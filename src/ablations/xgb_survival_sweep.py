"""Hyperparameter sweep for XGBoost-Survival (src/baselines_survival.py),
run alongside the patient-graph GNN sweep (patient_graph_gnn_sweep.py) for a
fair "tuned vs. tuned" comparison -- the study's own untuned XGBoost is
already the strongest model overall, so this checks whether that gap widens
or narrows once both sides get the same tuning effort.

Coordinate-wise sweep from the shipped defaults (n_estimators=250,
learning_rate=0.08, max_depth=6, subsample=0.85, colsample_bytree=0.7),
one axis at a time. Selection metric: mean AUROC at the 3-year horizon
across all 5 causes on the VALIDATION split -- XGBoost-Survival currently
has no validation-based selection at all (Section 3 of the review that
prompted this), so this sweep adds one, using exactly the same metric and
horizon the patient-graph sweep uses, for a like-for-like comparison. Test
data is touched exactly once, for the final best-combined config only.

Output: tkg_output/sweeps/xgb_survival_sweep.csv (screening results)
        tkg_output/sweeps/xgb_survival_best/test_metrics.csv (final config,
        test set)
"""
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.config import OUTPUT_DIR, SEED
from src.baselines_survival import _load, _build_X, _eval_horizon_auroc, CAUSES, HORIZON_DAYS

SWEEP_DIR = os.path.join(OUTPUT_DIR, "sweeps")
DEFAULTS = dict(n_estimators=250, learning_rate=0.08, max_depth=6, subsample=0.85, colsample_bytree=0.7)

AXES = {
    "n_estimators": [100, 250, 500, 800],
    "learning_rate": [0.02, 0.05, 0.08, 0.15],
    "max_depth": [3, 4, 6, 8],
    "subsample": [0.6, 0.85, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9],
}


def _fit_and_score(X_tr, y_xgb_tr, X_eval, labels_eval, config, seed=42):
    """Fit one XGBoost-Survival model per cause with `config`, score on
    X_eval/labels_eval at the 3-year horizon, return mean AUROC across causes."""
    aurocs = []
    for cause in CAUSES:
        clf = xgb.XGBRegressor(
            objective="survival:cox", eval_metric="cox-nloglik",
            n_estimators=config["n_estimators"], learning_rate=config["learning_rate"],
            max_depth=config["max_depth"], subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            tree_method="hist", n_jobs=-1, random_state=seed,
        )
        clf.fit(X_tr, y_xgb_tr[cause])
        risk = clf.predict(X_eval)
        rows = _eval_horizon_auroc(risk, labels_eval, cause, [1095])
        auroc = rows[0]["auroc"]
        if not np.isnan(auroc):
            aurocs.append(auroc)
    return float(np.mean(aurocs)) if aurocs else float("nan")


def run_screening() -> pd.DataFrame:
    os.makedirs(SWEEP_DIR, exist_ok=True)
    print("Loading data once (shared across all sweep configs)...")
    labels, static, events, nodes = _load()
    X, labels, tr_m, va_m, te_m = _build_X(labels, static, events, nodes)
    X = X.astype(np.float32)
    X_tr = X[tr_m]; X_va = X[va_m]
    labels_tr = labels[tr_m].reset_index(drop=True)
    labels_va = labels[va_m].reset_index(drop=True)

    y_xgb_tr = {}
    for cause in CAUSES:
        event_tr = (labels_tr["endpoint_type"] == cause).to_numpy().astype(bool)
        dur_tr = np.maximum(labels_tr["time_to_event_days"].to_numpy(dtype=float), 1.0)
        y_xgb_tr[cause] = np.where(event_tr, dur_tr, -dur_tr).astype(np.float32)

    rows = []
    seen = set()
    configs = [("baseline", dict(DEFAULTS))]
    for axis, values in AXES.items():
        for v in values:
            cfg = dict(DEFAULTS)
            cfg[axis] = v
            key = tuple(sorted(cfg.items()))
            if key in seen:
                continue
            seen.add(key)
            configs.append((f"{axis}={v}", cfg))

    print(f"\nScreening {len(configs)} configs (val-only selection, mean AUROC @ 3y across 5 causes)...\n")
    for name, cfg in configs:
        t0 = time.time()
        mean_auroc = _fit_and_score(X_tr, y_xgb_tr, X_va, labels_va, cfg, seed=SEED)
        dt = time.time() - t0
        print(f"  {name:24s} n_est={cfg['n_estimators']:4d} lr={cfg['learning_rate']:.3f} "
              f"depth={cfg['max_depth']} subsample={cfg['subsample']:.2f} "
              f"colsample={cfg['colsample_bytree']:.2f}  -> val_mean_AUROC@3y={mean_auroc:.4f} ({dt:.1f}s)")
        rows.append(dict(name=name, **cfg, val_mean_auroc_3y=mean_auroc, seconds=dt))

    result = pd.DataFrame(rows)
    out_path = os.path.join(SWEEP_DIR, "xgb_survival_sweep.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return result


def best_combined_config(result: pd.DataFrame) -> dict:
    best = dict(DEFAULTS)
    baseline_score = result.loc[result["name"] == "baseline", "val_mean_auroc_3y"].iloc[0]
    print(f"\nBaseline val_mean_AUROC@3y = {baseline_score:.4f}")
    for axis in AXES:
        pool = result[result["name"].str.startswith(f"{axis}=") | (result["name"] == "baseline")]
        top = pool.loc[pool["val_mean_auroc_3y"].idxmax()]
        best[axis] = top[axis]
        print(f"  best {axis}: {top[axis]} (val_mean_AUROC@3y={top['val_mean_auroc_3y']:.4f}, "
              f"{'improves' if top['val_mean_auroc_3y'] > baseline_score else 'no improvement'} over baseline)")
    return best


def run_final(best_cfg: dict) -> None:
    print(f"\nFinal best-combined config: {best_cfg}")
    out_dir = os.path.join(SWEEP_DIR, "xgb_survival_best")
    os.makedirs(out_dir, exist_ok=True)

    labels, static, events, nodes = _load()
    X, labels, tr_m, va_m, te_m = _build_X(labels, static, events, nodes)
    X = X.astype(np.float32)
    X_tr = X[tr_m]; X_te = X[te_m]
    labels_tr = labels[tr_m].reset_index(drop=True)
    labels_te = labels[te_m].reset_index(drop=True)

    all_rows = []
    for cause in CAUSES:
        event_tr = (labels_tr["endpoint_type"] == cause).to_numpy().astype(bool)
        dur_tr = np.maximum(labels_tr["time_to_event_days"].to_numpy(dtype=float), 1.0)
        y_xgb = np.where(event_tr, dur_tr, -dur_tr).astype(np.float32)
        clf = xgb.XGBRegressor(
            objective="survival:cox", eval_metric="cox-nloglik",
            n_estimators=best_cfg["n_estimators"], learning_rate=best_cfg["learning_rate"],
            max_depth=best_cfg["max_depth"], subsample=best_cfg["subsample"],
            colsample_bytree=best_cfg["colsample_bytree"],
            tree_method="hist", n_jobs=-1, random_state=SEED,
        )
        clf.fit(X_tr, y_xgb)
        risk = clf.predict(X_te)
        rows = _eval_horizon_auroc(risk, labels_te, cause, HORIZON_DAYS)
        for r in rows:
            r["model"] = "xgb_surv_tuned"
        all_rows.extend(rows)

    result = pd.DataFrame(all_rows)
    out_path = os.path.join(out_dir, "test_metrics.csv")
    result.to_csv(out_path, index=False)
    print("\n=== TUNED XGBOOST-SURVIVAL, TEST AUROC ===")
    print(result.pivot(index="cause", columns="horizon_days", values="auroc").round(4))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    screening = run_screening()
    best_cfg = best_combined_config(screening)
    run_final(best_cfg)

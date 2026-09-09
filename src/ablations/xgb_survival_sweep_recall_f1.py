"""Same hyperparameter sweep as xgb_survival_sweep.py, but selecting on
class-1 recall / F1 instead of AUROC -- a different question ("does this
config find more of the true positives at some threshold") than AUROC's
ranking question.

Recall alone turns out to be a degenerate selection criterion: for a
patient this rare (0.4%-5.2% test-set prevalence), recall=1.0 is always
achievable by flagging everyone (a threshold below the lowest score), so
every config ties at max_recall=1.0 and the sweep can't discriminate
between them. Confirmed this directly before falling back to F1, which
does not have this problem -- selection here uses best-achievable F1 at
the 3-year horizon, same aggregation convention (mean across 5 causes) as
the AUROC-based sweep.

Reuses the exact same coordinate-wise candidate grid as
xgb_survival_sweep.py, so results are directly comparable to that sweep --
only the selection metric differs.

Output: tkg_output/sweeps/xgb_survival_sweep_recall_f1.csv (screening)
        tkg_output/sweeps/xgb_survival_recall_f1_best/test_metrics.csv (final)
"""
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import OUTPUT_DIR, SEED
from src.baselines_survival import _load, _build_X, CAUSES, HORIZON_DAYS
from src.ablations.threshold_metrics import best_threshold_metrics

SWEEP_DIR = os.path.join(OUTPUT_DIR, "sweeps")
DEFAULTS = dict(n_estimators=250, learning_rate=0.08, max_depth=6, subsample=0.85, colsample_bytree=0.7)

AXES = {
    "n_estimators": [100, 250, 500, 800],
    "learning_rate": [0.02, 0.05, 0.08, 0.15],
    "max_depth": [3, 4, 6, 8],
    "subsample": [0.6, 0.85, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9],
}


def _labels_for(cause, h, evts, durs):
    is_pos = (evts == cause) & (durs <= h)
    survived = durs >= h
    competing = (durs < h) & (evts != cause) & (evts != "censored")
    is_neg = survived | competing
    mask = is_pos | is_neg
    y = is_pos[mask].astype(int)
    return y, mask


def _fit_and_score(X_tr, y_xgb_tr, X_eval, labels_eval, config, seed=42):
    """Fit one XGBoost-Survival model per cause with `config`, score on
    X_eval/labels_eval at the 3-year horizon. Returns (mean max_recall,
    mean best_f1) across causes."""
    evts = labels_eval["endpoint_type"].to_numpy()
    durs = labels_eval["time_to_event_days"].to_numpy(dtype=float)
    recalls, f1s = [], []
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
        y, mask = _labels_for(cause, 1095, evts, durs)
        m = best_threshold_metrics(y, risk[mask])
        if not np.isnan(m["max_recall"]):
            recalls.append(m["max_recall"])
            f1s.append(m["best_f1"])
    mean_recall = float(np.mean(recalls)) if recalls else float("nan")
    mean_f1 = float(np.mean(f1s)) if f1s else float("nan")
    return mean_recall, mean_f1


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

    print(f"\nScreening {len(configs)} configs (val-only selection, mean recall/F1 @ 3y across 5 causes)...\n")
    for name, cfg in configs:
        t0 = time.time()
        mean_recall, mean_f1 = _fit_and_score(X_tr, y_xgb_tr, X_va, labels_va, cfg, seed=SEED)
        dt = time.time() - t0
        print(f"  {name:24s} n_est={cfg['n_estimators']:4d} lr={cfg['learning_rate']:.3f} "
              f"depth={cfg['max_depth']} subsample={cfg['subsample']:.2f} "
              f"colsample={cfg['colsample_bytree']:.2f}  -> val_mean_recall@3y={mean_recall:.4f} "
              f"val_mean_best_f1@3y={mean_f1:.4f} ({dt:.1f}s)")
        rows.append(dict(name=name, **cfg, val_mean_recall_3y=mean_recall, val_mean_best_f1_3y=mean_f1, seconds=dt))

    result = pd.DataFrame(rows)
    out_path = os.path.join(SWEEP_DIR, "xgb_survival_sweep_recall_f1.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    recall_spread = result["val_mean_recall_3y"].max() - result["val_mean_recall_3y"].min()
    print(f"\nRecall spread across all {len(result)} configs: {recall_spread:.4f} "
          f"({'confirmed degenerate -- selecting on F1 instead' if recall_spread < 0.01 else 'not degenerate, selecting on recall'})")
    return result


def best_combined_config(result: pd.DataFrame, select_col: str) -> dict:
    best = dict(DEFAULTS)
    baseline_score = result.loc[result["name"] == "baseline", select_col].iloc[0]
    print(f"\nBaseline {select_col} = {baseline_score:.4f}")
    for axis in AXES:
        pool = result[result["name"].str.startswith(f"{axis}=") | (result["name"] == "baseline")]
        top = pool.loc[pool[select_col].idxmax()]
        best[axis] = top[axis]
        print(f"  best {axis}: {top[axis]} ({select_col}={top[select_col]:.4f}, "
              f"{'improves' if top[select_col] > baseline_score else 'no improvement'} over baseline)")
    return best


def run_final(best_cfg: dict) -> None:
    print(f"\nFinal best-combined config (selected by F1): {best_cfg}")
    out_dir = os.path.join(SWEEP_DIR, "xgb_survival_recall_f1_best")
    os.makedirs(out_dir, exist_ok=True)

    labels, static, events, nodes = _load()
    X, labels, tr_m, va_m, te_m = _build_X(labels, static, events, nodes)
    X = X.astype(np.float32)
    X_tr = X[tr_m]; X_te = X[te_m]
    labels_tr = labels[tr_m].reset_index(drop=True)
    labels_te = labels[te_m].reset_index(drop=True)
    evts = labels_te["endpoint_type"].to_numpy()
    durs = labels_te["time_to_event_days"].to_numpy(dtype=float)

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
        for h in HORIZON_DAYS:
            y, mask = _labels_for(cause, h, evts, durs)
            m = best_threshold_metrics(y, risk[mask])
            all_rows.append(dict(cause=cause, horizon_days=h, model="xgb_surv_recall_f1_tuned", **m))

    result = pd.DataFrame(all_rows)
    out_path = os.path.join(out_dir, "test_metrics.csv")
    result.to_csv(out_path, index=False)
    print("\n=== TUNED-FOR-F1 XGBOOST-SURVIVAL, TEST METRICS ===")
    print(result.pivot(index="cause", columns="horizon_days", values="best_f1").round(4))
    print("\n(precision/recall at the best-F1 threshold, and that threshold itself, are in the saved CSV)")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    screening = run_screening()
    recall_spread = screening["val_mean_recall_3y"].max() - screening["val_mean_recall_3y"].min()
    select_col = "val_mean_recall_3y" if recall_spread >= 0.01 else "val_mean_best_f1_3y"
    best_cfg = best_combined_config(screening, select_col)
    run_final(best_cfg)

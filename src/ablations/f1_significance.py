"""Is the F1 gap between the patient-graph GNN and each of Cox / XGBoost
real, or within noise?

Runs the identical two-test protocol against each baseline separately
(Bonferroni-corrected within each baseline's own 15-cell family, matching
this project's existing scoping convention -- see README's "Repeating
training five times, and correcting for that"):

  1. Paired bootstrap CI on (baseline_best_f1 - pg_best_f1), seed 42 for the
     graph model: resample test patients, recompute BOTH models' best-F1
     threshold freshly on each resample (not just the point estimate) so
     threshold-selection noise is propagated into the CI, not assumed away.
  2. One-sample t-test: the patient-graph model's 5 already-saved seed F1
     values (patient_graph_gnn_recall_f1_best/test_metrics.csv) against the
     fixed baseline F1 value -- identical protocol to multi_seed_summary.py.

XGBoost (F1-tuned config) and the patient-graph model (F1-tuned config,
seed 42) are retrained once each to get per-patient scores, then saved to
disk so this doesn't need repeating. Cox uses its already-tuned per-patient
risk scores directly from baselines_survival/predictions_test.csv -- Cox
has no separate F1-specific hyperparameter sweep in this project (it has no
architecture hyperparameters to sweep the way XGBoost/the GNN do; only its
regularization strength, tuned in cox_regularization_sweep.py).

Output: tkg_output/stats/f1_significance.csv
        tkg_output/stats/f1_tuned_predictions_test.csv  (raw per-patient
        scores, so future analyses don't need to retrain either model)
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from scipy import stats

from src.config import OUTPUT_DIR, SEED
from src.baselines_survival import _load, _build_X, CAUSES, HORIZON_DAYS
from src.ablations.threshold_metrics import best_threshold_metrics
from src.ablations.xgb_survival_sweep_recall_f1 import (
    best_combined_config as xgb_best_combined_config,
)
from src.ablations.patient_graph_gnn_sweep_recall_f1 import (
    best_combined_config as pg_best_combined_config,
)

SWEEP_DIR = os.path.join(OUTPUT_DIR, "sweeps")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
N_BOOT = 2000


def _labels_for(cause, h, evts, durs):
    is_pos = (evts == cause) & (durs <= h)
    survived = durs >= h
    competing = (durs < h) & (evts != cause) & (evts != "censored")
    is_neg = survived | competing
    mask = is_pos | is_neg
    y = is_pos[mask].astype(int)
    return y, mask


def _fit_xgb_f1_tuned_predictions() -> pd.DataFrame:
    """Refit the F1-tuned XGBoost (best_combined_config on the already-saved
    screening CSV, selecting on val_mean_best_f1_3y) per cause, return
    per-patient test risk scores."""
    screening = pd.read_csv(os.path.join(SWEEP_DIR, "xgb_survival_sweep_recall_f1.csv"))
    best_cfg = xgb_best_combined_config(screening, "val_mean_best_f1_3y")
    print(f"F1-tuned XGBoost config: {best_cfg}")

    labels, static, events, nodes = _load()
    X, labels, tr_m, va_m, te_m = _build_X(labels, static, events, nodes)
    X = X.astype(np.float32)
    X_tr, X_te = X[tr_m], X[te_m]
    labels_tr = labels[tr_m].reset_index(drop=True)
    labels_te = labels[te_m].reset_index(drop=True)

    preds = pd.DataFrame({"subject_id": labels_te["subject_id"].to_numpy()})
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
        preds[f"xgb_f1tuned_risk_{cause}"] = clf.predict(X_te)
    return preds


def _fit_pg_f1_tuned_predictions_seed42() -> pd.DataFrame:
    """Refit the F1-tuned patient-graph GNN (best_combined_config on the
    already-saved screening CSV) at seed 42, return per-patient test CIF
    probabilities. Reuses _train_one exactly as
    patient_graph_gnn_sweep_recall_f1.py's run_final does."""
    from src.tgn_survival import CAUSES as C, NUM_CAUSES, NUM_TIME_BINS
    from src.ablations.patient_graph_gnn_sweep_recall_f1 import _train_one
    from src.ablations.patient_graph_gnn import _prepare_patient_graph_data

    screening = pd.read_csv(os.path.join(SWEEP_DIR, "patient_graph_gnn_sweep_recall_f1.csv"))
    best_cfg = pg_best_combined_config(screening, "val_mean_best_f1_3y")
    print(f"F1-tuned patient-graph GNN config: {best_cfg}")

    d = _prepare_patient_graph_data()
    _, _, best_state, time_edges, model, device, _, _ = _train_one(d, best_cfg, seed=42)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        static_all = d["static_arr"].to(device)
        logits_flat = model(static_all)
        test_pos = np.array([d["pid_to_pos"][s] for s in d["splits"]["test"]])
        logits = logits_flat[test_pos].view(-1, NUM_CAUSES, NUM_TIME_BINS)
        probs = F.softmax(logits.reshape(logits.size(0), -1), dim=-1).view_as(logits)
        cif_test = torch.cumsum(probs, dim=-1).detach().cpu().numpy()
    test_sids = np.array(d["splits"]["test"])

    bin_idx_by_h = {h: max(0, min(int(np.searchsorted(time_edges, h, side="right") - 1),
                                   cif_test.shape[-1] - 1)) for h in HORIZON_DAYS}
    out = pd.DataFrame({"subject_id": test_sids})
    for i, cause in enumerate(C):
        for h in HORIZON_DAYS:
            out[f"pg_f1tuned_cif_{cause}_at_{h}d"] = cif_test[:, i, bin_idx_by_h[h]]
    return out


def _paired_bootstrap_f1_diff(y, s_a, s_b, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI on best_f1(s_a) - best_f1(s_b), resampling patients and
    re-selecting each model's own best-F1 threshold fresh on every resample."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    diffs = []
    for _ in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        yb = y[samp]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        fa = best_threshold_metrics(yb, s_a[samp])["best_f1"]
        fb = best_threshold_metrics(yb, s_b[samp])["best_f1"]
        if np.isnan(fa) or np.isnan(fb):
            continue
        diffs.append(fa - fb)
    if not diffs:
        return np.nan, np.nan, np.nan
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p-value: 2 * min(P(diff<=0), P(diff>=0))
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.0))


def _test_vs_baseline(baseline_label, s_baseline_col, df, mask_lookup, pg_5seed):
    """Run the paired-bootstrap + one-sample-t-test protocol for one
    baseline (xgb or cox) against the patient-graph model, across all 15
    disease/horizon cells. Returns a DataFrame of results plus its own
    Bonferroni-corrected significance flags (scoped to this baseline's own
    15-cell family, per this project's existing correction convention)."""
    rows = []
    for cause in CAUSES:
        for h in HORIZON_DAYS:
            y, mask = mask_lookup[(cause, h)]
            s_base = df.loc[mask, s_baseline_col(cause)].to_numpy(float)
            s_pg = df.loc[mask, f"pg_f1tuned_cif_{cause}_at_{h}d"].to_numpy(float)

            base_f1 = best_threshold_metrics(y, s_base)["best_f1"]
            pg_f1_seed42 = best_threshold_metrics(y, s_pg)["best_f1"]
            lo, hi, p_boot = _paired_bootstrap_f1_diff(y, s_base, s_pg)

            seed_vals = pg_5seed[(pg_5seed.cause == cause) & (pg_5seed.horizon_days == h)]["best_f1"].to_numpy()
            if len(seed_vals) >= 2 and np.isfinite(base_f1):
                _, p_ttest = stats.ttest_1samp(seed_vals, popmean=base_f1)
            else:
                p_ttest = np.nan

            rows.append(dict(
                cause=cause, horizon_days=h, baseline=baseline_label,
                baseline_best_f1=round(base_f1, 4),
                pg_f1tuned_best_f1_seed42=round(pg_f1_seed42, 4),
                diff_ci_low=round(lo, 4), diff_ci_high=round(hi, 4),
                p_paired_bootstrap=round(p_boot, 4),
                pg_5seed_mean=round(float(seed_vals.mean()), 4) if len(seed_vals) else np.nan,
                pg_5seed_std=round(float(seed_vals.std()), 4) if len(seed_vals) else np.nan,
                p_ttest_5seed_vs_baseline=round(p_ttest, 4) if np.isfinite(p_ttest) else np.nan,
            ))
            print(f"  {cause:8s} {h}d  vs {baseline_label:4s}  base_f1={base_f1:.4f}  "
                  f"pg_f1(seed42)={pg_f1_seed42:.4f}  diff_CI=[{lo:.4f},{hi:.4f}]  "
                  f"p_boot={p_boot:.4f}  p_ttest(5seed)="
                  + (f"{p_ttest:.4f}" if np.isfinite(p_ttest) else "n/a"))

    result = pd.DataFrame(rows)
    n_tests = len(result)
    alpha_bonf = 0.05 / n_tests
    result["sig_bootstrap_bonf"] = result["p_paired_bootstrap"] < alpha_bonf
    result["sig_ttest_bonf"] = result["p_ttest_5seed_vs_baseline"] < alpha_bonf
    return result, alpha_bonf


def run() -> None:
    os.makedirs(STATS_DIR, exist_ok=True)

    print("=== Refitting F1-tuned XGBoost (per cause) ===")
    xgb_preds = _fit_xgb_f1_tuned_predictions()
    print("\n=== Refitting F1-tuned patient-graph GNN (seed 42) ===")
    pg_preds = _fit_pg_f1_tuned_predictions_seed42()

    print("\n=== Loading Cox per-patient predictions (no retraining) ===")
    cox_path = os.path.join(OUTPUT_DIR, "baselines_survival", "predictions_test.csv")
    cox_preds = pd.read_csv(cox_path)[["subject_id"] + [f"cox_risk_{c}" for c in CAUSES]]

    labels_all = pd.read_csv(os.path.join(OUTPUT_DIR, "modeling", "labels.csv"))
    test_labels = labels_all[labels_all.split == "test"][
        ["subject_id", "endpoint_type", "time_to_event_days"]]
    df = test_labels.merge(xgb_preds, on="subject_id").merge(
        pg_preds, on="subject_id").merge(cox_preds, on="subject_id")
    evts = df["endpoint_type"].to_numpy()
    durs = df["time_to_event_days"].to_numpy(dtype=float)

    preds_out_path = os.path.join(STATS_DIR, "f1_tuned_predictions_test.csv")
    df.drop(columns=["endpoint_type", "time_to_event_days"]).to_csv(preds_out_path, index=False)
    print(f"  saved per-patient predictions: {preds_out_path}")

    pg_5seed = pd.read_csv(os.path.join(SWEEP_DIR, "patient_graph_gnn_recall_f1_best", "test_metrics.csv"))
    mask_lookup = {(cause, h): _labels_for(cause, h, evts, durs)
                   for cause in CAUSES for h in HORIZON_DAYS}
    mask_lookup = {k: v for k, v in mask_lookup.items() if v[0].sum() > 0 and v[0].sum() < len(v[0])}

    print("\n=== F1 gap vs XGBoost: paired bootstrap CI (seed 42) + one-sample t-test (5-seed dist) ===\n")
    result_xgb, alpha_xgb = _test_vs_baseline(
        "xgb", lambda c: f"xgb_f1tuned_risk_{c}", df, mask_lookup, pg_5seed)

    print("\n=== F1 gap vs Cox: paired bootstrap CI (seed 42) + one-sample t-test (5-seed dist) ===\n")
    result_cox, alpha_cox = _test_vs_baseline(
        "cox", lambda c: f"cox_risk_{c}", df, mask_lookup, pg_5seed)

    result = pd.concat([result_xgb, result_cox], ignore_index=True)
    out_path = os.path.join(STATS_DIR, "f1_significance.csv")
    result.to_csv(out_path, index=False)

    print(f"\n=== SUMMARY ===")
    for label, res, alpha in [("XGBoost", result_xgb, alpha_xgb), ("Cox", result_cox, alpha_cox)]:
        n_tests = len(res)
        n_sig_boot = int(res["sig_bootstrap_bonf"].sum())
        n_sig_ttest = int(res["sig_ttest_bonf"].sum())
        pg_wins = int((res["diff_ci_high"] < 0).sum())
        base_wins = int((res["diff_ci_low"] > 0).sum())
        print(f"\n--- vs {label} (Bonferroni alpha={alpha:.5f} across {n_tests} cells) ---")
        print(res[["cause", "horizon_days", "baseline_best_f1", "pg_f1tuned_best_f1_seed42",
                    "diff_ci_low", "diff_ci_high", "p_paired_bootstrap", "p_ttest_5seed_vs_baseline"]]
              .to_string(index=False))
        print(f"Paired bootstrap: significant in {n_sig_boot}/{n_tests} cells "
              f"({label} favored in {base_wins}, patient-graph favored in {pg_wins})")
        print(f"One-sample t-test (5-seed dist vs fixed {label}): significant in {n_sig_ttest}/{n_tests} cells")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()

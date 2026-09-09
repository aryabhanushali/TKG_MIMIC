"""Was Cox's regularization strength (l1_ratio=0.9, alphas=[0.01], a single
hardcoded value with no tuning) actually reasonable, or an accident that
happened to collapse every disease's model down to 1-4 non-zero coefficients
out of 17,695 available features?

Found while investigating the stroke-at-1-year result (Section 8.6): Cox's
Stroke model turned out to rely on exactly 2 non-zero coefficients (age,
and a small blood-pressure term) -- and checking the other four diseases as
a control showed this isn't unique to stroke: MI has 4 non-zero
coefficients, HF has 3, AF has 1 (age, alone), PAD has 1 (a single
blood-pressure feature, alone). Cox is functionally a 1-4-variable model
for every disease in this study, not really using the same feature space
XGBoost gets in practice, even though it's handed the same input.

This script checks whether that's a defensible choice or an artifact of one
untuned setting, using the discipline every other model in this study
follows: select purely on validation data, touch test exactly once.
CoxnetSurvivalAnalysis fits its entire regularization path in one call
(strongest to weakest penalty) and can predict at any point along that path
without refitting -- so this evaluates ~100 alpha values per disease from
one fit, not 100 separate model fits.

Output: tkg_output/stats/cox_regularization_sweep.csv (per-disease selected
        alpha, non-zero coefficient count, validation and test AUROC vs.
        the original alphas=[0.01] baseline)
"""
import os
import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis

from src.config import OUTPUT_DIR
from src.baselines_survival import _load, _build_X, _make_y, _eval_horizon_auroc, CAUSES, HORIZON_DAYS

STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
L1_RATIO = 0.9   # unchanged from the original setting -- only the penalty STRENGTH is being tuned here
BASELINE_ALPHA = 0.01


def run() -> None:
    os.makedirs(STATS_DIR, exist_ok=True)
    labels, static, events, nodes = _load()
    X, labels, tr_m, va_m, te_m = _build_X(labels, static, events, nodes)
    X = X.astype(np.float32)
    X_tr = X[tr_m]; X_va = X[va_m]; X_te = X[te_m]
    labels_tr = labels[tr_m].reset_index(drop=True)
    labels_va = labels[va_m].reset_index(drop=True)
    labels_te = labels[te_m].reset_index(drop=True)

    X_tr_dense = X_tr.toarray()
    X_va_dense = X_va.toarray()
    X_te_dense = X_te.toarray()

    rows = []
    for cause in CAUSES:
        print(f"\n=== {cause} ===")
        y_tr = _make_y(labels_tr, cause)

        print(f"  fitting full regularization path (l1_ratio={L1_RATIO})...")
        cox_path = CoxnetSurvivalAnalysis(l1_ratio=L1_RATIO, n_alphas=100,
                                            max_iter=2000, tol=1e-7)
        cox_path.fit(X_tr_dense, y_tr)
        alphas = cox_path.alphas_
        print(f"  path has {len(alphas)} alphas, range [{alphas.min():.6f}, {alphas.max():.6f}]")

        # Score every alpha on VALIDATION only, mean AUROC across the 3
        # horizons -- same selection discipline (val-only, no test peeking)
        # as every hyperparameter search in this study.
        best_alpha, best_val_score, best_nnz = None, -1.0, None
        for alpha in alphas:
            risk_va = cox_path.predict(X_va_dense, alpha=alpha)
            aurocs = []
            for h in HORIZON_DAYS:
                r = _eval_horizon_auroc(risk_va, labels_va, cause, [h])[0]
                if not np.isnan(r["auroc"]):
                    aurocs.append(r["auroc"])
            if not aurocs:
                continue
            mean_auroc = float(np.mean(aurocs))
            if mean_auroc > best_val_score:
                best_val_score = mean_auroc
                best_alpha = alpha
                coef = cox_path.coef_[:, list(alphas).index(alpha)]
                best_nnz = int((coef != 0).sum())

        print(f"  best alpha={best_alpha:.6f} (val mean AUROC={best_val_score:.4f}, "
              f"{best_nnz} non-zero coefficients)")

        # Baseline (original alphas=[0.01]) coefficient count + val score, for direct comparison
        baseline_idx = int(np.argmin(np.abs(alphas - BASELINE_ALPHA)))
        baseline_alpha_actual = alphas[baseline_idx]
        baseline_coef = cox_path.coef_[:, baseline_idx]
        baseline_nnz = int((baseline_coef != 0).sum())
        risk_va_baseline = cox_path.predict(X_va_dense, alpha=baseline_alpha_actual)
        baseline_val_aurocs = []
        for h in HORIZON_DAYS:
            r = _eval_horizon_auroc(risk_va_baseline, labels_va, cause, [h])[0]
            if not np.isnan(r["auroc"]):
                baseline_val_aurocs.append(r["auroc"])
        baseline_val_score = float(np.mean(baseline_val_aurocs)) if baseline_val_aurocs else np.nan

        # Final test evaluation, once, at the selected alpha only
        risk_te = cox_path.predict(X_te_dense, alpha=best_alpha)
        risk_te_baseline = cox_path.predict(X_te_dense, alpha=baseline_alpha_actual)
        for h in HORIZON_DAYS:
            r_tuned = _eval_horizon_auroc(risk_te, labels_te, cause, [h])[0]
            r_base = _eval_horizon_auroc(risk_te_baseline, labels_te, cause, [h])[0]
            rows.append(dict(
                cause=cause, horizon_days=h,
                baseline_alpha=round(float(baseline_alpha_actual), 6),
                baseline_nnz=baseline_nnz,
                baseline_val_mean_auroc=round(baseline_val_score, 4),
                baseline_test_auroc=round(r_base["auroc"], 4) if not np.isnan(r_base["auroc"]) else np.nan,
                tuned_alpha=round(float(best_alpha), 6),
                tuned_nnz=best_nnz,
                tuned_val_mean_auroc=round(best_val_score, 4),
                tuned_test_auroc=round(r_tuned["auroc"], 4) if not np.isnan(r_tuned["auroc"]) else np.nan,
            ))

    result = pd.DataFrame(rows)
    out_path = os.path.join(STATS_DIR, "cox_regularization_sweep.csv")
    result.to_csv(out_path, index=False)

    print("\n=== SUMMARY: original (alphas=0.01) vs. validation-selected alpha ===")
    print(result[["cause", "horizon_days", "baseline_nnz", "baseline_test_auroc",
                   "tuned_nnz", "tuned_test_auroc"]].to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()

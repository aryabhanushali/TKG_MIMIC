"""Statistical evaluation of the saved test predictions.

Adds the rigor an IEEE submission needs, computed entirely from the already-saved
per-patient test predictions (no retraining):

  1. The same corrected competing-risks horizon definition used everywhere else
     in this pipeline (src.tgn_survival._per_cause_auroc_at_horizons,
     src.baselines_survival._eval_horizon_auroc): positive iff (observed cause
     == c AND duration <= h); negative iff (duration >= h) OR (duration < h AND
     an observed COMPETING event); dropped only if administratively censored
     before h (status unknown; this part remains IPCW-free, see limitations).
     Recomputing it here (rather than trusting test_metrics.csv) keeps the
     bootstrap CIs and DeLong tests below defined on exactly the same patient
     set as the main results tables.

  2. Bootstrap 95% confidence intervals on every AUROC / AUPRC (resampling test
     patients, N_BOOT reps, percentile method).

  3. DeLong tests for paired AUROC differences (TGN vs XGBoost, TGN vs Cox,
     XGBoost vs Cox) -- same patients, correlated ROC curves.

  4. A valid reliability diagram for the TGN CIF (a genuine probability). Cox /
     XGBoost emit relative-risk scores, NOT probabilities, so they are excluded
     from calibration rather than min-max-rescaled into pseudo-probabilities.

Inputs : tkg_output/modeling/labels.csv
         tkg_output/baselines_survival/predictions_test.csv  (cox_risk_*, xgb_surv_risk_*)
         tkg_output/tgn_survival/predictions_test.csv         (cif_<cause>_at_<h>d)
Outputs: tkg_output/stats/test_metrics_with_ci.csv
         tkg_output/stats/delong_pairwise.csv
         tkg_output/figures/fig18_auroc_forest_ci.png
         tkg_output/figures/fig19_tgn_calibration_corrected.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import OUTPUT_DIR, FIGURES_DIR, SEED

MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
CAUSES = ["MI", "Stroke", "HF", "AF", "PAD"]
HORIZONS = [365, 1095, 1825]
HORIZON_LABEL = {365: "1y", 1095: "3y", 1825: "5y"}
N_BOOT = 2000
MODELS = ["cox", "xgb_surv", "tgn_surv"]
MODEL_LABEL = {"cox": "Cox", "xgb_surv": "XGB-Surv", "tgn_surv": "TGN-Surv"}
MODEL_COLOR = {"cox": "#888888", "xgb_surv": "#1f77b4", "tgn_surv": "#d62728"}
CAUSE_COLORS = {"MI": "#d62728", "Stroke": "#9467bd", "HF": "#ff7f0e",
                "AF": "#1f77b4", "PAD": "#2ca02c"}


# --------------------------------------------------------------------------- #
# Fast DeLong (Sun & Xu 2014) for two correlated AUCs                          #
# --------------------------------------------------------------------------- #
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    """preds_sorted: (2, N) with the m positives in the first m columns."""
    k, N = preds_sorted.shape
    n = N - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, N])
    for r in range(k):
        tx[r, :] = _midrank(pos[r, :])
        ty[r, :] = _midrank(neg[r, :])
        tz[r, :] = _midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_test(y, p1, p2):
    """Two-sided p-value for AUC(p1) == AUC(p2) on the same labels y."""
    y = np.asarray(y); p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    order = np.argsort(-y, kind="mergesort")     # positives (label 1) first
    m = int(y.sum())
    preds = np.vstack((p1, p2))[:, order]
    aucs, cov = _fast_delong(preds, m)
    l = np.array([[1.0, -1.0]])
    var = float((l @ cov @ l.T)[0, 0])
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), np.nan
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2 * norm.sf(abs(z)))


# --------------------------------------------------------------------------- #
# Corrected competing-risks horizon masks                                     #
# --------------------------------------------------------------------------- #
def _labels_for(cause, h, evts, durs):
    """Return (y, mask) for cause c at horizon h under the corrected rule:

      positive : observed cause == c AND duration <= h
      negative : duration >= h (event-free at/past h)  OR
                 (duration < h AND an observed COMPETING event != censored)
      dropped  : administratively censored with duration < h (status unknown)

    Boundary convention (duration == h for a censored patient counts as
    "survived to h", competing events use a strict "< h") matches
    src.tgn_survival._per_cause_auroc_at_horizons and
    src.baselines_survival._eval_horizon_auroc exactly, so this script's
    bootstrap CIs / DeLong tests are computed on the same patient set as the
    main results tables in test_metrics.csv.
    """
    is_pos = (evts == cause) & (durs <= h)
    survived = durs >= h
    competing = (durs < h) & (evts != cause) & (evts != "censored")
    is_neg = survived | competing
    mask = is_pos | is_neg
    y = is_pos[mask].astype(int)
    return y, mask


def _bootstrap_ci(y, s, metric, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        yb, sb = y[samp], s[samp]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        vals.append(metric(yb, sb))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _load_scores():
    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    test = labels[labels["split"] == "test"][
        ["subject_id", "endpoint_type", "time_to_event_days"]].copy()
    base = pd.read_csv(os.path.join(OUTPUT_DIR, "baselines_survival",
                                    "predictions_test.csv"))
    tgn = pd.read_csv(os.path.join(OUTPUT_DIR, "tgn_survival",
                                   "predictions_test.csv"))
    df = test.merge(base, on="subject_id", how="inner").merge(
        tgn.drop(columns=["endpoint_true", "duration_days"]),
        on="subject_id", how="inner")
    return df


def _score_col(df, model, cause, h):
    if model == "cox":
        return df[f"cox_risk_{cause}"].to_numpy(float)
    if model == "xgb_surv":
        return df[f"xgb_surv_risk_{cause}"].to_numpy(float)
    return df[f"cif_{cause}_at_{h}d"].to_numpy(float)


def run() -> None:
    os.makedirs(STATS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Loading saved test predictions...")
    df = _load_scores()
    evts = df["endpoint_type"].to_numpy()
    durs = df["time_to_event_days"].to_numpy(float)
    print(f"  aligned test patients across all 3 models: {len(df):,}")

    metric_rows, delong_rows = [], []
    for cause in CAUSES:
        for h in HORIZONS:
            y, mask = _labels_for(cause, h, evts, durs)
            n_pos, n = int(y.sum()), int(len(y))
            n_dropped = int((~mask).sum())
            if n_pos == 0 or n_pos == n:
                continue
            scores = {m: _score_col(df, m, cause, h)[mask] for m in MODELS}
            for m in MODELS:
                s = scores[m]
                auroc = roc_auc_score(y, s)
                auprc = average_precision_score(y, s)
                lo, hi = _bootstrap_ci(y, s, roc_auc_score)
                plo, phi = _bootstrap_ci(y, s, average_precision_score)
                metric_rows.append({
                    "cause": cause, "horizon_days": h, "model": m,
                    "auroc": auroc, "auroc_ci_low": lo, "auroc_ci_high": hi,
                    "auprc": auprc, "auprc_ci_low": plo, "auprc_ci_high": phi,
                    "n_pos": n_pos, "n": n, "n_dropped_censored": n_dropped,
                })
            for a, b in [("tgn_surv", "xgb_surv"), ("tgn_surv", "cox"),
                         ("xgb_surv", "cox")]:
                auc_a, auc_b, p = delong_test(y, scores[a], scores[b])
                delong_rows.append({
                    "cause": cause, "horizon_days": h,
                    "model_a": a, "model_b": b,
                    "auroc_a": auc_a, "auroc_b": auc_b,
                    "auroc_diff": auc_a - auc_b, "delong_p": p,
                    "significant_0.05": (p < 0.05) if pd.notna(p) else False,
                })

    metrics = pd.DataFrame(metric_rows)
    delong = pd.DataFrame(delong_rows)
    m_path = os.path.join(STATS_DIR, "test_metrics_with_ci.csv")
    d_path = os.path.join(STATS_DIR, "delong_pairwise.csv")
    metrics.to_csv(m_path, index=False)
    delong.to_csv(d_path, index=False)

    # Console summary: 3-year horizon AUROC with CI, all models.
    print("\n=== CORRECTED competing-risks AUROC @ 3y (95% bootstrap CI) ===")
    sub = metrics[metrics["horizon_days"] == 1095]
    for cause in CAUSES:
        line = f"  {cause:<7s}"
        for m in MODELS:
            r = sub[(sub["cause"] == cause) & (sub["model"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            line += (f"  {MODEL_LABEL[m]}={r.auroc:.3f}"
                     f"[{r.auroc_ci_low:.3f},{r.auroc_ci_high:.3f}]")
        print(line)

    print("\n=== TGN-Surv vs XGBoost-Surv (DeLong p, @3y) ===")
    ds = delong[(delong["horizon_days"] == 1095) &
                (delong["model_a"] == "tgn_surv") &
                (delong["model_b"] == "xgb_surv")]
    for r in ds.itertuples(index=False):
        sig = "SIGNIFICANT" if (pd.notna(r.delong_p) and r.delong_p < 0.05) else "n.s."
        print(f"  {r.cause:<7s} TGN={r.auroc_a:.3f} XGB={r.auroc_b:.3f}  "
              f"diff={r.auroc_diff:+.3f}  p={r.delong_p:.3f}  ({sig})")

    _forest_plot(metrics, os.path.join(FIGURES_DIR, "fig18_auroc_forest_ci.png"))
    _calibration_tgn(df, evts, durs,
                     os.path.join(FIGURES_DIR, "fig19_tgn_calibration_corrected.png"))

    print("\nSaved:")
    print(f"  {m_path}")
    print(f"  {d_path}")
    print(f"  {os.path.join(FIGURES_DIR, 'fig18_auroc_forest_ci.png')}")
    print(f"  {os.path.join(FIGURES_DIR, 'fig19_tgn_calibration_corrected.png')}")


def _forest_plot(metrics, out_path):
    """AUROC point estimates + 95% CI per cause/horizon, all three models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharex=True)
    for ax, h in zip(axes, HORIZONS):
        sub = metrics[metrics["horizon_days"] == h]
        ylabels, ypos = [], []
        y = 0
        for cause in CAUSES:
            for m in MODELS:
                r = sub[(sub["cause"] == cause) & (sub["model"] == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(
                    r.auroc, y,
                    xerr=[[r.auroc - r.auroc_ci_low], [r.auroc_ci_high - r.auroc]],
                    fmt="o", color=MODEL_COLOR[m], capsize=3, markersize=5)
                ylabels.append(f"{cause} · {MODEL_LABEL[m]}")
                ypos.append(y)
                y += 1
            y += 1  # gap between causes
        ax.axvline(0.5, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlim(0.45, 0.95)
        ax.set_xlabel("AUROC (corrected competing-risks)")
        ax.set_title(f"{HORIZON_LABEL[h]} horizon", fontweight="bold")
        ax.invert_yaxis()
    handles = [plt.Line2D([0], [0], marker="o", color=MODEL_COLOR[m], linestyle="",
                          label=MODEL_LABEL[m]) for m in MODELS]
    fig.suptitle("Test AUROC with 95% bootstrap CIs (corrected competing-risks evaluation)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _calibration_tgn(df, evts, durs, out_path):
    """Reliability diagram for the TGN CIF (a genuine probability) at each horizon.
    Quantile bins over the corrected eval set; baselines excluded (risk scores,
    not probabilities)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, h in zip(axes, HORIZONS):
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6, label="ideal")
        for cause in CAUSES:
            y, mask = _labels_for(cause, h, evts, durs)
            p = df[f"cif_{cause}_at_{h}d"].to_numpy(float)[mask]
            if y.sum() < 10:
                continue
            # quantile bins of predicted risk
            edges = np.quantile(p, np.linspace(0, 1, 6))
            edges[-1] += 1e-9
            binid = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 4)
            xs, ys = [], []
            for b in range(5):
                sel = binid == b
                if sel.sum() < 5:
                    continue
                xs.append(p[sel].mean()); ys.append(y[sel].mean())
            ax.plot(xs, ys, "o-", color=CAUSE_COLORS[cause], markersize=4,
                    linewidth=1.3, label=cause)
        ax.set_title(f"{HORIZON_LABEL[h]} horizon", fontweight="bold")
        ax.set_xlabel("mean predicted CIF"); ax.set_ylabel("observed event rate")
        ax.set_xlim(0, None); ax.set_ylim(0, None)
        if h == 365:
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("TGN-Survival CIF calibration (quantile-binned, corrected eval set)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()

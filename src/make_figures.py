"""Paper-quality figure generation.

Produces five additional figures from saved test metrics + per-patient
predictions, complementing the cohort/TKG figures already produced by
visualize_tkg.py and the explainability figures from explain*.py.

    fig5_model_comparison.png        master grouped-bar: 3 horizons x 5 causes x 3 models
    fig6_roc_5y.png                   ROC curves at 5y horizon, one panel per cause
    fig7_auroc_by_horizon.png         AUROC trajectory (1y/3y/5y) per cause, line plot
    fig12_calibration_5y.png          calibration plots at 5y horizon (reliability diagrams)

All figures are saved at 300 DPI as PNG. None of these scripts touch the model
or training; they read pre-computed test metrics + predictions only.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

from src.config import OUTPUT_DIR, FIGURES_DIR

CAUSES = ["MI", "Stroke", "HF", "AF", "PAD"]
HORIZONS = [365, 1095, 1825]
HORIZON_LABELS = {365: "1-year", 1095: "3-year", 1825: "5-year"}

MODEL_COLORS = {"cox": "#888888", "xgb_surv": "#1f77b4", "tgn_surv": "#d62728"}
MODEL_LABELS = {"cox": "Cox elastic-net", "xgb_surv": "XGBoost-Survival",
                "tgn_surv": "TGN-Survival (ours)"}
CAUSE_COLORS = {"MI": "#d62728", "Stroke": "#9467bd", "HF": "#ff7f0e",
                "AF": "#1f77b4", "PAD": "#2ca02c"}


# --------------------------------------------------------------------------- #
# Load metrics + predictions                                                  #
# --------------------------------------------------------------------------- #
def _load_metrics() -> pd.DataFrame:
    """Combine Cox / XGB-Surv / TGN-Surv test metrics into one long frame."""
    parts = []
    base_path = os.path.join(OUTPUT_DIR, "baselines_survival", "test_metrics.csv")
    tgn_path = os.path.join(OUTPUT_DIR, "tgn_survival", "test_metrics.csv")
    if os.path.exists(base_path):
        parts.append(pd.read_csv(base_path))
    if os.path.exists(tgn_path):
        tgn = pd.read_csv(tgn_path)
        tgn["model"] = "tgn_surv"
        parts.append(tgn)
    if not parts:
        raise FileNotFoundError("No test_metrics.csv found in either model dir")
    df = pd.concat(parts, ignore_index=True)
    return df


def _load_predictions():
    """Per-patient risk scores from baselines + CIF from TGN-Survival."""
    base_preds_path = os.path.join(OUTPUT_DIR, "baselines_survival",
                                     "predictions_test.csv")
    tgn_preds_path = os.path.join(OUTPUT_DIR, "tgn_survival",
                                    "predictions_test.csv")
    labels_path = os.path.join(OUTPUT_DIR, "modeling", "labels.csv")

    labels = pd.read_csv(labels_path)
    base = pd.read_csv(base_preds_path) if os.path.exists(base_preds_path) else None
    tgn  = pd.read_csv(tgn_preds_path)  if os.path.exists(tgn_preds_path)  else None
    return labels, base, tgn


# --------------------------------------------------------------------------- #
# Figure 5: master grouped-bar comparison                                     #
# --------------------------------------------------------------------------- #
def fig5_model_comparison(metrics: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    width = 0.27
    x = np.arange(len(CAUSES))
    for ax, h in zip(axes, HORIZONS):
        models_order = ["cox", "xgb_surv", "tgn_surv"]
        for i, model in enumerate(models_order):
            sub = metrics[(metrics["horizon_days"] == h)
                           & (metrics["model"] == model)
                           & (metrics["cause"].isin(CAUSES))]
            sub = sub.set_index("cause").reindex(CAUSES)
            heights = sub["auroc"].to_numpy()
            ax.bar(x + (i - 1) * width, heights, width,
                   color=MODEL_COLORS[model], label=MODEL_LABELS[model],
                   edgecolor="white", linewidth=0.5)
            for xi, hgt in zip(x + (i - 1) * width, heights):
                if not np.isnan(hgt):
                    ax.text(xi, hgt + 0.005, f"{hgt:.2f}",
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(CAUSES)
        ax.set_ylim(0.55, 0.90)
        ax.set_title(HORIZON_LABELS[h] + " horizon", fontweight="bold")
        ax.set_ylabel("test AUROC")
        ax.grid(axis="y", alpha=0.3)
        if h == HORIZONS[0]:
            ax.legend(loc="upper left", fontsize=9, frameon=True)
    fig.suptitle("Figure 5 — Per-cause test AUROC: Cox vs XGBoost-Survival vs TGN-Survival",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 6: ROC curves at 5-year horizon                                      #
# --------------------------------------------------------------------------- #
def _binary_labels_for_cause_at_horizon(labels: pd.DataFrame, cause: str,
                                         horizon_days: int):
    """y, mask in patient order (test only). y=1 if cause observed AND duration<=h.
    Mask drops censored-before-horizon patients (IPCW-free framing — same as the
    AUROC eval in tgn_survival.py)."""
    labels_test = labels[labels["split"] == "test"].sort_values("subject_id").reset_index(drop=True)
    d = labels_test["time_to_event_days"].to_numpy(dtype=float)
    e = labels_test["endpoint_type"].to_numpy()
    pos = (e == cause) & (d <= horizon_days)
    neg = (d >= horizon_days)
    mask = pos | neg
    return labels_test, pos.astype(int), mask


def fig6_roc_at_5y(labels, base_preds, tgn_preds, out_path: str) -> None:
    if base_preds is None and tgn_preds is None:
        print("  fig6: no predictions found; skipping")
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    h = 1825
    for k, cause in enumerate(CAUSES):
        ax = axes[k]
        # Build ground truth in canonical patient order
        labels_test, y_full, mask = _binary_labels_for_cause_at_horizon(labels, cause, h)
        if y_full.sum() == 0:
            ax.set_title(f"{cause} (no positives)"); continue

        # Cox
        if base_preds is not None and f"cox_risk_{cause}" in base_preds.columns:
            preds = (base_preds.set_index("subject_id")
                       .reindex(labels_test["subject_id"])
                       [f"cox_risk_{cause}"].to_numpy())
            if mask.sum() > 1 and preds[mask].size:
                fpr, tpr, _ = roc_curve(y_full[mask], preds[mask])
                auc = roc_auc_score(y_full[mask], preds[mask])
                ax.plot(fpr, tpr, label=f"Cox (AUC={auc:.3f})",
                        color=MODEL_COLORS["cox"], linewidth=1.5)

        # XGB-Surv
        if base_preds is not None and f"xgb_surv_risk_{cause}" in base_preds.columns:
            preds = (base_preds.set_index("subject_id")
                       .reindex(labels_test["subject_id"])
                       [f"xgb_surv_risk_{cause}"].to_numpy())
            if mask.sum() > 1 and preds[mask].size:
                fpr, tpr, _ = roc_curve(y_full[mask], preds[mask])
                auc = roc_auc_score(y_full[mask], preds[mask])
                ax.plot(fpr, tpr, label=f"XGB-Surv (AUC={auc:.3f})",
                        color=MODEL_COLORS["xgb_surv"], linewidth=1.8)

        # TGN-Surv (CIF column)
        if tgn_preds is not None and f"cif_{cause}_at_{h}d" in tgn_preds.columns:
            preds = (tgn_preds.set_index("subject_id")
                       .reindex(labels_test["subject_id"])
                       [f"cif_{cause}_at_{h}d"].to_numpy())
            if mask.sum() > 1 and preds[mask].size:
                fpr, tpr, _ = roc_curve(y_full[mask], preds[mask])
                auc = roc_auc_score(y_full[mask], preds[mask])
                ax.plot(fpr, tpr, label=f"TGN-Surv (AUC={auc:.3f})",
                        color=MODEL_COLORS["tgn_surv"], linewidth=2.5)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.5)
        ax.set_title(f"{cause}  (n_pos={int(y_full.sum())} / n={int(mask.sum())})",
                     fontweight="bold")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)
    axes[-1].axis("off")
    fig.suptitle("Figure 6 — Test-set ROC curves at 5-year horizon",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 7: AUROC trajectory by horizon (line plot per cause × model)         #
# --------------------------------------------------------------------------- #
def fig7_auroc_by_horizon(metrics: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)
    for ax, cause in zip(axes, CAUSES):
        sub = metrics[metrics["cause"] == cause]
        for model in ["cox", "xgb_surv", "tgn_surv"]:
            s = (sub[sub["model"] == model]
                   .sort_values("horizon_days"))
            if s.empty:
                continue
            ax.plot(s["horizon_days"], s["auroc"], "o-",
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                    linewidth=2,
                    markersize=8)
        ax.set_title(cause, fontweight="bold", color=CAUSE_COLORS[cause])
        ax.set_xlabel("horizon (days)")
        ax.set_xticks(HORIZONS)
        ax.set_xticklabels(["1y", "3y", "5y"])
        ax.set_ylim(0.55, 0.90)
        ax.grid(alpha=0.3)
        if cause == CAUSES[0]:
            ax.set_ylabel("test AUROC")
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Figure 7 — AUROC trajectory across 1-, 3-, 5-year horizons",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 12: calibration at 5y horizon                                        #
# --------------------------------------------------------------------------- #
def fig12_calibration_5y(labels, base_preds, tgn_preds, out_path: str) -> None:
    if tgn_preds is None and base_preds is None:
        print("  fig12: no predictions; skipping")
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    h = 1825
    for k, cause in enumerate(CAUSES):
        ax = axes[k]
        labels_test, y_full, mask = _binary_labels_for_cause_at_horizon(labels, cause, h)
        if y_full.sum() < 10:
            ax.set_title(f"{cause} (too few positives)"); continue
        for model_name, source_df, col in [
            ("Cox",      base_preds, f"cox_risk_{cause}"),
            ("XGB-Surv", base_preds, f"xgb_surv_risk_{cause}"),
            ("TGN-Surv", tgn_preds,  f"cif_{cause}_at_{h}d"),
        ]:
            if source_df is None or col not in source_df.columns:
                continue
            preds = (source_df.set_index("subject_id")
                       .reindex(labels_test["subject_id"])[col].to_numpy())
            scores = preds[mask]
            ys = y_full[mask]
            if scores.size < 10 or not np.isfinite(scores).all():
                continue
            # Rescale risk scores into [0,1] for plotting (Cox / XGB outputs are
            # raw partial hazards / log-hazards). TGN CIF is already 0..1.
            if model_name in ("Cox", "XGB-Surv"):
                s_min, s_max = scores.min(), scores.max()
                scores = (scores - s_min) / max(s_max - s_min, 1e-9)
            try:
                frac_pos, mean_pred = calibration_curve(ys, scores, n_bins=10,
                                                         strategy="quantile")
            except ValueError:
                continue
            colormap = {"Cox": MODEL_COLORS["cox"],
                        "XGB-Surv": MODEL_COLORS["xgb_surv"],
                        "TGN-Surv": MODEL_COLORS["tgn_surv"]}
            lw = 2.5 if model_name == "TGN-Surv" else 1.5
            ax.plot(mean_pred, frac_pos, "o-", label=model_name,
                    color=colormap[model_name], linewidth=lw, markersize=6)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.5,
                label="perfect calibration")
        ax.set_title(f"{cause}", fontweight="bold")
        ax.set_xlabel("mean predicted risk (rescaled)")
        ax.set_ylabel("observed event fraction")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[-1].axis("off")
    fig.suptitle("Figure 12 — Calibration at 5-year horizon (reliability diagrams)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #
def make_all() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    metrics = _load_metrics()
    labels, base_preds, tgn_preds = _load_predictions()

    fig5 = os.path.join(FIGURES_DIR, "fig5_model_comparison.png")
    fig6 = os.path.join(FIGURES_DIR, "fig6_roc_5y.png")
    fig7 = os.path.join(FIGURES_DIR, "fig7_auroc_by_horizon.png")
    fig12 = os.path.join(FIGURES_DIR, "fig12_calibration_5y.png")

    print("Figure 5: master grouped-bar (3 horizons x 5 causes x 3 models)...")
    fig5_model_comparison(metrics, fig5)
    print(f"  saved {fig5}")

    print("Figure 6: ROC curves at 5-year horizon (1 panel per cause)...")
    fig6_roc_at_5y(labels, base_preds, tgn_preds, fig6)
    print(f"  saved {fig6}")

    print("Figure 7: AUROC trajectory across horizons (line plot per cause)...")
    fig7_auroc_by_horizon(metrics, fig7)
    print(f"  saved {fig7}")

    print("Figure 12: calibration reliability diagrams at 5-year horizon...")
    fig12_calibration_5y(labels, base_preds, tgn_preds, fig12)
    print(f"  saved {fig12}")


if __name__ == "__main__":
    make_all()

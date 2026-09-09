"""Publication-ready figures summarizing this project's final results.

Five figures, each answering one question this project's results section
makes a claim about. Reuses only already-saved artifacts -- no retraining,
no new statistical computation. Every figure is 300 dpi, colorblind-aware
(model identity is always also encoded by x-position and a legend, never
color alone), and uses one consistent color-per-model convention throughout:

    Cox              #888888 (neutral gray -- the classical baseline)
    XGBoost           #1f77b4 (blue)
    TKG-Transformer   #d62728 (red)
    Patient-graph GNN #2ca02c (green)

Output: paper_figures/fig1_main_comparison.png
        paper_figures/fig2_cox_vs_xgb_equivalence.png
        paper_figures/fig3_delong_robustness.png
        paper_figures/fig4_explanation_fidelity.png
        paper_figures/fig5_f1_auprc.png
        paper_figures/fig6_feature_importance.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import OUTPUT_DIR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "paper_figures")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explain")

CAUSES = ["MI", "Stroke", "HF", "AF", "PAD"]
CAUSE_LABELS = {"MI": "Heart attack", "Stroke": "Stroke", "HF": "Heart failure",
                "AF": "Atrial fibrillation", "PAD": "Peripheral artery disease"}
HORIZONS = [365, 1095, 1825]
HORIZON_LABELS = {365: "1y", 1095: "3y", 1825: "5y"}

MODEL_COLOR = {"cox": "#888888", "xgb_surv": "#1f77b4", "tgn_surv": "#d62728", "patient_graph": "#2ca02c"}
MODEL_LABEL = {"cox": "Cox", "xgb_surv": "XGBoost", "tgn_surv": "TKG-Transformer", "patient_graph": "Patient-graph GNN"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _legend_handles(models):
    return [mpatches.Patch(color=MODEL_COLOR[m], label=MODEL_LABEL[m]) for m in models]


# --------------------------------------------------------------------------- #
# Figure 1: main model comparison                                             #
# --------------------------------------------------------------------------- #
def fig1_main_comparison():
    surv = pd.read_csv(os.path.join(OUTPUT_DIR, "survival_comparison_test.csv"))
    pg = pd.read_csv(os.path.join(OUTPUT_DIR, "patient_gnn_survival", "test_metrics.csv"))
    pg["model"] = "patient_graph"
    df = pd.concat([surv, pg], ignore_index=True)

    models = ["cox", "xgb_surv", "tgn_surv", "patient_graph"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharey=True)
    x = np.arange(len(HORIZONS))
    width = 0.2
    for ax, cause in zip(axes, CAUSES):
        for i, m in enumerate(models):
            vals = [df[(df.cause == cause) & (df.horizon_days == h) & (df.model == m)]["auroc"]
                    for h in HORIZONS]
            vals = [v.iloc[0] if len(v) else np.nan for v in vals]
            ax.bar(x + (i - 1.5) * width, vals, width, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([HORIZON_LABELS[h] for h in HORIZONS])
        ax.set_title(CAUSE_LABELS[cause], fontsize=10.5, fontweight="bold")
        ax.set_ylim(0.45, 0.90)
    axes[0].set_ylabel("Test AUROC")
    fig.legend(handles=_legend_handles(models), loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Test AUROC by disease, horizon, and model", fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_main_comparison.png"))
    plt.close(fig)
    print("  saved fig1_main_comparison.png")


# --------------------------------------------------------------------------- #
# Figure 2: Cox vs XGBoost statistical equivalence (forest plot)              #
# --------------------------------------------------------------------------- #
def fig2_cox_vs_xgb_equivalence():
    df = pd.read_csv(os.path.join(STATS_DIR, "test_metrics_with_ci.csv"))
    rows = []
    for cause in CAUSES:
        for h in HORIZONS:
            for m in ["cox", "xgb_surv"]:
                r = df[(df.cause == cause) & (df.horizon_days == h) & (df.model == m)]
                if len(r):
                    rows.append(dict(cause=cause, horizon=h, model=m,
                                      auroc=r.auroc.iloc[0], lo=r.auroc_ci_low.iloc[0], hi=r.auroc_ci_high.iloc[0]))
    d = pd.DataFrame(rows)
    cells = [(c, h) for c in CAUSES for h in HORIZONS]
    y = np.arange(len(cells))[::-1]

    fig, ax = plt.subplots(figsize=(7, 8.5))
    for i, (cause, h) in zip(y, cells):
        for off, m in [(0.15, "cox"), (-0.15, "xgb_surv")]:
            r = d[(d.cause == cause) & (d.horizon == h) & (d.model == m)]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.plot([r.lo, r.hi], [i + off, i + off], color=MODEL_COLOR[m], linewidth=2.2, solid_capstyle="round")
            ax.plot(r.auroc, i + off, "o", color=MODEL_COLOR[m], markersize=5, zorder=3)
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{CAUSE_LABELS[c]} — {HORIZON_LABELS[h]}" for c, h in cells], fontsize=9)
    ax.set_xlabel("Test AUROC (95% bootstrap CI)")
    ax.set_title("Cox vs. XGBoost: overlapping confidence intervals everywhere\n"
                  "(paired DeLong test finds no significant difference in any of 15 cells)",
                  fontsize=11.5, fontweight="bold")
    ax.legend(handles=_legend_handles(["cox", "xgb_surv"]), loc="lower right", frameon=False)
    ax.set_xlim(0.4, 0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_cox_vs_xgb_equivalence.png"))
    plt.close(fig)
    print("  saved fig2_cox_vs_xgb_equivalence.png")


# --------------------------------------------------------------------------- #
# Figure 3: DeLong robustness audit -- which "beats Cox" claims survive       #
# --------------------------------------------------------------------------- #
def fig3_delong_robustness():
    df = pd.read_csv(os.path.join(STATS_DIR, "delong_robustness_audit.csv"))
    models = ["TGN-Transformer", "PatientGraph"]
    model_label = {"TGN-Transformer": "TKG-Transformer", "PatientGraph": "Patient-graph GNN"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    cells = [(c, h) for c in CAUSES for h in HORIZONS]
    y = np.arange(len(cells))[::-1]
    for ax, model in zip(axes, models):
        sub = df[df.model == model]
        for i, (cause, h) in zip(y, cells):
            r = sub[(sub.cause == cause) & (sub.horizon_days == h)]
            if not len(r):
                continue
            r = r.iloc[0]
            wins = r.mean_model_auc > r.cox_auc
            if r.ROBUST_both_tests_agree:
                color, marker = ("#0ca30c" if wins else "#d03b3b"), "o"
            elif r.ttest_sig_bonferroni:
                color, marker = "#fab219", "s"
            else:
                color, marker = "#c3c2b7", "."
            ax.plot(r.n_seeds_beat_cox, i, marker, color=color, markersize=9,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{CAUSE_LABELS[c]} — {HORIZON_LABELS[h]}" for c, h in cells], fontsize=9)
        ax.set_xlim(-0.5, 5.5)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xlabel("Seeds (of 5) where model beat Cox")
        ax.set_title(model_label[model], fontsize=11.5, fontweight="bold")
        ax.axvline(2.5, color="black", linewidth=0.6, alpha=0.3)

    handles = [
        mpatches.Patch(color="#0ca30c", label="Robust win (t-test + ≥4/5 DeLong-significant seeds)"),
        mpatches.Patch(color="#d03b3b", label="Robust loss (same bar, Cox wins)"),
        mpatches.Patch(color="#fab219", label="t-test significant only — not DeLong-confirmed"),
        mpatches.Patch(color="#c3c2b7", label="Not significant"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.13), fontsize=9)
    fig.suptitle("Which \"beats Cox\" claims survive a stricter, paired significance test?",
                  fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_delong_robustness.png"))
    plt.close(fig)
    print("  saved fig3_delong_robustness.png")


# --------------------------------------------------------------------------- #
# Figure 4: explanation fidelity -- comprehensiveness vs sufficiency          #
# --------------------------------------------------------------------------- #
def fig4_explanation_fidelity():
    tgn = pd.read_csv(os.path.join(EXPLAIN_DIR, "gnn_explainer_fidelity_stats.csv"))
    pg = pd.read_csv(os.path.join(EXPLAIN_DIR, "patient_graph_fidelity_stats.csv"))
    tgn["model"] = "tgn_surv"
    pg["model"] = "patient_graph"
    df = pd.concat([tgn, pg], ignore_index=True)

    directions = ["comprehensiveness", "sufficiency"]
    models = ["tgn_surv", "patient_graph"]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(directions))
    width = 0.32
    for i, m in enumerate(models):
        vals, los, his = [], [], []
        for d in directions:
            r = df[(df.model == m) & (df.direction == d)].iloc[0]
            vals.append(r.win_rate)
            los.append(r.win_rate - r.win_rate_ci_lo)
            his.append(r.win_rate_ci_hi - r.win_rate)
        ax.bar(x + (i - 0.5) * width, vals, width, color=MODEL_COLOR[m], label=MODEL_LABEL[m],
               yerr=[los, his], capsize=4, error_kw=dict(linewidth=1.2))
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(1.62, 0.505, "chance (50%)", fontsize=8.5, color="#555", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(["Comprehensiveness", "Sufficiency"], fontsize=11)
    ax.set_ylabel("Win rate vs. random facts (95% CI)")
    ax.set_ylim(0, 1)
    ax.set_title("Explanation trustworthiness: comprehensiveness holds, sufficiency doesn't",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    fig.text(0.5, -0.02,
              "Comprehensiveness: does removing important facts hurt the prediction more than removing random facts?   "
              "Sufficiency: do the important facts alone preserve the prediction as well as a random set?",
              ha="center", fontsize=8.5, color="#555", wrap=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_explanation_fidelity.png"))
    plt.close(fig)
    print("  saved fig4_explanation_fidelity.png")


# --------------------------------------------------------------------------- #
# Figure 5: F1 and AUPRC -- no robust differences                             #
# --------------------------------------------------------------------------- #
def fig5_f1_auprc():
    """Two vertical forest plots (one row per disease/horizon cell), same
    visual language as fig2/fig3, instead of a dense continuous x-axis --
    15 cells x 3 series does not read cleanly on one horizontal axis."""
    auprc = pd.read_csv(os.path.join(STATS_DIR, "tuned_models_auprc_ci.csv"))
    model_map = {"cox": "cox", "xgb_surv": "xgb_surv", "patient_graph_5seed_mean": "patient_graph"}
    cells = [(c, h) for c in CAUSES for h in HORIZONS]
    y = np.arange(len(cells))[::-1]
    row_labels = [f"{CAUSE_LABELS[c]} — {HORIZON_LABELS[h]}" for c, h in cells]

    fig, axes = plt.subplots(1, 2, figsize=(13, 8.5))

    ax = axes[0]
    offsets = {"cox": 0.22, "xgb_surv": 0.0, "patient_graph": -0.22}
    for raw_m, m in model_map.items():
        for i, (cause, h) in zip(y, cells):
            r = auprc[(auprc.cause == cause) & (auprc.horizon_days == h) & (auprc.model == raw_m)]
            if not len(r):
                continue
            r = r.iloc[0]
            yy = i + offsets[m]
            ax.plot([r.auprc_ci_low, r.auprc_ci_high], [yy, yy], color=MODEL_COLOR[m], linewidth=1.8, alpha=0.85)
            ax.plot(r.auprc, yy, "o", color=MODEL_COLOR[m], markersize=4, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(row_labels, fontsize=8.5)
    ax.set_xlabel("AUPRC (95% bootstrap CI)")
    ax.set_title("AUPRC: overlapping CIs in every cell", fontsize=11.5, fontweight="bold")
    ax.legend(handles=_legend_handles(["cox", "xgb_surv", "patient_graph"]), loc="upper right",
              frameon=False, fontsize=8.5)

    ax2 = axes[1]
    f1 = pd.read_csv(os.path.join(STATS_DIR, "f1_significance.csv"))
    color_by_baseline = {"xgb": MODEL_COLOR["xgb_surv"], "cox": MODEL_COLOR["cox"]}
    label_by_baseline = {"xgb": f"vs. {MODEL_LABEL['xgb_surv']}", "cox": f"vs. {MODEL_LABEL['cox']}"}
    row_offset = {"xgb": 0.12, "cox": -0.12}
    for baseline in ["xgb", "cox"]:
        sub = f1[f1.baseline == baseline].reset_index(drop=True)
        for i, (cause, h) in zip(y, cells):
            r = sub[(sub.cause == cause) & (sub.horizon_days == h)]
            if not len(r):
                continue
            r = r.iloc[0]
            point = r["baseline_best_f1"] - r["pg_f1tuned_best_f1_seed42"]
            yy = i + row_offset[baseline]
            ax2.plot([r["diff_ci_low"], r["diff_ci_high"]], [yy, yy],
                      color=color_by_baseline[baseline], linewidth=1.8, alpha=0.85)
            ax2.plot(point, yy, "o", color=color_by_baseline[baseline], markersize=4, zorder=3)
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Best-F1 gap: baseline minus patient-graph (95% bootstrap CI)")
    ax2.set_title("F1 gap: every CI crosses zero\n(paired bootstrap, 30/30 not significant)",
                  fontsize=11.5, fontweight="bold")
    handles = [mpatches.Patch(color=color_by_baseline[b], label=label_by_baseline[b]) for b in ["xgb", "cox"]]
    ax2.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5)

    fig.suptitle("Threshold-based and threshold-free precision/recall metrics tell the same story",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_f1_auprc.png"))
    plt.close(fig)
    print("  saved fig5_f1_auprc.png")


# --------------------------------------------------------------------------- #
# Figure 6: top features per model per disease                               #
# --------------------------------------------------------------------------- #
def _format_icd9(code: str) -> str:
    """Standard ICD-9 decimal notation: 3-char category (E-codes: 4, V-codes: 3), then a decimal point."""
    if code.startswith("E") and len(code) > 4:
        return code[:4] + "." + code[4:]
    if code.startswith("V") and len(code) > 3:
        return code[:3] + "." + code[3:]
    if code[:1].isdigit() and len(code) > 3:
        return code[:3] + "." + code[3:]
    return code


def _format_icd10(code: str) -> str:
    """Standard ICD-10 decimal notation: 3-char category, then a decimal point."""
    return code[:3] + "." + code[3:] if len(code) > 3 else code


def _short_feature_name(name: str) -> str:
    """Shorten a raw feature/concept id for display in a table cell."""
    if name.startswith("VAL_LAB_"):
        return name[len("VAL_LAB_"):].replace("_", " ").title() + " (lab stat)"
    if name.startswith("VAL_OMR_"):
        return name[len("VAL_OMR_"):].replace("_", " ").title() + " (vital stat)"
    if name.startswith("LAB_"):
        return name[len("LAB_"):].replace("_", " ").title() + " (lab)"
    if name.startswith("DRUG_"):
        return name[len("DRUG_"):].replace("_", " ").title() + " (drug)"
    if name.startswith("ICD9_PROC_") or name.startswith("ICD10_PROC_"):
        return name.split("_")[-1] + " (procedure)"
    if name.startswith("ICD9_"):
        return _format_icd9(name[len("ICD9_"):]) + " (diagnosis)"
    if name.startswith("ICD10_"):
        return _format_icd10(name[len("ICD10_"):]) + " (diagnosis)"
    if name in ("age_at_index", "cci_score", "num_cardiometa_conditions", "had_icu_stay", "female"):
        return name.replace("_", " ") + " (demographic)"
    return name


def fig6_feature_importance():
    cox = pd.read_csv(os.path.join(EXPLAIN_DIR, "cox_feature_importance_by_cause.csv"))
    xgbi = pd.read_csv(os.path.join(EXPLAIN_DIR, "xgb_feature_importance_by_cause.csv"))
    tgn = pd.read_csv(os.path.join(EXPLAIN_DIR, "concept_importance_by_cause.csv"))
    pg = pd.read_csv(os.path.join(EXPLAIN_DIR, "gnn_concept_importance_by_cause.csv"))

    def top3(source, cause, sort_col, cause_col="cause", name_col="feature", n=3, shorten=True):
        if source is cox or source is xgbi:
            sub = source[(source[cause_col] == cause) & (source["rank"] <= n)].sort_values("rank")
            names = sub[name_col].tolist()
        else:
            sub = source[source[cause_col] == cause].sort_values(sort_col, ascending=False).head(n)
            names = sub[name_col].tolist()
        return [_short_feature_name(x) if shorten else x for x in names]

    models = ["cox", "xgb_surv", "tgn_surv", "patient_graph"]
    sources = {"cox": (cox, None), "xgb_surv": (xgbi, None),
               "tgn_surv": (tgn, "mean_attention"), "patient_graph": (pg, "mean_mask")}

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(CAUSES) + 0.6)
    ax.axis("off")

    col_w = 1.0
    row_h = 1.0
    header_y = len(CAUSES) + 0.1

    for j, m in enumerate(models):
        ax.add_patch(mpatches.Rectangle((j * col_w, header_y), col_w - 0.03, 0.5,
                                          color=MODEL_COLOR[m], alpha=0.85, zorder=1))
        ax.text(j * col_w + col_w / 2, header_y + 0.25, MODEL_LABEL[m], ha="center", va="center",
                fontsize=12, fontweight="bold", color="white", zorder=2)

    for i, cause in enumerate(CAUSES):
        y = len(CAUSES) - 1 - i
        ax.text(-0.05, y + row_h / 2, CAUSE_LABELS[cause], ha="right", va="center",
                fontsize=11, fontweight="bold")
        for j, m in enumerate(models):
            src, sort_col = sources[m]
            cause_col = "endpoint" if m == "patient_graph" else "cause"
            name_col = "concept_id" if m in ("tgn_surv", "patient_graph") else "feature"
            feats = top3(src, cause, sort_col, cause_col=cause_col, name_col=name_col)
            bg = "#f7f7f5" if i % 2 == 0 else "#ffffff"
            ax.add_patch(mpatches.Rectangle((j * col_w, y), col_w - 0.03, row_h - 0.03,
                                              facecolor=bg, edgecolor="#e1e0d9", linewidth=0.6, zorder=1))
            text = "\n".join(f"{k+1}. {f}" for k, f in enumerate(feats))
            ax.text(j * col_w + 0.05, y + row_h - 0.08, text, ha="left", va="top",
                    fontsize=8.3, linespacing=1.7, zorder=2)

    ax.set_title("Top 3 features by disease and model", fontsize=14.5, fontweight="bold", pad=28)
    fig.text(0.5, 0.005,
              "Cox and XGBoost draw on largely the same signal (age dominates stroke/HF/AF; disease-specific diagnosis codes lead MI/PAD).\n"
              "The patient-graph GNN's top features are noticeably less disease-specific — consistent with its separately-measured sufficiency failure (Figure 4).",
              ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=[0.09, 0.03, 1, 1])
    fig.savefig(os.path.join(OUT_DIR, "fig6_feature_importance.png"))
    plt.close(fig)
    print("  saved fig6_feature_importance.png")


def run() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Writing figures to {OUT_DIR}/")
    fig1_main_comparison()
    fig2_cox_vs_xgb_equivalence()
    fig3_delong_robustness()
    fig4_explanation_fidelity()
    fig5_f1_auprc()
    fig6_feature_importance()
    print("\nDone.")


if __name__ == "__main__":
    run()

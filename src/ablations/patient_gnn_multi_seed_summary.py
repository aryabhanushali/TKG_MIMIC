"""Multi-seed robustness summary: patient-graph GNN vs. Cox and XGBoost.

Mirrors src/multi_seed_summary.py's protocol exactly (same test, same
correction, same 30-cells-per-family framing), applied to the patient-graph
model (src/ablations/patient_graph_gnn.py, Section 6.3) instead of the plain
TKG-Transformer, so the two graph-vs-baseline comparisons use one consistent,
auditable statistical procedure.

This script was missing from the repository as committed -- the CSV it
produces (tkg_output/stats/patient_gnn_multi_seed_comparison.csv) existed as
an output artifact with no generating script anywhere in src/, which an
external audit flagged as a reproducibility gap (Section 8.6's headline
significance table could not be regenerated from the checked-in code). This
script was reconstructed to reproduce that artifact's numbers exactly
(verified bit-for-bit against the orphaned file before being added here) and
is now the single source of truth for regenerating it.

Welch's t-test (unequal variance, 5 vs. 5) per cell, Bonferroni-corrected
across all 30 cells tested in THIS family (15 vs-XGB + 15 vs-Cox). See
src/ablations/combined_significance_correction.py for the additional,
stricter correction across this family AND the plain-TGN family together --
report both when citing significance, since a reviewer correcting across the
union of the two families will reach a different (correct) answer than
either family's Bonferroni threshold alone.

Cox is not multi-seeded (CoxnetSurvivalAnalysis has no meaningful
random_state given a fixed design matrix), same as multi_seed_summary.py.

No retraining: reads already-completed test_metrics.csv per seed from
tkg_output/patient_gnn_survival[_seed{43..46}]/ and
tkg_output/baselines_survival[_seed{43..46}]/.

Outputs: tkg_output/stats/patient_gnn_multi_seed_comparison.csv
         tkg_output/figures/fig24_patient_gnn_multi_seed_auroc.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.config import OUTPUT_DIR, FIGURES_DIR

STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
SEEDS = [42, 43, 44, 45, 46]
CAUSES = ["MI", "Stroke", "HF", "AF", "PAD"]
HORIZONS = [365, 1095, 1825]
HORIZON_LABEL = {365: "1y", 1095: "3y", 1825: "5y"}
CAUSE_COLORS = {"MI": "#d62728", "Stroke": "#9467bd", "HF": "#ff7f0e",
                "AF": "#1f77b4", "PAD": "#2ca02c"}
N_CELLS_TESTED = len(CAUSES) * len(HORIZONS) * 2   # PatientGraph-vs-XGB and PatientGraph-vs-Cox
ALPHA_BONFERRONI = 0.05 / N_CELLS_TESTED


def _pg_dir(seed: int) -> str:
    return "patient_gnn_survival" if seed == 42 else f"patient_gnn_survival_seed{seed}"


def _baseline_dir(seed: int) -> str:
    return "baselines_survival" if seed == 42 else f"baselines_survival_seed{seed}"


def _load_seed_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pg_rows, xgb_rows = [], []
    for s in SEEDS:
        t = pd.read_csv(os.path.join(OUTPUT_DIR, _pg_dir(s), "test_metrics.csv"))
        t["seed"] = s
        pg_rows.append(t)
        b = pd.read_csv(os.path.join(OUTPUT_DIR, _baseline_dir(s), "test_metrics.csv"))
        x = b[b["model"] == "xgb_surv"].copy()
        x["seed"] = s
        xgb_rows.append(x)
    pg_all = pd.concat(pg_rows, ignore_index=True)
    xgb_all = pd.concat(xgb_rows, ignore_index=True)
    cox = pd.read_csv(os.path.join(OUTPUT_DIR, "baselines_survival", "test_metrics.csv"))
    cox = cox[cox["model"] == "cox"].copy()
    return pg_all, xgb_all, cox


def run() -> None:
    os.makedirs(STATS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Loading {len(SEEDS)} seed runs each for patient-graph GNN and "
          f"XGBoost-Survival ({SEEDS})...")
    pg_all, xgb_all, cox = _load_seed_metrics()

    rows = []
    for cause in CAUSES:
        cox_val = cox.loc[cox["cause"] == cause, :]
        for h in HORIZONS:
            t = (pg_all[(pg_all.cause == cause) & (pg_all.horizon_days == h)]
                 .sort_values("seed")["auroc"].to_numpy())
            x = (xgb_all[(xgb_all.cause == cause) & (xgb_all.horizon_days == h)]
                 .sort_values("seed")["auroc"].to_numpy())
            c = cox_val.loc[cox_val["horizon_days"] == h, "auroc"]
            c_val = float(c.iloc[0]) if len(c) else np.nan

            _, p_vs_xgb = stats.ttest_ind(t, x, equal_var=False)
            _, p_vs_cox = (stats.ttest_1samp(t, popmean=c_val)
                           if np.isfinite(c_val) else (np.nan, np.nan))

            rows.append(dict(
                cause=cause, horizon=h,
                cox=round(c_val, 4),
                xgb_mean=round(x.mean(), 4), xgb_std=round(x.std(ddof=1), 4),
                pg_mean=round(t.mean(), 4), pg_std=round(t.std(ddof=1), 4),
                pg_minus_xgb=round(t.mean() - x.mean(), 4),
                p_vs_xgb=round(p_vs_xgb, 4),
                pg_minus_cox=round(t.mean() - c_val, 4) if np.isfinite(c_val) else np.nan,
                p_vs_cox=round(p_vs_cox, 4) if np.isfinite(p_vs_cox) else np.nan,
                sig_vs_xgb=bool(p_vs_xgb < ALPHA_BONFERRONI),
                sig_vs_cox=bool(p_vs_cox < ALPHA_BONFERRONI) if np.isfinite(p_vs_cox) else False,
            ))
    result = pd.DataFrame(rows)
    out_path = os.path.join(STATS_DIR, "patient_gnn_multi_seed_comparison.csv")
    result.to_csv(out_path, index=False)

    print(f"\n=== MULTI-SEED (n={len(SEEDS)}) AUROC: Patient-graph GNN vs "
          f"XGBoost-Surv, Bonferroni alpha={ALPHA_BONFERRONI:.5f} across "
          f"{N_CELLS_TESTED} cells (this family only) ===\n")
    for h in HORIZONS:
        print(f"--- {HORIZON_LABEL[h]} horizon ---")
        sub = result[result.horizon == h]
        for _, r in sub.iterrows():
            flag_x = " *" if r.sig_vs_xgb else ""
            flag_c = " *" if r.sig_vs_cox else ""
            print(f"  {r.cause:8s} Cox={r.cox:.3f}  "
                  f"XGB={r.xgb_mean:.3f}+/-{r.xgb_std:.3f}  "
                  f"PatientGraph={r.pg_mean:.3f}+/-{r.pg_std:.3f}  "
                  f"| PG-XGB p={r.p_vs_xgb:.4f}{flag_x}  "
                  f"PG-Cox p={r.p_vs_cox:.4f}{flag_c}")
    print("\n  (* = survives Bonferroni correction for THIS family's 30 "
          "simultaneous tests only -- see combined_significance_correction.py "
          "for the stricter, cross-family bar)")

    _plot(result, os.path.join(FIGURES_DIR, "fig24_patient_gnn_multi_seed_auroc.png"))
    print(f"\nSaved:\n  {out_path}\n  "
          f"{os.path.join(FIGURES_DIR, 'fig24_patient_gnn_multi_seed_auroc.png')}")


def _plot(result: pd.DataFrame, out_path: str) -> None:
    """Per-cause AUROC across horizons: Cox (fixed point), XGB and
    patient-graph GNN as mean +/- std across 5 seeds. Significant
    (within-family Bonferroni) PatientGraph-vs-XGB gaps are starred."""
    fig, axes = plt.subplots(1, len(CAUSES), figsize=(4 * len(CAUSES), 4.5),
                              sharey=True)
    x = np.arange(len(HORIZONS))
    width = 0.25
    for ax, cause in zip(axes, CAUSES):
        sub = result[result.cause == cause].set_index("horizon").loc[HORIZONS]
        ax.bar(x - width, sub["cox"], width, label="Cox", color="#888888")
        ax.bar(x, sub["xgb_mean"], width, yerr=sub["xgb_std"], capsize=3,
               label="XGB-Surv", color="#1f77b4")
        ax.bar(x + width, sub["pg_mean"], width, yerr=sub["pg_std"], capsize=3,
               label="Patient-graph GNN", color="#2ca02c")
        for i, h in enumerate(HORIZONS):
            r = sub.loc[h]
            if r["sig_vs_xgb"]:
                y = max(r["pg_mean"] + r["pg_std"], r["xgb_mean"] + r["xgb_std"]) + 0.02
                ax.text(x[i], y, "*", ha="center", fontsize=14, fontweight="bold",
                        color=CAUSE_COLORS[cause])
        ax.set_xticks(x)
        ax.set_xticklabels([HORIZON_LABEL[h] for h in HORIZONS])
        ax.set_title(cause, fontweight="bold", color=CAUSE_COLORS[cause])
        ax.set_ylim(0.45, 0.85)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Test AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Multi-seed AUROC (n={len(SEEDS)} seeds, mean ± std); "
                 "* = Patient-graph vs. XGB significant after within-family "
                 "Bonferroni correction",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()

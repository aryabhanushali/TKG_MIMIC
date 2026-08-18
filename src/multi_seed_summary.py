"""Multi-seed robustness summary: TGN and XGBoost-Survival across 5 seeds.

A single training run of a neural model (TGN) can land on an unrepresentative
early-stopping checkpoint; XGBoost also has real seed sensitivity via its row/
column subsampling. This script aggregates already-completed seed runs (42
through 46, produced by `tgn_survival.py` / `baselines_survival.py` with
TKG_SEED set) into mean +/- std per (cause, horizon), and tests whether TGN's
apparent edge or deficit vs. XGBoost survives seed variance -- Welch's t-test
(unequal variance, 5 vs. 5) per cell, Bonferroni-corrected across all cells
tested (30 = 15 TGN-vs-XGB + 15 TGN-vs-Cox), since 5 seeds gives limited power
and reporting an uncorrected p-value here would overstate confidence.

Cox is not multi-seeded: CoxnetSurvivalAnalysis has no meaningful random_state
given a fixed design matrix, so its seed-42 value is used as a fixed reference
(one-sample t-test against the TGN seed distribution).

No retraining: this only reads test_metrics.csv from each already-completed
seed run under tkg_output/{tgn_survival,baselines_survival}[_seed{43..46}]/.

Outputs: tkg_output/stats/multi_seed_comparison.csv
         tkg_output/figures/fig20_multi_seed_auroc.png
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
N_CELLS_TESTED = len(CAUSES) * len(HORIZONS) * 2   # TGN-vs-XGB and TGN-vs-Cox
ALPHA_BONFERRONI = 0.05 / N_CELLS_TESTED


def _tgn_dir(seed: int) -> str:
    return "tgn_survival" if seed == 42 else f"tgn_survival_seed{seed}"


def _baseline_dir(seed: int) -> str:
    return "baselines_survival" if seed == 42 else f"baselines_survival_seed{seed}"


def _load_seed_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tgn_rows, xgb_rows = [], []
    for s in SEEDS:
        t = pd.read_csv(os.path.join(OUTPUT_DIR, _tgn_dir(s), "test_metrics.csv"))
        t["seed"] = s
        tgn_rows.append(t)
        b = pd.read_csv(os.path.join(OUTPUT_DIR, _baseline_dir(s), "test_metrics.csv"))
        x = b[b["model"] == "xgb_surv"].copy()
        x["seed"] = s
        xgb_rows.append(x)
    tgn_all = pd.concat(tgn_rows, ignore_index=True)
    xgb_all = pd.concat(xgb_rows, ignore_index=True)
    cox = pd.read_csv(os.path.join(OUTPUT_DIR, "baselines_survival", "test_metrics.csv"))
    cox = cox[cox["model"] == "cox"].copy()
    return tgn_all, xgb_all, cox


def run() -> None:
    os.makedirs(STATS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Loading {len(SEEDS)} seed runs each for TGN and XGBoost-Survival "
          f"({SEEDS})...")
    tgn_all, xgb_all, cox = _load_seed_metrics()

    rows = []
    for cause in CAUSES:
        cox_val = cox.loc[cox["cause"] == cause, :]
        for h in HORIZONS:
            t = (tgn_all[(tgn_all.cause == cause) & (tgn_all.horizon_days == h)]
                 .sort_values("seed")["auroc"].to_numpy())
            x = (xgb_all[(xgb_all.cause == cause) & (xgb_all.horizon_days == h)]
                 .sort_values("seed")["auroc"].to_numpy())
            c = cox_val.loc[cox_val["horizon_days"] == h, "auroc"]
            c_val = float(c.iloc[0]) if len(c) else np.nan

            _, p_vs_xgb = stats.ttest_ind(t, x, equal_var=False)
            _, p_vs_cox = (stats.ttest_1samp(t, popmean=c_val)
                           if np.isfinite(c_val) else (np.nan, np.nan))

            rows.append(dict(
                cause=cause, horizon_days=h,
                cox=round(c_val, 4),
                xgb_mean=round(x.mean(), 4), xgb_std=round(x.std(ddof=1), 4),
                tgn_mean=round(t.mean(), 4), tgn_std=round(t.std(ddof=1), 4),
                tgn_minus_xgb=round(t.mean() - x.mean(), 4),
                tgn_vs_xgb_p=round(p_vs_xgb, 4),
                tgn_vs_xgb_sig_bonferroni=bool(p_vs_xgb < ALPHA_BONFERRONI),
                tgn_minus_cox=round(t.mean() - c_val, 4) if np.isfinite(c_val) else np.nan,
                tgn_vs_cox_p=round(p_vs_cox, 4) if np.isfinite(p_vs_cox) else np.nan,
                tgn_vs_cox_sig_bonferroni=bool(p_vs_cox < ALPHA_BONFERRONI) if np.isfinite(p_vs_cox) else False,
            ))
    result = pd.DataFrame(rows)
    out_path = os.path.join(STATS_DIR, "multi_seed_comparison.csv")
    result.to_csv(out_path, index=False)

    print(f"\n=== MULTI-SEED (n={len(SEEDS)}) AUROC: TGN vs XGBoost-Surv, "
          f"Bonferroni alpha={ALPHA_BONFERRONI:.5f} across {N_CELLS_TESTED} cells ===\n")
    for h in HORIZONS:
        print(f"--- {HORIZON_LABEL[h]} horizon ---")
        sub = result[result.horizon_days == h]
        for _, r in sub.iterrows():
            flag_x = " *" if r.tgn_vs_xgb_sig_bonferroni else ""
            flag_c = " *" if r.tgn_vs_cox_sig_bonferroni else ""
            print(f"  {r.cause:8s} Cox={r.cox:.3f}  "
                  f"XGB={r.xgb_mean:.3f}+/-{r.xgb_std:.3f}  "
                  f"TGN={r.tgn_mean:.3f}+/-{r.tgn_std:.3f}  "
                  f"| TGN-XGB p={r.tgn_vs_xgb_p:.4f}{flag_x}  "
                  f"TGN-Cox p={r.tgn_vs_cox_p:.4f}{flag_c}")
    print("\n  (* = survives Bonferroni correction for 30 simultaneous tests)")

    _plot(result, os.path.join(FIGURES_DIR, "fig20_multi_seed_auroc.png"))
    print(f"\nSaved:\n  {out_path}\n  {os.path.join(FIGURES_DIR, 'fig20_multi_seed_auroc.png')}")


def _plot(result: pd.DataFrame, out_path: str) -> None:
    """Per-cause AUROC across horizons: Cox (fixed point), XGB and TGN as
    mean +/- std across 5 seeds. Significant (Bonferroni) TGN-vs-XGB gaps are
    starred."""
    fig, axes = plt.subplots(1, len(CAUSES), figsize=(4 * len(CAUSES), 4.5),
                              sharey=True)
    x = np.arange(len(HORIZONS))
    width = 0.25
    for ax, cause in zip(axes, CAUSES):
        sub = result[result.cause == cause].set_index("horizon_days").loc[HORIZONS]
        ax.bar(x - width, sub["cox"], width, label="Cox", color="#888888")
        ax.bar(x, sub["xgb_mean"], width, yerr=sub["xgb_std"], capsize=3,
               label="XGB-Surv", color="#1f77b4")
        ax.bar(x + width, sub["tgn_mean"], width, yerr=sub["tgn_std"], capsize=3,
               label="TGN-Surv", color="#d62728")
        for i, h in enumerate(HORIZONS):
            r = sub.loc[h]
            if r["tgn_vs_xgb_sig_bonferroni"]:
                y = max(r["tgn_mean"] + r["tgn_std"], r["xgb_mean"] + r["xgb_std"]) + 0.02
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
                 "* = TGN vs. XGB significant after Bonferroni correction",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()

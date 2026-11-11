"""Figures for the "who benefits from the TKG?" analyses.

  fig7_sample_size.png
     (a) MIMIC learning curve: mean AUROC over the 15 disease x horizon cells
         vs. number of training patients (same test set at every point).
     (b) Per-disease margin over Cox vs. training events for that disease,
         MIMIC runs (filled) and EHRSHOT CV (open), one panel per TKG model.
  fig8_subgroups.png
     Subgroup interaction (margin over Cox inside the subgroup minus outside),
     MIMIC vs EHRSHOT, for the pre-index subgroups discussed in the paper.

Inputs: tkg_output/stats/case_count/*.csv (case_count_analysis.py) and each
dataset's stats/discussion/subgroup_interaction_summary.csv
(discussion_findings.py). Missing inputs are skipped.

Colour: one colour per model, kept across figures, validated for CVD
separation (worst pair 9.2 OKLab x100) and paired with a distinct marker per
model plus direct labels, so identity is never colour-alone:
  Cox #52514e (square), XGBoost #2a78d6 (circle),
  TKG-Transformer #eb6834 (triangle), patient-graph #1baf7a (diamond).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CC = os.path.join(REPO, "tkg_output", "stats", "case_count")
OUT = os.path.join(REPO, "paper_figures")
COLOR = {"cox": "#52514e", "xgb": "#2a78d6", "tkg": "#eb6834", "pg": "#1baf7a"}
MARK = {"cox": "s", "xgb": "o", "tkg": "^", "pg": "D"}
LABEL = {"cox": "Cox", "xgb": "XGBoost", "tkg": "TKG-Transformer", "pg": "Patient-graph"}
INK, MUTED, GRID = "#0b0b0b", "#6b6a65", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 7, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _ring(**kw):
    return dict(markeredgecolor="white", markeredgewidth=1.2, **kw)


def fig_sample_size():
    lc_path, cells_path = os.path.join(CC, "learning_curve_auroc.csv"), os.path.join(CC, "case_count_cells.csv")
    if not (os.path.exists(lc_path) and os.path.exists(cells_path)):
        print("  fig7 skipped: run src.ablations.case_count_analysis first")
        return
    lc, cells = pd.read_csv(lc_path), pd.read_csv(cells_path)
    n_train_full = {}
    for (ds, fr), sub in cells.groupby(["dataset", "train_frac"]):
        n_train_full[(ds, fr)] = None
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.15, 1, 1]})

    # (a) learning curve, MIMIC only (same test set at every fraction)
    ax = axes[0]
    mim = lc[lc.dataset == "MIMIC"]
    # x = number of training patients, from the cells' run labels files via train events is
    # disease-specific; use the fraction of MIMIC's 23,516 train patients instead.
    for m in ["cox", "xgb", "tkg", "pg"]:
        s = mim[mim.model == m].sort_values("train_frac")
        if s.empty:
            continue
        x = s.train_frac * 23516
        ax.plot(x, s.mean_auroc, color=COLOR[m], marker=MARK[m], **_ring())
        ax.annotate(LABEL[m], (x.iloc[-1], s.mean_auroc.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ehr = lc[lc.dataset == "EHRSHOT"]
    if len(ehr):
        ax.axvline(1660, color=MUTED, linewidth=0.8, linestyle=":")
        ax.annotate("EHRSHOT train size\n(per CV fold)", (1660, ax.get_ylim()[0]), xytext=(4, 6),
                    textcoords="offset points", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("Training patients (MIMIC, log scale)")
    ax.set_ylabel("Mean AUROC, 15 disease × horizon cells")
    ax.set_title("(a) MIMIC learning curve, fixed test set", loc="left", fontsize=9, color=INK)
    ax.set_xlim(right=ax.get_xlim()[1] * 2.6)

    # (b, c) per-disease margin over Cox vs training events
    for ax, m, tag in [(axes[1], "tkg", "(b)"), (axes[2], "pg", "(c)")]:
        sub = cells[cells.pair == f"{m}_vs_cox"]
        if sub.empty:
            ax.set_visible(False)
            continue
        g = sub.groupby(["dataset", "train_frac", "cause"]).agg(delta=("delta", "mean"),
                                                                ev=("train_events", "first")).reset_index()
        for ds, filled in [("MIMIC", True), ("EHRSHOT", False)]:
            d = g[g.dataset == ds]
            ax.plot(d.ev, d.delta, linestyle="none", marker=MARK[m], color=COLOR[m],
                    markerfacecolor=COLOR[m] if filled else "white", markeredgewidth=1.4,
                    markeredgecolor=COLOR[m] if not filled else "white",
                    label=f"{ds}{' (all fractions)' if ds == 'MIMIC' else ' (5-fold CV)'}")
        for _, r in g[g.dataset == "EHRSHOT"].iterrows():
            ax.annotate(r.cause, (r.ev, r.delta), xytext=(5, -3), textcoords="offset points", fontsize=7, color=MUTED)
        x = np.log10(g.ev.to_numpy())
        b1, b0 = np.polyfit(x, g.delta.to_numpy(), 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(10 ** xs, b0 + b1 * xs, color=MUTED, linewidth=1, linestyle="--")
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Training events for that disease (log scale)")
        ax.set_ylabel(f"{LABEL[m]} − Cox AUROC")
        ax.set_title(f"{tag} {LABEL[m]} margin over Cox, per disease", loc="left", fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    p = os.path.join(OUT, "fig7_sample_size.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  saved {p}")


SUBGROUPS = [  # (prefix match in subgroup name, display label)
    ("very thin history", "Very thin history (bottom quartile)"),
    ("thin history", "Thin history (below median)"),
    ("no encounter before index", "No encounter before index admission"),
    ("no lab results", "No lab results"),
    ("very rich history", "Very rich history (top quartile)"),
    ("CCI >= 3", "CCI ≥ 3"),
    ("age >=", "Oldest age tertile"),
    ("age <", "Youngest age tertile"),
    ("female", "Female"),
    ("race: Black", "Race: Black"),
    ("race: Asian", "Race: Asian"),
    ("race: Hispanic/Latino", "Hispanic/Latino"),
]


def fig_subgroups():
    srcs = {"MIMIC": os.path.join(REPO, "tkg_output", "stats", "discussion", "subgroup_interaction_summary.csv"),
            "EHRSHOT": os.path.join(REPO, "tkg_output_ehrshot", "cv", "pooled", "stats", "discussion",
                                    "subgroup_interaction_summary.csv")}
    data = {k: pd.read_csv(v) for k, v in srcs.items() if os.path.exists(v)}
    if not data:
        print("  fig8 skipped: no subgroup_interaction_summary.csv")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, sharex=True, gridspec_kw={"wspace": 0.12})
    ds_style = {"MIMIC": dict(filled=True, off=-0.17), "EHRSHOT": dict(filled=False, off=0.17)}
    for ax, m in zip(axes, ["tkg", "pg"]):
        for ds, df in data.items():
            df = df[df.pair == f"{m}_vs_cox"]
            for i, (pref, lab) in enumerate(SUBGROUPS):
                hit = df[df.subgroup.str.startswith(pref)]
                if pref == "thin history":
                    hit = hit[~hit.subgroup.str.startswith("very")]
                if hit.empty:
                    continue
                r = hit.iloc[0]
                st = ds_style[ds]
                y = i + st["off"]
                ax.plot(r.mean_interaction, y, marker=MARK[m], color=COLOR[m], linestyle="none",
                        markerfacecolor=COLOR[m] if st["filled"] else "white",
                        markeredgecolor="white" if st["filled"] else COLOR[m], markeredgewidth=1.3)
                if r.wilcoxon_q_bh < 0.1:
                    ax.annotate("*", (r.mean_interaction, y), xytext=(6, -4), textcoords="offset points",
                                fontsize=10, color=INK)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_yticks(range(len(SUBGROUPS)))
        ax.set_yticklabels([l for _, l in SUBGROUPS])
        ax.invert_yaxis()
        ax.set_xlabel("Interaction (AUROC margin over Cox:\nsubgroup minus rest)")
        ax.set_title(f"{LABEL[m]} vs Cox", loc="left", fontsize=9, color=INK)
    # One legend above both panels: filled = MIMIC, open = EHRSHOT (marker shape = model)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], linestyle="none", marker="o", color=MUTED, markeredgecolor="white",
                      markersize=7, label="MIMIC (fixed test set)"),
               Line2D([], [], linestyle="none", marker="o", markerfacecolor="white", markeredgecolor=MUTED,
                      markeredgewidth=1.3, markersize=7, label="EHRSHOT (pooled 5-fold CV)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.04))
    fig.text(0.01, -0.10, "Positive = the TKG model's margin over Cox is larger inside the subgroup. "
             "Mean over 15 disease × horizon cells, seed-averaged. * BH q < 0.10 (Wilcoxon over cells). "
             "Subgroups with <200 patients are not shown for that dataset.", fontsize=7, color=MUTED)
    fig.tight_layout()
    p = os.path.join(OUT, "fig8_subgroups.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  saved {p}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_sample_size()
    fig_subgroups()

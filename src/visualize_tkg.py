"""Generate 4 exploratory figures for the TKG."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from src.config import OUTPUT_DIR, FIGURES_DIR


ENDPOINT_ORDER = ["MI", "Stroke", "HF", "AF", "PAD", "censored"]
ENDPOINT_PALETTE = {
    "MI": "#d62728", "Stroke": "#9467bd", "HF": "#ff7f0e",
    "AF": "#1f77b4", "PAD": "#2ca02c", "censored": "#7f7f7f",
}
FACT_TYPE_ORDER = ["diagnosis", "procedure", "prescription",
                   "lab", "icu", "omr_bp", "omr_bmi"]


def _figure1_consort(cohort: pd.DataFrame, out_dir: str) -> None:
    """CONSORT-style flow + endpoint pie."""
    # Numbers come from the cohort.csv summary. Hard-coded labels reflect
    # the same exclusion cascade printed by build_cohort().
    steps = [
        ("All MIMIC-IV patients",              364_627),
        ("Adult-age admissions",               223_452),
        ("Has cardiometabolic dx",             134_265),
        ("No endpoint before index",            94_823),
        (">= 2 admissions",                     51_080),
        ("Endpoint OR >= 90d follow-up",        len(cohort)),
    ]
    excl = [
        ("age < 18",                  364_627 - 223_452),
        ("no cardiometabolic dx",     223_452 - 134_265),
        ("endpoint before index",     134_265 - 94_823),
        ("< 2 admissions",             94_823 - 51_080),
        ("< 90d follow-up",            51_080 - len(cohort)),
    ]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.25)

    # Left: CONSORT flow
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) + 1)
    ax.invert_yaxis()
    ax.axis("off")
    for i, (label, n) in enumerate(steps):
        y = i + 0.5
        ax.add_patch(plt.Rectangle((1, y - 0.35), 5, 0.7,
                                    fill=True, color="#cfe2f3",
                                    edgecolor="#1f4e79", linewidth=1.5))
        ax.text(3.5, y, f"{label}\nN = {n:,}", ha="center", va="center",
                fontsize=10, fontweight="bold")
        if i < len(excl):
            dlabel, d_n = excl[i]
            ax.annotate("", xy=(3.5, y + 0.85), xytext=(3.5, y + 0.35),
                        arrowprops=dict(arrowstyle="->", color="black"))
            ax.text(6.4, y + 0.6, f"Excluded: {dlabel}\n(n = {d_n:,})",
                    ha="left", va="center", fontsize=9, color="#990000")
    ax.set_title("Cohort selection flow", fontsize=12, fontweight="bold")

    # Right: pie of endpoints
    ax2 = fig.add_subplot(gs[0, 1])
    cls = cohort["endpoint_type"].value_counts().reindex(ENDPOINT_ORDER).fillna(0).astype(int)
    colors = [ENDPOINT_PALETTE[e] for e in cls.index]
    wedges, _, autopct = ax2.pie(
        cls.values, labels=cls.index, autopct=lambda p: f"{p:.1f}%",
        colors=colors, startangle=90, textprops={"fontsize": 10},
    )
    ax2.set_title(f"Endpoint distribution (N = {len(cohort):,})",
                  fontsize=12, fontweight="bold")
    # legend with counts
    legend_labels = [f"{e}: {cls[e]:,}" for e in cls.index]
    ax2.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.05, 0.5),
               fontsize=9, frameon=False)

    fig.suptitle("Figure 1 — Cohort CONSORT", fontsize=13, fontweight="bold", y=0.98)
    out = os.path.join(out_dir, "fig1_consort.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def _figure2_facts_distribution(facts: pd.DataFrame, out_dir: str) -> None:
    """Stacked bar: x=endpoint_type, bars=fact_type."""
    pivot = (facts.groupby(["endpoint_type", "fact_type"]).size()
             .unstack(fill_value=0).reindex(ENDPOINT_ORDER)
             [FACT_TYPE_ORDER])
    # Normalize per row to mean per patient for fair comparison
    pt_counts = facts.groupby("endpoint_type")["subject_id"].nunique().reindex(ENDPOINT_ORDER)
    pivot_perpt = pivot.div(pt_counts, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    palette = sns.color_palette("tab10", n_colors=len(FACT_TYPE_ORDER))

    pivot.plot(kind="bar", stacked=True, ax=axes[0], color=palette, width=0.7)
    axes[0].set_title("Total facts per endpoint class", fontweight="bold")
    axes[0].set_xlabel("Endpoint class")
    axes[0].set_ylabel("Number of facts")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="fact_type", bbox_to_anchor=(1.02, 1), loc="upper left",
                   fontsize=9)

    pivot_perpt.plot(kind="bar", stacked=True, ax=axes[1], color=palette, width=0.7)
    axes[1].set_title("Mean facts per patient per endpoint class", fontweight="bold")
    axes[1].set_xlabel("Endpoint class")
    axes[1].set_ylabel("Mean facts / patient")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="fact_type", bbox_to_anchor=(1.02, 1), loc="upper left",
                   fontsize=9)

    fig.suptitle("Figure 2 — TKG composition by endpoint class",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(out_dir, "fig2_facts_distribution.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def _figure3_temporal_density(facts: pd.DataFrame, out_dir: str) -> None:
    """KDE of relative_days, per endpoint class, faceted by fact_type."""
    # Subsample for speed
    df = facts[["relative_days", "endpoint_type", "fact_type"]].dropna()
    if len(df) > 1_500_000:
        df = df.sample(n=1_500_000, random_state=42)
    # Clip x range so the heavy censored tail does not dominate
    df = df[(df["relative_days"] >= -365) & (df["relative_days"] <= 1500)]

    ft_present = [f for f in FACT_TYPE_ORDER if f in df["fact_type"].unique()]
    n = len(ft_present)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows),
                              sharex=True)
    axes = np.array(axes).reshape(-1)
    for i, ft in enumerate(ft_present):
        ax = axes[i]
        for ep in ENDPOINT_ORDER:
            sub = df[(df["fact_type"] == ft) & (df["endpoint_type"] == ep)]
            if len(sub) > 1000:
                sns.kdeplot(sub["relative_days"], ax=ax, label=ep,
                            color=ENDPOINT_PALETTE[ep], linewidth=1.5,
                            common_norm=False)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(ft, fontweight="bold")
        ax.set_xlabel("days from index")
        ax.set_ylabel("density")
        if i == 0:
            ax.legend(fontsize=8, title="endpoint", loc="upper right")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Figure 3 — Temporal density of facts relative to index date",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(out_dir, "fig3_temporal_density.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def _figure4_top_concepts(facts: pd.DataFrame, out_dir: str) -> None:
    """Top-20 concept nodes horizontal bar, colored by fact_type."""
    top = (facts.groupby(["concept_id", "fact_type"]).size()
           .reset_index(name="n")
           .sort_values("n", ascending=False).head(20))
    top = top.iloc[::-1]
    colors = sns.color_palette("tab10", n_colors=len(FACT_TYPE_ORDER))
    ft_to_color = dict(zip(FACT_TYPE_ORDER, colors))

    fig, ax = plt.subplots(figsize=(11, 8))
    bar_colors = [ft_to_color[ft] for ft in top["fact_type"]]
    ax.barh(top["concept_id"], top["n"], color=bar_colors)
    for i, (cnt, ft) in enumerate(zip(top["n"], top["fact_type"])):
        ax.text(cnt, i, f"  {cnt:,}", va="center", fontsize=8)
    ax.set_title("Figure 4 — Top-20 concept nodes by frequency",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("count")
    handles = [mpatches.Patch(color=ft_to_color[ft], label=ft)
               for ft in FACT_TYPE_ORDER if ft in top["fact_type"].values]
    ax.legend(handles=handles, title="fact_type", loc="lower right", fontsize=9)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig4_top_concepts.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def visualize_tkg() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Figures dir: {FIGURES_DIR}")
    print("Loading cohort + facts...")
    cohort = pd.read_csv(
        os.path.join(OUTPUT_DIR, "cohort.csv"),
        parse_dates=["index_date", "endpoint_date"],
    )
    facts = pd.read_csv(
        os.path.join(OUTPUT_DIR, "tkg_facts.csv"),
        usecols=["subject_id", "relation", "concept_id", "fact_type",
                 "endpoint_type", "relative_days"],
        low_memory=False,
    )
    print(f"  cohort: {len(cohort):,}, facts: {len(facts):,}")
    _figure1_consort(cohort, FIGURES_DIR)
    _figure2_facts_distribution(facts, FIGURES_DIR)
    _figure3_temporal_density(facts, FIGURES_DIR)
    _figure4_top_concepts(facts, FIGURES_DIR)
    print("Done.")


if __name__ == "__main__":
    visualize_tkg()

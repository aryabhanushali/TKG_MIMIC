"""Descriptive-statistics figures for the cohort and the modeling event table.

Complements `visualize_tkg.py` (which describes the raw TKG) with patient-level
summary statistics on the *modeling* cohort actually fed to the survival models:

  fig13_cohort_statistics.png   age / CCI / sex / comorbidity / follow-up /
                                time-to-event distributions, stratified by endpoint
  fig14_sequence_statistics.png per-patient event-count (sequence-length) and
                                per-patient fact-type composition statistics

  tkg_output/stats/table1_summary.csv   a Table-1-style per-endpoint summary

These read only the already-built modeling artifacts; no model is loaded.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import OUTPUT_DIR, FIGURES_DIR

STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")

ENDPOINT_ORDER = ["MI", "Stroke", "HF", "AF", "PAD", "censored"]
ENDPOINT_PALETTE = {
    "MI": "#d62728", "Stroke": "#9467bd", "HF": "#ff7f0e",
    "AF": "#1f77b4", "PAD": "#2ca02c", "censored": "#7f7f7f",
}
FACT_TYPE_ORDER = ["diagnosis", "procedure", "prescription",
                   "lab", "icu", "omr_bp", "omr_bmi",
                   "vital", "input", "output"]


def _read_events(usecols=None) -> pd.DataFrame:
    """prep_modeling.py writes events.parquet, falling back to events.csv only
    if pyarrow/fastparquet is unavailable at write time -- read whichever
    exists so this doesn't silently break when the write-time environment
    has parquet support."""
    parquet_path = os.path.join(MODELING_DIR, "events.parquet")
    csv_path = os.path.join(MODELING_DIR, "events.csv")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        return df[usecols] if usecols else df
    return pd.read_csv(csv_path, usecols=usecols, low_memory=False)


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-level frame (one row/patient) + long event frame (subject, fact_type)."""
    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    static = pd.read_csv(os.path.join(MODELING_DIR, "static_features.csv"))
    patients = labels.merge(static, on="subject_id", how="left")
    patients["sex"] = np.where(patients["female"] == 1, "Female", "Male")
    events = _read_events(usecols=["subject_id", "fact_type"])
    return patients, events


def _table1(patients: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Per-endpoint summary: counts, demographics, comorbidity, follow-up, events."""
    ev_per_pt = events.groupby("subject_id").size().rename("n_events")
    p = patients.merge(ev_per_pt, on="subject_id", how="left")
    p["n_events"] = p["n_events"].fillna(0).astype(int)

    def _q(s):
        return f"{s.median():.0f} [{s.quantile(.25):.0f}-{s.quantile(.75):.0f}]"

    rows = []
    for ep in ENDPOINT_ORDER:
        sub = p[p["endpoint_type"] == ep]
        if sub.empty:
            continue
        rows.append({
            "endpoint": ep,
            "n_patients": len(sub),
            "pct_of_cohort": round(len(sub) / len(p) * 100, 1),
            "pct_female": round((sub["female"] == 1).mean() * 100, 1),
            "age_median_iqr": _q(sub["age_at_index"]),
            "cci_median_iqr": _q(sub["cci_score"]),
            "n_cardiometa_median": float(sub["num_cardiometa_conditions"].median()),
            "pct_icu": round((sub["had_icu_stay"] == 1).mean() * 100, 1),
            "time_to_event_median_iqr": _q(sub["time_to_event_days"]),
            "n_events_median_iqr": _q(sub["n_events"]),
        })
    overall = {
        "endpoint": "ALL", "n_patients": len(p), "pct_of_cohort": 100.0,
        "pct_female": round((p["female"] == 1).mean() * 100, 1),
        "age_median_iqr": _q(p["age_at_index"]),
        "cci_median_iqr": _q(p["cci_score"]),
        "n_cardiometa_median": float(p["num_cardiometa_conditions"].median()),
        "pct_icu": round((p["had_icu_stay"] == 1).mean() * 100, 1),
        "time_to_event_median_iqr": _q(p["time_to_event_days"]),
        "n_events_median_iqr": _q(p["n_events"]),
    }
    return pd.DataFrame(rows + [overall])


def _figure13_cohort_statistics(patients: pd.DataFrame, out_dir: str) -> None:
    order = [e for e in ENDPOINT_ORDER if e in patients["endpoint_type"].unique()]
    pal = [ENDPOINT_PALETTE[e] for e in order]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (a) Age distribution by endpoint
    sns.violinplot(data=patients, x="endpoint_type", y="age_at_index",
                   order=order, hue="endpoint_type", hue_order=order, palette=pal,
                   legend=False, ax=axes[0, 0], cut=0, inner="quartile")
    axes[0, 0].set_title("Age at index by endpoint", fontweight="bold")
    axes[0, 0].set_xlabel(""); axes[0, 0].set_ylabel("age (years)")

    # (b) CCI by endpoint
    sns.boxplot(data=patients, x="endpoint_type", y="cci_score",
                order=order, hue="endpoint_type", hue_order=order, palette=pal,
                legend=False, ax=axes[0, 1], showfliers=False)
    axes[0, 1].set_title("Charlson Comorbidity Index by endpoint", fontweight="bold")
    axes[0, 1].set_xlabel(""); axes[0, 1].set_ylabel("CCI score")

    # (c) Sex composition by endpoint (stacked %)
    sex = (patients.groupby(["endpoint_type", "sex"]).size()
           .unstack(fill_value=0).reindex(order))
    sex_pct = sex.div(sex.sum(axis=1), axis=0) * 100
    sex_pct.plot(kind="bar", stacked=True, ax=axes[0, 2],
                 color=["#e377c2", "#17becf"], width=0.7)
    axes[0, 2].set_title("Sex composition by endpoint", fontweight="bold")
    axes[0, 2].set_xlabel(""); axes[0, 2].set_ylabel("% of patients")
    axes[0, 2].tick_params(axis="x", rotation=0)
    axes[0, 2].legend(title="", fontsize=9)

    # (d) Number of cardiometabolic index conditions
    nc = (patients.groupby(["endpoint_type", "num_cardiometa_conditions"]).size()
          .unstack(fill_value=0).reindex(order))
    nc_pct = nc.div(nc.sum(axis=1), axis=0) * 100
    nc_pct.plot(kind="bar", stacked=True, ax=axes[1, 0],
                colormap="viridis", width=0.7)
    axes[1, 0].set_title("Cardiometabolic conditions at index", fontweight="bold")
    axes[1, 0].set_xlabel(""); axes[1, 0].set_ylabel("% of patients")
    axes[1, 0].tick_params(axis="x", rotation=0)
    axes[1, 0].legend(title="n conditions", fontsize=8, ncol=2)

    # (e) Follow-up time (all) + ICU share annotation
    sns.histplot(data=patients, x="time_to_event_days", hue="endpoint_type",
                 hue_order=order, palette=ENDPOINT_PALETTE, element="step",
                 stat="density", common_norm=False, bins=40, ax=axes[1, 1])
    axes[1, 1].set_title("Time-to-event / censoring (days)", fontweight="bold")
    axes[1, 1].set_xlabel("days from index"); axes[1, 1].set_ylabel("density")

    # (f) Time-to-event for events only (exclude censored), by cause
    evonly = patients[patients["endpoint_type"] != "censored"]
    causes = [e for e in order if e != "censored"]
    sns.boxplot(data=evonly, x="endpoint_type", y="time_to_event_days",
                order=causes, hue="endpoint_type", hue_order=causes,
                palette=[ENDPOINT_PALETTE[c] for c in causes],
                legend=False, ax=axes[1, 2], showfliers=False)
    axes[1, 2].set_title("Time-to-event by cause (events only)", fontweight="bold")
    axes[1, 2].set_xlabel(""); axes[1, 2].set_ylabel("days from index")

    fig.suptitle("Cohort descriptive statistics (modeling cohort)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(out_dir, "fig13_cohort_statistics.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def _figure14_sequence_statistics(patients: pd.DataFrame, events: pd.DataFrame,
                                   out_dir: str) -> None:
    order = [e for e in ENDPOINT_ORDER if e in patients["endpoint_type"].unique()]
    ev_per_pt = events.groupby("subject_id").size().rename("n_events")
    p = patients.merge(ev_per_pt, on="subject_id", how="left")
    p["n_events"] = p["n_events"].fillna(0).astype(int)

    # per-patient fact-type counts
    ft_counts = (events.groupby(["subject_id", "fact_type"]).size()
                 .unstack(fill_value=0))
    ft_present = [f for f in FACT_TYPE_ORDER if f in ft_counts.columns]
    ft_counts = ft_counts[ft_present]
    ft_with_ep = ft_counts.merge(
        patients[["subject_id", "endpoint_type"]].set_index("subject_id"),
        left_index=True, right_index=True, how="left",
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # (a) Sequence-length (events/patient) distribution, log-x
    ax = axes[0, 0]
    bins = np.logspace(0, np.log10(max(p["n_events"].max(), 10)), 40)
    ax.hist(p["n_events"].clip(lower=1), bins=bins, color="#4c72b0",
            edgecolor="white")
    ax.axvline(256, color="#d62728", linestyle="--", linewidth=1.2,
               label="MAX_SEQ_LEN=256")
    ax.set_xscale("log")
    ax.set_title("Events per patient (sequence length)", fontweight="bold")
    ax.set_xlabel("# pre-index events (log scale)")
    ax.set_ylabel("# patients")
    med = int(p["n_events"].median())
    ax.legend(title=f"median={med}", fontsize=9)

    # (b) Events per patient by endpoint (box, log-y)
    ax = axes[0, 1]
    sns.boxplot(data=p, x="endpoint_type", y="n_events", order=order,
                hue="endpoint_type", hue_order=order,
                palette=[ENDPOINT_PALETTE[e] for e in order],
                legend=False, ax=ax, showfliers=False)
    ax.set_yscale("log")
    ax.set_title("Events per patient by endpoint", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("# events (log scale)")

    # (c) Mean fact-type composition per patient, by endpoint (stacked)
    ax = axes[1, 0]
    mean_ft = ft_with_ep.groupby("endpoint_type")[ft_present].mean().reindex(order)
    mean_ft.plot(kind="bar", stacked=True, ax=ax,
                 color=sns.color_palette("tab10", len(ft_present)), width=0.7)
    ax.set_title("Mean facts per patient by modality", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("mean # facts / patient")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="fact_type", fontsize=8, bbox_to_anchor=(1.01, 1),
              loc="upper left")

    # (d) Fraction of patients with >=1 fact of each modality
    ax = axes[1, 1]
    cov = (ft_counts > 0).mean().reindex(ft_present) * 100
    ax.barh(cov.index, cov.values,
            color=sns.color_palette("tab10", len(ft_present)))
    for i, v in enumerate(cov.values):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_title("Modality coverage (% patients with >=1 fact)",
                 fontweight="bold")
    ax.set_xlabel("% of patients")

    fig.suptitle("Per-patient event-sequence statistics",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(out_dir, "fig14_sequence_statistics.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def visualize_stats() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    print(f"Figures dir: {FIGURES_DIR}")
    print("Loading modeling cohort + event table...")
    patients, events = _load()
    print(f"  patients: {len(patients):,}, events: {len(events):,}")

    table1 = _table1(patients, events)
    out_csv = os.path.join(STATS_DIR, "table1_summary.csv")
    table1.to_csv(out_csv, index=False)
    print("\nTable 1 (per-endpoint summary):")
    print(table1.to_string(index=False))
    print(f"\n  saved {out_csv}")

    _figure13_cohort_statistics(patients, FIGURES_DIR)
    _figure14_sequence_statistics(patients, events, FIGURES_DIR)
    print("Done.")


if __name__ == "__main__":
    visualize_stats()

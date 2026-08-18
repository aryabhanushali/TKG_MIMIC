"""Heatmap views of the TGN-Survival attention explanations.

Reuses the per-patient top-attended events written by `explain.py`
(`tkg_output/explain/per_patient_top_events.csv`) plus the per-cause concept
aggregates, and renders them as heatmaps that make cross-cause structure easy
to read at a glance:

  fig15_concept_cause_heatmap.png   concept x cause attention matrix
                                    (row-normalized -> "which causes does the
                                     model rely on this concept for?")
  fig16_attention_heatmaps.png      (a) fact-type x cause attention share
                                    (b) time-bin x cause attention mass
                                        (when does the model look, per cause?)

Run `python -u -m src.explain` first to produce the inputs.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import OUTPUT_DIR, FIGURES_DIR, read_events_table

EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explain")
MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
CAUSES = ["MI", "Stroke", "HF", "AF", "PAD"]
TOP_N_PER_CAUSE = 12


def _emb_lookup(labels: pd.DataFrame, nodes: pd.DataFrame):
    """Reconstruct emb_idx -> (concept_id, fact_type), mirroring `_prepare_data`."""
    events = read_events_table(usecols=["subject_id", "concept_node_idx"])
    train_ids = set(labels.loc[labels["split"] == "train", "subject_id"])
    train_concepts = sorted(
        events.loc[events["subject_id"].isin(train_ids), "concept_node_idx"]
              .unique().tolist()
    )
    by_idx = nodes.set_index("node_idx")[["concept_id", "fact_type"]]
    cid, ft = {0: "<UNK>"}, {0: "unk"}
    for i, orig in enumerate(train_concepts):
        emb = i + 1
        if orig in by_idx.index:
            cid[emb] = by_idx.loc[orig, "concept_id"]
            ft[emb] = by_idx.loc[orig, "fact_type"]
    return cid, ft


def _figure15_concept_cause_heatmap(by_cause: pd.DataFrame, out_path: str) -> None:
    """Heatmap of mean attention for the union of each cause's top concepts."""
    keep = (by_cause.sort_values("sum_attention", ascending=False)
            .groupby("cause").head(TOP_N_PER_CAUSE)["concept_id"].unique())
    sub = by_cause[by_cause["concept_id"].isin(keep)]
    mat = (sub.pivot_table(index="concept_id", columns="cause",
                           values="mean_attention", aggfunc="mean")
           .reindex(columns=CAUSES).fillna(0.0))
    # Row-normalize so each concept's profile sums to 1 across causes.
    row_norm = mat.div(mat.sum(axis=1).replace(0, 1.0), axis=0)
    # Order rows by the cause they load most heavily on, then magnitude.
    row_norm = row_norm.assign(_top=row_norm.values.argmax(axis=1),
                               _mag=mat.max(axis=1).values)
    row_norm = row_norm.sort_values(["_top", "_mag"],
                                    ascending=[True, False]).drop(columns=["_top", "_mag"])

    h = max(6, 0.32 * len(row_norm))
    fig, ax = plt.subplots(figsize=(8, h))
    sns.heatmap(row_norm, cmap="rocket_r", annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.4, linecolor="white",
                cbar_kws={"label": "row-normalized attention share"}, ax=ax)
    ax.set_title("Concept x cause attention profile\n"
                 f"(union of top-{TOP_N_PER_CAUSE} concepts per cause, "
                 "row-normalized)", fontweight="bold")
    ax.set_xlabel("observed endpoint"); ax.set_ylabel("TKG concept")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _figure16_attention_heatmaps(per_patient: pd.DataFrame, labels: pd.DataFrame,
                                  emb_to_ft: dict, out_path: str) -> None:
    pp = per_patient.copy()
    sid_to_label = dict(zip(labels["subject_id"], labels["endpoint_type"]))
    pp["endpoint"] = pp["subject_id"].map(sid_to_label)
    pp["fact_type"] = pp["concept_emb_idx"].map(emb_to_ft)
    pp = pp[pp["endpoint"].isin(CAUSES)]

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    # (a) fact-type x cause attention share (column-normalized per cause)
    ft = (pp.groupby(["fact_type", "endpoint"])["attention"].sum()
          .unstack(fill_value=0).reindex(columns=CAUSES).fillna(0))
    ft = ft.loc[ft.sum(axis=1).sort_values(ascending=False).index]
    ft_share = ft.div(ft.sum(axis=0).replace(0, 1.0), axis=1)
    sns.heatmap(ft_share, cmap="mako_r", annot=True, fmt=".2f",
                annot_kws={"size": 8}, linewidths=0.4, linecolor="white",
                cbar_kws={"label": "attention share within cause"}, ax=axes[0])
    axes[0].set_title("Fact-type x cause attention share", fontweight="bold")
    axes[0].set_xlabel("observed endpoint"); axes[0].set_ylabel("fact type")

    # (b) time-bin x cause: where (in pre-index time) the model attends
    days = -pp["relative_days"].clip(upper=0)         # days BEFORE index (>=0)
    edges = np.array([0, 30, 90, 180, 365, 730, 1095, 1825])
    bin_labels = ["0-1m", "1-3m", "3-6m", "6-12m", "1-2y", "2-3y", "3-5y"]
    pp = pp.assign(time_bin=pd.cut(days, bins=edges, labels=bin_labels,
                                   include_lowest=True))
    tb = (pp.groupby(["time_bin", "endpoint"], observed=False)["attention"].sum()
          .unstack(fill_value=0).reindex(index=bin_labels, columns=CAUSES).fillna(0))
    tb_share = tb.div(tb.sum(axis=0).replace(0, 1.0), axis=1)
    sns.heatmap(tb_share, cmap="rocket_r", annot=True, fmt=".2f",
                annot_kws={"size": 8}, linewidths=0.4, linecolor="white",
                cbar_kws={"label": "attention share within cause"}, ax=axes[1])
    axes[1].set_title("Pre-index time-window x cause attention share",
                      fontweight="bold")
    axes[1].set_xlabel("observed endpoint")
    axes[1].set_ylabel("time before index")

    fig.suptitle("Where the TGN-Survival model attends",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def run() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pp_path = os.path.join(EXPLAIN_DIR, "per_patient_top_events.csv")
    bc_path = os.path.join(EXPLAIN_DIR, "concept_importance_by_cause.csv")
    if not (os.path.exists(pp_path) and os.path.exists(bc_path)):
        raise FileNotFoundError(
            "Missing explain outputs; run `python -u -m src.explain` first "
            f"(expected {pp_path} and {bc_path}).")

    print("Loading explain outputs + modeling metadata...")
    per_patient = pd.read_csv(pp_path)
    by_cause = pd.read_csv(bc_path)
    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    nodes = pd.read_csv(os.path.join(MODELING_DIR, "node_metadata.csv"))
    _, emb_to_ft = _emb_lookup(labels, nodes)

    _figure15_concept_cause_heatmap(
        by_cause, os.path.join(FIGURES_DIR, "fig15_concept_cause_heatmap.png"))
    _figure16_attention_heatmaps(
        per_patient, labels, emb_to_ft,
        os.path.join(FIGURES_DIR, "fig16_attention_heatmaps.png"))
    print("Done.")


if __name__ == "__main__":
    run()

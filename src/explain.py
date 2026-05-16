"""Attention-rollout explainability for TGN-Survival.

For each test patient, captures the attention-pool weights over the patient's
pre-index event sequence. These weights tell us *which past events the model
relied on most* when computing the patient's risk for each circulatory
endpoint. We aggregate them per true cause to produce:

    1. Top-N concepts per cause (concept_id -> mean attention)
    2. Top-N concepts per cause by frequency in the top-K per patient
    3. Fact-type breakdown per cause (diagnosis / drug / lab / vital / ...)
    4. One illustrative per-patient attention plot per cause
    5. A 5-panel concept-importance figure for the paper

This addresses the JBHI special-issue topic:
"Explainable and Trustworthy Biomedical AI Enabled by Knowledge Graph Reasoning".
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.config import OUTPUT_DIR, FIGURES_DIR
from src.tgn_model import (
    PatientEventsDataset, collate,
    _prepare_data, _set_seed,
    MAX_SEQ_LEN, BATCH_SIZE,
)
from src.tgn_survival import (
    TKGSurvivalNet, CAUSES, MODEL_DIR,
)

EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explain")
TOP_K_EVENTS_PER_PATIENT = 10
TOP_N_CONCEPTS_PER_CAUSE = 20

# Color palette matched to fig8_tgn_survival_per_cause.png
CAUSE_COLORS = {
    "MI":     "#d62728",
    "Stroke": "#9467bd",
    "HF":     "#ff7f0e",
    "AF":     "#1f77b4",
    "PAD":    "#2ca02c",
}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _load_concept_remap_lookup(labels_df: pd.DataFrame):
    """Reconstruct emb_idx -> concept_id mapping (must match _prepare_data
    exactly: train-only concepts indexed 1..N, with 0 = UNK)."""
    events = pd.read_csv(
        os.path.join(OUTPUT_DIR, "modeling", "events.csv"),
        usecols=["subject_id", "concept_node_idx"], low_memory=False,
    )
    nodes = pd.read_csv(os.path.join(OUTPUT_DIR, "modeling", "node_metadata.csv"))
    nodes_by_idx = nodes.set_index("node_idx")["concept_id"].to_dict()
    nodes_ft_by_idx = nodes.set_index("node_idx")["fact_type"].to_dict()

    train_ids = set(labels_df.loc[labels_df["split"] == "train", "subject_id"])
    train_concepts = sorted(
        events.loc[events["subject_id"].isin(train_ids), "concept_node_idx"]
              .unique().tolist()
    )
    emb_to_orig = {i + 1: int(c) for i, c in enumerate(train_concepts)}
    emb_to_concept = {0: "<UNK>"}
    emb_to_facttype = {0: "unk"}
    for emb_idx, orig in emb_to_orig.items():
        emb_to_concept[emb_idx] = nodes_by_idx.get(orig, f"node_{orig}")
        emb_to_facttype[emb_idx] = nodes_ft_by_idx.get(orig, "unknown")
    return emb_to_concept, emb_to_facttype


# --------------------------------------------------------------------------- #
# Inference w/ attention                                                      #
# --------------------------------------------------------------------------- #
def _extract_attention(model, loader, device):
    """Returns a long-format DataFrame: one row per (patient, top-K event)."""
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            for k in ("concept_idx", "edge_type_idx", "t",
                     "v_norm", "v_present", "static", "mask"):
                batch[k] = batch[k].to(device)
            _, attn = model(batch["concept_idx"], batch["edge_type_idx"],
                             batch["t"], batch["v_norm"], batch["v_present"],
                             batch["static"], batch["mask"],
                             return_attention=True)
            attn = attn.cpu().numpy()                       # (B, L)
            concept_emb = batch["concept_idx"].cpu().numpy()
            edge_type = batch["edge_type_idx"].cpu().numpy()
            t_arr = batch["t"].cpu().numpy()
            v_norm = batch["v_norm"].cpu().numpy()
            v_pres = batch["v_present"].cpu().numpy()
            mask_arr = batch["mask"].cpu().numpy()
            sids = batch["sid"].numpy()

            for i in range(len(sids)):
                valid = mask_arr[i].astype(bool)
                a = attn[i, valid]
                if a.size == 0:
                    continue
                k = min(TOP_K_EVENTS_PER_PATIENT, a.size)
                top_idx = np.argpartition(a, -k)[-k:]
                top_idx = top_idx[np.argsort(-a[top_idx])]
                idx_in_full = np.where(valid)[0][top_idx]
                for rank, idx in enumerate(idx_in_full):
                    rows.append({
                        "subject_id": int(sids[i]),
                        "rank": int(rank),
                        "concept_emb_idx": int(concept_emb[i, idx]),
                        "edge_type_idx": int(edge_type[i, idx]),
                        "relative_days": float(t_arr[i, idx]),
                        "value_norm": float(v_norm[i, idx]),
                        "has_value": int(v_pres[i, idx]),
                        "attention": float(a[top_idx[rank]]),
                    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #
def _aggregate_per_cause(per_patient: pd.DataFrame,
                          labels_df: pd.DataFrame,
                          emb_to_concept: dict,
                          emb_to_facttype: dict) -> dict[str, pd.DataFrame]:
    """For each cause, compute concept-level importance from patients with
    that observed endpoint (true positives only)."""
    sid_to_label = dict(zip(labels_df["subject_id"], labels_df["endpoint_type"]))
    per_patient = per_patient.copy()
    per_patient["concept_id"] = per_patient["concept_emb_idx"].map(emb_to_concept)
    per_patient["fact_type"] = per_patient["concept_emb_idx"].map(emb_to_facttype)
    per_patient["endpoint"] = per_patient["subject_id"].map(sid_to_label)

    out = {}
    for cause in CAUSES:
        sub = per_patient[per_patient["endpoint"] == cause]
        if sub.empty:
            out[cause] = pd.DataFrame()
            continue
        # Aggregate: per concept, mean attention + frequency in top-K
        agg = (sub.groupby(["concept_id", "fact_type"])
               .agg(mean_attention=("attention", "mean"),
                    sum_attention=("attention", "sum"),
                    n_patients=("subject_id", "nunique"),
                    n_appearances=("attention", "size"))
               .reset_index())
        agg["pct_patients_with_concept_in_topK"] = (
            agg["n_patients"] / sub["subject_id"].nunique() * 100.0
        )
        agg = agg.sort_values("sum_attention", ascending=False)
        out[cause] = agg
    return out


def _fact_type_breakdown(per_patient: pd.DataFrame,
                          labels_df: pd.DataFrame,
                          emb_to_facttype: dict) -> pd.DataFrame:
    sid_to_label = dict(zip(labels_df["subject_id"], labels_df["endpoint_type"]))
    per_patient = per_patient.copy()
    per_patient["fact_type"] = per_patient["concept_emb_idx"].map(emb_to_facttype)
    per_patient["endpoint"] = per_patient["subject_id"].map(sid_to_label)
    breakdown = (per_patient.groupby(["endpoint", "fact_type"])["attention"]
                 .sum().reset_index())
    pivot = breakdown.pivot(index="endpoint", columns="fact_type",
                             values="attention").fillna(0)
    # Normalize to within-endpoint proportions
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    return pivot


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def _plot_concept_importance(per_cause: dict[str, pd.DataFrame],
                              out_path: str) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 8))
    for ax, cause in zip(axes, CAUSES):
        agg = per_cause.get(cause)
        if agg is None or agg.empty:
            ax.set_title(cause); ax.axis("off"); continue
        top = agg.head(TOP_N_CONCEPTS_PER_CAUSE).iloc[::-1]
        ax.barh(top["concept_id"], top["sum_attention"],
                color=CAUSE_COLORS[cause])
        ax.set_title(f"{cause}\n(top-{TOP_N_CONCEPTS_PER_CAUSE} concepts)",
                     fontweight="bold")
        ax.set_xlabel("Σ attention from pool query")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Figure 9 — Top-attention TKG concepts by observed endpoint",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_fact_type_breakdown(pivot: pd.DataFrame, out_path: str) -> None:
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = pivot.reindex([c for c in CAUSES if c in pivot.index])
    pivot.plot(kind="bar", stacked=True, ax=ax,
                colormap="tab10", width=0.7)
    ax.set_ylabel("share of total attention")
    ax.set_xlabel("observed endpoint")
    ax.set_title("Figure 10 — Where the model attends, by fact type",
                 fontweight="bold")
    ax.legend(title="fact_type", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=9)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #
def run_explain() -> None:
    _set_seed()
    os.makedirs(EXPLAIN_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading data + model...")
    (events_by_sid, static_by_sid, label_by_sid,
     splits, n_concepts, n_edge_types, n_static,
     labels_df) = _prepare_data()

    # Concept remap for interpretability
    emb_to_concept, emb_to_facttype = _load_concept_remap_lookup(labels_df)

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    model = TKGSurvivalNet(
        n_concepts=n_concepts, n_edge_types=n_edge_types, n_static=n_static,
    ).to(device)
    ckpt = torch.load(os.path.join(MODEL_DIR, "best_model.pt"),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  loaded {os.path.join(MODEL_DIR, 'best_model.pt')} "
          f"(epoch {ckpt.get('best_epoch', '?')})  device={device}")

    test_ds = PatientEventsDataset(splits["test"], events_by_sid,
                                     static_by_sid, label_by_sid,
                                     max_len=MAX_SEQ_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate, num_workers=0)

    print("\nExtracting top-K attended events per test patient "
          f"(K={TOP_K_EVENTS_PER_PATIENT})...")
    per_patient = _extract_attention(model, test_loader, device)
    print(f"  rows: {len(per_patient):,}  "
          f"(patients × top-K = {per_patient['subject_id'].nunique()}×{TOP_K_EVENTS_PER_PATIENT})")

    print("\nAggregating per-cause concept importance (TP patients only)...")
    per_cause = _aggregate_per_cause(per_patient, labels_df,
                                       emb_to_concept, emb_to_facttype)
    for cause in CAUSES:
        agg = per_cause.get(cause)
        if agg is None or agg.empty:
            print(f"  {cause}: no TP patients in test set")
            continue
        print(f"\n  === {cause} top {TOP_N_CONCEPTS_PER_CAUSE} concepts ===")
        print(agg.head(TOP_N_CONCEPTS_PER_CAUSE)[
            ["concept_id", "fact_type", "sum_attention",
             "n_patients", "pct_patients_with_concept_in_topK"]
        ].round(3).to_string(index=False))

    # Persist
    per_patient.to_csv(os.path.join(EXPLAIN_DIR, "per_patient_top_events.csv"),
                       index=False)
    big = []
    for cause, df in per_cause.items():
        if df.empty:
            continue
        df = df.copy()
        df["cause"] = cause
        big.append(df.head(TOP_N_CONCEPTS_PER_CAUSE))
    if big:
        pd.concat(big, ignore_index=True).to_csv(
            os.path.join(EXPLAIN_DIR, "concept_importance_by_cause.csv"),
            index=False,
        )

    breakdown = _fact_type_breakdown(per_patient, labels_df, emb_to_facttype)
    breakdown.to_csv(os.path.join(EXPLAIN_DIR, "fact_type_breakdown.csv"))
    print("\nFact-type attention share by endpoint:")
    print(breakdown.round(3).to_string())

    # Figures
    fig9 = os.path.join(FIGURES_DIR, "fig9_concept_importance_by_cause.png")
    fig10 = os.path.join(FIGURES_DIR, "fig10_attention_fact_type_breakdown.png")
    _plot_concept_importance(per_cause, fig9)
    _plot_fact_type_breakdown(breakdown, fig10)

    print(f"\nSaved:")
    print(f"  {os.path.join(EXPLAIN_DIR, 'per_patient_top_events.csv')}")
    print(f"  {os.path.join(EXPLAIN_DIR, 'concept_importance_by_cause.csv')}")
    print(f"  {os.path.join(EXPLAIN_DIR, 'fact_type_breakdown.csv')}")
    print(f"  {fig9}")
    print(f"  {fig10}")


if __name__ == "__main__":
    run_explain()

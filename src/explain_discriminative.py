"""Discriminative attention analysis.

The absolute attention figure (`fig9_concept_importance_by_cause.png`) is
dominated by cohort-entry conditions (HTN, dyslipidemia, T2D) because those
are the patients' inclusion criteria. To surface endpoint-SPECIFIC signal we
compute, for each (cause, concept) pair:

    share(k | c)  = total attention on concept k from patients with cause c
                    ----------------------------------------------------
                    total attention from patients with cause c

    lift(k, c)    = share(k | c)  -  mean_{c' != c} share(k | c')

Concepts with high lift are attended to *more often when the model predicts
this specific endpoint* than for any other endpoint -- i.e., they are the
endpoint-discriminative risk factors the model has learned.

Outputs:
    tkg_output/explain/discriminative_concepts_by_cause.csv
    tkg_output/figures/fig11_discriminative_concepts_by_cause.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, FIGURES_DIR

EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explain")
MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
CAUSES = ["MI", "Stroke", "HF", "AF", "PAD"]
TOP_N = 15
MIN_PATIENTS = 8  # require at least this many TP patients with the concept in top-K
CAUSE_COLORS = {"MI": "#d62728", "Stroke": "#9467bd", "HF": "#ff7f0e",
                "AF": "#1f77b4", "PAD": "#2ca02c"}


def run() -> None:
    print("Loading per-patient attended events + labels...")
    per_patient = pd.read_csv(os.path.join(EXPLAIN_DIR, "per_patient_top_events.csv"))
    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    nodes = pd.read_csv(os.path.join(MODELING_DIR, "node_metadata.csv"))

    sid_to_label = dict(zip(labels["subject_id"], labels["endpoint_type"]))
    per_patient["endpoint"] = per_patient["subject_id"].map(sid_to_label)
    # Re-map concept_emb_idx to concept_id for readability (same logic as explain.py)
    events = pd.read_csv(
        os.path.join(MODELING_DIR, "events.csv"),
        usecols=["subject_id", "concept_node_idx"], low_memory=False,
    )
    train_ids = set(labels.loc[labels["split"] == "train", "subject_id"])
    train_concepts = sorted(
        events.loc[events["subject_id"].isin(train_ids), "concept_node_idx"]
              .unique().tolist()
    )
    emb_to_orig = {i + 1: int(c) for i, c in enumerate(train_concepts)}
    nodes_by_idx = nodes.set_index("node_idx")[["concept_id", "fact_type"]]
    emb_to_cid = {0: "<UNK>"}
    emb_to_ft = {0: "unk"}
    for emb_idx, orig in emb_to_orig.items():
        if orig in nodes_by_idx.index:
            emb_to_cid[emb_idx] = nodes_by_idx.loc[orig, "concept_id"]
            emb_to_ft[emb_idx] = nodes_by_idx.loc[orig, "fact_type"]
    per_patient["concept_id"] = per_patient["concept_emb_idx"].map(emb_to_cid)
    per_patient["fact_type"] = per_patient["concept_emb_idx"].map(emb_to_ft)

    # Compute within-cause share for each concept (sum_attention / cause_total_attention)
    shares = {}
    n_patients_per_cause = {}
    for cause in CAUSES:
        sub = per_patient[per_patient["endpoint"] == cause]
        if sub.empty:
            shares[cause] = {}
            n_patients_per_cause[cause] = 0
            continue
        total = sub["attention"].sum()
        per_concept = sub.groupby("concept_id").agg(
            sum_attention=("attention", "sum"),
            n_patients=("subject_id", "nunique"),
            fact_type=("fact_type", "first"),
        ).reset_index()
        per_concept["share"] = per_concept["sum_attention"] / max(total, 1e-9)
        shares[cause] = per_concept.set_index("concept_id")
        n_patients_per_cause[cause] = int(sub["subject_id"].nunique())

    # Build full concept universe (concepts that appeared in any cause's top-K)
    all_concepts = sorted(set().union(*[s.index for s in shares.values() if not s.empty]))
    print(f"  concept universe: {len(all_concepts):,} concepts")

    # For each cause, compute lift = share(k|c) - mean_{c'!=c} share(k|c')
    rows = []
    for cause in CAUSES:
        s_c = shares[cause]
        if s_c.empty:
            continue
        for k in all_concepts:
            share_c = float(s_c.loc[k, "share"]) if k in s_c.index else 0.0
            n_pts = int(s_c.loc[k, "n_patients"]) if k in s_c.index else 0
            ft = (s_c.loc[k, "fact_type"] if k in s_c.index
                  else emb_to_ft.get(0, "unknown"))
            others = [float(shares[oc].loc[k, "share"]) if (not shares[oc].empty
                       and k in shares[oc].index) else 0.0
                      for oc in CAUSES if oc != cause]
            share_others_mean = float(np.mean(others)) if others else 0.0
            lift = share_c - share_others_mean
            rows.append({
                "cause": cause,
                "concept_id": k,
                "fact_type": ft,
                "share_in_cause": share_c,
                "share_in_other_causes_mean": share_others_mean,
                "lift": lift,
                "n_tp_patients_with_concept": n_pts,
                "n_tp_patients_total": n_patients_per_cause[cause],
                "pct_tp_with_concept": (n_pts / max(n_patients_per_cause[cause], 1)
                                         * 100.0),
            })
    df = pd.DataFrame(rows)
    df = df[df["n_tp_patients_with_concept"] >= MIN_PATIENTS]
    out_csv = os.path.join(EXPLAIN_DIR, "discriminative_concepts_by_cause.csv")
    df.sort_values(["cause", "lift"], ascending=[True, False]).to_csv(out_csv, index=False)

    # Pretty-print top-N per cause
    print(f"\n=== TOP {TOP_N} DISCRIMINATIVE CONCEPTS PER CAUSE "
          f"(min {MIN_PATIENTS} TP patients) ===")
    for cause in CAUSES:
        sub = (df[df["cause"] == cause]
                .sort_values("lift", ascending=False).head(TOP_N))
        if sub.empty:
            print(f"\n  {cause}: (no eligible concepts)")
            continue
        print(f"\n  --- {cause} (n_tp={n_patients_per_cause[cause]}) ---")
        print(sub[["concept_id", "fact_type", "lift", "share_in_cause",
                   "share_in_other_causes_mean", "pct_tp_with_concept"]]
              .round(4).to_string(index=False))

    # Figure: per-cause top-N discriminative concepts
    fig, axes = plt.subplots(1, 5, figsize=(22, 8))
    for ax, cause in zip(axes, CAUSES):
        sub = (df[df["cause"] == cause]
                .sort_values("lift", ascending=False).head(TOP_N).iloc[::-1])
        if sub.empty:
            ax.set_title(cause); ax.axis("off"); continue
        ax.barh(sub["concept_id"], sub["lift"], color=CAUSE_COLORS[cause])
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(f"{cause}\n(lift over other causes)", fontweight="bold")
        ax.set_xlabel("attention share lift")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Figure 11 — Discriminative TKG concepts by endpoint "
                 "(model attention share over other causes)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_fig = os.path.join(FIGURES_DIR, "fig11_discriminative_concepts_by_cause.png")
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved:\n  {out_csv}\n  {out_fig}")


if __name__ == "__main__":
    run()

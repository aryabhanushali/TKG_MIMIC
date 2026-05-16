"""Prep TKG for prognostic modeling: pre-index events, splits, labels."""
import os
import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, PRE_INDEX_WINDOW_DAYS, SEED

MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
ENDPOINT_ORDER = ["MI", "Stroke", "HF", "AF", "PAD", "censored"]


def _stratified_split(df: pd.DataFrame, label_col: str,
                      ratios=(0.70, 0.15, 0.15), seed: int = SEED) -> dict[int, str]:
    """Patient-level stratified split. Returns {subject_id: 'train'|'val'|'test'}."""
    rng = np.random.default_rng(seed)
    assignment: dict[int, str] = {}
    for label, grp in df.groupby(label_col):
        ids = grp["subject_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        for sid in ids[:n_train]:
            assignment[int(sid)] = "train"
        for sid in ids[n_train:n_train + n_val]:
            assignment[int(sid)] = "val"
        for sid in ids[n_train + n_val:]:
            assignment[int(sid)] = "test"
    return assignment


def _to_parquet_or_csv(df: pd.DataFrame, base_path: str) -> str:
    """Try parquet; fall back to csv if pyarrow/fastparquet is unavailable."""
    try:
        path = base_path + ".parquet"
        df.to_parquet(path, index=False)
        return path
    except Exception as e:
        print(f"  parquet failed ({type(e).__name__}: {e}); writing csv")
        path = base_path + ".csv"
        df.to_csv(path, index=False)
        return path


def prep_modeling() -> None:
    os.makedirs(MODELING_DIR, exist_ok=True)
    print(f"Output: {MODELING_DIR}")

    print("\nLoading cohort, facts, nodes...")
    cohort = pd.read_csv(
        os.path.join(OUTPUT_DIR, "cohort.csv"),
        parse_dates=["index_date", "endpoint_date"],
    )
    facts = pd.read_csv(
        os.path.join(OUTPUT_DIR, "tkg_facts.csv"),
        parse_dates=["timestamp_start", "timestamp_end", "index_date"],
        low_memory=False,
    )
    nodes = pd.read_csv(os.path.join(OUTPUT_DIR, "node_index.csv"))
    if "value_num" not in facts.columns:
        facts["value_num"] = np.nan
    print(f"  cohort={len(cohort):,}, facts={len(facts):,}, nodes={len(nodes):,}")

    # --- 1. Pre-index window filter ----------------------------------------
    print(f"\nFiltering to pre-index window: relative_days in "
          f"[-{PRE_INDEX_WINDOW_DAYS}, 0]")
    pre = facts[(facts["relative_days"] >= -PRE_INDEX_WINDOW_DAYS)
                & (facts["relative_days"] <= 0)].copy()
    print(f"  events kept: {len(pre):,} ({len(pre)/len(facts)*100:.1f}% of all facts)")

    # Compute relative_days_end where timestamp_end exists (prescriptions, icu)
    pre = pre.merge(
        cohort[["subject_id", "index_date"]].rename(columns={"index_date": "_idx"}),
        on="subject_id", how="left",
    )
    pre["timestamp_end"] = pd.to_datetime(pre["timestamp_end"], errors="coerce")
    pre["relative_days_end"] = ((pre["timestamp_end"] - pre["_idx"]).dt.days)

    # --- 2. Node and edge type indexing ------------------------------------
    print("\nIndexing nodes and edge types...")
    concept_to_idx = dict(zip(nodes["concept_id"], nodes["node_idx"]))
    # patient nodes already exist in node_index as PATIENT_{subject_id}
    pre["concept_node_idx"] = pre["concept_id"].map(concept_to_idx)
    pre["patient_node_idx"] = pre["subject_id"].apply(
        lambda s: concept_to_idx[f"PATIENT_{s}"]
    )

    rel_types = sorted(pre["relation"].unique().tolist())
    rel_to_idx = {r: i for i, r in enumerate(rel_types)}
    pre["edge_type_idx"] = pre["relation"].map(rel_to_idx)
    print(f"  edge types ({len(rel_types)}): " + ", ".join(
        f"{r}={i}" for r, i in rel_to_idx.items()))

    # Restrict node_index to concept nodes that appear in pre-index OR are patient nodes
    active_concepts = set(pre["concept_id"].unique())
    pat_concepts = set(nodes[nodes["fact_type"] == "patient"]["concept_id"])
    active_nodes = nodes[nodes["concept_id"].isin(active_concepts | pat_concepts)].copy()
    active_nodes = active_nodes.reset_index(drop=True)
    print(f"  active concept nodes: {(active_nodes['fact_type']!='patient').sum():,}, "
          f"patient nodes: {(active_nodes['fact_type']=='patient').sum():,}")

    # --- 3. Stratified 70/15/15 split --------------------------------------
    print("\nStratified split (70/15/15 by endpoint_type, seed={})...".format(SEED))
    assignment = _stratified_split(cohort[["subject_id", "endpoint_type"]],
                                    "endpoint_type")
    splits = pd.DataFrame({
        "subject_id": list(assignment.keys()),
        "split": list(assignment.values()),
    })
    cohort_with_split = cohort.merge(splits, on="subject_id", how="left")

    print("  per-split endpoint distribution:")
    crosstab = pd.crosstab(cohort_with_split["endpoint_type"],
                            cohort_with_split["split"])[["train", "val", "test"]]
    crosstab = crosstab.reindex(ENDPOINT_ORDER)
    print(crosstab.to_string())
    print("  totals: train={}, val={}, test={}".format(
        (cohort_with_split["split"] == "train").sum(),
        (cohort_with_split["split"] == "val").sum(),
        (cohort_with_split["split"] == "test").sum()))

    # --- 4. Labels and static features -------------------------------------
    print("\nBuilding labels...")
    labels = cohort_with_split[[
        "subject_id", "endpoint_type", "follow_up_days", "split",
    ]].rename(columns={"follow_up_days": "time_to_event_days"})
    labels["event_observed"] = (labels["endpoint_type"] != "censored").astype(int)
    # multiclass label index (cause-specific). 'censored' -> -1 for prognostic loss.
    label_map = {ep: i for i, ep in enumerate(ENDPOINT_ORDER[:-1])}
    label_map["censored"] = -1
    labels["endpoint_idx"] = labels["endpoint_type"].map(label_map)

    print("Building static features (age, gender, CCI, ...)...")
    static = cohort_with_split[[
        "subject_id", "age_at_index", "gender",
        "cci_score", "num_cardiometa_conditions", "had_icu_stay",
    ]].copy()
    static["female"] = (static["gender"] == "F").astype(int)
    static["had_icu_stay"] = static["had_icu_stay"].astype(int)
    static = static.drop(columns=["gender"])

    # --- 5. Value normalization (z-score per concept, train-only stats) ----
    print("\nNormalizing event values (z-score per concept from train set)...")
    train_sids = set(cohort_with_split.loc[
        cohort_with_split["split"] == "train", "subject_id"])
    has_val = pre["value_num"].notna()
    train_vals = pre[has_val & pre["subject_id"].isin(train_sids)]
    val_stats = (train_vals.groupby("concept_id")["value_num"]
                  .agg(["mean", "std", "count"])
                  .reset_index())
    val_stats["std"] = val_stats["std"].replace(0, 1.0).fillna(1.0)
    print(f"  concepts with numeric values (train): {len(val_stats):,}")
    print(f"  total events with values: {int(has_val.sum()):,} "
          f"({has_val.mean()*100:.1f}% of pre-index events)")
    stats_map_mean = dict(zip(val_stats["concept_id"], val_stats["mean"]))
    stats_map_std  = dict(zip(val_stats["concept_id"], val_stats["std"]))
    pre["v_mean"] = pre["concept_id"].map(stats_map_mean)
    pre["v_std"]  = pre["concept_id"].map(stats_map_std).fillna(1.0)
    pre["value_norm"] = ((pre["value_num"] - pre["v_mean"]) / pre["v_std"]).fillna(0.0)
    # clip extreme outliers (>10 sigma) to ±10
    pre["value_norm"] = pre["value_norm"].clip(-10.0, 10.0)
    pre["value_present"] = pre["value_num"].notna().astype(np.int8)

    # Save the per-concept stats for reproducibility
    val_stats.to_csv(os.path.join(MODELING_DIR, "value_stats.csv"), index=False)

    # --- 6. Events table for GNN ------------------------------------------
    print("\nAssembling events table...")
    events = pre[[
        "subject_id", "patient_node_idx", "concept_node_idx",
        "edge_type_idx", "relation", "fact_type",
        "relative_days", "relative_days_end",
        "value_num", "value_norm", "value_present",
    ]].copy()
    events = events.sort_values(["subject_id", "relative_days"]).reset_index(drop=True)
    print(f"  events rows: {len(events):,}")

    # Coverage: how many patients have at least 1 pre-index event?
    pts_with_events = set(events["subject_id"].unique())
    n_empty = len(cohort) - len(pts_with_events)
    print(f"  patients with >=1 pre-index event: {len(pts_with_events):,} "
          f"({len(pts_with_events)/len(cohort)*100:.1f}%)")
    print(f"  patients with NO pre-index events: {n_empty:,}")
    per_pt = events.groupby("subject_id").size()
    print(f"  events/patient: median={int(per_pt.median())}, "
          f"p25={int(per_pt.quantile(0.25))}, p75={int(per_pt.quantile(0.75))}, "
          f"max={int(per_pt.max())}")

    # --- 6. Save -----------------------------------------------------------
    print("\nSaving...")
    splits_path = os.path.join(MODELING_DIR, "splits.csv")
    labels_path = os.path.join(MODELING_DIR, "labels.csv")
    static_path = os.path.join(MODELING_DIR, "static_features.csv")
    nodes_path = os.path.join(MODELING_DIR, "node_metadata.csv")
    events_base = os.path.join(MODELING_DIR, "events")
    edge_types_path = os.path.join(MODELING_DIR, "edge_types.csv")

    splits.to_csv(splits_path, index=False)
    labels.to_csv(labels_path, index=False)
    static.to_csv(static_path, index=False)
    active_nodes.to_csv(nodes_path, index=False)
    pd.DataFrame({"edge_type": list(rel_to_idx.keys()),
                  "edge_type_idx": list(rel_to_idx.values())}
                 ).to_csv(edge_types_path, index=False)
    events_path = _to_parquet_or_csv(events, events_base)

    print(f"  {splits_path}")
    print(f"  {labels_path}")
    print(f"  {static_path}")
    print(f"  {nodes_path}")
    print(f"  {edge_types_path}")
    print(f"  {events_path}")

    # --- 7. Model-ready summary -------------------------------------------
    print("\n=== MODELING DATA SUMMARY ===")
    print(f"Patients:         {len(cohort):,}")
    print(f"  train:          {(labels['split']=='train').sum():,}")
    print(f"  val:            {(labels['split']=='val').sum():,}")
    print(f"  test:           {(labels['split']=='test').sum():,}")
    print(f"Pre-index events: {len(events):,}")
    print(f"Concept nodes:    {(active_nodes['fact_type']!='patient').sum():,}")
    print(f"Patient nodes:    {(active_nodes['fact_type']=='patient').sum():,}")
    print(f"Edge types:       {len(rel_types)}")
    print(f"Pre-index window: [-{PRE_INDEX_WINDOW_DAYS}, 0] days from index_date "
          f"(inclusive)")
    print(f"Label scheme:     multiclass (5 endpoints + censored) and "
          f"time-to-event")


if __name__ == "__main__":
    prep_modeling()

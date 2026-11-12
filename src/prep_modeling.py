"""Prep TKG for prognostic modeling: pre-index events, splits, labels."""
import hashlib
import os
import numpy as np
import pandas as pd

from src.config import (DATA_DIR, OUTPUT_DIR, PRE_INDEX_WINDOW_DAYS, SEED,
                        SPLIT_MODE, TEMPORAL_TRAIN_MAX_YEAR,
                        TEMPORAL_VAL_MAX_YEAR, CV_FOLDS, CV_FOLD,
                        CV_HASH_SEED, SPLIT_TAG, TRAIN_FRAC,
                        TRAIN_SUBSAMPLE_HASH_SEED)

MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
SPLIT_MODE_MARKER = os.path.join(MODELING_DIR, "split_mode.txt")
ENDPOINT_ORDER = ["MI", "Stroke", "HF", "AF", "PAD", "censored"]


def _check_split_mode_marker() -> None:
    """Refuse to overwrite a modeling/ dir built under a different split mode.

    Every downstream script reads modeling/labels.csv, so silently replacing
    the main random split with a temporal one (or vice versa) would make every
    later rerun in that OUTPUT_DIR evaluate on a different test set than the
    results already reported from it. A dir with labels.csv but no marker
    predates this check and was necessarily built in random mode.
    """
    if os.path.exists(SPLIT_MODE_MARKER):
        with open(SPLIT_MODE_MARKER) as f:
            existing = f.read().strip()
    elif os.path.exists(os.path.join(MODELING_DIR, "labels.csv")):
        existing = "random"
    else:
        return
    if existing != SPLIT_TAG:
        raise RuntimeError(
            f"{MODELING_DIR} was built with split mode {existing!r} but "
            f"this run is {SPLIT_TAG!r}. Point MIMIC_OUTPUT_DIR at a "
            f"separate directory for this split mode / fold.")


def _temporal_split(cohort: pd.DataFrame) -> tuple[dict[int, str], pd.Series]:
    """Patient-level split by approximate real year of the index admission.

    Returns ({subject_id: 'train'|'val'|'test'}, era_year per subject_id).
    See the TKG_SPLIT_MODE note in config.py for the era_year definition and
    why the default cut points guarantee train index years < test index years.
    """
    patients = pd.read_csv(
        os.path.join(DATA_DIR, "patients.csv.gz"),
        usecols=["subject_id", "anchor_year", "anchor_year_group"],
    )
    df = cohort[["subject_id", "index_date"]].merge(
        patients, on="subject_id", how="left")
    if df["anchor_year_group"].isna().any():
        raise ValueError("cohort patients missing from patients.csv.gz")
    group_start = df["anchor_year_group"].str[:4].astype(int)
    era_year = group_start + (df["index_date"].dt.year - df["anchor_year"])
    split = np.where(era_year <= TEMPORAL_TRAIN_MAX_YEAR, "train",
                     np.where(era_year <= TEMPORAL_VAL_MAX_YEAR, "val", "test"))
    assignment = dict(zip(df["subject_id"].astype(int), split))
    return assignment, pd.Series(era_year.to_numpy(), index=df["subject_id"])


def _patient_bucket(subject_id: int, seed: int, n_buckets: int = 10_000) -> int:
    """Deterministic bucket in [0, n_buckets) for one (subject_id, seed) pair,
    independent of any other patient's presence, count, or ordering."""
    h = hashlib.sha256(f"{seed}:{subject_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") % n_buckets


def _stratified_split(df: pd.DataFrame, label_col: str,
                      ratios=(0.70, 0.15, 0.15), seed: int = SEED) -> dict[int, str]:
    """Patient-level stratified split. Returns {subject_id: 'train'|'val'|'test'}.

    Each patient's split is a pure function of (subject_id, seed) via a hash
    bucket, not a shared-RNG shuffle over the group they happen to fall in.
    This was previously np.random.default_rng(seed).shuffle(ids) applied once
    per stratification group in sequence -- since shuffle() consumes a number
    of draws proportional to group size, ANY upstream change to ANY group's
    patient count (even one unrelated to a given patient, e.g. one more/fewer
    censored patient from an ICD code fix elsewhere) shifted the shared RNG's
    internal state and reassigned splits for patients who were never touched
    by that change. Confirmed empirically: a 2-of-302-patient PAD code fix
    (removing two non-atherosclerotic codes) reshuffled enough of the test
    set that PAD's Cox AUROC at 3 years swung from 0.592 to 0.487 and a
    patient-graph-vs-Cox PAD comparison flipped from "mixed, losing at short
    horizons" to "wins at every horizon, p<0.0001" -- with no code change to
    PAD's own definition responsible for the second effect. Hash-bucketing
    makes each patient's assignment stable under any such change to anyone
    else's cohort membership; per-group ratios will deviate from the target
    70/15/15 by ordinary hash-partitioning noise (negligible at these group
    sizes) rather than exact rounding, which is the standard, accepted
    trade-off for this property.
    """
    assignment: dict[int, str] = {}
    for _, grp in df.groupby(label_col):
        ids = grp["subject_id"].to_numpy()
        n_buckets = 10_000
        train_cut = int(round(n_buckets * ratios[0]))
        val_cut = train_cut + int(round(n_buckets * ratios[1]))
        for sid in ids:
            b = _patient_bucket(int(sid), seed, n_buckets)
            if b < train_cut:
                assignment[int(sid)] = "train"
            elif b < val_cut:
                assignment[int(sid)] = "val"
            else:
                assignment[int(sid)] = "test"
    return assignment


def _cv_split(subject_ids, fold: int, n_folds: int,
              seed: int = CV_HASH_SEED) -> dict[int, str]:
    """K-fold assignment from the same per-patient hash as _stratified_split
    (stable under any other patient's cohort membership). test = `fold`,
    val = `fold + 1` (mod K), train = everything else."""
    if not 0 <= fold < n_folds:
        raise ValueError(f"TKG_CV_FOLD={fold} outside [0, {n_folds})")
    out = {}
    for sid in subject_ids:
        f = _patient_bucket(int(sid), seed) * n_folds // 10_000
        out[int(sid)] = ("test" if f == fold
                         else "val" if f == (fold + 1) % n_folds else "train")
    return out


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


def _subsample_train(cohort, facts, nodes):
    """TKG_TRAIN_FRAC < 1: drop all but a hashed fraction of TRAIN patients.

    The split is recomputed here exactly as step 3 below computes it (it is a
    pure per-patient hash, so the reduced cohort gets identical assignments),
    and only train patients are candidates for removal. Removed patients lose
    their cohort row, facts and PATIENT_ node, so no model sees them at all.
    """
    if SPLIT_MODE != "random":
        raise ValueError("TKG_TRAIN_FRAC is only supported with TKG_SPLIT_MODE=random")
    assignment = _stratified_split(cohort[["subject_id", "endpoint_type"]], "endpoint_type")
    cut = int(round(TRAIN_FRAC * 10_000))
    drop = {sid for sid, s in assignment.items()
            if s == "train" and _patient_bucket(sid, TRAIN_SUBSAMPLE_HASH_SEED) >= cut}
    n_train = sum(s == "train" for s in assignment.values())
    print(f"\nTKG_TRAIN_FRAC={TRAIN_FRAC:g}: keeping {n_train - len(drop):,} of "
          f"{n_train:,} train patients (val/test unchanged)")
    cohort = cohort[~cohort["subject_id"].isin(drop)].reset_index(drop=True)
    facts = facts[~facts["subject_id"].isin(drop)]
    nodes = nodes[~nodes["concept_id"].isin({f"PATIENT_{s}" for s in drop})]
    return cohort, facts, nodes


def prep_modeling() -> None:
    _check_split_mode_marker()
    os.makedirs(MODELING_DIR, exist_ok=True)
    print(f"Output: {MODELING_DIR}  (split mode: {SPLIT_TAG})")

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
    if TRAIN_FRAC < 1.0:
        cohort, facts, nodes = _subsample_train(cohort, facts, nodes)
        print(f"  after subsample: cohort={len(cohort):,}, facts={len(facts):,}")

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

    # --- 3. Split ----------------------------------------------------------
    if SPLIT_MODE == "temporal":
        print(f"\nTemporal split (train: era_year <= {TEMPORAL_TRAIN_MAX_YEAR}, "
              f"val: <= {TEMPORAL_VAL_MAX_YEAR}, test: later)...")
        assignment, era_year = _temporal_split(cohort)
    elif SPLIT_MODE == "random":
        print("\nStratified split (70/15/15 by endpoint_type, seed={})...".format(SEED))
        assignment = _stratified_split(cohort[["subject_id", "endpoint_type"]],
                                        "endpoint_type")
    elif SPLIT_MODE == "cv":
        print(f"\n{CV_FOLDS}-fold CV split: test=fold {CV_FOLD}, "
              f"val=fold {(CV_FOLD + 1) % CV_FOLDS}, train=rest (hash seed {CV_HASH_SEED})...")
        assignment = _cv_split(cohort["subject_id"], CV_FOLD, CV_FOLDS)
    else:
        raise ValueError(f"unknown TKG_SPLIT_MODE {SPLIT_MODE!r}")
    splits = pd.DataFrame({
        "subject_id": list(assignment.keys()),
        "split": list(assignment.values()),
    })
    if SPLIT_MODE == "temporal":
        splits["era_year"] = splits["subject_id"].map(era_year)
    cohort_with_split = cohort.merge(splits[["subject_id", "split"]],
                                     on="subject_id", how="left")

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

    # Vocabulary coverage: share of val/test events whose concept was seen in
    # train (anything else becomes UNK in tgn_model.py). Under the temporal
    # split this exposes the Oct-2015 ICD-9 -> ICD-10 switch directly.
    print("\nTrain-vocabulary coverage of events, by concept family:")
    split_of = dict(zip(cohort_with_split["subject_id"], cohort_with_split["split"]))
    cov = pre[["subject_id", "concept_id"]].copy()
    cov["split"] = cov["subject_id"].map(split_of)
    train_vocab = set(cov.loc[cov["split"] == "train", "concept_id"])
    cov["in_train"] = cov["concept_id"].isin(train_vocab)
    cov["family"] = np.where(cov["concept_id"].str.startswith("ICD10_"), "ICD10",
                    np.where(cov["concept_id"].str.startswith("ICD9_"), "ICD9",
                             "non-ICD"))
    print(cov.groupby(["family", "split"])["in_train"].mean()
             .unstack("split").reindex(columns=["val", "test"]).round(3)
             .to_string())

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
    with open(SPLIT_MODE_MARKER, "w") as f:
        f.write(SPLIT_TAG + "\n")

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

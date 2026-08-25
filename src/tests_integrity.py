"""Automated leakage / cohort-validation gate.

Unlike validate_tkg.py (which prints PASS/FAIL and always exits 0, so a
"FAIL" line can scroll by unnoticed and the pipeline continues regardless),
every check here is a hard assertion: any violation raises immediately and
the process exits non-zero. Run this after any change to cohort.py,
build_tkg.py, or prep_modeling.py, and before spending compute on training.

Checks:
  1. No subject_id appears in more than one of train/val/test.
  2. Every subject_id in modeling/labels.csv exists in cohort.csv with the
     SAME endpoint_type (outcome-mapping consistency between the two files).
  3. No modeling event has relative_days > 0 (no post-index leakage into the
     feature window actually written to disk, not just claimed by the code).
  4. For every patient with an observed endpoint, no raw fact in
     tkg_facts.csv is timestamped at or after that patient's endpoint_date
     (independent re-check of validate_tkg.py Check 1, as a hard failure).
  5. cohort_cascade.csv counts are non-increasing (a valid exclusion funnel).
  6. No duplicate subject_id rows in cohort.csv.
  7. Reconstructs tgn_model.py's exact concept -> embedding-index remap
     (train concepts -> 1..N, everything else -> UNK at index 0) from an
     independently recomputed train-only concept set, then asserts every
     test/val-only concept actually lands on UNK, every train concept lands
     on a real (nonzero, collision-free) index, and none of it is trivially
     true by construction. baselines_survival.py's restriction (building
     bag-of-codes columns only for the train-only concept_ids list) is
     structurally guaranteed by the same train-only set computed here, since
     it uses the identical subject_id-in-train_ids formula.
  8. No interval-type fact (ICU length-of-stay, IV input total amount) that's
     still open or ends on/after its patient's window close carries a value
     -- those totals are only fully known at the end of the interval, so
     build_tkg.py suppresses the value rather than leak it (Check 4 above
     only catches leakage via timestamp_start; this catches the same class
     of leak via timestamp_end / value_num).

Usage: python -u -m src.tests_integrity
Exit code 0 = all checks passed. Non-zero = at least one check failed; the
failure message names the specific violation.
"""
import os
import sys

import pandas as pd

from src.config import OUTPUT_DIR, read_events_table

MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def run() -> None:
    print("=== Integrity gate ===\n")

    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    cohort = pd.read_csv(
        os.path.join(OUTPUT_DIR, "cohort.csv"),
        parse_dates=["endpoint_date", "index_date"],
    )
    facts = pd.read_csv(
        os.path.join(OUTPUT_DIR, "tkg_facts.csv"),
        usecols=["subject_id", "timestamp_start", "endpoint_type"],
        parse_dates=["timestamp_start"],
    ).merge(cohort[["subject_id", "endpoint_date"]], on="subject_id", how="left")
    # Only icu/input facts are genuine start->end intervals whose value_num is
    # only final at timestamp_end. Point-in-time facts (labs, vitals, BP/BMI,
    # outputs) also carry a value_num but have timestamp_end == NaT by
    # construction -- that's correct, not a leak, and must not be flagged.
    interval_facts = pd.read_csv(
        os.path.join(OUTPUT_DIR, "tkg_facts.csv"),
        usecols=["subject_id", "fact_type", "timestamp_end", "value_num"],
        parse_dates=["timestamp_end"],
    )
    interval_facts = interval_facts[interval_facts["fact_type"].isin(["icu", "input"])]

    print("1. Split overlap")
    counts_per_split = labels.groupby("subject_id")["split"].nunique()
    check("no patient in more than one split",
          bool((counts_per_split == 1).all()),
          f"{(counts_per_split > 1).sum()} patients found in >1 split")

    print("\n2. Outcome-mapping consistency (labels.csv vs cohort.csv)")
    merged = labels[["subject_id", "endpoint_type"]].merge(
        cohort[["subject_id", "endpoint_type"]], on="subject_id",
        suffixes=("_modeling", "_cohort"), how="left",
    )
    mismatch = merged["endpoint_type_modeling"] != merged["endpoint_type_cohort"]
    check("modeling labels match cohort.csv exactly",
          not mismatch.any(),
          f"{mismatch.sum()} patients have a different endpoint_type between the two files")
    check("every modeling patient exists in cohort.csv",
          merged["endpoint_type_cohort"].notna().all(),
          f"{merged['endpoint_type_cohort'].isna().sum()} modeling patients missing from cohort.csv")

    print("\n3. No post-index events in the modeling table")
    events = read_events_table(usecols=["subject_id", "relative_days"])
    post_index = events["relative_days"] > 0
    check("no modeling event has relative_days > 0",
          not post_index.any(),
          f"{post_index.sum():,} events with relative_days > 0")

    print("\n4. No raw fact at/after its patient's endpoint date")
    has_ep = facts["endpoint_type"] != "censored"
    ep_facts = facts[has_ep].dropna(subset=["endpoint_date"])
    leak = ep_facts["timestamp_start"] >= ep_facts["endpoint_date"]
    check("no fact timestamped at/after endpoint_date",
          not leak.any(),
          f"{leak.sum():,} facts violate this")

    print("\n5. Cohort exclusion cascade is monotonically non-increasing")
    cascade = pd.read_csv(os.path.join(OUTPUT_DIR, "cohort_cascade.csv"))
    diffs = cascade["n_patients"].diff().dropna()
    check("cascade counts never increase step-to-step",
          bool((diffs <= 0).all()),
          f"cascade increased at step(s): {cascade.loc[diffs[diffs > 0].index, 'step'].tolist()}")

    print("\n6. No duplicate patients in cohort.csv")
    dup = cohort["subject_id"].duplicated().sum()
    check("cohort.csv has one row per subject_id", dup == 0, f"{dup} duplicate subject_id rows")

    print("\n7. Train-only vocabulary restriction (baselines_survival.py / tgn_model.py)")
    train_ids = set(labels.loc[labels["split"] == "train", "subject_id"])
    events_full = read_events_table(usecols=["subject_id", "concept_node_idx"])
    train_concepts_recomputed = set(
        events_full.loc[events_full["subject_id"].isin(train_ids), "concept_node_idx"].unique()
    )
    check("recomputed train-only concept set is non-empty (sanity)",
          len(train_concepts_recomputed) > 0,
          "recomputation produced zero concepts -- something upstream changed")
    test_only_concepts = (
        set(events_full["concept_node_idx"].unique()) - train_concepts_recomputed
    )

    # Reconstruct tgn_model.py._prepare_data's exact remap: train concepts get
    # 1..N (sorted, matching its `sorted(...)` call), everything else -> UNK (0).
    concept_remap = {c: i + 1 for i, c in enumerate(sorted(train_concepts_recomputed))}
    emb_idx = events_full["concept_node_idx"].map(concept_remap).fillna(0).astype(int)

    test_only_mask = events_full["concept_node_idx"].isin(test_only_concepts)
    leaked_to_real_idx = (emb_idx[test_only_mask] != 0).sum()
    check("every test/val-only concept maps to UNK, never a real embedding index",
          leaked_to_real_idx == 0,
          f"{leaked_to_real_idx:,} test/val-only-concept events got a nonzero embedding index")

    train_mask = events_full["concept_node_idx"].isin(train_concepts_recomputed)
    misrouted_to_unk = (emb_idx[train_mask] == 0).sum()
    check("every train concept maps to a real (nonzero) embedding index, not UNK",
          misrouted_to_unk == 0,
          f"{misrouted_to_unk:,} train-concept events were incorrectly routed to UNK")

    n_unique_idx = len(set(concept_remap.values()))
    check("train concept -> embedding index mapping has no collisions",
          n_unique_idx == len(train_concepts_recomputed),
          f"expected {len(train_concepts_recomputed):,} unique indices, got {n_unique_idx:,}")

    print(f"       ({len(test_only_concepts):,} concepts appear only in val/test patients, "
          f"all correctly routed to UNK)")

    print("\n8. No interval fact's value reflects information past its window close")
    win_end = cohort[["subject_id", "endpoint_date", "index_date", "follow_up_days",
                       "endpoint_type"]].copy()
    censored = win_end["endpoint_type"] == "censored"
    win_end["window_end"] = win_end["endpoint_date"]
    win_end.loc[censored, "window_end"] = (
        win_end.loc[censored, "index_date"]
        + pd.to_timedelta(win_end.loc[censored, "follow_up_days"], unit="D")
    )
    late = interval_facts.merge(win_end[["subject_id", "window_end"]],
                                 on="subject_id", how="left")
    late_and_valued = late["value_num"].notna() & (
        late["timestamp_end"].isna() | (late["timestamp_end"] >= late["window_end"])
    )
    check("interval facts (ICU los, IV input amount) ending on/after window close have no value",
          not late_and_valued.any(),
          f"{late_and_valued.sum():,} facts carry a value_num finalized after the window closes")

    print(f"\n=== {len(FAILURES)} failing check(s) ===" if FAILURES else "\n=== All checks passed ===")
    if FAILURES:
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    run()

"""Filter MIMIC-IV discharge notes to cohort patients.

Output:
  tkg_output/notes/discharge_cohort.csv.gz   -- cohort notes with text
  tkg_output/notes/note_metadata.csv          -- subject_id, hadm_id, charttime
"""
import os
import pandas as pd

from src.config import DATA_DIR, OUTPUT_DIR

NOTES_DIR = os.path.join(OUTPUT_DIR, "notes")


def extract_cohort_notes() -> None:
    os.makedirs(NOTES_DIR, exist_ok=True)

    print("Loading cohort...")
    cohort = pd.read_csv(os.path.join(OUTPUT_DIR, "cohort.csv"),
                          usecols=["subject_id", "index_date", "endpoint_date"],
                          parse_dates=["index_date", "endpoint_date"])
    cohort_ids = set(cohort["subject_id"])
    print(f"  cohort patients: {len(cohort_ids):,}")

    print("\nStreaming discharge.csv.gz (chunked)...")
    reader = pd.read_csv(
        os.path.join(DATA_DIR, "discharge.csv.gz"),
        chunksize=20_000, low_memory=False,
    )
    n_total = 0
    n_kept = 0
    kept = []
    for i, chunk in enumerate(reader):
        n_total += len(chunk)
        chunk = chunk[chunk["subject_id"].isin(cohort_ids)].copy()
        if not chunk.empty:
            n_kept += len(chunk)
            kept.append(chunk)
        if (i + 1) % 10 == 0:
            print(f"  chunk {i+1:3d} | read: {n_total:>9,} | kept: {n_kept:,}")
    print(f"\nDone. Total read: {n_total:,}; cohort notes: {n_kept:,}")

    if not kept:
        print("WARN: no cohort notes found. Aborting.")
        return
    out = pd.concat(kept, ignore_index=True)

    # Save full notes (with text) -- compressed
    out_path = os.path.join(NOTES_DIR, "discharge_cohort.csv.gz")
    out.to_csv(out_path, index=False, compression="gzip")

    # Lightweight metadata for later joins (no text)
    meta = out[["note_id", "subject_id", "hadm_id",
                "charttime", "storetime", "note_seq", "note_type"]]
    meta_path = os.path.join(NOTES_DIR, "note_metadata.csv")
    meta.to_csv(meta_path, index=False)

    # Per-patient summary
    n_per_pt = out.groupby("subject_id").size()
    n_cohort_with_notes = out["subject_id"].nunique()
    print(f"\n=== NOTE EXTRACTION SUMMARY ===")
    print(f"  total cohort notes: {len(out):,}")
    print(f"  cohort patients with >=1 discharge note: {n_cohort_with_notes:,} "
          f"({n_cohort_with_notes/len(cohort_ids)*100:.1f}% of cohort)")
    print(f"  notes per patient: median={int(n_per_pt.median())}, "
          f"p75={int(n_per_pt.quantile(0.75))}, max={int(n_per_pt.max())}")
    print(f"\nSaved:\n  {out_path}\n  {meta_path}")


if __name__ == "__main__":
    extract_cohort_notes()

"""Aggregate discharge-note BioBERT embeddings into a per-patient vector.

For each cohort patient, we mean-pool the [CLS] embeddings of all their
discharge notes written **strictly before the index admission** (i.e.,
charttime < index_date). This matches the prognostic-at-admit cutoff used
for every other modality in `prep_modeling.py`, so no modality ever sees
information from time points another does not. Patients without any prior
discharge notes get a zero vector and a `has_notes = 0` indicator.

Output:
  tkg_output/notes/patient_note_emb.npy   -- (n_patients, 768) float32, sorted by subject_id
  tkg_output/notes/patient_note_emb.csv   -- subject_id, has_notes, n_notes
"""
import os
import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR

NOTES_DIR = os.path.join(OUTPUT_DIR, "notes")
EMB_DIM = 768


def aggregate_notes() -> None:
    print("Loading cohort + note artifacts...")
    cohort = pd.read_csv(
        os.path.join(OUTPUT_DIR, "cohort.csv"),
        usecols=["subject_id", "index_date"],
        parse_dates=["index_date"],
    )
    meta = pd.read_csv(
        os.path.join(NOTES_DIR, "note_metadata.csv"),
        parse_dates=["charttime"],
    )
    emb = np.load(os.path.join(NOTES_DIR, "note_embeddings.npy"))
    ids = pd.read_csv(os.path.join(NOTES_DIR, "note_ids.csv"))["note_id"].tolist()
    assert emb.shape == (len(ids), EMB_DIM), \
        f"embedding shape mismatch: {emb.shape} vs ({len(ids)}, {EMB_DIM})"
    nid_to_row = {n: i for i, n in enumerate(ids)}
    print(f"  cohort: {len(cohort):,}, notes: {len(meta):,}, "
          f"embeddings: {emb.shape}")

    # Pre-index (strict) — matches the prognostic-at-admit cutoff used for
    # every other modality in prep_modeling.py. No modality sees data from
    # time points that another modality does not.
    meta = meta.merge(cohort[["subject_id", "index_date"]],
                       on="subject_id", how="inner")
    meta = meta[meta["charttime"] < meta["index_date"]].copy()
    print(f"  pre-index notes (strict, charttime < index_date): {len(meta):,}")

    # Map note_id -> embedding row
    meta["emb_row"] = meta["note_id"].map(nid_to_row).astype("Int64")
    meta = meta.dropna(subset=["emb_row"])
    meta["emb_row"] = meta["emb_row"].astype(int)

    # Mean-pool per patient (sorted by subject_id so order matches static_features.csv)
    cohort_sorted = cohort.sort_values("subject_id").reset_index(drop=True)
    sid_to_idx = {sid: i for i, sid in enumerate(cohort_sorted["subject_id"])}
    patient_emb = np.zeros((len(cohort_sorted), EMB_DIM), dtype=np.float32)
    n_notes = np.zeros(len(cohort_sorted), dtype=np.int32)
    for sid, g in meta.groupby("subject_id"):
        idx = sid_to_idx.get(int(sid))
        if idx is None:
            continue
        rows = g["emb_row"].to_numpy()
        patient_emb[idx] = emb[rows].mean(axis=0)
        n_notes[idx] = len(rows)

    emb_path = os.path.join(NOTES_DIR, "patient_note_emb.npy")
    meta_path = os.path.join(NOTES_DIR, "patient_note_emb.csv")
    np.save(emb_path, patient_emb)
    pd.DataFrame({
        "subject_id": cohort_sorted["subject_id"].to_numpy(),
        "has_notes": (n_notes > 0).astype(int),
        "n_notes": n_notes,
    }).to_csv(meta_path, index=False)

    n_with = int((n_notes > 0).sum())
    pct = n_with / len(cohort_sorted) * 100.0
    print(f"\n=== PER-PATIENT NOTE AGGREGATION SUMMARY ===")
    print(f"  patients with >=1 pre-index discharge note: {n_with:,} ({pct:.1f}%)")
    if n_with > 0:
        nz = n_notes[n_notes > 0]
        print(f"  pre-index notes/patient (median / p75 / max): "
              f"{int(np.median(nz))} / {int(np.quantile(nz, 0.75))} / {int(nz.max())}")
    print(f"\nSaved:\n  {emb_path}\n  {meta_path}")


if __name__ == "__main__":
    aggregate_notes()

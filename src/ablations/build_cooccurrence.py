"""Concept-concept co-occurrence edges, computed from TRAIN patients only.

build_ontology.py's edges are a naming hierarchy (a code belongs to its ICD
category/chapter, a drug belongs to its class) -- real relational structure,
but only 9.3% of drug concepts resolve to a class, since the hand-curated
dictionary only covers cardiometabolic drugs. Co-occurrence edges don't need
a hand-curated dictionary at all: two concepts get an edge if they show up
together in enough patients' pre-index history, which works uniformly across
every modality (drugs included) and reflects actual clinical usage patterns
rather than a naming convention.

Train-only by construction: co-occurrence counts are computed exclusively
from training-split patients, so no val/test patient's data shapes the graph
structure itself (only their own node features flow through it at
inference time) -- the same train-only discipline already used for the
concept vocabulary restriction elsewhere in this pipeline.

MIN_COOCCUR controls graph density: a pair must co-occur in at least this
many training patients to get an edge. This is a real, disclosed choice --
too low and the graph explodes in size relative to the ~23.5k training
patients available to learn from it (see the "graph too big for the data"
finding); too high and sparse-but-informative relationships get dropped.
"""
import os
import numpy as np
import pandas as pd
from scipy import sparse

from src.config import OUTPUT_DIR, read_events_table

MIN_COOCCUR = 30       # a concept pair must co-occur in >= this many train patients
MAX_EDGES_PER_CONCEPT = 15   # cap the top-K strongest co-occurrence edges per concept


def build_cooccurrence() -> None:
    print("Loading modeling labels + events...")
    labels = pd.read_csv(os.path.join(OUTPUT_DIR, "modeling", "labels.csv"))
    train_ids = set(labels.loc[labels["split"] == "train", "subject_id"])
    events = read_events_table(usecols=["subject_id", "concept_node_idx"])
    train_events = events[events["subject_id"].isin(train_ids)]

    pairs = train_events.drop_duplicates(["subject_id", "concept_node_idx"])
    print(f"  train patients: {len(train_ids):,}, "
          f"distinct (patient, concept) pairs: {len(pairs):,}")

    patients = pd.Categorical(pairs["subject_id"])
    concepts = pd.Categorical(pairs["concept_node_idx"])
    n_pat, n_con = len(patients.categories), len(concepts.categories)
    M = sparse.csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (patients.codes, concepts.codes)),
        shape=(n_pat, n_con),
    )
    print(f"  incidence matrix: {n_pat:,} patients x {n_con:,} concepts, "
          f"density {M.nnz / (n_pat * n_con) * 100:.3f}%")

    print("Computing concept-concept co-occurrence (sparse matmul)...")
    C = (M.T @ M).tocoo()   # concept x concept co-occurrence counts
    concept_ids = concepts.categories.to_numpy()

    mask = (C.row != C.col) & (C.data >= MIN_COOCCUR)
    rows, cols, counts = C.row[mask], C.col[mask], C.data[mask]
    print(f"  pairs with co-occurrence >= {MIN_COOCCUR}: {len(rows):,} "
          f"(of {C.nnz:,} raw nonzero entries)")

    df = pd.DataFrame({
        "src_concept_node_idx": concept_ids[rows],
        "dst_concept_node_idx": concept_ids[cols],
        "count": counts.astype(int),
    })
    # Keep only the strongest MAX_EDGES_PER_CONCEPT edges per source concept,
    # so a handful of extremely common concepts (e.g. a routine saline flush)
    # don't dominate every node's neighborhood.
    df = (df.sort_values(["src_concept_node_idx", "count"], ascending=[True, False])
            .groupby("src_concept_node_idx").head(MAX_EDGES_PER_CONCEPT))
    print(f"  after top-{MAX_EDGES_PER_CONCEPT}-per-concept cap: {len(df):,} directed edges")

    out_path = os.path.join(OUTPUT_DIR, "cooccurrence_edges.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  concepts with >=1 co-occurrence edge: {df['src_concept_node_idx'].nunique():,} "
          f"of {n_con:,} train concepts "
          f"({df['src_concept_node_idx'].nunique() / n_con * 100:.1f}%)")


if __name__ == "__main__":
    build_cooccurrence()

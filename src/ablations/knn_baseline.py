"""Item 5: does the patient-graph model do more than a simple, hand-built
k-nearest-neighbor case-based-reasoning baseline?

If a plain k-NN on concept overlap (no learning, no message-passing, no
embeddings) matches the patient-graph model's performance, the graph is not
earning its complexity. This is the single most direct test of that
question: same train-only concept vocabulary and same corrected
competing-risks evaluation as every other model in this study, but the
"model" is just "average what similar training patients experienced."

Method: for each test patient, find the k most similar training patients
by cosine similarity on the train-only bag-of-codes vector (identical
feature space the tabular baselines use). Risk score for (cause, horizon) =
fraction of the k nearest training neighbors who had that cause by that
horizon. k is chosen on the validation set only, from a small candidate
list, exactly as any other hyperparameter in this study would be.

Output: tkg_output/knn_baseline/test_metrics.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import sparse

from src.config import OUTPUT_DIR, read_events_table
from src.baseline import _build_bag_of_codes
from src.tgn_survival import CAUSES, HORIZON_DAYS, _per_cause_auroc_at_horizons, _make_time_bins, NUM_TIME_BINS

MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
KNN_DIR = os.path.join(OUTPUT_DIR, "knn_baseline")
K_CANDIDATES = [10, 25, 50, 100, 200]


def _risk_scores(sim_matrix: np.ndarray, train_labels: pd.DataFrame, k: int) -> dict:
    """sim_matrix: (n_query, n_train). Returns {(cause, horizon): np.array of scores}."""
    train_dur = train_labels["time_to_event_days"].to_numpy(dtype=float)
    train_evt = train_labels["endpoint_type"].to_numpy()
    nn_idx = np.argsort(-sim_matrix, axis=1)[:, :k]   # top-k most similar train patients

    scores = {}
    for cause in CAUSES:
        is_cause = (train_evt == cause)
        for h in HORIZON_DAYS:
            had_cause_by_h = is_cause & (train_dur <= h)
            frac = had_cause_by_h[nn_idx].mean(axis=1)
            scores[(cause, h)] = frac
    return scores


def run() -> None:
    os.makedirs(KNN_DIR, exist_ok=True)
    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    events = read_events_table()

    train_ids = labels.loc[labels["split"] == "train", "subject_id"].tolist()
    val_ids = labels.loc[labels["split"] == "val", "subject_id"].tolist()
    test_ids = labels.loc[labels["split"] == "test", "subject_id"].tolist()
    train_set = set(train_ids)

    train_concepts = sorted(
        events.loc[events["subject_id"].isin(train_set), "concept_node_idx"].unique().tolist()
    )
    print(f"  train-only concept vocabulary: {len(train_concepts):,}")

    all_ids = train_ids + val_ids + test_ids
    X = _build_bag_of_codes(events, all_ids, train_concepts)   # sparse counts -> binarize for Jaccard-like overlap
    X_bin = (X > 0).astype(np.float32)
    # Cosine similarity on binary concept-presence vectors (a simple, defensible
    # case-based-reasoning notion of "similar patient": shares similar concepts).
    norms = np.sqrt(X_bin.multiply(X_bin).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    X_norm = X_bin.multiply(1.0 / norms[:, None]).tocsr()

    n_train = len(train_ids)
    train_block = X_norm[:n_train]
    val_block = X_norm[n_train:n_train + len(val_ids)]
    test_block = X_norm[n_train + len(val_ids):]

    train_labels_df = labels.set_index("subject_id").loc[train_ids].reset_index()
    val_labels_df = labels.set_index("subject_id").loc[val_ids].reset_index()
    test_labels_df = labels.set_index("subject_id").loc[test_ids].reset_index()

    print("Selecting k on validation set (mean per-cause AUROC @ 3y)...")
    val_sim = (val_block @ train_block.T).toarray()
    best_k, best_metric = None, -1.0
    time_edges = _make_time_bins(
        train_labels_df["time_to_event_days"].to_numpy(dtype=np.float32), NUM_TIME_BINS)
    for k in K_CANDIDATES:
        scores = _risk_scores(val_sim, train_labels_df, k)
        # Build a (N, C, T)-shaped CIF-like array is unnecessary here; reuse the
        # per-horizon evaluator directly at the 3y horizon for model selection.
        aucs = []
        for cause in CAUSES:
            durs = val_labels_df["time_to_event_days"].to_numpy(dtype=float)
            evts = val_labels_df["endpoint_type"].to_numpy()
            h = 1095
            y_pos = (evts == cause) & (durs <= h)
            y_neg = (durs >= h) | ((evts != cause) & (evts != "censored") & (durs < h))
            mask = y_pos | y_neg
            y = y_pos[mask].astype(int)
            s = scores[(cause, h)][mask]
            if y.sum() == 0 or y.sum() == len(y):
                continue
            from sklearn.metrics import roc_auc_score
            aucs.append(roc_auc_score(y, s))
        mean_auc = float(np.mean(aucs)) if aucs else -1.0
        print(f"  k={k:4d}  val mean AUROC@3y={mean_auc:.4f}")
        if mean_auc > best_metric:
            best_metric, best_k = mean_auc, k
    print(f"  selected k={best_k} (val mean AUROC@3y={best_metric:.4f})")

    print("\nEvaluating on test set with selected k...")
    test_sim = (test_block @ train_block.T).toarray()
    test_scores = _risk_scores(test_sim, train_labels_df, best_k)

    rows = []
    durs = test_labels_df["time_to_event_days"].to_numpy(dtype=float)
    evts = test_labels_df["endpoint_type"].to_numpy()
    from sklearn.metrics import roc_auc_score, average_precision_score
    for cause in CAUSES:
        for h in HORIZON_DAYS:
            y_pos = (evts == cause) & (durs <= h)
            y_neg = (durs >= h) | ((evts != cause) & (evts != "censored") & (durs < h))
            mask = y_pos | y_neg
            y = y_pos[mask].astype(int)
            s = test_scores[(cause, h)][mask]
            if y.sum() == 0 or y.sum() == len(y):
                rows.append(dict(cause=cause, horizon_days=h, auroc=np.nan, auprc=np.nan,
                                  n_pos=int(y.sum()), n=len(y)))
                continue
            rows.append(dict(cause=cause, horizon_days=h,
                              auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s),
                              n_pos=int(y.sum()), n=len(y)))
    result = pd.DataFrame(rows)
    print("\n=== k-NN BASELINE TEST METRICS (k={}) ===".format(best_k))
    print(result.pivot(index="cause", columns="horizon_days", values="auroc").round(3).to_string())

    out_path = os.path.join(KNN_DIR, "test_metrics.csv")
    result.to_csv(out_path, index=False)
    with open(os.path.join(KNN_DIR, "selected_k.txt"), "w") as f:
        f.write(str(best_k))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()

"""Multiclass baselines (LogReg + XGBoost on bag-of-codes + value summaries).

The main pipeline uses the survival framing in `baselines_survival.py` and
`tgn_survival.py`. This module is kept because:

  - it provides the feature builders `_build_bag_of_codes` and
    `_build_value_summary_features`, both imported by `baselines_survival.py`;
  - `run_baseline()` produces the multiclass-softmax sensitivity analysis.

Concept space, normalization, and class weighting are all training-only;
test predictions are produced once at the end.
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, log_loss,
)
import matplotlib.pyplot as plt
import xgboost as xgb

from src.config import OUTPUT_DIR, FIGURES_DIR, SEED, read_events_table

warnings.filterwarnings("ignore", category=UserWarning)

MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
BASELINE_DIR = os.path.join(OUTPUT_DIR, "baseline")
ENDPOINT_ORDER = ["MI", "Stroke", "HF", "AF", "PAD", "censored"]
EP_TO_IDX = {ep: i for i, ep in enumerate(ENDPOINT_ORDER)}


def _load_data():
    print("Loading modeling artifacts...")
    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    static = pd.read_csv(os.path.join(MODELING_DIR, "static_features.csv"))
    events = read_events_table()
    nodes = pd.read_csv(os.path.join(MODELING_DIR, "node_metadata.csv"))
    print(f"  labels={len(labels):,}, static={len(static):,}, "
          f"events={len(events):,}, nodes={len(nodes):,}")
    return labels, static, events, nodes


def _build_bag_of_codes(events: pd.DataFrame, patient_order: list[int],
                        concept_ids: list[int]) -> csr_matrix:
    """Sparse count matrix (n_patients x n_concepts)."""
    pid_to_row = {pid: i for i, pid in enumerate(patient_order)}
    cid_to_col = {cid: j for j, cid in enumerate(concept_ids)}
    counts = (events.groupby(["subject_id", "concept_node_idx"]).size()
              .reset_index(name="n"))
    counts = counts[counts["subject_id"].isin(pid_to_row)
                    & counts["concept_node_idx"].isin(cid_to_col)]
    rows = counts["subject_id"].map(pid_to_row).to_numpy()
    cols = counts["concept_node_idx"].map(cid_to_col).to_numpy()
    data = counts["n"].to_numpy(dtype=np.float32)
    X = csr_matrix((data, (rows, cols)),
                    shape=(len(patient_order), len(concept_ids)))
    return X


def _build_value_summary_features(
    events: pd.DataFrame, patient_order: list[int], nodes: pd.DataFrame,
    train_ids: set,
) -> tuple[np.ndarray, list[str]]:
    """Per-patient summary stats for every LAB_*, OMR_BP*, OMR_BMI* concept.

    Returns a dense float32 array of shape (n_patients, n_concepts_with_values * 6)
    (mean, max, min, last, count, slope per concept) and the column names.
    Missing -> 0 (sentinel; XGBoost handles natively).
    """
    if "value_num" not in events.columns:
        return np.zeros((len(patient_order), 0), dtype=np.float32), []
    val_events = events[events["value_num"].notna()].copy()
    # Restrict to concepts present in training set (avoid leakage of test-only concepts)
    train_concepts = set(val_events.loc[
        val_events["subject_id"].isin(train_ids), "concept_node_idx"].unique())
    val_events = val_events[val_events["concept_node_idx"].isin(train_concepts)]
    if val_events.empty:
        return np.zeros((len(patient_order), 0), dtype=np.float32), []

    val_events = val_events.sort_values(["subject_id", "relative_days"])
    pid_to_row = {pid: i for i, pid in enumerate(patient_order)}
    nodes_by_idx = nodes.set_index("node_idx")

    feat_concepts = sorted(train_concepts)
    cid_to_col = {c: j for j, c in enumerate(feat_concepts)}
    n_stats = 6  # mean, max, min, last, count, slope
    n_pts = len(patient_order)
    arr = np.zeros((n_pts, len(feat_concepts) * n_stats), dtype=np.float32)

    grouped = val_events.groupby(["subject_id", "concept_node_idx"])["value_num"]
    aggs = grouped.agg(["mean", "max", "min", "last", "count"]).reset_index()
    aggs = aggs[aggs["subject_id"].isin(pid_to_row)]

    def _slope(g: pd.DataFrame) -> float:
        """OLS slope of value vs relative_days; 0 if <2 unique time points."""
        if len(g) < 2:
            return 0.0
        x = g["relative_days"].to_numpy(dtype=np.float64)
        y = g["value_num"].to_numpy(dtype=np.float64)
        if np.allclose(x.std(), 0.0):
            return 0.0
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        if denom == 0.0:
            return 0.0
        return float(((x - xm) * (y - ym)).sum() / denom)

    slopes = (val_events.groupby(["subject_id", "concept_node_idx"])
              [["relative_days", "value_num"]]
              .apply(_slope).reset_index(name="slope"))
    slopes = slopes[slopes["subject_id"].isin(pid_to_row)]
    slope_map = {(r.subject_id, r.concept_node_idx): r.slope
                 for r in slopes.itertuples(index=False)}

    for r in aggs.itertuples(index=False):
        i = pid_to_row[r.subject_id]
        j = cid_to_col[r.concept_node_idx]
        base = j * n_stats
        arr[i, base + 0] = r.mean
        arr[i, base + 1] = r.max
        arr[i, base + 2] = r.min
        arr[i, base + 3] = r.last
        arr[i, base + 4] = r.count
        arr[i, base + 5] = slope_map.get((r.subject_id, r.concept_node_idx), 0.0)

    train_mask = np.array([pid in train_ids for pid in patient_order])
    mu = arr[train_mask].mean(axis=0)
    sd = arr[train_mask].std(axis=0)
    sd[sd == 0] = 1.0
    arr = (arr - mu) / sd
    arr = np.clip(arr, -10.0, 10.0).astype(np.float32)

    stat_names = ["mean", "max", "min", "last", "count", "slope"]
    col_names: list[str] = []
    for c in feat_concepts:
        try:
            name = str(nodes_by_idx.loc[c, "concept_id"])
        except KeyError:
            name = f"node{c}"
        for s in stat_names:
            col_names.append(f"VAL_{name}_{s}")
    return arr, col_names


def _per_endpoint_metrics(y_true_idx: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """OvR AUROC and AUPRC per endpoint."""
    rows = []
    for ep, j in EP_TO_IDX.items():
        y_bin = (y_true_idx == j).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            rows.append({"endpoint": ep, "auroc": np.nan,
                         "auprc": np.nan, "support": int(y_bin.sum())})
            continue
        try:
            auroc = roc_auc_score(y_bin, proba[:, j])
        except ValueError:
            auroc = np.nan
        try:
            auprc = average_precision_score(y_bin, proba[:, j])
        except ValueError:
            auprc = np.nan
        rows.append({"endpoint": ep, "auroc": auroc, "auprc": auprc,
                     "support": int(y_bin.sum())})
    return pd.DataFrame(rows)


def _overall_metrics(y_true_idx: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true_idx, pred),
        "macro_f1": f1_score(y_true_idx, pred, average="macro"),
        "weighted_f1": f1_score(y_true_idx, pred, average="weighted"),
        "log_loss": log_loss(y_true_idx, proba, labels=list(range(len(ENDPOINT_ORDER)))),
    }


def _plot_roc(y_true_idx: np.ndarray, proba_by_model: dict[str, np.ndarray],
              out_path: str) -> None:
    from sklearn.metrics import roc_curve
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    for k, ep in enumerate(ENDPOINT_ORDER):
        ax = axes[k]
        j = EP_TO_IDX[ep]
        y_bin = (y_true_idx == j).astype(int)
        if y_bin.sum() == 0:
            ax.set_title(f"{ep} (no positives in test)")
            continue
        for name, proba in proba_by_model.items():
            fpr, tpr, _ = roc_curve(y_bin, proba[:, j])
            try:
                auc = roc_auc_score(y_bin, proba[:, j])
            except ValueError:
                auc = float("nan")
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=1.5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, alpha=0.5)
        ax.set_title(f"{ep} (n={int(y_bin.sum())})", fontweight="bold")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("Baseline ROC curves (test set, one-vs-rest)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_baseline() -> None:
    os.makedirs(BASELINE_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    labels, static, events, nodes = _load_data()

    labels = labels.sort_values("subject_id").reset_index(drop=True)
    static = static.sort_values("subject_id").reset_index(drop=True)
    assert (labels["subject_id"].to_numpy() == static["subject_id"].to_numpy()).all()
    patient_order = labels["subject_id"].tolist()

    y = labels["endpoint_type"].map(EP_TO_IDX).to_numpy()
    split = labels["split"].to_numpy()
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    print("\nBuilding bag-of-codes from pre-index events (train concepts only)...")
    train_ids_set = set(np.array(patient_order)[train_mask].tolist())
    train_events_mask = events["subject_id"].isin(train_ids_set)
    concept_ids = sorted(
        events.loc[train_events_mask, "concept_node_idx"].unique().tolist()
    )
    print(f"  concept feature space: {len(concept_ids):,} dims "
          f"(restricted from {events['concept_node_idx'].nunique():,} total)")
    X_bow = _build_bag_of_codes(events, patient_order, concept_ids)
    print(f"  bag-of-codes: shape={X_bow.shape}, "
          f"nnz={X_bow.nnz:,} ({X_bow.nnz / (X_bow.shape[0]*X_bow.shape[1]):.4%})")

    print("\nBuilding per-concept value summary features "
          "(mean/max/min/last/count/slope)...")
    X_values, value_col_names = _build_value_summary_features(
        events, patient_order, nodes, train_ids_set)
    print(f"  value summary features: {X_values.shape}")

    static_cols = ["age_at_index", "cci_score", "num_cardiometa_conditions",
                   "had_icu_stay", "female"]
    X_static = static[static_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(X_static[train_mask])
    X_static_scaled = scaler.transform(X_static)

    X = hstack([
        X_bow,
        sparse.csr_matrix(X_values),
        sparse.csr_matrix(X_static_scaled),
    ]).tocsr()
    print(f"  full design matrix: {X.shape}, nnz={X.nnz:,}")

    Xtr, ytr = X[train_mask], y[train_mask]
    Xva, yva = X[val_mask],   y[val_mask]
    Xte, yte = X[test_mask],  y[test_mask]

    print(f"\nTrain: {Xtr.shape[0]:,}  Val: {Xva.shape[0]:,}  Test: {Xte.shape[0]:,}")

    print("\n[1/2] Fitting LogisticRegression (multinomial, class-balanced)...")
    lr = LogisticRegression(
        solver="saga", penalty="l2", C=1.0, max_iter=200,
        class_weight="balanced", multi_class="multinomial",
        n_jobs=-1, random_state=SEED, tol=1e-3,
    )
    lr.fit(Xtr, ytr)
    proba_lr_va = lr.predict_proba(Xva)
    proba_lr_te = lr.predict_proba(Xte)
    print("  done.")

    print("\n[2/2] Fitting XGBoost (multi:softprob)...")
    n_classes = len(ENDPOINT_ORDER)
    counts = np.bincount(ytr, minlength=n_classes).astype(float)
    inv = (1.0 / counts) * counts.sum() / n_classes
    sample_weight = inv[ytr]
    xgb_clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.7,
        objective="multi:softprob", num_class=n_classes,
        tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="mlogloss",
        early_stopping_rounds=20,
    )
    xgb_clf.fit(Xtr, ytr, sample_weight=sample_weight,
                eval_set=[(Xva, yva)], verbose=False)
    proba_xgb_va = xgb_clf.predict_proba(Xva)
    proba_xgb_te = xgb_clf.predict_proba(Xte)
    print(f"  best iteration: {xgb_clf.best_iteration}")

    print("\n=== METRICS ===")
    rows = []
    for split_name, (y_eval, proba_lr, proba_xgb) in [
        ("val",  (yva, proba_lr_va, proba_xgb_va)),
        ("test", (yte, proba_lr_te, proba_xgb_te)),
    ]:
        for model_name, proba in [("logreg", proba_lr), ("xgboost", proba_xgb)]:
            overall = _overall_metrics(y_eval, proba)
            per_ep = _per_endpoint_metrics(y_eval, proba)
            for ep_row in per_ep.itertuples(index=False):
                rows.append({
                    "model": model_name, "split": split_name,
                    "endpoint": ep_row.endpoint,
                    "auroc": ep_row.auroc, "auprc": ep_row.auprc,
                    "support": ep_row.support,
                    **{k: overall[k] for k in
                       ("accuracy", "macro_f1", "weighted_f1", "log_loss")},
                })
    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(BASELINE_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("\nPer-endpoint AUROC / AUPRC on test:")
    for model_name in ("logreg", "xgboost"):
        sub = metrics_df[(metrics_df["model"] == model_name)
                          & (metrics_df["split"] == "test")]
        print(f"\n  {model_name}:")
        print("    {:<10s} {:>8s} {:>8s} {:>8s}".format(
            "endpoint", "AUROC", "AUPRC", "n_pos"))
        for r in sub.itertuples(index=False):
            print(f"    {r.endpoint:<10s} {r.auroc:>8.3f} {r.auprc:>8.3f} "
                  f"{r.support:>8d}")
        ov = sub.iloc[0]
        print(f"    overall: acc={ov['accuracy']:.3f}  "
              f"macroF1={ov['macro_f1']:.3f}  "
              f"weightedF1={ov['weighted_f1']:.3f}  "
              f"logloss={ov['log_loss']:.3f}")

    print("\nSaving test predictions...")
    preds = pd.DataFrame({
        "subject_id": labels.loc[test_mask, "subject_id"].to_numpy(),
        "endpoint_true": labels.loc[test_mask, "endpoint_type"].to_numpy(),
    })
    for j, ep in enumerate(ENDPOINT_ORDER):
        preds[f"logreg_p_{ep}"] = proba_lr_te[:, j]
        preds[f"xgb_p_{ep}"]    = proba_xgb_te[:, j]
    preds_path = os.path.join(BASELINE_DIR, "predictions_test.csv")
    preds.to_csv(preds_path, index=False)

    print("Saving XGBoost feature importance (top 50)...")
    booster = xgb_clf.get_booster()
    importance = booster.get_score(importance_type="gain")
    # Column layout: bag-of-codes | value summaries | static
    n_concepts = len(concept_ids)
    n_value = len(value_col_names)
    rows = []
    nodes_by_idx = nodes.set_index("node_idx")
    for fkey, val in importance.items():
        fidx = int(fkey.lstrip("f"))
        if fidx < n_concepts:
            nidx = concept_ids[fidx]
            row = nodes_by_idx.loc[nidx]
            rows.append({"feature": row["concept_id"],
                         "fact_type": row["fact_type"],
                         "kind": "concept", "gain": val})
        elif fidx < n_concepts + n_value:
            rows.append({"feature": value_col_names[fidx - n_concepts],
                         "fact_type": "value_summary",
                         "kind": "value", "gain": val})
        else:
            rows.append({"feature": static_cols[fidx - n_concepts - n_value],
                         "fact_type": "static",
                         "kind": "static", "gain": val})
    fi = pd.DataFrame(rows).sort_values("gain", ascending=False).head(50)
    fi_path = os.path.join(BASELINE_DIR, "xgb_feature_importance_top50.csv")
    fi.to_csv(fi_path, index=False)

    roc_path = os.path.join(FIGURES_DIR, "fig5_baseline_roc.png")
    _plot_roc(yte, {"logreg": proba_lr_te, "xgboost": proba_xgb_te}, roc_path)
    print(f"\nSaved:")
    print(f"  {metrics_path}")
    print(f"  {preds_path}")
    print(f"  {fi_path}")
    print(f"  {roc_path}")


if __name__ == "__main__":
    run_baseline()

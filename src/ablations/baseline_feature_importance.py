"""Per-disease feature importance for Cox and XGBoost -- the two models
that didn't have this anywhere in the project. The TKG-Transformer
(concept_importance_by_cause.csv, attention-based) and the patient-graph
model (gnn_concept_importance_by_cause.csv, GNNExplainer-based) already
have per-disease importance rankings; this closes the gap for the two
flat-feature baselines, using the exact tuned settings baselines_survival.py
fits with (same alphas, same XGBoost config) so these rankings describe the
same models reported everywhere else in this project.

Cox: signed coefficient magnitude (|coef|) -- the elastic-net path already
zeros out most features (see cox_regularization_sweep.py), so the non-zero
survivors are already a short, meaningful list per disease.
XGBoost: gain-based importance (average loss reduction per split on that
feature), the standard choice over raw split-count.

Every feature index is mapped back to a readable name: a concept ID + fact
type (from node_metadata.csv) for the bag-of-codes block, a value-summary
column name (e.g. "LAB_CREATININE_mean") for the per-concept statistics
block, or a static demographic/comorbidity name for the last block --
exactly the same column-layout convention baseline.py's original feature-
importance code already used for the (now-superseded) multiclass model.

Output: tkg_output/explain/cox_feature_importance_by_cause.csv
        tkg_output/explain/xgb_feature_importance_by_cause.csv
"""
import os

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis
import xgboost as xgb

from src.config import OUTPUT_DIR
from src.baseline import _build_bag_of_codes, _build_value_summary_features
from src.baselines_survival import (
    _load, _build_X, _make_y, CAUSES, COX_ALPHA, COX_L1_RATIO, XGB_CONFIG,
)

EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explain")
TOP_N = 30


def run() -> None:
    os.makedirs(EXPLAIN_DIR, exist_ok=True)
    print("Loading data + building features (shared across all fits)...")
    labels, static, events, nodes = _load()

    # Reproduce _build_X's exact column layout so feature indices line up.
    labels_sorted = labels.sort_values("subject_id").reset_index(drop=True)
    patient_order = labels_sorted["subject_id"].tolist()
    train_ids_set = set(labels_sorted.loc[labels_sorted["split"] == "train", "subject_id"])
    train_events_mask = events["subject_id"].isin(train_ids_set)
    concept_ids = sorted(events.loc[train_events_mask, "concept_node_idx"].unique().tolist())
    _, value_col_names = _build_value_summary_features(events, patient_order, nodes, train_ids_set)
    static_cols = ["age_at_index", "cci_score", "num_cardiometa_conditions", "had_icu_stay", "female"]
    n_concepts, n_value = len(concept_ids), len(value_col_names)
    nodes_by_idx = nodes.set_index("node_idx")

    def _feature_name(fidx: int) -> tuple:
        """Returns (feature_label, fact_type)."""
        if fidx < n_concepts:
            nidx = concept_ids[fidx]
            row = nodes_by_idx.loc[nidx]
            return row["concept_id"], row["fact_type"]
        elif fidx < n_concepts + n_value:
            return value_col_names[fidx - n_concepts], "value_summary"
        else:
            return static_cols[fidx - n_concepts - n_value], "static"

    X, labels_b, tr_m, va_m, te_m = _build_X(labels, static, events, nodes)
    X = X.astype(np.float32)
    X_tr = X[tr_m]
    labels_tr = labels_b[tr_m].reset_index(drop=True)
    X_tr_dense = X_tr.toarray()

    cox_rows, xgb_rows = [], []
    for cause in CAUSES:
        print(f"\n=== {cause} ===")
        y_tr = _make_y(labels_tr, cause)

        print(f"  fitting Cox (alpha={COX_ALPHA[cause]})...")
        cox_path = CoxnetSurvivalAnalysis(l1_ratio=COX_L1_RATIO, n_alphas=100, max_iter=2000, tol=1e-7)
        cox_path.fit(X_tr_dense, y_tr)
        alphas = cox_path.alphas_
        nearest_idx = int(np.argmin(np.abs(alphas - COX_ALPHA[cause])))
        coef = cox_path.coef_[:, nearest_idx]
        nz = np.nonzero(coef)[0]
        order = nz[np.argsort(-np.abs(coef[nz]))][:TOP_N]
        for rank, fidx in enumerate(order, start=1):
            name, ftype = _feature_name(int(fidx))
            cox_rows.append(dict(cause=cause, rank=rank, feature=name, fact_type=ftype,
                                  coefficient=round(float(coef[fidx]), 6)))
        print(f"    {len(nz)} non-zero coefficients; top feature: {_feature_name(int(order[0]))[0] if len(order) else 'n/a'}")

        print("  fitting XGBoost...")
        event_tr = (labels_tr["endpoint_type"] == cause).to_numpy().astype(bool)
        dur_tr = np.maximum(labels_tr["time_to_event_days"].to_numpy(dtype=float), 1.0)
        y_xgb = np.where(event_tr, dur_tr, -dur_tr).astype(np.float32)
        clf = xgb.XGBRegressor(
            objective="survival:cox", eval_metric="cox-nloglik",
            tree_method="hist", n_jobs=-1, random_state=42, **XGB_CONFIG,
        )
        clf.fit(X_tr, y_xgb)
        booster = clf.get_booster()
        importance = booster.get_score(importance_type="gain")
        ranked = sorted(importance.items(), key=lambda kv: -kv[1])[:TOP_N]
        for rank, (fkey, gain) in enumerate(ranked, start=1):
            fidx = int(fkey.lstrip("f"))
            name, ftype = _feature_name(fidx)
            xgb_rows.append(dict(cause=cause, rank=rank, feature=name, fact_type=ftype,
                                  gain=round(float(gain), 4)))
        print(f"    top feature: {ranked[0][0] if ranked else 'n/a'} "
              f"({_feature_name(int(ranked[0][0].lstrip('f')))[0] if ranked else 'n/a'})")

    cox_out = pd.DataFrame(cox_rows)
    xgb_out = pd.DataFrame(xgb_rows)
    cox_path_out = os.path.join(EXPLAIN_DIR, "cox_feature_importance_by_cause.csv")
    xgb_path_out = os.path.join(EXPLAIN_DIR, "xgb_feature_importance_by_cause.csv")
    cox_out.to_csv(cox_path_out, index=False)
    xgb_out.to_csv(xgb_path_out, index=False)

    print("\n=== Top 5 Cox features per disease ===")
    print(cox_out[cox_out["rank"] <= 5][["cause", "rank", "feature", "coefficient"]].to_string(index=False))
    print("\n=== Top 5 XGBoost features per disease ===")
    print(xgb_out[xgb_out["rank"] <= 5][["cause", "rank", "feature", "gain"]].to_string(index=False))
    print(f"\nSaved:\n  {cox_path_out}\n  {xgb_path_out}")


if __name__ == "__main__":
    run()

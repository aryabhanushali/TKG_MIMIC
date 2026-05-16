"""Survival baselines (Cox + Gradient Boosting Survival) using the same
features as baseline.py: bag-of-codes (train concepts) + per-concept value
summaries (mean/max/min/last/count/slope) + static features.

Trains one cause-specific model per endpoint (other events treated as censored
at their occurrence time) for fair per-cause comparison vs TGN-survival.
"""
import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
import xgboost as xgb

from src.config import OUTPUT_DIR, SEED
from src.baseline import (
    _build_bag_of_codes, _build_value_summary_features,
)

MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
SURV_DIR = os.path.join(OUTPUT_DIR, "baselines_survival")
CAUSES = ["MI", "Stroke", "HF", "AF", "PAD"]
HORIZON_DAYS = [365, 1095, 1825]   # match tgn_survival


def _load():
    print("Loading modeling artifacts...")
    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    static = pd.read_csv(os.path.join(MODELING_DIR, "static_features.csv"))
    events = pd.read_csv(os.path.join(MODELING_DIR, "events.csv"))
    nodes  = pd.read_csv(os.path.join(MODELING_DIR, "node_metadata.csv"))
    return labels, static, events, nodes


def _build_X(labels, static, events, nodes):
    labels = labels.sort_values("subject_id").reset_index(drop=True)
    static = static.sort_values("subject_id").reset_index(drop=True)
    assert (labels["subject_id"].to_numpy() == static["subject_id"].to_numpy()).all()
    patient_order = labels["subject_id"].tolist()
    split = labels["split"].to_numpy()
    train_mask = split == "train"
    val_mask   = split == "val"
    test_mask  = split == "test"
    train_ids_set = set(np.array(patient_order)[train_mask].tolist())

    train_events_mask = events["subject_id"].isin(train_ids_set)
    concept_ids = sorted(events.loc[train_events_mask, "concept_node_idx"]
                          .unique().tolist())
    print(f"  bag-of-codes dims: {len(concept_ids):,}")
    X_bow = _build_bag_of_codes(events, patient_order, concept_ids)

    X_values, value_col_names = _build_value_summary_features(
        events, patient_order, nodes, train_ids_set)
    print(f"  value summary dims: {X_values.shape[1]:,}")

    static_cols = ["age_at_index", "cci_score", "num_cardiometa_conditions",
                   "had_icu_stay", "female"]
    X_static = static[static_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(X_static[train_mask])
    X_static = scaler.transform(X_static)

    X = hstack([X_bow,
                sparse.csr_matrix(X_values),
                sparse.csr_matrix(X_static)]).tocsr()
    return X, labels, train_mask, val_mask, test_mask


def _make_y(labels: pd.DataFrame, cause: str):
    """Cause-specific Surv. Patients with a different observed event are
    treated as censored at their event time (standard cause-specific framing)."""
    event_bool = (labels["endpoint_type"] == cause).to_numpy().astype(bool)
    time = labels["time_to_event_days"].to_numpy(dtype=float)
    time = np.maximum(time, 1.0)  # sksurv requires positive durations
    return Surv.from_arrays(event_bool, time)


def _eval_horizon_auroc(risk_score, labels, cause, horizons):
    """Per-horizon AUROC: y=1 iff (cause observed and duration<=h), score=risk."""
    durs = labels["time_to_event_days"].to_numpy(dtype=float)
    evts = labels["endpoint_type"].to_numpy()
    rows = []
    for h in horizons:
        keep_pos = (evts == cause) & (durs <= h)
        keep_neg = (durs >= h)
        mask = keep_pos | keep_neg
        y = keep_pos[mask].astype(int)
        s = risk_score[mask]
        if y.sum() == 0 or y.sum() == len(y):
            rows.append({"cause": cause, "horizon_days": h,
                         "auroc": np.nan, "auprc": np.nan,
                         "n_pos": int(y.sum()), "n": int(len(y))})
            continue
        rows.append({"cause": cause, "horizon_days": h,
                     "auroc": roc_auc_score(y, s),
                     "auprc": average_precision_score(y, s),
                     "n_pos": int(y.sum()), "n": int(len(y))})
    return rows


def run() -> None:
    os.makedirs(SURV_DIR, exist_ok=True)
    labels, static, events, nodes = _load()
    X, labels, tr_m, va_m, te_m = _build_X(labels, static, events, nodes)
    print(f"  full X: shape={X.shape}, nnz={X.nnz:,}")
    X = X.astype(np.float32)
    X_tr = X[tr_m]; X_te = X[te_m]
    labels_tr = labels[tr_m].reset_index(drop=True)
    labels_te = labels[te_m].reset_index(drop=True)

    all_rows = []
    test_predictions = pd.DataFrame({"subject_id": labels_te["subject_id"].to_numpy()})
    for cause in CAUSES:
        print(f"\n=== {cause} ===")
        y_tr = _make_y(labels_tr, cause)
        y_te = _make_y(labels_te, cause)

        # ---- Cox (elastic-net regularized; single alpha, no path search) -
        # Single regularization point + relaxed tolerance = ~5x speedup over
        # the default 5-alpha path. Reviewers expect Cox as the baseline.
        print("  fitting CoxNet (single-alpha elastic-net Cox)...", flush=True)
        cox = CoxnetSurvivalAnalysis(
            l1_ratio=0.9, alphas=[0.01],
            max_iter=80, tol=1e-3,
        )
        try:
            X_tr_dense = X_tr.toarray()
            cox.fit(X_tr_dense, y_tr)
            cox_risk = cox.predict(X_te.toarray()).ravel()
            cox_rows = _eval_horizon_auroc(cox_risk, labels_te, cause, HORIZON_DAYS)
            for r in cox_rows:
                r["model"] = "cox"; all_rows.append(r)
            test_predictions[f"cox_risk_{cause}"] = cox_risk
            del X_tr_dense
        except Exception as e:
            print(f"    Cox failed: {e}")

        # ---- XGBoost survival (objective=survival:cox; sparse-native) -----
        # Stronger baseline — XGBoost was the hardest model to beat in the
        # multiclass setup. Survival:cox uses signed duration encoding:
        # positive duration = event observed, negative = censored at |t|.
        print("  fitting XGBoost (survival:cox)...", flush=True)
        try:
            event_tr = (labels_tr["endpoint_type"] == cause).to_numpy().astype(bool)
            dur_tr = labels_tr["time_to_event_days"].to_numpy(dtype=float)
            dur_tr = np.maximum(dur_tr, 1.0)
            y_xgb = np.where(event_tr, dur_tr, -dur_tr).astype(np.float32)

            xgb_clf = xgb.XGBRegressor(
                objective="survival:cox",
                eval_metric="cox-nloglik",
                n_estimators=250, learning_rate=0.08, max_depth=6,
                subsample=0.85, colsample_bytree=0.7,
                tree_method="hist", n_jobs=-1, random_state=SEED,
            )
            # XGBoost handles sparse natively — no .toarray() needed
            xgb_clf.fit(X_tr, y_xgb)
            xgb_risk = xgb_clf.predict(X_te)
            xgb_rows = _eval_horizon_auroc(xgb_risk, labels_te, cause, HORIZON_DAYS)
            for r in xgb_rows:
                r["model"] = "xgb_surv"; all_rows.append(r)
            test_predictions[f"xgb_surv_risk_{cause}"] = xgb_risk
        except Exception as e:
            print(f"    XGBoost-survival failed: {e}")

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(SURV_DIR, "test_metrics.csv")
    df.to_csv(out_path, index=False)
    preds_path = os.path.join(SURV_DIR, "predictions_test.csv")
    test_predictions.to_csv(preds_path, index=False)

    print("\n=== SURVIVAL BASELINE TEST METRICS ===")
    for m in df["model"].unique():
        print(f"\n{m.upper()} AUROC:")
        print(df[df["model"] == m].pivot(
            index="cause", columns="horizon_days", values="auroc").round(3).to_string())
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()

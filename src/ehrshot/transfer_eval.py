"""Apply the frozen MIMIC-IV Cox / XGBoost survival models to EHRSHOT.

Same idea as src/eicu/transfer_eval.py, but simpler and closer to MIMIC. The
EHRSHOT TKG is already in the MIMIC concept-ID scheme (src/ehrshot/build_tkg_ehrshot.py),
so the MIMIC feature matrix can be rebuilt from EHRSHOT's own
modeling/events.parquet (same pre-index window, same dedup) by looking concepts
up by their *string* ID:

  bag-of-codes : counts per concept in feature_meta["concept_id_to_bow_col"]
  values       : mean/max/min/last/count/slope per concept in
                 feature_meta["feat_concept_ids_str"], z-scored with MIMIC's
                 train mu/sd and clipped to +/-10 (as baseline.py does)
  static       : the 5 static columns, standardised with MIMIC's scaler

Scoring reuses src.eicu.transfer_eval._score. Evaluation reuses
baselines_survival._eval_horizon_auroc (same competing-risks rule, 1/3/5 y).
Two populations are reported: the full EHRSHOT cohort (no EHRSHOT data was
used to fit these models) and the EHRSHOT test split (directly comparable to
the natively-trained EHRSHOT models).

Requires: MIMIC_OUTPUT_DIR = EHRSHOT output dir (for modeling/), and
MIMIC_TRANSFER_DIR = the MIMIC run holding baselines_survival/transfer_artifacts.
Output: OUTPUT_DIR/transfer_from_mimic/{discrimination.csv, coverage.csv}
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import roc_auc_score

from src.config import OUTPUT_DIR, read_events_table
from src.baselines_survival import _eval_horizon_auroc, CAUSES, HORIZON_DAYS
from src.eicu.transfer_eval import _score
from src.ehrshot.config_ehrshot import MIMIC_TRANSFER_DIR

MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
OUT_DIR = os.path.join(OUTPUT_DIR, "transfer_from_mimic")
N_BOOT = 1000
RNG = np.random.default_rng(42)


def _load_artifacts():
    art = os.path.join(MIMIC_TRANSFER_DIR, "baselines_survival", "transfer_artifacts")
    with open(os.path.join(art, "feature_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    cox, xgbm = {}, {}
    for c in CAUSES:
        for d, prefix in ((cox, "cox"), (xgbm, "xgb")):
            p = os.path.join(art, f"{prefix}_{c}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    d[c] = pickle.load(f)
    print(f"  MIMIC artifacts: {art} (cox {len(cox)}, xgb {len(xgbm)}; "
          f"{meta['n_bow_cols']:,} bow + {meta['n_value_cols']:,} value + "
          f"{meta['n_static_cols']} static cols)")
    return meta, cox, xgbm


def _family(cid: str) -> str:
    for p in ("ICD10_PROC_", "ICD9_PROC_", "ICD10_", "ICD9_", "CPT4_", "DRUG_",
              "LAB_", "VITAL_", "OMR_", "ICU_"):
        if cid.startswith(p):
            return p.rstrip("_")
    return "other"


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 2 or np.allclose(x.std(), 0.0):
        return 0.0
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    return 0.0 if denom == 0 else float(((x - xm) * (y - ym)).sum() / denom)


def build_X(meta: dict, labels: pd.DataFrame, static: pd.DataFrame, events: pd.DataFrame):
    order = labels["subject_id"].tolist()
    row = {s: i for i, s in enumerate(order)}
    n = len(order)

    bow_col = meta["concept_id_to_bow_col"]
    cnt = events.groupby(["subject_id", "concept_id"]).size().reset_index(name="n")
    cnt = cnt[cnt["concept_id"].isin(bow_col) & cnt["subject_id"].isin(row)]
    X_bow = csr_matrix((cnt["n"].to_numpy(np.float32),
                        (cnt["subject_id"].map(row).to_numpy(), cnt["concept_id"].map(bow_col).to_numpy())),
                       shape=(n, meta["n_bow_cols"]))

    feat = meta["feat_concept_ids_str"]
    vcol = {c: j for j, c in enumerate(feat)}
    arr = np.zeros((n, len(feat) * 6), dtype=np.float32)
    ve = events[events["value_num"].notna() & events["concept_id"].isin(vcol)]
    ve = ve.sort_values(["subject_id", "relative_days"])
    for (sid, cid), g in ve.groupby(["subject_id", "concept_id"]):
        if sid not in row:
            continue
        v = g["value_num"].to_numpy(np.float64)
        d = g["relative_days"].to_numpy(np.float64)
        b = vcol[cid] * 6
        arr[row[sid], b:b + 6] = [v.mean(), v.max(), v.min(), v[-1], len(v), _slope(d, v)]
    mu, sd = np.asarray(meta["value_mu"]), np.asarray(meta["value_sd"]).copy()
    sd[sd == 0] = 1.0
    arr = np.clip((arr - mu) / sd, -10.0, 10.0).astype(np.float32)

    st = static.set_index("subject_id").loc[order, meta["static_cols"]].to_numpy(np.float32)
    st_sd = np.where(np.asarray(meta["static_sd"]) == 0, 1.0, meta["static_sd"])
    st = ((st - meta["static_mu"]) / st_sd).astype(np.float32)

    X = hstack([X_bow, csr_matrix(arr), csr_matrix(st)]).tocsr()
    assert X.shape[1] == meta["n_bow_cols"] + meta["n_value_cols"] + meta["n_static_cols"]
    return X


def _coverage(meta: dict, events: pd.DataFrame) -> pd.DataFrame:
    bow = set(meta["concept_id_to_bow_col"])
    ev = events.assign(family=events["concept_id"].map(_family),
                       in_mimic=events["concept_id"].isin(bow))
    ev_cov = ev.groupby("family").agg(events=("in_mimic", "size"), event_share_in_mimic_vocab=("in_mimic", "mean"))
    ehr_concepts = pd.Series(events["concept_id"].unique())
    mimic_hit = pd.Series(sorted(bow)).isin(set(ehr_concepts))
    cc = pd.DataFrame({"concept_id": sorted(bow), "hit": mimic_hit.to_numpy()})
    cc["family"] = cc["concept_id"].map(_family)
    col_cov = cc.groupby("family")["hit"].agg(mimic_cols="size", mimic_col_share_seen_in_ehrshot="mean")
    return ev_cov.join(col_cov, how="outer").reset_index()


def _boot_ci(risk, labels, cause, h):
    durs = labels["time_to_event_days"].to_numpy(float)
    evts = labels["endpoint_type"].to_numpy()
    pos = (evts == cause) & (durs <= h)
    neg = (durs >= h) | ((durs < h) & (evts != cause) & (evts != "censored"))
    m = pos | neg
    y, s = pos[m].astype(int), risk[m]
    if y.sum() < 2:
        return np.nan, np.nan
    boots = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boots.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[1] Loading MIMIC transfer artifacts...")
    meta, cox, xgbm = _load_artifacts()

    print("[2] Loading EHRSHOT modeling tables...")
    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv")).sort_values("subject_id").reset_index(drop=True)
    static = pd.read_csv(os.path.join(MODELING_DIR, "static_features.csv"))
    nodes = pd.read_csv(os.path.join(MODELING_DIR, "node_metadata.csv"))
    events = read_events_table(["subject_id", "concept_node_idx", "relative_days", "value_num"])
    events["concept_id"] = events["concept_node_idx"].map(nodes.set_index("node_idx")["concept_id"])

    cov = _coverage(meta, events)
    cov.to_csv(os.path.join(OUT_DIR, "coverage.csv"), index=False)
    print("  vocabulary coverage (event share whose concept is in MIMIC's bag-of-codes vocab):")
    print(cov.round(3).to_string(index=False))

    print("[3] Building MIMIC-schema feature matrix for EHRSHOT...")
    X = build_X(meta, labels, static, events)
    print(f"  X: {X.shape}, nnz={X.nnz:,}")

    print("[4] Scoring with frozen MIMIC models...")
    scores = _score(X, cox, xgbm, meta)
    # Per-patient scores (MIMIC column names) so paired tests against the
    # natively trained EHRSHOT models are possible (src/ehrshot/cv_pool.py).
    per_pt = pd.DataFrame({"subject_id": labels["subject_id"].to_numpy()})
    for cause in CAUSES:
        for model, col in (("cox", "cox_risk_{c}"), ("xgb", "xgb_surv_risk_{c}")):
            if scores[cause][model] is not None:
                per_pt[col.format(c=cause)] = scores[cause][model]
    per_pt.to_csv(os.path.join(OUT_DIR, "predictions_all.csv"), index=False)

    print("[5] Evaluating (competing-risks AUROC/AUPRC at 1/3/5 y)...")
    rows = []
    pops = {"ehrshot_all": np.ones(len(labels), bool),
            "ehrshot_test": (labels["split"] == "test").to_numpy()}
    for pop, m in pops.items():
        lab = labels[m].reset_index(drop=True)
        for cause in CAUSES:
            for model in ("cox", "xgb"):
                r = scores[cause][model]
                if r is None:
                    continue
                for res in _eval_horizon_auroc(r[m], lab, cause, HORIZON_DAYS):
                    lo, hi = (_boot_ci(r[m], lab, cause, res["horizon_days"])
                              if not np.isnan(res["auroc"]) else (np.nan, np.nan))
                    rows.append(dict(population=pop, model=f"mimic_{model}_transfer",
                                     **res, auroc_lo=lo, auroc_hi=hi))
    disc = pd.DataFrame(rows)
    disc.to_csv(os.path.join(OUT_DIR, "discrimination.csv"), index=False)
    for pop in pops:
        print(f"\n=== {pop}: AUROC ===")
        print(disc[disc["population"] == pop].pivot_table(
            index=["model", "cause"], columns="horizon_days", values="auroc").round(3).to_string())
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    run()

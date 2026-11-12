"""Pool K-fold CV results for the natively-trained EHRSHOT survival models.

Each fold k lives in <cv_root>/fold{k}/ (a full MIMIC_OUTPUT_DIR run with
TKG_SPLIT_MODE=cv TKG_CV_FOLD=k). Every patient is in exactly one test fold,
so concatenating the per-fold test predictions gives an out-of-fold (OOF)
prediction for the whole cohort. Two views are reported:

  pooled   : one AUROC/AUPRC per (model, cause, horizon) over all OOF
             predictions, with a 1000x bootstrap CI. Risk scores are
             percentile-ranked within each fold before pooling: Cox/XGB risk
             scales differ between independently fit fold models, and ranking
             removes that without changing any within-fold ordering.
  per_fold : mean +/- sd of the per-fold test AUROC (the usual CV summary).

When several model seeds were run (TKG_SEED, *_seed{N} dirs), pooled metrics
are computed per seed and then averaged; the CI comes from seed 42.
The frozen MIMIC -> EHRSHOT transfer models (src/ehrshot/transfer_eval.py, full
cohort) are appended for comparison when available.

Usage: python -m src.ehrshot.cv_summary <cv_root> [--folds 5]
Output: <cv_root>/cv_summary.csv
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.baselines_survival import _eval_horizon_auroc, CAUSES, HORIZON_DAYS

RNG = np.random.default_rng(42)
N_BOOT = 1000
MODELS = {  # model -> (results dir stem, score column template)
    "cox": ("baselines_survival", "cox_risk_{c}"),
    "xgb_surv": ("baselines_survival", "xgb_surv_risk_{c}"),
    "tkg_transformer": ("tgn_survival", "cif_{c}_at_{h}d"),
    "patient_graph_gnn": ("patient_gnn_survival", "cif_{c}_at_{h}d"),
}


def _seeds(fold_dir: str) -> list[int]:
    seeds = {42} if os.path.isdir(os.path.join(fold_dir, "baselines_survival")) else set()
    for p in glob.glob(os.path.join(fold_dir, "*_seed*")):
        m = re.search(r"_seed(\d+)$", p)
        if m:
            seeds.add(int(m.group(1)))
    return sorted(seeds)


def _pred_path(fold_dir, stem, seed):
    d = stem if seed == 42 else f"{stem}_seed{seed}"
    return os.path.join(fold_dir, d, "predictions_test.csv")


def _boot(score, labels, cause, h):
    durs = labels["time_to_event_days"].to_numpy(float)
    evts = labels["endpoint_type"].to_numpy()
    pos = (evts == cause) & (durs <= h)
    m = pos | (durs >= h) | ((durs < h) & (evts != cause) & (evts != "censored"))
    y, s = pos[m].astype(int), score[m]
    b = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            b.append(roc_auc_score(y[i], s[i]))
    return (np.percentile(b, 2.5), np.percentile(b, 97.5)) if b else (np.nan, np.nan)


def run(cv_root: str, n_folds: int) -> pd.DataFrame:
    folds = [os.path.join(cv_root, f"fold{k}") for k in range(n_folds)]
    labels = {k: pd.read_csv(os.path.join(f, "modeling", "labels.csv")) for k, f in enumerate(folds)}
    test_ids = [set(l.loc[l["split"] == "test", "subject_id"]) for l in labels.values()]
    assert all(not (a & b) for i, a in enumerate(test_ids) for b in test_ids[i + 1:]), \
        "a patient is in more than one test fold"
    n_all = len(labels[0])
    print(f"{n_folds} folds, {n_all:,} patients, test fold sizes "
          f"{[len(t) for t in test_ids]} (sum {sum(map(len, test_ids)):,})")
    seeds = sorted(set.intersection(*[set(_seeds(f)) for f in folds]))
    print(f"seeds present in every fold: {seeds}")

    pooled_rows, fold_rows = [], []
    for model, (stem, tmpl) in MODELS.items():
        for seed in seeds:
            parts = []
            for k, f in enumerate(folds):
                p = _pred_path(f, stem, seed)
                if not os.path.exists(p):
                    break
                pr = pd.read_csv(p)
                lab = labels[k][labels[k]["split"] == "test"][["subject_id", "endpoint_type", "time_to_event_days"]]
                pr = lab.merge(pr.drop(columns=[c for c in ("endpoint_true", "duration_days") if c in pr]),
                               on="subject_id", how="inner")
                pr["fold"] = k
                parts.append(pr)
            if len(parts) != n_folds:
                print(f"  skip {model} seed {seed}: predictions missing in some fold")
                continue
            for cause in CAUSES:
                for h in HORIZON_DAYS:
                    col = tmpl.format(c=cause, h=h)
                    missing = [int(pr["fold"].iloc[0]) for pr in parts if col not in pr]
                    if missing:   # e.g. a CoxNet fit that diverged in one fold
                        if len(missing) < n_folds:
                            print(f"  skip {model} seed {seed} {cause} @{h}d: no predictions in fold(s) {missing}")
                        continue
                    for pr in parts:   # per-fold metric, raw scores
                        r = _eval_horizon_auroc(pr[col].to_numpy(float), pr, cause, [h])[0]
                        fold_rows.append(dict(model=model, seed=seed, fold=int(pr["fold"].iloc[0]), **r))
                    allp = pd.concat([pr.assign(score=pr[col].rank(pct=True)) for pr in parts],
                                     ignore_index=True)
                    r = _eval_horizon_auroc(allp["score"].to_numpy(float), allp, cause, [h])[0]
                    lo, hi = (_boot(allp["score"].to_numpy(float), allp, cause, h)
                              if seed == 42 and not np.isnan(r["auroc"]) else (np.nan, np.nan))
                    pooled_rows.append(dict(model=model, seed=seed, **r, auroc_lo=lo, auroc_hi=hi))

    pooled = pd.DataFrame(pooled_rows)
    per_fold = pd.DataFrame(fold_rows)
    agg = (pooled.groupby(["model", "cause", "horizon_days"])
           .agg(pooled_auroc=("auroc", "mean"), pooled_auroc_seed_sd=("auroc", "std"),
                pooled_auprc=("auprc", "mean"), n_pos=("n_pos", "first"), n=("n", "first"),
                n_seeds=("seed", "nunique")).reset_index())
    ci = pooled[pooled["seed"] == 42][["model", "cause", "horizon_days", "auroc_lo", "auroc_hi"]]
    agg = agg.merge(ci, on=["model", "cause", "horizon_days"], how="left")
    pf = (per_fold.groupby(["model", "cause", "horizon_days"])["auroc"]
          .agg(fold_auroc_mean="mean", fold_auroc_sd="std").reset_index())
    out = agg.merge(pf, on=["model", "cause", "horizon_days"], how="left")

    tr = os.path.join(os.path.dirname(os.path.abspath(cv_root)), "transfer_from_mimic", "discrimination.csv")
    if os.path.exists(tr):
        t = pd.read_csv(tr)
        t = t[t["population"] == "ehrshot_all"].rename(columns={"auroc": "pooled_auroc", "auprc": "pooled_auprc"})
        t = t[["model", "cause", "horizon_days", "pooled_auroc", "pooled_auprc", "n_pos", "n",
               "auroc_lo", "auroc_hi"]]
        out = pd.concat([out, t], ignore_index=True)

    out.to_csv(os.path.join(cv_root, "cv_summary.csv"), index=False)
    per_fold.to_csv(os.path.join(cv_root, "cv_per_fold.csv"), index=False)
    for h in HORIZON_DAYS:
        print(f"\n=== Full-cohort (out-of-fold) AUROC, horizon {h} d ===")
        w = out[out["horizon_days"] == h].copy()
        w["cell"] = w.apply(lambda r: f"{r.pooled_auroc:.3f} [{r.auroc_lo:.2f}-{r.auroc_hi:.2f}]", axis=1)
        print(w.pivot(index="cause", columns="model", values="cell").to_string())
        print("  n_pos: " + ", ".join(f"{c}={int(v)}" for c, v in
              w.groupby("cause")["n_pos"].first().items()))
    print(f"\nSaved: {os.path.join(cv_root, 'cv_summary.csv')}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cv_root")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    run(a.cv_root, a.folds)

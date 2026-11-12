"""Does the TKG models' margin over the flat-feature baselines depend on how
many cases a disease has?

Unit of analysis: one (dataset/run, model pair, disease, horizon) cell, with
its seed-mean AUROC margin from paired_robustness.csv, and the number of
TRAINING events for that disease in that run (horizon-independent; it is what
the model learns from). Runs:

  MIMIC full           tkg_output/
  MIMIC learning curve tkg_output_lc/frac{0.05,0.1,0.25,0.5}/   (same test set,
                       fewer training patients; run_mimic_learning_curve.sbatch)
  EHRSHOT 5-fold CV    tkg_output_ehrshot/cv/pooled/   (training events =
                       mean over folds of that fold's train split)

Two questions, kept apart because they answer different things:
  1. Across diseases (observational): within one run, do diseases with fewer
     training events show a larger / smaller margin? Disease identity and
     case count are confounded here (PAD is both rare and atypical).
  2. Within disease (controlled): on MIMIC, shrinking the training set while
     holding disease, test set and pipeline fixed -- does the margin change?
     Fit margin ~ log10(train events) + disease fixed effects + horizon
     fixed effects on the MIMIC learning-curve cells, with cluster-robust
     SEs by disease; EHRSHOT is shown alongside as an out-of-sample check.

Outputs (OUTPUT root = tkg_output/stats/case_count/):
  case_count_cells.csv      every cell with margin, n_pos, train events
  case_count_models.csv     regression slopes per pair
  learning_curve_auroc.csv  mean AUROC per model per run
Usage: python -m src.ablations.case_count_analysis
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "tkg_output", "stats", "case_count")
CAUSES = ["MI", "Stroke", "HF", "AF", "PAD"]
PAIRS = ["tkg_vs_cox", "tkg_vs_xgb", "pg_vs_cox", "pg_vs_xgb", "xgb_vs_cox"]


def _train_events(label_paths) -> dict:
    """Mean (over label files) count of train-split events per cause."""
    counts = []
    for p in label_paths:
        lab = pd.read_csv(p)
        tr = lab[lab.split == "train"]
        counts.append({c: int((tr.endpoint_type == c).sum()) for c in CAUSES})
    return {c: float(np.mean([d[c] for d in counts])) for c in CAUSES}


def _runs():
    runs = [("MIMIC", 1.0, os.path.join(REPO, "tkg_output"),
             [os.path.join(REPO, "tkg_output", "modeling", "labels.csv")])]
    for d in sorted(glob.glob(os.path.join(REPO, "tkg_output_lc", "frac*"))):
        frac = float(re.search(r"frac([\d.]+)$", d).group(1))
        runs.append(("MIMIC", frac, d, [os.path.join(d, "modeling", "labels.csv")]))
    cv = os.path.join(REPO, "tkg_output_ehrshot", "cv")
    runs.append(("EHRSHOT", 1.0, os.path.join(cv, "pooled"),
                 sorted(glob.glob(os.path.join(cv, "fold*", "modeling", "labels.csv")))))
    return runs


def _ols_cluster(df: pd.DataFrame, y: str, x: str, fe: list, cluster: str):
    """OLS of y on x + fixed effects, cluster-robust (CR1) SE for x's slope."""
    X = pd.get_dummies(df[fe].astype(str), drop_first=True).astype(float)
    X.insert(0, x, df[x].to_numpy(float))
    X.insert(0, "const", 1.0)
    Xm, yv = X.to_numpy(), df[y].to_numpy(float)
    XtX_inv = np.linalg.pinv(Xm.T @ Xm)
    beta = XtX_inv @ Xm.T @ yv
    resid = yv - Xm @ beta
    meat = np.zeros((Xm.shape[1], Xm.shape[1]))
    groups = df[cluster].to_numpy()
    G = len(np.unique(groups))
    for g in np.unique(groups):
        s = Xm[groups == g].T @ resid[groups == g]
        meat += np.outer(s, s)
    n, k = Xm.shape
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    se = float(np.sqrt(V[1, 1]))
    t = beta[1] / se
    return float(beta[1]), se, float(2 * stats.t.sf(abs(t), G - 1))


def run() -> None:
    os.makedirs(OUT, exist_ok=True)
    cells, auroc_rows = [], []
    for dataset, frac, d, label_paths in _runs():
        pr = os.path.join(d, "stats", "paired_robustness.csv")
        if not os.path.exists(pr) or not all(os.path.exists(p) for p in label_paths):
            print(f"  skip {dataset} frac={frac}: {pr} not found (run paired_robustness first)")
            continue
        tr_ev = _train_events(label_paths)
        r = pd.read_csv(pr)
        r = r[r.pair.isin(PAIRS)].copy()
        r["dataset"], r["train_frac"], r["run"] = dataset, frac, f"{dataset} {frac:g}"
        r["train_events"] = r.cause.map(tr_ev)
        cells.append(r)
        # absolute AUROC per model in this run (model side of each pair, + baselines)
        a = pd.concat([r[["model", "cause", "horizon_days", "auc_model"]].rename(columns={"auc_model": "auc"}),
                       r[["comparator", "cause", "horizon_days", "auc_comparator"]]
                       .rename(columns={"comparator": "model", "auc_comparator": "auc"})])
        a = a.drop_duplicates(["model", "cause", "horizon_days"])
        for m, sub in a.groupby("model"):
            auroc_rows.append(dict(dataset=dataset, train_frac=frac, model=m, mean_auroc=sub.auc.mean(),
                                   n_cells=len(sub), mean_train_events=np.mean(list(tr_ev.values()))))
    if not cells:
        print("No runs with paired_robustness.csv found.")
        return
    cells = pd.concat(cells, ignore_index=True)
    cells["log10_train_events"] = np.log10(cells.train_events)
    cells.to_csv(os.path.join(OUT, "case_count_cells.csv"), index=False)
    lc = pd.DataFrame(auroc_rows).sort_values(["dataset", "train_frac", "model"])
    lc.to_csv(os.path.join(OUT, "learning_curve_auroc.csv"), index=False)

    rows = []
    print("\n=== 1. Across diseases within a run: Spearman(margin, train events) over 15 cells ===")
    for (run_name, pair), sub in cells.groupby(["run", "pair"]):
        rho, p = stats.spearmanr(sub.train_events, sub.delta)
        rarest = sub[sub.cause == sub.groupby("cause").train_events.first().idxmin()]
        rows.append(dict(analysis="across_disease", run=run_name, pair=pair, n_cells=len(sub),
                         spearman_rho=rho, p=p, margin_mean=sub.delta.mean(),
                         rarest_disease=rarest.cause.iloc[0], margin_rarest=rarest.delta.mean()))
    acr = pd.DataFrame(rows)
    print(acr[["run", "pair", "spearman_rho", "p", "margin_mean", "rarest_disease", "margin_rarest"]]
          .round(3).to_string(index=False))

    print("\n=== 2. Within disease (MIMIC learning curve): margin ~ log10(train events) + disease + horizon FE ===")
    mim = cells[cells.dataset == "MIMIC"]
    rows2 = []
    if mim.train_frac.nunique() >= 2:
        for pair, sub in mim.groupby("pair"):
            slope, se, p = _ols_cluster(sub, "delta", "log10_train_events", ["cause", "horizon_days"], "cause")
            by_frac = sub.groupby("train_frac").delta.mean()
            rows2.append(dict(analysis="within_disease_mimic_lc", pair=pair, n_cells=len(sub),
                              slope_per_10x_events=slope, se_cluster=se, p=p,
                              **{f"margin_frac{f:g}": v for f, v in by_frac.items()}))
        wd = pd.DataFrame(rows2)
        print(wd.round(4).to_string(index=False))
        print("  (negative slope = the TKG model's margin GROWS as training events shrink)")
    else:
        wd = pd.DataFrame()
        print("  learning-curve runs not available yet")
    pd.concat([acr, wd], ignore_index=True).to_csv(os.path.join(OUT, "case_count_models.csv"), index=False)

    print("\n=== Mean AUROC per model per run ===")
    print(lc.pivot_table(index=["dataset", "train_frac"], columns="model", values="mean_auroc").round(3).to_string())
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    run()

"""Paired, multi-seed robustness test for every model-vs-baseline comparison,
on whatever test set MIMIC_OUTPUT_DIR holds: the MIMIC fixed test set, or an
EHRSHOT pooled out-of-fold dir from src/ehrshot/cv_pool.py.

Same bar the paper already uses for MIMIC (delong_robustness_audit.py), made
self-contained so it runs on any dataset without the per-dataset summary
CSVs or checkpoint re-inference:

  1. Seed-level test, Bonferroni-corrected within each model's family of
     30 comparisons (15 disease x horizon cells x {Cox, XGBoost}):
       vs Cox     (deterministic, one fit): one-sample t-test of the model's
                  per-seed AUROCs against Cox's single AUROC;
       vs XGBoost (stochastic): paired t-test, seed s against seed s.
  2. DeLong (paired on the same patients) significant at p<0.05 with the
     same sign in >= 4 of the model's seeds.
  A comparison is ROBUST only if both hold.

Pairs: TKG-Transformer and patient-graph model vs Cox and XGBoost, plus
XGBoost vs Cox; on pooled EHRSHOT also every native model vs the frozen
MIMIC XGBoost. Bonferroni families are per model (15 cells x its number of
comparators, transfer pairs in their own 15-cell family). A deterministic-vs-deterministic pair (Cox vs MIMIC XGBoost)
has one seed, so its "seed-level test" is the DeLong p itself.

Output: OUTPUT_DIR/stats/paired_robustness.csv and a printed summary.
Usage:  MIMIC_OUTPUT_DIR=<dir> python -m src.ablations.paired_robustness
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from src.config import OUTPUT_DIR
from src.baselines_survival import CAUSES, HORIZON_DAYS
from src.evaluate_stats import delong_test, _labels_for
from src.ablations.discussion_findings import load_scores, MODEL_LABEL

PAIRS = [("tkg", "cox"), ("tkg", "xgb"), ("pg", "cox"), ("pg", "xgb"), ("xgb", "cox"),
         # native EHRSHOT models vs the frozen MIMIC XGBoost (only where
         # mimic_transfer/predictions_test.csv exists, i.e. pooled EHRSHOT)
         ("tkg", "mimic_xgb"), ("pg", "mimic_xgb"), ("xgb", "mimic_xgb"), ("cox", "mimic_xgb")]
LABEL = {**MODEL_LABEL, "mimic_xgb": "MIMIC XGBoost (frozen)", "mimic_cox": "MIMIC Cox (frozen)"}


def _load_transfer() -> dict:
    path = os.path.join(OUTPUT_DIR, "mimic_transfer", "predictions_test.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path).set_index("subject_id")
    return {"mimic_xgb": {42: {(c, h): df[f"xgb_surv_risk_{c}"] for c in CAUSES for h in HORIZON_DAYS}},
            "mimic_cox": {42: {(c, h): df[f"cox_risk_{c}"] for c in CAUSES for h in HORIZON_DAYS}}}
SEED_AGREE = 4
ALPHA = 0.05


def run() -> pd.DataFrame:
    labels = pd.read_csv(os.path.join(OUTPUT_DIR, "modeling", "labels.csv"))
    test = labels[labels.split == "test"].reset_index(drop=True)
    scores = {**load_scores(), **_load_transfer()}
    sids = test.subject_id.to_numpy()
    evts, durs = test.endpoint_type.to_numpy(), test.time_to_event_days.to_numpy(float)

    rows = []
    for a, b in PAIRS:
        if a not in scores or b not in scores:
            continue
        for cause in CAUSES:
            for h in HORIZON_DAYS:
                y, m = _labels_for(cause, h, evts, durs)
                y = y[m] if len(y) == len(m) else y
                per_seed = []
                for seed, cells in scores[a].items():
                    sa = cells[(cause, h)].reindex(sids).to_numpy(float)[m]
                    b_seed = seed if seed in scores[b] else min(scores[b])
                    sb = scores[b][b_seed][(cause, h)].reindex(sids).to_numpy(float)[m]
                    ok = ~(np.isnan(sa) | np.isnan(sb))
                    auc_a, auc_b, p = delong_test(y[ok], sa[ok], sb[ok])
                    per_seed.append((seed, auc_a, auc_b, p))
                ps = pd.DataFrame(per_seed, columns=["seed", "auc_a", "auc_b", "p"])
                diff = ps.auc_a - ps.auc_b
                if len(ps) == 1:      # deterministic vs deterministic: DeLong is the test
                    t_p = ps.p.iloc[0]
                elif len(scores[b]) == 1:
                    t_p = stats.ttest_1samp(ps.auc_a, ps.auc_b.iloc[0]).pvalue if len(ps) > 1 else np.nan
                else:
                    t_p = stats.ttest_rel(ps.auc_a, ps.auc_b).pvalue if len(ps) > 1 else np.nan
                sign = np.sign(diff.mean())
                n_delong = int(((ps.p < ALPHA) & (np.sign(diff) == sign)).sum())
                rows.append(dict(
                    pair=f"{a}_vs_{b}", model=a, comparator=b, cause=cause, horizon_days=h,
                    n_pos=int(y.sum()), n=int(len(y)), n_seeds=len(ps),
                    auc_model=ps.auc_a.mean(), auc_comparator=ps.auc_b.mean(),
                    delta=diff.mean(), delta_sd=diff.std(ddof=1) if len(ps) > 1 else np.nan,
                    seeds_model_better=int((diff > 0).sum()),
                    ttest_p=t_p, delong_sig_seeds_same_dir=n_delong,
                    delong_median_p=float(ps.p.median())))
    res = pd.DataFrame(rows)
    # Families: the paper's per-model family of native comparisons (30), and
    # a separate 15-cell family for each model's comparison with the frozen
    # MIMIC model, so adding transfer pairs never tightens the native bar.
    res["family"] = res.model + np.where(res.comparator.str.startswith("mimic_"), "_vs_transfer", "")
    fam_size = res.groupby("family").pair.transform("size")
    res["ttest_p_bonf"] = (res.ttest_p * fam_size).clip(upper=1.0)
    need = res.n_seeds.clip(upper=SEED_AGREE)
    res["robust"] = (res.ttest_p_bonf < ALPHA) & (res.delong_sig_seeds_same_dir >= need)
    res["robust_direction"] = np.where(res.robust, np.where(res.delta > 0, "model wins", "comparator wins"), "")
    os.makedirs(os.path.join(OUTPUT_DIR, "stats"), exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "stats", "paired_robustness.csv")
    res.to_csv(out, index=False)

    print(f"\n=== Paired robustness ({OUTPUT_DIR}) ===")
    for pair, sub in res.groupby("pair", sort=False):
        a, b = pair.split("_vs_")
        print(f"\n{LABEL[a]} vs {LABEL[b]}: mean delta {sub.delta.mean():+.3f}; "
              f"model ahead in {(sub.delta > 0).sum()}/{len(sub)} cells; "
              f"t-test Bonferroni-sig {int((sub.ttest_p_bonf < ALPHA).sum())}; "
              f"ROBUST {int(sub.robust.sum())} ({', '.join(f'{r.cause} {r.horizon_days // 365}y: {r.robust_direction}' for r in sub[sub.robust].itertuples()) or 'none'})")
    print(f"\nSaved: {out}")
    return res


if __name__ == "__main__":
    run()

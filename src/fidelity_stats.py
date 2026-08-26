"""Paired significance testing for explanation-fidelity checks.

The three fidelity scripts (explain_gnn.py, ablations/explain_gnn_concept_
graph.py, ablations/explain_patient_graph_fidelity.py) each compare a
"top-k important" KL divergence against a "random-k" KL divergence, per
patient, and previously reported only the mean over patients. A mean alone
can be dominated by a handful of outlier patients and hide the fact that the
"important beats random" direction may not hold for a typical patient. This
module adds what should sit next to any such mean: a paired t-test, a
Wilcoxon signed-rank test (robust to the skew that KL divergences often
have), a bootstrap CI on the per-patient win-rate, and a paired Cohen's d.

Report both tests together, not just one -- when they disagree (t-test
significant, Wilcoxon not, or vice versa), that disagreement is itself the
finding: it means the mean-level effect is driven by a subset of patients
rather than holding broadly, and the paper should say so rather than quoting
whichever test happens to support the claim.
"""
import numpy as np
import pandas as pd
from scipy import stats


def paired_fidelity_test(top: np.ndarray, random: np.ndarray, better: str,
                          n_boot: int = 5000, seed: int = 42) -> dict:
    """Compare per-patient top-k vs random-k KL divergence.

    better: "lower" if top-k should score lower than random (sufficiency),
            "higher" if top-k should score higher (comprehensiveness).
    """
    assert better in ("lower", "higher")
    top = np.asarray(top, dtype=float)
    random = np.asarray(random, dtype=float)
    n = len(top)
    diff = top - random
    win = (top < random) if better == "lower" else (top > random)
    win_rate = float(win.mean())

    t_stat, p_ttest = stats.ttest_rel(top, random)
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(top, random)
    except ValueError:
        w_stat, p_wilcoxon = np.nan, np.nan

    sd = diff.std(ddof=1)
    cohens_d = float(diff.mean() / sd) if sd > 0 else np.nan

    rng = np.random.default_rng(seed)
    boots = win[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    return dict(
        n=n, mean_top=float(top.mean()), mean_random=float(random.mean()),
        mean_diff=float(diff.mean()), win_rate=win_rate,
        win_rate_ci_lo=float(ci_lo), win_rate_ci_hi=float(ci_hi),
        t_stat=float(t_stat), p_ttest=float(p_ttest),
        wilcoxon_stat=float(w_stat) if np.isfinite(w_stat) else np.nan,
        p_wilcoxon=float(p_wilcoxon) if np.isfinite(p_wilcoxon) else np.nan,
        cohens_d=cohens_d,
    )


def summarize_fidelity(fidelity: pd.DataFrame) -> pd.DataFrame:
    """Run paired_fidelity_test for both sufficiency and comprehensiveness.

    Expects columns kl_keep_top, kl_keep_random, kl_drop_top, kl_drop_random
    (the schema shared by all three fidelity CSVs in this project).
    """
    suff = paired_fidelity_test(fidelity["kl_keep_top"], fidelity["kl_keep_random"], "lower")
    comp = paired_fidelity_test(fidelity["kl_drop_top"], fidelity["kl_drop_random"], "higher")
    suff["direction"] = "sufficiency"
    comp["direction"] = "comprehensiveness"
    return pd.DataFrame([suff, comp]).set_index("direction")


def print_fidelity_summary(summary: pd.DataFrame, label: str) -> None:
    print(f"\nFidelity significance testing ({label}):")
    for direction, r in summary.iterrows():
        verdict = ("robust" if (r["p_ttest"] < 0.05 and r["p_wilcoxon"] < 0.05)
                    else "mean-only / fragile" if (r["p_ttest"] < 0.05) != (r["p_wilcoxon"] < 0.05)
                    else "not significant")
        print(f"  {direction:18s} n={int(r['n'])}  win-rate={r['win_rate']:.1%} "
              f"(95% CI [{r['win_rate_ci_lo']:.1%}, {r['win_rate_ci_hi']:.1%}])  "
              f"paired-t p={r['p_ttest']:.4g}  wilcoxon p={r['p_wilcoxon']:.4g}  "
              f"d={r['cohens_d']:.3f}  -> {verdict}")

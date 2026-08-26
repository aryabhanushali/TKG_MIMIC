"""Cross-family multiple-comparisons correction for the two multi-seed
robustness checks (Section 8.4's plain TKG-Transformer family and Section
8.6's patient-graph family).

Why this script exists: multi_seed_summary.py and
patient_gnn_multi_seed_summary.py each independently Bonferroni-correct
across their own 30 comparisons (5 causes x 3 horizons x 2 baselines). That
is defensible if the two families answer genuinely separate questions asked
in isolation -- but both are the same underlying question ("does this
graph-derived model beat the classical baselines"), asked twice, about two
different models, and the paper's own prose cites findings from BOTH
families side by side as mutually reinforcing evidence (e.g. "stroke at 3
years... has now survived two full rebuilds of this pipeline"). A reviewer
who treats these as one 60-test family and corrects accordingly will reach a
stricter, and different, answer than either per-family Bonferroni threshold
alone -- this script computes that answer directly, plus a
Benjamini-Hochberg FDR alternative (less conservative, still a principled
correction, standard when 30-60 simultaneous tests are run and unadjusted
per-family Bonferroni is judged too strict).

The 45 single-seed DeLong pairwise tests (Section 8.3 / delong_pairwise.csv)
are deliberately EXCLUDED from this correction: the paper already frames
them as exploratory/preliminary ("too little to say much on its own,"
Section 8.3) and defers to the multi-seed checks for any confirmatory claim.
Pooling them in here would both overstate what they were used for and make
an already-conservative correction needlessly stricter. If any single-seed
DeLong result is ever promoted to a headline claim, it must be added to this
correction's pool first.

Run after both multi_seed_summary.py and patient_gnn_multi_seed_summary.py.

Outputs: tkg_output/stats/combined_significance_correction.csv
"""
import os

import pandas as pd

from src.config import OUTPUT_DIR

STATS_DIR = os.path.join(OUTPUT_DIR, "stats")


def run() -> None:
    tgn = pd.read_csv(os.path.join(STATS_DIR, "multi_seed_comparison.csv"))
    pg = pd.read_csv(os.path.join(STATS_DIR, "patient_gnn_multi_seed_comparison.csv"))

    rows = []
    for _, r in tgn.iterrows():
        rows.append(dict(model="TGN-Transformer", cause=r["cause"], horizon_days=r["horizon_days"],
                          comparison="vs_XGB", effect=r["tgn_minus_xgb"], p=r["tgn_vs_xgb_p"]))
        rows.append(dict(model="TGN-Transformer", cause=r["cause"], horizon_days=r["horizon_days"],
                          comparison="vs_Cox", effect=r["tgn_minus_cox"], p=r["tgn_vs_cox_p"]))
    for _, r in pg.iterrows():
        rows.append(dict(model="Patient-graph GNN", cause=r["cause"], horizon_days=r["horizon"],
                          comparison="vs_XGB", effect=r["pg_minus_xgb"], p=r["p_vs_xgb"]))
        rows.append(dict(model="Patient-graph GNN", cause=r["cause"], horizon_days=r["horizon"],
                          comparison="vs_Cox", effect=r["pg_minus_cox"], p=r["p_vs_cox"]))

    df = pd.DataFrame(rows)
    n = len(df)
    alpha_per_family = 0.05 / 30
    alpha_combined = 0.05 / n

    df["sig_per_family_bonferroni"] = df["p"] < alpha_per_family
    df["sig_combined_bonferroni"] = df["p"] < alpha_combined

    # Benjamini-Hochberg FDR across the combined pool, q=0.05
    df_sorted = df.sort_values("p").reset_index(drop=True)
    df_sorted["rank"] = df_sorted.index + 1
    df_sorted["bh_threshold"] = 0.05 * df_sorted["rank"] / n
    below = df_sorted["p"] < df_sorted["bh_threshold"]
    max_sig_rank = df_sorted.loc[below, "rank"].max() if below.any() else 0
    df_sorted["sig_bh_fdr"] = df_sorted["rank"] <= max_sig_rank
    df_sorted = df_sorted.drop(columns=["rank", "bh_threshold"])

    out_path = os.path.join(STATS_DIR, "combined_significance_correction.csv")
    df_sorted.to_csv(out_path, index=False)

    print(f"Combined pool: {n} tests (30 TGN-vs-baselines + 30 PatientGraph-vs-baselines)")
    print(f"  per-family alpha (0.05/30)      = {alpha_per_family:.6f}")
    print(f"  combined alpha   (0.05/{n})      = {alpha_combined:.6f}")
    print(f"\nSignificant under PER-FAMILY correction (current paper framing): "
          f"{df_sorted['sig_per_family_bonferroni'].sum()} / {n}")
    print(f"Significant under COMBINED Bonferroni correction:                 "
          f"{df_sorted['sig_combined_bonferroni'].sum()} / {n}")
    print(f"Significant under combined Benjamini-Hochberg FDR (q=0.05):       "
          f"{df_sorted['sig_bh_fdr'].sum()} / {n}")

    lost = df_sorted[df_sorted["sig_per_family_bonferroni"] & ~df_sorted["sig_combined_bonferroni"]]
    print(f"\nFindings that lose significance under the combined Bonferroni bar "
          f"({len(lost)}):")
    for _, r in lost.sort_values("p").iterrows():
        print(f"  {r['model']:18s} {r['cause']:8s} {r['horizon_days']}d {r['comparison']:8s} "
              f"effect={r['effect']:+.4f}  p={r['p']:.4f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()

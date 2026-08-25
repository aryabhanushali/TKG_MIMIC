"""Mechanism analysis: does task-level risk-factor concentration predict
where the patient-graph model beats the baselines?

Three linked pieces (paper "must-do" items 1, 2, 3):

  1. Risk-factor concentration (entropy) per disease -- computed from a
     PRE-MODEL, cheap diagnostic (univariate single-feature AUROC across the
     same train-only bag-of-codes + value-summary feature space the
     baselines use), NOT from any trained model's attention or feature
     importances. Using a trained model's own attention to explain that
     model's own advantage would be circular; this diagnostic is computable
     before any of the five candidate models are trained at all.
  2. Best-single-feature AUROC per disease -- the same univariate pass,
     reporting the single strongest feature's AUROC directly. A second,
     simpler operationalization of "how concentrated is this task's signal,"
     used to cross-check the entropy metric rather than rely on one
     definition of concentration.
  3. Case-count vs. graph-advantage relationship -- directly tests (and, on
     prior evidence from PAD, is expected to refute) the competing "graphs
     just help more with less data" explanation for the same result pattern.

All three are correlated (Spearman; n=5, explicitly and honestly
underpowered for a formal test) against "graph advantage" = patient-graph
model's 5-seed mean AUROC@3y minus XGBoost's 5-seed mean AUROC@3y, pulled
from the already-completed seed 42-46 runs (no retraining).

Outputs: tkg_output/stats/mechanism_analysis.csv
         tkg_output/figures/fig21_concentration_vs_advantage.png
"""
import os
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, FIGURES_DIR, read_events_table
from src.baseline import _build_bag_of_codes, _build_value_summary_features

MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
CAUSES = ["MI", "Stroke", "HF", "AF", "PAD"]
SEEDS = [42, 43, 44, 45, 46]
MIN_SUPPORT = 20   # a bag-of-codes concept must appear in >= this many train patients
                   # to be included -- guards against unstable AUROC estimates from
                   # rare concepts with near-perfect but meaningless separation


def _load_graph_advantage() -> pd.DataFrame:
    """Patient-graph 5-seed mean AUROC@3y minus XGBoost 5-seed mean AUROC@3y,
    per cause. Reuses already-completed seed runs; no retraining."""
    pg_rows, xgb_rows = [], []
    for s in SEEDS:
        pg_dir = "patient_gnn_survival" if s == 42 else f"patient_gnn_survival_seed{s}"
        df = pd.read_csv(os.path.join(OUTPUT_DIR, pg_dir, "test_metrics.csv"))
        df["seed"] = s
        pg_rows.append(df)
        bl_dir = "baselines_survival" if s == 42 else f"baselines_survival_seed{s}"
        b = pd.read_csv(os.path.join(OUTPUT_DIR, bl_dir, "test_metrics.csv"))
        b = b[b["model"] == "xgb_surv"].copy()
        b["seed"] = s
        xgb_rows.append(b)
    pg_all = pd.concat(pg_rows, ignore_index=True)
    xgb_all = pd.concat(xgb_rows, ignore_index=True)

    rows = []
    for cause in CAUSES:
        pg = pg_all[(pg_all.cause == cause) & (pg_all.horizon_days == 1095)]["auroc"]
        xg = xgb_all[(xgb_all.cause == cause) & (xgb_all.horizon_days == 1095)]["auroc"]
        rows.append(dict(cause=cause, pg_mean_3y=pg.mean(), xgb_mean_3y=xg.mean(),
                          graph_advantage=pg.mean() - xg.mean()))
    return pd.DataFrame(rows)


def _build_train_feature_matrix():
    """Same train-only feature space the baselines use: bag-of-codes +
    value-summary stats, restricted to training patients."""
    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    nodes = pd.read_csv(os.path.join(MODELING_DIR, "node_metadata.csv"))
    events = read_events_table()

    train_ids = set(labels.loc[labels["split"] == "train", "subject_id"])
    train_ids_set = train_ids
    patient_order = labels.loc[labels["subject_id"].isin(train_ids), "subject_id"].tolist()

    concept_counts = (events[events["subject_id"].isin(train_ids)]
                      .groupby("concept_node_idx")["subject_id"].nunique())
    concept_ids = sorted(concept_counts[concept_counts >= MIN_SUPPORT].index.tolist())
    print(f"  concepts with support >= {MIN_SUPPORT}: {len(concept_ids):,}")

    X_bow = _build_bag_of_codes(events, patient_order, concept_ids)
    X_vals, val_col_names = _build_value_summary_features(events, patient_order, nodes, train_ids_set)
    print(f"  bag-of-codes: {X_bow.shape}, value-summary: {X_vals.shape}")

    bow_col_names = [f"BOW_{c}" for c in concept_ids]
    X = sparse.hstack([X_bow, sparse.csr_matrix(X_vals)]).tocsc()
    col_names = bow_col_names + val_col_names

    label_by_sid = dict(zip(labels["subject_id"], labels["endpoint_type"]))
    y_by_cause = {c: np.array([1 if label_by_sid[s] == c else 0 for s in patient_order])
                  for c in CAUSES}
    return X, col_names, y_by_cause


def _univariate_auroc_per_feature(X, y: np.ndarray, col_names, min_pos: int = 10) -> np.ndarray:
    """Single-feature AUROC for every column against a binary outcome.
    Vectorized via rank-sum (Mann-Whitney U) rather than per-column sklearn
    calls, since the feature count (tens of thousands) makes a Python loop
    with sklearn.roc_auc_score too slow to be a "cheap, pre-model" diagnostic
    in practice."""
    from scipy.stats import rankdata
    n_pos = int(y.sum())
    if n_pos < min_pos:
        return np.full(X.shape[1], np.nan)
    Xd = X if isinstance(X, np.ndarray) else X.toarray()
    n = Xd.shape[0]
    aurocs = np.full(Xd.shape[1], np.nan)
    for j in range(Xd.shape[1]):
        col = Xd[:, j]
        if np.allclose(col, col[0]):
            continue
        ranks = rankdata(col)
        sum_ranks_pos = ranks[y == 1].sum()
        n_pos_j, n_neg_j = n_pos, n - n_pos
        auc = (sum_ranks_pos - n_pos_j * (n_pos_j + 1) / 2) / (n_pos_j * n_neg_j)
        aurocs[j] = auc
    return aurocs


def run() -> None:
    os.makedirs(STATS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading graph-advantage numbers (already-completed seed runs)...")
    adv = _load_graph_advantage()
    print(adv.round(4).to_string(index=False))

    print("\nLoading case counts (train split, for item 3)...")
    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    train_counts = labels[labels["split"] == "train"]["endpoint_type"].value_counts()
    adv["n_train_cases"] = adv["cause"].map(train_counts)

    print("\nBuilding train-only feature matrix (bag-of-codes + value-summary)...")
    X, col_names, y_by_cause = _build_train_feature_matrix()

    print("\nComputing univariate (single-feature) AUROC per disease "
          "(pre-model diagnostic -- no trained model's attention or importances used)...")
    concentration_rows = []
    for cause in CAUSES:
        y = y_by_cause[cause]
        print(f"  {cause}: n_pos={y.sum():,} ...")
        aurocs = _univariate_auroc_per_feature(X, y, col_names)
        strength = np.abs(aurocs - 0.5)
        strength = strength[~np.isnan(strength)]
        best_feature_auroc = 0.5 + strength.max() if len(strength) else np.nan

        p = strength / strength.sum() if strength.sum() > 0 else strength
        p = p[p > 0]
        entropy = float(-(p * np.log(p)).sum()) if len(p) else np.nan
        max_entropy = np.log(len(strength)) if len(strength) else np.nan
        norm_entropy = entropy / max_entropy if max_entropy else np.nan

        concentration_rows.append(dict(
            cause=cause, n_features_tested=len(strength),
            best_single_feature_auroc=round(best_feature_auroc, 4),
            entropy=round(entropy, 4), normalized_entropy=round(norm_entropy, 4),
        ))
    conc_df = pd.DataFrame(concentration_rows)
    print("\n" + conc_df.to_string(index=False))

    result = adv.merge(conc_df, on="cause")
    result["concentration_score"] = result["best_single_feature_auroc"] - 0.5  # higher = more concentrated

    out_path = os.path.join(STATS_DIR, "mechanism_analysis.csv")
    result.to_csv(out_path, index=False)

    print("\n=== Correlations across the 5 diseases (n=5 -- explicitly underpowered "
          "for formal significance; reported as a descriptive pattern) ===")
    for x_col, label in [
        ("normalized_entropy", "Item 1: normalized entropy (diffuseness) vs. graph advantage"),
        ("best_single_feature_auroc", "Item 2: best single-feature AUROC (concentration) vs. graph advantage"),
        ("n_train_cases", "Item 3: training case count vs. graph advantage"),
    ]:
        rho, p = stats.spearmanr(result[x_col], result["graph_advantage"])
        print(f"  {label}: Spearman rho={rho:.3f}, p={p:.3f} (n=5)")

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    specs = [
        ("normalized_entropy", "Risk-factor diffuseness (normalized entropy)", axes[0]),
        ("best_single_feature_auroc", "Best single-feature AUROC (concentration)", axes[1]),
        ("n_train_cases", "Training case count", axes[2]),
    ]
    for x_col, xlabel, ax in specs:
        ax.scatter(result[x_col], result["graph_advantage"], s=80, color="#1f77b4", zorder=3)
        for _, r in result.iterrows():
            ax.annotate(r["cause"], (r[x_col], r["graph_advantage"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=10, fontweight="bold")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Patient-graph AUROC minus XGBoost AUROC (3y)")
        ax.grid(alpha=0.3)
    fig.suptitle("Does task-level risk-factor concentration predict graph advantage? "
                 "(n=5 diseases; descriptive, not a powered test)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = os.path.join(FIGURES_DIR, "fig21_concentration_vs_advantage.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved:\n  {out_path}\n  {fig_path}")


if __name__ == "__main__":
    run()

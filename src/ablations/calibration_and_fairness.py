"""Two audit-flagged gaps closed in one pass: calibration (is a predicted
30% risk actually observed ~30% of the time?) and a basic fairness/subgroup
breakdown (does discrimination hold up equally across sex, age, and race?).
Neither was previously computed anywhere in this study -- the entire
evaluation was AUROC-only, which measures ranking, not whether predicted
probabilities mean what they say, and never checked whether performance is
uniform across demographic subgroups.

Calibration: DeepHit-style models (TKG-Transformer, patient-graph GNN)
output a genuine cumulative-incidence probability per cause/horizon, so a
standard reliability diagram (predicted-probability decile vs. observed
event rate) and Brier score apply directly. Cox and XGBoost-Survival output
a relative risk SCORE, not a probability (`sksurv`/`xgboost`'s survival:cox
objective is a log-hazard-proportional score with no fixed scale) -- for
these, a risk-decile stratification (does a higher risk decile correspond
to a higher observed event rate, monotonically) is the appropriate analogue
and is what's reported here instead of a Brier score.

Fairness: AUROC at the 3-year horizon, broken down by sex (already a model
feature), age tertile (computed on training data only, applied to all
splits), and self-reported race (from admissions.csv.gz, not currently used
by any model). Race buckets with very few disease-positive test patients are
reported with their n so a reader can judge reliability directly rather than
the analysis silently going ahead with an uninterpretable subgroup.

Test data is read once, at the end, exactly like every other analysis in
this study -- age tertile cutpoints and race bucketing use only definitions
computable without looking at outcomes.

Output: tkg_output/stats/calibration.csv, tkg_output/stats/fairness.csv
        tkg_output/figures/fig25_calibration.png
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.config import OUTPUT_DIR, FIGURES_DIR
from src.tgn_survival import CAUSES, NUM_CAUSES, NUM_TIME_BINS, HORIZON_DAYS, _make_time_bins
from src.ablations.patient_graph_gnn import PatientConceptGNN, _prepare_patient_graph_data, MODEL_DIR as PG_MODEL_DIR

STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
TGN_DIR = os.path.join(OUTPUT_DIR, "tgn_survival")
BASELINE_DIR = os.path.join(OUTPUT_DIR, "baselines_survival")
H3Y = 1095


# --------------------------------------------------------------------------- #
# Load per-patient probabilities / scores at the 3-year horizon              #
# --------------------------------------------------------------------------- #
def _cif_at_horizon(cif: np.ndarray, time_edges: np.ndarray, cause_idx: int, horizon_days: int) -> np.ndarray:
    """cif: (n_patients, n_causes, n_time_bins) cumulative incidence.
    Returns P(event of this cause by horizon_days) per patient."""
    bin_idx = int(np.searchsorted(time_edges, horizon_days, side="right") - 1)
    bin_idx = max(0, min(bin_idx, cif.shape[-1] - 1))
    return cif[:, cause_idx, bin_idx]


def _load_patient_graph_probs(labels_df: pd.DataFrame) -> dict:
    """Re-run inference (no retraining) from the saved canonical checkpoint
    to get per-patient CIF probabilities -- patient_graph_gnn.py only ever
    saved aggregate test_metrics.csv, not per-patient predictions."""
    ckpt_path = os.path.join(PG_MODEL_DIR, "best_model.pt")
    d = _prepare_patient_graph_data()
    train_sids = d["splits"]["train"]
    train_durations = labels_df.loc[
        labels_df["subject_id"].isin(train_sids), "time_to_event_days"
    ].to_numpy(dtype=np.float32)
    time_edges = _make_time_bins(train_durations, NUM_TIME_BINS)

    model = PatientConceptGNN(
        n_concepts=d["n_concepts"], n_patients=d["n_patients"], n_static=d["n_static"],
        n_relations=d["n_relations"], edge_index=d["edge_index"], edge_type=d["edge_type"],
        n_classes=NUM_CAUSES * NUM_TIME_BINS,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()
    with torch.no_grad():
        logits_flat = model(d["static_arr"])
        logits = logits_flat.view(-1, NUM_CAUSES, NUM_TIME_BINS)
        probs = F.softmax(logits.reshape(logits.size(0), -1), dim=-1).view_as(logits)
        cif = torch.cumsum(probs, dim=-1).numpy()

    test_sids = d["splits"]["test"]
    test_pos = np.array([d["pid_to_pos"][s] for s in test_sids])
    cif_test = cif[test_pos]
    out = {}
    for i, cause in enumerate(CAUSES):
        out[cause] = pd.DataFrame({
            "subject_id": test_sids,
            "prob": _cif_at_horizon(cif_test, time_edges, i, H3Y),
        })
    return out


def _load_tgn_probs() -> dict:
    preds = pd.read_csv(os.path.join(TGN_DIR, "predictions_test.csv"))
    out = {}
    for cause in CAUSES:
        col = [c for c in preds.columns if cause.lower() in c.lower() and "1095" in c]
        if not col:
            # fall back: try a cause_h1095-style or per-horizon-prob naming; skip if truly absent
            continue
        out[cause] = preds[["subject_id", col[0]]].rename(columns={col[0]: "prob"})
    return out


def _load_baseline_scores() -> dict:
    preds = pd.read_csv(os.path.join(BASELINE_DIR, "predictions_test.csv"))
    out = {"cox": {}, "xgb_surv": {}}
    for cause in CAUSES:
        out["cox"][cause] = preds[["subject_id", f"cox_risk_{cause}"]].rename(columns={f"cox_risk_{cause}": "score"})
        out["xgb_surv"][cause] = preds[["subject_id", f"xgb_surv_risk_{cause}"]].rename(columns={f"xgb_surv_risk_{cause}": "score"})
    return out


def _binary_label(labels_df: pd.DataFrame, cause: str, horizon_days: int) -> pd.DataFrame:
    """Same competing-risks rule used everywhere else in this study."""
    durs = labels_df["time_to_event_days"].to_numpy(dtype=float)
    evts = labels_df["endpoint_type"].to_numpy()
    keep_pos = (evts == cause) & (durs <= horizon_days)
    survived = durs >= horizon_days
    competing = (durs < horizon_days) & (evts != cause) & (evts != "censored")
    keep_neg = survived | competing
    mask = keep_pos | keep_neg
    out = labels_df.loc[mask, ["subject_id"]].copy()
    out["y"] = keep_pos[mask].astype(int)
    return out


# --------------------------------------------------------------------------- #
# Calibration                                                                 #
# --------------------------------------------------------------------------- #
def run_calibration(labels_df: pd.DataFrame) -> pd.DataFrame:
    print("=== Calibration @ 3-year horizon ===\n")
    prob_sources = {"patient_graph": _load_patient_graph_probs(labels_df)}
    tgn_probs = _load_tgn_probs()
    if tgn_probs:
        prob_sources["tgn"] = tgn_probs
    score_sources = _load_baseline_scores()

    rows = []
    fig, axes = plt.subplots(1, len(CAUSES), figsize=(4 * len(CAUSES), 4.5), sharey=True)
    for ax, cause in zip(axes, CAUSES):
        lab = _binary_label(labels_df, cause, H3Y)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)

        for name, sources in prob_sources.items():
            if cause not in sources:
                continue
            df = lab.merge(sources[cause], on="subject_id", how="inner")
            if df["y"].nunique() < 2 or len(df) < 20:
                continue
            df["decile"] = pd.qcut(df["prob"], q=min(10, df["prob"].nunique()), duplicates="drop")
            grp = df.groupby("decile", observed=True).agg(
                mean_pred=("prob", "mean"), obs_rate=("y", "mean"), n=("y", "size"))
            brier = brier_score_loss(df["y"], df["prob"])
            ax.plot(grp["mean_pred"], grp["obs_rate"], marker="o", label=f"{name} (Brier={brier:.4f})")
            rows.append(dict(cause=cause, model=name, metric="brier_score", value=brier, n=len(df)))
            for _, r in grp.iterrows():
                rows.append(dict(cause=cause, model=name, metric="decile_point",
                                  value=r["obs_rate"], mean_pred=r["mean_pred"], n=int(r["n"])))

        for name, sources in score_sources.items():
            if cause not in sources:
                continue
            df = lab.merge(sources[cause], on="subject_id", how="inner")
            if df["y"].nunique() < 2 or len(df) < 20:
                continue
            df["decile"] = pd.qcut(df["score"].rank(method="first"), q=10, duplicates="drop")
            grp = df.groupby("decile", observed=True).agg(obs_rate=("y", "mean"), n=("y", "size")).reset_index(drop=True)
            monotonic = bool(grp["obs_rate"].is_monotonic_increasing)
            print(f"  {cause:8s} {name:10s} risk-decile obs. rate monotonically increasing: {monotonic}  "
                  f"(deciles: {grp['obs_rate'].round(3).tolist()})")
            rows.append(dict(cause=cause, model=name, metric="risk_decile_monotonic", value=float(monotonic), n=len(df)))

        ax.set_title(cause, fontweight="bold")
        ax.set_xlabel("Mean predicted probability")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Observed event rate")
    fig.suptitle("Calibration @ 3-year horizon (probability-output models only; "
                  "Cox/XGBoost reported separately as risk-decile monotonicity)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = os.path.join(FIGURES_DIR, "fig25_calibration.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    result = pd.DataFrame(rows)
    out_path = os.path.join(STATS_DIR, "calibration.csv")
    result.to_csv(out_path, index=False)
    print(f"\nBrier scores (lower is better, 0=perfect, 0.25=uninformative-for-a-~2-3%-base-rate task):")
    print(result[result.metric == "brier_score"].pivot(index="cause", columns="model", values="value").round(4))
    print(f"\nSaved: {out_path}\n  {fig_path}")
    return result


# --------------------------------------------------------------------------- #
# Fairness / subgroup breakdown                                               #
# --------------------------------------------------------------------------- #
def _race_bucket(race: str) -> str:
    if pd.isna(race):
        return "Unknown"
    r = race.upper()
    if "WHITE" in r:
        return "White"
    if "BLACK" in r:
        return "Black"
    if "HISPANIC" in r or "LATINO" in r:
        return "Hispanic/Latino"
    if "ASIAN" in r:
        return "Asian"
    if "UNKNOWN" in r or "UNABLE" in r or "DECLINED" in r:
        return "Unknown"
    return "Other"


def run_fairness(labels_df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Fairness / subgroup breakdown @ 3-year horizon ===\n")
    static = pd.read_csv(os.path.join(MODELING_DIR, "static_features.csv"))
    cohort = pd.read_csv(os.path.join(OUTPUT_DIR, "cohort.csv"))

    adm = pd.read_csv(os.path.join(os.path.expanduser("~/Desktop/TKG_MIMIC/mimic_data"), "admissions.csv.gz"),
                       usecols=["subject_id", "race"])
    top_race = adm.groupby("subject_id")["race"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
    race_df = top_race.rename("race").reset_index()
    race_df["race_bucket"] = race_df["race"].apply(_race_bucket)

    # Age tertiles from TRAINING patients only (a decision rule fixed before
    # looking at any subgroup's outcomes, same discipline as every other
    # train-only statistic in this study).
    train_ids = set(labels_df.loc[labels_df["split"] == "train", "subject_id"])
    train_age = cohort.loc[cohort["subject_id"].isin(train_ids), "age_at_index"]
    q1, q2 = train_age.quantile([1 / 3, 2 / 3])
    print(f"Age tertile cutpoints (from training patients only): <{q1:.0f}, {q1:.0f}-{q2:.0f}, >{q2:.0f}\n")

    demo = cohort[["subject_id", "age_at_index"]].merge(
        static[["subject_id", "female"]], on="subject_id", how="left"
    ).merge(race_df[["subject_id", "race_bucket"]], on="subject_id", how="left")
    demo["age_tertile"] = pd.cut(demo["age_at_index"], bins=[-np.inf, q1, q2, np.inf],
                                  labels=["younger", "middle", "older"])
    demo["sex"] = np.where(demo["female"] > 0, "female", "male")

    prob_sources = {"patient_graph": _load_patient_graph_probs(labels_df)}
    tgn_probs = _load_tgn_probs()
    if tgn_probs:
        prob_sources["tgn"] = tgn_probs
    score_sources = _load_baseline_scores()
    all_sources = {**prob_sources, **score_sources}

    rows = []
    for cause in CAUSES:
        lab = _binary_label(labels_df, cause, H3Y).merge(demo, on="subject_id", how="left")
        for name, sources in all_sources.items():
            if cause not in sources:
                continue
            score_col = "prob" if "prob" in sources[cause].columns else "score"
            df = lab.merge(sources[cause], on="subject_id", how="inner")
            for dim, groups in [("sex", ["female", "male"]),
                                 ("age_tertile", ["younger", "middle", "older"]),
                                 ("race_bucket", ["White", "Black", "Hispanic/Latino", "Asian", "Other", "Unknown"])]:
                for g in groups:
                    sub = df[df[dim] == g]
                    n_pos = int(sub["y"].sum())
                    n = len(sub)
                    if n_pos < 2 or n_pos == n:
                        rows.append(dict(cause=cause, model=name, dimension=dim, subgroup=g,
                                          auroc=np.nan, n=n, n_pos=n_pos))
                        continue
                    auroc = roc_auc_score(sub["y"], sub[score_col])
                    rows.append(dict(cause=cause, model=name, dimension=dim, subgroup=g,
                                      auroc=auroc, n=n, n_pos=n_pos))

    result = pd.DataFrame(rows)
    out_path = os.path.join(STATS_DIR, "fairness.csv")
    result.to_csv(out_path, index=False)

    print("Sex breakdown (AUROC @ 3y), tuned models shown where available:\n")
    sex_view = result[result.dimension == "sex"].pivot_table(
        index=["cause", "model"], columns="subgroup", values="auroc")
    print(sex_view.round(3).to_string())

    print("\nAge tertile breakdown:\n")
    age_view = result[result.dimension == "age_tertile"].pivot_table(
        index=["cause", "model"], columns="subgroup", values="auroc")
    print(age_view.round(3).to_string())

    print("\nRace breakdown (n_pos shown for the two largest models only, White/Black -- "
          "smaller buckets are typically too small for a reliable AUROC, reported anyway for transparency):\n")
    race_n = result[(result.dimension == "race_bucket") & (result.model == "patient_graph")][
        ["cause", "subgroup", "n_pos", "n"]]
    print(race_n.to_string(index=False))

    print(f"\nSaved: {out_path}")
    return result


if __name__ == "__main__":
    labels_df = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    run_calibration(labels_df)
    run_fairness(labels_df)

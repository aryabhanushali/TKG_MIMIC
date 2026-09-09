"""Cross-checks every "beats Cox" claim in this study (TKG-Transformer and
the patient-graph model) against a stricter test than the one used to
report significance elsewhere.

The existing protocol (multi_seed_summary.py, patient_gnn_multi_seed_summary.py)
tests "vs Cox" with a one-sample t-test: 5 seeds' AUROC values against Cox's
single fixed AUROC. That question is "is Cox's number outside the range this
model's own seed-to-seed training noise would produce" -- it never looks at
whether the two models actually disagree on the same patients, and it
ignores Cox's own sampling uncertainty entirely.

This script computes the stricter, paired DeLong test (properly correlated
on the same test patients) for EVERY seed of EVERY "vs Cox" comparison
already claimed in this study -- not just one seed, so a finding can be
judged by how many of its 5 independently-trained instances actually beat
Cox on the same patients, not just whether the seed-mean sits far from
Cox's point estimate.

A finding is reported ROBUST only if BOTH hold:
  1. The existing one-sample-t-test claim is Bonferroni-significant
     (already computed, from multi_seed_comparison.csv /
     patient_gnn_multi_seed_comparison.csv).
  2. DeLong finds a significant, same-direction difference in at least
     4 of 5 seeds (uncorrected p<0.05 per seed -- requiring 4/5 independent
     replications is itself a strong bar, deliberately not stacking a
     second Bonferroni correction on top).

Inputs (no retraining -- pure inference from already-saved checkpoints):
  tkg_output/patient_gnn_survival[_seed43..46]/best_model.pt
  tkg_output/tgn_survival[_seed43..46]/predictions_test.csv
  tkg_output/baselines_survival/predictions_test.csv  (cox_risk_*)
  tkg_output/stats/multi_seed_comparison.csv
  tkg_output/stats/patient_gnn_multi_seed_comparison.csv

Output: tkg_output/stats/delong_robustness_audit.csv
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import OUTPUT_DIR
from src.tgn_survival import CAUSES, NUM_CAUSES, NUM_TIME_BINS, HORIZON_DAYS, _make_time_bins
from src.ablations.patient_graph_gnn import PatientConceptGNN, _prepare_patient_graph_data
from src.evaluate_stats import delong_test, _labels_for

STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
SEEDS = [42, 43, 44, 45, 46]
ALPHA_PER_FAMILY = 0.05 / 30   # matches the existing per-family Bonferroni bar


def _pg_dir(seed: int) -> str:
    return "patient_gnn_survival" if seed == 42 else f"patient_gnn_survival_seed{seed}"


def _tgn_dir(seed: int) -> str:
    return "tgn_survival" if seed == 42 else f"tgn_survival_seed{seed}"


def _cif_at_horizon(cif: np.ndarray, time_edges: np.ndarray, cause_idx: int, h: int) -> np.ndarray:
    bin_idx = int(np.searchsorted(time_edges, h, side="right") - 1)
    bin_idx = max(0, min(bin_idx, cif.shape[-1] - 1))
    return cif[:, cause_idx, bin_idx]


def _load_pg_probs_all_seeds() -> pd.DataFrame:
    """Pure inference from each seed's already-trained checkpoint -- no
    retraining. The underlying graph/splits are seed-independent (only the
    trained weights differ), so _prepare_patient_graph_data() is built once
    and reused for all 5 checkpoints."""
    print("Building patient-graph data (shared across all 5 seed checkpoints)...")
    d = _prepare_patient_graph_data()
    labels_df = pd.read_csv(os.path.join(OUTPUT_DIR, "modeling", "labels.csv"))
    train_sids = d["splits"]["train"]
    train_durations = labels_df.loc[
        labels_df["subject_id"].isin(train_sids), "time_to_event_days"
    ].to_numpy(dtype=np.float32)
    time_edges = _make_time_bins(train_durations, NUM_TIME_BINS)
    test_sids = d["splits"]["test"]
    test_pos = np.array([d["pid_to_pos"][s] for s in test_sids])

    merged = pd.DataFrame({"subject_id": test_sids})
    for seed in SEEDS:
        print(f"  seed {seed}...")
        ckpt = os.path.join(OUTPUT_DIR, _pg_dir(seed), "best_model.pt")
        model = PatientConceptGNN(
            n_concepts=d["n_concepts"], n_patients=d["n_patients"], n_static=d["n_static"],
            n_relations=d["n_relations"], edge_index=d["edge_index"], edge_type=d["edge_type"],
            n_classes=NUM_CAUSES * NUM_TIME_BINS,
        )
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        model.eval()
        with torch.no_grad():
            logits_flat = model(d["static_arr"])
            logits = logits_flat.view(-1, NUM_CAUSES, NUM_TIME_BINS)
            probs = F.softmax(logits.reshape(logits.size(0), -1), dim=-1).view_as(logits)
            cif = torch.cumsum(probs, dim=-1).numpy()
        cif_test = cif[test_pos]
        for i, cause in enumerate(CAUSES):
            for h in HORIZON_DAYS:
                merged[f"pg_cif_{cause}_at_{h}d_seed{seed}"] = _cif_at_horizon(cif_test, time_edges, i, h)
    return merged


def run() -> None:
    labels = pd.read_csv(os.path.join(OUTPUT_DIR, "modeling", "labels.csv"))
    test = labels[labels["split"] == "test"][["subject_id", "endpoint_type", "time_to_event_days"]]
    cox_preds = pd.read_csv(os.path.join(OUTPUT_DIR, "baselines_survival", "predictions_test.csv"))

    pg_preds = _load_pg_probs_all_seeds()

    df = test.merge(cox_preds, on="subject_id").merge(pg_preds, on="subject_id")
    for seed in SEEDS:
        tgn_df = pd.read_csv(os.path.join(OUTPUT_DIR, _tgn_dir(seed), "predictions_test.csv"))
        keep = ["subject_id"] + [c for c in tgn_df.columns if c.startswith("cif_")]
        tgn_df = tgn_df[keep].rename(columns={c: f"{c}_seed{seed}" for c in keep if c != "subject_id"})
        df = df.merge(tgn_df, on="subject_id", how="inner")

    evts = df["endpoint_type"].to_numpy()
    durs = df["time_to_event_days"].to_numpy(dtype=float)

    print(f"\nRunning per-seed DeLong tests ({len(SEEDS)} seeds x 5 causes x 3 horizons x "
          f"2 models = {len(SEEDS)*5*3*2} tests)...\n")
    rows = []
    for cause in CAUSES:
        for h in HORIZON_DAYS:
            y, mask = _labels_for(cause, h, evts, durs)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            cox_s = df.loc[mask, f"cox_risk_{cause}"].to_numpy(float)
            for seed in SEEDS:
                pg_s = df.loc[mask, f"pg_cif_{cause}_at_{h}d_seed{seed}"].to_numpy(float)
                pg_auc, cox_auc, p = delong_test(y, pg_s, cox_s)
                rows.append(dict(model="PatientGraph", cause=cause, horizon_days=h, seed=seed,
                                  model_auc=pg_auc, cox_auc=cox_auc, p_delong=p))

                tgn_col = f"cif_{cause}_at_{h}d_seed{seed}"
                if tgn_col in df.columns:
                    tgn_s = df.loc[mask, tgn_col].to_numpy(float)
                    tgn_auc, cox_auc2, p2 = delong_test(y, tgn_s, cox_s)
                    rows.append(dict(model="TGN-Transformer", cause=cause, horizon_days=h, seed=seed,
                                      model_auc=tgn_auc, cox_auc=cox_auc2, p_delong=p2))

    result = pd.DataFrame(rows)
    result["delong_sig_uncorrected"] = result["p_delong"] < 0.05
    result["delong_sig_bonferroni"] = result["p_delong"] < ALPHA_PER_FAMILY
    result["model_beats_cox"] = result["model_auc"] > result["cox_auc"]

    summary = result.groupby(["model", "cause", "horizon_days"]).agg(
        n_seeds_beat_cox=("model_beats_cox", "sum"),
        n_seeds_delong_sig_uncorrected=("delong_sig_uncorrected", "sum"),
        n_seeds_delong_sig_bonferroni=("delong_sig_bonferroni", "sum"),
        median_p_delong=("p_delong", "median"),
        mean_model_auc=("model_auc", "mean"),
        cox_auc=("cox_auc", "first"),
    ).reset_index()
    summary["delong_robust"] = (
        (summary["n_seeds_delong_sig_uncorrected"] >= 4)
        & (summary["n_seeds_beat_cox"].isin([0, 5]))   # all 5 agree on direction
    )

    tgn_ttest = pd.read_csv(os.path.join(STATS_DIR, "multi_seed_comparison.csv"))
    pg_ttest = pd.read_csv(os.path.join(STATS_DIR, "patient_gnn_multi_seed_comparison.csv"))
    ttest_lookup = {}
    for _, r in tgn_ttest.iterrows():
        ttest_lookup[("TGN-Transformer", r["cause"], r["horizon_days"])] = bool(r["tgn_vs_cox_sig_bonferroni"])
    for _, r in pg_ttest.iterrows():
        ttest_lookup[("PatientGraph", r["cause"], r["horizon"])] = bool(r["sig_vs_cox"])

    summary["ttest_sig_bonferroni"] = summary.apply(
        lambda r: ttest_lookup.get((r["model"], r["cause"], r["horizon_days"])), axis=1)
    summary["ROBUST_both_tests_agree"] = summary["ttest_sig_bonferroni"].fillna(False) & summary["delong_robust"]

    out_path = os.path.join(STATS_DIR, "delong_robustness_audit.csv")
    summary.to_csv(out_path, index=False)
    result.to_csv(os.path.join(STATS_DIR, "delong_robustness_audit_per_seed.csv"), index=False)

    print("=== ROBUSTNESS SUMMARY (vs Cox claims only; XGB comparisons unaffected, "
          "they already use a fair 2-distribution Welch test) ===\n")
    cols = ["model", "cause", "horizon_days", "mean_model_auc", "cox_auc",
            "n_seeds_beat_cox", "n_seeds_delong_sig_uncorrected", "ttest_sig_bonferroni", "ROBUST_both_tests_agree"]
    print(summary[cols].to_string(index=False))

    n_robust = int(summary["ROBUST_both_tests_agree"].sum())
    n_ttest_only = int((summary["ttest_sig_bonferroni"].fillna(False) & ~summary["delong_robust"]).sum())
    print(f"\n{n_robust} cells are robust under BOTH tests.")
    print(f"{n_ttest_only} cells are t-test-significant but do NOT survive the per-seed DeLong check "
          f"(claimed significant by the existing protocol, not confirmed by the stricter one).")
    print(f"\nSaved:\n  {out_path}\n  "
          f"{os.path.join(STATS_DIR, 'delong_robustness_audit_per_seed.csv')}")


if __name__ == "__main__":
    run()

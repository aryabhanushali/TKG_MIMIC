"""Materialise EHRSHOT K-fold out-of-fold (OOF) predictions as one directory
that looks like a single-test-set MIMIC_OUTPUT_DIR.

Every patient is in exactly one CV test fold, so concatenating the fold test
predictions gives one prediction per patient for the full cohort. Writing
them in the standard layout lets the analysis scripts written for MIMIC's
fixed test set run unchanged on EHRSHOT:

  src/ablations/discussion_findings.py   subgroup / complementarity scans
  src/ablations/paired_robustness.py     DeLong + multi-seed robustness bar
  src/ablations/case_count_analysis.py   per-cell margin vs. positive count

Layout written to <cv_root>/pooled/:
  cohort.csv                     symlink to the main EHRSHOT cohort
  modeling/labels.csv            all patients, split='test', plus 'cv_fold'
  modeling/static_features.csv   fold-invariant (checked), copied from fold 0
  modeling/events.parquet        fold-invariant columns, symlink to fold 0
  modeling/node_metadata.csv     symlink to fold 0
  baselines_survival[_seedN]/predictions_test.csv   Cox / XGBoost
  tgn_survival[_seedN]/predictions_test.csv         TKG-Transformer
  patient_gnn_survival[_seedN]/predictions_test.csv patient-graph model
  mimic_transfer/predictions_test.csv                frozen MIMIC Cox / XGBoost (if
                                                     transfer_eval.py saved per-patient scores)

Scores are percentile-ranked within each fold before pooling (as in
cv_summary.py): independently fit fold models put risk on different scales,
and ranking removes that without changing any within-fold ordering. So the
pooled AUROCs here equal cv_summary.csv's pooled_auroc.

Usage: python -m src.ehrshot.cv_pool <cv_root> [--folds 5]
"""
import argparse
import os
import shutil

import pandas as pd

from src.ehrshot.cv_summary import _seeds, _pred_path

STEMS = ["baselines_survival", "tgn_survival", "patient_gnn_survival"]


def _link(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


# OMOP "Race" vocabulary concept codes (CDC race categories) as they appear
# in EHRSHOT MEDS ("Race/5"); Hispanic ethnicity takes precedence, matching
# how MIMIC's single race field records Hispanic/Latino.
OMOP_RACE = {"1": "Other", "2": "Asian", "3": "Black", "4": "Other", "5": "White"}


def _write_race(out: str, subject_ids) -> None:
    """race.csv (subject_id, race_bucket) for discussion_findings' race subgroups."""
    try:
        from src.ehrshot import data_ehrshot as D
        r = D.load_rows(["Race/", "Ethnicity/"], [])
    except Exception as e:  # EHRSHOT_DATA_DIR not set / data not mounted
        print(f"  race.csv skipped ({type(e).__name__}: {e})")
        return
    r = r[r.subject_id.isin(set(subject_ids))]
    hisp = set(r.loc[r.code == "Ethnicity/Hispanic", "subject_id"])
    race = (r[r.code.str.startswith("Race/")].groupby("subject_id").code.first()
            .str.split("/").str[1].map(OMOP_RACE))
    df = pd.DataFrame({"subject_id": list(subject_ids)})
    df["race_bucket"] = df.subject_id.map(race).fillna("Unknown")
    df.loc[df.subject_id.isin(hisp), "race_bucket"] = "Hispanic/Latino"
    df.to_csv(os.path.join(out, "race.csv"), index=False)
    print("  race buckets:", df.race_bucket.value_counts().to_dict())


def run(cv_root: str, n_folds: int) -> str:
    folds = [os.path.join(cv_root, f"fold{k}") for k in range(n_folds)]
    out = os.path.join(cv_root, "pooled")
    mod = os.path.join(out, "modeling")
    os.makedirs(mod, exist_ok=True)

    labels = []
    for k, f in enumerate(folds):
        lab = pd.read_csv(os.path.join(f, "modeling", "labels.csv"))
        labels.append(lab[lab["split"] == "test"].assign(cv_fold=k))
    labels = pd.concat(labels, ignore_index=True)
    assert labels["subject_id"].is_unique, "a patient is in more than one test fold"
    labels.to_csv(os.path.join(mod, "labels.csv"), index=False)

    st0 = pd.read_csv(os.path.join(folds[0], "modeling", "static_features.csv"))
    for f in folds[1:]:
        st = pd.read_csv(os.path.join(f, "modeling", "static_features.csv"))
        key = ["subject_id"]
        assert st.sort_values(key).reset_index(drop=True).equals(
            st0.sort_values(key).reset_index(drop=True)), f"static features differ in {f}"
    shutil.copy(os.path.join(folds[0], "modeling", "static_features.csv"), mod)
    for name in ("events.parquet", "node_metadata.csv"):
        _link(os.path.join(folds[0], "modeling", name), os.path.join(mod, name))
    _link(os.path.join(folds[0], "cohort.csv"), os.path.join(out, "cohort.csv"))

    seeds = sorted(set.intersection(*[set(_seeds(f)) for f in folds]))
    written = 0
    for stem in STEMS:
        for seed in seeds:
            paths = [_pred_path(f, stem, seed) for f in folds]
            if not all(os.path.exists(p) for p in paths):
                print(f"  skip {stem} seed {seed}: missing in some fold")
                continue
            parts = []
            for p in paths:
                pr = pd.read_csv(p)
                score_cols = [c for c in pr.columns if c not in ("subject_id", "endpoint_true", "duration_days")]
                pr[score_cols] = pr[score_cols].rank(pct=True)
                parts.append(pr[["subject_id"] + score_cols])
            d = os.path.join(out, stem if seed == 42 else f"{stem}_seed{seed}")
            os.makedirs(d, exist_ok=True)
            pd.concat(parts, ignore_index=True).to_csv(os.path.join(d, "predictions_test.csv"), index=False)
            written += 1
    # Frozen MIMIC -> EHRSHOT transfer models: one fit, scored on everyone, so
    # no fold ranking is needed. Written as mimic_transfer/predictions_test.csv.
    tp = os.path.join(os.path.dirname(os.path.abspath(cv_root)), "transfer_from_mimic", "predictions_all.csv")
    if os.path.exists(tp):
        d = os.path.join(out, "mimic_transfer")
        os.makedirs(d, exist_ok=True)
        t = pd.read_csv(tp)
        t[t.subject_id.isin(labels.subject_id)].to_csv(os.path.join(d, "predictions_test.csv"), index=False)
        written += 1
    _write_race(out, labels.subject_id)
    print(f"Pooled {len(labels):,} OOF patients, {written} prediction sets (seeds {seeds}) -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cv_root")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    run(a.cv_root, a.folds)

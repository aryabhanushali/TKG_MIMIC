"""Build the cardiometabolic -> circulatory endpoint cohort from EHRSHOT.

Mirrors src/cohort.py step by step and writes cohort.csv / cohort_cascade.csv
with identical columns into src.config.OUTPUT_DIR. Differences forced by the
data (EHRSHOT is a longitudinal Stanford EHR, not a set of ICU admissions):

* "Admission" -> encounter (visit_id). The index date is the time of the
  earliest adult cardiometabolic ICD code in any setting. MIMIC's index is
  always a hospital admission, because MIMIC only contains admissions.
* Incident endpoint. MIMIC requires the endpoint code in the principal
  position (seq_num == 1) of a later admission. EHRSHOT has no principal-dx
  flag, so an endpoint is the first post-index endpoint ICD code that is
  either on an inpatient encounter (Stanford class "Inpatient", Visit/ERIP or
  ICU) or a hospital-account admission diagnosis. Outpatient / problem-list
  mentions don't count.
* Washout: any ENDPOINT_HISTORY code, any setting, at or before the index
  date. It only looks at rows at or before index. (The eICU port also scanned
  post-index stays, which washed out the true endpoints themselves.)
* ">= 2 adult admissions" -> ">= 2 adult encounters".
* Censoring: max(death, last recorded event) instead of max(dod, last admission).
* had_icu_stay: an ICU encounter (Visit/OMOP4822460) starting before index.
"""
import os

import numpy as np
import pandas as pd

from src.cohort import _classify_dx, _compute_cci
from src.ehrshot.config_ehrshot import (
    OUTPUT_DIR, MIN_FOLLOWUP_DAYS,
    CARDIOMETA_ICD10, CARDIOMETA_ICD9,
    ENDPOINTS_ICD10, ENDPOINTS_ICD9,
    ENDPOINT_HISTORY_ICD10, ENDPOINT_HISTORY_ICD9,
)
from src.ehrshot import data_ehrshot as D

CAUSE_ORDER = list(ENDPOINTS_ICD10.keys())   # MI, Stroke, HF, AF, PAD


def _encounter_key(dx: pd.DataFrame) -> pd.Series:
    """visit_id, or the calendar day for the ~10% of dx rows without one."""
    return dx["hadm_id"].fillna("DAY_" + dx["time"].dt.strftime("%Y-%m-%d")).astype(str)


def build_cohort() -> pd.DataFrame:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading EHRSHOT (meds_omop_ehrshot)...")
    demo = D.load_demographics()
    visits = D.visit_table()
    dx = D.load_diagnoses()
    n_total_patients = int(len(D.all_subjects()))
    print(f"  patients: {n_total_patients:,}, encounters: {len(visits):,}, "
          f"ICD dx rows: {len(dx):,}")

    # --- adults ------------------------------------------------------------
    birth = demo.set_index("subject_id")["birth"]
    dx["age"] = (dx["time"] - dx["subject_id"].map(birth)).dt.days / 365.25
    visits = visits.copy()
    visits["age"] = (visits["time"] - visits["subject_id"].map(birth)).dt.days / 365.25
    adult_visits = visits[visits["age"] >= 18]
    adult_dx = dx[dx["age"] >= 18].copy()
    adult_dx["enc"] = _encounter_key(adult_dx)
    n_after_adult = int(pd.Index(adult_visits["subject_id"]).union(
        pd.Index(adult_dx["subject_id"])).nunique())
    print(f"  patients with >=1 encounter at age >= 18: {n_after_adult:,}")

    # --- index encounter -----------------------------------------------------
    print("\nIdentifying cardiometabolic index encounters...")
    cm = _classify_dx(adult_dx, CARDIOMETA_ICD10, CARDIOMETA_ICD9)
    cm = cm.sort_values(["subject_id", "time"])
    index_rows = cm.groupby("subject_id").first().reset_index()
    cats = (cm.merge(index_rows[["subject_id", "enc"]], on=["subject_id", "enc"])
            .groupby("subject_id")["category"].agg(lambda s: sorted(set(s))))
    index_rows = index_rows.merge(cats.rename("cats").reset_index(), on="subject_id")
    index_rows["cardiometa_types"] = index_rows["cats"].apply(",".join)
    index_rows["num_cardiometa_conditions"] = index_rows["cats"].apply(len)
    n_with_cm = len(index_rows)
    print(f"  patients with any cardiometabolic dx: {n_with_cm:,}")
    idx_time = index_rows.set_index("subject_id")["time"]

    # --- incident endpoints ------------------------------------------------------
    print("\nIdentifying endpoint encounters (post-index, inpatient / admission dx)...")
    ep_src = dx[dx["on_inpatient_visit"] | dx["is_admit_dx"]]
    ep = _classify_dx(ep_src, ENDPOINTS_ICD10, ENDPOINTS_ICD9)
    ep = ep[ep["subject_id"].isin(idx_time.index)]
    # MIMIC: the endpoint is a *later, separate* admission (admittime > index
    # admittime). Here: a different encounter on a later calendar day, so an
    # endpoint code charted during the index encounter itself (or its same-day
    # work-up) never counts as an incident event.
    ep["enc"] = _encounter_key(ep)
    idx_enc = index_rows.set_index("subject_id")["enc"]
    ep = ep[(ep["time"].dt.normalize() > ep["subject_id"].map(idx_time).dt.normalize())
            & (ep["enc"] != ep["subject_id"].map(idx_enc))]
    ep["cause_rank"] = ep["category"].map({c: i for i, c in enumerate(CAUSE_ORDER)})
    first_ep = (ep.sort_values(["subject_id", "time", "cause_rank"])
                .groupby("subject_id").first().reset_index()
                .rename(columns={"time": "endpoint_date", "hadm_id": "endpoint_hadm_id",
                                 "category": "endpoint_type"})
                [["subject_id", "endpoint_date", "endpoint_hadm_id", "endpoint_type"]])

    # --- prevalent-disease washout (pre-index rows only) ---------------------
    hist = _classify_dx(dx[dx["subject_id"].isin(idx_time.index)],
                        ENDPOINT_HISTORY_ICD10, ENDPOINT_HISTORY_ICD9)
    washout_hits = hist[hist["time"] <= hist["subject_id"].map(idx_time)]
    pts_washout = set(washout_hits["subject_id"].unique())
    print("  prevalent-disease washout, patients excluded per cause "
          "(a patient can hit >1 cause):")
    for cause in ENDPOINT_HISTORY_ICD10:
        n_c = washout_hits.loc[washout_hits["category"] == cause, "subject_id"].nunique()
        print(f"    {cause:8s}: {n_c:,}")
    print(f"  total unique patients washed out: {len(pts_washout):,}")

    # --- cohort skeleton ----------------------------------------------------------
    cohort = index_rows[["subject_id", "hadm_id", "time", "age",
                         "cardiometa_types", "num_cardiometa_conditions"]].rename(
        columns={"hadm_id": "index_hadm_id", "time": "index_date", "age": "age_at_index"})
    cohort["age_at_index"] = np.floor(cohort["age_at_index"]).astype(int)
    cohort = cohort.merge(first_ep, on="subject_id", how="left")
    cohort = cohort.merge(demo[["subject_id", "gender", "dod"]], on="subject_id", how="left")
    last_evt = D.last_event_time().rename("last_event")
    cohort = cohort.merge(last_evt.reset_index(), on="subject_id", how="left")
    censor = cohort[["dod", "last_event"]].max(axis=1)
    cohort["follow_up_days"] = np.where(
        cohort["endpoint_date"].notna(),
        (cohort["endpoint_date"] - cohort["index_date"]).dt.days,
        (censor - cohort["index_date"]).dt.days,
    )

    cohort = cohort[~cohort["subject_id"].isin(pts_washout)]
    n_after_washout = len(cohort)
    n_enc = adult_visits.groupby("subject_id")["visit_id"].nunique().rename("n_enc")
    cohort = cohort.merge(n_enc.reset_index(), on="subject_id", how="left")
    cohort = cohort[cohort["n_enc"].fillna(0) >= 2]
    n_after_min_enc = len(cohort)
    cohort = cohort[cohort["endpoint_type"].notna()
                    | (cohort["follow_up_days"] >= MIN_FOLLOWUP_DAYS)]
    n_after_fu = len(cohort)
    cohort["endpoint_type"] = cohort["endpoint_type"].fillna("censored")

    # had_icu_stay: ICU encounter strictly before index (pre-index static feature)
    icu = visits[visits["is_icu"]][["subject_id", "time"]]
    icu = icu.merge(cohort[["subject_id", "index_date"]], on="subject_id")
    cohort["had_icu_stay"] = cohort["subject_id"].isin(
        set(icu.loc[icu["time"] < icu["index_date"], "subject_id"]))
    cohort["death_during_followup"] = cohort["dod"].notna() & (cohort["dod"] >= cohort["index_date"])

    # Charlson CCI from the index encounter's diagnoses
    print("\nComputing Charlson Comorbidity Index...")
    idx_enc = index_rows[["subject_id", "enc"]]
    dx_idx = adult_dx.merge(idx_enc, on=["subject_id", "enc"])
    dx_idx = dx_idx[dx_idx["subject_id"].isin(cohort["subject_id"])]
    cci = (dx_idx.groupby("subject_id")[["icd_code", "icd_version"]]
           .apply(_compute_cci).rename("cci_score").reset_index())
    cohort = cohort.merge(cci, on="subject_id", how="left")
    cohort["cci_score"] = cohort["cci_score"].fillna(0).astype(int)

    cohort = cohort[[
        "subject_id", "gender", "age_at_index", "index_hadm_id", "index_date",
        "cardiometa_types", "num_cardiometa_conditions",
        "endpoint_type", "endpoint_date", "endpoint_hadm_id",
        "follow_up_days", "had_icu_stay", "death_during_followup", "cci_score",
    ]].reset_index(drop=True)
    out_path = os.path.join(OUTPUT_DIR, "cohort.csv")
    cohort.to_csv(out_path, index=False)

    cascade = pd.DataFrame([
        ("All EHRSHOT patients", n_total_patients),
        ("Patients with >=1 encounter at age >= 18", n_after_adult),
        ("Has cardiometabolic index dx (adult)", n_with_cm),
        ("No prevalent endpoint disease at/before index (washout)", n_after_washout),
        (">= 2 adult encounters", n_after_min_enc),
        (f"Endpoint OR >= {MIN_FOLLOWUP_DAYS}d follow-up", n_after_fu),
    ], columns=["step", "n_patients"])
    cascade.to_csv(os.path.join(OUTPUT_DIR, "cohort_cascade.csv"), index=False)

    print("\n=== EHRSHOT COHORT SUMMARY ===")
    for step, n in zip(cascade["step"], cascade["n_patients"]):
        print(f"  {step:<58s}{n:>8,d}")
    print("\nEndpoint distribution:")
    vc = cohort["endpoint_type"].value_counts()
    for e in CAUSE_ORDER + ["censored"]:
        n = int(vc.get(e, 0))
        print(f"  {e:9s}: {n:6,d} ({n / max(len(cohort), 1) * 100:5.1f}%)")
    print(f"\nAge: {cohort['age_at_index'].mean():.1f} ± {cohort['age_at_index'].std():.1f}"
          f" | female {(cohort['gender'] == 'F').mean() * 100:.1f}%"
          f" | median follow-up {cohort['follow_up_days'].median():.0f} d"
          f" | mean CCI {cohort['cci_score'].mean():.2f}"
          f" | prior ICU {cohort['had_icu_stay'].mean() * 100:.1f}%")
    print(f"Saved cohort to: {out_path}")
    return cohort


if __name__ == "__main__":
    build_cohort()

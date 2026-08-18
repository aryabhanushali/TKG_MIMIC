"""Build the cardiometabolic -> circulatory endpoint cohort."""
import os
import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR, OUTPUT_DIR,
    CARDIOMETA_ICD10, CARDIOMETA_ICD9,
    ENDPOINTS_ICD10, ENDPOINTS_ICD9,
    ENDPOINT_HISTORY_ICD10, ENDPOINT_HISTORY_ICD9,
    CCI_WEIGHTS,
    MIN_FOLLOWUP_DAYS,
)


def _icd_matches_any(code_series: pd.Series, prefixes) -> pd.Series:
    """Boolean mask: True where the code starts with any of the given prefixes."""
    if len(prefixes) == 0:
        return pd.Series(False, index=code_series.index)
    return code_series.str.startswith(tuple(prefixes), na=False)


def _classify_dx(dx: pd.DataFrame, mapping_icd10, mapping_icd9,
                 primary_only: bool = False) -> pd.DataFrame:
    """Add a 'category' column to dx rows that match any cardiometa/endpoint cat.

    primary_only: restrict to the principal diagnosis (seq_num == 1) so an
    incident endpoint is the *reason for admission*, not a secondary/chronic
    comorbidity code that would misclassify prevalent disease as a new event.
    """
    dx = dx.copy()
    if primary_only:
        dx = dx[dx["seq_num"] == 1]
    dx["category"] = None
    for cat, prefixes in mapping_icd10.items():
        mask = (dx["icd_version"] == 10) & _icd_matches_any(dx["icd_code"], prefixes)
        dx.loc[mask & dx["category"].isna(), "category"] = cat
    for cat, prefixes in mapping_icd9.items():
        mask = (dx["icd_version"] == 9) & _icd_matches_any(dx["icd_code"], prefixes)
        dx.loc[mask & dx["category"].isna(), "category"] = cat
    return dx[dx["category"].notna()]


def _compute_cci(dx_at_index: pd.DataFrame) -> int:
    """Charlson Comorbidity Index from index-admission diagnoses."""
    score = 0
    flags = {}
    for cond, spec in CCI_WEIGHTS.items():
        m10 = (dx_at_index["icd_version"] == 10) & _icd_matches_any(
            dx_at_index["icd_code"], spec["icd10"])
        m9 = (dx_at_index["icd_version"] == 9) & _icd_matches_any(
            dx_at_index["icd_code"], spec["icd9"])
        flags[cond] = bool((m10 | m9).any())
    # CCI hierarchy: severe overrides mild; complicated overrides non-complicated.
    if flags["Liver_severe"]:
        flags["Liver_mild"] = False
    if flags["Diabetes_comp"]:
        flags["Diabetes_no_comp"] = False
    if flags["Mets"]:
        flags["Cancer"] = False
    for cond, present in flags.items():
        if present:
            score += CCI_WEIGHTS[cond]["w"]
    return score


def build_cohort() -> pd.DataFrame:
    print("Loading data...")
    admissions = pd.read_csv(
        os.path.join(DATA_DIR, "admissions.csv.gz"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                 "deathtime", "admission_type"],
        parse_dates=["admittime", "dischtime", "deathtime"],
        low_memory=False,
    )
    patients = pd.read_csv(
        os.path.join(DATA_DIR, "patients.csv.gz"),
        usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
        parse_dates=["dod"],
        low_memory=False,
    )
    dx = pd.read_csv(
        os.path.join(DATA_DIR, "diagnoses_icd.csv.gz"),
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version", "seq_num"],
        low_memory=False,
    )
    icu = pd.read_csv(
        os.path.join(DATA_DIR, "icustays.csv.gz"),
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
        low_memory=False,
    )
    dx["icd_code"] = dx["icd_code"].astype(str).str.strip()
    n_total_patients = patients["subject_id"].nunique()
    print(f"  admissions: {len(admissions):,}, patients: {n_total_patients:,}, "
          f"diagnoses: {len(dx):,}, icu stays: {len(icu):,}")

    print("\nComputing age at each admission...")
    adm = admissions.merge(
        patients[["subject_id", "anchor_age", "anchor_year"]], on="subject_id", how="left"
    )
    adm["age_at_admission"] = adm["anchor_age"] + (adm["admittime"].dt.year - adm["anchor_year"])
    # Drop individual admission ROWS taken at age < 18 (a patient with any
    # adult admission survives this step via their other rows; full removal
    # of never-adult patients doesn't happen until the age >= 18 AT INDEX
    # check below, since only an adult admission can become the index date).
    adult_mask = adm["age_at_admission"] >= 18
    adm = adm[adult_mask]
    n_after_adult = adm["subject_id"].nunique()
    print(f"  patients with >=1 admission at age >= 18: {n_after_adult:,}")

    print("\nIdentifying cardiometabolic index admissions...")
    cm_dx = _classify_dx(dx, CARDIOMETA_ICD10, CARDIOMETA_ICD9)
    # Each row: a (patient, hadm, cardiometa-cat). Join with admittime.
    cm_with_time = cm_dx.merge(
        adm[["subject_id", "hadm_id", "admittime", "age_at_admission"]],
        on=["subject_id", "hadm_id"], how="inner",
    )
    # Earliest cardiometabolic admission per patient
    cm_sorted = cm_with_time.sort_values(["subject_id", "admittime"])
    index_admit = cm_sorted.groupby("subject_id").first().reset_index()
    # All cardiometa categories present at the index admission
    cats_at_idx = (cm_with_time.merge(
        index_admit[["subject_id", "hadm_id"]], on=["subject_id", "hadm_id"]
    ).groupby("subject_id")["category"].agg(lambda s: sorted(set(s))))
    index_admit = index_admit.merge(
        cats_at_idx.rename("cardiometa_types_list").reset_index(),
        on="subject_id", how="left",
    )
    index_admit["cardiometa_types"] = index_admit["cardiometa_types_list"].apply(
        lambda lst: ",".join(lst))
    index_admit["num_cardiometa_conditions"] = index_admit["cardiometa_types_list"].apply(len)
    n_with_cm = len(index_admit)
    print(f"  patients with any cardiometabolic dx: {n_with_cm:,}")

    print("\nIdentifying endpoint admissions (post-index, principal diagnosis)...")
    # Incident endpoint = endpoint code in the PRINCIPAL position (seq_num==1)
    # of a post-index admission (i.e., the reason for that admission), not a
    # secondary/chronic comorbidity code.
    ep_dx = _classify_dx(dx, ENDPOINTS_ICD10, ENDPOINTS_ICD9, primary_only=True)
    ep_with_time = ep_dx.merge(
        adm[["subject_id", "hadm_id", "admittime"]], on=["subject_id", "hadm_id"], how="inner"
    )
    ep_with_idx = ep_with_time.merge(
        index_admit[["subject_id", "admittime"]].rename(columns={"admittime": "index_date"}),
        on="subject_id", how="inner",
    )
    ep_post = ep_with_idx[ep_with_idx["admittime"] > ep_with_idx["index_date"]]
    # Earliest endpoint per patient
    ep_post_sorted = ep_post.sort_values(["subject_id", "admittime"])
    first_ep = ep_post_sorted.groupby("subject_id").first().reset_index()
    first_ep = first_ep.rename(columns={
        "admittime": "endpoint_date",
        "hadm_id": "endpoint_hadm_id",
        "category": "endpoint_type",
    })[["subject_id", "endpoint_date", "endpoint_hadm_id", "endpoint_type"]]

    # Prevalent-disease washout: exclude any patient with a chronic/history form
    # of ANY endpoint disease (matched in ANY diagnosis position) on an admission
    # at/before the index date. This removes prevalent cases (e.g. chronic AF
    # coded I48.91, old MI 412, prior PAD 440.2x) that the principal-position
    # incident definition would otherwise leave in the at-risk set.
    hist_dx = _classify_dx(dx, ENDPOINT_HISTORY_ICD10, ENDPOINT_HISTORY_ICD9)
    hist_with_idx = hist_dx.merge(
        adm[["subject_id", "hadm_id", "admittime"]], on=["subject_id", "hadm_id"], how="inner"
    ).merge(
        index_admit[["subject_id", "admittime"]].rename(columns={"admittime": "index_date"}),
        on="subject_id", how="inner",
    )
    washout_hits = hist_with_idx.loc[
        hist_with_idx["admittime"] <= hist_with_idx["index_date"]
    ]
    pts_endpoint_before = set(washout_hits["subject_id"].unique())

    # Visibility: per-cause washout counts. HF history codes (I110/I130/I132,
    # ICD-9 402.0x/402.1x/402.9x = hypertensive heart disease) overlap with
    # the HTN cardiometabolic-index definition, so index patients whose HTN is
    # coded as hypertensive heart disease can be washed out here as
    # "prevalent HF" even though HF was never their endpoint. This is the
    # intended effect of an all-cause washout (they do have prevalent
    # cardiac disease) but the size of it should be visible, not silent.
    print("  prevalent-disease washout, patients excluded per cause "
          "(a patient can hit >1 cause):")
    for cause in ENDPOINT_HISTORY_ICD10:
        n_cause = washout_hits.loc[
            washout_hits["category"] == cause, "subject_id"
        ].nunique()
        print(f"    {cause:8s}: {n_cause:,}")
    print(f"  total unique patients washed out: {len(pts_endpoint_before):,}")

    # Build cohort skeleton
    cohort = index_admit[[
        "subject_id", "hadm_id", "admittime", "age_at_admission",
        "cardiometa_types", "num_cardiometa_conditions",
    ]].rename(columns={
        "hadm_id": "index_hadm_id",
        "admittime": "index_date",
        "age_at_admission": "age_at_index",
    })
    cohort = cohort.merge(first_ep, on="subject_id", how="left")

    # Determine censor date for patients without endpoint
    last_admit = (adm.groupby("subject_id")["admittime"].max()
                  .rename("last_admit_date").reset_index())
    cohort = cohort.merge(last_admit, on="subject_id", how="left")
    cohort = cohort.merge(patients[["subject_id", "gender", "dod"]], on="subject_id", how="left")

    # follow_up_days = (endpoint_date OR last_admit OR dod) - index_date
    def _followup(row):
        if pd.notna(row["endpoint_date"]):
            return (row["endpoint_date"] - row["index_date"]).days
        candidates = []
        if pd.notna(row["dod"]):
            candidates.append(row["dod"])
        if pd.notna(row["last_admit_date"]):
            candidates.append(row["last_admit_date"])
        if not candidates:
            return np.nan
        return (max(candidates) - row["index_date"]).days

    cohort["follow_up_days"] = cohort.apply(_followup, axis=1)

    # Apply exclusions
    cohort = cohort[~cohort["subject_id"].isin(pts_endpoint_before)]
    n_after_no_prior_ep = len(cohort)
    # Count of admissions per patient
    n_admits = adm.groupby("subject_id").size().rename("n_admits").reset_index()
    cohort = cohort.merge(n_admits, on="subject_id", how="left")
    cohort = cohort[cohort["n_admits"] >= 2]
    n_after_min_admits = len(cohort)
    # Adult at index
    cohort = cohort[cohort["age_at_index"] >= 18]
    n_after_adult2 = len(cohort)
    # Drop patients without enough follow-up AND no endpoint
    has_ep = cohort["endpoint_type"].notna()
    enough_fu = cohort["follow_up_days"] >= MIN_FOLLOWUP_DAYS
    cohort = cohort[has_ep | enough_fu]
    n_after_fu = len(cohort)

    cohort["endpoint_type"] = cohort["endpoint_type"].fillna("censored")
    cohort["had_icu_stay"] = cohort["subject_id"].isin(set(icu["subject_id"].unique()))
    cohort["death_during_followup"] = (
        cohort["dod"].notna()
        & (cohort["dod"] >= cohort["index_date"])
    )

    # Charlson CCI from diagnoses at the index admission
    print("\nComputing Charlson Comorbidity Index...")
    dx_indexed = dx.merge(
        cohort[["subject_id", "index_hadm_id"]].rename(columns={"index_hadm_id": "hadm_id"}),
        on=["subject_id", "hadm_id"], how="inner",
    )
    cci_by_pt = (dx_indexed.groupby("subject_id")[["icd_code", "icd_version"]]
                 .apply(_compute_cci).rename("cci_score").reset_index())
    cohort = cohort.merge(cci_by_pt, on="subject_id", how="left")
    cohort["cci_score"] = cohort["cci_score"].fillna(0).astype(int)

    # Final column ordering
    cohort = cohort[[
        "subject_id", "gender", "age_at_index", "index_hadm_id", "index_date",
        "cardiometa_types", "num_cardiometa_conditions",
        "endpoint_type", "endpoint_date", "endpoint_hadm_id",
        "follow_up_days", "had_icu_stay", "death_during_followup", "cci_score",
    ]].reset_index(drop=True)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "cohort.csv")
    cohort.to_csv(out_path, index=False)

    # Persist the real exclusion cascade so the CONSORT figure reads actual
    # counts instead of hardcoded constants.
    cascade = pd.DataFrame(
        [
            ("All MIMIC-IV patients", n_total_patients),
            ("Patients with >=1 admission at age >= 18", n_after_adult),
            ("Has cardiometabolic index dx (among adult admissions)", n_with_cm),
            ("No prevalent endpoint disease at/before index (washout)", n_after_no_prior_ep),
            (">= 2 admissions (any age)", n_after_min_admits),
            ("Adult (>=18) at the index admission itself", n_after_adult2),
            (f"Endpoint OR >= {MIN_FOLLOWUP_DAYS}d follow-up", n_after_fu),
        ],
        columns=["step", "n_patients"],
    )
    cascade.to_csv(os.path.join(OUTPUT_DIR, "cohort_cascade.csv"), index=False)

    # Report
    print("\n=== COHORT SUMMARY ===")
    print(f"Total patients: {len(cohort):,}")
    print("\nExclusion cascade:")
    for step, n in zip(cascade["step"], cascade["n_patients"]):
        print(f"  {step:<58s}{n:>10,d}")

    print("\nEndpoint distribution:")
    ep_counts = cohort["endpoint_type"].value_counts()
    for ep in ["MI", "Stroke", "HF", "AF", "PAD", "censored"]:
        n = int(ep_counts.get(ep, 0))
        pct = (n / len(cohort) * 100) if len(cohort) else 0
        print(f"  {ep:9s}: {n:6,d} ({pct:5.1f}%)")

    print(f"\nAge:                  {cohort['age_at_index'].mean():.1f} ± "
          f"{cohort['age_at_index'].std():.1f}")
    pct_female = (cohort["gender"] == "F").mean() * 100
    print(f"Gender:               {pct_female:.1f}% female")
    print(f"Mean follow-up:       {cohort['follow_up_days'].mean():.1f} days "
          f"(median {cohort['follow_up_days'].median():.0f})")
    print(f"Mean CCI:             {cohort['cci_score'].mean():.2f} ± "
          f"{cohort['cci_score'].std():.2f}")
    n_icu = int(cohort["had_icu_stay"].sum())
    print(f"Patients w/ ICU stay: {n_icu:,} ({n_icu/len(cohort)*100:.1f}%)")

    # Warnings
    small_classes = []
    for ep in ["MI", "Stroke", "HF", "AF", "PAD"]:
        n = int(ep_counts.get(ep, 0))
        if n < 200:
            small_classes.append((ep, n))
    if small_classes:
        print("\nWARNING: small endpoint classes (< 200 patients):")
        for ep, n in small_classes:
            print(f"  {ep}: {n}")
    hf_af_pad = sum(int(ep_counts.get(e, 0)) for e in ["HF", "AF", "PAD"])
    hf_n = int(ep_counts.get("HF", 0))
    af_n = int(ep_counts.get("AF", 0))
    pad_n = int(ep_counts.get("PAD", 0))
    if hf_n < 200 and af_n < 200 and pad_n < 200:
        print(f"\nSUGGESTION: HF+AF+PAD all small. Consider pooling into "
              f"'Other circulatory' ({hf_af_pad} total).")

    print(f"\nSaved cohort to: {out_path}")
    return cohort


if __name__ == "__main__":
    build_cohort()

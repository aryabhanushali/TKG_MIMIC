"""Build the eICU cardiometabolic → circulatory disease cohort.

Temporal design (disclosed explicitly because it differs from MIMIC-IV):
  - eICU covers 2014–2015 only. There are no absolute dates — only minute
    offsets from unit admission and an integer discharge year.
  - "Index event": earliest ICU unit stay per patient with a cardiometabolic
    diagnosis (any diagnosispriority position).
  - "Outcome event": a LATER unit stay in a STRICTLY LATER hospitaldischargeyear
    (i.e., 2014 → 2015) with a circulatory endpoint as the Primary diagnosis.
    Same-year cross-stay ordering is unreliable (no calendar dates), so those
    patients are excluded and counted in the cascade.
  - This gives a maximum follow-up of ~1 year.  The 1/3/5-year horizons used
    in MIMIC are not achievable; only a 1-year (approximate) horizon is used.
  - "Pre-index history" is limited to events within the index ICU unit stay
    itself (typically 1–3 days).  Prior unit stays from the SAME year as the
    index stay are optionally included but with very approximate timing.
  - Washout: patients excluded if pastHistory.csv OR any prior hospital-stay
    diagnosis row contains a prevalent-disease pattern for any of the 5 endpoints.
"""
import os
import re
import numpy as np
import pandas as pd

from src.eicu.config_eicu import (
    EICU_DATA_DIR, EICU_OUTPUT_DIR,
    CARDIOMETA_TEXT, ENDPOINT_TEXT, ENDPOINT_ADMITDX,
    WASHOUT_HISTORY_TEXT, WASHOUT_DX_TEXT,
)

CAUSES = ["MI", "Stroke", "HF", "AF", "PAD"]
MIN_AGE = 18


def _text_match(series: pd.Series, pattern: str) -> pd.Series:
    return series.str.contains(pattern, regex=True, na=False)


def _any_match(series: pd.Series, patterns: dict) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    for pat in patterns.values():
        mask |= _text_match(series, pat)
    return mask


def _first_endpoint_after(ep_stays: pd.DataFrame, index_row: pd.Series) -> pd.Series | None:
    """Return the earliest endpoint stay that is strictly later than index_row.

    'Strictly later' = hospitaldischargeyear of endpoint > index year.
    Same-year cross-stay ordering is not used (see module docstring).
    """
    later = ep_stays[ep_stays["hospitaldischargeyear"] > index_row["hospitaldischargeyear"]]
    if later.empty:
        return None
    return later.sort_values("hospitaldischargeyear").iloc[0]


def build_cohort_eicu() -> pd.DataFrame:
    os.makedirs(EICU_OUTPUT_DIR, exist_ok=True)
    print("=== eICU COHORT CONSTRUCTION ===\n")

    # -----------------------------------------------------------------------
    # 1. Patient table
    # -----------------------------------------------------------------------
    print("[1] Loading patient table...")
    pt = pd.read_csv(
        os.path.join(EICU_DATA_DIR, "patient.csv"),
        usecols=["patientunitstayid", "patienthealthsystemstayid", "uniquepid",
                 "gender", "age", "hospitalid",
                 "hospitaldischargeyear", "hospitaladmitoffset",
                 "hospitaldischargeoffset", "unitdischargeoffset", "unitvisitnumber",
                 "unitdischargestatus", "hospitaldischargestatus",
                 "admissionweight", "admissionheight"],
        low_memory=False,
    )
    # eICU age is stored as integer but capped at 89 for HIPAA compliance.
    # A string value "> 89" or "89+" may appear — coerce to 89.
    pt["age_num"] = pd.to_numeric(pt["age"], errors="coerce")
    pt["age_num"] = pt["age_num"].fillna(89)   # treat unparseable (">89" etc.) as 89
    n_total_stays = len(pt)
    n_total_patients = pt["uniquepid"].nunique()
    print(f"  total unit stays: {n_total_stays:,}, unique patients: {n_total_patients:,}")

    # Adult filter (>= 18)
    adult_mask = pt["age_num"] >= MIN_AGE
    pt_adult = pt[adult_mask].copy()
    n_adult = pt_adult["uniquepid"].nunique()
    print(f"  unique patients age >= {MIN_AGE}: {n_adult:,}")

    # -----------------------------------------------------------------------
    # 2. Diagnosis table — load once, used for cardiometa index and endpoint
    # -----------------------------------------------------------------------
    print("\n[2] Loading diagnosis table...")
    dx = pd.read_csv(
        os.path.join(EICU_DATA_DIR, "diagnosis.csv"),
        usecols=["patientunitstayid", "diagnosisoffset", "diagnosisstring",
                 "icd9code", "diagnosispriority"],
        low_memory=False,
    )
    dx["diagnosisstring"] = dx["diagnosisstring"].fillna("").str.strip()
    dx_lower = dx["diagnosisstring"].str.lower()
    print(f"  diagnosis rows: {len(dx):,}, "
          f"unit stays represented: {dx['patientunitstayid'].nunique():,}")

    # -----------------------------------------------------------------------
    # 3. Identify cardiometabolic stays (any diagnosispriority position)
    # -----------------------------------------------------------------------
    print("\n[3] Identifying cardiometabolic unit stays...")
    cm_mask = _any_match(dx_lower, CARDIOMETA_TEXT)
    cm_stays_set = set(dx.loc[cm_mask, "patientunitstayid"])
    print(f"  unit stays with any cardiometabolic dx: {len(cm_stays_set):,}")

    for cond, pat in CARDIOMETA_TEXT.items():
        n = dx.loc[_text_match(dx_lower, pat), "patientunitstayid"].nunique()
        print(f"    {cond}: {n:,} unit stays")

    # Join with patient table to get per-stay year and ordering info
    cm_pt = pt_adult[pt_adult["patientunitstayid"].isin(cm_stays_set)].copy()
    # Which cardiometa conditions are present at each stay
    for cond, pat in CARDIOMETA_TEXT.items():
        stays_with_cond = set(dx.loc[_text_match(dx_lower, pat), "patientunitstayid"])
        cm_pt[f"cm_{cond}"] = cm_pt["patientunitstayid"].isin(stays_with_cond)
    cm_cond_cols = [f"cm_{c}" for c in CARDIOMETA_TEXT]

    # -----------------------------------------------------------------------
    # 4. Index stay: earliest cardiometabolic unit stay per patient
    #    Ordering: first by hospitaldischargeyear, then unitvisitnumber
    # -----------------------------------------------------------------------
    print("\n[4] Selecting index stay (earliest cardiometabolic stay per patient)...")
    cm_pt_sorted = cm_pt.sort_values(["uniquepid", "hospitaldischargeyear", "unitvisitnumber"])
    index_stays = cm_pt_sorted.groupby("uniquepid").first().reset_index()
    n_with_cm = len(index_stays)
    print(f"  patients with cardiometabolic index stay: {n_with_cm:,}")
    print(f"  index year distribution:")
    print(f"    2014: {(index_stays['hospitaldischargeyear']==2014).sum():,}  "
          f"2015: {(index_stays['hospitaldischargeyear']==2015).sum():,}")

    # -----------------------------------------------------------------------
    # 5. Identify circulatory endpoint stays
    #    Primary diagnosispriority only; additionally checked against admissionDx
    # -----------------------------------------------------------------------
    print("\n[5] Identifying circulatory endpoint stays (Primary dx only)...")
    primary_mask = dx["diagnosispriority"] == "Primary"
    ep_rows = {ep: dx.loc[primary_mask & _text_match(dx_lower, pat)].copy()
               for ep, pat in ENDPOINT_TEXT.items()}
    for ep, rows in ep_rows.items():
        print(f"  {ep}: {rows['patientunitstayid'].nunique():,} Primary unit stays")

    # Secondary source: admissionDx table (checked only for MI/Stroke/HF/AF since PAD=0)
    print("  cross-checking with admissionDx table...")
    adx = pd.read_csv(
        os.path.join(EICU_DATA_DIR, "admissionDx.csv"),
        usecols=["patientunitstayid", "admitdxpath"],
        low_memory=False,
    )
    adx["admitdxpath"] = adx["admitdxpath"].fillna("").str.lower()
    for ep, pat in ENDPOINT_ADMITDX.items():
        adx_stays = set(adx.loc[_text_match(adx["admitdxpath"], pat),
                                 "patientunitstayid"])
        pre = len(ep_rows[ep]["patientunitstayid"].unique())
        ep_rows[ep] = pd.concat([
            ep_rows[ep],
            dx[dx["patientunitstayid"].isin(adx_stays)].assign(
                diagnosispriority="admitdx_primary")
        ], ignore_index=True).drop_duplicates("patientunitstayid")
        post = len(ep_rows[ep]["patientunitstayid"].unique())
        if post > pre:
            print(f"    {ep}: +{post-pre} stays from admissionDx")

    # Map patientunitstayid → endpoint type
    ep_stay_to_type: dict[int, str] = {}
    for ep, rows in ep_rows.items():
        for sid in rows["patientunitstayid"].unique():
            if sid not in ep_stay_to_type:
                ep_stay_to_type[int(sid)] = ep

    ep_pt = pt_adult[pt_adult["patientunitstayid"].isin(ep_stay_to_type)].copy()
    ep_pt["endpoint_type"] = ep_pt["patientunitstayid"].map(ep_stay_to_type)

    # -----------------------------------------------------------------------
    # 6. Prevalent-disease washout
    #    (a) pastHistory.csv free-text patterns
    #    (b) Any prior diagnosis stay (not just Primary) in the same patient
    # -----------------------------------------------------------------------
    print("\n[6] Prevalent-disease washout...")

    print("  (a) pastHistory.csv patterns...")
    ph = pd.read_csv(
        os.path.join(EICU_DATA_DIR, "pastHistory.csv"),
        usecols=["patientunitstayid", "pasthistorypath"],
        low_memory=False,
    )
    ph["pasthistorypath"] = ph["pasthistorypath"].fillna("").str.lower()
    ph_washout_stays: set[int] = set()
    washout_counts: dict[str, int] = {}
    for ep, pat in WASHOUT_HISTORY_TEXT.items():
        m = _text_match(ph["pasthistorypath"], pat)
        stays = set(ph.loc[m, "patientunitstayid"].astype(int).unique())
        # Any stay in this patient's record that has the pastHistory hit
        # means the PATIENT had prevalent disease; look up the uniquepid
        pts_hit = set(pt_adult.loc[
            pt_adult["patientunitstayid"].isin(stays), "uniquepid"])
        # Exclude from index_stays
        washout_counts[f"pastHistory_{ep}"] = len(pts_hit)
        ph_washout_stays |= stays

    ph_washout_pts = set(pt_adult.loc[
        pt_adult["patientunitstayid"].isin(ph_washout_stays), "uniquepid"])
    print(f"    patients with prevalent-disease pastHistory: {len(ph_washout_pts):,}")
    for k, v in washout_counts.items():
        print(f"      {k}: {v:,}")

    print("  (b) Prior diagnosis rows (any priority)...")
    dx_washout_mask = _any_match(dx_lower, WASHOUT_DX_TEXT)
    dx_washout_stays = set(dx.loc[dx_washout_mask, "patientunitstayid"].astype(int))
    dx_washout_pts = set(pt_adult.loc[
        pt_adult["patientunitstayid"].isin(dx_washout_stays), "uniquepid"])
    print(f"    patients with circulatory dx in ANY stay: {len(dx_washout_pts):,}")

    all_washout_pts = ph_washout_pts | dx_washout_pts
    print(f"  total unique patients washed out (union): {len(all_washout_pts):,}")

    # -----------------------------------------------------------------------
    # 7. Build cohort using unitvisitnumber ordering
    #    An endpoint stay is valid if its unitvisitnumber > index stay's
    #    unitvisitnumber for the same uniquepid.  Both same-year and
    #    cross-year pairings are included.
    #
    #    TTE approximation (no absolute dates in eICU):
    #      (a) Same patienthealthsystemstayid: use unitdischargeoffset diff.
    #      (b) Different hospital stay, cross-year: 365 × year_gap days.
    #      (c) Different hospital stay, same year: 180 days (midpoint
    #          estimate; flagged explicitly — exact gap is unknown).
    #    Censored patients: TTE = 365 days (end of observation window).
    # -----------------------------------------------------------------------
    print("\n[7] Matching index stays → subsequent circulatory endpoint stays...")
    print("    Ordering: unitvisitnumber (same-year and cross-year both included)")

    # All post-washout index stays (any year)
    idx_postw = index_stays[
        ~index_stays["uniquepid"].isin(all_washout_pts)
    ].copy()
    print(f"  post-washout index patients (all years): {len(idx_postw):,}")
    print(f"  index year distribution:")
    for yr, cnt in idx_postw["hospitaldischargeyear"].value_counts().sort_index().items():
        print(f"    {yr}: {cnt:,}")

    # Merge endpoint stays with index stays on uniquepid so we can filter by
    # unitvisitnumber vectorially (avoids slow per-row loop)
    idx_cols = ["uniquepid", "patientunitstayid", "patienthealthsystemstayid",
                "hospitaldischargeyear", "unitdischargeoffset",
                "hospitaldischargeoffset", "unitvisitnumber"]
    ep_cols  = ["uniquepid", "patientunitstayid", "patienthealthsystemstayid",
                "endpoint_type", "unitvisitnumber",
                "hospitaldischargeyear", "unitdischargeoffset",
                "hospitaldischargeoffset"]

    idx_sel = idx_postw[idx_cols].copy().rename(columns={
        "patientunitstayid":           "idx_unitstayid",
        "patienthealthsystemstayid":   "idx_hospstayid",
        "hospitaldischargeyear":       "idx_year",
        "unitdischargeoffset":         "idx_unitdischargeoffset",
        "hospitaldischargeoffset":     "idx_hosp_dischargeoffset",
        "unitvisitnumber":             "idx_visitnum",
    })

    ep_sel = ep_pt[ep_cols].copy().rename(columns={
        "patientunitstayid":           "ep_unitstayid",
        "patienthealthsystemstayid":   "ep_hospstayid",
        "unitvisitnumber":             "ep_visitnum",
        "hospitaldischargeyear":       "ep_year",
        "unitdischargeoffset":         "ep_unitdischargeoffset",
        "hospitaldischargeoffset":     "ep_hosp_dischargeoffset",
    })

    merged = idx_sel.merge(ep_sel, on="uniquepid", how="left")

    # Keep only endpoint stays that come AFTER the index stay
    valid_ep = merged[
        merged["ep_visitnum"].notna() & (merged["ep_visitnum"] > merged["idx_visitnum"])
    ].copy()

    # Among valid endpoints per patient take the earliest (lowest visitnum)
    valid_ep = (valid_ep
                .sort_values(["uniquepid", "ep_visitnum"])
                .groupby("uniquepid")
                .first()
                .reset_index())

    # --- TTE approximation ---
    def _compute_tte(row) -> float:
        if row["ep_hospstayid"] == row["idx_hospstayid"]:
            # Same hospital stay: use offset difference (minutes → days)
            tte = (row["ep_unitdischargeoffset"] - row["idx_unitdischargeoffset"]) / 1440.0
        elif row["ep_year"] != row["idx_year"]:
            # Cross-year, different hospital stay
            tte = float(365 * (row["ep_year"] - row["idx_year"]))
        else:
            # Same year, different hospital stay — exact gap unknown
            tte = 180.0   # conservative midpoint estimate; flagged below
        return max(tte, 1.0)

    valid_ep["tte_approx"] = valid_ep.apply(_compute_tte, axis=1)

    n_same_year_diff_hosp = int(
        ((valid_ep["ep_year"] == valid_ep["idx_year"]) &
         (valid_ep["ep_hospstayid"] != valid_ep["idx_hospstayid"])).sum()
    )
    n_cross_year = int((valid_ep["ep_year"] > valid_ep["idx_year"]).sum())
    n_same_hosp  = int((valid_ep["ep_hospstayid"] == valid_ep["idx_hospstayid"]).sum())
    print(f"  patients with valid subsequent endpoint: {len(valid_ep):,}")
    print(f"    same hospital stay (exact TTE from offsets): {n_same_hosp:,}")
    print(f"    cross-year, diff hospital stay (TTE ≈ 365 days): {n_cross_year:,}")
    print(f"    same-year, diff hospital stay (TTE ≈ 180 days, inexact): "
          f"{n_same_year_diff_hosp:,}")

    # Build per-patient mapping
    ep_lookup = valid_ep.set_index("uniquepid")

    rows = []
    for _, idx_row in idx_postw.iterrows():
        pid = idx_row["uniquepid"]
        if pid in ep_lookup.index:
            ep = ep_lookup.loc[pid]
            endpoint_type  = ep["endpoint_type"]
            endpoint_stay  = int(ep["ep_unitstayid"])
            tte_days_approx = float(ep["tte_approx"])
        else:
            endpoint_type  = "censored"
            endpoint_stay  = np.nan
            tte_days_approx = 365.0

        cond_list = [c.replace("cm_", "") for c in cm_cond_cols
                     if idx_row.get(c, False)]
        rows.append({
            "uniquepid":                   pid,
            "index_unitstayid":            int(idx_row["patientunitstayid"]),
            "index_year":                  int(idx_row["hospitaldischargeyear"]),
            "index_unitdischargeoffset":   int(idx_row["unitdischargeoffset"]),
            "index_hosp_dischargeoffset":  int(idx_row["hospitaldischargeoffset"]),
            "gender":                      idx_row["gender"],
            "age":                         float(idx_row["age_num"]),
            "hospitalid":                  int(idx_row["hospitalid"]),
            "admissionweight_kg":          idx_row.get("admissionweight", np.nan),
            "endpoint_type":               endpoint_type,
            "endpoint_unitstayid":         endpoint_stay,
            "time_to_event_days_approx":   tte_days_approx,
            "cardiometa_conditions":       ",".join(cond_list),
            "num_cardiometa_conditions":   len(cond_list),
        })

    cohort = pd.DataFrame(rows)
    print(f"  final cohort (all post-washout with valid temporal ordering): {len(cohort):,}")

    # -----------------------------------------------------------------------
    # 8. Compute CCI from index-stay diagnoses
    # -----------------------------------------------------------------------
    print("\n[8] Computing Charlson CCI from index-stay diagnoses...")
    from src.config import CCI_WEIGHTS

    index_stay_ids = set(cohort["index_unitstayid"])
    dx_index = dx[dx["patientunitstayid"].isin(index_stay_ids)].copy()
    # eICU icd9code: "414.00, I25.10" — split on comma, take the ICD-9 part
    # (first token, strip whitespace and dots).
    dx_index = dx_index[dx_index["icd9code"].notna()].copy()
    dx_index["icd9_clean"] = (dx_index["icd9code"].str.split(",").str[0]
                               .str.strip().str.replace(".", "", regex=False))
    dx_index["icd10_clean"] = (dx_index["icd9code"].str.split(",").str[-1]
                                .str.strip().str.replace(".", "", regex=False))

    def _cci_from_eicu_dx(stay_dx: pd.DataFrame) -> int:
        score = 0
        flags: dict[str, bool] = {}
        for cond, spec in CCI_WEIGHTS.items():
            found9 = any(stay_dx["icd9_clean"].str.startswith(p, na=False).any()
                         for p in spec["icd9"])
            found10 = any(stay_dx["icd10_clean"].str.startswith(p, na=False).any()
                          for p in spec["icd10"])
            flags[cond] = found9 or found10
        if flags.get("Liver_severe"):    flags["Liver_mild"]      = False
        if flags.get("Diabetes_comp"):   flags["Diabetes_no_comp"] = False
        if flags.get("Mets"):            flags["Cancer"]           = False
        for cond, present in flags.items():
            if present:
                score += CCI_WEIGHTS[cond]["w"]
        return score

    cci_map: dict[int, int] = {}
    for stay_id, grp in dx_index.groupby("patientunitstayid"):
        cci_map[int(stay_id)] = _cci_from_eicu_dx(grp)
    cohort["cci_score"] = cohort["index_unitstayid"].map(cci_map).fillna(0).astype(int)

    # -----------------------------------------------------------------------
    # 9. had_prior_icu_stay: whether the patient had a prior ICU unit stay
    #    BEFORE the index stay (i.e., unitvisitnumber > 1 at some prior stay).
    #    NOTE: In MIMIC, had_icu_stay = True if patient had any ICU stay
    #    BEFORE their index hospital admission.  In eICU, the index IS an ICU
    #    stay, so this feature answers a subtly different question (prior ICU
    #    exposure within the eICU dataset only).
    #    This is flagged explicitly in transfer_eval.py.
    # -----------------------------------------------------------------------
    index_visit_nums = dict(zip(cohort["index_unitstayid"],
                                cohort["index_unitstayid"].map(
                                    pt_adult.set_index("patientunitstayid")["unitvisitnumber"])))
    cohort["had_prior_icu_stay"] = cohort["index_unitstayid"].map(
        lambda sid: pt_adult.loc[
            pt_adult["patientunitstayid"] == sid, "unitvisitnumber"
        ].values[0] > 1 if (pt_adult["patientunitstayid"] == sid).any() else False
    )
    # Vectorised version is faster
    idx_visit_num = pt_adult.set_index("patientunitstayid")["unitvisitnumber"]
    cohort["had_prior_icu_stay"] = (
        cohort["index_unitstayid"].map(idx_visit_num).fillna(1) > 1
    ).astype(int)

    cohort["female"] = (cohort["gender"].str.lower() == "female").astype(int)

    # -----------------------------------------------------------------------
    # 10. Summary and export
    # -----------------------------------------------------------------------
    out_path = os.path.join(EICU_OUTPUT_DIR, "cohort_eicu.csv")
    cohort.to_csv(out_path, index=False)

    print("\n=== eICU COHORT SUMMARY ===")
    print(f"Final cohort: {len(cohort):,} patients")
    ep_counts = cohort["endpoint_type"].value_counts()
    for ep in CAUSES + ["censored"]:
        n = int(ep_counts.get(ep, 0))
        pct = n / len(cohort) * 100
        warn = "  *** TOO FEW FOR RELIABLE AUROC (<30)" if n < 30 and ep != "censored" else ""
        print(f"  {ep:9s}: {n:5,d} ({pct:5.1f}%){warn}")

    n_events = int((cohort["endpoint_type"] != "censored").sum())
    print(f"\nAny endpoint: {n_events:,} ({n_events/len(cohort)*100:.1f}%)")
    print(f"Age: {cohort['age'].mean():.1f} ± {cohort['age'].std():.1f} (note: capped at 89)")
    print(f"Female: {cohort['female'].mean()*100:.1f}%")
    print(f"Mean CCI: {cohort['cci_score'].mean():.2f}")
    print(f"Mean cardiometa conditions: {cohort['num_cardiometa_conditions'].mean():.2f}")
    print(f"\nTime-to-event (days, approximate):")
    tte = cohort["time_to_event_days_approx"]
    print(f"  min={tte.min():.0f}, median={tte.median():.0f}, "
          f"max={tte.max():.0f}")

    cascade = pd.DataFrame([
        ("All eICU patients", n_total_patients),
        (f"Adults (age >= {MIN_AGE})", n_adult),
        ("With any cardiometabolic dx", pt_adult[pt_adult["patientunitstayid"]
                                                  .isin(cm_stays_set)]["uniquepid"].nunique()),
        ("Post washout (no prevalent circulatory disease)", len(idx_postw)),
        ("Final cohort (unitvisitnumber ordering applied)", len(cohort)),
    ], columns=["step", "n_patients"])
    cascade.to_csv(os.path.join(EICU_OUTPUT_DIR, "cohort_eicu_cascade.csv"), index=False)

    print("\nExclusion cascade:")
    for step, n in zip(cascade["step"], cascade["n_patients"]):
        print(f"  {step:<60s} {n:>8,d}")

    print(f"\nSaved cohort to: {out_path}")
    print("\n*** STRUCTURAL LIMITATIONS (disclose in paper) ***")
    print("  1. eICU covers 2014-2015 only; max follow-up ~1 year vs. MIMIC's 5.")
    print("  2. Temporal ordering uses unitvisitnumber (same-year and cross-year).")
    print("  3. TTE for same-year, different hospital stays is approximated as 180 days.")
    print("  4. Pre-index feature window = index ICU stay only (~1-3 days median).")
    print("  5. PAD typically has 0 primary-admission cases in eICU.")
    print("  6. had_prior_icu_stay != had_icu_stay (MIMIC): different clinical meaning.")
    print("  7. Age capped at 89 in eICU (HIPAA).")
    return cohort


if __name__ == "__main__":
    build_cohort_eicu()

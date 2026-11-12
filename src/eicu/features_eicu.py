"""Map eICU tables into the MIMIC concept space used by the trained models.

Returns a feature matrix (bag-of-codes + value summaries + static) in exactly
the same column layout as the MIMIC baselines_survival.py training matrix,
allowing the MIMIC-trained Cox and XGBoost models to score eICU patients
without retraining.

FEATURE MISMATCH FLAGS (printed at runtime, included in the transfer report):
  M1  OMR_BP_* / OMR_BMI_*  – MIMIC's outpatient BP/BMI (from the OMR table)
      have NO equivalent in eICU, which is an ICU-only dataset.  These feature
      columns will be all-zero for every eICU patient.
  M2  had_icu_stay static feature – In MIMIC this means "had an ICU stay before
      the index general hospital admission."  In eICU the index IS an ICU stay,
      so the eICU analogue (had_prior_icu_stay) answers a subtly different
      clinical question; used as the best available proxy.
  M3  Drug concept IDs – MIMIC uses the full drug name string normalised to
      uppercase-underscore (e.g. DRUG_METOPROLOL_TARTRATE).  eICU drug names
      include dosage / route / formulation (e.g. "METOPROLOL TARTRATE 25 MG PO
      TABS"), which is stripped to the first word(s) before the dose number.
      Only drugs whose normalised prefix matches a MIMIC concept ID contribute
      to bag-of-codes; all others are counted in n_unmatched_drugs.
  M4  Pre-index window – MIMIC: up to 5 years.  eICU: index ICU stay only
      (~1–3 days typical).  Feature density is therefore far lower per patient.
  M5  ICD code format – eICU stores codes as "414.00, I25.10" (ICD-9 + ICD-10
      with dots); MIMIC stores them without dots.  Dots are stripped here.
  M6  vitalPeriodic BP (systemicsystolic/diastolic) is invasive arterial in eICU
      ICU context.  MIMIC chartevents mixes invasive and non-invasive BP.
      Mapped to the same VITAL_BP_SYS / _DIA concepts; the population-level
      value distributions will differ.
"""
import os
import re
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, hstack

from src.eicu.config_eicu import (
    EICU_DATA_DIR,
    EICU_LAB_MAP, EICU_VITAL_PERIODIC_MAP, EICU_VITAL_APERIODIC_MAP,
    EICU_INFUSION_MAP, EICU_OUTPUT_MAP,
    VITAL_PERIODIC_SAMPLE_RATE, LAB_SAMPLE_RATE, SEED,
)

STAT_NAMES = ["mean", "max", "min", "last", "count", "slope"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_icd_dots(code: str) -> str:
    return code.replace(".", "").strip()


def _drug_name_to_concept(raw_name: str) -> str:
    """Strip dosage / route / formulation from eICU drug name.

    e.g. "METOPROLOL TARTRATE 25 MG PO TABS" → "METOPROLOL_TARTRATE"
    Strategy: take all words before the first token that looks like a dose
    (a number or a unit abbreviation). This is intentionally conservative;
    ambiguous names that don't parse cleanly contribute to n_unmatched_drugs.
    """
    raw = raw_name.strip().upper()
    tokens = raw.split()
    drug_tokens = []
    for tok in tokens:
        if re.match(r"^\d", tok):       # starts with a digit → dose begins here
            break
        if tok in {"MG", "MCG", "MEQ", "ML", "GM", "IU", "UNIT", "UNITS",
                   "PO", "IV", "IM", "SQ", "PR", "IN", "SL", "TOP",
                   "TABS", "CAP", "SOLN", "SUSP", "INJ", "TAB", "EC",
                   "ER", "SR", "XR", "CR", "DR", "MDI"}:
            break
        drug_tokens.append(tok)
    return "DRUG_" + "_".join(drug_tokens).replace("/", "_") if drug_tokens else ""


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0.0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def _value_summaries(
    events: pd.DataFrame,
    concept_col: str,
    value_col: str,
    day_col: str,
    feat_concept_ids_str: list[str],
) -> np.ndarray:
    """Compute mean/max/min/last/count/slope per (patient, feat_concept).

    Returns shape (n_patients_in_events, len(feat_concept_ids_str) * 6).
    Caller must handle z-score normalisation with MIMIC training stats.
    """
    n_stats = len(STAT_NAMES)
    pid_order = events["uniquepid"].unique().tolist()
    pid_to_row = {p: i for i, p in enumerate(pid_order)}
    cid_to_col = {c: j for j, c in enumerate(feat_concept_ids_str)}

    val_ev = events[events[value_col].notna() & events[concept_col].isin(cid_to_col)].copy()
    arr = np.zeros((len(pid_order), len(feat_concept_ids_str) * n_stats), dtype=np.float32)

    if val_ev.empty:
        return arr, pid_order

    val_ev = val_ev.sort_values(["uniquepid", day_col])
    for (pid, cid), grp in val_ev.groupby(["uniquepid", concept_col]):
        i = pid_to_row[pid]
        j = cid_to_col[cid]
        base = j * n_stats
        vals = grp[value_col].to_numpy(dtype=np.float64)
        days = grp[day_col].to_numpy(dtype=np.float64)
        arr[i, base + 0] = vals.mean()
        arr[i, base + 1] = vals.max()
        arr[i, base + 2] = vals.min()
        arr[i, base + 3] = vals[-1]
        arr[i, base + 4] = len(vals)
        arr[i, base + 5] = _slope(days, vals)
    return arr, pid_order


# ---------------------------------------------------------------------------
# Per-modality event extractors
# ---------------------------------------------------------------------------

def _diagnosis_events(cohort: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """ICD diagnosis codes from the index ICU stay."""
    dx = pd.read_csv(
        os.path.join(EICU_DATA_DIR, "diagnosis.csv"),
        usecols=["patientunitstayid", "diagnosisoffset", "icd9code"],
        low_memory=False,
    )
    stay_to_pid = dict(zip(cohort["index_unitstayid"], cohort["uniquepid"]))
    stay_dur = dict(zip(cohort["index_unitstayid"], cohort["index_unitdischargeoffset"]))
    dx = dx[dx["patientunitstayid"].isin(stay_to_pid) & dx["icd9code"].notna()].copy()
    dx["uniquepid"] = dx["patientunitstayid"].map(stay_to_pid)
    dx["stay_los_min"] = dx["patientunitstayid"].map(stay_dur)

    # Relative days: (event_offset - stay_discharge_offset) / 1440
    # → events at admission are most negative; discharge = 0
    dx["relative_days"] = (
        (dx["diagnosisoffset"] - dx["stay_los_min"]) / 1440.0
    )

    rows = []
    n_unmatched = 0
    bow_col = meta["concept_id_to_bow_col"]
    for _, r in dx.iterrows():
        raw = str(r["icd9code"]).strip()
        parts = [p.strip() for p in raw.split(",")]
        for part in parts:
            clean = _strip_icd_dots(part)
            # Determine version: ICD-10 starts with a letter
            if clean and clean[0].isalpha():
                cid = f"ICD10_{clean}"
            else:
                cid = f"ICD9_{clean}"
            if cid in bow_col:
                rows.append({
                    "uniquepid":     r["uniquepid"],
                    "concept_id":    cid,
                    "relative_days": r["relative_days"],
                    "value_num":     np.nan,
                })
            else:
                n_unmatched += 1
    print(f"  diagnosis: {len(rows):,} matched events, {n_unmatched:,} unmatched concept IDs")
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["uniquepid", "concept_id", "relative_days", "value_num"])


def _medication_events(cohort: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Medication prescriptions from the index ICU stay."""
    med = pd.read_csv(
        os.path.join(EICU_DATA_DIR, "medication.csv"),
        usecols=["patientunitstayid", "drugstartoffset", "drugname"],
        low_memory=False,
    )
    stay_to_pid = dict(zip(cohort["index_unitstayid"], cohort["uniquepid"]))
    stay_dur = dict(zip(cohort["index_unitstayid"], cohort["index_unitdischargeoffset"]))
    med = med[med["patientunitstayid"].isin(stay_to_pid) & med["drugname"].notna()].copy()
    med["uniquepid"] = med["patientunitstayid"].map(stay_to_pid)
    med["stay_los_min"] = med["patientunitstayid"].map(stay_dur)
    med["relative_days"] = (med["drugstartoffset"] - med["stay_los_min"]) / 1440.0
    med["concept_id"] = med["drugname"].apply(_drug_name_to_concept)

    bow_col = meta["concept_id_to_bow_col"]
    matched = med[med["concept_id"].isin(bow_col)].copy()
    n_unmatched = int((~med["concept_id"].isin(bow_col)).sum())
    print(f"  medication: {len(matched):,} matched, {n_unmatched:,} unmatched "
          f"(drug name prefix not in MIMIC train vocab)")

    if matched.empty:
        return pd.DataFrame(columns=["uniquepid", "concept_id", "relative_days", "value_num"])
    matched = matched[["uniquepid", "concept_id", "relative_days"]].copy()
    matched["value_num"] = np.nan
    return matched


def _lab_events(cohort: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Lab results from the index ICU stay, sampled at LAB_SAMPLE_RATE."""
    lab = pd.read_csv(
        os.path.join(EICU_DATA_DIR, "lab.csv"),
        usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
        low_memory=False,
    )
    stay_to_pid = dict(zip(cohort["index_unitstayid"], cohort["uniquepid"]))
    stay_dur = dict(zip(cohort["index_unitstayid"], cohort["index_unitdischargeoffset"]))
    lab = lab[lab["patientunitstayid"].isin(stay_to_pid)
              & lab["labname"].notna()
              & lab["labresult"].notna()].copy()
    lab["uniquepid"] = lab["patientunitstayid"].map(stay_to_pid)
    lab["stay_los_min"] = lab["patientunitstayid"].map(stay_dur)
    lab["relative_days"] = (lab["labresultoffset"] - lab["stay_los_min"]) / 1440.0

    # Map lab name → MIMIC LAB_* concept
    lab["labname_lower"] = lab["labname"].str.lower()
    lab["concept_id"] = None
    for sub, concept in EICU_LAB_MAP.items():
        cid = f"LAB_{concept}"
        mask = lab["labname_lower"].str.contains(sub, na=False) & lab["concept_id"].isna()
        lab.loc[mask, "concept_id"] = cid
    lab_matched = lab[lab["concept_id"].notna()].copy()
    n_unmatched = int(lab["concept_id"].isna().sum())

    # Sample at LAB_SAMPLE_RATE
    rng = np.random.default_rng(SEED)
    if LAB_SAMPLE_RATE < 1.0:
        keep = rng.random(len(lab_matched)) < LAB_SAMPLE_RATE
        lab_matched = lab_matched[keep]

    print(f"  lab: {len(lab_matched):,} events after {LAB_SAMPLE_RATE:.0%} sample "
          f"({n_unmatched:,} unmatched lab names)")
    lab_matched = lab_matched.rename(columns={"labresult": "value_num"})
    return lab_matched[["uniquepid", "concept_id", "relative_days", "value_num"]]


def _vital_periodic_events(cohort: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Dense ICU vitals from vitalPeriodic (wide → long), sampled at VITAL_PERIODIC_SAMPLE_RATE.

    vitalPeriodic.csv is ~7 GB / 146 M rows — read in 500K-row chunks and
    filter to cohort stays immediately to avoid loading the full file into RAM.

    M6: systemicsystolic/diastolic in eICU is invasive arterial; MIMIC mixes
    invasive and non-invasive.  Mapped to the same VITAL_BP_SYS/DIA concepts.
    """
    vital_cols = list(EICU_VITAL_PERIODIC_MAP.keys())
    stay_to_pid = dict(zip(cohort["index_unitstayid"], cohort["uniquepid"]))
    stay_dur    = dict(zip(cohort["index_unitstayid"], cohort["index_unitdischargeoffset"]))
    cohort_stays = set(stay_to_pid.keys())

    rng = np.random.default_rng(SEED)
    kept_chunks: list[pd.DataFrame] = []
    n_rows_read = 0

    for chunk in pd.read_csv(
        os.path.join(EICU_DATA_DIR, "vitalPeriodic.csv"),
        usecols=["patientunitstayid", "observationoffset"] + vital_cols,
        low_memory=False,
        chunksize=500_000,
    ):
        n_rows_read += len(chunk)
        sub = chunk[chunk["patientunitstayid"].isin(cohort_stays)].copy()
        if sub.empty:
            continue
        # Down-sample before accumulating
        if VITAL_PERIODIC_SAMPLE_RATE < 1.0:
            keep = rng.random(len(sub)) < VITAL_PERIODIC_SAMPLE_RATE
            sub = sub[keep]
        if not sub.empty:
            kept_chunks.append(sub)

    print(f"  vitalPeriodic: read {n_rows_read:,} rows, "
          f"kept {sum(len(c) for c in kept_chunks):,} after filter+sample")

    if not kept_chunks:
        print("  vitalPeriodic: 0 events for cohort stays")
        return pd.DataFrame(columns=["uniquepid", "concept_id", "relative_days", "value_num"])

    vp = pd.concat(kept_chunks, ignore_index=True)
    del kept_chunks

    vp["uniquepid"]     = vp["patientunitstayid"].map(stay_to_pid)
    vp["stay_los_min"]  = vp["patientunitstayid"].map(stay_dur)
    vp["relative_days"] = (vp["observationoffset"] - vp["stay_los_min"]) / 1440.0

    rows = []
    for col, concept_suffix in EICU_VITAL_PERIODIC_MAP.items():
        cid = f"VITAL_{concept_suffix}"
        sub = vp[["uniquepid", "relative_days", col]].dropna(subset=[col]).copy()
        if sub.empty:
            continue
        sub = sub.rename(columns={col: "value_num"})
        sub["concept_id"] = cid
        rows.append(sub[["uniquepid", "concept_id", "relative_days", "value_num"]])

    if not rows:
        print("  vitalPeriodic: 0 events")
        return pd.DataFrame(columns=["uniquepid", "concept_id", "relative_days", "value_num"])
    out = pd.concat(rows, ignore_index=True)
    print(f"  vitalPeriodic: {len(out):,} long-format events")
    return out


def _vital_aperiodic_events(cohort: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Non-invasive BP from vitalAperiodic.

    These map to VITAL_BP_SYS/DIA/MEAN (not OMR_BP_HIGH/NORMAL) because eICU
    BP readings are in ICU context, not outpatient.  OMR_BP_* columns will be
    all-zero for eICU patients — flagged as M1.
    """
    vap = pd.read_csv(
        os.path.join(EICU_DATA_DIR, "vitalAperiodic.csv"),
        usecols=["patientunitstayid", "observationoffset",
                 "noninvasivesystolic", "noninvasivediastolic", "noninvasivemean"],
        low_memory=False,
    )
    stay_to_pid = dict(zip(cohort["index_unitstayid"], cohort["uniquepid"]))
    stay_dur = dict(zip(cohort["index_unitstayid"], cohort["index_unitdischargeoffset"]))
    vap = vap[vap["patientunitstayid"].isin(stay_to_pid)].copy()
    vap["uniquepid"] = vap["patientunitstayid"].map(stay_to_pid)
    vap["stay_los_min"] = vap["patientunitstayid"].map(stay_dur)
    vap["relative_days"] = (vap["observationoffset"] - vap["stay_los_min"]) / 1440.0

    rows = []
    for col, concept_suffix in EICU_VITAL_APERIODIC_MAP.items():
        cid = f"VITAL_{concept_suffix}"
        sub = vap[["uniquepid", "relative_days", col]].dropna(subset=[col]).copy()
        if sub.empty:
            continue
        sub = sub.rename(columns={col: "value_num"})
        sub["concept_id"] = cid
        rows.append(sub[["uniquepid", "concept_id", "relative_days", "value_num"]])
    if not rows:
        return pd.DataFrame(columns=["uniquepid", "concept_id", "relative_days", "value_num"])
    out = pd.concat(rows, ignore_index=True)
    print(f"  vitalAperiodic: {len(out):,} BP events (mapped to VITAL_BP_*, not OMR_BP_*)")
    return out


def _icu_stay_events(cohort: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """ICU unit stays → ICU_STAY concept (value = LOS in days)."""
    bow_col = meta["concept_id_to_bow_col"]
    cid = "ICU_STAY"
    if cid not in bow_col:
        print("  icu_stay: ICU_STAY concept not in MIMIC vocab — skipping")
        return pd.DataFrame(columns=["uniquepid", "concept_id", "relative_days", "value_num"])
    rows = []
    for _, row in cohort.iterrows():
        los_days = row["index_unitdischargeoffset"] / 1440.0
        rows.append({
            "uniquepid":     row["uniquepid"],
            "concept_id":    cid,
            "relative_days": -los_days,   # relative to discharge = 0
            "value_num":     los_days,
        })
    print(f"  icu_stay: {len(rows):,} ICU_STAY events")
    return pd.DataFrame(rows)


def _infusion_events(cohort: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """IV infusion drugs → INPUT_VASOPRESSOR / INPUT_IV_INSULIN etc."""
    inf = pd.read_csv(
        os.path.join(EICU_DATA_DIR, "infusionDrug.csv"),
        usecols=["patientunitstayid", "infusionoffset", "drugname", "drugamount"],
        low_memory=False,
    )
    stay_to_pid = dict(zip(cohort["index_unitstayid"], cohort["uniquepid"]))
    stay_dur = dict(zip(cohort["index_unitstayid"], cohort["index_unitdischargeoffset"]))
    inf = inf[inf["patientunitstayid"].isin(stay_to_pid) & inf["drugname"].notna()].copy()
    inf["uniquepid"] = inf["patientunitstayid"].map(stay_to_pid)
    inf["stay_los_min"] = inf["patientunitstayid"].map(stay_dur)
    inf["relative_days"] = (inf["infusionoffset"] - inf["stay_los_min"]) / 1440.0
    inf["drugname_lower"] = inf["drugname"].str.lower()

    bow_col = meta["concept_id_to_bow_col"]
    rows = []
    n_unmatched = 0
    for _, r in inf.iterrows():
        matched_cid = None
        for concept, terms in EICU_INFUSION_MAP.items():
            if any(t in r["drugname_lower"] for t in terms):
                cid = f"INPUT_{concept}"
                if cid in bow_col:
                    matched_cid = cid
                break
        if matched_cid:
            rows.append({
                "uniquepid":     r["uniquepid"],
                "concept_id":    matched_cid,
                "relative_days": r["relative_days"],
                "value_num":     r.get("drugamount", np.nan),
            })
        else:
            n_unmatched += 1
    print(f"  infusion: {len(rows):,} matched, {n_unmatched:,} unmatched infusion drugs")
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["uniquepid", "concept_id", "relative_days", "value_num"])


def _output_events(cohort: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Urine / drainage from intakeOutput."""
    io = pd.read_csv(
        os.path.join(EICU_DATA_DIR, "intakeOutput.csv"),
        usecols=["patientunitstayid", "intakeoutputoffset", "celllabel", "cellvaluenumeric"],
        low_memory=False,
    )
    stay_to_pid = dict(zip(cohort["index_unitstayid"], cohort["uniquepid"]))
    stay_dur = dict(zip(cohort["index_unitstayid"], cohort["index_unitdischargeoffset"]))
    io = io[io["patientunitstayid"].isin(stay_to_pid)
            & io["celllabel"].notna()
            & io["cellvaluenumeric"].notna()].copy()
    io["uniquepid"] = io["patientunitstayid"].map(stay_to_pid)
    io["stay_los_min"] = io["patientunitstayid"].map(stay_dur)
    io["relative_days"] = (io["intakeoutputoffset"] - io["stay_los_min"]) / 1440.0
    io["celllabel_lower"] = io["celllabel"].str.lower()

    bow_col = meta["concept_id_to_bow_col"]
    rows = []
    n_unmatched = 0
    for _, r in io.iterrows():
        matched_cid = None
        for concept, terms in EICU_OUTPUT_MAP.items():
            if any(t in r["celllabel_lower"] for t in terms):
                cid = f"OUTPUT_{concept}"
                if cid in bow_col:
                    matched_cid = cid
                break
        if matched_cid:
            rows.append({
                "uniquepid":     r["uniquepid"],
                "concept_id":    matched_cid,
                "relative_days": r["relative_days"],
                "value_num":     r["cellvaluenumeric"],
            })
        else:
            n_unmatched += 1
    print(f"  output: {len(rows):,} matched, {n_unmatched:,} unmatched output labels")
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["uniquepid", "concept_id", "relative_days", "value_num"])


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features_eicu(
    cohort: pd.DataFrame,
    feature_meta: dict,
) -> tuple[csr_matrix, list[str], dict]:
    """Build the transfer feature matrix X for eICU patients.

    Returns
    -------
    X : csr_matrix, shape (n_patients, n_mimic_features)
        Same column layout as the MIMIC training matrix.
    patient_order : list[str]
        uniquepid values in the row order of X.
    mismatch_report : dict
        Counts and descriptions of feature mismatches (M1–M6).
    """
    print("\n=== eICU FEATURE EXTRACTION ===")
    print("MISMATCH FLAGS:")
    print("  M1: OMR_BP_*/OMR_BMI_* (MIMIC outpatient) → all-zero for eICU patients")
    print("  M2: had_icu_stay mapped to had_prior_icu_stay (different clinical meaning)")
    print("  M3: Drug concept IDs use name-prefix matching; unmatched drugs → 0")
    print("  M4: Pre-index window = ICU stay only (~1–3 days vs. MIMIC's 5 years)")
    print("  M5: ICD code dots stripped to match MIMIC concept ID format")
    print("  M6: vitalPeriodic invasive BP mapped to same VITAL_BP_* as MIMIC chartevent BP\n")

    patient_order = cohort["uniquepid"].tolist()
    n_pts = len(patient_order)
    pid_to_row = {pid: i for i, pid in enumerate(patient_order)}

    n_bow = feature_meta["n_bow_cols"]
    n_val = feature_meta["n_value_cols"]
    n_sta = feature_meta["n_static_cols"]
    bow_col = feature_meta["concept_id_to_bow_col"]

    # -----------------------------------------------------------------------
    # 1. Collect all events across modalities
    # -----------------------------------------------------------------------
    print("[events] Extracting per-modality events...")
    all_events = pd.concat([
        _diagnosis_events(cohort, feature_meta),
        _medication_events(cohort, feature_meta),
        _lab_events(cohort, feature_meta),
        _vital_periodic_events(cohort, feature_meta),
        _vital_aperiodic_events(cohort, feature_meta),
        _icu_stay_events(cohort, feature_meta),
        _infusion_events(cohort, feature_meta),
        _output_events(cohort, feature_meta),
    ], ignore_index=True)

    # Keep only events that map to a MIMIC concept
    all_events = all_events[all_events["concept_id"].isin(bow_col)].copy()
    print(f"\n  Total events mapping to MIMIC concepts: {len(all_events):,}")
    print(f"  Patients with ≥1 event: {all_events['uniquepid'].nunique():,} / {n_pts:,}")

    per_pt = all_events.groupby("uniquepid").size()
    if len(per_pt):
        print(f"  Events/patient: median={int(per_pt.median())}, "
              f"p25={int(per_pt.quantile(0.25))}, p75={int(per_pt.quantile(0.75))}, "
              f"max={int(per_pt.max())}")

    # -----------------------------------------------------------------------
    # 2. Bag-of-codes  (n_pts × n_bow)
    # -----------------------------------------------------------------------
    print("\n[bow] Building bag-of-codes...")
    bow_data = (all_events.groupby(["uniquepid", "concept_id"])
                .size().reset_index(name="count"))
    rows_bow, cols_bow, data_bow = [], [], []
    for _, r in bow_data.iterrows():
        if r["uniquepid"] not in pid_to_row:
            continue
        ri = pid_to_row[r["uniquepid"]]
        ci = bow_col[r["concept_id"]]
        rows_bow.append(ri); cols_bow.append(ci); data_bow.append(r["count"])
    X_bow = csr_matrix(
        (np.array(data_bow, dtype=np.float32),
         (np.array(rows_bow), np.array(cols_bow))),
        shape=(n_pts, n_bow),
    )
    print(f"  bow: shape={X_bow.shape}, nnz={X_bow.nnz:,}")

    # -----------------------------------------------------------------------
    # 3. Value summary features  (n_pts × n_val)
    # -----------------------------------------------------------------------
    print("\n[values] Building value summary features...")
    feat_concept_ids_str = feature_meta["feat_concept_ids_str"]
    val_mu = feature_meta["value_mu"]
    val_sd = feature_meta["value_sd"]

    if n_val > 0 and feat_concept_ids_str:
        # Build per-patient summary for each MIMIC value concept
        all_events_vals = all_events[all_events["value_num"].notna()].copy()
        arr_val = np.zeros((n_pts, n_val), dtype=np.float32)
        cid_to_val_col = {c: j for j, c in enumerate(feat_concept_ids_str)}

        for (pid, cid), grp in all_events_vals[
            all_events_vals["concept_id"].isin(cid_to_val_col)
        ].groupby(["uniquepid", "concept_id"]):
            if pid not in pid_to_row:
                continue
            i = pid_to_row[pid]
            j = cid_to_val_col[cid]
            base = j * len(STAT_NAMES)
            vals = grp["value_num"].to_numpy(dtype=np.float64)
            days = grp["relative_days"].to_numpy(dtype=np.float64)
            arr_val[i, base + 0] = vals.mean()
            arr_val[i, base + 1] = vals.max()
            arr_val[i, base + 2] = vals.min()
            arr_val[i, base + 3] = vals[-1]
            arr_val[i, base + 4] = len(vals)
            arr_val[i, base + 5] = _slope(days, vals)

        # Apply MIMIC training normalisation (mu/sd computed on MIMIC train set only)
        if len(val_mu) == n_val:
            sd_safe = np.where(val_sd == 0, 1.0, val_sd)
            arr_val = (arr_val - val_mu) / sd_safe
            arr_val = np.clip(arr_val, -10.0, 10.0).astype(np.float32)
        else:
            print(f"  WARNING: value_mu length {len(val_mu)} != n_val {n_val}; "
                  f"skipping normalisation")
    else:
        arr_val = np.zeros((n_pts, n_val), dtype=np.float32)
    X_val = csr_matrix(arr_val)
    print(f"  value summary: shape={X_val.shape}")

    # -----------------------------------------------------------------------
    # 4. Static features  (n_pts × n_static)
    # -----------------------------------------------------------------------
    print("\n[static] Building static features...")
    static_cols = feature_meta["static_cols"]
    static_mu   = feature_meta["static_mu"]
    static_sd   = feature_meta["static_sd"]

    # eICU→MIMIC static mapping
    # age_at_index → age (direct; eICU caps at 89)
    # cci_score    → cci_score (computed by cohort_eicu.py)
    # num_cardiometa_conditions → num_cardiometa_conditions
    # had_icu_stay → had_prior_icu_stay (M2)
    # female       → female
    arr_sta = np.zeros((n_pts, len(static_cols)), dtype=np.float32)
    col_map = {c: i for i, c in enumerate(static_cols)}
    eicu_map = {
        "age_at_index":              "age",
        "cci_score":                 "cci_score",
        "num_cardiometa_conditions": "num_cardiometa_conditions",
        "had_icu_stay":              "had_prior_icu_stay",   # M2
        "female":                    "female",
    }
    cohort_idx = cohort.set_index("uniquepid")
    for i, pid in enumerate(patient_order):
        row = cohort_idx.loc[pid] if pid in cohort_idx.index else None
        if row is None:
            continue
        for mimic_col, eicu_col in eicu_map.items():
            if mimic_col in col_map and eicu_col in cohort.columns:
                val = row[eicu_col] if not isinstance(row, pd.DataFrame) else row[eicu_col].iloc[0]
                arr_sta[i, col_map[mimic_col]] = float(val) if pd.notna(val) else 0.0
    sd_safe = np.where(static_sd == 0, 1.0, static_sd)
    arr_sta = (arr_sta - static_mu) / sd_safe
    arr_sta = np.clip(arr_sta, -10.0, 10.0).astype(np.float32)
    X_sta = csr_matrix(arr_sta)
    print(f"  static: shape={X_sta.shape}")

    # -----------------------------------------------------------------------
    # 5. Assemble and return
    # -----------------------------------------------------------------------
    X = hstack([X_bow, X_val, X_sta]).tocsr()
    print(f"\n  Final X: shape={X.shape}, nnz={X.nnz:,}")

    mismatch_report = {
        "M1_OMR_BP_BMI_all_zero": True,
        "M2_had_icu_stay_proxy": True,
        "M3_drug_name_prefix_matching": True,
        "M4_preindex_window_days_median": float(
            cohort["index_unitdischargeoffset"].median() / 1440.0),
        "M5_ICD_dot_stripping": True,
        "M6_invasive_BP_mapped_to_VITAL_BP": True,
        "n_eicu_patients": n_pts,
        "n_patients_with_any_event": int(all_events["uniquepid"].nunique()),
        "n_total_matched_events": len(all_events),
        "bow_nnz": X_bow.nnz,
    }
    return X, patient_order, mismatch_report

"""Build the temporal knowledge graph fact table from EHRSHOT.

Emits tkg_facts.csv / node_index.csv in exactly the MIMIC schema (FACT_COLS,
relation names, concept-ID string scheme) through src.build_tkg.finalize_facts,
so prep_modeling.py and every model downstream run unchanged.

Concept-ID mapping (EHRSHOT code -> MIMIC-style concept):
  ICD10CM/I25.10        -> ICD10_I2510      (hasFinalDx if hospital admission dx
                                             on an inpatient visit, else hasDiagnosis)
  ICD9CM/401.9          -> ICD9_4019
  ICD10PCS/02703ZZ      -> ICD10_PROC_02703ZZ
  ICD9Proc/36.06        -> ICD9_PROC_3606
  CPT4/93000            -> CPT4_93000        (EHRSHOT-only; MIMIC has no CPT)
  RxNorm/314200         -> DRUG_METOPROLOL_TARTRATE   (ingredient words of the
                                             RxNorm description, as MIMIC
                                             prescriptions.drug is written)
  LOINC/2160-0          -> LAB_CREATININE    (config_ehrshot.LOINC_TO_LAB)
  LOINC/8867-4 (acute)  -> VITAL_HEART_RATE  (inpatient / ICU / ED encounters only,
                                             as MIMIC vitals are ICU chartevents)
  BP outside acute care -> OMR_BP_HIGH / OMR_BP_NORMAL (as MIMIC omr)
  LOINC/39156-5         -> OMR_BMI_OBESE / _OVERWEIGHT / _NORMAL
  ICU encounter         -> ICU_STAY (value = LOS days, suppressed if still open
                                             at window close)
There are no IV-input or urine-output facts (EHRSHOT has no equivalent table).
"""
import gc

import numpy as np
import pandas as pd

from src.build_tkg import FACT_COLS, _load_cohort_windows, finalize_facts
from src.ehrshot import data_ehrshot as D
from src.ehrshot.config_ehrshot import (
    SEED, LAB_SAMPLE_RATE, VITAL_SAMPLE_RATE,
    LOINC_TO_LAB, UNIT_RESCALE, UNIT_DROP, LOINC_TO_VITAL,
    LOINC_BP_SYS, LOINC_BP_DIA, LOINC_BMI,
)


def _in_window(df: pd.DataFrame, windows: pd.DataFrame, time_col: str = "time",
               inclusive_end: bool = False) -> pd.DataFrame:
    out = df.merge(windows[["subject_id", "window_start", "window_end"]],
                   on="subject_id", how="inner")
    end_ok = (out[time_col] <= out["window_end"]) if inclusive_end \
        else (out[time_col] < out["window_end"])
    return out[(out[time_col] >= out["window_start"]) & end_ok]


def _finish(df: pd.DataFrame, **const) -> pd.DataFrame:
    for k, v in const.items():
        df[k] = v
    for c in FACT_COLS:
        if c not in df.columns:
            df[c] = pd.NaT if c.startswith("timestamp") else np.nan
    return df[FACT_COLS].reset_index(drop=True)


def _diagnosis_facts(windows, subjects, **kw):
    print("\n[3] Diagnosis facts (ICD10CM / ICD9CM)...")
    dx = D.load_diagnoses(subjects)
    dx = _in_window(dx, windows, **kw)
    dx["concept_id"] = "ICD" + dx["icd_version"].astype(str) + "_" + dx["icd_code"]
    dx["relation"] = np.where(dx["seq_num"] == 1, "hasFinalDx", "hasDiagnosis")
    dx["source"] = dx["icd_version"].astype(str)
    dx["timestamp_start"] = dx["time"]
    out = _finish(dx, fact_type="diagnosis")
    print(f"  {len(out):,} (hasFinalDx {(out['relation'] == 'hasFinalDx').sum():,})")
    return out


def _procedure_facts(windows, subjects, **kw):
    print("\n[4] Procedure facts (ICD10PCS / ICD9Proc / CPT4)...")
    pr = D.load_rows(["ICD10PCS/", "ICD9Proc/", "CPT4/"], [], subjects)
    pr = _in_window(pr, windows, **kw)
    vocab = pr["code"].str.split("/", n=1).str[0]
    code = pr["code"].str.split("/", n=1).str[1].str.replace(".", "", regex=False).str.strip()
    pr["concept_id"] = np.select(
        [vocab == "ICD10PCS", vocab == "ICD9Proc"],
        ["ICD10_PROC_" + code, "ICD9_PROC_" + code], default="CPT4_" + code)
    pr["source"] = np.select([vocab == "ICD10PCS", vocab == "ICD9Proc"], ["10", "9"], default="cpt4")
    pr["timestamp_start"] = pr["time"]
    out = _finish(pr, relation="hasProcedure", fact_type="procedure")
    print(f"  {len(out):,}")
    return out


def _prescription_facts(windows, subjects, **kw):
    print("\n[5] Prescription facts (RxNorm)...")
    rx = D.load_rows(["RxNorm/"], ["end"], subjects)
    rx = _in_window(rx, windows, **kw)
    desc = D.rxnorm_descriptions()
    cmap = {c: D.rxnorm_to_drug_concept(c, desc) for c in rx["code"].unique()}
    rx["concept_id"] = rx["code"].map(cmap)
    n_fallback = rx["concept_id"].str.startswith("DRUG_RXNORM_").sum()
    rx["timestamp_start"] = rx["time"]
    rx["timestamp_end"] = rx["end"].where(rx["end"] > rx["time"])
    out = _finish(rx, relation="hasPrescription", fact_type="prescription", source="drug")
    print(f"  {len(out):,} ({len(set(cmap.values())):,} drug concepts; "
          f"{n_fallback / max(len(rx), 1) * 100:.1f}% of rows lack an RxNorm description)")
    return out


def _lab_facts(windows, subjects, **kw):
    print("\n[6] Lab facts (LOINC -> LAB_*)...")
    lab = D.load_rows(["LOINC/" + c for c in LOINC_TO_LAB], ["numeric_value", "unit"], subjects)
    lab = lab[lab["numeric_value"].notna()]
    lab = lab[lab["code"].str[6:].isin(LOINC_TO_LAB)]   # exact LOINC match, not prefix
    lab = _in_window(lab, windows, **kw)
    lab["concept"] = lab["code"].str[6:].map(LOINC_TO_LAB)
    unit = lab["unit"].fillna("").str.strip().str.lower()
    keys = list(zip(lab["concept"], unit))
    drop = np.array([k in UNIT_DROP for k in keys])
    scale = np.array([UNIT_RESCALE.get(k, 1.0) for k in keys])
    lab["value_num"] = lab["numeric_value"].astype(float) * scale
    lab = lab[~drop]
    if LAB_SAMPLE_RATE < 1.0:
        lab = lab[np.random.default_rng(SEED).random(len(lab)) < LAB_SAMPLE_RATE]
    lab["concept_id"] = "LAB_" + lab["concept"]
    lab["timestamp_start"] = lab["time"]
    out = _finish(lab, relation="hasLabEvent", fact_type="lab", source="labevents")
    print(f"  {len(out):,} (dropped {int(drop.sum()):,} incompatible-unit rows)")
    return out


def _vital_and_omr_facts(windows, subjects, visits, **kw):
    print("\n[7] Vital (acute care) and OMR-style BP / BMI facts...")
    codes = ["LOINC/" + c for c in LOINC_TO_VITAL] + ["LOINC/" + LOINC_BMI]
    v = D.load_rows(codes, ["numeric_value", "visit_id"], subjects)
    v = v[v["numeric_value"].notna()]
    v["loinc"] = v["code"].str[6:]
    v = v[v["loinc"].isin(set(LOINC_TO_VITAL) | {LOINC_BMI})]
    v = _in_window(v, windows, **kw)
    acute_ids = set(visits.loc[visits["is_acute"], "visit_id"])
    v["acute"] = v["visit_id"].isin(acute_ids)

    # Vitals: acute encounters only (MIMIC vitals = ICU chartevents)
    vit = v[v["acute"] & v["loinc"].isin(LOINC_TO_VITAL)].copy()
    if VITAL_SAMPLE_RATE < 1.0:
        vit = vit[np.random.default_rng(SEED).random(len(vit)) < VITAL_SAMPLE_RATE]
    vit["concept_id"] = "VITAL_" + vit["loinc"].map(LOINC_TO_VITAL)
    vit["timestamp_start"] = vit["time"]
    vit["value_num"] = vit["numeric_value"].astype(float)
    vit_out = _finish(vit, relation="hasVital", fact_type="vital", source="chartevents")

    # OMR BP: outpatient systolic/diastolic pairs at the same timestamp
    bp = v[~v["acute"] & v["loinc"].isin([LOINC_BP_SYS, LOINC_BP_DIA])]
    bp = (bp.pivot_table(index=["subject_id", "time"], columns="loinc",
                         values="numeric_value", aggfunc="first").reset_index())
    bp = bp.rename(columns={LOINC_BP_SYS: "sys", LOINC_BP_DIA: "dia"})
    if "dia" not in bp.columns:
        bp["dia"] = np.nan
    bp = bp.dropna(subset=["sys"]) if "sys" in bp.columns else bp.iloc[0:0]
    high = (bp["sys"] >= 140) | (bp["dia"] >= 90)        # same JNC7 rule as build_tkg
    bp["concept_id"] = np.where(high, "OMR_BP_HIGH", "OMR_BP_NORMAL")
    bp["timestamp_start"] = bp["time"]
    bp["value_num"] = bp["sys"].astype(float)
    bp_out = _finish(bp, relation="hasBP", fact_type="omr_bp", source="omr")

    bmi = v[v["loinc"] == LOINC_BMI].copy()
    bmi = bmi[(bmi["numeric_value"] > 8) & (bmi["numeric_value"] < 120)]
    bmi["concept_id"] = np.select(
        [bmi["numeric_value"] >= 30, bmi["numeric_value"] >= 25],
        ["OMR_BMI_OBESE", "OMR_BMI_OVERWEIGHT"], default="OMR_BMI_NORMAL")
    bmi["timestamp_start"] = bmi["time"]
    bmi["value_num"] = bmi["numeric_value"].astype(float)
    bmi_out = _finish(bmi, relation="hasBMI", fact_type="omr_bmi", source="omr")
    print(f"  vitals {len(vit_out):,} | OMR BP {len(bp_out):,} | BMI {len(bmi_out):,}")
    return [vit_out, bp_out, bmi_out]


def _icu_facts(windows, visits, **kw):
    print("\n[8] ICU stay facts...")
    icu = visits[visits["is_icu"]][["subject_id", "visit_id", "time", "end"]].copy()
    icu = _in_window(icu, windows, **kw)
    los = (icu["end"] - icu["time"]).dt.total_seconds() / 86400.0
    icu["value_num"] = los.where(los >= 0)
    late = icu["end"].isna() | (icu["end"] >= icu["window_end"])
    icu.loc[late, "value_num"] = np.nan
    icu["timestamp_start"] = icu["time"]
    icu["timestamp_end"] = icu["end"]
    out = _finish(icu, relation="hasICUStay", fact_type="icu", source="icu", concept_id="ICU_STAY")
    print(f"  {len(out):,}")
    return out


def build_fact_pieces(windows: pd.DataFrame, inclusive_end: bool = False) -> list[pd.DataFrame]:
    """All EHRSHOT facts for windows[subject_id, window_start, window_end].
    Shared with src/ehrshot/benchmark_binary.py (which uses inclusive_end=True,
    because EHRSHOT label times are 23:59 and data that day is admissible)."""
    subjects = windows["subject_id"].unique().tolist()
    visits = D.visit_table()
    kw = dict(inclusive_end=inclusive_end)
    pieces = [_diagnosis_facts(windows, subjects, **kw)]; gc.collect()
    pieces.append(_procedure_facts(windows, subjects, **kw)); gc.collect()
    pieces.append(_prescription_facts(windows, subjects, **kw)); gc.collect()
    pieces.append(_lab_facts(windows, subjects, **kw)); gc.collect()
    pieces.extend(_vital_and_omr_facts(windows, subjects, visits, **kw)); gc.collect()
    pieces.append(_icu_facts(windows, visits, **kw)); gc.collect()
    return [p for p in pieces if len(p)]


def build_tkg() -> pd.DataFrame:
    print("[1] Loading EHRSHOT cohort and computing per-patient windows...")
    cohort, windows = _load_cohort_windows()
    print(f"  cohort patients: {len(cohort):,}")
    return finalize_facts(build_fact_pieces(windows), cohort)


if __name__ == "__main__":
    build_tkg()

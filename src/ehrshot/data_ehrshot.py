"""Readers for meds_omop_ehrshot, shared by the EHRSHOT cohort / TKG builders."""
import functools
import json
import os
import re

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from src.ehrshot.config_ehrshot import (
    EHRSHOT_DATA_DIR, INPATIENT_STANFORD_CLASSES, INPATIENT_VISIT_CODES,
    ICU_VISIT_CODES, ACUTE_STANFORD_CLASSES, ACUTE_VISIT_CODES,
    ADMIT_DX_TABLES, CLMBR_TOKEN_2_CODE, CLMBR_TOKEN_2_DESC,
)


@functools.lru_cache(maxsize=1)
def _dataset() -> ds.Dataset:
    return ds.dataset(os.path.join(EHRSHOT_DATA_DIR, "data"), format="parquet")


def load_rows(prefixes, columns, subjects=None) -> pd.DataFrame:
    """All rows whose `code` starts with any of `prefixes` (pushed down to the
    parquet scan), optionally restricted to a set of subject_ids."""
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    expr = None
    for p in prefixes:
        e = pc.starts_with(pc.field("code"), p)
        expr = e if expr is None else (expr | e)
    if subjects is not None:
        expr = expr & pc.field("subject_id").isin(list(map(int, subjects)))
    cols = list(dict.fromkeys(["subject_id", "time", "code", *columns]))
    df = _dataset().to_table(columns=cols, filter=expr).to_pandas()
    for c in ("visit_id", "clarity_table", "unit"):
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df


def all_subjects() -> np.ndarray:
    return np.unique(_dataset().to_table(columns=["subject_id"])["subject_id"].to_numpy())


def last_event_time() -> pd.Series:
    """Per-patient time of the last recorded event of any kind (censoring)."""
    t = _dataset().to_table(columns=["subject_id", "time"]).to_pandas()
    return t.groupby("subject_id")["time"].max()


@functools.lru_cache(maxsize=1)
def visit_table() -> pd.DataFrame:
    """One row per visit_id with its Stanford class, OMOP visit code, start and
    end, plus is_inpatient / is_icu / is_acute flags."""
    v = load_rows(["STANFORD_VISIT/", "Visit/"], ["end", "visit_id"])
    v = v[v["visit_id"].notna()]
    sv = v[v["code"].str.startswith("STANFORD_VISIT/")].copy()
    sv["stanford_class"] = sv["code"].str.split("/", n=1).str[1].str.split("|").str[0]
    sv = sv.sort_values("time").drop_duplicates("visit_id")[
        ["visit_id", "subject_id", "time", "end", "stanford_class"]]
    ov = (v[v["code"].str.startswith("Visit/")]
          .groupby("visit_id")["code"].agg(lambda s: set(s)).rename("visit_codes"))
    ov_time = (v[v["code"].str.startswith("Visit/")]
               .groupby("visit_id").agg(o_subject=("subject_id", "first"),
                                        o_time=("time", "min"), o_end=("end", "max")))
    out = sv.merge(ov.reset_index(), on="visit_id", how="outer").merge(
        ov_time.reset_index(), on="visit_id", how="left")
    out["subject_id"] = out["subject_id"].fillna(out["o_subject"]).astype("int64")
    out["time"] = out["time"].fillna(out["o_time"])
    out["end"] = out["end"].fillna(out["o_end"])
    out["visit_codes"] = out["visit_codes"].apply(lambda s: s if isinstance(s, set) else set())
    out["stanford_class"] = out["stanford_class"].fillna("")
    out["is_inpatient"] = (out["stanford_class"].isin(INPATIENT_STANFORD_CLASSES)
                           | out["visit_codes"].apply(lambda s: bool(s & INPATIENT_VISIT_CODES)))
    out["is_icu"] = out["visit_codes"].apply(lambda s: bool(s & ICU_VISIT_CODES))
    out["is_acute"] = (out["is_inpatient"]
                       | out["stanford_class"].isin(ACUTE_STANFORD_CLASSES)
                       | out["visit_codes"].apply(lambda s: bool(s & ACUTE_VISIT_CODES)))
    return out[["visit_id", "subject_id", "time", "end", "stanford_class",
                "is_inpatient", "is_icu", "is_acute"]].reset_index(drop=True)


def load_diagnoses(subjects=None) -> pd.DataFrame:
    """ICD10CM / ICD9CM diagnosis rows in the MIMIC diagnoses_icd shape:
    subject_id, hadm_id (= visit_id), time, icd_code (no dots), icd_version,
    seq_num (1 = admission diagnosis on an inpatient visit, else 2),
    on_inpatient_visit, clarity_table."""
    dx = load_rows(["ICD10CM/", "ICD9CM/"], ["visit_id", "clarity_table"], subjects)
    dx["icd_version"] = np.where(dx["code"].str.startswith("ICD10CM/"), 10, 9)
    dx["icd_code"] = (dx["code"].str.split("/", n=1).str[1]
                      .str.replace(".", "", regex=False).str.strip().str.upper())
    vt = visit_table().set_index("visit_id")
    dx["on_inpatient_visit"] = dx["visit_id"].isin(set(vt.index[vt["is_inpatient"]]))
    is_admit_dx = dx["clarity_table"].isin(ADMIT_DX_TABLES).fillna(False).astype(bool)
    dx["seq_num"] = np.where(is_admit_dx & dx["on_inpatient_visit"], 1, 2)
    dx["is_admit_dx"] = is_admit_dx
    dx = dx.rename(columns={"visit_id": "hadm_id"})
    return dx[["subject_id", "hadm_id", "time", "icd_code", "icd_version",
               "seq_num", "on_inpatient_visit", "is_admit_dx", "clarity_table"]]


def load_demographics() -> pd.DataFrame:
    """subject_id, birth, gender ('F'/'M'), dod."""
    p = load_rows(["MEDS_BIRTH", "MEDS_DEATH", "Gender/"], [])
    birth = p[p["code"] == "MEDS_BIRTH"].groupby("subject_id")["time"].min().rename("birth")
    dod = p[p["code"] == "MEDS_DEATH"].groupby("subject_id")["time"].min().rename("dod")
    g = p[p["code"].str.startswith("Gender/")]
    gender = g.groupby("subject_id")["code"].first().str.split("/").str[1].rename("gender")
    return pd.concat([birth, gender, dod], axis=1).reset_index().rename(
        columns={"index": "subject_id"})


@functools.lru_cache(maxsize=1)
def rxnorm_descriptions() -> dict:
    """RxNorm code -> description, from the CLMBR vocabulary files."""
    with open(CLMBR_TOKEN_2_CODE) as f:
        t2c = json.load(f)
    with open(CLMBR_TOKEN_2_DESC) as f:
        t2d = json.load(f)
    return {code: t2d[tok] for tok, code in t2c.items()
            if code.startswith("RxNorm/") and t2d.get(tok)}


_DOSE_STOP = {"MG", "MCG", "MEQ", "ML", "GM", "IU", "UNIT", "UNITS", "ORAL",
              "INJECTABLE", "INJECTION", "TOPICAL", "TABLET", "CAPSULE",
              "SOLUTION", "SUSPENSION", "EXTENDED", "DELAYED", "PREFILLED",
              "METERED", "NASAL", "OPHTHALMIC", "RECTAL", "TRANSDERMAL",
              "AUTO-INJECTOR", "PEN", "SYRINGE", "CARTRIDGE", "PACK", "KIT",
              "CREAM", "OINTMENT", "GEL", "PATCH", "INHALANT", "INHALATION",
              "POWDER", "SPRAY", "DROPS", "CHEWABLE", "DISINTEGRATING"}


def rxnorm_to_drug_concept(code: str, desc_map: dict) -> str:
    """'RxNorm/314200' -> 'DRUG_METOPROLOL_TARTRATE' (MIMIC prescriptions style:
    ingredient words before the first dose / form token). Falls back to
    'DRUG_RXNORM_<id>' when the description is missing or unparseable."""
    desc = desc_map.get(code)
    rx_id = code.split("/", 1)[1]
    if not desc:
        return f"DRUG_RXNORM_{rx_id}"
    # Packs look like "{7 (drug A 1 MG Oral Tablet) / ...} Pack": take the first component.
    m = re.search(r"\(([^()]+)\)", desc) if desc.lstrip().startswith("{") else None
    if m:
        desc = m.group(1)
    toks = []
    for tok in desc.strip().upper().split():
        if re.match(r"^[\d.]", tok) or tok in _DOSE_STOP:
            break
        toks.append(tok.strip(",;"))
    if not toks:
        return f"DRUG_RXNORM_{rx_id}"
    return "DRUG_" + "_".join(toks).replace("/", "_")

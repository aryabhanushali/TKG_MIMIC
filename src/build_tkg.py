"""Build the temporal knowledge graph fact table from MIMIC-IV."""
import os
import gc
import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR, OUTPUT_DIR,
    CARDIOMETA_LAB_LABELS,
    CARDIOMETA_CHART_LABELS,
    CARDIOMETA_INPUT_LABELS,
    CARDIOMETA_OUTPUT_LABELS,
    PRE_INDEX_WINDOW_DAYS, LAB_SAMPLE_RATE, CHART_SAMPLE_RATE, SEED,
)


FACT_COLS = ["subject_id", "relation", "concept_id",
             "timestamp_start", "timestamp_end", "fact_type", "source",
             "value_num"]


def _load_cohort_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cohort and build per-patient time windows."""
    cohort = pd.read_csv(
        os.path.join(OUTPUT_DIR, "cohort.csv"),
        parse_dates=["index_date", "endpoint_date"],
    )
    # window_start = index_date - 365 days
    cohort["window_start"] = cohort["index_date"] - pd.Timedelta(days=PRE_INDEX_WINDOW_DAYS)
    # window_end: endpoint_date if endpoint, else index_date + follow_up_days
    cohort["window_end"] = cohort["endpoint_date"]
    censored = cohort["endpoint_type"] == "censored"
    cohort.loc[censored, "window_end"] = (
        cohort.loc[censored, "index_date"]
        + pd.to_timedelta(cohort.loc[censored, "follow_up_days"], unit="D")
    )
    windows = cohort[["subject_id", "index_date", "window_start", "window_end"]].copy()
    return cohort, windows


def _filter_to_window(df: pd.DataFrame, windows: pd.DataFrame,
                      time_col: str) -> pd.DataFrame:
    """Inner-join df to windows on subject_id and filter by per-patient time window."""
    out = df.merge(windows, on="subject_id", how="inner")
    out = out[(out[time_col] >= out["window_start"])
              & (out[time_col] < out["window_end"])]
    return out


def _diagnosis_facts(windows: pd.DataFrame) -> pd.DataFrame:
    print("\n[3] Building diagnosis facts...")
    dx = pd.read_csv(
        os.path.join(DATA_DIR, "diagnoses_icd.csv.gz"),
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        low_memory=False,
    )
    dx = dx[dx["subject_id"].isin(set(windows["subject_id"]))].copy()
    adm = pd.read_csv(
        os.path.join(DATA_DIR, "admissions.csv.gz"),
        usecols=["subject_id", "hadm_id", "admittime"],
        parse_dates=["admittime"], low_memory=False,
    )
    dx = dx.merge(adm, on=["subject_id", "hadm_id"], how="inner")
    dx = _filter_to_window(dx, windows, "admittime")
    dx["icd_code"] = dx["icd_code"].astype(str).str.strip()
    dx["concept_id"] = "ICD" + dx["icd_version"].astype(str) + "_" + dx["icd_code"]
    dx["relation"] = np.where(dx["seq_num"] == 1, "hasFinalDx", "hasDiagnosis")
    dx["fact_type"] = "diagnosis"
    dx["source"] = dx["icd_version"].astype(str)
    dx["timestamp_start"] = dx["admittime"]
    dx["timestamp_end"] = pd.NaT
    dx["value_num"] = np.nan
    out = dx[FACT_COLS]
    print(f"  Diagnosis facts: {len(out):,} "
          f"(hasFinalDx: {(out['relation']=='hasFinalDx').sum():,}, "
          f"hasDiagnosis: {(out['relation']=='hasDiagnosis').sum():,})")
    return out


def _procedure_facts(windows: pd.DataFrame) -> pd.DataFrame:
    print("\n[4] Building procedure facts...")
    pr = pd.read_csv(
        os.path.join(DATA_DIR, "procedures_icd.csv.gz"),
        usecols=["subject_id", "hadm_id", "chartdate", "icd_code", "icd_version"],
        parse_dates=["chartdate"], low_memory=False,
    )
    pr = pr[pr["subject_id"].isin(set(windows["subject_id"]))].copy()
    adm = pd.read_csv(
        os.path.join(DATA_DIR, "admissions.csv.gz"),
        usecols=["subject_id", "hadm_id", "admittime"],
        parse_dates=["admittime"], low_memory=False,
    )
    pr = pr.merge(adm, on=["subject_id", "hadm_id"], how="left")
    pr["timestamp_start"] = pr["chartdate"].fillna(pr["admittime"])
    pr = _filter_to_window(pr, windows, "timestamp_start")
    pr["icd_code"] = pr["icd_code"].astype(str).str.strip()
    pr["concept_id"] = "ICD" + pr["icd_version"].astype(str) + "_PROC_" + pr["icd_code"]
    pr["relation"] = "hasProcedure"
    pr["fact_type"] = "procedure"
    pr["source"] = pr["icd_version"].astype(str)
    pr["timestamp_end"] = pd.NaT
    pr["value_num"] = np.nan
    out = pr[FACT_COLS]
    print(f"  Procedure facts: {len(out):,}")
    return out


def _prescription_facts(windows: pd.DataFrame) -> pd.DataFrame:
    print("\n[5] Building prescription facts (chunked)...")
    rx_chunks = []
    cohort_ids = set(windows["subject_id"])
    win_by_id = windows.set_index("subject_id")[["window_start", "window_end"]]
    chunksize = 500_000
    reader = pd.read_csv(
        os.path.join(DATA_DIR, "prescriptions.csv.gz"),
        usecols=["subject_id", "drug", "starttime", "stoptime"],
        parse_dates=["starttime", "stoptime"],
        chunksize=chunksize, low_memory=False,
    )
    n_chunks = 0
    total_raw = 0
    truncated = False
    try:
        for chunk in reader:
            n_chunks += 1
            total_raw += len(chunk)
            chunk = chunk[chunk["subject_id"].isin(cohort_ids)].copy()
            if chunk.empty:
                continue
            chunk = chunk.dropna(subset=["starttime", "drug"])
            chunk = chunk.join(win_by_id, on="subject_id")
            chunk = chunk[(chunk["starttime"] >= chunk["window_start"])
                          & (chunk["starttime"] < chunk["window_end"])]
            if chunk.empty:
                continue
            chunk["concept_id"] = (
                "DRUG_"
                + chunk["drug"].astype(str).str.strip().str.upper()
                      .str.replace(" ", "_", regex=False)
                      .str.replace("/", "_", regex=False)
            )
            chunk["relation"] = "hasPrescription"
            chunk["fact_type"] = "prescription"
            chunk["source"] = "drug"
            chunk = chunk.rename(columns={"starttime": "timestamp_start",
                                           "stoptime": "timestamp_end"})
            chunk["value_num"] = np.nan
            rx_chunks.append(chunk[FACT_COLS])
            if n_chunks % 5 == 0:
                kept = sum(len(c) for c in rx_chunks)
                print(f"  chunk {n_chunks}: {total_raw:,} read, {kept:,} kept")
    except EOFError as e:
        truncated = True
        print(f"  WARN: prescriptions.csv.gz truncated after {total_raw:,} rows "
              f"(chunk {n_chunks}). Using what is readable. ({e})")
    out = pd.concat(rx_chunks, ignore_index=True) if rx_chunks else pd.DataFrame(columns=FACT_COLS)
    if truncated:
        print("  NOTE: prescriptions file is corrupted; downstream prescription "
              "facts are partial.")
    print(f"  Prescription facts: {len(out):,}")
    return out


def _resolve_itemids_by_label(d_items: pd.DataFrame,
                               label_dict: dict) -> dict[int, str]:
    """Map d_items.itemid to a concept name by substring-matching the label
    column. First match wins (dict iteration order matters)."""
    d_items = d_items.copy()
    d_items["label_l"] = d_items["label"].astype(str).str.lower()
    itemid_to_concept: dict[int, str] = {}
    for concept, substrings in label_dict.items():
        matched = d_items[d_items["label_l"].apply(
            lambda s: any(sub in s for sub in substrings))]
        for iid in matched["itemid"]:
            if iid not in itemid_to_concept:
                itemid_to_concept[iid] = concept
    return itemid_to_concept


def _resolve_lab_itemids() -> dict[int, str]:
    """Map MIMIC itemid -> our CARDIOMETA concept name (LOINC-equivalent label)."""
    labitems = pd.read_csv(os.path.join(DATA_DIR, "d_labitems.csv.gz"))
    labitems["label_l"] = labitems["label"].astype(str).str.lower()
    itemid_to_concept: dict[int, str] = {}
    matched_per_concept: dict[str, int] = {}
    for concept, substrings in CARDIOMETA_LAB_LABELS.items():
        matched = labitems[labitems["label_l"].apply(
            lambda s: any(sub in s for sub in substrings))]
        new = 0
        for itemid in matched["itemid"]:
            if itemid not in itemid_to_concept:
                itemid_to_concept[itemid] = concept
                new += 1
        matched_per_concept[concept] = new
    print("  Lab itemid resolution (CARDIOMETA concept -> n itemids):")
    for c, n in matched_per_concept.items():
        print(f"    {c:15s}: {n}")
    print(f"  Total resolved itemids: {len(itemid_to_concept):,}")
    return itemid_to_concept


def _labevent_facts(windows: pd.DataFrame) -> pd.DataFrame:
    print("\n[6] Building lab event facts (chunked)...")
    itemid_to_concept = _resolve_lab_itemids()
    item_ids = set(itemid_to_concept.keys())
    cohort_ids = set(windows["subject_id"])
    win_by_id = windows.set_index("subject_id")[["window_start", "window_end"]]
    rng = np.random.default_rng(SEED)
    chunksize = 500_000
    reader = pd.read_csv(
        os.path.join(DATA_DIR, "labevents.csv.gz"),
        usecols=["subject_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        chunksize=chunksize, low_memory=False,
    )
    kept_chunks = []
    n_chunks = 0
    total_raw = 0
    for chunk in reader:
        n_chunks += 1
        total_raw += len(chunk)
        chunk = chunk[chunk["subject_id"].isin(cohort_ids)
                      & chunk["itemid"].isin(item_ids)
                      & chunk["valuenum"].notna()]
        if chunk.empty:
            if n_chunks % 10 == 0:
                kept = sum(len(c) for c in kept_chunks)
                print(f"  chunk {n_chunks:3d} | rows read: {total_raw:>12,} | kept so far: {kept:,}")
            continue
        # per-patient time window
        chunk = chunk.join(win_by_id, on="subject_id")
        chunk = chunk[(chunk["charttime"] >= chunk["window_start"])
                      & (chunk["charttime"] < chunk["window_end"])]
        if chunk.empty:
            if n_chunks % 10 == 0:
                kept = sum(len(c) for c in kept_chunks)
                print(f"  chunk {n_chunks:3d} | rows read: {total_raw:>12,} | kept so far: {kept:,}")
            continue
        # sample LAB_SAMPLE_RATE per chunk (per-patient stratification is approximated by uniform sampling)
        if LAB_SAMPLE_RATE < 1.0:
            mask = rng.random(len(chunk)) < LAB_SAMPLE_RATE
            chunk = chunk[mask]
            if chunk.empty:
                if n_chunks % 10 == 0:
                    kept = sum(len(c) for c in kept_chunks)
                    print(f"  chunk {n_chunks:3d} | rows read: {total_raw:>12,} | kept so far: {kept:,}")
                continue
        chunk["concept"] = chunk["itemid"].map(itemid_to_concept)
        chunk["concept_id"] = "LAB_" + chunk["concept"].astype(str)
        chunk["relation"] = "hasLabEvent"
        chunk["fact_type"] = "lab"
        chunk["source"] = "labevents"
        chunk = chunk.rename(columns={"charttime": "timestamp_start",
                                       "valuenum": "value_num"})
        chunk["timestamp_end"] = pd.NaT
        kept_chunks.append(chunk[FACT_COLS])
        if n_chunks % 10 == 0:
            kept = sum(len(c) for c in kept_chunks)
            print(f"  chunk {n_chunks:3d} | rows read: {total_raw:>12,} | kept so far: {kept:,}")
    print(f"  total chunks processed: {n_chunks}, total rows read: {total_raw:,}")
    out = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame(columns=FACT_COLS)
    print(f"  Lab event facts: {len(out):,}")
    return out


def _icu_facts(windows: pd.DataFrame) -> pd.DataFrame:
    print("\n[7] Building ICU stay facts...")
    icu = pd.read_csv(
        os.path.join(DATA_DIR, "icustays.csv.gz"),
        usecols=["subject_id", "stay_id", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"], low_memory=False,
    )
    icu = icu[icu["subject_id"].isin(set(windows["subject_id"]))].copy()
    icu = _filter_to_window(icu, windows, "intime")
    icu["concept_id"] = "ICU_STAY"
    icu["relation"] = "hasICUStay"
    icu["fact_type"] = "icu"
    icu["source"] = "icu"
    icu = icu.rename(columns={"intime": "timestamp_start",
                              "outtime": "timestamp_end"})
    icu["value_num"] = icu["los"].astype(float)
    out = icu[FACT_COLS]
    print(f"  ICU stay facts: {len(out):,}")
    return out


def _omr_facts(windows: pd.DataFrame) -> pd.DataFrame:
    print("\n[8] Building OMR facts (BP, BMI)...")
    omr = pd.read_csv(
        os.path.join(DATA_DIR, "omr.csv.gz"),
        usecols=["subject_id", "chartdate", "result_name", "result_value"],
        parse_dates=["chartdate"], low_memory=False,
    )
    omr = omr[omr["subject_id"].isin(set(windows["subject_id"]))].copy()
    omr = _filter_to_window(omr, windows, "chartdate")
    name_l = omr["result_name"].astype(str).str.lower()
    is_bp = name_l.str.contains("blood pressure", na=False)
    is_bmi = name_l.str.contains("bmi", na=False)

    bp = omr[is_bp].copy()
    parts = bp["result_value"].astype(str).str.split("/", n=1, expand=True)
    bp["sys"] = pd.to_numeric(parts[0], errors="coerce")
    bp["dia"] = pd.to_numeric(parts[1], errors="coerce")
    bp = bp.dropna(subset=["sys"])
    bp["concept_id"] = np.where(bp["sys"] > 140, "OMR_BP_HIGH", "OMR_BP_NORMAL")
    bp["relation"] = "hasBP"
    bp["fact_type"] = "omr_bp"
    bp["source"] = "omr"
    bp = bp.rename(columns={"chartdate": "timestamp_start"})
    bp["timestamp_end"] = pd.NaT
    bp["value_num"] = bp["sys"].astype(float)
    bp_out = bp[FACT_COLS]

    bmi = omr[is_bmi].copy()
    bmi["bmi_val"] = pd.to_numeric(bmi["result_value"], errors="coerce")
    bmi = bmi.dropna(subset=["bmi_val"])
    bmi["concept_id"] = np.select(
        [bmi["bmi_val"] >= 30, bmi["bmi_val"] >= 25],
        ["OMR_BMI_OBESE", "OMR_BMI_OVERWEIGHT"],
        default="OMR_BMI_NORMAL",
    )
    bmi["relation"] = "hasBMI"
    bmi["fact_type"] = "omr_bmi"
    bmi["source"] = "omr"
    bmi = bmi.rename(columns={"chartdate": "timestamp_start"})
    bmi["timestamp_end"] = pd.NaT
    bmi["value_num"] = bmi["bmi_val"].astype(float)
    bmi_out = bmi[FACT_COLS]

    out = pd.concat([bp_out, bmi_out], ignore_index=True)
    print(f"  OMR facts: {len(out):,} (BP: {len(bp_out):,}, BMI: {len(bmi_out):,})")
    return out


def _chartevents_facts(windows: pd.DataFrame) -> pd.DataFrame:
    """Vital signs from ICU chartevents (HR, BP, SpO2, RR, temp, GCS).
    Chunked because the file is 3.3 GB compressed / ~250 GB rows."""
    print("\n[9] Building chartevents (ICU vital signs) facts (chunked)...")
    d_items = pd.read_csv(os.path.join(DATA_DIR, "d_items.csv.gz"))
    itemid_to_concept = _resolve_itemids_by_label(d_items, CARDIOMETA_CHART_LABELS)
    item_ids = set(itemid_to_concept.keys())
    print(f"  resolved {len(item_ids)} chartevent itemids across "
          f"{len(CARDIOMETA_CHART_LABELS)} concept groups")
    by_concept = {}
    for c in itemid_to_concept.values():
        by_concept[c] = by_concept.get(c, 0) + 1
    for c, n in by_concept.items():
        print(f"    {c:<14s}: {n} itemids")

    cohort_ids = set(windows["subject_id"])
    win_by_id = windows.set_index("subject_id")[["window_start", "window_end"]]
    rng = np.random.default_rng(SEED)
    chunksize = 1_000_000
    reader = pd.read_csv(
        os.path.join(DATA_DIR, "chartevents.csv.gz"),
        usecols=["subject_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        chunksize=chunksize, low_memory=False,
    )
    kept_chunks = []
    n_chunks = 0
    total_raw = 0
    for chunk in reader:
        n_chunks += 1
        total_raw += len(chunk)
        chunk = chunk[chunk["subject_id"].isin(cohort_ids)
                      & chunk["itemid"].isin(item_ids)
                      & chunk["valuenum"].notna()]
        if not chunk.empty:
            chunk = chunk.join(win_by_id, on="subject_id")
            chunk = chunk[(chunk["charttime"] >= chunk["window_start"])
                          & (chunk["charttime"] < chunk["window_end"])]
            if not chunk.empty and CHART_SAMPLE_RATE < 1.0:
                chunk = chunk[rng.random(len(chunk)) < CHART_SAMPLE_RATE]
            if not chunk.empty:
                chunk["concept"] = chunk["itemid"].map(itemid_to_concept)
                chunk["concept_id"] = "VITAL_" + chunk["concept"].astype(str)
                chunk["relation"] = "hasVital"
                chunk["fact_type"] = "vital"
                chunk["source"] = "chartevents"
                chunk = chunk.rename(columns={"charttime": "timestamp_start",
                                                "valuenum": "value_num"})
                chunk["timestamp_end"] = pd.NaT
                kept_chunks.append(chunk[FACT_COLS])
        if n_chunks % 50 == 0:
            kept = sum(len(c) for c in kept_chunks)
            print(f"  chunk {n_chunks:4d} | rows read: {total_raw:>14,} | "
                  f"kept so far: {kept:,}")
    print(f"  total chunks processed: {n_chunks}, rows read: {total_raw:,}")
    out = pd.concat(kept_chunks, ignore_index=True) if kept_chunks \
        else pd.DataFrame(columns=FACT_COLS)
    print(f"  Vital sign facts: {len(out):,}")
    return out


def _inputevents_facts(windows: pd.DataFrame) -> pd.DataFrame:
    """ICU IV drips / fluids: vasopressors, IV insulin, IV diuretics, fluid boluses.
    Time-interval facts (starttime -> endtime), value_num = amount."""
    print("\n[10] Building inputevents (ICU IV drips/fluids) facts...")
    d_items = pd.read_csv(os.path.join(DATA_DIR, "d_items.csv.gz"))
    itemid_to_concept = _resolve_itemids_by_label(d_items, CARDIOMETA_INPUT_LABELS)
    item_ids = set(itemid_to_concept.keys())
    print(f"  resolved {len(item_ids)} inputevent itemids across "
          f"{len(CARDIOMETA_INPUT_LABELS)} concept groups")
    if not item_ids:
        return pd.DataFrame(columns=FACT_COLS)

    cohort_ids = set(windows["subject_id"])
    inp = pd.read_csv(
        os.path.join(DATA_DIR, "inputevents.csv.gz"),
        usecols=["subject_id", "itemid", "starttime", "endtime", "amount"],
        parse_dates=["starttime", "endtime"], low_memory=False,
    )
    inp = inp[inp["subject_id"].isin(cohort_ids)
              & inp["itemid"].isin(item_ids)
              & inp["starttime"].notna()].copy()
    inp = inp.merge(windows, on="subject_id", how="inner")
    inp = inp[(inp["starttime"] >= inp["window_start"])
              & (inp["starttime"] < inp["window_end"])]
    if inp.empty:
        print("  IV input facts: 0")
        return pd.DataFrame(columns=FACT_COLS)
    inp["concept"] = inp["itemid"].map(itemid_to_concept)
    inp["concept_id"] = "INPUT_" + inp["concept"].astype(str)
    inp["relation"] = "hasIVInput"
    inp["fact_type"] = "input"
    inp["source"] = "inputevents"
    inp = inp.rename(columns={"starttime": "timestamp_start",
                              "endtime": "timestamp_end",
                              "amount": "value_num"})
    out = inp[FACT_COLS]
    print(f"  IV input facts: {len(out):,}")
    return out


def _outputevents_facts(windows: pd.DataFrame) -> pd.DataFrame:
    """Urine output / drainage from outputevents (renal trajectory signal).
    Time-point facts, value_num = volume."""
    print("\n[11] Building outputevents (urine / drainage) facts...")
    d_items = pd.read_csv(os.path.join(DATA_DIR, "d_items.csv.gz"))
    itemid_to_concept = _resolve_itemids_by_label(d_items, CARDIOMETA_OUTPUT_LABELS)
    item_ids = set(itemid_to_concept.keys())
    print(f"  resolved {len(item_ids)} outputevent itemids")
    if not item_ids:
        return pd.DataFrame(columns=FACT_COLS)

    out_df = pd.read_csv(
        os.path.join(DATA_DIR, "outputevents.csv.gz"),
        usecols=["subject_id", "itemid", "charttime", "value"],
        parse_dates=["charttime"], low_memory=False,
    )
    out_df = out_df[out_df["subject_id"].isin(set(windows["subject_id"]))
                    & out_df["itemid"].isin(item_ids)
                    & out_df["value"].notna()].copy()
    out_df = out_df.merge(windows, on="subject_id", how="inner")
    out_df = out_df[(out_df["charttime"] >= out_df["window_start"])
                    & (out_df["charttime"] < out_df["window_end"])]
    if out_df.empty:
        print("  Output facts: 0")
        return pd.DataFrame(columns=FACT_COLS)
    out_df["concept"] = out_df["itemid"].map(itemid_to_concept)
    out_df["concept_id"] = "OUTPUT_" + out_df["concept"].astype(str)
    out_df["relation"] = "hasOutput"
    out_df["fact_type"] = "output"
    out_df["source"] = "outputevents"
    out_df = out_df.rename(columns={"charttime": "timestamp_start",
                                     "value": "value_num"})
    out_df["timestamp_end"] = pd.NaT
    out = out_df[FACT_COLS]
    print(f"  Output facts: {len(out):,}")
    return out


def build_tkg(cohort_df: pd.DataFrame | None = None) -> pd.DataFrame:
    # cohort_df is accepted for orchestrator compatibility; the cohort is always
    # re-read from disk so this script can also be invoked standalone.
    del cohort_df
    print("[1] Loading cohort and computing per-patient windows...")
    cohort, windows = _load_cohort_windows()
    print(f"  cohort patients: {len(cohort):,}")

    facts_pieces = []
    facts_pieces.append(_diagnosis_facts(windows)); gc.collect()
    facts_pieces.append(_procedure_facts(windows)); gc.collect()
    facts_pieces.append(_prescription_facts(windows)); gc.collect()
    facts_pieces.append(_labevent_facts(windows)); gc.collect()
    facts_pieces.append(_icu_facts(windows)); gc.collect()
    facts_pieces.append(_omr_facts(windows)); gc.collect()
    facts_pieces.append(_chartevents_facts(windows)); gc.collect()
    facts_pieces.append(_inputevents_facts(windows)); gc.collect()
    facts_pieces.append(_outputevents_facts(windows)); gc.collect()

    print("\n[9] Combining all facts...")
    facts = pd.concat(facts_pieces, ignore_index=True)
    print(f"  pre-dedup: {len(facts):,} facts")

    # Merge index_date for relative_days
    facts = facts.merge(
        cohort[["subject_id", "index_date", "endpoint_type"]],
        on="subject_id", how="left",
    )
    facts["relative_days"] = (
        (facts["timestamp_start"] - facts["index_date"]).dt.days
    )

    # Deduplicate (same subject, relation, concept, timestamp_start)
    before = len(facts)
    facts = facts.drop_duplicates(
        subset=["subject_id", "relation", "concept_id", "timestamp_start"]
    )
    print(f"  dedup: {before - len(facts):,} duplicates removed -> {len(facts):,}")

    facts = facts.sort_values(["subject_id", "timestamp_start"]).reset_index(drop=True)

    # Build node index
    print("\n[10] Building node index...")
    concepts = facts[["concept_id", "fact_type", "source"]].drop_duplicates()
    concepts = concepts.reset_index(drop=True)
    concepts["node_idx"] = np.arange(len(concepts))
    # Patient nodes
    pt_nodes = pd.DataFrame({
        "concept_id": ["PATIENT_" + str(sid) for sid in cohort["subject_id"].unique()],
        "fact_type": "patient",
        "source": "patient",
    })
    pt_nodes["node_idx"] = np.arange(len(concepts), len(concepts) + len(pt_nodes))
    node_index = pd.concat([concepts, pt_nodes], ignore_index=True)
    print(f"  concept nodes: {len(concepts):,}, patient nodes: {len(pt_nodes):,}, "
          f"total: {len(node_index):,}")

    # Save
    facts_out = os.path.join(OUTPUT_DIR, "tkg_facts.csv")
    nodes_out = os.path.join(OUTPUT_DIR, "node_index.csv")
    print(f"\n[11] Saving facts to {facts_out} ...")
    facts.to_csv(facts_out, index=False)
    print(f"     Saving node index to {nodes_out} ...")
    node_index.to_csv(nodes_out, index=False)

    # === TKG SUMMARY ===
    print("\n=== TKG SUMMARY ===")
    print(f"Total facts: {len(facts):,}")
    print(f"Unique patients: {facts['subject_id'].nunique():,}")
    print(f"Unique concept nodes: {len(concepts):,}")
    print(f"Total nodes: {len(node_index):,}")

    rel_counts = facts["relation"].value_counts()
    total = len(facts)
    print("\nFacts by relation type:")
    for rel in ["hasDiagnosis", "hasFinalDx", "hasProcedure", "hasPrescription",
                "hasLabEvent", "hasICUStay", "hasBP", "hasBMI",
                "hasVital", "hasIVInput", "hasOutput"]:
        n = int(rel_counts.get(rel, 0))
        pct = n / total * 100 if total else 0
        extra = ""
        if rel == "hasPrescription":
            sub = facts[facts["relation"] == rel]
            if len(sub):
                intv = sub["timestamp_end"].notna().mean() * 100
                extra = f"  [time-interval: {intv:.1f}%]"
        elif rel == "hasICUStay":
            extra = "  [time-interval: 100%]"
        print(f"  {rel:16s}: {n:>10,} ({pct:5.1f}%){extra}")

    print(f"\nTime range: {facts['timestamp_start'].min()} -> "
          f"{facts['timestamp_start'].max()}")

    per_pt = facts.groupby("subject_id").size()
    print(f"Mean facts per patient:   {per_pt.mean():.1f} ± {per_pt.std():.1f}")
    print(f"Median facts per patient: {per_pt.median():.0f}")

    print("\nFacts by endpoint class (mean per patient):")
    by_ep = (facts.groupby(["endpoint_type", "subject_id"]).size()
             .groupby("endpoint_type").mean())
    for ep in ["MI", "Stroke", "HF", "AF", "PAD", "censored"]:
        v = by_ep.get(ep, 0.0)
        print(f"  {ep:9s}: mean {v:7.1f} facts/patient")

    return facts


if __name__ == "__main__":
    build_tkg()

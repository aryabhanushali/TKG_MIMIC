"""EHRSHOT-specific configuration.

The EHRSHOT port re-uses the MIMIC pipeline from prep_modeling.py onward
unchanged. To do that, src/ehrshot/cohort_ehrshot.py and build_tkg_ehrshot.py
write cohort.csv / tkg_facts.csv / node_index.csv in exactly the MIMIC schema
into src.config.OUTPUT_DIR. So for an EHRSHOT run, MIMIC_OUTPUT_DIR must point
at the EHRSHOT output dir (see run_ehrshot.sbatch).

Every clinical definition (cardiometabolic index codes, endpoint codes,
washout codes, CCI, relation names) is imported from src/config.py rather than
copied, so the two datasets can't drift apart. Only the EHRSHOT-specific code
maps (LOINC -> concept, visit classes) live here.

Data source: meds_omop_ehrshot (the MEDS-OMOP extract). Unlike EHRSHOT_MEDS it
keeps the original ICD10CM/ICD9CM codes plus visit_id / end / clarity_table.
"""
import os

from src.config import (  # noqa: F401  (re-exported for the ehrshot modules)
    OUTPUT_DIR, SEED, MIN_FOLLOWUP_DAYS, PRE_INDEX_WINDOW_DAYS,
    CARDIOMETA_ICD10, CARDIOMETA_ICD9,
    ENDPOINTS_ICD10, ENDPOINTS_ICD9,
    ENDPOINT_HISTORY_ICD10, ENDPOINT_HISTORY_ICD9,
    CCI_WEIGHTS,
)

_ROOT = "/storage/ice1/9/3/hsathu3/EHRSHOT"
EHRSHOT_DATA_DIR = os.environ.get(
    "EHRSHOT_DATA_DIR", os.path.join(_ROOT, "meds_omop_ehrshot"))
EHRSHOT_MEDS_DIR = os.environ.get(
    "EHRSHOT_MEDS_DIR", os.path.join(_ROOT, "EHRSHOT_MEDS"))
EHRSHOT_ASSETS_DIR = os.environ.get(
    "EHRSHOT_ASSETS_DIR", os.path.join(_ROOT, "EHRSHOT_ASSETS"))
# MIMIC run whose baselines_survival/transfer_artifacts/ are applied to EHRSHOT.
MIMIC_TRANSFER_DIR = os.environ.get(
    "MIMIC_TRANSFER_DIR", "/storage/ice1/9/3/hsathu3/TKG_MIMIC/tkg_output")

# EHRSHOT is ~100x smaller than MIMIC, so nothing has to be subsampled for
# memory. Defaults are 1.0; set these env vars to mimic MIMIC's 0.3 / 0.1.
LAB_SAMPLE_RATE = float(os.environ.get("EHRSHOT_LAB_SAMPLE_RATE", "1.0"))
VITAL_SAMPLE_RATE = float(os.environ.get("EHRSHOT_VITAL_SAMPLE_RATE", "1.0"))

# ---------------------------------------------------------------------------
# Encounter classes
# ---------------------------------------------------------------------------
# meds_omop_ehrshot has no Visit/IP code. Inpatient status comes from the
# STANFORD_VISIT/<class>|<type>|<encounter> code attached to each visit_id,
# plus the OMOP visit codes below.
INPATIENT_STANFORD_CLASSES = {"Inpatient"}
INPATIENT_VISIT_CODES = {"Visit/IP", "Visit/ERIP", "Visit/OMOP4822460"}
# Visit/OMOP4822460 = "Intensive Care": every STANFORD_VISIT_DEPARTMENT whose
# description is an ICU (CVICU, PICU, E2-ICU, ...) has it as its parent code.
ICU_VISIT_CODES = {"Visit/OMOP4822460"}
# Vitals in MIMIC come from ICU chartevents only. To keep that meaning, EHRSHOT
# vitals are taken only from inpatient / ICU / ED encounters. Outpatient BP and
# BMI become the MIMIC OMR_BP_* / OMR_BMI_* concepts, as in MIMIC's omr table.
ACUTE_STANFORD_CLASSES = INPATIENT_STANFORD_CLASSES | {"Emergency", "Emergency Services"}
ACUTE_VISIT_CODES = INPATIENT_VISIT_CODES | {"Visit/ER"}
# Diagnosis source tables that are hospital-account admission diagnoses.
# They are treated as the "reason for admission", the closest EHRSHOT analogue
# of MIMIC's seq_num == 1 principal diagnosis (relation hasFinalDx).
ADMIT_DX_TABLES = {"shc_hsp_acct_admit_dx", "shc_hsp_admit_diag"}

# ---------------------------------------------------------------------------
# LOINC -> MIMIC concept names (CARDIOMETA_LAB_LABELS keys in src/config.py)
# ---------------------------------------------------------------------------
# Units were checked against the data: every code below is >= 99% in the same
# unit MIMIC uses (mg/dL, mmol/L == mEq/L, K/uL, U/L, g/dL, %, ng/mL, pg/mL).
LOINC_TO_LAB = {
    "2345-7": "GLUCOSE", "2339-0": "GLUCOSE",
    "4548-4": "HBA1C", "17856-6": "HBA1C",
    "2093-3": "CHOLESTEROL",
    "2085-9": "HDL",
    "13457-7": "LDL", "18262-6": "LDL", "2089-1": "LDL",
    "2571-8": "TRIGLYCERIDES", "3043-7": "TRIGLYCERIDES",
    "2160-0": "CREATININE",
    "3094-0": "BUN",
    "33762-6": "NTPROBNP",
    "42637-9": "BNP", "30934-4": "BNP",
    "10839-9": "TROPONIN_I", "42757-5": "TROPONIN_I", "89579-7": "TROPONIN_I",
    "49563-0": "TROPONIN_I",
    "6598-7": "TROPONIN_T", "67151-1": "TROPONIN_T",
    "2951-2": "SODIUM", "2947-0": "SODIUM",
    "2823-3": "POTASSIUM", "6298-4": "POTASSIUM",
    "718-7": "HEMOGLOBIN",
    "4544-3": "HEMATOCRIT",
    "777-3": "PLATELETS",
    "6690-2": "WBC", "20584-9": "WBC", "26464-8": "WBC",
    "1742-6": "ALT",
    "1920-8": "AST",
    "1975-2": "BILIRUBIN",
    "2532-0": "LDH", "14804-9": "LDH",
    "2028-9": "BICARBONATE", "1963-8": "BICARBONATE",
}
# Rows whose (lower-cased, stripped) unit is in this map get rescaled to the
# MIMIC unit; any unit not listed (or null) is kept as-is.
UNIT_RESCALE = {
    ("TROPONIN_I", "ng/l"): 1e-3,
    ("TROPONIN_T", "ng/l"): 1e-3,
}
# Units that are incompatible with the concept: these rows are dropped.
UNIT_DROP = {
    ("HBA1C", "mg/dl"),
    ("HEMATOCRIT", "g/dl"),
}

# LOINC -> MIMIC VITAL_* concept names (CARDIOMETA_CHART_LABELS keys).
# There is no pulse-oximetry SpO2 LOINC in EHRSHOT (2708-6 / 59408-5 are
# absent), so VITAL_SPO2 is not emitted.
LOINC_TO_VITAL = {
    "8867-4": "HEART_RATE",
    "8480-6": "BP_SYS",
    "8462-4": "BP_DIA",
    "8478-0": "BP_MEAN",
    "9279-1": "RESP_RATE",
    "8310-5": "TEMP_F",        # EHRSHOT body temperature is in deg F
    "35088-4": "GCS_TOTAL",
}
LOINC_BP_SYS = "8480-6"
LOINC_BP_DIA = "8462-4"
LOINC_BMI = "39156-5"

# CLMBR vocabulary files: the only place EHRSHOT ships RxNorm descriptions
# (the meds_omop codes.parquet has none for RxNorm).
CLMBR_TOKEN_2_CODE = os.path.join(EHRSHOT_ASSETS_DIR, "models/clmbr/token_2_code.json")
CLMBR_TOKEN_2_DESC = os.path.join(EHRSHOT_ASSETS_DIR, "models/clmbr/token_2_description.json")

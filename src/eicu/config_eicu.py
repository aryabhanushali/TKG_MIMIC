"""eICU-specific paths and code/label mappings for the transfer pipeline."""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# eICU v2.0 flat CSV files (not compressed — downloaded from PhysioNet)
EICU_DATA_DIR = os.environ.get(
    "EICU_DATA_DIR",
    "/home/hice1/hsathu3/scratch/eICU_dataset/physionet.org/files/eicu-crd/2.0",
)

# Root output directory for all eICU transfer outputs
EICU_OUTPUT_DIR = os.environ.get(
    "EICU_OUTPUT_DIR",
    os.path.expanduser("~/scratch/eICU_transfer_output"),
)

# Root of the MIMIC TKG pipeline output (contains baselines_survival/transfer_artifacts/)
MIMIC_OUTPUT_DIR = os.environ.get(
    "MIMIC_OUTPUT_DIR",
    os.path.expanduser("~/Desktop/TKG_MIMIC/tkg_output"),
)

# ---------------------------------------------------------------------------
# Cardiometabolic index conditions
# Matched against diagnosis.csv `diagnosisstring` (free-text hierarchical path).
# These run case-insensitive regex against the full diagnosisstring.
# ---------------------------------------------------------------------------
CARDIOMETA_TEXT = {
    "T2D":      r"(?i)\bdiabetes\b(?!.*insipidus)",
    "HTN":      r"(?i)\bhypertension\b",
    "Dyslipid": r"(?i)\b(?:cholesterol|dyslipidemia|hyperlipidemia|hypercholesterol)\b",
    "Obesity":  r"(?i)\b(?:obesity|obese)\b(?!.*hypoventilation)",
    # Metabolic syndrome essentially absent in eICU text hierarchy — included
    # for completeness but expected match count is 0.
    "MetSyn":   r"(?i)\bmetabolic syndrome\b",
}

# ---------------------------------------------------------------------------
# Circulatory endpoints — matched only in Primary-priority diagnosis rows
# (diagnosispriority == "Primary"), mirroring MIMIC's seq_num == 1 rule.
# ---------------------------------------------------------------------------
ENDPOINT_TEXT = {
    "MI":     r"(?i)\b(?:myocardial infarction|stemi|nstemi|acute\s+mi)\b",
    "Stroke": r"(?i)\b(?:ischemic stroke|cerebral infarction|ischemic cerebrovascular"
              r"|stroke|cva)\b",
    "HF":     r"(?i)\b(?:heart failure|congestive heart failure|chf)\b",
    "AF":     r"(?i)\b(?:atrial fibrillation|atrial flutter)\b",
    # PAD rarely appears as a Primary ICU admission reason in eICU (confirmed
    # 0 matches in the full dataset); included so the code doesn't silently
    # skip it — the evaluation will report n_pos=0 explicitly.
    "PAD":    r"(?i)\b(?:peripheral artery disease|peripheral arterial disease"
              r"|peripheral vascular disease)\b",
}

# admissionDx table path patterns (admitdxpath column) for endpoint detection
# as a secondary source when diagnosis.csv Primary rows are sparse.
ENDPOINT_ADMITDX = {
    "MI":     r"(?i)myocardial.*infarction|(?:^|\|)stemi\b|(?:^|\|)nstemi\b",
    "Stroke": r"(?i)stroke|cerebrovascular accident|cva",
    "HF":     r"(?i)heart failure|congestive heart failure|(?:^|\|)chf(?:\||$)",
    "AF":     r"(?i)atrial fib|atrial flutter",
    "PAD":    r"(?i)peripheral artery|peripheral arterial|peripheral vascular",
}

# ---------------------------------------------------------------------------
# Washout: past-history text patterns (pastHistory.csv `pasthistorypath`)
# used to exclude patients with prevalent circulatory disease at index.
# ---------------------------------------------------------------------------
WASHOUT_HISTORY_TEXT = {
    "MI":     r"(?i)\b(?:myocardial infarction|/mi(?:\s|-|$))\b",
    "Stroke": r"(?i)\b(?:stroke|cerebrovascular accident|cva|cerebral infarction)\b",
    "HF":     r"(?i)\b(?:heart failure|congestive heart failure|chf)\b",
    "AF":     r"(?i)\b(?:atrial fibrillation|atrial flutter)\b",
    "PAD":    r"(?i)\b(?:peripheral vascular disease|peripheral artery|peripheral arterial)\b",
}

# Also wash out patients whose PRIOR diagnosis stays (earlier patienthealthsystemstayid)
# contain any circulatory disease code in any priority position — mirrors the MIMIC
# any-position washout in ENDPOINT_HISTORY_ICD10/9.
WASHOUT_DX_TEXT = {
    ep: pat for ep, pat in ENDPOINT_TEXT.items()
}
# Remove the Primary-only restriction: washout from ANY position
# (The patterns above are the same; apply them to ALL rows, not just Primary.)

# ---------------------------------------------------------------------------
# eICU lab.csv `labname` → MIMIC LAB_* concept name
# Applied with case-insensitive substring matching (not word-boundary, because
# eICU lab names use abbreviations like "PT - INR" where word-boundary would
# fail on "PT").
# ---------------------------------------------------------------------------
EICU_LAB_MAP: dict[str, str] = {
    "glucose":               "GLUCOSE",
    "hba1c":                 "HBA1C",
    "hemoglobin a1c":        "HBA1C",
    "cholesterol":           "CHOLESTEROL",
    "hdl":                   "HDL",
    "ldl":                   "LDL",
    "triglyceride":          "TRIGLYCERIDES",
    "creatinine":            "CREATININE",
    "urea nitrogen":         "BUN",
    # eICU uses "BUN" directly as a lab name
    " bun ":                 "BUN",
    "ntprobnp":              "NTPROBNP",
    "nt-probnp":             "NTPROBNP",
    "probnp":                "NTPROBNP",
    " bnp":                  "BNP",
    "troponin i":            "TROPONIN_I",
    "troponin-i":            "TROPONIN_I",
    "troponin t":            "TROPONIN_T",
    "troponin-t":            "TROPONIN_T",
    "sodium":                "SODIUM",
    "potassium":             "POTASSIUM",
    "hemoglobin":            "HEMOGLOBIN",
    "hematocrit":            "HEMATOCRIT",
    "platelet":              "PLATELETS",
    "wbc":                   "WBC",
    "white blood cell":      "WBC",
    "alt":                   "ALT",
    "alanine aminotransferase": "ALT",
    "ast":                   "AST",
    "aspartate aminotransferase": "AST",
    "bilirubin":             "BILIRUBIN",
    "ldh":                   "LDH",
    "lactate dehydrogenase": "LDH",
    "bicarbonate":           "BICARBONATE",
    "hco3":                  "BICARBONATE",
}

# ---------------------------------------------------------------------------
# eICU vitalPeriodic.csv column name → MIMIC VITAL_* concept name
# vitalPeriodic is wide-format; columns listed here are melted into long form.
# ---------------------------------------------------------------------------
EICU_VITAL_PERIODIC_MAP: dict[str, str] = {
    "heartrate":          "HEART_RATE",
    "systemicsystolic":   "BP_SYS",
    "systemicdiastolic":  "BP_DIA",
    "systemicmean":       "BP_MEAN",
    "sao2":               "SPO2",
    "respiration":        "RESP_RATE",
    # eICU temperature column is in Celsius; MIMIC's TEMP_C concept is Celsius too.
    "temperature":        "TEMP_C",
}

# eICU vitalAperiodic.csv → MIMIC VITAL_* for non-invasive BP readings.
# MIMIC maps these to OMR_BP_HIGH/NORMAL (outpatient context); eICU is ICU.
# We map to the same VITAL_BP_* concepts rather than OMR to be consistent with
# the ICU context and avoid silently inflating the OMR feature presence.
# MISMATCH FLAG: MIMIC OMR_BP_* features will have 0 events for all eICU patients.
EICU_VITAL_APERIODIC_MAP: dict[str, str] = {
    "noninvasivesystolic":  "BP_SYS",
    "noninvasivediastolic": "BP_DIA",
    "noninvasivemean":      "BP_MEAN",
}

# ---------------------------------------------------------------------------
# eICU infusionDrug.csv `drugname` substrings → MIMIC INPUT_* concept name
# Matched case-insensitive.
# ---------------------------------------------------------------------------
EICU_INFUSION_MAP: dict[str, list[str]] = {
    "VASOPRESSOR":    ["norepinephrine", "epinephrine", "dopamine",
                       "dobutamine", "phenylephrine", "vasopressin", "milrinone"],
    "IV_INSULIN":     ["insulin"],
    "IV_DIURETIC":    ["furosemide", "bumetanide", "torsemide"],
    "IV_FLUID_BOLUS": ["nacl", "saline", "lactated", "ringers", "d5w", "dextrose"],
}

# ---------------------------------------------------------------------------
# eICU intakeOutput.csv `celllabel` substrings → MIMIC OUTPUT_* concept name
# ---------------------------------------------------------------------------
EICU_OUTPUT_MAP: dict[str, list[str]] = {
    "URINE_OUTPUT": ["urine", "foley", "void", "catheter"],
}

# ---------------------------------------------------------------------------
# Sampling rates (mirror MIMIC defaults from config.py)
# ---------------------------------------------------------------------------
VITAL_PERIODIC_SAMPLE_RATE = 0.10   # vitalPeriodic is dense; 10% mirrors MIMIC chart rate
LAB_SAMPLE_RATE             = 0.30   # mirrors MIMIC LAB_SAMPLE_RATE
SEED                        = 42

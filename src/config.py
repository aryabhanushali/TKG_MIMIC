"""Config: all hardcoded values for the TKG pipeline."""
import os
import pandas as pd

DATA_DIR = os.path.expanduser("~/Desktop/TKG_MIMIC/mimic_data/")
OUTPUT_DIR = os.path.expanduser("~/Desktop/TKG_MIMIC/tkg_output/")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")


def read_events_table(usecols=None) -> pd.DataFrame:
    """prep_modeling.py writes events.parquet, falling back to events.csv only
    if pyarrow/fastparquet is unavailable at write time -- every consumer must
    read whichever exists so this doesn't silently break (or, for a plain
    read_csv, hard-crash with FileNotFoundError) whenever the write-time
    environment happens to have parquet support installed."""
    parquet_path = os.path.join(MODELING_DIR, "events.parquet")
    csv_path = os.path.join(MODELING_DIR, "events.csv")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        return df[usecols] if usecols else df
    return pd.read_csv(csv_path, usecols=usecols, low_memory=False)
# TKG_SEED overrides the model-training seed only (weight init, dropout,
# batch order via _set_seed()); it does NOT affect the train/val/test split,
# which prep_modeling.py fixes once from this same constant and saves to
# labels.csv -- multi-seed TGN runs read that fixed split and vary only the
# model's own stochasticity, which is the correct multi-seed protocol.
SEED = int(os.environ.get("TKG_SEED", "42"))

# Cardiometabolic index conditions (ICD-10)
CARDIOMETA_ICD10 = {
    "T2D":        ["E11", "E110", "E111", "E112", "E113", "E114",
                   "E115", "E116", "E117", "E118", "E119"],
    "HTN":        ["I10", "I110", "I119", "I120", "I129",
                   "I130", "I131", "I132", "I139"],
    "Dyslipid":   ["E780", "E781", "E782", "E783", "E784",
                   "E785", "E786", "E787", "E788", "E789"],
    "Obesity":    ["E660", "E661", "E662", "E668", "E669"],
    "MetSyn":     ["E881"],
}

# Cardiometabolic index conditions (ICD-9)
CARDIOMETA_ICD9 = {
    "T2D":        ["2500", "2501", "2502", "2503", "2504",
                   "2505", "2506", "2507", "2508", "2509"],
    "HTN":        ["4010", "4011", "4019", "4020", "4021", "4029",
                   "4030", "4031", "4039", "4040", "4041", "4049",
                   "4050", "4051", "4059"],
    "Dyslipid":   ["2720", "2721", "2722", "2723", "2724"],
    "Obesity":    ["2780", "27800", "27801", "27803"],
    "MetSyn":     ["2777"],
}

# Circulatory endpoints (ICD-10) -- incident acute events.
# NOTE (clinical review): these define the *incident* endpoint and are matched
# only in the PRINCIPAL diagnosis position of a post-index admission (see
# cohort.py). Verify against your target guideline before submission.
ENDPOINTS_ICD10 = {
    # I21 = acute MI (STEMI/NSTEMI/type-2), I22 = subsequent acute MI.
    "MI":     ["I21", "I22"],
    # Ischemic stroke only (I63.x). Hemorrhagic (I60-I62) excluded by design.
    "Stroke": ["I63"],
    "HF":     ["I50"],
    # I48.x atrial fibrillation/flutter, incl. I48.91 "unspecified AF" (was
    # previously omitted, which both under-ascertained AF and leaked I48.91 in
    # as a pre-index "predictor").
    "AF":     ["I48"],
    # I73.x peripheral vascular disease + I70.2x lower-extremity atherosclerosis
    # (the common PAD code).
    "PAD":    ["I730", "I731", "I738", "I739", "I702"],
}

# Circulatory endpoints (ICD-9) -- incident acute events.
ENDPOINTS_ICD9 = {
    "MI":     ["410"],
    # Ischemic stroke: require the 5th digit "1" (with cerebral infarction);
    # 433.x0/434.x0 (occlusion/stenosis WITHOUT infarction) are chronic and
    # excluded here, and 436 (acute ill-defined CVD).
    "Stroke": ["43301", "43311", "43321", "43331", "43381", "43391",
               "43401", "43411", "43491", "436"],
    "HF":     ["428"],
    # 42731 = AF, 42732 = atrial flutter (ICD-10 I48 covers both; ICD-9 was
    # previously AF-only, an asymmetry vs. the ICD-10 definition).
    "AF":     ["42731", "42732"],
    "PAD":    ["4402", "4430", "4431", "4432", "4433", "4434",
               "4438", "4439"],
}

# Prevalent/chronic forms of each endpoint disease, matched in ANY diagnosis
# position to EXCLUDE patients who already have the disease at/before index
# (prevalent-case washout). Must be a superset of the incident codes above
# plus chronic/history codes, so e.g. chronic AF (I48.91) or old MI
# (412/I25.2) coded before index removes the patient rather than leaking in
# as a feature. See _assert_history_covers_incident below, which checks this.
ENDPOINT_HISTORY_ICD10 = {
    "MI":     ["I21", "I22", "I23", "I252"],
    "Stroke": ["I63", "I65", "I66", "I693", "Z8673"],
    "HF":     ["I50", "I110", "I130", "I132"],
    "AF":     ["I48"],
    # "I73" (not "I731"/"I738"/"I739") so this also covers the incident code
    # I730, which the prior list omitted -- prevalent I73.0 patients were not
    # being washed out even though I73.0 counts as an incident PAD event.
    "PAD":    ["I70", "I73", "Z95820"],
}
ENDPOINT_HISTORY_ICD9 = {
    "MI":     ["410", "412", "V4581"],
    "Stroke": ["433", "434", "436", "438", "V1254"],
    "HF":     ["428", "39891", "40201", "40211", "40291"],
    "AF":     ["42731", "42732"],
    # "443" (not "4439" alone) so this covers the incident range
    # 4430-4438, which the prior list omitted.
    "PAD":    ["4402", "443", "4471", "V434"],
}


def _assert_history_covers_incident():
    """Every incident endpoint prefix must be covered by some history prefix.

    Otherwise a patient with the prevalent form of a disease coded before
    index is not washed out, but the same code family in principal position
    after index counts as an incident event -- a leakage path.
    """
    for label, inc, hist in (
        ("ICD-10", ENDPOINTS_ICD10, ENDPOINT_HISTORY_ICD10),
        ("ICD-9", ENDPOINTS_ICD9, ENDPOINT_HISTORY_ICD9),
    ):
        for cause, prefixes in inc.items():
            for p in prefixes:
                if not any(p.startswith(h) for h in hist[cause]):
                    raise AssertionError(
                        f"{label} {cause}: incident code {p!r} not covered by "
                        f"any ENDPOINT_HISTORY prefix {hist[cause]!r}"
                    )


_assert_history_covers_incident()

# TKG relation types
RELATIONS = {
    "hasDiagnosis":    "dx",
    "hasFinalDx":      "final_dx",
    "hasProcedure":    "proc",
    "hasLabEvent":     "lab",
    "hasPrescription": "rx",
    "hasICUStay":      "icu",
    "hasBP":           "omr_bp",
    "hasBMI":          "omr_bmi",
    "hasVital":        "vital",
    "hasIVInput":      "input",
    "hasOutput":       "output",
}

# Cardiometabolic-relevant chartevent labels (resolved to itemids at runtime)
CARDIOMETA_CHART_LABELS = {
    "HEART_RATE":  ["heart rate"],
    "BP_SYS":      ["non invasive blood pressure systolic",
                    "arterial blood pressure systolic",
                    "manual blood pressure systolic"],
    "BP_DIA":      ["non invasive blood pressure diastolic",
                    "arterial blood pressure diastolic",
                    "manual blood pressure diastolic"],
    "BP_MEAN":     ["non invasive blood pressure mean",
                    "arterial blood pressure mean"],
    "SPO2":        ["o2 saturation pulseoxymetry"],
    "RESP_RATE":   ["respiratory rate"],
    "TEMP_F":      ["temperature fahrenheit"],
    "TEMP_C":      ["temperature celsius"],
    "GCS_TOTAL":   ["gcs - total", "glasgow coma scale total"],
}

CARDIOMETA_INPUT_LABELS = {
    "VASOPRESSOR":   ["norepinephrine", "epinephrine", "dopamine",
                       "dobutamine", "phenylephrine", "vasopressin"],
    "IV_INSULIN":    ["insulin - regular", "insulin regular"],
    "IV_DIURETIC":   ["furosemide"],
    "IV_FLUID_BOLUS":["nacl 0.9", "lactated ringers", "d5w", "d5 1/2 ns"],
}

CARDIOMETA_OUTPUT_LABELS = {
    "URINE_OUTPUT": ["urine", "foley", "void"],
}

CHART_SAMPLE_RATE = 0.10  # vitals are dense (1-5 min); 10% preserves trajectories


# LOINC reference list; not used at runtime (MIMIC labevents lacks a LOINC
# column, so matching is done via the label substrings below).
CARDIOMETA_LOINC = [
    "2345-7", "4548-4", "2093-3", "2085-9", "13457-7", "2571-8",
    "2160-0", "3094-0", "33762-6", "42757-5", "10839-9", "6598-7",
    "2947-0", "2823-3", "718-7", "4544-3", "777-3", "6690-2",
    "1742-6", "1920-8", "1975-2", "2532-0", "14682-9", "2028-9",
]

# Label substrings (case-insensitive) matched against d_labitems.label.
# Key = LOINC-equivalent concept name used as the concept_id prefix.
CARDIOMETA_LAB_LABELS = {
    "GLUCOSE":       ["glucose"],
    "HBA1C":         ["hemoglobin a1c", "hba1c", "% hemoglobin a1c"],
    "CHOLESTEROL":   ["cholesterol, total", "total cholesterol"],
    "HDL":           ["cholesterol, hdl", "hdl"],
    "LDL":           ["cholesterol, ldl", "ldl, calculated", "ldl"],
    "TRIGLYCERIDES": ["triglycerides", "triglyceride"],
    "CREATININE":    ["creatinine"],
    "BUN":           ["urea nitrogen"],
    "NTPROBNP":      ["ntprobnp", "nt-probnp", "probnp"],
    "BNP":           ["bnp"],
    "TROPONIN_I":    ["troponin i"],
    "TROPONIN_T":    ["troponin t"],
    "SODIUM":        ["sodium"],
    "POTASSIUM":     ["potassium"],
    "HEMOGLOBIN":    ["hemoglobin"],
    "HEMATOCRIT":    ["hematocrit"],
    "PLATELETS":     ["platelet count"],
    "WBC":           ["white blood cells", "wbc"],
    "ALT":           ["alanine aminotransferase", "alt"],
    "AST":           ["asparate aminotransferase", "aspartate aminotransferase", "ast"],
    "BILIRUBIN":     ["bilirubin, total"],
    "LDH":           ["lactate dehydrogenase"],
    "BICARBONATE":   ["bicarbonate"],
}

# Charlson Comorbidity Index: per-condition weight + ICD-10 and ICD-9 prefixes.
CCI_WEIGHTS = {
    "MI":               {"w": 1, "icd10": ["I21", "I22", "I252"],
                         "icd9":  ["410", "412"]},
    "CHF":              {"w": 1, "icd10": ["I50", "I099", "I110", "I130", "I132",
                                            "I255", "I420", "I425", "I426", "I427",
                                            "I428", "I429", "P290"],
                         "icd9":  ["428", "39891", "40201", "40211", "40291",
                                    "40401", "40403", "40411", "40413", "40491",
                                    "40493", "4254", "4255", "4256", "4257",
                                    "4258", "4259"]},
    "PVD":              {"w": 1, "icd10": ["I70", "I71", "I731", "I738", "I739",
                                            "I771", "I790", "I792", "K551", "K558",
                                            "K559", "Z958", "Z959"],
                         "icd9":  ["4439", "4471", "5571", "5579", "V434"]},
    "Stroke":           {"w": 1, "icd10": ["G45", "G46", "H340", "I60", "I61",
                                            "I62", "I63", "I64", "I65", "I66",
                                            "I67", "I68", "I69"],
                         "icd9":  ["36234", "430", "431", "432", "433", "434",
                                   "435", "436", "437", "438"]},
    "Dementia":         {"w": 1, "icd10": ["F00", "F01", "F02", "F03", "F051",
                                            "G30", "G311"],
                         "icd9":  ["290", "2941", "3312"]},
    "Pulmonary":        {"w": 1, "icd10": ["I278", "I279", "J40", "J41", "J42",
                                            "J43", "J44", "J45", "J46", "J47",
                                            "J60", "J61", "J62", "J63", "J64",
                                            "J65", "J66", "J67", "J684", "J701",
                                            "J703"],
                         "icd9":  ["4168", "4169", "490", "491", "492", "493",
                                   "494", "495", "496", "500", "501", "502",
                                   "503", "504", "505", "5064", "5081", "5088"]},
    "Rheumatic":        {"w": 1, "icd10": ["M05", "M06", "M315", "M32", "M33",
                                            "M34", "M351", "M353", "M360"],
                         "icd9":  ["4465", "7100", "7101", "7102", "7103", "7104",
                                   "7140", "7141", "7142", "7148", "725"]},
    "PUD":              {"w": 1, "icd10": ["K25", "K26", "K27", "K28"],
                         "icd9":  ["531", "532", "533", "534"]},
    "Liver_mild":       {"w": 1, "icd10": ["B18", "K700", "K701", "K702", "K703",
                                            "K709", "K713", "K714", "K715", "K717",
                                            "K73", "K74", "K760", "K762", "K763",
                                            "K764", "K768", "K769", "Z944"],
                         "icd9":  ["07022", "07023", "07032", "07033", "07044",
                                   "07054", "0706", "0709", "570", "571", "5733",
                                   "5734", "5738", "5739", "V427"]},
    "Diabetes_no_comp": {"w": 1, "icd10": ["E100", "E101", "E106", "E108", "E109",
                                            "E110", "E111", "E116", "E118", "E119",
                                            "E120", "E121", "E126", "E128", "E129",
                                            "E130", "E131", "E136", "E138", "E139",
                                            "E140", "E141", "E146", "E148", "E149"],
                         "icd9":  ["2500", "2501", "2502", "2503", "2508", "2509"]},
    "Diabetes_comp":    {"w": 2, "icd10": ["E102", "E103", "E104", "E105", "E107",
                                            "E112", "E113", "E114", "E115", "E117",
                                            "E122", "E123", "E124", "E125", "E127",
                                            "E132", "E133", "E134", "E135", "E137",
                                            "E142", "E143", "E144", "E145", "E147"],
                         "icd9":  ["2504", "2505", "2506", "2507"]},
    "Hemiplegia":       {"w": 2, "icd10": ["G041", "G114", "G801", "G802", "G81",
                                            "G82", "G830", "G831", "G832", "G833",
                                            "G834", "G839"],
                         "icd9":  ["3341", "342", "343", "3440", "3441", "3442",
                                   "3443", "3444", "3445", "3446", "3449"]},
    "Renal":            {"w": 2, "icd10": ["I120", "I131", "N032", "N033", "N034",
                                            "N035", "N036", "N037", "N052", "N053",
                                            "N054", "N055", "N056", "N057", "N18",
                                            "N19", "N250", "Z490", "Z491", "Z492",
                                            "Z940", "Z992"],
                         "icd9":  ["40301", "40311", "40391", "40402", "40403",
                                   "40412", "40413", "40492", "40493", "582",
                                   "583", "585", "586", "5880", "V420", "V451",
                                   "V56"]},
    "Cancer":           {"w": 2, "icd10": ["C00", "C01", "C02", "C03", "C04", "C05",
                                            "C06", "C07", "C08", "C09", "C10",
                                            "C11", "C12", "C13", "C14", "C15",
                                            "C16", "C17", "C18", "C19", "C20",
                                            "C21", "C22", "C23", "C24", "C25",
                                            "C26", "C30", "C31", "C32", "C33",
                                            "C34", "C37", "C38", "C39", "C40",
                                            "C41", "C43", "C45", "C46", "C47",
                                            "C48", "C49", "C50", "C51", "C52",
                                            "C53", "C54", "C55", "C56", "C57",
                                            "C58", "C60", "C61", "C62", "C63",
                                            "C64", "C65", "C66", "C67", "C68",
                                            "C69", "C70", "C71", "C72", "C73",
                                            "C74", "C75", "C76", "C81", "C82",
                                            "C83", "C84", "C85", "C88", "C90",
                                            "C91", "C92", "C93", "C94", "C95",
                                            "C96", "C97"],
                         "icd9":  ["140", "141", "142", "143", "144", "145",
                                   "146", "147", "148", "149", "150", "151",
                                   "152", "153", "154", "155", "156", "157",
                                   "158", "159", "160", "161", "162", "163",
                                   "164", "165", "170", "171", "172", "174",
                                   "175", "176", "179", "180", "181", "182",
                                   "183", "184", "185", "186", "187", "188",
                                   "189", "190", "191", "192", "193", "194",
                                   "195", "200", "201", "202", "203", "204",
                                   "205", "206", "207", "208"]},
    "Liver_severe":     {"w": 3, "icd10": ["I850", "I859", "I864", "I982", "K704",
                                            "K711", "K721", "K729", "K765", "K766",
                                            "K767"],
                         "icd9":  ["4560", "4561", "4562", "5722", "5723", "5724",
                                   "5728"]},
    "Mets":             {"w": 6, "icd10": ["C77", "C78", "C79", "C80"],
                         "icd9":  ["196", "197", "198", "199"]},
    "HIV":              {"w": 6, "icd10": ["B20", "B21", "B22", "B24"],
                         "icd9":  ["042", "043", "044"]},
}

MIN_FOLLOWUP_DAYS = 90      # minimum follow-up to be included
PRE_INDEX_WINDOW_DAYS = 1825  # 5-yr pre-index window: gives trajectories room
LAB_SAMPLE_RATE = 0.3       # sample 30% of lab events per patient (memory mgmt)

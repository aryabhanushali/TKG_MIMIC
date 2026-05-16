"""Internal-only ontology layer: ICD hierarchy + drug classes + lab categories.

Emits concept-concept 'isA' edges to enrich the patient-centric TKG into a
heterogeneous knowledge graph suitable for GNN message passing.

No external files required:
  - ICD-10 chapters: derived from leading letter (CDC chapter ranges)
  - ICD-9 chapters:  derived from numeric ranges (CMS standard)
  - Drug classes:    hand-curated ATC-style dictionary (cardiometabolic focus)
  - Lab categories:  grouped from CARDIOMETA_LAB_LABELS in config
"""
import os
import re
import pandas as pd

from src.config import OUTPUT_DIR


# --------------------------------------------------------------------------- #
# 1. ICD-10 chapter map (letter-prefix ranges from CDC ICD-10-CM)             #
# --------------------------------------------------------------------------- #
# (letter, low_2digits, high_2digits) -> chapter_id, chapter_title
ICD10_CHAPTERS = [
    # (start_code, end_code, chapter_id, title)
    ("A00", "B99", "I",     "Infectious and parasitic"),
    ("C00", "D49", "II",    "Neoplasms"),
    ("D50", "D89", "III",   "Blood and immune"),
    ("E00", "E89", "IV",    "Endocrine, nutritional, metabolic"),
    ("F01", "F99", "V",     "Mental and behavioral"),
    ("G00", "G99", "VI",    "Nervous system"),
    ("H00", "H59", "VII",   "Eye"),
    ("H60", "H95", "VIII",  "Ear"),
    ("I00", "I99", "IX",    "Circulatory system"),
    ("J00", "J99", "X",     "Respiratory system"),
    ("K00", "K95", "XI",    "Digestive system"),
    ("L00", "L99", "XII",   "Skin"),
    ("M00", "M99", "XIII",  "Musculoskeletal"),
    ("N00", "N99", "XIV",   "Genitourinary"),
    ("O00", "O9A", "XV",    "Pregnancy"),
    ("P00", "P96", "XVI",   "Perinatal"),
    ("Q00", "Q99", "XVII",  "Congenital"),
    ("R00", "R99", "XVIII", "Symptoms and signs"),
    ("S00", "T88", "XIX",   "Injury and poisoning"),
    ("V00", "Y99", "XX",    "External causes"),
    ("Z00", "Z99", "XXI",   "Health status factors"),
]


def _icd10_chapter(category: str) -> str | None:
    """Map a 3-char ICD-10 category (e.g., 'I21', 'E11') to chapter ID."""
    if len(category) < 3:
        return None
    cat = category[:3].upper()
    for start, end, chid, _ in ICD10_CHAPTERS:
        if start <= cat <= end:
            return f"ICD10_CHAPTER_{chid}"
    return None


# --------------------------------------------------------------------------- #
# 2. ICD-9 chapter map (numeric ranges from CMS ICD-9-CM)                     #
# --------------------------------------------------------------------------- #
ICD9_CHAPTERS = [
    (1,   139, "I",     "Infectious and parasitic"),
    (140, 239, "II",    "Neoplasms"),
    (240, 279, "III",   "Endocrine and metabolic"),
    (280, 289, "IV",    "Blood and immune"),
    (290, 319, "V",     "Mental"),
    (320, 389, "VI",    "Nervous and sensory"),
    (390, 459, "VII",   "Circulatory system"),
    (460, 519, "VIII",  "Respiratory system"),
    (520, 579, "IX",    "Digestive system"),
    (580, 629, "X",     "Genitourinary"),
    (630, 679, "XI",    "Pregnancy"),
    (680, 709, "XII",   "Skin"),
    (710, 739, "XIII",  "Musculoskeletal"),
    (740, 759, "XIV",   "Congenital"),
    (760, 779, "XV",    "Perinatal"),
    (780, 799, "XVI",   "Symptoms"),
    (800, 999, "XVII",  "Injury and poisoning"),
]


def _icd9_chapter(category: str) -> str | None:
    """Map a 3-char ICD-9 numeric category or V/E code to chapter."""
    cat = category.upper()
    if cat.startswith("V"):
        return "ICD9_CHAPTER_V"
    if cat.startswith("E"):
        return "ICD9_CHAPTER_E"
    try:
        num = int(cat[:3])
    except ValueError:
        return None
    for lo, hi, chid, _ in ICD9_CHAPTERS:
        if lo <= num <= hi:
            return f"ICD9_CHAPTER_{chid}"
    return None


# --------------------------------------------------------------------------- #
# 3. ICD-10-PCS section map (procedure)                                       #
# --------------------------------------------------------------------------- #
ICD10_PCS_SECTIONS = {
    "0": "Med_Surgical", "1": "Obstetrics", "2": "Placement",
    "3": "Administration", "4": "Measurement", "5": "Ext_Assistance",
    "6": "Ext_Therapies", "7": "Osteopathic", "8": "Other",
    "9": "Chiropractic", "B": "Imaging", "C": "Nuclear_Med",
    "D": "Radiation", "F": "Physical_Rehab", "G": "Mental_Health",
    "H": "Substance_Abuse", "X": "New_Technology",
}


def _icd9_proc_chapter(category: str) -> str | None:
    try:
        num = int(category[:2])
    except ValueError:
        return None
    if 0 <= num <= 86:
        return "ICD9_PROC_CHAPTER_Surgical"
    if 87 <= num <= 99:
        return "ICD9_PROC_CHAPTER_Diagnostic"
    return None


# --------------------------------------------------------------------------- #
# 4. Drug class dictionary (cardiometabolic focus, ATC-style)                 #
# --------------------------------------------------------------------------- #
# Order matters: more specific patterns first, generic later.
DRUG_CLASSES: list[tuple[str, list[str]]] = [
    ("STATIN",            ["ATORVASTATIN", "ROSUVASTATIN", "SIMVASTATIN",
                             "PRAVASTATIN", "LOVASTATIN", "PITAVASTATIN",
                             "FLUVASTATIN"]),
    ("ACE_INHIBITOR",     ["LISINOPRIL", "ENALAPRIL", "CAPTOPRIL", "RAMIPRIL",
                             "BENAZEPRIL", "QUINAPRIL", "FOSINOPRIL",
                             "PERINDOPRIL", "TRANDOLAPRIL", "MOEXIPRIL"]),
    ("ARB",               ["LOSARTAN", "VALSARTAN", "IRBESARTAN", "OLMESARTAN",
                             "TELMISARTAN", "CANDESARTAN", "AZILSARTAN",
                             "EPROSARTAN"]),
    ("BETA_BLOCKER",      ["METOPROLOL", "ATENOLOL", "PROPRANOLOL", "CARVEDILOL",
                             "BISOPROLOL", "LABETALOL", "NEBIVOLOL", "ESMOLOL",
                             "NADOLOL", "TIMOLOL", "ACEBUTOLOL", "PINDOLOL"]),
    ("CCB",               ["AMLODIPINE", "NIFEDIPINE", "DILTIAZEM", "VERAPAMIL",
                             "FELODIPINE", "NICARDIPINE", "ISRADIPINE",
                             "NIMODIPINE", "CLEVIDIPINE"]),
    ("LOOP_DIURETIC",     ["FUROSEMIDE", "TORSEMIDE", "BUMETANIDE",
                             "ETHACRYNIC"]),
    ("THIAZIDE_DIURETIC", ["HYDROCHLOROTHIAZIDE", "CHLORTHALIDONE",
                             "METOLAZONE", "INDAPAMIDE", "CHLOROTHIAZIDE"]),
    ("K_SPARING_DIURETIC", ["SPIRONOLACTONE", "EPLERENONE", "AMILORIDE",
                             "TRIAMTERENE"]),
    ("BIGUANIDE",         ["METFORMIN"]),
    ("SULFONYLUREA",      ["GLIPIZIDE", "GLYBURIDE", "GLIMEPIRIDE", "GLICLAZIDE",
                             "TOLBUTAMIDE", "CHLORPROPAMIDE"]),
    ("DPP4",              ["SITAGLIPTIN", "SAXAGLIPTIN", "LINAGLIPTIN",
                             "ALOGLIPTIN"]),
    ("GLP1",              ["LIRAGLUTIDE", "EXENATIDE", "DULAGLUTIDE",
                             "SEMAGLUTIDE", "LIXISENATIDE", "ALBIGLUTIDE"]),
    ("SGLT2",             ["DAPAGLIFLOZIN", "EMPAGLIFLOZIN", "CANAGLIFLOZIN",
                             "ERTUGLIFLOZIN"]),
    ("THIAZOLIDINEDIONE", ["PIOGLITAZONE", "ROSIGLITAZONE"]),
    ("MEGLITINIDE",       ["REPAGLINIDE", "NATEGLINIDE"]),
    ("INSULIN",           ["INSULIN", "LANTUS", "HUMALOG", "NOVOLOG", "LEVEMIR",
                             "TRESIBA", "HUMULIN", "NOVOLIN", "APIDRA",
                             "GLARGINE", "LISPRO", "ASPART", "DETEMIR"]),
    ("ANTICOAG_VKA",      ["WARFARIN"]),
    ("ANTICOAG_HEPARIN",  ["HEPARIN", "ENOXAPARIN", "DALTEPARIN", "FONDAPARINUX",
                             "TINZAPARIN"]),
    ("ANTICOAG_DOAC",     ["RIVAROXABAN", "APIXABAN", "DABIGATRAN", "EDOXABAN"]),
    ("ANTIPLATELET",      ["ASPIRIN", "CLOPIDOGREL", "PRASUGREL", "TICAGRELOR",
                             "DIPYRIDAMOLE", "CILOSTAZOL"]),
    ("NITRATE",           ["NITROGLYCERIN", "ISOSORBIDE"]),
    ("FIBRATE",           ["GEMFIBROZIL", "FENOFIBRATE", "BEZAFIBRATE"]),
    ("PCSK9",             ["EVOLOCUMAB", "ALIROCUMAB"]),
    ("EZETIMIBE",         ["EZETIMIBE"]),
    ("DIGOXIN",           ["DIGOXIN"]),
    ("ANTIARRHYTHMIC",    ["AMIODARONE", "FLECAINIDE", "PROPAFENONE", "SOTALOL",
                             "DRONEDARONE", "MEXILETINE", "DOFETILIDE",
                             "QUINIDINE", "LIDOCAINE"]),
    ("ALPHA_BLOCKER",     ["PRAZOSIN", "DOXAZOSIN", "TERAZOSIN"]),
    ("CENTRAL_ALPHA",     ["CLONIDINE", "METHYLDOPA", "GUANFACINE"]),
]


def _drug_class(drug_name: str) -> str | None:
    """Match a drug string (no DRUG_ prefix) to a class label."""
    s = drug_name.upper()
    for cls, names in DRUG_CLASSES:
        for n in names:
            # word boundary match to avoid false hits (e.g., 'ASPIRIN' in 'ASPARTAME')
            if re.search(r"(^|[_\W])" + re.escape(n) + r"($|[_\W])", s):
                return f"DRUG_CLASS_{cls}"
    return None


# --------------------------------------------------------------------------- #
# 5. Lab category dictionary                                                  #
# --------------------------------------------------------------------------- #
LAB_CATEGORIES: dict[str, list[str]] = {
    "GLUCOSE_METABOLISM": ["GLUCOSE", "HBA1C"],
    "LIPID_PANEL":        ["CHOLESTEROL", "HDL", "LDL", "TRIGLYCERIDES"],
    "RENAL_FUNCTION":     ["CREATININE", "BUN"],
    "CARDIAC_BIOMARKERS": ["NTPROBNP", "BNP", "TROPONIN_I", "TROPONIN_T"],
    "ELECTROLYTES":       ["SODIUM", "POTASSIUM", "BICARBONATE"],
    "HEMATOLOGY":         ["HEMOGLOBIN", "HEMATOCRIT", "PLATELETS", "WBC"],
    "HEPATIC_FUNCTION":   ["ALT", "AST", "BILIRUBIN", "LDH"],
}


def _lab_category(lab_name: str) -> str | None:
    s = lab_name.upper()
    for cat, members in LAB_CATEGORIES.items():
        if s in members:
            return f"LAB_CAT_{cat}"
    return None


# --------------------------------------------------------------------------- #
# 6. Parse concept_id and emit (concept -> parent, edge_type) tuples          #
# --------------------------------------------------------------------------- #
def _ontology_parents(concept_id: str) -> list[tuple[str, str]]:
    """Return [(parent_concept_id, edge_type), ...] for one concept."""
    parents: list[tuple[str, str]] = []

    # ICD-10 diagnosis: ICD10_<code>
    m = re.match(r"^ICD10_([A-Z0-9]+)$", concept_id)
    if m:
        code = m.group(1)
        if len(code) >= 3:
            category = f"ICD10_CAT_{code[:3]}"
            parents.append((category, "isA_icd_category"))
            ch = _icd10_chapter(code)
            if ch:
                parents.append((ch, "isA_icd_chapter"))
        return parents

    # ICD-9 diagnosis: ICD9_<code>
    m = re.match(r"^ICD9_([A-Z0-9]+)$", concept_id)
    if m:
        code = m.group(1)
        # category = first 3 chars (or full V/E code if shorter)
        category = f"ICD9_CAT_{code[:3]}"
        parents.append((category, "isA_icd_category"))
        ch = _icd9_chapter(code)
        if ch:
            parents.append((ch, "isA_icd_chapter"))
        return parents

    # ICD-10-PCS procedure: ICD10_PROC_<code>
    m = re.match(r"^ICD10_PROC_([A-Z0-9]+)$", concept_id)
    if m:
        code = m.group(1)
        if len(code) >= 1:
            section = code[0]
            sec_name = ICD10_PCS_SECTIONS.get(section, "Unknown")
            parents.append((f"ICD10_PROC_SECTION_{sec_name}",
                             "isA_proc_section"))
        return parents

    # ICD-9 procedure: ICD9_PROC_<code>
    m = re.match(r"^ICD9_PROC_(\d+)$", concept_id)
    if m:
        code = m.group(1)
        ch = _icd9_proc_chapter(code)
        if ch:
            parents.append((ch, "isA_proc_chapter"))
        return parents

    # Drug: DRUG_<name>
    if concept_id.startswith("DRUG_"):
        cls = _drug_class(concept_id[5:])
        if cls:
            parents.append((cls, "isA_drug_class"))
        return parents

    # Lab: LAB_<name>
    if concept_id.startswith("LAB_"):
        cat = _lab_category(concept_id[4:])
        if cat:
            parents.append((cat, "isA_lab_category"))
        return parents

    # OMR / ICU / etc.: no ontology parent (leave as-is)
    return parents


# --------------------------------------------------------------------------- #
# 7. Pipeline                                                                 #
# --------------------------------------------------------------------------- #
def build_ontology() -> None:
    print("Loading node index...")
    nodes = pd.read_csv(os.path.join(OUTPUT_DIR, "node_index.csv"))
    print(f"  total nodes: {len(nodes):,}")

    concept_nodes = nodes[nodes["fact_type"] != "patient"].copy()
    print(f"  concept nodes: {len(concept_nodes):,}")

    # Build edges
    print("\nBuilding ontology edges...")
    edges: list[tuple[str, str, str]] = []
    coverage = {k: 0 for k in (
        "icd10_dx", "icd9_dx", "icd10_pcs", "icd9_proc", "drug", "lab", "other"
    )}
    coverage_missed = {k: 0 for k in coverage}
    for cid in concept_nodes["concept_id"]:
        parents = _ontology_parents(cid)
        # categorize for reporting
        if cid.startswith("ICD10_PROC_"):       key = "icd10_pcs"
        elif cid.startswith("ICD10_"):           key = "icd10_dx"
        elif cid.startswith("ICD9_PROC_"):       key = "icd9_proc"
        elif cid.startswith("ICD9_"):            key = "icd9_dx"
        elif cid.startswith("DRUG_"):            key = "drug"
        elif cid.startswith("LAB_"):             key = "lab"
        else:                                     key = "other"
        if parents:
            coverage[key] += 1
        else:
            coverage_missed[key] += 1
        for parent_id, etype in parents:
            edges.append((cid, parent_id, etype))

    edges_df = pd.DataFrame(edges,
                              columns=["src_concept_id", "dst_concept_id",
                                        "edge_type"])
    print(f"  ontology edges: {len(edges_df):,}")
    print("\n  coverage by concept type (matched / total):")
    for k in coverage:
        total = coverage[k] + coverage_missed[k]
        if total:
            pct = coverage[k] / total * 100
            print(f"    {k:12s}: {coverage[k]:>6,} / {total:>6,} ({pct:5.1f}%)")

    # Add new parent nodes to the node index
    new_concept_ids = sorted(set(edges_df["dst_concept_id"]))
    existing = set(nodes["concept_id"])
    truly_new = [c for c in new_concept_ids if c not in existing]

    next_idx = int(nodes["node_idx"].max()) + 1
    new_rows = []
    for c in truly_new:
        # fact_type from prefix
        if   c.startswith("ICD10_CHAPTER_"):       ft = "icd10_chapter"
        elif c.startswith("ICD10_CAT_"):           ft = "icd10_category"
        elif c.startswith("ICD9_CHAPTER_"):        ft = "icd9_chapter"
        elif c.startswith("ICD9_CAT_"):            ft = "icd9_category"
        elif c.startswith("ICD10_PROC_SECTION_"):  ft = "icd10_proc_section"
        elif c.startswith("ICD9_PROC_CHAPTER_"):   ft = "icd9_proc_chapter"
        elif c.startswith("DRUG_CLASS_"):          ft = "drug_class"
        elif c.startswith("LAB_CAT_"):             ft = "lab_category"
        else:                                       ft = "ontology"
        new_rows.append({"concept_id": c, "fact_type": ft,
                          "source": "ontology", "node_idx": next_idx})
        next_idx += 1
    new_nodes_df = pd.DataFrame(new_rows)
    extended_nodes = pd.concat([nodes, new_nodes_df], ignore_index=True)

    # Resolve concept_id -> node_idx on the extended index
    concept_to_idx = dict(zip(extended_nodes["concept_id"],
                                 extended_nodes["node_idx"]))
    edges_df["src_node_idx"] = edges_df["src_concept_id"].map(concept_to_idx).astype("Int64")
    edges_df["dst_node_idx"] = edges_df["dst_concept_id"].map(concept_to_idx).astype("Int64")

    # Drop any unresolved (should be 0)
    n_unresolved = edges_df[["src_node_idx", "dst_node_idx"]].isna().any(axis=1).sum()
    if n_unresolved:
        print(f"  WARN: {n_unresolved} edges had unresolved node_idx; dropping")
        edges_df = edges_df.dropna(subset=["src_node_idx", "dst_node_idx"])

    # Save
    edges_path = os.path.join(OUTPUT_DIR, "ontology_edges.csv")
    nodes_path = os.path.join(OUTPUT_DIR, "node_index_v2.csv")
    edges_df.to_csv(edges_path, index=False)
    extended_nodes.to_csv(nodes_path, index=False)

    # Summary
    print("\n=== ONTOLOGY SUMMARY ===")
    print(f"  new ontology hub nodes: {len(new_rows):,}")
    print("  hub nodes by type:")
    for ft, sub in (pd.DataFrame(new_rows).groupby("fact_type")
                     if new_rows else pd.DataFrame().groupby([])):
        print(f"    {ft:25s}: {len(sub):>6,}")
    print("  edges by type:")
    for etype, n in edges_df["edge_type"].value_counts().items():
        print(f"    {etype:22s}: {n:>8,}")
    print(f"  extended node index size: {len(extended_nodes):,} "
          f"(was {len(nodes):,})")
    print(f"\nSaved:")
    print(f"  {edges_path}")
    print(f"  {nodes_path}")


if __name__ == "__main__":
    build_ontology()

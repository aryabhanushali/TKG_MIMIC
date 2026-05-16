"""Validate the TKG before modeling. Prints PASS/FAIL per check."""
import os
import pandas as pd

from src.config import OUTPUT_DIR, CARDIOMETA_LAB_LABELS, PRE_INDEX_WINDOW_DAYS


def validate_tkg() -> None:
    cohort = pd.read_csv(
        os.path.join(OUTPUT_DIR, "cohort.csv"),
        parse_dates=["index_date", "endpoint_date"],
    )
    facts = pd.read_csv(
        os.path.join(OUTPUT_DIR, "tkg_facts.csv"),
        parse_dates=["timestamp_start", "timestamp_end", "index_date"],
        low_memory=False,
    )

    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("================ TKG VALIDATION ================")
    log(f"cohort rows:   {len(cohort):,}")
    log(f"facts rows:    {len(facts):,}")
    log("")

    # CHECK 1 - No temporal leakage
    log("CHECK 1: No temporal leakage (timestamp_start < endpoint_date)")
    cohort_ep = cohort[cohort["endpoint_type"] != "censored"][
        ["subject_id", "endpoint_date"]
    ]
    leak = facts.merge(cohort_ep, on="subject_id", how="inner")
    n_leak = int((leak["timestamp_start"] >= leak["endpoint_date"]).sum())
    if n_leak == 0:
        log(f"  PASS  no facts at/after endpoint_date")
    else:
        log(f"  FAIL  {n_leak} facts have timestamp_start >= endpoint_date")
    log("")

    # CHECK 2 - All patients have facts
    log("CHECK 2: All cohort patients appear in facts")
    cohort_ids = set(cohort["subject_id"])
    fact_ids = set(facts["subject_id"])
    missing = cohort_ids - fact_ids
    if not missing:
        log(f"  PASS  all {len(cohort_ids):,} patients have at least 1 fact")
    else:
        log(f"  FAIL  {len(missing):,} patients have zero facts")
        log(f"        (first 10): {sorted(missing)[:10]}")
    log("")

    # CHECK 3 - Minimum facts per patient (>= 5)
    log("CHECK 3: Minimum facts per patient (warn if < 5)")
    n_per = facts.groupby("subject_id").size()
    sparse = (n_per < 5).sum()
    log(f"  patients with < 5 facts: {sparse:,} "
        f"({sparse / len(cohort_ids) * 100:.1f}%)")
    log(f"  min={n_per.min()}, p25={int(n_per.quantile(0.25))}, "
        f"median={int(n_per.median())}, p75={int(n_per.quantile(0.75))}, "
        f"max={n_per.max()}")
    log("")

    # CHECK 4 - ICD version consistency
    log("CHECK 4: ICD-9 vs ICD-10 distribution")
    dx_or_pr = facts[facts["fact_type"].isin(["diagnosis", "procedure"])].copy()
    if len(dx_or_pr):
        v_counts = dx_or_pr["source"].value_counts()
        for v, n in v_counts.items():
            log(f"  source={v}: {n:,} facts")
        # check duplicates: same (subject, timestamp_start, base_code) under both versions
        same_event = dx_or_pr.assign(
            base=dx_or_pr["concept_id"].str.replace(r"^ICD\d+_", "", regex=True)
        ).groupby(["subject_id", "timestamp_start", "relation", "base"])["source"].nunique()
        n_dup_versions = int((same_event > 1).sum())
        if n_dup_versions == 0:
            log("  PASS  no patient has same code in both ICD-9 and ICD-10 at same ts")
        else:
            log(f"  WARN  {n_dup_versions} (subject,ts,code) appear under both versions")
    log("")

    # CHECK 5 - Endpoint class balance
    log("CHECK 5: Endpoint class balance")
    cls = cohort["endpoint_type"].value_counts()
    for ep in ["MI", "Stroke", "HF", "AF", "PAD", "censored"]:
        n = int(cls.get(ep, 0))
        flag = "  WARN: < 150" if n < 150 and ep != "censored" else ""
        log(f"  {ep:9s}: {n:>6,d}{flag}")
    log("")

    # CHECK 6 - Time window check
    log("CHECK 6: Time window (relative_days)")
    rd = facts["relative_days"].dropna()
    log(f"  min={int(rd.min())}, max={int(rd.max())} "
        f"(expected min >= -{PRE_INDEX_WINDOW_DAYS})")
    out_before = int((rd < -PRE_INDEX_WINDOW_DAYS).sum())
    log(f"  facts before window_start: {out_before}")
    if out_before == 0:
        log("  PASS  no facts before the pre-index window")
    else:
        log("  FAIL  some facts precede the pre-index window")
    log("")

    # CHECK 7 - Drug concept coverage
    log("CHECK 7: Drug concept coverage")
    rx = facts[facts["relation"] == "hasPrescription"]
    top = rx["concept_id"].value_counts().head(20)
    log("  Top-20 drug concepts:")
    for c, n in top.items():
        log(f"    {n:>9,}  {c}")
    log("")
    key_drugs = ["METFORMIN", "LISINOPRIL", "ATORVASTATIN", "METOPROLOL",
                 "FUROSEMIDE", "ASPIRIN", "WARFARIN", "INSULIN"]
    rx_uniq = set(rx["concept_id"].unique())
    log("  Key cardiometabolic drug presence:")
    for d in key_drugs:
        found = any(c.startswith(f"DRUG_{d}") or f"_{d}_" in c or c.endswith(f"_{d}")
                    for c in rx_uniq)
        status = "FOUND" if found else "MISSING"
        log(f"    {d:14s} {status}")
    log("")

    # CHECK 8 - Lab concept coverage
    log("CHECK 8: Lab concept coverage")
    lab = facts[facts["relation"] == "hasLabEvent"]
    lab_counts = lab["concept_id"].value_counts()
    log("  Lab concept presence:")
    for concept in CARDIOMETA_LAB_LABELS:
        cid = f"LAB_{concept}"
        n = int(lab_counts.get(cid, 0))
        flag = "  (ZERO)" if n == 0 else ""
        log(f"    {cid:25s} {n:>10,d}{flag}")
    log("")

    out_path = os.path.join(OUTPUT_DIR, "validation_report.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nValidation report saved: {out_path}")


if __name__ == "__main__":
    validate_tkg()

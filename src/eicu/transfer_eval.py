"""MIMIC-IV → eICU model-transfer / portability evaluation.

Loads the Cox (elastic-net) and XGBoost-Survival models trained on MIMIC-IV
and scores eICU patients in the same feature space WITHOUT retraining.
Evaluates discrimination (1-year AUROC under competing-risks rule) and
calibration (risk-decile observed event rates).

This is explicitly a covariate-shift / portability analysis, NOT an
independent external validation in the traditional epidemiological sense.
Key structural differences between MIMIC-IV and eICU are documented as
M1–M6 flags and included in the report.

Temporal limitation:
  eICU covers 2014–2015 only.  The cross-year design (2014 index → 2015
  outcome) gives a maximum follow-up of ~1 year.  Only the 1-year AUROC
  is computed; the 3-year and 5-year horizons used in MIMIC are not feasible.
"""
import os
import pickle
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score, average_precision_score

from src.eicu.config_eicu import (
    EICU_DATA_DIR, EICU_OUTPUT_DIR, MIMIC_OUTPUT_DIR,
)
from src.eicu.cohort_eicu import build_cohort_eicu, CAUSES
from src.eicu.features_eicu import build_features_eicu

HORIZON_DAYS = 365          # only 1-year feasible in eICU
N_BOOT       = 1000         # bootstrap replicates for 95% CI on AUROC
N_DECILES    = 10
RNG          = np.random.default_rng(42)

TRANSFER_ART_DIR = os.path.join(
    MIMIC_OUTPUT_DIR, "baselines_survival", "transfer_artifacts"
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_artifacts() -> tuple[dict, dict, dict]:
    """Load feature_meta and per-cause model objects from transfer_artifacts/.

    Returns (feature_meta, cox_models, xgb_models).
    Models may be None if the pkl is missing.
    """
    meta_path = os.path.join(TRANSFER_ART_DIR, "feature_meta.pkl")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"feature_meta.pkl not found at {meta_path}.\n"
            "Run src/baselines_survival.py on MIMIC-IV first (seed=42) to "
            "generate transfer_artifacts/."
        )
    with open(meta_path, "rb") as f:
        feature_meta = pickle.load(f)
    print(f"[artifacts] Loaded feature_meta from {meta_path}")
    print(f"  n_bow_cols   : {feature_meta['n_bow_cols']:,}")
    print(f"  n_value_cols : {feature_meta['n_value_cols']:,}")
    print(f"  n_static_cols: {feature_meta['n_static_cols']:,}")
    print(f"  causes       : {feature_meta['causes']}")

    cox_models: dict = {}
    xgb_models: dict = {}
    for cause in CAUSES:
        cox_path = os.path.join(TRANSFER_ART_DIR, f"cox_{cause}.pkl")
        xgb_path = os.path.join(TRANSFER_ART_DIR, f"xgb_{cause}.pkl")
        if os.path.isfile(cox_path):
            with open(cox_path, "rb") as f:
                cox_models[cause] = pickle.load(f)
            print(f"  cox_{cause}: loaded")
        else:
            print(f"  cox_{cause}: MISSING ({cox_path})")
        if os.path.isfile(xgb_path):
            with open(xgb_path, "rb") as f:
                xgb_models[cause] = pickle.load(f)
            print(f"  xgb_{cause}: loaded")
        else:
            print(f"  xgb_{cause}: MISSING ({xgb_path})")
    return feature_meta, cox_models, xgb_models


def _load_or_build_cohort() -> pd.DataFrame:
    cohort_path = os.path.join(EICU_OUTPUT_DIR, "cohort_eicu.csv")
    if os.path.isfile(cohort_path):
        cohort = pd.read_csv(cohort_path)
        print(f"[cohort] Loaded existing cohort from {cohort_path} "
              f"({len(cohort):,} patients)")
    else:
        print("[cohort] cohort_eicu.csv not found — running build_cohort_eicu()...")
        cohort = build_cohort_eicu()
    return cohort


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(X, cox_models, xgb_models, feature_meta) -> dict[str, dict[str, np.ndarray]]:
    """Score all patients for each cause.

    Returns {cause: {"cox": array | None, "xgb": array | None}}.
    """
    scores: dict[str, dict] = {}
    for cause in CAUSES:
        scores[cause] = {"cox": None, "xgb": None}

        if cause in cox_models:
            print(f"  [Cox] scoring {cause}...", end=" ", flush=True)
            try:
                alpha = feature_meta["cox_alpha"][cause]
                # Score in batches of 500 to avoid building a dense copy of
                # the full matrix at once (n_patients × 17,695 float32 ≈ large).
                BATCH = 500
                n = X.shape[0]
                parts = []
                for start in range(0, n, BATCH):
                    end = min(start + BATCH, n)
                    parts.append(
                        cox_models[cause].predict(
                            X[start:end].toarray(), alpha=alpha
                        ).ravel()
                    )
                cox_risk = np.concatenate(parts).astype(np.float64)
                scores[cause]["cox"] = cox_risk
                print(f"done (shape={cox_risk.shape})")
            except Exception as e:
                print(f"FAILED: {e}")

        if cause in xgb_models:
            print(f"  [XGB] scoring {cause}...", end=" ", flush=True)
            try:
                xgb_risk = xgb_models[cause].predict(X).astype(np.float64)
                scores[cause]["xgb"] = xgb_risk
                print(f"done (shape={xgb_risk.shape})")
            except Exception as e:
                print(f"FAILED: {e}")
    return scores


# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------

def _competing_risks_auroc(
    risk_score: np.ndarray,
    endpoint_type: np.ndarray,
    time_to_event: np.ndarray,
    cause: str,
    horizon: float,
) -> dict:
    """AUROC under the competing-risks rule from baselines_survival.py.

    positive = cause observed AND duration <= horizon
    negative = survived past horizon  OR  competing observed event before horizon
    dropped  = censored before horizon
    """
    keep_pos  = (endpoint_type == cause) & (time_to_event <= horizon)
    survived  = time_to_event >= horizon
    competing = (time_to_event < horizon) & (endpoint_type != cause) & (endpoint_type != "censored")
    keep_neg  = survived | competing
    mask      = keep_pos | keep_neg
    y         = keep_pos[mask].astype(int)
    s         = risk_score[mask]

    result = {
        "cause": cause, "horizon_days": horizon,
        "n_pos": int(y.sum()), "n_neg": int((~y.astype(bool)).sum()),
        "n": int(len(y)),
        "auroc": np.nan, "auroc_lo": np.nan, "auroc_hi": np.nan,
        "auprc": np.nan,
    }
    if y.sum() == 0 or y.sum() == len(y):
        result["note"] = "insufficient positives or all-positive — AUROC undefined"
        return result

    result["auroc"] = float(roc_auc_score(y, s))
    result["auprc"] = float(average_precision_score(y, s))

    # Bootstrap 95% CI on AUROC
    boot_aurocs = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)
        yb, sb = y[idx], s[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        try:
            boot_aurocs.append(roc_auc_score(yb, sb))
        except Exception:
            pass
    if boot_aurocs:
        result["auroc_lo"] = float(np.percentile(boot_aurocs, 2.5))
        result["auroc_hi"] = float(np.percentile(boot_aurocs, 97.5))
    return result


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _calibration_deciles(
    risk_score: np.ndarray,
    endpoint_type: np.ndarray,
    time_to_event: np.ndarray,
    cause: str,
    horizon: float,
) -> pd.DataFrame:
    """Compute observed event rate per risk decile at the given horizon.

    Uses the same competing-risks masking as AUROC: the 'eligible' population
    is patients who are either positive (event <= horizon) or negative (survived
    or competing event before horizon).  Patients censored before horizon are
    excluded to avoid diluting observed rates.
    """
    keep_pos  = (endpoint_type == cause) & (time_to_event <= horizon)
    survived  = time_to_event >= horizon
    competing = (time_to_event < horizon) & (endpoint_type != cause) & (endpoint_type != "censored")
    keep_neg  = survived | competing
    mask      = keep_pos | keep_neg

    s = risk_score[mask]
    y = keep_pos[mask].astype(int)

    if len(s) < N_DECILES:
        return pd.DataFrame()

    decile_edges = np.percentile(s, np.linspace(0, 100, N_DECILES + 1))
    rows = []
    for d in range(N_DECILES):
        lo, hi = decile_edges[d], decile_edges[d + 1]
        if d == N_DECILES - 1:
            in_bin = (s >= lo)
        else:
            in_bin = (s >= lo) & (s < hi)
        n_bin   = int(in_bin.sum())
        n_event = int(y[in_bin].sum())
        rows.append({
            "cause": cause,
            "decile": d + 1,
            "risk_lo": float(lo),
            "risk_hi": float(hi),
            "n": n_bin,
            "n_event": n_event,
            "obs_rate": float(n_event / n_bin) if n_bin > 0 else np.nan,
        })
    df = pd.DataFrame(rows)
    # Monotonicity score: fraction of adjacent pairs where obs_rate is non-decreasing
    rates = df["obs_rate"].dropna().to_numpy()
    if len(rates) > 1:
        pairs = len(rates) - 1
        non_decr = int(np.sum(np.diff(rates) >= 0))
        df.attrs["monotonicity"] = non_decr / pairs
    return df


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt_auroc(row: dict) -> str:
    if np.isnan(row.get("auroc", np.nan)):
        return "  AUROC=N/A (insufficient positives)"
    ci = ""
    if not np.isnan(row.get("auroc_lo", np.nan)):
        ci = f" (95% CI {row['auroc_lo']:.3f}–{row['auroc_hi']:.3f})"
    return (f"  AUROC={row['auroc']:.3f}{ci}, "
            f"AUPRC={row['auprc']:.3f}, "
            f"n_pos={row['n_pos']}, n={row['n']}")


def _write_report(
    cohort: pd.DataFrame,
    mismatch_report: dict,
    all_disc: list[dict],
    all_calib: list[pd.DataFrame],
    out_dir: str,
) -> str:
    lines = []

    def h(title):
        lines.append("\n" + "=" * 70)
        lines.append(title)
        lines.append("=" * 70)

    def p(text=""):
        lines.append(text)

    lines.append("MIMIC-IV → eICU MODEL-TRANSFER PORTABILITY REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        "\nINTERPRETATION NOTE\n"
        + textwrap.fill(
            "This report describes a covariate-shift / portability evaluation. "
            "MIMIC-IV-trained Cox and XGBoost models are applied directly to "
            "eICU patients scored in the same feature space (no retraining). "
            "This is NOT an independent external validation: the eICU cohort "
            "was not used to select models, tune hyperparameters, or define "
            "features. Structural differences (M1–M6 below) reduce the "
            "expected discrimination relative to the MIMIC hold-out set.",
            width=70, subsequent_indent="  ",
        )
    )

    h("1. ARTIFACT PATHS")
    p(f"MIMIC transfer artifacts : {TRANSFER_ART_DIR}")
    p(f"eICU output dir          : {out_dir}")

    h("2. FEATURE MISMATCH FLAGS")
    p("M1  OMR_BP_*/OMR_BMI_* — MIMIC outpatient vitals/BMI have no eICU")
    p("    equivalent.  These feature columns are all-zero for every eICU")
    p("    patient.  Expected effect: MIMIC models that rely on OMR features")
    p("    will fail to stratify risk using this signal on eICU.")
    p()
    p("M2  had_icu_stay static — In MIMIC: prior ICU stay before index")
    p("    hospital admission.  In eICU: the index IS an ICU stay; the proxy")
    p("    (unitvisitnumber > 1) captures prior eICU *unit* stays, not prior")
    p("    hospital-level ICU admissions.  Clinical meaning differs.")
    p()
    p("M3  Drug concept IDs — MIMIC uses full normalised drug names.  eICU")
    p("    drug names include dosage/route/formulation; a prefix-stripping")
    p("    heuristic is applied to match MIMIC concepts.  Unmatched drugs → 0.")
    p()
    p("M4  Pre-index feature window — MIMIC: up to 5 years of history.  eICU:")
    p("    index ICU stay only (~1–3 days typical).  Feature density is far")
    p(f"    lower per patient (median ICU LOS ≈ "
      f"{mismatch_report.get('M4_preindex_window_days_median', 'N/A'):.1f} days).")
    p()
    p("M5  ICD code format — eICU stores codes with dots ('414.00'); MIMIC")
    p("    omits dots.  Dots are stripped for concept ID matching.")
    p()
    p("M6  vitalPeriodic BP — eICU ICU context records invasive arterial BP.")
    p("    MIMIC chartevents mixes invasive and non-invasive.  Both mapped to")
    p("    VITAL_BP_SYS / _DIA; population-level distributions differ.")

    h("3. eICU COHORT STATISTICS")
    p(f"Total patients             : {len(cohort):,}")
    ep_counts = cohort["endpoint_type"].value_counts()
    for ep in CAUSES + ["censored"]:
        n = int(ep_counts.get(ep, 0))
        warn = "  [too few for reliable AUROC]" if n < 30 and ep != "censored" else ""
        p(f"  {ep:9s}               : {n:5,d} ({n/len(cohort)*100:.1f}%){warn}")
    p(f"Age (mean ± SD)            : {cohort['age'].mean():.1f} ± {cohort['age'].std():.1f}"
      f"  [capped at 89]")
    p(f"Female                     : {cohort['female'].mean()*100:.1f}%")
    p(f"Mean CCI score             : {cohort['cci_score'].mean():.2f}")
    p(f"Mean cardiometa conditions : {cohort['num_cardiometa_conditions'].mean():.2f}")
    tte = cohort["time_to_event_days_approx"]
    p(f"Time-to-event (days, approx): min={tte.min():.0f}, "
      f"median={tte.median():.0f}, max={tte.max():.0f}")
    p()
    p(f"Events matched to MIMIC concepts: {mismatch_report['n_total_matched_events']:,}")
    p(f"Patients with ≥1 matched event : {mismatch_report['n_patients_with_any_event']:,} "
      f"/ {mismatch_report['n_eicu_patients']:,}")
    p(f"Bag-of-codes non-zeros         : {mismatch_report['bow_nnz']:,}")

    h("4. TRANSFER DISCRIMINATION (1-year AUROC, competing-risks rule)")
    p("Positive = cause observed AND time_to_event <= 365 days")
    p("Negative = survived >= 365 days OR competing event < 365 days")
    p("Dropped  = censored before 365 days  [no IPCW applied]")
    p("95% CI via 1000-sample bootstrap on eligible subset")
    p()
    disc_df = pd.DataFrame(all_disc)
    for model in ["cox", "xgb"]:
        p(f"--- {model.upper()} ---")
        sub = disc_df[disc_df["model"] == model]
        if sub.empty:
            p("  (model not available)")
            continue
        for _, row in sub.iterrows():
            p(f"  {row['cause']:9s}: {_fmt_auroc(row.to_dict())}")

    h("5. CALIBRATION (1-year, eligible subset, 10 deciles)")
    p("Monotonicity = fraction of adjacent decile pairs with non-decreasing")
    p("observed event rate.  1.0 = perfectly monotone; 0.5 = chance level.")
    p()
    for df_cal in all_calib:
        if df_cal.empty:
            continue
        cause = df_cal["cause"].iloc[0]
        model = df_cal.attrs.get("model", "?")
        mono  = df_cal.attrs.get("monotonicity", np.nan)
        p(f"  {model.upper()} / {cause}  (monotonicity={mono:.2f})")
        for _, r in df_cal.iterrows():
            bar = "#" * int(round(r["obs_rate"] * 40)) if not np.isnan(r["obs_rate"]) else ""
            p(f"    D{r['decile']:02d} [{r['risk_lo']:.2f},{r['risk_hi']:.2f}]"
              f"  n={r['n']:4d}  obs={r['obs_rate']:.3f}  {bar}")
        p()

    h("6. STRUCTURAL LIMITATIONS AND INTERPRETATION")
    p(textwrap.fill(
        "1. Follow-up: eICU covers 2014–2015 only. Cross-year design "
        "(2014 index → 2015 outcome) gives ≤1 year follow-up. The 3-year "
        "and 5-year AUROC targets used in MIMIC are not computable.",
        width=70, initial_indent="", subsequent_indent="   ",
    ))
    p()
    p(textwrap.fill(
        "2. Feature sparsity: The pre-index window is the index ICU stay "
        "only (~1–3 days). MIMIC uses up to 5 years of EHR history. "
        "Most bag-of-codes features will be zero for eICU patients, biasing "
        "the Cox/XGBoost risk scores toward the intercept / baseline hazard.",
        width=70, initial_indent="", subsequent_indent="   ",
    ))
    p()
    p(textwrap.fill(
        "3. Population shift: eICU is a multi-centre ICU dataset spanning "
        "208 ICUs across the US.  MIMIC-IV is a single-centre dataset from "
        "Beth Israel Deaconess Medical Center.  Case-mix, coding practices, "
        "and cardiometabolic prevalence differ substantially.",
        width=70, initial_indent="", subsequent_indent="   ",
    ))
    p()
    p(textwrap.fill(
        "4. Outcome rarity: Given the short follow-up and the cross-year "
        "design requirement, positive cases per cause are few.  AUROC "
        "estimates have wide confidence intervals; n_pos < 30 are flagged.",
        width=70, initial_indent="", subsequent_indent="   ",
    ))
    p()
    p(textwrap.fill(
        "5. Reporting guidance for IEEE ICHI reviewers: This analysis "
        "demonstrates cross-dataset feature portability of the MIMIC-trained "
        "model vocabulary, not performance equivalence.  Reduced AUROC on "
        "eICU relative to MIMIC hold-out is expected given the structural "
        "mismatches above and should be reported as such.",
        width=70, initial_indent="", subsequent_indent="   ",
    ))

    report = "\n".join(lines)
    report_path = os.path.join(out_dir, "eicu_transfer_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n[report] Written to {report_path}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    os.makedirs(EICU_OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("MIMIC-IV → eICU MODEL-TRANSFER EVALUATION")
    print("=" * 70)

    # 1. Load artifacts
    print("\n[1] Loading transfer artifacts...")
    feature_meta, cox_models, xgb_models = _load_artifacts()

    # 2. Cohort
    print("\n[2] Building / loading eICU cohort...")
    cohort = _load_or_build_cohort()

    # 3. Features
    print("\n[3] Extracting eICU features in MIMIC concept space...")
    X, patient_order, mismatch_report = build_features_eicu(cohort, feature_meta)

    # Align cohort rows to patient_order
    cohort_idx = cohort.set_index("uniquepid").loc[patient_order].reset_index()
    endpoint_type = cohort_idx["endpoint_type"].to_numpy()
    time_to_event = cohort_idx["time_to_event_days_approx"].to_numpy(dtype=float)

    # 4. Score
    print("\n[4] Scoring patients with MIMIC-trained models...")
    scores = _score(X, cox_models, xgb_models, feature_meta)

    # 5. Evaluate
    print(f"\n[5] Evaluating at {HORIZON_DAYS}-day horizon...")
    all_disc: list[dict] = []
    all_calib: list[pd.DataFrame] = []
    causes_with_few_pos: list[str] = []

    for cause in CAUSES:
        for model_name in ("cox", "xgb"):
            risk = scores[cause][model_name]
            if risk is None:
                continue

            disc = _competing_risks_auroc(
                risk, endpoint_type, time_to_event, cause, float(HORIZON_DAYS)
            )
            disc["model"] = model_name
            all_disc.append(disc)
            n_warn = " [<30 positives — AUROC unreliable]" if disc["n_pos"] < 30 else ""
            print(f"  {model_name.upper()} / {cause:9s}: "
                  f"AUROC={disc['auroc']:.3f}  "
                  f"n_pos={disc['n_pos']}{n_warn}")
            if disc["n_pos"] < 30 and model_name == "cox":
                causes_with_few_pos.append(cause)
            # Per-cause calibration, computed here where `risk`/`cause` are the
            # current loop values (previously it ran after the fallback loop
            # and so only ever saw the stale last cause, PAD).
            cal = _calibration_deciles(
                risk, endpoint_type, time_to_event, cause, float(HORIZON_DAYS)
            )
            cal.attrs["model"] = model_name
            cal["cause"] = cause
            all_calib.append(cal)

    # ------------------------------------------------------------------
    # Fallback: "any circulatory event" AUROC when per-cause counts are low.
    # Collapses all 5 endpoints into a single binary positive label.
    # Use the mean of per-cause Cox risk scores as the composite score.
    # ------------------------------------------------------------------
    any_positives = (endpoint_type != "censored") & (time_to_event <= HORIZON_DAYS)
    survived_or_competing = (time_to_event >= HORIZON_DAYS)
    n_any_pos = int(any_positives.sum())
    print(f"\n  Any-endpoint positives at {HORIZON_DAYS}d: {n_any_pos}")

    if len(causes_with_few_pos) > 0 or n_any_pos >= 10:
        print(f"\n[5b] Fallback: any-circulatory-event AUROC "
              f"(triggered by {len(causes_with_few_pos)} causes with <30 positives)")
        for model_name in ("cox", "xgb"):
            available = [c for c in CAUSES if scores[c][model_name] is not None]
            if not available:
                continue
            # Average risk across causes (each already in comparable rank space)
            composite = np.mean(
                np.stack([scores[c][model_name] for c in available], axis=1), axis=1
            )
            mask = any_positives | survived_or_competing
            y = any_positives[mask].astype(int)
            s = composite[mask]
            if y.sum() < 5 or y.sum() == len(y):
                print(f"  {model_name.upper()} / any: AUROC=N/A "
                      f"(n_pos={int(y.sum())})")
                all_disc.append({"cause": "any", "horizon_days": HORIZON_DAYS,
                                 "model": model_name,
                                 "n_pos": int(y.sum()), "n": int(len(y)),
                                 "auroc": np.nan, "auprc": np.nan,
                                 "auroc_lo": np.nan, "auroc_hi": np.nan})
                continue
            auroc_any = float(roc_auc_score(y, s))
            auprc_any = float(average_precision_score(y, s))
            boot = []
            n_el = len(y)
            for _ in range(N_BOOT):
                idx = RNG.integers(0, n_el, size=n_el)
                yb, sb = y[idx], s[idx]
                if yb.sum() == 0 or yb.sum() == len(yb):
                    continue
                try:
                    boot.append(roc_auc_score(yb, sb))
                except Exception:
                    pass
            lo = float(np.percentile(boot, 2.5)) if boot else np.nan
            hi = float(np.percentile(boot, 97.5)) if boot else np.nan
            ci = f" (95% CI {lo:.3f}–{hi:.3f})" if not np.isnan(lo) else ""
            print(f"  {model_name.upper()} / any: AUROC={auroc_any:.3f}{ci}  "
                  f"AUPRC={auprc_any:.3f}  n_pos={int(y.sum())}  n={int(len(y))}")
            all_disc.append({"cause": "any", "horizon_days": HORIZON_DAYS,
                             "model": model_name,
                             "n_pos": int(y.sum()), "n": int(len(y)),
                             "auroc": auroc_any, "auprc": auprc_any,
                             "auroc_lo": lo, "auroc_hi": hi})

    # 6. Save CSVs
    print("\n[6] Saving output files...")
    disc_df = pd.DataFrame(all_disc)
    disc_path = os.path.join(EICU_OUTPUT_DIR, "eicu_transfer_discrimination.csv")
    disc_df.to_csv(disc_path, index=False)
    print(f"  discrimination metrics → {disc_path}")

    if all_calib:
        calib_df = pd.concat(
            [df.assign(model=df.attrs.get("model", "?"))
             for df in all_calib if not df.empty],
            ignore_index=True,
        )
        calib_path = os.path.join(EICU_OUTPUT_DIR, "eicu_transfer_calibration.csv")
        calib_df.to_csv(calib_path, index=False)
        print(f"  calibration deciles → {calib_path}")

    # 7. Text report
    print("\n[7] Writing transfer report...")
    report = _write_report(cohort, mismatch_report, all_disc, all_calib, EICU_OUTPUT_DIR)
    print(report)


if __name__ == "__main__":
    run()

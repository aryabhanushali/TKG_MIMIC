"""End-to-end orchestrator for the CardioMM-TKG pipeline.

Run order:
    1. build_cohort        -- cardiometabolic -> CV cohort + Charlson
    2. build_tkg           -- 13.4M temporal facts, 8 modalities
    3. validate_tkg        -- 8 sanity checks (no leakage, balance, coverage)
    4. visualize_tkg       -- Pillar-1 cohort / TKG figures
    5. extract_cohort_notes-- filter discharge notes to cohort
    6. embed_notes         -- Bio_ClinicalBERT [CLS] embeddings (~440 MB DL)
    7. aggregate_notes     -- per-patient note vector (pre-index discharge)
    8. prep_modeling       -- pre-index event table + labels + splits
    9. baselines_survival  -- Cox + XGBoost-Survival per cause
   10. tgn_survival        -- DeepHit-style temporal-graph competing risks
   11. compare_survival    -- final per-cause AUROC at 1y/3y/5y horizons

Each step can also be run independently (`python -u -m src.<module>`).
Test-set predictions are produced exactly once at the end of step 10.
"""
import time

from src.cohort import build_cohort
from src.build_tkg import build_tkg
from src.validate_tkg import validate_tkg
from src.visualize_tkg import visualize_tkg
from src.notes_extract import extract_cohort_notes
from src.notes_embed import embed_notes
from src.notes_aggregate import aggregate_notes
from src.prep_modeling import prep_modeling
from src.baselines_survival import run as run_baselines_survival
from src.tgn_survival import train_and_eval as train_tgn_survival
from src.compare_survival import compare as compare_survival


def _step(name: str, fn, *args, **kwargs):
    print(f"\n{'='*72}\n>>> {name}\n{'='*72}")
    t0 = time.time()
    out = fn(*args, **kwargs)
    print(f"<<< {name} done in {(time.time() - t0) / 60:.1f} min")
    return out


if __name__ == "__main__":
    cohort_df = _step("STEP 1: build_cohort", build_cohort)
    _step("STEP 2: build_tkg", build_tkg, cohort_df)
    _step("STEP 3: validate_tkg", validate_tkg)
    _step("STEP 4: visualize_tkg", visualize_tkg)
    _step("STEP 5: extract_cohort_notes", extract_cohort_notes)
    _step("STEP 6: embed_notes", embed_notes)
    _step("STEP 7: aggregate_notes", aggregate_notes)
    _step("STEP 8: prep_modeling", prep_modeling)
    _step("STEP 9: baselines_survival", run_baselines_survival)
    _step("STEP 10: tgn_survival", train_tgn_survival)
    _step("STEP 11: compare_survival", compare_survival)
    print("\nAll done. See tkg_output/survival_comparison_test.csv for Table 1.")

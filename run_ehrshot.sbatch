#!/bin/bash
#SBATCH --job-name=tkg_ehrshot
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/ehrshot_%j.log
#SBATCH --error=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/ehrshot_%j.log

# ---------------------------------------------------------------------------
# CardioMM-TKG on EHRSHOT (Stanford): native retraining + MIMIC transfer
#
# src/ehrshot/ writes cohort.csv / tkg_facts.csv / node_index.csv in the MIMIC
# schema, so everything from prep_modeling onward is the unchanged MIMIC code.
#
#   1. EHRSHOT cohort + TKG (src/ehrshot/)
#   2. prep_modeling (random 70/15/15), tests_integrity (hard gate),
#      build_ontology, build_cooccurrence
#   3. EHRSHOT's own validation sweeps for Cox alpha / XGBoost config
#      -> hparams_survival.json (read by baselines_survival.py)
#   4. 5 seeds x {Cox+XGBoost, TKG-Transformer, patient-graph model}
#   5. summaries (compare_survival, evaluate_stats, multi-seed)
#   6. MIMIC -> EHRSHOT transfer of the frozen Cox / XGBoost models
#   7. EHRSHOT benchmark task new_acutemi (binary head), 5 seeds
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=/home/hice1/hsathu3/scratch/TKG_MIMIC
PY=/home/hice1/hsathu3/.conda/envs/fedhealth/bin/python

export EHRSHOT_DATA_DIR=/storage/ice1/9/3/hsathu3/EHRSHOT/meds_omop_ehrshot
export EHRSHOT_MEDS_DIR=/storage/ice1/9/3/hsathu3/EHRSHOT/EHRSHOT_MEDS
export EHRSHOT_ASSETS_DIR=/storage/ice1/9/3/hsathu3/EHRSHOT/EHRSHOT_ASSETS
# All pipeline stages read/write MIMIC_OUTPUT_DIR; point it at the EHRSHOT run.
export MIMIC_OUTPUT_DIR=$REPO/tkg_output_ehrshot
# MIMIC run whose transfer_artifacts are applied in step 6.
export MIMIC_TRANSFER_DIR=$REPO/tkg_output

cd "$REPO"
mkdir -p "$REPO/logs" "$MIMIC_OUTPUT_DIR"
echo "Job ${SLURM_JOB_ID:-local}  started $(date)  output: $MIMIC_OUTPUT_DIR"

# 1. Cohort + TKG
$PY -u -m src.ehrshot.cohort_ehrshot
$PY -u -m src.ehrshot.build_tkg_ehrshot

# 2. Split, integrity gate, graph layers
$PY -u -m src.prep_modeling
$PY -u -m src.tests_integrity
$PY -u -m src.ablations.build_ontology
$PY -u -m src.ablations.build_cooccurrence

# 3. EHRSHOT-specific survival-baseline hyperparameters (validation-only)
rm -f "$MIMIC_OUTPUT_DIR/hparams_survival.json"
$PY -u -m src.ablations.cox_regularization_sweep
$PY -u -m src.ablations.xgb_survival_sweep
$PY -u -m src.ehrshot.select_hparams

# 4. Models, 5 seeds (seed changes model randomness only, never the split)
for s in 42 43 44 45 46; do
    echo "=== seed $s  $(date) ==="
    TKG_SEED=$s $PY -u -m src.baselines_survival
    TKG_SEED=$s $PY -u -m src.tgn_survival
    TKG_SEED=$s $PY -u -m src.ablations.patient_graph_gnn
done

# 5. Summaries
$PY -u -m src.compare_survival
$PY -u -m src.evaluate_stats
$PY -u -m src.multi_seed_summary
$PY -u -m src.ablations.patient_gnn_multi_seed_summary

# 6. Transfer: frozen MIMIC Cox / XGBoost -> EHRSHOT
$PY -u -m src.ehrshot.transfer_eval

# 7. EHRSHOT benchmark task with a binary head
for s in 42 43 44 45 46; do
    TKG_SEED=$s $PY -u -m src.ehrshot.benchmark_binary --task new_acutemi
done

echo "Done $(date)."
echo "  native:   $MIMIC_OUTPUT_DIR/survival_comparison_test.csv"
echo "  transfer: $MIMIC_OUTPUT_DIR/transfer_from_mimic/discrimination.csv"
echo "  benchmark:$MIMIC_OUTPUT_DIR/benchmark/new_acutemi*/test_metrics.csv"

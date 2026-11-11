#!/bin/bash
#SBATCH --job-name=tkg_temporal
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/temporal_%j.log
#SBATCH --error=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/temporal_%j.log

# ---------------------------------------------------------------------------
# CardioMM-TKG: temporal-split robustness experiment
#
# Train on index admissions from 2008-2015, validate on ~2014-2018, test on
# 2017+ (see TKG_SPLIT_MODE in src/config.py). Reuses the cohort and TKG facts
# from the main run -- only the split and everything downstream of it are
# recomputed, in a separate output dir so the main results are untouched.
#
# Steps:
#   0. link cohort / facts / node index from the main tkg_output
#   1. prep_modeling (temporal)  -- split + train-only normalization
#   2. tests_integrity           -- hard gate
#   3. build_ontology, build_cooccurrence (train-only co-occurrence)
#   4. 5 seeds x {Cox+XGBoost, TKG-Transformer, patient-graph model}
#   5. compare_survival + multi_seed_summary
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=/home/hice1/hsathu3/scratch/TKG_MIMIC
PY=/home/hice1/hsathu3/.conda/envs/fedhealth/bin/python
MAIN_OUT=/home/hice1/hsathu3/scratch/TKG_MIMIC/tkg_output

export MIMIC_HOSP_DIR=/storage/ice1/9/3/hsathu3/mimic_work/mimic_data/3.1/hosp
export MIMIC_ICU_DIR=/storage/ice1/9/3/hsathu3/mimic_work/mimic_data/3.1/icu
export MIMIC_OUTPUT_DIR=/home/hice1/hsathu3/scratch/TKG_MIMIC/tkg_output_temporal
export TKG_SPLIT_MODE=temporal

cd "$REPO"
mkdir -p "$REPO/logs" "$MIMIC_OUTPUT_DIR"

echo "Job ${SLURM_JOB_ID:-local}  started $(date)  output: $MIMIC_OUTPUT_DIR"

# 0. Split-independent inputs from the main run
for f in cohort.csv cohort_cascade.csv tkg_facts.csv node_index.csv; do
    ln -sfn "$MAIN_OUT/$f" "$MIMIC_OUTPUT_DIR/$f"
done

# 1-3. Split, integrity gate, graph layers
$PY -u -m src.prep_modeling
$PY -u -m src.tests_integrity
$PY -u -m src.ablations.build_ontology
$PY -u -m src.ablations.build_cooccurrence

# 4. Models, 5 seeds (seed changes model randomness only, never the split)
for s in 42 43 44 45 46; do
    echo "=== seed $s  $(date) ==="
    TKG_SEED=$s $PY -u -m src.baselines_survival
    TKG_SEED=$s $PY -u -m src.tgn_survival
    TKG_SEED=$s $PY -u -m src.ablations.patient_graph_gnn
done

# 5. Summaries
$PY -u -m src.compare_survival
$PY -u -m src.multi_seed_summary
$PY -u -m src.ablations.patient_gnn_multi_seed_summary

echo "Done $(date). Compare $MIMIC_OUTPUT_DIR/survival_comparison_test.csv"
echo "against $MAIN_OUT/survival_comparison_test.csv"

#!/bin/bash
#SBATCH --job-name=eicu_transfer
#SBATCH --partition=ice-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/eicu_%j.log
#SBATCH --error=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/eicu_%j.log

# ---------------------------------------------------------------------------
# eICU-only re-run after bug fixes.
# Prerequisites already complete:
#   - MIMIC pipeline (cohort → build_tkg → prep_modeling → tests_integrity)
#   - baselines_survival  → transfer_artifacts/ (Cox + XGBoost pickles)
#
# This job runs ONLY:
#   src.eicu.transfer_eval  (rebuilds cohort_eicu.csv + scores + report)
#
# Bugs fixed:
#   1. cohort: unitvisitnumber ordering replaces cross-year-only design
#   2. features: vitalPeriodic read in 500K-row chunks (avoids 7.4GB OOM)
#   3. scoring: Cox batched in 500-patient slices (avoids dense-matrix OOM)
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=/home/hice1/hsathu3/scratch/TKG_MIMIC
PY=/home/hice1/hsathu3/.conda/envs/fedhealth/bin/python

export MIMIC_HOSP_DIR=/storage/ice1/9/3/hsathu3/mimic_work/mimic_data/3.1/hosp
export MIMIC_ICU_DIR=/storage/ice1/9/3/hsathu3/mimic_work/mimic_data/3.1/icu
export MIMIC_OUTPUT_DIR=/home/hice1/hsathu3/scratch/TKG_MIMIC/tkg_output
export EICU_DATA_DIR=/home/hice1/hsathu3/scratch/eICU_dataset/physionet.org/files/eicu-crd/2.0
export EICU_OUTPUT_DIR=/home/hice1/hsathu3/scratch/eICU_transfer_output

mkdir -p "$EICU_OUTPUT_DIR"
cd "$REPO"

echo "======================================================================"
echo "Job $SLURM_JOB_ID  started $(date)"
echo "Node: $SLURMD_NODENAME   CPUs: $SLURM_CPUS_PER_TASK   Mem: 64G"
echo "MIMIC artifacts: $MIMIC_OUTPUT_DIR/baselines_survival/transfer_artifacts/"
echo "eICU data      : $EICU_DATA_DIR"
echo "eICU output    : $EICU_OUTPUT_DIR"
echo "======================================================================"

$PY -u -m src.eicu.transfer_eval

echo ""
echo "======================================================================"
echo "Done  $(date)"
echo "  eICU discrimination : $EICU_OUTPUT_DIR/eicu_transfer_discrimination.csv"
echo "  eICU calibration    : $EICU_OUTPUT_DIR/eicu_transfer_calibration.csv"
echo "  eICU report         : $EICU_OUTPUT_DIR/eicu_transfer_report.txt"
echo "======================================================================"

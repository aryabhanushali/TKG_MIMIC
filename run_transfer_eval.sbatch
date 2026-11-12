#!/bin/bash
#SBATCH --job-name=tkg_transfer
#SBATCH --partition=ice-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/transfer_%j.log
#SBATCH --error=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/transfer_%j.log

# ---------------------------------------------------------------------------
# CardioMM-TKG: MIMIC-IV pipeline + eICU portability evaluation
#
# Runs the minimum MIMIC-IV pipeline steps needed to produce transfer
# artifacts (feature_meta.pkl, cox_*.pkl, xgb_*.pkl), then immediately
# scores eICU patients with the trained models.
#
# Steps:
#   1. cohort          -- build MIMIC-IV cohort
#   2. build_tkg       -- build 14.5M-fact knowledge graph  [slowest step]
#   3. prep_modeling   -- train/val/test split + feature table
#   4. tests_integrity -- hard-gate sanity checks
#   5. baselines_survival -- fit Cox + XGBoost, write transfer_artifacts/
#   6. eicu.transfer_eval -- score eICU patients, write portability report
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=/home/hice1/hsathu3/scratch/TKG_MIMIC
PY=/home/hice1/hsathu3/.conda/envs/fedhealth/bin/python
LOG_DIR="$REPO/logs"

mkdir -p "$LOG_DIR"

echo "======================================================================"
echo "Job $SLURM_JOB_ID  started $(date)"
echo "Node: $SLURMD_NODENAME   CPUs: $SLURM_CPUS_PER_TASK   Mem: 48G"
echo "======================================================================"

# ---------------------------------------------------------------------------
# 0. Install scikit-survival if missing  (pip install into fedhealth env)
# ---------------------------------------------------------------------------
echo ""
echo "[0] Checking scikit-survival..."
if ! $PY -c "import sksurv" 2>/dev/null; then
    echo "  sksurv not found — installing into fedhealth env..."
    /home/hice1/hsathu3/.conda/envs/fedhealth/bin/pip install \
        --quiet scikit-survival
    echo "  installed."
else
    echo "  sksurv already present."
fi

# ---------------------------------------------------------------------------
# Environment variables for MIMIC-IV path overrides
# ---------------------------------------------------------------------------
export MIMIC_HOSP_DIR=/storage/ice1/9/3/hsathu3/mimic_work/mimic_data/3.1/hosp
export MIMIC_ICU_DIR=/storage/ice1/9/3/hsathu3/mimic_work/mimic_data/3.1/icu
export MIMIC_OUTPUT_DIR=/home/hice1/hsathu3/scratch/TKG_MIMIC/tkg_output

# eICU path overrides
export EICU_DATA_DIR=/home/hice1/hsathu3/scratch/eICU_dataset/physionet.org/files/eicu-crd/2.0
export EICU_OUTPUT_DIR=/home/hice1/hsathu3/scratch/eICU_transfer_output

cd "$REPO"

# Create output directories (cohort.py will fail if tkg_output/ doesn't exist)
mkdir -p "$MIMIC_OUTPUT_DIR"
mkdir -p "$EICU_OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 1. Cohort
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[1/6] Building MIMIC-IV cohort  $(date)"
echo "======================================================================"
$PY -u -m src.cohort

# ---------------------------------------------------------------------------
# 2. Build TKG  (largest step — ~1–3 hrs depending on chartevents size)
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[2/6] Building TKG facts  $(date)"
echo "======================================================================"
$PY -u -m src.build_tkg

# ---------------------------------------------------------------------------
# 3. Prep modeling
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[3/6] Preparing modeling artifacts  $(date)"
echo "======================================================================"
$PY -u -m src.prep_modeling

# ---------------------------------------------------------------------------
# 4. Integrity checks (hard gate)
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[4/6] Running integrity checks  $(date)"
echo "======================================================================"
$PY -u -m src.tests_integrity

# ---------------------------------------------------------------------------
# 5. Survival baselines (Cox + XGBoost) — writes transfer_artifacts/
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[5/6] Fitting Cox + XGBoost baselines  $(date)"
echo "======================================================================"
$PY -u -m src.baselines_survival

# ---------------------------------------------------------------------------
# 6. eICU portability evaluation
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[6/6] Running eICU model-transfer evaluation  $(date)"
echo "======================================================================"
$PY -u -m src.eicu.transfer_eval

echo ""
echo "======================================================================"
echo "Done  $(date)"
echo ""
echo "Key outputs:"
echo "  MIMIC transfer artifacts : $MIMIC_OUTPUT_DIR/baselines_survival/transfer_artifacts/"
echo "  eICU discrimination      : $EICU_OUTPUT_DIR/eicu_transfer_discrimination.csv"
echo "  eICU calibration         : $EICU_OUTPUT_DIR/eicu_transfer_calibration.csv"
echo "  eICU portability report  : $EICU_OUTPUT_DIR/eicu_transfer_report.txt"
echo "======================================================================"

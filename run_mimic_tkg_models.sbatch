#!/bin/bash
#SBATCH --job-name=tkg_mimic_models
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=42-46
#SBATCH --output=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/mimic_models_%A_s%a.log
#SBATCH --error=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/mimic_models_%A_s%a.log

# ---------------------------------------------------------------------------
# MIMIC-IV: (re)train all four model families for one seed per array task.
#
# Regenerates per-patient test predictions for the TKG-Transformer and the
# patient-graph model on this cluster (only seed-42 Cox / XGBoost predictions
# existed here), so subgroup / sample-size analyses can run on MIMIC.
# Uses the existing tkg_output/modeling split (unchanged random 70/15/15).
#
# The graph layers (ontology / co-occurrence) are built once by seed 42's
# task; the other seeds wait for them. Submit with:
#   sbatch run_mimic_tkg_models.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/home/hice1/hsathu3/scratch/TKG_MIMIC
PY=/home/hice1/hsathu3/.conda/envs/fedhealth/bin/python
export MIMIC_HOSP_DIR=/storage/ice1/9/3/hsathu3/mimic_work/mimic_data/3.1/hosp
export MIMIC_ICU_DIR=/storage/ice1/9/3/hsathu3/mimic_work/mimic_data/3.1/icu
export MIMIC_OUTPUT_DIR=$REPO/tkg_output
SEED=${SLURM_ARRAY_TASK_ID:-42}
export TKG_SEED=$SEED
cd "$REPO"
echo "Job ${SLURM_JOB_ID:-local} seed $SEED started $(date) on $(hostname)"

READY=$MIMIC_OUTPUT_DIR/.graph_layers_ready
if [ "$SEED" = "42" ]; then
    rm -f "$READY"
    $PY -u -m src.ablations.build_ontology
    $PY -u -m src.ablations.build_cooccurrence
    touch "$READY"
else
    while [ ! -f "$READY" ]; do sleep 60; done
fi

$PY -u -m src.baselines_survival
$PY -u -m src.tgn_survival
$PY -u -m src.ablations.patient_graph_gnn
echo "Done seed $SEED $(date)"

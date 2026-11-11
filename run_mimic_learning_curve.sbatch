#!/bin/bash
#SBATCH --job-name=tkg_mimic_lc
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
# NVIDIA nodes with CUDA 13 drivers only: the env has torch 2.11+cu130, which
# has no V100 (sm_70) kernels and cannot use the AMD (MI210 / R9700) nodes.
#SBATCH --constraint=cuda13
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/mimic_lc_%A_%a.log
#SBATCH --error=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/mimic_lc_%A_%a.log

# ---------------------------------------------------------------------------
# MIMIC-IV training-set-size learning curve.
#
# Question: does the TKG models' standing against Cox / XGBoost change as the
# number of training patients (and so training events per disease) shrinks?
# This isolates sample size from disease identity and from dataset (the
# MIMIC-vs-EHRSHOT comparison confounds both).
#
# Each array task keeps a nested, hashed fraction of TRAIN patients
# (TKG_TRAIN_FRAC; see config.py). Val and test are the unchanged MIMIC
# val/test sets, so every point on the curve is scored on identical patients.
# frac=1.0 is the main run (run_mimic_tkg_models.sbatch). 0.05 of MIMIC train
# (~1.2k patients) is below EHRSHOT's per-fold train size (~1.65k).
#
# Survival baselines are re-tuned on validation at each fraction (as for
# EHRSHOT); the TKG models keep their MIMIC configuration. This favours the
# baselines, so any relative TKG gain at small n is conservative.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/home/hice1/hsathu3/scratch/TKG_MIMIC
PY=/home/hice1/hsathu3/.conda/envs/fedhealth/bin/python
FRACS=(0.05 0.1 0.25 0.5)
FRAC=${FRACS[${SLURM_ARRAY_TASK_ID:-0}]}
SEEDS=${SEEDS:-"42 43 44"}
MAIN=$REPO/tkg_output
export MIMIC_HOSP_DIR=/storage/ice1/9/3/hsathu3/mimic_work/mimic_data/3.1/hosp
export MIMIC_ICU_DIR=/storage/ice1/9/3/hsathu3/mimic_work/mimic_data/3.1/icu
export MIMIC_OUTPUT_DIR=$REPO/tkg_output_lc/frac$FRAC
export TKG_TRAIN_FRAC=$FRAC
cd "$REPO"
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no usable CUDA GPU on this node -- refusing to train on CPU')"
$PY -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
mkdir -p "$MIMIC_OUTPUT_DIR"
echo "Job ${SLURM_JOB_ID:-local} frac $FRAC started $(date) on $(hostname)"
for f in cohort.csv cohort_cascade.csv tkg_facts.csv node_index.csv; do
    ln -sfn "$MAIN/$f" "$MIMIC_OUTPUT_DIR/$f"
done

$PY -u -m src.prep_modeling
$PY -u -m src.tests_integrity
$PY -u -m src.ablations.build_ontology
$PY -u -m src.ablations.build_cooccurrence
rm -f "$MIMIC_OUTPUT_DIR/hparams_survival.json"
$PY -u -m src.ablations.cox_regularization_sweep
$PY -u -m src.ablations.xgb_survival_sweep
$PY -u -m src.ehrshot.select_hparams
for s in $SEEDS; do
    echo "=== frac $FRAC seed $s  $(date)"
    TKG_SEED=$s $PY -u -m src.baselines_survival
    TKG_SEED=$s $PY -u -m src.tgn_survival
    TKG_SEED=$s $PY -u -m src.ablations.patient_graph_gnn
done
echo "Done frac $FRAC $(date)"

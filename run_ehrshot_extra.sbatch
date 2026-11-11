#!/bin/bash
#SBATCH --job-name=tkg_ehrshot_extra
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
# NVIDIA nodes with CUDA 13 drivers only: the env has torch 2.11+cu130, which
# has no V100 (sm_70) kernels and cannot use the AMD (MI210 / R9700) nodes.
#SBATCH --constraint=cuda13
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/ehrshot_extra_%A_%a.log
#SBATCH --error=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/ehrshot_extra_%A_%a.log

# ---------------------------------------------------------------------------
# Two EHRSHOT follow-ups (array task 0 / 1):
#
#  0. new_acutemi benchmark task, 5 seeds, full training (replaces the 3-epoch
#     smoke run; the TKG-Transformer now uses the same MIN_EPOCHS=15
#     checkpoint floor as the survival model).
#  1. Fact-density parity ablation: rebuild the EHRSHOT TKG with MIMIC's
#     subsampling (vitals 10%, labs 30%) and rerun the full 5-fold CV into
#     tkg_output_ehrshot_parity/. Unsubsampled EHRSHOT has vitals = 48.6% of
#     facts, which may crowd the TKG-Transformer's 256-event window.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=/home/hice1/hsathu3/scratch/TKG_MIMIC
PY=/home/hice1/hsathu3/.conda/envs/fedhealth/bin/python
export EHRSHOT_DATA_DIR=/storage/ice1/9/3/hsathu3/EHRSHOT/meds_omop_ehrshot
export EHRSHOT_MEDS_DIR=/storage/ice1/9/3/hsathu3/EHRSHOT/EHRSHOT_MEDS
export EHRSHOT_ASSETS_DIR=/storage/ice1/9/3/hsathu3/EHRSHOT/EHRSHOT_ASSETS
cd "$REPO"
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no usable CUDA GPU on this node -- refusing to train on CPU')"
$PY -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
TASK=${SLURM_ARRAY_TASK_ID:-0}
echo "Job ${SLURM_JOB_ID:-local} task $TASK started $(date) on $(hostname)"

if [ "$TASK" = "0" ]; then
    export MIMIC_OUTPUT_DIR=$REPO/tkg_output_ehrshot
    for s in 42 43 44 45 46; do
        echo "=== benchmark seed $s $(date)"
        TKG_SEED=$s $PY -u -m src.ehrshot.benchmark_binary --task new_acutemi
    done
else
    PAR=$REPO/tkg_output_ehrshot_parity
    mkdir -p "$PAR"
    for f in cohort.csv cohort_cascade.csv; do
        ln -sfn "$REPO/tkg_output_ehrshot/$f" "$PAR/$f"
    done
    MIMIC_OUTPUT_DIR=$PAR EHRSHOT_VITAL_SAMPLE_RATE=0.1 EHRSHOT_LAB_SAMPLE_RATE=0.3 \
        $PY -u -m src.ehrshot.build_tkg_ehrshot
    MAIN_OUT=$PAR bash "$REPO/run_ehrshot_cv.sbatch"
fi
echo "Done task $TASK $(date)"

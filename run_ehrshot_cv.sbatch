#!/bin/bash
#SBATCH --job-name=tkg_ehrshot_cv
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/ehrshot_cv_%j.log
#SBATCH --error=/home/hice1/hsathu3/scratch/TKG_MIMIC/logs/ehrshot_cv_%j.log

# ---------------------------------------------------------------------------
# EHRSHOT 5-fold cross-validation of the native survival models.
#
# EHRSHOT's cohort (~2.8k patients, 50-230 events per cause) is too small for
# a single 70/15/15 split: the test set held 4-20 events per cause. Here every
# patient is tested exactly once (fold k = test, fold k+1 = val, rest = train),
# and src/ehrshot/cv_summary.py reports both full-cohort out-of-fold AUROC
# (with bootstrap CI) and the per-fold mean +/- sd.
#
# Needs the main EHRSHOT run (run_ehrshot.sbatch steps 1 and 6) to exist:
# cohort / TKG are shared, read via symlinks.
# Per fold: prep_modeling (cv) -> tests_integrity -> ontology / co-occurrence
#           -> the fold's own validation sweeps -> 5 seeds x 3 model families.
# Usage: sbatch run_ehrshot_cv.sbatch   (or: bash run_ehrshot_cv.sbatch)
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=${REPO:-/home/hice1/hsathu3/scratch/TKG_MIMIC}
PY=${PY:-/home/hice1/hsathu3/.conda/envs/fedhealth/bin/python}
MAIN_OUT=${MAIN_OUT:-$REPO/tkg_output_ehrshot}
CV_ROOT=$MAIN_OUT/cv
FOLDS=${FOLDS:-5}
SEEDS=${SEEDS:-"42 43 44 45 46"}

export TKG_SPLIT_MODE=cv TKG_CV_FOLDS=$FOLDS
cd "$REPO"
mkdir -p "$REPO/logs" "$CV_ROOT"
echo "Job ${SLURM_JOB_ID:-local}  started $(date)  cv root: $CV_ROOT"

for k in $(seq 0 $((FOLDS - 1))); do
    export TKG_CV_FOLD=$k MIMIC_OUTPUT_DIR=$CV_ROOT/fold$k
    mkdir -p "$MIMIC_OUTPUT_DIR"
    echo "######## fold $k  $(date)"
    for f in cohort.csv cohort_cascade.csv tkg_facts.csv node_index.csv; do
        ln -sfn "$MAIN_OUT/$f" "$MIMIC_OUTPUT_DIR/$f"
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
        echo "=== fold $k seed $s  $(date)"
        TKG_SEED=$s $PY -u -m src.baselines_survival
        TKG_SEED=$s $PY -u -m src.tgn_survival
        TKG_SEED=$s $PY -u -m src.ablations.patient_graph_gnn
    done
done

$PY -u -m src.ehrshot.cv_summary "$CV_ROOT" --folds "$FOLDS"
echo "Done $(date). Summary: $CV_ROOT/cv_summary.csv"

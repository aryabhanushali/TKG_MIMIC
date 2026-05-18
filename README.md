# CardioMM-TKG

A multimodal temporal knowledge graph and benchmark for cardiometabolic-to-circulatory disease progression on MIMIC-IV. Submitted to the IEEE JBHI special issue on *Knowledge Graphs and Multimodal Data Fusion in Biomedical Informatics*.

## Cohort

39,950 MIMIC-IV patients with at least one cardiometabolic index admission (T2D, hypertension, dyslipidemia, obesity, or metabolic syndrome by ICD-9 / ICD-10), at least one subsequent admission, ≥90 days of follow-up, and no prior circulatory endpoint. Five competing-risks endpoints: MI, Stroke, HF, AF, PAD; otherwise censored.

## Pipeline

Run sequentially from the project root with `python -u -m src.<module>`, or `python main.py` to run all eleven steps in order.

| step | module | output |
|---|---|---|
| 1 | `src.cohort` | `tkg_output/cohort.csv` (cohort + Charlson) |
| 2 | `src.build_tkg` | `tkg_output/tkg_facts.csv` (~13M temporal facts, 8 modalities) + `node_index.csv` |
| 3 | `src.validate_tkg` | `tkg_output/validation_report.txt` (8 sanity checks) |
| 4 | `src.visualize_tkg` | `tkg_output/figures/fig{1,2,3,4}_*.png` |
| 5 | `src.notes_extract` | `tkg_output/notes/discharge_cohort.csv.gz` (cohort-filtered notes) |
| 6 | `src.notes_embed` | `tkg_output/notes/note_embeddings.npy` (Bio_ClinicalBERT [CLS]) |
| 7 | `src.notes_aggregate` | `tkg_output/notes/patient_note_emb.npy` (per-patient mean-pool) |
| 8 | `src.prep_modeling` | `tkg_output/modeling/{events,labels,splits,static_features,node_metadata,edge_types}.csv` |
| 9 | `src.baselines_survival` | `tkg_output/baselines_survival/test_metrics.csv` (Cox + XGBoost-Survival) |
| 10 | `src.tgn_survival` | `tkg_output/tgn_survival/test_metrics.csv` + `predictions_test.csv` (DeepHit-style TGN) |
| 11 | `src.compare_survival` | `tkg_output/survival_comparison_test.csv` (Table 1) |

Optional post-hoc analysis: `src.make_figures` (paper figures 5/6/7/12), `src.explain` and `src.explain_discriminative` (attention-based explainability, figures 9/10/11).

## Avoiding leakage

The test set is held out and used exactly once, at the final evaluation in step 10.

- **Stratified split.** Patient-level 70/15/15 by `endpoint_type`, fixed `SEED=42` in `config.py`, generated once in `prep_modeling.py`.
- **Train-only feature space.** The bag-of-codes vocabulary, the per-concept value-summary columns, and the TGN concept embedding table are restricted to concepts that appear in training patients. Test-only concepts are routed to a single UNK index and never receive a learned embedding (`tgn_model.py`, `baseline.py`).
- **Train-only normalization.** Per-concept value z-scores (`prep_modeling.py`), static-feature `StandardScaler` (`baseline.py`, `baselines_survival.py`), and BioBERT note z-scores (`tgn_model.py`) are all fit on training patients only.
- **Train-only time discretization.** DeepHit time-bin edges are quantile-fit on training durations only (`tgn_survival._make_time_bins`).
- **No temporal leakage.** `build_tkg.py` enforces `timestamp_start < endpoint_date` for every fact, and `prep_modeling.py` further restricts to the pre-index window `[index_date - 1825 days, index_date)`. Check 1 of `validate_tkg.py` reverifies this on the saved fact table.
- **Aligned prognostic cutoff.** All modalities (codes, drugs, labs, vitals, OMR, ICU charts / IV / output, notes) are filtered to `charttime < index_date`, so no modality sees information from a time point another modality does not.
- **Early stopping on validation.** `tgn_survival.py` selects the best epoch by mean per-cause AUROC at 3 years on the validation set; test predictions are never inspected during training.
- **Class weighting.** Inverse-frequency weights are computed from training cause counts only (`tgn_survival.py`).

## Repo layout

```
TKG_MIMIC/
|-- main.py                       # orchestrator
|-- README.md
|-- mimic_data/                   # raw MIMIC-IV csv.gz (not redistributed; gitignored)
|-- tkg_output/                   # derived artifacts (gitignored)
|   |-- cohort.csv
|   |-- tkg_facts.csv
|   |-- node_index.csv
|   |-- validation_report.txt
|   |-- figures/
|   |-- modeling/
|   |-- notes/
|   |-- baselines_survival/
|   |-- tgn_survival/
|   `-- survival_comparison_test.csv
`-- src/
    |-- config.py                 # hyperparameters + ICD / drug / lab vocabularies
    |-- cohort.py                 # cohort + Charlson Comorbidity Index
    |-- build_tkg.py              # 8-modality TKG with value_num
    |-- validate_tkg.py           # 8 sanity checks
    |-- visualize_tkg.py          # cohort + TKG figures
    |-- notes_{extract,embed,aggregate}.py
    |-- prep_modeling.py          # pre-index event table + stratified split
    |-- baseline.py               # multiclass baselines (LogReg, XGBoost) + feature helpers
    |-- baselines_survival.py     # Cox + XGBoost-Survival, per cause
    |-- tgn_model.py              # TKG-Transformer encoder
    |-- tgn_survival.py           # DeepHit-style competing-risks head + training
    |-- compare_survival.py       # final per-cause AUROC table
    |-- make_figures.py           # paper figures 5/6/7/12
    |-- explain.py                # attention-based concept importance
    |-- explain_discriminative.py # per-cause discriminative-lift analysis
    `-- ablations/                # not on the main path
        |-- build_ontology.py     # naive ICD / drug-class / lab-category ontology
        `-- hetero_gnn.py         # R-GCN over ontology edges + temporal encoder
```

## Environment

Python 3.10 (miniforge / conda env `tkg`):

```
pandas 2.3.3
numpy 2.2.6
scikit-learn 1.7.2
xgboost 3.2.0
torch 2.10.0    # Apple MPS or CUDA
torch_geometric 2.7.0
pycox / lifelines / scikit-survival
transformers   # Bio_ClinicalBERT
matplotlib / seaborn / tqdm
```

## Data

MIMIC-IV v3.1 (`hosp/` + `icu/`) and MIMIC-IV-Note v2.2 (`note/discharge.csv.gz`). Credentialed access via PhysioNet is required. The repository contains no patient data; only the scripts that build derived artifacts from the raw csv.gz files.

## Reproducing the paper's Table 1

```bash
PY=/path/to/miniforge3/envs/tkg/bin/python
$PY -u -m src.cohort
$PY -u -m src.build_tkg
$PY -u -m src.notes_extract
$PY -u -m src.notes_embed
$PY -u -m src.notes_aggregate
$PY -u -m src.prep_modeling
$PY -u -m src.baselines_survival
$PY -u -m src.tgn_survival
$PY -u -m src.compare_survival
```

The final table is written to `tkg_output/survival_comparison_test.csv`.

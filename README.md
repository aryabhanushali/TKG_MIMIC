# CardioMM-TKG

A multimodal temporal knowledge graph and benchmark for **cardiometabolic-to-circulatory disease progression** on MIMIC-IV. Submitted to the IEEE JBHI special issue on *Knowledge Graphs and Multimodal Data Fusion in Biomedical Informatics*.

## Cohort

- 39,950 MIMIC-IV patients with at least one cardiometabolic index admission (T2D, hypertension, dyslipidemia, obesity, metabolic syndrome by ICD-9 or ICD-10), at least one subsequent admission, ≥90 days of follow-up, and no prior circulatory endpoint.
- Five competing-risks circulatory endpoints: **MI, Stroke, HF, AF, PAD**; censored otherwise.

## Pipeline (end-to-end reproducible)

Run sequentially from project root with `python -u -m src.<module>`, or `python main.py` for the whole orchestration.

| step | module | output |
|---|---|---|
| 1 | `src.cohort` | `tkg_output/cohort.csv` (cohort + Charlson) |
| 2 | `src.build_tkg` | `tkg_output/tkg_facts.csv` (~13M temporal facts, 8 relations) + `node_index.csv` |
| 3 | `src.validate_tkg` | `tkg_output/validation_report.txt` (8 sanity checks) |
| 4 | `src.visualize_tkg` | `tkg_output/figures/fig{1,2,3,4}_*.png` |
| 5 | `src.notes_extract` | `tkg_output/notes/discharge_cohort.csv.gz` (cohort-filtered notes) |
| 6 | `src.notes_embed` | `tkg_output/notes/note_embeddings.npy` (Bio_ClinicalBERT [CLS]) |
| 7 | `src.notes_aggregate` | `tkg_output/notes/patient_note_emb.npy` (per-patient mean-pool) |
| 8 | `src.prep_modeling` | `tkg_output/modeling/{events,labels,splits,static_features,node_metadata,edge_types}.csv` |
| 9 | `src.baselines_survival` | `tkg_output/baselines_survival/test_metrics.csv` (Cox + XGBoost-Survival) |
| 10 | `src.tgn_survival` | `tkg_output/tgn_survival/test_metrics.csv` (DeepHit-style TGN) + `predictions_test.csv` |
| 11 | `src.compare_survival` | `tkg_output/survival_comparison_test.csv` (paper Table 1) |

## Scientific-validity guarantees

The test set is **used exactly once**, at the final evaluation of step 10.

- **Stratified split**: patient-level 70/15/15 by `endpoint_type`, fixed `SEED=42` in `config.py`. Generated once in `prep_modeling.py`.
- **Train-only feature space**: the bag-of-codes concept space, value-summary feature columns, and TGN concept-embedding table are restricted to concepts observed in *training* patients. Test-only concepts are routed to a single **UNK** index — they never get a learned embedding (`tgn_model.py`, `baseline.py`).
- **Train-only normalization**: per-concept value z-scores (`prep_modeling.py`), static-feature `StandardScaler` (`baseline.py`, `baselines_survival.py`), BioBERT note-embedding z-scores (`tgn_model.py`) — all use training-patient statistics only.
- **Train-only time discretization**: DeepHit time-bin edges are quantile-fit on `train_durations` only (`tgn_survival.py:_make_time_bins`).
- **No temporal leakage**: the TKG `build_tkg.py` enforces `timestamp_start < endpoint_date` for every fact, then `prep_modeling.py` further restricts to the pre-index window `[index_date - 1825 days, index_date)` for prognostic prediction. Check 1 of `validate_tkg.py` reverifies this on the saved fact table.
- **Single prognostic moment**: all modalities (codes, drugs, labs, vitals, OMR, ICU charts/IV/output, notes) are filtered to `charttime < index_date` — there is no modality that sees information from a time point another modality does not.
- **Early stopping on validation**: `tgn_survival.py` selects best epoch by **val mean per-cause AUROC at 3-yr horizon**; never references test predictions during training.
- **Class weighting**: inverse-frequency weights computed from training cause counts only (`tgn_survival.py`).

## Repo layout

```
TKG_MIMIC/
├── main.py                       # orchestrator
├── README.md
├── mimic_data/                   # raw MIMIC-IV csv.gz files (not redistributed)
├── tkg_output/                   # all derived artifacts
│   ├── cohort.csv
│   ├── tkg_facts.csv             # 13.4M temporal facts
│   ├── node_index.csv
│   ├── validation_report.txt
│   ├── figures/                  # 4 cohort/TKG figures + survival per-cause
│   ├── modeling/                 # events, labels, splits, static features
│   ├── notes/                    # cohort notes + BioBERT embeddings
│   ├── baselines_survival/       # Cox + XGB-Survival metrics + predictions
│   ├── tgn_survival/             # TGN model + metrics + predictions
│   └── survival_comparison_test.csv   # paper Table 1
└── src/
    ├── config.py                 # all hyperparameters + ICD/drug/lab vocabs
    ├── cohort.py                 # cohort + Charlson Comorbidity Index
    ├── build_tkg.py              # 8-modality TKG with value_num
    ├── validate_tkg.py           # 8 sanity checks
    ├── visualize_tkg.py          # cohort + TKG figures
    ├── notes_{extract,embed,aggregate}.py   # multimodal pillar
    ├── prep_modeling.py          # pre-index event table + stratified split
    ├── baseline.py               # multiclass baselines (LogReg, XGBoost) + feature helpers
    ├── baselines_survival.py     # Cox + XGBoost-Survival (per cause)
    ├── tgn_model.py              # TKG-Transformer encoder (multimodal-aware)
    ├── tgn_survival.py           # DeepHit-style competing-risks head + training
    ├── compare_survival.py       # paper Table 1 generator
    └── ablations/                # naive ontology + hetero-GNN (not in main path)
        ├── build_ontology.py
        └── hetero_gnn.py
```

## Environment

Python 3.10 (miniforge / conda env `tkg`):

```
pandas 2.3.3
numpy 2.2.6
scikit-learn 1.7.2
xgboost 3.2.0
torch 2.10.0  (Apple-MPS or CUDA)
torch_geometric 2.7.0
pycox / lifelines / scikit-survival
transformers (for Bio_ClinicalBERT)
matplotlib / seaborn / tqdm
```

## Data

MIMIC-IV v3.1 (`hosp/` + `icu/`) and MIMIC-IV-Note v2.2 (`note/discharge.csv.gz`). Credentialed access required via PhysioNet. The repo contains no patient data — only the scripts that build derived artifacts from the raw csv.gz files.

## Reproducing the paper Table 1

```bash
PY=/Users/.../miniforge3/envs/tkg/bin/python
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

Final table is at `tkg_output/survival_comparison_test.csv`.

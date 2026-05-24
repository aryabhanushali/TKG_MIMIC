# Temporal Knowledge Graph Modeling for Cardiometabolic Disease Progression using MIMIC-IV

This project builds a multimodal Temporal Knowledge Graph (TKG) from MIMIC-IV to model cardiometabolic disease progression and predict future adverse outcomes using temporal graph neural networks and survival modeling.

The pipeline integrates:
- longitudinal EHR events
- diagnoses
- procedures
- medications
- labs
- ICU trajectories
- discharge notes
- multimodal clinical embeddings

The goal is to model how patient conditions evolve over time and use those temporal patterns to predict future cardiometabolic complications.

---
Project represents each patient history as a sequence of timestamped clinical events inside a Temporal Knowledge Graph (TKG), allowing the model to reason about:
- event ordering
- progression patterns
- temporal relationships between diagnoses, labs, medications, and ICU events

The project focuses on predicting:
- Myocardial Infarction (MI)
- Stroke
- Heart Failure (HF)
- Atrial Fibrillation (AF)
- Peripheral Artery Disease (PAD)

---

# Dataset

Source:
- MIMIC-IV (Beth Israel Deaconess Medical Center ICU + hospital dataset)

Raw tables used:
- admissions
- patients
- diagnoses_icd
- procedures_icd
- prescriptions
- labevents
- icustays
- omr
- chartevents
- inputevents
- outputevents
- discharge notes

Final cohort:
- ~39,950 patients
- cardiometabolic disease population
- longitudinal multimodal EHR trajectories

---

# Pipeline Overview

```text
RAW MIMIC-IV
   admissions, patients, diagnoses_icd, procedures_icd,
   prescriptions, labevents, icustays, omr,
   chartevents, inputevents, outputevents, discharge

      |
      v

1. src/cohort.py
   - builds cardiometabolic cohort
   - computes Charlson Comorbidity Index
   --> cohort.csv

      |
      v

2. src/build_tkg.py
   - constructs temporal knowledge graph
   --> tkg_facts.csv
   --> node_index.csv

      |
      v

3. src/validate_tkg.py
4. src/visualize_tkg.py

      |
      v

NOTES PIPELINE

5. src/notes_extract.py
6. src/notes_embed.py
7. src/notes_aggregate.py

      |
      v

8. src/prep_modeling.py
   - train/val/test split
   - event tables
   - labels
   - static features

      |
      +---------------------+
      |                     |
      v                     v

9. baselines_survival.py
   - Cox Survival
   - XGBoost Survival

10. tgn_survival.py
   - Temporal Graph Transformer
   - DeepHit survival head

      |
      v

11. compare_survival.py
12. explain.py
13. explain_discriminative.py
```

---

# Temporal Knowledge Graph

Instead of flattening EHR data into summary statistics, every clinical event becomes a temporal fact:

(patient, relation, concept, timestamp, value)

Examples:
- diagnosis at a certain date
- medication prescription
- ICU vital sign
- lab result with numerical value


Final graph:
- ~13.4 million temporal facts
- ~32k clinical concepts

---

# Multimodal Notes Pipeline

Incorporates discharge summary notes using:
- Bio_ClinicalBERT

Pipeline:
1. Extract discharge notes
2. Generate note embeddings
3. Mean-pool embeddings per patient
4. Apply strict pre-index cutoff to prevent leakage

This allows the model to combine:
- structured EHR data
- unstructured clinical language

into a shared patient representation.

---

# Modeling

## Baseline Models
Two strong tabular survival baselines were implemented:
- Cox Proportional Hazards
- XGBoost-Survival

These models use:
- diagnosis counts
- lab summary statistics
- medication indicators
- demographic features

## TGN-Survival

The main model is a Temporal Graph Transformer that reads each patient's event sequence over time.

Each event includes:
- concept embedding
- relation embedding
- temporal encoding
- numerical value embedding (when available)

The transformer uses attention to learn which historical events are most important for future risk prediction.

Final prediction head:
- DeepHit survival objective
- competing-risk prediction across five cardiometabolic outcomes

---

# Training Setup

- 70 / 15 / 15 train-validation-test split
- fixed random seed
- all normalization statistics computed using training set only
- strict temporal cutoff to prevent leakage
- early stopping on validation AUROC
- rare outcomes upweighted during training

---

# Results Summary

We compared:
- Cox Proportional Hazards
- XGBoost-Survival
- TGN-Survival

using AUROC at:
- 1 year
- 3 years
- 5 years

---

# Main Findings

## 1. Temporal modeling helped most for Stroke and MI

TGN-Survival performed best on:
- Stroke
- Myocardial Infarction (MI)

These diseases depend heavily on:
- blood pressure progression
- medication escalation
- lab trends
- event ordering over time

Because the transformer reads patient history as a sequence, it captures these temporal relationships better than flattened models.

---

## 2. Cox regression performed worst overall

Cox regression consistently had the lowest AUROC values, especially for:
- Stroke
- PAD

This suggests cardiometabolic progression is:
- nonlinear
- time-dependent
- multimodal

which traditional linear survival models struggle to capture.

---

## 3. XGBoost remained strong for some endpoints

For:
- Heart Failure (HF)
- Atrial Fibrillation (AF)
- PAD

XGBoost-Survival performed similarly to or slightly better than the transformer.

These outcomes may depend more on:
- cumulative disease burden
- abnormal lab values
- static physiologic state

which can already be summarized effectively using tabular features.

---

# Example 3-Year AUROC Results

| Endpoint | Cox | XGB-Survival | TGN-Survival |
|---|---|---|---|
| MI | 0.74 | 0.78 | 0.78 |
| Stroke | 0.72 | 0.73 | 0.77 |
| HF | 0.75 | 0.82 | 0.78 |
| AF | 0.72 | 0.79 | 0.75 |
| PAD | 0.62 | 0.80 | 0.77 |

---

# Explainability

The transformer attention mechanism allows inspection of which historical events influenced predictions most strongly.

After aggregating attention across correctly predicted patients, the model recovered clinically meaningful concepts:

| Endpoint | Important Concepts |
|---|---|
| MI | coronary atherosclerosis, catheterization, beta blockers |
| Stroke | diastolic BP, hyperlipidemia, amlodipine |
| HF | creatinine, BUN, hemoglobin |
| AF | atrial fibrillation history, mitral valve disease |
| PAD | vascular procedures, aspirin, clopidogrel |

This suggests the model is learning medically meaningful relationships rather than random correlations.

---

# Repository Structure

```text
src/
├── cohort.py
├── build_tkg.py
├── validate_tkg.py
├── visualize_tkg.py
├── notes_extract.py
├── notes_embed.py
├── notes_aggregate.py
├── prep_modeling.py
├── baseline.py
├── baselines_survival.py
├── tgn_model.py
├── tgn_survival.py
├── explain.py
├── explain_discriminative.py
└── compare_survival.py
```

---


- Python
- PyTorch
- PyTorch Geometric
- Transformers
- Bio_ClinicalBERT
- XGBoost
- Pandas
- NumPy
- Scikit-learn

---

# Overall Conclusion

This project demonstrates that temporal graph deep learning can improve prediction of cardiometabolic complications by modeling:
- longitudinal patient trajectories
- event ordering
- multimodal clinical information

The temporal graph transformer:
- strongly outperformed classical Cox models
- matched or exceeded strong machine learning baselines on several endpoints
- performed especially well on Stroke and MI
- generated interpretable explanations through attention analysis

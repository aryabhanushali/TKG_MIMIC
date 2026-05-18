
# Temporal Knowledge Graph Modeling for Cardiometabolic Disease Progression using MIMIC-IV

This project builds a multimodal Temporal Knowledge Graph (TKG) from MIMIC-IV to model cardiometabolic disease progression and predict future adverse outcomes using temporal graph neural networks and survival modeling.

The pipeline integrates:
- longitudinal EHR events
- ICD diagnoses
- procedures
- medications
- labs
- ICU trajectories
- discharge notes
- multimodal clinical embeddings

The goal is to learn temporal representations of patient trajectories and identify clinically meaningful patterns associated with future cardiometabolic risk.

# Pipeline Overview

```text
RAW MIMIC-IV
   admissions, patients, diagnoses_icd, procedures_icd,
   prescriptions, labevents, icustays, omr,
   chartevents, inputevents, outputevents, discharge

      |
      v

1. src/cohort.py
   - keep patients with cardiometabolic dx, no prior endpoint, >=2 admissions,
     >=90 day follow-up
   - compute Charlson Comorbidity Index
   --> tkg_output/cohort.csv   (39,950 patients)

      |
      v

2. src/build_tkg.py
   - for each modality, filter to per-patient window [index - 1825 d, endpoint)
   - normalize schemas
   - attach value_num where available
   --> tkg_output/tkg_facts.csv    (~13.4M facts)
   --> tkg_output/node_index.csv   (~32k concepts + 39,950 patients)

      |
      v

3. src/validate_tkg.py
   --> validation_report.txt

4. src/visualize_tkg.py
   --> figures/fig1..fig4

      |
      v

NOTES PILLAR (multimodal)

5. src/notes_extract.py
   - filters discharge notes to cohort patients

6. src/notes_embed.py
   - Bio_ClinicalBERT [CLS] embeddings per note

7. src/notes_aggregate.py
   - mean-pool note embeddings per patient
   - strict pre-index cutoff
   --> patient_note_emb.npy

      |
      v

8. src/prep_modeling.py
   - clips all events to pre-index window
   - creates 70/15/15 stratified split
   - computes train-only normalization statistics
   - assembles:
       events
       labels
       static features
       edge types
       node metadata

   --> tkg_output/modeling/

      |
      |                       
      v                      

9. baselines_survival.py

   For each outcome:
   - Cox Proportional Hazards
   - XGBoost Survival

   --> baselines_survival/test_metrics.csv

10. tgn_survival.py

   Builds TKGTransformer from src/tgn_model.py
   - temporal graph attention
   - multimodal node representations
   - DeepHit survival head
   - early stopping on validation AUROC

   --> tgn_survival/
       test_metrics.csv
       predictions_test.csv
       best_model.pt

  
      +-----------+-----------+
                  |
                  v

11. src/compare_survival.py
    - compares baseline vs TKG performance
    --> survival_comparison_test.csv

    src/make_figures.py
    --> figures/fig5,6,7,12

      |
      v

EXPLAINABILITY

12. src/explain.py
    - extracts pool-query attention scores
    - identifies top concepts contributing to each outcome

13. src/explain_discriminative.py
    - computes concept specificity via share-lift
    - highlights discriminative disease mechanisms

    --> figures/fig9,10,11


# Results Summary

We compared three different approaches for predicting future cardiometabolic complications:

- Cox Proportional Hazards (traditional survival model)
- XGBoost-Survival (tabular machine learning baseline)
- TGN-Survival (our temporal graph transformer model)

The model predicts:
- Myocardial Infarction (MI)
- Stroke
- Heart Failure (HF)
- Atrial Fibrillation (AF)
- Peripheral Artery Disease (PAD)

We evaluated performance using AUROC at 1-year, 3-year, and 5-year prediction horizons.

---

# Main Takeaways

## 1. Temporal modeling helped most for Stroke and MI

Our TGN-Survival model performed best on:
- Stroke
- Myocardial Infarction (MI)

These diseases depend a lot on how a patient’s condition changes over time:
- blood pressure trends
- medication changes
- worsening labs
- sequences of clinical events

Because the transformer reads patient history as a sequence of events instead of a single flattened row, it can better capture these patterns.

---

## 2. Cox regression performed worst overall

The traditional Cox model consistently had the lowest AUROC scores, especially for:
- Stroke
- PAD

This suggests that cardiometabolic disease progression is:
- highly nonlinear
- time-dependent
- influenced by many interacting clinical factors

which are difficult for simple linear survival models to capture.

---

## 3. XGBoost was still very strong for some diseases

For:
- Heart Failure (HF)
- Atrial Fibrillation (AF)
- PAD

XGBoost-Survival performed similarly to or slightly better than the transformer model.

These diseases may depend more on:
- overall disease burden
- extreme lab values
- long-term physiologic state

which can already be captured well using summary statistics like:
- average lab values
- maximum values
- latest measurements
- trends/slopes

without needing full temporal reasoning.

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

# Explainability Results

One useful part of the model is that we can inspect the attention weights to see which historical events the model focused on most when making predictions.

The model learned clinically meaningful patterns:

| Endpoint | Important Concepts |
|---|---|
| MI | coronary atherosclerosis, catheterization, beta blockers |
| Stroke | high diastolic BP, hyperlipidemia, amlodipine |
| HF | creatinine, BUN, hemoglobin |
| AF | atrial fibrillation history, mitral valve disease |
| PAD | vascular procedures, aspirin, clopidogrel |

This suggests the model is learning medically reasonable relationships instead of random correlations.

---

# Overall Conclusion

This project shows that representing EHR data as a temporal knowledge graph can improve prediction of cardiometabolic complications, especially for diseases where the ordering and timing of events matters.

The temporal graph transformer:
- clearly outperformed classical Cox survival models
- matched or exceeded strong machine learning baselines on several tasks
- performed especially well on Stroke and MI
- produced interpretable clinical explanations through attention analysis

Overall, the project demonstrates how temporal graph deep learning and multimodal clinical data can be combined for more accurate and interpretable healthcare prediction models.

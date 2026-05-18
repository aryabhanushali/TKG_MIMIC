
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


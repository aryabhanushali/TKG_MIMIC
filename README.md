# CardioMM-TKG

**Can a patient's medical history predict which circulatory disease they'll develop next — and does turning that history into a connected knowledge graph help, compared to standard approaches?**

This project builds a benchmark from MIMIC-IV hospital records and tests that question with four models: tuned Cox regression, tuned XGBoost, a sequence-based transformer that reads a patient's history as an ordered timeline, and a graph neural network that makes patients themselves nodes in a shared graph. This document describes the final, current state of the codebase and its results.

## The question

For a patient newly diagnosed with a cardiometabolic condition — diabetes, high blood pressure, high cholesterol, obesity, or metabolic syndrome — can we predict which of five circulatory diseases they'll go on to develop, and roughly when? And does representing their medical history as a **temporal knowledge graph** (a connected, time-stamped record of everything that happened to them) help a model predict better than just handing it a flat list of facts?

The five diseases: heart attack (MI), ischemic stroke, heart failure (HF), atrial fibrillation (AF), and peripheral artery disease (PAD). A patient can only have one *first* event, so the five diseases compete to happen first — someone who has a stroke is no longer "at risk" of having their first heart attack in the same sense. Patients who develop none of them are "censored": followed for years with no event.

## The data

**33,655 patients** from MIMIC-IV (a large, de-identified hospital records dataset from a Boston academic medical center). Each patient's "index date" is their earliest hospital admission with one of the cardiometabolic conditions above. From that point, we look forward to see which of the five diseases (if any) shows up next as the *main reason* for a later hospital admission — not just mentioned in passing — and we look backward up to 5 years to build their medical history.

That history becomes a graph: 8 types of hospital data (diagnoses, procedures, prescriptions, lab results, ICU stays, vital signs, outpatient measurements, IV fluids) turned into simple building blocks — *this patient, had this fact, at this time, with this value if it has one.* The result is **14.5 million facts** connecting 33,655 patients to about 33,000 distinct medical concepts. Every fact is generated only from data strictly before a patient's disease event, so the model structurally cannot see the future.

| Disease | Patients | Median age | % female |
|---|---|---|---|
| Heart attack (MI) | 918 | 66 | 44% |
| Stroke | 710 | 69 | 53% |
| Heart failure (HF) | 751 | 69 | 58% |
| Atrial fibrillation (AF) | 682 | 69 | 50% |
| Peripheral artery disease (PAD) | 302 | 66 | 43% |
| No event (censored) | 30,293 | 62 | 55% |

PAD is the smallest group by a wide margin — worth keeping in mind, since its results below are noisier than the others.

## The models

**Cox regression** and **XGBoost** are the standard approaches: each patient is a flat feature vector (which diagnoses they have, summary statistics of their lab values, basic demographics), with no sense of order or connection between events. Both are tuned: Cox's regularization strength is selected per disease from a validation-scored 100-point elastic-net regularization path (`src/ablations/cox_regularization_sweep.py`); XGBoost's tree depth, learning rate, subsampling, and estimator count are selected by a coordinate-wise validation search (`src/ablations/xgb_survival_sweep.py`). Both final settings are hardcoded into `src/baselines_survival.py`, which is the single script that produces both models' reported numbers.

The **TKG-Transformer** reads a patient's most recent 256 medical events as an ordered sequence pulled from the knowledge graph and uses a small attention-based model (the same family behind modern language models) to make a prediction. It's a sequence model reading graph-derived data, not a graph neural network — the graph is what *generates* the sequence, but the model itself does no graph message-passing.

The **patient-graph model** does use real graph structure: patients themselves are nodes, connected to each other only through the medical concepts they share (deduplicated per patient-concept pair, split into 3 recency buckets — ≤90 days, 90–730 days, >730 days before index — so some timing signal survives the collapse from a sequence into a graph). Concepts are additionally connected to each other by a hand-built medical hierarchy (a diagnosis code belongs to a category, a drug belongs to a class) and by a data-driven co-occurrence layer (computed from training patients only). Two rounds of message-passing let one patient's prediction be shaped by patterns learned from other, similar patients — the one thing a sequence model structurally cannot do. It trains in about 1–2 minutes (one pass over the whole graph per epoch) versus 20–50 minutes for the sequence-based models. A separate hyperparameter search (`src/ablations/patient_graph_gnn_joint_sweep.py`, a bounded random search over the joint hyperparameter space) found a config that improves slightly on heart attack and is essentially flat everywhere else — reported model uses the untuned default settings, since tuning did not meaningfully change the picture.

## How results are measured

**AUROC**, a standard score from 0.5 to 1.0 for "how well does this model rank patients who will get the disease above patients who won't," is the primary metric — 0.5 is a coin flip, 1.0 is perfect. Competing risks are handled explicitly: if a model is being checked on whether it predicted heart attack, and a patient had a stroke first instead, that patient counts as a genuine "no" for heart attack, not thrown out of the analysis. Patients lost to follow-up before a given time window ends are excluded from that specific check, since what would have happened to them is genuinely unknown (administrative censoring) — this means the 5-year numbers rest on a smaller group of patients than the 1-year numbers.

Test data is used exactly once, per trained model, at the very end. Every decision about which model checkpoint to keep, how to scale numeric features, or which medical codes a model is even allowed to have learned representations for is made using only training and validation data; a code that only appears in the test set is treated as "unknown." An automated check (`src/tests_integrity.py`) runs eight hard-gate tests on every pipeline run — split overlap, no post-index events, no fact dated at or after a patient's outcome, and others — and stops the pipeline outright if any fail. Each patient's train/validation/test assignment is computed from a deterministic hash of their own patient ID and the random seed, independent of who else is in the dataset.

One exception, worth naming precisely: the patient-graph model's patients are nodes in one shared graph, so a test patient's own pre-index facts (never their outcome label) do shape their position in that graph during training — the standard, accepted way this style of model is trained, and a real, if narrow, structural difference from the other three models. Message-passing also normalizes each concept's incoming information by how many patients (across *all* splits) connect to it, so validation/test patients' presence has a small effect on how strongly a training patient's own signal updates shared weights.

**Two significance tests are used, and they don't always agree.** The simpler one is a one-sample t-test: a model's 5 independently-trained seeds against a fixed comparator's single AUROC value, Bonferroni-corrected within each model's own 30-comparison family (15 cells vs. Cox, 15 vs. XGBoost). This is standard practice, but it has a real blind spot: it asks whether the comparator's number falls outside the model's own seed-to-seed spread, without ever checking whether the two models actually disagree on the *same patients*, and it ignores the comparator's own sampling uncertainty entirely. The stricter test is DeLong's test (paired, correlation-aware, computed once per seed on the shared test set) — it asks whether a specific trained model instance would still look different from the comparator if you swapped which test patients you happened to draw. A finding is only reported **robust** here if it's Bonferroni-significant under the t-test *and* DeLong-significant in at least 4 of 5 independently-trained seeds, all agreeing on direction. Everywhere both tests are reported below, prefer the stricter one.

## Results

**AUROC, all four models, seed 42 (5-seed means for TGN-Transformer and the patient-graph model differ by at most ~0.03 from these):**

| Disease | Horizon | Cox | XGBoost | TKG-Transformer | Patient-graph |
|---|---|---|---|---|---|
| MI | 1y | 0.752 | 0.743 | 0.684 | 0.692 |
| MI | 3y | 0.740 | 0.732 | 0.647 | 0.670 |
| MI | 5y | 0.713 | 0.706 | 0.625 | 0.670 |
| Stroke | 1y | 0.772 | 0.775 | 0.525 | 0.681 |
| Stroke | 3y | 0.763 | 0.758 | 0.601 | 0.722 |
| Stroke | 5y | 0.734 | 0.727 | 0.562 | 0.713 |
| HF | 1y | 0.717 | 0.750 | 0.662 | 0.647 |
| HF | 3y | 0.695 | 0.721 | 0.634 | 0.637 |
| HF | 5y | 0.701 | 0.711 | 0.644 | 0.673 |
| AF | 1y | 0.810 | 0.823 | 0.671 | 0.760 |
| AF | 3y | 0.696 | 0.686 | 0.640 | 0.663 |
| AF | 5y | 0.681 | 0.686 | 0.598 | 0.659 |
| PAD | 1y | 0.623 | 0.710 | 0.589 | 0.708 |
| PAD | 3y | 0.646 | 0.700 | 0.628 | 0.657 |
| PAD | 5y | 0.666 | 0.716 | 0.664 | 0.675 |

**Cox and XGBoost are statistically indistinguishable.** A paired DeLong test across all 15 cells finds no significant difference anywhere (p ranges 0.23–0.90).

**The patient-graph model reliably loses to XGBoost, most clearly on heart failure.** Across a 5-seed multi-seed comparison, it loses to XGBoost significantly in 11 of 15 disease/horizon cells; heart failure is significant at all three horizons, every time this has been checked, and is the single most consistently confirmed result in this project. A naive "find similar training patients and copy their outcome" baseline (`src/ablations/knn_baseline.py`), with no learning at all, beats the patient-graph model on heart failure too (0.657 vs. 0.624 at 3 years) — exactly the disease where the graph model does worst overall.

**Against Cox, only a handful of the patient-graph model's differences are robust to the stricter paired test:**

| Disease | Horizon | Direction | Robust? |
|---|---|---|---|
| MI | 1y, 3y | Cox wins | Robust |
| HF | 1y, 3y | Cox wins | Robust |
| Stroke | 1y | Cox wins | Robust |
| PAD | 1y, 3y, 5y | Patient-graph wins in all 5 seeds, every time | **Not robust** — 0 of 5 individual DeLong tests reach significance at any horizon |

The PAD finding is worth stating precisely: the patient-graph model beats Cox in every one of 5 independently-trained instances, a completely consistent direction. But the size of that win, on this test set, is too small for a paired test on any single trained model to rule out chance. The direction is real and repeatable; the one-sample t-test's "significant" call came from how tightly clustered the 5 seeds are, not from a patient-level signal that survives the stricter check.

**The patient-graph model's advantage over Cox concentrates in patients with thin pre-index medical histories — the one place a theory of how this model should work actually predicts it should.** Patients were split into "thin" (below the train-set median pre-index fact count) and "rich" (at or above) groups, and the patient-graph-minus-Cox AUROC margin was computed separately within each, replicated across all 5 seeds (not a single snapshot) — a cell only counts as confirmed if the thin-history margin is larger in at least 4 of 5 independently-trained instances, the same bar every other robustness claim in this project has to clear. Message-passing between patients is supposed to help most exactly when a patient's own record is sparse, by borrowing signal from similar patients elsewhere in the graph. That is what happens, mostly: **12 of 15 disease/horizon cells confirm the pattern (sign test p=0.035)**. The exceptions are informative rather than just noise: heart failure at 5 years goes the other way (confirmed in only 1/5 seeds), and PAD's own multi-seed replication is more mixed than a single seed suggested — only the 1-year cell confirms cleanly (5/5 seeds, and the largest effect anywhere: +0.134 thin vs. +0.045 rich), while PAD at 3 and 5 years do not confirm (0/5 and 2/5). Splitting an already-small test set in half pushes some cells down to as few as 8–19 positive cases, so individual cell margins are noisy — but the cross-cell direction, tested the same way 5 times over, is unlikely to be chance. This is the clearest evidence in this project that the patient-graph model is doing something mechanistically distinct from the other three, not just a noisier version of the same prediction (`src/ablations/cold_start_stratification.py`).

**The TKG-Transformer has almost no robust wins or losses against Cox** — the one exception is a robust loss on heart attack at 3 and 5 years. Every other cell fails to reach the stricter bar in either direction.

**Correcting across both models' claims together is stricter still.** The two multi-seed families (TKG-Transformer vs. baselines, patient-graph model vs. baselines) are usually Bonferroni-corrected separately, 30 comparisons each — but both are really answering the same question ("does graph-derived structure help here"), so a reviewer correcting across their 60-comparison union reaches a different bar. Under that combined correction, 25 of 60 comparisons stay significant (vs. 29 under the looser per-family bar); a Benjamini-Hochberg FDR alternative keeps 55 of 60.

**F1 and precision/recall.** AUROC is a ranking metric; with events this rare (0.4%–5.2% test-set prevalence), a threshold-based metric like recall needs care — optimizing for recall alone is degenerate, since flagging every patient always achieves recall = 1.0 (confirmed directly: every hyperparameter config tried, for both XGBoost and the patient-graph model, tied at exactly 1.0). F1 is the practical alternative. A separate hyperparameter search selecting on F1 instead of AUROC (`src/ablations/xgb_survival_sweep_recall_f1.py`, `src/ablations/patient_graph_gnn_sweep_recall_f1.py`) found real differences in validation F1 across configs, but a paired bootstrap test on the resulting best-F1 gap between the patient-graph model and each baseline found **no significant difference in any of the 30 cells checked** (15 vs. XGBoost, 15 vs. Cox) — the same one-sample t-test that overstates AUROC significance also calls 10–11 of those 15 cells "significant" each, which is the same gap between the two tests seen everywhere else in this project. AUPRC (the threshold-free analogue) tells the same story: mean AUPRC across all 15 cells is 0.068 (Cox), 0.072 (XGBoost), 0.047 (patient-graph model), with bootstrapped 95% CIs that overlap in every single cell.

**Explanation trustworthiness.** Tested formally, not by eyeballing attention weights: take the facts a model calls "important" for a prediction, and check whether they actually matter more than a random set of facts of the same size. Two directions — does removing the important facts hurt the prediction more than removing random ones (**comprehensiveness**), and does keeping only the important facts preserve the prediction as well as keeping a random set (**sufficiency**). Comprehensiveness passes cleanly for every model. Sufficiency is consistently weaker — borderline for the sequence models, and for the patient-graph model specifically, it never passes: the "important" facts do no better than random at preserving the prediction. This pattern has replicated identically across every version of this analysis, which makes it the most trustworthy qualitative finding in the project. Checkpoint selection requires at least 15 full training passes before a model is eligible to be called "final" — earlier stopping points produce a model whose stated attention is statistically indistinguishable from random, even when its raw prediction accuracy still looks reasonable.

**Calibration.** Does a predicted 30% risk actually happen about 30% of the time? Checked for the two models that output a real probability (the TKG-Transformer and the patient-graph model) via reliability diagrams; Cox and XGBoost output a relative risk score rather than a probability, so they're checked instead by whether a higher predicted-risk decile has a higher observed event rate, moving up monotonically. For every model and every disease, there's at least one step where it doesn't move up smoothly — most likely too few cases at this horizon to measure calibration precisely, rather than a sign of real miscalibration, but not fully ruled out either way.

**Fairness.** This dataset does not have enough non-White patients with each disease (as few as 3–11 per group) to say anything reliable about performance across race — that gap is reported rather than papered over with numbers too small to trust. An apparent fairness gap in stroke prediction across age groups looked real at first glance but didn't hold up under scrutiny: the group in question had only 11 stroke-positive patients, and an AUROC built on 11 positive cases carries too much sampling noise to support a subgroup claim, even though the group's *total* size (hundreds of patients) looked reassuring.

## Limitations, honestly

- **Single hospital system.** All data is from one Boston hospital; how well any of this generalizes elsewhere is untested.
- **No model wins outright.** It depends on the disease and the time horizon — there's no single model to point to as "the best one." Cox and XGBoost are indistinguishable from each other; the patient-graph model's one robust advantage over the sequence model is real, but its advantage over Cox is not statistically confirmed at the level this project otherwise requires.
- **A one-sample significance test against a single seed's result is not enough on its own.** Every place in this project where the stricter paired test was run alongside the standard one, the standard test called more things "significant." Prefer the paired test where both are available.
- **No reliable way to check fairness gaps** by age or race, given this cohort's demographics — every subgroup breakdown attempted came down to too few disease-positive patients to trust.
- **Two of the five diseases (HF and AF) are defined strictly on purpose** — only counted as a new case when they're the *main reason* for a later hospital visit — which is more leak-resistant but discards some statistically usable cases where the disease was only mentioned in passing.
- **None of the four models were exhaustively tuned.** The hyperparameter searches described above were real but bounded, not exhaustive searches.
- **The patient-graph model works differently under the hood** — a test patient's own pre-index facts (never their outcome) shape their position in the shared graph during training. This is a real structural difference from the other three models. Tested directly (`src/ablations/patient_graph_gnn_inductive.py`, an inductive variant with every validation/test patient's edges structurally removed during training): the difference from the standard model is negligible (max 0.00046 AUROC across 15 cells), so this structural quirk does not explain any of this project's findings.
- **The knowledge-graph medication hierarchy is thin.** Only about 9% of medications resolve to a drug class in the hand-built hierarchy, even though medications are the single largest category of data (32% of all facts).
- **The 5-year results rest on a smaller group of patients than the 1-year results**, since many patients haven't been followed that long yet — standard practice, but worth knowing when comparing horizons.
- **Only 5 random seeds** were used for the main robustness checks — a reasonable minimum, not a generous number, especially for PAD's small sample.
- **Two known data-labeling quirks in MIMIC-IV**, disclosed rather than worked around: the specific heart-failure marker "BNP" isn't recorded (MIMIC-IV records the related NT-proBNP test under a different label), and the heart-attack marker captured is specifically Troponin T, not Troponin I.

## How to reproduce this

```bash
PY=/path/to/miniforge3/envs/tkg/bin/python

# Core pipeline
$PY -u -m src.cohort
$PY -u -m src.build_tkg
$PY -u -m src.prep_modeling
$PY -u -m src.tests_integrity     # hard gate -- stops here if any check fails
$PY -u -m src.validate_tkg
$PY -u -m src.ablations.build_ontology
$PY -u -m src.ablations.build_cooccurrence
$PY -u -m src.baselines_survival   # tuned Cox + tuned XGBoost
$PY -u -m src.tgn_survival
$PY -u -m src.ablations.patient_graph_gnn
$PY -u -m src.compare_survival
$PY -u -m src.evaluate_stats
$PY -u -m src.make_figures

# 5-seed robustness checks
for s in 42 43 44 45 46; do
  TKG_SEED=$s $PY -u -m src.tgn_survival
  TKG_SEED=$s $PY -u -m src.baselines_survival
  TKG_SEED=$s $PY -u -m src.ablations.patient_graph_gnn
done
$PY -u -m src.multi_seed_summary
$PY -u -m src.ablations.patient_gnn_multi_seed_summary
$PY -u -m src.ablations.combined_significance_correction

# Explanation-trustworthiness checks
$PY -u -m src.explain_gnn
$PY -u -m src.ablations.explain_patient_graph_fidelity

# Hyperparameter tuning (produces the settings hardcoded into baselines_survival.py)
$PY -u -m src.ablations.cox_regularization_sweep
$PY -u -m src.ablations.xgb_survival_sweep
$PY -u -m src.ablations.patient_graph_gnn_joint_sweep

# Calibration / fairness / naive baseline / mechanism analysis
$PY -u -m src.ablations.calibration_and_fairness
$PY -u -m src.ablations.knn_baseline
$PY -u -m src.ablations.mechanism_analysis

# Robustness of the "beats Cox" claims: stricter significance testing
$PY -u -m src.ablations.tuned_models_significance
$PY -u -m src.ablations.delong_robustness_audit
$PY -u -m src.ablations.tuned_models_auprc_ci

# Class-1 recall / F1 tuning (a metric distinct from AUROC -- see "Results")
$PY -u -m src.ablations.xgb_survival_sweep_recall_f1
$PY -u -m src.ablations.patient_graph_gnn_sweep_recall_f1
$PY -u -m src.ablations.f1_significance

# Ablation: does the patient-graph model need to see test patients during training?
for s in 42 43 44 45 46; do
  TKG_SEED=$s $PY -u -m src.ablations.patient_graph_gnn_inductive
done

# Per-disease feature importance for all four models
$PY -u -m src.ablations.baseline_feature_importance   # Cox + XGBoost (the two graph models already have this)

# Does the patient-graph model's edge over Cox concentrate in thin-history patients?
$PY -u -m src.ablations.cold_start_stratification

# Publication-ready figures (paper_figures/, not gitignored)
$PY -u -m src.make_paper_figures
```

**Settings you can change** via environment variables: `TKG_SEED` (changes model randomness only, never the train/val/test split).

## Repository layout

```
TKG_MIMIC/
|-- main.py                       # runs the core pipeline in order
|-- README.md
|-- paper_figures/                 # publication-ready figures (not gitignored)
|-- mimic_data/                   # raw MIMIC-IV files (not included; gitignored)
`-- tkg_output/                   # everything the pipeline produces (gitignored)
    |-- cohort.csv                       # the 33,655-patient cohort
    |-- tkg_facts.csv                    # the 14.5M-fact knowledge graph
    |-- figures/                         # all generated figures
    |-- modeling/                        # train/val/test split, features
    |-- explain/                         # explainability outputs
    |-- stats/                           # confidence intervals, significance tests
    |-- sweeps/                          # hyperparameter search results
    |-- baselines_survival[_seed*]/      # tuned Cox + tuned XGBoost, per seed
    |-- tgn_survival[_seed*]/            # TKG-Transformer per seed
    |-- patient_gnn_survival[_seed*]/    # patient-graph model per seed
    |-- knn_baseline/                    # naive similar-patients baseline
    `-- survival_comparison_test.csv     # main head-to-head results table
```

```
src/
|-- config.py                 # settings, medical code lists, the washout safety check
|-- cohort.py                 # builds the patient cohort
|-- build_tkg.py              # builds the knowledge graph from raw MIMIC-IV files
|-- validate_tkg.py           # informational sanity checks (never halts the pipeline)
|-- tests_integrity.py        # hard-gate sanity checks (halts on any violation -- run this first)
|-- prep_modeling.py          # builds the train/val/test split and the per-event feature table
|-- baseline.py               # shared feature-building code for Cox/XGBoost
|-- baselines_survival.py     # fits tuned Cox regression and tuned XGBoost
|-- tgn_model.py              # the TKG-Transformer architecture (a sequence model, not a GNN)
|-- tgn_survival.py           # trains the TKG-Transformer with the competing-risks prediction head
|-- compare_survival.py       # builds the main results table
|-- evaluate_stats.py         # confidence intervals and significance tests
|-- multi_seed_summary.py     # 5-seed comparison and significance testing for the TKG-Transformer
|-- fidelity_stats.py         # shared significance testing for the explanation-fidelity checks
|-- visualize_tkg.py, visualize_stats.py, make_figures.py  # figures
|-- make_paper_figures.py     # publication-ready figures (paper_figures/, not gitignored)
|-- explain.py, explain_discriminative.py, explain_heatmap.py, explain_gnn.py  # explainability
`-- ablations/
    |-- build_ontology.py                    # hand-built medical hierarchy (feeds the patient-graph model)
    |-- build_cooccurrence.py                # data-driven concept co-occurrence edges (train-only)
    |-- patient_graph_gnn.py                 # the patient-graph model
    |-- patient_graph_gnn_joint_sweep.py     # patient-graph model hyperparameter tuning
    |-- patient_graph_gnn_inductive.py       # ablation: does it need to see test patients while training?
    |-- patient_gnn_multi_seed_summary.py    # 5-seed comparison for the patient-graph model
    |-- combined_significance_correction.py  # cross-family statistical correction
    |-- mechanism_analysis.py                # tests a theory of when the graph helps
    |-- knn_baseline.py                      # a simple "similar patients" baseline
    |-- training_stability_figure.py         # overlays training curves across models
    |-- explain_patient_graph_fidelity.py    # fidelity check for the patient-graph model
    |-- cox_regularization_sweep.py          # Cox hyperparameter tuning
    |-- xgb_survival_sweep.py                # XGBoost hyperparameter tuning
    |-- tuned_models_significance.py         # bootstrap CI + DeLong + seed-distribution t-test, Cox vs. others
    |-- delong_robustness_audit.py           # per-seed DeLong check on every "beats Cox" claim
    |-- tuned_models_auprc_ci.py             # AUPRC + bootstrap CI, all three main models
    |-- threshold_metrics.py                 # shared recall/F1-at-best-threshold helper
    |-- xgb_survival_sweep_recall_f1.py      # XGBoost hyperparameter tuning, selecting on F1
    |-- patient_graph_gnn_sweep_recall_f1.py # patient-graph model hyperparameter tuning, selecting on F1
    |-- f1_significance.py                   # is the F1 gap real? (paired bootstrap + t-test)
    |-- calibration_and_fairness.py          # calibration curves + subgroup fairness breakdown
    |-- baseline_feature_importance.py       # per-disease feature importance for Cox + XGBoost
    `-- cold_start_stratification.py         # does the patient-graph model's edge concentrate in thin-history patients?
```

## Environment

Python 3.10 (conda environment `tkg`): `pandas, numpy, scikit-learn, xgboost, torch (MPS/CUDA), torch_geometric, pyarrow, pycox, lifelines, scikit-survival, matplotlib, seaborn, tqdm`.

**Data:** MIMIC-IV v3.1. Requires credentialed PhysioNet access. This repository contains no patient data — only the code that builds everything from the raw files.

## Key settings

| Setting | Value |
|---|---|
| How far back the model looks before the index date | 5 years |
| Minimum follow-up required | 90 days |
| Lab / vital sign sampling | 30% / 10% |
| Longest event sequence the TKG-Transformer reads | 256 events |
| Sequence model size | 128-dimensional, 4 attention heads, 2 layers |
| Prediction horizons checked | 1, 3, and 5 years |
| Minimum training rounds before a model can be selected | 15 |
| Number of random seeds used for robustness checks | 5 (seeds 42-46) |
| Patient-graph model: time buckets on patient-concept edges | recent (≤90 days), mid (90-730 days), old (>730 days) |
| Patient-graph model: rounds of message-passing | 2 |
| Patient-graph model: training time | 65-114 seconds across 5 seeds (full-batch, CPU) |
| Cox: elastic-net l1 ratio | 0.9 (regularization strength tuned per disease) |
| XGBoost: tuned config | 100 trees, depth 3, learning rate 0.02, subsample 1.0, colsample 0.5 |

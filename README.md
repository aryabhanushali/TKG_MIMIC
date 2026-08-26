# CardioMM-TKG

**Can a patient's medical history predict which circulatory disease they'll develop next — and does turning that history into a connected knowledge graph help, compared to standard approaches?**

This project builds a benchmark from MIMIC-IV hospital records and tests that question with five different models, including two graph neural networks. It also documents, in detail, three real bugs found and fixed along the way because one of them changed the study's main conclusion.

Part 1: current, correct numbers and conclusions
Part 2: understand how this was built, what was tried, and how the bugs were found

---

# Part 1 — Currently

## The question

For a patient newly diagnosed with a cardiometabolic condition,  diabetes, high blood pressure, high cholesterol, obesity, or metabolic syndrome, can we predict which of five circulatory diseases they'll go on to develop, and roughly when? And does representing their medical history as a **temporal knowledge graph** (a connected, time-stamped record of everything that happened to them) help a model predict better than just handing it a flat list of facts?

The five diseases: heart attack (MI), ischemic stroke, heart failure (HF), atrial fibrillation (AF), and peripheral artery disease (PAD). A patient can only have one *first* event, so the five diseases compete to happen first — someone who has a stroke is no longer "at risk" of having their first heart attack in the same sense. Patients who develop none of them are "censored": followed for years with no event.

## The data

**33,655 patients** from MIMIC-IV (a large, de-identified hospital records dataset from a Boston academic medical center). Each patient's "index date" is their earliest hospital admission with one of the cardiometabolic conditions above. From that point, we look forward to see which of the five diseases (if any) shows up next as the *main reason* for a later hospital admission — not just mentioned in passing — and we look backward up to 5 years to build their medical history.

That history becomes a graph: 8 types of hospital data (diagnoses, procedures, prescriptions, lab results, ICU stays, vital signs, outpatient measurements, IV fluids) turned into simple building blocks — *this patient, had this fact, at this time, with this value if it has one.* The result is **14.5 million facts** connecting 33,655 patients to about 33,000 distinct medical concepts.

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

Five models are compared, split into two groups:

**Standard approaches.** **Cox regression**, the traditional statistical method for this kind of problem, and **XGBoost**, a strong, widely-used machine learning model. Both see each patient as a flat list of features — which diagnoses they have, summary statistics of their lab values, their basic demographics — with no sense of order or connection between events.

**Graph-derived approaches.** A **TKG-Transformer** reads a patient's most recent 256 medical events as an ordered sequence pulled from the knowledge graph and uses a small attention-based model (the same family behind modern language models) to make a prediction. It's worth being precise about what "knowledge graph" means here: the graph is what *generates* the sequence of facts, but the model itself doesn't do graph message-passing between connected concepts — it's a sequence model reading graph-derived data, not a graph neural network.

Two further models *do* use real graph structure. The first enriches each medical concept's own representation using a hand-built medical hierarchy (a diagnosis code belongs to a category, a drug belongs to a class) — this one didn't help, and is described in Part 2. The second — the one that matters — makes **patients themselves nodes in the graph**, connected to each other only through the medical concepts they share, with two rounds of message-passing letting one patient's prediction be shaped by patterns learned from other, similar patients. We call this the **patient-graph model** throughout.

None of the five models had their settings tuned beyond a follow-up experiment described below — every model uses fixed, reasonable, hand-picked settings.

## What we found

Model accuracy below is measured with **AUROC**, a standard score from 0.5 to 1.0 for "how well does this model rank patients who will get the disease above patients who won't" — 0.5 is a coin flip, 1.0 is perfect, and anything reliably above 0.5 is doing real work. All the numbers below are AUROC unless stated otherwise.

**XGBoost is the strongest model overall.** Across almost every disease and time horizon, plain XGBoost — no graph, no sequence modeling, nothing fancy — matches or beats every other model, including both graph-based attempts. This has been true throughout this project and is the single most consistent finding here.

**The patient-graph model has one real, replicated strength: peripheral artery disease.** It beats Cox regression on PAD at every time horizon we checked (1, 3, and 5 years), with strong statistical confidence, and that result has held up on repeat checks — real evidence it's not a fluke. It still loses to XGBoost on PAD, so the honest way to put it is: *better than the old statistical method, not better than the best current method.*

**It has one real, replicated weakness: heart failure.** It loses to XGBoost on heart failure at every time horizon, every time we've checked. This is the single most consistently confirmed result in the whole project.

**It also loses to Cox regression on stroke, but only at the 1-year mark.** This is a real, repeatable finding (confirmed across 5 separate training runs with different random seeds) but we don't yet know *why* — see Part 2 for what we've ruled out so far.

**A follow-up experiment: is the graph model's weak showing about how it was built, or about the task itself?** Since none of the models were originally tuned, we ran two hyperparameter searches — one for XGBoost, one for the patient-graph model — to see whether more careful tuning would close the gap. XGBoost improved measurably across every disease with modest tuning. The patient-graph model, run through the same kind of search, found a config that was more stable across random seeds but changed **nothing** on heart failure — 0.624 AUROC before tuning, 0.624 after, to three decimal places, while XGBoost tuned scores 0.721 on the same task. That's about as clean a signal as this kind of experiment can give: heart failure's result looks like a real limit of this particular graph design, not a symptom of bad settings. Full details, including why a naive combination of "best settings" actually made things *worse*, are in Part 2.

**Can we trust the models' explanations?** We tested this formally rather than just showing attention weights: take the facts a model calls "important" for a prediction, and check whether they actually matter more than a random set of facts of the same size. Two directions were tested — does removing the important facts hurt the prediction more than removing random ones (**comprehensiveness**), and does keeping only the important facts preserve the prediction better than keeping a random set (**sufficiency**). Comprehensiveness passed cleanly for every model, every time we checked — a real, trustworthy result. Sufficiency was consistently weaker: borderline for the sequence models, and for the patient-graph model specifically, it never once passed — the "important" facts did no better than random at preserving the prediction, across every version of this analysis. This pattern — unlike almost every accuracy number in this project — replicated identically three separate times, which makes it the single most trustworthy finding here.

**Calibration:** does a predicted 30% risk actually happen about 30% of the time? We checked this for the two models that output a real probability (the TKG-Transformer and the patient-graph model), building on an earlier, narrower check in this project (`fig12`, which only covered the TKG-Transformer at 5 years). Cox and XGBoost don't output a true probability — just a relative risk score — so we checked them a different way instead: do patients in a higher predicted-risk group actually have a higher observed rate of the disease, moving up smoothly as risk increases? For every model and every disease we checked, the answer wasn't a clean yes — there was always at least one step where it didn't move up smoothly. That's most likely because there are too few actual cases at this horizon to measure it precisely, rather than a sign the models are badly miscalibrated, but we can't fully rule the second explanation out. Treat this as an open question, not a settled negative result.

**Fairness:** we checked whether performance holds up equally across patient subgroups — something not looked at anywhere else in this project. Two things stand out. First, a real and currently unexplained finding: for patients aged 56–69 specifically, stroke predictions are *worse than a coin flip* for Cox regression and the plain TKG-Transformer — a genuine gap, not an artifact of a small subgroup (each age group here has hundreds of patients). Second, an honest limitation: this dataset does not have enough non-White patients with each disease (as few as 3–11 per group) to say anything reliable about performance across race. We report that gap rather than paper over it with numbers too small to trust.

## Limitations, honestly

- **Single hospital system.** All data is from one Boston hospital; how well any of this generalizes elsewhere is untested.
- **No model wins outright.** It genuinely depends on the disease and the time horizon — there's no single model to point to as "the best one."
- **A real, unexplained fairness gap** in stroke prediction for middle-aged patients (above), and no reliable way (yet) to check for fairness gaps by race, given this cohort's demographics.
- **The clinical-notes version of this project was never evaluated** — a data-normalization bug in that code path means its numbers wouldn't be trustworthy, and it wasn't required to answer the main research question.
- **Two of the five diseases (HF and AF) are defined strictly on purpose** — only counted as a new case when they're the *main reason* for a later hospital visit — which is more leak-resistant but throws away some statistically usable cases where the disease was only mentioned in passing.
- **None of the five models were exhaustively tuned.** The hyperparameter searches described above were real but bounded, not an exhaustive search — see Part 2 for exactly what was and wasn't tried.
- **The patient-graph model works differently under the hood.** Every other model only ever sees one patient's own data. The patient-graph model's patients are nodes in one shared graph, so a test patient's own pre-index facts (never their outcome) shape their position in that graph during training. This is the standard way this kind of model is trained, and their outcome labels are never used, but it's a real structural difference worth understanding — see Part 2 for the full explanation.
- **The knowledge-graph medication hierarchy is thin.** Only about 9% of medications resolve to a drug class in our hand-built hierarchy, even though medications are the single largest category of data (32% of all facts).
- **The 5-year results rest on a smaller group of patients than the 1-year results**, since many patients haven't been followed long enough yet to know what happens to them at 5 years — standard practice, but worth knowing when comparing horizons.
- **Only 5 random seeds** were used for the main robustness checks — a reasonable minimum, not a generous number, especially for PAD's small sample.

## How to reproduce this

```bash
PY=/path/to/miniforge3/envs/tkg/bin/python

# Core pipeline
$PY -u -m src.cohort
$PY -u -m src.build_tkg
$PY -u -m src.prep_modeling
$PY -u -m src.tests_integrity     # hard gate -- stops here if any check fails
$PY -u -m src.validate_tkg
$PY -u -m src.baselines_survival
TKG_USE_NOTES=0 $PY -u -m src.tgn_survival
$PY -u -m src.compare_survival
$PY -u -m src.evaluate_stats

# 5-seed robustness check
for s in 42 43 44 45 46; do
  TKG_SEED=$s TKG_USE_NOTES=0 $PY -u -m src.tgn_survival
  TKG_SEED=$s $PY -u -m src.baselines_survival
done
$PY -u -m src.multi_seed_summary

# The two graph-structure attempts
$PY -u -m src.ablations.build_ontology
$PY -u -m src.ablations.build_cooccurrence
HETERO_VARIANT=full   $PY -u -m src.ablations.hetero_gnn
HETERO_VARIANT=static $PY -u -m src.ablations.hetero_gnn
for s in 42 43 44 45 46; do
  TKG_SEED=$s $PY -u -m src.ablations.patient_graph_gnn
done
$PY -u -m src.ablations.patient_gnn_multi_seed_summary
$PY -u -m src.ablations.combined_significance_correction

# Explanation-trustworthiness checks
$PY -u -m src.explain_gnn
$PY -u -m src.ablations.explain_gnn_concept_graph
$PY -u -m src.ablations.explain_patient_graph_fidelity

# Hyperparameter sweeps and calibration/fairness
$PY -u -m src.ablations.xgb_survival_sweep
$PY -u -m src.ablations.patient_graph_gnn_sweep
$PY -u -m src.ablations.patient_graph_gnn_joint_sweep
$PY -u -m src.ablations.calibration_and_fairness
```

**Settings you can change** via environment variables: `TKG_USE_NOTES` (0/1, structured-only vs. also using clinical notes — the latter isn't evaluated in this work), `TKG_SEED` (changes model randomness only, never the train/val/test split), `HETERO_VARIANT` (`full`/`static`, for the first graph attempt), `PATIENT_GNN_TIMING` (0/1, whether the patient-graph model gets timing information on its edges).

## Repository layout

```
TKG_MIMIC/
|-- main.py                       # runs the core pipeline in order
|-- README.md
|-- mimic_data/                   # raw MIMIC-IV files (not included; gitignored)
`-- tkg_output/                   # everything the pipeline produces (gitignored)
    |-- cohort.csv                       # the 33,655-patient cohort
    |-- tkg_facts.csv                    # the 14.5M-fact knowledge graph
    |-- figures/                         # all generated figures
    |-- modeling/                        # train/val/test split, features
    |-- explain/                         # explainability outputs
    |-- stats/                           # confidence intervals, significance tests
    |-- sweeps/                          # hyperparameter search results
    |-- baselines_survival[_seed*]/      # Cox + XGBoost per seed
    |-- tgn_survival[_seed*]/            # TKG-Transformer per seed
    |-- hetero_gnn_survival/             # graph attempt 1
    |-- patient_gnn_survival[_seed*]/    # graph attempt 2 (the one that worked)
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
|-- baselines_survival.py     # fits Cox regression and XGBoost
|-- tgn_model.py              # the TKG-Transformer architecture (a sequence model, not a GNN)
|-- tgn_survival.py           # trains the TKG-Transformer with the competing-risks prediction head
|-- compare_survival.py       # builds the main results table
|-- evaluate_stats.py         # confidence intervals and significance tests
|-- multi_seed_summary.py     # 5-seed comparison and significance testing for the TKG-Transformer
|-- fidelity_stats.py         # shared significance testing for the explanation-fidelity checks
|-- notes_extract.py, notes_embed.py, notes_aggregate.py   # the (unevaluated) clinical-notes path
|-- visualize_tkg.py, visualize_stats.py, make_figures.py  # figures
|-- explain.py, explain_discriminative.py, explain_heatmap.py, explain_gnn.py  # explainability
`-- ablations/
    |-- build_ontology.py                    # hand-built medical hierarchy
    |-- build_cooccurrence.py                # data-driven concept co-occurrence edges (train-only)
    |-- hetero_gnn.py                        # graph attempt 1 -- didn't help
    |-- patient_graph_gnn.py                 # graph attempt 2 -- the one that worked
    |-- patient_gnn_multi_seed_summary.py    # 5-seed comparison for the patient-graph model
    |-- combined_significance_correction.py  # cross-family statistical correction
    |-- mechanism_analysis.py                # tests a theory of when the graph helps
    |-- oversmoothing_diagnostic.py          # diagnoses why graph attempt 1 failed
    |-- knn_baseline.py                      # a simple "similar patients" baseline
    |-- training_stability_figure.py         # overlays training curves across models
    |-- explain_gnn_concept_graph.py         # fidelity check for graph attempt 1
    |-- explain_patient_graph_fidelity.py    # fidelity check for graph attempt 2
    |-- xgb_survival_sweep.py                # XGBoost hyperparameter sweep
    |-- patient_graph_gnn_sweep.py           # patient-graph model, coordinate-wise sweep
    |-- patient_graph_gnn_joint_sweep.py     # patient-graph model, joint-random sweep
    `-- calibration_and_fairness.py          # calibration curves + subgroup fairness breakdown
```

## Environment

Python 3.10 (conda environment `tkg`): `pandas, numpy, scikit-learn, xgboost, torch (MPS/CUDA), torch_geometric, pyarrow, pycox, lifelines, scikit-survival, transformers (notes only), matplotlib, seaborn, tqdm`.

**Data:** MIMIC-IV v3.1, plus MIMIC-IV-Note v2.2 for the (unevaluated) notes version. Both require credentialed PhysioNet access. This repository contains no patient data — only the code that builds everything from the raw files.

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

---

# Part 2 — How we got here: process, investigations, and design decisions

Everything in Part 1 is the destination. This part is the trip — what was built, what was tried, what broke, and how we found out. It matters for anyone extending this work, and honestly, the story of finding and fixing our own bugs is as much a part of this project's contribution as any single result.

## Building the cohort

Every MIMIC-IV hospital admission gets filtered down in five steps: keep adults only; find each patient's earliest cardiometabolic-diagnosis admission (their "index date"); find the first *later* admission where one of the five circulatory diseases is the main reason for that visit (not just mentioned); remove anyone who already had any of the five diseases, in any form, at or before their index date (a "washout," so we're only predicting genuinely new disease); and require at least 2 hospital visits and 90 days of follow-up, or an actual event.

**Why "main reason for admission" and the washout matter.** An earlier version of this logic counted a disease as "new" if it was mentioned *anywhere* on a later admission, even as a side note — which is a leak: a chronic condition mentioned in passing (say, "known atrial fibrillation") could get counted as a brand-new case, while that same loose matching let a patient's *earlier* chart show the identical condition as an apparent early warning sign for a disease they already had. The model would effectively be told the answer. Tightening this to "main reason for admission" plus a full washout closes that leak, at a real cost: heart failure and atrial fibrillation lose the most patients this way (down to 16% and 14% of their "mentioned anywhere" counts), because those two conditions are very often noted as a side issue during an admission for something else entirely.

Charlson Comorbidity Index (a standard overall-health-burden score) is computed at the index date, from index-admission diagnoses only.

## Building the knowledge graph

Every patient's facts — diagnoses, procedures, prescriptions, lab results, ICU stays, vital signs, outpatient measurements, IV fluids — are pulled from the 5 years before their index date and turned into simple triples: *this patient, had this fact, at this time, optionally with this value.* Facts are never generated on or after a patient's disease event — the model structurally cannot see the future, because a fact outside its allowed time window is never written into the graph in the first place, not filtered out afterward.

Eight automated checks confirm the finished graph is correct (no future-dated facts, every patient has at least one fact, no duplicate coding, disease counts match the cohort file, and so on). One honest limitation surfaced by these checks: the specific heart-failure marker "BNP" isn't in the data at all — MIMIC-IV records the related NT-proBNP test under a different label — and the heart-attack marker captured is specifically Troponin T, not Troponin I.

## Two attempts at real graph structure

The natural first attempt: keep the TKG-Transformer's sequence design, but make each *medical concept's* own representation smarter by letting it absorb information from a hand-built hierarchy (a diagnosis code belongs to a category; a drug belongs to a class). **This didn't help** — it was worse than the plain sequence model on nearly every disease. Two reasons stand out: the hand-built hierarchy barely touches medications (only 9% of drugs resolve to a class, even though medications are the single largest category of data), and a follow-up diagnostic showed the enriched model's concept embeddings genuinely collapsed toward each other after message-passing — specifically for the poorly-connected, mostly-medication concepts the hierarchy doesn't cover. The well-connected concepts, by contrast, actually became *more* distinguishable from each other. The failure was localized to exactly the part of the graph with the least real structure to lean on, not a general indictment of graph methods.

The second, more genuine attempt: make **patients themselves nodes in the graph**, connected to each other only through the concepts they share (split into three simple time buckets so some timing signal survives the collapse from a sequence into a graph), with concepts additionally connected to each other both by the same hand-built hierarchy and by a new, purely data-driven layer — two concepts get an edge if they co-occur often enough across training patients (computed from training patients only, so no test information shapes the graph). Two rounds of message-passing then let information flow from one patient, through a shared concept, to a *different* patient who also has that concept, and back — the one thing a sequence model structurally cannot do. This is the model referred to as the "patient-graph model" throughout Part 1, and it trains remarkably fast (about 1-2 minutes total, versus 20-50 minutes for the sequence-based models) because it's one pass over the whole graph per training round rather than reprocessing each patient separately.

**Is this graph model just under-tuned, or does the task genuinely not reward this kind of structure?** We tested this directly (see "The hyperparameter search" below), and a few other pieces of evidence point the same way: a naive "find similar training patients and copy their outcome" baseline, with no learning at all, actually beats the patient-graph model on heart failure — exactly the disease where the graph model does worst overall — while the graph model clearly beats that same naive baseline on the diseases where it wins. And a specific theory — that diseases whose risk is spread thin across many weak signals (rather than concentrated in one or two strong ones) are where a graph model should have the most room to add value — pointed in the right direction all three times we tested it across three different versions of the pipeline, even though the exact strength of that pattern swung wildly each time (more on why in "Statistical methodology decisions" below). Heart failure has this project's single most concentrated risk signal and its worst graph-model result; that's not a coincidence we'd expect from noise alone.

## Three bugs we found in ourselves

This project's numbers were computed from scratch three separate times, each time because a rigorous check turned up something wrong with the version before. That's worth walking through in detail, because the third one changed which scientific claims this project can actually make.

### Bug 1: an ICU-stay feature that leaked the answer

One of the patient features fed to every model — "did this patient ever have an ICU stay" — was originally computed with no time restriction at all: it counted an ICU stay *any time* in a patient's record, including during or after the very hospitalization that defined their disease event. For an acute condition like a heart attack or stroke, which very often triggers an ICU admission, that's the outcome leaking into a predictor. We measured it directly: the properly time-restricted version of this feature is true for under 2% of patients in every group; the leaky version was true for 35–69% of patients depending on the disease. Fixed by restricting the feature to ICU stays that started strictly before the index date, and every number in the project was recomputed. Heart failure's results changed the most.

A related, smaller leak was caught at the same time: values for an ICU stay's length or an IV infusion's total amount were being suppressed based only on when the fact *started*, not when it *ended* — so a stay that started in-window but continued past the window close could carry a value only fully knowable after the fact. Fixed by nulling those values out when that happens.

### Bug 2: two wrong disease codes

An external code review caught two more issues, both construct-validity errors rather than timing leaks — the code was matching the *wrong* medical condition, not the right one from the wrong time window.

The "metabolic syndrome" cardiometabolic-index condition was coded as ICD-10 `E88.1`, which is *lipodystrophy* — a completely different condition — instead of `E88.81`, the actual metabolic-syndrome code. Because the matching is done by string prefix, the old code could never have matched the real metabolic-syndrome code even by accident; it matched 24 lipodystrophy patients and zero true metabolic-syndrome patients. We measured the actual impact directly against the raw hospital records rather than assuming it was large or small: fixing it changes cohort membership for only 1-6 patients out of 33,655 (nearly everyone coded with metabolic syndrome also qualifies for the cohort through diabetes, high blood pressure, or high cholesterol independently), though it does shift a feature value for 25 patients already in the cohort and shifts the index date itself for 2 of them.

Separately, the PAD disease definition's code list included two conditions — Raynaud's syndrome and Buerger's disease — that are real medical conditions but are not atherosclerotic peripheral artery disease, which is what this project means by "PAD." Checked against the actual 302 PAD patients: 2 of them (0.7%) qualified only through the Raynaud's code; none through Buerger's. Both codes were removed.

### Bug 3: the big one — a train/test split that wasn't actually fixed

Fixing bug 2 should have been a minor, contained change — it touched 2 of 33,655 patients. Instead, when we reran the full pipeline afterward, results changed dramatically for diseases that had nothing to do with the fix. That sent us looking for a second bug, and we found one that mattered more than the first two combined.

The code that splits patients into training, validation, and test sets was supposed to use a "fixed random seed" for reproducibility. It did — but not in the way that guarantee actually requires. It worked by shuffling each disease group's patient list off *one shared* random-number stream, processed group by group. The amount of randomness that shuffle step consumes depends on how many patients are in that group. So changing the patient count in *any* group — even a group with nothing to do with whatever triggered the change — shifted the shared stream's internal state for every group processed afterward, and reassigned which patients ended up in the test set for diseases that were never touched by the original fix. "Fixed random seed" only guarantees a fixed split if the input population is identical down to the last patient, which is a much narrower promise than the phrase suggests.

This wasn't a subtle, theoretical concern. After the 2-patient PAD code fix, Cox's own PAD accuracy score moved from 0.592 to 0.487 — despite no change whatsoever to Cox's features or training procedure. A key PAD comparison flipped from "mixed, losing in some places" to "wins everywhere, with very high statistical confidence." And a statistical pattern we'd found earlier (see "Statistical methodology decisions" below) went from looking solid to looking meaningless, because one of its five data points had flipped sign. None of that reflected anything true about PAD, the graph model, or anything else — it was what a shared-randomness bug looks like when it lands on a small, sensitive part of the data.

The fix: each patient's train/validation/test assignment is now computed from a deterministic hash of their own patient ID and the random seed, completely independent of who else happens to be in the dataset. We verified this directly — simulating the exact same 2-patient removal against a 5,000-patient test cohort changes 0 of the remaining patients' assignments under the new method, versus a demonstrated cascade under the old one.

**What changed once this was fixed and the pipeline was run a third and final time:** the project's two originally-reported headline findings — "the patient-graph model beats Cox regression on stroke at 3 years" and "beats it on heart attack at 5 years" — turned out to be artifacts of exactly which patients the fragile split happened to place in the test set. Neither holds up under the stable split. What replaced them, holding up across two independent pipeline runs under two different (correct) split algorithms, is the picture described in Part 1: the graph model reliably beats Cox on PAD, reliably loses to XGBoost on heart failure, and — new to the final run — reliably loses to Cox on stroke specifically at the 1-year mark. The explanation-fidelity finding (comprehensiveness solid, sufficiency weak) is the one result that held up identically across all three versions of this pipeline, which is exactly why Part 1 calls it the most trustworthy finding in the project.

The practical lesson: a "fixed random seed" is not, by itself, a guarantee that a pipeline's split is stable under the kind of small, routine change a real research process generates — a bug fix, a data update, an added exclusion rule. This project's original split passed every check it was given, right up until an unrelated 2-patient fix reshuffled its results wholesale.

## The explainability investigation

Partway through this project, we noticed the TKG-Transformer's stated "explanations" for its predictions — the facts its attention mechanism claimed were most relevant — looked wrong. It was focusing heavily on things like routine IV saline flushes, given to nearly every hospitalized patient regardless of diagnosis, instead of anything disease-specific. We confirmed this wasn't just an unusual-but-valid pattern with a formal test: take the facts the model calls "important," and check whether keeping (or removing) them actually changes the prediction more than a random set of facts would. It didn't — statistically indistinguishable from random.

The cause traced back to how the "final" model was chosen during training. The original code stopped training as soon as a validation metric stopped improving, and every one of five independent training runs stopped after just 1-2 passes through the data — before the model had learned anything real, even though its raw prediction accuracy still looked reasonable. Requiring at least 15 full passes before a model becomes eligible to be called "done" fixed this completely: attention on actual diagnosis codes for heart attack jumped from about 9% to 66%, and the model's top-attended facts shifted to type-2 diabetes, coronary artery disease, high cholesterol, high blood pressure — real, medically sensible risk factors. This is a real, useful finding on its own: picking a "final" model based only on validation accuracy can silently select a version whose stated reasoning is meaningless, even when its predictions still look fine.

The same fidelity test was later extended to both graph models, with more careful per-patient statistics added (a single mean number, it turned out, can be misleading — see below) — which is how we found that the "sufficiency" direction of the test is consistently weaker than the "comprehensiveness" direction, especially for the patient-graph model, as described in Part 1.

**A methodological note on that per-patient statistic.** The original fidelity checks reported only a single mean KL-divergence number across patients, comparing "important" facts to "random" facts. A mean can be dominated by a handful of outlier patients and hide whether the effect holds for a typical patient. We added a proper per-patient paired significance test (a paired t-test and a Wilcoxon signed-rank test, plus a win-rate with a confidence interval) to every fidelity script. This is what revealed that the patient-graph model's sufficiency check, while its mean *looked* favorable, actually had a win-rate below 50% at the per-patient level and no significant Wilcoxon result — the mean was being pulled up by a small number of patients with unusually large effects, not something that held broadly. This pattern replicated across all three pipeline runs.

## The hyperparameter search

None of the five models in this project were originally tuned — every one uses fixed, reasonable, hand-picked settings, purely to keep the comparison manageable. That leaves an obvious follow-up question: is the patient-graph model's weaker showing (against XGBoost, on most diseases) a matter of it being under-tuned, or does the task itself just not reward this kind of structure? We ran two searches to find out, both selecting only on validation data, never touching the test set until a final config was chosen — the same discipline used everywhere else in this project.

**XGBoost responded to tuning.** A modest, coordinate-wise search (varying tree count, depth, learning rate, and subsampling one at a time from sensible defaults) found a config — shallower trees, a lower learning rate, more subsampling — that improved validation accuracy from 0.708 to 0.751, and that improvement carried through to the test set on every single disease.

**The patient-graph model responded very differently, and in an instructive way.** The same coordinate-wise approach — varying hidden size, message-passing depth, relation-decomposition count, dropout, and learning rate one at a time — found that each hyperparameter, changed in isolation, looked like an improvement. But combining the single best value from each axis into one config produced a model that did *worse* on 4 of 5 diseases, and was noticeably less stable: across five repeated training runs with different random seeds, its validation score split into two distinct clusters, some runs landing around 0.70-0.72 and others collapsing to about 0.63 — a spread several times wider than this project's typical run-to-run variation. Stacking "more capacity everywhere" made the model harder to train reliably, not better. This is a textbook failure of combining hyperparameters one axis at a time: it ignores how they interact with each other.

**A genuine joint search told a cleaner story.** We followed up with a bounded random search over the *joint* space (rather than one axis at a time), and found a config that was both better on validation than any single-axis result, *and* stable across all five seeds — no collapse this time, confirming the earlier instability was a property of that specific greedy combination, not of "bigger models" in general. Retrained and evaluated on the test set, this better-tuned patient-graph model improved slightly on heart attack prediction, stayed essentially flat everywhere else — and changed *nothing at all* on heart failure, matching the untuned model's score to three decimal places. That's a clean signal: heart failure isn't a case of an unlucky choice of settings. Neither search was exhaustive — a full Bayesian optimization run could still find something better than either attempt here found — but the balance of evidence, combined with the untuned comparisons and the naive-similarity-baseline result described above, points toward a real, task-level limit for this specific graph design on most of these diseases, not simply an under-tuned model.

## Statistical methodology decisions

A few choices about how results are measured and reported are worth explaining rather than just stating.

**Handling competing diseases correctly.** If a model is being checked on whether it predicted heart attack, and a patient had a stroke first instead, that patient counts as a genuine "no" for heart attack — not thrown out of the analysis, which a simpler and overly generous scoring method would do. This is applied identically for every model. Patients simply lost to follow-up before a given time window ends are excluded from that specific check, since what would have happened to them is genuinely unknown (standard practice, called administrative censoring) — this does mean the 5-year numbers rest on a meaningfully smaller group of patients than the 1-year numbers, since many patients haven't been followed that long yet.

**Repeating training five times, and correcting for that.** To rule out a lucky or unlucky one-off training run, the TKG-Transformer, the patient-graph model, and XGBoost were each retrained five times with different random seeds (Cox regression has no meaningful randomness to vary). Comparing five separate results against a baseline, across five diseases and three time horizons, means many statistical tests are effectively running at once — and if you don't correct for that, some will look "significant" purely from the sheer number of tests, not from any real effect. We use a standard correction (Bonferroni) scoped to each model's own set of comparisons.

That correction turned out to matter concretely. Both the TKG-Transformer's and the patient-graph model's headline positive results were tested and reported as belonging to two *separate* 30-comparison families — but this project's own writing treats findings from both models as mutually reinforcing evidence that "graph structure helps here," which really makes them one 60-comparison family in substance. Applying the stricter, honest combined correction, two additional borderline findings lose significance that looked solid under the narrower scope. We report both scopes rather than picking the more flattering one, and built a small script (`combined_significance_correction.py`) specifically to make that comparison reproducible rather than something we'd have to redo by hand each time.

**Why the mechanism-analysis correlation should be read with real caution.** Part 2's "two attempts" section above describes a theory — spread-out risk signals favor the graph model more than concentrated ones — that lined up with the data in the right direction every time we tested it. But it's a correlation across only 5 diseases, and we computed it three separate times as the pipeline evolved (before the split-fragility bug was found, in the interim after fixing the ICD codes but before finding the split bug, and in the final corrected run), and got three genuinely different strength estimates each time — from a fairly strong correlation, to essentially no correlation at all, to what looked like a near-perfect one. The direction stayed consistent all three times; the exact strength did not, and shouldn't be trusted at this sample size regardless of which run's p-value you'd look at.

**Test data is used exactly once**, per trained model, at the very end. Every decision about which model checkpoint to keep, how to scale numeric features, or which medical codes the model is even allowed to have learned representations for, is made using only training and validation data. A code that only appears in the test set is treated as "unknown" by every model — it never gets a real, learned representation, since a model with one would have learned something from data it wasn't supposed to see yet. An automated check (`tests_integrity.py`) runs eight hard-gate tests on every pipeline run — split overlap, no post-index events, no fact dated at or after a patient's outcome, and others — and stops the pipeline outright if any of them fail, specifically so none of this can quietly regress.

One exception, disclosed rather than hidden: the patient-graph model trains differently from the other four. Because patients are nodes in one shared graph, a test patient's own pre-index facts (never their outcome label) do shape their position in that graph during training — this is the standard, accepted way this style of model is trained, and it's a real, if narrow, structural difference from the strict "the model never touches anything related to a test patient" rule the other four models follow. It's worth naming precisely: message-passing in this model normalizes each concept's incoming information by how many patients (across *all* splits, not just training) connect to it, so validation and test patients' presence in the graph has a small, real effect on how strongly a training patient's own signal updates shared model weights — separate from, and in addition to, the "shapes their own position" effect. It never touches labels, and it's standard behavior for this entire class of model, but it should be named rather than glossed over.

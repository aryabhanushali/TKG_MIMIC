# CardioMM-TKG

**A benchmark and modeling study on MIMIC-IV: predicting circulatory disease in cardiometabolic patients, and testing whether a temporal knowledge graph helps.**
## Summary

For a patient with a cardiometabolic condition (diabetes, high blood pressure, high cholesterol, obesity, or metabolic syndrome), can we predict which of five circulatory diseases they'll develop next, and when? Does representing their medical history as a **temporal knowledge graph** — a connected, time-ordered record of everything that happened to them — improve predictions over standard approaches?

The five diseases: heart attack (MI), stroke, heart failure (HF), atrial fibrillation / irregular heartbeat (AF), and peripheral artery disease (PAD, narrowed leg arteries). Patients who don't develop any of these are "censored" — followed for years with no event.

Cohort: 33,656 patients from MIMIC-IV, built with a specific eye toward avoiding data leakage (Section 2). Graph: 8 data types, 14.5 million facts. Three models compared: **Cox regression** (the classical statistical approach), **XGBoost** (a strong standard machine-learning model that treats each patient as a flat list of features), and a **temporal knowledge graph model (TGN)** that reads each patient's history as an ordered sequence of connected, time-stamped events.

**Main finding:** the knowledge graph model does not reliably beat XGBoost. This holds up after proper statistics — correcting for running many comparisons at once, and repeating training 5 times with different random seeds to rule out a lucky run. XGBoost was the strongest model overall. TGN's one solid, repeatable win is against classical Cox regression on **peripheral artery disease (PAD)**. On heart attack and atrial fibrillation, TGN was significantly worse than the simpler models.

**Second finding:** partway through, the knowledge graph model's "explanations" — the facts it claimed to rely on for each prediction — turned out to be meaningless. It was focusing on things like routine IV saline flushes, given to nearly every hospital patient regardless of diagnosis, instead of anything disease-relevant. This was confirmed with a formal test: take the facts the model calls "important," compare them against a random set of facts, and check whether "important" actually performs better. It didn't — statistically identical to random. The cause traced back to how the model was picked during training: it locked in a version of itself after only 1-2 training rounds, before it had learned anything real. Requiring more training before a model could be selected fixed this. Afterward its explanations lined up with real medical knowledge — tying heart attack risk to high blood pressure, high cholesterol, and diabetes — but prediction accuracy on some diseases dropped once the model was properly trained instead of stopped early by accident.


---

## Table of contents

1. [The research question](#1-the-research-question)
2. [Building the patient cohort](#2-building-the-patient-cohort)
3. [Building the knowledge graph](#3-building-the-knowledge-graph)
4. [Checking the graph is correct](#4-checking-the-graph-is-correct)
5. [Preparing the data for modeling](#5-preparing-the-data-for-modeling)
6. [The three models](#6-the-three-models)
7. [How we measured success](#7-how-we-measured-success)
8. [Results](#8-results)
9. [Can we trust the model's explanations?](#9-can-we-trust-the-models-explanations)
10. [Limitations](#10-limitations)
11. [How we avoided data leakage](#11-how-we-avoided-data-leakage)
12. [How to reproduce this](#12-how-to-reproduce-this)
13. [Repository layout](#13-repository-layout)
14. [Environment](#14-environment)
15. [Key settings](#15-key-settings)

---

## 1. The research question

> Among patients with a new cardiometabolic diagnosis, can their medical history up to that point predict which circulatory disease they'll develop next, and how soon? Does representing that history as a time-ordered, connected knowledge graph improve predictions over standard models that just look at a flat list of facts about the patient?

The five outcomes (a patient can only have one *first* event, so the diseases compete to happen first):

| Code | Disease |
|---|---|
| MI | Heart attack |
| Stroke | Ischemic stroke |
| HF | Heart failure |
| AF | Atrial fibrillation |
| PAD | Peripheral artery disease |

Two parts to this project: a benchmark dataset — raw MIMIC-IV records turned into a time-stamped knowledge graph across 8 data types — and a comparison of three models under a strict no-leakage setup, with the model's explanations actually tested for validity rather than just shown.

---

## 2. Building the patient cohort

Every MIMIC-IV hospital admission gets filtered down through a series of steps:

1. Keep only adult patients (age 18+).
2. Find each patient's earliest admission with a cardiometabolic diagnosis (diabetes, high blood pressure, high cholesterol, obesity, or metabolic syndrome). That admission date becomes the **index date** — the starting point for prediction.
3. Find the first admission *after* the index date where a circulatory disease is the **main reason for that admission**, not just something mentioned in passing. This is the disease being predicted, and when it happened.
4. Remove anyone who already had a chronic or prior form of *any* of the five diseases, however it was recorded, at or before the index date — a "washout" that keeps the prediction target to genuinely new disease.
5. Require at least 2 hospital visits and at least 90 days of follow-up, or an actual disease event.

The Charlson Comorbidity Index (CCI), a standard score of overall health burden, is computed at the index date.

### Why steps 3 and 4 matter, and what they cost

An earlier version of this cohort logic counted a disease as "new" if it was mentioned *anywhere* on a later admission, even as a minor side note, and used that same loose rule to check for prior disease. That's a leak: a chronic condition mentioned in passing (say, "known atrial fibrillation") could get counted as a brand-new case, and the identical loose matching would let a patient's earlier chart show that exact condition as an apparent "warning sign" for a disease they already had. The model would effectively be predicting something it was secretly already being told.

Requiring the disease to be the main reason for the later admission, and removing anyone with any earlier record of it, closes that leak — at a real cost in patients:

| Disease | Patients where it's mentioned anywhere | Patients where it's the main reason for admission (what's actually counted) | % kept |
|---|---|---|---|
| MI | 13,152 | 7,700 | 59% |
| Stroke | 8,308 | 5,786 | 70% |
| HF | 31,369 | 5,081 | 16% |
| AF | 35,167 | 4,809 | 14% |
| PAD | 10,717 | 2,354 | 22% |

Heart failure and atrial fibrillation lose the most patients here, because they're very often noted as a side issue during a hospital stay for something else — a patient admitted for pneumonia who also happens to have ongoing heart failure, for example. That's a real pattern in how hospitals code these conditions, not a flaw in the extraction logic. Separately, the washout step removes 52,901 patients total (some for more than one disease): 17,081 for MI, 15,255 for stroke, 19,003 for HF, 23,758 for AF, 6,679 for PAD.

### Final cohort: 33,656 patients

| Disease | Patients | % of cohort | Median age | % female | Median CCI | % with ICU stay | Median years until event |
|---|---|---|---|---|---|---|---|
| MI | 918 | 2.7% | 66 | 43.9% | 1 | 63.6% | ~3.6 |
| Stroke | 710 | 2.1% | 69 | 53.1% | 1 | 54.9% | ~3.3 |
| HF | 751 | 2.2% | 69 | 57.8% | 2 | 68.6% | ~2.5 |
| AF | 682 | 2.0% | 69 | 49.9% | 1 | 50.3% | ~3.3 |
| PAD | 302 | 0.9% | 66 | 43.0% | 1 | 51.7% | ~2.7 |
| Censored (no event) | 30,293 | 90.0% | 62 | 55.1% | 1 | 35.5% | ~3.1 |
| **Total** | **33,656** | 100% | 62 | 54.6% | 1 | 37.8% | ~3.1 |

Heart failure patients are the sickest group overall — highest comorbidity score, most ICU stays — which fits HF's usual role as a later complication of long-running cardiometabolic disease.

Files: `tkg_output/cohort.csv` and `cohort_cascade.csv` (the exact filtering counts above, used directly to draw the flow-diagram figure — nothing hardcoded).

---

## 3. Building the knowledge graph

For every patient, events get pulled from 8 types of hospital data and turned into a simple building block: *this patient, had this fact, at this time, optionally with this value.*

| Data type | Where it comes from | Example | Has a number attached? |
|---|---|---|---|
| Diagnoses | Diagnosis codes | "Essential hypertension" | No |
| Procedures | Procedure codes | "Coronary artery procedure" | No |
| Prescriptions | Medication orders | "Metformin" | No |
| Lab results | Lab tests | "Glucose", "HbA1c" | Yes, the result |
| ICU stays | ICU records | "ICU admission" | Yes, length of stay |
| Outpatient measurements | Office visit vitals | High/normal blood pressure, BMI category | Yes, the measurement |
| Vital signs | ICU monitoring | Heart rate, oxygen level | Yes, the reading |
| IV fluids / drainage | ICU fluid tracking | IV fluids given, urine output | Yes, the amount |

Each patient's data is limited to the 5 years before their index date, and stops before their disease event (or their last known follow-up, if they never had one) — the model never sees the future. Lab results and ICU vital signs are large enough that only a portion is sampled (30% and 10%) to keep processing manageable while still capturing the overall pattern.

**Result: 14,480,074 facts** across 32,965 distinct medical concepts and 33,656 patients (66,621 total nodes). Average 430 facts per patient (median 221).

| Data type | Number of facts | Share |
|---|---|---|
| Prescriptions | 4,635,003 | 32.0% |
| Lab results | 4,349,030 | 30.0% |
| Diagnoses | 1,564,719 | 10.8% |
| Vital signs | 1,031,478 | 7.1% |
| Blood pressure readings | 785,797 | 5.4% |
| Urine/drainage output | 681,129 | 4.7% |
| BMI readings | 585,934 | 4.0% |
| IV fluids | 482,635 | 3.3% |
| Procedures | 206,007 | 1.4% |
| Main diagnosis (per visit) | 140,916 | 1.0% |
| ICU stays | 17,426 | 0.1% |

The raw medication data file was checked against PhysioNet's official checksum and downloaded completely and correctly — an earlier concern about partial corruption doesn't apply to these numbers.

---

## 4. Checking the graph is correct

8 automated checks run on the finished graph. All 8 pass:

1. No fact is dated on or after a patient's disease event — no seeing the future.
2. Every patient in the cohort has at least one fact (11 patients, 0.03%, have very few — a trivial edge case).
3. No patient has the same medical code entered under both old and new coding systems at the same time.
4. Disease counts in the graph match the cohort file exactly.
5. No fact falls outside the intended 5-year window.
6-8. Key cardiometabolic drugs and labs (metformin, statins, HbA1c, creatinine, etc.) are present with realistic counts.

One honest limitation: the specific lab test "BNP," a heart failure marker, doesn't appear in the data at all — MIMIC-IV records a related test (NT-proBNP, which is present, 5,822 facts) under a different label than the matching rules look for. The heart attack marker captured is specifically Troponin T (9,010 facts), not Troponin I (43 facts, negligible). Worth stating plainly: heart failure prediction is missing a natriuretic-peptide lab feature, and the heart attack marker is troponin T specifically.

---

## 5. Preparing the data for modeling

- Only facts from the 5 years before each patient's index date are used: 2,283,125 events out of the 14.5 million total.
- Patients are split 70% training / 15% validation / 15% test, keeping the same disease mix in each group, with a fixed random seed so the split is reproducible.
- Basic patient info used as features: age, sex, comorbidity score, number of cardiometabolic conditions, ICU-stay flag.
- Every numeric value (labs, vitals) is standardized using only the training patients' statistics, so nothing from validation or test leaks into how the data is scaled.

| Disease | Training | Validation | Test |
|---|---|---|---|
| MI | 643 | 138 | 137 |
| Stroke | 497 | 106 | 107 |
| HF | 526 | 113 | 112 |
| AF | 477 | 102 | 103 |
| PAD | 211 | 45 | 46 |
| Censored | 21,205 | 4,544 | 4,544 |
| **Total** | **23,559** | **5,048** | **5,049** |

The number of actual disease cases in the test set is small, especially for PAD (46 patients) — disclosed rather than hidden, and the reason confidence intervals and a 5-seed repeat (Section 8) matter more here than they would with a bigger cohort.

---

## 6. The three models

**Cox regression** — the traditional statistical method for this kind of problem. One model per disease.

**XGBoost** — a strong, widely-used machine learning model. Sees each patient as a flat list of features: which medical codes they have, summary statistics of their lab values (average, highest, lowest, most recent, trend), and their basic info, with no sense of order or connection between events.

**The temporal knowledge graph model (TGN)** — reads a patient's most recent 256 events in order, as a connected sequence. Each event combines what it is, what kind of relationship it represents, when it happened relative to the index date, and its value if it has one. A small Transformer (the same family of model behind modern language models) processes the sequence and boils it down into one summary vector per patient, using a learned attention mechanism that highlights the most relevant events. The output is a probability of each disease happening within each time window.

**A note on how the "final" version of this model was picked.** The first attempt used automatic stopping once a validation check stopped improving — and every one of 5 independent training runs stopped after just 1-2 passes through the data. Section 9 shows why that's a problem: the model saved at that point had explanations that were statistically meaningless. The fix was to require at least 15 full passes before a model becomes eligible to be called "done." With that change, all 5 runs trained for 21-23 passes before stopping. Every result in this document uses the properly-trained version. The comparison between the two versions is in Section 9 — it's a real finding, not a footnote.

---

## 7. How we measured success

- **The metric: AUROC.** A score from 0.5 (a coin flip) to 1.0 (perfect) measuring how well the model ranks patients who will get a disease above those who won't.
- **Handling "competing" diseases correctly.** If a model is being checked on whether it predicted heart attack, and a patient got a stroke first instead, that patient counts as a "no" for heart attack — not thrown out of the analysis, which is what a simpler, overly generous scoring method would do. Fixed everywhere in this project. Patients simply lost to follow-up before the time window ended are still excluded, since what would have happened to them is genuinely unknown.
- **Confidence intervals and significance testing.** The main results include a 95% confidence interval (resampling the test patients 2,000 times) and a standard statistical test (DeLong's) for whether one model is really better than another, or if the gap is just noise.
- **Repeating training 5 times.** To rule out a lucky or unlucky one-off run, TGN and XGBoost were each retrained five times with different random seeds (Cox has no meaningful randomness to vary, so it's reported once). The five results per model are compared with Welch's t-test, and because 30 comparisons are running at once (5 diseases × 3 time windows × 2 model comparisons), a strict correction (Bonferroni) is applied so nothing gets called "significant" just from sheer number of tests.
- **The test set is used exactly once** per trained model, at the very end. Every decision about normalization, which codes to use, or which model checkpoint to keep is made using only training and validation data.

---

## 8. Results

### 8.1 Test accuracy (AUROC) by disease, time window, and model

| Disease | Time window | Cox | XGBoost | TGN (knowledge graph) |
|---|---|---|---|---|
| MI | 1yr / 3yr / 5yr | 0.730 / 0.742 / 0.690 | 0.736 / 0.722 / 0.718 | 0.682 / 0.661 / 0.662 |
| Stroke | 1yr / 3yr / 5yr | 0.632 / 0.636 / 0.656 | 0.619 / 0.645 / 0.657 | 0.662 / 0.624 / 0.621 |
| HF | 1yr / 3yr / 5yr | 0.792 / 0.736 / 0.740 | 0.771 / 0.750 / 0.735 | 0.733 / 0.730 / 0.741 |
| AF | 1yr / 3yr / 5yr | 0.724 / 0.685 / 0.670 | 0.793 / 0.729 / 0.721 | 0.619 / 0.646 / 0.655 |
| PAD | 1yr / 3yr / 5yr | 0.634 / 0.592 / 0.568 | 0.692 / 0.670 / 0.636 | 0.685 / 0.673 / 0.685 |

### 8.2 3-year results with 95% confidence intervals

| Disease | Cox | XGBoost | TGN |
|---|---|---|---|
| MI | 0.742 [0.686 - 0.795] | 0.722 [0.661 - 0.780] | 0.661 [0.591 - 0.727] |
| Stroke | 0.636 [0.562 - 0.707] | 0.645 [0.565 - 0.717] | 0.624 [0.548 - 0.699] |
| HF | 0.736 [0.677 - 0.790] | 0.750 [0.687 - 0.810] | 0.730 [0.666 - 0.788] |
| AF | 0.685 [0.611 - 0.758] | 0.729 [0.662 - 0.790] | 0.646 [0.575 - 0.713] |
| PAD | 0.592 [0.489 - 0.685] | 0.670 [0.525 - 0.809] | 0.673 [0.554 - 0.784] |

### 8.3 Is TGN significantly different from XGBoost? (one model each, 3-year window)

| Disease | Difference (TGN minus XGBoost) | p-value | Result |
|---|---|---|---|
| MI | -0.061 | 0.025 | TGN significantly worse |
| AF | -0.083 | 0.019 | TGN significantly worse |
| Stroke | -0.021 | 0.658 | No real difference |
| HF | -0.021 | 0.603 | No real difference |
| PAD | +0.003 | 0.960 | No real difference |

### 8.4 The 5-seed check — the most trustworthy result

TGN and XGBoost were each trained five times with different random seeds, then compared with the strict correction described above. This is the most rigorous comparison in the study:

| Disease | Time window | Cox | XGBoost (average ± spread) | TGN (average ± spread) | Holds up after strict correction? |
|---|---|---|---|---|---|
| MI | 3yr | 0.742 | 0.721 ± 0.010 | 0.683 ± 0.015 | Yes — TGN worse than Cox |
| MI | 5yr | 0.690 | 0.717 ± 0.010 | 0.672 ± 0.006 | Yes — TGN worse than XGBoost |
| AF | 5yr | 0.670 | 0.677 ± 0.028 | 0.651 ± 0.004 | Yes — TGN worse than Cox |
| PAD | 3yr | 0.592 | 0.682 ± 0.019 | 0.661 ± 0.020 | Yes — TGN better than Cox |
| PAD | 5yr | 0.568 | 0.671 ± 0.021 | 0.675 ± 0.020 | Yes — TGN better than Cox |
| The other 10 of 15 comparisons | — | — | — | — | Not proven either way with only 5 seeds |

No version of "the knowledge graph model wins" survives this level of scrutiny. Where there's a real, provable difference, TGN is usually the worse model — heart attack, atrial fibrillation. The one genuine, repeatable win for TGN is against Cox on peripheral artery disease; it doesn't clearly beat XGBoost there, but it's no longer clearly behind it either. XGBoost is the strongest model overall on this data, and that's reported as the main finding, not softened.

### 8.5 The version with clinical notes

Not evaluated for these results. There's a specific problem in that part of the code: only about 9% of patients have a usable discharge note, and the way the note data gets standardized is thrown off by the 91% of patients who don't have one — the normalization statistics end up skewed toward "no note" rather than what a real note looks like. Flagged here for anyone picking this up, but it wasn't required to answer the main question, and with 9% coverage its effect either way would likely be small.

---

## 9. Can we trust the model's explanations?

### 9.1 Two different questions

"Where is the model looking?" (attention weights) is a different question from "does that actually matter to the prediction?" The second question was checked with a tool called GNNExplainer plus a fidelity test: take the top 20% of events the model calls "important," and check whether keeping only those events preserves the prediction better than keeping a random 20%. Separately, check whether removing the top 20% breaks the prediction more than removing a random 20%. A real explanation should pass both checks clearly. If "important" performs the same as "random," the explanation carries no information.

### 9.2 What the too-early model looked like

The first version of the model — the one that stopped training after just 1-2 passes — put most of its attention on things like a routine IV saline flush, given to nearly every hospitalized patient regardless of diagnosis, rather than on actual diagnoses. Only 6-11% of attention went to diagnosis codes; 34-48% went to medications like the saline flush. The fidelity test confirmed this wasn't just an odd-but-valid pattern: the "important" events performed statistically the same as randomly picked ones, on both checks. The explanations carried no real information, even though the raw predictions still looked reasonable.

### 9.3 What changed after requiring more training

| | Before (stopped after 1 round) | After (properly trained, 15+ rounds) |
|---|---|---|
| Attention on diagnosis codes (heart attack) | ~9% | 62% |
| Top facts for heart attack | Saline flush, routine lab values | High blood pressure, high cholesterol, coronary artery disease, diabetes |
| Does "important" beat "random" at preserving the prediction? | No — identical | Yes, clearly and measurably |
| Does removing "important" hurt more than removing "random"? | No — identical | Yes, over twice the damage |

Where the model's attention goes, by data type, after the fix:

| Disease | Diagnoses | Labs | Procedures | Medications | Blood pressure | BMI | Vitals |
|---|---|---|---|---|---|---|---|
| MI | 62.1% | 12.3% | 8.9% | 5.6% | 6.2% | 2.3% | 0.8% |
| Stroke | 61.3% | 15.8% | 6.2% | 5.4% | 6.1% | 3.8% | 0.9% |
| HF | 64.5% | 14.9% | 7.2% | 6.2% | 5.0% | 1.4% | 0.6% |
| AF | 59.4% | 15.4% | 9.3% | 4.7% | 6.5% | 3.7% | 0.5% |
| PAD | 64.6% | 8.6% | 7.6% | 8.9% | 7.1% | 2.5% | 0.4% |

Diagnosis codes are now the dominant, trustworthy signal for every disease.

### 9.4 What's distinctive about each disease, according to the model

Beyond raw attention, a check for which facts are disproportionately linked to one specific disease, rather than being generically common across all five:

| Disease | Cases in test set | Distinctive facts the model relies on |
|---|---|---|
| MI | 137 | Type-2 diabetes, chest pain history, HbA1c, coronary artery disease, high cholesterol |
| Stroke | 107 | Sodium level, normal blood pressure readings, kidney function, obesity, hypertension |
| HF | 112 | Kidney function (the heart-kidney link), diabetes, prior coronary artery disease (the classic path from heart attack to heart failure), obesity |
| AF | 103 | Overweight/obesity, bicarbonate level, LDL cholesterol, specific cardiac procedures |
| PAD | 46 | High blood pressure, hypertension diagnosis — only 3 facts had enough patients behind them to count as reliable, reflecting the small PAD sample |

Coronary artery disease flagging heart attack risk, and showing up again as a warning sign for heart failure, matches the well-known medical progression from heart attack to heart failure. Kidney function flagging heart failure risk matches the documented heart-kidney connection. This is the strongest evidence that the model is reasoning sensibly rather than picking up noise.

One caveat: GNNExplainer's own list of top facts per disease (figure 17) isn't filtered for how many patients each fact applies to, so some top entries are one-off quirks from a single patient. This doesn't affect the fidelity numbers above, which are the trustworthy part of this check — just the detailed per-fact list in that one figure.

### 9.5 The bottom line

Picking a "final" model based only on validation accuracy can select a version whose stated reasons for its predictions are meaningless, even when the predictions themselves look fine. Requiring a minimum amount of real training fixed this completely — the explanations became genuine and lined up with real medical knowledge — but cost some raw accuracy on certain diseases. Accuracy and trustworthy explanations weren't the same thing here, and optimizing only for accuracy would have quietly shipped a model that explained itself with noise.

---

## 10. Limitations

- **Single hospital system.** All data comes from one Boston hospital; how well this generalizes elsewhere is untested.
- **The clinical-notes version wasn't evaluated**, and has a known, unfixed issue (Section 8.5) for anyone extending this work.
- **No model here clearly wins overall** once tested rigorously, except TGN beating Cox on PAD specifically. XGBoost is the strongest model on this data.
- **Small numbers of actual disease cases in testing**, especially PAD (46 patients), hence the wide confidence intervals.
- **HF and AF are defined strictly on purpose** (main-reason-for-admission only, broad washout), trading away some statistical power for cleaner, leak-free labels — the exact cost is in Section 2.
- **Patients lost to follow-up are simply excluded** from a given time window's scoring rather than statistically reweighted — a standard, disclosed simplification.
- **Labs and vital signs are sampled** (30% / 10%) rather than fully complete, to keep processing manageable.
- **Only 5 random seeds** were used for the robustness check — a reasonable minimum, not a generous number. More would tighten the confidence estimates further, especially for PAD.

---

## 11. How we avoided data leakage

The test set is used exactly once per trained model, at the very end.

- The train/validation/test split is fixed once, with a fixed random seed, and never changes.
- Any medical code that only appears in the test set is treated as "unknown" — never given a real, learned representation, since that would mean the model learned something from data it shouldn't have seen yet.
- All numeric scaling (labs, vitals) is based on training patients only.
- The time windows used by the model's prediction head are also set using training patients only.
- No fact used for prediction is ever dated on or after the outcome it's predicting — enforced directly in the code and double-checked separately afterward.
- The washout step (Section 2) closes a specific leak where a chronic condition, mentioned casually, could act as both a warning sign before the fact and the outcome itself. A safety check runs automatically every time the code starts, confirming every disease code counted as an outcome is also covered by the washout rule — so this exact bug can't quietly come back.
- Which model checkpoint to keep is decided using validation data only; test results aren't looked at until the very end.
- The rarer diseases are up-weighted during training using training-set statistics only.

---

## 12. How to reproduce this

```bash
PY=/path/to/miniforge3/envs/tkg/bin/python
$PY -u -m src.cohort
$PY -u -m src.build_tkg
$PY -u -m src.validate_tkg
$PY -u -m src.prep_modeling
$PY -u -m src.baselines_survival
TKG_USE_NOTES=0 $PY -u -m src.tgn_survival
$PY -u -m src.compare_survival
$PY -u -m src.evaluate_stats
```

Main results table: `tkg_output/survival_comparison_test.csv`. Confidence intervals and significance tests: `tkg_output/stats/test_metrics_with_ci.csv` and `delong_pairwise.csv`.

To reproduce the 5-seed check (`tkg_output/stats/multi_seed_comparison.csv`):

```bash
for s in 42 43 44 45 46; do
  TKG_SEED=$s TKG_USE_NOTES=0 $PY -u -m src.tgn_survival
  TKG_SEED=$s $PY -u -m src.baselines_survival
done
$PY -u -m src.multi_seed_summary
```

**Settings you can change:**
- `TKG_USE_NOTES` — `0` (structured data only; used for every result here) or `1` (also uses clinical notes; not evaluated in this work, see Section 8.5).
- `TKG_SEED` — changes only the model's own randomness (weight init, dropout, batch order), not the train/validation/test split, which stays fixed no matter what seed is used. Seed 42 is the default everywhere; seeds 43-46 write to their own separate folders so they never overwrite it.

**Other scripts, once the steps above are done:**

| Script | Needs | Produces |
|---|---|---|
| `src.visualize_tkg` | build_tkg | Cohort and graph figures 1-4 |
| `src.visualize_stats` | prep_modeling | `stats/table1_summary.csv`, figures 13-14 |
| `src.make_figures` | baselines + TGN | Figures 5, 6, 7, 12 |
| `src.explain` | TGN | Attention-based importance, figures 9-10 |
| `src.explain_discriminative` | `src.explain` | Disease-specific fact analysis, figure 11 |
| `src.explain_heatmap` | `src.explain` | Attention heatmaps, figures 15-16 |
| `src.explain_gnn` | TGN | Fidelity check, figure 17 |
| `src.multi_seed_summary` | TGN + baselines, seeds 42-46 | 5-seed comparison table and figure 20 |

---

## 13. Repository layout

```
TKG_MIMIC/
|-- main.py                       # runs everything in order
|-- README.md                     # this file
|-- mimic_data/                   # raw MIMIC-IV files (not included; gitignored)
|-- tkg_output/                   # everything the pipeline produces (gitignored)
|   |-- cohort.csv
|   |-- cohort_cascade.csv
|   |-- tkg_facts.csv
|   |-- node_index.csv
|   |-- validation_report.txt
|   |-- figures/                  # fig1 through fig20
|   |-- modeling/
|   |-- baselines_survival/       # seed 42 (main results)
|   |-- baselines_survival_seed{43..46}/
|   |-- tgn_survival/             # seed 42 (main results)
|   |-- tgn_survival_seed{43..46}/
|   |-- stats/                    # confidence intervals, significance tests, 5-seed comparison
|   |-- explain/
|   `-- survival_comparison_test.csv
`-- src/
    |-- config.py                 # settings, medical code lists, shared helper functions
    |-- cohort.py                 # builds the patient cohort
    |-- build_tkg.py              # builds the knowledge graph
    |-- validate_tkg.py           # runs the 8 sanity checks
    |-- visualize_tkg.py          # cohort and graph figures
    |-- visualize_stats.py        # descriptive statistics (figures 13/14)
    |-- notes_{extract,embed,aggregate}.py  # optional clinical-notes add-on
    |-- prep_modeling.py          # builds the modeling dataset and train/val/test split
    |-- baseline.py               # shared feature-building code
    |-- baselines_survival.py     # Cox + XGBoost
    |-- tgn_model.py              # the knowledge graph model itself
    |-- tgn_survival.py           # trains the knowledge graph model
    |-- compare_survival.py       # builds the final results table
    |-- multi_seed_summary.py     # the 5-seed comparison and significance testing
    |-- make_figures.py           # figures 5/6/7/12
    |-- explain.py                # attention-based explanations
    |-- explain_discriminative.py # disease-specific fact analysis
    |-- explain_heatmap.py        # attention heatmaps
    |-- explain_gnn.py            # the fidelity check
    `-- ablations/                # experimental extensions, not part of the main results
```

---

## 14. Environment

Python 3.10 (conda environment `tkg`):

```
pandas, numpy, scikit-learn, xgboost
torch (Apple MPS or CUDA), torch_geometric
pyarrow
pycox, lifelines, scikit-survival
transformers   # only needed for the optional clinical-notes add-on
matplotlib, seaborn, tqdm
```

**Data:** MIMIC-IV v3.1, plus MIMIC-IV-Note v2.2 for the optional clinical-notes version. Both require separate, credentialed access through PhysioNet. This repository contains no patient data — only the code that builds everything from the raw files.

---

## 15. Key settings

| Setting | Value |
|---|---|
| How far back the model looks before the index date | 5 years |
| Minimum follow-up required | 90 days |
| Lab / vital sign sampling | 30% / 10% |
| Longest event sequence the model reads | 256 events |
| Model size | 128-dimensional, 4 attention heads, 2 layers |
| How the model represents time | Bochner/TGAT time encoding |
| Prediction horizons checked | 1, 3, and 5 years |
| Minimum training rounds before a model can be selected | 15 |
| How long training can run before stopping automatically | up to 30 rounds, stops after 6 rounds without improvement |
| Number of random seeds used for the robustness check | 5 (seeds 42-46) |
| Default seed | 42 |

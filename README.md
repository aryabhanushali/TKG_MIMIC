# CardioMM-TKG

**A benchmark and modeling study on MIMIC-IV: predicting circulatory disease in cardiometabolic patients, and testing whether a temporal knowledge graph helps.**



## Summary

**The question:** for a patient who has a cardiometabolic condition (diabetes, high blood pressure, high cholesterol, obesity, or metabolic syndrome), can we predict which of five circulatory diseases they'll develop next, and when? And does representing their medical history as a **temporal knowledge graph** (a connected, time-ordered record of everything that happened to them) improve predictions compared to standard approaches?

The five diseases: heart attack (MI), stroke, heart failure (HF), atrial fibrillation / irregular heartbeat (AF), and peripheral artery disease (PAD, narrowed leg arteries). Anyone who doesn't develop one of these is "censored" — followed for years with no event.

Built a cohort of 33,656 patients from MIMIC-IV, carefully checked to avoid data leakage (see [Section 2](#2-building-the-patient-cohort)), built an 8-type knowledge graph from their records (14.5 million facts), and compared three models: **Cox regression** (the classical statistical approach), **XGBoost** (a strong standard machine-learning model that treats each patient as a flat list of features), and our **temporal knowledge graph model (TGN)**, which reads each patient's history as an ordered sequence of connected, time-stamped events.

**Main finding:** after testing carefully — with statistics, correcting for running many comparisons at once, and repeating training 5 times with different random seeds to make sure results were accurate — the knowledge graph model did **not** reliably beat XGBoost. XGBoost was the strongest model overall. The knowledge graph model showed one solid, repeatable win: it reliably beat classical Cox regression on **peripheral artery disease (PAD)**. On heart attack and atrial fibrillation, it was actually significantly worse than the simpler models. This is a real, useful finding — showing that a more complex model doesn't automatically beat a well-tuned simple one is valuable and common in medical AI, not a failure.

**Second finding:** partway through, we discovered that our knowledge graph model's "explanations" — the facts it claimed to rely on for each prediction — were meaningless. It was focusing on things like routine IV saline flushes (given to almost every hospital patient, regardless of what's wrong with them) instead of actual disease-relevant facts. This was confirmed this with a test: comparing the facts the model called "important" against a random set of facts, and finding they performed the same. Rraced the cause to how the model was being selected during training — it locked in a version of itself after only 1-2 rounds of training, before it had learned anything meaningful. Fixed this by requiring more training before a model could be selected. Afterward, its explanations became trustworthy and lined up with real medical knowledge (e.g., tying heart attack risk to high blood pressure, high cholesterol, and diabetes). But this came at a cost: prediction accuracy on some diseases dropped once the model was trained properly instead of stopped early by accident.
\

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

> Among patients with a new cardiometabolic diagnosis, can we predict — using only their medical history up to that point — which circulatory disease they'll develop next, and how soon? Does representing that history as a time-ordered, connected knowledge graph improve predictions compared to standard models that just look at a flat list of facts about the patient?

The five outcomes we tried to predict (a patient can only have one *first* event, so the diseases "compete" to happen first):

| Code | Disease |
|---|---|
| MI | Heart attack |
| Stroke | Ischemic stroke |
| HF | Heart failure |
| AF | Atrial fibrillation |
| PAD | Peripheral artery disease |

This project has two parts: **(1)** a benchmark dataset — raw MIMIC-IV records turned into a time-stamped knowledge graph across 8 data types — and **(2)** a comparison of three models under a strict no-leakage setup, where the model's explanations are actually tested for validity rather than just shown.

---

## 2. Building the patient cohort

We scan every MIMIC-IV hospital admission and apply a series of filters:

1. Keep only adult patients (age 18+).
2. Find each patient's earliest admission with a cardiometabolic diagnosis (diabetes, high blood pressure, high cholesterol, obesity, or metabolic syndrome). This admission date becomes the **index date** — the starting point for prediction.
3. Find the first admission *after* the index date where a circulatory disease is the **main reason for that admission** (not just something mentioned in passing) — this is the disease we're trying to predict, and when it happened.
4. Remove anyone who already had a chronic or prior form of *any* of the five diseases, however it was recorded, at or before the index date. This is called a "washout" — it makes sure we're only trying to predict genuinely new disease, not disease someone already had.
5. Require at least 2 hospital visits and at least 90 days of follow-up (or an actual disease event).

We also calculate the Charlson Comorbidity Index (CCI), a standard score of overall health burden, at the index date.

### Why steps 3 and 4 matter, and what they cost us

An earlier version of this cohort logic counted a disease as "new" if it was mentioned *anywhere* on a later admission, even as a minor side-note — and used that same loose rule to check for prior disease. That's a problem: a chronic condition that's just mentioned in passing (like "known atrial fibrillation") could get counted as a *brand-new* case, and that same loose matching meant a patient's earlier chart could show that exact condition as an apparent "warning sign" for a disease they actually already had. That's a leak — the model would be predicting something it was secretly already being told about.

Fixing this (requiring the disease to be the *main* reason for the later admission, and removing anyone with any earlier record of it) closes that leak, but it does cost us some patients:

| Disease | Patients where it's mentioned anywhere | Patients where it's the main reason for admission (what we actually count) | % kept |
|---|---|---|---|
| MI | 13,152 | 7,700 | 59% |
| Stroke | 8,308 | 5,786 | 70% |
| HF | 31,369 | 5,081 | 16% |
| AF | 35,167 | 4,809 | 14% |
| PAD | 10,717 | 2,354 | 22% |

Heart failure and atrial fibrillation lose the most patients under this stricter rule. That's because they're very often noted as a side issue during a hospital stay for something else (like a patient admitted for pneumonia who also happens to have ongoing heart failure) — it's a real pattern in how hospitals code these conditions, not a bug in our logic. Separately, the washout step removes 52,901 patients total (some for more than one disease): 17,081 for MI, 15,255 for stroke, 19,003 for HF, 23,758 for AF, and 6,679 for PAD.

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

Heart failure patients are the sickest group overall — highest comorbidity score and the most ICU stays, which makes sense since HF is often a later complication of long-term cardiometabolic disease.

Files: `tkg_output/cohort.csv` and `cohort_cascade.csv` (the exact filtering counts above, used directly to draw the flow-diagram figure — nothing is hardcoded).

---

## 3. Building the knowledge graph

For every patient, we pull events from **8 different types of hospital data** and turn each one into a simple building block: *this patient, had this fact, at this time, optionally with this value.*

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

For each patient we only use data from the 5 years before their index date, and stop before their disease event (or their last known follow-up if they never had one) — so the model never sees the future. Lab results and ICU vital signs are large enough that we sample a portion of them (30% and 10%) to keep this manageable, while still capturing the overall pattern.

**Result: 14,480,074 facts** covering 32,965 distinct medical concepts and 33,656 patients (66,621 total "things" in the graph). On average, each patient has about 430 facts (typical patient: 221).

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

We double-checked the raw medication data file against PhysioNet's official checksum and confirmed it downloaded completely and correctly — an earlier concern that this file might be partially corrupted does not apply to the numbers here.

---

## 4. Checking the graph is correct

We ran 8 automated checks on the finished graph. **All 8 pass:**

1. No fact is dated on or after a patient's disease event (no "seeing the future").
2. Every patient in the cohort has at least one fact (11 patients, 0.03%, have very few — a trivial edge case).
3. No patient has the same medical code entered under both old and new coding systems at the same time.
4. Disease counts in the graph match the cohort file exactly.
5. No fact falls outside the intended 5-year window.
6-8. Key cardiometabolic drugs and labs (metformin, statins, HbA1c, creatinine, etc.) are present with realistic counts.

**One honest limitation:** the specific lab test "BNP" (a heart failure marker) doesn't appear at all in our data — MIMIC-IV records a related but different version of this test (NT-proBNP, which we do have, 5,822 facts) under a different label than what our matching rules look for. And for heart attack, the marker we capture is specifically "Troponin T" (9,010 facts) rather than "Troponin I" (only 43 facts). Anyone writing this up should say plainly: heart failure prediction is missing a natriuretic-peptide lab feature, and the heart attack marker used is troponin T specifically.

---

## 5. Preparing the data for modeling

- We keep only facts from the 5 years before each patient's index date: **2,283,125 events** used for modeling, out of the 14.5 million total facts.
- We split patients 70% training / 15% validation / 15% test, keeping the same disease mix in each group, using a fixed random seed so the split is reproducible.
- Basic patient info used as features: age, sex, comorbidity score, number of cardiometabolic conditions, whether they had an ICU stay.
- Every numeric value (labs, vitals, etc.) is standardized using only the training patients' statistics, so no information from the validation or test patients leaks into how the data is scaled.

| Disease | Training | Validation | Test |
|---|---|---|---|
| MI | 643 | 138 | 137 |
| Stroke | 497 | 106 | 107 |
| HF | 526 | 113 | 112 |
| AF | 477 | 102 | 103 |
| PAD | 211 | 45 | 46 |
| Censored | 21,205 | 4,544 | 4,544 |
| **Total** | **23,559** | **5,048** | **5,049** |

The number of actual disease cases in the test set is small, especially for PAD (46 patients). We say this plainly rather than hide it — it's exactly why we also report confidence intervals and repeat training with 5 different random seeds (see Section 8), rather than trusting a single run.

---

## 6. The three models

**Cox regression** — the traditional statistical method for this kind of prediction problem. One model is fit per disease.

**XGBoost** — a strong, widely-used machine learning model. It sees each patient as a flat list of features: which medical codes they have, summary statistics of their lab values (average, highest, lowest, most recent, trend), and their basic info — with no sense of order or connection between events.

**Our temporal knowledge graph model (TGN)** — reads a patient's most recent 256 events *in order*, as a connected sequence. Each event is represented by combining what it is, what kind of relationship it represents, when it happened relative to the index date, and its value if it has one. This sequence is processed by a small Transformer (the same family of model behind modern language models) and then boiled down into one summary vector for the patient using a learned "attention" mechanism that highlights the most relevant events. The final prediction is a probability of each disease happening within each time window, output all at once.

**An important note about how we picked the "final" version of this model.** In our first attempt, training was set up to stop automatically once it stopped improving on a validation check — and every one of 5 independent training runs stopped after just 1-2 passes through the data. We later discovered (Section 9) that the model saved at that point had explanations that were statistically meaningless. So we changed the rule: **the model must train for at least 15 full passes before it's allowed to be considered "done."** With that fix, all 5 runs trained for 21-23 passes before stopping. **Every result in this document uses the properly-trained version.** The full comparison between the too-early version and the properly-trained version is in Section 9, because it's an important finding in its own right, not just a technical footnote.

---

## 7. How we measured success

- **The metric: AUROC.** A score from 0.5 (no better than a coin flip) to 1.0 (perfect) measuring how well the model ranks patients who will get a disease above those who won't.
- **Correctly handling "competing" diseases.** If we're checking whether a model predicted heart attack well, and a patient instead got a stroke first, that patient should count as a "no" for heart attack — not be thrown out of the analysis, which is what a simpler (and overly generous) scoring method would do. We fixed this everywhere in the project. Patients who were simply lost to follow-up before the time window ended are still excluded, since we genuinely don't know what would have happened to them.
- **Confidence intervals and significance testing.** For our main results we compute a 95% confidence interval (by resampling the test patients 2,000 times) and use a standard statistical test (DeLong's test) to check whether one model is really better than another, or if the difference could just be noise.
- **Repeating training 5 times.** To make sure we aren't reporting a lucky (or unlucky) one-off result, we retrained the knowledge graph model and XGBoost five times each, with different random starting points (Cox regression doesn't have meaningful randomness to vary, so it's reported once). We then use a statistical test (Welch's t-test) to compare the five results from each model, and because we're running 30 comparisons at once (5 diseases x 3 time windows x 2 model comparisons), we apply a strict correction (Bonferroni) so we don't accidentally call something "significant" just because we tested so many things.
- **The test set is used exactly once** per trained model, at the very end. Every decision about how to normalize data, which codes to use, or which model checkpoint to keep, is made using only the training and validation patients.

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

We trained TGN and XGBoost five times each with different random seeds and compared the two sets of five results, applying a strict correction for testing many things at once. This is the most rigorous comparison in the study:

| Disease | Time window | Cox | XGBoost (average ± spread) | TGN (average ± spread) | Holds up after strict correction? |
|---|---|---|---|---|---|
| MI | 3yr | 0.742 | 0.721 ± 0.010 | 0.683 ± 0.015 | **Yes — TGN is worse than Cox** |
| MI | 5yr | 0.690 | 0.717 ± 0.010 | 0.672 ± 0.006 | **Yes — TGN is worse than XGBoost** |
| AF | 5yr | 0.670 | 0.677 ± 0.028 | 0.651 ± 0.004 | **Yes — TGN is worse than Cox** |
| PAD | 3yr | 0.592 | 0.682 ± 0.019 | 0.661 ± 0.020 | **Yes — TGN is better than Cox** |
| PAD | 5yr | 0.568 | 0.671 ± 0.021 | 0.675 ± 0.020 | **Yes — TGN is better than Cox** |
| The other 10 of 15 comparisons | — | — | — | — | Not proven either way with only 5 seeds |

**What this means:** no claim that "the knowledge graph model wins" survives this level of scrutiny. Where there's a real, provable difference, TGN is usually the *worse* model (heart attack, atrial fibrillation). The one genuine, repeatable win for TGN is against Cox regression on peripheral artery disease — it doesn't clearly beat XGBoost there, but it's no longer clearly behind it either. XGBoost is the strongest model overall on this data. We're reporting this as the real, main finding — not softening it.

### 8.5 The version with clinical notes

We did not evaluate the version of the model that also reads doctors' discharge notes (using a language model called BioBERT) for these results. We found a specific problem in that part of the code: only about 9% of patients have a usable note, and the way the note data gets standardized is thrown off by the 91% of patients who don't have one. We're flagging this clearly for anyone who wants to pick this up, but it wasn't required to answer our main question, and with only 9% coverage its effect either way would likely be small.

---

## 9. Can we trust the model's explanations?

### 9.1 Two different questions

"Where is the model looking?" (its attention weights) is a different question from "does that actually matter to its prediction?" To answer the second question, we used a tool called **GNNExplainer** together with a fidelity test: take the top 20% of events the model calls "important," and check — does keeping *only* those events preserve the prediction better than keeping a random 20%? And does removing *only* those events break the prediction more than removing a random 20%? A real, meaningful explanation should pass both checks clearly. If "important" performs the same as "random," the explanation isn't telling us anything real.

### 9.2 What we found with the too-early model

The first version of our model (the one that stopped training after just 1-2 passes) put most of its attention on things like a routine IV saline flush — something given to nearly every hospitalized patient regardless of their condition — rather than on actual diagnoses. Only 6-11% of its attention went to diagnosis codes; 34-48% went to medications like the saline flush. The fidelity test confirmed this wasn't just an odd-but-valid pattern: the "important" events performed statistically the same as randomly picked events, in both checks. The model's explanations carried no real information, even though its raw predictions still looked reasonable.

### 9.3 What changed after requiring more training

| | Before (stopped after 1 round) | After (properly trained, 15+ rounds) |
|---|---|---|
| Attention on diagnosis codes (heart attack) | ~9% | **62%** |
| Top facts for heart attack | Saline flush, routine lab values | High blood pressure, high cholesterol, coronary artery disease, diabetes |
| Does "important" beat "random" at preserving the prediction? | No — identical | **Yes — clearly and measurably better** |
| Does removing "important" hurt more than removing "random"? | No — identical | **Yes — over twice as much damage** |

Where the model's attention goes, by data type, after the fix:

| Disease | Diagnoses | Labs | Procedures | Medications | Blood pressure | BMI | Vitals |
|---|---|---|---|---|---|---|---|
| MI | 62.1% | 12.3% | 8.9% | 5.6% | 6.2% | 2.3% | 0.8% |
| Stroke | 61.3% | 15.8% | 6.2% | 5.4% | 6.1% | 3.8% | 0.9% |
| HF | 64.5% | 14.9% | 7.2% | 6.2% | 5.0% | 1.4% | 0.6% |
| AF | 59.4% | 15.4% | 9.3% | 4.7% | 6.5% | 3.7% | 0.5% |
| PAD | 64.6% | 8.6% | 7.6% | 8.9% | 7.1% | 2.5% | 0.4% |

Diagnosis codes are now clearly the dominant, trustworthy signal for every disease.

### 9.4 What's distinctive about each disease, according to the model

Beyond raw attention, we checked which facts are disproportionately linked to *one specific* disease rather than being generically common across all five:

| Disease | Cases in test set | Distinctive facts the model relies on |
|---|---|---|
| MI | 137 | Type-2 diabetes, chest pain history, HbA1c, **coronary artery disease**, high cholesterol |
| Stroke | 107 | Sodium level, normal blood pressure readings, kidney function, obesity, hypertension |
| HF | 112 | Kidney function (**the well-known heart-kidney link**), diabetes, **prior coronary artery disease** (the classic path from heart attack to heart failure), obesity |
| AF | 103 | Overweight/obesity, bicarbonate level, LDL cholesterol, specific cardiac procedures |
| PAD | 46 | High blood pressure, hypertension diagnosis (only 3 facts had enough patients behind them to count as reliable, since the PAD group is small) |

This is the strongest evidence that the model is reasoning sensibly: coronary artery disease specifically flags heart attack risk, and shows up again as a warning sign for heart failure — matching the well-known medical progression from heart attack to heart failure. Kidney function flagging heart failure risk matches the well-documented "heart-kidney" connection in medicine.

*One caveat:* GNNExplainer's own list of top facts per disease (feeding figure 17) isn't filtered for how many patients each fact applies to, so some of its top entries are one-off quirks from a single patient rather than a real pattern. This doesn't affect the overall fidelity numbers above, which are the trustworthy part of this check — just the detailed per-fact list in that one figure.

### 9.5 The bottom line

**Picking a "final" model based only on its validation accuracy can accidentally select a version whose stated reasons for its predictions are meaningless, even when the predictions themselves look fine.** Requiring a minimum amount of real training fixed this completely — its explanations became genuine and lined up with real medical knowledge — but cost some raw accuracy on certain diseases. Accuracy and trustworthy explanations were not the same thing here, and optimizing only for accuracy would have quietly shipped a model that explained itself with noise. We see this as a genuine lesson for how AI models should be checked before their explanations are trusted, not something to downplay.

---

## 10. Limitations

- **Single hospital system.** All data comes from one Boston hospital; how well this generalizes elsewhere is untested.
- **The clinical-notes version of the model wasn't evaluated**, and has a known, unfixed issue (Section 8.5) for anyone extending this work.
- **No model here clearly wins overall** once tested rigorously, except TGN beating Cox specifically on PAD. XGBoost is the strongest model on this data — we report that plainly rather than soften it.
- **Small numbers of actual disease cases in testing**, especially PAD (46 patients), which is why the confidence intervals are wide.
- **Heart failure and atrial fibrillation are defined strictly on purpose** (main-reason-for-admission only, broad washout), which trades away some statistical power for cleaner, leak-free labels — the exact cost is shown in Section 2.
- **Patients lost to follow-up are simply excluded** from a given time-window's scoring rather than statistically reweighted — a standard, disclosed simplification.
- **Labs and vital signs are sampled** (30% / 10%) rather than fully complete, purely to keep processing manageable.
- **Only 5 random seeds** were used for the robustness check — a reasonable minimum, not a generous number. More would tighten the confidence estimates further, especially for PAD.

---

## 11. How we avoided data leakage

The test set is used exactly once per trained model, at the very end.

- **The train/validation/test split is fixed once**, using a fixed random seed, and never changes.
- **Any medical code that only appears in the test set is treated as "unknown"** — the model never gets a real, learned understanding of it, since that would mean it learned something from data it shouldn't have seen yet.
- **All numeric scaling (labs, vitals, etc.) is based on training patients only.**
- **The time windows used by the model's prediction head are also set using training patients only.**
- **No fact used for prediction is ever dated on or after the outcome it's predicting** — this is enforced directly in the code and double-checked separately afterward.
- **The washout step** (Section 2) closes a specific leak where a chronic condition, mentioned casually, could act as both a "warning sign" before the fact and the "outcome" itself. A safety check runs automatically every time the code starts, confirming every disease code counted as an outcome is also covered by the washout rule — so this exact kind of bug can't quietly come back.
- **Which model checkpoint to keep is decided using validation data only**; test results are never looked at until the very end.
- **The rarer diseases are up-weighted during training using training-set statistics only**, so the model pays enough attention to them without any test-set information involved.

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
- `TKG_USE_NOTES` — `0` (structured data only; used for every result in this document) or `1` (also uses clinical notes; not evaluated here, see Section 8.5).
- `TKG_SEED` — changes only the model's own randomness (how its weights start, dropout, batch order). It does **not** change the train/validation/test split, which stays fixed no matter what seed you use — every seed sees exactly the same data, just trains the model slightly differently. Seed 42 is the main one used everywhere by default; seeds 43-46 write to their own separate folders so they never overwrite it.

**Other scripts you can run, once the steps above are done:**

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

**Data:** MIMIC-IV v3.1, plus MIMIC-IV-Note v2.2 if you want the optional clinical-notes version. Both require separate, credentialed access through PhysioNet. This repository contains no patient data — only the code that builds everything from the raw files.

---

## 15. Key settings

| Setting | Value |
|---|---|
| How far back we look before the index date | 5 years |
| Minimum follow-up required | 90 days |
| Lab / vital sign sampling | 30% / 10% |
| Longest event sequence the model reads | 256 events |
| Model size | 128-dimensional, 4 attention heads, 2 layers |
| How the model represents time | Bochner/TGAT time encoding |
| Prediction horizons checked | 1, 3, and 5 years |
| Minimum training rounds before a model can be selected | 15 |
| How long training can run before stopping automatically | up to 30 rounds, stops after 6 rounds without improvement |
| Number of random seeds used for the robustness check | 5 (seeds 42-46) |
| Main/default seed | 42 |

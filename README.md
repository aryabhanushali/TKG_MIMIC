# CardioMM-TKG

**A benchmark and modeling study on MIMIC-IV: predicting circulatory disease in cardiometabolic patients, and testing whether a temporal knowledge graph helps.**
## Summary

For a patient with a cardiometabolic condition (diabetes, high blood pressure, high cholesterol, obesity, or metabolic syndrome), can we predict which of five circulatory diseases they'll develop next, and when? Does representing their medical history as a **temporal knowledge graph** — a connected, time-ordered record of everything that happened to them — improve predictions over standard approaches?

The five diseases: heart attack (MI), stroke, heart failure (HF), atrial fibrillation / irregular heartbeat (AF), and peripheral artery disease (PAD, narrowed leg arteries). Patients who don't develop any of these are "censored" — followed for years with no event.

Cohort: 33,656 patients from MIMIC-IV, built with a specific eye toward avoiding data leakage (Section 2). Graph: 8 data types, 14.5 million facts. Five models compared: **Cox regression** and **XGBoost** (the two standard approaches), a **TKG-Transformer** that reads each patient's history as an ordered sequence of time-stamped events pulled from the graph, and two genuinely graph-structured models — one that turned out not to help, and one that did.

A naming note up front: the "TKG-Transformer" is not a "temporal graph neural network," on purpose. It's a Transformer that consumes facts drawn from the knowledge graph as a flat, ordered sequence — it does not do graph message-passing between connected concepts. Earlier drafts of this document called it "TGN" in a way that overstated that; see Section 6 for exactly what it does and doesn't do, and for the two models further down that *do* do real graph message-passing.

**A leak we found and fixed late, disclosed here rather than buried.** A code audit found that one of the patient features fed to every model — "did this patient ever have an ICU stay" — was computed with no time restriction at all: it counted ICU stays during or after the very hospitalization that defined a patient's disease event, not just before it. For an acute condition like a heart attack or stroke that very often triggers an ICU admission, that's the outcome leaking into a predictor. We measured it directly: the properly time-restricted version of this feature is true for under 2% of patients in every group; the leaky version was true for 35-69% of patients depending on the disease. Every number in this document was recomputed after fixing it (along with two smaller, related issues — see Section 11). Heart failure's results changed the most, dropping across all three original models; stroke and PAD changed the least. This is exactly the kind of thing a strict leakage audit is supposed to catch, and it's why Section 11 and `src/tests_integrity.py` exist.

**Main finding:** the plain TKG-Transformer does not reliably beat XGBoost — XGBoost is the strongest model overall for most of this project. But a later, more genuinely graph-structured model changed that picture in specific places. The first attempt at adding real graph structure — enriching each medical concept's representation using a hand-built hierarchy (a code belongs to a category, a drug belongs to a class), while still treating every patient as an isolated sequence — made things worse on nearly every disease, not better (Section 6.2). The second attempt — making patients themselves real nodes in the graph, connected to other patients only through the concepts they share, with real information passing between them — is the strongest result in the whole study: it beats classical Cox regression on stroke at 3 years and on heart attack at 5 years, holding up under the same strict statistical correction used everywhere else. It loses clearly to XGBoost on heart failure at every horizon, and the PAD result is genuinely mixed (wins in some places, losses in others, on a very small sample). This isn't "the graph model wins" across the board — it's specific and horizon-dependent, same as everything else in this study. See Section 6.3 and Section 8.6 for the full picture, including why the *design* of the graph mattered more than just adding one, and Section 8.7 for a mechanistic theory of exactly when it helps.

**Second finding:** partway through, the plain TKG-Transformer's "explanations" — the facts it claimed to rely on for each prediction — turned out to be meaningless. It was focusing on things like routine IV saline flushes, given to nearly every hospital patient regardless of diagnosis, instead of anything disease-relevant. This was confirmed with a formal test: take the facts the model calls "important," compare them against a random set of facts, and check whether "important" actually performs better. It didn't — statistically identical to random. The cause traced back to how the model was picked during training: it locked in a version of itself after only 1-2 training rounds, before it had learned anything real. Requiring more training before a model could be selected fixed this. Afterward its explanations lined up with real medical knowledge — tying heart attack risk to high blood pressure, high cholesterol, and diabetes. The same check was later extended to both graph attempts (Section 9.6): the one that failed on accuracy also came up short on one of the two fidelity checks, and the one that worked passed both cleanly.


---

## Table of contents

1. [The research question](#1-the-research-question)
2. [Building the patient cohort](#2-building-the-patient-cohort)
3. [Building the knowledge graph](#3-building-the-knowledge-graph)
4. [Checking the graph is correct](#4-checking-the-graph-is-correct)
5. [Preparing the data for modeling](#5-preparing-the-data-for-modeling)
6. [The models](#6-the-models)
7. [How we measured success](#7-how-we-measured-success)
8. [Results](#8-results)
   - [8.6 Two attempts at real graph structure](#86-two-attempts-at-real-graph-structure)
   - [8.7 When does the graph help? A mechanism analysis](#87-when-does-the-graph-help-a-mechanism-analysis)
9. [Can we trust the model's explanations?](#9-can-we-trust-the-models-explanations)
   - [9.6 Does this hold up for the two graph models?](#96-does-this-hold-up-for-the-two-graph-models)
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

Two parts to this project: a benchmark dataset — raw MIMIC-IV records turned into a time-stamped knowledge graph across 8 data types — and a comparison of several models (classical statistics, standard machine learning, and two different attempts at real graph structure) under a strict no-leakage setup, with the model's explanations actually tested for validity rather than just shown.

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

| Disease | Patients | % of cohort | Median age | % female | Median CCI | % with a pre-index ICU stay | Median years until event |
|---|---|---|---|---|---|---|---|
| MI | 918 | 2.7% | 66 | 43.9% | 1 | 0.4% | ~3.6 |
| Stroke | 710 | 2.1% | 69 | 53.1% | 1 | 0.8% | ~3.3 |
| HF | 751 | 2.2% | 69 | 57.8% | 2 | 0.5% | ~2.5 |
| AF | 682 | 2.0% | 69 | 49.9% | 1 | 1.0% | ~3.3 |
| PAD | 302 | 0.9% | 66 | 43.0% | 1 | 0.7% | ~2.7 |
| Censored (no event) | 30,293 | 90.0% | 62 | 55.1% | 1 | 1.6% | ~3.1 |
| **Total** | **33,656** | 100% | 62 | 54.6% | 1 | 1.5% | ~3.1 |

Heart failure patients are the sickest group overall by comorbidity score, which fits HF's usual role as a later complication of long-running cardiometabolic disease.

The "% with a pre-index ICU stay" column above is deliberately restricted to ICU stays that started *before* the index date — an earlier version of this column counted an ICU stay any time in a patient's record, including during or after the hospitalization that defined their disease event. That version showed MI at 63.6% vs. censored at 35.5%, which looked like a strong risk signal but was mostly circular: an ICU stay *during the heart attack admission itself* isn't a pre-existing risk factor, it's part of the outcome. Restricted to genuinely pre-index stays, the rate is a flat 0.4-1.6% across every group, which is what the models actually use as a feature now. See Section 11 for the fix and how it changed the results.

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

## 6. The models

### 6.1 The three original models

**Cox regression** — the traditional statistical method for this kind of problem. One model per disease.

**XGBoost** — a strong, widely-used machine learning model. Sees each patient as a flat list of features: which medical codes they have, summary statistics of their lab values (average, highest, lowest, most recent, trend), and their basic info, with no sense of order or connection between events.

**The TKG-Transformer** — reads a patient's most recent 256 events in order, as a flat time-ordered sequence pulled from the knowledge graph. Each event combines what it is, what kind of relationship it represents, when it happened relative to the index date, and its value if it has one. A small Transformer (the same family of model behind modern language models) processes the sequence and boils it down into one summary vector per patient, using a learned attention mechanism that highlights the most relevant events. The output is a probability of each disease happening within each time window.

To be precise about what "knowledge graph" means here: the graph structure (which concepts, which relationships, which timestamps) is what generates the facts fed into the model, but the model itself is a standard sequence Transformer — self-attention runs over every event in a patient's sequence uniformly, not restricted to graph neighbors, and there's no message-passing between connected concepts and no shared memory across patients. It's fair to call this "a model trained on knowledge-graph-derived data," not "a graph neural network." Sections 6.2 and 6.3 describe two models that actually are.

Only 256 of a patient's events are used, most-recent-first, even though the full pre-index history can be much longer. Across the cohort, 2.5% of patients have more than 256 eligible events, and truncating to 256 discards 12.1% of all eligible events overall — not a small amount. Truncation is not concentrated in the sickest patients: censored patients are truncated most often (2.6%), ahead of heart failure (2.3%) and well ahead of MI (0.7%), so it's not quietly hiding disease-relevant history right before an event. Still, no test was run at other sequence lengths (128, 512, 1024), so how much this specific choice affects the results is currently unmeasured.

**A note on how the "final" version of this model was picked.** The first attempt used automatic stopping once a validation check stopped improving — and every one of 5 independent training runs stopped after just 1-2 passes through the data. Section 9 shows why that's a problem: the model saved at that point had explanations that were statistically meaningless. The fix was to require at least 15 full passes before a model becomes eligible to be called "done." With that change, all 5 runs trained for 22-26 passes before stopping. Every result in this document uses the properly-trained version. The comparison between the two versions is in Section 9 — it's a real finding, not a footnote. The same 15-pass rule is applied identically to every model in Sections 6.2 and 6.3 below, for a fair comparison.

**A note on what each model is actually predicting.** Cox and XGBoost are fit one-per-disease: when fitting the heart attack model, a patient who instead had a stroke is treated as "censored" at the time of their stroke, not as a clean "no." That's a standard, defensible approach (called a "cause-specific" model), but it answers a subtly different question than the graph-based models, which are trained all at once across all five diseases and directly output a probability that accounts for the other four diseases competing to happen first. Both are legitimate ways to handle competing diseases, but they're not estimating exactly the same thing, so a difference in their scores isn't purely "which model is smarter" — part of it could just be "these are two different, both-reasonable definitions of risk." Every model is scored the same way at evaluation time (Section 7), which keeps the *comparison* fair even though the *training setup* differs.

None of the models had their settings tuned. Cox's penalty strength, XGBoost's tree count/depth/learning rate, and every neural model's size and learning rate are all fixed, hand-picked values — no systematic search was run for any of them, using only training and validation data, before arriving at these numbers. That means the results below compare *these specific configurations*, not necessarily the best each model family could do. This is a real gap: it's possible a properly tuned version of any model here would perform differently, and closing this gap is on the to-do list, not done.

### 6.2 First attempt at real graph structure: enrich the concepts, not the patients (didn't help)

The natural first thing to try is to keep the TKG-Transformer's sequence design but make each *concept's* representation smarter: instead of "heart attack code" being just a free-floating learned vector, let it absorb information from a small hand-built medical hierarchy — a specific diagnosis code belongs to a broader category, which belongs to a chapter; a specific drug belongs to a drug class; a specific lab belongs to a lab category. A graph technique called an R-GCN (a network that lets each concept's representation average in information from its connected neighbors) processes this hierarchy, and the enriched concept representations then feed into the same sequence Transformer as before.

Two versions of this were tested: one that kept the full time-ordered sequence on top of the enriched concepts, and one that dropped time and sequence entirely, just averaging a patient's enriched concepts together (isolating whether the graph structure by itself was doing anything, independent of timing).

**Essentially, neither helped.** The version that kept the sequence was worse than the plain TKG-Transformer on every single one of the five diseases, at 3 years. The version that dropped the sequence entirely was worse on four of five — but slightly *better* than the plain model on heart failure specifically, which is enough of an exception that "worse on every disease, no exceptions" would overstate the finding:

| Disease | Plain TKG-Transformer | + graph-enriched concepts, kept sequence | + graph-enriched concepts, dropped sequence |
|---|---|---|---|
| MI | 0.674 | 0.596 | 0.585 |
| Stroke | 0.663 | 0.570 | 0.532 |
| HF | 0.640 | 0.533 | **0.658** |
| AF | 0.652 | 0.628 | 0.599 |
| PAD | 0.618 | 0.585 | 0.589 |

Three honest reasons this is the likely explanation, not "graph structure never works" (this exact question, and the answer, is explored in more depth after the results in Section 8.6):

1. **The hierarchy barely touched the biggest category of data.** Medications are the single largest modality (32% of all facts), but the hand-curated drug-class dictionary only covers 9.3% of distinct medications — cardiometabolic drugs specifically. For 90% of medications, there was nothing to connect to.
2. **More moving parts, same amount of data.** Both graph variants have close to double the internal settings of the plain model, learning from the same ~23,500 patients — a textbook setup for memorizing training quirks rather than learning something general, and Section 8.7's oversmoothing check finds direct evidence of representational collapse in exactly the poorly-connected (mostly drug) concepts this dictionary doesn't cover.
3. **This is fundamentally a timing question**, and the hierarchy doesn't add timing information — it adds "how related are these two medical concepts," which turned out to matter less here than "when did this happen relative to the others."

### 6.3 Second attempt: make patients real nodes in the graph (this one worked, in specific places)

The first attempt never actually gave patients any connection to each other — every patient was still a fully isolated sequence; only the concepts they referenced got smarter. The bigger, more genuine use of graph structure is to make **patients themselves nodes in the graph**, connected to other patients only through the medical concepts they share. This is the one thing a sequence model structurally cannot do: let one patient's prediction be shaped by patterns learned from *other* patients, several hops away, through the graph.

Concretely: every patient becomes a node, connected to every concept they had (split into three simple time buckets — within 90 days of the index date, 90 days to 2 years, or older — so some timing signal survives), and concepts are additionally connected to each other two ways: the same hand-built hierarchy as before, plus a new, purely data-driven layer — two concepts get an edge if they co-occur often enough across training patients' histories (computed using training patients only, so no test information shapes the graph itself). This second, data-driven layer matters a lot: unlike the hand-built hierarchy, it works uniformly across every modality, including medications, without needing anyone to curate a drug dictionary.

Two rounds of message-passing then let information flow patient → concept → a *different* patient who shares that concept → back — a real network effect, not just per-patient enrichment. Training this way is also naturally more efficient: one pass over the whole graph per training round, rather than reprocessing each patient's sequence separately, and in practice it trained in about 1-2 minutes total (22-30 rounds at 2.5-3.8 seconds each, depending on seed; the seed-42 canonical run: 24 rounds in 65 seconds) versus 20-50 minutes for the sequence-based models (measured from each model's own saved training log: 20.2-48.4 minutes for the plain TKG-Transformer across its 5 seeds, 37.9 minutes for the first graph attempt's "full" variant, 21.7 minutes for its "static" variant).

**Results, 3-year AUROC, averaged over the same 5-seed check used everywhere else in this study:**

| Disease | Cox | XGBoost | Patient-graph model |
|---|---|---|---|
| MI | 0.730 | 0.712 | 0.721 |
| Stroke | 0.636 | 0.647 | **0.689** |
| HF | 0.628 | 0.697 | 0.635 |
| AF | 0.685 | 0.666 | 0.663 |
| PAD | 0.592 | 0.680 | 0.624 |

Full results, including 1- and 5-year horizons and which of these differences survive strict statistical correction, are in Section 8.6.

**In short:** this model beats Cox, with statistical confidence, on stroke at 3 years and on heart attack at 5 years — the two clearest, most repeatable wins for any graph-based model in this study. It loses clearly to XGBoost on heart failure at every horizon. PAD is genuinely mixed: it loses to both baselines at 1 year, loses to XGBoost at 3 years, but beats Cox at 5 years — on a test set of only 46-52 PAD events, so read that mixed pattern as noisy rather than as a clean story. This isn't "graph structure wins" — it's specific and horizon-dependent, exactly like every other finding here — but the stroke and heart-attack wins replicated across all 5 training seeds, and unlike every sequence-based model tried in this study, its practice-test score never declined during training at all (it climbed and then flattened out), which is itself a sign of healthier, more stable learning.

The same explanation-fidelity check from Section 9 was extended to this model too (Section 9.6): its implicit "which concepts drove this prediction" signal passes both directions of the check cleanly, the same as the plain TKG-Transformer. Section 8.7 also tests a specific theory of *why* the graph helps on some diseases and not others — it isn't a random pattern.

---

## 7. How we measured success

- **The metric: AUROC.** A score from 0.5 (a coin flip) to 1.0 (perfect) measuring how well the model ranks patients who will get a disease above those who won't.
- **Handling "competing" diseases correctly.** If a model is being checked on whether it predicted heart attack, and a patient got a stroke first instead, that patient counts as a "no" for heart attack — not thrown out of the analysis, which is what a simpler, overly generous scoring method would do. Fixed everywhere in this project. Patients simply lost to follow-up before the time window ended are still excluded, since what would have happened to them is genuinely unknown.
- **How many patients that excludes.** At the 1-year mark, 19.7% of the test set has been followed for less than a year and gets dropped from that specific check. At 3 years it's 44.3%, and at 5 years it's 59.6% — meaning the 5-year numbers below come from well under half the test set. This isn't a flaw unique to this study (it's standard practice, called "administrative censoring"), but it means the 5-year results rest on a smaller, and potentially different, group of patients than the 1-year results, and that's worth keeping in mind rather than treating all three time windows as equally solid.
- **Confidence intervals and significance testing.** The main results include a 95% confidence interval (resampling the test patients 2,000 times) and a standard statistical test (DeLong's) for whether one model is really better than another, or if the gap is just noise.
- **Repeating training 5 times.** To rule out a lucky or unlucky one-off run, the TKG-Transformer and XGBoost were each retrained five times with different random seeds (Cox has no meaningful randomness to vary, so it's reported once). The five results per model are compared with Welch's t-test, and because 30 comparisons are running at once (5 diseases × 3 time windows × 2 model comparisons), a strict correction (Bonferroni) is applied so nothing gets called "significant" just from sheer number of tests.
- **The test set is used exactly once** per trained model, at the very end. Every decision about normalization, which codes to use, or which model checkpoint to keep is made using only training and validation data.

---

## 8. Results

### 8.1 Test accuracy (AUROC) by disease, time window, and model

| Disease | Time window | Cox | XGBoost | TKG-Transformer |
|---|---|---|---|---|
| MI | 1yr / 3yr / 5yr | 0.709 / 0.730 / 0.684 | 0.726 / 0.704 / 0.693 | 0.639 / 0.674 / 0.670 |
| Stroke | 1yr / 3yr / 5yr | 0.632 / 0.636 / 0.656 | 0.650 / 0.660 / 0.674 | 0.761 / 0.663 / 0.640 |
| HF | 1yr / 3yr / 5yr | 0.674 / 0.628 / 0.617 | 0.702 / 0.703 / 0.693 | 0.623 / 0.640 / 0.650 |
| AF | 1yr / 3yr / 5yr | 0.724 / 0.685 / 0.670 | 0.713 / 0.677 / 0.675 | 0.660 / 0.652 / 0.650 |
| PAD | 1yr / 3yr / 5yr | 0.634 / 0.592 / 0.568 | 0.702 / 0.692 / 0.658 | 0.628 / 0.618 / 0.589 |

Remember from Section 7 that the 5-year column for every disease comes from only about 40% of the test set (the other 60% hadn't been followed long enough and were dropped from that check) — the 1-year column is on much steadier ground.

These numbers are noticeably lower for heart failure than an earlier version of this document reported (Cox went from 0.736 to 0.628 at 3 years, XGBoost from 0.750 to 0.703, TKG-Transformer from 0.730 to 0.640) — that's the `had_icu_stay` leak fix (see the note in the Summary and Section 11), and heart failure lost the most because it had the strongest association with the leaky version of that feature. Atrial fibrillation's XGBoost result dropped for the same reason (0.729 → 0.677 at 3 years). Stroke and PAD barely moved, since their Cox/XGBoost fits weren't leaning on that feature much either way.

### 8.2 3-year results with 95% confidence intervals

| Disease | Cox | XGBoost | TKG-Transformer |
|---|---|---|---|
| MI | 0.730 [0.660 - 0.790] | 0.704 [0.634 - 0.773] | 0.674 [0.597 - 0.750] |
| Stroke | 0.636 [0.561 - 0.702] | 0.660 [0.583 - 0.731] | 0.663 [0.588 - 0.737] |
| HF | 0.628 [0.541 - 0.707] | 0.703 [0.628 - 0.772] | 0.640 [0.572 - 0.704] |
| AF | 0.685 [0.611 - 0.755] | 0.677 [0.604 - 0.746] | 0.652 [0.577 - 0.725] |
| PAD | 0.592 [0.496 - 0.685] | 0.692 [0.555 - 0.831] | 0.618 [0.493 - 0.738] |

Look at how wide the PAD interval is compared to the others (0.555 to 0.831 for XGBoost alone) — that's the direct effect of PAD having only 21 observed events in this window of the test set. Every PAD number in this document should be read with that width in mind.

### 8.3 Is the TKG-Transformer significantly different from XGBoost? (one model each, 3-year window)

| Disease | Difference (TKG-Transformer minus XGBoost) | Result |
|---|---|---|
| MI | -0.031 | No real difference (wide, overlapping intervals) |
| Stroke | +0.002 | No real difference |
| HF | -0.063 | Overlapping intervals, but see the 5-seed check below |
| AF | -0.025 | No real difference |
| PAD | -0.074 | No real difference (PAD's interval is very wide) |

A single seed each is too little to say much on its own — Section 8.4's 5-seed check is the more trustworthy version of this comparison, so this section is kept short deliberately.

### 8.4 The 5-seed check — the most trustworthy result

The TKG-Transformer and XGBoost were each trained five times with different random seeds, then compared against Cox and against XGBoost with a strict correction (Bonferroni, 30 simultaneous comparisons). This is the most rigorous comparison in the study for the plain TKG-Transformer:

| Disease | Time window | Cox | XGBoost (average ± spread) | TKG-Transformer (average ± spread) | Holds up after strict correction? |
|---|---|---|---|---|---|
| Stroke | 3yr | 0.636 | 0.647 ± 0.019 | 0.659 ± 0.003 | Yes — better than Cox (small effect: +0.024) |
| HF | 1yr | 0.674 | 0.707 ± 0.009 | 0.616 ± 0.025 | Yes — worse than XGBoost |
| HF | 3yr | 0.628 | 0.697 ± 0.011 | 0.637 ± 0.014 | Yes — worse than XGBoost |
| AF | 1yr | 0.724 | 0.715 ± 0.026 | 0.676 ± 0.014 | Yes — worse than Cox |
| AF | 3yr | 0.685 | 0.666 ± 0.017 | 0.647 ± 0.010 | Yes — worse than Cox |
| AF | 5yr | 0.670 | 0.662 ± 0.011 | 0.643 ± 0.006 | Yes — worse than Cox |
| The other 9 of 15 comparisons | — | — | — | — | Not proven either way with only 5 seeds |

No version of "the knowledge graph model wins outright" survives this level of scrutiny. Where there's a real, provable difference, the TKG-Transformer is usually the worse model — heart failure, atrial fibrillation. The one exception is a small but statistically solid edge on stroke at 3 years. XGBoost is the strongest model overall on this data, and that's reported as the main finding, not softened.

**A note on what changed here.** An earlier version of this check (before the leakage fix) found the TKG-Transformer's only repeatable win on a small PAD test set, plus losses on MI and AF-at-5-years specifically. After the fix, that PAD win disappeared (PAD isn't in the significant list at all for the plain TKG-Transformer anymore), the MI losses disappeared, and AF's losses to Cox got both stronger and broader (all three horizons now, not just one). Heart failure — untouched by this comparison before, since it wasn't a significant finding either way — is now a clear, three-horizon loss to XGBoost. This is a good example of why disclosing a leak and rerunning everything matters: it didn't just shrink the numbers, it changed which findings are real.

### 8.5 The version with clinical notes

Not evaluated for these results. There's a specific problem in that part of the code: only about 9% of patients have a usable discharge note, and the way the note data gets standardized is thrown off by the 91% of patients who don't have one — the normalization statistics end up skewed toward "no note" rather than what a real note looks like. Flagged here for anyone picking this up, but it wasn't required to answer the main question, and with 9% coverage its effect either way would likely be small.

### 8.6 Two attempts at real graph structure

Section 6.2 and 6.3 describe *what* was tried; this is the full result set for both.

**Attempt 1 (concepts enriched, patients still isolated) — single seed, worse than the plain model on 9 of 10 disease/variant combinations:**

| Disease | Horizon | Cox | XGBoost | Plain TKG-Transformer | + graph concepts, kept sequence | + graph concepts, dropped sequence |
|---|---|---|---|---|---|---|
| MI | 1y/3y/5y | 0.709/0.730/0.684 | 0.726/0.704/0.693 | 0.639/0.674/0.670 | 0.617/0.596/0.575 | 0.598/0.585/0.581 |
| Stroke | 1y/3y/5y | 0.632/0.636/0.656 | 0.650/0.660/0.674 | 0.761/0.663/0.640 | 0.613/0.570/0.548 | 0.554/0.532/0.500 |
| HF | 1y/3y/5y | 0.674/0.628/0.617 | 0.702/0.703/0.693 | 0.623/0.640/0.650 | 0.554/0.533/0.554 | 0.671/**0.658**/0.639 |
| AF | 1y/3y/5y | 0.724/0.685/0.670 | 0.713/0.677/0.675 | 0.660/0.652/0.650 | 0.640/0.628/0.607 | 0.611/0.599/0.590 |
| PAD | 1y/3y/5y | 0.634/0.592/0.568 | 0.702/0.692/0.658 | 0.628/0.618/0.589 | 0.581/0.585/0.578 | 0.593/0.589/0.577 |

Given this is worse than the plain model in 29 of 30 disease/horizon/variant cells (the lone exception: "dropped sequence" on heart failure at 3 years, bolded above), this wasn't pursued further with a full 5-seed check. A single, consistent signal across nearly every disease and horizon is already fairly informative on its own, and the likely reasons (Section 6.2) point at the specific design, not at graph structure in general.

**Attempt 2 (patients as real graph nodes) — full 5-seed check, same strict correction as the rest of this study:**

| Disease | Horizon | Cox | XGBoost (mean ± spread) | Patient-graph model (mean ± spread) |
|---|---|---|---|---|
| MI | 1y | 0.709 | 0.728 ± 0.014 | 0.696 ± 0.025 |
| MI | 3y | 0.730 | 0.712 ± 0.009 | 0.721 ± 0.009 |
| MI | 5y | 0.684 | 0.700 ± 0.006 | **0.715 ± 0.008** |
| Stroke | 1y | 0.632 | 0.633 ± 0.020 | 0.676 ± 0.028 |
| Stroke | 3y | 0.636 | 0.647 ± 0.019 | **0.689 ± 0.014** |
| Stroke | 5y | 0.656 | 0.664 ± 0.017 | 0.699 ± 0.014 |
| HF | 1y | 0.674 | 0.707 ± 0.009 | **0.660 ± 0.013** |
| HF | 3y | 0.628 | 0.697 ± 0.011 | **0.635 ± 0.010** |
| HF | 5y | 0.617 | 0.678 ± 0.016 | **0.633 ± 0.008** |
| AF | 1y | 0.724 | 0.715 ± 0.026 | 0.684 ± 0.024 |
| AF | 3y | 0.685 | 0.666 ± 0.017 | 0.663 ± 0.013 |
| AF | 5y | 0.670 | 0.662 ± 0.011 | 0.658 ± 0.009 |
| PAD | 1y | 0.634 | 0.692 ± 0.021 | **0.565 ± 0.009** |
| PAD | 3y | 0.592 | 0.680 ± 0.014 | **0.624 ± 0.010** |
| PAD | 5y | 0.568 | 0.666 ± 0.015 | **0.637 ± 0.011** |

**What survives the same strict Bonferroni correction used everywhere else in this study** (30 comparisons tested at once, so the bar for "significant" is a p-value below 0.00167, not the usual 0.05):

| Finding | Effect size | p-value |
|---|---|---|
| Loses to XGBoost on PAD, 1 year | -0.127 | 0.0000 |
| Beats Cox on PAD, 5 years | +0.069 | 0.0002 |
| Loses to Cox on PAD, 1 year | -0.070 | 0.0001 |
| Loses to XGBoost on heart failure, 3 years | -0.062 | 0.0000 |
| Beats Cox on stroke, 3 years | +0.053 | 0.0011 |
| Loses to XGBoost on PAD, 3 years | -0.056 | 0.0001 |
| Loses to XGBoost on heart failure, 1 year | -0.047 | 0.0003 |
| Loses to XGBoost on heart failure, 5 years | -0.045 | 0.0014 |
| Beats Cox on heart attack, 5 years | +0.032 | 0.0009 |

Everything not listed above didn't reach this strict bar in either direction — real numeric differences exist (see the full table), but 5 seeds isn't enough to call them proven.

**What changed here after the leakage fix.** Before the fix, this model's significant wins included stroke at both 3 and 5 years and heart attack at 3 years (vs. XGBoost); after the fix, only stroke at 3 years and heart attack at 5 years survive — the 5-year stroke win and the 3-year heart-attack win are still real, numeric advantages, just no longer large enough (or the comparator's spread narrow enough) to clear the strict Bonferroni bar with only 5 seeds. Heart failure went from a single significant loss (to XGBoost at 3 years) to a loss at all three horizons — the clearest strengthening of any finding in this table. Both models use the same `had_icu_stay` static feature, and both dropped on heart failure after the fix, but XGBoost's drop was smaller and its spread across seeds narrower, which is enough on its own to turn a borderline gap into one that clears the strict Bonferroni bar at every horizon. PAD's pattern — losing at 1 and 3 years, winning at 5 years — held up essentially unchanged; treat it as noisy given the horizon has only 46-52 PAD test events, not as a real "gets better over time" story.

**The honest read:** stroke at 3 years and heart attack at 5 years are the clearest, most repeatable wins in this entire study — both beat Cox with high statistical confidence, and both replicated across all 5 seeds. Heart failure at every horizon and PAD at 1 and 3 years are real, equally confident losses to XGBoost or Cox. This is not "the graph model wins" — it's a specific, horizon-dependent pattern, same as everywhere else in this document — but stroke-at-3-years in particular has now survived two full rebuilds of this pipeline (before and after the leakage fix), which is about as replicated as any single finding gets in this study.

### 8.7 When does the graph help? A mechanism analysis

Section 6.3 and 8.6 show *that* the patient-graph model helps on some diseases and not others. This section tests a specific, falsifiable theory of *why*, using only train/validation-safe quantities computed before looking at any graph-vs-baseline comparison, then checking whether that theory lines up with the actual pattern of wins and losses.

**The theory: risk-factor concentration.** Some diseases have one or two facts that are, on their own, already fairly predictive (e.g., a specific diagnosis code that shows up in a large share of cases and rarely elsewhere). Others have no single strong signal — risk is spread thinly across many weakly-informative facts. The theory is that a graph model, which can combine weak signals across many concepts and across patients, should have more room to add value on the second kind of disease than the first, where a model that just looks at the concentrated signal directly (like XGBoost, feature by feature) is already close to as good as it's going to get.

To test this without peeking at the graph-vs-baseline comparison, `src/ablations/mechanism_analysis.py` computes, from training data only: the single best-performing individual fact for each disease (AUROC of that one fact alone, no model), and the Shannon entropy of how spread out the predictive signal is across all facts. "Graph advantage" is then defined separately, from already-saved test results, as the patient-graph model's AUROC minus XGBoost's AUROC at 3 years:

| Disease | Best single fact's AUROC (higher = more concentrated) | Normalized entropy (higher = more spread out) | Graph advantage (patient-graph minus XGBoost, 3y) |
|---|---|---|---|
| HF | **0.628** (most concentrated) | 0.845 | **-0.062** (graph hurts most) |
| PAD | 0.591 | 0.868 | -0.056 |
| MI | 0.579 | 0.859 | +0.009 |
| AF | 0.558 | 0.892 | -0.003 |
| Stroke | 0.554 (most spread out) | **0.893** (most spread out) | **+0.042** (graph helps most) |

(MI and AF swap places between the two middle rows depending on which metric you sort by — the correlation below is real but not a perfectly clean monotonic ordering.)

The pattern matches the theory in exactly the predicted direction: heart failure has the single most concentrated risk signal *and* the worst graph result; stroke has the most spread-out signal *and* the best graph result. With only 5 diseases this is a descriptive correlation, not a proof (Spearman's rho = -0.90 between concentration and graph advantage, p = 0.037 — technically under 0.05, but interpret a 5-point correlation cautiously regardless of the p-value), and it should be treated as a hypothesis for future work on more diseases or datasets, not a settled mechanism. A competing, simpler theory — "the graph just helps more on rarer diseases with less data" — is directly checked and does not hold: case count and graph advantage have a small *positive* correlation (rho = 0.20, p = 0.75, not significant), and PAD is both the smallest disease group and one of the worst graph results, the opposite of what that theory would predict.

**Why the first graph attempt failed: a closer look.** `src/ablations/oversmoothing_diagnostic.py` loads the already-trained concept-enrichment checkpoint (Section 6.2's "full" variant) and compares concept embeddings before and after message-passing, with no retraining. Effective rank (a measure of how many genuinely distinct directions the embeddings span) drops from 8.8 to 7.6 — a real, if modest, collapse toward fewer distinct representations. Splitting concepts by how well-connected they are in the hand-built hierarchy tells a sharper story: well-connected concepts actually became *more* distinguishable from each other after message-passing (average pairwise similarity 0.081 → 0.050), while poorly-connected concepts — mostly medications, which the hierarchy dictionary only covers for 9.3% of drugs — collapsed toward each other sharply (0.038 → 0.101). The mechanism isn't "message-passing blurs everything a little"; it's specific to exactly the part of the graph that has the least real structure to share.

**Is the patient-graph model doing more than simple pattern-matching?** `src/ablations/knn_baseline.py` builds the simplest possible "model" that could exploit the same information the patient-graph model has access to: for each test patient, find the k most similar training patients by shared concepts (k=200, chosen on validation data only) and use the fraction of them who developed a disease as the risk score. No learning, no embeddings, no message-passing.

| Disease | k-NN (3y AUROC) | Patient-graph model (3y) | Difference |
|---|---|---|---|
| Stroke | 0.568 | 0.689 | Graph +0.121 |
| MI | 0.670 | 0.721 | Graph +0.051 |
| AF | 0.625 | 0.663 | Graph +0.038 |
| HF | 0.658 | 0.635 | k-NN +0.023 |
| PAD | 0.685 | 0.624 | k-NN +0.061 |

On stroke and MI — the two diseases where the patient-graph model beats the classical baselines — it also clearly beats this naive similarity baseline, which is evidence the multi-hop message-passing is contributing something a "find similar patients and copy their outcome" approach can't. On heart failure and PAD, though, the naive baseline actually wins, which is a genuine, disclosed limitation of the "the graph earns its complexity" claim: it's true on the diseases where the graph helps at all, not universally.

**Does timing information on patient-concept edges matter?** The patient-graph model splits each patient-concept connection into three age buckets (recent / mid / old). Removing this distinction entirely (collapsing to one relation type, 3 seeds vs. the main 5-seed run) produces no statistically resolvable difference on any disease (all p > 0.4, Welch's t-test) — point estimates move by less than 0.008 AUROC in either direction for every disease. This is a genuinely inconclusive result, not a finding either way: 3-5 seeds is not enough power to detect a small effect, and it should be read as "we could not confirm timing information matters," not as "timing information doesn't matter."

**A visual summary of training stability** (figure 22) overlays every model's validation-accuracy curve during training on one plot. The three sequence-based models (plain TKG-Transformer, and both concept-graph variants) all peak early and then decline — the pattern behind the `MIN_EPOCHS` fix in Section 6.1. The patient-graph model climbs and then plateaus, never declining across its full training run, for every seed. This qualitative difference is consistent with (though doesn't by itself prove) the patient-graph model learning something more stable rather than something it has to be stopped early to avoid overfitting.

---

## 9. Can we trust the model's explanations?

### 9.1 Two different questions

"Where is the model looking?" (attention weights) is a different question from "does that actually matter to the prediction?" The second question was checked with a tool called GNNExplainer plus a fidelity test: take the top 20% of events the model calls "important," and check whether keeping only those events preserves the prediction better than keeping a random 20%. Separately, check whether removing the top 20% breaks the prediction more than removing a random 20%. A real explanation should pass both checks clearly. If "important" performs the same as "random," the explanation carries no information.

### 9.2 What the too-early model looked like

The first version of the model — the one that stopped training after just 1-2 passes — put most of its attention on things like a routine IV saline flush, given to nearly every hospitalized patient regardless of diagnosis, rather than on actual diagnoses. Only 6-11% of attention went to diagnosis codes; 34-48% went to medications like the saline flush. The fidelity test confirmed this wasn't just an odd-but-valid pattern: the "important" events performed statistically the same as randomly picked ones, on both checks. The explanations carried no real information, even though the raw predictions still looked reasonable.

### 9.3 What changed after requiring more training

| | Before (stopped after 1 round) | After (properly trained, 22-26 rounds) |
|---|---|---|
| Attention on diagnosis codes (heart attack) | ~9% | 66% |
| Top facts for heart attack | Saline flush, routine lab values | Type-2 diabetes, coronary artery disease, high cholesterol, high blood pressure, chronic kidney disease |
| Does "important" beat "random" at preserving the prediction? | No — identical | Yes, clearly and measurably |
| Does removing "important" hurt more than removing "random"? | No — identical | Yes, over twice the damage |

The "before" row is a preserved historical record from when this bug was first found and fixed — the exact undertrained checkpoint it describes can no longer be produced by the current code (the `MIN_EPOCHS` floor prevents it), so it isn't regenerated on every rerun. Everything else in this section reflects the current, properly-trained model.

Where the model's attention goes, by data type, after the fix:

| Disease | Diagnoses | Labs | Procedures | Medications | Blood pressure | BMI | Vitals |
|---|---|---|---|---|---|---|---|
| MI | 66.1% | 4.2% | 10.6% | 11.6% | 1.8% | 3.0% | 0.3% |
| Stroke | 66.0% | 7.4% | 7.0% | 12.0% | 2.0% | 5.1% | 0.3% |
| HF | 72.0% | 4.1% | 8.3% | 11.7% | 1.4% | 2.2% | 0.0% |
| AF | 64.2% | 7.7% | 11.0% | 10.7% | 3.8% | 1.8% | 0.1% |
| PAD | 70.0% | 2.2% | 11.5% | 10.3% | 2.4% | 3.1% | 0.0% |

Diagnosis codes are the dominant, trustworthy signal for every disease.

### 9.4 What's distinctive about each disease, according to the model

Beyond raw attention, a check for which facts are disproportionately linked to one specific disease, rather than being generically common across all five:

| Disease | Cases in test set | Distinctive facts the model relies on |
|---|---|---|
| MI | 137 | Type-2 diabetes, coronary artery disease, hyperlipidemia, essential hypertension, chronic kidney disease |
| Stroke | 107 | Normal BMI readings, hypertension, hyperlipidemia, hyponatremia, long-term aspirin use |
| HF | 112 | Type-2 diabetes, gout, coronary artery disease, hyperlipidemia, hypothyroidism, hypertensive chronic kidney disease, depression, obesity |
| AF | 103 | Essential hypertension, normal/high blood pressure readings, low bicarbonate, specific cardiac procedures, hyperlipidemia, GERD, history of tobacco use |
| PAD | 46 | Tobacco use disorder, hypertension diagnosis — only 2 facts had enough patients behind them to count as reliable, reflecting the small PAD sample |

Coronary artery disease flagging heart attack risk, and showing up again as a warning sign for heart failure, matches the well-known medical progression from heart attack to heart failure. This is the strongest evidence that the model is reasoning sensibly rather than picking up noise.

One caveat: GNNExplainer's own list of top facts per disease (figure 17) isn't filtered for how many patients each fact applies to, so some top entries are one-off quirks from a single patient — stroke's list, for instance, includes a routine IV saline flush at a similar rank to real risk factors, which is exactly the kind of noisy entry Section 9.2 flagged in the undertrained model, just at a much smaller scale now. This doesn't affect the fidelity numbers above, which are the trustworthy part of this check — just the detailed per-fact list in that one figure.

### 9.5 The bottom line

Picking a "final" model based only on validation accuracy can select a version whose stated reasons for its predictions are meaningless, even when the predictions themselves look fine. Requiring a minimum amount of real training fixed this completely — the explanations became genuine and lined up with real medical knowledge — but cost some raw accuracy on certain diseases. Accuracy and trustworthy explanations weren't the same thing here, and optimizing only for accuracy would have quietly shipped a model that explained itself with noise.

### 9.6 Does this hold up for the two graph models?

The same top-20%-vs-random-20% fidelity test (sufficiency: does keeping only the "important" facts preserve the prediction better than a random set of the same size; comprehensiveness: does removing them hurt the prediction more) was extended to both graph attempts.

| Model | Sufficiency (top-20% KL, lower is better) | vs. random | Comprehensiveness (drop-top-20% KL, higher is better) | vs. random |
|---|---|---|---|---|
| Plain TKG-Transformer | 1.293 | 1.440 (passes — top is lower) | 0.545 | 0.207 (passes — top is higher) |
| Concept-graph model (attempt 1, failed on accuracy) | 1.478 | 1.689 (passes) | 1.558 | 0.436 (passes) |
| Patient-graph model (attempt 2, the one that worked) | 0.00037 | 0.00153 (passes) | 0.00243 | 0.00006 (passes) |

All three models pass both directions of the check cleanly. The patient-graph model needed a different method to get here, worth spelling out: it scores every patient in one shared pass over the whole graph rather than one patient at a time, so the usual tool (GNNExplainer's iterative mask search) would need a full-graph forward pass per optimization step per patient — impractical at this graph's size. Instead, `src/ablations/explain_patient_graph_fidelity.py` uses one backward pass per patient to get a gradient-based importance score for each concept that patient is connected to, then tests it the same way: temporarily remove or restrict that one patient's own edges (leaving every other patient's edges untouched) and re-run the shared graph. The raw KL numbers for this model are far smaller than the other two's — expected, since one patient's own edges are a tiny fraction of a 2.7-million-edge shared graph, so removing them moves that patient's own prediction by a smaller absolute amount — but the ratio between "important" and "random" is what the test is actually about, and it's decisive in both directions.

The concept-graph model is a useful negative control: it fails on accuracy (Section 6.2) but still passes both fidelity checks, which is a reminder that these two properties are independent — an explanation can be faithful to what a model is doing even when what the model is doing isn't very good.

---

## 10. Limitations

- **Single hospital system.** All data comes from one Boston hospital; how well this generalizes elsewhere is untested.
- **The clinical-notes version wasn't evaluated**, and has a known, unfixed issue (Section 8.5) for anyone extending this work.
- **No model wins outright.** XGBoost is the strongest model overall, but the patient-graph model (Section 6.3, 8.6) beats it and Cox on specific diseases with real statistical confidence — stroke at 3 years and heart attack at 5 years — while losing just as clearly to XGBoost on heart failure at every horizon and to both on PAD at short horizons. The honest summary is "it depends which disease and which horizon," not a single winner.
- **The mechanism theory in Section 8.7 is a 5-point correlation.** "Risk-factor concentration predicts graph advantage" lines up with the data in the right direction and even clears p < 0.05, but with only 5 diseases that's a hypothesis worth testing on more diseases or datasets, not something this study can treat as proven.
- **A naive similarity baseline beats the patient-graph model on 2 of the 5 diseases.** Section 8.7's k-NN check shows the graph clearly earns its complexity on stroke and MI (the diseases where it also beats the classical baselines), but a simple "find similar patients and copy their outcome" approach actually wins on heart failure and PAD. "The graph does more than pattern-matching" is true only where the graph helps at all, not universally.
- **The timing-information ablation (Section 8.7) is inconclusive, not negative.** Removing all timing information from patient-concept edges moved every disease's AUROC by less than 0.008 with no statistically resolvable difference — but that's from 3 seeds against 5, nowhere near enough power to distinguish "doesn't matter" from "matters a little and we can't see it yet."
- **The patient-graph model works differently under the hood than every other model here, and that's worth understanding, not just noting.** Every other model in this study only ever sees one patient's own data — fully separate from every other patient, before or after training. The patient-graph model is different: patients are nodes in one shared graph, and a patient's final representation is built partly from *other* patients' structural position in that graph (which concepts they connect to), not just their own history in isolation. Test patients are present in this shared graph during training — their own facts (already properly time-limited to before their index date) shape their node's position, but their outcome labels are never used, and no patient gets a free, individually-learned parameter of their own to be tuned by the training signal — only shared settings that apply to every patient equally get adjusted. This is the standard, accepted way this style of graph model is trained, but it is a different assumption from the strict "the model never touches anything related to a test patient until final evaluation" rule that holds for every other model in this study, and that difference should be stated plainly, not glossed over.
- **None of the five models were tuned.** Every hyperparameter (Cox's penalty, XGBoost's tree count/depth/rate, every neural model's size/learning rate) is a fixed, hand-picked value, not the result of a systematic search. The comparison is between these specific configurations, not necessarily the best each model family could achieve.
- **Cox and the graph-based models don't estimate exactly the same thing** (Section 6): Cox and XGBoost treat a competing disease as "censored," while the graph-based models model all five diseases jointly. Both are standard, legitimate approaches, but the difference means part of any gap between them could reflect this setup difference rather than pure model quality.
- **The 5-year results rest on a shrinking test set.** 19.7% of test patients are excluded from the 1-year check (not followed long enough yet), rising to 44.3% at 3 years and 59.6% at 5 years. The 5-year numbers should be trusted less than the 1-year numbers for this reason alone.
- **The TKG-Transformer only reads a patient's most recent 256 events**, discarding 12.1% of all eligible events cohort-wide (2.5% of patients have more history than that). This isn't concentrated in the sickest patients, but no test was run at other sequence lengths, so its effect on the results is unmeasured. The patient-graph model doesn't have this limitation — it uses every eligible event, just collapsed into a most-recent-occurrence-per-concept summary rather than a full sequence.
- **HF and AF are defined strictly on purpose** (main-reason-for-admission only, broad washout), trading away some statistical power for cleaner, leak-free labels — the exact cost is in Section 2.
- **Patients lost to follow-up are simply excluded** from a given time window's scoring rather than statistically reweighted — a standard, disclosed simplification (see the censoring point above for how large this exclusion gets).
- **Labs and vital signs are sampled** (30% / 10%) rather than fully complete, to keep processing manageable.
- **Only 5 random seeds** were used for the robustness check — a reasonable minimum, not a generous number. More would tighten the confidence estimates further, especially for PAD.
- **Cox and XGBoost have less temporal detail available to them than the TKG-Transformer.** They see one whole-period summary per lab (average, highest, lowest, most recent, trend) rather than the exact timing of every event, and missing values are filled with zero rather than explicitly flagged as missing. XGBoost still wins overall despite this — which if anything makes that result more convincing — but it also means the comparison isn't yet perfectly even, and closing this gap is a planned next step, not done.
- **The medical-hierarchy dictionary used in Section 6.2/6.3 is thin for medications.** Only 9.3% of distinct medications resolve to a drug class; diagnoses, procedures, and labs are essentially fully covered. The co-occurrence edges added in the patient-graph model (Section 6.3) partly compensate for this — they don't need a hand-curated dictionary — but a genuinely richer medication ontology (real drug classification data, drug-drug interactions) is a plausible way to improve the patient-graph model further, not yet attempted.

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
- **One exception, disclosed rather than hidden:** the patient-graph model (Section 6.3) trains differently — test patients are structurally present in the shared graph during training (standard for this type of model), though their labels are never used and they receive no individually-tuned parameter of their own. See Section 10 for the full explanation of why this is still leak-safe and how it differs from every other model here.
- The co-occurrence edges used by the patient-graph model (Section 6.3) are computed from training patients only, so no validation or test patient's data shapes the graph's structure itself.
- `src/tests_integrity.py` runs 8 hard checks (split overlap, outcome-mapping consistency, no post-index events, no raw fact at/after the endpoint date, a valid exclusion cascade, no duplicate patients, the train-only vocabulary restriction actually routing test-only concepts to a shared "unknown" slot, and no interval-type fact's value reflecting information past its window close) and stops the pipeline with a non-zero exit code on any violation — unlike `validate_tkg.py` (Section 4), which only prints PASS/FAIL and keeps going. Run this one before spending time training anything.

### A leak we found through a code audit, not through these checks

Every check above was already in place, and none of them caught this, because it wasn't a timestamp or split-overlap problem — it was a feature computed without any time restriction at all. `had_icu_stay`, a static feature fed to every model in this study, was originally computed as "does this patient appear anywhere in the ICU stays table," with no join against the index date. For a patient whose qualifying event *was* an ICU-requiring admission (common for acute MI, stroke, HF, AF, and PAD), that counts an ICU stay *during or after the outcome itself* as if it were a pre-existing risk factor.

We found this through a deliberate code review, not through the automated checks, which is itself worth noting: automated leakage checks are good at catching timestamp and split-boundary mistakes, but a feature that's simply computed from the wrong subset of rows can slip past every one of them, because from the check's point of view no row is "late" — the feature just draws from an unrestricted pool. The fix: restrict `had_icu_stay` to ICU stays with an admission time strictly before the index date. Two related, smaller issues were fixed at the same time: `build_tkg.py` was suppressing the *value* of an ICU stay's length or an IV infusion's total amount only based on when the fact *started*, not when it *ended* — a stay or infusion that started in-window but continued past the window close would carry a value only fully known after the fact, so those values are now nulled out when that happens (affecting 855 ICU-stay and 1,565 IV-input facts out of the graph's 14.5 million, a small population but a real leak). `src/tests_integrity.py` gained a check for exactly that (Check 8).

Every number in this document was recomputed from scratch after these fixes — new cohort, new graph, all baselines and both graph architectures retrained across all 5 seeds. Section 8.1 quantifies how much changed (heart failure the most, stroke and PAD the least), and Section 8.4/8.6 show which statistically significant findings held up, which disappeared, and which appeared for the first time.

---

## 12. How to reproduce this

```bash
PY=/path/to/miniforge3/envs/tkg/bin/python
$PY -u -m src.cohort
$PY -u -m src.build_tkg
$PY -u -m src.prep_modeling
$PY -u -m src.tests_integrity     # hard gate -- stops here (non-zero exit) if any check fails
$PY -u -m src.validate_tkg        # informational, always exits 0
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

To reproduce the two graph-structure attempts (Section 6.2, 6.3 / 8.6):

```bash
$PY -u -m src.ablations.build_ontology        # hand-built medical hierarchy
$PY -u -m src.ablations.build_cooccurrence    # data-driven co-occurrence edges (train-only)

# Attempt 1: concepts enriched, patients still isolated (single seed each; worse than the
# plain model on 29 of 30 disease/horizon/variant cells, so a 5-seed check wasn't run -- see 8.6)
HETERO_VARIANT=full   $PY -u -m src.ablations.hetero_gnn
HETERO_VARIANT=static $PY -u -m src.ablations.hetero_gnn

# Attempt 2: patients as real graph nodes (the one that worked; 65-114 seconds per seed)
for s in 42 43 44 45 46; do
  TKG_SEED=$s $PY -u -m src.ablations.patient_graph_gnn
done

# Same model with all timing information removed (Section 8.7's timing ablation)
for s in 42 43 44; do
  TKG_SEED=$s PATIENT_GNN_TIMING=0 $PY -u -m src.ablations.patient_graph_gnn
done
```

To reproduce the mechanism-analysis and explanation-validity checks (Section 8.7, 9), once everything above has run:

```bash
$PY -u -m src.ablations.mechanism_analysis           # risk-factor concentration vs. graph advantage
$PY -u -m src.ablations.oversmoothing_diagnostic     # did graph attempt 1's embeddings collapse?
$PY -u -m src.ablations.knn_baseline                 # does the patient-graph model beat simple case-matching?
$PY -u -m src.ablations.training_stability_figure    # figure 22: every model's validation curve, overlaid
$PY -u -m src.explain_gnn                            # fidelity check, plain TKG-Transformer
$PY -u -m src.ablations.explain_gnn_concept_graph    # same check, graph attempt 1
$PY -u -m src.ablations.explain_patient_graph_fidelity  # same check, graph attempt 2
```

**Settings you can change:**
- `TKG_USE_NOTES` — `0` (structured data only; used for every result here) or `1` (also uses clinical notes; not evaluated in this work, see Section 8.5).
- `TKG_SEED` — changes only the model's own randomness (weight init, dropout, batch order), not the train/validation/test split, which stays fixed no matter what seed is used. Seed 42 is the default everywhere; seeds 43-46 write to their own separate folders so they never overwrite it.
- `HETERO_VARIANT` — for the first graph attempt only (Section 6.2): `full` (default, keeps the sequence) or `static` (drops sequence and time entirely). Each writes to its own folder.
- `PATIENT_GNN_TIMING` — for the second graph attempt only (Section 6.3): `1` (default, patient-concept edges are split into recent/mid/old) or `0` (all timing information removed, a single relation type). Writes to a separate `_notiming` folder so it never overwrites the main result.

**Other scripts, once the steps above are done:**

| Script | Needs | Produces |
|---|---|---|
| `src.visualize_tkg` | build_tkg | Cohort and graph figures 1-4 |
| `src.visualize_stats` | prep_modeling | `stats/table1_summary.csv`, figures 13-14 |
| `src.make_figures` | baselines + TKG-Transformer | Figures 5, 6, 7, 12 |
| `src.explain` | TKG-Transformer | Attention-based importance, figures 9-10 |
| `src.explain_discriminative` | `src.explain` | Disease-specific fact analysis, figure 11 |
| `src.explain_heatmap` | `src.explain` | Attention heatmaps, figures 15-16 |
| `src.explain_gnn` | TKG-Transformer | Fidelity check, figure 17 |
| `src.multi_seed_summary` | TKG-Transformer + baselines, seeds 42-46 | 5-seed comparison table and figure 20 |
| `src.ablations.mechanism_analysis` | baselines + patient-graph model, all seeds | `stats/mechanism_analysis.csv`, figure 21 |
| `src.ablations.oversmoothing_diagnostic` | `hetero_gnn_survival` checkpoint | `stats/oversmoothing_diagnostic.csv`, figure 23 |
| `src.ablations.knn_baseline` | prep_modeling | `knn_baseline/test_metrics.csv` |
| `src.ablations.training_stability_figure` | all four architectures trained | figure 22 |
| `src.ablations.explain_gnn_concept_graph` | `hetero_gnn_survival` checkpoint | `explain/gnn_explainer_fidelity_concept_graph.csv` |
| `src.ablations.explain_patient_graph_fidelity` | `patient_gnn_survival` checkpoint | `explain/patient_graph_fidelity.csv` |

---

## 13. Repository layout

Every script below can be run on its own with `python -u -m src.<name>` (or `src.ablations.<name>`), as long as the files it needs already exist. Section 12 gives the full run order.

```
TKG_MIMIC/
|-- main.py                       # runs everything in order
|-- README.md                     # this file
|-- mimic_data/                   # raw MIMIC-IV files (not included; gitignored)
`-- tkg_output/                   # everything the pipeline produces (gitignored)
    |-- cohort.csv                       # the 33,656-patient cohort (Section 2)
    |-- cohort_cascade.csv               # exact patient counts at each filtering step
    |-- tkg_facts.csv                    # the 14.5M-fact knowledge graph (Section 3)
    |-- node_index.csv, node_index_v2.csv  # concept ID <-> row-index lookup tables
    |-- validation_report.txt            # output of validate_tkg.py
    |-- ontology_edges.csv               # hand-built medical hierarchy edges
    |-- cooccurrence_edges.csv           # data-driven co-occurrence edges (train-only)
    |-- figures/                         # fig1 through fig23
    |-- modeling/                        # train/val/test split, features, event table
    |-- explain/                         # every explainability/fidelity-check output
    |-- stats/                           # confidence intervals, significance tests, mechanism analysis
    |-- knn_baseline/                    # the k-nearest-neighbor baseline's results
    |-- baselines_survival/              # Cox + XGBoost, seed 42 (main results)
    |-- baselines_survival_seed{43..46}/
    |-- tgn_survival/                    # TKG-Transformer, seed 42 (main results)
    |-- tgn_survival_seed{43..46}/
    |-- hetero_gnn_survival/             # graph attempt 1, "full" variant
    |-- static_gnn_survival/             # graph attempt 1, "static" variant
    |-- patient_gnn_survival/            # graph attempt 2, seed 42 (main results)
    |-- patient_gnn_survival_seed{43..46}/
    |-- patient_gnn_survival_notiming*/  # graph attempt 2 with timing information turned off (Section 8.7)
    `-- survival_comparison_test.csv     # the final head-to-head results table
```

```
src/
|-- config.py                 # settings, medical code lists, the washout safety check, shared file-loading helper
|-- cohort.py                 # builds the patient cohort (Section 2)
|-- build_tkg.py              # builds the knowledge graph from raw MIMIC-IV files (Section 3)
|-- validate_tkg.py           # the 8 sanity checks (Section 4) -- prints PASS/FAIL, never stops the pipeline
|-- tests_integrity.py        # a stricter version of the above: 8 checks that actually HALT the pipeline
|                              #   (non-zero exit code) on any violation -- run this, not validate_tkg.py,
|                              #   before spending time training anything
|-- prep_modeling.py          # cuts the graph down to the 5-year pre-index window, builds the train/70%
|                              #   / val/15% / test/15% split, and writes the per-event feature table every
|                              #   model reads from
|-- baseline.py               # shared feature-building code: turns a patient's events into either a
|                              #   "bag of codes" (which concepts they have) or a value-summary table
|                              #   (mean/max/min/last/count/trend per lab), both restricted to concepts
|                              #   seen in training patients only
|-- baselines_survival.py     # fits Cox regression and XGBoost, the two standard-approach models
|-- tgn_model.py              # the TKG-Transformer architecture itself (a sequence Transformer, not a
|                              #   graph neural network -- see Section 6.1)
|-- tgn_survival.py           # trains the TKG-Transformer with the competing-risks (DeepHit) prediction
|                              #   head; also defines the shared CAUSES/HORIZON_DAYS/MIN_EPOCHS settings
|                              #   and the competing-risks AUROC scoring function used everywhere else
|-- compare_survival.py       # builds the main results table (Section 8.1) from already-saved predictions
|-- evaluate_stats.py         # confidence intervals (bootstrap) and significance tests (DeLong's) on the
|                              #   main results, recomputed independently so they're guaranteed to use the
|                              #   same patient definitions as compare_survival.py
|-- multi_seed_summary.py     # the 5-seed comparison and Bonferroni-corrected significance testing (Section 8.4)
|-- notes_extract.py          # pulls each cohort patient's discharge notes from MIMIC-IV-Note
|-- notes_embed.py            # turns each note into a 768-number vector with a clinical language model
|-- notes_aggregate.py        # averages a patient's pre-index note vectors into one summary vector
|-- visualize_tkg.py          # cohort flow diagram and graph-composition figures (figures 1-4)
|-- visualize_stats.py        # descriptive statistics table and figures (figures 13-14)
|-- make_figures.py           # main results figures (5, 6, 7, 12)
|-- explain.py                # which facts the TKG-Transformer's attention focused on, per disease
|-- explain_discriminative.py # which facts are distinctively linked to one disease rather than generic (figure 11)
|-- explain_heatmap.py        # attention heatmaps by concept / data type / time (figures 15-16)
|-- explain_gnn.py            # the explanation-fidelity check (Section 9): are the TKG-Transformer's
|                              #   "important" facts actually more informative than random facts?
`-- ablations/
    |-- build_ontology.py                    # hand-built medical hierarchy (ICD chapter/category, drug
    |                                         #   class, lab category) used by both graph attempts
    |-- build_cooccurrence.py                # data-driven concept-concept co-occurrence edges, computed
    |                                         #   from training patients only
    |-- hetero_gnn.py                        # graph attempt 1: enrich concepts with the hierarchy, patients
    |                                         #   still isolated (Section 6.2) -- didn't help
    |-- patient_graph_gnn.py                 # graph attempt 2: patients as real graph nodes (Section 6.3)
    |                                         #   -- this is the one that worked
    |-- mechanism_analysis.py                # tests whether a disease's risk factors being spread out
    |                                         #   across many facts (vs. concentrated in a few) predicts
    |                                         #   whether the patient-graph model helps or hurts (Section 8.7)
    |-- oversmoothing_diagnostic.py           # checks whether graph attempt 1's concept embeddings collapsed
    |                                         #   into fewer effective directions after message-passing, and
    |                                         #   whether that collapse is worse near well-connected concepts
    |                                         #   or poorly-connected ones (Section 8.7)
    |-- knn_baseline.py                      # a plain "find similar training patients and copy their
    |                                         #   outcome rate" baseline -- checks whether the patient-graph
    |                                         #   model does more than simple case-based reasoning (Section 8.7)
    |-- training_stability_figure.py         # overlays every model's validation-accuracy curve during
    |                                         #   training on one plot (figure 22)
    |-- explain_gnn_concept_graph.py          # the same fidelity check as explain_gnn.py, run on graph
    |                                         #   attempt 1 instead, to see if the same checkpoint-selection
    |                                         #   problem (Section 9) shows up there too
    `-- explain_patient_graph_fidelity.py     # the same fidelity check, adapted for the patient-graph model
                                              #   -- since that model scores every patient in one shared pass
                                              #   instead of one patient at a time, this uses a one-pass
                                              #   gradient-saliency score plus literal edge removal instead
                                              #   of the GNNExplainer tool the other two scripts use
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

**Patient-graph model specifics (Section 6.3):**

| Setting | Value |
|---|---|
| Time buckets for patient-concept edges | recent (≤90 days before index), mid (90-730 days), old (>730 days) |
| Minimum co-occurrence to create a concept-concept edge | 30 training patients |
| Max co-occurrence edges kept per concept | 15 (strongest by count) |
| Rounds of message-passing | 2 (a patient's prediction can be shaped by a different patient's data 2 hops away) |
| Training style | full-batch — one pass over the whole graph per round, not per mini-batch |
| Training time | 65-114 seconds across the 5 seeds (22-30 rounds, CPU) |
| Timing-information ablation (Section 8.7) | 3 seeds, all timing buckets collapsed to one relation |

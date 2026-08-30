# Findings

Consolidated write-up of the study. Stage 8, documentation only: Amendment A7
dropped the serving endpoint, so nothing here is computed. Every figure below is
read from a file in `reports/` and matches it exactly. Sources are named per
section.

---

## 1. What was asked and what was found

**Q1 (prediction).** *Using only information available by day D of a course
presentation, how well can we identify students who will finish with Fail or
Withdrawn, on a presentation the model has never seen?*

Well enough to rank, not well enough to matter on the metric the protocol
pre-registered as primary. On the held-out 2014J presentation at day 28,
LightGBM reaches AUC-PR 0.6794 [0.6674, 0.6919] against a base rate of 0.4079,
and at a 5% alert budget recovers 0.1144 of non-completers against a ceiling of
0.1226 — about 93% of what that budget could possibly catch. But its expected
cost at the pre-committed 10:1 ratio is 0.5871 against 0.5921 for flagging every
student in the cohort, an improvement of 0.84%, and the regularised logistic
model at its frozen threshold (0.5922) is marginally worse than flagging
everyone. The model discriminates; at this base rate the cost metric cannot see
it.

**Q2 (decision).** *Does the event we would trigger on actually change the
outcome?*

Unanswered, and the reason is recorded rather than worked around. The
regression discontinuity at the assessment pass mark failed its pre-committed
validity check: both density tests reject a continuous density at 40, with
excess mass just above the mark (jackknife t = 2.4704, p = 0.0135; McCrary
z = +6.7811). Section 12 declared in advance that this invalidates the design,
so Arm 2 reduces to the power analysis. The design's minimum detectable effect
was 21.7 percentage points on robust inference, so it could not have found a
plausible intervention effect even had it been valid.

---

## 2. Results table

Test split (2014J), D = 28, all five ladder rungs. Source:
`reports/stage5_holdout.txt` section 2. Intervals are 95% stratified bootstrap,
1,000 resamples. Recall ceilings are budget / test base rate and depend only on
the cutoff, not the model.

| model | thr | expected cost | [95% CI] | AUC-PR | [95% CI] | Brier | AUC-ROC* |
|---|---|---|---|---|---|---|---|
| B0 | 0.01 | 0.5921 | [0.5921, 0.5921] | 0.4079 | [0.4079, 0.4079] | 0.2441 | 0.5000 |
| B1 | 0.01 | 0.5921 | [0.5921, 0.5921] | 0.5256 | [0.5135, 0.5385] | 0.2338 | 0.6221 |
| B2 | 0.01 | 0.5921 | [0.5921, 0.5921] | 0.5267 | [0.5136, 0.5412] | 0.2322 | 0.6253 |
| B3 | 0.07 | 0.5922 | [0.5837, 0.6020] | 0.6106 | [0.5970, 0.6250] | 0.2153 | 0.7291 |
| M1 | 0.14 | 0.5871 | [0.5789, 0.5967] | 0.6794 | [0.6674, 0.6919] | 0.2108 | 0.7498 |

| model | r@5% | [95% CI] | ceil@5% | r@10% | [95% CI] | ceil@10% | r@20% | [95% CI] | ceil@20% |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 0.0522 | [0.0444, 0.0551] | 0.1226 | 0.1022 | [0.0926, 0.1075] | 0.2452 | 0.1919 | [0.1903, 0.2100] | 0.4903 |
| B1 | 0.0878 | [0.0830, 0.0931] | 0.1226 | 0.1511 | [0.1429, 0.1583] | 0.2452 | 0.2869 | [0.2767, 0.2967] | 0.4903 |
| B2 | 0.0774 | [0.0726, 0.0830] | 0.1226 | 0.1453 | [0.1384, 0.1533] | 0.2452 | 0.2842 | [0.2733, 0.2940] | 0.4903 |
| B3 | 0.0830 | [0.0777, 0.0878] | 0.1226 | 0.1551 | [0.1480, 0.1623] | 0.2452 | 0.3260 | [0.3156, 0.3361] | 0.4903 |
| M1 | 0.1144 | [0.1112, 0.1168] | 0.1226 | 0.1985 | [0.1921, 0.2049] | 0.2452 | 0.3324 | [0.3233, 0.3425] | 0.4903 |

\* AUC-ROC is reported for completeness only. The temporal split (Section 4)
makes it non-comparable to published OULAD figures that use random splits.

B0, B1 and B2 all sit at the flag-everyone cost of 0.5921: their frozen
threshold of 0.01 is below every probability they emit, so they alert on the
whole cohort. That is not a bug in the ladder, it is the cost metric's behaviour
at this base rate, and it is the same behaviour that flattens the gap between B3
and M1.

Paired M1 − B3 expected cost difference at D = 28: **−0.0051, 95% CI
[−0.0167, 0.0056]** (source: `stage5_holdout.txt` section 4; negative means M1
is cheaper). The interval contains zero.

---

## 3. Accuracy versus timeliness

Source: `reports/stage5_holdout.txt` sections 2 and 8.

The cost-selected model **differs by cutoff** — B3 at D = 14, M1 at D = 28 and
D = 56 — because selection was made on validate expected cost and frozen before
the holdout opened. Comparing the selected models across cutoffs therefore
conflates a model swap with a timeliness cost. Both comparisons are given.

**Cost-selected model at each cutoff:**

| | D = 14 | D = 28 | D = 56 |
|---|---|---|---|
| selected model | B3 | M1 | M1 |
| frozen threshold | 0.09 | 0.14 | 0.13 |
| test base rate | 0.4165 | 0.4079 | 0.3854 |
| expected cost | 0.5790 | 0.5871 | 0.5856 |
| flag-everyone cost | 0.5835 | 0.5921 | 0.6146 |
| AUC-PR | 0.6363 | 0.6794 | 0.7299 |
| Brier | 0.2104 | 0.2108 | 0.1869 |
| AUC-ROC | 0.7256 | 0.7498 | 0.7975 |
| recall@10% (ceiling) | 0.1785 (0.2401) | 0.1985 (0.2452) | 0.2300 (0.2594) |

**Held fixed, D = 14 against D = 28** (change is D = 14 relative to D = 28;
lower is better for cost and Brier, higher for AUC-PR and recall):

| basis | metric | D=14 | D=28 | change |
|---|---|---|---|---|
| selected (B3 / M1) | expected cost | 0.5790 | 0.5871 | −1.4% |
| selected (B3 / M1) | AUC-PR | 0.6363 | 0.6794 | −6.3% |
| selected (B3 / M1) | Brier | 0.2104 | 0.2108 | −0.2% |
| selected (B3 / M1) | recall@10% | 0.1785 | 0.1985 | −10.1% |
| B3 at both cutoffs | expected cost | 0.5790 | 0.5922 | −2.2% |
| B3 at both cutoffs | AUC-PR | 0.6363 | 0.6106 | **+4.2%** |
| B3 at both cutoffs | Brier | 0.2104 | 0.2153 | −2.2% |
| B3 at both cutoffs | recall@10% | 0.1785 | 0.1551 | **+15.1%** |
| M1 at both cutoffs | expected cost | 0.5839 | 0.5871 | −0.5% |
| M1 at both cutoffs | AUC-PR | 0.6467 | 0.6794 | −4.8% |
| M1 at both cutoffs | Brier | 0.2133 | 0.2108 | +1.2% |
| M1 at both cutoffs | recall@10% | 0.1811 | 0.1985 | −8.8% |

Holding the model fixed changes the sign for B3: the logistic model is *better*
at day 14 than at day 28 on both AUC-PR and recall@10%, while M1 is worse at day
14 on both. The pooled comparison in the first pass of the Stage 5 report
reported a single "6.3% AUC-PR drop" and attributed all of it to timeliness. Two
weeks of extra data helps the gradient-boosted model and does not help the
linear one, and the pooled figure cannot show that.

No verdict is drawn from any of these numbers; see O3/O4 below.

---

## 4. Declared outcomes (Section 13)

Sources: `reports/stage5_holdout.txt` section 8, `reports/stage7_rdd.txt`
section 10.

| outcome | verdict | evidence |
|---|---|---|
| **O1** LightGBM materially beats logistic on cost at D=28, interval-separated | **Not met** | Paired M1 − B3 = −0.0051, 95% CI [−0.0167, 0.0056] contains zero. |
| **O2** LightGBM does not beat logistic by an interval-separated margin | **Met** | Same interval. Gradient boosting is not shown to be warranted on the primary metric. |
| **O3** Day 14 is close to day 28; early action costs little | **Undetermined** | Cost −1.4%, AUC-PR −6.3%, recall@10% −10.1% on the selected models; see section 3 for the held-fixed figures. |
| **O4** Day 14 is substantially worse; the timeliness cost is real | **Undetermined** | Same figures. |
| **O5** Performance degrades sharply on the test presentation | **Undetermined** | Two readings: cost 0.5047 → 0.5871 (+16.3%) while flag-everyone rose 0.5156 → 0.5921 over the same move; separately M1 AUC-PR 0.7870 → 0.6794 (−13.7%), B3 0.7843 → 0.6106 (−22.1%), M1 Brier 0.1883 → 0.2108 (+12.0%). |
| **O6** Slice reporting shows materially worse performance for one or more groups | **Undetermined** | Worst slice `imd_band = __NULL__` (n=366) at cost 0.6667, +13.6% against the overall 0.5871; best `disability = Y` (n=855) at 0.4702, −19.9%. All 18 slice gaps are tabulated. |
| **O7** A significant discontinuity at the pass mark | **Not met** | The design is invalid (O9). For the record and not as a finding, the primary robust estimate is −0.1284, 95% CI [−0.2803, +0.0234], p = 0.0973, which contains zero. |
| **O8** No detectable discontinuity, reported with the power analysis | **Not met (not reached)** | O9 supersedes it. A null drawn from a design whose identifying assumption fails is not the finding O8 declared. The power analysis it called for is reported regardless. |
| **O9** V1 fails and the design is invalid | **Met** | Manipulation test rejects a continuous density at 40: jackknife t = 2.4704, p = 0.0135, with density 0.001694 left and 0.002557 right. McCrary as originally specified: θ = +0.9516, z = +6.7811, p < 0.0001. |

### Why four outcomes are undetermined

Section 13 declared O3, O4, O5 and O6 in the language of "close",
"substantially worse", "sharp" degradation and "materially worse" performance.
Neither Section 13 nor any other part of the protocol defines what would count
as meeting them: no threshold, no metric on which to read one, no comparison
rule. Unlike D1's PSI bands in Section 10, nothing was pre-registered.

The first pass of the Stage 5 report supplied conventions anyway — a 10%
relative AUC-PR drop for O3/O4, and a 15% relative expected-cost degradation
used twice, once for O5 and once for O6 — and returned verdicts against them.
Each was chosen after the results were known, which makes it a cut-off selected
with knowledge of what it would decide, and each landed on the convenient side
of the number it was judging. **All three are withdrawn.** The underlying
figures are printed and no verdict is derived from them.

O1/O2 survives because its criterion — whether a paired bootstrap interval
contains zero — is one the protocol actually specified. O7, O8 and O9 survive
for the same reason: O7 and O8 turn on statistical significance, which Section
12 fixed by specifying robust bias-corrected inference, and O9 turns on whether
V1 fails, which V1 is a test of.

Two of the nine outcomes are decided as the protocol wrote them for Arm 1, and
three for Arm 2. Pre-registering an outcome does nothing unless the criterion
that decides it is pre-registered too. That is a finding about the protocol, not
about the model.

---

## 5. Amendment log

Source: `docs/PROTOCOL.md`, Amendments section. The "found by" column reports
what the protocol records; where it records nothing, that is stated rather than
inferred.

| # | Amends | What changed | Why | Stage found | Found by |
|---|---|---|---|---|---|
| **A1** | §6 Group E | All five Group E assessment features computed over `is_banked = 0` rows only | A banked score is carried from a previous presentation and is not evidence of engagement in the window being measured | Stage 1 validation | Manual. The specification was written from the published table schema without inspecting the column set. |
| **A2** | §4 | Students recurring across the split are retained, but the overlap between the test split and training presentations must be reported as a competing explanation | 32,593 rows over 28,785 distinct students, so a student can appear on both sides of the split. Not leakage, but per-student memorisation becomes an alternative explanation for test performance | Stage 1 validation | Not stated in the protocol. The counts come from the Stage 1 report; the implication was drawn from them. A2's reporting commitment went unmet until Stage 8; it is discharged in `reports/stage8_a2_overlap.txt`. |
| **A3** | §6 Group C | Engagement slope fitted by OLS over daily click counts from day 0 to D−1, not weekly | At D = 14 the weekly version is a difference between two points, not a regression, and is not comparable to D = 56 where it fits over eight | Before Stage 2 code | Manual. "A specification defect, found on reading the protocol back before implementation rather than from any data." |
| **A4** | §2 | Provenance recorded as the UCI mirror (dataset 349, DOI 10.24432/C5KK69) rather than the OU download page | The OU page was non-functional at time of ingest. Originating publication unchanged | Stage 1 ingest | Not stated in the protocol. Operational, discovered while fetching the data. |
| **A5** | §6 Groups B, C, D | The observation window has no lower bound: it is "everything known up to day D", not "the first D days" | `student_vle` contains activity from day −25, so five features include pre-start engagement. Retained rather than corrected: it is known at prediction time and breaches no leakage rule | Stage 3, Verification Stop 2 | **Manual.** "Found at Verification Stop 2 by manually tracing a sampled student against the raw table, not by any automated check." |
| **A6** | §6 Group F | `module_presentation` replaced by `code_module` | Under the temporal split the feature is degenerate — no training value ever recurs at validate or test — yet its dummies carried the two largest B3 coefficients (0.79 and −0.73) and shifted the model's effective intercept, and therefore its calibration | Stage 4 | **Manual.** "Found by reading the fitted coefficients in the Stage 4 report, not by any automated check." |
| **A7** | §11, §15 | E2, E3 and E4 dropped; the serving endpoint dropped | Scope reduction decided after Stage 5. None of the dropped items bears on a declared outcome, and drift analysis became the higher priority once Stage 5 found discrimination degrading | After Stage 5, recorded at Stage 6 | A decision, not a defect. No discovery mechanism applies. |

**No amendment was found by an automated check.** A5 and A6 say so explicitly;
A1 and A3 are specification defects that no assertion would have raised; A2 and
A4 record facts about the data and its provenance. The two manual verification
stops the protocol built in are what caught A5. An automated pipeline that
passed every assertion it contained would have shipped all six defects.

---

## 6. Limitations

### Known in advance (Section 14)

- **L1.** Population-target mismatch between the two arms, per Section 12.
  Quantified below.
- **L2.** Single institution, single provider, 2013 to 2014, distance learning.
  Generalisation to other settings is not claimed.
- **L3.** Clicks are a proxy for engagement. A student reading a printed handout
  registers as inactive.
- **L4.** Four presentations means four temporal units, and one test
  presentation is one draw. Conclusions about drift rest on a small number of
  periods.
- **L5.** No intervention exists, so the operating threshold is evaluated
  against an assumed cost ratio rather than a measured one.
- **L6.** Modules differ in structure, assessment weight and duration. Pooling
  them is a simplification; per-module reporting is the partial mitigation.
- **L7.** `imd_band` is a UK deprivation measure with no direct analogue
  elsewhere, which limits transfer of the fairness findings.

### Found during the study

**L8. The 10:1 cost ratio is inappropriate at this base rate, and it determined
the conclusion.**
Source: `reports/stage5_interpretation.md`, `reports/stage5_holdout.txt`
section 6.

Expected cost is (10 × FN + FP) / n. At a test base rate of 0.4079, missing a
student is punished ten times harder than a false alarm, so the cost-optimal
policy converges towards alerting on everyone and stops distinguishing between
models. Flag-everyone costs 0.5921; M1 costs 0.5871, an improvement of 0.84%;
B3 at its frozen threshold costs 0.5922, marginally worse than the constant
policy. This is a property of the cost assumption interacting with the base rate,
not of the models — on every secondary metric the two separate cleanly.

The pre-committed sweep from 2:1 to 20:1 shows where the frozen policy breaks:
at D = 28 the 10:1-optimal threshold beats flag-everyone by 3.04% at a ratio of
2, by 0.84% at 10, by 0.02% at 13, and **from a ratio of 14 upward it is worse
than flagging everyone** (−0.26% at 14, −1.91% at 20). The counterfactual
test-optimal threshold, which is not available at deployment time, never loses to
the trivial policy anywhere in the swept range (+0.40% at a ratio of 20).

The ratio was chosen by analogy to credit and fraud problems, where base rates
run 1–5%. At 41% it is the wrong ratio. It was fixed before any data was
examined, Section 9 stated it was asserted rather than measured, and it is not
revised after the fact.

**L9. The population gap between the two arms is larger than the caveat
implied.**
Source: `reports/stage7_rdd.txt` sections 1 and 9.

Arm 2 covers students who sat and were marked on the first TMA: 23,916
student-module-presentation rows over 22,261 distinct students, non-completion
rate 0.3825. Arm 1 at D = 28 covers 27,530 students across all splits.

| group | n | share | not_completed |
|---|---|---|---|
| Arm 1 cohort at D = 28 (all splits) | 27,530 | 1.0000 | 0.4414 |
| — also in the Arm 2 population | 23,637 | 0.8586 | 0.3752 |
| — **not** in the Arm 2 population | 3,893 | 0.1414 | **0.8433** |

A further 279 Arm 2 rows are absent from the Arm 1 D = 28 cohort, removed there
by E1, E2 or E3 but retained here because Arm 2 applies no such exclusion.

The students outside Arm 2 fail or withdraw at 0.8433 against 0.3752 for those
inside it. The population the causal design can speak to is the one that already
engaged enough to be marked; the students an early warning system exists to
reach are, by construction, the ones it cannot see. Section 12 predicted a
mismatch in the abstract. It is not resolved and is not resolvable within this
design.

**A third observation, recorded as a limitation of the monitoring rather than of
the study.**
Source: `reports/stage6_drift_explainability.txt` sections 2 and 3.

Discrimination fell between presentations, and Population Stability Index — the
monitor a production team would most likely be running — would have shown almost
nothing. The overlap between the top five features by PSI and the top five by
SHAP importance is **one of five** (`mean_submission_lateness`); at ten it is
five of ten. The model's most important feature, `n_due_not_submitted`, sits at
PSI rank 23 with a PSI of 0.0086, among the most stable things measured, and its
third most important, `clicks_percentile`, has the second-lowest PSI in the
study at 0.0002. PSI measures each feature's marginal distribution; what costs
accuracy is the conditional relationship moving, and PSI is blind to that. No
causal claim is made about the Stage 5 degradation; the overlap is reported as
an observation.

The highest PSI in the study, `code_module` at 1.7967, is not trustworthy as a
magnitude: module CCC has zero rows in the train split and 21.9% of the test
split, so the term exists only because of the constant used for empty bins. The
same feature reads 1.2872 at 1e-3 and 2.8072 at 1e-6. The compositional change
is real; the number is a property of the constant.

---

## 7. Reproducing the study

Environment:

```
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run order. Each stage is verified before the next begins; several stages split
computation from reporting, and both halves must run.

| # | Command | Produces |
|---|---|---|
| 1 | `src/stage1_ingest.py <path to oulad.zip>` | `data/CHECKSUMS.txt`, `data/oulad.duckdb` with the seven raw tables |
| 2 | `src/stage1_validate.py` | `reports/stage1_validation.txt` |
| 3 | `src/stage2_cohort.py` | `v_*` normalisation views and `cohort_d{14,28,56}` in DuckDB |
| 4 | `src/stage2_report.py` | `reports/stage2_cohort.txt` — **Verification Stop 1** |
| 5 | `src/stage3_features.py` | `features_d{14,28,56}` in DuckDB |
| 6 | `src/stage3_report.py` | `reports/stage3_features.txt`, `reports/stage3_sample_d28.csv` — **Verification Stop 2** |
| 7 | `src/stage4_ladder.py` | `models/stage4_refit_d{14,28,56}.joblib`, `reports/stage4_results.joblib`, `reports/frozen_threshold.json` |
| 8 | `src/stage4_report.py` | `reports/stage4_ladder.txt` |
| 9 | `src/stage5_holdout.py` | `reports/stage5_holdout.txt`, `reports/stage5_test_predictions.parquet` |
| 10 | `src/stage6_report.py` | `reports/stage6_drift_explainability.txt` |
| 11 | `src/stage7_report.py` | `reports/stage7_rdd.txt` |
| 12 | `src/stage8_a2_overlap.py` | `reports/stage8_a2_overlap.txt` — discharges Amendment A2 |

Stages 2, 3 and 4 split computation from reporting, and both halves must run.
`src/stage7_rdd.py`, `src/stage6_drift.py`, `src/stage4_models.py`,
`src/stage4_preprocess.py`, `src/stage4_guard.py`, `src/stage5_metrics.py` and
`src/stage2_views.py` are modules imported by the runners above, not entry
points.

### Immutable once created

- **`docs/PROTOCOL.md` above the Amendments section.** No section may be revised
  after data has been examined. Corrections are recorded as new amendments with
  the original wording intact.
- **`reports/frozen_threshold.json`.** Written at Stage 4 with the git commit it
  was generated at. Stage 5 verifies the loaded models' hyperparameters against
  it and refuses to run on a mismatch.
- **`models/stage4_refit_d{D}.joblib`.** The refit artefacts scored at Stage 5.
- **`reports/stage5_holdout.txt` and `reports/stage5_test_predictions.parquet`.**
  The holdout opens once. Stages 6 and 7 read the stored predictions and do not
  rescore; Stage 6 verifies that the model's own contributions reproduce the
  stored probabilities before reporting anything from them.

Stage 4 enforces the split rule in code: every data load goes through
`stage4_guard.load_split()`, which raises if `test` is requested. Stage 5
contains the only function in the project permitted to read it.

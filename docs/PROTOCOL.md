# Pre-Registered Protocol

## Early Warning for Course Non-Completion (OULAD)

**Author:** Arshin Sikka
**Status:** Committed as the repository root commit, before any code or data exists.

This document fixes every analytical decision in advance. It is committed first
so that the git history itself demonstrates the method preceded the analysis.
No section of this document may be revised after data has been examined. If a
decision here turns out to be wrong, the correction is recorded as a new commit
with the original wording intact and the reason for the change stated.

---

## 1. Question and Scope

**Primary question**

Can a student who will not complete a course be identified early enough, and
reliably enough, for an intervention to be worth running?

This decomposes into two questions that require different methods.

**Q1 (prediction).** Using only information available by day D of a course
presentation, how well can we identify students who will finish with Fail or
Withdrawn, on a presentation the model has never seen?

**Q2 (decision).** Does the event we would trigger on actually change the
outcome? A model that ranks students by risk says nothing about whether acting
on that ranking helps. Q2 is answered by a separate design, not by the model in
Q1.

**Scope boundary**

This is an offline study on a public dataset. No system is deployed and no
intervention is run. Every claim about operational value is conditional and
labelled as such.

---

## 2. Data and Provenance

Open University Learning Analytics Dataset (OULAD). Kuzilek, Hlosta and
Zdrahal, *Scientific Data*, 2017. Licensed CC-BY 4.0.

Seven tables: `studentInfo`, `studentRegistration`, `courses`, `assessments`,
`studentAssessment`, `vle`, `studentVle`.

Approximately 32,000 student-module-presentation records across 7 modules and 4
presentations (2013B, 2013J, 2014B, 2014J), where B denotes a February start
and J an October start.

All dates are integers relative to day 0 of the presentation. Negative values
precede the start.

Version pinned by SHA-256 of each downloaded CSV, recorded in
`data/CHECKSUMS.txt` at first ingest. Any later mismatch is a hard failure
rather than a warning.

Loaded into DuckDB. All aggregation and feature construction happens in SQL
against the relational tables. pandas is used for modelling input only. This is
a deliberate choice: the joins are the work, and doing them in SQL is both
honest to how this would be built and directly relevant to the skill the work
is meant to demonstrate.

---

## 3. Target Definition and Study Population

**Target**

```
not_completed = 1  where final_result is Fail or Withdrawn
not_completed = 0  where final_result is Pass or Distinction
```

*Rationale.* From the point of view of an early warning system, failure and
withdrawal are the same outcome: the student did not get the qualification. The
intervention triggered is the same in both cases.

*Considered and rejected:* a three-class target separating Withdrawn from Fail.
It complicates the cost model without changing the decision, since the action
taken at day 28 is identical.

**Population exclusions, applied per cutoff D**

- **E1.** Students with `date_unregistration < D` are excluded. They have
  already left. Their outcome is Withdrawn and it is recorded in the data
  before the prediction is made, so including them makes the model trivially
  accurate on a subgroup where no prediction was required. This is the single
  largest leakage trap in OULAD and it silently inflates every published
  headline figure that ignores it.
- **E2.** Students with `date_registration > D` are excluded, since they had not
  registered at the point of prediction.
- **E3.** Students with a null registration record are excluded.

**Explicitly NOT excluded**

Students who registered before D but recorded zero VLE activity before D are
**retained**, with zero-valued activity features. They are the highest-risk
group in the dataset. Dropping them would be selection on the outcome and would
remove exactly the students the system exists to find. Any analysis that
quietly requires at least one click has done this.

**Reported before any modelling**

Row counts and base rate after each exclusion, per presentation, per cutoff. If
E1 removes a materially different share of students in the test presentation
than in training, that is drift and it is reported under Section 10 rather than
smoothed over.

> **VERIFICATION STOP 1 (manual).** Confirm by hand, on a sample of at least 20
> retained students, that no row in the feature table has
> `date_unregistration` populated with a value earlier than the cutoff. This
> does not throw an error when it is wrong.

---

## 4. Temporal Split

Four presentations exist: 2013B, 2013J, 2014B, 2014J.

```
Train       2013B + 2013J
Validate    2014B
Test        2014J
```

The test presentation is the chronologically last one and is opened exactly
once, at Stage 5. Every model choice, hyperparameter, feature decision and
threshold is made on train and validation only.

*Why not a random split.* A random split lets the model learn from students in
the same presentation as the ones it is scored on, which shares cohort-level
effects (same materials, same tutors, same calendar) that would not be
available at deployment. Published OULAD results using random splits are not
comparable to this one and will not be cited as benchmarks.

**Module coverage caveat, resolved in advance.** Not all seven modules run in
all four presentations. Modules present in the test presentation but with fewer
than 200 training rows are reported separately rather than dropped, because
dropping them would hide exactly the generalisation problem a deployment would
hit when a new module launches.

**Refit rule.** After hyperparameters are fixed on validation, the final model
is refit on train + validation combined, then scored once on test. Declared
here so the refit cannot look like a post-hoc improvement.

---

## 5. Prediction Cutoffs and the Leakage Boundary

Cutoffs **D = 14, 28, 56** days from presentation start. Day 28 is the
headline. All three are reported.

*Rationale for reporting three.* The real design question is not "how accurate
is the model" but "how much accuracy do you give up to act a fortnight
earlier". That trade-off curve is the artefact. Picking a single cutoff because
it scored best would be choosing the question to fit the answer.

**THE LEAKAGE BOUNDARY.** A feature may enter the model at cutoff D only if its
value was determinable from data timestamped strictly before day D.

- **L1.** `studentVle` rows filtered to `date < D`.
- **L2.** `studentAssessment` rows filtered to `date_submitted < D`. Filtering
  on the assessment due date instead of the submission date leaks, because a
  due date before D says nothing about whether the student had submitted.
- **L3.** Assessments of type `Exam` are excluded from features at every
  cutoff. They occur at presentation end.
- **L4.** `date_unregistration` is never a feature. It is used only for the E1
  exclusion in Section 3.
- **L5.** `final_result` is never a feature and never a component of one.
- **L6.** No feature may be constructed using cohort statistics computed across
  the full presentation, since those aggregate information from after D.
  Cohort-relative features (Section 6, Group F) use only activity before D.
- **L7.** `studentInfo` fields are known at registration and are permitted:
  `gender`, `region`, `highest_education`, `imd_band`, `age_band`,
  `num_of_prev_attempts`, `studied_credits`, `disability`.

> **VERIFICATION STOP 2 (manual).** For each of the three cutoffs, take five
> students at random from the feature table and trace at least two engineered
> features back to the source rows by hand, confirming every contributing row
> is timestamped before D. Do this by reading the source rows, not by running
> an assertion written by the same process that built the feature. A leaked
> feature does not raise an error. It raises the AUC, which is why it is hard
> to notice.

---

## 6. Feature Specification

Declared now. No features are added after seeing validation results.

**Group A, registration and demographic (8)**

`gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`,
`num_of_prev_attempts`, `studied_credits`

**Group B, engagement volume (5)**

- total clicks before D
- distinct active days before D
- clicks in the 7 days before D
- mean clicks per active day
- days registered before presentation start

**Group C, engagement trajectory (4)**

- days since last activity as at D
- slope of weekly click counts (OLS over weeks 1..D/7)
- ratio of clicks in the second half of the window to the first half
- longest gap in days between consecutive active days

**Group D, engagement breadth (3)**

- distinct VLE activity types touched
- distinct VLE materials touched
- share of clicks on assessment-related material types

**Group E, assessment behaviour (5)**

- count of assessments submitted before D
- mean score on those submitted
- minimum score on those submitted
- mean days early or late relative to due date
- count of assessments due before D with no submission

**Group F, cohort-relative (3)**

- percentile of total clicks within own module-presentation
- percentile of mean assessment score within own module-presentation
- module-presentation identifier as a categorical

**Total: 28 features.**

*Group F rationale.* Click volumes differ by an order of magnitude between
modules, so raw counts are not comparable across a cohort. Ranking a student
against their own module is closer to how a tutor would read the data, and it
is the feature group most likely to survive to a new module. Computed strictly
on pre-D activity per L6.

*Prediction stated in advance:* Group E's last feature (assessments due before
D with no submission) is expected to dominate. Recorded here so that it reads
as a prediction rather than a discovery.

---

## 7. Models and Baselines

A ladder, all reported. The point is what each rung adds.

```
B0   Base rate. Predict cohort prevalence for everyone.
B1   Single feature: days since last activity, logistic.
B2   Demographics only (Group A), logistic.
B3   Full feature set, logistic with L2 regularisation.
M1   LightGBM, full feature set.
```

B2 establishes how much is predictable from who the student is rather than what
they did. It is referenced by the fairness slices in Section 8 and by E4 in
Section 11.

**B3 is the honest benchmark.** If M1 does not beat a regularised logistic
model by a margin that survives the bootstrap interval in Section 8, the
reported conclusion is that gradient boosting was not warranted here. That is a
legitimate finding and is declared valid in Section 13 (outcome O2).

**Hyperparameter search, declared in advance**

LightGBM: `num_leaves {15, 31, 63}`, `learning_rate {0.01, 0.05, 0.1}`,
`min_child_samples {20, 50, 100}`, `n_estimators` by early stopping on
validation, max 1000. Grid only, no adaptive search, so the number of
configurations tried is fixed at 27 and stated.

Class imbalance handled by `scale_pos_weight` at the observed ratio. No
resampling. SMOTE and its relatives distort the probability calibration that
Section 8 makes a primary metric, which would defeat the purpose.

Seed fixed at 42, with a control run at three other seeds to confirm the
ranking of the ladder is stable.

---

## 8. Metrics, Pre-Committed

**Primary**

- Expected cost at the operating threshold, at the 10:1 ratio. This is the
  metric the project optimises, because it is the only one that corresponds to
  a decision.

**Secondary**

- AUC-PR. Preferred to AUC-ROC under imbalance because it does not reward
  performance on the abundant negative class.
- Brier score and a calibration curve with expected calibration error. A
  probability that feeds a triage decision has to mean something, not merely
  rank correctly.
- Recall at fixed alert budgets: top 5%, 10%, 20% of the cohort. This is the
  number an operations team actually asks for, because their capacity is fixed
  regardless of what the model says.

**Reported for completeness**

- AUC-ROC, so the work is comparable to published OULAD results, with the split
  difference from Section 4 stated alongside.

**Uncertainty**

Stratified bootstrap, 1,000 resamples, on the test set, giving intervals on
every headline figure. Differences between ladder rungs reported as paired
intervals rather than point comparisons.

**Slice reporting, mandatory**

All primary and secondary metrics reported separately across `imd_band`,
`age_band`, `disability` and `gender`.

This is reported, not fixed. An early warning model that performs worse for
students from deprived postcodes is worse precisely where the intervention
matters most, and B2 in Section 7 exists to show how much of the signal is
demographic in the first place. No fairness intervention is attempted and none
is claimed.

---

## 9. Threshold Policy and Cost Assumptions

**Cost model**

```
C_FN = cost of failing to flag a student who does not complete
C_FP = cost of flagging a student who completes anyway

Headline ratio  C_FN : C_FP = 10 : 1
Swept across    2:1 to 20:1, reported as a curve
```

*Justification, stated as an assumption rather than a measurement.* C_FP is one
outreach contact: staff time, plus some nuisance to the student. C_FN is a
student who leaves. The ratio is asserted, not derived, and the sweep exists so
that a reader who disagrees with 10 can read their own number off the curve.

**Threshold selection.** The threshold minimising expected cost is chosen on
the validation presentation, frozen, and applied unchanged to the test
presentation. It is not re-optimised on test. The frozen value is recorded in
the protocol log before the holdout opens.

**Alert budget as a second policy.** Reported alongside: if capacity allows
contacting only the top K%, what recall is achieved. Cost-optimal and
budget-constrained thresholds will differ, and the gap between them is the
practical finding.

**Stability check.** How far the cost-optimal threshold moves between
validation and test. A threshold that shifts substantially is a warning that
the policy will not hold across presentations, and that is a result worth
reporting rather than hiding.

---

## 10. Drift Measurement

A model trained on 2013 presentations and deployed on a 2014 one assumes the
population and its behaviour did not move. That assumption is testable and the
test is cheap, so it goes in the protocol rather than being discovered later.

- **D1. Feature drift.** Population Stability Index per feature, train against
  test. Conventional reading: below 0.1 stable, 0.1 to 0.25 moderate, above
  0.25 significant. Reported per feature, ranked. The convention is stated as a
  convention, not a law.
- **D2. Base rate drift.** Non-completion rate per presentation, per module,
  with intervals. A shift in prevalence alone will move the cost-optimal
  threshold even if nothing about the students changed.
- **D3. Exclusion drift.** Share of students removed by exclusion E1 per
  presentation. If a materially different share of the test presentation had
  already left by day 28, the prediction task is not the same task.
- **D4. Performance decomposition.** Test performance reported per module, and
  separately for modules with thin training representation per Section 4. This
  separates "the model degraded over time" from "the model never saw this
  module".

**What is not done.** No drift correction, no reweighting, no recalibration on
test. Detecting drift and then fixing it using the holdout would consume the
holdout. Drift is measured and reported.

---

## 11. Explainability

- **E1. Global.** SHAP mean absolute value per feature, on test, at day 28.
  Compared against logistic coefficients from B3. Where the two disagree
  materially, that is reported rather than resolved, because the disagreement
  is informative about non-linearity.
- **E2. Local.** For a sample of flagged students, the top three contributing
  features with direction, rendered as a plain-language reason string. A tutor
  receiving an alert needs to know what to open the conversation with.
  "Flagged: no submission for the first assignment, no activity in 11 days" is
  actionable. A risk score alone is not.
- **E3. Counterfactual sanity check.** SHAP explains the model. It does not
  explain the world. If "days since last activity" drives a flag, that does not
  mean prompting a login reduces risk. Inactivity is a symptom. This is stated
  explicitly in the write-up next to the reason strings, because reason codes
  invite causal reading and that misreading is exactly what Section 12 exists
  to guard against.
- **E4. Demographic contribution.** Reported: how much of the model's output is
  attributable to Group A features. Ties directly to the B2 baseline and the
  slice reporting in Section 8. A model whose flags are substantially driven by
  postcode deprivation is a different product from one driven by observed
  behaviour, and the write-up says which one this is.

---

## 12. Arm 2, Causal Design

**The problem.** The model in Arm 1 ranks students by risk. It cannot tell you
whether acting on that ranking helps. OULAD contains no recorded intervention,
no treatment flag, and no experiment. So the honest options are: skip Arm 2, or
find a source of quasi-random variation already present in the data.

**The design: regression discontinuity at the assessment pass mark.**

OULAD assessments have a recorded pass threshold of 40. A student scoring 39
and a student scoring 41 on an early assignment are, in expectation,
near-identical in ability. What differs is which side of a salient line they
landed on, and therefore what feedback signal they received.

**Estimand.** The local effect, on final non-completion, of narrowly failing
versus narrowly passing the first assessed piece of work.

**Specification, fixed in advance**

- Running variable: score on the first assessment (TMA) of the presentation,
  centred at 40
- Outcome: `not_completed`
- Bandwidth: primary at +/- 10 marks; robustness at 5, 8, 15, 20
- Local linear, triangular kernel, separate slopes each side
- Robust bias-corrected inference (Calonico, Cattaneo, Titiunik)
- Population: students with a first-assessment score recorded, which is a
  different and smaller population than Arm 1

**Validity checks, all pre-committed and all reported whatever they show**

- **V1.** McCrary density test for manipulation of the running variable.
  Marking to a boundary is a real phenomenon. If markers push borderline
  students up to 40, the design is compromised and the result is reported as
  invalid rather than quietly retained. Heaping at round numbers is expected;
  whether it is asymmetric around 40 is the question.
- **V2.** Covariate balance at the cutoff across every Group A feature.
- **V3.** Placebo cutoffs at 30, 50 and 60. A discontinuity appearing where no
  rule exists means the specification is finding noise.
- **V4.** Donut specification excluding scores within 1 mark of 40, in case
  exact-40 scores are administratively assigned.

**Power analysis, the second half of Arm 2**

Given the observed effect size and variance, compute the sample size a
randomised trial would need to detect an intervention effect of that magnitude
at 80% power. Reported per plausible effect size, as a curve. This is the
artefact that answers "how would you design the experiment". It is honest about
the fact that no experiment was run and shows what running one would require.

**What this design does NOT establish**

It is a local effect at a specific score boundary, on students who sat the
first assessment. It says nothing about students who never submitted, who are
the highest-risk group in Arm 1. That disconnect between the population the
model targets and the population the causal estimate covers is a genuine
limitation, it is stated in Section 14, and it is not resolved.

---

## 13. Declared Outcomes, All Valid in Advance

**Arm 1**

- **O1.** LightGBM materially beats regularised logistic on expected cost at
  day 28, interval-separated. Reported.
- **O2.** LightGBM does not beat logistic by an interval-separated margin.
  Reported as gradient boosting not being warranted. This is a real finding,
  not a failure.
- **O3.** Performance at day 14 is close to day 28, meaning early action costs
  little. Reported.
- **O4.** Performance at day 14 is substantially worse, meaning the timeliness
  cost is real and quantified. Reported.
- **O5.** Model performance degrades sharply on the test presentation. Reported
  as drift, with Section 10's decomposition.
- **O6.** Slice reporting shows materially worse performance for one or more
  groups. Reported, not corrected, with the limitation stated.

**Arm 2**

- **O7.** A significant discontinuity at the pass mark. Reported with full
  validity checks.
- **O8.** No detectable discontinuity. Reported, with the power analysis
  showing what effect size the design could have detected. A null with a stated
  minimum detectable effect is a result.
- **O9.** V1 fails and the design is invalid. Reported as such, Arm 2 reduced
  to the power analysis alone, and the write-up says why.

Every one of these nine is a publishable outcome of the study. Committing to
the method only counts if you have committed to reporting whatever it produces.

---

## 14. Limitations Known in Advance

- **L1.** Population-target mismatch between arms, per Section 12.
- **L2.** Single institution, single provider, 2013 to 2014, distance learning.
  Generalisation to other settings is not claimed.
- **L3.** Clicks are a proxy for engagement. A student reading a printed
  handout registers as inactive.
- **L4.** Four presentations means four temporal units. One test presentation
  is one draw. Conclusions about drift over time rest on a small number of
  periods and the write-up says so.
- **L5.** No intervention exists, so Arm 1's operating threshold is evaluated
  against an assumed cost ratio rather than a measured one.
- **L6.** Modules differ in structure, assessment weight and duration. Pooling
  them is a simplification and per-module reporting is the partial mitigation.
- **L7.** `imd_band` is a UK deprivation measure with no direct analogue
  elsewhere, which limits transfer of the fairness findings.

---

## 15. Out of Scope

- Deep learning and sequence models over clickstream. Tabular gradient boosting
  is the right tool and reaching past it would be padding.
- PySpark. The data fits in memory. Distributed compute here would be theatre.
- Real-time serving infrastructure. A single prediction endpoint is built to
  demonstrate the lifecycle is understood. No orchestration, no feature store,
  no monitoring service.
- Fairness mitigation. Disparities are measured and reported. Fixing them is a
  separate project with its own protocol.
- Reject inference or any correction for students absent from the data.

---

## Execution Stages

Each stage is verified before the next begins.

```
Stage 1   Ingest, checksums, DuckDB load, table validation
Stage 2   Target, exclusions, cohort counts        (VERIFICATION STOP 1)
Stage 3   SQL feature construction at three cutoffs (VERIFICATION STOP 2)
Stage 4   Model ladder, validation only, threshold frozen
Stage 5   Holdout opened once, all metrics, slices, bootstrap
Stage 6   Drift and explainability
Stage 7   Arm 2, RDD and power analysis
Stage 8   Serving endpoint, write-up
```


---

## Amendments

Amendments are recorded after the root commit, with the original text above left
unchanged. Each states what changed and why. A1 to A3 were made after Stage 1
validation and before any Stage 2 code existed.

### A1. Group E must exclude banked assessment scores

**Amends:** Section 6, Group E.

`student_assessment` contains an `is_banked` flag, not referenced in the
original specification. A banked score is carried over from a previous
presentation and is not evidence of engagement in the current one. Counting a
banked score as a submission would credit a student with activity they did not
perform in the window being measured.

All five Group E features are computed over rows where `is_banked = 0`. The
count of excluded rows and the number of students affected is reported.

*Why it was missed:* the original specification was written from the published
table schema without inspecting the column set. Found at Stage 1 validation.

### A2. Students recur across the temporal split

**Amends:** Section 4.

Stage 1 found 32,593 rows over 28,785 distinct students, so a student can appear
in more than one module-presentation, and therefore on both sides of the
train/test split.

This is not leakage in the technical sense. No information from the test
presentation reaches the training data, and a returning student is a genuine
deployment case already encoded by `num_of_prev_attempts`. Such students are
therefore retained.

However, per-student memorisation becomes a competing explanation for test
performance. The count of distinct students appearing in both the 2014J test
split and either training presentation is reported as a raw number and as a
share of the test split. If that share is material, it is named in the write-up
as an alternative explanation for measured performance, alongside a comparison
of test performance on returning students versus first-time students.

### A3. Engagement slope is computed daily, not weekly

**Amends:** Section 6, Group C, second feature.

The original specification computes the slope of weekly click counts by OLS
over weeks 1 to D/7. At D = 14 this gives two points, which is a difference
between two numbers rather than a regression, and it is not comparable to the
same feature at D = 56 where it fits over eight points.

The slope is instead fitted by OLS over daily click counts from day 0 to day
D-1, with days of no activity entered as zero rather than omitted. This gives
14, 28 and 56 points at the three cutoffs and makes the feature comparable
across them.

*Why it was missed:* a specification defect, found on reading the protocol back
before implementation rather than from any data.

### A4. Provenance

**Amends:** Section 2.

The dataset copy used is the UCI Machine Learning Repository mirror (dataset
349, DOI 10.24432/C5KK69), not the Open University download page, which was
non-functional at time of ingest. The originating publication remains Kuzilek,
Hlosta and Zdrahal, *Scientific Data*, 2017. SHA-256 checksums of the seven CSVs
as retrieved are recorded in `data/CHECKSUMS.txt`.

### A5. The observation window has no lower bound

**Amends:** Section 6, Groups B, C and D.

Section 5 specifies the upper edge of the observation window (`date < D`) and
never specifies a lower one. `student_vle` contains activity from day -25, so
five features (`total_clicks`, `active_days`, `distinct_activity_types`,
`distinct_materials`, `click_slope_daily`) include engagement occurring before
the presentation began.

This is retained rather than corrected. Pre-registration activity is known at
the point of prediction, so it breaches no leakage rule, and early engagement
with course materials is exactly the behaviour an early warning system should
be reading. The window is therefore "everything known up to day D", not "the
first D days".

Consequence: `active_days` can exceed D. One traced student (FFF 2013J,
id 606428) records 31 active days at D=28, of which 11 precede day 0.
`click_slope_daily`, per Amendment A3, is still fitted over days 0 to D-1 only,
so the slope and the volume features cover different windows. That asymmetry is
noted and not resolved.

*Why it was missed:* the specification was written assuming activity begins at
day 0. Found at Verification Stop 2 by manually tracing a sampled student
against the raw table, not by any automated check.

### A6. module_presentation is replaced by code_module

**Amends:** Section 6, Group F, third feature.

Group F specifies `module_presentation` (the concatenation of `code_module` and
`code_presentation`) as a categorical. Under the temporal split in Section 4,
this feature is degenerate: training presentations are 2013B and 2013J,
validation is 2014B and test is 2014J, so no value observed in training ever
recurs at validation or test. Every learned dummy is zero at scoring time.

The first Stage 4 run made the consequence visible. In the B3 logistic model at
D=28, the two largest coefficients by absolute value were on
`module_presentation` dummies (0.79 and -0.73), both exceeding the largest
behavioural coefficient. Those coefficients contribute nothing to any
prediction the model is scored on, while shifting its effective intercept and
therefore its calibration, which Section 8 designates a headline metric.

The feature is replaced by `code_module` (7 values: AAA to GGG), which appears
in both the training and test presentations and preserves the original intent
of controlling for between-module differences in difficulty and activity
volume. `code_presentation` is not used as a feature in any form, being
perfectly collinear with the split.

Stage 4 is re-run in full under this amendment before the holdout is opened.
The original run is retained in git history.

*Why it was missed:* the feature was specified from the data schema without
checking its behaviour against the temporal split. Found by reading the fitted
coefficients in the Stage 4 report, not by any automated check.
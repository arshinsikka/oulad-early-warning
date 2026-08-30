# Early warning for course non-completion

A pre-registered study on the Open University Learning Analytics Dataset. Two questions: can you identify students who will not finish a course early enough to do something about it, and does acting on that identification help.

The first question has an answer. The second one doesn't, and finding out why was the more useful half.

**Findings:** the consolidated write-up — results, all nine declared outcomes, the amendment log and the limitations — is in [reports/FINDINGS.md](reports/FINDINGS.md).

The protocol is the first commit in this repository. Every decision that could have been tuned after seeing results was fixed before any code existed: the features, the split, the metrics, the cost assumption, the operating threshold, and the nine outcomes I committed to reporting whichever one occurred. Seven amendments follow, each recording something the protocol got wrong and when I found it.

## What the model does

Score a student at day 28 of a course using only what was known by then, and predict whether they finish with Fail or Withdrawn.

Training runs on the 2013 presentations, validation on 2014B, and the held-out test presentation is 2014J, opened once. 28 features built in SQL over seven relational tables, covering registration details, click volume and trajectory, breadth of material touched, and assessment submission behaviour.

Day 28 is the headline. Day 14 and day 56 are reported alongside it, because the interesting question is how much accuracy you give up to act a fortnight earlier.

## The result

On the held-out presentation at day 28, LightGBM reaches AUC-PR of 0.679 with a bootstrap interval of [0.667, 0.692], against 0.611 [0.597, 0.625] for regularised logistic regression. At an alert budget of the top 5% of the cohort, it recovers 0.114 of non-completers against a ceiling of 0.123, so roughly 93% of what that budget could possibly catch.

The model discriminates. On the metric I had pre-registered as primary, it is worth almost nothing.

Expected cost at a 10:1 penalty for missing a student came out at 0.587 for LightGBM. Flagging every student in the cohort costs 0.592. The model beats the trivial policy by 0.84%, and the logistic model at its frozen threshold is marginally worse than flagging everyone.

The mechanism is arithmetic. Expected cost is (10 × false negatives + false positives) / n. At a base rate of 0.408, missing anyone is punished ten times harder than a false alarm, so the cost-optimal policy collapses towards alerting on everybody and stops distinguishing between models. The pre-committed sweep from 2:1 to 20:1 shows where the frozen policy stops paying: at day 28 it beats flagging everyone by 0.84% at 10:1, by 0.02% at 13:1, and from 14:1 upward it is worse than flagging everyone. At day 56 that crossover is 16:1, and at day 14 it never happens inside the swept range. The optimum itself does not become flag-everyone anywhere in that range — the counterfactual test-optimal threshold still beats flagging everyone by 0.40% at 20:1. What breaks is the frozen threshold, applied at a ratio it was not chosen for.

I picked 10:1 by analogy to credit and fraud problems, where base rates run between 1 and 5%. At 41% it is the wrong ratio, and I fixed it before looking at any data, so it stays. Section 9 of the protocol says the ratio is asserted rather than measured. It turned out to determine the conclusion.

## Four of six outcomes are undecidable

Section 13 declared nine outcomes in advance and called each a valid finding. What it did not do, for most of them, was say what would count as meeting one.

Two of the six Arm 1 outcomes are decided. O1 and O2 turn on whether a bootstrap interval contains zero, which the protocol does specify. The paired difference between LightGBM and logistic at day 28 is -0.005 with an interval of [-0.017, 0.006], so O2 is met, O1 is not, and gradient boosting is not shown to be warranted on the primary metric.

O3 through O6 are worded as "substantially worse" and "materially worse" with no threshold attached anywhere. My first pass through the results invented three: a 10% relative AUC-PR drop, and 15% relative cost degradation used twice. All three were chosen after I could see the numbers, and each landed on the convenient side. They are withdrawn, and all four of those outcomes are reported as undetermined with the underlying figures printed and no verdict derived. Arm 2 is decidable throughout: O7, O8 and O9 turn on statistical significance and on whether the validity check fails, both of which Section 12 specified.

Pre-registering an outcome does nothing unless you also pre-register the criterion that decides it. That is the clearest thing this study taught me and it is a finding about my protocol rather than about the model.

## PSI would not have caught the degradation

Performance dropped between presentations. LightGBM's AUC-PR fell from 0.787 on validation to 0.679 on test, and logistic fell further, from 0.784 to 0.611. Some of the raw cost movement is prevalence, since the base rate itself fell from 0.484 to 0.408 and the flag-everyone floor rose almost as much. The discrimination fall is separate and real.

Population Stability Index is what production teams monitor for this. I computed it per feature, training against test.

The features that moved are not the features the model uses. Overlap between the top five by PSI and the top five by SHAP importance is one of five. The model's most important feature, a count of assessments due but not submitted, sits at PSI rank 23 and is among the most stable things measured. Its third most important feature has the second-lowest PSI in the study, 0.0002.

PSI measures the marginal distribution of each feature. What costs you accuracy is the conditional relationship between features and outcome moving, and PSI is blind to that. The monitor most likely to be running in production would have shown almost nothing while the model lost 14% of its discrimination.

One feature scored 1.80 and the number is not trustworthy. Module CCC has zero rows in the training split and 21.9% of the test split, so the PSI term exists only because of the constant used to fill empty bins. It reads 1.29 at 1e-3 and 2.81 at 1e-6. The compositional change is real, the magnitude is a property of my choice of constant, and the report says so.

## The causal arm failed, on schedule

The model ranks students by risk. It cannot tell you whether intervening on that ranking changes anything, and OULAD contains no recorded intervention, so I used a regression discontinuity at the assessment pass mark of 40. A student scoring 39 and one scoring 41 should be near-identical in ability, and they receive different signals.

The density test rejected. Both implementations found more mass just above 40 than just below, with McCrary at z = 6.78. That is the direction marking to a boundary produces.

Scores heap at round numbers generally, and 40 is not the worst offender: it sits at 4.93 times the mean of its neighbours while 50 is at 5.20 and 30 at 5.57. That is a real argument for keeping the design, and taking it would have meant choosing a criterion after seeing the test fail. Outcome O9, invalid design, was declared in advance for exactly this. So Arm 2 reduces to the power analysis.

The donut check makes the case anyway. Dropping the 201 rows within one mark of the cutoff moves the estimate from -0.128 to -0.025. Most of what the full-sample estimate measures is sitting in the mass point the density test flagged.

The design's minimum detectable effect is 21.7 percentage points. Anything smaller was never findable. A properly randomised trial detecting a 5 point effect would need 1,439 students per arm.

## The gap between the two arms

Arm 2 covers students who sat and submitted the first assignment. Arm 1 covers everyone registered at day 28.

3,893 of the 27,530 students in the day 28 cohort fall outside the Arm 2 population, 14.1% of it. Their non-completion rate is 0.843 against 0.375 for those inside.

The students an early warning system exists to reach are the ones the causal design cannot see. Section 12 predicted a population mismatch in the abstract. The numbers are worse than the caveat implied, and I have not resolved it.

## Running it

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/stage1_ingest.py <path to oulad.zip>
python src/stage1_validate.py
python src/stage2_cohort.py        # builds the cohort tables
python src/stage2_report.py        # writes the report — Verification Stop 1
python src/stage3_features.py      # builds the feature tables
python src/stage3_report.py        # writes the report — Verification Stop 2
python src/stage4_ladder.py        # fits the ladder, freezes the threshold
python src/stage4_report.py        # writes the report
python src/stage5_holdout.py
python src/stage6_report.py
python src/stage7_report.py
python src/stage8_a2_overlap.py
```

Stages 2, 3 and 4 split computation from reporting, and both halves must run. Everything else in `src/` is a module imported by one of these, not an entry point. Full detail, including which artefacts are immutable once written, is in [reports/FINDINGS.md](reports/FINDINGS.md) section 7.

Data comes from the UCI Machine Learning Repository, dataset 349, licensed CC-BY 4.0. Checksums for the seven source CSVs are in `data/CHECKSUMS.txt`. The raw data and the DuckDB database are not committed.

Reports land in `reports/`. `stage5_interpretation.md` records how I read the holdout results, including two readings I got wrong first.

## What this does not show

One institution, one provider, distance learning, four course presentations between 2013 and 2014. Clicks are a proxy for engagement and a student reading a printed handout registers as inactive. No intervention exists in the data, so the operating threshold is evaluated against an assumed cost ratio rather than a measured one. Fairness disparities across deprivation band, age, disability and gender are measured and reported per slice, and nothing is done to correct them.

Implementation was written by Claude Code against specifications I wrote and verified. The two manual verification stops in the protocol are mine: one caught that the observation window had no lower bound and was silently including activity from before the course started, which no automated check would have raised.
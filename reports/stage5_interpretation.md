# Stage 5 Interpretation

Written after the holdout was opened, before Stage 6. Records how the test
results are read, including two initial readings that were wrong.

## The primary metric is degenerate at this base rate

Expected cost is (10 x FN + FP) / n. At a test base rate of 0.4079 with a 10:1
penalty, missing a student is expensive and flagging one is cheap, so the
cost-optimal policy converges on flagging everyone. Flag-everyone costs
1 - 0.4079 = 0.5921.

Against that floor:

| model | expected cost | improvement over flag-everyone |
|-------|---------------|-------------------------------|
| B3    | 0.5922        | -0.0001 (worse)               |
| M1    | 0.5871        | +0.0050 (0.84%)               |

The regularised logistic model, at its pre-committed threshold, is very
slightly worse than a constant policy. Gradient boosting beats it by under 1%.

This is a property of the cost assumption interacting with the base rate, not
of the models. Section 9 stated the 10:1 ratio was asserted rather than
measured, and the pre-committed sweep from 2:1 to 20:1 exists so the
sensitivity is visible. It turned out the assumption determined the conclusion.

The design error is mine: 10:1 was chosen by analogy to credit and fraud
problems, where base rates run 1-5%. At 41% the ratio is inappropriate. It was
fixed before any data was examined and is not revised after the fact.

## O2 is met, and the reason matters

Paired bootstrap on expected cost, M1 vs B3 at D=28: the 95% interval contains
zero. Outcome O2 as declared in Section 13.

On every secondary metric the two models separate cleanly:

| metric   | M1     | 95% CI           | B3     | 95% CI           |
|----------|--------|------------------|--------|------------------|
| AUC-PR   | 0.6794 | [0.6674, 0.6919] | 0.6106 | [0.5970, 0.6250] |
| Brier    | 0.2108 | [0.2067, 0.2150] | 0.2153 | [0.2103, 0.2203] |
| AUC-ROC  | 0.7498 | [0.7404, 0.7596] | 0.7291 | [0.7188, 0.7393] |
| recall@5%| 0.1144 | [0.1112, 0.1168] | 0.0830 | [0.0777, 0.0878] |

These are not in tension. O2 is met because the primary metric cannot
distinguish models at this base rate, not because the models perform alike.
Reporting O2 without this qualification would be misleading.

## The model discriminates well

At the 5% alert budget, M1 achieves recall of 0.1144 against a ceiling of
0.1226 (budget divided by base rate). That is 93% of the maximum achievable,
meaning the top 5% of flagged students is almost entirely correct.

Raw recall at a fixed budget understates performance under a high base rate.
Every recall figure in this study is reported with its ceiling.

## O5: two readings, one of which was wrong

The first reading was that expected cost rose from 0.5047 on validate to
0.5871 on test, a 16.3% degradation, therefore O5.

That reading is mostly wrong. The trivial baseline moved almost as much,
because the base rate fell from 0.4844 to 0.4079:

| | validate | test |
|---|---|---|
| flag-everyone | 0.5156 | 0.5921 |
| selected model | 0.5047 | 0.5871 |
| model's edge | 0.0109 | 0.0050 |

Most of the apparent degradation is the floor rising. What actually fell is the
model's edge over the trivial policy, which roughly halved.

The second reading was that there was therefore no real degradation, only an
arithmetic artefact. That is also wrong. Discrimination fell:

| | validate | test |
|---|---|---|
| M1 AUC-PR | 0.7870 | 0.6794 |
| B3 AUC-PR | 0.7843 | 0.6106 |
| M1 Brier | 0.1883 | 0.2108 |

AUC-PR moves with prevalence, and a fall from 0.79 to 0.68 against a base rate
moving 0.48 to 0.41 is larger than prevalence accounts for.

**O5 is met**, on the evidence of falling discrimination across presentations,
not on the raw cost figure. Stage 6's PSI analysis should identify which
features moved.

Note: B3 degraded far more than M1 (0.7843 to 0.6106, against 0.7870 to
0.6794). The linear model generalised worse across presentations.

## Threshold stability

Validate-frozen thresholds against the test-optimal counterfactual:

| cutoff | frozen | test-optimal | gap |
|--------|--------|--------------|-----|
| D=14   | 0.09   | 0.09         | 0.00 |
| D=28   | 0.14   | 0.16         | 0.02 |
| D=56   | 0.13   | 0.12         | 0.01 |

Small. The test-optimal values are counterfactual and were not deployed.

## What the ratio sweep does and does not show

Above roughly 17:1, applying the 10:1-optimal threshold performs worse than
flagging everyone. This is not evidence that the threshold is fragile. It is
what happens when a threshold optimised for one cost ratio is applied at
another; the correct threshold for each ratio is in the sweep table and never
loses to the trivial policy.

Three claims are supported:

1. The cost-optimal threshold is sensitive to a ratio that was asserted rather
   than measured.
2. Above roughly 17:1 the optimal policy converges on flag-everyone regardless
   of model quality. This is a statement about the base rate.
3. Deploying a 10:1 threshold when the true ratio is 18:1 would perform worse
   than having no model. That is a deployment risk from miscalibrating an
   assumption.
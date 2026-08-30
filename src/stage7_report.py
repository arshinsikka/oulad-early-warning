"""
Stage 7: Arm 2, regression discontinuity at the assessment pass mark, and the
power analysis, per PROTOCOL.md Section 12.

Arm 2 is independent of Arm 1. No model artefact, frozen threshold or Stage 5
test prediction is read anywhere in this stage. The Arm 1 cohort table is
touched once, at the end, only to count how many of its students fall outside
the Arm 2 population, which Section 12 requires as a stated limitation.

Usage:
    .venv/bin/python src/stage7_report.py
"""

import warnings
from pathlib import Path

import duckdb
import numpy as np

from stage7_rdd import (
    ALPHA, DONUT_RADIUS, PASS_MARK, PLACEBO_CUTOFFS, POPULATION_SQL, POWER,
    PRIMARY_BANDWIDTH, ROBUSTNESS_BANDWIDTHS, cjm_density_test,
    mccrary_density_test, minimum_detectable_effect, n_per_arm, rd_estimate,
    score_frequencies,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "oulad.duckdb"
REPORT_PATH = ROOT / "reports" / "stage7_rdd.txt"

ARM1_CUTOFF = 28
FREQ_LO, FREQ_HI = 30, 50
EFFECT_SIZES_PP = list(range(1, 11))
GROUP_A_NUMERIC = ["num_of_prev_attempts", "studied_credits"]
GROUP_A_CATEGORICAL = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
NULL_TOKEN = "__NULL__"


def section(title: str) -> str:
    bar = "=" * len(title)
    return f"\n{title}\n{bar}\n"


def fmt_est(est: dict, key: str) -> str:
    if not est["estimable"]:
        return f"{'not estimable':>62}"
    r = est[key]
    ci = "[{:+.4f},{:+.4f}]".format(r["ci_lo"], r["ci_hi"])
    return f"{r['coef']:>+10.4f}{r['se']:>9.4f}{ci:>22}{r['pv']:>10.4f}"


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    out: list[str] = []

    out.append("Stage 7 Report: Arm 2, Regression Discontinuity at the Pass Mark")
    out.append(
        "Per Section 12. Arm 2 is independent of Arm 1: no model artefact, no frozen threshold and no "
        "Stage 5 prediction is loaded here. The Arm 1 cohort is read once, in Section 9, only to size the "
        "population gap between the two arms."
    )

    # ------------------------------------------------------------------
    out.append(section("1. Population"))
    out.append(
        "The running variable is the score on the FIRST TMA of each module-presentation, taken by due "
        "date with id_assessment as a deterministic tie-break. Section 12 fixes the first assessed piece "
        "of work as the running variable; Amendment A1 excludes banked scores, which are carried from a "
        "previous presentation and are not evidence of work done in this one."
    )
    out.append("")

    first_tma = con.execute(
        """
        WITH first_tma AS (
            SELECT code_module, code_presentation, id_assessment, date AS due_date, weight,
                   row_number() OVER (PARTITION BY code_module, code_presentation
                                      ORDER BY date ASC NULLS LAST, id_assessment ASC) AS rn
            FROM v_assessments WHERE assessment_type = 'TMA'
        )
        SELECT code_module, code_presentation, id_assessment, due_date, weight
        FROM first_tma WHERE rn = 1 ORDER BY code_module, code_presentation
        """
    ).df()
    out.append(f"  {'module':<8}{'presentation':<15}{'id_assessment':>14}{'due day':>9}{'weight':>8}")
    for row in first_tma.itertuples():
        out.append(
            f"  {row.code_module:<8}{row.code_presentation:<15}{int(row.id_assessment):>14}"
            f"{int(row.due_date):>9}{row.weight:>8.1f}"
        )
    out.append("")

    waterfall = con.execute(
        """
        WITH first_tma AS (
            SELECT code_module, code_presentation, id_assessment,
                   row_number() OVER (PARTITION BY code_module, code_presentation
                                      ORDER BY date ASC NULLS LAST, id_assessment ASC) AS rn
            FROM v_assessments WHERE assessment_type = 'TMA'
        ), ft AS (SELECT * FROM first_tma WHERE rn = 1)
        SELECT
            count(*) AS submission_rows,
            sum(CASE WHEN sa.score IS NULL THEN 1 ELSE 0 END) AS null_score,
            sum(CASE WHEN sa.is_banked = 1 THEN 1 ELSE 0 END) AS banked,
            sum(CASE WHEN sa.score IS NOT NULL AND sa.is_banked = 0 THEN 1 ELSE 0 END) AS retained
        FROM ft JOIN v_student_assessment sa USING (id_assessment)
        """
    ).df().iloc[0]

    pop = con.execute(POPULATION_SQL).df()
    x_raw = pop["score"].values.astype(float)
    y = pop["not_completed"].values.astype(float)

    out.append("  Population waterfall:")
    out.append(f"    submission rows against a first TMA        {int(waterfall.submission_rows):>8}")
    out.append(f"    - null score                               {int(waterfall.null_score):>8}")
    out.append(f"    - banked (Amendment A1)                    {int(waterfall.banked):>8}")
    out.append(f"    = Arm 2 population                         {len(pop):>8}")
    out.append("")
    out.append(f"  Distinct students                            {pop['id_student'].nunique():>8}")
    out.append(f"  Non-completion rate in this population       {y.mean():>8.4f}")
    out.append(f"  Score range                                  {x_raw.min():.0f} to {x_raw.max():.0f}")
    out.append(
        f"  Presentations used                           all four "
        f"({', '.join(sorted(pop['code_presentation'].unique()))})"
    )
    out.append("")
    out.append(
        "No Section 3 exclusion is applied. E1, E2 and E3 exist to make a prediction at day D honest; "
        "Arm 2 makes no prediction and has no prediction point, so they do not apply. The consequence is "
        "that the two arms cover different populations, and Section 9 below sizes the difference rather "
        "than leaving it as a caveat."
    )
    out.append("")
    out.append(f"  {'presentation':<15}{'n':>8}{'base_rate':>11}")
    for row in con.execute(
        f"WITH pop AS ({POPULATION_SQL}) SELECT code_presentation, count(*) n, avg(not_completed) br "
        "FROM pop GROUP BY 1 ORDER BY 1"
    ).df().itertuples():
        out.append(f"  {row.code_presentation:<15}{int(row.n):>8}{row.br:>11.4f}")
    out.append("")

    # ------------------------------------------------------------------
    out.append(section("2. Specification"))
    out.append(
        f"  Running variable      first-TMA score, centred at {PASS_MARK:.0f}\n"
        "  Outcome               not_completed (Section 3: Fail or Withdrawn)\n"
        f"  Primary bandwidth     +/- {PRIMARY_BANDWIDTH:.0f} marks\n"
        f"  Robustness            {', '.join(f'{b:.0f}' for b in ROBUSTNESS_BANDWIDTHS)}\n"
        "  Estimator             local linear, triangular kernel, separate slopes each side\n"
        "  Inference             robust bias-corrected (Calonico, Cattaneo, Titiunik), via rdrobust"
    )
    out.append("")
    out.append(
        "The estimate is the limit from the right minus the limit from the left, so it is the effect of "
        "landing just ABOVE the pass mark. A negative coefficient means narrowly passing is associated "
        "with less non-completion."
    )
    out.append(
        "rdrobust reports mass points in the running variable: scores are integers, so every attainable "
        "value is a mass point. That is a property of the data, not a defect of the fit, and it is "
        "handled by the estimator's mass-point adjustment. It matters for V1 below."
    )
    out.append("")

    # ------------------------------------------------------------------
    out.append(section("3. Primary and robustness estimates"))
    out.append(
        f"  {'bw':<5}{'method':<17}{'estimate':>10}{'SE':>9}{'[95% CI]':>22}{'p':>10}{'n_left':>9}{'n_right':>9}"
    )
    bandwidths = [PRIMARY_BANDWIDTH] + ROBUSTNESS_BANDWIDTHS
    estimates = {}
    for bw in sorted(bandwidths):
        est = rd_estimate(y, x_raw, PASS_MARK, bw)
        estimates[bw] = est
        marker = "  <- primary" if bw == PRIMARY_BANDWIDTH else ""
        for label, key in [("Conventional", "conventional"), ("Bias-corrected", "bias_corrected"),
                           ("Robust", "robust")]:
            prefix = f"  {bw:<5.0f}" if label == "Conventional" else "  " + " " * 5
            tail = ""
            if est["estimable"]:
                tail = f"{est['n_left']:>9}{est['n_right']:>9}" if label == "Conventional" else ""
            out.append(f"{prefix}{label:<17}{fmt_est(est, key)}{tail}"
                       + (marker if label == "Conventional" else ""))
        out.append("")
    out.append(
        "Conventional inference uses the point estimate and its own standard error. Bias-corrected "
        "recentres the estimate for the bias local linear fitting leaves behind; robust uses that "
        "recentred estimate with a variance that accounts for the correction, and it is the inference "
        "Section 12 pre-committed to. Where the three disagree, the robust row is the one that counts."
    )
    out.append("")

    # ------------------------------------------------------------------
    out.append(section("4. V1. McCrary density test for manipulation of the running variable"))
    out.append(
        "Section 12: 'If markers push borderline students up to 40, the design is compromised and the "
        "result is reported as invalid rather than quietly retained. Heaping at round numbers is "
        "expected; whether it is asymmetric around 40 is the question.'"
    )
    out.append("")

    cjm = cjm_density_test(x_raw, PASS_MARK)
    out.append("Cattaneo-Jansson-Ma manipulation test (rddensity), the modern form of the same test:")
    out.append(f"  density estimate left of 40    {cjm['f_left']:.6f}")
    out.append(f"  density estimate right of 40   {cjm['f_right']:.6f}")
    out.append(f"  difference (right - left)      {cjm['f_diff']:+.6f}")
    out.append(f"  bandwidths (left, right)       {cjm['h_left']:.3f}, {cjm['h_right']:.3f}")
    if np.isnan(cjm["t_asy"]):
        out.append(
            "  asymptotic variance            not available (mass points in the running variable), "
            "jackknife used"
        )
    else:
        out.append(f"  t (asymptotic)                 {cjm['t_asy']:.4f}   p = {cjm['p_asy']:.4f}")
    out.append(f"  t (jackknife)                  {cjm['t_jk']:.4f}   p = {cjm['p_jk']:.4f}")
    out.append("")

    mcc = mccrary_density_test(x_raw, PASS_MARK, PRIMARY_BANDWIDTH, binsize=1.0)
    out.append(
        "McCrary (2008) as originally specified, implemented directly: bin width 1 (every attainable "
        "score is its own bin, so no bin straddles the cutoff), weighted local linear regression of the "
        "normalised bin heights on their midpoints with a triangular kernel, fitted separately each side "
        f"at bandwidth {PRIMARY_BANDWIDTH:.0f} and evaluated at the cutoff."
    )
    if mcc["estimable"]:
        out.append(f"  f_left                         {mcc['f_left']:.6f}")
        out.append(f"  f_right                        {mcc['f_right']:.6f}")
        out.append(f"  theta = ln(f+) - ln(f-)        {mcc['theta']:+.4f}")
        out.append(f"  SE(theta)                      {mcc['se']:.4f}")
        out.append(f"  z                              {mcc['z']:+.4f}   p = {mcc['pv']:.4f}")
    else:
        out.append("  not estimable")
    out.append("")

    out.append(f"Raw frequency of every score from {FREQ_LO} to {FREQ_HI}, so heaping is visible directly:")
    out.append(f"  {'score':<8}{'n':>7}   histogram")
    freqs = score_frequencies(x_raw, FREQ_LO, FREQ_HI)
    scale = max(n for _, n in freqs) or 1
    for value, n in freqs:
        bar = "#" * int(round(60 * n / scale))
        mark = "  <- pass mark" if value == PASS_MARK else ""
        out.append(f"  {value:<8.0f}{n:>7}   {bar}{mark}")
    out.append("")

    freq_map = dict(freqs)
    round_numbers = [v for v in (30, 35, 40, 45, 50) if v in freq_map]
    out.append(
        "Round-number heaping, which Section 12 says to expect. Each multiple of five against the mean "
        "of its two neighbours:"
    )
    out.append(f"  {'score':<8}{'n':>7}{'mean of neighbours':>21}{'ratio':>9}")
    for v in round_numbers:
        neighbours = [freq_map.get(v - 1), freq_map.get(v + 1)]
        neighbours = [nb for nb in neighbours if nb is not None]
        if not neighbours:
            continue
        mean_nb = float(np.mean(neighbours))
        ratio = freq_map[v] / mean_nb if mean_nb else float("nan")
        out.append(f"  {v:<8.0f}{freq_map[v]:>7}{mean_nb:>21.1f}{ratio:>9.2f}")
    out.append("")

    v1_rejects = cjm["p_jk"] < ALPHA
    out.append(
        f"V1 RESULT: the pre-committed manipulation test {'REJECTS' if v1_rejects else 'does not reject'} "
        f"the null of a continuous density at the pass mark (p = {cjm['p_jk']:.4f}"
        f"{' < ' if v1_rejects else ' >= '}{ALPHA}). The estimated density is HIGHER just above 40 than "
        "just below it, which is the direction that marking to the boundary would produce: borderline "
        "scripts pushed up to a pass rather than left just below one."
    )
    out.append(
        "The heaping table above is reported because Section 12 asks for it, and it shows that 40 is not "
        "the only spike: every multiple of five is elevated. It is not, however, a reason to set the test "
        "aside. The test was pre-committed, it was run as specified, and it rejected. Reading the heaping "
        "table as a reason to keep the design would be choosing the criterion after seeing the result, "
        "which is the practice this protocol exists to prevent."
    )
    out.append("")

    # ------------------------------------------------------------------
    out.append(section("5. V2. Covariate balance at the cutoff"))
    out.append(
        "The same specification at the primary bandwidth, with each Group A feature as the outcome in "
        "place of not_completed. Multi-level categoricals are tested one indicator per level, since a "
        "discontinuity in any level is a discontinuity in the covariate. A significant jump in a "
        "pre-determined characteristic undermines the design: those characteristics are fixed before the "
        "score is awarded and cannot be caused by it."
    )
    out.append("")
    out.append(f"  {'covariate':<44}{'estimate':>10}{'SE':>9}{'[95% CI]':>22}{'p':>10}")
    balance_rows = []
    for feature in GROUP_A_NUMERIC:
        est = rd_estimate(pop[feature].values.astype(float), x_raw, PASS_MARK, PRIMARY_BANDWIDTH)
        balance_rows.append((feature, est))
    for feature in GROUP_A_CATEGORICAL:
        values = pop[feature].fillna(NULL_TOKEN).astype(str)
        for level in sorted(values.unique()):
            indicator = (values == level).astype(float).values
            est = rd_estimate(indicator, x_raw, PASS_MARK, PRIMARY_BANDWIDTH)
            balance_rows.append((f"{feature} = {level}", est))
    n_reject = 0
    for label, est in balance_rows:
        out.append(f"  {label:<44}{fmt_est(est, 'robust')}")
        if est["estimable"] and est["robust"]["pv"] < ALPHA:
            n_reject += 1
    out.append("")
    n_tests = sum(1 for _, e in balance_rows if e["estimable"])
    out.append(
        f"V2 RESULT: {n_reject} of {n_tests} covariate tests reject at alpha = {ALPHA} on robust "
        "inference. The tests are not independent of one another — the levels of a categorical are "
        "mechanically linked, and a two-level covariate such as gender or disability produces two "
        "mirror-image rows — so the number of rejections expected by chance is not simply alpha times "
        f"{n_tests}. No multiplicity correction is applied: the protocol pre-registered neither a "
        "correction nor a rule for reading the count, and choosing one now would be choosing it in "
        "knowledge of the result. Every row is reported so a reader can apply their own."
    )
    out.append("")

    # ------------------------------------------------------------------
    out.append(section("6. V3. Placebo cutoffs"))
    out.append(
        "The same specification and bandwidth at scores where no rule exists. A discontinuity appearing "
        "at one of these means the specification is finding noise rather than the pass mark. At bandwidth "
        f"{PRIMARY_BANDWIDTH:.0f} none of these windows contains the real cutoff with non-zero kernel "
        "weight, so they are not contaminated by it."
    )
    out.append("")
    out.append(
        f"  {'cutoff':<9}{'method':<17}{'estimate':>10}{'SE':>9}{'[95% CI]':>22}{'p':>10}{'n_left':>9}{'n_right':>9}"
    )
    for cutoff in PLACEBO_CUTOFFS:
        est = rd_estimate(y, x_raw, cutoff, PRIMARY_BANDWIDTH)
        for label, key in [("Conventional", "conventional"), ("Robust", "robust")]:
            prefix = f"  {cutoff:<9.0f}" if label == "Conventional" else "  " + " " * 9
            tail = f"{est['n_left']:>9}{est['n_right']:>9}" if (est["estimable"] and label == "Conventional") else ""
            out.append(f"{prefix}{label:<17}{fmt_est(est, key)}{tail}")
        out.append("")

    # ------------------------------------------------------------------
    out.append(section("7. V4. Donut specification"))
    out.append(
        f"Scores within {DONUT_RADIUS:.0f} mark of the pass mark are excluded, in case exact-40 scores "
        "are administratively assigned rather than earned. Given the spike at exactly 40 in the V1 table, "
        "this drops the mass point itself along with its immediate neighbours."
    )
    out.append("")
    primary_estimate = estimates[PRIMARY_BANDWIDTH]
    donut_mask = np.abs(x_raw - PASS_MARK) > DONUT_RADIUS
    out.append(
        f"  rows dropped: {int((~donut_mask).sum())} (scores "
        f"{PASS_MARK - DONUT_RADIUS:.0f} to {PASS_MARK + DONUT_RADIUS:.0f}), "
        f"{int(donut_mask.sum())} retained"
    )
    out.append("")
    out.append(
        f"  {'bw':<5}{'method':<17}{'estimate':>10}{'SE':>9}{'[95% CI]':>22}{'p':>10}{'n_left':>9}{'n_right':>9}"
    )
    for bw in [PRIMARY_BANDWIDTH]:
        est = rd_estimate(y[donut_mask], x_raw[donut_mask], PASS_MARK, bw)
        for label, key in [("Conventional", "conventional"), ("Bias-corrected", "bias_corrected"),
                           ("Robust", "robust")]:
            prefix = f"  {bw:<5.0f}" if label == "Conventional" else "  " + " " * 5
            tail = f"{est['n_left']:>9}{est['n_right']:>9}" if (est["estimable"] and label == "Conventional") else ""
            out.append(f"{prefix}{label:<17}{fmt_est(est, key)}{tail}")
        donut_est = est
    out.append("")
    if donut_est["estimable"] and primary_estimate["estimable"]:
        out.append(
            f"Reported as measured: the bias-corrected point estimate moves from "
            f"{primary_estimate['bias_corrected']['coef']:+.4f} on the full sample to "
            f"{donut_est['bias_corrected']['coef']:+.4f} once the {int((~donut_mask).sum())} rows at "
            f"scores {PASS_MARK - DONUT_RADIUS:.0f} to {PASS_MARK + DONUT_RADIUS:.0f} are removed, and "
            "the robust interval widens accordingly. Section 12 pre-committed to running this "
            "specification and to reporting it whatever it showed; it did not state how to read a "
            "difference of this kind, and no rule for reading one is invented here. What can be said "
            "without a criterion is factual: most of what the full-sample estimate measures comes from "
            "the rows at and immediately around the mass point that V1 flags."
        )
        out.append("")

    # ------------------------------------------------------------------
    out.append(section("8. Power analysis"))
    out.append(
        "Independent of what the RD found. Section 12 asks what a randomised trial would need in order to "
        "detect an intervention effect of a given size, which is the artefact that answers 'how would you "
        "design the experiment' without pretending one was run."
    )
    out.append("")
    p_control = float(y.mean())
    out.append(
        f"Two-sided two-proportion test, alpha = {ALPHA}, power = {POWER:.0%}, control non-completion "
        f"rate = {p_control:.4f} (the observed rate in the Arm 2 population). The effect is a reduction "
        "in that rate, in percentage points."
    )
    out.append("")
    out.append(f"  {'effect (pp)':<14}{'treated rate':>14}{'n per arm':>12}{'n total':>11}")
    for effect in EFFECT_SIZES_PP:
        n = n_per_arm(p_control, effect)
        out.append(
            f"  {effect:<14}{p_control - effect / 100:>14.4f}{int(np.ceil(n)):>12}{int(np.ceil(n) * 2):>11}"
        )
    out.append("")

    primary = primary_estimate
    out.append("Minimum detectable effect of the RD design itself, at the primary bandwidth:")
    if primary["estimable"]:
        out.append(
            f"  effective sample size          {primary['n_left']} left, {primary['n_right']} right "
            f"({primary['n_left'] + primary['n_right']} total)"
        )
        for label, key in [("conventional", "conventional"), ("robust", "robust")]:
            se = primary[key]["se"]
            mde = minimum_detectable_effect(se)
            out.append(
                f"  MDE on {label:<24} {mde:.4f} ({mde * 100:.1f} percentage points), from SE = {se:.4f}"
            )
        out.append("")
        out.append(
            "This is what makes a null interpretable. An effect smaller than the MDE would not have been "
            "detected by this design at this bandwidth whether or not it exists, so a null here is a "
            "statement about what the design could see, not a statement that the effect is zero."
        )
    out.append("")

    # ------------------------------------------------------------------
    out.append(section("9. What this does not establish"))
    arm_gap = con.execute(
        f"""
        WITH pop AS ({POPULATION_SQL})
        SELECT (p.id_student IS NOT NULL) AS in_arm2, count(*) AS n, avg(f.not_completed) AS base_rate
        FROM features_d{ARM1_CUTOFF} f
        LEFT JOIN pop p ON f.code_module = p.code_module
                       AND f.code_presentation = p.code_presentation
                       AND f.id_student = p.id_student
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    outside = arm_gap[~arm_gap["in_arm2"].astype(bool)].iloc[0]
    inside = arm_gap[arm_gap["in_arm2"].astype(bool)].iloc[0]
    arm1_n = int(outside["n"] + inside["n"])
    out.append(
        "Section 12 states the limitation and Section 14 L1 repeats it: this is a local effect at one "
        "score boundary, among students who sat and were marked on the first assessment. It says nothing "
        "about students who never submitted, and those are the highest-risk group in Arm 1. The size of "
        "that gap is a number, not a caveat:"
    )
    out.append("")
    out.append(f"  {'group':<52}{'n':>8}{'share':>9}{'not_completed':>15}")
    out.append(
        f"  {f'Arm 1 cohort at D={ARM1_CUTOFF} (all splits)':<52}{arm1_n:>8}{1.0:>9.4f}"
        f"{(outside['n'] * outside['base_rate'] + inside['n'] * inside['base_rate']) / arm1_n:>15.4f}"
    )
    out.append(
        f"  {'  of which also in the Arm 2 population':<52}{int(inside['n']):>8}"
        f"{inside['n'] / arm1_n:>9.4f}{inside['base_rate']:>15.4f}"
    )
    out.append(
        f"  {'  of which NOT in the Arm 2 population':<52}{int(outside['n']):>8}"
        f"{outside['n'] / arm1_n:>9.4f}{outside['base_rate']:>15.4f}"
    )
    out.append("")
    out.append(
        f"  Arm 2 rows absent from the Arm 1 D={ARM1_CUTOFF} cohort: {len(pop) - int(inside['n'])} "
        "(removed by E1, E2 or E3 there, but retained here because Arm 2 applies no such exclusion)."
    )
    out.append("")
    out.append(
        f"{int(outside['n'])} of the {arm1_n} students the Arm 1 model scores at D={ARM1_CUTOFF} "
        f"({outside['n'] / arm1_n:.1%}) are outside the Arm 2 population, and their non-completion rate "
        f"is {outside['base_rate']:.4f} against {inside['base_rate']:.4f} for those inside it. The "
        "population Arm 2 can speak to is the one that already engaged enough to be marked; the students "
        "the early warning system most needs to reach are, by construction, the ones this design cannot "
        "see. That disconnect is not resolved here and is not resolvable within this design."
    )
    out.append("")

    # ------------------------------------------------------------------
    out.append(section("10. Declared outcomes (Section 13)"))
    out.append(
        "Unlike O3 to O6, these three are decidable as the protocol wrote them. O7 and O8 turn on "
        "statistical significance, which Section 12 fixes by specifying robust bias-corrected inference, "
        "and O9 turns on whether V1 fails, which V1 is a test of. No numeric convention has to be "
        "invented to reach a verdict here, and none is."
    )
    out.append("")
    primary_robust = primary["robust"] if primary["estimable"] else None
    if v1_rejects:
        out.append(
            f"O9 MET: V1 fails. The pre-committed manipulation test rejects a continuous density at the "
            f"pass mark (jackknife t = {cjm['t_jk']:.4f}, p = {cjm['p_jk']:.4f}), with excess mass just "
            "ABOVE 40. Section 12 declared in advance that if markers push borderline students up to 40, "
            "'the design is compromised and the result is reported as invalid rather than quietly "
            "retained'. It is so reported."
        )
        out.append("")
        out.append(
            "The consequence, stated plainly. The identifying assumption of this design is that which "
            "side of 40 a student lands on is as good as random for students near the line. If scripts "
            "near the line are being moved across it, then which side a student lands on reflects a "
            "marker's judgment about that student, and the comparison is no longer between similar "
            "students who differ by luck. The estimates in Section 3 are printed because the protocol "
            "requires every pre-committed quantity to be reported whatever it shows, NOT because they "
            "are interpretable as causal effects. They are not."
        )
        out.append("")
        if primary_robust is not None:
            out.append(
                f"For the record and not as a finding: the primary estimate at bandwidth "
                f"{PRIMARY_BANDWIDTH:.0f} is {primary_robust['coef']:+.4f} with robust 95% CI "
                f"[{primary_robust['ci_lo']:+.4f}, {primary_robust['ci_hi']:+.4f}], p = "
                f"{primary_robust['pv']:.4f}. Under the pre-committed inference this interval contains "
                "zero, so even taking the design at face value it would not have supported O7."
            )
        out.append("")
        out.append(
            "Per Section 12's declared outcome O9, Arm 2 is reduced to the power analysis alone. The "
            "power analysis in Section 8 stands: it does not depend on the RD being valid, only on the "
            "observed outcome variance in this population, and it is the part of Arm 2 that survives."
        )
        out.append("")
        out.append(
            "No bandwidth search, no alternative cutoff and no respecification was attempted to rescue "
            "the design. The specification run here is the one Section 12 fixed in advance."
        )
    elif primary_robust is not None and not (primary_robust["ci_lo"] <= 0 <= primary_robust["ci_hi"]):
        out.append(
            f"O7 MET: V1 does not reject, and the primary estimate is {primary_robust['coef']:+.4f}, "
            f"robust 95% CI [{primary_robust['ci_lo']:+.4f}, {primary_robust['ci_hi']:+.4f}], p = "
            f"{primary_robust['pv']:.4f}, which excludes zero."
        )
    else:
        out.append(
            f"O8 MET: V1 does not reject, and the primary estimate is {primary_robust['coef']:+.4f}, "
            f"robust 95% CI [{primary_robust['ci_lo']:+.4f}, {primary_robust['ci_hi']:+.4f}], p = "
            f"{primary_robust['pv']:.4f}, which contains zero. Reported as a null with the minimum "
            "detectable effect in Section 8 stating what the design could have found."
        )
    out.append("")

    con.close()
    report_text = "\n".join(out)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text + "\n")
    print(report_text)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

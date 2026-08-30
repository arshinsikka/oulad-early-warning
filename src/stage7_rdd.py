"""
Stage 7 computation: Arm 2, regression discontinuity at the assessment pass
mark, per PROTOCOL.md Section 12.

Arm 2 is independent of Arm 1. No model artefact, frozen threshold or test
prediction is loaded anywhere in this module or its runner. The only inputs
are the raw relational views built at Stage 2.
"""

import numpy as np
import scipy.stats as st

PASS_MARK = 40.0
PRIMARY_BANDWIDTH = 10.0
ROBUSTNESS_BANDWIDTHS = [5.0, 8.0, 15.0, 20.0]
PLACEBO_CUTOFFS = [30.0, 50.0, 60.0]
DONUT_RADIUS = 1.0
ALPHA = 0.05
POWER = 0.80

# The first TMA of each module-presentation, by due date, with id_assessment
# as a deterministic tie-break. Section 12 fixes the running variable as the
# score on the first assessed piece of work; Amendment A1 excludes banked
# scores, which are carried over from a previous presentation and are not
# evidence of work done in this one.
POPULATION_SQL = """
WITH first_tma AS (
    SELECT
        code_module,
        code_presentation,
        id_assessment,
        date AS due_date,
        row_number() OVER (
            PARTITION BY code_module, code_presentation
            ORDER BY date ASC NULLS LAST, id_assessment ASC
        ) AS rn
    FROM v_assessments
    WHERE assessment_type = 'TMA'
),
ft AS (SELECT * FROM first_tma WHERE rn = 1)
SELECT
    ft.code_module,
    ft.code_presentation,
    ft.id_assessment,
    ft.due_date,
    sa.id_student,
    sa.score,
    sa.date_submitted,
    si.gender,
    si.region,
    si.highest_education,
    si.imd_band,
    si.age_band,
    si.disability,
    si.num_of_prev_attempts,
    si.studied_credits,
    si.final_result,
    CASE WHEN si.final_result IN ('Fail', 'Withdrawn') THEN 1 ELSE 0 END AS not_completed
FROM ft
JOIN v_student_assessment sa USING (id_assessment)
JOIN v_student_info si
  ON si.code_module = ft.code_module
 AND si.code_presentation = ft.code_presentation
 AND si.id_student = sa.id_student
WHERE sa.score IS NOT NULL
  AND sa.is_banked = 0
"""


def rd_estimate(y: np.ndarray, x: np.ndarray, cutoff: float, bandwidth: float) -> dict:
    """Local linear RD with a triangular kernel and separate slopes each side,
    with robust bias-corrected inference (Calonico, Cattaneo, Titiunik) as
    Section 12 specifies. x is the raw running variable; the cutoff is passed
    through rather than pre-centred so placebo cutoffs use the same path.

    Returns None-valued fields rather than raising when a specification is not
    estimable (a covariate constant inside the window, say), so that every
    pre-committed check still appears in the report.
    """
    from rdrobust import rdrobust

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    keep = ~(np.isnan(y) | np.isnan(x))
    try:
        out = rdrobust(y=y[keep], x=x[keep], c=cutoff, h=bandwidth, kernel="triangular", p=1)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return {"estimable": False, "error": f"{type(exc).__name__}: {exc}"}

    def row(label: str) -> dict:
        return {
            "coef": float(out.coef.loc[label].iloc[0]),
            "se": float(out.se.loc[label].iloc[0]),
            "ci_lo": float(out.ci.loc[label].iloc[0]),
            "ci_hi": float(out.ci.loc[label].iloc[1]),
            "pv": float(out.pv.loc[label].iloc[0]),
        }

    return {
        "estimable": True,
        "conventional": row("Conventional"),
        "bias_corrected": row("Bias-Corrected"),
        "robust": row("Robust"),
        "n_left": int(out.N_h[0]),
        "n_right": int(out.N_h[1]),
        "n_total_used": int(keep.sum()),
    }


def mccrary_density_test(x: np.ndarray, cutoff: float, bandwidth: float,
                         binsize: float = 1.0) -> dict:
    """McCrary (2008), implemented directly.

    Step 1: histogram with bins that do not straddle the cutoff. Scores here
    are integers, so a bin width of 1 puts each attainable score in its own
    bin and no bin can span the discontinuity.

    Step 2: weighted local linear regression of the normalised bin heights on
    the bin midpoints, triangular kernel, fitted separately each side and
    evaluated at the cutoff.

    theta = ln(f_right) - ln(f_left), with McCrary's variance
    Var(theta) = (1 / (n*h)) * (24/5) * (1/f_right + 1/f_left).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    centred = x - cutoff

    lo = np.floor(centred.min() / binsize) * binsize
    hi = np.ceil(centred.max() / binsize) * binsize
    edges = np.arange(lo, hi + binsize, binsize)
    # An edge exactly at the cutoff guarantees no bin straddles it.
    counts, _ = np.histogram(centred, bins=edges)
    midpoints = edges[:-1] + binsize / 2.0
    heights = counts / (n * binsize)

    def side_intercept(mask: np.ndarray) -> float | None:
        m, f = midpoints[mask], heights[mask]
        w = np.clip(1.0 - np.abs(m) / bandwidth, 0.0, None)
        keep = w > 0
        if keep.sum() < 2:
            return None
        m, f, w = m[keep], f[keep], w[keep]
        design = np.column_stack([np.ones_like(m), m])
        wls = np.linalg.lstsq(design * np.sqrt(w)[:, None], f * np.sqrt(w), rcond=None)[0]
        return float(wls[0])

    f_left = side_intercept(midpoints < 0)
    f_right = side_intercept(midpoints >= 0)
    if not f_left or not f_right or f_left <= 0 or f_right <= 0:
        return {"estimable": False, "f_left": f_left, "f_right": f_right}

    theta = float(np.log(f_right) - np.log(f_left))
    variance = (1.0 / (n * bandwidth)) * (24.0 / 5.0) * (1.0 / f_right + 1.0 / f_left)
    se = float(np.sqrt(variance))
    z = theta / se
    return {
        "estimable": True,
        "f_left": f_left, "f_right": f_right,
        "theta": theta, "se": se, "z": z,
        "pv": float(2 * (1 - st.norm.cdf(abs(z)))),
        "n": n, "binsize": binsize, "bandwidth": bandwidth,
    }


def cjm_density_test(x: np.ndarray, cutoff: float) -> dict:
    """The Cattaneo, Jansson and Ma manipulation test, the modern
    implementation of the same question McCrary asked. Reported alongside the
    original because the running variable has mass points at every integer,
    which is the case CJM's local polynomial density estimator handles and
    McCrary's binned estimator does not."""
    from rddensity import rddensity

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    out = rddensity(X=x, c=cutoff)
    test, hat = out.test, out.hat
    return {
        "t_asy": float(test["t_asy"]), "p_asy": float(test["p_asy"]),
        "t_jk": float(test["t_jk"]), "p_jk": float(test["p_jk"]),
        "f_left": float(hat["left"]), "f_right": float(hat["right"]),
        "f_diff": float(hat["diff"]),
        "h_left": float(out.h["left"]), "h_right": float(out.h["right"]),
        "n": int(out.n["full"]),
    }


def score_frequencies(x: np.ndarray, lo: float, hi: float) -> list[tuple[float, int]]:
    x = np.asarray(x, dtype=float)
    values = np.arange(lo, hi + 1)
    return [(float(v), int((x == v).sum())) for v in values]


def n_per_arm(p_control: float, effect_pp: float, alpha: float = ALPHA,
              power: float = POWER) -> float:
    """Sample size per arm for a two-sided two-proportion test.

    n = (z_{alpha/2} + z_{power})^2 * (p1(1-p1) + p2(1-p2)) / (p1 - p2)^2

    The effect is a reduction in the non-completion rate, in percentage
    points, so p2 = p1 - effect.
    """
    delta = effect_pp / 100.0
    p2 = p_control - delta
    if p2 <= 0 or p2 >= 1 or delta <= 0:
        return float("nan")
    z_a = st.norm.ppf(1 - alpha / 2)
    z_b = st.norm.ppf(power)
    var = p_control * (1 - p_control) + p2 * (1 - p2)
    return float((z_a + z_b) ** 2 * var / delta ** 2)


def minimum_detectable_effect(se: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """The smallest true effect this design would reject the null for, with
    the stated power: MDE = (z_{alpha/2} + z_{power}) * SE."""
    return float((st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)) * se)

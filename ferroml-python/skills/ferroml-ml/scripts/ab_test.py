"""A/B testing: experiment design and statistical analysis.

Usage: Claude runs design() before an experiment, analyze() after collecting data.
Output: Required sample size, significance tests, effect sizes, and recommendations.
"""
from __future__ import annotations

import numpy as np


def design(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    daily_traffic: int | None = None,
) -> dict:
    """Design an A/B test: compute required sample size.

    Parameters
    ----------
    baseline_rate : float
        Current metric value (e.g. 0.05 for 5% conversion).
    mde : float
        Minimum detectable effect as absolute change (e.g. 0.01 for +1pp).
    alpha : float
        Significance level (default 0.05).
    power : float
        Statistical power (default 0.8).
    daily_traffic : int or None
        Daily visitors per group (for duration estimate).

    Returns
    -------
    dict with sample_size_per_group, total_samples, duration_estimate_days
    """
    from ferroml.stats import sample_size_for_power

    # Effect size (Cohen's h for proportions)
    # d = |target - baseline| / pooled_std
    target_rate = baseline_rate + mde
    pooled_std = np.sqrt(baseline_rate * (1 - baseline_rate))

    if pooled_std < 1e-12:
        return {
            "error": "Baseline rate is 0 or 1; cannot compute sample size",
            "baseline_rate": baseline_rate,
            "mde": mde,
        }

    effect_size = abs(mde) / pooled_std

    # Use ferroml to compute sample size
    try:
        n_per_group = int(sample_size_for_power(effect_size, alpha=alpha, power=power))
    except Exception:
        # Fallback: manual calculation using normal approximation
        # n = (z_alpha/2 + z_beta)^2 / d^2
        z_alpha = _z_score(1 - alpha / 2)
        z_beta = _z_score(power)
        n_per_group = int(np.ceil((z_alpha + z_beta) ** 2 / effect_size ** 2))

    total_samples = n_per_group * 2
    duration_days = None
    if daily_traffic is not None and daily_traffic > 0:
        duration_days = int(np.ceil(n_per_group / daily_traffic))

    return {
        "baseline_rate": baseline_rate,
        "target_rate": round(target_rate, 6),
        "mde": mde,
        "effect_size": round(effect_size, 4),
        "alpha": alpha,
        "power": power,
        "sample_size_per_group": n_per_group,
        "total_samples": total_samples,
        "duration_estimate_days": duration_days,
        "daily_traffic": daily_traffic,
    }


def analyze(
    group_a: np.ndarray,
    group_b: np.ndarray,
    metric: str = "conversion",
    alpha: float = 0.05,
) -> dict:
    """Analyze A/B test results with statistical tests.

    Parameters
    ----------
    group_a : np.ndarray
        Outcomes for control group (0/1 for conversion, continuous for revenue).
    group_b : np.ndarray
        Outcomes for treatment group.
    metric : str
        "conversion" (binary) or "continuous" (e.g. revenue).
    alpha : float
        Significance level.

    Returns
    -------
    dict with means, difference, p_value, significant, effect_size, confidence_interval, recommendation
    """
    from ferroml.stats import bootstrap_ci, cohens_d, welch_ttest

    group_a = np.asarray(group_a, dtype=np.float64)
    group_b = np.asarray(group_b, dtype=np.float64)

    mean_a = float(np.mean(group_a))
    mean_b = float(np.mean(group_b))
    difference = mean_b - mean_a
    relative_lift = difference / mean_a if abs(mean_a) > 1e-12 else float("inf")

    # Statistical test
    if metric == "conversion":
        # Z-test for proportions
        n_a, n_b = len(group_a), len(group_b)
        p_pooled = (np.sum(group_a) + np.sum(group_b)) / (n_a + n_b)

        if p_pooled > 0 and p_pooled < 1:
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_a + 1 / n_b))
            z_stat = difference / se if se > 1e-12 else 0.0
            # Two-tailed p-value using normal approximation
            p_value = 2 * (1 - _normal_cdf(abs(z_stat)))
        else:
            z_stat = 0.0
            p_value = 1.0

        test_method = "z-test (proportions)"
    else:
        # Welch's t-test for continuous metrics
        try:
            result = welch_ttest(group_a, group_b)
            if isinstance(result, dict):
                p_value = float(result["p_value"])
                z_stat = float(result["statistic"])
            else:
                z_stat = float(result[0])
                p_value = float(result[1])
        except Exception:
            # Fallback manual Welch's t-test
            var_a = float(np.var(group_a, ddof=1))
            var_b = float(np.var(group_b, ddof=1))
            se = np.sqrt(var_a / len(group_a) + var_b / len(group_b))
            z_stat = difference / se if se > 1e-12 else 0.0
            p_value = 2 * (1 - _normal_cdf(abs(z_stat)))

        test_method = "Welch's t-test"

    significant = bool(p_value < alpha)

    # Effect size (Cohen's d)
    try:
        d_result = cohens_d(group_a, group_b)
        effect_size = float(d_result["d"]) if isinstance(d_result, dict) else float(d_result)
    except Exception:
        pooled_std = np.sqrt(
            ((len(group_a) - 1) * np.var(group_a, ddof=1) +
             (len(group_b) - 1) * np.var(group_b, ddof=1)) /
            (len(group_a) + len(group_b) - 2)
        )
        effect_size = difference / pooled_std if pooled_std > 1e-12 else 0.0

    # Confidence interval via bootstrap
    try:
        ci = bootstrap_ci(group_b - np.mean(group_a), confidence=1 - alpha)
        if isinstance(ci, dict):
            ci_result = (float(ci["lower"]), float(ci["upper"]))
        elif hasattr(ci, "__len__") and len(ci) >= 2:
            ci_result = (float(ci[0]), float(ci[1]))
        else:
            ci_result = _manual_ci(group_a, group_b, alpha)
    except Exception:
        ci_result = _manual_ci(group_a, group_b, alpha)

    # Recommendation
    if not significant:
        recommendation = (
            f"No statistically significant difference (p={p_value:.4f} > {alpha}). "
            f"Need more data or larger effect to reach significance."
        )
    elif difference > 0:
        recommendation = (
            f"Treatment (B) is significantly better (p={p_value:.4f}, "
            f"lift={relative_lift:+.2%}). Recommend rolling out treatment."
        )
    else:
        recommendation = (
            f"Treatment (B) is significantly worse (p={p_value:.4f}, "
            f"lift={relative_lift:+.2%}). Recommend keeping control."
        )

    # Effect size interpretation
    if abs(effect_size) < 0.2:
        effect_label = "negligible"
    elif abs(effect_size) < 0.5:
        effect_label = "small"
    elif abs(effect_size) < 0.8:
        effect_label = "medium"
    else:
        effect_label = "large"

    return {
        "group_a_mean": round(mean_a, 6),
        "group_b_mean": round(mean_b, 6),
        "difference": round(difference, 6),
        "relative_lift": round(relative_lift, 6),
        "test_method": test_method,
        "test_statistic": round(float(z_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": significant,
        "alpha": alpha,
        "effect_size": round(effect_size, 4),
        "effect_label": effect_label,
        "confidence_interval": (round(ci_result[0], 6), round(ci_result[1], 6)),
        "n_a": len(group_a),
        "n_b": len(group_b),
        "recommendation": recommendation,
    }


def _z_score(p: float) -> float:
    """Approximate inverse normal CDF (percent point function).

    Uses Abramowitz & Stegun rational approximation.
    """
    if p <= 0 or p >= 1:
        return 0.0
    if p < 0.5:
        return -_z_score(1 - p)

    t = np.sqrt(-2 * np.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function."""
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _erf(x: float) -> float:
    """Approximate error function (Horner form of Abramowitz & Stegun 7.1.26)."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592
    ) * t * np.exp(-x * x)
    return sign * y


def _manual_ci(
    group_a: np.ndarray, group_b: np.ndarray, alpha: float
) -> tuple[float, float]:
    """Manual confidence interval for difference in means."""
    diff = float(np.mean(group_b)) - float(np.mean(group_a))
    se = np.sqrt(
        np.var(group_a, ddof=1) / len(group_a)
        + np.var(group_b, ddof=1) / len(group_b)
    )
    z = _z_score(1 - alpha / 2)
    return (round(diff - z * se, 6), round(diff + z * se, 6))


def print_design(result: dict) -> None:
    """Print human-readable experiment design summary."""
    print(f"\n{'='*60}")
    print(f"A/B TEST DESIGN")
    print(f"{'='*60}")
    if "error" in result:
        print(f"  Error: {result['error']}")
        return
    print(f"  Baseline rate:       {result['baseline_rate']:.4f}")
    print(f"  Target rate:         {result['target_rate']:.4f}")
    print(f"  MDE:                 {result['mde']:+.4f}")
    print(f"  Effect size:         {result['effect_size']:.4f}")
    print(f"  Alpha:               {result['alpha']}")
    print(f"  Power:               {result['power']}")
    print(f"  Sample per group:    {result['sample_size_per_group']:,}")
    print(f"  Total samples:       {result['total_samples']:,}")
    if result["duration_estimate_days"] is not None:
        print(f"  Estimated duration:  {result['duration_estimate_days']} days "
              f"({result['daily_traffic']:,}/day/group)")
    print()


def print_analysis(result: dict) -> None:
    """Print human-readable A/B test results."""
    print(f"\n{'='*60}")
    print(f"A/B TEST RESULTS  |  {'SIGNIFICANT' if result['significant'] else 'NOT SIGNIFICANT'}")
    print(f"{'='*60}")
    print(f"  Group A (control):   {result['group_a_mean']:.6f}  (n={result['n_a']:,})")
    print(f"  Group B (treatment): {result['group_b_mean']:.6f}  (n={result['n_b']:,})")
    print(f"  Difference:          {result['difference']:+.6f}")
    print(f"  Relative lift:       {result['relative_lift']:+.2%}")
    print(f"\n  Test:                {result['test_method']}")
    print(f"  Statistic:           {result['test_statistic']:.4f}")
    print(f"  p-value:             {result['p_value']:.6f}")
    print(f"  Effect size:         {result['effect_size']:.4f} ({result['effect_label']})")
    print(f"  95% CI:              [{result['confidence_interval'][0]:.6f}, {result['confidence_interval'][1]:.6f}]")
    print(f"\n  {result['recommendation']}")
    print()

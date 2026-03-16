"""FerroML Statistical Testing -- hypothesis tests, effect sizes, CIs, bootstrap, power analysis.

This module provides scipy.stats-like functions for rigorous statistical inference.
Every test includes effect sizes, confidence intervals, and (where applicable) power analysis.

Functions
---------
ttest_ind(x, y, equal_var=True)
    Independent samples t-test with effect size and CI.
welch_ttest(x, y)
    Welch's t-test (unequal variances).
mann_whitney(x, y)
    Mann-Whitney U test (non-parametric).
cohens_d(x, y)
    Cohen's d effect size with CI.
hedges_g(x, y)
    Hedges' g (bias-corrected Cohen's d) with CI.
confidence_interval(data, level=0.95, method="t")
    Confidence interval for the mean.
bootstrap_ci(data, n_bootstrap=1000, confidence=0.95, seed=None)
    Bootstrap confidence interval (percentile + BCa).
adjust_pvalues(p_values, method="benjamini_hochberg", alpha=0.05)
    Multiple testing correction.
sample_size_for_power(effect_size, alpha=0.05, power=0.8)
    Required sample size per group for desired power.
power_for_sample_size(n, effect_size, alpha=0.05)
    Statistical power for given sample size.
durbin_watson(residuals)
    Durbin-Watson statistic for autocorrelation.
descriptive_stats(data)
    Descriptive statistics (mean, std, skewness, kurtosis, etc.).
normality_test(data)
    Normality test (skewness-kurtosis omnibus).
correlation(x, y, confidence=0.95)
    Pearson correlation with CI and significance test.
"""

from ferroml.ferroml import stats as _stats

# Hypothesis tests
ttest_ind = _stats.ttest_ind
welch_ttest = _stats.welch_ttest
mann_whitney = _stats.mann_whitney

# Effect sizes
cohens_d = _stats.cohens_d
hedges_g = _stats.hedges_g

# Confidence intervals
confidence_interval = _stats.confidence_interval
bootstrap_ci = _stats.bootstrap_ci

# Multiple testing
adjust_pvalues = _stats.adjust_pvalues

# Power analysis
sample_size_for_power = _stats.sample_size_for_power
power_for_sample_size = _stats.power_for_sample_size

# Diagnostics
durbin_watson = _stats.durbin_watson
descriptive_stats = _stats.descriptive_stats
normality_test = _stats.normality_test

# Correlation
correlation = _stats.correlation

__all__ = [
    "ttest_ind",
    "welch_ttest",
    "mann_whitney",
    "cohens_d",
    "hedges_g",
    "confidence_interval",
    "bootstrap_ci",
    "adjust_pvalues",
    "sample_size_for_power",
    "power_for_sample_size",
    "durbin_watson",
    "descriptive_stats",
    "normality_test",
    "correlation",
]

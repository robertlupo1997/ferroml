//! Python bindings for FerroML Statistical Testing
//!
//! This module exposes statistical functions as Python-callable functions
//! (not classes), following the scipy.stats API pattern:
//!
//! - Hypothesis tests: t-test, Welch's t-test, Mann-Whitney U
//! - Effect sizes: Cohen's d, Hedges' g
//! - Confidence intervals: parametric (t, z), bootstrap
//! - Multiple testing correction: Bonferroni, Holm, BH, BY
//! - Power analysis: sample size estimation, power calculation
//! - Diagnostics: Durbin-Watson, normality test, descriptive stats
//! - Correlation: Pearson r with CI and significance

use ferroml_core::stats::{
    self,
    bootstrap::Bootstrap,
    confidence::CIMethod,
    diagnostics::{NormalityTest, ShapiroWilkTest},
    effect_size::{CohensD, EffectSize, HedgesG},
    hypothesis::{HypothesisTest, TwoSampleTest},
    multiple_testing::MultipleTestingCorrection,
};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Convert PyReadonlyArray1 to owned ndarray Array1
fn to_array1(arr: PyReadonlyArray1<f64>) -> ndarray::Array1<f64> {
    arr.as_array().to_owned()
}

// =============================================================================
// Hypothesis Tests
// =============================================================================

/// Independent samples t-test with effect size and confidence interval.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First sample (1-D array of floats).
/// y : numpy.ndarray
///     Second sample (1-D array of floats).
/// equal_var : bool, optional (default=True)
///     If True, assumes equal population variances (Student's t-test).
///     If False, uses Welch's t-test (unequal variances).
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: statistic, p_value, df, effect_size, effect_interpretation,
///     effect_ci_lower, effect_ci_upper, ci_lower, ci_upper, power, test_name.
#[pyfunction]
#[pyo3(signature = (x, y, equal_var=true))]
fn ttest_ind(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    equal_var: bool,
) -> PyResult<PyObject> {
    let x_arr = to_array1(x);
    let y_arr = to_array1(y);

    let test = TwoSampleTest::t_test(x_arr, y_arr, equal_var);
    let result = test.test().map_err(crate::errors::ferro_to_pyerr)?;

    statistical_result_to_dict(py, &result)
}

/// Welch's t-test for two independent samples with unequal variances.
///
/// Equivalent to ``ttest_ind(x, y, equal_var=False)``.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First sample.
/// y : numpy.ndarray
///     Second sample.
///
/// Returns
/// -------
/// dict
///     Same format as ``ttest_ind``.
#[pyfunction]
#[pyo3(signature = (x, y))]
fn welch_ttest(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let x_arr = to_array1(x);
    let y_arr = to_array1(y);

    let test = TwoSampleTest::welch(x_arr, y_arr);
    let result = test.test().map_err(crate::errors::ferro_to_pyerr)?;

    statistical_result_to_dict(py, &result)
}

/// Mann-Whitney U test (non-parametric two-sample test).
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First sample.
/// y : numpy.ndarray
///     Second sample.
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: statistic, p_value, effect_size, effect_interpretation, test_name.
#[pyfunction]
#[pyo3(signature = (x, y))]
fn mann_whitney(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let x_arr = to_array1(x);
    let y_arr = to_array1(y);

    let test = TwoSampleTest::mann_whitney(x_arr, y_arr);
    let result = test.test().map_err(crate::errors::ferro_to_pyerr)?;

    statistical_result_to_dict(py, &result)
}

// =============================================================================
// Effect Sizes
// =============================================================================

/// Cohen's d effect size for two independent groups.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First group.
/// y : numpy.ndarray
///     Second group.
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: d, ci_lower, ci_upper, interpretation.
#[pyfunction]
#[pyo3(signature = (x, y))]
fn cohens_d(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let x_arr = to_array1(x);
    let y_arr = to_array1(y);

    let calc = CohensD::new(x_arr, y_arr);
    let result = calc.compute().map_err(crate::errors::ferro_to_pyerr)?;

    let dict = PyDict::new(py);
    dict.set_item("d", result.value)?;
    if let Some((lo, hi)) = result.ci {
        dict.set_item("ci_lower", lo)?;
        dict.set_item("ci_upper", hi)?;
    }
    dict.set_item("interpretation", &result.interpretation)?;
    Ok(dict.into())
}

/// Hedges' g effect size (bias-corrected Cohen's d).
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First group.
/// y : numpy.ndarray
///     Second group.
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: g, ci_lower, ci_upper, interpretation.
#[pyfunction]
#[pyo3(signature = (x, y))]
fn hedges_g(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let x_arr = to_array1(x);
    let y_arr = to_array1(y);

    let calc = HedgesG::new(x_arr, y_arr);
    let result = calc.compute().map_err(crate::errors::ferro_to_pyerr)?;

    let dict = PyDict::new(py);
    dict.set_item("g", result.value)?;
    if let Some((lo, hi)) = result.ci {
        dict.set_item("ci_lower", lo)?;
        dict.set_item("ci_upper", hi)?;
    }
    dict.set_item("interpretation", &result.interpretation)?;
    Ok(dict.into())
}

// =============================================================================
// Confidence Intervals
// =============================================================================

/// Confidence interval for the mean of a sample.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     Sample data (1-D array of floats).
/// level : float, optional (default=0.95)
///     Confidence level (e.g. 0.95 for 95% CI).
/// method : str, optional (default="t")
///     Method: "t" (Student's t), "z" (normal), or "bootstrap" (percentile bootstrap).
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: lower, upper, estimate, level, method.
#[pyfunction]
#[pyo3(signature = (data, level=0.95, method="t"))]
fn confidence_interval(
    py: Python<'_>,
    data: PyReadonlyArray1<f64>,
    level: f64,
    method: &str,
) -> PyResult<PyObject> {
    let data_arr = to_array1(data);

    let ci_method = match method {
        "t" => CIMethod::TDistribution,
        "z" | "normal" => CIMethod::Normal,
        "bootstrap" => CIMethod::BootstrapPercentile,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown method '{}'. Use 't', 'z', 'normal', or 'bootstrap'.",
                method
            )))
        }
    };

    let result = stats::confidence::confidence_interval(&data_arr, level, ci_method)
        .map_err(crate::errors::ferro_to_pyerr)?;

    let dict = PyDict::new(py);
    dict.set_item("lower", result.lower)?;
    dict.set_item("upper", result.upper)?;
    dict.set_item("estimate", result.estimate)?;
    dict.set_item("level", result.level)?;
    dict.set_item("method", method)?;
    Ok(dict.into())
}

// =============================================================================
// Bootstrap
// =============================================================================

/// Bootstrap confidence interval for the mean.
///
/// Computes percentile and BCa (bias-corrected and accelerated) confidence
/// intervals via bootstrap resampling.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     Sample data (1-D array of floats).
/// n_bootstrap : int, optional (default=1000)
///     Number of bootstrap resamples.
/// confidence : float, optional (default=0.95)
///     Confidence level.
/// seed : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: estimate, std_error, bias, ci_percentile_lower,
///     ci_percentile_upper, ci_bca_lower, ci_bca_upper.
#[pyfunction]
#[pyo3(signature = (data, n_bootstrap=1000, confidence=0.95, seed=None))]
fn bootstrap_ci(
    py: Python<'_>,
    data: PyReadonlyArray1<f64>,
    n_bootstrap: usize,
    confidence: f64,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    let data_arr = to_array1(data);

    let mut bootstrap = Bootstrap::new(n_bootstrap).with_confidence(confidence);
    if let Some(s) = seed {
        bootstrap = bootstrap.with_seed(s);
    }

    let result = bootstrap
        .mean(&data_arr)
        .map_err(crate::errors::ferro_to_pyerr)?;

    let dict = PyDict::new(py);
    dict.set_item("estimate", result.original)?;
    dict.set_item("std_error", result.std_error)?;
    dict.set_item("bias", result.bias)?;
    dict.set_item("ci_percentile_lower", result.ci_percentile.0)?;
    dict.set_item("ci_percentile_upper", result.ci_percentile.1)?;
    if let Some((lo, hi)) = result.ci_bca {
        dict.set_item("ci_bca_lower", lo)?;
        dict.set_item("ci_bca_upper", hi)?;
    } else {
        dict.set_item("ci_bca_lower", py.None())?;
        dict.set_item("ci_bca_upper", py.None())?;
    }
    Ok(dict.into())
}

// =============================================================================
// Multiple Testing
// =============================================================================

/// Adjust p-values for multiple testing.
///
/// Parameters
/// ----------
/// p_values : numpy.ndarray
///     Array of p-values to adjust.
/// method : str, optional (default="benjamini_hochberg")
///     Correction method. One of: "bonferroni", "holm", "hochberg",
///     "benjamini_hochberg" (or "bh", "fdr"), "benjamini_yekutieli" (or "by"), "none".
/// alpha : float, optional (default=0.05)
///     Significance level for rejection decisions.
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: adjusted (numpy array of adjusted p-values),
///     rejected (list of bools), method.
#[pyfunction]
#[pyo3(signature = (p_values, method="benjamini_hochberg", alpha=0.05))]
fn adjust_pvalues(
    py: Python<'_>,
    p_values: PyReadonlyArray1<f64>,
    method: &str,
    alpha: f64,
) -> PyResult<PyObject> {
    let pvals: Vec<f64> = p_values.as_array().to_vec();

    let correction = match method {
        "bonferroni" => MultipleTestingCorrection::Bonferroni,
        "holm" => MultipleTestingCorrection::Holm,
        "hochberg" => MultipleTestingCorrection::Hochberg,
        "benjamini_hochberg" | "bh" | "fdr" => MultipleTestingCorrection::BenjaminiHochberg,
        "benjamini_yekutieli" | "by" => MultipleTestingCorrection::BenjaminiYekutieli,
        "none" => MultipleTestingCorrection::None,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown method '{}'. Use 'bonferroni', 'holm', 'hochberg', 'benjamini_hochberg', 'benjamini_yekutieli', or 'none'.",
                method
            )))
        }
    };

    let result = stats::adjust_pvalues(&pvals, correction, alpha);

    let dict = PyDict::new(py);
    let adjusted_arr = ndarray::Array1::from_vec(result.adjusted);
    dict.set_item("adjusted", adjusted_arr.into_pyarray(py))?;
    dict.set_item("rejected", result.rejected)?;
    dict.set_item("method", method)?;
    Ok(dict.into())
}

// =============================================================================
// Power Analysis
// =============================================================================

/// Compute required sample size per group for desired statistical power.
///
/// Uses a two-sample t-test power formula.
///
/// Parameters
/// ----------
/// effect_size : float
///     Expected effect size (Cohen's d).
/// alpha : float, optional (default=0.05)
///     Significance level.
/// power : float, optional (default=0.8)
///     Desired statistical power (1 - beta).
///
/// Returns
/// -------
/// int
///     Required sample size per group.
#[pyfunction]
#[pyo3(signature = (effect_size, alpha=0.05, power=0.8))]
fn sample_size_for_power(effect_size: f64, alpha: f64, power: f64) -> PyResult<usize> {
    Ok(stats::power::sample_size_for_power(
        effect_size,
        alpha,
        power,
        "two_sample_t",
    ))
}

/// Compute statistical power for a given sample size.
///
/// Parameters
/// ----------
/// n : int
///     Sample size per group.
/// effect_size : float
///     Expected effect size (Cohen's d).
/// alpha : float, optional (default=0.05)
///     Significance level.
///
/// Returns
/// -------
/// float
///     Statistical power (between 0 and 1).
#[pyfunction]
#[pyo3(signature = (n, effect_size, alpha=0.05))]
fn power_for_sample_size(n: usize, effect_size: f64, alpha: f64) -> PyResult<f64> {
    Ok(stats::power::power_for_sample_size(n, effect_size, alpha))
}

// =============================================================================
// Diagnostics
// =============================================================================

/// Compute the Durbin-Watson statistic for autocorrelation in residuals.
///
/// Parameters
/// ----------
/// residuals : numpy.ndarray
///     Residuals from a regression model.
///
/// Returns
/// -------
/// float
///     Durbin-Watson statistic (range 0 to 4). Values near 2 indicate
///     no autocorrelation, near 0 positive, near 4 negative.
#[pyfunction]
#[pyo3(signature = (residuals,))]
fn durbin_watson(residuals: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = to_array1(residuals);
    Ok(stats::diagnostics::durbin_watson(&arr))
}

/// Compute descriptive statistics for a sample.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     Sample data (1-D array of floats).
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: n, mean, std, sem, min, max, median, q1, q3,
///     iqr, skewness, kurtosis.
#[pyfunction]
#[pyo3(signature = (data,))]
fn descriptive_stats(py: Python<'_>, data: PyReadonlyArray1<f64>) -> PyResult<PyObject> {
    let data_arr = to_array1(data);
    let result =
        stats::DescriptiveStats::compute(&data_arr).map_err(crate::errors::ferro_to_pyerr)?;

    let dict = PyDict::new(py);
    dict.set_item("n", result.n)?;
    dict.set_item("mean", result.mean)?;
    dict.set_item("std", result.std)?;
    dict.set_item("sem", result.sem)?;
    dict.set_item("min", result.min)?;
    dict.set_item("max", result.max)?;
    dict.set_item("median", result.median)?;
    dict.set_item("q1", result.q1)?;
    dict.set_item("q3", result.q3)?;
    dict.set_item("iqr", result.iqr)?;
    dict.set_item("skewness", result.skewness)?;
    dict.set_item("kurtosis", result.kurtosis)?;
    Ok(dict.into())
}

/// Normality test using skewness-kurtosis omnibus test.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     Sample data (1-D array of floats).
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: statistic, p_value, is_normal, skewness, kurtosis.
#[pyfunction]
#[pyo3(signature = (data,))]
fn normality_test(py: Python<'_>, data: PyReadonlyArray1<f64>) -> PyResult<PyObject> {
    let data_arr = to_array1(data);
    let result = ShapiroWilkTest
        .test(&data_arr)
        .map_err(crate::errors::ferro_to_pyerr)?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("p_value", result.p_value)?;
    dict.set_item("is_normal", result.is_normal)?;
    dict.set_item("skewness", result.skewness)?;
    dict.set_item("kurtosis", result.kurtosis)?;
    Ok(dict.into())
}

// =============================================================================
// Correlation
// =============================================================================

/// Pearson correlation with confidence interval and significance test.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     First variable.
/// y : numpy.ndarray
///     Second variable.
/// confidence : float, optional (default=0.95)
///     Confidence level for the CI on r.
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: r, p_value, ci_lower, ci_upper, r_squared, n.
#[pyfunction]
#[pyo3(signature = (x, y, confidence=0.95))]
fn correlation(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    confidence: f64,
) -> PyResult<PyObject> {
    let x_arr = to_array1(x);
    let y_arr = to_array1(y);

    let result =
        stats::correlation(&x_arr, &y_arr, confidence).map_err(crate::errors::ferro_to_pyerr)?;

    let dict = PyDict::new(py);
    dict.set_item("r", result.r)?;
    dict.set_item("p_value", result.p_value)?;
    dict.set_item("ci_lower", result.ci.0)?;
    dict.set_item("ci_upper", result.ci.1)?;
    dict.set_item("r_squared", result.r_squared)?;
    dict.set_item("n", result.n)?;
    Ok(dict.into())
}

// =============================================================================
// Helpers
// =============================================================================

/// Convert a StatisticalResult to a Python dict
fn statistical_result_to_dict(
    py: Python<'_>,
    result: &stats::StatisticalResult,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("p_value", result.p_value)?;
    dict.set_item("test_name", &result.test_name)?;
    dict.set_item("alternative", &result.alternative)?;
    dict.set_item("n", result.n)?;

    if let Some(df) = result.df {
        dict.set_item("df", df)?;
    }

    if let Some(ref es) = result.effect_size {
        dict.set_item("effect_size", es.value)?;
        dict.set_item("effect_name", &es.name)?;
        dict.set_item("effect_interpretation", &es.interpretation)?;
        if let Some((lo, hi)) = es.ci {
            dict.set_item("effect_ci_lower", lo)?;
            dict.set_item("effect_ci_upper", hi)?;
        }
    }

    if let Some((lo, hi)) = result.confidence_interval {
        dict.set_item("ci_lower", lo)?;
        dict.set_item("ci_upper", hi)?;
    }

    if let Some(power) = result.power {
        dict.set_item("power", power)?;
    }

    Ok(dict.into())
}

// =============================================================================
// Module Registration
// =============================================================================

/// Register the stats submodule
pub fn register_stats_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "stats")?;
    m.add_function(wrap_pyfunction!(ttest_ind, &m)?)?;
    m.add_function(wrap_pyfunction!(welch_ttest, &m)?)?;
    m.add_function(wrap_pyfunction!(mann_whitney, &m)?)?;
    m.add_function(wrap_pyfunction!(cohens_d, &m)?)?;
    m.add_function(wrap_pyfunction!(hedges_g, &m)?)?;
    m.add_function(wrap_pyfunction!(confidence_interval, &m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_ci, &m)?)?;
    m.add_function(wrap_pyfunction!(adjust_pvalues, &m)?)?;
    m.add_function(wrap_pyfunction!(sample_size_for_power, &m)?)?;
    m.add_function(wrap_pyfunction!(power_for_sample_size, &m)?)?;
    m.add_function(wrap_pyfunction!(durbin_watson, &m)?)?;
    m.add_function(wrap_pyfunction!(descriptive_stats, &m)?)?;
    m.add_function(wrap_pyfunction!(normality_test, &m)?)?;
    m.add_function(wrap_pyfunction!(correlation, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

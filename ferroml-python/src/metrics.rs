//! Python bindings for FerroML Evaluation Metrics
//!
//! This module exposes metrics as Python-callable functions following
//! the sklearn.metrics API pattern:
//!
//! - Classification: accuracy, precision, recall, f1, confusion matrix, MCC, Cohen's kappa
//! - Regression: mse, rmse, mae, r2, explained variance, max error, MAPE
//! - Probabilistic: roc_auc, log_loss, brier_score, pr_auc, average_precision
//! - Model comparison: paired t-test, corrected resampled t-test, McNemar's, Wilcoxon

use ferroml_core::metrics::{
    self,
    classification::{self, ClassificationReport, ConfusionMatrix},
    comparison, probabilistic, regression, Average,
};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Convert PyReadonlyArray1 to owned ndarray Array1
fn to_array1(arr: PyReadonlyArray1<f64>) -> ndarray::Array1<f64> {
    arr.as_array().to_owned()
}

/// Parse average parameter string to Average enum
fn parse_average(average: &str) -> PyResult<Average> {
    match average {
        "binary" | "micro" => Ok(Average::Micro),
        "macro" => Ok(Average::Macro),
        "weighted" => Ok(Average::Weighted),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown average '{}'. Use 'binary', 'micro', 'macro', or 'weighted'.",
            average
        ))),
    }
}

// =============================================================================
// Classification Metrics
// =============================================================================

/// Compute accuracy score.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     Ground truth labels.
/// y_pred : numpy.ndarray
///     Predicted labels.
///
/// Returns
/// -------
/// float
///     Fraction of correct predictions.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn accuracy_score(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    classification::accuracy(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute precision score.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     Ground truth labels.
/// y_pred : numpy.ndarray
///     Predicted labels.
/// average : str, optional (default="binary")
///     Averaging method: "binary"/"micro", "macro", or "weighted".
///
/// Returns
/// -------
/// float
///     Precision score.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="binary"))]
fn precision_score(
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
    average: &str,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    let avg = parse_average(average)?;
    classification::precision(&yt, &yp, avg).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute recall score.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     Ground truth labels.
/// y_pred : numpy.ndarray
///     Predicted labels.
/// average : str, optional (default="binary")
///     Averaging method: "binary"/"micro", "macro", or "weighted".
///
/// Returns
/// -------
/// float
///     Recall score.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="binary"))]
fn recall_score(
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
    average: &str,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    let avg = parse_average(average)?;
    classification::recall(&yt, &yp, avg).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute F1 score.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     Ground truth labels.
/// y_pred : numpy.ndarray
///     Predicted labels.
/// average : str, optional (default="binary")
///     Averaging method: "binary"/"micro", "macro", or "weighted".
///
/// Returns
/// -------
/// float
///     F1 score.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="binary"))]
fn f1_score(
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
    average: &str,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    let avg = parse_average(average)?;
    classification::f1_score(&yt, &yp, avg).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Matthews Correlation Coefficient.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     Ground truth labels.
/// y_pred : numpy.ndarray
///     Predicted labels.
///
/// Returns
/// -------
/// float
///     MCC in range [-1, 1].
#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn matthews_corrcoef(
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    classification::matthews_corrcoef(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Cohen's Kappa coefficient.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     Ground truth labels.
/// y_pred : numpy.ndarray
///     Predicted labels.
///
/// Returns
/// -------
/// float
///     Cohen's Kappa in range [-1, 1].
#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn cohen_kappa_score(
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    classification::cohen_kappa_score(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute balanced accuracy score.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     Ground truth labels.
/// y_pred : numpy.ndarray
///     Predicted labels.
///
/// Returns
/// -------
/// float
///     Balanced accuracy (macro-averaged recall).
#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn balanced_accuracy_score(
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    classification::balanced_accuracy(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute confusion matrix.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     Ground truth labels.
/// y_pred : numpy.ndarray
///     Predicted labels.
///
/// Returns
/// -------
/// numpy.ndarray
///     2D confusion matrix where row i, column j is the count of samples
///     with true label i predicted as j.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn confusion_matrix(
    py: Python<'_>,
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    let cm = ConfusionMatrix::compute(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)?;
    // Convert usize matrix to f64 for numpy compatibility
    let f64_matrix = cm.matrix.mapv(|v| v as f64);
    Ok(f64_matrix.into_pyarray(py).into())
}

/// Generate a classification report.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     Ground truth labels.
/// y_pred : numpy.ndarray
///     Predicted labels.
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: accuracy, macro_precision, macro_recall, macro_f1,
///     weighted_precision, weighted_recall, weighted_f1, labels, per_class.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn classification_report(
    py: Python<'_>,
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    let report = ClassificationReport::compute(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)?;

    let dict = PyDict::new(py);
    dict.set_item("accuracy", report.accuracy)?;
    dict.set_item("macro_precision", report.macro_precision)?;
    dict.set_item("macro_recall", report.macro_recall)?;
    dict.set_item("macro_f1", report.macro_f1)?;
    dict.set_item("weighted_precision", report.weighted_precision)?;
    dict.set_item("weighted_recall", report.weighted_recall)?;
    dict.set_item("weighted_f1", report.weighted_f1)?;
    dict.set_item("labels", &report.labels)?;

    // Per-class metrics
    let per_class = PyList::empty(py);
    for i in 0..report.labels.len() {
        let class_dict = PyDict::new(py);
        class_dict.set_item("label", report.labels[i])?;
        class_dict.set_item("precision", report.precision[i])?;
        class_dict.set_item("recall", report.recall[i])?;
        class_dict.set_item("f1", report.f1[i])?;
        class_dict.set_item("support", report.support[i])?;
        per_class.append(class_dict)?;
    }
    dict.set_item("per_class", per_class)?;

    Ok(dict.into())
}

// =============================================================================
// Regression Metrics
// =============================================================================

/// Compute Mean Squared Error.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     Ground truth values.
/// y_pred : numpy.ndarray
///     Predicted values.
///
/// Returns
/// -------
/// float
///     Mean squared error.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn mse(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    regression::mse(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Root Mean Squared Error.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn rmse(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    regression::rmse(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Mean Absolute Error.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn mae(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    regression::mae(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute R² (coefficient of determination).
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     Ground truth values.
/// y_pred : numpy.ndarray
///     Predicted values.
///
/// Returns
/// -------
/// float
///     R² score. Best possible score is 1.0, can be negative.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn r2_score(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    regression::r2_score(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Explained Variance Score.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn explained_variance(
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    regression::explained_variance(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Maximum Error.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn max_error(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    metrics::regression::max_error(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Mean Absolute Percentage Error.
///
/// Note: Undefined when y_true contains zeros.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn mape(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    regression::mape(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Median Absolute Error.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred))]
fn median_absolute_error(
    y_true: PyReadonlyArray1<f64>,
    y_pred: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_pred);
    regression::median_absolute_error(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)
}

// =============================================================================
// Probabilistic Metrics
// =============================================================================

/// Compute ROC-AUC score for binary classification.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     True binary labels (0 or 1).
/// y_score : numpy.ndarray
///     Predicted probabilities for the positive class.
///
/// Returns
/// -------
/// float
///     ROC-AUC score in [0, 1].
#[pyfunction]
#[pyo3(signature = (y_true, y_score))]
fn roc_auc_score(y_true: PyReadonlyArray1<f64>, y_score: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let ys = to_array1(y_score);
    probabilistic::roc_auc_score(&yt, &ys).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Precision-Recall AUC score.
#[pyfunction]
#[pyo3(signature = (y_true, y_score))]
fn pr_auc_score(y_true: PyReadonlyArray1<f64>, y_score: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let ys = to_array1(y_score);
    probabilistic::pr_auc_score(&yt, &ys).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Average Precision score.
#[pyfunction]
#[pyo3(signature = (y_true, y_score))]
fn average_precision_score(
    y_true: PyReadonlyArray1<f64>,
    y_score: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let ys = to_array1(y_score);
    probabilistic::average_precision_score(&yt, &ys).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Log Loss (Binary Cross-Entropy).
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     True binary labels (0 or 1).
/// y_prob : numpy.ndarray
///     Predicted probabilities for the positive class.
/// eps : float or None, optional (default=None)
///     Small value to clip probabilities to avoid log(0). Default 1e-15.
///
/// Returns
/// -------
/// float
///     Log loss value. Lower is better.
#[pyfunction]
#[pyo3(signature = (y_true, y_prob, eps=None))]
fn log_loss(
    y_true: PyReadonlyArray1<f64>,
    y_prob: PyReadonlyArray1<f64>,
    eps: Option<f64>,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_prob);
    probabilistic::log_loss(&yt, &yp, eps).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Brier Score.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     True binary labels (0 or 1).
/// y_prob : numpy.ndarray
///     Predicted probabilities for the positive class.
///
/// Returns
/// -------
/// float
///     Brier score. Lower is better, 0 is perfect.
#[pyfunction]
#[pyo3(signature = (y_true, y_prob))]
fn brier_score(y_true: PyReadonlyArray1<f64>, y_prob: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_prob);
    probabilistic::brier_score(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Brier Skill Score.
#[pyfunction]
#[pyo3(signature = (y_true, y_prob))]
fn brier_skill_score(
    y_true: PyReadonlyArray1<f64>,
    y_prob: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let yt = to_array1(y_true);
    let yp = to_array1(y_prob);
    probabilistic::brier_skill_score(&yt, &yp).map_err(crate::errors::ferro_to_pyerr)
}

// =============================================================================
// Curve Functions (for plotting)
// =============================================================================

/// Compute ROC curve for binary classification.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     True binary labels (0 or 1).
/// y_score : numpy.ndarray
///     Predicted probabilities for the positive class.
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: fpr (array), tpr (array), thresholds (array), auc (float).
#[pyfunction]
#[pyo3(signature = (y_true, y_score))]
fn roc_curve(
    py: Python<'_>,
    y_true: PyReadonlyArray1<f64>,
    y_score: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let yt = to_array1(y_true);
    let ys = to_array1(y_score);
    let curve =
        probabilistic::RocCurve::compute(&yt, &ys).map_err(crate::errors::ferro_to_pyerr)?;

    let dict = PyDict::new(py);
    let fpr_arr = ndarray::Array1::from_vec(curve.fpr);
    let tpr_arr = ndarray::Array1::from_vec(curve.tpr);
    let thresh_arr = ndarray::Array1::from_vec(curve.thresholds);
    dict.set_item("fpr", fpr_arr.into_pyarray(py))?;
    dict.set_item("tpr", tpr_arr.into_pyarray(py))?;
    dict.set_item("thresholds", thresh_arr.into_pyarray(py))?;
    dict.set_item("auc", curve.auc)?;
    Ok(dict.into())
}

/// Compute Precision-Recall curve for binary classification.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     True binary labels (0 or 1).
/// y_score : numpy.ndarray
///     Predicted probabilities for the positive class.
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: precision (array), recall (array), thresholds (array),
///     auc (float), average_precision (float).
#[pyfunction]
#[pyo3(signature = (y_true, y_score))]
fn precision_recall_curve(
    py: Python<'_>,
    y_true: PyReadonlyArray1<f64>,
    y_score: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let yt = to_array1(y_true);
    let ys = to_array1(y_score);
    let curve = probabilistic::PrCurve::compute(&yt, &ys).map_err(crate::errors::ferro_to_pyerr)?;

    let dict = PyDict::new(py);
    let prec_arr = ndarray::Array1::from_vec(curve.precision);
    let rec_arr = ndarray::Array1::from_vec(curve.recall);
    let thresh_arr = ndarray::Array1::from_vec(curve.thresholds);
    dict.set_item("precision", prec_arr.into_pyarray(py))?;
    dict.set_item("recall", rec_arr.into_pyarray(py))?;
    dict.set_item("thresholds", thresh_arr.into_pyarray(py))?;
    dict.set_item("auc", curve.auc)?;
    dict.set_item("average_precision", curve.average_precision)?;
    Ok(dict.into())
}

// =============================================================================
// Model Comparison
// =============================================================================

/// Convert ModelComparisonResult to Python dict
fn comparison_result_to_dict(
    py: Python<'_>,
    result: &comparison::ModelComparisonResult,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("test_name", &result.test_name)?;
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("p_value", result.p_value)?;
    if let Some(df) = result.df {
        dict.set_item("df", df)?;
    }
    dict.set_item("mean_difference", result.mean_difference)?;
    dict.set_item("std_error", result.std_error)?;
    dict.set_item("ci_lower", result.ci_95.0)?;
    dict.set_item("ci_upper", result.ci_95.1)?;
    dict.set_item("significant", result.significant)?;
    dict.set_item("interpretation", &result.interpretation)?;
    Ok(dict.into())
}

/// Paired t-test for comparing CV scores of two models.
///
/// Parameters
/// ----------
/// scores1 : numpy.ndarray
///     CV scores from model 1.
/// scores2 : numpy.ndarray
///     CV scores from model 2 (same folds).
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: test_name, statistic, p_value, df,
///     mean_difference, std_error, ci_lower, ci_upper, significant, interpretation.
#[pyfunction]
#[pyo3(signature = (scores1, scores2))]
fn paired_ttest(
    py: Python<'_>,
    scores1: PyReadonlyArray1<f64>,
    scores2: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let s1 = to_array1(scores1);
    let s2 = to_array1(scores2);
    let result = comparison::paired_ttest(&s1, &s2).map_err(crate::errors::ferro_to_pyerr)?;
    comparison_result_to_dict(py, &result)
}

/// Corrected resampled t-test (Nadeau & Bengio, 2003).
///
/// Corrects for dependence between CV fold scores.
///
/// Parameters
/// ----------
/// scores1 : numpy.ndarray
///     CV scores from model 1.
/// scores2 : numpy.ndarray
///     CV scores from model 2 (same folds).
/// n_train : int
///     Number of training samples in each fold.
/// n_test : int
///     Number of test samples in each fold.
///
/// Returns
/// -------
/// dict
///     Same format as paired_ttest.
#[pyfunction]
#[pyo3(signature = (scores1, scores2, n_train, n_test))]
fn corrected_resampled_ttest(
    py: Python<'_>,
    scores1: PyReadonlyArray1<f64>,
    scores2: PyReadonlyArray1<f64>,
    n_train: usize,
    n_test: usize,
) -> PyResult<PyObject> {
    let s1 = to_array1(scores1);
    let s2 = to_array1(scores2);
    let result = comparison::corrected_resampled_ttest(&s1, &s2, n_train, n_test)
        .map_err(crate::errors::ferro_to_pyerr)?;
    comparison_result_to_dict(py, &result)
}

/// McNemar's test for comparing classifier predictions.
///
/// Parameters
/// ----------
/// y_true : numpy.ndarray
///     True labels.
/// pred1 : numpy.ndarray
///     Predictions from classifier 1.
/// pred2 : numpy.ndarray
///     Predictions from classifier 2.
///
/// Returns
/// -------
/// dict
///     Same format as paired_ttest.
#[pyfunction]
#[pyo3(signature = (y_true, pred1, pred2))]
fn mcnemar_test(
    py: Python<'_>,
    y_true: PyReadonlyArray1<f64>,
    pred1: PyReadonlyArray1<f64>,
    pred2: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let yt = to_array1(y_true);
    let p1 = to_array1(pred1);
    let p2 = to_array1(pred2);
    let result = comparison::mcnemar_test(&yt, &p1, &p2).map_err(crate::errors::ferro_to_pyerr)?;
    comparison_result_to_dict(py, &result)
}

/// Wilcoxon signed-rank test for comparing paired samples.
///
/// A non-parametric alternative to the paired t-test.
///
/// Parameters
/// ----------
/// scores1 : numpy.ndarray
///     Scores from model 1.
/// scores2 : numpy.ndarray
///     Scores from model 2 (same samples).
///
/// Returns
/// -------
/// dict
///     Same format as paired_ttest.
#[pyfunction]
#[pyo3(signature = (scores1, scores2))]
fn wilcoxon_test(
    py: Python<'_>,
    scores1: PyReadonlyArray1<f64>,
    scores2: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let s1 = to_array1(scores1);
    let s2 = to_array1(scores2);
    let result =
        comparison::wilcoxon_signed_rank_test(&s1, &s2).map_err(crate::errors::ferro_to_pyerr)?;
    comparison_result_to_dict(py, &result)
}

// =============================================================================
// Module Registration
// =============================================================================

/// Register the metrics submodule
pub fn register_metrics_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "metrics")?;

    // Classification
    m.add_function(wrap_pyfunction!(accuracy_score, &m)?)?;
    m.add_function(wrap_pyfunction!(precision_score, &m)?)?;
    m.add_function(wrap_pyfunction!(recall_score, &m)?)?;
    m.add_function(wrap_pyfunction!(f1_score, &m)?)?;
    m.add_function(wrap_pyfunction!(matthews_corrcoef, &m)?)?;
    m.add_function(wrap_pyfunction!(cohen_kappa_score, &m)?)?;
    m.add_function(wrap_pyfunction!(balanced_accuracy_score, &m)?)?;
    m.add_function(wrap_pyfunction!(confusion_matrix, &m)?)?;
    m.add_function(wrap_pyfunction!(classification_report, &m)?)?;

    // Regression
    m.add_function(wrap_pyfunction!(mse, &m)?)?;
    m.add_function(wrap_pyfunction!(rmse, &m)?)?;
    m.add_function(wrap_pyfunction!(mae, &m)?)?;
    m.add_function(wrap_pyfunction!(r2_score, &m)?)?;
    m.add_function(wrap_pyfunction!(explained_variance, &m)?)?;
    m.add_function(wrap_pyfunction!(max_error, &m)?)?;
    m.add_function(wrap_pyfunction!(mape, &m)?)?;
    m.add_function(wrap_pyfunction!(median_absolute_error, &m)?)?;

    // Probabilistic
    m.add_function(wrap_pyfunction!(roc_auc_score, &m)?)?;
    m.add_function(wrap_pyfunction!(pr_auc_score, &m)?)?;
    m.add_function(wrap_pyfunction!(average_precision_score, &m)?)?;
    m.add_function(wrap_pyfunction!(log_loss, &m)?)?;
    m.add_function(wrap_pyfunction!(brier_score, &m)?)?;
    m.add_function(wrap_pyfunction!(brier_skill_score, &m)?)?;

    // Curve functions
    m.add_function(wrap_pyfunction!(roc_curve, &m)?)?;
    m.add_function(wrap_pyfunction!(precision_recall_curve, &m)?)?;

    // Model comparison
    m.add_function(wrap_pyfunction!(paired_ttest, &m)?)?;
    m.add_function(wrap_pyfunction!(corrected_resampled_ttest, &m)?)?;
    m.add_function(wrap_pyfunction!(mcnemar_test, &m)?)?;
    m.add_function(wrap_pyfunction!(wilcoxon_test, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}

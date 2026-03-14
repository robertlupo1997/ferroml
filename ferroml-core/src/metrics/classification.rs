//! Classification Metrics
//!
//! Metrics for evaluating classification models including accuracy, precision,
//! recall, F1-score, and confusion matrix-based metrics.

use crate::metrics::{validate_arrays, Average, Direction, Metric, MetricValue};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Confusion matrix for binary or multiclass classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// The confusion matrix values [n_classes x n_classes]
    /// Row i, column j is the count of samples with true label i predicted as j
    pub matrix: Array2<usize>,
    /// Unique class labels in sorted order
    pub labels: Vec<i64>,
}

impl ConfusionMatrix {
    /// Compute confusion matrix from true and predicted labels
    pub fn compute(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Self> {
        validate_arrays(y_true, y_pred)?;

        // Get unique labels
        let mut labels: Vec<i64> = y_true
            .iter()
            .chain(y_pred.iter())
            .map(|&x| x as i64)
            .collect();
        labels.sort();
        labels.dedup();

        let n_classes = labels.len();
        let label_to_idx: HashMap<i64, usize> =
            labels.iter().enumerate().map(|(i, &l)| (l, i)).collect();

        let mut matrix = Array2::zeros((n_classes, n_classes));

        for (t, p) in y_true.iter().zip(y_pred.iter()) {
            let t_idx = label_to_idx[&(*t as i64)];
            let p_idx = label_to_idx[&(*p as i64)];
            matrix[[t_idx, p_idx]] += 1;
        }

        Ok(Self { matrix, labels })
    }

    /// Get true positives for each class
    pub fn true_positives(&self) -> Vec<usize> {
        (0..self.labels.len())
            .map(|i| self.matrix[[i, i]])
            .collect()
    }

    /// Get false positives for each class
    pub fn false_positives(&self) -> Vec<usize> {
        (0..self.labels.len())
            .map(|j| {
                (0..self.labels.len())
                    .filter(|&i| i != j)
                    .map(|i| self.matrix[[i, j]])
                    .sum()
            })
            .collect()
    }

    /// Get false negatives for each class
    pub fn false_negatives(&self) -> Vec<usize> {
        (0..self.labels.len())
            .map(|i| {
                (0..self.labels.len())
                    .filter(|&j| j != i)
                    .map(|j| self.matrix[[i, j]])
                    .sum()
            })
            .collect()
    }

    /// Get true negatives for each class
    pub fn true_negatives(&self) -> Vec<usize> {
        let total: usize = self.matrix.sum();
        let tp = self.true_positives();
        let fp = self.false_positives();
        let fn_ = self.false_negatives();

        (0..self.labels.len())
            .map(|i| total - tp[i] - fp[i] - fn_[i])
            .collect()
    }

    /// Get support (number of true instances) for each class
    pub fn support(&self) -> Vec<usize> {
        (0..self.labels.len())
            .map(|i| self.matrix.row(i).sum())
            .collect()
    }

    /// Total number of samples
    pub fn total(&self) -> usize {
        self.matrix.sum()
    }
}

/// Classification report with per-class and aggregate metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationReport {
    /// Per-class precision
    pub precision: Vec<f64>,
    /// Per-class recall
    pub recall: Vec<f64>,
    /// Per-class F1 score
    pub f1: Vec<f64>,
    /// Per-class support
    pub support: Vec<usize>,
    /// Class labels
    pub labels: Vec<i64>,
    /// Overall accuracy
    pub accuracy: f64,
    /// Macro-averaged precision
    pub macro_precision: f64,
    /// Macro-averaged recall
    pub macro_recall: f64,
    /// Macro-averaged F1
    pub macro_f1: f64,
    /// Weighted-averaged precision
    pub weighted_precision: f64,
    /// Weighted-averaged recall
    pub weighted_recall: f64,
    /// Weighted-averaged F1
    pub weighted_f1: f64,
}

impl ClassificationReport {
    /// Generate classification report from confusion matrix
    pub fn from_confusion_matrix(cm: &ConfusionMatrix) -> Self {
        let tp = cm.true_positives();
        let fp = cm.false_positives();
        let fn_ = cm.false_negatives();
        let support = cm.support();
        let total = cm.total();

        let n_classes = cm.labels.len();

        // Per-class metrics
        let precision: Vec<f64> = (0..n_classes)
            .map(|i| {
                let denom = tp[i] + fp[i];
                if denom == 0 {
                    0.0
                } else {
                    tp[i] as f64 / denom as f64
                }
            })
            .collect();

        let recall: Vec<f64> = (0..n_classes)
            .map(|i| {
                let denom = tp[i] + fn_[i];
                if denom == 0 {
                    0.0
                } else {
                    tp[i] as f64 / denom as f64
                }
            })
            .collect();

        let f1: Vec<f64> = (0..n_classes)
            .map(|i| {
                if precision[i] + recall[i] == 0.0 {
                    0.0
                } else {
                    2.0 * precision[i] * recall[i] / (precision[i] + recall[i])
                }
            })
            .collect();

        // Accuracy
        let correct: usize = tp.iter().sum();
        let accuracy = correct as f64 / total as f64;

        // Macro averages
        let (macro_precision, macro_recall, macro_f1) = if n_classes == 0 {
            (0.0, 0.0, 0.0)
        } else {
            (
                precision.iter().sum::<f64>() / n_classes as f64,
                recall.iter().sum::<f64>() / n_classes as f64,
                f1.iter().sum::<f64>() / n_classes as f64,
            )
        };

        // Weighted averages
        let total_support: usize = support.iter().sum();
        let (weighted_precision, weighted_recall, weighted_f1) = if total_support == 0 {
            (0.0, 0.0, 0.0)
        } else {
            (
                (0..n_classes)
                    .map(|i| precision[i] * support[i] as f64)
                    .sum::<f64>()
                    / total_support as f64,
                (0..n_classes)
                    .map(|i| recall[i] * support[i] as f64)
                    .sum::<f64>()
                    / total_support as f64,
                (0..n_classes)
                    .map(|i| f1[i] * support[i] as f64)
                    .sum::<f64>()
                    / total_support as f64,
            )
        };

        Self {
            precision,
            recall,
            f1,
            support,
            labels: cm.labels.clone(),
            accuracy,
            macro_precision,
            macro_recall,
            macro_f1,
            weighted_precision,
            weighted_recall,
            weighted_f1,
        }
    }

    /// Generate from true and predicted labels
    pub fn compute(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Self> {
        let cm = ConfusionMatrix::compute(y_true, y_pred)?;
        Ok(Self::from_confusion_matrix(&cm))
    }

    /// Format as a string table
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "{:>12} {:>10} {:>10} {:>10} {:>10}\n",
            "", "precision", "recall", "f1-score", "support"
        ));
        s.push_str(&"-".repeat(54));
        s.push('\n');

        for (i, label) in self.labels.iter().enumerate() {
            s.push_str(&format!(
                "{:>12} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
                label, self.precision[i], self.recall[i], self.f1[i], self.support[i]
            ));
        }

        s.push_str(&"-".repeat(54));
        s.push('\n');

        let total_support: usize = self.support.iter().sum();
        s.push_str(&format!(
            "{:>12} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            "accuracy", "", "", self.accuracy, total_support
        ));
        s.push_str(&format!(
            "{:>12} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            "macro avg", self.macro_precision, self.macro_recall, self.macro_f1, total_support
        ));
        s.push_str(&format!(
            "{:>12} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            "weighted avg",
            self.weighted_precision,
            self.weighted_recall,
            self.weighted_f1,
            total_support
        ));

        s
    }
}

/// Compute accuracy score
pub fn accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
    validate_arrays(y_true, y_pred)?;
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| (t - p).abs() < 1e-10)
        .count();
    Ok(correct as f64 / y_true.len() as f64)
}

/// Compute precision score
pub fn precision(y_true: &Array1<f64>, y_pred: &Array1<f64>, average: Average) -> Result<f64> {
    let cm = ConfusionMatrix::compute(y_true, y_pred)?;
    let report = ClassificationReport::from_confusion_matrix(&cm);

    match average {
        Average::Macro => Ok(report.macro_precision),
        Average::Micro => {
            let tp: usize = cm.true_positives().iter().sum();
            let fp: usize = cm.false_positives().iter().sum();
            if tp + fp == 0 {
                Ok(0.0)
            } else {
                Ok(tp as f64 / (tp + fp) as f64)
            }
        }
        Average::Weighted => Ok(report.weighted_precision),
        Average::None => Err(FerroError::invalid_input(
            "Average::None not supported for scalar precision; use ClassificationReport",
        )),
    }
}

/// Compute recall score
pub fn recall(y_true: &Array1<f64>, y_pred: &Array1<f64>, average: Average) -> Result<f64> {
    let cm = ConfusionMatrix::compute(y_true, y_pred)?;
    let report = ClassificationReport::from_confusion_matrix(&cm);

    match average {
        Average::Macro => Ok(report.macro_recall),
        Average::Micro => {
            let tp: usize = cm.true_positives().iter().sum();
            let fn_: usize = cm.false_negatives().iter().sum();
            if tp + fn_ == 0 {
                Ok(0.0)
            } else {
                Ok(tp as f64 / (tp + fn_) as f64)
            }
        }
        Average::Weighted => Ok(report.weighted_recall),
        Average::None => Err(FerroError::invalid_input(
            "Average::None not supported for scalar recall; use ClassificationReport",
        )),
    }
}

/// Compute F1 score
pub fn f1_score(y_true: &Array1<f64>, y_pred: &Array1<f64>, average: Average) -> Result<f64> {
    let cm = ConfusionMatrix::compute(y_true, y_pred)?;
    let report = ClassificationReport::from_confusion_matrix(&cm);

    match average {
        Average::Macro => Ok(report.macro_f1),
        Average::Micro => {
            let p = precision(y_true, y_pred, Average::Micro)?;
            let r = recall(y_true, y_pred, Average::Micro)?;
            if p + r == 0.0 {
                Ok(0.0)
            } else {
                Ok(2.0 * p * r / (p + r))
            }
        }
        Average::Weighted => Ok(report.weighted_f1),
        Average::None => Err(FerroError::invalid_input(
            "Average::None not supported for scalar f1; use ClassificationReport",
        )),
    }
}

/// Compute confusion matrix (convenience function)
pub fn confusion_matrix(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<ConfusionMatrix> {
    ConfusionMatrix::compute(y_true, y_pred)
}

/// Accuracy metric implementing the Metric trait
#[derive(Debug, Clone, Default)]
pub struct AccuracyMetric;

impl Metric for AccuracyMetric {
    fn name(&self) -> &str {
        "accuracy"
    }

    fn direction(&self) -> Direction {
        Direction::Maximize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = accuracy(y_true, y_pred)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }
}

/// Precision metric implementing the Metric trait
#[derive(Debug, Clone)]
pub struct PrecisionMetric {
    /// Averaging strategy
    pub average: Average,
}

impl Default for PrecisionMetric {
    fn default() -> Self {
        Self {
            average: Average::Micro,
        }
    }
}

impl Metric for PrecisionMetric {
    fn name(&self) -> &str {
        "precision"
    }

    fn direction(&self) -> Direction {
        Direction::Maximize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = precision(y_true, y_pred, self.average)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }
}

/// Recall metric implementing the Metric trait
#[derive(Debug, Clone)]
pub struct RecallMetric {
    /// Averaging strategy
    pub average: Average,
}

impl Default for RecallMetric {
    fn default() -> Self {
        Self {
            average: Average::Micro,
        }
    }
}

impl Metric for RecallMetric {
    fn name(&self) -> &str {
        "recall"
    }

    fn direction(&self) -> Direction {
        Direction::Maximize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = recall(y_true, y_pred, self.average)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }
}

/// F1 Score metric implementing the Metric trait
#[derive(Debug, Clone)]
pub struct F1Metric {
    /// Averaging strategy
    pub average: Average,
}

impl Default for F1Metric {
    fn default() -> Self {
        Self {
            average: Average::Micro,
        }
    }
}

impl Metric for F1Metric {
    fn name(&self) -> &str {
        "f1"
    }

    fn direction(&self) -> Direction {
        Direction::Maximize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = f1_score(y_true, y_pred, self.average)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }
}

/// Compute balanced accuracy score
///
/// Balanced accuracy is the macro-averaged recall, which accounts for class imbalance.
/// For binary classification, it equals (sensitivity + specificity) / 2.
/// For multiclass, it is the arithmetic mean of per-class recall.
///
/// This metric is particularly useful for imbalanced datasets where accuracy
/// can be misleading.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels
///
/// # Returns
///
/// The balanced accuracy score in range [0, 1], where 1 is perfect prediction
/// and `1/n_classes` is expected for random guessing.
///
/// # Errors
///
/// Returns an error if `y_true` and `y_pred` have different lengths or are empty.
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use ferroml_core::metrics::classification::balanced_accuracy;
///
/// // Imbalanced dataset: 90 class 0, 10 class 1
/// // A classifier that always predicts 0 gets 90% accuracy but 50% balanced accuracy
/// let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
/// let y_pred = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
///
/// let ba = balanced_accuracy(&y_true, &y_pred).unwrap();
/// // Recall class 0: 2/2 = 1.0, Recall class 1: 1/2 = 0.5
/// // Balanced accuracy = (1.0 + 0.5) / 2 = 0.75
/// assert!((ba - 0.75).abs() < 1e-10);
/// ```
pub fn balanced_accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
    let cm = ConfusionMatrix::compute(y_true, y_pred)?;
    let report = ClassificationReport::from_confusion_matrix(&cm);
    // Balanced accuracy is the macro-averaged recall
    Ok(report.macro_recall)
}

/// Balanced Accuracy metric implementing the Metric trait
///
/// Balanced accuracy is designed for imbalanced datasets. It computes the
/// average recall across all classes, giving equal weight to each class
/// regardless of their size.
///
/// For binary classification with 50% prevalence, balanced accuracy equals
/// standard accuracy. For imbalanced data, balanced accuracy provides a
/// more reliable measure of model performance.
#[derive(Debug, Clone, Default)]
pub struct BalancedAccuracyMetric;

impl Metric for BalancedAccuracyMetric {
    fn name(&self) -> &str {
        "balanced_accuracy"
    }

    fn direction(&self) -> Direction {
        Direction::Maximize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = balanced_accuracy(y_true, y_pred)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }
}

/// Compute Matthews Correlation Coefficient (MCC)
///
/// MCC is a correlation coefficient between observed and predicted classifications.
/// It returns a value between -1 and +1:
/// - +1: Perfect prediction
/// -  0: No better than random prediction
/// - -1: Total disagreement between prediction and observation
///
/// MCC is particularly useful for imbalanced datasets as it takes into account
/// all four confusion matrix categories (TP, TN, FP, FN) and is considered a
/// balanced measure even when classes are of very different sizes.
///
/// For binary classification:
/// MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
///
/// For multiclass (uses R_k coefficient formulation):
/// MCC = (c×s - Σ(p_k × t_k)) / √((s² - Σp_k²)(s² - Σt_k²))
/// where:
/// - c = total correct predictions (trace of confusion matrix)
/// - s = total number of samples
/// - t_k = number of times class k truly occurred
/// - p_k = number of times class k was predicted
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels
///
/// # Returns
///
/// MCC value in range [-1, 1]
///
/// # Errors
///
/// Returns an error if arrays have different lengths or are empty.
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use ferroml_core::metrics::classification::matthews_corrcoef;
///
/// let y_true = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]);
/// let y_pred = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
///
/// let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
/// // MCC should be positive (better than random)
/// assert!(mcc > 0.0);
/// ```
pub fn matthews_corrcoef(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
    let cm = ConfusionMatrix::compute(y_true, y_pred)?;
    let n_classes = cm.labels.len();

    // Total number of samples
    let s = cm.total() as f64;
    if s == 0.0 {
        return Err(FerroError::invalid_input("Empty confusion matrix"));
    }

    // Total correct predictions (sum of diagonal)
    let c: f64 = (0..n_classes).map(|i| cm.matrix[[i, i]] as f64).sum();

    // t_k: true occurrences per class (row sums)
    let t: Vec<f64> = (0..n_classes)
        .map(|i| cm.matrix.row(i).sum() as f64)
        .collect();

    // p_k: predicted occurrences per class (column sums)
    let p: Vec<f64> = (0..n_classes)
        .map(|j| cm.matrix.column(j).sum() as f64)
        .collect();

    // Numerator: c*s - sum(p_k * t_k)
    let pk_tk_sum: f64 = (0..n_classes).map(|k| p[k] * t[k]).sum();
    let numerator = c.mul_add(s, -pk_tk_sum);

    // Denominator components
    let pk_sq_sum: f64 = p.iter().map(|&x| x * x).sum();
    let tk_sq_sum: f64 = t.iter().map(|&x| x * x).sum();

    let denom_left = s.mul_add(s, -pk_sq_sum);
    let denom_right = s.mul_add(s, -tk_sq_sum);

    // Handle edge cases where denominator would be zero
    if denom_left <= 0.0 || denom_right <= 0.0 {
        // This happens when all predictions or all true labels are the same class
        return Ok(0.0);
    }

    let denominator = (denom_left * denom_right).sqrt();

    if denominator == 0.0 {
        return Ok(0.0);
    }

    Ok(numerator / denominator)
}

/// Matthews Correlation Coefficient (MCC) metric implementing the Metric trait
///
/// MCC is a balanced measure that can be used even if the classes are of very
/// different sizes. Unlike F1 score which only considers positive predictions,
/// MCC considers all four quadrants of the confusion matrix.
///
/// MCC is essentially a correlation coefficient computed for binary classification,
/// generalized to multiclass. It is the only metric that takes into account the
/// ratio of the confusion matrix size.
#[derive(Debug, Clone, Default)]
pub struct MatthewsCorrCoefMetric;

impl Metric for MatthewsCorrCoefMetric {
    fn name(&self) -> &str {
        "matthews_corrcoef"
    }

    fn direction(&self) -> Direction {
        Direction::Maximize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = matthews_corrcoef(y_true, y_pred)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }
}

/// Compute Cohen's Kappa coefficient
///
/// Cohen's Kappa measures inter-rater agreement for categorical items,
/// accounting for agreement occurring by chance. It is more robust than
/// simple percent agreement because it accounts for the possibility of
/// agreement occurring by chance.
///
/// κ = (p_o - p_e) / (1 - p_e)
///
/// where:
/// - p_o = observed agreement (accuracy)
/// - p_e = expected agreement by chance
///
/// Interpretation:
/// - κ < 0: Less than chance agreement
/// - κ = 0: Chance agreement
/// - 0.01-0.20: Slight agreement
/// - 0.21-0.40: Fair agreement
/// - 0.41-0.60: Moderate agreement
/// - 0.61-0.80: Substantial agreement
/// - 0.81-1.00: Almost perfect agreement
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels
///
/// # Returns
///
/// Cohen's Kappa value, typically in range [-1, 1] where 1 is perfect agreement
///
/// # Errors
///
/// Returns an error if arrays have different lengths or are empty.
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use ferroml_core::metrics::classification::cohen_kappa_score;
///
/// let y_true = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
/// let y_pred = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
///
/// let kappa = cohen_kappa_score(&y_true, &y_pred).unwrap();
/// // Kappa should be positive (better than chance)
/// assert!(kappa > 0.0);
/// ```
pub fn cohen_kappa_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
    let cm = ConfusionMatrix::compute(y_true, y_pred)?;
    let n_classes = cm.labels.len();
    let n = cm.total() as f64;

    if n == 0.0 {
        return Err(FerroError::invalid_input("Empty confusion matrix"));
    }

    // Observed agreement: proportion of samples where y_true == y_pred
    let p_o: f64 = (0..n_classes)
        .map(|i| cm.matrix[[i, i]] as f64)
        .sum::<f64>()
        / n;

    // Expected agreement by chance
    // p_e = sum over classes of (P(true=k) * P(pred=k))
    // where P(true=k) = row_sum_k / n and P(pred=k) = col_sum_k / n
    let p_e: f64 = (0..n_classes)
        .map(|k| {
            let p_true_k = cm.matrix.row(k).sum() as f64 / n;
            let p_pred_k = cm.matrix.column(k).sum() as f64 / n;
            p_true_k * p_pred_k
        })
        .sum();

    // Handle perfect expected agreement (all same class)
    if (1.0 - p_e).abs() < 1e-10 {
        // If p_e == 1, kappa is undefined; return 1.0 if p_o == 1.0, else 0.0
        if (p_o - 1.0).abs() < 1e-10 {
            return Ok(1.0);
        }
        return Ok(0.0);
    }

    Ok((p_o - p_e) / (1.0 - p_e))
}

/// Cohen's Kappa coefficient metric implementing the Metric trait
///
/// Cohen's Kappa is a statistic that measures inter-rater agreement for
/// categorical items. It is generally thought to be a more robust measure
/// than simple percent agreement calculation, as κ takes into account the
/// possibility of the agreement occurring by chance.
///
/// This metric is particularly useful when:
/// - Comparing classifier performance to chance
/// - Evaluating agreement between human raters
/// - Dealing with imbalanced datasets where accuracy can be misleading
#[derive(Debug, Clone, Default)]
pub struct CohenKappaMetric;

impl Metric for CohenKappaMetric {
    fn name(&self) -> &str {
        "cohen_kappa"
    }

    fn direction(&self) -> Direction {
        Direction::Maximize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = cohen_kappa_score(y_true, y_pred)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix_binary() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 1.0, 1.0, 1.0, 0.0]);

        let cm = ConfusionMatrix::compute(&y_true, &y_pred).unwrap();

        // TN=1, FP=1, FN=1, TP=2
        assert_eq!(cm.matrix[[0, 0]], 1); // TN
        assert_eq!(cm.matrix[[0, 1]], 1); // FP
        assert_eq!(cm.matrix[[1, 0]], 1); // FN
        assert_eq!(cm.matrix[[1, 1]], 2); // TP
    }

    #[test]
    fn test_accuracy() {
        let y_true = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 1.0]);

        let acc = accuracy(&y_true, &y_pred).unwrap();
        assert!((acc - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_precision_recall_f1() {
        // Binary classification with known metrics
        let y_true = Array1::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        let y_pred = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        // TP=2, FP=1, FN=1, TN=2

        let p = precision(&y_true, &y_pred, Average::Micro).unwrap();
        let r = recall(&y_true, &y_pred, Average::Micro).unwrap();
        let f1 = f1_score(&y_true, &y_pred, Average::Micro).unwrap();

        // Micro for binary: precision = (TP0+TP1)/(TP0+TP1+FP0+FP1)
        // Accuracy = 4/6, which equals micro P, R, F1 for multiclass
        assert!((p - 4.0 / 6.0).abs() < 1e-10);
        assert!((r - 4.0 / 6.0).abs() < 1e-10);
        assert!((f1 - 4.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_classification_report() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 2.0, 2.0, 1.0]);

        let report = ClassificationReport::compute(&y_true, &y_pred).unwrap();

        assert_eq!(report.labels.len(), 3);
        assert!((report.accuracy - 4.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy_metric_trait() {
        let y_true = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);
        let y_pred = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);

        let metric = AccuracyMetric;
        let result = metric.compute(&y_true, &y_pred).unwrap();

        assert_eq!(result.name, "accuracy");
        assert!((result.value - 0.75).abs() < 1e-10);
        assert_eq!(result.direction, Direction::Maximize);
    }

    #[test]
    fn test_balanced_accuracy_binary() {
        // Imbalanced binary classification
        // Class 0: 2 samples, Class 1: 2 samples
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);

        // Recall for class 0: 2/2 = 1.0
        // Recall for class 1: 1/2 = 0.5
        // Balanced accuracy = (1.0 + 0.5) / 2 = 0.75
        let ba = balanced_accuracy(&y_true, &y_pred).unwrap();
        assert!((ba - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_balanced_accuracy_imbalanced() {
        // Highly imbalanced: 5 samples class 0, 1 sample class 1
        // A classifier predicting all 0 gets 83% accuracy but 50% balanced accuracy
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let acc = accuracy(&y_true, &y_pred).unwrap();
        let ba = balanced_accuracy(&y_true, &y_pred).unwrap();

        // Accuracy: 5/6 ≈ 0.833
        assert!((acc - 5.0 / 6.0).abs() < 1e-10);
        // Recall class 0: 5/5 = 1.0, Recall class 1: 0/1 = 0.0
        // Balanced accuracy: (1.0 + 0.0) / 2 = 0.5
        assert!((ba - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_balanced_accuracy_multiclass() {
        // Multiclass: 3 classes
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 2.0, 1.0]);

        // Recall class 0: 2/2 = 1.0
        // Recall class 1: 1/2 = 0.5
        // Recall class 2: 1/2 = 0.5
        // Balanced accuracy: (1.0 + 0.5 + 0.5) / 3 = 2/3
        let ba = balanced_accuracy(&y_true, &y_pred).unwrap();
        assert!((ba - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_balanced_accuracy_metric_trait() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);

        let metric = BalancedAccuracyMetric;
        let result = metric.compute(&y_true, &y_pred).unwrap();

        assert_eq!(result.name, "balanced_accuracy");
        assert!((result.value - 0.75).abs() < 1e-10);
        assert_eq!(result.direction, Direction::Maximize);
    }

    #[test]
    fn test_balanced_accuracy_perfect() {
        // Perfect classification
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

        let ba = balanced_accuracy(&y_true, &y_pred).unwrap();
        assert!((ba - 1.0).abs() < 1e-10);
    }

    // ==================== Matthews Correlation Coefficient Tests ====================

    #[test]
    fn test_mcc_perfect_binary() {
        // Perfect binary classification
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert!((mcc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_inverse_binary() {
        // Completely wrong binary classification (inverse of truth)
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]);

        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert!((mcc - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_binary_known_value() {
        // Known example: TP=2, TN=3, FP=1, FN=2
        // y_true: [1, 1, 1, 1, 0, 0, 0, 0] (4 pos, 4 neg)
        // y_pred: [1, 1, 0, 0, 0, 0, 0, 1] (TP=2, FN=2, TN=3, FP=1)
        let y_true = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let y_pred = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);

        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        // MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        // MCC = (2*3 - 1*2) / sqrt((2+1)(2+2)(3+1)(3+2))
        // MCC = (6 - 2) / sqrt(3 * 4 * 4 * 5) = 4 / sqrt(240)
        let expected = 4.0 / (240.0_f64).sqrt();
        assert!((mcc - expected).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_random_classifier() {
        // A classifier that predicts all one class should have MCC near 0
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]); // All predict 0

        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        // When all predictions are one class, MCC = 0
        assert!((mcc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_multiclass() {
        // Multiclass MCC
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        // Perfect multiclass classification should have MCC = 1
        assert!((mcc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_multiclass_imperfect() {
        // Imperfect multiclass
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 2.0, 0.0]);

        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        // Should be positive but less than 1
        assert!(mcc > 0.0);
        assert!(mcc < 1.0);
    }

    #[test]
    fn test_mcc_metric_trait() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let metric = MatthewsCorrCoefMetric;
        let result = metric.compute(&y_true, &y_pred).unwrap();

        assert_eq!(result.name, "matthews_corrcoef");
        assert!((result.value - 1.0).abs() < 1e-10);
        assert_eq!(result.direction, Direction::Maximize);
    }

    // ==================== Cohen's Kappa Tests ====================

    #[test]
    fn test_kappa_perfect() {
        // Perfect agreement
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let kappa = cohen_kappa_score(&y_true, &y_pred).unwrap();
        assert!((kappa - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kappa_no_agreement() {
        // Complete disagreement for balanced classes
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]);

        let kappa = cohen_kappa_score(&y_true, &y_pred).unwrap();
        // For balanced classes with complete disagreement, kappa = -1
        assert!((kappa - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_kappa_chance_level() {
        // Kappa should be 0 when agreement equals expected by chance
        // This is hard to construct exactly, but all-same predictions give 0
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]); // All predict 0

        let kappa = cohen_kappa_score(&y_true, &y_pred).unwrap();
        // p_o = 2/4 = 0.5
        // p_e = (2/4 * 4/4) + (2/4 * 0/4) = 0.5 + 0 = 0.5
        // kappa = (0.5 - 0.5) / (1 - 0.5) = 0
        assert!((kappa - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_kappa_known_value() {
        // Example with known kappa
        // y_true: [0, 0, 0, 1, 1, 1]
        // y_pred: [0, 0, 1, 0, 1, 1]
        // Confusion: [[2, 1], [1, 2]]
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0]);

        let kappa = cohen_kappa_score(&y_true, &y_pred).unwrap();
        // p_o = (2 + 2) / 6 = 4/6 = 0.6667
        // p_true_0 = 3/6 = 0.5, p_pred_0 = 3/6 = 0.5
        // p_true_1 = 3/6 = 0.5, p_pred_1 = 3/6 = 0.5
        // p_e = 0.5*0.5 + 0.5*0.5 = 0.5
        // kappa = (0.6667 - 0.5) / (1 - 0.5) = 0.1667 / 0.5 = 0.3333
        let expected = (4.0 / 6.0 - 0.5) / 0.5;
        assert!((kappa - expected).abs() < 1e-10);
    }

    #[test]
    fn test_kappa_multiclass() {
        // Multiclass Cohen's Kappa
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

        let kappa = cohen_kappa_score(&y_true, &y_pred).unwrap();
        // Perfect agreement
        assert!((kappa - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kappa_multiclass_imperfect() {
        // Imperfect multiclass
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 2.0, 0.0]);

        let kappa = cohen_kappa_score(&y_true, &y_pred).unwrap();
        // Should be positive (better than chance) but less than 1
        assert!(kappa > 0.0);
        assert!(kappa < 1.0);
    }

    #[test]
    fn test_kappa_metric_trait() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let metric = CohenKappaMetric;
        let result = metric.compute(&y_true, &y_pred).unwrap();

        assert_eq!(result.name, "cohen_kappa");
        assert!((result.value - 1.0).abs() < 1e-10);
        assert_eq!(result.direction, Direction::Maximize);
    }

    #[test]
    fn test_kappa_interpretation() {
        // Test that kappa values match interpretations
        // Substantial agreement: 0.61-0.80
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);

        let kappa = cohen_kappa_score(&y_true, &y_pred).unwrap();
        // 8 correct out of 10: p_o = 0.8
        // p_e = 0.5*0.5 + 0.5*0.5 = 0.5
        // kappa = (0.8 - 0.5) / (1 - 0.5) = 0.6 (substantial agreement boundary)
        let expected = 0.6;
        assert!((kappa - expected).abs() < 1e-10);
    }

    // ==================== MCC vs Kappa Comparison Tests ====================

    #[test]
    fn test_mcc_kappa_perfect_agreement() {
        // Both should be 1.0 for perfect agreement
        let y_true = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        let kappa = cohen_kappa_score(&y_true, &y_pred).unwrap();

        assert!((mcc - 1.0).abs() < 1e-10);
        assert!((kappa - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_kappa_both_positive_for_good_classifier() {
        // A reasonably good classifier should have positive MCC and Kappa
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);

        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        let kappa = cohen_kappa_score(&y_true, &y_pred).unwrap();

        // Both should be positive for a better-than-chance classifier
        assert!(mcc > 0.0, "MCC should be positive: {}", mcc);
        assert!(kappa > 0.0, "Kappa should be positive: {}", kappa);
    }
}

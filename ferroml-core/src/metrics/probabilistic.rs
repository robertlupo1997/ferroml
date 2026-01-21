//! Probabilistic Classification Metrics
//!
//! Metrics for evaluating probabilistic predictions including ROC-AUC, PR-AUC,
//! log loss, and Brier score. All metrics support confidence interval estimation
//! via bootstrapping.

use crate::metrics::{
    validate_arrays, Direction, Metric, MetricValue, MetricValueWithCI, ProbabilisticMetric,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Compute ROC-AUC score for binary classification
///
/// Area Under the Receiver Operating Characteristic Curve.
/// Measures the ability of the classifier to distinguish between classes.
///
/// # Arguments
/// * `y_true` - True binary labels (0 or 1)
/// * `y_score` - Predicted probabilities for the positive class
///
/// # Returns
/// ROC-AUC score in [0, 1]. A score of 0.5 indicates random guessing.
pub fn roc_auc_score(y_true: &Array1<f64>, y_score: &Array1<f64>) -> Result<f64> {
    validate_arrays(y_true, y_score)?;

    let n = y_true.len();
    if n < 2 {
        return Err(FerroError::invalid_input(
            "Need at least 2 samples for ROC-AUC",
        ));
    }

    // Check for valid labels
    let unique_labels: Vec<f64> = {
        let mut labels: Vec<f64> = y_true.iter().copied().collect();
        labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        labels.dedup();
        labels
    };

    if unique_labels.len() < 2 {
        return Err(FerroError::invalid_input(
            "ROC-AUC requires both positive and negative samples",
        ));
    }

    // Check for both positive and negative samples
    let n_pos = y_true.iter().filter(|&&y| y > 0.5).count();
    if n_pos == 0 || n_pos == n {
        return Err(FerroError::invalid_input(
            "ROC-AUC requires both positive and negative samples",
        ));
    }

    // Compute AUC using trapezoidal rule on ROC curve
    // This handles ties correctly and is numerically stable
    let (fpr, tpr) = compute_roc_curve(y_true, y_score)?;
    let auc = trapezoidal_auc(&fpr, &tpr);

    Ok(auc)
}

/// Compute ROC curve points (FPR, TPR)
fn compute_roc_curve(y_true: &Array1<f64>, y_score: &Array1<f64>) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = y_true.len();
    let n_pos = y_true.iter().filter(|&&y| y > 0.5).count() as f64;
    let n_neg = n as f64 - n_pos;

    // Sort by score descending
    let mut pairs: Vec<(f64, f64)> = y_score
        .iter()
        .zip(y_true.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut fpr = Vec::with_capacity(n + 2);
    let mut tpr = Vec::with_capacity(n + 2);

    fpr.push(0.0);
    tpr.push(0.0);

    let mut tp = 0.0;
    let mut fp = 0.0;

    let mut prev_score = f64::INFINITY;

    for (score, label) in &pairs {
        // Add point when threshold changes
        if (*score - prev_score).abs() > 1e-10 && (tp > 0.0 || fp > 0.0) {
            fpr.push(fp / n_neg);
            tpr.push(tp / n_pos);
        }
        prev_score = *score;

        if *label > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
    }

    // Add final point (1, 1)
    fpr.push(1.0);
    tpr.push(1.0);

    Ok((fpr, tpr))
}

/// Compute AUC using trapezoidal rule
fn trapezoidal_auc(x: &[f64], y: &[f64]) -> f64 {
    let mut auc = 0.0;
    for i in 1..x.len() {
        // Trapezoidal rule: area = (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2
        auc += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0;
    }
    auc
}

/// Compute ROC-AUC with confidence interval via bootstrap
pub fn roc_auc_with_ci(
    y_true: &Array1<f64>,
    y_score: &Array1<f64>,
    confidence: f64,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> Result<MetricValueWithCI> {
    let base_auc = roc_auc_score(y_true, y_score)?;

    let bootstrap_aucs = bootstrap_auc(y_true, y_score, n_bootstrap, seed)?;

    let (ci_lower, ci_upper) = percentile_ci(&bootstrap_aucs, confidence);
    let std_error = std_err(&bootstrap_aucs);

    Ok(MetricValueWithCI {
        value: base_auc,
        ci_lower,
        ci_upper,
        confidence_level: confidence,
        std_error,
        n_bootstrap,
    })
}

/// Bootstrap ROC-AUC scores
fn bootstrap_auc(
    y_true: &Array1<f64>,
    y_score: &Array1<f64>,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> Result<Vec<f64>> {
    let n = y_true.len();

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(rand::rngs::StdRng::seed_from_u64(s)),
        None => Box::new(rand::rng()),
    };

    let mut aucs = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Sample with replacement
        let indices: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();

        let y_true_sample = Array1::from_vec(indices.iter().map(|&i| y_true[i]).collect());
        let y_score_sample = Array1::from_vec(indices.iter().map(|&i| y_score[i]).collect());

        // Skip samples that don't have both classes
        if let Ok(auc) = roc_auc_score(&y_true_sample, &y_score_sample) {
            aucs.push(auc);
        }
    }

    if aucs.is_empty() {
        return Err(FerroError::numerical("All bootstrap samples failed"));
    }

    Ok(aucs)
}

/// Compute Precision-Recall AUC
///
/// Area Under the Precision-Recall Curve. More informative than ROC-AUC
/// for imbalanced datasets.
///
/// # Arguments
/// * `y_true` - True binary labels (0 or 1)
/// * `y_score` - Predicted probabilities for the positive class
pub fn pr_auc_score(y_true: &Array1<f64>, y_score: &Array1<f64>) -> Result<f64> {
    validate_arrays(y_true, y_score)?;

    let (precision, recall) = compute_pr_curve(y_true, y_score)?;

    // PR-AUC uses interpolated precision at each recall threshold
    let auc = trapezoidal_auc(&recall, &precision);

    Ok(auc)
}

/// Compute Precision-Recall curve points
fn compute_pr_curve(y_true: &Array1<f64>, y_score: &Array1<f64>) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = y_true.len();
    let n_pos = y_true.iter().filter(|&&y| y > 0.5).count() as f64;

    if n_pos == 0.0 {
        return Err(FerroError::invalid_input(
            "PR curve requires at least one positive sample",
        ));
    }

    // Sort by score descending
    let mut pairs: Vec<(f64, f64)> = y_score
        .iter()
        .zip(y_true.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut precision = Vec::with_capacity(n + 1);
    let mut recall = Vec::with_capacity(n + 1);

    // Start at recall=0, precision=1 (extrapolation)
    recall.push(0.0);
    precision.push(1.0);

    let mut tp = 0.0;
    let mut fp = 0.0;

    for (_, label) in &pairs {
        if *label > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        let p = tp / (tp + fp);
        let r = tp / n_pos;

        precision.push(p);
        recall.push(r);
    }

    Ok((precision, recall))
}

/// Compute Average Precision Score
///
/// Weighted mean of precisions at each threshold, with the increase in recall
/// from the previous threshold as the weight.
///
/// # Arguments
/// * `y_true` - True binary labels (0 or 1)
/// * `y_score` - Predicted probabilities for the positive class
pub fn average_precision_score(y_true: &Array1<f64>, y_score: &Array1<f64>) -> Result<f64> {
    validate_arrays(y_true, y_score)?;

    let (precision, recall) = compute_pr_curve(y_true, y_score)?;

    // AP = sum of (R_n - R_{n-1}) * P_n
    let mut ap = 0.0;
    for i in 1..precision.len() {
        ap += (recall[i] - recall[i - 1]) * precision[i];
    }

    Ok(ap)
}

/// Compute Log Loss (Binary Cross-Entropy)
///
/// Measures the performance of a classification model where the prediction
/// is a probability between 0 and 1. Lower is better.
///
/// # Arguments
/// * `y_true` - True binary labels (0 or 1)
/// * `y_pred` - Predicted probabilities for the positive class
/// * `eps` - Small value to clip probabilities to avoid log(0)
pub fn log_loss(y_true: &Array1<f64>, y_pred: &Array1<f64>, eps: Option<f64>) -> Result<f64> {
    validate_arrays(y_true, y_pred)?;

    let eps = eps.unwrap_or(1e-15);
    let n = y_true.len() as f64;

    let loss: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&y, &p)| {
            // Clip probabilities
            let p = p.clamp(eps, 1.0 - eps);
            // Binary cross-entropy
            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        })
        .sum();

    Ok(loss / n)
}

/// Compute Brier Score
///
/// Mean squared error between predicted probabilities and true labels.
/// Measures calibration quality. Lower is better, with 0 being perfect.
///
/// # Arguments
/// * `y_true` - True binary labels (0 or 1)
/// * `y_prob` - Predicted probabilities for the positive class
pub fn brier_score(y_true: &Array1<f64>, y_prob: &Array1<f64>) -> Result<f64> {
    validate_arrays(y_true, y_prob)?;

    let n = y_true.len() as f64;
    let score: f64 = y_true
        .iter()
        .zip(y_prob.iter())
        .map(|(&y, &p)| (p - y).powi(2))
        .sum();

    Ok(score / n)
}

/// Brier Skill Score
///
/// Measures improvement over a reference forecast (typically climatological
/// probability). BSS = 1 - (BS / BS_ref). A score of 1 is perfect, 0 means
/// no improvement over reference.
///
/// # Arguments
/// * `y_true` - True binary labels (0 or 1)
/// * `y_prob` - Predicted probabilities for the positive class
pub fn brier_skill_score(y_true: &Array1<f64>, y_prob: &Array1<f64>) -> Result<f64> {
    let bs = brier_score(y_true, y_prob)?;

    // Reference: climatological probability (base rate)
    let base_rate = y_true.mean().unwrap_or(0.5);
    let bs_ref = base_rate * (1.0 - base_rate);

    if bs_ref == 0.0 {
        return Ok(1.0); // Perfect reference (all same class)
    }

    Ok(1.0 - bs / bs_ref)
}

// ============================================================================
// Metric trait implementations
// ============================================================================

/// ROC-AUC metric implementing the Metric trait
#[derive(Debug, Clone, Default)]
pub struct RocAucMetric;

impl Metric for RocAucMetric {
    fn name(&self) -> &str {
        "roc_auc"
    }

    fn direction(&self) -> Direction {
        Direction::Maximize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = roc_auc_score(y_true, y_pred)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }

    fn requires_probabilities(&self) -> bool {
        true
    }
}

impl ProbabilisticMetric for RocAucMetric {
    fn compute_proba(&self, y_true: &Array1<f64>, y_proba: &Array2<f64>) -> Result<MetricValue> {
        // Use probability of positive class (column 1 for binary)
        let y_score = if y_proba.ncols() == 2 {
            y_proba.column(1).to_owned()
        } else if y_proba.ncols() == 1 {
            y_proba.column(0).to_owned()
        } else {
            return Err(FerroError::invalid_input(
                "ROC-AUC only supports binary classification",
            ));
        };
        self.compute(y_true, &y_score)
    }

    fn compute_proba_with_ci(
        &self,
        y_true: &Array1<f64>,
        y_proba: &Array2<f64>,
        confidence: f64,
        n_bootstrap: usize,
        seed: Option<u64>,
    ) -> Result<MetricValueWithCI> {
        let y_score = if y_proba.ncols() == 2 {
            y_proba.column(1).to_owned()
        } else if y_proba.ncols() == 1 {
            y_proba.column(0).to_owned()
        } else {
            return Err(FerroError::invalid_input(
                "ROC-AUC only supports binary classification",
            ));
        };
        roc_auc_with_ci(y_true, &y_score, confidence, n_bootstrap, seed)
    }
}

/// PR-AUC metric implementing the Metric trait
#[derive(Debug, Clone, Default)]
pub struct PrAucMetric;

impl Metric for PrAucMetric {
    fn name(&self) -> &str {
        "pr_auc"
    }

    fn direction(&self) -> Direction {
        Direction::Maximize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = pr_auc_score(y_true, y_pred)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }

    fn requires_probabilities(&self) -> bool {
        true
    }
}

/// Average Precision metric implementing the Metric trait
#[derive(Debug, Clone, Default)]
pub struct AveragePrecisionMetric;

impl Metric for AveragePrecisionMetric {
    fn name(&self) -> &str {
        "average_precision"
    }

    fn direction(&self) -> Direction {
        Direction::Maximize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = average_precision_score(y_true, y_pred)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }

    fn requires_probabilities(&self) -> bool {
        true
    }
}

/// Log Loss metric implementing the Metric trait
#[derive(Debug, Clone)]
pub struct LogLossMetric {
    /// Epsilon for clipping probabilities
    pub eps: f64,
}

impl Default for LogLossMetric {
    fn default() -> Self {
        Self { eps: 1e-15 }
    }
}

impl Metric for LogLossMetric {
    fn name(&self) -> &str {
        "log_loss"
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = log_loss(y_true, y_pred, Some(self.eps))?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }

    fn requires_probabilities(&self) -> bool {
        true
    }
}

/// Brier Score metric implementing the Metric trait
#[derive(Debug, Clone, Default)]
pub struct BrierScoreMetric;

impl Metric for BrierScoreMetric {
    fn name(&self) -> &str {
        "brier_score"
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = brier_score(y_true, y_pred)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }

    fn requires_probabilities(&self) -> bool {
        true
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Compute percentile confidence interval
fn percentile_ci(scores: &[f64], confidence: f64) -> (f64, f64) {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let n = sorted.len();

    let lower_idx = ((alpha / 2.0) * n as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n as f64).ceil() as usize;

    let lower_idx = lower_idx.min(n.saturating_sub(1));
    let upper_idx = upper_idx.min(n.saturating_sub(1));

    (sorted[lower_idx], sorted[upper_idx])
}

/// Compute standard error
fn std_err(scores: &[f64]) -> f64 {
    let n = scores.len() as f64;
    if n <= 1.0 {
        return 0.0;
    }

    let mean = scores.iter().sum::<f64>() / n;
    let variance = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    variance.sqrt()
}

/// ROC curve data for plotting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocCurve {
    /// False positive rates
    pub fpr: Vec<f64>,
    /// True positive rates
    pub tpr: Vec<f64>,
    /// Thresholds used
    pub thresholds: Vec<f64>,
    /// Area under the curve
    pub auc: f64,
}

impl RocCurve {
    /// Compute ROC curve from predictions
    pub fn compute(y_true: &Array1<f64>, y_score: &Array1<f64>) -> Result<Self> {
        let (fpr, tpr) = compute_roc_curve(y_true, y_score)?;
        let auc = trapezoidal_auc(&fpr, &tpr);

        // Extract thresholds (simplified - descending unique scores)
        let mut thresholds: Vec<f64> = y_score.iter().copied().collect();
        thresholds.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        thresholds.dedup();

        Ok(Self {
            fpr,
            tpr,
            thresholds,
            auc,
        })
    }
}

/// Precision-Recall curve data for plotting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrCurve {
    /// Precision values
    pub precision: Vec<f64>,
    /// Recall values
    pub recall: Vec<f64>,
    /// Thresholds used
    pub thresholds: Vec<f64>,
    /// Area under the curve
    pub auc: f64,
    /// Average precision score
    pub average_precision: f64,
}

impl PrCurve {
    /// Compute PR curve from predictions
    pub fn compute(y_true: &Array1<f64>, y_score: &Array1<f64>) -> Result<Self> {
        let (precision, recall) = compute_pr_curve(y_true, y_score)?;
        let auc = trapezoidal_auc(&recall, &precision);
        let ap = average_precision_score(y_true, y_score)?;

        // Extract thresholds
        let mut thresholds: Vec<f64> = y_score.iter().copied().collect();
        thresholds.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        thresholds.dedup();

        Ok(Self {
            precision,
            recall,
            thresholds,
            auc,
            average_precision: ap,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_roc_auc_perfect() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.2, 0.8, 0.9]);

        let auc = roc_auc_score(&y_true, &y_score).unwrap();
        assert_relative_eq!(auc, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_roc_auc_random() {
        let y_true = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);

        let auc = roc_auc_score(&y_true, &y_score).unwrap();
        assert_relative_eq!(auc, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_roc_auc_typical() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.4, 0.35, 0.8, 0.3, 0.7]);

        let auc = roc_auc_score(&y_true, &y_score).unwrap();
        // Should be between 0.5 and 1.0
        assert!(auc > 0.5);
        assert!(auc <= 1.0);
    }

    #[test]
    fn test_roc_auc_with_ci() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);

        let result = roc_auc_with_ci(&y_true, &y_score, 0.95, 100, Some(42)).unwrap();

        assert!(result.value > 0.5);
        assert!(result.ci_lower <= result.value);
        assert!(result.ci_upper >= result.value);
        assert_eq!(result.confidence_level, 0.95);
    }

    #[test]
    fn test_pr_auc() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.2, 0.8, 0.9]);

        let auc = pr_auc_score(&y_true, &y_score).unwrap();
        assert!(auc > 0.9); // Should be near 1 for perfect predictions
    }

    #[test]
    fn test_average_precision() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.2, 0.8, 0.9]);

        let ap = average_precision_score(&y_true, &y_score).unwrap();
        assert!(ap > 0.9);
    }

    #[test]
    fn test_log_loss_perfect() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0001, 0.0001, 0.9999, 0.9999]);

        let loss = log_loss(&y_true, &y_pred, None).unwrap();
        assert!(loss < 0.001); // Should be very small for near-perfect predictions
    }

    #[test]
    fn test_log_loss_random() {
        let y_true = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);

        let loss = log_loss(&y_true, &y_pred, None).unwrap();
        // Log loss of 0.5 for all predictions = -ln(0.5) ≈ 0.693
        assert_relative_eq!(loss, 0.693, epsilon = 0.01);
    }

    #[test]
    fn test_brier_score_perfect() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let score = brier_score(&y_true, &y_pred).unwrap();
        assert_relative_eq!(score, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_brier_score_typical() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.1, 0.2, 0.8, 0.9]);

        let score = brier_score(&y_true, &y_pred).unwrap();
        // (0.1² + 0.2² + 0.2² + 0.1²) / 4 = (0.01 + 0.04 + 0.04 + 0.01) / 4 = 0.025
        assert_relative_eq!(score, 0.025, epsilon = 1e-10);
    }

    #[test]
    fn test_brier_skill_score() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.1, 0.2, 0.8, 0.9]);

        let bss = brier_skill_score(&y_true, &y_pred).unwrap();
        // Should be positive (better than climatology)
        assert!(bss > 0.0);
        assert!(bss <= 1.0);
    }

    #[test]
    fn test_roc_auc_metric_trait() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.2, 0.8, 0.9]);

        let metric = RocAucMetric;
        let result = metric.compute(&y_true, &y_score).unwrap();

        assert_eq!(result.name, "roc_auc");
        assert_eq!(result.direction, Direction::Maximize);
        assert_relative_eq!(result.value, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_log_loss_metric_trait() {
        let y_true = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);

        let metric = LogLossMetric::default();
        let result = metric.compute(&y_true, &y_pred).unwrap();

        assert_eq!(result.name, "log_loss");
        assert_eq!(result.direction, Direction::Minimize);
    }

    #[test]
    fn test_brier_metric_trait() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let metric = BrierScoreMetric;
        let result = metric.compute(&y_true, &y_pred).unwrap();

        assert_eq!(result.name, "brier_score");
        assert_eq!(result.direction, Direction::Minimize);
        assert_relative_eq!(result.value, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_roc_curve_struct() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.4, 0.8, 0.9]);

        let curve = RocCurve::compute(&y_true, &y_score).unwrap();

        assert!(!curve.fpr.is_empty());
        assert!(!curve.tpr.is_empty());
        assert!(curve.auc > 0.9);
    }

    #[test]
    fn test_pr_curve_struct() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.4, 0.8, 0.9]);

        let curve = PrCurve::compute(&y_true, &y_score).unwrap();

        assert!(!curve.precision.is_empty());
        assert!(!curve.recall.is_empty());
        assert!(curve.auc > 0.9);
    }

    #[test]
    fn test_roc_auc_error_single_class() {
        let y_true = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.4, 0.8, 0.9]);

        let result = roc_auc_score(&y_true, &y_score);
        assert!(result.is_err());
    }

    #[test]
    fn test_roc_auc_error_empty() {
        let y_true: Array1<f64> = Array1::from_vec(vec![]);
        let y_score: Array1<f64> = Array1::from_vec(vec![]);

        let result = roc_auc_score(&y_true, &y_score);
        assert!(result.is_err());
    }
}

//! Uncertainty Quantification for Neural Networks
//!
//! This module provides uncertainty estimation using Monte Carlo Dropout
//! and other techniques.
//!
//! ## Features
//!
//! - MC Dropout for prediction intervals
//! - Confidence interval estimation
//! - Calibration analysis for probabilities

use crate::neural::Layer;
use crate::Result;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Prediction with uncertainty estimates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionUncertainty {
    /// Mean prediction (across MC samples)
    pub mean: Array1<f64>,
    /// Standard deviation of predictions
    pub std: Array1<f64>,
    /// Lower bound of confidence interval
    pub lower: Array1<f64>,
    /// Upper bound of confidence interval
    pub upper: Array1<f64>,
    /// Confidence level used
    pub confidence: f64,
    /// Number of MC samples
    pub n_samples: usize,
    /// Raw predictions from each MC sample (optional)
    pub samples: Option<Array2<f64>>,
}

impl PredictionUncertainty {
    /// Get prediction interval width
    pub fn interval_width(&self) -> Array1<f64> {
        &self.upper - &self.lower
    }

    /// Get coefficient of variation (std / |mean|)
    pub fn coefficient_of_variation(&self) -> Array1<f64> {
        let eps = 1e-10;
        &self.std / &self.mean.mapv(|m| m.abs().max(eps))
    }
}

/// Perform MC Dropout inference
///
/// Runs multiple forward passes with dropout enabled to estimate prediction uncertainty.
///
/// # Arguments
/// * `layers` - Network layers (will enable dropout during inference)
/// * `x` - Input data
/// * `n_samples` - Number of MC samples
/// * `dropout_rate` - Dropout rate to use
/// * `confidence` - Confidence level for intervals (e.g., 0.95)
/// * `seed` - Random seed
///
/// # Returns
/// `PredictionUncertainty` with mean, std, and confidence intervals
pub fn predict_with_uncertainty(
    layers: &mut [Layer],
    x: &Array2<f64>,
    n_samples: usize,
    dropout_rate: f64,
    confidence: f64,
    seed: Option<u64>,
) -> Result<PredictionUncertainty> {
    let n_inputs = x.nrows();
    let n_outputs = layers.last().map(|l| l.n_outputs).unwrap_or(0);

    // Collect predictions from multiple forward passes
    let mut all_predictions = Vec::with_capacity(n_samples);
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    };

    let n_layers = layers.len();
    for _ in 0..n_samples {
        let mut output = x.clone();

        for (i, layer) in layers.iter_mut().enumerate() {
            let is_output_layer = i == n_layers - 1;

            // Apply dropout to hidden layers
            if !is_output_layer && dropout_rate > 0.0 {
                output = layer.forward_with_dropout(&output, dropout_rate, true, &mut rng)?;
            } else {
                output = layer.forward(&output, false)?;
            }
        }

        all_predictions.push(output);
    }

    // Stack predictions: (n_samples, n_inputs, n_outputs)
    let samples = Array2::from_shape_fn((n_samples, n_inputs * n_outputs), |(s, i)| {
        let sample_idx = i / n_outputs;
        let output_idx = i % n_outputs;
        all_predictions[s][[sample_idx, output_idx]]
    });

    // Compute statistics across samples
    let mut mean = Array1::zeros(n_inputs);
    let mut std = Array1::zeros(n_inputs);

    for i in 0..n_inputs {
        // For now, just use first output dimension
        let values: Vec<f64> = all_predictions.iter().map(|p| p[[i, 0]]).collect();
        let m: f64 = values.iter().sum::<f64>() / n_samples as f64;
        let v: f64 = values.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / n_samples as f64;
        mean[i] = m;
        std[i] = v.sqrt();
    }

    // Compute confidence intervals using normal approximation
    // For 95% CI, z = 1.96
    let z = match confidence {
        c if c >= 0.99 => 2.576,
        c if c >= 0.95 => 1.96,
        c if c >= 0.90 => 1.645,
        c if c >= 0.80 => 1.282,
        _ => 1.96,
    };

    let lower = &mean - &std.mapv(|s| s * z);
    let upper = &mean + &std.mapv(|s| s * z);

    Ok(PredictionUncertainty {
        mean,
        std,
        lower,
        upper,
        confidence,
        n_samples,
        samples: Some(samples),
    })
}

/// Calibration analysis for probabilistic predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Predicted probability bins
    pub prob_bins: Vec<f64>,
    /// Observed frequencies in each bin
    pub observed_freq: Vec<f64>,
    /// Expected frequencies (should equal prob_bins for calibrated model)
    pub expected_freq: Vec<f64>,
    /// Expected Calibration Error (ECE)
    pub ece: f64,
    /// Maximum Calibration Error (MCE)
    pub mce: f64,
    /// Number of samples in each bin
    pub bin_counts: Vec<usize>,
}

/// Compute calibration statistics for classification
///
/// # Arguments
/// * `predicted_probs` - Predicted probabilities for positive class
/// * `true_labels` - True binary labels (0 or 1)
/// * `n_bins` - Number of probability bins (default 10)
pub fn calibration_analysis(
    predicted_probs: &Array1<f64>,
    true_labels: &Array1<f64>,
    n_bins: usize,
) -> CalibrationResult {
    let n_samples = predicted_probs.len();

    // Create bins
    let mut bin_sums = vec![0.0; n_bins];
    let mut bin_correct = vec![0.0; n_bins];
    let mut bin_counts = vec![0usize; n_bins];

    for (prob, label) in predicted_probs.iter().zip(true_labels.iter()) {
        let bin_idx = ((prob * n_bins as f64).floor() as usize).min(n_bins - 1);
        bin_sums[bin_idx] += prob;
        bin_correct[bin_idx] += label;
        bin_counts[bin_idx] += 1;
    }

    // Compute observed and expected frequencies, and ECE/MCE in one pass
    let mut prob_bins = Vec::new();
    let mut observed_freq = Vec::new();
    let mut expected_freq = Vec::new();
    let mut ece: f64 = 0.0;
    let mut mce: f64 = 0.0;

    for i in 0..n_bins {
        if bin_counts[i] > 0 {
            let avg_prob = bin_sums[i] / bin_counts[i] as f64;
            let accuracy = bin_correct[i] / bin_counts[i] as f64;

            prob_bins.push(avg_prob);
            observed_freq.push(accuracy);
            expected_freq.push(avg_prob);

            // Compute calibration error contribution
            let gap = (accuracy - avg_prob).abs();
            let weight = bin_counts[i] as f64 / n_samples as f64;
            ece += weight * gap;
            mce = mce.max(gap);
        }
    }

    CalibrationResult {
        prob_bins,
        observed_freq,
        expected_freq,
        ece,
        mce,
        bin_counts,
    }
}

/// Reliability diagram data for visualization
#[derive(Debug, Clone)]
pub struct ReliabilityDiagram {
    /// Bin edges
    pub bin_edges: Vec<f64>,
    /// Mean predicted probability in each bin
    pub mean_predicted: Vec<f64>,
    /// Fraction of positives in each bin
    pub fraction_positives: Vec<f64>,
    /// Number of samples in each bin
    pub counts: Vec<usize>,
}

impl From<CalibrationResult> for ReliabilityDiagram {
    fn from(cal: CalibrationResult) -> Self {
        let n = cal.prob_bins.len();
        let bin_edges: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();

        ReliabilityDiagram {
            bin_edges,
            mean_predicted: cal.expected_freq,
            fraction_positives: cal.observed_freq,
            counts: cal.bin_counts.iter().filter(|&&c| c > 0).cloned().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::{Activation, WeightInit};

    #[test]
    fn test_prediction_uncertainty_creation() {
        let mean = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let std = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let lower = &mean - &std.mapv(|s| s * 1.96);
        let upper = &mean + &std.mapv(|s| s * 1.96);

        let pred = PredictionUncertainty {
            mean,
            std,
            lower,
            upper,
            confidence: 0.95,
            n_samples: 100,
            samples: None,
        };

        assert_eq!(pred.n_samples, 100);
        assert!(pred.interval_width()[0] > 0.0);
    }

    #[test]
    fn test_mc_dropout() {
        let mut layers = vec![
            Layer::new(2, 4, Activation::ReLU, WeightInit::HeUniform, Some(42)),
            Layer::new(
                4,
                1,
                Activation::Linear,
                WeightInit::XavierUniform,
                Some(43),
            ),
        ];

        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = predict_with_uncertainty(&mut layers, &x, 50, 0.5, 0.95, Some(42)).unwrap();

        assert_eq!(result.mean.len(), 3);
        assert_eq!(result.std.len(), 3);
        assert!(result.std.iter().all(|&s| s >= 0.0));
        assert_eq!(result.n_samples, 50);
    }

    #[test]
    fn test_calibration_perfect() {
        // Perfectly calibrated model
        let predicted_probs = Array1::from_vec(vec![0.1, 0.3, 0.5, 0.7, 0.9]);
        let true_labels = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0]);

        let result = calibration_analysis(&predicted_probs, &true_labels, 5);

        // ECE should be relatively low for reasonable calibration
        assert!(result.ece < 0.5);
    }

    #[test]
    fn test_calibration_overconfident() {
        // Overconfident model (predicts high probs but only 50% correct)
        let predicted_probs =
            Array1::from_vec(vec![0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]);
        let true_labels = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let result = calibration_analysis(&predicted_probs, &true_labels, 10);

        // ECE should reflect the gap between predicted (0.9) and observed (0.5)
        // With single bin containing all samples, ECE = |0.9 - 0.5| = 0.4
        assert!(result.ece > 0.2, "ECE = {} should be > 0.2", result.ece);
    }

    #[test]
    fn test_reliability_diagram_from_calibration() {
        let predicted_probs = Array1::from_vec(vec![0.1, 0.5, 0.9]);
        let true_labels = Array1::from_vec(vec![0.0, 1.0, 1.0]);

        let cal = calibration_analysis(&predicted_probs, &true_labels, 10);
        let diagram = ReliabilityDiagram::from(cal);

        // bin_edges length = number of non-empty bins + 1
        assert!(diagram.bin_edges.len() > 1);
        assert!(diagram.bin_edges.len() <= 11);
    }

    #[test]
    fn test_coefficient_of_variation() {
        let pred = PredictionUncertainty {
            mean: Array1::from_vec(vec![1.0, 2.0, 0.0]),
            std: Array1::from_vec(vec![0.1, 0.4, 0.1]),
            lower: Array1::zeros(3),
            upper: Array1::zeros(3),
            confidence: 0.95,
            n_samples: 100,
            samples: None,
        };

        let cv = pred.coefficient_of_variation();
        assert!((cv[0] - 0.1).abs() < 1e-10);
        assert!((cv[1] - 0.2).abs() < 1e-10);
        // For mean=0, CV should be large but finite
        assert!(cv[2].is_finite());
    }
}

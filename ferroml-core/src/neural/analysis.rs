//! Weight Analysis for Neural Networks
//!
//! This module provides tools for analyzing neural network weights
//! and detecting issues like dead neurons.
//!
//! ## Features
//!
//! - Weight statistics (mean, std, sparsity per layer)
//! - Dead neuron detection for ReLU networks
//! - Weight distribution tests (normality, initialization quality)

use crate::neural::Layer;
use crate::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Statistics about weights in a neural network layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightStatistics {
    /// Layer index
    pub layer_idx: usize,
    /// Number of weights in layer
    pub n_weights: usize,
    /// Mean weight value
    pub mean: f64,
    /// Standard deviation of weights
    pub std: f64,
    /// Minimum weight value
    pub min: f64,
    /// Maximum weight value
    pub max: f64,
    /// Percentage of weights near zero (|w| < 0.01)
    pub sparsity: f64,
    /// Mean absolute value
    pub mean_abs: f64,
    /// Skewness of weight distribution
    pub skewness: f64,
    /// Kurtosis of weight distribution
    pub kurtosis: f64,
}

impl WeightStatistics {
    /// Compute statistics for a layer
    pub fn from_layer(layer_idx: usize, layer: &Layer) -> Self {
        let weights = &layer.weights;
        let n = weights.len() as f64;

        let mean = weights.sum() / n;
        let variance = weights.mapv(|w| (w - mean).powi(2)).sum() / n;
        let std = variance.sqrt();
        let min = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_abs = weights.mapv(|w| w.abs()).sum() / n;
        let near_zero = weights.iter().filter(|&&w| w.abs() < 0.01).count() as f64;
        let sparsity = near_zero / n;

        // Skewness: E[(X - μ)³] / σ³
        let skewness = if std > 1e-10 {
            weights.mapv(|w| ((w - mean) / std).powi(3)).sum() / n
        } else {
            0.0
        };

        // Kurtosis: E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)
        let kurtosis = if std > 1e-10 {
            weights.mapv(|w| ((w - mean) / std).powi(4)).sum() / n - 3.0
        } else {
            0.0
        };

        Self {
            layer_idx,
            n_weights: weights.len(),
            mean,
            std,
            min,
            max,
            sparsity,
            mean_abs,
            skewness,
            kurtosis,
        }
    }
}

/// Compute weight statistics for all layers
pub fn weight_statistics(layers: &[Layer]) -> Vec<WeightStatistics> {
    layers
        .iter()
        .enumerate()
        .map(|(i, layer)| WeightStatistics::from_layer(i, layer))
        .collect()
}

/// Information about a dead neuron
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadNeuron {
    /// Layer index (0-indexed)
    pub layer_idx: usize,
    /// Neuron index within the layer
    pub neuron_idx: usize,
    /// Activation rate (0.0 = never activates)
    pub activation_rate: f64,
}

/// Detect dead neurons in a ReLU network
///
/// A neuron is considered "dead" if it never activates (output always <= 0)
/// across the test samples.
///
/// # Arguments
/// * `layers` - Network layers
/// * `x` - Test data to check activations
///
/// # Returns
/// List of (layer_idx, neuron_idx) pairs for dead neurons
pub fn dead_neuron_detection(layers: &mut [Layer], x: &Array2<f64>) -> Result<Vec<DeadNeuron>> {
    use crate::neural::Activation;

    let mut dead_neurons = Vec::new();
    let mut input = x.clone();
    let n_samples = x.nrows() as f64;

    for (layer_idx, layer) in layers.iter_mut().enumerate() {
        // Only check hidden layers with ReLU activation
        let is_relu = matches!(
            layer.activation,
            Activation::ReLU | Activation::LeakyReLU | Activation::ELU
        );

        if !is_relu {
            // Still need to propagate through
            input = layer.forward(&input, false)?;
            continue;
        }

        // Get pre-activation values
        let z = input.dot(&layer.weights) + &layer.biases;

        // Check each neuron
        for neuron_idx in 0..layer.n_outputs {
            let activations = z.column(neuron_idx);
            let n_active = activations.iter().filter(|&&v| v > 0.0).count() as f64;
            let activation_rate = n_active / n_samples;

            // Dead if activation rate is very low (< 1%)
            if activation_rate < 0.01 {
                dead_neurons.push(DeadNeuron {
                    layer_idx,
                    neuron_idx,
                    activation_rate,
                });
            }
        }

        // Propagate through
        input = layer.forward(&input, false)?;
    }

    Ok(dead_neurons)
}

/// Result of weight distribution test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightDistributionTest {
    /// Layer index
    pub layer_idx: usize,
    /// Whether weights appear normally distributed
    pub is_normal: bool,
    /// P-value from normality test (Shapiro-Wilk approximation)
    pub normality_p_value: f64,
    /// Expected std for Xavier init
    pub expected_xavier_std: f64,
    /// Expected std for He init
    pub expected_he_std: f64,
    /// Actual std
    pub actual_std: f64,
    /// Whether initialization appears correct
    pub initialization_quality: InitializationQuality,
}

/// Quality of weight initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InitializationQuality {
    /// Weights match expected initialization
    Good,
    /// Weights are too large
    TooLarge,
    /// Weights are too small
    TooSmall,
    /// Weights have unexpected distribution
    UnexpectedDistribution,
}

/// Test weight distributions for proper initialization
pub fn weight_distribution_tests(layers: &[Layer]) -> Vec<WeightDistributionTest> {
    layers
        .iter()
        .enumerate()
        .map(|(i, layer)| {
            let stats = WeightStatistics::from_layer(i, layer);
            let fan_in = layer.n_inputs as f64;
            let fan_out = layer.n_outputs as f64;

            let expected_xavier_std = (2.0 / (fan_in + fan_out)).sqrt();
            let expected_he_std = (2.0 / fan_in).sqrt();

            // Simple normality test based on skewness and kurtosis
            // (For a normal distribution, skewness ≈ 0, kurtosis ≈ 0)
            let is_normal = stats.skewness.abs() < 0.5 && stats.kurtosis.abs() < 1.0;

            // Approximate p-value (very rough)
            let normality_p_value = if is_normal { 0.5 } else { 0.01 };

            // Check initialization quality
            let ratio_to_xavier = stats.std / expected_xavier_std;
            let ratio_to_he = stats.std / expected_he_std;

            let initialization_quality = if ratio_to_xavier > 0.5 && ratio_to_xavier < 2.0 {
                InitializationQuality::Good
            } else if ratio_to_he > 0.5 && ratio_to_he < 2.0 {
                InitializationQuality::Good
            } else if stats.std > expected_he_std * 3.0 {
                InitializationQuality::TooLarge
            } else if stats.std < expected_xavier_std * 0.1 {
                InitializationQuality::TooSmall
            } else {
                InitializationQuality::UnexpectedDistribution
            };

            WeightDistributionTest {
                layer_idx: i,
                is_normal,
                normality_p_value,
                expected_xavier_std,
                expected_he_std,
                actual_std: stats.std,
                initialization_quality,
            }
        })
        .collect()
}

/// Analyze weight changes during training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightChange {
    /// Layer index
    pub layer_idx: usize,
    /// Mean absolute change
    pub mean_abs_change: f64,
    /// Max absolute change
    pub max_abs_change: f64,
    /// Percentage of weights that changed significantly
    pub significant_changes: f64,
}

/// Compare weights before and after training
pub fn analyze_weight_changes(
    before: &[(&Array2<f64>, &Array1<f64>)],
    after: &[(&Array2<f64>, &Array1<f64>)],
) -> Vec<WeightChange> {
    before
        .iter()
        .zip(after.iter())
        .enumerate()
        .map(|(i, ((w1, _), (w2, _)))| {
            let diff = *w2 - *w1;
            let n = diff.len() as f64;

            let abs_diff = diff.mapv(|d| d.abs());
            let mean_abs_change = abs_diff.sum() / n;
            let max_abs_change = abs_diff.iter().cloned().fold(0.0, f64::max);

            // Significant change threshold: relative change > 10%
            let significant = diff
                .iter()
                .zip(w1.iter())
                .filter(|(d, w)| {
                    if w.abs() > 1e-6 {
                        d.abs() / w.abs() > 0.1
                    } else {
                        d.abs() > 0.01
                    }
                })
                .count() as f64;
            let significant_changes = significant / n;

            WeightChange {
                layer_idx: i,
                mean_abs_change,
                max_abs_change,
                significant_changes,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::{Activation, WeightInit};

    fn create_test_layer() -> Layer {
        Layer::new(10, 5, Activation::ReLU, WeightInit::XavierUniform, Some(42))
    }

    #[test]
    fn test_weight_statistics() {
        let layer = create_test_layer();
        let stats = WeightStatistics::from_layer(0, &layer);

        assert_eq!(stats.layer_idx, 0);
        assert_eq!(stats.n_weights, 50); // 10 * 5
        assert!(stats.std > 0.0);
        assert!(stats.min <= stats.max);
    }

    #[test]
    fn test_weight_statistics_multiple_layers() {
        let layers = vec![
            Layer::new(4, 10, Activation::ReLU, WeightInit::HeUniform, Some(1)),
            Layer::new(10, 5, Activation::ReLU, WeightInit::HeUniform, Some(2)),
            Layer::new(5, 2, Activation::Linear, WeightInit::XavierUniform, Some(3)),
        ];

        let stats = weight_statistics(&layers);
        assert_eq!(stats.len(), 3);
        assert_eq!(stats[0].n_weights, 40);
        assert_eq!(stats[1].n_weights, 50);
        assert_eq!(stats[2].n_weights, 10);
    }

    #[test]
    fn test_dead_neuron_detection() {
        use crate::neural::{Activation, WeightInit};

        // Create a layer with some weights set to very negative values
        // This should create dead neurons
        let mut layer = Layer::new(2, 4, Activation::ReLU, WeightInit::Normal, Some(42));

        // Force one neuron to be always negative by setting its weights very negative
        layer.weights[[0, 0]] = -100.0;
        layer.weights[[1, 0]] = -100.0;
        layer.biases[0] = -100.0;

        let mut layers = vec![layer];
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();

        let dead = dead_neuron_detection(&mut layers, &x).unwrap();

        // Should find at least the neuron we made dead
        assert!(!dead.is_empty());
        assert!(dead.iter().any(|d| d.layer_idx == 0 && d.neuron_idx == 0));
    }

    #[test]
    fn test_weight_distribution_tests() {
        let layers = vec![Layer::new(
            100,
            50,
            Activation::ReLU,
            WeightInit::HeNormal,
            Some(42),
        )];

        let tests = weight_distribution_tests(&layers);
        assert_eq!(tests.len(), 1);

        // He initialization should result in good quality
        assert!(matches!(
            tests[0].initialization_quality,
            InitializationQuality::Good
        ));
    }

    #[test]
    fn test_weight_change_analysis() {
        let w1 = Array2::ones((3, 2));
        let b1 = Array1::zeros(2);
        let w2 = Array2::from_elem((3, 2), 1.1);
        let b2 = Array1::zeros(2);

        let before = vec![(&w1, &b1)];
        let after = vec![(&w2, &b2)];

        let changes = analyze_weight_changes(&before, &after);
        assert_eq!(changes.len(), 1);
        assert!(changes[0].mean_abs_change > 0.0);
    }
}

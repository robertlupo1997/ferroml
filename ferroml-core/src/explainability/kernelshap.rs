//! KernelSHAP Implementation for Model-Agnostic Explanations
//!
//! This module implements the KernelSHAP algorithm for computing approximate Shapley values
//! for any black-box model. KernelSHAP uses a weighted linear regression approach to
//! approximate SHAP values without requiring access to model internals.
//!
//! ## Algorithm
//!
//! KernelSHAP approximates Shapley values by:
//! 1. Sampling coalitions (subsets of features)
//! 2. Creating perturbed instances by masking features with background values
//! 3. Getting model predictions for perturbed instances
//! 4. Fitting a weighted linear regression where weights follow the SHAP kernel
//!
//! The SHAP kernel weight for a coalition of size |z| out of M features is:
//! `w(z) = (M - 1) / (C(M, |z|) * |z| * (M - |z|))`
//!
//! ## Supported Models
//!
//! Works with any model implementing the `Model` trait from `ferroml_core::models`.
//!
//! ## Example
//!
//! ```
//! # fn main() -> ferroml_core::Result<()> {
//! use ferroml_core::explainability::{KernelExplainer, KernelSHAPConfig};
//! use ferroml_core::models::RandomForestRegressor;
//! # use ferroml_core::models::Model;
//! # use ndarray::{Array1, Array2};
//! # let x_train = Array2::from_shape_vec((20, 3), (0..60).map(|i| i as f64 / 60.0).collect()).unwrap();
//! # let y_train = Array1::from_vec((0..20).map(|i| i as f64).collect());
//! # let x_test = x_train.clone();
//!
//! let mut model = RandomForestRegressor::new();
//! model.fit(&x_train, &y_train)?;
//!
//! // Create KernelSHAP explainer with background data
//! let explainer = KernelExplainer::new(&model, &x_train, KernelSHAPConfig::default())?;
//!
//! // Explain a single prediction
//! let result = explainer.explain(&x_test.row(0).to_vec())?;
//! println!("Base value: {}", result.base_value);
//! println!("SHAP values: {:?}", result.shap_values);
//!
//! // Explain multiple predictions
//! let results = explainer.explain_batch(&x_test)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## References
//!
//! - Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting
//!   Model Predictions. NeurIPS 2017.

use crate::models::Model;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// Re-use result types from TreeSHAP
use super::treeshap::{SHAPBatchResult, SHAPResult};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for KernelSHAP explainer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSHAPConfig {
    /// Number of coalition samples to use for approximation
    /// Higher values give more accurate estimates but are slower
    /// Default: 2 * n_features + 2048
    pub n_samples: Option<usize>,

    /// Maximum number of background samples to use
    /// If the background data has more samples, a random subset is selected
    /// Default: 100
    pub max_background_samples: usize,

    /// Random seed for reproducibility
    pub random_state: Option<u64>,

    /// Regularization parameter for weighted regression
    /// Small positive value for numerical stability
    /// Default: 0.01
    pub regularization: f64,

    /// Whether to use paired sampling (sample coalition and its complement together)
    /// This reduces variance in the estimates
    /// Default: true
    pub paired_sampling: bool,
}

impl Default for KernelSHAPConfig {
    fn default() -> Self {
        Self {
            n_samples: None,
            max_background_samples: 100,
            random_state: None,
            regularization: 0.01,
            paired_sampling: true,
        }
    }
}

impl KernelSHAPConfig {
    /// Create a new configuration with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of coalition samples
    #[must_use]
    pub fn with_n_samples(mut self, n: usize) -> Self {
        self.n_samples = Some(n);
        self
    }

    /// Set the maximum number of background samples
    #[must_use]
    pub fn with_max_background_samples(mut self, n: usize) -> Self {
        self.max_background_samples = n;
        self
    }

    /// Set the random seed
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the regularization parameter
    #[must_use]
    pub fn with_regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Enable or disable paired sampling
    #[must_use]
    pub fn with_paired_sampling(mut self, enabled: bool) -> Self {
        self.paired_sampling = enabled;
        self
    }
}

// =============================================================================
// Simple RNG for reproducibility
// =============================================================================

/// Simple xorshift64 random number generator for reproducibility
#[derive(Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x853c_49e6_748f_ea9b
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a random f64 in [0, 1) (kept for API completeness)
    fn _next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    fn gen_range(&mut self, min: usize, max: usize) -> usize {
        if min >= max {
            return min;
        }
        min + (self.next_u64() as usize % (max - min))
    }

    /// Shuffle a slice in place using Fisher-Yates algorithm
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.gen_range(0, i + 1);
            slice.swap(i, j);
        }
    }
}

// =============================================================================
// KernelExplainer
// =============================================================================

/// KernelSHAP explainer for model-agnostic SHAP value computation
///
/// Uses weighted linear regression to approximate Shapley values for any model.
/// This is slower than TreeSHAP but works with any black-box model.
#[derive(Clone)]
pub struct KernelExplainer<'a, M: Model> {
    /// Reference to the model being explained
    model: &'a M,
    /// Background data for computing expected values
    background: Array2<f64>,
    /// Number of features
    n_features: usize,
    /// Expected model output (base value)
    base_value: f64,
    /// Configuration
    config: KernelSHAPConfig,
    /// Feature names (if available)
    feature_names: Option<Vec<String>>,
}

impl<'a, M: Model> KernelExplainer<'a, M> {
    /// Create a new KernelSHAP explainer
    ///
    /// # Arguments
    /// * `model` - The fitted model to explain
    /// * `background` - Background dataset for computing expected values
    /// * `config` - Configuration options
    ///
    /// # Returns
    /// A new `KernelExplainer` instance
    ///
    /// # Errors
    /// Returns an error if the model is not fitted or inputs are invalid
    pub fn new(model: &'a M, background: &Array2<f64>, config: KernelSHAPConfig) -> Result<Self> {
        if !model.is_fitted() {
            return Err(FerroError::not_fitted("KernelExplainer::new"));
        }

        let n_features = background.ncols();
        if n_features == 0 {
            return Err(FerroError::invalid_input(
                "Background data must have at least one feature",
            ));
        }

        // Subsample background if needed
        let mut rng = SimpleRng::new(config.random_state.unwrap_or(42));
        let background = if background.nrows() > config.max_background_samples {
            let mut indices: Vec<usize> = (0..background.nrows()).collect();
            rng.shuffle(&mut indices);
            let selected: Vec<usize> = indices
                .into_iter()
                .take(config.max_background_samples)
                .collect();
            background.select(Axis(0), &selected)
        } else {
            background.clone()
        };

        // Compute base value (expected model output over background)
        let predictions = model.predict(&background)?;
        let base_value = predictions.mean().unwrap_or(0.0);

        Ok(Self {
            model,
            background,
            n_features,
            base_value,
            config,
            feature_names: model.feature_names().map(|n| n.to_vec()),
        })
    }

    /// Get the base value (expected model output)
    #[must_use]
    pub fn base_value(&self) -> f64 {
        self.base_value
    }

    /// Get the number of features
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Set feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Explain a single prediction using KernelSHAP
    ///
    /// # Arguments
    /// * `x` - Feature values for a single sample
    ///
    /// # Returns
    /// SHAP values for the prediction
    pub fn explain(&self, x: &[f64]) -> Result<SHAPResult> {
        if x.len() != self.n_features {
            return Err(FerroError::shape_mismatch(
                format!("Expected {} features", self.n_features),
                format!("Got {} features", x.len()),
            ));
        }

        let shap_values = self.compute_shap_values(x)?;

        Ok(SHAPResult {
            base_value: self.base_value,
            shap_values,
            feature_values: Array1::from_vec(x.to_vec()),
            n_features: self.n_features,
            feature_names: self.feature_names.clone(),
        })
    }

    /// Explain multiple predictions in batch
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples, n_features)
    ///
    /// # Returns
    /// SHAP values for all predictions
    pub fn explain_batch(&self, x: &Array2<f64>) -> Result<SHAPBatchResult> {
        if x.ncols() != self.n_features {
            return Err(FerroError::shape_mismatch(
                format!("Expected {} features", self.n_features),
                format!("Got {} features", x.ncols()),
            ));
        }

        let n_samples = x.nrows();
        let mut shap_values = Array2::zeros((n_samples, self.n_features));

        for i in 0..n_samples {
            let sample: Vec<f64> = x.row(i).to_vec();
            let result = self.explain(&sample)?;
            for j in 0..self.n_features {
                shap_values[[i, j]] = result.shap_values[j];
            }
        }

        Ok(SHAPBatchResult {
            base_value: self.base_value,
            shap_values,
            feature_values: x.clone(),
            n_samples,
            n_features: self.n_features,
            feature_names: self.feature_names.clone(),
        })
    }

    /// Explain multiple predictions in parallel
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples, n_features)
    ///
    /// # Returns
    /// SHAP values for all predictions
    pub fn explain_batch_parallel(&self, x: &Array2<f64>) -> Result<SHAPBatchResult>
    where
        M: Sync,
    {
        if x.ncols() != self.n_features {
            return Err(FerroError::shape_mismatch(
                format!("Expected {} features", self.n_features),
                format!("Got {} features", x.ncols()),
            ));
        }

        let n_samples = x.nrows();

        // Compute SHAP values in parallel
        let sample_results: Vec<Result<Array1<f64>>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let sample: Vec<f64> = x.row(i).to_vec();
                self.compute_shap_values(&sample)
            })
            .collect();

        // Collect results and check for errors
        let mut shap_values = Array2::zeros((n_samples, self.n_features));
        for (i, result) in sample_results.into_iter().enumerate() {
            let vals = result?;
            for j in 0..self.n_features {
                shap_values[[i, j]] = vals[j];
            }
        }

        Ok(SHAPBatchResult {
            base_value: self.base_value,
            shap_values,
            feature_values: x.clone(),
            n_samples,
            n_features: self.n_features,
            feature_names: self.feature_names.clone(),
        })
    }

    /// Compute SHAP values for a single sample using KernelSHAP algorithm
    fn compute_shap_values(&self, x: &[f64]) -> Result<Array1<f64>> {
        let m = self.n_features;

        // Determine number of samples
        let n_samples = self
            .config
            .n_samples
            .unwrap_or_else(|| 2 * m + 2048)
            .max(2 * m);

        // Generate coalition samples and compute predictions
        let (coalitions, weights, predictions) = self.sample_coalitions(x, n_samples)?;

        // Solve weighted linear regression to get SHAP values
        self.solve_weighted_regression(&coalitions, &weights, &predictions)
    }

    /// Sample coalitions and compute model predictions for perturbed instances
    fn sample_coalitions(
        &self,
        x: &[f64],
        n_samples: usize,
    ) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
        let m = self.n_features;
        let _n_background = self.background.nrows();

        // Initialize RNG
        let mut rng = SimpleRng::new(self.config.random_state.unwrap_or(42));

        // Vectors to store coalition masks, weights, and predictions
        let mut coalitions: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        let mut weights: Vec<f64> = Vec::with_capacity(n_samples);
        let mut predictions: Vec<f64> = Vec::with_capacity(n_samples);

        // Always include the empty and full coalitions
        // Empty coalition (all background)
        coalitions.push(vec![0.0; m]);
        weights.push(1e6); // Very high weight for boundary conditions
        let empty_pred = self.predict_coalition(x, &vec![false; m])?;
        predictions.push(empty_pred);

        // Full coalition (all original)
        coalitions.push(vec![1.0; m]);
        weights.push(1e6);
        let full_pred = self.predict_coalition(x, &vec![true; m])?;
        predictions.push(full_pred);

        // Sample random coalitions
        let samples_to_generate = n_samples.saturating_sub(2);
        let pairs_to_generate = if self.config.paired_sampling {
            samples_to_generate / 2
        } else {
            samples_to_generate
        };

        for _ in 0..pairs_to_generate {
            // Generate random coalition size (exclude 0 and M)
            let size = rng.gen_range(1, m);

            // Generate random coalition of that size
            let mut indices: Vec<usize> = (0..m).collect();
            rng.shuffle(&mut indices);
            let selected: Vec<usize> = indices.into_iter().take(size).collect();

            // Create coalition mask
            let mut mask = vec![false; m];
            for &idx in &selected {
                mask[idx] = true;
            }

            // Compute SHAP kernel weight
            let weight = self.shap_kernel_weight(size, m);

            // Add coalition
            coalitions.push(mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect());
            weights.push(weight);
            let pred = self.predict_coalition(x, &mask)?;
            predictions.push(pred);

            // Add complementary coalition if paired sampling
            if self.config.paired_sampling && coalitions.len() < n_samples {
                let comp_mask: Vec<bool> = mask.iter().map(|&b| !b).collect();
                let comp_size = m - size;
                let comp_weight = self.shap_kernel_weight(comp_size, m);

                coalitions.push(
                    comp_mask
                        .iter()
                        .map(|&b| if b { 1.0 } else { 0.0 })
                        .collect(),
                );
                weights.push(comp_weight);
                let comp_pred = self.predict_coalition(x, &comp_mask)?;
                predictions.push(comp_pred);
            }
        }

        // Pad with random samples if needed
        while coalitions.len() < n_samples {
            let size = rng.gen_range(1, m);
            let mut indices: Vec<usize> = (0..m).collect();
            rng.shuffle(&mut indices);
            let selected: Vec<usize> = indices.into_iter().take(size).collect();

            let mut mask = vec![false; m];
            for &idx in &selected {
                mask[idx] = true;
            }

            let weight = self.shap_kernel_weight(size, m);
            coalitions.push(mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect());
            weights.push(weight);
            let pred = self.predict_coalition(x, &mask)?;
            predictions.push(pred);
        }

        // Convert to arrays
        let n_coalitions = coalitions.len();
        let coalitions_arr = Array2::from_shape_vec(
            (n_coalitions, m),
            coalitions.into_iter().flatten().collect(),
        )
        .map_err(|e| FerroError::numerical(format!("Failed to create coalition array: {}", e)))?;

        let weights_arr = Array1::from_vec(weights);
        let predictions_arr = Array1::from_vec(predictions);

        Ok((coalitions_arr, weights_arr, predictions_arr))
    }

    /// Compute model prediction for a coalition (masked instance)
    fn predict_coalition(&self, x: &[f64], mask: &[bool]) -> Result<f64> {
        let n_background = self.background.nrows();

        // Create perturbed instances by replacing masked features with background values
        // and average predictions over all background samples
        let mut total_pred = 0.0;

        for bg_idx in 0..n_background {
            let mut perturbed = vec![0.0; self.n_features];
            for j in 0..self.n_features {
                if mask[j] {
                    // Use original feature value
                    perturbed[j] = x[j];
                } else {
                    // Use background value
                    perturbed[j] = self.background[[bg_idx, j]];
                }
            }

            // Get prediction for this perturbed instance
            let x_perturbed =
                Array2::from_shape_vec((1, self.n_features), perturbed).map_err(|e| {
                    FerroError::numerical(format!("Failed to create perturbed array: {}", e))
                })?;

            let pred = self.model.predict(&x_perturbed)?;
            total_pred += pred[0];
        }

        Ok(total_pred / n_background as f64)
    }

    /// Compute SHAP kernel weight for a coalition
    ///
    /// The SHAP kernel weight for a coalition of size |z| out of M features is:
    /// w(z) = (M - 1) / (C(M, |z|) * |z| * (M - |z|))
    fn shap_kernel_weight(&self, coalition_size: usize, total_features: usize) -> f64 {
        let m = total_features as f64;
        let z = coalition_size as f64;

        if coalition_size == 0 || coalition_size == total_features {
            // Boundary conditions have infinite weight, we handle them separately
            return 1e6;
        }

        // C(M, |z|) = M! / (|z|! * (M-|z|)!)
        // Use log to avoid overflow for large M
        let log_binom = log_binomial(total_features, coalition_size);
        let binom = log_binom.exp();

        (m - 1.0) / (binom * z * (m - z))
    }

    /// Solve weighted linear regression to get SHAP values
    ///
    /// We solve: minimize sum_i w_i * (f(z_i) - (phi_0 + sum_j phi_j * z_ij))^2
    /// where phi_0 = base_value (fixed), phi_j are SHAP values
    ///
    /// This simplifies to solving: (Z^T W Z + lambda*I) phi = Z^T W (y - phi_0)
    fn solve_weighted_regression(
        &self,
        coalitions: &Array2<f64>,
        weights: &Array1<f64>,
        predictions: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n = coalitions.nrows();
        let m = self.n_features;

        // Adjust predictions by subtracting base value
        let y_centered: Array1<f64> = predictions - self.base_value;

        // Build weighted normal equations: (Z^T W Z + lambda*I) phi = Z^T W y
        // where Z is the coalition matrix, W is diagonal weight matrix

        // Compute Z^T W Z
        let mut zt_w_z = Array2::zeros((m, m));
        for i in 0..n {
            let w = weights[i];
            for j in 0..m {
                for k in 0..m {
                    zt_w_z[[j, k]] += w * coalitions[[i, j]] * coalitions[[i, k]];
                }
            }
        }

        // Add regularization to diagonal
        for j in 0..m {
            zt_w_z[[j, j]] += self.config.regularization;
        }

        // Compute Z^T W y
        let mut zt_w_y = Array1::zeros(m);
        for i in 0..n {
            let w = weights[i];
            for j in 0..m {
                zt_w_y[j] += w * coalitions[[i, j]] * y_centered[i];
            }
        }

        // Solve the system using Cholesky decomposition
        let phi = self.solve_cholesky(&zt_w_z, &zt_w_y)?;

        Ok(phi)
    }

    /// Solve a positive definite linear system using Cholesky decomposition
    fn solve_cholesky(&self, a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        let n = a.nrows();

        // Cholesky decomposition: A = L L^T
        let mut l = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }

                if i == j {
                    let diag = a[[i, i]] - sum;
                    if diag <= 0.0 {
                        // Matrix is not positive definite, fall back to regularized solution
                        return self.solve_regularized(a, b);
                    }
                    l[[i, j]] = diag.sqrt();
                } else {
                    l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        // Solve L y = b (forward substitution)
        let mut y = Array1::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[[i, j]] * y[j];
            }
            y[i] = (b[i] - sum) / l[[i, i]];
        }

        // Solve L^T x = y (backward substitution)
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += l[[j, i]] * x[j];
            }
            x[i] = (y[i] - sum) / l[[i, i]];
        }

        Ok(x)
    }

    /// Fallback regularized solver for non-positive-definite matrices
    fn solve_regularized(&self, a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        let n = a.nrows();

        // Add stronger regularization
        let mut a_reg = a.clone();
        for i in 0..n {
            a_reg[[i, i]] += 1.0;
        }

        // Simple iterative solver (conjugate gradient-like)
        let mut x: Array1<f64> = Array1::zeros(n);
        let max_iter = 100;
        let tol = 1e-6;

        for _ in 0..max_iter {
            let mut x_new: Array1<f64> = Array1::zeros(n);

            for i in 0..n {
                let mut sum = 0.0_f64;
                for j in 0..n {
                    if i != j {
                        sum += a_reg[[i, j]] * x[j];
                    }
                }
                x_new[i] = (b[i] - sum) / a_reg[[i, i]];
            }

            // Check convergence
            let mut diff = 0.0_f64;
            for i in 0..n {
                let val: f64 = x_new[i] - x[i];
                diff += val.abs();
            }
            x = x_new;

            if diff < tol * n as f64 {
                break;
            }
        }

        Ok(x)
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute log of binomial coefficient using Stirling's approximation for large values
fn log_binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    if k == 0 || k == n {
        return 0.0;
    }

    // log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
    log_factorial(n) - log_factorial(k) - log_factorial(n - k)
}

/// Compute log factorial using Stirling's approximation for large values
fn log_factorial(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }

    // For small n, compute directly
    if n <= 20 {
        let mut result = 0.0;
        for i in 2..=n {
            result += (i as f64).ln();
        }
        return result;
    }

    // Stirling's approximation for large n
    let n_f = n as f64;
    0.5f64.mul_add(
        (2.0 * std::f64::consts::PI * n_f).ln(),
        n_f.mul_add(n_f.ln(), -n_f),
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::linear::LinearRegression;
    use crate::models::Model;

    fn make_simple_regression_data() -> (Array2<f64>, Array1<f64>) {
        // Simple linear relationship: y = 2*x0 + 3*x1 + 1
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 4.0, 2.0, 5.0, 1.0, 6.0, 4.0, 7.0, 3.0, 8.0, 5.0,
                9.0, 2.0, 10.0, 4.0, 1.5, 3.0, 2.5, 2.0, 3.5, 4.0, 4.5, 1.0, 5.5, 3.0, 6.5, 2.0,
                7.5, 4.0, 8.5, 1.0, 9.5, 3.0, 10.5, 2.0,
            ],
        )
        .unwrap();
        let y: Array1<f64> = x.column(0).to_owned() * 2.0 + x.column(1).to_owned() * 3.0 + 1.0;
        (x, y)
    }

    #[test]
    fn test_kernel_explainer_creation() {
        let (x, y) = make_simple_regression_data();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let explainer = KernelExplainer::new(&model, &x, KernelSHAPConfig::default());
        assert!(explainer.is_ok());

        let explainer = explainer.unwrap();
        assert_eq!(explainer.n_features(), 2);
    }

    #[test]
    fn test_kernel_explainer_not_fitted() {
        let x = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();

        let model = LinearRegression::new();
        let result = KernelExplainer::new(&model, &x, KernelSHAPConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_kernel_explainer_explain() {
        let (x, y) = make_simple_regression_data();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = KernelSHAPConfig::new()
            .with_n_samples(512)
            .with_random_state(42);

        let explainer = KernelExplainer::new(&model, &x, config).unwrap();

        let sample = vec![5.0, 3.0];
        let result = explainer.explain(&sample).unwrap();

        assert_eq!(result.n_features, 2);
        assert_eq!(result.shap_values.len(), 2);

        // For a linear model, SHAP values should approximate the feature contributions
        // y = 2*x0 + 3*x1 + 1
        // The contribution of x0 should be roughly proportional to its coefficient
        // and the contribution of x1 should be roughly proportional to its coefficient
    }

    #[test]
    fn test_kernel_explainer_explain_batch() {
        let (x, y) = make_simple_regression_data();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = KernelSHAPConfig::new()
            .with_n_samples(256)
            .with_random_state(42);

        let explainer = KernelExplainer::new(&model, &x, config).unwrap();

        let test_x = x.slice(ndarray::s![0..3, ..]).to_owned();
        let result = explainer.explain_batch(&test_x).unwrap();

        assert_eq!(result.n_samples, 3);
        assert_eq!(result.n_features, 2);
        assert_eq!(result.shap_values.shape(), &[3, 2]);
    }

    #[test]
    fn test_kernel_explainer_wrong_features() {
        let (x, y) = make_simple_regression_data();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let explainer = KernelExplainer::new(&model, &x, KernelSHAPConfig::default()).unwrap();

        // Wrong number of features
        let sample = vec![1.0, 2.0, 3.0];
        let result = explainer.explain(&sample);
        assert!(result.is_err());
    }

    #[test]
    fn test_shap_kernel_weight() {
        // The SHAP kernel weight should be higher for coalitions closer to empty or full
        let (x, y) = make_simple_regression_data();
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let explainer = KernelExplainer::new(&model, &x, KernelSHAPConfig::default()).unwrap();

        // For M=10 features
        let m = 10;

        // Weight for coalition of size 1 should be higher than size 5
        let w1 = explainer.shap_kernel_weight(1, m);
        let w5 = explainer.shap_kernel_weight(5, m);

        assert!(w1 > w5, "w1={}, w5={}", w1, w5);
    }

    #[test]
    fn test_shap_result_additivity() {
        let (x, y) = make_simple_regression_data();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = KernelSHAPConfig::new()
            .with_n_samples(1024)
            .with_random_state(42);

        let explainer = KernelExplainer::new(&model, &x, config).unwrap();

        let sample = vec![5.0, 3.0];
        let result = explainer.explain(&sample).unwrap();

        // The sum of SHAP values + base value should approximately equal the prediction
        let reconstructed = result.prediction();

        let x_sample = Array2::from_shape_vec((1, 2), sample).unwrap();
        let actual = model.predict(&x_sample).unwrap()[0];

        // Allow some tolerance for the approximation
        let diff = (reconstructed - actual).abs();
        assert!(
            diff < 1.0,
            "Reconstructed {} vs actual {}, diff = {}",
            reconstructed,
            actual,
            diff
        );
    }

    #[test]
    fn test_config_builder() {
        let config = KernelSHAPConfig::new()
            .with_n_samples(500)
            .with_max_background_samples(50)
            .with_random_state(123)
            .with_regularization(0.1)
            .with_paired_sampling(false);

        assert_eq!(config.n_samples, Some(500));
        assert_eq!(config.max_background_samples, 50);
        assert_eq!(config.random_state, Some(123));
        assert!((config.regularization - 0.1).abs() < 1e-10);
        assert!(!config.paired_sampling);
    }

    #[test]
    fn test_log_binomial() {
        // C(5, 2) = 10
        let log_c = log_binomial(5, 2);
        assert!((log_c.exp() - 10.0).abs() < 1e-6);

        // C(10, 0) = 1
        let log_c = log_binomial(10, 0);
        assert!((log_c.exp() - 1.0).abs() < 1e-6);

        // C(10, 10) = 1
        let log_c = log_binomial(10, 10);
        assert!((log_c.exp() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_result_methods() {
        let (x, y) = make_simple_regression_data();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = KernelSHAPConfig::new()
            .with_n_samples(256)
            .with_random_state(42);

        let explainer = KernelExplainer::new(&model, &x, config).unwrap();

        let test_x = x.slice(ndarray::s![0..5, ..]).to_owned();
        let result = explainer.explain_batch(&test_x).unwrap();

        // Test mean_abs_shap
        let mean_abs = result.mean_abs_shap();
        assert_eq!(mean_abs.len(), 2);
        assert!(mean_abs[0] >= 0.0);
        assert!(mean_abs[1] >= 0.0);

        // Test get_sample
        let sample_result = result.get_sample(0);
        assert!(sample_result.is_some());

        let sample_result = result.get_sample(100);
        assert!(sample_result.is_none());

        // Test global_importance_sorted
        let sorted = result.global_importance_sorted();
        assert_eq!(sorted.len(), 2);
    }

    #[test]
    fn test_feature_names() {
        let (x, y) = make_simple_regression_data();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let explainer = KernelExplainer::new(&model, &x, KernelSHAPConfig::default())
            .unwrap()
            .with_feature_names(vec!["feature_a".to_string(), "feature_b".to_string()]);

        let sample = vec![5.0, 3.0];
        let result = explainer.explain(&sample).unwrap();

        assert_eq!(
            result.feature_names,
            Some(vec!["feature_a".to_string(), "feature_b".to_string()])
        );
    }

    #[test]
    fn test_background_subsampling() {
        // Create large background data
        let (x, y) = make_simple_regression_data();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // Create config with small max_background_samples
        let config = KernelSHAPConfig::new()
            .with_max_background_samples(5)
            .with_random_state(42);

        let explainer = KernelExplainer::new(&model, &x, config).unwrap();

        // Should still work with subsampled background
        let sample = vec![5.0, 3.0];
        let result = explainer.explain(&sample);
        assert!(result.is_ok());
    }
}

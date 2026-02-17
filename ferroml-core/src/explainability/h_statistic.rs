//! Friedman's H-statistic for Feature Interaction Detection
//!
//! This module implements the H-statistic for quantifying feature interactions,
//! as described in Friedman & Popescu (2008).
//!
//! ## Overview
//!
//! The H-statistic measures the proportion of variance in the prediction function
//! that is due to interaction effects between features. It ranges from 0 to 1:
//! - H² = 0: No interaction (features are additive)
//! - H² = 1: Pure interaction effect
//!
//! ## References
//!
//! Friedman, J. H., & Popescu, B. E. (2008). Predictive learning via rule ensembles.
//! The Annals of Applied Statistics, 2(3), 916-954.

use super::partial_dependence::{generate_grid, GridMethod};
use crate::models::Model;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Result of H-statistic computation for a pair of features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HStatisticResult {
    /// H² statistic (squared, ranges 0-1)
    pub h_squared: f64,
    /// H statistic (square root of H², ranges 0-1)
    pub h_statistic: f64,
    /// First feature index
    pub feature_idx_1: usize,
    /// Second feature index
    pub feature_idx_2: usize,
    /// First feature name (if available)
    pub feature_name_1: Option<String>,
    /// Second feature name (if available)
    pub feature_name_2: Option<String>,
    /// Bootstrap confidence interval for H² (if computed)
    pub ci: Option<(f64, f64)>,
    /// Bootstrap standard error (if computed)
    pub std_error: Option<f64>,
    /// P-value from permutation test (if computed)
    pub p_value: Option<f64>,
    /// Number of samples used
    pub n_samples: usize,
    /// Number of grid points used
    pub n_grid_points: usize,
}

impl HStatisticResult {
    /// Interpretation of the H-statistic value
    #[must_use]
    pub fn interpretation(&self) -> &'static str {
        if self.h_squared < 0.01 {
            "No interaction"
        } else if self.h_squared < 0.05 {
            "Weak interaction"
        } else if self.h_squared < 0.15 {
            "Moderate interaction"
        } else if self.h_squared < 0.30 {
            "Strong interaction"
        } else {
            "Very strong interaction"
        }
    }

    /// Check if the interaction is statistically significant
    ///
    /// Returns `None` if p-value was not computed.
    #[must_use]
    pub fn is_significant(&self, alpha: f64) -> Option<bool> {
        self.p_value.map(|p| p < alpha)
    }

    /// Create a summary string
    #[must_use]
    pub fn summary(&self) -> String {
        let name1 = self
            .feature_name_1
            .clone()
            .unwrap_or_else(|| format!("feature_{}", self.feature_idx_1));
        let name2 = self
            .feature_name_2
            .clone()
            .unwrap_or_else(|| format!("feature_{}", self.feature_idx_2));

        let mut s = format!(
            "H-statistic({}, {}): H² = {:.4} ({})",
            name1,
            name2,
            self.h_squared,
            self.interpretation()
        );

        if let Some((ci_low, ci_high)) = self.ci {
            s.push_str(&format!(", 95% CI: [{:.4}, {:.4}]", ci_low, ci_high));
        }

        if let Some(p) = self.p_value {
            s.push_str(&format!(", p = {:.4}", p));
        }

        s
    }
}

impl std::fmt::Display for HStatisticResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Result of overall H-statistic for a single feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HStatisticOverallResult {
    /// H² statistic for the feature's overall interaction strength
    pub h_squared: f64,
    /// H statistic (square root of H²)
    pub h_statistic: f64,
    /// Feature index
    pub feature_idx: usize,
    /// Feature name (if available)
    pub feature_name: Option<String>,
    /// Bootstrap confidence interval for H² (if computed)
    pub ci: Option<(f64, f64)>,
    /// Number of samples used
    pub n_samples: usize,
}

impl HStatisticOverallResult {
    /// Interpretation of the H-statistic value
    #[must_use]
    pub fn interpretation(&self) -> &'static str {
        if self.h_squared < 0.01 {
            "No interactions"
        } else if self.h_squared < 0.05 {
            "Weak interactions"
        } else if self.h_squared < 0.15 {
            "Moderate interactions"
        } else if self.h_squared < 0.30 {
            "Strong interactions"
        } else {
            "Very strong interactions"
        }
    }

    /// Create a summary string
    #[must_use]
    pub fn summary(&self) -> String {
        let name = self
            .feature_name
            .clone()
            .unwrap_or_else(|| format!("feature_{}", self.feature_idx));

        let mut s = format!(
            "Overall H-statistic({}): H² = {:.4} ({})",
            name,
            self.h_squared,
            self.interpretation()
        );

        if let Some((ci_low, ci_high)) = self.ci {
            s.push_str(&format!(", 95% CI: [{:.4}, {:.4}]", ci_low, ci_high));
        }

        s
    }
}

impl std::fmt::Display for HStatisticOverallResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Pairwise H-statistic matrix result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HStatisticMatrix {
    /// Pairwise H² values (n_features x n_features)
    pub h_squared_matrix: Array2<f64>,
    /// Feature indices included
    pub feature_indices: Vec<usize>,
    /// Feature names (if available)
    pub feature_names: Option<Vec<String>>,
    /// Number of samples used
    pub n_samples: usize,
}

impl HStatisticMatrix {
    /// Get the H² value for a pair of features
    #[must_use]
    pub fn get(&self, feature_idx_1: usize, feature_idx_2: usize) -> Option<f64> {
        let i = self
            .feature_indices
            .iter()
            .position(|&x| x == feature_idx_1)?;
        let j = self
            .feature_indices
            .iter()
            .position(|&x| x == feature_idx_2)?;
        Some(self.h_squared_matrix[[i, j]])
    }

    /// Get top-k feature pairs by interaction strength
    #[must_use]
    pub fn top_k(&self, k: usize) -> Vec<(usize, usize, f64)> {
        let n = self.feature_indices.len();
        let mut pairs: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                pairs.push((
                    self.feature_indices[i],
                    self.feature_indices[j],
                    self.h_squared_matrix[[i, j]],
                ));
            }
        }

        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(k);
        pairs
    }

    /// Create a summary string
    #[must_use]
    pub fn summary(&self) -> String {
        let n = self.feature_indices.len();
        let top = self.top_k(5);

        let mut s = format!(
            "H-statistic Matrix ({} features, {} pairs)\n",
            n,
            n * (n - 1) / 2
        );
        s.push_str("Top interactions:\n");

        for (i, j, h2) in top {
            let name_i = self
                .feature_names
                .as_ref()
                .and_then(|names| names.iter().find(|n| n.contains(&format!("{}", i))))
                .cloned()
                .unwrap_or_else(|| format!("feature_{}", i));
            let name_j = self
                .feature_names
                .as_ref()
                .and_then(|names| names.iter().find(|n| n.contains(&format!("{}", j))))
                .cloned()
                .unwrap_or_else(|| format!("feature_{}", j));

            s.push_str(&format!("  {} x {}: H² = {:.4}\n", name_i, name_j, h2));
        }

        s
    }
}

impl std::fmt::Display for HStatisticMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Configuration for H-statistic computation
#[derive(Debug, Clone)]
pub struct HStatisticConfig {
    /// Number of grid points for PDP computation
    pub n_grid_points: usize,
    /// Grid method for generating evaluation points
    pub grid_method: GridMethod,
    /// Number of bootstrap samples for CI (0 to disable)
    pub n_bootstrap: usize,
    /// Number of permutations for p-value (0 to disable)
    pub n_permutations: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Confidence level for CI
    pub confidence_level: f64,
    /// Maximum number of samples to use (for large datasets)
    pub max_samples: Option<usize>,
}

impl Default for HStatisticConfig {
    fn default() -> Self {
        Self {
            n_grid_points: 20,
            grid_method: GridMethod::Percentile,
            n_bootstrap: 0,
            n_permutations: 0,
            random_state: None,
            confidence_level: 0.95,
            max_samples: None,
        }
    }
}

impl HStatisticConfig {
    /// Create a new configuration with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of grid points
    #[must_use]
    pub fn with_grid_points(mut self, n: usize) -> Self {
        self.n_grid_points = n;
        self
    }

    /// Set the grid method
    #[must_use]
    pub fn with_grid_method(mut self, method: GridMethod) -> Self {
        self.grid_method = method;
        self
    }

    /// Enable bootstrap confidence intervals
    #[must_use]
    pub fn with_bootstrap(mut self, n_bootstrap: usize) -> Self {
        self.n_bootstrap = n_bootstrap;
        self
    }

    /// Enable permutation-based p-value
    #[must_use]
    pub fn with_permutation_test(mut self, n_permutations: usize) -> Self {
        self.n_permutations = n_permutations;
        self
    }

    /// Set random state for reproducibility
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set confidence level
    #[must_use]
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Set maximum number of samples
    #[must_use]
    pub fn with_max_samples(mut self, n: usize) -> Self {
        self.max_samples = Some(n);
        self
    }
}

/// Compute Friedman's H-statistic for interaction between two features
///
/// The H-statistic measures the strength of interaction between two features,
/// based on how much the 2D partial dependence deviates from the sum of
/// the individual 1D partial dependences.
///
/// # Algorithm
///
/// H²_{jk} = Σᵢ[PD_{jk}(xᵢⱼ, xᵢₖ) - PD_j(xᵢⱼ) - PD_k(xᵢₖ)]² / Σᵢ[PD_{jk}(xᵢⱼ, xᵢₖ)]²
///
/// # Arguments
///
/// * `model` - Fitted model implementing the `Model` trait
/// * `x` - Feature matrix (n_samples, n_features)
/// * `feature_idx_1` - Index of the first feature
/// * `feature_idx_2` - Index of the second feature
/// * `config` - Configuration for H-statistic computation
///
/// # Returns
///
/// `HStatisticResult` containing the H² value and optional CI/p-value.
///
/// # Example
///
/// ```
/// # fn main() -> ferroml_core::Result<()> {
/// use ferroml_core::explainability::{h_statistic, HStatisticConfig};
/// use ferroml_core::models::RandomForestRegressor;
/// # use ferroml_core::models::Model;
/// # use ndarray::{Array1, Array2};
/// # let x_train = Array2::from_shape_vec((20, 3), (0..60).map(|i| i as f64 / 60.0).collect()).unwrap();
/// # let y_train = Array1::from_vec((0..20).map(|i| i as f64).collect());
/// # let x_test = x_train.clone();
///
/// let mut model = RandomForestRegressor::new();
/// model.fit(&x_train, &y_train)?;
///
/// // Basic H-statistic
/// let result = h_statistic(&model, &x_test, 0, 1, HStatisticConfig::default())?;
/// println!("H² = {:.4}", result.h_squared);
///
/// // With bootstrap CI and permutation test
/// let config = HStatisticConfig::new()
///     .with_bootstrap(1000)
///     .with_permutation_test(500)
///     .with_random_state(42);
/// let result = h_statistic(&model, &x_test, 0, 1, config)?;
/// println!("{}", result);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - The model is not fitted
/// - Feature indices are out of bounds or equal
/// - Input data is empty
/// - `n_grid_points` is less than 2
///
/// # References
///
/// Friedman, J. H., & Popescu, B. E. (2008). Predictive learning via rule ensembles.
/// The Annals of Applied Statistics, 2(3), 916-954.
pub fn h_statistic<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    config: HStatisticConfig,
) -> Result<HStatisticResult>
where
    M: Model,
{
    // Validate inputs
    if !model.is_fitted() {
        return Err(FerroError::not_fitted("h_statistic"));
    }
    if x.is_empty() {
        return Err(FerroError::invalid_input("Empty input data"));
    }
    if feature_idx_1 >= x.ncols() {
        return Err(FerroError::invalid_input(format!(
            "feature_idx_1 {} is out of bounds (n_features = {})",
            feature_idx_1,
            x.ncols()
        )));
    }
    if feature_idx_2 >= x.ncols() {
        return Err(FerroError::invalid_input(format!(
            "feature_idx_2 {} is out of bounds (n_features = {})",
            feature_idx_2,
            x.ncols()
        )));
    }
    if feature_idx_1 == feature_idx_2 {
        return Err(FerroError::invalid_input(
            "feature_idx_1 and feature_idx_2 must be different",
        ));
    }
    if config.n_grid_points < 2 {
        return Err(FerroError::invalid_input(
            "n_grid_points must be at least 2",
        ));
    }

    // Subsample if needed
    let x_used = if let Some(max_samples) = config.max_samples {
        if x.nrows() > max_samples {
            let mut rng = match config.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_os_rng(),
            };
            let indices: Vec<usize> = (0..x.nrows())
                .collect::<Vec<_>>()
                .into_iter()
                .choose_multiple(&mut rng, max_samples);
            let mut x_sub = Array2::zeros((max_samples, x.ncols()));
            for (i, &idx) in indices.iter().enumerate() {
                x_sub.row_mut(i).assign(&x.row(idx));
            }
            x_sub
        } else {
            x.to_owned()
        }
    } else {
        x.to_owned()
    };

    let n_samples = x_used.nrows();

    // Compute the base H² statistic
    let h_squared = compute_h_squared(
        model,
        &x_used,
        feature_idx_1,
        feature_idx_2,
        config.n_grid_points,
        config.grid_method,
    )?;

    // Bootstrap CI if requested
    let (ci, std_error) = if config.n_bootstrap > 0 {
        let bootstrap_results = bootstrap_h_statistic(
            model,
            &x_used,
            feature_idx_1,
            feature_idx_2,
            config.n_grid_points,
            config.grid_method,
            config.n_bootstrap,
            config.random_state,
        )?;

        let alpha = 1.0 - config.confidence_level;
        let lower_idx = ((alpha / 2.0) * config.n_bootstrap as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * config.n_bootstrap as f64) as usize;

        let mean: f64 = bootstrap_results.iter().sum::<f64>() / config.n_bootstrap as f64;
        let variance = bootstrap_results
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (config.n_bootstrap - 1) as f64;

        (
            Some((
                bootstrap_results[lower_idx],
                bootstrap_results[upper_idx.min(config.n_bootstrap - 1)],
            )),
            Some(variance.sqrt()),
        )
    } else {
        (None, None)
    };

    // Permutation p-value if requested
    let p_value = if config.n_permutations > 0 {
        Some(permutation_test_h_statistic(
            model,
            &x_used,
            feature_idx_1,
            feature_idx_2,
            config.n_grid_points,
            config.grid_method,
            h_squared,
            config.n_permutations,
            config.random_state,
        )?)
    } else {
        None
    };

    // Get feature names
    let feature_name_1 = model
        .feature_names()
        .and_then(|names| names.get(feature_idx_1).cloned());
    let feature_name_2 = model
        .feature_names()
        .and_then(|names| names.get(feature_idx_2).cloned());

    Ok(HStatisticResult {
        h_squared,
        h_statistic: h_squared.sqrt(),
        feature_idx_1,
        feature_idx_2,
        feature_name_1,
        feature_name_2,
        ci,
        std_error,
        p_value,
        n_samples,
        n_grid_points: config.n_grid_points,
    })
}

/// Compute H-statistic with parallel execution using rayon
///
/// Same as `h_statistic` but parallelizes the grid point computations.
///
/// # Errors
///
/// Returns an error if:
/// - The model is not fitted
/// - Feature indices are out of bounds or equal
/// - Input data is empty
pub fn h_statistic_parallel<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    config: HStatisticConfig,
) -> Result<HStatisticResult>
where
    M: Model + Sync,
{
    // Validate inputs
    if !model.is_fitted() {
        return Err(FerroError::not_fitted("h_statistic_parallel"));
    }
    if x.is_empty() {
        return Err(FerroError::invalid_input("Empty input data"));
    }
    if feature_idx_1 >= x.ncols() {
        return Err(FerroError::invalid_input(format!(
            "feature_idx_1 {} is out of bounds (n_features = {})",
            feature_idx_1,
            x.ncols()
        )));
    }
    if feature_idx_2 >= x.ncols() {
        return Err(FerroError::invalid_input(format!(
            "feature_idx_2 {} is out of bounds (n_features = {})",
            feature_idx_2,
            x.ncols()
        )));
    }
    if feature_idx_1 == feature_idx_2 {
        return Err(FerroError::invalid_input(
            "feature_idx_1 and feature_idx_2 must be different",
        ));
    }
    if config.n_grid_points < 2 {
        return Err(FerroError::invalid_input(
            "n_grid_points must be at least 2",
        ));
    }

    // Subsample if needed
    let x_used = if let Some(max_samples) = config.max_samples {
        if x.nrows() > max_samples {
            let mut rng = match config.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_os_rng(),
            };
            let indices: Vec<usize> = (0..x.nrows())
                .collect::<Vec<_>>()
                .into_iter()
                .choose_multiple(&mut rng, max_samples);
            let mut x_sub = Array2::zeros((max_samples, x.ncols()));
            for (i, &idx) in indices.iter().enumerate() {
                x_sub.row_mut(i).assign(&x.row(idx));
            }
            x_sub
        } else {
            x.to_owned()
        }
    } else {
        x.to_owned()
    };

    let n_samples = x_used.nrows();

    // Compute the base H² statistic in parallel
    let h_squared = compute_h_squared_parallel(
        model,
        &x_used,
        feature_idx_1,
        feature_idx_2,
        config.n_grid_points,
        config.grid_method,
    )?;

    // Bootstrap CI if requested (parallelized)
    let (ci, std_error) = if config.n_bootstrap > 0 {
        let bootstrap_results = bootstrap_h_statistic_parallel(
            model,
            &x_used,
            feature_idx_1,
            feature_idx_2,
            config.n_grid_points,
            config.grid_method,
            config.n_bootstrap,
            config.random_state,
        )?;

        let alpha = 1.0 - config.confidence_level;
        let lower_idx = ((alpha / 2.0) * config.n_bootstrap as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * config.n_bootstrap as f64) as usize;

        let mean: f64 = bootstrap_results.iter().sum::<f64>() / config.n_bootstrap as f64;
        let variance = bootstrap_results
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (config.n_bootstrap - 1) as f64;

        (
            Some((
                bootstrap_results[lower_idx],
                bootstrap_results[upper_idx.min(config.n_bootstrap - 1)],
            )),
            Some(variance.sqrt()),
        )
    } else {
        (None, None)
    };

    // Permutation p-value if requested (parallelized)
    let p_value = if config.n_permutations > 0 {
        Some(permutation_test_h_statistic_parallel(
            model,
            &x_used,
            feature_idx_1,
            feature_idx_2,
            config.n_grid_points,
            config.grid_method,
            h_squared,
            config.n_permutations,
            config.random_state,
        )?)
    } else {
        None
    };

    // Get feature names
    let feature_name_1 = model
        .feature_names()
        .and_then(|names| names.get(feature_idx_1).cloned());
    let feature_name_2 = model
        .feature_names()
        .and_then(|names| names.get(feature_idx_2).cloned());

    Ok(HStatisticResult {
        h_squared,
        h_statistic: h_squared.sqrt(),
        feature_idx_1,
        feature_idx_2,
        feature_name_1,
        feature_name_2,
        ci,
        std_error,
        p_value,
        n_samples,
        n_grid_points: config.n_grid_points,
    })
}

/// Compute pairwise H-statistic matrix for multiple features
///
/// # Arguments
///
/// * `model` - Fitted model implementing the `Model` trait
/// * `x` - Feature matrix (n_samples, n_features)
/// * `feature_indices` - Indices of features to compute H-statistics for (None = all)
/// * `config` - Configuration for H-statistic computation
///
/// # Returns
///
/// `HStatisticMatrix` containing pairwise H² values.
///
/// # Errors
///
/// Returns an error if the model is not fitted or input is invalid.
pub fn h_statistic_matrix<M>(
    model: &M,
    x: &Array2<f64>,
    feature_indices: Option<&[usize]>,
    config: HStatisticConfig,
) -> Result<HStatisticMatrix>
where
    M: Model,
{
    if !model.is_fitted() {
        return Err(FerroError::not_fitted("h_statistic_matrix"));
    }
    if x.is_empty() {
        return Err(FerroError::invalid_input("Empty input data"));
    }

    let indices: Vec<usize> = feature_indices
        .map(|i| i.to_vec())
        .unwrap_or_else(|| (0..x.ncols()).collect());

    let n = indices.len();
    let mut h_squared_matrix = Array2::zeros((n, n));

    // Subsample if needed
    let x_used = if let Some(max_samples) = config.max_samples {
        if x.nrows() > max_samples {
            let mut rng = match config.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_os_rng(),
            };
            let sample_indices: Vec<usize> = (0..x.nrows())
                .collect::<Vec<_>>()
                .into_iter()
                .choose_multiple(&mut rng, max_samples);
            let mut x_sub = Array2::zeros((max_samples, x.ncols()));
            for (i, &idx) in sample_indices.iter().enumerate() {
                x_sub.row_mut(i).assign(&x.row(idx));
            }
            x_sub
        } else {
            x.to_owned()
        }
    } else {
        x.to_owned()
    };

    for i in 0..n {
        for j in (i + 1)..n {
            let h2 = compute_h_squared(
                model,
                &x_used,
                indices[i],
                indices[j],
                config.n_grid_points,
                config.grid_method,
            )?;
            h_squared_matrix[[i, j]] = h2;
            h_squared_matrix[[j, i]] = h2; // Symmetric
        }
    }

    let feature_names = model.feature_names().map(|names| {
        indices
            .iter()
            .filter_map(|&i| names.get(i).cloned())
            .collect()
    });

    Ok(HStatisticMatrix {
        h_squared_matrix,
        feature_indices: indices,
        feature_names,
        n_samples: x_used.nrows(),
    })
}

/// Compute pairwise H-statistic matrix with parallel execution
///
/// # Errors
///
/// Returns an error if the model is not fitted or input is invalid.
pub fn h_statistic_matrix_parallel<M>(
    model: &M,
    x: &Array2<f64>,
    feature_indices: Option<&[usize]>,
    config: HStatisticConfig,
) -> Result<HStatisticMatrix>
where
    M: Model + Sync,
{
    if !model.is_fitted() {
        return Err(FerroError::not_fitted("h_statistic_matrix_parallel"));
    }
    if x.is_empty() {
        return Err(FerroError::invalid_input("Empty input data"));
    }

    let indices: Vec<usize> = feature_indices
        .map(|i| i.to_vec())
        .unwrap_or_else(|| (0..x.ncols()).collect());

    let n = indices.len();

    // Subsample if needed
    let x_used = if let Some(max_samples) = config.max_samples {
        if x.nrows() > max_samples {
            let mut rng = match config.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_os_rng(),
            };
            let sample_indices: Vec<usize> = (0..x.nrows())
                .collect::<Vec<_>>()
                .into_iter()
                .choose_multiple(&mut rng, max_samples);
            let mut x_sub = Array2::zeros((max_samples, x.ncols()));
            for (i, &idx) in sample_indices.iter().enumerate() {
                x_sub.row_mut(i).assign(&x.row(idx));
            }
            x_sub
        } else {
            x.to_owned()
        }
    } else {
        x.to_owned()
    };

    // Generate all pairs
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    // Compute in parallel
    let results: Vec<(usize, usize, f64)> = pairs
        .into_par_iter()
        .map(|(i, j)| {
            let h2 = compute_h_squared_parallel(
                model,
                &x_used,
                indices[i],
                indices[j],
                config.n_grid_points,
                config.grid_method,
            )
            .unwrap_or(0.0);
            (i, j, h2)
        })
        .collect();

    // Assemble matrix
    let mut h_squared_matrix = Array2::zeros((n, n));
    for (i, j, h2) in results {
        h_squared_matrix[[i, j]] = h2;
        h_squared_matrix[[j, i]] = h2;
    }

    let feature_names = model.feature_names().map(|names| {
        indices
            .iter()
            .filter_map(|&i| names.get(i).cloned())
            .collect()
    });

    Ok(HStatisticMatrix {
        h_squared_matrix,
        feature_indices: indices,
        feature_names,
        n_samples: x_used.nrows(),
    })
}

/// Compute overall H-statistic for a single feature
///
/// Measures the total interaction strength of a feature with all other features.
/// This is useful for identifying features that participate in many interactions.
///
/// # Arguments
///
/// * `model` - Fitted model implementing the `Model` trait
/// * `x` - Feature matrix (n_samples, n_features)
/// * `feature_idx` - Index of the feature to analyze
/// * `config` - Configuration for H-statistic computation
///
/// # Returns
///
/// `HStatisticOverallResult` containing the overall H² value.
///
/// # Errors
///
/// Returns an error if the model is not fitted or input is invalid.
pub fn h_statistic_overall<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx: usize,
    config: HStatisticConfig,
) -> Result<HStatisticOverallResult>
where
    M: Model,
{
    if !model.is_fitted() {
        return Err(FerroError::not_fitted("h_statistic_overall"));
    }
    if x.is_empty() {
        return Err(FerroError::invalid_input("Empty input data"));
    }
    if feature_idx >= x.ncols() {
        return Err(FerroError::invalid_input(format!(
            "feature_idx {} is out of bounds (n_features = {})",
            feature_idx,
            x.ncols()
        )));
    }

    let n_features = x.ncols();

    // Subsample if needed
    let x_used = if let Some(max_samples) = config.max_samples {
        if x.nrows() > max_samples {
            let mut rng = match config.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_os_rng(),
            };
            let indices: Vec<usize> = (0..x.nrows())
                .collect::<Vec<_>>()
                .into_iter()
                .choose_multiple(&mut rng, max_samples);
            let mut x_sub = Array2::zeros((max_samples, x.ncols()));
            for (i, &idx) in indices.iter().enumerate() {
                x_sub.row_mut(i).assign(&x.row(idx));
            }
            x_sub
        } else {
            x.to_owned()
        }
    } else {
        x.to_owned()
    };

    let n_samples = x_used.nrows();

    // Compute H² with each other feature and aggregate
    let mut h_squared_sum = 0.0;
    let mut count = 0;

    for j in 0..n_features {
        if j != feature_idx {
            let h2 = compute_h_squared(
                model,
                &x_used,
                feature_idx,
                j,
                config.n_grid_points,
                config.grid_method,
            )?;
            h_squared_sum += h2;
            count += 1;
        }
    }

    let h_squared = if count > 0 {
        h_squared_sum / count as f64
    } else {
        0.0
    };

    let feature_name = model
        .feature_names()
        .and_then(|names| names.get(feature_idx).cloned());

    Ok(HStatisticOverallResult {
        h_squared,
        h_statistic: h_squared.sqrt(),
        feature_idx,
        feature_name,
        ci: None, // Could add bootstrap CI here
        n_samples,
    })
}

// ============================================================================
// Internal helper functions
// ============================================================================

/// Core H² computation using PDP values
///
/// The H-statistic formula (Friedman & Popescu, 2008):
/// H²_{jk} = Σ \[PD_{jk}(xᵢ, yᵢ) - PD_j(xᵢ) - PD_k(yᵢ) + E\[f\]\]² / Σ \[PD_{jk}(xᵢ, yᵢ) - E\[f\]\]²
///
/// This measures the proportion of variance in the joint effect that is due to
/// interaction (not explained by the additive main effects).
fn compute_h_squared<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    n_grid_points: usize,
    grid_method: GridMethod,
) -> Result<f64>
where
    M: Model,
{
    let n_samples = x.nrows();

    // Generate grid values for both features
    let grid_1 = generate_grid(
        &x.column(feature_idx_1).to_owned(),
        n_grid_points,
        grid_method,
    );
    let grid_2 = generate_grid(
        &x.column(feature_idx_2).to_owned(),
        n_grid_points,
        grid_method,
    );

    // First, compute the baseline mean prediction (E[f])
    let baseline_preds = model.predict(x)?;
    let baseline_mean = baseline_preds.mean().unwrap_or(0.0);

    // Compute 1D PDPs
    let mut pdp_1 = Array1::<f64>::zeros(n_grid_points);
    let mut pdp_2 = Array1::<f64>::zeros(n_grid_points);

    for (i, &g1) in grid_1.iter().enumerate() {
        let mut x_mod = x.to_owned();
        for k in 0..n_samples {
            x_mod[[k, feature_idx_1]] = g1;
        }
        let preds = model.predict(&x_mod)?;
        pdp_1[i] = preds.mean().unwrap_or(0.0);
    }

    for (j, &g2) in grid_2.iter().enumerate() {
        let mut x_mod = x.to_owned();
        for k in 0..n_samples {
            x_mod[[k, feature_idx_2]] = g2;
        }
        let preds = model.predict(&x_mod)?;
        pdp_2[j] = preds.mean().unwrap_or(0.0);
    }

    // Compute 2D PDP values (joint effect)
    let mut pdp_joint = Array2::<f64>::zeros((n_grid_points, n_grid_points));
    for (i, &g1) in grid_1.iter().enumerate() {
        for (j, &g2) in grid_2.iter().enumerate() {
            let mut x_mod = x.to_owned();
            for k in 0..n_samples {
                x_mod[[k, feature_idx_1]] = g1;
                x_mod[[k, feature_idx_2]] = g2;
            }
            let preds = model.predict(&x_mod)?;
            pdp_joint[[i, j]] = preds.mean().unwrap_or(0.0);
        }
    }

    // Compute H² using the corrected formula:
    // H²_{jk} = Σ [PD_{jk} - PD_j - PD_k + E[f]]² / Σ [PD_{jk} - E[f]]²
    // This accounts for the double subtraction of the baseline when we subtract both marginals
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n_grid_points {
        for j in 0..n_grid_points {
            // Centered 2D PDP (deviation from baseline)
            let joint_centered = pdp_joint[[i, j]] - baseline_mean;

            // Centered 1D PDPs (deviation from baseline)
            let marginal_1_centered = pdp_1[i] - baseline_mean;
            let marginal_2_centered = pdp_2[j] - baseline_mean;

            // Interaction term: what's left after removing additive effects
            // If additive: PD_12 ≈ PD_1 + PD_2 - E[f], so interaction ≈ 0
            let interaction = joint_centered - marginal_1_centered - marginal_2_centered;

            numerator += interaction * interaction;
            denominator += joint_centered * joint_centered;
        }
    }

    // Avoid division by zero
    if denominator < 1e-10 {
        return Ok(0.0);
    }

    Ok((numerator / denominator).min(1.0).max(0.0))
}

/// Parallel version of H² computation
fn compute_h_squared_parallel<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    n_grid_points: usize,
    grid_method: GridMethod,
) -> Result<f64>
where
    M: Model + Sync,
{
    let n_samples = x.nrows();

    // Generate grid values for both features
    let grid_1 = generate_grid(
        &x.column(feature_idx_1).to_owned(),
        n_grid_points,
        grid_method,
    );
    let grid_2 = generate_grid(
        &x.column(feature_idx_2).to_owned(),
        n_grid_points,
        grid_method,
    );

    // Compute the baseline mean prediction (E[f])
    let baseline_preds = model.predict(x)?;
    let baseline_mean = baseline_preds.mean().unwrap_or(0.0);

    // Compute 1D PDPs in parallel
    let pdp_1: Vec<f64> = grid_1
        .iter()
        .cloned()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|g1| {
            let mut x_mod = x.to_owned();
            for k in 0..n_samples {
                x_mod[[k, feature_idx_1]] = g1;
            }
            model
                .predict(&x_mod)
                .map(|p| p.mean().unwrap_or(0.0))
                .unwrap_or(0.0)
        })
        .collect();

    let pdp_2: Vec<f64> = grid_2
        .iter()
        .cloned()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|g2| {
            let mut x_mod = x.to_owned();
            for k in 0..n_samples {
                x_mod[[k, feature_idx_2]] = g2;
            }
            model
                .predict(&x_mod)
                .map(|p| p.mean().unwrap_or(0.0))
                .unwrap_or(0.0)
        })
        .collect();

    // Generate all grid combinations
    let combinations: Vec<(usize, usize, f64, f64)> = grid_1
        .iter()
        .enumerate()
        .flat_map(|(i, &g1)| {
            grid_2
                .iter()
                .enumerate()
                .map(move |(j, &g2)| (i, j, g1, g2))
        })
        .collect();

    // Compute 2D PDP values in parallel (joint effect)
    let pdp_joint_values: Vec<(usize, usize, f64)> = combinations
        .into_par_iter()
        .map(|(i, j, g1, g2)| {
            let mut x_mod = x.to_owned();
            for k in 0..n_samples {
                x_mod[[k, feature_idx_1]] = g1;
                x_mod[[k, feature_idx_2]] = g2;
            }
            let pdp_val = model
                .predict(&x_mod)
                .map(|p| p.mean().unwrap_or(0.0))
                .unwrap_or(0.0);
            (i, j, pdp_val)
        })
        .collect();

    // Compute H² using centered PDPs
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, j, pdp_joint) in pdp_joint_values {
        // Centered values
        let joint_centered = pdp_joint - baseline_mean;
        let marginal_1_centered = pdp_1[i] - baseline_mean;
        let marginal_2_centered = pdp_2[j] - baseline_mean;

        // Interaction term
        let interaction = joint_centered - marginal_1_centered - marginal_2_centered;

        numerator += interaction * interaction;
        denominator += joint_centered * joint_centered;
    }

    if denominator < 1e-10 {
        return Ok(0.0);
    }

    Ok((numerator / denominator).min(1.0).max(0.0))
}

/// Bootstrap H-statistic for confidence intervals
fn bootstrap_h_statistic<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    n_grid_points: usize,
    grid_method: GridMethod,
    n_bootstrap: usize,
    random_state: Option<u64>,
) -> Result<Vec<f64>>
where
    M: Model,
{
    let n_samples = x.nrows();
    let mut rng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };

    let mut results = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Bootstrap resample
        let mut x_boot = Array2::zeros((n_samples, x.ncols()));
        for i in 0..n_samples {
            let idx = rng.random_range(0..n_samples);
            x_boot.row_mut(i).assign(&x.row(idx));
        }

        let h2 = compute_h_squared(
            model,
            &x_boot,
            feature_idx_1,
            feature_idx_2,
            n_grid_points,
            grid_method,
        )?;
        results.push(h2);
    }

    results.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(results)
}

/// Parallel bootstrap H-statistic
fn bootstrap_h_statistic_parallel<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    n_grid_points: usize,
    grid_method: GridMethod,
    n_bootstrap: usize,
    random_state: Option<u64>,
) -> Result<Vec<f64>>
where
    M: Model + Sync,
{
    let n_samples = x.nrows();
    let base_seed = random_state.unwrap_or(0);

    let mut results: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .map(|b| {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(b as u64));

            // Bootstrap resample
            let mut x_boot = Array2::zeros((n_samples, x.ncols()));
            for i in 0..n_samples {
                let idx = rng.random_range(0..n_samples);
                x_boot.row_mut(i).assign(&x.row(idx));
            }

            compute_h_squared_parallel(
                model,
                &x_boot,
                feature_idx_1,
                feature_idx_2,
                n_grid_points,
                grid_method,
            )
            .unwrap_or(0.0)
        })
        .collect();

    results.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(results)
}

/// Permutation test for H-statistic significance
fn permutation_test_h_statistic<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    n_grid_points: usize,
    grid_method: GridMethod,
    observed_h2: f64,
    n_permutations: usize,
    random_state: Option<u64>,
) -> Result<f64>
where
    M: Model,
{
    let n_samples = x.nrows();
    let mut rng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };

    let mut n_extreme = 0;

    for _ in 0..n_permutations {
        // Permute feature_idx_2 to break the interaction
        let mut x_perm = x.to_owned();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        for i in 0..n_samples {
            x_perm[[i, feature_idx_2]] = x[[indices[i], feature_idx_2]];
        }

        let h2_perm = compute_h_squared(
            model,
            &x_perm,
            feature_idx_1,
            feature_idx_2,
            n_grid_points,
            grid_method,
        )?;

        if h2_perm >= observed_h2 {
            n_extreme += 1;
        }
    }

    Ok((n_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0))
}

/// Parallel permutation test
fn permutation_test_h_statistic_parallel<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    n_grid_points: usize,
    grid_method: GridMethod,
    observed_h2: f64,
    n_permutations: usize,
    random_state: Option<u64>,
) -> Result<f64>
where
    M: Model + Sync,
{
    let n_samples = x.nrows();
    let base_seed = random_state.unwrap_or(0);

    let n_extreme: usize = (0..n_permutations)
        .into_par_iter()
        .map(|p| {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(p as u64 + 10000));

            // Permute feature_idx_2
            let mut x_perm = x.to_owned();
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            for i in 0..n_samples {
                x_perm[[i, feature_idx_2]] = x[[indices[i], feature_idx_2]];
            }

            let h2_perm = compute_h_squared_parallel(
                model,
                &x_perm,
                feature_idx_1,
                feature_idx_2,
                n_grid_points,
                grid_method,
            )
            .unwrap_or(0.0);

            if h2_perm >= observed_h2 {
                1
            } else {
                0
            }
        })
        .sum();

    Ok((n_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::linear::LinearRegression;
    use ndarray::Axis;

    #[test]
    fn test_h_statistic_no_interaction() {
        // Create additive model: y = 2*x0 + 3*x1 + 1 (no interaction)
        let n_samples = 100;
        let x = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                i as f64 * 0.1
            } else {
                ((i * 7 + 11) % 19) as f64 * 0.1
            }
        });
        let y: Array1<f64> = x
            .axis_iter(Axis(0))
            .map(|row| 2.0 * row[0] + 3.0 * row[1] + 1.0)
            .collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = h_statistic(&model, &x, 0, 1, HStatisticConfig::default()).unwrap();

        // For additive model, H² should be very small
        assert!(
            result.h_squared < 0.05,
            "H² = {} should be small for additive model",
            result.h_squared
        );
        assert_eq!(result.interpretation(), "No interaction");
    }

    #[test]
    fn test_h_statistic_with_interaction() {
        // Create model with interaction: y = x0 * x1
        let n_samples = 100;
        let x = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                (i as f64 * 0.1).sin() + 1.0
            } else {
                (i as f64 * 0.05).cos() + 1.0
            }
        });
        let y: Array1<f64> = x.axis_iter(Axis(0)).map(|row| row[0] * row[1]).collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = h_statistic(&model, &x, 0, 1, HStatisticConfig::default()).unwrap();

        // For multiplicative interaction, we expect some H² (though linear model
        // can't fully capture it, the data itself has interaction structure)
        assert!(result.h_squared >= 0.0 && result.h_squared <= 1.0);
    }

    #[test]
    fn test_h_statistic_parallel() {
        let n_samples = 50;
        let x = Array2::from_shape_fn((n_samples, 3), |(i, j)| match j {
            0 => i as f64 * 0.1,
            1 => ((i * 7 + 11) % 19) as f64 * 0.1,
            _ => ((i * 11 + 3) % 23) as f64 * 0.1,
        });
        let y: Array1<f64> = x
            .axis_iter(Axis(0))
            .map(|row| row[0] + row[1] + 1.0)
            .collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = HStatisticConfig::default();
        let result_seq = h_statistic(&model, &x, 0, 1, config.clone()).unwrap();
        let result_par = h_statistic_parallel(&model, &x, 0, 1, config).unwrap();

        // Results should be identical
        assert!(
            (result_seq.h_squared - result_par.h_squared).abs() < 1e-6,
            "Sequential H² = {}, Parallel H² = {}",
            result_seq.h_squared,
            result_par.h_squared
        );
    }

    #[test]
    fn test_h_statistic_with_bootstrap() {
        let n_samples = 50;
        let x = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                i as f64 * 0.1
            } else {
                ((i * 7 + 11) % 19) as f64 * 0.1
            }
        });
        let y: Array1<f64> = x.axis_iter(Axis(0)).map(|row| row[0] + row[1]).collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = HStatisticConfig::new()
            .with_bootstrap(100)
            .with_random_state(42);
        let result = h_statistic(&model, &x, 0, 1, config).unwrap();

        // Should have CI and std_error
        assert!(result.ci.is_some());
        assert!(result.std_error.is_some());

        let (ci_low, ci_high) = result.ci.unwrap();
        assert!(ci_low <= result.h_squared);
        assert!(ci_high >= result.h_squared);
    }

    #[test]
    fn test_h_statistic_with_permutation_test() {
        let n_samples = 50;
        let x = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                i as f64 * 0.1
            } else {
                ((i * 7 + 11) % 19) as f64 * 0.1
            }
        });
        let y: Array1<f64> = x.axis_iter(Axis(0)).map(|row| row[0] + row[1]).collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = HStatisticConfig::new()
            .with_permutation_test(50)
            .with_random_state(42);
        let result = h_statistic(&model, &x, 0, 1, config).unwrap();

        // Should have p_value
        assert!(result.p_value.is_some());
        let p = result.p_value.unwrap();
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_h_statistic_matrix() {
        let n_samples = 50;
        let x = Array2::from_shape_fn((n_samples, 3), |(i, j)| match j {
            0 => i as f64 * 0.1,
            1 => ((i * 7 + 11) % 19) as f64 * 0.1,
            _ => ((i * 11 + 3) % 23) as f64 * 0.1,
        });
        let y: Array1<f64> = x.axis_iter(Axis(0)).map(|row| row[0] + row[1]).collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = h_statistic_matrix(&model, &x, None, HStatisticConfig::default()).unwrap();

        assert_eq!(result.h_squared_matrix.shape(), &[3, 3]);
        assert_eq!(result.feature_indices, vec![0, 1, 2]);

        // Matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (result.h_squared_matrix[[i, j]] - result.h_squared_matrix[[j, i]]).abs()
                        < 1e-10
                );
            }
            // Diagonal should be 0
            assert!(result.h_squared_matrix[[i, i]] < 1e-10);
        }
    }

    #[test]
    fn test_h_statistic_matrix_parallel() {
        let n_samples = 30;
        let x = Array2::from_shape_fn((n_samples, 3), |(i, j)| match j {
            0 => i as f64 * 0.1,
            1 => ((i * 7 + 11) % 19) as f64 * 0.1,
            _ => ((i * 11 + 3) % 23) as f64 * 0.1,
        });
        let y: Array1<f64> = x.axis_iter(Axis(0)).map(|row| row[0] + row[1]).collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = HStatisticConfig::default();
        let result_seq = h_statistic_matrix(&model, &x, None, config.clone()).unwrap();
        let result_par = h_statistic_matrix_parallel(&model, &x, None, config).unwrap();

        // Results should be identical
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (result_seq.h_squared_matrix[[i, j]] - result_par.h_squared_matrix[[i, j]])
                        .abs()
                        < 1e-6
                );
            }
        }
    }

    #[test]
    fn test_h_statistic_overall() {
        let n_samples = 50;
        let x = Array2::from_shape_fn((n_samples, 3), |(i, j)| match j {
            0 => i as f64 * 0.1,
            1 => ((i * 7 + 11) % 19) as f64 * 0.1,
            _ => ((i * 11 + 3) % 23) as f64 * 0.1,
        });
        let y: Array1<f64> = x.axis_iter(Axis(0)).map(|row| row[0] + row[1]).collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = h_statistic_overall(&model, &x, 0, HStatisticConfig::default()).unwrap();

        assert!(result.h_squared >= 0.0 && result.h_squared <= 1.0);
        assert_eq!(result.feature_idx, 0);
    }

    #[test]
    fn test_h_statistic_top_k() {
        let matrix = HStatisticMatrix {
            h_squared_matrix: Array2::from_shape_vec(
                (3, 3),
                vec![0.0, 0.5, 0.1, 0.5, 0.0, 0.3, 0.1, 0.3, 0.0],
            )
            .unwrap(),
            feature_indices: vec![0, 1, 2],
            feature_names: None,
            n_samples: 100,
        };

        let top = matrix.top_k(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0], (0, 1, 0.5)); // Highest interaction
        assert_eq!(top[1], (1, 2, 0.3)); // Second highest
    }

    #[test]
    fn test_h_statistic_result_interpretation() {
        let mut result = HStatisticResult {
            h_squared: 0.005,
            h_statistic: 0.07,
            feature_idx_1: 0,
            feature_idx_2: 1,
            feature_name_1: None,
            feature_name_2: None,
            ci: None,
            std_error: None,
            p_value: None,
            n_samples: 100,
            n_grid_points: 20,
        };

        assert_eq!(result.interpretation(), "No interaction");

        result.h_squared = 0.02;
        assert_eq!(result.interpretation(), "Weak interaction");

        result.h_squared = 0.10;
        assert_eq!(result.interpretation(), "Moderate interaction");

        result.h_squared = 0.20;
        assert_eq!(result.interpretation(), "Strong interaction");

        result.h_squared = 0.40;
        assert_eq!(result.interpretation(), "Very strong interaction");
    }

    #[test]
    fn test_h_statistic_is_significant() {
        let result = HStatisticResult {
            h_squared: 0.1,
            h_statistic: 0.316,
            feature_idx_1: 0,
            feature_idx_2: 1,
            feature_name_1: None,
            feature_name_2: None,
            ci: None,
            std_error: None,
            p_value: Some(0.03),
            n_samples: 100,
            n_grid_points: 20,
        };

        assert_eq!(result.is_significant(0.05), Some(true));
        assert_eq!(result.is_significant(0.01), Some(false));

        let result_no_p = HStatisticResult {
            p_value: None,
            ..result
        };
        assert_eq!(result_no_p.is_significant(0.05), None);
    }

    #[test]
    fn test_error_not_fitted() {
        let model = LinearRegression::new();
        let x = Array2::zeros((10, 2));

        let result = h_statistic(&model, &x, 0, 1, HStatisticConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_error_invalid_feature_idx() {
        let n_samples = 20;
        let x = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                i as f64 * 0.1
            } else {
                ((i * 7 + 11) % 19) as f64 * 0.1
            }
        });
        let y: Array1<f64> = x.axis_iter(Axis(0)).map(|row| row[0]).collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = h_statistic(&model, &x, 0, 5, HStatisticConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_error_same_feature() {
        let n_samples = 20;
        let x = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                i as f64 * 0.1
            } else {
                ((i * 7 + 11) % 19) as f64 * 0.1
            }
        });
        let y: Array1<f64> = x.axis_iter(Axis(0)).map(|row| row[0]).collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = h_statistic(&model, &x, 0, 0, HStatisticConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_h_statistic_config_builder() {
        let config = HStatisticConfig::new()
            .with_grid_points(30)
            .with_grid_method(GridMethod::Uniform)
            .with_bootstrap(500)
            .with_permutation_test(200)
            .with_random_state(123)
            .with_confidence_level(0.99)
            .with_max_samples(1000);

        assert_eq!(config.n_grid_points, 30);
        assert_eq!(config.grid_method, GridMethod::Uniform);
        assert_eq!(config.n_bootstrap, 500);
        assert_eq!(config.n_permutations, 200);
        assert_eq!(config.random_state, Some(123));
        assert!((config.confidence_level - 0.99).abs() < 1e-10);
        assert_eq!(config.max_samples, Some(1000));
    }

    #[test]
    fn test_h_statistic_with_max_samples() {
        let n_samples = 100;
        let x = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                i as f64 * 0.1
            } else {
                ((i * 7 + 11) % 19) as f64 * 0.1
            }
        });
        let y: Array1<f64> = x.axis_iter(Axis(0)).map(|row| row[0] + row[1]).collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = HStatisticConfig::new()
            .with_max_samples(50)
            .with_random_state(42);
        let result = h_statistic(&model, &x, 0, 1, config).unwrap();

        assert_eq!(result.n_samples, 50);
    }

    #[test]
    fn test_summary_display() {
        let result = HStatisticResult {
            h_squared: 0.10,
            h_statistic: 0.316,
            feature_idx_1: 0,
            feature_idx_2: 1,
            feature_name_1: Some("feature_a".to_string()),
            feature_name_2: Some("feature_b".to_string()),
            ci: Some((0.05, 0.15)),
            std_error: Some(0.025),
            p_value: Some(0.02),
            n_samples: 100,
            n_grid_points: 20,
        };

        let summary = result.summary();
        assert!(summary.contains("feature_a"));
        assert!(summary.contains("feature_b"));
        assert!(summary.contains("0.10"));
        assert!(summary.contains("Moderate interaction")); // H²=0.10 falls in [0.05, 0.15)
        assert!(summary.contains("CI"));
        assert!(summary.contains("p ="));
    }

    #[test]
    fn test_h_statistic_matrix_get() {
        let matrix = HStatisticMatrix {
            h_squared_matrix: Array2::from_shape_vec(
                (3, 3),
                vec![0.0, 0.5, 0.1, 0.5, 0.0, 0.3, 0.1, 0.3, 0.0],
            )
            .unwrap(),
            feature_indices: vec![0, 1, 2],
            feature_names: None,
            n_samples: 100,
        };

        assert_eq!(matrix.get(0, 1), Some(0.5));
        assert_eq!(matrix.get(1, 2), Some(0.3));
        assert_eq!(matrix.get(0, 2), Some(0.1));
        assert_eq!(matrix.get(5, 6), None); // Invalid indices
    }
}

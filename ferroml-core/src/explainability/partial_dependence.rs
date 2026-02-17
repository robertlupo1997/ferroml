//! Partial Dependence Plot (PDP) Implementation
//!
//! Model-agnostic visualization of marginal feature effects.
//! PDPs show the relationship between a feature and the predicted outcome,
//! marginalizing over all other features.

use crate::models::Model;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Result of partial dependence computation for a single feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PDPResult {
    /// Grid values for the feature
    pub grid_values: Array1<f64>,
    /// Average predictions at each grid point
    pub pdp_values: Array1<f64>,
    /// Standard deviation of predictions at each grid point (measure of heterogeneity)
    pub pdp_std: Array1<f64>,
    /// Individual Conditional Expectation (ICE) curves (n_samples x n_grid_points)
    /// Only populated if `return_ice` is true
    pub ice_curves: Option<Array2<f64>>,
    /// Feature index
    pub feature_idx: usize,
    /// Feature name (if available)
    pub feature_name: Option<String>,
    /// Number of samples used
    pub n_samples: usize,
    /// Grid method used
    pub grid_method: GridMethod,
}

impl PDPResult {
    /// Get the minimum PDP value
    #[must_use]
    pub fn min_effect(&self) -> f64 {
        self.pdp_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Get the maximum PDP value
    #[must_use]
    pub fn max_effect(&self) -> f64 {
        self.pdp_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get the range of the PDP effect
    #[must_use]
    pub fn effect_range(&self) -> f64 {
        self.max_effect() - self.min_effect()
    }

    /// Check if the feature has a monotonic increasing effect
    #[must_use]
    pub fn is_monotonic_increasing(&self) -> bool {
        self.pdp_values
            .as_slice()
            .unwrap_or(&[])
            .windows(2)
            .all(|w| w[1] >= w[0] - 1e-10)
    }

    /// Check if the feature has a monotonic decreasing effect
    #[must_use]
    pub fn is_monotonic_decreasing(&self) -> bool {
        self.pdp_values
            .as_slice()
            .unwrap_or(&[])
            .windows(2)
            .all(|w| w[1] <= w[0] + 1e-10)
    }

    /// Check if the feature has a monotonic effect (either direction)
    #[must_use]
    pub fn is_monotonic(&self) -> bool {
        self.is_monotonic_increasing() || self.is_monotonic_decreasing()
    }

    /// Get the grid point with maximum prediction
    #[must_use]
    pub fn argmax(&self) -> usize {
        self.pdp_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get the grid point with minimum prediction
    #[must_use]
    pub fn argmin(&self) -> usize {
        self.pdp_values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get the heterogeneity (average std across grid points)
    ///
    /// High heterogeneity suggests feature interactions may be present.
    #[must_use]
    pub fn heterogeneity(&self) -> f64 {
        self.pdp_std.mean().unwrap_or(0.0)
    }

    /// Create a summary string
    #[must_use]
    pub fn summary(&self) -> String {
        let default_name = format!("feature_{}", self.feature_idx);
        let name = self.feature_name.as_deref().unwrap_or(&default_name);

        let monotonicity = if self.is_monotonic_increasing() {
            "increasing"
        } else if self.is_monotonic_decreasing() {
            "decreasing"
        } else {
            "non-monotonic"
        };

        format!(
            "PDP for {}: effect range = {:.4}, trend = {}, heterogeneity = {:.4}",
            name,
            self.effect_range(),
            monotonicity,
            self.heterogeneity()
        )
    }
}

impl std::fmt::Display for PDPResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Result of 2D partial dependence computation (feature interaction)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PDP2DResult {
    /// Grid values for the first feature
    pub grid_values_1: Array1<f64>,
    /// Grid values for the second feature
    pub grid_values_2: Array1<f64>,
    /// Average predictions at each grid point (n_grid_1 x n_grid_2)
    pub pdp_values: Array2<f64>,
    /// First feature index
    pub feature_idx_1: usize,
    /// Second feature index
    pub feature_idx_2: usize,
    /// First feature name (if available)
    pub feature_name_1: Option<String>,
    /// Second feature name (if available)
    pub feature_name_2: Option<String>,
    /// Number of samples used
    pub n_samples: usize,
}

impl PDP2DResult {
    /// Get the minimum PDP value
    #[must_use]
    pub fn min_effect(&self) -> f64 {
        self.pdp_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Get the maximum PDP value
    #[must_use]
    pub fn max_effect(&self) -> f64 {
        self.pdp_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get the range of the PDP effect
    #[must_use]
    pub fn effect_range(&self) -> f64 {
        self.max_effect() - self.min_effect()
    }

    /// Get the grid coordinates with maximum prediction
    #[must_use]
    pub fn argmax(&self) -> (usize, usize) {
        let mut max_val = f64::NEG_INFINITY;
        let mut max_idx = (0, 0);
        for i in 0..self.pdp_values.nrows() {
            for j in 0..self.pdp_values.ncols() {
                if self.pdp_values[[i, j]] > max_val {
                    max_val = self.pdp_values[[i, j]];
                    max_idx = (i, j);
                }
            }
        }
        max_idx
    }

    /// Get the grid coordinates with minimum prediction
    #[must_use]
    pub fn argmin(&self) -> (usize, usize) {
        let mut min_val = f64::INFINITY;
        let mut min_idx = (0, 0);
        for i in 0..self.pdp_values.nrows() {
            for j in 0..self.pdp_values.ncols() {
                if self.pdp_values[[i, j]] < min_val {
                    min_val = self.pdp_values[[i, j]];
                    min_idx = (i, j);
                }
            }
        }
        min_idx
    }

    /// Create a summary string
    #[must_use]
    pub fn summary(&self) -> String {
        let default_name1 = format!("feature_{}", self.feature_idx_1);
        let default_name2 = format!("feature_{}", self.feature_idx_2);
        let name1 = self.feature_name_1.as_deref().unwrap_or(&default_name1);
        let name2 = self.feature_name_2.as_deref().unwrap_or(&default_name2);

        format!(
            "2D PDP for {} x {}: effect range = {:.4}, grid = {}x{}",
            name1,
            name2,
            self.effect_range(),
            self.grid_values_1.len(),
            self.grid_values_2.len()
        )
    }
}

impl std::fmt::Display for PDP2DResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Method for generating grid values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GridMethod {
    /// Use percentiles of the feature distribution (default)
    Percentile,
    /// Use uniform spacing between min and max
    Uniform,
}

impl Default for GridMethod {
    fn default() -> Self {
        Self::Percentile
    }
}

/// Compute partial dependence for a single feature
///
/// The partial dependence function shows the marginal effect of a feature
/// on the predicted outcome, averaging over all other features.
///
/// # Algorithm
///
/// For each grid point g:
/// 1. Create modified dataset with feature j set to g for all samples
/// 2. Compute predictions for modified dataset
/// 3. Average predictions to get PDP(g)
///
/// # Arguments
///
/// * `model` - Fitted model implementing the `Model` trait
/// * `x` - Feature matrix (n_samples, n_features)
/// * `feature_idx` - Index of the feature to compute PDP for
/// * `n_grid_points` - Number of grid points (default: 50)
/// * `grid_method` - Method for generating grid values
/// * `return_ice` - Whether to return individual ICE curves
///
/// # Returns
///
/// `PDPResult` containing grid values, PDP values, and optionally ICE curves.
///
/// # Example
///
/// ```
/// # fn main() -> ferroml_core::Result<()> {
/// use ferroml_core::explainability::partial_dependence;
/// use ferroml_core::models::RandomForestRegressor;
/// # use ferroml_core::explainability::GridMethod;
/// # use ferroml_core::models::Model;
/// # use ndarray::{Array1, Array2};
/// # let x_train = Array2::from_shape_vec((20, 3), (0..60).map(|i| i as f64 / 60.0).collect()).unwrap();
/// # let y_train = Array1::from_vec((0..20).map(|i| i as f64).collect());
/// # let x_test = x_train.clone();
///
/// let mut model = RandomForestRegressor::new();
/// model.fit(&x_train, &y_train)?;
///
/// let result = partial_dependence(&model, &x_test, 0, 50, GridMethod::Percentile, false)?;
/// println!("Grid: {:?}", result.grid_values);
/// println!("PDP: {:?}", result.pdp_values);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - The model is not fitted
/// - `feature_idx` is out of bounds
/// - Input data is empty
/// - `n_grid_points` is less than 2
///
/// # References
///
/// - Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.
///   Annals of Statistics, 29(5), 1189-1232.
pub fn partial_dependence<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx: usize,
    n_grid_points: usize,
    grid_method: GridMethod,
    return_ice: bool,
) -> Result<PDPResult>
where
    M: Model,
{
    // Validate inputs
    if !model.is_fitted() {
        return Err(FerroError::not_fitted("partial_dependence"));
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
    if n_grid_points < 2 {
        return Err(FerroError::invalid_input(
            "n_grid_points must be at least 2",
        ));
    }

    let n_samples = x.nrows();
    let feature_col = x.column(feature_idx);

    // Generate grid values
    let grid_values = generate_grid(&feature_col.to_owned(), n_grid_points, grid_method);

    // Compute PDP values
    let mut pdp_values = Array1::<f64>::zeros(n_grid_points);
    let mut pdp_std = Array1::<f64>::zeros(n_grid_points);
    let ice_curves = if return_ice {
        Some(Array2::<f64>::zeros((n_samples, n_grid_points)))
    } else {
        None
    };
    let mut ice_curves = ice_curves;

    for (grid_idx, &grid_val) in grid_values.iter().enumerate() {
        // Create modified dataset with feature set to grid value
        let mut x_modified = x.to_owned();
        for i in 0..n_samples {
            x_modified[[i, feature_idx]] = grid_val;
        }

        // Get predictions
        let predictions = model.predict(&x_modified)?;

        // Compute mean and std
        pdp_values[grid_idx] = predictions.mean().unwrap_or(0.0);
        let mean = pdp_values[grid_idx];
        let variance = predictions.iter().map(|&p| (p - mean).powi(2)).sum::<f64>()
            / (n_samples as f64 - 1.0).max(1.0);
        pdp_std[grid_idx] = variance.sqrt();

        // Store ICE curves if requested
        if let Some(ref mut ice) = ice_curves {
            for (i, &pred) in predictions.iter().enumerate() {
                ice[[i, grid_idx]] = pred;
            }
        }
    }

    // Get feature name
    let feature_name = model
        .feature_names()
        .and_then(|names| names.get(feature_idx).cloned());

    Ok(PDPResult {
        grid_values,
        pdp_values,
        pdp_std,
        ice_curves,
        feature_idx,
        feature_name,
        n_samples,
        grid_method,
    })
}

/// Compute partial dependence in parallel using rayon
///
/// Same as `partial_dependence` but parallelizes across grid points
/// for improved performance.
///
/// # Errors
///
/// Returns an error if:
/// - The model is not fitted
/// - `feature_idx` is out of bounds
/// - Input data is empty
/// - `n_grid_points` is less than 2
pub fn partial_dependence_parallel<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx: usize,
    n_grid_points: usize,
    grid_method: GridMethod,
    return_ice: bool,
) -> Result<PDPResult>
where
    M: Model + Sync,
{
    // Validate inputs
    if !model.is_fitted() {
        return Err(FerroError::not_fitted("partial_dependence_parallel"));
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
    if n_grid_points < 2 {
        return Err(FerroError::invalid_input(
            "n_grid_points must be at least 2",
        ));
    }

    let n_samples = x.nrows();
    let feature_col = x.column(feature_idx);

    // Generate grid values
    let grid_values = generate_grid(&feature_col.to_owned(), n_grid_points, grid_method);

    // Compute PDP values in parallel
    let results: Vec<(f64, f64, Vec<f64>)> = grid_values
        .iter()
        .cloned()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|grid_val| {
            // Create modified dataset with feature set to grid value
            let mut x_modified = x.to_owned();
            for i in 0..n_samples {
                x_modified[[i, feature_idx]] = grid_val;
            }

            // Get predictions
            let predictions = model
                .predict(&x_modified)
                .unwrap_or_else(|_| Array1::zeros(n_samples));

            // Compute mean and std
            let mean = predictions.mean().unwrap_or(0.0);
            let variance = predictions.iter().map(|&p| (p - mean).powi(2)).sum::<f64>()
                / (n_samples as f64 - 1.0).max(1.0);
            let std = variance.sqrt();

            let ice = if return_ice {
                predictions.to_vec()
            } else {
                Vec::new()
            };

            (mean, std, ice)
        })
        .collect();

    // Assemble results
    let mut pdp_values = Array1::<f64>::zeros(n_grid_points);
    let mut pdp_std = Array1::<f64>::zeros(n_grid_points);
    let mut ice_curves = if return_ice {
        Some(Array2::<f64>::zeros((n_samples, n_grid_points)))
    } else {
        None
    };

    for (grid_idx, (mean, std, ice)) in results.into_iter().enumerate() {
        pdp_values[grid_idx] = mean;
        pdp_std[grid_idx] = std;

        if let Some(ref mut ice_mat) = ice_curves {
            for (i, &val) in ice.iter().enumerate() {
                ice_mat[[i, grid_idx]] = val;
            }
        }
    }

    // Get feature name
    let feature_name = model
        .feature_names()
        .and_then(|names| names.get(feature_idx).cloned());

    Ok(PDPResult {
        grid_values,
        pdp_values,
        pdp_std,
        ice_curves,
        feature_idx,
        feature_name,
        n_samples,
        grid_method,
    })
}

/// Compute 2D partial dependence for feature interaction
///
/// Shows the joint effect of two features on the predicted outcome,
/// useful for detecting feature interactions.
///
/// # Arguments
///
/// * `model` - Fitted model implementing the `Model` trait
/// * `x` - Feature matrix (n_samples, n_features)
/// * `feature_idx_1` - Index of the first feature
/// * `feature_idx_2` - Index of the second feature
/// * `n_grid_points` - Number of grid points per feature (default: 20)
/// * `grid_method` - Method for generating grid values
///
/// # Returns
///
/// `PDP2DResult` containing grid values and 2D PDP matrix.
///
/// # Errors
///
/// Returns an error if:
/// - The model is not fitted
/// - Feature indices are out of bounds
/// - Feature indices are the same
/// - Input data is empty
/// - `n_grid_points` is less than 2
pub fn partial_dependence_2d<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    n_grid_points: usize,
    grid_method: GridMethod,
) -> Result<PDP2DResult>
where
    M: Model,
{
    // Validate inputs
    if !model.is_fitted() {
        return Err(FerroError::not_fitted("partial_dependence_2d"));
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
    if n_grid_points < 2 {
        return Err(FerroError::invalid_input(
            "n_grid_points must be at least 2",
        ));
    }

    let n_samples = x.nrows();

    // Generate grid values for both features
    let grid_values_1 = generate_grid(
        &x.column(feature_idx_1).to_owned(),
        n_grid_points,
        grid_method,
    );
    let grid_values_2 = generate_grid(
        &x.column(feature_idx_2).to_owned(),
        n_grid_points,
        grid_method,
    );

    // Compute 2D PDP values
    let mut pdp_values = Array2::<f64>::zeros((n_grid_points, n_grid_points));

    for (i, &grid_val_1) in grid_values_1.iter().enumerate() {
        for (j, &grid_val_2) in grid_values_2.iter().enumerate() {
            // Create modified dataset with both features set to grid values
            let mut x_modified = x.to_owned();
            for k in 0..n_samples {
                x_modified[[k, feature_idx_1]] = grid_val_1;
                x_modified[[k, feature_idx_2]] = grid_val_2;
            }

            // Get predictions and average
            let predictions = model.predict(&x_modified)?;
            pdp_values[[i, j]] = predictions.mean().unwrap_or(0.0);
        }
    }

    // Get feature names
    let feature_name_1 = model
        .feature_names()
        .and_then(|names| names.get(feature_idx_1).cloned());
    let feature_name_2 = model
        .feature_names()
        .and_then(|names| names.get(feature_idx_2).cloned());

    Ok(PDP2DResult {
        grid_values_1,
        grid_values_2,
        pdp_values,
        feature_idx_1,
        feature_idx_2,
        feature_name_1,
        feature_name_2,
        n_samples,
    })
}

/// Compute 2D partial dependence in parallel
///
/// # Errors
///
/// Returns an error if:
/// - The model is not fitted
/// - Feature indices are out of bounds
/// - Feature indices are the same
/// - Input data is empty
/// - `n_grid_points` is less than 2
pub fn partial_dependence_2d_parallel<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    n_grid_points: usize,
    grid_method: GridMethod,
) -> Result<PDP2DResult>
where
    M: Model + Sync,
{
    // Validate inputs
    if !model.is_fitted() {
        return Err(FerroError::not_fitted("partial_dependence_2d_parallel"));
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
    if n_grid_points < 2 {
        return Err(FerroError::invalid_input(
            "n_grid_points must be at least 2",
        ));
    }

    let n_samples = x.nrows();

    // Generate grid values for both features
    let grid_values_1 = generate_grid(
        &x.column(feature_idx_1).to_owned(),
        n_grid_points,
        grid_method,
    );
    let grid_values_2 = generate_grid(
        &x.column(feature_idx_2).to_owned(),
        n_grid_points,
        grid_method,
    );

    // Create all grid combinations
    let grid_combinations: Vec<(usize, usize, f64, f64)> = grid_values_1
        .iter()
        .enumerate()
        .flat_map(|(i, &v1)| {
            grid_values_2
                .iter()
                .enumerate()
                .map(move |(j, &v2)| (i, j, v1, v2))
        })
        .collect();

    // Compute PDP values in parallel
    let results: Vec<(usize, usize, f64)> = grid_combinations
        .into_par_iter()
        .map(|(i, j, grid_val_1, grid_val_2)| {
            // Create modified dataset with both features set to grid values
            let mut x_modified = x.to_owned();
            for k in 0..n_samples {
                x_modified[[k, feature_idx_1]] = grid_val_1;
                x_modified[[k, feature_idx_2]] = grid_val_2;
            }

            // Get predictions and average
            let predictions = model
                .predict(&x_modified)
                .unwrap_or_else(|_| Array1::zeros(n_samples));
            let mean = predictions.mean().unwrap_or(0.0);

            (i, j, mean)
        })
        .collect();

    // Assemble results
    let mut pdp_values = Array2::<f64>::zeros((n_grid_points, n_grid_points));
    for (i, j, mean) in results {
        pdp_values[[i, j]] = mean;
    }

    // Get feature names
    let feature_name_1 = model
        .feature_names()
        .and_then(|names| names.get(feature_idx_1).cloned());
    let feature_name_2 = model
        .feature_names()
        .and_then(|names| names.get(feature_idx_2).cloned());

    Ok(PDP2DResult {
        grid_values_1,
        grid_values_2,
        pdp_values,
        feature_idx_1,
        feature_idx_2,
        feature_name_1,
        feature_name_2,
        n_samples,
    })
}

/// Compute partial dependence for multiple features
///
/// Convenience function to compute PDPs for multiple features at once.
///
/// # Arguments
///
/// * `model` - Fitted model implementing the `Model` trait
/// * `x` - Feature matrix (n_samples, n_features)
/// * `feature_indices` - Indices of features to compute PDPs for
/// * `n_grid_points` - Number of grid points per feature
/// * `grid_method` - Method for generating grid values
///
/// # Returns
///
/// Vector of `PDPResult` for each requested feature.
///
/// # Errors
///
/// Returns an error if any feature index is invalid or model is not fitted.
pub fn partial_dependence_multi<M>(
    model: &M,
    x: &Array2<f64>,
    feature_indices: &[usize],
    n_grid_points: usize,
    grid_method: GridMethod,
) -> Result<Vec<PDPResult>>
where
    M: Model,
{
    feature_indices
        .iter()
        .map(|&idx| partial_dependence(model, x, idx, n_grid_points, grid_method, false))
        .collect()
}

/// Compute partial dependence for multiple features in parallel
///
/// # Errors
///
/// Returns an error if any feature index is invalid or model is not fitted.
pub fn partial_dependence_multi_parallel<M>(
    model: &M,
    x: &Array2<f64>,
    feature_indices: &[usize],
    n_grid_points: usize,
    grid_method: GridMethod,
) -> Result<Vec<PDPResult>>
where
    M: Model + Sync,
{
    // Validate model first
    if !model.is_fitted() {
        return Err(FerroError::not_fitted("partial_dependence_multi_parallel"));
    }

    // Compute PDPs in parallel across features
    let results: Vec<Result<PDPResult>> = feature_indices
        .par_iter()
        .map(|&idx| partial_dependence_parallel(model, x, idx, n_grid_points, grid_method, false))
        .collect();

    // Collect results, propagating any errors
    results.into_iter().collect()
}

/// Generate grid values for a feature
pub(super) fn generate_grid(
    feature_values: &Array1<f64>,
    n_points: usize,
    method: GridMethod,
) -> Array1<f64> {
    let mut sorted: Vec<f64> = feature_values.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    match method {
        GridMethod::Percentile => {
            // Use percentiles to handle non-uniform distributions
            (0..n_points)
                .map(|i| {
                    let p = i as f64 / (n_points - 1) as f64;
                    percentile(&sorted, p)
                })
                .collect()
        }
        GridMethod::Uniform => {
            // Uniform spacing between min and max
            let min = sorted.first().copied().unwrap_or(0.0);
            let max = sorted.last().copied().unwrap_or(1.0);
            let step = (max - min) / (n_points - 1) as f64;

            (0..n_points)
                .map(|i| (i as f64).mul_add(step, min))
                .collect()
        }
    }
}

/// Calculate percentile from sorted array
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let idx = p * (sorted.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower].mul_add(1.0 - frac, sorted[upper] * frac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::linear::LinearRegression;
    use ndarray::Axis;

    #[test]
    fn test_partial_dependence_basic() {
        // Create linear data: y = 2*x0 + 3*x1 + 1
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

        // Fit linear regression
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // Compute PDP for feature 0
        let result = partial_dependence(&model, &x, 0, 20, GridMethod::Percentile, false).unwrap();

        assert_eq!(result.grid_values.len(), 20);
        assert_eq!(result.pdp_values.len(), 20);
        assert_eq!(result.feature_idx, 0);

        // For linear model, PDP should be monotonic
        assert!(
            result.is_monotonic_increasing(),
            "PDP should be monotonic increasing for positive coefficient"
        );

        // Effect range should be approximately 2 * (max_x0 - min_x0)
        let x0_range = x.column(0).iter().fold(0.0_f64, |a, &b| a.max(b))
            - x.column(0).iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let expected_range = 2.0 * x0_range;
        assert!(
            (result.effect_range() - expected_range).abs() < 0.5,
            "Effect range {} should be close to {}",
            result.effect_range(),
            expected_range
        );
    }

    #[test]
    fn test_partial_dependence_with_ice() {
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

        // Request ICE curves
        let result = partial_dependence(&model, &x, 0, 10, GridMethod::Percentile, true).unwrap();

        assert!(result.ice_curves.is_some());
        let ice = result.ice_curves.as_ref().unwrap();
        assert_eq!(ice.shape(), &[n_samples, 10]);
    }

    #[test]
    fn test_partial_dependence_parallel() {
        let n_samples = 50;
        let x = Array2::from_shape_fn((n_samples, 3), |(i, j)| match j {
            0 => i as f64 * 0.1,
            1 => ((i * 7 + 11) % 19) as f64 * 0.1,
            _ => ((i * 11 + 3) % 23) as f64 * 0.1,
        });
        let y: Array1<f64> = x.axis_iter(Axis(0)).map(|row| row[0] * 2.0).collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result_seq =
            partial_dependence(&model, &x, 0, 15, GridMethod::Percentile, false).unwrap();
        let result_par =
            partial_dependence_parallel(&model, &x, 0, 15, GridMethod::Percentile, false).unwrap();

        // Results should be very similar (may differ slightly due to parallelization)
        for i in 0..15 {
            assert!(
                (result_seq.pdp_values[i] - result_par.pdp_values[i]).abs() < 1e-6,
                "Mismatch at grid point {}: {} vs {}",
                i,
                result_seq.pdp_values[i],
                result_par.pdp_values[i]
            );
        }
    }

    #[test]
    fn test_partial_dependence_2d() {
        let n_samples = 50;
        let x = Array2::from_shape_fn((n_samples, 3), |(i, j)| match j {
            0 => i as f64 * 0.1,
            1 => ((i * 7 + 11) % 19) as f64 * 0.1,
            _ => ((i * 11 + 3) % 23) as f64 * 0.1,
        });
        let y: Array1<f64> = x.axis_iter(Axis(0)).map(|row| row[0] * row[1]).collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = partial_dependence_2d(&model, &x, 0, 1, 10, GridMethod::Percentile).unwrap();

        assert_eq!(result.grid_values_1.len(), 10);
        assert_eq!(result.grid_values_2.len(), 10);
        assert_eq!(result.pdp_values.shape(), &[10, 10]);
        assert_eq!(result.feature_idx_1, 0);
        assert_eq!(result.feature_idx_2, 1);
    }

    #[test]
    fn test_partial_dependence_2d_parallel() {
        let n_samples = 30;
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

        let result_seq =
            partial_dependence_2d(&model, &x, 0, 1, 8, GridMethod::Percentile).unwrap();
        let result_par =
            partial_dependence_2d_parallel(&model, &x, 0, 1, 8, GridMethod::Percentile).unwrap();

        // Results should be identical
        for i in 0..8 {
            for j in 0..8 {
                assert!(
                    (result_seq.pdp_values[[i, j]] - result_par.pdp_values[[i, j]]).abs() < 1e-6,
                    "Mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_partial_dependence_multi() {
        // Create features with independent values to avoid collinearity
        let n_samples = 50;
        let x = Array2::from_shape_fn((n_samples, 3), |(i, j)| {
            match j {
                0 => i as f64 * 0.1,                   // range ~0-5, informative
                1 => ((i * 7 + 11) % 50) as f64 * 0.1, // range ~0-5, informative (different pattern)
                _ => ((i * 11 + 3) % 23) as f64 * 0.1, // noise feature (not used in y)
            }
        });
        // y = 2*x0 + x1 (feature 0 has 2x the coefficient of feature 1)
        let y: Array1<f64> = x
            .axis_iter(Axis(0))
            .map(|row| 2.0 * row[0] + row[1])
            .collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let results =
            partial_dependence_multi(&model, &x, &[0, 1, 2], 10, GridMethod::Percentile).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].feature_idx, 0);
        assert_eq!(results[1].feature_idx, 1);
        assert_eq!(results[2].feature_idx, 2);

        // Features 0 and 1 should have positive effects, feature 2 should be near zero
        // We just verify that informative features have larger effects than noise
        let min_informative_effect = results[0].effect_range().min(results[1].effect_range());
        assert!(
            min_informative_effect > results[2].effect_range() * 2.0,
            "Informative feature effect {} should be larger than noise effect {}",
            min_informative_effect,
            results[2].effect_range()
        );
    }

    #[test]
    fn test_grid_method_percentile() {
        // Create skewed data
        let values: Array1<f64> = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.3, 10.0]);
        let grid = generate_grid(&values, 5, GridMethod::Percentile);

        // Percentile method should capture the distribution
        assert!(grid[0] < 1.0); // First grid point should be in the dense region
        assert!(grid[4] > 5.0); // Last grid point should capture the outlier
    }

    #[test]
    fn test_grid_method_uniform() {
        let values: Array1<f64> = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.3, 10.0]);
        let grid = generate_grid(&values, 5, GridMethod::Uniform);

        // Uniform method should have equal spacing
        let step = 10.0 / 4.0;
        for i in 1..5 {
            let expected = i as f64 * step;
            assert!(
                (grid[i] - expected).abs() < 1e-10,
                "Grid point {} should be {}, got {}",
                i,
                expected,
                grid[i]
            );
        }
    }

    #[test]
    fn test_pdp_result_methods() {
        let result = PDPResult {
            grid_values: Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]),
            pdp_values: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            pdp_std: Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1]),
            ice_curves: None,
            feature_idx: 0,
            feature_name: Some("feature_x".to_string()),
            n_samples: 100,
            grid_method: GridMethod::Percentile,
        };

        assert!((result.min_effect() - 1.0).abs() < 1e-10);
        assert!((result.max_effect() - 5.0).abs() < 1e-10);
        assert!((result.effect_range() - 4.0).abs() < 1e-10);
        assert!(result.is_monotonic_increasing());
        assert!(result.is_monotonic());
        assert!(!result.is_monotonic_decreasing());
        assert_eq!(result.argmax(), 4);
        assert_eq!(result.argmin(), 0);
    }

    #[test]
    fn test_pdp_result_non_monotonic() {
        let result = PDPResult {
            grid_values: Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]),
            pdp_values: Array1::from_vec(vec![1.0, 3.0, 2.0, 4.0, 3.0]), // Non-monotonic
            pdp_std: Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1]),
            ice_curves: None,
            feature_idx: 0,
            feature_name: None,
            n_samples: 100,
            grid_method: GridMethod::Percentile,
        };

        assert!(!result.is_monotonic());
        assert!(!result.is_monotonic_increasing());
        assert!(!result.is_monotonic_decreasing());
    }

    #[test]
    fn test_pdp2d_result_methods() {
        let result = PDP2DResult {
            grid_values_1: Array1::from_vec(vec![0.0, 1.0, 2.0]),
            grid_values_2: Array1::from_vec(vec![0.0, 1.0, 2.0]),
            pdp_values: Array2::from_shape_vec(
                (3, 3),
                vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
            )
            .unwrap(),
            feature_idx_1: 0,
            feature_idx_2: 1,
            feature_name_1: None,
            feature_name_2: None,
            n_samples: 100,
        };

        assert!((result.min_effect() - 1.0).abs() < 1e-10);
        assert!((result.max_effect() - 9.0).abs() < 1e-10);
        assert_eq!(result.argmax(), (2, 2));
        assert_eq!(result.argmin(), (0, 0));
    }

    #[test]
    fn test_error_not_fitted() {
        let model = LinearRegression::new();
        let x = Array2::zeros((10, 2));

        let result = partial_dependence(&model, &x, 0, 10, GridMethod::Percentile, false);
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

        let result = partial_dependence(&model, &x, 5, 10, GridMethod::Percentile, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_same_feature_2d() {
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

        let result = partial_dependence_2d(&model, &x, 0, 0, 10, GridMethod::Percentile);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_empty_data() {
        let mut model = LinearRegression::new();
        // Need proper non-collinear data to fit
        let x = Array2::from_shape_fn((10, 2), |(i, j)| {
            if j == 0 {
                i as f64
            } else {
                ((i * 7) % 10) as f64
            }
        });
        let y: Array1<f64> = x.axis_iter(Axis(0)).map(|row| row[0]).collect();
        model.fit(&x, &y).unwrap();

        // Test with empty data
        let x_empty: Array2<f64> = Array2::zeros((0, 2));
        let result = partial_dependence(&model, &x_empty, 0, 10, GridMethod::Percentile, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_too_few_grid_points() {
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

        let result = partial_dependence(&model, &x, 0, 1, GridMethod::Percentile, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_summary_display() {
        let result = PDPResult {
            grid_values: Array1::from_vec(vec![0.0, 1.0, 2.0]),
            pdp_values: Array1::from_vec(vec![1.0, 2.0, 3.0]),
            pdp_std: Array1::from_vec(vec![0.1, 0.1, 0.1]),
            ice_curves: None,
            feature_idx: 0,
            feature_name: Some("important_feature".to_string()),
            n_samples: 100,
            grid_method: GridMethod::Percentile,
        };

        let summary = result.summary();
        assert!(summary.contains("important_feature"));
        assert!(summary.contains("increasing"));
    }

    #[test]
    fn test_heterogeneity() {
        let result = PDPResult {
            grid_values: Array1::from_vec(vec![0.0, 1.0, 2.0]),
            pdp_values: Array1::from_vec(vec![1.0, 2.0, 3.0]),
            pdp_std: Array1::from_vec(vec![0.5, 0.3, 0.4]),
            ice_curves: None,
            feature_idx: 0,
            feature_name: None,
            n_samples: 100,
            grid_method: GridMethod::Percentile,
        };

        let heterogeneity = result.heterogeneity();
        assert!((heterogeneity - 0.4).abs() < 1e-10);
    }
}

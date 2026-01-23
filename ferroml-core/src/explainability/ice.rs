//! Individual Conditional Expectation (ICE) Implementation
//!
//! ICE plots show the effect of a feature on predictions for each individual sample,
//! unlike PDP which shows the average effect. ICE curves reveal heterogeneity and
//! potential feature interactions that PDPs might mask.
//!
//! ## Variants
//!
//! - **ICE**: Raw individual predictions across feature grid
//! - **c-ICE** (Centered ICE): Curves centered at a reference point for easier comparison
//! - **d-ICE** (Derivative ICE): Derivatives of ICE curves for detecting non-linear effects

use crate::models::Model;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::partial_dependence::{generate_grid, GridMethod};

/// Configuration for ICE computation
#[derive(Debug, Clone)]
pub struct ICEConfig {
    /// Number of grid points for the feature
    pub n_grid_points: usize,
    /// Method for generating grid values
    pub grid_method: GridMethod,
    /// Whether to compute centered ICE (c-ICE)
    pub center: bool,
    /// Reference point index for centering (default: 0, i.e., minimum feature value)
    pub center_reference_idx: usize,
    /// Whether to compute derivative ICE (d-ICE)
    pub compute_derivative: bool,
    /// Optional subset of sample indices to compute ICE for (useful for large datasets)
    pub sample_indices: Option<Vec<usize>>,
}

impl Default for ICEConfig {
    fn default() -> Self {
        Self {
            n_grid_points: 50,
            grid_method: GridMethod::Percentile,
            center: false,
            center_reference_idx: 0,
            compute_derivative: false,
            sample_indices: None,
        }
    }
}

impl ICEConfig {
    /// Create a new ICEConfig with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of grid points
    #[must_use]
    pub fn with_n_grid_points(mut self, n: usize) -> Self {
        self.n_grid_points = n;
        self
    }

    /// Set the grid method
    #[must_use]
    pub fn with_grid_method(mut self, method: GridMethod) -> Self {
        self.grid_method = method;
        self
    }

    /// Enable centered ICE computation
    #[must_use]
    pub fn with_centering(mut self, center_reference_idx: usize) -> Self {
        self.center = true;
        self.center_reference_idx = center_reference_idx;
        self
    }

    /// Enable derivative ICE computation
    #[must_use]
    pub fn with_derivative(mut self) -> Self {
        self.compute_derivative = true;
        self
    }

    /// Set sample indices to compute ICE for (subset)
    #[must_use]
    pub fn with_sample_indices(mut self, indices: Vec<usize>) -> Self {
        self.sample_indices = Some(indices);
        self
    }
}

/// Result of ICE computation for a single feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ICEResult {
    /// Grid values for the feature
    pub grid_values: Array1<f64>,
    /// Raw ICE curves (n_samples x n_grid_points)
    /// Each row is one sample's prediction trajectory across feature values
    pub ice_curves: Array2<f64>,
    /// Centered ICE curves (n_samples x n_grid_points), if computed
    /// Centered at the reference point for easier comparison
    pub centered_ice: Option<Array2<f64>>,
    /// Derivative ICE curves (n_samples x (n_grid_points - 1)), if computed
    /// Shows rate of change of prediction with respect to feature value
    pub derivative_ice: Option<Array2<f64>>,
    /// PDP values (average of ICE curves)
    pub pdp_values: Array1<f64>,
    /// Centered PDP values (average of centered ICE), if centering was applied
    pub centered_pdp: Option<Array1<f64>>,
    /// Feature index
    pub feature_idx: usize,
    /// Feature name (if available)
    pub feature_name: Option<String>,
    /// Number of samples used
    pub n_samples: usize,
    /// Grid method used
    pub grid_method: GridMethod,
    /// Reference index used for centering
    pub center_reference_idx: Option<usize>,
    /// Sample indices used (if a subset was specified)
    pub sample_indices: Option<Vec<usize>>,
}

impl ICEResult {
    /// Get the minimum ICE value across all curves
    #[must_use]
    pub fn min_value(&self) -> f64 {
        self.ice_curves
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Get the maximum ICE value across all curves
    #[must_use]
    pub fn max_value(&self) -> f64 {
        self.ice_curves
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get the range of ICE values
    #[must_use]
    pub fn value_range(&self) -> f64 {
        self.max_value() - self.min_value()
    }

    /// Compute heterogeneity measure (std of ICE curves at each grid point)
    ///
    /// High heterogeneity suggests feature interactions may be present.
    #[must_use]
    pub fn heterogeneity(&self) -> Array1<f64> {
        let n_grid = self.ice_curves.ncols();
        let mut result = Array1::zeros(n_grid);

        for j in 0..n_grid {
            let col = self.ice_curves.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let variance = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
                / (col.len() as f64 - 1.0).max(1.0);
            result[j] = variance.sqrt();
        }

        result
    }

    /// Compute mean heterogeneity across all grid points
    #[must_use]
    pub fn mean_heterogeneity(&self) -> f64 {
        self.heterogeneity().mean().unwrap_or(0.0)
    }

    /// Detect potential interactions by checking if heterogeneity varies across grid
    ///
    /// Returns true if the coefficient of variation of heterogeneity exceeds threshold.
    #[must_use]
    pub fn has_interactions(&self, threshold: f64) -> bool {
        let heterogeneity = self.heterogeneity();
        let mean = heterogeneity.mean().unwrap_or(0.0);
        if mean < 1e-10 {
            return false;
        }
        let std = (heterogeneity
            .iter()
            .map(|&h| (h - mean).powi(2))
            .sum::<f64>()
            / heterogeneity.len() as f64)
            .sqrt();
        let cv = std / mean;
        cv > threshold
    }

    /// Get the sample with the strongest positive effect (largest slope)
    #[must_use]
    pub fn sample_with_strongest_positive_effect(&self) -> Option<usize> {
        if self.n_samples == 0 || self.ice_curves.ncols() < 2 {
            return None;
        }

        let mut max_slope = f64::NEG_INFINITY;
        let mut max_idx = 0;

        for i in 0..self.n_samples {
            // Compute overall slope (last - first) / range
            let first = self.ice_curves[[i, 0]];
            let last = self.ice_curves[[i, self.ice_curves.ncols() - 1]];
            let slope = last - first;

            if slope > max_slope {
                max_slope = slope;
                max_idx = i;
            }
        }

        Some(max_idx)
    }

    /// Get the sample with the strongest negative effect (largest negative slope)
    #[must_use]
    pub fn sample_with_strongest_negative_effect(&self) -> Option<usize> {
        if self.n_samples == 0 || self.ice_curves.ncols() < 2 {
            return None;
        }

        let mut min_slope = f64::INFINITY;
        let mut min_idx = 0;

        for i in 0..self.n_samples {
            let first = self.ice_curves[[i, 0]];
            let last = self.ice_curves[[i, self.ice_curves.ncols() - 1]];
            let slope = last - first;

            if slope < min_slope {
                min_slope = slope;
                min_idx = i;
            }
        }

        Some(min_idx)
    }

    /// Get the effect range per sample (max - min prediction across grid)
    #[must_use]
    pub fn per_sample_effect_range(&self) -> Array1<f64> {
        let mut result = Array1::zeros(self.n_samples);

        for i in 0..self.n_samples {
            let row = self.ice_curves.row(i);
            let min = row.iter().copied().fold(f64::INFINITY, f64::min);
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            result[i] = max - min;
        }

        result
    }

    /// Compute the fraction of curves that are monotonically increasing
    #[must_use]
    pub fn fraction_monotonic_increasing(&self) -> f64 {
        let n_grid = self.ice_curves.ncols();
        if n_grid < 2 {
            return 1.0;
        }

        let monotonic_count: usize = (0..self.n_samples)
            .filter(|&i| {
                let row = self.ice_curves.row(i);
                (0..n_grid - 1).all(|j| row[j + 1] >= row[j] - 1e-10)
            })
            .count();

        monotonic_count as f64 / self.n_samples as f64
    }

    /// Compute the fraction of curves that are monotonically decreasing
    #[must_use]
    pub fn fraction_monotonic_decreasing(&self) -> f64 {
        let n_grid = self.ice_curves.ncols();
        if n_grid < 2 {
            return 1.0;
        }

        let monotonic_count: usize = (0..self.n_samples)
            .filter(|&i| {
                let row = self.ice_curves.row(i);
                (0..n_grid - 1).all(|j| row[j + 1] <= row[j] + 1e-10)
            })
            .count();

        monotonic_count as f64 / self.n_samples as f64
    }

    /// Create a summary string
    #[must_use]
    pub fn summary(&self) -> String {
        let default_name = format!("feature_{}", self.feature_idx);
        let name = self.feature_name.as_deref().unwrap_or(&default_name);

        format!(
            "ICE for {}: {} samples, heterogeneity = {:.4}, \
             {:.1}% monotonic increasing, {:.1}% monotonic decreasing",
            name,
            self.n_samples,
            self.mean_heterogeneity(),
            self.fraction_monotonic_increasing() * 100.0,
            self.fraction_monotonic_decreasing() * 100.0,
        )
    }
}

impl std::fmt::Display for ICEResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Compute Individual Conditional Expectation (ICE) curves for a single feature
///
/// ICE curves show how predictions change for each individual sample as the feature
/// value varies across its range. Unlike PDP (which shows the average), ICE reveals
/// heterogeneity in feature effects across samples.
///
/// # Algorithm
///
/// For each sample i and grid point g:
/// 1. Create a copy of the sample with feature j set to g
/// 2. Compute prediction for this modified sample
/// 3. Store as ICE[i, g]
///
/// # Arguments
///
/// * `model` - Fitted model implementing the `Model` trait
/// * `x` - Feature matrix (n_samples, n_features)
/// * `feature_idx` - Index of the feature to compute ICE for
/// * `config` - Configuration for ICE computation
///
/// # Returns
///
/// `ICEResult` containing ICE curves and optionally centered/derivative versions.
///
/// # Example
///
/// ```ignore
/// use ferroml_core::explainability::{individual_conditional_expectation, ICEConfig};
/// use ferroml_core::models::RandomForestRegressor;
///
/// let mut model = RandomForestRegressor::new();
/// model.fit(&x_train, &y_train)?;
///
/// // Basic ICE
/// let result = individual_conditional_expectation(&model, &x_test, 0, ICEConfig::default())?;
///
/// // ICE with centering and derivatives
/// let config = ICEConfig::new()
///     .with_centering(0)  // Center at first grid point
///     .with_derivative();
/// let result = individual_conditional_expectation(&model, &x_test, 0, config)?;
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
/// - Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015). Peeking Inside
///   the Black Box: Visualizing Statistical Learning With Plots of Individual
///   Conditional Expectation. Journal of Computational and Graphical Statistics.
pub fn individual_conditional_expectation<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx: usize,
    config: ICEConfig,
) -> Result<ICEResult>
where
    M: Model,
{
    // Validate inputs
    if !model.is_fitted() {
        return Err(FerroError::not_fitted("individual_conditional_expectation"));
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
    if config.n_grid_points < 2 {
        return Err(FerroError::invalid_input(
            "n_grid_points must be at least 2",
        ));
    }

    // Determine which samples to use
    let sample_indices = match &config.sample_indices {
        Some(indices) => {
            // Validate indices
            for &idx in indices {
                if idx >= x.nrows() {
                    return Err(FerroError::invalid_input(format!(
                        "sample index {} is out of bounds (n_samples = {})",
                        idx,
                        x.nrows()
                    )));
                }
            }
            indices.clone()
        }
        None => (0..x.nrows()).collect(),
    };

    let n_samples = sample_indices.len();
    let feature_col = x.column(feature_idx);

    // Generate grid values
    let grid_values = generate_grid(
        &feature_col.to_owned(),
        config.n_grid_points,
        config.grid_method,
    );
    let n_grid = grid_values.len();

    // Compute ICE curves
    let mut ice_curves = Array2::<f64>::zeros((n_samples, n_grid));

    for (grid_idx, &grid_val) in grid_values.iter().enumerate() {
        // Create modified dataset with feature set to grid value
        let mut x_modified = x.to_owned();
        for i in 0..x.nrows() {
            x_modified[[i, feature_idx]] = grid_val;
        }

        // Get predictions
        let predictions = model.predict(&x_modified)?;

        // Store predictions for selected samples
        for (i, &sample_idx) in sample_indices.iter().enumerate() {
            ice_curves[[i, grid_idx]] = predictions[sample_idx];
        }
    }

    // Compute PDP (average of ICE curves)
    let pdp_values: Array1<f64> = ice_curves
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(n_grid));

    // Compute centered ICE if requested
    let (centered_ice, centered_pdp, center_reference_idx) = if config.center {
        let ref_idx = config.center_reference_idx.min(n_grid - 1);
        let centered = center_ice_curves(&ice_curves, ref_idx);
        let centered_pdp_vals: Array1<f64> = centered
            .mean_axis(Axis(0))
            .unwrap_or_else(|| Array1::zeros(n_grid));
        (Some(centered), Some(centered_pdp_vals), Some(ref_idx))
    } else {
        (None, None, None)
    };

    // Compute derivative ICE if requested
    let derivative_ice = if config.compute_derivative {
        Some(compute_derivative_ice(&ice_curves, &grid_values))
    } else {
        None
    };

    // Get feature name
    let feature_name = model
        .feature_names()
        .and_then(|names| names.get(feature_idx).cloned());

    Ok(ICEResult {
        grid_values,
        ice_curves,
        centered_ice,
        derivative_ice,
        pdp_values,
        centered_pdp,
        feature_idx,
        feature_name,
        n_samples,
        grid_method: config.grid_method,
        center_reference_idx,
        sample_indices: config.sample_indices,
    })
}

/// Compute Individual Conditional Expectation (ICE) curves in parallel
///
/// Same as `individual_conditional_expectation` but parallelizes across grid points.
///
/// # Errors
///
/// Returns an error if the model is not fitted, feature_idx is out of bounds,
/// input is empty, or n_grid_points < 2.
pub fn individual_conditional_expectation_parallel<M>(
    model: &M,
    x: &Array2<f64>,
    feature_idx: usize,
    config: ICEConfig,
) -> Result<ICEResult>
where
    M: Model + Sync,
{
    // Validate inputs
    if !model.is_fitted() {
        return Err(FerroError::not_fitted(
            "individual_conditional_expectation_parallel",
        ));
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
    if config.n_grid_points < 2 {
        return Err(FerroError::invalid_input(
            "n_grid_points must be at least 2",
        ));
    }

    // Determine which samples to use
    let sample_indices = match &config.sample_indices {
        Some(indices) => {
            for &idx in indices {
                if idx >= x.nrows() {
                    return Err(FerroError::invalid_input(format!(
                        "sample index {} is out of bounds (n_samples = {})",
                        idx,
                        x.nrows()
                    )));
                }
            }
            indices.clone()
        }
        None => (0..x.nrows()).collect(),
    };

    let n_samples = sample_indices.len();
    let n_total = x.nrows();
    let feature_col = x.column(feature_idx);

    // Generate grid values
    let grid_values = generate_grid(
        &feature_col.to_owned(),
        config.n_grid_points,
        config.grid_method,
    );
    let n_grid = grid_values.len();

    // Compute ICE curves in parallel
    let grid_vec: Vec<f64> = grid_values.iter().copied().collect();
    let results: Vec<Vec<f64>> = grid_vec
        .into_par_iter()
        .map(|grid_val| {
            // Create modified dataset with feature set to grid value
            let mut x_modified = x.to_owned();
            for i in 0..n_total {
                x_modified[[i, feature_idx]] = grid_val;
            }

            // Get predictions
            let predictions = model
                .predict(&x_modified)
                .unwrap_or_else(|_| Array1::zeros(n_total));

            // Extract predictions for selected samples
            sample_indices.iter().map(|&idx| predictions[idx]).collect()
        })
        .collect();

    // Assemble ICE curves
    let mut ice_curves = Array2::<f64>::zeros((n_samples, n_grid));
    for (grid_idx, preds) in results.into_iter().enumerate() {
        for (sample_idx, pred) in preds.into_iter().enumerate() {
            ice_curves[[sample_idx, grid_idx]] = pred;
        }
    }

    // Compute PDP (average of ICE curves)
    let pdp_values: Array1<f64> = ice_curves
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(n_grid));

    // Compute centered ICE if requested
    let (centered_ice, centered_pdp, center_reference_idx) = if config.center {
        let ref_idx = config.center_reference_idx.min(n_grid - 1);
        let centered = center_ice_curves(&ice_curves, ref_idx);
        let centered_pdp_vals: Array1<f64> = centered
            .mean_axis(Axis(0))
            .unwrap_or_else(|| Array1::zeros(n_grid));
        (Some(centered), Some(centered_pdp_vals), Some(ref_idx))
    } else {
        (None, None, None)
    };

    // Compute derivative ICE if requested
    let derivative_ice = if config.compute_derivative {
        Some(compute_derivative_ice(&ice_curves, &grid_values))
    } else {
        None
    };

    // Get feature name
    let feature_name = model
        .feature_names()
        .and_then(|names| names.get(feature_idx).cloned());

    Ok(ICEResult {
        grid_values,
        ice_curves,
        centered_ice,
        derivative_ice,
        pdp_values,
        centered_pdp,
        feature_idx,
        feature_name,
        n_samples,
        grid_method: config.grid_method,
        center_reference_idx,
        sample_indices: config.sample_indices,
    })
}

/// Center ICE curves at a reference point
///
/// Centered ICE (c-ICE) curves start at zero at the reference point, making it
/// easier to compare the relative effects across samples.
///
/// # Arguments
///
/// * `ice_curves` - Raw ICE curves (n_samples x n_grid_points)
/// * `reference_idx` - Index of the grid point to use as reference
///
/// # Returns
///
/// Centered ICE curves where each curve has been shifted so that
/// `centered_ice[i, reference_idx] = 0` for all samples i.
pub fn center_ice_curves(ice_curves: &Array2<f64>, reference_idx: usize) -> Array2<f64> {
    let n_samples = ice_curves.nrows();
    let n_grid = ice_curves.ncols();
    let ref_idx = reference_idx.min(n_grid.saturating_sub(1));

    let mut centered = Array2::<f64>::zeros((n_samples, n_grid));

    for i in 0..n_samples {
        let reference_value = ice_curves[[i, ref_idx]];
        for j in 0..n_grid {
            centered[[i, j]] = ice_curves[[i, j]] - reference_value;
        }
    }

    centered
}

/// Compute derivative ICE curves (d-ICE)
///
/// Derivative ICE shows the rate of change of predictions with respect to feature
/// values. This is useful for detecting non-linear effects and understanding where
/// the feature has the strongest impact.
///
/// # Arguments
///
/// * `ice_curves` - Raw ICE curves (n_samples x n_grid_points)
/// * `grid_values` - Grid values for the feature
///
/// # Returns
///
/// Derivative ICE curves (n_samples x (n_grid_points - 1)) computed using
/// finite differences.
pub fn compute_derivative_ice(ice_curves: &Array2<f64>, grid_values: &Array1<f64>) -> Array2<f64> {
    let n_samples = ice_curves.nrows();
    let n_grid = ice_curves.ncols();

    if n_grid < 2 {
        return Array2::zeros((n_samples, 0));
    }

    let n_derivatives = n_grid - 1;
    let mut derivatives = Array2::<f64>::zeros((n_samples, n_derivatives));

    for i in 0..n_samples {
        for j in 0..n_derivatives {
            let dy = ice_curves[[i, j + 1]] - ice_curves[[i, j]];
            let dx = grid_values[j + 1] - grid_values[j];
            derivatives[[i, j]] = if dx.abs() > 1e-10 { dy / dx } else { 0.0 };
        }
    }

    derivatives
}

/// Compute ICE for multiple features
///
/// Convenience function to compute ICE curves for multiple features at once.
///
/// # Errors
///
/// Returns an error if any feature index is invalid or model is not fitted.
pub fn ice_multi<M>(
    model: &M,
    x: &Array2<f64>,
    feature_indices: &[usize],
    config: ICEConfig,
) -> Result<Vec<ICEResult>>
where
    M: Model,
{
    feature_indices
        .iter()
        .map(|&idx| individual_conditional_expectation(model, x, idx, config.clone()))
        .collect()
}

/// Compute ICE for multiple features in parallel
///
/// # Errors
///
/// Returns an error if any feature index is invalid or model is not fitted.
pub fn ice_multi_parallel<M>(
    model: &M,
    x: &Array2<f64>,
    feature_indices: &[usize],
    config: ICEConfig,
) -> Result<Vec<ICEResult>>
where
    M: Model + Sync,
{
    // Validate model first
    if !model.is_fitted() {
        return Err(FerroError::not_fitted("ice_multi_parallel"));
    }

    // Compute ICE in parallel across features
    let results: Vec<Result<ICEResult>> = feature_indices
        .par_iter()
        .map(|&idx| individual_conditional_expectation_parallel(model, x, idx, config.clone()))
        .collect();

    results.into_iter().collect()
}

/// Convert existing raw ICE curves (from PDPResult) to ICEResult
///
/// This utility function converts ICE curves from a PDP computation
/// into a full ICEResult with analysis capabilities.
///
/// # Arguments
///
/// * `ice_curves` - Raw ICE curves (n_samples x n_grid_points)
/// * `grid_values` - Grid values for the feature
/// * `feature_idx` - Index of the feature
/// * `feature_name` - Optional feature name
/// * `center` - Whether to compute centered ICE
/// * `compute_derivative` - Whether to compute derivative ICE
///
/// # Returns
///
/// `ICEResult` with the provided curves and computed analyses.
#[must_use]
pub fn ice_from_curves(
    ice_curves: Array2<f64>,
    grid_values: Array1<f64>,
    feature_idx: usize,
    feature_name: Option<String>,
    center: bool,
    compute_derivative: bool,
) -> ICEResult {
    let n_samples = ice_curves.nrows();
    let n_grid = grid_values.len();

    // Compute PDP (average of ICE curves)
    let pdp_values: Array1<f64> = ice_curves
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(n_grid));

    // Compute centered ICE if requested
    let (centered_ice, centered_pdp, center_reference_idx) = if center {
        let centered = center_ice_curves(&ice_curves, 0);
        let centered_pdp_vals: Array1<f64> = centered
            .mean_axis(Axis(0))
            .unwrap_or_else(|| Array1::zeros(n_grid));
        (Some(centered), Some(centered_pdp_vals), Some(0))
    } else {
        (None, None, None)
    };

    // Compute derivative ICE if requested
    let derivative_ice = if compute_derivative {
        Some(compute_derivative_ice(&ice_curves, &grid_values))
    } else {
        None
    };

    ICEResult {
        grid_values,
        ice_curves,
        centered_ice,
        derivative_ice,
        pdp_values,
        centered_pdp,
        feature_idx,
        feature_name,
        n_samples,
        grid_method: GridMethod::Percentile, // Assume percentile if unknown
        center_reference_idx,
        sample_indices: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::linear::LinearRegression;
    use crate::testing::assertions::tolerances;
    use crate::assert_approx_eq;
    use ndarray::Axis as NdAxis;

    fn create_linear_dataset(n_samples: usize) -> (Array2<f64>, Array1<f64>) {
        // Create features with independent values
        let x = Array2::from_shape_fn((n_samples, 3), |(i, j)| match j {
            0 => i as f64 * 0.1,                   // x0: informative
            1 => ((i * 7 + 11) % 50) as f64 * 0.1, // x1: informative
            _ => ((i * 11 + 3) % 23) as f64 * 0.1, // x2: noise
        });
        // y = 2*x0 + 3*x1 + 1
        let y: Array1<f64> = x
            .axis_iter(NdAxis(0))
            .map(|row| 2.0 * row[0] + 3.0 * row[1] + 1.0)
            .collect();
        (x, y)
    }

    #[test]
    fn test_ice_basic() {
        let (x, y) = create_linear_dataset(50);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_n_grid_points(10);
        let result = individual_conditional_expectation(&model, &x, 0, config).unwrap();

        assert_eq!(result.grid_values.len(), 10);
        assert_eq!(result.ice_curves.shape(), &[50, 10]);
        assert_eq!(result.pdp_values.len(), 10);
        assert_eq!(result.feature_idx, 0);
        assert_eq!(result.n_samples, 50);
        assert!(result.centered_ice.is_none());
        assert!(result.derivative_ice.is_none());
    }

    #[test]
    fn test_ice_with_centering() {
        let (x, y) = create_linear_dataset(30);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_n_grid_points(10).with_centering(0); // Center at first grid point
        let result = individual_conditional_expectation(&model, &x, 0, config).unwrap();

        assert!(result.centered_ice.is_some());
        let centered = result.centered_ice.as_ref().unwrap();
        assert_eq!(centered.shape(), &[30, 10]);

        // Check that all curves start at 0 (centered at index 0)
        for i in 0..30 {
            assert!(
                centered[[i, 0]].abs() < 1e-10,
                "Sample {} should be centered at 0, got {}",
                i,
                centered[[i, 0]]
            );
        }

        assert!(result.center_reference_idx.is_some());
        assert_eq!(result.center_reference_idx.unwrap(), 0);
    }

    #[test]
    fn test_ice_with_derivative() {
        let (x, y) = create_linear_dataset(30);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_n_grid_points(10).with_derivative();
        let result = individual_conditional_expectation(&model, &x, 0, config).unwrap();

        assert!(result.derivative_ice.is_some());
        let derivatives = result.derivative_ice.as_ref().unwrap();
        assert_eq!(derivatives.shape(), &[30, 9]); // n_grid - 1

        // For linear model with coefficient 2, derivatives should be approximately 2
        for i in 0..30 {
            for j in 0..9 {
                assert!(
                    (derivatives[[i, j]] - 2.0).abs() < 0.5,
                    "Derivative at ({}, {}) should be ~2, got {}",
                    i,
                    j,
                    derivatives[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_ice_with_sample_subset() {
        let (x, y) = create_linear_dataset(100);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let sample_indices = vec![0, 10, 20, 30, 40];
        let config = ICEConfig::new()
            .with_n_grid_points(10)
            .with_sample_indices(sample_indices.clone());
        let result = individual_conditional_expectation(&model, &x, 0, config).unwrap();

        assert_eq!(result.n_samples, 5);
        assert_eq!(result.ice_curves.shape(), &[5, 10]);
        assert!(result.sample_indices.is_some());
        assert_eq!(result.sample_indices.as_ref().unwrap(), &sample_indices);
    }

    #[test]
    fn test_ice_parallel() {
        let (x, y) = create_linear_dataset(50);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_n_grid_points(15);
        let result_seq = individual_conditional_expectation(&model, &x, 0, config.clone()).unwrap();
        let result_par =
            individual_conditional_expectation_parallel(&model, &x, 0, config).unwrap();

        // Results should be very similar
        for i in 0..50 {
            for j in 0..15 {
                assert!(
                    (result_seq.ice_curves[[i, j]] - result_par.ice_curves[[i, j]]).abs() < 1e-6,
                    "Mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_ice_multi() {
        let (x, y) = create_linear_dataset(30);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_n_grid_points(10);
        let results = ice_multi(&model, &x, &[0, 1, 2], config).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].feature_idx, 0);
        assert_eq!(results[1].feature_idx, 1);
        assert_eq!(results[2].feature_idx, 2);
    }

    #[test]
    fn test_ice_heterogeneity() {
        let result = ICEResult {
            grid_values: Array1::from_vec(vec![0.0, 1.0, 2.0]),
            ice_curves: Array2::from_shape_vec(
                (4, 3),
                vec![
                    1.0, 2.0, 3.0, // Sample 0
                    0.5, 1.5, 2.5, // Sample 1
                    1.5, 2.5, 3.5, // Sample 2
                    1.0, 2.0, 3.0, // Sample 3
                ],
            )
            .unwrap(),
            centered_ice: None,
            derivative_ice: None,
            pdp_values: Array1::from_vec(vec![1.0, 2.0, 3.0]),
            centered_pdp: None,
            feature_idx: 0,
            feature_name: None,
            n_samples: 4,
            grid_method: GridMethod::Percentile,
            center_reference_idx: None,
            sample_indices: None,
        };

        let heterogeneity = result.heterogeneity();
        assert_eq!(heterogeneity.len(), 3);

        // All columns have same heterogeneity (std of [1.0, 0.5, 1.5, 1.0] = 0.408...)
        let expected_std = 0.408;
        for h in heterogeneity.iter() {
            assert!(
                (*h - expected_std).abs() < 0.01,
                "Expected heterogeneity ~{}, got {}",
                expected_std,
                h
            );
        }
    }

    #[test]
    fn test_ice_monotonicity_fractions() {
        let result = ICEResult {
            grid_values: Array1::from_vec(vec![0.0, 1.0, 2.0]),
            ice_curves: Array2::from_shape_vec(
                (4, 3),
                vec![
                    1.0, 2.0, 3.0, // Sample 0: increasing
                    3.0, 2.0, 1.0, // Sample 1: decreasing
                    1.0, 2.0, 3.0, // Sample 2: increasing
                    1.0, 3.0, 2.0, // Sample 3: neither
                ],
            )
            .unwrap(),
            centered_ice: None,
            derivative_ice: None,
            pdp_values: Array1::from_vec(vec![1.5, 2.25, 2.25]),
            centered_pdp: None,
            feature_idx: 0,
            feature_name: None,
            n_samples: 4,
            grid_method: GridMethod::Percentile,
            center_reference_idx: None,
            sample_indices: None,
        };

        assert!((result.fraction_monotonic_increasing() - 0.5).abs() < 0.01);
        assert!((result.fraction_monotonic_decreasing() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_ice_strongest_effects() {
        let result = ICEResult {
            grid_values: Array1::from_vec(vec![0.0, 1.0, 2.0]),
            ice_curves: Array2::from_shape_vec(
                (4, 3),
                vec![
                    1.0, 2.0, 3.0, // Sample 0: slope = 2
                    0.0, 0.0, 5.0, // Sample 1: slope = 5 (strongest positive)
                    5.0, 3.0, 1.0, // Sample 2: slope = -4 (strongest negative)
                    1.0, 1.5, 2.0, // Sample 3: slope = 1
                ],
            )
            .unwrap(),
            centered_ice: None,
            derivative_ice: None,
            pdp_values: Array1::from_vec(vec![1.75, 1.625, 2.75]),
            centered_pdp: None,
            feature_idx: 0,
            feature_name: None,
            n_samples: 4,
            grid_method: GridMethod::Percentile,
            center_reference_idx: None,
            sample_indices: None,
        };

        assert_eq!(result.sample_with_strongest_positive_effect(), Some(1));
        assert_eq!(result.sample_with_strongest_negative_effect(), Some(2));
    }

    #[test]
    fn test_ice_per_sample_effect_range() {
        let result = ICEResult {
            grid_values: Array1::from_vec(vec![0.0, 1.0, 2.0]),
            ice_curves: Array2::from_shape_vec(
                (3, 3),
                vec![
                    1.0, 2.0, 3.0, // Range = 2
                    0.0, 5.0, 2.0, // Range = 5
                    1.0, 1.0, 1.0, // Range = 0
                ],
            )
            .unwrap(),
            centered_ice: None,
            derivative_ice: None,
            pdp_values: Array1::from_vec(vec![0.67, 2.67, 2.0]),
            centered_pdp: None,
            feature_idx: 0,
            feature_name: None,
            n_samples: 3,
            grid_method: GridMethod::Percentile,
            center_reference_idx: None,
            sample_indices: None,
        };

        let ranges = result.per_sample_effect_range();
        assert_approx_eq!(ranges[0], 2.0, tolerances::CLOSED_FORM, "sample 0 effect range");
        assert_approx_eq!(ranges[1], 5.0, tolerances::CLOSED_FORM, "sample 1 effect range");
        assert_approx_eq!(ranges[2], 0.0, tolerances::CLOSED_FORM, "sample 2 effect range");
    }

    #[test]
    fn test_center_ice_curves() {
        let ice_curves = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, // Sample 0
                2.0, 3.0, 4.0, 5.0, // Sample 1
                0.0, 1.0, 2.0, 3.0, // Sample 2
            ],
        )
        .unwrap();

        // Center at index 0
        let centered = center_ice_curves(&ice_curves, 0);
        for i in 0..3 {
            assert!(centered[[i, 0]].abs() < 1e-10, "Should be 0 at reference");
        }
        assert!((centered[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((centered[[0, 2]] - 2.0).abs() < 1e-10);

        // Center at index 2
        let centered_mid = center_ice_curves(&ice_curves, 2);
        for i in 0..3 {
            assert!(
                centered_mid[[i, 2]].abs() < 1e-10,
                "Should be 0 at reference"
            );
        }
    }

    #[test]
    fn test_compute_derivative_ice() {
        let ice_curves = Array2::from_shape_vec(
            (2, 4),
            vec![
                0.0, 1.0, 3.0, 6.0, // Sample 0: increasing derivatives
                0.0, 2.0, 2.0, 2.0, // Sample 1: constant then zero derivative
            ],
        )
        .unwrap();
        let grid_values = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        let derivatives = compute_derivative_ice(&ice_curves, &grid_values);
        assert_eq!(derivatives.shape(), &[2, 3]);

        // Sample 0: derivatives are 1, 2, 3
        assert!((derivatives[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((derivatives[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((derivatives[[0, 2]] - 3.0).abs() < 1e-10);

        // Sample 1: derivatives are 2, 0, 0
        assert!((derivatives[[1, 0]] - 2.0).abs() < 1e-10);
        assert!(derivatives[[1, 1]].abs() < 1e-10);
        assert!(derivatives[[1, 2]].abs() < 1e-10);
    }

    #[test]
    fn test_ice_from_curves() {
        let ice_curves = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0],
        )
        .unwrap();
        let grid_values = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        let result = ice_from_curves(
            ice_curves.clone(),
            grid_values.clone(),
            0,
            Some("test_feature".to_string()),
            true,
            true,
        );

        assert_eq!(result.n_samples, 3);
        assert_eq!(result.ice_curves, ice_curves);
        assert_eq!(result.grid_values, grid_values);
        assert!(result.centered_ice.is_some());
        assert!(result.derivative_ice.is_some());
        assert_eq!(result.feature_name, Some("test_feature".to_string()));
    }

    #[test]
    fn test_ice_summary_display() {
        let result = ICEResult {
            grid_values: Array1::from_vec(vec![0.0, 1.0, 2.0]),
            ice_curves: Array2::from_shape_vec(
                (3, 3),
                vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            )
            .unwrap(),
            centered_ice: None,
            derivative_ice: None,
            pdp_values: Array1::from_vec(vec![1.0, 2.0, 3.0]),
            centered_pdp: None,
            feature_idx: 0,
            feature_name: Some("important_feature".to_string()),
            n_samples: 3,
            grid_method: GridMethod::Percentile,
            center_reference_idx: None,
            sample_indices: None,
        };

        let summary = result.summary();
        assert!(summary.contains("important_feature"));
        assert!(summary.contains("3 samples"));
    }

    #[test]
    fn test_error_not_fitted() {
        let model = LinearRegression::new();
        let x = Array2::zeros((10, 2));

        let config = ICEConfig::default();
        let result = individual_conditional_expectation(&model, &x, 0, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_invalid_feature_idx() {
        let (x, y) = create_linear_dataset(20);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::default();
        let result = individual_conditional_expectation(&model, &x, 10, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_invalid_sample_idx() {
        let (x, y) = create_linear_dataset(20);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_sample_indices(vec![0, 5, 100]); // 100 is invalid
        let result = individual_conditional_expectation(&model, &x, 0, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_empty_data() {
        let mut model = LinearRegression::new();
        let (x, y) = create_linear_dataset(10);
        model.fit(&x, &y).unwrap();

        let x_empty: Array2<f64> = Array2::zeros((0, 3));
        let config = ICEConfig::default();
        let result = individual_conditional_expectation(&model, &x_empty, 0, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_too_few_grid_points() {
        let (x, y) = create_linear_dataset(20);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_n_grid_points(1);
        let result = individual_conditional_expectation(&model, &x, 0, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_has_interactions() {
        // Homogeneous heterogeneity (no interaction)
        let result_no_interaction = ICEResult {
            grid_values: Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]),
            ice_curves: Array2::from_shape_vec(
                (4, 4),
                vec![
                    1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5, 1.5, 2.5, 3.5, 4.5, 1.0, 2.0, 3.0, 4.0,
                ],
            )
            .unwrap(),
            centered_ice: None,
            derivative_ice: None,
            pdp_values: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
            centered_pdp: None,
            feature_idx: 0,
            feature_name: None,
            n_samples: 4,
            grid_method: GridMethod::Percentile,
            center_reference_idx: None,
            sample_indices: None,
        };

        // With uniform heterogeneity, CV should be low
        assert!(!result_no_interaction.has_interactions(0.5));

        // Varying heterogeneity (potential interaction)
        let result_interaction = ICEResult {
            grid_values: Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]),
            ice_curves: Array2::from_shape_vec(
                (4, 4),
                vec![
                    1.0, 2.0, 3.0, 10.0, // Diverge at end
                    1.0, 2.0, 3.0, 0.0, // Opposite direction at end
                    1.0, 2.0, 3.0, 10.0, 1.0, 2.0, 3.0, 0.0,
                ],
            )
            .unwrap(),
            centered_ice: None,
            derivative_ice: None,
            pdp_values: Array1::from_vec(vec![1.0, 2.0, 3.0, 5.0]),
            centered_pdp: None,
            feature_idx: 0,
            feature_name: None,
            n_samples: 4,
            grid_method: GridMethod::Percentile,
            center_reference_idx: None,
            sample_indices: None,
        };

        // With varying heterogeneity, should detect interaction
        assert!(result_interaction.has_interactions(0.3));
    }
}

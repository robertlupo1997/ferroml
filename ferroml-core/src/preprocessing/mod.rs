//! Preprocessing Transformers
//!
//! This module provides feature preprocessing with proper statistical handling and
//! a consistent API across all transformer types.
//!
//! ## Design Philosophy
//!
//! - **Fit/Transform Separation**: All transformers learn parameters during `fit()`
//!   and apply them during `transform()`. This prevents data leakage in pipelines.
//!
//! - **Feature Name Tracking**: Transformers track feature names through transformations,
//!   essential for interpretability and debugging.
//!
//! - **Statistically Rigorous**: Transformers include proper handling of edge cases
//!   (constant features, missing values, etc.) with appropriate warnings.
//!
//! ## Module Structure
//!
//! - [`Transformer`] - Core trait that all preprocessing components implement
//! - [`scalers`] - Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
//! - [`encoders`] - Categorical encoding (OneHotEncoder, OrdinalEncoder, TargetEncoder)
//! - [`imputers`] - Missing value handling (SimpleImputer, KNNImputer)
//! - [`selection`] - Feature selection (VarianceThreshold, SelectKBest, RFE)
//! - [`power`] - Power transformations (Box-Cox, Yeo-Johnson)
//! - [`polynomial`] - Polynomial feature generation (PolynomialFeatures)
//!
//! ## Example
//!
//! ```
//! use ferroml_core::preprocessing::{Transformer, scalers::StandardScaler};
//! use ndarray::array;
//!
//! // Create and fit a scaler
//! let mut scaler = StandardScaler::new();
//! let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//!
//! // Fit learns the mean and std
//! scaler.fit(&x).unwrap();
//!
//! // Transform applies the learned parameters
//! let x_scaled = scaler.transform(&x).unwrap();
//!
//! // Inverse transform recovers original scale
//! let x_recovered = scaler.inverse_transform(&x_scaled).unwrap();
//! ```

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::{FerroError, Result};

// Submodules (to be implemented in subsequent tasks)
pub mod discretizers;
pub mod encoders;
pub mod imputers;
pub mod polynomial;
pub mod power;
pub mod quantile;
pub mod sampling;
pub mod scalers;
pub mod selection;

#[cfg(test)]
mod compliance_tests;

/// Core trait for all preprocessing transformers.
///
/// All transformers follow the fit/transform pattern:
/// 1. `fit()` learns parameters from training data
/// 2. `transform()` applies learned parameters to new data
/// 3. `inverse_transform()` reverses the transformation (when applicable)
///
/// # Thread Safety
///
/// Transformers implement `Send + Sync` to support parallel pipelines.
/// The `fit()` method takes `&mut self`, so fitting must be done single-threaded,
/// but `transform()` can be called concurrently from multiple threads.
///
/// # Example Implementation
///
/// ```ignore
/// struct MyScaler {
///     mean: Option<Array1<f64>>,
///     feature_names: Option<Vec<String>>,
/// }
///
/// impl Transformer for MyScaler {
///     fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
///         self.mean = Some(x.mean_axis(Axis(0)).unwrap());
///         Ok(())
///     }
///
///     fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
///         let mean = self.mean.as_ref()
///             .ok_or_else(|| FerroError::not_fitted("transform"))?;
///         Ok(x - mean)
///     }
///
///     fn is_fitted(&self) -> bool {
///         self.mean.is_some()
///     }
/// }
/// ```
pub trait Transformer: Send + Sync {
    /// Fit the transformer to training data.
    ///
    /// This method learns the parameters needed for transformation from the
    /// training data. For example, a `StandardScaler` learns mean and standard
    /// deviation during fitting.
    ///
    /// # Arguments
    ///
    /// * `x` - Training data of shape `(n_samples, n_features)`
    ///
    /// # Returns
    ///
    /// * `Ok(())` if fitting succeeds
    /// * `Err(FerroError)` if fitting fails (e.g., empty input, numerical issues)
    ///
    /// # Errors
    ///
    /// - `FerroError::InvalidInput` if the input is empty or malformed
    /// - `FerroError::NumericalError` if numerical issues occur (e.g., zero variance)
    fn fit(&mut self, x: &Array2<f64>) -> Result<()>;

    /// Transform data using fitted parameters.
    ///
    /// Applies the transformation learned during `fit()` to new data.
    /// The transformer must be fitted before calling this method.
    ///
    /// # Arguments
    ///
    /// * `x` - Data to transform of shape `(n_samples, n_features)`
    ///
    /// # Returns
    ///
    /// * Transformed data array
    ///
    /// # Errors
    ///
    /// - `FerroError::NotFitted` if `fit()` hasn't been called
    /// - `FerroError::ShapeMismatch` if input has wrong number of features
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>;

    /// Fit and transform in one step.
    ///
    /// Equivalent to calling `fit(x)` followed by `transform(x)`, but may be
    /// more efficient for some transformers.
    ///
    /// # Arguments
    ///
    /// * `x` - Data to fit and transform
    ///
    /// # Returns
    ///
    /// * Transformed data array
    fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Reverse the transformation.
    ///
    /// Recovers the original data from transformed data. Not all transformers
    /// support inverse transformation (e.g., feature selection is not reversible).
    ///
    /// # Arguments
    ///
    /// * `x` - Transformed data to invert
    ///
    /// # Returns
    ///
    /// * Data in the original feature space
    ///
    /// # Errors
    ///
    /// - `FerroError::NotImplemented` if inverse is not supported
    /// - `FerroError::NotFitted` if `fit()` hasn't been called
    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let _ = x;
        Err(FerroError::NotImplemented(
            "inverse_transform not supported for this transformer".to_string(),
        ))
    }

    /// Check if the transformer has been fitted.
    ///
    /// Returns `true` if `fit()` has been called successfully and the
    /// transformer is ready to transform data.
    fn is_fitted(&self) -> bool;

    /// Get the names of output features.
    ///
    /// Returns the names of features after transformation. This is important
    /// for interpretability, especially when transformers change the number
    /// of features (e.g., `OneHotEncoder`).
    ///
    /// # Arguments
    ///
    /// * `input_names` - Optional names of input features. If not provided,
    ///   default names like "x0", "x1", etc. are used.
    ///
    /// # Returns
    ///
    /// * `Some(Vec<String>)` with output feature names if fitted
    /// * `None` if not fitted
    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>>;

    /// Get the number of input features expected.
    ///
    /// Returns `None` if not fitted.
    fn n_features_in(&self) -> Option<usize>;

    /// Get the number of output features produced.
    ///
    /// Returns `None` if not fitted.
    fn n_features_out(&self) -> Option<usize>;
}

/// Configuration for handling unknown categories in encoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnknownCategoryHandling {
    /// Raise an error when encountering unknown categories
    Error,
    /// Ignore unknown categories (output zeros for one-hot)
    Ignore,
    /// Map unknown categories to a special "unknown" category
    InfrequentIfExist,
}

impl Default for UnknownCategoryHandling {
    fn default() -> Self {
        Self::Error
    }
}

/// Statistics computed during fitting, useful for diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitStatistics {
    /// Number of samples seen during fitting
    pub n_samples: usize,
    /// Number of features in the input
    pub n_features_in: usize,
    /// Number of features in the output
    pub n_features_out: usize,
    /// Indices of features with zero variance (if any)
    pub constant_features: Vec<usize>,
    /// Number of missing values per feature (if tracked)
    pub missing_counts: Option<Vec<usize>>,
}

/// Validates that the transformer is fitted before transformation.
///
/// # Arguments
///
/// * `is_fitted` - Whether the transformer is fitted
/// * `operation` - Name of the operation being attempted
///
/// # Returns
///
/// * `Ok(())` if fitted
/// * `Err(FerroError::NotFitted)` if not fitted
#[inline]
pub fn check_is_fitted(is_fitted: bool, operation: &str) -> Result<()> {
    if is_fitted {
        Ok(())
    } else {
        Err(FerroError::not_fitted(operation))
    }
}

/// Validates that input dimensions match expected dimensions from fitting.
///
/// # Arguments
///
/// * `x` - Input array to validate
/// * `expected_features` - Number of features expected
///
/// # Returns
///
/// * `Ok(())` if dimensions match
/// * `Err(FerroError::ShapeMismatch)` if dimensions don't match
#[inline]
pub fn check_shape(x: &Array2<f64>, expected_features: usize) -> Result<()> {
    let actual_features = x.ncols();
    if actual_features == expected_features {
        Ok(())
    } else {
        Err(FerroError::shape_mismatch(
            format!("({}, {})", x.nrows(), expected_features),
            format!("({}, {})", x.nrows(), actual_features),
        ))
    }
}

/// Validates that input is not empty.
///
/// # Arguments
///
/// * `x` - Input array to validate
///
/// # Returns
///
/// * `Ok(())` if not empty
/// * `Err(FerroError::InvalidInput)` if empty
#[inline]
pub fn check_non_empty(x: &Array2<f64>) -> Result<()> {
    if x.is_empty() {
        Err(FerroError::invalid_input("Input array cannot be empty"))
    } else if x.nrows() == 0 {
        Err(FerroError::invalid_input(
            "Input array must have at least one sample",
        ))
    } else if x.ncols() == 0 {
        Err(FerroError::invalid_input(
            "Input array must have at least one feature",
        ))
    } else {
        Ok(())
    }
}

/// Generates default feature names like ["x0", "x1", "x2", ...].
///
/// # Arguments
///
/// * `n_features` - Number of features
///
/// # Returns
///
/// * Vector of feature names
pub fn generate_feature_names(n_features: usize) -> Vec<String> {
    (0..n_features).map(|i| format!("x{}", i)).collect()
}

/// Computes column-wise statistics for fitting scalers.
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// * Tuple of (mean, variance, n_samples) per feature
pub fn compute_column_statistics(x: &Array2<f64>) -> (Array1<f64>, Array1<f64>, usize) {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    let mut mean: Array1<f64> = Array1::zeros(n_features);
    let mut m2: Array1<f64> = Array1::zeros(n_features); // Sum of squared differences from mean

    // Use Welford's online algorithm for numerical stability
    for (i, row) in x.rows().into_iter().enumerate() {
        let count = (i + 1) as f64;
        for (j, &val) in row.iter().enumerate() {
            let delta = val - mean[j];
            mean[j] += delta / count;
            let delta2 = val - mean[j];
            m2[j] += delta * delta2;
        }
    }

    // Compute population variance (n denominator) to match sklearn behavior
    let variance = if n_samples > 0 {
        m2.mapv(|v| v / n_samples as f64)
    } else {
        Array1::zeros(n_features)
    };

    (mean, variance, n_samples)
}

/// Computes the mean of each column.
pub fn column_mean(x: &Array2<f64>) -> Array1<f64> {
    x.mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(x.ncols()))
}

/// Computes the standard deviation of each column.
pub fn column_std(x: &Array2<f64>, ddof: usize) -> Array1<f64> {
    x.std_axis(Axis(0), ddof as f64)
}

/// Computes the minimum of each column.
pub fn column_min(x: &Array2<f64>) -> Array1<f64> {
    let n_features = x.ncols();
    let mut mins = Array1::from_elem(n_features, f64::INFINITY);

    for row in x.rows() {
        for (j, &val) in row.iter().enumerate() {
            if val < mins[j] {
                mins[j] = val;
            }
        }
    }

    mins
}

/// Computes the maximum of each column.
pub fn column_max(x: &Array2<f64>) -> Array1<f64> {
    let n_features = x.ncols();
    let mut maxs = Array1::from_elem(n_features, f64::NEG_INFINITY);

    for row in x.rows() {
        for (j, &val) in row.iter().enumerate() {
            if val > maxs[j] {
                maxs[j] = val;
            }
        }
    }

    maxs
}

/// Computes the median of each column.
pub fn column_median(x: &Array2<f64>) -> Array1<f64> {
    let n_features = x.ncols();
    let mut medians = Array1::zeros(n_features);

    for j in 0..n_features {
        let mut col: Vec<f64> = x.column(j).to_vec();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = col.len();
        medians[j] = if n % 2 == 0 {
            (col[n / 2 - 1] + col[n / 2]) / 2.0
        } else {
            col[n / 2]
        };
    }

    medians
}

/// Computes a specific quantile of each column.
///
/// # Arguments
///
/// * `x` - Input array
/// * `q` - Quantile in [0, 1]
///
/// # Returns
///
/// * Quantile value for each column
pub fn column_quantile(x: &Array2<f64>, q: f64) -> Array1<f64> {
    assert!((0.0..=1.0).contains(&q), "Quantile must be in [0, 1]");

    let n_features = x.ncols();
    let mut quantiles = Array1::zeros(n_features);

    for j in 0..n_features {
        let mut col: Vec<f64> = x.column(j).to_vec();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = col.len();

        if n == 0 {
            quantiles[j] = f64::NAN;
        } else if n == 1 {
            quantiles[j] = col[0];
        } else {
            // Linear interpolation between two nearest ranks
            let pos = q * (n - 1) as f64;
            let lower = pos.floor() as usize;
            let upper = pos.ceil() as usize;
            let frac = pos - lower as f64;

            quantiles[j] = if lower == upper {
                col[lower]
            } else {
                col[lower].mul_add(1.0 - frac, col[upper] * frac)
            };
        }
    }

    quantiles
}

/// Identifies constant features (zero variance).
///
/// # Arguments
///
/// * `x` - Input array
/// * `threshold` - Variance threshold below which a feature is considered constant
///
/// # Returns
///
/// * Indices of constant features
pub fn find_constant_features(x: &Array2<f64>, threshold: f64) -> Vec<usize> {
    let variance = x.var_axis(Axis(0), 0.0);
    variance
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v <= threshold { Some(i) } else { None })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_check_is_fitted() {
        assert!(check_is_fitted(true, "transform").is_ok());
        assert!(check_is_fitted(false, "transform").is_err());
    }

    #[test]
    fn test_check_shape() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(check_shape(&x, 2).is_ok());
        assert!(check_shape(&x, 3).is_err());
    }

    #[test]
    fn test_check_non_empty() {
        let x = array![[1.0, 2.0]];
        assert!(check_non_empty(&x).is_ok());

        let empty: Array2<f64> = Array2::zeros((0, 0));
        assert!(check_non_empty(&empty).is_err());
    }

    #[test]
    fn test_generate_feature_names() {
        let names = generate_feature_names(3);
        assert_eq!(names, vec!["x0", "x1", "x2"]);
    }

    #[test]
    fn test_column_mean() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mean = column_mean(&x);
        assert!((mean[0] - 3.0).abs() < 1e-10);
        assert!((mean[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_column_min_max() {
        let x = array![[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]];
        let min = column_min(&x);
        let max = column_max(&x);

        assert!((min[0] - 1.0).abs() < 1e-10);
        assert!((min[1] - 2.0).abs() < 1e-10);
        assert!((max[0] - 3.0).abs() < 1e-10);
        assert!((max[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_column_median() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 5.0]];
        let median = column_median(&x);
        assert!((median[0] - 2.0).abs() < 1e-10);
        assert!((median[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_column_quantile() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

        let q0 = column_quantile(&x, 0.0);
        assert!((q0[0] - 1.0).abs() < 1e-10);

        let q50 = column_quantile(&x, 0.5);
        assert!((q50[0] - 3.0).abs() < 1e-10);

        let q100 = column_quantile(&x, 1.0);
        assert!((q100[0] - 5.0).abs() < 1e-10);

        let q25 = column_quantile(&x, 0.25);
        assert!((q25[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_constant_features() {
        let x = array![[1.0, 5.0, 3.0], [1.0, 2.0, 3.0], [1.0, 8.0, 3.0]];
        let constant = find_constant_features(&x, 1e-10);
        assert_eq!(constant, vec![0, 2]); // First and third columns are constant
    }

    #[test]
    fn test_compute_column_statistics() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (mean, variance, n) = compute_column_statistics(&x);

        assert_eq!(n, 3);
        assert!((mean[0] - 3.0).abs() < 1e-10);
        assert!((mean[1] - 4.0).abs() < 1e-10);
        // Population variance (n): var([1,3,5]) = 8/3, var([2,4,6]) = 8/3
        assert!((variance[0] - 8.0 / 3.0).abs() < 1e-10);
        assert!((variance[1] - 8.0 / 3.0).abs() < 1e-10);
    }
}

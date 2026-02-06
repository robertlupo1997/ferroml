//! Power Transformations
//!
//! This module provides transformers that apply power transformations to make
//! data more Gaussian-like, which is beneficial for many machine learning algorithms.
//!
//! ## Available Transformers
//!
//! - [`PowerTransformer`] - Box-Cox and Yeo-Johnson transformations
//!
//! ## Power Transformation Methods
//!
//! | Method | Requirements | Notes |
//! |--------|-------------|-------|
//! | Box-Cox | Strictly positive data | Classic power transform |
//! | Yeo-Johnson | Any data | Extension of Box-Cox for non-positive values |
//!
//! ## When to Use Power Transformations
//!
//! Power transformations are useful when:
//! - Data is skewed and you want to make it more Gaussian-like
//! - You want to stabilize variance across the range of data
//! - Linear models require normally distributed features
//!
//! ## Example
//!
//! ```ignore
//! use ferroml_core::preprocessing::power::{PowerTransformer, PowerMethod};
//! use ferroml_core::preprocessing::Transformer;
//! use ndarray::array;
//!
//! // Yeo-Johnson handles any data (positive, negative, zero)
//! let mut transformer = PowerTransformer::new(PowerMethod::YeoJohnson);
//! let x = array![[1.0, -2.0], [4.0, 3.0], [9.0, -1.0]];
//!
//! let x_transformed = transformer.fit_transform(&x).unwrap();
//! // Data is now more Gaussian-like
//! ```

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::{check_is_fitted, check_non_empty, check_shape, generate_feature_names, Transformer};
use crate::{FerroError, Result};

/// Power transformation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerMethod {
    /// Box-Cox transformation.
    ///
    /// Requires strictly positive data (x > 0 for all x).
    ///
    /// Transformation:
    /// - For λ ≠ 0: y = (x^λ - 1) / λ
    /// - For λ = 0: y = ln(x)
    BoxCox,

    /// Yeo-Johnson transformation.
    ///
    /// Handles positive, negative, and zero values.
    ///
    /// Transformation:
    /// - For x ≥ 0, λ ≠ 0: y = ((x + 1)^λ - 1) / λ
    /// - For x ≥ 0, λ = 0: y = ln(x + 1)
    /// - For x < 0, λ ≠ 2: y = -((-x + 1)^(2-λ) - 1) / (2 - λ)
    /// - For x < 0, λ = 2: y = -ln(-x + 1)
    YeoJohnson,
}

impl Default for PowerMethod {
    fn default() -> Self {
        Self::YeoJohnson
    }
}

/// Power transformer that applies Box-Cox or Yeo-Johnson transformations.
///
/// Power transformations are useful for transforming data to be more Gaussian-like,
/// which can improve the performance of many machine learning algorithms that assume
/// normally distributed features.
///
/// # Lambda Optimization
///
/// The optimal lambda parameter for each feature is found via maximum likelihood
/// estimation (MLE). The search range is [-5, 5] by default, which covers most
/// practical use cases.
///
/// # Standardization
///
/// After the power transformation, the data can optionally be standardized
/// (zero mean, unit variance) which is enabled by default.
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::power::{PowerTransformer, PowerMethod};
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// // Create transformer with Yeo-Johnson method
/// let mut transformer = PowerTransformer::new(PowerMethod::YeoJohnson);
/// let x = array![[1.0], [4.0], [9.0], [16.0]];
///
/// transformer.fit(&x).unwrap();
/// let x_transformed = transformer.transform(&x).unwrap();
///
/// // Can recover original data (approximately)
/// let x_recovered = transformer.inverse_transform(&x_transformed).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerTransformer {
    /// The power transformation method to use
    method: PowerMethod,
    /// Whether to standardize output (zero mean, unit variance)
    standardize: bool,
    /// Learned lambda parameters for each feature
    lambdas: Option<Array1<f64>>,
    /// Mean of each feature after transformation (for standardization)
    mean: Option<Array1<f64>>,
    /// Std of each feature after transformation (for standardization)
    std: Option<Array1<f64>>,
    /// Number of features seen during fit
    n_features_in: Option<usize>,
    /// Number of samples seen during fit
    n_samples_seen: Option<usize>,
}

impl PowerTransformer {
    /// Create a new PowerTransformer with the specified method.
    ///
    /// Default configuration:
    /// - `standardize`: true (output has zero mean and unit variance)
    ///
    /// # Arguments
    ///
    /// * `method` - The power transformation method (BoxCox or YeoJohnson)
    ///
    /// # Example
    ///
    /// ```
    /// use ferroml_core::preprocessing::power::{PowerTransformer, PowerMethod};
    ///
    /// // Box-Cox (requires positive data)
    /// let transformer = PowerTransformer::new(PowerMethod::BoxCox);
    ///
    /// // Yeo-Johnson (handles any data)
    /// let transformer = PowerTransformer::new(PowerMethod::YeoJohnson);
    /// ```
    pub fn new(method: PowerMethod) -> Self {
        Self {
            method,
            standardize: true,
            lambdas: None,
            mean: None,
            std: None,
            n_features_in: None,
            n_samples_seen: None,
        }
    }

    /// Configure whether to standardize the transformed output.
    ///
    /// When enabled (default), the transformed data is centered and scaled
    /// to have zero mean and unit variance.
    ///
    /// # Arguments
    ///
    /// * `standardize` - Whether to standardize after transformation
    pub fn with_standardize(mut self, standardize: bool) -> Self {
        self.standardize = standardize;
        self
    }

    /// Get the power transformation method.
    pub fn method(&self) -> PowerMethod {
        self.method
    }

    /// Get the learned lambda parameters for each feature.
    ///
    /// Returns `None` if not fitted.
    pub fn lambdas(&self) -> Option<&Array1<f64>> {
        self.lambdas.as_ref()
    }

    /// Apply Box-Cox transformation to a single value.
    ///
    /// For λ ≠ 0: y = (x^λ - 1) / λ
    /// For λ = 0: y = ln(x)
    #[inline]
    fn box_cox_transform(x: f64, lambda: f64) -> f64 {
        if lambda.abs() < 1e-10 {
            x.ln()
        } else {
            (x.powf(lambda) - 1.0) / lambda
        }
    }

    /// Apply inverse Box-Cox transformation to a single value.
    ///
    /// For λ ≠ 0: x = (y * λ + 1)^(1/λ)
    /// For λ = 0: x = exp(y)
    #[inline]
    fn box_cox_inverse(y: f64, lambda: f64) -> f64 {
        if lambda.abs() < 1e-10 {
            y.exp()
        } else {
            y.mul_add(lambda, 1.0).powf(1.0 / lambda)
        }
    }

    /// Apply Yeo-Johnson transformation to a single value.
    ///
    /// For x ≥ 0, λ ≠ 0: y = ((x + 1)^λ - 1) / λ
    /// For x ≥ 0, λ = 0: y = ln(x + 1)
    /// For x < 0, λ ≠ 2: y = -((-x + 1)^(2-λ) - 1) / (2 - λ)
    /// For x < 0, λ = 2: y = -ln(-x + 1)
    #[inline]
    fn yeo_johnson_transform(x: f64, lambda: f64) -> f64 {
        if x >= 0.0 {
            if lambda.abs() < 1e-10 {
                x.ln_1p() // More precise than (x + 1.0).ln() for small x
            } else {
                ((x + 1.0).powf(lambda) - 1.0) / lambda
            }
        } else {
            // x < 0
            if (lambda - 2.0).abs() < 1e-10 {
                -(-x).ln_1p() // More precise than -((-x + 1.0).ln()) for small |x|
            } else {
                -((-x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda)
            }
        }
    }

    /// Apply inverse Yeo-Johnson transformation to a single value.
    #[inline]
    fn yeo_johnson_inverse(y: f64, lambda: f64) -> f64 {
        // We need to determine which branch we're in based on the transformed value
        // This is tricky because we don't know if original x was positive or negative
        // We use the fact that for the positive branch with standard lambdas,
        // the transformation is monotonic and we can use the inverse formulas

        // For simplicity, we try the positive branch first and check validity
        // Positive branch inverse: x = (y * λ + 1)^(1/λ) - 1 for λ ≠ 0
        //                          x = exp(y) - 1 for λ = 0

        if lambda.abs() < 1e-10 {
            // λ = 0 case: check if exp(y) - 1 >= 0
            let x = y.exp() - 1.0;
            if x >= 0.0 {
                return x;
            }
        } else {
            let arg = y.mul_add(lambda, 1.0);
            if arg > 0.0 {
                let x = arg.powf(1.0 / lambda) - 1.0;
                if x >= 0.0 {
                    return x;
                }
            }
        }

        // Negative branch inverse:
        // For λ ≠ 2: x = 1 - (y * (2-λ) + 1)^(1/(2-λ)) for x < 0
        // For λ = 2: x = 1 - exp(-y)
        if (lambda - 2.0).abs() < 1e-10 {
            1.0 - (-y).exp()
        } else {
            let two_minus_lambda = 2.0 - lambda;
            let arg = (-y).mul_add(two_minus_lambda, 1.0);
            if arg > 0.0 {
                1.0 - arg.powf(1.0 / two_minus_lambda)
            } else {
                // Fallback for numerical edge cases
                f64::NAN
            }
        }
    }

    /// Compute the log-likelihood for Box-Cox transformation.
    ///
    /// For MLE, we maximize:
    /// L(λ) = -n/2 * ln(σ²) + (λ - 1) * Σ ln(x_i)
    fn box_cox_log_likelihood(&self, x: &[f64], lambda: f64) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 {
            return f64::NEG_INFINITY;
        }

        // Transform data
        let transformed: Vec<f64> = x
            .iter()
            .map(|&xi| Self::box_cox_transform(xi, lambda))
            .collect();

        // Compute mean
        let mean: f64 = transformed.iter().sum::<f64>() / n;

        // Compute variance
        let var: f64 = transformed.iter().map(|&y| (y - mean).powi(2)).sum::<f64>() / n;

        if var <= 0.0 || var.is_nan() {
            return f64::NEG_INFINITY;
        }

        // Log-likelihood (ignoring constant terms)
        let log_jacobian: f64 = x.iter().map(|&xi| xi.ln()).sum();
        (-n / 2.0).mul_add(var.ln(), (lambda - 1.0) * log_jacobian)
    }

    /// Compute the log-likelihood for Yeo-Johnson transformation.
    fn yeo_johnson_log_likelihood(&self, x: &[f64], lambda: f64) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 {
            return f64::NEG_INFINITY;
        }

        // Transform data
        let transformed: Vec<f64> = x
            .iter()
            .map(|&xi| Self::yeo_johnson_transform(xi, lambda))
            .collect();

        // Compute mean
        let mean: f64 = transformed.iter().sum::<f64>() / n;

        // Compute variance
        let var: f64 = transformed.iter().map(|&y| (y - mean).powi(2)).sum::<f64>() / n;

        if var <= 0.0 || var.is_nan() {
            return f64::NEG_INFINITY;
        }

        // Jacobian for Yeo-Johnson:
        // For x >= 0: (x + 1)^(λ - 1)
        // For x < 0: (-x + 1)^(1 - λ)
        let log_jacobian: f64 = x
            .iter()
            .map(|&xi| {
                if xi >= 0.0 {
                    (lambda - 1.0) * xi.ln_1p() // More precise than (xi + 1.0).ln()
                } else {
                    (1.0 - lambda) * (-xi).ln_1p() // More precise than (-xi + 1.0).ln()
                }
            })
            .sum();

        (-n / 2.0).mul_add(var.ln(), log_jacobian)
    }

    /// Find optimal lambda using golden section search.
    ///
    /// This is more robust than grid search and faster for continuous optimization.
    fn find_optimal_lambda(&self, x: &[f64]) -> f64 {
        let log_likelihood = |lambda: f64| -> f64 {
            match self.method {
                PowerMethod::BoxCox => self.box_cox_log_likelihood(x, lambda),
                PowerMethod::YeoJohnson => self.yeo_johnson_log_likelihood(x, lambda),
            }
        };

        // Golden section search to find maximum
        // Search range [-5, 5] which covers most practical cases
        let mut a = -5.0_f64;
        let mut b = 5.0_f64;
        let golden_ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
        let tolerance = 1e-8;

        let mut c = b - golden_ratio * (b - a);
        let mut d = a + golden_ratio * (b - a);
        let mut fc = log_likelihood(c);
        let mut fd = log_likelihood(d);

        while (b - a).abs() > tolerance {
            if fc > fd {
                b = d;
                d = c;
                fd = fc;
                c = b - golden_ratio * (b - a);
                fc = log_likelihood(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + golden_ratio * (b - a);
                fd = log_likelihood(d);
            }
        }

        (a + b) / 2.0
    }

    /// Validate that data is strictly positive for Box-Cox transformation.
    fn validate_positive(x: &Array2<f64>) -> Result<()> {
        for (idx, &val) in x.iter().enumerate() {
            if val <= 0.0 {
                return Err(FerroError::invalid_input(format!(
                    "Box-Cox transformation requires strictly positive data, but found value {} at index {}",
                    val, idx
                )));
            }
        }
        Ok(())
    }

    /// Transform a column with power transformation.
    fn transform_column(&self, col: &[f64], lambda: f64) -> Vec<f64> {
        match self.method {
            PowerMethod::BoxCox => col
                .iter()
                .map(|&x| Self::box_cox_transform(x, lambda))
                .collect(),
            PowerMethod::YeoJohnson => col
                .iter()
                .map(|&x| Self::yeo_johnson_transform(x, lambda))
                .collect(),
        }
    }

    /// Inverse transform a column.
    fn inverse_transform_column(&self, col: &[f64], lambda: f64) -> Vec<f64> {
        match self.method {
            PowerMethod::BoxCox => col
                .iter()
                .map(|&y| Self::box_cox_inverse(y, lambda))
                .collect(),
            PowerMethod::YeoJohnson => col
                .iter()
                .map(|&y| Self::yeo_johnson_inverse(y, lambda))
                .collect(),
        }
    }
}

impl Default for PowerTransformer {
    fn default() -> Self {
        Self::new(PowerMethod::YeoJohnson)
    }
}

impl Transformer for PowerTransformer {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        // Validate positive data for Box-Cox
        if self.method == PowerMethod::BoxCox {
            Self::validate_positive(x)?;
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Find optimal lambda for each feature
        let mut lambdas = Array1::zeros(n_features);
        for j in 0..n_features {
            let col: Vec<f64> = x.column(j).to_vec();
            lambdas[j] = self.find_optimal_lambda(&col);
        }

        // Transform data to compute mean and std for standardization
        let mut transformed = Array2::zeros((n_samples, n_features));
        for j in 0..n_features {
            let col: Vec<f64> = x.column(j).to_vec();
            let transformed_col = self.transform_column(&col, lambdas[j]);
            for (i, &val) in transformed_col.iter().enumerate() {
                transformed[[i, j]] = val;
            }
        }

        // Compute mean and std of transformed data
        let mean = transformed
            .mean_axis(ndarray::Axis(0))
            .unwrap_or_else(|| Array1::zeros(n_features));

        let std = transformed.std_axis(ndarray::Axis(0), 0.0);

        self.lambdas = Some(lambdas);
        self.mean = Some(mean);
        self.std = Some(std);
        self.n_features_in = Some(n_features);
        self.n_samples_seen = Some(n_samples);

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        // Validate positive data for Box-Cox
        if self.method == PowerMethod::BoxCox {
            Self::validate_positive(x)?;
        }

        let lambdas = self.lambdas.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Apply power transformation
        let mut result = Array2::zeros((n_samples, n_features));
        for j in 0..n_features {
            let col: Vec<f64> = x.column(j).to_vec();
            let transformed_col = self.transform_column(&col, lambdas[j]);
            for (i, &val) in transformed_col.iter().enumerate() {
                result[[i, j]] = val;
            }
        }

        // Standardize if requested
        if self.standardize {
            let mean = self.mean.as_ref().unwrap();
            let std = self.std.as_ref().unwrap();

            for j in 0..n_features {
                let m = mean[j];
                let s = std[j];
                if s.abs() > 1e-10 {
                    for i in 0..n_samples {
                        result[[i, j]] = (result[[i, j]] - m) / s;
                    }
                } else {
                    // Zero variance feature - just center
                    for i in 0..n_samples {
                        result[[i, j]] -= m;
                    }
                }
            }
        }

        Ok(result)
    }

    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "inverse_transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let lambdas = self.lambdas.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut result = x.clone();

        // Inverse standardization if applied
        if self.standardize {
            let mean = self.mean.as_ref().unwrap();
            let std = self.std.as_ref().unwrap();

            for j in 0..n_features {
                let m = mean[j];
                let s = std[j];
                if s.abs() > 1e-10 {
                    for i in 0..n_samples {
                        result[[i, j]] = result[[i, j]].mul_add(s, m);
                    }
                } else {
                    for i in 0..n_samples {
                        result[[i, j]] += m;
                    }
                }
            }
        }

        // Apply inverse power transformation
        let mut final_result = Array2::zeros((n_samples, n_features));
        for j in 0..n_features {
            let col: Vec<f64> = result.column(j).to_vec();
            let inverse_col = self.inverse_transform_column(&col, lambdas[j]);
            for (i, &val) in inverse_col.iter().enumerate() {
                final_result[[i, j]] = val;
            }
        }

        Ok(final_result)
    }

    fn is_fitted(&self) -> bool {
        self.lambdas.is_some()
    }

    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_features = self.n_features_in?;
        Some(
            input_names
                .map(|names| names.to_vec())
                .unwrap_or_else(|| generate_feature_names(n_features)),
        )
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        self.n_features_in
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f64 = 1e-6;

    // ========== PowerMethod Tests ==========

    #[test]
    fn test_power_method_default() {
        assert_eq!(PowerMethod::default(), PowerMethod::YeoJohnson);
    }

    // ========== Box-Cox Transformation Tests ==========

    #[test]
    fn test_box_cox_lambda_zero() {
        // For λ = 0: y = ln(x)
        let y = PowerTransformer::box_cox_transform(std::f64::consts::E, 0.0);
        assert!((y - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_box_cox_lambda_one() {
        // For λ = 1: y = (x^1 - 1) / 1 = x - 1
        let y = PowerTransformer::box_cox_transform(5.0, 1.0);
        assert!((y - 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_box_cox_lambda_half() {
        // For λ = 0.5: y = (x^0.5 - 1) / 0.5 = 2 * (sqrt(x) - 1)
        let y = PowerTransformer::box_cox_transform(4.0, 0.5);
        let expected = 2.0 * (4.0_f64.sqrt() - 1.0);
        assert!((y - expected).abs() < EPSILON);
    }

    #[test]
    fn test_box_cox_inverse() {
        let x = 5.0;
        for &lambda in &[-1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let y = PowerTransformer::box_cox_transform(x, lambda);
            let x_recovered = PowerTransformer::box_cox_inverse(y, lambda);
            assert!(
                (x - x_recovered).abs() < EPSILON,
                "Failed for lambda = {}",
                lambda
            );
        }
    }

    #[test]
    fn test_box_cox_positive_only() {
        let mut transformer = PowerTransformer::new(PowerMethod::BoxCox);
        let x = array![[1.0], [0.0], [3.0]]; // Contains zero

        let result = transformer.fit(&x);
        assert!(result.is_err());

        let x = array![[1.0], [-1.0], [3.0]]; // Contains negative
        let result = transformer.fit(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_box_cox_basic() {
        let mut transformer = PowerTransformer::new(PowerMethod::BoxCox);
        let x = array![[1.0], [2.0], [4.0], [8.0], [16.0]];

        let x_transformed = transformer.fit_transform(&x).unwrap();

        // Check that transformation was applied (data changed)
        assert!((x_transformed[[0, 0]] - x[[0, 0]]).abs() > EPSILON);

        // Check that lambda was found
        let lambdas = transformer.lambdas().unwrap();
        assert!(lambdas[0].is_finite());
    }

    #[test]
    fn test_box_cox_roundtrip() {
        let mut transformer = PowerTransformer::new(PowerMethod::BoxCox).with_standardize(false);
        let x = array![[1.0], [2.0], [4.0], [8.0], [16.0]];

        transformer.fit(&x).unwrap();
        let x_transformed = transformer.transform(&x).unwrap();
        let x_recovered = transformer.inverse_transform(&x_transformed).unwrap();

        for i in 0..x.nrows() {
            assert!(
                (x[[i, 0]] - x_recovered[[i, 0]]).abs() < 0.01,
                "Failed at index {}: expected {}, got {}",
                i,
                x[[i, 0]],
                x_recovered[[i, 0]]
            );
        }
    }

    // ========== Yeo-Johnson Transformation Tests ==========

    #[test]
    fn test_yeo_johnson_positive_lambda_zero() {
        // For x >= 0, λ = 0: y = ln(x + 1)
        let y = PowerTransformer::yeo_johnson_transform(std::f64::consts::E - 1.0, 0.0);
        assert!((y - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_yeo_johnson_positive_lambda_one() {
        // For x >= 0, λ = 1: y = ((x + 1)^1 - 1) / 1 = x
        let y = PowerTransformer::yeo_johnson_transform(5.0, 1.0);
        assert!((y - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_yeo_johnson_negative_lambda_two() {
        // For x < 0, λ = 2: y = -ln(-x + 1)
        let y = PowerTransformer::yeo_johnson_transform(-std::f64::consts::E + 1.0, 2.0);
        assert!((y - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_yeo_johnson_negative() {
        // Test that Yeo-Johnson handles negative values
        let mut transformer = PowerTransformer::new(PowerMethod::YeoJohnson);
        let x = array![[-5.0], [-2.0], [0.0], [2.0], [5.0]];

        let result = transformer.fit_transform(&x);
        assert!(result.is_ok(), "Yeo-Johnson should handle negative values");
    }

    #[test]
    fn test_yeo_johnson_inverse() {
        let test_values = vec![-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0];
        let lambdas = vec![-1.0, 0.0, 0.5, 1.0, 2.0, 3.0];

        for &x in &test_values {
            for &lambda in &lambdas {
                let y = PowerTransformer::yeo_johnson_transform(x, lambda);
                let x_recovered = PowerTransformer::yeo_johnson_inverse(y, lambda);
                if x_recovered.is_finite() {
                    assert!(
                        (x - x_recovered).abs() < 0.01,
                        "Failed for x = {}, lambda = {}: got {}",
                        x,
                        lambda,
                        x_recovered
                    );
                }
            }
        }
    }

    #[test]
    fn test_yeo_johnson_basic() {
        let mut transformer = PowerTransformer::new(PowerMethod::YeoJohnson);
        let x = array![[-2.0], [-1.0], [0.0], [1.0], [2.0]];

        let x_transformed = transformer.fit_transform(&x).unwrap();

        // Check that output is standardized (mean ~ 0, std ~ 1)
        let mean: f64 = x_transformed.column(0).mean().unwrap();
        let var: f64 = x_transformed
            .column(0)
            .iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>()
            / x_transformed.nrows() as f64;
        let std = var.sqrt();

        assert!(mean.abs() < 0.1, "Mean should be near 0, got {}", mean);
        assert!((std - 1.0).abs() < 0.1, "Std should be near 1, got {}", std);
    }

    #[test]
    fn test_yeo_johnson_roundtrip() {
        let mut transformer =
            PowerTransformer::new(PowerMethod::YeoJohnson).with_standardize(false);
        let x = array![[-2.0], [-1.0], [0.0], [1.0], [2.0]];

        transformer.fit(&x).unwrap();
        let x_transformed = transformer.transform(&x).unwrap();
        let x_recovered = transformer.inverse_transform(&x_transformed).unwrap();

        for i in 0..x.nrows() {
            assert!(
                (x[[i, 0]] - x_recovered[[i, 0]]).abs() < 0.1,
                "Failed at index {}: expected {}, got {}",
                i,
                x[[i, 0]],
                x_recovered[[i, 0]]
            );
        }
    }

    // ========== Standardization Tests ==========

    #[test]
    fn test_standardize_enabled() {
        let mut transformer = PowerTransformer::new(PowerMethod::YeoJohnson).with_standardize(true);
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

        let x_transformed = transformer.fit_transform(&x).unwrap();

        let mean: f64 = x_transformed.column(0).mean().unwrap();
        assert!(mean.abs() < EPSILON, "Mean should be 0, got {}", mean);
    }

    #[test]
    fn test_standardize_disabled() {
        let mut transformer1 =
            PowerTransformer::new(PowerMethod::YeoJohnson).with_standardize(true);
        let mut transformer2 =
            PowerTransformer::new(PowerMethod::YeoJohnson).with_standardize(false);

        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

        let x1 = transformer1.fit_transform(&x).unwrap();
        let x2 = transformer2.fit_transform(&x).unwrap();

        // Results should differ due to standardization
        assert!((x1[[0, 0]] - x2[[0, 0]]).abs() > EPSILON);
    }

    // ========== Multiple Features Tests ==========

    #[test]
    fn test_multiple_features() {
        let mut transformer = PowerTransformer::new(PowerMethod::YeoJohnson);
        let x = array![
            [1.0, -2.0, 0.5],
            [2.0, -1.0, 1.0],
            [3.0, 0.0, 2.0],
            [4.0, 1.0, 4.0],
            [5.0, 2.0, 8.0]
        ];

        transformer.fit(&x).unwrap();

        // Check that we have lambdas for each feature
        let lambdas = transformer.lambdas().unwrap();
        assert_eq!(lambdas.len(), 3);

        // Transform
        let x_transformed = transformer.transform(&x).unwrap();
        assert_eq!(x_transformed.shape(), &[5, 3]);
    }

    // ========== Fitted State Tests ==========

    #[test]
    fn test_not_fitted() {
        let transformer = PowerTransformer::new(PowerMethod::YeoJohnson);
        let x = array![[1.0, 2.0]];

        assert!(transformer.transform(&x).is_err());
        assert!(!transformer.is_fitted());
    }

    #[test]
    fn test_shape_mismatch() {
        let mut transformer = PowerTransformer::new(PowerMethod::YeoJohnson);
        let x_fit = array![[1.0, 2.0], [3.0, 4.0]];
        transformer.fit(&x_fit).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
        assert!(transformer.transform(&x_wrong).is_err());
    }

    // ========== Feature Metadata Tests ==========

    #[test]
    fn test_feature_names() {
        let mut transformer = PowerTransformer::new(PowerMethod::YeoJohnson);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        transformer.fit(&x).unwrap();

        // Default names
        let names = transformer.get_feature_names_out(None).unwrap();
        assert_eq!(names, vec!["x0", "x1", "x2"]);

        // Custom names
        let custom = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let names = transformer.get_feature_names_out(Some(&custom)).unwrap();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_n_features() {
        let mut transformer = PowerTransformer::new(PowerMethod::YeoJohnson);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Before fit
        assert!(transformer.n_features_in().is_none());
        assert!(transformer.n_features_out().is_none());

        // After fit
        transformer.fit(&x).unwrap();
        assert_eq!(transformer.n_features_in(), Some(3));
        assert_eq!(transformer.n_features_out(), Some(3));
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_empty_input() {
        let empty: Array2<f64> = Array2::zeros((0, 0));
        let mut transformer = PowerTransformer::new(PowerMethod::YeoJohnson);
        assert!(transformer.fit(&empty).is_err());
    }

    #[test]
    fn test_single_sample() {
        let mut transformer = PowerTransformer::new(PowerMethod::YeoJohnson);
        let x = array![[1.0, 2.0, 3.0]]; // Single sample

        // Should still work
        transformer.fit(&x).unwrap();
        let x_transformed = transformer.transform(&x).unwrap();
        assert_eq!(x_transformed.shape(), &[1, 3]);
    }

    #[test]
    fn test_constant_feature() {
        let mut transformer = PowerTransformer::new(PowerMethod::YeoJohnson);
        let x = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]; // First column is constant

        transformer.fit(&x).unwrap();
        let x_transformed = transformer.transform(&x).unwrap();

        // Should handle constant feature gracefully
        assert!(x_transformed[[0, 0]].is_finite());
    }

    // ========== Serialization Tests ==========

    #[test]
    fn test_serialization() {
        let mut transformer = PowerTransformer::new(PowerMethod::YeoJohnson);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        transformer.fit(&x).unwrap();

        // Serialize to JSON
        let json = serde_json::to_string(&transformer).unwrap();
        assert!(!json.is_empty());

        // Deserialize
        let restored: PowerTransformer = serde_json::from_str(&json).unwrap();
        assert!(restored.is_fitted());
        assert_eq!(restored.lambdas().unwrap().len(), 2);
    }
}

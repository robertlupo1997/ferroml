//! Polynomial Feature Transformers
//!
//! This module provides transformers for generating polynomial and interaction features.
//!
//! ## Available Transformers
//!
//! - [`PolynomialFeatures`] - Generate polynomial and interaction features
//!
//! ## When to Use
//!
//! Polynomial features are useful for:
//! - Capturing non-linear relationships in linear models
//! - Creating interaction terms between features
//! - Feature engineering for regression and classification
//!
//! ## Example
//!
//! ```
//! use ferroml_core::preprocessing::polynomial::PolynomialFeatures;
//! use ferroml_core::preprocessing::Transformer;
//! use ndarray::array;
//!
//! let mut poly = PolynomialFeatures::new(2);
//! let x = array![[1.0, 2.0], [3.0, 4.0]];
//!
//! // Generates: [1, x0, x1, x0^2, x0*x1, x1^2]
//! let x_poly = poly.fit_transform(&x).unwrap();
//! assert_eq!(x_poly.ncols(), 6); // bias + 2 original + 3 degree-2 terms
//! ```

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::{check_is_fitted, check_non_empty, check_shape, generate_feature_names, Transformer};
use crate::Result;

/// Generate polynomial and interaction features.
///
/// Generates a new feature matrix consisting of all polynomial combinations
/// of the features with degree less than or equal to the specified degree.
///
/// For example, if the input has features `[a, b]` and `degree=2`, the output
/// features are `[1, a, b, a^2, a*b, b^2]` (with `include_bias=true`).
///
/// # Configuration
///
/// - `degree`: Maximum degree of polynomial features (default: 2)
/// - `interaction_only`: If true, only interaction features are produced
///   (products of distinct input features), not powers (default: false)
/// - `include_bias`: If true, include a bias column of all ones (default: true)
///
/// # Notes
///
/// - The number of output features grows combinatorially with input features and degree
/// - For n input features and degree d: output features = C(n+d, d)
/// - Consider using with caution for high-dimensional data or high degrees
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::polynomial::PolynomialFeatures;
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// // Create polynomial features of degree 2
/// let mut poly = PolynomialFeatures::new(2);
/// let x = array![[1.0, 2.0], [3.0, 4.0]];
///
/// let x_poly = poly.fit_transform(&x).unwrap();
///
/// // For degree=2, features are: [1, x0, x1, x0^2, x0*x1, x1^2]
/// assert_eq!(x_poly.ncols(), 6);
///
/// // First row: [1, 1, 2, 1, 2, 4]
/// assert!((x_poly[[0, 0]] - 1.0).abs() < 1e-10);  // bias
/// assert!((x_poly[[0, 1]] - 1.0).abs() < 1e-10);  // x0
/// assert!((x_poly[[0, 2]] - 2.0).abs() < 1e-10);  // x1
/// assert!((x_poly[[0, 3]] - 1.0).abs() < 1e-10);  // x0^2
/// assert!((x_poly[[0, 4]] - 2.0).abs() < 1e-10);  // x0*x1
/// assert!((x_poly[[0, 5]] - 4.0).abs() < 1e-10);  // x1^2
/// ```
///
/// ## Interaction Only
///
/// ```
/// use ferroml_core::preprocessing::polynomial::PolynomialFeatures;
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// // Only interaction terms, no powers
/// let mut poly = PolynomialFeatures::new(2).interaction_only(true);
/// let x = array![[1.0, 2.0, 3.0]];
///
/// let x_poly = poly.fit_transform(&x).unwrap();
///
/// // Features: [1, x0, x1, x2, x0*x1, x0*x2, x1*x2]
/// // No x0^2, x1^2, x2^2 terms
/// assert_eq!(x_poly.ncols(), 7);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialFeatures {
    /// Maximum degree of polynomial features
    degree: usize,
    /// Only produce interaction features (products of distinct features)
    interaction_only: bool,
    /// Include a bias column (all ones)
    include_bias: bool,
    /// Number of input features (learned during fit)
    n_features_in: Option<usize>,
    /// Number of output features (computed during fit)
    n_features_out: Option<usize>,
    /// Powers for each output feature: powers\[i\] = vec of (input_feature_idx, power)
    /// This encodes which input features and what powers to multiply
    powers: Option<Vec<Vec<(usize, usize)>>>,
}

impl PolynomialFeatures {
    /// Create a new PolynomialFeatures transformer with the specified degree.
    ///
    /// Default configuration:
    /// - `interaction_only`: false (include powers)
    /// - `include_bias`: true (include bias column)
    ///
    /// # Arguments
    ///
    /// * `degree` - Maximum degree of polynomial features (must be >= 1)
    ///
    /// # Panics
    ///
    /// Panics if `degree` is 0.
    pub fn new(degree: usize) -> Self {
        assert!(degree >= 1, "degree must be at least 1");
        Self {
            degree,
            interaction_only: false,
            include_bias: true,
            n_features_in: None,
            n_features_out: None,
            powers: None,
        }
    }

    /// Configure whether to only produce interaction features.
    ///
    /// When true, only products of distinct features are produced (e.g., x0*x1),
    /// not powers of individual features (e.g., x0^2).
    ///
    /// # Arguments
    ///
    /// * `interaction_only` - If true, only produce interaction features
    pub fn interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }

    /// Configure whether to include a bias column.
    ///
    /// When true, the first column of the output is all ones (the constant term).
    ///
    /// # Arguments
    ///
    /// * `include_bias` - If true, include a bias column
    pub fn include_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }

    /// Get the degree of polynomial features.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Check if interaction_only mode is enabled.
    pub fn is_interaction_only(&self) -> bool {
        self.interaction_only
    }

    /// Check if bias column is included.
    pub fn has_bias(&self) -> bool {
        self.include_bias
    }

    /// Get the powers matrix that defines the polynomial features.
    ///
    /// Each element is a vector of (feature_index, power) pairs.
    /// For example, `[(0, 2), (1, 1)]` represents x0^2 * x1.
    pub fn powers(&self) -> Option<&Vec<Vec<(usize, usize)>>> {
        self.powers.as_ref()
    }

    /// Generate all polynomial combinations for the given number of features.
    ///
    /// Returns a vector where each element describes one output feature as
    /// a vector of (input_feature_idx, power) pairs.
    ///
    /// Uses graded lexicographic ordering (grlex) to match sklearn's behavior:
    /// - First by total degree (0, 1, 2, ...)
    /// - Within each degree, combinations are ordered lexicographically
    fn generate_powers(&self, n_features: usize) -> Vec<Vec<(usize, usize)>> {
        let mut result = Vec::new();

        // Start from degree 0 (bias) or 1
        let min_degree = if self.include_bias { 0 } else { 1 };

        for total_degree in min_degree..=self.degree {
            if total_degree == 0 {
                // Bias term: no features, just constant 1
                result.push(vec![]);
            } else {
                // Generate all combinations for this total degree
                self.generate_combinations_for_degree(n_features, total_degree, &mut result);
            }
        }

        result
    }

    /// Generate all feature combinations that sum to the given total degree.
    ///
    /// Generates combinations in lexicographic order using "stars and bars"
    /// approach, where we distribute `total_degree` among `n_features` slots.
    fn generate_combinations_for_degree(
        &self,
        n_features: usize,
        total_degree: usize,
        result: &mut Vec<Vec<(usize, usize)>>,
    ) {
        // Generate combinations with replacement in lexicographic order
        // This is equivalent to distributing `total_degree` balls into `n_features` bins
        // We use a "stars and bars" approach but generate in lex order

        // For lexicographic order: we want combinations like (2,0), (1,1), (0,2) for degree 2
        // which corresponds to x0^2, x0*x1, x1^2

        // Use iterative combination generation
        self.generate_combinations_iterative(n_features, total_degree, result);
    }

    /// Generate combinations iteratively in lexicographic order.
    ///
    /// For degree d and n features, this generates all ways to choose d items
    /// (with replacement) from n features, in lexicographic order.
    fn generate_combinations_iterative(
        &self,
        n_features: usize,
        total_degree: usize,
        result: &mut Vec<Vec<(usize, usize)>>,
    ) {
        if n_features == 0 || total_degree == 0 {
            return;
        }

        // Generate all combinations with replacement of size `total_degree` from `n_features`
        // In lexicographic order, this is like choosing indices [0,0], [0,1], [0,2], [1,1], [1,2], [2,2] for n=3, d=2

        // Start with [0, 0, ..., 0] (total_degree zeros)
        let mut indices: Vec<usize> = vec![0; total_degree];

        loop {
            // Convert indices to power representation
            let combination = self.indices_to_powers(&indices, n_features);

            // Check interaction_only constraint
            if self.interaction_only {
                // All powers must be <= 1 (no repeated features)
                let all_powers_one = combination.iter().all(|(_, p)| *p <= 1);
                // Must have at least 2 distinct features for degree > 1
                let num_features = combination.len();
                if all_powers_one && (total_degree == 1 || num_features >= 2) {
                    result.push(combination);
                }
            } else {
                result.push(combination);
            }

            // Check if we're at the last combination (all indices at max)
            if indices.iter().all(|&idx| idx == n_features - 1) {
                break;
            }

            // Generate next combination (with replacement, non-decreasing)
            // Find rightmost position that can be incremented
            let mut pos = total_degree - 1;
            while indices[pos] == n_features - 1 {
                if pos == 0 {
                    break;
                }
                pos -= 1;
            }

            // Increment this position and reset all positions to the right
            let new_val = indices[pos] + 1;
            for i in pos..total_degree {
                indices[i] = new_val;
            }
        }
    }

    /// Convert a list of indices (with repetition) to a power representation.
    ///
    /// For example, [0, 0, 1] becomes [(0, 2), (1, 1)] meaning x0^2 * x1
    fn indices_to_powers(&self, indices: &[usize], n_features: usize) -> Vec<(usize, usize)> {
        let mut counts = vec![0usize; n_features];
        for &idx in indices {
            counts[idx] += 1;
        }

        counts
            .into_iter()
            .enumerate()
            .filter(|(_, count)| *count > 0)
            .collect()
    }

    /// Generate feature name for a given power combination.
    fn generate_feature_name(combination: &[(usize, usize)], input_names: &[String]) -> String {
        if combination.is_empty() {
            return "1".to_string();
        }

        let parts: Vec<String> = combination
            .iter()
            .map(|(idx, power)| {
                let name = &input_names[*idx];
                if *power == 1 {
                    name.clone()
                } else {
                    format!("{}^{}", name, power)
                }
            })
            .collect();

        parts.join(" ")
    }
}

impl Default for PolynomialFeatures {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Transformer for PolynomialFeatures {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        let n_features = x.ncols();
        let powers = self.generate_powers(n_features);
        let n_features_out = powers.len();

        self.n_features_in = Some(n_features);
        self.n_features_out = Some(n_features_out);
        self.powers = Some(powers);

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let n_samples = x.nrows();
        let powers = self.powers.as_ref().unwrap();
        let n_features_out = powers.len();

        let mut result = Array2::zeros((n_samples, n_features_out));

        for (col_idx, combination) in powers.iter().enumerate() {
            if combination.is_empty() {
                // Bias term: all ones
                result.column_mut(col_idx).fill(1.0);
            } else {
                // Compute product of features raised to their powers
                for row_idx in 0..n_samples {
                    let mut value = 1.0;
                    for &(feature_idx, power) in combination {
                        value *= x[[row_idx, feature_idx]].powi(power as i32);
                    }
                    result[[row_idx, col_idx]] = value;
                }
            }
        }

        Ok(result)
    }

    fn is_fitted(&self) -> bool {
        self.powers.is_some()
    }

    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_features_in = self.n_features_in?;
        let powers = self.powers.as_ref()?;

        let names = input_names
            .map(|n| n.to_vec())
            .unwrap_or_else(|| generate_feature_names(n_features_in));

        Some(
            powers
                .iter()
                .map(|combination| Self::generate_feature_name(combination, &names))
                .collect(),
        )
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        self.n_features_out
    }
}

// =============================================================================
// PipelineTransformer Implementation
// =============================================================================

use crate::pipeline::PipelineTransformer;

impl PipelineTransformer for PolynomialFeatures {
    fn clone_boxed(&self) -> Box<dyn PipelineTransformer> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        "PolynomialFeatures"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_polynomial_features_degree_2() {
        let mut poly = PolynomialFeatures::new(2);
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let x_poly = poly.fit_transform(&x).unwrap();

        // Features: [1, x0, x1, x0^2, x0*x1, x1^2]
        assert_eq!(x_poly.ncols(), 6);
        assert_eq!(x_poly.nrows(), 2);

        // First row: [1, 1, 2, 1, 2, 4]
        assert!((x_poly[[0, 0]] - 1.0).abs() < EPSILON); // bias
        assert!((x_poly[[0, 1]] - 1.0).abs() < EPSILON); // x0
        assert!((x_poly[[0, 2]] - 2.0).abs() < EPSILON); // x1
        assert!((x_poly[[0, 3]] - 1.0).abs() < EPSILON); // x0^2
        assert!((x_poly[[0, 4]] - 2.0).abs() < EPSILON); // x0*x1
        assert!((x_poly[[0, 5]] - 4.0).abs() < EPSILON); // x1^2

        // Second row: [1, 3, 4, 9, 12, 16]
        assert!((x_poly[[1, 0]] - 1.0).abs() < EPSILON); // bias
        assert!((x_poly[[1, 1]] - 3.0).abs() < EPSILON); // x0
        assert!((x_poly[[1, 2]] - 4.0).abs() < EPSILON); // x1
        assert!((x_poly[[1, 3]] - 9.0).abs() < EPSILON); // x0^2 = 9
        assert!((x_poly[[1, 4]] - 12.0).abs() < EPSILON); // x0*x1 = 12
        assert!((x_poly[[1, 5]] - 16.0).abs() < EPSILON); // x1^2 = 16
    }

    #[test]
    fn test_polynomial_features_degree_1() {
        let mut poly = PolynomialFeatures::new(1);
        let x = array![[1.0, 2.0, 3.0]];

        let x_poly = poly.fit_transform(&x).unwrap();

        // Features: [1, x0, x1, x2]
        assert_eq!(x_poly.ncols(), 4);
        assert!((x_poly[[0, 0]] - 1.0).abs() < EPSILON);
        assert!((x_poly[[0, 1]] - 1.0).abs() < EPSILON);
        assert!((x_poly[[0, 2]] - 2.0).abs() < EPSILON);
        assert!((x_poly[[0, 3]] - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_polynomial_features_no_bias() {
        let mut poly = PolynomialFeatures::new(2).include_bias(false);
        let x = array![[1.0, 2.0]];

        let x_poly = poly.fit_transform(&x).unwrap();

        // Features: [x0, x1, x0^2, x0*x1, x1^2] - no bias
        assert_eq!(x_poly.ncols(), 5);
        assert!((x_poly[[0, 0]] - 1.0).abs() < EPSILON); // x0
        assert!((x_poly[[0, 1]] - 2.0).abs() < EPSILON); // x1
        assert!((x_poly[[0, 2]] - 1.0).abs() < EPSILON); // x0^2
        assert!((x_poly[[0, 3]] - 2.0).abs() < EPSILON); // x0*x1
        assert!((x_poly[[0, 4]] - 4.0).abs() < EPSILON); // x1^2
    }

    #[test]
    fn test_polynomial_features_interaction_only() {
        let mut poly = PolynomialFeatures::new(2).interaction_only(true);
        let x = array![[1.0, 2.0, 3.0]];

        let x_poly = poly.fit_transform(&x).unwrap();

        // Features: [1, x0, x1, x2, x0*x1, x0*x2, x1*x2]
        // No x0^2, x1^2, x2^2 terms
        assert_eq!(x_poly.ncols(), 7);

        // Check first 4 are bias and original features
        assert!((x_poly[[0, 0]] - 1.0).abs() < EPSILON); // bias
        assert!((x_poly[[0, 1]] - 1.0).abs() < EPSILON); // x0
        assert!((x_poly[[0, 2]] - 2.0).abs() < EPSILON); // x1
        assert!((x_poly[[0, 3]] - 3.0).abs() < EPSILON); // x2

        // Check interaction terms
        assert!((x_poly[[0, 4]] - 2.0).abs() < EPSILON); // x0*x1 = 1*2
        assert!((x_poly[[0, 5]] - 3.0).abs() < EPSILON); // x0*x2 = 1*3
        assert!((x_poly[[0, 6]] - 6.0).abs() < EPSILON); // x1*x2 = 2*3
    }

    #[test]
    fn test_polynomial_features_interaction_only_no_bias() {
        let mut poly = PolynomialFeatures::new(2)
            .interaction_only(true)
            .include_bias(false);
        let x = array![[2.0, 3.0]];

        let x_poly = poly.fit_transform(&x).unwrap();

        // Features: [x0, x1, x0*x1]
        assert_eq!(x_poly.ncols(), 3);
        assert!((x_poly[[0, 0]] - 2.0).abs() < EPSILON); // x0
        assert!((x_poly[[0, 1]] - 3.0).abs() < EPSILON); // x1
        assert!((x_poly[[0, 2]] - 6.0).abs() < EPSILON); // x0*x1
    }

    #[test]
    fn test_polynomial_features_degree_3() {
        let mut poly = PolynomialFeatures::new(3).include_bias(false);
        let x = array![[2.0, 3.0]];

        let x_poly = poly.fit_transform(&x).unwrap();

        // Features: [x0, x1, x0^2, x0*x1, x1^2, x0^3, x0^2*x1, x0*x1^2, x1^3]
        assert_eq!(x_poly.ncols(), 9);

        // Verify some key values
        assert!((x_poly[[0, 0]] - 2.0).abs() < EPSILON); // x0
        assert!((x_poly[[0, 1]] - 3.0).abs() < EPSILON); // x1
        assert!((x_poly[[0, 2]] - 4.0).abs() < EPSILON); // x0^2 = 4
        assert!((x_poly[[0, 3]] - 6.0).abs() < EPSILON); // x0*x1 = 6
        assert!((x_poly[[0, 4]] - 9.0).abs() < EPSILON); // x1^2 = 9
        assert!((x_poly[[0, 5]] - 8.0).abs() < EPSILON); // x0^3 = 8
        assert!((x_poly[[0, 6]] - 12.0).abs() < EPSILON); // x0^2*x1 = 12
        assert!((x_poly[[0, 7]] - 18.0).abs() < EPSILON); // x0*x1^2 = 18
        assert!((x_poly[[0, 8]] - 27.0).abs() < EPSILON); // x1^3 = 27
    }

    #[test]
    fn test_polynomial_features_single_feature() {
        let mut poly = PolynomialFeatures::new(3);
        let x = array![[2.0], [3.0]];

        let x_poly = poly.fit_transform(&x).unwrap();

        // Features: [1, x0, x0^2, x0^3]
        assert_eq!(x_poly.ncols(), 4);

        // First row
        assert!((x_poly[[0, 0]] - 1.0).abs() < EPSILON); // bias
        assert!((x_poly[[0, 1]] - 2.0).abs() < EPSILON); // x0
        assert!((x_poly[[0, 2]] - 4.0).abs() < EPSILON); // x0^2
        assert!((x_poly[[0, 3]] - 8.0).abs() < EPSILON); // x0^3

        // Second row
        assert!((x_poly[[1, 0]] - 1.0).abs() < EPSILON); // bias
        assert!((x_poly[[1, 1]] - 3.0).abs() < EPSILON); // x0
        assert!((x_poly[[1, 2]] - 9.0).abs() < EPSILON); // x0^2
        assert!((x_poly[[1, 3]] - 27.0).abs() < EPSILON); // x0^3
    }

    #[test]
    fn test_polynomial_features_feature_names() {
        let mut poly = PolynomialFeatures::new(2);
        let x = array![[1.0, 2.0]];
        poly.fit(&x).unwrap();

        // Default names
        let names = poly.get_feature_names_out(None).unwrap();
        assert_eq!(names, vec!["1", "x0", "x1", "x0^2", "x0 x1", "x1^2"]);

        // Custom names
        let custom = vec!["a".to_string(), "b".to_string()];
        let names = poly.get_feature_names_out(Some(&custom)).unwrap();
        assert_eq!(names, vec!["1", "a", "b", "a^2", "a b", "b^2"]);
    }

    #[test]
    fn test_polynomial_features_feature_names_interaction_only() {
        let mut poly = PolynomialFeatures::new(2).interaction_only(true);
        let x = array![[1.0, 2.0, 3.0]];
        poly.fit(&x).unwrap();

        let names = poly.get_feature_names_out(None).unwrap();
        assert_eq!(
            names,
            vec!["1", "x0", "x1", "x2", "x0 x1", "x0 x2", "x1 x2"]
        );
    }

    #[test]
    fn test_polynomial_features_not_fitted() {
        let poly = PolynomialFeatures::new(2);
        let x = array![[1.0, 2.0]];

        assert!(!poly.is_fitted());
        assert!(poly.transform(&x).is_err());
        assert!(poly.get_feature_names_out(None).is_none());
    }

    #[test]
    fn test_polynomial_features_shape_mismatch() {
        let mut poly = PolynomialFeatures::new(2);
        let x_fit = array![[1.0, 2.0]];
        poly.fit(&x_fit).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
        assert!(poly.transform(&x_wrong).is_err());
    }

    #[test]
    fn test_polynomial_features_empty_input() {
        let mut poly = PolynomialFeatures::new(2);
        let empty: Array2<f64> = Array2::zeros((0, 0));

        assert!(poly.fit(&empty).is_err());
    }

    #[test]
    fn test_polynomial_features_n_features() {
        let mut poly = PolynomialFeatures::new(2);
        let x = array![[1.0, 2.0, 3.0]];

        // Before fit
        assert!(poly.n_features_in().is_none());
        assert!(poly.n_features_out().is_none());

        // After fit
        poly.fit(&x).unwrap();
        assert_eq!(poly.n_features_in(), Some(3));
        // For 3 features, degree 2 with bias: C(3+2, 2) = C(5,2) = 10
        assert_eq!(poly.n_features_out(), Some(10));
    }

    #[test]
    fn test_polynomial_features_getters() {
        let poly = PolynomialFeatures::new(3)
            .interaction_only(true)
            .include_bias(false);

        assert_eq!(poly.degree(), 3);
        assert!(poly.is_interaction_only());
        assert!(!poly.has_bias());
    }

    #[test]
    fn test_polynomial_features_transform_multiple_samples() {
        let mut poly = PolynomialFeatures::new(2);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let x_poly = poly.fit_transform(&x).unwrap();

        assert_eq!(x_poly.nrows(), 4);
        assert_eq!(x_poly.ncols(), 6);

        // Verify third row: [1, 5, 6, 25, 30, 36]
        assert!((x_poly[[2, 0]] - 1.0).abs() < EPSILON); // bias
        assert!((x_poly[[2, 1]] - 5.0).abs() < EPSILON); // x0
        assert!((x_poly[[2, 2]] - 6.0).abs() < EPSILON); // x1
        assert!((x_poly[[2, 3]] - 25.0).abs() < EPSILON); // x0^2
        assert!((x_poly[[2, 4]] - 30.0).abs() < EPSILON); // x0*x1
        assert!((x_poly[[2, 5]] - 36.0).abs() < EPSILON); // x1^2
    }

    #[test]
    fn test_polynomial_features_with_zeros() {
        let mut poly = PolynomialFeatures::new(2);
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

        let x_poly = poly.fit_transform(&x).unwrap();

        // First row: all zeros except bias
        assert!((x_poly[[0, 0]] - 1.0).abs() < EPSILON);
        for j in 1..x_poly.ncols() {
            assert!((x_poly[[0, j]] - 0.0).abs() < EPSILON);
        }
    }

    #[test]
    fn test_polynomial_features_with_negative() {
        let mut poly = PolynomialFeatures::new(2);
        let x = array![[-1.0, 2.0]];

        let x_poly = poly.fit_transform(&x).unwrap();

        // Features: [1, -1, 2, 1, -2, 4]
        assert!((x_poly[[0, 0]] - 1.0).abs() < EPSILON); // bias
        assert!((x_poly[[0, 1]] - (-1.0)).abs() < EPSILON); // x0
        assert!((x_poly[[0, 2]] - 2.0).abs() < EPSILON); // x1
        assert!((x_poly[[0, 3]] - 1.0).abs() < EPSILON); // x0^2 = 1
        assert!((x_poly[[0, 4]] - (-2.0)).abs() < EPSILON); // x0*x1 = -2
        assert!((x_poly[[0, 5]] - 4.0).abs() < EPSILON); // x1^2 = 4
    }

    #[test]
    fn test_polynomial_features_high_degree() {
        let mut poly = PolynomialFeatures::new(4).include_bias(false);
        let x = array![[2.0]];

        let x_poly = poly.fit_transform(&x).unwrap();

        // Features: [x0, x0^2, x0^3, x0^4]
        assert_eq!(x_poly.ncols(), 4);
        assert!((x_poly[[0, 0]] - 2.0).abs() < EPSILON); // x0
        assert!((x_poly[[0, 1]] - 4.0).abs() < EPSILON); // x0^2
        assert!((x_poly[[0, 2]] - 8.0).abs() < EPSILON); // x0^3
        assert!((x_poly[[0, 3]] - 16.0).abs() < EPSILON); // x0^4
    }

    #[test]
    fn test_polynomial_features_interaction_degree_3() {
        let mut poly = PolynomialFeatures::new(3)
            .interaction_only(true)
            .include_bias(false);
        let x = array![[1.0, 2.0, 3.0]];

        let x_poly = poly.fit_transform(&x).unwrap();

        // Degree 1: x0, x1, x2
        // Degree 2: x0*x1, x0*x2, x1*x2
        // Degree 3: x0*x1*x2
        assert_eq!(x_poly.ncols(), 7);

        // Check x0*x1*x2 = 1*2*3 = 6
        assert!((x_poly[[0, 6]] - 6.0).abs() < EPSILON);
    }

    #[test]
    #[should_panic(expected = "degree must be at least 1")]
    fn test_polynomial_features_degree_zero() {
        PolynomialFeatures::new(0);
    }

    #[test]
    fn test_default() {
        let poly = PolynomialFeatures::default();
        assert_eq!(poly.degree(), 2);
        assert!(!poly.is_interaction_only());
        assert!(poly.has_bias());
    }
}

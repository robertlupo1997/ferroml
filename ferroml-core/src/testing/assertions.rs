//! Assertion macros and tolerance constants for FerroML testing
//!
//! This module provides calibrated tolerance constants for different algorithm types
//! and assertion macros for float comparisons. Using these instead of hardcoded
//! tolerances ensures consistent and appropriate precision checks across the codebase.
//!
//! # Usage
//!
//! ```
//! # use ferroml_core::testing::assertions::{tolerances, assert_approx_eq, assert_array_approx_eq};
//! # use ndarray::array;
//! # let actual = 1.0_f64;
//! # let expected = 1.0_f64;
//! # let actual_array = array![1.0_f64, 2.0, 3.0];
//! # let expected_array = array![1.0_f64, 2.0, 3.0];
//! // For closed-form solutions (linear regression, QR decomposition)
//! assert_approx_eq!(actual, expected, tolerances::CLOSED_FORM);
//!
//! // For iterative algorithms (gradient descent, IRLS)
//! assert_approx_eq!(actual, expected, tolerances::ITERATIVE);
//!
//! // For array comparisons
//! assert_array_approx_eq!(actual_array, expected_array, tolerances::SKLEARN_COMPAT);
//! ```

use ndarray::{Array1, Array2};

/// Tolerance constants calibrated for different algorithm types
///
/// These values are carefully chosen based on the numerical characteristics
/// of each algorithm family and should be used consistently across tests.
pub mod tolerances {
    /// Closed-form solutions (QR decomposition, direct solve, least squares)
    ///
    /// Use for: LinearRegression, Ridge (closed form), direct matrix operations
    /// Expected precision: ~machine epsilon * condition number
    pub const CLOSED_FORM: f64 = 1e-10;

    /// Iterative algorithms (gradient descent, IRLS, coordinate descent)
    ///
    /// Use for: Lasso, ElasticNet, LogisticRegression, SGD variants
    /// Expected precision: depends on convergence tolerance (typically 1e-4)
    pub const ITERATIVE: f64 = 1e-4;

    /// Tree-based algorithms (deterministic splits)
    ///
    /// Use for: DecisionTree, RandomForest, GradientBoosting
    /// Expected precision: very high (discrete operations)
    pub const TREE: f64 = 1e-12;

    /// Probabilistic algorithms (sampling-based, MCMC)
    ///
    /// Use for: BayesianRidge, GaussianProcesses, ensemble with randomness
    /// Expected precision: lower due to stochastic nature
    pub const PROBABILISTIC: f64 = 1e-2;

    /// Neural network / deep learning
    ///
    /// Use for: MLP, deep learning models
    /// Expected precision: moderate due to floating point accumulation
    pub const NEURAL: f64 = 1e-3;

    /// sklearn comparison (accounts for implementation differences)
    ///
    /// Use for: comparing FerroML results against sklearn reference values
    /// Slightly looser to account for algorithmic differences
    pub const SKLEARN_COMPAT: f64 = 1e-5;

    /// Standard scaler and normalization operations
    ///
    /// Use for: StandardScaler, MinMaxScaler, etc.
    pub const SCALER: f64 = 1e-10;

    /// Decomposition operations (PCA, SVD)
    ///
    /// Use for: PCA, TruncatedSVD, etc.
    pub const DECOMPOSITION: f64 = 1e-8;

    /// Distance calculations (KNN, clustering)
    ///
    /// Use for: KNeighbors, KMeans, distance metrics
    pub const DISTANCE: f64 = 1e-10;

    /// Probability outputs (softmax, predict_proba)
    ///
    /// Use for: probability predictions that should sum to 1
    pub const PROBABILITY: f64 = 1e-6;

    /// Metric calculations (R2, MSE, accuracy)
    ///
    /// Use for: evaluation metric tests
    pub const METRIC: f64 = 1e-10;

    /// Cross-validation splits (should be exact)
    ///
    /// Use for: CV split indices and fold validation
    pub const CV_SPLIT: f64 = 1e-15;

    /// Loose tolerance for visual/display purposes
    ///
    /// Use for: tests that only need approximate agreement
    pub const LOOSE: f64 = 1e-2;

    /// Very strict tolerance for exact operations
    ///
    /// Use for: operations that should be bit-exact
    pub const STRICT: f64 = 1e-14;
}

/// Assert that two floating-point values are approximately equal within tolerance.
///
/// This macro provides better error messages than a simple boolean assertion,
/// showing the actual values and their difference.
///
/// # Arguments
///
/// * `$left` - The actual value
/// * `$right` - The expected value
/// * `$tol` - The tolerance (maximum allowed absolute difference)
///
/// # Example
///
/// ```
/// # use ferroml_core::testing::assertions::{assert_approx_eq, tolerances};
/// let actual = 2.0000001_f64;
/// let expected = 2.0_f64;
/// assert_approx_eq!(actual, expected, tolerances::ITERATIVE);
/// ```
///
/// # Panics
///
/// Panics if `|left - right| >= tol`, displaying the values and difference.
#[macro_export]
macro_rules! assert_approx_eq {
    ($left:expr, $right:expr, $tol:expr) => {{
        let left_val = $left;
        let right_val = $right;
        let tol_val = $tol;
        let diff = (left_val - right_val).abs();
        assert!(
            diff < tol_val,
            "assertion `left ≈ right` failed (tolerance: {:.2e})\n  left: {}\n right: {}\n  diff: {:.2e}",
            tol_val,
            left_val,
            right_val,
            diff
        );
    }};
    ($left:expr, $right:expr, $tol:expr, $($arg:tt)+) => {{
        let left_val = $left;
        let right_val = $right;
        let tol_val = $tol;
        let diff = (left_val - right_val).abs();
        assert!(
            diff < tol_val,
            "assertion `left ≈ right` failed (tolerance: {:.2e}): {}\n  left: {}\n right: {}\n  diff: {:.2e}",
            tol_val,
            format_args!($($arg)+),
            left_val,
            right_val,
            diff
        );
    }};
}

/// Assert that two arrays are approximately equal element-wise within tolerance.
///
/// This macro checks that:
/// 1. Both arrays have the same length
/// 2. Each pair of corresponding elements differs by less than the tolerance
///
/// # Arguments
///
/// * `$left` - The actual array
/// * `$right` - The expected array
/// * `$tol` - The tolerance (maximum allowed absolute difference per element)
///
/// # Example
///
/// ```
/// # use ferroml_core::testing::assertions::{assert_array_approx_eq, tolerances};
/// # use ndarray::array;
/// let actual = array![1.0_f64, 2.0, 3.0];
/// let expected = array![1.00001_f64, 2.00001, 3.00001];
/// assert_array_approx_eq!(actual, expected, tolerances::ITERATIVE);
/// ```
///
/// # Panics
///
/// Panics if arrays have different lengths or any element pair differs by >= tol.
#[macro_export]
macro_rules! assert_array_approx_eq {
    ($left:expr, $right:expr, $tol:expr) => {{
        let left_arr = &$left;
        let right_arr = &$right;
        let tol_val = $tol;
        assert_eq!(
            left_arr.len(),
            right_arr.len(),
            "assertion `left ≈ right` failed: array lengths differ\n  left length: {}\n right length: {}",
            left_arr.len(),
            right_arr.len()
        );
        for (i, (l, r)) in left_arr.iter().zip(right_arr.iter()).enumerate() {
            let diff = (l - r).abs();
            assert!(
                diff < tol_val,
                "assertion `left ≈ right` failed at index {} (tolerance: {:.2e})\n  left[{}]: {}\n right[{}]: {}\n  diff: {:.2e}",
                i, tol_val, i, l, i, r, diff
            );
        }
    }};
    ($left:expr, $right:expr, $tol:expr, $($arg:tt)+) => {{
        let left_arr = &$left;
        let right_arr = &$right;
        let tol_val = $tol;
        assert_eq!(
            left_arr.len(),
            right_arr.len(),
            "assertion `left ≈ right` failed: {}\n  array lengths differ: {} vs {}",
            format_args!($($arg)+),
            left_arr.len(),
            right_arr.len()
        );
        for (i, (l, r)) in left_arr.iter().zip(right_arr.iter()).enumerate() {
            let diff = (l - r).abs();
            assert!(
                diff < tol_val,
                "assertion `left ≈ right` failed at index {}: {} (tolerance: {:.2e})\n  left[{}]: {}\n right[{}]: {}\n  diff: {:.2e}",
                i, format_args!($($arg)+), tol_val, i, l, i, r, diff
            );
        }
    }};
}

/// Assert that two 2D arrays are approximately equal element-wise within tolerance.
///
/// # Arguments
///
/// * `$left` - The actual 2D array
/// * `$right` - The expected 2D array
/// * `$tol` - The tolerance (maximum allowed absolute difference per element)
#[macro_export]
macro_rules! assert_array2_approx_eq {
    ($left:expr, $right:expr, $tol:expr) => {{
        let left_arr = &$left;
        let right_arr = &$right;
        let tol_val = $tol;
        assert_eq!(
            left_arr.shape(),
            right_arr.shape(),
            "assertion `left ≈ right` failed: array shapes differ\n  left shape: {:?}\n right shape: {:?}",
            left_arr.shape(),
            right_arr.shape()
        );
        for ((idx, l), r) in left_arr.indexed_iter().zip(right_arr.iter()) {
            let diff = (l - r).abs();
            assert!(
                diff < tol_val,
                "assertion `left ≈ right` failed at {:?} (tolerance: {:.2e})\n  left: {}\n right: {}\n  diff: {:.2e}",
                idx, tol_val, l, r, diff
            );
        }
    }};
}

/// Check if two floats are approximately equal (returns bool instead of panicking).
///
/// Useful in conditional checks or when building up more complex assertions.
///
/// # Arguments
///
/// * `a` - First value
/// * `b` - Second value
/// * `tol` - Maximum allowed absolute difference
///
/// # Returns
///
/// `true` if `|a - b| < tol`, `false` otherwise.
#[inline]
pub fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// Check if two arrays are approximately equal (returns bool instead of panicking).
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
/// * `tol` - Maximum allowed absolute difference per element
///
/// # Returns
///
/// `true` if arrays have same length and all elements differ by less than tol.
pub fn arrays_approx_eq(a: &Array1<f64>, b: &Array1<f64>, tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() < tol || (x.is_nan() && y.is_nan()))
}

/// Check if two 2D arrays are approximately equal (returns bool instead of panicking).
pub fn arrays2_approx_eq(a: &Array2<f64>, b: &Array2<f64>, tol: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() < tol || (x.is_nan() && y.is_nan()))
}

/// Compute the maximum absolute difference between two arrays.
///
/// Useful for debugging when an assertion fails.
pub fn max_abs_diff(a: &Array1<f64>, b: &Array1<f64>) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(Some(0.0f64), |acc, diff| {
            acc.map(|a| if diff.is_finite() { a.max(diff) } else { a })
        })
}

/// Compute the root mean squared error between two arrays.
pub fn rmse(a: &Array1<f64>, b: &Array1<f64>) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let sum_sq: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    Some((sum_sq / a.len() as f64).sqrt())
}

/// Assert that a value is finite (not NaN or infinite).
#[macro_export]
macro_rules! assert_finite {
    ($val:expr) => {{
        let v: f64 = $val;
        assert!(
            v.is_finite(),
            "assertion failed: expected finite value, got {}",
            v
        );
    }};
    ($val:expr, $($arg:tt)+) => {{
        let v: f64 = $val;
        assert!(
            v.is_finite(),
            "assertion failed: expected finite value, got {}: {}",
            v,
            format_args!($($arg)+)
        );
    }};
}

/// Assert that all elements in an array are finite.
#[macro_export]
macro_rules! assert_all_finite {
    ($arr:expr) => {{
        let arr = &$arr;
        for (i, v) in arr.iter().enumerate() {
            assert!(
                v.is_finite(),
                "assertion failed: non-finite value at index {}: {}",
                i,
                v
            );
        }
    }};
}

/// Assert that a probability value is valid (between 0 and 1).
#[macro_export]
macro_rules! assert_probability {
    ($val:expr) => {{
        let v = $val;
        assert!(
            v >= 0.0 && v <= 1.0,
            "assertion failed: expected probability in [0, 1], got {}",
            v
        );
    }};
}

/// Assert that probabilities sum to 1 within tolerance.
#[macro_export]
macro_rules! assert_probabilities_sum_to_one {
    ($arr:expr) => {{
        let arr = &$arr;
        let sum: f64 = arr.iter().sum();
        let tol = $crate::testing::assertions::tolerances::PROBABILITY;
        assert!(
            (sum - 1.0).abs() < tol,
            "assertion failed: probabilities sum to {} (expected 1.0, tolerance {:.2e})",
            sum,
            tol
        );
    }};
    ($arr:expr, $tol:expr) => {{
        let arr = &$arr;
        let sum: f64 = arr.iter().sum();
        let tol = $tol;
        assert!(
            (sum - 1.0).abs() < tol,
            "assertion failed: probabilities sum to {} (expected 1.0, tolerance {:.2e})",
            sum,
            tol
        );
    }};
}

// Re-export macros at module level
pub use crate::{
    assert_all_finite, assert_approx_eq, assert_array2_approx_eq, assert_array_approx_eq,
    assert_finite, assert_probabilities_sum_to_one, assert_probability,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_approx_eq_passes() {
        assert_approx_eq!(1.0_f64, 1.0_f64 + 1e-11, tolerances::CLOSED_FORM);
    }

    #[test]
    #[should_panic(expected = "assertion `left ≈ right` failed")]
    fn test_approx_eq_fails() {
        assert_approx_eq!(1.0_f64, 2.0_f64, tolerances::CLOSED_FORM);
    }

    #[test]
    fn test_approx_eq_with_message() {
        assert_approx_eq!(
            1.0_f64,
            1.0_f64 + 1e-11,
            tolerances::CLOSED_FORM,
            "testing coefficient"
        );
    }

    #[test]
    fn test_array_approx_eq_passes() {
        let a: Array1<f64> = array![1.0, 2.0, 3.0];
        let b: Array1<f64> = array![1.0 + 1e-11, 2.0 + 1e-11, 3.0 + 1e-11];
        assert_array_approx_eq!(a, b, tolerances::CLOSED_FORM);
    }

    #[test]
    #[should_panic(expected = "array lengths differ")]
    fn test_array_approx_eq_length_mismatch() {
        let a: Array1<f64> = array![1.0, 2.0];
        let b: Array1<f64> = array![1.0, 2.0, 3.0];
        assert_array_approx_eq!(a, b, tolerances::CLOSED_FORM);
    }

    #[test]
    #[should_panic(expected = "assertion `left ≈ right` failed at index")]
    fn test_array_approx_eq_value_mismatch() {
        let a: Array1<f64> = array![1.0, 2.0, 3.0];
        let b: Array1<f64> = array![1.0, 2.0, 4.0];
        assert_array_approx_eq!(a, b, tolerances::CLOSED_FORM);
    }

    #[test]
    fn test_arrays_approx_eq_function() {
        let a: Array1<f64> = array![1.0, 2.0, 3.0];
        let b: Array1<f64> = array![1.0 + 1e-11, 2.0 + 1e-11, 3.0 + 1e-11];
        assert!(arrays_approx_eq(&a, &b, tolerances::CLOSED_FORM));
    }

    #[test]
    fn test_arrays_approx_eq_with_nan() {
        let a: Array1<f64> = array![1.0, f64::NAN, 3.0];
        let b: Array1<f64> = array![1.0, f64::NAN, 3.0];
        assert!(arrays_approx_eq(&a, &b, tolerances::CLOSED_FORM));
    }

    #[test]
    fn test_max_abs_diff() {
        let a: Array1<f64> = array![1.0, 2.0, 3.0];
        let b: Array1<f64> = array![1.1, 2.0, 3.2];
        let diff = max_abs_diff(&a, &b);
        assert!(approx_eq(diff.unwrap(), 0.2, 1e-10));
    }

    #[test]
    fn test_rmse() {
        let a: Array1<f64> = array![1.0, 2.0, 3.0];
        let b: Array1<f64> = array![1.0, 2.0, 3.0];
        let r = rmse(&a, &b);
        assert!(approx_eq(r.unwrap(), 0.0, 1e-10));
    }

    #[test]
    fn test_assert_finite() {
        assert_finite!(1.0_f64);
        assert_finite!(0.0_f64);
        assert_finite!(-1e10_f64);
    }

    #[test]
    #[should_panic(expected = "expected finite value")]
    fn test_assert_finite_nan() {
        assert_finite!(f64::NAN);
    }

    #[test]
    fn test_assert_all_finite() {
        let a: Array1<f64> = array![1.0, 2.0, 3.0];
        assert_all_finite!(a);
    }

    #[test]
    fn test_assert_probability() {
        assert_probability!(0.0_f64);
        assert_probability!(0.5_f64);
        assert_probability!(1.0_f64);
    }

    #[test]
    #[should_panic(expected = "expected probability")]
    fn test_assert_probability_fails() {
        assert_probability!(1.5_f64);
    }

    #[test]
    fn test_assert_probabilities_sum_to_one() {
        let probs: Array1<f64> = array![0.2, 0.3, 0.5];
        assert_probabilities_sum_to_one!(probs);
    }

    #[test]
    fn test_tolerance_values() {
        // Verify tolerance ordering makes sense
        assert!(tolerances::STRICT < tolerances::CLOSED_FORM);
        assert!(tolerances::CLOSED_FORM < tolerances::ITERATIVE);
        assert!(tolerances::ITERATIVE < tolerances::PROBABILISTIC);
        assert!(tolerances::PROBABILISTIC <= tolerances::LOOSE);
    }

    #[test]
    fn test_array2_approx_eq() {
        let a: Array2<f64> = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b: Array2<f64> =
            Array2::from_shape_vec((2, 2), vec![1.0 + 1e-11, 2.0, 3.0, 4.0]).unwrap();
        assert_array2_approx_eq!(a, b, tolerances::CLOSED_FORM);
    }
}

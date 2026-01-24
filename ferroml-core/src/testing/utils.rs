//! Test utilities and fixtures for FerroML testing
//!
//! This module provides data generation utilities for testing.
//! For assertion macros and tolerance constants, see [`crate::testing::assertions`].

use ndarray::{Array1, Array2};
use rand_chacha::ChaCha8Rng;

/// Generate reproducible regression data
pub fn make_regression(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    use rand::{Rng, SeedableRng};
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate random features
    let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.random_range(-10.0..10.0));

    // Generate true coefficients
    let true_coef: Vec<f64> = (0..n_features).map(|i| (i + 1) as f64 * 0.5).collect();

    // Generate targets with noise
    let y = Array1::from_shape_fn(n_samples, |i| {
        let row = x.row(i);
        let signal: f64 = row.iter().zip(true_coef.iter()).map(|(x, c)| x * c).sum();
        signal + rng.random_range(-noise..noise)
    });

    (x, y)
}

/// Generate reproducible classification data
pub fn make_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    use rand::{Rng, SeedableRng};
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let samples_per_class = n_samples / n_classes;
    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    for class in 0..n_classes {
        let center: Vec<f64> = (0..n_features).map(|f| (class * 3 + f) as f64).collect();

        for _ in 0..samples_per_class {
            for f in 0..n_features {
                x_data.push(center[f] + rng.random_range(-1.0..1.0));
            }
            y_data.push(class as f64);
        }
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
    let y = Array1::from_vec(y_data);

    (x, y)
}

/// Generate reproducible binary classification data
pub fn make_binary_classification(
    n_samples: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    make_classification(n_samples, n_features, 2, seed)
}

/// Create linearly separable data (for testing perfect classification)
pub fn make_linearly_separable(n_samples: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    use rand::{Rng, SeedableRng};
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let half = n_samples / 2;
    let mut x_data = Vec::with_capacity(n_samples * 2);
    let mut y_data = Vec::with_capacity(n_samples);

    // Class 0: centered at (0, 0)
    for _ in 0..half {
        x_data.push(rng.random_range(-2.0..0.0));
        x_data.push(rng.random_range(-2.0..0.0));
        y_data.push(0.0);
    }

    // Class 1: centered at (2, 2)
    for _ in 0..(n_samples - half) {
        x_data.push(rng.random_range(1.0..3.0));
        x_data.push(rng.random_range(1.0..3.0));
        y_data.push(1.0);
    }

    let x = Array2::from_shape_vec((n_samples, 2), x_data).unwrap();
    let y = Array1::from_vec(y_data);

    (x, y)
}

/// Perfect linear data: y = 2*x + 1 (for exact coefficient tests)
pub fn make_perfect_linear() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]); // y = 2x + 1
    (x, y)
}

/// Known sklearn reference datasets with expected values
pub mod sklearn_reference {
    use ndarray::{array, Array1, Array2};

    /// Simple 5-point regression: y ≈ 2x + 0.1
    pub mod simple_regression {
        use super::*;

        /// Feature matrix
        pub fn x() -> Array2<f64> {
            array![[1.0], [2.0], [3.0], [4.0], [5.0]]
        }

        /// Target values
        pub fn y() -> Array1<f64> {
            array![2.1, 3.9, 6.1, 7.9, 10.1]
        }

        /// sklearn LinearRegression expected coefficient
        pub const SKLEARN_COEF: f64 = 2.0;
        /// sklearn LinearRegression expected intercept
        pub const SKLEARN_INTERCEPT: f64 = 0.1;
        /// sklearn LinearRegression expected R²
        pub const SKLEARN_R2: f64 = 0.9998;
    }

    /// Binary classification dataset
    pub mod binary_classification {
        use super::*;

        /// Feature matrix
        pub fn x() -> Array2<f64> {
            array![
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 3.0],
                [4.0, 5.0],
                [5.0, 5.0],
                [1.0, 0.0],
                [2.0, 1.0],
                [3.0, 1.0],
                [4.0, 2.0],
                [5.0, 2.0],
            ]
        }

        /// Target values
        pub fn y() -> Array1<f64> {
            array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }

        /// sklearn LogisticRegression expected accuracy (perfectly separable)
        pub const SKLEARN_ACCURACY: f64 = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;
    use crate::testing::assertions::tolerances;

    #[test]
    fn test_make_regression() {
        let (x, y) = make_regression(100, 5, 0.1, 42);
        assert_eq!(x.nrows(), 100);
        assert_eq!(x.ncols(), 5);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_make_classification() {
        let (x, y) = make_classification(100, 5, 2, 42);
        assert_eq!(x.nrows(), 100);
        assert_eq!(x.ncols(), 5);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_make_perfect_linear() {
        let (x, y) = make_perfect_linear();
        assert_eq!(x.nrows(), 5);
        assert_eq!(y.len(), 5);
        // y = 2x + 1
        for i in 0..5 {
            let expected = 2.0 * x[[i, 0]] + 1.0;
            assert_approx_eq!(
                y[i],
                expected,
                tolerances::CLOSED_FORM,
                "y = 2x + 1 at index {}",
                i
            );
        }
    }
}

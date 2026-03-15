//! Naive Bayes Classifiers with Statistical Diagnostics
//!
//! This module provides Naive Bayes classifiers with comprehensive statistical
//! output - FerroML's key differentiator from sklearn.
//!
//! ## Classifiers
//!
//! - **GaussianNB**: For continuous features with Gaussian likelihood
//! - **MultinomialNB**: For discrete count features (e.g., word counts in text)
//! - **BernoulliNB**: For binary/boolean features
//! - **CategoricalNB**: For categorical features with discrete categories
//!
//! ## Features
//!
//! - **Class priors**: Automatic or user-specified prior probabilities
//! - **Incremental learning**: `partial_fit` for online/out-of-core learning
//! - **Smoothing**: Variance smoothing (Gaussian), Laplace/Lidstone smoothing (Multinomial/Bernoulli)
//! - **Feature log-probabilities**: Full probabilistic output
//!
//! ## Example - GaussianNB
//!
//! ```
//! use ferroml_core::models::naive_bayes::GaussianNB;
//! use ferroml_core::models::{Model, ProbabilisticModel};
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 6.0, 7.0, 7.0, 6.0, 8.0, 8.0
//! ]).unwrap();
//! let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
//!
//! let mut model = GaussianNB::new();
//! model.fit(&x, &y).unwrap();
//!
//! let predictions = model.predict(&x).unwrap();
//! assert_eq!(predictions.len(), 6);
//! ```
//!
//! ## Example - MultinomialNB (Text Classification)
//!
//! ```
//! use ferroml_core::models::naive_bayes::MultinomialNB;
//! use ferroml_core::models::{Model, ProbabilisticModel};
//! use ndarray::{Array1, Array2};
//!
//! // Word count features (e.g., bag-of-words)
//! let x = Array2::from_shape_vec((4, 3), vec![
//!     5.0, 1.0, 0.0,  // Document 1: 5 occurrences of word 0, etc.
//!     4.0, 2.0, 0.0,  // Document 2
//!     0.0, 1.0, 5.0,  // Document 3
//!     0.0, 0.0, 6.0,  // Document 4
//! ]).unwrap();
//! let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
//!
//! let mut model = MultinomialNB::new();
//! model.fit(&x, &y).unwrap();
//! let predictions = model.predict(&x).unwrap();
//! assert_eq!(predictions.len(), 4);
//! ```
//!
//! ## Example - BernoulliNB (Binary Features)
//!
//! ```
//! use ferroml_core::models::naive_bayes::BernoulliNB;
//! use ferroml_core::models::{Model, ProbabilisticModel};
//! use ndarray::{Array1, Array2};
//!
//! // Binary features (presence/absence)
//! let x = Array2::from_shape_vec((4, 3), vec![
//!     1.0, 1.0, 0.0,
//!     1.0, 0.0, 0.0,
//!     0.0, 1.0, 1.0,
//!     0.0, 0.0, 1.0,
//! ]).unwrap();
//! let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
//!
//! let mut model = BernoulliNB::new();
//! model.fit(&x, &y).unwrap();
//! let predictions = model.predict(&x).unwrap();
//! assert_eq!(predictions.len(), 4);
//! ```

mod bernoulli;
mod categorical;
mod gaussian;
mod multinomial;

pub use bernoulli::BernoulliNB;
pub use categorical::CategoricalNB;
pub use gaussian::GaussianNB;
pub use multinomial::MultinomialNB;

use ndarray::{Array1, Array2};

// =============================================================================
// Shared Helper Functions
// =============================================================================

/// Compute variance for each column
pub(crate) fn compute_variance(x: &Array2<f64>, mean: &Array1<f64>) -> Array1<f64> {
    let n = x.nrows() as f64;
    if n <= 1.0 {
        return Array1::zeros(x.ncols());
    }

    let n_features = x.ncols();
    let mut var = Array1::zeros(n_features);

    for j in 0..n_features {
        let sum_sq: f64 = x.column(j).iter().map(|&xi| (xi - mean[j]).powi(2)).sum();
        var[j] = sum_sq / n; // Population variance (sklearn uses n, not n-1)
    }

    var
}

/// Standard normal critical value (inverse CDF approximation)
pub(crate) fn z_critical(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let p_adj = if p > 0.5 { 1.0 - p } else { p };
    let t = (-2.0 * p_adj.ln()).sqrt();

    let c0 = 2.515_517;
    let c1 = 0.802_853;
    let c2 = 0.010_328;
    let d1 = 1.432_788;
    let d2 = 0.189_269;
    let d3 = 0.001_308;

    let z = t
        - (c2 * t).mul_add(t, c0 + c1 * t)
            / (d3 * t * t).mul_add(t, (d2 * t).mul_add(t, 1.0 + d1 * t));

    if p > 0.5 {
        z
    } else {
        -z
    }
}

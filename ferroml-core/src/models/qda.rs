//! Quadratic Discriminant Analysis (QDA)
//!
//! QDA fits per-class covariance matrices, producing quadratic decision boundaries.
//! Unlike LDA which assumes shared covariance, QDA allows each class to have its own
//! covariance structure.
//!
//! ## Decision Function
//!
//! For class k with mean μ_k and covariance Σ_k:
//!
//! δ_k(x) = log P(k) - 0.5 * log|Σ_k| - 0.5 * (x - μ_k)^T Σ_k^{-1} (x - μ_k)
//!
//! ## Example
//!
//! ```
//! use ferroml_core::models::QuadraticDiscriminantAnalysis;
//! use ferroml_core::models::Model;
//! use ndarray::array;
//!
//! let mut qda = QuadraticDiscriminantAnalysis::new();
//!
//! let x = array![
//!     [1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [1.5, 2.5],
//!     [6.0, 7.0], [7.0, 8.0], [8.0, 7.0], [7.5, 7.5]
//! ];
//! let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
//!
//! qda.fit(&x, &y).unwrap();
//! let predictions = qda.predict(&x).unwrap();
//! assert_eq!(predictions.len(), 8);
//! ```

use crate::hpo::SearchSpace;
use crate::preprocessing::{check_is_fitted, check_non_empty, check_shape};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Quadratic Discriminant Analysis classifier.
///
/// Fits per-class covariance matrices for quadratic decision boundaries.
/// Suitable when class covariances differ significantly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadraticDiscriminantAnalysis {
    // Config
    /// Regularization parameter: Sigma_k = (1 - reg_param) * Sigma_k + reg_param * I
    reg_param: f64,
    /// Class priors (None = estimated from data)
    priors: Option<Vec<f64>>,
    /// Whether to store full covariance matrices
    store_covariance: bool,
    /// Tolerance for eigenvalue truncation
    tol: f64,

    // Fitted state
    /// Per-class means (n_classes × n_features)
    means_: Option<Array2<f64>>,
    /// Per-class covariance matrices (only if store_covariance=true)
    covariances_: Option<Vec<Array2<f64>>>,
    /// Fitted class priors
    priors_: Option<Array1<f64>>,
    /// Unique sorted class labels
    classes_: Option<Vec<f64>>,
    /// Per-class eigenvectors (rotation matrices)
    rotations_: Option<Vec<Array2<f64>>>,
    /// Per-class eigenvalues (scalings along principal axes)
    scalings_: Option<Vec<Array1<f64>>>,
    /// Number of features
    n_features_in_: Option<usize>,
}

impl Default for QuadraticDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl QuadraticDiscriminantAnalysis {
    /// Create a new QDA with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            reg_param: 0.0,
            priors: None,
            store_covariance: false,
            tol: 1e-4,
            means_: None,
            covariances_: None,
            priors_: None,
            classes_: None,
            rotations_: None,
            scalings_: None,
            n_features_in_: None,
        }
    }

    /// Set regularization parameter (0 to 1).
    ///
    /// Regularizes each class covariance: Σ_k = (1 - r) * Σ_k + r * I
    #[must_use]
    pub fn with_reg_param(mut self, reg_param: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&reg_param),
            "reg_param must be in [0, 1]"
        );
        self.reg_param = reg_param;
        self
    }

    /// Set class prior probabilities (must sum to 1).
    #[must_use]
    pub fn with_priors(mut self, priors: Vec<f64>) -> Self {
        let sum: f64 = priors.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "priors must sum to 1, got {}",
            sum
        );
        self.priors = Some(priors);
        self
    }

    /// Set whether to store full covariance matrices.
    #[must_use]
    pub fn with_store_covariance(mut self, store: bool) -> Self {
        self.store_covariance = store;
        self
    }

    /// Set tolerance for eigenvalue truncation.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        assert!(tol > 0.0, "tol must be positive");
        self.tol = tol;
        self
    }

    /// Check if the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.means_.is_some()
    }

    /// Compute decision function values.
    ///
    /// Returns log-posterior-like scores for each class:
    /// δ_k(x) = log P(k) - 0.5 * log|Σ_k| - 0.5 * (x - μ_k)^T Σ_k^{-1} (x - μ_k)
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "decision_function")?;
        check_shape(x, self.n_features_in_.unwrap())?;

        let means = self.means_.as_ref().unwrap();
        let priors = self.priors_.as_ref().unwrap();
        let rotations = self.rotations_.as_ref().unwrap();
        let scalings = self.scalings_.as_ref().unwrap();

        let n_samples = x.nrows();
        let n_classes = means.nrows();

        let mut scores = Array2::zeros((n_samples, n_classes));

        for c in 0..n_classes {
            let mean_c = means.row(c);
            let rotation = &rotations[c];
            let scaling = &scalings[c];

            // log determinant = sum of log(eigenvalues)
            let log_det: f64 = scaling.iter().map(|&s| s.ln()).sum();

            // log prior
            let log_prior = priors[c].ln();

            for i in 0..n_samples {
                // Compute (x - mu) in rotated space
                let mut mahal_sq = 0.0;
                let n_components = rotation.ncols();

                for j in 0..n_components {
                    // Project (x_i - mean_c) onto j-th eigenvector
                    let mut proj = 0.0;
                    for f in 0..x.ncols() {
                        proj += (x[[i, f]] - mean_c[f]) * rotation[[f, j]];
                    }
                    // Mahalanobis: proj^2 / eigenvalue
                    mahal_sq += proj * proj / scaling[j];
                }

                scores[[i, c]] = log_prior - 0.5 * (log_det + mahal_sq);
            }
        }

        Ok(scores)
    }

    /// Predict class probabilities using softmax of decision function.
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let scores = self.decision_function(x)?;
        let n_samples = scores.nrows();
        let n_classes = scores.ncols();
        let mut proba = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let max_score = scores
                .row(i)
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0;

            for c in 0..n_classes {
                let exp_val = (scores[[i, c]] - max_score).exp();
                proba[[i, c]] = exp_val;
                sum_exp += exp_val;
            }

            for c in 0..n_classes {
                proba[[i, c]] /= sum_exp;
            }
        }

        Ok(proba)
    }

    // Accessors

    /// Get per-class means (n_classes × n_features).
    pub fn means(&self) -> Option<&Array2<f64>> {
        self.means_.as_ref()
    }

    /// Get per-class covariance matrices (only available if `store_covariance` was true).
    pub fn covariances(&self) -> Option<&Vec<Array2<f64>>> {
        self.covariances_.as_ref()
    }

    /// Get fitted class priors.
    pub fn priors_fitted(&self) -> Option<&Array1<f64>> {
        self.priors_.as_ref()
    }

    /// Get unique class labels.
    pub fn classes(&self) -> Option<&Vec<f64>> {
        self.classes_.as_ref()
    }
}

impl super::Model for QuadraticDiscriminantAnalysis {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        check_non_empty(x)?;

        let (n_samples, n_features) = x.dim();

        if y.len() != n_samples {
            return Err(FerroError::shape_mismatch(
                format!("({},)", n_samples),
                format!("({},)", y.len()),
            ));
        }

        // Find unique classes
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(FerroError::invalid_input("QDA requires at least 2 classes"));
        }

        // Compute class counts
        let mut class_counts = vec![0usize; n_classes];
        let class_to_idx: std::collections::HashMap<i64, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, &c)| (c.to_bits() as i64, i))
            .collect();

        for &yi in y.iter() {
            let idx = class_to_idx[&(yi.to_bits() as i64)];
            class_counts[idx] += 1;
        }

        // Compute priors
        let priors = if let Some(ref p) = self.priors {
            if p.len() != n_classes {
                return Err(FerroError::invalid_input(format!(
                    "priors length ({}) must match number of classes ({})",
                    p.len(),
                    n_classes
                )));
            }
            Array1::from_vec(p.clone())
        } else {
            Array1::from_iter(class_counts.iter().map(|&c| c as f64 / n_samples as f64))
        };

        // Compute per-class means
        let mut means = Array2::zeros((n_classes, n_features));
        for (k, &class) in classes.iter().enumerate() {
            let mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &yi)| if yi == class { Some(i) } else { None })
                .collect();

            for &i in &mask {
                for j in 0..n_features {
                    means[[k, j]] += x[[i, j]];
                }
            }
            let count = mask.len() as f64;
            for j in 0..n_features {
                means[[k, j]] /= count;
            }
        }

        // Compute per-class covariance matrices and their eigendecompositions
        let mut rotations = Vec::with_capacity(n_classes);
        let mut scalings = Vec::with_capacity(n_classes);
        let mut covariances = if self.store_covariance {
            Some(Vec::with_capacity(n_classes))
        } else {
            None
        };

        for (k, &class) in classes.iter().enumerate() {
            let mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &yi)| if yi == class { Some(i) } else { None })
                .collect();
            let n_k = mask.len();

            if n_k < 2 {
                return Err(FerroError::invalid_input(format!(
                    "Class {} has only {} sample(s); need at least 2 for covariance",
                    class, n_k
                )));
            }

            // Compute covariance matrix for class k
            let mut cov = Array2::<f64>::zeros((n_features, n_features));
            for &i in &mask {
                for j in 0..n_features {
                    let diff_j = x[[i, j]] - means[[k, j]];
                    for l in j..n_features {
                        let diff_l = x[[i, l]] - means[[k, l]];
                        let val = diff_j * diff_l;
                        cov[[j, l]] += val;
                        if j != l {
                            cov[[l, j]] += val;
                        }
                    }
                }
            }
            // Normalize by (n_k - 1) for unbiased estimate
            let denom = (n_k - 1) as f64;
            cov.mapv_inplace(|v| v / denom);

            // Apply regularization: Sigma = (1 - r) * Sigma + r * I
            if self.reg_param > 0.0 {
                for j in 0..n_features {
                    for l in 0..n_features {
                        if j == l {
                            cov[[j, l]] = (1.0 - self.reg_param) * cov[[j, l]] + self.reg_param;
                        } else {
                            cov[[j, l]] *= 1.0 - self.reg_param;
                        }
                    }
                }
            }

            if let Some(ref mut covs) = covariances {
                covs.push(cov.clone());
            }

            // Eigendecomposition of covariance
            let cov_mat = nalgebra::DMatrix::from_fn(n_features, n_features, |i, j| cov[[i, j]]);
            let eigen = cov_mat.symmetric_eigen();

            // Filter out near-zero eigenvalues and sort descending
            let mut eigen_pairs: Vec<(f64, Vec<f64>)> = (0..n_features)
                .map(|i| {
                    let eigval = eigen.eigenvalues[i];
                    let eigvec: Vec<f64> = (0..n_features)
                        .map(|j| eigen.eigenvectors[(j, i)])
                        .collect();
                    (eigval, eigvec)
                })
                .collect();

            eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Keep eigenvalues above tolerance
            let n_keep = eigen_pairs
                .iter()
                .filter(|(v, _)| *v > self.tol)
                .count()
                .max(1); // Keep at least 1

            let mut rotation = Array2::zeros((n_features, n_keep));
            let mut scaling = Array1::zeros(n_keep);

            for (idx, (eigval, eigvec)) in eigen_pairs.iter().take(n_keep).enumerate() {
                scaling[idx] = eigval.max(self.tol); // Floor at tol to avoid division by zero
                for f in 0..n_features {
                    rotation[[f, idx]] = eigvec[f];
                }
            }

            rotations.push(rotation);
            scalings.push(scaling);
        }

        self.means_ = Some(means);
        self.priors_ = Some(priors);
        self.classes_ = Some(classes);
        self.rotations_ = Some(rotations);
        self.scalings_ = Some(scalings);
        self.covariances_ = covariances;
        self.n_features_in_ = Some(n_features);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(self.is_fitted(), "predict")?;
        check_shape(x, self.n_features_in_.unwrap())?;

        let scores = self.decision_function(x)?;
        let classes = self.classes_.as_ref().unwrap();

        let mut predictions = Array1::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;

            for (c, &_class) in classes.iter().enumerate() {
                if scores[[i, c]] > best_score {
                    best_score = scores[[i, c]];
                    best_class = c;
                }
            }
            predictions[i] = classes[best_class];
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        QuadraticDiscriminantAnalysis::is_fitted(self)
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features_in_
    }

    fn model_name(&self) -> &str {
        "QuadraticDiscriminantAnalysis"
    }

    fn try_predict_proba(&self, x: &Array2<f64>) -> Option<Result<Array2<f64>>> {
        Some(self.predict_proba(x))
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new().float("reg_param", 0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Model;
    use ndarray::{array, Array2};

    fn make_two_class_data() -> (Array2<f64>, Array1<f64>) {
        // Two well-separated classes
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [1.5, 2.5],
            [2.0, 2.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 7.0],
            [7.5, 7.5],
            [6.5, 8.0],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        (x, y)
    }

    #[test]
    fn test_basic_fit_predict() {
        let (x, y) = make_two_class_data();
        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).unwrap();
        let preds = qda.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
        // Should classify training data correctly
        for i in 0..5 {
            assert_eq!(preds[i], 0.0, "sample {} should be class 0", i);
        }
        for i in 5..10 {
            assert_eq!(preds[i], 1.0, "sample {} should be class 1", i);
        }
    }

    #[test]
    fn test_predict_proba_sums_to_one() {
        let (x, y) = make_two_class_data();
        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).unwrap();
        let proba = qda.predict_proba(&x).unwrap();
        assert_eq!(proba.dim(), (10, 2));
        for i in 0..10 {
            let sum: f64 = proba.row(i).sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "row {} probabilities sum to {}, not 1.0",
                i,
                sum
            );
        }
    }

    #[test]
    fn test_predict_proba_in_range() {
        let (x, y) = make_two_class_data();
        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).unwrap();
        let proba = qda.predict_proba(&x).unwrap();
        for &p in proba.iter() {
            assert!(p >= 0.0 && p <= 1.0, "probability {} out of [0, 1]", p);
        }
    }

    #[test]
    fn test_decision_function_shape() {
        let (x, y) = make_two_class_data();
        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).unwrap();
        let scores = qda.decision_function(&x).unwrap();
        assert_eq!(scores.dim(), (10, 2));
    }

    #[test]
    fn test_multiclass() {
        let x = array![
            [1.0, 0.0],
            [1.5, 0.5],
            [2.0, 0.0],
            [5.0, 5.0],
            [5.5, 5.5],
            [6.0, 5.0],
            [0.0, 5.0],
            [0.5, 5.5],
            [0.0, 6.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let mut qda = QuadraticDiscriminantAnalysis::new().with_reg_param(0.5);
        qda.fit(&x, &y).unwrap();
        let preds = qda.predict(&x).unwrap();
        let scores = qda.decision_function(&x).unwrap();
        assert_eq!(scores.ncols(), 3);
        // Should classify most correctly
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(&p, &t)| (p - t).abs() < 1e-10)
            .count();
        assert!(correct >= 7, "expected >= 7 correct, got {}", correct);
    }

    #[test]
    fn test_reg_param_prevents_singular() {
        // Very few samples per class — regularization should prevent failure
        let x = array![[1.0, 2.0], [1.1, 2.1], [5.0, 6.0], [5.1, 6.1]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mut qda = QuadraticDiscriminantAnalysis::new().with_reg_param(0.5);
        assert!(qda.fit(&x, &y).is_ok());
        let preds = qda.predict(&x).unwrap();
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[3], 1.0);
    }

    #[test]
    fn test_custom_priors() {
        // Use overlapping data so priors have visible effect
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [3.5, 3.5],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [4.5, 4.5],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let proba_default = {
            let mut q = QuadraticDiscriminantAnalysis::new().with_reg_param(0.3);
            q.fit(&x, &y).unwrap();
            q.predict_proba(&x).unwrap()
        };
        // Heavily favor class 1
        let mut qda = QuadraticDiscriminantAnalysis::new()
            .with_reg_param(0.3)
            .with_priors(vec![0.01, 0.99]);
        qda.fit(&x, &y).unwrap();
        let proba = qda.predict_proba(&x).unwrap();
        // Class 1 probability should be higher with the biased prior for at least some samples
        let mut any_higher = false;
        for i in 0..8 {
            if proba[[i, 1]] > proba_default[[i, 1]] + 1e-10 {
                any_higher = true;
            }
        }
        assert!(
            any_higher,
            "biased priors should increase class 1 probability"
        );
    }

    #[test]
    fn test_single_feature() {
        let x = array![[1.0], [2.0], [1.5], [8.0], [9.0], [8.5]];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).unwrap();
        let preds = qda.predict(&x).unwrap();
        for i in 0..3 {
            assert_eq!(preds[i], 0.0);
        }
        for i in 3..6 {
            assert_eq!(preds[i], 1.0);
        }
    }

    #[test]
    fn test_unbalanced_classes() {
        let x = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [2.0, 2.0],
            [2.5, 2.5],
            [3.0, 3.0],
            [3.5, 3.5],
            [4.0, 4.0],
            [10.0, 10.0],
            [10.5, 10.5],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

        let mut qda = QuadraticDiscriminantAnalysis::new().with_reg_param(0.3);
        qda.fit(&x, &y).unwrap();
        let preds = qda.predict(&x).unwrap();
        // The two class-1 samples should be predicted as 1
        assert_eq!(preds[7], 1.0);
        assert_eq!(preds[8], 1.0);
    }

    #[test]
    fn test_not_fitted_error() {
        let qda = QuadraticDiscriminantAnalysis::new();
        let x = array![[1.0, 2.0]];
        assert!(qda.predict(&x).is_err());
        assert!(qda.predict_proba(&x).is_err());
        assert!(qda.decision_function(&x).is_err());
    }

    #[test]
    fn test_wrong_n_features() {
        let (x, y) = make_two_class_data();
        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).unwrap();
        let x_wrong = array![[1.0, 2.0, 3.0]];
        assert!(qda.predict(&x_wrong).is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 0.0];
        let mut qda = QuadraticDiscriminantAnalysis::new();
        assert!(qda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_store_covariance() {
        let (x, y) = make_two_class_data();
        let mut qda = QuadraticDiscriminantAnalysis::new().with_store_covariance(true);
        qda.fit(&x, &y).unwrap();
        let covs = qda.covariances().unwrap();
        assert_eq!(covs.len(), 2);
        // Each covariance is (n_features × n_features)
        assert_eq!(covs[0].dim(), (2, 2));
        assert_eq!(covs[1].dim(), (2, 2));
        // Covariances should be symmetric
        assert!((covs[0][[0, 1]] - covs[0][[1, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_is_fitted() {
        let mut qda = QuadraticDiscriminantAnalysis::new();
        assert!(!qda.is_fitted());
        let (x, y) = make_two_class_data();
        qda.fit(&x, &y).unwrap();
        assert!(qda.is_fitted());
    }

    #[test]
    fn test_n_features() {
        let (x, y) = make_two_class_data();
        let mut qda = QuadraticDiscriminantAnalysis::new();
        assert_eq!(qda.n_features(), None);
        qda.fit(&x, &y).unwrap();
        assert_eq!(qda.n_features(), Some(2));
    }

    #[test]
    fn test_reproducibility() {
        let (x, y) = make_two_class_data();
        let mut qda1 = QuadraticDiscriminantAnalysis::new();
        let mut qda2 = QuadraticDiscriminantAnalysis::new();
        qda1.fit(&x, &y).unwrap();
        qda2.fit(&x, &y).unwrap();
        let p1 = qda1.predict_proba(&x).unwrap();
        let p2 = qda2.predict_proba(&x).unwrap();
        for i in 0..p1.len() {
            assert!(
                (p1.iter().nth(i).unwrap() - p2.iter().nth(i).unwrap()).abs() < 1e-10,
                "predictions differ at index {}",
                i
            );
        }
    }

    #[test]
    fn test_classes_accessor() {
        let (x, y) = make_two_class_data();
        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).unwrap();
        let classes = qda.classes().unwrap();
        assert_eq!(classes, &[0.0, 1.0]);
    }

    #[test]
    fn test_means_accessor() {
        let (x, y) = make_two_class_data();
        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).unwrap();
        let means = qda.means().unwrap();
        assert_eq!(means.dim(), (2, 2));
        // Class 0 mean should be around (1.9, 2.3)
        assert!((means[[0, 0]] - 1.9).abs() < 0.1);
        // Class 1 mean should be around (7.0, 7.1)
        assert!((means[[1, 0]] - 7.0).abs() < 0.1);
    }

    #[test]
    fn test_qda_different_covariances() {
        // QDA should handle classes with very different covariance structures
        // Class 0: elongated along x-axis; Class 1: elongated along y-axis
        let x = array![
            [0.0, 0.0],
            [1.0, 0.1],
            [2.0, -0.1],
            [3.0, 0.2],
            [-1.0, 0.0],
            [10.0, 0.0],
            [10.1, 1.0],
            [9.9, 2.0],
            [10.0, 3.0],
            [10.2, -1.0],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).unwrap();
        let preds = qda.predict(&x).unwrap();
        for i in 0..5 {
            assert_eq!(preds[i], 0.0, "sample {} misclassified", i);
        }
        for i in 5..10 {
            assert_eq!(preds[i], 1.0, "sample {} misclassified", i);
        }
    }

    #[test]
    fn test_model_name() {
        let qda = QuadraticDiscriminantAnalysis::new();
        assert_eq!(qda.model_name(), "QuadraticDiscriminantAnalysis");
    }

    #[test]
    fn test_xy_length_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 2.0]; // wrong length
        let mut qda = QuadraticDiscriminantAnalysis::new();
        assert!(qda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_try_predict_proba() {
        let (x, y) = make_two_class_data();
        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).unwrap();
        let result = qda.try_predict_proba(&x);
        assert!(result.is_some());
        let proba = result.unwrap().unwrap();
        assert_eq!(proba.dim(), (10, 2));
    }

    #[test]
    fn test_matches_lda_when_covariances_equal() {
        // When both classes have equal covariance, QDA and LDA should agree
        use crate::decomposition::LDA;

        // Generate data where both classes have the same covariance
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [2.0, 2.0],
            [1.5, 3.5],
            [11.0, 12.0],
            [12.0, 13.0],
            [13.0, 14.0],
            [12.0, 12.0],
            [11.5, 13.5],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).unwrap();
        let qda_preds = qda.predict(&x).unwrap();

        let mut lda = LDA::new();
        lda.fit(&x, &y).unwrap();
        let lda_preds = lda.predict(&x).unwrap();

        // Both should classify identically on this well-separated, equal-covariance data
        for i in 0..10 {
            assert_eq!(
                qda_preds[i], lda_preds[i],
                "QDA and LDA disagree at sample {} (QDA={}, LDA={})",
                i, qda_preds[i], lda_preds[i]
            );
        }
    }

    #[test]
    fn test_outperforms_lda_different_covariances() {
        // When class covariances differ, QDA should do better than LDA
        use crate::decomposition::LDA;

        // Class 0: tight cluster; Class 1: spread along diagonal
        let x = array![
            // Class 0: tight cluster around (0, 0)
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.1],
            [0.1, -0.1],
            [-0.1, -0.1],
            [0.05, 0.0],
            // Class 1: spread cluster around (3, 3), elongated
            [1.0, 1.0],
            [5.0, 5.0],
            [2.0, 4.0],
            [4.0, 2.0],
            [3.0, 3.0],
            [2.5, 3.5],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut qda = QuadraticDiscriminantAnalysis::new();
        qda.fit(&x, &y).unwrap();
        let qda_preds = qda.predict(&x).unwrap();
        let qda_correct: usize = qda_preds
            .iter()
            .zip(y.iter())
            .filter(|(&p, &t)| (p - t).abs() < 1e-10)
            .count();

        let mut lda = LDA::new();
        lda.fit(&x, &y).unwrap();
        let lda_preds = lda.predict(&x).unwrap();
        let lda_correct: usize = lda_preds
            .iter()
            .zip(y.iter())
            .filter(|(&p, &t)| (p - t).abs() < 1e-10)
            .count();

        // QDA should get at least as many correct as LDA on training data
        assert!(
            qda_correct >= lda_correct,
            "QDA ({} correct) should be >= LDA ({} correct) with different covariances",
            qda_correct,
            lda_correct,
        );
    }
}

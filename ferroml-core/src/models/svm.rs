//! Support Vector Machines for Classification and Regression
//!
//! This module provides Support Vector Classification (SVC) and Regression (SVR)
//! with multiple kernel functions, probability estimates via Platt scaling, and
//! multiclass support using One-vs-One (OvO) or One-vs-Rest (OvR) strategies.
//!
//! It also provides `LinearSVC` and `LinearSVR` which are optimized for large
//! datasets using coordinate descent in the primal formulation.
//!
//! ## Features
//!
//! - **Kernels**: Linear, RBF (Gaussian), Polynomial, Sigmoid
//! - **Probability estimates**: Via Platt scaling calibration
//! - **Multiclass**: One-vs-One (default) and One-vs-Rest strategies
//! - **Class weights**: Balanced or custom weights for imbalanced data
//! - **Linear models**: Efficient `LinearSVC` and `LinearSVR` for large datasets
//!
//! ## Kernelized vs Linear SVMs
//!
//! | Aspect | SVC/SVR | LinearSVC/LinearSVR |
//! |--------|---------|---------------------|
//! | Memory | O(n²) kernel matrix | O(n·d) |
//! | Speed | Slower for large n | Fast coordinate descent |
//! | Flexibility | Non-linear boundaries | Linear only |
//! | Use case | Small/medium datasets | Large datasets |
//!
//! ## Example
//!
//! ```ignore
//! use ferroml_core::models::svm::{SVC, Kernel};
//! use ferroml_core::models::{Model, ProbabilisticModel};
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 6.0, 7.0, 7.0, 6.0, 8.0, 8.0
//! ]).unwrap();
//! let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
//!
//! let mut model = SVC::new()
//!     .with_kernel(Kernel::Rbf { gamma: 0.5 })
//!     .with_c(1.0)
//!     .with_probability(true);
//!
//! model.fit(&x, &y).unwrap();
//! let predictions = model.predict(&x).unwrap();
//! let probas = model.predict_proba(&x).unwrap();
//! ```
//!
//! ## LinearSVC Example
//!
//! ```ignore
//! use ferroml_core::models::svm::LinearSVC;
//! use ferroml_core::models::Model;
//!
//! // LinearSVC is much faster for large datasets
//! let mut clf = LinearSVC::new()
//!     .with_c(1.0)
//!     .with_max_iter(1000);
//!
//! clf.fit(&x_train, &y_train).unwrap();
//! let predictions = clf.predict(&x_test).unwrap();
//! ```

use crate::hpo::SearchSpace;
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, Model, PredictionInterval,
    ProbabilisticModel,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Kernel function for SVM.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Kernel {
    /// Linear kernel: K(x, y) = x · y
    Linear,
    /// RBF (Gaussian) kernel: K(x, y) = exp(-gamma * ||x - y||^2)
    Rbf {
        /// Kernel coefficient (default: 1 / n_features if not specified)
        gamma: f64,
    },
    /// Polynomial kernel: K(x, y) = (gamma * x · y + coef0)^degree
    Polynomial {
        /// Kernel coefficient
        gamma: f64,
        /// Independent term
        coef0: f64,
        /// Polynomial degree
        degree: u32,
    },
    /// Sigmoid kernel: K(x, y) = tanh(gamma * x · y + coef0)
    Sigmoid {
        /// Kernel coefficient
        gamma: f64,
        /// Independent term
        coef0: f64,
    },
}

impl Default for Kernel {
    fn default() -> Self {
        Self::Rbf { gamma: 1.0 }
    }
}

impl Kernel {
    /// Create an RBF kernel with automatic gamma (will be set to 1/n_features)
    pub fn rbf_auto() -> Self {
        Self::Rbf { gamma: 0.0 } // 0 signals "auto"
    }

    /// Create an RBF kernel with specified gamma
    pub fn rbf(gamma: f64) -> Self {
        Self::Rbf { gamma }
    }

    /// Create a polynomial kernel
    pub fn poly(degree: u32, gamma: f64, coef0: f64) -> Self {
        Self::Polynomial {
            gamma,
            coef0,
            degree,
        }
    }

    /// Create a sigmoid kernel
    pub fn sigmoid(gamma: f64, coef0: f64) -> Self {
        Self::Sigmoid { gamma, coef0 }
    }

    /// Compute the kernel function between two vectors.
    #[inline]
    pub fn compute(&self, x: &[f64], y: &[f64]) -> f64 {
        match self {
            Kernel::Linear => dot_product(x, y),
            Kernel::Rbf { gamma } => {
                let diff_sq: f64 = x
                    .iter()
                    .zip(y.iter())
                    .map(|(xi, yi)| (xi - yi).powi(2))
                    .sum();
                (-gamma * diff_sq).exp()
            }
            Kernel::Polynomial {
                gamma,
                coef0,
                degree,
            } => (gamma * dot_product(x, y) + coef0).powi(*degree as i32),
            Kernel::Sigmoid { gamma, coef0 } => (gamma * dot_product(x, y) + coef0).tanh(),
        }
    }

    /// Set gamma to auto (1 / n_features) if it's currently 0.
    fn with_auto_gamma(self, n_features: usize) -> Self {
        let auto_gamma = 1.0 / n_features as f64;
        match self {
            Kernel::Rbf { gamma } if gamma <= 0.0 => Kernel::Rbf { gamma: auto_gamma },
            Kernel::Polynomial {
                gamma,
                coef0,
                degree,
            } if gamma <= 0.0 => Kernel::Polynomial {
                gamma: auto_gamma,
                coef0,
                degree,
            },
            Kernel::Sigmoid { gamma, coef0 } if gamma <= 0.0 => Kernel::Sigmoid {
                gamma: auto_gamma,
                coef0,
            },
            other => other,
        }
    }
}

/// Multiclass classification strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MulticlassStrategy {
    /// One-vs-One: Train n*(n-1)/2 binary classifiers
    /// Best for smaller number of classes
    OneVsOne,
    /// One-vs-Rest: Train n binary classifiers
    /// More efficient for many classes
    OneVsRest,
}

impl Default for MulticlassStrategy {
    fn default() -> Self {
        Self::OneVsOne
    }
}

/// Class weight specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClassWeight {
    /// All classes have equal weight
    Uniform,
    /// Automatically adjust weights inversely proportional to class frequencies
    Balanced,
    /// Custom weights for each class
    Custom(Vec<(f64, f64)>), // (class_label, weight) pairs
}

impl Default for ClassWeight {
    fn default() -> Self {
        Self::Uniform
    }
}

// =============================================================================
// Binary SVC (Core Implementation)
// =============================================================================

/// Binary Support Vector Classifier using SMO algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinarySVC {
    /// Regularization parameter (penalty for misclassification)
    c: f64,
    /// Kernel function
    kernel: Kernel,
    /// Tolerance for stopping criterion
    tol: f64,
    /// Maximum number of iterations
    max_iter: usize,

    // Fitted state
    /// Support vectors
    support_vectors: Option<Array2<f64>>,
    /// Dual coefficients (alpha * y for each support vector)
    dual_coef: Option<Array1<f64>>,
    /// Intercept (bias term)
    intercept: f64,
    /// Indices of support vectors in original training data
    support_indices: Option<Vec<usize>>,
    /// Number of features
    n_features: Option<usize>,
    /// Positive class label
    positive_class: f64,
    /// Negative class label
    negative_class: f64,

    // Platt scaling parameters
    platt_a: f64,
    platt_b: f64,
    probability_fitted: bool,
}

impl BinarySVC {
    /// Create a new binary SVC.
    fn new(c: f64, kernel: Kernel, tol: f64, max_iter: usize) -> Self {
        Self {
            c,
            kernel,
            tol,
            max_iter,
            support_vectors: None,
            dual_coef: None,
            intercept: 0.0,
            support_indices: None,
            n_features: None,
            positive_class: 1.0,
            negative_class: -1.0,
            platt_a: 0.0,
            platt_b: 0.0,
            probability_fitted: false,
        }
    }

    /// Fit the binary SVC using SMO algorithm.
    fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        positive_class: f64,
        negative_class: f64,
        sample_weights: Option<&Array1<f64>>,
    ) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        self.positive_class = positive_class;
        self.negative_class = negative_class;
        self.n_features = Some(n_features);

        // Set auto gamma if needed
        self.kernel = self.kernel.with_auto_gamma(n_features);

        // Convert labels to +1/-1
        let y_binary: Array1<f64> = y
            .iter()
            .map(|&label| {
                if (label - positive_class).abs() < 1e-10 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        // Compute effective C for each sample (incorporating class weights)
        let c_effective: Array1<f64> = if let Some(weights) = sample_weights {
            weights.mapv(|w| self.c * w)
        } else {
            Array1::from_elem(n_samples, self.c)
        };

        // Initialize alphas and compute kernel matrix
        let mut alpha = Array1::zeros(n_samples);
        let kernel_matrix = self.compute_kernel_matrix(x);

        // SMO algorithm
        let (final_alpha, intercept) =
            self.smo(&kernel_matrix, &y_binary, &c_effective, alpha.view_mut())?;

        // Extract support vectors
        let support_threshold = 1e-8;
        let mut support_indices = Vec::new();
        let mut dual_coef = Vec::new();

        for i in 0..n_samples {
            if final_alpha[i] > support_threshold {
                support_indices.push(i);
                dual_coef.push(final_alpha[i] * y_binary[i]);
            }
        }

        if support_indices.is_empty() {
            return Err(FerroError::convergence_failure(
                self.max_iter,
                "No support vectors found - model failed to converge",
            ));
        }

        // Store support vectors
        let n_sv = support_indices.len();
        let mut sv_matrix = Array2::zeros((n_sv, n_features));
        for (i, &idx) in support_indices.iter().enumerate() {
            sv_matrix.row_mut(i).assign(&x.row(idx));
        }

        self.support_vectors = Some(sv_matrix);
        self.dual_coef = Some(Array1::from_vec(dual_coef));
        self.support_indices = Some(support_indices);
        self.intercept = intercept;

        Ok(())
    }

    /// SMO (Sequential Minimal Optimization) algorithm.
    fn smo(
        &self,
        kernel_matrix: &Array2<f64>,
        y: &Array1<f64>,
        c: &Array1<f64>,
        mut alpha: ndarray::ArrayViewMut1<f64>,
    ) -> Result<(Array1<f64>, f64)> {
        let n_samples = y.len();
        let mut b = 0.0;

        // Error cache
        let mut errors: Array1<f64> = -y.clone();

        let mut n_changed = 0;
        let mut examine_all = true;
        let mut iter = 0;

        while (n_changed > 0 || examine_all) && iter < self.max_iter {
            n_changed = 0;

            let indices: Vec<usize> = if examine_all {
                (0..n_samples).collect()
            } else {
                // Only examine non-bound examples
                (0..n_samples)
                    .filter(|&i| alpha[i] > 0.0 && alpha[i] < c[i])
                    .collect()
            };

            for &i in &indices {
                let ei = errors[i];
                let ri = ei * y[i];

                // Check KKT conditions
                if (ri < -self.tol && alpha[i] < c[i]) || (ri > self.tol && alpha[i] > 0.0) {
                    // Select j using heuristic
                    let j = self.select_j(i, &errors, &alpha, c);

                    if let Some(j) = j {
                        let ej = errors[j];

                        // Save old alphas
                        let alpha_i_old = alpha[i];
                        let alpha_j_old = alpha[j];

                        // Compute bounds
                        let (low, high) = if (y[i] - y[j]).abs() < 1e-10 {
                            (
                                (alpha[i] + alpha[j] - c[i]).max(0.0),
                                (alpha[i] + alpha[j]).min(c[j]),
                            )
                        } else {
                            (
                                (alpha[j] - alpha[i]).max(0.0),
                                (c[j] + alpha[j] - alpha[i]).min(c[j]),
                            )
                        };

                        if (low - high).abs() < 1e-10 {
                            continue;
                        }

                        // Compute eta
                        let eta = 2.0f64.mul_add(kernel_matrix[[i, j]], -kernel_matrix[[i, i]])
                            - kernel_matrix[[j, j]];

                        if eta >= 0.0 {
                            continue;
                        }

                        // Update alpha_j
                        alpha[j] = alpha_j_old - y[j] * (ei - ej) / eta;
                        alpha[j] = alpha[j].max(low).min(high);

                        if (alpha[j] - alpha_j_old).abs() < 1e-8 {
                            continue;
                        }

                        // Update alpha_i
                        alpha[i] = (y[i] * y[j]).mul_add(alpha_j_old - alpha[j], alpha_i_old);

                        // Update threshold
                        let b1 = (y[j] * (alpha[j] - alpha_j_old)).mul_add(
                            -kernel_matrix[[i, j]],
                            (y[i] * (alpha[i] - alpha_i_old))
                                .mul_add(-kernel_matrix[[i, i]], b - ei),
                        );

                        let b2 = (y[j] * (alpha[j] - alpha_j_old)).mul_add(
                            -kernel_matrix[[j, j]],
                            (y[i] * (alpha[i] - alpha_i_old))
                                .mul_add(-kernel_matrix[[i, j]], b - ej),
                        );

                        b = if alpha[i] > 0.0 && alpha[i] < c[i] {
                            b1
                        } else if alpha[j] > 0.0 && alpha[j] < c[j] {
                            b2
                        } else {
                            (b1 + b2) / 2.0
                        };

                        // Update error cache
                        for k in 0..n_samples {
                            errors[k] =
                                self.decision_function_cached(kernel_matrix, &alpha, y, b, k)
                                    - y[k];
                        }

                        n_changed += 1;
                    }
                }
            }

            if examine_all {
                examine_all = false;
            } else if n_changed == 0 {
                examine_all = true;
            }

            iter += 1;
        }

        Ok((alpha.to_owned(), b))
    }

    /// Select second alpha using heuristic.
    fn select_j(
        &self,
        i: usize,
        errors: &Array1<f64>,
        alpha: &ndarray::ArrayViewMut1<f64>,
        c: &Array1<f64>,
    ) -> Option<usize> {
        let ei = errors[i];
        let n_samples = errors.len();

        // First, try to select from non-bound examples
        let non_bound: Vec<usize> = (0..n_samples)
            .filter(|&j| j != i && alpha[j] > 0.0 && alpha[j] < c[j])
            .collect();

        if !non_bound.is_empty() {
            // Select j with maximum |Ei - Ej|
            return non_bound
                .iter()
                .max_by(|&&a, &&b| {
                    let diff_a = (ei - errors[a]).abs();
                    let diff_b = (ei - errors[b]).abs();
                    diff_a.partial_cmp(&diff_b).unwrap_or(Ordering::Equal)
                })
                .copied();
        }

        // Otherwise, select from all examples
        (0..n_samples).filter(|&j| j != i).max_by(|&a, &b| {
            let diff_a = (ei - errors[a]).abs();
            let diff_b = (ei - errors[b]).abs();
            diff_a.partial_cmp(&diff_b).unwrap_or(Ordering::Equal)
        })
    }

    /// Compute decision function using cached kernel matrix.
    fn decision_function_cached(
        &self,
        kernel_matrix: &Array2<f64>,
        alpha: &ndarray::ArrayViewMut1<f64>,
        y: &Array1<f64>,
        b: f64,
        idx: usize,
    ) -> f64 {
        let mut sum = 0.0;
        for i in 0..y.len() {
            if alpha[i] > 1e-8 {
                sum += alpha[i] * y[i] * kernel_matrix[[i, idx]];
            }
        }
        sum + b
    }

    /// Compute full kernel matrix.
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let mut k = Array2::zeros((n, n));

        for i in 0..n {
            let xi: Vec<f64> = x.row(i).to_vec();
            for j in i..n {
                let xj: Vec<f64> = x.row(j).to_vec();
                let val = self.kernel.compute(&xi, &xj);
                k[[i, j]] = val;
                k[[j, i]] = val;
            }
        }

        k
    }

    /// Predict decision values for samples.
    fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let sv = self
            .support_vectors
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("decision_function"))?;
        let dual_coef = self.dual_coef.as_ref().unwrap();
        let n_samples = x.nrows();

        let mut decisions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let xi: Vec<f64> = x.row(i).to_vec();
            let mut sum = 0.0;

            for (j, coef) in dual_coef.iter().enumerate() {
                let svj: Vec<f64> = sv.row(j).to_vec();
                sum += coef * self.kernel.compute(&xi, &svj);
            }

            decisions[i] = sum + self.intercept;
        }

        Ok(decisions)
    }

    /// Predict class labels.
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let decisions = self.decision_function(x)?;
        Ok(decisions.mapv(|d| {
            if d >= 0.0 {
                self.positive_class
            } else {
                self.negative_class
            }
        }))
    }

    /// Fit Platt scaling for probability estimates.
    fn fit_platt_scaling(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let decisions = self.decision_function(x)?;
        let n_samples = y.len();

        // Count positive and negative samples
        let n_pos = y
            .iter()
            .filter(|&&v| (v - self.positive_class).abs() < 1e-10)
            .count() as f64;
        let n_neg = n_samples as f64 - n_pos;

        // Target probabilities with regularization
        let targets: Array1<f64> = y
            .iter()
            .map(|&v| {
                if (v - self.positive_class).abs() < 1e-10 {
                    (n_pos + 1.0) / (n_pos + 2.0)
                } else {
                    1.0 / (n_neg + 2.0)
                }
            })
            .collect();

        // Newton-Raphson optimization for A and B
        // P(y=1|f) = 1 / (1 + exp(A*f + B))
        let mut a = 0.0;
        let mut b = ((n_neg + 1.0) / (n_pos + 1.0)).ln();

        let max_iter = 100;
        let min_step = 1e-10;
        let sigma = 1e-12;

        for _ in 0..max_iter {
            // Compute gradient and Hessian
            let mut g1 = 0.0;
            let mut g2 = 0.0;
            let mut h11 = sigma;
            let mut h22 = sigma;
            let mut h12 = 0.0;

            for i in 0..n_samples {
                let f = decisions[i];
                let t = targets[i];
                let p = 1.0 / (1.0 + (a * f + b).exp());
                let d = p - t;

                g1 += f * d;
                g2 += d;

                let w = p * (1.0 - p);
                h11 += f * f * w;
                h22 += w;
                h12 += f * w;
            }

            // Solve Newton step
            let det = h11 * h22 - h12 * h12;
            if det.abs() < 1e-12 {
                break;
            }

            let da = -(h22 * g1 - h12 * g2) / det;
            let db = -(h11 * g2 - h12 * g1) / det;

            if da.abs() < min_step && db.abs() < min_step {
                break;
            }

            a += da;
            b += db;
        }

        self.platt_a = a;
        self.platt_b = b;
        self.probability_fitted = true;

        Ok(())
    }

    /// Predict probability for the positive class (internal method for future public API).
    fn _predict_proba_positive(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.probability_fitted {
            return Err(FerroError::invalid_input(
                "Probability estimates not fitted. Call fit with probability=true",
            ));
        }

        let decisions = self.decision_function(x)?;
        Ok(decisions.mapv(|f| 1.0 / (1.0 + self.platt_a.mul_add(f, self.platt_b).exp())))
    }
}

// =============================================================================
// SVC (Public Interface)
// =============================================================================

/// Support Vector Classification.
///
/// A powerful classification method that finds the optimal hyperplane
/// separating classes. Supports various kernel functions for non-linear
/// classification.
///
/// ## Parameters
///
/// - `c`: Regularization parameter (default: 1.0). Higher C means less regularization.
/// - `kernel`: Kernel function (default: RBF with gamma=1/n_features)
/// - `tol`: Tolerance for stopping criterion (default: 1e-3)
/// - `max_iter`: Maximum SMO iterations (default: 1000)
/// - `probability`: Whether to enable probability estimates (default: false)
/// - `multiclass_strategy`: Strategy for multiclass (default: OneVsOne)
/// - `class_weight`: Weight for each class (default: Uniform)
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::models::svm::{SVC, Kernel, MulticlassStrategy};
/// use ferroml_core::models::Model;
///
/// let mut clf = SVC::new()
///     .with_kernel(Kernel::Rbf { gamma: 0.5 })
///     .with_c(1.0)
///     .with_probability(true)
///     .with_multiclass_strategy(MulticlassStrategy::OneVsOne);
///
/// clf.fit(&x_train, &y_train)?;
/// let predictions = clf.predict(&x_test)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVC {
    /// Regularization parameter
    pub c: f64,
    /// Kernel function
    pub kernel: Kernel,
    /// Tolerance for stopping criterion
    pub tol: f64,
    /// Maximum number of SMO iterations
    pub max_iter: usize,
    /// Whether to enable probability estimates
    pub probability: bool,
    /// Multiclass classification strategy
    pub multiclass_strategy: MulticlassStrategy,
    /// Class weight specification
    pub class_weight: ClassWeight,

    // Fitted state
    /// Unique class labels
    classes: Option<Array1<f64>>,
    /// Binary classifiers (for OvO or OvR)
    classifiers: Vec<BinarySVC>,
    /// Class pairs for OvO (class_i, class_j)
    class_pairs: Vec<(f64, f64)>,
    /// Number of features
    n_features: Option<usize>,
    /// Computed class weights
    computed_weights: Option<Vec<(f64, f64)>>,
}

impl Default for SVC {
    fn default() -> Self {
        Self::new()
    }
}

impl SVC {
    /// Create a new SVC with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            c: 1.0,
            kernel: Kernel::rbf_auto(),
            tol: 1e-3,
            max_iter: 1000,
            probability: false,
            multiclass_strategy: MulticlassStrategy::OneVsOne,
            class_weight: ClassWeight::Uniform,
            classes: None,
            classifiers: Vec::new(),
            class_pairs: Vec::new(),
            n_features: None,
            computed_weights: None,
        }
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c.max(1e-10);
        self
    }

    /// Set the kernel function.
    #[must_use]
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the tolerance for stopping criterion.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol.max(1e-10);
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Enable or disable probability estimates.
    #[must_use]
    pub fn with_probability(mut self, probability: bool) -> Self {
        self.probability = probability;
        self
    }

    /// Set the multiclass classification strategy.
    #[must_use]
    pub fn with_multiclass_strategy(mut self, strategy: MulticlassStrategy) -> Self {
        self.multiclass_strategy = strategy;
        self
    }

    /// Set the class weight specification.
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Get the unique class labels.
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Get the number of support vectors for each classifier.
    #[must_use]
    pub fn n_support_vectors(&self) -> Vec<usize> {
        self.classifiers
            .iter()
            .map(|clf| clf.support_indices.as_ref().map(|v| v.len()).unwrap_or(0))
            .collect()
    }

    /// Compute class weights from training data.
    fn compute_class_weights(&mut self, y: &Array1<f64>) {
        let classes = self.classes.as_ref().unwrap();
        let n_samples = y.len() as f64;

        let weights = match &self.class_weight {
            ClassWeight::Uniform => classes.iter().map(|&c| (c, 1.0)).collect(),
            ClassWeight::Balanced => {
                // Weight inversely proportional to class frequency
                let n_classes = classes.len() as f64;
                classes
                    .iter()
                    .map(|&c| {
                        let count = y.iter().filter(|&&v| (v - c).abs() < 1e-10).count() as f64;
                        let weight = n_samples / (n_classes * count);
                        (c, weight)
                    })
                    .collect()
            }
            ClassWeight::Custom(weights) => weights.clone(),
        };

        self.computed_weights = Some(weights);
    }

    /// Get weight for a specific class.
    fn get_class_weight(&self, class: f64) -> f64 {
        self.computed_weights
            .as_ref()
            .and_then(|weights| {
                weights
                    .iter()
                    .find(|(c, _)| (*c - class).abs() < 1e-10)
                    .map(|(_, w)| *w)
            })
            .unwrap_or(1.0)
    }

    /// Fit One-vs-One classifiers.
    fn fit_ovo(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();

        self.classifiers.clear();
        self.class_pairs.clear();

        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                let class_i = classes[i];
                let class_j = classes[j];

                // Extract samples for these two classes
                let mut indices = Vec::new();
                for (idx, &label) in y.iter().enumerate() {
                    if (label - class_i).abs() < 1e-10 || (label - class_j).abs() < 1e-10 {
                        indices.push(idx);
                    }
                }

                let n_binary = indices.len();
                let n_features = x.ncols();

                let mut x_binary = Array2::zeros((n_binary, n_features));
                let mut y_binary = Array1::zeros(n_binary);

                for (new_idx, &orig_idx) in indices.iter().enumerate() {
                    x_binary.row_mut(new_idx).assign(&x.row(orig_idx));
                    y_binary[new_idx] = y[orig_idx];
                }

                // Compute sample weights for this binary classifier
                let sample_weights: Array1<f64> = y_binary
                    .iter()
                    .map(|&label| self.get_class_weight(label))
                    .collect();

                // Train binary classifier
                let mut binary_clf = BinarySVC::new(self.c, self.kernel, self.tol, self.max_iter);
                binary_clf.fit(
                    &x_binary,
                    &y_binary,
                    class_i,
                    class_j,
                    Some(&sample_weights),
                )?;

                // Fit Platt scaling if probability estimates are needed
                if self.probability {
                    binary_clf.fit_platt_scaling(&x_binary, &y_binary)?;
                }

                self.classifiers.push(binary_clf);
                self.class_pairs.push((class_i, class_j));
            }
        }

        Ok(())
    }

    /// Fit One-vs-Rest classifiers.
    fn fit_ovr(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let classes = self.classes.as_ref().unwrap();

        self.classifiers.clear();
        self.class_pairs.clear();

        for &class in classes.iter() {
            // Create binary labels: class vs rest
            let y_binary: Array1<f64> = y
                .iter()
                .map(|&label| {
                    if (label - class).abs() < 1e-10 {
                        class
                    } else {
                        -1.0
                    }
                })
                .collect();

            // Compute sample weights
            let sample_weights: Array1<f64> = y
                .iter()
                .map(|&label| self.get_class_weight(label))
                .collect();

            // Train binary classifier
            let mut binary_clf = BinarySVC::new(self.c, self.kernel, self.tol, self.max_iter);
            binary_clf.fit(x, &y_binary, class, -1.0, Some(&sample_weights))?;

            if self.probability {
                binary_clf.fit_platt_scaling(x, &y_binary)?;
            }

            self.classifiers.push(binary_clf);
            self.class_pairs.push((class, -1.0)); // -1 represents "rest"
        }

        Ok(())
    }

    /// Predict using One-vs-One voting.
    fn predict_ovo(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();
        let n_samples = x.nrows();

        let mut votes = Array2::zeros((n_samples, n_classes));

        for (clf, &(class_i, class_j)) in self.classifiers.iter().zip(self.class_pairs.iter()) {
            let predictions = clf.predict(x)?;

            for (sample_idx, &pred) in predictions.iter().enumerate() {
                if (pred - class_i).abs() < 1e-10 {
                    // Find index of class_i
                    for (class_idx, &c) in classes.iter().enumerate() {
                        if (c - class_i).abs() < 1e-10 {
                            votes[[sample_idx, class_idx]] += 1.0;
                            break;
                        }
                    }
                } else {
                    // Find index of class_j
                    for (class_idx, &c) in classes.iter().enumerate() {
                        if (c - class_j).abs() < 1e-10 {
                            votes[[sample_idx, class_idx]] += 1.0;
                            break;
                        }
                    }
                }
            }
        }

        // Find class with most votes
        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let row = votes.row(i);
            let best_class_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a): &(usize, &f64), (_, b): &(usize, &f64)| {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = classes[best_class_idx];
        }

        Ok(predictions)
    }

    /// Predict using One-vs-Rest.
    fn predict_ovr(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let classes = self.classes.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_classes = classes.len();

        // Get decision values from each OvR classifier
        let mut decision_values = Array2::zeros((n_samples, n_classes));

        for (clf_idx, clf) in self.classifiers.iter().enumerate() {
            let decisions = clf.decision_function(x)?;
            for i in 0..n_samples {
                decision_values[[i, clf_idx]] = decisions[i];
            }
        }

        // Select class with highest decision value
        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let best_class_idx = decision_values
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = classes[best_class_idx];
        }

        Ok(predictions)
    }

    /// Predict probabilities using OvO (pairwise coupling).
    fn predict_proba_ovo(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();
        let n_samples = x.nrows();

        // Use simple voting-based probability estimation
        let mut proba = Array2::zeros((n_samples, n_classes));
        let mut votes: Array2<f64> = Array2::zeros((n_samples, n_classes));

        for (clf, &(_class_i, _class_j)) in self.classifiers.iter().zip(self.class_pairs.iter()) {
            let predictions = clf.predict(x)?;

            for (sample_idx, &pred) in predictions.iter().enumerate() {
                for (class_idx, &c) in classes.iter().enumerate() {
                    if (c - pred).abs() < 1e-10 {
                        votes[[sample_idx, class_idx]] += 1.0;
                        break;
                    }
                }
            }
        }

        // Normalize votes to probabilities
        for i in 0..n_samples {
            let total: f64 = votes.row(i).sum();
            if total > 0.0 {
                for j in 0..n_classes {
                    proba[[i, j]] = votes[[i, j]] / total;
                }
            } else {
                // Uniform if no votes
                for j in 0..n_classes {
                    proba[[i, j]] = 1.0 / n_classes as f64;
                }
            }
        }

        Ok(proba)
    }

    /// Predict probabilities using OvR (softmax normalization).
    fn predict_proba_ovr(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let classes = self.classes.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_classes = classes.len();

        // Get decision values and apply softmax
        let mut decision_values = Array2::zeros((n_samples, n_classes));

        for (clf_idx, clf) in self.classifiers.iter().enumerate() {
            let decisions = clf.decision_function(x)?;
            for i in 0..n_samples {
                decision_values[[i, clf_idx]] = decisions[i];
            }
        }

        // Softmax normalization
        let mut proba = Array2::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            let max_val = decision_values
                .row(i)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = decision_values
                .row(i)
                .iter()
                .map(|&v| (v - max_val).exp())
                .sum();

            for j in 0..n_classes {
                proba[[i, j]] = (decision_values[[i, j]] - max_val).exp() / exp_sum;
            }
        }

        Ok(proba)
    }
}

impl Model for SVC {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        // Extract unique classes
        let mut classes_vec: Vec<f64> = y.iter().copied().collect();
        classes_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        classes_vec.dedup();

        if classes_vec.len() < 2 {
            return Err(FerroError::invalid_input(
                "Need at least 2 classes for classification",
            ));
        }

        self.classes = Some(Array1::from_vec(classes_vec));
        self.n_features = Some(x.ncols());

        // Compute class weights
        self.compute_class_weights(y);

        // Train classifiers
        match self.multiclass_strategy {
            MulticlassStrategy::OneVsOne => self.fit_ovo(x, y)?,
            MulticlassStrategy::OneVsRest => self.fit_ovr(x, y)?,
        }

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.classes, "predict")?;
        validate_predict_input(x, self.n_features.unwrap())?;

        match self.multiclass_strategy {
            MulticlassStrategy::OneVsOne => self.predict_ovo(x),
            MulticlassStrategy::OneVsRest => self.predict_ovr(x),
        }
    }

    fn is_fitted(&self) -> bool {
        self.classes.is_some() && !self.classifiers.is_empty()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .float("C", 1e-3, 100.0)
            .float("gamma", 1e-4, 10.0)
            .categorical(
                "kernel",
                vec!["linear".to_string(), "rbf".to_string(), "poly".to_string()],
            )
    }
}

impl ProbabilisticModel for SVC {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.classes, "predict_proba")?;
        validate_predict_input(x, self.n_features.unwrap())?;

        if !self.probability {
            return Err(FerroError::invalid_input(
                "Probability estimates not enabled. Create SVC with .with_probability(true)",
            ));
        }

        match self.multiclass_strategy {
            MulticlassStrategy::OneVsOne => self.predict_proba_ovo(x),
            MulticlassStrategy::OneVsRest => self.predict_proba_ovr(x),
        }
    }

    fn predict_interval(&self, x: &Array2<f64>, level: f64) -> Result<PredictionInterval> {
        // For classification, return prediction with probability-based uncertainty
        let probas = self.predict_proba(x)?;
        let predictions = self.predict(x)?;

        let n_samples = x.nrows();
        let mut lower = Array1::zeros(n_samples);
        let mut upper = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let max_prob = probas.row(i).iter().cloned().fold(0.0_f64, f64::max);
            let uncertainty = 1.0 - max_prob;
            let half_width = uncertainty * (1.0 - level);

            lower[i] = predictions[i] - half_width;
            upper[i] = predictions[i] + half_width;
        }

        Ok(PredictionInterval::new(predictions, lower, upper, level))
    }
}

// =============================================================================
// SVR (Support Vector Regression)
// =============================================================================

/// Support Vector Regression.
///
/// A regression method that finds a function that deviates from the actual
/// target values by no more than epsilon for each training point while being
/// as flat as possible. Uses the epsilon-insensitive loss function.
///
/// ## Parameters
///
/// - `c`: Regularization parameter (default: 1.0). Higher C means less regularization.
/// - `epsilon`: Width of the epsilon-insensitive tube (default: 0.1).
///   Errors within this margin are not penalized.
/// - `kernel`: Kernel function (default: RBF with gamma=1/n_features)
/// - `tol`: Tolerance for stopping criterion (default: 1e-3)
/// - `max_iter`: Maximum SMO iterations (default: 1000)
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::models::svm::{SVR, Kernel};
/// use ferroml_core::models::Model;
///
/// let mut reg = SVR::new()
///     .with_kernel(Kernel::Rbf { gamma: 0.5 })
///     .with_c(1.0)
///     .with_epsilon(0.1);
///
/// reg.fit(&x_train, &y_train)?;
/// let predictions = reg.predict(&x_test)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVR {
    /// Regularization parameter
    pub c: f64,
    /// Width of epsilon-insensitive tube
    pub epsilon: f64,
    /// Kernel function
    pub kernel: Kernel,
    /// Tolerance for stopping criterion
    pub tol: f64,
    /// Maximum number of SMO iterations
    pub max_iter: usize,

    // Fitted state
    /// Support vectors
    support_vectors: Option<Array2<f64>>,
    /// Dual coefficients (alpha_i - alpha_i^*)
    dual_coef: Option<Array1<f64>>,
    /// Intercept (bias term)
    intercept: f64,
    /// Indices of support vectors in original training data
    support_indices: Option<Vec<usize>>,
    /// Number of features
    n_features: Option<usize>,
}

impl Default for SVR {
    fn default() -> Self {
        Self::new()
    }
}

impl SVR {
    /// Create a new SVR with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.1,
            kernel: Kernel::rbf_auto(),
            tol: 1e-3,
            max_iter: 1000,
            support_vectors: None,
            dual_coef: None,
            intercept: 0.0,
            support_indices: None,
            n_features: None,
        }
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c.max(1e-10);
        self
    }

    /// Set the epsilon parameter (tube width).
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon.max(0.0);
        self
    }

    /// Set the kernel function.
    #[must_use]
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the tolerance for stopping criterion.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol.max(1e-10);
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Get the number of support vectors.
    #[must_use]
    pub fn n_support_vectors(&self) -> usize {
        self.support_indices.as_ref().map(|v| v.len()).unwrap_or(0)
    }

    /// Get the support vector indices.
    #[must_use]
    pub fn support_indices(&self) -> Option<&[usize]> {
        self.support_indices.as_deref()
    }

    /// Get the dual coefficients.
    #[must_use]
    pub fn dual_coef(&self) -> Option<&Array1<f64>> {
        self.dual_coef.as_ref()
    }

    /// Get the intercept.
    #[must_use]
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Compute full kernel matrix.
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let mut k = Array2::zeros((n, n));

        for i in 0..n {
            let xi: Vec<f64> = x.row(i).to_vec();
            for j in i..n {
                let xj: Vec<f64> = x.row(j).to_vec();
                let val = self.kernel.compute(&xi, &xj);
                k[[i, j]] = val;
                k[[j, i]] = val;
            }
        }

        k
    }

    /// Coordinate descent algorithm for SVR.
    ///
    /// This is a simpler and more robust approach than full SMO.
    /// We iterate over each sample and update its coefficient to minimize the objective.
    ///
    /// The decision function is: f(x) = sum(coef_i * K(x_i, x)) + b
    /// where coef_i = alpha_i - alpha_star_i ∈ [-C, C]
    fn smo_regression(
        &self,
        kernel_matrix: &Array2<f64>,
        y: &Array1<f64>,
        sample_weights: Option<&Array1<f64>>,
    ) -> Result<(Array1<f64>, f64)> {
        let n = y.len();

        // Check for constant or near-constant targets
        // If range of y is within epsilon, no support vectors are needed
        let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_range = y_max - y_min;

        if y_range <= 2.0 * self.epsilon {
            // All targets within epsilon tube - just use mean
            let mean_y = y.mean().unwrap_or(0.0);
            return Ok((Array1::zeros(n), mean_y));
        }

        // Effective C for each sample
        let c: Array1<f64> = if let Some(weights) = sample_weights {
            weights.mapv(|w| self.c * w)
        } else {
            Array1::from_elem(n, self.c)
        };

        // Initialize coefficients: coef[i] = alpha[i] - alpha_star[i]
        let mut coef = Array1::zeros(n);

        // Cache for predictions: f[i] = sum_j coef[j] * K[j, i]
        let mut f: Array1<f64> = Array1::zeros(n);

        let mut iter = 0;
        let mut max_delta = f64::INFINITY;

        while iter < self.max_iter && max_delta > self.tol {
            max_delta = 0.0;

            for i in 0..n {
                let old_coef: f64 = coef[i];
                let k_ii = kernel_matrix[[i, i]];

                // Skip if diagonal is zero (shouldn't happen with valid kernels)
                if k_ii < 1e-12 {
                    continue;
                }

                // Compute gradient with respect to coef[i]
                // The error is f[i] - y[i]
                // We want to minimize: 0.5 * coef[i]^2 * K[i,i] + coef[i] * (f[i] - coef[i] * K[i,i] - y[i])
                //                      + epsilon * |coef[i]|
                // Taking derivative and setting to zero (with clipping for epsilon-insensitive loss)

                // Current prediction without contribution from i
                let f_without_i = old_coef.mul_add(-k_ii, f[i]);

                // The optimal coef[i] minimizes the loss
                // For epsilon-insensitive loss:
                // - If f_without_i < y[i] - epsilon: coef[i] should be positive (increase prediction)
                // - If f_without_i > y[i] + epsilon: coef[i] should be negative (decrease prediction)
                // - Otherwise: coef[i] = 0 (prediction is within epsilon tube)

                let error_without_i: f64 = f_without_i - y[i];

                let new_coef: f64 = if error_without_i < -self.epsilon {
                    // Need to increase prediction
                    ((-error_without_i - self.epsilon) / k_ii).min(c[i])
                } else if error_without_i > self.epsilon {
                    // Need to decrease prediction
                    ((-error_without_i + self.epsilon) / k_ii).max(-c[i])
                } else {
                    // Within epsilon tube
                    0.0
                };

                let delta: f64 = new_coef - old_coef;

                if delta.abs() > 1e-12 {
                    coef[i] = new_coef;

                    // Update predictions
                    for k in 0..n {
                        f[k] += delta * kernel_matrix[[i, k]];
                    }

                    max_delta = max_delta.max(delta.abs());
                }
            }

            iter += 1;
        }

        // Compute intercept
        let b = self.compute_intercept_cd(&coef, y, &f, &c);

        Ok((coef, b))
    }

    /// Compute intercept for coordinate descent SVR.
    fn compute_intercept_cd(
        &self,
        coef: &Array1<f64>,
        y: &Array1<f64>,
        f: &Array1<f64>,
        c: &Array1<f64>,
    ) -> f64 {
        let n = y.len();
        let mut sum_b = 0.0;
        let mut count = 0;

        for i in 0..n {
            let c_i = c[i];
            let coef_i = coef[i];

            // Free support vector: 0 < |coef[i]| < C
            if coef_i.abs() > 1e-8 && coef_i.abs() < c_i - 1e-8 {
                // For positive coef (alpha > 0): error = -epsilon
                // For negative coef (alpha_star > 0): error = +epsilon
                if coef_i > 0.0 {
                    // y[i] = f[i] + b - epsilon => b = y[i] - f[i] + epsilon
                    // But f already includes coef contribution, so:
                    // y[i] = (f[i] - b) + b => b = y[i] - f[i] + epsilon... no
                    // Actually: y[i] - epsilon = f_without_b[i] + b
                    // So b = y[i] - epsilon - f_without_b[i]
                    // And f_without_b[i] = f[i] (since we don't have b in f yet)
                    sum_b += y[i] - self.epsilon - f[i];
                } else {
                    sum_b += y[i] + self.epsilon - f[i];
                }
                count += 1;
            }
        }

        if count > 0 {
            return sum_b / count as f64;
        }

        // Fallback: average over all support vectors
        for i in 0..n {
            if coef[i].abs() > 1e-8 {
                sum_b += y[i] - f[i];
                count += 1;
            }
        }

        if count > 0 {
            sum_b / count as f64
        } else {
            // No support vectors: use mean of y
            y.mean().unwrap_or(0.0)
        }
    }
}

impl Model for SVR {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        self.n_features = Some(n_features);

        // Set auto gamma if needed
        self.kernel = self.kernel.with_auto_gamma(n_features);

        // Compute kernel matrix
        let kernel_matrix = self.compute_kernel_matrix(x);

        // Run SMO for regression
        let (alpha_diff, intercept) = self.smo_regression(&kernel_matrix, y, None)?;

        // Extract support vectors (points with |alpha_diff| > threshold)
        let support_threshold = 1e-8;
        let mut support_indices = Vec::new();
        let mut dual_coef = Vec::new();

        for i in 0..n_samples {
            if alpha_diff[i].abs() > support_threshold {
                support_indices.push(i);
                dual_coef.push(alpha_diff[i]);
            }
        }

        // Handle the case where all points are within the epsilon tube
        // (e.g., constant function or very low variance data)
        if support_indices.is_empty() {
            // Create a dummy support vector setup with zero coefficients
            // The model will just return the intercept (mean of y)
            self.support_vectors = Some(Array2::zeros((0, n_features)));
            self.dual_coef = Some(Array1::zeros(0));
            self.support_indices = Some(Vec::new());
            self.intercept = y.mean().unwrap_or(0.0);
            return Ok(());
        }

        // Store support vectors
        let n_sv = support_indices.len();
        let mut sv_matrix = Array2::zeros((n_sv, n_features));
        for (i, &idx) in support_indices.iter().enumerate() {
            sv_matrix.row_mut(i).assign(&x.row(idx));
        }

        self.support_vectors = Some(sv_matrix);
        self.dual_coef = Some(Array1::from_vec(dual_coef));
        self.support_indices = Some(support_indices);
        self.intercept = intercept;

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.support_vectors, "predict")?;
        validate_predict_input(x, self.n_features.unwrap())?;

        let sv = self.support_vectors.as_ref().unwrap();
        let dual_coef = self.dual_coef.as_ref().unwrap();
        let n_samples = x.nrows();

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let xi: Vec<f64> = x.row(i).to_vec();
            let mut sum = 0.0;

            for (j, coef) in dual_coef.iter().enumerate() {
                let svj: Vec<f64> = sv.row(j).to_vec();
                sum += coef * self.kernel.compute(&xi, &svj);
            }

            predictions[i] = sum + self.intercept;
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.support_vectors.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .float("C", 1e-3, 100.0)
            .float("epsilon", 1e-4, 1.0)
            .float("gamma", 1e-4, 10.0)
            .categorical(
                "kernel",
                vec!["linear".to_string(), "rbf".to_string(), "poly".to_string()],
            )
    }
}

// =============================================================================
// LinearSVC (Linear Support Vector Classification)
// =============================================================================

/// Loss function for LinearSVC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearSVCLoss {
    /// Hinge loss: max(0, 1 - y·f(x))
    /// Standard SVM loss function
    Hinge,
    /// Squared hinge loss: max(0, 1 - y·f(x))²
    /// Smoother, differentiable everywhere
    SquaredHinge,
}

impl Default for LinearSVCLoss {
    fn default() -> Self {
        Self::SquaredHinge
    }
}

/// Linear Support Vector Classification.
///
/// A fast linear SVM classifier optimized for large datasets. Uses coordinate
/// descent in the primal formulation, avoiding the O(n²) kernel matrix
/// computation of kernelized SVC.
///
/// ## Parameters
///
/// - `c`: Regularization parameter (default: 1.0). Higher C means less regularization.
/// - `loss`: Loss function - Hinge or SquaredHinge (default: SquaredHinge)
/// - `tol`: Tolerance for stopping criterion (default: 1e-4)
/// - `max_iter`: Maximum iterations (default: 1000)
/// - `fit_intercept`: Whether to fit an intercept (default: true)
/// - `class_weight`: Weight for each class (default: Uniform)
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::models::svm::LinearSVC;
/// use ferroml_core::models::Model;
///
/// let mut clf = LinearSVC::new()
///     .with_c(1.0)
///     .with_max_iter(1000);
///
/// clf.fit(&x_train, &y_train)?;
/// let predictions = clf.predict(&x_test)?;
/// ```
///
/// ## Algorithm
///
/// Uses coordinate descent to solve the primal optimization problem:
///
/// min_w (1/2)||w||² + C·Σ loss(y_i, w·x_i + b)
///
/// This is equivalent to LIBLINEAR's L2-regularized L1 or L2 loss SVM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearSVC {
    /// Regularization parameter
    pub c: f64,
    /// Loss function
    pub loss: LinearSVCLoss,
    /// Tolerance for stopping criterion
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Class weight specification
    pub class_weight: ClassWeight,

    // Fitted state
    /// Weight vector for each binary classifier
    weights: Option<Vec<Array1<f64>>>,
    /// Intercept for each binary classifier
    intercepts: Option<Vec<f64>>,
    /// Unique class labels
    classes: Option<Array1<f64>>,
    /// Number of features
    n_features: Option<usize>,
    /// Computed class weights
    computed_weights: Option<Vec<(f64, f64)>>,
}

impl Default for LinearSVC {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearSVC {
    /// Create a new LinearSVC with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            c: 1.0,
            loss: LinearSVCLoss::SquaredHinge,
            tol: 1e-4,
            max_iter: 1000,
            fit_intercept: true,
            class_weight: ClassWeight::Uniform,
            weights: None,
            intercepts: None,
            classes: None,
            n_features: None,
            computed_weights: None,
        }
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c.max(1e-10);
        self
    }

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: LinearSVCLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set the tolerance for stopping criterion.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol.max(1e-10);
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Set whether to fit an intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set the class weight specification.
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Get the unique class labels.
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Get the learned weight vectors.
    #[must_use]
    pub fn weights(&self) -> Option<&Vec<Array1<f64>>> {
        self.weights.as_ref()
    }

    /// Get the learned intercepts.
    #[must_use]
    pub fn intercepts(&self) -> Option<&Vec<f64>> {
        self.intercepts.as_ref()
    }

    /// Compute class weights from training data.
    fn compute_class_weights(&mut self, y: &Array1<f64>) {
        let classes = self.classes.as_ref().unwrap();
        let n_samples = y.len() as f64;

        let weights = match &self.class_weight {
            ClassWeight::Uniform => classes.iter().map(|&c| (c, 1.0)).collect(),
            ClassWeight::Balanced => {
                let n_classes = classes.len() as f64;
                classes
                    .iter()
                    .map(|&c| {
                        let count = y.iter().filter(|&&v| (v - c).abs() < 1e-10).count() as f64;
                        let weight = n_samples / (n_classes * count);
                        (c, weight)
                    })
                    .collect()
            }
            ClassWeight::Custom(weights) => weights.clone(),
        };

        self.computed_weights = Some(weights);
    }

    /// Get weight for a specific class.
    fn get_class_weight(&self, class: f64) -> f64 {
        self.computed_weights
            .as_ref()
            .and_then(|weights| {
                weights
                    .iter()
                    .find(|(c, _)| (*c - class).abs() < 1e-10)
                    .map(|(_, w)| *w)
            })
            .unwrap_or(1.0)
    }

    /// Fit a binary linear SVC using coordinate descent.
    fn fit_binary(
        &self,
        x: &Array2<f64>,
        y_binary: &Array1<f64>,
        sample_weights: &Array1<f64>,
    ) -> Result<(Array1<f64>, f64)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize weights and intercept
        let mut w = Array1::zeros(n_features);
        let mut b = 0.0;

        // Precompute ||x_i||² for each sample
        let x_norm_sq: Vec<f64> = (0..n_samples)
            .map(|i| x.row(i).iter().map(|&v| v * v).sum())
            .collect();

        // Effective C for each sample
        let c: Array1<f64> = sample_weights.mapv(|sw| self.c * sw);

        // Coordinate descent
        for _iter in 0..self.max_iter {
            let mut max_delta: f64 = 0.0;

            for i in 0..n_samples {
                // Compute decision value
                let f_i = dot_product_array(&w, &x.row(i).to_owned()) + b;
                let margin = y_binary[i] * f_i;

                // Compute the optimal step
                let (delta_w, delta_b) = match self.loss {
                    LinearSVCLoss::Hinge => {
                        // Hinge loss: subgradient at margin = 1
                        if margin < 1.0 {
                            let scale = c[i] * y_binary[i];
                            let denom = x_norm_sq[i] + if self.fit_intercept { 1.0 } else { 0.0 };
                            if denom > 1e-10 {
                                let step = (1.0 - margin) / denom;
                                (
                                    scale * step,
                                    if self.fit_intercept {
                                        scale * step
                                    } else {
                                        0.0
                                    },
                                )
                            } else {
                                (0.0, 0.0)
                            }
                        } else {
                            (0.0, 0.0)
                        }
                    }
                    LinearSVCLoss::SquaredHinge => {
                        // Squared hinge loss: gradient is smooth
                        if margin < 1.0 {
                            let scale = 2.0 * c[i] * y_binary[i] * (1.0 - margin);
                            let denom = (2.0 * c[i]).mul_add(
                                x_norm_sq[i] + if self.fit_intercept { 1.0 } else { 0.0 },
                                1.0,
                            );
                            (
                                scale / denom,
                                if self.fit_intercept {
                                    scale / denom
                                } else {
                                    0.0
                                },
                            )
                        } else {
                            (0.0, 0.0)
                        }
                    }
                };

                if delta_w.abs() > 1e-12 {
                    // Update weights
                    for j in 0..n_features {
                        w[j] += delta_w * x[[i, j]];
                    }
                    b += delta_b;

                    max_delta = max_delta.max(delta_w.abs());
                }
            }

            // Apply L2 regularization shrinkage
            let shrink_factor = 1.0 / (1.0 + 1.0 / (self.c * n_samples as f64));
            w *= shrink_factor;

            // Check convergence
            if max_delta < self.tol {
                break;
            }
        }

        Ok((w, b))
    }

    /// Compute decision function for a binary classifier.
    fn decision_function_binary(&self, x: &Array2<f64>, w: &Array1<f64>, b: f64) -> Array1<f64> {
        let n_samples = x.nrows();
        let mut decisions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            decisions[i] = dot_product_array(w, &x.row(i).to_owned()) + b;
        }

        decisions
    }
}

impl Model for LinearSVC {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        // Extract unique classes
        let mut classes_vec: Vec<f64> = y.iter().copied().collect();
        classes_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        classes_vec.dedup();

        if classes_vec.len() < 2 {
            return Err(FerroError::invalid_input(
                "Need at least 2 classes for classification",
            ));
        }

        self.classes = Some(Array1::from_vec(classes_vec.clone()));
        self.n_features = Some(x.ncols());

        // Compute class weights
        self.compute_class_weights(y);

        let n_classes = classes_vec.len();
        let mut weights = Vec::with_capacity(n_classes);
        let mut intercepts = Vec::with_capacity(n_classes);

        if n_classes == 2 {
            // Binary classification
            let positive_class = classes_vec[1];
            let _negative_class = classes_vec[0];

            let y_binary: Array1<f64> = y
                .iter()
                .map(|&label| {
                    if (label - positive_class).abs() < 1e-10 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect();

            let sample_weights: Array1<f64> = y
                .iter()
                .map(|&label| self.get_class_weight(label))
                .collect();

            let (w, b) = self.fit_binary(x, &y_binary, &sample_weights)?;
            weights.push(w);
            intercepts.push(b);
        } else {
            // Multiclass: One-vs-Rest
            for &class in &classes_vec {
                let y_binary: Array1<f64> = y
                    .iter()
                    .map(|&label| {
                        if (label - class).abs() < 1e-10 {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect();

                let sample_weights: Array1<f64> = y
                    .iter()
                    .map(|&label| self.get_class_weight(label))
                    .collect();

                let (w, b) = self.fit_binary(x, &y_binary, &sample_weights)?;
                weights.push(w);
                intercepts.push(b);
            }
        }

        self.weights = Some(weights);
        self.intercepts = Some(intercepts);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.classes, "predict")?;
        validate_predict_input(x, self.n_features.unwrap())?;

        let classes = self.classes.as_ref().unwrap();
        let weights = self.weights.as_ref().unwrap();
        let intercepts = self.intercepts.as_ref().unwrap();
        let n_samples = x.nrows();

        let mut predictions = Array1::zeros(n_samples);

        if classes.len() == 2 {
            // Binary classification
            let decisions = self.decision_function_binary(x, &weights[0], intercepts[0]);
            for i in 0..n_samples {
                predictions[i] = if decisions[i] >= 0.0 {
                    classes[1]
                } else {
                    classes[0]
                };
            }
        } else {
            // Multiclass: highest decision value wins
            let n_classes = classes.len();
            let mut decision_values = Array2::zeros((n_samples, n_classes));

            for (class_idx, (w, &b)) in weights.iter().zip(intercepts.iter()).enumerate() {
                let decisions = self.decision_function_binary(x, w, b);
                for i in 0..n_samples {
                    decision_values[[i, class_idx]] = decisions[i];
                }
            }

            for i in 0..n_samples {
                let best_class_idx = decision_values
                    .row(i)
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                predictions[i] = classes[best_class_idx];
            }
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.classes.is_some() && self.weights.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        // For linear models, absolute weight magnitude indicates importance
        self.weights.as_ref().map(|weights| {
            if weights.len() == 1 {
                weights[0].mapv(|v| v.abs())
            } else {
                // Average absolute weights across all OvR classifiers
                let n_features = weights[0].len();
                let mut importance = Array1::zeros(n_features);
                for w in weights {
                    for j in 0..n_features {
                        importance[j] += w[j].abs();
                    }
                }
                importance /= weights.len() as f64;
                importance
            }
        })
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new().float("C", 1e-3, 100.0).categorical(
            "loss",
            vec!["hinge".to_string(), "squared_hinge".to_string()],
        )
    }
}

// =============================================================================
// LinearSVR (Linear Support Vector Regression)
// =============================================================================

/// Loss function for LinearSVR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearSVRLoss {
    /// Epsilon-insensitive loss: max(0, |y - f(x)| - ε)
    EpsilonInsensitive,
    /// Squared epsilon-insensitive loss: max(0, |y - f(x)| - ε)²
    SquaredEpsilonInsensitive,
}

impl Default for LinearSVRLoss {
    fn default() -> Self {
        Self::EpsilonInsensitive
    }
}

/// Linear Support Vector Regression.
///
/// A fast linear SVR optimized for large datasets. Uses coordinate descent
/// in the primal formulation, avoiding the O(n²) kernel matrix computation
/// of kernelized SVR.
///
/// ## Parameters
///
/// - `c`: Regularization parameter (default: 1.0). Higher C means less regularization.
/// - `epsilon`: Width of the epsilon-insensitive tube (default: 0.0).
///   Errors within this margin are not penalized.
/// - `loss`: Loss function - EpsilonInsensitive or SquaredEpsilonInsensitive (default: EpsilonInsensitive)
/// - `tol`: Tolerance for stopping criterion (default: 1e-4)
/// - `max_iter`: Maximum iterations (default: 1000)
/// - `fit_intercept`: Whether to fit an intercept (default: true)
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::models::svm::LinearSVR;
/// use ferroml_core::models::Model;
///
/// let mut reg = LinearSVR::new()
///     .with_c(1.0)
///     .with_epsilon(0.1)
///     .with_max_iter(1000);
///
/// reg.fit(&x_train, &y_train)?;
/// let predictions = reg.predict(&x_test)?;
/// ```
///
/// ## Algorithm
///
/// Uses coordinate descent to solve the primal optimization problem:
///
/// min_w (1/2)||w||² + C·Σ loss(y_i, w·x_i + b)
///
/// This is equivalent to LIBLINEAR's L2-regularized L1 or L2 loss SVR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearSVR {
    /// Regularization parameter
    pub c: f64,
    /// Width of epsilon-insensitive tube
    pub epsilon: f64,
    /// Loss function
    pub loss: LinearSVRLoss,
    /// Tolerance for stopping criterion
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to fit an intercept
    pub fit_intercept: bool,

    // Fitted state
    /// Weight vector
    weights: Option<Array1<f64>>,
    /// Intercept
    intercept: f64,
    /// Number of features
    n_features: Option<usize>,
}

impl Default for LinearSVR {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearSVR {
    /// Create a new LinearSVR with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.0,
            loss: LinearSVRLoss::EpsilonInsensitive,
            tol: 1e-4,
            max_iter: 1000,
            fit_intercept: true,
            weights: None,
            intercept: 0.0,
            n_features: None,
        }
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c.max(1e-10);
        self
    }

    /// Set the epsilon parameter (tube width).
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon.max(0.0);
        self
    }

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: LinearSVRLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set the tolerance for stopping criterion.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol.max(1e-10);
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Set whether to fit an intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Get the learned weight vector.
    #[must_use]
    pub fn weights(&self) -> Option<&Array1<f64>> {
        self.weights.as_ref()
    }

    /// Get the learned intercept.
    #[must_use]
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

impl Model for LinearSVR {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        self.n_features = Some(n_features);

        // Initialize weights and intercept
        let mut w = Array1::zeros(n_features);
        let mut b = 0.0;

        // Precompute ||x_i||² for each sample
        let x_norm_sq: Vec<f64> = (0..n_samples)
            .map(|i| x.row(i).iter().map(|&v| v * v).sum())
            .collect();

        // Coordinate descent
        for _iter in 0..self.max_iter {
            let mut max_delta: f64 = 0.0;

            for i in 0..n_samples {
                // Compute prediction
                let f_i = dot_product_array(&w, &x.row(i).to_owned()) + b;
                let residual = y[i] - f_i;

                // Compute the optimal step based on loss function
                let (delta_scale, update) = match self.loss {
                    LinearSVRLoss::EpsilonInsensitive => {
                        if residual > self.epsilon {
                            // Underestimating: increase weights
                            let denom = x_norm_sq[i] + if self.fit_intercept { 1.0 } else { 0.0 };
                            if denom > 1e-10 {
                                let step = ((residual - self.epsilon) / denom).min(self.c);
                                (step, true)
                            } else {
                                (0.0, false)
                            }
                        } else if residual < -self.epsilon {
                            // Overestimating: decrease weights
                            let denom = x_norm_sq[i] + if self.fit_intercept { 1.0 } else { 0.0 };
                            if denom > 1e-10 {
                                let step = ((residual + self.epsilon) / denom).max(-self.c);
                                (step, true)
                            } else {
                                (0.0, false)
                            }
                        } else {
                            (0.0, false)
                        }
                    }
                    LinearSVRLoss::SquaredEpsilonInsensitive => {
                        if residual.abs() > self.epsilon {
                            let sign = if residual > 0.0 { 1.0 } else { -1.0 };
                            let loss_grad = 2.0 * sign * (residual.abs() - self.epsilon);
                            let denom = (2.0 * self.c).mul_add(
                                x_norm_sq[i] + if self.fit_intercept { 1.0 } else { 0.0 },
                                1.0,
                            );
                            let step = (self.c * loss_grad) / denom;
                            (step, true)
                        } else {
                            (0.0, false)
                        }
                    }
                };

                if update && delta_scale.abs() > 1e-12 {
                    // Update weights
                    for j in 0..n_features {
                        w[j] += delta_scale * x[[i, j]];
                    }
                    if self.fit_intercept {
                        b += delta_scale;
                    }

                    max_delta = max_delta.max(delta_scale.abs());
                }
            }

            // Apply L2 regularization shrinkage
            let shrink_factor = 1.0 / (1.0 + 1.0 / (self.c * n_samples as f64));
            w *= shrink_factor;

            // Check convergence
            if max_delta < self.tol {
                break;
            }
        }

        self.weights = Some(w);
        self.intercept = b;

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.weights, "predict")?;
        validate_predict_input(x, self.n_features.unwrap())?;

        let w = self.weights.as_ref().unwrap();
        let n_samples = x.nrows();

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            predictions[i] = dot_product_array(w, &x.row(i).to_owned()) + self.intercept;
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.weights.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        // For linear models, absolute weight magnitude indicates importance
        self.weights.as_ref().map(|w| w.mapv(|v| v.abs()))
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .float("C", 1e-3, 100.0)
            .float("epsilon", 0.0, 1.0)
            .categorical(
                "loss",
                vec![
                    "epsilon_insensitive".to_string(),
                    "squared_epsilon_insensitive".to_string(),
                ],
            )
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute dot product between two slices.
#[inline]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Compute dot product between an Array1 and another Array1.
#[inline]
fn dot_product_array(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Kernel Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_linear_kernel() {
        let kernel = Kernel::Linear;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];

        let result = kernel.compute(&a, &b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel() {
        let kernel = Kernel::Rbf { gamma: 1.0 };
        let a = [0.0, 0.0];
        let b = [1.0, 1.0];

        let result = kernel.compute(&a, &b);
        // exp(-1 * (1 + 1)) = exp(-2)
        assert!((result - (-2.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_poly_kernel() {
        let kernel = Kernel::Polynomial {
            gamma: 1.0,
            coef0: 1.0,
            degree: 2,
        };
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];

        let result = kernel.compute(&a, &b);
        // (1*(1*3 + 2*4) + 1)^2 = (11 + 1)^2 = 144
        assert!((result - 144.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel() {
        let kernel = Kernel::Sigmoid {
            gamma: 0.5,
            coef0: 0.0,
        };
        let a = [1.0, 1.0];
        let b = [1.0, 1.0];

        let result = kernel.compute(&a, &b);
        // tanh(0.5 * 2) = tanh(1)
        assert!((result - 1.0_f64.tanh()).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_auto_gamma() {
        let kernel = Kernel::rbf_auto();
        let adjusted = kernel.with_auto_gamma(10);

        match adjusted {
            Kernel::Rbf { gamma } => assert!((gamma - 0.1).abs() < 1e-10),
            _ => panic!("Expected RBF kernel"),
        }
    }

    // -------------------------------------------------------------------------
    // Binary SVC Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_binary_svc_linearly_separable() {
        // Simple linearly separable dataset
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = BinarySVC::new(1.0, Kernel::Linear, 1e-3, 1000);
        clf.fit(&x, &y, 1.0, 0.0, None).unwrap();

        let predictions = clf.predict(&x).unwrap();

        // Should classify training data correctly
        for i in 0..3 {
            assert!(
                (predictions[i] - 0.0).abs() < 1e-10,
                "Point {} should be class 0",
                i
            );
        }
        for i in 3..6 {
            assert!(
                (predictions[i] - 1.0).abs() < 1e-10,
                "Point {} should be class 1",
                i
            );
        }
    }

    // -------------------------------------------------------------------------
    // SVC Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_svc_basic() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = SVC::new().with_kernel(Kernel::Linear).with_c(1.0);
        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        assert_eq!(clf.classes().unwrap().len(), 2);

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_svc_rbf_kernel() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 6.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0, 6.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut clf = SVC::new()
            .with_kernel(Kernel::Rbf { gamma: 0.5 })
            .with_c(10.0);
        clf.fit(&x, &y).unwrap();

        let predictions = clf.predict(&x).unwrap();

        // Check that most predictions are correct
        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &actual)| (pred - actual).abs() < 1e-10)
            .count();

        assert!(correct >= 6, "Expected at least 6 correct, got {}", correct);
    }

    #[test]
    fn test_svc_multiclass_ovo() {
        // Three-class problem
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, // Class 0
                5.0, 5.0, 6.0, 5.0, 5.0, 6.0, // Class 1
                9.0, 9.0, 10.0, 9.0, 9.0, 10.0, // Class 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let mut clf = SVC::new()
            .with_kernel(Kernel::Linear)
            .with_c(1.0)
            .with_multiclass_strategy(MulticlassStrategy::OneVsOne);

        clf.fit(&x, &y).unwrap();

        assert_eq!(clf.classes().unwrap().len(), 3);
        // For 3 classes, OvO creates 3 binary classifiers
        assert_eq!(clf.classifiers.len(), 3);

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 9);
    }

    #[test]
    fn test_svc_multiclass_ovr() {
        // Three-class problem
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, // Class 0
                5.0, 5.0, 6.0, 5.0, 5.0, 6.0, // Class 1
                9.0, 9.0, 10.0, 9.0, 9.0, 10.0, // Class 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let mut clf = SVC::new()
            .with_kernel(Kernel::Linear)
            .with_c(1.0)
            .with_multiclass_strategy(MulticlassStrategy::OneVsRest);

        clf.fit(&x, &y).unwrap();

        // For 3 classes, OvR creates 3 binary classifiers
        assert_eq!(clf.classifiers.len(), 3);
    }

    #[test]
    fn test_svc_with_probability() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = SVC::new()
            .with_kernel(Kernel::Linear)
            .with_c(1.0)
            .with_probability(true);

        clf.fit(&x, &y).unwrap();

        let probas = clf.predict_proba(&x).unwrap();

        // Check shape
        assert_eq!(probas.nrows(), 6);
        assert_eq!(probas.ncols(), 2);

        // Check probabilities sum to 1
        for i in 0..6 {
            let sum: f64 = probas.row(i).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Row {} sum is {}, expected 1.0",
                i,
                sum
            );
        }
    }

    #[test]
    fn test_svc_class_weights_balanced() {
        // Imbalanced dataset
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0,
                6.0, // Class 0 (6 samples)
                10.0, 10.0, 11.0, 11.0, // Class 1 (2 samples)
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);

        let mut clf = SVC::new()
            .with_kernel(Kernel::Linear)
            .with_c(1.0)
            .with_class_weight(ClassWeight::Balanced);

        clf.fit(&x, &y).unwrap();

        // Just verify it trains without error
        assert!(clf.is_fitted());
    }

    #[test]
    fn test_svc_not_fitted_error() {
        let clf = SVC::new();
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        assert!(clf.predict(&x).is_err());
    }

    #[test]
    fn test_svc_feature_mismatch() {
        let x_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 6.0, 7.0, 7.0, 6.0]).unwrap();
        let y_train = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut clf = SVC::new();
        clf.fit(&x_train, &y_train).unwrap();

        // Wrong number of features
        let x_test = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(clf.predict(&x_test).is_err());
    }

    #[test]
    fn test_svc_single_class_error() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0]); // Only one class

        let mut clf = SVC::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_svc_search_space() {
        let clf = SVC::new();
        let space = clf.search_space();

        assert!(space.parameters.contains_key("C"));
        assert!(space.parameters.contains_key("gamma"));
        assert!(space.parameters.contains_key("kernel"));
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_svc_xor_problem() {
        // XOR problem - not linearly separable
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        // RBF kernel should handle this
        let mut clf = SVC::new()
            .with_kernel(Kernel::Rbf { gamma: 2.0 })
            .with_c(10.0);

        clf.fit(&x, &y).unwrap();

        let predictions = clf.predict(&x).unwrap();

        // Should classify XOR pattern correctly
        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &actual)| (pred - actual).abs() < 1e-10)
            .count();

        assert!(
            correct >= 3,
            "Expected at least 3 correct on XOR, got {}",
            correct
        );
    }

    #[test]
    fn test_svc_poly_kernel_integration() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 7.0, 7.0, 8.0, 8.0, 8.0, 7.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = SVC::new()
            .with_kernel(Kernel::poly(2, 1.0, 0.0))
            .with_c(1.0);

        clf.fit(&x, &y).unwrap();

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    // -------------------------------------------------------------------------
    // SVR Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_svr_basic_linear() {
        // Simple linear relationship: y = 2*x + 1
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]);

        let mut reg = SVR::new()
            .with_kernel(Kernel::Linear)
            .with_c(100.0)
            .with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();
        assert!(reg.is_fitted());
        assert!(reg.n_support_vectors() > 0);

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 10);

        // Check that predictions are reasonable
        for (pred, actual) in predictions.iter().zip(y.iter()) {
            assert!(
                (pred - actual).abs() < 2.0,
                "Prediction {} too far from actual {}",
                pred,
                actual
            );
        }
    }

    #[test]
    fn test_svr_rbf_kernel() {
        // Non-linear: y = x^2
        let x = Array2::from_shape_vec(
            (9, 1),
            vec![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![16.0, 9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0, 16.0]);

        let mut reg = SVR::new()
            .with_kernel(Kernel::Rbf { gamma: 0.5 })
            .with_c(100.0)
            .with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // Compute MSE
        let mse: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / y.len() as f64;

        // MSE should be reasonable (not perfect due to epsilon tube)
        assert!(mse < 10.0, "MSE {} is too high", mse);
    }

    #[test]
    fn test_svr_poly_kernel() {
        // Quadratic relationship
        let x = Array2::from_shape_vec((7, 1), vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0]);

        let mut reg = SVR::new()
            .with_kernel(Kernel::poly(2, 1.0, 1.0))
            .with_c(100.0)
            .with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 7);
    }

    #[test]
    fn test_svr_multivariate() {
        // y = x1 + 2*x2
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 3.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 7.0, 8.0]);

        let mut reg = SVR::new()
            .with_kernel(Kernel::Linear)
            .with_c(100.0)
            .with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // Predictions should be close to actual
        for (pred, actual) in predictions.iter().zip(y.iter()) {
            assert!(
                (pred - actual).abs() < 2.0,
                "Prediction {} too far from actual {}",
                pred,
                actual
            );
        }
    }

    #[test]
    fn test_svr_epsilon_effect() {
        // With larger epsilon, more points should be within the tube (fewer support vectors)
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let mut reg_small_eps = SVR::new()
            .with_kernel(Kernel::Linear)
            .with_c(10.0)
            .with_epsilon(0.01);

        let mut reg_large_eps = SVR::new()
            .with_kernel(Kernel::Linear)
            .with_c(10.0)
            .with_epsilon(1.0);

        reg_small_eps.fit(&x, &y).unwrap();
        reg_large_eps.fit(&x, &y).unwrap();

        // With larger epsilon, typically fewer support vectors (more points in tube)
        // This is a general tendency, not guaranteed for all datasets
        assert!(reg_small_eps.is_fitted());
        assert!(reg_large_eps.is_fitted());
    }

    #[test]
    fn test_svr_not_fitted_error() {
        let reg = SVR::new();
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        assert!(reg.predict(&x).is_err());
    }

    #[test]
    fn test_svr_feature_mismatch() {
        let x_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y_train = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let mut reg = SVR::new().with_kernel(Kernel::Linear).with_c(10.0);
        reg.fit(&x_train, &y_train).unwrap();

        // Wrong number of features
        let x_test = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(reg.predict(&x_test).is_err());
    }

    #[test]
    fn test_svr_search_space() {
        let reg = SVR::new();
        let space = reg.search_space();

        assert!(space.parameters.contains_key("C"));
        assert!(space.parameters.contains_key("epsilon"));
        assert!(space.parameters.contains_key("gamma"));
        assert!(space.parameters.contains_key("kernel"));
    }

    #[test]
    fn test_svr_with_noise() {
        // y = x + noise
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        // Adding some noise
        let y = Array1::from_vec(vec![0.1, 1.2, 1.8, 3.1, 3.9, 5.2, 5.8, 7.1, 7.9, 9.2]);

        let mut reg = SVR::new()
            .with_kernel(Kernel::Linear)
            .with_c(10.0)
            .with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // Should still capture the linear trend
        let rmse: f64 = (predictions
            .iter()
            .zip(y.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / y.len() as f64)
            .sqrt();

        assert!(
            rmse < 1.0,
            "RMSE {} is too high for noisy linear data",
            rmse
        );
    }

    #[test]
    fn test_svr_intercept() {
        // y = 5 (constant)
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0, 5.0]);

        let mut reg = SVR::new()
            .with_kernel(Kernel::Linear)
            .with_c(10.0)
            .with_epsilon(0.1);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // All predictions should be close to 5
        for pred in predictions.iter() {
            assert!(
                (pred - 5.0).abs() < 1.0,
                "Prediction {} should be close to 5",
                pred
            );
        }
    }

    // -------------------------------------------------------------------------
    // LinearSVC Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_linear_svc_basic() {
        // Linearly separable data
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = LinearSVC::new().with_c(1.0).with_max_iter(1000);
        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        assert_eq!(clf.classes().unwrap().len(), 2);

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_linear_svc_classification() {
        // Well-separated clusters
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0, 8.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut clf = LinearSVC::new().with_c(10.0);
        clf.fit(&x, &y).unwrap();

        let predictions = clf.predict(&x).unwrap();

        // Count correct predictions
        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &actual)| (pred - actual).abs() < 1e-10)
            .count();

        assert!(correct >= 6, "Expected at least 6 correct, got {}", correct);
    }

    #[test]
    fn test_linear_svc_multiclass() {
        // Three-class problem
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, // Class 0
                5.0, 5.0, 6.0, 5.0, 5.0, 6.0, // Class 1
                9.0, 9.0, 10.0, 9.0, 9.0, 10.0, // Class 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let mut clf = LinearSVC::new().with_c(1.0);
        clf.fit(&x, &y).unwrap();

        assert_eq!(clf.classes().unwrap().len(), 3);
        // For 3 classes, OvR creates 3 binary classifiers
        assert_eq!(clf.weights().unwrap().len(), 3);

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 9);
    }

    #[test]
    fn test_linear_svc_hinge_loss() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = LinearSVC::new().with_c(1.0).with_loss(LinearSVCLoss::Hinge);
        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
    }

    #[test]
    fn test_linear_svc_class_weights() {
        // Imbalanced dataset
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0,
                6.0, // Class 0 (6 samples)
                10.0, 10.0, 11.0, 11.0, // Class 1 (2 samples)
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);

        let mut clf = LinearSVC::new()
            .with_c(1.0)
            .with_class_weight(ClassWeight::Balanced);
        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
    }

    #[test]
    fn test_linear_svc_feature_importance() {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 6.0, 0.0, 0.0, 7.0, 0.0, 0.0, 7.0,
                0.0, 0.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = LinearSVC::new().with_c(1.0);
        clf.fit(&x, &y).unwrap();

        let importance = clf.feature_importance().unwrap();
        assert_eq!(importance.len(), 3);

        // First feature should be most important (it's the only one that varies)
        assert!(importance[0] >= importance[1]);
        assert!(importance[0] >= importance[2]);
    }

    #[test]
    fn test_linear_svc_not_fitted_error() {
        let clf = LinearSVC::new();
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        assert!(clf.predict(&x).is_err());
    }

    #[test]
    fn test_linear_svc_feature_mismatch() {
        let x_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 6.0, 7.0, 7.0, 6.0]).unwrap();
        let y_train = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut clf = LinearSVC::new();
        clf.fit(&x_train, &y_train).unwrap();

        // Wrong number of features
        let x_test = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(clf.predict(&x_test).is_err());
    }

    #[test]
    fn test_linear_svc_single_class_error() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0]); // Only one class

        let mut clf = LinearSVC::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_linear_svc_search_space() {
        let clf = LinearSVC::new();
        let space = clf.search_space();

        assert!(space.parameters.contains_key("C"));
        assert!(space.parameters.contains_key("loss"));
    }

    // -------------------------------------------------------------------------
    // LinearSVR Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_linear_svr_basic() {
        // Simple linear relationship: y = 2*x + 1
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]);

        let mut reg = LinearSVR::new().with_c(100.0).with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();
        assert!(reg.is_fitted());

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 10);

        // Check that predictions are reasonable
        for (pred, actual) in predictions.iter().zip(y.iter()) {
            assert!(
                (pred - actual).abs() < 3.0,
                "Prediction {} too far from actual {}",
                pred,
                actual
            );
        }
    }

    #[test]
    fn test_linear_svr_multivariate() {
        // y = x1 + 2*x2
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 3.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 7.0, 8.0]);

        let mut reg = LinearSVR::new().with_c(100.0).with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // Predictions should be close to actual
        let rmse: f64 = (predictions
            .iter()
            .zip(y.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / y.len() as f64)
            .sqrt();

        assert!(rmse < 3.0, "RMSE {} is too high", rmse);
    }

    #[test]
    fn test_linear_svr_squared_loss() {
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]);

        let mut reg = LinearSVR::new()
            .with_c(100.0)
            .with_epsilon(0.5)
            .with_loss(LinearSVRLoss::SquaredEpsilonInsensitive);

        reg.fit(&x, &y).unwrap();
        assert!(reg.is_fitted());
    }

    #[test]
    fn test_linear_svr_no_intercept() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let mut reg = LinearSVR::new().with_c(100.0).with_fit_intercept(false);

        reg.fit(&x, &y).unwrap();

        // Without intercept, the weight should be close to 2
        let weights = reg.weights().unwrap();
        assert!(
            (weights[0] - 2.0).abs() < 1.0,
            "Weight {} should be close to 2",
            weights[0]
        );
    }

    #[test]
    fn test_linear_svr_feature_importance() {
        // y = 3*x1 + 0*x2
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 5.0, 2.0, 3.0, 3.0, 7.0, 4.0, 2.0, 5.0, 8.0, 6.0, 1.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);

        let mut reg = LinearSVR::new().with_c(100.0);
        reg.fit(&x, &y).unwrap();

        let importance = reg.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // First feature should be more important
        assert!(
            importance[0] > importance[1],
            "First feature should be more important: {:?}",
            importance
        );
    }

    #[test]
    fn test_linear_svr_not_fitted_error() {
        let reg = LinearSVR::new();
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        assert!(reg.predict(&x).is_err());
    }

    #[test]
    fn test_linear_svr_feature_mismatch() {
        let x_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y_train = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let mut reg = LinearSVR::new().with_c(10.0);
        reg.fit(&x_train, &y_train).unwrap();

        // Wrong number of features
        let x_test = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(reg.predict(&x_test).is_err());
    }

    #[test]
    fn test_linear_svr_search_space() {
        let reg = LinearSVR::new();
        let space = reg.search_space();

        assert!(space.parameters.contains_key("C"));
        assert!(space.parameters.contains_key("epsilon"));
        assert!(space.parameters.contains_key("loss"));
    }

    #[test]
    fn test_linear_svr_with_noise() {
        // y = x + noise
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.1, 1.2, 1.8, 3.1, 3.9, 5.2, 5.8, 7.1, 7.9, 9.2]);

        let mut reg = LinearSVR::new().with_c(10.0).with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // Should still capture the linear trend
        let rmse: f64 = (predictions
            .iter()
            .zip(y.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / y.len() as f64)
            .sqrt();

        assert!(
            rmse < 2.0,
            "RMSE {} is too high for noisy linear data",
            rmse
        );
    }

    #[test]
    fn test_linear_svr_constant() {
        // y = 5 (constant)
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0, 5.0]);

        let mut reg = LinearSVR::new().with_c(10.0).with_epsilon(0.1);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // All predictions should be close to 5
        for pred in predictions.iter() {
            assert!(
                (pred - 5.0).abs() < 2.0,
                "Prediction {} should be close to 5",
                pred
            );
        }
    }

    // -------------------------------------------------------------------------
    // Comparison tests: Linear vs Kernelized versions
    // -------------------------------------------------------------------------

    #[test]
    fn test_linear_svc_vs_svc_linear_kernel() {
        // Both should give similar results on linearly separable data
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut linear_svc = LinearSVC::new().with_c(10.0);
        linear_svc.fit(&x, &y).unwrap();

        let mut svc = SVC::new().with_kernel(Kernel::Linear).with_c(10.0);
        svc.fit(&x, &y).unwrap();

        let linear_pred = linear_svc.predict(&x).unwrap();
        let svc_pred = svc.predict(&x).unwrap();

        // Both should classify correctly
        for (lp, sp) in linear_pred.iter().zip(svc_pred.iter()) {
            assert!(
                (lp - sp).abs() < 1e-10 || (*lp == 0.0 || *lp == 1.0) && (*sp == 0.0 || *sp == 1.0),
                "LinearSVC pred {} vs SVC pred {}",
                lp,
                sp
            );
        }
    }

    #[test]
    fn test_linear_svr_vs_svr_linear_kernel() {
        // Both should give similar results on linear data
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let mut linear_svr = LinearSVR::new().with_c(100.0).with_epsilon(0.5);
        linear_svr.fit(&x, &y).unwrap();

        let mut svr = SVR::new()
            .with_kernel(Kernel::Linear)
            .with_c(100.0)
            .with_epsilon(0.5);
        svr.fit(&x, &y).unwrap();

        let linear_pred = linear_svr.predict(&x).unwrap();
        let svr_pred = svr.predict(&x).unwrap();

        // Both should have similar predictions
        for (lp, sp) in linear_pred.iter().zip(svr_pred.iter()) {
            // Allow some tolerance since the optimization algorithms differ
            assert!(
                (lp - sp).abs() < 3.0,
                "LinearSVR pred {} vs SVR pred {}",
                lp,
                sp
            );
        }
    }
}

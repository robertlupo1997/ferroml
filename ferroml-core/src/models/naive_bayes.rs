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

use crate::hpo::SearchSpace;
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, ClassWeight, Model,
    PredictionInterval, ProbabilisticModel,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Gaussian Naive Bayes classifier
///
/// Implements the Gaussian Naive Bayes algorithm for classification.
/// The likelihood of features is assumed to be Gaussian:
///
/// P(x_i | y) = (1 / sqrt(2π σ²_y)) * exp(-(x_i - μ_y)² / (2σ²_y))
///
/// ## Assumptions
///
/// 1. **Feature independence**: Features are conditionally independent given the class
/// 2. **Gaussian distribution**: Each feature follows a Gaussian distribution per class
///
/// ## Incremental Learning
///
/// Supports `partial_fit` for online learning, updating mean/variance estimates
/// incrementally using Welford's algorithm for numerical stability.
///
/// ## Variance Smoothing
///
/// Adds a small value (`var_smoothing * max(variance)`) to all variances to
/// prevent division by zero and improve numerical stability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianNB {
    /// Variance smoothing: portion of largest variance added to all variances
    /// for numerical stability. Default: 1e-9
    pub var_smoothing: f64,
    /// User-specified class priors. If None, priors are estimated from data.
    pub priors: Option<Array1<f64>>,
    /// Class weights for handling imbalanced datasets
    pub class_weight: ClassWeight,

    // Fitted parameters (None before fit)
    /// Per-class mean for each feature: shape (n_classes, n_features)
    theta: Option<Array2<f64>>,
    /// Per-class variance for each feature: shape (n_classes, n_features)
    var: Option<Array2<f64>>,
    /// Actual variance used (with smoothing applied): shape (n_classes, n_features)
    var_smoothed: Option<Array2<f64>>,
    /// Class priors: shape (n_classes,)
    class_prior: Option<Array1<f64>>,
    /// Unique class labels
    classes: Option<Array1<f64>>,
    /// Number of samples per class (for incremental learning)
    class_count: Option<Array1<f64>>,
    /// Number of features
    n_features: Option<usize>,
    /// Epsilon: smoothing value computed from largest variance
    epsilon: Option<f64>,
}

impl Default for GaussianNB {
    fn default() -> Self {
        Self::new()
    }
}

impl GaussianNB {
    /// Create a new Gaussian Naive Bayes classifier with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            var_smoothing: 1e-9,
            priors: None,
            class_weight: ClassWeight::Uniform,
            theta: None,
            var: None,
            var_smoothed: None,
            class_prior: None,
            classes: None,
            class_count: None,
            n_features: None,
            epsilon: None,
        }
    }

    /// Set variance smoothing parameter
    ///
    /// Higher values increase numerical stability but may reduce accuracy.
    #[must_use]
    pub fn with_var_smoothing(mut self, var_smoothing: f64) -> Self {
        self.var_smoothing = var_smoothing;
        self
    }

    /// Set class weights for handling imbalanced data
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Set user-specified class priors
    ///
    /// Priors should sum to 1. If not provided, priors are estimated from data.
    #[must_use]
    pub fn with_priors(mut self, priors: Array1<f64>) -> Self {
        self.priors = Some(priors);
        self
    }

    /// Get the class means (theta)
    ///
    /// Shape: (n_classes, n_features)
    #[must_use]
    pub fn theta(&self) -> Option<&Array2<f64>> {
        self.theta.as_ref()
    }

    /// Get the class variances (before smoothing)
    ///
    /// Shape: (n_classes, n_features)
    #[must_use]
    pub fn var(&self) -> Option<&Array2<f64>> {
        self.var.as_ref()
    }

    /// Get the smoothed class variances (actually used for prediction)
    ///
    /// Shape: (n_classes, n_features)
    #[must_use]
    pub fn var_smoothed(&self) -> Option<&Array2<f64>> {
        self.var_smoothed.as_ref()
    }

    /// Get the class priors
    ///
    /// Shape: (n_classes,)
    #[must_use]
    pub fn class_prior(&self) -> Option<&Array1<f64>> {
        self.class_prior.as_ref()
    }

    /// Get the unique class labels
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Get the number of samples seen per class
    #[must_use]
    pub fn class_count(&self) -> Option<&Array1<f64>> {
        self.class_count.as_ref()
    }

    /// Get the smoothing epsilon value
    #[must_use]
    pub fn epsilon(&self) -> Option<f64> {
        self.epsilon
    }

    /// Incremental fit on a batch of samples
    ///
    /// This method allows for online/out-of-core learning by updating the
    /// model parameters incrementally.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - Target values of shape (n_samples,)
    /// * `classes` - All possible classes (required on first call, can be None afterwards)
    ///
    /// # Example
    ///
    /// ```
    /// # use ferroml_core::models::naive_bayes::GaussianNB;
    /// # use ndarray::{Array1, Array2};
    /// # let x1 = Array2::from_shape_vec((4, 2), vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]).unwrap();
    /// # let y1 = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
    /// # let x2 = x1.clone(); let y2 = y1.clone();
    /// # let x3 = x1.clone(); let y3 = y1.clone();
    /// let mut model = GaussianNB::new();
    ///
    /// // First batch - must specify all classes
    /// model.partial_fit(&x1, &y1, Some(vec![0.0, 1.0])).unwrap();
    ///
    /// // Subsequent batches
    /// model.partial_fit(&x2, &y2, None).unwrap();
    /// model.partial_fit(&x3, &y3, None).unwrap();
    /// ```
    pub fn partial_fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        classes: Option<Vec<f64>>,
    ) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_features = x.ncols();

        // Initialize on first call
        if !self.is_fitted() {
            let classes = classes.ok_or_else(|| {
                FerroError::invalid_input("Classes must be provided on first call to partial_fit")
            })?;

            if classes.is_empty() {
                return Err(FerroError::invalid_input("Classes cannot be empty"));
            }

            let n_classes = classes.len();
            let classes_arr = Array1::from_vec(classes);

            self.classes = Some(classes_arr);
            self.n_features = Some(n_features);
            self.theta = Some(Array2::zeros((n_classes, n_features)));
            self.var = Some(Array2::zeros((n_classes, n_features)));
            self.class_count = Some(Array1::zeros(n_classes));
        }

        // Validate feature count matches
        let expected_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;
        if expected_features != n_features {
            return Err(FerroError::shape_mismatch(
                format!("{} features", expected_features),
                format!("{} features", n_features),
            ));
        }

        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;

        // Update statistics for each class using Welford's algorithm
        for (class_idx, &class_label) in classes.iter().enumerate() {
            // Get samples for this class
            let class_mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &yi)| (yi - class_label).abs() < 1e-10)
                .map(|(i, _)| i)
                .collect();

            if class_mask.is_empty() {
                continue;
            }

            let n_new = class_mask.len() as f64;
            let n_old = self
                .class_count
                .as_ref()
                .ok_or_else(|| FerroError::not_fitted("partial_fit"))?[class_idx];
            let n_total = n_old + n_new;

            // Extract samples for this class
            let x_class: Array2<f64> =
                Array2::from_shape_fn((class_mask.len(), n_features), |(i, j)| {
                    x[[class_mask[i], j]]
                });

            // Compute new sample statistics
            let new_mean = x_class
                .mean_axis(Axis(0))
                .expect("class has at least one sample");
            let new_var = if class_mask.len() > 1 {
                compute_variance(&x_class, &new_mean)
            } else {
                Array1::zeros(n_features)
            };

            // Get current statistics
            let theta = self
                .theta
                .as_mut()
                .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;
            let var = self
                .var
                .as_mut()
                .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;
            let class_count = self
                .class_count
                .as_mut()
                .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;

            let old_mean = theta.row(class_idx).to_owned();
            let old_var = var.row(class_idx).to_owned();

            // Combine old and new using parallel axis formula
            // Combined mean: (n_old * old_mean + n_new * new_mean) / n_total
            let combined_mean = if n_old > 0.0 {
                (&old_mean * n_old + &new_mean * n_new) / n_total
            } else {
                new_mean.clone()
            };

            // Combined variance using parallel algorithm
            // var_combined = (n_old * (var_old + d_old^2) + n_new * (var_new + d_new^2)) / n_total
            // where d_old = old_mean - combined_mean, d_new = new_mean - combined_mean
            let combined_var = if n_old > 0.0 {
                let d_old = &old_mean - &combined_mean;
                let d_new = &new_mean - &combined_mean;

                let var_old_contribution = &old_var + &d_old.mapv(|x| x * x);
                let var_new_contribution = &new_var + &d_new.mapv(|x| x * x);

                (&var_old_contribution * n_old + &var_new_contribution * n_new) / n_total
            } else {
                new_var
            };

            // Update stored values
            theta.row_mut(class_idx).assign(&combined_mean);
            var.row_mut(class_idx).assign(&combined_var);
            class_count[class_idx] = n_total;
        }

        // Update priors and smoothed variance
        self.update_priors_and_variance();

        Ok(())
    }

    /// Update class priors and apply variance smoothing
    fn update_priors_and_variance(&mut self) {
        let class_count = self
            .class_count
            .as_ref()
            .expect("class_count set during fit");
        let var = self.var.as_ref().expect("var set during fit");

        // Compute class priors
        let total_samples: f64 = class_count.sum();
        if total_samples > 0.0 {
            if let Some(ref user_priors) = self.priors {
                // Use user-specified priors
                self.class_prior = Some(user_priors.clone());
            } else {
                // Estimate from data
                self.class_prior = Some(class_count / total_samples);
            }
        }

        // Compute epsilon (smoothing value based on largest variance)
        // Floor max_var to prevent epsilon=0 when all variances are zero
        // (e.g., single-sample classes). Matches sklearn behavior.
        let max_var = var.iter().copied().fold(0.0_f64, f64::max).max(1e-300);
        self.epsilon = Some(self.var_smoothing * max_var);

        // Apply variance smoothing
        let epsilon = self.epsilon.expect("epsilon was just computed");
        self.var_smoothed = Some(var.mapv(|v| v + epsilon));
    }

    /// Compute joint log-likelihood for each class
    ///
    /// Returns shape (n_samples, n_classes)
    fn joint_log_likelihood(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.theta, "joint_log_likelihood")?;

        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;
        validate_predict_input(x, n_features)?;

        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;
        let var_smoothed = self
            .var_smoothed
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;
        let class_prior = self
            .class_prior
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;

        let n_samples = x.nrows();
        let n_classes = theta.nrows();

        let mut jll = Array2::zeros((n_samples, n_classes));

        for class_idx in 0..n_classes {
            let mean = theta.row(class_idx);
            let var = var_smoothed.row(class_idx);

            // log P(y=class) + sum_i log P(x_i | y=class)
            // Gaussian log-likelihood:
            // log P(x_i | y) = -0.5 * (log(2π) + log(σ²) + (x - μ)² / σ²)

            let log_prior = class_prior[class_idx].max(1e-300).ln();
            let log_det = var.mapv(|v| v.max(1e-300).ln()).sum(); // sum of log variances
            let const_term =
                -0.5 * (n_features as f64).mul_add((2.0 * std::f64::consts::PI).ln(), log_det);

            for sample_idx in 0..n_samples {
                let xi = x.row(sample_idx);

                // Mahalanobis distance (diagonal covariance)
                let diff = &xi - &mean;
                let mahal: f64 = diff.iter().zip(var.iter()).map(|(&d, &v)| d * d / v).sum();

                jll[[sample_idx, class_idx]] = 0.5f64.mul_add(-mahal, log_prior + const_term);
            }
        }

        Ok(jll)
    }
}

impl Model for GaussianNB {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        // Find unique classes
        let classes = super::get_unique_classes(y);

        if classes.len() < 2 {
            return Err(FerroError::invalid_input(
                "GaussianNB requires at least 2 classes",
            ));
        }

        // Reset state
        self.theta = None;
        self.var = None;
        self.var_smoothed = None;
        self.class_prior = None;
        self.classes = None;
        self.class_count = None;
        self.n_features = None;
        self.epsilon = None;

        // Use partial_fit to do the actual work
        self.partial_fit(x, y, Some(classes.to_vec()))
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let jll = self.joint_log_likelihood(x)?;
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;

        // Find class with maximum log-likelihood for each sample
        let predictions: Array1<f64> = jll
            .rows()
            .into_iter()
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                classes[max_idx]
            })
            .collect();

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.theta.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        // For Naive Bayes, use ratio of between-class variance to within-class variance
        // as a simple feature importance measure (similar to F-ratio)
        let theta = self.theta.as_ref()?;
        let var = self.var.as_ref()?;
        let class_prior = self.class_prior.as_ref()?;

        let n_features = theta.ncols();
        let n_classes = theta.nrows();

        if n_classes < 2 {
            return None;
        }

        // Global mean per feature
        let global_mean: Array1<f64> = (0..n_features)
            .map(|j| (0..n_classes).map(|c| theta[[c, j]] * class_prior[c]).sum())
            .collect();

        // Between-class variance: sum_c prior_c * (theta_c - global_mean)^2
        let between_var: Array1<f64> = (0..n_features)
            .map(|j| {
                (0..n_classes)
                    .map(|c| {
                        let diff = theta[[c, j]] - global_mean[j];
                        class_prior[c] * diff * diff
                    })
                    .sum()
            })
            .collect();

        // Within-class variance: sum_c prior_c * var_c
        let within_var: Array1<f64> = (0..n_features)
            .map(|j| (0..n_classes).map(|c| class_prior[c] * var[[c, j]]).sum())
            .collect();

        // F-ratio: between / within (add small constant to avoid division by zero)
        let importance: Array1<f64> = between_var
            .iter()
            .zip(within_var.iter())
            .map(|(&b, &w)| b / (w + 1e-10))
            .collect();

        Some(importance)
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new().float_log("var_smoothing", 1e-12, 1e-3)
    }

    fn feature_names(&self) -> Option<&[String]> {
        None
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

impl ProbabilisticModel for GaussianNB {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let jll = self.joint_log_likelihood(x)?;

        // Convert log-likelihoods to probabilities using log-sum-exp trick
        let n_samples = jll.nrows();
        let n_classes = jll.ncols();
        let mut probas = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = jll.row(i);

            // Log-sum-exp trick for numerical stability
            let max_ll = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = row.iter().map(|&ll| (ll - max_ll).exp()).sum();
            let log_sum = max_ll + sum_exp.ln();

            for j in 0..n_classes {
                probas[[i, j]] = (jll[[i, j]] - log_sum).exp();
            }
        }

        Ok(probas)
    }

    fn predict_interval(&self, x: &Array2<f64>, level: f64) -> Result<PredictionInterval> {
        // For classification, return the predicted class probabilities as-is
        // with bounds based on confidence level (using simple binomial approximation)
        let probas = self.predict_proba(x)?;
        let predictions = self.predict(x)?;

        let n_samples = x.nrows();
        let z = z_critical(1.0 - (1.0 - level) / 2.0);

        // Get the probability of predicted class for each sample
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_interval"))?;
        let mut pred_probas = Array1::zeros(n_samples);
        let mut lower = Array1::zeros(n_samples);
        let mut upper = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let class_idx = classes
                .iter()
                .position(|&c| (c - predictions[i]).abs() < 1e-10)
                .unwrap_or(0);
            let p = probas[[i, class_idx]];
            pred_probas[i] = p;

            // Simple normal approximation for binomial CI (n=1)
            // This is a rough approximation since we have single prediction
            let se = (p * (1.0 - p)).sqrt().max(0.01); // min SE to avoid degenerate intervals
            lower[i] = (p - z * se).clamp(0.0, 1.0);
            upper[i] = (p + z * se).clamp(0.0, 1.0);
        }

        Ok(PredictionInterval::new(pred_probas, lower, upper, level))
    }
}

// =============================================================================
// Multinomial Naive Bayes
// =============================================================================

/// Multinomial Naive Bayes classifier
///
/// Implements the Multinomial Naive Bayes algorithm for classification.
/// Suitable for discrete features like word counts in text classification.
///
/// The likelihood of features is computed as:
/// P(x_i | y) = (N_yi + alpha) / (N_y + alpha * n_features)
///
/// Where:
/// - N_yi: Count of feature i in class y
/// - N_y: Total count of all features in class y
/// - alpha: Smoothing parameter (Laplace/Lidstone)
/// - n_features: Number of features
///
/// ## Assumptions
///
/// 1. **Feature independence**: Features are conditionally independent given the class
/// 2. **Non-negative counts**: Feature values should be non-negative counts
///
/// ## Incremental Learning
///
/// Supports `partial_fit` for online/out-of-core learning by accumulating
/// feature counts incrementally.
///
/// ## Smoothing
///
/// Uses Laplace smoothing (alpha=1.0) by default to prevent zero probabilities.
/// Set alpha=0 for no smoothing, alpha<1 for Lidstone smoothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultinomialNB {
    /// Additive (Laplace/Lidstone) smoothing parameter. Default: 1.0
    /// alpha=1.0: Laplace smoothing
    /// alpha=0.0: No smoothing (not recommended)
    /// 0 < alpha < 1: Lidstone smoothing
    pub alpha: f64,
    /// Whether to learn class priors from data. If false, uses uniform priors.
    pub fit_prior: bool,
    /// User-specified class priors. If None, priors are estimated from data.
    pub class_prior: Option<Array1<f64>>,
    /// Class weights for handling imbalanced datasets
    pub class_weight: ClassWeight,

    // Fitted parameters (None before fit)
    /// Log probability of each class: shape (n_classes,)
    class_log_prior: Option<Array1<f64>>,
    /// Log probability of features given class: shape (n_classes, n_features)
    feature_log_prob: Option<Array2<f64>>,
    /// Raw feature counts per class: shape (n_classes, n_features)
    feature_count: Option<Array2<f64>>,
    /// Number of samples per class: shape (n_classes,)
    class_count: Option<Array1<f64>>,
    /// Unique class labels
    classes: Option<Array1<f64>>,
    /// Number of features
    n_features: Option<usize>,
}

impl Default for MultinomialNB {
    fn default() -> Self {
        Self::new()
    }
}

impl MultinomialNB {
    /// Create a new Multinomial Naive Bayes classifier with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            fit_prior: true,
            class_prior: None,
            class_weight: ClassWeight::Uniform,
            class_log_prior: None,
            feature_log_prob: None,
            feature_count: None,
            class_count: None,
            classes: None,
            n_features: None,
        }
    }

    /// Set the smoothing parameter (alpha)
    ///
    /// Higher alpha values provide more smoothing.
    /// - alpha=1.0: Laplace smoothing (default)
    /// - alpha=0.0: No smoothing (may cause issues with unseen features)
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set class weights for handling imbalanced data
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Set whether to learn class priors from data
    ///
    /// If false, uses uniform class priors.
    #[must_use]
    pub fn with_fit_prior(mut self, fit_prior: bool) -> Self {
        self.fit_prior = fit_prior;
        self
    }

    /// Set user-specified class priors
    ///
    /// Priors should sum to 1. If provided, overrides fit_prior setting.
    #[must_use]
    pub fn with_class_prior(mut self, priors: Array1<f64>) -> Self {
        self.class_prior = Some(priors);
        self
    }

    /// Get the log probability of each class
    #[must_use]
    pub fn class_log_prior(&self) -> Option<&Array1<f64>> {
        self.class_log_prior.as_ref()
    }

    /// Get the log probability of features given each class
    ///
    /// Shape: (n_classes, n_features)
    #[must_use]
    pub fn feature_log_prob(&self) -> Option<&Array2<f64>> {
        self.feature_log_prob.as_ref()
    }

    /// Get the raw feature counts per class
    ///
    /// Shape: (n_classes, n_features)
    #[must_use]
    pub fn feature_count(&self) -> Option<&Array2<f64>> {
        self.feature_count.as_ref()
    }

    /// Get the number of samples per class
    #[must_use]
    pub fn class_count(&self) -> Option<&Array1<f64>> {
        self.class_count.as_ref()
    }

    /// Get the unique class labels
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Incremental fit on a batch of samples
    ///
    /// This method allows for online/out-of-core learning by updating the
    /// model parameters incrementally.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix of shape (n_samples, n_features) - should contain counts
    /// * `y` - Target values of shape (n_samples,)
    /// * `classes` - All possible classes (required on first call, can be None afterwards)
    pub fn partial_fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        classes: Option<Vec<f64>>,
    ) -> Result<()> {
        validate_fit_input(x, y)?;

        // Validate non-negative values
        if x.iter().any(|&v| v < 0.0) {
            return Err(FerroError::invalid_input(
                "MultinomialNB requires non-negative feature values",
            ));
        }

        let n_features = x.ncols();

        // Initialize on first call
        if !self.is_fitted() {
            let classes = classes.ok_or_else(|| {
                FerroError::invalid_input("Classes must be provided on first call to partial_fit")
            })?;

            if classes.is_empty() {
                return Err(FerroError::invalid_input("Classes cannot be empty"));
            }

            let n_classes = classes.len();
            let classes_arr = Array1::from_vec(classes);

            self.classes = Some(classes_arr);
            self.n_features = Some(n_features);
            self.feature_count = Some(Array2::zeros((n_classes, n_features)));
            self.class_count = Some(Array1::zeros(n_classes));
        }

        // Validate feature count matches
        let expected_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;
        if expected_features != n_features {
            return Err(FerroError::shape_mismatch(
                format!("{} features", expected_features),
                format!("{} features", n_features),
            ));
        }

        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;

        // Update counts for each class
        for (class_idx, &class_label) in classes.iter().enumerate() {
            // Get samples for this class
            let class_mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &yi)| (yi - class_label).abs() < 1e-10)
                .map(|(i, _)| i)
                .collect();

            if class_mask.is_empty() {
                continue;
            }

            let n_class_samples = class_mask.len();

            // Sum feature counts for this class
            let feature_count = self
                .feature_count
                .as_mut()
                .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;
            let class_count = self
                .class_count
                .as_mut()
                .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;

            for &idx in &class_mask {
                for j in 0..n_features {
                    feature_count[[class_idx, j]] += x[[idx, j]];
                }
            }

            class_count[class_idx] += n_class_samples as f64;
        }

        // Update log probabilities
        self.update_log_probabilities();

        Ok(())
    }

    /// Update log probabilities after fitting/partial fitting
    fn update_log_probabilities(&mut self) {
        let feature_count = self
            .feature_count
            .as_ref()
            .expect("feature_count set during fit");
        let class_count = self
            .class_count
            .as_ref()
            .expect("class_count set during fit");
        let n_classes = class_count.len();
        let n_features = self.n_features.expect("n_features set during fit");

        // Compute class log priors
        let total_samples: f64 = class_count.sum();
        if total_samples > 0.0 {
            if let Some(ref user_priors) = self.class_prior {
                // Use user-specified priors
                self.class_log_prior = Some(user_priors.mapv(|p| p.ln()));
            } else if self.fit_prior {
                // Estimate from data
                self.class_log_prior = Some(class_count.mapv(|c| (c / total_samples).ln()));
            } else {
                // Uniform priors
                let uniform = 1.0 / n_classes as f64;
                self.class_log_prior = Some(Array1::from_elem(n_classes, uniform.ln()));
            }
        }

        // Compute feature log probabilities with smoothing
        // P(x_i | y) = (count_yi + alpha) / (total_y + alpha * n_features)
        let mut feature_log_prob = Array2::zeros((n_classes, n_features));

        for class_idx in 0..n_classes {
            // Total count for this class
            let total_count: f64 = feature_count.row(class_idx).sum();
            let denominator = self.alpha.mul_add(n_features as f64, total_count);

            for j in 0..n_features {
                let count = feature_count[[class_idx, j]];
                let prob = (count + self.alpha) / denominator;
                feature_log_prob[[class_idx, j]] = prob.max(1e-300).ln();
            }
        }

        self.feature_log_prob = Some(feature_log_prob);
    }

    /// Compute joint log-likelihood for each class
    ///
    /// Returns shape (n_samples, n_classes)
    fn joint_log_likelihood(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.feature_log_prob, "joint_log_likelihood")?;

        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;
        validate_predict_input(x, n_features)?;

        let feature_log_prob = self
            .feature_log_prob
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;
        let class_log_prior = self
            .class_log_prior
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;

        let n_samples = x.nrows();
        let n_classes = feature_log_prob.nrows();

        let mut jll = Array2::zeros((n_samples, n_classes));

        for class_idx in 0..n_classes {
            let log_prior = class_log_prior[class_idx];
            let log_prob = feature_log_prob.row(class_idx);

            for sample_idx in 0..n_samples {
                // log P(y) + sum_i x_i * log P(x_i | y)
                let xi = x.row(sample_idx);
                let log_likelihood: f64 = xi
                    .iter()
                    .zip(log_prob.iter())
                    .map(|(&x_val, &lp)| x_val * lp)
                    .sum();

                jll[[sample_idx, class_idx]] = log_prior + log_likelihood;
            }
        }

        Ok(jll)
    }
}

impl Model for MultinomialNB {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        // Validate non-negative values
        if x.iter().any(|&v| v < 0.0) {
            return Err(FerroError::invalid_input(
                "MultinomialNB requires non-negative feature values",
            ));
        }

        // Find unique classes
        let classes = super::get_unique_classes(y);

        if classes.len() < 2 {
            return Err(FerroError::invalid_input(
                "MultinomialNB requires at least 2 classes",
            ));
        }

        // Reset state
        self.class_log_prior = None;
        self.feature_log_prob = None;
        self.feature_count = None;
        self.class_count = None;
        self.classes = None;
        self.n_features = None;

        // Use partial_fit to do the actual work
        self.partial_fit(x, y, Some(classes.to_vec()))
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let jll = self.joint_log_likelihood(x)?;
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;

        // Find class with maximum log-likelihood for each sample
        let predictions: Array1<f64> = jll
            .rows()
            .into_iter()
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                classes[max_idx]
            })
            .collect();

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.feature_log_prob.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        // For Multinomial NB, use the absolute deviation of feature log-probs
        // from the mean as a simple importance measure
        let feature_log_prob = self.feature_log_prob.as_ref()?;
        let n_features = feature_log_prob.ncols();

        // Compute variance of log-probs across classes for each feature
        let importance: Array1<f64> = (0..n_features)
            .map(|j| {
                let col: Vec<f64> = feature_log_prob.column(j).iter().copied().collect();
                let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
                col.iter().map(|&v| (v - mean).abs()).sum::<f64>() / col.len() as f64
            })
            .collect();

        Some(importance)
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new().float_log("alpha", 1e-3, 10.0)
    }

    fn feature_names(&self) -> Option<&[String]> {
        None
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

impl ProbabilisticModel for MultinomialNB {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let jll = self.joint_log_likelihood(x)?;

        // Convert log-likelihoods to probabilities using log-sum-exp trick
        let n_samples = jll.nrows();
        let n_classes = jll.ncols();
        let mut probas = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = jll.row(i);

            // Log-sum-exp trick for numerical stability
            let max_ll = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = row.iter().map(|&ll| (ll - max_ll).exp()).sum();
            let log_sum = max_ll + sum_exp.ln();

            for j in 0..n_classes {
                probas[[i, j]] = (jll[[i, j]] - log_sum).exp();
            }
        }

        Ok(probas)
    }

    fn predict_interval(&self, x: &Array2<f64>, level: f64) -> Result<PredictionInterval> {
        let probas = self.predict_proba(x)?;
        let predictions = self.predict(x)?;

        let n_samples = x.nrows();
        let z = z_critical(1.0 - (1.0 - level) / 2.0);

        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_interval"))?;
        let mut pred_probas = Array1::zeros(n_samples);
        let mut lower = Array1::zeros(n_samples);
        let mut upper = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let class_idx = classes
                .iter()
                .position(|&c| (c - predictions[i]).abs() < 1e-10)
                .unwrap_or(0);
            let p = probas[[i, class_idx]];
            pred_probas[i] = p;

            let se = (p * (1.0 - p)).sqrt().max(0.01);
            lower[i] = (p - z * se).clamp(0.0, 1.0);
            upper[i] = (p + z * se).clamp(0.0, 1.0);
        }

        Ok(PredictionInterval::new(pred_probas, lower, upper, level))
    }
}

// =============================================================================
// Bernoulli Naive Bayes
// =============================================================================

/// Bernoulli Naive Bayes classifier
///
/// Implements the Bernoulli Naive Bayes algorithm for classification.
/// Suitable for binary/boolean features like word presence in text classification.
///
/// The likelihood of features is computed as:
/// P(x_i | y) = P(i | y) * x_i + (1 - P(i | y)) * (1 - x_i)
///
/// Where:
/// - P(i | y): Probability of feature i being present in class y
/// - x_i: Binary feature value (0 or 1)
///
/// ## Key Difference from Multinomial
///
/// Unlike MultinomialNB, BernoulliNB explicitly penalizes the non-occurrence of
/// features that are indicative of a class. This makes it particularly suitable
/// for short documents where absence of a word is informative.
///
/// ## Binarization
///
/// By default, features are binarized at threshold 0.0 (i.e., any positive value
/// becomes 1). Set `binarize` to None to disable binarization (expects binary input).
///
/// ## Assumptions
///
/// 1. **Feature independence**: Features are conditionally independent given the class
/// 2. **Binary features**: Each feature is binary (present/absent)
///
/// ## Incremental Learning
///
/// Supports `partial_fit` for online/out-of-core learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BernoulliNB {
    /// Additive (Laplace/Lidstone) smoothing parameter. Default: 1.0
    pub alpha: f64,
    /// Threshold for binarizing features. If None, assumes input is already binary.
    /// Default: Some(0.0) - any value > 0 becomes 1.
    pub binarize: Option<f64>,
    /// Whether to learn class priors from data. If false, uses uniform priors.
    pub fit_prior: bool,
    /// User-specified class priors. If None, priors are estimated from data.
    pub class_prior: Option<Array1<f64>>,
    /// Class weights for handling imbalanced datasets
    pub class_weight: ClassWeight,

    // Fitted parameters (None before fit)
    /// Log probability of each class: shape (n_classes,)
    class_log_prior: Option<Array1<f64>>,
    /// Log probability of feature presence given class: shape (n_classes, n_features)
    feature_log_prob: Option<Array2<f64>>,
    /// Log probability of feature absence given class: shape (n_classes, n_features)
    feature_log_prob_neg: Option<Array2<f64>>,
    /// Count of feature occurrences per class: shape (n_classes, n_features)
    feature_count: Option<Array2<f64>>,
    /// Number of samples per class: shape (n_classes,)
    class_count: Option<Array1<f64>>,
    /// Unique class labels
    classes: Option<Array1<f64>>,
    /// Number of features
    n_features: Option<usize>,
}

impl Default for BernoulliNB {
    fn default() -> Self {
        Self::new()
    }
}

impl BernoulliNB {
    /// Create a new Bernoulli Naive Bayes classifier with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            binarize: Some(0.0),
            fit_prior: true,
            class_prior: None,
            class_weight: ClassWeight::Uniform,
            class_log_prior: None,
            feature_log_prob: None,
            feature_log_prob_neg: None,
            feature_count: None,
            class_count: None,
            classes: None,
            n_features: None,
        }
    }

    /// Set the smoothing parameter (alpha)
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set class weights for handling imbalanced data
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Set the binarization threshold
    ///
    /// Features with values > threshold become 1, others become 0.
    /// Set to None to disable binarization (expects binary input).
    #[must_use]
    pub fn with_binarize(mut self, threshold: Option<f64>) -> Self {
        self.binarize = threshold;
        self
    }

    /// Set whether to learn class priors from data
    #[must_use]
    pub fn with_fit_prior(mut self, fit_prior: bool) -> Self {
        self.fit_prior = fit_prior;
        self
    }

    /// Set user-specified class priors
    #[must_use]
    pub fn with_class_prior(mut self, priors: Array1<f64>) -> Self {
        self.class_prior = Some(priors);
        self
    }

    /// Get the log probability of each class
    #[must_use]
    pub fn class_log_prior(&self) -> Option<&Array1<f64>> {
        self.class_log_prior.as_ref()
    }

    /// Get the log probability of feature presence given each class
    ///
    /// Shape: (n_classes, n_features)
    #[must_use]
    pub fn feature_log_prob(&self) -> Option<&Array2<f64>> {
        self.feature_log_prob.as_ref()
    }

    /// Get the raw feature counts per class
    ///
    /// Shape: (n_classes, n_features)
    #[must_use]
    pub fn feature_count(&self) -> Option<&Array2<f64>> {
        self.feature_count.as_ref()
    }

    /// Get the number of samples per class
    #[must_use]
    pub fn class_count(&self) -> Option<&Array1<f64>> {
        self.class_count.as_ref()
    }

    /// Get the unique class labels
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Binarize the input matrix
    fn binarize_input(&self, x: &Array2<f64>) -> Array2<f64> {
        if let Some(threshold) = self.binarize {
            x.mapv(|v| if v > threshold { 1.0 } else { 0.0 })
        } else {
            x.clone()
        }
    }

    /// Incremental fit on a batch of samples
    pub fn partial_fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        classes: Option<Vec<f64>>,
    ) -> Result<()> {
        validate_fit_input(x, y)?;

        let x_bin = self.binarize_input(x);
        let n_features = x_bin.ncols();

        // Initialize on first call
        if !self.is_fitted() {
            let classes = classes.ok_or_else(|| {
                FerroError::invalid_input("Classes must be provided on first call to partial_fit")
            })?;

            if classes.is_empty() {
                return Err(FerroError::invalid_input("Classes cannot be empty"));
            }

            let n_classes = classes.len();
            let classes_arr = Array1::from_vec(classes);

            self.classes = Some(classes_arr);
            self.n_features = Some(n_features);
            self.feature_count = Some(Array2::zeros((n_classes, n_features)));
            self.class_count = Some(Array1::zeros(n_classes));
        }

        // Validate feature count matches
        let expected_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;
        if expected_features != n_features {
            return Err(FerroError::shape_mismatch(
                format!("{} features", expected_features),
                format!("{} features", n_features),
            ));
        }

        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;

        // Update counts for each class
        for (class_idx, &class_label) in classes.iter().enumerate() {
            // Get samples for this class
            let class_mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &yi)| (yi - class_label).abs() < 1e-10)
                .map(|(i, _)| i)
                .collect();

            if class_mask.is_empty() {
                continue;
            }

            let n_class_samples = class_mask.len();

            // Sum feature occurrences for this class
            let feature_count = self
                .feature_count
                .as_mut()
                .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;
            let class_count = self
                .class_count
                .as_mut()
                .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;

            for &idx in &class_mask {
                for j in 0..n_features {
                    feature_count[[class_idx, j]] += x_bin[[idx, j]];
                }
            }

            class_count[class_idx] += n_class_samples as f64;
        }

        // Update log probabilities
        self.update_log_probabilities();

        Ok(())
    }

    /// Update log probabilities after fitting/partial fitting
    fn update_log_probabilities(&mut self) {
        let feature_count = self
            .feature_count
            .as_ref()
            .expect("feature_count set during fit");
        let class_count = self
            .class_count
            .as_ref()
            .expect("class_count set during fit");
        let n_classes = class_count.len();
        let n_features = self.n_features.expect("n_features set during fit");

        // Compute class log priors
        let total_samples: f64 = class_count.sum();
        if total_samples > 0.0 {
            if let Some(ref user_priors) = self.class_prior {
                self.class_log_prior = Some(user_priors.mapv(|p| p.ln()));
            } else if self.fit_prior {
                self.class_log_prior = Some(class_count.mapv(|c| (c / total_samples).ln()));
            } else {
                let uniform = 1.0 / n_classes as f64;
                self.class_log_prior = Some(Array1::from_elem(n_classes, uniform.ln()));
            }
        }

        // Compute feature log probabilities with smoothing
        // P(x_i=1 | y) = (count + alpha) / (n_y + 2*alpha)
        let mut feature_log_prob = Array2::zeros((n_classes, n_features));
        let mut feature_log_prob_neg = Array2::zeros((n_classes, n_features));

        for class_idx in 0..n_classes {
            let n_class = class_count[class_idx];
            let denominator = 2.0f64.mul_add(self.alpha, n_class);

            for j in 0..n_features {
                let count = feature_count[[class_idx, j]];
                let prob = (count + self.alpha) / denominator;
                let prob_neg = 1.0 - prob;

                feature_log_prob[[class_idx, j]] = prob.max(1e-300).ln();
                feature_log_prob_neg[[class_idx, j]] = prob_neg.max(1e-300).ln();
            }
        }

        self.feature_log_prob = Some(feature_log_prob);
        self.feature_log_prob_neg = Some(feature_log_prob_neg);
    }

    /// Compute joint log-likelihood for each class
    fn joint_log_likelihood(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.feature_log_prob, "joint_log_likelihood")?;

        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;
        validate_predict_input(x, n_features)?;

        let x_bin = self.binarize_input(x);

        let feature_log_prob = self
            .feature_log_prob
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;
        let feature_log_prob_neg = self
            .feature_log_prob_neg
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;
        let class_log_prior = self
            .class_log_prior
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;

        let n_samples = x_bin.nrows();
        let n_classes = feature_log_prob.nrows();

        let mut jll = Array2::zeros((n_samples, n_classes));

        for class_idx in 0..n_classes {
            let log_prior = class_log_prior[class_idx];
            let log_prob = feature_log_prob.row(class_idx);
            let log_prob_neg = feature_log_prob_neg.row(class_idx);

            for sample_idx in 0..n_samples {
                // log P(y) + sum_i [x_i * log P(x_i=1|y) + (1-x_i) * log P(x_i=0|y)]
                let xi = x_bin.row(sample_idx);
                let log_likelihood: f64 = xi
                    .iter()
                    .zip(log_prob.iter().zip(log_prob_neg.iter()))
                    .map(|(&x_val, (&lp, &lp_neg))| x_val.mul_add(lp, (1.0 - x_val) * lp_neg))
                    .sum();

                jll[[sample_idx, class_idx]] = log_prior + log_likelihood;
            }
        }

        Ok(jll)
    }
}

impl Model for BernoulliNB {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        // Find unique classes
        let classes = super::get_unique_classes(y);

        if classes.len() < 2 {
            return Err(FerroError::invalid_input(
                "BernoulliNB requires at least 2 classes",
            ));
        }

        // Reset state
        self.class_log_prior = None;
        self.feature_log_prob = None;
        self.feature_log_prob_neg = None;
        self.feature_count = None;
        self.class_count = None;
        self.classes = None;
        self.n_features = None;

        // Use partial_fit to do the actual work
        self.partial_fit(x, y, Some(classes.to_vec()))
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let jll = self.joint_log_likelihood(x)?;
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;

        let predictions: Array1<f64> = jll
            .rows()
            .into_iter()
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                classes[max_idx]
            })
            .collect();

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.feature_log_prob.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        let feature_log_prob = self.feature_log_prob.as_ref()?;
        let n_features = feature_log_prob.ncols();

        // Use variance of log-probs across classes as importance
        let importance: Array1<f64> = (0..n_features)
            .map(|j| {
                let col: Vec<f64> = feature_log_prob.column(j).iter().copied().collect();
                let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
                col.iter().map(|&v| (v - mean).abs()).sum::<f64>() / col.len() as f64
            })
            .collect();

        Some(importance)
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .float_log("alpha", 1e-3, 10.0)
            .float("binarize", -1.0, 1.0) // -1 means None
    }

    fn feature_names(&self) -> Option<&[String]> {
        None
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

impl ProbabilisticModel for BernoulliNB {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let jll = self.joint_log_likelihood(x)?;

        let n_samples = jll.nrows();
        let n_classes = jll.ncols();
        let mut probas = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = jll.row(i);
            let max_ll = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = row.iter().map(|&ll| (ll - max_ll).exp()).sum();
            let log_sum = max_ll + sum_exp.ln();

            for j in 0..n_classes {
                probas[[i, j]] = (jll[[i, j]] - log_sum).exp();
            }
        }

        Ok(probas)
    }

    fn predict_interval(&self, x: &Array2<f64>, level: f64) -> Result<PredictionInterval> {
        let probas = self.predict_proba(x)?;
        let predictions = self.predict(x)?;

        let n_samples = x.nrows();
        let z = z_critical(1.0 - (1.0 - level) / 2.0);

        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_interval"))?;
        let mut pred_probas = Array1::zeros(n_samples);
        let mut lower = Array1::zeros(n_samples);
        let mut upper = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let class_idx = classes
                .iter()
                .position(|&c| (c - predictions[i]).abs() < 1e-10)
                .unwrap_or(0);
            let p = probas[[i, class_idx]];
            pred_probas[i] = p;

            let se = (p * (1.0 - p)).sqrt().max(0.01);
            lower[i] = (p - z * se).clamp(0.0, 1.0);
            upper[i] = (p + z * se).clamp(0.0, 1.0);
        }

        Ok(PredictionInterval::new(pred_probas, lower, upper, level))
    }
}

// =============================================================================
// Categorical Naive Bayes
// =============================================================================

/// Categorical Naive Bayes classifier
///
/// Implements the Categorical Naive Bayes algorithm for classification.
/// Each feature is assumed to be generated from a categorical distribution.
/// This classifier is suitable for classification with discrete features
/// that are categorically distributed (e.g., encoded categorical variables).
///
/// The likelihood of features is computed as:
/// P(x_j = c | y = k) = (N_{k,j,c} + alpha) / (N_k + alpha * n_categories_j)
///
/// Where:
/// - N_{k,j,c}: Count of feature j having value c in class k
/// - N_k: Total count of class k
/// - n_categories_j: Number of unique categories for feature j
/// - alpha: Smoothing parameter (Laplace/Lidstone)
///
/// ## Assumptions
///
/// 1. **Feature independence**: Features are conditionally independent given the class
/// 2. **Categorical features**: Each feature takes on discrete categorical values
///
/// ## Incremental Learning
///
/// Supports `partial_fit` for online/out-of-core learning by accumulating
/// category counts incrementally. New categories encountered in subsequent
/// batches are handled by expanding the count arrays.
///
/// ## Smoothing
///
/// Uses Laplace smoothing (alpha=1.0) by default to prevent zero probabilities
/// for unseen category-class combinations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalNB {
    /// Additive (Laplace/Lidstone) smoothing parameter. Default: 1.0
    pub alpha: f64,
    /// Whether to learn class priors from data. If false, uses uniform priors.
    pub fit_prior: bool,
    /// User-specified class priors. If None, priors are estimated from data.
    pub class_prior: Option<Array1<f64>>,

    // Fitted parameters (None before fit)
    /// Unique class labels
    classes: Option<Array1<f64>>,
    /// Number of samples per class: shape (n_classes,)
    class_count: Option<Array1<f64>>,
    /// Log probability of each class: shape (n_classes,)
    class_log_prior: Option<Array1<f64>>,
    /// Per feature: counts of each category per class.
    /// category_count[j] has shape (n_classes, n_categories_j)
    category_count: Option<Vec<Array2<f64>>>,
    /// Per feature: log probability of each category given class.
    /// feature_log_prob[j] has shape (n_classes, n_categories_j)
    feature_log_prob: Option<Vec<Array2<f64>>>,
    /// Sorted unique category values per feature
    feature_categories: Option<Vec<Vec<f64>>>,
    /// Number of features
    n_features: Option<usize>,
}

impl Default for CategoricalNB {
    fn default() -> Self {
        Self::new()
    }
}

impl CategoricalNB {
    /// Create a new Categorical Naive Bayes classifier with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            fit_prior: true,
            class_prior: None,
            classes: None,
            class_count: None,
            class_log_prior: None,
            category_count: None,
            feature_log_prob: None,
            feature_categories: None,
            n_features: None,
        }
    }

    /// Set the smoothing parameter (alpha)
    ///
    /// Higher alpha values provide more smoothing.
    /// - alpha=1.0: Laplace smoothing (default)
    /// - alpha=0.0: No smoothing (may cause issues with unseen categories)
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set whether to learn class priors from data
    ///
    /// If false, uses uniform class priors.
    #[must_use]
    pub fn with_fit_prior(mut self, fit_prior: bool) -> Self {
        self.fit_prior = fit_prior;
        self
    }

    /// Set user-specified class priors
    ///
    /// Priors should sum to 1. If provided, overrides fit_prior setting.
    #[must_use]
    pub fn with_class_prior(mut self, priors: Array1<f64>) -> Self {
        self.class_prior = Some(priors);
        self
    }

    /// Get the log probability of each class
    #[must_use]
    pub fn class_log_prior(&self) -> Option<&Array1<f64>> {
        self.class_log_prior.as_ref()
    }

    /// Get the per-feature category counts per class
    ///
    /// Returns a Vec where element j has shape (n_classes, n_categories_j)
    #[must_use]
    pub fn category_count(&self) -> Option<&Vec<Array2<f64>>> {
        self.category_count.as_ref()
    }

    /// Get the per-feature log probabilities of categories given class
    ///
    /// Returns a Vec where element j has shape (n_classes, n_categories_j)
    #[must_use]
    pub fn feature_log_prob(&self) -> Option<&Vec<Array2<f64>>> {
        self.feature_log_prob.as_ref()
    }

    /// Get the sorted unique category values per feature
    #[must_use]
    pub fn feature_categories(&self) -> Option<&Vec<Vec<f64>>> {
        self.feature_categories.as_ref()
    }

    /// Get the number of samples per class
    #[must_use]
    pub fn class_count(&self) -> Option<&Array1<f64>> {
        self.class_count.as_ref()
    }

    /// Get the unique class labels
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Find sorted unique values in a slice, using epsilon-aware comparison
    fn unique_sorted(values: &[f64]) -> Vec<f64> {
        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        sorted
    }

    /// Find the index of a category value in the sorted category list
    fn category_index(categories: &[f64], value: f64) -> Option<usize> {
        categories.iter().position(|&c| (c - value).abs() < 1e-10)
    }

    /// Incremental fit on a batch of samples
    ///
    /// This method allows for online/out-of-core learning by updating the
    /// model parameters incrementally.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix of shape (n_samples, n_features) - each value is a category
    /// * `y` - Target values of shape (n_samples,)
    /// * `classes` - All possible classes (required on first call, can be None afterwards)
    pub fn partial_fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        classes: Option<Vec<f64>>,
    ) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_features = x.ncols();

        // Initialize on first call
        if !self.is_fitted() {
            let classes = classes.ok_or_else(|| {
                FerroError::invalid_input("Classes must be provided on first call to partial_fit")
            })?;

            if classes.is_empty() {
                return Err(FerroError::invalid_input("Classes cannot be empty"));
            }

            let n_classes = classes.len();
            let classes_arr = Array1::from_vec(classes);

            self.classes = Some(classes_arr);
            self.n_features = Some(n_features);
            self.class_count = Some(Array1::zeros(n_classes));

            // Initialize per-feature category tracking
            // Discover categories from this first batch
            let mut feature_categories = Vec::with_capacity(n_features);
            let mut category_count = Vec::with_capacity(n_features);

            for j in 0..n_features {
                let col_values: Vec<f64> = x.column(j).iter().copied().collect();
                let cats = Self::unique_sorted(&col_values);
                let n_cats = cats.len();
                category_count.push(Array2::zeros((n_classes, n_cats)));
                feature_categories.push(cats);
            }

            self.feature_categories = Some(feature_categories);
            self.category_count = Some(category_count);
        }

        // Validate feature count matches
        let expected_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;
        if expected_features != n_features {
            return Err(FerroError::shape_mismatch(
                format!("{} features", expected_features),
                format!("{} features", n_features),
            ));
        }

        // Check for new categories and expand arrays if needed
        {
            let feature_categories = self
                .feature_categories
                .as_mut()
                .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;
            let category_count = self
                .category_count
                .as_mut()
                .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;
            let n_classes = self
                .classes
                .as_ref()
                .ok_or_else(|| FerroError::not_fitted("partial_fit"))?
                .len();

            for j in 0..n_features {
                // Find new categories in this batch
                let col_values: Vec<f64> = x.column(j).iter().copied().collect();
                let batch_cats = Self::unique_sorted(&col_values);

                let mut new_cats = Vec::new();
                for &cat in &batch_cats {
                    if Self::category_index(&feature_categories[j], cat).is_none() {
                        new_cats.push(cat);
                    }
                }

                // Expand if there are new categories
                if !new_cats.is_empty() {
                    // Save old categories before modification
                    let old_cats = feature_categories[j].clone();
                    let old_counts = category_count[j].clone();

                    // Merge and sort
                    feature_categories[j].extend_from_slice(&new_cats);
                    feature_categories[j]
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let new_n_cats = feature_categories[j].len();

                    // Rebuild count array mapping old positions to new positions
                    let mut new_counts = Array2::zeros((n_classes, new_n_cats));
                    for (old_idx, &old_cat_val) in old_cats.iter().enumerate() {
                        if let Some(new_idx) =
                            Self::category_index(&feature_categories[j], old_cat_val)
                        {
                            for class_idx in 0..n_classes {
                                new_counts[[class_idx, new_idx]] = old_counts[[class_idx, old_idx]];
                            }
                        }
                    }

                    category_count[j] = new_counts;
                }
            }
        }

        // Now accumulate counts
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;
        let feature_categories = self
            .feature_categories
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;

        for (class_idx, &class_label) in classes.iter().enumerate() {
            let class_mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &yi)| (yi - class_label).abs() < 1e-10)
                .map(|(i, _)| i)
                .collect();

            if class_mask.is_empty() {
                continue;
            }

            let n_class_samples = class_mask.len();

            let category_count = self
                .category_count
                .as_mut()
                .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;
            let class_count = self
                .class_count
                .as_mut()
                .ok_or_else(|| FerroError::not_fitted("partial_fit"))?;

            for &sample_idx in &class_mask {
                for j in 0..n_features {
                    let value = x[[sample_idx, j]];
                    if let Some(cat_idx) = Self::category_index(&feature_categories[j], value) {
                        category_count[j][[class_idx, cat_idx]] += 1.0;
                    }
                    // If category not found (shouldn't happen after expansion), skip
                }
            }

            class_count[class_idx] += n_class_samples as f64;
        }

        // Update log probabilities
        self.update_log_probabilities();

        Ok(())
    }

    /// Update log probabilities after fitting/partial fitting
    fn update_log_probabilities(&mut self) {
        let class_count = self
            .class_count
            .as_ref()
            .expect("class_count set during fit");
        let category_count = self
            .category_count
            .as_ref()
            .expect("category_count set during fit");
        let n_classes = class_count.len();
        let n_features = self.n_features.expect("n_features set during fit");

        // Compute class log priors
        let total_samples: f64 = class_count.sum();
        if total_samples > 0.0 {
            if let Some(ref user_priors) = self.class_prior {
                self.class_log_prior = Some(user_priors.mapv(|p| p.ln()));
            } else if self.fit_prior {
                self.class_log_prior = Some(class_count.mapv(|c| (c / total_samples).ln()));
            } else {
                let uniform = 1.0 / n_classes as f64;
                self.class_log_prior = Some(Array1::from_elem(n_classes, uniform.ln()));
            }
        }

        // Compute feature log probabilities with smoothing
        // P(x_j = c | y = k) = (N_{k,j,c} + alpha) / (N_k + alpha * n_categories_j)
        let mut feature_log_prob = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let counts = &category_count[j];
            let n_cats = counts.ncols();
            let mut log_prob = Array2::zeros((n_classes, n_cats));

            for class_idx in 0..n_classes {
                let n_k = class_count[class_idx];
                let denominator = self.alpha.mul_add(n_cats as f64, n_k);

                for cat_idx in 0..n_cats {
                    let count = counts[[class_idx, cat_idx]];
                    let prob = (count + self.alpha) / denominator;
                    log_prob[[class_idx, cat_idx]] = prob.ln();
                }
            }

            feature_log_prob.push(log_prob);
        }

        self.feature_log_prob = Some(feature_log_prob);
    }

    /// Compute joint log-likelihood for each class
    ///
    /// Returns shape (n_samples, n_classes)
    fn joint_log_likelihood(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.feature_log_prob, "joint_log_likelihood")?;

        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;
        validate_predict_input(x, n_features)?;

        let feature_log_prob = self
            .feature_log_prob
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;
        let feature_categories = self
            .feature_categories
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;
        let class_log_prior = self
            .class_log_prior
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("joint_log_likelihood"))?;

        let n_samples = x.nrows();
        let n_classes = class_log_prior.len();

        let mut jll = Array2::zeros((n_samples, n_classes));

        for class_idx in 0..n_classes {
            let log_prior = class_log_prior[class_idx];

            for sample_idx in 0..n_samples {
                let mut log_likelihood = log_prior;

                for j in 0..n_features {
                    let value = x[[sample_idx, j]];
                    let log_prob = &feature_log_prob[j];

                    if let Some(cat_idx) = Self::category_index(&feature_categories[j], value) {
                        log_likelihood += log_prob[[class_idx, cat_idx]];
                    } else {
                        // Unseen category: use uniform probability over n_categories + 1
                        // This gives a small but non-zero probability
                        let n_cats = feature_categories[j].len();
                        let n_k = self
                            .class_count
                            .as_ref()
                            .map(|cc| cc[class_idx])
                            .unwrap_or(0.0);
                        let denominator = self.alpha.mul_add((n_cats + 1) as f64, n_k);
                        log_likelihood += (self.alpha / denominator).ln();
                    }
                }

                jll[[sample_idx, class_idx]] = log_likelihood;
            }
        }

        Ok(jll)
    }
}

impl Model for CategoricalNB {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        // Find unique classes
        let classes = super::get_unique_classes(y);

        if classes.len() < 2 {
            return Err(FerroError::invalid_input(
                "CategoricalNB requires at least 2 classes",
            ));
        }

        // Reset state
        self.classes = None;
        self.class_count = None;
        self.class_log_prior = None;
        self.category_count = None;
        self.feature_log_prob = None;
        self.feature_categories = None;
        self.n_features = None;

        // Use partial_fit to do the actual work
        self.partial_fit(x, y, Some(classes.to_vec()))
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let jll = self.joint_log_likelihood(x)?;
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;

        let predictions: Array1<f64> = jll
            .rows()
            .into_iter()
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                classes[max_idx]
            })
            .collect();

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.feature_log_prob.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        // For Categorical NB, use the max variance of log-probs across categories
        // and classes for each feature as an importance measure
        let feature_log_prob = self.feature_log_prob.as_ref()?;
        let n_features = feature_log_prob.len();

        let importance: Array1<f64> = (0..n_features)
            .map(|j| {
                let log_prob = &feature_log_prob[j];
                let n_classes = log_prob.nrows();
                let n_cats = log_prob.ncols();

                if n_classes < 2 || n_cats < 1 {
                    return 0.0;
                }

                // Compute mean absolute deviation of log-probs across classes per category
                let mut total_dev = 0.0;
                for cat_idx in 0..n_cats {
                    let col: Vec<f64> = (0..n_classes).map(|c| log_prob[[c, cat_idx]]).collect();
                    let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
                    total_dev +=
                        col.iter().map(|&v| (v - mean).abs()).sum::<f64>() / col.len() as f64;
                }

                total_dev / n_cats as f64
            })
            .collect();

        Some(importance)
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new().float_log("alpha", 0.01, 10.0)
    }

    fn feature_names(&self) -> Option<&[String]> {
        None
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

impl ProbabilisticModel for CategoricalNB {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let jll = self.joint_log_likelihood(x)?;

        let n_samples = jll.nrows();
        let n_classes = jll.ncols();
        let mut probas = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = jll.row(i);
            let max_ll = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = row.iter().map(|&ll| (ll - max_ll).exp()).sum();
            let log_sum = max_ll + sum_exp.ln();

            for j in 0..n_classes {
                probas[[i, j]] = (jll[[i, j]] - log_sum).exp();
            }
        }

        Ok(probas)
    }

    fn predict_interval(&self, x: &Array2<f64>, level: f64) -> Result<PredictionInterval> {
        let probas = self.predict_proba(x)?;
        let predictions = self.predict(x)?;

        let n_samples = x.nrows();
        let z = z_critical(1.0 - (1.0 - level) / 2.0);

        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_interval"))?;
        let mut pred_probas = Array1::zeros(n_samples);
        let mut lower = Array1::zeros(n_samples);
        let mut upper = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let class_idx = classes
                .iter()
                .position(|&c| (c - predictions[i]).abs() < 1e-10)
                .unwrap_or(0);
            let p = probas[[i, class_idx]];
            pred_probas[i] = p;

            let se = (p * (1.0 - p)).sqrt().max(0.01);
            lower[i] = (p - z * se).clamp(0.0, 1.0);
            upper[i] = (p + z * se).clamp(0.0, 1.0);
        }

        Ok(PredictionInterval::new(pred_probas, lower, upper, level))
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute variance for each column
fn compute_variance(x: &Array2<f64>, mean: &Array1<f64>) -> Array1<f64> {
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
fn z_critical(p: f64) -> f64 {
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

// =============================================================================
// SparseModel Implementations
// =============================================================================

#[cfg(feature = "sparse")]
impl crate::models::traits::SparseModel for GaussianNB {
    fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("{} samples", n_samples),
                format!("{} targets", y.len()),
            ));
        }
        if n_samples == 0 {
            return Err(FerroError::invalid_input("Empty input"));
        }

        let classes = super::get_unique_classes(y);
        if classes.len() < 2 {
            return Err(FerroError::invalid_input(
                "GaussianNB requires at least 2 classes",
            ));
        }

        let n_classes = classes.len();

        // Build class index map
        let class_indices: Vec<Vec<usize>> = classes
            .iter()
            .map(|&c| {
                y.iter()
                    .enumerate()
                    .filter(|(_, &yi)| (yi - c).abs() < 1e-10)
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect();

        // Compute per-class mean and variance from sparse rows
        // For sparse data: most values are 0, so we track sum and sum_sq of nnz
        // and account for implicit zeros.
        let mut theta = Array2::zeros((n_classes, n_features));
        let mut var = Array2::zeros((n_classes, n_features));
        let mut class_count = Array1::zeros(n_classes);

        for (class_idx, indices) in class_indices.iter().enumerate() {
            let n_c = indices.len() as f64;
            if n_c == 0.0 {
                continue;
            }
            class_count[class_idx] = n_c;

            // Accumulate sum and sum_sq per feature using only nnz entries
            let mut sum: Array1<f64> = Array1::zeros(n_features);
            let mut sum_sq: Array1<f64> = Array1::zeros(n_features);

            for &i in indices {
                let row = x.row(i);
                for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                    sum[col] += val;
                    sum_sq[col] += val * val;
                }
            }

            // mean = sum / n_c  (zero entries contribute 0)
            // var = sum_sq / n_c - mean^2  (population variance)
            for j in 0..n_features {
                let mean_j = sum[j] / n_c;
                theta[[class_idx, j]] = mean_j;
                var[[class_idx, j]] = sum_sq[j] / n_c - mean_j * mean_j;
                // Clamp to 0 for floating-point rounding
                if var[[class_idx, j]] < 0.0 {
                    var[[class_idx, j]] = 0.0;
                }
            }
        }

        // Compute epsilon and smoothed variance
        // Floor max_var to prevent epsilon=0 when all variances are zero
        let max_var = var.iter().copied().fold(0.0_f64, f64::max).max(1e-300);
        let epsilon = self.var_smoothing * max_var;
        let var_smoothed = var.mapv(|v| v + epsilon);

        // Compute class priors
        let total_samples: f64 = class_count.sum();
        let class_prior = if let Some(ref user_priors) = self.priors {
            user_priors.clone()
        } else {
            class_count.mapv(|c| c / total_samples)
        };

        self.theta = Some(theta);
        self.var = Some(var);
        self.var_smoothed = Some(var_smoothed);
        self.class_prior = Some(class_prior);
        self.classes = Some(Array1::from_vec(classes.to_vec()));
        self.class_count = Some(class_count);
        self.n_features = Some(n_features);
        self.epsilon = Some(epsilon);

        Ok(())
    }

    fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array1<f64>> {
        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let var_smoothed = self
            .var_smoothed
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let class_prior = self
            .class_prior
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;

        if x.ncols() != n_features {
            return Err(FerroError::shape_mismatch(
                format!("{} features", n_features),
                format!("{} features", x.ncols()),
            ));
        }

        let n_samples = x.nrows();
        let n_classes = theta.nrows();

        // Pre-compute per-class constants:
        // const_term[c] = log_prior[c] - 0.5 * (n_features * log(2pi) + sum_j log(var[c][j]))
        // base_mahal[c] = sum_j (mean[c][j]^2 / var[c][j])  (contribution when x_j = 0)
        let mut const_term = Array1::zeros(n_classes);
        let mut base_mahal = Array1::zeros(n_classes);

        for c in 0..n_classes {
            let log_prior = class_prior[c].ln();
            let mut log_det = 0.0;
            let mut mahal_zero = 0.0;
            for j in 0..n_features {
                let v = var_smoothed[[c, j]];
                log_det += v.ln();
                let m = theta[[c, j]];
                mahal_zero += m * m / v;
            }
            const_term[c] = log_prior
                - 0.5 * ((n_features as f64) * (2.0 * std::f64::consts::PI).ln() + log_det)
                - 0.5 * mahal_zero;
            base_mahal[c] = mahal_zero;
        }

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);
            let nnz_indices = row.indices();
            let nnz_data = row.data();

            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;

            for c in 0..n_classes {
                // Start with the precomputed constant (assumes all x_j = 0)
                // Then correct for nnz features:
                // delta = -0.5 * [(x_j - mean)^2 / var - mean^2 / var]
                //       = -0.5 * [(x_j^2 - 2*x_j*mean) / var]
                let mut score = const_term[c];
                for (&col, &val) in nnz_indices.iter().zip(nnz_data.iter()) {
                    let m = theta[[c, col]];
                    let v = var_smoothed[[c, col]];
                    // Correction: replace (0 - mean)^2/var with (val - mean)^2/var
                    // delta = -0.5 * ((val-m)^2/v - m^2/v) = -0.5 * (val^2 - 2*val*m) / v
                    score -= 0.5 * (val * val - 2.0 * val * m) / v;
                }

                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }

            predictions[i] = classes[best_class];
        }

        Ok(predictions)
    }
}

#[cfg(feature = "sparse")]
impl crate::models::traits::SparseModel for MultinomialNB {
    fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("{} samples", n_samples),
                format!("{} targets", y.len()),
            ));
        }
        if n_samples == 0 {
            return Err(FerroError::invalid_input("Empty input"));
        }

        // Validate non-negative values
        if x.data().iter().any(|&v| v < 0.0) {
            return Err(FerroError::invalid_input(
                "MultinomialNB requires non-negative feature values",
            ));
        }

        let classes = super::get_unique_classes(y);
        if classes.len() < 2 {
            return Err(FerroError::invalid_input(
                "MultinomialNB requires at least 2 classes",
            ));
        }

        let n_classes = classes.len();
        let mut feature_count = Array2::zeros((n_classes, n_features));
        let mut class_count = Array1::zeros(n_classes);

        // Accumulate feature counts per class using only nnz entries
        for i in 0..n_samples {
            let yi = y[i];
            let class_idx = classes
                .iter()
                .position(|&c| (c - yi).abs() < 1e-10)
                .ok_or_else(|| FerroError::invalid_input("Unknown class label"))?;

            let row = x.row(i);
            for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                feature_count[[class_idx, col]] += val;
            }
            class_count[class_idx] += 1.0;
        }

        // Store fitted state
        self.classes = Some(Array1::from_vec(classes.to_vec()));
        self.n_features = Some(n_features);
        self.feature_count = Some(feature_count);
        self.class_count = Some(class_count);

        // Compute log probabilities (reuses existing method)
        self.update_log_probabilities();

        Ok(())
    }

    fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array1<f64>> {
        let feature_log_prob = self
            .feature_log_prob
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let class_log_prior = self
            .class_log_prior
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;

        if x.ncols() != n_features {
            return Err(FerroError::shape_mismatch(
                format!("{} features", n_features),
                format!("{} features", x.ncols()),
            ));
        }

        let n_samples = x.nrows();
        let n_classes = feature_log_prob.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);
            let nnz_indices = row.indices();
            let nnz_data = row.data();

            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;

            for c in 0..n_classes {
                // log P(y) + sum_j x_j * log P(x_j | y)
                // Only nnz features contribute (x_j=0 contributes 0)
                let mut score = class_log_prior[c];
                for (&col, &val) in nnz_indices.iter().zip(nnz_data.iter()) {
                    score += val * feature_log_prob[[c, col]];
                }

                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }

            predictions[i] = classes[best_class];
        }

        Ok(predictions)
    }
}

#[cfg(feature = "sparse")]
impl crate::models::traits::SparseModel for BernoulliNB {
    fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("{} samples", n_samples),
                format!("{} targets", y.len()),
            ));
        }
        if n_samples == 0 {
            return Err(FerroError::invalid_input("Empty input"));
        }

        let classes = super::get_unique_classes(y);
        if classes.len() < 2 {
            return Err(FerroError::invalid_input(
                "BernoulliNB requires at least 2 classes",
            ));
        }

        let n_classes = classes.len();
        let mut feature_count = Array2::zeros((n_classes, n_features));
        let mut class_count = Array1::zeros(n_classes);

        // For BernoulliNB, we count binary presence per feature per class.
        // With binarization: any stored value > threshold counts as present.
        // Without binarization: stored values are treated as-is (should be 0/1).
        let threshold = self.binarize;

        for i in 0..n_samples {
            let yi = y[i];
            let class_idx = classes
                .iter()
                .position(|&c| (c - yi).abs() < 1e-10)
                .ok_or_else(|| FerroError::invalid_input("Unknown class label"))?;

            let row = x.row(i);
            if let Some(thresh) = threshold {
                // Binarize: nnz values > threshold count as 1
                for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                    if val > thresh {
                        feature_count[[class_idx, col]] += 1.0;
                    }
                }
            } else {
                // No binarization: accumulate raw values (assumed binary)
                for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                    feature_count[[class_idx, col]] += val;
                }
            }
            class_count[class_idx] += 1.0;
        }

        // Store fitted state
        self.classes = Some(Array1::from_vec(classes.to_vec()));
        self.n_features = Some(n_features);
        self.feature_count = Some(feature_count);
        self.class_count = Some(class_count);

        // Compute log probabilities (reuses existing method)
        self.update_log_probabilities();

        Ok(())
    }

    fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array1<f64>> {
        let feature_log_prob = self
            .feature_log_prob
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let feature_log_prob_neg = self
            .feature_log_prob_neg
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let class_log_prior = self
            .class_log_prior
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;

        if x.ncols() != n_features {
            return Err(FerroError::shape_mismatch(
                format!("{} features", n_features),
                format!("{} features", x.ncols()),
            ));
        }

        let n_samples = x.nrows();
        let n_classes = feature_log_prob.nrows();

        // Pre-compute per-class baseline: sum of log(1-p) over ALL features
        // (the contribution when all features are 0)
        let mut base_neg_sum = Array1::zeros(n_classes);
        for c in 0..n_classes {
            let mut s = 0.0;
            for j in 0..n_features {
                s += feature_log_prob_neg[[c, j]];
            }
            base_neg_sum[c] = s;
        }

        let threshold = self.binarize;
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);
            let nnz_indices = row.indices();
            let nnz_data = row.data();

            // Determine which nnz features are "present" (binarized to 1)
            let present: Vec<usize> = if let Some(thresh) = threshold {
                nnz_indices
                    .iter()
                    .zip(nnz_data.iter())
                    .filter(|(_, &val)| val > thresh)
                    .map(|(&col, _)| col)
                    .collect()
            } else {
                // Without binarization, nnz entries with value > 0 are present
                nnz_indices
                    .iter()
                    .zip(nnz_data.iter())
                    .filter(|(_, &val)| val > 0.0)
                    .map(|(&col, _)| col)
                    .collect()
            };

            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;

            for c in 0..n_classes {
                // Start with log_prior + sum_j log(1-p_j) (all absent baseline)
                // For present features, replace log(1-p) with log(p):
                // delta = log(p) - log(1-p)
                let mut score = class_log_prior[c] + base_neg_sum[c];
                for &col in &present {
                    score += feature_log_prob[[c, col]] - feature_log_prob_neg[[c, col]];
                }

                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }

            predictions[i] = classes[best_class];
        }

        Ok(predictions)
    }
}

#[cfg(feature = "sparse")]
impl crate::models::traits::SparseModel for CategoricalNB {
    fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("{} samples", n_samples),
                format!("{} targets", y.len()),
            ));
        }
        if n_samples == 0 {
            return Err(FerroError::invalid_input("Empty input"));
        }

        let classes = super::get_unique_classes(y);
        if classes.len() < 2 {
            return Err(FerroError::invalid_input(
                "CategoricalNB requires at least 2 classes",
            ));
        }

        let n_classes = classes.len();

        // First pass: discover unique categories per feature from sparse data.
        // For sparse matrices, 0.0 is an implicit value for all non-stored entries,
        // so we always include 0.0 as a possible category.
        let mut feature_cats_set: Vec<std::collections::BTreeSet<i64>> =
            vec![std::collections::BTreeSet::new(); n_features];

        // Include 0.0 for every feature (implicit zeros in sparse)
        for cat_set in &mut feature_cats_set {
            cat_set.insert(0); // 0.0 encoded as i64 via to_bits-like scheme
        }

        // Collect all nnz values
        for i in 0..n_samples {
            let row = x.row(i);
            for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                // Use a quantized integer key for the float category
                let key = (val * 1e10).round() as i64;
                feature_cats_set[col].insert(key);
            }
        }

        // Build sorted category lists per feature
        let mut feature_categories: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        for cat_set in &feature_cats_set {
            let cats: Vec<f64> = cat_set.iter().map(|&k| k as f64 / 1e10).collect();
            feature_categories.push(cats);
        }

        // Initialize count arrays
        let mut category_count: Vec<Array2<f64>> = feature_categories
            .iter()
            .map(|cats| Array2::zeros((n_classes, cats.len())))
            .collect();
        let mut class_count = Array1::zeros(n_classes);

        // Second pass: accumulate counts
        for i in 0..n_samples {
            let yi = y[i];
            let class_idx = classes
                .iter()
                .position(|&c| (c - yi).abs() < 1e-10)
                .ok_or_else(|| FerroError::invalid_input("Unknown class label"))?;

            let row = x.row(i);

            // Track which features have nnz entries for this row
            let mut nnz_cols = std::collections::HashSet::new();

            for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                nnz_cols.insert(col);
                if let Some(cat_idx) = Self::category_index(&feature_categories[col], val) {
                    category_count[col][[class_idx, cat_idx]] += 1.0;
                }
            }

            // For features NOT in nnz, the value is 0.0
            for j in 0..n_features {
                if !nnz_cols.contains(&j) {
                    if let Some(cat_idx) = Self::category_index(&feature_categories[j], 0.0) {
                        category_count[j][[class_idx, cat_idx]] += 1.0;
                    }
                }
            }

            class_count[class_idx] += 1.0;
        }

        // Store fitted state
        self.classes = Some(Array1::from_vec(classes.to_vec()));
        self.n_features = Some(n_features);
        self.class_count = Some(class_count);
        self.feature_categories = Some(feature_categories);
        self.category_count = Some(category_count);

        // Compute log probabilities (reuses existing method)
        self.update_log_probabilities();

        Ok(())
    }

    fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array1<f64>> {
        let feature_log_prob = self
            .feature_log_prob
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let feature_categories = self
            .feature_categories
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let class_log_prior = self
            .class_log_prior
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;
        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("predict_sparse"))?;

        if x.ncols() != n_features {
            return Err(FerroError::shape_mismatch(
                format!("{} features", n_features),
                format!("{} features", x.ncols()),
            ));
        }

        let n_samples = x.nrows();
        let n_classes = class_log_prior.len();

        // Pre-compute per-class baseline: sum of log P(x_j=0|c) for all features
        // (the contribution when all features are 0)
        let mut base_zero_contrib = Array1::zeros(n_classes);
        for c in 0..n_classes {
            let mut s = 0.0;
            for j in 0..n_features {
                if let Some(cat_idx) = Self::category_index(&feature_categories[j], 0.0) {
                    s += feature_log_prob[j][[c, cat_idx]];
                } else {
                    // 0.0 not a known category: use unseen category probability
                    let n_cats = feature_categories[j].len();
                    let n_k = self.class_count.as_ref().map(|cc| cc[c]).unwrap_or(0.0);
                    let denominator = self.alpha.mul_add((n_cats + 1) as f64, n_k);
                    s += (self.alpha / denominator).ln();
                }
            }
            base_zero_contrib[c] = s;
        }

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);
            let nnz_indices = row.indices();
            let nnz_data = row.data();

            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;

            for c in 0..n_classes {
                // Start with log_prior + baseline (all features = 0)
                let mut score = class_log_prior[c] + base_zero_contrib[c];

                // For nnz features, replace the x_j=0 contribution with actual value
                for (&col, &val) in nnz_indices.iter().zip(nnz_data.iter()) {
                    // Subtract the 0-category contribution
                    if let Some(zero_cat_idx) = Self::category_index(&feature_categories[col], 0.0)
                    {
                        score -= feature_log_prob[col][[c, zero_cat_idx]];
                    } else {
                        let n_cats = feature_categories[col].len();
                        let n_k = self.class_count.as_ref().map(|cc| cc[c]).unwrap_or(0.0);
                        let denominator = self.alpha.mul_add((n_cats + 1) as f64, n_k);
                        score -= (self.alpha / denominator).ln();
                    }

                    // Add the actual value's contribution
                    if let Some(cat_idx) = Self::category_index(&feature_categories[col], val) {
                        score += feature_log_prob[col][[c, cat_idx]];
                    } else {
                        // Unseen category
                        let n_cats = feature_categories[col].len();
                        let n_k = self.class_count.as_ref().map(|cc| cc[c]).unwrap_or(0.0);
                        let denominator = self.alpha.mul_add((n_cats + 1) as f64, n_k);
                        score += (self.alpha / denominator).ln();
                    }
                }

                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }

            predictions[i] = classes[best_class];
        }

        Ok(predictions)
    }
}

// =============================================================================
// PipelineSparseModel Implementations
// =============================================================================

#[cfg(feature = "sparse")]
impl crate::pipeline::PipelineSparseModel for GaussianNB {
    fn fit_sparse(
        &mut self,
        x: &crate::sparse::CsrMatrix,
        y: &ndarray::Array1<f64>,
    ) -> crate::Result<()> {
        crate::models::traits::SparseModel::fit_sparse(self, x, y)
    }

    fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> crate::Result<ndarray::Array1<f64>> {
        crate::models::traits::SparseModel::predict_sparse(self, x)
    }

    fn search_space(&self) -> crate::hpo::SearchSpace {
        crate::models::Model::search_space(self)
    }

    fn clone_boxed(&self) -> Box<dyn crate::pipeline::PipelineSparseModel> {
        Box::new(self.clone())
    }

    fn set_param(&mut self, name: &str, _value: &crate::hpo::ParameterValue) -> crate::Result<()> {
        Err(crate::FerroError::invalid_input(format!(
            "Unknown parameter '{}'",
            name
        )))
    }

    fn name(&self) -> &str {
        "GaussianNB"
    }

    fn is_fitted(&self) -> bool {
        crate::models::Model::is_fitted(self)
    }
}

#[cfg(feature = "sparse")]
impl crate::pipeline::PipelineSparseModel for MultinomialNB {
    fn fit_sparse(
        &mut self,
        x: &crate::sparse::CsrMatrix,
        y: &ndarray::Array1<f64>,
    ) -> crate::Result<()> {
        crate::models::traits::SparseModel::fit_sparse(self, x, y)
    }

    fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> crate::Result<ndarray::Array1<f64>> {
        crate::models::traits::SparseModel::predict_sparse(self, x)
    }

    fn search_space(&self) -> crate::hpo::SearchSpace {
        crate::models::Model::search_space(self)
    }

    fn clone_boxed(&self) -> Box<dyn crate::pipeline::PipelineSparseModel> {
        Box::new(self.clone())
    }

    fn set_param(&mut self, name: &str, value: &crate::hpo::ParameterValue) -> crate::Result<()> {
        match name {
            "alpha" => {
                if let Some(v) = value.as_f64() {
                    self.alpha = v;
                    Ok(())
                } else {
                    Err(crate::FerroError::invalid_input("alpha must be a number"))
                }
            }
            _ => Err(crate::FerroError::invalid_input(format!(
                "Unknown parameter '{}'",
                name
            ))),
        }
    }

    fn name(&self) -> &str {
        "MultinomialNB"
    }

    fn is_fitted(&self) -> bool {
        crate::models::Model::is_fitted(self)
    }
}

#[cfg(feature = "sparse")]
impl crate::pipeline::PipelineSparseModel for BernoulliNB {
    fn fit_sparse(
        &mut self,
        x: &crate::sparse::CsrMatrix,
        y: &ndarray::Array1<f64>,
    ) -> crate::Result<()> {
        crate::models::traits::SparseModel::fit_sparse(self, x, y)
    }

    fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> crate::Result<ndarray::Array1<f64>> {
        crate::models::traits::SparseModel::predict_sparse(self, x)
    }

    fn search_space(&self) -> crate::hpo::SearchSpace {
        crate::models::Model::search_space(self)
    }

    fn clone_boxed(&self) -> Box<dyn crate::pipeline::PipelineSparseModel> {
        Box::new(self.clone())
    }

    fn set_param(&mut self, name: &str, value: &crate::hpo::ParameterValue) -> crate::Result<()> {
        match name {
            "alpha" => {
                if let Some(v) = value.as_f64() {
                    self.alpha = v;
                    Ok(())
                } else {
                    Err(crate::FerroError::invalid_input("alpha must be a number"))
                }
            }
            "binarize" => {
                if let Some(v) = value.as_f64() {
                    if v < 0.0 {
                        self.binarize = None;
                    } else {
                        self.binarize = Some(v);
                    }
                    Ok(())
                } else {
                    Err(crate::FerroError::invalid_input(
                        "binarize must be a number",
                    ))
                }
            }
            _ => Err(crate::FerroError::invalid_input(format!(
                "Unknown parameter '{}'",
                name
            ))),
        }
    }

    fn name(&self) -> &str {
        "BernoulliNB"
    }

    fn is_fitted(&self) -> bool {
        crate::models::Model::is_fitted(self)
    }
}

#[cfg(feature = "sparse")]
impl crate::pipeline::PipelineSparseModel for CategoricalNB {
    fn fit_sparse(
        &mut self,
        x: &crate::sparse::CsrMatrix,
        y: &ndarray::Array1<f64>,
    ) -> crate::Result<()> {
        crate::models::traits::SparseModel::fit_sparse(self, x, y)
    }

    fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> crate::Result<ndarray::Array1<f64>> {
        crate::models::traits::SparseModel::predict_sparse(self, x)
    }

    fn search_space(&self) -> crate::hpo::SearchSpace {
        crate::models::Model::search_space(self)
    }

    fn clone_boxed(&self) -> Box<dyn crate::pipeline::PipelineSparseModel> {
        Box::new(self.clone())
    }

    fn set_param(&mut self, name: &str, value: &crate::hpo::ParameterValue) -> crate::Result<()> {
        match name {
            "alpha" => {
                if let Some(v) = value.as_f64() {
                    self.alpha = v;
                    Ok(())
                } else {
                    Err(crate::FerroError::invalid_input("alpha must be a number"))
                }
            }
            _ => Err(crate::FerroError::invalid_input(format!(
                "Unknown parameter '{}'",
                name
            ))),
        }
    }

    fn name(&self) -> &str {
        "CategoricalNB"
    }

    fn is_fitted(&self) -> bool {
        crate::models::Model::is_fitted(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_simple_dataset() -> (Array2<f64>, Array1<f64>) {
        // Simple 2D dataset with two classes
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.0, 3.0, // Class 0
                7.0, 7.0, 7.5, 7.5, 8.0, 8.0, 8.5, 8.5, 9.0, 9.0, // Class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        (x, y)
    }

    #[test]
    fn test_gaussian_nb_fit_predict() {
        let (x, y) = create_simple_dataset();

        let mut model = GaussianNB::new();
        model.fit(&x, &y).unwrap();

        assert!(model.is_fitted());

        // Predict on training data
        let pred = model.predict(&x).unwrap();

        // Should classify correctly (classes are well-separated)
        for i in 0..5 {
            assert_eq!(pred[i], 0.0, "Sample {} should be class 0", i);
        }
        for i in 5..10 {
            assert_eq!(pred[i], 1.0, "Sample {} should be class 1", i);
        }
    }

    #[test]
    fn test_predict_proba() {
        let (x, y) = create_simple_dataset();

        let mut model = GaussianNB::new();
        model.fit(&x, &y).unwrap();

        let probas = model.predict_proba(&x).unwrap();

        // Check shape
        assert_eq!(probas.nrows(), 10);
        assert_eq!(probas.ncols(), 2);

        // Probabilities should sum to 1
        for i in 0..10 {
            let sum: f64 = probas.row(i).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }

        // Class 0 samples should have high P(y=0)
        for i in 0..5 {
            assert!(
                probas[[i, 0]] > 0.9,
                "P(y=0) should be high for sample {}",
                i
            );
        }

        // Class 1 samples should have high P(y=1)
        for i in 5..10 {
            assert!(
                probas[[i, 1]] > 0.9,
                "P(y=1) should be high for sample {}",
                i
            );
        }
    }

    #[test]
    fn test_theta_and_var() {
        let (x, y) = create_simple_dataset();

        let mut model = GaussianNB::new();
        model.fit(&x, &y).unwrap();

        let theta = model.theta().unwrap();
        let var = model.var().unwrap();

        // Check shape
        assert_eq!(theta.shape(), &[2, 2]); // 2 classes, 2 features
        assert_eq!(var.shape(), &[2, 2]);

        // Class 0 mean should be around 2.0 for both features
        assert_relative_eq!(theta[[0, 0]], 2.0, epsilon = 0.1);
        assert_relative_eq!(theta[[0, 1]], 2.0, epsilon = 0.1);

        // Class 1 mean should be around 8.0 for both features
        assert_relative_eq!(theta[[1, 0]], 8.0, epsilon = 0.1);
        assert_relative_eq!(theta[[1, 1]], 8.0, epsilon = 0.1);
    }

    #[test]
    fn test_class_priors() {
        let (x, y) = create_simple_dataset();

        let mut model = GaussianNB::new();
        model.fit(&x, &y).unwrap();

        let priors = model.class_prior().unwrap();

        // Equal class sizes -> equal priors
        assert_relative_eq!(priors[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(priors[1], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_custom_priors() {
        let (x, y) = create_simple_dataset();

        let custom_priors = Array1::from_vec(vec![0.3, 0.7]);
        let mut model = GaussianNB::new().with_priors(custom_priors.clone());
        model.fit(&x, &y).unwrap();

        let priors = model.class_prior().unwrap();
        assert_relative_eq!(priors[0], 0.3, epsilon = 1e-10);
        assert_relative_eq!(priors[1], 0.7, epsilon = 1e-10);
    }

    #[test]
    fn test_partial_fit() {
        // First batch
        let x1 =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0]).unwrap();
        let y1 = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = GaussianNB::new();
        model.partial_fit(&x1, &y1, Some(vec![0.0, 1.0])).unwrap();

        assert!(model.is_fitted());
        assert_eq!(model.class_count().unwrap()[0], 2.0);
        assert_eq!(model.class_count().unwrap()[1], 2.0);

        // Second batch
        let x2 =
            Array2::from_shape_vec((4, 2), vec![1.5, 1.5, 2.5, 2.5, 7.5, 7.5, 8.5, 8.5]).unwrap();
        let y2 = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        model.partial_fit(&x2, &y2, None).unwrap();

        assert_eq!(model.class_count().unwrap()[0], 4.0);
        assert_eq!(model.class_count().unwrap()[1], 4.0);

        // Should still predict correctly
        let pred = model.predict(&x1).unwrap();
        assert_eq!(pred[0], 0.0);
        assert_eq!(pred[1], 0.0);
        assert_eq!(pred[2], 1.0);
        assert_eq!(pred[3], 1.0);
    }

    #[test]
    fn test_partial_fit_unbalanced() {
        // Test incremental learning with unbalanced updates
        let x1 = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0],
        )
        .unwrap();
        let y1 = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = GaussianNB::new();
        model.partial_fit(&x1, &y1, Some(vec![0.0, 1.0])).unwrap();

        // Add only class 0 samples
        let x2 = Array2::from_shape_vec((2, 2), vec![1.5, 1.5, 2.5, 2.5]).unwrap();
        let y2 = Array1::from_vec(vec![0.0, 0.0]);

        model.partial_fit(&x2, &y2, None).unwrap();

        // Class 0 should have 5 samples, class 1 should have 3
        assert_eq!(model.class_count().unwrap()[0], 5.0);
        assert_eq!(model.class_count().unwrap()[1], 3.0);
    }

    #[test]
    fn test_var_smoothing() {
        let (x, y) = create_simple_dataset();

        let mut model_default = GaussianNB::new();
        model_default.fit(&x, &y).unwrap();

        let mut model_high_smooth = GaussianNB::new().with_var_smoothing(1.0);
        model_high_smooth.fit(&x, &y).unwrap();

        // Higher smoothing should result in larger epsilon
        assert!(model_high_smooth.epsilon().unwrap() > model_default.epsilon().unwrap());
    }

    #[test]
    fn test_multiclass() {
        // 3-class dataset
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, // Class 0
                5.0, 5.0, 5.5, 5.5, 6.0, 6.0, 6.5, 6.5, // Class 1
                9.0, 9.0, 9.5, 9.5, 10.0, 10.0, 10.5, 10.5, // Class 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
        ]);

        let mut model = GaussianNB::new();
        model.fit(&x, &y).unwrap();

        let classes = model.classes().unwrap();
        assert_eq!(classes.len(), 3);

        let probas = model.predict_proba(&x).unwrap();
        assert_eq!(probas.ncols(), 3);

        // Each row should sum to 1
        for i in 0..12 {
            let sum: f64 = probas.row(i).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_feature_importance() {
        let (x, y) = create_simple_dataset();

        let mut model = GaussianNB::new();
        model.fit(&x, &y).unwrap();

        let importance = model.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // Both features are equally discriminative in this dataset
        // so importance should be similar
        assert!(importance[0] > 0.0);
        assert!(importance[1] > 0.0);
    }

    #[test]
    fn test_prediction_interval() {
        let (x, y) = create_simple_dataset();

        let mut model = GaussianNB::new();
        model.fit(&x, &y).unwrap();

        let interval = model.predict_interval(&x, 0.95).unwrap();

        assert_eq!(interval.predictions.len(), 10);
        assert_eq!(interval.lower.len(), 10);
        assert_eq!(interval.upper.len(), 10);

        // Bounds should be valid
        for i in 0..10 {
            assert!(interval.lower[i] >= 0.0);
            assert!(interval.upper[i] <= 1.0);
            assert!(interval.lower[i] <= interval.predictions[i]);
            assert!(interval.predictions[i] <= interval.upper[i]);
        }
    }

    #[test]
    fn test_error_not_fitted() {
        let model = GaussianNB::new();
        let x = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();

        assert!(model.predict(&x).is_err());
        assert!(model.predict_proba(&x).is_err());
    }

    #[test]
    fn test_error_single_class() {
        let x = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
        let y = Array1::from_vec(vec![0.0; 5]);

        let mut model = GaussianNB::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_error_wrong_features() {
        let x_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 7.0, 7.0, 8.0, 8.0]).unwrap();
        let y_train = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = GaussianNB::new();
        model.fit(&x_train, &y_train).unwrap();

        // Try to predict with wrong number of features
        let x_test = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        assert!(model.predict(&x_test).is_err());
    }

    #[test]
    fn test_partial_fit_wrong_features() {
        let x1 =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 7.0, 7.0, 8.0, 8.0]).unwrap();
        let y1 = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = GaussianNB::new();
        model.partial_fit(&x1, &y1, Some(vec![0.0, 1.0])).unwrap();

        // Try to partial_fit with wrong number of features
        let x2 = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        let y2 = Array1::from_vec(vec![0.0, 1.0]);
        assert!(model.partial_fit(&x2, &y2, None).is_err());
    }

    #[test]
    fn test_partial_fit_missing_classes() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 7.0, 7.0, 8.0, 8.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = GaussianNB::new();

        // First call without classes should fail
        assert!(model.partial_fit(&x, &y, None).is_err());
    }

    #[test]
    fn test_log_probability_numerical_stability() {
        // Test with very separated classes to check for numerical stability
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.2, // Class 0
                100.0, 100.0, 100.1, 100.1, 100.2, 100.2, // Class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = GaussianNB::new();
        model.fit(&x, &y).unwrap();

        let probas = model.predict_proba(&x).unwrap();

        // Should not have NaN or Inf
        for i in 0..6 {
            for j in 0..2 {
                assert!(probas[[i, j]].is_finite(), "Probability should be finite");
            }
            let sum: f64 = probas.row(i).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_classes_sorted() {
        // Provide labels in non-sorted order
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);

        let mut model = GaussianNB::new();
        model.fit(&x, &y).unwrap();

        let classes = model.classes().unwrap();
        // Classes should be sorted
        assert_eq!(classes[0], 0.0);
        assert_eq!(classes[1], 1.0);
    }

    // ==========================================================================
    // MultinomialNB Tests
    // ==========================================================================

    fn create_multinomial_dataset() -> (Array2<f64>, Array1<f64>) {
        // Simulated word count data (bag-of-words style)
        // Class 0: high counts in features 0,1
        // Class 1: high counts in features 2,3
        let x = Array2::from_shape_vec(
            (8, 4),
            vec![
                5.0, 4.0, 0.0, 1.0, // Class 0
                6.0, 3.0, 1.0, 0.0, // Class 0
                4.0, 5.0, 0.0, 0.0, // Class 0
                5.0, 5.0, 1.0, 1.0, // Class 0
                0.0, 1.0, 5.0, 4.0, // Class 1
                1.0, 0.0, 6.0, 3.0, // Class 1
                0.0, 0.0, 4.0, 5.0, // Class 1
                1.0, 1.0, 5.0, 5.0, // Class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        (x, y)
    }

    #[test]
    fn test_multinomial_nb_fit_predict() {
        let (x, y) = create_multinomial_dataset();

        let mut model = MultinomialNB::new();
        model.fit(&x, &y).unwrap();

        assert!(model.is_fitted());

        let pred = model.predict(&x).unwrap();

        // Should classify correctly
        for i in 0..4 {
            assert_eq!(pred[i], 0.0, "Sample {} should be class 0", i);
        }
        for i in 4..8 {
            assert_eq!(pred[i], 1.0, "Sample {} should be class 1", i);
        }
    }

    #[test]
    fn test_multinomial_nb_predict_proba() {
        let (x, y) = create_multinomial_dataset();

        let mut model = MultinomialNB::new();
        model.fit(&x, &y).unwrap();

        let probas = model.predict_proba(&x).unwrap();

        // Check shape
        assert_eq!(probas.nrows(), 8);
        assert_eq!(probas.ncols(), 2);

        // Probabilities should sum to 1
        for i in 0..8 {
            let sum: f64 = probas.row(i).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multinomial_nb_partial_fit() {
        // First batch
        let x1 = Array2::from_shape_vec(
            (4, 3),
            vec![
                5.0, 1.0, 0.0, 4.0, 2.0, 0.0, // Class 0
                0.0, 1.0, 5.0, 0.0, 0.0, 6.0, // Class 1
            ],
        )
        .unwrap();
        let y1 = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = MultinomialNB::new();
        model.partial_fit(&x1, &y1, Some(vec![0.0, 1.0])).unwrap();

        assert!(model.is_fitted());
        assert_eq!(model.class_count().unwrap()[0], 2.0);
        assert_eq!(model.class_count().unwrap()[1], 2.0);

        // Second batch
        let x2 = Array2::from_shape_vec(
            (2, 3),
            vec![
                6.0, 2.0, 0.0, // Class 0
                0.0, 1.0, 7.0, // Class 1
            ],
        )
        .unwrap();
        let y2 = Array1::from_vec(vec![0.0, 1.0]);

        model.partial_fit(&x2, &y2, None).unwrap();

        assert_eq!(model.class_count().unwrap()[0], 3.0);
        assert_eq!(model.class_count().unwrap()[1], 3.0);
    }

    #[test]
    fn test_multinomial_nb_alpha_smoothing() {
        let (x, y) = create_multinomial_dataset();

        // With different alpha values
        let mut model_default = MultinomialNB::new();
        model_default.fit(&x, &y).unwrap();

        let mut model_low_alpha = MultinomialNB::new().with_alpha(0.1);
        model_low_alpha.fit(&x, &y).unwrap();

        // Both should predict correctly
        let pred_default = model_default.predict(&x).unwrap();
        let pred_low = model_low_alpha.predict(&x).unwrap();

        for i in 0..4 {
            assert_eq!(pred_default[i], 0.0);
            assert_eq!(pred_low[i], 0.0);
        }
    }

    #[test]
    fn test_multinomial_nb_negative_values() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, -1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = MultinomialNB::new();
        // Should fail because of negative values
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_multinomial_nb_uniform_priors() {
        let (x, y) = create_multinomial_dataset();

        let mut model = MultinomialNB::new().with_fit_prior(false);
        model.fit(&x, &y).unwrap();

        let log_priors = model.class_log_prior().unwrap();
        // Uniform priors -> log(0.5) for each class
        let expected_log_prior = 0.5_f64.ln();
        assert_relative_eq!(log_priors[0], expected_log_prior, epsilon = 1e-10);
        assert_relative_eq!(log_priors[1], expected_log_prior, epsilon = 1e-10);
    }

    #[test]
    fn test_multinomial_nb_feature_counts() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                3.0, 1.0, // Class 0
                2.0, 2.0, // Class 0
                1.0, 3.0, // Class 1
                0.0, 4.0, // Class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = MultinomialNB::new();
        model.fit(&x, &y).unwrap();

        let feature_count = model.feature_count().unwrap();
        // Class 0: feature 0 = 3+2 = 5, feature 1 = 1+2 = 3
        assert_relative_eq!(feature_count[[0, 0]], 5.0, epsilon = 1e-10);
        assert_relative_eq!(feature_count[[0, 1]], 3.0, epsilon = 1e-10);
        // Class 1: feature 0 = 1+0 = 1, feature 1 = 3+4 = 7
        assert_relative_eq!(feature_count[[1, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(feature_count[[1, 1]], 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multinomial_nb_feature_importance() {
        let (x, y) = create_multinomial_dataset();

        let mut model = MultinomialNB::new();
        model.fit(&x, &y).unwrap();

        let importance = model.feature_importance().unwrap();
        assert_eq!(importance.len(), 4);

        // All features should have positive importance
        for i in 0..4 {
            assert!(importance[i] >= 0.0);
        }
    }

    // ==========================================================================
    // BernoulliNB Tests
    // ==========================================================================

    fn create_bernoulli_dataset() -> (Array2<f64>, Array1<f64>) {
        // Binary presence/absence features
        // Class 0: features 0,1 tend to be present
        // Class 1: features 2,3 tend to be present
        let x = Array2::from_shape_vec(
            (8, 4),
            vec![
                1.0, 1.0, 0.0, 0.0, // Class 0
                1.0, 1.0, 0.0, 1.0, // Class 0
                1.0, 0.0, 0.0, 0.0, // Class 0
                1.0, 1.0, 1.0, 0.0, // Class 0
                0.0, 0.0, 1.0, 1.0, // Class 1
                0.0, 1.0, 1.0, 1.0, // Class 1
                0.0, 0.0, 1.0, 1.0, // Class 1
                1.0, 0.0, 1.0, 1.0, // Class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        (x, y)
    }

    #[test]
    fn test_bernoulli_nb_fit_predict() {
        let (x, y) = create_bernoulli_dataset();

        let mut model = BernoulliNB::new().with_binarize(None); // Already binary
        model.fit(&x, &y).unwrap();

        assert!(model.is_fitted());

        let pred = model.predict(&x).unwrap();

        // Should classify correctly
        for i in 0..4 {
            assert_eq!(pred[i], 0.0, "Sample {} should be class 0", i);
        }
        for i in 4..8 {
            assert_eq!(pred[i], 1.0, "Sample {} should be class 1", i);
        }
    }

    #[test]
    fn test_bernoulli_nb_predict_proba() {
        let (x, y) = create_bernoulli_dataset();

        let mut model = BernoulliNB::new().with_binarize(None);
        model.fit(&x, &y).unwrap();

        let probas = model.predict_proba(&x).unwrap();

        // Check shape
        assert_eq!(probas.nrows(), 8);
        assert_eq!(probas.ncols(), 2);

        // Probabilities should sum to 1
        for i in 0..8 {
            let sum: f64 = probas.row(i).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bernoulli_nb_binarization() {
        // Non-binary data that should be binarized
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                5.0, 3.0, 0.0, // -> 1, 1, 0 (Class 0)
                4.0, 2.0, 0.0, // -> 1, 1, 0 (Class 0)
                6.0, 4.0, 0.0, // -> 1, 1, 0 (Class 0)
                0.0, 0.0, 5.0, // -> 0, 0, 1 (Class 1)
                0.0, 0.0, 4.0, // -> 0, 0, 1 (Class 1)
                0.0, 0.0, 6.0, // -> 0, 0, 1 (Class 1)
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        // Default binarize threshold is 0.0
        let mut model = BernoulliNB::new();
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x).unwrap();
        for i in 0..3 {
            assert_eq!(pred[i], 0.0, "Sample {} should be class 0", i);
        }
        for i in 3..6 {
            assert_eq!(pred[i], 1.0, "Sample {} should be class 1", i);
        }
    }

    #[test]
    fn test_bernoulli_nb_custom_binarize_threshold() {
        // Data with values around threshold
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                3.0, 1.0, // threshold=2: -> 1, 0
                4.0, 0.5, // -> 1, 0
                0.5, 3.0, // -> 0, 1
                1.0, 4.0, // -> 0, 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = BernoulliNB::new().with_binarize(Some(2.0));
        model.fit(&x, &y).unwrap();

        let feature_count = model.feature_count().unwrap();
        // Class 0: feature 0 count = 2 (both > 2), feature 1 count = 0 (both <= 2)
        assert_relative_eq!(feature_count[[0, 0]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(feature_count[[0, 1]], 0.0, epsilon = 1e-10);
        // Class 1: feature 0 count = 0 (both <= 2), feature 1 count = 2 (both > 2)
        assert_relative_eq!(feature_count[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(feature_count[[1, 1]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bernoulli_nb_partial_fit() {
        let x1 = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 0.0, 1.0, 0.0, // Class 0
                0.0, 1.0, 0.0, 1.0, // Class 1
            ],
        )
        .unwrap();
        let y1 = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = BernoulliNB::new().with_binarize(None);
        model.partial_fit(&x1, &y1, Some(vec![0.0, 1.0])).unwrap();

        assert!(model.is_fitted());
        assert_eq!(model.class_count().unwrap()[0], 2.0);
        assert_eq!(model.class_count().unwrap()[1], 2.0);

        // Second batch
        let x2 = Array2::from_shape_vec(
            (2, 2),
            vec![
                1.0, 1.0, // Class 0
                0.0, 1.0, // Class 1
            ],
        )
        .unwrap();
        let y2 = Array1::from_vec(vec![0.0, 1.0]);

        model.partial_fit(&x2, &y2, None).unwrap();

        assert_eq!(model.class_count().unwrap()[0], 3.0);
        assert_eq!(model.class_count().unwrap()[1], 3.0);
    }

    #[test]
    fn test_bernoulli_nb_alpha_smoothing() {
        let (x, y) = create_bernoulli_dataset();

        let mut model_high_alpha = BernoulliNB::new().with_alpha(2.0).with_binarize(None);
        model_high_alpha.fit(&x, &y).unwrap();

        let mut model_low_alpha = BernoulliNB::new().with_alpha(0.01).with_binarize(None);
        model_low_alpha.fit(&x, &y).unwrap();

        // Both should still predict correctly on well-separated data
        let pred_high = model_high_alpha.predict(&x).unwrap();
        let pred_low = model_low_alpha.predict(&x).unwrap();

        for i in 0..4 {
            assert_eq!(pred_high[i], 0.0);
            assert_eq!(pred_low[i], 0.0);
        }
    }

    #[test]
    fn test_bernoulli_nb_feature_log_prob() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 0.0, 1.0, 0.0, // Class 0: feature 0 always present, feature 1 never
                0.0, 1.0, 0.0, 1.0, // Class 1: feature 0 never, feature 1 always
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = BernoulliNB::new().with_alpha(1.0).with_binarize(None);
        model.fit(&x, &y).unwrap();

        let log_prob = model.feature_log_prob().unwrap();

        // With alpha=1, n_class=2:
        // P(feature 0 = 1 | class 0) = (2 + 1) / (2 + 2) = 0.75
        // P(feature 1 = 1 | class 0) = (0 + 1) / (2 + 2) = 0.25
        assert_relative_eq!(log_prob[[0, 0]], 0.75_f64.ln(), epsilon = 1e-10);
        assert_relative_eq!(log_prob[[0, 1]], 0.25_f64.ln(), epsilon = 1e-10);

        // P(feature 0 = 1 | class 1) = (0 + 1) / (2 + 2) = 0.25
        // P(feature 1 = 1 | class 1) = (2 + 1) / (2 + 2) = 0.75
        assert_relative_eq!(log_prob[[1, 0]], 0.25_f64.ln(), epsilon = 1e-10);
        assert_relative_eq!(log_prob[[1, 1]], 0.75_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_bernoulli_nb_feature_importance() {
        let (x, y) = create_bernoulli_dataset();

        let mut model = BernoulliNB::new().with_binarize(None);
        model.fit(&x, &y).unwrap();

        let importance = model.feature_importance().unwrap();
        assert_eq!(importance.len(), 4);

        // All features should have non-negative importance
        for i in 0..4 {
            assert!(importance[i] >= 0.0);
        }
    }

    #[test]
    fn test_bernoulli_nb_prediction_interval() {
        let (x, y) = create_bernoulli_dataset();

        let mut model = BernoulliNB::new().with_binarize(None);
        model.fit(&x, &y).unwrap();

        let interval = model.predict_interval(&x, 0.95).unwrap();

        assert_eq!(interval.predictions.len(), 8);
        assert_eq!(interval.lower.len(), 8);
        assert_eq!(interval.upper.len(), 8);

        for i in 0..8 {
            assert!(interval.lower[i] >= 0.0);
            assert!(interval.upper[i] <= 1.0);
            assert!(interval.lower[i] <= interval.predictions[i]);
            assert!(interval.predictions[i] <= interval.upper[i]);
        }
    }

    #[test]
    fn test_bernoulli_nb_multiclass() {
        // 3-class dataset
        let x = Array2::from_shape_vec(
            (9, 3),
            vec![
                1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, // Class 0: feature 0
                0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, // Class 1: feature 1
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, // Class 2: feature 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let mut model = BernoulliNB::new().with_binarize(None);
        model.fit(&x, &y).unwrap();

        let classes = model.classes().unwrap();
        assert_eq!(classes.len(), 3);

        let probas = model.predict_proba(&x).unwrap();
        assert_eq!(probas.ncols(), 3);

        // Each row should sum to 1
        for i in 0..9 {
            let sum: f64 = probas.row(i).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bernoulli_nb_not_fitted_error() {
        let model = BernoulliNB::new();
        let x = Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap();

        assert!(model.predict(&x).is_err());
        assert!(model.predict_proba(&x).is_err());
    }

    #[test]
    fn test_bernoulli_nb_single_class_error() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
        let y = Array1::from_vec(vec![0.0; 4]);

        let mut model = BernoulliNB::new();
        assert!(model.fit(&x, &y).is_err());
    }

    // =========================================================================
    // CategoricalNB Tests
    // =========================================================================

    /// Create a simple categorical dataset with 2 features, 2 classes
    fn create_categorical_dataset() -> (Array2<f64>, Array1<f64>) {
        // Feature 0: categories {0, 1, 2}
        // Feature 1: categories {0, 1}
        // Class 0 tends to have feat0=0,1 and feat1=0
        // Class 1 tends to have feat0=1,2 and feat1=1
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, // class 0
                1.0, 0.0, // class 0
                0.0, 0.0, // class 0
                1.0, 0.0, // class 0
                2.0, 1.0, // class 1
                1.0, 1.0, // class 1
                2.0, 1.0, // class 1
                2.0, 1.0, // class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        (x, y)
    }

    #[test]
    fn test_categorical_nb_basic_fit_predict() {
        let (x, y) = create_categorical_dataset();

        let mut model = CategoricalNB::new();
        model.fit(&x, &y).unwrap();

        assert!(model.is_fitted());

        let pred = model.predict(&x).unwrap();
        assert_eq!(pred.len(), 8);

        // Class 0 samples should be predicted as 0
        for i in 0..4 {
            assert_eq!(pred[i], 0.0, "Sample {} should be class 0", i);
        }
        // Class 1 samples should be predicted as 1
        for i in 4..8 {
            assert_eq!(pred[i], 1.0, "Sample {} should be class 1", i);
        }
    }

    #[test]
    fn test_categorical_nb_predict_proba_sums_to_one() {
        let (x, y) = create_categorical_dataset();

        let mut model = CategoricalNB::new();
        model.fit(&x, &y).unwrap();

        let probas = model.predict_proba(&x).unwrap();
        assert_eq!(probas.nrows(), 8);
        assert_eq!(probas.ncols(), 2);

        for i in 0..8 {
            let sum: f64 = probas.row(i).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_categorical_nb_laplace_smoothing() {
        let (x, y) = create_categorical_dataset();

        let mut model = CategoricalNB::new().with_alpha(1.0);
        model.fit(&x, &y).unwrap();

        // Test with unseen category: feature 0 = 3.0 (never seen)
        let x_test = Array2::from_shape_vec((1, 2), vec![3.0, 0.0]).unwrap();
        let probas = model.predict_proba(&x_test).unwrap();

        // Should get non-zero probabilities for both classes
        assert!(
            probas[[0, 0]] > 0.0,
            "Should have non-zero prob for class 0"
        );
        assert!(
            probas[[0, 1]] > 0.0,
            "Should have non-zero prob for class 1"
        );

        let sum: f64 = probas.row(0).sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_categorical_nb_alpha_zero() {
        let (x, y) = create_categorical_dataset();

        let mut model = CategoricalNB::new().with_alpha(0.0);
        model.fit(&x, &y).unwrap();

        // With alpha=0, category counts should directly correspond to probabilities
        let category_count = model.category_count().unwrap();
        // Feature 0: class 0 has cats {0:2, 1:2, 2:0}, class 1 has {0:0, 1:1, 2:3}
        assert_eq!(category_count[0][[0, 0]], 2.0); // class 0, cat 0
        assert_eq!(category_count[0][[0, 1]], 2.0); // class 0, cat 1
        assert_eq!(category_count[0][[0, 2]], 0.0); // class 0, cat 2
        assert_eq!(category_count[0][[1, 0]], 0.0); // class 1, cat 0
        assert_eq!(category_count[0][[1, 1]], 1.0); // class 1, cat 1
        assert_eq!(category_count[0][[1, 2]], 3.0); // class 1, cat 2
    }

    #[test]
    fn test_categorical_nb_alpha_effect() {
        let (x, y) = create_categorical_dataset();

        let mut model_low = CategoricalNB::new().with_alpha(0.01);
        model_low.fit(&x, &y).unwrap();

        let mut model_high = CategoricalNB::new().with_alpha(10.0);
        model_high.fit(&x, &y).unwrap();

        let probas_low = model_low.predict_proba(&x).unwrap();
        let probas_high = model_high.predict_proba(&x).unwrap();

        // Higher alpha should make probabilities more uniform (closer to 0.5)
        // For a clearly class-0 sample:
        let diff_low = (probas_low[[0, 0]] - probas_low[[0, 1]]).abs();
        let diff_high = (probas_high[[0, 0]] - probas_high[[0, 1]]).abs();
        assert!(
            diff_low > diff_high,
            "Higher alpha should produce more uniform probabilities"
        );
    }

    #[test]
    fn test_categorical_nb_fit_prior_false() {
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);

        let mut model = CategoricalNB::new().with_fit_prior(false);
        model.fit(&x, &y).unwrap();

        let class_log_prior = model.class_log_prior().unwrap();
        let n_classes = class_log_prior.len();
        let uniform_log = (1.0 / n_classes as f64).ln();

        for i in 0..n_classes {
            assert_relative_eq!(class_log_prior[i], uniform_log, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_categorical_nb_custom_prior() {
        let (x, y) = create_categorical_dataset();

        let priors = Array1::from_vec(vec![0.3, 0.7]);
        let mut model = CategoricalNB::new().with_class_prior(priors.clone());
        model.fit(&x, &y).unwrap();

        let class_log_prior = model.class_log_prior().unwrap();
        assert_relative_eq!(class_log_prior[0], 0.3_f64.ln(), epsilon = 1e-10);
        assert_relative_eq!(class_log_prior[1], 0.7_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_categorical_nb_partial_fit() {
        let (x, y) = create_categorical_dataset();

        // Split into two batches
        let x1 = x.slice(ndarray::s![0..4, ..]).to_owned();
        let y1 = y.slice(ndarray::s![0..4]).to_owned();
        let x2 = x.slice(ndarray::s![4..8, ..]).to_owned();
        let y2 = y.slice(ndarray::s![4..8]).to_owned();

        // Incremental fit
        let mut model_inc = CategoricalNB::new();
        model_inc
            .partial_fit(&x1, &y1, Some(vec![0.0, 1.0]))
            .unwrap();
        model_inc.partial_fit(&x2, &y2, None).unwrap();

        // Batch fit
        let mut model_batch = CategoricalNB::new();
        model_batch.fit(&x, &y).unwrap();

        // Predictions should match
        let pred_inc = model_inc.predict(&x).unwrap();
        let pred_batch = model_batch.predict(&x).unwrap();
        for i in 0..8 {
            assert_eq!(
                pred_inc[i], pred_batch[i],
                "Incremental and batch predictions should match at sample {}",
                i
            );
        }

        // Class counts should match
        let cc_inc = model_inc.class_count().unwrap();
        let cc_batch = model_batch.class_count().unwrap();
        assert_relative_eq!(cc_inc[0], cc_batch[0], epsilon = 1e-10);
        assert_relative_eq!(cc_inc[1], cc_batch[1], epsilon = 1e-10);
    }

    #[test]
    fn test_categorical_nb_partial_fit_new_categories() {
        // First batch has categories {0, 1} for feature 0
        let x1 = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        let y1 = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = CategoricalNB::new();
        model.partial_fit(&x1, &y1, Some(vec![0.0, 1.0])).unwrap();

        // Second batch introduces category 2
        let x2 = Array2::from_shape_vec((2, 1), vec![2.0, 2.0]).unwrap();
        let y2 = Array1::from_vec(vec![1.0, 1.0]);

        model.partial_fit(&x2, &y2, None).unwrap();

        // Feature should now have 3 categories
        let cats = model.feature_categories().unwrap();
        assert_eq!(cats[0].len(), 3);
        assert_relative_eq!(cats[0][0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cats[0][1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(cats[0][2], 2.0, epsilon = 1e-10);

        // Category 2 should have counts only for class 1
        let cc = model.category_count().unwrap();
        assert_eq!(cc[0][[0, 2]], 0.0); // class 0, cat 2 = 0
        assert_eq!(cc[0][[1, 2]], 2.0); // class 1, cat 2 = 2
    }

    #[test]
    fn test_categorical_nb_single_feature() {
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = CategoricalNB::new();
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x).unwrap();
        for i in 0..3 {
            assert_eq!(pred[i], 0.0);
        }
        for i in 3..6 {
            assert_eq!(pred[i], 1.0);
        }
    }

    #[test]
    fn test_categorical_nb_binary() {
        // Binary features (0/1) - should work just like BernoulliNB without binarization
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 0.0, 1.0, 0.0, 1.0, 1.0, // class 0
                0.0, 1.0, 0.0, 1.0, 0.0, 0.0, // class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = CategoricalNB::new();
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x).unwrap();
        for i in 0..3 {
            assert_eq!(pred[i], 0.0, "Sample {} should be class 0", i);
        }
        for i in 3..6 {
            assert_eq!(pred[i], 1.0, "Sample {} should be class 1", i);
        }
    }

    #[test]
    fn test_categorical_nb_multiclass() {
        // 4 classes, 2 features
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // class 0 -> (0, 0)
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // class 1 -> (1, 1)
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, // class 2 -> (2, 2)
                3.0, 3.0, 3.0, 3.0, 3.0, 3.0, // class 3 -> (3, 3)
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
        ]);

        let mut model = CategoricalNB::new();
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x).unwrap();
        for i in 0..3 {
            assert_eq!(pred[i], 0.0);
        }
        for i in 3..6 {
            assert_eq!(pred[i], 1.0);
        }
        for i in 6..9 {
            assert_eq!(pred[i], 2.0);
        }
        for i in 9..12 {
            assert_eq!(pred[i], 3.0);
        }

        let probas = model.predict_proba(&x).unwrap();
        assert_eq!(probas.ncols(), 4);
        for i in 0..12 {
            let sum: f64 = probas.row(i).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_categorical_nb_many_categories() {
        // 1 feature with 10 categories
        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();

        // Class 0: categories 0-4
        for cat in 0..5 {
            for _ in 0..3 {
                x_vals.push(cat as f64);
                y_vals.push(0.0);
            }
        }
        // Class 1: categories 5-9
        for cat in 5..10 {
            for _ in 0..3 {
                x_vals.push(cat as f64);
                y_vals.push(1.0);
            }
        }

        let x = Array2::from_shape_vec((30, 1), x_vals).unwrap();
        let y = Array1::from_vec(y_vals);

        let mut model = CategoricalNB::new();
        model.fit(&x, &y).unwrap();

        let cats = model.feature_categories().unwrap();
        assert_eq!(cats[0].len(), 10);

        // Test prediction
        let x_test = Array2::from_shape_vec((2, 1), vec![2.0, 7.0]).unwrap();
        let pred = model.predict(&x_test).unwrap();
        assert_eq!(pred[0], 0.0); // cat 2 -> class 0
        assert_eq!(pred[1], 1.0); // cat 7 -> class 1
    }

    #[test]
    fn test_categorical_nb_not_fitted_error() {
        let model = CategoricalNB::new();
        let x = Array2::from_shape_vec((2, 2), vec![0.0; 4]).unwrap();

        assert!(model.predict(&x).is_err());
        assert!(model.predict_proba(&x).is_err());
    }

    #[test]
    fn test_categorical_nb_empty_input() {
        let x = Array2::from_shape_vec((0, 2), vec![]).unwrap();
        let y = Array1::from_vec(vec![]);

        let mut model = CategoricalNB::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_categorical_nb_class_count() {
        let (x, y) = create_categorical_dataset();

        let mut model = CategoricalNB::new();
        model.fit(&x, &y).unwrap();

        let class_count = model.class_count().unwrap();
        assert_eq!(class_count[0], 4.0);
        assert_eq!(class_count[1], 4.0);
    }

    #[test]
    fn test_categorical_nb_feature_log_prob_shape() {
        let (x, y) = create_categorical_dataset();

        let mut model = CategoricalNB::new();
        model.fit(&x, &y).unwrap();

        let flp = model.feature_log_prob().unwrap();
        assert_eq!(flp.len(), 2); // 2 features

        // Feature 0 has 3 categories (0, 1, 2)
        assert_eq!(flp[0].nrows(), 2); // 2 classes
        assert_eq!(flp[0].ncols(), 3); // 3 categories

        // Feature 1 has 2 categories (0, 1)
        assert_eq!(flp[1].nrows(), 2); // 2 classes
        assert_eq!(flp[1].ncols(), 2); // 2 categories
    }

    #[test]
    fn test_categorical_nb_deterministic() {
        let (x, y) = create_categorical_dataset();

        let mut model1 = CategoricalNB::new();
        model1.fit(&x, &y).unwrap();
        let pred1 = model1.predict(&x).unwrap();
        let probas1 = model1.predict_proba(&x).unwrap();

        let mut model2 = CategoricalNB::new();
        model2.fit(&x, &y).unwrap();
        let pred2 = model2.predict(&x).unwrap();
        let probas2 = model2.predict_proba(&x).unwrap();

        for i in 0..8 {
            assert_eq!(pred1[i], pred2[i]);
            for j in 0..2 {
                assert_relative_eq!(probas1[[i, j]], probas2[[i, j]], epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn test_categorical_nb_search_space() {
        let model = CategoricalNB::new();
        let space = model.search_space();
        assert!(!space.parameters.is_empty());
    }

    #[test]
    fn test_categorical_nb_is_fitted() {
        let (x, y) = create_categorical_dataset();
        let mut model = CategoricalNB::new();

        assert!(!model.is_fitted());
        model.fit(&x, &y).unwrap();
        assert!(model.is_fitted());
    }

    #[test]
    fn test_categorical_nb_category_count_values() {
        let (x, y) = create_categorical_dataset();

        let mut model = CategoricalNB::new();
        model.fit(&x, &y).unwrap();

        let cc = model.category_count().unwrap();

        // Feature 0 categories: [0, 1, 2]
        // Class 0: {0:2, 1:2, 2:0}
        // Class 1: {0:0, 1:1, 2:3}
        assert_eq!(cc[0][[0, 0]], 2.0);
        assert_eq!(cc[0][[0, 1]], 2.0);
        assert_eq!(cc[0][[0, 2]], 0.0);
        assert_eq!(cc[0][[1, 0]], 0.0);
        assert_eq!(cc[0][[1, 1]], 1.0);
        assert_eq!(cc[0][[1, 2]], 3.0);

        // Feature 1 categories: [0, 1]
        // Class 0: {0:4, 1:0}
        // Class 1: {0:0, 1:4}
        assert_eq!(cc[1][[0, 0]], 4.0);
        assert_eq!(cc[1][[0, 1]], 0.0);
        assert_eq!(cc[1][[1, 0]], 0.0);
        assert_eq!(cc[1][[1, 1]], 4.0);
    }

    #[test]
    fn test_categorical_nb_two_features() {
        // Two independent features: each alone predicts the class
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // class 0
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut model = CategoricalNB::new();
        model.fit(&x, &y).unwrap();

        // Both features agree -> high confidence prediction
        let x_test = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let probas = model.predict_proba(&x_test).unwrap();

        assert!(probas[[0, 0]] > 0.9); // strongly class 0
        assert!(probas[[1, 1]] > 0.9); // strongly class 1
    }

    #[test]
    fn test_categorical_nb_predict_matches_proba() {
        let (x, y) = create_categorical_dataset();

        let mut model = CategoricalNB::new();
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x).unwrap();
        let probas = model.predict_proba(&x).unwrap();
        let classes = model.classes().unwrap();

        for i in 0..8 {
            // Find argmax of probabilities
            let max_idx = probas
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            assert_eq!(
                pred[i], classes[max_idx],
                "predict should match argmax of predict_proba at sample {}",
                i
            );
        }
    }

    #[test]
    fn test_categorical_nb_single_class_error() {
        let x = Array2::from_shape_vec((4, 2), vec![0.0; 8]).unwrap();
        let y = Array1::from_vec(vec![0.0; 4]);

        let mut model = CategoricalNB::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_categorical_nb_n_features() {
        let (x, y) = create_categorical_dataset();

        let mut model = CategoricalNB::new();
        assert_eq!(model.n_features(), None);

        model.fit(&x, &y).unwrap();
        assert_eq!(model.n_features(), Some(2));
    }

    #[test]
    fn test_categorical_nb_feature_importance() {
        let (x, y) = create_categorical_dataset();

        let mut model = CategoricalNB::new();
        model.fit(&x, &y).unwrap();

        let importance = model.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // Both features should have non-zero importance
        assert!(importance[0] > 0.0);
        assert!(importance[1] > 0.0);
    }

    #[test]
    fn test_categorical_nb_feature_mismatch_error() {
        let (x, y) = create_categorical_dataset();

        let mut model = CategoricalNB::new();
        model.fit(&x, &y).unwrap();

        // Try to predict with wrong number of features
        let x_bad = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        assert!(model.predict(&x_bad).is_err());
    }

    #[test]
    fn test_gaussian_nb_single_sample_per_class() {
        // Single sample per class → zero variance before smoothing.
        // Without the max_var floor fix, epsilon=0 and log(var)=-inf → NaN predictions.
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0]);

        let mut model = GaussianNB::new();
        let result = model.fit(&x, &y);
        assert!(
            result.is_ok(),
            "GaussianNB should handle single-sample classes: {:?}",
            result.err()
        );

        // Predictions must be valid class labels, not NaN
        let preds = model.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(!p.is_nan(), "prediction should not be NaN");
            assert!(
                p == 0.0 || p == 1.0,
                "prediction should be a valid class label, got {}",
                p
            );
        }

        // Probabilities should sum to 1 and not contain NaN
        let proba = model.predict_proba(&x).unwrap();
        for row in proba.rows() {
            let sum: f64 = row.iter().sum();
            assert!(!sum.is_nan(), "probability sum should not be NaN");
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "probabilities should sum to 1, got {}",
                sum
            );
        }
    }
}

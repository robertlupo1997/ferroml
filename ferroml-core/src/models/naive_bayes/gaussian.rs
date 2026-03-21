use crate::hpo::SearchSpace;
use crate::linalg::logsumexp;
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, ClassWeight, Model,
    PredictionInterval, ProbabilisticModel,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use super::{compute_variance, z_critical};

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
        let classes = crate::models::get_unique_classes(y);

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

        crate::models::validate_output(&predictions, "GaussianNB")?;
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

        let n_samples = jll.nrows();
        let n_classes = jll.ncols();
        let mut probas = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = jll.row(i);
            let log_sum = logsumexp(row.as_slice().unwrap_or(&row.to_vec()));
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
// SparseModel Implementation
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

        let classes = crate::models::get_unique_classes(y);
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
        #[allow(unused_variables)]
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

// =============================================================================
// PipelineSparseModel Implementation
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ProbabilisticModel;
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

    #[test]
    fn test_gaussian_nb_single_sample_per_class() {
        // Single sample per class -> zero variance before smoothing.
        // Without the max_var floor fix, epsilon=0 and log(var)=-inf -> NaN predictions.
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

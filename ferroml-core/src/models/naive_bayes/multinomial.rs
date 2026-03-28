use crate::hpo::SearchSpace;
use crate::linalg::logsumexp;
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, ClassWeight, Model,
    PredictionInterval, ProbabilisticModel,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::z_critical;

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

        if self.alpha < 0.0 {
            return Err(FerroError::invalid_input("alpha must be non-negative"));
        }

        // Validate non-negative values
        if x.iter().any(|&v| v < 0.0) {
            return Err(FerroError::invalid_input(
                "MultinomialNB requires non-negative feature values",
            ));
        }

        // Find unique classes
        let classes = crate::models::get_unique_classes(y);

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
// SparseModel Implementation
// =============================================================================

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

        let classes = crate::models::get_unique_classes(y);
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

// =============================================================================
// PipelineSparseModel Implementation
// =============================================================================

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ProbabilisticModel;
    use approx::assert_relative_eq;

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
}

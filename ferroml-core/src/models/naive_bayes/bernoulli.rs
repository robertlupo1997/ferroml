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

        if self.alpha < 0.0 {
            return Err(FerroError::invalid_input("alpha must be non-negative"));
        }

        // Find unique classes
        let classes = crate::models::get_unique_classes(y);

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

        let classes = crate::models::get_unique_classes(y);
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

// =============================================================================
// PipelineSparseModel Implementation
// =============================================================================

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ProbabilisticModel;
    use approx::assert_relative_eq;

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
}

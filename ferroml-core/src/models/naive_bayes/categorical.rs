use crate::hpo::SearchSpace;
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, Model, PredictionInterval,
    ProbabilisticModel,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::z_critical;

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
    pub(crate) fn category_index(categories: &[f64], value: f64) -> Option<usize> {
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
                    log_prob[[class_idx, cat_idx]] = prob.max(1e-300).ln();
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
        let classes = crate::models::get_unique_classes(y);

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
// SparseModel Implementation
// =============================================================================

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

        let classes = crate::models::get_unique_classes(y);
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
// PipelineSparseModel Implementation
// =============================================================================

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
    use crate::models::ProbabilisticModel;
    use approx::assert_relative_eq;

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
    fn test_categorical_nb_zero_count_no_nan() {
        // Create data where alpha=0 would produce zero probabilities
        // Use very small alpha to push probabilities near zero
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, // class 0
                0.0, 0.0, // class 0
                0.0, 0.0, // class 0
                1.0, 1.0, // class 1
                1.0, 1.0, // class 1
                1.0, 1.0, // class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        // With very small alpha, P(x_j=1 | class=0) ~ alpha / (3 + 2*alpha) near 0
        let mut model = CategoricalNB::new().with_alpha(1e-300);
        model.fit(&x, &y).unwrap();

        // Predict on data with unseen category combinations
        let x_test = Array2::from_shape_vec(
            (2, 2),
            vec![
                1.0, 0.0, // mixed
                0.0, 1.0, // mixed
            ],
        )
        .unwrap();
        let preds = model.predict(&x_test).unwrap();
        for &p in preds.iter() {
            assert!(p.is_finite(), "Prediction must be finite, got {}", p);
        }

        let proba = model.predict_proba(&x_test).unwrap();
        for row in proba.rows() {
            for &p in row.iter() {
                assert!(!p.is_nan(), "Probability must not be NaN");
                assert!(p.is_finite(), "Probability must be finite");
            }
        }
    }
}

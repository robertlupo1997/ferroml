//! Feature Selection Transformers
//!
//! This module provides transformers for selecting important features.
//!
//! ## Available Selectors
//!
//! - [`VarianceThreshold`] - Remove low-variance features
//! - [`SelectKBest`] - Select top k features by statistical score
//! - [`SelectFromModel`] - Select features based on model importance
//!
//! ## Selection Methods
//!
//! | Method | Best For | Notes |
//! |--------|----------|-------|
//! | `VarianceThreshold` | Removing constant features | Fast, simple |
//! | `SelectKBest` | Statistical feature ranking | Uses F-test, chi2, etc. |
//! | `SelectFromModel` | Model-based importance | Uses tree importance, etc. |
//!
//! ## Statistical Tests for SelectKBest
//!
//! - **f_classif**: ANOVA F-value for classification
//! - **f_regression**: F-value for regression (correlation-based)
//! - **chi2**: Chi-squared stats for non-negative features
//!
//! ## Example
//!
//! ```
//! use ferroml_core::preprocessing::selection::{VarianceThreshold, SelectKBest, ScoreFunction};
//! use ferroml_core::preprocessing::Transformer;
//! use ndarray::array;
//!
//! // Remove constant features
//! let mut selector = VarianceThreshold::new(0.0);
//! let x = array![[1.0, 5.0, 3.0], [1.0, 2.0, 3.0], [1.0, 8.0, 3.0]];
//! let x_selected = selector.fit_transform(&x).unwrap();
//! assert_eq!(x_selected.ncols(), 1); // Only column 1 has variance > 0
//! ```

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{check_is_fitted, check_non_empty, check_shape, generate_feature_names, Transformer};
use crate::{FerroError, Result};

/// Remove features with variance below a threshold.
///
/// Features with variance lower than or equal to the threshold are removed.
/// This is useful for removing constant or near-constant features that provide
/// no discriminative information.
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::selection::VarianceThreshold;
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// let mut selector = VarianceThreshold::new(0.0);
/// let x = array![
///     [1.0, 5.0, 3.0],
///     [1.0, 2.0, 3.0],
///     [1.0, 8.0, 3.0]
/// ];
///
/// let x_selected = selector.fit_transform(&x).unwrap();
/// assert_eq!(x_selected.ncols(), 1); // Only column 1 (variance > 0) is kept
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceThreshold {
    /// Variance threshold (features with variance <= threshold are removed)
    threshold: f64,
    /// Variance of each feature (learned during fit)
    variances: Option<Array1<f64>>,
    /// Indices of selected features (variance > threshold)
    selected_indices: Option<Vec<usize>>,
    /// Number of features seen during fit
    n_features_in: Option<usize>,
}

impl VarianceThreshold {
    /// Create a new VarianceThreshold selector.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Features with variance <= threshold will be removed.
    ///   Use 0.0 to remove only constant features.
    ///
    /// # Panics
    ///
    /// Panics if threshold is negative.
    pub fn new(threshold: f64) -> Self {
        assert!(threshold >= 0.0, "Variance threshold must be non-negative");
        Self {
            threshold,
            variances: None,
            selected_indices: None,
            n_features_in: None,
        }
    }

    /// Get the variance of each input feature.
    ///
    /// Returns `None` if not fitted.
    pub fn variances(&self) -> Option<&Array1<f64>> {
        self.variances.as_ref()
    }

    /// Get the indices of selected features.
    ///
    /// Returns `None` if not fitted.
    pub fn selected_indices(&self) -> Option<&[usize]> {
        self.selected_indices.as_deref()
    }

    /// Get a boolean mask of which features are selected.
    ///
    /// Returns `None` if not fitted.
    pub fn get_support(&self) -> Option<Vec<bool>> {
        let n_features = self.n_features_in?;
        let selected = self.selected_indices.as_ref()?;
        let mut mask = vec![false; n_features];
        for &idx in selected {
            mask[idx] = true;
        }
        Some(mask)
    }
}

impl Default for VarianceThreshold {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Transformer for VarianceThreshold {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        let n_features = x.ncols();
        let variances = x.var_axis(Axis(0), 0.0);

        // Find features with variance > threshold
        let selected_indices: Vec<usize> = variances
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v > self.threshold { Some(i) } else { None })
            .collect();

        if selected_indices.is_empty() {
            return Err(FerroError::invalid_input(format!(
                "No features have variance > {}. All features would be removed.",
                self.threshold
            )));
        }

        self.variances = Some(variances);
        self.selected_indices = Some(selected_indices);
        self.n_features_in = Some(n_features);

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let selected = self.selected_indices.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected.len();

        let mut result = Array2::zeros((n_samples, n_selected));
        for (new_j, &old_j) in selected.iter().enumerate() {
            result.column_mut(new_j).assign(&x.column(old_j));
        }

        Ok(result)
    }

    fn is_fitted(&self) -> bool {
        self.variances.is_some() && self.selected_indices.is_some()
    }

    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_features = self.n_features_in?;
        let selected = self.selected_indices.as_ref()?;

        let input_names = input_names
            .map(|n| n.to_vec())
            .unwrap_or_else(|| generate_feature_names(n_features));

        Some(selected.iter().map(|&i| input_names[i].clone()).collect())
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        self.selected_indices.as_ref().map(|s| s.len())
    }
}

/// Score function type for feature selection.
///
/// Different score functions are suitable for different problem types:
/// - `FClassif`: ANOVA F-value for classification problems
/// - `FRegression`: F-value based on correlation for regression problems
/// - `Chi2`: Chi-squared statistic for non-negative features in classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreFunction {
    /// ANOVA F-value for classification (compares group means)
    FClassif,
    /// F-value for regression (based on Pearson correlation)
    FRegression,
    /// Chi-squared statistic for classification with non-negative features
    Chi2,
}

impl Default for ScoreFunction {
    fn default() -> Self {
        Self::FClassif
    }
}

/// Result of feature scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureScores {
    /// Score for each feature
    pub scores: Array1<f64>,
    /// p-value for each feature (if applicable)
    pub p_values: Option<Array1<f64>>,
}

/// Select top k features by statistical score.
///
/// This transformer scores each feature using a statistical test and selects
/// the k features with the highest scores.
///
/// # Score Functions
///
/// - `FClassif`: ANOVA F-value between feature and target classes
/// - `FRegression`: F-value based on Pearson correlation with target
/// - `Chi2`: Chi-squared statistic (features must be non-negative)
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::selection::{SelectKBest, ScoreFunction};
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::{array, Array1};
///
/// let mut selector = SelectKBest::new(ScoreFunction::FRegression, 2);
/// let x = array![
///     [1.0, 2.0, 0.5],
///     [2.0, 4.0, 0.3],
///     [3.0, 6.0, 0.8],
///     [4.0, 8.0, 0.2]
/// ];
/// let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
///
/// selector.fit_with_target(&x, &y).unwrap();
/// let x_selected = selector.transform(&x).unwrap();
/// assert_eq!(x_selected.ncols(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectKBest {
    /// Score function to use
    score_func: ScoreFunction,
    /// Number of top features to select
    k: usize,
    /// Scores for each feature (learned during fit)
    scores: Option<FeatureScores>,
    /// Indices of selected features
    selected_indices: Option<Vec<usize>>,
    /// Number of features seen during fit
    n_features_in: Option<usize>,
}

impl SelectKBest {
    /// Create a new SelectKBest selector.
    ///
    /// # Arguments
    ///
    /// * `score_func` - The statistical test to use for scoring features
    /// * `k` - Number of top features to select
    ///
    /// # Panics
    ///
    /// Panics if k is 0.
    pub fn new(score_func: ScoreFunction, k: usize) -> Self {
        assert!(k > 0, "k must be at least 1");
        Self {
            score_func,
            k,
            scores: None,
            selected_indices: None,
            n_features_in: None,
        }
    }

    /// Fit the selector with features and target.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - Target array of shape (n_samples,)
    pub fn fit_with_target(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        check_non_empty(x)?;

        if x.nrows() != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("x samples: {}", x.nrows()),
                format!("y length: {}", y.len()),
            ));
        }

        let n_features = x.ncols();
        let k = self.k.min(n_features); // Can't select more features than exist

        // Compute scores based on score function
        let scores = match self.score_func {
            ScoreFunction::FClassif => compute_f_classif(x, y)?,
            ScoreFunction::FRegression => compute_f_regression(x, y)?,
            ScoreFunction::Chi2 => compute_chi2(x, y)?,
        };

        // Select top k features by score
        let mut indices_scores: Vec<(usize, f64)> = scores
            .scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, if s.is_nan() { f64::NEG_INFINITY } else { s }))
            .collect();
        indices_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected_indices: Vec<usize> = indices_scores.iter().take(k).map(|(i, _)| *i).collect();

        // Sort indices for consistent ordering
        let mut selected_indices = selected_indices;
        selected_indices.sort();

        self.scores = Some(scores);
        self.selected_indices = Some(selected_indices);
        self.n_features_in = Some(n_features);

        Ok(())
    }

    /// Get the scores for each feature.
    ///
    /// Returns `None` if not fitted.
    pub fn scores(&self) -> Option<&FeatureScores> {
        self.scores.as_ref()
    }

    /// Get the indices of selected features.
    ///
    /// Returns `None` if not fitted.
    pub fn selected_indices(&self) -> Option<&[usize]> {
        self.selected_indices.as_deref()
    }

    /// Get a boolean mask of which features are selected.
    ///
    /// Returns `None` if not fitted.
    pub fn get_support(&self) -> Option<Vec<bool>> {
        let n_features = self.n_features_in?;
        let selected = self.selected_indices.as_ref()?;
        let mut mask = vec![false; n_features];
        for &idx in selected {
            mask[idx] = true;
        }
        Some(mask)
    }

    /// Get p-values for each feature (if available).
    pub fn p_values(&self) -> Option<&Array1<f64>> {
        self.scores.as_ref()?.p_values.as_ref()
    }
}

impl Transformer for SelectKBest {
    fn fit(&mut self, _x: &Array2<f64>) -> Result<()> {
        // SelectKBest requires a target, so this is a fallback that creates a dummy
        // This allows it to work in pipelines where fit() is called without target
        Err(FerroError::invalid_input(
            "SelectKBest requires a target variable. Use fit_with_target() instead.",
        ))
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let selected = self.selected_indices.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected.len();

        let mut result = Array2::zeros((n_samples, n_selected));
        for (new_j, &old_j) in selected.iter().enumerate() {
            result.column_mut(new_j).assign(&x.column(old_j));
        }

        Ok(result)
    }

    fn is_fitted(&self) -> bool {
        self.scores.is_some() && self.selected_indices.is_some()
    }

    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_features = self.n_features_in?;
        let selected = self.selected_indices.as_ref()?;

        let input_names = input_names
            .map(|n| n.to_vec())
            .unwrap_or_else(|| generate_feature_names(n_features));

        Some(selected.iter().map(|&i| input_names[i].clone()).collect())
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        self.selected_indices.as_ref().map(|s| s.len())
    }
}

/// Importance threshold for SelectFromModel.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ImportanceThreshold {
    /// Select features with importance >= mean importance
    Mean,
    /// Select features with importance >= median importance
    Median,
    /// Select features with importance >= given value
    Value(f64),
    /// Select features with importance >= mean + factor * std
    MeanPlusStd(f64),
}

impl Default for ImportanceThreshold {
    fn default() -> Self {
        Self::Mean
    }
}

/// Select features based on model feature importances.
///
/// This transformer selects features based on importance values provided
/// by a model (e.g., feature_importances_ from tree models or coefficients
/// from linear models).
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::selection::{SelectFromModel, ImportanceThreshold};
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::{array, Array1};
///
/// let importances = Array1::from_vec(vec![0.1, 0.5, 0.2, 0.8, 0.05]);
/// let mut selector = SelectFromModel::new(importances, ImportanceThreshold::Mean);
///
/// let x = array![
///     [1.0, 2.0, 3.0, 4.0, 5.0],
///     [6.0, 7.0, 8.0, 9.0, 10.0]
/// ];
///
/// selector.fit(&x).unwrap();
/// let x_selected = selector.transform(&x).unwrap();
/// // Features with importance >= mean(0.33) are selected: indices 1, 3 (and possibly 2)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectFromModel {
    /// Feature importances from the model
    importances: Array1<f64>,
    /// Threshold for selection
    threshold: ImportanceThreshold,
    /// Computed threshold value
    threshold_value: Option<f64>,
    /// Indices of selected features
    selected_indices: Option<Vec<usize>>,
    /// Number of features seen during fit
    n_features_in: Option<usize>,
    /// Optional maximum number of features to select
    max_features: Option<usize>,
}

impl SelectFromModel {
    /// Create a new SelectFromModel selector.
    ///
    /// # Arguments
    ///
    /// * `importances` - Feature importance values from a fitted model
    /// * `threshold` - Threshold for selecting features
    pub fn new(importances: Array1<f64>, threshold: ImportanceThreshold) -> Self {
        Self {
            importances,
            threshold,
            threshold_value: None,
            selected_indices: None,
            n_features_in: None,
            max_features: None,
        }
    }

    /// Set the maximum number of features to select.
    ///
    /// If set, even if more features pass the threshold, only the top
    /// `max_features` by importance will be selected.
    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = Some(max_features);
        self
    }

    /// Get the computed threshold value.
    pub fn threshold_value(&self) -> Option<f64> {
        self.threshold_value
    }

    /// Get the feature importances.
    pub fn importances(&self) -> &Array1<f64> {
        &self.importances
    }

    /// Get the indices of selected features.
    pub fn selected_indices(&self) -> Option<&[usize]> {
        self.selected_indices.as_deref()
    }

    /// Get a boolean mask of which features are selected.
    pub fn get_support(&self) -> Option<Vec<bool>> {
        let n_features = self.n_features_in?;
        let selected = self.selected_indices.as_ref()?;
        let mut mask = vec![false; n_features];
        for &idx in selected {
            mask[idx] = true;
        }
        Some(mask)
    }
}

impl Transformer for SelectFromModel {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        let n_features = x.ncols();

        if self.importances.len() != n_features {
            return Err(FerroError::shape_mismatch(
                format!("importances length: {}", self.importances.len()),
                format!("n_features: {}", n_features),
            ));
        }

        // Compute threshold value
        let threshold_value = match self.threshold {
            ImportanceThreshold::Mean => self.importances.mean().unwrap_or(0.0),
            ImportanceThreshold::Median => {
                let mut sorted: Vec<f64> = self.importances.iter().copied().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sorted.len();
                if n % 2 == 0 {
                    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                } else {
                    sorted[n / 2]
                }
            }
            ImportanceThreshold::Value(v) => v,
            ImportanceThreshold::MeanPlusStd(factor) => {
                let mean = self.importances.mean().unwrap_or(0.0);
                let std = self.importances.std(1.0);
                factor.mul_add(std, mean)
            }
        };

        // Select features with importance >= threshold
        let mut indices_importances: Vec<(usize, f64)> = self
            .importances
            .iter()
            .enumerate()
            .filter(|(_, &imp)| imp >= threshold_value)
            .map(|(i, &imp)| (i, imp))
            .collect();

        // Sort by importance (descending) for max_features selection
        indices_importances
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply max_features limit if set
        if let Some(max_feat) = self.max_features {
            indices_importances.truncate(max_feat);
        }

        if indices_importances.is_empty() {
            return Err(FerroError::invalid_input(format!(
                "No features have importance >= threshold ({:.4}). All features would be removed.",
                threshold_value
            )));
        }

        // Get indices and sort for consistent ordering
        let mut selected_indices: Vec<usize> =
            indices_importances.iter().map(|(i, _)| *i).collect();
        selected_indices.sort();

        self.threshold_value = Some(threshold_value);
        self.selected_indices = Some(selected_indices);
        self.n_features_in = Some(n_features);

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let selected = self.selected_indices.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected.len();

        let mut result = Array2::zeros((n_samples, n_selected));
        for (new_j, &old_j) in selected.iter().enumerate() {
            result.column_mut(new_j).assign(&x.column(old_j));
        }

        Ok(result)
    }

    fn is_fitted(&self) -> bool {
        self.selected_indices.is_some()
    }

    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_features = self.n_features_in?;
        let selected = self.selected_indices.as_ref()?;

        let input_names = input_names
            .map(|n| n.to_vec())
            .unwrap_or_else(|| generate_feature_names(n_features));

        Some(selected.iter().map(|&i| input_names[i].clone()).collect())
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        self.selected_indices.as_ref().map(|s| s.len())
    }
}

// ============================================================================
// Score Functions
// ============================================================================

/// Compute ANOVA F-value for classification.
///
/// The F-value measures the ratio of variance between classes to variance within classes.
/// Higher F-values indicate the feature is more discriminative for the target classes.
fn compute_f_classif(x: &Array2<f64>, y: &Array1<f64>) -> Result<FeatureScores> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 2 {
        return Err(FerroError::invalid_input(
            "Need at least 2 samples for f_classif",
        ));
    }

    // Get unique classes
    let mut classes: Vec<f64> = y.iter().copied().collect();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    classes.dedup();

    let n_classes = classes.len();
    if n_classes < 2 {
        return Err(FerroError::invalid_input(
            "Need at least 2 classes for f_classif",
        ));
    }

    // Group samples by class
    let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in y.iter().enumerate() {
        let class_idx = classes.iter().position(|&c| c == label).unwrap();
        class_indices.entry(class_idx).or_default().push(i);
    }

    let mut scores = Array1::zeros(n_features);
    let mut p_values = Array1::zeros(n_features);

    for j in 0..n_features {
        let col = x.column(j);
        let grand_mean = col.mean().unwrap_or(0.0);

        // Between-group sum of squares
        let mut ss_between = 0.0;
        // Within-group sum of squares
        let mut ss_within = 0.0;

        for class_idx in 0..n_classes {
            let indices = &class_indices[&class_idx];
            let n_k = indices.len() as f64;

            // Class mean
            let class_mean: f64 = indices.iter().map(|&i| col[i]).sum::<f64>() / n_k;

            // Between-group: n_k * (class_mean - grand_mean)^2
            ss_between += n_k * (class_mean - grand_mean).powi(2);

            // Within-group: sum of (x_i - class_mean)^2
            for &i in indices {
                ss_within += (col[i] - class_mean).powi(2);
            }
        }

        let df_between = (n_classes - 1) as f64;
        let df_within = (n_samples - n_classes) as f64;

        if df_within <= 0.0 || ss_within < 1e-15 {
            scores[j] = f64::NAN;
            p_values[j] = 1.0;
        } else {
            let ms_between = ss_between / df_between;
            let ms_within = ss_within / df_within;
            let f_stat = ms_between / ms_within;

            scores[j] = f_stat;
            p_values[j] = 1.0 - f_cdf(f_stat, df_between, df_within);
        }
    }

    Ok(FeatureScores {
        scores,
        p_values: Some(p_values),
    })
}

/// Compute F-value for regression (based on Pearson correlation).
///
/// For regression, we use the F-statistic from the linear regression of y on each feature.
/// F = (r^2 / 1) / ((1 - r^2) / (n - 2)) = r^2 * (n - 2) / (1 - r^2)
fn compute_f_regression(x: &Array2<f64>, y: &Array1<f64>) -> Result<FeatureScores> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 3 {
        return Err(FerroError::invalid_input(
            "Need at least 3 samples for f_regression",
        ));
    }

    let y_mean = y.mean().unwrap_or(0.0);
    let y_centered: Array1<f64> = y - y_mean;
    let y_ss: f64 = y_centered.iter().map(|&v| v * v).sum();

    let mut scores = Array1::zeros(n_features);
    let mut p_values = Array1::zeros(n_features);

    for j in 0..n_features {
        let x_col = x.column(j);
        let x_mean = x_col.mean().unwrap_or(0.0);
        let x_centered: Array1<f64> = x_col.to_owned() - x_mean;
        let x_ss: f64 = x_centered.iter().map(|&v| v * v).sum();

        if x_ss < 1e-15 || y_ss < 1e-15 {
            scores[j] = f64::NAN;
            p_values[j] = 1.0;
        } else {
            // Pearson correlation
            let xy_sum: f64 = x_centered
                .iter()
                .zip(y_centered.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let r = xy_sum / (x_ss.sqrt() * y_ss.sqrt());
            let r_squared = r * r;

            // F-statistic
            let df = (n_samples - 2) as f64;
            if (1.0 - r_squared).abs() < 1e-15 {
                scores[j] = f64::INFINITY;
                p_values[j] = 0.0;
            } else {
                let f_stat = r_squared * df / (1.0 - r_squared);
                scores[j] = f_stat;
                p_values[j] = 1.0 - f_cdf(f_stat, 1.0, df);
            }
        }
    }

    Ok(FeatureScores {
        scores,
        p_values: Some(p_values),
    })
}

/// Compute chi-squared statistic for classification.
///
/// Features must be non-negative (counts or frequencies).
/// Chi-squared measures the independence between feature and target.
fn compute_chi2(x: &Array2<f64>, y: &Array1<f64>) -> Result<FeatureScores> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Check for non-negative values
    for j in 0..n_features {
        for i in 0..n_samples {
            if x[[i, j]] < 0.0 {
                return Err(FerroError::invalid_input(format!(
                    "Chi-squared requires non-negative features. Found negative value at ({}, {})",
                    i, j
                )));
            }
        }
    }

    // Get unique classes
    let mut classes: Vec<f64> = y.iter().copied().collect();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    classes.dedup();

    let n_classes = classes.len();
    if n_classes < 2 {
        return Err(FerroError::invalid_input(
            "Need at least 2 classes for chi2",
        ));
    }

    // Compute observed and expected frequencies
    let mut scores = Array1::zeros(n_features);
    let mut p_values = Array1::zeros(n_features);

    for j in 0..n_features {
        let col = x.column(j);

        // Sum by class (observed)
        let mut class_sums: Vec<f64> = vec![0.0; n_classes];
        let mut class_counts: Vec<usize> = vec![0; n_classes];

        for (i, &label) in y.iter().enumerate() {
            let class_idx = classes.iter().position(|&c| c == label).unwrap();
            class_sums[class_idx] += col[i];
            class_counts[class_idx] += 1;
        }

        let total_sum: f64 = class_sums.iter().sum();

        if total_sum < 1e-15 {
            scores[j] = 0.0;
            p_values[j] = 1.0;
            continue;
        }

        // Chi-squared statistic
        let mut chi2 = 0.0;
        for class_idx in 0..n_classes {
            let observed = class_sums[class_idx];
            let expected = total_sum * (class_counts[class_idx] as f64) / (n_samples as f64);

            if expected > 1e-15 {
                chi2 += (observed - expected).powi(2) / expected;
            }
        }

        scores[j] = chi2;
        let df = (n_classes - 1) as f64;
        p_values[j] = 1.0 - chi2_cdf(chi2, df);
    }

    Ok(FeatureScores {
        scores,
        p_values: Some(p_values),
    })
}

// ============================================================================
// Statistical Distribution Functions
// ============================================================================

/// F-distribution CDF approximation
fn f_cdf(f: f64, df1: f64, df2: f64) -> f64 {
    if f <= 0.0 {
        return 0.0;
    }
    let x = df1 * f / df1.mul_add(f, df2);
    incomplete_beta(df1 / 2.0, df2 / 2.0, x)
}

/// Chi-squared CDF approximation
fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Chi-squared is a special case of gamma distribution
    lower_incomplete_gamma(df / 2.0, x / 2.0) / gamma(df / 2.0)
}

/// Lower incomplete gamma function
fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    // Series expansion for small x
    if x < a + 1.0 {
        let mut sum = 1.0 / a;
        let mut term = 1.0 / a;
        for n in 1..100 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < 1e-10 {
                break;
            }
        }
        return sum * x.powf(a) * (-x).exp();
    }

    // Continued fraction for large x
    gamma(a) - upper_incomplete_gamma(a, x)
}

/// Upper incomplete gamma function (continued fraction)
fn upper_incomplete_gamma(a: f64, x: f64) -> f64 {
    let mut c = 1.0;
    let mut d = 1.0 / (x + 1.0 - a);
    let mut h = d;

    for n in 1..100 {
        let an = n as f64 * (a - n as f64);
        let bn = x + (2 * n + 1) as f64 - a;
        d = an.mul_add(d, bn);
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = bn + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = c * d;
        h *= delta;
        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    x.powf(a) * (-x).exp() * h
}

/// Gamma function
fn gamma(x: f64) -> f64 {
    gamma_ln(x).exp()
}

/// Log gamma function (Lanczos approximation)
fn gamma_ln(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    let tmp = (x + 0.5).mul_add(-(x + 5.5).ln(), x + 5.5);
    let mut ser = 1.000000000190015;
    for (i, &c) in coeffs.iter().enumerate() {
        ser += c / (x + i as f64 + 1.0);
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// Incomplete beta function (regularized)
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry for numerical stability
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - incomplete_beta(b, a, 1.0 - x);
    }

    // Continued fraction (Lentz's algorithm)
    let prefix = x.powf(a) * (1.0 - x).powf(b) / (a * beta(a, b));

    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..100 {
        let m = m as f64;

        // Even step
        let num = m * (b - m) * x / ((2.0f64.mul_add(m, a) - 1.0) * 2.0f64.mul_add(m, a));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let num =
            -(a + m) * (a + b + m) * x / (2.0f64.mul_add(m, a) * (2.0f64.mul_add(m, a) + 1.0));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    prefix * h
}

/// Beta function
fn beta(a: f64, b: f64) -> f64 {
    (gamma_ln(a) + gamma_ln(b) - gamma_ln(a + b)).exp()
}

// ============================================================================
// Recursive Feature Elimination (RFE)
// ============================================================================

/// Trait for estimators that can provide feature importances.
///
/// This trait is implemented by models that can be used with `RecursiveFeatureElimination`.
/// Models should provide feature importances after being fitted, which are used to
/// determine which features to eliminate at each iteration.
///
/// # Example Implementation
///
/// ```ignore
/// struct SimpleLinearModel {
///     coefficients: Option<Array1<f64>>,
/// }
///
/// impl FeatureImportanceEstimator for SimpleLinearModel {
///     fn fit_and_get_importances(
///         &mut self,
///         x: &Array2<f64>,
///         y: &Array1<f64>,
///     ) -> Result<Array1<f64>> {
///         // Fit model and return absolute coefficients as importances
///         // ...
///         Ok(self.coefficients.as_ref().unwrap().mapv(f64::abs))
///     }
/// }
/// ```
pub trait FeatureImportanceEstimator: Send + Sync {
    /// Fit the estimator on the given data and return feature importances.
    ///
    /// This method should:
    /// 1. Fit the model on the provided feature subset
    /// 2. Return the absolute importance of each feature
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - Target array of shape (n_samples,)
    ///
    /// # Returns
    ///
    /// Feature importances of shape (n_features,). Larger values indicate
    /// more important features.
    fn fit_and_get_importances(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>>;
}

/// A simple importance provider that uses a closure.
///
/// This allows using RFE with any importance-computing function without
/// implementing the `FeatureImportanceEstimator` trait explicitly.
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::selection::{ClosureEstimator, RecursiveFeatureElimination};
/// use ndarray::{Array1, Array2};
///
/// // Create an estimator from a closure that returns feature variances as importances
/// let estimator = ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| {
///     Ok(x.var_axis(ndarray::Axis(0), 0.0))
/// });
/// ```
pub struct ClosureEstimator<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync,
{
    func: F,
}

impl<F> ClosureEstimator<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync,
{
    /// Create a new closure-based estimator.
    ///
    /// # Arguments
    ///
    /// * `func` - A closure that takes (X, y) and returns feature importances
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> FeatureImportanceEstimator for ClosureEstimator<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync,
{
    fn fit_and_get_importances(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        (self.func)(x, y)
    }
}

/// Recursive Feature Elimination (RFE).
///
/// RFE selects features by recursively fitting an estimator and eliminating
/// the least important features at each iteration until the desired number
/// of features is reached.
///
/// # Algorithm
///
/// 1. Start with all features
/// 2. Fit the estimator on current features
/// 3. Get feature importances from the fitted estimator
/// 4. Remove the `step` features with the lowest importance
/// 5. Repeat steps 2-4 until `n_features_to_select` features remain
/// 6. Track feature rankings (1 = selected, higher = eliminated earlier)
///
/// # Statistical Considerations
///
/// - RFE can be unstable with correlated features; consider using regularization
/// - Feature rankings may vary with different random states in the estimator
/// - Consider using RFECV for automatic selection of the optimal number of features
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::selection::{ClosureEstimator, RecursiveFeatureElimination};
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::{array, Array1, Axis};
///
/// // Create an estimator that uses variance as importance
/// let estimator = ClosureEstimator::new(|x: &ndarray::Array2<f64>, _y: &Array1<f64>| {
///     Ok(x.var_axis(Axis(0), 0.0))
/// });
///
/// // Create RFE to select 2 features, removing 1 at a time
/// let mut rfe = RecursiveFeatureElimination::new(Box::new(estimator))
///     .with_n_features_to_select(2)
///     .with_step(1);
///
/// let x = array![
///     [1.0, 10.0, 100.0],  // Feature 2 has highest variance
///     [1.5, 11.0, 200.0],
///     [2.0, 12.0, 300.0],
///     [2.5, 13.0, 400.0]
/// ];
/// let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
///
/// rfe.fit_with_target(&x, &y).unwrap();
///
/// // Get selected feature indices
/// let selected = rfe.selected_indices().unwrap();
/// assert_eq!(selected.len(), 2);
///
/// // Transform data to keep only selected features
/// let x_selected = rfe.transform(&x).unwrap();
/// assert_eq!(x_selected.ncols(), 2);
/// ```
#[derive(Serialize, Deserialize)]
pub struct RecursiveFeatureElimination {
    /// Number of features to select. If None, selects half of the features.
    n_features_to_select: Option<usize>,
    /// Number of features to remove at each iteration
    step: usize,
    /// Feature rankings (1 = selected/most important, higher = eliminated earlier)
    #[serde(skip)]
    ranking: Option<Array1<usize>>,
    /// Boolean mask of selected features
    support: Option<Vec<bool>>,
    /// Indices of selected features (sorted)
    selected_indices: Option<Vec<usize>>,
    /// Number of features seen during fit
    n_features_in: Option<usize>,
    /// Number of eliminations performed
    n_iterations: Option<usize>,
    /// The estimator used for fitting (not serializable, must be re-provided)
    #[serde(skip)]
    estimator: Option<Box<dyn FeatureImportanceEstimator>>,
}

impl std::fmt::Debug for RecursiveFeatureElimination {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecursiveFeatureElimination")
            .field("n_features_to_select", &self.n_features_to_select)
            .field("step", &self.step)
            .field("ranking", &self.ranking)
            .field("support", &self.support)
            .field("selected_indices", &self.selected_indices)
            .field("n_features_in", &self.n_features_in)
            .field("n_iterations", &self.n_iterations)
            .field("estimator", &"<dyn FeatureImportanceEstimator>")
            .finish()
    }
}

impl Clone for RecursiveFeatureElimination {
    fn clone(&self) -> Self {
        Self {
            n_features_to_select: self.n_features_to_select,
            step: self.step,
            ranking: self.ranking.clone(),
            support: self.support.clone(),
            selected_indices: self.selected_indices.clone(),
            n_features_in: self.n_features_in,
            n_iterations: self.n_iterations,
            estimator: None, // Cannot clone trait object
        }
    }
}

impl RecursiveFeatureElimination {
    /// Create a new RFE selector with the given estimator.
    ///
    /// # Arguments
    ///
    /// * `estimator` - An estimator that implements `FeatureImportanceEstimator`
    ///
    /// # Default Configuration
    ///
    /// - `n_features_to_select`: None (selects half of features)
    /// - `step`: 1 (remove one feature at a time)
    pub fn new(estimator: Box<dyn FeatureImportanceEstimator>) -> Self {
        Self {
            n_features_to_select: None,
            step: 1,
            ranking: None,
            support: None,
            selected_indices: None,
            n_features_in: None,
            n_iterations: None,
            estimator: Some(estimator),
        }
    }

    /// Set the number of features to select.
    ///
    /// If not set, half of the features will be selected.
    pub fn with_n_features_to_select(mut self, n: usize) -> Self {
        assert!(n > 0, "n_features_to_select must be at least 1");
        self.n_features_to_select = Some(n);
        self
    }

    /// Set the step size (number of features to remove at each iteration).
    ///
    /// # Arguments
    ///
    /// * `step` - Number of features to remove at each iteration.
    ///   If >= 1, removes exactly `step` features.
    ///   Must be at least 1.
    ///
    /// # Panics
    ///
    /// Panics if step is 0.
    pub fn with_step(mut self, step: usize) -> Self {
        assert!(step >= 1, "step must be at least 1");
        self.step = step;
        self
    }

    /// Fit the RFE on data with target.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - Target array of shape (n_samples,)
    ///
    /// # Returns
    ///
    /// * `Ok(())` if fitting succeeds
    /// * `Err` if estimator is not set or other errors occur
    pub fn fit_with_target(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        check_non_empty(x)?;

        if x.nrows() != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("x samples: {}", x.nrows()),
                format!("y length: {}", y.len()),
            ));
        }

        let estimator = self.estimator.as_mut().ok_or_else(|| {
            FerroError::invalid_input("RFE requires an estimator. Call new() with an estimator.")
        })?;

        let n_features = x.ncols();
        let n_features_to_select = self
            .n_features_to_select
            .unwrap_or_else(|| (n_features + 1) / 2) // Default: select half (rounded up)
            .min(n_features);

        if n_features_to_select == 0 {
            return Err(FerroError::invalid_input(
                "n_features_to_select must be at least 1",
            ));
        }

        // Initialize tracking
        // ranking[i] = rank of feature i (1 = selected, higher = eliminated earlier)
        let mut ranking = Array1::ones(n_features);
        // support[i] = true if feature i is still in consideration
        let mut support = vec![true; n_features];
        // Map from current feature index to original feature index
        let mut current_to_original: Vec<usize> = (0..n_features).collect();

        let mut n_current_features = n_features;
        let mut current_rank = n_features; // Start with highest rank for eliminated features
        let mut n_iterations = 0;

        // Iteratively eliminate features
        while n_current_features > n_features_to_select {
            // Extract current feature subset
            let x_subset = Self::extract_features(x, &current_to_original);

            // Fit estimator and get importances
            let importances = estimator.fit_and_get_importances(&x_subset, y)?;

            if importances.len() != n_current_features {
                return Err(FerroError::shape_mismatch(
                    format!("importances length: {}", importances.len()),
                    format!("n_current_features: {}", n_current_features),
                ));
            }

            // Determine how many features to eliminate this iteration
            let n_to_eliminate = self.step.min(n_current_features - n_features_to_select);

            // Find features with lowest importance
            let mut indexed_importances: Vec<(usize, f64)> = importances
                .iter()
                .enumerate()
                .map(|(i, &imp)| (i, if imp.is_nan() { f64::NEG_INFINITY } else { imp }))
                .collect();

            // Sort by importance (ascending) to find least important
            indexed_importances
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Eliminate the least important features
            let to_eliminate: Vec<usize> = indexed_importances
                .iter()
                .take(n_to_eliminate)
                .map(|(idx, _)| *idx)
                .collect();

            // Update rankings for eliminated features
            for &current_idx in &to_eliminate {
                let original_idx = current_to_original[current_idx];
                ranking[original_idx] = current_rank;
                support[original_idx] = false;
            }
            current_rank -= n_to_eliminate;

            // Remove eliminated features from current_to_original
            let mut new_current_to_original = Vec::new();
            for (current_idx, &original_idx) in current_to_original.iter().enumerate() {
                if !to_eliminate.contains(&current_idx) {
                    new_current_to_original.push(original_idx);
                }
            }
            current_to_original = new_current_to_original;
            n_current_features = current_to_original.len();
            n_iterations += 1;
        }

        // Set ranking for selected features to 1
        for &original_idx in &current_to_original {
            ranking[original_idx] = 1;
        }

        // Get selected indices (sorted)
        let mut selected_indices = current_to_original;
        selected_indices.sort();

        self.ranking = Some(ranking);
        self.support = Some(support);
        self.selected_indices = Some(selected_indices);
        self.n_features_in = Some(n_features);
        self.n_iterations = Some(n_iterations);

        Ok(())
    }

    /// Extract features at the specified indices from the input matrix.
    fn extract_features(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        let n_samples = x.nrows();
        let n_features = indices.len();
        let mut result = Array2::zeros((n_samples, n_features));
        for (new_j, &old_j) in indices.iter().enumerate() {
            result.column_mut(new_j).assign(&x.column(old_j));
        }
        result
    }

    /// Get the feature rankings.
    ///
    /// Rankings indicate the elimination order:
    /// - 1 = selected (most important)
    /// - Higher values = eliminated earlier (less important)
    ///
    /// Returns `None` if not fitted.
    pub fn ranking(&self) -> Option<&Array1<usize>> {
        self.ranking.as_ref()
    }

    /// Get a boolean mask of selected features.
    ///
    /// Returns `None` if not fitted.
    pub fn get_support(&self) -> Option<&[bool]> {
        self.support.as_deref()
    }

    /// Get the indices of selected features.
    ///
    /// Returns `None` if not fitted.
    pub fn selected_indices(&self) -> Option<&[usize]> {
        self.selected_indices.as_deref()
    }

    /// Get the number of iterations performed during fitting.
    ///
    /// Returns `None` if not fitted.
    pub fn n_iterations(&self) -> Option<usize> {
        self.n_iterations
    }

    /// Set a new estimator (useful after deserialization).
    pub fn set_estimator(&mut self, estimator: Box<dyn FeatureImportanceEstimator>) {
        self.estimator = Some(estimator);
    }
}

impl Transformer for RecursiveFeatureElimination {
    fn fit(&mut self, _x: &Array2<f64>) -> Result<()> {
        Err(FerroError::invalid_input(
            "RecursiveFeatureElimination requires a target variable. Use fit_with_target() instead.",
        ))
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let selected = self.selected_indices.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected.len();

        let mut result = Array2::zeros((n_samples, n_selected));
        for (new_j, &old_j) in selected.iter().enumerate() {
            result.column_mut(new_j).assign(&x.column(old_j));
        }

        Ok(result)
    }

    fn is_fitted(&self) -> bool {
        self.ranking.is_some() && self.selected_indices.is_some()
    }

    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_features = self.n_features_in?;
        let selected = self.selected_indices.as_ref()?;

        let input_names = input_names
            .map(|n| n.to_vec())
            .unwrap_or_else(|| generate_feature_names(n_features));

        Some(selected.iter().map(|&i| input_names[i].clone()).collect())
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        self.selected_indices.as_ref().map(|s| s.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ========== VarianceThreshold Tests ==========

    #[test]
    fn test_variance_threshold_basic() {
        let mut selector = VarianceThreshold::new(0.0);
        let x = array![[1.0, 5.0, 3.0], [1.0, 2.0, 3.0], [1.0, 8.0, 3.0]];

        let x_selected = selector.fit_transform(&x).unwrap();

        // Only column 1 has variance > 0
        assert_eq!(x_selected.ncols(), 1);
        assert_eq!(selector.selected_indices(), Some(&[1][..]));
    }

    #[test]
    fn test_variance_threshold_with_threshold() {
        let mut selector = VarianceThreshold::new(1.0);
        let x = array![[1.0, 5.0, 3.0], [2.0, 2.0, 3.5], [3.0, 8.0, 4.0]];

        selector.fit(&x).unwrap();

        // Features with variance > 1.0
        let variances = selector.variances().unwrap();
        let selected = selector.selected_indices().unwrap();

        for &idx in selected {
            assert!(variances[idx] > 1.0);
        }
    }

    #[test]
    fn test_variance_threshold_all_constant() {
        let mut selector = VarianceThreshold::new(0.0);
        let x = array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];

        let result = selector.fit(&x);
        assert!(result.is_err()); // All features are constant
    }

    #[test]
    fn test_variance_threshold_transform_shape() {
        let mut selector = VarianceThreshold::new(0.0);
        let x_train = array![
            [1.0, 5.0, 3.0, 7.0],
            [1.0, 2.0, 3.0, 8.0],
            [1.0, 8.0, 3.0, 9.0]
        ];

        selector.fit(&x_train).unwrap();

        // Transform new data
        let x_test = array![[1.0, 10.0, 3.0, 20.0], [1.0, 11.0, 3.0, 21.0]];

        let x_transformed = selector.transform(&x_test).unwrap();
        assert_eq!(x_transformed.nrows(), 2);
        assert_eq!(x_transformed.ncols(), selector.n_features_out().unwrap());
    }

    #[test]
    fn test_variance_threshold_feature_names() {
        let mut selector = VarianceThreshold::new(0.0);
        let x = array![[1.0, 5.0, 3.0], [1.0, 2.0, 3.0], [1.0, 8.0, 3.0]];

        selector.fit(&x).unwrap();

        let input_names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let output_names = selector.get_feature_names_out(Some(&input_names)).unwrap();
        assert_eq!(output_names, vec!["b"]);
    }

    #[test]
    fn test_variance_threshold_get_support() {
        let mut selector = VarianceThreshold::new(0.0);
        let x = array![[1.0, 5.0, 3.0], [1.0, 2.0, 3.0], [1.0, 8.0, 3.0]];

        selector.fit(&x).unwrap();

        let support = selector.get_support().unwrap();
        assert_eq!(support, vec![false, true, false]);
    }

    // ========== SelectKBest Tests ==========

    #[test]
    fn test_select_k_best_f_regression() {
        let mut selector = SelectKBest::new(ScoreFunction::FRegression, 2);
        let x = array![
            [1.0, 0.1, 100.0],
            [2.0, 0.2, 200.0],
            [3.0, 0.3, 300.0],
            [4.0, 0.4, 400.0],
            [5.0, 0.5, 500.0]
        ];
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        selector.fit_with_target(&x, &y).unwrap();
        let x_selected = selector.transform(&x).unwrap();

        assert_eq!(x_selected.ncols(), 2);
        // Columns 0 and 2 are perfectly correlated with y
    }

    #[test]
    fn test_select_k_best_f_classif() {
        let mut selector = SelectKBest::new(ScoreFunction::FClassif, 1);
        let x = array![
            [1.0, 10.0],
            [1.5, 11.0],
            [2.0, 12.0],
            [10.0, 13.0],
            [10.5, 14.0],
            [11.0, 15.0]
        ];
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        selector.fit_with_target(&x, &y).unwrap();
        let x_selected = selector.transform(&x).unwrap();

        assert_eq!(x_selected.ncols(), 1);
        // Column 0 should be selected (clear separation between classes)
        assert_eq!(selector.selected_indices().unwrap(), &[0]);
    }

    #[test]
    fn test_select_k_best_chi2() {
        let mut selector = SelectKBest::new(ScoreFunction::Chi2, 1);
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 10.0],
            [0.0, 20.0],
            [0.0, 30.0]
        ];
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        selector.fit_with_target(&x, &y).unwrap();

        // Both features should have high chi2 scores
        let scores = selector.scores().unwrap();
        assert!(scores.scores[0] > 0.0);
        assert!(scores.scores[1] > 0.0);
    }

    #[test]
    fn test_select_k_best_chi2_negative_values() {
        let mut selector = SelectKBest::new(ScoreFunction::Chi2, 1);
        let x = array![
            [-1.0, 0.0], // Negative value
            [2.0, 0.0],
        ];
        let y = Array1::from_vec(vec![0.0, 1.0]);

        let result = selector.fit_with_target(&x, &y);
        assert!(result.is_err()); // Chi2 requires non-negative features
    }

    #[test]
    fn test_select_k_best_p_values() {
        let mut selector = SelectKBest::new(ScoreFunction::FRegression, 1);
        let x = array![[1.0, 0.5], [2.0, 0.3], [3.0, 0.8], [4.0, 0.2], [5.0, 0.9]];
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        selector.fit_with_target(&x, &y).unwrap();

        let p_values = selector.p_values().unwrap();
        assert!(p_values[0] < 0.05); // Column 0 is perfectly correlated
        assert!(p_values[1] > p_values[0]); // Column 1 is less correlated
    }

    #[test]
    fn test_select_k_best_without_target() {
        let mut selector = SelectKBest::new(ScoreFunction::FRegression, 1);
        let x = array![[1.0, 2.0]];

        let result = selector.fit(&x);
        assert!(result.is_err()); // Should error without target
    }

    // ========== SelectFromModel Tests ==========

    #[test]
    fn test_select_from_model_mean() {
        let importances = Array1::from_vec(vec![0.1, 0.5, 0.2, 0.8, 0.05]);
        let mut selector = SelectFromModel::new(importances, ImportanceThreshold::Mean);

        let x = array![[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]];

        selector.fit(&x).unwrap();
        let x_selected = selector.transform(&x).unwrap();

        // Mean importance = 0.33
        // Features 1 and 3 have importance >= 0.33
        assert!(x_selected.ncols() >= 2);
    }

    #[test]
    fn test_select_from_model_median() {
        let importances = Array1::from_vec(vec![0.1, 0.5, 0.2, 0.8, 0.05]);
        let mut selector = SelectFromModel::new(importances, ImportanceThreshold::Median);

        let x = array![[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]];

        selector.fit(&x).unwrap();
        let x_selected = selector.transform(&x).unwrap();

        // Median importance = 0.2
        // Features 0.5 and 0.8 are >= 0.2, plus 0.2 itself
        assert!(x_selected.ncols() >= 3);
    }

    #[test]
    fn test_select_from_model_value() {
        let importances = Array1::from_vec(vec![0.1, 0.5, 0.2, 0.8, 0.05]);
        let mut selector = SelectFromModel::new(importances, ImportanceThreshold::Value(0.5));

        let x = array![[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]];

        selector.fit(&x).unwrap();
        let x_selected = selector.transform(&x).unwrap();

        // Features with importance >= 0.5: indices 1 (0.5) and 3 (0.8)
        assert_eq!(x_selected.ncols(), 2);
        assert_eq!(selector.selected_indices().unwrap(), &[1, 3]);
    }

    #[test]
    fn test_select_from_model_max_features() {
        let importances = Array1::from_vec(vec![0.1, 0.5, 0.2, 0.8, 0.6]);
        let mut selector =
            SelectFromModel::new(importances, ImportanceThreshold::Value(0.0)).with_max_features(2);

        let x = array![[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]];

        selector.fit(&x).unwrap();
        let x_selected = selector.transform(&x).unwrap();

        // All pass threshold, but only top 2 by importance selected
        assert_eq!(x_selected.ncols(), 2);
        // Top 2 are indices 3 (0.8) and 4 (0.6), sorted: [3, 4]
        assert_eq!(selector.selected_indices().unwrap(), &[3, 4]);
    }

    #[test]
    fn test_select_from_model_threshold_value() {
        let importances = Array1::from_vec(vec![0.1, 0.5, 0.2, 0.8, 0.05]);
        let mut selector = SelectFromModel::new(importances, ImportanceThreshold::Mean);

        let x = array![[1.0, 2.0, 3.0, 4.0, 5.0]];
        selector.fit(&x).unwrap();

        let threshold = selector.threshold_value().unwrap();
        assert!((threshold - 0.33).abs() < 0.01); // Mean of [0.1, 0.5, 0.2, 0.8, 0.05]
    }

    #[test]
    fn test_select_from_model_mismatched_importances() {
        let importances = Array1::from_vec(vec![0.1, 0.5, 0.2]); // 3 importances
        let mut selector = SelectFromModel::new(importances, ImportanceThreshold::Mean);

        let x = array![[1.0, 2.0, 3.0, 4.0, 5.0]]; // 5 features

        let result = selector.fit(&x);
        assert!(result.is_err()); // Mismatch in number of features
    }

    #[test]
    fn test_select_from_model_all_below_threshold() {
        let importances = Array1::from_vec(vec![0.01, 0.02, 0.03]);
        let mut selector = SelectFromModel::new(importances, ImportanceThreshold::Value(0.5));

        let x = array![[1.0, 2.0, 3.0]];

        let result = selector.fit(&x);
        assert!(result.is_err()); // No features pass threshold
    }

    // ========== Score Function Tests ==========

    #[test]
    fn test_f_cdf() {
        // F(1, 10) at x=4.96 should be approximately 0.95
        let p = f_cdf(4.96, 1.0, 10.0);
        assert!((p - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_chi2_cdf() {
        // Chi-squared(2) at x=5.99 should be approximately 0.95
        let p = chi2_cdf(5.99, 2.0);
        assert!((p - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_f_classif_perfect_separation() {
        let x = array![
            [0.0, 100.0],
            [0.0, 101.0],
            [0.0, 102.0],
            [100.0, 100.0],
            [101.0, 101.0],
            [102.0, 102.0]
        ];
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let scores = compute_f_classif(&x, &y).unwrap();

        // Column 0 has perfect class separation, column 1 does not
        assert!(scores.scores[0] > scores.scores[1]);
        assert!(scores.p_values.as_ref().unwrap()[0] < 0.05);
    }

    #[test]
    fn test_f_regression_perfect_correlation() {
        let x = array![[1.0, 0.5], [2.0, 0.3], [3.0, 0.8], [4.0, 0.2], [5.0, 0.9]];
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let scores = compute_f_regression(&x, &y).unwrap();

        // Column 0 is perfectly correlated with y
        assert!(scores.scores[0] > 100.0);
        assert!(scores.p_values.as_ref().unwrap()[0] < 0.001);
    }

    // ========== RecursiveFeatureElimination Tests ==========

    #[test]
    fn test_rfe_basic() {
        // Create an estimator that uses variance as importance
        let estimator =
            ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

        let mut rfe = RecursiveFeatureElimination::new(Box::new(estimator))
            .with_n_features_to_select(2)
            .with_step(1);

        // Features: low variance, medium variance, high variance
        let x = array![
            [1.0, 10.0, 100.0],
            [1.5, 15.0, 200.0],
            [2.0, 20.0, 300.0],
            [2.5, 25.0, 400.0]
        ];
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        rfe.fit_with_target(&x, &y).unwrap();

        // Should select the 2 features with highest variance (indices 1 and 2)
        let selected = rfe.selected_indices().unwrap();
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&1)); // Medium variance
        assert!(selected.contains(&2)); // High variance
    }

    #[test]
    fn test_rfe_ranking() {
        let estimator =
            ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

        let mut rfe = RecursiveFeatureElimination::new(Box::new(estimator))
            .with_n_features_to_select(1)
            .with_step(1);

        let x = array![
            [1.0, 10.0, 100.0], // Variances: low, medium, high
            [1.5, 15.0, 200.0],
            [2.0, 20.0, 300.0],
            [2.5, 25.0, 400.0]
        ];
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        rfe.fit_with_target(&x, &y).unwrap();

        // Check rankings
        let ranking = rfe.ranking().unwrap();
        // Feature 2 (highest variance) should have rank 1 (selected)
        assert_eq!(ranking[2], 1);
        // Feature 1 (medium variance) should have rank 2 (eliminated second to last)
        assert_eq!(ranking[1], 2);
        // Feature 0 (lowest variance) should have rank 3 (eliminated first)
        assert_eq!(ranking[0], 3);
    }

    #[test]
    fn test_rfe_support_mask() {
        let estimator =
            ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

        let mut rfe = RecursiveFeatureElimination::new(Box::new(estimator))
            .with_n_features_to_select(2)
            .with_step(1);

        let x = array![
            [1.0, 10.0, 100.0],
            [1.5, 15.0, 200.0],
            [2.0, 20.0, 300.0],
            [2.5, 25.0, 400.0]
        ];
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        rfe.fit_with_target(&x, &y).unwrap();

        let support = rfe.get_support().unwrap();
        assert_eq!(support.len(), 3);
        assert!(!support[0]); // Low variance - not selected
        assert!(support[1]); // Medium variance - selected
        assert!(support[2]); // High variance - selected
    }

    #[test]
    fn test_rfe_transform() {
        let estimator =
            ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

        let mut rfe = RecursiveFeatureElimination::new(Box::new(estimator))
            .with_n_features_to_select(2)
            .with_step(1);

        let x = array![
            [1.0, 10.0, 100.0],
            [1.5, 15.0, 200.0],
            [2.0, 20.0, 300.0],
            [2.5, 25.0, 400.0]
        ];
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        rfe.fit_with_target(&x, &y).unwrap();

        let x_transformed = rfe.transform(&x).unwrap();
        assert_eq!(x_transformed.ncols(), 2);
        assert_eq!(x_transformed.nrows(), 4);
    }

    #[test]
    fn test_rfe_step_greater_than_one() {
        let estimator =
            ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

        let mut rfe = RecursiveFeatureElimination::new(Box::new(estimator))
            .with_n_features_to_select(1)
            .with_step(2); // Remove 2 features at a time

        let x = array![
            [1.0, 10.0, 100.0, 1000.0, 10000.0], // 5 features
            [1.5, 15.0, 200.0, 2000.0, 20000.0],
            [2.0, 20.0, 300.0, 3000.0, 30000.0],
            [2.5, 25.0, 400.0, 4000.0, 40000.0]
        ];
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        rfe.fit_with_target(&x, &y).unwrap();

        // Should have 2 iterations: 5->3->1
        assert_eq!(rfe.n_iterations(), Some(2));
        assert_eq!(rfe.selected_indices().unwrap().len(), 1);
        // Feature 4 (highest variance) should be selected
        assert_eq!(rfe.selected_indices().unwrap(), &[4]);
    }

    #[test]
    fn test_rfe_default_n_features() {
        // Test that default selects half of features
        let estimator =
            ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

        let mut rfe = RecursiveFeatureElimination::new(Box::new(estimator));
        // Don't set n_features_to_select - should default to half

        let x = array![
            [1.0, 10.0, 100.0, 1000.0], // 4 features
            [1.5, 15.0, 200.0, 2000.0],
            [2.0, 20.0, 300.0, 3000.0]
        ];
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        rfe.fit_with_target(&x, &y).unwrap();

        // Default should select half (4 / 2 = 2)
        assert_eq!(rfe.selected_indices().unwrap().len(), 2);
    }

    #[test]
    fn test_rfe_feature_names() {
        let estimator =
            ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

        let mut rfe = RecursiveFeatureElimination::new(Box::new(estimator))
            .with_n_features_to_select(2)
            .with_step(1);

        let x = array![[1.0, 10.0, 100.0], [1.5, 15.0, 200.0], [2.0, 20.0, 300.0]];
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        rfe.fit_with_target(&x, &y).unwrap();

        let input_names = vec![
            "low_var".to_string(),
            "med_var".to_string(),
            "high_var".to_string(),
        ];
        let output_names = rfe.get_feature_names_out(Some(&input_names)).unwrap();

        assert_eq!(output_names.len(), 2);
        assert!(output_names.contains(&"med_var".to_string()));
        assert!(output_names.contains(&"high_var".to_string()));
    }

    #[test]
    fn test_rfe_not_fitted_error() {
        let estimator =
            ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

        let rfe = RecursiveFeatureElimination::new(Box::new(estimator));
        let x = array![[1.0, 2.0, 3.0]];

        // Should error because not fitted
        assert!(rfe.transform(&x).is_err());
    }

    #[test]
    fn test_rfe_fit_without_target_error() {
        let estimator =
            ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

        let mut rfe = RecursiveFeatureElimination::new(Box::new(estimator));
        let x = array![[1.0, 2.0, 3.0]];

        // Should error because fit() requires target
        assert!(rfe.fit(&x).is_err());
    }

    #[test]
    fn test_rfe_mismatched_x_y() {
        let estimator =
            ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

        let mut rfe =
            RecursiveFeatureElimination::new(Box::new(estimator)).with_n_features_to_select(1);

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Mismatched size

        assert!(rfe.fit_with_target(&x, &y).is_err());
    }

    #[test]
    fn test_rfe_with_correlation_importance() {
        // Use correlation with target as importance (more realistic)
        let estimator = ClosureEstimator::new(|x: &Array2<f64>, y: &Array1<f64>| {
            let n_features = x.ncols();
            let y_mean = y.mean().unwrap_or(0.0);
            let y_centered: Array1<f64> = y - y_mean;
            let y_ss: f64 = y_centered.iter().map(|&v| v * v).sum();

            let mut importances = Array1::zeros(n_features);

            for j in 0..n_features {
                let x_col = x.column(j);
                let x_mean = x_col.mean().unwrap_or(0.0);
                let x_centered: Array1<f64> = x_col.to_owned() - x_mean;
                let x_ss: f64 = x_centered.iter().map(|&v| v * v).sum();

                if x_ss < 1e-15 || y_ss < 1e-15 {
                    importances[j] = 0.0;
                } else {
                    let xy_sum: f64 = x_centered
                        .iter()
                        .zip(y_centered.iter())
                        .map(|(&a, &b)| a * b)
                        .sum();
                    let r = xy_sum / (x_ss.sqrt() * y_ss.sqrt());
                    importances[j] = r.abs(); // Absolute correlation as importance
                }
            }

            Ok(importances)
        });

        let mut rfe = RecursiveFeatureElimination::new(Box::new(estimator))
            .with_n_features_to_select(2)
            .with_step(1);

        // Feature 0: perfectly correlated with y
        // Feature 1: random noise
        // Feature 2: also correlated with y
        let x = array![
            [1.0, 0.5, 2.0],
            [2.0, 0.3, 4.0],
            [3.0, 0.8, 6.0],
            [4.0, 0.2, 8.0],
            [5.0, 0.9, 10.0]
        ];
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        rfe.fit_with_target(&x, &y).unwrap();

        // Features 0 and 2 are correlated with y, feature 1 is noise
        let selected = rfe.selected_indices().unwrap();
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&0));
        assert!(selected.contains(&2));
        assert!(!selected.contains(&1));
    }

    #[test]
    fn test_rfe_n_features_in_out() {
        let estimator =
            ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

        let mut rfe = RecursiveFeatureElimination::new(Box::new(estimator))
            .with_n_features_to_select(2)
            .with_step(1);

        let x = array![[1.0, 10.0, 100.0, 1000.0], [1.5, 15.0, 200.0, 2000.0]];
        let y = Array1::from_vec(vec![1.0, 2.0]);

        rfe.fit_with_target(&x, &y).unwrap();

        assert_eq!(rfe.n_features_in(), Some(4));
        assert_eq!(rfe.n_features_out(), Some(2));
    }

    #[test]
    fn test_rfe_is_fitted() {
        let estimator =
            ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

        let mut rfe =
            RecursiveFeatureElimination::new(Box::new(estimator)).with_n_features_to_select(1);

        assert!(!rfe.is_fitted());

        let x = array![[1.0, 10.0], [1.5, 15.0]];
        let y = Array1::from_vec(vec![1.0, 2.0]);

        rfe.fit_with_target(&x, &y).unwrap();

        assert!(rfe.is_fitted());
    }

    #[test]
    fn test_rfe_select_all_features() {
        let estimator =
            ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

        let mut rfe =
            RecursiveFeatureElimination::new(Box::new(estimator)).with_n_features_to_select(3); // Select all 3 features

        let x = array![[1.0, 10.0, 100.0], [1.5, 15.0, 200.0], [2.0, 20.0, 300.0]];
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        rfe.fit_with_target(&x, &y).unwrap();

        // All features should be selected
        assert_eq!(rfe.selected_indices().unwrap().len(), 3);
        assert_eq!(rfe.n_iterations(), Some(0)); // No iterations needed
    }

    #[test]
    fn test_rfe_select_more_than_available() {
        let estimator =
            ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

        let mut rfe =
            RecursiveFeatureElimination::new(Box::new(estimator)).with_n_features_to_select(10); // More than available

        let x = array![[1.0, 10.0, 100.0], [1.5, 15.0, 200.0]];
        let y = Array1::from_vec(vec![1.0, 2.0]);

        rfe.fit_with_target(&x, &y).unwrap();

        // Should select all 3 features (capped at available)
        assert_eq!(rfe.selected_indices().unwrap().len(), 3);
    }

    #[test]
    fn test_closure_estimator() {
        // Test ClosureEstimator directly
        let mut estimator = ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| {
            Ok(x.mean_axis(Axis(0)).unwrap())
        });

        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let importances = estimator.fit_and_get_importances(&x, &y).unwrap();

        assert_eq!(importances.len(), 2);
        assert!((importances[0] - 2.0).abs() < 1e-10);
        assert!((importances[1] - 20.0).abs() < 1e-10);
    }
}

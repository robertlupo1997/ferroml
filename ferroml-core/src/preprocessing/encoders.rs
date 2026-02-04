//! Categorical Encoding Transformers
//!
//! This module provides transformers for encoding categorical features.
//!
//! ## Available Encoders
//!
//! - [`OneHotEncoder`] - Encode categorical features as one-hot vectors
//! - [`OrdinalEncoder`] - Encode categorical features as integers
//! - [`LabelEncoder`] - Encode target labels as integers
//! - [`TargetEncoder`] - Encode categories using target statistics
//!
//! ## When to Use Which Encoder
//!
//! | Encoder | Best For | Notes |
//! |---------|----------|-------|
//! | `OneHotEncoder` | Nominal categories with few levels | Creates sparse features |
//! | `OrdinalEncoder` | Ordinal categories or tree models | Preserves feature count |
//! | `LabelEncoder` | Target variable encoding | Single column only |
//! | `TargetEncoder` | High-cardinality categories | Requires smoothing |
//!
//! ## Category Representation
//!
//! These encoders work with `f64` arrays where each unique value represents a category.
//! This matches how most ML pipelines represent categorical data after initial parsing.
//!
//! ## Example
//!
//! ```
//! use ferroml_core::preprocessing::encoders::{OneHotEncoder, OrdinalEncoder, LabelEncoder};
//! use ferroml_core::preprocessing::Transformer;
//! use ndarray::{array, Array1};
//!
//! // OneHotEncoder: Creates binary columns for each category
//! let mut ohe = OneHotEncoder::new();
//! let x = array![[0.0], [1.0], [2.0], [1.0]];  // 3 categories
//! let x_encoded = ohe.fit_transform(&x).unwrap();
//! assert_eq!(x_encoded.ncols(), 3);  // One column per category
//!
//! // OrdinalEncoder: Maps categories to integers
//! let mut oe = OrdinalEncoder::new();
//! let x = array![[1.0], [3.0], [2.0], [1.0]];
//! let x_encoded = oe.fit_transform(&x).unwrap();
//! // Categories are mapped to 0, 1, 2 in order of first appearance
//!
//! // LabelEncoder: For 1D label arrays
//! let mut le = LabelEncoder::new();
//! let labels = array![2.0, 0.0, 1.0, 2.0, 1.0];
//! le.fit_1d(&labels).unwrap();
//! let encoded = le.transform_1d(&labels).unwrap();
//! ```

use std::collections::{HashMap, HashSet};

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::{check_is_fitted, check_non_empty, check_shape, Transformer, UnknownCategoryHandling};
use crate::{FerroError, Result};

/// Strategy for dropping categories in OneHotEncoder to avoid collinearity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DropStrategy {
    /// Don't drop any category (may cause collinearity)
    None,
    /// Drop the first category in each feature
    First,
    /// Drop a specific category value for each feature
    IfBinary,
}

impl Default for DropStrategy {
    fn default() -> Self {
        Self::None
    }
}

/// Encode categorical features as one-hot numeric arrays.
///
/// Each unique value in each input feature becomes a separate binary column.
/// For a feature with n unique values, this creates n columns (or n-1 if
/// `drop` is configured).
///
/// # Configuration
///
/// - `handle_unknown`: How to handle unknown categories during transform
///   - `Error`: Raise an error (default)
///   - `Ignore`: Output all zeros for the feature
/// - `drop`: Strategy for dropping categories to avoid collinearity
///   - `None`: Don't drop any category (default)
///   - `First`: Drop the first category
///   - `IfBinary`: Drop first category only for binary features
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::encoders::OneHotEncoder;
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// let mut encoder = OneHotEncoder::new();
/// let x = array![[0.0, 0.0], [1.0, 1.0], [0.0, 2.0]];
///
/// let x_encoded = encoder.fit_transform(&x).unwrap();
///
/// // First feature has 2 categories -> 2 columns
/// // Second feature has 3 categories -> 3 columns
/// // Total: 5 columns
/// assert_eq!(x_encoded.ncols(), 5);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneHotEncoder {
    /// How to handle unknown categories during transform
    handle_unknown: UnknownCategoryHandling,
    /// Strategy for dropping categories
    drop: DropStrategy,
    /// Categories for each feature: feature_idx -> sorted list of unique values
    categories: Option<Vec<Vec<f64>>>,
    /// Mapping from (feature_idx, category_value) -> column index in output
    category_to_column: Option<HashMap<(usize, OrderedF64), usize>>,
    /// Number of input features
    n_features_in: Option<usize>,
    /// Number of output features (total columns after encoding)
    n_features_out: Option<usize>,
    /// Feature names for output columns
    feature_names_out: Option<Vec<String>>,
}

/// Wrapper for f64 that implements Eq and Hash for use in HashMap keys.
/// Uses bitwise comparison which treats NaN values as equal.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct OrderedF64(f64);

impl PartialEq for OrderedF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for OrderedF64 {}

impl std::hash::Hash for OrderedF64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl OneHotEncoder {
    /// Create a new OneHotEncoder with default settings.
    pub fn new() -> Self {
        Self {
            handle_unknown: UnknownCategoryHandling::Error,
            drop: DropStrategy::None,
            categories: None,
            category_to_column: None,
            n_features_in: None,
            n_features_out: None,
            feature_names_out: None,
        }
    }

    /// Set how to handle unknown categories during transform.
    pub fn with_handle_unknown(mut self, handling: UnknownCategoryHandling) -> Self {
        self.handle_unknown = handling;
        self
    }

    /// Set the strategy for dropping categories.
    pub fn with_drop(mut self, drop: DropStrategy) -> Self {
        self.drop = drop;
        self
    }

    /// Get the learned categories for each feature.
    ///
    /// Returns a vector where each element contains the sorted unique values
    /// for that feature.
    pub fn categories(&self) -> Option<&[Vec<f64>]> {
        self.categories.as_deref()
    }

    /// Check if a category should be dropped based on the drop strategy.
    fn should_drop(&self, feature_idx: usize, category_idx: usize) -> bool {
        match &self.drop {
            DropStrategy::None => false,
            DropStrategy::First => category_idx == 0,
            DropStrategy::IfBinary => {
                if let Some(cats) = &self.categories {
                    cats[feature_idx].len() == 2 && category_idx == 0
                } else {
                    false
                }
            }
        }
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for OneHotEncoder {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        let n_features = x.ncols();
        let mut categories: Vec<Vec<f64>> = Vec::with_capacity(n_features);

        // First pass: collect all categories for all features
        for j in 0..n_features {
            let mut unique: HashSet<OrderedF64> = HashSet::new();
            for val in x.column(j) {
                unique.insert(OrderedF64(*val));
            }

            // Sort categories for deterministic ordering
            let mut cats: Vec<f64> = unique.into_iter().map(|of| of.0).collect();
            cats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            categories.push(cats);
        }

        // Store categories first so should_drop can access them
        self.categories = Some(categories);

        // Second pass: build the mapping using drop strategy
        let mut category_to_column: HashMap<(usize, OrderedF64), usize> = HashMap::new();
        let mut feature_names: Vec<String> = Vec::new();
        let mut col_idx = 0;

        let categories = self.categories.as_ref().unwrap();
        for (j, cats) in categories.iter().enumerate() {
            for (cat_idx, &cat_val) in cats.iter().enumerate() {
                if !self.should_drop(j, cat_idx) {
                    category_to_column.insert((j, OrderedF64(cat_val)), col_idx);
                    feature_names.push(format!("x{}_{}", j, cat_val));
                    col_idx += 1;
                }
            }
        }

        self.category_to_column = Some(category_to_column);
        self.n_features_in = Some(n_features);
        self.n_features_out = Some(col_idx);
        self.feature_names_out = Some(feature_names);

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let n_samples = x.nrows();
        let n_features_out = self.n_features_out.unwrap();
        let category_to_column = self.category_to_column.as_ref().unwrap();
        let categories = self.categories.as_ref().unwrap();

        let mut result = Array2::zeros((n_samples, n_features_out));

        for i in 0..n_samples {
            for j in 0..x.ncols() {
                let val = x[[i, j]];
                let key = (j, OrderedF64(val));

                if let Some(&col) = category_to_column.get(&key) {
                    result[[i, col]] = 1.0;
                } else {
                    // Check if it's a known category that was dropped
                    let is_known = categories[j]
                        .iter()
                        .any(|&cat| OrderedF64(cat) == OrderedF64(val));

                    if is_known {
                        // Known but dropped category -> all zeros (do nothing)
                    } else {
                        // Unknown category
                        match self.handle_unknown {
                            UnknownCategoryHandling::Error => {
                                return Err(FerroError::invalid_input(format!(
                                    "Unknown category {} in feature {}",
                                    val, j
                                )));
                            }
                            UnknownCategoryHandling::Ignore
                            | UnknownCategoryHandling::InfrequentIfExist => {
                                // Leave all zeros for this feature
                            }
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "inverse_transform")?;

        let n_samples = x.nrows();
        let n_features_in = self.n_features_in.unwrap();
        let categories = self.categories.as_ref().unwrap();
        let category_to_column = self.category_to_column.as_ref().unwrap();

        let mut result = Array2::zeros((n_samples, n_features_in));

        for i in 0..n_samples {
            for j in 0..n_features_in {
                // Find which category is "hot" for this feature
                let mut found = false;

                for &cat_val in &categories[j] {
                    let key = (j, OrderedF64(cat_val));
                    if let Some(&col) = category_to_column.get(&key) {
                        if x[[i, col]] > 0.5 {
                            result[[i, j]] = cat_val;
                            found = true;
                            break;
                        }
                    }
                }

                // If no category found (dropped category or all zeros), use first category
                if !found {
                    result[[i, j]] = categories[j][0];
                }
            }
        }

        Ok(result)
    }

    fn is_fitted(&self) -> bool {
        self.categories.is_some()
    }

    fn get_feature_names_out(&self, _input_names: Option<&[String]>) -> Option<Vec<String>> {
        self.feature_names_out.clone()
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        self.n_features_out
    }
}

/// Encode categorical features as integers.
///
/// Each unique value in each feature is mapped to an integer (0, 1, 2, ...).
/// Categories are ordered by their first appearance in the training data.
///
/// # Configuration
///
/// - `handle_unknown`: How to handle unknown categories during transform
///   - `Error`: Raise an error (default)
///   - `Ignore`: Map unknown categories to -1.0
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::encoders::OrdinalEncoder;
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// let mut encoder = OrdinalEncoder::new();
/// let x = array![[1.0, 0.5], [2.0, 1.5], [1.0, 0.5]];
///
/// let x_encoded = encoder.fit_transform(&x).unwrap();
///
/// // First feature: 1.0 -> 0, 2.0 -> 1 (order of first appearance)
/// assert_eq!(x_encoded[[0, 0]], 0.0);
/// assert_eq!(x_encoded[[1, 0]], 1.0);
/// assert_eq!(x_encoded[[2, 0]], 0.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrdinalEncoder {
    /// How to handle unknown categories during transform
    handle_unknown: UnknownCategoryHandling,
    /// Categories for each feature: feature_idx -> list of unique values in order
    categories: Option<Vec<Vec<f64>>>,
    /// Mapping from (feature_idx, category_value) -> ordinal code
    category_to_code: Option<HashMap<(usize, OrderedF64), usize>>,
    /// Number of input features
    n_features_in: Option<usize>,
}

impl OrdinalEncoder {
    /// Create a new OrdinalEncoder with default settings.
    pub fn new() -> Self {
        Self {
            handle_unknown: UnknownCategoryHandling::Error,
            categories: None,
            category_to_code: None,
            n_features_in: None,
        }
    }

    /// Set how to handle unknown categories during transform.
    ///
    /// - `Error`: Raise an error for unknown categories (default)
    /// - `Ignore`: Map unknown categories to -1.0
    pub fn with_handle_unknown(mut self, handling: UnknownCategoryHandling) -> Self {
        self.handle_unknown = handling;
        self
    }

    /// Get the learned categories for each feature.
    ///
    /// Categories are in the order they were first seen during fit.
    pub fn categories(&self) -> Option<&[Vec<f64>]> {
        self.categories.as_deref()
    }

    /// Get the number of categories for each feature.
    pub fn n_categories(&self) -> Option<Vec<usize>> {
        self.categories
            .as_ref()
            .map(|cats| cats.iter().map(|c| c.len()).collect())
    }
}

impl Default for OrdinalEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for OrdinalEncoder {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        let n_features = x.ncols();
        let mut categories: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        let mut category_to_code: HashMap<(usize, OrderedF64), usize> = HashMap::new();

        for j in 0..n_features {
            let mut cats: Vec<f64> = Vec::new();
            let mut seen: HashSet<OrderedF64> = HashSet::new();

            // Preserve order of first appearance
            for val in x.column(j) {
                let key = OrderedF64(*val);
                if seen.insert(key) {
                    // insert() returns true if value was newly inserted
                    category_to_code.insert((j, key), cats.len());
                    cats.push(*val);
                }
            }

            categories.push(cats);
        }

        self.categories = Some(categories);
        self.category_to_code = Some(category_to_code);
        self.n_features_in = Some(n_features);

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let category_to_code = self.category_to_code.as_ref().unwrap();
        let mut result = Array2::zeros(x.raw_dim());

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let val = x[[i, j]];
                let key = (j, OrderedF64(val));

                if let Some(&code) = category_to_code.get(&key) {
                    result[[i, j]] = code as f64;
                } else {
                    // Unknown category
                    match self.handle_unknown {
                        UnknownCategoryHandling::Error => {
                            return Err(FerroError::invalid_input(format!(
                                "Unknown category {} in feature {}",
                                val, j
                            )));
                        }
                        UnknownCategoryHandling::Ignore
                        | UnknownCategoryHandling::InfrequentIfExist => {
                            result[[i, j]] = -1.0;
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "inverse_transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let categories = self.categories.as_ref().unwrap();
        let mut result = Array2::zeros(x.raw_dim());

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let code = x[[i, j]].round() as isize;

                if code < 0 {
                    // Unknown category marker, return NaN
                    result[[i, j]] = f64::NAN;
                } else if (code as usize) < categories[j].len() {
                    result[[i, j]] = categories[j][code as usize];
                } else {
                    return Err(FerroError::invalid_input(format!(
                        "Invalid ordinal code {} for feature {} (max: {})",
                        code,
                        j,
                        categories[j].len() - 1
                    )));
                }
            }
        }

        Ok(result)
    }

    fn is_fitted(&self) -> bool {
        self.categories.is_some()
    }

    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_features = self.n_features_in?;
        Some(
            input_names
                .map(|names| names.to_vec())
                .unwrap_or_else(|| (0..n_features).map(|i| format!("x{}", i)).collect()),
        )
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        self.n_features_in
    }
}

/// Encode target labels as integers.
///
/// A simple encoder that converts labels to integers (0 to n_classes-1).
/// This is designed for 1D arrays (target variables) rather than 2D feature matrices.
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::encoders::LabelEncoder;
/// use ndarray::array;
///
/// let mut encoder = LabelEncoder::new();
/// let labels = array![2.0, 0.0, 1.0, 2.0, 1.0];
///
/// encoder.fit_1d(&labels).unwrap();
/// let encoded = encoder.transform_1d(&labels).unwrap();
///
/// // Labels are mapped to 0, 1, 2 in order of first appearance
/// // 2.0 -> 0, 0.0 -> 1, 1.0 -> 2
/// assert_eq!(encoded[0], 0.0);  // 2.0 was first
/// assert_eq!(encoded[1], 1.0);  // 0.0 was second
/// assert_eq!(encoded[2], 2.0);  // 1.0 was third
///
/// // Inverse transform
/// let recovered = encoder.inverse_transform_1d(&encoded).unwrap();
/// assert_eq!(recovered, labels);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelEncoder {
    /// Unique classes in order of first appearance
    classes: Option<Vec<f64>>,
    /// Mapping from class value -> code
    class_to_code: Option<HashMap<OrderedF64, usize>>,
}

impl LabelEncoder {
    /// Create a new LabelEncoder.
    pub fn new() -> Self {
        Self {
            classes: None,
            class_to_code: None,
        }
    }

    /// Get the unique classes in the order they were learned.
    pub fn classes(&self) -> Option<&[f64]> {
        self.classes.as_deref()
    }

    /// Get the number of unique classes.
    pub fn n_classes(&self) -> Option<usize> {
        self.classes.as_ref().map(|c| c.len())
    }

    /// Check if the encoder is fitted.
    pub fn is_fitted(&self) -> bool {
        self.classes.is_some()
    }

    /// Fit the encoder to 1D label data.
    pub fn fit_1d(&mut self, y: &Array1<f64>) -> Result<()> {
        if y.is_empty() {
            return Err(FerroError::invalid_input("Label array cannot be empty"));
        }

        let mut classes: Vec<f64> = Vec::new();
        let mut class_to_code: HashMap<OrderedF64, usize> = HashMap::new();
        let mut seen: HashSet<OrderedF64> = HashSet::new();

        for &val in y.iter() {
            let key = OrderedF64(val);
            if seen.insert(key) {
                // insert() returns true if value was newly inserted
                class_to_code.insert(key, classes.len());
                classes.push(val);
            }
        }

        self.classes = Some(classes);
        self.class_to_code = Some(class_to_code);

        Ok(())
    }

    /// Transform 1D label data to encoded integers.
    pub fn transform_1d(&self, y: &Array1<f64>) -> Result<Array1<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;

        let class_to_code = self.class_to_code.as_ref().unwrap();
        let mut result = Array1::zeros(y.len());

        for (i, &val) in y.iter().enumerate() {
            let key = OrderedF64(val);
            if let Some(&code) = class_to_code.get(&key) {
                result[i] = code as f64;
            } else {
                return Err(FerroError::invalid_input(format!("Unknown class: {}", val)));
            }
        }

        Ok(result)
    }

    /// Fit and transform in one step.
    pub fn fit_transform_1d(&mut self, y: &Array1<f64>) -> Result<Array1<f64>> {
        self.fit_1d(y)?;
        self.transform_1d(y)
    }

    /// Inverse transform encoded labels back to original values.
    pub fn inverse_transform_1d(&self, y: &Array1<f64>) -> Result<Array1<f64>> {
        check_is_fitted(self.is_fitted(), "inverse_transform")?;

        let classes = self.classes.as_ref().unwrap();
        let mut result = Array1::zeros(y.len());

        for (i, &code) in y.iter().enumerate() {
            let idx = code.round() as usize;
            if idx < classes.len() {
                result[i] = classes[idx];
            } else {
                return Err(FerroError::invalid_input(format!(
                    "Invalid label code {} (max: {})",
                    idx,
                    classes.len() - 1
                )));
            }
        }

        Ok(result)
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// LabelEncoder can also implement Transformer for 2D arrays (single column)
impl Transformer for LabelEncoder {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.ncols() != 1 {
            return Err(FerroError::invalid_input(
                "LabelEncoder expects a single column (n_features=1)",
            ));
        }
        self.fit_1d(&x.column(0).to_owned())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != 1 {
            return Err(FerroError::invalid_input(
                "LabelEncoder expects a single column (n_features=1)",
            ));
        }
        let result_1d = self.transform_1d(&x.column(0).to_owned())?;
        Ok(result_1d.insert_axis(ndarray::Axis(1)))
    }

    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != 1 {
            return Err(FerroError::invalid_input(
                "LabelEncoder expects a single column (n_features=1)",
            ));
        }
        let result_1d = self.inverse_transform_1d(&x.column(0).to_owned())?;
        Ok(result_1d.insert_axis(ndarray::Axis(1)))
    }

    fn is_fitted(&self) -> bool {
        self.classes.is_some()
    }

    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>> {
        if !self.is_fitted() {
            return None;
        }
        Some(
            input_names
                .map(|names| names.to_vec())
                .unwrap_or_else(|| vec!["label".to_string()]),
        )
    }

    fn n_features_in(&self) -> Option<usize> {
        if self.is_fitted() {
            Some(1)
        } else {
            None
        }
    }

    fn n_features_out(&self) -> Option<usize> {
        if self.is_fitted() {
            Some(1)
        } else {
            None
        }
    }
}

/// Encode categorical features using target statistics (mean) with smoothing.
///
/// Target encoding replaces each category with a blend of the category's mean target
/// value and the global mean. This is particularly effective for high-cardinality
/// categorical features.
///
/// # Smoothing
///
/// To prevent overfitting for rare categories, we blend the category mean with
/// the global mean using the formula:
///
/// ```text
/// encoded = (count * category_mean + smooth * global_mean) / (count + smooth)
/// ```
///
/// Where:
/// - `count`: Number of samples with this category
/// - `category_mean`: Mean target value for samples with this category
/// - `smooth`: Smoothing parameter (regularization strength)
/// - `global_mean`: Overall mean target value
///
/// Higher smoothing values pull rare category encodings closer to the global mean.
///
/// # Target Leakage Prevention
///
/// When using `fit_transform()`, k-fold cross-validation is used internally:
/// each sample's encoding is computed using only out-of-fold data, preventing
/// target leakage during training.
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::encoders::TargetEncoder;
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::{array, Array1};
///
/// let mut encoder = TargetEncoder::new();
/// let x = array![[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]];
/// let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
///
/// // fit_transform uses CV to prevent leakage
/// let x_encoded = encoder.fit_transform_with_target(&x, &y).unwrap();
///
/// // After fitting, transform uses smoothed statistics
/// let x_new = array![[0.0], [1.0]];
/// let x_new_encoded = encoder.transform(&x_new).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetEncoder {
    /// Smoothing parameter (regularization strength). Default: 1.0
    /// Higher values pull rare category encodings closer to the global mean.
    smooth: f64,
    /// How to handle unknown categories during transform. Default: use global mean
    handle_unknown: UnknownCategoryHandling,
    /// Number of CV folds for internal cross-validation during fit_transform. Default: 5
    cv: usize,
    /// Global mean of target values
    global_mean: Option<f64>,
    /// Category statistics: (feature_idx, category_value) -> (sum, count)
    category_stats: Option<HashMap<(usize, OrderedF64), (f64, usize)>>,
    /// Pre-computed encoded values: (feature_idx, category_value) -> encoded_value
    encoding_map: Option<HashMap<(usize, OrderedF64), f64>>,
    /// Number of input features
    n_features_in: Option<usize>,
}

impl TargetEncoder {
    /// Create a new TargetEncoder with default settings.
    ///
    /// Default configuration:
    /// - `smooth`: 1.0
    /// - `handle_unknown`: `Error` (raises error for unknown categories)
    /// - `cv`: 5 (folds for internal cross-validation)
    pub fn new() -> Self {
        Self {
            smooth: 1.0,
            handle_unknown: UnknownCategoryHandling::Error,
            cv: 5,
            global_mean: None,
            category_stats: None,
            encoding_map: None,
            n_features_in: None,
        }
    }

    /// Set the smoothing parameter.
    ///
    /// Higher values (e.g., 10.0) provide more regularization, pulling encodings
    /// toward the global mean. Lower values (e.g., 0.1) trust category statistics more.
    ///
    /// # Arguments
    ///
    /// * `smooth` - Smoothing parameter, must be non-negative
    pub fn with_smooth(mut self, smooth: f64) -> Self {
        assert!(smooth >= 0.0, "Smoothing parameter must be non-negative");
        self.smooth = smooth;
        self
    }

    /// Set how to handle unknown categories during transform.
    ///
    /// - `Error`: Raise an error for unknown categories
    /// - `Ignore` or `InfrequentIfExist`: Use global mean for unknown categories
    pub fn with_handle_unknown(mut self, handling: UnknownCategoryHandling) -> Self {
        self.handle_unknown = handling;
        self
    }

    /// Set the number of CV folds for internal cross-validation.
    ///
    /// Used during `fit_transform_with_target()` to prevent target leakage.
    ///
    /// # Arguments
    ///
    /// * `cv` - Number of folds, must be at least 2
    pub fn with_cv(mut self, cv: usize) -> Self {
        assert!(cv >= 2, "Number of CV folds must be at least 2");
        self.cv = cv;
        self
    }

    /// Get the global mean of target values.
    pub fn global_mean(&self) -> Option<f64> {
        self.global_mean
    }

    /// Get the smoothing parameter.
    pub fn smooth(&self) -> f64 {
        self.smooth
    }

    /// Fit the encoder to data with target values.
    ///
    /// This method learns the category statistics needed for encoding.
    /// Use this when you have separate training and test data.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature data of shape `(n_samples, n_features)`
    /// * `y` - Target values of shape `(n_samples,)`
    pub fn fit_with_target(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        check_non_empty(x)?;

        if x.nrows() != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("({},)", x.nrows()),
                format!("({},)", y.len()),
            ));
        }

        let n_features = x.ncols();

        // Compute global mean
        let global_mean = y.mean().unwrap_or(0.0);

        // Compute category statistics
        let mut category_stats: HashMap<(usize, OrderedF64), (f64, usize)> = HashMap::new();

        for i in 0..x.nrows() {
            let target = y[i];
            for j in 0..n_features {
                let key = (j, OrderedF64(x[[i, j]]));
                let entry = category_stats.entry(key).or_insert((0.0, 0));
                entry.0 += target; // sum
                entry.1 += 1; // count
            }
        }

        // Pre-compute encoded values using smoothing formula
        let mut encoding_map: HashMap<(usize, OrderedF64), f64> = HashMap::new();

        for (&key, &(sum, count)) in &category_stats {
            let category_mean = sum / count as f64;
            // Smoothed encoding: (count * category_mean + smooth * global_mean) / (count + smooth)
            let encoded = (count as f64 * category_mean + self.smooth * global_mean)
                / (count as f64 + self.smooth);
            encoding_map.insert(key, encoded);
        }

        self.global_mean = Some(global_mean);
        self.category_stats = Some(category_stats);
        self.encoding_map = Some(encoding_map);
        self.n_features_in = Some(n_features);

        Ok(())
    }

    /// Fit and transform with target leakage prevention via k-fold CV.
    ///
    /// This is the recommended method for training data. Each sample's encoding
    /// is computed using only out-of-fold data to prevent target leakage.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature data of shape `(n_samples, n_features)`
    /// * `y` - Target values of shape `(n_samples,)`
    ///
    /// # Returns
    ///
    /// * Encoded feature matrix
    pub fn fit_transform_with_target(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        check_non_empty(x)?;

        if x.nrows() != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("({},)", x.nrows()),
                format!("({},)", y.len()),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // If we have fewer samples than folds, reduce folds
        let actual_cv = self.cv.min(n_samples);

        // First, fit on all data (for later transform calls)
        self.fit_with_target(x, y)?;

        // Global mean for fallback
        let global_mean = self.global_mean.unwrap();

        // Create result array
        let mut result = Array2::zeros((n_samples, n_features));

        if actual_cv < 2 {
            // Not enough samples for CV, just use fitted values
            return self.transform(x);
        }

        // Create fold assignments
        let fold_sizes: Vec<usize> = (0..actual_cv)
            .map(|i| {
                let base = n_samples / actual_cv;
                let remainder = n_samples % actual_cv;
                if i < remainder {
                    base + 1
                } else {
                    base
                }
            })
            .collect();

        // Assign each sample to a fold
        let mut fold_assignment = vec![0usize; n_samples];
        let mut idx = 0;
        for (fold_idx, &size) in fold_sizes.iter().enumerate() {
            for _ in 0..size {
                fold_assignment[idx] = fold_idx;
                idx += 1;
            }
        }

        // For each fold, compute statistics from out-of-fold data and encode fold samples
        for fold in 0..actual_cv {
            // Compute out-of-fold statistics
            let mut oof_global_sum = 0.0;
            let mut oof_global_count = 0usize;
            let mut oof_category_stats: HashMap<(usize, OrderedF64), (f64, usize)> = HashMap::new();

            for i in 0..n_samples {
                if fold_assignment[i] != fold {
                    let target = y[i];
                    oof_global_sum += target;
                    oof_global_count += 1;

                    for j in 0..n_features {
                        let key = (j, OrderedF64(x[[i, j]]));
                        let entry = oof_category_stats.entry(key).or_insert((0.0, 0));
                        entry.0 += target;
                        entry.1 += 1;
                    }
                }
            }

            let oof_global_mean = if oof_global_count > 0 {
                oof_global_sum / oof_global_count as f64
            } else {
                global_mean
            };

            // Encode samples in this fold using out-of-fold statistics
            for i in 0..n_samples {
                if fold_assignment[i] == fold {
                    for j in 0..n_features {
                        let key = (j, OrderedF64(x[[i, j]]));
                        if let Some(&(sum, count)) = oof_category_stats.get(&key) {
                            let category_mean = sum / count as f64;
                            let encoded = (count as f64 * category_mean
                                + self.smooth * oof_global_mean)
                                / (count as f64 + self.smooth);
                            result[[i, j]] = encoded;
                        } else {
                            // Category only appears in this fold - use global mean
                            result[[i, j]] = oof_global_mean;
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Encode a single value using the smoothing formula.
    fn encode_value(&self, feature_idx: usize, value: f64) -> Result<f64> {
        let global_mean = self.global_mean.unwrap();
        let encoding_map = self.encoding_map.as_ref().unwrap();

        let key = (feature_idx, OrderedF64(value));

        if let Some(&encoded) = encoding_map.get(&key) {
            Ok(encoded)
        } else {
            // Unknown category
            match self.handle_unknown {
                UnknownCategoryHandling::Error => Err(FerroError::invalid_input(format!(
                    "Unknown category {} in feature {}",
                    value, feature_idx
                ))),
                UnknownCategoryHandling::Ignore | UnknownCategoryHandling::InfrequentIfExist => {
                    // Use global mean for unknown categories
                    Ok(global_mean)
                }
            }
        }
    }
}

impl Default for TargetEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for TargetEncoder {
    fn fit(&mut self, _x: &Array2<f64>) -> Result<()> {
        // TargetEncoder needs target values to fit
        // This implementation allows fitting without target for API compatibility
        // but the encoder won't be usable until fit_with_target is called
        Err(FerroError::invalid_input(
            "TargetEncoder requires target values. Use fit_with_target(x, y) instead.",
        ))
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut result = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                result[[i, j]] = self.encode_value(j, x[[i, j]])?;
            }
        }

        Ok(result)
    }

    fn inverse_transform(&self, _x: &Array2<f64>) -> Result<Array2<f64>> {
        Err(FerroError::NotImplemented(
            "TargetEncoder does not support inverse_transform (many-to-one mapping)".to_string(),
        ))
    }

    fn is_fitted(&self) -> bool {
        self.encoding_map.is_some()
    }

    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_features = self.n_features_in?;
        Some(
            input_names
                .map(|names| names.iter().map(|n| format!("{}_target_enc", n)).collect())
                .unwrap_or_else(|| {
                    (0..n_features)
                        .map(|i| format!("x{}_target_enc", i))
                        .collect()
                }),
        )
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        self.n_features_in
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f64 = 1e-10;

    // ========== OneHotEncoder Tests ==========

    #[test]
    fn test_onehot_encoder_basic() {
        let mut encoder = OneHotEncoder::new();
        let x = array![[0.0], [1.0], [2.0], [1.0]];

        let x_encoded = encoder.fit_transform(&x).unwrap();

        // 3 unique values -> 3 columns
        assert_eq!(x_encoded.ncols(), 3);
        assert_eq!(x_encoded.nrows(), 4);

        // Check one-hot encoding
        assert!((x_encoded[[0, 0]] - 1.0).abs() < EPSILON); // 0.0 -> [1,0,0]
        assert!((x_encoded[[0, 1]] - 0.0).abs() < EPSILON);
        assert!((x_encoded[[0, 2]] - 0.0).abs() < EPSILON);

        assert!((x_encoded[[1, 0]] - 0.0).abs() < EPSILON); // 1.0 -> [0,1,0]
        assert!((x_encoded[[1, 1]] - 1.0).abs() < EPSILON);
        assert!((x_encoded[[1, 2]] - 0.0).abs() < EPSILON);

        assert!((x_encoded[[2, 0]] - 0.0).abs() < EPSILON); // 2.0 -> [0,0,1]
        assert!((x_encoded[[2, 1]] - 0.0).abs() < EPSILON);
        assert!((x_encoded[[2, 2]] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_onehot_encoder_multiple_features() {
        let mut encoder = OneHotEncoder::new();
        let x = array![[0.0, 0.0], [1.0, 1.0], [0.0, 2.0]];

        let x_encoded = encoder.fit_transform(&x).unwrap();

        // First feature: 2 categories (0, 1)
        // Second feature: 3 categories (0, 1, 2)
        // Total: 5 columns
        assert_eq!(x_encoded.ncols(), 5);
    }

    #[test]
    fn test_onehot_encoder_inverse_transform() {
        let mut encoder = OneHotEncoder::new();
        let x = array![[0.0], [1.0], [2.0], [1.0]];

        let x_encoded = encoder.fit_transform(&x).unwrap();
        let x_recovered = encoder.inverse_transform(&x_encoded).unwrap();

        for i in 0..x.nrows() {
            assert!((x[[i, 0]] - x_recovered[[i, 0]]).abs() < EPSILON);
        }
    }

    #[test]
    fn test_onehot_encoder_unknown_error() {
        let mut encoder = OneHotEncoder::new();
        let x_train = array![[0.0], [1.0]];
        let x_test = array![[2.0]]; // Unknown category

        encoder.fit(&x_train).unwrap();
        assert!(encoder.transform(&x_test).is_err());
    }

    #[test]
    fn test_onehot_encoder_unknown_ignore() {
        let mut encoder = OneHotEncoder::new().with_handle_unknown(UnknownCategoryHandling::Ignore);
        let x_train = array![[0.0], [1.0]];
        let x_test = array![[2.0]]; // Unknown category

        encoder.fit(&x_train).unwrap();
        let x_encoded = encoder.transform(&x_test).unwrap();

        // Unknown category should result in all zeros
        assert!((x_encoded[[0, 0]] - 0.0).abs() < EPSILON);
        assert!((x_encoded[[0, 1]] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_onehot_encoder_drop_first() {
        let mut encoder = OneHotEncoder::new().with_drop(DropStrategy::First);
        let x = array![[0.0], [1.0], [2.0]];

        let x_encoded = encoder.fit_transform(&x).unwrap();

        // 3 categories but drop first -> 2 columns
        assert_eq!(x_encoded.ncols(), 2);

        // 0.0 category is dropped -> all zeros
        assert!((x_encoded[[0, 0]] - 0.0).abs() < EPSILON);
        assert!((x_encoded[[0, 1]] - 0.0).abs() < EPSILON);

        // 1.0 -> [1, 0]
        assert!((x_encoded[[1, 0]] - 1.0).abs() < EPSILON);
        assert!((x_encoded[[1, 1]] - 0.0).abs() < EPSILON);

        // 2.0 -> [0, 1]
        assert!((x_encoded[[2, 0]] - 0.0).abs() < EPSILON);
        assert!((x_encoded[[2, 1]] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_onehot_encoder_drop_if_binary() {
        let mut encoder = OneHotEncoder::new().with_drop(DropStrategy::IfBinary);

        // Binary feature
        let x = array![[0.0], [1.0], [0.0]];
        let x_encoded = encoder.fit_transform(&x).unwrap();
        assert_eq!(x_encoded.ncols(), 1); // Dropped one

        // Non-binary feature
        let mut encoder2 = OneHotEncoder::new().with_drop(DropStrategy::IfBinary);
        let x2 = array![[0.0], [1.0], [2.0]];
        let x_encoded2 = encoder2.fit_transform(&x2).unwrap();
        assert_eq!(x_encoded2.ncols(), 3); // No dropping
    }

    #[test]
    fn test_onehot_encoder_feature_names() {
        let mut encoder = OneHotEncoder::new();
        let x = array![[0.0, 5.0], [1.0, 10.0]];

        encoder.fit(&x).unwrap();
        let names = encoder.get_feature_names_out(None).unwrap();

        assert_eq!(names.len(), 4); // 2 + 2 categories
        assert!(names[0].starts_with("x0_")); // Feature 0
        assert!(names[2].starts_with("x1_")); // Feature 1
    }

    // ========== OrdinalEncoder Tests ==========

    #[test]
    fn test_ordinal_encoder_basic() {
        let mut encoder = OrdinalEncoder::new();
        let x = array![[1.0], [3.0], [2.0], [1.0]];

        let x_encoded = encoder.fit_transform(&x).unwrap();

        // Categories in order of first appearance: 1.0 -> 0, 3.0 -> 1, 2.0 -> 2
        assert!((x_encoded[[0, 0]] - 0.0).abs() < EPSILON);
        assert!((x_encoded[[1, 0]] - 1.0).abs() < EPSILON);
        assert!((x_encoded[[2, 0]] - 2.0).abs() < EPSILON);
        assert!((x_encoded[[3, 0]] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_ordinal_encoder_multiple_features() {
        let mut encoder = OrdinalEncoder::new();
        let x = array![[1.0, 0.5], [2.0, 1.5], [1.0, 0.5]];

        let x_encoded = encoder.fit_transform(&x).unwrap();

        // Same number of features
        assert_eq!(x_encoded.ncols(), 2);

        // Feature 0: 1.0 -> 0, 2.0 -> 1
        assert!((x_encoded[[0, 0]] - 0.0).abs() < EPSILON);
        assert!((x_encoded[[1, 0]] - 1.0).abs() < EPSILON);

        // Feature 1: 0.5 -> 0, 1.5 -> 1
        assert!((x_encoded[[0, 1]] - 0.0).abs() < EPSILON);
        assert!((x_encoded[[1, 1]] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_ordinal_encoder_inverse_transform() {
        let mut encoder = OrdinalEncoder::new();
        let x = array![[1.0, 0.5], [2.0, 1.5], [1.0, 0.5]];

        let x_encoded = encoder.fit_transform(&x).unwrap();
        let x_recovered = encoder.inverse_transform(&x_encoded).unwrap();

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((x[[i, j]] - x_recovered[[i, j]]).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_ordinal_encoder_unknown_error() {
        let mut encoder = OrdinalEncoder::new();
        let x_train = array![[0.0], [1.0]];
        let x_test = array![[2.0]]; // Unknown

        encoder.fit(&x_train).unwrap();
        assert!(encoder.transform(&x_test).is_err());
    }

    #[test]
    fn test_ordinal_encoder_unknown_ignore() {
        let mut encoder =
            OrdinalEncoder::new().with_handle_unknown(UnknownCategoryHandling::Ignore);
        let x_train = array![[0.0], [1.0]];
        let x_test = array![[2.0]]; // Unknown

        encoder.fit(&x_train).unwrap();
        let x_encoded = encoder.transform(&x_test).unwrap();

        // Unknown -> -1.0
        assert!((x_encoded[[0, 0]] - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_ordinal_encoder_categories() {
        let mut encoder = OrdinalEncoder::new();
        let x = array![[5.0], [3.0], [1.0], [3.0]];

        encoder.fit(&x).unwrap();

        let cats = encoder.categories().unwrap();
        assert_eq!(cats.len(), 1);
        // Order of first appearance: 5.0, 3.0, 1.0
        assert!((cats[0][0] - 5.0).abs() < EPSILON);
        assert!((cats[0][1] - 3.0).abs() < EPSILON);
        assert!((cats[0][2] - 1.0).abs() < EPSILON);
    }

    // ========== LabelEncoder Tests ==========

    #[test]
    fn test_label_encoder_basic() {
        let mut encoder = LabelEncoder::new();
        let y = array![2.0, 0.0, 1.0, 2.0, 1.0];

        let y_encoded = encoder.fit_transform_1d(&y).unwrap();

        // Order of first appearance: 2.0 -> 0, 0.0 -> 1, 1.0 -> 2
        assert!((y_encoded[0] - 0.0).abs() < EPSILON);
        assert!((y_encoded[1] - 1.0).abs() < EPSILON);
        assert!((y_encoded[2] - 2.0).abs() < EPSILON);
        assert!((y_encoded[3] - 0.0).abs() < EPSILON);
        assert!((y_encoded[4] - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_label_encoder_inverse_transform() {
        let mut encoder = LabelEncoder::new();
        let y = array![2.0, 0.0, 1.0, 2.0, 1.0];

        let y_encoded = encoder.fit_transform_1d(&y).unwrap();
        let y_recovered = encoder.inverse_transform_1d(&y_encoded).unwrap();

        for i in 0..y.len() {
            assert!((y[i] - y_recovered[i]).abs() < EPSILON);
        }
    }

    #[test]
    fn test_label_encoder_unknown() {
        let mut encoder = LabelEncoder::new();
        let y_train = array![0.0, 1.0];
        let y_test = array![2.0]; // Unknown

        encoder.fit_1d(&y_train).unwrap();
        assert!(encoder.transform_1d(&y_test).is_err());
    }

    #[test]
    fn test_label_encoder_classes() {
        let mut encoder = LabelEncoder::new();
        let y = array![5.0, 3.0, 1.0, 3.0];

        encoder.fit_1d(&y).unwrap();

        let classes = encoder.classes().unwrap();
        assert_eq!(classes.len(), 3);
        // Order of first appearance
        assert!((classes[0] - 5.0).abs() < EPSILON);
        assert!((classes[1] - 3.0).abs() < EPSILON);
        assert!((classes[2] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_label_encoder_2d_interface() {
        let mut encoder = LabelEncoder::new();
        let y = array![[2.0], [0.0], [1.0]];

        let y_encoded = encoder.fit_transform(&y).unwrap();

        assert_eq!(y_encoded.ncols(), 1);
        assert!((y_encoded[[0, 0]] - 0.0).abs() < EPSILON);
        assert!((y_encoded[[1, 0]] - 1.0).abs() < EPSILON);
        assert!((y_encoded[[2, 0]] - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_label_encoder_2d_invalid_shape() {
        let mut encoder = LabelEncoder::new();
        let y = array![[1.0, 2.0], [3.0, 4.0]]; // 2 columns

        assert!(encoder.fit(&y).is_err());
    }

    // ========== Common Tests ==========

    #[test]
    fn test_not_fitted() {
        let ohe = OneHotEncoder::new();
        let oe = OrdinalEncoder::new();
        let le = LabelEncoder::new();

        let x = array![[1.0]];
        let y = array![1.0];

        assert!(!ohe.is_fitted());
        assert!(ohe.transform(&x).is_err());

        assert!(!oe.is_fitted());
        assert!(oe.transform(&x).is_err());

        assert!(!le.is_fitted());
        assert!(le.transform_1d(&y).is_err());
    }

    #[test]
    fn test_empty_input() {
        let mut ohe = OneHotEncoder::new();
        let mut oe = OrdinalEncoder::new();
        let mut le = LabelEncoder::new();

        let empty_2d: Array2<f64> = Array2::zeros((0, 0));
        let empty_1d: Array1<f64> = Array1::zeros(0);

        assert!(ohe.fit(&empty_2d).is_err());
        assert!(oe.fit(&empty_2d).is_err());
        assert!(le.fit_1d(&empty_1d).is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let mut ohe = OneHotEncoder::new();
        let mut oe = OrdinalEncoder::new();

        let x_train = array![[1.0, 2.0]];
        let x_test = array![[1.0, 2.0, 3.0]]; // Wrong number of features

        ohe.fit(&x_train).unwrap();
        assert!(ohe.transform(&x_test).is_err());

        oe.fit(&x_train).unwrap();
        assert!(oe.transform(&x_test).is_err());
    }

    #[test]
    fn test_nan_handling() {
        // NaN values should be treated as a distinct category
        let mut ohe = OneHotEncoder::new();
        let x = array![[1.0], [f64::NAN], [1.0]];

        let x_encoded = ohe.fit_transform(&x).unwrap();

        // 2 unique categories: 1.0 and NaN
        assert_eq!(x_encoded.ncols(), 2);

        // Both samples with 1.0 should have same encoding
        assert_eq!(x_encoded[[0, 0]], x_encoded[[2, 0]]);
        assert_eq!(x_encoded[[0, 1]], x_encoded[[2, 1]]);

        // NaN sample should have different encoding
        assert_ne!(x_encoded[[1, 0]], x_encoded[[0, 0]]);
    }

    // ========== TargetEncoder Tests ==========

    #[test]
    fn test_target_encoder_basic() {
        let mut encoder = TargetEncoder::new();
        // 3 categories: 0.0, 1.0, 2.0
        let x = array![[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]];
        // Targets: category 0 has mean 1.5, category 1 has mean 3.5, category 2 has mean 5.5
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        encoder.fit_with_target(&x, &y).unwrap();

        // Global mean = (1+2+3+4+5+6)/6 = 21/6 = 3.5
        assert!((encoder.global_mean().unwrap() - 3.5).abs() < EPSILON);

        // Transform should use smoothed values
        let x_encoded = encoder.transform(&x).unwrap();
        assert_eq!(x_encoded.ncols(), 1);
        assert_eq!(x_encoded.nrows(), 6);

        // With smooth=1.0:
        // Category 0: (2 * 1.5 + 1 * 3.5) / (2 + 1) = 6.5 / 3 = 2.166...
        // Category 1: (2 * 3.5 + 1 * 3.5) / (2 + 1) = 10.5 / 3 = 3.5
        // Category 2: (2 * 5.5 + 1 * 3.5) / (2 + 1) = 14.5 / 3 = 4.833...
        let expected_cat0 = (2.0 * 1.5 + 1.0 * 3.5) / 3.0;
        let expected_cat1 = (2.0 * 3.5 + 1.0 * 3.5) / 3.0;
        let expected_cat2 = (2.0 * 5.5 + 1.0 * 3.5) / 3.0;

        assert!((x_encoded[[0, 0]] - expected_cat0).abs() < EPSILON);
        assert!((x_encoded[[1, 0]] - expected_cat0).abs() < EPSILON);
        assert!((x_encoded[[2, 0]] - expected_cat1).abs() < EPSILON);
        assert!((x_encoded[[3, 0]] - expected_cat1).abs() < EPSILON);
        assert!((x_encoded[[4, 0]] - expected_cat2).abs() < EPSILON);
        assert!((x_encoded[[5, 0]] - expected_cat2).abs() < EPSILON);
    }

    #[test]
    fn test_target_encoder_smoothing_effect() {
        // With high smoothing, rare categories should be pulled toward global mean
        let x = array![[0.0], [0.0], [0.0], [0.0], [1.0]]; // Category 1 is rare
        let y = array![1.0, 1.0, 1.0, 1.0, 100.0]; // Category 1 has extreme value

        // Low smoothing: trust category mean more
        let mut encoder_low = TargetEncoder::new().with_smooth(0.1);
        encoder_low.fit_with_target(&x, &y).unwrap();
        let encoded_low = encoder_low.transform(&array![[1.0]]).unwrap();

        // High smoothing: pull toward global mean
        let mut encoder_high = TargetEncoder::new().with_smooth(10.0);
        encoder_high.fit_with_target(&x, &y).unwrap();
        let encoded_high = encoder_high.transform(&array![[1.0]]).unwrap();

        let global_mean = (4.0 + 100.0) / 5.0; // = 20.8

        // With low smoothing, category 1 encoding should be closer to 100
        // With high smoothing, category 1 encoding should be closer to global mean
        assert!(encoded_low[[0, 0]] > encoded_high[[0, 0]]);
        assert!(
            (encoded_high[[0, 0]] - global_mean).abs() < (encoded_low[[0, 0]] - global_mean).abs()
        );
    }

    #[test]
    fn test_target_encoder_multiple_features() {
        let mut encoder = TargetEncoder::new();
        let x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        encoder.fit_with_target(&x, &y).unwrap();
        let x_encoded = encoder.transform(&x).unwrap();

        assert_eq!(x_encoded.ncols(), 2);
        assert_eq!(x_encoded.nrows(), 4);

        // Feature 0: category 0 has mean 1.5, category 1 has mean 3.5
        // Feature 1: category 0 has mean 2.0, category 1 has mean 3.0
        // Global mean = 2.5
        let global_mean = 2.5;

        // With smooth=1.0, feature 0, category 0: (2 * 1.5 + 1 * 2.5) / 3 = 5.5/3
        let f0_cat0 = (2.0 * 1.5 + 1.0 * global_mean) / 3.0;
        let f0_cat1 = (2.0 * 3.5 + 1.0 * global_mean) / 3.0;
        let f1_cat0 = (2.0 * 2.0 + 1.0 * global_mean) / 3.0;
        let f1_cat1 = (2.0 * 3.0 + 1.0 * global_mean) / 3.0;

        assert!((x_encoded[[0, 0]] - f0_cat0).abs() < EPSILON);
        assert!((x_encoded[[1, 0]] - f0_cat0).abs() < EPSILON);
        assert!((x_encoded[[2, 0]] - f0_cat1).abs() < EPSILON);
        assert!((x_encoded[[3, 0]] - f0_cat1).abs() < EPSILON);

        assert!((x_encoded[[0, 1]] - f1_cat0).abs() < EPSILON);
        assert!((x_encoded[[1, 1]] - f1_cat1).abs() < EPSILON);
        assert!((x_encoded[[2, 1]] - f1_cat0).abs() < EPSILON);
        assert!((x_encoded[[3, 1]] - f1_cat1).abs() < EPSILON);
    }

    #[test]
    fn test_target_encoder_fit_transform_with_cv() {
        let mut encoder = TargetEncoder::new().with_cv(2);
        let x = array![[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // fit_transform_with_target uses CV to prevent leakage
        let x_encoded = encoder.fit_transform_with_target(&x, &y).unwrap();

        assert_eq!(x_encoded.ncols(), 1);
        assert_eq!(x_encoded.nrows(), 6);

        // After fit_transform, the encoder should be fitted for future transform calls
        assert!(encoder.is_fitted());

        // transform on new data should work
        let x_new = array![[0.0], [1.0], [2.0]];
        let x_new_encoded = encoder.transform(&x_new).unwrap();
        assert_eq!(x_new_encoded.nrows(), 3);
    }

    #[test]
    fn test_target_encoder_unknown_error() {
        let mut encoder = TargetEncoder::new();
        let x = array![[0.0], [1.0]];
        let y = array![1.0, 2.0];

        encoder.fit_with_target(&x, &y).unwrap();

        let x_new = array![[2.0]]; // Unknown category
        assert!(encoder.transform(&x_new).is_err());
    }

    #[test]
    fn test_target_encoder_unknown_ignore() {
        let mut encoder = TargetEncoder::new().with_handle_unknown(UnknownCategoryHandling::Ignore);
        let x = array![[0.0], [1.0]];
        let y = array![1.0, 2.0];

        encoder.fit_with_target(&x, &y).unwrap();

        let x_new = array![[2.0]]; // Unknown category
        let x_encoded = encoder.transform(&x_new).unwrap();

        // Unknown category should use global mean
        let global_mean = 1.5;
        assert!((x_encoded[[0, 0]] - global_mean).abs() < EPSILON);
    }

    #[test]
    fn test_target_encoder_fit_without_target_fails() {
        let mut encoder = TargetEncoder::new();
        let x = array![[0.0], [1.0]];

        // Regular fit should fail - needs target
        assert!(encoder.fit(&x).is_err());
    }

    #[test]
    fn test_target_encoder_not_fitted() {
        let encoder = TargetEncoder::new();
        let x = array![[0.0]];

        assert!(!encoder.is_fitted());
        assert!(encoder.transform(&x).is_err());
    }

    #[test]
    fn test_target_encoder_shape_mismatch() {
        let mut encoder = TargetEncoder::new();
        let x = array![[0.0, 1.0]];
        let y = array![1.0];

        encoder.fit_with_target(&x, &y).unwrap();

        let x_wrong = array![[0.0]]; // Wrong number of features
        assert!(encoder.transform(&x_wrong).is_err());
    }

    #[test]
    fn test_target_encoder_x_y_length_mismatch() {
        let mut encoder = TargetEncoder::new();
        let x = array![[0.0], [1.0]];
        let y = array![1.0, 2.0, 3.0]; // Wrong length

        assert!(encoder.fit_with_target(&x, &y).is_err());
    }

    #[test]
    fn test_target_encoder_inverse_transform_not_supported() {
        let mut encoder = TargetEncoder::new();
        let x = array![[0.0], [1.0]];
        let y = array![1.0, 2.0];

        encoder.fit_with_target(&x, &y).unwrap();
        let x_encoded = encoder.transform(&x).unwrap();

        // inverse_transform should fail (many-to-one mapping)
        assert!(encoder.inverse_transform(&x_encoded).is_err());
    }

    #[test]
    fn test_target_encoder_feature_names() {
        let mut encoder = TargetEncoder::new();
        let x = array![[0.0], [1.0]];
        let y = array![1.0, 2.0];

        encoder.fit_with_target(&x, &y).unwrap();

        // Default names
        let names = encoder.get_feature_names_out(None).unwrap();
        assert_eq!(names, vec!["x0_target_enc"]);

        // Custom names
        let input_names = vec!["category".to_string()];
        let names = encoder.get_feature_names_out(Some(&input_names)).unwrap();
        assert_eq!(names, vec!["category_target_enc"]);
    }

    #[test]
    fn test_target_encoder_builder_methods() {
        let encoder = TargetEncoder::new()
            .with_smooth(5.0)
            .with_handle_unknown(UnknownCategoryHandling::Ignore)
            .with_cv(3);

        assert!((encoder.smooth() - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_target_encoder_single_sample_category() {
        // Test behavior when a category has only one sample
        let mut encoder = TargetEncoder::new().with_smooth(1.0);
        let x = array![[0.0], [0.0], [0.0], [1.0]]; // Category 1 has single sample
        let y = array![1.0, 2.0, 3.0, 10.0];

        encoder.fit_with_target(&x, &y).unwrap();
        let x_encoded = encoder.transform(&array![[1.0]]).unwrap();

        // Global mean = (1+2+3+10)/4 = 4.0
        // Category 1 has count=1, mean=10
        // Encoded = (1 * 10 + 1 * 4) / (1 + 1) = 14/2 = 7
        let expected = (1.0 * 10.0 + 1.0 * 4.0) / 2.0;
        assert!((x_encoded[[0, 0]] - expected).abs() < EPSILON);
    }

    #[test]
    fn test_target_encoder_zero_smooth() {
        // With smooth=0, should just use category mean
        let mut encoder = TargetEncoder::new().with_smooth(0.0);
        let x = array![[0.0], [0.0], [1.0], [1.0]];
        let y = array![1.0, 2.0, 5.0, 6.0];

        encoder.fit_with_target(&x, &y).unwrap();
        let x_encoded = encoder.transform(&x).unwrap();

        // Category 0 mean = 1.5, Category 1 mean = 5.5
        // With smooth=0: encoded = (count * mean + 0) / count = mean
        assert!((x_encoded[[0, 0]] - 1.5).abs() < EPSILON);
        assert!((x_encoded[[2, 0]] - 5.5).abs() < EPSILON);
    }
}

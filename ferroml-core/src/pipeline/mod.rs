//! Pipeline Execution
//!
//! This module provides Pipeline for chaining transformers and estimators into
//! reusable workflows with automatic fit/transform sequencing, and FeatureUnion
//! for parallel feature extraction.
//!
//! ## Features
//!
//! - **Sequential steps**: Chain transformers and models (transform → transform → model)
//! - **Named steps**: Access and modify individual steps by name
//! - **Combined search space**: Merge hyperparameter spaces from all steps for HPO
//! - **Caching**: Optional caching of intermediate transformations
//! - **Nested parameters**: Set parameters with "step__param" syntax
//! - **FeatureUnion**: Parallel feature extraction with concatenation
//!
//! ## Example
//!
//! ```
//! # fn main() -> ferroml_core::Result<()> {
//! use ferroml_core::pipeline::{Pipeline, PipelineStep, FeatureUnion};
//! use ferroml_core::preprocessing::scalers::{StandardScaler, MinMaxScaler};
//! use ferroml_core::preprocessing::Transformer;
//! use ferroml_core::models::LinearRegression;
//! # use ndarray::{Array1, Array2};
//! # let x_train = Array2::from_shape_vec((10, 2), (0..20).map(|i| (i as f64 * 0.7).sin()).collect()).unwrap();
//! # let y_train = Array1::from_vec((0..10).map(|i| i as f64).collect());
//! # let x_test = x_train.clone();
//!
//! // Create a pipeline with scaler and model
//! let mut pipeline = Pipeline::new()
//!     .add_transformer("scaler", StandardScaler::new())
//!     .add_model("regressor", LinearRegression::new());
//!
//! // Fit the entire pipeline
//! pipeline.fit(&x_train, &y_train)?;
//!
//! // Predict transforms data through all steps
//! let predictions = pipeline.predict(&x_test)?;
//!
//! // Access combined search space for HPO
//! let space = pipeline.search_space();
//!
//! // FeatureUnion for parallel feature extraction
//! let mut feature_union = FeatureUnion::new()
//!     .add_transformer("scaler", StandardScaler::new())
//!     .add_transformer("minmax", MinMaxScaler::new());
//!
//! // Fit and transform - runs transformers in parallel and concatenates
//! let combined_features = feature_union.fit_transform(&x_train)?;
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "sparse")]
pub mod text_pipeline;

#[cfg(feature = "sparse")]
pub use text_pipeline::{
    PipelineSparseModel, PipelineSparseTransformer, PipelineTextTransformer, TextPipeline,
    TextPipelineStep,
};

use crate::hpo::{ParameterValue, SearchSpace};
use crate::models::Model;
use crate::preprocessing::Transformer;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};

// =============================================================================
// Pipeline Step Types
// =============================================================================

/// A step in a pipeline that can be either a transformer or a model.
pub enum PipelineStep {
    /// A feature transformer (preprocessing step)
    Transform(Box<dyn PipelineTransformer>),
    /// A final model (must be the last step)
    Model(Box<dyn PipelineModel>),
}

impl fmt::Debug for PipelineStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transform(_) => write!(f, "PipelineStep::Transform(<dyn Transformer>)"),
            Self::Model(_) => write!(f, "PipelineStep::Model(<dyn Model>)"),
        }
    }
}

/// Trait for transformers that can be used in pipelines.
///
/// Extends the base Transformer trait with additional methods needed for
/// pipeline integration like search space access and cloning.
pub trait PipelineTransformer: Transformer {
    /// Get the hyperparameter search space for this transformer
    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    /// Clone into a boxed trait object
    fn clone_boxed(&self) -> Box<dyn PipelineTransformer>;

    /// Set a hyperparameter by name
    fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()> {
        let _ = (name, value);
        Err(FerroError::invalid_input(format!(
            "Parameter '{}' not supported",
            name
        )))
    }

    /// Get transformer type name
    fn name(&self) -> &str {
        "Transformer"
    }
}

/// Trait for models that can be used in pipelines.
///
/// Extends the base Model trait with additional methods needed for
/// pipeline integration.
pub trait PipelineModel: Model {
    /// Clone into a boxed trait object
    fn clone_boxed(&self) -> Box<dyn PipelineModel>;

    /// Set a hyperparameter by name
    fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()> {
        let _ = (name, value);
        Err(FerroError::invalid_input(format!(
            "Parameter '{}' not supported",
            name
        )))
    }

    /// Get model type name
    fn name(&self) -> &str {
        "Model"
    }
}

// =============================================================================
// Pipeline Cache
// =============================================================================

/// Caching strategy for intermediate transformations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheStrategy {
    /// No caching
    None,
    /// Cache in memory
    Memory,
}

impl Default for CacheStrategy {
    fn default() -> Self {
        Self::None
    }
}

/// Cache for intermediate pipeline transformations
#[derive(Debug, Clone)]
pub struct PipelineCache {
    /// Cached transformations by step name
    cache: HashMap<String, CachedTransform>,
    /// Cache strategy
    strategy: CacheStrategy,
}

/// A cached transformation result
#[derive(Debug, Clone)]
struct CachedTransform {
    /// The transformed data
    data: Array2<f64>,
    /// Hash of the input data for validation
    input_hash: u64,
}

impl PipelineCache {
    /// Create a new cache with the given strategy
    pub fn new(strategy: CacheStrategy) -> Self {
        Self {
            cache: HashMap::new(),
            strategy,
        }
    }

    /// Check if caching is enabled
    pub fn is_enabled(&self) -> bool {
        self.strategy != CacheStrategy::None
    }

    /// Get cached data for a step if valid
    pub fn get(&self, step_name: &str, input: &Array2<f64>) -> Option<&Array2<f64>> {
        if !self.is_enabled() {
            return None;
        }

        self.cache.get(step_name).and_then(|cached| {
            let current_hash = Self::hash_array(input);
            if cached.input_hash == current_hash {
                Some(&cached.data)
            } else {
                None
            }
        })
    }

    /// Store cached data for a step
    pub fn set(&mut self, step_name: &str, input: &Array2<f64>, output: Array2<f64>) {
        if !self.is_enabled() {
            return;
        }

        let input_hash = Self::hash_array(input);
        self.cache.insert(
            step_name.to_string(),
            CachedTransform {
                data: output,
                input_hash,
            },
        );
    }

    /// Clear all cached data
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Invalidate cache for a specific step and all following steps
    pub fn invalidate_from(&mut self, step_names: &[String], starting_step: &str) {
        let mut should_invalidate = false;
        for name in step_names {
            if name == starting_step {
                should_invalidate = true;
            }
            if should_invalidate {
                self.cache.remove(name);
            }
        }
    }

    /// Compute a hash for an array (for cache validation)
    fn hash_array(arr: &Array2<f64>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();

        // Hash shape
        arr.shape().hash(&mut hasher);

        // Hash a sample of values for efficiency
        let total = arr.len();
        let sample_size = total.min(1000);
        let step = if total > sample_size {
            total / sample_size
        } else {
            1
        };

        for (i, val) in arr.iter().enumerate() {
            if i % step == 0 {
                val.to_bits().hash(&mut hasher);
            }
        }

        hasher.finish()
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new(CacheStrategy::None)
    }
}

// =============================================================================
// Pipeline
// =============================================================================

/// A machine learning pipeline that chains transformers and a final model.
///
/// Pipelines ensure proper sequencing of fit/transform operations and prevent
/// data leakage by always fitting on training data first.
///
/// ## Structure
///
/// A pipeline consists of:
/// - Zero or more **transformers** that modify features
/// - Optionally, a final **model** that makes predictions
///
/// ## Fit/Transform Flow
///
/// ```text
/// fit():
///   X_train ─→ transformer1.fit_transform() ─→ X1 ─→ transformer2.fit_transform() ─→ X2 ─→ model.fit()
///
/// predict():
///   X_test ─→ transformer1.transform() ─→ X1 ─→ transformer2.transform() ─→ X2 ─→ model.predict()
/// ```
///
/// ## Named Steps
///
/// Each step has a name for accessing/modifying it. Parameter names use
/// double-underscore convention: `"step_name__param_name"`.
pub struct Pipeline {
    /// Named steps: (name, step) pairs in order
    steps: Vec<(String, PipelineStep)>,
    /// Cache for intermediate transformations
    cache: PipelineCache,
    /// Whether the pipeline is fitted
    fitted: bool,
    /// Number of features expected
    n_features_in: Option<usize>,
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let step_names: Vec<_> = self.steps.iter().map(|(name, _)| name.as_str()).collect();
        f.debug_struct("Pipeline")
            .field("steps", &step_names)
            .field("cache_strategy", &self.cache.strategy)
            .field("fitted", &self.fitted)
            .field("n_features_in", &self.n_features_in)
            .finish()
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Pipeline {
    /// Create a new empty pipeline
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            cache: PipelineCache::default(),
            fitted: false,
            n_features_in: None,
        }
    }

    /// Add a transformer step to the pipeline
    ///
    /// # Arguments
    ///
    /// * `name` - Unique name for this step
    /// * `transformer` - The transformer to add
    ///
    /// # Panics
    ///
    /// Panics if a step with this name already exists, or if a model has
    /// already been added (models must be the final step).
    pub fn add_transformer<T: PipelineTransformer + 'static>(
        mut self,
        name: impl Into<String>,
        transformer: T,
    ) -> Self {
        let name = name.into();

        // Check for duplicate names
        assert!(
            !self.steps.iter().any(|(n, _)| n == &name),
            "Step name '{}' already exists in pipeline",
            name
        );

        // Check that no model has been added yet
        assert!(
            !self.has_model(),
            "Cannot add transformer after model. Models must be the final step."
        );

        self.steps
            .push((name, PipelineStep::Transform(Box::new(transformer))));
        self
    }

    /// Add a model as the final step of the pipeline
    ///
    /// # Arguments
    ///
    /// * `name` - Unique name for this step
    /// * `model` - The model to add
    ///
    /// # Panics
    ///
    /// Panics if a step with this name already exists, or if a model has
    /// already been added.
    pub fn add_model<M: PipelineModel + 'static>(
        mut self,
        name: impl Into<String>,
        model: M,
    ) -> Self {
        let name = name.into();

        // Check for duplicate names
        assert!(
            !self.steps.iter().any(|(n, _)| n == &name),
            "Step name '{}' already exists in pipeline",
            name
        );

        // Check that no model has been added yet
        assert!(
            !self.has_model(),
            "Pipeline already has a model. Only one model is allowed."
        );

        self.steps
            .push((name, PipelineStep::Model(Box::new(model))));
        self
    }

    /// Enable caching of intermediate transformations
    pub fn with_cache(mut self, strategy: CacheStrategy) -> Self {
        self.cache = PipelineCache::new(strategy);
        self
    }

    /// Check if the pipeline has a model as the final step
    pub fn has_model(&self) -> bool {
        self.steps
            .last()
            .map(|(_, step)| matches!(step, PipelineStep::Model(_)))
            .unwrap_or(false)
    }

    /// Get the names of all steps
    pub fn step_names(&self) -> Vec<&str> {
        self.steps.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// Get the number of steps
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Check if the pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Check if the pipeline is fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Get the number of input features (after fitting)
    pub fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    /// Fit the pipeline to training data
    ///
    /// For each transformer: fit_transform on the data, passing the transformed
    /// data to the next step. For the final model: fit on the transformed data.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - Target values of shape (n_samples,)
    ///
    /// # Returns
    ///
    /// * `Ok(())` on success
    /// * `Err(FerroError)` on failure
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        if self.steps.is_empty() {
            return Err(FerroError::invalid_input(
                "Pipeline is empty. Add at least one step.",
            ));
        }

        // Validate input
        if x.nrows() != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("X has {} rows", x.nrows()),
                format!("y has {} elements", y.len()),
            ));
        }

        self.n_features_in = Some(x.ncols());
        self.cache.clear();

        // Transform through all steps
        let mut current_x = x.clone();

        for (name, step) in &mut self.steps {
            match step {
                PipelineStep::Transform(transformer) => {
                    // Save input BEFORE transformation for cache key
                    let input_for_cache = current_x.clone();
                    current_x = transformer.fit_transform(&current_x)?;
                    // Cache with input before transformation as key (not original x)
                    self.cache.set(name, &input_for_cache, current_x.clone());
                }
                PipelineStep::Model(model) => {
                    model.fit(&current_x, y)?;
                }
            }
        }

        self.fitted = true;
        Ok(())
    }

    /// Transform data through all transformer steps
    ///
    /// Does not apply the final model (if any). Use `predict` for that.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// * Transformed data
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(FerroError::not_fitted("transform"));
        }

        self.validate_n_features(x)?;

        let mut current_x = x.clone();

        for (name, step) in &self.steps {
            match step {
                PipelineStep::Transform(transformer) => {
                    // Check cache first
                    if let Some(cached) = self.cache.get(name, &current_x) {
                        current_x = cached.clone();
                    } else {
                        current_x = transformer.transform(&current_x)?;
                    }
                }
                PipelineStep::Model(_) => {
                    // Stop before the model
                    break;
                }
            }
        }

        Ok(current_x)
    }

    /// Fit and transform in one step
    ///
    /// More efficient than calling `fit(x, y)` followed by `transform(x)`
    /// because it avoids recomputing transformations.
    pub fn fit_transform(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>> {
        // Inline implementation to avoid double-computation
        if self.steps.is_empty() {
            return Err(FerroError::invalid_input(
                "Pipeline is empty. Add at least one step.",
            ));
        }

        if x.nrows() != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("X has {} rows", x.nrows()),
                format!("y has {} elements", y.len()),
            ));
        }

        self.n_features_in = Some(x.ncols());
        self.cache.clear();

        let mut current_x = x.clone();

        for (name, step) in &mut self.steps {
            match step {
                PipelineStep::Transform(transformer) => {
                    let input_for_cache = current_x.clone();
                    current_x = transformer.fit_transform(&current_x)?;
                    self.cache.set(name, &input_for_cache, current_x.clone());
                }
                PipelineStep::Model(model) => {
                    model.fit(&current_x, y)?;
                    // Return the transformed data before the model
                    self.fitted = true;
                    return Ok(current_x);
                }
            }
        }

        self.fitted = true;
        Ok(current_x)
    }

    /// Predict using the pipeline
    ///
    /// Transforms data through all transformers, then applies the final model.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// * Predictions from the final model
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline doesn't have a model, or isn't fitted.
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.fitted {
            return Err(FerroError::not_fitted("predict"));
        }

        if !self.has_model() {
            return Err(FerroError::invalid_input(
                "Pipeline has no model. Use transform() for transformer-only pipelines.",
            ));
        }

        // Transform through all transformers
        let transformed = self.transform(x)?;

        // Apply the final model
        if let Some((_, PipelineStep::Model(model))) = self.steps.last() {
            model.predict(&transformed)
        } else {
            unreachable!("has_model() returned true but no model found")
        }
    }

    /// Get the combined search space from all steps
    ///
    /// Parameter names are prefixed with step name: `"step_name__param_name"`.
    pub fn search_space(&self) -> SearchSpace {
        let mut combined = SearchSpace::new();

        for (name, step) in &self.steps {
            let step_space = match step {
                PipelineStep::Transform(t) => t.search_space(),
                PipelineStep::Model(m) => m.search_space(),
            };

            for (param_name, param) in step_space.parameters {
                let prefixed_name = format!("{}__{}", name, param_name);
                combined.parameters.insert(prefixed_name, param);
            }
        }

        combined
    }

    /// Set hyperparameters using nested naming convention
    ///
    /// Parameter names use double-underscore: `"step_name__param_name"`.
    ///
    /// # Arguments
    ///
    /// * `params` - Map of parameter names to values
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> ferroml_core::Result<()> {
    /// # use ferroml_core::pipeline::Pipeline;
    /// # use ferroml_core::preprocessing::scalers::StandardScaler;
    /// # use ferroml_core::models::LinearRegression;
    /// # use ferroml_core::hpo::ParameterValue;
    /// # use std::collections::HashMap;
    /// # let mut pipeline = Pipeline::new()
    /// #     .add_transformer("scaler", StandardScaler::new())
    /// #     .add_model("model", LinearRegression::new());
    /// let mut params = HashMap::new();
    /// params.insert("scaler__with_mean".to_string(), ParameterValue::Bool(true));
    /// params.insert("model__alpha".to_string(), ParameterValue::Float(0.1));
    /// pipeline.set_params(&params)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_params(&mut self, params: &HashMap<String, ParameterValue>) -> Result<()> {
        // Collect step names first to avoid borrow issues
        let step_names: Vec<String> = self.steps.iter().map(|(n, _)| n.clone()).collect();

        for (full_name, value) in params {
            let parts: Vec<&str> = full_name.splitn(2, "__").collect();
            if parts.len() != 2 {
                return Err(FerroError::invalid_input(format!(
                    "Invalid parameter name '{}'. Expected format: 'step__param'",
                    full_name
                )));
            }

            let (step_name, param_name) = (parts[0], parts[1]);

            // Find the step and set the parameter
            let step = self
                .steps
                .iter_mut()
                .find(|(name, _)| name == step_name)
                .ok_or_else(|| {
                    FerroError::invalid_input(format!(
                        "Step '{}' not found in pipeline. Available: {:?}",
                        step_name, step_names
                    ))
                })?;

            match &mut step.1 {
                PipelineStep::Transform(t) => t.set_param(param_name, value)?,
                PipelineStep::Model(m) => m.set_param(param_name, value)?,
            }

            // Invalidate cache from this step onwards
            self.cache.invalidate_from(&step_names, step_name);
        }

        // Mark as unfitted since parameters changed
        self.fitted = false;

        Ok(())
    }

    /// Get parameters from all steps
    ///
    /// Returns a map of `"step__param"` to string representation of values.
    pub fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();

        for (name, step) in &self.steps {
            match step {
                PipelineStep::Transform(t) => {
                    params.insert(format!("{}__type", name), t.name().to_string());
                }
                PipelineStep::Model(m) => {
                    params.insert(format!("{}__type", name), m.name().to_string());
                }
            }
        }

        params
    }

    /// Validate that input has the expected number of features
    fn validate_n_features(&self, x: &Array2<f64>) -> Result<()> {
        if let Some(expected) = self.n_features_in {
            if x.ncols() != expected {
                return Err(FerroError::shape_mismatch(
                    format!("{} features", expected),
                    format!("{} features", x.ncols()),
                ));
            }
        }
        Ok(())
    }
}

// =============================================================================
// Make Pipeline
// =============================================================================

/// Create a pipeline from a sequence of steps.
///
/// A convenience function for creating pipelines with auto-generated names.
///
/// # Arguments
///
/// * `transformers` - Vector of transformers to chain
/// * `model` - Optional final model
///
/// # Example
///
/// ```
/// use ferroml_core::pipeline::make_pipeline;
/// use ferroml_core::preprocessing::scalers::StandardScaler;
/// use ferroml_core::models::LinearRegression;
///
/// let pipeline = make_pipeline(
///     vec![Box::new(StandardScaler::new())],
///     Some(Box::new(LinearRegression::new())),
/// );
/// ```
pub fn make_pipeline(
    transformers: Vec<Box<dyn PipelineTransformer>>,
    model: Option<Box<dyn PipelineModel>>,
) -> Pipeline {
    let mut pipeline = Pipeline::new();

    for (i, transformer) in transformers.into_iter().enumerate() {
        let name = format!("step_{}", i);
        pipeline
            .steps
            .push((name, PipelineStep::Transform(transformer)));
    }

    if let Some(m) = model {
        let name = "model".to_string();
        pipeline.steps.push((name, PipelineStep::Model(m)));
    }

    pipeline
}

// =============================================================================
// FeatureUnion
// =============================================================================

/// A transformer that applies multiple transformers in parallel and concatenates
/// their outputs horizontally.
///
/// FeatureUnion is useful when you want to combine different feature extraction
/// methods on the same input data. Each transformer processes the original input
/// independently (in parallel via rayon), and the resulting features are concatenated.
///
/// ## Example
///
/// ```
/// # fn main() -> ferroml_core::Result<()> {
/// use ferroml_core::pipeline::FeatureUnion;
/// use ferroml_core::preprocessing::scalers::{StandardScaler, MinMaxScaler};
/// use ferroml_core::preprocessing::Transformer;
/// # use ndarray::Array2;
/// # let x_train = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
///
/// // Create a feature union with two transformers
/// let mut feature_union = FeatureUnion::new()
///     .add_transformer("scaler", StandardScaler::new())
///     .add_transformer("minmax", MinMaxScaler::new());
///
/// // Fit and transform - runs in parallel
/// let combined = feature_union.fit_transform(&x_train)?;
/// # Ok(())
/// # }
/// ```
pub struct FeatureUnion {
    /// Named transformers: (name, transformer) pairs
    transformers: Vec<(String, Box<dyn PipelineTransformer>)>,
    /// Whether the union is fitted
    fitted: bool,
    /// Number of input features
    n_features_in: Option<usize>,
    /// Number of output features (sum of all transformer outputs)
    n_features_out: Option<usize>,
    /// Feature names for output columns
    feature_names_out: Option<Vec<String>>,
    /// Weights for each transformer's output (optional)
    transformer_weights: Option<HashMap<String, f64>>,
}

impl fmt::Debug for FeatureUnion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let transformer_names: Vec<_> = self
            .transformers
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        f.debug_struct("FeatureUnion")
            .field("transformers", &transformer_names)
            .field("fitted", &self.fitted)
            .field("n_features_in", &self.n_features_in)
            .field("n_features_out", &self.n_features_out)
            .finish()
    }
}

impl Default for FeatureUnion {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureUnion {
    /// Create a new empty FeatureUnion
    pub fn new() -> Self {
        Self {
            transformers: Vec::new(),
            fitted: false,
            n_features_in: None,
            n_features_out: None,
            feature_names_out: None,
            transformer_weights: None,
        }
    }

    /// Add a named transformer to the union
    ///
    /// # Arguments
    ///
    /// * `name` - Unique name for this transformer
    /// * `transformer` - The transformer to add
    ///
    /// # Panics
    ///
    /// Panics if a transformer with this name already exists.
    pub fn add_transformer<T: PipelineTransformer + 'static>(
        mut self,
        name: impl Into<String>,
        transformer: T,
    ) -> Self {
        let name = name.into();

        // Check for duplicate names
        assert!(
            !self.transformers.iter().any(|(n, _)| n == &name),
            "Transformer name '{}' already exists in FeatureUnion",
            name
        );

        self.transformers.push((name, Box::new(transformer)));
        self
    }

    /// Set weights for each transformer's output
    ///
    /// Weights are applied as multipliers to the transformed features.
    /// If not set, all transformers have equal weight (1.0).
    ///
    /// # Arguments
    ///
    /// * `weights` - Map of transformer name to weight
    pub fn with_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.transformer_weights = Some(weights);
        self
    }

    /// Get the names of all transformers
    pub fn transformer_names(&self) -> Vec<&str> {
        self.transformers
            .iter()
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get the number of transformers
    pub fn n_transformers(&self) -> usize {
        self.transformers.len()
    }

    /// Check if the union is empty
    pub fn is_empty(&self) -> bool {
        self.transformers.is_empty()
    }

    /// Get the combined search space from all transformers
    ///
    /// Parameter names are prefixed with transformer name: `"transformer_name__param_name"`.
    pub fn search_space(&self) -> SearchSpace {
        let mut combined = SearchSpace::new();

        for (name, transformer) in &self.transformers {
            let transformer_space = transformer.search_space();

            for (param_name, param) in transformer_space.parameters {
                let prefixed_name = format!("{}__{}", name, param_name);
                combined.parameters.insert(prefixed_name, param);
            }
        }

        combined
    }

    /// Set hyperparameters using nested naming convention
    ///
    /// Parameter names use double-underscore: `"transformer_name__param_name"`.
    ///
    /// # Arguments
    ///
    /// * `params` - Map of parameter names to values
    pub fn set_params(&mut self, params: &HashMap<String, ParameterValue>) -> Result<()> {
        let transformer_names: Vec<String> =
            self.transformers.iter().map(|(n, _)| n.clone()).collect();

        for (full_name, value) in params {
            let parts: Vec<&str> = full_name.splitn(2, "__").collect();
            if parts.len() != 2 {
                return Err(FerroError::invalid_input(format!(
                    "Invalid parameter name '{}'. Expected format: 'transformer__param'",
                    full_name
                )));
            }

            let (transformer_name, param_name) = (parts[0], parts[1]);

            // Find the transformer and set the parameter
            let transformer = self
                .transformers
                .iter_mut()
                .find(|(name, _)| name == transformer_name)
                .ok_or_else(|| {
                    FerroError::invalid_input(format!(
                        "Transformer '{}' not found in FeatureUnion. Available: {:?}",
                        transformer_name, transformer_names
                    ))
                })?;

            transformer.1.set_param(param_name, value)?;
        }

        // Mark as unfitted since parameters changed
        self.fitted = false;

        Ok(())
    }

    /// Get parameters from all transformers
    ///
    /// Returns a map of `"transformer__param"` to string representation of values.
    pub fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();

        for (name, transformer) in &self.transformers {
            params.insert(format!("{}__type", name), transformer.name().to_string());
        }

        params
    }

    /// Validate that input has the expected number of features
    fn validate_n_features(&self, x: &Array2<f64>) -> Result<()> {
        if let Some(expected) = self.n_features_in {
            if x.ncols() != expected {
                return Err(FerroError::shape_mismatch(
                    format!("{} features", expected),
                    format!("{} features", x.ncols()),
                ));
            }
        }
        Ok(())
    }
}

impl Transformer for FeatureUnion {
    /// Fit all transformers to the input data (in parallel)
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        if self.transformers.is_empty() {
            return Err(FerroError::invalid_input(
                "FeatureUnion is empty. Add at least one transformer.",
            ));
        }

        self.n_features_in = Some(x.ncols());

        // Clone transformers with indices for parallel processing
        let transformer_clones: Vec<_> = self
            .transformers
            .iter()
            .enumerate()
            .map(|(idx, (name, t))| (idx, name.clone(), t.clone_boxed()))
            .collect();

        // Fit all transformers in parallel
        let results: Vec<Result<(usize, String, Box<dyn PipelineTransformer>)>> =
            transformer_clones
                .into_par_iter()
                .map(|(idx, name, mut transformer)| {
                    transformer.fit(x)?;
                    Ok((idx, name, transformer))
                })
                .collect();

        // Check for errors and collect results
        let mut fitted_results: Vec<(usize, String, Box<dyn PipelineTransformer>)> =
            Vec::with_capacity(results.len());
        for result in results {
            fitted_results.push(result?);
        }

        // Sort back to original order by index
        fitted_results.sort_by_key(|(idx, _, _)| *idx);

        // Build feature names and count total features
        let mut total_features = 0;
        let mut feature_names = Vec::new();

        for (_, name, transformer) in &fitted_results {
            if let Some(n_out) = transformer.n_features_out() {
                // Generate feature names
                if let Some(names) = transformer.get_feature_names_out(None) {
                    for fname in names {
                        feature_names.push(format!("{}_{}", name, fname));
                    }
                } else {
                    for i in 0..n_out {
                        feature_names.push(format!("{}_feature_{}", name, i));
                    }
                }
                total_features += n_out;
            }
        }

        // Replace transformers with fitted versions
        self.transformers = fitted_results
            .into_iter()
            .map(|(_, name, transformer)| (name, transformer))
            .collect();

        self.n_features_out = Some(total_features);
        self.feature_names_out = Some(feature_names);
        self.fitted = true;

        Ok(())
    }

    /// Transform data through all transformers (in parallel) and concatenate results
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(FerroError::not_fitted("transform"));
        }

        self.validate_n_features(x)?;

        // Transform through each transformer in parallel
        let results: Vec<Result<(String, Array2<f64>)>> = self
            .transformers
            .par_iter()
            .map(|(name, transformer)| {
                let transformed = transformer.transform(x)?;
                Ok((name.clone(), transformed))
            })
            .collect();

        // Check for errors and collect transformed data in order
        let mut transformed_data: Vec<(String, Array2<f64>)> = Vec::with_capacity(results.len());
        for result in results {
            transformed_data.push(result?);
        }

        // Sort back to original order
        let mut ordered_data: Vec<Array2<f64>> = Vec::with_capacity(transformed_data.len());
        for (orig_name, _) in &self.transformers {
            if let Some((_, data)) = transformed_data.iter().find(|(n, _)| n == orig_name) {
                let mut data = data.clone();

                // Apply weights if specified
                if let Some(ref weights) = self.transformer_weights {
                    if let Some(&weight) = weights.get(orig_name) {
                        data *= weight;
                    }
                }

                ordered_data.push(data);
            }
        }

        // Concatenate all transformed data horizontally
        if ordered_data.is_empty() {
            return Err(FerroError::invalid_input(
                "No transformer outputs to concatenate",
            ));
        }

        let n_samples = x.nrows();
        let total_cols: usize = ordered_data.iter().map(|d| d.ncols()).sum();

        let mut result = Array2::zeros((n_samples, total_cols));
        let mut col_offset = 0;

        for data in ordered_data {
            let n_cols = data.ncols();
            result
                .slice_mut(ndarray::s![.., col_offset..col_offset + n_cols])
                .assign(&data);
            col_offset += n_cols;
        }

        Ok(result)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
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

impl PipelineTransformer for FeatureUnion {
    fn search_space(&self) -> SearchSpace {
        self.search_space()
    }

    fn clone_boxed(&self) -> Box<dyn PipelineTransformer> {
        let cloned_transformers: Vec<_> = self
            .transformers
            .iter()
            .map(|(name, t)| (name.clone(), t.clone_boxed()))
            .collect();

        Box::new(FeatureUnion {
            transformers: cloned_transformers,
            fitted: self.fitted,
            n_features_in: self.n_features_in,
            n_features_out: self.n_features_out,
            feature_names_out: self.feature_names_out.clone(),
            transformer_weights: self.transformer_weights.clone(),
        })
    }

    fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()> {
        // Handle nested parameters (transformer__param)
        let parts: Vec<&str> = name.splitn(2, "__").collect();
        if parts.len() == 2 {
            let (transformer_name, param_name) = (parts[0], parts[1]);

            for (t_name, transformer) in &mut self.transformers {
                if t_name == transformer_name {
                    return transformer.set_param(param_name, value);
                }
            }

            return Err(FerroError::invalid_input(format!(
                "Transformer '{}' not found",
                transformer_name
            )));
        }

        Err(FerroError::invalid_input(format!(
            "Parameter '{}' not supported. Use 'transformer__param' format for nested parameters.",
            name
        )))
    }

    fn name(&self) -> &str {
        "FeatureUnion"
    }
}

/// Create a FeatureUnion from a list of (name, transformer) pairs.
///
/// A convenience function for creating feature unions with named transformers.
///
/// # Arguments
///
/// * `transformers` - Vector of (name, transformer) pairs
///
/// # Example
///
/// ```
/// use ferroml_core::pipeline::make_feature_union;
/// use ferroml_core::preprocessing::scalers::{StandardScaler, MinMaxScaler};
///
/// let union = make_feature_union(vec![
///     ("scaler".to_string(), Box::new(StandardScaler::new())),
///     ("minmax".to_string(), Box::new(MinMaxScaler::new())),
/// ]);
/// ```
pub fn make_feature_union(
    transformers: Vec<(String, Box<dyn PipelineTransformer>)>,
) -> FeatureUnion {
    let mut union = FeatureUnion::new();
    union.transformers = transformers;
    union
}

// =============================================================================
// ColumnTransformer
// =============================================================================

/// Column selection for ColumnTransformer.
///
/// Specifies which columns a transformer should be applied to.
#[derive(Debug, Clone)]
pub enum ColumnSelector {
    /// Select columns by their indices (0-based)
    Indices(Vec<usize>),
    /// Select columns by a boolean mask (true = include)
    Mask(Vec<bool>),
    /// Select all columns
    All,
    /// Select all remaining columns not used by other transformers
    Remainder,
}

impl ColumnSelector {
    /// Create a selector from column indices.
    pub fn indices(indices: impl IntoIterator<Item = usize>) -> Self {
        Self::Indices(indices.into_iter().collect())
    }

    /// Create a selector from a boolean mask.
    pub fn mask(mask: impl IntoIterator<Item = bool>) -> Self {
        Self::Mask(mask.into_iter().collect())
    }

    /// Create a selector that selects all columns.
    pub fn all() -> Self {
        Self::All
    }

    /// Resolve the selector to actual column indices.
    ///
    /// # Arguments
    ///
    /// * `n_features` - Total number of features
    /// * `used_columns` - Columns already used by other transformers (for Remainder)
    fn resolve(&self, n_features: usize, used_columns: &[usize]) -> Vec<usize> {
        match self {
            ColumnSelector::Indices(indices) => indices.clone(),
            ColumnSelector::Mask(mask) => mask
                .iter()
                .enumerate()
                .filter_map(|(i, &included)| if included { Some(i) } else { None })
                .collect(),
            ColumnSelector::All => (0..n_features).collect(),
            ColumnSelector::Remainder => (0..n_features)
                .filter(|i| !used_columns.contains(i))
                .collect(),
        }
    }
}

/// How to handle columns not explicitly assigned to any transformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemainderHandling {
    /// Drop unassigned columns (exclude from output)
    Drop,
    /// Pass unassigned columns through unchanged
    Passthrough,
}

impl Default for RemainderHandling {
    fn default() -> Self {
        Self::Drop
    }
}

/// A transformer specification for ColumnTransformer.
struct TransformerSpec {
    /// Name of this transformer
    name: String,
    /// The transformer to apply
    transformer: Box<dyn PipelineTransformer>,
    /// Which columns to apply it to
    columns: ColumnSelector,
    /// Resolved column indices (set during fit)
    resolved_indices: Option<Vec<usize>>,
}

impl fmt::Debug for TransformerSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransformerSpec")
            .field("name", &self.name)
            .field("columns", &self.columns)
            .field("resolved_indices", &self.resolved_indices)
            .finish()
    }
}

/// Applies different transformers to different subsets of columns and concatenates
/// the results.
///
/// This is useful when different features need different preprocessing. For example,
/// you might want to scale numeric features while one-hot encoding categorical features.
///
/// ## Column Selection
///
/// Columns can be selected by:
/// - **Indices**: Specific column positions (e.g., `[0, 2, 5]`)
/// - **Mask**: Boolean array (e.g., `[true, false, true, ...]`)
/// - **All**: All columns
/// - **Remainder**: Columns not used by other transformers
///
/// ## Remainder Handling
///
/// Columns not assigned to any transformer can be:
/// - **Dropped**: Excluded from output (default)
/// - **Passthrough**: Included unchanged in output
///
/// ## Example
///
/// ```
/// # fn main() -> ferroml_core::Result<()> {
/// use ferroml_core::pipeline::{ColumnTransformer, ColumnSelector, RemainderHandling};
/// use ferroml_core::preprocessing::scalers::{StandardScaler, MinMaxScaler};
/// use ferroml_core::preprocessing::Transformer;
/// # use ndarray::Array2;
/// # let x = Array2::from_shape_vec((10, 4), (0..40).map(|i| i as f64).collect()).unwrap();
///
/// // Scale columns 0-2 (numeric), min-max scale column 3
/// let mut ct = ColumnTransformer::new()
///     .add_transformer("scaler", StandardScaler::new(), ColumnSelector::indices([0, 1, 2]))
///     .add_transformer("minmax", MinMaxScaler::new(), ColumnSelector::indices([3]))
///     .with_remainder(RemainderHandling::Drop);
///
/// let x_transformed = ct.fit_transform(&x)?;
/// # Ok(())
/// # }
/// ```
pub struct ColumnTransformer {
    /// Named transformers with their column specifications
    transformers: Vec<TransformerSpec>,
    /// How to handle columns not assigned to any transformer
    remainder: RemainderHandling,
    /// Whether the transformer is fitted
    fitted: bool,
    /// Number of input features
    n_features_in: Option<usize>,
    /// Number of output features
    n_features_out: Option<usize>,
    /// Feature names for output columns
    feature_names_out: Option<Vec<String>>,
    /// Indices of remainder columns (columns not assigned to any transformer)
    remainder_indices: Option<Vec<usize>>,
    /// Output slices: (start, end) for each transformer's output in the final array
    output_slices: Vec<(usize, usize)>,
    /// Remainder output slice (if passthrough)
    remainder_slice: Option<(usize, usize)>,
}

impl fmt::Debug for ColumnTransformer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let transformer_names: Vec<_> = self.transformers.iter().map(|t| t.name.as_str()).collect();
        f.debug_struct("ColumnTransformer")
            .field("transformers", &transformer_names)
            .field("remainder", &self.remainder)
            .field("fitted", &self.fitted)
            .field("n_features_in", &self.n_features_in)
            .field("n_features_out", &self.n_features_out)
            .finish()
    }
}

impl Default for ColumnTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl ColumnTransformer {
    /// Create a new empty ColumnTransformer.
    pub fn new() -> Self {
        Self {
            transformers: Vec::new(),
            remainder: RemainderHandling::default(),
            fitted: false,
            n_features_in: None,
            n_features_out: None,
            feature_names_out: None,
            remainder_indices: None,
            output_slices: Vec::new(),
            remainder_slice: None,
        }
    }

    /// Add a named transformer that operates on specific columns.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique name for this transformer
    /// * `transformer` - The transformer to apply
    /// * `columns` - Which columns to apply it to
    ///
    /// # Panics
    ///
    /// Panics if a transformer with this name already exists.
    pub fn add_transformer<T: PipelineTransformer + 'static>(
        mut self,
        name: impl Into<String>,
        transformer: T,
        columns: ColumnSelector,
    ) -> Self {
        let name = name.into();

        // Check for duplicate names
        assert!(
            !self.transformers.iter().any(|t| t.name == name),
            "Transformer name '{}' already exists in ColumnTransformer",
            name
        );

        self.transformers.push(TransformerSpec {
            name,
            transformer: Box::new(transformer),
            columns,
            resolved_indices: None,
        });
        self
    }

    /// Configure how to handle columns not assigned to any transformer.
    ///
    /// # Arguments
    ///
    /// * `handling` - Either `Drop` (exclude from output) or `Passthrough` (include unchanged)
    pub fn with_remainder(mut self, handling: RemainderHandling) -> Self {
        self.remainder = handling;
        self
    }

    /// Get the names of all transformers.
    pub fn transformer_names(&self) -> Vec<&str> {
        self.transformers.iter().map(|t| t.name.as_str()).collect()
    }

    /// Get the number of transformers.
    pub fn n_transformers(&self) -> usize {
        self.transformers.len()
    }

    /// Check if the transformer is empty.
    pub fn is_empty(&self) -> bool {
        self.transformers.is_empty()
    }

    /// Get the combined search space from all transformers.
    ///
    /// Parameter names are prefixed with transformer name: `"transformer_name__param_name"`.
    pub fn search_space(&self) -> SearchSpace {
        let mut combined = SearchSpace::new();

        for spec in &self.transformers {
            let transformer_space = spec.transformer.search_space();

            for (param_name, param) in transformer_space.parameters {
                let prefixed_name = format!("{}__{}", spec.name, param_name);
                combined.parameters.insert(prefixed_name, param);
            }
        }

        combined
    }

    /// Set hyperparameters using nested naming convention.
    ///
    /// Parameter names use double-underscore: `"transformer_name__param_name"`.
    ///
    /// # Arguments
    ///
    /// * `params` - Map of parameter names to values
    pub fn set_params(&mut self, params: &HashMap<String, ParameterValue>) -> Result<()> {
        let transformer_names: Vec<String> =
            self.transformers.iter().map(|t| t.name.clone()).collect();

        for (full_name, value) in params {
            let parts: Vec<&str> = full_name.splitn(2, "__").collect();
            if parts.len() != 2 {
                return Err(FerroError::invalid_input(format!(
                    "Invalid parameter name '{}'. Expected format: 'transformer__param'",
                    full_name
                )));
            }

            let (transformer_name, param_name) = (parts[0], parts[1]);

            // Find the transformer and set the parameter
            let spec = self
                .transformers
                .iter_mut()
                .find(|t| t.name == transformer_name)
                .ok_or_else(|| {
                    FerroError::invalid_input(format!(
                        "Transformer '{}' not found in ColumnTransformer. Available: {:?}",
                        transformer_name, transformer_names
                    ))
                })?;

            spec.transformer.set_param(param_name, value)?;
        }

        // Mark as unfitted since parameters changed
        self.fitted = false;

        Ok(())
    }

    /// Get parameters from all transformers.
    ///
    /// Returns a map of `"transformer__param"` to string representation of values.
    pub fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();

        for spec in &self.transformers {
            params.insert(
                format!("{}__type", spec.name),
                spec.transformer.name().to_string(),
            );
        }

        params
    }

    /// Extract columns from input array based on indices.
    fn extract_columns(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        if indices.is_empty() {
            return Array2::zeros((x.nrows(), 0));
        }

        let n_samples = x.nrows();
        let n_cols = indices.len();
        let mut result = Array2::zeros((n_samples, n_cols));

        for (new_col, &old_col) in indices.iter().enumerate() {
            result.column_mut(new_col).assign(&x.column(old_col));
        }

        result
    }

    /// Validate that input has the expected number of features.
    fn validate_n_features(&self, x: &Array2<f64>) -> Result<()> {
        if let Some(expected) = self.n_features_in {
            if x.ncols() != expected {
                return Err(FerroError::shape_mismatch(
                    format!("{} features", expected),
                    format!("{} features", x.ncols()),
                ));
            }
        }
        Ok(())
    }
}

impl Transformer for ColumnTransformer {
    /// Fit all transformers to their respective column subsets.
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        if self.transformers.is_empty() {
            return Err(FerroError::invalid_input(
                "ColumnTransformer is empty. Add at least one transformer.",
            ));
        }

        let n_features = x.ncols();
        self.n_features_in = Some(n_features);

        // First pass: resolve column indices for non-remainder selectors
        let mut used_columns: Vec<usize> = Vec::new();
        for spec in &mut self.transformers {
            if !matches!(spec.columns, ColumnSelector::Remainder) {
                let indices = spec.columns.resolve(n_features, &[]);
                // Validate indices are in range
                for &idx in &indices {
                    if idx >= n_features {
                        return Err(FerroError::invalid_input(format!(
                            "Column index {} is out of range (n_features={})",
                            idx, n_features
                        )));
                    }
                }
                used_columns.extend(&indices);
                spec.resolved_indices = Some(indices);
            }
        }

        // Second pass: resolve remainder selectors
        for spec in &mut self.transformers {
            if matches!(spec.columns, ColumnSelector::Remainder) {
                let indices = spec.columns.resolve(n_features, &used_columns);
                spec.resolved_indices = Some(indices);
            }
        }

        // Compute remainder indices (columns not assigned to any transformer)
        let all_used: std::collections::HashSet<usize> = self
            .transformers
            .iter()
            .flat_map(|t| t.resolved_indices.as_ref().unwrap().iter().copied())
            .collect();
        self.remainder_indices = Some((0..n_features).filter(|i| !all_used.contains(i)).collect());

        // Clone transformers with indices for parallel processing
        let transformer_data: Vec<_> = self
            .transformers
            .iter()
            .enumerate()
            .map(|(idx, spec)| {
                (
                    idx,
                    spec.name.clone(),
                    spec.transformer.clone_boxed(),
                    spec.resolved_indices.clone().unwrap(),
                )
            })
            .collect();

        // Fit all transformers in parallel
        let results: Vec<Result<(usize, String, Box<dyn PipelineTransformer>, Vec<usize>)>> =
            transformer_data
                .into_par_iter()
                .map(|(idx, name, mut transformer, indices)| {
                    let x_subset = Self::extract_columns(x, &indices);
                    if !x_subset.is_empty() && x_subset.ncols() > 0 {
                        transformer.fit(&x_subset)?;
                    }
                    Ok((idx, name, transformer, indices))
                })
                .collect();

        // Check for errors and collect results
        let mut fitted_results: Vec<_> = Vec::with_capacity(results.len());
        for result in results {
            fitted_results.push(result?);
        }

        // Sort back to original order
        fitted_results.sort_by_key(|(idx, _, _, _)| *idx);

        // Build output slices and feature names
        let mut total_features = 0;
        let mut feature_names = Vec::new();
        self.output_slices = Vec::with_capacity(fitted_results.len());

        for (_, name, transformer, indices) in &fitted_results {
            let n_out = if indices.is_empty() {
                0
            } else {
                transformer.n_features_out().unwrap_or(indices.len())
            };
            let start = total_features;
            let end = total_features + n_out;
            self.output_slices.push((start, end));

            // Generate feature names
            if let Some(names) = transformer.get_feature_names_out(None) {
                for fname in names {
                    feature_names.push(format!("{}_{}", name, fname));
                }
            } else {
                for i in 0..n_out {
                    feature_names.push(format!("{}_feature_{}", name, i));
                }
            }
            total_features += n_out;
        }

        // Handle remainder passthrough
        if self.remainder == RemainderHandling::Passthrough {
            if let Some(ref remainder_indices) = self.remainder_indices {
                if !remainder_indices.is_empty() {
                    let start = total_features;
                    let end = total_features + remainder_indices.len();
                    self.remainder_slice = Some((start, end));

                    for &idx in remainder_indices {
                        feature_names.push(format!("remainder_x{}", idx));
                    }
                    total_features += remainder_indices.len();
                }
            }
        }

        // Update transformers with fitted versions
        for (idx, spec) in self.transformers.iter_mut().enumerate() {
            let (_, _, transformer, indices) = fitted_results.remove(
                fitted_results
                    .iter()
                    .position(|(i, _, _, _)| *i == idx)
                    .unwrap(),
            );
            spec.transformer = transformer;
            spec.resolved_indices = Some(indices);
        }

        self.n_features_out = Some(total_features);
        self.feature_names_out = Some(feature_names);
        self.fitted = true;

        Ok(())
    }

    /// Transform data through all column transformers and concatenate results.
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(FerroError::not_fitted("transform"));
        }

        self.validate_n_features(x)?;

        let n_samples = x.nrows();
        let n_features_out = self.n_features_out.unwrap();

        // Allocate output array
        let mut result = Array2::zeros((n_samples, n_features_out));

        // Transform through each transformer in parallel
        let transformed_parts: Vec<Result<(usize, Array2<f64>)>> = self
            .transformers
            .par_iter()
            .enumerate()
            .map(|(idx, spec)| {
                let indices = spec.resolved_indices.as_ref().unwrap();
                if indices.is_empty() {
                    Ok((idx, Array2::zeros((n_samples, 0))))
                } else {
                    let x_subset = Self::extract_columns(x, indices);
                    let transformed = spec.transformer.transform(&x_subset)?;
                    Ok((idx, transformed))
                }
            })
            .collect();

        // Check for errors and place results
        for part_result in transformed_parts {
            let (idx, transformed) = part_result?;
            let (start, end) = self.output_slices[idx];
            if end > start {
                result
                    .slice_mut(ndarray::s![.., start..end])
                    .assign(&transformed);
            }
        }

        // Handle remainder passthrough
        if let Some((start, end)) = self.remainder_slice {
            if let Some(ref remainder_indices) = self.remainder_indices {
                let remainder = Self::extract_columns(x, remainder_indices);
                result
                    .slice_mut(ndarray::s![.., start..end])
                    .assign(&remainder);
            }
        }

        Ok(result)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
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

impl PipelineTransformer for ColumnTransformer {
    fn search_space(&self) -> SearchSpace {
        self.search_space()
    }

    fn clone_boxed(&self) -> Box<dyn PipelineTransformer> {
        let cloned_transformers: Vec<_> = self
            .transformers
            .iter()
            .map(|spec| TransformerSpec {
                name: spec.name.clone(),
                transformer: spec.transformer.clone_boxed(),
                columns: spec.columns.clone(),
                resolved_indices: spec.resolved_indices.clone(),
            })
            .collect();

        Box::new(ColumnTransformer {
            transformers: cloned_transformers,
            remainder: self.remainder,
            fitted: self.fitted,
            n_features_in: self.n_features_in,
            n_features_out: self.n_features_out,
            feature_names_out: self.feature_names_out.clone(),
            remainder_indices: self.remainder_indices.clone(),
            output_slices: self.output_slices.clone(),
            remainder_slice: self.remainder_slice,
        })
    }

    fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()> {
        // Handle nested parameters (transformer__param)
        let parts: Vec<&str> = name.splitn(2, "__").collect();
        if parts.len() == 2 {
            let (transformer_name, param_name) = (parts[0], parts[1]);

            for spec in &mut self.transformers {
                if spec.name == transformer_name {
                    return spec.transformer.set_param(param_name, value);
                }
            }

            return Err(FerroError::invalid_input(format!(
                "Transformer '{}' not found",
                transformer_name
            )));
        }

        Err(FerroError::invalid_input(format!(
            "Parameter '{}' not supported. Use 'transformer__param' format for nested parameters.",
            name
        )))
    }

    fn name(&self) -> &str {
        "ColumnTransformer"
    }
}

/// Create a ColumnTransformer from a list of (name, transformer, columns) tuples.
///
/// A convenience function for creating column transformers.
///
/// # Arguments
///
/// * `transformers` - Vector of (name, transformer, columns) tuples
///
/// # Example
///
/// ```
/// use ferroml_core::pipeline::{make_column_transformer, ColumnSelector};
/// use ferroml_core::preprocessing::scalers::StandardScaler;
///
/// let ct = make_column_transformer(vec![
///     ("scaler".to_string(), Box::new(StandardScaler::new()), ColumnSelector::indices([0, 1])),
/// ]);
/// ```
pub fn make_column_transformer(
    transformers: Vec<(String, Box<dyn PipelineTransformer>, ColumnSelector)>,
) -> ColumnTransformer {
    let mut ct = ColumnTransformer::new();
    ct.transformers = transformers
        .into_iter()
        .map(|(name, transformer, columns)| TransformerSpec {
            name,
            transformer,
            columns,
            resolved_indices: None,
        })
        .collect();
    ct
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    // Test transformer that doubles values
    #[derive(Clone)]
    struct DoublingTransformer {
        fitted: bool,
        n_features: Option<usize>,
    }

    impl DoublingTransformer {
        fn new() -> Self {
            Self {
                fitted: false,
                n_features: None,
            }
        }
    }

    impl Transformer for DoublingTransformer {
        fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
            self.n_features = Some(x.ncols());
            self.fitted = true;
            Ok(())
        }

        fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
            if !self.fitted {
                return Err(FerroError::not_fitted("transform"));
            }
            Ok(x * 2.0)
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn get_feature_names_out(&self, _input_names: Option<&[String]>) -> Option<Vec<String>> {
            self.n_features
                .map(|n| (0..n).map(|i| format!("x{}", i)).collect())
        }

        fn n_features_in(&self) -> Option<usize> {
            self.n_features
        }

        fn n_features_out(&self) -> Option<usize> {
            self.n_features
        }
    }

    impl PipelineTransformer for DoublingTransformer {
        fn clone_boxed(&self) -> Box<dyn PipelineTransformer> {
            Box::new(self.clone())
        }

        fn name(&self) -> &str {
            "DoublingTransformer"
        }
    }

    // Test transformer that adds a constant
    #[derive(Clone)]
    struct AddConstantTransformer {
        constant: f64,
        fitted: bool,
        n_features: Option<usize>,
    }

    impl AddConstantTransformer {
        fn new(constant: f64) -> Self {
            Self {
                constant,
                fitted: false,
                n_features: None,
            }
        }
    }

    impl Transformer for AddConstantTransformer {
        fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
            self.n_features = Some(x.ncols());
            self.fitted = true;
            Ok(())
        }

        fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
            if !self.fitted {
                return Err(FerroError::not_fitted("transform"));
            }
            Ok(x + self.constant)
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn get_feature_names_out(&self, _input_names: Option<&[String]>) -> Option<Vec<String>> {
            self.n_features
                .map(|n| (0..n).map(|i| format!("x{}", i)).collect())
        }

        fn n_features_in(&self) -> Option<usize> {
            self.n_features
        }

        fn n_features_out(&self) -> Option<usize> {
            self.n_features
        }
    }

    impl PipelineTransformer for AddConstantTransformer {
        fn clone_boxed(&self) -> Box<dyn PipelineTransformer> {
            Box::new(self.clone())
        }

        fn name(&self) -> &str {
            "AddConstantTransformer"
        }

        fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()> {
            match name {
                "constant" => {
                    if let Some(v) = value.as_f64() {
                        self.constant = v;
                        Ok(())
                    } else {
                        Err(FerroError::invalid_input("constant must be a number"))
                    }
                }
                _ => Err(FerroError::invalid_input(format!(
                    "Unknown parameter: {}",
                    name
                ))),
            }
        }

        fn search_space(&self) -> SearchSpace {
            SearchSpace::new().float("constant", -10.0, 10.0)
        }
    }

    // Simple test model that just returns the mean of each row
    #[derive(Clone)]
    struct MeanModel {
        fitted: bool,
        n_features: Option<usize>,
    }

    impl MeanModel {
        fn new() -> Self {
            Self {
                fitted: false,
                n_features: None,
            }
        }
    }

    impl Model for MeanModel {
        fn fit(&mut self, x: &Array2<f64>, _y: &Array1<f64>) -> Result<()> {
            self.n_features = Some(x.ncols());
            self.fitted = true;
            Ok(())
        }

        fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
            if !self.fitted {
                return Err(FerroError::not_fitted("predict"));
            }
            Ok(x.mean_axis(ndarray::Axis(1)).unwrap())
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn n_features(&self) -> Option<usize> {
            self.n_features
        }
    }

    impl PipelineModel for MeanModel {
        fn clone_boxed(&self) -> Box<dyn PipelineModel> {
            Box::new(self.clone())
        }

        fn name(&self) -> &str {
            "MeanModel"
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = Pipeline::new()
            .add_transformer("doubler", DoublingTransformer::new())
            .add_transformer("adder", AddConstantTransformer::new(1.0))
            .add_model("model", MeanModel::new());

        assert_eq!(pipeline.n_steps(), 3);
        assert_eq!(pipeline.step_names(), vec!["doubler", "adder", "model"]);
        assert!(pipeline.has_model());
        assert!(!pipeline.is_fitted());
    }

    #[test]
    fn test_pipeline_fit_transform() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        let mut pipeline = Pipeline::new()
            .add_transformer("doubler", DoublingTransformer::new())
            .add_transformer("adder", AddConstantTransformer::new(1.0));

        pipeline.fit(&x, &y).unwrap();
        assert!(pipeline.is_fitted());

        let transformed = pipeline.transform(&x).unwrap();

        // Expected: (x * 2) + 1
        let expected = array![[3.0, 5.0], [7.0, 9.0], [11.0, 13.0]];
        assert_eq!(transformed, expected);
    }

    #[test]
    fn test_pipeline_predict() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        let mut pipeline = Pipeline::new()
            .add_transformer("doubler", DoublingTransformer::new())
            .add_model("model", MeanModel::new());

        pipeline.fit(&x, &y).unwrap();

        let predictions = pipeline.predict(&x).unwrap();

        // Doubled: [[2, 4], [6, 8], [10, 12]]
        // Mean per row: [3, 7, 11]
        let expected = array![3.0, 7.0, 11.0];
        assert_eq!(predictions, expected);
    }

    #[test]
    fn test_pipeline_search_space() {
        let pipeline = Pipeline::new()
            .add_transformer("adder", AddConstantTransformer::new(1.0))
            .add_model("model", MeanModel::new());

        let space = pipeline.search_space();
        assert!(space.parameters.contains_key("adder__constant"));
    }

    #[test]
    fn test_pipeline_set_params() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let mut pipeline =
            Pipeline::new().add_transformer("adder", AddConstantTransformer::new(1.0));

        pipeline.fit(&x, &y).unwrap();

        // Change the constant
        let mut params = HashMap::new();
        params.insert("adder__constant".to_string(), ParameterValue::Float(5.0));
        pipeline.set_params(&params).unwrap();

        // Pipeline should now be unfitted
        assert!(!pipeline.is_fitted());

        // Refit and check new transformation
        pipeline.fit(&x, &y).unwrap();
        let transformed = pipeline.transform(&x).unwrap();

        let expected = array![[6.0, 7.0], [8.0, 9.0]];
        assert_eq!(transformed, expected);
    }

    #[test]
    fn test_pipeline_caching() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let mut pipeline = Pipeline::new()
            .add_transformer("doubler", DoublingTransformer::new())
            .with_cache(CacheStrategy::Memory);

        pipeline.fit(&x, &y).unwrap();

        // First transform populates cache
        let result1 = pipeline.transform(&x).unwrap();

        // Second transform should use cache (same result)
        let result2 = pipeline.transform(&x).unwrap();

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_empty_pipeline_error() {
        let mut pipeline = Pipeline::new();
        let x = array![[1.0, 2.0]];
        let y = array![1.0];

        let result = pipeline.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_without_model_error() {
        let x = array![[1.0, 2.0]];
        let y = array![1.0];

        let mut pipeline = Pipeline::new().add_transformer("doubler", DoublingTransformer::new());

        pipeline.fit(&x, &y).unwrap();

        let result = pipeline.predict(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_before_fit_error() {
        let x = array![[1.0, 2.0]];

        let pipeline = Pipeline::new()
            .add_transformer("doubler", DoublingTransformer::new())
            .add_model("model", MeanModel::new());

        let result = pipeline.predict(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![1.0, 2.0];
        let x_test = array![[1.0, 2.0, 3.0]]; // Wrong number of features

        let mut pipeline = Pipeline::new().add_transformer("doubler", DoublingTransformer::new());

        pipeline.fit(&x_train, &y_train).unwrap();

        let result = pipeline.transform(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_param_name_error() {
        let mut pipeline =
            Pipeline::new().add_transformer("adder", AddConstantTransformer::new(1.0));

        let mut params = HashMap::new();
        params.insert("invalid".to_string(), ParameterValue::Float(5.0)); // Missing __

        let result = pipeline.set_params(&params);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_step_error() {
        let mut pipeline =
            Pipeline::new().add_transformer("adder", AddConstantTransformer::new(1.0));

        let mut params = HashMap::new();
        params.insert("unknown__constant".to_string(), ParameterValue::Float(5.0));

        let result = pipeline.set_params(&params);
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "already exists")]
    fn test_duplicate_step_name_panics() {
        let _ = Pipeline::new()
            .add_transformer("step", DoublingTransformer::new())
            .add_transformer("step", DoublingTransformer::new());
    }

    #[test]
    #[should_panic(expected = "Cannot add transformer after model")]
    fn test_transformer_after_model_panics() {
        let _ = Pipeline::new()
            .add_model("model", MeanModel::new())
            .add_transformer("step", DoublingTransformer::new());
    }

    #[test]
    fn test_make_pipeline() {
        let transformers: Vec<Box<dyn PipelineTransformer>> = vec![
            Box::new(DoublingTransformer::new()),
            Box::new(AddConstantTransformer::new(1.0)),
        ];
        let model: Option<Box<dyn PipelineModel>> = Some(Box::new(MeanModel::new()));

        let pipeline = make_pipeline(transformers, model);

        assert_eq!(pipeline.n_steps(), 3);
        assert_eq!(pipeline.step_names(), vec!["step_0", "step_1", "model"]);
    }

    #[test]
    fn test_get_params() {
        let pipeline = Pipeline::new()
            .add_transformer("doubler", DoublingTransformer::new())
            .add_model("model", MeanModel::new());

        let params = pipeline.get_params();

        assert_eq!(
            params.get("doubler__type"),
            Some(&"DoublingTransformer".to_string())
        );
        assert_eq!(params.get("model__type"), Some(&"MeanModel".to_string()));
    }

    #[test]
    fn test_fit_transform_combined() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let mut pipeline = Pipeline::new().add_transformer("doubler", DoublingTransformer::new());

        let transformed = pipeline.fit_transform(&x, &y).unwrap();

        let expected = array![[2.0, 4.0], [6.0, 8.0]];
        assert_eq!(transformed, expected);
        assert!(pipeline.is_fitted());
    }

    // =============================================================================
    // FeatureUnion Tests
    // =============================================================================

    #[test]
    fn test_feature_union_creation() {
        let union = FeatureUnion::new()
            .add_transformer("doubler", DoublingTransformer::new())
            .add_transformer("adder", AddConstantTransformer::new(1.0));

        assert_eq!(union.n_transformers(), 2);
        assert_eq!(union.transformer_names(), vec!["doubler", "adder"]);
        assert!(!union.is_fitted());
    }

    #[test]
    fn test_feature_union_fit_transform() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let mut union = FeatureUnion::new()
            .add_transformer("doubler", DoublingTransformer::new())
            .add_transformer("adder", AddConstantTransformer::new(10.0));

        union.fit(&x).unwrap();
        assert!(union.is_fitted());
        assert_eq!(union.n_features_in(), Some(2));
        assert_eq!(union.n_features_out(), Some(4)); // 2 from doubler + 2 from adder

        let transformed = union.transform(&x).unwrap();

        // Expected: [doubled | added]
        // Row 0: [2, 4, 11, 12]
        // Row 1: [6, 8, 13, 14]
        // Row 2: [10, 12, 15, 16]
        assert_eq!(transformed.shape(), &[3, 4]);
        assert_eq!(transformed[[0, 0]], 2.0); // doubled
        assert_eq!(transformed[[0, 1]], 4.0); // doubled
        assert_eq!(transformed[[0, 2]], 11.0); // added (1 + 10)
        assert_eq!(transformed[[0, 3]], 12.0); // added (2 + 10)
    }

    #[test]
    fn test_feature_union_fit_transform_method() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let mut union = FeatureUnion::new().add_transformer("doubler", DoublingTransformer::new());

        let transformed = union.fit_transform(&x).unwrap();

        assert!(union.is_fitted());
        assert_eq!(transformed, array![[2.0, 4.0], [6.0, 8.0]]);
    }

    #[test]
    fn test_feature_union_search_space() {
        let union = FeatureUnion::new()
            .add_transformer("adder1", AddConstantTransformer::new(1.0))
            .add_transformer("adder2", AddConstantTransformer::new(2.0));

        let space = union.search_space();

        // Both adders should contribute their "constant" parameter
        assert!(space.parameters.contains_key("adder1__constant"));
        assert!(space.parameters.contains_key("adder2__constant"));
    }

    #[test]
    fn test_feature_union_set_params() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let mut union =
            FeatureUnion::new().add_transformer("adder", AddConstantTransformer::new(1.0));

        union.fit(&x).unwrap();

        // Change the constant
        let mut params = HashMap::new();
        params.insert("adder__constant".to_string(), ParameterValue::Float(100.0));
        union.set_params(&params).unwrap();

        // Union should now be unfitted
        assert!(!union.is_fitted());

        // Refit and check new transformation
        union.fit(&x).unwrap();
        let transformed = union.transform(&x).unwrap();

        // x + 100
        let expected = array![[101.0, 102.0], [103.0, 104.0]];
        assert_eq!(transformed, expected);
    }

    #[test]
    fn test_feature_union_weights() {
        let x = array![[1.0], [2.0]];

        let mut weights = HashMap::new();
        weights.insert("doubler".to_string(), 0.5);
        weights.insert("adder".to_string(), 2.0);

        let mut union = FeatureUnion::new()
            .add_transformer("doubler", DoublingTransformer::new())
            .add_transformer("adder", AddConstantTransformer::new(10.0))
            .with_weights(weights);

        let transformed = union.fit_transform(&x).unwrap();

        // doubler: [2, 4] * 0.5 = [1, 2]
        // adder: [11, 12] * 2.0 = [22, 24]
        assert_eq!(transformed[[0, 0]], 1.0); // 2 * 0.5
        assert_eq!(transformed[[1, 0]], 2.0); // 4 * 0.5
        assert_eq!(transformed[[0, 1]], 22.0); // 11 * 2.0
        assert_eq!(transformed[[1, 1]], 24.0); // 12 * 2.0
    }

    #[test]
    fn test_feature_union_get_params() {
        let union = FeatureUnion::new()
            .add_transformer("doubler", DoublingTransformer::new())
            .add_transformer("adder", AddConstantTransformer::new(1.0));

        let params = union.get_params();

        assert_eq!(
            params.get("doubler__type"),
            Some(&"DoublingTransformer".to_string())
        );
        assert_eq!(
            params.get("adder__type"),
            Some(&"AddConstantTransformer".to_string())
        );
    }

    #[test]
    fn test_feature_union_feature_names() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let mut union = FeatureUnion::new()
            .add_transformer("doubler", DoublingTransformer::new())
            .add_transformer("adder", AddConstantTransformer::new(1.0));

        union.fit(&x).unwrap();

        let names = union.get_feature_names_out(None).unwrap();
        assert_eq!(names.len(), 4);
        // Feature names should be prefixed with transformer name
        assert!(names[0].starts_with("doubler_"));
        assert!(names[2].starts_with("adder_"));
    }

    #[test]
    fn test_feature_union_empty_error() {
        let mut union = FeatureUnion::new();
        let x = array![[1.0, 2.0]];

        let result = union.fit(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_union_transform_before_fit_error() {
        let union = FeatureUnion::new().add_transformer("doubler", DoublingTransformer::new());

        let x = array![[1.0, 2.0]];
        let result = union.transform(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_union_shape_mismatch_error() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let x_test = array![[1.0, 2.0, 3.0]]; // Wrong number of features

        let mut union = FeatureUnion::new().add_transformer("doubler", DoublingTransformer::new());

        union.fit(&x_train).unwrap();

        let result = union.transform(&x_test);
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "already exists")]
    fn test_feature_union_duplicate_name_panics() {
        let _ = FeatureUnion::new()
            .add_transformer("step", DoublingTransformer::new())
            .add_transformer("step", DoublingTransformer::new());
    }

    #[test]
    fn test_feature_union_set_param_via_trait() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let mut union =
            FeatureUnion::new().add_transformer("adder", AddConstantTransformer::new(1.0));

        union.fit(&x).unwrap();

        // Use the PipelineTransformer trait method
        let transformer: &mut dyn PipelineTransformer = &mut union;
        transformer
            .set_param("adder__constant", &ParameterValue::Float(50.0))
            .unwrap();

        union.fit(&x).unwrap();
        let transformed = union.transform(&x).unwrap();

        let expected = array![[51.0, 52.0], [53.0, 54.0]];
        assert_eq!(transformed, expected);
    }

    #[test]
    fn test_feature_union_clone_boxed() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let mut union = FeatureUnion::new().add_transformer("doubler", DoublingTransformer::new());

        union.fit(&x).unwrap();

        // Clone via PipelineTransformer trait
        let cloned = union.clone_boxed();

        // Both should transform the same way
        let original_result = union.transform(&x).unwrap();
        let cloned_result = cloned.transform(&x).unwrap();

        assert_eq!(original_result, cloned_result);
    }

    #[test]
    fn test_feature_union_unknown_transformer_error() {
        let mut union =
            FeatureUnion::new().add_transformer("adder", AddConstantTransformer::new(1.0));

        let mut params = HashMap::new();
        params.insert("unknown__constant".to_string(), ParameterValue::Float(5.0));

        let result = union.set_params(&params);
        assert!(result.is_err());
    }

    #[test]
    fn test_make_feature_union() {
        let transformers: Vec<(String, Box<dyn PipelineTransformer>)> = vec![
            ("doubler".to_string(), Box::new(DoublingTransformer::new())),
            (
                "adder".to_string(),
                Box::new(AddConstantTransformer::new(1.0)),
            ),
        ];

        let union = make_feature_union(transformers);

        assert_eq!(union.n_transformers(), 2);
        assert_eq!(union.transformer_names(), vec!["doubler", "adder"]);
    }

    #[test]
    fn test_feature_union_in_pipeline() {
        // Test that FeatureUnion can be used as a step in a Pipeline
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        let union = FeatureUnion::new()
            .add_transformer("doubler", DoublingTransformer::new())
            .add_transformer("adder", AddConstantTransformer::new(10.0));

        let mut pipeline = Pipeline::new()
            .add_transformer("union", union)
            .add_model("model", MeanModel::new());

        pipeline.fit(&x, &y).unwrap();

        let predictions = pipeline.predict(&x).unwrap();

        // Union output: [doubled | added]
        // Row 0: [2, 4, 11, 12] -> mean = 7.25
        // Row 1: [6, 8, 13, 14] -> mean = 10.25
        // Row 2: [10, 12, 15, 16] -> mean = 13.25
        assert!((predictions[0] - 7.25).abs() < 1e-10);
        assert!((predictions[1] - 10.25).abs() < 1e-10);
        assert!((predictions[2] - 13.25).abs() < 1e-10);
    }

    // =============================================================================
    // ColumnTransformer Tests
    // =============================================================================

    #[test]
    fn test_column_transformer_creation() {
        let ct = ColumnTransformer::new()
            .add_transformer(
                "doubler",
                DoublingTransformer::new(),
                ColumnSelector::indices([0]),
            )
            .add_transformer(
                "adder",
                AddConstantTransformer::new(5.0),
                ColumnSelector::indices([1]),
            );

        assert_eq!(ct.n_transformers(), 2);
        assert_eq!(ct.transformer_names(), vec!["doubler", "adder"]);
        assert!(!ct.is_fitted());
    }

    #[test]
    fn test_column_transformer_fit_transform() {
        // Input: 4 features, apply different transformers to different columns
        let x = array![
            [1.0, 10.0, 100.0, 1000.0],
            [2.0, 20.0, 200.0, 2000.0],
            [3.0, 30.0, 300.0, 3000.0]
        ];

        let mut ct = ColumnTransformer::new()
            .add_transformer(
                "doubler",
                DoublingTransformer::new(),
                ColumnSelector::indices([0, 1]),
            ) // cols 0,1 -> doubled
            .add_transformer(
                "adder",
                AddConstantTransformer::new(1.0),
                ColumnSelector::indices([2, 3]),
            ); // cols 2,3 -> +1

        ct.fit(&x).unwrap();
        assert!(ct.is_fitted());
        assert_eq!(ct.n_features_in(), Some(4));
        assert_eq!(ct.n_features_out(), Some(4)); // 2 from doubler + 2 from adder

        let transformed = ct.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[3, 4]);
        // First two columns are doubled
        assert_eq!(transformed[[0, 0]], 2.0); // 1 * 2
        assert_eq!(transformed[[0, 1]], 20.0); // 10 * 2
                                               // Last two columns have +1 added
        assert_eq!(transformed[[0, 2]], 101.0); // 100 + 1
        assert_eq!(transformed[[0, 3]], 1001.0); // 1000 + 1
    }

    #[test]
    fn test_column_transformer_with_mask() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Use mask: select first and last columns
        let mut ct = ColumnTransformer::new().add_transformer(
            "doubler",
            DoublingTransformer::new(),
            ColumnSelector::mask([true, false, true]),
        );

        let transformed = ct.fit_transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[2, 2]);
        assert_eq!(transformed[[0, 0]], 2.0); // 1 * 2
        assert_eq!(transformed[[0, 1]], 6.0); // 3 * 2
    }

    #[test]
    fn test_column_transformer_with_remainder_drop() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Only transform column 0, drop remainder
        let mut ct = ColumnTransformer::new()
            .add_transformer(
                "doubler",
                DoublingTransformer::new(),
                ColumnSelector::indices([0]),
            )
            .with_remainder(RemainderHandling::Drop);

        let transformed = ct.fit_transform(&x).unwrap();

        // Only column 0 is kept (and doubled), columns 1 and 2 are dropped
        assert_eq!(transformed.shape(), &[2, 1]);
        assert_eq!(transformed[[0, 0]], 2.0); // 1 * 2
        assert_eq!(transformed[[1, 0]], 8.0); // 4 * 2
    }

    #[test]
    fn test_column_transformer_with_remainder_passthrough() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Transform column 0, passthrough remainder (columns 1 and 2)
        let mut ct = ColumnTransformer::new()
            .add_transformer(
                "doubler",
                DoublingTransformer::new(),
                ColumnSelector::indices([0]),
            )
            .with_remainder(RemainderHandling::Passthrough);

        let transformed = ct.fit_transform(&x).unwrap();

        // Output: [doubled_col0, col1, col2]
        assert_eq!(transformed.shape(), &[2, 3]);
        assert_eq!(transformed[[0, 0]], 2.0); // 1 * 2 (doubled)
        assert_eq!(transformed[[0, 1]], 2.0); // passthrough
        assert_eq!(transformed[[0, 2]], 3.0); // passthrough
    }

    #[test]
    fn test_column_transformer_search_space() {
        let ct = ColumnTransformer::new()
            .add_transformer(
                "adder1",
                AddConstantTransformer::new(1.0),
                ColumnSelector::indices([0]),
            )
            .add_transformer(
                "adder2",
                AddConstantTransformer::new(2.0),
                ColumnSelector::indices([1]),
            );

        let space = ct.search_space();

        assert!(space.parameters.contains_key("adder1__constant"));
        assert!(space.parameters.contains_key("adder2__constant"));
    }

    #[test]
    fn test_column_transformer_set_params() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let mut ct = ColumnTransformer::new().add_transformer(
            "adder",
            AddConstantTransformer::new(1.0),
            ColumnSelector::indices([0, 1]),
        );

        ct.fit(&x).unwrap();

        // Change the constant
        let mut params = HashMap::new();
        params.insert("adder__constant".to_string(), ParameterValue::Float(100.0));
        ct.set_params(&params).unwrap();

        // Should now be unfitted
        assert!(!ct.is_fitted());

        // Refit and check new transformation
        ct.fit(&x).unwrap();
        let transformed = ct.transform(&x).unwrap();

        // x + 100
        assert_eq!(transformed[[0, 0]], 101.0);
        assert_eq!(transformed[[0, 1]], 102.0);
    }

    #[test]
    fn test_column_transformer_get_params() {
        let ct = ColumnTransformer::new()
            .add_transformer(
                "doubler",
                DoublingTransformer::new(),
                ColumnSelector::indices([0]),
            )
            .add_transformer(
                "adder",
                AddConstantTransformer::new(1.0),
                ColumnSelector::indices([1]),
            );

        let params = ct.get_params();

        assert_eq!(
            params.get("doubler__type"),
            Some(&"DoublingTransformer".to_string())
        );
        assert_eq!(
            params.get("adder__type"),
            Some(&"AddConstantTransformer".to_string())
        );
    }

    #[test]
    fn test_column_transformer_feature_names() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let mut ct = ColumnTransformer::new()
            .add_transformer(
                "first",
                DoublingTransformer::new(),
                ColumnSelector::indices([0]),
            )
            .add_transformer(
                "second",
                AddConstantTransformer::new(1.0),
                ColumnSelector::indices([1, 2]),
            );

        ct.fit(&x).unwrap();

        let names = ct.get_feature_names_out(None).unwrap();
        assert_eq!(names.len(), 3);
        assert!(names[0].starts_with("first_"));
        assert!(names[1].starts_with("second_"));
        assert!(names[2].starts_with("second_"));
    }

    #[test]
    fn test_column_transformer_empty_error() {
        let mut ct = ColumnTransformer::new();
        let x = array![[1.0, 2.0]];

        let result = ct.fit(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_column_transformer_transform_before_fit_error() {
        let ct = ColumnTransformer::new().add_transformer(
            "doubler",
            DoublingTransformer::new(),
            ColumnSelector::indices([0]),
        );

        let x = array![[1.0, 2.0]];
        let result = ct.transform(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_column_transformer_shape_mismatch_error() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let x_test = array![[1.0, 2.0, 3.0]]; // Wrong number of features

        let mut ct = ColumnTransformer::new().add_transformer(
            "doubler",
            DoublingTransformer::new(),
            ColumnSelector::indices([0]),
        );

        ct.fit(&x_train).unwrap();

        let result = ct.transform(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_column_transformer_invalid_column_index() {
        let x = array![[1.0, 2.0]];

        let mut ct = ColumnTransformer::new().add_transformer(
            "doubler",
            DoublingTransformer::new(),
            ColumnSelector::indices([5]),
        ); // Out of range

        let result = ct.fit(&x);
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "already exists")]
    fn test_column_transformer_duplicate_name_panics() {
        let _ = ColumnTransformer::new()
            .add_transformer(
                "step",
                DoublingTransformer::new(),
                ColumnSelector::indices([0]),
            )
            .add_transformer(
                "step",
                DoublingTransformer::new(),
                ColumnSelector::indices([1]),
            );
    }

    #[test]
    fn test_column_transformer_set_param_via_trait() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let mut ct = ColumnTransformer::new().add_transformer(
            "adder",
            AddConstantTransformer::new(1.0),
            ColumnSelector::indices([0, 1]),
        );

        ct.fit(&x).unwrap();

        // Use the PipelineTransformer trait method
        let transformer: &mut dyn PipelineTransformer = &mut ct;
        transformer
            .set_param("adder__constant", &ParameterValue::Float(50.0))
            .unwrap();

        ct.fit(&x).unwrap();
        let transformed = ct.transform(&x).unwrap();

        assert_eq!(transformed[[0, 0]], 51.0);
        assert_eq!(transformed[[0, 1]], 52.0);
    }

    #[test]
    fn test_column_transformer_clone_boxed() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let mut ct = ColumnTransformer::new().add_transformer(
            "doubler",
            DoublingTransformer::new(),
            ColumnSelector::indices([0, 1]),
        );

        ct.fit(&x).unwrap();

        // Clone via PipelineTransformer trait
        let cloned = ct.clone_boxed();

        // Both should transform the same way
        let original_result = ct.transform(&x).unwrap();
        let cloned_result = cloned.transform(&x).unwrap();

        assert_eq!(original_result, cloned_result);
    }

    #[test]
    fn test_column_transformer_in_pipeline() {
        // Test that ColumnTransformer can be used as a step in a Pipeline
        let x = array![[1.0, 2.0, 100.0], [3.0, 4.0, 200.0], [5.0, 6.0, 300.0]];
        let y = array![1.0, 2.0, 3.0];

        let ct = ColumnTransformer::new()
            .add_transformer(
                "doubler",
                DoublingTransformer::new(),
                ColumnSelector::indices([0, 1]),
            )
            .add_transformer(
                "adder",
                AddConstantTransformer::new(1.0),
                ColumnSelector::indices([2]),
            );

        let mut pipeline = Pipeline::new()
            .add_transformer("ct", ct)
            .add_model("model", MeanModel::new());

        pipeline.fit(&x, &y).unwrap();

        let predictions = pipeline.predict(&x).unwrap();

        // CT output for row 0: [2, 4, 101] -> mean = 35.666...
        assert!((predictions[0] - (2.0 + 4.0 + 101.0) / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_make_column_transformer() {
        let transformers: Vec<(String, Box<dyn PipelineTransformer>, ColumnSelector)> = vec![
            (
                "doubler".to_string(),
                Box::new(DoublingTransformer::new()),
                ColumnSelector::indices([0]),
            ),
            (
                "adder".to_string(),
                Box::new(AddConstantTransformer::new(1.0)),
                ColumnSelector::indices([1]),
            ),
        ];

        let ct = make_column_transformer(transformers);

        assert_eq!(ct.n_transformers(), 2);
        assert_eq!(ct.transformer_names(), vec!["doubler", "adder"]);
    }

    #[test]
    fn test_column_selector_all() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let mut ct = ColumnTransformer::new().add_transformer(
            "doubler",
            DoublingTransformer::new(),
            ColumnSelector::All,
        );

        let transformed = ct.fit_transform(&x).unwrap();

        // All columns are doubled
        assert_eq!(transformed.shape(), &[2, 3]);
        assert_eq!(transformed[[0, 0]], 2.0);
        assert_eq!(transformed[[0, 1]], 4.0);
        assert_eq!(transformed[[0, 2]], 6.0);
    }

    #[test]
    fn test_column_transformer_overlapping_columns() {
        // Test that overlapping column selections work (same column can go to multiple transformers)
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let mut ct = ColumnTransformer::new()
            .add_transformer(
                "first",
                DoublingTransformer::new(),
                ColumnSelector::indices([0]),
            )
            .add_transformer(
                "second",
                AddConstantTransformer::new(10.0),
                ColumnSelector::indices([0]),
            );

        let transformed = ct.fit_transform(&x).unwrap();

        // Output: [doubled_col0, added_col0]
        assert_eq!(transformed.shape(), &[2, 2]);
        assert_eq!(transformed[[0, 0]], 2.0); // 1 * 2
        assert_eq!(transformed[[0, 1]], 11.0); // 1 + 10
    }
}

//! Automatic Preprocessing Selection for AutoML
//!
//! This module provides automatic selection and configuration of preprocessing steps
//! based on data characteristics and algorithm requirements. It builds preprocessing
//! pipelines automatically for each algorithm in the AutoML portfolio.
//!
//! # Features
//!
//! - **Data-Aware Selection**: Automatically detects missing values, categorical features,
//!   scale differences, and class imbalance to select appropriate preprocessors.
//!
//! - **Algorithm-Specific Pipelines**: Builds custom preprocessing pipelines based on each
//!   algorithm's requirements (scaling for SVM, non-negativity for MultinomialNB, etc.).
//!
//! - **Strategy-Based Configuration**: Multiple strategies (Auto, Conservative, Thorough)
//!   to balance preprocessing thoroughness with computational cost.
//!
//! # Example
//!
//! ```ignore
//! use ferroml_core::automl::{
//!     PreprocessingSelector, PreprocessingConfig, PreprocessingStrategy,
//!     DataCharacteristics, PreprocessingRequirement,
//! };
//!
//! // Create a selector with automatic strategy
//! let config = PreprocessingConfig::new()
//!     .with_strategy(PreprocessingStrategy::Auto)
//!     .with_handle_imbalance(true);
//!
//! let selector = PreprocessingSelector::new(config);
//!
//! // Analyze data characteristics
//! let chars = DataCharacteristics::from_data(&x, &y);
//!
//! // Build preprocessing pipeline for specific requirements
//! let requirements = vec![
//!     PreprocessingRequirement::HandleMissing,
//!     PreprocessingRequirement::Scaling,
//! ];
//!
//! let pipeline_spec = selector.build_pipeline(&chars, &requirements);
//! ```

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{DataCharacteristics, PreprocessingRequirement};

/// Strategy for automatic preprocessing selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PreprocessingStrategy {
    /// Automatically choose based on data characteristics (default)
    #[default]
    Auto,
    /// Minimal preprocessing - only what's strictly required
    Conservative,
    /// Standard preprocessing with sensible defaults
    Standard,
    /// Thorough preprocessing - all beneficial transformations
    Thorough,
    /// No preprocessing - pass data through unchanged
    Passthrough,
}

/// Configuration for automatic preprocessing selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Strategy for selecting preprocessing steps
    pub strategy: PreprocessingStrategy,
    /// Whether to handle class imbalance (for classification)
    pub handle_imbalance: bool,
    /// Imbalance ratio threshold to trigger resampling
    pub imbalance_threshold: f64,
    /// Preferred scaler type
    pub scaler_type: ScalerType,
    /// Preferred imputation strategy
    pub imputation_strategy: ImputationStrategy,
    /// Preferred categorical encoding
    pub encoding_type: EncodingType,
    /// Whether to apply dimensionality reduction for high-dimensional data
    pub reduce_dimensions: bool,
    /// Feature/sample ratio threshold for dimensionality reduction
    pub dimension_reduction_threshold: f64,
    /// Variance threshold for feature selection
    pub variance_threshold: f64,
    /// Whether to use power transforms for skewed features
    pub use_power_transform: bool,
    /// Skewness threshold to apply power transform
    pub skewness_threshold: f64,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            strategy: PreprocessingStrategy::Auto,
            handle_imbalance: true,
            imbalance_threshold: 3.0, // Ratio of 3:1 or higher
            scaler_type: ScalerType::Standard,
            imputation_strategy: ImputationStrategy::Median,
            encoding_type: EncodingType::OneHot,
            reduce_dimensions: true,
            dimension_reduction_threshold: 0.5, // Features/samples > 0.5
            variance_threshold: 0.0,            // No variance filtering by default
            use_power_transform: false,
            skewness_threshold: 2.0,
        }
    }
}

impl PreprocessingConfig {
    /// Create a new default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the preprocessing strategy
    pub fn with_strategy(mut self, strategy: PreprocessingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set whether to handle class imbalance
    pub fn with_handle_imbalance(mut self, handle: bool) -> Self {
        self.handle_imbalance = handle;
        self
    }

    /// Set the imbalance threshold
    pub fn with_imbalance_threshold(mut self, threshold: f64) -> Self {
        self.imbalance_threshold = threshold;
        self
    }

    /// Set the scaler type
    pub fn with_scaler(mut self, scaler: ScalerType) -> Self {
        self.scaler_type = scaler;
        self
    }

    /// Set the imputation strategy
    pub fn with_imputation(mut self, strategy: ImputationStrategy) -> Self {
        self.imputation_strategy = strategy;
        self
    }

    /// Set the encoding type
    pub fn with_encoding(mut self, encoding: EncodingType) -> Self {
        self.encoding_type = encoding;
        self
    }

    /// Set whether to reduce dimensions
    pub fn with_dimension_reduction(mut self, reduce: bool) -> Self {
        self.reduce_dimensions = reduce;
        self
    }

    /// Set the dimension reduction threshold
    pub fn with_dimension_reduction_threshold(mut self, threshold: f64) -> Self {
        self.dimension_reduction_threshold = threshold;
        self
    }

    /// Set the variance threshold for feature filtering
    pub fn with_variance_threshold(mut self, threshold: f64) -> Self {
        self.variance_threshold = threshold;
        self
    }

    /// Set whether to use power transforms
    pub fn with_power_transform(mut self, use_power: bool) -> Self {
        self.use_power_transform = use_power;
        self
    }

    /// Set the skewness threshold for power transforms
    pub fn with_skewness_threshold(mut self, threshold: f64) -> Self {
        self.skewness_threshold = threshold;
        self
    }
}

/// Type of scaler to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ScalerType {
    /// StandardScaler (z-score normalization)
    #[default]
    Standard,
    /// MinMaxScaler (scale to [0, 1])
    MinMax,
    /// RobustScaler (using median and IQR)
    Robust,
    /// MaxAbsScaler (scale by max absolute value)
    MaxAbs,
    /// No scaling
    None,
}

/// Strategy for missing value imputation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ImputationStrategy {
    /// Use mean of each feature
    Mean,
    /// Use median of each feature (default, robust to outliers)
    #[default]
    Median,
    /// Use mode (most frequent value)
    Mode,
    /// Use a constant value (typically 0)
    Constant,
    /// Use KNN imputation
    KNN,
}

/// Type of categorical encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum EncodingType {
    /// One-hot encoding (creates binary columns)
    #[default]
    OneHot,
    /// Ordinal encoding (assigns integers)
    Ordinal,
    /// Target encoding (uses target statistics)
    Target,
}

/// A specification for a preprocessing step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingStepSpec {
    /// Name of the preprocessing step
    pub name: String,
    /// Type of the step
    pub step_type: PreprocessingStepType,
    /// Configuration parameters for the step
    pub params: HashMap<String, StepParam>,
    /// Column indices this step applies to (None = all columns)
    pub column_indices: Option<Vec<usize>>,
}

/// Type of preprocessing step
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreprocessingStepType {
    /// Simple imputation (mean, median, mode, constant)
    SimpleImputer,
    /// K-nearest neighbors imputation
    KNNImputer,
    /// Standard scaler (z-score normalization)
    StandardScaler,
    /// Min-max scaler (scale to [0, 1])
    MinMaxScaler,
    /// Robust scaler (using median and IQR)
    RobustScaler,
    /// Max absolute scaler
    MaxAbsScaler,
    /// One-hot encoding for categorical features
    OneHotEncoder,
    /// Ordinal encoding for categorical features
    OrdinalEncoder,
    /// Target encoding for categorical features
    TargetEncoder,
    /// Power transformation (Box-Cox, Yeo-Johnson)
    PowerTransformer,
    /// Quantile transformation
    QuantileTransformer,
    /// Variance threshold filter
    VarianceThreshold,
    /// Select K best features by score
    SelectKBest,
    /// Principal Component Analysis
    PCA,
    /// SMOTE oversampling
    SMOTE,
    /// Random undersampling
    RandomUnderSampler,
}

/// Parameter value for a preprocessing step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepParam {
    /// Integer parameter
    Int(i64),
    /// Floating-point parameter
    Float(f64),
    /// String parameter
    String(String),
    /// Boolean parameter
    Bool(bool),
    /// List of integers parameter
    IntList(Vec<i64>),
}

impl PreprocessingStepSpec {
    /// Create a new step specification
    pub fn new(name: impl Into<String>, step_type: PreprocessingStepType) -> Self {
        Self {
            name: name.into(),
            step_type,
            params: HashMap::new(),
            column_indices: None,
        }
    }

    /// Add a parameter
    pub fn with_param(mut self, name: impl Into<String>, value: StepParam) -> Self {
        self.params.insert(name.into(), value);
        self
    }

    /// Set column indices
    pub fn with_columns(mut self, indices: Vec<usize>) -> Self {
        self.column_indices = Some(indices);
        self
    }
}

/// Complete preprocessing pipeline specification
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreprocessingPipelineSpec {
    /// Ordered list of preprocessing steps
    pub steps: Vec<PreprocessingStepSpec>,
    /// Detected data characteristics that influenced the pipeline
    pub detected_characteristics: DetectedCharacteristics,
}

/// Characteristics detected during preprocessing analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DetectedCharacteristics {
    /// Indices of features with missing values
    pub missing_feature_indices: Vec<usize>,
    /// Percentage of missing values per feature
    pub missing_percentages: Vec<f64>,
    /// Indices of categorical features (if detected)
    pub categorical_feature_indices: Vec<usize>,
    /// Indices of features with high variance relative to others
    pub high_variance_indices: Vec<usize>,
    /// Indices of features with low variance (near constant)
    pub low_variance_indices: Vec<usize>,
    /// Indices of skewed features
    pub skewed_feature_indices: Vec<usize>,
    /// Skewness values for each feature
    pub feature_skewness: Vec<f64>,
    /// Whether the data has negative values
    pub has_negative_values: bool,
    /// Whether scaling is recommended
    pub scaling_recommended: bool,
    /// Whether class imbalance was detected
    pub class_imbalance_detected: bool,
    /// Class distribution (class -> count)
    pub class_distribution: HashMap<i64, usize>,
}

/// Automatic preprocessing selector
///
/// Analyzes data characteristics and builds appropriate preprocessing pipelines
/// based on algorithm requirements and configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingSelector {
    /// Configuration for preprocessing selection
    pub config: PreprocessingConfig,
}

impl Default for PreprocessingSelector {
    fn default() -> Self {
        Self::new(PreprocessingConfig::default())
    }
}

impl PreprocessingSelector {
    /// Create a new preprocessing selector with the given configuration
    pub fn new(config: PreprocessingConfig) -> Self {
        Self { config }
    }

    /// Create a selector with automatic strategy
    pub fn auto() -> Self {
        Self::new(PreprocessingConfig::new().with_strategy(PreprocessingStrategy::Auto))
    }

    /// Create a selector with conservative strategy
    pub fn conservative() -> Self {
        Self::new(PreprocessingConfig::new().with_strategy(PreprocessingStrategy::Conservative))
    }

    /// Create a selector with thorough strategy
    pub fn thorough() -> Self {
        Self::new(PreprocessingConfig::new().with_strategy(PreprocessingStrategy::Thorough))
    }

    /// Analyze data characteristics beyond what DataCharacteristics provides
    pub fn analyze_data(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
    ) -> DetectedCharacteristics {
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Detect missing values
        let mut missing_feature_indices = Vec::new();
        let mut missing_percentages = vec![0.0; n_features];

        for j in 0..n_features {
            let col = x.column(j);
            let missing_count = col.iter().filter(|v| v.is_nan()).count();
            let pct = missing_count as f64 / n_samples as f64;
            missing_percentages[j] = pct;
            if missing_count > 0 {
                missing_feature_indices.push(j);
            }
        }

        // Compute variance for each feature
        let mut variances = Vec::with_capacity(n_features);
        let mut has_negative = false;

        for j in 0..n_features {
            let col = x.column(j);
            let valid_vals: Vec<f64> = col.iter().copied().filter(|v| !v.is_nan()).collect();

            if valid_vals.is_empty() {
                variances.push(0.0);
                continue;
            }

            let mean = valid_vals.iter().sum::<f64>() / valid_vals.len() as f64;
            let var = valid_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / valid_vals.len() as f64;
            variances.push(var);

            if valid_vals.iter().any(|&v| v < 0.0) {
                has_negative = true;
            }
        }

        // Find low/high variance features
        let max_var = variances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let low_variance_indices: Vec<usize> = variances
            .iter()
            .enumerate()
            .filter(|(_, &v)| v < self.config.variance_threshold || v < 1e-10)
            .map(|(i, _)| i)
            .collect();

        let high_variance_indices: Vec<usize> = variances
            .iter()
            .enumerate()
            .filter(|(_, &v)| max_var > 0.0 && v / max_var > 100.0)
            .map(|(i, _)| i)
            .collect();

        // Compute skewness for each feature
        let mut feature_skewness = Vec::with_capacity(n_features);
        let mut skewed_feature_indices = Vec::new();

        for j in 0..n_features {
            let col = x.column(j);
            let valid_vals: Vec<f64> = col.iter().copied().filter(|v| !v.is_nan()).collect();

            if valid_vals.len() < 3 || variances[j] < 1e-10 {
                feature_skewness.push(0.0);
                continue;
            }

            let mean = valid_vals.iter().sum::<f64>() / valid_vals.len() as f64;
            let std = variances[j].sqrt();
            let n = valid_vals.len() as f64;

            let skew = valid_vals
                .iter()
                .map(|v| ((v - mean) / std).powi(3))
                .sum::<f64>()
                / n;
            feature_skewness.push(skew);

            if skew.abs() > self.config.skewness_threshold {
                skewed_feature_indices.push(j);
            }
        }

        // Check for scaling recommendation
        let variance_ratio = if let (Some(&max), Some(&min)) = (
            variances
                .iter()
                .filter(|&&v| v > 1e-10)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
            variances
                .iter()
                .filter(|&&v| v > 1e-10)
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
        ) {
            max / min.max(1e-10)
        } else {
            1.0
        };

        let scaling_recommended = variance_ratio > 10.0;

        // Analyze class distribution if y is provided
        let (class_imbalance_detected, class_distribution) = if let Some(y) = y {
            let mut dist: HashMap<i64, usize> = HashMap::new();
            for &val in y.iter() {
                *dist.entry(val as i64).or_insert(0) += 1;
            }

            let max_count = *dist.values().max().unwrap_or(&1) as f64;
            let min_count = *dist.values().min().unwrap_or(&1) as f64;
            let imbalance_ratio = max_count / min_count.max(1.0);

            (imbalance_ratio >= self.config.imbalance_threshold, dist)
        } else {
            (false, HashMap::new())
        };

        DetectedCharacteristics {
            missing_feature_indices,
            missing_percentages,
            categorical_feature_indices: Vec::new(), // Hard to detect from f64 array
            high_variance_indices,
            low_variance_indices,
            skewed_feature_indices,
            feature_skewness,
            has_negative_values: has_negative,
            scaling_recommended,
            class_imbalance_detected,
            class_distribution,
        }
    }

    /// Build a preprocessing pipeline based on data characteristics and requirements
    pub fn build_pipeline(
        &self,
        data_chars: &DataCharacteristics,
        requirements: &[PreprocessingRequirement],
        detected: Option<&DetectedCharacteristics>,
    ) -> PreprocessingPipelineSpec {
        let mut steps = Vec::new();

        // Use detected characteristics or create default
        let detected_chars = match detected {
            Some(d) => d.clone(),
            None => DetectedCharacteristics::default(),
        };

        // Determine effective strategy based on Auto mode
        let effective_strategy = match self.config.strategy {
            PreprocessingStrategy::Auto => self.determine_auto_strategy(data_chars),
            other => other,
        };

        if effective_strategy == PreprocessingStrategy::Passthrough {
            return PreprocessingPipelineSpec {
                steps: Vec::new(),
                detected_characteristics: detected_chars,
            };
        }

        // Step 1: Handle missing values (if required)
        if requirements.contains(&PreprocessingRequirement::HandleMissing)
            || data_chars.has_missing_values
        {
            steps.push(self.create_imputer_step(&detected_chars, effective_strategy));
        }

        // Step 2: Encode categorical features (if required)
        if requirements.contains(&PreprocessingRequirement::EncodeCategorical)
            || data_chars.has_categorical
        {
            steps.push(self.create_encoder_step(effective_strategy));
        }

        // Step 3: Apply power transform for skewed features (if thorough)
        if effective_strategy == PreprocessingStrategy::Thorough
            && self.config.use_power_transform
            && !detected_chars.skewed_feature_indices.is_empty()
        {
            steps.push(self.create_power_transform_step(&detected_chars));
        }

        // Step 4: Handle non-negative requirement
        if requirements.contains(&PreprocessingRequirement::NonNegative)
            && detected_chars.has_negative_values
        {
            // Use MinMax scaler to ensure [0, 1] range
            steps.push(
                PreprocessingStepSpec::new("minmax_nonneg", PreprocessingStepType::MinMaxScaler)
                    .with_param("feature_range_min", StepParam::Float(0.0))
                    .with_param("feature_range_max", StepParam::Float(1.0)),
            );
        }

        // Step 5: Apply scaling (if required)
        if requirements.contains(&PreprocessingRequirement::Scaling)
            || (detected_chars.scaling_recommended
                && effective_strategy != PreprocessingStrategy::Conservative)
        {
            // Don't duplicate if we already added MinMax for non-negative
            if !requirements.contains(&PreprocessingRequirement::NonNegative)
                || !detected_chars.has_negative_values
            {
                steps.push(self.create_scaler_step(effective_strategy));
            }
        }

        // Step 6: Feature selection (for Thorough strategy or low variance features)
        if effective_strategy == PreprocessingStrategy::Thorough
            && !detected_chars.low_variance_indices.is_empty()
        {
            steps.push(
                PreprocessingStepSpec::new(
                    "variance_filter",
                    PreprocessingStepType::VarianceThreshold,
                )
                .with_param(
                    "threshold",
                    StepParam::Float(self.config.variance_threshold),
                ),
            );
        }

        // Step 7: Dimensionality reduction (if required or high-dimensional)
        if requirements.contains(&PreprocessingRequirement::DimensionalityReduction)
            || (self.config.reduce_dimensions
                && effective_strategy == PreprocessingStrategy::Thorough
                && self.should_reduce_dimensions(data_chars))
        {
            steps.push(self.create_dimensionality_reduction_step(data_chars));
        }

        PreprocessingPipelineSpec {
            steps,
            detected_characteristics: detected_chars,
        }
    }

    /// Build preprocessing pipeline for a specific algorithm
    pub fn build_pipeline_for_algorithm(
        &self,
        data_chars: &DataCharacteristics,
        requirements: &[PreprocessingRequirement],
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
    ) -> PreprocessingPipelineSpec {
        let detected = self.analyze_data(x, y);
        self.build_pipeline(data_chars, requirements, Some(&detected))
    }

    /// Determine the automatic strategy based on data characteristics
    fn determine_auto_strategy(&self, chars: &DataCharacteristics) -> PreprocessingStrategy {
        // For very small datasets, use conservative to avoid overfitting preprocessing
        if chars.n_samples < 100 {
            return PreprocessingStrategy::Conservative;
        }

        // For high-dimensional data, use thorough to help with feature reduction
        if chars.n_features > chars.n_samples / 2 {
            return PreprocessingStrategy::Thorough;
        }

        // For large datasets with many features, use standard
        if chars.n_samples > 10000 && chars.n_features > 50 {
            return PreprocessingStrategy::Standard;
        }

        // Default to standard
        PreprocessingStrategy::Standard
    }

    /// Check if dimensionality reduction should be applied
    fn should_reduce_dimensions(&self, chars: &DataCharacteristics) -> bool {
        let ratio = chars.n_features as f64 / chars.n_samples as f64;
        ratio > self.config.dimension_reduction_threshold
    }

    /// Create an imputer step based on strategy
    fn create_imputer_step(
        &self,
        detected: &DetectedCharacteristics,
        strategy: PreprocessingStrategy,
    ) -> PreprocessingStepSpec {
        // Calculate average missing percentage
        let avg_missing: f64 = if detected.missing_percentages.is_empty() {
            0.0
        } else {
            detected.missing_percentages.iter().sum::<f64>()
                / detected.missing_percentages.len() as f64
        };

        // Choose imputation strategy
        let impute_strategy = match strategy {
            PreprocessingStrategy::Conservative => ImputationStrategy::Median,
            PreprocessingStrategy::Standard => self.config.imputation_strategy,
            PreprocessingStrategy::Thorough => {
                // Use KNN for moderate missing, simpler for sparse/heavy missing
                if avg_missing > 0.01 && avg_missing < 0.3 {
                    ImputationStrategy::KNN
                } else {
                    ImputationStrategy::Median
                }
            }
            _ => self.config.imputation_strategy,
        };

        match impute_strategy {
            ImputationStrategy::KNN => {
                PreprocessingStepSpec::new("imputer", PreprocessingStepType::KNNImputer)
                    .with_param("n_neighbors", StepParam::Int(5))
            }
            ImputationStrategy::Mean => {
                PreprocessingStepSpec::new("imputer", PreprocessingStepType::SimpleImputer)
                    .with_param("strategy", StepParam::String("mean".to_string()))
            }
            ImputationStrategy::Median => {
                PreprocessingStepSpec::new("imputer", PreprocessingStepType::SimpleImputer)
                    .with_param("strategy", StepParam::String("median".to_string()))
            }
            ImputationStrategy::Mode => {
                PreprocessingStepSpec::new("imputer", PreprocessingStepType::SimpleImputer)
                    .with_param("strategy", StepParam::String("most_frequent".to_string()))
            }
            ImputationStrategy::Constant => {
                PreprocessingStepSpec::new("imputer", PreprocessingStepType::SimpleImputer)
                    .with_param("strategy", StepParam::String("constant".to_string()))
                    .with_param("fill_value", StepParam::Float(0.0))
            }
        }
    }

    /// Create an encoder step based on strategy
    fn create_encoder_step(&self, strategy: PreprocessingStrategy) -> PreprocessingStepSpec {
        match (strategy, self.config.encoding_type) {
            (PreprocessingStrategy::Conservative, _) => {
                PreprocessingStepSpec::new("encoder", PreprocessingStepType::OrdinalEncoder)
            }
            (_, EncodingType::OneHot) => {
                PreprocessingStepSpec::new("encoder", PreprocessingStepType::OneHotEncoder)
                    .with_param("handle_unknown", StepParam::String("ignore".to_string()))
            }
            (_, EncodingType::Ordinal) => {
                PreprocessingStepSpec::new("encoder", PreprocessingStepType::OrdinalEncoder)
            }
            (_, EncodingType::Target) => {
                PreprocessingStepSpec::new("encoder", PreprocessingStepType::TargetEncoder)
                    .with_param("smoothing", StepParam::Float(10.0))
            }
        }
    }

    /// Create a scaler step based on strategy
    fn create_scaler_step(&self, strategy: PreprocessingStrategy) -> PreprocessingStepSpec {
        let scaler_type = match strategy {
            PreprocessingStrategy::Conservative => ScalerType::Standard,
            PreprocessingStrategy::Standard => self.config.scaler_type,
            PreprocessingStrategy::Thorough => ScalerType::Robust, // More robust to outliers
            _ => self.config.scaler_type,
        };

        match scaler_type {
            ScalerType::Standard => {
                PreprocessingStepSpec::new("scaler", PreprocessingStepType::StandardScaler)
            }
            ScalerType::MinMax => {
                PreprocessingStepSpec::new("scaler", PreprocessingStepType::MinMaxScaler)
            }
            ScalerType::Robust => {
                PreprocessingStepSpec::new("scaler", PreprocessingStepType::RobustScaler)
            }
            ScalerType::MaxAbs => {
                PreprocessingStepSpec::new("scaler", PreprocessingStepType::MaxAbsScaler)
            }
            ScalerType::None => {
                // Return a passthrough (no-op)
                PreprocessingStepSpec::new("scaler", PreprocessingStepType::StandardScaler)
            }
        }
    }

    /// Create a power transform step for skewed features
    fn create_power_transform_step(
        &self,
        detected: &DetectedCharacteristics,
    ) -> PreprocessingStepSpec {
        // Use Yeo-Johnson if there are negative values, Box-Cox otherwise
        let method = if detected.has_negative_values {
            "yeo-johnson"
        } else {
            "box-cox"
        };

        PreprocessingStepSpec::new("power_transform", PreprocessingStepType::PowerTransformer)
            .with_param("method", StepParam::String(method.to_string()))
            .with_param("standardize", StepParam::Bool(true))
    }

    /// Create a dimensionality reduction step
    fn create_dimensionality_reduction_step(
        &self,
        chars: &DataCharacteristics,
    ) -> PreprocessingStepSpec {
        // Keep enough components to capture most variance
        let n_components = (chars.n_features / 2).max(10).min(chars.n_features);

        PreprocessingStepSpec::new("dim_reduction", PreprocessingStepType::PCA)
            .with_param("n_components", StepParam::Int(n_components as i64))
    }

    /// Create a resampling step for class imbalance
    pub fn create_resampling_step(
        &self,
        detected: &DetectedCharacteristics,
    ) -> Option<PreprocessingStepSpec> {
        if !self.config.handle_imbalance || !detected.class_imbalance_detected {
            return None;
        }

        // Calculate total samples in each class
        let total_samples: usize = detected.class_distribution.values().sum();
        let n_classes = detected.class_distribution.len();

        if n_classes < 2 || total_samples < 10 {
            return None;
        }

        let min_class_count = *detected.class_distribution.values().min().unwrap_or(&0);

        // Use SMOTE for moderate imbalance, undersampling for severe imbalance
        if min_class_count >= 6 {
            // Enough samples for SMOTE (needs k neighbors)
            Some(
                PreprocessingStepSpec::new("resampler", PreprocessingStepType::SMOTE)
                    .with_param("k_neighbors", StepParam::Int(5))
                    .with_param("sampling_strategy", StepParam::String("auto".to_string())),
            )
        } else {
            // Too few minority samples, use undersampling
            Some(
                PreprocessingStepSpec::new("resampler", PreprocessingStepType::RandomUnderSampler)
                    .with_param(
                        "sampling_strategy",
                        StepParam::String("majority".to_string()),
                    ),
            )
        }
    }
}

/// Result of preprocessing selection for an algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingSelection {
    /// The preprocessing pipeline specification
    pub pipeline_spec: PreprocessingPipelineSpec,
    /// Resampling step (applied separately, not in pipeline)
    pub resampling_step: Option<PreprocessingStepSpec>,
    /// Whether preprocessing is needed at all
    pub needs_preprocessing: bool,
    /// Summary of why each step was selected
    pub selection_reasons: Vec<String>,
}

impl PreprocessingSelection {
    /// Check if the selection includes any steps
    pub fn has_steps(&self) -> bool {
        !self.pipeline_spec.steps.is_empty() || self.resampling_step.is_some()
    }

    /// Get all step names
    pub fn step_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self
            .pipeline_spec
            .steps
            .iter()
            .map(|s| s.name.clone())
            .collect();
        if let Some(ref rs) = self.resampling_step {
            names.push(rs.name.clone());
        }
        names
    }
}

/// Build a complete preprocessing selection for an algorithm
pub fn select_preprocessing(
    selector: &PreprocessingSelector,
    data_chars: &DataCharacteristics,
    requirements: &[PreprocessingRequirement],
    x: &Array2<f64>,
    y: Option<&Array1<f64>>,
) -> PreprocessingSelection {
    let detected = selector.analyze_data(x, y);
    let pipeline_spec = selector.build_pipeline(data_chars, requirements, Some(&detected));
    let resampling_step = if !data_chars.is_regression {
        selector.create_resampling_step(&detected)
    } else {
        None
    };

    let mut reasons = Vec::new();

    // Document why each step was selected
    for step in &pipeline_spec.steps {
        let reason = match step.step_type {
            PreprocessingStepType::SimpleImputer | PreprocessingStepType::KNNImputer => {
                format!(
                    "Imputation: {} features have missing values",
                    detected.missing_feature_indices.len()
                )
            }
            PreprocessingStepType::OneHotEncoder
            | PreprocessingStepType::OrdinalEncoder
            | PreprocessingStepType::TargetEncoder => {
                "Encoding: categorical features detected or required by algorithm".to_string()
            }
            PreprocessingStepType::StandardScaler
            | PreprocessingStepType::MinMaxScaler
            | PreprocessingStepType::RobustScaler
            | PreprocessingStepType::MaxAbsScaler => {
                if requirements.contains(&PreprocessingRequirement::Scaling) {
                    "Scaling: required by algorithm".to_string()
                } else {
                    "Scaling: high variance ratio detected between features".to_string()
                }
            }
            PreprocessingStepType::PowerTransformer => format!(
                "Power transform: {} features have high skewness",
                detected.skewed_feature_indices.len()
            ),
            PreprocessingStepType::VarianceThreshold => format!(
                "Variance filter: {} features have near-zero variance",
                detected.low_variance_indices.len()
            ),
            PreprocessingStepType::PCA => format!(
                "PCA: high-dimensional data ({} features / {} samples)",
                data_chars.n_features, data_chars.n_samples
            ),
            _ => format!("Step: {:?}", step.step_type),
        };
        reasons.push(reason);
    }

    if resampling_step.is_some() {
        reasons.push(format!(
            "Resampling: class imbalance ratio {:.2}",
            data_chars.class_imbalance_ratio
        ));
    }

    // Compute needs_preprocessing before moving values
    let needs_preprocessing = !pipeline_spec.steps.is_empty() || resampling_step.is_some();

    PreprocessingSelection {
        pipeline_spec,
        resampling_step,
        needs_preprocessing,
        selection_reasons: reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn create_test_data() -> (Array2<f64>, Array1<f64>) {
        // Create data with various characteristics
        let x = array![
            [1.0, 100.0, 0.5],
            [2.0, 200.0, 0.6],
            [3.0, 300.0, f64::NAN], // Missing value
            [4.0, 400.0, 0.8],
            [5.0, 500.0, 0.9],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 0.0];
        (x, y)
    }

    #[test]
    fn test_preprocessing_config_default() {
        let config = PreprocessingConfig::default();
        assert_eq!(config.strategy, PreprocessingStrategy::Auto);
        assert!(config.handle_imbalance);
        assert_eq!(config.scaler_type, ScalerType::Standard);
    }

    #[test]
    fn test_preprocessing_config_builder() {
        let config = PreprocessingConfig::new()
            .with_strategy(PreprocessingStrategy::Thorough)
            .with_scaler(ScalerType::Robust)
            .with_imputation(ImputationStrategy::KNN)
            .with_handle_imbalance(false);

        assert_eq!(config.strategy, PreprocessingStrategy::Thorough);
        assert_eq!(config.scaler_type, ScalerType::Robust);
        assert_eq!(config.imputation_strategy, ImputationStrategy::KNN);
        assert!(!config.handle_imbalance);
    }

    #[test]
    fn test_selector_creation() {
        let selector = PreprocessingSelector::auto();
        assert_eq!(selector.config.strategy, PreprocessingStrategy::Auto);

        let selector = PreprocessingSelector::conservative();
        assert_eq!(
            selector.config.strategy,
            PreprocessingStrategy::Conservative
        );

        let selector = PreprocessingSelector::thorough();
        assert_eq!(selector.config.strategy, PreprocessingStrategy::Thorough);
    }

    #[test]
    fn test_analyze_data_missing_values() {
        let (x, y) = create_test_data();
        let selector = PreprocessingSelector::auto();
        let detected = selector.analyze_data(&x, Some(&y));

        assert!(!detected.missing_feature_indices.is_empty());
        assert!(detected.missing_feature_indices.contains(&2)); // Column 2 has NaN
        assert!(detected.missing_percentages[2] > 0.0);
    }

    #[test]
    fn test_analyze_data_variance() {
        let (x, y) = create_test_data();
        let selector = PreprocessingSelector::auto();
        let detected = selector.analyze_data(&x, Some(&y));

        // Column 1 has much higher variance than column 2
        assert!(detected.scaling_recommended);
    }

    #[test]
    fn test_analyze_data_class_distribution() {
        let (x, y) = create_test_data();
        let selector = PreprocessingSelector::auto();
        let detected = selector.analyze_data(&x, Some(&y));

        assert!(!detected.class_distribution.is_empty());
        assert_eq!(detected.class_distribution.get(&0), Some(&3));
        assert_eq!(detected.class_distribution.get(&1), Some(&2));
    }

    #[test]
    fn test_build_pipeline_with_missing_requirement() {
        let selector = PreprocessingSelector::new(
            PreprocessingConfig::new().with_strategy(PreprocessingStrategy::Standard),
        );

        let chars = DataCharacteristics::new(100, 5)
            .with_missing_values(true)
            .with_n_classes(2);

        let requirements = vec![PreprocessingRequirement::HandleMissing];
        let pipeline = selector.build_pipeline(&chars, &requirements, None);

        // Should have an imputer step
        assert!(!pipeline.steps.is_empty());
        assert!(pipeline.steps.iter().any(|s| matches!(
            s.step_type,
            PreprocessingStepType::SimpleImputer | PreprocessingStepType::KNNImputer
        )));
    }

    #[test]
    fn test_build_pipeline_with_scaling_requirement() {
        let selector = PreprocessingSelector::new(
            PreprocessingConfig::new().with_strategy(PreprocessingStrategy::Standard),
        );

        let chars = DataCharacteristics::new(100, 5).with_n_classes(2);
        let requirements = vec![PreprocessingRequirement::Scaling];
        let pipeline = selector.build_pipeline(&chars, &requirements, None);

        // Should have a scaler step
        assert!(!pipeline.steps.is_empty());
        assert!(pipeline.steps.iter().any(|s| matches!(
            s.step_type,
            PreprocessingStepType::StandardScaler
                | PreprocessingStepType::MinMaxScaler
                | PreprocessingStepType::RobustScaler
        )));
    }

    #[test]
    fn test_build_pipeline_with_encoding_requirement() {
        let selector = PreprocessingSelector::new(
            PreprocessingConfig::new()
                .with_strategy(PreprocessingStrategy::Standard)
                .with_encoding(EncodingType::OneHot),
        );

        let chars = DataCharacteristics::new(100, 5)
            .with_categorical(true)
            .with_n_classes(2);

        let requirements = vec![PreprocessingRequirement::EncodeCategorical];
        let pipeline = selector.build_pipeline(&chars, &requirements, None);

        // Should have an encoder step
        assert!(!pipeline.steps.is_empty());
        assert!(pipeline.steps.iter().any(|s| matches!(
            s.step_type,
            PreprocessingStepType::OneHotEncoder
                | PreprocessingStepType::OrdinalEncoder
                | PreprocessingStepType::TargetEncoder
        )));
    }

    #[test]
    fn test_build_pipeline_nonnegative() {
        let selector = PreprocessingSelector::new(
            PreprocessingConfig::new().with_strategy(PreprocessingStrategy::Standard),
        );

        let chars = DataCharacteristics::new(100, 5).with_n_classes(2);
        let requirements = vec![PreprocessingRequirement::NonNegative];

        // Need detected characteristics with negative values
        let detected = DetectedCharacteristics {
            has_negative_values: true,
            ..Default::default()
        };

        let pipeline = selector.build_pipeline(&chars, &requirements, Some(&detected));

        // Should have MinMaxScaler for non-negative
        assert!(!pipeline.steps.is_empty());
        assert!(pipeline
            .steps
            .iter()
            .any(|s| matches!(s.step_type, PreprocessingStepType::MinMaxScaler)));
    }

    #[test]
    fn test_build_pipeline_passthrough_strategy() {
        let selector = PreprocessingSelector::new(
            PreprocessingConfig::new().with_strategy(PreprocessingStrategy::Passthrough),
        );

        let chars = DataCharacteristics::new(100, 5)
            .with_missing_values(true)
            .with_n_classes(2);

        let requirements = vec![
            PreprocessingRequirement::HandleMissing,
            PreprocessingRequirement::Scaling,
        ];
        let pipeline = selector.build_pipeline(&chars, &requirements, None);

        // Passthrough should have no steps
        assert!(pipeline.steps.is_empty());
    }

    #[test]
    fn test_build_pipeline_conservative_vs_thorough() {
        let conservative = PreprocessingSelector::conservative();
        let thorough = PreprocessingSelector::thorough();

        let chars = DataCharacteristics::new(1000, 50)
            .with_missing_values(true)
            .with_n_classes(2);

        let requirements = vec![
            PreprocessingRequirement::HandleMissing,
            PreprocessingRequirement::Scaling,
        ];

        let cons_pipeline = conservative.build_pipeline(&chars, &requirements, None);
        let thor_pipeline = thorough.build_pipeline(&chars, &requirements, None);

        // Thorough should generally have more steps
        // (variance threshold, maybe power transform, etc.)
        assert!(thor_pipeline.steps.len() >= cons_pipeline.steps.len());
    }

    #[test]
    fn test_resampling_step_creation() {
        let selector = PreprocessingSelector::new(
            PreprocessingConfig::new()
                .with_handle_imbalance(true)
                .with_imbalance_threshold(2.0),
        );

        // Imbalanced class distribution
        let mut dist = HashMap::new();
        dist.insert(0, 100);
        dist.insert(1, 20);

        let detected = DetectedCharacteristics {
            class_imbalance_detected: true,
            class_distribution: dist,
            ..Default::default()
        };

        let resampling = selector.create_resampling_step(&detected);

        assert!(resampling.is_some());
        let step = resampling.unwrap();
        assert!(matches!(
            step.step_type,
            PreprocessingStepType::SMOTE | PreprocessingStepType::RandomUnderSampler
        ));
    }

    #[test]
    fn test_resampling_step_not_created_when_disabled() {
        let selector =
            PreprocessingSelector::new(PreprocessingConfig::new().with_handle_imbalance(false));

        let mut dist = HashMap::new();
        dist.insert(0, 100);
        dist.insert(1, 20);

        let detected = DetectedCharacteristics {
            class_imbalance_detected: true,
            class_distribution: dist,
            ..Default::default()
        };

        let resampling = selector.create_resampling_step(&detected);
        assert!(resampling.is_none());
    }

    #[test]
    fn test_select_preprocessing_function() {
        let (x, y) = create_test_data();
        let selector = PreprocessingSelector::new(
            PreprocessingConfig::new().with_strategy(PreprocessingStrategy::Standard),
        );

        let chars = DataCharacteristics::from_data(&x, &y);
        let requirements = vec![
            PreprocessingRequirement::HandleMissing,
            PreprocessingRequirement::Scaling,
        ];

        let selection = select_preprocessing(&selector, &chars, &requirements, &x, Some(&y));

        assert!(selection.needs_preprocessing);
        assert!(!selection.pipeline_spec.steps.is_empty());
        assert!(!selection.selection_reasons.is_empty());
    }

    #[test]
    fn test_selection_step_names() {
        let selector = PreprocessingSelector::new(
            PreprocessingConfig::new().with_strategy(PreprocessingStrategy::Standard),
        );

        let chars = DataCharacteristics::new(100, 5)
            .with_missing_values(true)
            .with_n_classes(2);

        let requirements = vec![
            PreprocessingRequirement::HandleMissing,
            PreprocessingRequirement::Scaling,
        ];

        let pipeline = selector.build_pipeline(&chars, &requirements, None);

        let selection = PreprocessingSelection {
            pipeline_spec: pipeline,
            resampling_step: None,
            needs_preprocessing: true,
            selection_reasons: vec![],
        };

        let names = selection.step_names();
        assert!(!names.is_empty());
    }

    #[test]
    fn test_auto_strategy_small_dataset() {
        let selector = PreprocessingSelector::auto();
        let chars = DataCharacteristics::new(50, 10).with_n_classes(2);

        let effective = selector.determine_auto_strategy(&chars);
        assert_eq!(effective, PreprocessingStrategy::Conservative);
    }

    #[test]
    fn test_auto_strategy_high_dimensional() {
        let selector = PreprocessingSelector::auto();
        let chars = DataCharacteristics::new(100, 80).with_n_classes(2);

        let effective = selector.determine_auto_strategy(&chars);
        assert_eq!(effective, PreprocessingStrategy::Thorough);
    }

    #[test]
    fn test_dimensionality_reduction_threshold() {
        let selector = PreprocessingSelector::new(
            PreprocessingConfig::new()
                .with_strategy(PreprocessingStrategy::Thorough)
                .with_dimension_reduction(true)
                .with_dimension_reduction_threshold(0.3),
        );

        let chars = DataCharacteristics::new(100, 50).with_n_classes(2);
        let requirements = vec![PreprocessingRequirement::DimensionalityReduction];
        let pipeline = selector.build_pipeline(&chars, &requirements, None);

        // Should have PCA step
        assert!(pipeline
            .steps
            .iter()
            .any(|s| matches!(s.step_type, PreprocessingStepType::PCA)));
    }

    #[test]
    fn test_preprocessing_step_spec_builder() {
        let spec = PreprocessingStepSpec::new("my_imputer", PreprocessingStepType::SimpleImputer)
            .with_param("strategy", StepParam::String("median".to_string()))
            .with_param("n_neighbors", StepParam::Int(5))
            .with_columns(vec![0, 1, 2]);

        assert_eq!(spec.name, "my_imputer");
        assert_eq!(spec.step_type, PreprocessingStepType::SimpleImputer);
        assert!(spec.params.contains_key("strategy"));
        assert!(spec.params.contains_key("n_neighbors"));
        assert_eq!(spec.column_indices, Some(vec![0, 1, 2]));
    }

    #[test]
    fn test_multiple_requirements() {
        let selector = PreprocessingSelector::new(
            PreprocessingConfig::new().with_strategy(PreprocessingStrategy::Standard),
        );

        let chars = DataCharacteristics::new(100, 10)
            .with_missing_values(true)
            .with_categorical(true)
            .with_n_classes(2);

        let requirements = vec![
            PreprocessingRequirement::HandleMissing,
            PreprocessingRequirement::EncodeCategorical,
            PreprocessingRequirement::Scaling,
        ];

        let pipeline = selector.build_pipeline(&chars, &requirements, None);

        // Should have imputer, encoder, and scaler
        let step_types: Vec<_> = pipeline.steps.iter().map(|s| s.step_type).collect();

        let has_imputer = step_types.iter().any(|t| {
            matches!(
                t,
                PreprocessingStepType::SimpleImputer | PreprocessingStepType::KNNImputer
            )
        });
        let has_encoder = step_types.iter().any(|t| {
            matches!(
                t,
                PreprocessingStepType::OneHotEncoder
                    | PreprocessingStepType::OrdinalEncoder
                    | PreprocessingStepType::TargetEncoder
            )
        });
        let has_scaler = step_types.iter().any(|t| {
            matches!(
                t,
                PreprocessingStepType::StandardScaler
                    | PreprocessingStepType::MinMaxScaler
                    | PreprocessingStepType::RobustScaler
            )
        });

        assert!(has_imputer, "Should have imputer step");
        assert!(has_encoder, "Should have encoder step");
        assert!(has_scaler, "Should have scaler step");
    }

    #[test]
    fn test_skewness_detection() {
        // Create highly skewed data
        let x = Array2::from_shape_vec(
            (100, 2),
            (0..200)
                .map(|i| {
                    if i < 100 {
                        // First column: exponential (skewed)
                        (i as f64 / 10.0).exp()
                    } else {
                        // Second column: normal
                        i as f64 - 150.0
                    }
                })
                .collect(),
        )
        .unwrap();

        let selector = PreprocessingSelector::new(
            PreprocessingConfig::new()
                .with_strategy(PreprocessingStrategy::Thorough)
                .with_power_transform(true)
                .with_skewness_threshold(1.0),
        );

        let detected = selector.analyze_data(&x, None);

        // First column should be detected as skewed
        assert!(!detected.skewed_feature_indices.is_empty());
    }

    #[test]
    fn test_regression_no_resampling() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        // Use non-integer values to trigger regression detection
        let y = array![1.1, 2.2, 3.3, 4.4, 5.5]; // Continuous target

        let selector =
            PreprocessingSelector::new(PreprocessingConfig::new().with_handle_imbalance(true));

        let chars = DataCharacteristics::from_data(&x, &y);
        assert!(chars.is_regression);

        let requirements = vec![];
        let selection = select_preprocessing(&selector, &chars, &requirements, &x, Some(&y));

        // Regression tasks shouldn't have resampling
        assert!(selection.resampling_step.is_none());
    }
}

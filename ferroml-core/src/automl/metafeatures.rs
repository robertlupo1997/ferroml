//! Dataset Metafeature Extraction for Meta-Learning
//!
//! This module provides comprehensive dataset characterization through metafeatures,
//! enabling meta-learning for warm-starting AutoML and algorithm selection.
//!
//! # Metafeature Categories
//!
//! - **Statistical**: Mean, std, skewness, kurtosis per feature
//! - **Information-theoretic**: Entropy, mutual information
//! - **Landmarking**: Performance of simple models (decision stump, 1-NN, Naive Bayes)
//!
//! # Example
//!
//! ```ignore
//! use ferroml_core::automl::metafeatures::{DatasetMetafeatures, MetafeatureConfig};
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((100, 5), (0..500).map(|i| i as f64).collect()).unwrap();
//! let y = Array1::from_vec((0..100).map(|i| (i % 3) as f64).collect());
//!
//! // Extract metafeatures with default configuration
//! let metafeatures = DatasetMetafeatures::extract(&x, &y, true, None).unwrap();
//!
//! println!("Dataset has {} features", metafeatures.simple.n_features);
//! println!("Skewness: {:?}", metafeatures.statistical.skewness_mean);
//! println!("Target entropy: {}", metafeatures.information.target_entropy);
//! if let Some(landmarking) = &metafeatures.landmarking {
//!     println!("1-NN accuracy: {}", landmarking.one_nn_accuracy);
//! }
//! ```

use crate::models::{
    DecisionTreeClassifier, DecisionTreeRegressor, GaussianNB, KNeighborsClassifier,
    KNeighborsRegressor, Model, SplitCriterion,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for metafeature extraction
#[derive(Debug, Clone)]
pub struct MetafeatureConfig {
    /// Whether to compute landmarking features (slower but informative)
    pub compute_landmarking: bool,
    /// Number of CV folds for landmarking (default: 3)
    pub landmarking_cv_folds: usize,
    /// Maximum samples to use for landmarking (subsampling for large datasets)
    pub landmarking_max_samples: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

impl Default for MetafeatureConfig {
    fn default() -> Self {
        Self {
            compute_landmarking: true,
            landmarking_cv_folds: 3,
            landmarking_max_samples: 5000,
            random_state: None,
        }
    }
}

impl MetafeatureConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Disable landmarking computation (faster extraction)
    pub fn without_landmarking(mut self) -> Self {
        self.compute_landmarking = false;
        self
    }

    /// Set the number of CV folds for landmarking
    pub fn with_landmarking_cv_folds(mut self, folds: usize) -> Self {
        self.landmarking_cv_folds = folds;
        self
    }

    /// Set maximum samples for landmarking
    pub fn with_landmarking_max_samples(mut self, max_samples: usize) -> Self {
        self.landmarking_max_samples = max_samples;
        self
    }

    /// Set random state for reproducibility
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

/// Complete dataset metafeatures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetafeatures {
    /// Simple/general metafeatures
    pub simple: SimpleMetafeatures,
    /// Statistical metafeatures
    pub statistical: StatisticalMetafeatures,
    /// Information-theoretic metafeatures
    pub information: InformationMetafeatures,
    /// Landmarking metafeatures (performance of simple models)
    pub landmarking: Option<LandmarkingMetafeatures>,
}

impl DatasetMetafeatures {
    /// Extract all metafeatures from a dataset
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples, n_features)
    /// * `y` - Target values (n_samples,)
    /// * `is_classification` - Whether this is a classification task
    /// * `config` - Optional configuration (uses defaults if None)
    ///
    /// # Returns
    /// Complete metafeature extraction results
    pub fn extract(
        x: &Array2<f64>,
        y: &Array1<f64>,
        is_classification: bool,
        config: Option<MetafeatureConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();

        // Validate inputs
        if x.nrows() != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("X has {} rows", x.nrows()),
                format!("y has {} elements", y.len()),
            ));
        }
        if x.is_empty() || y.is_empty() {
            return Err(FerroError::invalid_input("Empty input data"));
        }

        // Extract each category of metafeatures
        let simple = SimpleMetafeatures::extract(x, y, is_classification);
        let statistical = StatisticalMetafeatures::extract(x);
        let information = InformationMetafeatures::extract(x, y, is_classification);

        let landmarking = if config.compute_landmarking {
            Some(LandmarkingMetafeatures::extract(
                x,
                y,
                is_classification,
                &config,
            )?)
        } else {
            None
        };

        Ok(Self {
            simple,
            statistical,
            information,
            landmarking,
        })
    }

    /// Get a flat vector of all numeric metafeatures for similarity computation
    pub fn to_vector(&self) -> Vec<f64> {
        let mut vec = Vec::new();

        // Simple metafeatures
        vec.push(self.simple.n_samples as f64);
        vec.push(self.simple.n_features as f64);
        vec.push(self.simple.dimensionality);
        if let Some(nc) = self.simple.n_classes {
            vec.push(nc as f64);
        }
        if let Some(ir) = self.simple.imbalance_ratio {
            vec.push(ir);
        }
        vec.push(self.simple.missing_ratio);

        // Statistical metafeatures
        vec.push(self.statistical.mean_mean);
        vec.push(self.statistical.mean_std);
        vec.push(self.statistical.std_mean);
        vec.push(self.statistical.std_std);
        vec.push(self.statistical.skewness_mean);
        vec.push(self.statistical.skewness_std);
        vec.push(self.statistical.kurtosis_mean);
        vec.push(self.statistical.kurtosis_std);
        vec.push(self.statistical.outlier_ratio);
        vec.push(self.statistical.correlation_mean);
        vec.push(self.statistical.correlation_std);

        // Information-theoretic metafeatures
        vec.push(self.information.target_entropy);
        vec.push(self.information.feature_entropy_mean);
        vec.push(self.information.feature_entropy_std);
        vec.push(self.information.mutual_info_mean);
        vec.push(self.information.mutual_info_std);
        vec.push(self.information.normalized_entropy);

        // Landmarking metafeatures (if available)
        if let Some(ref lm) = self.landmarking {
            vec.push(lm.one_nn_score);
            vec.push(lm.decision_stump_score);
            vec.push(lm.naive_bayes_score);
            vec.push(lm.linear_model_score);
        }

        vec
    }

    /// Compute similarity to another dataset's metafeatures (cosine similarity)
    pub fn similarity(&self, other: &DatasetMetafeatures) -> f64 {
        let v1 = self.to_vector();
        let v2 = other.to_vector();

        // Ensure same length (use min length)
        let len = v1.len().min(v2.len());
        if len == 0 {
            return 0.0;
        }

        let dot: f64 = v1.iter().zip(v2.iter()).take(len).map(|(a, b)| a * b).sum();
        let norm1: f64 = v1.iter().take(len).map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = v2.iter().take(len).map(|x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot / (norm1 * norm2)
        }
    }
}

/// Simple/general metafeatures describing the dataset structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleMetafeatures {
    /// Number of samples (instances)
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Number of numeric features
    pub n_numeric_features: usize,
    /// Number of classes (classification only)
    pub n_classes: Option<usize>,
    /// Dimensionality ratio: n_features / n_samples
    pub dimensionality: f64,
    /// Class imbalance ratio: max_count / min_count (classification only)
    pub imbalance_ratio: Option<f64>,
    /// Ratio of missing values (NaN)
    pub missing_ratio: f64,
    /// Whether the dataset is high-dimensional (n_features > n_samples)
    pub is_high_dimensional: bool,
    /// Whether the dataset is large (n_samples * n_features > 1M)
    pub is_large: bool,
    /// Log of dataset size
    pub log_size: f64,
}

impl SimpleMetafeatures {
    /// Extract simple metafeatures from dataset
    pub fn extract(x: &Array2<f64>, y: &Array1<f64>, is_classification: bool) -> Self {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_numeric_features = n_features; // Assuming all numeric for now

        // Count missing values
        let missing_count = x.iter().filter(|v| v.is_nan()).count();
        let total_values = n_samples * n_features;
        let missing_ratio = if total_values > 0 {
            missing_count as f64 / total_values as f64
        } else {
            0.0
        };

        // Classification-specific metrics
        let (n_classes, imbalance_ratio) = if is_classification {
            let class_counts = count_classes(y);
            let n_classes = class_counts.len();
            let max_count = class_counts.values().max().copied().unwrap_or(1) as f64;
            let min_count = class_counts.values().min().copied().unwrap_or(1).max(1) as f64;
            let imbalance = max_count / min_count;
            (Some(n_classes), Some(imbalance))
        } else {
            (None, None)
        };

        let dimensionality = n_features as f64 / n_samples.max(1) as f64;
        let is_high_dimensional = n_features > n_samples;
        let is_large = n_samples * n_features > 1_000_000;
        let log_size = ((n_samples * n_features) as f64).ln();

        Self {
            n_samples,
            n_features,
            n_numeric_features,
            n_classes,
            dimensionality,
            imbalance_ratio,
            missing_ratio,
            is_high_dimensional,
            is_large,
            log_size,
        }
    }
}

/// Statistical metafeatures capturing distributional properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalMetafeatures {
    /// Mean of feature means
    pub mean_mean: f64,
    /// Std of feature means
    pub mean_std: f64,
    /// Mean of feature standard deviations
    pub std_mean: f64,
    /// Std of feature standard deviations
    pub std_std: f64,
    /// Mean of feature skewness values
    pub skewness_mean: f64,
    /// Std of feature skewness values
    pub skewness_std: f64,
    /// Mean of feature kurtosis values
    pub kurtosis_mean: f64,
    /// Std of feature kurtosis values
    pub kurtosis_std: f64,
    /// Minimum of feature minimums
    pub min_min: f64,
    /// Maximum of feature maximums
    pub max_max: f64,
    /// Ratio of outliers (values > 3 std from mean)
    pub outlier_ratio: f64,
    /// Mean of absolute pairwise correlations
    pub correlation_mean: f64,
    /// Std of pairwise correlations
    pub correlation_std: f64,
    /// Per-feature statistics (if needed for detailed analysis)
    pub per_feature_stats: Vec<FeatureStatistics>,
}

/// Statistics for a single feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStatistics {
    /// Feature index
    pub index: usize,
    /// Mean
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis (excess)
    pub kurtosis: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Number of unique values
    pub n_unique: usize,
    /// Whether feature is constant
    pub is_constant: bool,
}

impl StatisticalMetafeatures {
    /// Extract statistical metafeatures from feature matrix
    pub fn extract(x: &Array2<f64>) -> Self {
        let n_features = x.ncols();
        let _n_samples = x.nrows();

        // Compute per-feature statistics
        let mut per_feature_stats = Vec::with_capacity(n_features);
        let mut means = Vec::with_capacity(n_features);
        let mut stds = Vec::with_capacity(n_features);
        let mut skewnesses = Vec::with_capacity(n_features);
        let mut kurtosises = Vec::with_capacity(n_features);
        let mut mins = Vec::with_capacity(n_features);
        let mut maxs = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col = x.column(j);
            let stats = compute_feature_statistics(&col, j);

            means.push(stats.mean);
            stds.push(stats.std);
            skewnesses.push(stats.skewness);
            kurtosises.push(stats.kurtosis);
            mins.push(stats.min);
            maxs.push(stats.max);

            per_feature_stats.push(stats);
        }

        // Aggregate statistics
        let mean_mean = mean_finite(&means);
        let mean_std = std_finite(&means);
        let std_mean = mean_finite(&stds);
        let std_std = std_finite(&stds);
        let skewness_mean = mean_finite(&skewnesses);
        let skewness_std = std_finite(&skewnesses);
        let kurtosis_mean = mean_finite(&kurtosises);
        let kurtosis_std = std_finite(&kurtosises);

        let min_min = mins
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::INFINITY, f64::min);
        let max_max = maxs
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);

        // Compute outlier ratio
        let outlier_ratio = compute_outlier_ratio(x, &means, &stds);

        // Compute pairwise correlations (if not too many features)
        let (correlation_mean, correlation_std) = if n_features <= 100 && n_features > 1 {
            compute_correlation_stats(x)
        } else if n_features > 1 {
            // For high-dimensional data, sample feature pairs
            compute_sampled_correlation_stats(x, 500)
        } else {
            (0.0, 0.0)
        };

        Self {
            mean_mean,
            mean_std,
            std_mean,
            std_std,
            skewness_mean,
            skewness_std,
            kurtosis_mean,
            kurtosis_std,
            min_min: if min_min.is_finite() { min_min } else { 0.0 },
            max_max: if max_max.is_finite() { max_max } else { 0.0 },
            outlier_ratio,
            correlation_mean,
            correlation_std,
            per_feature_stats,
        }
    }
}

/// Information-theoretic metafeatures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationMetafeatures {
    /// Entropy of the target variable
    pub target_entropy: f64,
    /// Maximum possible entropy for target (log2(n_classes))
    pub max_target_entropy: f64,
    /// Normalized entropy: target_entropy / max_target_entropy
    pub normalized_entropy: f64,
    /// Mean entropy of discretized features
    pub feature_entropy_mean: f64,
    /// Std of feature entropies
    pub feature_entropy_std: f64,
    /// Mean mutual information between features and target
    pub mutual_info_mean: f64,
    /// Std of mutual information values
    pub mutual_info_std: f64,
    /// Maximum mutual information (best single feature)
    pub mutual_info_max: f64,
    /// Sum of mutual information (feature redundancy indicator)
    pub mutual_info_sum: f64,
    /// Equivalent number of features (sum MI / max MI)
    pub equivalent_n_features: f64,
    /// Noise-to-signal ratio estimate
    pub noise_signal_ratio: f64,
}

impl InformationMetafeatures {
    /// Extract information-theoretic metafeatures
    pub fn extract(x: &Array2<f64>, y: &Array1<f64>, is_classification: bool) -> Self {
        let n_features = x.ncols();

        // Compute target entropy
        let (target_entropy, max_target_entropy) = if is_classification {
            let class_counts = count_classes(y);
            let n = y.len() as f64;
            let entropy: f64 = class_counts
                .values()
                .map(|&count| {
                    let p = count as f64 / n;
                    if p > 0.0 {
                        -p * p.log2()
                    } else {
                        0.0
                    }
                })
                .sum();
            let max_entropy = (class_counts.len() as f64).log2();
            (entropy, max_entropy)
        } else {
            // For regression, use discretized target
            let discretized = discretize_array(y, 10);
            let entropy = compute_entropy(&discretized);
            let max_entropy = 10.0_f64.log2();
            (entropy, max_entropy)
        };

        let normalized_entropy = if max_target_entropy > 0.0 {
            target_entropy / max_target_entropy
        } else {
            0.0
        };

        // Compute feature entropies and mutual information
        let mut feature_entropies = Vec::with_capacity(n_features);
        let mut mutual_infos = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col = x.column(j);
            let col_array = col.to_owned();

            // Discretize feature for entropy computation
            let discretized = discretize_array(&col_array, 10);
            let feature_entropy = compute_entropy(&discretized);
            feature_entropies.push(feature_entropy);

            // Compute mutual information
            let mi = compute_mutual_information(&discretized, y, is_classification);
            mutual_infos.push(mi);
        }

        let feature_entropy_mean = mean_finite(&feature_entropies);
        let feature_entropy_std = std_finite(&feature_entropies);
        let mutual_info_mean = mean_finite(&mutual_infos);
        let mutual_info_std = std_finite(&mutual_infos);
        let mutual_info_max = mutual_infos
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0, f64::max);
        let mutual_info_sum = mutual_infos
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .sum::<f64>();

        let equivalent_n_features = if mutual_info_max > 0.0 {
            mutual_info_sum / mutual_info_max
        } else {
            0.0
        };

        // Noise-to-signal ratio: complement of normalized MI
        let noise_signal_ratio = if target_entropy > 0.0 && mutual_info_max > 0.0 {
            1.0 - (mutual_info_max / target_entropy).min(1.0)
        } else {
            1.0
        };

        Self {
            target_entropy,
            max_target_entropy,
            normalized_entropy,
            feature_entropy_mean,
            feature_entropy_std,
            mutual_info_mean,
            mutual_info_std,
            mutual_info_max,
            mutual_info_sum,
            equivalent_n_features,
            noise_signal_ratio,
        }
    }
}

/// Landmarking metafeatures based on performance of simple models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandmarkingMetafeatures {
    /// 1-Nearest Neighbor accuracy/R² (how well local structure predicts)
    pub one_nn_score: f64,
    /// 1-NN accuracy for classification
    pub one_nn_accuracy: f64,
    /// Decision stump (depth-1 tree) accuracy/R² (best single split)
    pub decision_stump_score: f64,
    /// Random decision stump score (baseline)
    pub random_stump_score: f64,
    /// Naive Bayes accuracy (feature independence assumption)
    pub naive_bayes_score: f64,
    /// Linear model score (linearity assumption)
    pub linear_model_score: f64,
    /// Best single feature score
    pub best_feature_score: f64,
    /// Index of best single feature
    pub best_feature_index: usize,
    /// Relative performance: 1-NN vs linear
    pub nn_vs_linear: f64,
    /// Relative performance: NB vs linear
    pub nb_vs_linear: f64,
}

impl LandmarkingMetafeatures {
    /// Extract landmarking metafeatures using simple models
    pub fn extract(
        x: &Array2<f64>,
        y: &Array1<f64>,
        is_classification: bool,
        config: &MetafeatureConfig,
    ) -> Result<Self> {
        let n_samples = x.nrows();

        // Subsample if dataset is too large
        let (x_sub, y_sub) = if n_samples > config.landmarking_max_samples {
            subsample(x, y, config.landmarking_max_samples, config.random_state)
        } else {
            (x.clone(), y.clone())
        };

        if is_classification {
            Self::extract_classification(&x_sub, &y_sub, config)
        } else {
            Self::extract_regression(&x_sub, &y_sub, config)
        }
    }

    fn extract_classification(
        x: &Array2<f64>,
        y: &Array1<f64>,
        config: &MetafeatureConfig,
    ) -> Result<Self> {
        // 1-Nearest Neighbor
        let one_nn_score = cross_validate_classifier(
            || KNeighborsClassifier::new(1).with_weights(crate::models::KNNWeights::Uniform),
            x,
            y,
            config.landmarking_cv_folds,
            config.random_state,
        )?;

        // Decision stump (depth-1 tree)
        let decision_stump_score = cross_validate_classifier(
            || {
                DecisionTreeClassifier::new()
                    .with_max_depth(Some(1))
                    .with_criterion(SplitCriterion::Gini)
            },
            x,
            y,
            config.landmarking_cv_folds,
            config.random_state,
        )?;

        // Random stump (random feature, random split)
        let random_stump_score = cross_validate_classifier(
            || {
                DecisionTreeClassifier::new()
                    .with_max_depth(Some(1))
                    .with_max_features(Some(1))
            },
            x,
            y,
            config.landmarking_cv_folds,
            config.random_state,
        )?;

        // Gaussian Naive Bayes
        let naive_bayes_score = cross_validate_classifier(
            GaussianNB::new,
            x,
            y,
            config.landmarking_cv_folds,
            config.random_state,
        )?;

        // Linear model (depth-3 tree as proxy for linear relationship detection)
        let linear_model_score = cross_validate_classifier(
            || {
                DecisionTreeClassifier::new()
                    .with_max_depth(Some(3))
                    .with_criterion(SplitCriterion::Gini)
            },
            x,
            y,
            config.landmarking_cv_folds,
            config.random_state,
        )?;

        // Best single feature
        let (best_feature_score, best_feature_index) =
            find_best_single_feature_classification(x, y, config)?;

        // Relative performances
        let nn_vs_linear = if linear_model_score > 0.0 {
            one_nn_score / linear_model_score
        } else {
            1.0
        };
        let nb_vs_linear = if linear_model_score > 0.0 {
            naive_bayes_score / linear_model_score
        } else {
            1.0
        };

        Ok(Self {
            one_nn_score,
            one_nn_accuracy: one_nn_score,
            decision_stump_score,
            random_stump_score,
            naive_bayes_score,
            linear_model_score,
            best_feature_score,
            best_feature_index,
            nn_vs_linear,
            nb_vs_linear,
        })
    }

    fn extract_regression(
        x: &Array2<f64>,
        y: &Array1<f64>,
        config: &MetafeatureConfig,
    ) -> Result<Self> {
        // 1-Nearest Neighbor
        let one_nn_score = cross_validate_regressor(
            || KNeighborsRegressor::new(1).with_weights(crate::models::KNNWeights::Uniform),
            x,
            y,
            config.landmarking_cv_folds,
            config.random_state,
        )?;

        // Decision stump (depth-1 tree)
        let decision_stump_score = cross_validate_regressor(
            || DecisionTreeRegressor::new().with_max_depth(Some(1)),
            x,
            y,
            config.landmarking_cv_folds,
            config.random_state,
        )?;

        // Random stump
        let random_stump_score = cross_validate_regressor(
            || {
                DecisionTreeRegressor::new()
                    .with_max_depth(Some(1))
                    .with_max_features(Some(1))
            },
            x,
            y,
            config.landmarking_cv_folds,
            config.random_state,
        )?;

        // For regression, use tree-based proxy for Naive Bayes
        // (simple model assuming feature independence)
        let naive_bayes_score = cross_validate_regressor(
            || DecisionTreeRegressor::new().with_max_depth(Some(2)),
            x,
            y,
            config.landmarking_cv_folds,
            config.random_state,
        )?;

        // Linear model (depth-3 tree for regression)
        let linear_model_score = cross_validate_regressor(
            || DecisionTreeRegressor::new().with_max_depth(Some(3)),
            x,
            y,
            config.landmarking_cv_folds,
            config.random_state,
        )?;

        // Best single feature
        let (best_feature_score, best_feature_index) =
            find_best_single_feature_regression(x, y, config)?;

        // Relative performances
        let nn_vs_linear = if linear_model_score > 0.0 {
            one_nn_score / linear_model_score
        } else {
            1.0
        };
        let nb_vs_linear = if linear_model_score > 0.0 {
            naive_bayes_score / linear_model_score
        } else {
            1.0
        };

        Ok(Self {
            one_nn_score,
            one_nn_accuracy: one_nn_score, // For regression, this is R²
            decision_stump_score,
            random_stump_score,
            naive_bayes_score,
            linear_model_score,
            best_feature_score,
            best_feature_index,
            nn_vs_linear,
            nb_vs_linear,
        })
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Count occurrences of each class in target array
fn count_classes(y: &Array1<f64>) -> HashMap<i64, usize> {
    let mut counts = HashMap::new();
    for &val in y.iter() {
        let key = val as i64;
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

/// Compute statistics for a single feature column
fn compute_feature_statistics(col: &ndarray::ArrayView1<f64>, index: usize) -> FeatureStatistics {
    let n = col.len();
    if n == 0 {
        return FeatureStatistics {
            index,
            mean: 0.0,
            std: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            min: 0.0,
            max: 0.0,
            n_unique: 0,
            is_constant: true,
        };
    }

    // Filter out NaN values for statistics
    let valid: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
    let n_valid = valid.len();

    if n_valid == 0 {
        return FeatureStatistics {
            index,
            mean: f64::NAN,
            std: f64::NAN,
            skewness: f64::NAN,
            kurtosis: f64::NAN,
            min: f64::NAN,
            max: f64::NAN,
            n_unique: 0,
            is_constant: true,
        };
    }

    let mean = valid.iter().sum::<f64>() / n_valid as f64;

    let variance = if n_valid > 1 {
        valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n_valid - 1) as f64
    } else {
        0.0
    };
    let std = variance.sqrt();

    // Skewness and kurtosis
    let (skewness, kurtosis) = if std > 1e-10 && n_valid > 2 {
        let m3 = valid
            .iter()
            .map(|x| ((x - mean) / std).powi(3))
            .sum::<f64>()
            / n_valid as f64;
        let m4 = valid
            .iter()
            .map(|x| ((x - mean) / std).powi(4))
            .sum::<f64>()
            / n_valid as f64;
        (m3, m4 - 3.0) // Excess kurtosis
    } else {
        (0.0, 0.0)
    };

    let min = valid.iter().copied().fold(f64::INFINITY, f64::min);
    let max = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Count unique values (rounded for numerical stability)
    let mut unique: Vec<i64> = valid.iter().map(|x| (x * 1e6).round() as i64).collect();
    unique.sort();
    unique.dedup();
    let n_unique = unique.len();

    let is_constant = n_unique <= 1 || std < 1e-10;

    FeatureStatistics {
        index,
        mean,
        std,
        skewness,
        kurtosis,
        min,
        max,
        n_unique,
        is_constant,
    }
}

/// Compute mean of finite values
fn mean_finite(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        0.0
    } else {
        finite.iter().sum::<f64>() / finite.len() as f64
    }
}

/// Compute std of finite values
fn std_finite(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let n = finite.len();
    if n <= 1 {
        return 0.0;
    }
    let mean = finite.iter().sum::<f64>() / n as f64;
    let variance = finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    variance.sqrt()
}

/// Compute ratio of outliers (values > 3 std from mean)
fn compute_outlier_ratio(x: &Array2<f64>, means: &[f64], stds: &[f64]) -> f64 {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let total = n_samples * n_features;

    if total == 0 {
        return 0.0;
    }

    let mut outlier_count = 0;
    for j in 0..n_features {
        let mean = means[j];
        let std = stds[j];
        if std < 1e-10 {
            continue;
        }
        for i in 0..n_samples {
            let val = x[[i, j]];
            if val.is_finite() && (val - mean).abs() > 3.0 * std {
                outlier_count += 1;
            }
        }
    }

    outlier_count as f64 / total as f64
}

/// Compute pairwise correlation statistics
fn compute_correlation_stats(x: &Array2<f64>) -> (f64, f64) {
    let n_features = x.ncols();
    let mut correlations = Vec::new();

    for i in 0..n_features {
        for j in (i + 1)..n_features {
            let corr = pearson_correlation(&x.column(i), &x.column(j));
            if corr.is_finite() {
                correlations.push(corr.abs());
            }
        }
    }

    if correlations.is_empty() {
        return (0.0, 0.0);
    }

    (mean_finite(&correlations), std_finite(&correlations))
}

/// Compute sampled pairwise correlation statistics for high-dimensional data
fn compute_sampled_correlation_stats(x: &Array2<f64>, n_pairs: usize) -> (f64, f64) {
    let n_features = x.ncols();
    if n_features < 2 {
        return (0.0, 0.0);
    }

    let mut correlations = Vec::with_capacity(n_pairs);
    let mut rng = SimpleRng::new(42);

    for _ in 0..n_pairs {
        let i = rng.next_usize() % n_features;
        let mut j = rng.next_usize() % n_features;
        while j == i {
            j = rng.next_usize() % n_features;
        }

        let corr = pearson_correlation(&x.column(i), &x.column(j));
        if corr.is_finite() {
            correlations.push(corr.abs());
        }
    }

    if correlations.is_empty() {
        return (0.0, 0.0);
    }

    (mean_finite(&correlations), std_finite(&correlations))
}

/// Compute Pearson correlation coefficient
fn pearson_correlation(x: &ndarray::ArrayView1<f64>, y: &ndarray::ArrayView1<f64>) -> f64 {
    let n = x.len();
    if n == 0 || x.len() != y.len() {
        return 0.0;
    }

    // Filter pairs where both are finite
    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(&a, &b)| (a, b))
        .collect();

    let n = pairs.len();
    if n < 3 {
        return 0.0;
    }

    let mean_x = pairs.iter().map(|(a, _)| a).sum::<f64>() / n as f64;
    let mean_y = pairs.iter().map(|(_, b)| b).sum::<f64>() / n as f64;

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for (a, b) in pairs {
        let dx = a - mean_x;
        let dy = b - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }

    let denom = (sum_x2 * sum_y2).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        sum_xy / denom
    }
}

/// Discretize continuous array into bins
fn discretize_array(values: &Array1<f64>, n_bins: usize) -> Vec<usize> {
    let valid: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if valid.is_empty() {
        return vec![0; values.len()];
    }

    let min = valid.iter().copied().fold(f64::INFINITY, f64::min);
    let max = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < 1e-10 {
        return vec![0; values.len()];
    }

    let bin_width = (max - min) / n_bins as f64;

    values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                0
            } else {
                let bin = ((v - min) / bin_width).floor() as usize;
                bin.min(n_bins - 1)
            }
        })
        .collect()
}

/// Compute entropy of discrete values
fn compute_entropy(values: &[usize]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }

    let mut counts = HashMap::new();
    for &v in values {
        *counts.entry(v).or_insert(0usize) += 1;
    }

    counts
        .values()
        .map(|&count| {
            let p = count as f64 / n as f64;
            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum()
}

/// Compute mutual information between discretized feature and target
fn compute_mutual_information(
    feature: &[usize],
    target: &Array1<f64>,
    is_classification: bool,
) -> f64 {
    let n = feature.len();
    if n == 0 || n != target.len() {
        return 0.0;
    }

    // Discretize target if regression
    let target_discrete: Vec<usize> = if is_classification {
        target.iter().map(|&v| v as usize).collect()
    } else {
        discretize_array(target, 10)
    };

    // Count joint and marginal frequencies
    let mut joint_counts: HashMap<(usize, usize), usize> = HashMap::new();
    let mut feature_counts: HashMap<usize, usize> = HashMap::new();
    let mut target_counts: HashMap<usize, usize> = HashMap::new();

    for i in 0..n {
        let f = feature[i];
        let t = target_discrete[i];
        *joint_counts.entry((f, t)).or_insert(0) += 1;
        *feature_counts.entry(f).or_insert(0) += 1;
        *target_counts.entry(t).or_insert(0) += 1;
    }

    // Compute MI
    let n_f64 = n as f64;
    let mut mi = 0.0;

    for ((f, t), &joint_count) in &joint_counts {
        let p_ft = joint_count as f64 / n_f64;
        let p_f = feature_counts[f] as f64 / n_f64;
        let p_t = target_counts[t] as f64 / n_f64;

        if p_ft > 0.0 && p_f > 0.0 && p_t > 0.0 {
            mi += p_ft * (p_ft / (p_f * p_t)).log2();
        }
    }

    mi.max(0.0)
}

/// Subsample dataset for large datasets
fn subsample(
    x: &Array2<f64>,
    y: &Array1<f64>,
    max_samples: usize,
    random_state: Option<u64>,
) -> (Array2<f64>, Array1<f64>) {
    let n_samples = x.nrows();
    if n_samples <= max_samples {
        return (x.clone(), y.clone());
    }

    let mut rng = SimpleRng::new(random_state.unwrap_or(42));
    let mut indices: Vec<usize> = (0..n_samples).collect();

    // Fisher-Yates shuffle
    for i in (1..n_samples).rev() {
        let j = rng.next_usize() % (i + 1);
        indices.swap(i, j);
    }

    let selected: Vec<usize> = indices.into_iter().take(max_samples).collect();

    let x_sub = Array2::from_shape_fn((max_samples, x.ncols()), |(i, j)| x[[selected[i], j]]);
    let y_sub = Array1::from_shape_fn(max_samples, |i| y[selected[i]]);

    (x_sub, y_sub)
}

/// Simple PRNG for reproducibility
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }
}

/// Cross-validate a classifier with simple k-fold
fn cross_validate_classifier<F, M>(
    model_fn: F,
    x: &Array2<f64>,
    y: &Array1<f64>,
    n_folds: usize,
    random_state: Option<u64>,
) -> Result<f64>
where
    F: Fn() -> M,
    M: Model,
{
    let n_samples = x.nrows();
    if n_samples < n_folds {
        return Ok(0.0);
    }

    // Create shuffled indices
    let mut rng = SimpleRng::new(random_state.unwrap_or(42));
    let mut indices: Vec<usize> = (0..n_samples).collect();
    for i in (1..n_samples).rev() {
        let j = rng.next_usize() % (i + 1);
        indices.swap(i, j);
    }

    let fold_size = n_samples / n_folds;
    let mut scores = Vec::with_capacity(n_folds);

    for fold in 0..n_folds {
        let test_start = fold * fold_size;
        let test_end = if fold == n_folds - 1 {
            n_samples
        } else {
            (fold + 1) * fold_size
        };

        // Split indices
        let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
        let train_indices: Vec<usize> = indices[..test_start]
            .iter()
            .chain(indices[test_end..].iter())
            .copied()
            .collect();

        if train_indices.is_empty() || test_indices.is_empty() {
            continue;
        }

        // Create train/test sets
        let x_train = Array2::from_shape_fn((train_indices.len(), x.ncols()), |(i, j)| {
            x[[train_indices[i], j]]
        });
        let y_train = Array1::from_shape_fn(train_indices.len(), |i| y[train_indices[i]]);
        let x_test = Array2::from_shape_fn((test_indices.len(), x.ncols()), |(i, j)| {
            x[[test_indices[i], j]]
        });
        let y_test = Array1::from_shape_fn(test_indices.len(), |i| y[test_indices[i]]);

        // Train and evaluate
        let mut model = model_fn();
        if model.fit(&x_train, &y_train).is_err() {
            continue;
        }

        match model.predict(&x_test) {
            Ok(predictions) => {
                // Compute accuracy
                let correct = predictions
                    .iter()
                    .zip(y_test.iter())
                    .filter(|(pred, actual)| (**pred - **actual).abs() < 0.5)
                    .count();
                scores.push(correct as f64 / test_indices.len() as f64);
            }
            Err(_) => continue,
        }
    }

    if scores.is_empty() {
        Ok(0.0)
    } else {
        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }
}

/// Cross-validate a regressor with simple k-fold
fn cross_validate_regressor<F, M>(
    model_fn: F,
    x: &Array2<f64>,
    y: &Array1<f64>,
    n_folds: usize,
    random_state: Option<u64>,
) -> Result<f64>
where
    F: Fn() -> M,
    M: Model,
{
    let n_samples = x.nrows();
    if n_samples < n_folds {
        return Ok(0.0);
    }

    // Create shuffled indices
    let mut rng = SimpleRng::new(random_state.unwrap_or(42));
    let mut indices: Vec<usize> = (0..n_samples).collect();
    for i in (1..n_samples).rev() {
        let j = rng.next_usize() % (i + 1);
        indices.swap(i, j);
    }

    let fold_size = n_samples / n_folds;
    let mut scores = Vec::with_capacity(n_folds);

    for fold in 0..n_folds {
        let test_start = fold * fold_size;
        let test_end = if fold == n_folds - 1 {
            n_samples
        } else {
            (fold + 1) * fold_size
        };

        // Split indices
        let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
        let train_indices: Vec<usize> = indices[..test_start]
            .iter()
            .chain(indices[test_end..].iter())
            .copied()
            .collect();

        if train_indices.is_empty() || test_indices.is_empty() {
            continue;
        }

        // Create train/test sets
        let x_train = Array2::from_shape_fn((train_indices.len(), x.ncols()), |(i, j)| {
            x[[train_indices[i], j]]
        });
        let y_train = Array1::from_shape_fn(train_indices.len(), |i| y[train_indices[i]]);
        let x_test = Array2::from_shape_fn((test_indices.len(), x.ncols()), |(i, j)| {
            x[[test_indices[i], j]]
        });
        let y_test = Array1::from_shape_fn(test_indices.len(), |i| y[test_indices[i]]);

        // Train and evaluate
        let mut model = model_fn();
        if model.fit(&x_train, &y_train).is_err() {
            continue;
        }

        match model.predict(&x_test) {
            Ok(predictions) => {
                // Compute R²
                let y_mean = y_test.mean().unwrap_or(0.0);
                let ss_tot: f64 = y_test.iter().map(|y| (y - y_mean).powi(2)).sum();
                let ss_res: f64 = predictions
                    .iter()
                    .zip(y_test.iter())
                    .map(|(pred, actual)| (actual - pred).powi(2))
                    .sum();

                let r2 = if ss_tot > 1e-10 {
                    1.0 - ss_res / ss_tot
                } else {
                    0.0
                };
                scores.push(r2.max(0.0)); // Clamp negative R² to 0
            }
            Err(_) => continue,
        }
    }

    if scores.is_empty() {
        Ok(0.0)
    } else {
        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }
}

/// Find the best single feature for classification
fn find_best_single_feature_classification(
    x: &Array2<f64>,
    y: &Array1<f64>,
    config: &MetafeatureConfig,
) -> Result<(f64, usize)> {
    let n_features = x.ncols();
    let mut best_score = 0.0;
    let mut best_index = 0;

    for j in 0..n_features {
        let x_single = x.column(j).to_owned().insert_axis(Axis(1));
        let score = cross_validate_classifier(
            || DecisionTreeClassifier::new().with_max_depth(Some(3)),
            &x_single,
            y,
            config.landmarking_cv_folds,
            config.random_state,
        )?;

        if score > best_score {
            best_score = score;
            best_index = j;
        }
    }

    Ok((best_score, best_index))
}

/// Find the best single feature for regression
fn find_best_single_feature_regression(
    x: &Array2<f64>,
    y: &Array1<f64>,
    config: &MetafeatureConfig,
) -> Result<(f64, usize)> {
    let n_features = x.ncols();
    let mut best_score = f64::NEG_INFINITY;
    let mut best_index = 0;

    for j in 0..n_features {
        let x_single = x.column(j).to_owned().insert_axis(Axis(1));
        let score = cross_validate_regressor(
            || DecisionTreeRegressor::new().with_max_depth(Some(3)),
            &x_single,
            y,
            config.landmarking_cv_folds,
            config.random_state,
        )?;

        if score > best_score {
            best_score = score;
            best_index = j;
        }
    }

    Ok((best_score.max(0.0), best_index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_classification_data() -> (Array2<f64>, Array1<f64>) {
        // Simple linearly separable data
        let x = Array2::from_shape_vec(
            (20, 3),
            vec![
                // Class 0
                1.0, 2.0, 0.5, 1.5, 2.5, 0.3, 0.8, 1.8, 0.6, 1.2, 2.2, 0.4, 1.1, 2.1, 0.55, 0.9,
                1.9, 0.45, 1.3, 2.3, 0.35, 1.4, 2.4, 0.65, 0.7, 1.7, 0.7, 1.6, 2.6, 0.25,
                // Class 1
                4.0, 5.0, 0.9, 4.5, 5.5, 0.85, 3.8, 4.8, 0.95, 4.2, 5.2, 0.88, 4.1, 5.1, 0.92, 3.9,
                4.9, 0.87, 4.3, 5.3, 0.93, 4.4, 5.4, 0.91, 3.7, 4.7, 0.96, 4.6, 5.6, 0.86,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]);
        (x, y)
    }

    fn create_test_regression_data() -> (Array2<f64>, Array1<f64>) {
        // Simple linear relationship: y = 2*x1 + 0.5*x2 + noise
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5,
                4.5, 4.5, 5.5, 5.5, 1.2, 0.8, 2.2, 1.8, 3.2, 2.8, 4.2, 3.8, 5.2, 4.8, 1.8, 1.2,
                2.8, 2.2, 3.8, 3.2, 4.8, 4.2, 5.8, 5.2,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            2.5, 4.5, 6.5, 8.5, 10.5, 3.75, 5.75, 7.75, 9.75, 11.75, 2.8, 5.3, 7.8, 10.3, 12.8,
            4.2, 6.7, 9.2, 11.7, 14.2,
        ]);
        (x, y)
    }

    #[test]
    fn test_simple_metafeatures() {
        let (x, y) = create_test_classification_data();
        let simple = SimpleMetafeatures::extract(&x, &y, true);

        assert_eq!(simple.n_samples, 20);
        assert_eq!(simple.n_features, 3);
        assert_eq!(simple.n_classes, Some(2));
        assert_relative_eq!(simple.dimensionality, 0.15, epsilon = 0.01);
        assert!(simple.imbalance_ratio.unwrap() >= 1.0);
        assert_relative_eq!(simple.missing_ratio, 0.0, epsilon = 1e-10);
        assert!(!simple.is_high_dimensional);
    }

    #[test]
    fn test_statistical_metafeatures() {
        let (x, _) = create_test_classification_data();
        let stats = StatisticalMetafeatures::extract(&x);

        // Should have computed stats for each feature
        assert_eq!(stats.per_feature_stats.len(), 3);

        // Mean and std should be reasonable
        assert!(stats.mean_mean.is_finite());
        assert!(stats.std_mean.is_finite());
        assert!(stats.std_mean > 0.0);
    }

    #[test]
    fn test_information_metafeatures() {
        let (x, y) = create_test_classification_data();
        let info = InformationMetafeatures::extract(&x, &y, true);

        // Target entropy for binary classification should be <= 1.0
        assert!(info.target_entropy >= 0.0);
        assert!(info.target_entropy <= 1.0);
        assert_relative_eq!(info.max_target_entropy, 1.0, epsilon = 0.01);

        // Feature entropies should be non-negative
        assert!(info.feature_entropy_mean >= 0.0);

        // Mutual information should be non-negative
        assert!(info.mutual_info_mean >= 0.0);
    }

    #[test]
    fn test_landmarking_classification() {
        let (x, y) = create_test_classification_data();
        let config = MetafeatureConfig::new()
            .with_landmarking_cv_folds(2)
            .with_random_state(42);

        let landmarking =
            LandmarkingMetafeatures::extract(&x, &y, true, &config).expect("Landmarking failed");

        // For well-separated classes, 1-NN should do well
        assert!(landmarking.one_nn_score >= 0.0);
        assert!(landmarking.one_nn_score <= 1.0);

        // Decision stump should have some predictive power
        assert!(landmarking.decision_stump_score >= 0.0);
    }

    #[test]
    fn test_landmarking_regression() {
        let (x, y) = create_test_regression_data();
        let config = MetafeatureConfig::new()
            .with_landmarking_cv_folds(2)
            .with_random_state(42);

        let landmarking =
            LandmarkingMetafeatures::extract(&x, &y, false, &config).expect("Landmarking failed");

        // R² should be non-negative (we clamp it)
        assert!(landmarking.one_nn_score >= 0.0);
        assert!(landmarking.decision_stump_score >= 0.0);
    }

    #[test]
    fn test_full_extraction() {
        let (x, y) = create_test_classification_data();
        let config = MetafeatureConfig::new()
            .with_landmarking_cv_folds(2)
            .with_random_state(42);

        let metafeatures =
            DatasetMetafeatures::extract(&x, &y, true, Some(config)).expect("Extraction failed");

        assert_eq!(metafeatures.simple.n_samples, 20);
        assert!(metafeatures.statistical.mean_mean.is_finite());
        assert!(metafeatures.information.target_entropy >= 0.0);
        assert!(metafeatures.landmarking.is_some());
    }

    #[test]
    fn test_without_landmarking() {
        let (x, y) = create_test_classification_data();
        let config = MetafeatureConfig::new().without_landmarking();

        let metafeatures =
            DatasetMetafeatures::extract(&x, &y, true, Some(config)).expect("Extraction failed");

        assert!(metafeatures.landmarking.is_none());
    }

    #[test]
    fn test_to_vector() {
        let (x, y) = create_test_classification_data();
        let config = MetafeatureConfig::new().without_landmarking();

        let metafeatures =
            DatasetMetafeatures::extract(&x, &y, true, Some(config)).expect("Extraction failed");

        let vec = metafeatures.to_vector();
        assert!(!vec.is_empty());
        // All values should be finite
        assert!(vec.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_similarity() {
        let (x, y) = create_test_classification_data();
        let config = MetafeatureConfig::new().without_landmarking();

        let mf1 =
            DatasetMetafeatures::extract(&x, &y, true, Some(config.clone())).expect("Failed mf1");
        let mf2 = DatasetMetafeatures::extract(&x, &y, true, Some(config)).expect("Failed mf2");

        // Same dataset should have similarity ~1.0
        let sim = mf1.similarity(&mf2);
        assert_relative_eq!(sim, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_discretize_array() {
        let values = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let bins = discretize_array(&values, 4);

        // First value should be bin 0, last should be bin 3
        assert_eq!(bins[0], 0);
        assert_eq!(bins[4], 3);
    }

    #[test]
    fn test_compute_entropy() {
        // Uniform distribution has maximum entropy
        let uniform = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let entropy_uniform = compute_entropy(&uniform);
        assert!(entropy_uniform > 1.5); // log2(4) = 2.0

        // Single value has zero entropy
        let single = vec![0, 0, 0, 0];
        let entropy_single = compute_entropy(&single);
        assert_relative_eq!(entropy_single, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let corr = pearson_correlation(&x.view(), &y.view());
        assert_relative_eq!(corr, 1.0, epsilon = 1e-10);

        // Negative correlation
        let y_neg = Array1::from_vec(vec![10.0, 8.0, 6.0, 4.0, 2.0]);
        let corr_neg = pearson_correlation(&x.view(), &y_neg.view());
        assert_relative_eq!(corr_neg, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_count_classes() {
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let counts = count_classes(&y);

        assert_eq!(counts.get(&0), Some(&2));
        assert_eq!(counts.get(&1), Some(&3));
        assert_eq!(counts.get(&2), Some(&1));
    }

    #[test]
    fn test_subsample() {
        let x = Array2::from_shape_fn((100, 3), |(i, j)| (i * 3 + j) as f64);
        let y = Array1::from_shape_fn(100, |i| (i % 2) as f64);

        let (x_sub, y_sub) = subsample(&x, &y, 50, Some(42));

        assert_eq!(x_sub.nrows(), 50);
        assert_eq!(y_sub.len(), 50);
        assert_eq!(x_sub.ncols(), 3);
    }

    #[test]
    fn test_feature_statistics() {
        let col = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = compute_feature_statistics(&col.view(), 0);

        assert_eq!(stats.index, 0);
        assert_relative_eq!(stats.mean, 3.0, epsilon = 0.01);
        assert!(stats.std > 0.0);
        assert_relative_eq!(stats.min, 1.0, epsilon = 1e-10);
        assert_relative_eq!(stats.max, 5.0, epsilon = 1e-10);
        assert!(!stats.is_constant);
    }

    #[test]
    fn test_mutual_information() {
        // Perfect mutual information: feature exactly predicts target
        let feature = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let target = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mi = compute_mutual_information(&feature, &target, true);
        assert!(mi > 0.9); // Should be close to 1.0
    }
}

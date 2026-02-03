//! Warm-Starting from Similar Datasets for Meta-Learning
//!
//! This module provides warm-starting capabilities for AutoML by:
//! - Storing metafeatures and best configurations from previous runs
//! - Finding similar datasets using metafeature similarity
//! - Initializing HPO with configurations that worked well on similar datasets
//!
//! # Example
//!
//! ```ignore
//! use ferroml_core::automl::{
//!     MetaLearningStore, WarmStartConfig, DatasetMetafeatures, MetafeatureConfig,
//! };
//!
//! // Create or load a meta-learning store
//! let mut store = MetaLearningStore::new();
//!
//! // After running AutoML, store the results
//! store.add_dataset("my_dataset", metafeatures, &trial_results);
//!
//! // For a new dataset, get warm-start configurations
//! let new_metafeatures = DatasetMetafeatures::extract(&x, &y, true, None)?;
//! let config = WarmStartConfig::new()
//!     .with_k_nearest(3)
//!     .with_min_similarity(0.5);
//!
//! let warm_start = store.get_warm_start_configs(&new_metafeatures, &config)?;
//! println!("Found {} initial configurations from {} similar datasets",
//!          warm_start.configurations.len(), warm_start.similar_datasets.len());
//! ```

use crate::automl::{AlgorithmType, DatasetMetafeatures, ParamValue, TrialResult};
use crate::{FerroError, Result, Task};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for warm-starting
#[derive(Debug, Clone)]
pub struct WarmStartConfig {
    /// Number of nearest datasets to consider
    pub k_nearest: usize,
    /// Minimum similarity threshold (0.0 to 1.0)
    pub min_similarity: f64,
    /// Maximum configurations to return per similar dataset
    pub max_configs_per_dataset: usize,
    /// Maximum total configurations to return
    pub max_total_configs: usize,
    /// Weight by similarity when selecting configs
    pub weight_by_similarity: bool,
    /// Normalize metafeatures before similarity computation
    pub normalize_metafeatures: bool,
    /// Whether to compute landmarking features for faster extraction
    pub compute_landmarking: bool,
}

impl Default for WarmStartConfig {
    fn default() -> Self {
        Self {
            k_nearest: 5,
            min_similarity: 0.3,
            max_configs_per_dataset: 5,
            max_total_configs: 25,
            weight_by_similarity: true,
            normalize_metafeatures: true,
            compute_landmarking: false, // Faster by default
        }
    }
}

impl WarmStartConfig {
    /// Create a new warm-start configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of nearest datasets to consider
    pub fn with_k_nearest(mut self, k: usize) -> Self {
        self.k_nearest = k.max(1);
        self
    }

    /// Set the minimum similarity threshold
    pub fn with_min_similarity(mut self, threshold: f64) -> Self {
        self.min_similarity = threshold.max(0.0).min(1.0);
        self
    }

    /// Set maximum configurations per dataset
    pub fn with_max_configs_per_dataset(mut self, max: usize) -> Self {
        self.max_configs_per_dataset = max.max(1);
        self
    }

    /// Set maximum total configurations
    pub fn with_max_total_configs(mut self, max: usize) -> Self {
        self.max_total_configs = max.max(1);
        self
    }

    /// Enable/disable similarity weighting
    pub fn with_weight_by_similarity(mut self, enable: bool) -> Self {
        self.weight_by_similarity = enable;
        self
    }

    /// Enable/disable metafeature normalization
    pub fn with_normalize_metafeatures(mut self, enable: bool) -> Self {
        self.normalize_metafeatures = enable;
        self
    }

    /// Enable/disable landmarking computation
    pub fn with_landmarking(mut self, enable: bool) -> Self {
        self.compute_landmarking = enable;
        self
    }
}

/// A stored configuration record from a previous dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationRecord {
    /// Algorithm type
    pub algorithm: AlgorithmType,
    /// Hyperparameter values
    pub params: HashMap<String, ParamValue>,
    /// Cross-validation score achieved
    pub cv_score: f64,
    /// Standard deviation of CV scores
    pub cv_std: f64,
    /// Whether higher scores are better
    pub maximize: bool,
    /// Rank within dataset results (1 = best)
    pub rank: usize,
}

impl ConfigurationRecord {
    /// Create a new configuration record
    pub fn new(
        algorithm: AlgorithmType,
        params: HashMap<String, ParamValue>,
        cv_score: f64,
        cv_std: f64,
        maximize: bool,
        rank: usize,
    ) -> Self {
        Self {
            algorithm,
            params,
            cv_score,
            cv_std,
            maximize,
            rank,
        }
    }

    /// Create from a trial result
    pub fn from_trial(trial: &TrialResult, maximize: bool, rank: usize) -> Self {
        Self {
            algorithm: trial.algorithm,
            params: trial.params.clone(),
            cv_score: trial.cv_score,
            cv_std: trial.cv_std,
            maximize,
            rank,
        }
    }
}

/// A stored dataset record containing metafeatures and best configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRecord {
    /// Unique identifier for the dataset
    pub name: String,
    /// Dataset metafeatures
    pub metafeatures: DatasetMetafeatures,
    /// Task type (classification/regression)
    pub task: Task,
    /// Best configurations from previous runs
    pub configurations: Vec<ConfigurationRecord>,
    /// Timestamp when record was created
    pub created_at: u64,
    /// Timestamp when record was last updated
    pub updated_at: u64,
}

impl DatasetRecord {
    /// Create a new dataset record
    pub fn new(name: impl Into<String>, metafeatures: DatasetMetafeatures, task: Task) -> Self {
        let now = current_timestamp();
        Self {
            name: name.into(),
            metafeatures,
            task,
            configurations: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add configurations from trial results
    pub fn add_configurations(&mut self, trials: &[TrialResult], maximize: bool) {
        // Sort trials by score (best first)
        let mut sorted_trials: Vec<_> = trials.iter().filter(|t| t.success).collect();
        sorted_trials.sort_by(|a, b| {
            let cmp = a.cv_score.partial_cmp(&b.cv_score).unwrap();
            if maximize {
                cmp.reverse()
            } else {
                cmp
            }
        });

        // Add each configuration with its rank
        for (rank, trial) in sorted_trials.iter().enumerate() {
            let config = ConfigurationRecord::from_trial(trial, maximize, rank + 1);
            self.configurations.push(config);
        }

        self.updated_at = current_timestamp();
    }

    /// Get top N configurations
    pub fn top_configurations(&self, n: usize) -> Vec<&ConfigurationRecord> {
        self.configurations.iter().take(n).collect()
    }
}

/// Information about a similar dataset found during warm-starting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarDataset {
    /// Dataset name
    pub name: String,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f64,
    /// Number of configurations available
    pub n_configurations: usize,
    /// Task type
    pub task: Task,
}

/// Result of warm-starting with configurations from similar datasets
#[derive(Debug, Clone)]
pub struct WarmStartResult {
    /// Similar datasets found (sorted by similarity, most similar first)
    pub similar_datasets: Vec<SimilarDataset>,
    /// Initial configurations to try (weighted by similarity)
    pub configurations: Vec<WeightedConfiguration>,
    /// Total number of configurations found
    pub total_configs_found: usize,
    /// Average similarity of similar datasets
    pub mean_similarity: f64,
}

impl WarmStartResult {
    /// Check if warm-start found useful configurations
    pub fn has_configurations(&self) -> bool {
        !self.configurations.is_empty()
    }

    /// Get configurations grouped by algorithm
    pub fn configurations_by_algorithm(
        &self,
    ) -> HashMap<AlgorithmType, Vec<&WeightedConfiguration>> {
        let mut grouped: HashMap<AlgorithmType, Vec<&WeightedConfiguration>> = HashMap::new();
        for config in &self.configurations {
            grouped
                .entry(config.config.algorithm)
                .or_default()
                .push(config);
        }
        grouped
    }

    /// Get the algorithms to prioritize based on warm-start results
    pub fn prioritized_algorithms(&self) -> Vec<AlgorithmType> {
        // Count algorithm occurrences weighted by priority
        let mut algo_scores: HashMap<AlgorithmType, f64> = HashMap::new();
        for config in &self.configurations {
            let score = config.priority_weight;
            *algo_scores.entry(config.config.algorithm).or_default() += score;
        }

        // Sort by score
        let mut algos: Vec<_> = algo_scores.into_iter().collect();
        algos.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        algos.into_iter().map(|(algo, _)| algo).collect()
    }
}

/// A configuration with associated weight from warm-starting
#[derive(Debug, Clone)]
pub struct WeightedConfiguration {
    /// The configuration record
    pub config: ConfigurationRecord,
    /// Dataset similarity that provided this config
    pub source_similarity: f64,
    /// Dataset name that provided this config
    pub source_dataset: String,
    /// Combined priority weight for ordering
    pub priority_weight: f64,
}

/// In-memory store for dataset metafeatures and configurations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetaLearningStore {
    /// Stored dataset records
    datasets: HashMap<String, DatasetRecord>,
    /// Normalization parameters (computed from all stored datasets)
    #[serde(skip)]
    normalization_params: Option<NormalizationParams>,
}

/// Parameters for normalizing metafeature vectors
#[derive(Debug, Clone)]
struct NormalizationParams {
    /// Mean of each metafeature
    means: Vec<f64>,
    /// Standard deviation of each metafeature
    stds: Vec<f64>,
}

impl MetaLearningStore {
    /// Create a new empty meta-learning store
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
            normalization_params: None,
        }
    }

    /// Get the number of stored datasets
    pub fn len(&self) -> usize {
        self.datasets.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.datasets.is_empty()
    }

    /// Add a dataset record to the store
    pub fn add_record(&mut self, record: DatasetRecord) {
        self.datasets.insert(record.name.clone(), record);
        self.normalization_params = None; // Invalidate cached params
    }

    /// Add a dataset with metafeatures and trial results
    pub fn add_dataset(
        &mut self,
        name: impl Into<String>,
        metafeatures: DatasetMetafeatures,
        task: Task,
        trials: &[TrialResult],
        maximize: bool,
    ) {
        let name = name.into();
        let mut record = DatasetRecord::new(name, metafeatures, task);
        record.add_configurations(trials, maximize);
        self.add_record(record);
    }

    /// Get a dataset record by name
    pub fn get(&self, name: &str) -> Option<&DatasetRecord> {
        self.datasets.get(name)
    }

    /// Remove a dataset record
    pub fn remove(&mut self, name: &str) -> Option<DatasetRecord> {
        let result = self.datasets.remove(name);
        if result.is_some() {
            self.normalization_params = None;
        }
        result
    }

    /// Get all dataset names
    pub fn dataset_names(&self) -> Vec<&str> {
        self.datasets.keys().map(|s| s.as_str()).collect()
    }

    /// Find similar datasets based on metafeature similarity
    ///
    /// # Arguments
    /// * `query_metafeatures` - Metafeatures of the query dataset
    /// * `config` - Warm-start configuration
    /// * `task_filter` - Optional task type to filter by
    ///
    /// # Returns
    /// Vector of (similarity, dataset_record) pairs, sorted by similarity (highest first)
    pub fn find_similar(
        &self,
        query_metafeatures: &DatasetMetafeatures,
        config: &WarmStartConfig,
        task_filter: Option<Task>,
    ) -> Vec<(f64, &DatasetRecord)> {
        if self.is_empty() {
            return Vec::new();
        }

        // Get query vector
        let query_vec = if config.normalize_metafeatures {
            self.normalize_vector(&query_metafeatures.to_vector())
        } else {
            query_metafeatures.to_vector()
        };

        // Compute similarity to all stored datasets
        let mut similarities: Vec<(f64, &DatasetRecord)> = self
            .datasets
            .values()
            .filter(|record| task_filter.is_none() || task_filter == Some(record.task))
            .map(|record| {
                let stored_vec = if config.normalize_metafeatures {
                    self.normalize_vector(&record.metafeatures.to_vector())
                } else {
                    record.metafeatures.to_vector()
                };
                let sim = cosine_similarity(&query_vec, &stored_vec);
                (sim, record)
            })
            .filter(|(sim, _)| *sim >= config.min_similarity)
            .collect();

        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take top k
        similarities.truncate(config.k_nearest);

        similarities
    }

    /// Get warm-start configurations from similar datasets
    ///
    /// # Arguments
    /// * `query_metafeatures` - Metafeatures of the new dataset
    /// * `config` - Warm-start configuration
    /// * `task` - Task type (classification/regression)
    ///
    /// # Returns
    /// `WarmStartResult` with configurations weighted by similarity
    pub fn get_warm_start_configs(
        &self,
        query_metafeatures: &DatasetMetafeatures,
        config: &WarmStartConfig,
        task: Task,
    ) -> Result<WarmStartResult> {
        let similar = self.find_similar(query_metafeatures, config, Some(task));

        if similar.is_empty() {
            return Ok(WarmStartResult {
                similar_datasets: Vec::new(),
                configurations: Vec::new(),
                total_configs_found: 0,
                mean_similarity: 0.0,
            });
        }

        // Build similar datasets info
        let similar_datasets: Vec<SimilarDataset> = similar
            .iter()
            .map(|(sim, record)| SimilarDataset {
                name: record.name.clone(),
                similarity: *sim,
                n_configurations: record.configurations.len(),
                task: record.task,
            })
            .collect();

        // Collect weighted configurations
        let mut weighted_configs: Vec<WeightedConfiguration> = Vec::new();
        let mut total_configs = 0;

        for (similarity, record) in &similar {
            let top_configs = record.top_configurations(config.max_configs_per_dataset);
            total_configs += top_configs.len();

            for config_record in top_configs {
                // Compute priority weight combining similarity and rank
                let rank_weight = 1.0 / (config_record.rank as f64);
                let priority_weight = if config.weight_by_similarity {
                    *similarity * rank_weight
                } else {
                    rank_weight
                };

                weighted_configs.push(WeightedConfiguration {
                    config: config_record.clone(),
                    source_similarity: *similarity,
                    source_dataset: record.name.clone(),
                    priority_weight,
                });
            }
        }

        // Sort by priority weight
        weighted_configs.sort_by(|a, b| b.priority_weight.partial_cmp(&a.priority_weight).unwrap());

        // Truncate to max total
        weighted_configs.truncate(config.max_total_configs);

        // Compute mean similarity
        let mean_similarity = similar_datasets.iter().map(|s| s.similarity).sum::<f64>()
            / similar_datasets.len() as f64;

        Ok(WarmStartResult {
            similar_datasets,
            configurations: weighted_configs,
            total_configs_found: total_configs,
            mean_similarity,
        })
    }

    /// Compute normalization parameters from stored datasets
    fn compute_normalization_params(&self) -> NormalizationParams {
        if self.is_empty() {
            return NormalizationParams {
                means: Vec::new(),
                stds: Vec::new(),
            };
        }

        // Collect all vectors
        let vectors: Vec<Vec<f64>> = self
            .datasets
            .values()
            .map(|r| r.metafeatures.to_vector())
            .collect();

        // Find max length
        let max_len = vectors.iter().map(|v| v.len()).max().unwrap_or(0);

        if max_len == 0 {
            return NormalizationParams {
                means: Vec::new(),
                stds: Vec::new(),
            };
        }

        // Compute means and stds for each feature
        let mut means = vec![0.0; max_len];
        let mut stds = vec![1.0; max_len];

        for i in 0..max_len {
            let values: Vec<f64> = vectors
                .iter()
                .filter_map(|v| v.get(i).copied())
                .filter(|v| v.is_finite())
                .collect();

            if !values.is_empty() {
                let n = values.len() as f64;
                let mean = values.iter().sum::<f64>() / n;
                means[i] = mean;

                if values.len() > 1 {
                    let variance =
                        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
                    stds[i] = variance.sqrt().max(1e-10);
                }
            }
        }

        NormalizationParams { means, stds }
    }

    /// Normalize a metafeature vector
    fn normalize_vector(&self, vec: &[f64]) -> Vec<f64> {
        // Use cached params or compute them
        let params = self
            .normalization_params
            .clone()
            .unwrap_or_else(|| self.compute_normalization_params());

        vec.iter()
            .enumerate()
            .map(|(i, &v)| {
                if i < params.means.len() && v.is_finite() {
                    (v - params.means[i]) / params.stds[i]
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Serialize the store to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            FerroError::SerializationError(format!("Failed to serialize store: {}", e))
        })
    }

    /// Deserialize a store from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            FerroError::SerializationError(format!("Failed to deserialize store: {}", e))
        })
    }

    /// Merge another store into this one
    pub fn merge(&mut self, other: MetaLearningStore) {
        for (name, record) in other.datasets {
            // Keep the record with more configurations, or the newer one
            if let Some(existing) = self.datasets.get(&name) {
                if record.configurations.len() > existing.configurations.len()
                    || record.updated_at > existing.updated_at
                {
                    self.datasets.insert(name, record);
                }
            } else {
                self.datasets.insert(name, record);
            }
        }
        self.normalization_params = None;
    }

    /// Get statistics about the store
    pub fn stats(&self) -> StoreStatistics {
        let total_configs: usize = self.datasets.values().map(|r| r.configurations.len()).sum();
        let classification_count = self
            .datasets
            .values()
            .filter(|r| r.task == Task::Classification)
            .count();
        let regression_count = self
            .datasets
            .values()
            .filter(|r| r.task == Task::Regression)
            .count();

        StoreStatistics {
            total_datasets: self.len(),
            total_configurations: total_configs,
            classification_datasets: classification_count,
            regression_datasets: regression_count,
            avg_configs_per_dataset: if self.is_empty() {
                0.0
            } else {
                total_configs as f64 / self.len() as f64
            },
        }
    }
}

/// Statistics about a meta-learning store
#[derive(Debug, Clone)]
pub struct StoreStatistics {
    /// Total number of datasets
    pub total_datasets: usize,
    /// Total number of configurations
    pub total_configurations: usize,
    /// Number of classification datasets
    pub classification_datasets: usize,
    /// Number of regression datasets
    pub regression_datasets: usize,
    /// Average configurations per dataset
    pub avg_configs_per_dataset: f64,
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).take(len).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().take(len).map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().take(len).map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).max(-1.0).min(1.0)
    }
}

/// Get current timestamp in seconds since Unix epoch
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automl::metafeatures::{
        InformationMetafeatures, MetafeatureConfig, SimpleMetafeatures, StatisticalMetafeatures,
    };
    use ndarray::{Array1, Array2};

    fn create_test_metafeatures(n_samples: usize, n_features: usize) -> DatasetMetafeatures {
        DatasetMetafeatures {
            simple: SimpleMetafeatures {
                n_samples,
                n_features,
                n_numeric_features: n_features,
                n_classes: Some(2),
                dimensionality: n_features as f64 / n_samples as f64,
                imbalance_ratio: Some(1.0),
                missing_ratio: 0.0,
                is_high_dimensional: n_features > n_samples,
                is_large: n_samples * n_features > 1_000_000,
                log_size: ((n_samples * n_features) as f64).ln(),
            },
            statistical: StatisticalMetafeatures {
                mean_mean: 0.5,
                mean_std: 0.1,
                std_mean: 0.3,
                std_std: 0.05,
                skewness_mean: 0.0,
                skewness_std: 0.1,
                kurtosis_mean: 0.0,
                kurtosis_std: 0.2,
                min_min: 0.0,
                max_max: 1.0,
                outlier_ratio: 0.01,
                correlation_mean: 0.2,
                correlation_std: 0.1,
                per_feature_stats: Vec::new(),
            },
            information: InformationMetafeatures {
                target_entropy: 1.0,
                max_target_entropy: 1.0,
                normalized_entropy: 1.0,
                feature_entropy_mean: 2.0,
                feature_entropy_std: 0.5,
                mutual_info_mean: 0.3,
                mutual_info_std: 0.1,
                mutual_info_max: 0.5,
                mutual_info_sum: 1.5,
                equivalent_n_features: 3.0,
                noise_signal_ratio: 0.5,
            },
            landmarking: None,
        }
    }

    fn create_test_trials() -> Vec<TrialResult> {
        vec![
            TrialResult::new(
                0,
                AlgorithmType::LogisticRegression,
                0.85,
                0.02,
                vec![0.83, 0.87],
            ),
            TrialResult::new(
                1,
                AlgorithmType::RandomForestClassifier,
                0.90,
                0.03,
                vec![0.88, 0.92],
            ),
            TrialResult::new(2, AlgorithmType::GaussianNB, 0.75, 0.05, vec![0.72, 0.78]),
        ]
    }

    #[test]
    fn test_warm_start_config_builder() {
        let config = WarmStartConfig::new()
            .with_k_nearest(10)
            .with_min_similarity(0.5)
            .with_max_configs_per_dataset(3)
            .with_max_total_configs(15);

        assert_eq!(config.k_nearest, 10);
        assert_eq!(config.min_similarity, 0.5);
        assert_eq!(config.max_configs_per_dataset, 3);
        assert_eq!(config.max_total_configs, 15);
    }

    #[test]
    fn test_configuration_record() {
        let trials = create_test_trials();
        let record = ConfigurationRecord::from_trial(&trials[0], true, 1);

        assert_eq!(record.algorithm, AlgorithmType::LogisticRegression);
        assert_eq!(record.cv_score, 0.85);
        assert!(record.maximize);
        assert_eq!(record.rank, 1);
    }

    #[test]
    fn test_dataset_record() {
        let mf = create_test_metafeatures(100, 10);
        let trials = create_test_trials();

        let mut record = DatasetRecord::new("test_dataset", mf, Task::Classification);
        record.add_configurations(&trials, true);

        assert_eq!(record.name, "test_dataset");
        assert_eq!(record.task, Task::Classification);
        assert_eq!(record.configurations.len(), 3);

        // Best config should be first (RandomForest with 0.90)
        let top = record.top_configurations(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].algorithm, AlgorithmType::RandomForestClassifier);
    }

    #[test]
    fn test_meta_learning_store_basic() {
        let mut store = MetaLearningStore::new();
        assert!(store.is_empty());

        let mf = create_test_metafeatures(100, 10);
        let trials = create_test_trials();

        store.add_dataset("dataset1", mf.clone(), Task::Classification, &trials, true);

        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
        assert!(store.get("dataset1").is_some());
        assert!(store.get("nonexistent").is_none());
    }

    #[test]
    fn test_find_similar() {
        let mut store = MetaLearningStore::new();

        // Add a few datasets with slightly different metafeatures
        for i in 0..5 {
            let mf = create_test_metafeatures(100 + i * 10, 10);
            let trials = create_test_trials();
            store.add_dataset(
                format!("dataset_{}", i),
                mf,
                Task::Classification,
                &trials,
                true,
            );
        }

        // Query with similar metafeatures
        let query_mf = create_test_metafeatures(105, 10);
        let config = WarmStartConfig::new()
            .with_k_nearest(3)
            .with_min_similarity(0.0);

        let similar = store.find_similar(&query_mf, &config, Some(Task::Classification));

        assert!(!similar.is_empty());
        assert!(similar.len() <= 3);
        // First result should be most similar
        for window in similar.windows(2) {
            assert!(window[0].0 >= window[1].0);
        }
    }

    #[test]
    fn test_get_warm_start_configs() {
        let mut store = MetaLearningStore::new();

        // Add datasets
        for i in 0..3 {
            let mf = create_test_metafeatures(100 + i * 10, 10);
            let trials = create_test_trials();
            store.add_dataset(
                format!("dataset_{}", i),
                mf,
                Task::Classification,
                &trials,
                true,
            );
        }

        let query_mf = create_test_metafeatures(105, 10);
        let config = WarmStartConfig::new()
            .with_k_nearest(3)
            .with_min_similarity(0.0)
            .with_max_configs_per_dataset(2);

        let result = store
            .get_warm_start_configs(&query_mf, &config, Task::Classification)
            .unwrap();

        assert!(result.has_configurations());
        assert!(!result.similar_datasets.is_empty());
        assert!(result.mean_similarity > 0.0);
    }

    #[test]
    fn test_prioritized_algorithms() {
        let mut store = MetaLearningStore::new();

        let mf = create_test_metafeatures(100, 10);
        let trials = create_test_trials();
        store.add_dataset("dataset1", mf.clone(), Task::Classification, &trials, true);

        let query_mf = create_test_metafeatures(100, 10);
        // Disable normalization since with a single dataset, normalized vectors
        // become all zeros, making similarity 0.0 and all priority weights equal
        let config = WarmStartConfig::new()
            .with_min_similarity(0.0)
            .with_normalize_metafeatures(false);

        let result = store
            .get_warm_start_configs(&query_mf, &config, Task::Classification)
            .unwrap();

        let prioritized = result.prioritized_algorithms();
        assert!(!prioritized.is_empty());
        // RandomForest should be first (best score in our test data)
        assert_eq!(prioritized[0], AlgorithmType::RandomForestClassifier);
    }

    #[test]
    fn test_empty_store() {
        let store = MetaLearningStore::new();
        let query_mf = create_test_metafeatures(100, 10);
        let config = WarmStartConfig::new();

        let result = store
            .get_warm_start_configs(&query_mf, &config, Task::Classification)
            .unwrap();

        assert!(!result.has_configurations());
        assert!(result.similar_datasets.is_empty());
    }

    #[test]
    fn test_task_filtering() {
        let mut store = MetaLearningStore::new();

        // Add classification dataset
        let mf1 = create_test_metafeatures(100, 10);
        let trials1 = create_test_trials();
        store.add_dataset("clf_dataset", mf1, Task::Classification, &trials1, true);

        // Add regression dataset
        let mf2 = create_test_metafeatures(100, 10);
        let trials2 = vec![TrialResult::new(
            0,
            AlgorithmType::LinearRegression,
            0.85,
            0.02,
            vec![0.83, 0.87],
        )];
        store.add_dataset("reg_dataset", mf2, Task::Regression, &trials2, true);

        let query_mf = create_test_metafeatures(100, 10);
        let config = WarmStartConfig::new().with_min_similarity(0.0);

        // Should only find classification datasets
        let clf_result = store
            .get_warm_start_configs(&query_mf, &config, Task::Classification)
            .unwrap();
        assert_eq!(clf_result.similar_datasets.len(), 1);
        assert_eq!(clf_result.similar_datasets[0].task, Task::Classification);

        // Should only find regression datasets
        let reg_result = store
            .get_warm_start_configs(&query_mf, &config, Task::Regression)
            .unwrap();
        assert_eq!(reg_result.similar_datasets.len(), 1);
        assert_eq!(reg_result.similar_datasets[0].task, Task::Regression);
    }

    #[test]
    fn test_serialization() {
        let mut store = MetaLearningStore::new();
        let mf = create_test_metafeatures(100, 10);
        let trials = create_test_trials();
        store.add_dataset("test", mf, Task::Classification, &trials, true);

        let json = store.to_json().unwrap();
        let restored = MetaLearningStore::from_json(&json).unwrap();

        assert_eq!(restored.len(), 1);
        assert!(restored.get("test").is_some());
    }

    #[test]
    fn test_store_merge() {
        let mut store1 = MetaLearningStore::new();
        let mut store2 = MetaLearningStore::new();

        let mf1 = create_test_metafeatures(100, 10);
        let mf2 = create_test_metafeatures(200, 20);
        let trials = create_test_trials();

        store1.add_dataset("dataset1", mf1.clone(), Task::Classification, &trials, true);
        store2.add_dataset("dataset2", mf2.clone(), Task::Classification, &trials, true);

        store1.merge(store2);

        assert_eq!(store1.len(), 2);
        assert!(store1.get("dataset1").is_some());
        assert!(store1.get("dataset2").is_some());
    }

    #[test]
    fn test_store_stats() {
        let mut store = MetaLearningStore::new();

        let mf = create_test_metafeatures(100, 10);
        let trials = create_test_trials();
        store.add_dataset("clf", mf.clone(), Task::Classification, &trials, true);

        let reg_trials = vec![TrialResult::new(
            0,
            AlgorithmType::LinearRegression,
            0.9,
            0.01,
            vec![0.89, 0.91],
        )];
        store.add_dataset("reg", mf, Task::Regression, &reg_trials, true);

        let stats = store.stats();
        assert_eq!(stats.total_datasets, 2);
        assert_eq!(stats.classification_datasets, 1);
        assert_eq!(stats.regression_datasets, 1);
        assert_eq!(stats.total_configurations, 4); // 3 from clf + 1 from reg
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-10);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_min_similarity_threshold() {
        let mut store = MetaLearningStore::new();

        // Add dataset with specific metafeatures
        let mf = create_test_metafeatures(100, 10);
        let trials = create_test_trials();
        store.add_dataset("dataset1", mf, Task::Classification, &trials, true);

        // Query with very different metafeatures
        let query_mf = create_test_metafeatures(10000, 1000);
        let config = WarmStartConfig::new().with_min_similarity(0.99); // Very high threshold

        let similar = store.find_similar(&query_mf, &config, Some(Task::Classification));

        // May be empty if similarity is below threshold
        for (sim, _) in &similar {
            assert!(*sim >= 0.99);
        }
    }

    #[test]
    fn test_configurations_by_algorithm() {
        let mut store = MetaLearningStore::new();

        let mf = create_test_metafeatures(100, 10);
        let trials = create_test_trials();
        store.add_dataset("test", mf.clone(), Task::Classification, &trials, true);

        let query_mf = create_test_metafeatures(100, 10);
        let config = WarmStartConfig::new().with_min_similarity(0.0);

        let result = store
            .get_warm_start_configs(&query_mf, &config, Task::Classification)
            .unwrap();

        let by_algo = result.configurations_by_algorithm();
        assert!(by_algo.contains_key(&AlgorithmType::RandomForestClassifier));
        assert!(by_algo.contains_key(&AlgorithmType::LogisticRegression));
    }

    #[test]
    fn test_real_metafeature_extraction() {
        // Create actual test data
        let x = Array2::from_shape_vec((50, 3), (0..150).map(|i| (i as f64) / 150.0).collect())
            .unwrap();
        let y = Array1::from_vec((0..50).map(|i| (i % 2) as f64).collect());

        let config = MetafeatureConfig::new().without_landmarking();
        let mf = DatasetMetafeatures::extract(&x, &y, true, Some(config)).unwrap();

        assert_eq!(mf.simple.n_samples, 50);
        assert_eq!(mf.simple.n_features, 3);

        // Use for warm-starting
        let mut store = MetaLearningStore::new();
        let trials = create_test_trials();
        store.add_dataset("real_data", mf.clone(), Task::Classification, &trials, true);

        let ws_config = WarmStartConfig::new().with_min_similarity(0.0);
        let result = store
            .get_warm_start_configs(&mf, &ws_config, Task::Classification)
            .unwrap();

        assert!(result.has_configurations());
    }
}

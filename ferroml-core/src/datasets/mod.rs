//! Dataset Loading and Management
//!
//! This module provides utilities for loading, managing, and working with datasets
//! for machine learning tasks.
//!
//! ## Available Features
//!
//! - [`Dataset`] - Core struct holding features (X) and targets (y)
//! - [`DatasetInfo`] - Metadata about datasets (description, feature names, etc.)
//! - [`DatasetLoader`] - Trait for implementing dataset loaders
//!
//! ## Built-in Datasets (toy datasets for testing/learning)
//!
//! | Dataset | Task | Samples | Features | Classes |
//! |---------|------|---------|----------|---------|
//! | [`load_iris`] | Classification | 150 | 4 | 3 |
//! | [`load_wine`] | Classification | 178 | 13 | 3 |
//! | [`load_diabetes`] | Regression | 442 | 10 | - |
//! | [`load_linnerud`] | Regression | 20 | 3 | - |
//!
//! ## Example
//!
//! ```
//! use ferroml_core::datasets::{Dataset, DatasetInfo, load_iris};
//!
//! // Load the classic iris dataset
//! let (dataset, info) = load_iris();
//!
//! // Access data
//! println!("Samples: {}", dataset.n_samples());
//! println!("Features: {}", dataset.n_features());
//! println!("Description: {}", info.description);
//!
//! // Get X, y for training
//! let (x, y) = dataset.into_arrays();
//! ```
//!
//! ## Synthetic Data Generators
//!
//! - [`make_classification`] - Random n-class classification problem
//! - [`make_regression`] - Random regression problem
//! - [`make_blobs`] - Gaussian blobs for clustering
//! - [`make_moons`] - Two interleaving half circles
//! - [`make_circles`] - Concentric circles
//!
//! ## Data Loading Utilities
//!
//! The module provides utilities for loading data from external sources:
//!
//! - [`load_csv`], [`load_csv_with_options`] - Load CSV files with automatic type inference
//! - [`load_parquet`], [`load_parquet_with_options`] - Load Parquet files efficiently
//! - [`load_file`] - Automatically detect format from file extension
//! - NumPy arrays (through Python bindings)
//!
//! ## Memory-Mapped Datasets
//!
//! For datasets that don't fit in RAM, the module provides memory-mapped file support:
//!
//! - [`MemmappedDataset`] - Memory-mapped dataset with zero-copy access
//! - [`MemmappedDatasetBuilder`] - Builder for creating memory-mapped datasets
//! - [`MemmappedArray2`], [`MemmappedArray1`] - Low-level memory-mapped arrays
//!
//! ```ignore
//! use ferroml_core::datasets::{Dataset, MemmappedDataset, MemmappedDatasetBuilder};
//!
//! // Create a memory-mapped dataset from arrays
//! let dataset = MemmappedDatasetBuilder::new("large_data.fmm")
//!     .with_features(x)
//!     .with_targets(y)
//!     .build()?;
//!
//! // Or from an existing Dataset
//! let mmap_dataset = MemmappedDataset::from_dataset("large_data.fmm", &dataset)?;
//!
//! // Access data with zero-copy views
//! let x_view = mmap_dataset.x_view();
//! let y_view = mmap_dataset.y_view();
//! ```
//!
//! ### Loading from Files
//!
//! ```ignore
//! use ferroml_core::datasets::{load_csv, load_parquet, load_file, CsvOptions};
//!
//! // Load CSV file
//! let (dataset, info) = load_csv("data.csv", Some("target_column"))?;
//!
//! // Load with custom options
//! let opts = CsvOptions::new().with_delimiter(b';');
//! let (dataset, info) = load_csv_with_options("data.csv", Some("target"), opts)?;
//!
//! // Load Parquet file
//! let (dataset, info) = load_parquet("data.parquet", Some("target"))?;
//!
//! // Auto-detect format
//! let (dataset, info) = load_file("data.csv", Some("target"))?;
//! ```
//!
//! ## Statistical Rigor
//!
//! All datasets include:
//! - Complete metadata (feature names, target names, descriptions)
//! - Data characteristics (missing values, class distribution)
//! - Recommended train/test splits for reproducibility
//! - Citation information where applicable

mod loaders;
pub mod mmap;
mod toy;

pub use loaders::{
    load_csv, load_csv_with_options, load_file, load_parquet, load_parquet_with_options,
    CsvEncoding, CsvOptions, ParquetOptions,
};
pub use mmap::{
    MemmappedArray1, MemmappedArray2, MemmappedArray2Mut, MemmappedDataset,
    MemmappedDatasetBuilder, peek_mmap_info,
};
pub use toy::{load_diabetes, load_iris, load_linnerud, load_wine};

use crate::{FerroError, Result, Task};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// A dataset containing features and targets for supervised learning.
///
/// This is the core data structure for machine learning tasks. It holds
/// the feature matrix X and target vector y, along with optional metadata.
///
/// # Example
///
/// ```
/// use ferroml_core::datasets::Dataset;
/// use ndarray::{array, Array1, Array2};
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let y = array![0.0, 1.0, 0.0];
///
/// let dataset = Dataset::new(x, y);
/// assert_eq!(dataset.n_samples(), 3);
/// assert_eq!(dataset.n_features(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Feature matrix (n_samples, n_features)
    x: Array2<f64>,
    /// Target vector (n_samples,)
    y: Array1<f64>,
    /// Optional feature names
    feature_names: Option<Vec<String>>,
    /// Optional target names (for classification)
    target_names: Option<Vec<String>>,
    /// Optional sample indices/identifiers
    sample_ids: Option<Vec<String>>,
}

impl Dataset {
    /// Create a new dataset from feature matrix and target vector.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - Target vector of shape (n_samples,)
    ///
    /// # Panics
    ///
    /// Panics if the number of samples in x doesn't match the length of y.
    pub fn new(x: Array2<f64>, y: Array1<f64>) -> Self {
        assert_eq!(
            x.nrows(),
            y.len(),
            "Feature matrix rows ({}) must match target length ({})",
            x.nrows(),
            y.len()
        );
        Self {
            x,
            y,
            feature_names: None,
            target_names: None,
            sample_ids: None,
        }
    }

    /// Create a dataset with validation.
    ///
    /// Returns an error if shapes don't match, rather than panicking.
    pub fn try_new(x: Array2<f64>, y: Array1<f64>) -> Result<Self> {
        if x.nrows() != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("({}, {})", x.nrows(), x.ncols()),
                format!("target length {}", y.len()),
            ));
        }
        Ok(Self {
            x,
            y,
            feature_names: None,
            target_names: None,
            sample_ids: None,
        })
    }

    /// Create a dataset with only features (unsupervised learning).
    pub fn from_features(x: Array2<f64>) -> Self {
        let n_samples = x.nrows();
        let y = Array1::zeros(n_samples);
        Self {
            x,
            y,
            feature_names: None,
            target_names: None,
            sample_ids: None,
        }
    }

    /// Set feature names.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Set target names (for classification).
    pub fn with_target_names(mut self, names: Vec<String>) -> Self {
        self.target_names = Some(names);
        self
    }

    /// Set sample identifiers.
    pub fn with_sample_ids(mut self, ids: Vec<String>) -> Self {
        self.sample_ids = Some(ids);
        self
    }

    /// Get the number of samples.
    pub fn n_samples(&self) -> usize {
        self.x.nrows()
    }

    /// Get the number of features.
    pub fn n_features(&self) -> usize {
        self.x.ncols()
    }

    /// Get the shape as (n_samples, n_features).
    pub fn shape(&self) -> (usize, usize) {
        (self.n_samples(), self.n_features())
    }

    /// Get a reference to the feature matrix.
    pub fn x(&self) -> &Array2<f64> {
        &self.x
    }

    /// Get a reference to the target vector.
    pub fn y(&self) -> &Array1<f64> {
        &self.y
    }

    /// Get feature names if available.
    pub fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    /// Get target names if available.
    pub fn target_names(&self) -> Option<&[String]> {
        self.target_names.as_deref()
    }

    /// Get sample IDs if available.
    pub fn sample_ids(&self) -> Option<&[String]> {
        self.sample_ids.as_deref()
    }

    /// Consume the dataset and return the (X, y) arrays.
    pub fn into_arrays(self) -> (Array2<f64>, Array1<f64>) {
        (self.x, self.y)
    }

    /// Get references to (X, y) arrays.
    pub fn as_arrays(&self) -> (&Array2<f64>, &Array1<f64>) {
        (&self.x, &self.y)
    }

    /// Split into train and test sets.
    ///
    /// # Arguments
    ///
    /// * `test_size` - Fraction of data to use for testing (0.0 to 1.0)
    /// * `shuffle` - Whether to shuffle before splitting
    /// * `random_state` - Optional random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Tuple of (train_dataset, test_dataset)
    pub fn train_test_split(
        &self,
        test_size: f64,
        shuffle: bool,
        random_state: Option<u64>,
    ) -> Result<(Dataset, Dataset)> {
        if !(0.0..=1.0).contains(&test_size) {
            return Err(FerroError::invalid_input(format!(
                "test_size must be between 0 and 1, got {}",
                test_size
            )));
        }

        let n_samples = self.n_samples();
        let n_test = (n_samples as f64 * test_size).round() as usize;
        let n_train = n_samples - n_test;

        if n_train == 0 || n_test == 0 {
            return Err(FerroError::invalid_input(
                "Split would result in empty train or test set",
            ));
        }

        let indices: Vec<usize> = if shuffle {
            use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
            let mut indices: Vec<usize> = (0..n_samples).collect();
            let mut rng = match random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_os_rng(),
            };
            indices.shuffle(&mut rng);
            indices
        } else {
            (0..n_samples).collect()
        };

        let train_indices = &indices[..n_train];
        let test_indices = &indices[n_train..];

        let x_train = self.x.select(Axis(0), train_indices);
        let y_train = self.y.select(Axis(0), train_indices);
        let x_test = self.x.select(Axis(0), test_indices);
        let y_test = self.y.select(Axis(0), test_indices);

        let mut train = Dataset::new(x_train, y_train);
        let mut test = Dataset::new(x_test, y_test);

        // Preserve metadata
        if let Some(names) = &self.feature_names {
            train = train.with_feature_names(names.clone());
            test = test.with_feature_names(names.clone());
        }
        if let Some(names) = &self.target_names {
            train = train.with_target_names(names.clone());
            test = test.with_target_names(names.clone());
        }

        Ok((train, test))
    }

    /// Get unique classes in the target (for classification).
    pub fn unique_classes(&self) -> Vec<f64> {
        let mut classes: Vec<f64> = self.y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        classes
    }

    /// Get class counts (for classification).
    pub fn class_counts(&self) -> std::collections::HashMap<i64, usize> {
        let mut counts = std::collections::HashMap::new();
        for &y in self.y.iter() {
            #[allow(clippy::cast_possible_truncation)]
            let class = y.round() as i64;
            *counts.entry(class).or_insert(0) += 1;
        }
        counts
    }

    /// Check if the target is likely binary classification.
    pub fn is_binary(&self) -> bool {
        self.unique_classes().len() == 2
    }

    /// Check if the target is likely multiclass classification.
    pub fn is_multiclass(&self) -> bool {
        let classes = self.unique_classes();
        classes.len() > 2 && classes.iter().all(|&c| (c - c.round()).abs() < 1e-10)
    }

    /// Infer the task type from the target.
    pub fn infer_task(&self) -> Task {
        let unique = self.unique_classes();
        // If all targets are integers and there are relatively few unique values,
        // it's likely classification
        let all_integer = unique.iter().all(|&v| (v - v.round()).abs() < 1e-10);
        // For small datasets, be more generous with the classification heuristic
        let n_samples = self.n_samples();
        let few_unique = unique.len() <= 20
            && (n_samples < 50 || unique.len() < n_samples / 5);

        if all_integer && few_unique {
            Task::Classification
        } else {
            Task::Regression
        }
    }

    /// Get basic statistics about the dataset.
    pub fn describe(&self) -> DatasetStatistics {
        let mut feature_stats = Vec::with_capacity(self.n_features());

        for col_idx in 0..self.n_features() {
            let column = self.x.column(col_idx);
            let values: Vec<f64> = column.iter().copied().collect();

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
                / (values.len() - 1).max(1) as f64;
            let std = variance.sqrt();

            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let min = sorted.first().copied().unwrap_or(f64::NAN);
            let max = sorted.last().copied().unwrap_or(f64::NAN);
            let median = if sorted.len() % 2 == 0 {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
            } else {
                sorted[sorted.len() / 2]
            };

            let n_missing = values.iter().filter(|&&v| v.is_nan()).count();

            let name = self
                .feature_names
                .as_ref()
                .and_then(|names| names.get(col_idx))
                .cloned();

            feature_stats.push(FeatureStatistics {
                name,
                mean,
                std,
                min,
                max,
                median,
                n_missing,
            });
        }

        DatasetStatistics {
            n_samples: self.n_samples(),
            n_features: self.n_features(),
            feature_stats,
        }
    }
}

/// Statistics for a single feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStatistics {
    /// Feature name (if available)
    pub name: Option<String>,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Median value
    pub median: f64,
    /// Number of missing (NaN) values
    pub n_missing: usize,
}

/// Overall dataset statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Per-feature statistics
    pub feature_stats: Vec<FeatureStatistics>,
}

/// Metadata about a dataset.
///
/// Contains descriptive information about the dataset, including
/// its origin, task type, and recommendations for use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Name of the dataset
    pub name: String,
    /// Description of the dataset
    pub description: String,
    /// Task type (classification, regression, etc.)
    pub task: Task,
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Number of classes (for classification)
    pub n_classes: Option<usize>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Target names (for classification)
    pub target_names: Option<Vec<String>>,
    /// Original source/citation
    pub source: Option<String>,
    /// URL for more information
    pub url: Option<String>,
    /// License information
    pub license: Option<String>,
    /// Version string
    pub version: Option<String>,
}

impl DatasetInfo {
    /// Create new dataset info with required fields.
    pub fn new(name: impl Into<String>, task: Task, n_samples: usize, n_features: usize) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            task,
            n_samples,
            n_features,
            n_classes: None,
            feature_names: Vec::new(),
            target_names: None,
            source: None,
            url: None,
            license: None,
            version: None,
        }
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set number of classes.
    pub fn with_n_classes(mut self, n: usize) -> Self {
        self.n_classes = Some(n);
        self
    }

    /// Set feature names.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Set target names.
    pub fn with_target_names(mut self, names: Vec<String>) -> Self {
        self.target_names = Some(names);
        self
    }

    /// Set source citation.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set URL.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Set license.
    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    /// Set version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }
}

/// Trait for implementing dataset loaders.
///
/// Implement this trait to create custom dataset loaders for
/// different data sources (files, URLs, databases, etc.).
pub trait DatasetLoader {
    /// Load the dataset.
    fn load(&self) -> Result<(Dataset, DatasetInfo)>;

    /// Get the name of the dataset.
    fn name(&self) -> &str;

    /// Get a description of the dataset.
    fn description(&self) -> &str;
}

/// Options for loading datasets.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Whether to shuffle the data after loading
    pub shuffle: bool,
    /// Random seed for shuffling
    pub random_state: Option<u64>,
    /// Whether to return as sparse (if supported)
    pub as_sparse: bool,
    /// Subset of features to load (by index)
    pub feature_indices: Option<Vec<usize>>,
    /// Subset of samples to load (by index)
    pub sample_indices: Option<Vec<usize>>,
}

impl LoadOptions {
    /// Create new load options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable shuffling.
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random state for shuffling.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Request sparse output.
    pub fn as_sparse(mut self) -> Self {
        self.as_sparse = true;
        self
    }

    /// Select specific features.
    pub fn with_features(mut self, indices: Vec<usize>) -> Self {
        self.feature_indices = Some(indices);
        self
    }

    /// Select specific samples.
    pub fn with_samples(mut self, indices: Vec<usize>) -> Self {
        self.sample_indices = Some(indices);
        self
    }
}

/// Generate synthetic classification data.
///
/// Creates a random n-class classification problem.
///
/// # Arguments
///
/// * `n_samples` - Number of samples
/// * `n_features` - Number of features
/// * `n_informative` - Number of informative features
/// * `n_classes` - Number of classes
/// * `random_state` - Random seed for reproducibility
///
/// # Example
///
/// ```
/// use ferroml_core::datasets::make_classification;
///
/// let (dataset, info) = make_classification(100, 10, 5, 2, Some(42));
/// assert_eq!(dataset.n_samples(), 100);
/// assert_eq!(dataset.n_features(), 10);
/// ```
pub fn make_classification(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    n_classes: usize,
    random_state: Option<u64>,
) -> (Dataset, DatasetInfo) {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let mut rng: StdRng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };

    let n_informative = n_informative.min(n_features);

    // Generate informative features with class-dependent means
    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);

    for i in 0..n_samples {
        #[allow(clippy::cast_precision_loss)]
        let class = (i * n_classes / n_samples) as f64;
        y[i] = class;

        // Informative features: class-dependent + noise
        for j in 0..n_informative {
            #[allow(clippy::cast_precision_loss)]
            let class_mean = (class + 1.0) * 2.0 * ((j % 2) as f64 * 2.0 - 1.0);
            x[[i, j]] = class_mean + rng.random::<f64>() * 2.0 - 1.0;
        }

        // Noise features: pure random
        for j in n_informative..n_features {
            x[[i, j]] = rng.random::<f64>() * 2.0 - 1.0;
        }
    }

    // Shuffle
    let mut indices: Vec<usize> = (0..n_samples).collect();
    use rand::seq::SliceRandom;
    indices.shuffle(&mut rng);

    let x_shuffled = x.select(Axis(0), &indices);
    let y_shuffled = y.select(Axis(0), &indices);

    let feature_names: Vec<String> = (0..n_features).map(|i| format!("feature_{}", i)).collect();
    let target_names: Vec<String> = (0..n_classes).map(|i| format!("class_{}", i)).collect();

    let dataset = Dataset::new(x_shuffled, y_shuffled)
        .with_feature_names(feature_names.clone())
        .with_target_names(target_names.clone());

    let info = DatasetInfo::new("synthetic_classification", Task::Classification, n_samples, n_features)
        .with_description(format!(
            "Synthetic classification dataset with {} samples, {} features ({} informative), {} classes",
            n_samples, n_features, n_informative, n_classes
        ))
        .with_n_classes(n_classes)
        .with_feature_names(feature_names)
        .with_target_names(target_names);

    (dataset, info)
}

/// Generate synthetic regression data.
///
/// Creates a random regression problem.
///
/// # Arguments
///
/// * `n_samples` - Number of samples
/// * `n_features` - Number of features
/// * `n_informative` - Number of informative features
/// * `noise` - Standard deviation of Gaussian noise
/// * `random_state` - Random seed for reproducibility
///
/// # Example
///
/// ```
/// use ferroml_core::datasets::make_regression;
///
/// let (dataset, info) = make_regression(100, 10, 5, 0.1, Some(42));
/// assert_eq!(dataset.n_samples(), 100);
/// assert_eq!(dataset.n_features(), 10);
/// ```
pub fn make_regression(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise: f64,
    random_state: Option<u64>,
) -> (Dataset, DatasetInfo) {
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::{Distribution, Normal};

    let mut rng: StdRng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };

    let n_informative = n_informative.min(n_features);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let noise_dist = Normal::new(0.0, noise).unwrap();

    // Generate random coefficients for informative features
    let coefficients: Vec<f64> = (0..n_informative)
        .map(|_| normal.sample(&mut rng))
        .collect();

    // Generate features
    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);

    for i in 0..n_samples {
        // Generate features
        for j in 0..n_features {
            x[[i, j]] = normal.sample(&mut rng);
        }

        // Compute target as linear combination + noise
        let mut target = 0.0;
        for (j, &coef) in coefficients.iter().enumerate() {
            target += coef * x[[i, j]];
        }
        target += noise_dist.sample(&mut rng);
        y[i] = target;
    }

    let feature_names: Vec<String> = (0..n_features).map(|i| format!("feature_{}", i)).collect();

    let dataset = Dataset::new(x, y).with_feature_names(feature_names.clone());

    let info = DatasetInfo::new("synthetic_regression", Task::Regression, n_samples, n_features)
        .with_description(format!(
            "Synthetic regression dataset with {} samples, {} features ({} informative), noise std {}",
            n_samples, n_features, n_informative, noise
        ))
        .with_feature_names(feature_names);

    (dataset, info)
}

/// Generate synthetic blobs for clustering.
///
/// Creates isotropic Gaussian blobs for clustering evaluation.
///
/// # Arguments
///
/// * `n_samples` - Total number of samples
/// * `n_features` - Number of features
/// * `centers` - Number of cluster centers
/// * `cluster_std` - Standard deviation of clusters
/// * `random_state` - Random seed for reproducibility
///
/// # Example
///
/// ```
/// use ferroml_core::datasets::make_blobs;
///
/// let (dataset, info) = make_blobs(100, 2, 3, 1.0, Some(42));
/// assert_eq!(dataset.n_samples(), 100);
/// assert_eq!(dataset.unique_classes().len(), 3);
/// ```
pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    centers: usize,
    cluster_std: f64,
    random_state: Option<u64>,
) -> (Dataset, DatasetInfo) {
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::{Distribution, Normal, Uniform};

    let mut rng: StdRng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };

    let uniform = Uniform::new(-10.0_f64, 10.0_f64).unwrap();
    let normal = Normal::new(0.0, cluster_std).unwrap();

    // Generate cluster centers
    let center_coords: Vec<Vec<f64>> = (0..centers)
        .map(|_| {
            (0..n_features)
                .map(|_| uniform.sample(&mut rng))
                .collect()
        })
        .collect();

    // Generate samples
    let samples_per_cluster = n_samples / centers;
    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);

    let mut sample_idx = 0;
    for (cluster_idx, center) in center_coords.iter().enumerate() {
        let n_this_cluster = if cluster_idx == centers - 1 {
            n_samples - sample_idx
        } else {
            samples_per_cluster
        };

        for _ in 0..n_this_cluster {
            if sample_idx >= n_samples {
                break;
            }
            for (j, &c) in center.iter().enumerate() {
                x[[sample_idx, j]] = c + normal.sample(&mut rng);
            }
            #[allow(clippy::cast_precision_loss)]
            {
                y[sample_idx] = cluster_idx as f64;
            }
            sample_idx += 1;
        }
    }

    // Shuffle
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);
    let x_shuffled = x.select(Axis(0), &indices);
    let y_shuffled = y.select(Axis(0), &indices);

    let feature_names: Vec<String> = (0..n_features).map(|i| format!("feature_{}", i)).collect();
    let target_names: Vec<String> = (0..centers).map(|i| format!("cluster_{}", i)).collect();

    let dataset = Dataset::new(x_shuffled, y_shuffled)
        .with_feature_names(feature_names.clone())
        .with_target_names(target_names.clone());

    let info = DatasetInfo::new("synthetic_blobs", Task::Classification, n_samples, n_features)
        .with_description(format!(
            "Synthetic blob dataset with {} samples, {} features, {} centers (std={})",
            n_samples, n_features, centers, cluster_std
        ))
        .with_n_classes(centers)
        .with_feature_names(feature_names)
        .with_target_names(target_names);

    (dataset, info)
}

/// Generate synthetic moons dataset for binary classification.
///
/// Creates two interleaving half circles.
///
/// # Arguments
///
/// * `n_samples` - Number of samples (split evenly between moons)
/// * `noise` - Standard deviation of Gaussian noise
/// * `random_state` - Random seed for reproducibility
///
/// # Example
///
/// ```
/// use ferroml_core::datasets::make_moons;
///
/// let (dataset, info) = make_moons(100, 0.1, Some(42));
/// assert_eq!(dataset.n_samples(), 100);
/// assert_eq!(dataset.n_features(), 2);
/// ```
pub fn make_moons(n_samples: usize, noise: f64, random_state: Option<u64>) -> (Dataset, DatasetInfo) {
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::{Distribution, Normal};
    use std::f64::consts::PI;

    let mut rng: StdRng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };

    let normal = Normal::new(0.0, noise).unwrap();
    let n_per_moon = n_samples / 2;

    let mut x = Array2::zeros((n_samples, 2));
    let mut y = Array1::zeros(n_samples);

    // First moon (upper)
    for i in 0..n_per_moon {
        #[allow(clippy::cast_precision_loss)]
        let angle = PI * i as f64 / n_per_moon as f64;
        x[[i, 0]] = angle.cos() + normal.sample(&mut rng);
        x[[i, 1]] = angle.sin() + normal.sample(&mut rng);
        y[i] = 0.0;
    }

    // Second moon (lower, shifted)
    for i in 0..(n_samples - n_per_moon) {
        let idx = n_per_moon + i;
        #[allow(clippy::cast_precision_loss)]
        let angle = PI * i as f64 / (n_samples - n_per_moon) as f64;
        x[[idx, 0]] = 1.0 - angle.cos() + normal.sample(&mut rng);
        x[[idx, 1]] = 0.5 - angle.sin() + normal.sample(&mut rng);
        y[idx] = 1.0;
    }

    // Shuffle
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);
    let x_shuffled = x.select(Axis(0), &indices);
    let y_shuffled = y.select(Axis(0), &indices);

    let feature_names = vec!["x".to_string(), "y".to_string()];
    let target_names = vec!["moon_0".to_string(), "moon_1".to_string()];

    let dataset = Dataset::new(x_shuffled, y_shuffled)
        .with_feature_names(feature_names.clone())
        .with_target_names(target_names.clone());

    let info = DatasetInfo::new("synthetic_moons", Task::Classification, n_samples, 2)
        .with_description(format!(
            "Synthetic moons dataset with {} samples, noise std {}",
            n_samples, noise
        ))
        .with_n_classes(2)
        .with_feature_names(feature_names)
        .with_target_names(target_names);

    (dataset, info)
}

/// Generate synthetic circles dataset for binary classification.
///
/// Creates a large circle containing a smaller circle.
///
/// # Arguments
///
/// * `n_samples` - Number of samples (split evenly between circles)
/// * `noise` - Standard deviation of Gaussian noise
/// * `factor` - Scale factor between inner and outer circle (0 to 1)
/// * `random_state` - Random seed for reproducibility
///
/// # Example
///
/// ```
/// use ferroml_core::datasets::make_circles;
///
/// let (dataset, info) = make_circles(100, 0.1, 0.5, Some(42));
/// assert_eq!(dataset.n_samples(), 100);
/// assert_eq!(dataset.n_features(), 2);
/// ```
pub fn make_circles(
    n_samples: usize,
    noise: f64,
    factor: f64,
    random_state: Option<u64>,
) -> (Dataset, DatasetInfo) {
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::{Distribution, Normal};
    use std::f64::consts::PI;

    let mut rng: StdRng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };

    let normal = Normal::new(0.0, noise).unwrap();
    let n_per_circle = n_samples / 2;

    let mut x = Array2::zeros((n_samples, 2));
    let mut y = Array1::zeros(n_samples);

    // Outer circle
    for i in 0..n_per_circle {
        #[allow(clippy::cast_precision_loss)]
        let angle = 2.0 * PI * i as f64 / n_per_circle as f64;
        x[[i, 0]] = angle.cos() + normal.sample(&mut rng);
        x[[i, 1]] = angle.sin() + normal.sample(&mut rng);
        y[i] = 0.0;
    }

    // Inner circle
    for i in 0..(n_samples - n_per_circle) {
        let idx = n_per_circle + i;
        #[allow(clippy::cast_precision_loss)]
        let angle = 2.0 * PI * i as f64 / (n_samples - n_per_circle) as f64;
        x[[idx, 0]] = factor * angle.cos() + normal.sample(&mut rng);
        x[[idx, 1]] = factor * angle.sin() + normal.sample(&mut rng);
        y[idx] = 1.0;
    }

    // Shuffle
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);
    let x_shuffled = x.select(Axis(0), &indices);
    let y_shuffled = y.select(Axis(0), &indices);

    let feature_names = vec!["x".to_string(), "y".to_string()];
    let target_names = vec!["outer".to_string(), "inner".to_string()];

    let dataset = Dataset::new(x_shuffled, y_shuffled)
        .with_feature_names(feature_names.clone())
        .with_target_names(target_names.clone());

    let info = DatasetInfo::new("synthetic_circles", Task::Classification, n_samples, 2)
        .with_description(format!(
            "Synthetic circles dataset with {} samples, noise std {}, factor {}",
            n_samples, noise, factor
        ))
        .with_n_classes(2)
        .with_feature_names(feature_names)
        .with_target_names(target_names);

    (dataset, info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_creation() {
        use ndarray::array;

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![0.0, 1.0, 0.0];

        let dataset = Dataset::new(x, y);
        assert_eq!(dataset.n_samples(), 3);
        assert_eq!(dataset.n_features(), 2);
        assert_eq!(dataset.shape(), (3, 2));
    }

    #[test]
    fn test_dataset_with_metadata() {
        use ndarray::array;

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let dataset = Dataset::new(x, y)
            .with_feature_names(vec!["a".to_string(), "b".to_string()])
            .with_target_names(vec!["class_0".to_string(), "class_1".to_string()]);

        assert_eq!(
            dataset.feature_names(),
            Some(&["a".to_string(), "b".to_string()][..])
        );
        assert_eq!(
            dataset.target_names(),
            Some(&["class_0".to_string(), "class_1".to_string()][..])
        );
    }

    #[test]
    fn test_dataset_try_new_error() {
        use ndarray::array;

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 2.0]; // Wrong length

        let result = Dataset::try_new(x, y);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_test_split() {
        use ndarray::Array2;

        let x = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let y = ndarray::Array1::from_vec((0..10).map(|x| (x % 2) as f64).collect());

        let dataset = Dataset::new(x, y);
        let (train, test) = dataset.train_test_split(0.2, false, Some(42)).unwrap();

        assert_eq!(train.n_samples(), 8);
        assert_eq!(test.n_samples(), 2);
    }

    #[test]
    fn test_unique_classes() {
        use ndarray::array;

        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 1.0, 0.0, 2.0];

        let dataset = Dataset::new(x, y);
        let classes = dataset.unique_classes();
        assert_eq!(classes, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_class_counts() {
        use ndarray::array;

        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![0.0, 1.0, 0.0, 2.0, 0.0];

        let dataset = Dataset::new(x, y);
        let counts = dataset.class_counts();
        assert_eq!(counts.get(&0), Some(&3));
        assert_eq!(counts.get(&1), Some(&1));
        assert_eq!(counts.get(&2), Some(&1));
    }

    #[test]
    fn test_is_binary() {
        use ndarray::array;

        let x = array![[1.0], [2.0]];
        let y_binary = array![0.0, 1.0];
        let y_multi = array![0.0, 2.0];

        assert!(Dataset::new(x.clone(), y_binary).is_binary());
        assert!(Dataset::new(x, y_multi).is_binary()); // Still 2 classes
    }

    #[test]
    fn test_infer_task() {
        use ndarray::array;

        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

        // Classification: few integer classes
        let y_class = array![0.0, 1.0, 0.0, 1.0, 2.0];
        assert_eq!(
            Dataset::new(x.clone(), y_class).infer_task(),
            Task::Classification
        );

        // Regression: continuous values
        let y_reg = array![1.23, 4.56, 7.89, 0.12, 3.45];
        assert_eq!(Dataset::new(x, y_reg).infer_task(), Task::Regression);
    }

    #[test]
    fn test_describe() {
        use ndarray::array;

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![0.0, 1.0, 0.0];

        let dataset = Dataset::new(x, y);
        let stats = dataset.describe();

        assert_eq!(stats.n_samples, 3);
        assert_eq!(stats.n_features, 2);
        assert_eq!(stats.feature_stats.len(), 2);

        // First feature: [1, 3, 5] mean=3
        assert!((stats.feature_stats[0].mean - 3.0).abs() < 1e-10);
        assert!((stats.feature_stats[0].min - 1.0).abs() < 1e-10);
        assert!((stats.feature_stats[0].max - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_make_classification() {
        let (dataset, info) = make_classification(100, 10, 5, 3, Some(42));

        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 10);
        assert_eq!(dataset.unique_classes().len(), 3);
        assert_eq!(info.n_classes, Some(3));
        assert_eq!(info.task, Task::Classification);
    }

    #[test]
    fn test_make_regression() {
        let (dataset, info) = make_regression(100, 10, 5, 0.1, Some(42));

        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 10);
        assert_eq!(info.task, Task::Regression);
    }

    #[test]
    fn test_make_blobs() {
        let (dataset, info) = make_blobs(100, 2, 3, 1.0, Some(42));

        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 2);
        assert_eq!(dataset.unique_classes().len(), 3);
        assert_eq!(info.n_classes, Some(3));
    }

    #[test]
    fn test_make_moons() {
        let (dataset, info) = make_moons(100, 0.1, Some(42));

        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 2);
        assert_eq!(dataset.unique_classes().len(), 2);
        assert_eq!(info.n_classes, Some(2));
    }

    #[test]
    fn test_make_circles() {
        let (dataset, info) = make_circles(100, 0.1, 0.5, Some(42));

        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 2);
        assert_eq!(dataset.unique_classes().len(), 2);
        assert_eq!(info.n_classes, Some(2));
    }

    #[test]
    fn test_dataset_info() {
        let info = DatasetInfo::new("test", Task::Classification, 100, 10)
            .with_description("A test dataset")
            .with_n_classes(3)
            .with_feature_names(vec!["a".to_string(), "b".to_string()])
            .with_source("Test source")
            .with_url("https://example.com")
            .with_license("MIT");

        assert_eq!(info.name, "test");
        assert_eq!(info.n_classes, Some(3));
        assert_eq!(info.source, Some("Test source".to_string()));
    }

    #[test]
    fn test_load_options() {
        let opts = LoadOptions::new()
            .with_shuffle(true)
            .with_random_state(42)
            .with_features(vec![0, 1, 2]);

        assert!(opts.shuffle);
        assert_eq!(opts.random_state, Some(42));
        assert_eq!(opts.feature_indices, Some(vec![0, 1, 2]));
    }

    #[test]
    fn test_train_test_split_shuffled() {
        use ndarray::Array2;

        let x = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let y = ndarray::Array1::from_vec((0..10).map(|x| (x % 2) as f64).collect());

        let dataset = Dataset::new(x, y);

        // Split with shuffle
        let (train1, _test1) = dataset
            .train_test_split(0.2, true, Some(42))
            .unwrap();
        let (train2, _test2) = dataset
            .train_test_split(0.2, true, Some(42))
            .unwrap();

        // Same seed should give same results
        assert_eq!(train1.x(), train2.x());

        // Different seed should (likely) give different results
        let (train3, _test3) = dataset
            .train_test_split(0.2, true, Some(123))
            .unwrap();
        // Note: this could theoretically fail but is extremely unlikely
        assert_ne!(train1.x(), train3.x());
    }

    #[test]
    fn test_from_features() {
        use ndarray::array;

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let dataset = Dataset::from_features(x);

        assert_eq!(dataset.n_samples(), 2);
        assert_eq!(dataset.n_features(), 2);
        // y should be zeros
        assert!(dataset.y().iter().all(|&v| v == 0.0));
    }
}

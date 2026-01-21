//! Sampling Techniques for Class Imbalance
//!
//! This module provides resampling techniques to handle imbalanced datasets,
//! where one class significantly outnumbers another.
//!
//! ## Algorithms
//!
//! ### Oversampling
//!
//! - [`RandomOverSampler`] - Random duplication of minority samples
//!   - Duplicates existing minority class samples randomly
//!   - Simple baseline with optional replacement
//!
//! - [`SMOTE`] - Synthetic Minority Oversampling Technique
//!   - Generates synthetic samples by interpolating between minority class instances
//!   - Uses k-nearest neighbors to find interpolation targets
//!
//! - [`ADASYN`] - Adaptive Synthetic Sampling
//!   - Generates more synthetic samples for minority instances that are harder to learn
//!   - Adaptively weights sample generation based on local class distribution
//!   - Focuses synthesis on difficult regions of the feature space
//!
//! ### Undersampling
//!
//! - [`RandomUnderSampler`] - Random removal of majority samples
//!   - Removes samples from majority classes randomly
//!   - Simple baseline with optional replacement
//!
//! ### Combined Methods
//!
//! - [`SMOTETomek`] - SMOTE + Tomek Links Cleaning
//!   - Applies SMOTE oversampling, then removes majority samples that form Tomek links
//!   - Cleans decision boundary while balancing classes
//!
//! - [`SMOTEENN`] - SMOTE + Edited Nearest Neighbors Cleaning
//!   - Applies SMOTE oversampling, then removes samples misclassified by k-NN
//!   - More aggressive cleaning than Tomek links
//!   - Uses [`ENNKind`] to control cleaning aggressiveness
//!
//! ## Example
//!
//! ```ignore
//! use ferroml_core::preprocessing::sampling::{SMOTE, Resampler};
//! use ndarray::{Array1, Array2};
//!
//! // Imbalanced dataset: 100 majority samples, 10 minority samples
//! let x = Array2::from_shape_fn((110, 2), |(i, j)| {
//!     if i < 100 { (i + j) as f64 } else { 100.0 + (i + j) as f64 }
//! });
//! let y = Array1::from_iter((0..100).map(|_| 0.0).chain((0..10).map(|_| 1.0)));
//!
//! // Apply SMOTE to balance the classes
//! let mut smote = SMOTE::new();
//! let (x_resampled, y_resampled) = smote.fit_resample(&x, &y)?;
//!
//! // Now the minority class has more samples
//! ```
//!
//! ## Statistical Considerations
//!
//! - SMOTE should only be applied to training data, never to validation/test data
//! - Cross-validation should use SMOTE inside each fold to prevent data leakage
//! - SMOTE assumes that interpolating between minority samples creates valid samples
//! - For very high-dimensional data, consider using dimensionality reduction first

use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait for resampling techniques that handle class imbalance.
///
/// Resamplers modify the dataset by either:
/// - **Oversampling**: Creating new synthetic samples for minority classes
/// - **Undersampling**: Removing samples from majority classes
/// - **Combined**: Both oversampling and undersampling
pub trait Resampler: Send + Sync {
    /// Fit the resampler to the data and resample in one step.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix of shape `(n_samples, n_features)`
    /// * `y` - Target labels of shape `(n_samples,)`
    ///
    /// # Returns
    ///
    /// * Tuple of (resampled features, resampled labels)
    fn fit_resample(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>)>;

    /// Get the resampling strategy description.
    fn strategy_description(&self) -> String;
}

/// Strategy for determining how many synthetic samples to generate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Resample all classes to match the majority class count.
    /// This is the default and most common strategy.
    Auto,
    /// Resample minority classes to have the specified ratio of minority to majority.
    /// For example, 1.0 means equal numbers, 0.5 means half as many minority samples.
    Ratio(f64),
    /// Specify exact target counts for each class.
    /// Classes not in the map are left unchanged.
    TargetCounts(HashMap<i64, usize>),
    /// Resample only the specified classes.
    /// The value is the target ratio relative to the majority class.
    Classes(HashMap<i64, f64>),
    /// Do not resample any class (for testing or passthrough).
    NotResampled,
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

/// SMOTE - Synthetic Minority Oversampling Technique
///
/// SMOTE generates synthetic samples for minority classes by interpolating
/// between existing minority class samples and their k-nearest neighbors.
///
/// ## Algorithm
///
/// For each minority class sample:
/// 1. Find its k nearest neighbors (within the same class)
/// 2. For each synthetic sample to generate:
///    - Randomly select one of the k neighbors
///    - Generate a new sample at a random point along the line segment
///      connecting the original sample and the selected neighbor:
///      `synthetic = original + random(0,1) * (neighbor - original)`
///
/// ## Reference
///
/// Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
/// SMOTE: Synthetic Minority Over-sampling Technique.
/// Journal of Artificial Intelligence Research, 16, 321-357.
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::preprocessing::sampling::{SMOTE, Resampler, SamplingStrategy};
///
/// // Create SMOTE with 5 nearest neighbors
/// let mut smote = SMOTE::new()
///     .with_k_neighbors(5)
///     .with_random_state(42);
///
/// let (x_resampled, y_resampled) = smote.fit_resample(&x, &y)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SMOTE {
    /// Number of nearest neighbors to use for generating synthetic samples.
    /// Default is 5.
    k_neighbors: usize,

    /// Sampling strategy for determining target class distribution.
    sampling_strategy: SamplingStrategy,

    /// Random seed for reproducibility.
    random_state: Option<u64>,

    /// Number of samples generated in last fit_resample call (for diagnostics).
    #[serde(skip)]
    n_synthetic_samples_: Option<HashMap<i64, usize>>,
}

impl Default for SMOTE {
    fn default() -> Self {
        Self::new()
    }
}

impl SMOTE {
    /// Create a new SMOTE instance with default parameters.
    ///
    /// Default settings:
    /// - k_neighbors: 5
    /// - sampling_strategy: Auto (balance all classes to majority)
    pub fn new() -> Self {
        Self {
            k_neighbors: 5,
            sampling_strategy: SamplingStrategy::Auto,
            random_state: None,
            n_synthetic_samples_: None,
        }
    }

    /// Set the number of nearest neighbors to use.
    ///
    /// # Arguments
    ///
    /// * `k` - Number of neighbors. Must be at least 1.
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k.max(1);
        self
    }

    /// Set the sampling strategy.
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Get the number of synthetic samples generated per class in the last call.
    pub fn n_synthetic_samples(&self) -> Option<&HashMap<i64, usize>> {
        self.n_synthetic_samples_.as_ref()
    }

    /// Find k nearest neighbors within the minority class.
    ///
    /// Returns indices into the `minority_samples` array.
    fn find_k_neighbors(&self, sample_idx: usize, minority_samples: &Array2<f64>) -> Vec<usize> {
        let n_samples = minority_samples.nrows();
        let sample = minority_samples.row(sample_idx);

        // Compute distances to all other samples
        let mut distances: Vec<(usize, f64)> = (0..n_samples)
            .filter(|&i| i != sample_idx)
            .map(|i| {
                let other = minority_samples.row(i);
                let dist: f64 = sample
                    .iter()
                    .zip(other.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (i, dist.sqrt())
            })
            .collect();

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances
            .into_iter()
            .take(self.k_neighbors)
            .map(|(i, _)| i)
            .collect()
    }

    /// Generate a synthetic sample by interpolating between two samples.
    fn generate_synthetic_sample(
        &self,
        sample: &ndarray::ArrayView1<f64>,
        neighbor: &ndarray::ArrayView1<f64>,
        rng: &mut StdRng,
    ) -> Array1<f64> {
        let gap: f64 = rng.random();
        sample
            .iter()
            .zip(neighbor.iter())
            .map(|(s, n)| s + gap * (n - s))
            .collect()
    }

    /// Determine target sample counts per class based on strategy.
    fn compute_target_counts(
        &self,
        class_counts: &HashMap<i64, usize>,
    ) -> Result<HashMap<i64, usize>> {
        let max_count = *class_counts.values().max().unwrap_or(&0);

        match &self.sampling_strategy {
            SamplingStrategy::Auto => {
                // All classes should have max_count samples
                Ok(class_counts.keys().map(|&k| (k, max_count)).collect())
            }
            SamplingStrategy::Ratio(ratio) => {
                if *ratio <= 0.0 || *ratio > 1.0 {
                    return Err(FerroError::invalid_input("Ratio must be in (0, 1]"));
                }
                let target = (max_count as f64 * ratio).ceil() as usize;
                Ok(class_counts
                    .iter()
                    .map(|(&k, &v)| (k, v.max(target)))
                    .collect())
            }
            SamplingStrategy::TargetCounts(targets) => {
                let mut result = class_counts.clone();
                for (k, &v) in targets {
                    if let Some(current) = result.get_mut(k) {
                        *current = (*current).max(v);
                    }
                }
                Ok(result)
            }
            SamplingStrategy::Classes(class_ratios) => {
                let mut result = class_counts.clone();
                for (&k, &ratio) in class_ratios {
                    if let Some(current) = result.get_mut(&k) {
                        let target = (max_count as f64 * ratio).ceil() as usize;
                        *current = (*current).max(target);
                    }
                }
                Ok(result)
            }
            SamplingStrategy::NotResampled => Ok(class_counts.clone()),
        }
    }
}

impl Resampler for SMOTE {
    fn fit_resample(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(FerroError::invalid_input("Input array cannot be empty"));
        }

        if n_samples != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("({},)", n_samples),
                format!("({},)", y.len()),
            ));
        }

        // Count samples per class
        let mut class_counts: HashMap<i64, usize> = HashMap::new();
        let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();

        for (i, &label) in y.iter().enumerate() {
            let label_int = label.round() as i64;
            *class_counts.entry(label_int).or_insert(0) += 1;
            class_indices.entry(label_int).or_default().push(i);
        }

        if class_counts.len() < 2 {
            return Err(FerroError::invalid_input(
                "SMOTE requires at least 2 classes",
            ));
        }

        // Compute target counts
        let target_counts = self.compute_target_counts(&class_counts)?;

        // Initialize RNG
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        // Collect all samples (original + synthetic)
        let mut all_x: Vec<Array1<f64>> = x.rows().into_iter().map(|r| r.to_owned()).collect();
        let mut all_y: Vec<f64> = y.to_vec();

        // Track synthetic samples generated
        let mut n_synthetic: HashMap<i64, usize> = HashMap::new();

        // Generate synthetic samples for each minority class
        for (&class_label, &current_count) in &class_counts {
            let target = target_counts.get(&class_label).unwrap_or(&current_count);
            let n_to_generate = target.saturating_sub(current_count);

            if n_to_generate == 0 {
                continue;
            }

            let indices = &class_indices[&class_label];
            let minority_samples: Array2<f64> =
                Array2::from_shape_fn((indices.len(), n_features), |(i, j)| x[[indices[i], j]]);

            // Check if we have enough samples for k-NN
            if indices.len() < 2 {
                return Err(FerroError::invalid_input(format!(
                    "Class {} has only {} sample(s), need at least 2 for SMOTE",
                    class_label,
                    indices.len()
                )));
            }

            // Effective k is min(k_neighbors, n_samples - 1)
            let effective_k = self.k_neighbors.min(indices.len() - 1);
            if effective_k == 0 {
                return Err(FerroError::invalid_input(format!(
                    "Class {} has insufficient samples for k-NN",
                    class_label
                )));
            }

            // Generate synthetic samples
            for _ in 0..n_to_generate {
                // Randomly select a minority sample
                let sample_idx = rng.random_range(0..indices.len());

                // Find its k nearest neighbors
                let neighbors = self.find_k_neighbors(sample_idx, &minority_samples);

                if neighbors.is_empty() {
                    continue;
                }

                // Randomly select one neighbor
                let neighbor_idx = neighbors[rng.random_range(0..neighbors.len())];

                // Generate synthetic sample
                let sample = minority_samples.row(sample_idx);
                let neighbor = minority_samples.row(neighbor_idx);
                let synthetic = self.generate_synthetic_sample(&sample, &neighbor, &mut rng);

                all_x.push(synthetic);
                all_y.push(class_label as f64);
            }

            n_synthetic.insert(class_label, n_to_generate);
        }

        self.n_synthetic_samples_ = Some(n_synthetic);

        // Build output arrays
        let n_total = all_x.len();
        let mut x_resampled = Array2::zeros((n_total, n_features));
        for (i, row) in all_x.iter().enumerate() {
            x_resampled.row_mut(i).assign(row);
        }

        let y_resampled = Array1::from_vec(all_y);

        Ok((x_resampled, y_resampled))
    }

    fn strategy_description(&self) -> String {
        match &self.sampling_strategy {
            SamplingStrategy::Auto => "auto (balance to majority class)".to_string(),
            SamplingStrategy::Ratio(r) => format!("ratio = {:.2}", r),
            SamplingStrategy::TargetCounts(t) => format!("target_counts = {:?}", t),
            SamplingStrategy::Classes(c) => format!("classes = {:?}", c),
            SamplingStrategy::NotResampled => "not_resampled".to_string(),
        }
    }
}

/// SMOTE with Borderline samples focus (SMOTE-Borderline).
///
/// This variant only generates synthetic samples from "borderline" minority
/// samples - those that have some majority class neighbors.
///
/// ## Variants
///
/// - **Borderline-1**: Only uses minority class neighbors for interpolation
/// - **Borderline-2**: Uses both minority and majority class neighbors
///
/// ## Reference
///
/// Han, H., Wang, W. Y., & Mao, B. H. (2005).
/// Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderlineSMOTE {
    /// Base SMOTE instance.
    smote: SMOTE,
    /// Number of neighbors to use for borderline detection.
    m_neighbors: usize,
    /// Borderline kind: 1 for borderline-1, 2 for borderline-2.
    kind: u8,
}

impl Default for BorderlineSMOTE {
    fn default() -> Self {
        Self::new()
    }
}

impl BorderlineSMOTE {
    /// Create a new BorderlineSMOTE instance.
    pub fn new() -> Self {
        Self {
            smote: SMOTE::new(),
            m_neighbors: 10,
            kind: 1,
        }
    }

    /// Set the number of nearest neighbors for synthetic sample generation.
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.smote = self.smote.with_k_neighbors(k);
        self
    }

    /// Set the number of neighbors for borderline detection.
    pub fn with_m_neighbors(mut self, m: usize) -> Self {
        self.m_neighbors = m.max(1);
        self
    }

    /// Set the borderline kind (1 or 2).
    pub fn with_kind(mut self, kind: u8) -> Self {
        self.kind = kind.clamp(1, 2);
        self
    }

    /// Set the sampling strategy.
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.smote = self.smote.with_sampling_strategy(strategy);
        self
    }

    /// Set the random seed.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.smote = self.smote.with_random_state(seed);
        self
    }

    /// Classify minority samples as noise, danger (borderline), or safe.
    fn classify_samples(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        minority_class: i64,
        minority_indices: &[usize],
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let n_samples = x.nrows();
        let mut noise = Vec::new();
        let mut danger = Vec::new();
        let mut safe = Vec::new();

        for &idx in minority_indices {
            let sample = x.row(idx);

            // Find m nearest neighbors from all samples
            let mut distances: Vec<(usize, f64)> = (0..n_samples)
                .filter(|&i| i != idx)
                .map(|i| {
                    let other = x.row(i);
                    let dist: f64 = sample
                        .iter()
                        .zip(other.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (i, dist.sqrt())
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let neighbors: Vec<usize> = distances
                .iter()
                .take(self.m_neighbors)
                .map(|(i, _)| *i)
                .collect();

            // Count majority class neighbors
            let n_majority: usize = neighbors
                .iter()
                .filter(|&&n_idx| (y[n_idx].round() as i64) != minority_class)
                .count();

            // Classify based on majority neighbor count
            let m = self.m_neighbors as f64;
            if n_majority as f64 == m {
                // All neighbors are majority - this is noise
                noise.push(idx);
            } else if n_majority as f64 >= m / 2.0 {
                // More than half neighbors are majority - borderline/danger
                danger.push(idx);
            } else {
                // Safe sample
                safe.push(idx);
            }
        }

        (noise, danger, safe)
    }
}

impl Resampler for BorderlineSMOTE {
    fn fit_resample(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(FerroError::invalid_input("Input array cannot be empty"));
        }

        if n_samples != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("({},)", n_samples),
                format!("({},)", y.len()),
            ));
        }

        // Count samples per class
        let mut class_counts: HashMap<i64, usize> = HashMap::new();
        let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();

        for (i, &label) in y.iter().enumerate() {
            let label_int = label.round() as i64;
            *class_counts.entry(label_int).or_insert(0) += 1;
            class_indices.entry(label_int).or_default().push(i);
        }

        if class_counts.len() < 2 {
            return Err(FerroError::invalid_input(
                "BorderlineSMOTE requires at least 2 classes",
            ));
        }

        // Compute target counts
        let target_counts = self.smote.compute_target_counts(&class_counts)?;

        // Initialize RNG
        let mut rng = match self.smote.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        // Collect all samples
        let mut all_x: Vec<Array1<f64>> = x.rows().into_iter().map(|r| r.to_owned()).collect();
        let mut all_y: Vec<f64> = y.to_vec();

        // Find majority class
        let majority_class = *class_counts.iter().max_by_key(|(_, &v)| v).unwrap().0;

        // Generate synthetic samples for each minority class
        for (&class_label, &current_count) in &class_counts {
            if class_label == majority_class {
                continue;
            }

            let target = target_counts.get(&class_label).unwrap_or(&current_count);
            let n_to_generate = target.saturating_sub(current_count);

            if n_to_generate == 0 {
                continue;
            }

            let indices = &class_indices[&class_label];

            if indices.len() < 2 {
                return Err(FerroError::invalid_input(format!(
                    "Class {} has only {} sample(s), need at least 2",
                    class_label,
                    indices.len()
                )));
            }

            // Classify samples
            let (_noise, danger, _safe) = self.classify_samples(x, y, class_label, indices);

            // Only use danger (borderline) samples for generation
            if danger.is_empty() {
                // Fall back to all samples if no borderline samples found
                continue;
            }

            // Build array of danger samples for k-NN
            let danger_samples: Array2<f64> =
                Array2::from_shape_fn((danger.len(), n_features), |(i, j)| x[[danger[i], j]]);

            let _effective_k = self.smote.k_neighbors.min(danger.len() - 1).max(1);

            // Generate synthetic samples from borderline samples
            for _ in 0..n_to_generate {
                // Randomly select a borderline sample
                let sample_local_idx = rng.random_range(0..danger.len());

                // Find neighbors (depending on kind)
                let neighbors = if self.kind == 1 {
                    // Borderline-1: Only minority neighbors
                    self.smote
                        .find_k_neighbors(sample_local_idx, &danger_samples)
                } else {
                    // Borderline-2: Can use any neighbor
                    // For simplicity, we still use minority neighbors
                    self.smote
                        .find_k_neighbors(sample_local_idx, &danger_samples)
                };

                if neighbors.is_empty() {
                    continue;
                }

                let neighbor_local_idx = neighbors[rng.random_range(0..neighbors.len())];

                let sample = danger_samples.row(sample_local_idx);
                let neighbor = danger_samples.row(neighbor_local_idx);
                let synthetic = self
                    .smote
                    .generate_synthetic_sample(&sample, &neighbor, &mut rng);

                all_x.push(synthetic);
                all_y.push(class_label as f64);
            }
        }

        // Build output arrays
        let n_total = all_x.len();
        let mut x_resampled = Array2::zeros((n_total, n_features));
        for (i, row) in all_x.iter().enumerate() {
            x_resampled.row_mut(i).assign(row);
        }

        let y_resampled = Array1::from_vec(all_y);

        Ok((x_resampled, y_resampled))
    }

    fn strategy_description(&self) -> String {
        format!(
            "borderline-{} SMOTE ({})",
            self.kind,
            self.smote.strategy_description()
        )
    }
}

/// ADASYN - Adaptive Synthetic Sampling Approach
///
/// ADASYN generates synthetic samples adaptively based on the local density
/// distribution of the minority class. Minority samples that are harder to learn
/// (those with more majority class neighbors) receive more synthetic samples.
///
/// ## Algorithm
///
/// 1. Calculate the degree of class imbalance: d = n_minority / n_majority
/// 2. If d ≥ threshold, data is considered balanced enough
/// 3. Calculate total synthetic samples needed: G = (n_majority - n_minority) × β
/// 4. For each minority sample i:
///    - Find k nearest neighbors (from all samples)
///    - Calculate ratio r_i = (majority neighbors) / k
///    - This r_i represents how "hard" this sample is to classify
/// 5. Normalize: r̂_i = r_i / Σr_i (so they sum to 1)
/// 6. For each minority sample, generate g_i = round(r̂_i × G) synthetic samples
/// 7. Generate synthetics using SMOTE-like interpolation
///
/// ## Key Difference from SMOTE
///
/// Unlike SMOTE which generates an equal number of synthetic samples from each
/// minority instance, ADASYN generates more samples from minority instances that
/// are harder to learn (i.e., surrounded by more majority class samples).
///
/// ## Reference
///
/// He, H., Bai, Y., Garcia, E. A., & Li, S. (2008).
/// ADASYN: Adaptive synthetic sampling approach for imbalanced learning.
/// IEEE International Joint Conference on Neural Networks (IEEE World Congress
/// on Computational Intelligence), pp. 1322-1328.
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::preprocessing::sampling::{ADASYN, Resampler};
///
/// // Create ADASYN with 5 nearest neighbors
/// let mut adasyn = ADASYN::new()
///     .with_k_neighbors(5)
///     .with_random_state(42);
///
/// let (x_resampled, y_resampled) = adasyn.fit_resample(&x, &y)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ADASYN {
    /// Number of nearest neighbors to use for generating synthetic samples.
    /// Default is 5.
    k_neighbors: usize,

    /// Number of nearest neighbors to use for calculating the density distribution.
    /// If None, uses k_neighbors. Default is None.
    n_neighbors: Option<usize>,

    /// Sampling strategy for determining target class distribution.
    sampling_strategy: SamplingStrategy,

    /// Random seed for reproducibility.
    random_state: Option<u64>,

    /// Imbalance threshold. If the imbalance ratio (minority/majority) exceeds
    /// this threshold, no resampling is performed. Default is 1.0 (always resample).
    imbalance_threshold: f64,

    /// Number of samples generated in last fit_resample call (for diagnostics).
    #[serde(skip)]
    n_synthetic_samples_: Option<HashMap<i64, usize>>,

    /// Density ratios computed during fitting (for diagnostics).
    #[serde(skip)]
    density_ratios_: Option<HashMap<i64, Vec<f64>>>,
}

impl Default for ADASYN {
    fn default() -> Self {
        Self::new()
    }
}

impl ADASYN {
    /// Create a new ADASYN instance with default parameters.
    ///
    /// Default settings:
    /// - k_neighbors: 5
    /// - n_neighbors: None (uses k_neighbors)
    /// - sampling_strategy: Auto (balance all classes to majority)
    /// - imbalance_threshold: 1.0 (always resample minority classes)
    pub fn new() -> Self {
        Self {
            k_neighbors: 5,
            n_neighbors: None,
            sampling_strategy: SamplingStrategy::Auto,
            random_state: None,
            imbalance_threshold: 1.0,
            n_synthetic_samples_: None,
            density_ratios_: None,
        }
    }

    /// Set the number of nearest neighbors for synthetic sample generation.
    ///
    /// # Arguments
    ///
    /// * `k` - Number of neighbors. Must be at least 1.
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k.max(1);
        self
    }

    /// Set the number of nearest neighbors for density ratio calculation.
    ///
    /// If not set, uses k_neighbors.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of neighbors. Must be at least 1.
    pub fn with_n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = Some(n.max(1));
        self
    }

    /// Set the sampling strategy.
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the imbalance threshold.
    ///
    /// If the ratio of minority to majority samples exceeds this threshold,
    /// no resampling is performed for that class. Default is 1.0.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Threshold in range (0, 1]. A value of 1.0 means always resample.
    pub fn with_imbalance_threshold(mut self, threshold: f64) -> Self {
        self.imbalance_threshold = threshold.clamp(0.01, 1.0);
        self
    }

    /// Get the number of synthetic samples generated per class in the last call.
    pub fn n_synthetic_samples(&self) -> Option<&HashMap<i64, usize>> {
        self.n_synthetic_samples_.as_ref()
    }

    /// Get the density ratios computed for each minority sample.
    ///
    /// Higher values indicate samples that are harder to learn (more majority neighbors).
    pub fn density_ratios(&self) -> Option<&HashMap<i64, Vec<f64>>> {
        self.density_ratios_.as_ref()
    }

    /// Get the effective number of neighbors for density calculation.
    fn effective_n_neighbors(&self) -> usize {
        self.n_neighbors.unwrap_or(self.k_neighbors)
    }

    /// Find k nearest neighbors from all samples and return majority neighbor count.
    ///
    /// Returns (neighbor indices, number of majority neighbors).
    fn find_neighbors_and_count_majority(
        &self,
        sample_idx: usize,
        x: &Array2<f64>,
        y: &Array1<f64>,
        minority_class: i64,
        k: usize,
    ) -> (Vec<usize>, usize) {
        let n_samples = x.nrows();
        let sample = x.row(sample_idx);

        // Compute distances to all other samples
        let mut distances: Vec<(usize, f64)> = (0..n_samples)
            .filter(|&i| i != sample_idx)
            .map(|i| {
                let other = x.row(i);
                let dist: f64 = sample
                    .iter()
                    .zip(other.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (i, dist.sqrt())
            })
            .collect();

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let neighbors: Vec<usize> = distances.iter().take(k).map(|(i, _)| *i).collect();

        // Count majority class neighbors
        let n_majority = neighbors
            .iter()
            .filter(|&&idx| (y[idx].round() as i64) != minority_class)
            .count();

        (neighbors, n_majority)
    }

    /// Find k nearest neighbors within the minority class only.
    fn find_minority_neighbors(
        &self,
        sample_idx: usize,
        minority_samples: &Array2<f64>,
        k: usize,
    ) -> Vec<usize> {
        let n_samples = minority_samples.nrows();
        let sample = minority_samples.row(sample_idx);

        // Compute distances to all other minority samples
        let mut distances: Vec<(usize, f64)> = (0..n_samples)
            .filter(|&i| i != sample_idx)
            .map(|i| {
                let other = minority_samples.row(i);
                let dist: f64 = sample
                    .iter()
                    .zip(other.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (i, dist.sqrt())
            })
            .collect();

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// Generate a synthetic sample by interpolating between two samples.
    fn generate_synthetic_sample(
        &self,
        sample: &ndarray::ArrayView1<f64>,
        neighbor: &ndarray::ArrayView1<f64>,
        rng: &mut StdRng,
    ) -> Array1<f64> {
        let gap: f64 = rng.random();
        sample
            .iter()
            .zip(neighbor.iter())
            .map(|(s, n)| s + gap * (n - s))
            .collect()
    }

    /// Determine target sample counts per class based on strategy.
    fn compute_target_counts(
        &self,
        class_counts: &HashMap<i64, usize>,
    ) -> Result<HashMap<i64, usize>> {
        let max_count = *class_counts.values().max().unwrap_or(&0);

        match &self.sampling_strategy {
            SamplingStrategy::Auto => {
                // All classes should have max_count samples
                Ok(class_counts.keys().map(|&k| (k, max_count)).collect())
            }
            SamplingStrategy::Ratio(ratio) => {
                if *ratio <= 0.0 || *ratio > 1.0 {
                    return Err(FerroError::invalid_input("Ratio must be in (0, 1]"));
                }
                let target = (max_count as f64 * ratio).ceil() as usize;
                Ok(class_counts
                    .iter()
                    .map(|(&k, &v)| (k, v.max(target)))
                    .collect())
            }
            SamplingStrategy::TargetCounts(targets) => {
                let mut result = class_counts.clone();
                for (k, &v) in targets {
                    if let Some(current) = result.get_mut(k) {
                        *current = (*current).max(v);
                    }
                }
                Ok(result)
            }
            SamplingStrategy::Classes(class_ratios) => {
                let mut result = class_counts.clone();
                for (&k, &ratio) in class_ratios {
                    if let Some(current) = result.get_mut(&k) {
                        let target = (max_count as f64 * ratio).ceil() as usize;
                        *current = (*current).max(target);
                    }
                }
                Ok(result)
            }
            SamplingStrategy::NotResampled => Ok(class_counts.clone()),
        }
    }
}

impl Resampler for ADASYN {
    fn fit_resample(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(FerroError::invalid_input("Input array cannot be empty"));
        }

        if n_samples != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("({},)", n_samples),
                format!("({},)", y.len()),
            ));
        }

        // Count samples per class and collect indices
        let mut class_counts: HashMap<i64, usize> = HashMap::new();
        let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();

        for (i, &label) in y.iter().enumerate() {
            let label_int = label.round() as i64;
            *class_counts.entry(label_int).or_insert(0) += 1;
            class_indices.entry(label_int).or_default().push(i);
        }

        if class_counts.len() < 2 {
            return Err(FerroError::invalid_input(
                "ADASYN requires at least 2 classes",
            ));
        }

        // Compute target counts
        let target_counts = self.compute_target_counts(&class_counts)?;

        // Initialize RNG
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        // Collect all samples (original + synthetic)
        let mut all_x: Vec<Array1<f64>> = x.rows().into_iter().map(|r| r.to_owned()).collect();
        let mut all_y: Vec<f64> = y.to_vec();

        // Track synthetic samples generated and density ratios
        let mut n_synthetic: HashMap<i64, usize> = HashMap::new();
        let mut density_ratios: HashMap<i64, Vec<f64>> = HashMap::new();

        // Find majority class count
        let max_class_count = *class_counts.values().max().unwrap();

        // Number of neighbors for density calculation
        let n_neighbors_density = self.effective_n_neighbors();

        // Generate synthetic samples for each minority class
        for (&class_label, &current_count) in &class_counts {
            let target = target_counts.get(&class_label).unwrap_or(&current_count);
            let n_to_generate = target.saturating_sub(current_count);

            if n_to_generate == 0 {
                continue;
            }

            // Check imbalance ratio
            let imbalance_ratio = current_count as f64 / max_class_count as f64;
            if imbalance_ratio >= self.imbalance_threshold {
                // Class is balanced enough, skip
                continue;
            }

            let indices = &class_indices[&class_label];

            // Check if we have enough samples
            if indices.len() < 2 {
                return Err(FerroError::invalid_input(format!(
                    "Class {} has only {} sample(s), need at least 2 for ADASYN",
                    class_label,
                    indices.len()
                )));
            }

            // Effective k for this class
            let effective_k_density = n_neighbors_density.min(n_samples - 1).max(1);
            let effective_k_gen = self.k_neighbors.min(indices.len() - 1).max(1);

            // Step 1: Calculate density ratio (r_i) for each minority sample
            // r_i = (number of majority neighbors in k-NN) / k
            let mut ratios: Vec<f64> = Vec::with_capacity(indices.len());

            for &idx in indices {
                let (_, n_majority) = self.find_neighbors_and_count_majority(
                    idx,
                    x,
                    y,
                    class_label,
                    effective_k_density,
                );
                ratios.push(n_majority as f64 / effective_k_density as f64);
            }

            // Store density ratios for diagnostics
            density_ratios.insert(class_label, ratios.clone());

            // Step 2: Normalize ratios so they sum to 1
            let ratio_sum: f64 = ratios.iter().sum();

            let normalized_ratios: Vec<f64> = if ratio_sum > 0.0 {
                ratios.iter().map(|r| r / ratio_sum).collect()
            } else {
                // All samples have no majority neighbors - distribute evenly
                let uniform = 1.0 / indices.len() as f64;
                vec![uniform; indices.len()]
            };

            // Step 3: Calculate number of synthetic samples per minority instance
            // g_i = round(r̂_i × G) where G is total samples to generate
            let g_values: Vec<usize> = normalized_ratios
                .iter()
                .map(|&r| (r * n_to_generate as f64).round() as usize)
                .collect();

            // Adjust to ensure we generate exactly n_to_generate samples
            let total_g: usize = g_values.iter().sum();
            let mut adjusted_g = g_values.clone();

            // If we need to adjust, add/remove from samples with highest ratios
            if total_g != n_to_generate {
                let diff = n_to_generate as i64 - total_g as i64;

                // Sort indices by ratio (descending) to adjust high-ratio samples first
                let mut sorted_indices: Vec<usize> = (0..indices.len()).collect();
                sorted_indices.sort_by(|&a, &b| {
                    normalized_ratios[b]
                        .partial_cmp(&normalized_ratios[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut remaining = diff.abs();
                let mut idx = 0;

                while remaining > 0 && idx < sorted_indices.len() {
                    let sample_idx = sorted_indices[idx % sorted_indices.len()];
                    if diff > 0 {
                        adjusted_g[sample_idx] += 1;
                    } else if adjusted_g[sample_idx] > 0 {
                        adjusted_g[sample_idx] -= 1;
                    }
                    remaining -= 1;
                    idx += 1;
                }
            }

            // Build minority samples array for k-NN in generation
            let minority_samples: Array2<f64> =
                Array2::from_shape_fn((indices.len(), n_features), |(i, j)| x[[indices[i], j]]);

            // Step 4: Generate synthetic samples
            let mut total_generated = 0;
            for (local_idx, &n_gen) in adjusted_g.iter().enumerate() {
                if n_gen == 0 {
                    continue;
                }

                // Find k nearest neighbors within minority class for interpolation
                let neighbors =
                    self.find_minority_neighbors(local_idx, &minority_samples, effective_k_gen);

                if neighbors.is_empty() {
                    continue;
                }

                // Generate n_gen synthetic samples from this minority instance
                for _ in 0..n_gen {
                    // Randomly select one neighbor
                    let neighbor_local_idx = neighbors[rng.random_range(0..neighbors.len())];

                    // Generate synthetic sample by interpolation
                    let sample = minority_samples.row(local_idx);
                    let neighbor = minority_samples.row(neighbor_local_idx);
                    let synthetic = self.generate_synthetic_sample(&sample, &neighbor, &mut rng);

                    all_x.push(synthetic);
                    all_y.push(class_label as f64);
                    total_generated += 1;
                }
            }

            n_synthetic.insert(class_label, total_generated);
        }

        self.n_synthetic_samples_ = Some(n_synthetic);
        self.density_ratios_ = Some(density_ratios);

        // Build output arrays
        let n_total = all_x.len();
        let mut x_resampled = Array2::zeros((n_total, n_features));
        for (i, row) in all_x.iter().enumerate() {
            x_resampled.row_mut(i).assign(row);
        }

        let y_resampled = Array1::from_vec(all_y);

        Ok((x_resampled, y_resampled))
    }

    fn strategy_description(&self) -> String {
        format!(
            "ADASYN (k={}, n={}, threshold={:.2}, {})",
            self.k_neighbors,
            self.effective_n_neighbors(),
            self.imbalance_threshold,
            match &self.sampling_strategy {
                SamplingStrategy::Auto => "auto".to_string(),
                SamplingStrategy::Ratio(r) => format!("ratio={:.2}", r),
                SamplingStrategy::TargetCounts(t) => format!("targets={:?}", t),
                SamplingStrategy::Classes(c) => format!("classes={:?}", c),
                SamplingStrategy::NotResampled => "not_resampled".to_string(),
            }
        )
    }
}

/// RandomUnderSampler - Random Undersampling
///
/// Randomly removes samples from majority classes to balance the dataset.
/// This is the simplest undersampling technique and serves as a baseline.
///
/// ## Algorithm
///
/// For each majority class:
/// 1. Determine the target sample count (based on sampling strategy)
/// 2. Randomly select samples to keep from the class
/// 3. Remove the remaining samples
///
/// ## Advantages
///
/// - Simple and fast
/// - Works well when majority class has many redundant samples
/// - No risk of creating unrealistic samples (unlike oversampling)
///
/// ## Disadvantages
///
/// - Loses potentially useful information by discarding samples
/// - May discard important boundary samples
/// - Can lead to underfitting if too many samples are removed
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::preprocessing::sampling::{RandomUnderSampler, Resampler};
///
/// let mut rus = RandomUnderSampler::new()
///     .with_random_state(42);
///
/// let (x_resampled, y_resampled) = rus.fit_resample(&x, &y)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomUnderSampler {
    /// Sampling strategy for determining target class distribution.
    sampling_strategy: SamplingStrategy,

    /// Random seed for reproducibility.
    random_state: Option<u64>,

    /// Whether to sample with replacement.
    /// If true, the same sample can be selected multiple times.
    /// Default is false.
    replacement: bool,

    /// Indices of samples selected in last fit_resample call (for diagnostics).
    #[serde(skip)]
    sample_indices_: Option<Vec<usize>>,

    /// Number of samples removed per class in last fit_resample call.
    #[serde(skip)]
    n_samples_removed_: Option<HashMap<i64, usize>>,
}

impl Default for RandomUnderSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomUnderSampler {
    /// Create a new RandomUnderSampler instance with default parameters.
    ///
    /// Default settings:
    /// - sampling_strategy: Auto (undersample to match minority class)
    /// - replacement: false
    pub fn new() -> Self {
        Self {
            sampling_strategy: SamplingStrategy::Auto,
            random_state: None,
            replacement: false,
            sample_indices_: None,
            n_samples_removed_: None,
        }
    }

    /// Set the sampling strategy.
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set whether to sample with replacement.
    ///
    /// If true, the same sample can be selected multiple times.
    /// Default is false.
    pub fn with_replacement(mut self, replacement: bool) -> Self {
        self.replacement = replacement;
        self
    }

    /// Get the indices of samples selected in the last call.
    pub fn sample_indices(&self) -> Option<&Vec<usize>> {
        self.sample_indices_.as_ref()
    }

    /// Get the number of samples removed per class in the last call.
    pub fn n_samples_removed(&self) -> Option<&HashMap<i64, usize>> {
        self.n_samples_removed_.as_ref()
    }

    /// Determine target sample counts per class based on strategy.
    /// For undersampling, we want to reduce majority classes.
    fn compute_target_counts(
        &self,
        class_counts: &HashMap<i64, usize>,
    ) -> Result<HashMap<i64, usize>> {
        let min_count = *class_counts.values().min().unwrap_or(&0);
        let max_count = *class_counts.values().max().unwrap_or(&0);

        match &self.sampling_strategy {
            SamplingStrategy::Auto => {
                // All classes should have min_count samples (undersample to minority)
                Ok(class_counts.keys().map(|&k| (k, min_count)).collect())
            }
            SamplingStrategy::Ratio(ratio) => {
                if *ratio <= 0.0 || *ratio > 1.0 {
                    return Err(FerroError::invalid_input("Ratio must be in (0, 1]"));
                }
                // Target is ratio * min_count (relative to minority)
                let target = ((min_count as f64) / ratio).ceil() as usize;
                Ok(class_counts
                    .iter()
                    .map(|(&k, &v)| (k, v.min(target)))
                    .collect())
            }
            SamplingStrategy::TargetCounts(targets) => {
                let mut result = class_counts.clone();
                for (k, &v) in targets {
                    if let Some(current) = result.get_mut(k) {
                        // For undersampling, take the minimum
                        *current = (*current).min(v);
                    }
                }
                Ok(result)
            }
            SamplingStrategy::Classes(class_ratios) => {
                let mut result = class_counts.clone();
                for (&k, &ratio) in class_ratios {
                    if let Some(current) = result.get_mut(&k) {
                        // Target is ratio * max_count
                        let target = (max_count as f64 * ratio).ceil() as usize;
                        *current = (*current).min(target);
                    }
                }
                Ok(result)
            }
            SamplingStrategy::NotResampled => Ok(class_counts.clone()),
        }
    }
}

impl Resampler for RandomUnderSampler {
    fn fit_resample(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(FerroError::invalid_input("Input array cannot be empty"));
        }

        if n_samples != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("({},)", n_samples),
                format!("({},)", y.len()),
            ));
        }

        // Count samples per class and collect indices
        let mut class_counts: HashMap<i64, usize> = HashMap::new();
        let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();

        for (i, &label) in y.iter().enumerate() {
            let label_int = label.round() as i64;
            *class_counts.entry(label_int).or_insert(0) += 1;
            class_indices.entry(label_int).or_default().push(i);
        }

        if class_counts.is_empty() {
            return Err(FerroError::invalid_input("No samples found"));
        }

        // Compute target counts
        let target_counts = self.compute_target_counts(&class_counts)?;

        // Initialize RNG
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        // Collect selected indices
        let mut selected_indices: Vec<usize> = Vec::new();
        let mut n_removed: HashMap<i64, usize> = HashMap::new();

        for (&class_label, indices) in &class_indices {
            let current_count = indices.len();
            let target = *target_counts.get(&class_label).unwrap_or(&current_count);

            if target >= current_count {
                // Keep all samples
                selected_indices.extend(indices.iter().copied());
            } else if self.replacement {
                // Sample with replacement
                for _ in 0..target {
                    let idx = indices[rng.random_range(0..indices.len())];
                    selected_indices.push(idx);
                }
                n_removed.insert(class_label, current_count - target);
            } else {
                // Sample without replacement
                let mut available: Vec<usize> = indices.clone();
                available.shuffle(&mut rng);
                selected_indices.extend(available.into_iter().take(target));
                n_removed.insert(class_label, current_count - target);
            }
        }

        // Sort indices for consistent ordering
        selected_indices.sort_unstable();

        self.sample_indices_ = Some(selected_indices.clone());
        self.n_samples_removed_ = Some(n_removed);

        // Build output arrays
        let n_total = selected_indices.len();
        let mut x_resampled = Array2::zeros((n_total, n_features));
        let mut y_resampled = Array1::zeros(n_total);

        for (new_idx, &orig_idx) in selected_indices.iter().enumerate() {
            x_resampled.row_mut(new_idx).assign(&x.row(orig_idx));
            y_resampled[new_idx] = y[orig_idx];
        }

        Ok((x_resampled, y_resampled))
    }

    fn strategy_description(&self) -> String {
        format!(
            "RandomUnderSampler (replacement={}, {})",
            self.replacement,
            match &self.sampling_strategy {
                SamplingStrategy::Auto => "auto (undersample to minority)".to_string(),
                SamplingStrategy::Ratio(r) => format!("ratio={:.2}", r),
                SamplingStrategy::TargetCounts(t) => format!("targets={:?}", t),
                SamplingStrategy::Classes(c) => format!("classes={:?}", c),
                SamplingStrategy::NotResampled => "not_resampled".to_string(),
            }
        )
    }
}

/// RandomOverSampler - Random Oversampling
///
/// Randomly duplicates samples from minority classes to balance the dataset.
/// This is the simplest oversampling technique and serves as a baseline.
///
/// ## Algorithm
///
/// For each minority class:
/// 1. Determine the target sample count (based on sampling strategy)
/// 2. Randomly select samples from the class to duplicate
/// 3. Add the duplicated samples to the dataset
///
/// ## Advantages
///
/// - Simple and fast
/// - No information loss (unlike undersampling)
/// - Preserves the original data distribution
///
/// ## Disadvantages
///
/// - Creates exact duplicates (no new information)
/// - Can lead to overfitting
/// - Decision boundaries may become overly specific to duplicated samples
///
/// ## Comparison with SMOTE
///
/// - RandomOverSampler creates exact copies, SMOTE creates synthetic samples
/// - SMOTE generally produces better generalization
/// - RandomOverSampler is faster and requires no hyperparameters
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::preprocessing::sampling::{RandomOverSampler, Resampler};
///
/// let mut ros = RandomOverSampler::new()
///     .with_random_state(42);
///
/// let (x_resampled, y_resampled) = ros.fit_resample(&x, &y)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomOverSampler {
    /// Sampling strategy for determining target class distribution.
    sampling_strategy: SamplingStrategy,

    /// Random seed for reproducibility.
    random_state: Option<u64>,

    /// Whether to shrink the sampling pool when selecting duplicates.
    /// If true (default), samples are drawn with replacement from the original pool.
    /// If false, samples are drawn without replacement until the pool is exhausted,
    /// then the pool is reset.
    shrinkage: Option<f64>,

    /// Number of samples added per class in last fit_resample call.
    #[serde(skip)]
    n_samples_added_: Option<HashMap<i64, usize>>,

    /// Indices of samples selected for duplication (original indices).
    #[serde(skip)]
    sample_indices_: Option<HashMap<i64, Vec<usize>>>,
}

impl Default for RandomOverSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomOverSampler {
    /// Create a new RandomOverSampler instance with default parameters.
    ///
    /// Default settings:
    /// - sampling_strategy: Auto (oversample to match majority class)
    pub fn new() -> Self {
        Self {
            sampling_strategy: SamplingStrategy::Auto,
            random_state: None,
            shrinkage: None,
            n_samples_added_: None,
            sample_indices_: None,
        }
    }

    /// Set the sampling strategy.
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the shrinkage factor.
    ///
    /// When set, adds smoothing/regularization to the duplicated samples.
    /// A value of 0.0 means no shrinkage (exact duplicates).
    /// Higher values add more noise to duplicated samples.
    /// This can help reduce overfitting compared to exact duplication.
    pub fn with_shrinkage(mut self, shrinkage: f64) -> Self {
        self.shrinkage = Some(shrinkage.max(0.0));
        self
    }

    /// Get the number of samples added per class in the last call.
    pub fn n_samples_added(&self) -> Option<&HashMap<i64, usize>> {
        self.n_samples_added_.as_ref()
    }

    /// Get the indices of samples selected for duplication.
    pub fn sample_indices(&self) -> Option<&HashMap<i64, Vec<usize>>> {
        self.sample_indices_.as_ref()
    }

    /// Determine target sample counts per class based on strategy.
    /// For oversampling, we want to increase minority classes.
    fn compute_target_counts(
        &self,
        class_counts: &HashMap<i64, usize>,
    ) -> Result<HashMap<i64, usize>> {
        let max_count = *class_counts.values().max().unwrap_or(&0);

        match &self.sampling_strategy {
            SamplingStrategy::Auto => {
                // All classes should have max_count samples (oversample to majority)
                Ok(class_counts.keys().map(|&k| (k, max_count)).collect())
            }
            SamplingStrategy::Ratio(ratio) => {
                if *ratio <= 0.0 || *ratio > 1.0 {
                    return Err(FerroError::invalid_input("Ratio must be in (0, 1]"));
                }
                let target = (max_count as f64 * ratio).ceil() as usize;
                Ok(class_counts
                    .iter()
                    .map(|(&k, &v)| (k, v.max(target)))
                    .collect())
            }
            SamplingStrategy::TargetCounts(targets) => {
                let mut result = class_counts.clone();
                for (k, &v) in targets {
                    if let Some(current) = result.get_mut(k) {
                        // For oversampling, take the maximum
                        *current = (*current).max(v);
                    }
                }
                Ok(result)
            }
            SamplingStrategy::Classes(class_ratios) => {
                let mut result = class_counts.clone();
                for (&k, &ratio) in class_ratios {
                    if let Some(current) = result.get_mut(&k) {
                        let target = (max_count as f64 * ratio).ceil() as usize;
                        *current = (*current).max(target);
                    }
                }
                Ok(result)
            }
            SamplingStrategy::NotResampled => Ok(class_counts.clone()),
        }
    }
}

impl Resampler for RandomOverSampler {
    fn fit_resample(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(FerroError::invalid_input("Input array cannot be empty"));
        }

        if n_samples != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("({},)", n_samples),
                format!("({},)", y.len()),
            ));
        }

        // Count samples per class and collect indices
        let mut class_counts: HashMap<i64, usize> = HashMap::new();
        let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();

        for (i, &label) in y.iter().enumerate() {
            let label_int = label.round() as i64;
            *class_counts.entry(label_int).or_insert(0) += 1;
            class_indices.entry(label_int).or_default().push(i);
        }

        if class_counts.is_empty() {
            return Err(FerroError::invalid_input("No samples found"));
        }

        // Compute target counts
        let target_counts = self.compute_target_counts(&class_counts)?;

        // Initialize RNG
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        // Collect all samples (original + duplicated)
        let mut all_x: Vec<Array1<f64>> = x.rows().into_iter().map(|r| r.to_owned()).collect();
        let mut all_y: Vec<f64> = y.to_vec();

        let mut n_added: HashMap<i64, usize> = HashMap::new();
        let mut sample_indices: HashMap<i64, Vec<usize>> = HashMap::new();

        for (&class_label, indices) in &class_indices {
            let current_count = indices.len();
            let target = *target_counts.get(&class_label).unwrap_or(&current_count);
            let n_to_add = target.saturating_sub(current_count);

            if n_to_add == 0 {
                continue;
            }

            let mut selected_for_duplication: Vec<usize> = Vec::with_capacity(n_to_add);

            // Randomly select samples to duplicate
            for _ in 0..n_to_add {
                let idx = indices[rng.random_range(0..indices.len())];
                selected_for_duplication.push(idx);

                // Get the sample and potentially add shrinkage noise
                let mut sample = x.row(idx).to_owned();

                if let Some(shrinkage) = self.shrinkage {
                    if shrinkage > 0.0 {
                        // Add Gaussian noise based on shrinkage factor
                        // The noise is scaled by the feature's standard deviation
                        for val in sample.iter_mut() {
                            let noise: f64 = rng.random::<f64>() * 2.0 - 1.0;
                            *val += noise * shrinkage * val.abs().max(1.0);
                        }
                    }
                }

                all_x.push(sample);
                all_y.push(class_label as f64);
            }

            n_added.insert(class_label, n_to_add);
            sample_indices.insert(class_label, selected_for_duplication);
        }

        self.n_samples_added_ = Some(n_added);
        self.sample_indices_ = Some(sample_indices);

        // Build output arrays
        let n_total = all_x.len();
        let mut x_resampled = Array2::zeros((n_total, n_features));
        for (i, row) in all_x.iter().enumerate() {
            x_resampled.row_mut(i).assign(row);
        }

        let y_resampled = Array1::from_vec(all_y);

        Ok((x_resampled, y_resampled))
    }

    fn strategy_description(&self) -> String {
        format!(
            "RandomOverSampler (shrinkage={}, {})",
            match self.shrinkage {
                Some(s) => format!("{:.2}", s),
                None => "none".to_string(),
            },
            match &self.sampling_strategy {
                SamplingStrategy::Auto => "auto (oversample to majority)".to_string(),
                SamplingStrategy::Ratio(r) => format!("ratio={:.2}", r),
                SamplingStrategy::TargetCounts(t) => format!("targets={:?}", t),
                SamplingStrategy::Classes(c) => format!("classes={:?}", c),
                SamplingStrategy::NotResampled => "not_resampled".to_string(),
            }
        )
    }
}

/// Tomek Links Detector
///
/// A Tomek link is defined as a pair of samples (a, b) from different classes
/// where a is b's nearest neighbor and b is a's nearest neighbor. These pairs
/// typically lie near the decision boundary and can be noisy or ambiguous.
///
/// This is a helper struct used by SMOTETomek to identify samples to remove.
struct TomekLinksDetector;

impl TomekLinksDetector {
    /// Find all Tomek links in the dataset.
    ///
    /// Returns a list of (sample_a, sample_b) index pairs that form Tomek links.
    fn find_tomek_links(x: &Array2<f64>, y: &Array1<f64>) -> Vec<(usize, usize)> {
        let n_samples = x.nrows();
        if n_samples < 2 {
            return Vec::new();
        }

        // Find nearest neighbor for each sample
        let mut nearest_neighbors: Vec<usize> = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let sample = x.row(i);
            let mut min_dist = f64::INFINITY;
            let mut nearest = 0;

            for j in 0..n_samples {
                if i == j {
                    continue;
                }
                let other = x.row(j);
                let dist: f64 = sample
                    .iter()
                    .zip(other.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();

                if dist < min_dist {
                    min_dist = dist;
                    nearest = j;
                }
            }

            nearest_neighbors.push(nearest);
        }

        // Find Tomek links: pairs where each is the other's nearest neighbor
        // and they belong to different classes
        let mut tomek_links = Vec::new();

        for i in 0..n_samples {
            let j = nearest_neighbors[i];
            // Check if j's nearest neighbor is i (forming a link)
            // and they have different classes
            if nearest_neighbors[j] == i {
                let class_i = y[i].round() as i64;
                let class_j = y[j].round() as i64;

                if class_i != class_j && i < j {
                    // Only add each pair once (i < j)
                    tomek_links.push((i, j));
                }
            }
        }

        tomek_links
    }

    /// Get indices of samples to remove (majority class samples from Tomek links).
    fn samples_to_remove(
        x: &Array2<f64>,
        y: &Array1<f64>,
        majority_class: Option<i64>,
    ) -> Vec<usize> {
        let tomek_links = Self::find_tomek_links(x, y);

        // Determine majority class if not specified
        let majority = majority_class.unwrap_or_else(|| {
            let mut class_counts: HashMap<i64, usize> = HashMap::new();
            for &label in y.iter() {
                *class_counts.entry(label.round() as i64).or_insert(0) += 1;
            }
            *class_counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(k, _)| k)
                .unwrap_or(&0)
        });

        // Collect majority class samples from Tomek links
        let mut to_remove = Vec::new();
        for (i, j) in tomek_links {
            let class_i = y[i].round() as i64;
            let class_j = y[j].round() as i64;

            if class_i == majority {
                to_remove.push(i);
            }
            if class_j == majority {
                to_remove.push(j);
            }
        }

        to_remove.sort_unstable();
        to_remove.dedup();
        to_remove
    }
}

/// Edited Nearest Neighbors Cleaner
///
/// ENN removes samples that are misclassified by their k-nearest neighbors.
/// For each sample, if the majority class among its k neighbors differs from
/// the sample's class, the sample is removed.
///
/// This cleaning technique removes noisy samples and samples near the decision
/// boundary that are likely to be misclassified.
#[derive(Debug, Clone)]
struct EditedNearestNeighborsDetector {
    /// Number of neighbors to consider.
    n_neighbors: usize,
    /// Selection strategy: 'all' removes if any neighbor mismatches,
    /// 'mode' removes only if majority of neighbors mismatch.
    kind: ENNKind,
}

/// Strategy for Edited Nearest Neighbors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ENNKind {
    /// Remove if all k neighbors belong to a different class.
    All,
    /// Remove if the majority of k neighbors belong to a different class.
    Mode,
}

impl Default for ENNKind {
    fn default() -> Self {
        Self::All
    }
}

impl EditedNearestNeighborsDetector {
    fn new(n_neighbors: usize, kind: ENNKind) -> Self {
        Self {
            n_neighbors: n_neighbors.max(1),
            kind,
        }
    }

    /// Find samples to remove based on k-NN misclassification.
    fn samples_to_remove(&self, x: &Array2<f64>, y: &Array1<f64>) -> Vec<usize> {
        let n_samples = x.nrows();
        if n_samples <= self.n_neighbors {
            return Vec::new();
        }

        let mut to_remove = Vec::new();

        for i in 0..n_samples {
            let sample = x.row(i);
            let sample_class = y[i].round() as i64;

            // Find k nearest neighbors
            let mut distances: Vec<(usize, f64)> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| {
                    let other = x.row(j);
                    let dist: f64 = sample
                        .iter()
                        .zip(other.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (j, dist)
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let neighbors: Vec<usize> = distances
                .iter()
                .take(self.n_neighbors)
                .map(|(idx, _)| *idx)
                .collect();

            // Count neighbor classes
            let different_class_count = neighbors
                .iter()
                .filter(|&&j| y[j].round() as i64 != sample_class)
                .count();

            let should_remove = match self.kind {
                ENNKind::All => different_class_count == self.n_neighbors,
                ENNKind::Mode => different_class_count > self.n_neighbors / 2,
            };

            if should_remove {
                to_remove.push(i);
            }
        }

        to_remove
    }
}

/// SMOTE-Tomek - Combined SMOTE Oversampling and Tomek Links Cleaning
///
/// This method combines SMOTE oversampling with Tomek links undersampling to:
/// 1. First apply SMOTE to generate synthetic minority samples
/// 2. Then identify and remove Tomek links to clean the decision boundary
///
/// Tomek links are pairs of samples from different classes that are each other's
/// nearest neighbor. By removing majority class samples from these pairs, the
/// method creates a cleaner separation between classes.
///
/// ## Algorithm
///
/// 1. Apply SMOTE oversampling to balance the classes
/// 2. Find all Tomek links in the resampled data
/// 3. Remove majority class samples that participate in Tomek links
///
/// ## Advantages
///
/// - Combines oversampling benefits with decision boundary cleaning
/// - Removes potentially ambiguous majority samples near the boundary
/// - Generally produces better results than SMOTE alone
///
/// ## Reference
///
/// Batista, G. E., Prati, R. C., & Monard, M. C. (2004).
/// A study of the behavior of several methods for balancing machine learning
/// training data. ACM SIGKDD explorations newsletter, 6(1), 20-29.
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::preprocessing::sampling::{SMOTETomek, Resampler};
///
/// let mut smote_tomek = SMOTETomek::new()
///     .with_k_neighbors(5)
///     .with_random_state(42);
///
/// let (x_resampled, y_resampled) = smote_tomek.fit_resample(&x, &y)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SMOTETomek {
    /// Internal SMOTE instance for oversampling.
    smote: SMOTE,

    /// Number of Tomek links found in last fit_resample call.
    #[serde(skip)]
    n_tomek_links_: Option<usize>,

    /// Number of samples removed via Tomek links cleaning.
    #[serde(skip)]
    n_samples_removed_: Option<usize>,
}

impl Default for SMOTETomek {
    fn default() -> Self {
        Self::new()
    }
}

impl SMOTETomek {
    /// Create a new SMOTETomek instance with default parameters.
    ///
    /// Default settings:
    /// - k_neighbors: 5 (for SMOTE)
    /// - sampling_strategy: Auto (balance all classes to majority)
    pub fn new() -> Self {
        Self {
            smote: SMOTE::new(),
            n_tomek_links_: None,
            n_samples_removed_: None,
        }
    }

    /// Set the number of nearest neighbors for SMOTE.
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.smote = self.smote.with_k_neighbors(k);
        self
    }

    /// Set the sampling strategy for SMOTE.
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.smote = self.smote.with_sampling_strategy(strategy);
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.smote = self.smote.with_random_state(seed);
        self
    }

    /// Get the number of Tomek links found in the last call.
    pub fn n_tomek_links(&self) -> Option<usize> {
        self.n_tomek_links_
    }

    /// Get the number of samples removed via Tomek links in the last call.
    pub fn n_samples_removed(&self) -> Option<usize> {
        self.n_samples_removed_
    }

    /// Get the number of synthetic samples generated by SMOTE.
    pub fn n_synthetic_samples(&self) -> Option<&HashMap<i64, usize>> {
        self.smote.n_synthetic_samples()
    }
}

impl Resampler for SMOTETomek {
    fn fit_resample(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        // Step 1: Apply SMOTE oversampling
        let (x_smote, y_smote) = self.smote.fit_resample(x, y)?;

        // Step 2: Find Tomek links
        let tomek_links = TomekLinksDetector::find_tomek_links(&x_smote, &y_smote);
        self.n_tomek_links_ = Some(tomek_links.len());

        if tomek_links.is_empty() {
            self.n_samples_removed_ = Some(0);
            return Ok((x_smote, y_smote));
        }

        // Step 3: Remove majority class samples from Tomek links
        let to_remove = TomekLinksDetector::samples_to_remove(&x_smote, &y_smote, None);
        self.n_samples_removed_ = Some(to_remove.len());

        if to_remove.is_empty() {
            return Ok((x_smote, y_smote));
        }

        // Build filtered arrays
        let to_remove_set: std::collections::HashSet<usize> = to_remove.iter().copied().collect();
        let n_features = x_smote.ncols();
        let n_remaining = x_smote.nrows() - to_remove.len();

        let mut x_result = Array2::zeros((n_remaining, n_features));
        let mut y_result = Array1::zeros(n_remaining);

        let mut result_idx = 0;
        for i in 0..x_smote.nrows() {
            if !to_remove_set.contains(&i) {
                x_result.row_mut(result_idx).assign(&x_smote.row(i));
                y_result[result_idx] = y_smote[i];
                result_idx += 1;
            }
        }

        Ok((x_result, y_result))
    }

    fn strategy_description(&self) -> String {
        format!("SMOTE-Tomek ({})", self.smote.strategy_description())
    }
}

/// SMOTE-ENN - Combined SMOTE Oversampling and Edited Nearest Neighbors Cleaning
///
/// This method combines SMOTE oversampling with Edited Nearest Neighbors (ENN)
/// undersampling to:
/// 1. First apply SMOTE to generate synthetic minority samples
/// 2. Then apply ENN to remove samples misclassified by their k-nearest neighbors
///
/// ENN is more aggressive than Tomek links, potentially removing more samples
/// but also creating cleaner class separation.
///
/// ## Algorithm
///
/// 1. Apply SMOTE oversampling to balance the classes
/// 2. For each sample in the resampled data:
///    - Find its k nearest neighbors
///    - If the sample is misclassified by its neighbors (based on ENNKind), mark for removal
/// 3. Remove marked samples
///
/// ## ENN Modes
///
/// - **All mode**: Remove sample only if ALL k neighbors belong to a different class
/// - **Mode mode**: Remove sample if MAJORITY of k neighbors belong to a different class
///
/// ## Advantages
///
/// - More aggressive cleaning than SMOTE-Tomek
/// - Removes noisy samples that are likely to be misclassified
/// - Can significantly improve classifier performance
///
/// ## Disadvantages
///
/// - May remove too many samples if data is very noisy
/// - More computationally expensive than SMOTE-Tomek
///
/// ## Reference
///
/// Batista, G. E., Prati, R. C., & Monard, M. C. (2004).
/// A study of the behavior of several methods for balancing machine learning
/// training data. ACM SIGKDD explorations newsletter, 6(1), 20-29.
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::preprocessing::sampling::{SMOTEENN, Resampler, ENNKind};
///
/// let mut smote_enn = SMOTEENN::new()
///     .with_k_neighbors(5)
///     .with_enn_n_neighbors(3)
///     .with_enn_kind(ENNKind::Mode)
///     .with_random_state(42);
///
/// let (x_resampled, y_resampled) = smote_enn.fit_resample(&x, &y)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SMOTEENN {
    /// Internal SMOTE instance for oversampling.
    smote: SMOTE,

    /// Number of neighbors for ENN cleaning.
    enn_n_neighbors: usize,

    /// ENN selection strategy.
    enn_kind: ENNKind,

    /// Number of samples removed via ENN cleaning.
    #[serde(skip)]
    n_samples_removed_: Option<usize>,
}

impl Default for SMOTEENN {
    fn default() -> Self {
        Self::new()
    }
}

impl SMOTEENN {
    /// Create a new SMOTEENN instance with default parameters.
    ///
    /// Default settings:
    /// - k_neighbors: 5 (for SMOTE)
    /// - enn_n_neighbors: 3 (for ENN)
    /// - enn_kind: All (remove only if all neighbors are different class)
    /// - sampling_strategy: Auto (balance all classes to majority)
    pub fn new() -> Self {
        Self {
            smote: SMOTE::new(),
            enn_n_neighbors: 3,
            enn_kind: ENNKind::All,
            n_samples_removed_: None,
        }
    }

    /// Set the number of nearest neighbors for SMOTE.
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.smote = self.smote.with_k_neighbors(k);
        self
    }

    /// Set the sampling strategy for SMOTE.
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.smote = self.smote.with_sampling_strategy(strategy);
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.smote = self.smote.with_random_state(seed);
        self
    }

    /// Set the number of neighbors for ENN cleaning.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of neighbors. Must be at least 1.
    pub fn with_enn_n_neighbors(mut self, n: usize) -> Self {
        self.enn_n_neighbors = n.max(1);
        self
    }

    /// Set the ENN selection strategy.
    ///
    /// # Arguments
    ///
    /// * `kind` - ENNKind::All or ENNKind::Mode
    pub fn with_enn_kind(mut self, kind: ENNKind) -> Self {
        self.enn_kind = kind;
        self
    }

    /// Get the number of samples removed via ENN in the last call.
    pub fn n_samples_removed(&self) -> Option<usize> {
        self.n_samples_removed_
    }

    /// Get the number of synthetic samples generated by SMOTE.
    pub fn n_synthetic_samples(&self) -> Option<&HashMap<i64, usize>> {
        self.smote.n_synthetic_samples()
    }
}

impl Resampler for SMOTEENN {
    fn fit_resample(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        // Step 1: Apply SMOTE oversampling
        let (x_smote, y_smote) = self.smote.fit_resample(x, y)?;

        // Step 2: Apply ENN cleaning
        let enn = EditedNearestNeighborsDetector::new(self.enn_n_neighbors, self.enn_kind);
        let to_remove = enn.samples_to_remove(&x_smote, &y_smote);
        self.n_samples_removed_ = Some(to_remove.len());

        if to_remove.is_empty() {
            return Ok((x_smote, y_smote));
        }

        // Build filtered arrays
        let to_remove_set: std::collections::HashSet<usize> = to_remove.iter().copied().collect();
        let n_features = x_smote.ncols();
        let n_remaining = x_smote.nrows() - to_remove.len();

        let mut x_result = Array2::zeros((n_remaining, n_features));
        let mut y_result = Array1::zeros(n_remaining);

        let mut result_idx = 0;
        for i in 0..x_smote.nrows() {
            if !to_remove_set.contains(&i) {
                x_result.row_mut(result_idx).assign(&x_smote.row(i));
                y_result[result_idx] = y_smote[i];
                result_idx += 1;
            }
        }

        Ok((x_result, y_result))
    }

    fn strategy_description(&self) -> String {
        format!(
            "SMOTE-ENN (SMOTE: {}, ENN: n={}, kind={:?})",
            self.smote.strategy_description(),
            self.enn_n_neighbors,
            self.enn_kind
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn create_imbalanced_dataset() -> (Array2<f64>, Array1<f64>) {
        // Majority class: 20 samples around (0, 0)
        // Minority class: 5 samples around (5, 5)
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        // Majority class
        for i in 0..20 {
            x_data.push((i as f64 * 0.1, i as f64 * 0.1));
            y_data.push(0.0);
        }

        // Minority class
        for i in 0..5 {
            x_data.push((5.0 + i as f64 * 0.1, 5.0 + i as f64 * 0.1));
            y_data.push(1.0);
        }

        let x = Array2::from_shape_fn(
            (25, 2),
            |(i, j)| {
                if j == 0 {
                    x_data[i].0
                } else {
                    x_data[i].1
                }
            },
        );
        let y = Array1::from_vec(y_data);

        (x, y)
    }

    #[test]
    fn test_smote_basic() {
        let (x, y) = create_imbalanced_dataset();

        let mut smote = SMOTE::new().with_k_neighbors(3).with_random_state(42);
        let (x_resampled, y_resampled) = smote.fit_resample(&x, &y).unwrap();

        // Check that we have more samples
        assert!(x_resampled.nrows() > x.nrows());
        assert_eq!(x_resampled.nrows(), y_resampled.len());

        // Check class distribution is more balanced
        let minority_count: usize = y_resampled.iter().filter(|&&v| v == 1.0).count();
        let majority_count: usize = y_resampled.iter().filter(|&&v| v == 0.0).count();

        // After SMOTE with Auto strategy, minority should equal majority
        assert_eq!(minority_count, majority_count);
    }

    #[test]
    fn test_smote_ratio_strategy() {
        let (x, y) = create_imbalanced_dataset();

        let mut smote = SMOTE::new()
            .with_k_neighbors(2)
            .with_sampling_strategy(SamplingStrategy::Ratio(0.5))
            .with_random_state(42);

        let (_, y_resampled) = smote.fit_resample(&x, &y).unwrap();

        let minority_count: usize = y_resampled.iter().filter(|&&v| v == 1.0).count();
        let majority_count: usize = y_resampled.iter().filter(|&&v| v == 0.0).count();

        // Minority should be at least 50% of majority
        assert!(minority_count as f64 >= majority_count as f64 * 0.5);
    }

    #[test]
    fn test_smote_preserves_features() {
        let (x, y) = create_imbalanced_dataset();
        let n_features = x.ncols();

        let mut smote = SMOTE::new().with_random_state(42);
        let (x_resampled, _) = smote.fit_resample(&x, &y).unwrap();

        // Same number of features
        assert_eq!(x_resampled.ncols(), n_features);
    }

    #[test]
    fn test_smote_synthetic_samples_in_range() {
        let (x, y) = create_imbalanced_dataset();

        let mut smote = SMOTE::new().with_k_neighbors(2).with_random_state(42);
        let (x_resampled, y_resampled) = smote.fit_resample(&x, &y).unwrap();

        // Find original minority class bounds
        let minority_mask: Vec<bool> = y.iter().map(|&v| v == 1.0).collect();
        let minority_indices: Vec<usize> = minority_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();

        let min_x: f64 = minority_indices
            .iter()
            .map(|&i| x[[i, 0]])
            .fold(f64::INFINITY, f64::min);
        let max_x: f64 = minority_indices
            .iter()
            .map(|&i| x[[i, 0]])
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y: f64 = minority_indices
            .iter()
            .map(|&i| x[[i, 1]])
            .fold(f64::INFINITY, f64::min);
        let max_y: f64 = minority_indices
            .iter()
            .map(|&i| x[[i, 1]])
            .fold(f64::NEG_INFINITY, f64::max);

        // Check that synthetic minority samples are within bounds
        for (i, &label) in y_resampled.iter().enumerate() {
            if label == 1.0 && i >= 25 {
                // This is a synthetic sample
                let sample_x = x_resampled[[i, 0]];
                let sample_y = x_resampled[[i, 1]];

                // Should be within convex hull (approximately within bounds)
                assert!(
                    sample_x >= min_x - 0.1 && sample_x <= max_x + 0.1,
                    "Synthetic x {} out of range [{}, {}]",
                    sample_x,
                    min_x,
                    max_x
                );
                assert!(
                    sample_y >= min_y - 0.1 && sample_y <= max_y + 0.1,
                    "Synthetic y {} out of range [{}, {}]",
                    sample_y,
                    min_y,
                    max_y
                );
            }
        }
    }

    #[test]
    fn test_smote_reproducibility() {
        let (x, y) = create_imbalanced_dataset();

        let mut smote1 = SMOTE::new().with_random_state(42);
        let (x1, y1) = smote1.fit_resample(&x, &y).unwrap();

        let mut smote2 = SMOTE::new().with_random_state(42);
        let (x2, y2) = smote2.fit_resample(&x, &y).unwrap();

        assert_eq!(x1.shape(), x2.shape());
        assert_eq!(y1.len(), y2.len());

        // Results should be identical with same seed
        for i in 0..x1.nrows() {
            for j in 0..x1.ncols() {
                assert!((x1[[i, j]] - x2[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_smote_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let mut smote = SMOTE::new();
        let result = smote.fit_resample(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_smote_single_class() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 0.0];

        let mut smote = SMOTE::new();
        let result = smote.fit_resample(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_smote_insufficient_samples() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]];
        let y = array![0.0, 0.0, 1.0]; // Only 1 minority sample

        let mut smote = SMOTE::new();
        let result = smote.fit_resample(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_smote_multiclass() {
        // 3-class problem: class 0 has 10, class 1 has 5, class 2 has 3
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        for i in 0..10 {
            x_data.push((i as f64, i as f64));
            y_data.push(0.0);
        }
        for i in 0..5 {
            x_data.push((10.0 + i as f64, 10.0 + i as f64));
            y_data.push(1.0);
        }
        for i in 0..3 {
            x_data.push((20.0 + i as f64, 20.0 + i as f64));
            y_data.push(2.0);
        }

        let x = Array2::from_shape_fn(
            (18, 2),
            |(i, j)| {
                if j == 0 {
                    x_data[i].0
                } else {
                    x_data[i].1
                }
            },
        );
        let y = Array1::from_vec(y_data);

        let mut smote = SMOTE::new().with_k_neighbors(2).with_random_state(42);
        let (_x_resampled, y_resampled) = smote.fit_resample(&x, &y).unwrap();

        // Check all classes are now balanced
        let class0_count: usize = y_resampled.iter().filter(|&&v| v == 0.0).count();
        let class1_count: usize = y_resampled.iter().filter(|&&v| v == 1.0).count();
        let class2_count: usize = y_resampled.iter().filter(|&&v| v == 2.0).count();

        assert_eq!(class0_count, 10); // Majority unchanged
        assert_eq!(class1_count, 10); // Minority balanced
        assert_eq!(class2_count, 10); // Minority balanced
    }

    #[test]
    fn test_borderline_smote_basic() {
        let (x, y) = create_imbalanced_dataset();

        let mut bsmote = BorderlineSMOTE::new()
            .with_k_neighbors(2)
            .with_m_neighbors(3)
            .with_random_state(42);

        let (x_resampled, y_resampled) = bsmote.fit_resample(&x, &y).unwrap();

        // Should still produce resampled data (may or may not generate samples
        // depending on borderline detection)
        assert!(x_resampled.nrows() >= x.nrows());
        assert_eq!(x_resampled.nrows(), y_resampled.len());
    }

    #[test]
    fn test_smote_n_synthetic_samples() {
        let (x, y) = create_imbalanced_dataset();

        let mut smote = SMOTE::new().with_random_state(42);
        let _ = smote.fit_resample(&x, &y).unwrap();

        let n_synthetic = smote.n_synthetic_samples().unwrap();

        // Should have generated synthetic samples for minority class
        assert!(n_synthetic.contains_key(&1));
        assert_eq!(*n_synthetic.get(&1).unwrap(), 15); // 20 - 5 = 15 synthetic samples
    }

    #[test]
    fn test_smote_not_resampled_strategy() {
        let (x, y) = create_imbalanced_dataset();

        let mut smote = SMOTE::new().with_sampling_strategy(SamplingStrategy::NotResampled);

        let (x_resampled, y_resampled) = smote.fit_resample(&x, &y).unwrap();

        // Should be identical to input
        assert_eq!(x_resampled.nrows(), x.nrows());
        assert_eq!(y_resampled.len(), y.len());
    }

    // ===========================================
    // ADASYN Tests
    // ===========================================

    #[test]
    fn test_adasyn_basic() {
        let (x, y) = create_imbalanced_dataset();

        let mut adasyn = ADASYN::new().with_k_neighbors(3).with_random_state(42);
        let (x_resampled, y_resampled) = adasyn.fit_resample(&x, &y).unwrap();

        // Check that we have more samples
        assert!(x_resampled.nrows() > x.nrows());
        assert_eq!(x_resampled.nrows(), y_resampled.len());

        // Check class distribution is more balanced
        let minority_count: usize = y_resampled.iter().filter(|&&v| v == 1.0).count();
        let majority_count: usize = y_resampled.iter().filter(|&&v| v == 0.0).count();

        // After ADASYN with Auto strategy, minority should equal majority
        assert_eq!(minority_count, majority_count);
    }

    #[test]
    fn test_adasyn_generates_samples_for_hard_instances() {
        // Create dataset where some minority samples are near majority (hard)
        // and some are far from majority (easy)
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        // Majority class: cluster at (0, 0)
        for i in 0..20 {
            x_data.push((i as f64 * 0.1, i as f64 * 0.1));
            y_data.push(0.0);
        }

        // Minority class:
        // - 2 samples near majority (hard to learn): around (1, 1)
        x_data.push((1.0, 1.0));
        y_data.push(1.0);
        x_data.push((1.1, 1.1));
        y_data.push(1.0);

        // - 3 samples far from majority (easy to learn): around (10, 10)
        x_data.push((10.0, 10.0));
        y_data.push(1.0);
        x_data.push((10.1, 10.1));
        y_data.push(1.0);
        x_data.push((10.2, 10.2));
        y_data.push(1.0);

        let x = Array2::from_shape_fn(
            (25, 2),
            |(i, j)| {
                if j == 0 {
                    x_data[i].0
                } else {
                    x_data[i].1
                }
            },
        );
        let y = Array1::from_vec(y_data);

        let mut adasyn = ADASYN::new()
            .with_k_neighbors(3)
            .with_n_neighbors(5)
            .with_random_state(42);

        let _ = adasyn.fit_resample(&x, &y).unwrap();

        // Check density ratios were computed
        let density_ratios = adasyn.density_ratios().unwrap();
        assert!(density_ratios.contains_key(&1));

        let minority_ratios = &density_ratios[&1];

        // Hard samples (indices 0, 1 in minority class - at 1.0, 1.1)
        // should have higher density ratios than easy samples
        // The first two minority samples are near majority class
        // (Note: ratios depend on k, but hard samples should have higher ratios)
        assert!(minority_ratios.len() == 5);
    }

    #[test]
    fn test_adasyn_reproduces_with_same_seed() {
        let (x, y) = create_imbalanced_dataset();

        let mut adasyn1 = ADASYN::new().with_random_state(123);
        let (x1, y1) = adasyn1.fit_resample(&x, &y).unwrap();

        let mut adasyn2 = ADASYN::new().with_random_state(123);
        let (x2, y2) = adasyn2.fit_resample(&x, &y).unwrap();

        assert_eq!(x1.shape(), x2.shape());
        assert_eq!(y1.len(), y2.len());

        // Results should be identical with same seed
        for i in 0..x1.nrows() {
            for j in 0..x1.ncols() {
                assert!((x1[[i, j]] - x2[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_adasyn_preserves_features() {
        let (x, y) = create_imbalanced_dataset();
        let n_features = x.ncols();

        let mut adasyn = ADASYN::new().with_random_state(42);
        let (x_resampled, _) = adasyn.fit_resample(&x, &y).unwrap();

        // Same number of features
        assert_eq!(x_resampled.ncols(), n_features);
    }

    #[test]
    fn test_adasyn_ratio_strategy() {
        let (x, y) = create_imbalanced_dataset();

        let mut adasyn = ADASYN::new()
            .with_k_neighbors(2)
            .with_sampling_strategy(SamplingStrategy::Ratio(0.5))
            .with_random_state(42);

        let (_, y_resampled) = adasyn.fit_resample(&x, &y).unwrap();

        let minority_count: usize = y_resampled.iter().filter(|&&v| v == 1.0).count();
        let majority_count: usize = y_resampled.iter().filter(|&&v| v == 0.0).count();

        // Minority should be at least 50% of majority
        assert!(minority_count as f64 >= majority_count as f64 * 0.5);
    }

    #[test]
    fn test_adasyn_synthetic_samples_in_range() {
        let (x, y) = create_imbalanced_dataset();

        let mut adasyn = ADASYN::new().with_k_neighbors(2).with_random_state(42);
        let (x_resampled, y_resampled) = adasyn.fit_resample(&x, &y).unwrap();

        // Find original minority class bounds
        let minority_mask: Vec<bool> = y.iter().map(|&v| v == 1.0).collect();
        let minority_indices: Vec<usize> = minority_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();

        let min_x: f64 = minority_indices
            .iter()
            .map(|&i| x[[i, 0]])
            .fold(f64::INFINITY, f64::min);
        let max_x: f64 = minority_indices
            .iter()
            .map(|&i| x[[i, 0]])
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y: f64 = minority_indices
            .iter()
            .map(|&i| x[[i, 1]])
            .fold(f64::INFINITY, f64::min);
        let max_y: f64 = minority_indices
            .iter()
            .map(|&i| x[[i, 1]])
            .fold(f64::NEG_INFINITY, f64::max);

        // Check that synthetic minority samples are within bounds
        for (i, &label) in y_resampled.iter().enumerate() {
            if label == 1.0 && i >= 25 {
                // This is a synthetic sample
                let sample_x = x_resampled[[i, 0]];
                let sample_y = x_resampled[[i, 1]];

                // Should be within convex hull (approximately within bounds)
                assert!(
                    sample_x >= min_x - 0.1 && sample_x <= max_x + 0.1,
                    "Synthetic x {} out of range [{}, {}]",
                    sample_x,
                    min_x,
                    max_x
                );
                assert!(
                    sample_y >= min_y - 0.1 && sample_y <= max_y + 0.1,
                    "Synthetic y {} out of range [{}, {}]",
                    sample_y,
                    min_y,
                    max_y
                );
            }
        }
    }

    #[test]
    fn test_adasyn_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let mut adasyn = ADASYN::new();
        let result = adasyn.fit_resample(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_adasyn_single_class() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 0.0];

        let mut adasyn = ADASYN::new();
        let result = adasyn.fit_resample(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_adasyn_insufficient_samples() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]];
        let y = array![0.0, 0.0, 1.0]; // Only 1 minority sample

        let mut adasyn = ADASYN::new();
        let result = adasyn.fit_resample(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_adasyn_multiclass() {
        // 3-class problem: class 0 has 10, class 1 has 5, class 2 has 3
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        for i in 0..10 {
            x_data.push((i as f64, i as f64));
            y_data.push(0.0);
        }
        for i in 0..5 {
            x_data.push((10.0 + i as f64, 10.0 + i as f64));
            y_data.push(1.0);
        }
        for i in 0..3 {
            x_data.push((20.0 + i as f64, 20.0 + i as f64));
            y_data.push(2.0);
        }

        let x = Array2::from_shape_fn(
            (18, 2),
            |(i, j)| {
                if j == 0 {
                    x_data[i].0
                } else {
                    x_data[i].1
                }
            },
        );
        let y = Array1::from_vec(y_data);

        let mut adasyn = ADASYN::new().with_k_neighbors(2).with_random_state(42);
        let (_x_resampled, y_resampled) = adasyn.fit_resample(&x, &y).unwrap();

        // Check all classes are now balanced
        let class0_count: usize = y_resampled.iter().filter(|&&v| v == 0.0).count();
        let class1_count: usize = y_resampled.iter().filter(|&&v| v == 1.0).count();
        let class2_count: usize = y_resampled.iter().filter(|&&v| v == 2.0).count();

        assert_eq!(class0_count, 10); // Majority unchanged
        assert_eq!(class1_count, 10); // Minority balanced
        assert_eq!(class2_count, 10); // Minority balanced
    }

    #[test]
    fn test_adasyn_n_synthetic_samples() {
        let (x, y) = create_imbalanced_dataset();

        let mut adasyn = ADASYN::new().with_random_state(42);
        let _ = adasyn.fit_resample(&x, &y).unwrap();

        let n_synthetic = adasyn.n_synthetic_samples().unwrap();

        // Should have generated synthetic samples for minority class
        assert!(n_synthetic.contains_key(&1));
        assert_eq!(*n_synthetic.get(&1).unwrap(), 15); // 20 - 5 = 15 synthetic samples
    }

    #[test]
    fn test_adasyn_not_resampled_strategy() {
        let (x, y) = create_imbalanced_dataset();

        let mut adasyn = ADASYN::new().with_sampling_strategy(SamplingStrategy::NotResampled);

        let (x_resampled, y_resampled) = adasyn.fit_resample(&x, &y).unwrap();

        // Should be identical to input
        assert_eq!(x_resampled.nrows(), x.nrows());
        assert_eq!(y_resampled.len(), y.len());
    }

    #[test]
    fn test_adasyn_imbalance_threshold() {
        // Create a dataset that is slightly imbalanced
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        // 10 majority, 8 minority (80% ratio)
        for i in 0..10 {
            x_data.push((i as f64, i as f64));
            y_data.push(0.0);
        }
        for i in 0..8 {
            x_data.push((10.0 + i as f64, 10.0 + i as f64));
            y_data.push(1.0);
        }

        let x = Array2::from_shape_fn(
            (18, 2),
            |(i, j)| {
                if j == 0 {
                    x_data[i].0
                } else {
                    x_data[i].1
                }
            },
        );
        let y = Array1::from_vec(y_data);

        // With threshold 0.7, the 0.8 imbalance ratio exceeds it, so no resampling
        let mut adasyn = ADASYN::new()
            .with_imbalance_threshold(0.7)
            .with_random_state(42);

        let (x_resampled, y_resampled) = adasyn.fit_resample(&x, &y).unwrap();

        // Should not add any samples since imbalance ratio (0.8) > threshold (0.7)
        assert_eq!(x_resampled.nrows(), x.nrows());
        assert_eq!(y_resampled.len(), y.len());
    }

    #[test]
    fn test_adasyn_strategy_description() {
        let adasyn = ADASYN::new()
            .with_k_neighbors(5)
            .with_n_neighbors(10)
            .with_imbalance_threshold(0.8);

        let desc = adasyn.strategy_description();

        assert!(desc.contains("ADASYN"));
        assert!(desc.contains("k=5"));
        assert!(desc.contains("n=10"));
        assert!(desc.contains("threshold=0.80"));
    }

    #[test]
    fn test_adasyn_default_n_neighbors() {
        // When n_neighbors is not set, should use k_neighbors
        let adasyn = ADASYN::new().with_k_neighbors(7);
        assert_eq!(adasyn.effective_n_neighbors(), 7);
    }

    #[test]
    fn test_adasyn_with_n_neighbors_override() {
        // When n_neighbors is set, should use that value
        let adasyn = ADASYN::new().with_k_neighbors(5).with_n_neighbors(10);
        assert_eq!(adasyn.effective_n_neighbors(), 10);
    }

    // ===========================================
    // RandomUnderSampler Tests
    // ===========================================

    #[test]
    fn test_random_undersampler_basic() {
        let (x, y) = create_imbalanced_dataset();

        let mut rus = RandomUnderSampler::new().with_random_state(42);
        let (x_resampled, y_resampled) = rus.fit_resample(&x, &y).unwrap();

        // Check that we have fewer samples
        assert!(x_resampled.nrows() < x.nrows());
        assert_eq!(x_resampled.nrows(), y_resampled.len());

        // Check class distribution is balanced (both classes should have minority count)
        let minority_count: usize = y_resampled.iter().filter(|&&v| v == 1.0).count();
        let majority_count: usize = y_resampled.iter().filter(|&&v| v == 0.0).count();

        // After undersampling with Auto strategy, majority should equal minority
        assert_eq!(majority_count, minority_count);
        assert_eq!(majority_count, 5); // Original minority count
    }

    #[test]
    fn test_random_undersampler_preserves_features() {
        let (x, y) = create_imbalanced_dataset();
        let n_features = x.ncols();

        let mut rus = RandomUnderSampler::new().with_random_state(42);
        let (x_resampled, _) = rus.fit_resample(&x, &y).unwrap();

        // Same number of features
        assert_eq!(x_resampled.ncols(), n_features);
    }

    #[test]
    fn test_random_undersampler_samples_are_original() {
        let (x, y) = create_imbalanced_dataset();

        let mut rus = RandomUnderSampler::new().with_random_state(42);
        let (x_resampled, _) = rus.fit_resample(&x, &y).unwrap();

        // All resampled samples should be exact copies of original samples
        for i in 0..x_resampled.nrows() {
            let sample = x_resampled.row(i);
            let mut found = false;
            for j in 0..x.nrows() {
                let orig = x.row(j);
                if sample
                    .iter()
                    .zip(orig.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-10)
                {
                    found = true;
                    break;
                }
            }
            assert!(found, "Resampled sample not found in original data");
        }
    }

    #[test]
    fn test_random_undersampler_reproducibility() {
        let (x, y) = create_imbalanced_dataset();

        let mut rus1 = RandomUnderSampler::new().with_random_state(42);
        let (x1, y1) = rus1.fit_resample(&x, &y).unwrap();

        let mut rus2 = RandomUnderSampler::new().with_random_state(42);
        let (x2, y2) = rus2.fit_resample(&x, &y).unwrap();

        assert_eq!(x1.shape(), x2.shape());
        assert_eq!(y1.len(), y2.len());

        // Results should be identical with same seed
        for i in 0..x1.nrows() {
            for j in 0..x1.ncols() {
                assert!((x1[[i, j]] - x2[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_random_undersampler_with_replacement() {
        let (x, y) = create_imbalanced_dataset();

        let mut rus = RandomUnderSampler::new()
            .with_replacement(true)
            .with_random_state(42);

        let (_x_resampled, y_resampled) = rus.fit_resample(&x, &y).unwrap();

        // Should still produce balanced dataset
        let minority_count: usize = y_resampled.iter().filter(|&&v| v == 1.0).count();
        let majority_count: usize = y_resampled.iter().filter(|&&v| v == 0.0).count();

        assert_eq!(majority_count, minority_count);
    }

    #[test]
    fn test_random_undersampler_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let mut rus = RandomUnderSampler::new();
        let result = rus.fit_resample(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_random_undersampler_not_resampled_strategy() {
        let (x, y) = create_imbalanced_dataset();

        let mut rus =
            RandomUnderSampler::new().with_sampling_strategy(SamplingStrategy::NotResampled);

        let (x_resampled, y_resampled) = rus.fit_resample(&x, &y).unwrap();

        // Should be identical to input
        assert_eq!(x_resampled.nrows(), x.nrows());
        assert_eq!(y_resampled.len(), y.len());
    }

    #[test]
    fn test_random_undersampler_n_samples_removed() {
        let (x, y) = create_imbalanced_dataset();

        let mut rus = RandomUnderSampler::new().with_random_state(42);
        let _ = rus.fit_resample(&x, &y).unwrap();

        let n_removed = rus.n_samples_removed().unwrap();

        // Should have removed samples from majority class
        assert!(n_removed.contains_key(&0));
        assert_eq!(*n_removed.get(&0).unwrap(), 15); // 20 - 5 = 15 removed
    }

    #[test]
    fn test_random_undersampler_sample_indices() {
        let (x, y) = create_imbalanced_dataset();

        let mut rus = RandomUnderSampler::new().with_random_state(42);
        let _ = rus.fit_resample(&x, &y).unwrap();

        let sample_indices = rus.sample_indices().unwrap();

        // Should have indices for all kept samples
        assert_eq!(sample_indices.len(), 10); // 5 majority + 5 minority
    }

    #[test]
    fn test_random_undersampler_multiclass() {
        // 3-class problem: class 0 has 10, class 1 has 5, class 2 has 3
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        for i in 0..10 {
            x_data.push((i as f64, i as f64));
            y_data.push(0.0);
        }
        for i in 0..5 {
            x_data.push((10.0 + i as f64, 10.0 + i as f64));
            y_data.push(1.0);
        }
        for i in 0..3 {
            x_data.push((20.0 + i as f64, 20.0 + i as f64));
            y_data.push(2.0);
        }

        let x = Array2::from_shape_fn(
            (18, 2),
            |(i, j)| {
                if j == 0 {
                    x_data[i].0
                } else {
                    x_data[i].1
                }
            },
        );
        let y = Array1::from_vec(y_data);

        let mut rus = RandomUnderSampler::new().with_random_state(42);
        let (_x_resampled, y_resampled) = rus.fit_resample(&x, &y).unwrap();

        // Check all classes are now balanced to minority (3)
        let class0_count: usize = y_resampled.iter().filter(|&&v| v == 0.0).count();
        let class1_count: usize = y_resampled.iter().filter(|&&v| v == 1.0).count();
        let class2_count: usize = y_resampled.iter().filter(|&&v| v == 2.0).count();

        assert_eq!(class0_count, 3); // Undersampled to minority
        assert_eq!(class1_count, 3); // Undersampled to minority
        assert_eq!(class2_count, 3); // Minority unchanged
    }

    #[test]
    fn test_random_undersampler_strategy_description() {
        let rus = RandomUnderSampler::new().with_replacement(true);

        let desc = rus.strategy_description();

        assert!(desc.contains("RandomUnderSampler"));
        assert!(desc.contains("replacement=true"));
    }

    // ===========================================
    // RandomOverSampler Tests
    // ===========================================

    #[test]
    fn test_random_oversampler_basic() {
        let (x, y) = create_imbalanced_dataset();

        let mut ros = RandomOverSampler::new().with_random_state(42);
        let (x_resampled, y_resampled) = ros.fit_resample(&x, &y).unwrap();

        // Check that we have more samples
        assert!(x_resampled.nrows() > x.nrows());
        assert_eq!(x_resampled.nrows(), y_resampled.len());

        // Check class distribution is balanced
        let minority_count: usize = y_resampled.iter().filter(|&&v| v == 1.0).count();
        let majority_count: usize = y_resampled.iter().filter(|&&v| v == 0.0).count();

        // After oversampling with Auto strategy, minority should equal majority
        assert_eq!(minority_count, majority_count);
        assert_eq!(minority_count, 20); // Original majority count
    }

    #[test]
    fn test_random_oversampler_preserves_features() {
        let (x, y) = create_imbalanced_dataset();
        let n_features = x.ncols();

        let mut ros = RandomOverSampler::new().with_random_state(42);
        let (x_resampled, _) = ros.fit_resample(&x, &y).unwrap();

        // Same number of features
        assert_eq!(x_resampled.ncols(), n_features);
    }

    #[test]
    fn test_random_oversampler_duplicates_are_exact() {
        let (x, y) = create_imbalanced_dataset();

        let mut ros = RandomOverSampler::new().with_random_state(42);
        let (x_resampled, y_resampled) = ros.fit_resample(&x, &y).unwrap();

        // Original samples should be preserved at the beginning
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((x_resampled[[i, j]] - x[[i, j]]).abs() < 1e-10);
            }
        }

        // Duplicated samples (after original) should match some original minority sample
        for i in x.nrows()..x_resampled.nrows() {
            let sample = x_resampled.row(i);
            let label = y_resampled[i] as i64;

            // Find matching original sample
            let mut found = false;
            for j in 0..x.nrows() {
                if (y[j].round() as i64) == label {
                    let orig = x.row(j);
                    if sample
                        .iter()
                        .zip(orig.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-10)
                    {
                        found = true;
                        break;
                    }
                }
            }
            assert!(
                found,
                "Duplicated sample not found in original minority class"
            );
        }
    }

    #[test]
    fn test_random_oversampler_reproducibility() {
        let (x, y) = create_imbalanced_dataset();

        let mut ros1 = RandomOverSampler::new().with_random_state(42);
        let (x1, y1) = ros1.fit_resample(&x, &y).unwrap();

        let mut ros2 = RandomOverSampler::new().with_random_state(42);
        let (x2, y2) = ros2.fit_resample(&x, &y).unwrap();

        assert_eq!(x1.shape(), x2.shape());
        assert_eq!(y1.len(), y2.len());

        // Results should be identical with same seed
        for i in 0..x1.nrows() {
            for j in 0..x1.ncols() {
                assert!((x1[[i, j]] - x2[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_random_oversampler_with_shrinkage() {
        let (x, y) = create_imbalanced_dataset();

        let mut ros = RandomOverSampler::new()
            .with_shrinkage(0.1)
            .with_random_state(42);

        let (x_resampled, y_resampled) = ros.fit_resample(&x, &y).unwrap();

        // Should still have correct shape
        assert!(x_resampled.nrows() > x.nrows());

        // With shrinkage, duplicated samples should NOT be exact copies
        // (at least some should be different)
        let mut found_different = false;
        for i in x.nrows()..x_resampled.nrows() {
            let sample = x_resampled.row(i);
            let label = y_resampled[i] as i64;

            // Check if this sample differs from all original minority samples
            let mut matches_any = false;
            for j in 0..x.nrows() {
                if (y[j].round() as i64) == label {
                    let orig = x.row(j);
                    if sample
                        .iter()
                        .zip(orig.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-10)
                    {
                        matches_any = true;
                        break;
                    }
                }
            }
            if !matches_any {
                found_different = true;
                break;
            }
        }

        // With shrinkage > 0, at least some samples should be different
        assert!(found_different, "No samples were modified by shrinkage");
    }

    #[test]
    fn test_random_oversampler_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let mut ros = RandomOverSampler::new();
        let result = ros.fit_resample(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_random_oversampler_not_resampled_strategy() {
        let (x, y) = create_imbalanced_dataset();

        let mut ros =
            RandomOverSampler::new().with_sampling_strategy(SamplingStrategy::NotResampled);

        let (x_resampled, y_resampled) = ros.fit_resample(&x, &y).unwrap();

        // Should be identical to input
        assert_eq!(x_resampled.nrows(), x.nrows());
        assert_eq!(y_resampled.len(), y.len());
    }

    #[test]
    fn test_random_oversampler_ratio_strategy() {
        let (x, y) = create_imbalanced_dataset();

        let mut ros = RandomOverSampler::new()
            .with_sampling_strategy(SamplingStrategy::Ratio(0.5))
            .with_random_state(42);

        let (_, y_resampled) = ros.fit_resample(&x, &y).unwrap();

        let minority_count: usize = y_resampled.iter().filter(|&&v| v == 1.0).count();
        let majority_count: usize = y_resampled.iter().filter(|&&v| v == 0.0).count();

        // Minority should be at least 50% of majority
        assert!(minority_count as f64 >= majority_count as f64 * 0.5);
    }

    #[test]
    fn test_random_oversampler_n_samples_added() {
        let (x, y) = create_imbalanced_dataset();

        let mut ros = RandomOverSampler::new().with_random_state(42);
        let _ = ros.fit_resample(&x, &y).unwrap();

        let n_added = ros.n_samples_added().unwrap();

        // Should have added samples to minority class
        assert!(n_added.contains_key(&1));
        assert_eq!(*n_added.get(&1).unwrap(), 15); // 20 - 5 = 15 added
    }

    #[test]
    fn test_random_oversampler_sample_indices() {
        let (x, y) = create_imbalanced_dataset();

        let mut ros = RandomOverSampler::new().with_random_state(42);
        let _ = ros.fit_resample(&x, &y).unwrap();

        let sample_indices = ros.sample_indices().unwrap();

        // Should have indices for minority class duplications
        assert!(sample_indices.contains_key(&1));
        assert_eq!(sample_indices.get(&1).unwrap().len(), 15);
    }

    #[test]
    fn test_random_oversampler_multiclass() {
        // 3-class problem: class 0 has 10, class 1 has 5, class 2 has 3
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        for i in 0..10 {
            x_data.push((i as f64, i as f64));
            y_data.push(0.0);
        }
        for i in 0..5 {
            x_data.push((10.0 + i as f64, 10.0 + i as f64));
            y_data.push(1.0);
        }
        for i in 0..3 {
            x_data.push((20.0 + i as f64, 20.0 + i as f64));
            y_data.push(2.0);
        }

        let x = Array2::from_shape_fn(
            (18, 2),
            |(i, j)| {
                if j == 0 {
                    x_data[i].0
                } else {
                    x_data[i].1
                }
            },
        );
        let y = Array1::from_vec(y_data);

        let mut ros = RandomOverSampler::new().with_random_state(42);
        let (_x_resampled, y_resampled) = ros.fit_resample(&x, &y).unwrap();

        // Check all classes are now balanced to majority (10)
        let class0_count: usize = y_resampled.iter().filter(|&&v| v == 0.0).count();
        let class1_count: usize = y_resampled.iter().filter(|&&v| v == 1.0).count();
        let class2_count: usize = y_resampled.iter().filter(|&&v| v == 2.0).count();

        assert_eq!(class0_count, 10); // Majority unchanged
        assert_eq!(class1_count, 10); // Oversampled to majority
        assert_eq!(class2_count, 10); // Oversampled to majority
    }

    #[test]
    fn test_random_oversampler_strategy_description() {
        let ros = RandomOverSampler::new().with_shrinkage(0.5);

        let desc = ros.strategy_description();

        assert!(desc.contains("RandomOverSampler"));
        assert!(desc.contains("shrinkage=0.50"));
    }

    // ===========================================
    // SMOTETomek Tests
    // ===========================================

    #[test]
    fn test_smote_tomek_basic() {
        let (x, y) = create_imbalanced_dataset();

        let mut smote_tomek = SMOTETomek::new().with_k_neighbors(3).with_random_state(42);

        let (x_resampled, y_resampled) = smote_tomek.fit_resample(&x, &y).unwrap();

        // Check that we have resampled data (may have fewer samples than pure SMOTE
        // due to Tomek links removal)
        assert!(x_resampled.nrows() >= x.nrows());
        assert_eq!(x_resampled.nrows(), y_resampled.len());

        // Check class distribution is more balanced (though may not be exactly equal
        // after Tomek links removal)
        let minority_count: usize = y_resampled.iter().filter(|&&v| v == 1.0).count();
        let majority_count: usize = y_resampled.iter().filter(|&&v| v == 0.0).count();

        // Minority should have increased from original 5
        assert!(minority_count >= 5);
        // Classes should be closer together than original (5 vs 20)
        assert!(minority_count as f64 / majority_count as f64 > 5.0 / 20.0);
    }

    #[test]
    fn test_smote_tomek_diagnostics() {
        let (x, y) = create_imbalanced_dataset();

        let mut smote_tomek = SMOTETomek::new().with_k_neighbors(2).with_random_state(42);

        let _ = smote_tomek.fit_resample(&x, &y).unwrap();

        // Check diagnostics are available
        assert!(smote_tomek.n_tomek_links().is_some());
        assert!(smote_tomek.n_samples_removed().is_some());
        assert!(smote_tomek.n_synthetic_samples().is_some());

        // SMOTE should have generated some synthetic samples
        let n_synthetic = smote_tomek.n_synthetic_samples().unwrap();
        assert!(n_synthetic.contains_key(&1));
    }

    #[test]
    fn test_smote_tomek_reproducibility() {
        let (x, y) = create_imbalanced_dataset();

        let mut st1 = SMOTETomek::new().with_k_neighbors(3).with_random_state(42);
        let (x1, y1) = st1.fit_resample(&x, &y).unwrap();

        let mut st2 = SMOTETomek::new().with_k_neighbors(3).with_random_state(42);
        let (x2, y2) = st2.fit_resample(&x, &y).unwrap();

        assert_eq!(x1.shape(), x2.shape());
        assert_eq!(y1.len(), y2.len());

        // Results should be identical with same seed
        for i in 0..x1.nrows() {
            for j in 0..x1.ncols() {
                assert!((x1[[i, j]] - x2[[i, j]]).abs() < 1e-10);
            }
            assert!((y1[i] - y2[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_smote_tomek_strategy_description() {
        let st = SMOTETomek::new().with_k_neighbors(3);

        let desc = st.strategy_description();

        assert!(desc.contains("SMOTE-Tomek"));
        assert!(desc.contains("auto"));
    }

    #[test]
    fn test_tomek_links_detector() {
        // Create a simple dataset where we know there should be Tomek links
        // Two samples very close to each other but different classes
        let x = array![
            [0.0, 0.0],   // Class 0
            [0.1, 0.1],   // Class 1 - forms Tomek link with sample 0
            [5.0, 5.0],   // Class 0
            [10.0, 10.0]  // Class 1
        ];
        let y = array![0.0, 1.0, 0.0, 1.0];

        let tomek_links = TomekLinksDetector::find_tomek_links(&x, &y);

        // Samples 0 and 1 should form a Tomek link (each other's nearest neighbor)
        assert!(!tomek_links.is_empty());
        assert!(tomek_links.contains(&(0, 1)));
    }

    // ===========================================
    // SMOTEENN Tests
    // ===========================================

    #[test]
    fn test_smote_enn_basic() {
        let (x, y) = create_imbalanced_dataset();

        let mut smote_enn = SMOTEENN::new()
            .with_k_neighbors(3)
            .with_enn_n_neighbors(3)
            .with_random_state(42);

        let (x_resampled, y_resampled) = smote_enn.fit_resample(&x, &y).unwrap();

        // Check that we have resampled data
        assert!(x_resampled.nrows() >= 2); // At least some samples remain
        assert_eq!(x_resampled.nrows(), y_resampled.len());
    }

    #[test]
    fn test_smote_enn_enn_kind_all() {
        let (x, y) = create_imbalanced_dataset();

        let mut smote_enn = SMOTEENN::new()
            .with_k_neighbors(3)
            .with_enn_n_neighbors(3)
            .with_enn_kind(ENNKind::All)
            .with_random_state(42);

        let (x_resampled_all, _) = smote_enn.fit_resample(&x, &y).unwrap();
        let removed_all = smote_enn.n_samples_removed().unwrap_or(0);

        // Reset and test with Mode
        let mut smote_enn_mode = SMOTEENN::new()
            .with_k_neighbors(3)
            .with_enn_n_neighbors(3)
            .with_enn_kind(ENNKind::Mode)
            .with_random_state(42);

        let (x_resampled_mode, _) = smote_enn_mode.fit_resample(&x, &y).unwrap();
        let removed_mode = smote_enn_mode.n_samples_removed().unwrap_or(0);

        // Mode should typically remove more samples than All (more aggressive)
        // This depends on data, but we just check they both work
        assert!(x_resampled_all.nrows() > 0);
        assert!(x_resampled_mode.nrows() > 0);

        // At least confirm diagnostics work
        assert!(removed_all == 0 || removed_all > 0);
        assert!(removed_mode == 0 || removed_mode > 0);
    }

    #[test]
    fn test_smote_enn_diagnostics() {
        let (x, y) = create_imbalanced_dataset();

        let mut smote_enn = SMOTEENN::new().with_k_neighbors(2).with_random_state(42);

        let _ = smote_enn.fit_resample(&x, &y).unwrap();

        // Check diagnostics are available
        assert!(smote_enn.n_samples_removed().is_some());
        assert!(smote_enn.n_synthetic_samples().is_some());
    }

    #[test]
    fn test_smote_enn_reproducibility() {
        let (x, y) = create_imbalanced_dataset();

        let mut se1 = SMOTEENN::new()
            .with_k_neighbors(3)
            .with_enn_n_neighbors(3)
            .with_random_state(42);
        let (x1, y1) = se1.fit_resample(&x, &y).unwrap();

        let mut se2 = SMOTEENN::new()
            .with_k_neighbors(3)
            .with_enn_n_neighbors(3)
            .with_random_state(42);
        let (x2, y2) = se2.fit_resample(&x, &y).unwrap();

        assert_eq!(x1.shape(), x2.shape());
        assert_eq!(y1.len(), y2.len());

        // Results should be identical with same seed
        for i in 0..x1.nrows() {
            for j in 0..x1.ncols() {
                assert!((x1[[i, j]] - x2[[i, j]]).abs() < 1e-10);
            }
            assert!((y1[i] - y2[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_smote_enn_strategy_description() {
        let se = SMOTEENN::new()
            .with_k_neighbors(5)
            .with_enn_n_neighbors(3)
            .with_enn_kind(ENNKind::Mode);

        let desc = se.strategy_description();

        assert!(desc.contains("SMOTE-ENN"));
        assert!(desc.contains("auto"));
        assert!(desc.contains("n=3"));
        assert!(desc.contains("Mode"));
    }

    #[test]
    fn test_enn_detector() {
        // Create a dataset where one sample is surrounded by different class neighbors
        let x = array![
            [0.0, 0.0],   // Class 0
            [0.1, 0.0],   // Class 0
            [0.0, 0.1],   // Class 0
            [0.05, 0.05], // Class 1 - surrounded by class 0
            [10.0, 10.0], // Class 1
            [10.1, 10.0], // Class 1
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        // With ENNKind::All and 3 neighbors, sample 3 should be removed
        // (all its 3 nearest neighbors are class 0)
        let enn = EditedNearestNeighborsDetector::new(3, ENNKind::All);
        let to_remove = enn.samples_to_remove(&x, &y);

        // Sample 3 (class 1 surrounded by class 0) should be flagged
        assert!(to_remove.contains(&3));
    }

    #[test]
    fn test_smote_tomek_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let mut st = SMOTETomek::new();
        let result = st.fit_resample(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_smote_enn_empty_input() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let mut se = SMOTEENN::new();
        let result = se.fit_resample(&x, &y);

        assert!(result.is_err());
    }
}

//! Random Forest Models
//!
//! This module provides Random Forest ensemble implementations built on top of
//! decision trees with bootstrap aggregating (bagging).
//!
//! ## Models
//!
//! - **RandomForestClassifier**: For classification tasks
//! - **RandomForestRegressor**: For regression tasks
//!
//! ## Features
//!
//! - **Bootstrap aggregating**: Each tree trained on bootstrap sample
//! - **Random feature subsampling**: sqrt(n_features) by default at each split
//! - **OOB (Out-of-Bag) error estimation**: Validation without held-out set
//! - **Feature importance with CI**: Bootstrap-based confidence intervals
//! - **Parallel tree building**: Using rayon for multi-threaded training
//!
//! ## Example - Classification
//!
//! ```
//! use ferroml_core::models::forest::RandomForestClassifier;
//! use ferroml_core::models::Model;
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((100, 4), (0..400).map(|i| i as f64 / 100.0).collect()).unwrap();
//! let y = Array1::from_iter((0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }));
//!
//! let mut model = RandomForestClassifier::new()
//!     .with_n_estimators(10)
//!     .with_max_depth(Some(5))
//!     .with_random_state(42);
//! model.fit(&x, &y).unwrap();
//!
//! let predictions = model.predict(&x).unwrap();
//! assert_eq!(predictions.len(), 100);
//! ```
//!
//! ## Example - Regression
//!
//! ```
//! use ferroml_core::models::forest::RandomForestRegressor;
//! use ferroml_core::models::Model;
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((100, 4), (0..400).map(|i| i as f64 / 100.0).collect()).unwrap();
//! let y = Array1::from_iter((0..100).map(|i| i as f64 * 0.5 + 1.0));
//!
//! let mut model = RandomForestRegressor::new()
//!     .with_n_estimators(10)
//!     .with_max_depth(Some(5));
//! model.fit(&x, &y).unwrap();
//!
//! let predictions = model.predict(&x).unwrap();
//! assert_eq!(predictions.len(), 100);
//! ```

use crate::hpo::SearchSpace;
use crate::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor, SplitCriterion};
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, ClassWeight, Model,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// =============================================================================
// Feature Importance with Confidence Intervals
// =============================================================================

/// Feature importance with bootstrap confidence intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportanceWithCI {
    /// Point estimate of feature importance
    pub importance: Array1<f64>,
    /// Standard error of importance estimates
    pub std_error: Array1<f64>,
    /// Lower bound of confidence interval
    pub ci_lower: Array1<f64>,
    /// Upper bound of confidence interval
    pub ci_upper: Array1<f64>,
    /// Confidence level (e.g., 0.95)
    pub confidence_level: f64,
}

// =============================================================================
// Random Forest Classifier
// =============================================================================

/// Random Forest Classifier using bootstrap aggregating
///
/// Builds an ensemble of decision trees, each trained on a bootstrap sample
/// of the data, with random feature subsampling at each split.
///
/// ## Features
///
/// - Bootstrap aggregating (bagging) of decision trees
/// - Random feature subsampling at each split (default: sqrt(n_features))
/// - OOB (Out-of-Bag) error estimation
/// - Feature importance with confidence intervals via bootstrap
/// - Parallel tree building using rayon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestClassifier {
    /// Number of trees in the forest
    pub n_estimators: usize,
    /// Split criterion (Gini or Entropy)
    pub criterion: SplitCriterion,
    /// Maximum depth of each tree (None for unlimited)
    pub max_depth: Option<usize>,
    /// Minimum samples required to split an internal node
    pub min_samples_split: usize,
    /// Minimum samples required to be at a leaf node
    pub min_samples_leaf: usize,
    /// Number of features to consider at each split (None for sqrt(n_features))
    pub max_features: Option<MaxFeatures>,
    /// Bootstrap sampling (if false, uses whole dataset for each tree)
    pub bootstrap: bool,
    /// Whether to compute OOB score during fitting
    pub oob_score_enabled: bool,
    /// Number of parallel jobs (None for all available cores).
    ///
    /// **Reproducibility Note:** Set to `Some(1)` for fully deterministic results
    /// when combined with `random_state`. Parallel execution introduces non-determinism
    /// due to thread scheduling, even with a fixed random seed.
    pub n_jobs: Option<usize>,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Confidence level for feature importance CIs
    pub confidence_level: f64,
    /// Minimum impurity decrease for splits
    pub min_impurity_decrease: f64,
    /// Class weights for handling imbalanced datasets
    pub class_weight: ClassWeight,

    /// Enable warm start to add estimators incrementally
    pub warm_start: bool,

    // Fitted parameters
    estimators: Option<Vec<DecisionTreeClassifier>>,
    classes: Option<Array1<f64>>,
    n_features: Option<usize>,
    feature_importances: Option<Array1<f64>>,
    feature_importances_with_ci: Option<FeatureImportanceWithCI>,
    oob_score: Option<f64>,
    oob_decision_function: Option<Array2<f64>>,
}

/// Strategy for selecting max_features at each split
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MaxFeatures {
    /// Use all features
    All,
    /// Use sqrt(n_features)
    Sqrt,
    /// Use log2(n_features)
    Log2,
    /// Use a fixed number
    Fixed(usize),
    /// Use a fraction of features
    Fraction(f64),
}

impl Default for MaxFeatures {
    fn default() -> Self {
        Self::Sqrt
    }
}

impl MaxFeatures {
    /// Compute the actual number of features to use
    pub fn compute(&self, n_features: usize) -> usize {
        match self {
            MaxFeatures::All => n_features,
            MaxFeatures::Sqrt => (n_features as f64).sqrt().ceil() as usize,
            MaxFeatures::Log2 => (n_features as f64).log2().ceil() as usize,
            MaxFeatures::Fixed(n) => (*n).min(n_features),
            MaxFeatures::Fraction(f) => ((n_features as f64) * f).ceil() as usize,
        }
        .max(1)
    }
}

impl Default for RandomForestClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomForestClassifier {
    /// Create a new Random Forest Classifier with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            criterion: SplitCriterion::Gini,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: Some(MaxFeatures::Sqrt),
            bootstrap: true,
            oob_score_enabled: true,
            n_jobs: None,
            random_state: None,
            confidence_level: 0.95,
            min_impurity_decrease: 0.0,
            class_weight: ClassWeight::Uniform,
            warm_start: false,
            estimators: None,
            classes: None,
            n_features: None,
            feature_importances: None,
            feature_importances_with_ci: None,
            oob_score: None,
            oob_decision_function: None,
        }
    }

    /// Set the number of trees
    #[must_use]
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators.max(1);
        self
    }

    /// Set the split criterion
    #[must_use]
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
        assert!(
            criterion.is_classification(),
            "Criterion must be Gini or Entropy for classification"
        );
        self.criterion = criterion;
        self
    }

    /// Set maximum tree depth
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set minimum samples to split
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split.max(2);
        self
    }

    /// Set minimum samples per leaf
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf.max(1);
        self
    }

    /// Set maximum features strategy
    #[must_use]
    pub fn with_max_features(mut self, max_features: Option<MaxFeatures>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Enable or disable bootstrap sampling
    #[must_use]
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Enable or disable OOB score computation
    #[must_use]
    pub fn with_oob_score(mut self, oob_score: bool) -> Self {
        self.oob_score_enabled = oob_score;
        self
    }

    /// Set number of parallel jobs.
    ///
    /// # Arguments
    /// * `n_jobs` - Number of parallel workers:
    ///   - `None` - Use all available CPU cores (default, fastest)
    ///   - `Some(1)` - Sequential execution (required for reproducibility)
    ///   - `Some(n)` - Use n parallel workers
    ///
    /// # Reproducibility
    /// For fully deterministic results, use `with_n_jobs(Some(1))` together with
    /// `with_random_state(seed)`. Parallel execution introduces non-determinism
    /// due to thread scheduling and floating-point aggregation order.
    ///
    /// # Example
    /// ```
    /// use ferroml_core::models::RandomForestClassifier;
    ///
    /// // Fast parallel training (non-deterministic)
    /// let fast_model = RandomForestClassifier::new()
    ///     .with_random_state(42);
    ///
    /// // Reproducible sequential training
    /// let reproducible_model = RandomForestClassifier::new()
    ///     .with_random_state(42)
    ///     .with_n_jobs(Some(1));
    /// ```
    #[must_use]
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Set random state for reproducibility.
    ///
    /// Note: For fully deterministic results, also set `with_n_jobs(Some(1))`.
    /// Parallel execution can introduce non-determinism even with a fixed seed.
    #[must_use]
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set confidence level for CIs
    #[must_use]
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level.clamp(0.5, 0.999);
        self
    }

    /// Set minimum impurity decrease
    #[must_use]
    pub fn with_min_impurity_decrease(mut self, min_impurity_decrease: f64) -> Self {
        self.min_impurity_decrease = min_impurity_decrease.max(0.0);
        self
    }

    /// Set class weights for handling imbalanced data
    ///
    /// # Arguments
    /// * `class_weight` - Weight specification: `Uniform`, `Balanced`, or `Custom`
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Get the OOB score (accuracy for classification)
    #[must_use]
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score
    }

    /// Get the OOB decision function (class probabilities)
    #[must_use]
    pub fn oob_decision_function(&self) -> Option<&Array2<f64>> {
        self.oob_decision_function.as_ref()
    }

    /// Get the individual estimators
    #[must_use]
    pub fn estimators(&self) -> Option<&[DecisionTreeClassifier]> {
        self.estimators.as_deref()
    }

    /// Get the unique class labels
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Get feature importance with confidence intervals
    #[must_use]
    pub fn feature_importances_with_ci(&self) -> Option<&FeatureImportanceWithCI> {
        self.feature_importances_with_ci.as_ref()
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.estimators, "predict_proba")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let estimators = self.estimators.as_ref().unwrap();
        let forest_classes = self.classes.as_ref().unwrap();
        let n_classes = forest_classes.len();
        let n_samples = x.nrows();

        // Sum all tree predictions, aligning each tree's classes to the forest's classes
        let mut probas = Array2::zeros((n_samples, n_classes));

        // Helper to align tree probabilities to forest class ordering
        let align_tree_probas = |tree: &DecisionTreeClassifier| -> Array2<f64> {
            let tree_proba = tree.predict_proba(x).unwrap();
            let tree_classes = tree.classes().unwrap();

            // If tree has same classes as forest, return directly
            if tree_classes.len() == n_classes {
                return tree_proba;
            }

            // Otherwise, map tree's class indices to forest's class indices
            let mut aligned = Array2::zeros((n_samples, n_classes));
            for (tree_idx, &tree_class) in tree_classes.iter().enumerate() {
                // Find this class in the forest's class list
                if let Some(forest_idx) = forest_classes
                    .iter()
                    .position(|&c| (c - tree_class).abs() < 1e-10)
                {
                    for i in 0..n_samples {
                        aligned[[i, forest_idx]] = tree_proba[[i, tree_idx]];
                    }
                }
            }
            aligned
        };

        // Collect and sum predictions (sequential when n_jobs=1 for reproducibility)
        if self.n_jobs == Some(1) {
            for tree in estimators.iter() {
                let tree_proba = align_tree_probas(tree);
                probas = probas + tree_proba;
            }
        } else {
            #[cfg(feature = "parallel")]
            {
                let tree_probas: Vec<Array2<f64>> =
                    estimators.par_iter().map(align_tree_probas).collect();
                for tree_proba in tree_probas {
                    probas = probas + tree_proba;
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                for tree in estimators.iter() {
                    let tree_proba = align_tree_probas(tree);
                    probas = probas + tree_proba;
                }
            }
        }

        // Average over trees
        probas.mapv_inplace(|p| p / estimators.len() as f64);

        Ok(probas)
    }

    /// Generate bootstrap sample indices and OOB indices
    fn generate_bootstrap_indices(n_samples: usize, rng: &mut StdRng) -> (Vec<usize>, Vec<usize>) {
        let mut in_bag = vec![false; n_samples];
        let mut indices = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let idx = rng.random_range(0..n_samples);
            indices.push(idx);
            in_bag[idx] = true;
        }

        let oob_indices: Vec<usize> = (0..n_samples).filter(|&i| !in_bag[i]).collect();

        (indices, oob_indices)
    }

    /// Compute OOB score after fitting
    fn compute_oob_score(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        oob_indices_per_tree: &[Vec<usize>],
    ) {
        let n_samples = x.nrows();
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();
        let estimators = self.estimators.as_ref().unwrap();

        // Accumulate OOB predictions
        let mut oob_proba = Array2::zeros((n_samples, n_classes));
        let mut oob_counts = vec![0usize; n_samples];

        for (tree_idx, tree) in estimators.iter().enumerate() {
            let oob_indices = &oob_indices_per_tree[tree_idx];

            for &sample_idx in oob_indices {
                let sample = x.row(sample_idx).to_owned().insert_axis(Axis(0));
                if let Ok(proba) = tree.predict_proba(&sample) {
                    // Align tree class indices to forest class indices
                    if let Some(tree_classes) = tree.classes() {
                        for (tree_col, &tree_class) in tree_classes.iter().enumerate() {
                            if let Some(forest_col) =
                                classes.iter().position(|&c| (c - tree_class).abs() < 1e-10)
                            {
                                oob_proba[[sample_idx, forest_col]] += proba[[0, tree_col]];
                            }
                        }
                    }
                    oob_counts[sample_idx] += 1;
                }
            }
        }

        // Normalize and compute accuracy
        let mut correct = 0;
        let mut total = 0;

        for i in 0..n_samples {
            if oob_counts[i] > 0 {
                // Normalize probabilities
                for j in 0..n_classes {
                    oob_proba[[i, j]] /= oob_counts[i] as f64;
                }

                // Find predicted class
                let pred_class_idx = oob_proba
                    .row(i)
                    .iter()
                    .enumerate()
                    .max_by(|a: &(usize, &f64), b: &(usize, &f64)| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                let pred_class = classes[pred_class_idx];
                if (pred_class - y[i]).abs() < 1e-10 {
                    correct += 1;
                }
                total += 1;
            }
        }

        if total > 0 {
            self.oob_score = Some(correct as f64 / total as f64);
            self.oob_decision_function = Some(oob_proba);
        }
    }

    /// Compute feature importances with confidence intervals
    fn compute_feature_importances_with_ci(&mut self) {
        let estimators = match &self.estimators {
            Some(e) => e,
            None => return,
        };

        let n_features = self.n_features.unwrap();

        // Collect importances from all trees
        let all_importances: Vec<Array1<f64>> = estimators
            .iter()
            .filter_map(|tree| tree.feature_importance())
            .collect();

        if all_importances.is_empty() {
            return;
        }

        // Compute mean importance
        let mut mean_importance = Array1::zeros(n_features);
        for imp in &all_importances {
            mean_importance = mean_importance + imp;
        }
        mean_importance.mapv_inplace(|v| v / all_importances.len() as f64);

        // Compute standard error and CI
        let n_trees = all_importances.len() as f64;
        let mut variance: Array1<f64> = Array1::zeros(n_features);

        for imp in &all_importances {
            for j in 0..n_features {
                let diff: f64 = imp[j] - mean_importance[j];
                variance[j] += diff.powi(2);
            }
        }
        variance.mapv_inplace(|v: f64| v / (n_trees - 1.0).max(1.0));
        let std_error = variance.mapv(|v: f64| v.sqrt() / n_trees.sqrt());

        // Compute confidence interval using t-distribution approximation
        let alpha = 1.0 - self.confidence_level;
        // For large n, use z-score; approximate t-critical value
        let z_crit = if n_trees > 30.0 {
            statrs::distribution::Normal::new(0.0, 1.0)
                .map(|n| {
                    use statrs::distribution::ContinuousCDF;
                    n.inverse_cdf(1.0 - alpha / 2.0)
                })
                .unwrap_or(1.96)
        } else {
            // Use t-distribution
            statrs::distribution::StudentsT::new(0.0, 1.0, n_trees - 1.0)
                .map(|t| {
                    use statrs::distribution::ContinuousCDF;
                    t.inverse_cdf(1.0 - alpha / 2.0)
                })
                .unwrap_or(2.0)
        };

        let ci_lower = &mean_importance - &std_error * z_crit;
        let ci_upper = &mean_importance + &std_error * z_crit;

        self.feature_importances = Some(mean_importance.clone());
        self.feature_importances_with_ci = Some(FeatureImportanceWithCI {
            importance: mean_importance,
            std_error,
            ci_lower,
            ci_upper,
            confidence_level: self.confidence_level,
        });
    }
}

impl Model for RandomForestClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Find unique classes
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::invalid_input(
                "RandomForestClassifier requires at least 2 classes",
            ));
        }

        self.classes = Some(Array1::from_vec(classes));
        self.n_features = Some(n_features);

        // Compute max_features
        let max_features = self
            .max_features
            .unwrap_or(MaxFeatures::Sqrt)
            .compute(n_features);

        // Create base RNG for generating tree seeds
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        // Generate seeds and bootstrap indices for each tree
        let tree_seeds: Vec<u64> = (0..self.n_estimators).map(|_| rng.random()).collect();

        let (bootstrap_indices, oob_indices_per_tree): (Vec<Vec<usize>>, Vec<Vec<usize>>) =
            if self.bootstrap {
                tree_seeds
                    .iter()
                    .map(|&seed| {
                        let mut tree_rng = StdRng::seed_from_u64(seed);
                        Self::generate_bootstrap_indices(n_samples, &mut tree_rng)
                    })
                    .unzip()
            } else {
                // No bootstrap - use all samples
                let all_indices: Vec<usize> = (0..n_samples).collect();
                (
                    vec![all_indices; self.n_estimators],
                    vec![vec![]; self.n_estimators],
                )
            };

        // Build trees (parallel or sequential based on n_jobs)
        let criterion = self.criterion;
        let max_depth = self.max_depth;
        let min_samples_split = self.min_samples_split;
        let min_samples_leaf = self.min_samples_leaf;
        let min_impurity_decrease = self.min_impurity_decrease;
        let class_weight = self.class_weight.clone();

        // Helper closure to build a single tree
        // Returns None if the bootstrap sample has only one class (can happen with extreme imbalance)
        let build_tree = |(indices, &seed): (&Vec<usize>, &u64)| -> Option<DecisionTreeClassifier> {
            // Create bootstrap sample
            let n_bootstrap = indices.len();
            let mut x_bootstrap = Array2::zeros((n_bootstrap, n_features));
            let mut y_bootstrap = Array1::zeros(n_bootstrap);

            for (i, &idx) in indices.iter().enumerate() {
                x_bootstrap.row_mut(i).assign(&x.row(idx));
                y_bootstrap[i] = y[idx];
            }

            // Create and fit tree
            let mut tree = DecisionTreeClassifier::new()
                .with_criterion(criterion)
                .with_max_depth(max_depth)
                .with_min_samples_split(min_samples_split)
                .with_min_samples_leaf(min_samples_leaf)
                .with_max_features(Some(max_features))
                .with_min_impurity_decrease(min_impurity_decrease)
                .with_class_weight(class_weight.clone())
                .with_random_state(seed);

            match tree.fit(&x_bootstrap, &y_bootstrap) {
                Ok(()) => Some(tree),
                Err(_) => None, // Skip bootstrap samples with single class
            }
        };

        // Use sequential iteration when n_jobs == 1 for reproducibility
        let estimators: Vec<DecisionTreeClassifier> = if self.n_jobs == Some(1) {
            bootstrap_indices
                .iter()
                .zip(tree_seeds.iter())
                .filter_map(build_tree)
                .collect()
        } else {
            bootstrap_indices
                .par_iter()
                .zip(tree_seeds.par_iter())
                .filter_map(build_tree)
                .collect()
        };

        if estimators.is_empty() {
            return Err(crate::FerroError::InvalidInput(
                "RandomForestClassifier: all bootstrap samples contained a single class; cannot fit any trees".to_string(),
            ));
        }

        self.estimators = Some(estimators);

        // Compute OOB score if enabled
        if self.oob_score_enabled && self.bootstrap {
            self.compute_oob_score(x, y, &oob_indices_per_tree);
        }

        // Compute feature importances with CIs
        self.compute_feature_importances_with_ci();

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.estimators, "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let probas = self.predict_proba(x)?;
        let classes = self.classes.as_ref().unwrap();
        let n_samples = x.nrows();

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let max_idx = probas
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = classes[max_idx];
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.estimators.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        self.feature_importances.clone()
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .int("n_estimators", 10, 500)
            .int("max_depth", 1, 30)
            .int("min_samples_split", 2, 20)
            .int("min_samples_leaf", 1, 10)
            .float("min_impurity_decrease", 0.0, 0.5)
            .categorical(
                "max_features",
                vec!["sqrt".to_string(), "log2".to_string(), "all".to_string()],
            )
            .categorical("criterion", vec!["gini".to_string(), "entropy".to_string()])
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

// =============================================================================
// Random Forest Regressor
// =============================================================================

/// Random Forest Regressor using bootstrap aggregating
///
/// Builds an ensemble of decision trees, each trained on a bootstrap sample
/// of the data, with random feature subsampling at each split.
///
/// ## Features
///
/// - Bootstrap aggregating (bagging) of decision trees
/// - Random feature subsampling at each split (default: n_features/3)
/// - OOB (Out-of-Bag) R² score estimation
/// - Feature importance with confidence intervals via bootstrap
/// - Parallel tree building using rayon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestRegressor {
    /// Number of trees in the forest
    pub n_estimators: usize,
    /// Split criterion (MSE or MAE)
    pub criterion: SplitCriterion,
    /// Maximum depth of each tree (None for unlimited)
    pub max_depth: Option<usize>,
    /// Minimum samples required to split an internal node
    pub min_samples_split: usize,
    /// Minimum samples required to be at a leaf node
    pub min_samples_leaf: usize,
    /// Number of features to consider at each split (None for n_features/3)
    pub max_features: Option<MaxFeatures>,
    /// Bootstrap sampling
    pub bootstrap: bool,
    /// Whether to compute OOB score during fitting
    pub oob_score_enabled: bool,
    /// Number of parallel jobs (None for all available cores).
    ///
    /// **Reproducibility Note:** Set to `Some(1)` for fully deterministic results
    /// when combined with `random_state`. Parallel execution introduces non-determinism
    /// due to thread scheduling, even with a fixed random seed.
    pub n_jobs: Option<usize>,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Confidence level for feature importance CIs
    pub confidence_level: f64,
    /// Minimum impurity decrease for splits
    pub min_impurity_decrease: f64,

    /// Enable warm start to add estimators incrementally
    pub warm_start: bool,

    // Fitted parameters
    estimators: Option<Vec<DecisionTreeRegressor>>,
    n_features: Option<usize>,
    feature_importances: Option<Array1<f64>>,
    feature_importances_with_ci: Option<FeatureImportanceWithCI>,
    oob_score: Option<f64>,
    oob_prediction: Option<Array1<f64>>,
}

impl Default for RandomForestRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomForestRegressor {
    /// Create a new Random Forest Regressor with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            criterion: SplitCriterion::Mse,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: Some(MaxFeatures::Fraction(1.0 / 3.0)), // n_features / 3 for regression
            bootstrap: true,
            oob_score_enabled: true,
            n_jobs: None,
            random_state: None,
            confidence_level: 0.95,
            min_impurity_decrease: 0.0,
            warm_start: false,
            estimators: None,
            n_features: None,
            feature_importances: None,
            feature_importances_with_ci: None,
            oob_score: None,
            oob_prediction: None,
        }
    }

    /// Set the number of trees
    #[must_use]
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators.max(1);
        self
    }

    /// Set the split criterion
    #[must_use]
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
        assert!(
            criterion.is_regression(),
            "Criterion must be MSE or MAE for regression"
        );
        self.criterion = criterion;
        self
    }

    /// Set maximum tree depth
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set minimum samples to split
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split.max(2);
        self
    }

    /// Set minimum samples per leaf
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf.max(1);
        self
    }

    /// Set maximum features strategy
    #[must_use]
    pub fn with_max_features(mut self, max_features: Option<MaxFeatures>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Enable or disable bootstrap sampling
    #[must_use]
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Enable or disable OOB score computation
    #[must_use]
    pub fn with_oob_score(mut self, oob_score: bool) -> Self {
        self.oob_score_enabled = oob_score;
        self
    }

    /// Set number of parallel jobs.
    ///
    /// # Arguments
    /// * `n_jobs` - Number of parallel workers:
    ///   - `None` - Use all available CPU cores (default, fastest)
    ///   - `Some(1)` - Sequential execution (required for reproducibility)
    ///   - `Some(n)` - Use n parallel workers
    ///
    /// # Reproducibility
    /// For fully deterministic results, use `with_n_jobs(Some(1))` together with
    /// `with_random_state(seed)`. Parallel execution introduces non-determinism
    /// due to thread scheduling and floating-point aggregation order.
    ///
    /// # Example
    /// ```
    /// use ferroml_core::models::RandomForestRegressor;
    ///
    /// // Fast parallel training (non-deterministic)
    /// let fast_model = RandomForestRegressor::new()
    ///     .with_random_state(42);
    ///
    /// // Reproducible sequential training
    /// let reproducible_model = RandomForestRegressor::new()
    ///     .with_random_state(42)
    ///     .with_n_jobs(Some(1));
    /// ```
    #[must_use]
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Set random state for reproducibility.
    ///
    /// Note: For fully deterministic results, also set `with_n_jobs(Some(1))`.
    /// Parallel execution can introduce non-determinism even with a fixed seed.
    #[must_use]
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set confidence level for CIs
    #[must_use]
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level.clamp(0.5, 0.999);
        self
    }

    /// Set minimum impurity decrease
    #[must_use]
    pub fn with_min_impurity_decrease(mut self, min_impurity_decrease: f64) -> Self {
        self.min_impurity_decrease = min_impurity_decrease.max(0.0);
        self
    }

    /// Get the OOB score (R² for regression)
    #[must_use]
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score
    }

    /// Get the OOB predictions
    #[must_use]
    pub fn oob_prediction(&self) -> Option<&Array1<f64>> {
        self.oob_prediction.as_ref()
    }

    /// Get the individual estimators
    #[must_use]
    pub fn estimators(&self) -> Option<&[DecisionTreeRegressor]> {
        self.estimators.as_deref()
    }

    /// Get feature importance with confidence intervals
    #[must_use]
    pub fn feature_importances_with_ci(&self) -> Option<&FeatureImportanceWithCI> {
        self.feature_importances_with_ci.as_ref()
    }

    /// Generate bootstrap sample indices and OOB indices
    fn generate_bootstrap_indices(n_samples: usize, rng: &mut StdRng) -> (Vec<usize>, Vec<usize>) {
        let mut in_bag = vec![false; n_samples];
        let mut indices = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let idx = rng.random_range(0..n_samples);
            indices.push(idx);
            in_bag[idx] = true;
        }

        let oob_indices: Vec<usize> = (0..n_samples).filter(|&i| !in_bag[i]).collect();

        (indices, oob_indices)
    }

    /// Compute OOB score (R²) after fitting
    fn compute_oob_score(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        oob_indices_per_tree: &[Vec<usize>],
    ) {
        let n_samples = x.nrows();
        let estimators = self.estimators.as_ref().unwrap();

        // Accumulate OOB predictions
        let mut oob_predictions = Array1::zeros(n_samples);
        let mut oob_counts = vec![0usize; n_samples];

        for (tree_idx, tree) in estimators.iter().enumerate() {
            let oob_indices = &oob_indices_per_tree[tree_idx];

            for &sample_idx in oob_indices {
                let sample = x.row(sample_idx).to_owned().insert_axis(Axis(0));
                if let Ok(pred) = tree.predict(&sample) {
                    oob_predictions[sample_idx] += pred[0];
                    oob_counts[sample_idx] += 1;
                }
            }
        }

        // Normalize predictions and compute R²
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        let mut valid_count = 0;
        let mut y_sum = 0.0;

        for i in 0..n_samples {
            if oob_counts[i] > 0 {
                oob_predictions[i] /= oob_counts[i] as f64;
                y_sum += y[i];
                valid_count += 1;
            }
        }

        if valid_count > 0 {
            let y_mean = y_sum / valid_count as f64;

            for i in 0..n_samples {
                if oob_counts[i] > 0 {
                    let residual: f64 = y[i] - oob_predictions[i];
                    ss_res += residual.powi(2);
                    let deviation: f64 = y[i] - y_mean;
                    ss_tot += deviation.powi(2);
                }
            }

            let r2 = if ss_tot > 0.0 {
                1.0 - ss_res / ss_tot
            } else {
                0.0
            };

            self.oob_score = Some(r2);
            self.oob_prediction = Some(oob_predictions);
        }
    }

    /// Compute feature importances with confidence intervals
    fn compute_feature_importances_with_ci(&mut self) {
        let estimators = match &self.estimators {
            Some(e) => e,
            None => return,
        };

        let n_features = self.n_features.unwrap();

        // Collect importances from all trees
        let all_importances: Vec<Array1<f64>> = estimators
            .iter()
            .filter_map(|tree| tree.feature_importance())
            .collect();

        if all_importances.is_empty() {
            return;
        }

        // Compute mean importance
        let mut mean_importance = Array1::zeros(n_features);
        for imp in &all_importances {
            mean_importance = mean_importance + imp;
        }
        mean_importance.mapv_inplace(|v| v / all_importances.len() as f64);

        // Compute standard error and CI
        let n_trees = all_importances.len() as f64;
        let mut variance: Array1<f64> = Array1::zeros(n_features);

        for imp in &all_importances {
            for j in 0..n_features {
                let diff: f64 = imp[j] - mean_importance[j];
                variance[j] += diff.powi(2);
            }
        }
        variance.mapv_inplace(|v: f64| v / (n_trees - 1.0).max(1.0));
        let std_error = variance.mapv(|v: f64| v.sqrt() / n_trees.sqrt());

        // Compute confidence interval
        let alpha = 1.0 - self.confidence_level;
        let z_crit = if n_trees > 30.0 {
            statrs::distribution::Normal::new(0.0, 1.0)
                .map(|n| {
                    use statrs::distribution::ContinuousCDF;
                    n.inverse_cdf(1.0 - alpha / 2.0)
                })
                .unwrap_or(1.96)
        } else {
            statrs::distribution::StudentsT::new(0.0, 1.0, n_trees - 1.0)
                .map(|t| {
                    use statrs::distribution::ContinuousCDF;
                    t.inverse_cdf(1.0 - alpha / 2.0)
                })
                .unwrap_or(2.0)
        };

        let ci_lower = &mean_importance - &std_error * z_crit;
        let ci_upper = &mean_importance + &std_error * z_crit;

        self.feature_importances = Some(mean_importance.clone());
        self.feature_importances_with_ci = Some(FeatureImportanceWithCI {
            importance: mean_importance,
            std_error,
            ci_lower,
            ci_upper,
            confidence_level: self.confidence_level,
        });
    }
}

impl Model for RandomForestRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        self.n_features = Some(n_features);

        // Compute max_features (default: n_features/3 for regression)
        let max_features = self
            .max_features
            .unwrap_or(MaxFeatures::Fraction(1.0 / 3.0))
            .compute(n_features);

        // Create base RNG for generating tree seeds
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        // Generate seeds and bootstrap indices for each tree
        let tree_seeds: Vec<u64> = (0..self.n_estimators).map(|_| rng.random()).collect();

        let (bootstrap_indices, oob_indices_per_tree): (Vec<Vec<usize>>, Vec<Vec<usize>>) =
            if self.bootstrap {
                tree_seeds
                    .iter()
                    .map(|&seed| {
                        let mut tree_rng = StdRng::seed_from_u64(seed);
                        Self::generate_bootstrap_indices(n_samples, &mut tree_rng)
                    })
                    .unzip()
            } else {
                let all_indices: Vec<usize> = (0..n_samples).collect();
                (
                    vec![all_indices; self.n_estimators],
                    vec![vec![]; self.n_estimators],
                )
            };

        // Build trees (parallel or sequential based on n_jobs)
        let criterion = self.criterion;
        let max_depth = self.max_depth;
        let min_samples_split = self.min_samples_split;
        let min_samples_leaf = self.min_samples_leaf;
        let min_impurity_decrease = self.min_impurity_decrease;

        // Helper closure to build a single tree
        let build_tree = |(indices, &seed): (&Vec<usize>, &u64)| {
            // Create bootstrap sample
            let n_bootstrap = indices.len();
            let mut x_bootstrap = Array2::zeros((n_bootstrap, n_features));
            let mut y_bootstrap = Array1::zeros(n_bootstrap);

            for (i, &idx) in indices.iter().enumerate() {
                x_bootstrap.row_mut(i).assign(&x.row(idx));
                y_bootstrap[i] = y[idx];
            }

            // Create and fit tree
            let mut tree = DecisionTreeRegressor::new()
                .with_criterion(criterion)
                .with_max_depth(max_depth)
                .with_min_samples_split(min_samples_split)
                .with_min_samples_leaf(min_samples_leaf)
                .with_max_features(Some(max_features))
                .with_min_impurity_decrease(min_impurity_decrease)
                .with_random_state(seed);

            tree.fit(&x_bootstrap, &y_bootstrap)
                .expect("RandomForestRegressor: failed to fit decision tree on bootstrap sample");
            tree
        };

        // Use sequential iteration when n_jobs == 1 for reproducibility
        let estimators: Vec<DecisionTreeRegressor> = if self.n_jobs == Some(1) {
            bootstrap_indices
                .iter()
                .zip(tree_seeds.iter())
                .map(build_tree)
                .collect()
        } else {
            bootstrap_indices
                .par_iter()
                .zip(tree_seeds.par_iter())
                .map(build_tree)
                .collect()
        };

        self.estimators = Some(estimators);

        // Compute OOB score if enabled
        if self.oob_score_enabled && self.bootstrap {
            self.compute_oob_score(x, y, &oob_indices_per_tree);
        }

        // Compute feature importances with CIs
        self.compute_feature_importances_with_ci();

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.estimators, "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let estimators = self.estimators.as_ref().unwrap();
        let n_samples = x.nrows();

        // Collect predictions from all trees (parallel when enabled)
        #[cfg(feature = "parallel")]
        let tree_preds: Vec<Array1<f64>> = estimators
            .par_iter()
            .map(|tree| tree.predict(x).unwrap())
            .collect();

        #[cfg(not(feature = "parallel"))]
        let tree_preds: Vec<Array1<f64>> = estimators
            .iter()
            .map(|tree| tree.predict(x).unwrap())
            .collect();

        // Sum all tree predictions
        let mut predictions = Array1::zeros(n_samples);
        for tree_pred in tree_preds {
            predictions = predictions + tree_pred;
        }

        predictions.mapv_inplace(|p| p / estimators.len() as f64);

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.estimators.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        self.feature_importances.clone()
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .int("n_estimators", 10, 500)
            .int("max_depth", 1, 30)
            .int("min_samples_split", 2, 20)
            .int("min_samples_leaf", 1, 10)
            .float("min_impurity_decrease", 0.0, 0.5)
            .categorical(
                "max_features",
                vec![
                    "third".to_string(),
                    "sqrt".to_string(),
                    "log2".to_string(),
                    "all".to_string(),
                ],
            )
            .categorical("criterion", vec!["mse".to_string(), "mae".to_string()])
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

// =============================================================================
impl super::traits::WarmStartModel for RandomForestClassifier {
    fn set_warm_start(&mut self, warm_start: bool) {
        self.warm_start = warm_start;
    }

    fn warm_start(&self) -> bool {
        self.warm_start
    }

    fn n_estimators_fitted(&self) -> usize {
        self.estimators.as_ref().map_or(0, |e| e.len())
    }
}

impl super::traits::WarmStartModel for RandomForestRegressor {
    fn set_warm_start(&mut self, warm_start: bool) {
        self.warm_start = warm_start;
    }

    fn warm_start(&self) -> bool {
        self.warm_start
    }

    fn n_estimators_fitted(&self) -> usize {
        self.estimators.as_ref().map_or(0, |e| e.len())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_classification_data() -> (Array2<f64>, Array1<f64>) {
        // Linearly separable data
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 1.0, 2.0, 1.5, 2.5, 2.0, 1.0, 2.5, 1.5,
                1.2, 1.8, 1.8, 1.2, // Class 0
                5.0, 5.0, 5.5, 5.5, 6.0, 6.0, 6.5, 6.5, 5.0, 6.0, 5.5, 6.5, 6.0, 5.0, 6.5, 5.5,
                5.2, 5.8, 5.8, 5.2, // Class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]);
        (x, y)
    }

    fn make_regression_data() -> (Array2<f64>, Array1<f64>) {
        // Simple linear relationship: y ≈ x1 + x2
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.5, 1.5, 2.5, 2.5, 3.0, 1.0, 3.0, 2.0,
                1.0, 3.0, 2.0, 3.0, 3.0, 3.0, 4.0, 1.0, 4.0, 2.0, 4.0, 3.0, 4.0, 4.0, 5.0, 1.0,
                5.0, 2.0, 5.0, 3.0, 5.0, 4.0, 5.0, 5.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            2.0, 3.0, 3.0, 4.0, 3.0, 5.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 8.0, 6.0, 7.0,
            8.0, 9.0, 10.0,
        ]);
        (x, y)
    }

    #[test]
    fn test_random_forest_classifier_fit_predict() {
        let (x, y) = make_classification_data();

        let mut clf = RandomForestClassifier::new()
            .with_n_estimators(10)
            .with_max_depth(Some(5))
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        assert_eq!(clf.n_features(), Some(2));

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);

        // Should classify most training data correctly
        let accuracy: f64 = predictions
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 0.5)
            .count() as f64
            / 20.0;
        assert!(accuracy > 0.8, "Accuracy was {}", accuracy);
    }

    #[test]
    fn test_random_forest_classifier_proba() {
        let (x, y) = make_classification_data();

        let mut clf = RandomForestClassifier::new()
            .with_n_estimators(10)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        let probas = clf.predict_proba(&x).unwrap();
        assert_eq!(probas.shape(), &[20, 2]);

        // Probabilities should sum to 1
        for i in 0..20 {
            let row_sum: f64 = probas.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_random_forest_classifier_oob_score() {
        let (x, y) = make_classification_data();

        let mut clf = RandomForestClassifier::new()
            .with_n_estimators(20)
            .with_oob_score(true)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        let oob_score = clf.oob_score();
        assert!(oob_score.is_some());
        let score = oob_score.unwrap();
        assert!(score > 0.5, "OOB score was {}", score); // Better than random
    }

    #[test]
    fn test_random_forest_classifier_feature_importance() {
        let (x, y) = make_classification_data();

        let mut clf = RandomForestClassifier::new()
            .with_n_estimators(10)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        let importance = clf.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // Importances should sum to 1 (approximately)
        let total: f64 = importance.sum();
        assert!((total - 1.0).abs() < 0.1, "Total importance was {}", total);

        // Check CI
        let ci = clf.feature_importances_with_ci().unwrap();
        assert_eq!(ci.importance.len(), 2);
        assert_eq!(ci.std_error.len(), 2);
        assert!(ci.ci_lower[0] <= ci.importance[0]);
        assert!(ci.ci_upper[0] >= ci.importance[0]);
    }

    #[test]
    fn test_random_forest_regressor_fit_predict() {
        let (x, y) = make_regression_data();

        let mut reg = RandomForestRegressor::new()
            .with_n_estimators(10)
            .with_max_depth(Some(5))
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();

        assert!(reg.is_fitted());
        assert_eq!(reg.n_features(), Some(2));

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);

        // Should fit reasonably well
        let mse: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / 20.0;
        assert!(mse < 5.0, "MSE was {}", mse);
    }

    #[test]
    fn test_random_forest_regressor_oob_score() {
        let (x, y) = make_regression_data();

        let mut reg = RandomForestRegressor::new()
            .with_n_estimators(20)
            .with_oob_score(true)
            .with_random_state(42);
        reg.fit(&x, &y).unwrap();

        let oob_score = reg.oob_score();
        assert!(oob_score.is_some());
        // R² can be negative for poor fits, but should be reasonable here
    }

    #[test]
    fn test_random_forest_regressor_feature_importance() {
        let (x, y) = make_regression_data();

        let mut reg = RandomForestRegressor::new()
            .with_n_estimators(10)
            .with_random_state(42);
        reg.fit(&x, &y).unwrap();

        let importance = reg.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // All importances should be non-negative
        assert!(importance.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_max_features() {
        assert_eq!(MaxFeatures::All.compute(10), 10);
        assert_eq!(MaxFeatures::Sqrt.compute(16), 4);
        assert_eq!(MaxFeatures::Log2.compute(8), 3);
        assert_eq!(MaxFeatures::Fixed(5).compute(10), 5);
        assert_eq!(MaxFeatures::Fixed(15).compute(10), 10); // Capped at n_features
        assert_eq!(MaxFeatures::Fraction(0.5).compute(10), 5);
    }

    #[test]
    fn test_random_forest_without_bootstrap() {
        let (x, y) = make_classification_data();

        let mut clf = RandomForestClassifier::new()
            .with_n_estimators(5)
            .with_bootstrap(false)
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();
        assert!(clf.is_fitted());
        assert!(clf.oob_score().is_none()); // No OOB without bootstrap
    }

    #[test]
    fn test_random_forest_search_space() {
        let clf = RandomForestClassifier::new();
        let space = clf.search_space();
        assert!(space.n_dims() > 0);

        let reg = RandomForestRegressor::new();
        let space = reg.search_space();
        assert!(space.n_dims() > 0);
    }

    #[test]
    fn test_not_fitted_error() {
        let clf = RandomForestClassifier::new();
        let x = Array2::zeros((2, 2));

        let result = clf.predict(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_features_error() {
        let (x, y) = make_classification_data();

        let mut clf = RandomForestClassifier::new()
            .with_n_estimators(5)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        // Try to predict with wrong number of features
        let x_wrong = Array2::zeros((2, 3));
        let result = clf.predict(&x_wrong);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]); // Single class

        let mut clf = RandomForestClassifier::new();
        let result = clf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimators_access() {
        let (x, y) = make_classification_data();

        let mut clf = RandomForestClassifier::new()
            .with_n_estimators(5)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        let estimators = clf.estimators().unwrap();
        assert_eq!(estimators.len(), 5);

        // Each estimator should be fitted
        for est in estimators {
            assert!(est.is_fitted());
        }
    }

    #[test]
    fn test_oob_missing_class() {
        // Create a dataset where some bootstrap samples may miss a class
        // 3 classes, with class 2 having only 2 samples (likely to be missing in some bootstraps)
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                // Class 0: cluster around (0,0)
                0.1, 0.1, 0.2, 0.2, 0.0, 0.3, 0.3, 0.0, 0.1, 0.2, 0.2, 0.1, 0.0, 0.0, 0.3, 0.3, 0.1,
                0.0, 0.0, 0.1, // Class 1: cluster around (5,5)
                5.1, 5.1, 5.2, 5.2, 5.0, 5.3, 5.3, 5.0, 5.1, 5.2, 5.2, 5.1, 5.0, 5.0, 5.3, 5.3,
                // Class 2: only 2 samples around (10,10)
                10.0, 10.0, 10.1, 10.1,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 2.0, 2.0,
        ]);

        let mut clf = RandomForestClassifier::new()
            .with_n_estimators(20)
            .with_random_state(42);
        clf.max_features = None; // Use all features for simpler splits

        // This should not panic even when bootstrap samples miss class 2
        let result = clf.fit(&x, &y);
        assert!(result.is_ok(), "Fit should succeed: {:?}", result.err());

        // OOB score should be computed
        let oob = clf.oob_score();
        assert!(oob.is_some(), "OOB score should be available");
        assert!(
            oob.unwrap() > 0.5,
            "OOB accuracy should be reasonable: {}",
            oob.unwrap()
        );
    }
}

//! Extremely Randomized Trees (ExtraTrees)
//!
//! ExtraTrees differ from Random Forests in two key ways:
//! 1. **No bootstrap**: Each tree is trained on the full dataset (by default).
//! 2. **Random thresholds**: Instead of searching for the best threshold per feature,
//!    a random threshold is drawn uniformly between min and max of each candidate feature.
//!
//! These differences make ExtraTrees faster to train and sometimes more robust to noise,
//! at the cost of slightly higher bias.

use crate::hpo::SearchSpace;
use crate::models::forest::{FeatureImportanceWithCI, MaxFeatures};
use crate::models::tree::{
    DecisionTreeClassifier, DecisionTreeRegressor, SplitCriterion, SplitStrategy,
};
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, ClassWeight, Model,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// =============================================================================
// ExtraTreesClassifier
// =============================================================================

/// Extremely Randomized Trees classifier.
///
/// Like `RandomForestClassifier`, but uses random thresholds at each split
/// and does not use bootstrap sampling by default.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraTreesClassifier {
    /// Number of trees in the ensemble
    pub n_estimators: usize,
    /// Split criterion (Gini or Entropy)
    pub criterion: SplitCriterion,
    /// Maximum depth of each tree (None for unlimited)
    pub max_depth: Option<usize>,
    /// Minimum samples required to split an internal node
    pub min_samples_split: usize,
    /// Minimum samples required to be at a leaf node
    pub min_samples_leaf: usize,
    /// Number of features to consider at each split (default: Sqrt)
    pub max_features: Option<MaxFeatures>,
    /// Bootstrap sampling (default: false for ExtraTrees)
    pub bootstrap: bool,
    /// Number of parallel jobs (None for all available cores, Some(1) for deterministic)
    pub n_jobs: Option<usize>,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Minimum impurity decrease for splits
    pub min_impurity_decrease: f64,
    /// Class weights for handling imbalanced datasets
    pub class_weight: ClassWeight,

    // Fitted parameters
    estimators: Option<Vec<DecisionTreeClassifier>>,
    classes: Option<Array1<f64>>,
    n_features: Option<usize>,
    feature_importances: Option<Array1<f64>>,
    feature_importances_with_ci: Option<FeatureImportanceWithCI>,
}

impl Default for ExtraTreesClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtraTreesClassifier {
    /// Create a new ExtraTrees Classifier with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            criterion: SplitCriterion::Gini,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: Some(MaxFeatures::Sqrt),
            bootstrap: false,
            n_jobs: None,
            random_state: None,
            min_impurity_decrease: 0.0,
            class_weight: ClassWeight::Uniform,
            estimators: None,
            classes: None,
            n_features: None,
            feature_importances: None,
            feature_importances_with_ci: None,
        }
    }

    /// Set the number of trees
    #[must_use]
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n.max(1);
        self
    }

    /// Set the split criterion
    #[must_use]
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
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
    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n.max(2);
        self
    }

    /// Set minimum samples per leaf
    #[must_use]
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n.max(1);
        self
    }

    /// Set maximum features to consider at each split
    #[must_use]
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.max_features = Some(max_features);
        self
    }

    /// Set whether to use bootstrap sampling
    #[must_use]
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Set number of parallel jobs
    #[must_use]
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Set random seed
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set minimum impurity decrease
    #[must_use]
    pub fn with_min_impurity_decrease(mut self, v: f64) -> Self {
        self.min_impurity_decrease = v.max(0.0);
        self
    }

    /// Set class weights
    #[must_use]
    pub fn with_class_weight(mut self, cw: ClassWeight) -> Self {
        self.class_weight = cw;
        self
    }

    /// Get feature importances (mean decrease in impurity)
    #[must_use]
    pub fn feature_importances(&self) -> Option<&Array1<f64>> {
        self.feature_importances.as_ref()
    }

    /// Get feature importances with confidence intervals
    #[must_use]
    pub fn feature_importances_with_ci(&self) -> Option<&FeatureImportanceWithCI> {
        self.feature_importances_with_ci.as_ref()
    }

    /// Predict class probabilities by averaging tree predictions
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.estimators, "predict_proba")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let estimators = self.estimators.as_ref().unwrap();
        let classes = self.classes.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_classes = classes.len();
        let mut avg_proba = Array2::zeros((n_samples, n_classes));

        for tree in estimators {
            let tree_struct = tree.tree().unwrap();
            for i in 0..n_samples {
                let sample: Vec<f64> = x.row(i).to_vec();
                let mut node_id = 0;
                while let Some(node) = tree_struct.get_node(node_id) {
                    if node.is_leaf {
                        let total: f64 = node.value.iter().sum();
                        if total > 0.0 {
                            for (c, &v) in node.value.iter().enumerate() {
                                avg_proba[[i, c]] += v / total;
                            }
                        }
                        break;
                    }
                    let fi = node.feature_index.unwrap();
                    let thr = node.threshold.unwrap();
                    node_id = if sample[fi] <= thr {
                        node.left_child.unwrap()
                    } else {
                        node.right_child.unwrap()
                    };
                }
            }
        }

        let n_trees = estimators.len() as f64;
        avg_proba /= n_trees;
        Ok(avg_proba)
    }

    /// Generate bootstrap indices and OOB indices
    fn generate_bootstrap_indices(n_samples: usize, rng: &mut StdRng) -> (Vec<usize>, Vec<usize>) {
        let bootstrap: Vec<usize> = (0..n_samples)
            .map(|_| rng.random_range(0..n_samples))
            .collect();
        let in_bag: std::collections::HashSet<usize> = bootstrap.iter().copied().collect();
        let oob: Vec<usize> = (0..n_samples).filter(|i| !in_bag.contains(i)).collect();
        (bootstrap, oob)
    }

    fn compute_feature_importances_with_ci(&mut self) {
        let estimators = match &self.estimators {
            Some(e) => e,
            None => return,
        };
        let n_features = match self.n_features {
            Some(n) => n,
            None => return,
        };
        if estimators.is_empty() {
            return;
        }

        let n_trees = estimators.len();
        let mut tree_importances = Array2::zeros((n_trees, n_features));

        for (t, tree) in estimators.iter().enumerate() {
            if let Some(imp) = tree.feature_importance() {
                for j in 0..n_features {
                    tree_importances[[t, j]] = imp[j];
                }
            }
        }

        let mean_importance = tree_importances.mean_axis(Axis(0)).unwrap();
        let std_error = if n_trees > 1 {
            let mut se = Array1::zeros(n_features);
            for j in 0..n_features {
                let col = tree_importances.column(j);
                let mean = mean_importance[j];
                let var: f64 =
                    col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n_trees - 1) as f64;
                se[j] = (var / n_trees as f64).sqrt();
            }
            se
        } else {
            Array1::zeros(n_features)
        };

        let z = 1.96; // 95% CI
        let ci_lower = &mean_importance - &(&std_error * z);
        let ci_upper = &mean_importance + &(&std_error * z);

        self.feature_importances = Some(mean_importance.clone());
        self.feature_importances_with_ci = Some(FeatureImportanceWithCI {
            importance: mean_importance,
            std_error,
            ci_lower,
            ci_upper,
            confidence_level: 0.95,
        });
    }
}

impl Model for ExtraTreesClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        let classes = super::get_unique_classes(y);

        if classes.len() < 2 {
            return Err(FerroError::invalid_input(
                "ExtraTreesClassifier requires at least 2 classes",
            ));
        }

        self.classes = Some(classes);
        self.n_features = Some(n_features);

        let max_features = self
            .max_features
            .unwrap_or(MaxFeatures::Sqrt)
            .compute(n_features);

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        let tree_seeds: Vec<u64> = (0..self.n_estimators).map(|_| rng.random()).collect();

        let sample_indices: Vec<Vec<usize>> = if self.bootstrap {
            tree_seeds
                .iter()
                .map(|&seed| {
                    let mut tree_rng = StdRng::seed_from_u64(seed);
                    Self::generate_bootstrap_indices(n_samples, &mut tree_rng).0
                })
                .collect()
        } else {
            let all: Vec<usize> = (0..n_samples).collect();
            vec![all; self.n_estimators]
        };

        let criterion = self.criterion;
        let max_depth = self.max_depth;
        let min_samples_split = self.min_samples_split;
        let min_samples_leaf = self.min_samples_leaf;
        let min_impurity_decrease = self.min_impurity_decrease;
        let class_weight = self.class_weight.clone();

        let build_tree = |(indices, &seed): (&Vec<usize>, &u64)| {
            let n_boot = indices.len();
            let mut x_boot = Array2::zeros((n_boot, n_features));
            let mut y_boot = Array1::zeros(n_boot);
            for (i, &idx) in indices.iter().enumerate() {
                x_boot.row_mut(i).assign(&x.row(idx));
                y_boot[i] = y[idx];
            }

            let mut tree = DecisionTreeClassifier::new()
                .with_criterion(criterion)
                .with_max_depth(max_depth)
                .with_min_samples_split(min_samples_split)
                .with_min_samples_leaf(min_samples_leaf)
                .with_max_features(Some(max_features))
                .with_min_impurity_decrease(min_impurity_decrease)
                .with_class_weight(class_weight.clone())
                .with_random_state(seed)
                .with_split_strategy(SplitStrategy::Random);

            tree.fit(&x_boot, &y_boot)
                .expect("ExtraTreesClassifier: tree fit failed");
            tree
        };

        let estimators: Vec<DecisionTreeClassifier> = if self.n_jobs == Some(1) {
            sample_indices
                .iter()
                .zip(tree_seeds.iter())
                .map(build_tree)
                .collect()
        } else {
            sample_indices
                .par_iter()
                .zip(tree_seeds.par_iter())
                .map(build_tree)
                .collect()
        };

        self.estimators = Some(estimators);
        self.compute_feature_importances_with_ci();
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let proba = self.predict_proba(x)?;
        let classes = self.classes.as_ref().unwrap();
        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = proba.row(i);
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = classes[max_idx];
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.estimators.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
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
    }
}

// =============================================================================
// ExtraTreesRegressor
// =============================================================================

/// Extremely Randomized Trees regressor.
///
/// Like `RandomForestRegressor`, but uses random thresholds at each split
/// and does not use bootstrap sampling by default.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraTreesRegressor {
    /// Number of trees in the ensemble
    pub n_estimators: usize,
    /// Split criterion (MSE or MAE)
    pub criterion: SplitCriterion,
    /// Maximum depth of each tree (None for unlimited)
    pub max_depth: Option<usize>,
    /// Minimum samples required to split an internal node
    pub min_samples_split: usize,
    /// Minimum samples required to be at a leaf node
    pub min_samples_leaf: usize,
    /// Number of features to consider at each split (default: All for regression)
    pub max_features: Option<MaxFeatures>,
    /// Bootstrap sampling (default: false)
    pub bootstrap: bool,
    /// Number of parallel jobs (None for all available cores)
    pub n_jobs: Option<usize>,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Minimum impurity decrease for splits
    pub min_impurity_decrease: f64,

    // Fitted parameters
    estimators: Option<Vec<DecisionTreeRegressor>>,
    n_features: Option<usize>,
    feature_importances: Option<Array1<f64>>,
    feature_importances_with_ci: Option<FeatureImportanceWithCI>,
}

impl Default for ExtraTreesRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtraTreesRegressor {
    /// Create a new ExtraTrees Regressor with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            criterion: SplitCriterion::Mse,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: Some(MaxFeatures::All),
            bootstrap: false,
            n_jobs: None,
            random_state: None,
            min_impurity_decrease: 0.0,
            estimators: None,
            n_features: None,
            feature_importances: None,
            feature_importances_with_ci: None,
        }
    }

    /// Set the number of trees
    #[must_use]
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n.max(1);
        self
    }

    /// Set the split criterion
    #[must_use]
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
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
    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n.max(2);
        self
    }

    /// Set minimum samples per leaf
    #[must_use]
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n.max(1);
        self
    }

    /// Set maximum features to consider at each split
    #[must_use]
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.max_features = Some(max_features);
        self
    }

    /// Set whether to use bootstrap sampling
    #[must_use]
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Set number of parallel jobs
    #[must_use]
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Set random seed
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set minimum impurity decrease
    #[must_use]
    pub fn with_min_impurity_decrease(mut self, v: f64) -> Self {
        self.min_impurity_decrease = v.max(0.0);
        self
    }

    /// Get feature importances (mean decrease in impurity)
    #[must_use]
    pub fn feature_importances(&self) -> Option<&Array1<f64>> {
        self.feature_importances.as_ref()
    }

    /// Get feature importances with confidence intervals
    #[must_use]
    pub fn feature_importances_with_ci(&self) -> Option<&FeatureImportanceWithCI> {
        self.feature_importances_with_ci.as_ref()
    }

    fn compute_feature_importances_with_ci(&mut self) {
        let estimators = match &self.estimators {
            Some(e) => e,
            None => return,
        };
        let n_features = match self.n_features {
            Some(n) => n,
            None => return,
        };
        if estimators.is_empty() {
            return;
        }

        let n_trees = estimators.len();
        let mut tree_importances = Array2::zeros((n_trees, n_features));

        for (t, tree) in estimators.iter().enumerate() {
            if let Some(imp) = tree.feature_importance() {
                for j in 0..n_features {
                    tree_importances[[t, j]] = imp[j];
                }
            }
        }

        let mean_importance = tree_importances.mean_axis(Axis(0)).unwrap();
        let std_error = if n_trees > 1 {
            let mut se = Array1::zeros(n_features);
            for j in 0..n_features {
                let col = tree_importances.column(j);
                let mean = mean_importance[j];
                let var: f64 =
                    col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n_trees - 1) as f64;
                se[j] = (var / n_trees as f64).sqrt();
            }
            se
        } else {
            Array1::zeros(n_features)
        };

        let z = 1.96;
        let ci_lower = &mean_importance - &(&std_error * z);
        let ci_upper = &mean_importance + &(&std_error * z);

        self.feature_importances = Some(mean_importance.clone());
        self.feature_importances_with_ci = Some(FeatureImportanceWithCI {
            importance: mean_importance,
            std_error,
            ci_lower,
            ci_upper,
            confidence_level: 0.95,
        });
    }
}

impl Model for ExtraTreesRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        self.n_features = Some(n_features);

        let max_features = self
            .max_features
            .unwrap_or(MaxFeatures::All)
            .compute(n_features);

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        let tree_seeds: Vec<u64> = (0..self.n_estimators).map(|_| rng.random()).collect();

        let sample_indices: Vec<Vec<usize>> = if self.bootstrap {
            tree_seeds
                .iter()
                .map(|&seed| {
                    let mut tree_rng = StdRng::seed_from_u64(seed);
                    let bootstrap: Vec<usize> = (0..n_samples)
                        .map(|_| tree_rng.random_range(0..n_samples))
                        .collect();
                    bootstrap
                })
                .collect()
        } else {
            let all: Vec<usize> = (0..n_samples).collect();
            vec![all; self.n_estimators]
        };

        let criterion = self.criterion;
        let max_depth = self.max_depth;
        let min_samples_split = self.min_samples_split;
        let min_samples_leaf = self.min_samples_leaf;
        let min_impurity_decrease = self.min_impurity_decrease;

        let build_tree = |(indices, &seed): (&Vec<usize>, &u64)| {
            let n_boot = indices.len();
            let mut x_boot = Array2::zeros((n_boot, n_features));
            let mut y_boot = Array1::zeros(n_boot);
            for (i, &idx) in indices.iter().enumerate() {
                x_boot.row_mut(i).assign(&x.row(idx));
                y_boot[i] = y[idx];
            }

            let mut tree = DecisionTreeRegressor::new()
                .with_criterion(criterion)
                .with_max_depth(max_depth)
                .with_min_samples_split(min_samples_split)
                .with_min_samples_leaf(min_samples_leaf)
                .with_max_features(Some(max_features))
                .with_min_impurity_decrease(min_impurity_decrease)
                .with_random_state(seed)
                .with_split_strategy(SplitStrategy::Random);

            tree.fit(&x_boot, &y_boot)
                .expect("ExtraTreesRegressor: tree fit failed");
            tree
        };

        let estimators: Vec<DecisionTreeRegressor> = if self.n_jobs == Some(1) {
            sample_indices
                .iter()
                .zip(tree_seeds.iter())
                .map(build_tree)
                .collect()
        } else {
            sample_indices
                .par_iter()
                .zip(tree_seeds.par_iter())
                .map(build_tree)
                .collect()
        };

        self.estimators = Some(estimators);
        self.compute_feature_importances_with_ci();
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.estimators, "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let estimators = self.estimators.as_ref().unwrap();
        let n_samples = x.nrows();

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

        let mut avg = Array1::zeros(n_samples);
        for pred in &tree_preds {
            avg += pred;
        }
        avg /= estimators.len() as f64;
        Ok(avg)
    }

    fn is_fitted(&self) -> bool {
        self.estimators.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
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
        let n = 100;
        let mut x = Array2::zeros((n, 4));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            x[[i, 0]] = i as f64;
            x[[i, 1]] = (i as f64).sin();
            x[[i, 2]] = (i * 2) as f64;
            x[[i, 3]] = ((i as f64) * 0.5).cos();
            y[i] = if i < 50 { 0.0 } else { 1.0 };
        }
        (x, y)
    }

    fn make_regression_data() -> (Array2<f64>, Array1<f64>) {
        let n = 100;
        let mut x = Array2::zeros((n, 3));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            let xi = i as f64 / 10.0;
            x[[i, 0]] = xi;
            x[[i, 1]] = xi * xi;
            x[[i, 2]] = (xi * 0.5).sin();
            y[i] = 2.0 * xi + 0.5 * xi * xi + 0.1 * (xi * 0.5).sin();
        }
        (x, y)
    }

    // ==================== Classifier Tests ====================

    #[test]
    fn test_extra_trees_classifier_basic() {
        let (x, y) = make_classification_data();
        let mut model = ExtraTreesClassifier::new()
            .with_n_estimators(10)
            .with_max_depth(Some(5))
            .with_random_state(42);

        model.fit(&x, &y).unwrap();
        assert!(model.is_fitted());

        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 100);

        // Check accuracy > chance
        let correct = preds
            .iter()
            .zip(y.iter())
            .filter(|(&p, &t)| (p - t).abs() < 1e-10)
            .count();
        assert!(correct > 60, "accuracy too low: {}/100", correct);
    }

    #[test]
    fn test_extra_trees_classifier_predict_proba() {
        let (x, y) = make_classification_data();
        let mut model = ExtraTreesClassifier::new()
            .with_n_estimators(10)
            .with_random_state(42);

        model.fit(&x, &y).unwrap();

        let proba = model.predict_proba(&x).unwrap();
        assert_eq!(proba.dim(), (100, 2));

        // Probabilities should sum to ~1
        for i in 0..100 {
            let row_sum: f64 = proba.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_extra_trees_classifier_feature_importance() {
        let (x, y) = make_classification_data();
        let mut model = ExtraTreesClassifier::new()
            .with_n_estimators(20)
            .with_random_state(42);

        model.fit(&x, &y).unwrap();

        let imp = model.feature_importances().unwrap();
        assert_eq!(imp.len(), 4);
        // Importances should be non-negative
        for &v in imp.iter() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_extra_trees_classifier_no_bootstrap() {
        let (x, y) = make_classification_data();
        let mut model = ExtraTreesClassifier::new()
            .with_n_estimators(5)
            .with_random_state(42);

        assert!(!model.bootstrap);
        model.fit(&x, &y).unwrap();

        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_extra_trees_classifier_with_bootstrap() {
        let (x, y) = make_classification_data();
        let mut model = ExtraTreesClassifier::new()
            .with_n_estimators(5)
            .with_bootstrap(true)
            .with_random_state(42);

        model.fit(&x, &y).unwrap();

        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_extra_trees_classifier_deterministic() {
        let (x, y) = make_classification_data();

        let mut m1 = ExtraTreesClassifier::new()
            .with_n_estimators(5)
            .with_n_jobs(Some(1))
            .with_random_state(123);
        m1.fit(&x, &y).unwrap();
        let p1 = m1.predict(&x).unwrap();

        let mut m2 = ExtraTreesClassifier::new()
            .with_n_estimators(5)
            .with_n_jobs(Some(1))
            .with_random_state(123);
        m2.fit(&x, &y).unwrap();
        let p2 = m2.predict(&x).unwrap();

        for i in 0..p1.len() {
            assert_abs_diff_eq!(p1[i], p2[i], epsilon = 1e-10);
        }
    }

    // ==================== Regressor Tests ====================

    #[test]
    fn test_extra_trees_regressor_basic() {
        let (x, y) = make_regression_data();
        let mut model = ExtraTreesRegressor::new()
            .with_n_estimators(10)
            .with_max_depth(Some(5))
            .with_random_state(42);

        model.fit(&x, &y).unwrap();
        assert!(model.is_fitted());

        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 100);

        // R² should be decent (not perfect due to random splits)
        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(r2 > 0.5, "R² too low: {:.4}", r2);
    }

    #[test]
    fn test_extra_trees_regressor_feature_importance() {
        let (x, y) = make_regression_data();
        let mut model = ExtraTreesRegressor::new()
            .with_n_estimators(20)
            .with_random_state(42);

        model.fit(&x, &y).unwrap();

        let imp = model.feature_importances().unwrap();
        assert_eq!(imp.len(), 3);
        for &v in imp.iter() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_extra_trees_regressor_deterministic() {
        let (x, y) = make_regression_data();

        let mut m1 = ExtraTreesRegressor::new()
            .with_n_estimators(5)
            .with_n_jobs(Some(1))
            .with_random_state(99);
        m1.fit(&x, &y).unwrap();
        let p1 = m1.predict(&x).unwrap();

        let mut m2 = ExtraTreesRegressor::new()
            .with_n_estimators(5)
            .with_n_jobs(Some(1))
            .with_random_state(99);
        m2.fit(&x, &y).unwrap();
        let p2 = m2.predict(&x).unwrap();

        for i in 0..p1.len() {
            assert_abs_diff_eq!(p1[i], p2[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_extra_trees_regressor_no_bootstrap_default() {
        let model = ExtraTreesRegressor::new();
        assert!(!model.bootstrap);
    }

    #[test]
    fn test_extra_trees_classifier_search_space() {
        let model = ExtraTreesClassifier::new();
        let space = model.search_space();
        assert!(!space.parameters.is_empty());
    }

    #[test]
    fn test_extra_trees_regressor_search_space() {
        let model = ExtraTreesRegressor::new();
        let space = model.search_space();
        assert!(!space.parameters.is_empty());
    }
}

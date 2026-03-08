//! Isolation Forest for anomaly detection
//!
//! Uses random recursive partitioning — anomalies are isolated in fewer splits
//! (shorter average path length across trees).
//!
//! ## Sign conventions (matching sklearn)
//!
//! - `score_samples()`: negative of the anomaly score. Lower = more anomalous.
//! - `decision_function()`: `score_samples() - offset_`. Negative = outlier.
//! - `predict()`: returns +1 (inlier) or -1 (outlier).

use crate::models::traits::OutlierDetector;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

// =============================================================================
// Configuration enums
// =============================================================================

/// How many samples to draw per tree.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MaxSamples {
    /// `min(256, n_samples)` — sklearn default.
    Auto,
    /// Fixed absolute count.
    Count(usize),
    /// Fraction of n_samples in (0, 1].
    Fraction(f64),
}

impl Default for MaxSamples {
    fn default() -> Self {
        Self::Auto
    }
}

/// How to set the decision threshold.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Contamination {
    /// Offset = −0.5 (no fitting of threshold).
    Auto,
    /// Expected proportion of outliers in (0, 0.5].
    Proportion(f64),
}

impl Default for Contamination {
    fn default() -> Self {
        Self::Auto
    }
}

// =============================================================================
// Isolation tree structures
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IsoNode {
    feature: usize,
    threshold: f64,
    left: Option<usize>,
    right: Option<usize>,
    size: usize, // samples that reached this node
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IsolationTree {
    nodes: Vec<IsoNode>,
}

// =============================================================================
// IsolationForest
// =============================================================================

/// Isolation Forest anomaly detector.
///
/// # Example
/// ```
/// use ferroml_core::models::isolation_forest::IsolationForest;
/// use ferroml_core::models::OutlierDetector;
/// use ndarray::Array2;
///
/// let mut data = vec![0.0; 200]; // 100 inliers near 0
/// for i in 0..100 { data[i*2] = i as f64 * 0.01; data[i*2+1] = i as f64 * 0.01; }
/// let mut x = Array2::from_shape_vec((100, 2), data).unwrap();
///
/// let mut model = IsolationForest::new(100).with_random_state(42);
/// model.fit_unsupervised(&x).unwrap();
/// let preds = model.predict_outliers(&x).unwrap();
/// assert!(preds.iter().all(|&p| p == 1 || p == -1));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationForest {
    n_estimators: usize,
    max_samples: MaxSamples,
    contamination: Contamination,
    max_features: f64,
    bootstrap: bool,
    random_state: Option<u64>,
    // Fitted state
    trees: Option<Vec<IsolationTree>>,
    offset_: Option<f64>,
    max_samples_: Option<usize>,
    n_features_in_: Option<usize>,
}

impl IsolationForest {
    /// Create a new IsolationForest with the given number of trees.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            max_samples: MaxSamples::default(),
            contamination: Contamination::default(),
            max_features: 1.0,
            bootstrap: false,
            random_state: None,
            trees: None,
            offset_: None,
            max_samples_: None,
            n_features_in_: None,
        }
    }

    pub fn with_max_samples(mut self, max_samples: MaxSamples) -> Self {
        self.max_samples = max_samples;
        self
    }

    pub fn with_contamination(mut self, contamination: Contamination) -> Self {
        self.contamination = contamination;
        self
    }

    pub fn with_max_features(mut self, max_features: f64) -> Self {
        self.max_features = max_features;
        self
    }

    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Get the fitted offset threshold.
    pub fn offset_value(&self) -> Option<f64> {
        self.offset_
    }

    /// Get the resolved max_samples used during fitting.
    pub fn max_samples_value(&self) -> Option<usize> {
        self.max_samples_
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    fn resolve_max_samples(&self, n_samples: usize) -> usize {
        match self.max_samples {
            MaxSamples::Auto => n_samples.min(256),
            MaxSamples::Count(n) => n.min(n_samples),
            MaxSamples::Fraction(f) => ((n_samples as f64 * f).ceil() as usize).max(1),
        }
    }

    /// Average path length of unsuccessful search in a BST with n elements.
    /// This is the normalisation factor c(n).
    fn average_path_length(n: f64) -> f64 {
        if n <= 1.0 {
            0.0
        } else if n <= 2.0 {
            1.0
        } else {
            let euler_gamma = 0.5772156649015329;
            2.0 * ((n - 1.0).ln() + euler_gamma) - 2.0 * (n - 1.0) / n
        }
    }

    /// Build one isolation tree on a subsample.
    fn build_tree(
        x: &Array2<f64>,
        indices: &[usize],
        max_depth: usize,
        n_features: usize,
        max_features_count: usize,
        rng: &mut StdRng,
    ) -> IsolationTree {
        let mut nodes = Vec::new();
        Self::build_node(
            x,
            indices,
            0,
            max_depth,
            n_features,
            max_features_count,
            rng,
            &mut nodes,
        );
        IsolationTree { nodes }
    }

    fn build_node(
        x: &Array2<f64>,
        indices: &[usize],
        depth: usize,
        max_depth: usize,
        n_features: usize,
        max_features_count: usize,
        rng: &mut StdRng,
        nodes: &mut Vec<IsoNode>,
    ) -> usize {
        let node_idx = nodes.len();

        // Terminal: max depth or too few samples
        if depth >= max_depth || indices.len() <= 1 {
            nodes.push(IsoNode {
                feature: 0,
                threshold: 0.0,
                left: None,
                right: None,
                size: indices.len(),
            });
            return node_idx;
        }

        // Select random subset of features
        let features: Vec<usize> = if max_features_count >= n_features {
            (0..n_features).collect()
        } else {
            let mut feats = Vec::with_capacity(max_features_count);
            while feats.len() < max_features_count {
                let f = rng.random_range(0..n_features);
                if !feats.contains(&f) {
                    feats.push(f);
                }
            }
            feats
        };

        // Try features until we find one with non-zero range
        for &feat in &features {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for &idx in indices {
                let v = x[[idx, feat]];
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
            }

            if (max_val - min_val).abs() < f64::EPSILON {
                continue; // constant feature, try next
            }

            let threshold = rng.random_range(min_val..max_val);

            // Placeholder node
            nodes.push(IsoNode {
                feature: feat,
                threshold,
                left: None,
                right: None,
                size: indices.len(),
            });

            let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
                .iter()
                .partition(|&&idx| x[[idx, feat]] <= threshold);

            let left_child = Self::build_node(
                x,
                &left_indices,
                depth + 1,
                max_depth,
                n_features,
                max_features_count,
                rng,
                nodes,
            );
            let right_child = Self::build_node(
                x,
                &right_indices,
                depth + 1,
                max_depth,
                n_features,
                max_features_count,
                rng,
                nodes,
            );

            nodes[node_idx].left = Some(left_child);
            nodes[node_idx].right = Some(right_child);
            return node_idx;
        }

        // All selected features were constant — make leaf
        nodes.push(IsoNode {
            feature: 0,
            threshold: 0.0,
            left: None,
            right: None,
            size: indices.len(),
        });
        node_idx
    }

    /// Compute path length for a single sample through a single tree.
    fn path_length(tree: &IsolationTree, sample: &[f64]) -> f64 {
        let mut node_idx = 0;
        let mut depth = 0.0;

        loop {
            let node = &tree.nodes[node_idx];

            // Leaf node
            if node.left.is_none() && node.right.is_none() {
                // Add expected additional path length for remaining samples
                return depth + Self::average_path_length(node.size as f64);
            }

            if sample[node.feature] <= node.threshold {
                node_idx = node.left.unwrap();
            } else {
                node_idx = node.right.unwrap();
            }
            depth += 1.0;
        }
    }

    /// Compute raw anomaly scores for all samples.
    /// Returns the negative of the sklearn anomaly score (lower = more anomalous).
    fn compute_score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let trees = self
            .trees
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("IsolationForest"))?;

        let n_samples = x.nrows();
        let max_samples = self.max_samples_.unwrap();
        let c_n = Self::average_path_length(max_samples as f64);

        if c_n == 0.0 {
            // Degenerate case: all scores are 0
            return Ok(Array1::zeros(n_samples));
        }

        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample: Vec<f64> = x.row(i).to_vec();
            let mean_path: f64 = trees
                .iter()
                .map(|tree| Self::path_length(tree, &sample))
                .sum::<f64>()
                / trees.len() as f64;

            // anomaly_score = 2^(-mean_path / c(n))
            // score_samples = -anomaly_score (lower = more anomalous)
            let anomaly_score = 2.0_f64.powf(-mean_path / c_n);
            scores[i] = -anomaly_score;
        }

        Ok(scores)
    }
}

impl OutlierDetector for IsolationForest {
    fn fit_unsupervised(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(FerroError::InvalidInput("Cannot fit on empty data".into()));
        }

        self.n_features_in_ = Some(n_features);
        let max_samples = self.resolve_max_samples(n_samples);
        self.max_samples_ = Some(max_samples);

        let max_depth = (max_samples as f64).log2().ceil() as usize;
        let max_features_count = ((n_features as f64 * self.max_features).ceil() as usize).max(1);

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        let mut trees = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            // Subsample indices
            let indices: Vec<usize> = if self.bootstrap {
                (0..max_samples)
                    .map(|_| rng.random_range(0..n_samples))
                    .collect()
            } else if max_samples < n_samples {
                // Sample without replacement using Fisher-Yates partial shuffle
                let mut pool: Vec<usize> = (0..n_samples).collect();
                for i in 0..max_samples {
                    let j = rng.random_range(i..n_samples);
                    pool.swap(i, j);
                }
                pool[..max_samples].to_vec()
            } else {
                (0..n_samples).collect()
            };

            let tree = Self::build_tree(
                x,
                &indices,
                max_depth,
                n_features,
                max_features_count,
                &mut rng,
            );
            trees.push(tree);
        }

        self.trees = Some(trees);

        // Set offset based on contamination
        match self.contamination {
            Contamination::Auto => {
                self.offset_ = Some(-0.5);
            }
            Contamination::Proportion(c) => {
                // Fit threshold on training data
                let scores = self.compute_score_samples(x)?;
                let mut sorted: Vec<f64> = scores.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                // Threshold at the c-th percentile (lower scores = outliers)
                let idx = ((c * n_samples as f64).ceil() as usize)
                    .min(n_samples)
                    .max(1)
                    - 1;
                self.offset_ = Some(sorted[idx]);
            }
        }

        Ok(())
    }

    fn predict_outliers(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let decision = self.decision_function(x)?;
        Ok(decision.mapv(|d| if d >= 0.0 { 1 } else { -1 }))
    }

    fn score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("IsolationForest"));
        }

        // Validate features
        let expected = self.n_features_in_.unwrap();
        if x.ncols() != expected {
            return Err(FerroError::shape_mismatch(
                format!("(*, {expected})"),
                format!("(*, {})", x.ncols()),
            ));
        }

        self.compute_score_samples(x)
    }

    fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let scores = self.score_samples(x)?;
        let offset = self.offset_.unwrap();
        Ok(scores.mapv(|s| s - offset))
    }

    fn is_fitted(&self) -> bool {
        self.trees.is_some()
    }

    fn offset(&self) -> f64 {
        self.offset_.unwrap_or(-0.5)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Helper: create a cluster of inliers + a few far-away outliers.
    fn make_test_data() -> Array2<f64> {
        let mut data = Vec::new();
        // 90 inliers near origin
        for i in 0..90 {
            let x = (i as f64 * 0.01) - 0.45;
            let y = (i as f64 * 0.011) - 0.5;
            data.push(x);
            data.push(y);
        }
        // 10 outliers far away
        for i in 0..10 {
            data.push(10.0 + i as f64);
            data.push(10.0 + i as f64);
        }
        Array2::from_shape_vec((100, 2), data).unwrap()
    }

    #[test]
    fn test_basic_fit_predict() {
        let x = make_test_data();
        let mut model = IsolationForest::new(100).with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        let preds = model.predict_outliers(&x).unwrap();
        assert_eq!(preds.len(), 100);
        assert!(preds.iter().all(|&p| p == 1 || p == -1));
    }

    #[test]
    fn test_detects_outliers() {
        let x = make_test_data();
        let mut model = IsolationForest::new(200)
            .with_contamination(Contamination::Proportion(0.1))
            .with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        let preds = model.predict_outliers(&x).unwrap();

        // Last 10 samples are outliers — most should be detected
        let outlier_count: usize = preds
            .slice(ndarray::s![90..])
            .iter()
            .filter(|&&p| p == -1)
            .count();
        assert!(
            outlier_count >= 7,
            "Expected most far-away points to be outliers, got {outlier_count}/10"
        );
    }

    #[test]
    fn test_score_samples_range() {
        let x = make_test_data();
        let mut model = IsolationForest::new(100).with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        let scores = model.score_samples(&x).unwrap();

        // score_samples = -anomaly_score, so in [-1, 0]
        for &s in scores.iter() {
            assert!(s >= -1.0 && s <= 0.0, "Score {s} out of range [-1, 0]");
        }
    }

    #[test]
    fn test_outliers_have_lower_scores() {
        let x = make_test_data();
        let mut model = IsolationForest::new(200).with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        let scores = model.score_samples(&x).unwrap();

        let inlier_mean: f64 = scores.slice(ndarray::s![..90]).mean().unwrap();
        let outlier_mean: f64 = scores.slice(ndarray::s![90..]).mean().unwrap();
        assert!(
            outlier_mean < inlier_mean,
            "Outlier scores ({outlier_mean}) should be lower than inlier scores ({inlier_mean})"
        );
    }

    #[test]
    fn test_decision_function() {
        let x = make_test_data();
        let mut model = IsolationForest::new(100).with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        let scores = model.score_samples(&x).unwrap();
        let decision = model.decision_function(&x).unwrap();
        let offset = model.offset();

        for i in 0..x.nrows() {
            let expected = scores[i] - offset;
            assert!(
                (decision[i] - expected).abs() < 1e-10,
                "decision[{i}]={} != score[{i}]={} - offset={offset}",
                decision[i],
                scores[i]
            );
        }
    }

    #[test]
    fn test_predict_matches_decision_function() {
        let x = make_test_data();
        let mut model = IsolationForest::new(100).with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        let decision = model.decision_function(&x).unwrap();
        let preds = model.predict_outliers(&x).unwrap();

        for i in 0..x.nrows() {
            let expected = if decision[i] >= 0.0 { 1 } else { -1 };
            assert_eq!(preds[i], expected, "Mismatch at index {i}");
        }
    }

    #[test]
    fn test_contamination_auto() {
        let x = make_test_data();
        let mut model = IsolationForest::new(50).with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        assert_eq!(model.offset(), -0.5);
    }

    #[test]
    fn test_contamination_proportion() {
        let x = make_test_data();
        let mut model = IsolationForest::new(100)
            .with_contamination(Contamination::Proportion(0.1))
            .with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        // Offset should be different from -0.5
        assert!(model.offset() != -0.5);
    }

    #[test]
    fn test_reproducibility() {
        let x = make_test_data();
        let mut m1 = IsolationForest::new(50).with_random_state(123);
        let mut m2 = IsolationForest::new(50).with_random_state(123);
        m1.fit_unsupervised(&x).unwrap();
        m2.fit_unsupervised(&x).unwrap();

        let s1 = m1.score_samples(&x).unwrap();
        let s2 = m2.score_samples(&x).unwrap();
        assert_eq!(s1, s2, "Same seed should produce identical scores");
    }

    #[test]
    fn test_different_seeds_differ() {
        let x = make_test_data();
        let mut m1 = IsolationForest::new(50).with_random_state(1);
        let mut m2 = IsolationForest::new(50).with_random_state(2);
        m1.fit_unsupervised(&x).unwrap();
        m2.fit_unsupervised(&x).unwrap();

        let s1 = m1.score_samples(&x).unwrap();
        let s2 = m2.score_samples(&x).unwrap();
        assert_ne!(s1, s2, "Different seeds should produce different scores");
    }

    #[test]
    fn test_more_trees_more_stable() {
        let x = make_test_data();
        let mut m1 = IsolationForest::new(10).with_random_state(42);
        let mut m2 = IsolationForest::new(10).with_random_state(99);
        let mut m3 = IsolationForest::new(500).with_random_state(42);
        let mut m4 = IsolationForest::new(500).with_random_state(99);

        m1.fit_unsupervised(&x).unwrap();
        m2.fit_unsupervised(&x).unwrap();
        m3.fit_unsupervised(&x).unwrap();
        m4.fit_unsupervised(&x).unwrap();

        let diff_10: f64 = (&m1.score_samples(&x).unwrap() - &m2.score_samples(&x).unwrap())
            .mapv(|v| v.abs())
            .mean()
            .unwrap();
        let diff_500: f64 = (&m3.score_samples(&x).unwrap() - &m4.score_samples(&x).unwrap())
            .mapv(|v| v.abs())
            .mean()
            .unwrap();

        assert!(
            diff_500 < diff_10,
            "500 trees diff ({diff_500}) should be less than 10 trees diff ({diff_10})"
        );
    }

    #[test]
    fn test_max_samples_count() {
        let x = make_test_data();
        let mut model = IsolationForest::new(50)
            .with_max_samples(MaxSamples::Count(50))
            .with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        assert_eq!(model.max_samples_value(), Some(50));
        let preds = model.predict_outliers(&x).unwrap();
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_max_samples_fraction() {
        let x = make_test_data();
        let mut model = IsolationForest::new(50)
            .with_max_samples(MaxSamples::Fraction(0.5))
            .with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        assert_eq!(model.max_samples_value(), Some(50));
    }

    #[test]
    fn test_max_samples_auto() {
        let x = make_test_data();
        let mut model = IsolationForest::new(50).with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        // 100 samples, auto = min(256, 100) = 100
        assert_eq!(model.max_samples_value(), Some(100));
    }

    #[test]
    fn test_bootstrap() {
        let x = make_test_data();
        let mut model = IsolationForest::new(50)
            .with_bootstrap(true)
            .with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        let preds = model.predict_outliers(&x).unwrap();
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_single_feature() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let mut x = Array2::from_shape_vec((100, 1), data).unwrap();
        // Add outlier
        x[[99, 0]] = 100.0;

        let mut model = IsolationForest::new(100)
            .with_contamination(Contamination::Proportion(0.05))
            .with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        let preds = model.predict_outliers(&x).unwrap();
        assert_eq!(preds[99], -1, "Far-away point should be an outlier");
    }

    #[test]
    fn test_high_dimensional() {
        // 50 samples, 20 features
        let n = 50;
        let d = 20;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f64> = (0..n * d).map(|_| rng.random_range(-1.0..1.0)).collect();
        let x = Array2::from_shape_vec((n, d), data).unwrap();

        let mut model = IsolationForest::new(100).with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        let preds = model.predict_outliers(&x).unwrap();
        assert_eq!(preds.len(), n);
    }

    #[test]
    fn test_all_same_values() {
        let x = Array2::ones((50, 3));
        let mut model = IsolationForest::new(50).with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        let _preds = model.predict_outliers(&x).unwrap();
        // All same — all should have similar scores
        let scores = model.score_samples(&x).unwrap();
        let first = scores[0];
        for &s in scores.iter() {
            assert!(
                (s - first).abs() < 1e-10,
                "All-same data should have identical scores"
            );
        }
    }

    #[test]
    fn test_not_fitted_error() {
        let model = IsolationForest::new(10);
        let x = Array2::zeros((5, 2));
        assert!(model.predict_outliers(&x).is_err());
        assert!(model.score_samples(&x).is_err());
        assert!(model.decision_function(&x).is_err());
    }

    #[test]
    fn test_empty_data_error() {
        let x = Array2::zeros((0, 2));
        let mut model = IsolationForest::new(10);
        assert!(model.fit_unsupervised(&x).is_err());
    }

    #[test]
    fn test_feature_mismatch_error() {
        let x_train = Array2::zeros((50, 3));
        let x_test = Array2::zeros((10, 5));
        let mut model = IsolationForest::new(10).with_random_state(42);
        model.fit_unsupervised(&x_train).unwrap();
        assert!(model.predict_outliers(&x_test).is_err());
    }

    #[test]
    fn test_is_fitted() {
        let mut model = IsolationForest::new(10);
        assert!(!model.is_fitted());
        let x = Array2::zeros((20, 2));
        model.fit_unsupervised(&x).unwrap();
        assert!(model.is_fitted());
    }

    #[test]
    fn test_fit_predict_outliers() {
        let x = make_test_data();
        let mut model = IsolationForest::new(100)
            .with_contamination(Contamination::Proportion(0.1))
            .with_random_state(42);
        let preds = model.fit_predict_outliers(&x).unwrap();
        assert_eq!(preds.len(), 100);
        assert!(model.is_fitted());
    }

    #[test]
    fn test_max_features_subset() {
        let x = make_test_data();
        let mut model = IsolationForest::new(100)
            .with_max_features(0.5)
            .with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        let preds = model.predict_outliers(&x).unwrap();
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_predict_on_new_data() {
        let x_train = make_test_data();
        let mut model = IsolationForest::new(100).with_random_state(42);
        model.fit_unsupervised(&x_train).unwrap();

        // New inlier
        let x_inlier = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let score_inlier = model.score_samples(&x_inlier).unwrap();

        // New outlier
        let x_outlier = Array2::from_shape_vec((1, 2), vec![100.0, 100.0]).unwrap();
        let score_outlier = model.score_samples(&x_outlier).unwrap();

        assert!(
            score_outlier[0] < score_inlier[0],
            "Outlier score ({}) should be lower than inlier score ({})",
            score_outlier[0],
            score_inlier[0]
        );
    }

    #[test]
    fn test_small_dataset() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.3, 0.7, 100.0, 100.0],
        )
        .unwrap();
        let mut model = IsolationForest::new(200).with_random_state(42);
        model.fit_unsupervised(&x).unwrap();
        let scores = model.score_samples(&x).unwrap();
        // The far-away point should have the lowest score
        let outlier_score = scores[4];
        let max_inlier_score = scores
            .slice(ndarray::s![..4])
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            outlier_score < max_inlier_score,
            "Outlier score ({outlier_score}) should be lower than inlier scores (max={max_inlier_score})"
        );
    }

    #[test]
    fn test_two_clusters_with_outlier() {
        let mut data = Vec::new();
        // Cluster 1 near (0, 0)
        for i in 0..40 {
            data.push(i as f64 * 0.01);
            data.push(i as f64 * 0.01);
        }
        // Cluster 2 near (5, 5)
        for i in 0..40 {
            data.push(5.0 + i as f64 * 0.01);
            data.push(5.0 + i as f64 * 0.01);
        }
        // Outlier at (50, 50)
        data.push(50.0);
        data.push(50.0);

        let x = Array2::from_shape_vec((81, 2), data).unwrap();
        let mut model = IsolationForest::new(200)
            .with_contamination(Contamination::Proportion(0.05))
            .with_random_state(42);
        model.fit_unsupervised(&x).unwrap();

        let scores = model.score_samples(&x).unwrap();
        // Outlier at index 80 should have the lowest score
        let outlier_score = scores[80];
        let min_inlier_score = scores
            .slice(ndarray::s![..80])
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert!(
            outlier_score < min_inlier_score,
            "Outlier score ({outlier_score}) should be lower than all inlier scores (min={min_inlier_score})"
        );
    }

    #[test]
    fn test_average_path_length() {
        assert_eq!(IsolationForest::average_path_length(0.0), 0.0);
        assert_eq!(IsolationForest::average_path_length(1.0), 0.0);
        assert_eq!(IsolationForest::average_path_length(2.0), 1.0);
        // c(256) ≈ 10.24 (2*(ln(255)+γ) - 2*255/256)
        let c256 = IsolationForest::average_path_length(256.0);
        assert!(
            (c256 - 10.244).abs() < 0.1,
            "c(256) = {c256}, expected ~10.244"
        );
    }

    #[test]
    fn test_offset_accessor() {
        let x = make_test_data();
        let mut model = IsolationForest::new(50).with_random_state(42);
        // Before fit, offset returns default
        assert_eq!(model.offset(), -0.5);
        model.fit_unsupervised(&x).unwrap();
        assert_eq!(model.offset(), -0.5);
    }

    #[test]
    fn test_n_features_validation() {
        let x = Array2::from_shape_vec((10, 3), vec![0.0; 30]).unwrap();
        let mut model = IsolationForest::new(10).with_random_state(42);
        model.fit_unsupervised(&x).unwrap();

        // Correct features work
        let x_ok = Array2::from_shape_vec((2, 3), vec![0.0; 6]).unwrap();
        assert!(model.score_samples(&x_ok).is_ok());

        // Wrong features fail
        let x_bad = Array2::from_shape_vec((2, 4), vec![0.0; 8]).unwrap();
        assert!(model.score_samples(&x_bad).is_err());
    }
}

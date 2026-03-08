//! Local Outlier Factor (LOF) for anomaly / novelty detection
//!
//! LOF measures local density deviation of a point relative to its neighbors.
//! Points with substantially lower density than neighbors are outliers.
//!
//! ## Modes
//!
//! - **novelty=false** (default): outlier detection on training data via `fit_predict_outliers()`
//! - **novelty=true**: novelty detection — `predict_outliers()` / `score_samples()` on new data
//!
//! ## Sign conventions (matching sklearn)
//!
//! - `negative_outlier_factor_`: opposite of LOF. Near −1 = inlier, large negative = outlier.
//! - `score_samples()`: same as `negative_outlier_factor_` for training data.
//! - `decision_function()`: `score_samples() - offset_`. Negative = outlier.
//! - `predict_outliers()`: +1 (inlier) or −1 (outlier).

use crate::models::knn::{BallTree, DistanceMetric, KDTree, KNNAlgorithm};
use crate::models::traits::OutlierDetector;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::isolation_forest::Contamination;

// =============================================================================
// LocalOutlierFactor
// =============================================================================

/// Local Outlier Factor anomaly/novelty detector.
///
/// # Example
/// ```
/// use ferroml_core::models::lof::LocalOutlierFactor;
/// use ferroml_core::models::OutlierDetector;
/// use ndarray::Array2;
///
/// let mut data = vec![0.0; 200];
/// for i in 0..100 { data[i*2] = (i % 10) as f64 * 0.1; data[i*2+1] = (i / 10) as f64 * 0.1; }
/// let x = Array2::from_shape_vec((100, 2), data).unwrap();
///
/// let mut model = LocalOutlierFactor::new(20);
/// let preds = model.fit_predict_outliers(&x).unwrap();
/// assert!(preds.iter().all(|&p| p == 1 || p == -1));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalOutlierFactor {
    n_neighbors: usize,
    contamination: Contamination,
    metric: DistanceMetric,
    algorithm: KNNAlgorithm,
    novelty: bool,
    // Fitted state
    x_train_: Option<Array2<f64>>,
    negative_outlier_factor_: Option<Array1<f64>>,
    offset_: Option<f64>,
    n_features_in_: Option<usize>,
    // Precomputed neighbor info for training data
    k_distances_: Option<Array1<f64>>,
    lrd_: Option<Array1<f64>>,
    neighbor_indices_: Option<Vec<Vec<usize>>>,
    neighbor_dists_: Option<Vec<Vec<f64>>>,
}

impl LocalOutlierFactor {
    /// Create a new LOF with the given number of neighbors.
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            contamination: Contamination::Auto,
            metric: DistanceMetric::Euclidean,
            algorithm: KNNAlgorithm::Auto,
            novelty: false,
            x_train_: None,
            negative_outlier_factor_: None,
            offset_: None,
            n_features_in_: None,
            k_distances_: None,
            lrd_: None,
            neighbor_indices_: None,
            neighbor_dists_: None,
        }
    }

    pub fn with_contamination(mut self, contamination: Contamination) -> Self {
        self.contamination = contamination;
        self
    }

    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    pub fn with_algorithm(mut self, algorithm: KNNAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    pub fn with_novelty(mut self, novelty: bool) -> Self {
        self.novelty = novelty;
        self
    }

    /// Get the negative outlier factor for each training sample.
    /// Near −1 = inlier, large negative = outlier.
    pub fn negative_outlier_factor(&self) -> Option<&Array1<f64>> {
        self.negative_outlier_factor_.as_ref()
    }

    /// Get the fitted offset threshold.
    pub fn offset_value(&self) -> Option<f64> {
        self.offset_
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /// Effective k, clamped to n_samples - 1.
    fn effective_k(&self, n_samples: usize) -> usize {
        self.n_neighbors.min(n_samples - 1).max(1)
    }

    /// Find k nearest neighbors for all points in x, using training data.
    /// Returns (indices, distances) for each query point.
    /// When querying training data (is_training=true), excludes self-matches.
    fn find_neighbors(
        &self,
        x: &Array2<f64>,
        x_train: &Array2<f64>,
        k: usize,
        is_training: bool,
    ) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
        let query_k = if is_training { k + 1 } else { k };

        let effective_alg = match self.algorithm {
            KNNAlgorithm::Auto => {
                if x_train.ncols() <= 15 {
                    KNNAlgorithm::KDTree
                } else {
                    KNNAlgorithm::BallTree
                }
            }
            other => other,
        };

        let n_queries = x.nrows();
        let mut all_indices = Vec::with_capacity(n_queries);
        let mut all_dists = Vec::with_capacity(n_queries);

        match effective_alg {
            KNNAlgorithm::KDTree => {
                let kdtree = KDTree::build(x_train.clone(), 30);
                for i in 0..n_queries {
                    let query: Vec<f64> = x.row(i).to_vec();
                    let results = kdtree.query(&query, query_k, &self.metric);
                    let (idxs, dists): (Vec<usize>, Vec<f64>) = if is_training {
                        results
                            .into_iter()
                            .filter(|(idx, _)| *idx != i)
                            .take(k)
                            .unzip()
                    } else {
                        results.into_iter().take(k).unzip()
                    };
                    all_indices.push(idxs);
                    all_dists.push(dists);
                }
            }
            KNNAlgorithm::BallTree => {
                let balltree = BallTree::build(x_train.clone(), 30);
                for i in 0..n_queries {
                    let query: Vec<f64> = x.row(i).to_vec();
                    let results = balltree.query(&query, query_k, &self.metric);
                    let (idxs, dists): (Vec<usize>, Vec<f64>) = if is_training {
                        results
                            .into_iter()
                            .filter(|(idx, _)| *idx != i)
                            .take(k)
                            .unzip()
                    } else {
                        results.into_iter().take(k).unzip()
                    };
                    all_indices.push(idxs);
                    all_dists.push(dists);
                }
            }
            KNNAlgorithm::BruteForce | KNNAlgorithm::Auto => {
                for i in 0..n_queries {
                    let query: Vec<f64> = x.row(i).to_vec();
                    let mut distances: Vec<(usize, f64)> = x_train
                        .rows()
                        .into_iter()
                        .enumerate()
                        .map(|(idx, row)| {
                            let point: Vec<f64> = row.to_vec();
                            (idx, self.metric.compute(&query, &point))
                        })
                        .collect();
                    distances
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                    let (idxs, dists): (Vec<usize>, Vec<f64>) = if is_training {
                        distances
                            .into_iter()
                            .filter(|(idx, _)| *idx != i)
                            .take(k)
                            .unzip()
                    } else {
                        distances.into_iter().take(k).unzip()
                    };
                    all_indices.push(idxs);
                    all_dists.push(dists);
                }
            }
        }

        (all_indices, all_dists)
    }

    /// Compute local reachability density for a set of points.
    /// k_distances: k-distance of each training point.
    /// neighbor_indices/dists: neighbors for each query point.
    fn compute_lrd(
        &self,
        neighbor_indices: &[Vec<usize>],
        neighbor_dists: &[Vec<f64>],
        k_distances: &Array1<f64>,
    ) -> Array1<f64> {
        let n = neighbor_indices.len();
        let mut lrd = Array1::zeros(n);

        for i in 0..n {
            let mut reach_dist_sum = 0.0;
            let k_actual = neighbor_indices[i].len();
            if k_actual == 0 {
                lrd[i] = f64::INFINITY;
                continue;
            }

            for j in 0..k_actual {
                let neighbor_idx = neighbor_indices[i][j];
                let dist = neighbor_dists[i][j];
                // reach-dist(A, B) = max(k-distance(B), d(A, B))
                let reach_dist = k_distances[neighbor_idx].max(dist);
                reach_dist_sum += reach_dist;
            }

            let mean_reach_dist = reach_dist_sum / k_actual as f64;
            if mean_reach_dist == 0.0 {
                lrd[i] = f64::INFINITY;
            } else {
                lrd[i] = 1.0 / mean_reach_dist;
            }
        }

        lrd
    }

    /// Compute LOF scores given lrd values and neighbor info.
    /// Returns negative_outlier_factor (opposite of LOF: near -1 = inlier).
    fn compute_lof(
        &self,
        lrd: &Array1<f64>,
        neighbor_indices: &[Vec<usize>],
        training_lrd: &Array1<f64>,
    ) -> Array1<f64> {
        let n = lrd.len();
        let mut nof = Array1::zeros(n);

        for i in 0..n {
            let k_actual = neighbor_indices[i].len();
            if k_actual == 0 || lrd[i].is_infinite() {
                nof[i] = -1.0; // Treat as inlier if we can't compute
                continue;
            }

            let mut lof_sum = 0.0;
            for &neighbor_idx in &neighbor_indices[i] {
                if training_lrd[neighbor_idx].is_infinite() {
                    // Neighbor has zero reach distance — skip or treat as 1
                    lof_sum += 1.0;
                } else {
                    lof_sum += training_lrd[neighbor_idx] / lrd[i];
                }
            }

            let lof = lof_sum / k_actual as f64;
            nof[i] = -lof; // negative: near -1 = inlier, large negative = outlier
        }

        nof
    }

    /// Compute score_samples for new data points (novelty mode).
    fn compute_score_samples_new(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let x_train = self
            .x_train_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("LocalOutlierFactor"))?;
        let k_distances = self
            .k_distances_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("LocalOutlierFactor"))?;
        let training_lrd = self
            .lrd_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("LocalOutlierFactor"))?;

        let k = self.effective_k(x_train.nrows());
        let (neighbor_indices, neighbor_dists) = self.find_neighbors(x, x_train, k, false);

        let lrd = self.compute_lrd(&neighbor_indices, &neighbor_dists, k_distances);
        let nof = self.compute_lof(&lrd, &neighbor_indices, training_lrd);

        Ok(nof)
    }
}

impl OutlierDetector for LocalOutlierFactor {
    fn fit_unsupervised(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples < 2 {
            return Err(FerroError::InvalidInput(
                "LOF requires at least 2 samples".into(),
            ));
        }

        self.n_features_in_ = Some(n_features);
        self.x_train_ = Some(x.to_owned());

        let k = self.effective_k(n_samples);

        // Find k nearest neighbors for all training points
        let (neighbor_indices, neighbor_dists) = self.find_neighbors(x, x, k, true);

        // k-distance for each point = distance to k-th nearest neighbor
        let k_distances = Array1::from_vec(
            neighbor_dists
                .iter()
                .map(|dists| {
                    if dists.is_empty() {
                        0.0
                    } else {
                        *dists.last().unwrap()
                    }
                })
                .collect(),
        );

        // Local reachability density
        let lrd = self.compute_lrd(&neighbor_indices, &neighbor_dists, &k_distances);

        // LOF scores (negative)
        let nof = self.compute_lof(&lrd, &neighbor_indices, &lrd);

        // Set offset
        match self.contamination {
            Contamination::Auto => {
                self.offset_ = Some(-1.5);
            }
            Contamination::Proportion(c) => {
                let mut sorted: Vec<f64> = nof.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = ((c * n_samples as f64).ceil() as usize)
                    .min(n_samples)
                    .max(1)
                    - 1;
                self.offset_ = Some(sorted[idx]);
            }
        }

        self.negative_outlier_factor_ = Some(nof);
        self.k_distances_ = Some(k_distances);
        self.lrd_ = Some(lrd);
        self.neighbor_indices_ = Some(neighbor_indices);
        self.neighbor_dists_ = Some(neighbor_dists);

        Ok(())
    }

    fn predict_outliers(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        if !self.novelty {
            return Err(FerroError::InvalidInput(
                "predict_outliers is not available when novelty=false. \
                 Use fit_predict_outliers() for outlier detection, or \
                 set novelty=true for novelty detection."
                    .into(),
            ));
        }
        let decision = self.decision_function(x)?;
        Ok(decision.mapv(|d| if d >= 0.0 { 1 } else { -1 }))
    }

    fn fit_predict_outliers(&mut self, x: &Array2<f64>) -> Result<Array1<i32>> {
        self.fit_unsupervised(x)?;
        let nof = self.negative_outlier_factor_.as_ref().unwrap();
        let offset = self.offset_.unwrap();
        Ok(nof.mapv(|s| if s - offset >= 0.0 { 1 } else { -1 }))
    }

    fn score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("LocalOutlierFactor"));
        }

        let expected = self.n_features_in_.unwrap();
        if x.ncols() != expected {
            return Err(FerroError::shape_mismatch(
                format!("(*, {expected})"),
                format!("(*, {})", x.ncols()),
            ));
        }

        if self.novelty {
            self.compute_score_samples_new(x)
        } else {
            Err(FerroError::InvalidInput(
                "score_samples on new data is not available when novelty=false. \
                 Use negative_outlier_factor() for training scores."
                    .into(),
            ))
        }
    }

    fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let scores = self.score_samples(x)?;
        let offset = self.offset_.unwrap();
        Ok(scores.mapv(|s| s - offset))
    }

    fn is_fitted(&self) -> bool {
        self.x_train_.is_some()
    }

    fn offset(&self) -> f64 {
        self.offset_.unwrap_or(-1.5)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Helper: cluster of inliers + a few far-away outliers.
    fn make_test_data() -> Array2<f64> {
        let mut data = Vec::new();
        // 90 inliers in a grid near origin
        for i in 0..90 {
            let x = (i % 10) as f64 * 0.1;
            let y = (i / 10) as f64 * 0.1;
            data.push(x);
            data.push(y);
        }
        // 10 outliers far away
        for i in 0..10 {
            data.push(20.0 + i as f64 * 2.0);
            data.push(20.0 + i as f64 * 2.0);
        }
        Array2::from_shape_vec((100, 2), data).unwrap()
    }

    #[test]
    fn test_basic_fit_predict() {
        let x = make_test_data();
        let mut model = LocalOutlierFactor::new(20);
        let preds = model.fit_predict_outliers(&x).unwrap();
        assert_eq!(preds.len(), 100);
        assert!(preds.iter().all(|&p| p == 1 || p == -1));
    }

    #[test]
    fn test_detects_outliers() {
        let x = make_test_data();
        let mut model =
            LocalOutlierFactor::new(20).with_contamination(Contamination::Proportion(0.1));
        let preds = model.fit_predict_outliers(&x).unwrap();

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
    fn test_negative_outlier_factor_near_minus_one_for_inliers() {
        // Dense uniform grid — all points should have LOF ~ 1 (NOF ~ -1)
        let mut data = Vec::new();
        for i in 0..100 {
            data.push((i % 10) as f64);
            data.push((i / 10) as f64);
        }
        let x = Array2::from_shape_vec((100, 2), data).unwrap();
        let mut model = LocalOutlierFactor::new(10);
        model.fit_unsupervised(&x).unwrap();

        let nof = model.negative_outlier_factor().unwrap();
        // Interior points should have NOF close to -1
        // Edge points may deviate slightly
        let interior_nof: Vec<f64> = nof
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let row = i / 10;
                let col = i % 10;
                row >= 2 && row <= 7 && col >= 2 && col <= 7
            })
            .map(|(_, &v)| v)
            .collect();

        let mean_nof: f64 = interior_nof.iter().sum::<f64>() / interior_nof.len() as f64;
        assert!(
            (mean_nof - (-1.0)).abs() < 0.15,
            "Interior mean NOF = {mean_nof}, expected ~-1.0"
        );
    }

    #[test]
    fn test_outliers_have_more_negative_nof() {
        let x = make_test_data();
        let mut model = LocalOutlierFactor::new(20);
        model.fit_unsupervised(&x).unwrap();
        let nof = model.negative_outlier_factor().unwrap();

        let inlier_mean: f64 = nof.slice(ndarray::s![..90]).mean().unwrap();
        let outlier_mean: f64 = nof.slice(ndarray::s![90..]).mean().unwrap();
        assert!(
            outlier_mean < inlier_mean,
            "Outlier NOF ({outlier_mean}) should be more negative than inlier NOF ({inlier_mean})"
        );
    }

    #[test]
    fn test_novelty_mode_predict() {
        let x = make_test_data();
        let mut model = LocalOutlierFactor::new(20)
            .with_novelty(true)
            .with_contamination(Contamination::Proportion(0.1));
        model.fit_unsupervised(&x).unwrap();

        // New inlier
        let x_new = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();
        let pred = model.predict_outliers(&x_new).unwrap();
        assert_eq!(pred[0], 1, "Inlier should be +1");

        // New outlier
        let x_out = Array2::from_shape_vec((1, 2), vec![100.0, 100.0]).unwrap();
        let pred_out = model.predict_outliers(&x_out).unwrap();
        assert_eq!(pred_out[0], -1, "Far-away point should be -1");
    }

    #[test]
    fn test_novelty_false_rejects_predict() {
        let x = make_test_data();
        let mut model = LocalOutlierFactor::new(20);
        model.fit_unsupervised(&x).unwrap();

        let x_new = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();
        assert!(
            model.predict_outliers(&x_new).is_err(),
            "predict_outliers should fail when novelty=false"
        );
    }

    #[test]
    fn test_novelty_false_rejects_score_samples() {
        let x = make_test_data();
        let mut model = LocalOutlierFactor::new(20);
        model.fit_unsupervised(&x).unwrap();

        let x_new = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();
        assert!(
            model.score_samples(&x_new).is_err(),
            "score_samples should fail when novelty=false"
        );
    }

    #[test]
    fn test_novelty_score_samples() {
        let x = make_test_data();
        let mut model = LocalOutlierFactor::new(20).with_novelty(true);
        model.fit_unsupervised(&x).unwrap();

        let x_inlier = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();
        let x_outlier = Array2::from_shape_vec((1, 2), vec![100.0, 100.0]).unwrap();

        let score_in = model.score_samples(&x_inlier).unwrap();
        let score_out = model.score_samples(&x_outlier).unwrap();

        assert!(
            score_out[0] < score_in[0],
            "Outlier score ({}) should be lower than inlier score ({})",
            score_out[0],
            score_in[0]
        );
    }

    #[test]
    fn test_decision_function_novelty() {
        let x = make_test_data();
        let mut model = LocalOutlierFactor::new(20)
            .with_novelty(true)
            .with_contamination(Contamination::Proportion(0.1));
        model.fit_unsupervised(&x).unwrap();

        let x_new = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 100.0, 100.0]).unwrap();
        let scores = model.score_samples(&x_new).unwrap();
        let decision = model.decision_function(&x_new).unwrap();
        let offset = model.offset();

        for i in 0..2 {
            assert!(
                (decision[i] - (scores[i] - offset)).abs() < 1e-10,
                "decision_function should be score_samples - offset"
            );
        }
    }

    #[test]
    fn test_n_neighbors_affects_sensitivity() {
        let x = make_test_data();

        let mut model_small = LocalOutlierFactor::new(5);
        model_small.fit_unsupervised(&x).unwrap();
        let nof_small = model_small.negative_outlier_factor().unwrap().clone();

        let mut model_large = LocalOutlierFactor::new(50);
        model_large.fit_unsupervised(&x).unwrap();
        let nof_large = model_large.negative_outlier_factor().unwrap().clone();

        // Different k should produce different scores
        assert_ne!(
            nof_small, nof_large,
            "Different n_neighbors should give different NOFs"
        );
    }

    #[test]
    fn test_different_metrics() {
        let x = make_test_data();

        let mut model_euc = LocalOutlierFactor::new(20).with_metric(DistanceMetric::Euclidean);
        model_euc.fit_unsupervised(&x).unwrap();

        let mut model_man = LocalOutlierFactor::new(20).with_metric(DistanceMetric::Manhattan);
        model_man.fit_unsupervised(&x).unwrap();

        let nof_euc = model_euc.negative_outlier_factor().unwrap();
        let nof_man = model_man.negative_outlier_factor().unwrap();
        assert_ne!(
            nof_euc, nof_man,
            "Different metrics should give different NOFs"
        );
    }

    #[test]
    fn test_brute_force_algorithm() {
        let x = make_test_data();
        let mut model = LocalOutlierFactor::new(20).with_algorithm(KNNAlgorithm::BruteForce);
        let preds = model.fit_predict_outliers(&x).unwrap();
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_ball_tree_algorithm() {
        let x = make_test_data();
        let mut model = LocalOutlierFactor::new(20).with_algorithm(KNNAlgorithm::BallTree);
        let preds = model.fit_predict_outliers(&x).unwrap();
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_kdtree_algorithm() {
        let x = make_test_data();
        let mut model = LocalOutlierFactor::new(20).with_algorithm(KNNAlgorithm::KDTree);
        let preds = model.fit_predict_outliers(&x).unwrap();
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_contamination_auto() {
        let x = make_test_data();
        let mut model = LocalOutlierFactor::new(20);
        model.fit_unsupervised(&x).unwrap();
        assert_eq!(model.offset(), -1.5);
    }

    #[test]
    fn test_contamination_proportion() {
        let x = make_test_data();
        let mut model =
            LocalOutlierFactor::new(20).with_contamination(Contamination::Proportion(0.1));
        model.fit_unsupervised(&x).unwrap();
        assert!(
            model.offset() != -1.5,
            "Proportion contamination should set a different offset"
        );
    }

    #[test]
    fn test_deterministic() {
        let x = make_test_data();
        let mut m1 = LocalOutlierFactor::new(20);
        let mut m2 = LocalOutlierFactor::new(20);
        m1.fit_unsupervised(&x).unwrap();
        m2.fit_unsupervised(&x).unwrap();

        let nof1 = m1.negative_outlier_factor().unwrap();
        let nof2 = m2.negative_outlier_factor().unwrap();
        assert_eq!(nof1, nof2, "LOF should be deterministic for same input");
    }

    #[test]
    fn test_not_fitted_error() {
        let model = LocalOutlierFactor::new(20).with_novelty(true);
        let x = Array2::zeros((5, 2));
        assert!(model.score_samples(&x).is_err());
        assert!(model.decision_function(&x).is_err());
    }

    #[test]
    fn test_too_few_samples() {
        let x = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let mut model = LocalOutlierFactor::new(20);
        assert!(model.fit_unsupervised(&x).is_err());
    }

    #[test]
    fn test_feature_mismatch() {
        let x_train = Array2::zeros((50, 3));
        let x_test = Array2::zeros((10, 5));
        let mut model = LocalOutlierFactor::new(10).with_novelty(true);
        model.fit_unsupervised(&x_train).unwrap();
        assert!(model.score_samples(&x_test).is_err());
    }

    #[test]
    fn test_is_fitted() {
        let mut model = LocalOutlierFactor::new(10);
        assert!(!model.is_fitted());
        let x = Array2::zeros((20, 2));
        model.fit_unsupervised(&x).unwrap();
        assert!(model.is_fitted());
    }

    #[test]
    fn test_k_greater_than_n_samples() {
        // k=20 but only 5 samples — should clamp to k=4
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )
        .unwrap();
        let mut model = LocalOutlierFactor::new(20);
        let preds = model.fit_predict_outliers(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_single_dense_cluster() {
        // All points in tight cluster — all should be inliers
        let mut data = Vec::new();
        for i in 0..50 {
            data.push((i % 5) as f64 * 0.01);
            data.push((i / 5) as f64 * 0.01);
        }
        let x = Array2::from_shape_vec((50, 2), data).unwrap();
        let mut model = LocalOutlierFactor::new(10).with_contamination(Contamination::Auto);
        let preds = model.fit_predict_outliers(&x).unwrap();

        let outlier_count: usize = preds.iter().filter(|&&p| p == -1).count();
        // With auto contamination, most should be inliers in a uniform cluster
        assert!(
            outlier_count <= 10,
            "Dense cluster should have very few outliers, got {outlier_count}/50"
        );
    }

    #[test]
    fn test_isolated_point_highest_lof() {
        let mut data = Vec::new();
        // 20 points in tight cluster
        for i in 0..20 {
            data.push((i % 5) as f64 * 0.1);
            data.push((i / 5) as f64 * 0.1);
        }
        // 1 isolated point
        data.push(100.0);
        data.push(100.0);

        let x = Array2::from_shape_vec((21, 2), data).unwrap();
        let mut model = LocalOutlierFactor::new(10);
        model.fit_unsupervised(&x).unwrap();

        let nof = model.negative_outlier_factor().unwrap();
        // Isolated point (index 20) should have the most negative NOF
        let isolated_nof = nof[20];
        let min_cluster_nof = nof
            .slice(ndarray::s![..20])
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert!(
            isolated_nof < min_cluster_nof,
            "Isolated point NOF ({isolated_nof}) should be more negative than cluster ({min_cluster_nof})"
        );
    }

    #[test]
    fn test_two_clusters_boundary() {
        let mut data = Vec::new();
        // Cluster A near (0,0)
        for i in 0..30 {
            data.push((i % 6) as f64 * 0.1);
            data.push((i / 6) as f64 * 0.1);
        }
        // Cluster B near (10,10)
        for i in 0..30 {
            data.push(10.0 + (i % 6) as f64 * 0.1);
            data.push(10.0 + (i / 6) as f64 * 0.1);
        }
        let x = Array2::from_shape_vec((60, 2), data).unwrap();
        let mut model = LocalOutlierFactor::new(10);
        model.fit_unsupervised(&x).unwrap();

        // Both clusters should have similar NOF (both dense)
        let nof = model.negative_outlier_factor().unwrap();
        let mean_a: f64 = nof.slice(ndarray::s![..30]).mean().unwrap();
        let mean_b: f64 = nof.slice(ndarray::s![30..]).mean().unwrap();
        assert!(
            (mean_a - mean_b).abs() < 0.5,
            "Two equal clusters should have similar NOF: A={mean_a}, B={mean_b}"
        );
    }

    #[test]
    fn test_predict_returns_only_valid_labels() {
        let x = make_test_data();
        let mut model = LocalOutlierFactor::new(20);
        let preds = model.fit_predict_outliers(&x).unwrap();
        for &p in preds.iter() {
            assert!(p == 1 || p == -1, "Prediction must be +1 or -1, got {p}");
        }
    }

    #[test]
    fn test_high_dimensional() {
        let n = 50;
        let d = 15;
        let data: Vec<f64> = (0..n * d).map(|i| (i as f64 * 0.1).sin()).collect();
        let x = Array2::from_shape_vec((n, d), data).unwrap();

        let mut model = LocalOutlierFactor::new(10);
        let preds = model.fit_predict_outliers(&x).unwrap();
        assert_eq!(preds.len(), n);
    }

    #[test]
    fn test_novelty_score_vs_nof() {
        // In novelty mode, score_samples on training-like data should be similar to NOF
        let x = make_test_data();
        let mut model = LocalOutlierFactor::new(20).with_novelty(true);
        model.fit_unsupervised(&x).unwrap();

        // Score a point in the inlier cluster
        let x_in = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();
        let score = model.score_samples(&x_in).unwrap();
        // Should be near -1 (inlier)
        assert!(
            score[0] > -3.0,
            "Inlier score should be near -1, got {}",
            score[0]
        );
    }
}

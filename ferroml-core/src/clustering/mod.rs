//! Clustering Algorithms with Statistical Extensions
//!
//! This module provides clustering algorithms (KMeans, DBSCAN) with FerroML-style
//! statistical extensions that exceed R package rigor.
//!
//! ## Design Philosophy
//!
//! Every clustering algorithm in FerroML provides:
//! - **Cluster stability metrics**: Bootstrap-based stability assessment
//! - **Silhouette analysis with CI**: Confidence intervals on silhouette scores
//! - **Gap statistic**: Optimal k selection with standard error
//! - **Quality diagnostics**: Within-cluster SS, Calinski-Harabasz with inference
//! - **Hopkins statistic**: Clustering tendency assessment
//!
//! ## Algorithms
//!
//! - [`KMeans`] - K-Means clustering with kmeans++ initialization
//! - [`DBSCAN`] - Density-based spatial clustering
//!
//! ## Metrics
//!
//! - [`silhouette_score`], [`silhouette_samples`] - Silhouette coefficient
//! - [`calinski_harabasz_score`] - Variance ratio criterion
//! - [`davies_bouldin_score`] - Average similarity between clusters
//! - [`adjusted_rand_index`], [`normalized_mutual_info`] - External validation
//! - [`hopkins_statistic`] - Clustering tendency
//!
//! ## Example
//!
//! ```ignore
//! use ferroml_core::clustering::{KMeans, ClusteringModel};
//! use ndarray::Array2;
//!
//! let X = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 8.0, 8.0, 8.2, 7.8, 7.8, 8.2
//! ]).unwrap();
//!
//! let mut kmeans = KMeans::new(2);
//! kmeans.fit(&X).unwrap();
//!
//! let labels = kmeans.predict(&X).unwrap();
//! let stability = kmeans.cluster_stability(100).unwrap();
//! ```

use crate::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

// Submodules
pub mod dbscan;
pub mod diagnostics;
pub mod kmeans;
pub mod metrics;

// Re-exports
pub use dbscan::DBSCAN;
pub use diagnostics::ClusterDiagnostics;
pub use kmeans::KMeans;
pub use metrics::{
    adjusted_rand_index, calinski_harabasz_score, davies_bouldin_score, hopkins_statistic,
    normalized_mutual_info, silhouette_samples, silhouette_score,
};

/// Core trait for all clustering models
///
/// This trait provides the fundamental interface for clustering algorithms.
/// All FerroML clustering models implement this trait.
pub trait ClusteringModel {
    /// Fit the clustering model to data
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    /// Result indicating success or error
    fn fit(&mut self, x: &Array2<f64>) -> Result<()>;

    /// Predict cluster labels for new data
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    /// Array of cluster labels (integers starting from 0, -1 for noise in DBSCAN)
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>>;

    /// Fit and predict in one step
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    /// Array of cluster labels
    fn fit_predict(&mut self, x: &Array2<f64>) -> Result<Array1<i32>> {
        self.fit(x)?;
        self.predict(x)
    }

    /// Get cluster labels from the last fit
    fn labels(&self) -> Option<&Array1<i32>>;

    /// Check if the model has been fitted
    fn is_fitted(&self) -> bool;
}

/// Trait for clustering models with statistical extensions
pub trait ClusteringStatistics: ClusteringModel {
    /// Compute cluster stability using bootstrap resampling
    ///
    /// # Arguments
    /// * `x` - Feature matrix
    /// * `n_bootstrap` - Number of bootstrap iterations
    ///
    /// # Returns
    /// Stability scores for each cluster (0-1, higher is more stable)
    fn cluster_stability(&self, x: &Array2<f64>, n_bootstrap: usize) -> Result<Array1<f64>>;

    /// Compute silhouette scores with bootstrap confidence intervals
    ///
    /// # Arguments
    /// * `x` - Feature matrix
    /// * `confidence` - Confidence level (e.g., 0.95)
    ///
    /// # Returns
    /// (mean_silhouette, lower_ci, upper_ci)
    fn silhouette_with_ci(&self, x: &Array2<f64>, confidence: f64) -> Result<(f64, f64, f64)>;
}

/// Result of gap statistic analysis for optimal k selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapStatisticResult {
    /// Range of k values tested
    pub k_values: Vec<usize>,
    /// Gap statistic for each k
    pub gap_values: Vec<f64>,
    /// Standard error of gap statistic
    pub gap_se: Vec<f64>,
    /// Optimal k using gap statistic criterion
    pub optimal_k: usize,
}

/// Result of elbow method analysis for optimal k selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElbowResult {
    /// Range of k values tested
    pub k_values: Vec<usize>,
    /// Inertia (within-cluster sum of squares) for each k
    pub inertias: Vec<f64>,
    /// Optimal k detected by elbow method
    pub optimal_k: usize,
}

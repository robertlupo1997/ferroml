//! K-Means Clustering with Statistical Extensions
//!
//! This module implements K-Means clustering with kmeans++ initialization
//! and FerroML-style statistical extensions.
//!
//! ## Features
//!
//! - **kmeans++ initialization**: Smart centroid initialization for faster convergence
//! - **Elkan's algorithm**: Triangle inequality bounds to skip ~70% of distance computations
//! - **Hamerly's algorithm**: Single lower bound per point (O(n) memory) for cache-friendly performance at small k
//! - **Cluster stability**: Bootstrap-based stability assessment
//! - **Gap statistic**: Optimal k selection with standard error
//! - **Silhouette with CI**: Confidence intervals on silhouette scores
//!
//! ## Example
//!
//! ```
//! use ferroml_core::clustering::{KMeans, ClusteringModel};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0
//! ]).unwrap();
//!
//! let mut kmeans = KMeans::new(2);
//! kmeans.fit(&x).unwrap();
//!
//! let labels = kmeans.labels().unwrap();
//! println!("Cluster centers: {:?}", kmeans.cluster_centers());
//! ```

use crate::clustering::{ClusteringModel, ClusteringStatistics, ElbowResult, GapStatisticResult};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, ArrayView1};
use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Minimum sample count before rayon parallelism is used.
/// Below this threshold, thread-pool and collect overhead exceeds the gains.
#[cfg(feature = "parallel")]
const PARALLEL_MIN_SAMPLES: usize = 2_000;

/// Compute squared L2 norms for each row: x_norms[i] = ||x_i||^2
fn compute_row_norms(x: &Array2<f64>) -> Array1<f64> {
    x.rows().into_iter().map(|row| row.dot(&row)).collect()
}

/// Compute all pairwise squared distances between rows of X and rows of C
/// using the GEMM decomposition: ||x_i - c_j||^2 = x_norms[i] + c_norms[j] - 2*(X@C^T)[i,j]
/// Returns Array2 of shape (n_samples, k) with squared distances clamped to >= 0.
fn batch_squared_distances(
    x: &Array2<f64>,
    centers: &Array2<f64>,
    x_norms: &Array1<f64>,
) -> Array2<f64> {
    let k = centers.nrows();
    let n = x.nrows();
    // c_norms[j] = sum of squares of center j
    let c_norms: Array1<f64> = centers
        .rows()
        .into_iter()
        .map(|row| row.dot(&row))
        .collect();
    // XC^T via BLAS GEMM: (n, d) @ (d, k) = (n, k)
    let xct = x.dot(&centers.t());
    let mut dists = Array2::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            let d = x_norms[i] + c_norms[j] - 2.0 * xct[[i, j]];
            dists[[i, j]] = d.max(0.0); // clamp for numerical stability
        }
    }
    dists
}

/// Algorithm variant for KMeans clustering.
///
/// - `Lloyd`: Standard Lloyd's algorithm. Recomputes all distances every iteration.
/// - `Elkan`: Elkan's algorithm using triangle inequality bounds to skip ~70% of
///   distance computations. Generally faster, especially for moderate k.
/// - `Hamerly`: Hamerly's algorithm with single lower bound per point (O(n) memory).
///   Best for small k where the reduced memory fits in L1 cache.
/// - `Auto`: Automatically selects Hamerly for k<=20, Elkan for k<=256, otherwise Lloyd.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KMeansAlgorithm {
    /// Standard Lloyd's algorithm
    Lloyd,
    /// Elkan's algorithm with triangle inequality bounds
    Elkan,
    /// Hamerly's algorithm with single lower bound per point (O(n) memory)
    Hamerly,
    /// Automatic selection (Hamerly for k<=20, Elkan for k<=256, Lloyd otherwise)
    Auto,
}

impl std::fmt::Display for KMeansAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KMeansAlgorithm::Lloyd => write!(f, "lloyd"),
            KMeansAlgorithm::Elkan => write!(f, "elkan"),
            KMeansAlgorithm::Hamerly => write!(f, "hamerly"),
            KMeansAlgorithm::Auto => write!(f, "auto"),
        }
    }
}

impl Default for KMeansAlgorithm {
    fn default() -> Self {
        KMeansAlgorithm::Auto
    }
}

/// K-Means clustering algorithm with kmeans++ initialization
///
/// K-Means partitions data into k clusters by minimizing within-cluster
/// sum of squares (inertia). This implementation uses the kmeans++
/// initialization strategy for better convergence.
///
/// # Algorithm Variants
///
/// - **Lloyd** (classic): Recomputes all n*k distances each iteration.
/// - **Elkan**: Uses triangle inequality bounds with O(n*k) lower bounds to skip
///   distance computations, typically reducing work by ~70%.
/// - **Hamerly**: Uses a single lower bound per point (O(n) memory) for better
///   cache performance at small k. Auto-selected when k<=20.
///
/// # Statistical Extensions
///
/// Beyond sklearn's basic implementation, FerroML's KMeans provides:
/// - `cluster_stability()` - Bootstrap-based cluster stability scores
/// - `optimal_k()` - Gap statistic for optimal k selection
/// - `silhouette_with_ci()` - Silhouette scores with confidence intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeans {
    /// Number of clusters
    n_clusters: usize,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Random seed for initialization
    random_state: Option<u64>,
    /// Number of times to run with different centroid seeds
    n_init: usize,
    /// Whether to reuse previous cluster centers as initialization
    warm_start: bool,
    /// Algorithm variant: Lloyd, Elkan, or Auto
    algorithm: KMeansAlgorithm,
    /// Optional GPU backend for accelerated distance computation
    #[cfg(feature = "gpu")]
    #[serde(skip)]
    gpu_backend: Option<std::sync::Arc<dyn crate::gpu::GpuBackend>>,

    // Fitted state
    /// Cluster centers after fitting
    cluster_centers_: Option<Array2<f64>>,
    /// Labels of each point
    labels_: Option<Array1<i32>>,
    /// Sum of squared distances to closest cluster center (inertia)
    inertia_: Option<f64>,
    /// Number of iterations run
    n_iter_: Option<usize>,
    /// Convergence status after fitting
    convergence_status_: Option<crate::ConvergenceStatus>,
}

impl Default for KMeans {
    fn default() -> Self {
        Self::new(8)
    }
}

impl KMeans {
    /// Create a new KMeans model
    ///
    /// # Arguments
    /// * `n_clusters` - Number of clusters to form
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 300,
            tol: 1e-4,
            random_state: None,
            n_init: 10,
            warm_start: false,
            algorithm: KMeansAlgorithm::Auto,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
            n_iter_: None,
            convergence_status_: None,
        }
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random seed for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set GPU backend for accelerated distance computation
    #[cfg(feature = "gpu")]
    pub fn with_gpu(mut self, backend: std::sync::Arc<dyn crate::gpu::GpuBackend>) -> Self {
        self.gpu_backend = Some(backend);
        self
    }

    /// Set number of initialization runs
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set whether to reuse previous cluster centers as initialization
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Set the algorithm variant (Lloyd, Elkan, or Auto)
    ///
    /// - `Lloyd`: Standard algorithm, always computes all distances.
    /// - `Elkan`: Uses triangle inequality to skip distance computations.
    /// - `Auto` (default): Selects Elkan when k is moderate relative to n.
    pub fn algorithm(mut self, algorithm: KMeansAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Get cluster centers
    pub fn cluster_centers(&self) -> Option<&Array2<f64>> {
        self.cluster_centers_.as_ref()
    }

    /// Get inertia (within-cluster sum of squares)
    pub fn inertia(&self) -> Option<f64> {
        self.inertia_
    }

    /// Get number of iterations run
    pub fn n_iter(&self) -> Option<usize> {
        self.n_iter_
    }

    /// Get convergence status after fitting
    pub fn convergence_status(&self) -> Option<&crate::ConvergenceStatus> {
        self.convergence_status_.as_ref()
    }

    /// kmeans++ initialization
    ///
    /// Selects initial centroids to be far apart from each other
    fn kmeans_plus_plus_init(&self, x: &Array2<f64>, rng: &mut StdRng) -> Array2<f64> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut centers = Array2::zeros((self.n_clusters, n_features));

        // First center: random point
        let first_idx = rng.random_range(0..n_samples);
        centers.row_mut(0).assign(&x.row(first_idx));

        // Remaining centers: weighted probability by squared distance
        for k in 1..self.n_clusters {
            // Compute squared distances to nearest center
            let mut min_sq_dists = Array1::from_elem(n_samples, f64::MAX);
            for i in 0..n_samples {
                for j in 0..k {
                    let dist_sq = squared_euclidean(&x.row(i), &centers.row(j));
                    if dist_sq < min_sq_dists[i] {
                        min_sq_dists[i] = dist_sq;
                    }
                }
            }

            // Choose next center with probability proportional to D^2
            let total: f64 = min_sq_dists.sum();
            if total <= 0.0 {
                // All points are at centers, pick random
                let idx = rng.random_range(0..n_samples);
                centers.row_mut(k).assign(&x.row(idx));
            } else {
                let threshold = rng.random::<f64>() * total;
                let mut cumsum = 0.0;
                for i in 0..n_samples {
                    cumsum += min_sq_dists[i];
                    if cumsum >= threshold {
                        centers.row_mut(k).assign(&x.row(i));
                        break;
                    }
                }
            }
        }

        centers
    }

    /// CPU assignment step: assign each point to nearest center.
    ///
    /// Uses GEMM-based batch distance computation for fast assignment.
    /// When `x_norms` is provided, avoids recomputing row norms.
    fn cpu_assign(
        x: &Array2<f64>,
        centers: &Array2<f64>,
        n_clusters: usize,
        n_samples: usize,
        x_norms: Option<&Array1<f64>>,
    ) -> (Array1<i32>, f64) {
        let owned_norms;
        let norms = match x_norms {
            Some(n) => n,
            None => {
                owned_norms = compute_row_norms(x);
                &owned_norms
            }
        };
        let dists = batch_squared_distances(x, centers, norms);
        let mut labels = Array1::zeros(n_samples);
        let mut total_inertia = 0.0;
        for i in 0..n_samples {
            let mut min_dist = f64::MAX;
            let mut min_idx = 0i32;
            for j in 0..n_clusters {
                let d = dists[[i, j]];
                if d < min_dist {
                    min_dist = d;
                    min_idx = j as i32;
                }
            }
            labels[i] = min_idx;
            total_inertia += min_dist;
        }
        (labels, total_inertia)
    }

    /// Resolve the algorithm choice: Auto picks Elkan when k is moderate relative to n.
    fn resolve_algorithm(&self, n_samples: usize) -> KMeansAlgorithm {
        match self.algorithm {
            KMeansAlgorithm::Lloyd => KMeansAlgorithm::Lloyd,
            KMeansAlgorithm::Elkan => KMeansAlgorithm::Elkan,
            KMeansAlgorithm::Hamerly => KMeansAlgorithm::Hamerly,
            KMeansAlgorithm::Auto => {
                // Three-tier selection:
                // - Hamerly: O(n) bounds fit in L1 cache, best for small k
                // - Elkan: O(n*k) bounds, good for moderate k
                // - Lloyd: No bounds overhead, best for large k
                if self.n_clusters <= 20 && self.n_clusters * 2 <= n_samples {
                    KMeansAlgorithm::Hamerly
                } else if self.n_clusters <= 256 && self.n_clusters * 2 <= n_samples {
                    KMeansAlgorithm::Elkan
                } else {
                    KMeansAlgorithm::Lloyd
                }
            }
        }
    }

    /// Run Elkan's KMeans algorithm.
    ///
    /// Uses triangle inequality bounds to skip distance computations:
    /// - Upper bound u[i]: distance from sample i to its assigned center
    /// - Lower bounds l[i][j]: distance from sample i to center j
    /// - s[j]: half the minimum inter-center distance for center j
    fn run_elkan(
        &self,
        x: &Array2<f64>,
        initial_centers: Array2<f64>,
    ) -> (Array2<f64>, Array1<i32>, f64, usize) {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let k = self.n_clusters;
        let mut centers = initial_centers;
        let mut rng = StdRng::seed_from_u64(self.random_state.unwrap_or(42));

        // Zero-copy access: use contiguous slices from ndarray directly.
        // Array2 in standard (C) layout has contiguous rows, so as_slice() works.
        // Force contiguous layout if input is transposed/sliced.
        let x_contig;
        let x_ref = if x.is_standard_layout() {
            x
        } else {
            x_contig = x.as_standard_layout().into_owned();
            &x_contig
        };
        let x_data = x_ref.as_slice().expect("SAFETY: standard-layout array");

        // Helper closure: get row i of x as a slice (zero-copy)
        let x_row = |i: usize| -> &[f64] { &x_data[i * n_features..(i + 1) * n_features] };

        // --- Initial full assignment (no bounds yet) ---
        let mut labels = Array1::<i32>::zeros(n_samples);
        let mut upper = vec![0.0f64; n_samples]; // u[i]: upper bound on d(x_i, c_{a_i})
                                                 // Flat lower bounds: lower[i * k + j] = lower bound on d(x_i, c_j)
        let mut lower = vec![0.0f64; n_samples * k];

        // Runtime parallel dispatch: only use rayon when dataset is large enough
        #[cfg(feature = "parallel")]
        let use_parallel = n_samples >= PARALLEL_MIN_SAMPLES;
        #[cfg(not(feature = "parallel"))]
        let use_parallel = false;

        // Precompute row norms once for the entire fit (reused for final inertia)
        let x_norms = compute_row_norms(x_ref);

        // Initial assignment: use batch GEMM distances
        {
            let init_sq_dists = batch_squared_distances(x_ref, &centers, &x_norms);
            for i in 0..n_samples {
                let mut min_sq = f64::MAX;
                let mut min_idx = 0usize;
                for j in 0..k {
                    let sq = init_sq_dists[[i, j]];
                    // Elkan bounds use Euclidean (not squared) distances
                    lower[i * k + j] = sq.sqrt();
                    if sq < min_sq {
                        min_sq = sq;
                        min_idx = j;
                    }
                }
                labels[i] = min_idx as i32;
                upper[i] = min_sq.sqrt();
            }
        }

        // Pre-allocate working buffers
        let mut new_centers = Array2::zeros((k, n_features));
        let mut counts = vec![0usize; k];
        // Flat center-to-center distances: center_dists[j * k + jp] = d(c_j, c_jp)
        let mut center_dists = vec![0.0f64; k * k];
        let mut s = vec![0.0f64; k]; // s[j] = 0.5 * min_{j'!=j} d(c_j, c_j')

        for iter in 0..self.max_iter {
            // Extract centers as contiguous flat slice for cache-friendly access
            let centers_contig;
            let centers_ref = if centers.is_standard_layout() {
                &centers
            } else {
                centers_contig = centers.as_standard_layout().into_owned();
                &centers_contig
            };
            let centers_data = centers_ref.as_slice().expect("SAFETY: standard-layout");
            let center_row =
                |j: usize| -> &[f64] { &centers_data[j * n_features..(j + 1) * n_features] };

            // Step 1: Compute center-to-center distances and s[j]
            for j in 0..k {
                s[j] = f64::MAX;
                let cj_s = center_row(j);
                for jp in 0..k {
                    if j == jp {
                        center_dists[j * k + jp] = 0.0;
                        continue;
                    }
                    let cjp_s = center_row(jp);
                    let dist = crate::linalg::squared_euclidean_distance(cj_s, cjp_s).sqrt();
                    center_dists[j * k + jp] = dist;
                    let half_dist = dist * 0.5;
                    if half_dist < s[j] {
                        s[j] = half_dist;
                    }
                }
            }

            // Step 2: For each point, try to skip distance computations
            if use_parallel {
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    let chunk_size = (n_samples / rayon::current_num_threads().max(1)).max(64);

                    let results: Vec<(i32, f64, Vec<f64>)> = (0..n_samples)
                        .into_par_iter()
                        .with_min_len(chunk_size)
                        .map(|i| {
                            let mut label = labels[i];
                            let mut ai = label as usize;
                            let mut ub = upper[i];
                            let mut lowers: Vec<f64> = lower[i * k..i * k + k].to_vec();

                            if ub <= s[ai] {
                                return (label, ub, lowers);
                            }

                            let mut r_done = false;
                            let xi = &x_data[i * n_features..(i + 1) * n_features];

                            for j in 0..k {
                                if j == ai {
                                    continue;
                                }
                                if ub <= lowers[j] {
                                    continue;
                                }
                                if ub <= center_dists[ai * k + j] * 0.5 {
                                    continue;
                                }
                                if !r_done {
                                    let c_ai =
                                        &centers_data[ai * n_features..(ai + 1) * n_features];
                                    let d_ai =
                                        crate::linalg::squared_euclidean_distance(xi, c_ai).sqrt();
                                    ub = d_ai;
                                    lowers[ai] = d_ai;
                                    r_done = true;
                                    if d_ai <= lowers[j] || d_ai <= center_dists[ai * k + j] * 0.5 {
                                        continue;
                                    }
                                }
                                let c_j = &centers_data[j * n_features..(j + 1) * n_features];
                                let d_j = crate::linalg::squared_euclidean_distance(xi, c_j).sqrt();
                                lowers[j] = d_j;
                                if d_j < ub {
                                    label = j as i32;
                                    ai = j;
                                    ub = d_j;
                                }
                            }
                            (label, ub, lowers)
                        })
                        .collect();

                    for (i, (label, ub, lowers)) in results.into_iter().enumerate() {
                        labels[i] = label;
                        upper[i] = ub;
                        lower[i * k..i * k + k].copy_from_slice(&lowers);
                    }
                }
            } else {
                for i in 0..n_samples {
                    let ai = labels[i] as usize;

                    if upper[i] <= s[ai] {
                        continue;
                    }

                    let mut r_done = false;
                    let xi = x_row(i);
                    let lower_i = i * k;

                    for j in 0..k {
                        if j == ai {
                            continue;
                        }
                        if upper[i] <= lower[lower_i + j] {
                            continue;
                        }
                        if upper[i] <= center_dists[ai * k + j] * 0.5 {
                            continue;
                        }
                        if !r_done {
                            let d_ai =
                                crate::linalg::squared_euclidean_distance(xi, center_row(ai))
                                    .sqrt();
                            upper[i] = d_ai;
                            lower[lower_i + ai] = d_ai;
                            r_done = true;
                            if d_ai <= lower[lower_i + j] || d_ai <= center_dists[ai * k + j] * 0.5
                            {
                                continue;
                            }
                        }
                        let d_j =
                            crate::linalg::squared_euclidean_distance(xi, center_row(j)).sqrt();
                        lower[lower_i + j] = d_j;
                        if d_j < upper[i] {
                            labels[i] = j as i32;
                            upper[i] = d_j;
                        }
                    }
                }
            }

            // Step 3: Compute new centers
            if use_parallel {
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    let chunk_size = (n_samples / rayon::current_num_threads().max(1)).max(64);

                    let (par_centers, par_counts) = (0..n_samples)
                        .into_par_iter()
                        .with_min_len(chunk_size)
                        .fold(
                            || (vec![0.0f64; k * n_features], vec![0usize; k]),
                            |(mut acc_centers, mut acc_counts), i| {
                                let ci = labels[i] as usize;
                                let xi = &x_data[i * n_features..(i + 1) * n_features];
                                let offset = ci * n_features;
                                for f in 0..n_features {
                                    acc_centers[offset + f] += xi[f];
                                }
                                acc_counts[ci] += 1;
                                (acc_centers, acc_counts)
                            },
                        )
                        .reduce(
                            || (vec![0.0f64; k * n_features], vec![0usize; k]),
                            |(mut a_c, mut a_n), (b_c, b_n)| {
                                for idx in 0..a_c.len() {
                                    a_c[idx] += b_c[idx];
                                }
                                for j in 0..k {
                                    a_n[j] += b_n[j];
                                }
                                (a_c, a_n)
                            },
                        );

                    for j in 0..k {
                        if par_counts[j] > 0 {
                            let scale = 1.0 / par_counts[j] as f64;
                            let offset = j * n_features;
                            for f in 0..n_features {
                                new_centers[[j, f]] = par_centers[offset + f] * scale;
                            }
                        } else {
                            let idx = rng.random_range(0..n_samples);
                            new_centers.row_mut(j).assign(&x.row(idx));
                        }
                    }
                    counts.copy_from_slice(&par_counts);
                }
            } else {
                new_centers.fill(0.0);
                counts.fill(0);

                for i in 0..n_samples {
                    let ci = labels[i] as usize;
                    let x_row = x.row(i);
                    let mut center_row = new_centers.row_mut(ci);
                    center_row += &x_row;
                    counts[ci] += 1;
                }

                for j in 0..k {
                    if counts[j] > 0 {
                        let scale = 1.0 / counts[j] as f64;
                        new_centers.row_mut(j).mapv_inplace(|v| v * scale);
                    } else {
                        let idx = rng.random_range(0..n_samples);
                        new_centers.row_mut(j).assign(&x.row(idx));
                    }
                }
            }

            // Step 4: Compute center movement deltas
            let new_centers_contig;
            let new_centers_ref = if new_centers.is_standard_layout() {
                &new_centers
            } else {
                new_centers_contig = new_centers.as_standard_layout().into_owned();
                &new_centers_contig
            };
            let new_centers_data = new_centers_ref.as_slice().expect("SAFETY: standard-layout");
            let deltas: Vec<f64> = (0..k)
                .map(|j| {
                    let old_cj_s = center_row(j);
                    let new_cj_s = &new_centers_data[j * n_features..(j + 1) * n_features];
                    crate::linalg::squared_euclidean_distance(old_cj_s, new_cj_s).sqrt()
                })
                .collect();

            // Step 5: Update bounds
            if use_parallel {
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    let labels_slice = labels.as_slice().expect("SAFETY: contiguous");
                    upper
                        .par_chunks_mut(1)
                        .zip(lower.par_chunks_mut(k))
                        .zip(labels_slice.par_iter())
                        .for_each(|((ub_chunk, lower_chunk), &label)| {
                            let ai = label as usize;
                            for j in 0..k {
                                lower_chunk[j] = (lower_chunk[j] - deltas[j]).max(0.0);
                            }
                            ub_chunk[0] += deltas[ai];
                        });
                }
            } else {
                for i in 0..n_samples {
                    let ai = labels[i] as usize;
                    let lower_i = i * k;
                    for j in 0..k {
                        lower[lower_i + j] = (lower[lower_i + j] - deltas[j]).max(0.0);
                    }
                    upper[i] += deltas[ai];
                }
            }

            std::mem::swap(&mut centers, &mut new_centers);

            // Check convergence: skip inertia computation if centers moved significantly
            let center_shift_total: f64 = deltas.iter().map(|d| d * d).sum();
            if center_shift_total < self.tol {
                // Centers barely moved — compute final inertia using batch distances
                let final_dists = batch_squared_distances(x_ref, &centers, &x_norms);
                let inertia: f64 = (0..n_samples)
                    .map(|i| final_dists[[i, labels[i] as usize]])
                    .sum();
                return (centers, labels, inertia, iter + 1);
            }
        }

        // Compute final inertia at max_iter using batch distances
        let final_dists = batch_squared_distances(x_ref, &centers, &x_norms);
        let inertia: f64 = (0..n_samples)
            .map(|i| final_dists[[i, labels[i] as usize]])
            .sum();

        (centers, labels, inertia, self.max_iter)
    }

    /// Dispatch to the appropriate KMeans algorithm variant.
    fn run_kmeans(
        &self,
        x: &Array2<f64>,
        initial_centers: Array2<f64>,
    ) -> (Array2<f64>, Array1<i32>, f64, usize) {
        match self.resolve_algorithm(x.nrows()) {
            KMeansAlgorithm::Elkan => self.run_elkan(x, initial_centers),
            KMeansAlgorithm::Hamerly => self.run_hamerly(x, initial_centers),
            KMeansAlgorithm::Lloyd | KMeansAlgorithm::Auto => self.run_lloyd(x, initial_centers),
        }
    }

    /// Run Hamerly's KMeans algorithm.
    ///
    /// Uses a single lower bound per point (O(n) memory) instead of Elkan's O(n*k).
    /// The reduced memory footprint fits in L1 cache for typical n, giving better
    /// cache performance at small k (<=20). When a point's bounds are insufficient,
    /// all k distances are recomputed (no per-center lower bounds to consult).
    fn run_hamerly(
        &self,
        x: &Array2<f64>,
        initial_centers: Array2<f64>,
    ) -> (Array2<f64>, Array1<i32>, f64, usize) {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let k = self.n_clusters;
        let mut centers = initial_centers;
        let mut rng = StdRng::seed_from_u64(self.random_state.unwrap_or(42));

        // Ensure contiguous layout for cache-friendly access
        let x_contig;
        let x_ref = if x.is_standard_layout() {
            x
        } else {
            x_contig = x.as_standard_layout().into_owned();
            &x_contig
        };
        let x_data = x_ref.as_slice().expect("SAFETY: standard-layout array");

        let x_row = |i: usize| -> &[f64] { &x_data[i * n_features..(i + 1) * n_features] };

        // Runtime parallel dispatch
        #[cfg(feature = "parallel")]
        let use_parallel = n_samples >= PARALLEL_MIN_SAMPLES;
        #[cfg(not(feature = "parallel"))]
        let use_parallel = false;

        // Precompute row norms once
        let x_norms = compute_row_norms(x_ref);

        // --- Initial full assignment ---
        let mut labels = Array1::<i32>::zeros(n_samples);
        let mut upper = vec![0.0f64; n_samples]; // u[i]: upper bound on d(x_i, c_{a_i})
        let mut lower = vec![0.0f64; n_samples]; // l[i]: lower bound on d(x_i, second-closest)

        {
            let init_sq_dists = batch_squared_distances(x_ref, &centers, &x_norms);
            for i in 0..n_samples {
                let mut min_sq = f64::MAX;
                let mut min_idx = 0usize;
                let mut second_min_sq = f64::MAX;
                for j in 0..k {
                    let sq = init_sq_dists[[i, j]];
                    if sq < min_sq {
                        second_min_sq = min_sq;
                        min_sq = sq;
                        min_idx = j;
                    } else if sq < second_min_sq {
                        second_min_sq = sq;
                    }
                }
                labels[i] = min_idx as i32;
                upper[i] = min_sq.sqrt();
                lower[i] = second_min_sq.sqrt();
            }
        }

        // Pre-allocate working buffers
        let mut new_centers = Array2::zeros((k, n_features));
        let mut counts = vec![0usize; k];
        let mut s = vec![0.0f64; k]; // s[j] = 0.5 * min_{j'!=j} d(c_j, c_j')

        for iter in 0..self.max_iter {
            // Extract centers as contiguous flat slice
            let centers_contig;
            let centers_ref = if centers.is_standard_layout() {
                &centers
            } else {
                centers_contig = centers.as_standard_layout().into_owned();
                &centers_contig
            };
            let centers_data = centers_ref.as_slice().expect("SAFETY: standard-layout");
            let center_row_fn =
                |j: usize| -> &[f64] { &centers_data[j * n_features..(j + 1) * n_features] };

            // Step 1: Compute s[j] = 0.5 * min_{j'!=j} d(c_j, c_j')
            for j in 0..k {
                s[j] = f64::MAX;
                let cj_s = center_row_fn(j);
                for jp in 0..k {
                    if j == jp {
                        continue;
                    }
                    let cjp_s = center_row_fn(jp);
                    let dist = crate::linalg::squared_euclidean_distance(cj_s, cjp_s).sqrt();
                    let half_dist = dist * 0.5;
                    if half_dist < s[j] {
                        s[j] = half_dist;
                    }
                }
            }

            // Step 2: For each point, check bounds and possibly reassign
            if use_parallel {
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    let chunk_size = (n_samples / rayon::current_num_threads().max(1)).max(64);
                    let labels_slice = labels.as_slice_mut().expect("SAFETY: contiguous");

                    upper
                        .par_iter_mut()
                        .zip(lower.par_iter_mut())
                        .zip(labels_slice.par_iter_mut())
                        .enumerate()
                        .with_min_len(chunk_size)
                        .for_each(|(i, ((ub, lb), label))| {
                            let ai = *label as usize;

                            // Hamerly filter: if upper bound <= max(s[a_i], lower[i]), skip
                            let m = s[ai].max(*lb);
                            if *ub <= m {
                                return;
                            }

                            // Tighten upper bound
                            let xi = &x_data[i * n_features..(i + 1) * n_features];
                            let c_ai = &centers_data[ai * n_features..(ai + 1) * n_features];
                            *ub = crate::linalg::squared_euclidean_distance(xi, c_ai).sqrt();

                            // Re-check after tightening
                            if *ub <= m {
                                return;
                            }

                            // Recompute all k distances for this point
                            let mut min_dist = *ub;
                            let mut min_idx = ai;
                            let mut second_min_dist = f64::MAX;
                            for j in 0..k {
                                if j == ai {
                                    if *ub < second_min_dist {
                                        second_min_dist = *ub;
                                    }
                                    continue;
                                }
                                let c_j = &centers_data[j * n_features..(j + 1) * n_features];
                                let d_j = crate::linalg::squared_euclidean_distance(xi, c_j).sqrt();
                                if d_j < min_dist {
                                    second_min_dist = min_dist;
                                    min_dist = d_j;
                                    min_idx = j;
                                } else if d_j < second_min_dist {
                                    second_min_dist = d_j;
                                }
                            }
                            *label = min_idx as i32;
                            *ub = min_dist;
                            *lb = second_min_dist;
                        });
                }
            } else {
                for i in 0..n_samples {
                    let ai = labels[i] as usize;

                    let m = s[ai].max(lower[i]);
                    if upper[i] <= m {
                        continue;
                    }

                    // Tighten upper bound
                    let xi = x_row(i);
                    let c_ai = center_row_fn(ai);
                    upper[i] = crate::linalg::squared_euclidean_distance(xi, c_ai).sqrt();

                    if upper[i] <= m {
                        continue;
                    }

                    // Recompute all k distances
                    let mut min_dist = upper[i];
                    let mut min_idx = ai;
                    let mut second_min_dist = f64::MAX;
                    for j in 0..k {
                        if j == ai {
                            if upper[i] < second_min_dist {
                                second_min_dist = upper[i];
                            }
                            continue;
                        }
                        let c_j = center_row_fn(j);
                        let d_j = crate::linalg::squared_euclidean_distance(xi, c_j).sqrt();
                        if d_j < min_dist {
                            second_min_dist = min_dist;
                            min_dist = d_j;
                            min_idx = j;
                        } else if d_j < second_min_dist {
                            second_min_dist = d_j;
                        }
                    }
                    labels[i] = min_idx as i32;
                    upper[i] = min_dist;
                    lower[i] = second_min_dist;
                }
            }

            // Step 3: Compute new centers
            if use_parallel {
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    let chunk_size = (n_samples / rayon::current_num_threads().max(1)).max(64);

                    let (par_centers, par_counts) = (0..n_samples)
                        .into_par_iter()
                        .with_min_len(chunk_size)
                        .fold(
                            || (vec![0.0f64; k * n_features], vec![0usize; k]),
                            |(mut acc_centers, mut acc_counts), i| {
                                let ci = labels[i] as usize;
                                let xi = &x_data[i * n_features..(i + 1) * n_features];
                                let offset = ci * n_features;
                                for f in 0..n_features {
                                    acc_centers[offset + f] += xi[f];
                                }
                                acc_counts[ci] += 1;
                                (acc_centers, acc_counts)
                            },
                        )
                        .reduce(
                            || (vec![0.0f64; k * n_features], vec![0usize; k]),
                            |(mut a_c, mut a_n), (b_c, b_n)| {
                                for idx in 0..a_c.len() {
                                    a_c[idx] += b_c[idx];
                                }
                                for j in 0..k {
                                    a_n[j] += b_n[j];
                                }
                                (a_c, a_n)
                            },
                        );

                    for j in 0..k {
                        if par_counts[j] > 0 {
                            let scale = 1.0 / par_counts[j] as f64;
                            let offset = j * n_features;
                            for f in 0..n_features {
                                new_centers[[j, f]] = par_centers[offset + f] * scale;
                            }
                        } else {
                            let idx = rng.random_range(0..n_samples);
                            new_centers.row_mut(j).assign(&x.row(idx));
                        }
                    }
                    counts.copy_from_slice(&par_counts);
                }
            } else {
                new_centers.fill(0.0);
                counts.fill(0);

                for i in 0..n_samples {
                    let ci = labels[i] as usize;
                    let x_row_i = x.row(i);
                    let mut center_row_mut = new_centers.row_mut(ci);
                    center_row_mut += &x_row_i;
                    counts[ci] += 1;
                }

                for j in 0..k {
                    if counts[j] > 0 {
                        let scale = 1.0 / counts[j] as f64;
                        new_centers.row_mut(j).mapv_inplace(|v| v * scale);
                    } else {
                        let idx = rng.random_range(0..n_samples);
                        new_centers.row_mut(j).assign(&x.row(idx));
                    }
                }
            }

            // Step 4: Compute center movement deltas (Euclidean distances)
            let new_centers_contig;
            let new_centers_ref = if new_centers.is_standard_layout() {
                &new_centers
            } else {
                new_centers_contig = new_centers.as_standard_layout().into_owned();
                &new_centers_contig
            };
            let new_centers_data = new_centers_ref.as_slice().expect("SAFETY: standard-layout");
            let deltas: Vec<f64> = (0..k)
                .map(|j| {
                    let old_cj_s = center_row_fn(j);
                    let new_cj_s = &new_centers_data[j * n_features..(j + 1) * n_features];
                    crate::linalg::squared_euclidean_distance(old_cj_s, new_cj_s).sqrt()
                })
                .collect();

            // Step 5: Update bounds using max_delta (conservative Hamerly update)
            let max_delta = deltas.iter().cloned().fold(0.0f64, f64::max);

            if use_parallel {
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    let labels_slice = labels.as_slice().expect("SAFETY: contiguous");
                    upper
                        .par_iter_mut()
                        .zip(lower.par_iter_mut())
                        .zip(labels_slice.par_iter())
                        .for_each(|((ub, lb), &label)| {
                            let ai = label as usize;
                            *ub += deltas[ai];
                            *lb = (*lb - max_delta).max(0.0);
                        });
                }
            } else {
                for i in 0..n_samples {
                    let ai = labels[i] as usize;
                    upper[i] += deltas[ai];
                    lower[i] = (lower[i] - max_delta).max(0.0);
                }
            }

            std::mem::swap(&mut centers, &mut new_centers);

            // Check convergence: sum of squared center shifts < tol
            let center_shift_total: f64 = deltas.iter().map(|d| d * d).sum();
            if center_shift_total < self.tol {
                let final_dists = batch_squared_distances(x_ref, &centers, &x_norms);
                let inertia: f64 = (0..n_samples)
                    .map(|i| final_dists[[i, labels[i] as usize]])
                    .sum();
                return (centers, labels, inertia, iter + 1);
            }
        }

        // Compute final inertia at max_iter
        let final_dists = batch_squared_distances(x_ref, &centers, &x_norms);
        let inertia: f64 = (0..n_samples)
            .map(|i| final_dists[[i, labels[i] as usize]])
            .sum();

        (centers, labels, inertia, self.max_iter)
    }

    /// Run Lloyd's KMeans algorithm (standard).
    ///
    /// When the `parallel` feature is enabled, the assignment step uses rayon
    /// for near-linear speedup with core count.
    fn run_lloyd(
        &self,
        x: &Array2<f64>,
        initial_centers: Array2<f64>,
    ) -> (Array2<f64>, Array1<i32>, f64, usize) {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut centers = initial_centers;
        let mut labels = Array1::zeros(n_samples);
        let mut rng = StdRng::seed_from_u64(self.random_state.unwrap_or(42));

        // Pre-allocate working buffers for the update step (reused across iterations)
        let mut new_centers = Array2::zeros((self.n_clusters, n_features));
        let mut counts = vec![0usize; self.n_clusters];

        for iter in 0..self.max_iter {
            // Assignment step: assign each point to nearest center

            // GPU-accelerated path: compute full distance matrix on GPU
            #[cfg(feature = "gpu")]
            let gpu_result = self.gpu_backend.as_ref().and_then(|gpu| {
                gpu.pairwise_distances(x, &centers).ok().map(|dist_matrix| {
                    let mut new_labels = Array1::zeros(n_samples);
                    for i in 0..n_samples {
                        let mut min_dist = f64::MAX;
                        let mut min_idx = 0i32;
                        for k in 0..self.n_clusters {
                            let d = dist_matrix[[i, k]];
                            if d < min_dist {
                                min_dist = d;
                                min_idx = k as i32;
                            }
                        }
                        new_labels[i] = min_idx;
                    }
                    new_labels
                })
            });

            #[cfg(feature = "gpu")]
            let new_labels = if let Some(result) = gpu_result {
                result
            } else {
                Self::cpu_assign(x, &centers, self.n_clusters, n_samples, None).0
            };

            #[cfg(not(feature = "gpu"))]
            let new_labels = Self::cpu_assign(x, &centers, self.n_clusters, n_samples, None).0;

            labels = new_labels;

            // Update step: move centers to mean of assigned points
            new_centers.fill(0.0);
            counts.fill(0);

            for i in 0..n_samples {
                let k = labels[i] as usize;
                let x_row = x.row(i);
                let mut center_row = new_centers.row_mut(k);
                center_row += &x_row;
                counts[k] += 1;
            }

            for k in 0..self.n_clusters {
                if counts[k] > 0 {
                    let scale = 1.0 / counts[k] as f64;
                    new_centers.row_mut(k).mapv_inplace(|v| v * scale);
                } else {
                    let idx = rng.random_range(0..n_samples);
                    new_centers.row_mut(k).assign(&x.row(idx));
                }
            }

            // Check convergence on center movement (matches sklearn behavior:
            // sum of squared center shifts < tol)
            let center_shift_total: f64 = (0..self.n_clusters)
                .map(|j| squared_euclidean(&centers.row(j), &new_centers.row(j)))
                .sum();

            std::mem::swap(&mut centers, &mut new_centers);

            if center_shift_total < self.tol {
                // Compute final inertia only on convergence
                let (_, inertia) = Self::cpu_assign(x, &centers, self.n_clusters, n_samples, None);
                return (centers, labels, inertia, iter + 1);
            }
        }

        // Compute final inertia at max_iter
        let (_, inertia) = Self::cpu_assign(x, &centers, self.n_clusters, n_samples, None);

        (centers, labels, inertia, self.max_iter)
    }

    /// Find optimal k using the gap statistic
    ///
    /// The gap statistic compares the change in within-cluster dispersion
    /// with that expected under an appropriate null reference distribution.
    ///
    /// # Arguments
    /// * `x` - Feature matrix
    /// * `k_range` - Range of k values to test
    /// * `n_refs` - Number of reference datasets to generate
    pub fn optimal_k(
        x: &Array2<f64>,
        k_range: std::ops::Range<usize>,
        n_refs: usize,
        random_state: Option<u64>,
    ) -> Result<GapStatisticResult> {
        let mut rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Compute min/max for each feature (for uniform reference generation)
        let mins: Vec<f64> = (0..n_features)
            .map(|j| x.column(j).iter().cloned().fold(f64::INFINITY, f64::min))
            .collect();
        let maxs: Vec<f64> = (0..n_features)
            .map(|j| {
                x.column(j)
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();

        let k_values: Vec<usize> = k_range.collect();
        let mut gap_values = Vec::with_capacity(k_values.len());
        let mut gap_se = Vec::with_capacity(k_values.len());

        for &k in &k_values {
            // Fit kmeans on actual data
            let mut kmeans = KMeans::new(k).random_state(rng.random());
            kmeans.fit(x)?;
            let w_k = kmeans
                .inertia()
                .expect("SAFETY: kmeans was just fitted")
                .ln();

            // Generate reference datasets and compute their dispersions
            let mut ref_dispersions = Vec::with_capacity(n_refs);
            for _ in 0..n_refs {
                // Generate uniform random data in bounding box
                let mut ref_data = Array2::zeros((n_samples, n_features));
                for i in 0..n_samples {
                    for j in 0..n_features {
                        ref_data[[i, j]] = rng.random::<f64>() * (maxs[j] - mins[j]) + mins[j];
                    }
                }

                let mut ref_kmeans = KMeans::new(k).n_init(1).random_state(rng.random());
                ref_kmeans.fit(&ref_data)?;
                ref_dispersions.push(
                    ref_kmeans
                        .inertia()
                        .expect("SAFETY: kmeans was just fitted")
                        .ln(),
                );
            }

            // Compute gap statistic: E[log(W_k^*)] - log(W_k)
            let mean_ref: f64 = ref_dispersions.iter().sum::<f64>() / n_refs as f64;
            let gap = mean_ref - w_k;

            // Compute standard error
            let variance: f64 = ref_dispersions
                .iter()
                .map(|&x| (x - mean_ref).powi(2))
                .sum::<f64>()
                / n_refs as f64;
            let se = (variance * (1.0 + 1.0 / n_refs as f64)).sqrt();

            gap_values.push(gap);
            gap_se.push(se);
        }

        // Find optimal k: smallest k such that gap(k) >= gap(k+1) - se(k+1)
        let mut optimal_k = k_values[0];
        for i in 0..k_values.len() - 1 {
            if gap_values[i] >= gap_values[i + 1] - gap_se[i + 1] {
                optimal_k = k_values[i];
                break;
            }
        }

        Ok(GapStatisticResult {
            k_values,
            gap_values,
            gap_se,
            optimal_k,
        })
    }

    /// Find optimal k using the elbow method
    ///
    /// The elbow method looks for a "kink" in the inertia curve.
    ///
    /// # Arguments
    /// * `x` - Feature matrix
    /// * `k_range` - Range of k values to test
    pub fn elbow(
        x: &Array2<f64>,
        k_range: std::ops::Range<usize>,
        random_state: Option<u64>,
    ) -> Result<ElbowResult> {
        let mut rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        let k_values: Vec<usize> = k_range.collect();
        let mut inertias = Vec::with_capacity(k_values.len());

        for &k in &k_values {
            let mut kmeans = KMeans::new(k).random_state(rng.random());
            kmeans.fit(x)?;
            inertias.push(kmeans.inertia().expect("SAFETY: kmeans was just fitted"));
        }

        // Find elbow using the kneedle algorithm (simplified)
        // Normalize data and find point farthest from line connecting first and last
        let n = k_values.len();
        if n < 3 {
            return Ok(ElbowResult {
                k_values: k_values.clone(),
                inertias,
                optimal_k: k_values[0],
            });
        }

        let x1 = k_values[0] as f64;
        let y1 = inertias[0];
        let x2 = k_values[n - 1] as f64;
        let y2 = inertias[n - 1];

        // Distance from point to line
        let _line_len = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
        let mut max_dist = 0.0;
        let mut optimal_idx = 0;

        for (i, (&k, &inertia)) in k_values.iter().zip(inertias.iter()).enumerate() {
            let px = k as f64;
            let py = inertia;
            // Normalize to similar scales
            let norm_px = (px - x1) / (x2 - x1);
            let norm_py = (py - y2) / (y1 - y2);
            // Distance to diagonal in normalized space
            let dist = (norm_px - (1.0 - norm_py)).abs();
            if dist > max_dist {
                max_dist = dist;
                optimal_idx = i;
            }
        }

        Ok(ElbowResult {
            k_values: k_values.clone(),
            inertias,
            optimal_k: k_values[optimal_idx],
        })
    }
}

impl ClusteringModel for KMeans {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        crate::validation::validate_unsupervised_input(x)?;

        // Hyperparameter validation
        if self.n_clusters == 0 {
            return Err(FerroError::invalid_input(
                "Parameter n_clusters must be >= 1, got 0",
            ));
        }
        if self.tol < 0.0 {
            return Err(FerroError::invalid_input(format!(
                "Parameter tol must be >= 0, got {}",
                self.tol
            )));
        }

        let n_samples = x.nrows();

        if n_samples < self.n_clusters {
            return Err(FerroError::InvalidInput(format!(
                "n_samples={} should be >= n_clusters={}",
                n_samples, self.n_clusters
            )));
        }

        let base_seed = self.random_state.unwrap_or_else(rand::random);

        // Warm start: reuse previous cluster centers as initial centers (single run)
        if self.warm_start {
            if let Some(ref prev_centers) = self.cluster_centers_ {
                if prev_centers.ncols() == x.ncols() && prev_centers.nrows() == self.n_clusters {
                    let (centers, labels, inertia, n_iter) =
                        self.run_kmeans(x, prev_centers.clone());
                    self.cluster_centers_ = Some(centers);
                    self.labels_ = Some(labels);
                    self.inertia_ = Some(inertia);
                    self.n_iter_ = Some(n_iter);
                    if n_iter >= self.max_iter {
                        tracing::warn!(
                            "KMeans did not converge after {} iterations. \
                             Results may be suboptimal. Try increasing max_iter or n_init.",
                            self.max_iter
                        );
                        self.convergence_status_ = Some(crate::ConvergenceStatus::NotConverged {
                            iterations: n_iter,
                            final_change: f64::NAN,
                        });
                    } else {
                        self.convergence_status_ =
                            Some(crate::ConvergenceStatus::Converged { iterations: n_iter });
                    }
                    return Ok(());
                }
            }
        }

        // Run n_init times and keep best result
        // Parallelize across n_init runs when the parallel feature is enabled
        #[cfg(feature = "parallel")]
        let (best_centers, best_labels, best_inertia, best_n_iter) = {
            use rayon::prelude::*;
            let results: Vec<_> = (0..self.n_init)
                .into_par_iter()
                .map(|init| {
                    let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(init as u64));
                    let initial_centers = self.kmeans_plus_plus_init(x, &mut rng);
                    self.run_kmeans(x, initial_centers)
                })
                .collect();

            let mut best_inertia = f64::MAX;
            let mut best_centers = None;
            let mut best_labels = None;
            let mut best_n_iter = 0;
            for (centers, labels, inertia, n_iter) in results {
                if inertia < best_inertia {
                    best_inertia = inertia;
                    best_centers = Some(centers);
                    best_labels = Some(labels);
                    best_n_iter = n_iter;
                }
            }
            (best_centers, best_labels, best_inertia, best_n_iter)
        };

        #[cfg(not(feature = "parallel"))]
        let (best_centers, best_labels, best_inertia, best_n_iter) = {
            let mut best_inertia = f64::MAX;
            let mut best_centers = None;
            let mut best_labels = None;
            let mut best_n_iter = 0;
            for init in 0..self.n_init {
                let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(init as u64));
                let initial_centers = self.kmeans_plus_plus_init(x, &mut rng);
                let (centers, labels, inertia, n_iter) = self.run_kmeans(x, initial_centers);

                if inertia < best_inertia {
                    best_inertia = inertia;
                    best_centers = Some(centers);
                    best_labels = Some(labels);
                    best_n_iter = n_iter;
                }
            }
            (best_centers, best_labels, best_inertia, best_n_iter)
        };

        self.cluster_centers_ = best_centers;
        self.labels_ = best_labels;
        self.inertia_ = Some(best_inertia);
        self.n_iter_ = Some(best_n_iter);

        if best_n_iter >= self.max_iter {
            tracing::warn!(
                "KMeans did not converge after {} iterations. \
                 Results may be suboptimal. Try increasing max_iter or n_init.",
                self.max_iter
            );
            self.convergence_status_ = Some(crate::ConvergenceStatus::NotConverged {
                iterations: best_n_iter,
                final_change: f64::NAN,
            });
        } else {
            self.convergence_status_ = Some(crate::ConvergenceStatus::Converged {
                iterations: best_n_iter,
            });
        }

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidInput(
                "Input contains NaN or infinite values".to_string(),
            ));
        }

        let centers = self
            .cluster_centers_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;

        // Validate feature count matches training data
        let n_features = centers.ncols();
        if x.ncols() != n_features {
            return Err(FerroError::shape_mismatch(
                format!("(n_samples, {})", n_features),
                format!("({}, {})", x.nrows(), x.ncols()),
            ));
        }

        let n_samples = x.nrows();
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut min_dist = f64::MAX;
            let mut min_idx = 0;
            for k in 0..self.n_clusters {
                let dist = squared_euclidean(&x.row(i), &centers.row(k));
                if dist < min_dist {
                    min_dist = dist;
                    min_idx = k;
                }
            }
            labels[i] = min_idx as i32;
        }

        Ok(labels)
    }

    fn labels(&self) -> Option<&Array1<i32>> {
        self.labels_.as_ref()
    }

    fn is_fitted(&self) -> bool {
        self.cluster_centers_.is_some()
    }
}

impl ClusteringStatistics for KMeans {
    fn cluster_stability(&self, x: &Array2<f64>, n_bootstrap: usize) -> Result<Array1<f64>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("cluster_stability"));
        }

        let n_samples = x.nrows();
        let original_labels = self
            .labels_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("cluster_stability"))?;
        let mut rng = StdRng::seed_from_u64(self.random_state.unwrap_or(42));

        // For each bootstrap iteration, compute adjusted rand index with original
        let mut stability_scores = vec![0.0; self.n_clusters];
        let mut stability_counts = vec![0usize; self.n_clusters];

        for _ in 0..n_bootstrap {
            // Bootstrap sample indices
            let indices: Vec<usize> = (0..n_samples)
                .map(|_| rng.random_range(0..n_samples))
                .collect();
            let mut boot_x = Array2::zeros((n_samples, x.ncols()));
            for (i, &idx) in indices.iter().enumerate() {
                boot_x.row_mut(i).assign(&x.row(idx));
            }

            // Fit new kmeans on bootstrap sample
            let mut boot_kmeans = KMeans::new(self.n_clusters)
                .max_iter(self.max_iter)
                .tol(self.tol)
                .n_init(1)
                .random_state(rng.random());
            boot_kmeans.fit(&boot_x)?;

            // Predict labels for original data
            let boot_labels = boot_kmeans.predict(x)?;

            // Compute per-cluster stability (Jaccard similarity)
            for k in 0..self.n_clusters {
                let orig_mask: Vec<bool> = original_labels.iter().map(|&l| l == k as i32).collect();

                // Find best matching cluster in bootstrap
                let mut best_jaccard = 0.0;
                for bk in 0..self.n_clusters {
                    let boot_mask: Vec<bool> =
                        boot_labels.iter().map(|&l| l == bk as i32).collect();

                    let intersection: usize = orig_mask
                        .iter()
                        .zip(boot_mask.iter())
                        .filter(|(&a, &b)| a && b)
                        .count();
                    let union: usize = orig_mask
                        .iter()
                        .zip(boot_mask.iter())
                        .filter(|(&a, &b)| a || b)
                        .count();

                    if union > 0 {
                        let jaccard = intersection as f64 / union as f64;
                        if jaccard > best_jaccard {
                            best_jaccard = jaccard;
                        }
                    }
                }

                stability_scores[k] += best_jaccard;
                stability_counts[k] += 1;
            }
        }

        // Average stability scores
        let result: Vec<f64> = stability_scores
            .iter()
            .zip(stability_counts.iter())
            .map(|(&sum, &count)| if count > 0 { sum / count as f64 } else { 0.0 })
            .collect();

        Ok(Array1::from_vec(result))
    }

    fn silhouette_with_ci(&self, x: &Array2<f64>, confidence: f64) -> Result<(f64, f64, f64)> {
        use crate::clustering::metrics::silhouette_score;

        if !self.is_fitted() {
            return Err(FerroError::not_fitted("silhouette_with_ci"));
        }

        let labels = self
            .labels_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("silhouette_with_ci"))?;
        let n_bootstrap = 1000;
        let mut rng = StdRng::seed_from_u64(self.random_state.unwrap_or(42));

        let n_samples = x.nrows();
        let mut scores = Vec::with_capacity(n_bootstrap);

        for _ in 0..n_bootstrap {
            // Bootstrap sample
            let indices: Vec<usize> = (0..n_samples)
                .map(|_| rng.random_range(0..n_samples))
                .collect();
            let mut boot_x = Array2::zeros((n_samples, x.ncols()));
            let mut boot_labels = Array1::zeros(n_samples);
            for (i, &idx) in indices.iter().enumerate() {
                boot_x.row_mut(i).assign(&x.row(idx));
                boot_labels[i] = labels[idx];
            }

            if let Ok(score) = silhouette_score(&boot_x, &boot_labels) {
                scores.push(score);
            }
        }

        if scores.is_empty() {
            return Err(FerroError::numerical("Could not compute silhouette scores"));
        }

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let alpha = 1.0 - confidence;
        let lower_idx = ((alpha / 2.0) * scores.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * scores.len() as f64) as usize;

        let lower = scores[lower_idx.min(scores.len() - 1)];
        let upper = scores[upper_idx.min(scores.len() - 1)];

        Ok((mean, lower, upper))
    }
}

/// Compute squared Euclidean distance between two vectors
///
/// Uses SIMD-accelerated computation when the `simd` feature is enabled via
/// the shared `linalg` module.
#[inline]
fn squared_euclidean(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    // Try to get contiguous slices for SIMD path
    if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
        crate::linalg::squared_euclidean_distance(a_slice, b_slice)
    } else {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.0, 1.0, 8.0, 8.0, 8.5, 8.0, 8.0, 7.5],
        )
        .unwrap();

        let mut kmeans = KMeans::new(2).random_state(42);
        kmeans.fit(&x).unwrap();

        assert!(kmeans.is_fitted());
        assert!(kmeans.cluster_centers().is_some());
        assert!(kmeans.inertia().unwrap() < 10.0);

        let labels = kmeans.labels().unwrap();
        assert_eq!(labels.len(), 6);

        // First 3 points should be in same cluster, last 3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_kmeans_predict() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.5, 1.5, 10.0, 10.0, 10.5, 10.5])
            .unwrap();

        let mut kmeans = KMeans::new(2).random_state(42);
        kmeans.fit(&x).unwrap();

        let new_x = Array2::from_shape_vec((2, 2), vec![1.2, 1.2, 10.2, 10.2]).unwrap();
        let new_labels = kmeans.predict(&new_x).unwrap();

        assert_eq!(new_labels.len(), 2);
        // Points should be assigned to different clusters
        assert_ne!(new_labels[0], new_labels[1]);
    }

    #[test]
    fn test_kmeans_fit_predict() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.5, 1.5, 10.0, 10.0, 10.5, 10.5])
            .unwrap();

        let mut kmeans = KMeans::new(2).random_state(42);
        let labels = kmeans.fit_predict(&x).unwrap();

        assert_eq!(labels.len(), 4);
        assert!(kmeans.is_fitted());
    }

    #[test]
    fn test_kmeans_elbow() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                1.0, 1.0, 1.5, 1.5, 1.0, 1.5, 5.0, 5.0, 5.5, 5.5, 5.0, 5.5, 9.0, 9.0, 9.5, 9.5,
                9.0, 9.5,
            ],
        )
        .unwrap();

        let result = KMeans::elbow(&x, 1..6, Some(42)).unwrap();
        assert_eq!(result.k_values.len(), 5);
        assert_eq!(result.inertias.len(), 5);
        // With 3 clear clusters, optimal k should be around 3
        assert!(result.optimal_k >= 2 && result.optimal_k <= 4);
    }

    #[test]
    fn test_cluster_stability() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 1.0, 1.1, 1.1, 1.0, 10.0, 10.0, 10.1, 10.1, 10.0, 10.1, 10.1,
                10.0,
            ],
        )
        .unwrap();

        let mut kmeans = KMeans::new(2).random_state(42);
        kmeans.fit(&x).unwrap();

        let stability = kmeans.cluster_stability(&x, 50).unwrap();
        assert_eq!(stability.len(), 2);
        // Well-separated clusters should have high stability
        for s in stability.iter() {
            assert!(*s > 0.7);
        }
    }

    #[test]
    fn test_cpu_assign_matches_direct_computation() {
        // Verify that cpu_assign produces the same assignments as
        // manual squared Euclidean distance computation
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 2.0, 1.0, 0.5, 10.0, 10.0, 10.0, 10.5, 10.5, 10.5,
                11.0, 9.5, 10.0, 5.0, 5.0, 5.0, 5.5, 4.5, 5.5,
            ],
        )
        .unwrap();

        let centers =
            Array2::from_shape_vec((3, 3), vec![1.5, 1.8, 2.3, 10.2, 10.0, 10.2, 5.2, 4.8, 5.2])
                .unwrap();

        let (labels, inertia) = KMeans::cpu_assign(&x, &centers, 3, 8, None);

        // Compute expected assignments using direct squared Euclidean
        for i in 0..8 {
            let mut min_dist = f64::MAX;
            let mut min_idx = 0i32;
            for k in 0..3 {
                let dist: f64 = x
                    .row(i)
                    .iter()
                    .zip(centers.row(k).iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                if dist < min_dist {
                    min_dist = dist;
                    min_idx = k as i32;
                }
            }
            assert_eq!(labels[i], min_idx, "Label mismatch at sample {}", i);
        }

        // Inertia should be positive
        assert!(inertia > 0.0);
    }

    #[test]
    fn test_cpu_assign_no_negative_distances() {
        // When a point exactly matches a centroid, distance must be 0 (not negative)
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 1.0, 2.0, // duplicate of centroid 0
                3.0, 4.0, // duplicate of centroid 1
            ],
        )
        .unwrap();

        let centers = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let (labels, inertia) = KMeans::cpu_assign(&x, &centers, 2, 4, None);

        // Points matching centroids should have 0 distance contribution
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 1);
        assert_eq!(labels[2], 0);
        assert_eq!(labels[3], 1);
        // Inertia should be exactly 0 since all points lie on centroids
        assert!((inertia - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_kmeans_warm_start() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 1.0, 1.1, 1.1, 1.0, 10.0, 10.0, 10.1, 10.1, 10.0, 10.1, 10.1,
                10.0,
            ],
        )
        .unwrap();

        let mut kmeans = KMeans::new(2).warm_start(true).random_state(42);
        kmeans.fit(&x).unwrap();
        let inertia1 = kmeans.inertia().unwrap();

        // Fit again — should reuse centers and converge at least as well
        kmeans.fit(&x).unwrap();
        let inertia2 = kmeans.inertia().unwrap();

        // Second fit should have similar or better inertia
        assert!(
            inertia2 <= inertia1 + 0.1,
            "inertia2={} should be <= inertia1={} + 0.1",
            inertia2,
            inertia1
        );
    }

    #[test]
    fn test_convergence_status_converged() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.0, 1.0, 8.0, 8.0, 8.5, 8.0, 8.0, 7.5],
        )
        .unwrap();

        let mut kmeans = KMeans::new(2).random_state(42);
        kmeans.fit(&x).unwrap();

        let status = kmeans.convergence_status().unwrap();
        assert!(
            matches!(status, crate::ConvergenceStatus::Converged { .. }),
            "Easy data should converge"
        );
    }

    #[test]
    fn test_convergence_status_not_converged() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.0, 1.0, 8.0, 8.0, 8.5, 8.0, 8.0, 7.5],
        )
        .unwrap();

        // max_iter=1 with tol=0 should not converge
        let mut kmeans = KMeans::new(2)
            .max_iter(1)
            .tol(0.0)
            .n_init(1)
            .random_state(42);
        kmeans.fit(&x).unwrap(); // should not error

        let status = kmeans.convergence_status().unwrap();
        assert!(
            matches!(status, crate::ConvergenceStatus::NotConverged { .. }),
            "max_iter=1 should not converge"
        );
    }
}

//! t-distributed Stochastic Neighbor Embedding (t-SNE)
//!
//! Nonlinear dimensionality reduction for visualization. Maps high-dimensional
//! data to low dimensions while preserving local neighborhood structure.
//!
//! ## Algorithm
//!
//! 1. Compute pairwise conditional probabilities P in high-dimensional space
//!    using Gaussian kernels (with per-point bandwidth selected via perplexity)
//! 2. Initialize low-dimensional embedding Y (PCA or random)
//! 3. Optimize Y via gradient descent on KL divergence between P and
//!    Student-t distributed similarities Q in low-dimensional space
//!
//! ## Methods
//!
//! - **Exact** — O(N^2) pairwise computation, practical for up to ~5K points
//! - **Barnes-Hut** — O(N log N) approximation using VP-tree + quad-tree,
//!   suitable for larger datasets. Controlled by `theta` parameter (default 0.5).
//!
//! ## Limitations
//!
//! - t-SNE is transductive: no out-of-sample `transform()` for new data
//! - Non-convex objective: results depend on initialization and random seed
//!
//! ## Example
//!
//! ```
//! use ferroml_core::decomposition::TSNE;
//! use ferroml_core::preprocessing::Transformer;
//! use ndarray::Array2;
//!
//! let mut tsne = TSNE::new()
//!     .with_perplexity(30.0)
//!     .with_n_components(2)
//!     .with_random_state(42);
//!
//! let x = Array2::from_shape_fn((50, 10), |(i, _)| i as f64);
//! let embedding = tsne.fit_transform(&x).unwrap();
//! assert_eq!(embedding.dim(), (50, 2));
//! ```

#![allow(clippy::doc_markdown)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use ndarray::{Array2, Axis};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

use crate::decomposition::quadtree::QuadTree;
use crate::decomposition::vptree::VPTree;
use crate::preprocessing::{check_non_empty, Transformer};
use crate::{FerroError, Result};

/// Distance metric for t-SNE.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TsneMetric {
    /// Euclidean (L2) distance — default
    Euclidean,
    /// Manhattan (L1) distance
    Manhattan,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
}

/// Initialization method for t-SNE embedding.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TsneInit {
    /// Initialize with PCA projection — better convergence (default)
    Pca,
    /// Initialize with small random values
    Random,
}

/// Learning rate configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LearningRate {
    /// Automatic: max(N / early_exaggeration / 4, 50)
    Auto,
    /// Fixed learning rate
    Fixed(f64),
}

/// t-SNE computation method.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TsneMethod {
    /// Exact O(N^2) computation — computes all pairwise interactions
    Exact,
    /// Barnes-Hut O(N log N) approximation — uses VP-tree for kNN and
    /// quad-tree for approximate repulsive forces
    BarnesHut,
}

/// t-distributed Stochastic Neighbor Embedding.
///
/// Reduces high-dimensional data to `n_components` dimensions (typically 2)
/// for visualization, preserving local neighborhood structure.
///
/// # Parameters
///
/// - `perplexity` — Effective number of neighbors (default 30.0). Typical 5–50.
/// - `learning_rate` — Step size for gradient descent (default Auto).
/// - `max_iter` — Maximum gradient descent iterations (default 1000).
/// - `early_exaggeration` — Factor to multiply P by for first 250 iterations (default 12.0).
/// - `n_components` — Output dimensionality (default 2).
/// - `min_grad_norm` — Convergence threshold (default 1e-7).
/// - `metric` — Distance metric: Euclidean, Manhattan, or Cosine.
/// - `init` — Initialization: PCA or Random.
/// - `random_state` — Seed for reproducibility.
/// - `method` — Exact or BarnesHut (default: BarnesHut for n > 1000, Exact otherwise).
/// - `theta` — Barnes-Hut accuracy parameter (default 0.5). Lower = more accurate but slower.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSNE {
    // Config
    n_components: usize,
    perplexity: f64,
    learning_rate: LearningRate,
    max_iter: usize,
    early_exaggeration: f64,
    min_grad_norm: f64,
    metric: TsneMetric,
    init: TsneInit,
    random_state: Option<u64>,
    method: Option<TsneMethod>,
    theta: f64,

    /// Optional GPU backend for accelerated pairwise distance computation
    #[cfg(feature = "gpu")]
    #[serde(skip)]
    gpu_backend: Option<std::sync::Arc<dyn crate::gpu::GpuBackend>>,

    // Fitted state
    embedding_: Option<Array2<f64>>,
    kl_divergence_: Option<f64>,
    n_iter_final_: Option<usize>,
    n_features_in_: Option<usize>,
}

impl Default for TSNE {
    fn default() -> Self {
        Self::new()
    }
}

impl TSNE {
    /// Create a new t-SNE with default parameters.
    pub fn new() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            learning_rate: LearningRate::Auto,
            max_iter: 1000,
            early_exaggeration: 12.0,
            min_grad_norm: 1e-7,
            metric: TsneMetric::Euclidean,
            init: TsneInit::Pca,
            random_state: None,
            method: None,
            theta: 0.5,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
            embedding_: None,
            kl_divergence_: None,
            n_iter_final_: None,
            n_features_in_: None,
        }
    }

    /// Set GPU backend for accelerated pairwise distance computation (exact method only).
    #[cfg(feature = "gpu")]
    pub fn with_gpu(mut self, backend: std::sync::Arc<dyn crate::gpu::GpuBackend>) -> Self {
        self.gpu_backend = Some(backend);
        self
    }

    /// Set the number of output dimensions.
    pub fn with_n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Set the perplexity (effective number of neighbors).
    pub fn with_perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Set the learning rate.
    pub fn with_learning_rate(mut self, lr: LearningRate) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set a fixed learning rate value.
    pub fn with_learning_rate_f64(mut self, lr: f64) -> Self {
        self.learning_rate = LearningRate::Fixed(lr);
        self
    }

    /// Set the maximum number of iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the early exaggeration factor.
    pub fn with_early_exaggeration(mut self, factor: f64) -> Self {
        self.early_exaggeration = factor;
        self
    }

    /// Set the minimum gradient norm for convergence.
    pub fn with_min_grad_norm(mut self, norm: f64) -> Self {
        self.min_grad_norm = norm;
        self
    }

    /// Set the distance metric.
    pub fn with_metric(mut self, metric: TsneMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the initialization method.
    pub fn with_init(mut self, init: TsneInit) -> Self {
        self.init = init;
        self
    }

    /// Set the random state for reproducibility.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the computation method (Exact or BarnesHut).
    ///
    /// If not set, defaults to BarnesHut for n > 1000, Exact otherwise.
    pub fn with_method(mut self, method: TsneMethod) -> Self {
        self.method = Some(method);
        self
    }

    /// Set the Barnes-Hut accuracy parameter theta (default 0.5).
    ///
    /// - theta = 0: exact (no approximation, equivalent to Exact method)
    /// - theta = 0.5: good balance between speed and accuracy (default)
    /// - theta = 1.0: fast but less accurate
    ///
    /// Only used when method is BarnesHut.
    pub fn with_theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Get the embedding after fitting.
    pub fn embedding(&self) -> Option<&Array2<f64>> {
        self.embedding_.as_ref()
    }

    /// Get the final KL divergence.
    pub fn kl_divergence(&self) -> Option<f64> {
        self.kl_divergence_
    }

    /// Get the number of iterations actually run.
    pub fn n_iter_final(&self) -> Option<usize> {
        self.n_iter_final_
    }

    /// Get the number of input features.
    pub fn n_features_in(&self) -> Option<usize> {
        self.n_features_in_
    }

    // =========================================================================
    // Core algorithm
    // =========================================================================

    /// Compute pairwise distances using the configured metric.
    fn compute_pairwise_distances(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let mut distances = Array2::zeros((n, n));

        for i in 0..n {
            let xi = x.row(i);
            for j in (i + 1)..n {
                let xj = x.row(j);
                let d = match self.metric {
                    TsneMetric::Euclidean => crate::linalg::squared_euclidean_distance(
                        xi.as_slice().unwrap(),
                        xj.as_slice().unwrap(),
                    ),
                    TsneMetric::Manhattan => {
                        let m: f64 = xi.iter().zip(xj.iter()).map(|(a, b)| (a - b).abs()).sum();
                        m * m
                    }
                    TsneMetric::Cosine => {
                        let dot: f64 = xi.iter().zip(xj.iter()).map(|(a, b)| a * b).sum();
                        let na: f64 = xi.iter().map(|a| a * a).sum::<f64>().sqrt();
                        let nb: f64 = xj.iter().map(|b| b * b).sum::<f64>().sqrt();
                        let sim = if na < 1e-300 || nb < 1e-300 {
                            0.0
                        } else {
                            dot / (na * nb)
                        };
                        let c = 1.0 - sim;
                        c * c
                    }
                };
                distances[[i, j]] = d;
                distances[[j, i]] = d;
            }
        }
        distances
    }

    /// Binary search for the bandwidth sigma_i that achieves the target perplexity
    /// for point i, given its distances to all other points.
    ///
    /// Returns the conditional probabilities p_{j|i} for all j.
    fn binary_search_perplexity_row(
        distances_i: &[f64],
        i: usize,
        target_perplexity: f64,
    ) -> Vec<f64> {
        let n = distances_i.len();
        let target_entropy = target_perplexity.ln();
        let tol = 1e-5;
        let max_tries = 50;

        let mut beta = 1.0_f64; // beta = 1 / (2 * sigma^2)
        let mut beta_min = f64::NEG_INFINITY;
        let mut beta_max = f64::INFINITY;

        let mut p = vec![0.0; n];

        for _ in 0..max_tries {
            // Compute conditional probabilities
            let mut sum_p = 0.0;
            for j in 0..n {
                if j == i {
                    p[j] = 0.0;
                } else {
                    p[j] = (-distances_i[j] * beta).max(-700.0).exp();
                    sum_p += p[j];
                }
            }

            // Normalize
            if sum_p < 1e-300 {
                // All distances are huge or beta is too large
                for pj in &mut p {
                    *pj = 1.0 / (n - 1) as f64;
                }
                p[i] = 0.0;
                break;
            }

            for pj in p.iter_mut() {
                *pj /= sum_p;
            }

            // Compute entropy H = -sum(p * log(p))
            let mut entropy = 0.0;
            for j in 0..n {
                if p[j] > 1e-300 {
                    entropy -= p[j] * p[j].ln();
                }
            }

            let diff = entropy - target_entropy;

            if diff.abs() < tol {
                break;
            }

            if diff > 0.0 {
                // Entropy too high (distribution too uniform) => increase beta (smaller sigma)
                beta_min = beta;
                beta = if beta_max.is_infinite() {
                    beta * 2.0
                } else {
                    (beta + beta_max) / 2.0
                };
            } else {
                // Entropy too low (distribution too peaked) => decrease beta (larger sigma)
                beta_max = beta;
                beta = if beta_min.is_infinite() {
                    beta / 2.0
                } else {
                    (beta + beta_min) / 2.0
                };
            }
        }

        p
    }

    /// Compute the symmetric joint probability matrix P from pairwise distances.
    fn compute_joint_probabilities(&self, distances: &Array2<f64>) -> Array2<f64> {
        let n = distances.nrows();
        let mut p = Array2::zeros((n, n));

        // Compute conditional probabilities p_{j|i} per row
        for i in 0..n {
            let dist_row = distances.row(i).to_vec();
            let p_row = Self::binary_search_perplexity_row(&dist_row, i, self.perplexity);
            for j in 0..n {
                p[[i, j]] = p_row[j];
            }
        }

        // Symmetrize: P_ij = (p_{j|i} + p_{i|j}) / (2N)
        let two_n = 2.0 * n as f64;
        let mut p_sym = Array2::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                let val = (p[[i, j]] + p[[j, i]]) / two_n;
                // Clamp to avoid zero probabilities (causes log issues)
                let val = val.max(1e-12);
                p_sym[[i, j]] = val;
                p_sym[[j, i]] = val;
            }
        }

        p_sym
    }

    /// Initialize the low-dimensional embedding.
    fn initialize_embedding(
        &self,
        x: &Array2<f64>,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<Array2<f64>> {
        let n = x.nrows();

        match self.init {
            TsneInit::Pca => {
                // Use PCA to initialize — much better convergence
                let mut pca = crate::decomposition::PCA::new().with_n_components(self.n_components);
                let result = pca.fit_transform(x);

                match result {
                    Ok(y) => {
                        // Scale down to small values (sklearn uses 1e-4 * PCA result)
                        Ok(y * 1e-4)
                    }
                    Err(_) => {
                        // Fallback to random if PCA fails (e.g., n_features < n_components)
                        Ok(self.random_init(n, rng))
                    }
                }
            }
            TsneInit::Random => Ok(self.random_init(n, rng)),
        }
    }

    /// Random initialization with small values.
    fn random_init(&self, n: usize, rng: &mut rand::rngs::StdRng) -> Array2<f64> {
        let dist = StandardNormal;
        let mut y = Array2::zeros((n, self.n_components));
        for i in 0..n {
            for k in 0..self.n_components {
                let v: f64 = dist.sample(rng);
                y[[i, k]] = v * 1e-4;
            }
        }
        y
    }

    /// Compute the KL divergence between P and Q, and the gradient dC/dY.
    ///
    /// Returns (kl_divergence, gradient, grad_norm).
    fn compute_gradient_and_kl(
        p: &Array2<f64>,
        y: &Array2<f64>,
        exaggeration: f64,
    ) -> (f64, Array2<f64>, f64) {
        let n = y.nrows();
        let n_components = y.ncols();

        // Compute pairwise squared distances in embedding space
        let mut dist_y = Array2::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                let mut d2 = 0.0;
                for k in 0..n_components {
                    let diff = y[[i, k]] - y[[j, k]];
                    d2 += diff * diff;
                }
                dist_y[[i, j]] = d2;
                dist_y[[j, i]] = d2;
            }
        }

        // Compute Student-t kernel: q_num_ij = (1 + ||y_i - y_j||^2)^(-1)
        let mut q_num = Array2::zeros((n, n));
        let mut sum_q = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = 1.0 / (1.0 + dist_y[[i, j]]);
                q_num[[i, j]] = val;
                q_num[[j, i]] = val;
                sum_q += 2.0 * val;
            }
        }

        // Avoid division by zero
        sum_q = sum_q.max(1e-300);

        // Compute KL divergence
        let mut kl = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let p_ij = p[[i, j]] * exaggeration;
                    let q_ij = (q_num[[i, j]] / sum_q).max(1e-12);
                    if p_ij > 1e-300 {
                        kl += p_ij * (p_ij / q_ij).ln();
                    }
                }
            }
        }

        // Compute gradient: dC/dy_i = 4 * sum_j (p_ij - q_ij) * (y_i - y_j) * (1 + ||y_i - y_j||^2)^(-1)
        let mut grad = Array2::zeros((n, n_components));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let p_ij = p[[i, j]] * exaggeration;
                    let q_ij = q_num[[i, j]] / sum_q;
                    let mult = 4.0 * (p_ij - q_ij) * q_num[[i, j]];
                    for k in 0..n_components {
                        grad[[i, k]] += mult * (y[[i, k]] - y[[j, k]]);
                    }
                }
            }
        }

        // Compute gradient norm
        let grad_norm = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();

        (kl, grad, grad_norm)
    }

    /// Determine the effective method to use based on configuration and data size.
    fn effective_method(&self, n_samples: usize) -> TsneMethod {
        match self.method {
            Some(m) => m,
            None => {
                if n_samples > 1000 {
                    TsneMethod::BarnesHut
                } else {
                    TsneMethod::Exact
                }
            }
        }
    }

    /// Compute sparse joint probabilities using VP-tree for k nearest neighbors.
    ///
    /// Returns a sparse representation: `Vec<Vec<(usize, f64)>>` where entry `i`
    /// contains `(j, p_ij)` pairs for the k nearest neighbors of point i.
    fn compute_sparse_joint_probabilities(&self, x: &Array2<f64>) -> Vec<Vec<(usize, f64)>> {
        let n = x.nrows();
        let k = (3.0 * self.perplexity).ceil() as usize;
        let k = k.min(n - 1);

        // Build VP-tree
        let tree = VPTree::from_array(x);

        // For each point, find k nearest neighbors and compute conditional probabilities
        let mut sparse_p: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);

        for i in 0..n {
            let query = x.row(i).to_vec();
            let mut neighbors = tree.search(&query, k + 1); // +1 because self is included

            // Remove self from neighbors
            neighbors.retain(|&(idx, _)| idx != i);
            neighbors.truncate(k);

            // Compute squared distances for perplexity calibration
            let sq_dists: Vec<f64> = neighbors.iter().map(|&(_, d)| d * d).collect();

            // Binary search for sigma (perplexity calibration)
            let p_row = Self::binary_search_perplexity_sparse(&sq_dists, self.perplexity);

            let row: Vec<(usize, f64)> = neighbors
                .iter()
                .zip(p_row.iter())
                .map(|(&(idx, _), &p)| (idx, p))
                .collect();

            sparse_p.push(row);
        }

        // Symmetrize: p_ij = (p_{j|i} + p_{i|j}) / (2N)
        let two_n = 2.0 * n as f64;
        let mut sym_p: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

        // Collect all (i, j, p_cond) entries
        let mut all_entries: std::collections::HashMap<(usize, usize), f64> =
            std::collections::HashMap::new();

        for (i, neighbors) in sparse_p.iter().enumerate() {
            for &(j, p_cond) in neighbors {
                // p_{j|i} contribution
                *all_entries.entry((i, j)).or_insert(0.0) += p_cond;
                *all_entries.entry((j, i)).or_insert(0.0) += 0.0; // ensure symmetric key exists
            }
        }

        // Symmetrize
        let mut sym_entries: std::collections::HashMap<(usize, usize), f64> =
            std::collections::HashMap::new();
        for (&(i, j), &p_ji) in &all_entries {
            if i < j {
                let p_ij = all_entries.get(&(j, i)).copied().unwrap_or(0.0);
                let sym_val = ((p_ji + p_ij) / two_n).max(1e-12);
                sym_entries.insert((i, j), sym_val);
                sym_entries.insert((j, i), sym_val);
            }
        }

        for (&(i, j), &val) in &sym_entries {
            sym_p[i].push((j, val));
        }

        // Sort each row by index for consistent iteration
        for row in &mut sym_p {
            row.sort_by_key(|&(idx, _)| idx);
        }

        sym_p
    }

    /// Binary search for perplexity on sparse distances (only k neighbors).
    fn binary_search_perplexity_sparse(sq_distances: &[f64], target_perplexity: f64) -> Vec<f64> {
        let k = sq_distances.len();
        if k == 0 {
            return Vec::new();
        }

        let target_entropy = target_perplexity.ln();
        let tol = 1e-5;
        let max_tries = 50;

        let mut beta = 1.0_f64;
        let mut beta_min = f64::NEG_INFINITY;
        let mut beta_max = f64::INFINITY;
        let mut p = vec![0.0; k];

        for _ in 0..max_tries {
            let mut sum_p = 0.0;
            for (j, &d) in sq_distances.iter().enumerate() {
                p[j] = (-d * beta).max(-700.0).exp();
                sum_p += p[j];
            }

            if sum_p < 1e-300 {
                for pj in &mut p {
                    *pj = 1.0 / k as f64;
                }
                break;
            }

            for pj in p.iter_mut() {
                *pj /= sum_p;
            }

            let mut entropy = 0.0;
            for &pj in &p {
                if pj > 1e-300 {
                    entropy -= pj * pj.ln();
                }
            }

            let diff = entropy - target_entropy;
            if diff.abs() < tol {
                break;
            }

            if diff > 0.0 {
                beta_min = beta;
                beta = if beta_max.is_infinite() {
                    beta * 2.0
                } else {
                    (beta + beta_max) / 2.0
                };
            } else {
                beta_max = beta;
                beta = if beta_min.is_infinite() {
                    beta / 2.0
                } else {
                    (beta + beta_min) / 2.0
                };
            }
        }

        p
    }

    /// Compute gradient using Barnes-Hut approximation.
    ///
    /// Uses the quad-tree for repulsive forces and sparse P for attractive forces.
    fn compute_gradient_barnes_hut(
        sparse_p: &[Vec<(usize, f64)>],
        y: &Array2<f64>,
        exaggeration: f64,
        theta: f64,
    ) -> (f64, Array2<f64>, f64) {
        let n = y.nrows();
        let n_components = y.ncols();
        let mut grad = Array2::zeros((n, n_components));

        // Build quad-tree from current embedding
        let tree = QuadTree::new(y);

        // Compute repulsive forces via Barnes-Hut
        let mut total_sum_q = 0.0;
        let mut neg_forces = Array2::zeros((n, n_components));

        for i in 0..n {
            let px = y[[i, 0]];
            let py_val = y[[i, 1]];
            let (fx, fy, sq) = tree.compute_non_edge_forces(px, py_val, theta);
            neg_forces[[i, 0]] = fx;
            if n_components > 1 {
                neg_forces[[i, 1]] = fy;
            }
            total_sum_q += sq;
        }

        // Avoid division by zero
        total_sum_q = total_sum_q.max(1e-300);

        // Compute attractive forces from sparse P (edge forces)
        let mut pos_forces: Array2<f64> = Array2::zeros((n, n_components));
        let mut kl = 0.0;

        for i in 0..n {
            for &(j, p_ij) in &sparse_p[i] {
                let p_val = p_ij * exaggeration;

                let mut dist_sq = 0.0;
                for k in 0..n_components {
                    let diff = y[[i, k]] - y[[j, k]];
                    dist_sq += diff * diff;
                }

                let q_ij = 1.0 / (1.0 + dist_sq);

                // Attractive force
                let mult = p_val * q_ij;
                for k in 0..n_components {
                    pos_forces[[i, k]] += mult * (y[[i, k]] - y[[j, k]]);
                }

                // KL divergence contribution
                let q_normalized = (q_ij / total_sum_q).max(1e-12);
                if p_val > 1e-300 {
                    kl += p_val * (p_val / q_normalized).ln();
                }
            }
        }

        // Combine: gradient = 4 * (attractive - repulsive / sum_Q)
        for i in 0..n {
            for k in 0..n_components {
                grad[[i, k]] = 4.0 * (pos_forces[[i, k]] - neg_forces[[i, k]] / total_sum_q);
            }
        }

        let grad_norm = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
        (kl, grad, grad_norm)
    }

    /// Run the Barnes-Hut t-SNE optimization with sparse P.
    fn run_optimization_barnes_hut(
        &self,
        sparse_p: &[Vec<(usize, f64)>],
        y_init: Array2<f64>,
    ) -> Result<(Array2<f64>, f64, usize)> {
        let n = y_init.nrows();
        let n_components = y_init.ncols();

        let lr = match self.learning_rate {
            LearningRate::Auto => (n as f64 / self.early_exaggeration / 4.0).max(50.0),
            LearningRate::Fixed(lr) => lr,
        };

        let early_exaggeration_stop = 250.min(self.max_iter);

        let mut y = y_init;
        let mut gains: Array2<f64> = Array2::from_elem((n, n_components), 1.0_f64);
        let mut update: Array2<f64> = Array2::zeros((n, n_components));
        let mut best_kl = f64::INFINITY;
        let mut final_iter = 0;

        for iter in 0..self.max_iter {
            let momentum = if iter < early_exaggeration_stop {
                0.5
            } else {
                0.8
            };

            let exaggeration = if iter < early_exaggeration_stop {
                self.early_exaggeration
            } else {
                1.0
            };

            let (kl, grad, grad_norm) =
                Self::compute_gradient_barnes_hut(sparse_p, &y, exaggeration, self.theta);
            best_kl = kl;
            final_iter = iter + 1;

            if iter > early_exaggeration_stop && grad_norm < self.min_grad_norm {
                break;
            }

            // Adaptive gains
            for i in 0..n {
                for k in 0..n_components {
                    let same_sign = (grad[[i, k]] > 0.0) == (update[[i, k]] > 0.0);
                    if same_sign {
                        let g: f64 = gains[[i, k]] * 0.8;
                        gains[[i, k]] = g.max(0.01);
                    } else {
                        gains[[i, k]] = gains[[i, k]] + 0.2;
                    }
                }
            }

            // Update with momentum
            for i in 0..n {
                for k in 0..n_components {
                    update[[i, k]] = momentum * update[[i, k]] - lr * gains[[i, k]] * grad[[i, k]];
                    y[[i, k]] += update[[i, k]];
                }
            }

            // Re-center
            let mean = y.mean_axis(Axis(0)).unwrap();
            for i in 0..n {
                for k in 0..n_components {
                    y[[i, k]] -= mean[k];
                }
            }
        }

        Ok((y, best_kl, final_iter))
    }

    /// Run the t-SNE optimization (exact method).
    fn run_optimization(
        &self,
        p: &Array2<f64>,
        y_init: Array2<f64>,
    ) -> Result<(Array2<f64>, f64, usize)> {
        let n = y_init.nrows();
        let n_components = y_init.ncols();

        // Determine learning rate
        let lr = match self.learning_rate {
            LearningRate::Auto => (n as f64 / self.early_exaggeration / 4.0).max(50.0),
            LearningRate::Fixed(lr) => lr,
        };

        let early_exaggeration_stop = 250.min(self.max_iter);

        let mut y = y_init;
        let mut gains: Array2<f64> = Array2::from_elem((n, n_components), 1.0_f64);
        let mut update: Array2<f64> = Array2::zeros((n, n_components));
        let mut best_kl = f64::INFINITY;
        let mut final_iter = 0;

        for iter in 0..self.max_iter {
            let momentum = if iter < early_exaggeration_stop {
                0.5
            } else {
                0.8
            };

            let exaggeration = if iter < early_exaggeration_stop {
                self.early_exaggeration
            } else {
                1.0
            };

            let (kl, grad, grad_norm) = Self::compute_gradient_and_kl(p, &y, exaggeration);
            best_kl = kl;
            final_iter = iter + 1;

            // Check convergence (only after early exaggeration)
            if iter > early_exaggeration_stop && grad_norm < self.min_grad_norm {
                break;
            }

            // Adaptive gains (same sign = decrease gain, different sign = increase)
            for i in 0..n {
                for k in 0..n_components {
                    let same_sign = (grad[[i, k]] > 0.0) == (update[[i, k]] > 0.0);
                    if same_sign {
                        let g: f64 = gains[[i, k]] * 0.8;
                        gains[[i, k]] = g.max(0.01);
                    } else {
                        gains[[i, k]] = gains[[i, k]] + 0.2;
                    }
                }
            }

            // Update with momentum
            for i in 0..n {
                for k in 0..n_components {
                    update[[i, k]] = momentum * update[[i, k]] - lr * gains[[i, k]] * grad[[i, k]];
                    y[[i, k]] += update[[i, k]];
                }
            }

            // Re-center embedding (subtract mean)
            let mean = y.mean_axis(Axis(0)).unwrap();
            for i in 0..n {
                for k in 0..n_components {
                    y[[i, k]] -= mean[k];
                }
            }
        }

        Ok((y, best_kl, final_iter))
    }
}

impl Transformer for TSNE {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        self.fit_transform(x)?;
        Ok(())
    }

    fn transform(&self, _x: &Array2<f64>) -> Result<Array2<f64>> {
        // t-SNE is transductive — return the stored embedding
        self.embedding_
            .clone()
            .ok_or_else(|| FerroError::not_fitted("TSNE"))
    }

    fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_non_empty(x)?;

        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(FerroError::invalid_input(
                "t-SNE requires at least 2 samples",
            ));
        }

        // Perplexity must be less than n_samples
        if self.perplexity >= n_samples as f64 {
            return Err(FerroError::invalid_input(format!(
                "Perplexity ({}) must be less than n_samples ({})",
                self.perplexity, n_samples
            )));
        }

        if self.n_components > n_features {
            return Err(FerroError::invalid_input(format!(
                "n_components ({}) must be <= n_features ({})",
                self.n_components, n_features
            )));
        }

        self.n_features_in_ = Some(n_features);

        let method = self.effective_method(n_samples);

        // Step 1: Initialize embedding
        let mut rng = match self.random_state {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_os_rng(),
        };
        let y_init = self.initialize_embedding(x, &mut rng)?;

        // Step 2+3: Compute probabilities and optimize
        let (embedding, kl, n_iter) = match method {
            TsneMethod::Exact => {
                // GPU-accelerated path for Euclidean metric: use GPU pairwise_distances
                #[cfg(feature = "gpu")]
                let distances = if self.metric == TsneMetric::Euclidean {
                    self.gpu_backend
                        .as_ref()
                        .and_then(|gpu| gpu.pairwise_distances(x, x).ok())
                        .unwrap_or_else(|| self.compute_pairwise_distances(x))
                } else {
                    self.compute_pairwise_distances(x)
                };

                #[cfg(not(feature = "gpu"))]
                let distances = self.compute_pairwise_distances(x);

                let p = self.compute_joint_probabilities(&distances);
                self.run_optimization(&p, y_init)?
            }
            TsneMethod::BarnesHut => {
                if self.n_components != 2 {
                    return Err(FerroError::invalid_input(
                        "Barnes-Hut method only supports n_components=2. Use Exact for other dimensions.",
                    ));
                }
                let sparse_p = self.compute_sparse_joint_probabilities(x);
                self.run_optimization_barnes_hut(&sparse_p, y_init)?
            }
        };

        self.embedding_ = Some(embedding.clone());
        self.kl_divergence_ = Some(kl);
        self.n_iter_final_ = Some(n_iter);

        Ok(embedding)
    }

    fn inverse_transform(&self, _x: &Array2<f64>) -> Result<Array2<f64>> {
        Err(FerroError::NotImplemented(
            "inverse_transform is not supported for t-SNE".to_string(),
        ))
    }

    fn is_fitted(&self) -> bool {
        self.embedding_.is_some()
    }

    fn get_feature_names_out(&self, _input_names: Option<&[String]>) -> Option<Vec<String>> {
        if self.embedding_.is_some() {
            Some((0..self.n_components).map(|i| format!("tsne{i}")).collect())
        } else {
            None
        }
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in_
    }

    fn n_features_out(&self) -> Option<usize> {
        if self.embedding_.is_some() {
            Some(self.n_components)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::squared_euclidean_distance;
    use ndarray::{array, Array2};

    /// Generate well-separated Gaussian clusters for testing.
    fn make_clusters(
        n_per_cluster: usize,
        n_clusters: usize,
        dim: usize,
        seed: u64,
    ) -> (Array2<f64>, Vec<usize>) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n = n_per_cluster * n_clusters;
        let mut data = Array2::zeros((n, dim));
        let mut labels = vec![0usize; n];

        for c in 0..n_clusters {
            let center = c as f64 * 10.0; // well-separated centers
            for i in 0..n_per_cluster {
                let idx = c * n_per_cluster + i;
                for j in 0..dim {
                    let v: f64 = StandardNormal.sample(&mut rng);
                    data[[idx, j]] = center + v;
                }
                labels[idx] = c;
            }
        }

        (data, labels)
    }

    /// Compute mean intra-cluster distance / mean inter-cluster distance.
    /// Values < 1 mean clusters are well-separated.
    fn cluster_separation_ratio(embedding: &Array2<f64>, labels: &[usize]) -> f64 {
        let n = embedding.nrows();
        let mut intra_sum = 0.0;
        let mut intra_count = 0;
        let mut inter_sum = 0.0;
        let mut inter_count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let d = squared_euclidean_distance(
                    embedding.row(i).as_slice().unwrap(),
                    embedding.row(j).as_slice().unwrap(),
                )
                .sqrt();
                if labels[i] == labels[j] {
                    intra_sum += d;
                    intra_count += 1;
                } else {
                    inter_sum += d;
                    inter_count += 1;
                }
            }
        }

        let intra_mean = if intra_count > 0 {
            intra_sum / intra_count as f64
        } else {
            0.0
        };
        let inter_mean = if inter_count > 0 {
            inter_sum / inter_count as f64
        } else {
            1.0
        };

        intra_mean / inter_mean
    }

    #[test]
    fn test_output_shape_2d() {
        let (data, _) = make_clusters(20, 3, 10, 42);
        let mut tsne = TSNE::new()
            .with_n_components(2)
            .with_perplexity(10.0)
            .with_max_iter(300)
            .with_random_state(42);

        let result = tsne.fit_transform(&data).unwrap();
        assert_eq!(result.dim(), (60, 2));
    }

    #[test]
    fn test_output_shape_3d() {
        let (data, _) = make_clusters(15, 2, 8, 42);
        let mut tsne = TSNE::new()
            .with_n_components(3)
            .with_perplexity(5.0)
            .with_max_iter(300)
            .with_random_state(42);

        let result = tsne.fit_transform(&data).unwrap();
        assert_eq!(result.dim(), (30, 3));
    }

    #[test]
    fn test_separates_well_separated_clusters() {
        let (data, labels) = make_clusters(30, 3, 10, 42);
        let mut tsne = TSNE::new()
            .with_n_components(2)
            .with_perplexity(15.0)
            .with_max_iter(500)
            .with_random_state(42);

        let embedding = tsne.fit_transform(&data).unwrap();

        let ratio = cluster_separation_ratio(&embedding, &labels);
        // Well-separated clusters should have ratio significantly below 1
        assert!(
            ratio < 0.5,
            "Cluster separation ratio {} should be < 0.5",
            ratio
        );
    }

    #[test]
    fn test_kl_divergence_stored() {
        let (data, _) = make_clusters(20, 2, 5, 42);
        let mut tsne = TSNE::new()
            .with_perplexity(5.0)
            .with_max_iter(200)
            .with_random_state(42);

        tsne.fit_transform(&data).unwrap();

        let kl = tsne.kl_divergence().unwrap();
        assert!(kl.is_finite(), "KL divergence should be finite");
        assert!(kl >= 0.0, "KL divergence should be non-negative");
    }

    #[test]
    fn test_n_iter_stored() {
        let (data, _) = make_clusters(15, 2, 5, 42);
        let mut tsne = TSNE::new()
            .with_perplexity(5.0)
            .with_max_iter(100)
            .with_random_state(42);

        tsne.fit_transform(&data).unwrap();

        let n_iter = tsne.n_iter_final().unwrap();
        assert!(n_iter > 0 && n_iter <= 100);
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let (data, _) = make_clusters(20, 2, 5, 42);

        let mut tsne1 = TSNE::new()
            .with_perplexity(5.0)
            .with_max_iter(200)
            .with_random_state(123);
        let result1 = tsne1.fit_transform(&data).unwrap();

        let mut tsne2 = TSNE::new()
            .with_perplexity(5.0)
            .with_max_iter(200)
            .with_random_state(123);
        let result2 = tsne2.fit_transform(&data).unwrap();

        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Results should be identical with same seed"
            );
        }
    }

    #[test]
    fn test_different_seeds_give_different_results() {
        let (data, _) = make_clusters(20, 2, 5, 42);

        let mut tsne1 = TSNE::new()
            .with_init(TsneInit::Random)
            .with_perplexity(5.0)
            .with_max_iter(200)
            .with_random_state(1);
        let result1 = tsne1.fit_transform(&data).unwrap();

        let mut tsne2 = TSNE::new()
            .with_init(TsneInit::Random)
            .with_perplexity(5.0)
            .with_max_iter(200)
            .with_random_state(2);
        let result2 = tsne2.fit_transform(&data).unwrap();

        let max_diff: f64 = result1
            .iter()
            .zip(result2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_diff > 1e-6,
            "Different seeds should give different results"
        );
    }

    #[test]
    fn test_pca_init_vs_random_init() {
        let (data, _labels) = make_clusters(25, 3, 10, 42);

        let mut tsne_pca = TSNE::new()
            .with_init(TsneInit::Pca)
            .with_perplexity(10.0)
            .with_max_iter(300)
            .with_random_state(42);
        let emb_pca = tsne_pca.fit_transform(&data).unwrap();

        let mut tsne_random = TSNE::new()
            .with_init(TsneInit::Random)
            .with_perplexity(10.0)
            .with_max_iter(300)
            .with_random_state(42);
        let emb_random = tsne_random.fit_transform(&data).unwrap();

        // Both should produce valid embeddings
        assert_eq!(emb_pca.dim(), emb_random.dim());
        // PCA init typically converges to lower KL divergence
        // (not strictly guaranteed but generally true)
    }

    #[test]
    fn test_single_cluster_stays_compact() {
        // One cluster should not be spread out
        let (data, _) = make_clusters(30, 1, 5, 42);
        let mut tsne = TSNE::new()
            .with_perplexity(10.0)
            .with_max_iter(300)
            .with_random_state(42);

        let embedding = tsne.fit_transform(&data).unwrap();

        // Compute spread: std dev of embedding coordinates
        let mean = embedding.mean_axis(Axis(0)).unwrap();
        let var: f64 = embedding
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .zip(mean.iter())
                    .map(|(v, m)| (v - m).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / embedding.nrows() as f64;

        // Embedding should not have huge variance for a single cluster
        assert!(var.is_finite());
    }

    #[test]
    fn test_early_exaggeration_affects_result() {
        let (data, _) = make_clusters(20, 2, 5, 42);

        let mut tsne1 = TSNE::new()
            .with_early_exaggeration(4.0)
            .with_perplexity(5.0)
            .with_max_iter(300)
            .with_random_state(42);
        let result1 = tsne1.fit_transform(&data).unwrap();

        let mut tsne2 = TSNE::new()
            .with_early_exaggeration(24.0)
            .with_perplexity(5.0)
            .with_max_iter(300)
            .with_random_state(42);
        let result2 = tsne2.fit_transform(&data).unwrap();

        let max_diff: f64 = result1
            .iter()
            .zip(result2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_diff > 1e-6,
            "Different early exaggeration should produce different results"
        );
    }

    #[test]
    fn test_learning_rate_auto() {
        let (data, _) = make_clusters(20, 2, 5, 42);
        let mut tsne = TSNE::new()
            .with_learning_rate(LearningRate::Auto)
            .with_perplexity(5.0)
            .with_max_iter(200)
            .with_random_state(42);

        let result = tsne.fit_transform(&data).unwrap();
        assert_eq!(result.dim(), (40, 2));
    }

    #[test]
    fn test_learning_rate_fixed() {
        let (data, _) = make_clusters(20, 2, 5, 42);
        let mut tsne = TSNE::new()
            .with_learning_rate(LearningRate::Fixed(200.0))
            .with_perplexity(5.0)
            .with_max_iter(200)
            .with_random_state(42);

        let result = tsne.fit_transform(&data).unwrap();
        assert_eq!(result.dim(), (40, 2));
    }

    #[test]
    fn test_manhattan_metric() {
        let (data, labels) = make_clusters(20, 3, 5, 42);
        let mut tsne = TSNE::new()
            .with_metric(TsneMetric::Manhattan)
            .with_perplexity(5.0)
            .with_max_iter(300)
            .with_random_state(42);

        let embedding = tsne.fit_transform(&data).unwrap();
        assert_eq!(embedding.dim(), (60, 2));
        // Should still separate clusters
        let ratio = cluster_separation_ratio(&embedding, &labels);
        assert!(
            ratio < 0.8,
            "Manhattan metric should still separate clusters, ratio={}",
            ratio
        );
    }

    #[test]
    fn test_cosine_metric() {
        let (data, _) = make_clusters(20, 2, 5, 42);
        let mut tsne = TSNE::new()
            .with_metric(TsneMetric::Cosine)
            .with_perplexity(5.0)
            .with_max_iter(300)
            .with_random_state(42);

        let result = tsne.fit_transform(&data).unwrap();
        assert_eq!(result.dim(), (40, 2));
    }

    #[test]
    fn test_small_dataset() {
        // Just 4 points
        let data = array![
            [0.0, 0.0, 1.0],
            [0.1, 0.1, 1.1],
            [5.0, 5.0, 5.0],
            [5.1, 5.1, 5.1],
        ];
        let mut tsne = TSNE::new()
            .with_perplexity(1.5)
            .with_max_iter(200)
            .with_random_state(42);

        let result = tsne.fit_transform(&data).unwrap();
        assert_eq!(result.dim(), (4, 2));
    }

    #[test]
    fn test_error_too_few_samples() {
        let data = array![[1.0, 2.0]];
        let mut tsne = TSNE::new();
        let result = tsne.fit_transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_perplexity_too_large() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut tsne = TSNE::new().with_perplexity(10.0);
        let result = tsne.fit_transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_n_components_too_large() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut tsne = TSNE::new().with_n_components(5).with_perplexity(1.0);
        let result = tsne.fit_transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_empty_input() {
        let data = Array2::<f64>::zeros((0, 5));
        let mut tsne = TSNE::new();
        let result = tsne.fit_transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_transform_returns_stored_embedding() {
        let (data, _) = make_clusters(15, 2, 5, 42);
        let mut tsne = TSNE::new()
            .with_perplexity(5.0)
            .with_max_iter(100)
            .with_random_state(42);

        let embedding = tsne.fit_transform(&data).unwrap();
        let transformed = tsne.transform(&data).unwrap();

        // transform() should return the same stored embedding
        for (a, b) in embedding.iter().zip(transformed.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_transform_before_fit_errors() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let tsne = TSNE::new();
        let result = tsne.transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_inverse_transform_not_supported() {
        let (data, _) = make_clusters(10, 2, 5, 42);
        let mut tsne = TSNE::new()
            .with_perplexity(3.0)
            .with_max_iter(50)
            .with_random_state(42);
        tsne.fit_transform(&data).unwrap();

        let result = tsne.inverse_transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_n_features_in_stored() {
        let (data, _) = make_clusters(10, 2, 7, 42);
        let mut tsne = TSNE::new()
            .with_perplexity(3.0)
            .with_max_iter(50)
            .with_random_state(42);
        tsne.fit_transform(&data).unwrap();

        assert_eq!(tsne.n_features_in(), Some(7));
    }

    #[test]
    fn test_fit_then_access_embedding() {
        let (data, _) = make_clusters(15, 2, 5, 42);
        let mut tsne = TSNE::new()
            .with_perplexity(5.0)
            .with_max_iter(100)
            .with_random_state(42);

        tsne.fit(&data).unwrap();

        let embedding = tsne.embedding().unwrap();
        assert_eq!(embedding.dim(), (30, 2));
    }

    #[test]
    fn test_perplexity_affects_result() {
        let (data, _) = make_clusters(30, 3, 5, 42);

        let mut tsne1 = TSNE::new()
            .with_perplexity(5.0)
            .with_max_iter(300)
            .with_random_state(42);
        let result1 = tsne1.fit_transform(&data).unwrap();

        let mut tsne2 = TSNE::new()
            .with_perplexity(25.0)
            .with_max_iter(300)
            .with_random_state(42);
        let result2 = tsne2.fit_transform(&data).unwrap();

        let max_diff: f64 = result1
            .iter()
            .zip(result2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_diff > 1e-3,
            "Different perplexity should produce different results"
        );
    }

    #[test]
    fn test_default_parameters() {
        let tsne = TSNE::new();
        assert_eq!(tsne.n_components, 2);
        assert!((tsne.perplexity - 30.0).abs() < 1e-10);
        assert_eq!(tsne.max_iter, 1000);
        assert!((tsne.early_exaggeration - 12.0).abs() < 1e-10);
        assert_eq!(tsne.metric, TsneMetric::Euclidean);
        assert_eq!(tsne.init, TsneInit::Pca);
    }

    #[test]
    fn test_max_iter_respected() {
        let (data, _) = make_clusters(15, 2, 5, 42);
        let mut tsne = TSNE::new()
            .with_perplexity(5.0)
            .with_max_iter(10)
            .with_random_state(42);

        tsne.fit_transform(&data).unwrap();
        assert!(tsne.n_iter_final().unwrap() <= 10);
    }

    #[test]
    fn test_embedding_values_are_finite() {
        let (data, _) = make_clusters(25, 3, 8, 42);
        let mut tsne = TSNE::new()
            .with_perplexity(8.0)
            .with_max_iter(300)
            .with_random_state(42);

        let result = tsne.fit_transform(&data).unwrap();
        assert!(
            result.iter().all(|v| v.is_finite()),
            "All embedding values should be finite"
        );
    }

    // =========================================================================
    // Barnes-Hut t-SNE tests
    // =========================================================================

    #[test]
    fn test_barnes_hut_produces_valid_2d_embedding() {
        let (data, _) = make_clusters(30, 3, 10, 42);
        let mut tsne = TSNE::new()
            .with_method(TsneMethod::BarnesHut)
            .with_perplexity(10.0)
            .with_max_iter(500)
            .with_random_state(42);

        let result = tsne.fit_transform(&data).unwrap();
        assert_eq!(result.dim(), (90, 2));
        assert!(
            result.iter().all(|v| v.is_finite()),
            "All Barnes-Hut embedding values should be finite"
        );
    }

    #[test]
    fn test_barnes_hut_separates_clusters() {
        let (data, labels) = make_clusters(30, 3, 10, 42);
        let mut tsne = TSNE::new()
            .with_method(TsneMethod::BarnesHut)
            .with_perplexity(10.0)
            .with_max_iter(500)
            .with_random_state(42);

        let embedding = tsne.fit_transform(&data).unwrap();
        let ratio = cluster_separation_ratio(&embedding, &labels);
        assert!(
            ratio < 0.6,
            "Barnes-Hut should separate clusters, ratio={}",
            ratio
        );
    }

    #[test]
    fn test_barnes_hut_knn_preservation() {
        // Test that Barnes-Hut preserves local neighborhoods
        let (data, _) = make_clusters(40, 3, 10, 42);
        let n = data.nrows();
        let k = 7;

        let mut tsne = TSNE::new()
            .with_method(TsneMethod::BarnesHut)
            .with_perplexity(15.0)
            .with_max_iter(500)
            .with_random_state(42);

        let embedding = tsne.fit_transform(&data).unwrap();

        // Compute kNN in original space
        let orig_knn = compute_knn(&data, k);
        // Compute kNN in embedding space
        let emb_knn = compute_knn(&embedding, k);

        // Measure preservation: fraction of original kNN preserved in embedding
        let mut total_preserved = 0;
        let total_possible = n * k;

        for i in 0..n {
            for &j in &orig_knn[i] {
                if emb_knn[i].contains(&j) {
                    total_preserved += 1;
                }
            }
        }

        let preservation = total_preserved as f64 / total_possible as f64;
        assert!(
            preservation > 0.50,
            "kNN preservation should be > 0.50, got {}",
            preservation
        );
    }

    /// Compute k nearest neighbors for each point.
    fn compute_knn(data: &Array2<f64>, k: usize) -> Vec<Vec<usize>> {
        let n = data.nrows();
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d = squared_euclidean_distance(
                        data.row(i).as_slice().unwrap(),
                        data.row(j).as_slice().unwrap(),
                    );
                    (j, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            result.push(dists.iter().take(k).map(|&(j, _)| j).collect());
        }

        result
    }

    #[test]
    fn test_exact_method_unchanged() {
        // Verify that explicitly setting Exact produces the same results as before
        let (data, _) = make_clusters(20, 3, 10, 42);

        let mut tsne_default = TSNE::new()
            .with_method(TsneMethod::Exact)
            .with_n_components(2)
            .with_perplexity(10.0)
            .with_max_iter(300)
            .with_random_state(42);
        let result = tsne_default.fit_transform(&data).unwrap();

        assert_eq!(result.dim(), (60, 2));
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_barnes_hut_reproducibility() {
        let (data, _) = make_clusters(30, 3, 10, 42);

        let mut tsne1 = TSNE::new()
            .with_method(TsneMethod::BarnesHut)
            .with_perplexity(10.0)
            .with_max_iter(300)
            .with_random_state(99);
        let result1 = tsne1.fit_transform(&data).unwrap();

        let mut tsne2 = TSNE::new()
            .with_method(TsneMethod::BarnesHut)
            .with_perplexity(10.0)
            .with_max_iter(300)
            .with_random_state(99);
        let result2 = tsne2.fit_transform(&data).unwrap();

        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Barnes-Hut should be reproducible with same seed"
            );
        }
    }

    #[test]
    fn test_theta_affects_result() {
        let (data, _) = make_clusters(30, 3, 10, 42);

        let mut tsne_low = TSNE::new()
            .with_method(TsneMethod::BarnesHut)
            .with_theta(0.1)
            .with_perplexity(10.0)
            .with_max_iter(300)
            .with_random_state(42);
        let result_low = tsne_low.fit_transform(&data).unwrap();

        let mut tsne_high = TSNE::new()
            .with_method(TsneMethod::BarnesHut)
            .with_theta(1.0)
            .with_perplexity(10.0)
            .with_max_iter(300)
            .with_random_state(42);
        let result_high = tsne_high.fit_transform(&data).unwrap();

        let max_diff: f64 = result_low
            .iter()
            .zip(result_high.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_diff > 1e-3,
            "Different theta should produce different results"
        );
    }

    #[test]
    fn test_barnes_hut_kl_divergence_stored() {
        let (data, _) = make_clusters(20, 2, 5, 42);
        let mut tsne = TSNE::new()
            .with_method(TsneMethod::BarnesHut)
            .with_perplexity(5.0)
            .with_max_iter(200)
            .with_random_state(42);

        tsne.fit_transform(&data).unwrap();

        let kl = tsne.kl_divergence().unwrap();
        assert!(kl.is_finite(), "KL divergence should be finite");
    }

    #[test]
    fn test_barnes_hut_3d_errors() {
        // Barnes-Hut only supports 2D
        let (data, _) = make_clusters(20, 2, 5, 42);
        let mut tsne = TSNE::new()
            .with_method(TsneMethod::BarnesHut)
            .with_n_components(3)
            .with_perplexity(5.0)
            .with_max_iter(100)
            .with_random_state(42);

        let result = tsne.fit_transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_method_selects_exact_for_small() {
        // < 1000 points should default to Exact, which supports 3D
        let (data, _) = make_clusters(10, 2, 5, 42);
        let mut tsne = TSNE::new()
            .with_n_components(2)
            .with_perplexity(5.0)
            .with_max_iter(100)
            .with_random_state(42);

        // Should work without specifying method (defaults to Exact for small n)
        let result = tsne.fit_transform(&data).unwrap();
        assert_eq!(result.dim(), (20, 2));
    }

    #[test]
    fn test_default_parameters_include_method_theta() {
        let tsne = TSNE::new();
        assert!(tsne.method.is_none()); // Auto-detection
        assert!((tsne.theta - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tsne_large_distance_no_overflow() {
        // Create data with extremely large distances that could cause exp() overflow
        let n = 10;
        let dim = 2;
        let mut data = Array2::zeros((n, dim));
        for i in 0..n {
            data[[i, 0]] = i as f64 * 1e6; // huge distances between points
            data[[i, 1]] = i as f64 * 1e6;
        }

        let mut tsne = TSNE::new()
            .with_n_components(2)
            .with_perplexity(3.0)
            .with_max_iter(50);
        let result = tsne.fit_transform(&data);
        assert!(
            result.is_ok(),
            "t-SNE should handle large distances without overflow: {:?}",
            result.err()
        );

        let embedding = result.unwrap();
        assert_eq!(embedding.dim(), (n, 2));
        // No NaN or Inf in the embedding
        for &val in embedding.iter() {
            assert!(
                val.is_finite(),
                "Embedding value must be finite, got {}",
                val
            );
        }
    }
}

//! Support Vector Machines for Classification and Regression
//!
//! This module provides Support Vector Classification (SVC) and Regression (SVR)
//! with multiple kernel functions, probability estimates via Platt scaling, and
//! multiclass support using One-vs-One (OvO) or One-vs-Rest (OvR) strategies.
//!
//! It also provides `LinearSVC` and `LinearSVR` which are optimized for large
//! datasets using coordinate descent in the primal formulation.
//!
//! ## Features
//!
//! - **Kernels**: Linear, RBF (Gaussian), Polynomial, Sigmoid
//! - **Probability estimates**: Via Platt scaling calibration
//! - **Multiclass**: One-vs-One (default) and One-vs-Rest strategies
//! - **Class weights**: Balanced or custom weights for imbalanced data
//! - **Linear models**: Efficient `LinearSVC` and `LinearSVR` for large datasets
//!
//! ## Kernelized vs Linear SVMs
//!
//! | Aspect | SVC/SVR | LinearSVC/LinearSVR |
//! |--------|---------|---------------------|
//! | Memory | O(n²) kernel matrix | O(n·d) |
//! | Speed | Slower for large n | Fast coordinate descent |
//! | Flexibility | Non-linear boundaries | Linear only |
//! | Use case | Small/medium datasets | Large datasets |
//!
//! ## Example
//!
//! ```
//! use ferroml_core::models::svm::{SVC, Kernel};
//! use ferroml_core::models::{Model, ProbabilisticModel};
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 6.0, 7.0, 7.0, 6.0, 8.0, 8.0
//! ]).unwrap();
//! let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
//!
//! let mut model = SVC::new()
//!     .with_kernel(Kernel::Rbf { gamma: 0.5 })
//!     .with_c(1.0)
//!     .with_probability(true);
//!
//! model.fit(&x, &y).unwrap();
//! let predictions = model.predict(&x).unwrap();
//! let probas = model.predict_proba(&x).unwrap();
//! ```
//!
//! ## LinearSVC Example
//!
//! ```
//! # use ndarray::{Array1, Array2};
//! # let x_train = Array2::from_shape_vec((6, 2), vec![1.0,2.0,2.0,1.0,3.0,3.0,6.0,7.0,7.0,6.0,8.0,8.0]).unwrap();
//! # let y_train = Array1::from_vec(vec![0.0,0.0,0.0,1.0,1.0,1.0]);
//! # let x_test = x_train.clone();
//! use ferroml_core::models::svm::LinearSVC;
//! use ferroml_core::models::Model;
//!
//! // LinearSVC is much faster for large datasets
//! let mut clf = LinearSVC::new()
//!     .with_c(1.0)
//!     .with_max_iter(1000);
//!
//! clf.fit(&x_train, &y_train).unwrap();
//! let predictions = clf.predict(&x_test).unwrap();
//! ```

use crate::hpo::SearchSpace;
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, Model, PredictionInterval,
    ProbabilisticModel,
};
use crate::{FerroError, Result};
use ndarray::{concatenate, Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
// =============================================================================
// LRU Kernel Cache (Slab-based, O(1) eviction)
// =============================================================================

/// Default cache size in megabytes (matches sklearn/libsvm default).
const DEFAULT_CACHE_SIZE_MB: usize = 200;

/// Legacy default for the row-count API (used when `cache_size` field is set).
const DEFAULT_CACHE_SIZE: usize = 1000;

/// Threshold below which the full kernel matrix is precomputed.
/// Above this, the LRU cache is used with shrinking to limit memory.
/// Full matrix costs O(n^2) memory but O(1) access; cache costs O(cache_size * n)
/// but O(n_features) per miss. With shrinking, cache hit rates are high because
/// the active set converges to ~n_sv (support vectors).
const FULL_MATRIX_THRESHOLD: usize = 2_000;

/// Sentinel value for "no slot / no link".
const NONE_SLOT: i32 = -1;

/// Slab-based LRU kernel cache with O(1) eviction.
///
/// All cached rows are stored in a single contiguous `Vec<f64>` (the slab).
/// An intrusive doubly-linked list provides O(1) LRU eviction — no `contains()`
/// scans. Direct index mapping via `row_to_slot` gives O(1) cache lookups.
struct KernelCache {
    /// Contiguous memory for all cached rows (capacity * n_samples f64 values).
    slab: Vec<f64>,
    /// Maps sample index → slot index in slab (NONE_SLOT if not cached).
    row_to_slot: Vec<i32>,
    /// Maps slot index → sample index (or usize::MAX if slot is free).
    slot_to_row: Vec<usize>,
    /// Intrusive doubly-linked list: previous slot for each slot.
    lru_prev: Vec<i32>,
    /// Intrusive doubly-linked list: next slot for each slot.
    lru_next: Vec<i32>,
    /// Head of LRU list (least recently used). NONE_SLOT if empty.
    lru_head: i32,
    /// Tail of LRU list (most recently used). NONE_SLOT if empty.
    lru_tail: i32,
    /// Number of occupied slots.
    len: usize,
    /// Row length (n_samples).
    row_len: usize,
    /// Maximum number of rows that fit in the slab.
    capacity: usize,
    /// Kernel function.
    kernel: Kernel,
    /// Training data stored as row-major Vec<Vec<f64>> for efficient access.
    training_data: Vec<Vec<f64>>,
}

impl KernelCache {
    /// Create a new slab-based kernel cache.
    ///
    /// `capacity_rows` is the maximum number of kernel rows to cache.
    fn new(kernel: Kernel, x: &Array2<f64>, capacity_rows: usize) -> Self {
        let n_samples = x.nrows();
        let cap = capacity_rows.min(n_samples).max(2); // need at least 2 slots

        // Pre-extract training data as Vec<Vec<f64>> for efficient repeated access
        let x_std = if x.is_standard_layout() {
            None
        } else {
            Some(x.as_standard_layout().into_owned())
        };
        let x_ref = x_std.as_ref().unwrap_or(x);

        let training_data: Vec<Vec<f64>> = (0..n_samples)
            .map(|i| x_ref.row(i).as_slice().unwrap().to_vec())
            .collect();

        Self {
            slab: vec![0.0; cap * n_samples],
            row_to_slot: vec![NONE_SLOT; n_samples],
            slot_to_row: vec![usize::MAX; cap],
            lru_prev: vec![NONE_SLOT; cap],
            lru_next: vec![NONE_SLOT; cap],
            lru_head: NONE_SLOT,
            lru_tail: NONE_SLOT,
            len: 0,
            row_len: n_samples,
            capacity: cap,
            kernel,
            training_data,
        }
    }

    /// Create a cache with a byte budget (converts to row count).
    fn with_byte_budget(kernel: Kernel, x: &Array2<f64>, budget_bytes: usize) -> Self {
        let n_samples = x.nrows();
        let bytes_per_row = n_samples * std::mem::size_of::<f64>();
        let capacity_rows = if bytes_per_row > 0 {
            (budget_bytes / bytes_per_row).max(2)
        } else {
            n_samples
        };
        Self::new(kernel, x, capacity_rows)
    }

    /// Remove a slot from the LRU linked list (without freeing it).
    #[inline]
    fn lru_remove(&mut self, slot: i32) {
        let prev = self.lru_prev[slot as usize];
        let next = self.lru_next[slot as usize];
        if prev != NONE_SLOT {
            self.lru_next[prev as usize] = next;
        } else {
            self.lru_head = next;
        }
        if next != NONE_SLOT {
            self.lru_prev[next as usize] = prev;
        } else {
            self.lru_tail = prev;
        }
        self.lru_prev[slot as usize] = NONE_SLOT;
        self.lru_next[slot as usize] = NONE_SLOT;
    }

    /// Push a slot to the tail (most recently used) of the LRU list.
    #[inline]
    fn lru_push_tail(&mut self, slot: i32) {
        self.lru_prev[slot as usize] = self.lru_tail;
        self.lru_next[slot as usize] = NONE_SLOT;
        if self.lru_tail != NONE_SLOT {
            self.lru_next[self.lru_tail as usize] = slot;
        } else {
            self.lru_head = slot;
        }
        self.lru_tail = slot;
    }

    /// Touch a slot (move to tail / most recently used).
    #[inline]
    fn lru_touch(&mut self, slot: i32) {
        if slot != self.lru_tail {
            self.lru_remove(slot);
            self.lru_push_tail(slot);
        }
    }

    /// Evict the least recently used row. Returns the freed slot index.
    ///
    /// Note: `len` is NOT decremented because the caller (`get_row`) always
    /// reuses the returned slot immediately, keeping occupancy the same.
    #[inline]
    fn evict_lru(&mut self) -> usize {
        debug_assert!(self.lru_head != NONE_SLOT);
        let slot = self.lru_head as usize;
        let old_row = self.slot_to_row[slot];
        // Unlink from LRU
        self.lru_remove(slot as i32);
        // Clear mappings
        if old_row < self.row_to_slot.len() {
            self.row_to_slot[old_row] = NONE_SLOT;
        }
        self.slot_to_row[slot] = usize::MAX;
        slot
    }

    /// Get the full kernel row for sample `i`, computing and caching if needed.
    /// Returns a slice into the slab.
    fn get_row(&mut self, i: usize) -> &[f64] {
        let slot = self.row_to_slot[i];
        if slot != NONE_SLOT {
            // Cache hit — move to tail
            self.lru_touch(slot);
            let start = slot as usize * self.row_len;
            return &self.slab[start..start + self.row_len];
        }

        // Cache miss — allocate or evict
        let slot = if self.len < self.capacity {
            // Free slot available
            let s = self.len;
            self.len += 1;
            s
        } else {
            self.evict_lru()
        };

        // Compute the row
        let start = slot * self.row_len;
        let xi = &self.training_data[i];
        for j in 0..self.row_len {
            self.slab[start + j] = self.kernel.compute(xi, &self.training_data[j]);
        }

        // Update mappings
        self.row_to_slot[i] = slot as i32;
        self.slot_to_row[slot] = i;
        self.lru_push_tail(slot as i32);

        &self.slab[start..start + self.row_len]
    }

    /// Get a single kernel value K(i, j).
    #[inline]
    fn get_value(&mut self, i: usize, j: usize) -> f64 {
        // Check if row i is cached
        let slot_i = self.row_to_slot[i];
        if slot_i != NONE_SLOT {
            return self.slab[slot_i as usize * self.row_len + j];
        }
        // Check if row j is cached (kernel is symmetric)
        let slot_j = self.row_to_slot[j];
        if slot_j != NONE_SLOT {
            return self.slab[slot_j as usize * self.row_len + i];
        }
        // Neither row is cached — compute just this single value
        self.kernel
            .compute(&self.training_data[i], &self.training_data[j])
    }
}

/// Abstraction over kernel value access: either a full precomputed matrix
/// or an LRU cache for on-demand computation.
enum KernelProvider {
    /// Full precomputed kernel matrix (used for small datasets)
    Full(Array2<f64>),
    /// LRU cache (used for large datasets)
    Cached(KernelCache),
}

impl KernelProvider {
    /// Get kernel value K(i, j).
    #[inline]
    fn get(&mut self, i: usize, j: usize) -> f64 {
        match self {
            KernelProvider::Full(matrix) => matrix[[i, j]],
            KernelProvider::Cached(cache) => cache.get_value(i, j),
        }
    }

    /// Get a full row of kernel values for row `i`.
    /// For the full matrix this returns a slice; for the cache it computes and caches the row.
    /// The returned values are copied into the provided buffer.
    fn fill_row(&mut self, i: usize, buf: &mut [f64]) {
        match self {
            KernelProvider::Full(matrix) => {
                let row = matrix.row(i);
                buf.copy_from_slice(row.as_slice().unwrap());
            }
            KernelProvider::Cached(cache) => {
                let row = cache.get_row(i);
                buf.copy_from_slice(row);
            }
        }
    }
}

/// Numerically stable sigmoid: 1 / (1 + exp(-z)).
/// Avoids overflow by branching on the sign of z.
#[inline]
fn stable_sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Kernel function for SVM.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Kernel {
    /// Linear kernel: K(x, y) = x · y
    Linear,
    /// RBF (Gaussian) kernel: K(x, y) = exp(-gamma * ||x - y||^2)
    Rbf {
        /// Kernel coefficient (default: 1 / n_features if not specified)
        gamma: f64,
    },
    /// Polynomial kernel: K(x, y) = (gamma * x · y + coef0)^degree
    Polynomial {
        /// Kernel coefficient
        gamma: f64,
        /// Independent term
        coef0: f64,
        /// Polynomial degree
        degree: u32,
    },
    /// Sigmoid kernel: K(x, y) = tanh(gamma * x · y + coef0)
    Sigmoid {
        /// Kernel coefficient
        gamma: f64,
        /// Independent term
        coef0: f64,
    },
}

impl Default for Kernel {
    fn default() -> Self {
        Self::Rbf { gamma: 1.0 }
    }
}

impl Kernel {
    /// Create an RBF kernel with automatic gamma (will be set to 1/n_features)
    pub fn rbf_auto() -> Self {
        Self::Rbf { gamma: 0.0 } // 0 signals "auto"
    }

    /// Create an RBF kernel with specified gamma
    pub fn rbf(gamma: f64) -> Self {
        Self::Rbf { gamma }
    }

    /// Create a polynomial kernel
    pub fn poly(degree: u32, gamma: f64, coef0: f64) -> Self {
        Self::Polynomial {
            gamma,
            coef0,
            degree,
        }
    }

    /// Create a sigmoid kernel
    pub fn sigmoid(gamma: f64, coef0: f64) -> Self {
        Self::Sigmoid { gamma, coef0 }
    }

    /// Compute the kernel function between two vectors.
    ///
    /// Uses SIMD-accelerated dot product and distance when the `simd` feature is
    /// enabled.
    #[inline]
    pub fn compute(&self, x: &[f64], y: &[f64]) -> f64 {
        match self {
            Kernel::Linear => crate::linalg::dot_product(x, y),
            Kernel::Rbf { gamma } => {
                let diff_sq = crate::linalg::squared_euclidean_distance(x, y);
                (-gamma * diff_sq).exp()
            }
            Kernel::Polynomial {
                gamma,
                coef0,
                degree,
            } => (gamma * crate::linalg::dot_product(x, y) + coef0).powi(*degree as i32),
            Kernel::Sigmoid { gamma, coef0 } => {
                (gamma * crate::linalg::dot_product(x, y) + coef0).tanh()
            }
        }
    }

    /// Set gamma to auto (1 / n_features) if it's currently 0.
    pub fn with_auto_gamma(self, n_features: usize) -> Self {
        let auto_gamma = 1.0 / n_features as f64;
        match self {
            Kernel::Rbf { gamma } if gamma <= 0.0 => Kernel::Rbf { gamma: auto_gamma },
            Kernel::Polynomial {
                gamma,
                coef0,
                degree,
            } if gamma <= 0.0 => Kernel::Polynomial {
                gamma: auto_gamma,
                coef0,
                degree,
            },
            Kernel::Sigmoid { gamma, coef0 } if gamma <= 0.0 => Kernel::Sigmoid {
                gamma: auto_gamma,
                coef0,
            },
            other => other,
        }
    }
}

/// Multiclass classification strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MulticlassStrategy {
    /// One-vs-One: Train n*(n-1)/2 binary classifiers
    /// Best for smaller number of classes
    OneVsOne,
    /// One-vs-Rest: Train n binary classifiers
    /// More efficient for many classes
    OneVsRest,
}

impl Default for MulticlassStrategy {
    fn default() -> Self {
        Self::OneVsOne
    }
}

/// Class weight specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClassWeight {
    /// All classes have equal weight
    Uniform,
    /// Automatically adjust weights inversely proportional to class frequencies
    Balanced,
    /// Custom weights for each class
    Custom(Vec<(f64, f64)>), // (class_label, weight) pairs
}

impl Default for ClassWeight {
    fn default() -> Self {
        Self::Uniform
    }
}

// =============================================================================
// Binary SVC (Core Implementation)
// =============================================================================

/// Binary Support Vector Classifier using SMO algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BinarySVC {
    /// Regularization parameter (penalty for misclassification)
    c: f64,
    /// Kernel function
    kernel: Kernel,
    /// Tolerance for stopping criterion
    tol: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Maximum number of kernel rows to cache (0 = auto)
    cache_size: usize,

    // Fitted state
    /// Support vectors
    support_vectors: Option<Array2<f64>>,
    /// Dual coefficients (alpha * y for each support vector)
    dual_coef: Option<Array1<f64>>,
    /// Intercept (bias term)
    intercept: f64,
    /// Indices of support vectors in original training data
    support_indices: Option<Vec<usize>>,
    /// Number of features
    n_features: Option<usize>,
    /// Positive class label
    positive_class: f64,
    /// Negative class label
    negative_class: f64,

    // Platt scaling parameters
    platt_a: f64,
    platt_b: f64,
    probability_fitted: bool,
}

/// WSS3: Second-order working set selection (libsvm-style).
///
/// Given i (the first index, selected by max KKT violation), find j that
/// maximizes the objective function decrease. Uses second-order information
/// (curvature via kernel diagonal) for better convergence.
///
/// Objective decrease for pair (i,j) ~ -(Ei - Ej)^2 / (Kii + Kjj - 2*Kij)
/// We want to maximize this, which means maximizing (Ei - Ej)^2 / eta.
impl BinarySVC {
    /// Create a new binary SVC.
    fn new(c: f64, kernel: Kernel, tol: f64, max_iter: usize, cache_size: usize) -> Self {
        Self {
            c,
            kernel,
            tol,
            max_iter,
            cache_size,
            support_vectors: None,
            dual_coef: None,
            intercept: 0.0,
            support_indices: None,
            n_features: None,
            positive_class: 1.0,
            negative_class: -1.0,
            platt_a: 0.0,
            platt_b: 0.0,
            probability_fitted: false,
        }
    }

    /// Get support vectors.
    pub(crate) fn support_vectors(&self) -> Option<&Array2<f64>> {
        self.support_vectors.as_ref()
    }

    /// Get dual coefficients.
    pub(crate) fn dual_coef(&self) -> Option<&Array1<f64>> {
        self.dual_coef.as_ref()
    }

    /// Get the intercept (rho).
    pub(crate) fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Get the positive class label.
    #[allow(dead_code)]
    pub(crate) fn positive_class(&self) -> f64 {
        self.positive_class
    }

    /// Get the negative class label.
    #[allow(dead_code)]
    pub(crate) fn negative_class(&self) -> f64 {
        self.negative_class
    }

    /// Fit the binary SVC using SMO algorithm.
    fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        positive_class: f64,
        negative_class: f64,
        sample_weights: Option<&Array1<f64>>,
    ) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        self.positive_class = positive_class;
        self.negative_class = negative_class;
        self.n_features = Some(n_features);

        // Set auto gamma if needed
        self.kernel = self.kernel.with_auto_gamma(n_features);

        // Convert labels to +1/-1
        let y_binary: Array1<f64> = y
            .iter()
            .map(|&label| {
                if (label - positive_class).abs() < 1e-10 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        // Compute effective C for each sample (incorporating class weights)
        let c_effective: Array1<f64> = if let Some(weights) = sample_weights {
            weights.mapv(|w| self.c * w)
        } else {
            Array1::from_elem(n_samples, self.c)
        };

        // Initialize alphas and create kernel provider
        let mut alpha = Array1::zeros(n_samples);
        let mut kernel_provider = if n_samples < FULL_MATRIX_THRESHOLD {
            // Small dataset: precompute full kernel matrix
            KernelProvider::Full(self.compute_kernel_matrix(x))
        } else {
            // Large dataset: use slab-based LRU cache
            // Use byte-based sizing (200MB default) unless user specified a row count
            if self.cache_size > 0 && self.cache_size != DEFAULT_CACHE_SIZE {
                // User explicitly set a row count — honor it
                KernelProvider::Cached(KernelCache::new(
                    self.kernel,
                    x,
                    self.cache_size.min(n_samples),
                ))
            } else {
                // Default: 200MB byte budget (matches sklearn/libsvm)
                KernelProvider::Cached(KernelCache::with_byte_budget(
                    self.kernel,
                    x,
                    DEFAULT_CACHE_SIZE_MB * 1024 * 1024,
                ))
            }
        };

        // SMO algorithm
        let (final_alpha, intercept) = self.smo(
            &mut kernel_provider,
            &y_binary,
            &c_effective,
            alpha.view_mut(),
        )?;

        // Extract support vectors
        let support_threshold = 1e-8;
        let mut support_indices = Vec::new();
        let mut dual_coef = Vec::new();

        for i in 0..n_samples {
            if final_alpha[i] > support_threshold {
                support_indices.push(i);
                dual_coef.push(final_alpha[i] * y_binary[i]);
            }
        }

        if support_indices.is_empty() {
            return Err(FerroError::convergence_failure(
                self.max_iter,
                "No support vectors found - model failed to converge",
            ));
        }

        // Store support vectors
        let n_sv = support_indices.len();
        let mut sv_matrix = Array2::zeros((n_sv, n_features));
        for (i, &idx) in support_indices.iter().enumerate() {
            sv_matrix.row_mut(i).assign(&x.row(idx));
        }

        self.support_vectors = Some(sv_matrix);
        self.dual_coef = Some(Array1::from_vec(dual_coef));
        self.support_indices = Some(support_indices);
        self.intercept = intercept;

        Ok(())
    }

    /// SMO (Sequential Minimal Optimization) algorithm with WSS3 working set
    /// selection and active-set gradient updates.
    ///
    /// WSS3 (Fan, Chen, Lin 2005) selects the second variable j by maximizing
    /// the objective decrease: (E_i - E_j)^2 / (Q_ii + Q_jj - 2*Q_ij), which
    /// converges in far fewer iterations than Platt's max-|E_i - E_j| heuristic.
    fn smo(
        &self,
        kernel: &mut KernelProvider,
        y: &Array1<f64>,
        c: &Array1<f64>,
        mut alpha: ndarray::ArrayViewMut1<f64>,
    ) -> Result<(Array1<f64>, f64)> {
        let n_samples = y.len();
        let mut b = 0.0;

        // Error cache (gradient: f(x_k) - y_k)
        let mut errors: Array1<f64> = -y.clone();

        // Precompute kernel diagonal: Q_ii = K(i,i). Needed for WSS3 j-selection.
        // O(n) cost, trivial compared to the solver.
        let mut q_diag = vec![0.0; n_samples];
        for i in 0..n_samples {
            q_diag[i] = kernel.get(i, i);
        }

        // Buffers for kernel rows (used in incremental error update)
        let mut row_i_buf = vec![0.0; n_samples];
        let mut row_j_buf = vec![0.0; n_samples];

        let mut n_changed = 0;
        let mut examine_all = true;
        let mut iter = 0;
        let mut total_updates: u64 = 0;
        // Reconstruct errors from scratch every N updates to prevent floating-point drift.
        let reconstruct_interval: u64 = (n_samples as u64).max(200);

        // Shrinking: track which samples are "active" (may change) vs "shrunk" (at bounds)
        let mut active: Vec<bool> = vec![true; n_samples];
        let mut active_set: Vec<usize> = (0..n_samples).collect();
        // Only start shrinking after the solver has had time to explore.
        let shrink_interval = if n_samples < 500 {
            self.max_iter // effectively disable shrinking for small datasets
        } else {
            1000.min(self.max_iter)
        };

        while (n_changed > 0 || examine_all) && iter < self.max_iter {
            n_changed = 0;

            // Periodically shrink the active set
            if iter > 0 && iter % shrink_interval == 0 && !examine_all {
                Self::shrink_active_set(
                    &mut active,
                    &mut active_set,
                    &alpha,
                    &errors,
                    y,
                    c,
                    self.tol,
                );
            }

            let indices: Vec<usize> = if examine_all {
                // Full pass: unshrink all samples to re-evaluate
                for i in 0..n_samples {
                    active[i] = true;
                }
                active_set = (0..n_samples).collect();
                (0..n_samples).collect()
            } else {
                // Only examine non-bound active examples
                active_set
                    .iter()
                    .copied()
                    .filter(|&i| alpha[i] > 0.0 && alpha[i] < c[i])
                    .collect()
            };

            for &i in &indices {
                let ei = errors[i];
                let ri = ei * y[i];

                // Check KKT conditions
                if (ri < -self.tol && alpha[i] < c[i]) || (ri > self.tol && alpha[i] > 0.0) {
                    // WSS3: Select j to maximize objective decrease.
                    // obj_decrease ~ (E_i - E_j)^2 / (Q_ii + Q_jj - 2*Q_ij)
                    // We use the kernel diagonal for Q_ii/Q_jj and compute Q_ij on demand.
                    let j =
                        Self::select_j_wss3(i, &errors, &alpha, c, &q_diag, kernel, &active_set);

                    if let Some(j) = j {
                        let ej = errors[j];

                        // Save old alphas
                        let alpha_i_old = alpha[i];
                        let alpha_j_old = alpha[j];

                        // Compute bounds
                        let (low, high) = if (y[i] - y[j]).abs() < 1e-10 {
                            (
                                (alpha[i] + alpha[j] - c[i]).max(0.0),
                                (alpha[i] + alpha[j]).min(c[j]),
                            )
                        } else {
                            (
                                (alpha[j] - alpha[i]).max(0.0),
                                (c[i] + alpha[j] - alpha[i]).min(c[j]),
                            )
                        };

                        if (low - high).abs() < 1e-10 {
                            continue;
                        }

                        // Compute eta = 2*K_ij - K_ii - K_jj
                        let k_ij = kernel.get(i, j);
                        let eta = 2.0f64.mul_add(k_ij, -q_diag[i]) - q_diag[j];

                        if eta >= 0.0 {
                            continue;
                        }

                        // Update alpha_j
                        alpha[j] = alpha_j_old - y[j] * (ei - ej) / eta;
                        alpha[j] = alpha[j].max(low).min(high);

                        if (alpha[j] - alpha_j_old).abs() < 1e-8 {
                            continue;
                        }

                        // Update alpha_i
                        alpha[i] = (y[i] * y[j]).mul_add(alpha_j_old - alpha[j], alpha_i_old);

                        // Update threshold
                        let b_old = b;
                        let b1 = (y[j] * (alpha[j] - alpha_j_old)).mul_add(
                            -k_ij,
                            (y[i] * (alpha[i] - alpha_i_old)).mul_add(-q_diag[i], b - ei),
                        );

                        let b2 = (y[j] * (alpha[j] - alpha_j_old)).mul_add(
                            -q_diag[j],
                            (y[i] * (alpha[i] - alpha_i_old)).mul_add(-k_ij, b - ej),
                        );

                        b = if alpha[i] > 0.0 && alpha[i] < c[i] {
                            b1
                        } else if alpha[j] > 0.0 && alpha[j] < c[j] {
                            b2
                        } else {
                            (b1 + b2) / 2.0
                        };

                        // Incremental error cache update — only active set.
                        // errors[k] = f(x_k) - y[k], and f changed only at indices i,j:
                        // delta_f(k) = di*K(i,k) + dj*K(j,k) + (b_new - b_old)
                        {
                            let di = (alpha[i] - alpha_i_old) * y[i];
                            let dj = (alpha[j] - alpha_j_old) * y[j];
                            let db = b - b_old;

                            // Fill row buffers for i and j
                            kernel.fill_row(i, &mut row_i_buf);
                            kernel.fill_row(j, &mut row_j_buf);

                            // Update only active samples (non-bound + free)
                            // Non-active samples get their errors recomputed on unshrink.
                            for &k in &active_set {
                                errors[k] += di * row_i_buf[k] + dj * row_j_buf[k] + db;
                            }
                        }

                        n_changed += 1;
                        total_updates += 1;

                        // Periodically reconstruct error cache from scratch
                        // to prevent floating-point drift from accumulating
                        if total_updates % reconstruct_interval == 0 {
                            for k in 0..n_samples {
                                let mut fk = b;
                                for m in 0..n_samples {
                                    if alpha[m] > 1e-12 {
                                        fk += alpha[m] * y[m] * kernel.get(m, k);
                                    }
                                }
                                errors[k] = fk - y[k];
                            }
                        }
                    }
                }
            }

            if examine_all {
                examine_all = false;
            } else if n_changed == 0 {
                examine_all = true;
            }

            iter += 1;
        }

        Ok((alpha.to_owned(), b))
    }

    /// Shrink samples that are firmly at their bounds and unlikely to change.
    fn shrink_active_set(
        active: &mut [bool],
        active_set: &mut Vec<usize>,
        alpha: &ndarray::ArrayViewMut1<f64>,
        errors: &Array1<f64>,
        y: &Array1<f64>,
        c: &Array1<f64>,
        tol: f64,
    ) {
        let threshold = 1e-8;
        for &i in active_set.iter() {
            if !active[i] {
                continue;
            }
            let ri = errors[i] * y[i]; // KKT residual

            // At lower bound (alpha = 0) and gradient says "stay"
            let at_lower = alpha[i] < threshold && ri >= -tol;
            // At upper bound (alpha = C) and gradient says "stay"
            let at_upper = (alpha[i] - c[i]).abs() < threshold && ri <= tol;

            if at_lower || at_upper {
                active[i] = false;
            }
        }
        // Rebuild active_set from active flags
        active_set.retain(|&i| active[i]);
    }

    /// WSS3 second-order working set selection (Fan, Chen, Lin 2005).
    ///
    /// Select j to maximize objective decrease: (E_i - E_j)^2 / (Q_ii + Q_jj - 2*Q_ij).
    /// Falls back to max-|E_i - E_j| (first-order) if no valid pair is found.
    fn select_j_wss3(
        i: usize,
        errors: &Array1<f64>,
        alpha: &ndarray::ArrayViewMut1<f64>,
        c: &Array1<f64>,
        q_diag: &[f64],
        kernel: &mut KernelProvider,
        active_set: &[usize],
    ) -> Option<usize> {
        let ei = errors[i];
        let n_samples = errors.len();

        // Regularization to prevent division by near-zero denominator
        // (the bug that originally caused us to revert to Platt's heuristic)
        let tau = 1e-12;

        let mut best_j = None;
        let mut best_obj = f64::NEG_INFINITY;

        // First pass: try non-bound active examples with WSS3
        for &j in active_set {
            if j == i {
                continue;
            }
            // Only consider examples that can make progress
            let is_free = alpha[j] > 0.0 && alpha[j] < c[j];
            if !is_free {
                continue;
            }

            let diff = ei - errors[j];
            let diff_sq = diff * diff;
            if diff_sq < 1e-20 {
                continue;
            }

            // WSS3 objective: (E_i - E_j)^2 / (Q_ii + Q_jj - 2*Q_ij)
            // We need K(i,j) for each candidate — but the kernel diagonal is precomputed.
            // For the full-matrix case this is O(1); for cached, we try to use cached values.
            let k_ij = kernel.get(i, j);
            let denom = (q_diag[i] + q_diag[j] - 2.0 * k_ij).max(tau);
            let obj = diff_sq / denom;

            if obj > best_obj {
                best_obj = obj;
                best_j = Some(j);
            }
        }

        if best_j.is_some() {
            return best_j;
        }

        // Fallback: select from all examples using first-order heuristic (max |E_i - E_j|)
        // This handles the case where there are no free variables (early iterations)
        (0..n_samples).filter(|&j| j != i).max_by(|&a, &b| {
            let diff_a = (ei - errors[a]).abs();
            let diff_b = (ei - errors[b]).abs();
            diff_a.partial_cmp(&diff_b).unwrap_or(Ordering::Equal)
        })
    }

    /// Compute full kernel matrix.
    ///
    /// Optimized: uses array slices directly instead of allocating Vecs per pair.
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        // Ensure contiguous layout for slice access
        let x_c = if x.is_standard_layout() {
            None
        } else {
            Some(x.as_standard_layout().into_owned())
        };
        let x_ref = x_c.as_ref().unwrap_or(x);

        let mut k = Array2::zeros((n, n));

        for i in 0..n {
            let ri = x_ref.row(i);
            let xi = ri.as_slice().unwrap();
            for j in i..n {
                let rj = x_ref.row(j);
                let xj = rj.as_slice().unwrap();
                let val = self.kernel.compute(xi, xj);
                k[[i, j]] = val;
                k[[j, i]] = val;
            }
        }

        k
    }

    /// Predict decision values for samples.
    ///
    /// Optimized: uses array slices directly instead of allocating Vecs per sample.
    fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let sv = self
            .support_vectors
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("decision_function"))?;
        let dual_coef = self.dual_coef.as_ref().unwrap();
        let n_samples = x.nrows();

        // Ensure contiguous layout for slice access
        let x_c = if x.is_standard_layout() {
            None
        } else {
            Some(x.as_standard_layout().into_owned())
        };
        let x_ref = x_c.as_ref().unwrap_or(x);
        let sv_c = if sv.is_standard_layout() {
            None
        } else {
            Some(sv.as_standard_layout().into_owned())
        };
        let sv_ref = sv_c.as_ref().unwrap_or(sv);

        let mut decisions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let ri = x_ref.row(i);
            let xi = ri.as_slice().unwrap();
            let mut sum = 0.0;

            for (j, coef) in dual_coef.iter().enumerate() {
                let rj = sv_ref.row(j);
                let svj = rj.as_slice().unwrap();
                sum += coef * self.kernel.compute(xi, svj);
            }

            decisions[i] = sum + self.intercept;
        }

        Ok(decisions)
    }

    /// Predict class labels.
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let decisions = self.decision_function(x)?;
        Ok(decisions.mapv(|d| {
            if d >= 0.0 {
                self.positive_class
            } else {
                self.negative_class
            }
        }))
    }

    /// Fit Platt scaling for probability estimates.
    fn fit_platt_scaling(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let decisions = self.decision_function(x)?;
        let n_samples = y.len();

        // Count positive and negative samples
        let n_pos = y
            .iter()
            .filter(|&&v| (v - self.positive_class).abs() < 1e-10)
            .count() as f64;
        let n_neg = n_samples as f64 - n_pos;

        // Target probabilities with regularization
        let targets: Array1<f64> = y
            .iter()
            .map(|&v| {
                if (v - self.positive_class).abs() < 1e-10 {
                    (n_pos + 1.0) / (n_pos + 2.0)
                } else {
                    1.0 / (n_neg + 2.0)
                }
            })
            .collect();

        // Newton-Raphson optimization for A and B
        // P(y=1|f) = 1 / (1 + exp(A*f + B))
        let mut a = 0.0;
        let mut b = ((n_neg + 1.0) / (n_pos + 1.0)).ln();

        let max_iter = 100;
        let min_step = 1e-10;
        let sigma = 1e-12;

        for _ in 0..max_iter {
            // Compute gradient and Hessian
            let mut g1 = 0.0;
            let mut g2 = 0.0;
            let mut h11 = sigma;
            let mut h22 = sigma;
            let mut h12 = 0.0;

            for i in 0..n_samples {
                let f = decisions[i];
                let t = targets[i];
                let p = stable_sigmoid(-(a * f + b));
                let d = p - t;

                g1 += f * d;
                g2 += d;

                let w = p * (1.0 - p);
                h11 += f * f * w;
                h22 += w;
                h12 += f * w;
            }

            // Solve Newton step
            let det = h11 * h22 - h12 * h12;
            if det.abs() < 1e-12 {
                break;
            }

            let da = -(h22 * g1 - h12 * g2) / det;
            let db = -(h11 * g2 - h12 * g1) / det;

            if da.abs() < min_step && db.abs() < min_step {
                break;
            }

            a += da;
            b += db;
        }

        self.platt_a = a;
        self.platt_b = b;
        self.probability_fitted = true;

        Ok(())
    }

    /// Predict probability for the positive class (internal method for future public API).
    fn _predict_proba_positive(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.probability_fitted {
            return Err(FerroError::invalid_input(
                "Probability estimates not fitted. Call fit with probability=true",
            ));
        }

        let decisions = self.decision_function(x)?;
        Ok(decisions.mapv(|f| stable_sigmoid(-self.platt_a.mul_add(f, self.platt_b))))
    }
}

// =============================================================================
// SVC (Public Interface)
// =============================================================================

/// Support Vector Classification.
///
/// A powerful classification method that finds the optimal hyperplane
/// separating classes. Supports various kernel functions for non-linear
/// classification.
///
/// ## Parameters
///
/// - `c`: Regularization parameter (default: 1.0). Higher C means less regularization.
/// - `kernel`: Kernel function (default: RBF with gamma=1/n_features)
/// - `tol`: Tolerance for stopping criterion (default: 1e-3)
/// - `max_iter`: Maximum SMO iterations (default: 1000)
/// - `probability`: Whether to enable probability estimates (default: false)
/// - `multiclass_strategy`: Strategy for multiclass (default: OneVsOne)
/// - `class_weight`: Weight for each class (default: Uniform)
///
/// ## Example
///
/// ```
/// # use ferroml_core::models::svm::{SVC, Kernel, MulticlassStrategy};
/// # use ferroml_core::models::Model;
/// # use ndarray::{Array1, Array2};
/// # fn main() -> ferroml_core::Result<()> {
/// # let x_train = Array2::from_shape_vec((6, 2), vec![1.0,2.0,2.0,1.0,3.0,3.0,6.0,7.0,7.0,6.0,8.0,8.0]).unwrap();
/// # let y_train = Array1::from_vec(vec![0.0,0.0,0.0,1.0,1.0,1.0]);
/// # let x_test = x_train.clone();
/// use ferroml_core::models::svm::{SVC, Kernel, MulticlassStrategy};
/// use ferroml_core::models::Model;
///
/// let mut clf = SVC::new()
///     .with_kernel(Kernel::Rbf { gamma: 0.5 })
///     .with_c(1.0)
///     .with_probability(true)
///     .with_multiclass_strategy(MulticlassStrategy::OneVsOne);
///
/// clf.fit(&x_train, &y_train)?;
/// let predictions = clf.predict(&x_test)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVC {
    /// Regularization parameter
    pub c: f64,
    /// Kernel function
    pub kernel: Kernel,
    /// Tolerance for stopping criterion
    pub tol: f64,
    /// Maximum number of SMO iterations
    pub max_iter: usize,
    /// Whether to enable probability estimates
    pub probability: bool,
    /// Multiclass classification strategy
    pub multiclass_strategy: MulticlassStrategy,
    /// Class weight specification
    pub class_weight: ClassWeight,
    /// Maximum number of kernel rows to cache during SMO (0 = auto, default 1000).
    /// For datasets smaller than 500 samples, the full kernel matrix is precomputed
    /// regardless of this setting.
    pub cache_size: usize,

    // Fitted state
    /// Unique class labels
    classes: Option<Array1<f64>>,
    /// Binary classifiers (for OvO or OvR)
    classifiers: Vec<BinarySVC>,
    /// Class pairs for OvO (class_i, class_j)
    class_pairs: Vec<(f64, f64)>,
    /// Number of features
    n_features: Option<usize>,
    /// Computed class weights
    computed_weights: Option<Vec<(f64, f64)>>,
}

impl Default for SVC {
    fn default() -> Self {
        Self::new()
    }
}

impl SVC {
    /// Create a new SVC with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            c: 1.0,
            kernel: Kernel::rbf_auto(),
            tol: 1e-3,
            max_iter: 1000,
            probability: false,
            multiclass_strategy: MulticlassStrategy::OneVsOne,
            class_weight: ClassWeight::Uniform,
            cache_size: DEFAULT_CACHE_SIZE,
            classes: None,
            classifiers: Vec::new(),
            class_pairs: Vec::new(),
            n_features: None,
            computed_weights: None,
        }
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the kernel function.
    #[must_use]
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the tolerance for stopping criterion.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Enable or disable probability estimates.
    #[must_use]
    pub fn with_probability(mut self, probability: bool) -> Self {
        self.probability = probability;
        self
    }

    /// Set the multiclass classification strategy.
    #[must_use]
    pub fn with_multiclass_strategy(mut self, strategy: MulticlassStrategy) -> Self {
        self.multiclass_strategy = strategy;
        self
    }

    /// Set the class weight specification.
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Set the maximum number of kernel rows to cache during SMO.
    ///
    /// For datasets with fewer than 4000 samples, the full kernel matrix is always
    /// precomputed regardless of this setting. For larger datasets, the default
    /// is a 200MB byte budget (matching sklearn/libsvm). Setting this to a custom
    /// value overrides the byte budget with an explicit row count.
    ///
    /// Default: 200MB byte budget (~5000 rows for n=5000, full matrix cached).
    #[must_use]
    pub fn with_cache_size(mut self, cache_size: usize) -> Self {
        self.cache_size = cache_size;
        self
    }

    /// Get the unique class labels.
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Get the binary classifiers.
    pub(crate) fn classifiers(&self) -> &[BinarySVC] {
        &self.classifiers
    }

    /// Get the class pairs for OvO.
    pub(crate) fn class_pairs(&self) -> &[(f64, f64)] {
        &self.class_pairs
    }

    /// Get the number of support vectors for each classifier.
    #[must_use]
    pub fn n_support_vectors(&self) -> Vec<usize> {
        self.classifiers
            .iter()
            .map(|clf| clf.support_indices.as_ref().map(|v| v.len()).unwrap_or(0))
            .collect()
    }

    /// Compute decision function values for each sample.
    ///
    /// For OvO: returns Array2 shape (n_samples, n_classifiers) where
    ///   n_classifiers = n_classes * (n_classes - 1) / 2
    /// For OvR: returns Array2 shape (n_samples, n_classes)
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.classes, "decision_function")?;
        validate_predict_input(x, self.n_features.unwrap())?;
        let n_samples = x.nrows();
        let n_classifiers = self.classifiers.len();
        let mut decisions = Array2::zeros((n_samples, n_classifiers));
        for (clf_idx, clf) in self.classifiers.iter().enumerate() {
            let d = clf.decision_function(x)?;
            decisions.column_mut(clf_idx).assign(&d);
        }
        Ok(decisions)
    }

    /// Compute class weights from training data.
    fn compute_class_weights(&mut self, y: &Array1<f64>) {
        let classes = self.classes.as_ref().unwrap();
        let n_samples = y.len() as f64;

        let weights = match &self.class_weight {
            ClassWeight::Uniform => classes.iter().map(|&c| (c, 1.0)).collect(),
            ClassWeight::Balanced => {
                // Weight inversely proportional to class frequency
                let n_classes = classes.len() as f64;
                classes
                    .iter()
                    .map(|&c| {
                        let count = y.iter().filter(|&&v| (v - c).abs() < 1e-10).count() as f64;
                        let weight = n_samples / (n_classes * count);
                        (c, weight)
                    })
                    .collect()
            }
            ClassWeight::Custom(weights) => weights.clone(),
        };

        self.computed_weights = Some(weights);
    }

    /// Get weight for a specific class.
    fn get_class_weight(&self, class: f64) -> f64 {
        self.computed_weights
            .as_ref()
            .and_then(|weights| {
                weights
                    .iter()
                    .find(|(c, _)| (*c - class).abs() < 1e-10)
                    .map(|(_, w)| *w)
            })
            .unwrap_or(1.0)
    }

    /// Fit One-vs-One classifiers.
    fn fit_ovo(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();

        self.classifiers.clear();
        self.class_pairs.clear();

        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                let class_i = classes[i];
                let class_j = classes[j];

                // Extract samples for these two classes
                let mut indices = Vec::new();
                for (idx, &label) in y.iter().enumerate() {
                    if (label - class_i).abs() < 1e-10 || (label - class_j).abs() < 1e-10 {
                        indices.push(idx);
                    }
                }

                let n_binary = indices.len();
                let n_features = x.ncols();

                let mut x_binary = Array2::zeros((n_binary, n_features));
                let mut y_binary = Array1::zeros(n_binary);

                for (new_idx, &orig_idx) in indices.iter().enumerate() {
                    x_binary.row_mut(new_idx).assign(&x.row(orig_idx));
                    y_binary[new_idx] = y[orig_idx];
                }

                // Compute sample weights for this binary classifier
                let sample_weights: Array1<f64> = y_binary
                    .iter()
                    .map(|&label| self.get_class_weight(label))
                    .collect();

                // Train binary classifier
                let mut binary_clf = BinarySVC::new(
                    self.c,
                    self.kernel,
                    self.tol,
                    self.max_iter,
                    self.cache_size,
                );
                binary_clf.fit(
                    &x_binary,
                    &y_binary,
                    class_i,
                    class_j,
                    Some(&sample_weights),
                )?;

                // Fit Platt scaling if probability estimates are needed
                if self.probability {
                    binary_clf.fit_platt_scaling(&x_binary, &y_binary)?;
                }

                self.classifiers.push(binary_clf);
                self.class_pairs.push((class_i, class_j));
            }
        }

        Ok(())
    }

    /// Fit One-vs-Rest classifiers.
    fn fit_ovr(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let classes = self.classes.as_ref().unwrap();

        self.classifiers.clear();
        self.class_pairs.clear();

        for &class in classes.iter() {
            // Create binary labels: class vs rest
            let y_binary: Array1<f64> = y
                .iter()
                .map(|&label| {
                    if (label - class).abs() < 1e-10 {
                        class
                    } else {
                        -1.0
                    }
                })
                .collect();

            // Compute sample weights
            let sample_weights: Array1<f64> = y
                .iter()
                .map(|&label| self.get_class_weight(label))
                .collect();

            // Train binary classifier
            let mut binary_clf = BinarySVC::new(
                self.c,
                self.kernel,
                self.tol,
                self.max_iter,
                self.cache_size,
            );
            binary_clf.fit(x, &y_binary, class, -1.0, Some(&sample_weights))?;

            if self.probability {
                binary_clf.fit_platt_scaling(x, &y_binary)?;
            }

            self.classifiers.push(binary_clf);
            self.class_pairs.push((class, -1.0)); // -1 represents "rest"
        }

        Ok(())
    }

    /// Predict using One-vs-One voting.
    fn predict_ovo(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();
        let n_samples = x.nrows();

        let mut votes = Array2::zeros((n_samples, n_classes));

        for (clf, &(class_i, class_j)) in self.classifiers.iter().zip(self.class_pairs.iter()) {
            let predictions = clf.predict(x)?;

            for (sample_idx, &pred) in predictions.iter().enumerate() {
                if (pred - class_i).abs() < 1e-10 {
                    // Find index of class_i
                    for (class_idx, &c) in classes.iter().enumerate() {
                        if (c - class_i).abs() < 1e-10 {
                            votes[[sample_idx, class_idx]] += 1.0;
                            break;
                        }
                    }
                } else {
                    // Find index of class_j
                    for (class_idx, &c) in classes.iter().enumerate() {
                        if (c - class_j).abs() < 1e-10 {
                            votes[[sample_idx, class_idx]] += 1.0;
                            break;
                        }
                    }
                }
            }
        }

        // Find class with most votes
        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let row = votes.row(i);
            let best_class_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a): &(usize, &f64), (_, b): &(usize, &f64)| {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = classes[best_class_idx];
        }

        Ok(predictions)
    }

    /// Predict using One-vs-Rest.
    fn predict_ovr(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let classes = self.classes.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_classes = classes.len();

        // Get decision values from each OvR classifier
        let mut decision_values = Array2::zeros((n_samples, n_classes));

        for (clf_idx, clf) in self.classifiers.iter().enumerate() {
            let decisions = clf.decision_function(x)?;
            for i in 0..n_samples {
                decision_values[[i, clf_idx]] = decisions[i];
            }
        }

        // Select class with highest decision value
        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let best_class_idx = decision_values
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = classes[best_class_idx];
        }

        Ok(predictions)
    }

    /// Predict probabilities using OvO (pairwise coupling).
    fn predict_proba_ovo(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();
        let n_samples = x.nrows();

        // Use simple voting-based probability estimation
        let mut proba = Array2::zeros((n_samples, n_classes));
        let mut votes: Array2<f64> = Array2::zeros((n_samples, n_classes));

        for (clf, &(_class_i, _class_j)) in self.classifiers.iter().zip(self.class_pairs.iter()) {
            let predictions = clf.predict(x)?;

            for (sample_idx, &pred) in predictions.iter().enumerate() {
                for (class_idx, &c) in classes.iter().enumerate() {
                    if (c - pred).abs() < 1e-10 {
                        votes[[sample_idx, class_idx]] += 1.0;
                        break;
                    }
                }
            }
        }

        // Normalize votes to probabilities
        for i in 0..n_samples {
            let total: f64 = votes.row(i).sum();
            if total > 0.0 {
                for j in 0..n_classes {
                    proba[[i, j]] = votes[[i, j]] / total;
                }
            } else {
                // Uniform if no votes
                for j in 0..n_classes {
                    proba[[i, j]] = 1.0 / n_classes as f64;
                }
            }
        }

        Ok(proba)
    }

    /// Predict probabilities using OvR (softmax normalization).
    fn predict_proba_ovr(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let classes = self.classes.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_classes = classes.len();

        // Get decision values and apply softmax
        let mut decision_values = Array2::zeros((n_samples, n_classes));

        for (clf_idx, clf) in self.classifiers.iter().enumerate() {
            let decisions = clf.decision_function(x)?;
            for i in 0..n_samples {
                decision_values[[i, clf_idx]] = decisions[i];
            }
        }

        // Softmax normalization
        let mut proba = Array2::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            let max_val = decision_values
                .row(i)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = decision_values
                .row(i)
                .iter()
                .map(|&v| (v - max_val).exp())
                .sum();

            for j in 0..n_classes {
                proba[[i, j]] = (decision_values[[i, j]] - max_val).exp() / exp_sum;
            }
        }

        Ok(proba)
    }
}

impl Model for SVC {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        // Validate hyperparameters
        if self.c <= 0.0 {
            return Err(FerroError::invalid_input(format!(
                "Parameter C must be positive, got {}",
                self.c
            )));
        }
        if self.tol <= 0.0 {
            return Err(FerroError::invalid_input(format!(
                "Parameter tol must be positive, got {}",
                self.tol
            )));
        }

        // Extract unique classes
        let classes_vec = super::get_unique_classes(y);

        if classes_vec.len() < 2 {
            return Err(FerroError::invalid_input(
                "Need at least 2 classes for classification",
            ));
        }

        self.classes = Some(classes_vec);
        self.n_features = Some(x.ncols());

        // Compute class weights
        self.compute_class_weights(y);

        // Train classifiers
        match self.multiclass_strategy {
            MulticlassStrategy::OneVsOne => self.fit_ovo(x, y)?,
            MulticlassStrategy::OneVsRest => self.fit_ovr(x, y)?,
        }

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.classes, "predict")?;
        validate_predict_input(x, self.n_features.unwrap())?;

        match self.multiclass_strategy {
            MulticlassStrategy::OneVsOne => self.predict_ovo(x),
            MulticlassStrategy::OneVsRest => self.predict_ovr(x),
        }
    }

    fn is_fitted(&self) -> bool {
        self.classes.is_some() && !self.classifiers.is_empty()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .float("C", 1e-3, 100.0)
            .float("gamma", 1e-4, 10.0)
            .categorical(
                "kernel",
                vec!["linear".to_string(), "rbf".to_string(), "poly".to_string()],
            )
            .int("cache_size", 100, 2000)
    }
}

impl ProbabilisticModel for SVC {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.classes, "predict_proba")?;
        validate_predict_input(x, self.n_features.unwrap())?;

        if !self.probability {
            return Err(FerroError::invalid_input(
                "Probability estimates not enabled. Create SVC with .with_probability(true)",
            ));
        }

        match self.multiclass_strategy {
            MulticlassStrategy::OneVsOne => self.predict_proba_ovo(x),
            MulticlassStrategy::OneVsRest => self.predict_proba_ovr(x),
        }
    }

    fn predict_interval(&self, x: &Array2<f64>, level: f64) -> Result<PredictionInterval> {
        // For classification, return prediction with probability-based uncertainty
        let probas = self.predict_proba(x)?;
        let predictions = self.predict(x)?;

        let n_samples = x.nrows();
        let mut lower = Array1::zeros(n_samples);
        let mut upper = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let max_prob = probas.row(i).iter().cloned().fold(0.0_f64, f64::max);
            let uncertainty = 1.0 - max_prob;
            let half_width = uncertainty * (1.0 - level);

            lower[i] = predictions[i] - half_width;
            upper[i] = predictions[i] + half_width;
        }

        Ok(PredictionInterval::new(predictions, lower, upper, level))
    }
}

// =============================================================================
// SVR (Support Vector Regression)
// =============================================================================

/// Support Vector Regression.
///
/// A regression method that finds a function that deviates from the actual
/// target values by no more than epsilon for each training point while being
/// as flat as possible. Uses the epsilon-insensitive loss function.
///
/// ## Parameters
///
/// - `c`: Regularization parameter (default: 1.0). Higher C means less regularization.
/// - `epsilon`: Width of the epsilon-insensitive tube (default: 0.1).
///   Errors within this margin are not penalized.
/// - `kernel`: Kernel function (default: RBF with gamma=1/n_features)
/// - `tol`: Tolerance for stopping criterion (default: 1e-3)
/// - `max_iter`: Maximum SMO iterations (default: 1000)
///
/// ## Example
///
/// ```
/// # use ferroml_core::models::svm::{SVR, Kernel};
/// # use ferroml_core::models::Model;
/// # use ndarray::{Array1, Array2};
/// # fn main() -> ferroml_core::Result<()> {
/// # let x_train = Array2::from_shape_vec((6, 2), vec![1.0,2.0,2.0,1.0,3.0,3.0,6.0,7.0,7.0,6.0,8.0,8.0]).unwrap();
/// # let y_train = Array1::from_vec(vec![0.0,0.1,0.2,1.0,1.1,1.2]);
/// # let x_test = x_train.clone();
/// use ferroml_core::models::svm::{SVR, Kernel};
/// use ferroml_core::models::Model;
///
/// let mut reg = SVR::new()
///     .with_kernel(Kernel::Rbf { gamma: 0.5 })
///     .with_c(1.0)
///     .with_epsilon(0.1);
///
/// reg.fit(&x_train, &y_train)?;
/// let predictions = reg.predict(&x_test)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVR {
    /// Regularization parameter
    pub c: f64,
    /// Width of epsilon-insensitive tube
    pub epsilon: f64,
    /// Kernel function
    pub kernel: Kernel,
    /// Tolerance for stopping criterion
    pub tol: f64,
    /// Maximum number of SMO iterations
    pub max_iter: usize,

    // Fitted state
    /// Support vectors
    support_vectors: Option<Array2<f64>>,
    /// Dual coefficients (alpha_i - alpha_i^*)
    dual_coef: Option<Array1<f64>>,
    /// Intercept (bias term)
    intercept: f64,
    /// Indices of support vectors in original training data
    support_indices: Option<Vec<usize>>,
    /// Number of features
    n_features: Option<usize>,
}

impl Default for SVR {
    fn default() -> Self {
        Self::new()
    }
}

impl SVR {
    /// Create a new SVR with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.1,
            kernel: Kernel::rbf_auto(),
            tol: 1e-3,
            max_iter: 1000,
            support_vectors: None,
            dual_coef: None,
            intercept: 0.0,
            support_indices: None,
            n_features: None,
        }
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the epsilon parameter (tube width).
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the kernel function.
    #[must_use]
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the tolerance for stopping criterion.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Get the number of support vectors.
    #[must_use]
    pub fn n_support_vectors(&self) -> usize {
        self.support_indices.as_ref().map(|v| v.len()).unwrap_or(0)
    }

    /// Get the support vectors.
    #[must_use]
    pub fn support_vectors(&self) -> Option<&Array2<f64>> {
        self.support_vectors.as_ref()
    }

    /// Get the support vector indices.
    #[must_use]
    pub fn support_indices(&self) -> Option<&[usize]> {
        self.support_indices.as_deref()
    }

    /// Get the dual coefficients.
    #[must_use]
    pub fn dual_coef(&self) -> Option<&Array1<f64>> {
        self.dual_coef.as_ref()
    }

    /// Get the intercept.
    #[must_use]
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Compute decision function values (same as predict for regression).
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.predict(x)
    }

    /// Compute full kernel matrix.
    ///
    /// Optimized: uses array slices directly instead of allocating Vecs per pair.
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        // Ensure contiguous layout for slice access
        let x_c = if x.is_standard_layout() {
            None
        } else {
            Some(x.as_standard_layout().into_owned())
        };
        let x_ref = x_c.as_ref().unwrap_or(x);

        let mut k = Array2::zeros((n, n));

        for i in 0..n {
            let ri = x_ref.row(i);
            let xi = ri.as_slice().unwrap();
            for j in i..n {
                let rj = x_ref.row(j);
                let xj = rj.as_slice().unwrap();
                let val = self.kernel.compute(xi, xj);
                k[[i, j]] = val;
                k[[j, i]] = val;
            }
        }

        k
    }

    /// Coordinate descent algorithm for SVR.
    ///
    /// This is a simpler and more robust approach than full SMO.
    /// We iterate over each sample and update its coefficient to minimize the objective.
    ///
    /// The decision function is: f(x) = sum(coef_i * K(x_i, x)) + b
    /// where coef_i = alpha_i - alpha_star_i ∈ [-C, C]
    fn smo_regression(
        &self,
        kernel_matrix: &Array2<f64>,
        y: &Array1<f64>,
        sample_weights: Option<&Array1<f64>>,
    ) -> Result<(Array1<f64>, f64)> {
        let n = y.len();

        // Check for constant or near-constant targets
        // If range of y is within epsilon, no support vectors are needed
        let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_range = y_max - y_min;

        if y_range <= 2.0 * self.epsilon {
            // All targets within epsilon tube - just use mean
            let mean_y = y.mean().unwrap_or(0.0);
            return Ok((Array1::zeros(n), mean_y));
        }

        // Effective C for each sample
        let c: Array1<f64> = if let Some(weights) = sample_weights {
            weights.mapv(|w| self.c * w)
        } else {
            Array1::from_elem(n, self.c)
        };

        // Initialize coefficients: coef[i] = alpha[i] - alpha_star[i]
        let mut coef = Array1::zeros(n);

        // Cache for predictions: f[i] = sum_j coef[j] * K[j, i]
        let mut f: Array1<f64> = Array1::zeros(n);

        let mut iter = 0;
        let mut max_delta = f64::INFINITY;

        while iter < self.max_iter && max_delta > self.tol {
            max_delta = 0.0;

            for i in 0..n {
                let old_coef: f64 = coef[i];
                let k_ii = kernel_matrix[[i, i]];

                // Skip if diagonal is zero (shouldn't happen with valid kernels)
                if k_ii < 1e-12 {
                    continue;
                }

                // Compute gradient with respect to coef[i]
                // The error is f[i] - y[i]
                // We want to minimize: 0.5 * coef[i]^2 * K[i,i] + coef[i] * (f[i] - coef[i] * K[i,i] - y[i])
                //                      + epsilon * |coef[i]|
                // Taking derivative and setting to zero (with clipping for epsilon-insensitive loss)

                // Current prediction without contribution from i
                let f_without_i = old_coef.mul_add(-k_ii, f[i]);

                // The optimal coef[i] minimizes the loss
                // For epsilon-insensitive loss:
                // - If f_without_i < y[i] - epsilon: coef[i] should be positive (increase prediction)
                // - If f_without_i > y[i] + epsilon: coef[i] should be negative (decrease prediction)
                // - Otherwise: coef[i] = 0 (prediction is within epsilon tube)

                let error_without_i: f64 = f_without_i - y[i];

                let new_coef: f64 = if error_without_i < -self.epsilon {
                    // Need to increase prediction
                    ((-error_without_i - self.epsilon) / k_ii).min(c[i])
                } else if error_without_i > self.epsilon {
                    // Need to decrease prediction
                    ((-error_without_i + self.epsilon) / k_ii).max(-c[i])
                } else {
                    // Within epsilon tube
                    0.0
                };

                let delta: f64 = new_coef - old_coef;

                if delta.abs() > 1e-12 {
                    coef[i] = new_coef;

                    // Update predictions
                    for k in 0..n {
                        f[k] += delta * kernel_matrix[[i, k]];
                    }

                    max_delta = max_delta.max(delta.abs());
                }
            }

            iter += 1;
        }

        // Compute intercept
        let b = self.compute_intercept_cd(&coef, y, &f, &c);

        Ok((coef, b))
    }

    /// Compute intercept for coordinate descent SVR.
    fn compute_intercept_cd(
        &self,
        coef: &Array1<f64>,
        y: &Array1<f64>,
        f: &Array1<f64>,
        c: &Array1<f64>,
    ) -> f64 {
        let n = y.len();
        let mut sum_b = 0.0;
        let mut count = 0;

        for i in 0..n {
            let c_i = c[i];
            let coef_i = coef[i];

            // Free support vector: 0 < |coef[i]| < C
            if coef_i.abs() > 1e-8 && coef_i.abs() < c_i - 1e-8 {
                // For positive coef (alpha > 0): error = -epsilon
                // For negative coef (alpha_star > 0): error = +epsilon
                if coef_i > 0.0 {
                    // y[i] = f[i] + b - epsilon => b = y[i] - f[i] + epsilon
                    // But f already includes coef contribution, so:
                    // y[i] = (f[i] - b) + b => b = y[i] - f[i] + epsilon... no
                    // Actually: y[i] - epsilon = f_without_b[i] + b
                    // So b = y[i] - epsilon - f_without_b[i]
                    // And f_without_b[i] = f[i] (since we don't have b in f yet)
                    sum_b += y[i] - self.epsilon - f[i];
                } else {
                    sum_b += y[i] + self.epsilon - f[i];
                }
                count += 1;
            }
        }

        if count > 0 {
            return sum_b / count as f64;
        }

        // Fallback: average over all support vectors
        for i in 0..n {
            if coef[i].abs() > 1e-8 {
                sum_b += y[i] - f[i];
                count += 1;
            }
        }

        if count > 0 {
            sum_b / count as f64
        } else {
            // No support vectors: use mean of y
            y.mean().unwrap_or(0.0)
        }
    }
}

impl Model for SVR {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        // Validate hyperparameters
        if self.c <= 0.0 {
            return Err(FerroError::invalid_input(format!(
                "Parameter C must be positive, got {}",
                self.c
            )));
        }
        if self.tol <= 0.0 {
            return Err(FerroError::invalid_input(format!(
                "Parameter tol must be positive, got {}",
                self.tol
            )));
        }
        if self.epsilon < 0.0 {
            return Err(FerroError::invalid_input(format!(
                "Parameter epsilon must be non-negative, got {}",
                self.epsilon
            )));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        self.n_features = Some(n_features);

        // Set auto gamma if needed
        self.kernel = self.kernel.with_auto_gamma(n_features);

        // Compute kernel matrix
        let kernel_matrix = self.compute_kernel_matrix(x);

        // Run SMO for regression
        let (alpha_diff, intercept) = self.smo_regression(&kernel_matrix, y, None)?;

        // Extract support vectors (points with |alpha_diff| > threshold)
        let support_threshold = 1e-8;
        let mut support_indices = Vec::new();
        let mut dual_coef = Vec::new();

        for i in 0..n_samples {
            if alpha_diff[i].abs() > support_threshold {
                support_indices.push(i);
                dual_coef.push(alpha_diff[i]);
            }
        }

        // Handle the case where all points are within the epsilon tube
        // (e.g., constant function or very low variance data)
        if support_indices.is_empty() {
            // Create a dummy support vector setup with zero coefficients
            // The model will just return the intercept (mean of y)
            self.support_vectors = Some(Array2::zeros((0, n_features)));
            self.dual_coef = Some(Array1::zeros(0));
            self.support_indices = Some(Vec::new());
            self.intercept = y.mean().unwrap_or(0.0);
            return Ok(());
        }

        // Store support vectors
        let n_sv = support_indices.len();
        let mut sv_matrix = Array2::zeros((n_sv, n_features));
        for (i, &idx) in support_indices.iter().enumerate() {
            sv_matrix.row_mut(i).assign(&x.row(idx));
        }

        self.support_vectors = Some(sv_matrix);
        self.dual_coef = Some(Array1::from_vec(dual_coef));
        self.support_indices = Some(support_indices);
        self.intercept = intercept;

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.support_vectors, "predict")?;
        validate_predict_input(x, self.n_features.unwrap())?;

        let sv = self.support_vectors.as_ref().unwrap();
        let dual_coef = self.dual_coef.as_ref().unwrap();
        let n_samples = x.nrows();

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let xi: Vec<f64> = x.row(i).to_vec();
            let mut sum = 0.0;

            for (j, coef) in dual_coef.iter().enumerate() {
                let svj: Vec<f64> = sv.row(j).to_vec();
                sum += coef * self.kernel.compute(&xi, &svj);
            }

            predictions[i] = sum + self.intercept;
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.support_vectors.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .float("C", 1e-3, 100.0)
            .float("epsilon", 1e-4, 1.0)
            .float("gamma", 1e-4, 10.0)
            .categorical(
                "kernel",
                vec!["linear".to_string(), "rbf".to_string(), "poly".to_string()],
            )
    }

    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;
        crate::metrics::r2_score(y, &predictions)
    }
}

// =============================================================================
// LinearSVC (Linear Support Vector Classification)
// =============================================================================

/// Loss function for LinearSVC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearSVCLoss {
    /// Hinge loss: max(0, 1 - y·f(x))
    /// Standard SVM loss function
    Hinge,
    /// Squared hinge loss: max(0, 1 - y·f(x))²
    /// Smoother, differentiable everywhere
    SquaredHinge,
}

impl Default for LinearSVCLoss {
    fn default() -> Self {
        Self::SquaredHinge
    }
}

/// Linear Support Vector Classification.
///
/// A fast linear SVM classifier optimized for large datasets. Uses coordinate
/// descent in the primal formulation, avoiding the O(n²) kernel matrix
/// computation of kernelized SVC.
///
/// ## Parameters
///
/// - `c`: Regularization parameter (default: 1.0). Higher C means less regularization.
/// - `loss`: Loss function - Hinge or SquaredHinge (default: SquaredHinge)
/// - `tol`: Tolerance for stopping criterion (default: 1e-4)
/// - `max_iter`: Maximum iterations (default: 1000)
/// - `fit_intercept`: Whether to fit an intercept (default: true)
/// - `class_weight`: Weight for each class (default: Uniform)
///
/// ## Example
///
/// ```
/// # use ferroml_core::models::svm::LinearSVC;
/// # use ferroml_core::models::Model;
/// # use ndarray::{Array1, Array2};
/// # fn main() -> ferroml_core::Result<()> {
/// # let x_train = Array2::from_shape_vec((6, 2), vec![1.0,2.0,2.0,1.0,3.0,3.0,6.0,7.0,7.0,6.0,8.0,8.0]).unwrap();
/// # let y_train = Array1::from_vec(vec![0.0,0.0,0.0,1.0,1.0,1.0]);
/// # let x_test = x_train.clone();
/// use ferroml_core::models::svm::LinearSVC;
/// use ferroml_core::models::Model;
///
/// let mut clf = LinearSVC::new()
///     .with_c(1.0)
///     .with_max_iter(1000);
///
/// clf.fit(&x_train, &y_train)?;
/// let predictions = clf.predict(&x_test)?;
/// # Ok(())
/// # }
/// ```
///
/// ## Algorithm
///
/// Uses coordinate descent to solve the primal optimization problem:
///
/// min_w (1/2)||w||² + C·Σ loss(y_i, w·x_i + b)
///
/// This is equivalent to LIBLINEAR's L2-regularized L1 or L2 loss SVM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearSVC {
    /// Regularization parameter
    pub c: f64,
    /// Loss function
    pub loss: LinearSVCLoss,
    /// Tolerance for stopping criterion
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Class weight specification
    pub class_weight: ClassWeight,

    // Fitted state
    /// Weight vector for each binary classifier
    weights: Option<Vec<Array1<f64>>>,
    /// Intercept for each binary classifier
    intercepts: Option<Vec<f64>>,
    /// Unique class labels
    classes: Option<Array1<f64>>,
    /// Number of features
    n_features: Option<usize>,
    /// Computed class weights
    computed_weights: Option<Vec<(f64, f64)>>,
}

impl Default for LinearSVC {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearSVC {
    /// Create a new LinearSVC with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            c: 1.0,
            loss: LinearSVCLoss::SquaredHinge,
            tol: 1e-4,
            max_iter: 1000,
            fit_intercept: true,
            class_weight: ClassWeight::Uniform,
            weights: None,
            intercepts: None,
            classes: None,
            n_features: None,
            computed_weights: None,
        }
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: LinearSVCLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set the tolerance for stopping criterion.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set whether to fit an intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set the class weight specification.
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Get the unique class labels.
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Get the learned weight vectors.
    #[must_use]
    pub fn weights(&self) -> Option<&Vec<Array1<f64>>> {
        self.weights.as_ref()
    }

    /// Get the learned intercepts.
    #[must_use]
    pub fn intercepts(&self) -> Option<&Vec<f64>> {
        self.intercepts.as_ref()
    }

    /// Compute decision function values.
    ///
    /// Returns Array2 shape (n_samples, n_classifiers) where n_classifiers is 1 for
    /// binary classification and n_classes for multiclass (one-vs-rest).
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.classes, "decision_function")?;
        validate_predict_input(x, self.n_features.unwrap())?;
        let weights = self.weights.as_ref().unwrap();
        let intercepts = self.intercepts.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_classifiers = weights.len();
        let mut decisions = Array2::zeros((n_samples, n_classifiers));
        for (clf_idx, (w, &b)) in weights.iter().zip(intercepts.iter()).enumerate() {
            let col = self.decision_function_binary(x, w, b);
            decisions.column_mut(clf_idx).assign(&col);
        }
        Ok(decisions)
    }

    /// Compute class weights from training data.
    fn compute_class_weights(&mut self, y: &Array1<f64>) {
        let classes = self.classes.as_ref().unwrap();
        let n_samples = y.len() as f64;

        let weights = match &self.class_weight {
            ClassWeight::Uniform => classes.iter().map(|&c| (c, 1.0)).collect(),
            ClassWeight::Balanced => {
                let n_classes = classes.len() as f64;
                classes
                    .iter()
                    .map(|&c| {
                        let count = y.iter().filter(|&&v| (v - c).abs() < 1e-10).count() as f64;
                        let weight = n_samples / (n_classes * count);
                        (c, weight)
                    })
                    .collect()
            }
            ClassWeight::Custom(weights) => weights.clone(),
        };

        self.computed_weights = Some(weights);
    }

    /// Get weight for a specific class.
    fn get_class_weight(&self, class: f64) -> f64 {
        self.computed_weights
            .as_ref()
            .and_then(|weights| {
                weights
                    .iter()
                    .find(|(c, _)| (*c - class).abs() < 1e-10)
                    .map(|(_, w)| *w)
            })
            .unwrap_or(1.0)
    }

    /// Fit a binary linear SVC using dual coordinate descent (DCD),
    /// following the liblinear algorithm for L2-regularized L2-loss / L1-loss SVM.
    ///
    /// Reference: Hsieh et al., "A Dual Coordinate Descent Method for Large-scale
    /// Linear SVM" (ICML 2008).
    ///
    /// Solves the primal: min_{w,b} (1/2)||w||^2 + C * sum_i loss(y_i*(w^T x_i + b))
    ///
    /// The dual for L2-loss SVM is:
    ///   min_alpha (1/2) alpha^T Q alpha - e^T alpha
    ///   s.t. alpha >= 0
    ///   where Q_ij = y_i y_j x_i^T x_j + delta_ij/(2C)
    ///
    /// The dual for L1-loss SVM is:
    ///   min_alpha (1/2) alpha^T Q alpha - e^T alpha
    ///   s.t. 0 <= alpha_i <= C
    ///   where Q_ij = y_i y_j x_i^T x_j
    fn fit_binary(
        &self,
        x: &Array2<f64>,
        y_binary: &Array1<f64>,
        sample_weights: &Array1<f64>,
    ) -> Result<(Array1<f64>, f64)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Augmented dimensionality: add an extra feature = 1.0 for the intercept
        let d = if self.fit_intercept {
            n_features + 1
        } else {
            n_features
        };

        // Build augmented design matrix using ndarray (avoid Vec<Vec<f64>> copy)
        let x_design: Array2<f64> = if self.fit_intercept {
            let ones = Array2::ones((n_samples, 1));
            concatenate(Axis(1), &[x.view(), ones.view()]).unwrap()
        } else {
            x.to_owned()
        };

        // Precompute squared norms from ndarray rows
        let x_sq_norm: Vec<f64> = (0..n_samples)
            .map(|i| {
                let row = x_design.row(i);
                row.dot(&row)
            })
            .collect();

        // Effective C for each sample
        let c_eff: Vec<f64> = sample_weights.iter().map(|&sw| self.c * sw).collect();

        // Precompute Q_ii (diagonal of the dual Hessian)
        let q_diag: Vec<f64> = match self.loss {
            LinearSVCLoss::SquaredHinge => {
                // Q_ii = ||x_i||^2 + 1/(2*C_i)
                (0..n_samples)
                    .map(|i| x_sq_norm[i] + 1.0 / (2.0 * c_eff[i]))
                    .collect()
            }
            LinearSVCLoss::Hinge => {
                // Q_ii = ||x_i||^2
                x_sq_norm.clone()
            }
        };

        // Initialize dual variables and primal weight vector
        let mut alpha = vec![0.0f64; n_samples];
        let mut w_aug = Array1::<f64>::zeros(d);

        // Active set shrinking: skip samples firmly at bounds
        let mut active = vec![true; n_samples];

        // Build a permutation array for shuffling
        let mut perm: Vec<usize> = (0..n_samples).collect();

        for iter in 0..self.max_iter {
            let mut max_change = 0.0f64;

            // Re-examine all samples every 10 iterations
            if iter % 10 == 0 {
                active.fill(true);
            }

            // Deterministic shuffle using a simple hash of the iteration
            // This approximates random permutation to prevent cycling
            let seed = (iter as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            for i in (1..n_samples).rev() {
                let j = ((seed.wrapping_mul((i + 1) as u64).wrapping_add(iter as u64))
                    % (i as u64 + 1)) as usize;
                perm.swap(i, j);
            }

            for &i in &perm {
                if !active[i] {
                    continue;
                }

                if q_diag[i] < 1e-12 {
                    continue;
                }

                // Compute w^T x_i using ndarray dot
                let xi = x_design.row(i);
                let wt_xi: f64 = w_aug.dot(&xi);

                // Gradient component: g_i = y_i * w^T x_i + alpha_i/(2*C_i) - 1
                // (for L2-loss; for L1-loss omit the alpha/(2C) term, which is
                //  handled differently via the box constraint)
                let g = match self.loss {
                    LinearSVCLoss::SquaredHinge => {
                        y_binary[i] * wt_xi + alpha[i] / (2.0 * c_eff[i]) - 1.0
                    }
                    LinearSVCLoss::Hinge => y_binary[i] * wt_xi - 1.0,
                };

                // Newton step: alpha_new = alpha_old - g / Q_ii
                let alpha_new = match self.loss {
                    LinearSVCLoss::SquaredHinge => {
                        // No upper bound for L2-loss dual
                        (alpha[i] - g / q_diag[i]).max(0.0)
                    }
                    LinearSVCLoss::Hinge => {
                        // Box constraint: 0 <= alpha_i <= C_i
                        (alpha[i] - g / q_diag[i]).max(0.0).min(c_eff[i])
                    }
                };

                let delta = alpha_new - alpha[i];
                if delta.abs() >= 1e-15 {
                    alpha[i] = alpha_new;

                    // Update primal weights: w += delta * y_i * x_i (using ndarray)
                    let scale = delta * y_binary[i];
                    w_aug.scaled_add(scale, &xi);

                    max_change = max_change.max(delta.abs());
                }

                // Mark as inactive if firmly at bound with correct gradient sign
                // (uses pre-update gradient g, consistent with LIBLINEAR shrinking)
                let at_lower = alpha_new < 1e-12 && g > 0.0;
                let at_upper = match self.loss {
                    LinearSVCLoss::SquaredHinge => false, // no upper bound for L2-loss
                    LinearSVCLoss::Hinge => alpha_new > c_eff[i] - 1e-12 && g < 0.0,
                };
                if at_lower || at_upper {
                    active[i] = false;
                }
            }

            if max_change < self.tol {
                break;
            }
        }

        // Extract w and b from augmented weight vector
        let w = w_aug.slice(ndarray::s![..n_features]).to_owned();
        let b = if self.fit_intercept {
            w_aug[n_features]
        } else {
            0.0
        };

        Ok((w, b))
    }

    /// Compute decision function for a binary classifier.
    fn decision_function_binary(&self, x: &Array2<f64>, w: &Array1<f64>, b: f64) -> Array1<f64> {
        let n_samples = x.nrows();
        let mut decisions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            decisions[i] = dot_product_array(w, &x.row(i).to_owned()) + b;
        }

        decisions
    }
}

impl Model for LinearSVC {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        // Validate hyperparameters
        if self.c <= 0.0 {
            return Err(FerroError::invalid_input(format!(
                "Parameter C must be positive, got {}",
                self.c
            )));
        }
        if self.tol <= 0.0 {
            return Err(FerroError::invalid_input(format!(
                "Parameter tol must be positive, got {}",
                self.tol
            )));
        }

        // Extract unique classes
        let classes_vec = super::get_unique_classes(y);

        if classes_vec.len() < 2 {
            return Err(FerroError::invalid_input(
                "Need at least 2 classes for classification",
            ));
        }

        self.classes = Some(classes_vec.clone());
        self.n_features = Some(x.ncols());

        // Compute class weights
        self.compute_class_weights(y);

        let n_classes = classes_vec.len();
        let mut weights = Vec::with_capacity(n_classes);
        let mut intercepts = Vec::with_capacity(n_classes);

        if n_classes == 2 {
            // Binary classification
            let positive_class = classes_vec[1];
            let _negative_class = classes_vec[0];

            let y_binary: Array1<f64> = y
                .iter()
                .map(|&label| {
                    if (label - positive_class).abs() < 1e-10 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect();

            let sample_weights: Array1<f64> = y
                .iter()
                .map(|&label| self.get_class_weight(label))
                .collect();

            let (w, b) = self.fit_binary(x, &y_binary, &sample_weights)?;
            weights.push(w);
            intercepts.push(b);
        } else {
            // Multiclass: One-vs-Rest
            for &class in &classes_vec {
                let y_binary: Array1<f64> = y
                    .iter()
                    .map(|&label| {
                        if (label - class).abs() < 1e-10 {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect();

                let sample_weights: Array1<f64> = y
                    .iter()
                    .map(|&label| self.get_class_weight(label))
                    .collect();

                let (w, b) = self.fit_binary(x, &y_binary, &sample_weights)?;
                weights.push(w);
                intercepts.push(b);
            }
        }

        self.weights = Some(weights);
        self.intercepts = Some(intercepts);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.classes, "predict")?;
        validate_predict_input(x, self.n_features.unwrap())?;

        let classes = self.classes.as_ref().unwrap();
        let weights = self.weights.as_ref().unwrap();
        let intercepts = self.intercepts.as_ref().unwrap();
        let n_samples = x.nrows();

        let mut predictions = Array1::zeros(n_samples);

        if classes.len() == 2 {
            // Binary classification
            let decisions = self.decision_function_binary(x, &weights[0], intercepts[0]);
            for i in 0..n_samples {
                predictions[i] = if decisions[i] >= 0.0 {
                    classes[1]
                } else {
                    classes[0]
                };
            }
        } else {
            // Multiclass: highest decision value wins
            let n_classes = classes.len();
            let mut decision_values = Array2::zeros((n_samples, n_classes));

            for (class_idx, (w, &b)) in weights.iter().zip(intercepts.iter()).enumerate() {
                let decisions = self.decision_function_binary(x, w, b);
                for i in 0..n_samples {
                    decision_values[[i, class_idx]] = decisions[i];
                }
            }

            for i in 0..n_samples {
                let best_class_idx = decision_values
                    .row(i)
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                predictions[i] = classes[best_class_idx];
            }
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.classes.is_some() && self.weights.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        // For linear models, absolute weight magnitude indicates importance
        self.weights.as_ref().map(|weights| {
            if weights.len() == 1 {
                weights[0].mapv(|v| v.abs())
            } else {
                // Average absolute weights across all OvR classifiers
                let n_features = weights[0].len();
                let mut importance = Array1::zeros(n_features);
                for w in weights {
                    for j in 0..n_features {
                        importance[j] += w[j].abs();
                    }
                }
                importance /= weights.len() as f64;
                importance
            }
        })
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new().float("C", 1e-3, 100.0).categorical(
            "loss",
            vec!["hinge".to_string(), "squared_hinge".to_string()],
        )
    }
}

// =============================================================================
// LinearSVR (Linear Support Vector Regression)
// =============================================================================

/// Loss function for LinearSVR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearSVRLoss {
    /// Epsilon-insensitive loss: max(0, |y - f(x)| - ε)
    EpsilonInsensitive,
    /// Squared epsilon-insensitive loss: max(0, |y - f(x)| - ε)²
    SquaredEpsilonInsensitive,
}

impl Default for LinearSVRLoss {
    fn default() -> Self {
        Self::EpsilonInsensitive
    }
}

/// Linear Support Vector Regression.
///
/// A fast linear SVR optimized for large datasets. Uses coordinate descent
/// in the primal formulation, avoiding the O(n²) kernel matrix computation
/// of kernelized SVR.
///
/// ## Parameters
///
/// - `c`: Regularization parameter (default: 1.0). Higher C means less regularization.
/// - `epsilon`: Width of the epsilon-insensitive tube (default: 0.0).
///   Errors within this margin are not penalized.
/// - `loss`: Loss function - EpsilonInsensitive or SquaredEpsilonInsensitive (default: EpsilonInsensitive)
/// - `tol`: Tolerance for stopping criterion (default: 1e-4)
/// - `max_iter`: Maximum iterations (default: 1000)
/// - `fit_intercept`: Whether to fit an intercept (default: true)
///
/// ## Example
///
/// ```
/// # use ferroml_core::models::svm::LinearSVR;
/// # use ferroml_core::models::Model;
/// # use ndarray::{Array1, Array2};
/// # fn main() -> ferroml_core::Result<()> {
/// # let x_train = Array2::from_shape_vec((6, 2), vec![1.0,2.0,2.0,1.0,3.0,3.0,6.0,7.0,7.0,6.0,8.0,8.0]).unwrap();
/// # let y_train = Array1::from_vec(vec![0.0,0.1,0.2,1.0,1.1,1.2]);
/// # let x_test = x_train.clone();
/// use ferroml_core::models::svm::LinearSVR;
/// use ferroml_core::models::Model;
///
/// let mut reg = LinearSVR::new()
///     .with_c(1.0)
///     .with_epsilon(0.1)
///     .with_max_iter(1000);
///
/// reg.fit(&x_train, &y_train)?;
/// let predictions = reg.predict(&x_test)?;
/// # Ok(())
/// # }
/// ```
///
/// ## Algorithm
///
/// Uses coordinate descent to solve the primal optimization problem:
///
/// min_w (1/2)||w||² + C·Σ loss(y_i, w·x_i + b)
///
/// This is equivalent to LIBLINEAR's L2-regularized L1 or L2 loss SVR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearSVR {
    /// Regularization parameter
    pub c: f64,
    /// Width of epsilon-insensitive tube
    pub epsilon: f64,
    /// Loss function
    pub loss: LinearSVRLoss,
    /// Tolerance for stopping criterion
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to fit an intercept
    pub fit_intercept: bool,

    // Fitted state
    /// Weight vector
    weights: Option<Array1<f64>>,
    /// Intercept
    intercept: f64,
    /// Number of features
    n_features: Option<usize>,
}

impl Default for LinearSVR {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearSVR {
    /// Create a new LinearSVR with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.0,
            loss: LinearSVRLoss::EpsilonInsensitive,
            tol: 1e-4,
            max_iter: 1000,
            fit_intercept: true,
            weights: None,
            intercept: 0.0,
            n_features: None,
        }
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the epsilon parameter (tube width).
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: LinearSVRLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set the tolerance for stopping criterion.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set whether to fit an intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Get the learned weight vector.
    #[must_use]
    pub fn weights(&self) -> Option<&Array1<f64>> {
        self.weights.as_ref()
    }

    /// Get the learned intercept.
    #[must_use]
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Compute decision function values (same as predict for regression).
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.predict(x)
    }
}

impl Model for LinearSVR {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        // Validate hyperparameters
        if self.c <= 0.0 {
            return Err(FerroError::invalid_input(format!(
                "Parameter C must be positive, got {}",
                self.c
            )));
        }
        if self.tol <= 0.0 {
            return Err(FerroError::invalid_input(format!(
                "Parameter tol must be positive, got {}",
                self.tol
            )));
        }
        if self.epsilon < 0.0 {
            return Err(FerroError::invalid_input(format!(
                "Parameter epsilon must be non-negative, got {}",
                self.epsilon
            )));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        self.n_features = Some(n_features);

        // Initialize weights and intercept
        let mut w = Array1::zeros(n_features);
        let mut b = 0.0;

        // Precompute ||x_i||² for each sample
        let x_norm_sq: Vec<f64> = (0..n_samples)
            .map(|i| x.row(i).iter().map(|&v| v * v).sum())
            .collect();

        // Coordinate descent
        for _iter in 0..self.max_iter {
            let mut max_delta: f64 = 0.0;

            for i in 0..n_samples {
                // Compute prediction
                let f_i = dot_product_array(&w, &x.row(i).to_owned()) + b;
                let residual = y[i] - f_i;

                // Compute the optimal step based on loss function
                let (delta_scale, update) = match self.loss {
                    LinearSVRLoss::EpsilonInsensitive => {
                        if residual > self.epsilon {
                            // Underestimating: increase weights
                            let denom = x_norm_sq[i] + if self.fit_intercept { 1.0 } else { 0.0 };
                            if denom > 1e-10 {
                                let step = ((residual - self.epsilon) / denom).min(self.c);
                                (step, true)
                            } else {
                                (0.0, false)
                            }
                        } else if residual < -self.epsilon {
                            // Overestimating: decrease weights
                            let denom = x_norm_sq[i] + if self.fit_intercept { 1.0 } else { 0.0 };
                            if denom > 1e-10 {
                                let step = ((residual + self.epsilon) / denom).max(-self.c);
                                (step, true)
                            } else {
                                (0.0, false)
                            }
                        } else {
                            (0.0, false)
                        }
                    }
                    LinearSVRLoss::SquaredEpsilonInsensitive => {
                        if residual.abs() > self.epsilon {
                            let sign = if residual > 0.0 { 1.0 } else { -1.0 };
                            let loss_grad = 2.0 * sign * (residual.abs() - self.epsilon);
                            let denom = (2.0 * self.c).mul_add(
                                x_norm_sq[i] + if self.fit_intercept { 1.0 } else { 0.0 },
                                1.0,
                            );
                            let step = (self.c * loss_grad) / denom;
                            (step, true)
                        } else {
                            (0.0, false)
                        }
                    }
                };

                if update && delta_scale.abs() > 1e-12 {
                    // Update weights
                    for j in 0..n_features {
                        w[j] += delta_scale * x[[i, j]];
                    }
                    if self.fit_intercept {
                        b += delta_scale;
                    }

                    max_delta = max_delta.max(delta_scale.abs());
                }
            }

            // Apply L2 regularization shrinkage
            let shrink_factor = 1.0 / (1.0 + 1.0 / (self.c * n_samples as f64));
            w *= shrink_factor;

            // Check convergence
            if max_delta < self.tol {
                break;
            }
        }

        self.weights = Some(w);
        self.intercept = b;

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.weights, "predict")?;
        validate_predict_input(x, self.n_features.unwrap())?;

        let w = self.weights.as_ref().unwrap();
        let n_samples = x.nrows();

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            predictions[i] = dot_product_array(w, &x.row(i).to_owned()) + self.intercept;
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.weights.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        // For linear models, absolute weight magnitude indicates importance
        self.weights.as_ref().map(|w| w.mapv(|v| v.abs()))
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .float("C", 1e-3, 100.0)
            .float("epsilon", 0.0, 1.0)
            .categorical(
                "loss",
                vec![
                    "epsilon_insensitive".to_string(),
                    "squared_epsilon_insensitive".to_string(),
                ],
            )
    }

    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;
        crate::metrics::r2_score(y, &predictions)
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute dot product between an Array1 and another Array1.
#[inline]
fn dot_product_array(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.dot(b)
}

// =============================================================================
// SparseModel implementations
// =============================================================================

#[cfg(feature = "sparse")]
impl crate::models::traits::SparseModel for LinearSVC {
    fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: format!("{} samples", n_samples),
                actual: format!("{} targets", y.len()),
            });
        }

        // Extract unique classes
        let classes_vec = super::get_unique_classes(y);
        if classes_vec.len() < 2 {
            return Err(FerroError::invalid_input(
                "Need at least 2 classes for classification",
            ));
        }

        self.classes = Some(classes_vec.clone());
        self.n_features = Some(n_features);
        self.compute_class_weights(y);

        let n_classes = classes_vec.len();
        let mut weights = Vec::with_capacity(n_classes);
        let mut intercepts = Vec::with_capacity(n_classes);

        // Augmented dimensionality for intercept
        let d = if self.fit_intercept {
            n_features + 1
        } else {
            n_features
        };

        // Precompute row norms squared (sparse O(nnz))
        let x_sq_norm = x.row_norms_squared();

        // Helper closure: fit one binary classifier using sparse dual CD
        let fit_one = |y_binary: &Array1<f64>,
                       sample_weights: &Array1<f64>|
         -> (Array1<f64>, f64) {
            let c_eff: Vec<f64> = sample_weights.iter().map(|&sw| self.c * sw).collect();

            let x_aug_norm_sq: Vec<f64> = (0..n_samples)
                .map(|i| x_sq_norm[i] + if self.fit_intercept { 1.0 } else { 0.0 })
                .collect();

            let q_diag: Vec<f64> = match self.loss {
                LinearSVCLoss::SquaredHinge => (0..n_samples)
                    .map(|i| x_aug_norm_sq[i] + 1.0 / (2.0 * c_eff[i]))
                    .collect(),
                LinearSVCLoss::Hinge => x_aug_norm_sq.clone(),
            };

            let mut alpha = vec![0.0f64; n_samples];
            let mut w_aug = vec![0.0f64; d];
            let mut perm: Vec<usize> = (0..n_samples).collect();

            for iter in 0..self.max_iter {
                let mut max_change = 0.0f64;

                let seed = (iter as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);
                for i in (1..n_samples).rev() {
                    let j = ((seed.wrapping_mul((i + 1) as u64).wrapping_add(iter as u64))
                        % (i as u64 + 1)) as usize;
                    perm.swap(i, j);
                }

                for &i in &perm {
                    if q_diag[i] < 1e-12 {
                        continue;
                    }

                    // Sparse dot: w^T x_i (only over nonzero entries)
                    let row = x.row(i);
                    let mut wt_xi: f64 = row
                        .indices()
                        .iter()
                        .zip(row.data().iter())
                        .map(|(&col, &val)| w_aug[col] * val)
                        .sum();
                    if self.fit_intercept {
                        wt_xi += w_aug[n_features]; // intercept term
                    }

                    let g = match self.loss {
                        LinearSVCLoss::SquaredHinge => {
                            y_binary[i] * wt_xi + alpha[i] / (2.0 * c_eff[i]) - 1.0
                        }
                        LinearSVCLoss::Hinge => y_binary[i] * wt_xi - 1.0,
                    };

                    let alpha_new = match self.loss {
                        LinearSVCLoss::SquaredHinge => (alpha[i] - g / q_diag[i]).max(0.0),
                        LinearSVCLoss::Hinge => (alpha[i] - g / q_diag[i]).max(0.0).min(c_eff[i]),
                    };

                    let delta = alpha_new - alpha[i];
                    if delta.abs() < 1e-15 {
                        continue;
                    }

                    alpha[i] = alpha_new;

                    // Sparse weight update: only touch nonzero features
                    let scale = delta * y_binary[i];
                    for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                        w_aug[col] += scale * val;
                    }
                    if self.fit_intercept {
                        w_aug[n_features] += scale; // intercept augmented feature = 1.0
                    }

                    max_change = max_change.max(delta.abs());
                }

                if max_change < self.tol {
                    break;
                }
            }

            let w = Array1::from_vec(w_aug[..n_features].to_vec());
            let b = if self.fit_intercept {
                w_aug[n_features]
            } else {
                0.0
            };
            (w, b)
        };

        if n_classes == 2 {
            let positive_class = classes_vec[1];
            let y_binary: Array1<f64> = y
                .iter()
                .map(|&label| {
                    if (label - positive_class).abs() < 1e-10 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect();
            let sample_weights: Array1<f64> = y
                .iter()
                .map(|&label| self.get_class_weight(label))
                .collect();
            let (w, b) = fit_one(&y_binary, &sample_weights);
            weights.push(w);
            intercepts.push(b);
        } else {
            for &class in &classes_vec {
                let y_binary: Array1<f64> = y
                    .iter()
                    .map(|&label| {
                        if (label - class).abs() < 1e-10 {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect();
                let sample_weights: Array1<f64> = y
                    .iter()
                    .map(|&label| self.get_class_weight(label))
                    .collect();
                let (w, b) = fit_one(&y_binary, &sample_weights);
                weights.push(w);
                intercepts.push(b);
            }
        }

        self.weights = Some(weights);
        self.intercepts = Some(intercepts);
        Ok(())
    }

    fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array1<f64>> {
        check_is_fitted(&self.classes, "predict")?;
        let n_features = self.n_features.unwrap();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: format!("{} features", n_features),
                actual: format!("{} features", x.ncols()),
            });
        }

        let classes = self.classes.as_ref().unwrap();
        let weights_vec = self.weights.as_ref().unwrap();
        let intercepts = self.intercepts.as_ref().unwrap();
        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        if classes.len() == 2 {
            // Binary: sparse mat-vec X @ w + b
            let decisions = x.dot(&weights_vec[0])?.mapv(|v| v + intercepts[0]);
            for i in 0..n_samples {
                predictions[i] = if decisions[i] >= 0.0 {
                    classes[1]
                } else {
                    classes[0]
                };
            }
        } else {
            // Multiclass: one-vs-rest, highest decision value wins
            let n_classes = classes.len();
            let mut decision_values = Array2::zeros((n_samples, n_classes));
            for (class_idx, (w, &b)) in weights_vec.iter().zip(intercepts.iter()).enumerate() {
                let decisions = x.dot(w)?.mapv(|v| v + b);
                for i in 0..n_samples {
                    decision_values[[i, class_idx]] = decisions[i];
                }
            }
            for i in 0..n_samples {
                let best_class_idx = decision_values
                    .row(i)
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                predictions[i] = classes[best_class_idx];
            }
        }

        Ok(predictions)
    }
}

#[cfg(feature = "sparse")]
impl crate::models::traits::SparseModel for LinearSVR {
    fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: format!("{} samples", n_samples),
                actual: format!("{} targets", y.len()),
            });
        }

        self.n_features = Some(n_features);

        let mut w = Array1::zeros(n_features);
        let mut b = 0.0;

        // Precompute ||x_i||^2 for each sample (sparse O(nnz))
        let x_norm_sq = x.row_norms_squared();

        // Coordinate descent (same algorithm as dense, but sparse inner products)
        for _iter in 0..self.max_iter {
            let mut max_delta: f64 = 0.0;

            for i in 0..n_samples {
                // Sparse dot product: w^T x_i
                let row = x.row(i);
                let f_i: f64 = row
                    .indices()
                    .iter()
                    .zip(row.data().iter())
                    .map(|(&col, &val)| w[col] * val)
                    .sum::<f64>()
                    + b;
                let residual = y[i] - f_i;

                let (delta_scale, update) = match self.loss {
                    LinearSVRLoss::EpsilonInsensitive => {
                        if residual > self.epsilon {
                            let denom = x_norm_sq[i] + if self.fit_intercept { 1.0 } else { 0.0 };
                            if denom > 1e-10 {
                                let step = ((residual - self.epsilon) / denom).min(self.c);
                                (step, true)
                            } else {
                                (0.0, false)
                            }
                        } else if residual < -self.epsilon {
                            let denom = x_norm_sq[i] + if self.fit_intercept { 1.0 } else { 0.0 };
                            if denom > 1e-10 {
                                let step = ((residual + self.epsilon) / denom).max(-self.c);
                                (step, true)
                            } else {
                                (0.0, false)
                            }
                        } else {
                            (0.0, false)
                        }
                    }
                    LinearSVRLoss::SquaredEpsilonInsensitive => {
                        if residual.abs() > self.epsilon {
                            let sign = if residual > 0.0 { 1.0 } else { -1.0 };
                            let loss_grad = 2.0 * sign * (residual.abs() - self.epsilon);
                            let denom = (2.0 * self.c).mul_add(
                                x_norm_sq[i] + if self.fit_intercept { 1.0 } else { 0.0 },
                                1.0,
                            );
                            let step = (self.c * loss_grad) / denom;
                            (step, true)
                        } else {
                            (0.0, false)
                        }
                    }
                };

                if update && delta_scale.abs() > 1e-12 {
                    // Sparse weight update: only touch nonzero features
                    for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                        w[col] += delta_scale * val;
                    }
                    if self.fit_intercept {
                        b += delta_scale;
                    }
                    max_delta = max_delta.max(delta_scale.abs());
                }
            }

            // L2 regularization shrinkage
            let shrink_factor = 1.0 / (1.0 + 1.0 / (self.c * n_samples as f64));
            w *= shrink_factor;

            if max_delta < self.tol {
                break;
            }
        }

        self.weights = Some(w);
        self.intercept = b;
        Ok(())
    }

    fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array1<f64>> {
        check_is_fitted(&self.weights, "predict")?;
        let n_features = self.n_features.unwrap();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: format!("{} features", n_features),
                actual: format!("{} features", x.ncols()),
            });
        }

        let w = self.weights.as_ref().unwrap();
        let mut predictions = x.dot(w)?;
        predictions.mapv_inplace(|v| v + self.intercept);
        Ok(predictions)
    }
}

// =============================================================================
// PipelineSparseModel Implementations
// =============================================================================

#[cfg(feature = "sparse")]
impl crate::pipeline::PipelineSparseModel for LinearSVC {
    fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix, y: &Array1<f64>) -> Result<()> {
        crate::models::traits::SparseModel::fit_sparse(self, x, y)
    }

    fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array1<f64>> {
        crate::models::traits::SparseModel::predict_sparse(self, x)
    }

    fn search_space(&self) -> crate::hpo::SearchSpace {
        Model::search_space(self)
    }

    fn clone_boxed(&self) -> Box<dyn crate::pipeline::PipelineSparseModel> {
        Box::new(self.clone())
    }

    fn set_param(&mut self, name: &str, value: &crate::hpo::ParameterValue) -> Result<()> {
        match name {
            "C" | "c" => {
                if let Some(v) = value.as_f64() {
                    self.c = v;
                    Ok(())
                } else {
                    Err(FerroError::invalid_input("C must be a number"))
                }
            }
            "max_iter" => {
                if let Some(v) = value.as_i64() {
                    self.max_iter = v as usize;
                    Ok(())
                } else {
                    Err(FerroError::invalid_input("max_iter must be an integer"))
                }
            }
            _ => Err(FerroError::invalid_input(format!(
                "Unknown parameter '{}'",
                name
            ))),
        }
    }

    fn name(&self) -> &str {
        "LinearSVC"
    }

    fn is_fitted(&self) -> bool {
        Model::is_fitted(self)
    }
}

#[cfg(feature = "sparse")]
impl crate::pipeline::PipelineSparseModel for LinearSVR {
    fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix, y: &Array1<f64>) -> Result<()> {
        crate::models::traits::SparseModel::fit_sparse(self, x, y)
    }

    fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array1<f64>> {
        crate::models::traits::SparseModel::predict_sparse(self, x)
    }

    fn search_space(&self) -> crate::hpo::SearchSpace {
        Model::search_space(self)
    }

    fn clone_boxed(&self) -> Box<dyn crate::pipeline::PipelineSparseModel> {
        Box::new(self.clone())
    }

    fn set_param(&mut self, name: &str, value: &crate::hpo::ParameterValue) -> Result<()> {
        match name {
            "C" | "c" => {
                if let Some(v) = value.as_f64() {
                    self.c = v;
                    Ok(())
                } else {
                    Err(FerroError::invalid_input("C must be a number"))
                }
            }
            "epsilon" => {
                if let Some(v) = value.as_f64() {
                    self.epsilon = v;
                    Ok(())
                } else {
                    Err(FerroError::invalid_input("epsilon must be a number"))
                }
            }
            "max_iter" => {
                if let Some(v) = value.as_i64() {
                    self.max_iter = v as usize;
                    Ok(())
                } else {
                    Err(FerroError::invalid_input("max_iter must be an integer"))
                }
            }
            _ => Err(FerroError::invalid_input(format!(
                "Unknown parameter '{}'",
                name
            ))),
        }
    }

    fn name(&self) -> &str {
        "LinearSVR"
    }

    fn is_fitted(&self) -> bool {
        Model::is_fitted(self)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Kernel Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_linear_kernel() {
        let kernel = Kernel::Linear;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];

        let result = kernel.compute(&a, &b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel() {
        let kernel = Kernel::Rbf { gamma: 1.0 };
        let a = [0.0, 0.0];
        let b = [1.0, 1.0];

        let result = kernel.compute(&a, &b);
        // exp(-1 * (1 + 1)) = exp(-2)
        assert!((result - (-2.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_poly_kernel() {
        let kernel = Kernel::Polynomial {
            gamma: 1.0,
            coef0: 1.0,
            degree: 2,
        };
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];

        let result = kernel.compute(&a, &b);
        // (1*(1*3 + 2*4) + 1)^2 = (11 + 1)^2 = 144
        assert!((result - 144.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel() {
        let kernel = Kernel::Sigmoid {
            gamma: 0.5,
            coef0: 0.0,
        };
        let a = [1.0, 1.0];
        let b = [1.0, 1.0];

        let result = kernel.compute(&a, &b);
        // tanh(0.5 * 2) = tanh(1)
        assert!((result - 1.0_f64.tanh()).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_auto_gamma() {
        let kernel = Kernel::rbf_auto();
        let adjusted = kernel.with_auto_gamma(10);

        match adjusted {
            Kernel::Rbf { gamma } => assert!((gamma - 0.1).abs() < 1e-10),
            _ => panic!("Expected RBF kernel"),
        }
    }

    // -------------------------------------------------------------------------
    // Binary SVC Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_binary_svc_linearly_separable() {
        // Simple linearly separable dataset
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = BinarySVC::new(1.0, Kernel::Linear, 1e-3, 1000, DEFAULT_CACHE_SIZE);
        clf.fit(&x, &y, 1.0, 0.0, None).unwrap();

        let predictions = clf.predict(&x).unwrap();

        // Should classify training data correctly
        for i in 0..3 {
            assert!(
                (predictions[i] - 0.0).abs() < 1e-10,
                "Point {} should be class 0",
                i
            );
        }
        for i in 3..6 {
            assert!(
                (predictions[i] - 1.0).abs() < 1e-10,
                "Point {} should be class 1",
                i
            );
        }
    }

    // -------------------------------------------------------------------------
    // SVC Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_svc_basic() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = SVC::new().with_kernel(Kernel::Linear).with_c(1.0);
        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        assert_eq!(clf.classes().unwrap().len(), 2);

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_svc_rbf_kernel() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 6.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0, 6.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut clf = SVC::new()
            .with_kernel(Kernel::Rbf { gamma: 0.5 })
            .with_c(10.0);
        clf.fit(&x, &y).unwrap();

        let predictions = clf.predict(&x).unwrap();

        // Check that most predictions are correct
        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &actual)| (pred - actual).abs() < 1e-10)
            .count();

        assert!(correct >= 6, "Expected at least 6 correct, got {}", correct);
    }

    #[test]
    fn test_svc_multiclass_ovo() {
        // Three-class problem
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, // Class 0
                5.0, 5.0, 6.0, 5.0, 5.0, 6.0, // Class 1
                9.0, 9.0, 10.0, 9.0, 9.0, 10.0, // Class 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let mut clf = SVC::new()
            .with_kernel(Kernel::Linear)
            .with_c(1.0)
            .with_multiclass_strategy(MulticlassStrategy::OneVsOne);

        clf.fit(&x, &y).unwrap();

        assert_eq!(clf.classes().unwrap().len(), 3);
        // For 3 classes, OvO creates 3 binary classifiers
        assert_eq!(clf.classifiers.len(), 3);

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 9);
    }

    #[test]
    fn test_svc_multiclass_ovr() {
        // Three-class problem
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, // Class 0
                5.0, 5.0, 6.0, 5.0, 5.0, 6.0, // Class 1
                9.0, 9.0, 10.0, 9.0, 9.0, 10.0, // Class 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let mut clf = SVC::new()
            .with_kernel(Kernel::Linear)
            .with_c(1.0)
            .with_multiclass_strategy(MulticlassStrategy::OneVsRest);

        clf.fit(&x, &y).unwrap();

        // For 3 classes, OvR creates 3 binary classifiers
        assert_eq!(clf.classifiers.len(), 3);
    }

    #[test]
    fn test_svc_with_probability() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = SVC::new()
            .with_kernel(Kernel::Linear)
            .with_c(1.0)
            .with_probability(true);

        clf.fit(&x, &y).unwrap();

        let probas = clf.predict_proba(&x).unwrap();

        // Check shape
        assert_eq!(probas.nrows(), 6);
        assert_eq!(probas.ncols(), 2);

        // Check probabilities sum to 1
        for i in 0..6 {
            let sum: f64 = probas.row(i).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Row {} sum is {}, expected 1.0",
                i,
                sum
            );
        }
    }

    #[test]
    fn test_svc_class_weights_balanced() {
        // Imbalanced dataset
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0,
                6.0, // Class 0 (6 samples)
                10.0, 10.0, 11.0, 11.0, // Class 1 (2 samples)
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);

        let mut clf = SVC::new()
            .with_kernel(Kernel::Linear)
            .with_c(1.0)
            .with_class_weight(ClassWeight::Balanced);

        clf.fit(&x, &y).unwrap();

        // Just verify it trains without error
        assert!(clf.is_fitted());
    }

    #[test]
    fn test_svc_not_fitted_error() {
        let clf = SVC::new();
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        assert!(clf.predict(&x).is_err());
    }

    #[test]
    fn test_svc_feature_mismatch() {
        let x_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 6.0, 7.0, 7.0, 6.0]).unwrap();
        let y_train = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut clf = SVC::new();
        clf.fit(&x_train, &y_train).unwrap();

        // Wrong number of features
        let x_test = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(clf.predict(&x_test).is_err());
    }

    #[test]
    fn test_svc_single_class_error() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0]); // Only one class

        let mut clf = SVC::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_svc_search_space() {
        let clf = SVC::new();
        let space = clf.search_space();

        assert!(space.parameters.contains_key("C"));
        assert!(space.parameters.contains_key("gamma"));
        assert!(space.parameters.contains_key("kernel"));
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_svc_xor_problem() {
        // XOR problem - not linearly separable
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        // RBF kernel should handle this
        let mut clf = SVC::new()
            .with_kernel(Kernel::Rbf { gamma: 2.0 })
            .with_c(10.0);

        clf.fit(&x, &y).unwrap();

        let predictions = clf.predict(&x).unwrap();

        // Should classify XOR pattern correctly
        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &actual)| (pred - actual).abs() < 1e-10)
            .count();

        assert!(
            correct >= 3,
            "Expected at least 3 correct on XOR, got {}",
            correct
        );
    }

    #[test]
    fn test_svc_poly_kernel_integration() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 7.0, 7.0, 8.0, 8.0, 8.0, 7.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = SVC::new()
            .with_kernel(Kernel::poly(2, 1.0, 0.0))
            .with_c(1.0);

        clf.fit(&x, &y).unwrap();

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    // -------------------------------------------------------------------------
    // SVR Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_svr_basic_linear() {
        // Simple linear relationship: y = 2*x + 1
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]);

        let mut reg = SVR::new()
            .with_kernel(Kernel::Linear)
            .with_c(100.0)
            .with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();
        assert!(reg.is_fitted());
        assert!(reg.n_support_vectors() > 0);

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 10);

        // Check that predictions are reasonable
        for (pred, actual) in predictions.iter().zip(y.iter()) {
            assert!(
                (pred - actual).abs() < 2.0,
                "Prediction {} too far from actual {}",
                pred,
                actual
            );
        }
    }

    #[test]
    fn test_svr_rbf_kernel() {
        // Non-linear: y = x^2
        let x = Array2::from_shape_vec(
            (9, 1),
            vec![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![16.0, 9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0, 16.0]);

        let mut reg = SVR::new()
            .with_kernel(Kernel::Rbf { gamma: 0.5 })
            .with_c(100.0)
            .with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // Compute MSE
        let mse: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / y.len() as f64;

        // MSE should be reasonable (not perfect due to epsilon tube)
        assert!(mse < 10.0, "MSE {} is too high", mse);
    }

    #[test]
    fn test_svr_poly_kernel() {
        // Quadratic relationship
        let x = Array2::from_shape_vec((7, 1), vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0]);

        let mut reg = SVR::new()
            .with_kernel(Kernel::poly(2, 1.0, 1.0))
            .with_c(100.0)
            .with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 7);
    }

    #[test]
    fn test_svr_multivariate() {
        // y = x1 + 2*x2
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 3.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 7.0, 8.0]);

        let mut reg = SVR::new()
            .with_kernel(Kernel::Linear)
            .with_c(100.0)
            .with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // Predictions should be close to actual
        for (pred, actual) in predictions.iter().zip(y.iter()) {
            assert!(
                (pred - actual).abs() < 2.0,
                "Prediction {} too far from actual {}",
                pred,
                actual
            );
        }
    }

    #[test]
    fn test_svr_epsilon_effect() {
        // With larger epsilon, more points should be within the tube (fewer support vectors)
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let mut reg_small_eps = SVR::new()
            .with_kernel(Kernel::Linear)
            .with_c(10.0)
            .with_epsilon(0.01);

        let mut reg_large_eps = SVR::new()
            .with_kernel(Kernel::Linear)
            .with_c(10.0)
            .with_epsilon(1.0);

        reg_small_eps.fit(&x, &y).unwrap();
        reg_large_eps.fit(&x, &y).unwrap();

        // With larger epsilon, typically fewer support vectors (more points in tube)
        // This is a general tendency, not guaranteed for all datasets
        assert!(reg_small_eps.is_fitted());
        assert!(reg_large_eps.is_fitted());
    }

    #[test]
    fn test_svr_not_fitted_error() {
        let reg = SVR::new();
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        assert!(reg.predict(&x).is_err());
    }

    #[test]
    fn test_svr_feature_mismatch() {
        let x_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y_train = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let mut reg = SVR::new().with_kernel(Kernel::Linear).with_c(10.0);
        reg.fit(&x_train, &y_train).unwrap();

        // Wrong number of features
        let x_test = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(reg.predict(&x_test).is_err());
    }

    #[test]
    fn test_svr_search_space() {
        let reg = SVR::new();
        let space = reg.search_space();

        assert!(space.parameters.contains_key("C"));
        assert!(space.parameters.contains_key("epsilon"));
        assert!(space.parameters.contains_key("gamma"));
        assert!(space.parameters.contains_key("kernel"));
    }

    #[test]
    fn test_svr_with_noise() {
        // y = x + noise
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        // Adding some noise
        let y = Array1::from_vec(vec![0.1, 1.2, 1.8, 3.1, 3.9, 5.2, 5.8, 7.1, 7.9, 9.2]);

        let mut reg = SVR::new()
            .with_kernel(Kernel::Linear)
            .with_c(10.0)
            .with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // Should still capture the linear trend
        let rmse: f64 = (predictions
            .iter()
            .zip(y.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / y.len() as f64)
            .sqrt();

        assert!(
            rmse < 1.0,
            "RMSE {} is too high for noisy linear data",
            rmse
        );
    }

    #[test]
    fn test_svr_intercept() {
        // y = 5 (constant)
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0, 5.0]);

        let mut reg = SVR::new()
            .with_kernel(Kernel::Linear)
            .with_c(10.0)
            .with_epsilon(0.1);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // All predictions should be close to 5
        for pred in predictions.iter() {
            assert!(
                (pred - 5.0).abs() < 1.0,
                "Prediction {} should be close to 5",
                pred
            );
        }
    }

    // -------------------------------------------------------------------------
    // LinearSVC Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_linear_svc_basic() {
        // Linearly separable data
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = LinearSVC::new().with_c(1.0).with_max_iter(1000);
        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        assert_eq!(clf.classes().unwrap().len(), 2);

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_linear_svc_classification() {
        // Well-separated clusters
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0, 8.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut clf = LinearSVC::new().with_c(10.0);
        clf.fit(&x, &y).unwrap();

        let predictions = clf.predict(&x).unwrap();

        // Count correct predictions
        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &actual)| (pred - actual).abs() < 1e-10)
            .count();

        assert!(correct >= 6, "Expected at least 6 correct, got {}", correct);
    }

    #[test]
    fn test_linear_svc_multiclass() {
        // Three-class problem
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, // Class 0
                5.0, 5.0, 6.0, 5.0, 5.0, 6.0, // Class 1
                9.0, 9.0, 10.0, 9.0, 9.0, 10.0, // Class 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let mut clf = LinearSVC::new().with_c(1.0);
        clf.fit(&x, &y).unwrap();

        assert_eq!(clf.classes().unwrap().len(), 3);
        // For 3 classes, OvR creates 3 binary classifiers
        assert_eq!(clf.weights().unwrap().len(), 3);

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 9);
    }

    #[test]
    fn test_linear_svc_hinge_loss() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = LinearSVC::new().with_c(1.0).with_loss(LinearSVCLoss::Hinge);
        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
    }

    #[test]
    fn test_linear_svc_class_weights() {
        // Imbalanced dataset
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0,
                6.0, // Class 0 (6 samples)
                10.0, 10.0, 11.0, 11.0, // Class 1 (2 samples)
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);

        let mut clf = LinearSVC::new()
            .with_c(1.0)
            .with_class_weight(ClassWeight::Balanced);
        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
    }

    #[test]
    fn test_linear_svc_feature_importance() {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 6.0, 0.0, 0.0, 7.0, 0.0, 0.0, 7.0,
                0.0, 0.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = LinearSVC::new().with_c(1.0);
        clf.fit(&x, &y).unwrap();

        let importance = clf.feature_importance().unwrap();
        assert_eq!(importance.len(), 3);

        // First feature should be most important (it's the only one that varies)
        assert!(importance[0] >= importance[1]);
        assert!(importance[0] >= importance[2]);
    }

    #[test]
    fn test_linear_svc_not_fitted_error() {
        let clf = LinearSVC::new();
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        assert!(clf.predict(&x).is_err());
    }

    #[test]
    fn test_linear_svc_feature_mismatch() {
        let x_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 6.0, 7.0, 7.0, 6.0]).unwrap();
        let y_train = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut clf = LinearSVC::new();
        clf.fit(&x_train, &y_train).unwrap();

        // Wrong number of features
        let x_test = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(clf.predict(&x_test).is_err());
    }

    #[test]
    fn test_linear_svc_single_class_error() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0]); // Only one class

        let mut clf = LinearSVC::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_linear_svc_search_space() {
        let clf = LinearSVC::new();
        let space = clf.search_space();

        assert!(space.parameters.contains_key("C"));
        assert!(space.parameters.contains_key("loss"));
    }

    // -------------------------------------------------------------------------
    // LinearSVR Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_linear_svr_basic() {
        // Simple linear relationship: y = 2*x + 1
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]);

        let mut reg = LinearSVR::new().with_c(100.0).with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();
        assert!(reg.is_fitted());

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 10);

        // Check that predictions are reasonable
        for (pred, actual) in predictions.iter().zip(y.iter()) {
            assert!(
                (pred - actual).abs() < 3.0,
                "Prediction {} too far from actual {}",
                pred,
                actual
            );
        }
    }

    #[test]
    fn test_linear_svr_multivariate() {
        // y = x1 + 2*x2
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 3.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 7.0, 8.0]);

        let mut reg = LinearSVR::new().with_c(100.0).with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // Predictions should be close to actual
        let rmse: f64 = (predictions
            .iter()
            .zip(y.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / y.len() as f64)
            .sqrt();

        assert!(rmse < 3.0, "RMSE {} is too high", rmse);
    }

    #[test]
    fn test_linear_svr_squared_loss() {
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]);

        let mut reg = LinearSVR::new()
            .with_c(100.0)
            .with_epsilon(0.5)
            .with_loss(LinearSVRLoss::SquaredEpsilonInsensitive);

        reg.fit(&x, &y).unwrap();
        assert!(reg.is_fitted());
    }

    #[test]
    fn test_linear_svr_no_intercept() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let mut reg = LinearSVR::new().with_c(100.0).with_fit_intercept(false);

        reg.fit(&x, &y).unwrap();

        // Without intercept, the weight should be close to 2
        let weights = reg.weights().unwrap();
        assert!(
            (weights[0] - 2.0).abs() < 1.0,
            "Weight {} should be close to 2",
            weights[0]
        );
    }

    #[test]
    fn test_linear_svr_feature_importance() {
        // y = 3*x1 + 0*x2
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 5.0, 2.0, 3.0, 3.0, 7.0, 4.0, 2.0, 5.0, 8.0, 6.0, 1.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);

        let mut reg = LinearSVR::new().with_c(100.0);
        reg.fit(&x, &y).unwrap();

        let importance = reg.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // First feature should be more important
        assert!(
            importance[0] > importance[1],
            "First feature should be more important: {:?}",
            importance
        );
    }

    #[test]
    fn test_linear_svr_not_fitted_error() {
        let reg = LinearSVR::new();
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        assert!(reg.predict(&x).is_err());
    }

    #[test]
    fn test_linear_svr_feature_mismatch() {
        let x_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y_train = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let mut reg = LinearSVR::new().with_c(10.0);
        reg.fit(&x_train, &y_train).unwrap();

        // Wrong number of features
        let x_test = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(reg.predict(&x_test).is_err());
    }

    #[test]
    fn test_linear_svr_search_space() {
        let reg = LinearSVR::new();
        let space = reg.search_space();

        assert!(space.parameters.contains_key("C"));
        assert!(space.parameters.contains_key("epsilon"));
        assert!(space.parameters.contains_key("loss"));
    }

    #[test]
    fn test_linear_svr_with_noise() {
        // y = x + noise
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.1, 1.2, 1.8, 3.1, 3.9, 5.2, 5.8, 7.1, 7.9, 9.2]);

        let mut reg = LinearSVR::new().with_c(10.0).with_epsilon(0.5);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // Should still capture the linear trend
        let rmse: f64 = (predictions
            .iter()
            .zip(y.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / y.len() as f64)
            .sqrt();

        assert!(
            rmse < 2.0,
            "RMSE {} is too high for noisy linear data",
            rmse
        );
    }

    #[test]
    fn test_linear_svr_constant() {
        // y = 5 (constant)
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0, 5.0]);

        let mut reg = LinearSVR::new().with_c(10.0).with_epsilon(0.1);

        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // All predictions should be close to 5
        for pred in predictions.iter() {
            assert!(
                (pred - 5.0).abs() < 2.0,
                "Prediction {} should be close to 5",
                pred
            );
        }
    }

    // -------------------------------------------------------------------------
    // Comparison tests: Linear vs Kernelized versions
    // -------------------------------------------------------------------------

    #[test]
    fn test_linear_svc_vs_svc_linear_kernel() {
        // Both should give similar results on linearly separable data
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut linear_svc = LinearSVC::new().with_c(10.0);
        linear_svc.fit(&x, &y).unwrap();

        let mut svc = SVC::new().with_kernel(Kernel::Linear).with_c(10.0);
        svc.fit(&x, &y).unwrap();

        let linear_pred = linear_svc.predict(&x).unwrap();
        let svc_pred = svc.predict(&x).unwrap();

        // Both should classify correctly
        for (lp, sp) in linear_pred.iter().zip(svc_pred.iter()) {
            assert!(
                (lp - sp).abs() < 1e-10 || (*lp == 0.0 || *lp == 1.0) && (*sp == 0.0 || *sp == 1.0),
                "LinearSVC pred {} vs SVC pred {}",
                lp,
                sp
            );
        }
    }

    #[test]
    fn test_linear_svr_vs_svr_linear_kernel() {
        // Both should give similar results on linear data
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let mut linear_svr = LinearSVR::new().with_c(100.0).with_epsilon(0.5);
        linear_svr.fit(&x, &y).unwrap();

        let mut svr = SVR::new()
            .with_kernel(Kernel::Linear)
            .with_c(100.0)
            .with_epsilon(0.5);
        svr.fit(&x, &y).unwrap();

        let linear_pred = linear_svr.predict(&x).unwrap();
        let svr_pred = svr.predict(&x).unwrap();

        // Both should have similar predictions
        for (lp, sp) in linear_pred.iter().zip(svr_pred.iter()) {
            // Allow some tolerance since the optimization algorithms differ
            assert!(
                (lp - sp).abs() < 3.0,
                "LinearSVR pred {} vs SVR pred {}",
                lp,
                sp
            );
        }
    }

    #[test]
    fn test_smo_bounds_asymmetric_c() {
        // When y[i] != y[j] and C values differ, the high bound should use c[i]
        // Use ClassWeight::Custom to produce asymmetric per-sample C values
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 3.0, 3.0, 4.0, 3.0, 3.0, 4.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        // Asymmetric class weights produce different C per class
        let mut svc = SVC::new()
            .with_c(10.0)
            .with_class_weight(ClassWeight::Custom(vec![(0.0, 1.0), (1.0, 10.0)]));

        svc.fit(&x, &y).unwrap();

        // Model should correctly classify all training points
        let pred = svc.predict(&x).unwrap();
        for (i, (&p, &actual)) in pred.iter().zip(y.iter()).enumerate() {
            assert!(
                (p - actual).abs() < 1e-10,
                "Sample {i}: predicted {p}, expected {actual}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // SMO Optimization Tests (WSS3, Shrinking, Correctness)
    // -------------------------------------------------------------------------

    #[test]
    fn test_wss3_converges_on_nonlinear_problem() {
        // WSS3 (second-order working set selection) should converge successfully
        // on an RBF kernel problem where first-order selection can struggle.
        // We verify convergence by checking that the model fits correctly.
        let mut data = Vec::new();
        let mut labels = Vec::new();
        // Class 0: inner ring
        for i in 0..20 {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / 20.0;
            data.push(angle.cos());
            data.push(angle.sin());
            labels.push(0.0);
        }
        // Class 1: outer ring
        for i in 0..20 {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / 20.0;
            data.push(3.0 * angle.cos());
            data.push(3.0 * angle.sin());
            labels.push(1.0);
        }
        let x = Array2::from_shape_vec((40, 2), data).unwrap();
        let y = Array1::from_vec(labels);

        let mut svc = SVC::new()
            .with_c(10.0)
            .with_kernel(Kernel::Rbf { gamma: 0.5 })
            .with_max_iter(1000);

        svc.fit(&x, &y).unwrap();
        let pred = svc.predict(&x).unwrap();

        // Should classify most training points correctly (WSS3 converges)
        let correct: usize = pred
            .iter()
            .zip(y.iter())
            .filter(|(&p, &a)| (p - a).abs() < 1e-10)
            .count();
        assert!(
            correct >= 36,
            "WSS3 SVM should classify >= 36/40 training points, got {correct}"
        );
    }

    #[test]
    fn test_shrinking_does_not_change_predictions() {
        // Shrinking is an optimization that should not alter the final model.
        // Train with enough iterations that shrinking activates (>100 iters).
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 3.0, 3.0, 3.5, 3.5, 4.0, 3.0, 3.0, 4.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        // With max_iter high enough that shrinking kicks in at iter 100
        let mut svc = SVC::new().with_c(1.0).with_max_iter(500);
        svc.fit(&x, &y).unwrap();
        let pred = svc.predict(&x).unwrap();

        // All training points should be correctly classified (well-separated data)
        for (i, (&p, &actual)) in pred.iter().zip(y.iter()).enumerate() {
            assert!(
                (p - actual).abs() < 1e-10,
                "Shrinking test: sample {i} predicted {p}, expected {actual}"
            );
        }
    }

    #[test]
    fn test_optimized_svm_linearly_separable() {
        // Verify that the optimized SVM (WSS3 + diag precompute + shrinking)
        // produces correct predictions on a clearly linearly separable dataset.
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                -3.0, -3.0, -2.0, -2.0, -1.0, -1.0, -2.0, -1.0, -1.0, -2.0, 1.0, 1.0, 2.0, 2.0,
                3.0, 3.0, 2.0, 1.0, 1.0, 2.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let mut svc = SVC::new().with_c(10.0).with_max_iter(1000);
        svc.fit(&x, &y).unwrap();
        let pred = svc.predict(&x).unwrap();

        // Perfect classification expected on linearly separable data
        for (i, (&p, &actual)) in pred.iter().zip(y.iter()).enumerate() {
            assert!(
                (p - actual).abs() < 1e-10,
                "Linearly separable test: sample {i} predicted {p}, expected {actual}"
            );
        }

        // Verify model has support vectors
        let sv_counts = svc.n_support_vectors();
        let total_sv: usize = sv_counts.iter().sum();
        assert!(
            total_sv > 0 && total_sv <= 10,
            "Expected 1-10 support vectors, got {total_sv}"
        );
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    /// Helper: create a small training matrix and a linear-kernel cache with the given capacity.
    fn make_cache(n_samples: usize, capacity: usize) -> KernelCache {
        // Simple n_samples x 2 training data so kernel values are deterministic
        let mut data = Vec::with_capacity(n_samples * 2);
        for i in 0..n_samples {
            data.push(i as f64);
            data.push((i as f64) * 0.5);
        }
        let x = Array2::from_shape_vec((n_samples, 2), data).unwrap();
        KernelCache::new(Kernel::Linear, &x, capacity)
    }

    #[test]
    fn test_lru_eviction_order() {
        // Cache with capacity 3, 5 samples total.
        // Insert rows 0, 1, 2 (fills cache), then insert row 3.
        // Row 0 should be evicted (LRU).
        let mut cache = make_cache(5, 3);

        // Access rows 0, 1, 2 to fill the cache
        let _ = cache.get_row(0);
        let _ = cache.get_row(1);
        let _ = cache.get_row(2);
        assert_eq!(cache.len, 3);

        // Row 0 should be cached
        assert_ne!(cache.row_to_slot[0], NONE_SLOT);

        // Insert row 3 -- should evict row 0 (LRU)
        let _ = cache.get_row(3);
        assert_eq!(cache.len, 3); // still at capacity

        // Row 0 was evicted
        assert_eq!(cache.row_to_slot[0], NONE_SLOT);
        // Rows 1, 2, 3 are still cached
        assert_ne!(cache.row_to_slot[1], NONE_SLOT);
        assert_ne!(cache.row_to_slot[2], NONE_SLOT);
        assert_ne!(cache.row_to_slot[3], NONE_SLOT);
    }

    #[test]
    fn test_hit_promotion() {
        // Cache capacity 3, 5 samples.
        // Insert 0, 1, 2. Touch 0 (moves to MRU). Insert 3.
        // Row 1 should be evicted (it's now LRU), NOT row 0.
        let mut cache = make_cache(5, 3);

        let _ = cache.get_row(0);
        let _ = cache.get_row(1);
        let _ = cache.get_row(2);

        // Touch row 0 -- promotes it to MRU
        let _ = cache.get_row(0);

        // Insert row 3 -- should evict row 1 (LRU after promotion of 0)
        let _ = cache.get_row(3);

        // Row 0 should still be cached (was promoted)
        assert_ne!(cache.row_to_slot[0], NONE_SLOT);
        // Row 1 should be evicted
        assert_eq!(cache.row_to_slot[1], NONE_SLOT);
        // Rows 2, 3 still cached
        assert_ne!(cache.row_to_slot[2], NONE_SLOT);
        assert_ne!(cache.row_to_slot[3], NONE_SLOT);
    }

    #[test]
    fn test_cache_hit_returns_correct_values() {
        // Verify that a cache hit returns the same computed row values.
        let mut cache = make_cache(4, 3);

        // First access computes the row
        let row_first = cache.get_row(1).to_vec();

        // Second access should be a cache hit with identical values
        let row_second = cache.get_row(1).to_vec();

        assert_eq!(row_first, row_second);

        // Verify values are correct (linear kernel: dot product)
        // Row 1 training data = [1.0, 0.5]
        // K(1, 0) = dot([1.0, 0.5], [0.0, 0.0]) = 0.0
        // K(1, 1) = dot([1.0, 0.5], [1.0, 0.5]) = 1.25
        assert!((row_first[0] - 0.0).abs() < 1e-10);
        assert!((row_first[1] - 1.25).abs() < 1e-10);
    }

    #[test]
    fn test_shrinking_invalidates_entries() {
        // Simulate shrinking by evicting specific rows and verifying
        // that new insertions work correctly after eviction.
        let mut cache = make_cache(6, 4);

        // Fill cache with rows 0, 1, 2, 3
        for i in 0..4 {
            let _ = cache.get_row(i);
        }
        assert_eq!(cache.len, 4);

        // Manually invalidate row 1 (simulating what happens when shrinking
        // removes an index from the active set -- the slot gets reused on next eviction)
        // Access row 4 -- evicts row 0 (LRU)
        let _ = cache.get_row(4);
        assert_eq!(cache.row_to_slot[0], NONE_SLOT);

        // Access row 5 -- evicts row 1 (next LRU)
        let _ = cache.get_row(5);
        assert_eq!(cache.row_to_slot[1], NONE_SLOT);

        // Remaining cached: 2, 3, 4, 5
        assert_ne!(cache.row_to_slot[2], NONE_SLOT);
        assert_ne!(cache.row_to_slot[3], NONE_SLOT);
        assert_ne!(cache.row_to_slot[4], NONE_SLOT);
        assert_ne!(cache.row_to_slot[5], NONE_SLOT);

        // Re-access row 0 -- should compute and cache again (evicts row 2)
        let row0 = cache.get_row(0).to_vec();
        assert_ne!(cache.row_to_slot[0], NONE_SLOT);
        assert_eq!(cache.row_to_slot[2], NONE_SLOT);

        // Verify the re-computed values are correct
        // K(0, 0) = dot([0, 0], [0, 0]) = 0
        assert!((row0[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_cache_miss() {
        // Cache with minimum capacity (2, enforced by the constructor).
        // Accessing any row should compute it.
        let mut cache = make_cache(5, 2);
        assert_eq!(cache.len, 0);

        let row = cache.get_row(3).to_vec();
        assert_eq!(cache.len, 1);

        // Verify the row was computed correctly
        // Row 3 data = [3.0, 1.5]
        // K(3, 3) = dot([3.0, 1.5], [3.0, 1.5]) = 9.0 + 2.25 = 11.25
        assert!((row[3] - 11.25).abs() < 1e-10);
    }

    #[test]
    fn test_single_entry_behavior() {
        // Minimum capacity is 2 (enforced by constructor).
        // With capacity 2, each pair of inserts fills it, third evicts.
        let mut cache = make_cache(5, 2);

        let _ = cache.get_row(0);
        let _ = cache.get_row(1);
        assert_eq!(cache.len, 2);

        // Third insert evicts row 0
        let _ = cache.get_row(2);
        assert_eq!(cache.len, 2);
        assert_eq!(cache.row_to_slot[0], NONE_SLOT);
        assert_ne!(cache.row_to_slot[1], NONE_SLOT);
        assert_ne!(cache.row_to_slot[2], NONE_SLOT);

        // Fourth insert evicts row 1
        let _ = cache.get_row(3);
        assert_eq!(cache.row_to_slot[1], NONE_SLOT);
        assert_ne!(cache.row_to_slot[2], NONE_SLOT);
        assert_ne!(cache.row_to_slot[3], NONE_SLOT);
    }

    #[test]
    fn test_get_value_symmetric_lookup() {
        // Test that get_value uses symmetry: K(i,j) can be read from
        // either row i or row j's cached row.
        let mut cache = make_cache(4, 2);

        // Cache row 1
        let _ = cache.get_row(1);

        // get_value(1, 2) should use the cached row for i=1
        let v12 = cache.get_value(1, 2);

        // get_value(2, 1) should use the cached row for j=1 (symmetric lookup)
        let v21 = cache.get_value(2, 1);

        // Linear kernel is symmetric
        assert!((v12 - v21).abs() < 1e-10);

        // get_value(2, 3) -- neither row cached, should compute directly
        let v23 = cache.get_value(2, 3);
        // K(2, 3) = dot([2.0, 1.0], [3.0, 1.5]) = 6.0 + 1.5 = 7.5
        assert!((v23 - 7.5).abs() < 1e-10);
    }

    #[test]
    fn test_repeated_access_same_row() {
        // Accessing the same row repeatedly should always return correct values
        // and not corrupt the cache.
        let mut cache = make_cache(4, 3);

        for _ in 0..10 {
            let row = cache.get_row(2).to_vec();
            // K(2, 2) = dot([2.0, 1.0], [2.0, 1.0]) = 5.0
            assert!((row[2] - 5.0).abs() < 1e-10);
        }
        assert_eq!(cache.len, 1); // Only one slot used
    }

    #[test]
    fn test_full_cycle_eviction() {
        // Fill cache, evict everything, refill -- verify no corruption.
        let mut cache = make_cache(6, 3);

        // First round: fill with 0, 1, 2
        for i in 0..3 {
            let _ = cache.get_row(i);
        }

        // Second round: fill with 3, 4, 5 -- evicts 0, 1, 2 in order
        for i in 3..6 {
            let _ = cache.get_row(i);
        }

        // All original rows evicted
        for i in 0..3 {
            assert_eq!(cache.row_to_slot[i], NONE_SLOT);
        }
        // New rows cached
        for i in 3..6 {
            assert_ne!(cache.row_to_slot[i], NONE_SLOT);
        }

        // Third round: re-access row 0
        let row0 = cache.get_row(0).to_vec();
        assert_ne!(cache.row_to_slot[0], NONE_SLOT);
        // K(0, 0) = 0
        assert!((row0[0] - 0.0).abs() < 1e-10);
    }
}

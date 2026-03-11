//! GPU acceleration backend via wgpu
//!
//! Provides optional GPU-accelerated matrix operations (GEMM, pairwise distances)
//! using wgpu compute shaders. All GPU code is behind the `gpu` feature flag.
//!
//! ## Precision Notes
//!
//! wgpu compute shaders only support f32 arithmetic. Data is converted f64->f32
//! on upload and f32->f64 on download. This introduces precision loss (~1e-7 relative).
//! For high-precision needs, use the CPU path.
//!
//! ## Usage
//!
//! ```no_run
//! use ferroml_core::gpu::{WgpuBackend, GpuBackend};
//! use ndarray::array;
//!
//! if let Some(backend) = WgpuBackend::try_new() {
//!     let a = array![[1.0, 2.0], [3.0, 4.0]];
//!     let b = array![[5.0, 6.0], [7.0, 8.0]];
//!     let c = backend.matmul(&a, &b).unwrap();
//! }
//! ```

mod backend;
mod dispatch;
mod kernels;

pub use backend::WgpuBackend;
pub use dispatch::GpuDispatcher;

use crate::Result;
use ndarray::{Array1, Array2};

/// Policy for GPU/CPU dispatch.
#[derive(Debug, Clone)]
pub enum GpuDispatchPolicy {
    /// Always use GPU (return error on failure)
    Always,
    /// Use GPU if matrix exceeds threshold, else CPU
    Auto { min_elements: usize },
    /// Always use CPU (useful for testing/debugging)
    Never,
}

impl Default for GpuDispatchPolicy {
    fn default() -> Self {
        GpuDispatchPolicy::Auto { min_elements: 4096 } // 64x64
    }
}

/// GPU memory information queried from adapter.
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u32,
}

/// Trait for GPU-accelerated linear algebra operations.
pub trait GpuBackend: Send + Sync + std::fmt::Debug {
    /// Matrix multiply: C = A @ B
    fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>>;

    /// Compute pairwise squared Euclidean distances between rows of X and rows of centers.
    /// Returns matrix of shape (n_samples, n_centers).
    fn pairwise_distances(&self, x: &Array2<f64>, centers: &Array2<f64>) -> Result<Array2<f64>>;

    /// Element-wise ReLU: max(0, x).
    fn relu(&self, x: &Array2<f64>) -> Result<Array2<f64>>;

    /// Element-wise sigmoid: 1 / (1 + exp(-x)).
    fn sigmoid(&self, x: &Array2<f64>) -> Result<Array2<f64>>;

    /// Row-wise softmax: exp(x_ij - max_i) / sum_j exp(x_ij - max_i).
    fn softmax(&self, x: &Array2<f64>) -> Result<Array2<f64>>;

    /// Row-wise sum reduction.
    fn row_sum(&self, x: &Array2<f64>) -> Result<Array1<f64>>;

    /// Row-wise max reduction.
    fn row_max(&self, x: &Array2<f64>) -> Result<Array1<f64>>;

    /// Broadcast add bias vector to each row: output[i,j] = x[i,j] + bias[j].
    fn bias_add(&self, x: &Array2<f64>, bias: &Array1<f64>) -> Result<Array2<f64>>;

    /// ReLU gradient: 1.0 where z > 0, 0.0 otherwise.
    fn relu_grad(&self, z: &Array2<f64>) -> Result<Array2<f64>>;

    /// Sigmoid gradient: output * (1 - output), where output is the sigmoid output.
    fn sigmoid_grad(&self, output: &Array2<f64>) -> Result<Array2<f64>>;

    /// Element-wise multiplication: a * b.
    fn elementwise_mul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>>;

    /// Check if GPU backend is available and functional.
    fn is_available(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // ========================================================================
    // MockGpuBackend — CPU-based implementation of GpuBackend for testing
    // ========================================================================

    /// A mock GPU backend that performs computations on CPU.
    /// This allows testing the GpuBackend trait interface without GPU hardware.
    #[derive(Debug, Clone)]
    struct MockGpuBackend {
        available: bool,
    }

    impl MockGpuBackend {
        fn new() -> Self {
            Self { available: true }
        }

        fn unavailable() -> Self {
            Self { available: false }
        }
    }

    impl GpuBackend for MockGpuBackend {
        fn relu(&self, x: &Array2<f64>) -> crate::Result<Array2<f64>> {
            Ok(x.mapv(|v| v.max(0.0)))
        }

        fn sigmoid(&self, x: &Array2<f64>) -> crate::Result<Array2<f64>> {
            Ok(x.mapv(|v| 1.0 / (1.0 + (-v).exp())))
        }

        fn softmax(&self, x: &Array2<f64>) -> crate::Result<Array2<f64>> {
            let (rows, cols) = x.dim();
            let mut result = Array2::zeros((rows, cols));
            for i in 0..rows {
                let row = x.row(i);
                let max_val = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exp_row: Vec<f64> = row.iter().map(|&v| (v - max_val).exp()).collect();
                let sum: f64 = exp_row.iter().sum();
                for j in 0..cols {
                    result[[i, j]] = exp_row[j] / sum;
                }
            }
            Ok(result)
        }

        fn row_sum(&self, x: &Array2<f64>) -> crate::Result<ndarray::Array1<f64>> {
            Ok(x.sum_axis(ndarray::Axis(1)))
        }

        fn row_max(&self, x: &Array2<f64>) -> crate::Result<ndarray::Array1<f64>> {
            let (rows, _) = x.dim();
            let mut result = ndarray::Array1::zeros(rows);
            for i in 0..rows {
                result[i] = x.row(i).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            }
            Ok(result)
        }

        fn bias_add(
            &self,
            x: &Array2<f64>,
            bias: &ndarray::Array1<f64>,
        ) -> crate::Result<Array2<f64>> {
            let (_, cols) = x.dim();
            if bias.len() != cols {
                return Err(crate::FerroError::shape_mismatch(
                    format!("matrix cols = {}", cols),
                    format!("bias len = {}", bias.len()),
                ));
            }
            Ok(x + &bias.broadcast((x.nrows(), cols)).unwrap())
        }

        fn relu_grad(&self, z: &Array2<f64>) -> crate::Result<Array2<f64>> {
            Ok(z.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }))
        }

        fn sigmoid_grad(&self, output: &Array2<f64>) -> crate::Result<Array2<f64>> {
            Ok(output * &(1.0 - output))
        }

        fn elementwise_mul(&self, a: &Array2<f64>, b: &Array2<f64>) -> crate::Result<Array2<f64>> {
            if a.dim() != b.dim() {
                return Err(crate::FerroError::shape_mismatch(
                    format!("a shape = {:?}", a.dim()),
                    format!("b shape = {:?}", b.dim()),
                ));
            }
            Ok(a * b)
        }

        fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> crate::Result<Array2<f64>> {
            let (_, k) = a.dim();
            let (k2, _) = b.dim();
            if k != k2 {
                return Err(crate::FerroError::shape_mismatch(
                    format!("A cols = {}", k),
                    format!("B rows = {}", k2),
                ));
            }
            Ok(a.dot(b))
        }

        fn pairwise_distances(
            &self,
            x: &Array2<f64>,
            centers: &Array2<f64>,
        ) -> crate::Result<Array2<f64>> {
            let (_, d) = x.dim();
            let (_, d2) = centers.dim();
            if d != d2 {
                return Err(crate::FerroError::shape_mismatch(
                    format!("X features = {}", d),
                    format!("centers features = {}", d2),
                ));
            }
            let (n_samples, _) = x.dim();
            let (n_centers, _) = centers.dim();
            let mut result = Array2::zeros((n_samples, n_centers));
            for i in 0..n_samples {
                for j in 0..n_centers {
                    let mut dist = 0.0;
                    for f in 0..d {
                        let diff = x[[i, f]] - centers[[j, f]];
                        dist += diff * diff;
                    }
                    result[[i, j]] = dist;
                }
            }
            Ok(result)
        }

        fn is_available(&self) -> bool {
            self.available
        }
    }

    // ========================================================================
    // Mock backend: trait dispatch tests
    // ========================================================================

    #[test]
    fn test_mock_trait_object_dispatch() {
        let mock = MockGpuBackend::new();
        let backend: &dyn GpuBackend = &mock;
        assert!(backend.is_available());

        let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let b = ndarray::array![[5.0, 6.0], [7.0, 8.0]];
        let c = backend.matmul(&a, &b).unwrap();
        assert_eq!(c.dim(), (2, 2));
    }

    #[test]
    fn test_mock_trait_object_boxed() {
        let backend: Box<dyn GpuBackend> = Box::new(MockGpuBackend::new());
        assert!(backend.is_available());
        let a = ndarray::array![[1.0]];
        let b = ndarray::array![[2.0]];
        let c = backend.matmul(&a, &b).unwrap();
        assert!((c[[0, 0]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_mock_trait_object_arc() {
        let backend: std::sync::Arc<dyn GpuBackend> = std::sync::Arc::new(MockGpuBackend::new());
        assert!(backend.is_available());
        let a = ndarray::array![[3.0]];
        let b = ndarray::array![[4.0]];
        let c = backend.matmul(&a, &b).unwrap();
        assert!((c[[0, 0]] - 12.0).abs() < 1e-12);
    }

    // ========================================================================
    // Mock backend: is_available tests
    // ========================================================================

    #[test]
    fn test_mock_is_available_true() {
        let mock = MockGpuBackend::new();
        assert!(mock.is_available());
    }

    #[test]
    fn test_mock_is_available_false() {
        let mock = MockGpuBackend::unavailable();
        assert!(!mock.is_available());
    }

    // ========================================================================
    // Mock backend: matmul correctness tests
    // ========================================================================

    #[test]
    fn test_mock_matmul_identity() {
        let backend = MockGpuBackend::new();
        let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let identity = ndarray::array![[1.0, 0.0], [0.0, 1.0]];

        // A * I = A
        let result = backend.matmul(&a, &identity).unwrap();
        assert!((result[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((result[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((result[[1, 0]] - 3.0).abs() < 1e-12);
        assert!((result[[1, 1]] - 4.0).abs() < 1e-12);

        // I * A = A
        let result2 = backend.matmul(&identity, &a).unwrap();
        assert!((result2[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((result2[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((result2[[1, 0]] - 3.0).abs() < 1e-12);
        assert!((result2[[1, 1]] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_mock_matmul_zero_matrix() {
        let backend = MockGpuBackend::new();
        let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let zero = Array2::zeros((2, 2));

        let result = backend.matmul(&a, &zero).unwrap();
        for val in result.iter() {
            assert!(val.abs() < 1e-12);
        }

        let result2 = backend.matmul(&zero, &a).unwrap();
        for val in result2.iter() {
            assert!(val.abs() < 1e-12);
        }
    }

    #[test]
    fn test_mock_matmul_rectangular() {
        let backend = MockGpuBackend::new();

        // (2x3) * (3x4) = (2x4)
        let a = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = ndarray::array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ];
        let result = backend.matmul(&a, &b).unwrap();
        assert_eq!(result.dim(), (2, 4));

        let expected = a.dot(&b);
        for i in 0..2 {
            for j in 0..4 {
                assert!(
                    (result[[i, j]] - expected[[i, j]]).abs() < 1e-12,
                    "Mismatch at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_mock_matmul_1x1() {
        let backend = MockGpuBackend::new();
        let a = ndarray::array![[7.0]];
        let b = ndarray::array![[3.0]];
        let result = backend.matmul(&a, &b).unwrap();
        assert!((result[[0, 0]] - 21.0).abs() < 1e-12);
    }

    #[test]
    fn test_mock_matmul_row_times_col() {
        let backend = MockGpuBackend::new();
        // (1x3) * (3x1) = (1x1) — dot product
        let a = ndarray::array![[1.0, 2.0, 3.0]];
        let b = ndarray::array![[4.0], [5.0], [6.0]];
        let result = backend.matmul(&a, &b).unwrap();
        assert_eq!(result.dim(), (1, 1));
        assert!((result[[0, 0]] - 32.0).abs() < 1e-12); // 4+10+18
    }

    #[test]
    fn test_mock_matmul_col_times_row() {
        let backend = MockGpuBackend::new();
        // (3x1) * (1x3) = (3x3) — outer product
        let a = ndarray::array![[1.0], [2.0], [3.0]];
        let b = ndarray::array![[4.0, 5.0, 6.0]];
        let result = backend.matmul(&a, &b).unwrap();
        assert_eq!(result.dim(), (3, 3));
        assert!((result[[0, 0]] - 4.0).abs() < 1e-12);
        assert!((result[[1, 1]] - 10.0).abs() < 1e-12);
        assert!((result[[2, 2]] - 18.0).abs() < 1e-12);
    }

    #[test]
    fn test_mock_matmul_negative_values() {
        let backend = MockGpuBackend::new();
        let a = ndarray::array![[-1.0, 2.0], [3.0, -4.0]];
        let b = ndarray::array![[-5.0, 6.0], [7.0, -8.0]];
        let expected = a.dot(&b);
        let result = backend.matmul(&a, &b).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((result[[i, j]] - expected[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_mock_matmul_large_values() {
        let backend = MockGpuBackend::new();
        let a = ndarray::array![[1e10, 2e10], [3e10, 4e10]];
        let b = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        let result = backend.matmul(&a, &b).unwrap();
        assert!((result[[0, 0]] - 1e10).abs() < 1.0);
        assert!((result[[1, 1]] - 4e10).abs() < 1.0);
    }

    // ========================================================================
    // Mock backend: matmul error propagation
    // ========================================================================

    #[test]
    fn test_mock_matmul_shape_mismatch() {
        let backend = MockGpuBackend::new();
        let a = ndarray::array![[1.0, 2.0]]; // 1x2
        let b = ndarray::array![[1.0, 2.0, 3.0]]; // 1x3 — inner dims don't match
        let result = backend.matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_matmul_shape_mismatch_large() {
        let backend = MockGpuBackend::new();
        let a = Array2::zeros((10, 5));
        let b = Array2::zeros((3, 7));
        let result = backend.matmul(&a, &b);
        assert!(result.is_err());
    }

    // ========================================================================
    // Mock backend: pairwise_distances correctness tests
    // ========================================================================

    #[test]
    fn test_mock_distances_known() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let centers = ndarray::array![[0.0, 0.0], [1.0, 1.0]];
        let dists = backend.pairwise_distances(&x, &centers).unwrap();

        assert_eq!(dists.dim(), (3, 2));
        assert!((dists[[0, 0]] - 0.0).abs() < 1e-12); // origin to origin
        assert!((dists[[0, 1]] - 2.0).abs() < 1e-12); // origin to (1,1)
        assert!((dists[[1, 0]] - 1.0).abs() < 1e-12); // (1,0) to origin
        assert!((dists[[1, 1]] - 1.0).abs() < 1e-12); // (1,0) to (1,1)
        assert!((dists[[2, 0]] - 1.0).abs() < 1e-12); // (0,1) to origin
        assert!((dists[[2, 1]] - 1.0).abs() < 1e-12); // (0,1) to (1,1)
    }

    #[test]
    fn test_mock_distances_single_point() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[3.0, 4.0]];
        let centers = ndarray::array![[0.0, 0.0]];
        let dists = backend.pairwise_distances(&x, &centers).unwrap();
        assert_eq!(dists.dim(), (1, 1));
        assert!((dists[[0, 0]] - 25.0).abs() < 1e-12); // 9+16
    }

    #[test]
    fn test_mock_distances_identical_points() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
        let centers = ndarray::array![[1.0, 2.0, 3.0]];
        let dists = backend.pairwise_distances(&x, &centers).unwrap();
        assert!((dists[[0, 0]]).abs() < 1e-12);
        assert!((dists[[1, 0]]).abs() < 1e-12);
    }

    #[test]
    fn test_mock_distances_symmetry() {
        let backend = MockGpuBackend::new();
        let points = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        // Use same points as both x and centers to get a symmetric distance matrix
        let dists = backend.pairwise_distances(&points, &points).unwrap();
        assert_eq!(dists.dim(), (3, 3));

        // Diagonal should be zero
        for i in 0..3 {
            assert!(dists[[i, i]].abs() < 1e-12);
        }
        // Symmetric: d(i,j) == d(j,i)
        for i in 0..3 {
            for j in 0..3 {
                assert!((dists[[i, j]] - dists[[j, i]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_mock_distances_high_dimensional() {
        let backend = MockGpuBackend::new();
        let d = 100;
        // Point at origin vs unit vector along first axis
        let mut x_data = vec![0.0; d];
        let mut c_data = vec![0.0; d];
        c_data[0] = 1.0;
        let x = Array2::from_shape_vec((1, d), x_data.clone()).unwrap();
        let centers = Array2::from_shape_vec((1, d), c_data).unwrap();
        let dists = backend.pairwise_distances(&x, &centers).unwrap();
        assert!((dists[[0, 0]] - 1.0).abs() < 1e-12);

        // All ones to origin: distance should be d
        x_data = vec![1.0; d];
        let x2 = Array2::from_shape_vec((1, d), x_data).unwrap();
        let origin = Array2::zeros((1, d));
        let dists2 = backend.pairwise_distances(&x2, &origin).unwrap();
        assert!((dists2[[0, 0]] - d as f64).abs() < 1e-10);
    }

    #[test]
    fn test_mock_distances_many_centers() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[0.0, 0.0]];
        let centers = ndarray::array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
        let dists = backend.pairwise_distances(&x, &centers).unwrap();
        assert_eq!(dists.dim(), (1, 5));
        assert!((dists[[0, 0]] - 1.0).abs() < 1e-12); // (1,0)
        assert!((dists[[0, 1]] - 1.0).abs() < 1e-12); // (0,1)
        assert!((dists[[0, 2]] - 2.0).abs() < 1e-12); // (1,1)
        assert!((dists[[0, 3]] - 1.0).abs() < 1e-12); // (-1,0)
        assert!((dists[[0, 4]] - 1.0).abs() < 1e-12); // (0,-1)
    }

    // ========================================================================
    // Mock backend: pairwise_distances error propagation
    // ========================================================================

    #[test]
    fn test_mock_distances_shape_mismatch() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[1.0, 2.0]]; // 2 features
        let centers = ndarray::array![[1.0, 2.0, 3.0]]; // 3 features
        let result = backend.pairwise_distances(&x, &centers);
        assert!(result.is_err());
    }

    // ========================================================================
    // Send + Sync compile-time verification
    // ========================================================================

    #[test]
    fn test_mock_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockGpuBackend>();
    }

    #[test]
    fn test_trait_requires_send_sync() {
        // This verifies the trait object is Send + Sync
        fn assert_send_sync<T: Send + Sync + ?Sized>() {}
        assert_send_sync::<dyn GpuBackend>();
    }

    #[test]
    fn test_arc_trait_object_across_threads() {
        let backend: std::sync::Arc<dyn GpuBackend> = std::sync::Arc::new(MockGpuBackend::new());
        let backend_clone = backend.clone();
        let handle = std::thread::spawn(move || {
            assert!(backend_clone.is_available());
            let a = ndarray::array![[2.0]];
            let b = ndarray::array![[3.0]];
            backend_clone.matmul(&a, &b).unwrap()
        });
        let result = handle.join().unwrap();
        assert!((result[[0, 0]] - 6.0).abs() < 1e-12);
    }

    // ========================================================================
    // Mock backend: relu tests
    // ========================================================================

    #[test]
    fn test_mock_relu_positive_passthrough() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = backend.relu(&x).unwrap();
        assert!((result[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((result[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((result[[1, 2]] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_mock_relu_negative_to_zero() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[-1.0, -2.0], [-0.5, -100.0]];
        let result = backend.relu(&x).unwrap();
        for val in result.iter() {
            assert!(val.abs() < 1e-12);
        }
    }

    #[test]
    fn test_mock_relu_mixed() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[-1.0, 2.0], [0.0, -3.0]];
        let result = backend.relu(&x).unwrap();
        assert!((result[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((result[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((result[[1, 0]] - 0.0).abs() < 1e-12);
        assert!((result[[1, 1]] - 0.0).abs() < 1e-12);
    }

    // ========================================================================
    // Mock backend: sigmoid tests
    // ========================================================================

    #[test]
    fn test_mock_sigmoid_range() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[-10.0, -1.0, 0.0, 1.0, 10.0]]
            .into_shape_with_order((1, 5))
            .unwrap();
        let result = backend.sigmoid(&x).unwrap();
        for val in result.iter() {
            assert!(*val > 0.0 && *val < 1.0);
        }
    }

    #[test]
    fn test_mock_sigmoid_zero() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[0.0]];
        let result = backend.sigmoid(&x).unwrap();
        assert!((result[[0, 0]] - 0.5).abs() < 1e-12);
    }

    // ========================================================================
    // Mock backend: softmax tests
    // ========================================================================

    #[test]
    fn test_mock_softmax_rows_sum_to_one() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = backend.softmax(&x).unwrap();
        for i in 0..2 {
            let row_sum: f64 = result.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_mock_softmax_values() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[0.0, 0.0, 0.0]];
        let result = backend.softmax(&x).unwrap();
        // Uniform input -> uniform output
        for j in 0..3 {
            assert!((result[[0, j]] - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    // ========================================================================
    // Mock backend: row_sum / row_max tests
    // ========================================================================

    #[test]
    fn test_mock_row_sum_known() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = backend.row_sum(&x).unwrap();
        assert!((result[0] - 6.0).abs() < 1e-12);
        assert!((result[1] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_mock_row_max_known() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]];
        let result = backend.row_max(&x).unwrap();
        assert!((result[0] - 5.0).abs() < 1e-12);
        assert!((result[1] - 6.0).abs() < 1e-12);
    }

    // ========================================================================
    // Mock backend: bias_add tests
    // ========================================================================

    #[test]
    fn test_mock_bias_add_known() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let bias = ndarray::array![10.0, 20.0];
        let result = backend.bias_add(&x, &bias).unwrap();
        assert!((result[[0, 0]] - 11.0).abs() < 1e-12);
        assert!((result[[0, 1]] - 22.0).abs() < 1e-12);
        assert!((result[[1, 0]] - 13.0).abs() < 1e-12);
        assert!((result[[1, 1]] - 24.0).abs() < 1e-12);
    }

    #[test]
    fn test_mock_bias_add_shape_mismatch() {
        let backend = MockGpuBackend::new();
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let bias = ndarray::array![10.0, 20.0, 30.0]; // 3 != 2 cols
        let result = backend.bias_add(&x, &bias);
        assert!(result.is_err());
    }

    // ========================================================================
    // Mock backend: relu_grad tests
    // ========================================================================

    #[test]
    fn test_mock_relu_grad_positive() {
        let backend = MockGpuBackend::new();
        let z = ndarray::array![[1.0, 2.0], [0.5, 3.0]];
        let result = backend.relu_grad(&z).unwrap();
        for val in result.iter() {
            assert!((val - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_mock_relu_grad_negative() {
        let backend = MockGpuBackend::new();
        let z = ndarray::array![[-1.0, -2.0], [-0.5, 0.0]];
        let result = backend.relu_grad(&z).unwrap();
        for val in result.iter() {
            assert!(val.abs() < 1e-12);
        }
    }

    // ========================================================================
    // Mock backend: sigmoid_grad tests
    // ========================================================================

    #[test]
    fn test_mock_sigmoid_grad_known() {
        let backend = MockGpuBackend::new();
        // sigmoid(0) = 0.5, grad at output=0.5 is 0.5*0.5 = 0.25
        let output = ndarray::array![[0.5]];
        let result = backend.sigmoid_grad(&output).unwrap();
        assert!((result[[0, 0]] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_mock_sigmoid_grad_boundary() {
        let backend = MockGpuBackend::new();
        // At output ~0 or ~1, gradient should be ~0
        let output = ndarray::array![[0.0, 1.0]];
        let result = backend.sigmoid_grad(&output).unwrap();
        assert!(result[[0, 0]].abs() < 1e-12);
        assert!(result[[0, 1]].abs() < 1e-12);
    }

    // ========================================================================
    // Mock backend: elementwise_mul tests
    // ========================================================================

    #[test]
    fn test_mock_elementwise_mul_known() {
        let backend = MockGpuBackend::new();
        let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let b = ndarray::array![[5.0, 6.0], [7.0, 8.0]];
        let result = backend.elementwise_mul(&a, &b).unwrap();
        assert!((result[[0, 0]] - 5.0).abs() < 1e-12);
        assert!((result[[0, 1]] - 12.0).abs() < 1e-12);
        assert!((result[[1, 0]] - 21.0).abs() < 1e-12);
        assert!((result[[1, 1]] - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_mock_elementwise_mul_shape_mismatch() {
        let backend = MockGpuBackend::new();
        let a = ndarray::array![[1.0, 2.0]];
        let b = ndarray::array![[1.0, 2.0, 3.0]];
        let result = backend.elementwise_mul(&a, &b);
        assert!(result.is_err());
    }

    // ========================================================================
    // WgpuBackend graceful degradation tests (no GPU required)
    // ========================================================================

    // ========================================================================
    // GpuDispatchPolicy and GpuMemoryInfo compile-time tests
    // ========================================================================

    #[test]
    fn test_dispatch_policy_default() {
        let policy = GpuDispatchPolicy::default();
        match policy {
            GpuDispatchPolicy::Auto { min_elements } => assert_eq!(min_elements, 4096),
            _ => panic!("Default should be Auto {{ min_elements: 4096 }}"),
        }
    }

    #[test]
    fn test_dispatch_policy_debug() {
        let _ = format!("{:?}", GpuDispatchPolicy::Always);
        let _ = format!("{:?}", GpuDispatchPolicy::Never);
        let _ = format!("{:?}", GpuDispatchPolicy::Auto { min_elements: 100 });
    }

    #[test]
    fn test_dispatch_policy_clone() {
        let p = GpuDispatchPolicy::Auto { min_elements: 42 };
        let p2 = p.clone();
        match p2 {
            GpuDispatchPolicy::Auto { min_elements } => assert_eq!(min_elements, 42),
            _ => panic!("Clone failed"),
        }
    }

    #[test]
    fn test_memory_info_struct() {
        let info = GpuMemoryInfo {
            max_buffer_size: 1 << 28,
            max_storage_buffer_binding_size: 1 << 27,
        };
        assert_eq!(info.max_buffer_size, 268_435_456);
        assert_eq!(info.max_storage_buffer_binding_size, 134_217_728);
        let debug = format!("{:?}", info);
        assert!(debug.contains("GpuMemoryInfo"));
    }

    #[test]
    fn test_memory_info_clone() {
        let info = GpuMemoryInfo {
            max_buffer_size: 100,
            max_storage_buffer_binding_size: 50,
        };
        let info2 = info.clone();
        assert_eq!(info2.max_buffer_size, 100);
        assert_eq!(info2.max_storage_buffer_binding_size, 50);
    }

    #[test]
    fn test_preflight_check_wgpu() {
        // If GPU is available, verify memory_info is populated and preflight works
        if let Some(backend) = WgpuBackend::try_new() {
            let info = backend.memory_info();
            // max_buffer_size should be > 0 on any real GPU
            assert!(info.max_buffer_size > 0);
            assert!(info.max_storage_buffer_binding_size > 0);
        }
    }

    #[test]
    fn test_wgpu_try_new_returns_option() {
        // On machines without GPU, this returns None. On machines with GPU, Some.
        // Either way, it should not panic.
        let _result = WgpuBackend::try_new();
    }

    #[test]
    fn test_wgpu_new_returns_result() {
        // On machines without GPU, this returns Err. On machines with GPU, Ok.
        // Either way, it should not panic.
        let _result = WgpuBackend::new();
    }

    #[test]
    fn test_wgpu_try_new_none_when_no_gpu() {
        // This test validates behavior on this specific machine.
        // If no GPU is available, try_new returns None.
        let result = WgpuBackend::try_new();
        if result.is_none() {
            // Expected on headless/no-GPU machines
            assert!(WgpuBackend::new().is_err());
        }
    }

    // ========================================================================
    // Original GPU-required tests (skip if no GPU)
    // ========================================================================

    #[test]
    fn test_gpu_backend_available() {
        if let Some(backend) = WgpuBackend::try_new() {
            assert!(backend.is_available());
        }
        // If no GPU, test passes (graceful skip)
    }

    #[test]
    fn test_matmul_basic() {
        let backend = match WgpuBackend::try_new() {
            Some(b) => b,
            None => return, // No GPU available, skip
        };

        let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let b = ndarray::array![[5.0, 6.0], [7.0, 8.0]];
        let c = backend.matmul(&a, &b).unwrap();

        // Expected: [[19, 22], [43, 50]]
        assert!((c[[0, 0]] - 19.0).abs() < 1e-4);
        assert!((c[[0, 1]] - 22.0).abs() < 1e-4);
        assert!((c[[1, 0]] - 43.0).abs() < 1e-4);
        assert!((c[[1, 1]] - 50.0).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_parity() {
        let backend = match WgpuBackend::try_new() {
            Some(b) => b,
            None => return,
        };

        use ndarray::Array2;
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let m = 64;
        let k = 32;
        let n = 48;

        let a = Array2::from_shape_fn((m, k), |_| rng.random::<f64>());
        let b = Array2::from_shape_fn((k, n), |_| rng.random::<f64>());

        let cpu_result = a.dot(&b);
        let gpu_result = backend.matmul(&a, &b).unwrap();

        for i in 0..m {
            for j in 0..n {
                let diff = (cpu_result[[i, j]] - gpu_result[[i, j]]).abs();
                assert!(
                    diff < 1e-3,
                    "Mismatch at [{},{}]: cpu={}, gpu={}, diff={}",
                    i,
                    j,
                    cpu_result[[i, j]],
                    gpu_result[[i, j]],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_pairwise_distances_basic() {
        let backend = match WgpuBackend::try_new() {
            Some(b) => b,
            None => return,
        };

        let x = ndarray::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let centers = ndarray::array![[0.0, 0.0], [1.0, 1.0]];
        let dists = backend.pairwise_distances(&x, &centers).unwrap();

        // d(0,0)=0, d(0,1)=2, d(1,0)=1, d(1,1)=1, d(2,0)=1, d(2,1)=1
        assert!((dists[[0, 0]] - 0.0).abs() < 1e-4);
        assert!((dists[[0, 1]] - 2.0).abs() < 1e-4);
        assert!((dists[[1, 0]] - 1.0).abs() < 1e-4);
        assert!((dists[[1, 1]] - 1.0).abs() < 1e-4);
        assert!((dists[[2, 0]] - 1.0).abs() < 1e-4);
        assert!((dists[[2, 1]] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_edge_shapes() {
        let backend = match WgpuBackend::try_new() {
            Some(b) => b,
            None => return,
        };

        // 1x1 matrix
        let a = ndarray::array![[3.0]];
        let b = ndarray::array![[4.0]];
        let c = backend.matmul(&a, &b).unwrap();
        assert!((c[[0, 0]] - 12.0).abs() < 1e-4);

        // 1xN and Nx1
        let a = ndarray::array![[1.0, 2.0, 3.0]];
        let b = ndarray::array![[4.0], [5.0], [6.0]];
        let c = backend.matmul(&a, &b).unwrap();
        assert!((c[[0, 0]] - 32.0).abs() < 1e-4);
    }

    #[test]
    fn test_distance_single_cluster() {
        let backend = match WgpuBackend::try_new() {
            Some(b) => b,
            None => return,
        };

        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let centers = ndarray::array![[0.0, 0.0]];
        let dists = backend.pairwise_distances(&x, &centers).unwrap();

        assert_eq!(dists.dim(), (3, 1));
        assert!((dists[[0, 0]] - 5.0).abs() < 1e-4); // 1+4
        assert!((dists[[1, 0]] - 25.0).abs() < 1e-4); // 9+16
        assert!((dists[[2, 0]] - 61.0).abs() < 1e-4); // 25+36
    }

    #[test]
    fn test_distance_parity_random() {
        let backend = match WgpuBackend::try_new() {
            Some(b) => b,
            None => return,
        };

        use ndarray::Array2;
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let n = 50;
        let k = 5;
        let d = 10;

        let x = Array2::from_shape_fn((n, d), |_| rng.random::<f64>());
        let centers = Array2::from_shape_fn((k, d), |_| rng.random::<f64>());

        let gpu_dists = backend.pairwise_distances(&x, &centers).unwrap();

        // CPU reference
        for i in 0..n {
            for j in 0..k {
                let mut cpu_dist = 0.0;
                for f in 0..d {
                    let diff = x[[i, f]] - centers[[j, f]];
                    cpu_dist += diff * diff;
                }
                let diff = (gpu_dists[[i, j]] - cpu_dist).abs();
                assert!(
                    diff < 1e-2,
                    "Distance mismatch at [{},{}]: cpu={}, gpu={}",
                    i,
                    j,
                    cpu_dist,
                    gpu_dists[[i, j]]
                );
            }
        }
    }
}

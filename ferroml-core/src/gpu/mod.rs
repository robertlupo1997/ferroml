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
mod kernels;

pub use backend::WgpuBackend;

use crate::Result;
use ndarray::Array2;

/// Trait for GPU-accelerated linear algebra operations.
pub trait GpuBackend: Send + Sync + std::fmt::Debug {
    /// Matrix multiply: C = A @ B
    fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>>;

    /// Compute pairwise squared Euclidean distances between rows of X and rows of centers.
    /// Returns matrix of shape (n_samples, n_centers).
    fn pairwise_distances(&self, x: &Array2<f64>, centers: &Array2<f64>) -> Result<Array2<f64>>;

    /// Check if GPU backend is available and functional.
    fn is_available(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

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

//! Tests that verify models work correctly WITHOUT GPU hardware.
//!
//! These tests do NOT require the `gpu` feature flag. They verify that
//! CPU fallback paths function correctly, and that the codebase compiles
//! and runs without GPU support.

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::clustering::{ClusteringModel, KMeans};

    // ========================================================================
    // KMeans works without GPU
    // ========================================================================

    #[test]
    fn test_kmeans_cpu_fallback_basic() {
        let x = ndarray::array![
            [1.0, 1.0],
            [1.1, 1.1],
            [0.9, 0.9],
            [5.0, 5.0],
            [5.1, 5.1],
            [4.9, 4.9]
        ];

        let mut kmeans = KMeans::new(2);
        kmeans.fit(&x).unwrap();
        let labels = kmeans.predict(&x).unwrap();

        // Should find 2 clusters: one near (1,1), one near (5,5)
        assert_eq!(labels.len(), 6);
        // First 3 points should be in one cluster, last 3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_kmeans_cpu_fallback_deterministic() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let n = 100;
        let d = 5;
        let x = Array2::from_shape_fn((n, d), |_| rng.random::<f64>());

        let mut kmeans1 = KMeans::new(3).random_state(123);
        kmeans1.fit(&x).unwrap();
        let labels1 = kmeans1.predict(&x).unwrap();

        let mut kmeans2 = KMeans::new(3).random_state(123);
        kmeans2.fit(&x).unwrap();
        let labels2 = kmeans2.predict(&x).unwrap();

        assert_eq!(
            labels1, labels2,
            "KMeans should be deterministic with same seed"
        );
    }

    #[test]
    fn test_kmeans_cpu_fallback_predict_shape() {
        let x_train = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let mut kmeans = KMeans::new(2);
        kmeans.fit(&x_train).unwrap();

        let x_test = ndarray::array![[2.0, 3.0], [6.0, 7.0]];
        let labels = kmeans.predict(&x_test).unwrap();
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn test_kmeans_cpu_fallback_single_cluster() {
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let mut kmeans = KMeans::new(1);
        kmeans.fit(&x).unwrap();
        let labels = kmeans.predict(&x).unwrap();

        // All points should be in cluster 0
        for &label in labels.iter() {
            assert_eq!(label, 0);
        }
    }

    // ========================================================================
    // Verify GPU feature is optional (compile-time tests)
    // ========================================================================

    #[test]
    fn test_gpu_module_is_conditionally_compiled() {
        // When gpu feature is NOT enabled, the gpu module should not exist.
        // When it IS enabled, it should. This test just verifies the cfg works.
        #[cfg(feature = "gpu")]
        {
            // GPU module is available — try_new should return Option
            let _: Option<crate::gpu::WgpuBackend> = crate::gpu::WgpuBackend::try_new();
        }

        #[cfg(not(feature = "gpu"))]
        {
            // GPU module is not available — this branch compiles without gpu feature
            // Just verify KMeans still works without any GPU code being compiled
            let x = ndarray::array![[1.0], [2.0], [3.0]];
            let mut km = KMeans::new(1);
            km.fit(&x).unwrap();
            let _ = km.predict(&x).unwrap();
        }
    }

    // ========================================================================
    // KMeans with_gpu builder pattern (only when gpu feature enabled)
    // ========================================================================

    #[cfg(feature = "gpu")]
    #[test]
    fn test_kmeans_with_gpu_none_falls_back_to_cpu() {
        // When GPU backend returns None, KMeans should still work via CPU
        let x = ndarray::array![[1.0, 1.0], [1.1, 0.9], [5.0, 5.0], [4.9, 5.1]];

        // Try to create a GPU backend — will be None on this machine
        let gpu = crate::gpu::WgpuBackend::try_new();
        let mut kmeans = KMeans::new(2);
        if let Some(backend) = gpu {
            kmeans = kmeans.with_gpu(std::sync::Arc::new(backend));
        }
        // Either way, fit/predict should work
        kmeans.fit(&x).unwrap();
        let labels = kmeans.predict(&x).unwrap();
        assert_eq!(labels.len(), 4);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_kmeans_with_mock_gpu_backend() {
        use crate::gpu::GpuBackend;

        /// Mock backend that performs CPU computation via the GpuBackend trait
        #[derive(Debug)]
        struct CpuMockBackend;

        impl GpuBackend for CpuMockBackend {
            fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> crate::Result<Array2<f64>> {
                Ok(a.dot(b))
            }

            fn pairwise_distances(
                &self,
                x: &Array2<f64>,
                centers: &Array2<f64>,
            ) -> crate::Result<Array2<f64>> {
                let (n, d) = x.dim();
                let (c, _) = centers.dim();
                let mut result = Array2::zeros((n, c));
                for i in 0..n {
                    for j in 0..c {
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
                true
            }
        }

        let x = ndarray::array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]];

        let mock_gpu: std::sync::Arc<dyn GpuBackend> = std::sync::Arc::new(CpuMockBackend);
        let mut kmeans = KMeans::new(2).with_gpu(mock_gpu);
        kmeans.fit(&x).unwrap();
        let labels = kmeans.predict(&x).unwrap();
        assert_eq!(labels.len(), 4);
        // Points near (0,0) in one cluster, near (10,10) in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    // ========================================================================
    // Linear models work without GPU (sanity check)
    // ========================================================================

    #[test]
    fn test_linear_regression_cpu_only() {
        use crate::models::{LinearRegression, Model};

        let x = ndarray::array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = ndarray::array![2.0, 4.0, 6.0, 8.0, 10.0];

        let mut lr = LinearRegression::new();
        lr.fit(&x, &y).unwrap();
        let preds = lr.predict(&x).unwrap();

        // Should be close to y = 2x
        for i in 0..5 {
            assert!(
                (preds[i] - y[i]).abs() < 0.5,
                "Linear regression prediction too far: pred={}, expected={}",
                preds[i],
                y[i]
            );
        }
    }
}

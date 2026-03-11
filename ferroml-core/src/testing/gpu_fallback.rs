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
                Ok(x + &bias.broadcast((x.nrows(), cols)).unwrap())
            }

            fn relu_grad(&self, z: &Array2<f64>) -> crate::Result<Array2<f64>> {
                Ok(z.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }))
            }

            fn sigmoid_grad(&self, output: &Array2<f64>) -> crate::Result<Array2<f64>> {
                Ok(output * &(1.0 - output))
            }

            fn elementwise_mul(
                &self,
                a: &Array2<f64>,
                b: &Array2<f64>,
            ) -> crate::Result<Array2<f64>> {
                Ok(a * b)
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

    // ========================================================================
    // DBSCAN works without GPU
    // ========================================================================

    #[test]
    fn test_dbscan_cpu_fallback_basic() {
        use crate::clustering::DBSCAN;

        let x = ndarray::array![
            [1.0, 1.0],
            [1.1, 1.0],
            [1.0, 1.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1]
        ];

        let mut dbscan = DBSCAN::new(0.5, 2);
        dbscan.fit(&x).unwrap();
        let labels = dbscan.labels().unwrap();
        assert_eq!(labels.len(), 6);
        assert_eq!(dbscan.n_clusters(), Some(2));
    }

    // ========================================================================
    // HDBSCAN works without GPU
    // ========================================================================

    #[test]
    fn test_hdbscan_cpu_fallback_basic() {
        use crate::clustering::HDBSCAN;

        let x = ndarray::array![
            [1.0, 1.0],
            [1.1, 1.0],
            [1.0, 1.1],
            [1.1, 1.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [5.1, 5.1]
        ];

        let mut hdbscan = HDBSCAN::new(3);
        let labels = hdbscan.fit_predict(&x).unwrap();
        assert_eq!(labels.len(), 8);
    }

    // ========================================================================
    // t-SNE works without GPU
    // ========================================================================

    #[test]
    fn test_tsne_cpu_fallback_basic() {
        use crate::decomposition::TSNE;
        use crate::preprocessing::Transformer;

        let x = Array2::from_shape_fn((20, 4), |(i, j)| (i * 4 + j) as f64);

        let mut tsne = TSNE::new()
            .with_perplexity(5.0)
            .with_n_components(2)
            .with_max_iter(50)
            .with_random_state(42);

        let embedding = tsne.fit_transform(&x).unwrap();
        assert_eq!(embedding.dim(), (20, 2));
    }

    // ========================================================================
    // GradientBoosting works without GPU
    // ========================================================================

    #[test]
    fn test_gradient_boosting_cpu_fallback_basic() {
        use crate::models::boosting::GradientBoostingRegressor;
        use crate::models::Model;

        let x = ndarray::array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0]
        ];
        let y = ndarray::array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut model = GradientBoostingRegressor::new()
            .with_n_estimators(10)
            .with_learning_rate(0.1)
            .with_max_depth(Some(2));
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    // ========================================================================
    // GPU model integration tests (only when gpu feature enabled)
    // ========================================================================

    #[cfg(feature = "gpu")]
    mod gpu_model_integration {
        use crate::gpu::GpuBackend;
        use ndarray::Array2;

        /// Shared mock GPU backend that performs CPU computation.
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
                Ok(x + &bias.broadcast((x.nrows(), cols)).unwrap())
            }

            fn relu_grad(&self, z: &Array2<f64>) -> crate::Result<Array2<f64>> {
                Ok(z.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }))
            }

            fn sigmoid_grad(&self, output: &Array2<f64>) -> crate::Result<Array2<f64>> {
                Ok(output * &(1.0 - output))
            }

            fn elementwise_mul(
                &self,
                a: &Array2<f64>,
                b: &Array2<f64>,
            ) -> crate::Result<Array2<f64>> {
                Ok(a * b)
            }

            fn is_available(&self) -> bool {
                true
            }
        }

        /// Mock backend that always returns errors, to test fallback behavior.
        #[derive(Debug)]
        struct FailingMockBackend;

        impl GpuBackend for FailingMockBackend {
            fn matmul(&self, _a: &Array2<f64>, _b: &Array2<f64>) -> crate::Result<Array2<f64>> {
                Err(crate::FerroError::NumericalError(
                    "GPU simulated error".into(),
                ))
            }
            fn pairwise_distances(
                &self,
                _x: &Array2<f64>,
                _c: &Array2<f64>,
            ) -> crate::Result<Array2<f64>> {
                Err(crate::FerroError::NumericalError(
                    "GPU simulated error".into(),
                ))
            }
            fn relu(&self, _x: &Array2<f64>) -> crate::Result<Array2<f64>> {
                Err(crate::FerroError::NumericalError(
                    "GPU simulated error".into(),
                ))
            }
            fn sigmoid(&self, _x: &Array2<f64>) -> crate::Result<Array2<f64>> {
                Err(crate::FerroError::NumericalError(
                    "GPU simulated error".into(),
                ))
            }
            fn softmax(&self, _x: &Array2<f64>) -> crate::Result<Array2<f64>> {
                Err(crate::FerroError::NumericalError(
                    "GPU simulated error".into(),
                ))
            }
            fn row_sum(&self, _x: &Array2<f64>) -> crate::Result<ndarray::Array1<f64>> {
                Err(crate::FerroError::NumericalError(
                    "GPU simulated error".into(),
                ))
            }
            fn row_max(&self, _x: &Array2<f64>) -> crate::Result<ndarray::Array1<f64>> {
                Err(crate::FerroError::NumericalError(
                    "GPU simulated error".into(),
                ))
            }
            fn bias_add(
                &self,
                _x: &Array2<f64>,
                _b: &ndarray::Array1<f64>,
            ) -> crate::Result<Array2<f64>> {
                Err(crate::FerroError::NumericalError(
                    "GPU simulated error".into(),
                ))
            }
            fn relu_grad(&self, _z: &Array2<f64>) -> crate::Result<Array2<f64>> {
                Err(crate::FerroError::NumericalError(
                    "GPU simulated error".into(),
                ))
            }
            fn sigmoid_grad(&self, _o: &Array2<f64>) -> crate::Result<Array2<f64>> {
                Err(crate::FerroError::NumericalError(
                    "GPU simulated error".into(),
                ))
            }
            fn elementwise_mul(
                &self,
                _a: &Array2<f64>,
                _b: &Array2<f64>,
            ) -> crate::Result<Array2<f64>> {
                Err(crate::FerroError::NumericalError(
                    "GPU simulated error".into(),
                ))
            }
            fn is_available(&self) -> bool {
                false
            }
        }

        fn mock_gpu() -> std::sync::Arc<dyn GpuBackend> {
            std::sync::Arc::new(CpuMockBackend)
        }

        fn failing_gpu() -> std::sync::Arc<dyn GpuBackend> {
            std::sync::Arc::new(FailingMockBackend)
        }

        fn two_cluster_data() -> Array2<f64> {
            ndarray::array![
                [1.0, 1.0],
                [1.1, 1.0],
                [1.0, 1.1],
                [1.1, 1.1],
                [5.0, 5.0],
                [5.1, 5.0],
                [5.0, 5.1],
                [5.1, 5.1]
            ]
        }

        // ====================================================================
        // DBSCAN GPU integration tests
        // ====================================================================

        #[test]
        fn test_dbscan_without_gpu() {
            use crate::clustering::{ClusteringModel, DBSCAN};
            let x = two_cluster_data();
            let mut dbscan = DBSCAN::new(0.5, 2);
            dbscan.fit(&x).unwrap();
            assert_eq!(dbscan.n_clusters(), Some(2));
        }

        #[test]
        fn test_dbscan_with_mock_gpu() {
            use crate::clustering::{ClusteringModel, DBSCAN};
            let x = two_cluster_data();
            let mut dbscan = DBSCAN::new(0.5, 2).with_gpu(mock_gpu());
            dbscan.fit(&x).unwrap();
            assert_eq!(dbscan.n_clusters(), Some(2));
        }

        #[test]
        fn test_dbscan_gpu_cpu_parity() {
            use crate::clustering::{ClusteringModel, DBSCAN};
            let x = two_cluster_data();

            let mut cpu_dbscan = DBSCAN::new(0.5, 2);
            cpu_dbscan.fit(&x).unwrap();
            let cpu_labels = cpu_dbscan.labels().unwrap().clone();

            let mut gpu_dbscan = DBSCAN::new(0.5, 2).with_gpu(mock_gpu());
            gpu_dbscan.fit(&x).unwrap();
            let gpu_labels = gpu_dbscan.labels().unwrap().clone();

            assert_eq!(cpu_labels, gpu_labels, "GPU and CPU labels must match");
        }

        #[test]
        fn test_dbscan_gpu_error_fallback() {
            use crate::clustering::{ClusteringModel, DBSCAN};
            let x = two_cluster_data();
            // Failing GPU should fall back to CPU gracefully
            let mut dbscan = DBSCAN::new(0.5, 2).with_gpu(failing_gpu());
            dbscan.fit(&x).unwrap();
            assert_eq!(dbscan.n_clusters(), Some(2));
        }

        // ====================================================================
        // HDBSCAN GPU integration tests
        // ====================================================================

        #[test]
        fn test_hdbscan_without_gpu() {
            use crate::clustering::{ClusteringModel, HDBSCAN};
            let x = two_cluster_data();
            let mut hdbscan = HDBSCAN::new(3);
            hdbscan.fit(&x).unwrap();
            assert!(hdbscan.is_fitted());
        }

        #[test]
        fn test_hdbscan_with_mock_gpu() {
            use crate::clustering::{ClusteringModel, HDBSCAN};
            let x = two_cluster_data();
            let mut hdbscan = HDBSCAN::new(3).with_gpu(mock_gpu());
            hdbscan.fit(&x).unwrap();
            assert!(hdbscan.is_fitted());
        }

        #[test]
        fn test_hdbscan_gpu_cpu_parity() {
            use crate::clustering::{ClusteringModel, HDBSCAN};
            let x = two_cluster_data();

            let mut cpu = HDBSCAN::new(3);
            cpu.fit(&x).unwrap();
            let cpu_labels = cpu.labels().unwrap().clone();

            let mut gpu = HDBSCAN::new(3).with_gpu(mock_gpu());
            gpu.fit(&x).unwrap();
            let gpu_labels = gpu.labels().unwrap().clone();

            assert_eq!(
                cpu_labels, gpu_labels,
                "HDBSCAN GPU and CPU labels must match"
            );
        }

        #[test]
        fn test_hdbscan_gpu_error_fallback() {
            use crate::clustering::{ClusteringModel, HDBSCAN};
            let x = two_cluster_data();
            let mut hdbscan = HDBSCAN::new(3).with_gpu(failing_gpu());
            hdbscan.fit(&x).unwrap();
            assert!(hdbscan.is_fitted());
        }

        // ====================================================================
        // t-SNE GPU integration tests
        // ====================================================================

        fn tsne_data() -> Array2<f64> {
            Array2::from_shape_fn((20, 4), |(i, j)| (i * 4 + j) as f64)
        }

        #[test]
        fn test_tsne_without_gpu() {
            use crate::decomposition::TSNE;
            use crate::preprocessing::Transformer;
            let x = tsne_data();
            let mut tsne = TSNE::new()
                .with_perplexity(5.0)
                .with_n_components(2)
                .with_max_iter(50)
                .with_random_state(42);
            let embedding = tsne.fit_transform(&x).unwrap();
            assert_eq!(embedding.dim(), (20, 2));
        }

        #[test]
        fn test_tsne_with_mock_gpu() {
            use crate::decomposition::TSNE;
            use crate::preprocessing::Transformer;
            let x = tsne_data();
            let mut tsne = TSNE::new()
                .with_perplexity(5.0)
                .with_n_components(2)
                .with_max_iter(50)
                .with_random_state(42)
                .with_gpu(mock_gpu());
            let embedding = tsne.fit_transform(&x).unwrap();
            assert_eq!(embedding.dim(), (20, 2));
        }

        #[test]
        fn test_tsne_gpu_cpu_parity() {
            use crate::decomposition::TsneMethod;
            use crate::decomposition::TSNE;
            use crate::preprocessing::Transformer;
            let x = tsne_data();

            let mut cpu_tsne = TSNE::new()
                .with_perplexity(5.0)
                .with_n_components(2)
                .with_max_iter(50)
                .with_random_state(42)
                .with_method(TsneMethod::Exact);
            let cpu_embedding = cpu_tsne.fit_transform(&x).unwrap();

            let mut gpu_tsne = TSNE::new()
                .with_perplexity(5.0)
                .with_n_components(2)
                .with_max_iter(50)
                .with_random_state(42)
                .with_method(TsneMethod::Exact)
                .with_gpu(mock_gpu());
            let gpu_embedding = gpu_tsne.fit_transform(&x).unwrap();

            // Results should match since CpuMockBackend computes exact same distances
            assert_eq!(cpu_embedding.dim(), gpu_embedding.dim());
            for i in 0..cpu_embedding.nrows() {
                for j in 0..cpu_embedding.ncols() {
                    assert!(
                        (cpu_embedding[[i, j]] - gpu_embedding[[i, j]]).abs() < 1e-6,
                        "t-SNE parity mismatch at [{},{}]: cpu={}, gpu={}",
                        i,
                        j,
                        cpu_embedding[[i, j]],
                        gpu_embedding[[i, j]]
                    );
                }
            }
        }

        #[test]
        fn test_tsne_gpu_error_fallback() {
            use crate::decomposition::TsneMethod;
            use crate::decomposition::TSNE;
            use crate::preprocessing::Transformer;
            let x = tsne_data();
            let mut tsne = TSNE::new()
                .with_perplexity(5.0)
                .with_n_components(2)
                .with_max_iter(50)
                .with_random_state(42)
                .with_method(TsneMethod::Exact)
                .with_gpu(failing_gpu());
            // Should fall back to CPU gracefully
            let embedding = tsne.fit_transform(&x).unwrap();
            assert_eq!(embedding.dim(), (20, 2));
        }

        // ====================================================================
        // GradientBoosting GPU integration tests
        // ====================================================================

        #[test]
        fn test_gradient_boosting_without_gpu() {
            use crate::models::boosting::GradientBoostingRegressor;
            use crate::models::Model;
            let x = ndarray::array![
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
                [6.0, 7.0],
                [7.0, 8.0],
                [8.0, 9.0]
            ];
            let y = ndarray::array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let mut model = GradientBoostingRegressor::new()
                .with_n_estimators(10)
                .with_learning_rate(0.1)
                .with_max_depth(Some(2));
            model.fit(&x, &y).unwrap();
            let preds = model.predict(&x).unwrap();
            assert_eq!(preds.len(), 8);
        }

        #[test]
        fn test_gradient_boosting_with_mock_gpu() {
            use crate::models::boosting::GradientBoostingRegressor;
            use crate::models::Model;
            let x = ndarray::array![
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
                [6.0, 7.0],
                [7.0, 8.0],
                [8.0, 9.0]
            ];
            let y = ndarray::array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let mut model = GradientBoostingRegressor::new()
                .with_n_estimators(10)
                .with_learning_rate(0.1)
                .with_max_depth(Some(2))
                .with_gpu(mock_gpu());
            model.fit(&x, &y).unwrap();
            let preds = model.predict(&x).unwrap();
            assert_eq!(preds.len(), 8);
        }

        #[test]
        fn test_gradient_boosting_gpu_cpu_parity() {
            use crate::models::boosting::GradientBoostingRegressor;
            use crate::models::Model;
            let x = ndarray::array![
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
                [6.0, 7.0],
                [7.0, 8.0],
                [8.0, 9.0]
            ];
            let y = ndarray::array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            let mut cpu_model = GradientBoostingRegressor::new()
                .with_n_estimators(10)
                .with_learning_rate(0.1)
                .with_max_depth(Some(2));
            cpu_model.fit(&x, &y).unwrap();
            let cpu_preds = cpu_model.predict(&x).unwrap();

            let mut gpu_model = GradientBoostingRegressor::new()
                .with_n_estimators(10)
                .with_learning_rate(0.1)
                .with_max_depth(Some(2))
                .with_gpu(mock_gpu());
            gpu_model.fit(&x, &y).unwrap();
            let gpu_preds = gpu_model.predict(&x).unwrap();

            // GB doesn't use GPU for compute currently, so predictions should be identical
            for i in 0..8 {
                assert!(
                    (cpu_preds[i] - gpu_preds[i]).abs() < 1e-10,
                    "GB parity mismatch at [{}]: cpu={}, gpu={}",
                    i,
                    cpu_preds[i],
                    gpu_preds[i]
                );
            }
        }

        #[test]
        fn test_gradient_boosting_gpu_error_fallback() {
            use crate::models::boosting::GradientBoostingRegressor;
            use crate::models::Model;
            let x = ndarray::array![
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
                [6.0, 7.0],
                [7.0, 8.0],
                [8.0, 9.0]
            ];
            let y = ndarray::array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let mut model = GradientBoostingRegressor::new()
                .with_n_estimators(10)
                .with_learning_rate(0.1)
                .with_max_depth(Some(2))
                .with_gpu(failing_gpu());
            model.fit(&x, &y).unwrap();
            let preds = model.predict(&x).unwrap();
            assert_eq!(preds.len(), 8);
        }

        // ====================================================================
        // MLP Neural Network GPU integration tests (Q.3)
        // ====================================================================

        /// Helper: MSE loss function for MLP training
        fn mse_loss(output: &Array2<f64>, target: &Array2<f64>) -> (f64, Array2<f64>) {
            let n = output.nrows() as f64;
            let diff = output - target;
            let loss = (&diff * &diff).sum() / n;
            let grad = 2.0 * &diff / n;
            (loss, grad)
        }

        #[test]
        fn test_mlp_forward_gpu_parity() {
            use crate::neural::{Activation, MLP};
            let seed = 42u64;
            let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [-1.0, 0.5]];

            // CPU forward
            let mut cpu_mlp = MLP::new()
                .hidden_layer_sizes(&[4, 3])
                .activation(Activation::ReLU)
                .output_activation(Activation::Linear)
                .random_state(seed);
            cpu_mlp.initialize(2, 1).unwrap();
            let cpu_output = cpu_mlp.forward(&x, false).unwrap();

            // GPU forward (mock backend does exact same math)
            let mut gpu_mlp = MLP::new()
                .hidden_layer_sizes(&[4, 3])
                .activation(Activation::ReLU)
                .output_activation(Activation::Linear)
                .random_state(seed)
                .with_gpu(mock_gpu());
            gpu_mlp.initialize(2, 1).unwrap();
            let gpu_output = gpu_mlp.forward(&x, false).unwrap();

            assert_eq!(cpu_output.dim(), gpu_output.dim());
            for i in 0..cpu_output.nrows() {
                for j in 0..cpu_output.ncols() {
                    assert!(
                        (cpu_output[[i, j]] - gpu_output[[i, j]]).abs() < 1e-10,
                        "Forward parity mismatch at [{},{}]: cpu={}, gpu={}",
                        i,
                        j,
                        cpu_output[[i, j]],
                        gpu_output[[i, j]]
                    );
                }
            }
        }

        #[test]
        fn test_mlp_backward_gpu_parity() {
            use crate::neural::{Activation, MLP};
            let seed = 42u64;
            let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

            // CPU
            let mut cpu_mlp = MLP::new()
                .hidden_layer_sizes(&[4])
                .activation(Activation::ReLU)
                .output_activation(Activation::Linear)
                .random_state(seed);
            cpu_mlp.initialize(2, 1).unwrap();
            let cpu_output = cpu_mlp.forward(&x, true).unwrap();
            let loss_grad = 2.0 * (&cpu_output - 1.0) / x.nrows() as f64;
            let cpu_grads = cpu_mlp.backward(&loss_grad).unwrap();

            // GPU
            let mut gpu_mlp = MLP::new()
                .hidden_layer_sizes(&[4])
                .activation(Activation::ReLU)
                .output_activation(Activation::Linear)
                .random_state(seed)
                .with_gpu(mock_gpu());
            gpu_mlp.initialize(2, 1).unwrap();
            let gpu_output = gpu_mlp.forward(&x, true).unwrap();
            let loss_grad_gpu = 2.0 * (&gpu_output - 1.0) / x.nrows() as f64;
            let gpu_grads = gpu_mlp.backward(&loss_grad_gpu).unwrap();

            assert_eq!(cpu_grads.len(), gpu_grads.len());
            for (layer_idx, ((cw, cb), (gw, gb))) in
                cpu_grads.iter().zip(gpu_grads.iter()).enumerate()
            {
                for ((i, j), &cv) in cw.indexed_iter() {
                    let gv = gw[[i, j]];
                    assert!(
                        (cv - gv).abs() < 1e-10,
                        "Backward weight grad mismatch layer {} at [{},{}]: cpu={}, gpu={}",
                        layer_idx,
                        i,
                        j,
                        cv,
                        gv
                    );
                }
                for (i, (&cv, &gv)) in cb.iter().zip(gb.iter()).enumerate() {
                    assert!(
                        (cv - gv).abs() < 1e-10,
                        "Backward bias grad mismatch layer {} at [{}]: cpu={}, gpu={}",
                        layer_idx,
                        i,
                        cv,
                        gv
                    );
                }
            }
        }

        #[test]
        fn test_mlp_full_train_loop_parity() {
            use crate::neural::{Activation, Solver, MLP};
            let seed = 42u64;
            let x = ndarray::array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
            let y = ndarray::array![[0.0], [1.0], [1.0], [0.0]]; // XOR-like

            let n_epochs = 10;

            // CPU training
            let mut cpu_mlp = MLP::new()
                .hidden_layer_sizes(&[4])
                .activation(Activation::ReLU)
                .output_activation(Activation::Linear)
                .solver(Solver::Adam)
                .learning_rate(0.01)
                .random_state(seed)
                .batch_size(4);
            cpu_mlp.initialize(2, 1).unwrap();
            let mut cpu_losses = Vec::new();
            for _ in 0..n_epochs {
                let loss = cpu_mlp.train_epoch(&x, &y, mse_loss).unwrap();
                cpu_losses.push(loss);
            }

            // GPU training
            let mut gpu_mlp = MLP::new()
                .hidden_layer_sizes(&[4])
                .activation(Activation::ReLU)
                .output_activation(Activation::Linear)
                .solver(Solver::Adam)
                .learning_rate(0.01)
                .random_state(seed)
                .batch_size(4)
                .with_gpu(mock_gpu());
            gpu_mlp.initialize(2, 1).unwrap();
            let mut gpu_losses = Vec::new();
            for _ in 0..n_epochs {
                let loss = gpu_mlp.train_epoch(&x, &y, mse_loss).unwrap();
                gpu_losses.push(loss);
            }

            // Loss trajectories should match closely
            for (epoch, (cl, gl)) in cpu_losses.iter().zip(gpu_losses.iter()).enumerate() {
                assert!(
                    (cl - gl).abs() < 1e-8,
                    "Loss mismatch at epoch {}: cpu={}, gpu={}",
                    epoch,
                    cl,
                    gl
                );
            }
        }

        #[test]
        fn test_mlp_relu_activation_gpu() {
            use crate::neural::{Activation, MLP};
            let x = ndarray::array![[1.0, -1.0], [2.0, -2.0]];

            let mut mlp = MLP::new()
                .hidden_layer_sizes(&[3])
                .activation(Activation::ReLU)
                .output_activation(Activation::Linear)
                .random_state(42)
                .with_gpu(mock_gpu());
            mlp.initialize(2, 1).unwrap();
            let output = mlp.forward(&x, true).unwrap();

            // Should produce valid output (not NaN/Inf)
            assert_eq!(output.dim(), (2, 1));
            for val in output.iter() {
                assert!(val.is_finite(), "ReLU GPU output should be finite");
            }

            // Hidden layer z values should have ReLU applied
            let hidden_output = mlp.layers[0].last_output.as_ref().unwrap();
            for val in hidden_output.iter() {
                assert!(*val >= 0.0, "ReLU output must be non-negative, got {}", val);
            }
        }

        #[test]
        fn test_mlp_sigmoid_activation_gpu() {
            use crate::neural::{Activation, MLP};
            let x = ndarray::array![[1.0, -1.0], [2.0, -2.0], [0.0, 0.0]];

            let mut mlp = MLP::new()
                .hidden_layer_sizes(&[3])
                .activation(Activation::Sigmoid)
                .output_activation(Activation::Linear)
                .random_state(42)
                .with_gpu(mock_gpu());
            mlp.initialize(2, 1).unwrap();
            let output = mlp.forward(&x, true).unwrap();

            assert_eq!(output.dim(), (3, 1));

            // Hidden layer should have sigmoid outputs in (0, 1)
            let hidden_output = mlp.layers[0].last_output.as_ref().unwrap();
            for val in hidden_output.iter() {
                assert!(
                    *val > 0.0 && *val < 1.0,
                    "Sigmoid output must be in (0,1), got {}",
                    val
                );
            }
        }

        #[test]
        fn test_mlp_softmax_output_gpu() {
            use crate::neural::{Activation, MLP};
            let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

            let mut mlp = MLP::new()
                .hidden_layer_sizes(&[4])
                .activation(Activation::ReLU)
                .output_activation(Activation::Softmax)
                .random_state(42)
                .with_gpu(mock_gpu());
            mlp.initialize(2, 3).unwrap();
            let output = mlp.forward(&x, false).unwrap();

            assert_eq!(output.dim(), (3, 3));
            // Each row should sum to ~1
            for i in 0..3 {
                let row_sum: f64 = output.row(i).sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-10,
                    "Softmax row {} sum = {}, expected 1.0",
                    i,
                    row_sum
                );
            }
            // All values should be in (0, 1)
            for val in output.iter() {
                assert!(
                    *val > 0.0 && *val < 1.0,
                    "Softmax output should be in (0,1), got {}",
                    val
                );
            }
        }

        #[test]
        fn test_mlp_gpu_error_fallback() {
            use crate::neural::{Activation, Solver, MLP};
            let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
            let y = ndarray::array![[1.0], [2.0], [3.0], [4.0]];

            // MLP with failing GPU should fall back to CPU gracefully
            let mut mlp = MLP::new()
                .hidden_layer_sizes(&[4])
                .activation(Activation::ReLU)
                .output_activation(Activation::Linear)
                .solver(Solver::Adam)
                .learning_rate(0.01)
                .random_state(42)
                .batch_size(4)
                .with_gpu(failing_gpu());
            mlp.initialize(2, 1).unwrap();

            // Forward should fall back to CPU
            let output = mlp.forward(&x, true).unwrap();
            assert_eq!(output.dim(), (4, 1));

            // Training should also work via CPU fallback
            let loss = mlp.train_epoch(&x, &y, mse_loss).unwrap();
            assert!(
                loss.is_finite(),
                "Loss should be finite after fallback training"
            );
        }

        #[test]
        fn test_mlp_gpu_single_sample() {
            use crate::neural::{Activation, MLP};
            let x = ndarray::array![[1.0, 2.0, 3.0]];

            let mut mlp = MLP::new()
                .hidden_layer_sizes(&[4])
                .activation(Activation::ReLU)
                .output_activation(Activation::Linear)
                .random_state(42)
                .with_gpu(mock_gpu());
            mlp.initialize(3, 1).unwrap();
            let output = mlp.forward(&x, false).unwrap();
            assert_eq!(output.dim(), (1, 1));
            assert!(output[[0, 0]].is_finite());
        }

        #[test]
        fn test_mlp_gpu_single_feature() {
            use crate::neural::{Activation, MLP};
            let x = ndarray::array![[1.0], [2.0], [3.0]];

            let mut mlp = MLP::new()
                .hidden_layer_sizes(&[2])
                .activation(Activation::Sigmoid)
                .output_activation(Activation::Linear)
                .random_state(42)
                .with_gpu(mock_gpu());
            mlp.initialize(1, 1).unwrap();
            let output = mlp.forward(&x, false).unwrap();
            assert_eq!(output.dim(), (3, 1));
            for val in output.iter() {
                assert!(val.is_finite());
            }
        }

        #[test]
        fn test_mlp_gpu_large_batch() {
            use crate::neural::{Activation, MLP};
            let n = 200;
            let x = Array2::from_shape_fn((n, 5), |(i, j)| (i * 5 + j) as f64 * 0.01);

            let mut mlp = MLP::new()
                .hidden_layer_sizes(&[10, 5])
                .activation(Activation::ReLU)
                .output_activation(Activation::Linear)
                .random_state(42)
                .with_gpu(mock_gpu());
            mlp.initialize(5, 2).unwrap();
            let output = mlp.forward(&x, false).unwrap();
            assert_eq!(output.dim(), (n, 2));
            for val in output.iter() {
                assert!(val.is_finite(), "Large batch output should be finite");
            }
        }

        #[test]
        fn test_mlp_gpu_sigmoid_backward_parity() {
            use crate::neural::{Activation, MLP};
            let seed = 99u64;
            let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];

            // CPU
            let mut cpu_mlp = MLP::new()
                .hidden_layer_sizes(&[3])
                .activation(Activation::Sigmoid)
                .output_activation(Activation::Linear)
                .random_state(seed);
            cpu_mlp.initialize(2, 1).unwrap();
            let cpu_out = cpu_mlp.forward(&x, true).unwrap();
            let grad = 2.0 * (&cpu_out - 1.0) / 2.0;
            let cpu_grads = cpu_mlp.backward(&grad).unwrap();

            // GPU
            let mut gpu_mlp = MLP::new()
                .hidden_layer_sizes(&[3])
                .activation(Activation::Sigmoid)
                .output_activation(Activation::Linear)
                .random_state(seed)
                .with_gpu(mock_gpu());
            gpu_mlp.initialize(2, 1).unwrap();
            let gpu_out = gpu_mlp.forward(&x, true).unwrap();
            let grad = 2.0 * (&gpu_out - 1.0) / 2.0;
            let gpu_grads = gpu_mlp.backward(&grad).unwrap();

            for (layer_idx, ((cw, _cb), (gw, _gb))) in
                cpu_grads.iter().zip(gpu_grads.iter()).enumerate()
            {
                for ((i, j), &cv) in cw.indexed_iter() {
                    let gv = gw[[i, j]];
                    assert!(
                        (cv - gv).abs() < 1e-10,
                        "Sigmoid backward parity: layer {} [{},{}]: cpu={}, gpu={}",
                        layer_idx,
                        i,
                        j,
                        cv,
                        gv
                    );
                }
            }
        }

        #[test]
        fn test_mlp_gpu_multi_hidden_layers() {
            use crate::neural::{Activation, Solver, MLP};
            let x = ndarray::array![
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2]
            ];
            let y = ndarray::array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];

            let mut mlp = MLP::new()
                .hidden_layer_sizes(&[8, 6, 4])
                .activation(Activation::ReLU)
                .output_activation(Activation::Linear)
                .solver(Solver::Adam)
                .learning_rate(0.01)
                .random_state(42)
                .batch_size(4)
                .with_gpu(mock_gpu());
            mlp.initialize(3, 2).unwrap();

            // Train for a few epochs
            let mut prev_loss = f64::MAX;
            for epoch in 0..20 {
                let loss = mlp.train_epoch(&x, &y, mse_loss).unwrap();
                assert!(loss.is_finite(), "Loss should be finite at epoch {}", epoch);
                if epoch > 5 {
                    // After some warmup, loss should generally decrease
                    // (not guaranteed every epoch, but should not explode)
                    assert!(
                        loss < prev_loss * 2.0,
                        "Loss should not explode: epoch {} loss={}, prev={}",
                        epoch,
                        loss,
                        prev_loss
                    );
                }
                prev_loss = loss;
            }
        }
    }
}

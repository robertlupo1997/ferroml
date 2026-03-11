//! Automatic CPU/GPU dispatch based on matrix size and policy.

use super::{GpuBackend, GpuDispatchPolicy};
use crate::Result;
use ndarray::{Array1, Array2, Axis};
use std::sync::Arc;

/// Wraps a GpuBackend with automatic CPU/GPU dispatch based on matrix size.
#[derive(Debug)]
pub struct GpuDispatcher {
    backend: Arc<dyn GpuBackend>,
    policy: GpuDispatchPolicy,
}

impl GpuDispatcher {
    pub fn new(backend: Arc<dyn GpuBackend>, policy: GpuDispatchPolicy) -> Self {
        Self { backend, policy }
    }

    pub fn with_auto_policy(backend: Arc<dyn GpuBackend>) -> Self {
        Self::new(backend, GpuDispatchPolicy::default())
    }

    /// Returns the current dispatch policy.
    pub fn policy(&self) -> &GpuDispatchPolicy {
        &self.policy
    }

    /// Returns a reference to the underlying GPU backend.
    pub fn backend(&self) -> &Arc<dyn GpuBackend> {
        &self.backend
    }

    fn should_use_gpu(&self, total_elements: usize) -> bool {
        match &self.policy {
            GpuDispatchPolicy::Always => true,
            GpuDispatchPolicy::Auto { min_elements } => total_elements >= *min_elements,
            GpuDispatchPolicy::Never => false,
        }
    }

    /// Matrix multiply with dispatch.
    pub fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
        let elements = a.len() + b.len();
        if self.should_use_gpu(elements) {
            self.backend.matmul(a, b)
        } else {
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
    }

    /// Pairwise squared Euclidean distances with dispatch.
    pub fn pairwise_distances(
        &self,
        x: &Array2<f64>,
        centers: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let elements = x.len() + centers.len();
        if self.should_use_gpu(elements) {
            self.backend.pairwise_distances(x, centers)
        } else {
            let (n, d) = x.dim();
            let (c, d2) = centers.dim();
            if d != d2 {
                return Err(crate::FerroError::shape_mismatch(
                    format!("X features = {}", d),
                    format!("centers features = {}", d2),
                ));
            }
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
    }

    /// Element-wise ReLU with dispatch.
    pub fn relu(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.should_use_gpu(x.len()) {
            self.backend.relu(x)
        } else {
            Ok(x.mapv(|v| v.max(0.0)))
        }
    }

    /// Element-wise sigmoid with dispatch.
    pub fn sigmoid(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.should_use_gpu(x.len()) {
            self.backend.sigmoid(x)
        } else {
            Ok(x.mapv(|v| 1.0 / (1.0 + (-v).exp())))
        }
    }

    /// Row-wise softmax with dispatch.
    pub fn softmax(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.should_use_gpu(x.len()) {
            self.backend.softmax(x)
        } else {
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
    }

    /// Row-wise sum with dispatch.
    pub fn row_sum(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if self.should_use_gpu(x.len()) {
            self.backend.row_sum(x)
        } else {
            Ok(x.sum_axis(Axis(1)))
        }
    }

    /// Row-wise max with dispatch.
    pub fn row_max(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if self.should_use_gpu(x.len()) {
            self.backend.row_max(x)
        } else {
            let (rows, _) = x.dim();
            let mut result = Array1::zeros(rows);
            for i in 0..rows {
                result[i] = x.row(i).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            }
            Ok(result)
        }
    }

    /// Broadcast bias add with dispatch.
    pub fn bias_add(&self, x: &Array2<f64>, bias: &Array1<f64>) -> Result<Array2<f64>> {
        if self.should_use_gpu(x.len()) {
            self.backend.bias_add(x, bias)
        } else {
            let (_, cols) = x.dim();
            if bias.len() != cols {
                return Err(crate::FerroError::shape_mismatch(
                    format!("matrix cols = {}", cols),
                    format!("bias len = {}", bias.len()),
                ));
            }
            Ok(x + &bias.broadcast((x.nrows(), cols)).unwrap())
        }
    }

    /// ReLU gradient with dispatch.
    pub fn relu_grad(&self, z: &Array2<f64>) -> Result<Array2<f64>> {
        if self.should_use_gpu(z.len()) {
            self.backend.relu_grad(z)
        } else {
            Ok(z.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }))
        }
    }

    /// Sigmoid gradient with dispatch.
    pub fn sigmoid_grad(&self, output: &Array2<f64>) -> Result<Array2<f64>> {
        if self.should_use_gpu(output.len()) {
            self.backend.sigmoid_grad(output)
        } else {
            Ok(output * &(1.0 - output))
        }
    }

    /// Element-wise multiplication with dispatch.
    pub fn elementwise_mul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
        let elements = a.len() + b.len();
        if self.should_use_gpu(elements) {
            self.backend.elementwise_mul(a, b)
        } else {
            if a.dim() != b.dim() {
                return Err(crate::FerroError::shape_mismatch(
                    format!("a shape = {:?}", a.dim()),
                    format!("b shape = {:?}", b.dim()),
                ));
            }
            Ok(a * b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};
    use std::sync::Arc;

    /// A mock GPU backend that performs computations on CPU, for testing dispatch logic.
    #[derive(Debug, Clone)]
    struct MockGpuBackend {
        available: bool,
    }

    impl MockGpuBackend {
        fn new() -> Self {
            Self { available: true }
        }
    }

    impl GpuBackend for MockGpuBackend {
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
            let (n, d) = x.dim();
            let (c, d2) = centers.dim();
            if d != d2 {
                return Err(crate::FerroError::shape_mismatch(
                    format!("X features = {}", d),
                    format!("centers features = {}", d2),
                ));
            }
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

        fn row_sum(&self, x: &Array2<f64>) -> crate::Result<Array1<f64>> {
            Ok(x.sum_axis(Axis(1)))
        }

        fn row_max(&self, x: &Array2<f64>) -> crate::Result<Array1<f64>> {
            let (rows, _) = x.dim();
            let mut result = Array1::zeros(rows);
            for i in 0..rows {
                result[i] = x.row(i).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            }
            Ok(result)
        }

        fn bias_add(&self, x: &Array2<f64>, bias: &Array1<f64>) -> crate::Result<Array2<f64>> {
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

        fn is_available(&self) -> bool {
            self.available
        }
    }

    fn mock_backend() -> Arc<dyn GpuBackend> {
        Arc::new(MockGpuBackend::new())
    }

    // ========================================================================
    // Dispatch policy tests
    // ========================================================================

    #[test]
    fn test_default_policy() {
        let policy = GpuDispatchPolicy::default();
        match policy {
            GpuDispatchPolicy::Auto { min_elements } => assert_eq!(min_elements, 4096),
            _ => panic!("Default policy should be Auto with 4096"),
        }
    }

    #[test]
    fn test_dispatcher_debug() {
        let dispatcher = GpuDispatcher::new(mock_backend(), GpuDispatchPolicy::Never);
        let debug_str = format!("{:?}", dispatcher);
        assert!(debug_str.contains("GpuDispatcher"));
    }

    // ========================================================================
    // Never policy — always uses CPU
    // ========================================================================

    #[test]
    fn test_dispatch_never_uses_cpu_matmul() {
        let dispatcher = GpuDispatcher::new(mock_backend(), GpuDispatchPolicy::Never);
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let result = dispatcher.matmul(&a, &b).unwrap();
        let expected = a.dot(&b);
        assert!((result[[0, 0]] - expected[[0, 0]]).abs() < 1e-12);
        assert!((result[[1, 1]] - expected[[1, 1]]).abs() < 1e-12);
    }

    #[test]
    fn test_dispatch_never_uses_cpu_pairwise() {
        let dispatcher = GpuDispatcher::new(mock_backend(), GpuDispatchPolicy::Never);
        let x = array![[0.0, 0.0], [1.0, 0.0]];
        let centers = array![[0.0, 0.0]];
        let dists = dispatcher.pairwise_distances(&x, &centers).unwrap();
        assert!((dists[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((dists[[1, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_dispatch_relu_cpu_fallback() {
        let dispatcher = GpuDispatcher::new(mock_backend(), GpuDispatchPolicy::Never);
        let x = array![[-1.0, 2.0], [0.0, -3.0]];
        let result = dispatcher.relu(&x).unwrap();
        assert!((result[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((result[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((result[[1, 0]] - 0.0).abs() < 1e-12);
        assert!((result[[1, 1]] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_dispatch_sigmoid_cpu_fallback() {
        let dispatcher = GpuDispatcher::new(mock_backend(), GpuDispatchPolicy::Never);
        let x = array![[0.0]];
        let result = dispatcher.sigmoid(&x).unwrap();
        assert!((result[[0, 0]] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_dispatch_softmax_cpu_fallback() {
        let dispatcher = GpuDispatcher::new(mock_backend(), GpuDispatchPolicy::Never);
        let x = array![[1.0, 2.0, 3.0]];
        let result = dispatcher.softmax(&x).unwrap();
        let row_sum: f64 = result.row(0).sum();
        assert!((row_sum - 1.0).abs() < 1e-12);
    }

    // ========================================================================
    // Always policy — always uses GPU
    // ========================================================================

    #[test]
    fn test_dispatch_always_uses_gpu() {
        let dispatcher = GpuDispatcher::new(mock_backend(), GpuDispatchPolicy::Always);
        // Even a tiny 1x1 matrix should go through the GPU backend
        let a = array![[3.0]];
        let b = array![[4.0]];
        let result = dispatcher.matmul(&a, &b).unwrap();
        assert!((result[[0, 0]] - 12.0).abs() < 1e-12);
    }

    // ========================================================================
    // Auto policy — threshold-based
    // ========================================================================

    #[test]
    fn test_dispatch_auto_small_uses_cpu() {
        // 2x2 = 4 elements per matrix, total 8 < 4096 threshold
        let dispatcher = GpuDispatcher::with_auto_policy(mock_backend());
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let result = dispatcher.matmul(&a, &b).unwrap();
        let expected = a.dot(&b);
        for i in 0..2 {
            for j in 0..2 {
                assert!((result[[i, j]] - expected[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_dispatch_auto_large_uses_gpu() {
        // Create matrices large enough to exceed 4096 threshold
        let dispatcher = GpuDispatcher::with_auto_policy(mock_backend());
        let a = Array2::ones((64, 64)); // 4096 elements
        let b = Array2::ones((64, 64)); // 4096 elements, total 8192 >= 4096
        let result = dispatcher.matmul(&a, &b).unwrap();
        assert_eq!(result.dim(), (64, 64));
        // Each element should be 64.0 (sum of 64 ones)
        assert!((result[[0, 0]] - 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_dispatch_custom_threshold() {
        // Set threshold to 2 elements — even small matrices go to GPU
        let dispatcher =
            GpuDispatcher::new(mock_backend(), GpuDispatchPolicy::Auto { min_elements: 2 });
        let a = array![[3.0]];
        let b = array![[4.0]];
        // 1 + 1 = 2 elements >= 2 threshold → GPU
        let result = dispatcher.matmul(&a, &b).unwrap();
        assert!((result[[0, 0]] - 12.0).abs() < 1e-12);
    }

    // ========================================================================
    // Parity tests — GPU and CPU give same results
    // ========================================================================

    #[test]
    fn test_dispatch_matmul_parity() {
        let backend = mock_backend();
        let gpu_dispatcher = GpuDispatcher::new(backend, GpuDispatchPolicy::Always);
        let cpu_dispatcher = GpuDispatcher::new(mock_backend(), GpuDispatchPolicy::Never);

        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];

        let gpu_result = gpu_dispatcher.matmul(&a, &b).unwrap();
        let cpu_result = cpu_dispatcher.matmul(&a, &b).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (gpu_result[[i, j]] - cpu_result[[i, j]]).abs() < 1e-12,
                    "Matmul parity mismatch at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_dispatch_pairwise_parity() {
        let gpu_dispatcher = GpuDispatcher::new(mock_backend(), GpuDispatchPolicy::Always);
        let cpu_dispatcher = GpuDispatcher::new(mock_backend(), GpuDispatchPolicy::Never);

        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let centers = array![[0.0, 0.0], [1.0, 1.0]];

        let gpu_result = gpu_dispatcher.pairwise_distances(&x, &centers).unwrap();
        let cpu_result = cpu_dispatcher.pairwise_distances(&x, &centers).unwrap();

        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (gpu_result[[i, j]] - cpu_result[[i, j]]).abs() < 1e-12,
                    "Pairwise parity mismatch at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    // ========================================================================
    // All operations work in both modes
    // ========================================================================

    #[test]
    fn test_dispatcher_all_ops_never() {
        let dispatcher = GpuDispatcher::new(mock_backend(), GpuDispatchPolicy::Never);

        // matmul
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let _ = dispatcher.matmul(&a, &b).unwrap();

        // pairwise_distances
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let c = array![[0.0, 0.0]];
        let _ = dispatcher.pairwise_distances(&x, &c).unwrap();

        // relu
        let _ = dispatcher.relu(&a).unwrap();

        // sigmoid
        let _ = dispatcher.sigmoid(&a).unwrap();

        // softmax
        let _ = dispatcher.softmax(&a).unwrap();

        // row_sum
        let _ = dispatcher.row_sum(&a).unwrap();

        // row_max
        let _ = dispatcher.row_max(&a).unwrap();

        // bias_add
        let bias = ndarray::array![10.0, 20.0];
        let _ = dispatcher.bias_add(&a, &bias).unwrap();

        // relu_grad
        let _ = dispatcher.relu_grad(&a).unwrap();

        // sigmoid_grad
        let sig = array![[0.5, 0.7], [0.3, 0.9]];
        let _ = dispatcher.sigmoid_grad(&sig).unwrap();

        // elementwise_mul
        let _ = dispatcher.elementwise_mul(&a, &b).unwrap();
    }

    #[test]
    fn test_dispatcher_all_ops_always() {
        let dispatcher = GpuDispatcher::new(mock_backend(), GpuDispatchPolicy::Always);

        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let _ = dispatcher.matmul(&a, &b).unwrap();

        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let c = array![[0.0, 0.0]];
        let _ = dispatcher.pairwise_distances(&x, &c).unwrap();

        let _ = dispatcher.relu(&a).unwrap();
        let _ = dispatcher.sigmoid(&a).unwrap();
        let _ = dispatcher.softmax(&a).unwrap();
        let _ = dispatcher.row_sum(&a).unwrap();
        let _ = dispatcher.row_max(&a).unwrap();

        let bias = ndarray::array![10.0, 20.0];
        let _ = dispatcher.bias_add(&a, &bias).unwrap();

        let _ = dispatcher.relu_grad(&a).unwrap();
        let sig = array![[0.5, 0.7], [0.3, 0.9]];
        let _ = dispatcher.sigmoid_grad(&sig).unwrap();
        let _ = dispatcher.elementwise_mul(&a, &b).unwrap();
    }

    // ========================================================================
    // GpuMemoryInfo tests
    // ========================================================================

    #[test]
    fn test_memory_info_fields() {
        use super::super::GpuMemoryInfo;
        let info = GpuMemoryInfo {
            max_buffer_size: 1 << 30,                   // 1 GB
            max_storage_buffer_binding_size: 128 << 20, // 128 MB
        };
        assert_eq!(info.max_buffer_size, 1 << 30);
        assert_eq!(info.max_storage_buffer_binding_size, 128 << 20);
        // Debug impl works
        let _ = format!("{:?}", info);
    }
}

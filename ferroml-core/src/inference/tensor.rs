//! Tensor implementation for inference runtime
//!
//! Provides a simple tensor type for holding data during inference.

use super::InferenceError;
use std::ops::{Index, IndexMut};

/// A multi-dimensional array of f32 values
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The data stored in row-major order
    data: Vec<f32>,
    /// The shape of the tensor
    shape: Vec<usize>,
    /// Strides for indexing
    strides: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor from data and shape
    ///
    /// # Panics
    /// Panics if the data length doesn't match the shape
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self::from_vec(vec![0.0; len], shape)
    }

    /// Create a tensor filled with a single value
    pub fn full(shape: Vec<usize>, value: f32) -> Self {
        let len: usize = shape.iter().product();
        Self::from_vec(vec![value; len], shape)
    }

    /// Create a 1D tensor from a slice
    pub fn from_slice(data: &[f32]) -> Self {
        Self::from_vec(data.to_vec(), vec![data.len()])
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get raw data as slice
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get raw data as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get raw data as f32 slice (alias for as_slice)
    pub fn as_f32_slice(&self) -> &[f32] {
        &self.data
    }

    /// Convert to Vec
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Get element at multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> Option<f32> {
        if indices.len() != self.shape.len() {
            return None;
        }
        let flat_idx = self.flat_index(indices)?;
        self.data.get(flat_idx).copied()
    }

    /// Set element at multi-dimensional index
    pub fn set(&mut self, indices: &[usize], value: f32) -> Option<()> {
        if indices.len() != self.shape.len() {
            return None;
        }
        let flat_idx = self.flat_index(indices)?;
        if flat_idx < self.data.len() {
            self.data[flat_idx] = value;
            Some(())
        } else {
            None
        }
    }

    /// Compute flat index from multi-dimensional indices
    fn flat_index(&self, indices: &[usize]) -> Option<usize> {
        let mut idx = 0;
        for (i, &dim_idx) in indices.iter().enumerate() {
            if dim_idx >= self.shape[i] {
                return None;
            }
            idx += dim_idx * self.strides[i];
        }
        Some(idx)
    }

    /// Reshape the tensor (in-place if possible)
    pub fn reshape(self, new_shape: Vec<usize>) -> Result<Self, InferenceError> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.data.len() {
            return Err(InferenceError::ShapeMismatch(format!(
                "Cannot reshape tensor of {} elements to shape {:?} ({} elements)",
                self.data.len(),
                new_shape,
                new_len
            )));
        }
        Ok(Self::from_vec(self.data, new_shape))
    }

    /// Squeeze dimensions of size 1 at specified axes
    pub fn squeeze(self, axes: &[i64]) -> Result<Self, InferenceError> {
        let mut new_shape = Vec::new();
        for (i, &dim) in self.shape.iter().enumerate() {
            let should_squeeze = axes.iter().any(|&ax| {
                let ax = if ax < 0 {
                    (self.shape.len() as i64 + ax) as usize
                } else {
                    ax as usize
                };
                ax == i
            });
            if !should_squeeze || dim != 1 {
                new_shape.push(dim);
            }
        }
        if new_shape.is_empty() {
            new_shape.push(1);
        }
        Ok(Self::from_vec(self.data, new_shape))
    }

    /// Flatten the tensor starting at given axis
    pub fn flatten(self, axis: i64) -> Result<Self, InferenceError> {
        let axis = if axis < 0 {
            (self.ndim() as i64 + axis) as usize
        } else {
            axis as usize
        };

        if axis > self.ndim() {
            return Err(InferenceError::ShapeMismatch(format!(
                "Axis {} out of range for tensor with {} dimensions",
                axis,
                self.ndim()
            )));
        }

        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis..].iter().product();
        let outer = if outer == 0 { 1 } else { outer };

        Ok(Self::from_vec(self.data, vec![outer, inner]))
    }

    /// Matrix multiplication: self @ other
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, InferenceError> {
        // Handle 1D @ 2D, 2D @ 1D, 2D @ 2D, and batched matmul
        match (self.ndim(), other.ndim()) {
            (2, 2) => self.matmul_2d(other),
            (2, 1) => {
                // [M, K] @ [K] -> [M]
                let other_2d = other.clone().reshape(vec![other.shape[0], 1])?;
                let result = self.matmul_2d(&other_2d)?;
                result.squeeze(&[1])
            }
            (1, 2) => {
                // [K] @ [K, N] -> [N]
                let self_2d = self.clone().reshape(vec![1, self.shape[0]])?;
                let result = self_2d.matmul_2d(other)?;
                result.squeeze(&[0])
            }
            _ => Err(InferenceError::ShapeMismatch(format!(
                "Unsupported matmul shapes: {:?} @ {:?}",
                self.shape, other.shape
            ))),
        }
    }

    /// 2D matrix multiplication
    fn matmul_2d(&self, other: &Tensor) -> Result<Tensor, InferenceError> {
        let m = self.shape[0];
        let k1 = self.shape[1];
        let k2 = other.shape[0];
        let n = other.shape[1];

        if k1 != k2 {
            return Err(InferenceError::ShapeMismatch(format!(
                "Matrix multiplication dimension mismatch: [{}, {}] @ [{}, {}]",
                m, k1, k2, n
            )));
        }

        let mut result = vec![0.0f32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k in 0..k1 {
                    sum += self.data[i * k1 + k] * other.data[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(Tensor::from_vec(result, vec![m, n]))
    }

    /// Element-wise addition with broadcasting
    pub fn add(&self, other: &Tensor) -> Result<Tensor, InferenceError> {
        // Simple broadcasting: if shapes match, add directly
        // If other is 1D and matches last dimension, broadcast
        if self.shape == other.shape {
            let data: Vec<f32> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect();
            return Ok(Tensor::from_vec(data, self.shape.clone()));
        }

        // Handle broadcasting for bias addition [batch, features] + [features]
        if other.ndim() == 1 && self.ndim() >= 1 && *self.shape.last().unwrap() == other.shape[0] {
            let mut data = self.data.clone();
            let n = other.shape[0];
            for (i, val) in data.iter_mut().enumerate() {
                *val += other.data[i % n];
            }
            return Ok(Tensor::from_vec(data, self.shape.clone()));
        }

        // Handle [batch, 1] + [1] broadcasting
        if other.shape == vec![1] {
            let data: Vec<f32> = self.data.iter().map(|a| a + other.data[0]).collect();
            return Ok(Tensor::from_vec(data, self.shape.clone()));
        }

        Err(InferenceError::ShapeMismatch(format!(
            "Cannot broadcast shapes {:?} and {:?}",
            self.shape, other.shape
        )))
    }

    /// Apply sigmoid function element-wise
    pub fn sigmoid(&self) -> Tensor {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Tensor::from_vec(data, self.shape.clone())
    }

    /// Apply softmax along specified axis
    pub fn softmax(&self, axis: i64) -> Result<Tensor, InferenceError> {
        let axis = if axis < 0 {
            (self.ndim() as i64 + axis) as usize
        } else {
            axis as usize
        };

        if axis >= self.ndim() {
            return Err(InferenceError::ShapeMismatch(format!(
                "Softmax axis {} out of range for {} dimensions",
                axis,
                self.ndim()
            )));
        }

        // For 2D: softmax along axis=1 (rows)
        if self.ndim() == 2 && axis == 1 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            let mut data = vec![0.0f32; self.data.len()];

            for i in 0..rows {
                // Find max for numerical stability
                let row_start = i * cols;
                let row_end = row_start + cols;
                let max_val = self.data[row_start..row_end]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                // Compute exp(x - max) and sum
                let mut sum = 0.0f32;
                for j in 0..cols {
                    let exp_val = (self.data[row_start + j] - max_val).exp();
                    data[row_start + j] = exp_val;
                    sum += exp_val;
                }

                // Normalize
                for j in 0..cols {
                    data[row_start + j] /= sum;
                }
            }

            return Ok(Tensor::from_vec(data, self.shape.clone()));
        }

        // For 1D: just apply softmax to entire array
        if self.ndim() == 1 {
            let max_val = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_data: Vec<f32> = self.data.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exp_data.iter().sum();
            let data: Vec<f32> = exp_data.iter().map(|&x| x / sum).collect();
            return Ok(Tensor::from_vec(data, self.shape.clone()));
        }

        Err(InferenceError::ShapeMismatch(format!(
            "Softmax not implemented for ndim={} axis={}",
            self.ndim(),
            axis
        )))
    }
}

impl Index<usize> for Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

/// A multi-dimensional array of i64 values (for integer outputs like labels)
#[derive(Debug, Clone)]
pub struct TensorI64 {
    /// The data stored in row-major order
    data: Vec<i64>,
    /// The shape of the tensor
    shape: Vec<usize>,
}

impl TensorI64 {
    /// Create a new tensor from data and shape
    pub fn from_vec(data: Vec<i64>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?}",
            data.len(),
            shape
        );
        Self { data, shape }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self::from_vec(vec![0; len], shape)
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get raw data as slice
    pub fn as_slice(&self) -> &[i64] {
        &self.data
    }

    /// Get raw data as i64 slice (alias)
    pub fn as_i64_slice(&self) -> &[i64] {
        &self.data
    }

    /// Convert to Vec
    pub fn into_vec(self) -> Vec<i64> {
        self.data
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Compute strides for row-major layout
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.len(), 6);
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(vec![3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
        assert!(t.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_indexing() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.get(&[0, 0]), Some(1.0));
        assert_eq!(t.get(&[0, 2]), Some(3.0));
        assert_eq!(t.get(&[1, 0]), Some(4.0));
        assert_eq!(t.get(&[1, 2]), Some(6.0));
    }

    #[test]
    fn test_matmul_2d() {
        // [2, 3] @ [3, 2] = [2, 2]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        // Row 0: [1,2,3] @ [[1,2],[3,4],[5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        // Row 1: [4,5,6] @ [[1,2],[3,4],[5,6]] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        assert_eq!(c.as_slice(), &[22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_add_same_shape() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_broadcast() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.as_slice(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_sigmoid() {
        let t = Tensor::from_vec(vec![0.0], vec![1]);
        let s = t.sigmoid();
        assert!((s.as_slice()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let s = t.softmax(1).unwrap();
        let sum: f32 = s.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_squeeze() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
        let s = t.squeeze(&[0, 2]).unwrap();
        assert_eq!(s.shape(), &[3]);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = t.reshape(vec![3, 2]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}

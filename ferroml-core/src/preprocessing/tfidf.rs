//! TF-IDF Transformer for sparse text data.
//!
//! Converts term-frequency matrices to TF-IDF representation.
//! Users pass pre-tokenized count matrices (from CountVectorizer or manual construction).

use ndarray::{Array1, Array2};

use crate::{FerroError, Result};

/// Normalization for TF-IDF.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TfidfNorm {
    /// L1 normalization (rows sum to 1 in absolute value)
    L1,
    /// L2 normalization (rows have unit L2 norm)
    L2,
    /// No normalization
    None,
}

/// TF-IDF transformer: converts term-frequency matrices to TF-IDF representation.
///
/// Users pass pre-tokenized count matrices (from CountVectorizer or manual construction).
/// This transformer applies TF weighting and IDF weighting.
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::tfidf::{TfidfTransformer, TfidfNorm};
/// use ndarray::array;
///
/// let mut tfidf = TfidfTransformer::new();
/// let x = array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]];
/// let result = tfidf.fit_transform(&x).unwrap();
/// // Each row has unit L2 norm by default
/// ```
#[derive(Debug, Clone)]
pub struct TfidfTransformer {
    norm: TfidfNorm,
    use_idf: bool,
    smooth_idf: bool,
    sublinear_tf: bool,
    idf_: Option<Array1<f64>>,
}

impl Default for TfidfTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl TfidfTransformer {
    /// Create a new TfidfTransformer with default settings.
    ///
    /// Defaults: L2 norm, use_idf=true, smooth_idf=true, sublinear_tf=false.
    pub fn new() -> Self {
        Self {
            norm: TfidfNorm::L2,
            use_idf: true,
            smooth_idf: true,
            sublinear_tf: false,
            idf_: None,
        }
    }

    /// Set the normalization type.
    pub fn with_norm(mut self, norm: TfidfNorm) -> Self {
        self.norm = norm;
        self
    }

    /// Set whether to use IDF weighting.
    pub fn with_use_idf(mut self, use_idf: bool) -> Self {
        self.use_idf = use_idf;
        self
    }

    /// Set whether to smooth IDF weights.
    pub fn with_smooth_idf(mut self, smooth_idf: bool) -> Self {
        self.smooth_idf = smooth_idf;
        self
    }

    /// Set whether to apply sublinear TF scaling (1 + log(tf)).
    pub fn with_sublinear_tf(mut self, sublinear_tf: bool) -> Self {
        self.sublinear_tf = sublinear_tf;
        self
    }

    /// Fit the transformer by computing IDF weights from the input matrix.
    ///
    /// For each feature j, counts the number of documents where it is non-zero (df_j),
    /// then computes:
    /// - smooth_idf=true:  idf_j = ln((1 + n) / (1 + df_j)) + 1
    /// - smooth_idf=false: idf_j = ln(n / max(1, df_j)) + 1
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(FerroError::invalid_input(
                "Input array must have at least one sample and one feature",
            ));
        }

        let n = x.nrows();
        let n_features = x.ncols();
        let mut df: Array1<f64> = Array1::zeros(n_features);

        for row in x.rows() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0.0 {
                    df[j] += 1.0;
                }
            }
        }

        if self.use_idf {
            let mut idf = Array1::zeros(n_features);
            for j in 0..n_features {
                if self.smooth_idf {
                    idf[j] = ((1.0 + n as f64) / (1.0 + df[j])).ln() + 1.0;
                } else {
                    idf[j] = (n as f64 / df[j].max(1.0)).ln() + 1.0;
                }
            }
            self.idf_ = Some(idf);
        } else {
            // Store a dummy to mark as fitted
            self.idf_ = Some(Array1::ones(n_features));
        }

        Ok(())
    }

    /// Transform a count matrix to TF-IDF representation.
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("TfidfTransformer"));
        }

        let idf = self.idf_.as_ref().unwrap();
        if x.ncols() != idf.len() {
            return Err(FerroError::shape_mismatch(
                format!("(*, {})", idf.len()),
                format!("(*, {})", x.ncols()),
            ));
        }

        // Apply TF weighting
        let mut result = x.to_owned();
        if self.sublinear_tf {
            result.mapv_inplace(|v| if v > 0.0 { 1.0 + v.ln() } else { 0.0 });
        }

        // Apply IDF weighting
        if self.use_idf {
            for mut row in result.rows_mut() {
                row *= idf;
            }
        }

        // Apply normalization
        match self.norm {
            TfidfNorm::L2 => {
                for mut row in result.rows_mut() {
                    let norm = row.dot(&row).sqrt();
                    if norm > 0.0 {
                        row /= norm;
                    }
                }
            }
            TfidfNorm::L1 => {
                for mut row in result.rows_mut() {
                    let norm: f64 = row.iter().map(|v| v.abs()).sum();
                    if norm > 0.0 {
                        row /= norm;
                    }
                }
            }
            TfidfNorm::None => {}
        }

        Ok(result)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Get the learned IDF weights. Returns None if not fitted.
    pub fn idf(&self) -> Option<&Array1<f64>> {
        if self.use_idf {
            self.idf_.as_ref()
        } else {
            None
        }
    }

    /// Check if the transformer has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.idf_.is_some()
    }

    /// Get the normalization type.
    pub fn norm(&self) -> TfidfNorm {
        self.norm
    }

    /// Get whether IDF weighting is used.
    pub fn use_idf(&self) -> bool {
        self.use_idf
    }

    /// Get whether smooth IDF is used.
    pub fn smooth_idf(&self) -> bool {
        self.smooth_idf
    }

    /// Get whether sublinear TF is used.
    pub fn sublinear_tf(&self) -> bool {
        self.sublinear_tf
    }

    /// Get the number of features (vocabulary size) if fitted.
    pub fn n_features(&self) -> Option<usize> {
        self.idf_.as_ref().map(|idf| idf.len())
    }
}

// Transformer trait implementation
impl super::Transformer for TfidfTransformer {
    fn fit(&mut self, x: &Array2<f64>) -> crate::Result<()> {
        TfidfTransformer::fit(self, x)
    }

    fn transform(&self, x: &Array2<f64>) -> crate::Result<Array2<f64>> {
        TfidfTransformer::transform(self, x)
    }

    fn is_fitted(&self) -> bool {
        TfidfTransformer::is_fitted(self)
    }

    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>> {
        if !self.is_fitted() {
            return None;
        }
        let n = self.idf_.as_ref()?.len();
        Some(
            input_names
                .map(|names| names.to_vec())
                .unwrap_or_else(|| super::generate_feature_names(n)),
        )
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features()
    }

    fn n_features_out(&self) -> Option<usize> {
        self.n_features()
    }
}

// PipelineTransformer implementation
impl crate::pipeline::PipelineTransformer for TfidfTransformer {
    fn clone_boxed(&self) -> Box<dyn crate::pipeline::PipelineTransformer> {
        Box::new(self.clone())
    }

    fn set_param(&mut self, name: &str, value: &crate::hpo::ParameterValue) -> crate::Result<()> {
        match name {
            "norm" => {
                if let Some(s) = value.as_str() {
                    self.norm = match s {
                        "l1" => TfidfNorm::L1,
                        "l2" => TfidfNorm::L2,
                        "none" => TfidfNorm::None,
                        _ => {
                            return Err(crate::FerroError::invalid_input(
                                "norm must be 'l1', 'l2', or 'none'",
                            ))
                        }
                    };
                    Ok(())
                } else {
                    Err(crate::FerroError::invalid_input("norm must be a string"))
                }
            }
            "use_idf" => {
                if let Some(v) = value.as_bool() {
                    self.use_idf = v;
                    Ok(())
                } else {
                    Err(crate::FerroError::invalid_input(
                        "use_idf must be a boolean",
                    ))
                }
            }
            "smooth_idf" => {
                if let Some(v) = value.as_bool() {
                    self.smooth_idf = v;
                    Ok(())
                } else {
                    Err(crate::FerroError::invalid_input(
                        "smooth_idf must be a boolean",
                    ))
                }
            }
            "sublinear_tf" => {
                if let Some(v) = value.as_bool() {
                    self.sublinear_tf = v;
                    Ok(())
                } else {
                    Err(crate::FerroError::invalid_input(
                        "sublinear_tf must be a boolean",
                    ))
                }
            }
            _ => Err(crate::FerroError::invalid_input(format!(
                "Unknown parameter '{}'",
                name
            ))),
        }
    }

    fn name(&self) -> &str {
        "TfidfTransformer"
    }
}

// Sparse versions
#[cfg(feature = "sparse")]
impl TfidfTransformer {
    /// Fit the transformer from a sparse CSR matrix.
    ///
    /// Native O(nnz) implementation: computes document frequencies directly from
    /// the sparse structure without densifying.
    pub fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix) -> Result<()> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(FerroError::invalid_input(
                "Input array must have at least one sample and one feature",
            ));
        }

        let n_samples = x.nrows() as f64;
        let n_features = x.ncols();
        let df = crate::sparse::sparse_column_nnz(x);

        if self.use_idf {
            let mut idf = Array1::zeros(n_features);
            for j in 0..n_features {
                if self.smooth_idf {
                    idf[j] = ((1.0 + n_samples) / (1.0 + df[j] as f64)).ln() + 1.0;
                } else {
                    idf[j] = (n_samples / (df[j] as f64).max(1.0)).ln() + 1.0;
                }
            }
            self.idf_ = Some(idf);
        } else {
            self.idf_ = Some(Array1::ones(n_features));
        }

        Ok(())
    }

    /// Transform a sparse CSR matrix to TF-IDF, returning a sparse CsrMatrix.
    ///
    /// Native O(nnz) computation: applies TF weighting, IDF weighting, and
    /// normalization directly on the sparse structure without densifying.
    pub fn transform_sparse_native(
        &self,
        x: &crate::sparse::CsrMatrix,
    ) -> Result<crate::sparse::CsrMatrix> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("TfidfTransformer"));
        }

        let idf = self.idf_.as_ref().unwrap();
        if x.ncols() != idf.len() {
            return Err(FerroError::shape_mismatch(
                format!("(*, {})", idf.len()),
                format!("(*, {})", x.ncols()),
            ));
        }

        let n_rows = x.nrows();
        let n_cols = x.ncols();
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();

        for i in 0..n_rows {
            let row = x.row(i);
            let row_indices = row.indices();
            let row_data = row.data();

            // First pass: compute weighted values for this row
            let mut row_vals: Vec<(usize, f64)> = Vec::with_capacity(row_indices.len());
            for (&col_idx, &val) in row_indices.iter().zip(row_data.iter()) {
                let tf = if self.sublinear_tf {
                    if val > 0.0 {
                        1.0 + val.ln()
                    } else {
                        0.0
                    }
                } else {
                    val
                };
                let weighted = if self.use_idf { tf * idf[col_idx] } else { tf };
                if weighted != 0.0 {
                    row_vals.push((col_idx, weighted));
                }
            }

            // Compute row norm for normalization
            let row_norm = match self.norm {
                TfidfNorm::L2 => {
                    let sum_sq: f64 = row_vals.iter().map(|(_, v)| v * v).sum();
                    sum_sq.sqrt()
                }
                TfidfNorm::L1 => row_vals.iter().map(|(_, v)| v.abs()).sum(),
                TfidfNorm::None => 1.0,
            };

            // Second pass: emit normalized values
            for (col_idx, weighted) in row_vals {
                let final_val = if row_norm > 0.0 && self.norm != TfidfNorm::None {
                    weighted / row_norm
                } else {
                    weighted
                };
                rows.push(i);
                cols.push(col_idx);
                values.push(final_val);
            }
        }

        crate::sparse::CsrMatrix::from_triplets((n_rows, n_cols), &rows, &cols, &values)
    }

    /// Transform a sparse CSR matrix to TF-IDF (returns dense).
    ///
    /// Native O(nnz) computation: only touches non-zero entries when building
    /// the result, then applies row-wise normalization.
    pub fn transform_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array2<f64>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("TfidfTransformer"));
        }

        let idf = self.idf_.as_ref().unwrap();
        if x.ncols() != idf.len() {
            return Err(FerroError::shape_mismatch(
                format!("(*, {})", idf.len()),
                format!("(*, {})", x.ncols()),
            ));
        }

        let mut result = Array2::zeros((x.nrows(), x.ncols()));
        for i in 0..x.nrows() {
            let row = x.row(i);
            for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                let tf = if self.sublinear_tf {
                    if val > 0.0 {
                        1.0 + val.ln()
                    } else {
                        0.0
                    }
                } else {
                    val
                };
                let weighted = if self.use_idf { tf * idf[col] } else { tf };
                result[[i, col]] = weighted;
            }

            // Apply normalization to this row
            match self.norm {
                TfidfNorm::L2 => {
                    let row_norm = result.row(i).mapv(|v| v * v).sum().sqrt();
                    if row_norm > 0.0 {
                        result.row_mut(i).mapv_inplace(|v| v / row_norm);
                    }
                }
                TfidfNorm::L1 => {
                    let row_norm: f64 = result.row(i).iter().map(|v| v.abs()).sum();
                    if row_norm > 0.0 {
                        result.row_mut(i).mapv_inplace(|v| v / row_norm);
                    }
                }
                TfidfNorm::None => {}
            }
        }

        Ok(result)
    }
}

// SparseTransformer trait implementation
#[cfg(feature = "sparse")]
impl super::SparseTransformer for TfidfTransformer {
    fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix) -> crate::Result<()> {
        TfidfTransformer::fit_sparse(self, x)
    }

    fn transform_sparse(
        &self,
        x: &crate::sparse::CsrMatrix,
    ) -> crate::Result<crate::sparse::CsrMatrix> {
        self.transform_sparse_native(x)
    }

    fn is_fitted(&self) -> bool {
        TfidfTransformer::is_fitted(self)
    }

    fn n_features_out(&self) -> Option<usize> {
        self.n_features()
    }
}

#[cfg(feature = "sparse")]
impl crate::pipeline::PipelineSparseTransformer for TfidfTransformer {
    fn clone_boxed(&self) -> Box<dyn crate::pipeline::PipelineSparseTransformer> {
        Box::new(self.clone())
    }

    fn set_param(&mut self, name: &str, value: &crate::hpo::ParameterValue) -> crate::Result<()> {
        // Delegate to existing PipelineTransformer impl which has the same set_param logic
        crate::pipeline::PipelineTransformer::set_param(self, name, value)
    }

    fn name(&self) -> &str {
        "TfidfTransformer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_basic_fit_transform() {
        let mut tfidf = TfidfTransformer::new();
        let x = array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]];
        let result = tfidf.fit_transform(&x).unwrap();
        assert_eq!(result.dim(), (3, 3));

        // Each row should have unit L2 norm (default)
        for row in result.rows() {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                assert!(
                    (norm - 1.0).abs() < 1e-10,
                    "L2 norm should be 1.0, got {}",
                    norm
                );
            }
        }
    }

    #[test]
    fn test_smooth_idf_vs_unsmoothed() {
        let x = array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]];

        let mut smooth = TfidfTransformer::new()
            .with_smooth_idf(true)
            .with_norm(TfidfNorm::None);
        smooth.fit(&x).unwrap();

        let mut unsmoothed = TfidfTransformer::new()
            .with_smooth_idf(false)
            .with_norm(TfidfNorm::None);
        unsmoothed.fit(&x).unwrap();

        let smooth_idf = smooth.idf().unwrap();
        let unsmoothed_idf = unsmoothed.idf().unwrap();

        // They should be different
        assert!(
            (smooth_idf[0] - unsmoothed_idf[0]).abs() > 1e-10,
            "Smooth and unsmoothed IDF should differ"
        );
    }

    #[test]
    fn test_sublinear_tf() {
        let x = array![[1.0, 0.0, 4.0], [0.0, 1.0, 1.0]];

        let mut normal = TfidfTransformer::new()
            .with_sublinear_tf(false)
            .with_use_idf(false)
            .with_norm(TfidfNorm::None);
        let result_normal = normal.fit_transform(&x).unwrap();

        let mut sublinear = TfidfTransformer::new()
            .with_sublinear_tf(true)
            .with_use_idf(false)
            .with_norm(TfidfNorm::None);
        let result_sub = sublinear.fit_transform(&x).unwrap();

        // With sublinear TF, tf=4 becomes 1 + ln(4) ~ 2.386, while tf=1 stays 1 + ln(1) = 1
        assert!((result_sub[[0, 2]] - (1.0 + 4.0_f64.ln())).abs() < 1e-10);
        assert!((result_sub[[1, 1]] - 1.0).abs() < 1e-10); // 1 + ln(1) = 1
                                                           // Zero stays zero
        assert!((result_sub[[0, 1]]).abs() < 1e-10);

        // Normal should just pass through the raw values
        assert!((result_normal[[0, 2]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_l1_norm() {
        let mut tfidf = TfidfTransformer::new().with_norm(TfidfNorm::L1);
        let x = array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]];
        let result = tfidf.fit_transform(&x).unwrap();

        // Each row should have L1 norm = 1
        for row in result.rows() {
            let norm: f64 = row.iter().map(|v| v.abs()).sum();
            if norm > 0.0 {
                assert!(
                    (norm - 1.0).abs() < 1e-10,
                    "L1 norm should be 1.0, got {}",
                    norm
                );
            }
        }
    }

    #[test]
    fn test_l2_norm() {
        let mut tfidf = TfidfTransformer::new().with_norm(TfidfNorm::L2);
        let x = array![[3.0, 0.0, 4.0], [0.0, 5.0, 0.0]];
        let result = tfidf.fit_transform(&x).unwrap();

        for row in result.rows() {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                assert!(
                    (norm - 1.0).abs() < 1e-10,
                    "L2 norm should be 1.0, got {}",
                    norm
                );
            }
        }
    }

    #[test]
    fn test_no_norm() {
        let mut tfidf = TfidfTransformer::new()
            .with_norm(TfidfNorm::None)
            .with_use_idf(false);
        let x = array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]];
        let result = tfidf.fit_transform(&x).unwrap();

        // Without IDF and without normalization, result should equal input
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result[[0, 2]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_use_idf_false() {
        let mut tfidf = TfidfTransformer::new()
            .with_use_idf(false)
            .with_norm(TfidfNorm::None);
        let x = array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]];
        let result = tfidf.fit_transform(&x).unwrap();

        // Without IDF, result should be raw TF (no normalization either)
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result[[0, 2]] - 2.0).abs() < 1e-10);

        // idf() should return None when use_idf=false
        assert!(tfidf.idf().is_none());
    }

    #[test]
    fn test_not_fitted_error() {
        let tfidf = TfidfTransformer::new();
        let x = array![[1.0, 0.0], [0.0, 1.0]];
        let result = tfidf.transform(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_document() {
        let mut tfidf = TfidfTransformer::new();
        let x = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]];
        let result = tfidf.fit_transform(&x).unwrap();

        // All-zero rows should remain all-zero (no division by zero)
        for j in 0..3 {
            assert!((result[[0, j]]).abs() < 1e-10);
            assert!((result[[2, j]]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_idf_values_smooth() {
        let mut tfidf = TfidfTransformer::new()
            .with_norm(TfidfNorm::None)
            .with_smooth_idf(true);
        // 3 documents, feature 0 appears in 2 docs, feature 1 in 1 doc, feature 2 in 3 docs
        let x = array![[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]];
        tfidf.fit(&x).unwrap();

        let idf = tfidf.idf().unwrap();
        // idf[0] = ln((1+3)/(1+2)) + 1 = ln(4/3) + 1
        assert!((idf[0] - (4.0_f64 / 3.0).ln() - 1.0).abs() < 1e-10);
        // idf[1] = ln((1+3)/(1+1)) + 1 = ln(2) + 1
        assert!((idf[1] - 2.0_f64.ln() - 1.0).abs() < 1e-10);
        // idf[2] = ln((1+3)/(1+3)) + 1 = ln(1) + 1 = 1
        assert!((idf[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_shape_mismatch_on_transform() {
        let mut tfidf = TfidfTransformer::new();
        let x_fit = array![[1.0, 0.0], [0.0, 1.0]];
        tfidf.fit(&x_fit).unwrap();

        let x_bad = array![[1.0, 0.0, 0.0]];
        assert!(tfidf.transform(&x_bad).is_err());
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_fit_transform() {
        use crate::sparse::CsrMatrix;

        let x_dense = array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]];
        let x_sparse = CsrMatrix::from_dense(&x_dense);

        let mut tfidf_dense = TfidfTransformer::new();
        let result_dense = tfidf_dense.fit_transform(&x_dense).unwrap();

        let mut tfidf_sparse = TfidfTransformer::new();
        tfidf_sparse.fit_sparse(&x_sparse).unwrap();
        let result_sparse = tfidf_sparse.transform_sparse(&x_sparse).unwrap();

        // Results should match
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (result_dense[[i, j]] - result_sparse[[i, j]]).abs() < 1e-10,
                    "Mismatch at ({}, {}): dense={}, sparse={}",
                    i,
                    j,
                    result_dense[[i, j]],
                    result_sparse[[i, j]]
                );
            }
        }
    }

    // ========================================================================
    // SparseTransformer trait tests
    // ========================================================================

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_transformer_fit_transform_sparse() {
        use crate::preprocessing::SparseTransformer;
        use crate::sparse::CsrMatrix;

        let x_dense = array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]];
        let x_sparse = CsrMatrix::from_dense(&x_dense);

        let mut tfidf = TfidfTransformer::new();
        let result = SparseTransformer::fit_transform_sparse(&mut tfidf, &x_sparse).unwrap();
        let result_dense = result.to_dense();

        // Each row should have unit L2 norm
        for i in 0..3 {
            let norm: f64 = result_dense
                .row(i)
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 {
                assert!(
                    (norm - 1.0).abs() < 1e-10,
                    "Row {} L2 norm should be 1.0, got {}",
                    i,
                    norm
                );
            }
        }
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_transformer_is_fitted() {
        use crate::preprocessing::SparseTransformer;
        use crate::sparse::CsrMatrix;

        let mut tfidf = TfidfTransformer::new();
        assert!(!SparseTransformer::is_fitted(&tfidf));

        let x = CsrMatrix::from_dense(&array![[1.0, 0.0], [0.0, 1.0]]);
        SparseTransformer::fit_sparse(&mut tfidf, &x).unwrap();
        assert!(SparseTransformer::is_fitted(&tfidf));
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_transformer_n_features_out() {
        use crate::preprocessing::SparseTransformer;
        use crate::sparse::CsrMatrix;

        let mut tfidf = TfidfTransformer::new();
        assert_eq!(SparseTransformer::n_features_out(&tfidf), None);

        let x = CsrMatrix::from_dense(&array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]);
        SparseTransformer::fit_sparse(&mut tfidf, &x).unwrap();
        assert_eq!(SparseTransformer::n_features_out(&tfidf), Some(3));
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_transformer_not_fitted_error() {
        use crate::preprocessing::SparseTransformer;
        use crate::sparse::CsrMatrix;

        let tfidf = TfidfTransformer::new();
        let x = CsrMatrix::from_dense(&array![[1.0, 0.0], [0.0, 1.0]]);
        let result = SparseTransformer::transform_sparse(&tfidf, &x);
        assert!(result.is_err());
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_transformer_shape_mismatch() {
        use crate::preprocessing::SparseTransformer;
        use crate::sparse::CsrMatrix;

        let mut tfidf = TfidfTransformer::new();
        let x_fit = CsrMatrix::from_dense(&array![[1.0, 0.0], [0.0, 1.0]]);
        SparseTransformer::fit_sparse(&mut tfidf, &x_fit).unwrap();

        let x_bad = CsrMatrix::from_dense(&array![[1.0, 0.0, 0.0]]);
        let result = SparseTransformer::transform_sparse(&tfidf, &x_bad);
        assert!(result.is_err());
    }

    // ========================================================================
    // Sparse-to-sparse output correctness tests
    // ========================================================================

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_native_matches_dense() {
        use crate::sparse::CsrMatrix;

        let x_dense = array![
            [3.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0, 3.0]
        ];
        let x_sparse = CsrMatrix::from_dense(&x_dense);

        let mut tfidf_dense = TfidfTransformer::new();
        let result_dense = tfidf_dense.fit_transform(&x_dense).unwrap();

        let mut tfidf_sparse = TfidfTransformer::new();
        tfidf_sparse.fit_sparse(&x_sparse).unwrap();
        let result_sparse = tfidf_sparse.transform_sparse_native(&x_sparse).unwrap();
        let result_sparse_dense = result_sparse.to_dense();

        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (result_dense[[i, j]] - result_sparse_dense[[i, j]]).abs() < 1e-10,
                    "Mismatch at ({}, {}): dense={}, sparse={}",
                    i,
                    j,
                    result_dense[[i, j]],
                    result_sparse_dense[[i, j]]
                );
            }
        }
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_native_l1_norm() {
        use crate::sparse::CsrMatrix;

        let x_dense = array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]];
        let x_sparse = CsrMatrix::from_dense(&x_dense);

        let mut tfidf = TfidfTransformer::new().with_norm(TfidfNorm::L1);
        tfidf.fit_sparse(&x_sparse).unwrap();
        let result = tfidf.transform_sparse_native(&x_sparse).unwrap();
        let result_dense = result.to_dense();

        for i in 0..3 {
            let norm: f64 = result_dense.row(i).iter().map(|v| v.abs()).sum();
            if norm > 0.0 {
                assert!(
                    (norm - 1.0).abs() < 1e-10,
                    "Row {} L1 norm should be 1.0, got {}",
                    i,
                    norm
                );
            }
        }
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_native_l2_norm() {
        use crate::sparse::CsrMatrix;

        let x_dense = array![[3.0, 0.0, 4.0], [0.0, 5.0, 0.0], [1.0, 2.0, 3.0]];
        let x_sparse = CsrMatrix::from_dense(&x_dense);

        let mut tfidf = TfidfTransformer::new().with_norm(TfidfNorm::L2);
        tfidf.fit_sparse(&x_sparse).unwrap();
        let result = tfidf.transform_sparse_native(&x_sparse).unwrap();
        let result_dense = result.to_dense();

        for i in 0..3 {
            let norm: f64 = result_dense
                .row(i)
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 {
                assert!(
                    (norm - 1.0).abs() < 1e-10,
                    "Row {} L2 norm should be 1.0, got {}",
                    i,
                    norm
                );
            }
        }
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_native_no_norm() {
        use crate::sparse::CsrMatrix;

        let x_dense = array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]];
        let x_sparse = CsrMatrix::from_dense(&x_dense);

        let mut tfidf = TfidfTransformer::new()
            .with_norm(TfidfNorm::None)
            .with_use_idf(false);
        tfidf.fit_sparse(&x_sparse).unwrap();
        let result = tfidf.transform_sparse_native(&x_sparse).unwrap();
        let result_dense = result.to_dense();

        // Without IDF and without normalization, result should equal input
        assert!((result_dense[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result_dense[[0, 1]]).abs() < 1e-10);
        assert!((result_dense[[0, 2]] - 2.0).abs() < 1e-10);
        assert!((result_dense[[1, 0]]).abs() < 1e-10);
        assert!((result_dense[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((result_dense[[1, 2]] - 1.0).abs() < 1e-10);
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_native_sublinear_tf() {
        use crate::sparse::CsrMatrix;

        let x_dense = array![[1.0, 0.0, 4.0], [0.0, 1.0, 1.0]];
        let x_sparse = CsrMatrix::from_dense(&x_dense);

        let mut tfidf_dense = TfidfTransformer::new()
            .with_sublinear_tf(true)
            .with_norm(TfidfNorm::None);
        let result_dense = tfidf_dense.fit_transform(&x_dense).unwrap();

        let mut tfidf_sparse = TfidfTransformer::new()
            .with_sublinear_tf(true)
            .with_norm(TfidfNorm::None);
        tfidf_sparse.fit_sparse(&x_sparse).unwrap();
        let result_sparse = tfidf_sparse.transform_sparse_native(&x_sparse).unwrap();
        let result_sparse_dense = result_sparse.to_dense();

        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    (result_dense[[i, j]] - result_sparse_dense[[i, j]]).abs() < 1e-10,
                    "Sublinear TF mismatch at ({}, {}): dense={}, sparse={}",
                    i,
                    j,
                    result_dense[[i, j]],
                    result_sparse_dense[[i, j]]
                );
            }
        }
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_native_no_idf() {
        use crate::sparse::CsrMatrix;

        let x_dense = array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]];
        let x_sparse = CsrMatrix::from_dense(&x_dense);

        let mut tfidf_dense = TfidfTransformer::new()
            .with_use_idf(false)
            .with_norm(TfidfNorm::L2);
        let result_dense = tfidf_dense.fit_transform(&x_dense).unwrap();

        let mut tfidf_sparse = TfidfTransformer::new()
            .with_use_idf(false)
            .with_norm(TfidfNorm::L2);
        tfidf_sparse.fit_sparse(&x_sparse).unwrap();
        let result_sparse = tfidf_sparse.transform_sparse_native(&x_sparse).unwrap();
        let result_sparse_dense = result_sparse.to_dense();

        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    (result_dense[[i, j]] - result_sparse_dense[[i, j]]).abs() < 1e-10,
                    "No-IDF mismatch at ({}, {}): dense={}, sparse={}",
                    i,
                    j,
                    result_dense[[i, j]],
                    result_sparse_dense[[i, j]]
                );
            }
        }
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_native_preserves_sparsity() {
        use crate::sparse::CsrMatrix;

        let x_dense = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        let x_sparse = CsrMatrix::from_dense(&x_dense);
        let input_nnz = x_sparse.nnz();

        let mut tfidf = TfidfTransformer::new();
        tfidf.fit_sparse(&x_sparse).unwrap();
        let result = tfidf.transform_sparse_native(&x_sparse).unwrap();

        // Output nnz should not exceed input nnz
        assert!(
            result.nnz() <= input_nnz,
            "Output nnz ({}) should not exceed input nnz ({})",
            result.nnz(),
            input_nnz
        );
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn test_sparse_native_empty_row() {
        use crate::sparse::CsrMatrix;

        let x_dense = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]];
        let x_sparse = CsrMatrix::from_dense(&x_dense);

        let mut tfidf = TfidfTransformer::new();
        tfidf.fit_sparse(&x_sparse).unwrap();
        let result = tfidf.transform_sparse_native(&x_sparse).unwrap();
        let result_dense = result.to_dense();

        // All-zero rows should remain all-zero
        for j in 0..3 {
            assert!(
                result_dense[[0, j]].abs() < 1e-10,
                "Row 0 col {} should be zero, got {}",
                j,
                result_dense[[0, j]]
            );
            assert!(
                result_dense[[2, j]].abs() < 1e-10,
                "Row 2 col {} should be zero, got {}",
                j,
                result_dense[[2, j]]
            );
        }
    }

    // ========================================================================
    // Transformer trait tests
    // ========================================================================

    #[test]
    fn test_transformer_trait_fit_transform() {
        use crate::preprocessing::Transformer;

        let mut tfidf = TfidfTransformer::new();
        let x = array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]];

        Transformer::fit(&mut tfidf, &x).unwrap();
        let result = Transformer::transform(&tfidf, &x).unwrap();

        assert_eq!(result.dim(), (3, 3));
        // Each row should have unit L2 norm
        for row in result.rows() {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                assert!((norm - 1.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_transformer_trait_is_fitted() {
        use crate::preprocessing::Transformer;

        let mut tfidf = TfidfTransformer::new();
        assert!(!Transformer::is_fitted(&tfidf));

        let x = array![[1.0, 0.0], [0.0, 1.0]];
        Transformer::fit(&mut tfidf, &x).unwrap();
        assert!(Transformer::is_fitted(&tfidf));
    }

    #[test]
    fn test_transformer_trait_feature_names() {
        use crate::preprocessing::Transformer;

        let mut tfidf = TfidfTransformer::new();

        // Before fitting, should return None
        assert!(tfidf.get_feature_names_out(None).is_none());

        let x = array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]];
        Transformer::fit(&mut tfidf, &x).unwrap();

        // With no input names, should generate default names
        let names = tfidf.get_feature_names_out(None).unwrap();
        assert_eq!(names.len(), 3);

        // With input names, should pass them through
        let input_names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let names = tfidf.get_feature_names_out(Some(&input_names)).unwrap();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    // ========================================================================
    // PipelineTransformer tests
    // ========================================================================

    #[test]
    fn test_pipeline_transformer_clone_boxed() {
        use crate::pipeline::PipelineTransformer;

        let mut tfidf = TfidfTransformer::new().with_norm(TfidfNorm::L1);
        let x = array![[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]];
        tfidf.fit(&x).unwrap();

        let cloned = PipelineTransformer::clone_boxed(&tfidf);
        assert_eq!(cloned.name(), "TfidfTransformer");
        // The clone should be fitted and usable
        assert!(tfidf.is_fitted());
    }

    #[test]
    fn test_pipeline_transformer_set_param() {
        use crate::hpo::ParameterValue;
        use crate::pipeline::PipelineTransformer;

        let mut tfidf = TfidfTransformer::new();

        // Set norm to l1
        PipelineTransformer::set_param(
            &mut tfidf,
            "norm",
            &ParameterValue::Categorical("l1".to_string()),
        )
        .unwrap();
        assert_eq!(tfidf.norm(), TfidfNorm::L1);

        // Set use_idf to false
        PipelineTransformer::set_param(&mut tfidf, "use_idf", &ParameterValue::Bool(false))
            .unwrap();
        assert!(!tfidf.use_idf());

        // Set smooth_idf to false
        PipelineTransformer::set_param(&mut tfidf, "smooth_idf", &ParameterValue::Bool(false))
            .unwrap();
        assert!(!tfidf.smooth_idf());

        // Set sublinear_tf to true
        PipelineTransformer::set_param(&mut tfidf, "sublinear_tf", &ParameterValue::Bool(true))
            .unwrap();
        assert!(tfidf.sublinear_tf());
    }

    #[test]
    fn test_pipeline_transformer_name() {
        use crate::pipeline::PipelineTransformer;

        let tfidf = TfidfTransformer::new();
        assert_eq!(PipelineTransformer::name(&tfidf), "TfidfTransformer");
    }

    #[test]
    fn test_pipeline_transformer_unknown_param_error() {
        use crate::hpo::ParameterValue;
        use crate::pipeline::PipelineTransformer;

        let mut tfidf = TfidfTransformer::new();
        let result = PipelineTransformer::set_param(
            &mut tfidf,
            "nonexistent_param",
            &ParameterValue::Bool(true),
        );
        assert!(result.is_err());
    }
}

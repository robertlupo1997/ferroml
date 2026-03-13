//! Text Pipeline for chaining text transformers and sparse/dense models.
//!
//! This module provides `TextPipeline` for text classification/regression
//! workflows that accept `&[String]` input and chain text transformers,
//! sparse transformers, and a final model.
//!
//! ## Pipeline Flow
//!
//! ```text
//! fit():
//!   documents ─→ text_transformer.fit_transform_text() ─→ CsrMatrix
//!              ─→ sparse_transformer.fit_transform_sparse() ─→ CsrMatrix
//!              ─→ model.fit_sparse(csr, y)  OR  model.fit(csr.to_dense(), y)
//!
//! predict():
//!   documents ─→ text_transformer.transform_text() ─→ CsrMatrix
//!              ─→ sparse_transformer.transform_sparse() ─→ CsrMatrix
//!              ─→ model.predict_sparse(csr)  OR  model.predict(csr.to_dense())
//! ```

use crate::hpo::{ParameterValue, SearchSpace};
use crate::preprocessing::count_vectorizer::TextTransformer;
use crate::preprocessing::SparseTransformer;
use crate::sparse::CsrMatrix;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::fmt;

// =============================================================================
// Pipeline Traits
// =============================================================================

/// Trait for text transformers usable in TextPipeline.
///
/// Extends `TextTransformer` with pipeline integration methods for
/// hyperparameter search, cloning, and introspection.
pub trait PipelineTextTransformer: TextTransformer {
    /// Get the hyperparameter search space for this transformer.
    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    /// Clone into a boxed trait object.
    fn clone_boxed(&self) -> Box<dyn PipelineTextTransformer>;

    /// Set a hyperparameter by name.
    fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()> {
        let _ = (name, value);
        Err(FerroError::invalid_input(format!(
            "Parameter '{}' not supported",
            name
        )))
    }

    /// Get transformer type name.
    fn name(&self) -> &str;

    /// Get the number of output features produced (after fitting).
    fn n_features_out(&self) -> Option<usize>;
}

/// Trait for sparse transformers usable in TextPipeline.
///
/// Extends `SparseTransformer` with pipeline integration methods.
pub trait PipelineSparseTransformer: SparseTransformer {
    /// Get the hyperparameter search space for this transformer.
    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    /// Clone into a boxed trait object.
    fn clone_boxed(&self) -> Box<dyn PipelineSparseTransformer>;

    /// Set a hyperparameter by name.
    fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()> {
        let _ = (name, value);
        Err(FerroError::invalid_input(format!(
            "Parameter '{}' not supported",
            name
        )))
    }

    /// Get transformer type name.
    fn name(&self) -> &str;
}

/// Trait for models that accept sparse input in pipelines.
pub trait PipelineSparseModel: Send + Sync {
    /// Fit on sparse CSR matrix.
    fn fit_sparse(&mut self, x: &CsrMatrix, y: &Array1<f64>) -> Result<()>;

    /// Predict from sparse CSR matrix.
    fn predict_sparse(&self, x: &CsrMatrix) -> Result<Array1<f64>>;

    /// Get the hyperparameter search space.
    fn search_space(&self) -> SearchSpace;

    /// Clone into a boxed trait object.
    fn clone_boxed(&self) -> Box<dyn PipelineSparseModel>;

    /// Set a hyperparameter by name.
    fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()>;

    /// Get model type name.
    fn name(&self) -> &str;

    /// Check if the model has been fitted.
    fn is_fitted(&self) -> bool;
}

// =============================================================================
// TextPipelineStep
// =============================================================================

/// A step in a text pipeline.
pub enum TextPipelineStep {
    /// Text documents to sparse matrix (e.g., CountVectorizer).
    TextToSparse(Box<dyn PipelineTextTransformer>),
    /// Sparse-to-sparse transformation (e.g., TfidfTransformer).
    SparseToSparse(Box<dyn PipelineSparseTransformer>),
    /// Final model that accepts sparse input directly.
    SparseModel(Box<dyn PipelineSparseModel>),
    /// Final model that accepts dense input (sparse is densified).
    DenseModel(Box<dyn super::PipelineModel>),
}

impl fmt::Debug for TextPipelineStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TextToSparse(_) => write!(f, "TextPipelineStep::TextToSparse(<dyn>)"),
            Self::SparseToSparse(_) => write!(f, "TextPipelineStep::SparseToSparse(<dyn>)"),
            Self::SparseModel(_) => write!(f, "TextPipelineStep::SparseModel(<dyn>)"),
            Self::DenseModel(_) => write!(f, "TextPipelineStep::DenseModel(<dyn>)"),
        }
    }
}

// =============================================================================
// TextPipeline
// =============================================================================

/// A machine learning pipeline for text classification/regression workflows.
///
/// Accepts `&[String]` input and chains text transformers, sparse transformers,
/// and a final model.
///
/// ## Structure
///
/// A text pipeline consists of:
/// - One or more **text transformers** that convert documents to sparse matrices
/// - Zero or more **sparse transformers** that transform sparse matrices
/// - Optionally, a final **model** (sparse or dense) that makes predictions
///
/// ## Named Steps
///
/// Each step has a name. Parameter names use double-underscore convention:
/// `"step_name__param_name"`.
pub struct TextPipeline {
    /// Named steps: (name, step) pairs in order.
    steps: Vec<(String, TextPipelineStep)>,
    /// Whether the pipeline has been fitted.
    fitted: bool,
}

impl fmt::Debug for TextPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let step_names: Vec<_> = self.steps.iter().map(|(name, _)| name.as_str()).collect();
        f.debug_struct("TextPipeline")
            .field("steps", &step_names)
            .field("fitted", &self.fitted)
            .finish()
    }
}

impl Default for TextPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl TextPipeline {
    /// Create a new empty text pipeline.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            fitted: false,
        }
    }

    /// Add a text transformer step to the pipeline.
    ///
    /// Text transformers convert `&[String]` documents into sparse `CsrMatrix`.
    /// They must come before any sparse transformers or models.
    ///
    /// # Panics
    ///
    /// Panics if a step with this name already exists, or if the pipeline
    /// already has a model (models must be the final step).
    pub fn add_text_transformer<T: PipelineTextTransformer + 'static>(
        mut self,
        name: impl Into<String>,
        t: T,
    ) -> Self {
        let name = name.into();
        self.assert_no_duplicate_name(&name);
        assert!(
            !self.has_model(),
            "Cannot add transformer after model. Models must be the final step."
        );
        // Text transformers must come before sparse transformers
        assert!(
            !self.has_sparse_transformer(),
            "Cannot add text transformer after sparse transformer. \
             Text transformers must come first."
        );
        self.steps
            .push((name, TextPipelineStep::TextToSparse(Box::new(t))));
        self
    }

    /// Add a sparse transformer step to the pipeline.
    ///
    /// Sparse transformers convert `CsrMatrix` to `CsrMatrix`.
    /// They must come after at least one text transformer and before any model.
    ///
    /// # Panics
    ///
    /// Panics if a step with this name already exists, if no text transformer
    /// has been added, or if the pipeline already has a model.
    pub fn add_sparse_transformer<T: PipelineSparseTransformer + 'static>(
        mut self,
        name: impl Into<String>,
        t: T,
    ) -> Self {
        let name = name.into();
        self.assert_no_duplicate_name(&name);
        assert!(
            !self.has_model(),
            "Cannot add transformer after model. Models must be the final step."
        );
        assert!(
            self.has_text_transformer(),
            "Cannot add sparse transformer before text transformer. \
             Add a text transformer first."
        );
        self.steps
            .push((name, TextPipelineStep::SparseToSparse(Box::new(t))));
        self
    }

    /// Add a sparse model as the final step of the pipeline.
    ///
    /// # Panics
    ///
    /// Panics if a step with this name already exists, if no text transformer
    /// has been added, or if the pipeline already has a model.
    pub fn add_sparse_model<M: PipelineSparseModel + 'static>(
        mut self,
        name: impl Into<String>,
        m: M,
    ) -> Self {
        let name = name.into();
        self.assert_no_duplicate_name(&name);
        assert!(
            !self.has_model(),
            "Pipeline already has a model. Only one model is allowed."
        );
        assert!(
            self.has_text_transformer(),
            "Cannot add model before text transformer. Add a text transformer first."
        );
        self.steps
            .push((name, TextPipelineStep::SparseModel(Box::new(m))));
        self
    }

    /// Add a dense model as the final step of the pipeline.
    ///
    /// The sparse features will be densified before being passed to the model.
    ///
    /// # Panics
    ///
    /// Panics if a step with this name already exists, if no text transformer
    /// has been added, or if the pipeline already has a model.
    pub fn add_dense_model<M: super::PipelineModel + 'static>(
        mut self,
        name: impl Into<String>,
        m: M,
    ) -> Self {
        let name = name.into();
        self.assert_no_duplicate_name(&name);
        assert!(
            !self.has_model(),
            "Pipeline already has a model. Only one model is allowed."
        );
        assert!(
            self.has_text_transformer(),
            "Cannot add model before text transformer. Add a text transformer first."
        );
        self.steps
            .push((name, TextPipelineStep::DenseModel(Box::new(m))));
        self
    }

    /// Fit the pipeline on text documents and labels.
    ///
    /// 1. First text transformer(s): `fit_transform_text` -> `CsrMatrix`
    /// 2. Sparse transformers: `fit_transform_sparse` -> `CsrMatrix`
    /// 3. Final model: `fit_sparse(csr, y)` or `fit(csr.to_dense(), y)`
    pub fn fit(&mut self, documents: &[String], y: &Array1<f64>) -> Result<()> {
        if self.steps.is_empty() {
            return Err(FerroError::invalid_input(
                "TextPipeline is empty. Add at least one step.",
            ));
        }

        if documents.is_empty() {
            return Err(FerroError::invalid_input("Cannot fit on empty documents."));
        }

        if documents.len() != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("{} documents", documents.len()),
                format!("y has {} elements", y.len()),
            ));
        }

        // Validate that first step is a text transformer
        if !matches!(self.steps[0].1, TextPipelineStep::TextToSparse(_)) {
            return Err(FerroError::invalid_input(
                "First step in TextPipeline must be a TextToSparse transformer.",
            ));
        }

        let mut current_sparse: Option<CsrMatrix> = None;

        for (_name, step) in &mut self.steps {
            match step {
                TextPipelineStep::TextToSparse(transformer) => {
                    if current_sparse.is_none() {
                        // First text transformer: fit on documents
                        current_sparse = Some(transformer.fit_transform_text(documents)?);
                    } else {
                        // Subsequent text transformers shouldn't appear here,
                        // but handle gracefully — re-fit on documents
                        current_sparse = Some(transformer.fit_transform_text(documents)?);
                    }
                }
                TextPipelineStep::SparseToSparse(transformer) => {
                    let csr = current_sparse.as_ref().ok_or_else(|| {
                        FerroError::invalid_input(
                            "SparseToSparse step encountered before any sparse data was produced.",
                        )
                    })?;
                    current_sparse = Some(transformer.fit_transform_sparse(csr)?);
                }
                TextPipelineStep::SparseModel(model) => {
                    let csr = current_sparse.as_ref().ok_or_else(|| {
                        FerroError::invalid_input(
                            "Model step encountered before any sparse data was produced.",
                        )
                    })?;
                    model.fit_sparse(csr, y)?;
                }
                TextPipelineStep::DenseModel(model) => {
                    let csr = current_sparse.as_ref().ok_or_else(|| {
                        FerroError::invalid_input(
                            "Model step encountered before any sparse data was produced.",
                        )
                    })?;
                    let dense = csr.to_dense();
                    model.fit(&dense, y)?;
                }
            }
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict from text documents.
    ///
    /// Transforms through text->sparse chain, then predicts with the final model.
    pub fn predict(&self, documents: &[String]) -> Result<Array1<f64>> {
        if !self.fitted {
            return Err(FerroError::not_fitted("predict"));
        }

        if !self.has_model() {
            return Err(FerroError::invalid_input(
                "TextPipeline has no model. Use transform() for transformer-only pipelines.",
            ));
        }

        let mut current_sparse: Option<CsrMatrix> = None;

        for (_name, step) in &self.steps {
            match step {
                TextPipelineStep::TextToSparse(transformer) => {
                    current_sparse = Some(transformer.transform_text(documents)?);
                }
                TextPipelineStep::SparseToSparse(transformer) => {
                    let csr = current_sparse.as_ref().ok_or_else(|| {
                        FerroError::invalid_input("No sparse data for SparseToSparse step.")
                    })?;
                    current_sparse = Some(transformer.transform_sparse(csr)?);
                }
                TextPipelineStep::SparseModel(model) => {
                    let csr = current_sparse
                        .as_ref()
                        .ok_or_else(|| FerroError::invalid_input("No sparse data for model."))?;
                    return model.predict_sparse(csr);
                }
                TextPipelineStep::DenseModel(model) => {
                    let csr = current_sparse
                        .as_ref()
                        .ok_or_else(|| FerroError::invalid_input("No sparse data for model."))?;
                    let dense = csr.to_dense();
                    return model.predict(&dense);
                }
            }
        }

        Err(FerroError::invalid_input(
            "TextPipeline has no model step to produce predictions.",
        ))
    }

    /// Transform text to sparse features (without the final model step).
    pub fn transform(&self, documents: &[String]) -> Result<CsrMatrix> {
        if !self.fitted {
            return Err(FerroError::not_fitted("transform"));
        }

        let mut current_sparse: Option<CsrMatrix> = None;

        for (_name, step) in &self.steps {
            match step {
                TextPipelineStep::TextToSparse(transformer) => {
                    current_sparse = Some(transformer.transform_text(documents)?);
                }
                TextPipelineStep::SparseToSparse(transformer) => {
                    let csr = current_sparse.as_ref().ok_or_else(|| {
                        FerroError::invalid_input("No sparse data for SparseToSparse step.")
                    })?;
                    current_sparse = Some(transformer.transform_sparse(csr)?);
                }
                TextPipelineStep::SparseModel(_) | TextPipelineStep::DenseModel(_) => {
                    // Stop before the model
                    break;
                }
            }
        }

        current_sparse.ok_or_else(|| {
            FerroError::invalid_input("TextPipeline has no text transformer to produce features.")
        })
    }

    /// Transform text to dense features.
    pub fn transform_dense(&self, documents: &[String]) -> Result<Array2<f64>> {
        let sparse = self.transform(documents)?;
        Ok(sparse.to_dense())
    }

    /// Get combined search space from all steps (prefixed with "step_name__").
    pub fn search_space(&self) -> SearchSpace {
        let mut combined = SearchSpace::new();

        for (name, step) in &self.steps {
            let step_space = match step {
                TextPipelineStep::TextToSparse(t) => t.search_space(),
                TextPipelineStep::SparseToSparse(t) => t.search_space(),
                TextPipelineStep::SparseModel(m) => m.search_space(),
                TextPipelineStep::DenseModel(m) => m.search_space(),
            };

            for (param_name, param) in step_space.parameters {
                let prefixed_name = format!("{}__{}", name, param_name);
                combined.parameters.insert(prefixed_name, param);
            }
        }

        combined
    }

    /// Set hyperparameters with "step__param" convention.
    pub fn set_params(&mut self, params: &HashMap<String, ParameterValue>) -> Result<()> {
        let step_names: Vec<String> = self.steps.iter().map(|(n, _)| n.clone()).collect();

        for (full_name, value) in params {
            let parts: Vec<&str> = full_name.splitn(2, "__").collect();
            if parts.len() != 2 {
                return Err(FerroError::invalid_input(format!(
                    "Invalid parameter name '{}'. Expected format: 'step__param'",
                    full_name
                )));
            }

            let (step_name, param_name) = (parts[0], parts[1]);

            let step = self
                .steps
                .iter_mut()
                .find(|(name, _)| name == step_name)
                .ok_or_else(|| {
                    FerroError::invalid_input(format!(
                        "Step '{}' not found in pipeline. Available: {:?}",
                        step_name, step_names
                    ))
                })?;

            match &mut step.1 {
                TextPipelineStep::TextToSparse(t) => t.set_param(param_name, value)?,
                TextPipelineStep::SparseToSparse(t) => t.set_param(param_name, value)?,
                TextPipelineStep::SparseModel(m) => m.set_param(param_name, value)?,
                TextPipelineStep::DenseModel(m) => m.set_param(param_name, value)?,
            }
        }

        // Mark as unfitted since parameters changed
        self.fitted = false;

        Ok(())
    }

    /// Get the names of all steps.
    pub fn step_names(&self) -> Vec<&str> {
        self.steps.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// Get the number of steps.
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Check if the pipeline is fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Check if the pipeline has a model as the final step.
    pub fn has_model(&self) -> bool {
        self.steps.last().map_or(false, |(_, step)| {
            matches!(
                step,
                TextPipelineStep::SparseModel(_) | TextPipelineStep::DenseModel(_)
            )
        })
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    fn has_text_transformer(&self) -> bool {
        self.steps
            .iter()
            .any(|(_, step)| matches!(step, TextPipelineStep::TextToSparse(_)))
    }

    fn has_sparse_transformer(&self) -> bool {
        self.steps
            .iter()
            .any(|(_, step)| matches!(step, TextPipelineStep::SparseToSparse(_)))
    }

    fn assert_no_duplicate_name(&self, name: &str) {
        assert!(
            !self.steps.iter().any(|(n, _)| n == name),
            "Step name '{}' already exists in pipeline",
            name
        );
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpo::search_space::Parameter;
    use crate::sparse::CsrMatrix;
    use ndarray::Array1;

    // =========================================================================
    // Test Doubles
    // =========================================================================

    #[derive(Clone)]
    struct MockTextTransformer {
        fitted: bool,
        vocab_size: usize,
    }

    impl MockTextTransformer {
        fn new(vocab_size: usize) -> Self {
            Self {
                fitted: false,
                vocab_size,
            }
        }
    }

    impl TextTransformer for MockTextTransformer {
        fn fit_text(&mut self, _documents: &[String]) -> Result<()> {
            self.fitted = true;
            Ok(())
        }

        fn transform_text(&self, documents: &[String]) -> Result<CsrMatrix> {
            if !self.fitted {
                return Err(FerroError::not_fitted("MockTextTransformer"));
            }
            // Create a simple identity-like sparse matrix:
            // n_docs rows, vocab_size cols, one entry per row
            let n_docs = documents.len();
            let mut indptr = vec![0usize];
            let mut indices = Vec::new();
            let mut data = Vec::new();

            for i in 0..n_docs {
                let col = i % self.vocab_size;
                indices.push(col);
                data.push(1.0);
                indptr.push(indptr.last().unwrap() + 1);
            }

            CsrMatrix::new((n_docs, self.vocab_size), indptr, indices, data)
        }
    }

    impl PipelineTextTransformer for MockTextTransformer {
        fn search_space(&self) -> SearchSpace {
            SearchSpace::new().add("vocab_size", Parameter::int(10, 1000))
        }

        fn clone_boxed(&self) -> Box<dyn PipelineTextTransformer> {
            Box::new(self.clone())
        }

        fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()> {
            match name {
                "vocab_size" => {
                    if let Some(v) = value.as_i64() {
                        self.vocab_size = v as usize;
                        Ok(())
                    } else {
                        Err(FerroError::invalid_input("vocab_size must be int"))
                    }
                }
                _ => Err(FerroError::invalid_input(format!(
                    "Unknown parameter '{}'",
                    name
                ))),
            }
        }

        fn name(&self) -> &str {
            "MockTextTransformer"
        }

        fn n_features_out(&self) -> Option<usize> {
            if self.fitted {
                Some(self.vocab_size)
            } else {
                None
            }
        }
    }

    #[derive(Clone)]
    struct MockSparseTransformer {
        fitted: bool,
    }

    impl MockSparseTransformer {
        fn new() -> Self {
            Self { fitted: false }
        }
    }

    impl SparseTransformer for MockSparseTransformer {
        fn fit_sparse(&mut self, _x: &CsrMatrix) -> Result<()> {
            self.fitted = true;
            Ok(())
        }

        fn transform_sparse(&self, x: &CsrMatrix) -> Result<CsrMatrix> {
            if !self.fitted {
                return Err(FerroError::not_fitted("MockSparseTransformer"));
            }
            // Pass-through with doubled values
            let (nrows, ncols) = x.shape();
            CsrMatrix::new(
                (nrows, ncols),
                x.indptr(),
                x.indices().to_vec(),
                x.data().iter().map(|v| v * 2.0).collect(),
            )
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn n_features_out(&self) -> Option<usize> {
            None
        }
    }

    impl PipelineSparseTransformer for MockSparseTransformer {
        fn search_space(&self) -> SearchSpace {
            SearchSpace::new()
        }

        fn clone_boxed(&self) -> Box<dyn PipelineSparseTransformer> {
            Box::new(self.clone())
        }

        fn name(&self) -> &str {
            "MockSparseTransformer"
        }
    }

    #[derive(Clone)]
    struct MockSparseModel {
        fitted: bool,
    }

    impl MockSparseModel {
        fn new() -> Self {
            Self { fitted: false }
        }
    }

    impl PipelineSparseModel for MockSparseModel {
        fn fit_sparse(&mut self, _x: &CsrMatrix, _y: &Array1<f64>) -> Result<()> {
            self.fitted = true;
            Ok(())
        }

        fn predict_sparse(&self, x: &CsrMatrix) -> Result<Array1<f64>> {
            if !self.fitted {
                return Err(FerroError::not_fitted("MockSparseModel"));
            }
            // Return sum of each row as prediction
            let (nrows, _) = x.shape();
            let mut predictions = Array1::zeros(nrows);
            for row in 0..nrows {
                let start = x.indptr()[row];
                let end = x.indptr()[row + 1];
                let sum: f64 = x.data()[start..end].iter().sum();
                predictions[row] = sum;
            }
            Ok(predictions)
        }

        fn search_space(&self) -> SearchSpace {
            SearchSpace::new().add("alpha", Parameter::float(0.01, 10.0))
        }

        fn clone_boxed(&self) -> Box<dyn PipelineSparseModel> {
            Box::new(self.clone())
        }

        fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()> {
            match name {
                "alpha" => {
                    let _ = value
                        .as_f64()
                        .ok_or_else(|| FerroError::invalid_input("alpha must be numeric"))?;
                    Ok(())
                }
                _ => Err(FerroError::invalid_input(format!(
                    "Unknown parameter '{}'",
                    name
                ))),
            }
        }

        fn name(&self) -> &str {
            "MockSparseModel"
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }
    }

    /// A mock dense model for testing DenseModel path.
    #[derive(Clone)]
    struct MockDenseModel {
        fitted: bool,
    }

    impl MockDenseModel {
        fn new() -> Self {
            Self { fitted: false }
        }
    }

    impl crate::models::Model for MockDenseModel {
        fn fit(&mut self, _x: &Array2<f64>, _y: &Array1<f64>) -> Result<()> {
            self.fitted = true;
            Ok(())
        }

        fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
            if !self.fitted {
                return Err(FerroError::not_fitted("MockDenseModel"));
            }
            // Return row sums
            Ok(x.sum_axis(ndarray::Axis(1)))
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn n_features(&self) -> Option<usize> {
            None
        }

        fn search_space(&self) -> SearchSpace {
            SearchSpace::new().add("lr", Parameter::float(0.001, 1.0))
        }
    }

    impl super::super::PipelineModel for MockDenseModel {
        fn clone_boxed(&self) -> Box<dyn super::super::PipelineModel> {
            Box::new(self.clone())
        }

        fn name(&self) -> &str {
            "MockDenseModel"
        }
    }

    // =========================================================================
    // Helper
    // =========================================================================

    fn sample_docs() -> Vec<String> {
        vec![
            "hello world".to_string(),
            "foo bar baz".to_string(),
            "hello foo".to_string(),
        ]
    }

    fn sample_labels() -> Array1<f64> {
        Array1::from_vec(vec![0.0, 1.0, 0.0])
    }

    // =========================================================================
    // Construction tests
    // =========================================================================

    #[test]
    fn test_new_pipeline_empty() {
        let pipe = TextPipeline::new();
        assert_eq!(pipe.n_steps(), 0);
        assert!(!pipe.is_fitted());
        assert!(!pipe.has_model());
    }

    #[test]
    fn test_default_pipeline() {
        let pipe = TextPipeline::default();
        assert_eq!(pipe.n_steps(), 0);
    }

    #[test]
    fn test_add_text_transformer() {
        let pipe = TextPipeline::new().add_text_transformer("cv", MockTextTransformer::new(100));
        assert_eq!(pipe.n_steps(), 1);
        assert_eq!(pipe.step_names(), vec!["cv"]);
    }

    #[test]
    fn test_add_text_and_sparse_transformer() {
        let pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(100))
            .add_sparse_transformer("tfidf", MockSparseTransformer::new());
        assert_eq!(pipe.n_steps(), 2);
        assert_eq!(pipe.step_names(), vec!["cv", "tfidf"]);
    }

    #[test]
    fn test_add_text_sparse_and_model() {
        let pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(100))
            .add_sparse_transformer("tfidf", MockSparseTransformer::new())
            .add_sparse_model("clf", MockSparseModel::new());
        assert_eq!(pipe.n_steps(), 3);
        assert!(pipe.has_model());
    }

    #[test]
    fn test_add_text_and_dense_model() {
        let pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(100))
            .add_dense_model("clf", MockDenseModel::new());
        assert_eq!(pipe.n_steps(), 2);
        assert!(pipe.has_model());
    }

    #[test]
    #[should_panic(expected = "already exists")]
    fn test_duplicate_step_name_panics() {
        let _ = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(100))
            .add_text_transformer("cv", MockTextTransformer::new(50));
    }

    #[test]
    #[should_panic(expected = "Cannot add transformer after model")]
    fn test_add_transformer_after_model_panics() {
        let _ = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(100))
            .add_sparse_model("clf", MockSparseModel::new())
            .add_sparse_transformer("tfidf", MockSparseTransformer::new());
    }

    #[test]
    #[should_panic(expected = "Only one model")]
    fn test_add_two_models_panics() {
        let _ = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(100))
            .add_sparse_model("clf1", MockSparseModel::new())
            .add_sparse_model("clf2", MockSparseModel::new());
    }

    #[test]
    #[should_panic(expected = "Add a text transformer first")]
    fn test_sparse_transformer_before_text_panics() {
        let _ = TextPipeline::new().add_sparse_transformer("tfidf", MockSparseTransformer::new());
    }

    #[test]
    #[should_panic(expected = "Add a text transformer first")]
    fn test_model_before_text_panics() {
        let _ = TextPipeline::new().add_sparse_model("clf", MockSparseModel::new());
    }

    #[test]
    #[should_panic(expected = "text transformer after sparse transformer")]
    fn test_text_after_sparse_panics() {
        let _ = TextPipeline::new()
            .add_text_transformer("cv1", MockTextTransformer::new(100))
            .add_sparse_transformer("tfidf", MockSparseTransformer::new())
            .add_text_transformer("cv2", MockTextTransformer::new(50));
    }

    // =========================================================================
    // Fit/predict tests
    // =========================================================================

    #[test]
    fn test_fit_predict_sparse_model() {
        let mut pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_sparse_model("clf", MockSparseModel::new());

        let docs = sample_docs();
        let y = sample_labels();

        pipe.fit(&docs, &y).unwrap();
        assert!(pipe.is_fitted());

        let preds = pipe.predict(&docs).unwrap();
        assert_eq!(preds.len(), docs.len());
    }

    #[test]
    fn test_fit_predict_with_sparse_transformer() {
        let mut pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_sparse_transformer("tfidf", MockSparseTransformer::new())
            .add_sparse_model("clf", MockSparseModel::new());

        let docs = sample_docs();
        let y = sample_labels();

        pipe.fit(&docs, &y).unwrap();
        let preds = pipe.predict(&docs).unwrap();
        assert_eq!(preds.len(), docs.len());

        // With sparse transformer doubling values, predictions should be 2x
        // compared to without
        let mut pipe_no_tfidf = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_sparse_model("clf", MockSparseModel::new());
        pipe_no_tfidf.fit(&docs, &y).unwrap();
        let preds_no_tfidf = pipe_no_tfidf.predict(&docs).unwrap();

        for (p, p_no) in preds.iter().zip(preds_no_tfidf.iter()) {
            assert!((p - 2.0 * p_no).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fit_predict_dense_model() {
        let mut pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_dense_model("clf", MockDenseModel::new());

        let docs = sample_docs();
        let y = sample_labels();

        pipe.fit(&docs, &y).unwrap();
        let preds = pipe.predict(&docs).unwrap();
        assert_eq!(preds.len(), docs.len());
    }

    #[test]
    fn test_fit_predict_dense_model_with_sparse_transformer() {
        let mut pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_sparse_transformer("tfidf", MockSparseTransformer::new())
            .add_dense_model("clf", MockDenseModel::new());

        let docs = sample_docs();
        let y = sample_labels();

        pipe.fit(&docs, &y).unwrap();
        let preds = pipe.predict(&docs).unwrap();
        assert_eq!(preds.len(), docs.len());
    }

    // =========================================================================
    // Transform tests
    // =========================================================================

    #[test]
    fn test_transform_returns_sparse() {
        let mut pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_sparse_model("clf", MockSparseModel::new());

        let docs = sample_docs();
        let y = sample_labels();

        pipe.fit(&docs, &y).unwrap();
        let csr = pipe.transform(&docs).unwrap();
        let (nrows, ncols) = csr.shape();
        assert_eq!(nrows, docs.len());
        assert_eq!(ncols, 5);
    }

    #[test]
    fn test_transform_without_model() {
        let mut pipe = TextPipeline::new().add_text_transformer("cv", MockTextTransformer::new(5));

        let docs = sample_docs();
        let y = sample_labels();

        pipe.fit(&docs, &y).unwrap();
        let csr = pipe.transform(&docs).unwrap();
        assert_eq!(csr.shape().0, docs.len());
    }

    #[test]
    fn test_transform_dense_returns_array2() {
        let mut pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_sparse_model("clf", MockSparseModel::new());

        let docs = sample_docs();
        let y = sample_labels();

        pipe.fit(&docs, &y).unwrap();
        let dense = pipe.transform_dense(&docs).unwrap();
        assert_eq!(dense.nrows(), docs.len());
        assert_eq!(dense.ncols(), 5);
    }

    #[test]
    fn test_transform_with_sparse_transformer() {
        let mut pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_sparse_transformer("tfidf", MockSparseTransformer::new())
            .add_sparse_model("clf", MockSparseModel::new());

        let docs = sample_docs();
        let y = sample_labels();

        pipe.fit(&docs, &y).unwrap();
        let csr = pipe.transform(&docs).unwrap();
        // Sparse transformer doubles values, so each entry should be 2.0
        for &val in csr.data() {
            assert!((val - 2.0).abs() < 1e-10);
        }
    }

    // =========================================================================
    // Error tests
    // =========================================================================

    #[test]
    fn test_predict_not_fitted_error() {
        let pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_sparse_model("clf", MockSparseModel::new());

        let docs = sample_docs();
        let err = pipe.predict(&docs).unwrap_err();
        assert!(
            err.to_string().contains("not fitted") || err.to_string().contains("Not fitted"),
            "Expected 'not fitted' error, got: {}",
            err
        );
    }

    #[test]
    fn test_transform_not_fitted_error() {
        let pipe = TextPipeline::new().add_text_transformer("cv", MockTextTransformer::new(5));

        let docs = sample_docs();
        let err = pipe.transform(&docs).unwrap_err();
        assert!(
            err.to_string().contains("not fitted") || err.to_string().contains("Not fitted"),
            "Expected 'not fitted' error, got: {}",
            err
        );
    }

    #[test]
    fn test_fit_empty_pipeline_error() {
        let mut pipe = TextPipeline::new();
        let docs = sample_docs();
        let y = sample_labels();
        let err = pipe.fit(&docs, &y).unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "Expected 'empty' error, got: {}",
            err
        );
    }

    #[test]
    fn test_fit_empty_documents_error() {
        let mut pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_sparse_model("clf", MockSparseModel::new());

        let docs: Vec<String> = vec![];
        let y = Array1::from_vec(vec![]);
        let err = pipe.fit(&docs, &y).unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "Expected 'empty' error, got: {}",
            err
        );
    }

    #[test]
    fn test_fit_shape_mismatch_error() {
        let mut pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_sparse_model("clf", MockSparseModel::new());

        let docs = sample_docs(); // 3 docs
        let y = Array1::from_vec(vec![0.0, 1.0]); // 2 labels
        let err = pipe.fit(&docs, &y).unwrap_err();
        assert!(
            err.to_string().contains("3") && err.to_string().contains("2"),
            "Expected shape mismatch error, got: {}",
            err
        );
    }

    #[test]
    fn test_predict_no_model_error() {
        let mut pipe = TextPipeline::new().add_text_transformer("cv", MockTextTransformer::new(5));

        let docs = sample_docs();
        let y = sample_labels();

        pipe.fit(&docs, &y).unwrap();
        let err = pipe.predict(&docs).unwrap_err();
        assert!(
            err.to_string().contains("no model") || err.to_string().contains("No model"),
            "Expected 'no model' error, got: {}",
            err
        );
    }

    // =========================================================================
    // Search space + set_params tests
    // =========================================================================

    #[test]
    fn test_search_space_combined() {
        let pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(100))
            .add_sparse_model("clf", MockSparseModel::new());

        let space = pipe.search_space();
        assert!(space.parameters.contains_key("cv__vocab_size"));
        assert!(space.parameters.contains_key("clf__alpha"));
        assert_eq!(space.n_dims(), 2);
    }

    #[test]
    fn test_search_space_empty_pipeline() {
        let pipe = TextPipeline::new();
        let space = pipe.search_space();
        assert_eq!(space.n_dims(), 0);
    }

    #[test]
    fn test_search_space_with_sparse_transformer() {
        let pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(100))
            .add_sparse_transformer("tfidf", MockSparseTransformer::new())
            .add_sparse_model("clf", MockSparseModel::new());

        let space = pipe.search_space();
        // MockSparseTransformer has empty search space
        assert!(space.parameters.contains_key("cv__vocab_size"));
        assert!(space.parameters.contains_key("clf__alpha"));
        assert_eq!(space.n_dims(), 2);
    }

    #[test]
    fn test_set_params() {
        let mut pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(100))
            .add_sparse_model("clf", MockSparseModel::new());

        // Fit first
        let docs = sample_docs();
        let y = sample_labels();
        pipe.fit(&docs, &y).unwrap();
        assert!(pipe.is_fitted());

        // Set params should mark as unfitted
        let mut params = HashMap::new();
        params.insert("cv__vocab_size".to_string(), ParameterValue::Int(50));
        params.insert("clf__alpha".to_string(), ParameterValue::Float(0.5));
        pipe.set_params(&params).unwrap();
        assert!(!pipe.is_fitted());
    }

    #[test]
    fn test_set_params_invalid_format() {
        let mut pipe =
            TextPipeline::new().add_text_transformer("cv", MockTextTransformer::new(100));

        let mut params = HashMap::new();
        params.insert("nodelimiter".to_string(), ParameterValue::Int(5));
        let err = pipe.set_params(&params).unwrap_err();
        assert!(
            err.to_string().contains("step__param"),
            "Expected format error, got: {}",
            err
        );
    }

    #[test]
    fn test_set_params_unknown_step() {
        let mut pipe =
            TextPipeline::new().add_text_transformer("cv", MockTextTransformer::new(100));

        let mut params = HashMap::new();
        params.insert("nonexistent__param".to_string(), ParameterValue::Int(5));
        let err = pipe.set_params(&params).unwrap_err();
        assert!(
            err.to_string().contains("not found"),
            "Expected 'not found' error, got: {}",
            err
        );
    }

    #[test]
    fn test_set_params_unknown_param() {
        let mut pipe =
            TextPipeline::new().add_text_transformer("cv", MockTextTransformer::new(100));

        let mut params = HashMap::new();
        params.insert("cv__nonexistent".to_string(), ParameterValue::Int(5));
        let err = pipe.set_params(&params).unwrap_err();
        assert!(
            err.to_string().contains("Unknown"),
            "Expected 'Unknown' error, got: {}",
            err
        );
    }

    // =========================================================================
    // Step introspection tests
    // =========================================================================

    #[test]
    fn test_step_names() {
        let pipe = TextPipeline::new()
            .add_text_transformer("vectorizer", MockTextTransformer::new(100))
            .add_sparse_transformer("normalizer", MockSparseTransformer::new())
            .add_sparse_model("classifier", MockSparseModel::new());

        assert_eq!(
            pipe.step_names(),
            vec!["vectorizer", "normalizer", "classifier"]
        );
    }

    #[test]
    fn test_n_steps() {
        let pipe = TextPipeline::new().add_text_transformer("cv", MockTextTransformer::new(100));
        assert_eq!(pipe.n_steps(), 1);

        let pipe = pipe.add_sparse_model("clf", MockSparseModel::new());
        assert_eq!(pipe.n_steps(), 2);
    }

    #[test]
    fn test_is_fitted_transitions() {
        let mut pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_sparse_model("clf", MockSparseModel::new());

        assert!(!pipe.is_fitted());

        pipe.fit(&sample_docs(), &sample_labels()).unwrap();
        assert!(pipe.is_fitted());

        // set_params resets fitted
        let mut params = HashMap::new();
        params.insert("clf__alpha".to_string(), ParameterValue::Float(0.1));
        pipe.set_params(&params).unwrap();
        assert!(!pipe.is_fitted());
    }

    #[test]
    fn test_has_model() {
        let pipe = TextPipeline::new().add_text_transformer("cv", MockTextTransformer::new(100));
        assert!(!pipe.has_model());

        let pipe = pipe.add_sparse_model("clf", MockSparseModel::new());
        assert!(pipe.has_model());
    }

    #[test]
    fn test_debug_format() {
        let pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_sparse_model("clf", MockSparseModel::new());

        let debug_str = format!("{:?}", pipe);
        assert!(debug_str.contains("TextPipeline"));
        assert!(debug_str.contains("cv"));
        assert!(debug_str.contains("clf"));
    }

    #[test]
    fn test_dense_model_search_space_in_pipeline() {
        let pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(100))
            .add_dense_model("clf", MockDenseModel::new());

        let space = pipe.search_space();
        assert!(space.parameters.contains_key("cv__vocab_size"));
        assert!(space.parameters.contains_key("clf__lr"));
        assert_eq!(space.n_dims(), 2);
    }

    #[test]
    fn test_fit_transformer_only_pipeline() {
        let mut pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_sparse_transformer("tfidf", MockSparseTransformer::new());

        let docs = sample_docs();
        let y = sample_labels();

        // Should succeed (no model is fine for transform-only pipelines)
        pipe.fit(&docs, &y).unwrap();
        assert!(pipe.is_fitted());
    }

    #[test]
    fn test_refit_pipeline() {
        let mut pipe = TextPipeline::new()
            .add_text_transformer("cv", MockTextTransformer::new(5))
            .add_sparse_model("clf", MockSparseModel::new());

        let docs = sample_docs();
        let y = sample_labels();

        pipe.fit(&docs, &y).unwrap();
        let preds1 = pipe.predict(&docs).unwrap();

        // Fit again
        pipe.fit(&docs, &y).unwrap();
        let preds2 = pipe.predict(&docs).unwrap();

        // Predictions should be the same
        for (a, b) in preds1.iter().zip(preds2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}

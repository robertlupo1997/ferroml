//! TF-IDF Vectorizer: CountVectorizer + TfidfTransformer combined.
//!
//! Converts raw text documents directly to TF-IDF weighted sparse matrices.
//! This is equivalent to using CountVectorizer followed by TfidfTransformer,
//! but in a single, convenient step.
//!
//! # Example
//!
//! ```
//! use ferroml_core::preprocessing::tfidf_vectorizer::TfidfVectorizer;
//! use ferroml_core::preprocessing::count_vectorizer::TextTransformer;
//!
//! let corpus = vec![
//!     "the cat sat on the mat".to_string(),
//!     "the dog sat on the log".to_string(),
//! ];
//!
//! let mut tv = TfidfVectorizer::new();
//! let tfidf_matrix = tv.fit_transform_text(&corpus).unwrap();
//! // tfidf_matrix is a sparse CsrMatrix of TF-IDF values
//! ```

use std::collections::HashMap;

use ndarray::{Array1, Array2};

use crate::preprocessing::count_vectorizer::{CountVectorizer, DocFrequency, TextTransformer};
use crate::preprocessing::tfidf::{TfidfNorm, TfidfTransformer};
use crate::sparse::CsrMatrix;
use crate::Result;

/// TF-IDF Vectorizer: wraps CountVectorizer + TfidfTransformer into a single step.
///
/// Converts raw text documents directly to TF-IDF weighted sparse matrices.
/// This is the most common way to convert text to numeric features for ML models.
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::tfidf_vectorizer::TfidfVectorizer;
/// use ferroml_core::preprocessing::count_vectorizer::TextTransformer;
///
/// let corpus = vec![
///     "the cat sat on the mat".to_string(),
///     "the dog sat on the log".to_string(),
/// ];
///
/// let mut tv = TfidfVectorizer::new();
/// let tfidf_matrix = tv.fit_transform_text(&corpus).unwrap();
/// assert_eq!(tfidf_matrix.nrows(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct TfidfVectorizer {
    count_vectorizer: CountVectorizer,
    tfidf_transformer: TfidfTransformer,
}

impl Default for TfidfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TfidfVectorizer {
    /// Create a new `TfidfVectorizer` with default settings.
    ///
    /// Defaults: no max_features, min_df=1, max_df=1.0, ngram_range=(1,1),
    /// binary=false, lowercase=true, L2 norm, use_idf=true, smooth_idf=true,
    /// sublinear_tf=false.
    pub fn new() -> Self {
        Self {
            count_vectorizer: CountVectorizer::new(),
            tfidf_transformer: TfidfTransformer::new(),
        }
    }

    // ========================================================================
    // Builder methods forwarding to CountVectorizer
    // ========================================================================

    /// Set the maximum number of features (vocabulary size).
    pub fn with_max_features(mut self, n: usize) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_max_features(n);
        self
    }

    /// Set the n-gram range (min_n, max_n).
    pub fn with_ngram_range(mut self, range: (usize, usize)) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_ngram_range(range);
        self
    }

    /// Set the minimum document frequency threshold.
    pub fn with_min_df(mut self, min_df: DocFrequency) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_min_df(min_df);
        self
    }

    /// Set the maximum document frequency threshold.
    pub fn with_max_df(mut self, max_df: DocFrequency) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_max_df(max_df);
        self
    }

    /// Set binary mode. If true, all non-zero counts are clamped to 1 before TF-IDF.
    pub fn with_binary(mut self, binary: bool) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_binary(binary);
        self
    }

    /// Set stop words to filter out during tokenization.
    pub fn with_stop_words(mut self, sw: Vec<String>) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_stop_words(sw);
        self
    }

    /// Set whether to convert all text to lowercase before tokenization.
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.count_vectorizer = self.count_vectorizer.with_lowercase(lowercase);
        self
    }

    // ========================================================================
    // Builder methods forwarding to TfidfTransformer
    // ========================================================================

    /// Set the normalization type (L1, L2, or None).
    pub fn with_norm(mut self, norm: TfidfNorm) -> Self {
        self.tfidf_transformer = self.tfidf_transformer.with_norm(norm);
        self
    }

    /// Set whether to use IDF weighting.
    pub fn with_use_idf(mut self, use_idf: bool) -> Self {
        self.tfidf_transformer = self.tfidf_transformer.with_use_idf(use_idf);
        self
    }

    /// Set whether to smooth IDF weights.
    pub fn with_smooth_idf(mut self, smooth: bool) -> Self {
        self.tfidf_transformer = self.tfidf_transformer.with_smooth_idf(smooth);
        self
    }

    /// Set whether to apply sublinear TF scaling (1 + log(tf)).
    pub fn with_sublinear_tf(mut self, sub: bool) -> Self {
        self.tfidf_transformer = self.tfidf_transformer.with_sublinear_tf(sub);
        self
    }

    // ========================================================================
    // Getters forwarding to inner components
    // ========================================================================

    /// Get the learned vocabulary mapping (term -> index).
    /// Returns `None` if not fitted.
    pub fn vocabulary(&self) -> Option<&HashMap<String, usize>> {
        self.count_vectorizer.vocabulary()
    }

    /// Get the sorted feature names (vocabulary terms).
    /// Returns `None` if not fitted.
    pub fn get_feature_names(&self) -> Option<&[String]> {
        self.count_vectorizer.get_feature_names()
    }

    /// Get the learned IDF weights. Returns `None` if not fitted or use_idf=false.
    pub fn idf(&self) -> Option<&Array1<f64>> {
        self.tfidf_transformer.idf()
    }

    /// Check if the vectorizer has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.count_vectorizer.is_fitted() && self.tfidf_transformer.is_fitted()
    }

    // ========================================================================
    // Dense convenience methods
    // ========================================================================

    /// Fit and transform documents to a dense TF-IDF matrix in one step.
    pub fn fit_transform_text_dense(&mut self, documents: &[String]) -> Result<Array2<f64>> {
        self.fit_text(documents)?;
        self.transform_text_dense(documents)
    }

    /// Transform documents to a dense TF-IDF matrix.
    pub fn transform_text_dense(&self, documents: &[String]) -> Result<Array2<f64>> {
        let sparse = self.transform_text(documents)?;
        Ok(sparse.to_dense())
    }
}

impl TextTransformer for TfidfVectorizer {
    fn fit_text(&mut self, documents: &[String]) -> Result<()> {
        self.count_vectorizer.fit_text(documents)?;
        let counts = self.count_vectorizer.transform_text(documents)?;
        self.tfidf_transformer.fit_sparse(&counts)?;
        Ok(())
    }

    fn transform_text(&self, documents: &[String]) -> Result<CsrMatrix> {
        let counts = self.count_vectorizer.transform_text(documents)?;
        self.tfidf_transformer.transform_sparse_native(&counts)
    }
}

#[cfg(feature = "sparse")]
impl crate::pipeline::PipelineTextTransformer for TfidfVectorizer {
    fn clone_boxed(&self) -> Box<dyn crate::pipeline::PipelineTextTransformer> {
        Box::new(self.clone())
    }

    fn set_param(&mut self, name: &str, value: &crate::hpo::ParameterValue) -> crate::Result<()> {
        match name {
            // CountVectorizer params
            "max_features" | "binary" | "lowercase" => {
                crate::pipeline::PipelineTextTransformer::set_param(
                    &mut self.count_vectorizer,
                    name,
                    value,
                )
            }
            // TfidfTransformer params
            "norm" | "use_idf" | "smooth_idf" | "sublinear_tf" => {
                crate::pipeline::PipelineTransformer::set_param(
                    &mut self.tfidf_transformer,
                    name,
                    value,
                )
            }
            _ => Err(crate::FerroError::invalid_input(format!(
                "Unknown parameter '{}'",
                name
            ))),
        }
    }

    fn name(&self) -> &str {
        "TfidfVectorizer"
    }

    fn n_features_out(&self) -> Option<usize> {
        self.count_vectorizer.vocabulary().map(|v| v.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_corpus() -> Vec<String> {
        vec![
            "the cat sat on the mat".to_string(),
            "the dog sat on the log".to_string(),
            "the cat and the dog".to_string(),
        ]
    }

    #[test]
    fn test_basic_fit_transform_text() {
        let corpus = sample_corpus();
        let mut tv = TfidfVectorizer::new();
        let result = tv.fit_transform_text(&corpus).unwrap();

        assert_eq!(result.nrows(), 3);
        assert!(result.ncols() > 0);
        assert!(result.nnz() > 0);

        // Dense values should be non-negative
        let dense = result.to_dense();
        for val in dense.iter() {
            assert!(*val >= 0.0);
        }
    }

    #[test]
    fn test_matches_manual_chain() {
        let corpus = sample_corpus();

        // Manual chain: CountVectorizer -> TfidfTransformer
        let mut cv = CountVectorizer::new();
        cv.fit_text(&corpus).unwrap();
        let counts = cv.transform_text(&corpus).unwrap();
        let mut tfidf = TfidfTransformer::new();
        tfidf.fit_sparse(&counts).unwrap();
        let manual_result = tfidf.transform_sparse_native(&counts).unwrap();

        // TfidfVectorizer combined
        let mut tv = TfidfVectorizer::new();
        let combined_result = tv.fit_transform_text(&corpus).unwrap();

        let manual_dense = manual_result.to_dense();
        let combined_dense = combined_result.to_dense();

        assert_eq!(manual_dense.dim(), combined_dense.dim());
        for i in 0..manual_dense.nrows() {
            for j in 0..manual_dense.ncols() {
                assert!(
                    (manual_dense[[i, j]] - combined_dense[[i, j]]).abs() < 1e-10,
                    "Mismatch at ({}, {}): manual={}, combined={}",
                    i,
                    j,
                    manual_dense[[i, j]],
                    combined_dense[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_vocabulary_accessible() {
        let corpus = sample_corpus();
        let mut tv = TfidfVectorizer::new();

        assert!(tv.vocabulary().is_none());
        tv.fit_text(&corpus).unwrap();

        let vocab = tv.vocabulary().unwrap();
        assert!(!vocab.is_empty());
        assert!(vocab.contains_key("cat"));
        assert!(vocab.contains_key("dog"));
        assert!(vocab.contains_key("the"));
    }

    #[test]
    fn test_feature_names_accessible() {
        let corpus = vec!["zebra apple mango".to_string()];
        let mut tv = TfidfVectorizer::new();

        assert!(tv.get_feature_names().is_none());
        tv.fit_text(&corpus).unwrap();

        let names = tv.get_feature_names().unwrap();
        // Feature names should be sorted alphabetically
        assert_eq!(names, &["apple", "mango", "zebra"]);
    }

    #[test]
    fn test_idf_accessible() {
        let corpus = sample_corpus();
        let mut tv = TfidfVectorizer::new();

        assert!(tv.idf().is_none());
        tv.fit_text(&corpus).unwrap();

        let idf = tv.idf().unwrap();
        let vocab = tv.vocabulary().unwrap();
        assert_eq!(idf.len(), vocab.len());

        // All IDF values should be positive
        for &val in idf.iter() {
            assert!(val > 0.0);
        }

        // "the" appears in all 3 docs -> lowest IDF
        // Terms appearing in fewer docs should have higher IDF
        let the_idx = vocab["the"];
        let cat_idx = vocab["cat"];
        // "cat" appears in 2/3 docs, "the" in 3/3 -> cat IDF > the IDF
        assert!(idf[cat_idx] > idf[the_idx]);
    }

    #[test]
    fn test_max_features() {
        let corpus = vec![
            "alpha bravo charlie".to_string(),
            "alpha bravo delta".to_string(),
            "alpha echo foxtrot".to_string(),
        ];
        let mut tv = TfidfVectorizer::new().with_max_features(2);
        let result = tv.fit_transform_text(&corpus).unwrap();

        let vocab = tv.vocabulary().unwrap();
        assert_eq!(vocab.len(), 2);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn test_ngram_range() {
        let corpus = vec!["the cat sat".to_string(), "the dog sat".to_string()];
        let mut tv = TfidfVectorizer::new().with_ngram_range((1, 2));
        tv.fit_text(&corpus).unwrap();

        let vocab = tv.vocabulary().unwrap();
        // Should have unigrams and bigrams
        assert!(vocab.contains_key("the"));
        assert!(vocab.contains_key("cat"));
        assert!(vocab.contains_key("the cat"));
        assert!(vocab.contains_key("cat sat"));
    }

    #[test]
    fn test_min_df() {
        let corpus = vec![
            "apple banana".to_string(),
            "apple cherry".to_string(),
            "apple date".to_string(),
        ];
        let mut tv = TfidfVectorizer::new().with_min_df(DocFrequency::Count(2));
        tv.fit_text(&corpus).unwrap();

        let vocab = tv.vocabulary().unwrap();
        // "apple" appears in 3 docs (>= 2): included
        assert!(vocab.contains_key("apple"));
        // Others appear in 1 doc: excluded
        assert!(!vocab.contains_key("banana"));
        assert!(!vocab.contains_key("cherry"));
        assert!(!vocab.contains_key("date"));
    }

    #[test]
    fn test_max_df() {
        let corpus = vec![
            "common rare1".to_string(),
            "common rare2".to_string(),
            "common rare3".to_string(),
            "common rare4".to_string(),
            "common rare5".to_string(),
        ];
        let mut tv = TfidfVectorizer::new().with_max_df(DocFrequency::Fraction(0.8));
        tv.fit_text(&corpus).unwrap();

        let vocab = tv.vocabulary().unwrap();
        // "common" appears in 5/5 = 100% (> 80%): excluded
        assert!(!vocab.contains_key("common"));
        // Each "rareX" appears in 1/5 = 20%: included
        assert!(vocab.contains_key("rare1"));
    }

    #[test]
    fn test_stop_words() {
        let corpus = vec!["the cat sat on the mat".to_string()];
        let mut tv =
            TfidfVectorizer::new().with_stop_words(vec!["the".to_string(), "on".to_string()]);
        tv.fit_text(&corpus).unwrap();

        let vocab = tv.vocabulary().unwrap();
        assert!(!vocab.contains_key("the"));
        assert!(!vocab.contains_key("on"));
        assert!(vocab.contains_key("cat"));
        assert!(vocab.contains_key("sat"));
        assert!(vocab.contains_key("mat"));
    }

    #[test]
    fn test_binary_mode() {
        let corpus = vec![
            "hello hello hello world".to_string(),
            "world world goodbye".to_string(),
        ];

        // With binary=true, counts are clamped to 0/1 before TF-IDF
        let mut tv_binary = TfidfVectorizer::new()
            .with_binary(true)
            .with_use_idf(false)
            .with_norm(TfidfNorm::None);
        let result_binary = tv_binary.fit_transform_text_dense(&corpus).unwrap();

        // With binary=false (default), raw counts are used
        let mut tv_normal = TfidfVectorizer::new()
            .with_binary(false)
            .with_use_idf(false)
            .with_norm(TfidfNorm::None);
        let result_normal = tv_normal.fit_transform_text_dense(&corpus).unwrap();

        // In binary mode, "hello" count in doc 0 should be 1.0 (not 3.0)
        let vocab = tv_binary.vocabulary().unwrap();
        let hello_idx = vocab["hello"];
        assert!((result_binary[[0, hello_idx]] - 1.0).abs() < 1e-10);
        assert!((result_normal[[0, hello_idx]] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_l1() {
        let corpus = sample_corpus();
        let mut tv = TfidfVectorizer::new().with_norm(TfidfNorm::L1);
        let result = tv.fit_transform_text_dense(&corpus).unwrap();

        // Each row should have L1 norm = 1
        for i in 0..result.nrows() {
            let norm: f64 = result.row(i).iter().map(|v| v.abs()).sum();
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

    #[test]
    fn test_norm_none() {
        let corpus = sample_corpus();
        let mut tv = TfidfVectorizer::new().with_norm(TfidfNorm::None);
        let result = tv.fit_transform_text_dense(&corpus).unwrap();

        // Without normalization, rows should NOT generally sum to 1
        // Just check that values are non-negative and some are > 1
        let mut has_value_gt_1 = false;
        for val in result.iter() {
            assert!(*val >= 0.0);
            if *val > 1.0 {
                has_value_gt_1 = true;
            }
        }
        assert!(
            has_value_gt_1,
            "Without normalization, some values should exceed 1.0"
        );
    }

    #[test]
    fn test_no_idf() {
        let corpus = sample_corpus();
        let mut tv = TfidfVectorizer::new()
            .with_use_idf(false)
            .with_norm(TfidfNorm::None);
        let result = tv.fit_transform_text_dense(&corpus).unwrap();

        // Without IDF and normalization, result should be raw term counts
        let mut cv = CountVectorizer::new();
        let counts = cv.fit_transform_text_dense(&corpus).unwrap();

        assert_eq!(result.dim(), counts.dim());
        for i in 0..result.nrows() {
            for j in 0..result.ncols() {
                assert!(
                    (result[[i, j]] - counts[[i, j]]).abs() < 1e-10,
                    "Mismatch at ({}, {}): tfidf={}, counts={}",
                    i,
                    j,
                    result[[i, j]],
                    counts[[i, j]]
                );
            }
        }

        // idf() should return None
        assert!(tv.idf().is_none());
    }

    #[test]
    fn test_sublinear_tf() {
        let corpus = vec![
            "hello hello hello hello world".to_string(),
            "world goodbye".to_string(),
        ];

        let mut tv_normal = TfidfVectorizer::new()
            .with_sublinear_tf(false)
            .with_norm(TfidfNorm::None);
        let result_normal = tv_normal.fit_transform_text_dense(&corpus).unwrap();

        let mut tv_sub = TfidfVectorizer::new()
            .with_sublinear_tf(true)
            .with_norm(TfidfNorm::None);
        let result_sub = tv_sub.fit_transform_text_dense(&corpus).unwrap();

        // With sublinear TF, high counts are compressed
        let vocab = tv_normal.vocabulary().unwrap();
        let hello_idx = vocab["hello"];

        // "hello" has tf=4 in doc 0
        // Normal: 4 * idf, Sublinear: (1 + ln(4)) * idf
        // The sublinear value should be smaller
        assert!(
            result_sub[[0, hello_idx]] < result_normal[[0, hello_idx]],
            "Sublinear TF should compress high counts"
        );
    }

    #[test]
    fn test_smooth_idf() {
        let corpus = sample_corpus();

        let mut tv_smooth = TfidfVectorizer::new()
            .with_smooth_idf(true)
            .with_norm(TfidfNorm::None);
        tv_smooth.fit_text(&corpus).unwrap();

        let mut tv_unsmooth = TfidfVectorizer::new()
            .with_smooth_idf(false)
            .with_norm(TfidfNorm::None);
        tv_unsmooth.fit_text(&corpus).unwrap();

        let idf_smooth = tv_smooth.idf().unwrap();
        let idf_unsmooth = tv_unsmooth.idf().unwrap();

        // Smooth and unsmoothed IDF should differ
        let mut differs = false;
        for i in 0..idf_smooth.len() {
            if (idf_smooth[i] - idf_unsmooth[i]).abs() > 1e-10 {
                differs = true;
                break;
            }
        }
        assert!(
            differs,
            "Smooth and unsmoothed IDF should produce different values"
        );
    }

    #[test]
    fn test_empty_corpus_error() {
        let corpus: Vec<String> = vec![];
        let mut tv = TfidfVectorizer::new();
        assert!(tv.fit_text(&corpus).is_err());
    }

    #[test]
    fn test_not_fitted_error() {
        let tv = TfidfVectorizer::new();
        let docs = vec!["hello world".to_string()];
        assert!(tv.transform_text(&docs).is_err());
        assert!(!tv.is_fitted());
    }
}

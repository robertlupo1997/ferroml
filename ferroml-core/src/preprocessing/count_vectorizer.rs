//! Count Vectorizer for text feature extraction.
//!
//! Converts a collection of text documents to a matrix of token counts.
//! This is the text equivalent of `OneHotEncoder` for categorical features.
//!
//! # Example
//!
//! ```
//! use ferroml_core::preprocessing::count_vectorizer::{CountVectorizer, TextTransformer};
//!
//! let corpus = vec![
//!     "the cat sat on the mat".to_string(),
//!     "the dog sat on the log".to_string(),
//! ];
//!
//! let mut cv = CountVectorizer::new();
//! let counts = cv.fit_transform_text_dense(&corpus).unwrap();
//! // counts is a (2, n_features) matrix of token counts
//! ```

use std::collections::{HashMap, HashSet};

use ndarray::Array2;

use crate::sparse::CsrMatrix;
use crate::{FerroError, Result};

/// How to interpret min_df / max_df thresholds.
#[derive(Debug, Clone)]
pub enum DocFrequency {
    /// Absolute count threshold.
    Count(usize),
    /// Fraction of total documents [0.0, 1.0].
    Fraction(f64),
}

impl DocFrequency {
    /// Resolve the threshold to an absolute count given the total number of documents.
    fn resolve(&self, n_docs: usize) -> usize {
        match self {
            DocFrequency::Count(c) => *c,
            DocFrequency::Fraction(f) => {
                // Ceiling for min_df fraction, floor for max_df fraction
                // We return the raw value; callers decide how to use it.
                (*f * n_docs as f64) as usize
            }
        }
    }
}

/// Token extraction pattern.
#[derive(Debug, Clone)]
pub enum TokenPattern {
    /// Default: match alphanumeric sequences of length >= 2.
    Word,
}

/// Trait for text-based transformers that operate on string documents
/// rather than numeric arrays.
pub trait TextTransformer: Send + Sync {
    /// Fit the transformer on a collection of text documents.
    fn fit_text(&mut self, documents: &[String]) -> Result<()>;

    /// Transform documents into a sparse term-count matrix.
    fn transform_text(&self, documents: &[String]) -> Result<CsrMatrix>;

    /// Fit and transform in one step.
    fn fit_transform_text(&mut self, documents: &[String]) -> Result<CsrMatrix> {
        self.fit_text(documents)?;
        self.transform_text(documents)
    }
}

/// Text feature extraction via bag-of-words with count or binary encoding.
///
/// Tokenizes documents, builds vocabulary from training data, and transforms
/// documents into term-count matrices (dense `Array2<f64>` or sparse `CsrMatrix`).
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::count_vectorizer::{CountVectorizer, TextTransformer};
///
/// let corpus = vec![
///     "the cat sat on the mat".to_string(),
///     "the dog sat on the log".to_string(),
/// ];
///
/// let mut cv = CountVectorizer::new();
/// let counts = cv.fit_transform_text_dense(&corpus).unwrap();
/// assert_eq!(counts.nrows(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct CountVectorizer {
    max_features: Option<usize>,
    min_df: DocFrequency,
    max_df: DocFrequency,
    ngram_range: (usize, usize),
    binary: bool,
    lowercase: bool,
    token_pattern: TokenPattern,
    stop_words: Option<HashSet<String>>,
    // Fitted state
    vocabulary_: Option<HashMap<String, usize>>,
    feature_names_: Option<Vec<String>>,
}

impl Default for CountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

impl CountVectorizer {
    /// Create a new `CountVectorizer` with default settings.
    ///
    /// Defaults: no max_features, min_df=1 (count), max_df=1.0 (fraction),
    /// ngram_range=(1,1), binary=false, lowercase=true, Word token pattern.
    pub fn new() -> Self {
        Self {
            max_features: None,
            min_df: DocFrequency::Count(1),
            max_df: DocFrequency::Fraction(1.0),
            ngram_range: (1, 1),
            binary: false,
            lowercase: true,
            token_pattern: TokenPattern::Word,
            stop_words: None,
            vocabulary_: None,
            feature_names_: None,
        }
    }

    /// Set the maximum number of features (vocabulary size).
    /// The top `max_features` by document frequency are kept.
    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = Some(max_features);
        self
    }

    /// Set the minimum document frequency threshold.
    pub fn with_min_df(mut self, min_df: DocFrequency) -> Self {
        self.min_df = min_df;
        self
    }

    /// Set the maximum document frequency threshold.
    pub fn with_max_df(mut self, max_df: DocFrequency) -> Self {
        self.max_df = max_df;
        self
    }

    /// Set the n-gram range (min_n, max_n). For unigrams, use (1, 1).
    /// For unigrams + bigrams, use (1, 2).
    pub fn with_ngram_range(mut self, ngram_range: (usize, usize)) -> Self {
        self.ngram_range = ngram_range;
        self
    }

    /// Set binary mode. If true, all non-zero counts are set to 1.
    pub fn with_binary(mut self, binary: bool) -> Self {
        self.binary = binary;
        self
    }

    /// Set whether to convert all text to lowercase before tokenization.
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Set the token extraction pattern.
    pub fn with_token_pattern(mut self, token_pattern: TokenPattern) -> Self {
        self.token_pattern = token_pattern;
        self
    }

    /// Set stop words to filter out during tokenization.
    pub fn with_stop_words(mut self, stop_words: Vec<String>) -> Self {
        self.stop_words = Some(stop_words.into_iter().collect());
        self
    }

    /// Get the learned vocabulary mapping (term -> index).
    /// Returns `None` if not fitted.
    pub fn vocabulary(&self) -> Option<&HashMap<String, usize>> {
        self.vocabulary_.as_ref()
    }

    /// Get the sorted feature names (vocabulary terms).
    /// Returns `None` if not fitted.
    pub fn get_feature_names(&self) -> Option<&[String]> {
        self.feature_names_.as_deref()
    }

    /// Check if the vectorizer has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.vocabulary_.is_some()
    }

    /// Transform documents to a dense count matrix.
    /// Calls `transform_text` and then converts the sparse result to dense.
    pub fn transform_text_dense(&self, documents: &[String]) -> Result<Array2<f64>> {
        let sparse = self.transform_text(documents)?;
        Ok(sparse.to_dense())
    }

    /// Fit and transform to a dense count matrix in one step.
    pub fn fit_transform_text_dense(&mut self, documents: &[String]) -> Result<Array2<f64>> {
        self.fit_text(documents)?;
        self.transform_text_dense(documents)
    }

    /// Tokenize a single document into unigram tokens.
    fn tokenize(&self, doc: &str) -> Vec<String> {
        let text = if self.lowercase {
            doc.to_lowercase()
        } else {
            doc.to_string()
        };

        let tokens: Vec<String> = match &self.token_pattern {
            TokenPattern::Word => {
                // Split on non-alphanumeric, keep tokens of length >= 2
                let mut result = Vec::new();
                let mut current = String::new();

                for ch in text.chars() {
                    if ch.is_alphanumeric() {
                        current.push(ch);
                    } else {
                        if current.len() >= 2 {
                            result.push(std::mem::take(&mut current));
                        } else {
                            current.clear();
                        }
                    }
                }
                // Don't forget trailing token
                if current.len() >= 2 {
                    result.push(current);
                }

                result
            }
        };

        // Filter stop words
        if let Some(ref stop_words) = self.stop_words {
            tokens
                .into_iter()
                .filter(|t| !stop_words.contains(t))
                .collect()
        } else {
            tokens
        }
    }

    /// Generate n-grams from a list of tokens for the configured ngram_range.
    fn generate_ngrams(&self, tokens: &[String]) -> Vec<String> {
        let (min_n, max_n) = self.ngram_range;
        let mut ngrams = Vec::new();

        for n in min_n..=max_n {
            if n == 0 || tokens.len() < n {
                continue;
            }
            for window in tokens.windows(n) {
                ngrams.push(window.join(" "));
            }
        }

        ngrams
    }
}

impl TextTransformer for CountVectorizer {
    fn fit_text(&mut self, documents: &[String]) -> Result<()> {
        if documents.is_empty() {
            return Err(FerroError::invalid_input(
                "Cannot fit CountVectorizer on empty document list",
            ));
        }

        let n_docs = documents.len();

        // Step 1: Tokenize all documents and count document frequencies
        let mut doc_freq: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            let tokens = self.tokenize(doc);
            let ngrams = self.generate_ngrams(&tokens);

            // Count unique terms per document (for document frequency)
            let unique_terms: HashSet<String> = ngrams.into_iter().collect();
            for term in unique_terms {
                *doc_freq.entry(term).or_insert(0) += 1;
            }
        }

        // Step 2: Apply min_df / max_df filtering
        let min_df_abs = self.min_df.resolve(n_docs);
        let max_df_abs = self.max_df.resolve(n_docs);

        let mut filtered: Vec<(String, usize)> = doc_freq
            .into_iter()
            .filter(|(_, df)| *df >= min_df_abs && *df <= max_df_abs)
            .collect();

        if filtered.is_empty() {
            return Err(FerroError::invalid_input(
                "No features remain after min_df/max_df filtering. Try lower min_df or higher max_df.",
            ));
        }

        // Step 3: Apply max_features cap (keep top by document frequency)
        if let Some(max_features) = self.max_features {
            if filtered.len() > max_features {
                // Sort by document frequency descending, then alphabetically for ties
                filtered.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                filtered.truncate(max_features);
            }
        }

        // Step 4: Build vocabulary (sorted alphabetically for determinism)
        let mut terms: Vec<String> = filtered.into_iter().map(|(term, _)| term).collect();
        terms.sort();

        let vocabulary: HashMap<String, usize> = terms
            .iter()
            .enumerate()
            .map(|(idx, term)| (term.clone(), idx))
            .collect();

        self.feature_names_ = Some(terms);
        self.vocabulary_ = Some(vocabulary);

        Ok(())
    }

    fn transform_text(&self, documents: &[String]) -> Result<CsrMatrix> {
        let vocabulary = self
            .vocabulary_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("CountVectorizer.transform_text"))?;

        let n_features = vocabulary.len();
        let n_docs = documents.len();

        // Build sparse matrix via triplets
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();

        for (doc_idx, doc) in documents.iter().enumerate() {
            let tokens = self.tokenize(doc);
            let ngrams = self.generate_ngrams(&tokens);

            // Count term occurrences in this document
            let mut term_counts: HashMap<usize, f64> = HashMap::new();
            for ngram in &ngrams {
                if let Some(&col_idx) = vocabulary.get(ngram) {
                    *term_counts.entry(col_idx).or_insert(0.0) += 1.0;
                }
            }

            // Add to triplets
            for (col_idx, count) in term_counts {
                rows.push(doc_idx);
                cols.push(col_idx);
                values.push(if self.binary { 1.0 } else { count });
            }
        }

        CsrMatrix::from_triplets((n_docs, n_features), &rows, &cols, &values)
    }
}

#[cfg(feature = "sparse")]
impl crate::pipeline::PipelineTextTransformer for CountVectorizer {
    fn clone_boxed(&self) -> Box<dyn crate::pipeline::PipelineTextTransformer> {
        Box::new(self.clone())
    }

    fn set_param(&mut self, name: &str, value: &crate::hpo::ParameterValue) -> crate::Result<()> {
        match name {
            "max_features" => {
                if let Some(v) = value.as_i64() {
                    self.max_features = Some(v as usize);
                    Ok(())
                } else {
                    Err(crate::FerroError::invalid_input(
                        "max_features must be an integer",
                    ))
                }
            }
            "binary" => {
                if let Some(v) = value.as_bool() {
                    self.binary = v;
                    Ok(())
                } else {
                    Err(crate::FerroError::invalid_input("binary must be a boolean"))
                }
            }
            "lowercase" => {
                if let Some(v) = value.as_bool() {
                    self.lowercase = v;
                    Ok(())
                } else {
                    Err(crate::FerroError::invalid_input(
                        "lowercase must be a boolean",
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
        "CountVectorizer"
    }

    fn n_features_out(&self) -> Option<usize> {
        self.vocabulary_.as_ref().map(|v| v.len())
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
    fn test_basic_tokenization() {
        let cv = CountVectorizer::new();
        let tokens = cv.tokenize("the cat sat on the mat");
        assert_eq!(tokens, vec!["the", "cat", "sat", "on", "the", "mat"]);
    }

    #[test]
    fn test_lowercase_tokenization() {
        let cv = CountVectorizer::new();
        let tokens = cv.tokenize("The CAT Sat");
        assert_eq!(tokens, vec!["the", "cat", "sat"]);
    }

    #[test]
    fn test_case_sensitive_tokenization() {
        let cv = CountVectorizer::new().with_lowercase(false);
        let tokens = cv.tokenize("The CAT Sat");
        assert_eq!(tokens, vec!["The", "CAT", "Sat"]);
    }

    #[test]
    fn test_short_tokens_filtered() {
        let cv = CountVectorizer::new();
        let tokens = cv.tokenize("I am a cat");
        // "I", "a" are length 1 and should be filtered
        assert_eq!(tokens, vec!["am", "cat"]);
    }

    #[test]
    fn test_basic_fit_transform() {
        let corpus = sample_corpus();
        let mut cv = CountVectorizer::new();
        let result = cv.fit_transform_text_dense(&corpus).unwrap();

        assert_eq!(result.nrows(), 3);
        assert!(result.ncols() > 0);

        // "the" appears in all 3 docs
        let vocab = cv.vocabulary().unwrap();
        let the_idx = vocab["the"];
        // "the" appears twice in doc 0
        assert!((result[[0, the_idx]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_unigram_counts() {
        let corpus = vec!["hello world hello".to_string()];
        let mut cv = CountVectorizer::new();
        let result = cv.fit_transform_text_dense(&corpus).unwrap();

        let vocab = cv.vocabulary().unwrap();
        let hello_idx = vocab["hello"];
        let world_idx = vocab["world"];

        assert!((result[[0, hello_idx]] - 2.0).abs() < 1e-10);
        assert!((result[[0, world_idx]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bigram_generation() {
        let corpus = vec!["the cat sat".to_string(), "the dog sat".to_string()];
        let mut cv = CountVectorizer::new().with_ngram_range((1, 2));
        cv.fit_text(&corpus).unwrap();

        let vocab = cv.vocabulary().unwrap();
        // Should have unigrams and bigrams
        assert!(vocab.contains_key("the"));
        assert!(vocab.contains_key("cat"));
        assert!(vocab.contains_key("the cat"));
        assert!(vocab.contains_key("cat sat"));
        assert!(vocab.contains_key("the dog"));
    }

    #[test]
    fn test_min_df_filtering() {
        let corpus = vec![
            "apple banana".to_string(),
            "apple cherry".to_string(),
            "apple date".to_string(),
        ];
        let mut cv = CountVectorizer::new().with_min_df(DocFrequency::Count(2));
        cv.fit_text(&corpus).unwrap();

        let vocab = cv.vocabulary().unwrap();
        // "apple" appears in 3 docs (>= 2): included
        assert!(vocab.contains_key("apple"));
        // "banana", "cherry", "date" appear in 1 doc (< 2): excluded
        assert!(!vocab.contains_key("banana"));
        assert!(!vocab.contains_key("cherry"));
        assert!(!vocab.contains_key("date"));
    }

    #[test]
    fn test_max_df_filtering() {
        // 5 documents, max_df=0.8 means max_df_abs = 4
        let corpus = vec![
            "common rare1".to_string(),
            "common rare2".to_string(),
            "common rare3".to_string(),
            "common rare4".to_string(),
            "common rare5".to_string(),
        ];
        let mut cv = CountVectorizer::new().with_max_df(DocFrequency::Fraction(0.8));
        cv.fit_text(&corpus).unwrap();

        let vocab = cv.vocabulary().unwrap();
        // "common" appears in 5/5 = 100% docs (> 80%): excluded
        assert!(!vocab.contains_key("common"));
        // Each "rareX" appears in 1/5 = 20% docs: included
        assert!(vocab.contains_key("rare1"));
    }

    #[test]
    fn test_max_features_cap() {
        let corpus = vec![
            "alpha bravo charlie".to_string(),
            "alpha bravo delta".to_string(),
            "alpha echo foxtrot".to_string(),
        ];
        let mut cv = CountVectorizer::new().with_max_features(2);
        cv.fit_text(&corpus).unwrap();

        let vocab = cv.vocabulary().unwrap();
        assert_eq!(vocab.len(), 2);
        // "alpha" has df=3 (highest), "bravo" has df=2 (second highest)
        assert!(vocab.contains_key("alpha"));
        assert!(vocab.contains_key("bravo"));
    }

    #[test]
    fn test_binary_mode() {
        let corpus = vec!["hello hello hello world".to_string()];
        let mut cv = CountVectorizer::new().with_binary(true);
        let result = cv.fit_transform_text_dense(&corpus).unwrap();

        // All non-zero counts should be 1.0
        for val in result.iter() {
            assert!(*val == 0.0 || *val == 1.0);
        }

        let vocab = cv.vocabulary().unwrap();
        let hello_idx = vocab["hello"];
        assert!((result[[0, hello_idx]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_stop_words_filtering() {
        let corpus = vec!["the cat sat on the mat".to_string()];
        let mut cv =
            CountVectorizer::new().with_stop_words(vec!["the".to_string(), "on".to_string()]);
        cv.fit_text(&corpus).unwrap();

        let vocab = cv.vocabulary().unwrap();
        assert!(!vocab.contains_key("the"));
        assert!(!vocab.contains_key("on"));
        assert!(vocab.contains_key("cat"));
        assert!(vocab.contains_key("sat"));
        assert!(vocab.contains_key("mat"));
    }

    #[test]
    fn test_empty_document_in_corpus() {
        let corpus = vec![
            "hello world".to_string(),
            "".to_string(),
            "hello again".to_string(),
        ];
        let mut cv = CountVectorizer::new();
        let result = cv.fit_transform_text_dense(&corpus).unwrap();

        assert_eq!(result.nrows(), 3);
        // Empty doc row should be all zeros
        for j in 0..result.ncols() {
            assert!((result[[1, j]]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_single_document() {
        let corpus = vec!["alpha bravo charlie".to_string()];
        let mut cv = CountVectorizer::new();
        let result = cv.fit_transform_text_dense(&corpus).unwrap();

        assert_eq!(result.nrows(), 1);
        assert_eq!(result.ncols(), 3);
    }

    #[test]
    fn test_empty_corpus_error() {
        let corpus: Vec<String> = vec![];
        let mut cv = CountVectorizer::new();
        assert!(cv.fit_text(&corpus).is_err());
    }

    #[test]
    fn test_vocabulary_getter() {
        let corpus = vec!["cat dog".to_string()];
        let mut cv = CountVectorizer::new();
        assert!(cv.vocabulary().is_none());

        cv.fit_text(&corpus).unwrap();
        let vocab = cv.vocabulary().unwrap();
        assert_eq!(vocab.len(), 2);
        assert!(vocab.contains_key("cat"));
        assert!(vocab.contains_key("dog"));
    }

    #[test]
    fn test_feature_names_sorted() {
        let corpus = vec!["zebra apple mango".to_string()];
        let mut cv = CountVectorizer::new();
        cv.fit_text(&corpus).unwrap();

        let names = cv.get_feature_names().unwrap();
        assert_eq!(names, &["apple", "mango", "zebra"]);
    }

    #[test]
    fn test_transform_unseen_terms() {
        let train = vec!["cat dog".to_string()];
        let test = vec!["cat elephant".to_string()];

        let mut cv = CountVectorizer::new();
        cv.fit_text(&train).unwrap();
        let result = cv.transform_text_dense(&test).unwrap();

        let vocab = cv.vocabulary().unwrap();
        let cat_idx = vocab["cat"];
        // "cat" present
        assert!((result[[0, cat_idx]] - 1.0).abs() < 1e-10);
        // "elephant" not in vocab, should be ignored
        assert!(!vocab.contains_key("elephant"));
    }

    #[test]
    fn test_not_fitted_error() {
        let cv = CountVectorizer::new();
        let docs = vec!["hello".to_string()];
        assert!(cv.transform_text(&docs).is_err());
    }

    #[test]
    fn test_sparse_output_matches_dense() {
        let corpus = sample_corpus();
        let mut cv = CountVectorizer::new();
        cv.fit_text(&corpus).unwrap();

        let sparse = cv.transform_text(&corpus).unwrap();
        let dense = cv.transform_text_dense(&corpus).unwrap();
        let sparse_dense = sparse.to_dense();

        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                assert!(
                    (dense[[i, j]] - sparse_dense[[i, j]]).abs() < 1e-10,
                    "Mismatch at ({}, {}): dense={}, sparse={}",
                    i,
                    j,
                    dense[[i, j]],
                    sparse_dense[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_ngram_range_only_bigrams() {
        let corpus = vec!["the cat sat".to_string()];
        let mut cv = CountVectorizer::new().with_ngram_range((2, 2));
        cv.fit_text(&corpus).unwrap();

        let vocab = cv.vocabulary().unwrap();
        // Only bigrams, no unigrams
        assert!(!vocab.contains_key("the"));
        assert!(!vocab.contains_key("cat"));
        assert!(vocab.contains_key("the cat"));
        assert!(vocab.contains_key("cat sat"));
    }

    #[test]
    fn test_min_df_fraction() {
        // 4 docs, min_df=0.5 means min_df_abs = 2
        let corpus = vec![
            "alpha bravo".to_string(),
            "alpha charlie".to_string(),
            "alpha delta".to_string(),
            "alpha echo".to_string(),
        ];
        let mut cv = CountVectorizer::new().with_min_df(DocFrequency::Fraction(0.5));
        cv.fit_text(&corpus).unwrap();

        let vocab = cv.vocabulary().unwrap();
        // "alpha" appears in 4 docs (>= 2): included
        assert!(vocab.contains_key("alpha"));
        // Each other term appears in 1 doc (< 2): excluded
        assert!(!vocab.contains_key("bravo"));
    }
}

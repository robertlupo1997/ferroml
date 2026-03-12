//! Tests for text feature extraction pipeline (CountVectorizer + TfidfTransformer).

#[cfg(test)]
mod tests {
    use crate::preprocessing::count_vectorizer::{CountVectorizer, DocFrequency, TextTransformer};
    use crate::preprocessing::tfidf::TfidfTransformer;

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
        let corpus = vec!["hello world test".to_string()];
        let mut cv_clone = cv;
        let result = cv_clone.fit_transform_text_dense(&corpus).unwrap();
        assert_eq!(result.nrows(), 1);
        assert_eq!(result.ncols(), 3); // "hello", "test", "world"
    }

    #[test]
    fn test_unigram_counts_manual() {
        let corpus = vec![
            "apple banana apple".to_string(),
            "banana cherry".to_string(),
        ];
        let mut cv = CountVectorizer::new();
        let result = cv.fit_transform_text_dense(&corpus).unwrap();

        let vocab = cv.vocabulary().unwrap();
        // "apple" appears twice in doc 0
        assert!((result[[0, vocab["apple"]]] - 2.0).abs() < 1e-10);
        // "banana" appears once in doc 0
        assert!((result[[0, vocab["banana"]]] - 1.0).abs() < 1e-10);
        // "cherry" appears zero times in doc 0
        assert!((result[[0, vocab["cherry"]]]).abs() < 1e-10);
        // "banana" appears once in doc 1
        assert!((result[[1, vocab["banana"]]] - 1.0).abs() < 1e-10);
        // "cherry" appears once in doc 1
        assert!((result[[1, vocab["cherry"]]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bigram_generation() {
        let corpus = vec!["the cat sat".to_string(), "the dog sat".to_string()];
        let mut cv = CountVectorizer::new().with_ngram_range((1, 2));
        cv.fit_text(&corpus).unwrap();

        let vocab = cv.vocabulary().unwrap();
        // Should have both unigrams and bigrams
        assert!(vocab.contains_key("the"));
        assert!(vocab.contains_key("cat"));
        assert!(vocab.contains_key("sat"));
        assert!(vocab.contains_key("dog"));
        assert!(vocab.contains_key("the cat"));
        assert!(vocab.contains_key("cat sat"));
        assert!(vocab.contains_key("the dog"));
        assert!(vocab.contains_key("dog sat"));
    }

    #[test]
    fn test_min_df_filtering() {
        let corpus = vec![
            "alpha bravo charlie".to_string(),
            "alpha bravo delta".to_string(),
            "alpha echo foxtrot".to_string(),
        ];
        let mut cv = CountVectorizer::new().with_min_df(DocFrequency::Count(2));
        cv.fit_text(&corpus).unwrap();

        let vocab = cv.vocabulary().unwrap();
        // "alpha" df=3, "bravo" df=2: included
        assert!(vocab.contains_key("alpha"));
        assert!(vocab.contains_key("bravo"));
        // others df=1: excluded
        assert!(!vocab.contains_key("charlie"));
        assert!(!vocab.contains_key("delta"));
        assert!(!vocab.contains_key("echo"));
        assert!(!vocab.contains_key("foxtrot"));
    }

    #[test]
    fn test_max_df_filtering() {
        // 5 docs, max_df = 0.6 means max_df_abs = 3
        let corpus = vec![
            "common unique1".to_string(),
            "common unique2".to_string(),
            "common unique3".to_string(),
            "common unique4".to_string(),
            "rare unique5".to_string(),
        ];
        let mut cv = CountVectorizer::new().with_max_df(DocFrequency::Fraction(0.6));
        cv.fit_text(&corpus).unwrap();

        let vocab = cv.vocabulary().unwrap();
        // "common" appears in 4/5 = 80% > 60%: excluded
        assert!(!vocab.contains_key("common"));
        // "rare" appears in 1/5 = 20%: included
        assert!(vocab.contains_key("rare"));
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
        // "alpha" df=3 (highest), "bravo" df=2 (second highest)
        assert!(vocab.contains_key("alpha"));
        assert!(vocab.contains_key("bravo"));
    }

    #[test]
    fn test_binary_mode() {
        let corpus = vec!["hello hello hello world".to_string()];
        let mut cv = CountVectorizer::new().with_binary(true);
        let result = cv.fit_transform_text_dense(&corpus).unwrap();

        let vocab = cv.vocabulary().unwrap();
        assert!((result[[0, vocab["hello"]]] - 1.0).abs() < 1e-10);
        assert!((result[[0, vocab["world"]]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lowercase_vs_case_sensitive() {
        let corpus = vec!["Hello HELLO hello".to_string()];

        // Lowercase (default)
        let mut cv_lower = CountVectorizer::new();
        let result_lower = cv_lower.fit_transform_text_dense(&corpus).unwrap();
        let vocab_lower = cv_lower.vocabulary().unwrap();
        assert_eq!(vocab_lower.len(), 1);
        assert!((result_lower[[0, vocab_lower["hello"]]] - 3.0).abs() < 1e-10);

        // Case sensitive
        let mut cv_case = CountVectorizer::new().with_lowercase(false);
        let result_case = cv_case.fit_transform_text_dense(&corpus).unwrap();
        let vocab_case = cv_case.vocabulary().unwrap();
        assert_eq!(vocab_case.len(), 3); // "Hello", "HELLO", "hello"
        for val in result_case.iter() {
            assert!((*val - 1.0).abs() < 1e-10);
        }
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
    fn test_empty_document_handling() {
        let corpus = vec![
            "hello world".to_string(),
            "".to_string(),
            "hello again".to_string(),
        ];
        let mut cv = CountVectorizer::new();
        let result = cv.fit_transform_text_dense(&corpus).unwrap();

        assert_eq!(result.nrows(), 3);
        // Empty doc should be all zeros
        for j in 0..result.ncols() {
            assert!((result[[1, j]]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_single_document() {
        let corpus = vec!["one two three".to_string()];
        let mut cv = CountVectorizer::new();
        let result = cv.fit_transform_text_dense(&corpus).unwrap();

        assert_eq!(result.nrows(), 1);
        assert_eq!(result.ncols(), 3);
    }

    #[test]
    fn test_count_vectorizer_to_tfidf_pipeline() {
        let corpus = sample_corpus();
        let mut cv = CountVectorizer::new();
        let counts = cv.fit_transform_text_dense(&corpus).unwrap();

        // Now apply TF-IDF on the count matrix
        let mut tfidf = TfidfTransformer::new();
        let tfidf_result = tfidf.fit_transform(&counts).unwrap();

        // Same shape
        assert_eq!(tfidf_result.dim(), counts.dim());

        // Each row should have unit L2 norm (default)
        for row in tfidf_result.rows() {
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
    fn test_vocabulary_getter() {
        let mut cv = CountVectorizer::new();
        assert!(cv.vocabulary().is_none());

        cv.fit_text(&vec!["cat dog".to_string()]).unwrap();
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
    fn test_transform_unseen_document() {
        let train = vec!["cat dog fish".to_string()];
        let test = vec!["cat elephant giraffe".to_string()];

        let mut cv = CountVectorizer::new();
        cv.fit_text(&train).unwrap();
        let result = cv.transform_text_dense(&test).unwrap();

        let vocab = cv.vocabulary().unwrap();
        assert_eq!(result.ncols(), 3); // cat, dog, fish
        assert!((result[[0, vocab["cat"]]] - 1.0).abs() < 1e-10);
        assert!((result[[0, vocab["dog"]]]).abs() < 1e-10); // not in test doc
        assert!((result[[0, vocab["fish"]]]).abs() < 1e-10); // not in test doc
    }

    #[test]
    fn test_sparse_output() {
        let corpus = sample_corpus();
        let mut cv = CountVectorizer::new();
        cv.fit_text(&corpus).unwrap();

        let sparse = cv.transform_text(&corpus).unwrap();
        let dense = cv.transform_text_dense(&corpus).unwrap();
        let sparse_dense = sparse.to_dense();

        assert_eq!(dense.dim(), sparse_dense.dim());
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                assert!(
                    (dense[[i, j]] - sparse_dense[[i, j]]).abs() < 1e-10,
                    "Mismatch at ({}, {})",
                    i,
                    j,
                );
            }
        }
    }
}

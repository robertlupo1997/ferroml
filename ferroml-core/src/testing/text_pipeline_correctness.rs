//! Correctness tests for TextPipeline that compare FerroML output against sklearn fixtures.
//!
//! These tests load JSON fixtures from `ferroml-core/fixtures/text_pipeline/` and verify
//! that FerroML's TfidfVectorizer and TextPipeline produce results matching sklearn.

#[cfg(test)]
#[cfg(feature = "sparse")]
mod tests {
    use crate::models::{LogisticRegression, MultinomialNB};
    use crate::pipeline::TextPipeline;
    use crate::preprocessing::count_vectorizer::{CountVectorizer, TextTransformer};
    use crate::preprocessing::tfidf::TfidfNorm;
    use crate::preprocessing::tfidf::TfidfTransformer;
    use crate::preprocessing::tfidf_vectorizer::TfidfVectorizer;
    use ndarray::Array1;
    use std::collections::HashMap;

    // =========================================================================
    // Fixture parsing helpers
    // =========================================================================

    fn parse_matrix(val: &serde_json::Value) -> Vec<Vec<f64>> {
        val.as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect()
            })
            .collect()
    }

    fn parse_vec(val: &serde_json::Value) -> Vec<f64> {
        val.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    }

    fn parse_string_vec(val: &serde_json::Value) -> Vec<String> {
        val.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    }

    fn parse_vocab(val: &serde_json::Value) -> HashMap<String, usize> {
        val.as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_u64().unwrap() as usize))
            .collect()
    }

    // =========================================================================
    // TF-IDF default parameter tests
    // =========================================================================

    #[test]
    fn test_tfidf_default_values_match_sklearn() {
        let fixture_str = include_str!("../../fixtures/text_pipeline/tfidf_default.json");
        let fixture: serde_json::Value = serde_json::from_str(fixture_str).unwrap();

        let corpus = parse_string_vec(&fixture["corpus"]);
        let expected_matrix = parse_matrix(&fixture["matrix"]);

        let mut vectorizer = TfidfVectorizer::new();
        vectorizer.fit_text(&corpus).unwrap();
        let result = vectorizer.transform_text(&corpus).unwrap();
        let dense = result.to_dense();

        assert_eq!(dense.nrows(), expected_matrix.len(), "Row count mismatch");
        assert_eq!(
            dense.ncols(),
            expected_matrix[0].len(),
            "Column count mismatch"
        );

        // To compare element-by-element, we need to align by vocabulary.
        // Both should have same vocab, so we compare via feature names.
        let expected_vocab = parse_vocab(&fixture["vocabulary"]);
        let ferro_vocab = vectorizer.vocabulary().unwrap().clone();

        // Verify vocab size matches
        assert_eq!(
            ferro_vocab.len(),
            expected_vocab.len(),
            "Vocabulary size mismatch: ferro={}, sklearn={}",
            ferro_vocab.len(),
            expected_vocab.len()
        );

        // Compare matrix values using vocabulary index mapping
        for (term, &sklearn_idx) in &expected_vocab {
            let ferro_idx = ferro_vocab
                .get(term)
                .unwrap_or_else(|| panic!("Term '{}' not found in FerroML vocabulary", term));
            for row in 0..dense.nrows() {
                let ferro_val = dense[[row, *ferro_idx]];
                let sklearn_val = expected_matrix[row][sklearn_idx];
                assert!(
                    (ferro_val - sklearn_val).abs() < 1e-5,
                    "Mismatch at row={}, term='{}': ferro={}, sklearn={}",
                    row,
                    term,
                    ferro_val,
                    sklearn_val
                );
            }
        }
    }

    #[test]
    fn test_tfidf_default_vocabulary_matches_sklearn() {
        let fixture_str = include_str!("../../fixtures/text_pipeline/tfidf_default.json");
        let fixture: serde_json::Value = serde_json::from_str(fixture_str).unwrap();

        let corpus = parse_string_vec(&fixture["corpus"]);
        let expected_vocab = parse_vocab(&fixture["vocabulary"]);

        let mut vectorizer = TfidfVectorizer::new();
        vectorizer.fit_text(&corpus).unwrap();
        let ferro_vocab = vectorizer.vocabulary().unwrap().clone();

        // Same terms
        let mut expected_terms: Vec<&String> = expected_vocab.keys().collect();
        expected_terms.sort();
        let mut ferro_terms: Vec<&String> = ferro_vocab.keys().collect();
        ferro_terms.sort();

        assert_eq!(
            expected_terms, ferro_terms,
            "Vocabulary terms differ.\nsklearn: {:?}\nferro: {:?}",
            expected_terms, ferro_terms
        );

        // Same indices
        for (term, &sklearn_idx) in &expected_vocab {
            let ferro_idx = ferro_vocab[term];
            assert_eq!(
                ferro_idx, sklearn_idx,
                "Index mismatch for term '{}': ferro={}, sklearn={}",
                term, ferro_idx, sklearn_idx
            );
        }
    }

    #[test]
    fn test_tfidf_default_idf_matches_sklearn() {
        let fixture_str = include_str!("../../fixtures/text_pipeline/tfidf_default.json");
        let fixture: serde_json::Value = serde_json::from_str(fixture_str).unwrap();

        let corpus = parse_string_vec(&fixture["corpus"]);
        let expected_idf = parse_vec(&fixture["idf"]);

        let mut vectorizer = TfidfVectorizer::new();
        vectorizer.fit_text(&corpus).unwrap();
        let ferro_idf = vectorizer.idf().unwrap();

        assert_eq!(
            ferro_idf.len(),
            expected_idf.len(),
            "IDF length mismatch: ferro={}, sklearn={}",
            ferro_idf.len(),
            expected_idf.len()
        );

        for i in 0..expected_idf.len() {
            assert!(
                (ferro_idf[i] - expected_idf[i]).abs() < 1e-6,
                "IDF mismatch at index {}: ferro={}, sklearn={}",
                i,
                ferro_idf[i],
                expected_idf[i]
            );
        }
    }

    // =========================================================================
    // TF-IDF non-default parameter tests
    // =========================================================================

    #[test]
    fn test_tfidf_nondefault_values_match_sklearn() {
        let fixture_str = include_str!("../../fixtures/text_pipeline/tfidf_nondefault.json");
        let fixture: serde_json::Value = serde_json::from_str(fixture_str).unwrap();

        let corpus = parse_string_vec(&fixture["corpus"]);
        let expected_matrix = parse_matrix(&fixture["matrix"]);

        // Configure with non-default params: sublinear_tf=true, norm=L1, max_features=10, ngram_range=(1,2)
        let mut vectorizer = TfidfVectorizer::new()
            .with_sublinear_tf(true)
            .with_norm(TfidfNorm::L1)
            .with_max_features(10)
            .with_ngram_range((1, 2));

        vectorizer.fit_text(&corpus).unwrap();
        let result = vectorizer.transform_text(&corpus).unwrap();
        let dense = result.to_dense();

        assert_eq!(dense.nrows(), expected_matrix.len(), "Row count mismatch");
        assert_eq!(
            dense.ncols(),
            expected_matrix[0].len(),
            "Column count mismatch"
        );

        let expected_vocab = parse_vocab(&fixture["vocabulary"]);
        let ferro_vocab = vectorizer.vocabulary().unwrap().clone();

        // Compare matrix values using vocabulary index mapping
        for (term, &sklearn_idx) in &expected_vocab {
            let ferro_idx = ferro_vocab
                .get(term)
                .unwrap_or_else(|| panic!("Term '{}' not found in FerroML vocabulary", term));
            for row in 0..dense.nrows() {
                let ferro_val = dense[[row, *ferro_idx]];
                let sklearn_val = expected_matrix[row][sklearn_idx];
                assert!(
                    (ferro_val - sklearn_val).abs() < 1e-5,
                    "Mismatch at row={}, term='{}': ferro={}, sklearn={}",
                    row,
                    term,
                    ferro_val,
                    sklearn_val
                );
            }
        }
    }

    #[test]
    fn test_tfidf_nondefault_vocabulary_matches_sklearn() {
        let fixture_str = include_str!("../../fixtures/text_pipeline/tfidf_nondefault.json");
        let fixture: serde_json::Value = serde_json::from_str(fixture_str).unwrap();

        let corpus = parse_string_vec(&fixture["corpus"]);
        let expected_vocab = parse_vocab(&fixture["vocabulary"]);

        let mut vectorizer = TfidfVectorizer::new()
            .with_sublinear_tf(true)
            .with_norm(TfidfNorm::L1)
            .with_max_features(10)
            .with_ngram_range((1, 2));

        vectorizer.fit_text(&corpus).unwrap();
        let ferro_vocab = vectorizer.vocabulary().unwrap().clone();

        let mut expected_terms: Vec<&String> = expected_vocab.keys().collect();
        expected_terms.sort();
        let mut ferro_terms: Vec<&String> = ferro_vocab.keys().collect();
        ferro_terms.sort();

        assert_eq!(
            expected_terms, ferro_terms,
            "Vocabulary terms differ.\nsklearn: {:?}\nferro: {:?}",
            expected_terms, ferro_terms
        );
    }

    // =========================================================================
    // Pipeline prediction correctness tests
    // =========================================================================

    #[test]
    fn test_pipeline_multinomialnb_predictions_match_sklearn() {
        let fixture_str = include_str!("../../fixtures/text_pipeline/pipeline_multinomialnb.json");
        let fixture: serde_json::Value = serde_json::from_str(fixture_str).unwrap();

        let corpus_train = parse_string_vec(&fixture["corpus_train"]);
        let y_train_vec = parse_vec(&fixture["y_train"]);
        let y_train = Array1::from_vec(y_train_vec);
        let expected_train_preds = parse_vec(&fixture["train_predictions"]);

        let mut pipeline = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_sparse_model("model", MultinomialNB::new());

        pipeline.fit(&corpus_train, &y_train).unwrap();

        // Compare train predictions
        let train_preds = pipeline.predict(&corpus_train).unwrap();
        assert_eq!(
            train_preds.len(),
            expected_train_preds.len(),
            "Train prediction count mismatch"
        );
        for (i, (&ferro, &sklearn)) in train_preds
            .iter()
            .zip(expected_train_preds.iter())
            .enumerate()
        {
            assert!(
                (ferro - sklearn).abs() < 1e-10,
                "Train prediction mismatch at index {}: ferro={}, sklearn={}",
                i,
                ferro,
                sklearn
            );
        }

        // Compare test predictions if available
        if let Some(corpus_test_val) = fixture.get("corpus_test") {
            let corpus_test = parse_string_vec(corpus_test_val);
            let expected_test_preds = parse_vec(&fixture["test_predictions"]);

            let test_preds = pipeline.predict(&corpus_test).unwrap();
            assert_eq!(
                test_preds.len(),
                expected_test_preds.len(),
                "Test prediction count mismatch"
            );
            for (i, (&ferro, &sklearn)) in test_preds
                .iter()
                .zip(expected_test_preds.iter())
                .enumerate()
            {
                assert!(
                    (ferro - sklearn).abs() < 1e-10,
                    "Test prediction mismatch at index {}: ferro={}, sklearn={}",
                    i,
                    ferro,
                    sklearn
                );
            }
        }
    }

    #[test]
    fn test_pipeline_logistic_predictions_match_sklearn() {
        let fixture_str = include_str!("../../fixtures/text_pipeline/pipeline_logistic.json");
        let fixture: serde_json::Value = serde_json::from_str(fixture_str).unwrap();

        let corpus_train = parse_string_vec(&fixture["corpus_train"]);
        let y_train_vec = parse_vec(&fixture["y_train"]);
        let y_train = Array1::from_vec(y_train_vec);
        let expected_train_preds = parse_vec(&fixture["train_predictions"]);

        // Pipeline: CountVectorizer + TfidfTransformer + LogisticRegression
        let mut pipeline = TextPipeline::new()
            .add_text_transformer("cv", CountVectorizer::new())
            .add_sparse_transformer("tfidf", TfidfTransformer::new())
            .add_sparse_model("model", LogisticRegression::new());

        pipeline.fit(&corpus_train, &y_train).unwrap();

        let train_preds = pipeline.predict(&corpus_train).unwrap();
        assert_eq!(
            train_preds.len(),
            expected_train_preds.len(),
            "Train prediction count mismatch"
        );
        for (i, (&ferro, &sklearn)) in train_preds
            .iter()
            .zip(expected_train_preds.iter())
            .enumerate()
        {
            assert!(
                (ferro - sklearn).abs() < 1e-10,
                "Train prediction mismatch at index {}: ferro={}, sklearn={}",
                i,
                ferro,
                sklearn
            );
        }
    }
}

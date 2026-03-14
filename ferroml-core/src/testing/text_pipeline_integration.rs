//! Integration tests for TextPipeline end-to-end workflows.
//!
//! These tests verify that TextPipeline correctly chains text transformers,
//! sparse transformers, and models for classification and regression tasks.

#[cfg(test)]
#[cfg(feature = "sparse")]
mod tests {
    use crate::hpo::ParameterValue;
    use crate::models::{
        BernoulliNB, KNeighborsClassifier, LinearSVC, LogisticRegression, MultinomialNB,
        RidgeRegression,
    };
    use crate::pipeline::{PipelineSparseModel, TextPipeline};
    use crate::preprocessing::count_vectorizer::{CountVectorizer, TextTransformer};
    use crate::preprocessing::tfidf::TfidfTransformer;
    use crate::preprocessing::tfidf_vectorizer::TfidfVectorizer;
    use crate::preprocessing::SparseTransformer;
    use ndarray::Array1;
    use std::collections::HashMap;

    fn sample_corpus() -> Vec<String> {
        vec![
            "the cat sat on the mat".to_string(),
            "the dog sat on the log".to_string(),
            "the cat and the dog".to_string(),
            "a bird flew over the house".to_string(),
        ]
    }

    fn binary_labels() -> Array1<f64> {
        Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0])
    }

    fn regression_targets() -> Array1<f64> {
        Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0])
    }

    // =========================================================================
    // End-to-end classification pipelines
    // =========================================================================

    #[test]
    fn test_pipeline_multinomial_nb_end_to_end() {
        let mut pipeline = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_sparse_model("model", MultinomialNB::new());

        let corpus = sample_corpus();
        let y = binary_labels();

        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();
        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite(), "Prediction should be finite, got {}", p);
        }
    }

    #[test]
    fn test_pipeline_bernoulli_nb_end_to_end() {
        let mut pipeline = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_sparse_model("model", BernoulliNB::new());

        let corpus = sample_corpus();
        let y = binary_labels();

        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();
        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite(), "Prediction should be finite, got {}", p);
        }
    }

    #[test]
    fn test_pipeline_logistic_regression_end_to_end() {
        let mut pipeline = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_sparse_model("model", LogisticRegression::new());

        let corpus = sample_corpus();
        let y = binary_labels();

        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();
        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite(), "Prediction should be finite, got {}", p);
        }
    }

    #[test]
    fn test_pipeline_linear_svc_end_to_end() {
        let mut pipeline = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_sparse_model("model", LinearSVC::new());

        let corpus = sample_corpus();
        let y = binary_labels();

        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();
        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite(), "Prediction should be finite, got {}", p);
        }
    }

    #[test]
    fn test_pipeline_kneighbors_end_to_end() {
        let mut pipeline = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_sparse_model("model", KNeighborsClassifier::new(2));

        let corpus = sample_corpus();
        let y = binary_labels();

        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();
        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite(), "Prediction should be finite, got {}", p);
        }
    }

    // =========================================================================
    // Regression pipeline
    // =========================================================================

    #[test]
    fn test_pipeline_ridge_regression_end_to_end() {
        let mut pipeline = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_sparse_model("model", RidgeRegression::new(1.0));

        let corpus = sample_corpus();
        let y = regression_targets();

        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();
        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite(), "Prediction should be finite, got {}", p);
        }
    }

    // =========================================================================
    // Three-step pipeline: CountVectorizer + TfidfTransformer + MultinomialNB
    // =========================================================================

    #[test]
    fn test_pipeline_cv_tfidf_mnb_three_step() {
        let mut pipeline = TextPipeline::new()
            .add_text_transformer("cv", CountVectorizer::new())
            .add_sparse_transformer("tfidf", TfidfTransformer::new())
            .add_sparse_model("model", MultinomialNB::new());

        let corpus = sample_corpus();
        let y = binary_labels();

        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();
        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite(), "Prediction should be finite, got {}", p);
        }
    }

    // =========================================================================
    // Dense model fallback via add_dense_model
    // =========================================================================

    #[test]
    fn test_pipeline_dense_model_fallback() {
        // Use LogisticRegression via add_dense_model to test the dense fallback path
        // (densifies sparse features before passing to model).
        let corpus = sample_corpus();
        let y = binary_labels();

        let mut pipeline = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_dense_model("model", LogisticRegression::new());

        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();
        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite(), "Prediction should be finite, got {}", p);
        }
    }

    // =========================================================================
    // Manual vs pipeline comparison
    // =========================================================================

    #[test]
    fn test_pipeline_manual_vs_pipeline_match() {
        let corpus = sample_corpus();
        let y = binary_labels();

        // --- Manual step-by-step ---
        let mut cv = CountVectorizer::new();
        cv.fit_text(&corpus).unwrap();
        let sparse_cv = cv.transform_text(&corpus).unwrap();

        let mut tfidf = TfidfTransformer::new();
        SparseTransformer::fit_sparse(&mut tfidf, &sparse_cv).unwrap();
        let sparse_tfidf = SparseTransformer::transform_sparse(&tfidf, &sparse_cv).unwrap();

        let mut mnb = MultinomialNB::new();
        PipelineSparseModel::fit_sparse(&mut mnb, &sparse_tfidf, &y).unwrap();
        let manual_preds = PipelineSparseModel::predict_sparse(&mnb, &sparse_tfidf).unwrap();

        // --- Pipeline ---
        let mut pipeline = TextPipeline::new()
            .add_text_transformer("cv", CountVectorizer::new())
            .add_sparse_transformer("tfidf", TfidfTransformer::new())
            .add_sparse_model("model", MultinomialNB::new());

        pipeline.fit(&corpus, &y).unwrap();
        let pipeline_preds = pipeline.predict(&corpus).unwrap();

        // Predictions should match exactly
        assert_eq!(manual_preds.len(), pipeline_preds.len());
        for (i, (&m, &p)) in manual_preds.iter().zip(pipeline_preds.iter()).enumerate() {
            assert!(
                (m - p).abs() < 1e-12,
                "Prediction mismatch at index {}: manual={}, pipeline={}",
                i,
                m,
                p
            );
        }
    }

    // =========================================================================
    // Large-scale test: 100 documents
    // =========================================================================

    #[test]
    fn test_pipeline_large_scale_100_docs() {
        let words = [
            "cat", "dog", "bird", "fish", "car", "bus", "train", "plane", "red", "blue",
        ];
        let mut docs = Vec::new();
        for i in 0..100 {
            let doc = format!(
                "{} {} {} {}",
                words[i % 10],
                words[(i + 1) % 10],
                words[(i + 2) % 10],
                words[(i + 3) % 10]
            );
            docs.push(doc);
        }
        let y = Array1::from_vec((0..100).map(|i| (i % 2) as f64).collect());

        let mut pipeline = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_sparse_model("model", MultinomialNB::new());

        pipeline.fit(&docs, &y).unwrap();
        let preds = pipeline.predict(&docs).unwrap();
        assert_eq!(preds.len(), 100);
        for &p in preds.iter() {
            assert!(p.is_finite(), "Prediction should be finite, got {}", p);
        }
    }

    // =========================================================================
    // Transform returns CsrMatrix
    // =========================================================================

    #[test]
    fn test_pipeline_transform_returns_csr() {
        let corpus = sample_corpus();
        let y = binary_labels();

        let mut pipeline = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_sparse_model("model", MultinomialNB::new());

        pipeline.fit(&corpus, &y).unwrap();

        let csr = pipeline.transform(&corpus).unwrap();
        let dense = csr.to_dense();
        assert_eq!(dense.nrows(), corpus.len());
        assert!(dense.ncols() > 0, "Feature matrix should have columns");
    }

    // =========================================================================
    // Search space includes all steps
    // =========================================================================

    #[test]
    fn test_pipeline_search_space_includes_all_steps() {
        // CountVectorizer and TfidfTransformer use default (empty) search_space,
        // so only MultinomialNB contributes params (alpha).
        let pipeline = TextPipeline::new()
            .add_text_transformer("cv", CountVectorizer::new())
            .add_sparse_transformer("tfidf", TfidfTransformer::new())
            .add_sparse_model("model", MultinomialNB::new());

        let space = pipeline.search_space();
        let param_names: Vec<String> = space.parameters.keys().cloned().collect();

        // Model params should be prefixed with step name
        let has_model_param = param_names.iter().any(|n| n.starts_with("model__"));
        assert!(
            has_model_param,
            "Search space should contain model__ params, got: {:?}",
            param_names
        );
    }

    // =========================================================================
    // set_params updates step
    // =========================================================================

    #[test]
    fn test_pipeline_set_params_updates_step() {
        let corpus = sample_corpus();
        let y = binary_labels();

        // Pipeline with use_idf=true (default)
        let mut pipeline_with_idf = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_sparse_model("model", MultinomialNB::new());

        pipeline_with_idf.fit(&corpus, &y).unwrap();
        let preds_with_idf = pipeline_with_idf.predict(&corpus).unwrap();

        // Pipeline with use_idf=false via set_params
        let mut pipeline_no_idf = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_sparse_model("model", MultinomialNB::new());

        let mut params = HashMap::new();
        params.insert("tfidf__use_idf".to_string(), ParameterValue::Bool(false));
        pipeline_no_idf.set_params(&params).unwrap();
        pipeline_no_idf.fit(&corpus, &y).unwrap();
        let preds_no_idf = pipeline_no_idf.predict(&corpus).unwrap();

        // The two pipelines should produce different feature representations,
        // which may or may not lead to different predictions on this tiny corpus.
        // At minimum, verify both produce valid predictions.
        assert_eq!(preds_with_idf.len(), corpus.len());
        assert_eq!(preds_no_idf.len(), corpus.len());

        // Verify the transform outputs differ (IDF vs no IDF should produce different values)
        let csr_with = pipeline_with_idf.transform(&corpus).unwrap();
        let csr_without = pipeline_no_idf.transform(&corpus).unwrap();
        let dense_with = csr_with.to_dense();
        let dense_without = csr_without.to_dense();

        // The matrices should differ since IDF reweighting changes values
        let mut any_diff = false;
        for (a, b) in dense_with.iter().zip(dense_without.iter()) {
            if (a - b).abs() > 1e-10 {
                any_diff = true;
                break;
            }
        }
        assert!(
            any_diff,
            "Disabling IDF should produce different feature values"
        );
    }
}

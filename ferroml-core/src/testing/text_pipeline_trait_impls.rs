//! Tests for PipelineTextTransformer, PipelineSparseTransformer, and PipelineSparseModel
//! trait implementations added in Phase S.4.

#[cfg(test)]
#[cfg(feature = "sparse")]
mod tests {
    use crate::hpo::ParameterValue;
    use crate::pipeline::{
        PipelineSparseModel, PipelineSparseTransformer, PipelineTextTransformer,
    };
    use crate::preprocessing::count_vectorizer::{CountVectorizer, TextTransformer};
    use crate::preprocessing::tfidf::TfidfTransformer;
    use crate::preprocessing::tfidf_vectorizer::TfidfVectorizer;
    use ndarray::Array1;

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
    // CountVectorizer PipelineTextTransformer tests
    // =========================================================================

    #[test]
    fn test_count_vectorizer_name() {
        let cv = CountVectorizer::new();
        assert_eq!(PipelineTextTransformer::name(&cv), "CountVectorizer");
    }

    #[test]
    fn test_count_vectorizer_clone_boxed() {
        let mut cv = CountVectorizer::new();
        cv.fit_text(&sample_corpus()).unwrap();
        let cloned = PipelineTextTransformer::clone_boxed(&cv);
        assert_eq!(cloned.name(), "CountVectorizer");
        assert!(cloned.n_features_out().is_some());
        assert_eq!(cloned.n_features_out(), cv.n_features_out());
    }

    #[test]
    fn test_count_vectorizer_n_features_out_before_fit() {
        let cv = CountVectorizer::new();
        assert!(PipelineTextTransformer::n_features_out(&cv).is_none());
    }

    #[test]
    fn test_count_vectorizer_n_features_out_after_fit() {
        let mut cv = CountVectorizer::new();
        cv.fit_text(&sample_corpus()).unwrap();
        let n = PipelineTextTransformer::n_features_out(&cv);
        assert!(n.is_some());
        assert!(n.unwrap() > 0);
    }

    #[test]
    fn test_count_vectorizer_set_param_max_features() {
        let mut cv = CountVectorizer::new();
        PipelineTextTransformer::set_param(&mut cv, "max_features", &ParameterValue::Int(5))
            .unwrap();
        cv.fit_text(&sample_corpus()).unwrap();
        let n = PipelineTextTransformer::n_features_out(&cv).unwrap();
        assert!(n <= 5);
    }

    #[test]
    fn test_count_vectorizer_set_param_binary() {
        let mut cv = CountVectorizer::new();
        PipelineTextTransformer::set_param(&mut cv, "binary", &ParameterValue::Bool(true)).unwrap();
        let corpus = vec!["hello hello hello world".to_string()];
        cv.fit_text(&corpus).unwrap();
        let result = cv.transform_text(&corpus).unwrap();
        let dense = result.to_dense();
        for val in dense.iter() {
            assert!(*val == 0.0 || *val == 1.0);
        }
    }

    #[test]
    fn test_count_vectorizer_set_param_lowercase() {
        let mut cv = CountVectorizer::new();
        PipelineTextTransformer::set_param(&mut cv, "lowercase", &ParameterValue::Bool(false))
            .unwrap();
        let corpus = vec!["Hello WORLD".to_string()];
        cv.fit_text(&corpus).unwrap();
        let vocab = cv.vocabulary().unwrap();
        assert!(vocab.contains_key("Hello"));
        assert!(vocab.contains_key("WORLD"));
    }

    #[test]
    fn test_count_vectorizer_set_param_unknown() {
        let mut cv = CountVectorizer::new();
        let result =
            PipelineTextTransformer::set_param(&mut cv, "nonexistent", &ParameterValue::Bool(true));
        assert!(result.is_err());
    }

    // =========================================================================
    // TfidfVectorizer PipelineTextTransformer tests
    // =========================================================================

    #[test]
    fn test_tfidf_vectorizer_name() {
        let tv = TfidfVectorizer::new();
        assert_eq!(PipelineTextTransformer::name(&tv), "TfidfVectorizer");
    }

    #[test]
    fn test_tfidf_vectorizer_clone_boxed() {
        let mut tv = TfidfVectorizer::new();
        tv.fit_text(&sample_corpus()).unwrap();
        let cloned = PipelineTextTransformer::clone_boxed(&tv);
        assert_eq!(cloned.name(), "TfidfVectorizer");
        assert!(cloned.n_features_out().is_some());
    }

    #[test]
    fn test_tfidf_vectorizer_n_features_out() {
        let mut tv = TfidfVectorizer::new();
        assert!(PipelineTextTransformer::n_features_out(&tv).is_none());
        tv.fit_text(&sample_corpus()).unwrap();
        assert!(PipelineTextTransformer::n_features_out(&tv).unwrap() > 0);
    }

    #[test]
    fn test_tfidf_vectorizer_set_param_max_features() {
        let mut tv = TfidfVectorizer::new();
        PipelineTextTransformer::set_param(&mut tv, "max_features", &ParameterValue::Int(3))
            .unwrap();
        tv.fit_text(&sample_corpus()).unwrap();
        assert!(PipelineTextTransformer::n_features_out(&tv).unwrap() <= 3);
    }

    #[test]
    fn test_tfidf_vectorizer_set_param_use_idf() {
        let mut tv = TfidfVectorizer::new();
        PipelineTextTransformer::set_param(&mut tv, "use_idf", &ParameterValue::Bool(false))
            .unwrap();
        tv.fit_text(&sample_corpus()).unwrap();
        assert!(tv.idf().is_none());
    }

    #[test]
    fn test_tfidf_vectorizer_set_param_unknown() {
        let mut tv = TfidfVectorizer::new();
        let result =
            PipelineTextTransformer::set_param(&mut tv, "nonexistent", &ParameterValue::Bool(true));
        assert!(result.is_err());
    }

    // =========================================================================
    // TfidfTransformer PipelineSparseTransformer tests
    // =========================================================================

    #[test]
    fn test_tfidf_transformer_sparse_transformer_name() {
        let t = TfidfTransformer::new();
        assert_eq!(PipelineSparseTransformer::name(&t), "TfidfTransformer");
    }

    #[test]
    fn test_tfidf_transformer_sparse_transformer_clone_boxed() {
        let mut t = TfidfTransformer::new();
        let x = crate::sparse::CsrMatrix::from_dense(&ndarray::array![
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0]
        ]);
        crate::preprocessing::SparseTransformer::fit_sparse(&mut t, &x).unwrap();
        let cloned = PipelineSparseTransformer::clone_boxed(&t);
        assert_eq!(cloned.name(), "TfidfTransformer");
    }

    #[test]
    fn test_tfidf_transformer_sparse_transformer_set_param() {
        let mut t = TfidfTransformer::new();
        PipelineSparseTransformer::set_param(&mut t, "use_idf", &ParameterValue::Bool(false))
            .unwrap();
        PipelineSparseTransformer::set_param(&mut t, "sublinear_tf", &ParameterValue::Bool(true))
            .unwrap();
        assert!(!t.use_idf());
        assert!(t.sublinear_tf());
    }

    #[test]
    fn test_tfidf_transformer_sparse_transformer_set_param_unknown() {
        let mut t = TfidfTransformer::new();
        let result = PipelineSparseTransformer::set_param(
            &mut t,
            "nonexistent",
            &ParameterValue::Bool(true),
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Helper: fit a text transformer and produce sparse X for model fitting
    // =========================================================================

    fn fit_cv_and_transform(corpus: &[String]) -> crate::sparse::CsrMatrix {
        let mut cv = CountVectorizer::new();
        cv.fit_text(corpus).unwrap();
        cv.transform_text(corpus).unwrap()
    }

    // =========================================================================
    // PipelineSparseModel smoke tests (fit + predict) for all 11 models
    // =========================================================================

    #[test]
    fn test_multinomial_nb_pipeline_sparse_model() {
        use crate::models::MultinomialNB;
        let corpus = sample_corpus();
        let x = fit_cv_and_transform(&corpus);
        let y = binary_labels();
        let mut model = MultinomialNB::new();
        PipelineSparseModel::fit_sparse(&mut model, &x, &y).unwrap();
        assert!(PipelineSparseModel::is_fitted(&model));
        let preds = PipelineSparseModel::predict_sparse(&model, &x).unwrap();
        assert_eq!(preds.len(), corpus.len());
        assert_eq!(PipelineSparseModel::name(&model), "MultinomialNB");
    }

    #[test]
    fn test_bernoulli_nb_pipeline_sparse_model() {
        use crate::models::BernoulliNB;
        let corpus = sample_corpus();
        let x = fit_cv_and_transform(&corpus);
        let y = binary_labels();
        let mut model = BernoulliNB::new();
        PipelineSparseModel::fit_sparse(&mut model, &x, &y).unwrap();
        assert!(PipelineSparseModel::is_fitted(&model));
        let preds = PipelineSparseModel::predict_sparse(&model, &x).unwrap();
        assert_eq!(preds.len(), corpus.len());
        assert_eq!(PipelineSparseModel::name(&model), "BernoulliNB");
    }

    #[test]
    fn test_gaussian_nb_pipeline_sparse_model() {
        use crate::models::GaussianNB;
        let corpus = sample_corpus();
        let x = fit_cv_and_transform(&corpus);
        let y = binary_labels();
        let mut model = GaussianNB::new();
        PipelineSparseModel::fit_sparse(&mut model, &x, &y).unwrap();
        assert!(PipelineSparseModel::is_fitted(&model));
        let preds = PipelineSparseModel::predict_sparse(&model, &x).unwrap();
        assert_eq!(preds.len(), corpus.len());
        assert_eq!(PipelineSparseModel::name(&model), "GaussianNB");
    }

    #[test]
    fn test_categorical_nb_pipeline_sparse_model() {
        use crate::models::CategoricalNB;
        let corpus = sample_corpus();
        let x = fit_cv_and_transform(&corpus);
        let y = binary_labels();
        let mut model = CategoricalNB::new();
        PipelineSparseModel::fit_sparse(&mut model, &x, &y).unwrap();
        assert!(PipelineSparseModel::is_fitted(&model));
        let preds = PipelineSparseModel::predict_sparse(&model, &x).unwrap();
        assert_eq!(preds.len(), corpus.len());
        assert_eq!(PipelineSparseModel::name(&model), "CategoricalNB");
    }

    #[test]
    fn test_logistic_regression_pipeline_sparse_model() {
        use crate::models::LogisticRegression;
        let corpus = sample_corpus();
        let x = fit_cv_and_transform(&corpus);
        let y = binary_labels();
        let mut model = LogisticRegression::new();
        PipelineSparseModel::fit_sparse(&mut model, &x, &y).unwrap();
        assert!(PipelineSparseModel::is_fitted(&model));
        let preds = PipelineSparseModel::predict_sparse(&model, &x).unwrap();
        assert_eq!(preds.len(), corpus.len());
        assert_eq!(PipelineSparseModel::name(&model), "LogisticRegression");
    }

    #[test]
    fn test_linear_svc_pipeline_sparse_model() {
        use crate::models::LinearSVC;
        let corpus = sample_corpus();
        let x = fit_cv_and_transform(&corpus);
        let y = binary_labels();
        let mut model = LinearSVC::new();
        PipelineSparseModel::fit_sparse(&mut model, &x, &y).unwrap();
        assert!(PipelineSparseModel::is_fitted(&model));
        let preds = PipelineSparseModel::predict_sparse(&model, &x).unwrap();
        assert_eq!(preds.len(), corpus.len());
        assert_eq!(PipelineSparseModel::name(&model), "LinearSVC");
    }

    #[test]
    fn test_linear_svr_pipeline_sparse_model() {
        use crate::models::LinearSVR;
        let corpus = sample_corpus();
        let x = fit_cv_and_transform(&corpus);
        let y = regression_targets();
        let mut model = LinearSVR::new();
        PipelineSparseModel::fit_sparse(&mut model, &x, &y).unwrap();
        assert!(PipelineSparseModel::is_fitted(&model));
        let preds = PipelineSparseModel::predict_sparse(&model, &x).unwrap();
        assert_eq!(preds.len(), corpus.len());
        assert_eq!(PipelineSparseModel::name(&model), "LinearSVR");
    }

    #[test]
    fn test_kneighbors_classifier_pipeline_sparse_model() {
        use crate::models::KNeighborsClassifier;
        let corpus = sample_corpus();
        let x = fit_cv_and_transform(&corpus);
        let y = binary_labels();
        let mut model = KNeighborsClassifier::new(2);
        PipelineSparseModel::fit_sparse(&mut model, &x, &y).unwrap();
        assert!(PipelineSparseModel::is_fitted(&model));
        let preds = PipelineSparseModel::predict_sparse(&model, &x).unwrap();
        assert_eq!(preds.len(), corpus.len());
        assert_eq!(PipelineSparseModel::name(&model), "KNeighborsClassifier");
    }

    #[test]
    fn test_kneighbors_regressor_pipeline_sparse_model() {
        use crate::models::KNeighborsRegressor;
        let corpus = sample_corpus();
        let x = fit_cv_and_transform(&corpus);
        let y = regression_targets();
        let mut model = KNeighborsRegressor::new(2);
        PipelineSparseModel::fit_sparse(&mut model, &x, &y).unwrap();
        assert!(PipelineSparseModel::is_fitted(&model));
        let preds = PipelineSparseModel::predict_sparse(&model, &x).unwrap();
        assert_eq!(preds.len(), corpus.len());
        assert_eq!(PipelineSparseModel::name(&model), "KNeighborsRegressor");
    }

    #[test]
    fn test_nearest_centroid_pipeline_sparse_model() {
        use crate::models::NearestCentroid;
        let corpus = sample_corpus();
        let x = fit_cv_and_transform(&corpus);
        let y = binary_labels();
        let mut model = NearestCentroid::new();
        PipelineSparseModel::fit_sparse(&mut model, &x, &y).unwrap();
        assert!(PipelineSparseModel::is_fitted(&model));
        let preds = PipelineSparseModel::predict_sparse(&model, &x).unwrap();
        assert_eq!(preds.len(), corpus.len());
        assert_eq!(PipelineSparseModel::name(&model), "NearestCentroid");
    }

    #[test]
    fn test_ridge_regression_pipeline_sparse_model() {
        use crate::models::RidgeRegression;
        let corpus = sample_corpus();
        let x = fit_cv_and_transform(&corpus);
        let y = regression_targets();
        let mut model = RidgeRegression::new(1.0);
        PipelineSparseModel::fit_sparse(&mut model, &x, &y).unwrap();
        assert!(PipelineSparseModel::is_fitted(&model));
        let preds = PipelineSparseModel::predict_sparse(&model, &x).unwrap();
        assert_eq!(preds.len(), corpus.len());
        assert_eq!(PipelineSparseModel::name(&model), "RidgeRegression");
    }

    // =========================================================================
    // clone_boxed tests for models
    // =========================================================================

    #[test]
    fn test_multinomial_nb_clone_boxed() {
        use crate::models::MultinomialNB;
        let corpus = sample_corpus();
        let x = fit_cv_and_transform(&corpus);
        let y = binary_labels();
        let mut model = MultinomialNB::new();
        PipelineSparseModel::fit_sparse(&mut model, &x, &y).unwrap();
        let cloned = PipelineSparseModel::clone_boxed(&model);
        assert!(cloned.is_fitted());
        let preds = cloned.predict_sparse(&x).unwrap();
        assert_eq!(preds.len(), corpus.len());
    }

    #[test]
    fn test_logistic_regression_clone_boxed() {
        use crate::models::LogisticRegression;
        let corpus = sample_corpus();
        let x = fit_cv_and_transform(&corpus);
        let y = binary_labels();
        let mut model = LogisticRegression::new();
        PipelineSparseModel::fit_sparse(&mut model, &x, &y).unwrap();
        let cloned = PipelineSparseModel::clone_boxed(&model);
        assert!(cloned.is_fitted());
        let preds = cloned.predict_sparse(&x).unwrap();
        assert_eq!(preds.len(), corpus.len());
    }

    // =========================================================================
    // set_param tests for models
    // =========================================================================

    #[test]
    fn test_multinomial_nb_set_param_alpha() {
        use crate::models::MultinomialNB;
        let mut model = MultinomialNB::new();
        PipelineSparseModel::set_param(&mut model, "alpha", &ParameterValue::Float(2.0)).unwrap();
        assert!((model.alpha - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_bernoulli_nb_set_param_alpha_and_binarize() {
        use crate::models::BernoulliNB;
        let mut model = BernoulliNB::new();
        PipelineSparseModel::set_param(&mut model, "alpha", &ParameterValue::Float(0.5)).unwrap();
        assert!((model.alpha - 0.5).abs() < 1e-10);

        PipelineSparseModel::set_param(&mut model, "binarize", &ParameterValue::Float(0.5))
            .unwrap();
        assert_eq!(model.binarize, Some(0.5));

        // Negative value sets to None
        PipelineSparseModel::set_param(&mut model, "binarize", &ParameterValue::Float(-1.0))
            .unwrap();
        assert_eq!(model.binarize, None);
    }

    #[test]
    fn test_categorical_nb_set_param_alpha() {
        use crate::models::CategoricalNB;
        let mut model = CategoricalNB::new();
        PipelineSparseModel::set_param(&mut model, "alpha", &ParameterValue::Float(0.1)).unwrap();
        assert!((model.alpha - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_nb_set_param_unknown() {
        use crate::models::GaussianNB;
        let mut model = GaussianNB::new();
        let result =
            PipelineSparseModel::set_param(&mut model, "alpha", &ParameterValue::Float(1.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_logistic_regression_set_params() {
        use crate::models::LogisticRegression;
        let mut model = LogisticRegression::new();
        PipelineSparseModel::set_param(&mut model, "fit_intercept", &ParameterValue::Bool(false))
            .unwrap();
        assert!(!model.fit_intercept);

        PipelineSparseModel::set_param(&mut model, "l2_penalty", &ParameterValue::Float(0.01))
            .unwrap();
        assert!((model.l2_penalty - 0.01).abs() < 1e-10);

        PipelineSparseModel::set_param(&mut model, "max_iter", &ParameterValue::Int(200)).unwrap();
        assert_eq!(model.max_iter, 200);
    }

    #[test]
    fn test_linear_svc_set_params() {
        use crate::models::LinearSVC;
        let mut model = LinearSVC::new();
        PipelineSparseModel::set_param(&mut model, "C", &ParameterValue::Float(10.0)).unwrap();
        assert!((model.c - 10.0).abs() < 1e-10);

        PipelineSparseModel::set_param(&mut model, "c", &ParameterValue::Float(5.0)).unwrap();
        assert!((model.c - 5.0).abs() < 1e-10);

        PipelineSparseModel::set_param(&mut model, "max_iter", &ParameterValue::Int(500)).unwrap();
        assert_eq!(model.max_iter, 500);
    }

    #[test]
    fn test_linear_svr_set_params() {
        use crate::models::LinearSVR;
        let mut model = LinearSVR::new();
        PipelineSparseModel::set_param(&mut model, "C", &ParameterValue::Float(2.0)).unwrap();
        assert!((model.c - 2.0).abs() < 1e-10);

        PipelineSparseModel::set_param(&mut model, "epsilon", &ParameterValue::Float(0.5)).unwrap();
        assert!((model.epsilon - 0.5).abs() < 1e-10);

        PipelineSparseModel::set_param(&mut model, "max_iter", &ParameterValue::Int(300)).unwrap();
        assert_eq!(model.max_iter, 300);
    }

    #[test]
    fn test_kneighbors_classifier_set_param() {
        use crate::models::KNeighborsClassifier;
        let mut model = KNeighborsClassifier::new(5);
        PipelineSparseModel::set_param(&mut model, "n_neighbors", &ParameterValue::Int(3)).unwrap();
        assert_eq!(model.n_neighbors, 3);
    }

    #[test]
    fn test_kneighbors_regressor_set_param() {
        use crate::models::KNeighborsRegressor;
        let mut model = KNeighborsRegressor::new(5);
        PipelineSparseModel::set_param(&mut model, "n_neighbors", &ParameterValue::Int(7)).unwrap();
        assert_eq!(model.n_neighbors, 7);
    }

    #[test]
    fn test_nearest_centroid_set_param_unknown() {
        use crate::models::NearestCentroid;
        let mut model = NearestCentroid::new();
        let result =
            PipelineSparseModel::set_param(&mut model, "anything", &ParameterValue::Float(1.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_regression_set_param_alpha() {
        use crate::models::RidgeRegression;
        let mut model = RidgeRegression::new(1.0);
        PipelineSparseModel::set_param(&mut model, "alpha", &ParameterValue::Float(10.0)).unwrap();
        assert!((model.alpha - 10.0).abs() < 1e-10);
    }

    // =========================================================================
    // is_fitted tests
    // =========================================================================

    #[test]
    fn test_not_fitted_before_training() {
        use crate::models::MultinomialNB;
        let model = MultinomialNB::new();
        assert!(!PipelineSparseModel::is_fitted(&model));
    }

    // =========================================================================
    // TextPipeline integration smoke tests
    // =========================================================================

    #[test]
    fn test_text_pipeline_with_count_vectorizer_and_multinomial_nb() {
        use crate::models::MultinomialNB;
        use crate::pipeline::TextPipeline;

        let pipeline = TextPipeline::new()
            .add_text_transformer("cv", CountVectorizer::new())
            .add_sparse_model("model", MultinomialNB::new());

        let corpus = sample_corpus();
        let y = binary_labels();

        let mut pipeline = pipeline;
        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();
        assert_eq!(preds.len(), corpus.len());
    }

    #[test]
    fn test_text_pipeline_with_tfidf_vectorizer_and_logistic_regression() {
        use crate::models::LogisticRegression;
        use crate::pipeline::TextPipeline;

        let pipeline = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_sparse_model("model", LogisticRegression::new());

        let corpus = sample_corpus();
        let y = binary_labels();

        let mut pipeline = pipeline;
        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();
        assert_eq!(preds.len(), corpus.len());
    }

    #[test]
    fn test_text_pipeline_with_count_vectorizer_tfidf_transformer_and_linear_svc() {
        use crate::models::LinearSVC;
        use crate::pipeline::TextPipeline;

        let pipeline = TextPipeline::new()
            .add_text_transformer("cv", CountVectorizer::new())
            .add_sparse_transformer("tfidf", TfidfTransformer::new())
            .add_sparse_model("model", LinearSVC::new());

        let corpus = sample_corpus();
        let y = binary_labels();

        let mut pipeline = pipeline;
        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();
        assert_eq!(preds.len(), corpus.len());
    }

    #[test]
    fn test_text_pipeline_with_ridge_regression() {
        use crate::models::RidgeRegression;
        use crate::pipeline::TextPipeline;

        let pipeline = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_sparse_model("model", RidgeRegression::new(1.0));

        let corpus = sample_corpus();
        let y = regression_targets();

        let mut pipeline = pipeline;
        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();
        assert_eq!(preds.len(), corpus.len());
    }
}

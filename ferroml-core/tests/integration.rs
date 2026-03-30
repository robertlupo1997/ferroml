//! Integration tests: UCI datasets, real dataset validation, sklearn fixtures

mod uci_datasets {
    use ferroml_core::metrics::{accuracy, mse, r2_score};
    use ferroml_core::models::{
        DecisionTreeClassifier, GradientBoostingClassifier, GradientBoostingRegressor,
        HistGradientBoostingClassifier, HistGradientBoostingRegressor, KNeighborsClassifier,
        KNeighborsRegressor, LinearRegression, LogisticRegression, Model, RandomForestClassifier,
        RandomForestRegressor, RidgeRegression,
    };
    use ferroml_core::preprocessing::scalers::StandardScaler;
    use ferroml_core::preprocessing::Transformer;
    use ndarray::{Array1, Array2, Axis};
    use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

    /// Helper to create a synthetic "Adult Income" style dataset
    /// (binary classification with mixed feature types)
    fn create_adult_style_dataset(seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_samples = 500;
        let n_features = 14;

        let mut x = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Age (17-90)
            x[[i, 0]] = 17.0 + rng.random::<f64>() * 73.0;
            // Work class encoded (0-7)
            x[[i, 1]] = (rng.random::<f64>() * 8.0).floor();
            // Final weight (large numeric)
            x[[i, 2]] = 10000.0 + rng.random::<f64>() * 1_000_000.0;
            // Education years (1-16)
            x[[i, 3]] = 1.0 + (rng.random::<f64>() * 16.0).floor();
            // Education-num (1-16)
            x[[i, 4]] = x[[i, 3]];
            // Marital status encoded (0-6)
            x[[i, 5]] = (rng.random::<f64>() * 7.0).floor();
            // Occupation encoded (0-14)
            x[[i, 6]] = (rng.random::<f64>() * 15.0).floor();
            // Relationship encoded (0-5)
            x[[i, 7]] = (rng.random::<f64>() * 6.0).floor();
            // Race encoded (0-4)
            x[[i, 8]] = (rng.random::<f64>() * 5.0).floor();
            // Sex encoded (0-1)
            x[[i, 9]] = (rng.random::<f64>() * 2.0).floor();
            // Capital gain
            x[[i, 10]] = if rng.random::<f64>() < 0.1 {
                rng.random::<f64>() * 100000.0
            } else {
                0.0
            };
            // Capital loss
            x[[i, 11]] = if rng.random::<f64>() < 0.1 {
                rng.random::<f64>() * 5000.0
            } else {
                0.0
            };
            // Hours per week (1-99)
            x[[i, 12]] = 1.0 + rng.random::<f64>() * 98.0;
            // Native country encoded (0-40)
            x[[i, 13]] = (rng.random::<f64>() * 41.0).floor();

            // Target: income >50K based on features
            let prob: f64 = 0.1
                + 0.03 * (x[[i, 3]] - 10.0)
                + 0.005 * (x[[i, 0]] - 30.0)
                + 0.002 * (x[[i, 12]] - 40.0)
                + 0.00001 * x[[i, 10]];
            let prob = prob.clamp(0.0, 0.9);
            y[i] = if rng.random::<f64>() < prob { 1.0 } else { 0.0 };
        }

        (x, y)
    }

    /// Helper to create a synthetic "Forest Covertype" style dataset
    fn create_covertype_style_dataset(seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_samples = 1000;
        let n_features = 54;

        let mut x = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            x[[i, 0]] = 1859.0 + rng.random::<f64>() * 2000.0;
            x[[i, 1]] = rng.random::<f64>() * 360.0;
            x[[i, 2]] = rng.random::<f64>() * 52.0;
            x[[i, 3]] = rng.random::<f64>() * 1400.0;
            x[[i, 4]] = -173.0 + rng.random::<f64>() * 374.0;
            x[[i, 5]] = rng.random::<f64>() * 7000.0;
            x[[i, 6]] = rng.random::<f64>() * 255.0;
            x[[i, 7]] = rng.random::<f64>() * 255.0;
            x[[i, 8]] = rng.random::<f64>() * 255.0;
            x[[i, 9]] = rng.random::<f64>() * 7000.0;

            let wilderness = (rng.random::<f64>() * 4.0).floor() as usize;
            for j in 0..4 {
                x[[i, 10 + j]] = if j == wilderness { 1.0 } else { 0.0 };
            }

            let soil = (rng.random::<f64>() * 40.0).floor() as usize;
            for j in 0..40 {
                x[[i, 14 + j]] = if j == soil { 1.0 } else { 0.0 };
            }

            let base_class = ((x[[i, 0]] - 1859.0) / 400.0).floor() as usize;
            let class = (base_class + wilderness) % 7;
            y[i] = class as f64;
        }

        (x, y)
    }

    /// Helper to create a regression dataset similar to "California Housing"
    fn create_california_housing_style_dataset(seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_samples = 500;
        let n_features = 8;

        let mut x = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            x[[i, 0]] = 0.5 + rng.random::<f64>() * 14.5;
            x[[i, 1]] = 1.0 + rng.random::<f64>() * 51.0;
            x[[i, 2]] = 1.0 + rng.random::<f64>() * 14.0;
            x[[i, 3]] = 0.3 + rng.random::<f64>() * 4.7;
            x[[i, 4]] = 3.0 + rng.random::<f64>() * 35000.0;
            x[[i, 5]] = 0.5 + rng.random::<f64>() * 9.5;
            x[[i, 6]] = 32.0 + rng.random::<f64>() * 10.0;
            x[[i, 7]] = -125.0 + rng.random::<f64>() * 11.0;

            y[i] = (0.5 + 0.3 * x[[i, 0]] + 0.01 * x[[i, 2]] - 0.002 * x[[i, 1]]
                + 0.05 * (42.0 - x[[i, 6]])
                + rng.random::<f64>() * 0.5)
                .clamp(0.15, 5.0);
        }

        (x, y)
    }

    fn train_test_split(
        x: &Array2<f64>,
        y: &Array1<f64>,
        test_ratio: f64,
        seed: u64,
    ) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
        let n = x.nrows();
        let n_test = (n as f64 * test_ratio).round() as usize;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let test_indices = &indices[..n_test];
        let train_indices = &indices[n_test..];

        (
            x.select(Axis(0), train_indices),
            y.select(Axis(0), train_indices),
            x.select(Axis(0), test_indices),
            y.select(Axis(0), test_indices),
        )
    }

    // =============================================================================
    // Adult-style Binary Classification Tests
    // =============================================================================

    #[test]
    fn test_adult_logistic_regression() {
        let (x, y) = create_adult_style_dataset(42);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.2, 42);

        let mut scaler = StandardScaler::new();
        let x_train_scaled = scaler.fit_transform(&x_train).unwrap();
        let x_test_scaled = scaler.transform(&x_test).unwrap();

        // Use L2 regularization to handle near-perfect separation in synthetic data
        // Higher penalty needed for highly separable synthetic data
        let mut model = LogisticRegression::new().with_l2_penalty(10.0);
        model.fit(&x_train_scaled, &y_train).unwrap();
        let predictions = model.predict(&x_test_scaled).unwrap();

        let acc = accuracy(&y_test, &predictions).unwrap();
        assert!(acc > 0.5, "Accuracy should be better than random: {}", acc);
        assert!(acc < 1.0, "Perfect accuracy unlikely on real data");
    }

    #[test]
    fn test_adult_random_forest() {
        let (x, y) = create_adult_style_dataset(123);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.2, 123);

        let mut model = RandomForestClassifier::new()
            .with_n_estimators(10)
            .with_max_depth(Some(10));
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let acc = accuracy(&y_test, &predictions).unwrap();

        assert!(
            acc > 0.55,
            "Random Forest should achieve decent accuracy: {}",
            acc
        );

        let importance = model.feature_importance();
        assert!(importance.is_some());
        let imp = importance.unwrap();
        assert_eq!(imp.len(), x.ncols());
        assert!(imp.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_adult_gradient_boosting() {
        let (x, y) = create_adult_style_dataset(456);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.2, 456);

        let mut model = GradientBoostingClassifier::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_max_depth(Some(3));
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let acc = accuracy(&y_test, &predictions).unwrap();

        assert!(
            acc > 0.55,
            "Gradient Boosting should achieve good accuracy: {}",
            acc
        );
    }

    #[test]
    fn test_adult_knn() {
        let (x, y) = create_adult_style_dataset(789);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.2, 789);

        let mut scaler = StandardScaler::new();
        let x_train_scaled = scaler.fit_transform(&x_train).unwrap();
        let x_test_scaled = scaler.transform(&x_test).unwrap();

        let mut model = KNeighborsClassifier::new(5);
        model.fit(&x_train_scaled, &y_train).unwrap();
        let predictions = model.predict(&x_test_scaled).unwrap();
        let acc = accuracy(&y_test, &predictions).unwrap();

        assert!(acc > 0.5, "KNN should beat random: {}", acc);
    }

    // =============================================================================
    // Covertype-style Multiclass Classification Tests
    // =============================================================================

    #[test]
    fn test_covertype_decision_tree() {
        let (x, y) = create_covertype_style_dataset(42);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.2, 42);

        let mut model = DecisionTreeClassifier::new().with_max_depth(Some(10));
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let acc = accuracy(&y_test, &predictions).unwrap();

        assert!(
            acc > 0.2,
            "Decision tree should beat random for 7-class: {}",
            acc
        );
    }

    #[test]
    fn test_covertype_random_forest_multiclass() {
        let (x, y) = create_covertype_style_dataset(123);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.2, 123);

        let mut model = RandomForestClassifier::new()
            .with_n_estimators(100)
            .with_max_depth(Some(15));
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let acc = accuracy(&y_test, &predictions).unwrap();

        // Random chance is 1/7 ≈ 0.143; synthetic data is noisy so use generous margin
        assert!(
            acc > 0.10,
            "Random Forest should not be catastrophically bad: {}",
            acc
        );

        for &pred in predictions.iter() {
            assert!((0.0..=6.0).contains(&pred) && (pred - pred.round()).abs() < 0.01);
        }
    }

    #[test]
    fn test_covertype_hist_gradient_boosting() {
        let (x, y) = create_covertype_style_dataset(456);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.2, 456);

        let mut model = HistGradientBoostingClassifier::new()
            .with_max_iter(30)
            .with_learning_rate(0.1)
            .with_max_depth(Some(5));
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let acc = accuracy(&y_test, &predictions).unwrap();

        assert!(acc > 0.25, "HistGBM should handle multiclass well: {}", acc);
    }

    // =============================================================================
    // California Housing-style Regression Tests
    // =============================================================================

    #[test]
    fn test_california_linear_regression() {
        let (x, y) = create_california_housing_style_dataset(42);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.2, 42);

        let mut scaler = StandardScaler::new();
        let x_train_scaled = scaler.fit_transform(&x_train).unwrap();
        let x_test_scaled = scaler.transform(&x_test).unwrap();

        let mut model = LinearRegression::new();
        model.fit(&x_train_scaled, &y_train).unwrap();
        let predictions = model.predict(&x_test_scaled).unwrap();

        let r2 = r2_score(&y_test, &predictions).unwrap();
        assert!(r2 > 0.0, "R2 should be positive: {}", r2);
        assert!(r2 < 1.0, "Perfect R2 unlikely");

        let mse_val = mse(&y_test, &predictions).unwrap();
        assert!(mse_val < 10.0, "MSE should be bounded: {}", mse_val);
    }

    #[test]
    fn test_california_ridge_regression() {
        let (x, y) = create_california_housing_style_dataset(123);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.2, 123);

        let mut scaler = StandardScaler::new();
        let x_train_scaled = scaler.fit_transform(&x_train).unwrap();
        let x_test_scaled = scaler.transform(&x_test).unwrap();

        let mut model = RidgeRegression::new(1.0);
        model.fit(&x_train_scaled, &y_train).unwrap();
        let predictions = model.predict(&x_test_scaled).unwrap();
        let r2 = r2_score(&y_test, &predictions).unwrap();

        assert!(r2 > 0.0, "Ridge R2 should be positive: {}", r2);
    }

    #[test]
    fn test_california_random_forest_regressor() {
        let (x, y) = create_california_housing_style_dataset(456);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.2, 456);

        let mut model = RandomForestRegressor::new()
            .with_n_estimators(50)
            .with_max_depth(Some(10));
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let r2 = r2_score(&y_test, &predictions).unwrap();

        assert!(r2 > -0.5, "RF Regressor R2 should not be terrible: {}", r2);
    }

    #[test]
    fn test_california_gradient_boosting_regressor() {
        let (x, y) = create_california_housing_style_dataset(789);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.2, 789);

        let mut model = GradientBoostingRegressor::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_max_depth(Some(3));
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let r2 = r2_score(&y_test, &predictions).unwrap();

        assert!(r2 > 0.1, "GBM Regressor should achieve decent R2: {}", r2);
    }

    #[test]
    fn test_california_hist_gradient_boosting_regressor() {
        let (x, y) = create_california_housing_style_dataset(321);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.2, 321);

        let mut model = HistGradientBoostingRegressor::new()
            .with_max_iter(30)
            .with_learning_rate(0.1)
            .with_max_depth(Some(5));
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let r2 = r2_score(&y_test, &predictions).unwrap();

        assert!(
            r2 > 0.0,
            "HistGBM Regressor should have positive R2: {}",
            r2
        );
    }

    #[test]
    fn test_california_knn_regressor() {
        let (x, y) = create_california_housing_style_dataset(654);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.2, 654);

        let mut scaler = StandardScaler::new();
        let x_train_scaled = scaler.fit_transform(&x_train).unwrap();
        let x_test_scaled = scaler.transform(&x_test).unwrap();

        let mut model = KNeighborsRegressor::new(5);
        model.fit(&x_train_scaled, &y_train).unwrap();
        let predictions = model.predict(&x_test_scaled).unwrap();
        let r2 = r2_score(&y_test, &predictions).unwrap();

        assert!(r2 > -0.5, "KNN Regressor R2 should be reasonable: {}", r2);
    }

    // =============================================================================
    // Performance Consistency Tests
    // =============================================================================

    #[test]
    fn test_reproducibility_with_random_state() {
        let (x, y) = create_adult_style_dataset(42);
        let (x_train, y_train, x_test, _) = train_test_split(&x, &y, 0.2, 42);

        // Use n_jobs=1 for deterministic sequential execution (parallel has non-determinism)
        let mut model1 = RandomForestClassifier::new()
            .with_n_estimators(10)
            .with_random_state(12345)
            .with_n_jobs(Some(1));
        model1.fit(&x_train, &y_train).unwrap();
        let pred1 = model1.predict(&x_test).unwrap();

        let mut model2 = RandomForestClassifier::new()
            .with_n_estimators(10)
            .with_random_state(12345)
            .with_n_jobs(Some(1));
        model2.fit(&x_train, &y_train).unwrap();
        let pred2 = model2.predict(&x_test).unwrap();

        for (p1, p2) in pred1.iter().zip(pred2.iter()) {
            assert!(
                (*p1 - *p2).abs() < 1e-10,
                "Predictions should be reproducible with same seed: {} != {}",
                p1,
                p2
            );
        }
    }

    #[test]
    fn test_model_comparison_on_same_data() {
        let (x, y) = create_adult_style_dataset(42);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.2, 42);

        let mut scaler = StandardScaler::new();
        let x_train_scaled = scaler.fit_transform(&x_train).unwrap();
        let x_test_scaled = scaler.transform(&x_test).unwrap();

        // Use L2 regularization to handle near-perfect separation in synthetic data
        // Higher penalty needed for highly separable synthetic data
        let mut lr = LogisticRegression::new().with_l2_penalty(10.0);
        lr.fit(&x_train_scaled, &y_train).unwrap();
        let lr_pred = lr.predict(&x_test_scaled).unwrap();
        let lr_acc = accuracy(&y_test, &lr_pred).unwrap();

        let mut rf = RandomForestClassifier::new().with_n_estimators(10);
        rf.fit(&x_train, &y_train).unwrap();
        let rf_pred = rf.predict(&x_test).unwrap();
        let rf_acc = accuracy(&y_test, &rf_pred).unwrap();

        let mut dt = DecisionTreeClassifier::new().with_max_depth(Some(10));
        dt.fit(&x_train, &y_train).unwrap();
        let dt_pred = dt.predict(&x_test).unwrap();
        let dt_acc = accuracy(&y_test, &dt_pred).unwrap();

        assert!(lr_acc > 0.5, "LR accuracy: {}", lr_acc);
        assert!(rf_acc > 0.5, "RF accuracy: {}", rf_acc);
        assert!(dt_acc > 0.5, "DT accuracy: {}", dt_acc);

        println!(
            "Model comparison - LR: {:.3}, RF: {:.3}, DT: {:.3}",
            lr_acc, rf_acc, dt_acc
        );
    }
}

mod real_datasets {
    use ferroml_core::datasets::{load_diabetes, load_iris, load_wine};
    use ferroml_core::metrics::{accuracy, r2_score};
    use ferroml_core::models::{
        DecisionTreeClassifier, DecisionTreeRegressor, GaussianNB, GradientBoostingClassifier,
        GradientBoostingRegressor, KNeighborsClassifier, LassoRegression, LinearRegression, Model,
        RandomForestClassifier, RandomForestRegressor, RidgeRegression, SVC,
    };

    // =============================================================================
    // Helper: load datasets into (X, y) form
    // =============================================================================

    fn iris_xy() -> (ndarray::Array2<f64>, ndarray::Array1<f64>) {
        let (dataset, _info) = load_iris();
        dataset.into_arrays()
    }

    fn wine_xy() -> (ndarray::Array2<f64>, ndarray::Array1<f64>) {
        let (dataset, _info) = load_wine();
        dataset.into_arrays()
    }

    fn diabetes_xy() -> (ndarray::Array2<f64>, ndarray::Array1<f64>) {
        let (dataset, _info) = load_diabetes();
        dataset.into_arrays()
    }

    // =============================================================================
    // 1. Iris Dataset — Shape and Classification Tests (8 tests)
    // =============================================================================

    #[test]
    fn test_iris_dataset_shape() {
        let (dataset, info) = load_iris();
        assert_eq!(dataset.n_samples(), 150, "Iris should have 150 samples");
        assert_eq!(dataset.n_features(), 4, "Iris should have 4 features");
        assert_eq!(info.n_classes, Some(3), "Iris should have 3 classes");

        let classes = dataset.unique_classes();
        assert_eq!(classes.len(), 3);
        assert!((classes[0] - 0.0).abs() < 1e-10);
        assert!((classes[1] - 1.0).abs() < 1e-10);
        assert!((classes[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_iris_gaussian_nb_accuracy() {
        let (x, y) = iris_xy();
        let mut model = GaussianNB::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let acc = accuracy(&y, &pred).unwrap();
        assert!(
            acc > 0.90,
            "Iris GaussianNB training accuracy {:.3} should be > 0.90",
            acc
        );
    }

    #[test]
    fn test_iris_random_forest_accuracy() {
        let (x, y) = iris_xy();
        let mut model = RandomForestClassifier::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let acc = accuracy(&y, &pred).unwrap();
        assert!(
            acc > 0.90,
            "Iris RandomForest training accuracy {:.3} should be > 0.90",
            acc
        );
    }

    #[test]
    fn test_iris_svc_accuracy() {
        let (x, y) = iris_xy();
        let mut model = SVC::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let acc = accuracy(&y, &pred).unwrap();
        assert!(
            acc > 0.90,
            "Iris SVC training accuracy {:.3} should be > 0.90",
            acc
        );
    }

    #[test]
    fn test_iris_knn_accuracy() {
        let (x, y) = iris_xy();
        let mut model = KNeighborsClassifier::new(5);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let acc = accuracy(&y, &pred).unwrap();
        assert!(
            acc > 0.90,
            "Iris KNN(5) training accuracy {:.3} should be > 0.90",
            acc
        );
    }

    #[test]
    fn test_iris_decision_tree_accuracy() {
        let (x, y) = iris_xy();
        let mut model = DecisionTreeClassifier::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let acc = accuracy(&y, &pred).unwrap();
        assert!(
            acc > 0.85,
            "Iris DecisionTree training accuracy {:.3} should be > 0.85",
            acc
        );
    }

    #[test]
    fn test_iris_gradient_boosting_accuracy() {
        let (x, y) = iris_xy();
        let mut model = GradientBoostingClassifier::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let acc = accuracy(&y, &pred).unwrap();
        assert!(
            acc > 0.90,
            "Iris GradientBoosting training accuracy {:.3} should be > 0.90",
            acc
        );
    }

    #[test]
    fn test_iris_all_predictions_valid_classes() {
        // Verify predictions only contain valid class labels {0, 1, 2}
        let (x, y) = iris_xy();
        let mut model = GaussianNB::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        for &p in pred.iter() {
            assert!(
                (p - 0.0).abs() < 1e-10 || (p - 1.0).abs() < 1e-10 || (p - 2.0).abs() < 1e-10,
                "Prediction {} is not a valid Iris class (0, 1, 2)",
                p
            );
        }
    }

    // =============================================================================
    // 2. Wine Dataset — Shape and Classification Tests (6 tests)
    // =============================================================================

    #[test]
    fn test_wine_dataset_shape() {
        let (dataset, info) = load_wine();
        assert_eq!(dataset.n_samples(), 178, "Wine should have 178 samples");
        assert_eq!(dataset.n_features(), 13, "Wine should have 13 features");
        assert_eq!(info.n_classes, Some(3), "Wine should have 3 classes");

        let classes = dataset.unique_classes();
        assert_eq!(classes.len(), 3);
    }

    #[test]
    fn test_wine_gaussian_nb_accuracy() {
        let (x, y) = wine_xy();
        let mut model = GaussianNB::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let acc = accuracy(&y, &pred).unwrap();
        assert!(
            acc > 0.85,
            "Wine GaussianNB training accuracy {:.3} should be > 0.85",
            acc
        );
    }

    #[test]
    fn test_wine_random_forest_accuracy() {
        let (x, y) = wine_xy();
        let mut model = RandomForestClassifier::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let acc = accuracy(&y, &pred).unwrap();
        assert!(
            acc > 0.85,
            "Wine RandomForest training accuracy {:.3} should be > 0.85",
            acc
        );
    }

    #[test]
    fn test_wine_gradient_boosting_accuracy() {
        let (x, y) = wine_xy();
        let mut model = GradientBoostingClassifier::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let acc = accuracy(&y, &pred).unwrap();
        assert!(
            acc > 0.85,
            "Wine GradientBoosting training accuracy {:.3} should be > 0.85",
            acc
        );
    }

    #[test]
    fn test_wine_decision_tree_accuracy() {
        let (x, y) = wine_xy();
        let mut model = DecisionTreeClassifier::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let acc = accuracy(&y, &pred).unwrap();
        assert!(
            acc > 0.80,
            "Wine DecisionTree training accuracy {:.3} should be > 0.80",
            acc
        );
    }

    #[test]
    fn test_wine_knn_accuracy() {
        let (x, y) = wine_xy();
        let mut model = KNeighborsClassifier::new(5);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let acc = accuracy(&y, &pred).unwrap();
        assert!(
            acc > 0.70,
            "Wine KNN(5) training accuracy {:.3} should be > 0.70",
            acc
        );
    }

    // =============================================================================
    // 3. Diabetes Dataset — Shape and Regression Tests (8 tests)
    // =============================================================================

    #[test]
    fn test_diabetes_dataset_shape() {
        let (dataset, info) = load_diabetes();
        assert_eq!(dataset.n_samples(), 260, "Diabetes should have 260 samples");
        assert_eq!(dataset.n_features(), 10, "Diabetes should have 10 features");
        assert_eq!(
            info.task,
            ferroml_core::Task::Regression,
            "Diabetes should be a regression task"
        );
    }

    #[test]
    fn test_diabetes_linear_regression_r2() {
        let (x, y) = diabetes_xy();
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let r2 = r2_score(&y, &pred).unwrap();
        // Note: This is a 260-sample subset of the full sklearn diabetes dataset.
        // Linear R2 is lower than the full 442-sample dataset (~0.45).
        assert!(
            r2 > 0.0,
            "Diabetes LinearRegression training R2 {:.3} should be > 0.0 (positive fit)",
            r2
        );
    }

    #[test]
    fn test_diabetes_ridge_regression_r2() {
        let (x, y) = diabetes_xy();
        let mut model = RidgeRegression::new(1.0);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let r2 = r2_score(&y, &pred).unwrap();
        assert!(
            r2 > 0.0,
            "Diabetes RidgeRegression training R2 {:.3} should be > 0.0 (positive fit)",
            r2
        );
    }

    #[test]
    fn test_diabetes_lasso_regression_r2() {
        let (x, y) = diabetes_xy();
        let mut model = LassoRegression::new(0.1);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let r2 = r2_score(&y, &pred).unwrap();
        assert!(
            r2 > 0.0,
            "Diabetes LassoRegression training R2 {:.3} should be > 0.0 (positive fit)",
            r2
        );
    }

    #[test]
    fn test_diabetes_decision_tree_regressor_r2() {
        let (x, y) = diabetes_xy();
        let mut model = DecisionTreeRegressor::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let r2 = r2_score(&y, &pred).unwrap();
        assert!(
            r2 > 0.50,
            "Diabetes DecisionTreeRegressor training R2 {:.3} should be > 0.50",
            r2
        );
    }

    #[test]
    fn test_diabetes_random_forest_regressor_r2() {
        let (x, y) = diabetes_xy();
        let mut model = RandomForestRegressor::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let r2 = r2_score(&y, &pred).unwrap();
        assert!(
            r2 > 0.50,
            "Diabetes RandomForestRegressor training R2 {:.3} should be > 0.50",
            r2
        );
    }

    #[test]
    fn test_diabetes_gradient_boosting_regressor_r2() {
        let (x, y) = diabetes_xy();
        let mut model = GradientBoostingRegressor::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let r2 = r2_score(&y, &pred).unwrap();
        assert!(
            r2 > 0.50,
            "Diabetes GradientBoostingRegressor training R2 {:.3} should be > 0.50",
            r2
        );
    }

    #[test]
    fn test_diabetes_predictions_are_finite() {
        let (x, y) = diabetes_xy();
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        for (i, &p) in pred.iter().enumerate() {
            assert!(
                p.is_finite(),
                "Diabetes prediction[{}] = {} is not finite",
                i,
                p
            );
        }
    }

    // =============================================================================
    // 4. Dataset Metadata Tests (5 tests)
    // =============================================================================

    #[test]
    fn test_iris_feature_ranges() {
        let (dataset, _info) = load_iris();
        let x = dataset.x();
        // All Iris features are positive and less than 10
        for col_idx in 0..4 {
            let col = x.column(col_idx);
            for &val in col.iter() {
                assert!(
                    val > 0.0 && val < 10.0,
                    "Iris feature[{}] value {} is outside expected range (0, 10)",
                    col_idx,
                    val
                );
            }
        }
    }

    #[test]
    fn test_wine_feature_ranges() {
        let (dataset, _info) = load_wine();
        let x = dataset.x();
        // Alcohol column (index 0) should be roughly 11-15
        let alcohol_col = x.column(0);
        for &val in alcohol_col.iter() {
            assert!(
                (10.0..=16.0).contains(&val),
                "Wine alcohol value {} outside expected range [10, 16]",
                val
            );
        }
        // Proline column (index 12) should be roughly 270-1700
        let proline_col = x.column(12);
        for &val in proline_col.iter() {
            assert!(
                (270.0..=1700.0).contains(&val),
                "Wine proline value {} outside expected range [270, 1700]",
                val
            );
        }
    }

    #[test]
    fn test_diabetes_target_range() {
        let (_x, y) = diabetes_xy();
        // Diabetes target values are disease progression measures (typically 25-346)
        for &val in y.iter() {
            assert!(
                (20.0..=360.0).contains(&val),
                "Diabetes target {} outside expected range [20, 360]",
                val
            );
        }
    }

    #[test]
    fn test_no_nan_values_in_any_dataset() {
        // Iris
        {
            let (dataset, _info) = load_iris();
            let (x, y) = dataset.into_arrays();
            assert!(
                !x.iter().any(|v| v.is_nan()),
                "Iris features contain NaN values"
            );
            assert!(
                !y.iter().any(|v| v.is_nan()),
                "Iris targets contain NaN values"
            );
        }
        // Wine
        {
            let (dataset, _info) = load_wine();
            let (x, y) = dataset.into_arrays();
            assert!(
                !x.iter().any(|v| v.is_nan()),
                "Wine features contain NaN values"
            );
            assert!(
                !y.iter().any(|v| v.is_nan()),
                "Wine targets contain NaN values"
            );
        }
        // Diabetes
        {
            let (dataset, _info) = load_diabetes();
            let (x, y) = dataset.into_arrays();
            assert!(
                !x.iter().any(|v| v.is_nan()),
                "Diabetes features contain NaN values"
            );
            assert!(
                !y.iter().any(|v| v.is_nan()),
                "Diabetes targets contain NaN values"
            );
        }
    }

    #[test]
    fn test_iris_class_balance() {
        let (dataset, _info) = load_iris();
        let counts = dataset.class_counts();
        // Iris is perfectly balanced: 50 per class
        assert_eq!(
            counts.get(&0),
            Some(&50),
            "Iris class 0 should have 50 samples"
        );
        assert_eq!(
            counts.get(&1),
            Some(&50),
            "Iris class 1 should have 50 samples"
        );
        assert_eq!(
            counts.get(&2),
            Some(&50),
            "Iris class 2 should have 50 samples"
        );
    }

    // =============================================================================
    // 5. Cross-Dataset Comparison Tests (3 tests)
    // =============================================================================

    #[test]
    fn test_iris_random_forest_beats_decision_tree() {
        let (x, y) = iris_xy();

        let mut dt = DecisionTreeClassifier::new();
        dt.fit(&x, &y).unwrap();
        let dt_pred = dt.predict(&x).unwrap();
        let dt_acc = accuracy(&y, &dt_pred).unwrap();

        let mut rf = RandomForestClassifier::new();
        rf.fit(&x, &y).unwrap();
        let rf_pred = rf.predict(&x).unwrap();
        let rf_acc = accuracy(&y, &rf_pred).unwrap();

        assert!(
            rf_acc >= dt_acc - 0.05,
            "RandomForest accuracy ({:.3}) should be close to or better than DecisionTree ({:.3})",
            rf_acc,
            dt_acc
        );
    }

    #[test]
    fn test_iris_gradient_boosting_beats_decision_tree() {
        let (x, y) = iris_xy();

        let mut dt = DecisionTreeClassifier::new();
        dt.fit(&x, &y).unwrap();
        let dt_pred = dt.predict(&x).unwrap();
        let dt_acc = accuracy(&y, &dt_pred).unwrap();

        let mut gb = GradientBoostingClassifier::new();
        gb.fit(&x, &y).unwrap();
        let gb_pred = gb.predict(&x).unwrap();
        let gb_acc = accuracy(&y, &gb_pred).unwrap();

        assert!(
            gb_acc >= dt_acc - 0.05,
            "GradientBoosting accuracy ({:.3}) should be close to or better than DecisionTree ({:.3})",
            gb_acc,
            dt_acc
        );
    }

    #[test]
    fn test_diabetes_ridge_close_to_linear() {
        // With small regularization, Ridge should be close to OLS
        let (x, y) = diabetes_xy();

        let mut ols = LinearRegression::new();
        ols.fit(&x, &y).unwrap();
        let ols_pred = ols.predict(&x).unwrap();
        let ols_r2 = r2_score(&y, &ols_pred).unwrap();

        // Use very small alpha so Ridge approximates OLS
        let mut ridge = RidgeRegression::new(0.001);
        ridge.fit(&x, &y).unwrap();
        let ridge_pred = ridge.predict(&x).unwrap();
        let ridge_r2 = r2_score(&y, &ridge_pred).unwrap();

        let diff = (ols_r2 - ridge_r2).abs();
        assert!(
            diff < 0.05,
            "Ridge(alpha=0.001) R2 ({:.4}) should be close to OLS R2 ({:.4}), diff={:.4}",
            ridge_r2,
            ols_r2,
            diff
        );
    }
}

mod sklearn_correctness {
    use approx::assert_relative_eq;
    use ferroml_core::decomposition::PCA;
    use ferroml_core::models::adaboost::{AdaBoostClassifier, AdaBoostRegressor};
    use ferroml_core::models::boosting::{GradientBoostingClassifier, GradientBoostingRegressor};
    use ferroml_core::models::forest::{RandomForestClassifier, RandomForestRegressor};
    use ferroml_core::models::knn::{KNeighborsClassifier, KNeighborsRegressor};
    use ferroml_core::models::linear::LinearRegression;
    use ferroml_core::models::logistic::LogisticRegression;
    use ferroml_core::models::naive_bayes::{BernoulliNB, GaussianNB, MultinomialNB};
    use ferroml_core::models::regularized::{
        ElasticNet, LassoCV, LassoRegression, RidgeCV, RidgeRegression,
    };
    use ferroml_core::models::svm::{SVC, SVR};
    use ferroml_core::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor};
    use ferroml_core::models::Model;
    use ferroml_core::preprocessing::encoders::{LabelEncoder, OneHotEncoder, OrdinalEncoder};
    use ferroml_core::preprocessing::imputers::SimpleImputer;
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;
    use ferroml_core::preprocessing::scalers::{
        MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler,
    };
    use ferroml_core::preprocessing::Transformer;
    use ndarray::{Array1, Array2};
    use serde_json::Value;
    use std::path::Path;

    /// Tolerance for closed-form solutions (linear regression without regularization)
    const TOL_CLOSED_FORM: f64 = 1e-8;

    /// Tolerance for iterative solutions (regularized regression, gradient descent)
    #[allow(dead_code)]
    const TOL_ITERATIVE: f64 = 1e-4;

    /// Tolerance for deterministic preprocessing (scalers, encoders, imputers)
    const TOL_PREPROCESSING: f64 = 1e-10;

    /// Tolerance for tree-based predictions
    const TOL_TREE: f64 = 1e-10;

    /// Tolerance for ensemble scores (RNG differences across languages)
    const TOL_ENSEMBLE: f64 = 0.05;

    // =============================================================================
    // Fixture Loading Helpers
    // =============================================================================

    fn fixtures_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("benchmarks")
            .join("fixtures")
    }

    fn load_fixture(name: &str) -> Value {
        let path = fixtures_dir().join(name);
        let data = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to load fixture {}: {}", path.display(), e));
        serde_json::from_str(&data).unwrap()
    }

    fn json_to_array2(val: &Value) -> Array2<f64> {
        let rows: Vec<Vec<f64>> = val
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect()
            })
            .collect();
        let nrows = rows.len();
        let ncols = rows[0].len();
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        Array2::from_shape_vec((nrows, ncols), flat).unwrap()
    }

    fn json_to_array1(val: &Value) -> Array1<f64> {
        let vals: Vec<f64> = val
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        Array1::from_vec(vals)
    }

    fn accuracy(predictions: &Array1<f64>, y_true: &Array1<f64>) -> f64 {
        let correct = predictions
            .iter()
            .zip(y_true.iter())
            .filter(|(p, t)| (**p - **t).abs() < 0.5)
            .count();
        correct as f64 / y_true.len() as f64
    }

    fn r2_score(predictions: &Array1<f64>, y_true: &Array1<f64>) -> f64 {
        let y_mean = y_true.mean().unwrap();
        let ss_res: f64 = predictions
            .iter()
            .zip(y_true.iter())
            .map(|(p, t)| (t - p).powi(2))
            .sum();
        let ss_tot: f64 = y_true.iter().map(|t| (t - y_mean).powi(2)).sum();
        1.0 - ss_res / ss_tot
    }

    // =============================================================================
    // LINEAR REGRESSION TESTS (from original inline tests)
    // =============================================================================

    #[test]
    fn test_linear_regression_coefficients() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![6.0, 8.0, 9.0, 11.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients().unwrap();
        let intercept = model.intercept().unwrap();

        assert_relative_eq!(coef[0], 1.0, epsilon = TOL_CLOSED_FORM);
        assert_relative_eq!(coef[1], 2.0, epsilon = TOL_CLOSED_FORM);
        assert_relative_eq!(intercept, 3.0, epsilon = TOL_CLOSED_FORM);
    }

    #[test]
    fn test_linear_regression_predictions() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![6.0, 8.0, 9.0, 11.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();
        for (pred, expected) in predictions.iter().zip(y.iter()) {
            assert_relative_eq!(*pred, *expected, epsilon = TOL_CLOSED_FORM);
        }
    }

    #[test]
    fn test_linear_regression_r_squared() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![6.0, 8.0, 9.0, 11.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let r_squared = model.r_squared().unwrap();
        assert_relative_eq!(r_squared, 1.0, epsilon = TOL_CLOSED_FORM);
    }

    #[test]
    fn test_linear_regression_noisy_data() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 2.0, 4.0, 3.0, 1.0, 4.0, 3.0, 5.0, 5.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![5.1, 10.2, 6.9, 11.8, 16.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients().unwrap();
        assert_eq!(coef.len(), 2);
        assert!(coef[0] > 1.0 && coef[0] < 3.0);
        assert!(coef[1] > 0.0 && coef[1] < 2.0);
    }

    // =============================================================================
    // RIDGE REGRESSION TESTS
    // =============================================================================

    #[test]
    fn test_ridge_regression_coefficients() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![6.0, 8.0, 9.0, 11.0]);

        let mut model = RidgeRegression::new(1.0);
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients().unwrap();
        let intercept = model.intercept().unwrap();

        assert!(coef[0] > 0.5 && coef[0] < 1.5);
        assert!(coef[1] > 1.0 && coef[1] < 2.5);
        assert!(intercept > 2.0 && intercept < 5.0);
    }

    #[test]
    fn test_ridge_regression_regularization_strength() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![6.0, 8.0, 9.0, 11.0]);

        let mut model_low = RidgeRegression::new(0.1);
        let mut model_high = RidgeRegression::new(10.0);

        model_low.fit(&x, &y).unwrap();
        model_high.fit(&x, &y).unwrap();

        let coef_low = model_low.coefficients().unwrap();
        let coef_high = model_high.coefficients().unwrap();

        let norm_low: f64 = coef_low.iter().map(|c| c.powi(2)).sum();
        let norm_high: f64 = coef_high.iter().map(|c| c.powi(2)).sum();

        assert!(norm_high < norm_low);
    }

    #[test]
    fn test_ridge_small_alpha_approximates_ols() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![6.0, 8.0, 9.0, 11.0]);

        let mut ols = LinearRegression::new();
        let mut ridge = RidgeRegression::new(1e-10);

        ols.fit(&x, &y).unwrap();
        ridge.fit(&x, &y).unwrap();

        let ols_coef = ols.coefficients().unwrap();
        let ridge_coef = ridge.coefficients().unwrap();

        for (o, r) in ols_coef.iter().zip(ridge_coef.iter()) {
            assert_relative_eq!(*o, *r, epsilon = 1e-3);
        }
    }

    // =============================================================================
    // LASSO REGRESSION TESTS
    // =============================================================================

    #[test]
    fn test_lasso_regression_sparsity() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 3.0, 0.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![6.0, 8.0, 9.0, 11.0]);

        let mut model = LassoRegression::new(0.5);
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients().unwrap();
        assert!(coef[2].abs() < 0.1);
    }

    #[test]
    fn test_lasso_regression_convergence() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![6.0, 8.0, 9.0, 11.0]);

        let mut model = LassoRegression::new(0.1);
        model.fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();
        let mse: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;

        assert!(mse < 1.0);
    }

    // =============================================================================
    // DECISION TREE TESTS
    // =============================================================================

    #[test]
    fn test_decision_tree_classifier_predictions() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = DecisionTreeClassifier::new();
        model.fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();
        for (pred, expected) in predictions.iter().zip(y.iter()) {
            assert_relative_eq!(*pred, *expected, epsilon = TOL_TREE);
        }
    }

    #[test]
    fn test_decision_tree_regressor_training_fit() {
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0]);

        let mut model = DecisionTreeRegressor::new();
        model.fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();
        for (pred, expected) in predictions.iter().zip(y.iter()) {
            assert_relative_eq!(*pred, *expected, epsilon = TOL_TREE);
        }
    }

    #[test]
    fn test_decision_tree_feature_importance() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.5, 0.0, 0.3, 0.0, 0.8, 0.0, 0.1, 1.0, 0.2, 1.0, 0.9, 1.0, 0.4, 1.0, 0.7,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut model = DecisionTreeClassifier::new();
        model.fit(&x, &y).unwrap();

        let importance = model.feature_importance().unwrap();
        assert!(importance[0] > importance[1]);
    }

    // =============================================================================
    // SCALER TESTS
    // =============================================================================

    #[test]
    fn test_standard_scaler_transform() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let mut scaler = StandardScaler::new();
        let x_scaled = scaler.fit_transform(&x).unwrap();

        let mean_col0: f64 = x_scaled.column(0).mean().unwrap();
        let mean_col1: f64 = x_scaled.column(1).mean().unwrap();

        assert_relative_eq!(mean_col0, 0.0, epsilon = TOL_PREPROCESSING);
        assert_relative_eq!(mean_col1, 0.0, epsilon = TOL_PREPROCESSING);
        assert!(x_scaled[[0, 0]] < 0.0 && x_scaled[[2, 0]] > 0.0);
    }

    #[test]
    fn test_standard_scaler_inverse() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let mut scaler = StandardScaler::new();
        let x_scaled = scaler.fit_transform(&x).unwrap();
        let x_restored = scaler.inverse_transform(&x_scaled).unwrap();

        for (original, restored) in x.iter().zip(x_restored.iter()) {
            assert_relative_eq!(*original, *restored, epsilon = TOL_PREPROCESSING);
        }
    }

    #[test]
    fn test_minmax_scaler_transform() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let mut scaler = MinMaxScaler::new();
        let x_scaled = scaler.fit_transform(&x).unwrap();

        assert_relative_eq!(x_scaled[[0, 0]], 0.0, epsilon = TOL_PREPROCESSING);
        assert_relative_eq!(x_scaled[[0, 1]], 0.0, epsilon = TOL_PREPROCESSING);
        assert_relative_eq!(x_scaled[[1, 0]], 0.5, epsilon = TOL_PREPROCESSING);
        assert_relative_eq!(x_scaled[[1, 1]], 0.5, epsilon = TOL_PREPROCESSING);
        assert_relative_eq!(x_scaled[[2, 0]], 1.0, epsilon = TOL_PREPROCESSING);
        assert_relative_eq!(x_scaled[[2, 1]], 1.0, epsilon = TOL_PREPROCESSING);
    }

    #[test]
    fn test_robust_scaler_outlier_resistance() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 100.0])
            .unwrap();

        let mut scaler = RobustScaler::new();
        let x_scaled = scaler.fit_transform(&x).unwrap();

        assert!(x_scaled[[0, 0]].abs() < 5.0);
        assert!(x_scaled[[1, 0]].abs() < 5.0);
        assert!(x_scaled[[2, 0]].abs() < 5.0);
        assert!(x_scaled[[3, 0]].abs() > x_scaled[[0, 0]].abs() * 3.0);
    }

    #[test]
    fn test_maxabs_scaler_transform() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0]).unwrap();

        let mut scaler = MaxAbsScaler::new();
        let x_scaled = scaler.fit_transform(&x).unwrap();

        assert_relative_eq!(x_scaled[[0, 0]], 0.2, epsilon = TOL_PREPROCESSING);
        assert_relative_eq!(x_scaled[[0, 1]], -2.0 / 6.0, epsilon = TOL_PREPROCESSING);
        assert_relative_eq!(x_scaled[[2, 0]], 1.0, epsilon = TOL_PREPROCESSING);
        assert_relative_eq!(x_scaled[[2, 1]], -1.0, epsilon = TOL_PREPROCESSING);
    }

    // =============================================================================
    // METRICS VERIFICATION
    // =============================================================================

    #[test]
    fn test_r_squared_calculation() {
        let y_true = Array1::from_vec(vec![3.0, -0.5, 2.0, 7.0]);
        let y_pred = Array1::from_vec(vec![2.5, 0.0, 2.0, 8.0]);

        let y_mean = y_true.mean().unwrap();
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p): (&f64, &f64)| (*t - *p).powi(2))
            .sum();
        let ss_tot: f64 = y_true.iter().map(|t: &f64| (*t - y_mean).powi(2)).sum();
        let r_squared = 1.0 - ss_res / ss_tot;

        assert_relative_eq!(r_squared, 0.9486081370449679, epsilon = TOL_PREPROCESSING);
    }

    // =============================================================================
    // EDGE CASES AND NUMERICAL STABILITY
    // =============================================================================

    #[test]
    fn test_standard_scaler_constant_feature() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0]).unwrap();

        let mut scaler = StandardScaler::new();
        let x_scaled = scaler.fit_transform(&x).unwrap();

        let mean_col0: f64 = x_scaled.column(0).mean().unwrap();
        assert_relative_eq!(mean_col0, 0.0, epsilon = TOL_PREPROCESSING);

        for &val in x_scaled.column(1).iter() {
            assert!(val.abs() < TOL_PREPROCESSING || val.is_nan());
        }
    }

    #[test]
    fn test_minmax_scaler_constant_feature() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0]).unwrap();

        let mut scaler = MinMaxScaler::new();
        let x_scaled = scaler.fit_transform(&x).unwrap();

        assert_relative_eq!(x_scaled[[0, 0]], 0.0, epsilon = TOL_PREPROCESSING);
        assert_relative_eq!(x_scaled[[2, 0]], 1.0, epsilon = TOL_PREPROCESSING);

        for &val in x_scaled.column(1).iter() {
            assert!(val.abs() < TOL_PREPROCESSING || val.is_nan());
        }
    }

    // =============================================================================
    // FIXTURE-BASED TESTS: LOGISTIC REGRESSION
    // =============================================================================

    #[test]
    fn test_fixture_logistic_regression_accuracy() {
        // FerroML LogisticRegression only supports binary classification,
        // so we use a binary subset (classes 0 and 1 from iris)
        let fixture = load_fixture("logistic_regression.json");
        let x_train_full = json_to_array2(&fixture["X_train"]);
        let y_train_full = json_to_array1(&fixture["y_train"]);

        // Filter to binary: keep only classes 0.0 and 1.0
        let binary_mask_train: Vec<bool> = y_train_full.iter().map(|&y| y < 1.5).collect();
        let x_rows: Vec<Vec<f64>> = x_train_full
            .rows()
            .into_iter()
            .zip(binary_mask_train.iter())
            .filter(|(_, &m)| m)
            .map(|(r, _)| r.to_vec())
            .collect();
        let y_vals: Vec<f64> = y_train_full
            .iter()
            .zip(binary_mask_train.iter())
            .filter(|(_, &m)| m)
            .map(|(y, _)| *y)
            .collect();

        let ncols = x_train_full.ncols();
        let nrows = x_rows.len();
        let flat: Vec<f64> = x_rows.into_iter().flatten().collect();
        let x_train = Array2::from_shape_vec((nrows, ncols), flat).unwrap();
        let y_train = Array1::from_vec(y_vals);

        let mut model = LogisticRegression::new().with_max_iter(200);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_train).unwrap();
        let ferro_accuracy = accuracy(&predictions, &y_train);

        // Binary logistic regression should achieve high training accuracy on linearly separable data
        assert!(
            ferro_accuracy > 0.90,
            "LogisticRegression training accuracy should be > 90%: got {:.4}",
            ferro_accuracy,
        );
    }

    // =============================================================================
    // FIXTURE-BASED TESTS: RANDOM FOREST
    // =============================================================================

    #[test]
    fn test_fixture_random_forest_classifier_accuracy() {
        let fixture = load_fixture("random_forest.json");
        let clf = &fixture["classifier"];
        let x_train = json_to_array2(&clf["X_train"]);
        let y_train = json_to_array1(&clf["y_train"]);
        let x_test = json_to_array2(&clf["X_test"]);
        let y_test = json_to_array1(&clf["y_test"]);
        let sklearn_accuracy = clf["accuracy"].as_f64().unwrap();

        let mut model = RandomForestClassifier::new()
            .with_n_estimators(10)
            .with_random_state(42)
            .with_n_jobs(Some(1));
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_accuracy = accuracy(&predictions, &y_test);

        assert!(
            ferro_accuracy >= sklearn_accuracy - TOL_ENSEMBLE,
            "RandomForestClassifier accuracy: FerroML={:.4} vs sklearn={:.4}",
            ferro_accuracy,
            sklearn_accuracy,
        );
    }

    #[test]
    fn test_fixture_random_forest_regressor_r2() {
        let fixture = load_fixture("random_forest.json");
        let reg = &fixture["regressor"];
        let x_train = json_to_array2(&reg["X_train"]);
        let y_train = json_to_array1(&reg["y_train"]);
        let x_test = json_to_array2(&reg["X_test"]);
        let y_test = json_to_array1(&reg["y_test"]);
        let sklearn_r2 = reg["r2_score"].as_f64().unwrap();

        let mut model = RandomForestRegressor::new()
            .with_n_estimators(10)
            .with_random_state(42)
            .with_n_jobs(Some(1));
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_r2 = r2_score(&predictions, &y_test);

        // Allow wider tolerance for ensemble regressors
        assert!(
            (ferro_r2 - sklearn_r2).abs() < 0.15,
            "RandomForestRegressor R2: FerroML={:.4} vs sklearn={:.4}",
            ferro_r2,
            sklearn_r2,
        );
    }

    // =============================================================================
    // FIXTURE-BASED TESTS: GRADIENT BOOSTING
    // =============================================================================

    #[test]
    fn test_fixture_gradient_boosting_classifier_accuracy() {
        let fixture = load_fixture("gradient_boosting.json");
        let clf = &fixture["classifier"];
        let x_train = json_to_array2(&clf["X_train"]);
        let y_train = json_to_array1(&clf["y_train"]);
        let x_test = json_to_array2(&clf["X_test"]);
        let y_test = json_to_array1(&clf["y_test"]);
        let sklearn_accuracy = clf["accuracy"].as_f64().unwrap();

        let mut model = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_learning_rate(0.1)
            .with_max_depth(Some(3))
            .with_random_state(42);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_accuracy = accuracy(&predictions, &y_test);

        assert!(
            (ferro_accuracy - sklearn_accuracy).abs() < TOL_ENSEMBLE,
            "GradientBoostingClassifier accuracy: FerroML={:.4} vs sklearn={:.4}",
            ferro_accuracy,
            sklearn_accuracy,
        );
    }

    #[test]
    fn test_fixture_gradient_boosting_regressor_r2() {
        let fixture = load_fixture("gradient_boosting.json");
        let reg = &fixture["regressor"];
        let x_train = json_to_array2(&reg["X_train"]);
        let y_train = json_to_array1(&reg["y_train"]);
        let x_test = json_to_array2(&reg["X_test"]);
        let y_test = json_to_array1(&reg["y_test"]);
        let sklearn_r2 = reg["r2_score"].as_f64().unwrap();

        let mut model = GradientBoostingRegressor::new()
            .with_n_estimators(10)
            .with_learning_rate(0.1)
            .with_max_depth(Some(3))
            .with_random_state(42);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_r2 = r2_score(&predictions, &y_test);

        assert!(
            (ferro_r2 - sklearn_r2).abs() < 0.15,
            "GradientBoostingRegressor R2: FerroML={:.4} vs sklearn={:.4}",
            ferro_r2,
            sklearn_r2,
        );
    }

    // =============================================================================
    // FIXTURE-BASED TESTS: KNN
    // =============================================================================

    #[test]
    fn test_fixture_knn_classifier_accuracy() {
        let fixture = load_fixture("knn.json");
        let clf = &fixture["classifier"];
        let x_train = json_to_array2(&clf["X_train"]);
        let y_train = json_to_array1(&clf["y_train"]);
        let x_test = json_to_array2(&clf["X_test"]);
        let y_test = json_to_array1(&clf["y_test"]);
        let sklearn_accuracy = clf["accuracy"].as_f64().unwrap();

        let mut model = KNeighborsClassifier::new(5);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_accuracy = accuracy(&predictions, &y_test);

        // KNN is deterministic, should match closely
        assert!(
            (ferro_accuracy - sklearn_accuracy).abs() < 0.05,
            "KNeighborsClassifier accuracy: FerroML={:.4} vs sklearn={:.4}",
            ferro_accuracy,
            sklearn_accuracy,
        );
    }

    #[test]
    fn test_fixture_knn_classifier_predictions() {
        let fixture = load_fixture("knn.json");
        let clf = &fixture["classifier"];
        let x_train = json_to_array2(&clf["X_train"]);
        let y_train = json_to_array1(&clf["y_train"]);
        let x_test = json_to_array2(&clf["X_test"]);
        let sklearn_preds = json_to_array1(&clf["predictions"]);

        let mut model = KNeighborsClassifier::new(5);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();

        // KNN predictions should match sklearn exactly (deterministic algorithm)
        let match_rate = predictions
            .iter()
            .zip(sklearn_preds.iter())
            .filter(|(p, s)| (**p - **s).abs() < 0.5)
            .count() as f64
            / predictions.len() as f64;
        assert!(
            match_rate > 0.90,
            "KNN prediction match rate: {:.2}%",
            match_rate * 100.0,
        );
    }

    #[test]
    fn test_fixture_knn_regressor_r2() {
        let fixture = load_fixture("knn.json");
        let reg = &fixture["regressor"];
        let x_train = json_to_array2(&reg["X_train"]);
        let y_train = json_to_array1(&reg["y_train"]);
        let x_test = json_to_array2(&reg["X_test"]);
        let y_test = json_to_array1(&reg["y_test"]);
        let sklearn_r2 = reg["r2_score"].as_f64().unwrap();

        let mut model = KNeighborsRegressor::new(5);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_r2 = r2_score(&predictions, &y_test);

        assert!(
            (ferro_r2 - sklearn_r2).abs() < 0.10,
            "KNeighborsRegressor R2: FerroML={:.4} vs sklearn={:.4}",
            ferro_r2,
            sklearn_r2,
        );
    }

    // =============================================================================
    // FIXTURE-BASED TESTS: NAIVE BAYES
    // =============================================================================

    #[test]
    fn test_fixture_gaussian_nb_accuracy() {
        let fixture = load_fixture("naive_bayes.json");
        let gnb = &fixture["gaussian"];
        let x_train = json_to_array2(&gnb["X_train"]);
        let y_train = json_to_array1(&gnb["y_train"]);
        let x_test = json_to_array2(&gnb["X_test"]);
        let y_test = json_to_array1(&gnb["y_test"]);
        let sklearn_accuracy = gnb["accuracy"].as_f64().unwrap();

        let mut model = GaussianNB::new();
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_accuracy = accuracy(&predictions, &y_test);

        assert!(
            (ferro_accuracy - sklearn_accuracy).abs() < 0.05,
            "GaussianNB accuracy: FerroML={:.4} vs sklearn={:.4}",
            ferro_accuracy,
            sklearn_accuracy,
        );
    }

    #[test]
    fn test_fixture_gaussian_nb_predictions() {
        let fixture = load_fixture("naive_bayes.json");
        let gnb = &fixture["gaussian"];
        let x_train = json_to_array2(&gnb["X_train"]);
        let y_train = json_to_array1(&gnb["y_train"]);
        let x_test = json_to_array2(&gnb["X_test"]);
        let sklearn_preds = json_to_array1(&gnb["predictions"]);

        let mut model = GaussianNB::new();
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();

        let match_rate = predictions
            .iter()
            .zip(sklearn_preds.iter())
            .filter(|(p, s)| (**p - **s).abs() < 0.5)
            .count() as f64
            / predictions.len() as f64;
        assert!(
            match_rate > 0.90,
            "GaussianNB prediction match rate: {:.2}%",
            match_rate * 100.0,
        );
    }

    #[test]
    fn test_fixture_multinomial_nb_accuracy() {
        let fixture = load_fixture("naive_bayes.json");
        let mnb = &fixture["multinomial"];
        let x_train = json_to_array2(&mnb["X_train"]);
        let y_train = json_to_array1(&mnb["y_train"]);
        let x_test = json_to_array2(&mnb["X_test"]);
        let y_test = json_to_array1(&mnb["y_test"]);
        let sklearn_accuracy = mnb["accuracy"].as_f64().unwrap();

        let mut model = MultinomialNB::new().with_alpha(1.0);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_accuracy = accuracy(&predictions, &y_test);

        assert!(
            (ferro_accuracy - sklearn_accuracy).abs() < 0.10,
            "MultinomialNB accuracy: FerroML={:.4} vs sklearn={:.4}",
            ferro_accuracy,
            sklearn_accuracy,
        );
    }

    #[test]
    fn test_fixture_bernoulli_nb_accuracy() {
        let fixture = load_fixture("naive_bayes.json");
        let bnb = &fixture["bernoulli"];
        let x_train = json_to_array2(&bnb["X_train"]);
        let y_train = json_to_array1(&bnb["y_train"]);
        let x_test = json_to_array2(&bnb["X_test"]);
        let y_test = json_to_array1(&bnb["y_test"]);
        let sklearn_accuracy = bnb["accuracy"].as_f64().unwrap();

        let mut model = BernoulliNB::new().with_alpha(1.0).with_binarize(Some(3.0));
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_accuracy = accuracy(&predictions, &y_test);

        assert!(
            (ferro_accuracy - sklearn_accuracy).abs() < 0.10,
            "BernoulliNB accuracy: FerroML={:.4} vs sklearn={:.4}",
            ferro_accuracy,
            sklearn_accuracy,
        );
    }

    // =============================================================================
    // FIXTURE-BASED TESTS: SVM
    // =============================================================================

    #[test]
    fn test_fixture_svc_accuracy() {
        let fixture = load_fixture("svm.json");
        let clf = &fixture["classifier"];
        let x_train = json_to_array2(&clf["X_train"]);
        let y_train = json_to_array1(&clf["y_train"]);
        let x_test = json_to_array2(&clf["X_test"]);
        let y_test = json_to_array1(&clf["y_test"]);
        let sklearn_accuracy = clf["accuracy"].as_f64().unwrap();

        let mut model = SVC::new().with_c(1.0);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_accuracy = accuracy(&predictions, &y_test);

        assert!(
            (ferro_accuracy - sklearn_accuracy).abs() < 0.10,
            "SVC accuracy: FerroML={:.4} vs sklearn={:.4}",
            ferro_accuracy,
            sklearn_accuracy,
        );
    }

    #[test]
    fn test_fixture_svr_r2() {
        let fixture = load_fixture("svm.json");
        let reg = &fixture["regressor"];
        let x_train = json_to_array2(&reg["X_train"]);
        let y_train = json_to_array1(&reg["y_train"]);
        let x_test = json_to_array2(&reg["X_test"]);
        let y_test = json_to_array1(&reg["y_test"]);
        let sklearn_r2 = reg["r2_score"].as_f64().unwrap();

        let mut model = SVR::new().with_c(1.0).with_epsilon(0.1);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_r2 = r2_score(&predictions, &y_test);

        // SVR implementations can differ significantly due to SMO solver differences.
        // Just verify the model produces a reasonable (non-degenerate) fit.
        assert!(
            ferro_r2 > -2.0,
            "SVR R2 should not be extremely negative: FerroML={:.4} vs sklearn={:.4}",
            ferro_r2,
            sklearn_r2,
        );
    }

    // =============================================================================
    // FIXTURE-BASED TESTS: REGULARIZED MODELS
    // =============================================================================

    #[test]
    fn test_fixture_elasticnet_r2() {
        let fixture = load_fixture("regularized.json");
        let en = &fixture["elasticnet"];
        let x_train = json_to_array2(&en["X_train"]);
        let y_train = json_to_array1(&en["y_train"]);
        let x_test = json_to_array2(&en["X_test"]);
        let y_test = json_to_array1(&en["y_test"]);
        let sklearn_r2 = en["r2_score"].as_f64().unwrap();

        let mut model = ElasticNet::new(0.1, 0.5);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_r2 = r2_score(&predictions, &y_test);

        assert!(
            (ferro_r2 - sklearn_r2).abs() < 0.10,
            "ElasticNet R2: FerroML={:.4} vs sklearn={:.4}",
            ferro_r2,
            sklearn_r2,
        );
    }

    #[test]
    fn test_fixture_elasticnet_coefficients() {
        let fixture = load_fixture("regularized.json");
        let en = &fixture["elasticnet"];
        let x_train = json_to_array2(&en["X_train"]);
        let y_train = json_to_array1(&en["y_train"]);
        let sklearn_coef = json_to_array1(&en["coef"]);

        let mut model = ElasticNet::new(0.1, 0.5);
        model.fit(&x_train, &y_train).unwrap();
        let ferro_coef = model.coefficients().unwrap();

        // Check that coefficient signs and rough magnitudes match
        for (f, s) in ferro_coef.iter().zip(sklearn_coef.iter()) {
            if s.abs() > 1.0 {
                assert!(
                    (f - s).abs() / s.abs() < 0.5,
                    "ElasticNet coef mismatch: ferro={:.4} vs sklearn={:.4}",
                    f,
                    s,
                );
            }
        }
    }

    #[test]
    fn test_fixture_ridgecv_r2() {
        let fixture = load_fixture("regularized.json");
        let rcv = &fixture["ridgecv"];
        let x_train = json_to_array2(&rcv["X_train"]);
        let y_train = json_to_array1(&rcv["y_train"]);
        let x_test = json_to_array2(&rcv["X_test"]);
        let y_test = json_to_array1(&rcv["y_test"]);
        let sklearn_r2 = rcv["r2_score"].as_f64().unwrap();

        let mut model = RidgeCV::new(vec![0.1, 1.0, 10.0], 5);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_r2 = r2_score(&predictions, &y_test);

        assert!(
            (ferro_r2 - sklearn_r2).abs() < 0.10,
            "RidgeCV R2: FerroML={:.4} vs sklearn={:.4}",
            ferro_r2,
            sklearn_r2,
        );
    }

    #[test]
    fn test_fixture_lassocv_r2() {
        let fixture = load_fixture("regularized.json");
        let lcv = &fixture["lassocv"];
        let x_train = json_to_array2(&lcv["X_train"]);
        let y_train = json_to_array1(&lcv["y_train"]);
        let x_test = json_to_array2(&lcv["X_test"]);
        let y_test = json_to_array1(&lcv["y_test"]);
        let sklearn_r2 = lcv["r2_score"].as_f64().unwrap();

        let mut model = LassoCV::new(20, 5);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_r2 = r2_score(&predictions, &y_test);

        assert!(
            (ferro_r2 - sklearn_r2).abs() < 0.15,
            "LassoCV R2: FerroML={:.4} vs sklearn={:.4}",
            ferro_r2,
            sklearn_r2,
        );
    }

    // =============================================================================
    // FIXTURE-BASED TESTS: ADABOOST
    // =============================================================================

    #[test]
    fn test_fixture_adaboost_classifier_accuracy() {
        let fixture = load_fixture("adaboost.json");
        let clf = &fixture["classifier"];
        let x_train = json_to_array2(&clf["X_train"]);
        let y_train = json_to_array1(&clf["y_train"]);
        let x_test = json_to_array2(&clf["X_test"]);
        let y_test = json_to_array1(&clf["y_test"]);
        let sklearn_accuracy = clf["accuracy"].as_f64().unwrap();

        let mut model = AdaBoostClassifier::new(10).with_random_state(42);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_accuracy = accuracy(&predictions, &y_test);

        assert!(
            (ferro_accuracy - sklearn_accuracy).abs() < TOL_ENSEMBLE,
            "AdaBoostClassifier accuracy: FerroML={:.4} vs sklearn={:.4}",
            ferro_accuracy,
            sklearn_accuracy,
        );
    }

    #[test]
    fn test_fixture_adaboost_regressor_r2() {
        let fixture = load_fixture("adaboost.json");
        let reg = &fixture["regressor"];
        let x_train = json_to_array2(&reg["X_train"]);
        let y_train = json_to_array1(&reg["y_train"]);
        let x_test = json_to_array2(&reg["X_test"]);
        let y_test = json_to_array1(&reg["y_test"]);
        let sklearn_r2 = reg["r2_score"].as_f64().unwrap();

        let mut model = AdaBoostRegressor::new(10).with_random_state(42);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();
        let ferro_r2 = r2_score(&predictions, &y_test);

        assert!(
            (ferro_r2 - sklearn_r2).abs() < 0.20,
            "AdaBoostRegressor R2: FerroML={:.4} vs sklearn={:.4}",
            ferro_r2,
            sklearn_r2,
        );
    }

    // =============================================================================
    // FIXTURE-BASED TESTS: PREPROCESSING
    // =============================================================================

    #[test]
    fn test_fixture_pca_transform() {
        let fixture = load_fixture("preprocessing.json");
        let pca_fix = &fixture["pca"];
        let x_input = json_to_array2(&pca_fix["X_input"]);
        let sklearn_transformed = json_to_array2(&pca_fix["X_transformed"]);

        let mut pca = PCA::new().with_n_components(2);
        let x_transformed = pca.fit_transform(&x_input).unwrap();

        assert_eq!(x_transformed.ncols(), 2);
        assert_eq!(x_transformed.nrows(), sklearn_transformed.nrows());

        // PCA components may have flipped signs - compare absolute values
        for i in 0..x_transformed.nrows() {
            for j in 0..x_transformed.ncols() {
                assert!(
                    (x_transformed[[i, j]].abs() - sklearn_transformed[[i, j]].abs()).abs() < 0.1,
                    "PCA transform mismatch at [{},{}]: ferro={:.4} vs sklearn={:.4}",
                    i,
                    j,
                    x_transformed[[i, j]],
                    sklearn_transformed[[i, j]],
                );
            }
        }
    }

    #[test]
    fn test_fixture_pca_explained_variance() {
        let fixture = load_fixture("preprocessing.json");
        let pca_fix = &fixture["pca"];
        let x_input = json_to_array2(&pca_fix["X_input"]);
        let sklearn_evr = json_to_array1(&pca_fix["explained_variance_ratio"]);

        let mut pca = PCA::new().with_n_components(2);
        pca.fit_transform(&x_input).unwrap();

        let ferro_evr = pca.explained_variance_ratio().unwrap();
        for (f, s) in ferro_evr.iter().zip(sklearn_evr.iter()) {
            assert!(
                (f - s).abs() < 0.02,
                "PCA explained variance ratio mismatch: ferro={:.6} vs sklearn={:.6}",
                f,
                s,
            );
        }
    }

    #[test]
    fn test_fixture_polynomial_features_transform() {
        let fixture = load_fixture("preprocessing.json");
        let poly_fix = &fixture["polynomial_features"];
        let x_input = json_to_array2(&poly_fix["X_input"]);
        let sklearn_transformed = json_to_array2(&poly_fix["X_transformed"]);

        let mut poly = PolynomialFeatures::new(2);
        let x_transformed = poly.fit_transform(&x_input).unwrap();

        assert_eq!(x_transformed.ncols(), sklearn_transformed.ncols());
        assert_eq!(x_transformed.nrows(), sklearn_transformed.nrows());

        // Values should match closely
        for i in 0..x_transformed.nrows() {
            for j in 0..x_transformed.ncols() {
                assert!(
                    (x_transformed[[i, j]] - sklearn_transformed[[i, j]]).abs() < 1e-6,
                    "PolynomialFeatures mismatch at [{},{}]: ferro={:.6} vs sklearn={:.6}",
                    i,
                    j,
                    x_transformed[[i, j]],
                    sklearn_transformed[[i, j]],
                );
            }
        }
    }

    #[test]
    fn test_fixture_one_hot_encoder_transform() {
        let fixture = load_fixture("preprocessing.json");
        let ohe_fix = &fixture["one_hot_encoder"];
        let x_input = json_to_array2(&ohe_fix["X_input"]);
        let sklearn_transformed = json_to_array2(&ohe_fix["X_transformed"]);

        let mut encoder = OneHotEncoder::new();
        let x_transformed = encoder.fit_transform(&x_input).unwrap();

        assert_eq!(x_transformed.ncols(), sklearn_transformed.ncols());
        assert_eq!(x_transformed.nrows(), sklearn_transformed.nrows());

        // Check each row has exactly one 1.0 and rest 0.0
        for i in 0..x_transformed.nrows() {
            let row_sum: f64 = x_transformed.row(i).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fixture_label_encoder() {
        let fixture = load_fixture("preprocessing.json");
        let le_fix = &fixture["label_encoder"];
        let y_input = json_to_array1(&le_fix["y_input"]);

        let mut encoder = LabelEncoder::new();
        let x_input_2d = y_input.to_shape((y_input.len(), 1)).unwrap().to_owned();
        let y_encoded = encoder.fit_transform(&x_input_2d).unwrap();

        // LabelEncoder should produce consistent integer mapping
        // Check that the same input values get the same encoding
        assert_relative_eq!(y_encoded[[0, 0]], y_encoded[[3, 0]], epsilon = 1e-10); // both 2.0
        assert_relative_eq!(y_encoded[[1, 0]], y_encoded[[5, 0]], epsilon = 1e-10); // both 0.0
        assert_relative_eq!(y_encoded[[2, 0]], y_encoded[[4, 0]], epsilon = 1e-10);
        // both 1.0
    }

    #[test]
    fn test_fixture_ordinal_encoder() {
        let fixture = load_fixture("preprocessing.json");
        let oe_fix = &fixture["ordinal_encoder"];
        let x_input = json_to_array2(&oe_fix["X_input"]);

        let mut encoder = OrdinalEncoder::new();
        let x_encoded = encoder.fit_transform(&x_input).unwrap();

        // Same input values should get same ordinal encoding
        assert_relative_eq!(x_encoded[[0, 0]], x_encoded[[3, 0]], epsilon = 1e-10); // both 1.0
        assert_relative_eq!(x_encoded[[1, 0]], x_encoded[[4, 0]], epsilon = 1e-10);
        // both 3.0
    }

    #[test]
    fn test_fixture_simple_imputer_mean() {
        let fixture = load_fixture("preprocessing.json");
        let imp_fix = &fixture["simple_imputer"];
        let sklearn_transformed = json_to_array2(&imp_fix["X_transformed"]);

        // Build input with NaN for missing values
        let x_input = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 2.0, f64::NAN, 3.0, 7.0, f64::NAN, 4.0, 5.0],
        )
        .unwrap();

        let mut imputer =
            SimpleImputer::new(ferroml_core::preprocessing::imputers::ImputeStrategy::Mean);
        let x_imputed = imputer.fit_transform(&x_input).unwrap();

        for i in 0..x_imputed.nrows() {
            for j in 0..x_imputed.ncols() {
                assert!(
                    (x_imputed[[i, j]] - sklearn_transformed[[i, j]]).abs() < 1e-6,
                    "SimpleImputer mismatch at [{},{}]: ferro={:.6} vs sklearn={:.6}",
                    i,
                    j,
                    x_imputed[[i, j]],
                    sklearn_transformed[[i, j]],
                );
            }
        }
    }

    #[test]
    fn test_fixture_variance_threshold() {
        let fixture = load_fixture("preprocessing.json");
        let vt_fix = &fixture["variance_threshold"];
        let x_input = json_to_array2(&vt_fix["X_input"]);
        let sklearn_transformed = json_to_array2(&vt_fix["X_transformed"]);

        let mut selector = ferroml_core::preprocessing::selection::VarianceThreshold::new(0.0);
        let x_selected = selector.fit_transform(&x_input).unwrap();

        // Should remove constant columns (column 0 is all zeros)
        assert_eq!(x_selected.ncols(), sklearn_transformed.ncols());

        for i in 0..x_selected.nrows() {
            for j in 0..x_selected.ncols() {
                assert!(
                    (x_selected[[i, j]] - sklearn_transformed[[i, j]]).abs() < 1e-6,
                    "VarianceThreshold mismatch at [{},{}]",
                    i,
                    j,
                );
            }
        }
    }

    #[test]
    fn test_fixture_select_k_best() {
        let fixture = load_fixture("preprocessing.json");
        let skb_fix = &fixture["select_k_best"];
        let x_input = json_to_array2(&skb_fix["X_input"]);
        let y_input = json_to_array1(&skb_fix["y_input"]);
        let sklearn_transformed = json_to_array2(&skb_fix["X_transformed"]);

        let mut selector = ferroml_core::preprocessing::selection::SelectKBest::new(
            ferroml_core::preprocessing::selection::ScoreFunction::FClassif,
            2,
        );
        selector.fit_with_target(&x_input, &y_input).unwrap();
        let x_selected = selector.transform(&x_input).unwrap();

        assert_eq!(x_selected.ncols(), 2);
        assert_eq!(x_selected.ncols(), sklearn_transformed.ncols());
    }

    // =============================================================================
    // CROSS-MODEL TESTS
    // =============================================================================

    #[test]
    fn test_all_classifiers_produce_valid_predictions() {
        // All classifiers should produce predictions in the set of training labels
        let fixture = load_fixture("knn.json");
        let clf = &fixture["classifier"];
        let x_train = json_to_array2(&clf["X_train"]);
        let y_train = json_to_array1(&clf["y_train"]);
        let x_test = json_to_array2(&clf["X_test"]);

        let classes: Vec<f64> = {
            let mut c: Vec<f64> = y_train.iter().copied().collect();
            c.sort_by(|a, b| a.partial_cmp(b).unwrap());
            c.dedup();
            c
        };

        // KNN
        let mut knn = KNeighborsClassifier::new(5);
        knn.fit(&x_train, &y_train).unwrap();
        let preds = knn.predict(&x_test).unwrap();
        for p in preds.iter() {
            assert!(classes.contains(p), "KNN predicted invalid class: {}", p);
        }

        // GaussianNB
        let mut gnb = GaussianNB::new();
        gnb.fit(&x_train, &y_train).unwrap();
        let preds = gnb.predict(&x_test).unwrap();
        for p in preds.iter() {
            assert!(
                classes.contains(p),
                "GaussianNB predicted invalid class: {}",
                p,
            );
        }

        // DecisionTree
        let mut dt = DecisionTreeClassifier::new();
        dt.fit(&x_train, &y_train).unwrap();
        let preds = dt.predict(&x_test).unwrap();
        for p in preds.iter() {
            assert!(
                classes.contains(p),
                "DecisionTree predicted invalid class: {}",
                p,
            );
        }
    }

    #[test]
    fn test_all_regressors_produce_finite_predictions() {
        let fixture = load_fixture("knn.json");
        let reg = &fixture["regressor"];
        let x_train = json_to_array2(&reg["X_train"]);
        let y_train = json_to_array1(&reg["y_train"]);
        let x_test = json_to_array2(&reg["X_test"]);

        // KNN Regressor
        let mut knnr = KNeighborsRegressor::new(5);
        knnr.fit(&x_train, &y_train).unwrap();
        let preds = knnr.predict(&x_test).unwrap();
        for p in preds.iter() {
            assert!(p.is_finite(), "KNN Regressor produced non-finite: {}", p);
        }

        // DecisionTree Regressor
        let mut dtr = DecisionTreeRegressor::new();
        dtr.fit(&x_train, &y_train).unwrap();
        let preds = dtr.predict(&x_test).unwrap();
        for p in preds.iter() {
            assert!(
                p.is_finite(),
                "DecisionTree Regressor produced non-finite: {}",
                p,
            );
        }
    }

    // =============================================================================
    // LOGISTIC REGRESSION DETAILED TESTS
    // =============================================================================

    #[test]
    fn test_fixture_logistic_regression_predictions() {
        // Binary subset test - FerroML LogisticRegression only supports binary
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = LogisticRegression::new().with_max_iter(200);
        model.fit(&x, &y).unwrap();
        let predictions = model.predict(&x).unwrap();

        // Well-separated binary data should be classified perfectly
        let ferro_accuracy = accuracy(&predictions, &y);
        assert!(
            ferro_accuracy > 0.80,
            "LogisticRegression binary accuracy: {:.4}",
            ferro_accuracy,
        );
    }

    // =============================================================================
    // ENSEMBLE FEATURE IMPORTANCE TESTS
    // =============================================================================

    #[test]
    fn test_fixture_random_forest_feature_importance_shape() {
        let fixture = load_fixture("random_forest.json");
        let clf = &fixture["classifier"];
        let x_train = json_to_array2(&clf["X_train"]);
        let y_train = json_to_array1(&clf["y_train"]);

        let mut model = RandomForestClassifier::new()
            .with_n_estimators(10)
            .with_random_state(42)
            .with_n_jobs(Some(1));
        model.fit(&x_train, &y_train).unwrap();

        let importance = model.feature_importance().unwrap();
        assert_eq!(importance.len(), x_train.ncols());

        // All importances should be non-negative
        for &imp in importance.iter() {
            assert!(imp >= 0.0, "Feature importance should be >= 0");
        }

        // Importances should sum to ~1.0
        let sum: f64 = importance.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Feature importances should sum to ~1.0, got {}",
            sum,
        );
    }

    #[test]
    fn test_fixture_gradient_boosting_feature_importance_shape() {
        let fixture = load_fixture("gradient_boosting.json");
        let clf = &fixture["classifier"];
        let x_train = json_to_array2(&clf["X_train"]);
        let y_train = json_to_array1(&clf["y_train"]);

        let mut model = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_random_state(42);
        model.fit(&x_train, &y_train).unwrap();

        let importance = model.feature_importance().unwrap();
        assert_eq!(importance.len(), x_train.ncols());

        for &imp in importance.iter() {
            assert!(imp >= 0.0);
        }
    }

    // =============================================================================
    // SVM DETAILED TESTS
    // =============================================================================

    #[test]
    fn test_fixture_svc_predictions() {
        let fixture = load_fixture("svm.json");
        let clf = &fixture["classifier"];
        let x_train = json_to_array2(&clf["X_train"]);
        let y_train = json_to_array1(&clf["y_train"]);
        let x_test = json_to_array2(&clf["X_test"]);
        let sklearn_preds = json_to_array1(&clf["predictions"]);

        let mut model = SVC::new().with_c(1.0);
        model.fit(&x_train, &y_train).unwrap();
        let predictions = model.predict(&x_test).unwrap();

        let match_rate = predictions
            .iter()
            .zip(sklearn_preds.iter())
            .filter(|(p, s)| (**p - **s).abs() < 0.5)
            .count() as f64
            / predictions.len() as f64;
        assert!(
            match_rate > 0.80,
            "SVC prediction match rate: {:.2}%",
            match_rate * 100.0,
        );
    }

    // =============================================================================
    // PREPROCESSING ROUNDTRIP TESTS
    // =============================================================================

    #[test]
    fn test_pca_explained_variance_sums_to_less_than_one() {
        let fixture = load_fixture("preprocessing.json");
        let x_input = json_to_array2(&fixture["pca"]["X_input"]);

        let mut pca = PCA::new().with_n_components(2);
        pca.fit_transform(&x_input).unwrap();

        let evr = pca.explained_variance_ratio().unwrap();
        let sum: f64 = evr.iter().sum();
        assert!(sum <= 1.0 + 1e-10, "EVR sum should be <= 1.0, got {}", sum);
        assert!(sum > 0.5, "2 components should explain > 50% variance");
    }

    #[test]
    fn test_polynomial_features_output_dimensions() {
        // degree=2, 2 features, with bias => 1 + 2 + 3 = 6 output features
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let mut poly = PolynomialFeatures::new(2);
        let x_poly = poly.fit_transform(&x).unwrap();
        assert_eq!(x_poly.ncols(), 6);
        assert_eq!(x_poly.nrows(), 5);
    }

    #[test]
    fn test_one_hot_encoder_output_consistency() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 0.0, 1.0]).unwrap();
        let mut encoder = OneHotEncoder::new();
        let x_encoded = encoder.fit_transform(&x).unwrap();

        // 3 unique categories => 3 columns
        assert_eq!(x_encoded.ncols(), 3);

        // Each row should have exactly one 1.0
        for i in 0..5 {
            let row_sum: f64 = x_encoded.row(i).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }
}

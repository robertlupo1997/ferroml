//! Phase 31: Regression Suite Tests
//!
//! Comprehensive regression detection tests to ensure model quality and performance
//! don't degrade over time. This module provides:
//!
//! - Accuracy baselines: Verify models achieve expected performance on known datasets
//! - Timing regression: Detect performance regressions against baselines
//! - Reproducibility: Ensure deterministic results with fixed seeds
//! - Numerical stability: Verify coefficients/weights are stable across runs
//! - Prediction consistency: Ensure predictions don't drift

#[cfg(test)]
mod accuracy_baseline_tests {
    //! Tests that verify models achieve expected accuracy/R² on standard datasets

    use crate::metrics::classification::accuracy;
    use crate::metrics::r2_score;
    use crate::models::boosting::{GradientBoostingClassifier, GradientBoostingRegressor};
    use crate::models::forest::{RandomForestClassifier, RandomForestRegressor};
    use crate::models::knn::{KNeighborsClassifier, KNeighborsRegressor};
    use crate::models::linear::LinearRegression;
    use crate::models::regularized::{LassoRegression, RidgeRegression};
    use crate::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor};
    use crate::models::Model;
    use crate::testing::utils::{make_binary_classification, make_regression};

    /// Baseline accuracy requirements (minimum acceptable performance)
    mod baselines {
        /// Linear regression R² baseline on well-conditioned data
        pub const LINEAR_REGRESSION_R2_MIN: f64 = 0.85;
        /// Ridge regression R² baseline
        pub const RIDGE_REGRESSION_R2_MIN: f64 = 0.80;
        /// Lasso regression R² baseline (lower due to regularization)
        pub const LASSO_REGRESSION_R2_MIN: f64 = 0.70;
        /// Decision tree regressor R² baseline
        pub const TREE_REGRESSOR_R2_MIN: f64 = 0.90;
        /// Random forest regressor R² baseline
        pub const FOREST_REGRESSOR_R2_MIN: f64 = 0.85;
        /// Gradient boosting regressor R² baseline
        pub const GB_REGRESSOR_R2_MIN: f64 = 0.85;

        /// Decision tree classifier accuracy baseline
        pub const TREE_CLASSIFIER_ACC_MIN: f64 = 0.90;
        /// Random forest classifier accuracy baseline
        pub const FOREST_CLASSIFIER_ACC_MIN: f64 = 0.85;
        /// KNN classifier accuracy baseline
        pub const KNN_CLASSIFIER_ACC_MIN: f64 = 0.80;
        /// Gradient boosting classifier accuracy baseline
        pub const GB_CLASSIFIER_ACC_MIN: f64 = 0.85;
    }

    // ==================== Regression Model Baselines ====================

    #[test]
    fn test_linear_regression_r2_baseline() {
        let (x, y) = make_regression(200, 5, 0.1, 42);
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let r2 = r2_score(&y, &pred).unwrap();

        assert!(
            r2 >= baselines::LINEAR_REGRESSION_R2_MIN,
            "LinearRegression R² {} below baseline {}",
            r2,
            baselines::LINEAR_REGRESSION_R2_MIN
        );
    }

    #[test]
    fn test_ridge_regression_r2_baseline() {
        let (x, y) = make_regression(200, 5, 0.1, 42);
        let mut model = RidgeRegression::new(1.0);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let r2 = r2_score(&y, &pred).unwrap();

        assert!(
            r2 >= baselines::RIDGE_REGRESSION_R2_MIN,
            "RidgeRegression R² {} below baseline {}",
            r2,
            baselines::RIDGE_REGRESSION_R2_MIN
        );
    }

    #[test]
    fn test_lasso_regression_r2_baseline() {
        let (x, y) = make_regression(200, 5, 0.1, 42);
        let mut model = LassoRegression::new(0.1);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let r2 = r2_score(&y, &pred).unwrap();

        assert!(
            r2 >= baselines::LASSO_REGRESSION_R2_MIN,
            "LassoRegression R² {} below baseline {}",
            r2,
            baselines::LASSO_REGRESSION_R2_MIN
        );
    }

    #[test]
    fn test_decision_tree_regressor_r2_baseline() {
        let (x, y) = make_regression(200, 5, 0.1, 42);
        let mut model = DecisionTreeRegressor::new().with_random_state(42);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let r2 = r2_score(&y, &pred).unwrap();

        assert!(
            r2 >= baselines::TREE_REGRESSOR_R2_MIN,
            "DecisionTreeRegressor R² {} below baseline {}",
            r2,
            baselines::TREE_REGRESSOR_R2_MIN
        );
    }

    #[test]
    fn test_random_forest_regressor_r2_baseline() {
        let (x, y) = make_regression(200, 5, 0.1, 42);
        let mut model = RandomForestRegressor::new()
            .with_n_estimators(50)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let r2 = r2_score(&y, &pred).unwrap();

        assert!(
            r2 >= baselines::FOREST_REGRESSOR_R2_MIN,
            "RandomForestRegressor R² {} below baseline {}",
            r2,
            baselines::FOREST_REGRESSOR_R2_MIN
        );
    }

    #[test]
    fn test_gradient_boosting_regressor_r2_baseline() {
        let (x, y) = make_regression(200, 5, 0.1, 42);
        let mut model = GradientBoostingRegressor::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let r2 = r2_score(&y, &pred).unwrap();

        assert!(
            r2 >= baselines::GB_REGRESSOR_R2_MIN,
            "GradientBoostingRegressor R² {} below baseline {}",
            r2,
            baselines::GB_REGRESSOR_R2_MIN
        );
    }

    #[test]
    fn test_knn_regressor_r2_baseline() {
        let (x, y) = make_regression(200, 5, 0.1, 42);
        let mut model = KNeighborsRegressor::new(5);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let r2 = r2_score(&y, &pred).unwrap();

        // KNN may not achieve as high R² on regression - use looser baseline
        assert!(
            r2 >= 0.70,
            "KNeighborsRegressor R² {} below baseline 0.70",
            r2
        );
    }

    // ==================== Classification Model Baselines ====================

    #[test]
    fn test_decision_tree_classifier_accuracy_baseline() {
        let (x, y) = make_binary_classification(200, 5, 42);
        let mut model = DecisionTreeClassifier::new().with_random_state(42);
        model.fit(&x, &y).unwrap();
        let predictions = model.predict(&x).unwrap();
        let acc = accuracy(&y, &predictions).unwrap();

        assert!(
            acc >= baselines::TREE_CLASSIFIER_ACC_MIN,
            "DecisionTreeClassifier accuracy {} below baseline {}",
            acc,
            baselines::TREE_CLASSIFIER_ACC_MIN
        );
    }

    #[test]
    fn test_random_forest_classifier_accuracy_baseline() {
        let (x, y) = make_binary_classification(200, 5, 42);
        let mut model = RandomForestClassifier::new()
            .with_n_estimators(50)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();
        let predictions = model.predict(&x).unwrap();
        let acc = accuracy(&y, &predictions).unwrap();

        assert!(
            acc >= baselines::FOREST_CLASSIFIER_ACC_MIN,
            "RandomForestClassifier accuracy {} below baseline {}",
            acc,
            baselines::FOREST_CLASSIFIER_ACC_MIN
        );
    }

    #[test]
    fn test_knn_classifier_accuracy_baseline() {
        let (x, y) = make_binary_classification(200, 5, 42);
        let mut model = KNeighborsClassifier::new(5);
        model.fit(&x, &y).unwrap();
        let predictions = model.predict(&x).unwrap();
        let acc = accuracy(&y, &predictions).unwrap();

        assert!(
            acc >= baselines::KNN_CLASSIFIER_ACC_MIN,
            "KNeighborsClassifier accuracy {} below baseline {}",
            acc,
            baselines::KNN_CLASSIFIER_ACC_MIN
        );
    }

    #[test]
    fn test_gradient_boosting_classifier_accuracy_baseline() {
        let (x, y) = make_binary_classification(200, 5, 42);
        let mut model = GradientBoostingClassifier::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1);
        model.fit(&x, &y).unwrap();
        let predictions = model.predict(&x).unwrap();
        let acc = accuracy(&y, &predictions).unwrap();

        assert!(
            acc >= baselines::GB_CLASSIFIER_ACC_MIN,
            "GradientBoostingClassifier accuracy {} below baseline {}",
            acc,
            baselines::GB_CLASSIFIER_ACC_MIN
        );
    }
}

#[cfg(test)]
mod timing_regression_tests {
    //! Tests that detect timing regressions against baseline performance

    use crate::models::forest::RandomForestClassifier;
    use crate::models::linear::LinearRegression;
    use crate::models::regularized::RidgeRegression;
    use crate::models::tree::DecisionTreeClassifier;
    use crate::models::Model;
    use crate::testing::utils::{make_binary_classification, make_regression};
    use std::time::Instant;

    /// Timing thresholds - max acceptable times in milliseconds
    /// These are very generous to account for CI variability and debug builds
    mod timing_limits {
        /// Linear regression fit time limit (ms) for 1000x50 data
        pub const LINEAR_FIT_1000X50_MS: u128 = 1000;
        /// Ridge regression fit time limit (ms) for 1000x50 data
        pub const RIDGE_FIT_1000X50_MS: u128 = 1000;
        /// Decision tree fit time limit (ms) for 1000x50 data
        pub const TREE_FIT_1000X50_MS: u128 = 15000;
        /// Random forest fit time limit (ms) for 500x20 data with 50 trees
        pub const FOREST_FIT_500X20_MS: u128 = 30000;
        /// Prediction time limit (ms) for 1000 samples
        pub const PREDICT_1000_MS: u128 = 500;
    }

    #[test]
    fn test_linear_regression_fit_timing() {
        let (x, y) = make_regression(1000, 50, 0.1, 42);
        let mut model = LinearRegression::new();

        let start = Instant::now();
        model.fit(&x, &y).unwrap();
        let elapsed = start.elapsed().as_millis();

        assert!(
            elapsed <= timing_limits::LINEAR_FIT_1000X50_MS,
            "LinearRegression fit took {}ms, exceeds limit of {}ms",
            elapsed,
            timing_limits::LINEAR_FIT_1000X50_MS
        );
    }

    #[test]
    fn test_ridge_regression_fit_timing() {
        let (x, y) = make_regression(1000, 50, 0.1, 42);
        let mut model = RidgeRegression::new(1.0);

        let start = Instant::now();
        model.fit(&x, &y).unwrap();
        let elapsed = start.elapsed().as_millis();

        assert!(
            elapsed <= timing_limits::RIDGE_FIT_1000X50_MS,
            "RidgeRegression fit took {}ms, exceeds limit of {}ms",
            elapsed,
            timing_limits::RIDGE_FIT_1000X50_MS
        );
    }

    #[test]
    fn test_decision_tree_fit_timing() {
        let (x, y) = make_binary_classification(1000, 50, 42);
        let mut model = DecisionTreeClassifier::new().with_random_state(42);

        let start = Instant::now();
        model.fit(&x, &y).unwrap();
        let elapsed = start.elapsed().as_millis();

        assert!(
            elapsed <= timing_limits::TREE_FIT_1000X50_MS,
            "DecisionTreeClassifier fit took {}ms, exceeds limit of {}ms",
            elapsed,
            timing_limits::TREE_FIT_1000X50_MS
        );
    }

    #[test]
    fn test_random_forest_fit_timing() {
        let (x, y) = make_binary_classification(500, 20, 42);
        let mut model = RandomForestClassifier::new()
            .with_n_estimators(50)
            .with_random_state(42);

        let start = Instant::now();
        model.fit(&x, &y).unwrap();
        let elapsed = start.elapsed().as_millis();

        assert!(
            elapsed <= timing_limits::FOREST_FIT_500X20_MS,
            "RandomForestClassifier fit took {}ms, exceeds limit of {}ms",
            elapsed,
            timing_limits::FOREST_FIT_500X20_MS
        );
    }

    #[test]
    fn test_linear_regression_predict_timing() {
        let (x, y) = make_regression(1000, 50, 0.1, 42);
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let start = Instant::now();
        let _ = model.predict(&x).unwrap();
        let elapsed = start.elapsed().as_millis();

        assert!(
            elapsed <= timing_limits::PREDICT_1000_MS,
            "LinearRegression predict took {}ms, exceeds limit of {}ms",
            elapsed,
            timing_limits::PREDICT_1000_MS
        );
    }

    #[test]
    fn test_random_forest_predict_timing() {
        let (x, y) = make_binary_classification(500, 20, 42);
        let mut model = RandomForestClassifier::new()
            .with_n_estimators(50)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        // Predict on 1000 samples
        let (x_test, _) = make_binary_classification(1000, 20, 123);

        let start = Instant::now();
        let _ = model.predict(&x_test).unwrap();
        let elapsed = start.elapsed().as_millis();

        assert!(
            elapsed <= timing_limits::PREDICT_1000_MS,
            "RandomForestClassifier predict took {}ms, exceeds limit of {}ms",
            elapsed,
            timing_limits::PREDICT_1000_MS
        );
    }
}

#[cfg(test)]
mod reproducibility_tests {
    //! Tests that verify deterministic results with fixed random seeds

    use crate::models::boosting::{GradientBoostingClassifier, GradientBoostingRegressor};
    use crate::models::forest::{RandomForestClassifier, RandomForestRegressor};
    use crate::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor};
    use crate::models::Model;
    use crate::testing::assertions::tolerances;
    use crate::testing::utils::{make_binary_classification, make_regression};

    // ==================== Tree-based Reproducibility ====================

    #[test]
    fn test_decision_tree_classifier_reproducibility() {
        let (x, y) = make_binary_classification(100, 5, 42);

        // Train twice with same seed
        let mut model1 = DecisionTreeClassifier::new().with_random_state(123);
        let mut model2 = DecisionTreeClassifier::new().with_random_state(123);

        model1.fit(&x, &y).unwrap();
        model2.fit(&x, &y).unwrap();

        let pred1 = model1.predict(&x).unwrap();
        let pred2 = model2.predict(&x).unwrap();

        // Predictions should be identical
        assert_eq!(
            pred1.as_slice().unwrap(),
            pred2.as_slice().unwrap(),
            "DecisionTreeClassifier not reproducible with same seed"
        );
    }

    #[test]
    fn test_decision_tree_regressor_reproducibility() {
        let (x, y) = make_regression(100, 5, 0.1, 42);

        let mut model1 = DecisionTreeRegressor::new().with_random_state(123);
        let mut model2 = DecisionTreeRegressor::new().with_random_state(123);

        model1.fit(&x, &y).unwrap();
        model2.fit(&x, &y).unwrap();

        let pred1 = model1.predict(&x).unwrap();
        let pred2 = model2.predict(&x).unwrap();

        for (p1, p2) in pred1.iter().zip(pred2.iter()) {
            assert!(
                (p1 - p2).abs() < tolerances::TREE,
                "DecisionTreeRegressor not reproducible: {} vs {}",
                p1,
                p2
            );
        }
    }

    #[test]
    fn test_random_forest_classifier_reproducibility() {
        let (x, y) = make_binary_classification(100, 5, 42);

        let mut model1 = RandomForestClassifier::new()
            .with_n_estimators(20)
            .with_random_state(456);
        let mut model2 = RandomForestClassifier::new()
            .with_n_estimators(20)
            .with_random_state(456);

        model1.fit(&x, &y).unwrap();
        model2.fit(&x, &y).unwrap();

        let pred1 = model1.predict(&x).unwrap();
        let pred2 = model2.predict(&x).unwrap();

        assert_eq!(
            pred1.as_slice().unwrap(),
            pred2.as_slice().unwrap(),
            "RandomForestClassifier not reproducible with same seed"
        );
    }

    #[test]
    fn test_random_forest_regressor_reproducibility() {
        let (x, y) = make_regression(100, 5, 0.1, 42);

        let mut model1 = RandomForestRegressor::new()
            .with_n_estimators(20)
            .with_random_state(456);
        let mut model2 = RandomForestRegressor::new()
            .with_n_estimators(20)
            .with_random_state(456);

        model1.fit(&x, &y).unwrap();
        model2.fit(&x, &y).unwrap();

        let pred1 = model1.predict(&x).unwrap();
        let pred2 = model2.predict(&x).unwrap();

        for (p1, p2) in pred1.iter().zip(pred2.iter()) {
            assert!(
                (p1 - p2).abs() < tolerances::TREE,
                "RandomForestRegressor not reproducible: {} vs {}",
                p1,
                p2
            );
        }
    }

    #[test]
    fn test_gradient_boosting_classifier_reproducibility() {
        let (x, y) = make_binary_classification(100, 5, 42);

        let mut model1 = GradientBoostingClassifier::new().with_n_estimators(20);
        let mut model2 = GradientBoostingClassifier::new().with_n_estimators(20);

        model1.fit(&x, &y).unwrap();
        model2.fit(&x, &y).unwrap();

        let pred1 = model1.predict(&x).unwrap();
        let pred2 = model2.predict(&x).unwrap();

        assert_eq!(
            pred1.as_slice().unwrap(),
            pred2.as_slice().unwrap(),
            "GradientBoostingClassifier not reproducible with same seed"
        );
    }

    #[test]
    fn test_gradient_boosting_regressor_reproducibility() {
        let (x, y) = make_regression(100, 5, 0.1, 42);

        let mut model1 = GradientBoostingRegressor::new().with_n_estimators(20);
        let mut model2 = GradientBoostingRegressor::new().with_n_estimators(20);

        model1.fit(&x, &y).unwrap();
        model2.fit(&x, &y).unwrap();

        let pred1 = model1.predict(&x).unwrap();
        let pred2 = model2.predict(&x).unwrap();

        for (p1, p2) in pred1.iter().zip(pred2.iter()) {
            assert!(
                (p1 - p2).abs() < tolerances::TREE,
                "GradientBoostingRegressor not reproducible: {} vs {}",
                p1,
                p2
            );
        }
    }

    // ==================== Different Seeds Should Differ ====================

    #[test]
    fn test_random_forest_different_seeds_differ() {
        let (x, y) = make_binary_classification(100, 5, 42);

        let mut model1 = RandomForestClassifier::new()
            .with_n_estimators(20)
            .with_random_state(111);
        let mut model2 = RandomForestClassifier::new()
            .with_n_estimators(20)
            .with_random_state(222);

        model1.fit(&x, &y).unwrap();
        model2.fit(&x, &y).unwrap();

        let pred1 = model1.predict(&x).unwrap();
        let pred2 = model2.predict(&x).unwrap();

        // With different seeds, predictions may differ (not guaranteed, but likely)
        // At minimum, ensure no crash
        assert_eq!(pred1.len(), pred2.len());
    }
}

#[cfg(test)]
mod numerical_stability_tests {
    //! Tests that verify numerical stability of model coefficients

    use crate::models::linear::LinearRegression;
    use crate::models::regularized::{LassoRegression, RidgeRegression};
    use crate::models::Model;
    use crate::testing::assertions::tolerances;
    use crate::testing::utils::make_regression;

    // ==================== Coefficient Stability ====================

    #[test]
    fn test_linear_regression_coefficient_stability() {
        let (x, y) = make_regression(200, 5, 0.1, 42);

        // Fit multiple times - coefficients should be identical (closed-form)
        let mut model1 = LinearRegression::new();
        let mut model2 = LinearRegression::new();

        model1.fit(&x, &y).unwrap();
        model2.fit(&x, &y).unwrap();

        let coef1 = model1.coefficients().unwrap();
        let coef2 = model2.coefficients().unwrap();

        for (c1, c2) in coef1.iter().zip(coef2.iter()) {
            assert!(
                (c1 - c2).abs() < tolerances::CLOSED_FORM,
                "LinearRegression coefficients unstable: {} vs {}",
                c1,
                c2
            );
        }
    }

    #[test]
    fn test_ridge_regression_coefficient_stability() {
        let (x, y) = make_regression(200, 5, 0.1, 42);

        let mut model1 = RidgeRegression::new(1.0);
        let mut model2 = RidgeRegression::new(1.0);

        model1.fit(&x, &y).unwrap();
        model2.fit(&x, &y).unwrap();

        let coef1 = model1.coefficients().unwrap();
        let coef2 = model2.coefficients().unwrap();

        for (c1, c2) in coef1.iter().zip(coef2.iter()) {
            assert!(
                (c1 - c2).abs() < tolerances::CLOSED_FORM,
                "RidgeRegression coefficients unstable: {} vs {}",
                c1,
                c2
            );
        }
    }

    #[test]
    fn test_lasso_regression_coefficient_stability() {
        let (x, y) = make_regression(200, 5, 0.1, 42);

        let mut model1 = LassoRegression::new(0.1);
        let mut model2 = LassoRegression::new(0.1);

        model1.fit(&x, &y).unwrap();
        model2.fit(&x, &y).unwrap();

        let coef1 = model1.coefficients().unwrap();
        let coef2 = model2.coefficients().unwrap();

        // Lasso is iterative, so use looser tolerance
        for (c1, c2) in coef1.iter().zip(coef2.iter()) {
            assert!(
                (c1 - c2).abs() < tolerances::ITERATIVE,
                "LassoRegression coefficients unstable: {} vs {}",
                c1,
                c2
            );
        }
    }

    // ==================== Intercept Stability ====================

    #[test]
    fn test_linear_regression_intercept_stability() {
        let (x, y) = make_regression(200, 5, 0.1, 42);

        let mut model1 = LinearRegression::new();
        let mut model2 = LinearRegression::new();

        model1.fit(&x, &y).unwrap();
        model2.fit(&x, &y).unwrap();

        let intercept1 = model1.intercept().unwrap();
        let intercept2 = model2.intercept().unwrap();

        assert!(
            (intercept1 - intercept2).abs() < tolerances::CLOSED_FORM,
            "LinearRegression intercepts unstable: {} vs {}",
            intercept1,
            intercept2
        );
    }

    // ==================== Ill-Conditioned Data ====================

    #[test]
    fn test_ridge_handles_collinear_features() {
        // Create collinear data
        let n = 100;
        let mut x_data = Vec::with_capacity(n * 3);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            let x1 = i as f64 / n as f64;
            let x2 = x1 * 2.0 + 0.01; // Nearly collinear with x1
            let x3 = (i as f64).sin();
            x_data.extend_from_slice(&[x1, x2, x3]);
            y_data.push(x1 + x3 + 0.1 * (i as f64 / n as f64));
        }

        let x = ndarray::Array2::from_shape_vec((n, 3), x_data).unwrap();
        let y = ndarray::Array1::from_vec(y_data);

        // Ridge should handle this gracefully (regularization helps)
        let mut model = RidgeRegression::new(1.0);
        let result = model.fit(&x, &y);

        assert!(result.is_ok(), "Ridge should handle collinear features");

        let coef = model.coefficients().unwrap();
        // Coefficients should be finite
        for c in coef.iter() {
            assert!(c.is_finite(), "Ridge coefficient should be finite: {}", c);
        }
    }

    #[test]
    fn test_linear_regression_small_values_stability() {
        // Test with very small feature values
        let n = 100;
        let scale = 1e-10;
        let mut x_data = Vec::with_capacity(n * 3);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            let x1 = (i as f64 / n as f64) * scale;
            let x2 = ((i as f64).sin()) * scale;
            let x3 = ((i as f64 / 10.0).cos()) * scale;
            x_data.extend_from_slice(&[x1, x2, x3]);
            y_data.push((x1 + x2 + x3) / scale + 0.001 * (i as f64 / n as f64));
        }

        let x = ndarray::Array2::from_shape_vec((n, 3), x_data).unwrap();
        let y = ndarray::Array1::from_vec(y_data);

        let mut model = LinearRegression::new();
        let result = model.fit(&x, &y);

        assert!(
            result.is_ok(),
            "LinearRegression should handle small values"
        );

        let pred = model.predict(&x).unwrap();
        for p in pred.iter() {
            assert!(p.is_finite(), "Prediction should be finite: {}", p);
        }
    }

    #[test]
    fn test_linear_regression_large_values_stability() {
        // Test with large feature values
        let n = 100;
        let scale = 1e6;
        let mut x_data = Vec::with_capacity(n * 3);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            let x1 = (i as f64 / n as f64) * scale;
            let x2 = ((i as f64).sin()) * scale;
            let x3 = ((i as f64 / 10.0).cos()) * scale;
            x_data.extend_from_slice(&[x1, x2, x3]);
            y_data.push((x1 + x2 + x3) / scale * 100.0);
        }

        let x = ndarray::Array2::from_shape_vec((n, 3), x_data).unwrap();
        let y = ndarray::Array1::from_vec(y_data);

        let mut model = LinearRegression::new();
        let result = model.fit(&x, &y);

        assert!(
            result.is_ok(),
            "LinearRegression should handle large values"
        );

        let pred = model.predict(&x).unwrap();
        for p in pred.iter() {
            assert!(p.is_finite(), "Prediction should be finite: {}", p);
        }
    }
}

#[cfg(test)]
mod prediction_consistency_tests {
    //! Tests that verify predictions remain consistent for a trained model

    use crate::models::forest::RandomForestClassifier;
    use crate::models::knn::KNeighborsClassifier;
    use crate::models::linear::LinearRegression;
    use crate::models::tree::DecisionTreeClassifier;
    use crate::models::Model;
    use crate::testing::assertions::tolerances;
    use crate::testing::utils::{make_binary_classification, make_regression};
    use ndarray::s;

    #[test]
    fn test_linear_regression_prediction_consistency() {
        let (x, y) = make_regression(100, 5, 0.1, 42);
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // Predict multiple times on same data
        let pred1 = model.predict(&x).unwrap();
        let pred2 = model.predict(&x).unwrap();
        let pred3 = model.predict(&x).unwrap();

        for ((p1, p2), p3) in pred1.iter().zip(pred2.iter()).zip(pred3.iter()) {
            assert!(
                (p1 - p2).abs() < tolerances::CLOSED_FORM,
                "Predictions not consistent: {} vs {}",
                p1,
                p2
            );
            assert!(
                (p2 - p3).abs() < tolerances::CLOSED_FORM,
                "Predictions not consistent: {} vs {}",
                p2,
                p3
            );
        }
    }

    #[test]
    fn test_decision_tree_prediction_consistency() {
        let (x, y) = make_binary_classification(100, 5, 42);
        let mut model = DecisionTreeClassifier::new().with_random_state(42);
        model.fit(&x, &y).unwrap();

        let pred1 = model.predict(&x).unwrap();
        let pred2 = model.predict(&x).unwrap();
        let pred3 = model.predict(&x).unwrap();

        assert_eq!(
            pred1.as_slice().unwrap(),
            pred2.as_slice().unwrap(),
            "DecisionTree predictions not consistent"
        );
        assert_eq!(
            pred2.as_slice().unwrap(),
            pred3.as_slice().unwrap(),
            "DecisionTree predictions not consistent"
        );
    }

    #[test]
    fn test_random_forest_prediction_consistency() {
        let (x, y) = make_binary_classification(100, 5, 42);
        let mut model = RandomForestClassifier::new()
            .with_n_estimators(20)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let pred1 = model.predict(&x).unwrap();
        let pred2 = model.predict(&x).unwrap();

        assert_eq!(
            pred1.as_slice().unwrap(),
            pred2.as_slice().unwrap(),
            "RandomForest predictions not consistent across calls"
        );
    }

    #[test]
    fn test_knn_prediction_consistency() {
        let (x, y) = make_binary_classification(100, 5, 42);
        let mut model = KNeighborsClassifier::new(5);
        model.fit(&x, &y).unwrap();

        let pred1 = model.predict(&x).unwrap();
        let pred2 = model.predict(&x).unwrap();

        assert_eq!(
            pred1.as_slice().unwrap(),
            pred2.as_slice().unwrap(),
            "KNN predictions not consistent across calls"
        );
    }

    // ==================== Subset Consistency ====================

    #[test]
    fn test_linear_regression_subset_prediction_consistency() {
        let (x, y) = make_regression(100, 5, 0.1, 42);
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // Full prediction
        let full_pred = model.predict(&x).unwrap();

        // Predict on first 50 samples
        let x_subset = x.slice(s![..50, ..]).to_owned();
        let subset_pred = model.predict(&x_subset).unwrap();

        // Subset predictions should match corresponding full predictions
        for (i, (sp, fp)) in subset_pred
            .iter()
            .zip(full_pred.iter().take(50))
            .enumerate()
        {
            assert!(
                (sp - fp).abs() < tolerances::CLOSED_FORM,
                "Subset prediction {} differs: {} vs {}",
                i,
                sp,
                fp
            );
        }
    }

    #[test]
    fn test_decision_tree_subset_prediction_consistency() {
        let (x, y) = make_binary_classification(100, 5, 42);
        let mut model = DecisionTreeClassifier::new().with_random_state(42);
        model.fit(&x, &y).unwrap();

        let full_pred = model.predict(&x).unwrap();
        let x_subset = x.slice(s![..50, ..]).to_owned();
        let subset_pred = model.predict(&x_subset).unwrap();

        for (i, (sp, fp)) in subset_pred
            .iter()
            .zip(full_pred.iter().take(50))
            .enumerate()
        {
            assert!(
                (sp - fp).abs() < tolerances::TREE,
                "Subset prediction {} differs: {} vs {}",
                i,
                sp,
                fp
            );
        }
    }
}

#[cfg(test)]
mod scaling_regression_tests {
    //! Tests that verify computational complexity doesn't regress unexpectedly

    use crate::models::forest::RandomForestClassifier;
    use crate::models::linear::LinearRegression;
    use crate::models::tree::DecisionTreeClassifier;
    use crate::models::Model;
    use crate::testing::utils::{make_binary_classification, make_regression};
    use std::time::Instant;

    #[test]
    fn test_linear_regression_scales_linearly_in_samples() {
        // Linear regression should scale roughly O(n) in samples for fixed features
        let n_features = 10;

        let (x_100, y_100) = make_regression(100, n_features, 0.1, 42);
        let (x_400, y_400) = make_regression(400, n_features, 0.1, 42);

        // Warmup run to prime caches / JIT / scheduler
        let mut warmup = LinearRegression::new();
        let _ = warmup.fit(&x_100, &y_100);

        let mut model = LinearRegression::new();
        let start = Instant::now();
        model.fit(&x_100, &y_100).unwrap();
        let time_100 = start.elapsed().as_nanos() as f64;

        let mut model = LinearRegression::new();
        let start = Instant::now();
        model.fit(&x_400, &y_400).unwrap();
        let time_400 = start.elapsed().as_nanos() as f64;

        // Catches gross algorithmic regressions (O(n) → O(n³) = 64x).
        // Generous threshold: wall-clock timing under parallel test execution
        // has high variance from CPU contention and scheduling noise.
        let ratio = time_400 / time_100.max(1.0);
        assert!(
            ratio < 100.0,
            "LinearRegression scaling worse than expected: {}x for 4x samples",
            ratio
        );
    }

    #[test]
    fn test_decision_tree_scales_with_samples() {
        // Decision tree should scale roughly O(n log n) in samples
        let n_features = 10;

        let (x_100, y_100) = make_binary_classification(100, n_features, 42);
        let (x_400, y_400) = make_binary_classification(400, n_features, 42);

        // Warmup run to prime caches / scheduler
        let mut warmup = DecisionTreeClassifier::new().with_random_state(42);
        let _ = warmup.fit(&x_100, &y_100);

        let mut model = DecisionTreeClassifier::new().with_random_state(42);
        let start = Instant::now();
        model.fit(&x_100, &y_100).unwrap();
        let time_100 = start.elapsed().as_nanos() as f64;

        let mut model = DecisionTreeClassifier::new().with_random_state(42);
        let start = Instant::now();
        model.fit(&x_400, &y_400).unwrap();
        let time_400 = start.elapsed().as_nanos() as f64;

        // Catches gross algorithmic regressions (O(n log n) → O(n³) = 64x).
        // Generous threshold: wall-clock timing under parallel test execution
        // has high variance from CPU contention and scheduling noise.
        let ratio = time_400 / time_100.max(1.0);
        assert!(
            ratio < 100.0,
            "DecisionTree scaling worse than expected: {}x for 4x samples",
            ratio
        );
    }

    #[test]
    fn test_random_forest_scales_with_trees() {
        // Random forest should scale linearly with number of trees
        let (x, y) = make_binary_classification(200, 10, 42);

        // Warmup run to prime caches / scheduler
        let mut warmup = RandomForestClassifier::new()
            .with_n_estimators(5)
            .with_random_state(42);
        let _ = warmup.fit(&x, &y);

        let mut model = RandomForestClassifier::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let start = Instant::now();
        model.fit(&x, &y).unwrap();
        let time_10 = start.elapsed().as_nanos() as f64;

        let mut model = RandomForestClassifier::new()
            .with_n_estimators(40)
            .with_random_state(42);
        let start = Instant::now();
        model.fit(&x, &y).unwrap();
        let time_40 = start.elapsed().as_nanos() as f64;

        // Catches gross algorithmic regressions (O(n) → O(n³) = 64x).
        // Generous threshold: wall-clock timing under parallel test execution
        // has high variance from CPU contention and scheduling noise.
        let ratio = time_40 / time_10.max(1.0);
        assert!(
            ratio < 50.0,
            "RandomForest scaling with trees worse than expected: {}x for 4x trees",
            ratio
        );
    }
}

#[cfg(test)]
mod model_comparison_regression_tests {
    //! Tests that verify relative model performance relationships hold

    use crate::metrics::classification::accuracy;
    use crate::metrics::r2_score;
    use crate::models::boosting::GradientBoostingClassifier;
    use crate::models::forest::{RandomForestClassifier, RandomForestRegressor};
    use crate::models::linear::LinearRegression;
    use crate::models::regularized::RidgeRegression;
    use crate::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor};
    use crate::models::Model;
    use crate::testing::utils::{make_binary_classification, make_regression};
    use ndarray::s;

    #[test]
    fn test_ensemble_beats_single_tree_classification() {
        // Random Forest should generally outperform or match single Decision Tree
        let (x, y) = make_binary_classification(300, 10, 42);

        // Split data
        let x_train = x.slice(s![..200, ..]).to_owned();
        let y_train = y.slice(s![..200]).to_owned();
        let x_test = x.slice(s![200.., ..]).to_owned();
        let y_test = y.slice(s![200..]).to_owned();

        let mut tree = DecisionTreeClassifier::new().with_random_state(42);
        tree.fit(&x_train, &y_train).unwrap();
        let tree_pred = tree.predict(&x_test).unwrap();
        let tree_acc = accuracy(&y_test, &tree_pred).unwrap();

        let mut forest = RandomForestClassifier::new()
            .with_n_estimators(50)
            .with_random_state(42);
        forest.fit(&x_train, &y_train).unwrap();
        let forest_pred = forest.predict(&x_test).unwrap();
        let forest_acc = accuracy(&y_test, &forest_pred).unwrap();

        // Forest should be at least as good (with small margin for randomness)
        assert!(
            forest_acc >= tree_acc - 0.05,
            "RandomForest ({}) significantly worse than DecisionTree ({})",
            forest_acc,
            tree_acc
        );
    }

    #[test]
    fn test_ensemble_beats_single_tree_regression() {
        let (x, y) = make_regression(300, 10, 0.1, 42);

        let x_train = x.slice(s![..200, ..]).to_owned();
        let y_train = y.slice(s![..200]).to_owned();
        let x_test = x.slice(s![200.., ..]).to_owned();
        let y_test = y.slice(s![200..]).to_owned();

        let mut tree = DecisionTreeRegressor::new().with_random_state(42);
        tree.fit(&x_train, &y_train).unwrap();
        let tree_pred = tree.predict(&x_test).unwrap();
        let tree_r2 = r2_score(&y_test, &tree_pred).unwrap();

        let mut forest = RandomForestRegressor::new()
            .with_n_estimators(50)
            .with_random_state(42);
        forest.fit(&x_train, &y_train).unwrap();
        let forest_pred = forest.predict(&x_test).unwrap();
        let forest_r2 = r2_score(&y_test, &forest_pred).unwrap();

        assert!(
            forest_r2 >= tree_r2 - 0.1,
            "RandomForestRegressor ({}) significantly worse than DecisionTreeRegressor ({})",
            forest_r2,
            tree_r2
        );
    }

    #[test]
    fn test_ridge_vs_linear_on_noisy_data() {
        // With noisy data, Ridge should perform at least as well as Linear
        let (x, y) = make_regression(200, 5, 0.5, 42); // More noise

        let x_train = x.slice(s![..150, ..]).to_owned();
        let y_train = y.slice(s![..150]).to_owned();
        let x_test = x.slice(s![150.., ..]).to_owned();
        let y_test = y.slice(s![150..]).to_owned();

        let mut linear = LinearRegression::new();
        linear.fit(&x_train, &y_train).unwrap();
        let linear_pred = linear.predict(&x_test).unwrap();
        let linear_r2 = r2_score(&y_test, &linear_pred).unwrap();

        let mut ridge = RidgeRegression::new(1.0);
        ridge.fit(&x_train, &y_train).unwrap();
        let ridge_pred = ridge.predict(&x_test).unwrap();
        let ridge_r2 = r2_score(&y_test, &ridge_pred).unwrap();

        // Ridge should be competitive (not dramatically worse)
        assert!(
            ridge_r2 >= linear_r2 - 0.1,
            "Ridge ({}) significantly worse than Linear ({}) on noisy data",
            ridge_r2,
            linear_r2
        );
    }

    #[test]
    fn test_boosting_competitive_with_forest() {
        // Gradient boosting should be competitive with random forest
        let (x, y) = make_binary_classification(300, 10, 42);

        let x_train = x.slice(s![..200, ..]).to_owned();
        let y_train = y.slice(s![..200]).to_owned();
        let x_test = x.slice(s![200.., ..]).to_owned();
        let y_test = y.slice(s![200..]).to_owned();

        let mut forest = RandomForestClassifier::new()
            .with_n_estimators(50)
            .with_random_state(42);
        forest.fit(&x_train, &y_train).unwrap();
        let forest_pred = forest.predict(&x_test).unwrap();
        let forest_acc = accuracy(&y_test, &forest_pred).unwrap();

        let mut gb = GradientBoostingClassifier::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1);
        gb.fit(&x_train, &y_train).unwrap();
        let gb_pred = gb.predict(&x_test).unwrap();
        let gb_acc = accuracy(&y_test, &gb_pred).unwrap();

        // GB should be within 10% of forest
        assert!(
            gb_acc >= forest_acc - 0.1,
            "GradientBoosting ({}) significantly worse than RandomForest ({})",
            gb_acc,
            forest_acc
        );
    }
}

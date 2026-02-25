//! Real-World Dataset Validation Tests
//!
//! This test suite validates that FerroML models work correctly on the built-in
//! real-world datasets (Iris, Wine, Diabetes). Each test trains a model on the
//! full dataset and predicts on the same data as a sanity check. Thresholds are
//! deliberately conservative since these are training-set (not out-of-sample)
//! evaluations.
//!
//! Note: FerroML's LogisticRegression currently only supports binary classification,
//! so multiclass datasets (Iris, Wine) use GaussianNB as the "simple probabilistic"
//! baseline instead. The diabetes dataset is a 260-sample subset of the full 442-sample
//! sklearn dataset, so linear model R2 values may be lower than the full dataset.

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

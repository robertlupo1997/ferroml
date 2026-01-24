//! UCI Dataset Integration Tests
//!
//! Tests FerroML models on UCI-style datasets with realistic characteristics:
//! - Mixed numeric features with various distributions
//! - Class imbalance
//! - Multiple classes
//!
//! These tests verify that the full pipeline works on real-world-like data.

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

    let mut model = LogisticRegression::new();
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
        .with_n_estimators(20)
        .with_max_depth(Some(15));
    model.fit(&x_train, &y_train).unwrap();
    let predictions = model.predict(&x_test).unwrap();
    let acc = accuracy(&y_test, &predictions).unwrap();

    assert!(
        acc > 0.25,
        "Random Forest should handle 7 classes reasonably: {}",
        acc
    );

    for &pred in predictions.iter() {
        assert!(pred >= 0.0 && pred <= 6.0 && (pred - pred.round()).abs() < 0.01);
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
        .with_n_estimators(10)
        .with_max_depth(Some(10));
    model.fit(&x_train, &y_train).unwrap();
    let predictions = model.predict(&x_test).unwrap();
    let r2 = r2_score(&y_test, &predictions).unwrap();

    assert!(r2 > 0.0, "RF Regressor R2 should be positive: {}", r2);
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

    let mut model1 = RandomForestClassifier::new()
        .with_n_estimators(10)
        .with_random_state(12345);
    model1.fit(&x_train, &y_train).unwrap();
    let pred1 = model1.predict(&x_test).unwrap();

    let mut model2 = RandomForestClassifier::new()
        .with_n_estimators(10)
        .with_random_state(12345);
    model2.fit(&x_train, &y_train).unwrap();
    let pred2 = model2.predict(&x_test).unwrap();

    for (p1, p2) in pred1.iter().zip(pred2.iter()) {
        assert_eq!(
            *p1, *p2,
            "Predictions should be reproducible with same seed"
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

    let mut lr = LogisticRegression::new();
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

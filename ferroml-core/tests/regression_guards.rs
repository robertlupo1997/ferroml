//! Regression Guard Tests for FerroML
//!
//! These tests prevent regressions by verifying:
//! 1. **Deterministic snapshots** — model outputs on fixed data match expected values
//! 2. **Benchmark stability** — model quality on standard datasets stays above baselines
//! 3. **API contracts** — predict-before-fit errors, shape consistency, clone equivalence
//!
//! If any of these tests fail after a code change, it signals an unintended behavioral
//! change that needs investigation.

use ferroml_core::datasets::load_iris;
use ferroml_core::decomposition::PCA;
use ferroml_core::metrics::{accuracy, r2_score};
use ferroml_core::models::forest::RandomForestClassifier;
use ferroml_core::models::knn::KNeighborsClassifier;
use ferroml_core::models::linear::LinearRegression;
use ferroml_core::models::naive_bayes::GaussianNB;
use ferroml_core::models::regularized::RidgeRegression;
use ferroml_core::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor};
use ferroml_core::models::Model;
use ferroml_core::preprocessing::scalers::{MinMaxScaler, StandardScaler};
use ferroml_core::preprocessing::Transformer;
use ndarray::{array, Array2};

// =============================================================================
// Category 1: Deterministic Snapshot Tests
// =============================================================================

/// Linear regression on perfectly linear data must produce exact predictions.
/// y = 1 + 2*x1 + 3*x2. Non-collinear features ensure the design matrix is
/// full rank. On perfectly linear data the residuals should be at machine epsilon.
#[test]
fn snapshot_linear_regression_coefficients() {
    // Non-collinear features: x1 increasing, x2 decreasing
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 5.0, 2.0, 3.0, 3.0, 4.0, 4.0, 2.0, 5.0, 1.0],
    )
    .unwrap();
    // y = 1 + 2*x1 + 3*x2
    let y = array![
        1.0 + 2.0 * 1.0 + 3.0 * 5.0,
        1.0 + 2.0 * 2.0 + 3.0 * 3.0,
        1.0 + 2.0 * 3.0 + 3.0 * 4.0,
        1.0 + 2.0 * 4.0 + 3.0 * 2.0,
        1.0 + 2.0 * 5.0 + 3.0 * 1.0
    ];

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();
    let pred = model.predict(&x).unwrap();

    for (p, t) in pred.iter().zip(y.iter()) {
        assert!(
            (p - t).abs() < 1e-10,
            "Linear regression prediction mismatch: expected {}, got {}",
            t,
            p
        );
    }
}

/// Decision tree classifier on linearly separable data must achieve 100% train accuracy.
#[test]
fn snapshot_decision_tree_classifier_separable() {
    // Two well-separated clusters: class 0 at low values, class 1 at high values
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, 1.5, 2.0, 2.0, 1.5, 2.5, 2.5, // class 0
            7.0, 7.0, 7.5, 8.0, 8.0, 7.5, 8.5, 8.5, // class 1
        ],
    )
    .unwrap();
    let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let mut model = DecisionTreeClassifier::new();
    model.fit(&x, &y).unwrap();
    let pred = model.predict(&x).unwrap();

    let acc = accuracy(&y, &pred).unwrap();
    assert!(
        (acc - 1.0).abs() < 1e-10,
        "DecisionTreeClassifier should achieve 100% on separable data, got {:.4}",
        acc
    );
}

/// KNN classifier on well-separated clusters with k=3 must achieve 100% train accuracy.
#[test]
fn snapshot_knn_classifier_separated_clusters() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, // class 0
            10.0, 10.0, 10.1, 10.1, 10.2, 10.0, 10.0, 10.2, // class 1
        ],
    )
    .unwrap();
    let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let mut model = KNeighborsClassifier::new(3);
    model.fit(&x, &y).unwrap();
    let pred = model.predict(&x).unwrap();

    let acc = accuracy(&y, &pred).unwrap();
    assert!(
        (acc - 1.0).abs() < 1e-10,
        "KNN should achieve 100% on well-separated clusters, got {:.4}",
        acc
    );
}

/// GaussianNB on well-separated Gaussian clusters should achieve 100% train accuracy.
#[test]
fn snapshot_gaussian_nb_separated_clusters() {
    // Two Gaussian clusters with zero overlap
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, // class 0
            20.0, 20.0, 20.5, 20.5, 21.0, 20.0, 20.0, 21.0, // class 1
        ],
    )
    .unwrap();
    let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let mut model = GaussianNB::new();
    model.fit(&x, &y).unwrap();
    let pred = model.predict(&x).unwrap();

    let acc = accuracy(&y, &pred).unwrap();
    assert!(
        (acc - 1.0).abs() < 1e-10,
        "GaussianNB should achieve 100% on well-separated Gaussian clusters, got {:.4}",
        acc
    );
}

/// StandardScaler on known data must produce mean=0 and std=1 per feature.
#[test]
fn snapshot_standard_scaler_output() {
    let x = array![
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0]
    ];

    let mut scaler = StandardScaler::new();
    let x_scaled = scaler.fit_transform(&x).unwrap();

    // Check mean is ~0 for each feature
    for j in 0..x_scaled.ncols() {
        let col_mean: f64 = x_scaled.column(j).mean().unwrap();
        assert!(
            col_mean.abs() < 1e-10,
            "StandardScaler column {} mean = {}, expected ~0",
            j,
            col_mean
        );
    }

    // Check that population std is ~1 for each feature (using ddof=0 to match sklearn)
    for j in 0..x_scaled.ncols() {
        let col = x_scaled.column(j);
        let mean = col.mean().unwrap();
        let var: f64 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / col.len() as f64;
        let std = var.sqrt();
        assert!(
            (std - 1.0).abs() < 1e-10,
            "StandardScaler column {} population std = {}, expected ~1",
            j,
            std
        );
    }
}

/// MinMaxScaler on known data must produce values in [0, 1].
#[test]
fn snapshot_min_max_scaler_output() {
    let x = array![
        [1.0, 100.0],
        [2.0, 200.0],
        [3.0, 300.0],
        [4.0, 400.0],
        [5.0, 500.0]
    ];

    let mut scaler = MinMaxScaler::new();
    let x_scaled = scaler.fit_transform(&x).unwrap();

    // All values should be in [0, 1]
    for &v in x_scaled.iter() {
        assert!(
            (-1e-10..=1.0 + 1e-10).contains(&v),
            "MinMaxScaler value {} outside [0, 1]",
            v
        );
    }

    // Min row should map to 0, max row should map to 1
    assert!(
        (x_scaled[[0, 0]] - 0.0).abs() < 1e-10,
        "MinMaxScaler min should be 0, got {}",
        x_scaled[[0, 0]]
    );
    assert!(
        (x_scaled[[4, 0]] - 1.0).abs() < 1e-10,
        "MinMaxScaler max should be 1, got {}",
        x_scaled[[4, 0]]
    );
    assert!(
        (x_scaled[[0, 1]] - 0.0).abs() < 1e-10,
        "MinMaxScaler min should be 0, got {}",
        x_scaled[[0, 1]]
    );
    assert!(
        (x_scaled[[4, 1]] - 1.0).abs() < 1e-10,
        "MinMaxScaler max should be 1, got {}",
        x_scaled[[4, 1]]
    );

    // Middle values should be linearly interpolated: (v - min) / (max - min)
    // For column 0: (3 - 1) / (5 - 1) = 0.5
    assert!(
        (x_scaled[[2, 0]] - 0.5).abs() < 1e-10,
        "MinMaxScaler midpoint should be 0.5, got {}",
        x_scaled[[2, 0]]
    );
}

/// PCA on correlated data: first component should capture most of the variance.
#[test]
fn snapshot_pca_first_component_dominates() {
    // Highly correlated data: x2 ~ 2 * x1
    let x = array![
        [1.0, 2.1],
        [2.0, 3.9],
        [3.0, 6.1],
        [4.0, 7.9],
        [5.0, 10.1],
        [6.0, 11.9],
        [7.0, 14.1],
        [8.0, 15.9]
    ];

    let mut pca = PCA::new().with_n_components(2);
    pca.fit(&x).unwrap();

    let var_ratio = pca.explained_variance_ratio().unwrap();

    // First component should capture > 99% of variance for nearly perfectly correlated data
    assert!(
        var_ratio[0] > 0.99,
        "PCA first component should capture > 99% variance on correlated data, got {:.4}",
        var_ratio[0]
    );

    // Second component should capture < 1%
    assert!(
        var_ratio[1] < 0.01,
        "PCA second component should capture < 1% variance on correlated data, got {:.4}",
        var_ratio[1]
    );

    // Sum of ratios should be ~1
    let total: f64 = var_ratio.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-6,
        "PCA explained variance ratios should sum to ~1, got {}",
        total
    );
}

/// Decision tree regressor on step-function data must capture the steps exactly.
#[test]
fn snapshot_decision_tree_regressor_step_function() {
    // Step function: y = 0 for x < 5, y = 10 for x >= 5
    let x = Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
    let y = array![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];

    let mut model = DecisionTreeRegressor::new();
    model.fit(&x, &y).unwrap();
    let pred = model.predict(&x).unwrap();

    // Tree should perfectly learn the step function on training data
    for (i, (p, t)) in pred.iter().zip(y.iter()).enumerate() {
        assert!(
            (p - t).abs() < 1e-10,
            "DecisionTreeRegressor step function mismatch at index {}: expected {}, got {}",
            i,
            t,
            p
        );
    }
}

// =============================================================================
// Category 2: Benchmark Stability Tests
// =============================================================================

/// Random forest on Iris training data should achieve > 95% accuracy.
/// This is a property-based check (not exact), because RF uses randomness.
#[test]
fn benchmark_iris_random_forest_train_accuracy() {
    let (dataset, _info) = load_iris();
    let (x, y) = dataset.into_arrays();

    let mut model = RandomForestClassifier::new()
        .with_n_estimators(100)
        .with_random_state(42);
    model.fit(&x, &y).unwrap();
    let pred = model.predict(&x).unwrap();
    let acc = accuracy(&y, &pred).unwrap();

    assert!(
        acc > 0.95,
        "Iris RF training accuracy {:.3} is below 0.95 baseline",
        acc
    );
}

/// Ridge regression on Iris (treated as regression on class labels) should achieve
/// a reasonable R-squared. Using alpha=1.0 on the 4-feature Iris dataset.
#[test]
fn benchmark_iris_ridge_r2() {
    let (dataset, _info) = load_iris();
    let (x, y) = dataset.into_arrays();

    let mut model = RidgeRegression::new(1.0);
    model.fit(&x, &y).unwrap();
    let pred = model.predict(&x).unwrap();
    let r2 = r2_score(&y, &pred).unwrap();

    assert!(
        r2 > 0.90,
        "Iris Ridge R-squared {:.3} is below 0.90 baseline",
        r2
    );
}

/// All predictions on standard data must be finite (no overflow, no NaN).
#[test]
fn benchmark_no_overflow_on_standard_data() {
    let (dataset, _info) = load_iris();
    let (x, y) = dataset.into_arrays();

    // LinearRegression
    let mut lr = LinearRegression::new();
    lr.fit(&x, &y).unwrap();
    let pred_lr = lr.predict(&x).unwrap();
    assert!(
        pred_lr.iter().all(|v| v.is_finite()),
        "LinearRegression produced non-finite predictions"
    );

    // DecisionTreeClassifier
    let mut dtc = DecisionTreeClassifier::new();
    dtc.fit(&x, &y).unwrap();
    let pred_dtc = dtc.predict(&x).unwrap();
    assert!(
        pred_dtc.iter().all(|v| v.is_finite()),
        "DecisionTreeClassifier produced non-finite predictions"
    );

    // KNeighborsClassifier
    let mut knn = KNeighborsClassifier::new(5);
    knn.fit(&x, &y).unwrap();
    let pred_knn = knn.predict(&x).unwrap();
    assert!(
        pred_knn.iter().all(|v| v.is_finite()),
        "KNeighborsClassifier produced non-finite predictions"
    );

    // GaussianNB
    let mut gnb = GaussianNB::new();
    gnb.fit(&x, &y).unwrap();
    let pred_gnb = gnb.predict(&x).unwrap();
    assert!(
        pred_gnb.iter().all(|v| v.is_finite()),
        "GaussianNB produced non-finite predictions"
    );
}

/// predict() output must have the same number of rows as the input.
#[test]
fn benchmark_shape_consistency() {
    let (dataset, _info) = load_iris();
    let (x, y) = dataset.into_arrays();
    let n_samples = x.nrows();

    // LinearRegression
    let mut lr = LinearRegression::new();
    lr.fit(&x, &y).unwrap();
    let pred = lr.predict(&x).unwrap();
    assert_eq!(
        pred.len(),
        n_samples,
        "LinearRegression predict output has wrong length"
    );

    // DecisionTreeClassifier
    let mut dtc = DecisionTreeClassifier::new();
    dtc.fit(&x, &y).unwrap();
    let pred = dtc.predict(&x).unwrap();
    assert_eq!(
        pred.len(),
        n_samples,
        "DecisionTreeClassifier predict output has wrong length"
    );

    // KNeighborsClassifier
    let mut knn = KNeighborsClassifier::new(5);
    knn.fit(&x, &y).unwrap();
    let pred = knn.predict(&x).unwrap();
    assert_eq!(
        pred.len(),
        n_samples,
        "KNeighborsClassifier predict output has wrong length"
    );

    // GaussianNB
    let mut gnb = GaussianNB::new();
    gnb.fit(&x, &y).unwrap();
    let pred = gnb.predict(&x).unwrap();
    assert_eq!(
        pred.len(),
        n_samples,
        "GaussianNB predict output has wrong length"
    );

    // RidgeRegression
    let mut ridge = RidgeRegression::new(1.0);
    ridge.fit(&x, &y).unwrap();
    let pred = ridge.predict(&x).unwrap();
    assert_eq!(
        pred.len(),
        n_samples,
        "RidgeRegression predict output has wrong length"
    );
}

// =============================================================================
// Category 3: API Contract Tests
// =============================================================================

/// Calling predict before fit must return an Err, not panic.
#[test]
fn contract_predict_before_fit_returns_error() {
    let x = Array2::zeros((5, 3));

    // LinearRegression
    let lr = LinearRegression::new();
    assert!(
        lr.predict(&x).is_err(),
        "LinearRegression predict before fit should return Err"
    );

    // DecisionTreeClassifier
    let dtc = DecisionTreeClassifier::new();
    assert!(
        dtc.predict(&x).is_err(),
        "DecisionTreeClassifier predict before fit should return Err"
    );

    // DecisionTreeRegressor
    let dtr = DecisionTreeRegressor::new();
    assert!(
        dtr.predict(&x).is_err(),
        "DecisionTreeRegressor predict before fit should return Err"
    );

    // KNeighborsClassifier
    let knn = KNeighborsClassifier::new(3);
    assert!(
        knn.predict(&x).is_err(),
        "KNeighborsClassifier predict before fit should return Err"
    );

    // GaussianNB
    let gnb = GaussianNB::new();
    assert!(
        gnb.predict(&x).is_err(),
        "GaussianNB predict before fit should return Err"
    );

    // RidgeRegression
    let ridge = RidgeRegression::new(1.0);
    assert!(
        ridge.predict(&x).is_err(),
        "RidgeRegression predict before fit should return Err"
    );
}

/// fit-then-predict roundtrip must succeed for all models.
#[test]
fn contract_fit_predict_roundtrip() {
    // Non-collinear 2D features for regression models
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 8.0, 2.0, 6.0, 3.0, 7.0, 4.0, 5.0, 5.0, 3.0, 6.0, 4.0, 7.0, 2.0, 8.0, 1.0,
        ],
    )
    .unwrap();
    let y_class = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let y_reg = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // LinearRegression
    let mut lr = LinearRegression::new();
    lr.fit(&x, &y_reg).unwrap();
    assert!(
        lr.predict(&x).is_ok(),
        "LinearRegression fit-predict failed"
    );

    // DecisionTreeClassifier
    let mut dtc = DecisionTreeClassifier::new();
    dtc.fit(&x, &y_class).unwrap();
    assert!(
        dtc.predict(&x).is_ok(),
        "DecisionTreeClassifier fit-predict failed"
    );

    // DecisionTreeRegressor
    let mut dtr = DecisionTreeRegressor::new();
    dtr.fit(&x, &y_reg).unwrap();
    assert!(
        dtr.predict(&x).is_ok(),
        "DecisionTreeRegressor fit-predict failed"
    );

    // KNeighborsClassifier
    let mut knn = KNeighborsClassifier::new(3);
    knn.fit(&x, &y_class).unwrap();
    assert!(
        knn.predict(&x).is_ok(),
        "KNeighborsClassifier fit-predict failed"
    );

    // GaussianNB
    let mut gnb = GaussianNB::new();
    gnb.fit(&x, &y_class).unwrap();
    assert!(gnb.predict(&x).is_ok(), "GaussianNB fit-predict failed");

    // RidgeRegression
    let mut ridge = RidgeRegression::new(1.0);
    ridge.fit(&x, &y_reg).unwrap();
    assert!(
        ridge.predict(&x).is_ok(),
        "RidgeRegression fit-predict failed"
    );
}

/// A cloned model must produce identical predictions to the original.
#[test]
fn contract_clone_equivalence() {
    // Non-collinear 2D features for regression models
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 8.0, 2.0, 6.0, 3.0, 7.0, 4.0, 5.0, 5.0, 3.0, 6.0, 4.0, 7.0, 2.0, 8.0, 1.0,
        ],
    )
    .unwrap();
    let y_class = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let y_reg = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // LinearRegression
    let mut lr = LinearRegression::new();
    lr.fit(&x, &y_reg).unwrap();
    let lr_clone = lr.clone();
    let pred_orig = lr.predict(&x).unwrap();
    let pred_clone = lr_clone.predict(&x).unwrap();
    assert_eq!(
        pred_orig, pred_clone,
        "LinearRegression clone should produce identical predictions"
    );

    // DecisionTreeClassifier
    let mut dtc = DecisionTreeClassifier::new();
    dtc.fit(&x, &y_class).unwrap();
    let dtc_clone = dtc.clone();
    let pred_orig = dtc.predict(&x).unwrap();
    let pred_clone = dtc_clone.predict(&x).unwrap();
    assert_eq!(
        pred_orig, pred_clone,
        "DecisionTreeClassifier clone should produce identical predictions"
    );

    // KNeighborsClassifier
    let mut knn = KNeighborsClassifier::new(3);
    knn.fit(&x, &y_class).unwrap();
    let knn_clone = knn.clone();
    let pred_orig = knn.predict(&x).unwrap();
    let pred_clone = knn_clone.predict(&x).unwrap();
    assert_eq!(
        pred_orig, pred_clone,
        "KNeighborsClassifier clone should produce identical predictions"
    );

    // GaussianNB
    let mut gnb = GaussianNB::new();
    gnb.fit(&x, &y_class).unwrap();
    let gnb_clone = gnb.clone();
    let pred_orig = gnb.predict(&x).unwrap();
    let pred_clone = gnb_clone.predict(&x).unwrap();
    assert_eq!(
        pred_orig, pred_clone,
        "GaussianNB clone should produce identical predictions"
    );
}

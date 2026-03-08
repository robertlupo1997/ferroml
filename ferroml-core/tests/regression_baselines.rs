//! Regression Test Baselines for FerroML
//!
//! This integration test loads model performance baselines from `regression/baselines.json`
//! and verifies that each model meets or exceeds its minimum metric threshold on
//! deterministic synthetic data. Any failure here indicates an unintended behavioral
//! regression — the model's quality has degraded relative to a known-good state.
//!
//! ## How it works
//!
//! 1. Loads `baselines.json` which defines model name, dataset params, metric, and min value.
//! 2. Generates the synthetic dataset using the same seed every time.
//! 3. Trains the model on the full dataset (train = test for baseline stability).
//! 4. Computes the metric and asserts it meets the baseline.
//!
//! ## Adding a new baseline
//!
//! Add an entry to `regression/baselines.json` and a corresponding match arm in
//! `train_and_evaluate`.

use ferroml_core::metrics::{accuracy, r2_score};
use ferroml_core::models::adaboost::AdaBoostClassifier;
use ferroml_core::models::boosting::{GradientBoostingClassifier, GradientBoostingRegressor};
use ferroml_core::models::extra_trees::{ExtraTreesClassifier, ExtraTreesRegressor};
use ferroml_core::models::forest::{RandomForestClassifier, RandomForestRegressor};
use ferroml_core::models::hist_boosting::HistGradientBoostingRegressor;
use ferroml_core::models::knn::{KNeighborsClassifier, KNeighborsRegressor};
use ferroml_core::models::linear::LinearRegression;
use ferroml_core::models::logistic::LogisticRegression;
use ferroml_core::models::naive_bayes::GaussianNB;
use ferroml_core::models::regularized::{ElasticNet, LassoRegression, RidgeRegression};
use ferroml_core::models::sgd::SGDClassifier;
use ferroml_core::models::svm::LinearSVC;
use ferroml_core::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor};
use ferroml_core::models::Model;
use ndarray::{Array1, Array2};
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use std::path::Path;

// ============================================================================
// Baseline schema
// ============================================================================

#[derive(Deserialize)]
struct BaselinesFile {
    baselines: Vec<Baseline>,
}

#[derive(Deserialize)]
struct Baseline {
    name: String,
    #[allow(dead_code)]
    task: String,
    dataset: DatasetSpec,
    metric: String,
    min_value: f64,
    #[allow(dead_code)]
    notes: Option<String>,
}

#[derive(Deserialize)]
struct DatasetSpec {
    #[serde(rename = "type")]
    dtype: String,
    n_samples: usize,
    n_features: usize,
    // Regression-specific
    noise: Option<f64>,
    seed: u64,
    // Classification-specific
    n_classes: Option<usize>,
}

// ============================================================================
// Synthetic data generators (deterministic via ChaCha8Rng)
// ============================================================================

fn make_regression_data(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    use rand::{Rng, SeedableRng};
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.random_range(-10.0..10.0));

    let true_coef: Vec<f64> = (0..n_features).map(|i| (i + 1) as f64 * 0.5).collect();

    let y = Array1::from_shape_fn(n_samples, |i| {
        let row = x.row(i);
        let signal: f64 = row.iter().zip(true_coef.iter()).map(|(xi, c)| xi * c).sum();
        signal + rng.random_range(-noise..noise)
    });

    (x, y)
}

fn make_classification_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    use rand::{Rng, SeedableRng};
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let samples_per_class = n_samples / n_classes;
    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    for class in 0..n_classes {
        let center: Vec<f64> = (0..n_features).map(|f| (class * 3 + f) as f64).collect();
        for _ in 0..samples_per_class {
            for val in center.iter().take(n_features) {
                x_data.push(val + rng.random_range(-1.0..1.0));
            }
            y_data.push(class as f64);
        }
    }

    let actual_samples = samples_per_class * n_classes;
    let x = Array2::from_shape_vec((actual_samples, n_features), x_data).unwrap();
    let y = Array1::from_vec(y_data);
    (x, y)
}

// ============================================================================
// Model training dispatcher
// ============================================================================

fn train_and_evaluate(
    name: &str,
    x: &Array2<f64>,
    y: &Array1<f64>,
    metric: &str,
) -> Result<f64, String> {
    let pred = match name {
        // --- Regressors ---
        "LinearRegression" => {
            let mut m = LinearRegression::new();
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "RidgeRegression" => {
            let mut m = RidgeRegression::new(1.0);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "LassoRegression" => {
            let mut m = LassoRegression::new(0.1);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "ElasticNet" => {
            let mut m = ElasticNet::new(0.1, 0.5);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "DecisionTreeRegressor" => {
            let mut m = DecisionTreeRegressor::new().with_random_state(42);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "RandomForestRegressor" => {
            let mut m = RandomForestRegressor::new()
                .with_n_estimators(50)
                .with_random_state(42);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "GradientBoostingRegressor" => {
            let mut m = GradientBoostingRegressor::new()
                .with_n_estimators(100)
                .with_random_state(42);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "KNeighborsRegressor" => {
            let mut m = KNeighborsRegressor::new(5);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "ExtraTreesRegressor" => {
            let mut m = ExtraTreesRegressor::new()
                .with_n_estimators(50)
                .with_random_state(42);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "HistGradientBoostingRegressor" => {
            let mut m = HistGradientBoostingRegressor::new()
                .with_max_iter(100)
                .with_random_state(42);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        // --- Classifiers ---
        "LogisticRegression" => {
            let mut m = LogisticRegression::new();
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "DecisionTreeClassifier" => {
            let mut m = DecisionTreeClassifier::new().with_random_state(42);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "RandomForestClassifier" => {
            let mut m = RandomForestClassifier::new()
                .with_n_estimators(50)
                .with_random_state(42);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "GradientBoostingClassifier" => {
            let mut m = GradientBoostingClassifier::new()
                .with_n_estimators(100)
                .with_random_state(42);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "KNeighborsClassifier" => {
            let mut m = KNeighborsClassifier::new(5);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "GaussianNB" => {
            let mut m = GaussianNB::new();
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "ExtraTreesClassifier" => {
            let mut m = ExtraTreesClassifier::new()
                .with_n_estimators(50)
                .with_random_state(42);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "AdaBoostClassifier" => {
            let mut m = AdaBoostClassifier::new(50).with_random_state(42);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "SGDClassifier" => {
            let mut m = SGDClassifier::new()
                .with_max_iter(2000)
                .with_random_state(42);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        "LinearSVC" => {
            let mut m = LinearSVC::new().with_max_iter(2000);
            m.fit(x, y).map_err(|e| format!("{e}"))?;
            m.predict(x).map_err(|e| format!("{e}"))?
        }
        _ => return Err(format!("Unknown model: {name}")),
    };

    let score = match metric {
        "r2" => r2_score(y, &pred).map_err(|e| format!("{e}"))?,
        "accuracy" => accuracy(y, &pred).map_err(|e| format!("{e}"))?,
        other => return Err(format!("Unknown metric: {other}")),
    };

    Ok(score)
}

// ============================================================================
// Main test: load baselines and run all checks
// ============================================================================

#[test]
fn regression_baselines_all_models() {
    let baselines_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("regression")
        .join("baselines.json");

    let content = std::fs::read_to_string(&baselines_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read baselines file at {}: {}",
            baselines_path.display(),
            e
        )
    });

    let baselines: BaselinesFile =
        serde_json::from_str(&content).expect("Failed to parse baselines.json");

    let mut failures: Vec<String> = Vec::new();
    let mut successes: Vec<String> = Vec::new();

    for baseline in &baselines.baselines {
        let (x, y) = if baseline.dataset.dtype == "regression" {
            make_regression_data(
                baseline.dataset.n_samples,
                baseline.dataset.n_features,
                baseline.dataset.noise.unwrap_or(1.0),
                baseline.dataset.seed,
            )
        } else {
            make_classification_data(
                baseline.dataset.n_samples,
                baseline.dataset.n_features,
                baseline.dataset.n_classes.unwrap_or(2),
                baseline.dataset.seed,
            )
        };

        match train_and_evaluate(&baseline.name, &x, &y, &baseline.metric) {
            Ok(score) => {
                if score >= baseline.min_value {
                    successes.push(format!(
                        "  PASS: {} — {} = {:.4} (baseline: {:.4})",
                        baseline.name, baseline.metric, score, baseline.min_value
                    ));
                } else {
                    failures.push(format!(
                        "  FAIL: {} — {} = {:.4} < baseline {:.4} (regression detected!)",
                        baseline.name, baseline.metric, score, baseline.min_value
                    ));
                }
            }
            Err(e) => {
                failures.push(format!(
                    "  ERROR: {} — failed to train/evaluate: {}",
                    baseline.name, e
                ));
            }
        }
    }

    // Print full report
    eprintln!("\n=== Regression Baseline Report ===");
    eprintln!("Models tested: {}", baselines.baselines.len());
    eprintln!("Passed: {}", successes.len());
    eprintln!("Failed: {}", failures.len());
    eprintln!();
    for s in &successes {
        eprintln!("{s}");
    }
    for f in &failures {
        eprintln!("{f}");
    }
    eprintln!("=================================\n");

    assert!(
        failures.is_empty(),
        "\n{} regression baseline(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ============================================================================
// Individual model tests for granular CI reporting
// ============================================================================

macro_rules! baseline_test {
    ($test_name:ident, $model_name:literal, $task:literal, $metric:literal, $min:expr) => {
        #[test]
        fn $test_name() {
            let (x, y) = if $task == "regression" {
                make_regression_data(200, 5, 0.5, 42)
            } else {
                make_classification_data(200, 5, 2, 42)
            };

            let score = train_and_evaluate($model_name, &x, &y, $metric)
                .unwrap_or_else(|e| panic!("{} failed: {}", $model_name, e));

            assert!(
                score >= $min,
                "{}: {} = {:.4} is below baseline {:.4}",
                $model_name,
                $metric,
                score,
                $min,
            );
        }
    };
}

// Regressors
baseline_test!(
    baseline_linear_regression,
    "LinearRegression",
    "regression",
    "r2",
    0.95
);
baseline_test!(
    baseline_ridge_regression,
    "RidgeRegression",
    "regression",
    "r2",
    0.94
);
baseline_test!(
    baseline_lasso_regression,
    "LassoRegression",
    "regression",
    "r2",
    0.90
);
baseline_test!(baseline_elastic_net, "ElasticNet", "regression", "r2", 0.88);
baseline_test!(
    baseline_dt_regressor,
    "DecisionTreeRegressor",
    "regression",
    "r2",
    0.85
);
baseline_test!(
    baseline_rf_regressor,
    "RandomForestRegressor",
    "regression",
    "r2",
    0.90
);
baseline_test!(
    baseline_gb_regressor,
    "GradientBoostingRegressor",
    "regression",
    "r2",
    0.90
);
baseline_test!(
    baseline_knn_regressor,
    "KNeighborsRegressor",
    "regression",
    "r2",
    0.80
);
baseline_test!(
    baseline_et_regressor,
    "ExtraTreesRegressor",
    "regression",
    "r2",
    0.90
);
baseline_test!(
    baseline_hgb_regressor,
    "HistGradientBoostingRegressor",
    "regression",
    "r2",
    0.85
);

// Classifiers
baseline_test!(
    baseline_logistic_regression,
    "LogisticRegression",
    "classification",
    "accuracy",
    0.90
);
baseline_test!(
    baseline_dt_classifier,
    "DecisionTreeClassifier",
    "classification",
    "accuracy",
    0.95
);
baseline_test!(
    baseline_rf_classifier,
    "RandomForestClassifier",
    "classification",
    "accuracy",
    0.95
);
baseline_test!(
    baseline_gb_classifier,
    "GradientBoostingClassifier",
    "classification",
    "accuracy",
    0.90
);
baseline_test!(
    baseline_knn_classifier,
    "KNeighborsClassifier",
    "classification",
    "accuracy",
    0.90
);
baseline_test!(
    baseline_gaussian_nb,
    "GaussianNB",
    "classification",
    "accuracy",
    0.85
);
baseline_test!(
    baseline_et_classifier,
    "ExtraTreesClassifier",
    "classification",
    "accuracy",
    0.95
);
baseline_test!(
    baseline_adaboost_classifier,
    "AdaBoostClassifier",
    "classification",
    "accuracy",
    0.90
);
baseline_test!(
    baseline_sgd_classifier,
    "SGDClassifier",
    "classification",
    "accuracy",
    0.80
);
baseline_test!(
    baseline_linear_svc,
    "LinearSVC",
    "classification",
    "accuracy",
    0.85
);

//! Pipeline Example - Preprocessing + Model Workflow
//!
//! This example demonstrates FerroML's Pipeline for chaining preprocessing
//! transformers and models into a single reusable workflow.
//!
//! Run with: `cargo run --example pipeline`

use ferroml_core::datasets::{load_diabetes, load_iris, make_regression};
use ferroml_core::metrics::{accuracy, r2_score};
use ferroml_core::models::linear::LinearRegression;
use ferroml_core::models::logistic::LogisticRegression;
use ferroml_core::pipeline::{ColumnSelector, ColumnTransformer, Pipeline};
use ferroml_core::preprocessing::imputers::{ImputeStrategy, SimpleImputer};
use ferroml_core::preprocessing::scalers::StandardScaler;
use ferroml_core::preprocessing::Transformer;
use ndarray::{Array1, Array2};

fn main() -> ferroml_core::Result<()> {
    println!("=============================================================");
    println!("FerroML Pipeline - Preprocessing + Model Workflow Demo");
    println!("=============================================================\n");

    // =========================================================================
    // 1. Basic Pipeline: Scaler + Linear Regression
    // =========================================================================
    println!("1. BASIC PIPELINE: StandardScaler + LinearRegression");
    println!("-----------------------------------------------------");

    // Load diabetes dataset (regression task)
    let (diabetes_dataset, diabetes_info) = load_diabetes();
    println!("Dataset: {}", diabetes_info.name);
    println!("Samples: {}", diabetes_dataset.n_samples());
    println!("Features: {}", diabetes_dataset.n_features());

    // Split data
    let (train_data, test_data) = diabetes_dataset.train_test_split(0.2, true, Some(42))?;
    let (x_train, y_train) = train_data.into_arrays();
    let (x_test, y_test) = test_data.into_arrays();

    println!("Train samples: {}", x_train.nrows());
    println!("Test samples: {}", x_test.nrows());

    // Create a pipeline with scaler and linear regression
    let mut regression_pipeline = Pipeline::new()
        .add_transformer("scaler", StandardScaler::new())
        .add_model("regressor", LinearRegression::new());

    println!("\nPipeline steps: {:?}", regression_pipeline.step_names());
    println!("Has model: {}", regression_pipeline.has_model());

    // Fit the entire pipeline
    regression_pipeline.fit(&x_train, &y_train)?;
    println!("\nPipeline fitted: {}", regression_pipeline.is_fitted());

    // Predict
    let y_pred = regression_pipeline.predict(&x_test)?;
    let r2 = r2_score(&y_test, &y_pred)?;
    println!("Test R^2 Score: {:.4}", r2);

    // =========================================================================
    // 2. Pipeline with Missing Value Imputation
    // =========================================================================
    println!("\n\n2. PIPELINE WITH IMPUTATION");
    println!("----------------------------");

    // Create synthetic data with missing values
    let (x_with_missing, y_regression) = create_data_with_missing(100, 5, 0.1);
    let n_missing = x_with_missing.iter().filter(|v| v.is_nan()).count();
    println!("Created data with {} missing values (10% of data)", n_missing);

    // Split
    let split_idx = 80;
    let x_train_missing = x_with_missing.slice(ndarray::s![..split_idx, ..]).to_owned();
    let x_test_missing = x_with_missing.slice(ndarray::s![split_idx.., ..]).to_owned();
    let y_train_missing = y_regression.slice(ndarray::s![..split_idx]).to_owned();
    let y_test_missing = y_regression.slice(ndarray::s![split_idx..]).to_owned();

    // Create pipeline: Imputation -> Scaling -> Linear Regression
    let mut imputation_pipeline = Pipeline::new()
        .add_transformer("imputer", SimpleImputer::new(ImputeStrategy::Mean))
        .add_transformer("scaler", StandardScaler::new())
        .add_model("regressor", LinearRegression::new());

    println!("Pipeline steps: {:?}", imputation_pipeline.step_names());

    imputation_pipeline.fit(&x_train_missing, &y_train_missing)?;
    let y_pred_imputed = imputation_pipeline.predict(&x_test_missing)?;
    let r2_imputed = r2_score(&y_test_missing, &y_pred_imputed)?;
    println!("Test R^2 Score (with imputation): {:.4}", r2_imputed);

    // =========================================================================
    // 3. Transform-Only Pipeline (No Model)
    // =========================================================================
    println!("\n\n3. TRANSFORM-ONLY PIPELINE");
    println!("---------------------------");

    // A pipeline can also be just transformers (for preprocessing)
    let mut preprocessing_pipeline = Pipeline::new()
        .add_transformer("imputer", SimpleImputer::new(ImputeStrategy::Median))
        .add_transformer("scaler", StandardScaler::new());

    println!("Pipeline steps: {:?}", preprocessing_pipeline.step_names());
    println!("Has model: {}", preprocessing_pipeline.has_model());

    // Fit and transform
    let x_transformed = preprocessing_pipeline.fit_transform(&x_train_missing, &y_train_missing)?;
    println!("\nTransformed data shape: {:?}", x_transformed.dim());

    // Check that missing values are handled
    let n_missing_after = x_transformed.iter().filter(|v| v.is_nan()).count();
    println!("Missing values after transform: {}", n_missing_after);

    // Check scaling
    let col_means: Vec<f64> = (0..x_transformed.ncols())
        .map(|j| x_transformed.column(j).mean().unwrap())
        .collect();
    let col_stds: Vec<f64> = (0..x_transformed.ncols())
        .map(|j| x_transformed.column(j).std(0.0))
        .collect();
    println!(
        "Column means (should be ~0): {:?}",
        col_means.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>()
    );
    println!(
        "Column stds (should be ~1): {:?}",
        col_stds.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>()
    );

    // =========================================================================
    // 4. Classification Pipeline
    // =========================================================================
    println!("\n\n4. CLASSIFICATION PIPELINE");
    println!("--------------------------");

    // Load Iris dataset for classification
    let (iris_dataset, iris_info) = load_iris();
    println!("Dataset: {}", iris_info.name);
    println!("Classes: {:?}", iris_info.n_classes);

    // For binary classification, convert to setosa vs rest
    let (iris_train, iris_test) = iris_dataset.train_test_split(0.3, true, Some(42))?;
    let (x_iris_train, y_iris_train) = iris_train.into_arrays();
    let (x_iris_test, y_iris_test) = iris_test.into_arrays();

    // Convert to binary (setosa = 1, rest = 0)
    let y_train_binary = y_iris_train.mapv(|y| if y == 0.0 { 1.0 } else { 0.0 });
    let y_test_binary = y_iris_test.mapv(|y| if y == 0.0 { 1.0 } else { 0.0 });

    // Create classification pipeline
    let mut classification_pipeline = Pipeline::new()
        .add_transformer("scaler", StandardScaler::new())
        .add_model("classifier", LogisticRegression::new().with_confidence_level(0.95));

    classification_pipeline.fit(&x_iris_train, &y_train_binary)?;

    let y_pred_class = classification_pipeline.predict(&x_iris_test)?;
    let acc = accuracy(&y_test_binary, &y_pred_class)?;
    println!("Test Accuracy: {:.4}", acc);

    // =========================================================================
    // 5. ColumnTransformer for Different Feature Types
    // =========================================================================
    println!("\n\n5. COLUMNTRANSFORMER EXAMPLE");
    println!("-----------------------------");
    println!("Apply different transformers to different columns\n");

    // Create some synthetic data with different feature types
    // make_regression(n_samples, n_features, n_informative, noise, random_state)
    let (dataset_mixed, _) = make_regression(100, 6, 4, 0.0, Some(42));
    let (x_mixed, _) = dataset_mixed.into_arrays();

    // Assume columns 0-2 need StandardScaler, columns 3-5 need different scaling
    let mut column_transformer = ColumnTransformer::new()
        .add_transformer(
            "std_scaler",
            StandardScaler::new(),
            ColumnSelector::indices([0, 1, 2]),
        )
        .add_transformer(
            "robust_scaler",
            StandardScaler::new().with_mean(false), // Just for demo
            ColumnSelector::indices([3, 4, 5]),
        );

    println!("Transformers: {:?}", column_transformer.transformer_names());

    let x_col_transformed = column_transformer.fit_transform(&x_mixed)?;
    println!(
        "Input shape: {:?} -> Output shape: {:?}",
        x_mixed.dim(),
        x_col_transformed.dim()
    );

    // =========================================================================
    // 6. Pipeline with Caching
    // =========================================================================
    println!("\n\n6. PIPELINE WITH CACHING");
    println!("------------------------");

    use ferroml_core::pipeline::CacheStrategy;

    // Enable caching for intermediate transformations
    let mut cached_pipeline = Pipeline::new()
        .add_transformer("scaler", StandardScaler::new())
        .add_model("regressor", LinearRegression::new())
        .with_cache(CacheStrategy::Memory);

    cached_pipeline.fit(&x_train, &y_train)?;

    // First prediction - populates cache
    let _ = cached_pipeline.predict(&x_test)?;
    println!("First prediction completed (cache populated)");

    // Second prediction - uses cache for transform step
    let y_pred_cached = cached_pipeline.predict(&x_test)?;
    let r2_cached = r2_score(&y_test, &y_pred_cached)?;
    println!("Second prediction R^2: {:.4} (from cache)", r2_cached);

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=============================================================");
    println!("SUMMARY: FerroML Pipeline Capabilities");
    println!("=============================================================");
    println!(
        "
FerroML's Pipeline module provides:

  Pipeline:
    - Chain multiple transformers and a final model
    - Automatic fit/transform sequencing (prevents data leakage)
    - Named steps for easy access and modification
    - Memory caching for intermediate transformations
    - Combined search space for HPO
    - set_params with 'step__param' syntax

  ColumnTransformer:
    - Apply different transformers to different column subsets
    - Column selection: indices, mask, or all
    - Remainder handling: drop or passthrough
    - Parallel execution via rayon

  Transform-only pipelines:
    - Use for reusable preprocessing workflows
    - Combine imputation, scaling, encoding steps

This modular approach ensures reproducible ML workflows with proper
data handling at each step.
"
    );

    Ok(())
}

/// Create synthetic data with missing values
fn create_data_with_missing(n_samples: usize, n_features: usize, missing_rate: f64) -> (Array2<f64>, Array1<f64>) {
    // Generate data using make_regression
    // make_regression(n_samples, n_features, n_informative, noise, random_state)
    let (dataset, _) = make_regression(n_samples, n_features, n_features, 0.0, Some(42));
    let (mut x, y) = dataset.into_arrays();

    // Simple LCG for reproducibility
    let mut seed: u64 = 123;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f64) / (u32::MAX as f64)
    };

    // Introduce missing values
    for i in 0..n_samples {
        for j in 0..n_features {
            if rand() < missing_rate {
                x[[i, j]] = f64::NAN;
            }
        }
    }

    (x, y)
}

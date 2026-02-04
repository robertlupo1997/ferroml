//! Gradient Boosting with Monotonic Constraints
//!
//! This example demonstrates FerroML's histogram-based gradient boosting with
//! monotonic constraints - a key feature for incorporating domain knowledge
//! into machine learning models.
//!
//! ## When to Use Monotonic Constraints
//!
//! Monotonic constraints enforce that the model's prediction changes in only
//! one direction (increasing or decreasing) as a feature increases. This is
//! useful when you have domain knowledge about feature relationships:
//!
//! - **Credit Risk**: Risk should increase with debt-to-income ratio
//! - **Pricing**: Price should increase with demand indicators
//! - **Medical**: Disease severity should increase with certain biomarkers
//! - **Insurance**: Premium should increase with risk factors
//!
//! Run with: `cargo run --example gradient_boosting`

use ferroml_core::metrics::{mae, mse, r2_score};
use ferroml_core::models::hist_boosting::{
    GrowthStrategy, HistEarlyStopping, HistGradientBoostingRegressor, HistRegressionLoss,
    MonotonicConstraint,
};
use ferroml_core::models::Model;
use ndarray::{Array1, Array2};

fn main() -> ferroml_core::Result<()> {
    println!("=============================================================");
    println!("FerroML Gradient Boosting - Monotonic Constraints Demo");
    println!("=============================================================\n");

    // =========================================================================
    // 1. Create Synthetic Dataset with Known Monotonic Relationships
    // =========================================================================
    println!("1. CREATING SYNTHETIC DATASET");
    println!("-----------------------------");
    println!("Simulating a credit risk scenario with 4 features:");
    println!("  Feature 0: debt_to_income   (should have POSITIVE effect on risk)");
    println!("  Feature 1: credit_history   (should have NEGATIVE effect on risk)");
    println!("  Feature 2: num_delinquencies (should have POSITIVE effect on risk)");
    println!("  Feature 3: account_age      (should have NEGATIVE effect on risk)\n");

    // Generate base data
    let (x, y) = generate_credit_risk_data(1000, 42)?;

    // Split into train/test
    let n_train = 800;
    let x_train = x.slice(ndarray::s![..n_train, ..]).to_owned();
    let y_train = y.slice(ndarray::s![..n_train]).to_owned();
    let x_test = x.slice(ndarray::s![n_train.., ..]).to_owned();
    let y_test = y.slice(ndarray::s![n_train..]).to_owned();

    println!("Train samples: {}", x_train.nrows());
    println!("Test samples: {}", x_test.nrows());

    // =========================================================================
    // 2. Standard Gradient Boosting (No Constraints)
    // =========================================================================
    println!("\n\n2. HISTOGRAM GRADIENT BOOSTING (NO CONSTRAINTS)");
    println!("------------------------------------------------");

    let mut model_unconstrained = HistGradientBoostingRegressor::new()
        .with_max_iter(100)
        .with_learning_rate(0.1)
        .with_max_depth(Some(5))
        .with_random_state(42);

    model_unconstrained.fit(&x_train, &y_train)?;

    let y_pred_unconstrained = model_unconstrained.predict(&x_test)?;
    print_regression_metrics("Unconstrained", &y_test, &y_pred_unconstrained)?;

    // Verify monotonicity (may violate domain knowledge)
    println!("\nMonotonicity Check (unconstrained):");
    check_monotonicity(&model_unconstrained, &x_test)?;

    // =========================================================================
    // 3. Gradient Boosting with Monotonic Constraints
    // =========================================================================
    println!("\n\n3. HISTOGRAM GRADIENT BOOSTING (WITH MONOTONIC CONSTRAINTS)");
    println!("------------------------------------------------------------");
    println!("Applying constraints:");
    println!("  Feature 0 (debt_to_income):   Positive (+) - higher debt = higher risk");
    println!("  Feature 1 (credit_history):   Negative (-) - longer history = lower risk");
    println!("  Feature 2 (num_delinquencies): Positive (+) - more delinquencies = higher risk");
    println!("  Feature 3 (account_age):      Negative (-) - older accounts = lower risk\n");

    let mut model_constrained = HistGradientBoostingRegressor::new()
        .with_max_iter(100)
        .with_learning_rate(0.1)
        .with_max_depth(Some(5))
        .with_monotonic_constraints(vec![
            MonotonicConstraint::Positive, // debt_to_income: higher = higher risk
            MonotonicConstraint::Negative, // credit_history: longer = lower risk
            MonotonicConstraint::Positive, // num_delinquencies: more = higher risk
            MonotonicConstraint::Negative, // account_age: older = lower risk
        ])
        .with_random_state(42);

    model_constrained.fit(&x_train, &y_train)?;

    let y_pred_constrained = model_constrained.predict(&x_test)?;
    print_regression_metrics("Constrained", &y_test, &y_pred_constrained)?;

    // Verify monotonicity is enforced
    println!("\nMonotonicity Check (constrained):");
    check_monotonicity(&model_constrained, &x_test)?;

    // =========================================================================
    // 4. Feature Interaction Constraints
    // =========================================================================
    println!("\n\n4. FEATURE INTERACTION CONSTRAINTS");
    println!("-----------------------------------");
    println!("Limiting which features can interact in the same tree:");
    println!("  Group 1: [debt_to_income, credit_history] - financial ratios");
    println!("  Group 2: [num_delinquencies, account_age] - account history\n");

    let mut model_interaction = HistGradientBoostingRegressor::new()
        .with_max_iter(100)
        .with_learning_rate(0.1)
        .with_max_depth(Some(5))
        .with_monotonic_constraints(vec![
            MonotonicConstraint::Positive,
            MonotonicConstraint::Negative,
            MonotonicConstraint::Positive,
            MonotonicConstraint::Negative,
        ])
        .with_interaction_constraints(vec![
            vec![0, 1], // debt_to_income can only interact with credit_history
            vec![2, 3], // num_delinquencies can only interact with account_age
        ])
        .with_random_state(42);

    model_interaction.fit(&x_train, &y_train)?;

    let y_pred_interaction = model_interaction.predict(&x_test)?;
    print_regression_metrics("Interaction Constrained", &y_test, &y_pred_interaction)?;

    // =========================================================================
    // 5. Early Stopping with Validation Set
    // =========================================================================
    println!("\n\n5. EARLY STOPPING");
    println!("-----------------");
    println!("Training with early stopping to prevent overfitting...\n");

    // Further split train into train/validation for early stopping
    let n_valid = 100;
    let x_subtrain = x_train
        .slice(ndarray::s![..n_train - n_valid, ..])
        .to_owned();
    let y_subtrain = y_train.slice(ndarray::s![..n_train - n_valid]).to_owned();
    let _x_valid = x_train
        .slice(ndarray::s![n_train - n_valid.., ..])
        .to_owned();
    let _y_valid = y_train.slice(ndarray::s![n_train - n_valid..]).to_owned();

    let mut model_early = HistGradientBoostingRegressor::new()
        .with_max_iter(500) // Allow many iterations
        .with_learning_rate(0.1)
        .with_max_depth(Some(5))
        .with_monotonic_constraints(vec![
            MonotonicConstraint::Positive,
            MonotonicConstraint::Negative,
            MonotonicConstraint::Positive,
            MonotonicConstraint::Negative,
        ])
        .with_early_stopping(HistEarlyStopping {
            validation_fraction: 0.2, // Use 20% of training data for validation
            patience: 10,
            tol: 1e-4,
        })
        .with_random_state(42);

    // Fit with validation monitoring
    model_early.fit(&x_subtrain, &y_subtrain)?;

    println!("Max iterations: 500");
    println!("Early stopping patience: 10 iterations");

    // Check training history
    if let Some(history) = model_early.train_loss_history() {
        println!("Actual iterations: {}", history.len());
        println!("Initial loss: {:.4}", history.first().unwrap_or(&0.0));
        println!("Final loss: {:.4}", history.last().unwrap_or(&0.0));
    }

    let y_pred_early = model_early.predict(&x_test)?;
    print_regression_metrics("Early Stopping", &y_test, &y_pred_early)?;

    // =========================================================================
    // 6. Different Loss Functions
    // =========================================================================
    println!("\n\n6. LOSS FUNCTION COMPARISON");
    println!("---------------------------");
    println!("Comparing different loss functions on the same data:\n");

    // Squared Error (L2) - standard, sensitive to outliers
    let mut model_l2 = HistGradientBoostingRegressor::new()
        .with_max_iter(100)
        .with_learning_rate(0.1)
        .with_loss(HistRegressionLoss::SquaredError)
        .with_monotonic_constraints(vec![
            MonotonicConstraint::Positive,
            MonotonicConstraint::Negative,
            MonotonicConstraint::Positive,
            MonotonicConstraint::Negative,
        ])
        .with_random_state(42);

    model_l2.fit(&x_train, &y_train)?;
    let y_pred_l2 = model_l2.predict(&x_test)?;
    let mse_l2 = mse(&y_test, &y_pred_l2)?;
    let mae_l2 = mae(&y_test, &y_pred_l2)?;

    // Absolute Error (L1) - robust to outliers
    let mut model_l1 = HistGradientBoostingRegressor::new()
        .with_max_iter(100)
        .with_learning_rate(0.1)
        .with_loss(HistRegressionLoss::AbsoluteError)
        .with_monotonic_constraints(vec![
            MonotonicConstraint::Positive,
            MonotonicConstraint::Negative,
            MonotonicConstraint::Positive,
            MonotonicConstraint::Negative,
        ])
        .with_random_state(42);

    model_l1.fit(&x_train, &y_train)?;
    let y_pred_l1 = model_l1.predict(&x_test)?;
    let mse_l1 = mse(&y_test, &y_pred_l1)?;
    let mae_l1 = mae(&y_test, &y_pred_l1)?;

    // Huber Loss - balanced (L2 for small errors, L1 for large)
    let mut model_huber = HistGradientBoostingRegressor::new()
        .with_max_iter(100)
        .with_learning_rate(0.1)
        .with_loss(HistRegressionLoss::Huber)
        .with_huber_delta(1.0)
        .with_monotonic_constraints(vec![
            MonotonicConstraint::Positive,
            MonotonicConstraint::Negative,
            MonotonicConstraint::Positive,
            MonotonicConstraint::Negative,
        ])
        .with_random_state(42);

    model_huber.fit(&x_train, &y_train)?;
    let y_pred_huber = model_huber.predict(&x_test)?;
    let mse_huber = mse(&y_test, &y_pred_huber)?;
    let mae_huber = mae(&y_test, &y_pred_huber)?;

    println!("{:<20} {:>12} {:>12}", "Loss Function", "MSE", "MAE");
    println!("{}", "-".repeat(46));
    println!(
        "{:<20} {:>12.4} {:>12.4}",
        "Squared Error (L2)", mse_l2, mae_l2
    );
    println!(
        "{:<20} {:>12.4} {:>12.4}",
        "Absolute Error (L1)", mse_l1, mae_l1
    );
    println!("{:<20} {:>12.4} {:>12.4}", "Huber", mse_huber, mae_huber);

    // =========================================================================
    // 7. Growth Strategy Comparison
    // =========================================================================
    println!("\n\n7. GROWTH STRATEGY COMPARISON");
    println!("------------------------------");
    println!("Leaf-wise (LightGBM-style) vs Depth-first (traditional CART):\n");

    // Leaf-wise growth (default, LightGBM-style)
    let mut model_leaf = HistGradientBoostingRegressor::new()
        .with_max_iter(100)
        .with_learning_rate(0.1)
        .with_max_depth(Some(5))
        .with_growth_strategy(GrowthStrategy::LeafWise)
        .with_monotonic_constraints(vec![
            MonotonicConstraint::Positive,
            MonotonicConstraint::Negative,
            MonotonicConstraint::Positive,
            MonotonicConstraint::Negative,
        ])
        .with_random_state(42);

    model_leaf.fit(&x_train, &y_train)?;
    let y_pred_leaf = model_leaf.predict(&x_test)?;

    // Depth-first growth (traditional)
    let mut model_depth = HistGradientBoostingRegressor::new()
        .with_max_iter(100)
        .with_learning_rate(0.1)
        .with_max_depth(Some(5))
        .with_growth_strategy(GrowthStrategy::DepthFirst)
        .with_monotonic_constraints(vec![
            MonotonicConstraint::Positive,
            MonotonicConstraint::Negative,
            MonotonicConstraint::Positive,
            MonotonicConstraint::Negative,
        ])
        .with_random_state(42);

    model_depth.fit(&x_train, &y_train)?;
    let y_pred_depth = model_depth.predict(&x_test)?;

    println!("{:<20} {:>12} {:>12}", "Strategy", "MSE", "R²");
    println!("{}", "-".repeat(46));
    println!(
        "{:<20} {:>12.4} {:>12.4}",
        "Leaf-wise",
        mse(&y_test, &y_pred_leaf)?,
        r2_score(&y_test, &y_pred_leaf)?
    );
    println!(
        "{:<20} {:>12.4} {:>12.4}",
        "Depth-first",
        mse(&y_test, &y_pred_depth)?,
        r2_score(&y_test, &y_pred_depth)?
    );

    // =========================================================================
    // 8. Feature Importance
    // =========================================================================
    println!("\n\n8. FEATURE IMPORTANCE");
    println!("---------------------");

    if let Some(importance) = model_constrained.feature_importance() {
        let feature_names = [
            "debt_to_income",
            "credit_history",
            "num_delinquencies",
            "account_age",
        ];

        let mut pairs: Vec<_> = feature_names.iter().zip(importance.iter()).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!("{:<20} {:>12}", "Feature", "Importance");
        println!("{}", "-".repeat(34));
        for (name, &imp) in pairs {
            let bar_len = (imp * 40.0).max(0.0) as usize;
            let bar = "#".repeat(bar_len);
            println!("{:<20} {:>12.4} {}", name, imp, bar);
        }
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n\n=============================================================");
    println!("SUMMARY: Monotonic Constraints Benefits");
    println!("=============================================================");
    println!(
        "
FerroML's monotonic constraints provide:

  1. Domain Knowledge Integration:
     - Enforce known relationships (e.g., more debt = higher risk)
     - Prevent counterintuitive model behavior
     - Build trust with stakeholders

  2. Regulatory Compliance:
     - Required for many financial models
     - Ensures explainable, defensible predictions
     - Meets fairness requirements

  3. Robustness:
     - Reduces overfitting to noise
     - More stable predictions on new data
     - Better generalization

  4. Key Configuration Options:
     - MonotonicConstraint::Positive - feature effect must be non-decreasing
     - MonotonicConstraint::Negative - feature effect must be non-increasing
     - MonotonicConstraint::None - no constraint (default)

  5. Complementary Features:
     - Feature interaction constraints for additional control
     - Multiple loss functions (L2, L1, Huber)
     - Early stopping for optimal model complexity
     - Leaf-wise or depth-first growth strategies
"
    );

    Ok(())
}

/// Generate synthetic credit risk data with known monotonic relationships
fn generate_credit_risk_data(
    n_samples: usize,
    seed: u64,
) -> ferroml_core::Result<(Array2<f64>, Array1<f64>)> {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 0.2).unwrap();

    let mut x_data = Vec::with_capacity(n_samples * 4);
    let mut y_data = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // Features (all scaled to roughly [0, 1] range)
        let debt_to_income: f64 = rng.random(); // 0-1, higher = more debt
        let credit_history: f64 = rng.random(); // 0-1, higher = longer history
        let num_delinquencies: f64 = rng.random(); // 0-1, higher = more delinquencies
        let account_age: f64 = rng.random(); // 0-1, higher = older account

        // Target: risk score with known monotonic relationships
        // Higher debt_to_income -> higher risk (positive)
        // Higher credit_history -> lower risk (negative)
        // Higher num_delinquencies -> higher risk (positive)
        // Higher account_age -> lower risk (negative)
        let risk = 0.0
            + 2.0 * debt_to_income          // Positive effect
            - 1.5 * credit_history          // Negative effect
            + 2.5 * num_delinquencies       // Positive effect
            - 1.0 * account_age             // Negative effect
            + 0.5 * debt_to_income * num_delinquencies  // Interaction term
            + normal.sample(&mut rng); // Noise

        x_data.push(debt_to_income);
        x_data.push(credit_history);
        x_data.push(num_delinquencies);
        x_data.push(account_age);
        y_data.push(risk);
    }

    let x = Array2::from_shape_vec((n_samples, 4), x_data)
        .map_err(|e| ferroml_core::FerroError::InvalidInput(e.to_string()))?;
    let y = Array1::from_vec(y_data);

    Ok((x, y))
}

/// Print regression metrics
fn print_regression_metrics(
    name: &str,
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
) -> ferroml_core::Result<()> {
    let mse = mse(y_true, y_pred)?;
    let mae = mae(y_true, y_pred)?;
    let r2 = r2_score(y_true, y_pred)?;

    println!("{} Model Performance:", name);
    println!("  MSE:  {:.4}", mse);
    println!("  RMSE: {:.4}", mse.sqrt());
    println!("  MAE:  {:.4}", mae);
    println!("  R²:   {:.4}", r2);

    Ok(())
}

/// Check if model predictions are monotonic with respect to each feature
fn check_monotonicity(
    model: &HistGradientBoostingRegressor,
    x: &Array2<f64>,
) -> ferroml_core::Result<()> {
    let feature_names = [
        "debt_to_income",
        "credit_history",
        "num_delinquencies",
        "account_age",
    ];
    let expected_directions = ["Positive", "Negative", "Positive", "Negative"];

    for (feature_idx, (name, expected)) in feature_names
        .iter()
        .zip(expected_directions.iter())
        .enumerate()
    {
        // Create test points varying only this feature
        let n_points = 20;
        let mut violations = 0;
        let mut total_checks = 0;

        // Use the mean of other features
        let means: Vec<f64> = (0..4).map(|i| x.column(i).mean().unwrap_or(0.5)).collect();

        // Generate predictions at different feature values
        let mut prev_pred = None;
        for i in 0..n_points {
            let value = i as f64 / (n_points - 1) as f64;

            let mut point = Array2::zeros((1, 4));
            for j in 0..4 {
                if j == feature_idx {
                    point[[0, j]] = value;
                } else {
                    point[[0, j]] = means[j];
                }
            }

            let pred = model.predict(&point)?[0];

            if let Some(prev) = prev_pred {
                total_checks += 1;
                let should_increase = *expected == "Positive";
                let did_increase = pred >= prev;

                if should_increase != did_increase {
                    violations += 1;
                }
            }
            prev_pred = Some(pred);
        }

        let violation_rate = if total_checks > 0 {
            100.0 * violations as f64 / total_checks as f64
        } else {
            0.0
        };

        let status = if violations == 0 { "OK" } else { "VIOLATED" };
        println!(
            "  {:<20} Expected: {:<8} Violations: {:>2}/{:<2} ({:>5.1}%) [{}]",
            name, expected, violations, total_checks, violation_rate, status
        );
    }

    Ok(())
}

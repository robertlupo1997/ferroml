//! AutoML with Statistical Output
//!
//! This example demonstrates FerroML's AutoML system, showcasing:
//! - Automatic algorithm selection and hyperparameter optimization
//! - Statistical confidence intervals on model scores
//! - Model comparison with significance testing
//! - Ensemble construction from top models
//! - Aggregated feature importance across models
//!
//! Run with: `cargo run --example automl`

use ferroml_core::datasets::{load_diabetes, load_iris};
use ferroml_core::{AutoML, AutoMLConfig, Metric, Task};

fn main() -> ferroml_core::Result<()> {
    println!("=============================================================");
    println!("FerroML AutoML - Statistically Rigorous Model Selection Demo");
    println!("=============================================================\n");

    // =========================================================================
    // 1. Classification Example
    // =========================================================================
    println!("1. AUTOML FOR CLASSIFICATION");
    println!("============================\n");

    run_classification_example()?;

    println!("\n");

    // =========================================================================
    // 2. Regression Example
    // =========================================================================
    println!("2. AUTOML FOR REGRESSION");
    println!("========================\n");

    run_regression_example()?;

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=============================================================");
    println!("SUMMARY: FerroML AutoML Capabilities");
    println!("=============================================================");
    println!("
FerroML's AutoML provides statistically rigorous model selection:

  Algorithm Selection:
    - Intelligent portfolio of algorithms (Quick/Balanced/Thorough presets)
    - Data-aware search space adaptation
    - Bandit-based time budget allocation

  Statistical Rigor (Key Differentiator):
    - Confidence intervals on cross-validation scores
    - Corrected resampled t-test (Nadeau-Bengio) for model comparison
    - Holm-Bonferroni multiple testing correction
    - Identifies statistically competitive models

  Ensemble Learning:
    - Greedy ensemble selection from trial results
    - Diversity-weighted model selection
    - Coordinate descent weight optimization

  Feature Importance:
    - Aggregated importance across all successful models
    - Weighted by model performance
    - Confidence intervals for importance values

This goes beyond typical AutoML systems by providing statistical
guarantees and interpretability, not just \"best\" model selection.
");

    Ok(())
}

/// Run AutoML classification example using Iris dataset
fn run_classification_example() -> ferroml_core::Result<()> {
    // Load the Iris dataset
    let (dataset, info) = load_iris();

    println!("Dataset: {}", info.name);
    println!("Samples: {}", dataset.n_samples());
    println!("Features: {}", dataset.n_features());
    println!("Classes: {:?}", info.n_classes);
    println!("Feature names: {:?}", info.feature_names);

    // Split into train/test
    let (train_dataset, test_dataset) = dataset.train_test_split(0.2, true, Some(42))?;
    let (x_train, y_train) = train_dataset.into_arrays();
    let (x_test, _y_test) = test_dataset.into_arrays();

    println!("\nTrain samples: {}", x_train.nrows());
    println!("Test samples: {}", x_test.nrows());

    // -------------------------------------------------------------------------
    // Configure AutoML
    // -------------------------------------------------------------------------
    println!("\n--- Configuring AutoML ---");

    let config = AutoMLConfig {
        task: Task::Classification,
        metric: Metric::Accuracy,
        time_budget_seconds: 60, // 1 minute for demo
        cv_folds: 5,
        statistical_tests: true,
        confidence_level: 0.95,
        seed: Some(42),
        ..Default::default()
    };

    println!("Task: {:?}", config.task);
    println!("Metric: {:?}", config.metric);
    println!("Time budget: {}s", config.time_budget_seconds);
    println!("CV folds: {}", config.cv_folds);
    println!("Statistical tests: {}", config.statistical_tests);

    // -------------------------------------------------------------------------
    // Run AutoML
    // -------------------------------------------------------------------------
    println!("\n--- Running AutoML ---");
    println!("(This may take up to {} seconds...)\n", config.time_budget_seconds);

    let automl = AutoML::new(config);
    let result = automl.fit(&x_train, &y_train)?;

    // -------------------------------------------------------------------------
    // Display Results
    // -------------------------------------------------------------------------
    println!("--- AutoML Results ---\n");

    // Summary statistics
    println!("Run Statistics:");
    println!("  Total time: {:.2}s", result.total_time_seconds);
    println!("  Successful trials: {}", result.n_successful_trials);
    println!("  Failed trials: {}", result.n_failed_trials);
    println!("  Unique algorithms tried: {}", result.n_successful_algorithms());

    // Best model
    if let Some(best) = result.best_model() {
        println!("\nBest Model:");
        println!("  Algorithm: {:?}", best.algorithm);
        println!("  CV Score: {:.4} +/- {:.4}", best.cv_score, best.cv_std);
        println!("  95% CI: [{:.4}, {:.4}]", best.ci_lower, best.ci_upper);
        println!("  Training time: {:.2}s", best.training_time_seconds);
    }

    // Leaderboard with confidence intervals
    println!("\nLeaderboard (with 95% CIs):");
    println!("{:<5} {:<30} {:>12} {:>10} {:>24}",
             "Rank", "Algorithm", "CV Score", "Std", "95% CI");
    println!("{}", "-".repeat(85));

    for entry in result.leaderboard.iter().take(10) {
        println!("{:<5} {:<30} {:>12.4} {:>10.4} [{:>10.4}, {:>10.4}]",
                 entry.rank,
                 format!("{:?}", entry.algorithm),
                 entry.cv_score,
                 entry.cv_std,
                 entry.ci_lower,
                 entry.ci_upper);
    }

    // Statistical model comparisons
    if let Some(comparisons) = &result.model_comparisons {
        println!("\nModel Comparison (vs Best):");
        println!("  Correction method: {}", comparisons.correction_method);
        println!("  Corrected alpha: {:.4}", comparisons.corrected_alpha);
        println!("  Best significantly better than {} models", comparisons.n_significantly_worse);

        println!("\nPairwise Comparisons:");
        println!("{:<35} {:>10} {:>10} {:>10} {:>8}",
                 "Comparison", "Diff", "p-value", "Corrected", "Signif");
        println!("{}", "-".repeat(75));

        for comp in &comparisons.pairwise_comparisons {
            let sig = if comp.significant_corrected { "**" }
                     else if comp.significant { "*" }
                     else { "" };
            println!("{:<35} {:>10.4} {:>10.4} {:>10.4} {:>8}",
                     format!("{} vs {}", comp.model1_name, comp.model2_name),
                     comp.mean_difference,
                     comp.p_value,
                     comp.p_value_corrected,
                     sig);
        }
        println!("\n  * significant at alpha=0.05");
        println!("  ** significant after Holm-Bonferroni correction");
    }

    // Competitive models (not significantly worse than best)
    let competitive = result.competitive_models();
    if !competitive.is_empty() {
        println!("\nStatistically Competitive Models:");
        println!("  (These are not significantly worse than the best model)");
        for model in &competitive {
            println!("    - {:?} (score: {:.4})", model.algorithm, model.cv_score);
        }
    }

    // Ensemble results
    if let Some(ensemble) = &result.ensemble {
        println!("\nEnsemble:");
        println!("  Ensemble score: {:.4}", ensemble.ensemble_score);
        println!("  Improvement over best: {:.4} ({:.2}%)",
                 ensemble.improvement,
                 if result.best_model().map_or(0.0, |b| b.cv_score) != 0.0 {
                     (ensemble.improvement / result.best_model().unwrap().cv_score.abs()) * 100.0
                 } else { 0.0 });
        println!("  Number of models in ensemble: {}", ensemble.members.len());
        println!("\n  Ensemble members:");
        for member in &ensemble.members {
            println!("    - {:?} (weight: {:.4})", member.algorithm, member.weight);
        }
    }

    // Feature importance
    if let Some(importance) = &result.aggregated_importance {
        println!("\nAggregated Feature Importance:");
        println!("  (Weighted across {} models)", importance.n_models);
        println!("{:<20} {:>12} {:>10} {:>24}",
                 "Feature", "Importance", "Std", "95% CI");
        println!("{}", "-".repeat(70));

        let feature_names = &info.feature_names;
        for idx in importance.sorted_indices() {
            let name = feature_names.get(idx)
                .map(String::as_str)
                .unwrap_or(&importance.feature_names[idx]);
            let sig = if importance.is_significant(idx) { "*" } else { "" };
            println!("{:<20} {:>12.4} {:>10.4} [{:>10.4}, {:>10.4}]{}",
                     name,
                     importance.importance_mean[idx],
                     importance.importance_std[idx],
                     importance.ci_lower[idx],
                     importance.ci_upper[idx],
                     sig);
        }
        println!("\n  * indicates CI excludes zero (statistically significant)");
    }

    // Print the full summary
    println!("\n--- Full Summary ---");
    println!("{}", result.summary());

    // -------------------------------------------------------------------------
    // Evaluate on test set (using best model's algorithm)
    // -------------------------------------------------------------------------
    println!("\n--- Test Set Evaluation ---");
    println!("Note: In production, you would use the fitted ensemble or best model.");
    println!("This demo shows the AutoML result structure, not model persistence.");

    Ok(())
}

/// Run AutoML regression example using Diabetes dataset
fn run_regression_example() -> ferroml_core::Result<()> {
    // Load the Diabetes dataset
    let (dataset, info) = load_diabetes();

    println!("Dataset: {}", info.name);
    println!("Samples: {}", dataset.n_samples());
    println!("Features: {}", dataset.n_features());

    // Split into train/test
    let (train_dataset, test_dataset) = dataset.train_test_split(0.2, true, Some(42))?;
    let (x_train, y_train) = train_dataset.into_arrays();
    let (_x_test, _y_test) = test_dataset.into_arrays();

    println!("\nTrain samples: {}", x_train.nrows());
    println!("Test samples: {}", _x_test.nrows());

    // -------------------------------------------------------------------------
    // Configure AutoML for Regression
    // -------------------------------------------------------------------------
    println!("\n--- Configuring AutoML ---");

    let config = AutoMLConfig {
        task: Task::Regression,
        metric: Metric::R2,
        time_budget_seconds: 60, // 1 minute for demo
        cv_folds: 5,
        statistical_tests: true,
        confidence_level: 0.95,
        seed: Some(42),
        ..Default::default()
    };

    println!("Task: {:?}", config.task);
    println!("Metric: {:?}", config.metric);
    println!("Time budget: {}s", config.time_budget_seconds);

    // -------------------------------------------------------------------------
    // Run AutoML
    // -------------------------------------------------------------------------
    println!("\n--- Running AutoML ---");
    println!("(This may take up to {} seconds...)\n", config.time_budget_seconds);

    let automl = AutoML::new(config);
    let result = automl.fit(&x_train, &y_train)?;

    // -------------------------------------------------------------------------
    // Display Results
    // -------------------------------------------------------------------------
    println!("--- AutoML Results ---\n");

    // Summary statistics
    println!("Run Statistics:");
    println!("  Total time: {:.2}s", result.total_time_seconds);
    println!("  Successful trials: {}", result.n_successful_trials);
    println!("  Failed trials: {}", result.n_failed_trials);
    println!("  Unique algorithms tried: {}", result.n_successful_algorithms());

    // Best model
    if let Some(best) = result.best_model() {
        println!("\nBest Model:");
        println!("  Algorithm: {:?}", best.algorithm);
        println!("  CV R²: {:.4} +/- {:.4}", best.cv_score, best.cv_std);
        println!("  95% CI: [{:.4}, {:.4}]", best.ci_lower, best.ci_upper);
        println!("  Training time: {:.2}s", best.training_time_seconds);
    }

    // Leaderboard
    println!("\nTop 5 Models (Leaderboard):");
    println!("{:<5} {:<30} {:>12} {:>10} {:>24}",
             "Rank", "Algorithm", "CV R²", "Std", "95% CI");
    println!("{}", "-".repeat(85));

    for entry in result.leaderboard.iter().take(5) {
        println!("{:<5} {:<30} {:>12.4} {:>10.4} [{:>10.4}, {:>10.4}]",
                 entry.rank,
                 format!("{:?}", entry.algorithm),
                 entry.cv_score,
                 entry.cv_std,
                 entry.ci_lower,
                 entry.ci_upper);
    }

    // Data characteristics detected
    println!("\nData Characteristics Detected:");
    let chars = &result.data_characteristics;
    println!("  Samples: {}", chars.n_samples);
    println!("  Features: {}", chars.n_features);
    println!("  Has missing values: {}", chars.has_missing_values);
    println!("  Variance ratio (max/min): {:.2}", chars.feature_variance_ratio);

    // Model comparisons for regression
    if let Some(comparisons) = &result.model_comparisons {
        println!("\nModel Significance Testing:");
        println!("  {} pairwise comparisons performed", comparisons.pairwise_comparisons.len());
        println!("  Best model significantly better than {} others", comparisons.n_significantly_worse);

        let competitive_count = result.competitive_models().len();
        println!("  {} models statistically competitive with best", competitive_count);
    }

    // Ensemble
    if let Some(ensemble) = &result.ensemble {
        println!("\nEnsemble Performance:");
        println!("  Score: {:.4}", ensemble.ensemble_score);
        println!("  Members: {}", ensemble.members.len());
        println!("  Improvement: {:.4}", ensemble.improvement);
    }

    // Print compact summary
    println!("\n--- Compact Summary ---");
    println!("{}", result.summary());

    Ok(())
}

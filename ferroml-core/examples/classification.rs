//! Classification Workflow with Full Statistical Diagnostics
//!
//! This example demonstrates FerroML's comprehensive classification capabilities,
//! showing how to train, evaluate, and compare classification models with
//! statistical rigor.
//!
//! Run with: `cargo run --example classification`

use ferroml_core::datasets::load_iris;
use ferroml_core::metrics::{
    accuracy, balanced_accuracy, confusion_matrix, f1_score, matthews_corrcoef, precision, recall,
    roc_auc_score, Average,
};
use ferroml_core::models::forest::{MaxFeatures, RandomForestClassifier};
use ferroml_core::models::logistic::LogisticRegression;
use ferroml_core::models::tree::DecisionTreeClassifier;
use ferroml_core::models::{Model, ProbabilisticModel, StatisticalModel};
use ndarray::Array1;

fn main() -> ferroml_core::Result<()> {
    println!("=============================================================");
    println!("FerroML Classification - Full Workflow Demo");
    println!("=============================================================\n");

    // =========================================================================
    // 1. Load and Prepare Data
    // =========================================================================
    println!("1. LOADING AND PREPARING DATA");
    println!("-----------------------------");

    let (dataset, info) = load_iris();
    println!("Dataset: {}", info.name);
    println!("Samples: {}", dataset.n_samples());
    println!("Features: {}", dataset.n_features());
    println!("Classes: {:?}", info.n_classes);
    println!("Feature names: {:?}", info.feature_names);
    println!("Target names: {:?}", info.target_names);

    // Split into train/test using Dataset's method
    let (train_dataset, test_dataset) = dataset.train_test_split(0.3, true, Some(42))?;

    println!("\nTrain samples: {}", train_dataset.n_samples());
    println!("Test samples: {}", test_dataset.n_samples());

    // Convert to arrays
    let (x_train, y_train) = train_dataset.into_arrays();
    let (x_test, y_test) = test_dataset.into_arrays();

    // For binary classification demo, let's convert to binary (setosa vs rest)
    let y_train_binary = y_train.mapv(|y| if y == 0.0 { 1.0 } else { 0.0 });
    let y_test_binary = y_test.mapv(|y| if y == 0.0 { 1.0 } else { 0.0 });

    // =========================================================================
    // 2. Logistic Regression with Full Statistical Diagnostics
    // =========================================================================
    println!("\n\n2. LOGISTIC REGRESSION (Binary: Setosa vs Rest)");
    println!("------------------------------------------------");

    let feature_names: Vec<String> = if info.feature_names.is_empty() {
        (0..x_train.ncols()).map(|i| format!("X{}", i)).collect()
    } else {
        info.feature_names.clone()
    };

    let mut logistic = LogisticRegression::new()
        .with_confidence_level(0.95)
        .with_feature_names(feature_names.clone());

    logistic.fit(&x_train, &y_train_binary)?;

    // Print summary
    println!("\nModel Summary:");
    println!("{}", logistic.summary());

    // Odds Ratios (key for interpretation)
    println!("\nOdds Ratios with 95% CI:");
    println!(
        "{:<20} {:>12} {:>12} {:>12}",
        "Variable", "Odds Ratio", "CI Lower", "CI Upper"
    );
    println!("{}", "-".repeat(60));

    let odds_ratios = logistic.odds_ratios_with_ci(0.95);
    for or in &odds_ratios {
        println!(
            "{:<20} {:>12.4} {:>12.4} {:>12.4}",
            or.name, or.odds_ratio, or.ci_lower, or.ci_upper
        );
    }

    println!("\nInterpretation:");
    println!("  - Odds ratio > 1: Higher values increase odds of being Setosa");
    println!("  - Odds ratio < 1: Higher values decrease odds of being Setosa");

    // Model fit statistics
    println!("\nModel Fit Statistics:");
    if let Some(pseudo_r2) = logistic.pseudo_r_squared() {
        println!("  McFadden's Pseudo R²: {:.4}", pseudo_r2);
        if pseudo_r2 > 0.4 {
            println!("  -> Excellent fit (pseudo R² > 0.4)");
        } else if pseudo_r2 > 0.2 {
            println!("  -> Good fit (0.2 < pseudo R² < 0.4)");
        } else {
            println!("  -> Moderate fit (pseudo R² < 0.2)");
        }
    }

    if let Some((lr_stat, p_value)) = logistic.likelihood_ratio_test() {
        println!(
            "  Likelihood Ratio Test: chi² = {:.4}, p = {:.6}",
            lr_stat, p_value
        );
        if p_value < 0.05 {
            println!("  -> Model is significant at alpha=0.05");
        }
    }

    if let Some(aic) = logistic.aic() {
        println!("  AIC: {:.4}", aic);
    }
    if let Some(bic) = logistic.bic() {
        println!("  BIC: {:.4}", bic);
    }

    // Predictions and evaluation
    let y_pred_logistic = logistic.predict(&x_test)?;
    let y_proba_logistic = logistic.predict_proba(&x_test)?;
    // For binary classification, predict_proba may return 1 or 2 columns
    // If 1 column: it's P(y=1), If 2 columns: second is P(y=1)
    let y_proba_positive: Array1<f64> = if y_proba_logistic.ncols() == 1 {
        y_proba_logistic.column(0).to_owned()
    } else {
        y_proba_logistic.column(1).to_owned()
    };

    println!("\nTest Set Evaluation:");
    print_binary_metrics(&y_test_binary, &y_pred_logistic, &y_proba_positive)?;

    // =========================================================================
    // 3. Random Forest Classifier
    // =========================================================================
    println!("\n\n3. RANDOM FOREST CLASSIFIER (Multiclass)");
    println!("-----------------------------------------");

    let mut rf = RandomForestClassifier::new()
        .with_n_estimators(100)
        .with_max_depth(Some(5))
        .with_max_features(Some(MaxFeatures::Sqrt))
        .with_random_state(42)
        .with_oob_score(true);

    rf.fit(&x_train, &y_train)?;

    // OOB Error
    if let Some(oob_score) = rf.oob_score() {
        println!("Out-of-Bag Score (accuracy): {:.4}", oob_score);
        println!("  -> Validation without held-out set!");
    }

    // Feature importance with confidence intervals
    println!("\nFeature Importance (with CIs):");
    if let Some(importance_with_ci) = rf.feature_importances_with_ci() {
        println!(
            "{:<20} {:>12} {:>10} {:>12} {:>12}",
            "Feature", "Importance", "Std Error", "CI Lower", "CI Upper"
        );
        println!("{}", "-".repeat(70));

        let mut indices: Vec<usize> = (0..importance_with_ci.importance.len()).collect();
        indices.sort_by(|&a, &b| {
            importance_with_ci.importance[b]
                .partial_cmp(&importance_with_ci.importance[a])
                .unwrap()
        });

        for &i in &indices {
            let name = feature_names.get(i).map(String::as_str).unwrap_or("?");
            println!(
                "{:<20} {:>12.4} {:>10.4} {:>12.4} {:>12.4}",
                name,
                importance_with_ci.importance[i],
                importance_with_ci.std_error[i],
                importance_with_ci.ci_lower[i],
                importance_with_ci.ci_upper[i]
            );
        }
    }

    // Multiclass predictions
    let y_pred_rf = rf.predict(&x_test)?;

    println!("\nTest Set Evaluation (Multiclass):");
    print_multiclass_metrics(&y_test, &y_pred_rf)?;

    // =========================================================================
    // 4. Decision Tree Classifier
    // =========================================================================
    println!("\n\n4. DECISION TREE CLASSIFIER");
    println!("---------------------------");

    let mut tree = DecisionTreeClassifier::new()
        .with_max_depth(Some(4))
        .with_min_samples_split(5);

    tree.fit(&x_train, &y_train)?;

    println!("Tree Statistics:");
    println!("  Max Depth: {:?}", tree.max_depth);
    println!("  Min Samples Split: {}", tree.min_samples_split);

    // Feature importance
    if let Some(importance) = tree.feature_importance() {
        println!("\nFeature Importance (Gini):");
        let mut pairs: Vec<_> = feature_names.iter().zip(importance.iter()).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        for (name, &imp) in pairs {
            let bar_len = (imp * 40.0) as usize;
            let bar = "#".repeat(bar_len);
            println!("  {:<20} {:>8.4} {}", name, imp, bar);
        }
    }

    let y_pred_tree = tree.predict(&x_test)?;

    println!("\nTest Set Evaluation:");
    print_multiclass_metrics(&y_test, &y_pred_tree)?;

    // =========================================================================
    // 5. Model Comparison
    // =========================================================================
    println!("\n\n5. MODEL COMPARISON");
    println!("-------------------");

    let acc_logistic = accuracy(&y_test_binary, &y_pred_logistic)?;
    let acc_rf = accuracy(&y_test, &y_pred_rf)?;
    let acc_tree = accuracy(&y_test, &y_pred_tree)?;

    println!(
        "{:<25} {:>15} {:>15}",
        "Model", "Task", "Accuracy"
    );
    println!("{}", "-".repeat(55));
    println!(
        "{:<25} {:>15} {:>15.4}",
        "Logistic Regression", "Binary", acc_logistic
    );
    println!(
        "{:<25} {:>15} {:>15.4}",
        "Random Forest (100 trees)", "Multiclass", acc_rf
    );
    println!(
        "{:<25} {:>15} {:>15.4}",
        "Decision Tree", "Multiclass", acc_tree
    );

    // =========================================================================
    // 6. Confusion Matrix Analysis
    // =========================================================================
    println!("\n\n6. CONFUSION MATRIX ANALYSIS (Random Forest)");
    println!("---------------------------------------------");

    let cm = confusion_matrix(&y_test, &y_pred_rf)?;
    let target_names = info.target_names.as_ref();

    println!("\nConfusion Matrix:");
    print!("{:>12}", "");
    for j in 0..cm.labels.len() {
        let label_str = format!("{}", cm.labels[j]);
        let name = target_names
            .and_then(|n| n.get(j))
            .map(|s| s.as_str())
            .unwrap_or(&label_str);
        print!("{:>12}", name);
    }
    println!();

    for i in 0..cm.labels.len() {
        let label_str = format!("{}", cm.labels[i]);
        let name = target_names
            .and_then(|n| n.get(i))
            .map(|s| s.as_str())
            .unwrap_or(&label_str);
        print!("{:>12}", name);
        for j in 0..cm.labels.len() {
            print!("{:>12}", cm.matrix[[i, j]]);
        }
        println!();
    }

    println!("\nPer-Class Metrics:");
    let tp = cm.true_positives();
    let fp = cm.false_positives();
    let fn_ = cm.false_negatives();
    let support = cm.support();

    println!(
        "{:<15} {:>10} {:>10} {:>10} {:>10}",
        "Class", "Precision", "Recall", "F1", "Support"
    );
    println!("{}", "-".repeat(60));

    for i in 0..cm.labels.len() {
        let prec = if tp[i] + fp[i] > 0 {
            tp[i] as f64 / (tp[i] + fp[i]) as f64
        } else {
            0.0
        };
        let rec = if tp[i] + fn_[i] > 0 {
            tp[i] as f64 / (tp[i] + fn_[i]) as f64
        } else {
            0.0
        };
        let f1 = if prec + rec > 0.0 {
            2.0 * prec * rec / (prec + rec)
        } else {
            0.0
        };

        let label_str = format!("{}", cm.labels[i]);
        let name = target_names
            .and_then(|n| n.get(i))
            .map(|s| s.as_str())
            .unwrap_or(&label_str);

        println!(
            "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10}",
            name, prec, rec, f1, support[i]
        );
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=============================================================");
    println!("SUMMARY: FerroML Classification Capabilities");
    println!("=============================================================");
    println!(
        "
FerroML provides comprehensive classification with:

  Logistic Regression:
    - Coefficient inference (z-stats, p-values, CIs)
    - Odds ratios with confidence intervals
    - Model fit: Pseudo R², AIC, BIC, likelihood ratio test
    - ROC-AUC with bootstrap CI

  Random Forest:
    - Out-of-bag error estimation (no held-out set needed!)
    - Feature importance with bootstrap confidence intervals
    - Parallel training via rayon

  Evaluation Metrics:
    - Confusion matrix with derived metrics
    - Accuracy, Balanced Accuracy
    - Precision, Recall, F1 (micro/macro/weighted)
    - Matthews Correlation Coefficient
    - ROC-AUC with CI

This statistical rigor goes beyond sklearn's basic functionality,
providing insights needed for production ML systems.
"
    );

    Ok(())
}

/// Print binary classification metrics
fn print_binary_metrics(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    y_proba: &Array1<f64>,
) -> ferroml_core::Result<()> {
    let acc = accuracy(y_true, y_pred)?;
    let bal_acc = balanced_accuracy(y_true, y_pred)?;
    // For binary, Micro averaging gives the same as binary-specific metrics
    let prec = precision(y_true, y_pred, Average::Micro)?;
    let rec = recall(y_true, y_pred, Average::Micro)?;
    let f1 = f1_score(y_true, y_pred, Average::Micro)?;
    let mcc = matthews_corrcoef(y_true, y_pred)?;
    let roc_auc = roc_auc_score(y_true, y_proba)?;

    println!("  Accuracy:          {:.4}", acc);
    println!("  Balanced Accuracy: {:.4}", bal_acc);
    println!("  Precision:         {:.4}", prec);
    println!("  Recall:            {:.4}", rec);
    println!("  F1 Score:          {:.4}", f1);
    println!("  MCC:               {:.4}", mcc);
    println!("  ROC-AUC:           {:.4}", roc_auc);

    Ok(())
}

/// Print multiclass classification metrics
fn print_multiclass_metrics(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
) -> ferroml_core::Result<()> {
    let acc = accuracy(y_true, y_pred)?;
    let bal_acc = balanced_accuracy(y_true, y_pred)?;
    let prec_macro = precision(y_true, y_pred, Average::Macro)?;
    let rec_macro = recall(y_true, y_pred, Average::Macro)?;
    let f1_macro = f1_score(y_true, y_pred, Average::Macro)?;
    let f1_weighted = f1_score(y_true, y_pred, Average::Weighted)?;
    let mcc = matthews_corrcoef(y_true, y_pred)?;

    println!("  Accuracy:          {:.4}", acc);
    println!("  Balanced Accuracy: {:.4}", bal_acc);
    println!("  Precision (macro): {:.4}", prec_macro);
    println!("  Recall (macro):    {:.4}", rec_macro);
    println!("  F1 (macro):        {:.4}", f1_macro);
    println!("  F1 (weighted):     {:.4}", f1_weighted);
    println!("  MCC:               {:.4}", mcc);

    Ok(())
}

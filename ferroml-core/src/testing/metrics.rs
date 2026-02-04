//! Phase 28: Metrics Tests
//!
//! Comprehensive tests for evaluation metrics including:
//! - Multi-class classification metrics (per-class precision, recall, F1)
//! - Calibration curve analysis (ECE, MCE, reliability diagrams)
//! - Custom scorers and the Metric trait
//! - Confusion matrix analysis for multi-class
//! - ROC/PR curve edge cases
//! - Regression metric edge cases

#[cfg(test)]
mod multiclass_metrics_tests {
    use crate::metrics::classification::{
        accuracy, balanced_accuracy, cohen_kappa_score, f1_score, matthews_corrcoef, precision,
        recall, ClassificationReport,
    };
    use crate::metrics::Average;
    use ndarray::Array1;

    // ==================== Per-class Metrics via ClassificationReport ====================

    #[test]
    fn test_classification_report_per_class_binary() {
        // Binary classification with known per-class metrics
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        // Class 0: TP=2, FP=1, FN=1 -> P=2/3, R=2/3, F1=2/3
        // Class 1: TP=2, FP=1, FN=1 -> P=2/3, R=2/3, F1=2/3

        let report = ClassificationReport::compute(&y_true, &y_pred).unwrap();

        assert_eq!(report.labels.len(), 2);
        assert!((report.precision[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((report.precision[1] - 2.0 / 3.0).abs() < 1e-10);
        assert!((report.recall[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((report.recall[1] - 2.0 / 3.0).abs() < 1e-10);
        assert!((report.f1[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((report.f1[1] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_classification_report_per_class_multiclass() {
        // 3-class classification
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 2.0, 0.0]);
        // Class 0: TP=1, FP=1, FN=1 -> P=1/2, R=1/2, F1=1/2
        // Class 1: TP=1, FP=1, FN=1 -> P=1/2, R=1/2, F1=1/2
        // Class 2: TP=1, FP=1, FN=1 -> P=1/2, R=1/2, F1=1/2

        let report = ClassificationReport::compute(&y_true, &y_pred).unwrap();

        assert_eq!(report.labels.len(), 3);
        for i in 0..3 {
            assert!(
                (report.precision[i] - 0.5).abs() < 1e-10,
                "Class {} precision wrong",
                i
            );
            assert!(
                (report.recall[i] - 0.5).abs() < 1e-10,
                "Class {} recall wrong",
                i
            );
            assert!((report.f1[i] - 0.5).abs() < 1e-10, "Class {} F1 wrong", i);
        }
    }

    #[test]
    fn test_classification_report_imbalanced_classes() {
        // Imbalanced: 5 class 0, 2 class 1, 3 class 2
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0]);

        let report = ClassificationReport::compute(&y_true, &y_pred).unwrap();

        // Class 0: TP=4, FP=1 (from class 2), FN=1 (pred as 1)
        // P=4/5=0.8, R=4/5=0.8
        assert!((report.precision[0] - 0.8).abs() < 1e-10);
        assert!((report.recall[0] - 0.8).abs() < 1e-10);

        // Support should reflect class sizes
        assert_eq!(report.support[0], 5);
        assert_eq!(report.support[1], 2);
        assert_eq!(report.support[2], 3);
    }

    #[test]
    fn test_classification_report_macro_vs_weighted() {
        // Imbalanced data where macro != weighted
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]); // All predict 0

        let report = ClassificationReport::compute(&y_true, &y_pred).unwrap();

        // Class 0: P=4/5=0.8, R=4/4=1.0, F1=2*0.8*1/(0.8+1)=0.889
        // Class 1: P=0/0=0, R=0/1=0, F1=0
        // Macro: (0.8+0)/2=0.4 for precision
        // Weighted: 0.8*4/5 + 0*1/5 = 0.64 for precision
        assert!((report.macro_precision - 0.4).abs() < 1e-10);
        assert!((report.weighted_precision - 0.64).abs() < 1e-10);
    }

    #[test]
    fn test_classification_report_summary_format() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let report = ClassificationReport::compute(&y_true, &y_pred).unwrap();
        let summary = report.summary();

        // Summary should contain key elements
        assert!(summary.contains("precision"));
        assert!(summary.contains("recall"));
        assert!(summary.contains("f1-score"));
        assert!(summary.contains("support"));
        assert!(summary.contains("accuracy"));
        assert!(summary.contains("macro avg"));
        assert!(summary.contains("weighted avg"));
    }

    // ==================== Averaging Strategies ====================

    #[test]
    fn test_precision_all_averaging_modes() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

        let macro_p = precision(&y_true, &y_pred, Average::Macro).unwrap();
        let micro_p = precision(&y_true, &y_pred, Average::Micro).unwrap();
        let weighted_p = precision(&y_true, &y_pred, Average::Weighted).unwrap();

        // All should be valid values
        assert!(macro_p >= 0.0 && macro_p <= 1.0);
        assert!(micro_p >= 0.0 && micro_p <= 1.0);
        assert!(weighted_p >= 0.0 && weighted_p <= 1.0);

        // Micro should equal accuracy for multiclass
        let acc = accuracy(&y_true, &y_pred).unwrap();
        assert!((micro_p - acc).abs() < 1e-10);
    }

    #[test]
    fn test_recall_all_averaging_modes() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

        let macro_r = recall(&y_true, &y_pred, Average::Macro).unwrap();
        let micro_r = recall(&y_true, &y_pred, Average::Micro).unwrap();
        let weighted_r = recall(&y_true, &y_pred, Average::Weighted).unwrap();

        assert!(macro_r >= 0.0 && macro_r <= 1.0);
        assert!(micro_r >= 0.0 && micro_r <= 1.0);
        assert!(weighted_r >= 0.0 && weighted_r <= 1.0);

        // Balanced accuracy equals macro recall
        let ba = balanced_accuracy(&y_true, &y_pred).unwrap();
        assert!((macro_r - ba).abs() < 1e-10);
    }

    #[test]
    fn test_f1_all_averaging_modes() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

        let macro_f1 = f1_score(&y_true, &y_pred, Average::Macro).unwrap();
        let micro_f1 = f1_score(&y_true, &y_pred, Average::Micro).unwrap();
        let weighted_f1 = f1_score(&y_true, &y_pred, Average::Weighted).unwrap();

        assert!(macro_f1 >= 0.0 && macro_f1 <= 1.0);
        assert!(micro_f1 >= 0.0 && micro_f1 <= 1.0);
        assert!(weighted_f1 >= 0.0 && weighted_f1 <= 1.0);
    }

    #[test]
    fn test_average_none_returns_error() {
        let y_true = Array1::from_vec(vec![0.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 1.0]);

        // Average::None should return error for scalar functions
        assert!(precision(&y_true, &y_pred, Average::None).is_err());
        assert!(recall(&y_true, &y_pred, Average::None).is_err());
        assert!(f1_score(&y_true, &y_pred, Average::None).is_err());
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_metrics_with_unseen_class_in_predictions() {
        // y_pred has class 2 that's not in y_true
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 2.0, 1.0, 2.0]);

        let report = ClassificationReport::compute(&y_true, &y_pred).unwrap();

        // Should handle class 2 appearing only in predictions
        assert_eq!(report.labels.len(), 3);
    }

    #[test]
    fn test_metrics_with_unseen_class_in_true() {
        // y_true has class 2 that model never predicts
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let report = ClassificationReport::compute(&y_true, &y_pred).unwrap();

        // Class 2 recall should be 0 (never correctly predicted)
        let class_2_idx = report.labels.iter().position(|&l| l == 2).unwrap();
        assert!((report.recall[class_2_idx] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_vs_kappa_relationship() {
        // Both MCC and Kappa are correlation metrics, but computed differently
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0]);

        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        let kappa = cohen_kappa_score(&y_true, &y_pred).unwrap();

        // Both should be positive for this reasonably good classifier
        assert!(mcc > 0.0);
        assert!(kappa > 0.0);

        // They should have the same sign
        assert!((mcc > 0.0) == (kappa > 0.0));
    }
}

#[cfg(test)]
mod confusion_matrix_tests {
    use crate::metrics::classification::ConfusionMatrix;
    use ndarray::Array1;

    #[test]
    fn test_confusion_matrix_multiclass_structure() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 2.0, 0.0]);

        let cm = ConfusionMatrix::compute(&y_true, &y_pred).unwrap();

        // Should be 3x3
        assert_eq!(cm.matrix.nrows(), 3);
        assert_eq!(cm.matrix.ncols(), 3);

        // Diagonal = correct predictions
        assert_eq!(cm.matrix[[0, 0]], 1); // Class 0 correct
        assert_eq!(cm.matrix[[1, 1]], 1); // Class 1 correct
        assert_eq!(cm.matrix[[2, 2]], 1); // Class 2 correct
    }

    #[test]
    fn test_confusion_matrix_row_col_sums() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 0.0]);

        let cm = ConfusionMatrix::compute(&y_true, &y_pred).unwrap();

        // Row sums = actual class counts
        let row_0_sum: usize = cm.matrix.row(0).sum();
        let row_1_sum: usize = cm.matrix.row(1).sum();
        assert_eq!(row_0_sum, 3); // 3 actual class 0
        assert_eq!(row_1_sum, 2); // 2 actual class 1

        // Column sums = predicted class counts
        let col_0_sum: usize = cm.matrix.column(0).sum();
        let col_1_sum: usize = cm.matrix.column(1).sum();
        assert_eq!(col_0_sum, 3); // 3 predicted class 0
        assert_eq!(col_1_sum, 2); // 2 predicted class 1

        // Total should equal n_samples
        assert_eq!(cm.total(), 5);
    }

    #[test]
    fn test_confusion_matrix_tp_fp_fn_tn() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 1.0, 1.0, 1.0, 0.0]);
        // TN=1, FP=1, FN=1, TP=2 for class 1

        let cm = ConfusionMatrix::compute(&y_true, &y_pred).unwrap();

        let tp = cm.true_positives();
        let fp = cm.false_positives();
        let fn_ = cm.false_negatives();
        let tn = cm.true_negatives();

        // For class 1 (index 1)
        assert_eq!(tp[1], 2);
        assert_eq!(fp[1], 1);
        assert_eq!(fn_[1], 1);
        assert_eq!(tn[1], 1);

        // TP + FP + FN + TN = n_samples for each class
        for i in 0..cm.labels.len() {
            assert_eq!(tp[i] + fp[i] + fn_[i] + tn[i], 5);
        }
    }

    #[test]
    fn test_confusion_matrix_support() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0]);

        let cm = ConfusionMatrix::compute(&y_true, &y_pred).unwrap();
        let support = cm.support();

        assert_eq!(support[0], 3); // 3 samples of class 0
        assert_eq!(support[1], 1); // 1 sample of class 1
        assert_eq!(support[2], 2); // 2 samples of class 2
    }

    #[test]
    fn test_confusion_matrix_many_classes() {
        // 5-class problem
        let y_true = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
        let y_pred = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 0.0]);

        let cm = ConfusionMatrix::compute(&y_true, &y_pred).unwrap();

        assert_eq!(cm.labels.len(), 5);
        assert_eq!(cm.matrix.nrows(), 5);
        assert_eq!(cm.matrix.ncols(), 5);

        // 5 correct predictions on diagonal
        let diagonal_sum: usize = (0..5).map(|i| cm.matrix[[i, i]]).sum();
        assert_eq!(diagonal_sum, 5);
    }

    #[test]
    fn test_confusion_matrix_non_contiguous_labels() {
        // Labels are 0, 5, 10 (non-contiguous)
        let y_true = Array1::from_vec(vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0]);
        let y_pred = Array1::from_vec(vec![0.0, 5.0, 5.0, 10.0, 10.0, 0.0]);

        let cm = ConfusionMatrix::compute(&y_true, &y_pred).unwrap();

        // Should have 3 unique labels
        assert_eq!(cm.labels.len(), 3);
        assert!(cm.labels.contains(&0));
        assert!(cm.labels.contains(&5));
        assert!(cm.labels.contains(&10));
    }
}

#[cfg(test)]
mod calibration_curve_tests {
    use crate::models::calibration::calibration_curve;
    use ndarray::Array1;

    #[test]
    fn test_calibration_curve_well_calibrated() {
        // Well-calibrated predictions: predicted proba ≈ actual frequency
        let y_true = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 10 negatives
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 10 positives
        ]);
        // Predictions roughly match true frequencies
        let y_prob = Array1::from_vec(vec![
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.48, // Low for negatives
            0.52, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, // High for positives
        ]);

        let result = calibration_curve(&y_true, &y_prob, 10).unwrap();

        // ECE should be low for well-calibrated predictions
        assert!(
            result.ece < 0.3,
            "ECE too high for well-calibrated: {}",
            result.ece
        );
        // MCE should also be reasonable
        assert!(result.mce < 0.5, "MCE too high: {}", result.mce);
    }

    #[test]
    fn test_calibration_curve_overconfident() {
        // Overconfident predictions: always predicts near 0 or 1
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let y_prob = Array1::from_vec(vec![
            0.01, 0.02, 0.03, 0.04, 0.05, 0.95, 0.96, 0.97, 0.98, 0.99,
        ]);

        let result = calibration_curve(&y_true, &y_prob, 10).unwrap();

        // Brier score should be very low (good predictions)
        assert!(result.brier_score < 0.01);
    }

    #[test]
    fn test_calibration_curve_underconfident() {
        // Underconfident predictions: predictions cluster around 0.5
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let y_prob = Array1::from_vec(vec![
            0.45, 0.47, 0.48, 0.49, 0.50, 0.50, 0.51, 0.52, 0.53, 0.55,
        ]);

        let result = calibration_curve(&y_true, &y_prob, 10).unwrap();

        // ECE should be moderate (predictions don't match extremes)
        assert!(result.ece > 0.0);
    }

    #[test]
    fn test_calibration_curve_bin_counts() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let y_prob = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]);

        let result = calibration_curve(&y_true, &y_prob, 10).unwrap();

        // Total bin counts should equal n_samples
        let total_count: usize = result.bin_counts.iter().sum();
        assert_eq!(total_count, 10);
    }

    #[test]
    fn test_calibration_curve_bin_edges() {
        let y_true = Array1::from_vec(vec![0.0, 1.0]);
        let y_prob = Array1::from_vec(vec![0.3, 0.7]);

        let result = calibration_curve(&y_true, &y_prob, 5).unwrap();

        // Should have n_bins + 1 edges
        assert_eq!(result.bin_edges.len(), 6);
        // First edge should be 0
        assert!((result.bin_edges[0] - 0.0).abs() < 1e-10);
        // Last edge should be 1
        assert!((result.bin_edges[5] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_calibration_curve_ece_bounds() {
        // ECE is weighted absolute error, should be in [0, 1]
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_prob = Array1::from_vec(vec![0.9, 0.8, 0.2, 0.1]); // Completely wrong

        let result = calibration_curve(&y_true, &y_prob, 4).unwrap();

        assert!(result.ece >= 0.0 && result.ece <= 1.0);
        assert!(result.mce >= 0.0 && result.mce <= 1.0);
    }

    #[test]
    fn test_calibration_curve_empty_bins() {
        // Some bins will be empty
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_prob = Array1::from_vec(vec![0.1, 0.15, 0.85, 0.9]); // All in extreme bins

        let result = calibration_curve(&y_true, &y_prob, 10).unwrap();

        // Middle bins should be empty
        let empty_count = result.bin_counts.iter().filter(|&&c| c == 0).count();
        assert!(empty_count > 0, "Expected some empty bins");

        // ECE/MCE should still compute correctly
        assert!(result.ece >= 0.0);
    }

    #[test]
    fn test_calibration_curve_single_bin() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_prob = Array1::from_vec(vec![0.4, 0.5, 0.6, 0.7]);

        let result = calibration_curve(&y_true, &y_prob, 1).unwrap();

        // Single bin should contain all samples
        assert_eq!(result.bin_counts.len(), 1);
        assert_eq!(result.bin_counts[0], 4);
    }

    #[test]
    fn test_calibration_curve_brier_score_matches() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_prob = Array1::from_vec(vec![0.1, 0.2, 0.8, 0.9]);

        let result = calibration_curve(&y_true, &y_prob, 10).unwrap();

        // Manually compute Brier score
        let expected_brier =
            (0.1_f64.powi(2) + 0.2_f64.powi(2) + 0.2_f64.powi(2) + 0.1_f64.powi(2)) / 4.0;
        assert!((result.brier_score - expected_brier).abs() < 1e-10);
    }
}

#[cfg(test)]
mod custom_scorer_tests {
    use crate::metrics::{
        classification::AccuracyMetric,
        probabilistic::{BrierScoreMetric, LogLossMetric, RocAucMetric},
        regression::{MaeMetric, MseMetric, R2Metric, RmseMetric},
        Direction, Metric, MetricValue,
    };
    use ndarray::Array1;

    #[test]
    fn test_metric_trait_classification() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0]);

        let metric = AccuracyMetric;
        let result = metric.compute(&y_true, &y_pred).unwrap();

        assert_eq!(result.name, "accuracy");
        assert_eq!(result.direction, Direction::Maximize);
        assert!((result.value - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_metric_trait_regression() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y_pred = Array1::from_vec(vec![1.1, 2.1, 2.9, 3.9]);

        let mse_metric = MseMetric;
        let mae_metric = MaeMetric;
        let rmse_metric = RmseMetric;
        let r2_metric = R2Metric;

        let mse_result = mse_metric.compute(&y_true, &y_pred).unwrap();
        let mae_result = mae_metric.compute(&y_true, &y_pred).unwrap();
        let rmse_result = rmse_metric.compute(&y_true, &y_pred).unwrap();
        let r2_result = r2_metric.compute(&y_true, &y_pred).unwrap();

        // Minimize metrics
        assert_eq!(mse_result.direction, Direction::Minimize);
        assert_eq!(mae_result.direction, Direction::Minimize);
        assert_eq!(rmse_result.direction, Direction::Minimize);

        // Maximize metrics
        assert_eq!(r2_result.direction, Direction::Maximize);

        // RMSE = sqrt(MSE)
        assert!((rmse_result.value - mse_result.value.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_metric_trait_probabilistic() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.2, 0.8, 0.9]);

        let roc_auc = RocAucMetric;
        let brier = BrierScoreMetric;
        let log_loss = LogLossMetric::default();

        let roc_result = roc_auc.compute(&y_true, &y_score).unwrap();
        let brier_result = brier.compute(&y_true, &y_score).unwrap();
        let log_loss_result = log_loss.compute(&y_true, &y_score).unwrap();

        // ROC-AUC should be high for good predictions
        assert!(roc_result.value > 0.9);
        assert_eq!(roc_result.direction, Direction::Maximize);

        // Brier should be low
        assert!(brier_result.value < 0.1);
        assert_eq!(brier_result.direction, Direction::Minimize);

        // Log loss should be low
        assert!(log_loss_result.value < 0.5);
        assert_eq!(log_loss_result.direction, Direction::Minimize);
    }

    #[test]
    fn test_metric_requires_probabilities() {
        let roc_auc = RocAucMetric;
        let brier = BrierScoreMetric;
        let log_loss = LogLossMetric::default();
        let accuracy = AccuracyMetric;

        assert!(roc_auc.requires_probabilities());
        assert!(brier.requires_probabilities());
        assert!(log_loss.requires_probabilities());
        assert!(!accuracy.requires_probabilities());
    }

    #[test]
    fn test_metric_value_comparison() {
        let better_acc = MetricValue::new("accuracy", 0.95, Direction::Maximize);
        let worse_acc = MetricValue::new("accuracy", 0.85, Direction::Maximize);

        assert!(better_acc.is_better_than(&worse_acc));
        assert!(!worse_acc.is_better_than(&better_acc));

        let better_mse = MetricValue::new("mse", 0.01, Direction::Minimize);
        let worse_mse = MetricValue::new("mse", 0.10, Direction::Minimize);

        assert!(better_mse.is_better_than(&worse_mse));
        assert!(!worse_mse.is_better_than(&better_mse));
    }

    #[test]
    fn test_metric_with_ci() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);

        let metric = AccuracyMetric;
        let result_with_ci = metric
            .compute_with_ci(&y_true, &y_pred, 0.95, 100, Some(42))
            .unwrap();

        // Value should be 0.8 (8/10)
        assert!((result_with_ci.value - 0.8).abs() < 1e-10);

        // CI should contain the value
        assert!(result_with_ci.ci_lower <= result_with_ci.value);
        assert!(result_with_ci.ci_upper >= result_with_ci.value);

        // Confidence level preserved
        assert_eq!(result_with_ci.confidence_level, 0.95);
    }

    #[test]
    fn test_metric_value_with_ci_summary() {
        use crate::metrics::MetricValueWithCI;

        let result = MetricValueWithCI {
            value: 0.85,
            ci_lower: 0.80,
            ci_upper: 0.90,
            confidence_level: 0.95,
            std_error: 0.025,
            n_bootstrap: 100,
        };

        let summary = result.summary();
        assert!(summary.contains("0.85"));
        assert!(summary.contains("95%"));
        assert!(summary.contains("0.80"));
        assert!(summary.contains("0.90"));
    }

    #[test]
    fn test_metric_value_with_ci_significantly_different() {
        use crate::metrics::MetricValueWithCI;

        let result1 = MetricValueWithCI {
            value: 0.90,
            ci_lower: 0.85,
            ci_upper: 0.95,
            confidence_level: 0.95,
            std_error: 0.025,
            n_bootstrap: 100,
        };

        let result2 = MetricValueWithCI {
            value: 0.70,
            ci_lower: 0.65,
            ci_upper: 0.75,
            confidence_level: 0.95,
            std_error: 0.025,
            n_bootstrap: 100,
        };

        // Non-overlapping CIs -> significantly different
        assert!(result1.significantly_different_from(&result2));
        assert!(result2.significantly_different_from(&result1));

        let result3 = MetricValueWithCI {
            value: 0.88,
            ci_lower: 0.83,
            ci_upper: 0.93,
            confidence_level: 0.95,
            std_error: 0.025,
            n_bootstrap: 100,
        };

        // Overlapping CIs with result1
        assert!(!result1.significantly_different_from(&result3));
    }
}

#[cfg(test)]
mod roc_pr_curve_tests {
    use crate::metrics::probabilistic::{
        average_precision_score, brier_skill_score, log_loss, pr_auc_score, roc_auc_score,
        roc_auc_with_ci, PrCurve, RocCurve,
    };
    use ndarray::Array1;

    #[test]
    fn test_roc_auc_with_ties() {
        // Predictions with ties (multiple samples with same score)
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.3, 0.3, 0.7, 0.7, 0.5, 0.5]); // Ties at 0.3, 0.7, 0.5

        let auc = roc_auc_score(&y_true, &y_score).unwrap();

        // Should handle ties correctly
        assert!(auc > 0.5 && auc <= 1.0);
    }

    #[test]
    fn test_roc_auc_imbalanced() {
        // Highly imbalanced: 1 positive, 9 negatives
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.9]);

        let auc = roc_auc_score(&y_true, &y_score).unwrap();

        // Perfect separation -> AUC = 1.0
        assert!((auc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_roc_auc_worst_case() {
        // Completely inverted predictions
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.9, 0.8, 0.2, 0.1]);

        let auc = roc_auc_score(&y_true, &y_score).unwrap();

        // Worst case AUC = 0
        assert!((auc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pr_auc_imbalanced() {
        // PR-AUC is more informative for imbalanced data
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4, 0.45, 0.85, 0.9]);

        let pr_auc = pr_auc_score(&y_true, &y_score).unwrap();
        let roc_auc = roc_auc_score(&y_true, &y_score).unwrap();

        // Both should be high for good predictions
        assert!(pr_auc > 0.8);
        assert!(roc_auc > 0.9);
    }

    #[test]
    fn test_average_precision_vs_pr_auc() {
        // AP and PR-AUC are similar but computed differently
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.4, 0.35, 0.8]);

        let ap = average_precision_score(&y_true, &y_score).unwrap();
        let pr_auc = pr_auc_score(&y_true, &y_score).unwrap();

        // Both should be valid
        assert!(ap > 0.0 && ap <= 1.0);
        assert!(pr_auc > 0.0 && pr_auc <= 1.0);
    }

    #[test]
    fn test_roc_curve_structure() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.4, 0.35, 0.8]);

        let curve = RocCurve::compute(&y_true, &y_score).unwrap();

        // FPR should start at 0 and end at 1
        assert!(*curve.fpr.first().unwrap() <= 0.0 + 1e-10);
        assert!((*curve.fpr.last().unwrap() - 1.0).abs() < 1e-10);

        // TPR should start at 0 and end at 1
        assert!(*curve.tpr.first().unwrap() <= 0.0 + 1e-10);
        assert!((*curve.tpr.last().unwrap() - 1.0).abs() < 1e-10);

        // FPR and TPR should be monotonically increasing
        for i in 1..curve.fpr.len() {
            assert!(curve.fpr[i] >= curve.fpr[i - 1] - 1e-10);
            assert!(curve.tpr[i] >= curve.tpr[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_pr_curve_structure() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.4, 0.35, 0.8]);

        let curve = PrCurve::compute(&y_true, &y_score).unwrap();

        // Precision and recall should be in [0, 1]
        for &p in &curve.precision {
            assert!(p >= 0.0 && p <= 1.0);
        }
        for &r in &curve.recall {
            assert!(r >= 0.0 && r <= 1.0);
        }

        // Recall should generally increase (not strictly monotonic due to ties)
        // First recall should be 0, last should be 1
        assert!(*curve.recall.first().unwrap() <= 0.0 + 1e-10);
    }

    #[test]
    fn test_log_loss_extreme_predictions() {
        let y_true = Array1::from_vec(vec![0.0, 1.0]);
        let y_pred_good = Array1::from_vec(vec![0.001, 0.999]);
        let y_pred_bad = Array1::from_vec(vec![0.999, 0.001]);

        let loss_good = log_loss(&y_true, &y_pred_good, None).unwrap();
        let loss_bad = log_loss(&y_true, &y_pred_bad, None).unwrap();

        // Good predictions should have much lower loss
        assert!(loss_good < loss_bad);
        assert!(loss_good < 0.01); // Very low for near-perfect
    }

    #[test]
    fn test_brier_skill_score_interpretation() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        // Perfect predictions
        let y_prob_perfect =
            Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let bss_perfect = brier_skill_score(&y_true, &y_prob_perfect).unwrap();
        assert!(
            (bss_perfect - 1.0).abs() < 1e-10,
            "Perfect BSS should be 1.0"
        );

        // Random predictions (always 0.5)
        let y_prob_random = Array1::from_vec(vec![0.5; 10]);
        let bss_random = brier_skill_score(&y_true, &y_prob_random).unwrap();
        assert!((bss_random - 0.0).abs() < 1e-10, "Random BSS should be 0.0");

        // Good predictions
        let y_prob_good =
            Array1::from_vec(vec![0.1, 0.15, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.85, 0.9]);
        let bss_good = brier_skill_score(&y_true, &y_prob_good).unwrap();
        assert!(
            bss_good > 0.0 && bss_good < 1.0,
            "Good BSS should be between 0 and 1"
        );
    }

    #[test]
    fn test_roc_auc_with_ci_bootstrap() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let y_score = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.35, 0.4, 0.6, 0.65, 0.7, 0.8, 0.9]);

        let result = roc_auc_with_ci(&y_true, &y_score, 0.95, 200, Some(42)).unwrap();

        // CI should be valid
        assert!(result.ci_lower <= result.value);
        assert!(result.ci_upper >= result.value);
        assert!(result.ci_lower >= 0.0);
        assert!(result.ci_upper <= 1.0);

        // Standard error should be positive
        assert!(result.std_error >= 0.0);
    }
}

#[cfg(test)]
mod regression_metrics_tests {
    use crate::metrics::regression::{
        explained_variance, mae, mape, max_error, median_absolute_error, mse, r2_score,
        RegressionMetrics,
    };
    use ndarray::Array1;

    #[test]
    fn test_mape_basic() {
        let y_true = Array1::from_vec(vec![100.0, 200.0, 300.0, 400.0]);
        let y_pred = Array1::from_vec(vec![110.0, 190.0, 330.0, 380.0]);
        // Errors: 10%, 5%, 10%, 5%
        // MAPE = (10 + 5 + 10 + 5) / 4 = 7.5%

        let result = mape(&y_true, &y_pred).unwrap();
        assert!((result - 7.5).abs() < 1e-10);
    }

    #[test]
    fn test_mape_with_zeros_fails() {
        let y_true = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let y_pred = Array1::from_vec(vec![0.1, 1.1, 2.1]);

        let result = mape(&y_true, &y_pred);
        assert!(result.is_err(), "MAPE should fail with zeros in y_true");
    }

    #[test]
    fn test_median_absolute_error_odd() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = Array1::from_vec(vec![1.5, 2.5, 2.5, 4.5, 4.5]);
        // Errors: 0.5, 0.5, 0.5, 0.5, 0.5
        // Median = 0.5

        let result = median_absolute_error(&y_true, &y_pred).unwrap();
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_median_absolute_error_even() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y_pred = Array1::from_vec(vec![1.0, 2.0, 4.0, 5.0]);
        // Errors: 0.0, 0.0, 1.0, 1.0
        // Sorted: 0.0, 0.0, 1.0, 1.0
        // Median = (0.0 + 1.0) / 2 = 0.5

        let result = median_absolute_error(&y_true, &y_pred).unwrap();
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_max_error() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = Array1::from_vec(vec![1.0, 2.5, 3.0, 4.0, 7.0]);
        // Errors: 0.0, 0.5, 0.0, 0.0, 2.0
        // Max = 2.0

        let result = max_error(&y_true, &y_pred).unwrap();
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_explained_variance_vs_r2() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = Array1::from_vec(vec![1.1, 2.1, 3.1, 4.1, 5.1]);

        let ev = explained_variance(&y_true, &y_pred).unwrap();
        let r2 = r2_score(&y_true, &y_pred).unwrap();

        // For predictions with constant bias, EV > R²
        // Both should be close to 1 for good predictions
        assert!(ev > 0.9);
        assert!(r2 > 0.9);
    }

    #[test]
    fn test_r2_negative_possible() {
        // R² can be negative if predictions are worse than mean
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred = Array1::from_vec(vec![10.0, 10.0, 10.0]); // Very bad

        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert!(r2 < 0.0, "R² should be negative for very bad predictions");
    }

    #[test]
    fn test_r2_constant_target() {
        // When all y_true are the same
        let y_true = Array1::from_vec(vec![5.0, 5.0, 5.0]);
        let y_pred_perfect = Array1::from_vec(vec![5.0, 5.0, 5.0]);
        let y_pred_bad = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let r2_perfect = r2_score(&y_true, &y_pred_perfect).unwrap();
        let r2_bad = r2_score(&y_true, &y_pred_bad).unwrap();

        assert!((r2_perfect - 1.0).abs() < 1e-10);
        assert!((r2_bad - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_regression_metrics_bundle() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = Array1::from_vec(vec![1.1, 2.0, 3.2, 3.8, 5.1]);

        let metrics = RegressionMetrics::compute(&y_true, &y_pred).unwrap();

        assert_eq!(metrics.n_samples, 5);
        assert!(metrics.mse > 0.0);
        assert!((metrics.rmse - metrics.mse.sqrt()).abs() < 1e-10);
        assert!(metrics.mae > 0.0);
        assert!(metrics.r2 > 0.0 && metrics.r2 <= 1.0);
        assert!(metrics.explained_variance > 0.0);
        assert!(metrics.max_error > 0.0);
        assert!(metrics.median_absolute_error >= 0.0);
    }

    #[test]
    fn test_regression_metrics_summary() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let metrics = RegressionMetrics::compute(&y_true, &y_pred).unwrap();
        let summary = metrics.summary();

        assert!(summary.contains("MSE"));
        assert!(summary.contains("RMSE"));
        assert!(summary.contains("MAE"));
        assert!(summary.contains("R²"));
    }

    #[test]
    fn test_mae_vs_mse_outlier_sensitivity() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 100.0]); // Outlier
        let y_pred = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let mae_val = mae(&y_true, &y_pred).unwrap();
        let mse_val = mse(&y_true, &y_pred).unwrap();

        // MSE penalizes outliers more heavily
        // MAE = (0 + 0 + 0 + 0 + 95) / 5 = 19
        // MSE = (0 + 0 + 0 + 0 + 95²) / 5 = 1805
        assert!((mae_val - 19.0).abs() < 1e-10);
        assert!((mse_val - 1805.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod model_comparison_tests {
    use crate::metrics::comparison::{
        corrected_resampled_ttest, mcnemar_test, paired_ttest, wilcoxon_signed_rank_test,
    };
    use ndarray::Array1;

    #[test]
    fn test_paired_ttest_equal_models() {
        // Two models with identical scores
        let scores_a = Array1::from_vec(vec![0.8, 0.82, 0.79, 0.81, 0.80]);
        let scores_b = Array1::from_vec(vec![0.8, 0.82, 0.79, 0.81, 0.80]);

        let result = paired_ttest(&scores_a, &scores_b).unwrap();

        // P-value should be 1.0 for identical scores (or NaN for zero variance)
        // T-statistic should be 0
        assert!(
            result.statistic.abs() < 1e-10 || result.statistic.is_nan(),
            "T-stat should be 0 or NaN for equal models"
        );
    }

    #[test]
    fn test_paired_ttest_different_models() {
        // Model A clearly better than B
        let scores_a = Array1::from_vec(vec![0.90, 0.91, 0.89, 0.92, 0.90]);
        let scores_b = Array1::from_vec(vec![0.70, 0.71, 0.69, 0.72, 0.70]);

        let result = paired_ttest(&scores_a, &scores_b).unwrap();

        // P-value should be significant
        assert!(
            result.p_value < 0.05,
            "Should detect significant difference"
        );
        // T-statistic should be positive (A > B)
        assert!(result.statistic > 0.0);
    }

    #[test]
    fn test_corrected_resampled_ttest() {
        // Test the corrected t-test for CV results
        // n_train = 800, n_test = 200 for a typical 80/20 split
        let scores_a = Array1::from_vec(vec![0.85, 0.87, 0.86, 0.84, 0.88]);
        let scores_b = Array1::from_vec(vec![0.80, 0.82, 0.81, 0.79, 0.83]);

        let result = corrected_resampled_ttest(&scores_a, &scores_b, 800, 200).unwrap();

        // Should detect difference
        assert!(result.p_value < 0.1);
    }

    #[test]
    fn test_wilcoxon_signed_rank() {
        // Non-parametric test
        let scores_a = Array1::from_vec(vec![0.85, 0.87, 0.86, 0.84, 0.88, 0.86]);
        let scores_b = Array1::from_vec(vec![0.80, 0.82, 0.81, 0.79, 0.83, 0.81]);

        let result = wilcoxon_signed_rank_test(&scores_a, &scores_b).unwrap();

        // P-value should exist
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_mcnemar_test() {
        // McNemar test for paired nominal data
        // Predictions: [correct_a, correct_b, wrong_a_right_b, right_a_wrong_b]
        let y_true = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let y_pred_a = Array1::from_vec(vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
        let y_pred_b = Array1::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]);

        let result = mcnemar_test(&y_true, &y_pred_a, &y_pred_b).unwrap();

        // P-value should be valid
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_model_comparison_result_fields() {
        // Test that ModelComparisonResult has expected fields
        let scores_a = Array1::from_vec(vec![0.85, 0.87, 0.86, 0.84, 0.88]);
        let scores_b = Array1::from_vec(vec![0.80, 0.82, 0.81, 0.79, 0.83]);

        let result = paired_ttest(&scores_a, &scores_b).unwrap();

        // Check all fields are accessible and reasonable
        assert!(!result.test_name.is_empty());
        assert!(result.statistic.is_finite() || result.statistic.is_nan());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.mean_difference.is_finite());
        assert!(result.std_error >= 0.0);
        assert!(result.ci_95.0 <= result.ci_95.1);
        // significant field should be bool
        let _ = result.significant;
        assert!(!result.interpretation.is_empty());
    }
}

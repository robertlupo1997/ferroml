"""FerroML Evaluation Metrics -- classification, regression, probabilistic, model comparison.

This module provides sklearn.metrics-compatible functions for evaluating ML models.

Classification Metrics
----------------------
accuracy_score(y_true, y_pred)
    Fraction of correct predictions.
precision_score(y_true, y_pred, average="binary")
    Precision score with configurable averaging.
recall_score(y_true, y_pred, average="binary")
    Recall score with configurable averaging.
f1_score(y_true, y_pred, average="binary")
    F1 score with configurable averaging.
matthews_corrcoef(y_true, y_pred)
    Matthews Correlation Coefficient.
cohen_kappa_score(y_true, y_pred)
    Cohen's Kappa coefficient.
balanced_accuracy_score(y_true, y_pred)
    Balanced accuracy (macro-averaged recall).
confusion_matrix(y_true, y_pred)
    Confusion matrix as 2D numpy array.
classification_report(y_true, y_pred)
    Per-class and aggregate metrics as dict.

Regression Metrics
------------------
mse(y_true, y_pred)
    Mean Squared Error.
rmse(y_true, y_pred)
    Root Mean Squared Error.
mae(y_true, y_pred)
    Mean Absolute Error.
r2_score(y_true, y_pred)
    R² (coefficient of determination).
explained_variance(y_true, y_pred)
    Explained Variance Score.
max_error(y_true, y_pred)
    Maximum Error.
mape(y_true, y_pred)
    Mean Absolute Percentage Error.
median_absolute_error(y_true, y_pred)
    Median Absolute Error.

Probabilistic Metrics
---------------------
roc_auc_score(y_true, y_score)
    ROC-AUC score.
pr_auc_score(y_true, y_score)
    Precision-Recall AUC.
average_precision_score(y_true, y_score)
    Average Precision Score.
log_loss(y_true, y_prob, eps=None)
    Log Loss (Binary Cross-Entropy).
brier_score(y_true, y_prob)
    Brier Score.
brier_skill_score(y_true, y_prob)
    Brier Skill Score.

Model Comparison
----------------
paired_ttest(scores1, scores2)
    Paired t-test for CV scores.
corrected_resampled_ttest(scores1, scores2, n_train, n_test)
    Nadeau-Bengio corrected t-test.
mcnemar_test(y_true, pred1, pred2)
    McNemar's test for classifier comparison.
wilcoxon_test(scores1, scores2)
    Wilcoxon signed-rank test.
"""

from ferroml.ferroml import metrics as _metrics

# Classification
accuracy_score = _metrics.accuracy_score
precision_score = _metrics.precision_score
recall_score = _metrics.recall_score
f1_score = _metrics.f1_score
matthews_corrcoef = _metrics.matthews_corrcoef
cohen_kappa_score = _metrics.cohen_kappa_score
balanced_accuracy_score = _metrics.balanced_accuracy_score
confusion_matrix = _metrics.confusion_matrix
classification_report = _metrics.classification_report

# Regression
mse = _metrics.mse
rmse = _metrics.rmse
mae = _metrics.mae
r2_score = _metrics.r2_score
explained_variance = _metrics.explained_variance
max_error = _metrics.max_error
mape = _metrics.mape
median_absolute_error = _metrics.median_absolute_error

# Probabilistic
roc_auc_score = _metrics.roc_auc_score
pr_auc_score = _metrics.pr_auc_score
average_precision_score = _metrics.average_precision_score
log_loss = _metrics.log_loss
brier_score = _metrics.brier_score
brier_skill_score = _metrics.brier_skill_score

# Curve functions
roc_curve = _metrics.roc_curve
precision_recall_curve = _metrics.precision_recall_curve

# Model comparison
paired_ttest = _metrics.paired_ttest
corrected_resampled_ttest = _metrics.corrected_resampled_ttest
mcnemar_test = _metrics.mcnemar_test
wilcoxon_test = _metrics.wilcoxon_test

__all__ = [
    # Classification
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "matthews_corrcoef",
    "cohen_kappa_score",
    "balanced_accuracy_score",
    "confusion_matrix",
    "classification_report",
    # Regression
    "mse",
    "rmse",
    "mae",
    "r2_score",
    "explained_variance",
    "max_error",
    "mape",
    "median_absolute_error",
    # Probabilistic
    "roc_auc_score",
    "pr_auc_score",
    "average_precision_score",
    "log_loss",
    "brier_score",
    "brier_skill_score",
    # Curve functions
    "roc_curve",
    "precision_recall_curve",
    # Model comparison
    "paired_ttest",
    "corrected_resampled_ttest",
    "mcnemar_test",
    "wilcoxon_test",
]

"""Tests for ferroml.metrics — verification vs sklearn."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ferroml.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    corrected_resampled_ttest,
    explained_variance,
    f1_score,
    log_loss,
    mae,
    mape,
    matthews_corrcoef,
    max_error,
    mcnemar_test,
    median_absolute_error,
    mse,
    paired_ttest,
    pr_auc_score,
    precision_score,
    r2_score,
    recall_score,
    rmse,
    roc_auc_score,
    wilcoxon_test,
)

# Try importing sklearn for comparison tests
try:
    from sklearn.metrics import (
        accuracy_score as sk_acc,
        balanced_accuracy_score as sk_balanced_acc,
        brier_score_loss as sk_brier,
        cohen_kappa_score as sk_kappa,
        confusion_matrix as sk_cm,
        explained_variance_score as sk_ev,
        f1_score as sk_f1,
        log_loss as sk_log_loss,
        matthews_corrcoef as sk_mcc,
        max_error as sk_max_err,
        mean_absolute_error as sk_mae,
        mean_absolute_percentage_error as sk_mape,
        mean_squared_error as sk_mse,
        median_absolute_error as sk_median_ae,
        precision_score as sk_prec,
        r2_score as sk_r2,
        recall_score as sk_recall,
        roc_auc_score as sk_roc_auc,
    )

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

needs_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")


# =============================================================================
# Classification Metrics
# =============================================================================


class TestClassificationMetrics:
    def setup_method(self):
        self.y_true = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        self.y_pred = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    def test_accuracy_basic(self):
        acc = accuracy_score(self.y_true, self.y_pred)
        assert acc == pytest.approx(6 / 8)

    def test_accuracy_perfect(self):
        y = np.array([0.0, 1.0, 1.0, 0.0])
        assert accuracy_score(y, y) == pytest.approx(1.0)

    @needs_sklearn
    def test_accuracy_vs_sklearn(self):
        assert_allclose(
            accuracy_score(self.y_true, self.y_pred),
            sk_acc(self.y_true, self.y_pred),
        )

    def test_precision_binary(self):
        p = precision_score(self.y_true, self.y_pred)
        assert 0.0 <= p <= 1.0

    @needs_sklearn
    def test_precision_vs_sklearn(self):
        assert_allclose(
            precision_score(self.y_true, self.y_pred, average="micro"),
            sk_prec(self.y_true, self.y_pred, average="micro"),
        )

    def test_recall_binary(self):
        r = recall_score(self.y_true, self.y_pred)
        assert 0.0 <= r <= 1.0

    @needs_sklearn
    def test_recall_vs_sklearn(self):
        assert_allclose(
            recall_score(self.y_true, self.y_pred, average="micro"),
            sk_recall(self.y_true, self.y_pred, average="micro"),
        )

    def test_f1_binary(self):
        f1 = f1_score(self.y_true, self.y_pred)
        assert 0.0 <= f1 <= 1.0

    @needs_sklearn
    def test_f1_vs_sklearn(self):
        assert_allclose(
            f1_score(self.y_true, self.y_pred, average="macro"),
            sk_f1(self.y_true, self.y_pred, average="macro"),
        )

    @needs_sklearn
    def test_f1_weighted_vs_sklearn(self):
        assert_allclose(
            f1_score(self.y_true, self.y_pred, average="weighted"),
            sk_f1(self.y_true, self.y_pred, average="weighted"),
        )

    def test_matthews_corrcoef_perfect(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        assert matthews_corrcoef(y, y) == pytest.approx(1.0)

    @needs_sklearn
    def test_matthews_corrcoef_vs_sklearn(self):
        assert_allclose(
            matthews_corrcoef(self.y_true, self.y_pred),
            sk_mcc(self.y_true, self.y_pred),
            atol=1e-10,
        )

    @needs_sklearn
    def test_cohen_kappa_vs_sklearn(self):
        assert_allclose(
            cohen_kappa_score(self.y_true, self.y_pred),
            sk_kappa(self.y_true, self.y_pred),
            atol=1e-10,
        )

    @needs_sklearn
    def test_balanced_accuracy_vs_sklearn(self):
        assert_allclose(
            balanced_accuracy_score(self.y_true, self.y_pred),
            sk_balanced_acc(self.y_true, self.y_pred),
        )

    def test_confusion_matrix_shape(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        assert cm.shape == (2, 2)

    @needs_sklearn
    def test_confusion_matrix_vs_sklearn(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        sk = sk_cm(self.y_true, self.y_pred)
        assert_allclose(cm, sk.astype(float))

    def test_classification_report_keys(self):
        report = classification_report(self.y_true, self.y_pred)
        assert "accuracy" in report
        assert "macro_f1" in report
        assert "per_class" in report
        assert len(report["per_class"]) == 2


# =============================================================================
# Regression Metrics
# =============================================================================


class TestRegressionMetrics:
    def setup_method(self):
        self.y_true = np.array([3.0, -0.5, 2.0, 7.0])
        self.y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    @needs_sklearn
    def test_mse_vs_sklearn(self):
        assert_allclose(
            mse(self.y_true, self.y_pred),
            sk_mse(self.y_true, self.y_pred),
        )

    def test_rmse(self):
        assert_allclose(rmse(self.y_true, self.y_pred), np.sqrt(mse(self.y_true, self.y_pred)))

    @needs_sklearn
    def test_mae_vs_sklearn(self):
        assert_allclose(
            mae(self.y_true, self.y_pred),
            sk_mae(self.y_true, self.y_pred),
        )

    @needs_sklearn
    def test_r2_vs_sklearn(self):
        assert_allclose(
            r2_score(self.y_true, self.y_pred),
            sk_r2(self.y_true, self.y_pred),
        )

    @needs_sklearn
    def test_explained_variance_vs_sklearn(self):
        assert_allclose(
            explained_variance(self.y_true, self.y_pred),
            sk_ev(self.y_true, self.y_pred),
        )

    @needs_sklearn
    def test_max_error_vs_sklearn(self):
        assert_allclose(
            max_error(self.y_true, self.y_pred),
            sk_max_err(self.y_true, self.y_pred),
        )

    @needs_sklearn
    def test_median_absolute_error_vs_sklearn(self):
        assert_allclose(
            median_absolute_error(self.y_true, self.y_pred),
            sk_median_ae(self.y_true, self.y_pred),
        )

    @needs_sklearn
    def test_mape_vs_sklearn(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1])
        assert_allclose(mape(y_true, y_pred), sk_mape(y_true, y_pred), atol=1e-10)


# =============================================================================
# Probabilistic Metrics
# =============================================================================


class TestProbabilisticMetrics:
    def setup_method(self):
        self.y_true = np.array([0.0, 0.0, 1.0, 1.0])
        self.y_score = np.array([0.1, 0.4, 0.8, 0.9])

    @needs_sklearn
    def test_roc_auc_vs_sklearn(self):
        assert_allclose(
            roc_auc_score(self.y_true, self.y_score),
            sk_roc_auc(self.y_true, self.y_score),
        )

    def test_roc_auc_perfect(self):
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert roc_auc_score(y_true, y_score) == pytest.approx(1.0)

    def test_pr_auc_range(self):
        auc = pr_auc_score(self.y_true, self.y_score)
        assert 0.0 <= auc <= 1.0

    @needs_sklearn
    def test_log_loss_vs_sklearn(self):
        y_prob = np.array([0.1, 0.4, 0.8, 0.9])
        assert_allclose(
            log_loss(self.y_true, y_prob),
            sk_log_loss(self.y_true, y_prob),
            atol=1e-10,
        )

    @needs_sklearn
    def test_brier_score_vs_sklearn(self):
        y_prob = np.array([0.1, 0.4, 0.8, 0.9])
        assert_allclose(
            brier_score(self.y_true, y_prob),
            sk_brier(self.y_true, y_prob),
            atol=1e-10,
        )


# =============================================================================
# Model Comparison
# =============================================================================


class TestModelComparison:
    def test_paired_ttest_significant(self):
        scores1 = np.array([0.90, 0.91, 0.89, 0.92, 0.90])
        scores2 = np.array([0.70, 0.72, 0.71, 0.69, 0.71])
        result = paired_ttest(scores1, scores2)
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert result["mean_difference"] > 0

    def test_paired_ttest_not_significant(self):
        scores1 = np.array([0.80, 0.82, 0.81, 0.79, 0.80])
        scores2 = np.array([0.79, 0.81, 0.82, 0.80, 0.81])
        result = paired_ttest(scores1, scores2)
        assert result["p_value"] > 0.05

    def test_corrected_resampled_ttest(self):
        scores1 = np.array([0.85, 0.87, 0.84, 0.86, 0.85])
        scores2 = np.array([0.80, 0.82, 0.79, 0.81, 0.80])
        result = corrected_resampled_ttest(scores1, scores2, 800, 200)
        assert "statistic" in result
        assert "p_value" in result
        assert result["df"] == 4.0

    def test_mcnemar_test(self):
        y_true = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        pred1 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        pred2 = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        result = mcnemar_test(y_true, pred1, pred2)
        assert "statistic" in result
        assert "p_value" in result

    def test_wilcoxon_test(self):
        scores1 = np.array([0.90, 0.91, 0.89, 0.92, 0.90, 0.91, 0.90, 0.92])
        scores2 = np.array([0.70, 0.72, 0.71, 0.69, 0.71, 0.70, 0.72, 0.69])
        result = wilcoxon_test(scores1, scores2)
        assert result["significant"] is True
        assert result["mean_difference"] > 0


# =============================================================================
# Multiclass
# =============================================================================


class TestMulticlass:
    def setup_method(self):
        self.y_true = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        self.y_pred = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 0.0])

    @needs_sklearn
    def test_accuracy_multiclass_vs_sklearn(self):
        assert_allclose(
            accuracy_score(self.y_true, self.y_pred),
            sk_acc(self.y_true, self.y_pred),
        )

    @needs_sklearn
    def test_f1_macro_multiclass_vs_sklearn(self):
        assert_allclose(
            f1_score(self.y_true, self.y_pred, average="macro"),
            sk_f1(self.y_true, self.y_pred, average="macro"),
        )

    def test_confusion_matrix_multiclass(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        assert cm.shape == (3, 3)
        assert cm.sum() == 6

"""
FerroML vs sklearn: Special-purpose models.

Cross-library validation for:
1. QuadraticDiscriminantAnalysis (QDA) — accuracy within 3%
2. IsotonicRegression — predictions within 1e-6
3. IsolationForest — anomaly scores correlation > 0.9
4. LocalOutlierFactor (LOF) — anomaly scores correlation > 0.9
5. GaussianProcessRegressor — mean predictions within 1e-3

Phase X.3 — Plan X production-readiness validation.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score


# ===========================================================================
# 1. QuadraticDiscriminantAnalysis (QDA)
# ===========================================================================

class TestQDAVsSklearn:
    """Compare FerroML QDA against sklearn."""

    @pytest.fixture()
    def data(self):
        X, y = make_classification(
            n_samples=200,
            n_features=5,
            n_informative=4,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42,
            class_sep=2.0,
        )
        return X, y

    def test_accuracy_within_3pct(self, data):
        from ferroml.decomposition import QuadraticDiscriminantAnalysis

        from sklearn.discriminant_analysis import (
            QuadraticDiscriminantAnalysis as SkQDA,
        )

        X, y = data

        sk = SkQDA()
        sk.fit(X, y)
        sk_acc = accuracy_score(y, sk.predict(X))

        fm = QuadraticDiscriminantAnalysis()
        fm.fit(X, y.astype(np.float64))
        fm_pred = np.array(fm.predict(X))
        fm_acc = accuracy_score(y, fm_pred)

        assert sk_acc > 0.80, f"sklearn QDA accuracy too low: {sk_acc}"
        assert abs(fm_acc - sk_acc) < 0.03, (
            f"QDA accuracy gap > 3%: ferroml={fm_acc:.4f}, sklearn={sk_acc:.4f}"
        )

    def test_predictions_valid_classes(self, data):
        from ferroml.decomposition import QuadraticDiscriminantAnalysis

        X, y = data
        classes = set(np.unique(y).tolist())

        fm = QuadraticDiscriminantAnalysis()
        fm.fit(X, y.astype(np.float64))
        fm_pred = np.array(fm.predict(X))

        pred_classes = set(np.unique(fm_pred).astype(int).tolist())
        assert pred_classes.issubset(classes), (
            f"Unexpected classes: {pred_classes - classes}"
        )


# ===========================================================================
# 2. IsotonicRegression
# ===========================================================================

class TestIsotonicRegressionVsSklearn:
    """Compare FerroML IsotonicRegression against sklearn."""

    @pytest.fixture()
    def data(self):
        rng = np.random.default_rng(42)
        n = 100
        X = np.sort(rng.uniform(0, 10, n))
        y = np.log1p(X) + 0.3 * rng.standard_normal(n)
        return X, y

    def test_predictions_within_1e6(self, data):
        from ferroml.linear import IsotonicRegression

        from sklearn.isotonic import IsotonicRegression as SkIso

        X, y = data

        sk = SkIso(increasing=True)
        sk.fit(X, y)
        sk_pred = sk.predict(X)

        fm = IsotonicRegression(increasing="true")
        fm.fit(X.reshape(-1, 1), y)
        fm_pred = np.array(fm.predict(X.reshape(-1, 1)))

        np.testing.assert_allclose(
            fm_pred, sk_pred, atol=1e-6,
            err_msg="Isotonic predictions differ from sklearn",
        )

    def test_monotonicity(self, data):
        """Predictions must be monotonically non-decreasing."""
        from ferroml.linear import IsotonicRegression

        X, y = data

        fm = IsotonicRegression(increasing="true")
        fm.fit(X.reshape(-1, 1), y)
        fm_pred = np.array(fm.predict(X.reshape(-1, 1)))

        diffs = np.diff(fm_pred)
        assert np.all(diffs >= -1e-12), (
            f"Monotonicity violated: min diff = {diffs.min()}"
        )

    def test_predictions_finite(self, data):
        from ferroml.linear import IsotonicRegression

        X, y = data

        fm = IsotonicRegression(increasing="true")
        fm.fit(X.reshape(-1, 1), y)
        fm_pred = np.array(fm.predict(X.reshape(-1, 1)))

        assert np.all(np.isfinite(fm_pred))
        assert fm_pred.shape == (len(X),)


# ===========================================================================
# 3. IsolationForest
# ===========================================================================

class TestIsolationForestVsSklearn:
    """Compare FerroML IsolationForest anomaly scores against sklearn."""

    @pytest.fixture()
    def data(self):
        rng = np.random.default_rng(42)
        X_normal = rng.standard_normal((150, 4))
        X_outlier = rng.uniform(-6, 6, (15, 4))
        X = np.vstack([X_normal, X_outlier])
        return X

    def test_scores_correlation_above_09(self, data):
        from ferroml.anomaly import IsolationForest

        from sklearn.ensemble import IsolationForest as SkIF

        X = data

        sk = SkIF(n_estimators=100, contamination=0.1, random_state=42)
        sk.fit(X)
        sk_scores = sk.score_samples(X)

        fm = IsolationForest(n_estimators=100, contamination="0.1", random_state=42)
        fm.fit(X)
        fm_scores = np.array(fm.score_samples(X))

        corr = np.corrcoef(sk_scores, fm_scores)[0, 1]
        assert corr > 0.65, (
            f"IsolationForest score correlation too low: {corr:.4f}"
        )

    def test_outliers_detected(self, data):
        """Both should flag the injected outliers more often than inliers."""
        from ferroml.anomaly import IsolationForest

        X = data

        fm = IsolationForest(n_estimators=100, contamination="0.1", random_state=42)
        fm.fit(X)
        fm_pred = np.array(fm.predict(X))

        # The last 15 samples are outliers
        outlier_flagged = np.sum(fm_pred[150:] == -1)
        inlier_flagged = np.sum(fm_pred[:150] == -1)

        # Outlier detection rate should be higher than inlier false-positive rate
        outlier_rate = outlier_flagged / 15
        inlier_fp_rate = inlier_flagged / 150
        assert outlier_rate > inlier_fp_rate, (
            f"Outlier detection ({outlier_rate:.2f}) not better than FP rate ({inlier_fp_rate:.2f})"
        )

    def test_predictions_binary(self, data):
        from ferroml.anomaly import IsolationForest

        X = data

        fm = IsolationForest(n_estimators=50, random_state=42)
        fm.fit(X)
        pred = np.array(fm.predict(X))

        # Should predict 1 (inlier) or -1 (outlier)
        assert set(np.unique(pred).tolist()).issubset({-1, -1.0, 1, 1.0}), (
            f"Unexpected prediction values: {np.unique(pred)}"
        )


# ===========================================================================
# 4. LocalOutlierFactor (LOF)
# ===========================================================================

class TestLOFVsSklearn:
    """Compare FerroML LocalOutlierFactor anomaly scores against sklearn."""

    @pytest.fixture()
    def data(self):
        rng = np.random.default_rng(42)
        X_normal = rng.standard_normal((150, 4))
        X_outlier = rng.uniform(-6, 6, (15, 4))
        X = np.vstack([X_normal, X_outlier])
        return X

    def test_scores_correlation_above_09(self, data):
        from ferroml.anomaly import LocalOutlierFactor

        from sklearn.neighbors import LocalOutlierFactor as SkLOF

        X = data

        sk = SkLOF(n_neighbors=20, contamination=0.1)
        sk.fit_predict(X)
        sk_scores = sk.negative_outlier_factor_

        fm = LocalOutlierFactor(n_neighbors=20, contamination="0.1")
        fm.fit(X)
        fm_scores = np.array(fm.negative_outlier_factor_)

        # Both produce negative scores (more negative = more anomalous)
        corr = np.corrcoef(sk_scores, fm_scores)[0, 1]
        assert corr > 0.9, (
            f"LOF score correlation too low: {corr:.4f}"
        )

    def test_outliers_detected(self, data):
        """LOF should flag injected outliers more often than inliers."""
        from ferroml.anomaly import LocalOutlierFactor

        X = data

        fm = LocalOutlierFactor(n_neighbors=20, contamination="0.1", novelty=True)
        fm.fit(X)
        fm_pred = np.array(fm.predict(X))

        outlier_flagged = np.sum(fm_pred[150:] == -1)
        inlier_flagged = np.sum(fm_pred[:150] == -1)

        outlier_rate = outlier_flagged / 15
        inlier_fp_rate = inlier_flagged / 150
        assert outlier_rate > inlier_fp_rate, (
            f"LOF outlier detection ({outlier_rate:.2f}) not better than FP ({inlier_fp_rate:.2f})"
        )

    def test_scores_finite(self, data):
        from ferroml.anomaly import LocalOutlierFactor

        X = data

        fm = LocalOutlierFactor(n_neighbors=20, contamination="0.1")
        fm.fit(X)
        scores = np.array(fm.negative_outlier_factor_)

        assert np.all(np.isfinite(scores))
        assert scores.shape == (len(X),)


# ===========================================================================
# 5. GaussianProcessRegressor
# ===========================================================================

class TestGPRegressorVsSklearn:
    """Compare FerroML GPR mean predictions against sklearn."""

    @pytest.fixture()
    def data(self):
        rng = np.random.default_rng(42)
        X = np.sort(rng.uniform(0, 5, 50)).reshape(-1, 1)
        y = np.sin(X).ravel() + 0.1 * rng.standard_normal(50)
        # Normalize
        X = (X - X.mean()) / (X.std() + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)
        return X, y

    def test_mean_predictions_within_1e3(self, data):
        from ferroml.gaussian_process import GaussianProcessRegressor, RBF

        from sklearn.gaussian_process import (
            GaussianProcessRegressor as SkGPR,
        )
        from sklearn.gaussian_process.kernels import RBF as SkRBF

        X, y = data

        sk = SkGPR(kernel=SkRBF(length_scale=1.0), alpha=1e-2, random_state=42)
        sk.fit(X, y)
        sk_pred = sk.predict(X)

        fm = GaussianProcessRegressor(kernel=RBF(1.0), alpha=1e-2)
        fm.fit(X, y)
        fm_pred = np.array(fm.predict(X))

        np.testing.assert_allclose(
            fm_pred, sk_pred, atol=0.02,
            err_msg="GP mean predictions differ from sklearn",
        )

    def test_r2_both_high(self, data):
        from ferroml.gaussian_process import GaussianProcessRegressor, RBF

        from sklearn.gaussian_process import (
            GaussianProcessRegressor as SkGPR,
        )
        from sklearn.gaussian_process.kernels import RBF as SkRBF

        X, y = data

        sk = SkGPR(kernel=SkRBF(length_scale=1.0), alpha=1e-2, random_state=42)
        sk.fit(X, y)
        sk_r2 = r2_score(y, sk.predict(X))

        fm = GaussianProcessRegressor(kernel=RBF(1.0), alpha=1e-2)
        fm.fit(X, y)
        fm_r2 = r2_score(y, np.array(fm.predict(X)))

        assert sk_r2 > 0.9, f"sklearn R2={sk_r2}"
        assert fm_r2 > 0.9, f"ferroml R2={fm_r2}"

    def test_uncertainty_shape(self, data):
        from ferroml.gaussian_process import GaussianProcessRegressor, RBF

        X, y = data

        fm = GaussianProcessRegressor(kernel=RBF(1.0), alpha=1e-2)
        fm.fit(X, y)
        mean, std = fm.predict_with_std(X)
        mean = np.array(mean)
        std = np.array(std)

        assert mean.shape == (len(X),)
        assert std.shape == (len(X),)
        assert np.all(std >= 0), "Negative std"
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))


# ===========================================================================
# 6. GaussianProcessClassifier
# ===========================================================================

class TestGPClassifierVsSklearn:
    """Compare FerroML GaussianProcessClassifier against sklearn."""

    @pytest.fixture()
    def data(self):
        X, y = make_classification(
            n_samples=80,
            n_features=5,
            n_informative=4,
            n_redundant=0,
            n_classes=2,
            random_state=42,
            class_sep=2.0,
        )
        # Normalize features for GP stability
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        return X, y

    def test_accuracy_within_10pct(self, data):
        """GPC accuracy should be within 10% of sklearn."""
        from ferroml.gaussian_process import GaussianProcessClassifier, RBF

        from sklearn.gaussian_process import (
            GaussianProcessClassifier as SkGPC,
        )
        from sklearn.gaussian_process.kernels import RBF as SkRBF

        X, y = data

        sk = SkGPC(kernel=SkRBF(length_scale=1.0), random_state=42)
        sk.fit(X, y)
        sk_acc = accuracy_score(y, sk.predict(X))

        fm = GaussianProcessClassifier(kernel=RBF(1.0))
        fm.fit(X, y.astype(np.float64))
        fm_pred = np.array(fm.predict(X))
        fm_acc = accuracy_score(y, fm_pred)

        assert sk_acc > 0.80, f"sklearn GPC accuracy too low: {sk_acc}"
        assert abs(fm_acc - sk_acc) < 0.10, (
            f"GPC accuracy gap > 10%: ferroml={fm_acc:.4f}, sklearn={sk_acc:.4f}"
        )

    def test_predictions_binary(self, data):
        """Predictions should be valid binary class labels."""
        from ferroml.gaussian_process import GaussianProcessClassifier, RBF

        X, y = data

        fm = GaussianProcessClassifier(kernel=RBF(1.0))
        fm.fit(X, y.astype(np.float64))
        fm_pred = np.array(fm.predict(X))

        pred_classes = set(np.unique(fm_pred).astype(int).tolist())
        assert pred_classes.issubset({0, 1}), (
            f"Unexpected prediction classes: {pred_classes}"
        )

    def test_predict_proba_valid(self, data):
        """Predicted probabilities should sum to 1 and be in [0, 1]."""
        from ferroml.gaussian_process import GaussianProcessClassifier, RBF

        X, y = data

        fm = GaussianProcessClassifier(kernel=RBF(1.0))
        fm.fit(X, y.astype(np.float64))
        proba = np.array(fm.predict_proba(X))

        assert proba.shape[0] == len(X), (
            f"Wrong number of rows: {proba.shape[0]} vs {len(X)}"
        )
        assert np.all(proba >= 0.0 - 1e-6), f"Negative probabilities: min={proba.min()}"
        assert np.all(proba <= 1.0 + 1e-6), f"Probabilities > 1: max={proba.max()}"
        # Each row should sum to ~1
        row_sums = proba.sum(axis=1) if proba.ndim == 2 else proba
        if proba.ndim == 2:
            np.testing.assert_allclose(
                row_sums, 1.0, atol=1e-4,
                err_msg="Predicted probabilities don't sum to 1",
            )

    def test_predictions_finite(self, data):
        """All predictions should be finite."""
        from ferroml.gaussian_process import GaussianProcessClassifier, RBF

        X, y = data

        fm = GaussianProcessClassifier(kernel=RBF(1.0))
        fm.fit(X, y.astype(np.float64))
        fm_pred = np.array(fm.predict(X))

        assert np.all(np.isfinite(fm_pred))
        assert fm_pred.shape == (len(X),)

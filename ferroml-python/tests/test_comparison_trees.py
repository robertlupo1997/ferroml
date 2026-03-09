"""
FerroML vs sklearn comparison tests for tree and ensemble models.

Validates that FerroML tree-based models achieve comparable performance
to their sklearn equivalents on standard datasets.
"""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from conftest_comparison import (
    get_iris, get_wine, get_breast_cancer, get_diabetes,
    get_classification_data, get_regression_data,
    r2_score, accuracy_score,
)

# ── FerroML imports ──────────────────────────────────────────────────────────

from ferroml.trees import (
    DecisionTreeClassifier as FDTClassifier,
    DecisionTreeRegressor as FDTRegressor,
    RandomForestClassifier as FRFClassifier,
    RandomForestRegressor as FRFRegressor,
    GradientBoostingClassifier as FGBClassifier,
    GradientBoostingRegressor as FGBRegressor,
    HistGradientBoostingClassifier as FHGBClassifier,
    HistGradientBoostingRegressor as FHGBRegressor,
)
from ferroml.ensemble import (
    ExtraTreesClassifier as FETClassifier,
    ExtraTreesRegressor as FETRegressor,
    AdaBoostClassifier as FABClassifier,
    AdaBoostRegressor as FABRegressor,
    SGDClassifier as FSGDClassifier,
    SGDRegressor as FSGDRegressor,
)

# ── sklearn imports ──────────────────────────────────────────────────────────

from sklearn.tree import (
    DecisionTreeClassifier as SKDTClassifier,
    DecisionTreeRegressor as SKDTRegressor,
)
from sklearn.linear_model import (
    SGDClassifier as SKSGDClassifier,
    SGDRegressor as SKSGDRegressor,
)
from sklearn.ensemble import (
    RandomForestClassifier as SKRFClassifier,
    RandomForestRegressor as SKRFRegressor,
    GradientBoostingClassifier as SKGBClassifier,
    GradientBoostingRegressor as SKGBRegressor,
    HistGradientBoostingClassifier as SKHGBClassifier,
    HistGradientBoostingRegressor as SKHGBRegressor,
    ExtraTreesClassifier as SKETClassifier,
    ExtraTreesRegressor as SKETRegressor,
    AdaBoostClassifier as SKABClassifier,
    AdaBoostRegressor as SKABRegressor,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

PERF_TOL = 0.05  # 5% performance tolerance for stochastic models


def _scale(X_train, X_test=None):
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    if X_test is not None:
        return Xt, scaler.transform(X_test)
    return Xt


def _assert_clf_comparable(ferro_model, sklearn_model, X, y, tol=PERF_TOL):
    """Assert both classifiers achieve comparable accuracy."""
    ferro_model.fit(X, y)
    sklearn_model.fit(X, y)
    ferro_acc = accuracy_score(y, ferro_model.predict(X))
    sklearn_acc = accuracy_score(y, sklearn_model.predict(X))
    assert ferro_acc >= sklearn_acc - tol, (
        f"FerroML accuracy {ferro_acc:.4f} too far below "
        f"sklearn {sklearn_acc:.4f} (tol={tol})"
    )


def _assert_reg_comparable(ferro_model, sklearn_model, X, y, tol=PERF_TOL):
    """Assert both regressors achieve comparable R2."""
    ferro_model.fit(X, y)
    sklearn_model.fit(X, y)
    ferro_r2 = r2_score(y, ferro_model.predict(X))
    sklearn_r2 = r2_score(y, sklearn_model.predict(X))
    assert ferro_r2 >= sklearn_r2 - tol, (
        f"FerroML R2 {ferro_r2:.4f} too far below "
        f"sklearn {sklearn_r2:.4f} (tol={tol})"
    )


# ==========================================================================
# DecisionTreeClassifier
# ==========================================================================

class TestDecisionTreeClassifier:

    def test_iris_exact(self):
        X, y = get_iris()
        ferro = FDTClassifier(max_depth=5, random_state=42)
        sk = SKDTClassifier(max_depth=5, random_state=42)
        ferro.fit(X, y)
        sk.fit(X, y)
        ferro_acc = accuracy_score(y, ferro.predict(X))
        sk_acc = accuracy_score(y, sk.predict(X))
        # Both should achieve near-perfect on iris with depth 5
        assert ferro_acc >= 0.95
        assert abs(ferro_acc - sk_acc) < 0.02

    def test_breast_cancer(self):
        X, y = get_breast_cancer()
        _assert_clf_comparable(
            FDTClassifier(max_depth=5, random_state=42),
            SKDTClassifier(max_depth=5, random_state=42),
            X, y, tol=0.02,
        )

    def test_feature_importances_shape_iris(self):
        X, y = get_iris()
        ferro = FDTClassifier(max_depth=5, random_state=42)
        ferro.fit(X, y)
        fi = ferro.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert np.all(fi >= 0)
        assert abs(fi.sum() - 1.0) < 1e-6

    def test_synthetic_multiclass(self):
        X, y = get_classification_data(500, 10, n_classes=4)
        _assert_clf_comparable(
            FDTClassifier(max_depth=6, random_state=42),
            SKDTClassifier(max_depth=6, random_state=42),
            X, y, tol=0.05,
        )


# ==========================================================================
# DecisionTreeRegressor
# ==========================================================================

class TestDecisionTreeRegressor:

    def test_diabetes(self):
        X, y = get_diabetes()
        _assert_reg_comparable(
            FDTRegressor(max_depth=5, random_state=42),
            SKDTRegressor(max_depth=5, random_state=42),
            X, y, tol=0.02,
        )

    def test_feature_importances_shape_diabetes(self):
        X, y = get_diabetes()
        ferro = FDTRegressor(max_depth=5, random_state=42)
        ferro.fit(X, y)
        fi = ferro.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert np.all(fi >= 0)

    def test_synthetic_regression(self):
        X, y = get_regression_data(500, 10)
        _assert_reg_comparable(
            FDTRegressor(max_depth=8, random_state=42),
            SKDTRegressor(max_depth=8, random_state=42),
            X, y, tol=0.05,
        )


# ==========================================================================
# RandomForestClassifier
# ==========================================================================

class TestRandomForestClassifier:

    def test_iris(self):
        X, y = get_iris()
        _assert_clf_comparable(
            FRFClassifier(n_estimators=100, max_depth=5, random_state=42),
            SKRFClassifier(n_estimators=100, max_depth=5, random_state=42),
            X, y,
        )

    def test_breast_cancer(self):
        X, y = get_breast_cancer()
        X = _scale(X)
        _assert_clf_comparable(
            FRFClassifier(n_estimators=100, max_depth=8, random_state=42),
            SKRFClassifier(n_estimators=100, max_depth=8, random_state=42),
            X, y,
        )

    def test_wine(self):
        X, y = get_wine()
        X = _scale(X)
        _assert_clf_comparable(
            FRFClassifier(n_estimators=100, max_depth=6, random_state=42),
            SKRFClassifier(n_estimators=100, max_depth=6, random_state=42),
            X, y,
        )

    def test_feature_importances_shape(self):
        X, y = get_iris()
        ferro = FRFClassifier(n_estimators=50, max_depth=5, random_state=42)
        ferro.fit(X, y)
        fi = ferro.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert np.all(fi >= 0)


# ==========================================================================
# RandomForestRegressor
# ==========================================================================

class TestRandomForestRegressor:

    def test_diabetes(self):
        X, y = get_diabetes()
        _assert_reg_comparable(
            FRFRegressor(n_estimators=100, max_depth=8, random_state=42),
            SKRFRegressor(n_estimators=100, max_depth=8, random_state=42),
            X, y,
        )

    def test_synthetic(self):
        X, y = get_regression_data(500, 10)
        # Wider tolerance: RNG differs, so RF on synthetic data can diverge more
        _assert_reg_comparable(
            FRFRegressor(n_estimators=100, max_depth=8, random_state=42),
            SKRFRegressor(n_estimators=100, max_depth=8, random_state=42),
            X, y, tol=0.20,
        )

    def test_feature_importances_shape(self):
        X, y = get_diabetes()
        ferro = FRFRegressor(n_estimators=50, max_depth=5, random_state=42)
        ferro.fit(X, y)
        fi = ferro.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert np.all(fi >= 0)


# ==========================================================================
# GradientBoostingClassifier
# ==========================================================================

class TestGradientBoostingClassifier:

    def test_iris(self):
        X, y = get_iris()
        _assert_clf_comparable(
            FGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
            SKGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
            X, y,
        )

    def test_breast_cancer(self):
        X, y = get_breast_cancer()
        X = _scale(X)
        _assert_clf_comparable(
            FGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
            SKGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
            X, y,
        )

    def test_feature_importances_shape(self):
        X, y = get_iris()
        ferro = FGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        ferro.fit(X, y)
        fi = ferro.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert np.all(fi >= 0)

    def test_synthetic_binary(self):
        X, y = get_classification_data(500, 10, n_classes=2)
        # Wider tolerance: tree-split RNG differs between implementations
        _assert_clf_comparable(
            FGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
            SKGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
            X, y, tol=0.10,
        )


# ==========================================================================
# GradientBoostingRegressor
# ==========================================================================

class TestGradientBoostingRegressor:

    def test_diabetes(self):
        X, y = get_diabetes()
        _assert_reg_comparable(
            FGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
            SKGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
            X, y,
        )

    def test_feature_importances_shape(self):
        X, y = get_diabetes()
        ferro = FGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        ferro.fit(X, y)
        fi = ferro.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert np.all(fi >= 0)

    def test_synthetic(self):
        X, y = get_regression_data(500, 10)
        _assert_reg_comparable(
            FGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
            SKGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
            X, y,
        )


# ==========================================================================
# HistGradientBoostingClassifier
# ==========================================================================

class TestHistGradientBoostingClassifier:

    def test_breast_cancer(self):
        X, y = get_breast_cancer()
        X = _scale(X)
        _assert_clf_comparable(
            FHGBClassifier(max_iter=100, max_depth=4, learning_rate=0.1, random_state=42),
            SKHGBClassifier(max_iter=100, max_depth=4, learning_rate=0.1, random_state=42),
            X, y,
        )

    def test_wine(self):
        X, y = get_wine()
        X = _scale(X)
        _assert_clf_comparable(
            FHGBClassifier(max_iter=100, max_depth=4, learning_rate=0.1, random_state=42),
            SKHGBClassifier(max_iter=100, max_depth=4, learning_rate=0.1, random_state=42),
            X, y,
        )

    def test_feature_importances_shape(self):
        X, y = get_breast_cancer()
        ferro = FHGBClassifier(max_iter=50, max_depth=3, learning_rate=0.1, random_state=42)
        ferro.fit(X, y)
        fi = ferro.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert np.all(fi >= 0)


# ==========================================================================
# HistGradientBoostingRegressor
# ==========================================================================

class TestHistGradientBoostingRegressor:

    def test_diabetes(self):
        X, y = get_diabetes()
        _assert_reg_comparable(
            FHGBRegressor(max_iter=100, max_depth=4, learning_rate=0.1, random_state=42),
            SKHGBRegressor(max_iter=100, max_depth=4, learning_rate=0.1, random_state=42),
            X, y,
        )

    def test_feature_importances_shape(self):
        X, y = get_diabetes()
        ferro = FHGBRegressor(max_iter=50, max_depth=3, learning_rate=0.1, random_state=42)
        ferro.fit(X, y)
        fi = ferro.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert np.all(fi >= 0)

    def test_synthetic(self):
        X, y = get_regression_data(500, 10)
        _assert_reg_comparable(
            FHGBRegressor(max_iter=100, max_depth=4, learning_rate=0.1, random_state=42),
            SKHGBRegressor(max_iter=100, max_depth=4, learning_rate=0.1, random_state=42),
            X, y,
        )


# ==========================================================================
# ExtraTreesClassifier
# ==========================================================================

class TestExtraTreesClassifier:

    def test_iris(self):
        X, y = get_iris()
        _assert_clf_comparable(
            FETClassifier(n_estimators=100, max_depth=5, random_state=42),
            SKETClassifier(n_estimators=100, max_depth=5, random_state=42),
            X, y,
        )

    def test_wine(self):
        X, y = get_wine()
        X = _scale(X)
        _assert_clf_comparable(
            FETClassifier(n_estimators=100, max_depth=6, random_state=42),
            SKETClassifier(n_estimators=100, max_depth=6, random_state=42),
            X, y,
        )

    def test_feature_importances_shape(self):
        X, y = get_iris()
        ferro = FETClassifier(n_estimators=50, max_depth=5, random_state=42)
        ferro.fit(X, y)
        fi = ferro.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert np.all(fi >= 0)


# ==========================================================================
# ExtraTreesRegressor
# ==========================================================================

class TestExtraTreesRegressor:

    def test_diabetes(self):
        X, y = get_diabetes()
        _assert_reg_comparable(
            FETRegressor(n_estimators=100, max_depth=8, random_state=42),
            SKETRegressor(n_estimators=100, max_depth=8, random_state=42),
            X, y,
        )

    def test_feature_importances_shape(self):
        X, y = get_diabetes()
        ferro = FETRegressor(n_estimators=50, max_depth=5, random_state=42)
        ferro.fit(X, y)
        fi = ferro.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert np.all(fi >= 0)

    def test_synthetic(self):
        X, y = get_regression_data(500, 10)
        _assert_reg_comparable(
            FETRegressor(n_estimators=50, max_depth=6, random_state=42),
            SKETRegressor(n_estimators=50, max_depth=6, random_state=42),
            X, y,
        )


# ==========================================================================
# AdaBoostClassifier
# ==========================================================================

class TestAdaBoostClassifier:

    def test_iris(self):
        X, y = get_iris()
        _assert_clf_comparable(
            FABClassifier(n_estimators=50, learning_rate=1.0, random_state=42),
            SKABClassifier(n_estimators=50, learning_rate=1.0, random_state=42),
            X, y,
        )

    def test_breast_cancer(self):
        X, y = get_breast_cancer()
        X = _scale(X)
        _assert_clf_comparable(
            FABClassifier(n_estimators=50, learning_rate=1.0, random_state=42),
            SKABClassifier(n_estimators=50, learning_rate=1.0, random_state=42),
            X, y,
        )

    def test_synthetic_binary(self):
        X, y = get_classification_data(500, 10, n_classes=2)
        _assert_clf_comparable(
            FABClassifier(n_estimators=50, learning_rate=0.5, random_state=42),
            SKABClassifier(n_estimators=50, learning_rate=0.5, random_state=42),
            X, y,
        )


# ==========================================================================
# AdaBoostRegressor
# ==========================================================================

class TestAdaBoostRegressor:

    def test_diabetes(self):
        X, y = get_diabetes()
        _assert_reg_comparable(
            FABRegressor(n_estimators=50, learning_rate=1.0, random_state=42),
            SKABRegressor(n_estimators=50, learning_rate=1.0, random_state=42),
            X, y,
        )

    def test_synthetic(self):
        X, y = get_regression_data(500, 10)
        _assert_reg_comparable(
            FABRegressor(n_estimators=50, learning_rate=0.5, random_state=42),
            SKABRegressor(n_estimators=50, learning_rate=0.5, random_state=42),
            X, y,
        )


# ==========================================================================
# Cross-model comparison: prediction distribution sanity checks
# ==========================================================================

class TestEnsemblePredictionDistributions:

    def test_classifiers_all_beat_majority_baseline_iris(self):
        """All ensemble classifiers should beat majority-class baseline on iris."""
        X, y = get_iris()
        majority_acc = max(np.bincount(y.astype(int))) / len(y)
        models = [
            ("RF", FRFClassifier(n_estimators=50, max_depth=5, random_state=42)),
            ("GB", FGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)),
            ("ET", FETClassifier(n_estimators=50, max_depth=5, random_state=42)),
            ("AB", FABClassifier(n_estimators=50, learning_rate=1.0, random_state=42)),
        ]
        for name, model in models:
            model.fit(X, y)
            acc = accuracy_score(y, model.predict(X))
            assert acc > majority_acc, (
                f"{name} accuracy {acc:.4f} did not beat majority baseline {majority_acc:.4f}"
            )

    def test_regressors_all_beat_mean_baseline_diabetes(self):
        """All ensemble regressors should beat mean-prediction baseline on diabetes."""
        X, y = get_diabetes()
        mean_r2 = 0.0  # R2 of constant mean predictor
        models = [
            ("RF", FRFRegressor(n_estimators=50, max_depth=8, random_state=42)),
            ("GB", FGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)),
            ("ET", FETRegressor(n_estimators=50, max_depth=8, random_state=42)),
            ("AB", FABRegressor(n_estimators=50, learning_rate=1.0, random_state=42)),
        ]
        for name, model in models:
            model.fit(X, y)
            r2 = r2_score(y, model.predict(X))
            assert r2 > mean_r2, (
                f"{name} R2 {r2:.4f} did not beat mean baseline"
            )

    def test_hgb_comparable_to_gb_breast_cancer(self):
        """HGB and GB should achieve similar accuracy on breast cancer."""
        X, y = get_breast_cancer()
        X = _scale(X)
        gb = FGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        hgb = FHGBClassifier(max_iter=100, max_depth=3, learning_rate=0.1, random_state=42)
        gb.fit(X, y)
        hgb.fit(X, y)
        gb_acc = accuracy_score(y, gb.predict(X))
        hgb_acc = accuracy_score(y, hgb.predict(X))
        assert abs(gb_acc - hgb_acc) < 0.05, (
            f"GB ({gb_acc:.4f}) and HGB ({hgb_acc:.4f}) too different"
        )

    def test_hgb_comparable_to_gb_diabetes(self):
        """HGB and GB should achieve similar R2 on diabetes."""
        X, y = get_diabetes()
        gb = FGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        hgb = FHGBRegressor(max_iter=100, max_depth=3, learning_rate=0.1, random_state=42)
        gb.fit(X, y)
        hgb.fit(X, y)
        gb_r2 = r2_score(y, gb.predict(X))
        hgb_r2 = r2_score(y, hgb.predict(X))
        assert abs(gb_r2 - hgb_r2) < 0.10, (
            f"GB R2 ({gb_r2:.4f}) and HGB R2 ({hgb_r2:.4f}) too different"
        )


# ==========================================================================
# SGDClassifier
# ==========================================================================

SGD_TOL = 0.10  # 10% tolerance for SGD (stochastic + different implementations)


class TestSGDClassifier:

    def test_sgd_classifier_breast_cancer(self):
        """SGDClassifier should achieve comparable accuracy to sklearn on breast cancer."""
        X, y = get_breast_cancer()
        X = _scale(X)
        ferro = FSGDClassifier(loss="hinge", penalty="l2", alpha=0.0001,
                               max_iter=1000, tol=1e-3, random_state=42)
        sk = SKSGDClassifier(loss="hinge", penalty="l2", alpha=0.0001,
                             max_iter=1000, tol=1e-3, random_state=42)
        _assert_clf_comparable(ferro, sk, X, y, tol=SGD_TOL)

    def test_sgd_classifier_log_loss_breast_cancer(self):
        """SGDClassifier with log loss should achieve comparable accuracy."""
        X, y = get_breast_cancer()
        X = _scale(X)
        ferro = FSGDClassifier(loss="log", penalty="l2", alpha=0.0001,
                               max_iter=1000, tol=1e-3, random_state=42)
        sk = SKSGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001,
                             max_iter=1000, tol=1e-3, random_state=42)
        _assert_clf_comparable(ferro, sk, X, y, tol=SGD_TOL)

    def test_sgd_classifier_beats_majority(self):
        """SGDClassifier should beat majority-class baseline."""
        X, y = get_breast_cancer()
        X = _scale(X)
        majority_acc = max(np.mean(y == 0), np.mean(y == 1))
        ferro = FSGDClassifier(loss="hinge", random_state=42)
        ferro.fit(X, y)
        acc = accuracy_score(y, ferro.predict(X))
        assert acc > majority_acc, (
            f"SGDClassifier accuracy {acc:.4f} did not beat majority baseline {majority_acc:.4f}"
        )


# ==========================================================================
# SGDRegressor
# ==========================================================================


class TestSGDRegressor:

    def test_sgd_regressor_diabetes(self):
        """SGDRegressor should achieve comparable R2 to sklearn on diabetes."""
        X, y = get_diabetes()
        X = _scale(X)
        ferro = FSGDRegressor(loss="squared_error", penalty="l2", alpha=0.0001,
                              max_iter=1000, tol=1e-3, random_state=42)
        sk = SKSGDRegressor(loss="squared_error", penalty="l2", alpha=0.0001,
                            max_iter=1000, tol=1e-3, random_state=42)
        _assert_reg_comparable(ferro, sk, X, y, tol=SGD_TOL)

    def test_sgd_regressor_huber_diabetes(self):
        """SGDRegressor with Huber loss should achieve comparable R2."""
        X, y = get_diabetes()
        X = _scale(X)
        ferro = FSGDRegressor(loss="huber", penalty="l2", alpha=0.0001,
                              max_iter=1000, tol=1e-3, random_state=42)
        sk = SKSGDRegressor(loss="huber", penalty="l2", alpha=0.0001,
                            max_iter=1000, tol=1e-3, random_state=42)
        _assert_reg_comparable(ferro, sk, X, y, tol=SGD_TOL)

    def test_sgd_regressor_beats_mean(self):
        """SGDRegressor should beat mean-prediction baseline (R2 > 0)."""
        X, y = get_diabetes()
        X = _scale(X)
        ferro = FSGDRegressor(loss="squared_error", random_state=42)
        ferro.fit(X, y)
        r2 = r2_score(y, ferro.predict(X))
        assert r2 > 0.0, (
            f"SGDRegressor R2 {r2:.4f} did not beat mean baseline"
        )

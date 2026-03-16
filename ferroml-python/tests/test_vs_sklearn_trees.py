"""
FerroML vs sklearn: Tree-based ensemble models.

Cross-library validation for:
1. ExtraTreesClassifier / ExtraTreesRegressor
2. GradientBoostingClassifier / GradientBoostingRegressor
3. HistGradientBoostingClassifier / HistGradientBoostingRegressor

Phase X.3 — Plan X production-readiness validation.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cls_data():
    """Binary classification dataset with good separability."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        class_sep=1.5,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture()
def reg_data():
    """Regression dataset with moderate noise."""
    X, y = make_regression(
        n_samples=300,
        n_features=10,
        n_informative=6,
        noise=10.0,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


# ===========================================================================
# 1. ExtraTreesClassifier
# ===========================================================================

class TestExtraTreesClassifierVsSklearn:
    """Compare FerroML ExtraTreesClassifier against sklearn."""

    def test_accuracy_within_5pct(self, cls_data):
        from ferroml.ensemble import ExtraTreesClassifier

        from sklearn.ensemble import ExtraTreesClassifier as SkETC

        X_train, X_test, y_train, y_test = cls_data

        sk = SkETC(n_estimators=100, max_depth=10, random_state=42)
        sk.fit(X_train, y_train)
        sk_acc = accuracy_score(y_test, sk.predict(X_test))

        fm = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42)
        fm.fit(X_train, y_train.astype(np.float64))
        fm_pred = np.array(fm.predict(X_test))
        fm_acc = accuracy_score(y_test, fm_pred)

        assert sk_acc > 0.75, f"sklearn baseline too low: {sk_acc}"
        assert abs(fm_acc - sk_acc) < 0.05, (
            f"Accuracy gap > 5%: ferroml={fm_acc:.4f}, sklearn={sk_acc:.4f}"
        )

    def test_predictions_finite(self, cls_data):
        from ferroml.ensemble import ExtraTreesClassifier

        X_train, X_test, y_train, y_test = cls_data

        fm = ExtraTreesClassifier(n_estimators=50, random_state=42)
        fm.fit(X_train, y_train.astype(np.float64))
        pred = np.array(fm.predict(X_test))

        assert np.all(np.isfinite(pred))
        assert set(np.unique(pred)).issubset({0.0, 1.0})


# ===========================================================================
# 2. ExtraTreesRegressor
# ===========================================================================

class TestExtraTreesRegressorVsSklearn:
    """Compare FerroML ExtraTreesRegressor against sklearn."""

    def test_r2_within_005(self, reg_data):
        from ferroml.ensemble import ExtraTreesRegressor

        from sklearn.ensemble import ExtraTreesRegressor as SkETR

        X_train, X_test, y_train, y_test = reg_data

        sk = SkETR(n_estimators=100, max_depth=10, random_state=42)
        sk.fit(X_train, y_train)
        sk_r2 = r2_score(y_test, sk.predict(X_test))

        fm = ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42)
        fm.fit(X_train, y_train.astype(np.float64))
        fm_pred = np.array(fm.predict(X_test))
        fm_r2 = r2_score(y_test, fm_pred)

        assert sk_r2 > 0.5, f"sklearn baseline too low: {sk_r2}"
        assert abs(fm_r2 - sk_r2) < 0.05, (
            f"R2 gap > 0.05: ferroml={fm_r2:.4f}, sklearn={sk_r2:.4f}"
        )

    def test_predictions_finite(self, reg_data):
        from ferroml.ensemble import ExtraTreesRegressor

        X_train, X_test, y_train, y_test = reg_data

        fm = ExtraTreesRegressor(n_estimators=50, random_state=42)
        fm.fit(X_train, y_train.astype(np.float64))
        pred = np.array(fm.predict(X_test))

        assert np.all(np.isfinite(pred))
        assert pred.shape == (len(X_test),)


# ===========================================================================
# 3. GradientBoostingClassifier
# ===========================================================================

class TestGradientBoostingClassifierVsSklearn:
    """Compare FerroML GradientBoostingClassifier against sklearn."""

    def test_accuracy_within_5pct(self, cls_data):
        from ferroml.trees import GradientBoostingClassifier

        from sklearn.ensemble import GradientBoostingClassifier as SkGBC

        X_train, X_test, y_train, y_test = cls_data

        sk = SkGBC(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        sk.fit(X_train, y_train)
        sk_acc = accuracy_score(y_test, sk.predict(X_test))

        fm = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        fm.fit(X_train, y_train.astype(np.float64))
        fm_pred = np.array(fm.predict(X_test))
        fm_acc = accuracy_score(y_test, fm_pred)

        assert sk_acc > 0.75, f"sklearn baseline too low: {sk_acc}"
        assert abs(fm_acc - sk_acc) < 0.05, (
            f"Accuracy gap > 5%: ferroml={fm_acc:.4f}, sklearn={sk_acc:.4f}"
        )

    def test_increasing_estimators_improves(self, cls_data):
        """More estimators should generally not hurt performance."""
        from ferroml.trees import GradientBoostingClassifier

        X_train, X_test, y_train, y_test = cls_data

        accs = []
        for n_est in [10, 50, 100]:
            fm = GradientBoostingClassifier(
                n_estimators=n_est,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
            )
            fm.fit(X_train, y_train.astype(np.float64))
            pred = np.array(fm.predict(X_test))
            accs.append(accuracy_score(y_test, pred))

        # Last should be >= first - small tolerance for variance
        assert accs[-1] >= accs[0] - 0.03, (
            f"More estimators hurt: {accs}"
        )


# ===========================================================================
# 4. GradientBoostingRegressor
# ===========================================================================

class TestGradientBoostingRegressorVsSklearn:
    """Compare FerroML GradientBoostingRegressor against sklearn."""

    def test_r2_within_005(self, reg_data):
        from ferroml.trees import GradientBoostingRegressor

        from sklearn.ensemble import GradientBoostingRegressor as SkGBR

        X_train, X_test, y_train, y_test = reg_data

        sk = SkGBR(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        sk.fit(X_train, y_train)
        sk_r2 = r2_score(y_test, sk.predict(X_test))

        fm = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        fm.fit(X_train, y_train.astype(np.float64))
        fm_pred = np.array(fm.predict(X_test))
        fm_r2 = r2_score(y_test, fm_pred)

        assert sk_r2 > 0.5, f"sklearn baseline too low: {sk_r2}"
        assert abs(fm_r2 - sk_r2) < 0.05, (
            f"R2 gap > 0.05: ferroml={fm_r2:.4f}, sklearn={sk_r2:.4f}"
        )

    def test_predictions_finite(self, reg_data):
        from ferroml.trees import GradientBoostingRegressor

        X_train, X_test, y_train, y_test = reg_data

        fm = GradientBoostingRegressor(
            n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42,
        )
        fm.fit(X_train, y_train.astype(np.float64))
        pred = np.array(fm.predict(X_test))

        assert np.all(np.isfinite(pred))
        assert pred.shape == (len(X_test),)


# ===========================================================================
# 5. HistGradientBoostingClassifier
# ===========================================================================

class TestHistGBClassifierVsSklearn:
    """Compare FerroML HistGradientBoostingClassifier against sklearn."""

    def test_accuracy_within_5pct(self, cls_data):
        from ferroml.trees import HistGradientBoostingClassifier

        from sklearn.ensemble import HistGradientBoostingClassifier as SkHGBC

        X_train, X_test, y_train, y_test = cls_data

        sk = SkHGBC(
            max_iter=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
        sk.fit(X_train, y_train)
        sk_acc = accuracy_score(y_test, sk.predict(X_test))

        fm = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
        fm.fit(X_train, y_train.astype(np.float64))
        fm_pred = np.array(fm.predict(X_test))
        fm_acc = accuracy_score(y_test, fm_pred)

        assert sk_acc > 0.75, f"sklearn baseline too low: {sk_acc}"
        assert abs(fm_acc - sk_acc) < 0.05, (
            f"Accuracy gap > 5%: ferroml={fm_acc:.4f}, sklearn={sk_acc:.4f}"
        )

    def test_predictions_binary(self, cls_data):
        from ferroml.trees import HistGradientBoostingClassifier

        X_train, X_test, y_train, y_test = cls_data

        fm = HistGradientBoostingClassifier(max_iter=50, random_state=42)
        fm.fit(X_train, y_train.astype(np.float64))
        pred = np.array(fm.predict(X_test))

        assert set(np.unique(pred)).issubset({0.0, 1.0})


# ===========================================================================
# 6. HistGradientBoostingRegressor
# ===========================================================================

class TestHistGBRegressorVsSklearn:
    """Compare FerroML HistGradientBoostingRegressor against sklearn."""

    def test_r2_within_005(self, reg_data):
        from ferroml.trees import HistGradientBoostingRegressor

        from sklearn.ensemble import HistGradientBoostingRegressor as SkHGBR

        X_train, X_test, y_train, y_test = reg_data

        sk = SkHGBR(
            max_iter=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
        sk.fit(X_train, y_train)
        sk_r2 = r2_score(y_test, sk.predict(X_test))

        fm = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
        fm.fit(X_train, y_train.astype(np.float64))
        fm_pred = np.array(fm.predict(X_test))
        fm_r2 = r2_score(y_test, fm_pred)

        assert sk_r2 > 0.5, f"sklearn baseline too low: {sk_r2}"
        assert abs(fm_r2 - sk_r2) < 0.05, (
            f"R2 gap > 0.05: ferroml={fm_r2:.4f}, sklearn={sk_r2:.4f}"
        )

    def test_predictions_finite(self, reg_data):
        from ferroml.trees import HistGradientBoostingRegressor

        X_train, X_test, y_train, y_test = reg_data

        fm = HistGradientBoostingRegressor(max_iter=50, random_state=42)
        fm.fit(X_train, y_train.astype(np.float64))
        pred = np.array(fm.predict(X_test))

        assert np.all(np.isfinite(pred))
        assert pred.shape == (len(X_test),)

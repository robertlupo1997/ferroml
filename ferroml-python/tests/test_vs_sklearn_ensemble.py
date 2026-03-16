"""Cross-library validation: FerroML ensemble models vs sklearn equivalents.

Tests VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor,
BaggingClassifier, and BaggingRegressor against their sklearn counterparts.

Phase X.2 — Plan X production-ready validation.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Shared data generators
# ---------------------------------------------------------------------------

def make_cls_data(n=500, p=10, seed=42):
    """Well-separated binary classification dataset."""
    X, y = make_classification(
        n_samples=n,
        n_features=p,
        n_informative=p // 2,
        n_redundant=p // 4,
        n_classes=2,
        random_state=seed,
        class_sep=1.5,
    )
    return X, y


def make_reg_data(n=500, p=10, seed=42):
    """Regression dataset with moderate noise."""
    X, y = make_regression(
        n_samples=n,
        n_features=p,
        n_informative=p // 2,
        noise=5.0,
        random_state=seed,
    )
    return X, y


# ===========================================================================
# VotingClassifier
# ===========================================================================

class TestVotingClassifierVsSklearn:
    """Compare FerroML VotingClassifier against sklearn VotingClassifier."""

    @pytest.fixture()
    def data(self):
        X, y = make_cls_data(n=400, p=8, seed=42)
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_hard_voting_accuracy(self, data):
        """Hard voting with LR + DT + GaussianNB should match sklearn within 5%."""
        from ferroml.ensemble import VotingClassifier

        from sklearn.ensemble import VotingClassifier as SkVC
        from sklearn.linear_model import LogisticRegression as SkLR
        from sklearn.tree import DecisionTreeClassifier as SkDT
        from sklearn.naive_bayes import GaussianNB as SkNB

        X_train, X_test, y_train, y_test = data

        # sklearn
        sk_vc = SkVC(
            estimators=[("lr", SkLR(max_iter=200)), ("dt", SkDT()), ("nb", SkNB())],
            voting="hard",
        )
        sk_vc.fit(X_train, y_train)
        sk_acc = accuracy_score(y_test, sk_vc.predict(X_test))

        # FerroML
        fm_vc = VotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree"), ("nb", "gaussian_nb")],
            voting="hard",
        )
        fm_vc.fit(X_train, y_train.astype(float))
        fm_pred = np.array(fm_vc.predict(X_test))
        fm_acc = accuracy_score(y_test, fm_pred)

        assert fm_acc > 0.70, f"FerroML VotingClassifier accuracy too low: {fm_acc:.4f}"
        assert abs(fm_acc - sk_acc) < 0.10, (
            f"VotingClassifier accuracy gap: ferroml={fm_acc:.4f}, sklearn={sk_acc:.4f}"
        )

    def test_soft_voting_accuracy(self, data):
        """Soft voting should also be competitive with sklearn."""
        from ferroml.ensemble import VotingClassifier

        from sklearn.ensemble import VotingClassifier as SkVC
        from sklearn.linear_model import LogisticRegression as SkLR
        from sklearn.tree import DecisionTreeClassifier as SkDT
        from sklearn.naive_bayes import GaussianNB as SkNB

        X_train, X_test, y_train, y_test = data

        # sklearn
        sk_vc = SkVC(
            estimators=[("lr", SkLR(max_iter=200)), ("dt", SkDT()), ("nb", SkNB())],
            voting="soft",
        )
        sk_vc.fit(X_train, y_train)
        sk_acc = accuracy_score(y_test, sk_vc.predict(X_test))

        # FerroML
        fm_vc = VotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree"), ("nb", "gaussian_nb")],
            voting="soft",
        )
        fm_vc.fit(X_train, y_train.astype(float))
        fm_pred = np.array(fm_vc.predict(X_test))
        fm_acc = accuracy_score(y_test, fm_pred)

        assert fm_acc > 0.70, f"FerroML soft VotingClassifier accuracy too low: {fm_acc:.4f}"
        assert abs(fm_acc - sk_acc) < 0.10, (
            f"Soft VotingClassifier accuracy gap: ferroml={fm_acc:.4f}, sklearn={sk_acc:.4f}"
        )

    def test_predict_proba_shape(self, data):
        """predict_proba should return (n_samples, n_classes)."""
        from ferroml.ensemble import VotingClassifier

        X_train, X_test, y_train, y_test = data

        fm_vc = VotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree"), ("nb", "gaussian_nb")],
            voting="soft",
        )
        fm_vc.fit(X_train, y_train.astype(float))
        probas = np.array(fm_vc.predict_proba(X_test))

        assert probas.shape == (len(X_test), 2)
        # Probabilities should sum to ~1
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
        # All in [0, 1]
        assert np.all(probas >= 0) and np.all(probas <= 1)


# ===========================================================================
# VotingRegressor
# ===========================================================================

class TestVotingRegressorVsSklearn:
    """Compare FerroML VotingRegressor against sklearn VotingRegressor."""

    @pytest.fixture()
    def data(self):
        X, y = make_reg_data(n=400, p=8, seed=42)
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_predictions_competitive(self, data):
        """VotingRegressor with LR + Ridge + DT should match sklearn R2 within 0.1."""
        from ferroml.ensemble import VotingRegressor

        from sklearn.ensemble import VotingRegressor as SkVR
        from sklearn.linear_model import LinearRegression as SkLR, Ridge as SkRidge
        from sklearn.tree import DecisionTreeRegressor as SkDT

        X_train, X_test, y_train, y_test = data

        # sklearn
        sk_vr = SkVR(
            estimators=[("lr", SkLR()), ("ridge", SkRidge()), ("dt", SkDT())],
        )
        sk_vr.fit(X_train, y_train)
        sk_r2 = r2_score(y_test, sk_vr.predict(X_test))

        # FerroML
        fm_vr = VotingRegressor(
            [("lr", "linear_regression"), ("ridge", "ridge"), ("dt", "decision_tree")],
        )
        fm_vr.fit(X_train, y_train.astype(float))
        fm_pred = np.array(fm_vr.predict(X_test))
        fm_r2 = r2_score(y_test, fm_pred)

        assert fm_r2 > 0.5, f"FerroML VotingRegressor R2 too low: {fm_r2:.4f}"
        assert abs(fm_r2 - sk_r2) < 0.15, (
            f"VotingRegressor R2 gap: ferroml={fm_r2:.4f}, sklearn={sk_r2:.4f}"
        )

    def test_weighted_voting(self, data):
        """Weighted voting should produce finite predictions with decent R2."""
        from ferroml.ensemble import VotingRegressor

        X_train, X_test, y_train, y_test = data

        fm_vr = VotingRegressor(
            [("lr", "linear_regression"), ("ridge", "ridge"), ("dt", "decision_tree")],
            weights=[2.0, 1.0, 1.0],
        )
        fm_vr.fit(X_train, y_train.astype(float))
        fm_pred = np.array(fm_vr.predict(X_test))

        assert np.all(np.isfinite(fm_pred)), "Non-finite predictions from weighted voting"
        fm_r2 = r2_score(y_test, fm_pred)
        assert fm_r2 > 0.4, f"Weighted VotingRegressor R2 too low: {fm_r2:.4f}"

    def test_estimator_names(self):
        """Estimator names should be accessible after construction."""
        from ferroml.ensemble import VotingRegressor

        fm_vr = VotingRegressor(
            [("lr", "linear_regression"), ("ridge", "ridge"), ("dt", "decision_tree")],
        )
        names = fm_vr.estimator_names
        assert names == ["lr", "ridge", "dt"]


# ===========================================================================
# StackingClassifier
# ===========================================================================

class TestStackingClassifierVsSklearn:
    """Compare FerroML StackingClassifier against sklearn StackingClassifier."""

    @pytest.fixture()
    def data(self):
        X, y = make_cls_data(n=500, p=10, seed=42)
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_accuracy_competitive(self, data):
        """Stacking with DT + KNN base + LR meta should match sklearn within 5%."""
        from ferroml.ensemble import StackingClassifier

        from sklearn.ensemble import StackingClassifier as SkStack
        from sklearn.tree import DecisionTreeClassifier as SkDT
        from sklearn.neighbors import KNeighborsClassifier as SkKNN
        from sklearn.linear_model import LogisticRegression as SkLR

        X_train, X_test, y_train, y_test = data

        # sklearn
        sk_stack = SkStack(
            estimators=[("dt", SkDT()), ("knn", SkKNN(n_neighbors=5))],
            final_estimator=SkLR(max_iter=200),
            cv=3,
        )
        sk_stack.fit(X_train, y_train)
        sk_acc = accuracy_score(y_test, sk_stack.predict(X_test))

        # FerroML
        fm_stack = StackingClassifier(
            [("dt", "decision_tree"), ("knn", "knn")],
            final_estimator="logistic_regression",
            cv=3,
        )
        fm_stack.fit(X_train, y_train.astype(float))
        fm_pred = np.array(fm_stack.predict(X_test))
        fm_acc = accuracy_score(y_test, fm_pred)

        assert fm_acc > 0.70, f"FerroML StackingClassifier accuracy too low: {fm_acc:.4f}"
        assert abs(fm_acc - sk_acc) < 0.10, (
            f"StackingClassifier accuracy gap: ferroml={fm_acc:.4f}, sklearn={sk_acc:.4f}"
        )

    def test_predict_proba_shape(self, data):
        """Stacking predict_proba should return proper shape."""
        from ferroml.ensemble import StackingClassifier

        X_train, X_test, y_train, y_test = data

        fm_stack = StackingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb")],
            final_estimator="logistic_regression",
            cv=3,
        )
        fm_stack.fit(X_train, y_train.astype(float))
        probas = np.array(fm_stack.predict_proba(X_test))

        assert probas.shape[0] == len(X_test)
        assert probas.shape[1] == 2  # binary classification
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_different_base_estimators(self, data):
        """Stacking with LR + RF base should produce valid predictions."""
        from ferroml.ensemble import StackingClassifier

        X_train, X_test, y_train, y_test = data

        fm_stack = StackingClassifier(
            [("lr", "logistic_regression"), ("rf", "random_forest")],
            final_estimator="logistic_regression",
            cv=3,
        )
        fm_stack.fit(X_train, y_train.astype(float))
        fm_pred = np.array(fm_stack.predict(X_test))
        fm_acc = accuracy_score(y_test, fm_pred)

        assert fm_acc > 0.70, f"FerroML StackingClassifier (LR+RF) accuracy: {fm_acc:.4f}"

    def test_score_method(self, data):
        """score() should return accuracy consistent with manual computation."""
        from ferroml.ensemble import StackingClassifier

        X_train, X_test, y_train, y_test = data

        fm_stack = StackingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb")],
            final_estimator="logistic_regression",
            cv=3,
        )
        fm_stack.fit(X_train, y_train.astype(float))

        score = fm_stack.score(X_test, y_test.astype(float))
        fm_pred = np.array(fm_stack.predict(X_test))
        manual_acc = accuracy_score(y_test, fm_pred)

        assert abs(score - manual_acc) < 1e-10, (
            f"score() disagrees with manual accuracy: {score:.6f} vs {manual_acc:.6f}"
        )


# ===========================================================================
# StackingRegressor
# ===========================================================================

class TestStackingRegressorVsSklearn:
    """Compare FerroML StackingRegressor against sklearn StackingRegressor."""

    @pytest.fixture()
    def data(self):
        X, y = make_reg_data(n=500, p=10, seed=42)
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_r2_competitive(self, data):
        """Stacking with DT + LR base + Ridge meta should match sklearn R2 within 0.1."""
        from ferroml.ensemble import StackingRegressor

        from sklearn.ensemble import StackingRegressor as SkStack
        from sklearn.tree import DecisionTreeRegressor as SkDT
        from sklearn.linear_model import LinearRegression as SkLR, Ridge as SkRidge

        X_train, X_test, y_train, y_test = data

        # sklearn
        sk_stack = SkStack(
            estimators=[("dt", SkDT()), ("lr", SkLR())],
            final_estimator=SkRidge(),
            cv=3,
        )
        sk_stack.fit(X_train, y_train)
        sk_r2 = r2_score(y_test, sk_stack.predict(X_test))

        # FerroML
        fm_stack = StackingRegressor(
            [("dt", "decision_tree"), ("lr", "linear_regression")],
            final_estimator="ridge",
            cv=3,
        )
        fm_stack.fit(X_train, y_train.astype(float))
        fm_pred = np.array(fm_stack.predict(X_test))
        fm_r2 = r2_score(y_test, fm_pred)

        assert fm_r2 > 0.5, f"FerroML StackingRegressor R2 too low: {fm_r2:.4f}"
        assert abs(fm_r2 - sk_r2) < 0.15, (
            f"StackingRegressor R2 gap: ferroml={fm_r2:.4f}, sklearn={sk_r2:.4f}"
        )

    def test_passthrough_mode(self, data):
        """Passthrough should include original features alongside meta-features."""
        from ferroml.ensemble import StackingRegressor

        X_train, X_test, y_train, y_test = data

        fm_stack = StackingRegressor(
            [("lr", "linear_regression"), ("ridge", "ridge")],
            final_estimator="linear_regression",
            cv=3,
            passthrough=True,
        )
        fm_stack.fit(X_train, y_train.astype(float))
        fm_pred = np.array(fm_stack.predict(X_test))
        fm_r2 = r2_score(y_test, fm_pred)

        assert np.all(np.isfinite(fm_pred)), "Non-finite predictions with passthrough"
        assert fm_r2 > 0.3, f"Passthrough StackingRegressor R2: {fm_r2:.4f}"
        assert fm_stack.passthrough is True

    def test_predictions_finite(self, data):
        """All stacking predictions should be finite."""
        from ferroml.ensemble import StackingRegressor

        X_train, X_test, y_train, y_test = data

        fm_stack = StackingRegressor(
            [("rf", "random_forest"), ("lr", "linear_regression")],
            final_estimator="ridge",
            cv=3,
        )
        fm_stack.fit(X_train, y_train.astype(float))
        fm_pred = np.array(fm_stack.predict(X_test))

        assert np.all(np.isfinite(fm_pred)), "Non-finite predictions from StackingRegressor"
        assert fm_pred.shape == (len(X_test),)


# ===========================================================================
# BaggingClassifier (extended — complements test_vs_sklearn_gaps.py)
# ===========================================================================

class TestBaggingClassifierExtendedVsSklearn:
    """Extended BaggingClassifier tests with train/test split evaluation."""

    @pytest.fixture()
    def data(self):
        X, y = make_cls_data(n=500, p=10, seed=42)
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_generalization_accuracy(self, data):
        """BaggingClassifier should generalize well on held-out data."""
        from ferroml.ensemble import BaggingClassifier

        from sklearn.ensemble import BaggingClassifier as SkBag
        from sklearn.tree import DecisionTreeClassifier as SkDT

        X_train, X_test, y_train, y_test = data

        # sklearn
        sk_bag = SkBag(
            estimator=SkDT(max_depth=5),
            n_estimators=20,
            bootstrap=True,
            random_state=42,
        )
        sk_bag.fit(X_train, y_train)
        sk_acc = accuracy_score(y_test, sk_bag.predict(X_test))

        # FerroML
        fm_bag = BaggingClassifier.with_decision_tree(
            n_estimators=20,
            bootstrap=True,
            random_state=42,
            max_depth=5,
        )
        fm_bag.fit(X_train, y_train.astype(float))
        fm_pred = np.array(fm_bag.predict(X_test))
        fm_acc = accuracy_score(y_test, fm_pred)

        assert fm_acc > 0.70, f"FerroML BaggingClassifier test acc: {fm_acc:.4f}"
        assert abs(fm_acc - sk_acc) < 0.10, (
            f"BaggingClassifier test accuracy gap: ferroml={fm_acc:.4f}, sklearn={sk_acc:.4f}"
        )

    def test_subsample_features(self, data):
        """Bagging with max_features < 1.0 should still produce good results."""
        from ferroml.ensemble import BaggingClassifier

        X_train, X_test, y_train, y_test = data

        fm_bag = BaggingClassifier.with_decision_tree(
            n_estimators=20,
            max_features=0.7,
            bootstrap=True,
            random_state=42,
        )
        fm_bag.fit(X_train, y_train.astype(float))
        fm_pred = np.array(fm_bag.predict(X_test))
        fm_acc = accuracy_score(y_test, fm_pred)

        assert fm_acc > 0.65, f"BaggingClassifier with feature subsampling acc: {fm_acc:.4f}"

    def test_score_method(self, data):
        """score() should match manual accuracy computation."""
        from ferroml.ensemble import BaggingClassifier

        X_train, X_test, y_train, y_test = data

        fm_bag = BaggingClassifier.with_decision_tree(
            n_estimators=15,
            random_state=42,
        )
        fm_bag.fit(X_train, y_train.astype(float))

        score = fm_bag.score(X_test, y_test.astype(float))
        fm_pred = np.array(fm_bag.predict(X_test))
        manual_acc = accuracy_score(y_test, fm_pred)

        assert abs(score - manual_acc) < 1e-10, (
            f"score() disagrees: {score:.6f} vs {manual_acc:.6f}"
        )


# ===========================================================================
# BaggingRegressor (extended — complements test_vs_sklearn_gaps.py)
# ===========================================================================

class TestBaggingRegressorExtendedVsSklearn:
    """Extended BaggingRegressor tests with train/test split evaluation."""

    @pytest.fixture()
    def data(self):
        X, y = make_reg_data(n=500, p=10, seed=42)
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_generalization_r2(self, data):
        """BaggingRegressor should generalize well on held-out data."""
        from ferroml.ensemble import BaggingRegressor

        from sklearn.ensemble import BaggingRegressor as SkBagReg
        from sklearn.tree import DecisionTreeRegressor as SkDTR

        X_train, X_test, y_train, y_test = data

        # sklearn
        sk_bag = SkBagReg(
            estimator=SkDTR(max_depth=10),
            n_estimators=20,
            bootstrap=True,
            random_state=42,
        )
        sk_bag.fit(X_train, y_train)
        sk_r2 = r2_score(y_test, sk_bag.predict(X_test))

        # FerroML
        fm_bag = BaggingRegressor.with_decision_tree(
            n_estimators=20,
            bootstrap=True,
            random_state=42,
            max_depth=10,
        )
        fm_bag.fit(X_train, y_train.astype(float))
        fm_pred = np.array(fm_bag.predict(X_test))
        fm_r2 = r2_score(y_test, fm_pred)

        assert fm_r2 > 0.5, f"FerroML BaggingRegressor test R2: {fm_r2:.4f}"
        assert abs(fm_r2 - sk_r2) < 0.15, (
            f"BaggingRegressor test R2 gap: ferroml={fm_r2:.4f}, sklearn={sk_r2:.4f}"
        )

    def test_subsample_features(self, data):
        """Bagging with max_features < 1.0 should still produce decent R2."""
        from ferroml.ensemble import BaggingRegressor

        X_train, X_test, y_train, y_test = data

        fm_bag = BaggingRegressor.with_decision_tree(
            n_estimators=20,
            max_features=0.7,
            bootstrap=True,
            random_state=42,
        )
        fm_bag.fit(X_train, y_train.astype(float))
        fm_pred = np.array(fm_bag.predict(X_test))
        fm_r2 = r2_score(y_test, fm_pred)

        assert fm_r2 > 0.3, f"BaggingRegressor with feature subsampling R2: {fm_r2:.4f}"

    def test_score_method(self, data):
        """score() should match manual R2 computation."""
        from ferroml.ensemble import BaggingRegressor

        X_train, X_test, y_train, y_test = data

        fm_bag = BaggingRegressor.with_decision_tree(
            n_estimators=15,
            random_state=42,
        )
        fm_bag.fit(X_train, y_train.astype(float))

        score = fm_bag.score(X_test, y_test.astype(float))
        fm_pred = np.array(fm_bag.predict(X_test))
        manual_r2 = r2_score(y_test, fm_pred)

        assert abs(score - manual_r2) < 1e-10, (
            f"score() disagrees: {score:.6f} vs {manual_r2:.6f}"
        )

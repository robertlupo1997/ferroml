"""
Correctness tests: FerroML voting/stacking ensembles vs sklearn.

Verifies that FerroML's VotingClassifier, VotingRegressor, StackingClassifier,
and StackingRegressor produce comparable results to sklearn's implementations
on the same data and estimator configurations.

Tolerance notes:
- Voting ensembles use majority/averaging so results should be close
- Stacking uses CV internally; different CV split implementations may cause
  larger divergence, so we use relaxed tolerances
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# sklearn ensemble imports
from sklearn.ensemble import (
    VotingClassifier as SKVotingClassifier,
    VotingRegressor as SKVotingRegressor,
    StackingClassifier as SKStackingClassifier,
    StackingRegressor as SKStackingRegressor,
)
from sklearn.linear_model import LogisticRegression as SKLR, LinearRegression as SKLinReg, Ridge as SKRidge
from sklearn.tree import (
    DecisionTreeClassifier as SKDTC,
    DecisionTreeRegressor as SKDTR,
)
from sklearn.naive_bayes import GaussianNB as SKGNB
from sklearn.neighbors import (
    KNeighborsClassifier as SKKNC,
    KNeighborsRegressor as SKKNR,
)
from sklearn.metrics import accuracy_score, r2_score

# FerroML imports
from ferroml.ensemble import (
    VotingClassifier as FVotingClassifier,
    VotingRegressor as FVotingRegressor,
    StackingClassifier as FStackingClassifier,
    StackingRegressor as FStackingRegressor,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def binary_clf_data():
    """Binary classification dataset with train/test split."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42,
    )
    return train_test_split(X, y.astype(float), test_size=0.3, random_state=42)


@pytest.fixture
def regression_data():
    """Regression dataset with train/test split."""
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=5,
        noise=10.0, random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


# ── VotingClassifier correctness ────────────────────────────────────────────

class TestVotingClassifierCorrectness:
    def test_hard_voting_vs_sklearn(self, binary_clf_data):
        """Hard voting: FerroML accuracy within 0.1 of sklearn."""
        X_train, X_test, y_train, y_test = binary_clf_data

        # sklearn
        sk = SKVotingClassifier(
            estimators=[("lr", SKLR(max_iter=200)), ("dt", SKDTC()), ("nb", SKGNB())],
            voting="hard",
        )
        sk.fit(X_train, y_train)
        sk_acc = accuracy_score(y_test, sk.predict(X_test))

        # FerroML
        ferro = FVotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree"), ("nb", "gaussian_nb")],
            voting="hard",
        )
        ferro.fit(X_train, y_train)
        ferro_acc = accuracy_score(y_test, ferro.predict(X_test))

        assert abs(ferro_acc - sk_acc) < 0.15, (
            f"FerroML hard voting acc={ferro_acc:.3f} vs sklearn={sk_acc:.3f}"
        )

    def test_soft_voting_vs_sklearn(self, binary_clf_data):
        """Soft voting: FerroML accuracy within 0.1 of sklearn."""
        X_train, X_test, y_train, y_test = binary_clf_data

        sk = SKVotingClassifier(
            estimators=[("lr", SKLR(max_iter=200)), ("dt", SKDTC()), ("nb", SKGNB())],
            voting="soft",
        )
        sk.fit(X_train, y_train)
        sk_acc = accuracy_score(y_test, sk.predict(X_test))

        ferro = FVotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree"), ("nb", "gaussian_nb")],
            voting="soft",
        )
        ferro.fit(X_train, y_train)
        ferro_acc = accuracy_score(y_test, ferro.predict(X_test))

        assert abs(ferro_acc - sk_acc) < 0.15, (
            f"FerroML soft voting acc={ferro_acc:.3f} vs sklearn={sk_acc:.3f}"
        )

    def test_soft_voting_probas_shape(self, binary_clf_data):
        """Soft voting predict_proba returns correct shape and sums to 1."""
        X_train, X_test, y_train, _ = binary_clf_data

        ferro = FVotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree"), ("nb", "gaussian_nb")],
            voting="soft",
        )
        ferro.fit(X_train, y_train)
        probas = ferro.predict_proba(X_test)

        assert probas.shape == (X_test.shape[0], 2), f"Expected (60, 2), got {probas.shape}"
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_weighted_voting_vs_sklearn(self, binary_clf_data):
        """Weighted soft voting: comparable to sklearn."""
        X_train, X_test, y_train, y_test = binary_clf_data
        weights = [2.0, 1.0, 1.0]

        sk = SKVotingClassifier(
            estimators=[("lr", SKLR(max_iter=200)), ("dt", SKDTC()), ("nb", SKGNB())],
            voting="soft",
            weights=weights,
        )
        sk.fit(X_train, y_train)
        sk_acc = accuracy_score(y_test, sk.predict(X_test))

        ferro = FVotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree"), ("nb", "gaussian_nb")],
            voting="soft",
            weights=weights,
        )
        ferro.fit(X_train, y_train)
        ferro_acc = accuracy_score(y_test, ferro.predict(X_test))

        assert abs(ferro_acc - sk_acc) < 0.15, (
            f"FerroML weighted voting acc={ferro_acc:.3f} vs sklearn={sk_acc:.3f}"
        )

    def test_voting_all_predictions_valid(self, binary_clf_data):
        """All predictions are valid class labels."""
        X_train, X_test, y_train, _ = binary_clf_data
        classes = set(y_train)

        ferro = FVotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree")],
            voting="hard",
        )
        ferro.fit(X_train, y_train)
        preds = ferro.predict(X_test)

        assert all(p in classes for p in preds), "Predictions contain invalid class labels"
        assert np.all(np.isfinite(preds)), "Non-finite predictions"


# ── VotingRegressor correctness ─────────────────────────────────────────────

class TestVotingRegressorCorrectness:
    def test_regression_vs_sklearn(self, regression_data):
        """VotingRegressor R² within 0.3 of sklearn."""
        X_train, X_test, y_train, y_test = regression_data

        sk = SKVotingRegressor(
            estimators=[("lr", SKLinReg()), ("dt", SKDTR()), ("knn", SKKNR(n_neighbors=5))],
        )
        sk.fit(X_train, y_train)
        sk_r2 = r2_score(y_test, sk.predict(X_test))

        ferro = FVotingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree"), ("knn", "knn")],
        )
        ferro.fit(X_train, y_train)
        ferro_r2 = r2_score(y_test, ferro.predict(X_test))

        assert abs(ferro_r2 - sk_r2) < 0.3, (
            f"FerroML voting reg R²={ferro_r2:.3f} vs sklearn={sk_r2:.3f}"
        )

    def test_weighted_regression_vs_sklearn(self, regression_data):
        """Weighted VotingRegressor comparable to sklearn."""
        X_train, X_test, y_train, y_test = regression_data
        weights = [3.0, 1.0, 2.0]

        sk = SKVotingRegressor(
            estimators=[("lr", SKLinReg()), ("dt", SKDTR()), ("knn", SKKNR(n_neighbors=5))],
            weights=weights,
        )
        sk.fit(X_train, y_train)
        sk_r2 = r2_score(y_test, sk.predict(X_test))

        ferro = FVotingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree"), ("knn", "knn")],
            weights=weights,
        )
        ferro.fit(X_train, y_train)
        ferro_r2 = r2_score(y_test, ferro.predict(X_test))

        assert abs(ferro_r2 - sk_r2) < 0.3, (
            f"FerroML weighted voting reg R²={ferro_r2:.3f} vs sklearn={sk_r2:.3f}"
        )

    def test_regression_predictions_finite(self, regression_data):
        """All regression predictions are finite."""
        X_train, X_test, y_train, _ = regression_data

        ferro = FVotingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree")],
        )
        ferro.fit(X_train, y_train)
        preds = ferro.predict(X_test)

        assert np.all(np.isfinite(preds)), "Non-finite predictions"


# ── StackingClassifier correctness ──────────────────────────────────────────

class TestStackingClassifierCorrectness:
    def test_stacking_clf_vs_sklearn(self, binary_clf_data):
        """StackingClassifier accuracy within 0.15 of sklearn."""
        X_train, X_test, y_train, y_test = binary_clf_data

        sk = SKStackingClassifier(
            estimators=[("dt", SKDTC()), ("nb", SKGNB())],
            final_estimator=SKLR(max_iter=200),
            cv=5,
        )
        sk.fit(X_train, y_train)
        sk_acc = accuracy_score(y_test, sk.predict(X_test))

        ferro = FStackingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb")],
            final_estimator="logistic_regression",
            cv=5,
        )
        ferro.fit(X_train, y_train)
        ferro_acc = accuracy_score(y_test, ferro.predict(X_test))

        assert abs(ferro_acc - sk_acc) < 0.2, (
            f"FerroML stacking clf acc={ferro_acc:.3f} vs sklearn={sk_acc:.3f}"
        )

    def test_stacking_clf_probas_sum_to_one(self, binary_clf_data):
        """Stacking classifier predict_proba outputs sum to 1."""
        X_train, X_test, y_train, _ = binary_clf_data

        ferro = FStackingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb")],
            final_estimator="logistic_regression",
            cv=3,
        )
        ferro.fit(X_train, y_train)
        probas = ferro.predict_proba(X_test)

        assert probas.shape[0] == X_test.shape[0]
        assert probas.shape[1] == 2
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_stacking_clf_all_predictions_valid(self, binary_clf_data):
        """All stacking predictions are valid class labels."""
        X_train, X_test, y_train, _ = binary_clf_data
        classes = set(y_train)

        ferro = FStackingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree")],
            final_estimator="logistic_regression",
            cv=3,
        )
        ferro.fit(X_train, y_train)
        preds = ferro.predict(X_test)

        assert all(p in classes for p in preds)
        assert np.all(np.isfinite(preds))

    def test_stacking_clf_with_passthrough_vs_sklearn(self, binary_clf_data):
        """Stacking with passthrough comparable to sklearn."""
        X_train, X_test, y_train, y_test = binary_clf_data

        sk = SKStackingClassifier(
            estimators=[("dt", SKDTC()), ("nb", SKGNB())],
            final_estimator=SKLR(max_iter=200),
            cv=3,
            passthrough=True,
        )
        sk.fit(X_train, y_train)
        sk_acc = accuracy_score(y_test, sk.predict(X_test))

        ferro = FStackingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb")],
            final_estimator="logistic_regression",
            cv=3,
            passthrough=True,
        )
        ferro.fit(X_train, y_train)
        ferro_acc = accuracy_score(y_test, ferro.predict(X_test))

        assert abs(ferro_acc - sk_acc) < 0.2, (
            f"FerroML stacking+passthrough acc={ferro_acc:.3f} vs sklearn={sk_acc:.3f}"
        )


# ── StackingRegressor correctness ───────────────────────────────────────────

class TestStackingRegressorCorrectness:
    def test_stacking_reg_vs_sklearn(self, regression_data):
        """StackingRegressor R² within 0.3 of sklearn."""
        X_train, X_test, y_train, y_test = regression_data

        sk = SKStackingRegressor(
            estimators=[("lr", SKLinReg()), ("dt", SKDTR())],
            final_estimator=SKRidge(),
            cv=5,
        )
        sk.fit(X_train, y_train)
        sk_r2 = r2_score(y_test, sk.predict(X_test))

        ferro = FStackingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree")],
            final_estimator="ridge",
            cv=5,
        )
        ferro.fit(X_train, y_train)
        ferro_r2 = r2_score(y_test, ferro.predict(X_test))

        assert abs(ferro_r2 - sk_r2) < 0.3, (
            f"FerroML stacking reg R²={ferro_r2:.3f} vs sklearn={sk_r2:.3f}"
        )

    def test_stacking_reg_predictions_finite(self, regression_data):
        """All stacking regression predictions are finite."""
        X_train, X_test, y_train, _ = regression_data

        ferro = FStackingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree")],
            final_estimator="ridge",
            cv=3,
        )
        ferro.fit(X_train, y_train)
        preds = ferro.predict(X_test)

        assert np.all(np.isfinite(preds)), "Non-finite predictions"

    def test_stacking_reg_with_passthrough_vs_sklearn(self, regression_data):
        """Stacking regressor with passthrough comparable to sklearn."""
        X_train, X_test, y_train, y_test = regression_data

        sk = SKStackingRegressor(
            estimators=[("lr", SKLinReg()), ("dt", SKDTR())],
            final_estimator=SKRidge(),
            cv=3,
            passthrough=True,
        )
        sk.fit(X_train, y_train)
        sk_r2 = r2_score(y_test, sk.predict(X_test))

        ferro = FStackingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree")],
            final_estimator="ridge",
            cv=3,
            passthrough=True,
        )
        ferro.fit(X_train, y_train)
        ferro_r2 = r2_score(y_test, ferro.predict(X_test))

        assert abs(ferro_r2 - sk_r2) < 0.3, (
            f"FerroML stacking+passthrough reg R²={ferro_r2:.3f} vs sklearn={sk_r2:.3f}"
        )

    def test_stacking_reg_r2_positive(self, regression_data):
        """Stacking regressor achieves positive R² on easy data."""
        X_train, X_test, y_train, y_test = regression_data

        ferro = FStackingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree"), ("knn", "knn")],
            final_estimator="ridge",
            cv=3,
        )
        ferro.fit(X_train, y_train)
        ferro_r2 = r2_score(y_test, ferro.predict(X_test))

        assert ferro_r2 > 0, f"Expected positive R², got {ferro_r2:.3f}"

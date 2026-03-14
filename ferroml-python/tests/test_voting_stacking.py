"""Test FerroML Voting and Stacking ensemble models."""

import numpy as np
import pytest
from ferroml.ensemble import (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 4)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
    return X[:150], y[:150], X[150:], y[150:]


@pytest.fixture
def multiclass_data():
    rng = np.random.RandomState(42)
    X = rng.randn(300, 4)
    y = np.where(X[:, 0] > 0.5, 2.0, np.where(X[:, 0] < -0.5, 0.0, 1.0))
    return X[:220], y[:220], X[220:], y[220:]


@pytest.fixture
def regression_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 4)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.5 * rng.randn(200)
    return X[:150], y[:150], X[150:], y[150:]


# ===========================================================================
# VotingClassifier tests
# ===========================================================================

class TestVotingClassifier:
    def test_hard_voting_basic(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        vc = VotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree")],
            voting="hard",
        )
        vc.fit(X_tr, y_tr)
        preds = vc.predict(X_te)
        assert preds.shape == (50,)
        assert set(np.unique(preds)).issubset({0.0, 1.0})

    def test_soft_voting_basic(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        vc = VotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree")],
            voting="soft",
        )
        vc.fit(X_tr, y_tr)
        preds = vc.predict(X_te)
        assert preds.shape == (50,)
        assert set(np.unique(preds)).issubset({0.0, 1.0})

    def test_soft_voting_predict_proba(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        vc = VotingClassifier(
            [("lr", "logistic_regression"), ("nb", "gaussian_nb")],
            voting="soft",
        )
        vc.fit(X_tr, y_tr)
        probas = vc.predict_proba(X_te)
        assert probas.shape == (50, 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(probas >= 0) and np.all(probas <= 1)

    def test_weighted_voting(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        vc1 = VotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree")],
            voting="soft",
            weights=[1.0, 1.0],
        )
        vc2 = VotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree")],
            voting="soft",
            weights=[10.0, 0.1],
        )
        vc1.fit(X_tr, y_tr)
        vc2.fit(X_tr, y_tr)
        # Different weights should generally produce different probabilities
        p1 = vc1.predict_proba(X_te)
        p2 = vc2.predict_proba(X_te)
        assert not np.allclose(p1, p2, atol=1e-4)

    def test_voting_binary_accuracy(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        vc = VotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree"), ("nb", "gaussian_nb")],
            voting="soft",
        )
        vc.fit(X_tr, y_tr)
        preds = vc.predict(X_te)
        acc = np.mean(preds == y_te)
        assert acc > 0.7, f"Accuracy too low: {acc}"

    def test_voting_multiclass(self, multiclass_data):
        X_tr, y_tr, X_te, y_te = multiclass_data
        vc = VotingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb"), ("knn", "knn")],
            voting="soft",
        )
        vc.fit(X_tr, y_tr)
        preds = vc.predict(X_te)
        assert set(np.unique(preds)).issubset({0.0, 1.0, 2.0})

    def test_voting_multiple_estimators(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        vc = VotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree"),
             ("nb", "gaussian_nb"), ("knn", "knn")],
            voting="soft",
        )
        vc.fit(X_tr, y_tr)
        preds = vc.predict(X_te)
        assert preds.shape == (50,)

    def test_voting_estimator_names(self, binary_data):
        X_tr, y_tr, _, _ = binary_data
        vc = VotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree")],
        )
        assert vc.estimator_names == ["lr", "dt"]

    def test_voting_unfitted_raises(self, binary_data):
        _, _, X_te, _ = binary_data
        vc = VotingClassifier([("lr", "logistic_regression")])
        with pytest.raises(RuntimeError):
            vc.predict(X_te)

    def test_voting_invalid_estimator_raises(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            VotingClassifier([("bad", "nonexistent_model")])

    def test_voting_repr(self):
        vc = VotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree")],
            voting="soft",
        )
        r = repr(vc)
        assert "VotingClassifier" in r
        assert "soft" in r

    def test_voting_single_estimator(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        vc = VotingClassifier([("lr", "logistic_regression")])
        vc.fit(X_tr, y_tr)
        preds = vc.predict(X_te)
        assert preds.shape == (50,)

    def test_voting_hard_proba_raises(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        vc = VotingClassifier(
            [("lr", "logistic_regression")], voting="hard"
        )
        vc.fit(X_tr, y_tr)
        with pytest.raises(ValueError, match="soft voting"):
            vc.predict_proba(X_te)

    def test_voting_invalid_voting_raises(self):
        with pytest.raises(ValueError, match="Unknown voting"):
            VotingClassifier([("lr", "logistic_regression")], voting="invalid")

    def test_voting_hard_vs_soft_can_differ(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        vc_hard = VotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree"), ("nb", "gaussian_nb")],
            voting="hard",
        )
        vc_soft = VotingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree"), ("nb", "gaussian_nb")],
            voting="soft",
        )
        vc_hard.fit(X_tr, y_tr)
        vc_soft.fit(X_tr, y_tr)
        p_hard = vc_hard.predict(X_te)
        p_soft = vc_soft.predict(X_te)
        # They may or may not differ, but both should be valid
        assert p_hard.shape == p_soft.shape


# ===========================================================================
# VotingRegressor tests
# ===========================================================================

class TestVotingRegressor:
    def test_regression_basic(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        vr = VotingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree")],
        )
        vr.fit(X_tr, y_tr)
        preds = vr.predict(X_te)
        assert preds.shape == (50,)
        assert np.all(np.isfinite(preds))

    def test_regression_weighted(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        vr = VotingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree")],
            weights=[2.0, 1.0],
        )
        vr.fit(X_tr, y_tr)
        preds = vr.predict(X_te)
        assert np.all(np.isfinite(preds))

    def test_regression_r2_positive(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        vr = VotingRegressor(
            [("lr", "linear_regression"), ("ridge", "ridge"), ("dt", "decision_tree")],
        )
        vr.fit(X_tr, y_tr)
        preds = vr.predict(X_te)
        ss_res = np.sum((y_te - preds) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0, f"R² too low: {r2}"

    def test_regression_multiple_estimators(self, regression_data):
        X_tr, y_tr, X_te, _ = regression_data
        vr = VotingRegressor(
            [("lr", "linear_regression"), ("ridge", "ridge"),
             ("dt", "decision_tree"), ("knn", "knn")],
        )
        vr.fit(X_tr, y_tr)
        preds = vr.predict(X_te)
        assert preds.shape == (50,)

    def test_regression_estimator_names(self):
        vr = VotingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree")],
        )
        assert vr.estimator_names == ["lr", "dt"]

    def test_regression_unfitted_raises(self, regression_data):
        _, _, X_te, _ = regression_data
        vr = VotingRegressor([("lr", "linear_regression")])
        with pytest.raises(RuntimeError):
            vr.predict(X_te)

    def test_regression_invalid_estimator_raises(self):
        with pytest.raises(ValueError, match="Unknown regressor"):
            VotingRegressor([("bad", "nonexistent_model")])

    def test_regression_repr(self):
        vr = VotingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree")],
        )
        assert "VotingRegressor" in repr(vr)

    def test_regression_single_estimator(self, regression_data):
        X_tr, y_tr, X_te, _ = regression_data
        vr = VotingRegressor([("lr", "linear_regression")])
        vr.fit(X_tr, y_tr)
        preds = vr.predict(X_te)
        assert preds.shape == (50,)

    def test_regression_all_estimator_types(self, regression_data):
        """Verify all 10 regressor estimator types work."""
        X_tr, y_tr, X_te, _ = regression_data
        types = [
            "linear_regression", "ridge", "lasso", "elastic_net",
            "decision_tree", "random_forest", "knn", "svr",
            "gradient_boosting", "hist_gradient_boosting",
        ]
        for i, t in enumerate(types):
            vr = VotingRegressor([(f"e{i}", t)])
            vr.fit(X_tr, y_tr)
            preds = vr.predict(X_te)
            assert np.all(np.isfinite(preds)), f"Non-finite predictions for {t}"


# ===========================================================================
# StackingClassifier tests
# ===========================================================================

class TestStackingClassifier:
    def test_stacking_clf_basic(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        sc = StackingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb")],
        )
        sc.fit(X_tr, y_tr)
        preds = sc.predict(X_te)
        assert preds.shape == (50,)
        assert set(np.unique(preds)).issubset({0.0, 1.0})

    def test_stacking_clf_predict_proba(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        sc = StackingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb")],
        )
        sc.fit(X_tr, y_tr)
        probas = sc.predict_proba(X_te)
        assert probas.shape[0] == 50
        assert probas.shape[1] >= 2

    def test_stacking_clf_with_passthrough(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        sc = StackingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb")],
            passthrough=True,
        )
        sc.fit(X_tr, y_tr)
        preds = sc.predict(X_te)
        assert preds.shape == (50,)
        assert sc.passthrough is True

    def test_stacking_clf_predict_method(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        sc = StackingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb")],
            stack_method="predict",
        )
        sc.fit(X_tr, y_tr)
        preds = sc.predict(X_te)
        assert preds.shape == (50,)
        assert sc.stack_method_name == "predict"

    def test_stacking_clf_custom_final(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        sc = StackingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb")],
            final_estimator="decision_tree",
        )
        sc.fit(X_tr, y_tr)
        preds = sc.predict(X_te)
        assert preds.shape == (50,)

    def test_stacking_clf_cv_folds(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        sc3 = StackingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb")],
            cv=3,
        )
        sc3.fit(X_tr, y_tr)
        preds = sc3.predict(X_te)
        assert preds.shape == (50,)

    def test_stacking_clf_binary_accuracy(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        sc = StackingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree"), ("nb", "gaussian_nb")],
        )
        sc.fit(X_tr, y_tr)
        preds = sc.predict(X_te)
        acc = np.mean(preds == y_te)
        assert acc > 0.6, f"Accuracy too low: {acc}"

    def test_stacking_clf_multiclass(self, multiclass_data):
        X_tr, y_tr, X_te, _ = multiclass_data
        sc = StackingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb")],
            final_estimator="decision_tree",
        )
        sc.fit(X_tr, y_tr)
        preds = sc.predict(X_te)
        assert set(np.unique(preds)).issubset({0.0, 1.0, 2.0})

    def test_stacking_clf_multiple_estimators(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        sc = StackingClassifier(
            [("lr", "logistic_regression"), ("dt", "decision_tree"),
             ("nb", "gaussian_nb"), ("knn", "knn")],
        )
        sc.fit(X_tr, y_tr)
        preds = sc.predict(X_te)
        assert preds.shape == (50,)

    def test_stacking_clf_estimator_names(self):
        sc = StackingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb")],
        )
        assert sc.estimator_names == ["dt", "nb"]

    def test_stacking_clf_unfitted_raises(self, binary_data):
        _, _, X_te, _ = binary_data
        sc = StackingClassifier([("dt", "decision_tree")])
        with pytest.raises(RuntimeError):
            sc.predict(X_te)

    def test_stacking_clf_invalid_estimator_raises(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            StackingClassifier([("bad", "nonexistent_model")])

    def test_stacking_clf_invalid_final_raises(self):
        with pytest.raises(ValueError, match="Unknown classifier final"):
            StackingClassifier(
                [("dt", "decision_tree")],
                final_estimator="nonexistent",
            )

    def test_stacking_clf_invalid_stack_method_raises(self):
        with pytest.raises(ValueError, match="Unknown stack_method"):
            StackingClassifier(
                [("dt", "decision_tree")],
                stack_method="invalid",
            )

    def test_stacking_clf_repr(self):
        sc = StackingClassifier(
            [("dt", "decision_tree"), ("nb", "gaussian_nb")],
            final_estimator="logistic_regression",
        )
        r = repr(sc)
        assert "StackingClassifier" in r
        assert "logistic_regression" in r


# ===========================================================================
# StackingRegressor tests
# ===========================================================================

class TestStackingRegressor:
    def test_stacking_reg_basic(self, regression_data):
        X_tr, y_tr, X_te, _ = regression_data
        sr = StackingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree")],
        )
        sr.fit(X_tr, y_tr)
        preds = sr.predict(X_te)
        assert preds.shape == (50,)
        assert np.all(np.isfinite(preds))

    def test_stacking_reg_with_passthrough(self, regression_data):
        X_tr, y_tr, X_te, _ = regression_data
        sr = StackingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree")],
            passthrough=True,
        )
        sr.fit(X_tr, y_tr)
        preds = sr.predict(X_te)
        assert preds.shape == (50,)
        assert sr.passthrough is True

    def test_stacking_reg_custom_final(self, regression_data):
        X_tr, y_tr, X_te, _ = regression_data
        sr = StackingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree")],
            final_estimator="ridge",
        )
        sr.fit(X_tr, y_tr)
        preds = sr.predict(X_te)
        assert preds.shape == (50,)

    def test_stacking_reg_cv_folds(self, regression_data):
        X_tr, y_tr, X_te, _ = regression_data
        sr = StackingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree")],
            cv=3,
        )
        sr.fit(X_tr, y_tr)
        preds = sr.predict(X_te)
        assert preds.shape == (50,)

    def test_stacking_reg_r2_positive(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        sr = StackingRegressor(
            [("lr", "linear_regression"), ("ridge", "ridge"), ("dt", "decision_tree")],
        )
        sr.fit(X_tr, y_tr)
        preds = sr.predict(X_te)
        ss_res = np.sum((y_te - preds) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0, f"R² too low: {r2}"

    def test_stacking_reg_multiple_estimators(self, regression_data):
        X_tr, y_tr, X_te, _ = regression_data
        sr = StackingRegressor(
            [("lr", "linear_regression"), ("ridge", "ridge"),
             ("dt", "decision_tree"), ("knn", "knn")],
        )
        sr.fit(X_tr, y_tr)
        preds = sr.predict(X_te)
        assert preds.shape == (50,)

    def test_stacking_reg_estimator_names(self):
        sr = StackingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree")],
        )
        assert sr.estimator_names == ["lr", "dt"]

    def test_stacking_reg_unfitted_raises(self, regression_data):
        _, _, X_te, _ = regression_data
        sr = StackingRegressor([("lr", "linear_regression")])
        with pytest.raises(RuntimeError):
            sr.predict(X_te)

    def test_stacking_reg_invalid_estimator_raises(self):
        with pytest.raises(ValueError, match="Unknown regressor"):
            StackingRegressor([("bad", "nonexistent_model")])

    def test_stacking_reg_invalid_final_raises(self):
        with pytest.raises(ValueError, match="Unknown regressor final"):
            StackingRegressor(
                [("dt", "decision_tree")],
                final_estimator="nonexistent",
            )

    def test_stacking_reg_repr(self):
        sr = StackingRegressor(
            [("lr", "linear_regression"), ("dt", "decision_tree")],
            final_estimator="ridge",
        )
        r = repr(sr)
        assert "StackingRegressor" in r
        assert "ridge" in r

    def test_stacking_reg_predictions_finite(self, regression_data):
        X_tr, y_tr, X_te, _ = regression_data
        sr = StackingRegressor(
            [("lr", "linear_regression"), ("ridge", "ridge")],
        )
        sr.fit(X_tr, y_tr)
        preds = sr.predict(X_te)
        assert np.all(np.isfinite(preds))


# ===========================================================================
# Cross-model tests
# ===========================================================================

class TestAllEstimatorTypes:
    def test_all_classifier_types(self, binary_data):
        """Verify all 10 classifier estimator types work in VotingClassifier."""
        X_tr, y_tr, X_te, _ = binary_data
        types = [
            "logistic_regression", "decision_tree", "random_forest",
            "gaussian_nb", "multinomial_nb", "bernoulli_nb", "knn",
            "svc", "gradient_boosting", "hist_gradient_boosting",
        ]
        # MultinomialNB needs non-negative features
        X_tr_pos = np.abs(X_tr)
        X_te_pos = np.abs(X_te)
        for i, t in enumerate(types):
            vc = VotingClassifier([(f"e{i}", t)], voting="hard")
            if t == "multinomial_nb":
                vc.fit(X_tr_pos, y_tr)
                preds = vc.predict(X_te_pos)
            else:
                vc.fit(X_tr, y_tr)
                preds = vc.predict(X_te)
            assert preds.shape == (50,), f"Failed for estimator type: {t}"

    def test_empty_estimators_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            VotingClassifier([])

    def test_empty_reg_estimators_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            VotingRegressor([])

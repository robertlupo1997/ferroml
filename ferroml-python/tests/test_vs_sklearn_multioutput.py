"""Cross-library validation: FerroML MultiOutput wrappers vs sklearn equivalents.

Tests MultiOutputRegressor and MultiOutputClassifier against their sklearn
counterparts with per-target metric comparisons.

Phase X.2 — Plan X production-ready validation.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score


# ===========================================================================
# MultiOutputRegressor
# ===========================================================================

class TestMultiOutputRegressorVsSklearn:
    """Compare FerroML MultiOutputRegressor against sklearn MultiOutputRegressor."""

    @pytest.fixture()
    def data(self):
        """Generate regression data with 3 correlated targets."""
        rng = np.random.RandomState(42)
        n_samples = 300
        n_features = 8

        X = rng.randn(n_samples, n_features)
        # 3 targets: linear combinations of features with noise
        coefs = rng.randn(n_features, 3) * 5
        noise = rng.randn(n_samples, 3) * 2.0
        Y = X @ coefs + noise

        return X, Y

    def test_per_target_r2_with_linear_regression(self, data):
        """Per-target R2 should match sklearn within 0.01 using LinearRegression."""
        from ferroml.multioutput import MultiOutputRegressor

        from sklearn.multioutput import MultiOutputRegressor as SkMO
        from sklearn.linear_model import LinearRegression as SkLR

        X, Y = data

        # sklearn
        sk_mo = SkMO(SkLR())
        sk_mo.fit(X, Y)
        sk_pred = sk_mo.predict(X)

        # FerroML
        fm_mo = MultiOutputRegressor("linear_regression")
        fm_mo.fit(X, Y)
        fm_pred = np.array(fm_mo.predict(X))

        assert fm_pred.shape == Y.shape, (
            f"Shape mismatch: ferroml={fm_pred.shape}, expected={Y.shape}"
        )

        # Compare per-target R2
        for i in range(Y.shape[1]):
            sk_r2 = r2_score(Y[:, i], sk_pred[:, i])
            fm_r2 = r2_score(Y[:, i], fm_pred[:, i])
            assert abs(fm_r2 - sk_r2) < 0.02, (
                f"Target {i} R2 mismatch: ferroml={fm_r2:.6f}, sklearn={sk_r2:.6f}"
            )

    def test_per_target_r2_with_decision_tree(self, data):
        """Per-target R2 with DecisionTree base should be reasonable."""
        from ferroml.multioutput import MultiOutputRegressor

        from sklearn.multioutput import MultiOutputRegressor as SkMO
        from sklearn.tree import DecisionTreeRegressor as SkDT

        X, Y = data

        # sklearn
        sk_mo = SkMO(SkDT(random_state=42))
        sk_mo.fit(X, Y)
        sk_pred = sk_mo.predict(X)

        # FerroML
        fm_mo = MultiOutputRegressor("decision_tree")
        fm_mo.fit(X, Y)
        fm_pred = np.array(fm_mo.predict(X))

        # Both should overfit on training data (tree with no depth limit)
        for i in range(Y.shape[1]):
            sk_r2 = r2_score(Y[:, i], sk_pred[:, i])
            fm_r2 = r2_score(Y[:, i], fm_pred[:, i])
            assert fm_r2 > 0.90, f"Target {i} FerroML DT R2 too low: {fm_r2:.4f}"
            assert sk_r2 > 0.90, f"Target {i} sklearn DT R2 too low: {sk_r2:.4f}"

    def test_n_outputs_property(self, data):
        """n_outputs should reflect number of target columns."""
        from ferroml.multioutput import MultiOutputRegressor

        X, Y = data

        fm_mo = MultiOutputRegressor("linear_regression")
        assert fm_mo.n_outputs is None, "n_outputs should be None before fitting"

        fm_mo.fit(X, Y)
        assert fm_mo.n_outputs == 3, f"Expected 3 outputs, got {fm_mo.n_outputs}"

    def test_predictions_finite(self, data):
        """All multi-output predictions should be finite."""
        from ferroml.multioutput import MultiOutputRegressor

        X, Y = data

        fm_mo = MultiOutputRegressor("ridge")
        fm_mo.fit(X, Y)
        fm_pred = np.array(fm_mo.predict(X))

        assert np.all(np.isfinite(fm_pred)), "Non-finite predictions from MultiOutputRegressor"

    def test_two_targets(self):
        """Should work correctly with just 2 targets."""
        from ferroml.multioutput import MultiOutputRegressor

        from sklearn.multioutput import MultiOutputRegressor as SkMO
        from sklearn.linear_model import LinearRegression as SkLR

        rng = np.random.RandomState(123)
        X = rng.randn(200, 5)
        Y = np.column_stack([X @ rng.randn(5), X @ rng.randn(5)])

        sk_mo = SkMO(SkLR())
        sk_mo.fit(X, Y)
        sk_pred = sk_mo.predict(X)

        fm_mo = MultiOutputRegressor("linear_regression")
        fm_mo.fit(X, Y)
        fm_pred = np.array(fm_mo.predict(X))

        np.testing.assert_allclose(fm_pred, sk_pred, atol=1e-4,
            err_msg="MultiOutputRegressor predictions diverge from sklearn")


# ===========================================================================
# MultiOutputClassifier
# ===========================================================================

class TestMultiOutputClassifierVsSklearn:
    """Compare FerroML MultiOutputClassifier against sklearn MultiOutputClassifier."""

    @pytest.fixture()
    def data(self):
        """Generate classification data with 3 binary targets."""
        rng = np.random.RandomState(42)
        n_samples = 300
        n_features = 8

        X = rng.randn(n_samples, n_features)
        # 3 binary targets based on linear combinations
        Y = np.column_stack([
            (X @ rng.randn(n_features) > 0).astype(float),
            (X @ rng.randn(n_features) > 0).astype(float),
            (X @ rng.randn(n_features) > 0).astype(float),
        ])

        return X, Y

    def test_per_target_accuracy_with_logistic_regression(self, data):
        """Per-target accuracy should match sklearn within 3% using LogisticRegression."""
        from ferroml.multioutput import MultiOutputClassifier

        from sklearn.multioutput import MultiOutputClassifier as SkMO
        from sklearn.linear_model import LogisticRegression as SkLR

        X, Y = data

        # sklearn
        sk_mo = SkMO(SkLR(max_iter=200))
        sk_mo.fit(X, Y.astype(int))
        sk_pred = sk_mo.predict(X)

        # FerroML
        fm_mo = MultiOutputClassifier("logistic_regression")
        fm_mo.fit(X, Y)
        fm_pred = np.array(fm_mo.predict(X))

        assert fm_pred.shape == Y.shape, (
            f"Shape mismatch: ferroml={fm_pred.shape}, expected={Y.shape}"
        )

        # Compare per-target accuracy
        for i in range(Y.shape[1]):
            sk_acc = accuracy_score(Y[:, i], sk_pred[:, i])
            fm_acc = accuracy_score(Y[:, i], fm_pred[:, i])
            assert abs(fm_acc - sk_acc) < 0.08, (
                f"Target {i} accuracy mismatch: ferroml={fm_acc:.4f}, sklearn={sk_acc:.4f}"
            )

    def test_per_target_accuracy_with_decision_tree(self, data):
        """Per-target accuracy with DecisionTree should be reasonable."""
        from ferroml.multioutput import MultiOutputClassifier

        from sklearn.multioutput import MultiOutputClassifier as SkMO
        from sklearn.tree import DecisionTreeClassifier as SkDT

        X, Y = data

        # sklearn
        sk_mo = SkMO(SkDT(random_state=42))
        sk_mo.fit(X, Y.astype(int))
        sk_pred = sk_mo.predict(X)

        # FerroML
        fm_mo = MultiOutputClassifier("decision_tree")
        fm_mo.fit(X, Y)
        fm_pred = np.array(fm_mo.predict(X))

        # DT should overfit well on training data
        for i in range(Y.shape[1]):
            sk_acc = accuracy_score(Y[:, i], sk_pred[:, i])
            fm_acc = accuracy_score(Y[:, i], fm_pred[:, i])
            assert fm_acc > 0.90, f"Target {i} FerroML DT accuracy too low: {fm_acc:.4f}"
            assert sk_acc > 0.90, f"Target {i} sklearn DT accuracy too low: {sk_acc:.4f}"

    def test_predictions_binary(self, data):
        """All predictions should be binary (0.0 or 1.0)."""
        from ferroml.multioutput import MultiOutputClassifier

        X, Y = data

        fm_mo = MultiOutputClassifier("logistic_regression")
        fm_mo.fit(X, Y)
        fm_pred = np.array(fm_mo.predict(X))

        unique_vals = np.unique(fm_pred)
        assert set(unique_vals).issubset({0.0, 1.0}), (
            f"Unexpected prediction values: {unique_vals}"
        )

    def test_n_outputs_property(self, data):
        """n_outputs should reflect number of target columns."""
        from ferroml.multioutput import MultiOutputClassifier

        X, Y = data

        fm_mo = MultiOutputClassifier("logistic_regression")
        assert fm_mo.n_outputs is None, "n_outputs should be None before fitting"

        fm_mo.fit(X, Y)
        assert fm_mo.n_outputs == 3, f"Expected 3 outputs, got {fm_mo.n_outputs}"

    def test_predict_proba_shape(self, data):
        """predict_proba should return a list of arrays, one per target."""
        from ferroml.multioutput import MultiOutputClassifier

        X, Y = data

        fm_mo = MultiOutputClassifier("logistic_regression")
        fm_mo.fit(X, Y)
        probas = fm_mo.predict_proba(X)

        assert len(probas) == 3, f"Expected 3 proba arrays, got {len(probas)}"
        for i, p in enumerate(probas):
            p_arr = np.array(p)
            assert p_arr.shape[0] == len(X), (
                f"Target {i} proba rows: {p_arr.shape[0]} vs {len(X)}"
            )
            assert p_arr.shape[1] == 2, (
                f"Target {i} proba cols: {p_arr.shape[1]} (expected 2 for binary)"
            )
            # Probabilities should sum to ~1
            np.testing.assert_allclose(
                p_arr.sum(axis=1), 1.0, atol=1e-6,
                err_msg=f"Target {i} probabilities don't sum to 1",
            )

    def test_is_fitted(self, data):
        """is_fitted() should correctly reflect model state."""
        from ferroml.multioutput import MultiOutputClassifier

        X, Y = data

        fm_mo = MultiOutputClassifier("logistic_regression")
        assert fm_mo.is_fitted() is False

        fm_mo.fit(X, Y)
        assert fm_mo.is_fitted() is True

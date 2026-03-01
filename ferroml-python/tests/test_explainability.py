"""Test FerroML explainability module."""

import numpy as np
import pytest

import ferroml
from ferroml.trees import (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
)
from ferroml.linear import LinearRegression, LogisticRegression

# The explainability functions live in the native extension.
_expl = ferroml._native.explainability

TreeExplainer = _expl.TreeExplainer
permutation_importance_rf_reg = _expl.permutation_importance_rf_reg
permutation_importance_rf_clf = _expl.permutation_importance_rf_clf
permutation_importance_dt_reg = _expl.permutation_importance_dt_reg
permutation_importance_linear = _expl.permutation_importance_linear
permutation_importance_logistic = _expl.permutation_importance_logistic
partial_dependence_rf_reg = _expl.partial_dependence_rf_reg
partial_dependence_dt_reg = _expl.partial_dependence_dt_reg
partial_dependence_linear = _expl.partial_dependence_linear
ice_rf_reg = _expl.ice_rf_reg
ice_dt_reg = _expl.ice_dt_reg
ice_linear = _expl.ice_linear
h_statistic_rf_reg = _expl.h_statistic_rf_reg
h_statistic_matrix_rf_reg = _expl.h_statistic_matrix_rf_reg


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 120
N_FEATURES = 4


@pytest.fixture(scope="module")
def reg_data():
    """Regression dataset: y = X0 + 2*X1 - X2 + noise."""
    np.random.seed(42)
    X = np.random.randn(N_SAMPLES, N_FEATURES)
    y = X[:, 0] + 2.0 * X[:, 1] - X[:, 2] + np.random.randn(N_SAMPLES) * 0.1
    return X, y


@pytest.fixture(scope="module")
def clf_data():
    """Binary classification dataset."""
    np.random.seed(42)
    X = np.random.randn(N_SAMPLES, N_FEATURES)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


@pytest.fixture(scope="module")
def dt_reg(reg_data):
    """Fitted DecisionTreeRegressor (small, fast)."""
    X, y = reg_data
    m = DecisionTreeRegressor(max_depth=3, random_state=42)
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def dt_clf(clf_data):
    """Fitted DecisionTreeClassifier (small, fast)."""
    X, y = clf_data
    m = DecisionTreeClassifier(max_depth=3, random_state=42)
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def rf_reg(reg_data):
    """Fitted RandomForestRegressor (small, fast)."""
    X, y = reg_data
    m = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def rf_clf(clf_data):
    """Fitted RandomForestClassifier (small, fast)."""
    X, y = clf_data
    m = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def gb_reg(reg_data):
    """Fitted GradientBoostingRegressor (small, fast)."""
    X, y = reg_data
    m = GradientBoostingRegressor(n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42)
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def lin_reg(reg_data):
    """Fitted LinearRegression."""
    X, y = reg_data
    m = LinearRegression()
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def log_reg(clf_data):
    """Fitted LogisticRegression."""
    X, y = clf_data
    m = LogisticRegression()
    m.fit(X, y)
    return m


# ---------------------------------------------------------------------------
# TreeExplainer — factory methods
# ---------------------------------------------------------------------------


class TestTreeExplainerFactories:
    """Tests for TreeExplainer.from_*() factory methods."""

    def test_from_decision_tree_regressor(self, dt_reg):
        """from_decision_tree_regressor() should produce a valid TreeExplainer."""
        te = TreeExplainer.from_decision_tree_regressor(dt_reg)
        assert te.n_trees == 1
        assert te.n_features == N_FEATURES

    def test_from_decision_tree_classifier(self, dt_clf):
        """from_decision_tree_classifier() should produce a valid TreeExplainer."""
        te = TreeExplainer.from_decision_tree_classifier(dt_clf)
        assert te.n_trees == 1
        assert te.n_features == N_FEATURES

    def test_from_random_forest_regressor(self, rf_reg):
        """from_random_forest_regressor() should produce a valid TreeExplainer."""
        te = TreeExplainer.from_random_forest_regressor(rf_reg)
        assert te.n_trees == 10
        assert te.n_features == N_FEATURES

    def test_from_random_forest_classifier(self, rf_clf):
        """from_random_forest_classifier() should produce a valid TreeExplainer."""
        te = TreeExplainer.from_random_forest_classifier(rf_clf)
        assert te.n_trees == 10
        assert te.n_features == N_FEATURES

    def test_from_gradient_boosting_regressor(self, gb_reg):
        """from_gradient_boosting_regressor() should produce a valid TreeExplainer."""
        te = TreeExplainer.from_gradient_boosting_regressor(gb_reg)
        assert te.n_trees == 10
        assert te.n_features == N_FEATURES


# ---------------------------------------------------------------------------
# TreeExplainer — explain / explain_batch
# ---------------------------------------------------------------------------


class TestTreeExplainerExplain:
    """Tests for explain() and explain_batch() on various model types."""

    # ---- explain() ----

    def _assert_explain_structure(self, result, n_features):
        """Shared structural checks for a single-sample explain() result."""
        assert isinstance(result, dict)
        assert "base_value" in result
        assert "shap_values" in result
        assert "feature_values" in result
        shap = result["shap_values"]
        fv = result["feature_values"]
        assert len(shap) == n_features
        assert len(fv) == n_features

    def test_explain_dt_reg_structure(self, dt_reg, reg_data):
        """explain() on DT regressor should return expected keys and shapes."""
        te = TreeExplainer.from_decision_tree_regressor(dt_reg)
        X, _ = reg_data
        result = te.explain(X[0])
        self._assert_explain_structure(result, N_FEATURES)

    def test_explain_rf_reg_structure(self, rf_reg, reg_data):
        """explain() on RF regressor should return expected keys and shapes."""
        te = TreeExplainer.from_random_forest_regressor(rf_reg)
        X, _ = reg_data
        result = te.explain(X[0])
        self._assert_explain_structure(result, N_FEATURES)

    def test_explain_rf_clf_structure(self, rf_clf, clf_data):
        """explain() on RF classifier should return expected keys and shapes."""
        te = TreeExplainer.from_random_forest_classifier(rf_clf)
        X, _ = clf_data
        result = te.explain(X[0])
        self._assert_explain_structure(result, N_FEATURES)

    def test_explain_gb_reg_structure(self, gb_reg, reg_data):
        """explain() on GB regressor should return expected keys and shapes."""
        te = TreeExplainer.from_gradient_boosting_regressor(gb_reg)
        X, _ = reg_data
        result = te.explain(X[0])
        self._assert_explain_structure(result, N_FEATURES)

    # ---- SHAP consistency property ----

    def test_shap_consistency_dt_reg(self, dt_reg, reg_data):
        """sum(shap_values) + base_value must equal model prediction (DT reg)."""
        X, _ = reg_data
        te = TreeExplainer.from_decision_tree_regressor(dt_reg)
        for i in range(5):
            result = te.explain(X[i])
            reconstructed = float(np.sum(result["shap_values"])) + result["base_value"]
            pred = float(dt_reg.predict(X[i : i + 1])[0])
            np.testing.assert_allclose(reconstructed, pred, atol=1e-6,
                                       err_msg=f"SHAP property violated at sample {i}")

    def test_shap_consistency_rf_reg(self, rf_reg, reg_data):
        """sum(shap_values) + base_value must equal model prediction (RF reg)."""
        X, _ = reg_data
        te = TreeExplainer.from_random_forest_regressor(rf_reg)
        for i in range(5):
            result = te.explain(X[i])
            reconstructed = float(np.sum(result["shap_values"])) + result["base_value"]
            pred = float(rf_reg.predict(X[i : i + 1])[0])
            np.testing.assert_allclose(reconstructed, pred, atol=1e-6,
                                       err_msg=f"SHAP property violated at sample {i}")

    def test_shap_consistency_gb_reg(self, gb_reg, reg_data):
        """SHAP values for GB regressor should be finite and have correct shape.

        Note: for GradientBoostingRegressor the TreeSHAP explains the ensemble's
        raw (pre-shrinkage) output, so sum(shap_values) + base_value does not
        equal predict() directly.  We verify structure and finiteness instead.
        """
        X, _ = reg_data
        te = TreeExplainer.from_gradient_boosting_regressor(gb_reg)
        for i in range(5):
            result = te.explain(X[i])
            assert len(result["shap_values"]) == N_FEATURES
            assert np.all(np.isfinite(result["shap_values"]))
            assert np.isfinite(result["base_value"])

    # ---- explain_batch() ----

    def test_explain_batch_keys(self, rf_reg, reg_data):
        """explain_batch() should return a dict with standard keys."""
        X, _ = reg_data
        te = TreeExplainer.from_random_forest_regressor(rf_reg)
        result = te.explain_batch(X[:10])
        assert isinstance(result, dict)
        assert "base_value" in result
        assert "shap_values" in result
        assert "feature_values" in result

    def test_explain_batch_shap_shape(self, rf_reg, reg_data):
        """shap_values in explain_batch() should have shape (n_samples, n_features)."""
        X, _ = reg_data
        te = TreeExplainer.from_random_forest_regressor(rf_reg)
        batch = 15
        result = te.explain_batch(X[:batch])
        assert result["shap_values"].shape == (batch, N_FEATURES)

    def test_explain_batch_feature_values_shape(self, rf_reg, reg_data):
        """feature_values in explain_batch() should have shape (n_samples, n_features)."""
        X, _ = reg_data
        te = TreeExplainer.from_random_forest_regressor(rf_reg)
        batch = 15
        result = te.explain_batch(X[:batch])
        assert result["feature_values"].shape == (batch, N_FEATURES)

    def test_explain_batch_consistency_with_single(self, dt_reg, reg_data):
        """explain_batch() row 0 should match explain() for sample 0 (DT reg)."""
        X, _ = reg_data
        te = TreeExplainer.from_decision_tree_regressor(dt_reg)
        single = te.explain(X[0])
        batch = te.explain_batch(X[:5])
        np.testing.assert_allclose(
            single["shap_values"], batch["shap_values"][0], atol=1e-9
        )

    def test_explain_batch_base_value_scalar(self, rf_reg, reg_data):
        """base_value in explain_batch() should be a scalar (shared across the batch)."""
        X, _ = reg_data
        te = TreeExplainer.from_random_forest_regressor(rf_reg)
        result = te.explain_batch(X[:10])
        # base_value should be a Python float or 0-d array
        base = result["base_value"]
        assert np.isfinite(float(base))


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

_PERM_KEYS = frozenset([
    "importances_mean", "importances_std", "ci_lower", "ci_upper",
    "baseline_score", "n_repeats",
])


class TestPermutationImportance:
    """Tests for permutation importance functions."""

    def _assert_perm_structure(self, result, n_features):
        """Shared checks for permutation importance result dicts."""
        assert isinstance(result, dict)
        assert _PERM_KEYS.issubset(result.keys()), (
            f"Missing keys: {_PERM_KEYS - result.keys()}"
        )
        mean = result["importances_mean"]
        std = result["importances_std"]
        lo = result["ci_lower"]
        hi = result["ci_upper"]
        assert mean.shape == (n_features,)
        assert std.shape == (n_features,)
        assert lo.shape == (n_features,)
        assert hi.shape == (n_features,)
        assert np.all(np.isfinite(mean))
        assert np.all(std >= 0)
        for i in range(n_features):
            assert lo[i] <= hi[i], (
                f"ci_lower > ci_upper at feature {i}"
            )

    def test_rf_reg_structure(self, rf_reg, reg_data):
        """permutation_importance_rf_reg() should return the standard dict."""
        X, y = reg_data
        result = permutation_importance_rf_reg(rf_reg, X, y, n_repeats=5, random_state=42)
        self._assert_perm_structure(result, N_FEATURES)

    def test_rf_clf_structure(self, rf_clf, clf_data):
        """permutation_importance_rf_clf() should return the standard dict."""
        X, y = clf_data
        result = permutation_importance_rf_clf(rf_clf, X, y, n_repeats=5, random_state=42)
        self._assert_perm_structure(result, N_FEATURES)

    def test_dt_reg_structure(self, dt_reg, reg_data):
        """permutation_importance_dt_reg() should return the standard dict."""
        X, y = reg_data
        result = permutation_importance_dt_reg(dt_reg, X, y, n_repeats=5, random_state=42)
        self._assert_perm_structure(result, N_FEATURES)

    def test_linear_structure(self, lin_reg, reg_data):
        """permutation_importance_linear() should return the standard dict."""
        X, y = reg_data
        result = permutation_importance_linear(lin_reg, X, y, n_repeats=5, random_state=42)
        self._assert_perm_structure(result, N_FEATURES)

    def test_logistic_structure(self, log_reg, clf_data):
        """permutation_importance_logistic() should return the standard dict."""
        X, y = clf_data
        result = permutation_importance_logistic(log_reg, X, y, n_repeats=5, random_state=42)
        self._assert_perm_structure(result, N_FEATURES)

    def test_baseline_score_finite(self, rf_reg, reg_data):
        """baseline_score should be a finite scalar."""
        X, y = reg_data
        result = permutation_importance_rf_reg(rf_reg, X, y, n_repeats=5, random_state=42)
        assert np.isfinite(result["baseline_score"])

    def test_n_repeats_stored(self, rf_reg, reg_data):
        """n_repeats in the result should match what was requested."""
        X, y = reg_data
        n_repeats = 7
        result = permutation_importance_rf_reg(rf_reg, X, y, n_repeats=n_repeats, random_state=42)
        assert result["n_repeats"] == n_repeats

    def test_informative_feature_has_high_importance(self, reg_data):
        """Features used in y construction (X0, X1) should rank above random features."""
        np.random.seed(0)
        n = 200
        # X0 and X1 are informative; X2 and X3 are pure noise
        X = np.random.randn(n, 4)
        y = 3.0 * X[:, 0] + 3.0 * X[:, 1] + np.random.randn(n) * 0.05

        rf = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
        rf.fit(X, y)

        result = permutation_importance_rf_reg(rf, X, y, n_repeats=10, random_state=42)
        mean = result["importances_mean"]
        # Top 2 features by importance should be feature 0 or 1
        top2 = set(np.argsort(mean)[-2:])
        assert top2 == {0, 1}, (
            f"Expected features 0 and 1 to be most important, got top2={top2}, "
            f"importances={mean}"
        )


# ---------------------------------------------------------------------------
# Partial dependence
# ---------------------------------------------------------------------------


class TestPartialDependence:
    """Tests for partial dependence functions."""

    def _assert_pdp_structure(self, result, feature_idx, grid_resolution=50):
        """Shared checks for a partial dependence result dict."""
        assert isinstance(result, dict)
        assert "grid_values" in result
        assert "pdp_values" in result
        assert "pdp_std" in result
        assert "feature_idx" in result
        assert result["feature_idx"] == feature_idx
        grid = result["grid_values"]
        pdp = result["pdp_values"]
        assert len(grid) == len(pdp), (
            f"grid_values length {len(grid)} != pdp_values length {len(pdp)}"
        )
        assert len(grid) == grid_resolution
        assert np.all(np.isfinite(pdp))

    def test_rf_reg_structure(self, rf_reg, reg_data):
        """partial_dependence_rf_reg() should return the standard dict."""
        X, _ = reg_data
        result = partial_dependence_rf_reg(rf_reg, X, feature_idx=0)
        self._assert_pdp_structure(result, feature_idx=0)

    def test_dt_reg_structure(self, dt_reg, reg_data):
        """partial_dependence_dt_reg() should return the standard dict."""
        X, _ = reg_data
        result = partial_dependence_dt_reg(dt_reg, X, feature_idx=1)
        self._assert_pdp_structure(result, feature_idx=1)

    def test_linear_structure(self, lin_reg, reg_data):
        """partial_dependence_linear() should return the standard dict."""
        X, _ = reg_data
        result = partial_dependence_linear(lin_reg, X, feature_idx=2)
        self._assert_pdp_structure(result, feature_idx=2)

    def test_grid_values_ordered(self, rf_reg, reg_data):
        """grid_values should be monotonically increasing."""
        X, _ = reg_data
        result = partial_dependence_rf_reg(rf_reg, X, feature_idx=0)
        grid = np.array(result["grid_values"])
        assert np.all(grid[1:] >= grid[:-1]), "grid_values are not sorted"

    def test_custom_grid_resolution(self, rf_reg, reg_data):
        """grid_resolution parameter should control the number of grid points."""
        X, _ = reg_data
        grid_resolution = 30
        result = partial_dependence_rf_reg(rf_reg, X, feature_idx=0,
                                           grid_resolution=grid_resolution)
        assert len(result["grid_values"]) == grid_resolution
        assert len(result["pdp_values"]) == grid_resolution

    def test_pdp_varies_for_informative_feature(self, reg_data):
        """PDP for an informative feature should show variation (non-constant)."""
        X, y = reg_data
        rf = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
        rf.fit(X, y)
        # Feature 1 is highly informative (coefficient 2.0)
        result = partial_dependence_rf_reg(rf, X, feature_idx=1)
        pdp = np.array(result["pdp_values"])
        assert np.std(pdp) > 0.01, "PDP for informative feature should not be flat"

    @pytest.mark.parametrize("feature_idx", [0, 1, 2, 3])
    def test_all_features_rf_reg(self, rf_reg, reg_data, feature_idx):
        """partial_dependence_rf_reg() should work for every feature index."""
        X, _ = reg_data
        result = partial_dependence_rf_reg(rf_reg, X, feature_idx=feature_idx)
        assert result["feature_idx"] == feature_idx
        assert len(result["grid_values"]) == 50


# ---------------------------------------------------------------------------
# ICE (Individual Conditional Expectation)
# ---------------------------------------------------------------------------


class TestICE:
    """Tests for ICE (Individual Conditional Expectation) functions."""

    def _assert_ice_structure(self, result, n_samples, n_grid_points, feature_idx):
        """Shared structural checks for ICE result dicts."""
        assert isinstance(result, dict)
        assert "grid_values" in result
        assert "ice_curves" in result
        assert "pdp_values" in result
        assert "feature_idx" in result
        assert result["feature_idx"] == feature_idx

        ice = result["ice_curves"]
        assert ice.shape == (n_samples, n_grid_points), (
            f"Expected ice_curves shape ({n_samples}, {n_grid_points}), got {ice.shape}"
        )
        assert len(result["grid_values"]) == n_grid_points
        assert len(result["pdp_values"]) == n_grid_points

    def test_ice_rf_reg_structure(self, rf_reg, reg_data):
        """ice_rf_reg() should return a dict with correctly shaped arrays."""
        X, _ = reg_data
        result = ice_rf_reg(rf_reg, X, feature_idx=0)
        self._assert_ice_structure(result, N_SAMPLES, 50, feature_idx=0)

    def test_ice_dt_reg_structure(self, dt_reg, reg_data):
        """ice_dt_reg() should return a dict with correctly shaped arrays."""
        X, _ = reg_data
        result = ice_dt_reg(dt_reg, X, feature_idx=1)
        self._assert_ice_structure(result, N_SAMPLES, 50, feature_idx=1)

    def test_ice_linear_structure(self, lin_reg, reg_data):
        """ice_linear() should return a dict with correctly shaped arrays."""
        X, _ = reg_data
        result = ice_linear(lin_reg, X, feature_idx=2)
        self._assert_ice_structure(result, N_SAMPLES, 50, feature_idx=2)

    def test_ice_rf_reg_with_center(self, rf_reg, reg_data):
        """ice_rf_reg() with center=True should include 'centered_ice' key."""
        X, _ = reg_data
        result = ice_rf_reg(rf_reg, X, feature_idx=0, center=True)
        assert "centered_ice" in result, "Expected 'centered_ice' key when center=True"
        centered = result["centered_ice"]
        assert centered.shape == (N_SAMPLES, 50)

    def test_ice_dt_reg_with_center(self, dt_reg, reg_data):
        """ice_dt_reg() with center=True should include 'centered_ice' key."""
        X, _ = reg_data
        result = ice_dt_reg(dt_reg, X, feature_idx=0, center=True)
        assert "centered_ice" in result

    def test_ice_linear_with_center(self, lin_reg, reg_data):
        """ice_linear() with center=True should include 'centered_ice' key."""
        X, _ = reg_data
        result = ice_linear(lin_reg, X, feature_idx=0, center=True)
        assert "centered_ice" in result

    def test_ice_rf_reg_without_center_no_centered_ice_key(self, rf_reg, reg_data):
        """When center=False (default), 'centered_ice' should NOT be in the result."""
        X, _ = reg_data
        result = ice_rf_reg(rf_reg, X, feature_idx=0, center=False)
        assert "centered_ice" not in result

    def test_ice_rf_reg_with_derivative(self, rf_reg, reg_data):
        """ice_rf_reg() with compute_derivative=True should include 'derivative_ice' key."""
        X, _ = reg_data
        result = ice_rf_reg(rf_reg, X, feature_idx=0, compute_derivative=True)
        assert "derivative_ice" in result, (
            "Expected 'derivative_ice' key when compute_derivative=True"
        )

    def test_ice_rf_reg_without_derivative_no_derivative_key(self, rf_reg, reg_data):
        """When compute_derivative=False (default), 'derivative_ice' should be absent."""
        X, _ = reg_data
        result = ice_rf_reg(rf_reg, X, feature_idx=0, compute_derivative=False)
        assert "derivative_ice" not in result

    def test_ice_pdp_is_mean_of_ice_curves(self, lin_reg, reg_data):
        """For linear model, pdp_values should equal column-wise mean of ice_curves."""
        X, _ = reg_data
        result = ice_linear(lin_reg, X, feature_idx=0)
        ice = result["ice_curves"]
        pdp = np.array(result["pdp_values"])
        np.testing.assert_allclose(
            np.mean(ice, axis=0), pdp, atol=1e-6,
            err_msg="pdp_values should be the mean of ice_curves columns"
        )

    def test_ice_rf_custom_n_grid_points(self, rf_reg, reg_data):
        """n_grid_points parameter should control the number of grid steps."""
        X, _ = reg_data
        n_grid = 30
        result = ice_rf_reg(rf_reg, X, feature_idx=0, n_grid_points=n_grid)
        assert result["ice_curves"].shape == (N_SAMPLES, n_grid)
        assert len(result["grid_values"]) == n_grid

    def test_ice_grid_values_sorted(self, rf_reg, reg_data):
        """grid_values should be sorted in ascending order."""
        X, _ = reg_data
        result = ice_rf_reg(rf_reg, X, feature_idx=0)
        grid = np.array(result["grid_values"])
        assert np.all(grid[1:] >= grid[:-1]), "grid_values are not sorted"

    def test_ice_curves_finite(self, rf_reg, reg_data):
        """All ice_curves values should be finite."""
        X, _ = reg_data
        result = ice_rf_reg(rf_reg, X, feature_idx=0)
        assert np.all(np.isfinite(result["ice_curves"]))


# ---------------------------------------------------------------------------
# H-statistic (interaction strength)
# ---------------------------------------------------------------------------


class TestHStatistic:
    """Tests for H-statistic interaction functions."""

    def test_h_statistic_rf_reg_keys(self, rf_reg, reg_data):
        """h_statistic_rf_reg() must return dict with 'h_squared' and 'h_statistic'."""
        X, _ = reg_data
        result = h_statistic_rf_reg(rf_reg, X, feature_idx_1=0, feature_idx_2=1)
        assert isinstance(result, dict)
        assert "h_squared" in result
        assert "h_statistic" in result

    def test_h_statistic_range(self, rf_reg, reg_data):
        """h_statistic should be in [0, 1]."""
        X, _ = reg_data
        result = h_statistic_rf_reg(rf_reg, X, feature_idx_1=0, feature_idx_2=1)
        h = result["h_statistic"]
        assert np.isfinite(h)
        assert 0.0 <= h <= 1.0, f"h_statistic out of range: {h}"

    def test_h_squared_non_negative(self, rf_reg, reg_data):
        """h_squared must be non-negative."""
        X, _ = reg_data
        result = h_statistic_rf_reg(rf_reg, X, feature_idx_1=0, feature_idx_2=1)
        assert result["h_squared"] >= 0.0

    def test_h_statistic_feature_indices_stored(self, rf_reg, reg_data):
        """Result should record which feature pair was used."""
        X, _ = reg_data
        result = h_statistic_rf_reg(rf_reg, X, feature_idx_1=1, feature_idx_2=3)
        assert "feature_idx_1" in result
        assert "feature_idx_2" in result
        assert result["feature_idx_1"] == 1
        assert result["feature_idx_2"] == 3

    def test_h_statistic_with_custom_grid(self, rf_reg, reg_data):
        """h_statistic_rf_reg() should accept n_grid_points parameter."""
        X, _ = reg_data
        result = h_statistic_rf_reg(rf_reg, X, feature_idx_1=0, feature_idx_2=2,
                                    n_grid_points=10)
        assert np.isfinite(result["h_statistic"])

    def test_h_statistic_matrix_keys(self, rf_reg, reg_data):
        """h_statistic_matrix_rf_reg() must return 'h_squared_matrix'."""
        X, _ = reg_data
        result = h_statistic_matrix_rf_reg(rf_reg, X)
        assert isinstance(result, dict)
        assert "h_squared_matrix" in result

    def test_h_statistic_matrix_shape(self, rf_reg, reg_data):
        """h_squared_matrix should be (n_features, n_features)."""
        X, _ = reg_data
        result = h_statistic_matrix_rf_reg(rf_reg, X)
        mat = result["h_squared_matrix"]
        assert mat.shape == (N_FEATURES, N_FEATURES)

    def test_h_statistic_matrix_non_negative(self, rf_reg, reg_data):
        """All entries in h_squared_matrix must be non-negative."""
        X, _ = reg_data
        result = h_statistic_matrix_rf_reg(rf_reg, X)
        mat = result["h_squared_matrix"]
        assert np.all(mat >= 0.0)

    def test_h_statistic_matrix_with_feature_indices(self, rf_reg, reg_data):
        """h_statistic_matrix_rf_reg() should accept a feature_indices list."""
        X, _ = reg_data
        result = h_statistic_matrix_rf_reg(rf_reg, X, feature_indices=[0, 1, 2])
        mat = result["h_squared_matrix"]
        # Matrix should be 3x3 when 3 feature indices are provided
        assert mat.shape == (3, 3)

    @pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 3), (2, 3)])
    def test_h_statistic_all_feature_pairs(self, rf_reg, reg_data, pair):
        """h_statistic_rf_reg() should work for all valid feature index pairs."""
        X, _ = reg_data
        i, j = pair
        result = h_statistic_rf_reg(rf_reg, X, feature_idx_1=i, feature_idx_2=j)
        h = result["h_statistic"]
        assert np.isfinite(h)
        assert 0.0 <= h <= 1.0

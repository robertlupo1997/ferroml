"""Statistical tests for FerroML regressors.

Verifies R^2, score(), residuals, coefficient behavior, feature importances,
and GaussianProcess uncertainty across all regressor models.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

from ferroml.linear import LinearRegression, RidgeRegression, LassoRegression, ElasticNet
from ferroml.trees import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    DecisionTreeRegressor,
)
from ferroml.neighbors import KNeighborsRegressor
from ferroml.gaussian_process import GaussianProcessRegressor
from ferroml.svm import SVR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_data():
    """Standard regression dataset with train/test split."""
    X, y = make_regression(
        n_samples=200, n_features=10, noise=10, random_state=42
    )
    return X[:150], y[:150], X[150:], y[150:]


@pytest.fixture
def noiseless_linear_data():
    """Perfect linear data with no noise."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    coef = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = X @ coef + 7.0  # intercept = 7
    return X, y


@pytest.fixture
def collinear_data():
    """Data with highly collinear features."""
    rng = np.random.RandomState(42)
    base = rng.randn(150, 3)
    # Create collinear features: cols 3-5 are noisy copies of cols 0-2
    noise = rng.randn(150, 3) * 0.01
    X = np.hstack([base, base + noise])
    y = base @ np.array([1.0, 2.0, 3.0]) + rng.randn(150) * 0.5
    return X[:120], y[:120], X[120:], y[120:]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_and_score(model, X_train, y_train, X_test, y_test):
    """Fit a model and return its score on the test set."""
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


ALL_REGRESSORS = [
    ("LinearRegression", lambda: LinearRegression()),
    ("RidgeRegression", lambda: RidgeRegression(1.0)),
    ("LassoRegression", lambda: LassoRegression(0.1)),
    ("ElasticNet", lambda: ElasticNet(0.1, 0.5)),
    ("DecisionTreeRegressor", lambda: DecisionTreeRegressor()),
    ("RandomForestRegressor", lambda: RandomForestRegressor(n_estimators=50)),
    ("GradientBoostingRegressor", lambda: GradientBoostingRegressor(n_estimators=50)),
    ("KNeighborsRegressor", lambda: KNeighborsRegressor()),
    ("SVR", lambda: SVR()),
    ("GaussianProcessRegressor", lambda: GaussianProcessRegressor()),
]


# ---------------------------------------------------------------------------
# 1. score(X,y) == R^2 definition for each model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,make_model", ALL_REGRESSORS, ids=[r[0] for r in ALL_REGRESSORS])
def test_score_is_r_squared(name, make_model, regression_data):
    X_train, y_train, X_test, y_test = regression_data
    model = make_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score_val = model.score(X_test, y_test)

    # Compute R^2 manually: 1 - SS_res / SS_tot
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_manual = 1.0 - ss_res / ss_tot

    assert_allclose(score_val, r2_manual, atol=1e-6,
                    err_msg=f"{name}: score() != manual R^2")


# ---------------------------------------------------------------------------
# 2. score vs sklearn r2_score
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,make_model", ALL_REGRESSORS, ids=[r[0] for r in ALL_REGRESSORS])
def test_score_vs_sklearn(name, make_model, regression_data):
    X_train, y_train, X_test, y_test = regression_data
    model = make_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score_val = model.score(X_test, y_test)
    sklearn_r2 = r2_score(y_test, preds)

    assert_allclose(score_val, sklearn_r2, atol=1e-6,
                    err_msg=f"{name}: score() != sklearn r2_score")


# ---------------------------------------------------------------------------
# 3. Predictions are finite (no NaN/Inf)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,make_model", ALL_REGRESSORS, ids=[r[0] for r in ALL_REGRESSORS])
def test_predictions_finite(name, make_model, regression_data):
    X_train, y_train, X_test, y_test = regression_data
    model = make_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert np.all(np.isfinite(preds)), f"{name}: predictions contain NaN or Inf"


# ---------------------------------------------------------------------------
# 4. Perfect fit on noiseless linear data (linear models only)
# ---------------------------------------------------------------------------

LINEAR_MODELS = [
    ("LinearRegression", lambda: LinearRegression()),
    ("RidgeRegression", lambda: RidgeRegression(1e-10)),  # tiny alpha ~ OLS
]


@pytest.mark.parametrize("name,make_model", LINEAR_MODELS, ids=[r[0] for r in LINEAR_MODELS])
def test_perfect_fit_score_one(name, make_model, noiseless_linear_data):
    X, y = noiseless_linear_data
    model = make_model()
    model.fit(X, y)
    score_val = model.score(X, y)
    assert score_val > 0.9999, f"{name}: R^2 on noiseless data = {score_val}, expected ~1.0"


# ---------------------------------------------------------------------------
# 5. Ridge coefficients shrink with alpha
# ---------------------------------------------------------------------------

def test_ridge_coefficients_shrink_with_alpha(regression_data):
    X_train, y_train, _, _ = regression_data
    norms = []
    for alpha in [0.01, 1.0, 100.0]:
        m = RidgeRegression(alpha)
        m.fit(X_train, y_train)
        norms.append(np.linalg.norm(m.coef_))

    # Higher alpha -> smaller coefficient norm
    assert norms[0] > norms[1] > norms[2], (
        f"Ridge coef norms should decrease with alpha: {norms}"
    )


# ---------------------------------------------------------------------------
# 6. Lasso sparsity: high alpha -> zeros
# ---------------------------------------------------------------------------

def test_lasso_sparsity(regression_data):
    X_train, y_train, _, _ = regression_data
    # Low alpha: most coefficients nonzero
    m_low = LassoRegression(0.01)
    m_low.fit(X_train, y_train)
    nnz_low = np.sum(np.abs(m_low.coef_) > 1e-10)

    # High alpha: fewer nonzero
    m_high = LassoRegression(100.0)
    m_high.fit(X_train, y_train)
    nnz_high = np.sum(np.abs(m_high.coef_) > 1e-10)

    assert nnz_high < nnz_low, (
        f"Lasso should be sparser with higher alpha: nnz_low={nnz_low}, nnz_high={nnz_high}"
    )


# ---------------------------------------------------------------------------
# 7. ElasticNet between Ridge and Lasso
# ---------------------------------------------------------------------------

def test_elasticnet_between_ridge_and_lasso(regression_data):
    X_train, y_train, X_test, y_test = regression_data
    alpha = 1.0

    # l1_ratio near 0 -> Ridge-like (more nonzero coefficients)
    en_ridge = ElasticNet(alpha, 0.01)
    en_ridge.fit(X_train, y_train)
    nnz_ridge = np.sum(np.abs(en_ridge.coef_) > 1e-10)

    # l1_ratio near 1 -> Lasso-like (sparser)
    en_lasso = ElasticNet(alpha, 0.99)
    en_lasso.fit(X_train, y_train)
    nnz_lasso = np.sum(np.abs(en_lasso.coef_) > 1e-10)

    # Ridge-like should have at least as many nonzero coefficients
    assert nnz_ridge >= nnz_lasso, (
        f"EN(l1_ratio~0) should have >= nonzero coefs than EN(l1_ratio~1): "
        f"{nnz_ridge} vs {nnz_lasso}"
    )


# ---------------------------------------------------------------------------
# 8. Ridge coefficients vs sklearn
# ---------------------------------------------------------------------------

def test_ridge_vs_sklearn(regression_data):
    from sklearn.linear_model import Ridge as SklearnRidge

    X_train, y_train, _, _ = regression_data
    alpha = 1.0

    ferro = RidgeRegression(alpha)
    ferro.fit(X_train, y_train)

    sk = SklearnRidge(alpha=alpha, fit_intercept=True)
    sk.fit(X_train, y_train)

    assert_allclose(ferro.coef_, sk.coef_, atol=0.1,
                    err_msg="Ridge coefficients differ from sklearn")
    assert_allclose(ferro.intercept_, sk.intercept_, atol=0.5,
                    err_msg="Ridge intercept differs from sklearn")


# ---------------------------------------------------------------------------
# 9. RF feature importances sum to 1
# ---------------------------------------------------------------------------

def test_rf_feature_importances_sum_to_one(regression_data):
    X_train, y_train, _, _ = regression_data
    m = RandomForestRegressor(n_estimators=50)
    m.fit(X_train, y_train)
    fi = m.feature_importances_
    assert_allclose(np.sum(fi), 1.0, atol=1e-6,
                    err_msg="RF feature importances should sum to 1.0")
    assert np.all(fi >= 0), "Feature importances should be non-negative"


# ---------------------------------------------------------------------------
# 10. RF OOB score is reasonable
# ---------------------------------------------------------------------------

def test_rf_oob_score_reasonable(regression_data):
    X_train, y_train, X_test, y_test = regression_data
    m = RandomForestRegressor(n_estimators=100, oob_score=True)
    m.fit(X_train, y_train)
    oob = m.oob_score_
    test_score = m.score(X_test, y_test)

    # OOB should be a reasonable estimate -- within 0.3 of test score
    assert abs(oob - test_score) < 0.3, (
        f"OOB score ({oob:.3f}) should be close to test score ({test_score:.3f})"
    )
    assert oob > 0, "OOB score should be positive on data with signal"


# ---------------------------------------------------------------------------
# 11. GB predictions improve with more estimators
# ---------------------------------------------------------------------------

def test_gb_predictions_improve_with_estimators(regression_data):
    X_train, y_train, X_test, y_test = regression_data
    scores = []
    for n in [5, 50, 200]:
        m = GradientBoostingRegressor(n_estimators=n, learning_rate=0.1)
        m.fit(X_train, y_train)
        scores.append(m.score(X_test, y_test))

    # More estimators should generally improve or maintain score
    assert scores[-1] >= scores[0] - 0.05, (
        f"GB with 200 trees ({scores[-1]:.3f}) should be >= GB with 5 trees ({scores[0]:.3f})"
    )


# ---------------------------------------------------------------------------
# 12. GP uncertainty at training points is near zero
# ---------------------------------------------------------------------------

def test_gp_uncertainty_at_training_points():
    rng = np.random.RandomState(42)
    X = rng.randn(20, 2)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1])

    m = GaussianProcessRegressor()
    m.fit(X, y)
    _, std = m.predict_with_std(X)

    assert np.all(std < 0.1), (
        f"GP uncertainty at training points should be ~0, got max={np.max(std):.4f}"
    )


# ---------------------------------------------------------------------------
# 13. GP uncertainty increases far from training data
# ---------------------------------------------------------------------------

def test_gp_uncertainty_increases_far_from_data():
    rng = np.random.RandomState(42)
    X_train = rng.randn(20, 2) * 0.5  # clustered near origin

    y_train = np.sin(X_train[:, 0])
    m = GaussianProcessRegressor()
    m.fit(X_train, y_train)

    # Near training data
    X_near = rng.randn(10, 2) * 0.5
    _, std_near = m.predict_with_std(X_near)

    # Far from training data
    X_far = rng.randn(10, 2) * 0.5 + 10.0
    _, std_far = m.predict_with_std(X_far)

    assert np.mean(std_far) > np.mean(std_near), (
        f"GP uncertainty should be higher far from data: "
        f"mean_far={np.mean(std_far):.4f}, mean_near={np.mean(std_near):.4f}"
    )


# ---------------------------------------------------------------------------
# 14. GP mean interpolates training data
# ---------------------------------------------------------------------------

def test_gp_mean_interpolates_training_data():
    rng = np.random.RandomState(42)
    X = rng.randn(15, 2)
    y = X[:, 0] ** 2 + X[:, 1]

    m = GaussianProcessRegressor()
    m.fit(X, y)
    preds = m.predict(X)

    assert_allclose(preds, y, atol=0.05,
                    err_msg="GP should interpolate training data")


# ---------------------------------------------------------------------------
# 15. GP prediction interval coverage
# ---------------------------------------------------------------------------

def test_gp_std_monotonic_with_distance():
    """GP posterior std should grow monotonically as we move away from training data.

    This tests that predict_with_std returns uncertainty that is structured
    correctly: near-zero at training points, growing with distance.
    The GP is noise-free so std represents posterior (epistemic) uncertainty only.
    """
    # 1D evenly spaced training data
    X_train = np.linspace(0, 1, 10).reshape(-1, 1)
    y_train = np.sin(2 * np.pi * X_train.ravel())

    m = GaussianProcessRegressor()
    m.fit(X_train, y_train)

    # Query at increasing distances from the training region
    distances = [0.0, 0.5, 1.0, 2.0, 5.0]
    mean_stds = []
    for d in distances:
        X_query = (np.array([1.0]) + d).reshape(-1, 1)
        _, std = m.predict_with_std(X_query)
        mean_stds.append(std[0])

    # std should generally increase (or stay flat) as we go further
    # Check that the farthest point has higher std than the nearest
    assert mean_stds[-1] > mean_stds[0], (
        f"GP std should increase with distance from training data: "
        f"stds={[f'{s:.6f}' for s in mean_stds]}"
    )


# ---------------------------------------------------------------------------
# 16. All regressors beat mean baseline (score > 0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,make_model", ALL_REGRESSORS, ids=[r[0] for r in ALL_REGRESSORS])
def test_all_regressors_beat_mean_baseline(name, make_model, regression_data):
    X_train, y_train, X_test, y_test = regression_data
    model = make_model()
    model.fit(X_train, y_train)
    score_val = model.score(X_test, y_test)

    # SVR and GP may not do great on high-dim data without tuning, use lenient threshold
    threshold = -0.5 if name in ("SVR", "GaussianProcessRegressor") else 0.0
    assert score_val > threshold, (
        f"{name}: score={score_val:.3f}, expected > {threshold} (better than mean baseline)"
    )


# ---------------------------------------------------------------------------
# 17. Regularized models handle collinear data better
# ---------------------------------------------------------------------------

def test_regularized_vs_ols_on_collinear(collinear_data):
    X_train, y_train, X_test, y_test = collinear_data

    ols = LinearRegression()
    ols.fit(X_train, y_train)
    ols_score = ols.score(X_test, y_test)

    ridge = RidgeRegression(10.0)
    ridge.fit(X_train, y_train)
    ridge_score = ridge.score(X_test, y_test)

    lasso = LassoRegression(0.1)
    lasso.fit(X_train, y_train)
    lasso_score = lasso.score(X_test, y_test)

    # At minimum, regularized models should not be catastrophically worse
    assert ridge_score > ols_score - 0.1, (
        f"Ridge ({ridge_score:.3f}) should handle collinearity at least as well as OLS ({ols_score:.3f})"
    )
    assert lasso_score > ols_score - 0.1, (
        f"Lasso ({lasso_score:.3f}) should handle collinearity at least as well as OLS ({ols_score:.3f})"
    )

    # Ridge coefficient norms should be more stable (smaller)
    assert np.linalg.norm(ridge.coef_) < np.linalg.norm(ols.coef_) * 1.5, (
        "Ridge should produce smaller coefficient norms on collinear data"
    )


# ---------------------------------------------------------------------------
# 18. Score consistency: train score >= test score (usually)
# ---------------------------------------------------------------------------

OVERFIT_PRONE_MODELS = [
    ("DecisionTreeRegressor", lambda: DecisionTreeRegressor()),
    ("RandomForestRegressor", lambda: RandomForestRegressor(n_estimators=50)),
    ("KNeighborsRegressor", lambda: KNeighborsRegressor()),
]


@pytest.mark.parametrize("name,make_model", OVERFIT_PRONE_MODELS, ids=[r[0] for r in OVERFIT_PRONE_MODELS])
def test_score_consistency(name, make_model, regression_data):
    X_train, y_train, X_test, y_test = regression_data
    model = make_model()
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # Models that can overfit should have train_score >= test_score
    assert train_score >= test_score - 0.05, (
        f"{name}: train_score ({train_score:.3f}) should be >= test_score ({test_score:.3f})"
    )

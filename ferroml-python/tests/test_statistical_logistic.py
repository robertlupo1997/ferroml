"""Statistical validation of FerroML LogisticRegression against statsmodels and scipy.

Verifies coefficients, standard errors, odds ratios, log-likelihood, AIC, BIC,
pseudo R-squared, likelihood ratio test, predict_proba, and decision_function
against reference implementations.
"""

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import accuracy_score

from ferroml.linear import LogisticRegression


@pytest.fixture
def logistic_models():
    """Fit both FerroML and statsmodels logistic regression on the same data."""
    np.random.seed(42)
    n = 300
    X = np.random.randn(n, 3)
    z = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]
    prob = 1 / (1 + np.exp(-z))
    y = (np.random.rand(n) < prob).astype(float)

    X_sm = sm.add_constant(X)
    sm_model = sm.Logit(y, X_sm).fit(disp=0)

    ferro = LogisticRegression()
    ferro.fit(X, y)
    return ferro, sm_model, X, y


# ---- Test 1: Coefficients vs statsmodels ----
def test_coefficients_vs_statsmodels(logistic_models):
    ferro, sm_model, X, y = logistic_models
    # statsmodels params: [intercept, coef0, coef1, coef2]
    sm_coefs = sm_model.params[1:]
    sm_intercept = sm_model.params[0]

    np.testing.assert_allclose(ferro.coef_, sm_coefs, atol=1e-2,
                               err_msg="Coefficients differ from statsmodels")
    np.testing.assert_allclose(ferro.intercept_, sm_intercept, atol=1e-2,
                               err_msg="Intercept differs from statsmodels")


# ---- Test 2: Standard errors vs statsmodels (via odds_ratios_with_ci) ----
def test_standard_errors_vs_statsmodels(logistic_models):
    ferro, sm_model, X, y = logistic_models
    # Extract SEs indirectly: CI width on log scale relates to SE
    # statsmodels bse gives standard errors for [intercept, coef0, coef1, coef2]
    sm_bse = sm_model.bse[1:]  # feature SEs only

    # FerroML odds_ratios_with_ci at 95% => CI on odds ratio scale
    # CI on log scale: log(OR) +/- z*SE => SE = (log(ci_upper) - log(ci_lower)) / (2*1.96)
    or_info = ferro.odds_ratios_with_ci(0.95)
    ferro_ses = []
    for info in or_info:
        ci_lower = info["ci_lower"]
        ci_upper = info["ci_upper"]
        se = (np.log(ci_upper) - np.log(ci_lower)) / (2 * 1.96)
        ferro_ses.append(se)

    np.testing.assert_allclose(ferro_ses, sm_bse, atol=0.1,
                               err_msg="Standard errors differ from statsmodels")


# ---- Test 3: Odds ratios match exp(coef) ----
def test_odds_ratios_match_exp_coef(logistic_models):
    ferro, sm_model, X, y = logistic_models
    expected_or = np.exp(ferro.coef_)
    actual_or = ferro.odds_ratios()
    np.testing.assert_allclose(actual_or, expected_or, rtol=1e-6,
                               err_msg="Odds ratios should be exp(coef)")


# ---- Test 4: Odds ratio CIs vs statsmodels ----
def test_odds_ratio_cis_vs_statsmodels(logistic_models):
    ferro, sm_model, X, y = logistic_models
    # statsmodels conf_int on log-odds scale, then exponentiate
    sm_ci = sm_model.conf_int(alpha=0.05)
    sm_or_ci_lower = np.exp(sm_ci[1:, 0])  # skip intercept
    sm_or_ci_upper = np.exp(sm_ci[1:, 1])

    or_info = ferro.odds_ratios_with_ci(0.95)
    ferro_ci_lower = np.array([info["ci_lower"] for info in or_info])
    ferro_ci_upper = np.array([info["ci_upper"] for info in or_info])

    np.testing.assert_allclose(ferro_ci_lower, sm_or_ci_lower, atol=0.5,
                               err_msg="Odds ratio CI lower bounds differ")
    np.testing.assert_allclose(ferro_ci_upper, sm_or_ci_upper, atol=0.5,
                               err_msg="Odds ratio CI upper bounds differ")


# ---- Test 5: Log-likelihood vs statsmodels ----
def test_log_likelihood_vs_statsmodels(logistic_models):
    ferro, sm_model, X, y = logistic_models
    np.testing.assert_allclose(ferro.log_likelihood(), sm_model.llf, atol=1e-1,
                               err_msg="Log-likelihood differs from statsmodels")


# ---- Test 6: AIC vs statsmodels ----
def test_aic_vs_statsmodels(logistic_models):
    ferro, sm_model, X, y = logistic_models
    np.testing.assert_allclose(ferro.aic(), sm_model.aic, atol=1.0,
                               err_msg="AIC differs from statsmodels")


# ---- Test 7: BIC vs statsmodels ----
def test_bic_vs_statsmodels(logistic_models):
    ferro, sm_model, X, y = logistic_models
    np.testing.assert_allclose(ferro.bic(), sm_model.bic, atol=1.0,
                               err_msg="BIC differs from statsmodels")


# ---- Test 8: Pseudo R-squared vs statsmodels ----
def test_pseudo_r_squared_vs_statsmodels(logistic_models):
    ferro, sm_model, X, y = logistic_models
    # McFadden's pseudo R^2 = 1 - llf/llnull
    np.testing.assert_allclose(ferro.pseudo_r_squared(), sm_model.prsquared, atol=0.02,
                               err_msg="Pseudo R-squared differs from statsmodels")


# ---- Test 9: Likelihood ratio test ----
def test_likelihood_ratio_test(logistic_models):
    ferro, sm_model, X, y = logistic_models
    lr_stat, p_value = ferro.likelihood_ratio_test()

    # LR stat = 2 * (llf - llnull)
    sm_lr_stat = sm_model.llr
    sm_p_value = sm_model.llr_pvalue

    np.testing.assert_allclose(lr_stat, sm_lr_stat, atol=1.0,
                               err_msg="LR statistic differs from statsmodels")
    # p-value should be very small for both
    assert p_value < 0.001, f"LR test p-value should be significant, got {p_value}"
    assert sm_p_value < 0.001, f"statsmodels LR test p-value should be significant"


# ---- Test 10: Predict_proba calibration on well-separated data ----
def test_predict_proba_calibration():
    np.random.seed(123)
    X = np.vstack([np.random.randn(50, 2) + 5, np.random.randn(50, 2) - 5])
    y = np.array([1.0] * 50 + [0.0] * 50)

    m = LogisticRegression()
    m.fit(X, y)
    proba = m.predict_proba(X)

    # Class 1 samples should have high probability
    assert np.all(proba[:50, 1] > 0.9), "Well-separated class 1 should have P > 0.9"
    # Class 0 samples should have low probability for class 1
    assert np.all(proba[50:, 1] < 0.1), "Well-separated class 0 should have P < 0.1"


# ---- Test 11: Predict_proba sums to 1 ----
def test_predict_proba_sums_to_one(logistic_models):
    ferro, sm_model, X, y = logistic_models
    proba = ferro.predict_proba(X)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10,
                               err_msg="Predicted probabilities should sum to 1")


# ---- Test 12: Perfect separation produces large coefficients ----
def test_perfect_separation_large_coefficients():
    np.random.seed(99)
    X = np.array([[i] for i in range(-10, 11)], dtype=float)
    y = (X[:, 0] >= 0).astype(float)

    m = LogisticRegression()
    m.fit(X, y)
    # With perfect separation, coefficient magnitude should be large
    assert np.abs(m.coef_[0]) > 1.0, (
        f"Coefficient should be large for perfectly separated data, got {m.coef_[0]}"
    )


# ---- Test 13: Score matches sklearn accuracy ----
def test_score_matches_sklearn_accuracy(logistic_models):
    ferro, sm_model, X, y = logistic_models
    preds = ferro.predict(X)
    expected_accuracy = accuracy_score(y, preds)
    actual_score = ferro.score(X, y)
    np.testing.assert_allclose(actual_score, expected_accuracy, atol=1e-10,
                               err_msg="score() should match accuracy_score")


# ---- Test 14: Decision function sign matches prediction ----
def test_decision_function_sign_matches_predict(logistic_models):
    ferro, sm_model, X, y = logistic_models
    decision = ferro.decision_function(X)
    preds = ferro.predict(X)

    # Positive decision => class 1, negative => class 0
    predicted_from_decision = (decision >= 0).astype(float)
    np.testing.assert_array_equal(predicted_from_decision, preds,
                                  err_msg="sign(decision_function) should match predict")


# ---- Test 15: Null model probability calibration ----
def test_null_model_probability_near_half():
    np.random.seed(77)
    n = 500
    X = np.random.randn(n, 3)
    y = np.random.binomial(1, 0.5, size=n).astype(float)

    m = LogisticRegression()
    m.fit(X, y)
    proba = m.predict_proba(X)

    # With random labels, predicted probabilities should be near 0.5
    mean_prob = proba[:, 1].mean()
    assert 0.3 < mean_prob < 0.7, (
        f"Mean predicted probability on random data should be near 0.5, got {mean_prob}"
    )
    # Standard deviation of probabilities should be small
    std_prob = proba[:, 1].std()
    assert std_prob < 0.2, (
        f"Probability std on random data should be small, got {std_prob}"
    )

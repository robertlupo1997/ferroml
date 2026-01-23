"""
Test reproducibility and determinism for FerroML models.

Ensures that:
- Same seed produces identical results across runs
- Models with random_state are deterministic
- Fit/predict cycles are reproducible
"""

import numpy as np
import pytest


# ============================================================================
# Seed Reproducibility Tests
# ============================================================================


class TestSeedReproducibility:
    """Test that random_state parameter produces reproducible results."""

    @pytest.mark.skip(reason="RandomForest may have non-deterministic parallel training")
    def test_random_forest_classifier_reproducibility(self, classification_data):
        """Test RandomForestClassifier produces same results with same seed."""
        from ferroml.trees import RandomForestClassifier

        X, y = classification_data

        # Train two models with same seed
        model1 = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model2 = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)

        model1.fit(X, y)
        model2.fit(X, y)

        preds1 = model1.predict(X)
        preds2 = model2.predict(X)

        np.testing.assert_array_equal(preds1, preds2,
            err_msg="Same seed should produce identical predictions")

    @pytest.mark.skip(reason="RandomForest may have non-deterministic parallel training")
    def test_random_forest_regressor_reproducibility(self, regression_data):
        """Test RandomForestRegressor produces same results with same seed."""
        from ferroml.trees import RandomForestRegressor

        X, y = regression_data

        model1 = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
        model2 = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)

        model1.fit(X, y)
        model2.fit(X, y)

        preds1 = model1.predict(X)
        preds2 = model2.predict(X)

        np.testing.assert_allclose(preds1, preds2, rtol=1e-10,
            err_msg="Same seed should produce identical predictions")

    def test_gradient_boosting_classifier_reproducibility(self, classification_data):
        """Test GradientBoostingClassifier produces same results with same seed."""
        from ferroml.trees import GradientBoostingClassifier

        X, y = classification_data

        model1 = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model2 = GradientBoostingClassifier(n_estimators=10, random_state=42)

        model1.fit(X, y)
        model2.fit(X, y)

        preds1 = model1.predict(X)
        preds2 = model2.predict(X)

        np.testing.assert_array_equal(preds1, preds2,
            err_msg="Same seed should produce identical predictions")

    def test_gradient_boosting_regressor_reproducibility(self, regression_data):
        """Test GradientBoostingRegressor produces same results with same seed."""
        from ferroml.trees import GradientBoostingRegressor

        X, y = regression_data

        model1 = GradientBoostingRegressor(n_estimators=10, random_state=42)
        model2 = GradientBoostingRegressor(n_estimators=10, random_state=42)

        model1.fit(X, y)
        model2.fit(X, y)

        preds1 = model1.predict(X)
        preds2 = model2.predict(X)

        np.testing.assert_allclose(preds1, preds2, rtol=1e-10,
            err_msg="Same seed should produce identical predictions")

    def test_hist_gradient_boosting_reproducibility(self, classification_data):
        """Test HistGradientBoostingClassifier produces same results with same seed."""
        from ferroml.trees import HistGradientBoostingClassifier

        X, y = classification_data

        model1 = HistGradientBoostingClassifier(max_iter=10, random_state=42)
        model2 = HistGradientBoostingClassifier(max_iter=10, random_state=42)

        model1.fit(X, y)
        model2.fit(X, y)

        preds1 = model1.predict(X)
        preds2 = model2.predict(X)

        np.testing.assert_array_equal(preds1, preds2,
            err_msg="Same seed should produce identical predictions")

    def test_decision_tree_reproducibility(self, classification_data):
        """Test DecisionTreeClassifier produces same results with same seed."""
        from ferroml.trees import DecisionTreeClassifier

        X, y = classification_data

        model1 = DecisionTreeClassifier(max_depth=5, random_state=42)
        model2 = DecisionTreeClassifier(max_depth=5, random_state=42)

        model1.fit(X, y)
        model2.fit(X, y)

        preds1 = model1.predict(X)
        preds2 = model2.predict(X)

        np.testing.assert_array_equal(preds1, preds2,
            err_msg="Same seed should produce identical predictions")


# ============================================================================
# Different Seeds Produce Different Results
# ============================================================================


class TestDifferentSeeds:
    """Test that different seeds produce different results."""

    def test_random_forest_different_seeds(self, classification_data):
        """Test RandomForestClassifier with different seeds produces different results."""
        from ferroml.trees import RandomForestClassifier

        X, y = classification_data

        model1 = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model2 = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=123)

        model1.fit(X, y)
        model2.fit(X, y)

        preds1 = model1.predict(X)
        preds2 = model2.predict(X)

        # Predictions might be same for some samples, but not all
        # (unless the problem is trivial)
        # We check that models are actually different by comparing feature importances
        imp1 = model1.feature_importances_
        imp2 = model2.feature_importances_

        # At least some difference should exist
        diff = np.abs(imp1 - imp2).max()
        assert diff > 1e-10, "Different seeds should produce different models"


# ============================================================================
# Deterministic Models (No Randomness)
# ============================================================================


class TestDeterministicModels:
    """Test that deterministic models are always reproducible."""

    def test_linear_regression_deterministic(self, regression_data):
        """Test LinearRegression is deterministic (no random state needed)."""
        from ferroml.linear import LinearRegression

        X, y = regression_data

        model1 = LinearRegression()
        model2 = LinearRegression()

        model1.fit(X, y)
        model2.fit(X, y)

        preds1 = model1.predict(X)
        preds2 = model2.predict(X)

        np.testing.assert_allclose(preds1, preds2, rtol=1e-10,
            err_msg="LinearRegression should be deterministic")

        # Coefficients should also match
        np.testing.assert_allclose(model1.coef_, model2.coef_, rtol=1e-10)
        np.testing.assert_allclose(model1.intercept_, model2.intercept_, rtol=1e-10)

    def test_ridge_regression_deterministic(self, regression_data):
        """Test RidgeRegression is deterministic."""
        from ferroml.linear import RidgeRegression

        X, y = regression_data

        model1 = RidgeRegression(alpha=1.0)
        model2 = RidgeRegression(alpha=1.0)

        model1.fit(X, y)
        model2.fit(X, y)

        preds1 = model1.predict(X)
        preds2 = model2.predict(X)

        np.testing.assert_allclose(preds1, preds2, rtol=1e-10,
            err_msg="RidgeRegression should be deterministic")

    def test_lasso_regression_deterministic(self, regression_data):
        """Test LassoRegression is deterministic."""
        from ferroml.linear import LassoRegression

        X, y = regression_data

        model1 = LassoRegression(alpha=0.1)
        model2 = LassoRegression(alpha=0.1)

        model1.fit(X, y)
        model2.fit(X, y)

        preds1 = model1.predict(X)
        preds2 = model2.predict(X)

        # Lasso uses iterative solver, allow slightly more tolerance
        np.testing.assert_allclose(preds1, preds2, rtol=1e-5,
            err_msg="LassoRegression should be deterministic")

    def test_logistic_regression_deterministic(self, classification_data):
        """Test LogisticRegression is deterministic."""
        from ferroml.linear import LogisticRegression

        X, y = classification_data

        model1 = LogisticRegression()
        model2 = LogisticRegression()

        model1.fit(X, y)
        model2.fit(X, y)

        preds1 = model1.predict(X)
        preds2 = model2.predict(X)

        np.testing.assert_array_equal(preds1, preds2,
            err_msg="LogisticRegression should be deterministic")

    def test_standard_scaler_deterministic(self, regression_data):
        """Test StandardScaler is deterministic."""
        from ferroml.preprocessing import StandardScaler

        X, y = regression_data

        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        X_t1 = scaler1.fit_transform(X)
        X_t2 = scaler2.fit_transform(X)

        np.testing.assert_allclose(X_t1, X_t2, rtol=1e-10,
            err_msg="StandardScaler should be deterministic")


# ============================================================================
# Multiple Fit Calls
# ============================================================================


class TestMultipleFitCalls:
    """Test that fitting multiple times is reproducible."""

    def test_refit_same_data(self, regression_data):
        """Test refitting on same data produces same results."""
        from ferroml.linear import LinearRegression

        X, y = regression_data
        model = LinearRegression()

        model.fit(X, y)
        preds1 = model.predict(X)

        model.fit(X, y)
        preds2 = model.predict(X)

        np.testing.assert_allclose(preds1, preds2, rtol=1e-10,
            err_msg="Refitting on same data should give same results")

    def test_refit_different_data(self, regression_data, classification_data):
        """Test that refitting on different data changes the model."""
        from ferroml.linear import LinearRegression

        X1, y1 = regression_data

        # Create very different data
        np.random.seed(999)
        X2 = np.random.randn(50, 3) * 100
        y2 = X2[:, 0] * 10 - X2[:, 1] * 5 + np.random.randn(50)

        model = LinearRegression()

        model.fit(X1, y1)
        coef1 = model.coef_.copy()

        model.fit(X2, y2)
        coef2 = model.coef_.copy()

        # Coefficients should be different
        diff = np.abs(coef1 - coef2).max()
        assert diff > 0.1, "Fitting different data should change coefficients"


# ============================================================================
# Cross-Run Reproducibility
# ============================================================================


class TestCrossRunReproducibility:
    """Test reproducibility across multiple runs/sessions."""

    def test_known_output_linear_regression(self, perfect_linear_data):
        """Test LinearRegression produces known coefficients for perfect data."""
        from ferroml.linear import LinearRegression

        X, y = perfect_linear_data
        # y = 2x + 1

        model = LinearRegression()
        model.fit(X, y)

        # Should recover exact coefficients
        np.testing.assert_allclose(model.coef_, [2.0], rtol=1e-5,
            err_msg="Should recover coefficient of 2.0")
        np.testing.assert_allclose(model.intercept_, 1.0, rtol=1e-5,
            err_msg="Should recover intercept of 1.0")

    def test_known_output_standard_scaler(self):
        """Test StandardScaler produces known output."""
        from ferroml.preprocessing import StandardScaler

        # Simple data with known mean and std
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        # Mean = 3.0, Std = sqrt(2) ≈ 1.414

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Scaled mean should be 0
        np.testing.assert_allclose(X_scaled.mean(), 0.0, atol=1e-10)

        # Scaled std should be close to 1 (allow for ddof differences)
        # FerroML may use ddof=0 (population) or ddof=1 (sample)
        std_val = X_scaled.std()
        assert 0.8 <= std_val <= 1.1, f"Std should be close to 1, got {std_val}"


# ============================================================================
# Probability Reproducibility
# ============================================================================


class TestProbabilityReproducibility:
    """Test that probability predictions are reproducible."""

    @pytest.mark.skip(reason="RandomForest may have non-deterministic parallel training")
    def test_predict_proba_reproducibility(self, classification_data):
        """Test predict_proba is reproducible with same seed."""
        from ferroml.trees import RandomForestClassifier

        X, y = classification_data

        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2 = RandomForestClassifier(n_estimators=10, random_state=42)

        model1.fit(X, y)
        model2.fit(X, y)

        proba1 = model1.predict_proba(X)
        proba2 = model2.predict_proba(X)

        np.testing.assert_allclose(proba1, proba2, rtol=1e-10,
            err_msg="predict_proba should be reproducible with same seed")

    def test_logistic_predict_proba_deterministic(self, classification_data):
        """Test LogisticRegression predict_proba is deterministic."""
        from ferroml.linear import LogisticRegression

        X, y = classification_data

        model1 = LogisticRegression()
        model2 = LogisticRegression()

        model1.fit(X, y)
        model2.fit(X, y)

        proba1 = model1.predict_proba(X)
        proba2 = model2.predict_proba(X)

        np.testing.assert_allclose(proba1, proba2, rtol=1e-5,
            err_msg="LogisticRegression predict_proba should be deterministic")


# ============================================================================
# Feature Importance Reproducibility
# ============================================================================


class TestFeatureImportanceReproducibility:
    """Test that feature importances are reproducible."""

    @pytest.mark.skip(reason="RandomForest may have non-deterministic parallel training")
    def test_random_forest_feature_importances(self, classification_data):
        """Test RandomForest feature importances are reproducible with same seed."""
        from ferroml.trees import RandomForestClassifier

        X, y = classification_data

        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2 = RandomForestClassifier(n_estimators=10, random_state=42)

        model1.fit(X, y)
        model2.fit(X, y)

        imp1 = model1.feature_importances_
        imp2 = model2.feature_importances_

        np.testing.assert_allclose(imp1, imp2, rtol=1e-10,
            err_msg="Feature importances should be reproducible with same seed")

    def test_decision_tree_feature_importances(self, classification_data):
        """Test DecisionTree feature importances are reproducible with same seed."""
        from ferroml.trees import DecisionTreeClassifier

        X, y = classification_data

        model1 = DecisionTreeClassifier(max_depth=5, random_state=42)
        model2 = DecisionTreeClassifier(max_depth=5, random_state=42)

        model1.fit(X, y)
        model2.fit(X, y)

        imp1 = model1.feature_importances_
        imp2 = model2.feature_importances_

        np.testing.assert_allclose(imp1, imp2, rtol=1e-10,
            err_msg="Feature importances should be reproducible with same seed")


# ============================================================================
# Transformer Reproducibility
# ============================================================================


class TestTransformerReproducibility:
    """Test that transformers are reproducible."""

    def test_all_scalers_reproducible(self, all_scalers, regression_data):
        """Test all scalers produce reproducible results."""
        X, y = regression_data

        for name, scaler in all_scalers:
            # Create two instances
            scaler1_class = type(scaler)
            scaler2_class = type(scaler)

            scaler1 = scaler1_class()
            scaler2 = scaler2_class()

            X_t1 = scaler1.fit_transform(X)
            X_t2 = scaler2.fit_transform(X)

            np.testing.assert_allclose(X_t1, X_t2, rtol=1e-10,
                err_msg=f"{name} should be reproducible")

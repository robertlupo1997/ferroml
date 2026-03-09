"""
System-level end-to-end tests for the AutoML pipeline.

These tests exercise the full AutoML Python API as a real user would:
configuration, fitting, leaderboard inspection, ensemble access, prediction,
and error handling. Each test runs a complete AutoML search, so they are
marked as slow.
"""

import numpy as np
import pytest

from ferroml.automl import AutoMLConfig, AutoML


# ============================================================================
# Data generation helpers
# ============================================================================


def make_classification_data(n_samples=150, n_features=4, n_classes=3, seed=42):
    """Well-separated clusters for reliable classification."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    per_class = n_samples // n_classes
    for c in range(n_classes):
        start = c * per_class
        end = start + per_class
        y[start:end] = float(c)
        X[start:end] = rng.randn(per_class, n_features) + c * 2.0
    return X, y


def make_regression_data(n_samples=200, n_features=5, seed=42):
    """Linear relationship with moderate noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = 2.0 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + rng.randn(n_samples) * 0.5
    return X, y


# ============================================================================
# Shared config helpers
# ============================================================================


def _clf_config(seed=42):
    return AutoMLConfig(
        task="classification",
        metric="accuracy",
        time_budget_seconds=15,
        cv_folds=3,
        seed=seed,
    )


def _reg_config(seed=42):
    return AutoMLConfig(
        task="regression",
        metric="r2",
        time_budget_seconds=15,
        cv_folds=3,
        seed=seed,
    )


# ============================================================================
# End-to-end workflow tests
# ============================================================================


@pytest.mark.slow
class TestClassificationEndToEnd:
    """Classification pipeline from config to prediction."""

    def test_classification_end_to_end(self):
        """Fit on multiclass data, predict, verify shape and valid labels."""
        X, y = make_classification_data()
        config = _clf_config()
        automl = AutoML(config)
        result = automl.fit(X, y)

        assert result.is_successful()

        # Predict on a held-out slice
        X_test = X[:20]
        preds = result.predict(X, y, X_test)
        assert preds.shape == (20,)
        unique_preds = set(np.unique(preds))
        valid_labels = {0.0, 1.0, 2.0}
        assert unique_preds.issubset(valid_labels), (
            f"Predictions contain invalid labels: {unique_preds - valid_labels}"
        )

    def test_classification_score_quality(self):
        """Best model should achieve > 0.65 on well-separated data."""
        X, y = make_classification_data()
        result = AutoML(_clf_config()).fit(X, y)
        best = result.best_model()
        assert best is not None
        assert best.cv_score > 0.65, (
            f"Best score {best.cv_score:.4f} too low on easy data"
        )


@pytest.mark.slow
class TestRegressionEndToEnd:
    """Regression pipeline from config to prediction."""

    def test_regression_end_to_end(self):
        """Fit on regression data, predict, verify finite values."""
        X, y = make_regression_data()
        config = _reg_config()
        result = AutoML(config).fit(X, y)

        assert result.is_successful()

        preds = result.predict(X, y, X[:30])
        assert preds.shape == (30,)
        assert np.all(np.isfinite(preds)), "Predictions contain non-finite values"

    def test_regression_score_quality(self):
        """R-squared > 0.15 on data with clear linear signal."""
        X, y = make_regression_data()
        result = AutoML(_reg_config()).fit(X, y)
        best = result.best_model()
        assert best is not None
        assert best.cv_score > 0.15, (
            f"Best R2 {best.cv_score:.4f} too low on linear data"
        )


# ============================================================================
# Leaderboard and result inspection tests
# ============================================================================


@pytest.mark.slow
class TestLeaderboardInspection:
    """Verify leaderboard structure and ordering."""

    def test_leaderboard_fields(self):
        """Leaderboard entries expose rank, algorithm, cv_score, cv_std."""
        X, y = make_classification_data()
        result = AutoML(_clf_config()).fit(X, y)
        lb = result.leaderboard
        assert len(lb) > 0, "Leaderboard is empty"

        entry = lb[0]
        assert isinstance(entry.rank, int)
        assert isinstance(entry.algorithm, str) and len(entry.algorithm) > 0
        assert isinstance(entry.cv_score, float)
        assert isinstance(entry.cv_std, float)

    def test_leaderboard_sorted_by_rank(self):
        """Leaderboard entries are sorted by ascending rank."""
        X, y = make_classification_data()
        result = AutoML(_clf_config()).fit(X, y)
        ranks = [e.rank for e in result.leaderboard]
        assert ranks == sorted(ranks), f"Ranks not sorted: {ranks}"

    def test_ci_bounds(self):
        """ci_lower <= cv_score <= ci_upper for every entry."""
        X, y = make_classification_data()
        result = AutoML(_clf_config()).fit(X, y)
        for entry in result.leaderboard:
            assert entry.ci_lower <= entry.cv_score + 1e-12, (
                f"{entry.algorithm}: ci_lower {entry.ci_lower} > cv_score {entry.cv_score}"
            )
            assert entry.cv_score <= entry.ci_upper + 1e-12, (
                f"{entry.algorithm}: cv_score {entry.cv_score} > ci_upper {entry.ci_upper}"
            )


# ============================================================================
# Ensemble tests
# ============================================================================


@pytest.mark.slow
class TestEnsembleAccess:
    """Inspect ensemble construction results."""

    def test_ensemble_members_weights(self):
        """If ensemble exists, members have positive weights summing to ~1."""
        X, y = make_classification_data()
        result = AutoML(_clf_config()).fit(X, y)
        ens = result.ensemble
        if ens is not None:
            members = ens.members
            assert len(members) > 0
            for m in members:
                assert m.weight > 0, f"Non-positive weight for {m.algorithm}"
            total_weight = sum(m.weight for m in members)
            assert abs(total_weight - 1.0) < 0.05, (
                f"Weights sum to {total_weight}, expected ~1.0"
            )

    def test_ensemble_score_and_improvement(self):
        """Ensemble score and improvement are accessible."""
        X, y = make_classification_data()
        result = AutoML(_clf_config()).fit(X, y)
        score = result.ensemble_score()
        improvement = result.ensemble_improvement()
        # Both are None (no ensemble) or both are floats
        if score is not None:
            assert isinstance(score, float)
            assert isinstance(improvement, float)


# ============================================================================
# Result metadata and summary tests
# ============================================================================


@pytest.mark.slow
class TestResultMetadata:
    """Verify result metadata attributes and summary."""

    def test_summary_output(self):
        """summary() returns a non-empty string."""
        X, y = make_classification_data()
        result = AutoML(_clf_config()).fit(X, y)
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 20, "Summary is unexpectedly short"

    def test_predict_shape_matches(self):
        """Prediction shape matches number of test samples."""
        X, y = make_regression_data()
        result = AutoML(_reg_config()).fit(X, y)
        for n_test in [1, 10, 50]:
            preds = result.predict(X, y, X[:n_test])
            assert preds.shape == (n_test,), (
                f"Expected shape ({n_test},), got {preds.shape}"
            )

    def test_result_metadata_fields(self):
        """total_time_seconds > 0, n_successful_trials > 0, is_successful."""
        X, y = make_classification_data()
        result = AutoML(_clf_config()).fit(X, y)
        assert result.total_time_seconds > 0
        assert result.n_successful_trials > 0
        assert result.is_successful()
        assert result.task == "classification"
        assert result.metric_name == "accuracy"

    def test_top_features(self):
        """top_features(k) returns a list of tuples with non-negative importances."""
        X, y = make_classification_data()
        result = AutoML(_clf_config()).fit(X, y)
        feats = result.top_features(3)
        if feats is not None:
            assert len(feats) <= 3
            for item in feats:
                # (feature_name, importance, ci_lower, ci_upper)
                assert len(item) == 4
                name, imp, ci_l, ci_u = item
                assert isinstance(name, str)
                assert imp >= 0.0, f"Negative importance for {name}"

    def test_competitive_models(self):
        """competitive_models() is non-empty and includes the best."""
        X, y = make_classification_data()
        result = AutoML(_clf_config()).fit(X, y)
        comp = result.competitive_models()
        assert len(comp) >= 1, "No competitive models"
        best = result.best_model()
        if best is not None:
            comp_algos = [m.algorithm for m in comp]
            assert best.algorithm in comp_algos, (
                f"Best model {best.algorithm} not in competitive set"
            )


# ============================================================================
# Reproducibility test
# ============================================================================


@pytest.mark.slow
class TestReproducibility:
    """Same seed should yield identical results."""

    def test_reproducibility(self):
        """Two runs with the same seed produce the same best scores."""
        X, y = make_classification_data()
        r1 = AutoML(_clf_config(seed=123)).fit(X, y)
        r2 = AutoML(_clf_config(seed=123)).fit(X, y)

        best1 = r1.best_model()
        best2 = r2.best_model()
        assert best1 is not None and best2 is not None
        assert best1.algorithm == best2.algorithm, (
            f"Different best algorithms: {best1.algorithm} vs {best2.algorithm}"
        )
        assert abs(best1.cv_score - best2.cv_score) < 1e-10, (
            f"Scores differ: {best1.cv_score} vs {best2.cv_score}"
        )


# ============================================================================
# Error handling tests
# ============================================================================


@pytest.mark.slow
class TestErrorHandling:
    """Validate that invalid inputs raise appropriate errors."""

    def test_error_empty_array(self):
        """Empty X should raise an error."""
        X = np.empty((0, 4))
        y = np.empty(0)
        with pytest.raises(Exception):
            AutoML(_clf_config()).fit(X, y)

    def test_error_mismatched_shapes(self):
        """X.shape[0] != y.shape[0] should raise an error."""
        X = np.random.randn(100, 4)
        y = np.random.randn(50)
        with pytest.raises(Exception):
            AutoML(_clf_config()).fit(X, y)

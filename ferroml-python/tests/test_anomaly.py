"""Tests for anomaly detection: IsolationForest and LocalOutlierFactor."""

import numpy as np
import pytest

from ferroml.anomaly import IsolationForest, LocalOutlierFactor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_data(n_inliers=90, n_outliers=10, seed=42):
    """Create a cluster of inliers near origin + outliers far away."""
    rng = np.random.RandomState(seed)
    inliers = rng.randn(n_inliers, 2) * 0.5
    outliers = rng.randn(n_outliers, 2) * 0.5 + 20.0
    return np.vstack([inliers, outliers])


# ===========================================================================
# IsolationForest tests
# ===========================================================================

class TestIsolationForest:
    def test_fit_predict(self):
        X = make_test_data()
        model = IsolationForest(random_state=42)
        model.fit(X)
        preds = model.predict(X)
        assert preds.shape == (100,)
        assert set(np.unique(preds)).issubset({1, -1})

    def test_fit_predict_shortcut(self):
        X = make_test_data()
        model = IsolationForest(random_state=42)
        preds = model.fit_predict(X)
        assert preds.shape == (100,)

    def test_detects_outliers(self):
        X = make_test_data()
        model = IsolationForest(contamination="0.1", random_state=42)
        model.fit(X)
        preds = model.predict(X)
        outlier_count = np.sum(preds[90:] == -1)
        assert outlier_count >= 7, f"Expected most outliers detected, got {outlier_count}/10"

    def test_score_samples_range(self):
        X = make_test_data()
        model = IsolationForest(random_state=42)
        model.fit(X)
        scores = model.score_samples(X)
        assert np.all(scores >= -1.0)
        assert np.all(scores <= 0.0)

    def test_outliers_have_lower_scores(self):
        X = make_test_data()
        model = IsolationForest(n_estimators=200, random_state=42)
        model.fit(X)
        scores = model.score_samples(X)
        assert np.mean(scores[90:]) < np.mean(scores[:90])

    def test_decision_function(self):
        X = make_test_data()
        model = IsolationForest(random_state=42)
        model.fit(X)
        scores = model.score_samples(X)
        decision = model.decision_function(X)
        offset = model.offset_
        np.testing.assert_allclose(decision, scores - offset, atol=1e-10)

    def test_predict_matches_decision(self):
        X = make_test_data()
        model = IsolationForest(random_state=42)
        model.fit(X)
        decision = model.decision_function(X)
        preds = model.predict(X)
        expected = np.where(decision >= 0, 1, -1)
        np.testing.assert_array_equal(preds, expected)

    def test_reproducibility(self):
        X = make_test_data()
        m1 = IsolationForest(random_state=123)
        m2 = IsolationForest(random_state=123)
        m1.fit(X)
        m2.fit(X)
        np.testing.assert_array_equal(
            m1.score_samples(X), m2.score_samples(X)
        )

    def test_contamination_auto_offset(self):
        X = make_test_data()
        model = IsolationForest(random_state=42)
        model.fit(X)
        assert model.offset_ == -0.5

    def test_contamination_proportion(self):
        X = make_test_data()
        model = IsolationForest(contamination="0.1", random_state=42)
        model.fit(X)
        assert model.offset_ != -0.5

    def test_max_samples_count(self):
        X = make_test_data()
        model = IsolationForest(max_samples="50", random_state=42)
        model.fit(X)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_max_samples_fraction(self):
        X = make_test_data()
        model = IsolationForest(max_samples="0.5", random_state=42)
        model.fit(X)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_predict_new_data(self):
        X = make_test_data()
        model = IsolationForest(n_estimators=200, random_state=42)
        model.fit(X)
        x_in = np.array([[0.0, 0.0]])
        x_out = np.array([[100.0, 100.0]])
        assert model.score_samples(x_out)[0] < model.score_samples(x_in)[0]

    def test_repr(self):
        model = IsolationForest()
        assert "IsolationForest" in repr(model)

    def test_not_fitted_error(self):
        model = IsolationForest()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 2)))


# ===========================================================================
# LocalOutlierFactor tests
# ===========================================================================

class TestLocalOutlierFactor:
    def test_fit_predict(self):
        X = make_test_data()
        model = LocalOutlierFactor(n_neighbors=20)
        preds = model.fit_predict(X)
        assert preds.shape == (100,)
        assert set(np.unique(preds)).issubset({1, -1})

    def test_detects_outliers(self):
        X = make_test_data()
        model = LocalOutlierFactor(n_neighbors=20, contamination="0.1")
        preds = model.fit_predict(X)
        outlier_count = np.sum(preds[90:] == -1)
        assert outlier_count >= 7, f"Expected most outliers detected, got {outlier_count}/10"

    def test_negative_outlier_factor(self):
        X = make_test_data()
        model = LocalOutlierFactor(n_neighbors=20)
        model.fit(X)
        nof = model.negative_outlier_factor_
        assert nof is not None
        assert nof.shape == (100,)
        # Outliers should have more negative NOF
        assert np.mean(nof[90:]) < np.mean(nof[:90])

    def test_novelty_mode_predict(self):
        X = make_test_data()
        model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="0.1")
        model.fit(X)
        x_in = np.array([[0.0, 0.0]])
        pred = model.predict(x_in)
        assert pred[0] == 1

        x_out = np.array([[100.0, 100.0]])
        pred_out = model.predict(x_out)
        assert pred_out[0] == -1

    def test_novelty_false_rejects_predict(self):
        X = make_test_data()
        model = LocalOutlierFactor(n_neighbors=20, novelty=False)
        model.fit(X)
        with pytest.raises(RuntimeError):
            model.predict(np.array([[0.0, 0.0]]))

    def test_novelty_score_samples(self):
        X = make_test_data()
        model = LocalOutlierFactor(n_neighbors=20, novelty=True)
        model.fit(X)
        x_in = np.array([[0.0, 0.0]])
        x_out = np.array([[100.0, 100.0]])
        assert model.score_samples(x_out)[0] < model.score_samples(x_in)[0]

    def test_decision_function(self):
        X = make_test_data()
        model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="0.1")
        model.fit(X)
        x_new = np.array([[0.0, 0.0], [100.0, 100.0]])
        scores = model.score_samples(x_new)
        decision = model.decision_function(x_new)
        offset = model.offset_
        np.testing.assert_allclose(decision, scores - offset, atol=1e-10)

    def test_deterministic(self):
        X = make_test_data()
        m1 = LocalOutlierFactor(n_neighbors=20)
        m2 = LocalOutlierFactor(n_neighbors=20)
        m1.fit(X)
        m2.fit(X)
        np.testing.assert_array_equal(
            m1.negative_outlier_factor_, m2.negative_outlier_factor_
        )

    def test_contamination_auto(self):
        X = make_test_data()
        model = LocalOutlierFactor(n_neighbors=20)
        model.fit(X)
        assert model.offset_ == -1.5

    def test_contamination_proportion(self):
        X = make_test_data()
        model = LocalOutlierFactor(n_neighbors=20, contamination="0.1")
        model.fit(X)
        assert model.offset_ != -1.5

    def test_different_metrics(self):
        X = make_test_data()
        m_euc = LocalOutlierFactor(n_neighbors=20, metric="euclidean")
        m_man = LocalOutlierFactor(n_neighbors=20, metric="manhattan")
        m_euc.fit(X)
        m_man.fit(X)
        assert not np.array_equal(
            m_euc.negative_outlier_factor_, m_man.negative_outlier_factor_
        )

    def test_brute_algorithm(self):
        X = make_test_data()
        model = LocalOutlierFactor(n_neighbors=20, algorithm="brute")
        preds = model.fit_predict(X)
        assert preds.shape == (100,)

    def test_repr(self):
        model = LocalOutlierFactor()
        assert "LocalOutlierFactor" in repr(model)

    def test_not_fitted_score(self):
        model = LocalOutlierFactor(novelty=True)
        with pytest.raises(RuntimeError):
            model.score_samples(np.zeros((5, 2)))

    def test_k_larger_than_n(self):
        X = np.random.randn(5, 2)
        model = LocalOutlierFactor(n_neighbors=20)
        preds = model.fit_predict(X)
        assert preds.shape == (5,)

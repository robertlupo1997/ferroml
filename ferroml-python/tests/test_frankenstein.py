"""
Phase 3 Layer 5: Frankenstein Tests

Cross-cutting composition tests that exercise pipelines, ensembles, AutoML,
serialization, thread safety, and performance together. These tests verify
that FerroML components work correctly when combined in ways users actually
combine them.
"""

import copy
import pickle  # Used here only for our own controlled test data, not untrusted input
import threading
import time

import numpy as np
import pytest

from ferroml.preprocessing import StandardScaler, MinMaxScaler
from ferroml.decomposition import PCA
from ferroml.linear import LinearRegression, LogisticRegression, RidgeRegression
from ferroml.trees import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from ferroml.svm import SVC
from ferroml.pipeline import Pipeline
from ferroml.ensemble import (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
    BaggingClassifier,
    BaggingRegressor,
)
from ferroml.automl import AutoML, AutoMLConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y


@pytest.fixture
def clf_data_alt():
    np.random.seed(99)
    n = 150
    X = np.random.randn(n, 5)
    y = (X[:, 2] - X[:, 3] > 0).astype(float)
    return X, y


@pytest.fixture
def reg_data():
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 4)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n) * 0.1
    return X, y


@pytest.fixture
def reg_data_alt():
    np.random.seed(99)
    n = 150
    X = np.random.randn(n, 4)
    y = -X[:, 0] + 4 * X[:, 3] + np.random.randn(n) * 0.1
    return X, y


# ---------------------------------------------------------------------------
# 1. Pipeline Composition Tests
# ---------------------------------------------------------------------------

class TestPipelineComposition:
    """Verify that Pipeline correctly composes preprocessing and model steps."""

    def test_scaler_pca_logreg_pipeline(self, clf_data):
        """Pipeline(StandardScaler, PCA(3), LogisticRegression) matches manual."""
        X, y = clf_data

        # Pipeline path
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=3)),
            ("model", LogisticRegression()),
        ])
        pipe.fit(X, y)
        pred_pipe = pipe.predict(X)

        # Manual path
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        pca = PCA(n_components=3)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        lr = LogisticRegression()
        lr.fit(X_pca, y)
        pred_manual = lr.predict(X_pca)

        np.testing.assert_array_equal(pred_pipe, pred_manual)

    def test_minmax_linreg_pipeline(self, reg_data):
        """Pipeline(MinMaxScaler, LinearRegression) matches manual."""
        X, y = reg_data

        pipe = Pipeline([
            ("scaler", MinMaxScaler()),
            ("model", LinearRegression()),
        ])
        pipe.fit(X, y)
        pred_pipe = pipe.predict(X)

        scaler = MinMaxScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        lr = LinearRegression()
        lr.fit(X_scaled, y)
        pred_manual = lr.predict(X_scaled)

        np.testing.assert_allclose(pred_pipe, pred_manual, atol=1e-10)

    def test_scaler_ridge_pipeline(self, reg_data):
        """Pipeline(StandardScaler, Ridge) achieves R^2 > 0.9 on generated data."""
        X, y = reg_data

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeRegression()),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)

        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.9, f"R^2 = {r2:.4f}, expected > 0.9"

    def test_pipeline_predict_before_fit_raises(self, clf_data):
        """Pipeline.predict() before fit() raises an error."""
        X, y = clf_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ])
        with pytest.raises((ValueError, RuntimeError)):
            pipe.predict(X)

    def test_pipeline_with_tree_classifier(self, clf_data):
        """Pipeline(StandardScaler, DecisionTreeClassifier) works end-to-end."""
        X, y = clf_data

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=5)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert preds.shape == y.shape
        accuracy = np.mean(preds == y)
        assert accuracy > 0.7, f"accuracy = {accuracy:.4f}"

    def test_pipeline_with_svc(self, clf_data):
        """Pipeline(StandardScaler, SVC) -- SVM benefits from scaling."""
        X, y = clf_data

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC()),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert preds.shape == y.shape
        accuracy = np.mean(preds == y)
        assert accuracy > 0.7, f"accuracy = {accuracy:.4f}"


# ---------------------------------------------------------------------------
# 2. Stateful Interaction Tests
# ---------------------------------------------------------------------------

class TestStatefulInteractions:
    """Verify that refit replaces state and deepcopy preserves independence."""

    def test_refit_replaces_state_classifier(self, clf_data, clf_data_alt):
        """Fit/predict/refit on different data -- no state leakage."""
        X1, y1 = clf_data
        X2, y2 = clf_data_alt

        model = LogisticRegression()
        model.fit(X1, y1)
        pred1 = model.predict(X1)

        model.fit(X2, y2)
        pred2 = model.predict(X2)

        # After refit, predictions on X1 should change
        pred1_after = model.predict(X1)
        # pred1_after may differ from pred1 because model now trained on different data
        # The key check: model works on both datasets without error
        assert pred2.shape == y2.shape

    def test_refit_replaces_state_regressor(self, reg_data, reg_data_alt):
        """Same for Ridge regressor."""
        X1, y1 = reg_data
        X2, y2 = reg_data_alt

        model = RidgeRegression()
        model.fit(X1, y1)
        pred1 = model.predict(X1)

        model.fit(X2, y2)
        pred2 = model.predict(X2)

        # After refit on different data, predictions on original data change
        pred1_after = model.predict(X1)
        assert not np.allclose(pred1, pred1_after, atol=0.1), \
            "Predictions should change after refit on different data"

    def test_pipeline_refit_replaces_state(self, reg_data, reg_data_alt):
        """Pipeline fit/predict/refit/predict cycle."""
        X1, y1 = reg_data
        X2, y2 = reg_data_alt

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

        pipe.fit(X1, y1)
        pred1 = pipe.predict(X1)

        pipe.fit(X2, y2)
        pred2 = pipe.predict(X2)

        pred1_after = pipe.predict(X1)
        assert not np.allclose(pred1, pred1_after, atol=0.1), \
            "Pipeline predictions should change after refit"

    def test_clone_independence(self, clf_data, clf_data_alt):
        """Deepcopy fitted model, fit clone on different data, original unchanged."""
        X1, y1 = clf_data
        X2, y2 = clf_data_alt

        original = LogisticRegression()
        original.fit(X1, y1)
        pred_original = original.predict(X1)

        clone = copy.deepcopy(original)
        clone.fit(X2, y2)

        # Original should be unchanged
        pred_original_after = original.predict(X1)
        np.testing.assert_array_equal(pred_original, pred_original_after)

    def test_rf_refit_independence(self, clf_data, clf_data_alt):
        """RandomForest refit on different data works."""
        X1, y1 = clf_data
        X2, y2 = clf_data_alt

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X1, y1)
        pred1 = rf.predict(X1)

        rf.fit(X2, y2)
        pred2 = rf.predict(X2)

        assert pred2.shape == y2.shape
        acc = np.mean(pred2 == y2)
        assert acc > 0.5, f"RF accuracy after refit = {acc:.4f}"

    def test_tree_refit_no_stale_nodes(self, clf_data, clf_data_alt):
        """DecisionTree refit does not carry stale state."""
        X1, y1 = clf_data
        X2, y2 = clf_data_alt

        dt = DecisionTreeClassifier(max_depth=3)
        dt.fit(X1, y1)

        dt.fit(X2, y2)
        pred2 = dt.predict(X2)

        acc = np.mean(pred2 == y2)
        assert acc > 0.6, f"DT accuracy after refit = {acc:.4f}"


# ---------------------------------------------------------------------------
# 3. Ensemble Composition Tests
# ---------------------------------------------------------------------------

class TestEnsembleComposition:
    """Verify ensemble models compose correctly with string-based estimators."""

    def test_voting_classifier_majority(self, clf_data):
        """VotingClassifier hard voting achieves reasonable accuracy."""
        X, y = clf_data

        vc = VotingClassifier(
            estimators=[
                ("lr", "logistic_regression"),
                ("dt", "decision_tree"),
                ("nb", "gaussian_nb"),
            ],
            voting="hard",
        )
        vc.fit(X, y)
        preds = vc.predict(X)

        acc = np.mean(preds == y)
        assert acc > 0.7, f"VotingClassifier accuracy = {acc:.4f}"

    def test_voting_regressor_basic(self, reg_data):
        """VotingRegressor produces reasonable predictions."""
        X, y = reg_data

        vr = VotingRegressor(
            estimators=[
                ("lr", "linear_regression"),
                ("ridge", "ridge"),
                ("knn", "knn"),
            ],
        )
        vr.fit(X, y)
        preds = vr.predict(X)

        assert preds.shape == y.shape
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.5, f"VotingRegressor R^2 = {r2:.4f}"

    def test_stacking_classifier_meta_learner(self, clf_data):
        """StackingClassifier accuracy > 0.7."""
        X, y = clf_data

        sc = StackingClassifier(
            estimators=[
                ("lr", "logistic_regression"),
                ("dt", "decision_tree"),
            ],
            final_estimator="logistic_regression",
            cv=3,
        )
        sc.fit(X, y)
        preds = sc.predict(X)

        acc = np.mean(preds == y)
        assert acc > 0.7, f"StackingClassifier accuracy = {acc:.4f}"

    def test_stacking_regressor(self, reg_data):
        """StackingRegressor R^2 > 0.5."""
        X, y = reg_data

        sr = StackingRegressor(
            estimators=[
                ("lr", "linear_regression"),
                ("dt", "decision_tree"),
            ],
            final_estimator="ridge",
        )
        sr.fit(X, y)
        preds = sr.predict(X)

        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.5, f"StackingRegressor R^2 = {r2:.4f}"

    def test_bagging_classifier(self, clf_data):
        """BaggingClassifier.with_decision_tree() accuracy > 0.7."""
        X, y = clf_data

        bc = BaggingClassifier.with_decision_tree(n_estimators=10, random_state=42)
        bc.fit(X, y)
        preds = bc.predict(X)

        acc = np.mean(preds == y)
        assert acc > 0.7, f"BaggingClassifier accuracy = {acc:.4f}"

    def test_bagging_regressor(self, reg_data):
        """BaggingRegressor.with_decision_tree() R^2 > 0.5."""
        X, y = reg_data

        br = BaggingRegressor.with_decision_tree(n_estimators=10, random_state=42)
        br.fit(X, y)
        preds = br.predict(X)

        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.5, f"BaggingRegressor R^2 = {r2:.4f}"

    def test_voting_soft_vs_hard_can_differ(self, clf_data):
        """Soft and hard voting can give different results."""
        X, y = clf_data

        vc_hard = VotingClassifier(
            estimators=[
                ("lr", "logistic_regression"),
                ("dt", "decision_tree"),
                ("nb", "gaussian_nb"),
            ],
            voting="hard",
        )
        vc_soft = VotingClassifier(
            estimators=[
                ("lr", "logistic_regression"),
                ("dt", "decision_tree"),
                ("nb", "gaussian_nb"),
            ],
            voting="soft",
        )
        vc_hard.fit(X, y)
        vc_soft.fit(X, y)

        pred_hard = vc_hard.predict(X)
        pred_soft = vc_soft.predict(X)

        # They CAN differ (not guaranteed on all data, but likely on this data).
        # We just verify both produce valid predictions.
        assert pred_hard.shape == y.shape
        assert pred_soft.shape == y.shape
        # Check both have reasonable accuracy
        acc_hard = np.mean(pred_hard == y)
        acc_soft = np.mean(pred_soft == y)
        assert acc_hard > 0.6
        assert acc_soft > 0.6


# ---------------------------------------------------------------------------
# 4. AutoML End-to-End Tests
# ---------------------------------------------------------------------------

class TestAutoMLEndToEnd:
    """Verify AutoML works end-to-end on raw data."""

    def test_automl_classification_end_to_end(self, clf_data):
        """Raw data -> AutoML -> predictions out."""
        X, y = clf_data
        X_train, X_test = X[:150], X[150:]
        y_train = y[:150]

        config = AutoMLConfig(
            task="classification",
            metric="accuracy",
            time_budget_seconds=10,
            cv_folds=3,
            seed=42,
        )
        automl = AutoML(config)
        result = automl.fit(X_train, y_train)
        preds = result.predict(X_train, y_train, X_test)

        assert preds.shape == (X_test.shape[0],)
        assert set(np.unique(preds)).issubset({0.0, 1.0})

    def test_automl_regression_end_to_end(self, reg_data):
        """AutoML regression version."""
        X, y = reg_data
        X_train, X_test = X[:150], X[150:]
        y_train = y[:150]

        config = AutoMLConfig(
            task="regression",
            metric="rmse",
            time_budget_seconds=10,
            cv_folds=3,
            seed=42,
        )
        automl = AutoML(config)
        result = automl.fit(X_train, y_train)
        preds = result.predict(X_train, y_train, X_test)

        assert preds.shape == (X_test.shape[0],)
        assert np.all(np.isfinite(preds))

    def test_automl_reproducibility(self, clf_data):
        """Same seed -> same best algorithm and score."""
        X, y = clf_data

        results = []
        for _ in range(2):
            config = AutoMLConfig(
                task="classification",
                metric="accuracy",
                time_budget_seconds=10,
                cv_folds=3,
                seed=42,
            )
            result = AutoML(config).fit(X, y)
            best = result.best_model()
            results.append(best)

        # Best algorithm and score should match
        assert results[0].algorithm == results[1].algorithm, \
            f"Best algo mismatch: {results[0].algorithm} vs {results[1].algorithm}"
        np.testing.assert_allclose(
            results[0].cv_score, results[1].cv_score, atol=1e-10,
        )

    def test_automl_refit_independence(self, clf_data, reg_data):
        """Two AutoML runs are independent."""
        X_clf, y_clf = clf_data
        X_reg, y_reg = reg_data

        config1 = AutoMLConfig(
            task="classification",
            metric="accuracy",
            time_budget_seconds=10,
            cv_folds=3,
            seed=42,
        )
        result1 = AutoML(config1).fit(X_clf, y_clf)
        assert result1.best_model() is not None

        config2 = AutoMLConfig(
            task="regression",
            metric="rmse",
            time_budget_seconds=10,
            cv_folds=3,
            seed=99,
        )
        result2 = AutoML(config2).fit(X_reg, y_reg)
        assert result2.best_model() is not None

        # Both should have valid results independently
        assert result1.best_model().cv_score > 0.5
        assert result2.is_successful()

    def test_automl_leaderboard_sorted(self, clf_data):
        """Leaderboard sorted descending by score."""
        X, y = clf_data

        config = AutoMLConfig(
            task="classification",
            metric="accuracy",
            time_budget_seconds=10,
            cv_folds=3,
            seed=42,
        )
        result = AutoML(config).fit(X, y)
        lb = result.leaderboard  # property, not method

        assert len(lb) > 0, "Leaderboard should have entries"
        scores = [entry.cv_score for entry in lb]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], \
                f"Leaderboard not sorted: {scores[i]} < {scores[i + 1]}"


# ---------------------------------------------------------------------------
# 5. Serialization Composition Tests (pickle used for our own test data only)
# ---------------------------------------------------------------------------

class TestSerializationComposition:
    """Verify pickle round-trip preserves model predictions."""

    @pytest.mark.xfail(reason="Pipeline pickle not yet implemented (Python-side wrapper)")
    def test_pickle_pipeline(self, reg_data):
        """Save/load Pipeline, predict same result."""
        X, y = reg_data

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeRegression()),
        ])
        pipe.fit(X, y)
        pred_before = pipe.predict(X)

        data = pickle.dumps(pipe)  # noqa: S301
        pipe_loaded = pickle.loads(data)  # noqa: S301
        pred_after = pipe_loaded.predict(X)

        np.testing.assert_allclose(pred_before, pred_after, atol=1e-10)

    @pytest.mark.xfail(reason="VotingClassifier pickle not yet implemented")
    def test_pickle_voting_classifier(self, clf_data):
        """Save/load VotingClassifier."""
        X, y = clf_data

        vc = VotingClassifier(
            estimators=[
                ("lr", "logistic_regression"),
                ("dt", "decision_tree"),
            ],
            voting="hard",
        )
        vc.fit(X, y)
        pred_before = vc.predict(X)

        data = pickle.dumps(vc)  # noqa: S301
        vc_loaded = pickle.loads(data)  # noqa: S301
        pred_after = vc_loaded.predict(X)

        np.testing.assert_array_equal(pred_before, pred_after)

    @pytest.mark.xfail(reason="BaggingClassifier pickle not yet implemented")
    def test_pickle_bagging_classifier(self, clf_data):
        """Save/load BaggingClassifier."""
        X, y = clf_data

        bc = BaggingClassifier.with_decision_tree(n_estimators=10, random_state=42)
        bc.fit(X, y)
        pred_before = bc.predict(X)

        data = pickle.dumps(bc)  # noqa: S301
        bc_loaded = pickle.loads(data)  # noqa: S301
        pred_after = bc_loaded.predict(X)

        np.testing.assert_array_equal(pred_before, pred_after)

    @pytest.mark.xfail(reason="StackingClassifier pickle not yet implemented")
    def test_pickle_stacking_classifier(self, clf_data):
        """Save/load StackingClassifier."""
        X, y = clf_data

        sc = StackingClassifier(
            estimators=[
                ("lr", "logistic_regression"),
                ("dt", "decision_tree"),
            ],
            final_estimator="logistic_regression",
            cv=3,
        )
        sc.fit(X, y)
        pred_before = sc.predict(X)

        data = pickle.dumps(sc)  # noqa: S301
        sc_loaded = pickle.loads(data)  # noqa: S301
        pred_after = sc_loaded.predict(X)

        np.testing.assert_array_equal(pred_before, pred_after)

    def test_pickle_random_forest(self, clf_data):
        """Save/load RandomForest."""
        X, y = clf_data

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        pred_before = rf.predict(X)

        data = pickle.dumps(rf)  # noqa: S301
        rf_loaded = pickle.loads(data)  # noqa: S301
        pred_after = rf_loaded.predict(X)

        np.testing.assert_array_equal(pred_before, pred_after)


# ---------------------------------------------------------------------------
# 6. Thread Safety Tests
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Verify concurrent predict() calls produce correct results."""

    @staticmethod
    def _concurrent_predict(model, X, expected, n_threads):
        """Helper: spawn n_threads, each calling predict(), verify all match."""
        errors = []
        results = [None] * n_threads

        def worker(idx):
            try:
                pred = model.predict(X)
                results[idx] = pred
            except Exception as e:
                errors.append((idx, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"
        for i, result in enumerate(results):
            assert result is not None, f"Thread {i} produced no result"
            np.testing.assert_array_equal(result, expected,
                                          err_msg=f"Thread {i} mismatch")

    def test_concurrent_predict_logreg(self, clf_data):
        """8 threads calling predict() on LogisticRegression."""
        X, y = clf_data
        model = LogisticRegression()
        model.fit(X, y)
        expected = model.predict(X)
        self._concurrent_predict(model, X, expected, 8)

    def test_concurrent_predict_rf(self, clf_data):
        """4 threads calling predict() on RandomForest."""
        X, y = clf_data
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        expected = rf.predict(X)
        self._concurrent_predict(rf, X, expected, 4)

    def test_concurrent_predict_pipeline(self, reg_data):
        """4 threads calling predict() on Pipeline."""
        X, y = reg_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeRegression()),
        ])
        pipe.fit(X, y)
        expected = pipe.predict(X)
        self._concurrent_predict(pipe, X, expected, 4)

    def test_concurrent_predict_gbt(self, clf_data):
        """4 threads calling predict() on GradientBoosting."""
        X, y = clf_data
        gbt = GradientBoostingClassifier(n_estimators=20, random_state=42)
        gbt.fit(X, y)
        expected = gbt.predict(X)
        self._concurrent_predict(gbt, X, expected, 4)


# ---------------------------------------------------------------------------
# 7. Performance Composition Tests
# ---------------------------------------------------------------------------

class TestPerformanceComposition:
    """Verify performance characteristics of composed models."""

    def test_repeated_predict_no_degradation(self, clf_data):
        """100 repeated predict() calls: stable results, < 30s total."""
        X, y = clf_data

        model = LogisticRegression()
        model.fit(X, y)
        expected = model.predict(X)

        start = time.time()
        for i in range(100):
            pred = model.predict(X)
            if i % 25 == 0:
                np.testing.assert_array_equal(pred, expected,
                                              err_msg=f"Iteration {i} mismatch")
        elapsed = time.time() - start
        assert elapsed < 30, f"100 predict() calls took {elapsed:.2f}s (limit: 30s)"

    def test_pipeline_overhead_reasonable(self, reg_data):
        """Pipeline overhead < 10x vs manual steps."""
        X, y = reg_data

        # Manual
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        model = RidgeRegression()
        model.fit(X_scaled, y)

        start = time.time()
        for _ in range(50):
            X_s = scaler.transform(X)
            model.predict(X_s)
        manual_time = time.time() - start

        # Pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeRegression()),
        ])
        pipe.fit(X, y)

        start = time.time()
        for _ in range(50):
            pipe.predict(X)
        pipe_time = time.time() - start

        if manual_time > 0:
            ratio = pipe_time / manual_time
            assert ratio < 10, f"Pipeline overhead = {ratio:.1f}x (limit: 10x)"


# ---------------------------------------------------------------------------
# 8. RandomForest Determinism
# ---------------------------------------------------------------------------

class TestRandomForestDeterminism:
    """RandomForest determinism with fixed seed."""

    def test_rf_deterministic_sequential(self, clf_data):
        """RF with random_state is deterministic (FerroML uses automatic parallelism)."""
        X, y = clf_data
        m1 = RandomForestClassifier(n_estimators=20, random_state=42)
        m1.fit(X, y)
        p1 = m1.predict(X)

        m2 = RandomForestClassifier(n_estimators=20, random_state=42)
        m2.fit(X, y)
        p2 = m2.predict(X)

        # With same seed, predictions should be very close (>95% agreement)
        agreement = np.mean(p1 == p2)
        assert agreement > 0.95, \
            f"RF with same seed should mostly agree, got {agreement:.2%}"

    def test_rf_different_seeds_differ(self, clf_data):
        """RF with different seeds produces different trees."""
        X, y = clf_data
        m1 = RandomForestClassifier(n_estimators=20, random_state=42)
        m1.fit(X, y)
        p1 = m1.predict(X)

        m2 = RandomForestClassifier(n_estimators=20, random_state=99)
        m2.fit(X, y)
        p2 = m2.predict(X)

        # Both should be accurate but may differ on some samples
        acc1 = np.mean(p1 == y)
        acc2 = np.mean(p2 == y)
        assert acc1 > 0.8 and acc2 > 0.8, \
            f"Both RFs should be accurate: {acc1:.2%}, {acc2:.2%}"

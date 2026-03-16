"""Tests for ferroml.model_selection (train_test_split, learning_curve, validation_curve)."""

import numpy as np
import pytest


# =============================================================================
# train_test_split
# =============================================================================

class TestTrainTestSplit:
    def test_shapes(self):
        from ferroml.model_selection import train_test_split
        X = np.random.RandomState(0).randn(100, 5)
        y = np.random.RandomState(0).randn(100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        assert X_train.shape[0] + X_test.shape[0] == 100
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20
        assert X_train.shape[1] == 5

    def test_reproducibility(self):
        from ferroml.model_selection import train_test_split
        X = np.random.RandomState(0).randn(100, 5)
        y = np.random.RandomState(0).randn(100)
        r1 = train_test_split(X, y, test_size=0.3, random_state=42)
        r2 = train_test_split(X, y, test_size=0.3, random_state=42)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)

    def test_no_overlap(self):
        from ferroml.model_selection import train_test_split
        X = np.arange(50).reshape(10, 5).astype(float)
        y = np.arange(10).astype(float)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # Check that no training row appears in the test set
        train_set = set(map(tuple, X_train.tolist()))
        test_set = set(map(tuple, X_test.tolist()))
        assert len(train_set & test_set) == 0

    def test_different_test_sizes(self):
        from ferroml.model_selection import train_test_split
        X = np.random.RandomState(0).randn(200, 3)
        y = np.random.RandomState(0).randn(200)
        for test_size in [0.1, 0.2, 0.5]:
            X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size, random_state=0)
            expected_test = int(round(200 * test_size))
            assert abs(X_test.shape[0] - expected_test) <= 1

    def test_no_shuffle(self):
        from ferroml.model_selection import train_test_split
        X = np.arange(50).reshape(10, 5).astype(float)
        y = np.arange(10).astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )
        # Without shuffle, last 3 rows should be test
        np.testing.assert_array_equal(y_test, np.array([7., 8., 9.]))
        np.testing.assert_array_equal(y_train, np.arange(7).astype(float))

    def test_stratified(self):
        from ferroml.model_selection import train_test_split
        X = np.random.RandomState(42).randn(100, 3)
        y = np.array([0.0] * 80 + [1.0] * 20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        # Class ratio in test should be roughly maintained
        test_ratio = np.mean(y_test == 1.0)
        original_ratio = np.mean(y == 1.0)
        assert abs(test_ratio - original_ratio) < 0.15  # within 15%

    def test_invalid_test_size(self):
        from ferroml.model_selection import train_test_split
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=0.0)
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=1.0)

    def test_vs_sklearn(self):
        """Compare train_test_split output properties with sklearn."""
        pytest.importorskip("sklearn")
        from sklearn.model_selection import train_test_split as sk_split
        from ferroml.model_selection import train_test_split

        X = np.random.RandomState(42).randn(200, 5)
        y = (X[:, 0] > 0).astype(float)

        # Same shapes
        ferro_result = train_test_split(X, y, test_size=0.25, random_state=42)
        sk_result = sk_split(X, y, test_size=0.25, random_state=42)
        assert ferro_result[0].shape == sk_result[0].shape
        assert ferro_result[1].shape == sk_result[1].shape


# =============================================================================
# ROC/PR curves
# =============================================================================

class TestMetricsCurves:
    def test_roc_curve_keys(self):
        from ferroml.metrics import roc_curve
        y_true = np.array([0., 0., 1., 1.])
        y_score = np.array([0.1, 0.4, 0.8, 0.9])
        roc = roc_curve(y_true, y_score)
        assert "fpr" in roc
        assert "tpr" in roc
        assert "thresholds" in roc
        assert "auc" in roc

    def test_roc_curve_perfect(self):
        from ferroml.metrics import roc_curve
        y_true = np.array([0., 0., 1., 1.])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        roc = roc_curve(y_true, y_score)
        assert abs(roc["auc"] - 1.0) < 1e-6

    def test_roc_curve_bounds(self):
        from ferroml.metrics import roc_curve
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100).astype(float)
        y_score = rng.rand(100)
        roc = roc_curve(y_true, y_score)
        assert 0.0 <= roc["auc"] <= 1.0
        assert all(0.0 <= f <= 1.0 for f in roc["fpr"])
        assert all(0.0 <= t <= 1.0 for t in roc["tpr"])

    def test_roc_curve_vs_sklearn(self):
        pytest.importorskip("sklearn")
        from sklearn.metrics import roc_auc_score as sk_auc
        from ferroml.metrics import roc_curve
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200).astype(float)
        y_score = rng.rand(200)
        roc = roc_curve(y_true, y_score)
        sk_auc_val = sk_auc(y_true, y_score)
        assert abs(roc["auc"] - sk_auc_val) < 0.01

    def test_precision_recall_curve_keys(self):
        from ferroml.metrics import precision_recall_curve
        y_true = np.array([0., 0., 1., 1.])
        y_score = np.array([0.1, 0.4, 0.8, 0.9])
        pr = precision_recall_curve(y_true, y_score)
        assert "precision" in pr
        assert "recall" in pr
        assert "thresholds" in pr
        assert "auc" in pr
        assert "average_precision" in pr

    def test_precision_recall_curve_vs_sklearn(self):
        pytest.importorskip("sklearn")
        from sklearn.metrics import average_precision_score as sk_ap
        from ferroml.metrics import precision_recall_curve
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200).astype(float)
        y_score = rng.rand(200)
        pr = precision_recall_curve(y_true, y_score)
        sk_ap_val = sk_ap(y_true, y_score)
        assert abs(pr["average_precision"] - sk_ap_val) < 0.05


# =============================================================================
# CV Splitters
# =============================================================================

class TestCVSplitters:
    def test_repeated_kfold_count(self):
        from ferroml.cv import RepeatedKFold
        X = np.random.randn(50, 3)
        rkf = RepeatedKFold(n_folds=5, n_repeats=3, random_state=42)
        splits = rkf.split(X)
        assert len(splits) == 15  # 5 * 3

    def test_repeated_kfold_coverage(self):
        from ferroml.cv import RepeatedKFold
        X = np.random.randn(20, 3)
        rkf = RepeatedKFold(n_folds=4, n_repeats=2, random_state=0)
        splits = rkf.split(X)
        for train_idx, test_idx in splits:
            assert len(set(train_idx) & set(test_idx)) == 0
            assert len(train_idx) + len(test_idx) == 20

    def test_shuffle_split_sizes(self):
        from ferroml.cv import ShuffleSplit
        X = np.random.randn(100, 3)
        ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        splits = ss.split(X)
        assert len(splits) == 5
        for train_idx, test_idx in splits:
            assert abs(len(test_idx) - 20) <= 1
            assert abs(len(train_idx) - 80) <= 1

    def test_shuffle_split_reproducibility(self):
        from ferroml.cv import ShuffleSplit
        X = np.random.randn(50, 3)
        ss1 = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
        ss2 = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
        splits1 = ss1.split(X)
        splits2 = ss2.split(X)
        for (t1, v1), (t2, v2) in zip(splits1, splits2):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)

    def test_group_kfold_no_group_leakage(self):
        from ferroml.cv import GroupKFold
        X = np.random.randn(100, 3)
        groups = np.array([i // 10 for i in range(100)])
        gkf = GroupKFold(n_folds=5)
        splits = gkf.split(X, groups=groups)
        assert len(splits) == 5
        for train_idx, test_idx in splits:
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert len(train_groups & test_groups) == 0, "Group leakage detected!"

    def test_leave_p_out_basic(self):
        from ferroml.cv import LeavePOut
        X = np.random.randn(5, 2)
        lpo = LeavePOut(p=2)
        splits = lpo.split(X)
        assert len(splits) == 10  # C(5, 2) = 10
        for train_idx, test_idx in splits:
            assert len(test_idx) == 2
            assert len(train_idx) == 3


# =============================================================================
# Learning Curve / Validation Curve
# =============================================================================

class TestDiagnosticCurves:
    def test_learning_curve_returns_correct_length(self):
        from ferroml.cv import learning_curve
        from ferroml.linear import LogisticRegression
        X = np.random.RandomState(42).randn(100, 3)
        y = (X[:, 0] > 0).astype(float)
        sizes, train_scores, test_scores = learning_curve(
            LogisticRegression(), X, y, cv=3, train_sizes=[0.2, 0.5, 1.0]
        )
        assert len(sizes) == 3
        assert len(train_scores) == 3
        assert len(test_scores) == 3

    def test_learning_curve_increasing_trend(self):
        from ferroml.cv import learning_curve
        from ferroml.linear import LogisticRegression
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
        sizes, train_scores, test_scores = learning_curve(
            LogisticRegression(), X, y, cv=3, train_sizes=[0.1, 0.5, 1.0]
        )
        # Test scores should generally increase (or at least not decrease dramatically)
        # With more data, model should be at least as good with full data vs 10%
        assert test_scores[-1] >= test_scores[0] - 0.15

    def test_validation_curve_returns_correct_length(self):
        from ferroml.cv import validation_curve
        from ferroml.linear import LogisticRegression
        X = np.random.RandomState(42).randn(100, 3)
        y = (X[:, 0] > 0).astype(float)
        param_range = [0.001, 0.01, 0.1, 1.0]
        params, train_scores, test_scores = validation_curve(
            LogisticRegression(), X, y, param_name="l2_penalty",
            param_range=param_range, cv=3
        )
        assert len(params) == 4
        assert len(train_scores) == 4
        assert len(test_scores) == 4


# =============================================================================
# Normalizer
# =============================================================================

class TestNormalizer:
    def test_l2_norm(self):
        from ferroml.preprocessing import Normalizer
        X = np.array([[1., 2., 3.], [4., 5., 6.]])
        norm = Normalizer(norm="l2")
        X_normed = norm.fit_transform(X)
        norms = np.linalg.norm(X_normed, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_l1_norm(self):
        from ferroml.preprocessing import Normalizer
        X = np.array([[1., 2., 3.], [4., 5., 6.]])
        norm = Normalizer(norm="l1")
        X_normed = norm.fit_transform(X)
        norms = np.sum(np.abs(X_normed), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_max_norm(self):
        from ferroml.preprocessing import Normalizer
        X = np.array([[1., 2., 3.], [4., 5., 6.]])
        norm = Normalizer(norm="max")
        X_normed = norm.fit_transform(X)
        max_vals = np.max(np.abs(X_normed), axis=1)
        np.testing.assert_allclose(max_vals, 1.0, atol=1e-10)


# =============================================================================
# model_selection re-exports
# =============================================================================

class TestModelSelectionReExports:
    def test_all_imports(self):
        from ferroml.model_selection import (
            train_test_split,
            cross_val_score,
            KFold,
            StratifiedKFold,
            TimeSeriesSplit,
            LeaveOneOut,
            RepeatedKFold,
            ShuffleSplit,
            GroupKFold,
            LeavePOut,
            GridSearchCV,
            RandomSearchCV,
        )
        # Just verify they are all importable
        assert train_test_split is not None
        assert cross_val_score is not None
        assert KFold is not None
        assert GridSearchCV is not None

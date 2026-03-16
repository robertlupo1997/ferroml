"""Tests for ferroml.cv — cross-validation splitters and cross_val_score."""

import numpy as np
import pytest

from ferroml.cv import KFold, LeaveOneOut, StratifiedKFold, TimeSeriesSplit, cross_val_score
from ferroml.linear import LogisticRegression, LinearRegression


class TestKFold:
    def test_basic_split(self):
        kf = KFold(n_folds=5)
        X = np.random.randn(100, 3)
        splits = kf.split(X)
        assert len(splits) == 5
        for train, test in splits:
            assert len(train) + len(test) == 100

    def test_no_overlap(self):
        kf = KFold(n_folds=5)
        X = np.random.randn(50, 3)
        splits = kf.split(X)
        all_test = np.concatenate([test for _, test in splits])
        assert len(np.unique(all_test)) == 50

    def test_shuffle_reproducibility(self):
        kf = KFold(n_folds=3, shuffle=True, random_state=42)
        X = np.random.randn(30, 3)
        splits1 = kf.split(X)
        splits2 = kf.split(X)
        np.testing.assert_array_equal(splits1[0][0], splits2[0][0])

    def test_n_splits_property(self):
        kf = KFold(n_folds=7)
        assert kf.n_splits == 7


class TestStratifiedKFold:
    def test_requires_y(self):
        skf = StratifiedKFold(n_folds=3)
        X = np.random.randn(30, 3)
        with pytest.raises(ValueError, match="requires y"):
            skf.split(X)

    def test_basic_split(self):
        skf = StratifiedKFold(n_folds=3)
        X = np.random.randn(30, 3)
        y = np.array([0.0] * 20 + [1.0] * 10)
        splits = skf.split(X, y)
        assert len(splits) == 3

    def test_class_balance(self):
        """Each fold should have approximately the same class distribution."""
        skf = StratifiedKFold(n_folds=5, shuffle=True, random_state=42)
        X = np.random.randn(100, 3)
        y = np.array([0.0] * 70 + [1.0] * 30)
        splits = skf.split(X, y)
        for _, test in splits:
            test_y = y[test.astype(int)]
            # Each fold should have roughly 30% class 1
            ratio = np.mean(test_y == 1.0)
            assert 0.1 <= ratio <= 0.5  # Loose bounds


class TestTimeSeriesSplit:
    def test_basic_split(self):
        tss = TimeSeriesSplit(n_splits=3)
        X = np.random.randn(40, 3)
        splits = tss.split(X)
        assert len(splits) == 3

    def test_temporal_ordering(self):
        """Training indices should always precede test indices."""
        tss = TimeSeriesSplit(n_splits=3)
        X = np.random.randn(40, 3)
        splits = tss.split(X)
        for train, test in splits:
            assert train.max() < test.min()


class TestLeaveOneOut:
    def test_basic(self):
        loo = LeaveOneOut()
        X = np.random.randn(10, 3)
        splits = loo.split(X)
        assert len(splits) == 10
        for train, test in splits:
            assert len(test) == 1
            assert len(train) == 9


class TestCrossValScore:
    def test_classification(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(float)
        scores = cross_val_score(LogisticRegression(), X, y, cv=5)
        assert len(scores) == 5
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert np.mean(scores) > 0.7

    def test_regression_r2(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(100) * 0.1
        scores = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")
        assert len(scores) == 5
        assert np.mean(scores) > 0.9

    def test_regression_mse(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(100) * 0.1
        scores = cross_val_score(LinearRegression(), X, y, cv=5, scoring="mse")
        assert len(scores) == 5
        # MSE scores should be negative (sklearn convention)
        assert all(s <= 0 for s in scores)

    def test_invalid_scoring(self):
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        with pytest.raises(ValueError, match="Unknown scoring"):
            cross_val_score(LinearRegression(), X, y, scoring="invalid")

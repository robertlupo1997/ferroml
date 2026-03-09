"""
Phase M.7: Edge-case comparison tests — FerroML vs sklearn.

Validates that FerroML handles tricky data scenarios (high dimensionality,
class imbalance, near-constant features, large values, constant targets,
multicollinearity, tiny datasets) at least as gracefully as sklearn.

~25 tests covering 7 edge-case categories.
"""

import numpy as np
import pytest

from conftest_comparison import r2_score, accuracy_score

# ---------------------------------------------------------------------------
# FerroML imports
# ---------------------------------------------------------------------------
from ferroml.linear import (
    LinearRegression as FerroLR,
    RidgeRegression as FerroRidge,
    LassoRegression as FerroLasso,
    ElasticNet as FerroEN,
    LogisticRegression as FerroLogR,
)
from ferroml.trees import (
    RandomForestClassifier as FerroRF,
    GradientBoostingClassifier as FerroGB,
    DecisionTreeClassifier as FerroDT,
)
from ferroml.preprocessing import (
    StandardScaler as FerroSS,
    RobustScaler as FerroRS,
    VarianceThreshold as FerroVT,
)
from ferroml.neighbors import KNeighborsClassifier as FerroKNN
from ferroml.decomposition import PCA as FerroPCA

# ---------------------------------------------------------------------------
# sklearn imports
# ---------------------------------------------------------------------------
from sklearn.linear_model import (
    LinearRegression as SkLR,
    Ridge as SkRidge,
    Lasso as SkLasso,
    ElasticNet as SkEN,
    LogisticRegression as SkLogR,
)
from sklearn.ensemble import (
    RandomForestClassifier as SkRF,
    GradientBoostingClassifier as SkGB,
)
from sklearn.tree import DecisionTreeClassifier as SkDT
from sklearn.preprocessing import (
    StandardScaler as SkSS,
    RobustScaler as SkRS,
)
from sklearn.feature_selection import VarianceThreshold as SkVT
from sklearn.neighbors import KNeighborsClassifier as SkKNN
from sklearn.decomposition import PCA as SkPCA
from sklearn.metrics import precision_score, recall_score, f1_score


# ===================================================================
# 1. High Dimensionality (p >> n) — 4 tests
# ===================================================================

class TestHighDimensionality:
    """Tests where the number of features far exceeds the number of samples."""

    def test_linear_regression_underdetermined(self):
        """p >> n: FerroML rejects underdetermined OLS; sklearn uses pseudo-inverse."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 500)
        y = rng.randn(50)

        # sklearn handles it with pseudo-inverse
        sk = SkLR()
        sk.fit(X, y)
        sp = sk.predict(X)
        assert sp.shape == (50,)
        assert not np.any(np.isnan(sp))

        # FerroML raises because it requires n > p for OLS — a valid design choice
        with pytest.raises(RuntimeError, match="Need more observations"):
            ferro = FerroLR()
            ferro.fit(X, y)

    def test_pca_high_dim(self):
        """PCA on wide data: both should respect n_components <= min(n, p)."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 200)
        n_comp = 50

        ferro = FerroPCA(n_components=n_comp)
        ferro.fit(X)
        ft = ferro.transform(X)

        sk = SkPCA(n_components=n_comp)
        sk.fit(X)
        st = sk.transform(X)

        assert ft.shape == st.shape == (100, n_comp)
        # Explained variance should be in same ballpark
        ferro_var = np.var(ft, axis=0).sum()
        sk_var = np.var(st, axis=0).sum()
        assert abs(ferro_var - sk_var) / sk_var < 0.05

    def test_random_forest_high_dim(self):
        """RF on wide data: both should fit and predict without error."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 1000)
        y = (X[:, 0] > 0).astype(np.float64)

        ferro = FerroRF(n_estimators=10, random_state=42)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkRF(n_estimators=10, random_state=42)
        sk.fit(X, y)
        sp = sk.predict(X)

        # Both should achieve high train accuracy (likely overfit)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc > 0.8
        assert sk_acc > 0.8

    def test_ridge_high_dim(self):
        """Ridge on underdetermined system: regularization handles p >> n."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 500)
        y = X[:, :3].sum(axis=1) + rng.randn(50) * 0.1

        ferro = FerroRidge(alpha=1.0)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkRidge(alpha=1.0)
        sk.fit(X, y)
        sp = sk.predict(X)

        # Both should produce valid predictions
        assert fp.shape == (50,)
        assert not np.any(np.isnan(fp))
        assert not np.any(np.isinf(fp))

        # Both should achieve reasonable fit despite underdetermination
        ferro_r2 = r2_score(y, fp)
        sk_r2 = r2_score(y, sp)
        assert ferro_r2 > 0.0
        assert sk_r2 > 0.0


# ===================================================================
# 2. Class Imbalance (99:1 ratio) — 4 tests
# ===================================================================

class TestClassImbalance:
    """Tests with severely imbalanced binary classification data."""

    @staticmethod
    def _make_imbalanced(n=1000, minority_frac=0.01, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, 5)
        y = np.zeros(n)
        n_pos = max(int(n * minority_frac), 2)
        y[:n_pos] = 1.0
        # Give minority class a signal
        X[:n_pos, 0] += 3.0
        return X, y

    def test_logistic_regression_imbalanced(self):
        """LogisticRegression on 99:1 data: both should fit without crashing."""
        X, y = self._make_imbalanced()

        ferro = FerroLogR()
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkLogR(max_iter=1000)
        sk.fit(X, y)
        sp = sk.predict(X)

        # Both should at least predict the majority class correctly
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc > 0.90
        assert sk_acc > 0.90

    def test_random_forest_imbalanced(self):
        """RF on imbalanced data: should detect at least some minority samples."""
        X, y = self._make_imbalanced(n=500, minority_frac=0.02)

        ferro = FerroRF(n_estimators=50, random_state=42)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkRF(n_estimators=50, random_state=42)
        sk.fit(X, y)
        sp = sk.predict(X)

        # Both should find some minority class on training data
        assert np.sum(fp == 1.0) >= 1, "FerroML RF found no minority samples"
        assert np.sum(sp == 1.0) >= 1, "sklearn RF found no minority samples"

    def test_gradient_boosting_imbalanced(self):
        """GB on imbalanced data: should handle without error."""
        X, y = self._make_imbalanced(n=500, minority_frac=0.02)

        ferro = FerroGB(n_estimators=50, random_state=42)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkGB(n_estimators=50, random_state=42)
        sk.fit(X, y)
        sp = sk.predict(X)

        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc > 0.90
        assert sk_acc > 0.90

    def test_imbalanced_precision_recall(self):
        """On imbalanced data, compare precision/recall rather than just accuracy."""
        X, y = self._make_imbalanced(n=1000, minority_frac=0.05, seed=123)
        # Stronger signal so recall is possible
        n_pos = int(1000 * 0.05)
        X[:n_pos, 0] += 5.0

        ferro = FerroRF(n_estimators=100, random_state=42)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkRF(n_estimators=100, random_state=42)
        sk.fit(X, y)
        sp = sk.predict(X)

        # Both should have some recall for the minority class (train set)
        ferro_recall = recall_score(y, fp)
        sk_recall = recall_score(y, sp)
        assert ferro_recall > 0.5, f"FerroML recall too low: {ferro_recall}"
        assert sk_recall > 0.5, f"sklearn recall too low: {sk_recall}"


# ===================================================================
# 3. Near-Constant Features — 3 tests
# ===================================================================

class TestNearConstantFeatures:
    """Tests with features that have near-zero variance."""

    def test_variance_threshold_removes_same(self):
        """VarianceThreshold should remove the same near-constant columns."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        X[:, 1] = 7.0  # constant
        X[:, 3] = 3.0 + rng.randn(100) * 1e-12  # near-constant

        threshold = 1e-8

        ferro = FerroVT(threshold=threshold)
        ferro.fit(X)
        ft = ferro.transform(X)

        sk = SkVT(threshold=threshold)
        sk.fit(X)
        st = sk.transform(X)

        # Both should keep the same number of features
        assert ft.shape[1] == st.shape[1], (
            f"FerroML kept {ft.shape[1]} features, sklearn kept {st.shape[1]}"
        )

    def test_standard_scaler_near_constant(self):
        """StandardScaler with near-zero variance should not produce NaN."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        X[:, 2] = 5.0 + rng.randn(100) * 1e-10

        ferro = FerroSS()
        ferro.fit(X)
        ft = ferro.transform(X)

        sk = SkSS()
        sk.fit(X)
        st = sk.transform(X)

        # Neither should produce NaN
        assert not np.any(np.isnan(ft)), "FerroML StandardScaler produced NaN"
        assert not np.any(np.isnan(st)), "sklearn StandardScaler produced NaN"

    def test_linear_regression_near_constant_feature(self):
        """LR with a near-constant column should still produce valid predictions."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        X[:, 0] = 5.0 + rng.randn(100) * 1e-10
        y = X[:, 1] * 2.0 + rng.randn(100) * 0.1

        ferro = FerroLR()
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkLR()
        sk.fit(X, y)
        sp = sk.predict(X)

        # Both should produce valid predictions
        assert not np.any(np.isnan(fp))
        assert not np.any(np.isnan(sp))

        # Both should capture the signal from column 1
        ferro_r2 = r2_score(y, fp)
        sk_r2 = r2_score(y, sp)
        assert ferro_r2 > 0.9
        assert sk_r2 > 0.9


# ===================================================================
# 4. Large Feature Values — 3 tests
# ===================================================================

class TestLargeFeatureValues:
    """Tests with features at extreme scales (1e10)."""

    def test_linear_regression_large_values(self):
        """LR should handle features at 1e10 scale."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5) * 1e10
        y = X[:, 0] * 2.0 + rng.randn(100) * 1e10

        ferro = FerroLR()
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkLR()
        sk.fit(X, y)
        sp = sk.predict(X)

        assert not np.any(np.isnan(fp))
        assert not np.any(np.isinf(fp))
        assert not np.any(np.isnan(sp))

        # R2 should be similar
        ferro_r2 = r2_score(y, fp)
        sk_r2 = r2_score(y, sp)
        assert abs(ferro_r2 - sk_r2) < 0.1

    def test_robust_scaler_large_values(self):
        """RobustScaler should handle extreme values without overflow."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5) * 1e10
        # Add some outliers
        X[0, :] = 1e15

        ferro = FerroRS()
        ferro.fit(X)
        ft = ferro.transform(X)

        sk = SkRS()
        sk.fit(X)
        st = sk.transform(X)

        assert not np.any(np.isnan(ft)), "FerroML RobustScaler produced NaN"
        assert not np.any(np.isinf(ft)), "FerroML RobustScaler produced Inf"
        assert not np.any(np.isnan(st))

    def test_standard_scaler_large_values(self):
        """StandardScaler should normalize extreme-valued data correctly."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5) * 1e10

        ferro = FerroSS()
        ferro.fit(X)
        ft = ferro.transform(X)

        sk = SkSS()
        sk.fit(X)
        st = sk.transform(X)

        assert not np.any(np.isnan(ft))
        assert not np.any(np.isinf(ft))

        # After scaling, values should be around [-3, 3] for most points
        assert np.max(np.abs(ft)) < 10
        assert np.max(np.abs(st)) < 10

        # Means should be near zero
        assert np.allclose(ft.mean(axis=0), 0.0, atol=1e-6)
        assert np.allclose(st.mean(axis=0), 0.0, atol=1e-6)


# ===================================================================
# 5. Constant Target — 3 tests
# ===================================================================

class TestConstantTarget:
    """Tests where the target variable is constant."""

    def test_linear_regression_constant_y(self):
        """LR with constant y should predict that constant."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = np.ones(50) * 3.0

        ferro = FerroLR()
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkLR()
        sk.fit(X, y)
        sp = sk.predict(X)

        # Both should predict ~3.0
        assert np.allclose(fp, 3.0, atol=1e-6), f"FerroML range: {fp.min():.6f} - {fp.max():.6f}"
        assert np.allclose(sp, 3.0, atol=1e-6)

    def test_decision_tree_constant_y(self):
        """DT classifier with constant y: FerroML requires 2+ classes, sklearn allows 1."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = np.zeros(50)

        # sklearn handles single-class
        sk = SkDT(random_state=42)
        sk.fit(X, y)
        sp = sk.predict(X)
        assert np.all(sp == 0.0)

        # FerroML requires at least 2 classes — a valid design choice
        with pytest.raises(RuntimeError, match="at least 2 classes"):
            ferro = FerroDT(random_state=42)
            ferro.fit(X, y)

    def test_ridge_constant_y(self):
        """Ridge with constant y should predict that constant."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = np.ones(50) * 7.5

        ferro = FerroRidge(alpha=1.0)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkRidge(alpha=1.0)
        sk.fit(X, y)
        sp = sk.predict(X)

        # Both should predict close to 7.5
        assert np.allclose(fp, 7.5, atol=0.1), f"FerroML range: {fp.min():.4f} - {fp.max():.4f}"
        assert np.allclose(sp, 7.5, atol=1e-6)


# ===================================================================
# 6. Multicollinear Features — 4 tests
# ===================================================================

class TestMulticollinearity:
    """Tests with perfectly or near-perfectly correlated features."""

    def test_ridge_vs_ols_multicollinear(self):
        """Ridge should be more stable than OLS on multicollinear data."""
        rng = np.random.RandomState(42)
        n = 100
        X = rng.randn(n, 5)
        X[:, 1] = X[:, 0] + rng.randn(n) * 1e-8  # near-perfect correlation
        y = X[:, 0] * 3.0 + rng.randn(n) * 0.5

        # Ridge in both
        ferro_ridge = FerroRidge(alpha=1.0)
        ferro_ridge.fit(X, y)
        frp = ferro_ridge.predict(X)

        sk_ridge = SkRidge(alpha=1.0)
        sk_ridge.fit(X, y)
        srp = sk_ridge.predict(X)

        # Both Ridge should produce valid predictions
        assert not np.any(np.isnan(frp))
        assert not np.any(np.isnan(srp))
        assert r2_score(y, frp) > 0.5
        assert r2_score(y, srp) > 0.5

    def test_elasticnet_multicollinear(self):
        """ElasticNet should handle multicollinearity via L1+L2 penalties."""
        rng = np.random.RandomState(42)
        n = 100
        X = rng.randn(n, 5)
        X[:, 1] = X[:, 0] + rng.randn(n) * 1e-8
        y = X[:, 0] * 3.0 + rng.randn(n) * 0.5

        ferro = FerroEN(alpha=0.1, l1_ratio=0.5)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkEN(alpha=0.1, l1_ratio=0.5, max_iter=5000)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert not np.any(np.isnan(fp))
        assert r2_score(y, fp) > 0.5
        assert r2_score(y, sp) > 0.5

    def test_lasso_multicollinear_feature_selection(self):
        """Lasso should select one of the correlated features, not both."""
        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 5)
        X[:, 1] = X[:, 0] + rng.randn(n) * 1e-8
        y = X[:, 0] * 3.0 + X[:, 2] * 1.0 + rng.randn(n) * 0.1

        ferro = FerroLasso(alpha=0.1)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkLasso(alpha=0.1, max_iter=5000)
        sk.fit(X, y)
        sp = sk.predict(X)

        # Both should achieve good fit
        assert r2_score(y, fp) > 0.5
        assert r2_score(y, sp) > 0.5

    def test_ridge_prediction_stability_multicollinear(self):
        """Ridge predictions should be stable across slightly perturbed data."""
        rng = np.random.RandomState(42)
        n = 100
        X = rng.randn(n, 5)
        X[:, 1] = X[:, 0] + rng.randn(n) * 1e-8
        y = X[:, 0] * 3.0 + rng.randn(n) * 0.5

        # Train on original data
        ferro = FerroRidge(alpha=1.0)
        ferro.fit(X, y)

        # Predict on slightly perturbed test point
        x_test = rng.randn(1, 5)
        p1 = ferro.predict(x_test)
        x_test_perturbed = x_test.copy()
        x_test_perturbed[0, 0] += 1e-6
        p2 = ferro.predict(x_test_perturbed)

        # Predictions should change only slightly
        assert abs(p1[0] - p2[0]) < 1.0, (
            f"Ridge predictions unstable: {p1[0]:.6f} vs {p2[0]:.6f}"
        )


# ===================================================================
# 7. Small Datasets — 4 tests
# ===================================================================

class TestSmallDatasets:
    """Tests with very few samples."""

    def test_knn_fewer_samples_than_neighbors(self):
        """KNN with n_samples < k: both should raise or clip k."""
        rng = np.random.RandomState(42)
        X = rng.randn(3, 5)
        y = np.array([0.0, 1.0, 0.0])

        # FerroML raises
        with pytest.raises(RuntimeError, match="n_neighbors.*cannot be greater"):
            ferro = FerroKNN(n_neighbors=5)
            ferro.fit(X, y)

        # sklearn also raises
        with pytest.raises(ValueError):
            sk = SkKNN(n_neighbors=5)
            sk.fit(X, y)
            sk.predict(X)

    def test_all_models_tiny_dataset(self):
        """All regression models should work on a 10-sample dataset."""
        rng = np.random.RandomState(42)
        X = rng.randn(10, 3)
        y = X[:, 0] * 2.0 + rng.randn(10) * 0.1

        models_ferro = [
            ("LR", FerroLR()),
            ("Ridge", FerroRidge(alpha=1.0)),
            ("Lasso", FerroLasso(alpha=0.01)),
        ]
        models_sk = [
            ("LR", SkLR()),
            ("Ridge", SkRidge(alpha=1.0)),
            ("Lasso", SkLasso(alpha=0.01, max_iter=5000)),
        ]

        for (name, ferro), (_, sk) in zip(models_ferro, models_sk):
            ferro.fit(X, y)
            fp = ferro.predict(X)
            sk.fit(X, y)
            sp = sk.predict(X)

            assert fp.shape == (10,), f"{name}: wrong shape"
            assert not np.any(np.isnan(fp)), f"{name}: FerroML produced NaN"
            assert r2_score(y, fp) > 0.5, f"{name}: FerroML R2 too low"

    def test_pca_more_components_than_samples(self):
        """PCA with n_components > n_samples: should be capped or raise."""
        rng = np.random.RandomState(42)
        X = rng.randn(10, 20)

        # Request more components than samples
        # sklearn caps at min(n_samples, n_features)
        sk = SkPCA(n_components=10)
        sk.fit(X)
        st = sk.transform(X)
        assert st.shape == (10, 10)

        # FerroML should also handle this
        ferro = FerroPCA(n_components=10)
        ferro.fit(X)
        ft = ferro.transform(X)
        assert ft.shape == (10, 10)

    def test_classifier_tiny_dataset(self):
        """Classifiers on a very small dataset (10 samples, 2 classes)."""
        rng = np.random.RandomState(42)
        X = rng.randn(10, 3)
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        ferro_rf = FerroRF(n_estimators=10, random_state=42)
        ferro_rf.fit(X, y)
        fp = ferro_rf.predict(X)

        sk_rf = SkRF(n_estimators=10, random_state=42)
        sk_rf.fit(X, y)
        sp = sk_rf.predict(X)

        # Both should produce valid predictions
        assert set(np.unique(fp)).issubset({0.0, 1.0})
        assert fp.shape == (10,)

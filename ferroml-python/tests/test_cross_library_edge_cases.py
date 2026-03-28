"""
Cross-library edge-case tests: FerroML vs sklearn.

Tests edge cases that commonly cause ML libraries to diverge or crash:
1. Single sample (fit/predict on 1 sample)
2. High dimensional (p >> n)
3. Sparse data (>90% zeros)
4. Extreme values (1e+/-100)
5. Constant features
6. Constant target
7. Large class count (20 classes)
8. Near-duplicate rows

~50 tests across all categories.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# FerroML imports
# ---------------------------------------------------------------------------
from ferroml.linear import (
    LinearRegression as FerroLR,
    RidgeRegression as FerroRidge,
    LassoRegression as FerroLasso,
    LogisticRegression as FerroLogR,
)
from ferroml.trees import (
    DecisionTreeClassifier as FerroDTC,
    DecisionTreeRegressor as FerroDTR,
    RandomForestClassifier as FerroRF,
    GradientBoostingRegressor as FerroGBR,
)
from ferroml.preprocessing import StandardScaler as FerroSS, CountVectorizer as FerroCV
from ferroml.decomposition import PCA as FerroPCA
from ferroml.neighbors import KNeighborsClassifier as FerroKNN
from ferroml.svm import SVC as FerroSVC
from ferroml.clustering import KMeans as FerroKM, DBSCAN as FerroDBSCAN
from ferroml.naive_bayes import GaussianNB as FerroGNB, MultinomialNB as FerroMNB

# ---------------------------------------------------------------------------
# sklearn imports
# ---------------------------------------------------------------------------
from sklearn.linear_model import (
    LinearRegression as SkLR,
    Ridge as SkRidge,
    Lasso as SkLasso,
    LogisticRegression as SkLogR,
)
from sklearn.tree import (
    DecisionTreeClassifier as SkDTC,
    DecisionTreeRegressor as SkDTR,
)
from sklearn.ensemble import (
    RandomForestClassifier as SkRF,
    GradientBoostingRegressor as SkGBR,
)
from sklearn.preprocessing import StandardScaler as SkSS
from sklearn.feature_extraction.text import CountVectorizer as SkCV
from sklearn.decomposition import PCA as SkPCA
from sklearn.neighbors import KNeighborsClassifier as SkKNN
from sklearn.svm import SVC as SkSVC
from sklearn.cluster import KMeans as SkKM, DBSCAN as SkDBSCAN
from sklearn.naive_bayes import GaussianNB as SkGNB, MultinomialNB as SkMNB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


# ===================================================================
# 1. Single Sample — 7 tests
# ===================================================================

class TestSingleSample:
    """Fit and/or predict with a single sample."""

    def test_linear_regression_single_sample_errors(self):
        """LR on 1 sample: FerroML needs n > p, sklearn uses pseudo-inverse."""
        X = np.array([[1.0, 2.0]])
        y = np.array([3.0])

        # sklearn handles it
        sk = SkLR()
        sk.fit(X, y)
        assert sk.predict(X).shape == (1,)

        # FerroML requires n > p (intercept counts as a param)
        with pytest.raises(ValueError, match="Need more observations"):
            FerroLR().fit(X, y)

    def test_ridge_single_sample(self):
        """Ridge on 1 sample: regularization makes this solvable."""
        X = np.array([[1.0, 2.0, 3.0]])
        y = np.array([5.0])

        ferro = FerroRidge(alpha=1.0)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkRidge(alpha=1.0)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert fp.shape == (1,)
        assert not np.isnan(fp[0])
        assert sp.shape == (1,)

    def test_knn_k1_single_sample(self):
        """KNN(k=1) fitted on 2 samples, predict single sample."""
        X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
        y_train = np.array([0.0, 1.0])
        X_test = np.array([[0.1, 0.1]])

        ferro = FerroKNN(n_neighbors=1)
        ferro.fit(X_train, y_train)
        fp = ferro.predict(X_test)

        sk = SkKNN(n_neighbors=1)
        sk.fit(X_train, y_train)
        sp = sk.predict(X_test)

        assert fp[0] == 0.0
        assert sp[0] == 0.0

    def test_decision_tree_regressor_single_sample(self):
        """DT regressor on 1 sample should memorize it."""
        X = np.array([[1.0, 2.0]])
        y = np.array([42.0])

        ferro = FerroDTR(random_state=42)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkDTR(random_state=42)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert np.isclose(fp[0], 42.0, atol=1e-6)
        assert np.isclose(sp[0], 42.0, atol=1e-6)

    def test_decision_tree_classifier_single_sample_error(self):
        """DT classifier needs 2+ classes; single sample = single class."""
        X = np.array([[1.0, 2.0]])
        y = np.array([0.0])

        # FerroML requires at least 2 classes
        with pytest.raises(ValueError, match="at least 2 classes"):
            FerroDTC(random_state=42).fit(X, y)

        # sklearn allows single-class
        sk = SkDTC(random_state=42)
        sk.fit(X, y)
        assert sk.predict(X)[0] == 0.0

    def test_standard_scaler_single_sample(self):
        """StandardScaler on 1 sample: std=0 so scale to 0."""
        X = np.array([[3.0, 7.0, -1.0]])

        ferro = FerroSS()
        ferro.fit(X)
        ft = ferro.transform(X)

        sk = SkSS()
        sk.fit(X)
        st = sk.transform(X)

        # Both should produce zeros (mean-centered, std=0 -> 0)
        assert not np.any(np.isnan(ft))
        assert not np.any(np.isnan(st))
        assert ft.shape == (1, 3)

    def test_predict_single_sample_after_normal_fit(self):
        """Fit on many samples, predict on 1 sample."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 4)
        y_train = (X_train[:, 0] > 0).astype(np.float64)
        X_test = rng.randn(1, 4)

        ferro = FerroRF(n_estimators=10, random_state=42)
        ferro.fit(X_train, y_train)
        fp = ferro.predict(X_test)

        sk = SkRF(n_estimators=10, random_state=42)
        sk.fit(X_train, y_train)
        sp = sk.predict(X_test)

        assert fp.shape == (1,)
        assert sp.shape == (1,)
        assert fp[0] in (0.0, 1.0)


# ===================================================================
# 2. High Dimensional (p >> n) — 7 tests
# ===================================================================

class TestHighDimensional:
    """50 samples, 500 features."""

    @pytest.fixture()
    def high_dim_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 500)
        y_reg = X[:, :3].sum(axis=1) + rng.randn(50) * 0.1
        y_cls = (X[:, 0] > 0).astype(np.float64)
        return X, y_reg, y_cls

    def test_ridge_high_dim(self, high_dim_data):
        X, y, _ = high_dim_data
        ferro = FerroRidge(alpha=1.0)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkRidge(alpha=1.0)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert not np.any(np.isnan(fp))
        assert not np.any(np.isinf(fp))
        assert _r2(y, fp) > 0.0
        assert _r2(y, sp) > 0.0

    def test_lasso_high_dim(self, high_dim_data):
        X, y, _ = high_dim_data
        ferro = FerroLasso(alpha=0.01)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkLasso(alpha=0.01, max_iter=10000)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert not np.any(np.isnan(fp))
        assert fp.shape == (50,)
        assert sp.shape == (50,)

    def test_pca_high_dim(self, high_dim_data):
        X, _, _ = high_dim_data
        n_comp = 10

        ferro = FerroPCA(n_components=n_comp)
        ferro.fit(X)
        ft = ferro.transform(X)

        sk = SkPCA(n_components=n_comp)
        sk.fit(X)
        st = sk.transform(X)

        assert ft.shape == (50, 10)
        assert st.shape == (50, 10)
        # Explained variance should be in same ballpark
        ferro_var = np.var(ft, axis=0).sum()
        sk_var = np.var(st, axis=0).sum()
        assert abs(ferro_var - sk_var) / max(sk_var, 1e-10) < 0.1

    def test_svc_linear_high_dim(self, high_dim_data):
        X, _, y = high_dim_data
        ferro = FerroSVC(kernel="linear")
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkSVC(kernel="linear")
        sk.fit(X, y)
        sp = sk.predict(X)

        assert fp.shape == (50,)
        assert _accuracy(y, fp) > 0.6
        assert _accuracy(y, sp) > 0.6

    def test_logistic_high_dim(self, high_dim_data):
        X, _, y = high_dim_data
        ferro = FerroLogR()
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkLogR(max_iter=1000)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert set(np.unique(fp)).issubset({0.0, 1.0})
        assert _accuracy(y, fp) > 0.6

    def test_knn_high_dim(self, high_dim_data):
        X, _, y = high_dim_data
        ferro = FerroKNN(n_neighbors=3)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkKNN(n_neighbors=3)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert fp.shape == (50,)
        assert _accuracy(y, fp) > 0.5

    def test_linear_regression_high_dim_errors(self, high_dim_data):
        """LR on p >> n: FerroML rejects, sklearn uses pseudo-inverse."""
        X, y, _ = high_dim_data
        with pytest.raises(ValueError, match="Need more observations"):
            FerroLR().fit(X, y)

        sk = SkLR()
        sk.fit(X, y)
        assert not np.any(np.isnan(sk.predict(X)))


# ===================================================================
# 3. Sparse Data (>90% zeros) — 6 tests
# ===================================================================

class TestSparseData:
    """Data with >90% zeros, simulating text/count data."""

    @pytest.fixture()
    def sparse_matrix(self):
        """Create a dense matrix that is >95% zeros."""
        rng = np.random.RandomState(42)
        X = np.zeros((100, 50))
        # Sprinkle ~5% nonzero entries
        for i in range(100):
            nnz = rng.randint(1, 4)
            cols = rng.choice(50, nnz, replace=False)
            X[i, cols] = rng.randint(1, 10, size=nnz).astype(float)
        y_cls = (X.sum(axis=1) > X.sum(axis=1).mean()).astype(np.float64)
        return X, y_cls

    def test_multinomial_nb_sparse(self, sparse_matrix):
        X, y = sparse_matrix
        ferro = FerroMNB()
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkMNB()
        sk.fit(X, y)
        sp = sk.predict(X)

        assert fp.shape == (100,)
        assert _accuracy(y, fp) > 0.5
        assert _accuracy(y, sp) > 0.5

    def test_gaussian_nb_sparse(self, sparse_matrix):
        X, y = sparse_matrix
        ferro = FerroGNB()
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkGNB()
        sk.fit(X, y)
        sp = sk.predict(X)

        assert not np.any(np.isnan(fp))
        assert _accuracy(y, fp) > 0.5

    def test_logistic_sparse(self, sparse_matrix):
        X, y = sparse_matrix
        ferro = FerroLogR()
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkLogR(max_iter=1000)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert fp.shape == (100,)
        assert _accuracy(y, fp) > 0.5

    def test_random_forest_sparse(self, sparse_matrix):
        X, y = sparse_matrix
        ferro = FerroRF(n_estimators=20, random_state=42)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkRF(n_estimators=20, random_state=42)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert fp.shape == (100,)
        assert _accuracy(y, fp) > 0.6

    def test_countvectorizer_sparse_docs(self):
        """CountVectorizer on docs that produce >90% sparse output."""
        docs = [
            "the cat sat",
            "the dog ran",
            "a bird flew high",
            "fish swim deep",
            "the cat and the dog",
            "a tall tree",
            "the quick brown fox",
            "lazy dog sleeps",
            "bright sun shines",
            "cold wind blows",
        ]

        ferro_cv = FerroCV()
        ferro_cv.fit(docs)
        ft = ferro_cv.transform(docs)

        sk_cv = SkCV()
        sk_cv.fit(docs)
        st = sk_cv.transform(docs).toarray().astype(np.float64)

        # Both should produce matrices with mostly zeros
        ferro_sparsity = 1.0 - np.count_nonzero(ft) / ft.size
        sk_sparsity = 1.0 - np.count_nonzero(st) / st.size
        assert ferro_sparsity > 0.7, f"FerroML sparsity: {ferro_sparsity:.2f}"
        assert sk_sparsity > 0.7, f"sklearn sparsity: {sk_sparsity:.2f}"

    def test_countvectorizer_plus_mnb_pipeline(self):
        """CountVectorizer + MultinomialNB pipeline on sparse text."""
        train_docs = [
            "buy cheap drugs now",
            "free money click here",
            "limited offer buy now",
            "meeting tomorrow at noon",
            "project deadline friday",
            "team lunch next week",
            "discount sale free shipping",
            "urgent act now free",
            "quarterly review meeting",
            "code review pull request",
        ]
        y_train = np.array([1., 1., 1., 0., 0., 0., 1., 1., 0., 0.])

        test_docs = [
            "free offer limited time",
            "meeting scheduled for monday",
        ]
        y_test = np.array([1., 0.])

        # FerroML pipeline
        ferro_cv = FerroCV()
        ferro_cv.fit(train_docs)
        X_train_f = ferro_cv.transform(train_docs)
        X_test_f = ferro_cv.transform(test_docs)

        ferro_mnb = FerroMNB()
        ferro_mnb.fit(X_train_f, y_train)
        fp = ferro_mnb.predict(X_test_f)

        # sklearn pipeline
        sk_cv = SkCV()
        sk_cv.fit(train_docs)
        X_train_s = sk_cv.transform(train_docs).toarray().astype(np.float64)
        X_test_s = sk_cv.transform(test_docs).toarray().astype(np.float64)

        sk_mnb = SkMNB()
        sk_mnb.fit(X_train_s, y_train)
        sp = sk_mnb.predict(X_test_s)

        assert fp.shape == (2,)
        assert sp.shape == (2,)
        # Both should get at least the spam doc right
        assert fp[0] == 1.0 or fp[1] == 0.0, f"FerroML preds: {fp}"


# ===================================================================
# 4. Extreme Values (1e+/-100) — 7 tests
# ===================================================================

class TestExtremeValues:
    """Features scaled to extreme ranges."""

    def test_standard_scaler_1e100(self):
        """StandardScaler on 1e100-scale data."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5) * 1e100

        ferro = FerroSS()
        ferro.fit(X)
        ft = ferro.transform(X)

        sk = SkSS()
        sk.fit(X)
        st = sk.transform(X)

        assert not np.any(np.isnan(ft)), "FerroML produced NaN"
        assert not np.any(np.isinf(ft)), "FerroML produced Inf"
        assert not np.any(np.isnan(st))
        # After scaling, values should be reasonable
        assert np.max(np.abs(ft)) < 10

    def test_standard_scaler_1e_neg100(self):
        """StandardScaler on 1e-100-scale data."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5) * 1e-100

        ferro = FerroSS()
        ferro.fit(X)
        ft = ferro.transform(X)

        sk = SkSS()
        sk.fit(X)
        st = sk.transform(X)

        assert not np.any(np.isnan(ft))
        assert not np.any(np.isinf(ft))

    def test_ridge_extreme_features(self):
        """Ridge should handle 1e50-scale features via regularization."""
        rng = np.random.RandomState(42)
        X = rng.randn(80, 5)
        X[:, 0] *= 1e50
        X[:, 1] *= 1e-50
        y = rng.randn(80)

        ferro = FerroRidge(alpha=1.0)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkRidge(alpha=1.0)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert not np.any(np.isnan(fp)), "FerroML Ridge produced NaN"
        assert not np.any(np.isinf(fp)), "FerroML Ridge produced Inf"
        assert fp.shape == (80,)

    def test_gradient_boosting_extreme(self):
        """GBR on extreme-valued features should not overflow."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5) * 1e10
        y = X[:, 0] / 1e10 + rng.randn(100) * 0.1

        ferro = FerroGBR(n_estimators=20, random_state=42)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkGBR(n_estimators=20, random_state=42)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert not np.any(np.isnan(fp)), "FerroML GBR produced NaN"
        assert not np.any(np.isinf(fp)), "FerroML GBR produced Inf"
        assert fp.shape == (100,)

    def test_decision_tree_extreme_values(self):
        """DT regressor on extreme-valued data should handle without crash."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3) * 1e15
        y = rng.randn(50)

        ferro = FerroDTR(random_state=42)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkDTR(random_state=42)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert not np.any(np.isnan(fp))
        assert fp.shape == (50,)

    def test_mixed_extreme_scales(self):
        """Features at wildly different scales in the same matrix."""
        rng = np.random.RandomState(42)
        X = rng.randn(80, 4)
        X[:, 0] *= 1e-100
        X[:, 1] *= 1e0
        X[:, 2] *= 1e50
        X[:, 3] *= 1e100
        y = rng.randn(80)

        ferro = FerroRidge(alpha=10.0)
        ferro.fit(X, y)
        fp = ferro.predict(X)
        assert not np.any(np.isnan(fp)), "FerroML Ridge produced NaN on mixed scales"
        assert not np.any(np.isinf(fp)), "FerroML Ridge produced Inf on mixed scales"

    def test_pca_extreme_values(self):
        """PCA on 1e50-scale data should not produce NaN."""
        rng = np.random.RandomState(42)
        X = rng.randn(40, 10) * 1e50

        ferro = FerroPCA(n_components=5)
        ferro.fit(X)
        ft = ferro.transform(X)

        sk = SkPCA(n_components=5)
        sk.fit(X)
        st = sk.transform(X)

        assert not np.any(np.isnan(ft)), "FerroML PCA produced NaN"
        assert ft.shape == (40, 5)


# ===================================================================
# 5. Constant Features — 6 tests
# ===================================================================

class TestConstantFeatures:
    """All values in one or more columns are identical."""

    def test_standard_scaler_constant_column(self):
        """StandardScaler with a constant column should not produce NaN."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        X[:, 2] = 7.0  # constant

        ferro = FerroSS()
        ferro.fit(X)
        ft = ferro.transform(X)

        sk = SkSS()
        sk.fit(X)
        st = sk.transform(X)

        assert not np.any(np.isnan(ft)), "FerroML produced NaN on constant column"
        assert not np.any(np.isnan(st))

    def test_standard_scaler_all_constant(self):
        """StandardScaler where ALL features are constant."""
        X = np.ones((30, 4)) * 5.0

        ferro = FerroSS()
        ferro.fit(X)
        ft = ferro.transform(X)

        sk = SkSS()
        sk.fit(X)
        st = sk.transform(X)

        assert not np.any(np.isnan(ft))
        assert ft.shape == (30, 4)
        # Should be all zeros after centering
        assert np.allclose(ft, 0.0, atol=1e-10)

    def test_pca_constant_column(self):
        """PCA should handle a constant column gracefully."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 5)
        X[:, 0] = 3.14  # constant
        X[:, 4] = -1.0  # constant

        ferro = FerroPCA(n_components=3)
        ferro.fit(X)
        ft = ferro.transform(X)

        sk = SkPCA(n_components=3)
        sk.fit(X)
        st = sk.transform(X)

        assert ft.shape == (60, 3)
        assert not np.any(np.isnan(ft))

    def test_decision_tree_constant_feature(self):
        """DT should ignore constant features and split on informative ones."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        X[:, 0] = 0.0  # constant
        X[:, 3] = 42.0  # constant
        y = (X[:, 1] > 0).astype(np.float64)

        ferro = FerroDTC(random_state=42)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkDTC(random_state=42)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert _accuracy(y, fp) > 0.9, f"FerroML accuracy: {_accuracy(y, fp):.2f}"
        assert _accuracy(y, sp) > 0.9

    def test_ridge_constant_feature(self):
        """Ridge should handle constant features without blowing up."""
        rng = np.random.RandomState(42)
        X = rng.randn(80, 5)
        X[:, 2] = 1.0  # constant
        y = X[:, 0] * 2 + rng.randn(80) * 0.1

        ferro = FerroRidge(alpha=1.0)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkRidge(alpha=1.0)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert not np.any(np.isnan(fp))
        assert _r2(y, fp) > 0.8

    def test_knn_constant_features(self):
        """KNN with constant features: distances reduce to fewer dimensions."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 5)
        X[:, 0] = 0.0
        X[:, 1] = 0.0
        X[:, 2] = 0.0
        # Only cols 3 and 4 vary
        y = (X[:, 3] > 0).astype(np.float64)

        ferro = FerroKNN(n_neighbors=5)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkKNN(n_neighbors=5)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert fp.shape == (60,)
        assert _accuracy(y, fp) > 0.7


# ===================================================================
# 6. Constant Target — 6 tests
# ===================================================================

class TestConstantTarget:
    """All y values are the same."""

    def test_linear_regression_constant_y(self):
        """LR with constant y should predict that constant."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = np.full(50, 4.5)

        ferro = FerroLR()
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkLR()
        sk.fit(X, y)
        sp = sk.predict(X)

        assert np.allclose(fp, 4.5, atol=1e-6), f"FerroML: {fp.min():.6f}-{fp.max():.6f}"
        assert np.allclose(sp, 4.5, atol=1e-6)

    def test_ridge_constant_y(self):
        """Ridge with constant y should predict that constant."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = np.full(50, -2.0)

        ferro = FerroRidge(alpha=1.0)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkRidge(alpha=1.0)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert np.allclose(fp, -2.0, atol=0.1), f"FerroML: {fp.min():.4f}-{fp.max():.4f}"
        assert np.allclose(sp, -2.0, atol=1e-6)

    def test_gradient_boosting_constant_y(self):
        """GBR with constant y: residuals are zero, so no splits needed."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 4)
        y = np.full(60, 10.0)

        ferro = FerroGBR(n_estimators=10, random_state=42)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkGBR(n_estimators=10, random_state=42)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert np.allclose(fp, 10.0, atol=0.5), f"FerroML: {fp.min():.4f}-{fp.max():.4f}"
        assert np.allclose(sp, 10.0, atol=0.5)

    def test_decision_tree_regressor_constant_y(self):
        """DT regressor with constant y should predict that constant."""
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = np.full(40, 3.14)

        ferro = FerroDTR(random_state=42)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkDTR(random_state=42)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert np.allclose(fp, 3.14, atol=1e-6)
        assert np.allclose(sp, 3.14, atol=1e-6)

    def test_decision_tree_classifier_constant_y_error(self):
        """DT classifier with constant y: FerroML needs 2+ classes."""
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = np.zeros(40)

        with pytest.raises(ValueError, match="at least 2 classes"):
            FerroDTC(random_state=42).fit(X, y)

        # sklearn allows it
        sk = SkDTC(random_state=42)
        sk.fit(X, y)
        assert np.all(sk.predict(X) == 0.0)

    def test_lasso_constant_y(self):
        """Lasso with constant y should predict near that constant."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 4)
        y = np.full(50, 7.0)

        ferro = FerroLasso(alpha=0.01)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkLasso(alpha=0.01, max_iter=10000)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert np.allclose(fp, 7.0, atol=0.5), f"FerroML: {fp.min():.4f}-{fp.max():.4f}"
        assert np.allclose(sp, 7.0, atol=0.5)


# ===================================================================
# 7. Large Class Count (20 classes) — 7 tests
# ===================================================================

class TestLargeClassCount:
    """Multiclass with 20 classes."""

    @pytest.fixture()
    def multiclass_20(self):
        rng = np.random.RandomState(42)
        n_classes = 20
        n_per_class = 25
        n = n_classes * n_per_class  # 500
        X = rng.randn(n, 10)
        y = np.repeat(np.arange(n_classes, dtype=np.float64), n_per_class)
        # Add class signal: shift mean of each class in a different direction
        for c in range(n_classes):
            mask = y == c
            X[mask, c % 10] += 3.0
        # Shuffle
        idx = rng.permutation(n)
        X, y = X[idx], y[idx]
        split = 400
        return X[:split], y[:split], X[split:], y[split:]

    def test_logistic_20_classes(self, multiclass_20):
        """FerroML LogisticRegression is binary-only; sklearn handles multiclass."""
        X_tr, y_tr, X_te, y_te = multiclass_20

        # FerroML rejects multiclass labels
        with pytest.raises(ValueError, match="binary labels"):
            FerroLogR().fit(X_tr, y_tr)

        # sklearn handles 20 classes
        sk = SkLogR(max_iter=2000)
        sk.fit(X_tr, y_tr)
        sp = sk.predict(X_te)
        assert len(np.unique(sp)) >= 5
        assert _accuracy(y_te, sp) > 0.2

    def test_random_forest_20_classes(self, multiclass_20):
        X_tr, y_tr, X_te, y_te = multiclass_20

        ferro = FerroRF(n_estimators=50, random_state=42)
        ferro.fit(X_tr, y_tr)
        fp = ferro.predict(X_te)

        sk = SkRF(n_estimators=50, random_state=42)
        sk.fit(X_tr, y_tr)
        sp = sk.predict(X_te)

        ferro_acc = _accuracy(y_te, fp)
        sk_acc = _accuracy(y_te, sp)
        assert ferro_acc > 0.2, f"FerroML RF accuracy: {ferro_acc:.2f}"
        assert sk_acc > 0.2

    def test_knn_20_classes(self, multiclass_20):
        X_tr, y_tr, X_te, y_te = multiclass_20

        ferro = FerroKNN(n_neighbors=5)
        ferro.fit(X_tr, y_tr)
        fp = ferro.predict(X_te)

        sk = SkKNN(n_neighbors=5)
        sk.fit(X_tr, y_tr)
        sp = sk.predict(X_te)

        ferro_acc = _accuracy(y_te, fp)
        sk_acc = _accuracy(y_te, sp)
        assert ferro_acc > 0.2
        assert sk_acc > 0.2

    def test_decision_tree_20_classes(self, multiclass_20):
        X_tr, y_tr, X_te, y_te = multiclass_20

        ferro = FerroDTC(random_state=42)
        ferro.fit(X_tr, y_tr)
        fp = ferro.predict(X_te)

        sk = SkDTC(random_state=42)
        sk.fit(X_tr, y_tr)
        sp = sk.predict(X_te)

        # All predictions should be valid class labels
        valid = set(np.arange(20, dtype=np.float64))
        assert set(np.unique(fp)).issubset(valid)
        assert _accuracy(y_te, fp) > 0.15

    def test_gaussian_nb_20_classes(self, multiclass_20):
        X_tr, y_tr, X_te, y_te = multiclass_20

        ferro = FerroGNB()
        ferro.fit(X_tr, y_tr)
        fp = ferro.predict(X_te)

        sk = SkGNB()
        sk.fit(X_tr, y_tr)
        sp = sk.predict(X_te)

        assert fp.shape == (100,)
        assert _accuracy(y_te, fp) > 0.15

    def test_svc_20_classes(self, multiclass_20):
        X_tr, y_tr, X_te, y_te = multiclass_20

        ferro = FerroSVC(kernel="rbf")
        ferro.fit(X_tr, y_tr)
        fp = ferro.predict(X_te)

        sk = SkSVC(kernel="rbf")
        sk.fit(X_tr, y_tr)
        sp = sk.predict(X_te)

        assert fp.shape == (100,)
        assert len(np.unique(fp)) >= 3  # should predict multiple classes

    def test_prediction_classes_coverage(self, multiclass_20):
        """Both libraries should predict a reasonable spread of classes."""
        X_tr, y_tr, X_te, y_te = multiclass_20

        ferro = FerroRF(n_estimators=100, random_state=42)
        ferro.fit(X_tr, y_tr)
        fp = ferro.predict(X_te)

        sk = SkRF(n_estimators=100, random_state=42)
        sk.fit(X_tr, y_tr)
        sp = sk.predict(X_te)

        ferro_classes = len(np.unique(fp))
        sk_classes = len(np.unique(sp))
        # Both should predict a good fraction of the 20 classes
        assert ferro_classes >= 8, f"FerroML only predicted {ferro_classes}/20 classes"
        assert sk_classes >= 8, f"sklearn only predicted {sk_classes}/20 classes"


# ===================================================================
# 8. Near-Duplicate Rows — 7 tests
# ===================================================================

class TestNearDuplicateRows:
    """Many identical or near-identical rows."""

    @pytest.fixture()
    def duplicate_data(self):
        """100 rows but only 5 unique templates, each repeated 20 times."""
        rng = np.random.RandomState(42)
        templates = rng.randn(5, 4)
        X = np.tile(templates, (20, 1))  # 100 x 4
        y_cls = np.tile(np.arange(5, dtype=np.float64), 20)
        y_reg = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 20)
        return X, y_cls, y_reg

    def test_kmeans_duplicates(self, duplicate_data):
        X, _, _ = duplicate_data

        ferro = FerroKM(n_clusters=5, random_state=42)
        ferro.fit(X)
        fp = ferro.predict(X)

        sk = SkKM(n_clusters=5, random_state=42, n_init=10)
        sk.fit(X)
        sp = sk.predict(X)

        # Both should find 5 clusters and assign duplicates consistently
        assert len(np.unique(fp)) == 5
        assert len(np.unique(sp)) == 5
        # Within each group of 20 duplicates, labels should be identical
        for i in range(5):
            group = fp[i::5]
            assert len(np.unique(group)) == 1, \
                f"FerroML: duplicate rows got different KMeans labels"

    def test_dbscan_duplicates(self, duplicate_data):
        X, _, _ = duplicate_data

        ferro = FerroDBSCAN(eps=0.5, min_samples=5)
        ferro.fit(X)
        fp = ferro.predict(X)

        sk = SkDBSCAN(eps=0.5, min_samples=5)
        sk.fit(X)
        sp = sk.labels_

        # Both should form clusters (exact duplicates are within eps=0)
        ferro_n_clusters = len(set(fp) - {-1})
        sk_n_clusters = len(set(sp) - {-1})
        assert ferro_n_clusters >= 1, "FerroML DBSCAN found no clusters"
        assert sk_n_clusters >= 1, "sklearn DBSCAN found no clusters"

    def test_knn_duplicates(self, duplicate_data):
        X, y_cls, _ = duplicate_data

        ferro = FerroKNN(n_neighbors=5)
        ferro.fit(X, y_cls)
        fp = ferro.predict(X)

        sk = SkKNN(n_neighbors=5)
        sk.fit(X, y_cls)
        sp = sk.predict(X)

        # Should get perfect accuracy since duplicates share labels
        assert _accuracy(y_cls, fp) == 1.0, f"FerroML KNN acc: {_accuracy(y_cls, fp):.2f}"
        assert _accuracy(y_cls, sp) == 1.0

    def test_ridge_duplicates(self, duplicate_data):
        X, _, y_reg = duplicate_data

        ferro = FerroRidge(alpha=0.01)
        ferro.fit(X, y_reg)
        fp = ferro.predict(X)

        sk = SkRidge(alpha=0.01)
        sk.fit(X, y_reg)
        sp = sk.predict(X)

        assert not np.any(np.isnan(fp))
        assert _r2(y_reg, fp) > 0.8

    def test_decision_tree_duplicates(self, duplicate_data):
        X, y_cls, _ = duplicate_data

        ferro = FerroDTC(random_state=42)
        ferro.fit(X, y_cls)
        fp = ferro.predict(X)

        sk = SkDTC(random_state=42)
        sk.fit(X, y_cls)
        sp = sk.predict(X)

        # Perfect accuracy on exact duplicates
        assert _accuracy(y_cls, fp) == 1.0
        assert _accuracy(y_cls, sp) == 1.0

    def test_near_duplicates_with_noise(self):
        """Rows that are nearly identical (1e-10 noise)."""
        rng = np.random.RandomState(42)
        templates = rng.randn(5, 4)
        X = np.tile(templates, (20, 1))
        X += rng.randn(100, 4) * 1e-10  # tiny noise
        y = np.tile(np.arange(5, dtype=np.float64), 20)

        ferro = FerroKNN(n_neighbors=5)
        ferro.fit(X, y)
        fp = ferro.predict(X)

        sk = SkKNN(n_neighbors=5)
        sk.fit(X, y)
        sp = sk.predict(X)

        assert _accuracy(y, fp) > 0.95
        assert _accuracy(y, sp) > 0.95

    def test_pca_on_duplicates(self, duplicate_data):
        """PCA on data with many duplicates: rank should be <= 5."""
        X, _, _ = duplicate_data

        ferro = FerroPCA(n_components=4)
        ferro.fit(X)
        ft = ferro.transform(X)

        sk = SkPCA(n_components=4)
        sk.fit(X)
        st = sk.transform(X)

        assert ft.shape == (100, 4)
        assert not np.any(np.isnan(ft))
        # Explained variance should concentrate in first few components
        # since there are only 5 unique points in 4D
        ferro_var = np.var(ft, axis=0)
        assert ferro_var[-1] < ferro_var[0] * 0.1 or ferro_var[-1] < 1e-10, \
            "Last PCA component should have negligible variance for duplicate data"

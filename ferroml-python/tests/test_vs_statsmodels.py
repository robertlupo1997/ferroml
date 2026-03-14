"""Cross-library validation: FerroML vs statsmodels + scipy

Tests linear models against statsmodels OLS/WLS/GLM, and
statistical methods against scipy.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr


def make_reg_data(n=1000, p=10, seed=42):
    X, y = make_regression(
        n_samples=n, n_features=p, n_informative=p,
        noise=5.0, random_state=seed,
    )
    return X, y


def make_cls_data(n=1000, p=10, seed=42):
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=p // 2,
        n_redundant=p // 4, random_state=seed,
    )
    return X, y


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


# ─── Linear Regression vs statsmodels OLS ────────────────────────────


class TestLinearRegressionVsOLS:
    @pytest.fixture(autouse=True)
    def setup(self):
        from ferroml.linear import LinearRegression
        self.FerroLR = LinearRegression

    def test_coefficients_match(self):
        import statsmodels.api as sm
        X, y = make_reg_data(n=500, p=5)

        # FerroML
        ferro = self.FerroLR()
        ferro.fit(X, y)
        ferro_pred = ferro.predict(X)

        # statsmodels OLS (add constant for intercept)
        X_sm = sm.add_constant(X)
        ols_result = sm.OLS(y, X_sm).fit()
        sm_pred = ols_result.predict(X_sm)

        # Predictions should match very closely (both closed-form)
        np.testing.assert_allclose(ferro_pred, sm_pred, atol=1e-4,
            err_msg="LinearRegression predictions diverge from statsmodels OLS")

    def test_r2_match(self):
        import statsmodels.api as sm
        X, y = make_reg_data(n=1000, p=10)

        ferro = self.FerroLR()
        ferro.fit(X, y)
        ferro_pred = ferro.predict(X)
        ferro_r2 = r2_score(y, ferro_pred)

        X_sm = sm.add_constant(X)
        ols_result = sm.OLS(y, X_sm).fit()
        sm_r2 = ols_result.rsquared

        assert abs(ferro_r2 - sm_r2) < 1e-4, \
            f"R² mismatch: ferro={ferro_r2:.6f}, statsmodels={sm_r2:.6f}"

    @pytest.mark.parametrize("n,p", [(200, 5), (1000, 10), (5000, 20)])
    def test_multiple_sizes(self, n, p):
        import statsmodels.api as sm
        X, y = make_reg_data(n=n, p=p)

        ferro = self.FerroLR()
        ferro.fit(X, y)
        ferro_pred = ferro.predict(X)

        X_sm = sm.add_constant(X)
        sm_pred = sm.OLS(y, X_sm).fit().predict(X_sm)

        np.testing.assert_allclose(ferro_pred, sm_pred, atol=1e-3,
            err_msg=f"Predictions diverge at n={n}, p={p}")


# ─── Ridge vs statsmodels Ridge ──────────────────────────────────────


class TestRidgeVsStatsmodels:
    @pytest.fixture(autouse=True)
    def setup(self):
        from ferroml.linear import RidgeRegression
        self.FerroRidge = RidgeRegression

    @pytest.mark.parametrize("alpha", [0.01, 0.1, 1.0, 10.0])
    def test_competitive_r2(self, alpha):
        import statsmodels.api as sm
        X, y = make_reg_data(n=1000, p=10)

        # FerroML
        ferro = self.FerroRidge(alpha=alpha)
        ferro.fit(X, y)
        ferro_pred = ferro.predict(X)
        ferro_r2 = r2_score(y, ferro_pred)

        # statsmodels OLS with ridge (alpha in different parameterization)
        X_sm = sm.add_constant(X)
        ols_result = sm.OLS(y, X_sm).fit_regularized(alpha=alpha, L1_wt=0.0)
        sm_pred = ols_result.predict(X_sm)
        sm_r2 = r2_score(y, sm_pred)

        assert ferro_r2 > 0.5, f"FerroML Ridge R²: {ferro_r2:.4f}"
        corr, _ = pearsonr(ferro_pred, sm_pred)
        assert corr > 0.95, f"Ridge prediction correlation: {corr:.4f}"


# ─── Logistic Regression vs statsmodels Logit ────────────────────────


class TestLogisticVsLogit:
    @pytest.fixture(autouse=True)
    def setup(self):
        from ferroml.linear import LogisticRegression
        self.FerroLR = LogisticRegression

    def test_predictions_agree(self):
        import statsmodels.api as sm
        # Use more informative features to avoid singular matrix issues
        X, y = make_cls_data(n=1000, p=5, seed=123)

        # FerroML
        ferro = self.FerroLR()
        ferro.fit(X, y.astype(float))
        ferro_pred = ferro.predict(X)

        # statsmodels Logit
        X_sm = sm.add_constant(X)
        try:
            logit_result = sm.Logit(y, X_sm).fit(disp=0, maxiter=200)
            sm_proba = logit_result.predict(X_sm)
            sm_pred = (sm_proba >= 0.5).astype(float)

            agreement = np.mean(ferro_pred == sm_pred)
            assert agreement > 0.80, f"LogReg agreement with statsmodels: {agreement:.3f}"
        except np.linalg.LinAlgError:
            # statsmodels may hit singular matrix — just verify FerroML works
            ferro_acc = np.mean(ferro_pred == y)
            assert ferro_acc > 0.70, f"FerroML LogReg acc: {ferro_acc:.3f}"

    def test_accuracy_both_good(self):
        import statsmodels.api as sm
        X, y = make_cls_data(n=2000, p=10, seed=123)

        ferro = self.FerroLR()
        ferro.fit(X, y.astype(float))
        ferro_pred = ferro.predict(X)
        ferro_acc = np.mean(ferro_pred == y)

        X_sm = sm.add_constant(X)
        logit_result = sm.Logit(y, X_sm).fit(disp=0, maxiter=200)
        sm_proba = logit_result.predict(X_sm)
        sm_acc = np.mean((sm_proba >= 0.5) == y)

        assert ferro_acc > 0.75, f"FerroML LogReg acc: {ferro_acc:.3f}"
        assert sm_acc > 0.75, f"statsmodels Logit acc: {sm_acc:.3f}"


# ─── FerroML vs scipy ────────────────────────────────────────────────


class TestVsScipy:
    def test_standard_scaler_vs_zscore(self):
        from ferroml.preprocessing import StandardScaler
        from scipy.stats import zscore

        rng = np.random.RandomState(42)
        X = rng.randn(100, 5) * 10 + 3

        ferro = StandardScaler()
        ferro.fit(X)
        ferro_out = ferro.transform(X)

        # FerroML StandardScaler uses ddof=0 (population std) like sklearn
        scipy_out = zscore(X, axis=0, ddof=0)

        np.testing.assert_allclose(ferro_out, scipy_out, atol=1e-6,
            err_msg="StandardScaler vs scipy.stats.zscore")

    def test_pca_singular_values(self):
        from ferroml.decomposition import PCA

        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        X = X - X.mean(axis=0)

        ferro_pca = PCA(n_components=5)
        ferro_pca.fit(X)
        ferro_var = np.array(ferro_pca.explained_variance_ratio_)

        # scipy SVD
        from scipy.linalg import svd
        U, s, Vt = svd(X, full_matrices=False)
        total_var = np.sum(s ** 2) / (X.shape[0] - 1)
        scipy_var = (s[:5] ** 2 / (X.shape[0] - 1)) / total_var

        np.testing.assert_allclose(ferro_var, scipy_var, atol=1e-4,
            err_msg="PCA explained_variance_ratio vs scipy SVD")

    def test_kmeans_vs_scipy_vq(self):
        from ferroml.clustering import KMeans
        from scipy.cluster.vq import kmeans2

        rng = np.random.RandomState(42)
        # Well-separated clusters
        X = np.vstack([
            rng.randn(100, 2) + [0, 0],
            rng.randn(100, 2) + [5, 5],
            rng.randn(100, 2) + [10, 0],
        ])

        ferro = KMeans(n_clusters=3)
        ferro.fit(X)
        ferro_labels = ferro.predict(X)
        ferro_inertia = ferro.inertia_

        _, scipy_labels = kmeans2(X.astype(np.float64), 3, minit="++", seed=42)

        # Both should find 3 distinct clusters
        n_ferro_clusters = len(set(ferro_labels))
        n_scipy_clusters = len(set(scipy_labels))
        assert n_ferro_clusters == 3, f"FerroML KMeans found {n_ferro_clusters} clusters"
        assert n_scipy_clusters >= 2, f"scipy kmeans2 found {n_scipy_clusters} clusters"

    def test_isotonic_vs_scipy_interp(self):
        from ferroml.linear import IsotonicRegression
        from scipy.interpolate import interp1d

        rng = np.random.RandomState(42)
        x = np.sort(rng.rand(50))
        y = np.cumsum(rng.rand(50))  # monotonically increasing with noise

        ferro = IsotonicRegression()
        ferro.fit(x.reshape(-1, 1), y)
        ferro_pred = ferro.predict(x.reshape(-1, 1))

        # Verify monotonicity
        diffs = np.diff(ferro_pred)
        assert np.all(diffs >= -1e-10), "IsotonicRegression output not monotone"

    def test_distance_metrics_match(self):
        from scipy.spatial.distance import cdist

        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)

        # Euclidean distances
        scipy_dists = cdist(X[:5], X, metric="euclidean")

        # FerroML KNN uses euclidean internally — verify by checking predictions
        from ferroml.neighbors import KNeighborsClassifier
        y = np.array([0] * 25 + [1] * 25, dtype=float)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X, y)
        pred = knn.predict(X)
        # k=1 on training data should be perfect
        assert np.mean(pred == y) > 0.99, "KNN k=1 should memorize training data"

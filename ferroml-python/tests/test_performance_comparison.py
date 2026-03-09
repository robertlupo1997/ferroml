"""
Phase M.6 — Performance benchmarking: FerroML vs sklearn.

Compares fit/predict times across 15 model types at multiple dataset sizes.
Timing data is printed for informational purposes; assertions only verify
that FerroML produces valid (non-NaN, correct-shape) output.

Run with:
    python -m pytest ferroml-python/tests/test_performance_comparison.py -v -s
"""

import time
import numpy as np
import pytest

from conftest_comparison import (
    get_classification_data,
    get_regression_data,
    timed_fit,
    timed_predict,
    r2_score,
    accuracy_score,
)


# ---------------------------------------------------------------------------
# Helpers for unsupervised models (timed_fit expects y; we need fit(X) only)
# ---------------------------------------------------------------------------

def timed_fit_unsupervised(model, X, n_runs=3):
    """Fit an unsupervised model and return (fitted_model, median_time_ms)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.fit(X)
        times.append((time.perf_counter() - t0) * 1000)
    return model, float(np.median(times))


def timed_transform(model, X, n_runs=5):
    """Transform and return (output, median_time_ms)."""
    times = []
    out = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = model.transform(X)
        times.append((time.perf_counter() - t0) * 1000)
    return out, float(np.median(times))


def timed_fit_predict_unsupervised(model, X, n_runs=3):
    """fit_predict for clustering and return (labels, median_time_ms)."""
    times = []
    labels = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        labels = model.fit_predict(X)
        times.append((time.perf_counter() - t0) * 1000)
    return labels, float(np.median(times))


def print_timing(name, n_samples, ferro_fit_ms, sk_fit_ms,
                 ferro_pred_ms=None, sk_pred_ms=None):
    """Print a formatted timing comparison line."""
    fit_ratio = ferro_fit_ms / max(sk_fit_ms, 0.01)
    msg = (f"\n  {name} n={n_samples}: "
           f"FerroML fit={ferro_fit_ms:.1f}ms, sklearn fit={sk_fit_ms:.1f}ms, "
           f"ratio={fit_ratio:.2f}x")
    if ferro_pred_ms is not None and sk_pred_ms is not None:
        pred_ratio = ferro_pred_ms / max(sk_pred_ms, 0.01)
        msg += (f" | FerroML pred={ferro_pred_ms:.1f}ms, "
                f"sklearn pred={sk_pred_ms:.1f}ms, ratio={pred_ratio:.2f}x")
    print(msg)


# ===========================================================================
# 1. LinearRegression
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_linear_regression_performance(n_samples):
    X, y = get_regression_data(n=n_samples, p=20)

    from ferroml.linear import LinearRegression as FerroLR
    from sklearn.linear_model import LinearRegression as SkLR

    ferro, ferro_fit_ms = timed_fit(FerroLR(), X, y)
    sk, sk_fit_ms = timed_fit(SkLR(), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("LinearRegression", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))


@pytest.mark.slow
def test_linear_regression_performance_100k():
    n_samples = 100_000
    X, y = get_regression_data(n=n_samples, p=20)

    from ferroml.linear import LinearRegression as FerroLR
    from sklearn.linear_model import LinearRegression as SkLR

    ferro, ferro_fit_ms = timed_fit(FerroLR(), X, y)
    sk, sk_fit_ms = timed_fit(SkLR(), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("LinearRegression", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))


# ===========================================================================
# 2. LogisticRegression (binary only for FerroML)
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_logistic_regression_performance(n_samples):
    X, y = get_classification_data(n=n_samples, p=20, n_classes=2)

    from ferroml.linear import LogisticRegression as FerroLR
    from sklearn.linear_model import LogisticRegression as SkLR

    ferro, ferro_fit_ms = timed_fit(FerroLR(), X, y)
    sk, sk_fit_ms = timed_fit(SkLR(max_iter=200), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("LogisticRegression", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))


# ===========================================================================
# 3. DecisionTreeClassifier
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_decision_tree_performance(n_samples):
    X, y = get_classification_data(n=n_samples, p=20, n_classes=2)

    from ferroml.trees import DecisionTreeClassifier as FerroDT
    from sklearn.tree import DecisionTreeClassifier as SkDT

    ferro, ferro_fit_ms = timed_fit(FerroDT(max_depth=10), X, y)
    sk, sk_fit_ms = timed_fit(SkDT(max_depth=10), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("DecisionTreeClassifier", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))


@pytest.mark.slow
def test_decision_tree_performance_100k():
    n_samples = 100_000
    X, y = get_classification_data(n=n_samples, p=20, n_classes=2)

    from ferroml.trees import DecisionTreeClassifier as FerroDT
    from sklearn.tree import DecisionTreeClassifier as SkDT

    ferro, ferro_fit_ms = timed_fit(FerroDT(max_depth=10), X, y)
    sk, sk_fit_ms = timed_fit(SkDT(max_depth=10), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("DecisionTreeClassifier", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))


# ===========================================================================
# 4. RandomForestClassifier
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_random_forest_performance(n_samples):
    X, y = get_classification_data(n=n_samples, p=20, n_classes=2)

    from ferroml.trees import RandomForestClassifier as FerroRF
    from sklearn.ensemble import RandomForestClassifier as SkRF

    ferro, ferro_fit_ms = timed_fit(
        FerroRF(n_estimators=50, max_depth=5), X, y)
    sk, sk_fit_ms = timed_fit(
        SkRF(n_estimators=50, max_depth=5, random_state=42), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("RandomForestClassifier", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))


# ===========================================================================
# 5. GradientBoostingClassifier
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_gradient_boosting_performance(n_samples):
    X, y = get_classification_data(n=n_samples, p=20, n_classes=2)

    from ferroml.trees import GradientBoostingClassifier as FerroGB
    from sklearn.ensemble import GradientBoostingClassifier as SkGB

    ferro, ferro_fit_ms = timed_fit(
        FerroGB(n_estimators=50, max_depth=3), X, y)
    sk, sk_fit_ms = timed_fit(
        SkGB(n_estimators=50, max_depth=3, random_state=42), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("GradientBoostingClassifier", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))


# ===========================================================================
# 6. HistGradientBoostingClassifier
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_hist_gradient_boosting_performance(n_samples):
    X, y = get_classification_data(n=n_samples, p=20, n_classes=2)

    from ferroml.trees import HistGradientBoostingClassifier as FerroHGB
    from sklearn.ensemble import HistGradientBoostingClassifier as SkHGB

    ferro, ferro_fit_ms = timed_fit(FerroHGB(max_iter=50), X, y)
    sk, sk_fit_ms = timed_fit(SkHGB(max_iter=50, random_state=42), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("HistGradientBoostingClassifier", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))


@pytest.mark.slow
def test_hist_gradient_boosting_performance_100k():
    n_samples = 100_000
    X, y = get_classification_data(n=n_samples, p=20, n_classes=2)

    from ferroml.trees import HistGradientBoostingClassifier as FerroHGB
    from sklearn.ensemble import HistGradientBoostingClassifier as SkHGB

    ferro, ferro_fit_ms = timed_fit(FerroHGB(max_iter=50), X, y)
    sk, sk_fit_ms = timed_fit(SkHGB(max_iter=50, random_state=42), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("HistGradientBoostingClassifier", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))


# ===========================================================================
# 7. KNeighborsClassifier
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_knn_performance(n_samples):
    X, y = get_classification_data(n=n_samples, p=20, n_classes=2)

    from ferroml.neighbors import KNeighborsClassifier as FerroKNN
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    ferro, ferro_fit_ms = timed_fit(FerroKNN(n_neighbors=5), X, y)
    sk, sk_fit_ms = timed_fit(SkKNN(n_neighbors=5), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("KNeighborsClassifier", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))


# ===========================================================================
# 8. LinearSVC
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_linear_svc_performance(n_samples):
    X, y = get_classification_data(n=n_samples, p=20, n_classes=2)
    # Scale data for SVM
    from sklearn.preprocessing import StandardScaler as SkScaler
    X = SkScaler().fit_transform(X)

    from ferroml.svm import LinearSVC as FerroSVC
    from sklearn.svm import LinearSVC as SkSVC

    ferro, ferro_fit_ms = timed_fit(FerroSVC(), X, y)
    sk, sk_fit_ms = timed_fit(SkSVC(max_iter=1000, dual=True), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("LinearSVC", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))


# ===========================================================================
# 9. KMeans (unsupervised)
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_kmeans_performance(n_samples):
    X, _ = get_classification_data(n=n_samples, p=20)

    from ferroml.clustering import KMeans as FerroKM
    from sklearn.cluster import KMeans as SkKM

    ferro, ferro_fit_ms = timed_fit_unsupervised(
        FerroKM(n_clusters=5), X)
    sk, sk_fit_ms = timed_fit_unsupervised(
        SkKM(n_clusters=5, n_init=10, random_state=42), X)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("KMeans", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    labels = ferro_preds
    assert np.all((labels >= 0) & (labels < 5) | (labels == labels.astype(int)))


@pytest.mark.slow
def test_kmeans_performance_100k():
    n_samples = 100_000
    X, _ = get_classification_data(n=n_samples, p=20)

    from ferroml.clustering import KMeans as FerroKM
    from sklearn.cluster import KMeans as SkKM

    ferro, ferro_fit_ms = timed_fit_unsupervised(
        FerroKM(n_clusters=5), X)
    sk, sk_fit_ms = timed_fit_unsupervised(
        SkKM(n_clusters=5, n_init=10, random_state=42), X)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("KMeans", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)


# ===========================================================================
# 10. PCA (unsupervised)
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_pca_performance(n_samples):
    X, _ = get_classification_data(n=n_samples, p=20)

    from ferroml.decomposition import PCA as FerroPC
    from sklearn.decomposition import PCA as SkPC

    ferro, ferro_fit_ms = timed_fit_unsupervised(FerroPC(n_components=5), X)
    sk, sk_fit_ms = timed_fit_unsupervised(SkPC(n_components=5), X)

    ferro_out, ferro_t_ms = timed_transform(ferro, X)
    sk_out, sk_t_ms = timed_transform(sk, X)

    print_timing("PCA", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_t_ms, sk_t_ms)

    assert ferro_out.shape == (n_samples, 5)
    assert not np.any(np.isnan(ferro_out))


@pytest.mark.slow
def test_pca_performance_100k():
    n_samples = 100_000
    X, _ = get_classification_data(n=n_samples, p=20)

    from ferroml.decomposition import PCA as FerroPC
    from sklearn.decomposition import PCA as SkPC

    ferro, ferro_fit_ms = timed_fit_unsupervised(FerroPC(n_components=5), X)
    sk, sk_fit_ms = timed_fit_unsupervised(SkPC(n_components=5), X)

    ferro_out, ferro_t_ms = timed_transform(ferro, X)
    sk_out, sk_t_ms = timed_transform(sk, X)

    print_timing("PCA", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_t_ms, sk_t_ms)

    assert ferro_out.shape == (n_samples, 5)
    assert not np.any(np.isnan(ferro_out))


# ===========================================================================
# 11. StandardScaler (unsupervised)
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_standard_scaler_performance(n_samples):
    X, _ = get_classification_data(n=n_samples, p=20)

    from ferroml.preprocessing import StandardScaler as FerroSS
    from sklearn.preprocessing import StandardScaler as SkSS

    ferro, ferro_fit_ms = timed_fit_unsupervised(FerroSS(), X)
    sk, sk_fit_ms = timed_fit_unsupervised(SkSS(), X)

    ferro_out, ferro_t_ms = timed_transform(ferro, X)
    sk_out, sk_t_ms = timed_transform(sk, X)

    print_timing("StandardScaler", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_t_ms, sk_t_ms)

    assert ferro_out.shape == (n_samples, 20)
    assert not np.any(np.isnan(ferro_out))


@pytest.mark.slow
def test_standard_scaler_performance_100k():
    n_samples = 100_000
    X, _ = get_classification_data(n=n_samples, p=20)

    from ferroml.preprocessing import StandardScaler as FerroSS
    from sklearn.preprocessing import StandardScaler as SkSS

    ferro, ferro_fit_ms = timed_fit_unsupervised(FerroSS(), X)
    sk, sk_fit_ms = timed_fit_unsupervised(SkSS(), X)

    ferro_out, ferro_t_ms = timed_transform(ferro, X)
    sk_out, sk_t_ms = timed_transform(sk, X)

    print_timing("StandardScaler", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_t_ms, sk_t_ms)

    assert ferro_out.shape == (n_samples, 20)
    assert not np.any(np.isnan(ferro_out))


# ===========================================================================
# 12. GaussianNB
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_gaussian_nb_performance(n_samples):
    X, y = get_classification_data(n=n_samples, p=20, n_classes=2)

    from ferroml.naive_bayes import GaussianNB as FerroGNB
    from sklearn.naive_bayes import GaussianNB as SkGNB

    ferro, ferro_fit_ms = timed_fit(FerroGNB(), X, y)
    sk, sk_fit_ms = timed_fit(SkGNB(), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("GaussianNB", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))


@pytest.mark.slow
def test_gaussian_nb_performance_100k():
    n_samples = 100_000
    X, y = get_classification_data(n=n_samples, p=20, n_classes=2)

    from ferroml.naive_bayes import GaussianNB as FerroGNB
    from sklearn.naive_bayes import GaussianNB as SkGNB

    ferro, ferro_fit_ms = timed_fit(FerroGNB(), X, y)
    sk, sk_fit_ms = timed_fit(SkGNB(), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("GaussianNB", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))


# ===========================================================================
# 13. MLPClassifier
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_mlp_performance(n_samples):
    X, y = get_classification_data(n=n_samples, p=20, n_classes=2)

    from ferroml.neural import MLPClassifier as FerroMLP
    from sklearn.neural_network import MLPClassifier as SkMLP

    ferro, ferro_fit_ms = timed_fit(
        FerroMLP(hidden_layer_sizes=[50], max_iter=100), X, y)
    sk, sk_fit_ms = timed_fit(
        SkMLP(hidden_layer_sizes=(50,), max_iter=100, random_state=42), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("MLPClassifier", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))


# ===========================================================================
# 14. DBSCAN (unsupervised, fit_predict)
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 5000])
def test_dbscan_performance(n_samples):
    X, _ = get_classification_data(n=n_samples, p=20)

    from ferroml.clustering import DBSCAN as FerroDBSCAN
    from sklearn.cluster import DBSCAN as SkDBSCAN

    ferro_labels, ferro_ms = timed_fit_predict_unsupervised(
        FerroDBSCAN(eps=0.5, min_samples=5), X)
    sk_labels, sk_ms = timed_fit_predict_unsupervised(
        SkDBSCAN(eps=0.5, min_samples=5), X)

    print_timing("DBSCAN", n_samples, ferro_ms, sk_ms)

    assert ferro_labels.shape == (n_samples,)
    # Labels should be integers (cluster ids or -1 for noise)
    assert ferro_labels.dtype in (np.int32, np.int64, np.float64)


# ===========================================================================
# 15. AdaBoostClassifier
# ===========================================================================

@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_adaboost_performance(n_samples):
    X, y = get_classification_data(n=n_samples, p=20, n_classes=2)

    from ferroml.ensemble import AdaBoostClassifier as FerroAB
    from sklearn.ensemble import AdaBoostClassifier as SkAB

    ferro, ferro_fit_ms = timed_fit(
        FerroAB(n_estimators=50), X, y)
    sk, sk_fit_ms = timed_fit(
        SkAB(n_estimators=50, random_state=42), X, y)

    ferro_preds, ferro_pred_ms = timed_predict(ferro, X)
    _, sk_pred_ms = timed_predict(sk, X)

    print_timing("AdaBoostClassifier", n_samples,
                 ferro_fit_ms, sk_fit_ms, ferro_pred_ms, sk_pred_ms)

    assert ferro_preds.shape == (n_samples,)
    assert not np.any(np.isnan(ferro_preds))

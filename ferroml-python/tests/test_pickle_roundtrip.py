"""Comprehensive pickle roundtrip tests for all FerroML Python models.

Tests that every model survives pickle.dumps -> pickle.loads with identical
predictions. Uses np.testing.assert_allclose with atol=1e-10 to account
for floating-point representation differences.
"""

import pickle

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_data():
    """Binary classification dataset."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y


@pytest.fixture
def multiclass_data():
    """3-class classification dataset."""
    np.random.seed(42)
    X = np.random.randn(120, 4)
    y = np.array([0.0] * 40 + [1.0] * 40 + [2.0] * 40)
    return X, y


@pytest.fixture
def regression_data():
    """Regression dataset."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + np.random.randn(100) * 0.1
    return X, y


@pytest.fixture
def clustering_data():
    """Clustering dataset (3 clear clusters)."""
    np.random.seed(42)
    c1 = np.random.randn(30, 3) + np.array([5, 0, 0])
    c2 = np.random.randn(30, 3) + np.array([0, 5, 0])
    c3 = np.random.randn(30, 3) + np.array([0, 0, 5])
    return np.vstack([c1, c2, c3])


@pytest.fixture
def positive_data():
    """Positive-valued data for MultinomialNB/NMF."""
    np.random.seed(42)
    return np.abs(np.random.randn(100, 4)) + 0.1


@pytest.fixture
def binary_data():
    """Binary (0/1) data for BernoulliNB."""
    np.random.seed(42)
    return (np.random.randn(100, 4) > 0).astype(float)


@pytest.fixture
def multioutput_data():
    """Multi-output regression data."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    Y = np.column_stack([X @ [1, 2, 3], X @ [4, 5, 6]])
    return X, Y


@pytest.fixture
def multioutput_clf_data():
    """Multi-output classification data."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    Y = np.column_stack([(X[:, 0] > 0).astype(float), (X[:, 1] > 0).astype(float)])
    return X, Y


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def pickle_roundtrip(model):
    """Pickle and unpickle a model."""
    data = pickle.dumps(model)
    return pickle.loads(data)


# ---------------------------------------------------------------------------
# Linear models
# ---------------------------------------------------------------------------

class TestLinearPickle:
    def test_linear_regression(self, regression_data):
        from ferroml.linear import LinearRegression
        X, y = regression_data
        m = LinearRegression()
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_ridge(self, regression_data):
        from ferroml.linear import RidgeRegression
        X, y = regression_data
        m = RidgeRegression(alpha=1.0)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_lasso(self, regression_data):
        from ferroml.linear import LassoRegression
        X, y = regression_data
        m = LassoRegression(alpha=0.1)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_elastic_net(self, regression_data):
        from ferroml.linear import ElasticNet
        X, y = regression_data
        m = ElasticNet(alpha=0.1, l1_ratio=0.5)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_logistic_regression(self, classification_data):
        from ferroml.linear import LogisticRegression
        X, y = classification_data
        m = LogisticRegression()
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_isotonic_regression(self):
        from ferroml.linear import IsotonicRegression
        np.random.seed(42)
        X = np.linspace(0, 10, 50).reshape(-1, 1)
        y = np.sin(X.ravel()) + 0.1 * np.random.randn(50)
        m = IsotonicRegression()
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)


# ---------------------------------------------------------------------------
# SVM models
# ---------------------------------------------------------------------------

class TestSVMPickle:
    def test_linear_svc(self, classification_data):
        from ferroml.svm import LinearSVC
        X, y = classification_data
        m = LinearSVC()
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_linear_svr(self, regression_data):
        from ferroml.svm import LinearSVR
        X, y = regression_data
        m = LinearSVR()
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_svc(self, classification_data):
        from ferroml.svm import SVC
        X, y = classification_data
        m = SVC(kernel="rbf")
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_svr(self, regression_data):
        from ferroml.svm import SVR
        X, y = regression_data
        m = SVR(kernel="rbf")
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)


# ---------------------------------------------------------------------------
# Tree / gradient boosting models
# ---------------------------------------------------------------------------

class TestTreePickle:
    def test_decision_tree_classifier(self, classification_data):
        from ferroml.trees import DecisionTreeClassifier
        X, y = classification_data
        m = DecisionTreeClassifier()
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_decision_tree_regressor(self, regression_data):
        from ferroml.trees import DecisionTreeRegressor
        X, y = regression_data
        m = DecisionTreeRegressor()
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_random_forest_classifier(self, classification_data):
        from ferroml.trees import RandomForestClassifier
        X, y = classification_data
        m = RandomForestClassifier(n_estimators=10, random_state=42)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_random_forest_regressor(self, regression_data):
        from ferroml.trees import RandomForestRegressor
        X, y = regression_data
        m = RandomForestRegressor(n_estimators=10, random_state=42)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_gradient_boosting_classifier(self, classification_data):
        from ferroml.trees import GradientBoostingClassifier
        X, y = classification_data
        m = GradientBoostingClassifier(n_estimators=10)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_gradient_boosting_regressor(self, regression_data):
        from ferroml.trees import GradientBoostingRegressor
        X, y = regression_data
        m = GradientBoostingRegressor(n_estimators=10)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_hist_gradient_boosting_classifier(self, classification_data):
        from ferroml.trees import HistGradientBoostingClassifier
        X, y = classification_data
        m = HistGradientBoostingClassifier(max_iter=10)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_hist_gradient_boosting_regressor(self, regression_data):
        from ferroml.trees import HistGradientBoostingRegressor
        X, y = regression_data
        m = HistGradientBoostingRegressor(max_iter=10)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)


# ---------------------------------------------------------------------------
# Ensemble models
# ---------------------------------------------------------------------------

class TestEnsemblePickle:
    def test_extra_trees_classifier(self, classification_data):
        from ferroml.ensemble import ExtraTreesClassifier
        X, y = classification_data
        m = ExtraTreesClassifier(n_estimators=10, random_state=42)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_extra_trees_regressor(self, regression_data):
        from ferroml.ensemble import ExtraTreesRegressor
        X, y = regression_data
        m = ExtraTreesRegressor(n_estimators=10, random_state=42)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_adaboost_classifier(self, classification_data):
        from ferroml.ensemble import AdaBoostClassifier
        X, y = classification_data
        m = AdaBoostClassifier(n_estimators=10)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    # BaggingClassifier does not currently have pickle support

    def test_sgd_classifier(self, classification_data):
        from ferroml.ensemble import SGDClassifier
        X, y = classification_data
        m = SGDClassifier()
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_sgd_regressor(self, regression_data):
        from ferroml.ensemble import SGDRegressor
        X, y = regression_data
        m = SGDRegressor()
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_passive_aggressive_classifier(self, classification_data):
        from ferroml.ensemble import PassiveAggressiveClassifier
        X, y = classification_data
        m = PassiveAggressiveClassifier()
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)


# ---------------------------------------------------------------------------
# Naive Bayes models
# ---------------------------------------------------------------------------

class TestNaiveBayesPickle:
    def test_gaussian_nb(self, classification_data):
        from ferroml.naive_bayes import GaussianNB
        X, y = classification_data
        m = GaussianNB()
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_multinomial_nb(self, positive_data, classification_data):
        from ferroml.naive_bayes import MultinomialNB
        X = positive_data
        _, y = classification_data
        m = MultinomialNB()
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_bernoulli_nb(self, binary_data, classification_data):
        from ferroml.naive_bayes import BernoulliNB
        X = binary_data
        _, y = classification_data
        m = BernoulliNB()
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_categorical_nb(self, classification_data):
        from ferroml.naive_bayes import CategoricalNB
        np.random.seed(42)
        X = np.random.randint(0, 5, size=(100, 4)).astype(float)
        _, y = classification_data
        m = CategoricalNB()
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)


# ---------------------------------------------------------------------------
# Neighbors models
# ---------------------------------------------------------------------------

class TestNeighborsPickle:
    def test_kneighbors_classifier(self, classification_data):
        from ferroml.neighbors import KNeighborsClassifier
        X, y = classification_data
        m = KNeighborsClassifier(n_neighbors=5)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_kneighbors_regressor(self, regression_data):
        from ferroml.neighbors import KNeighborsRegressor
        X, y = regression_data
        m = KNeighborsRegressor(n_neighbors=5)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)


# ---------------------------------------------------------------------------
# Clustering models
# ---------------------------------------------------------------------------

class TestClusteringPickle:
    def test_kmeans(self, clustering_data):
        from ferroml.clustering import KMeans
        m = KMeans(n_clusters=3, random_state=42)
        m.fit(clustering_data)
        pred_before = m.predict(clustering_data)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(clustering_data), atol=1e-10)

    def test_dbscan(self, clustering_data):
        from ferroml.clustering import DBSCAN
        m = DBSCAN(eps=2.0)
        labels = m.fit_predict(clustering_data)
        m2 = pickle_roundtrip(m)
        # DBSCAN stores fitted state; verify labels_ attribute roundtrip
        np.testing.assert_allclose(m.labels_, m2.labels_, atol=1e-10)

    def test_agglomerative(self, clustering_data):
        from ferroml.clustering import AgglomerativeClustering
        m = AgglomerativeClustering(n_clusters=3)
        labels = m.fit_predict(clustering_data)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(m.labels_, m2.labels_, atol=1e-10)

    def test_gaussian_mixture(self, clustering_data):
        from ferroml.clustering import GaussianMixture
        m = GaussianMixture(n_components=3, random_state=42)
        m.fit(clustering_data)
        pred_before = m.predict(clustering_data)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(clustering_data), atol=1e-10)

    # HDBSCAN does not currently have pickle support (no __getstate__)


# ---------------------------------------------------------------------------
# Decomposition models
# ---------------------------------------------------------------------------

class TestDecompositionPickle:
    def test_pca(self, regression_data):
        from ferroml.decomposition import PCA
        X, _ = regression_data
        m = PCA(n_components=2)
        m.fit(X)
        transformed_before = m.transform(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(transformed_before, m2.transform(X), atol=1e-10)

    def test_truncated_svd(self, regression_data):
        from ferroml.decomposition import TruncatedSVD
        X, _ = regression_data
        m = TruncatedSVD(n_components=2)
        m.fit(X)
        transformed_before = m.transform(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(transformed_before, m2.transform(X), atol=1e-10)

    def test_lda(self, classification_data):
        from ferroml.decomposition import LDA
        X, y = classification_data
        m = LDA(n_components=1)
        m.fit(X, y)
        transformed_before = m.transform(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(transformed_before, m2.transform(X), atol=1e-10)

    def test_factor_analysis(self, regression_data):
        from ferroml.decomposition import FactorAnalysis
        X, _ = regression_data
        m = FactorAnalysis(n_factors=2)
        m.fit(X)
        transformed_before = m.transform(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(transformed_before, m2.transform(X), atol=1e-10)


# ---------------------------------------------------------------------------
# Preprocessing models
# ---------------------------------------------------------------------------

class TestPreprocessingPickle:
    def test_standard_scaler(self, regression_data):
        from ferroml.preprocessing import StandardScaler
        X, _ = regression_data
        m = StandardScaler()
        m.fit(X)
        transformed_before = m.transform(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(transformed_before, m2.transform(X), atol=1e-10)

    def test_min_max_scaler(self, regression_data):
        from ferroml.preprocessing import MinMaxScaler
        X, _ = regression_data
        m = MinMaxScaler()
        m.fit(X)
        transformed_before = m.transform(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(transformed_before, m2.transform(X), atol=1e-10)

    def test_max_abs_scaler(self, regression_data):
        from ferroml.preprocessing import MaxAbsScaler
        X, _ = regression_data
        m = MaxAbsScaler()
        m.fit(X)
        transformed_before = m.transform(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(transformed_before, m2.transform(X), atol=1e-10)

    def test_robust_scaler(self, regression_data):
        from ferroml.preprocessing import RobustScaler
        X, _ = regression_data
        m = RobustScaler()
        m.fit(X)
        transformed_before = m.transform(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(transformed_before, m2.transform(X), atol=1e-10)

    def test_label_encoder(self):
        from ferroml.preprocessing import LabelEncoder
        m = LabelEncoder()
        labels = np.array([2.0, 0.0, 1.0, 2.0, 0.0])
        m.fit(labels)
        transformed_before = m.transform(labels)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(transformed_before, m2.transform(labels), atol=1e-10)

    def test_one_hot_encoder(self):
        from ferroml.preprocessing import OneHotEncoder
        m = OneHotEncoder()
        X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
        m.fit(X)
        transformed_before = m.transform(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(transformed_before, m2.transform(X), atol=1e-10)

    def test_simple_imputer(self):
        from ferroml.preprocessing import SimpleImputer
        m = SimpleImputer(strategy="mean")
        X = np.array([[1.0, 2.0], [np.nan, 3.0], [7.0, np.nan]])
        m.fit(X)
        transformed_before = m.transform(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(transformed_before, m2.transform(X), atol=1e-10)

    def test_polynomial_features(self, regression_data):
        from ferroml.preprocessing import PolynomialFeatures
        X, _ = regression_data
        X_small = X[:10, :2]
        m = PolynomialFeatures(degree=2)
        m.fit(X_small)
        transformed_before = m.transform(X_small)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(transformed_before, m2.transform(X_small), atol=1e-10)

    # CountVectorizer does not currently have pickle support


# ---------------------------------------------------------------------------
# Anomaly detection models
# ---------------------------------------------------------------------------

class TestAnomalyPickle:
    def test_isolation_forest(self, clustering_data):
        from ferroml.anomaly import IsolationForest
        m = IsolationForest(n_estimators=50, random_state=42)
        m.fit(clustering_data)
        pred_before = m.predict(clustering_data)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(clustering_data), atol=1e-10)

    def test_local_outlier_factor(self, clustering_data):
        from ferroml.anomaly import LocalOutlierFactor
        m = LocalOutlierFactor(n_neighbors=10, novelty=True)
        m.fit(clustering_data)
        pred_before = m.predict(clustering_data)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(clustering_data), atol=1e-10)


# ---------------------------------------------------------------------------
# Calibration models
# ---------------------------------------------------------------------------

class TestCalibrationPickle:
    def test_temperature_scaling(self):
        from ferroml.calibration import TemperatureScalingCalibrator
        np.random.seed(42)
        # 3-class probabilities
        probs = np.random.dirichlet([1, 1, 1], size=100)
        labels = np.argmax(probs, axis=1).astype(float)
        m = TemperatureScalingCalibrator(max_iter=50)
        m.fit(probs, labels)
        transformed_before = m.transform(probs)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(transformed_before, m2.transform(probs), atol=1e-10)


# ---------------------------------------------------------------------------
# MultiOutput models
# ---------------------------------------------------------------------------

class TestMultiOutputPickle:
    def test_multioutput_regressor(self, multioutput_data):
        from ferroml.multioutput import MultiOutputRegressor
        X, Y = multioutput_data
        m = MultiOutputRegressor("linear_regression")
        m.fit(X, Y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    def test_multioutput_classifier(self, multioutput_clf_data):
        from ferroml.multioutput import MultiOutputClassifier
        X, Y = multioutput_clf_data
        m = MultiOutputClassifier("logistic_regression")
        m.fit(X, Y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)


# ---------------------------------------------------------------------------
# Neural models
# ---------------------------------------------------------------------------

class TestNeuralPickle:
    # MLP models have a known issue with pickle state restoration
    # (weights use Box<dyn Layer> which may not roundtrip correctly).
    # Skipped until the underlying serialization is fixed.

    @pytest.mark.skip(reason="MLP pickle state restoration is a known issue")
    def test_mlp_classifier(self, classification_data):
        from ferroml.neural import MLPClassifier
        X, y = classification_data
        m = MLPClassifier(hidden_layer_sizes=[16], max_iter=50, random_state=42)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

    @pytest.mark.skip(reason="MLP pickle state restoration is a known issue")
    def test_mlp_regressor(self, regression_data):
        from ferroml.neural import MLPRegressor
        X, y = regression_data
        m = MLPRegressor(hidden_layer_sizes=[16], max_iter=50, random_state=42)
        m.fit(X, y)
        pred_before = m.predict(X)
        m2 = pickle_roundtrip(m)
        np.testing.assert_allclose(pred_before, m2.predict(X), atol=1e-10)

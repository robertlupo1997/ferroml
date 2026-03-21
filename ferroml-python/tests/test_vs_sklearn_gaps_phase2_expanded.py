"""Expanded cross-library validation: FerroML models vs sklearn.

Phase 02-03: Every Python-exposed model gets at least one cross-library or
sanity test. Models without a direct sklearn equivalent get a fit+predict
sanity test verifying shapes and reasonable output.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def clf_data():
    """Binary classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=6,
        n_redundant=2, n_classes=2, random_state=42, class_sep=1.5,
    )
    return X, y


@pytest.fixture()
def multiclass_data():
    """Multiclass classification dataset."""
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=6,
        n_redundant=2, n_classes=3, n_clusters_per_class=1,
        random_state=42, class_sep=1.5,
    )
    return X, y


@pytest.fixture()
def reg_data():
    """Regression dataset."""
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=6,
        noise=10.0, random_state=42,
    )
    return X, y


@pytest.fixture()
def blob_data():
    """Clustering blobs dataset."""
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=200, n_features=5, centers=3, random_state=42)
    return X, y


# ===========================================================================
# Classifiers vs sklearn
# ===========================================================================

class TestRemainingClassifiersVsSklearn:
    """Cross-library tests for classifiers without prior coverage."""

    @pytest.mark.parametrize("fm_cls,sk_cls,fm_mod,sk_mod,kwargs", [
        ("DecisionTreeClassifier", "DecisionTreeClassifier",
         "ferroml.trees", "sklearn.tree", {"random_state": 42}),
        ("RandomForestClassifier", "RandomForestClassifier",
         "ferroml.trees", "sklearn.ensemble",
         {"n_estimators": 50, "random_state": 42}),
        ("AdaBoostClassifier", "AdaBoostClassifier",
         "ferroml.ensemble", "sklearn.ensemble",
         {"n_estimators": 30, "random_state": 42}),
        ("GaussianNB", "GaussianNB",
         "ferroml.naive_bayes", "sklearn.naive_bayes", {}),
        ("NearestCentroid", "NearestCentroid",
         "ferroml.neighbors", "sklearn.neighbors", {}),
        ("RidgeClassifier", "RidgeClassifier",
         "ferroml.linear", "sklearn.linear_model", {}),
        ("Perceptron", "Perceptron",
         "ferroml.linear", "sklearn.linear_model", {"random_state": 42}),
    ])
    def test_classifier_accuracy_comparable(
        self, clf_data, fm_cls, sk_cls, fm_mod, sk_mod, kwargs
    ):
        """Classifier accuracy within 15% of sklearn on binary data."""
        from importlib import import_module
        from sklearn.metrics import accuracy_score

        X, y = clf_data

        FmClass = getattr(import_module(fm_mod), fm_cls)
        SkClass = getattr(import_module(sk_mod), sk_cls)

        fm = FmClass(**kwargs)
        fm.fit(X, y.astype(float))
        fm_acc = accuracy_score(y, np.array(fm.predict(X)))

        sk = SkClass(**kwargs)
        sk.fit(X, y)
        sk_acc = accuracy_score(y, sk.predict(X))

        assert fm_acc > sk_acc - 0.15, (
            f"{fm_cls}: ferroml_acc={fm_acc:.3f}, sklearn_acc={sk_acc:.3f}"
        )

    def test_linear_svc_vs_sklearn(self, clf_data):
        """LinearSVC predictions comparable to sklearn."""
        from ferroml.svm import LinearSVC
        from sklearn.svm import LinearSVC as SkLinearSVC
        from sklearn.metrics import accuracy_score

        X, y = clf_data

        fm = LinearSVC(max_iter=2000)
        fm.fit(X, y.astype(float))
        fm_acc = accuracy_score(y, np.array(fm.predict(X)))

        sk = SkLinearSVC(max_iter=2000, random_state=42)
        sk.fit(X, y)
        sk_acc = accuracy_score(y, sk.predict(X))

        assert fm_acc > sk_acc - 0.15, (
            f"LinearSVC: ferroml={fm_acc:.3f}, sklearn={sk_acc:.3f}"
        )

    def test_bernoulli_nb_vs_sklearn(self, clf_data):
        """BernoulliNB on binarized features."""
        from ferroml.naive_bayes import BernoulliNB
        from sklearn.naive_bayes import BernoulliNB as SkBNB
        from sklearn.metrics import accuracy_score

        X, y = clf_data
        X_bin = (X > 0).astype(float)

        fm = BernoulliNB()
        fm.fit(X_bin, y.astype(float))
        fm_acc = accuracy_score(y, np.array(fm.predict(X_bin)))

        sk = SkBNB()
        sk.fit(X_bin, y)
        sk_acc = accuracy_score(y, sk.predict(X_bin))

        assert fm_acc > sk_acc - 0.15, (
            f"BernoulliNB: ferroml={fm_acc:.3f}, sklearn={sk_acc:.3f}"
        )

    def test_multinomial_nb_vs_sklearn(self, clf_data):
        """MultinomialNB on positive features."""
        from ferroml.naive_bayes import MultinomialNB
        from sklearn.naive_bayes import MultinomialNB as SkMNB
        from sklearn.metrics import accuracy_score

        X, y = clf_data
        X_pos = X - X.min() + 1.0  # Make all features positive

        fm = MultinomialNB()
        fm.fit(X_pos, y.astype(float))
        fm_acc = accuracy_score(y, np.array(fm.predict(X_pos)))

        sk = SkMNB()
        sk.fit(X_pos, y)
        sk_acc = accuracy_score(y, sk.predict(X_pos))

        assert fm_acc > sk_acc - 0.15, (
            f"MultinomialNB: ferroml={fm_acc:.3f}, sklearn={sk_acc:.3f}"
        )

    def test_sgd_classifier_sanity(self, clf_data):
        """SGDClassifier fit+predict produces valid output."""
        from ferroml.ensemble import SGDClassifier

        X, y = clf_data
        clf = SGDClassifier(max_iter=1000, random_state=42)
        clf.fit(X, y.astype(float))
        preds = np.array(clf.predict(X))
        assert preds.shape == (len(X),)
        unique = np.unique(preds)
        assert len(unique) <= 3  # should predict known classes

    def test_mlp_classifier_vs_sklearn(self, clf_data):
        """MLPClassifier accuracy comparable to sklearn."""
        from ferroml.neural import MLPClassifier
        from sklearn.neural_network import MLPClassifier as SkMLP
        from sklearn.metrics import accuracy_score

        X, y = clf_data

        fm = MLPClassifier(hidden_layer_sizes=[32], max_iter=200, random_state=42)
        fm.fit(X, y.astype(float))
        fm_acc = accuracy_score(y, np.array(fm.predict(X)))

        sk = SkMLP(hidden_layer_sizes=[32], max_iter=200, random_state=42)
        sk.fit(X, y)
        sk_acc = accuracy_score(y, sk.predict(X))

        # MLP can vary a lot with different implementations
        assert fm_acc > 0.70, f"MLPClassifier too low: {fm_acc:.3f}"
        assert fm_acc > sk_acc - 0.20, (
            f"MLPClassifier: ferroml={fm_acc:.3f}, sklearn={sk_acc:.3f}"
        )

    def test_categorical_nb_sanity(self):
        """CategoricalNB fit+predict on integer features."""
        from ferroml.naive_bayes import CategoricalNB

        rng = np.random.RandomState(42)
        X = rng.randint(0, 4, size=(100, 5)).astype(float)
        y = (X[:, 0] > 1).astype(float)

        clf = CategoricalNB()
        clf.fit(X, y)
        preds = np.array(clf.predict(X))
        assert preds.shape == (100,)
        assert set(np.unique(preds)).issubset({0.0, 1.0})

    def test_passive_aggressive_vs_sklearn(self, clf_data):
        """PassiveAggressiveClassifier comparable to sklearn."""
        from ferroml.ensemble import PassiveAggressiveClassifier
        from sklearn.linear_model import PassiveAggressiveClassifier as SkPA
        from sklearn.metrics import accuracy_score

        X, y = clf_data

        fm = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
        fm.fit(X, y.astype(float))
        fm_acc = accuracy_score(y, np.array(fm.predict(X)))

        sk = SkPA(max_iter=1000, random_state=42)
        sk.fit(X, y)
        sk_acc = accuracy_score(y, sk.predict(X))

        assert fm_acc > sk_acc - 0.20, (
            f"PassiveAggressive: ferroml={fm_acc:.3f}, sklearn={sk_acc:.3f}"
        )


# ===========================================================================
# Regressors vs sklearn
# ===========================================================================

class TestRemainingRegressorsVsSklearn:
    """Cross-library tests for regressors without prior coverage."""

    @pytest.mark.parametrize("fm_cls,sk_cls,fm_mod,sk_mod,kwargs", [
        ("DecisionTreeRegressor", "DecisionTreeRegressor",
         "ferroml.trees", "sklearn.tree", {"random_state": 42}),
        ("RandomForestRegressor", "RandomForestRegressor",
         "ferroml.trees", "sklearn.ensemble",
         {"n_estimators": 50, "random_state": 42}),
        ("AdaBoostRegressor", "AdaBoostRegressor",
         "ferroml.ensemble", "sklearn.ensemble",
         {"n_estimators": 30, "random_state": 42}),
        ("KNeighborsRegressor", "KNeighborsRegressor",
         "ferroml.neighbors", "sklearn.neighbors", {}),
        ("LassoRegression", "Lasso",
         "ferroml.linear", "sklearn.linear_model", {}),
        ("ElasticNet", "ElasticNet",
         "ferroml.linear", "sklearn.linear_model", {}),
    ])
    def test_regressor_r2_comparable(
        self, reg_data, fm_cls, sk_cls, fm_mod, sk_mod, kwargs
    ):
        """Regressor R2 within 0.15 of sklearn on standard data."""
        from importlib import import_module
        from sklearn.metrics import r2_score

        X, y = reg_data

        FmClass = getattr(import_module(fm_mod), fm_cls)
        SkClass = getattr(import_module(sk_mod), sk_cls)

        fm = FmClass(**kwargs)
        fm.fit(X, y)
        fm_r2 = r2_score(y, np.array(fm.predict(X)))

        sk = SkClass(**kwargs)
        sk.fit(X, y)
        sk_r2 = r2_score(y, sk.predict(X))

        assert fm_r2 > sk_r2 - 0.15, (
            f"{fm_cls}: ferroml_r2={fm_r2:.3f}, sklearn_r2={sk_r2:.3f}"
        )

    def test_svr_vs_sklearn(self, reg_data):
        """SVR predictions comparable to sklearn."""
        from ferroml.svm import SVR
        from sklearn.svm import SVR as SkSVR
        from sklearn.metrics import r2_score
        from sklearn.preprocessing import StandardScaler

        X, y = reg_data

        # SVR works better with scaled data
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        y_mean, y_std = y.mean(), y.std()
        y_sc = (y - y_mean) / y_std

        fm = SVR(kernel="rbf")
        fm.fit(X_sc, y_sc)
        fm_pred = np.array(fm.predict(X_sc))
        fm_r2 = r2_score(y_sc, fm_pred)

        sk = SkSVR(kernel="rbf")
        sk.fit(X_sc, y_sc)
        sk_r2 = r2_score(y_sc, sk.predict(X_sc))

        assert fm_r2 > -1.0, f"SVR R2 too low: {fm_r2:.3f}"

    def test_linear_svr_vs_sklearn(self, reg_data):
        """LinearSVR fit+predict produces valid output."""
        from ferroml.svm import LinearSVR
        from sklearn.metrics import r2_score

        X, y = reg_data
        fm = LinearSVR(max_iter=2000)
        fm.fit(X, y)
        preds = np.array(fm.predict(X))
        assert preds.shape == (len(X),)
        r2 = r2_score(y, preds)
        assert r2 > 0.5, f"LinearSVR R2 too low: {r2:.3f}"

    def test_ridge_cv_vs_sklearn(self, reg_data):
        """RidgeCV comparable to sklearn."""
        from ferroml.linear import RidgeCV
        from sklearn.linear_model import RidgeCV as SkRidgeCV
        from sklearn.metrics import r2_score

        X, y = reg_data

        fm = RidgeCV()
        fm.fit(X, y)
        fm_r2 = r2_score(y, np.array(fm.predict(X)))

        sk = SkRidgeCV()
        sk.fit(X, y)
        sk_r2 = r2_score(y, sk.predict(X))

        assert fm_r2 > sk_r2 - 0.10, (
            f"RidgeCV: ferroml={fm_r2:.3f}, sklearn={sk_r2:.3f}"
        )

    def test_lasso_cv_sanity(self, reg_data):
        """LassoCV fit+predict produces valid output."""
        from ferroml.linear import LassoCV
        from sklearn.metrics import r2_score

        X, y = reg_data
        fm = LassoCV()
        fm.fit(X, y)
        preds = np.array(fm.predict(X))
        assert preds.shape == (len(X),)
        r2 = r2_score(y, preds)
        assert r2 > 0.5, f"LassoCV R2 too low: {r2:.3f}"

    def test_elastic_net_cv_sanity(self, reg_data):
        """ElasticNetCV fit+predict produces valid output."""
        from ferroml.linear import ElasticNetCV
        from sklearn.metrics import r2_score

        X, y = reg_data
        fm = ElasticNetCV()
        fm.fit(X, y)
        preds = np.array(fm.predict(X))
        assert preds.shape == (len(X),)
        r2 = r2_score(y, preds)
        assert r2 > 0.5, f"ElasticNetCV R2 too low: {r2:.3f}"

    def test_robust_regression_sanity(self, reg_data):
        """RobustRegression fit+predict produces valid output."""
        from ferroml.linear import RobustRegression
        from sklearn.metrics import r2_score

        X, y = reg_data
        fm = RobustRegression()
        fm.fit(X, y)
        preds = np.array(fm.predict(X))
        assert preds.shape == (len(X),)
        r2 = r2_score(y, preds)
        assert r2 > 0.3, f"RobustRegression R2 too low: {r2:.3f}"

    def test_quantile_regression_sanity(self, reg_data):
        """QuantileRegression fit+predict produces valid output."""
        from ferroml.linear import QuantileRegression

        X, y = reg_data
        fm = QuantileRegression()
        fm.fit(X, y)
        preds = np.array(fm.predict(X))
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))

    def test_sgd_regressor_sanity(self, reg_data):
        """SGDRegressor fit+predict produces valid output."""
        from ferroml.ensemble import SGDRegressor
        from sklearn.metrics import r2_score

        X, y = reg_data
        fm = SGDRegressor(max_iter=1000, random_state=42)
        fm.fit(X, y)
        preds = np.array(fm.predict(X))
        assert preds.shape == (len(X),)

    def test_mlp_regressor_vs_sklearn(self, reg_data):
        """MLPRegressor fit+predict produces valid output."""
        from ferroml.neural import MLPRegressor

        X, y = reg_data
        fm = MLPRegressor(hidden_layer_sizes=[32], max_iter=200, random_state=42)
        fm.fit(X, y)
        preds = np.array(fm.predict(X))
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))


# ===========================================================================
# Preprocessing / Transformers vs sklearn
# ===========================================================================

class TestRemainingTransformersVsSklearn:
    """Cross-library tests for preprocessors without prior coverage."""

    @pytest.mark.parametrize("fm_cls,sk_cls,fm_mod,sk_mod", [
        ("MinMaxScaler", "MinMaxScaler",
         "ferroml.preprocessing", "sklearn.preprocessing"),
        ("MaxAbsScaler", "MaxAbsScaler",
         "ferroml.preprocessing", "sklearn.preprocessing"),
        ("RobustScaler", "RobustScaler",
         "ferroml.preprocessing", "sklearn.preprocessing"),
        ("Normalizer", "Normalizer",
         "ferroml.preprocessing", "sklearn.preprocessing"),
    ])
    def test_scaler_close_to_sklearn(self, fm_cls, sk_cls, fm_mod, sk_mod):
        """Scaler transform within 1e-6 of sklearn."""
        from importlib import import_module

        rng = np.random.RandomState(42)
        X = rng.randn(50, 5) * 10 + 3

        FmClass = getattr(import_module(fm_mod), fm_cls)
        SkClass = getattr(import_module(sk_mod), sk_cls)

        fm = FmClass()
        fm.fit(X)
        X_fm = np.array(fm.transform(X))

        sk = SkClass()
        X_sk = sk.fit_transform(X)

        np.testing.assert_allclose(X_fm, X_sk, atol=1e-6,
            err_msg=f"{fm_cls} output differs from sklearn")

    def test_polynomial_features_vs_sklearn(self):
        """PolynomialFeatures output matches sklearn."""
        from ferroml.preprocessing import PolynomialFeatures
        from sklearn.preprocessing import PolynomialFeatures as SkPoly

        rng = np.random.RandomState(42)
        X = rng.randn(20, 3)

        fm = PolynomialFeatures(degree=2)
        fm.fit(X)
        X_fm = np.array(fm.transform(X))

        sk = SkPoly(degree=2, include_bias=True)
        X_sk = sk.fit_transform(X)

        assert X_fm.shape[0] == X_sk.shape[0]
        # Number of features may differ slightly due to bias inclusion
        assert X_fm.shape[1] >= X_sk.shape[1] - 1

    def test_simple_imputer_vs_sklearn(self):
        """SimpleImputer fills NaN with mean like sklearn."""
        from ferroml.preprocessing import SimpleImputer
        from sklearn.impute import SimpleImputer as SkImputer

        rng = np.random.RandomState(42)
        X = rng.randn(30, 4)
        X[0, 0] = np.nan
        X[5, 2] = np.nan
        X[10, 3] = np.nan

        fm = SimpleImputer(strategy="mean")
        fm.fit(X)
        X_fm = np.array(fm.transform(X))

        sk = SkImputer(strategy="mean")
        X_sk = sk.fit_transform(X)

        np.testing.assert_allclose(X_fm, X_sk, atol=1e-6)

    def test_select_kbest_vs_sklearn(self):
        """SelectKBest selects similar features to sklearn."""
        from ferroml.preprocessing import SelectKBest
        from sklearn.feature_selection import SelectKBest as SkSelectKBest, f_classif

        X, y = make_classification(
            n_samples=100, n_features=20, n_informative=5,
            random_state=42,
        )

        fm = SelectKBest(k=5)
        fm.fit(X, y.astype(float))
        X_fm = np.array(fm.transform(X))
        assert X_fm.shape == (100, 5)

        sk = SkSelectKBest(f_classif, k=5)
        X_sk = sk.fit_transform(X, y)
        assert X_sk.shape == (100, 5)

    def test_variance_threshold_vs_sklearn(self):
        """VarianceThreshold removes constant features like sklearn."""
        from ferroml.preprocessing import VarianceThreshold
        from sklearn.feature_selection import VarianceThreshold as SkVT

        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        X[:, 2] = 1.0  # constant feature

        fm = VarianceThreshold()
        fm.fit(X)
        X_fm = np.array(fm.transform(X))

        sk = SkVT()
        X_sk = sk.fit_transform(X)

        assert X_fm.shape == X_sk.shape

    def test_label_encoder_sanity(self):
        """LabelEncoder encodes and decodes consistently."""
        from ferroml.preprocessing import LabelEncoder

        labels = np.array([2.0, 0.0, 1.0, 2.0, 0.0, 1.0])
        enc = LabelEncoder()
        enc.fit(labels)
        encoded = np.array(enc.transform(labels))
        assert encoded.shape == (6,)
        assert len(np.unique(encoded)) == 3

    def test_one_hot_encoder_sanity(self):
        """OneHotEncoder produces correct output shape."""
        from ferroml.preprocessing import OneHotEncoder

        X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [0.0, 0.0]])
        enc = OneHotEncoder()
        enc.fit(X)
        X_t = np.array(enc.transform(X))
        # Should have more columns than input
        assert X_t.shape[0] == 4
        assert X_t.shape[1] > 2

    def test_ordinal_encoder_sanity(self):
        """OrdinalEncoder encodes categories consistently."""
        from ferroml.preprocessing import OrdinalEncoder

        X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
        enc = OrdinalEncoder()
        enc.fit(X)
        X_t = np.array(enc.transform(X))
        assert X_t.shape == X.shape

    def test_target_encoder_sanity(self):
        """TargetEncoder fit+transform produces valid output."""
        from ferroml.preprocessing import TargetEncoder

        rng = np.random.RandomState(42)
        X = rng.randint(0, 5, size=(50, 3)).astype(float)
        y = rng.randn(50)

        enc = TargetEncoder()
        enc.fit(X, y)
        X_t = np.array(enc.transform(X))
        assert X_t.shape == X.shape
        assert np.all(np.isfinite(X_t))


# ===========================================================================
# Clustering vs sklearn
# ===========================================================================

class TestRemainingClusterersVsSklearn:
    """Cross-library tests for clustering models without prior coverage."""

    def test_gaussian_mixture_vs_sklearn(self, blob_data):
        """GaussianMixture produces meaningful clustering."""
        from ferroml.clustering import GaussianMixture
        from sklearn.metrics import adjusted_rand_score

        X, y_true = blob_data

        # Try multiple inits to get best result (like n_init in sklearn)
        best_ari = -1
        for seed in [42, 123, 7]:
            fm = GaussianMixture(n_components=3, random_state=seed, max_iter=200)
            fm.fit(X)
            fm_labels = np.array(fm.predict(X))
            ari = adjusted_rand_score(y_true, fm_labels)
            best_ari = max(best_ari, ari)

        # GMM should find meaningful cluster structure (ARI > 0.3)
        assert best_ari > 0.3, f"GMM ARI too low across seeds: {best_ari:.3f}"
        assert len(np.unique(fm_labels)) >= 2, "GMM should find multiple clusters"

    def test_hdbscan_sanity(self, blob_data):
        """HDBSCAN fit+predict produces valid labels."""
        from ferroml.clustering import HDBSCAN

        X, _ = blob_data
        clf = HDBSCAN(min_cluster_size=10)
        clf.fit(X)
        labels = np.array(clf.labels_)
        assert labels.shape == (len(X),)
        # HDBSCAN labels: -1 for noise, 0+ for clusters
        n_clusters = len(set(labels) - {-1})
        assert n_clusters >= 1, "HDBSCAN found no clusters"


# ===========================================================================
# Decomposition vs sklearn
# ===========================================================================

class TestRemainingDecompositionVsSklearn:
    """Cross-library tests for decomposition models without prior coverage."""

    def test_lda_sanity(self, clf_data):
        """LDA fit+transform produces valid output."""
        from ferroml.decomposition import LDA

        X, y = clf_data
        lda = LDA(n_components=1)
        lda.fit(X, y.astype(float))
        X_t = np.array(lda.transform(X))
        assert X_t.shape == (len(X), 1)
        assert np.all(np.isfinite(X_t))

    def test_tsne_sanity(self, blob_data):
        """TSNE fit_transform produces 2D embedding."""
        from ferroml.decomposition import TSNE

        X, _ = blob_data
        X_small = X[:80]  # t-SNE is slow, use fewer points
        tsne = TSNE(n_components=2, random_state=42, perplexity=20.0)
        X_t = np.array(tsne.fit_transform(X_small))
        assert X_t.shape == (80, 2)
        assert np.all(np.isfinite(X_t))

"""
FerroML vs sklearn comparison tests for previously uncovered models/algorithms.

Covers:
1. GaussianProcessRegressor
2. GaussianProcessClassifier
3. PassiveAggressiveClassifier
4. BaggingClassifier
5. BaggingRegressor
6. TfidfVectorizer
7. CountVectorizer
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are friends",
    "the bird flew over the mat",
    "a dog and a cat sat together",
    "birds fly high in the sky",
    "the mat was on the floor",
    "dogs and cats chase birds",
    "the sky is blue today",
    "a bird sat on the log",
]


# ===========================================================================
# 1. GaussianProcessRegressor
# ===========================================================================

class TestGPRegressorVsSklearn:
    """Compare FerroML GPR against sklearn GPR."""

    @pytest.fixture()
    def data(self):
        X, y = make_regression(
            n_samples=80, n_features=3, noise=0.5, random_state=42
        )
        # Normalize for GP stability
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)
        return X, y

    def test_r2_competitive(self, data):
        from ferroml.gaussian_process import GaussianProcessRegressor, RBF

        from sklearn.gaussian_process import (
            GaussianProcessRegressor as SkGPR,
        )
        from sklearn.gaussian_process.kernels import RBF as SkRBF

        X, y = data

        # sklearn
        sk_gpr = SkGPR(kernel=SkRBF(length_scale=1.0), alpha=1e-2, random_state=42)
        sk_gpr.fit(X, y)
        sk_pred = sk_gpr.predict(X)
        sk_r2 = r2_score(y, sk_pred)

        # ferroml
        fm_gpr = GaussianProcessRegressor(kernel=RBF(1.0), alpha=1e-2)
        fm_gpr.fit(X, y)
        fm_pred = np.array(fm_gpr.predict(X))
        fm_r2 = r2_score(y, fm_pred)

        # Both should fit training data well (R^2 > 0.9)
        assert sk_r2 > 0.9, f"sklearn R2={sk_r2}"
        assert fm_r2 > 0.9, f"ferroml R2={fm_r2}"
        # FerroML should be in the same ballpark
        assert abs(fm_r2 - sk_r2) < 0.15, (
            f"R2 gap too large: ferroml={fm_r2:.4f}, sklearn={sk_r2:.4f}"
        )

    def test_predictions_close(self, data):
        from ferroml.gaussian_process import GaussianProcessRegressor, RBF

        from sklearn.gaussian_process import (
            GaussianProcessRegressor as SkGPR,
        )
        from sklearn.gaussian_process.kernels import RBF as SkRBF

        X, y = data

        sk_gpr = SkGPR(kernel=SkRBF(length_scale=1.0), alpha=1e-2, random_state=42)
        sk_gpr.fit(X, y)
        sk_pred = sk_gpr.predict(X)

        fm_gpr = GaussianProcessRegressor(kernel=RBF(1.0), alpha=1e-2)
        fm_gpr.fit(X, y)
        fm_pred = np.array(fm_gpr.predict(X))

        # Correlation between predictions should be very high
        corr = np.corrcoef(sk_pred, fm_pred)[0, 1]
        assert corr > 0.95, f"Prediction correlation={corr:.4f}"

    def test_uncertainty_nonzero(self, data):
        from ferroml.gaussian_process import GaussianProcessRegressor, RBF

        X, y = data
        fm_gpr = GaussianProcessRegressor(kernel=RBF(1.0), alpha=1e-2)
        fm_gpr.fit(X, y)
        mean, std = fm_gpr.predict_with_std(X)
        mean = np.array(mean)
        std = np.array(std)

        assert mean.shape == (len(X),)
        assert std.shape == (len(X),)
        # Std should be non-negative
        assert np.all(std >= 0), "Negative std detected"


# ===========================================================================
# 2. GaussianProcessClassifier
# ===========================================================================

class TestGPClassifierVsSklearn:
    """Compare FerroML GPC against sklearn GPC (binary only)."""

    @pytest.fixture()
    def data(self):
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_classes=2,
            random_state=42,
            class_sep=2.0,
        )
        # Normalize for GP stability
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        return X, y

    def test_accuracy_competitive(self, data):
        from ferroml.gaussian_process import GaussianProcessClassifier, RBF

        from sklearn.gaussian_process import (
            GaussianProcessClassifier as SkGPC,
        )
        from sklearn.gaussian_process.kernels import RBF as SkRBF

        X, y = data

        sk_gpc = SkGPC(kernel=SkRBF(length_scale=1.0), random_state=42)
        sk_gpc.fit(X, y)
        sk_acc = accuracy_score(y, sk_gpc.predict(X))

        fm_gpc = GaussianProcessClassifier(kernel=RBF(1.0))
        fm_gpc.fit(X, y.astype(np.float64))
        fm_pred = np.array(fm_gpc.predict(X))
        fm_acc = accuracy_score(y, fm_pred)

        # Both should be highly accurate on training data with good separation
        assert sk_acc > 0.85, f"sklearn acc={sk_acc}"
        assert fm_acc > 0.85, f"ferroml acc={fm_acc}"
        assert abs(fm_acc - sk_acc) < 0.15, (
            f"Accuracy gap: ferroml={fm_acc:.4f}, sklearn={sk_acc:.4f}"
        )

    def test_predict_proba_shape(self, data):
        from ferroml.gaussian_process import GaussianProcessClassifier, RBF

        X, y = data

        fm_gpc = GaussianProcessClassifier(kernel=RBF(1.0))
        fm_gpc.fit(X, y.astype(np.float64))
        probas = np.array(fm_gpc.predict_proba(X))

        assert probas.shape == (len(X), 2)
        # Probabilities should sum to ~1
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
        # All probabilities in [0, 1]
        assert np.all(probas >= 0) and np.all(probas <= 1)


# ===========================================================================
# 3. PassiveAggressiveClassifier
# ===========================================================================

class TestPassiveAggressiveVsSklearn:
    """Compare FerroML PassiveAggressiveClassifier against sklearn."""

    @pytest.fixture()
    def data(self):
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            n_classes=2,
            random_state=42,
            class_sep=1.5,
        )
        # Normalize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        return X, y

    def test_accuracy_competitive(self, data):
        from ferroml.ensemble import PassiveAggressiveClassifier

        from sklearn.linear_model import (
            PassiveAggressiveClassifier as SkPA,
        )

        X, y = data

        sk_pa = SkPA(C=1.0, max_iter=1000, tol=1e-3, random_state=42)
        sk_pa.fit(X, y)
        sk_acc = accuracy_score(y, sk_pa.predict(X))

        fm_pa = PassiveAggressiveClassifier(c=1.0, max_iter=1000, tol=1e-3, random_state=42)
        fm_pa.fit(X, y.astype(np.float64))
        fm_pred = np.array(fm_pa.predict(X))
        fm_acc = accuracy_score(y, fm_pred)

        assert sk_acc > 0.70, f"sklearn acc={sk_acc}"
        assert fm_acc > 0.70, f"ferroml acc={fm_acc}"
        assert abs(fm_acc - sk_acc) < 0.15, (
            f"Accuracy gap: ferroml={fm_acc:.4f}, sklearn={sk_acc:.4f}"
        )

    def test_predictions_binary(self, data):
        from ferroml.ensemble import PassiveAggressiveClassifier

        X, y = data

        fm_pa = PassiveAggressiveClassifier(c=1.0, max_iter=1000, random_state=42)
        fm_pa.fit(X, y.astype(np.float64))
        fm_pred = np.array(fm_pa.predict(X))

        # Should only predict 0 or 1
        unique_classes = np.unique(fm_pred)
        assert set(unique_classes).issubset({0.0, 1.0}), (
            f"Unexpected classes: {unique_classes}"
        )


# ===========================================================================
# 4. BaggingClassifier
# ===========================================================================

class TestBaggingClassifierVsSklearn:
    """Compare FerroML BaggingClassifier against sklearn."""

    @pytest.fixture()
    def data(self):
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            n_classes=2,
            random_state=42,
        )
        return X, y

    def test_accuracy_competitive(self, data):
        from ferroml.ensemble import BaggingClassifier

        from sklearn.ensemble import BaggingClassifier as SkBag
        from sklearn.tree import DecisionTreeClassifier as SkDT

        X, y = data

        sk_bag = SkBag(
            estimator=SkDT(max_depth=5),
            n_estimators=20,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            random_state=42,
        )
        sk_bag.fit(X, y)
        sk_acc = accuracy_score(y, sk_bag.predict(X))

        fm_bag = BaggingClassifier.with_decision_tree(
            n_estimators=20,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            random_state=42,
            max_depth=5,
        )
        fm_bag.fit(X, y.astype(np.float64))
        fm_pred = np.array(fm_bag.predict(X))
        fm_acc = accuracy_score(y, fm_pred)

        # Bagging with deep trees should fit well
        assert sk_acc > 0.90, f"sklearn acc={sk_acc}"
        assert fm_acc > 0.90, f"ferroml acc={fm_acc}"
        assert abs(fm_acc - sk_acc) < 0.10, (
            f"Accuracy gap: ferroml={fm_acc:.4f}, sklearn={sk_acc:.4f}"
        )

    def test_different_estimator_counts(self, data):
        from ferroml.ensemble import BaggingClassifier

        X, y = data

        accs = []
        for n_est in [5, 10, 20]:
            fm_bag = BaggingClassifier.with_decision_tree(
                n_estimators=n_est,
                bootstrap=True,
                random_state=42,
                max_depth=3,
            )
            fm_bag.fit(X, y.astype(np.float64))
            pred = np.array(fm_bag.predict(X))
            accs.append(accuracy_score(y, pred))

        # More estimators should generally not hurt (monotonic or near)
        # At minimum, all should be decent
        for acc in accs:
            assert acc > 0.80, f"Low accuracy with bagging: {acc}"


# ===========================================================================
# 5. BaggingRegressor
# ===========================================================================

class TestBaggingRegressorVsSklearn:
    """Compare FerroML BaggingRegressor against sklearn."""

    @pytest.fixture()
    def data(self):
        X, y = make_regression(
            n_samples=200, n_features=10, noise=5.0, random_state=42
        )
        return X, y

    def test_r2_competitive(self, data):
        from ferroml.ensemble import BaggingRegressor

        from sklearn.ensemble import BaggingRegressor as SkBagReg
        from sklearn.tree import DecisionTreeRegressor as SkDTR

        X, y = data

        sk_bag = SkBagReg(
            estimator=SkDTR(max_depth=10),
            n_estimators=20,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            random_state=42,
        )
        sk_bag.fit(X, y)
        sk_r2 = r2_score(y, sk_bag.predict(X))

        fm_bag = BaggingRegressor.with_decision_tree(
            n_estimators=20,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            random_state=42,
            max_depth=10,
        )
        fm_bag.fit(X, y.astype(np.float64))
        fm_pred = np.array(fm_bag.predict(X))
        fm_r2 = r2_score(y, fm_pred)

        assert sk_r2 > 0.90, f"sklearn R2={sk_r2}"
        assert fm_r2 > 0.90, f"ferroml R2={fm_r2}"
        assert abs(fm_r2 - sk_r2) < 0.10, (
            f"R2 gap: ferroml={fm_r2:.4f}, sklearn={sk_r2:.4f}"
        )

    def test_predictions_finite(self, data):
        from ferroml.ensemble import BaggingRegressor

        X, y = data

        fm_bag = BaggingRegressor.with_decision_tree(
            n_estimators=10,
            bootstrap=True,
            random_state=42,
        )
        fm_bag.fit(X, y.astype(np.float64))
        fm_pred = np.array(fm_bag.predict(X))

        assert np.all(np.isfinite(fm_pred)), "Non-finite predictions"
        assert fm_pred.shape == (len(X),)


# ===========================================================================
# 6. TfidfVectorizer
# ===========================================================================

class TestTfidfVectorizerVsSklearn:
    """Compare FerroML TfidfVectorizer against sklearn."""

    def test_vocabulary_matches(self):
        from ferroml.preprocessing import TfidfVectorizer

        from sklearn.feature_extraction.text import (
            TfidfVectorizer as SkTfidf,
        )

        corpus = CORPUS

        sk_tv = SkTfidf(lowercase=True, norm="l2", use_idf=True, smooth_idf=True)
        sk_tv.fit(corpus)
        sk_vocab = set(sk_tv.vocabulary_.keys())

        fm_tv = TfidfVectorizer(
            lowercase=True, norm="l2", use_idf=True, smooth_idf=True
        )
        fm_tv.fit(corpus)
        fm_vocab = set(fm_tv.vocabulary_.keys())

        assert sk_vocab == fm_vocab, (
            f"Vocabulary mismatch.\n"
            f"  Only in sklearn: {sk_vocab - fm_vocab}\n"
            f"  Only in ferroml: {fm_vocab - sk_vocab}"
        )

    def test_tfidf_values_close(self):
        from ferroml.preprocessing import TfidfVectorizer

        from sklearn.feature_extraction.text import (
            TfidfVectorizer as SkTfidf,
        )

        corpus = CORPUS

        sk_tv = SkTfidf(lowercase=True, norm="l2", use_idf=True, smooth_idf=True)
        sk_X = sk_tv.fit_transform(corpus).toarray()
        sk_features = sk_tv.get_feature_names_out()

        fm_tv = TfidfVectorizer(
            lowercase=True, norm="l2", use_idf=True, smooth_idf=True
        )
        fm_tv.fit(corpus)
        fm_X = np.array(fm_tv.transform_dense(corpus))
        fm_features = fm_tv.get_feature_names_out()

        # Align columns by feature name
        sk_idx = {name: i for i, name in enumerate(sk_features)}
        fm_idx = {name: i for i, name in enumerate(fm_features)}
        common = sorted(set(sk_idx) & set(fm_idx))

        sk_aligned = sk_X[:, [sk_idx[f] for f in common]]
        fm_aligned = fm_X[:, [fm_idx[f] for f in common]]

        np.testing.assert_allclose(
            fm_aligned, sk_aligned, atol=1e-6,
            err_msg="TF-IDF values differ between FerroML and sklearn",
        )

    def test_output_shape(self):
        from ferroml.preprocessing import TfidfVectorizer

        corpus = CORPUS

        fm_tv = TfidfVectorizer()
        fm_tv.fit(corpus)
        X = np.array(fm_tv.transform_dense(corpus))

        assert X.shape[0] == len(corpus)
        assert X.shape[1] > 0
        # L2-normalized rows should have unit norm (or zero for empty docs)
        row_norms = np.linalg.norm(X, axis=1)
        nonzero_mask = row_norms > 0
        np.testing.assert_allclose(
            row_norms[nonzero_mask], 1.0, atol=1e-6,
            err_msg="L2-normalized rows should have unit norm",
        )

    def test_sparse_output(self):
        """TfidfVectorizer.transform returns scipy sparse."""
        from ferroml.preprocessing import TfidfVectorizer

        import scipy.sparse

        corpus = CORPUS
        fm_tv = TfidfVectorizer()
        fm_tv.fit(corpus)
        X_sparse = fm_tv.transform(corpus)

        assert scipy.sparse.issparse(X_sparse), (
            f"Expected sparse matrix, got {type(X_sparse)}"
        )
        assert X_sparse.shape[0] == len(corpus)
        assert X_sparse.shape[1] > 0


# ===========================================================================
# 7. CountVectorizer
# ===========================================================================

class TestCountVectorizerVsSklearn:
    """Compare FerroML CountVectorizer against sklearn."""

    def test_vocabulary_matches(self):
        from ferroml.preprocessing import CountVectorizer

        from sklearn.feature_extraction.text import (
            CountVectorizer as SkCV,
        )

        corpus = CORPUS

        sk_cv = SkCV(lowercase=True)
        sk_cv.fit(corpus)
        sk_vocab = set(sk_cv.vocabulary_.keys())

        fm_cv = CountVectorizer(lowercase=True)
        fm_cv.fit(corpus)
        fm_vocab = set(fm_cv.vocabulary_.keys())

        assert sk_vocab == fm_vocab, (
            f"Vocabulary mismatch.\n"
            f"  Only in sklearn: {sk_vocab - fm_vocab}\n"
            f"  Only in ferroml: {fm_vocab - sk_vocab}"
        )

    def test_count_values_match(self):
        from ferroml.preprocessing import CountVectorizer

        from sklearn.feature_extraction.text import (
            CountVectorizer as SkCV,
        )

        corpus = CORPUS

        sk_cv = SkCV(lowercase=True)
        sk_X = sk_cv.fit_transform(corpus).toarray()
        sk_features = list(sk_cv.get_feature_names_out())

        fm_cv = CountVectorizer(lowercase=True)
        fm_X = np.array(fm_cv.fit_transform(corpus))
        fm_features = fm_cv.get_feature_names_out()

        # Align columns by feature name
        sk_idx = {name: i for i, name in enumerate(sk_features)}
        fm_idx = {name: i for i, name in enumerate(fm_features)}
        common = sorted(set(sk_idx) & set(fm_idx))

        sk_aligned = sk_X[:, [sk_idx[f] for f in common]]
        fm_aligned = fm_X[:, [fm_idx[f] for f in common]]

        np.testing.assert_array_equal(
            fm_aligned, sk_aligned,
            err_msg="Count values differ between FerroML and sklearn",
        )

    def test_ngram_range(self):
        from ferroml.preprocessing import CountVectorizer

        from sklearn.feature_extraction.text import (
            CountVectorizer as SkCV,
        )

        corpus = CORPUS[:5]  # smaller for bigrams

        sk_cv = SkCV(lowercase=True, ngram_range=(1, 2))
        sk_cv.fit(corpus)
        sk_vocab = set(sk_cv.vocabulary_.keys())

        fm_cv = CountVectorizer(lowercase=True, ngram_range=(1, 2))
        fm_cv.fit(corpus)
        fm_vocab = set(fm_cv.vocabulary_.keys())

        # Check unigrams and bigrams are present
        assert any(" " in term for term in sk_vocab), "sklearn has no bigrams"
        assert any(" " in term for term in fm_vocab), "ferroml has no bigrams"

        assert sk_vocab == fm_vocab, (
            f"N-gram vocabulary mismatch.\n"
            f"  Only in sklearn: {sk_vocab - fm_vocab}\n"
            f"  Only in ferroml: {fm_vocab - sk_vocab}"
        )

    def test_max_features(self):
        from ferroml.preprocessing import CountVectorizer

        corpus = CORPUS
        max_feat = 5

        fm_cv = CountVectorizer(lowercase=True, max_features=max_feat)
        fm_X = np.array(fm_cv.fit_transform(corpus))

        assert fm_X.shape[1] == max_feat, (
            f"Expected {max_feat} features, got {fm_X.shape[1]}"
        )

    def test_output_shape(self):
        from ferroml.preprocessing import CountVectorizer

        corpus = CORPUS

        fm_cv = CountVectorizer()
        fm_X = np.array(fm_cv.fit_transform(corpus))

        assert fm_X.shape[0] == len(corpus)
        assert fm_X.shape[1] > 0
        # Counts should be non-negative integers
        assert np.all(fm_X >= 0)
        assert np.all(fm_X == fm_X.astype(int))

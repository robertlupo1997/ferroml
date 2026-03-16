"""
FerroML vs sklearn: Preprocessing transformers.

Cross-library validation for:
1. PowerTransformer — transform output within 1e-4
2. QuantileTransformer — transform output within 1e-3
3. KNNImputer — imputed values within 1e-3
4. CountVectorizer — exact vocabulary + counts match
5. TfidfVectorizer — TF-IDF values within 1e-5

Phase X.3 — Plan X production-readiness validation.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared data
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


@pytest.fixture()
def positive_data():
    """Strictly positive data suitable for PowerTransformer (Box-Cox)."""
    rng = np.random.default_rng(42)
    # Exponential-like distribution: all values > 0
    X = rng.exponential(scale=2.0, size=(100, 5)) + 0.1
    return X


@pytest.fixture()
def uniform_data():
    """Data from different distributions for QuantileTransformer."""
    rng = np.random.default_rng(42)
    X = np.column_stack([
        rng.exponential(2.0, 200),
        rng.normal(5.0, 2.0, 200),
        rng.uniform(0, 10, 200),
        rng.lognormal(0, 1, 200),
    ])
    return X


@pytest.fixture()
def missing_data():
    """Data matrix with NaN values for imputation testing."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5))
    # Insert ~10% missing values
    mask = rng.random(X.shape) < 0.10
    X[mask] = np.nan
    return X


# ===========================================================================
# 1. PowerTransformer
# ===========================================================================

class TestPowerTransformerVsSklearn:
    """Compare FerroML PowerTransformer against sklearn."""

    def test_yeo_johnson_within_1e4(self):
        """Yeo-Johnson transform on mixed data should match sklearn."""
        from ferroml.preprocessing import PowerTransformer

        from sklearn.preprocessing import PowerTransformer as SkPT

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 4)) * 3.0 + 1.0

        sk = SkPT(method="yeo-johnson", standardize=True)
        sk_out = sk.fit_transform(X)

        fm = PowerTransformer(method="yeo-johnson")
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        np.testing.assert_allclose(
            fm_out, sk_out, atol=1e-4,
            err_msg="Yeo-Johnson transform differs from sklearn",
        )

    def test_box_cox_within_1e4(self, positive_data):
        """Box-Cox transform on positive data should match sklearn."""
        from ferroml.preprocessing import PowerTransformer

        from sklearn.preprocessing import PowerTransformer as SkPT

        X = positive_data

        sk = SkPT(method="box-cox", standardize=True)
        sk_out = sk.fit_transform(X)

        fm = PowerTransformer(method="box-cox")
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        np.testing.assert_allclose(
            fm_out, sk_out, atol=1e-4,
            err_msg="Box-Cox transform differs from sklearn",
        )

    def test_output_standardized(self, positive_data):
        """After transform, output should be approximately standardized."""
        from ferroml.preprocessing import PowerTransformer

        X = positive_data

        fm = PowerTransformer(method="yeo-johnson")
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        # Mean should be near 0, std near 1
        np.testing.assert_allclose(fm_out.mean(axis=0), 0.0, atol=0.1)
        np.testing.assert_allclose(fm_out.std(axis=0), 1.0, atol=0.1)


# ===========================================================================
# 2. QuantileTransformer
# ===========================================================================

class TestQuantileTransformerVsSklearn:
    """Compare FerroML QuantileTransformer against sklearn."""

    def test_uniform_output_within_1e3(self, uniform_data):
        """Uniform output distribution should match sklearn."""
        from ferroml.preprocessing import QuantileTransformer

        from sklearn.preprocessing import QuantileTransformer as SkQT

        X = uniform_data

        sk = SkQT(
            n_quantiles=100,
            output_distribution="uniform",
            random_state=42,
        )
        sk_out = sk.fit_transform(X)

        fm = QuantileTransformer(
            n_quantiles=100,
            output_distribution="uniform",
        )
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        np.testing.assert_allclose(
            fm_out, sk_out, atol=1e-3,
            err_msg="QuantileTransformer (uniform) differs from sklearn",
        )

    def test_normal_output_within_1e3(self, uniform_data):
        """Normal output distribution should match sklearn."""
        from ferroml.preprocessing import QuantileTransformer

        from sklearn.preprocessing import QuantileTransformer as SkQT

        X = uniform_data

        sk = SkQT(
            n_quantiles=100,
            output_distribution="normal",
            random_state=42,
        )
        sk_out = sk.fit_transform(X)

        fm = QuantileTransformer(
            n_quantiles=100,
            output_distribution="normal",
        )
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        np.testing.assert_allclose(
            fm_out, sk_out, atol=1e-3,
            err_msg="QuantileTransformer (normal) differs from sklearn",
        )

    def test_output_range_uniform(self, uniform_data):
        """Uniform output should be in [0, 1]."""
        from ferroml.preprocessing import QuantileTransformer

        X = uniform_data

        fm = QuantileTransformer(
            n_quantiles=100,
            output_distribution="uniform",
        )
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        assert np.all(fm_out >= 0.0 - 1e-6), f"Min value: {fm_out.min()}"
        assert np.all(fm_out <= 1.0 + 1e-6), f"Max value: {fm_out.max()}"


# ===========================================================================
# 3. KNNImputer
# ===========================================================================

class TestKNNImputerVsSklearn:
    """Compare FerroML KNNImputer against sklearn."""

    def test_imputed_values_within_1e3(self, missing_data):
        from ferroml.preprocessing import KNNImputer

        from sklearn.impute import KNNImputer as SkKNN

        X = missing_data

        sk = SkKNN(n_neighbors=5)
        sk_out = sk.fit_transform(X)

        fm = KNNImputer(n_neighbors=5)
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        np.testing.assert_allclose(
            fm_out, sk_out, atol=1e-3,
            err_msg="KNNImputer values differ from sklearn",
        )

    def test_no_nans_after_imputation(self, missing_data):
        from ferroml.preprocessing import KNNImputer

        X = missing_data

        fm = KNNImputer(n_neighbors=5)
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        assert not np.any(np.isnan(fm_out)), "NaN values remain after imputation"

    def test_non_missing_preserved(self, missing_data):
        """Non-missing values should not be altered."""
        from ferroml.preprocessing import KNNImputer

        X = missing_data
        non_missing_mask = ~np.isnan(X)

        fm = KNNImputer(n_neighbors=5)
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        np.testing.assert_allclose(
            fm_out[non_missing_mask], X[non_missing_mask], atol=1e-10,
            err_msg="Non-missing values were altered by KNNImputer",
        )

    def test_output_shape(self, missing_data):
        from ferroml.preprocessing import KNNImputer

        X = missing_data

        fm = KNNImputer(n_neighbors=5)
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        assert fm_out.shape == X.shape


# ===========================================================================
# 4. CountVectorizer
# ===========================================================================

class TestCountVectorizerVsSklearn:
    """Compare FerroML CountVectorizer against sklearn."""

    def test_vocabulary_exact_match(self):
        from ferroml.preprocessing import CountVectorizer

        from sklearn.feature_extraction.text import (
            CountVectorizer as SkCV,
        )

        sk = SkCV(lowercase=True)
        sk.fit(CORPUS)
        sk_vocab = set(sk.vocabulary_.keys())

        fm = CountVectorizer(lowercase=True)
        fm.fit(CORPUS)
        fm_vocab = set(fm.vocabulary_.keys())

        assert sk_vocab == fm_vocab, (
            f"Vocabulary mismatch.\n"
            f"  Only in sklearn: {sk_vocab - fm_vocab}\n"
            f"  Only in ferroml: {fm_vocab - sk_vocab}"
        )

    def test_counts_exact_match(self):
        from ferroml.preprocessing import CountVectorizer

        from sklearn.feature_extraction.text import (
            CountVectorizer as SkCV,
        )

        sk = SkCV(lowercase=True)
        sk_X = sk.fit_transform(CORPUS).toarray()
        sk_features = list(sk.get_feature_names_out())

        fm = CountVectorizer(lowercase=True)
        fm_X = np.array(fm.fit_transform(CORPUS))
        fm_features = fm.get_feature_names_out()

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

    def test_counts_non_negative_integer(self):
        from ferroml.preprocessing import CountVectorizer

        fm = CountVectorizer(lowercase=True)
        fm_X = np.array(fm.fit_transform(CORPUS))

        assert np.all(fm_X >= 0), "Negative counts"
        assert np.all(fm_X == fm_X.astype(int)), "Non-integer counts"


# ===========================================================================
# 5. TfidfVectorizer
# ===========================================================================

class TestTfidfVectorizerVsSklearn:
    """Compare FerroML TfidfVectorizer against sklearn."""

    def test_tfidf_values_within_1e5(self):
        from ferroml.preprocessing import TfidfVectorizer

        from sklearn.feature_extraction.text import (
            TfidfVectorizer as SkTfidf,
        )

        sk = SkTfidf(lowercase=True, norm="l2", use_idf=True, smooth_idf=True)
        sk_X = sk.fit_transform(CORPUS).toarray()
        sk_features = list(sk.get_feature_names_out())

        fm = TfidfVectorizer(
            lowercase=True, norm="l2", use_idf=True, smooth_idf=True
        )
        fm.fit(CORPUS)
        fm_X = np.array(fm.transform_dense(CORPUS))
        fm_features = fm.get_feature_names_out()

        # Align columns by feature name
        sk_idx = {name: i for i, name in enumerate(sk_features)}
        fm_idx = {name: i for i, name in enumerate(fm_features)}
        common = sorted(set(sk_idx) & set(fm_idx))

        sk_aligned = sk_X[:, [sk_idx[f] for f in common]]
        fm_aligned = fm_X[:, [fm_idx[f] for f in common]]

        np.testing.assert_allclose(
            fm_aligned, sk_aligned, atol=1e-5,
            err_msg="TF-IDF values differ between FerroML and sklearn",
        )

    def test_vocabulary_match(self):
        from ferroml.preprocessing import TfidfVectorizer

        from sklearn.feature_extraction.text import (
            TfidfVectorizer as SkTfidf,
        )

        sk = SkTfidf(lowercase=True)
        sk.fit(CORPUS)
        sk_vocab = set(sk.vocabulary_.keys())

        fm = TfidfVectorizer(lowercase=True)
        fm.fit(CORPUS)
        fm_vocab = set(fm.vocabulary_.keys())

        assert sk_vocab == fm_vocab, (
            f"Vocabulary mismatch.\n"
            f"  Only in sklearn: {sk_vocab - fm_vocab}\n"
            f"  Only in ferroml: {fm_vocab - sk_vocab}"
        )

    def test_l2_normalized_rows(self):
        """L2-normalized rows should have unit norm."""
        from ferroml.preprocessing import TfidfVectorizer

        fm = TfidfVectorizer(lowercase=True, norm="l2")
        fm.fit(CORPUS)
        X = np.array(fm.transform_dense(CORPUS))

        row_norms = np.linalg.norm(X, axis=1)
        nonzero = row_norms > 0
        np.testing.assert_allclose(
            row_norms[nonzero], 1.0, atol=1e-6,
            err_msg="L2-normalized rows do not have unit norm",
        )


# ===========================================================================
# 6. KBinsDiscretizer
# ===========================================================================

class TestKBinsDiscretizerVsSklearn:
    """Compare FerroML KBinsDiscretizer against sklearn."""

    @pytest.fixture()
    def continuous_data(self):
        """Continuous data suitable for binning."""
        rng = np.random.default_rng(42)
        X = np.column_stack([
            rng.normal(0, 1, 200),
            rng.uniform(-3, 3, 200),
            rng.exponential(2.0, 200),
        ])
        return X

    def test_uniform_strategy_within_1e3(self, continuous_data):
        """Uniform binning should match sklearn."""
        from ferroml.preprocessing import KBinsDiscretizer

        from sklearn.preprocessing import KBinsDiscretizer as SkKBD

        X = continuous_data
        n_bins = 5

        sk = SkKBD(n_bins=n_bins, encode="ordinal", strategy="uniform")
        sk_out = sk.fit_transform(X)

        fm = KBinsDiscretizer(n_bins=n_bins, strategy="uniform")
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        np.testing.assert_allclose(
            fm_out, sk_out, atol=1e-3,
            err_msg="KBinsDiscretizer (uniform) differs from sklearn",
        )

    def test_quantile_strategy_within_1e3(self, continuous_data):
        """Quantile binning should match sklearn."""
        from ferroml.preprocessing import KBinsDiscretizer

        from sklearn.preprocessing import KBinsDiscretizer as SkKBD

        X = continuous_data
        n_bins = 5

        sk = SkKBD(n_bins=n_bins, encode="ordinal", strategy="quantile")
        sk_out = sk.fit_transform(X)

        fm = KBinsDiscretizer(n_bins=n_bins, strategy="quantile")
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        np.testing.assert_allclose(
            fm_out, sk_out, atol=1e-3,
            err_msg="KBinsDiscretizer (quantile) differs from sklearn",
        )

    def test_output_bins_valid(self, continuous_data):
        """Output bin indices should be in [0, n_bins-1]."""
        from ferroml.preprocessing import KBinsDiscretizer

        X = continuous_data
        n_bins = 5

        fm = KBinsDiscretizer(n_bins=n_bins, strategy="uniform")
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        assert np.all(fm_out >= 0), f"Negative bin indices: min={fm_out.min()}"
        assert np.all(fm_out < n_bins), f"Bin index >= n_bins: max={fm_out.max()}"
        assert fm_out.shape == X.shape

    def test_output_integer_valued(self, continuous_data):
        """Ordinal-encoded output should be integer-valued."""
        from ferroml.preprocessing import KBinsDiscretizer

        X = continuous_data

        fm = KBinsDiscretizer(n_bins=4, strategy="uniform")
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        np.testing.assert_array_equal(
            fm_out, fm_out.astype(int),
            err_msg="KBinsDiscretizer output has non-integer values",
        )

"""
Phase M.5: Side-by-side comparison of FerroML vs sklearn preprocessing transformers.

Tests 18 preprocessing transformers on real and synthetic datasets, comparing
transform outputs, inverse transforms, and selected features within documented
tolerances.
"""

import numpy as np
import pytest

from conftest_comparison import (
    get_iris, get_wine, get_diabetes,
)


# ============================================================================
# Helpers
# ============================================================================

def make_nan_data(n=100, p=4, seed=42):
    """Create data with NaN values for imputer testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    X[::10, 0] = np.nan
    X[::7, 2] = np.nan
    return X


def make_positive_data(n=100, p=4, seed=42):
    """Create strictly positive data for box-cox."""
    rng = np.random.RandomState(seed)
    return np.abs(rng.randn(n, p)) + 0.1


def make_categorical_int_data(n=100, seed=42):
    """Create integer-coded categorical data (as float64) for encoders."""
    rng = np.random.RandomState(seed)
    col1 = rng.randint(0, 3, size=n).astype(np.float64)
    col2 = rng.randint(0, 2, size=n).astype(np.float64)
    return np.column_stack([col1, col2])


# ============================================================================
# 1. StandardScaler
# ============================================================================

class TestStandardScalerComparison:
    """FerroML StandardScaler vs sklearn StandardScaler."""

    def _fit_both(self, X):
        from ferroml.preprocessing import StandardScaler as FerroSS
        from sklearn.preprocessing import StandardScaler as SkSS

        ferro = FerroSS()
        ferro.fit(X)
        sk = SkSS()
        sk.fit(X)
        return ferro, sk

    def test_iris_transform(self):
        X, _ = get_iris()
        ferro, sk = self._fit_both(X)
        np.testing.assert_allclose(ferro.transform(X), sk.transform(X), atol=1e-10)

    def test_iris_inverse_transform(self):
        X, _ = get_iris()
        ferro, sk = self._fit_both(X)
        Xt = ferro.transform(X)
        X_back = ferro.inverse_transform(Xt)
        np.testing.assert_allclose(X_back, X, atol=1e-10)

    def test_diabetes_transform(self):
        X, _ = get_diabetes()
        ferro, sk = self._fit_both(X)
        np.testing.assert_allclose(ferro.transform(X), sk.transform(X), atol=1e-10)

    def test_fit_transform(self):
        X, _ = get_iris()
        from ferroml.preprocessing import StandardScaler as FerroSS
        from sklearn.preprocessing import StandardScaler as SkSS
        ferro_out = FerroSS().fit_transform(X)
        sk_out = SkSS().fit_transform(X)
        np.testing.assert_allclose(ferro_out, sk_out, atol=1e-10)


# ============================================================================
# 2. MinMaxScaler
# ============================================================================

class TestMinMaxScalerComparison:
    """FerroML MinMaxScaler vs sklearn MinMaxScaler."""

    def _fit_both(self, X):
        from ferroml.preprocessing import MinMaxScaler as FerroMM
        from sklearn.preprocessing import MinMaxScaler as SkMM

        ferro = FerroMM()
        ferro.fit(X)
        sk = SkMM()
        sk.fit(X)
        return ferro, sk

    def test_iris_transform(self):
        X, _ = get_iris()
        ferro, sk = self._fit_both(X)
        np.testing.assert_allclose(ferro.transform(X), sk.transform(X), atol=1e-10)

    def test_iris_inverse_transform(self):
        X, _ = get_iris()
        ferro, sk = self._fit_both(X)
        Xt = ferro.transform(X)
        X_back = ferro.inverse_transform(Xt)
        np.testing.assert_allclose(X_back, X, atol=1e-10)

    def test_diabetes_transform(self):
        X, _ = get_diabetes()
        ferro, sk = self._fit_both(X)
        np.testing.assert_allclose(ferro.transform(X), sk.transform(X), atol=1e-10)


# ============================================================================
# 3. RobustScaler
# ============================================================================

class TestRobustScalerComparison:
    """FerroML RobustScaler vs sklearn RobustScaler."""

    def _fit_both(self, X):
        from ferroml.preprocessing import RobustScaler as FerroRS
        from sklearn.preprocessing import RobustScaler as SkRS

        ferro = FerroRS()
        ferro.fit(X)
        sk = SkRS()
        sk.fit(X)
        return ferro, sk

    def test_iris_transform(self):
        X, _ = get_iris()
        ferro, sk = self._fit_both(X)
        np.testing.assert_allclose(ferro.transform(X), sk.transform(X), atol=1e-10)

    def test_iris_inverse_transform(self):
        X, _ = get_iris()
        ferro, sk = self._fit_both(X)
        Xt = ferro.transform(X)
        X_back = ferro.inverse_transform(Xt)
        np.testing.assert_allclose(X_back, X, atol=1e-10)


# ============================================================================
# 4. MaxAbsScaler
# ============================================================================

class TestMaxAbsScalerComparison:
    """FerroML MaxAbsScaler vs sklearn MaxAbsScaler."""

    def _fit_both(self, X):
        from ferroml.preprocessing import MaxAbsScaler as FerroMA
        from sklearn.preprocessing import MaxAbsScaler as SkMA

        ferro = FerroMA()
        ferro.fit(X)
        sk = SkMA()
        sk.fit(X)
        return ferro, sk

    def test_iris_transform(self):
        X, _ = get_iris()
        ferro, sk = self._fit_both(X)
        np.testing.assert_allclose(ferro.transform(X), sk.transform(X), atol=1e-10)

    def test_iris_inverse_transform(self):
        X, _ = get_iris()
        ferro, sk = self._fit_both(X)
        Xt = ferro.transform(X)
        X_back = ferro.inverse_transform(Xt)
        np.testing.assert_allclose(X_back, X, atol=1e-10)


# ============================================================================
# 5. OneHotEncoder
# ============================================================================

class TestOneHotEncoderComparison:
    """FerroML OneHotEncoder vs sklearn OneHotEncoder."""

    def test_transform_output(self):
        from ferroml.preprocessing import OneHotEncoder as FerroOHE
        from sklearn.preprocessing import OneHotEncoder as SkOHE

        X = make_categorical_int_data(n=50)

        ferro = FerroOHE()
        ferro.fit(X)
        ferro_out = ferro.transform(X)

        sk = SkOHE(sparse_output=False)
        sk.fit(X)
        sk_out = sk.transform(X)

        assert ferro_out.shape == sk_out.shape, (
            f"Shape mismatch: ferro {ferro_out.shape} vs sklearn {sk_out.shape}"
        )
        np.testing.assert_allclose(ferro_out, sk_out, atol=1e-10)


# ============================================================================
# 6. OrdinalEncoder
# ============================================================================

class TestOrdinalEncoderComparison:
    """FerroML OrdinalEncoder vs sklearn OrdinalEncoder."""

    def test_transform_output(self):
        from ferroml.preprocessing import OrdinalEncoder as FerroOE
        from sklearn.preprocessing import OrdinalEncoder as SkOE

        X = make_categorical_int_data(n=50)

        ferro = FerroOE()
        ferro.fit(X)
        ferro_out = ferro.transform(X)

        sk = SkOE()
        sk.fit(X)
        sk_out = sk.transform(X)

        assert ferro_out.shape == sk_out.shape
        # FerroML and sklearn may assign ordinal codes in different order.
        # Verify that the mapping is consistent (a bijection) per column.
        for col in range(X.shape[1]):
            # Build mapping from ferro code -> sklearn code
            mapping = {}
            for i in range(X.shape[0]):
                fv, sv = ferro_out[i, col], sk_out[i, col]
                if fv in mapping:
                    assert mapping[fv] == sv, (
                        f"Inconsistent mapping col {col}: "
                        f"ferro {fv} -> {mapping[fv]} and {sv}"
                    )
                else:
                    mapping[fv] = sv
            # Verify bijection (unique values map uniquely)
            assert len(set(mapping.values())) == len(mapping), (
                f"Non-bijective mapping in col {col}"
            )


# ============================================================================
# 7. LabelEncoder
# ============================================================================

class TestLabelEncoderComparison:
    """FerroML LabelEncoder vs sklearn LabelEncoder."""

    def test_transform_output(self):
        from ferroml.preprocessing import LabelEncoder as FerroLE
        from sklearn.preprocessing import LabelEncoder as SkLE

        y = np.array([2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0])

        ferro = FerroLE()
        ferro.fit(y)
        ferro_out = ferro.transform(y)

        sk = SkLE()
        sk.fit(y)
        sk_out = sk.transform(y).astype(np.float64)

        # FerroML and sklearn may assign label codes in different order.
        # Verify the mapping is a consistent bijection.
        mapping = {}
        for fv, sv in zip(ferro_out, sk_out):
            if fv in mapping:
                assert mapping[fv] == sv, (
                    f"Inconsistent label mapping: ferro {fv} -> {mapping[fv]} and {sv}"
                )
            else:
                mapping[fv] = sv
        assert len(set(mapping.values())) == len(mapping), "Non-bijective label mapping"

    def test_inverse_transform(self):
        from ferroml.preprocessing import LabelEncoder as FerroLE

        y = np.array([2.0, 0.0, 1.0, 2.0, 0.0])

        ferro = FerroLE()
        ferro.fit(y)
        encoded = ferro.transform(y)
        decoded = ferro.inverse_transform(encoded)
        np.testing.assert_array_equal(decoded, y)


# ============================================================================
# 8. TargetEncoder
# ============================================================================

class TestTargetEncoderComparison:
    """FerroML TargetEncoder vs sklearn TargetEncoder."""

    def test_transform_output(self):
        from ferroml.preprocessing import TargetEncoder as FerroTE
        from sklearn.preprocessing import TargetEncoder as SkTE

        rng = np.random.RandomState(42)
        n = 200
        X = rng.randint(0, 4, size=(n, 2)).astype(np.float64)
        y = X[:, 0] * 1.5 + rng.randn(n) * 0.3

        ferro = FerroTE(smooth=1.0)
        ferro.fit(X, y)
        ferro_out = ferro.transform(X)

        sk = SkTE(smooth=1.0, random_state=42)
        sk.fit(X, y)
        sk_out = sk.transform(X)

        assert ferro_out.shape == sk_out.shape
        # TargetEncoder implementations can differ due to cross-fitting in sklearn
        # vs simple encoding in FerroML, so we use a loose tolerance and check
        # that the general mapping direction is preserved per-column
        for col in range(X.shape[1]):
            ferro_corr = np.corrcoef(ferro_out[:, col], sk_out[:, col])[0, 1]
            assert ferro_corr > 0.8, (
                f"Column {col}: correlation {ferro_corr:.4f} too low"
            )


# ============================================================================
# 9. SimpleImputer (mean)
# ============================================================================

class TestSimpleImputerMeanComparison:
    """FerroML SimpleImputer(mean) vs sklearn SimpleImputer(mean)."""

    def test_transform(self):
        from ferroml.preprocessing import SimpleImputer as FerroSI
        from sklearn.impute import SimpleImputer as SkSI

        X = make_nan_data()

        ferro = FerroSI(strategy="mean")
        ferro.fit(X)
        ferro_out = ferro.transform(X)

        sk = SkSI(strategy="mean")
        sk.fit(X)
        sk_out = sk.transform(X)

        assert not np.any(np.isnan(ferro_out)), "FerroML output contains NaN"
        np.testing.assert_allclose(ferro_out, sk_out, atol=1e-10)


# ============================================================================
# 10. SimpleImputer (median)
# ============================================================================

class TestSimpleImputerMedianComparison:
    """FerroML SimpleImputer(median) vs sklearn SimpleImputer(median)."""

    def test_transform(self):
        from ferroml.preprocessing import SimpleImputer as FerroSI
        from sklearn.impute import SimpleImputer as SkSI

        X = make_nan_data()

        ferro = FerroSI(strategy="median")
        ferro.fit(X)
        ferro_out = ferro.transform(X)

        sk = SkSI(strategy="median")
        sk.fit(X)
        sk_out = sk.transform(X)

        assert not np.any(np.isnan(ferro_out)), "FerroML output contains NaN"
        np.testing.assert_allclose(ferro_out, sk_out, atol=1e-10)


# ============================================================================
# 11. KNNImputer
# ============================================================================

class TestKNNImputerComparison:
    """FerroML KNNImputer vs sklearn KNNImputer."""

    def test_transform(self):
        from ferroml.preprocessing import KNNImputer as FerroKNN
        from sklearn.impute import KNNImputer as SkKNN

        X = make_nan_data(n=80, p=4)

        ferro = FerroKNN(n_neighbors=5)
        ferro.fit(X)
        ferro_out = ferro.transform(X)

        sk = SkKNN(n_neighbors=5)
        sk.fit(X)
        sk_out = sk.transform(X)

        assert not np.any(np.isnan(ferro_out)), "FerroML output contains NaN"
        np.testing.assert_allclose(ferro_out, sk_out, atol=1e-6)


# ============================================================================
# 12. PowerTransformer (yeo-johnson)
# ============================================================================

class TestPowerTransformerYJComparison:
    """FerroML PowerTransformer(yeo-johnson) vs sklearn."""

    def test_transform(self):
        from ferroml.preprocessing import PowerTransformer as FerroPT
        from sklearn.preprocessing import PowerTransformer as SkPT

        X, _ = get_iris()

        ferro = FerroPT(method="yeo-johnson")
        ferro.fit(X)
        ferro_out = ferro.transform(X)

        sk = SkPT(method="yeo-johnson")
        sk.fit(X)
        sk_out = sk.transform(X)

        assert ferro_out.shape == sk_out.shape
        np.testing.assert_allclose(ferro_out, sk_out, atol=1e-4)

    def test_inverse_transform(self):
        from ferroml.preprocessing import PowerTransformer as FerroPT

        X, _ = get_iris()
        ferro = FerroPT(method="yeo-johnson")
        ferro.fit(X)
        Xt = ferro.transform(X)
        X_back = ferro.inverse_transform(Xt)
        np.testing.assert_allclose(X_back, X, atol=1e-4)


# ============================================================================
# 13. PowerTransformer (box-cox)
# ============================================================================

class TestPowerTransformerBCComparison:
    """FerroML PowerTransformer(box-cox) vs sklearn."""

    def test_transform(self):
        from ferroml.preprocessing import PowerTransformer as FerroPT
        from sklearn.preprocessing import PowerTransformer as SkPT

        X = make_positive_data(n=150, p=4)

        ferro = FerroPT(method="box-cox")
        ferro.fit(X)
        ferro_out = ferro.transform(X)

        sk = SkPT(method="box-cox")
        sk.fit(X)
        sk_out = sk.transform(X)

        assert ferro_out.shape == sk_out.shape
        np.testing.assert_allclose(ferro_out, sk_out, atol=1e-4)


# ============================================================================
# 14. QuantileTransformer
# ============================================================================

class TestQuantileTransformerComparison:
    """FerroML QuantileTransformer vs sklearn QuantileTransformer."""

    def test_transform_uniform(self):
        from ferroml.preprocessing import QuantileTransformer as FerroQT
        from sklearn.preprocessing import QuantileTransformer as SkQT

        rng = np.random.RandomState(42)
        X = rng.randn(300, 4)  # larger dataset reduces interpolation differences

        ferro = FerroQT(n_quantiles=100, output_distribution="uniform")
        ferro.fit(X)
        ferro_out = ferro.transform(X)

        sk = SkQT(n_quantiles=100, output_distribution="uniform")
        sk.fit(X)
        sk_out = sk.transform(X)

        assert ferro_out.shape == sk_out.shape
        # Quantile transformers may use different interpolation methods internally.
        # Verify that the rank-order is preserved and values are close in aggregate.
        for col in range(X.shape[1]):
            ferro_col = ferro_out[:, col]
            sk_col = sk_out[:, col]
            # Rank correlation should be very high
            rank_corr = np.corrcoef(ferro_col, sk_col)[0, 1]
            assert rank_corr > 0.99, (
                f"Column {col}: rank correlation {rank_corr:.4f} too low"
            )
            # Both should map to [0, 1] range
            assert ferro_col.min() >= -1e-7
            assert ferro_col.max() <= 1.0 + 1e-7

    def test_output_range(self):
        from ferroml.preprocessing import QuantileTransformer as FerroQT

        X, _ = get_iris()
        ferro = FerroQT(n_quantiles=50, output_distribution="uniform")
        ferro.fit(X)
        Xt = ferro.transform(X)
        assert Xt.min() >= 0.0 - 1e-7
        assert Xt.max() <= 1.0 + 1e-7


# ============================================================================
# 15. PolynomialFeatures
# ============================================================================

class TestPolynomialFeaturesComparison:
    """FerroML PolynomialFeatures vs sklearn PolynomialFeatures."""

    def test_degree2_shape_and_values(self):
        from ferroml.preprocessing import PolynomialFeatures as FerroPF
        from sklearn.preprocessing import PolynomialFeatures as SkPF

        rng = np.random.RandomState(42)
        X = rng.randn(30, 3)

        ferro = FerroPF(degree=2)
        ferro_out = ferro.fit_transform(X)

        # FerroML may or may not include bias/interaction_only options.
        # Determine sklearn params to match FerroML output shape.
        sk_no_bias = SkPF(degree=2, include_bias=False)
        sk_no_bias_out = sk_no_bias.fit_transform(X)
        sk_with_bias = SkPF(degree=2, include_bias=True)
        sk_with_bias_out = sk_with_bias.fit_transform(X)

        if ferro_out.shape[1] == sk_no_bias_out.shape[1]:
            np.testing.assert_allclose(ferro_out, sk_no_bias_out, atol=1e-10)
        elif ferro_out.shape[1] == sk_with_bias_out.shape[1]:
            np.testing.assert_allclose(ferro_out, sk_with_bias_out, atol=1e-10)
        else:
            pytest.fail(
                f"FerroML output shape {ferro_out.shape} does not match "
                f"sklearn with bias {sk_with_bias_out.shape} or "
                f"without bias {sk_no_bias_out.shape}"
            )

    def test_degree3(self):
        from ferroml.preprocessing import PolynomialFeatures as FerroPF
        from sklearn.preprocessing import PolynomialFeatures as SkPF

        rng = np.random.RandomState(42)
        X = rng.randn(20, 2)

        ferro = FerroPF(degree=3)
        ferro_out = ferro.fit_transform(X)

        sk_no_bias = SkPF(degree=3, include_bias=False)
        sk_no_bias_out = sk_no_bias.fit_transform(X)
        sk_with_bias = SkPF(degree=3, include_bias=True)
        sk_with_bias_out = sk_with_bias.fit_transform(X)

        if ferro_out.shape[1] == sk_no_bias_out.shape[1]:
            np.testing.assert_allclose(ferro_out, sk_no_bias_out, atol=1e-10)
        elif ferro_out.shape[1] == sk_with_bias_out.shape[1]:
            np.testing.assert_allclose(ferro_out, sk_with_bias_out, atol=1e-10)
        else:
            pytest.fail(
                f"FerroML degree=3 shape {ferro_out.shape} mismatches sklearn"
            )


# ============================================================================
# 16. KBinsDiscretizer
# ============================================================================

class TestKBinsDiscretizerComparison:
    """FerroML KBinsDiscretizer vs sklearn KBinsDiscretizer."""

    def test_uniform_strategy(self):
        from ferroml.preprocessing import KBinsDiscretizer as FerroKB
        from sklearn.preprocessing import KBinsDiscretizer as SkKB

        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)

        ferro = FerroKB(n_bins=5, strategy="uniform")
        ferro.fit(X)
        ferro_out = ferro.transform(X)

        sk = SkKB(n_bins=5, strategy="uniform", encode="ordinal")
        sk.fit(X)
        sk_out = sk.transform(X)

        assert ferro_out.shape == sk_out.shape
        np.testing.assert_allclose(ferro_out, sk_out, atol=1e-6)

    def test_quantile_strategy(self):
        from ferroml.preprocessing import KBinsDiscretizer as FerroKB
        from sklearn.preprocessing import KBinsDiscretizer as SkKB

        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)

        ferro = FerroKB(n_bins=4, strategy="quantile")
        ferro.fit(X)
        ferro_out = ferro.transform(X)

        sk = SkKB(n_bins=4, strategy="quantile", encode="ordinal")
        sk.fit(X)
        sk_out = sk.transform(X)

        assert ferro_out.shape == sk_out.shape
        np.testing.assert_allclose(ferro_out, sk_out, atol=1e-6)


# ============================================================================
# 17. VarianceThreshold
# ============================================================================

class TestVarianceThresholdComparison:
    """FerroML VarianceThreshold vs sklearn VarianceThreshold."""

    def test_selected_features(self):
        from ferroml.preprocessing import VarianceThreshold as FerroVT
        from sklearn.feature_selection import VarianceThreshold as SkVT

        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        X[:, 2] = 0.0  # zero variance column
        X[:, 4] = 0.01 * rng.randn(100)  # near-zero variance

        ferro = FerroVT(threshold=0.01)
        ferro.fit(X)
        ferro_out = ferro.transform(X)

        sk = SkVT(threshold=0.01)
        sk.fit(X)
        sk_out = sk.transform(X)

        assert ferro_out.shape == sk_out.shape, (
            f"Shape mismatch: ferro {ferro_out.shape} vs sklearn {sk_out.shape}"
        )
        np.testing.assert_allclose(ferro_out, sk_out, atol=1e-10)

    def test_default_threshold(self):
        from ferroml.preprocessing import VarianceThreshold as FerroVT
        from sklearn.feature_selection import VarianceThreshold as SkVT

        X, _ = get_iris()

        ferro = FerroVT()
        ferro.fit(X)
        ferro_out = ferro.transform(X)

        sk = SkVT()
        sk.fit(X)
        sk_out = sk.transform(X)

        assert ferro_out.shape == sk_out.shape
        np.testing.assert_allclose(ferro_out, sk_out, atol=1e-10)


# ============================================================================
# 18. SelectKBest
# ============================================================================

class TestSelectKBestComparison:
    """FerroML SelectKBest vs sklearn SelectKBest."""

    def test_selected_features(self):
        from ferroml.preprocessing import SelectKBest as FerroSKB
        from sklearn.feature_selection import SelectKBest as SkSKB
        from sklearn.feature_selection import f_classif

        X, y = get_iris()

        ferro = FerroSKB(k=2)
        ferro.fit(X, y)
        ferro_out = ferro.transform(X)

        sk = SkSKB(score_func=f_classif, k=2)
        sk.fit(X, y)
        sk_out = sk.transform(X)

        assert ferro_out.shape == sk_out.shape, (
            f"Shape mismatch: ferro {ferro_out.shape} vs sklearn {sk_out.shape}"
        )
        # Values of selected features should match exactly
        np.testing.assert_allclose(ferro_out, sk_out, atol=1e-6)

    def test_scores(self):
        from ferroml.preprocessing import SelectKBest as FerroSKB
        from sklearn.feature_selection import SelectKBest as SkSKB
        from sklearn.feature_selection import f_classif

        X, y = get_iris()

        ferro = FerroSKB(k=2)
        ferro.fit(X, y)

        sk = SkSKB(score_func=f_classif, k=2)
        sk.fit(X, y)

        # Scores should be close (both use F-statistic)
        np.testing.assert_allclose(ferro.scores_, sk.scores_, atol=1e-6)


# ============================================================================
# 19. Integration: scaler roundtrip on multiple datasets
# ============================================================================

class TestScalerRoundtripIntegration:
    """Verify fit -> transform -> inverse_transform roundtrip for all scalers."""

    @pytest.mark.parametrize("scaler_name", [
        "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    ])
    def test_roundtrip_wine(self, scaler_name):
        import ferroml.preprocessing as fp

        ScalerClass = getattr(fp, scaler_name)
        X, _ = get_wine()

        scaler = ScalerClass()
        scaler.fit(X)
        Xt = scaler.transform(X)
        X_back = scaler.inverse_transform(Xt)
        np.testing.assert_allclose(X_back, X, atol=1e-10,
                                   err_msg=f"{scaler_name} roundtrip failed on wine")


# ============================================================================
# 20. Integration: fit_transform consistency
# ============================================================================

class TestFitTransformConsistency:
    """Verify fit_transform == fit then transform for transformers that support it."""

    @pytest.mark.parametrize("scaler_name", [
        "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    ])
    def test_fit_transform_equals_separate(self, scaler_name):
        import ferroml.preprocessing as fp

        ScalerClass = getattr(fp, scaler_name)
        X, _ = get_iris()

        # fit_transform path
        s1 = ScalerClass()
        out1 = s1.fit_transform(X)

        # separate fit + transform path
        s2 = ScalerClass()
        s2.fit(X)
        out2 = s2.transform(X)

        np.testing.assert_allclose(out1, out2, atol=1e-14,
                                   err_msg=f"{scaler_name} fit_transform != fit+transform")

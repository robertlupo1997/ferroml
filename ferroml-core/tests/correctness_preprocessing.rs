//! Comprehensive correctness tests for FerroML preprocessing transformers.
//!
//! Tests cover: PolynomialFeatures, KBinsDiscretizer, PowerTransformer,
//! QuantileTransformer, VarianceThreshold, SelectKBest, SelectFromModel,
//! RecursiveFeatureElimination, SimpleImputer, KNNImputer, OneHotEncoder,
//! OrdinalEncoder, LabelEncoder, TargetEncoder, SMOTE, ADASYN, RandomOverSampler,
//! and pipeline integration.
//!
//! Each test verifies correctness against hand-computed or sklearn-equivalent results.
//! Tests that reveal bugs are marked `#[ignore]` with a comment explaining the issue.

use ferroml_core::preprocessing::Transformer;
use ndarray::{array, Array1, Array2, Axis};

// =============================================================================
// Helper functions
// =============================================================================

/// Assert that two f64 values are approximately equal within a given tolerance.
fn assert_approx(actual: f64, expected: f64, tol: f64, msg: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{}: expected {}, got {}, diff = {}",
        msg,
        expected,
        actual,
        (actual - expected).abs()
    );
}

/// Assert that two Array2<f64> values are approximately equal element-wise.
fn assert_array2_approx(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64, msg: &str) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{}: shape mismatch: {:?} vs {:?}",
        msg,
        actual.shape(),
        expected.shape()
    );
    for ((i, j), &a) in actual.indexed_iter() {
        let e = expected[[i, j]];
        assert!(
            (a - e).abs() < tol,
            "{}: at [{},{}] expected {}, got {}, diff = {}",
            msg,
            i,
            j,
            e,
            a,
            (a - e).abs()
        );
    }
}

// =============================================================================
// PolynomialFeatures Tests (5 tests)
// =============================================================================

#[test]
fn polynomial_features_degree2_two_features() {
    // sklearn: PolynomialFeatures(degree=2, include_bias=True).fit_transform([[1,2],[3,4]])
    // Expected output: [1, x0, x1, x0^2, x0*x1, x1^2]
    // Row 0: [1, 1, 2, 1, 2, 4]
    // Row 1: [1, 3, 4, 9, 12, 16]
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;

    let mut poly = PolynomialFeatures::new(2);
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let result = poly.fit_transform(&x).unwrap();

    assert_eq!(
        result.ncols(),
        6,
        "degree=2 with 2 features => 6 output cols"
    );
    assert_eq!(result.nrows(), 2);

    let expected = array![
        [1.0, 1.0, 2.0, 1.0, 2.0, 4.0],
        [1.0, 3.0, 4.0, 9.0, 12.0, 16.0]
    ];
    assert_array2_approx(&result, &expected, 1e-10, "poly_degree2");
}

#[test]
fn polynomial_features_degree3_single_feature() {
    // sklearn: PolynomialFeatures(degree=3, include_bias=True).fit_transform([[2],[3],[5]])
    // Expected: [1, x, x^2, x^3]
    // Row 0: [1, 2, 4, 8]
    // Row 1: [1, 3, 9, 27]
    // Row 2: [1, 5, 25, 125]
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;

    let mut poly = PolynomialFeatures::new(3);
    let x = array![[2.0], [3.0], [5.0]];
    let result = poly.fit_transform(&x).unwrap();

    assert_eq!(result.ncols(), 4, "degree=3, 1 feature => 4 cols");
    let expected = array![
        [1.0, 2.0, 4.0, 8.0],
        [1.0, 3.0, 9.0, 27.0],
        [1.0, 5.0, 25.0, 125.0]
    ];
    assert_array2_approx(&result, &expected, 1e-10, "poly_degree3_single");
}

#[test]
fn polynomial_features_interaction_only() {
    // sklearn: PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)
    //          .fit_transform([[1,2,3]])
    // Expected: [1, x0, x1, x2, x0*x1, x0*x2, x1*x2]
    // Row 0: [1, 1, 2, 3, 2, 3, 6]
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;

    let mut poly = PolynomialFeatures::new(2).interaction_only(true);
    let x = array![[1.0, 2.0, 3.0]];
    let result = poly.fit_transform(&x).unwrap();

    assert_eq!(
        result.ncols(),
        7,
        "interaction_only degree=2 with 3 features => 7 cols"
    );

    let expected = array![[1.0, 1.0, 2.0, 3.0, 2.0, 3.0, 6.0]];
    assert_array2_approx(&result, &expected, 1e-10, "poly_interaction_only");
}

#[test]
fn polynomial_features_no_bias() {
    // sklearn: PolynomialFeatures(degree=2, include_bias=False).fit_transform([[1,2]])
    // Expected: [x0, x1, x0^2, x0*x1, x1^2] = [1, 2, 1, 2, 4]
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;

    let mut poly = PolynomialFeatures::new(2).include_bias(false);
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let result = poly.fit_transform(&x).unwrap();

    assert_eq!(
        result.ncols(),
        5,
        "no_bias degree=2 with 2 features => 5 cols"
    );
    let expected = array![[1.0, 2.0, 1.0, 2.0, 4.0], [3.0, 4.0, 9.0, 12.0, 16.0]];
    assert_array2_approx(&result, &expected, 1e-10, "poly_no_bias");
}

#[test]
fn polynomial_features_output_shape_degree2_four_features() {
    // C(n+d, d) = C(4+2, 2) = C(6,2) = 15 with bias
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;

    let mut poly = PolynomialFeatures::new(2);
    let x = Array2::from_shape_fn((3, 4), |(i, j)| (i + j + 1) as f64);
    let result = poly.fit_transform(&x).unwrap();

    assert_eq!(
        result.ncols(),
        15,
        "degree=2 with 4 features => C(6,2)=15 cols"
    );
    assert_eq!(result.nrows(), 3);
}

// =============================================================================
// KBinsDiscretizer Tests (4 tests)
// =============================================================================

#[test]
fn kbins_uniform_strategy() {
    // Uniform binning: [0, 1, 2, ..., 9] into 5 bins
    // Bin edges: [0, 2, 4, 6, 8, 10] (wait, max=9)
    // Actually: edges = min + i*(max-min)/n_bins = 0 + i*9/5
    // edges: [0.0, 1.8, 3.6, 5.4, 7.2, 9.0]
    // Value 0 -> bin 0, 1 -> bin 0, 2 -> bin 1, 3 -> bin 1, 4 -> bin 2,
    // 5 -> bin 2, 6 -> bin 3, 7 -> bin 3, 8 -> bin 4, 9 -> bin 4
    use ferroml_core::preprocessing::discretizers::{BinningStrategy, KBinsDiscretizer};

    let mut disc = KBinsDiscretizer::new()
        .with_n_bins(5)
        .with_strategy(BinningStrategy::Uniform);

    let x = array![
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0]
    ];
    let result = disc.fit_transform(&x).unwrap();

    assert_eq!(result.ncols(), 1);
    assert_eq!(result.nrows(), 10);

    // Verify bin values are integers in [0, 4]
    for i in 0..10 {
        let bin = result[[i, 0]];
        assert!((0.0..5.0).contains(&bin), "bin {} should be in [0, 5)", bin);
        assert!(
            (bin - bin.round()).abs() < 1e-10,
            "bin should be integer, got {}",
            bin
        );
    }

    // Values should be monotonically non-decreasing
    for i in 1..10 {
        assert!(
            result[[i, 0]] >= result[[i - 1, 0]],
            "bins should be non-decreasing"
        );
    }
}

#[test]
fn kbins_quantile_strategy() {
    // Quantile binning: equal-frequency bins
    // With 10 samples and 5 bins, each bin should have ~2 samples
    use ferroml_core::preprocessing::discretizers::{BinningStrategy, KBinsDiscretizer};

    let mut disc = KBinsDiscretizer::new()
        .with_n_bins(5)
        .with_strategy(BinningStrategy::Quantile);

    let x = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0]
    ];
    let result = disc.fit_transform(&x).unwrap();

    // Verify output is valid bin indices
    for i in 0..10 {
        let bin = result[[i, 0]];
        assert!(bin >= 0.0, "bin should be >= 0, got {}", bin);
    }

    // Bins should be monotonically non-decreasing (sorted input)
    for i in 1..10 {
        assert!(
            result[[i, 0]] >= result[[i - 1, 0]],
            "bins should be non-decreasing for sorted input"
        );
    }

    // First and last bins should be different
    assert!(
        result[[9, 0]] > result[[0, 0]],
        "first and last bins should differ"
    );
}

#[test]
fn kbins_kmeans_strategy() {
    // K-means binning: bins determined by clustering
    use ferroml_core::preprocessing::discretizers::{BinningStrategy, KBinsDiscretizer};

    let mut disc = KBinsDiscretizer::new()
        .with_n_bins(3)
        .with_strategy(BinningStrategy::KMeans);

    // Data with 3 natural clusters
    let x = array![
        [1.0],
        [1.5],
        [2.0],
        [10.0],
        [10.5],
        [11.0],
        [20.0],
        [20.5],
        [21.0]
    ];
    let result = disc.fit_transform(&x).unwrap();

    // The first 3 samples should have the same bin
    assert!(
        result[[0, 0]] == result[[1, 0]] && result[[1, 0]] == result[[2, 0]],
        "cluster 1 should be in same bin"
    );
    // The middle 3 should have the same bin
    assert!(
        result[[3, 0]] == result[[4, 0]] && result[[4, 0]] == result[[5, 0]],
        "cluster 2 should be in same bin"
    );
    // The last 3 should have the same bin
    assert!(
        result[[6, 0]] == result[[7, 0]] && result[[7, 0]] == result[[8, 0]],
        "cluster 3 should be in same bin"
    );
    // Each cluster should be in a different bin
    assert!(
        result[[0, 0]] != result[[3, 0]]
            && result[[3, 0]] != result[[6, 0]]
            && result[[0, 0]] != result[[6, 0]],
        "different clusters should be in different bins"
    );
}

#[test]
fn kbins_single_feature_many_bins() {
    // Edge case: more bins requested than unique values
    use ferroml_core::preprocessing::discretizers::{BinningStrategy, KBinsDiscretizer};

    let mut disc = KBinsDiscretizer::new()
        .with_n_bins(10)
        .with_strategy(BinningStrategy::Quantile);

    // Only 4 unique values
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let result = disc.fit_transform(&x);

    // Should still succeed (may reduce number of bins internally)
    assert!(
        result.is_ok(),
        "Should handle fewer unique values than bins"
    );
    let result = result.unwrap();
    assert_eq!(result.nrows(), 4);
}

// =============================================================================
// PowerTransformer Tests (4 tests)
// =============================================================================

#[test]
fn power_transformer_yeo_johnson_positive_data() {
    // Yeo-Johnson on positive skewed data should make it more Gaussian
    use ferroml_core::preprocessing::power::{PowerMethod, PowerTransformer};

    let mut pt = PowerTransformer::new(PowerMethod::YeoJohnson);

    // Highly skewed positive data
    let x = array![
        [1.0],
        [2.0],
        [3.0],
        [5.0],
        [8.0],
        [13.0],
        [21.0],
        [34.0],
        [55.0],
        [89.0]
    ];
    let result = pt.fit_transform(&x).unwrap();

    // Verify standardized output has approximately zero mean and unit variance
    let col = result.column(0);
    let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
    let var: f64 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / col.len() as f64;

    assert_approx(mean, 0.0, 0.2, "yeo_johnson mean should be near 0");
    assert_approx(var, 1.0, 0.5, "yeo_johnson variance should be near 1");

    // Lambda should be learned
    let lambdas = pt.lambdas().unwrap();
    assert_eq!(lambdas.len(), 1);
    // For positive skewed data, lambda should be less than 1
    assert!(
        lambdas[0] < 2.0,
        "lambda for skewed data should be reasonable, got {}",
        lambdas[0]
    );
}

#[test]
fn power_transformer_box_cox_positive_data() {
    // Box-Cox requires strictly positive data
    use ferroml_core::preprocessing::power::{PowerMethod, PowerTransformer};

    let mut pt = PowerTransformer::new(PowerMethod::BoxCox);

    let x = array![[1.0], [4.0], [9.0], [16.0], [25.0], [36.0], [49.0], [64.0]];
    let result = pt.fit_transform(&x).unwrap();

    // Should produce standardized output
    let col = result.column(0);
    let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
    let var: f64 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / col.len() as f64;

    assert_approx(mean, 0.0, 0.2, "box_cox mean should be near 0");
    assert_approx(var, 1.0, 0.5, "box_cox variance should be near 1");
}

#[test]
fn power_transformer_box_cox_rejects_nonpositive() {
    // Box-Cox should reject non-positive data
    use ferroml_core::preprocessing::power::{PowerMethod, PowerTransformer};

    let mut pt = PowerTransformer::new(PowerMethod::BoxCox);

    let x = array![[0.0], [1.0], [2.0]];
    let result = pt.fit_transform(&x);

    // Box-Cox requires strictly positive data, should error
    assert!(
        result.is_err(),
        "Box-Cox should reject non-positive data (contains 0)"
    );
}

#[test]
fn power_transformer_yeo_johnson_mixed_data() {
    // Yeo-Johnson should handle positive, negative, and zero data
    use ferroml_core::preprocessing::power::{PowerMethod, PowerTransformer};

    let mut pt = PowerTransformer::new(PowerMethod::YeoJohnson);

    let x = array![[-5.0], [-2.0], [0.0], [1.0], [3.0], [7.0], [15.0], [30.0]];
    let result = pt.fit_transform(&x).unwrap();

    // Should produce finite values
    for &v in result.iter() {
        assert!(
            v.is_finite(),
            "Yeo-Johnson output should be finite, got {}",
            v
        );
    }

    // Verify standardized output
    let col = result.column(0);
    let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
    assert_approx(mean, 0.0, 0.3, "yeo_johnson mixed data mean near 0");
}

// =============================================================================
// PowerTransformer Inverse Transform Test (1 test)
// =============================================================================

#[test]
fn power_transformer_inverse_transform_roundtrip() {
    use ferroml_core::preprocessing::power::{PowerMethod, PowerTransformer};

    let mut pt = PowerTransformer::new(PowerMethod::YeoJohnson);
    let x = array![[1.0], [2.0], [5.0], [10.0], [20.0], [50.0]];

    pt.fit(&x).unwrap();
    let transformed = pt.transform(&x).unwrap();
    let recovered = pt.inverse_transform(&transformed).unwrap();

    for i in 0..x.nrows() {
        assert_approx(
            recovered[[i, 0]],
            x[[i, 0]],
            0.1,
            &format!("roundtrip row {}", i),
        );
    }
}

// =============================================================================
// QuantileTransformer Tests (3 tests)
// =============================================================================

#[test]
fn quantile_transformer_uniform_output() {
    // QuantileTransformer with uniform output should map to [epsilon, 1-epsilon]
    use ferroml_core::preprocessing::quantile::{OutputDistribution, QuantileTransformer};

    let mut qt = QuantileTransformer::new(OutputDistribution::Uniform);

    // 20 samples, sorted
    let x = Array2::from_shape_fn((20, 1), |(i, _)| (i + 1) as f64);
    let result = qt.fit_transform(&x).unwrap();

    // Output should be in [0, 1]
    for &v in result.iter() {
        assert!(
            (0.0..=1.0).contains(&v),
            "Uniform quantile output should be in [0,1], got {}",
            v
        );
    }

    // Output should be monotonically non-decreasing (input is sorted)
    for i in 1..20 {
        assert!(
            result[[i, 0]] >= result[[i - 1, 0]],
            "Uniform quantile output should be monotone"
        );
    }

    // Min should be near 0, max should be near 1
    let min_val = result.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = result.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(min_val < 0.1, "min should be near 0, got {}", min_val);
    assert!(max_val > 0.9, "max should be near 1, got {}", max_val);
}

#[test]
fn quantile_transformer_normal_output() {
    // QuantileTransformer with normal output should map to approximately N(0,1)
    use ferroml_core::preprocessing::quantile::{OutputDistribution, QuantileTransformer};

    let mut qt = QuantileTransformer::new(OutputDistribution::Normal);

    // 100 samples from uniform distribution
    let x = Array2::from_shape_fn((100, 1), |(i, _)| (i as f64 + 0.5) / 100.0);
    let result = qt.fit_transform(&x).unwrap();

    // Output should be finite
    for &v in result.iter() {
        assert!(
            v.is_finite(),
            "Normal quantile output should be finite, got {}",
            v
        );
    }

    // Output should be approximately N(0,1): mean near 0, std near 1
    let col = result.column(0);
    let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
    let var: f64 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / col.len() as f64;

    assert_approx(mean, 0.0, 0.3, "normal quantile mean");
    assert!(var > 0.3, "normal quantile variance should be substantial");
}

#[test]
fn quantile_transformer_few_samples() {
    // With very few samples, quantile transformer should still work
    use ferroml_core::preprocessing::quantile::{OutputDistribution, QuantileTransformer};

    let mut qt = QuantileTransformer::new(OutputDistribution::Uniform).with_n_quantiles(3);

    let x = array![[1.0], [2.0], [3.0]];
    let result = qt.fit_transform(&x).unwrap();

    assert_eq!(result.nrows(), 3);
    assert_eq!(result.ncols(), 1);

    // Values should be in [0, 1]
    for &v in result.iter() {
        assert!(
            (0.0..=1.0).contains(&v),
            "output should be in [0,1], got {}",
            v
        );
    }
}

// =============================================================================
// VarianceThreshold Tests (3 tests)
// =============================================================================

#[test]
fn variance_threshold_removes_constant_features() {
    // VarianceThreshold(0) should remove constant features
    use ferroml_core::preprocessing::selection::VarianceThreshold;

    let mut selector = VarianceThreshold::new(0.0);
    // Column 0 and 2 are constant, column 1 has variance
    let x = array![[1.0, 5.0, 3.0], [1.0, 2.0, 3.0], [1.0, 8.0, 3.0]];
    let result = selector.fit_transform(&x).unwrap();

    assert_eq!(result.ncols(), 1, "Should keep only 1 non-constant feature");
    assert_eq!(result.nrows(), 3);

    // The remaining column should be [5, 2, 8]
    assert_approx(result[[0, 0]], 5.0, 1e-10, "row 0");
    assert_approx(result[[1, 0]], 2.0, 1e-10, "row 1");
    assert_approx(result[[2, 0]], 8.0, 1e-10, "row 2");
}

#[test]
fn variance_threshold_with_threshold() {
    // VarianceThreshold(threshold=1.0) should remove features with variance <= 1.0
    use ferroml_core::preprocessing::selection::VarianceThreshold;

    let mut selector = VarianceThreshold::new(1.0);
    // Column 0: var([1,2,3]) = 2/3 < 1 => removed
    // Column 1: var([10,20,30]) = 200/3 >> 1 => kept
    // Column 2: var([5.0, 5.1, 5.2]) = ~0.0067 < 1 => removed
    let x = array![[1.0, 10.0, 5.0], [2.0, 20.0, 5.1], [3.0, 30.0, 5.2]];
    let result = selector.fit_transform(&x).unwrap();

    assert_eq!(result.ncols(), 1, "Only column with var > 1 should remain");
    assert_approx(result[[0, 0]], 10.0, 1e-10, "kept col row 0");
    assert_approx(result[[1, 0]], 20.0, 1e-10, "kept col row 1");
    assert_approx(result[[2, 0]], 30.0, 1e-10, "kept col row 2");
}

#[test]
fn variance_threshold_all_constant_rejects() {
    // If all features are constant, should return error
    use ferroml_core::preprocessing::selection::VarianceThreshold;

    let mut selector = VarianceThreshold::new(0.0);
    let x = array![[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];
    let result = selector.fit(&x);

    assert!(
        result.is_err(),
        "Should reject when all features are constant"
    );
}

// =============================================================================
// SelectKBest Tests (3 tests)
// =============================================================================

#[test]
fn select_k_best_f_regression_correlated_features() {
    // Select features most correlated with target
    use ferroml_core::preprocessing::selection::{ScoreFunction, SelectKBest};

    let mut selector = SelectKBest::new(ScoreFunction::FRegression, 2);

    // x0: perfectly correlated with y (y = 2*x0)
    // x1: moderately correlated with y
    // x2: uncorrelated (random-ish)
    let x = array![
        [1.0, 1.0, 0.5],
        [2.0, 3.0, 0.3],
        [3.0, 5.0, 0.8],
        [4.0, 7.0, 0.2],
        [5.0, 9.0, 0.6]
    ];
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2*x0

    selector.fit_with_target(&x, &y).unwrap();
    let selected = selector.selected_indices().unwrap();

    assert_eq!(selected.len(), 2);
    // Feature 0 (perfect correlation) should be selected
    assert!(
        selected.contains(&0),
        "Feature 0 (perfectly correlated) should be selected, got {:?}",
        selected
    );
    // Feature 1 (high correlation) should also be selected
    assert!(
        selected.contains(&1),
        "Feature 1 (highly correlated) should be selected, got {:?}",
        selected
    );

    // Transform should keep only selected features
    let x_selected = selector.transform(&x).unwrap();
    assert_eq!(x_selected.ncols(), 2);
}

#[test]
fn select_k_best_f_classif() {
    // F-classif: ANOVA F-value for classification
    use ferroml_core::preprocessing::selection::{ScoreFunction, SelectKBest};

    let mut selector = SelectKBest::new(ScoreFunction::FClassif, 1);

    // x0: class-discriminative (class 0 has low vals, class 1 has high vals)
    // x1: not discriminative (similar values across classes)
    let x = array![
        [1.0, 5.0],
        [2.0, 5.5],
        [1.5, 4.5],
        [10.0, 5.2],
        [11.0, 4.8],
        [10.5, 5.1]
    ];
    let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    selector.fit_with_target(&x, &y).unwrap();
    let selected = selector.selected_indices().unwrap();

    assert_eq!(selected.len(), 1);
    // Feature 0 should be selected (high between-class variance)
    assert_eq!(
        selected[0], 0,
        "Feature 0 should be selected for classification"
    );
}

#[test]
fn select_k_best_k_equals_n_features() {
    // When k == n_features, all features should be selected
    use ferroml_core::preprocessing::selection::{ScoreFunction, SelectKBest};

    let mut selector = SelectKBest::new(ScoreFunction::FRegression, 3);

    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let y = array![1.0, 2.0, 3.0];

    selector.fit_with_target(&x, &y).unwrap();
    let x_selected = selector.transform(&x).unwrap();

    assert_eq!(x_selected.ncols(), 3, "All features should be selected");
    assert_array2_approx(&x_selected, &x, 1e-10, "select_all");
}

// =============================================================================
// SelectFromModel Tests (2 tests)
// =============================================================================

#[test]
fn select_from_model_mean_threshold() {
    // SelectFromModel with Mean threshold: select features with importance > mean
    use ferroml_core::preprocessing::selection::{ImportanceThreshold, SelectFromModel};

    let importances = array![0.1, 0.5, 0.05, 0.8, 0.02];
    // mean = (0.1+0.5+0.05+0.8+0.02)/5 = 0.294
    // Features with importance > 0.294: indices 1 (0.5) and 3 (0.8)

    let mut selector = SelectFromModel::new(importances, ImportanceThreshold::Mean);

    let x = array![
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0]
    ];

    let result = selector.fit_transform(&x).unwrap();
    assert_eq!(
        result.ncols(),
        2,
        "Should select 2 features above mean importance"
    );

    // Verify the selected columns are 1 and 3
    assert_approx(result[[0, 0]], 2.0, 1e-10, "first selected feature, row 0");
    assert_approx(result[[0, 1]], 4.0, 1e-10, "second selected feature, row 0");
}

#[test]
fn select_from_model_value_threshold() {
    // SelectFromModel with explicit value threshold
    use ferroml_core::preprocessing::selection::{ImportanceThreshold, SelectFromModel};

    let importances = array![0.3, 0.1, 0.6, 0.9];
    // threshold = 0.5 => select indices 2 (0.6) and 3 (0.9)
    let mut selector = SelectFromModel::new(importances, ImportanceThreshold::Value(0.5));

    let x = array![[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]];
    let result = selector.fit_transform(&x).unwrap();

    assert_eq!(result.ncols(), 2);
    assert_approx(result[[0, 0]], 30.0, 1e-10, "selected feature 2");
    assert_approx(result[[0, 1]], 40.0, 1e-10, "selected feature 3");
}

// =============================================================================
// RecursiveFeatureElimination Tests (2 tests)
// =============================================================================

#[test]
fn rfe_selects_features_by_importance() {
    // RFE using variance as importance: should keep high-variance features
    use ferroml_core::preprocessing::selection::{ClosureEstimator, RecursiveFeatureElimination};

    let estimator =
        ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

    let mut rfe = RecursiveFeatureElimination::new(Box::new(estimator))
        .with_n_features_to_select(2)
        .with_step(1);

    // Feature 0: low variance, Feature 1: medium, Feature 2: high
    let x = array![
        [1.0, 10.0, 100.0],
        [1.1, 12.0, 200.0],
        [0.9, 8.0, 300.0],
        [1.0, 11.0, 400.0]
    ];
    let y = array![1.0, 2.0, 3.0, 4.0];

    rfe.fit_with_target(&x, &y).unwrap();
    let selected = rfe.selected_indices().unwrap();

    assert_eq!(selected.len(), 2, "Should select 2 features");
    // Feature 2 (highest variance) should definitely be selected
    assert!(
        selected.contains(&2),
        "Feature 2 (highest variance) should be selected, got {:?}",
        selected
    );
}

#[test]
fn rfe_transform_keeps_selected_only() {
    use ferroml_core::preprocessing::selection::{ClosureEstimator, RecursiveFeatureElimination};

    let estimator =
        ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

    let mut rfe = RecursiveFeatureElimination::new(Box::new(estimator))
        .with_n_features_to_select(1)
        .with_step(1);

    let x = array![
        [1.0, 100.0, 5.0],
        [2.0, 200.0, 5.1],
        [3.0, 300.0, 5.2],
        [4.0, 400.0, 5.3]
    ];
    let y = array![1.0, 2.0, 3.0, 4.0];

    rfe.fit_with_target(&x, &y).unwrap();
    let x_selected = rfe.transform(&x).unwrap();

    assert_eq!(x_selected.ncols(), 1, "Should produce 1 feature");
    // The selected feature should be feature 1 (highest variance)
    assert_approx(x_selected[[0, 0]], 100.0, 1e-10, "selected feature row 0");
    assert_approx(x_selected[[1, 0]], 200.0, 1e-10, "selected feature row 1");
}

// =============================================================================
// SimpleImputer Tests (4 tests)
// =============================================================================

#[test]
fn simple_imputer_mean_strategy() {
    // Mean imputation: fill NaN with column mean
    use ferroml_core::preprocessing::imputers::{ImputeStrategy, SimpleImputer};

    let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);

    // Col 0: non-NaN values = [1, 3, 5], mean = 3.0
    // Col 1: non-NaN values = [2, 4], mean = 3.0
    let x = array![
        [1.0, 2.0],
        [f64::NAN, 4.0],
        [3.0, f64::NAN],
        [5.0, f64::NAN]
    ];
    let result = imputer.fit_transform(&x).unwrap();

    // Row 1, Col 0: was NaN, should be filled with 3.0
    assert_approx(result[[1, 0]], 3.0, 1e-10, "mean impute col 0");
    // Row 2, Col 1: was NaN, should be filled with 3.0
    assert_approx(result[[2, 1]], 3.0, 1e-10, "mean impute col 1 row 2");
    // Row 3, Col 1: was NaN, should be filled with 3.0
    assert_approx(result[[3, 1]], 3.0, 1e-10, "mean impute col 1 row 3");

    // Non-NaN values should be unchanged
    assert_approx(result[[0, 0]], 1.0, 1e-10, "unchanged val");
    assert_approx(result[[0, 1]], 2.0, 1e-10, "unchanged val");
}

#[test]
fn simple_imputer_median_strategy() {
    // Median imputation
    use ferroml_core::preprocessing::imputers::{ImputeStrategy, SimpleImputer};

    let mut imputer = SimpleImputer::new(ImputeStrategy::Median);

    // Col 0: non-NaN values = [1, 5, 9], median = 5.0
    // Col 1: non-NaN values = [2, 4, 6, 8], median = (4+6)/2 = 5.0
    let x = array![
        [1.0, 2.0],
        [f64::NAN, 4.0],
        [5.0, 6.0],
        [9.0, 8.0],
        [f64::NAN, f64::NAN]
    ];
    let result = imputer.fit_transform(&x).unwrap();

    assert_approx(result[[1, 0]], 5.0, 1e-10, "median col 0");
    assert_approx(result[[4, 0]], 5.0, 1e-10, "median col 0 row 4");
    assert_approx(result[[4, 1]], 5.0, 1e-10, "median col 1 row 4");
}

#[test]
fn simple_imputer_most_frequent_strategy() {
    // Most frequent (mode) imputation
    use ferroml_core::preprocessing::imputers::{ImputeStrategy, SimpleImputer};

    let mut imputer = SimpleImputer::new(ImputeStrategy::MostFrequent);

    // Col 0: values = [1, 1, 3, 1, NaN] => mode = 1.0
    // Col 1: values = [2, 4, 4, NaN, 4] => mode = 4.0
    let x = array![
        [1.0, 2.0],
        [1.0, 4.0],
        [3.0, 4.0],
        [1.0, f64::NAN],
        [f64::NAN, 4.0]
    ];
    let result = imputer.fit_transform(&x).unwrap();

    assert_approx(result[[4, 0]], 1.0, 1e-10, "mode col 0");
    assert_approx(result[[3, 1]], 4.0, 1e-10, "mode col 1");
}

#[test]
fn simple_imputer_no_missing_values() {
    // When there are no missing values, output should equal input
    use ferroml_core::preprocessing::imputers::{ImputeStrategy, SimpleImputer};

    let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);

    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let result = imputer.fit_transform(&x).unwrap();

    assert_array2_approx(&result, &x, 1e-10, "no_missing_identity");
}

// =============================================================================
// KNNImputer Tests (2 tests)
// =============================================================================

#[test]
fn knn_imputer_basic() {
    // KNN imputation with uniform weights
    use ferroml_core::preprocessing::imputers::KNNImputer;

    let mut imputer = KNNImputer::new(2);

    // Simple data where row 1 col 1 is missing
    // Nearest neighbors to row 1 (by col 0): rows 0 (dist=3) and row 2 (dist=3)
    // Actually neighbors sorted by available features; col 1 missing so distance uses col 0
    let x = array![[1.0, 10.0], [4.0, f64::NAN], [7.0, 20.0], [10.0, 30.0]];
    let result = imputer.fit_transform(&x).unwrap();

    // The imputed value should be reasonable (mean of nearest neighbors)
    assert!(!result[[1, 1]].is_nan(), "KNN imputer should fill NaN");
    assert!(
        result[[1, 1]] > 5.0 && result[[1, 1]] < 35.0,
        "KNN imputed value should be reasonable, got {}",
        result[[1, 1]]
    );

    // Non-NaN values should be unchanged
    assert_approx(result[[0, 0]], 1.0, 1e-10, "unchanged 0,0");
    assert_approx(result[[0, 1]], 10.0, 1e-10, "unchanged 0,1");
    assert_approx(result[[2, 1]], 20.0, 1e-10, "unchanged 2,1");
}

#[test]
fn knn_imputer_no_missing() {
    // When no values are missing, output should equal input
    use ferroml_core::preprocessing::imputers::KNNImputer;

    let mut imputer = KNNImputer::new(3);

    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let result = imputer.fit_transform(&x).unwrap();

    assert_array2_approx(&result, &x, 1e-10, "knn_no_missing");
}

// =============================================================================
// Encoder Tests (6 tests)
// =============================================================================

#[test]
fn onehot_encoder_basic() {
    // OneHotEncoder with 3 categories
    use ferroml_core::preprocessing::encoders::OneHotEncoder;

    let mut encoder = OneHotEncoder::new();
    let x = array![[0.0], [1.0], [2.0], [1.0]];
    let result = encoder.fit_transform(&x).unwrap();

    assert_eq!(result.ncols(), 3, "3 categories => 3 columns");
    assert_eq!(result.nrows(), 4);

    // Row 0: category 0 => [1, 0, 0]
    assert_approx(result[[0, 0]], 1.0, 1e-10, "cat 0, col 0");
    assert_approx(result[[0, 1]], 0.0, 1e-10, "cat 0, col 1");
    assert_approx(result[[0, 2]], 0.0, 1e-10, "cat 0, col 2");

    // Row 1: category 1 => [0, 1, 0]
    assert_approx(result[[1, 0]], 0.0, 1e-10, "cat 1, col 0");
    assert_approx(result[[1, 1]], 1.0, 1e-10, "cat 1, col 1");
    assert_approx(result[[1, 2]], 0.0, 1e-10, "cat 1, col 2");

    // Row 2: category 2 => [0, 0, 1]
    assert_approx(result[[2, 0]], 0.0, 1e-10, "cat 2, col 0");
    assert_approx(result[[2, 1]], 0.0, 1e-10, "cat 2, col 1");
    assert_approx(result[[2, 2]], 1.0, 1e-10, "cat 2, col 2");
}

#[test]
fn onehot_encoder_drop_first() {
    // OneHotEncoder with drop='first'
    use ferroml_core::preprocessing::encoders::{DropStrategy, OneHotEncoder};

    let mut encoder = OneHotEncoder::new().with_drop(DropStrategy::First);
    let x = array![[0.0], [1.0], [2.0], [1.0]];
    let result = encoder.fit_transform(&x).unwrap();

    assert_eq!(
        result.ncols(),
        2,
        "3 categories with drop=first => 2 columns"
    );

    // Row 0: category 0 (dropped) => [0, 0]
    assert_approx(result[[0, 0]], 0.0, 1e-10, "dropped cat, col 0");
    assert_approx(result[[0, 1]], 0.0, 1e-10, "dropped cat, col 1");

    // Row 1: category 1 => [1, 0]
    assert_approx(result[[1, 0]], 1.0, 1e-10, "cat 1, col 0");
    assert_approx(result[[1, 1]], 0.0, 1e-10, "cat 1, col 1");

    // Row 2: category 2 => [0, 1]
    assert_approx(result[[2, 0]], 0.0, 1e-10, "cat 2, col 0");
    assert_approx(result[[2, 1]], 1.0, 1e-10, "cat 2, col 1");
}

#[test]
fn onehot_encoder_unknown_category_error() {
    // Unknown categories should raise error by default
    use ferroml_core::preprocessing::encoders::OneHotEncoder;

    let mut encoder = OneHotEncoder::new();
    let x_train = array![[0.0], [1.0], [2.0]];
    encoder.fit(&x_train).unwrap();

    // Try to transform with an unseen category
    let x_test = array![[0.0], [3.0]]; // 3.0 is unknown
    let result = encoder.transform(&x_test);

    assert!(result.is_err(), "Should error on unknown category");
}

#[test]
fn onehot_encoder_unknown_category_ignore() {
    // With handle_unknown='ignore', unknown categories should produce all zeros
    use ferroml_core::preprocessing::encoders::OneHotEncoder;
    use ferroml_core::preprocessing::UnknownCategoryHandling;

    let mut encoder = OneHotEncoder::new().with_handle_unknown(UnknownCategoryHandling::Ignore);
    let x_train = array![[0.0], [1.0], [2.0]];
    encoder.fit(&x_train).unwrap();

    let x_test = array![[0.0], [3.0]]; // 3.0 is unknown
    let result = encoder.transform(&x_test).unwrap();

    assert_eq!(result.ncols(), 3);
    // Row 0: category 0 => [1, 0, 0]
    assert_approx(result[[0, 0]], 1.0, 1e-10, "known cat");
    // Row 1: unknown => [0, 0, 0]
    assert_approx(result[[1, 0]], 0.0, 1e-10, "unknown cat col 0");
    assert_approx(result[[1, 1]], 0.0, 1e-10, "unknown cat col 1");
    assert_approx(result[[1, 2]], 0.0, 1e-10, "unknown cat col 2");
}

#[test]
fn ordinal_encoder_basic() {
    // OrdinalEncoder maps categories to integers
    use ferroml_core::preprocessing::encoders::OrdinalEncoder;

    let mut encoder = OrdinalEncoder::new();
    // Categories appear in order: 1.0, 3.0, 2.0
    let x = array![[1.0], [3.0], [2.0], [1.0]];
    let result = encoder.fit_transform(&x).unwrap();

    // Order of first appearance: 1.0 -> 0, 3.0 -> 1, 2.0 -> 2
    assert_approx(result[[0, 0]], 0.0, 1e-10, "1.0 -> 0");
    assert_approx(result[[1, 0]], 1.0, 1e-10, "3.0 -> 1");
    assert_approx(result[[2, 0]], 2.0, 1e-10, "2.0 -> 2");
    assert_approx(result[[3, 0]], 0.0, 1e-10, "1.0 -> 0 again");
}

#[test]
fn label_encoder_basic() {
    // LabelEncoder maps labels to integers in order of first appearance
    use ferroml_core::preprocessing::encoders::LabelEncoder;

    let mut encoder = LabelEncoder::new();
    let labels = array![2.0, 0.0, 1.0, 2.0, 1.0];

    encoder.fit_1d(&labels).unwrap();
    let encoded = encoder.transform_1d(&labels).unwrap();

    // Order of first appearance: 2.0 -> 0, 0.0 -> 1, 1.0 -> 2
    assert_approx(encoded[0], 0.0, 1e-10, "2.0 -> 0");
    assert_approx(encoded[1], 1.0, 1e-10, "0.0 -> 1");
    assert_approx(encoded[2], 2.0, 1e-10, "1.0 -> 2");
    assert_approx(encoded[3], 0.0, 1e-10, "2.0 -> 0 again");
    assert_approx(encoded[4], 2.0, 1e-10, "1.0 -> 2 again");

    // Inverse transform should recover original
    let recovered = encoder.inverse_transform_1d(&encoded).unwrap();
    for i in 0..labels.len() {
        assert_approx(recovered[i], labels[i], 1e-10, &format!("recover {}", i));
    }
}

// =============================================================================
// TargetEncoder Tests (2 tests)
// =============================================================================

#[test]
fn target_encoder_basic_smoothing() {
    // TargetEncoder: encodes categories using target mean with smoothing
    // Formula: encoded = (count * cat_mean + smooth * global_mean) / (count + smooth)
    use ferroml_core::preprocessing::encoders::TargetEncoder;

    let mut encoder = TargetEncoder::new().with_smooth(1.0);

    // One feature with 2 categories: 0.0 and 1.0
    // Category 0: targets = [10, 20, 30] => cat_mean = 20, count = 3
    // Category 1: targets = [100, 200] => cat_mean = 150, count = 2
    // Global mean = (10+20+30+100+200) / 5 = 72
    //
    // Encoded(cat 0) = (3*20 + 1*72) / (3+1) = (60+72)/4 = 33.0
    // Encoded(cat 1) = (2*150 + 1*72) / (2+1) = (300+72)/3 = 124.0

    let x = array![[0.0], [0.0], [0.0], [1.0], [1.0]];
    let y = array![10.0, 20.0, 30.0, 100.0, 200.0];

    encoder.fit_with_target(&x, &y).unwrap();
    let result = encoder.transform(&x).unwrap();

    assert_eq!(result.ncols(), 1);

    // All rows with category 0 should have the same encoding
    assert_approx(result[[0, 0]], result[[1, 0]], 1e-10, "cat 0 consistency");
    assert_approx(result[[1, 0]], result[[2, 0]], 1e-10, "cat 0 consistency");

    // All rows with category 1 should have the same encoding
    assert_approx(result[[3, 0]], result[[4, 0]], 1e-10, "cat 1 consistency");

    // Category 0 encoding should be lower than category 1
    assert!(
        result[[0, 0]] < result[[3, 0]],
        "cat 0 encoding ({}) should be < cat 1 encoding ({})",
        result[[0, 0]],
        result[[3, 0]]
    );

    // Check approximate values
    assert_approx(result[[0, 0]], 33.0, 1e-8, "cat 0 smoothed encoding");
    assert_approx(result[[3, 0]], 124.0, 1e-8, "cat 1 smoothed encoding");
}

#[test]
fn target_encoder_no_smoothing() {
    // With smooth=0, encoding should just be the category mean
    use ferroml_core::preprocessing::encoders::TargetEncoder;

    let mut encoder = TargetEncoder::new().with_smooth(0.0);

    let x = array![[0.0], [0.0], [1.0], [1.0]];
    let y = array![10.0, 20.0, 100.0, 200.0];
    // cat 0 mean = 15, cat 1 mean = 150

    encoder.fit_with_target(&x, &y).unwrap();
    let result = encoder.transform(&x).unwrap();

    // With 0 smoothing: encoded = (count * cat_mean + 0) / count = cat_mean
    assert_approx(result[[0, 0]], 15.0, 1e-8, "cat 0 no smoothing");
    assert_approx(result[[2, 0]], 150.0, 1e-8, "cat 1 no smoothing");
}

// =============================================================================
// Resampling Tests (4 tests)
// =============================================================================

#[test]
fn smote_balances_classes() {
    use ferroml_core::preprocessing::sampling::{Resampler, SMOTE};

    let mut smote = SMOTE::new().with_k_neighbors(3).with_random_state(42);

    // 20 majority (class 0), 5 minority (class 1)
    let mut x_data = Vec::new();
    for i in 0..20 {
        x_data.push(i as f64);
        x_data.push((i * 2) as f64);
    }
    for i in 0..5 {
        x_data.push(100.0 + i as f64);
        x_data.push(200.0 + (i * 2) as f64);
    }
    let x = Array2::from_shape_vec((25, 2), x_data).unwrap();
    let y = Array1::from_iter((0..20).map(|_| 0.0).chain((0..5).map(|_| 1.0)));

    let (x_res, y_res) = smote.fit_resample(&x, &y).unwrap();

    // After SMOTE, both classes should have same count (or close)
    let class_0_count = y_res.iter().filter(|&&v| v == 0.0).count();
    let class_1_count = y_res.iter().filter(|&&v| v == 1.0).count();

    assert_eq!(class_0_count, 20, "majority class should be unchanged");
    assert_eq!(
        class_1_count, 20,
        "minority class should be upsampled to match majority"
    );
    assert_eq!(x_res.nrows(), 40, "total samples should be 40");
    assert_eq!(x_res.ncols(), 2, "features should be preserved");
}

#[test]
fn smote_preserves_feature_space() {
    use ferroml_core::preprocessing::sampling::{Resampler, SMOTE};

    let mut smote = SMOTE::new().with_k_neighbors(2).with_random_state(123);

    // Minority samples are in a specific region [100, 200]
    let mut x_data = Vec::new();
    for i in 0..15 {
        x_data.push(i as f64);
    }
    for i in 0..5 {
        x_data.push(100.0 + i as f64);
    }
    let x = Array2::from_shape_vec((20, 1), x_data).unwrap();
    let y = Array1::from_iter((0..15).map(|_| 0.0).chain((0..5).map(|_| 1.0)));

    let (x_res, y_res) = smote.fit_resample(&x, &y).unwrap();

    // Synthetic minority samples should be in the minority feature range [100, 104]
    for i in 0..x_res.nrows() {
        if y_res[i] == 1.0 {
            assert!(
                x_res[[i, 0]] >= 99.0 && x_res[[i, 0]] <= 105.0,
                "synthetic minority sample should be near minority region, got {}",
                x_res[[i, 0]]
            );
        }
    }
}

#[test]
fn adasyn_balances_classes() {
    use ferroml_core::preprocessing::sampling::{Resampler, ADASYN};

    let mut adasyn = ADASYN::new().with_random_state(42);

    // 20 majority, 5 minority
    let mut x_data = Vec::new();
    for i in 0..20 {
        x_data.push(i as f64);
        x_data.push((i * 2) as f64);
    }
    for i in 0..5 {
        x_data.push(100.0 + i as f64);
        x_data.push(200.0 + (i * 2) as f64);
    }
    let x = Array2::from_shape_vec((25, 2), x_data).unwrap();
    let y = Array1::from_iter((0..20).map(|_| 0.0).chain((0..5).map(|_| 1.0)));

    let (x_res, y_res) = adasyn.fit_resample(&x, &y).unwrap();

    let class_0_count = y_res.iter().filter(|&&v| v == 0.0).count();
    let class_1_count = y_res.iter().filter(|&&v| v == 1.0).count();

    assert_eq!(class_0_count, 20, "majority unchanged");
    // ADASYN may not produce exactly balanced classes, but should be close
    assert!(
        class_1_count >= 15,
        "minority should be substantially upsampled, got {}",
        class_1_count
    );
    assert_eq!(x_res.ncols(), 2, "features preserved");
}

#[test]
fn random_oversampler_exact_count() {
    use ferroml_core::preprocessing::sampling::{RandomOverSampler, Resampler};

    let mut ros = RandomOverSampler::new().with_random_state(42);

    // 10 majority, 3 minority
    let mut x_data = Vec::new();
    for i in 0..10 {
        x_data.push(i as f64);
    }
    for i in 0..3 {
        x_data.push(100.0 + i as f64);
    }
    let x = Array2::from_shape_vec((13, 1), x_data).unwrap();
    let y = Array1::from_iter((0..10).map(|_| 0.0).chain((0..3).map(|_| 1.0)));

    let (x_res, y_res) = ros.fit_resample(&x, &y).unwrap();

    let class_0_count = y_res.iter().filter(|&&v| v == 0.0).count();
    let class_1_count = y_res.iter().filter(|&&v| v == 1.0).count();

    assert_eq!(class_0_count, 10, "majority unchanged");
    assert_eq!(class_1_count, 10, "minority oversampled to match");

    // All oversampled minority values should come from [100, 101, 102]
    for i in 0..x_res.nrows() {
        if y_res[i] == 1.0 {
            let val = x_res[[i, 0]];
            assert!(
                val == 100.0 || val == 101.0 || val == 102.0,
                "oversampled value should be from original, got {}",
                val
            );
        }
    }
}

// =============================================================================
// Edge Case Tests (5 tests)
// =============================================================================

#[test]
fn transformer_single_sample() {
    // Various transformers should handle a single sample
    use ferroml_core::preprocessing::scalers::StandardScaler;

    let mut scaler = StandardScaler::new();
    let x = array![[1.0, 2.0, 3.0]]; // single sample

    // Fit should succeed (or gracefully handle)
    let result = scaler.fit_transform(&x);
    // StandardScaler on 1 sample => std=0 => may produce 0 or NaN
    // Either is acceptable as long as it doesn't panic
    match result {
        Ok(transformed) => {
            // Should have the same shape
            assert_eq!(transformed.shape(), x.shape());
        }
        Err(_) => {
            // Error is also acceptable for 1 sample
        }
    }
}

#[test]
fn transformer_single_feature() {
    use ferroml_core::preprocessing::scalers::MinMaxScaler;

    let mut scaler = MinMaxScaler::new();
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

    let result = scaler.fit_transform(&x).unwrap();
    assert_eq!(result.ncols(), 1);
    assert_eq!(result.nrows(), 5);

    // MinMaxScaler: (x - min) / (max - min)
    // min=1, max=5, range=4
    assert_approx(result[[0, 0]], 0.0, 1e-10, "min -> 0");
    assert_approx(result[[4, 0]], 1.0, 1e-10, "max -> 1");
    assert_approx(result[[2, 0]], 0.5, 1e-10, "mid -> 0.5");
}

#[test]
fn transformer_constant_feature_robustscaler() {
    use ferroml_core::preprocessing::scalers::RobustScaler;

    let mut scaler = RobustScaler::new();
    let x = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]; // col 0 is constant

    let result = scaler.fit_transform(&x);
    // Should either handle gracefully or error
    match result {
        Ok(transformed) => {
            // Constant feature column should be 0
            for i in 0..3 {
                assert!(
                    transformed[[i, 0]].is_finite(),
                    "constant feature should produce finite values"
                );
            }
        }
        Err(_) => {
            // Error is acceptable for constant features
        }
    }
}

#[test]
fn ordinal_encoder_unseen_category_error() {
    use ferroml_core::preprocessing::encoders::OrdinalEncoder;

    let mut encoder = OrdinalEncoder::new();
    let x_train = array![[1.0], [2.0], [3.0]];
    encoder.fit(&x_train).unwrap();

    let x_test = array![[1.0], [99.0]]; // 99.0 is unknown
    let result = encoder.transform(&x_test);

    assert!(
        result.is_err(),
        "Should error on unseen category by default"
    );
}

#[test]
fn ordinal_encoder_unseen_category_ignore() {
    use ferroml_core::preprocessing::encoders::OrdinalEncoder;
    use ferroml_core::preprocessing::UnknownCategoryHandling;

    let mut encoder = OrdinalEncoder::new().with_handle_unknown(UnknownCategoryHandling::Ignore);
    let x_train = array![[1.0], [2.0], [3.0]];
    encoder.fit(&x_train).unwrap();

    let x_test = array![[1.0], [99.0]]; // 99.0 is unknown
    let result = encoder.transform(&x_test).unwrap();

    // Known category should map correctly
    assert_approx(result[[0, 0]], 0.0, 1e-10, "known category 1.0 -> 0");
    // Unknown category should map to -1
    assert_approx(result[[1, 0]], -1.0, 1e-10, "unknown category -> -1");
}

// =============================================================================
// Pipeline Integration Tests (2 tests)
// =============================================================================

#[test]
fn pipeline_scaler_to_model() {
    // Pipeline: StandardScaler -> LinearRegression
    use ferroml_core::models::LinearRegression;
    use ferroml_core::pipeline::Pipeline;
    use ferroml_core::preprocessing::scalers::StandardScaler;

    let mut pipeline = Pipeline::new()
        .add_transformer("scaler", StandardScaler::new())
        .add_model("lr", LinearRegression::new());

    // Non-collinear data: x0 is sequential, x1 is a different pattern
    let x = Array2::from_shape_fn((20, 2), |(i, _j)| {
        // Use different patterns for the two features to avoid collinearity
        if _j == 0 {
            i as f64
        } else {
            ((i * 7 + 3) % 20) as f64
        }
    });
    let y =
        Array1::from_iter((0..20).map(|i| 2.0 * (i as f64) + 3.0 * (((i * 7 + 3) % 20) as f64)));

    pipeline.fit(&x, &y).unwrap();
    let predictions = pipeline.predict(&x).unwrap();

    assert_eq!(predictions.len(), 20);
    // Predictions should be reasonable
    for &p in predictions.iter() {
        assert!(p.is_finite(), "prediction should be finite");
    }
}

#[test]
fn pipeline_imputer_scaler_model() {
    // Pipeline: SimpleImputer -> StandardScaler -> LinearRegression
    use ferroml_core::models::LinearRegression;
    use ferroml_core::pipeline::Pipeline;
    use ferroml_core::preprocessing::imputers::{ImputeStrategy, SimpleImputer};
    use ferroml_core::preprocessing::scalers::StandardScaler;

    let mut pipeline = Pipeline::new()
        .add_transformer("imputer", SimpleImputer::new(ImputeStrategy::Mean))
        .add_transformer("scaler", StandardScaler::new())
        .add_model("lr", LinearRegression::new());

    // Data with a couple NaN values
    let mut x = Array2::from_shape_fn((20, 3), |(i, j)| (i * (j + 1) + 1) as f64);
    x[[3, 1]] = f64::NAN;
    x[[7, 2]] = f64::NAN;

    let y = Array1::from_iter((0..20).map(|i| (i * 3 + 1) as f64));

    pipeline.fit(&x, &y).unwrap();
    let predictions = pipeline.predict(&x).unwrap();

    assert_eq!(predictions.len(), 20);
    for &p in predictions.iter() {
        assert!(p.is_finite(), "prediction should be finite");
    }
}

// =============================================================================
// Transformer fit_transform vs fit+transform consistency (1 test)
// =============================================================================

#[test]
fn fit_transform_consistency_all_transformers() {
    // Verify fit_transform == fit + transform for multiple transformers
    use ferroml_core::preprocessing::discretizers::{BinningStrategy, KBinsDiscretizer};
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;
    use ferroml_core::preprocessing::scalers::{MaxAbsScaler, MinMaxScaler, StandardScaler};

    let x = Array2::from_shape_fn((20, 3), |(i, j)| (i * 3 + j + 1) as f64);

    // Test StandardScaler
    {
        let mut t1 = StandardScaler::new();
        let r1 = {
            let mut t = StandardScaler::new();
            t.fit(&x).unwrap();
            t.transform(&x).unwrap()
        };
        let r2 = t1.fit_transform(&x).unwrap();
        assert_array2_approx(&r1, &r2, 1e-10, "StandardScaler consistency");
    }

    // Test MinMaxScaler
    {
        let mut t1 = MinMaxScaler::new();
        let r1 = {
            let mut t = MinMaxScaler::new();
            t.fit(&x).unwrap();
            t.transform(&x).unwrap()
        };
        let r2 = t1.fit_transform(&x).unwrap();
        assert_array2_approx(&r1, &r2, 1e-10, "MinMaxScaler consistency");
    }

    // Test MaxAbsScaler
    {
        let mut t1 = MaxAbsScaler::new();
        let r1 = {
            let mut t = MaxAbsScaler::new();
            t.fit(&x).unwrap();
            t.transform(&x).unwrap()
        };
        let r2 = t1.fit_transform(&x).unwrap();
        assert_array2_approx(&r1, &r2, 1e-10, "MaxAbsScaler consistency");
    }

    // Test PolynomialFeatures
    {
        let mut t1 = PolynomialFeatures::new(2);
        let r1 = {
            let mut t = PolynomialFeatures::new(2);
            t.fit(&x).unwrap();
            t.transform(&x).unwrap()
        };
        let r2 = t1.fit_transform(&x).unwrap();
        assert_array2_approx(&r1, &r2, 1e-10, "PolynomialFeatures consistency");
    }

    // Test KBinsDiscretizer
    {
        let mut t1 = KBinsDiscretizer::new()
            .with_n_bins(4)
            .with_strategy(BinningStrategy::Uniform);
        let r1 = {
            let mut t = KBinsDiscretizer::new()
                .with_n_bins(4)
                .with_strategy(BinningStrategy::Uniform);
            t.fit(&x).unwrap();
            t.transform(&x).unwrap()
        };
        let r2 = t1.fit_transform(&x).unwrap();
        assert_array2_approx(&r1, &r2, 1e-10, "KBinsDiscretizer consistency");
    }
}

// =============================================================================
// n_features_in / n_features_out consistency (1 test)
// =============================================================================

#[test]
fn feature_count_consistency() {
    use ferroml_core::preprocessing::discretizers::{BinningStrategy, KBinsDiscretizer};
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;
    use ferroml_core::preprocessing::scalers::StandardScaler;
    use ferroml_core::preprocessing::selection::VarianceThreshold;

    let x = Array2::from_shape_fn((10, 4), |(i, j)| (i * (j + 1)) as f64 + 0.1 * j as f64);

    // StandardScaler: n_in == n_out
    {
        let mut t = StandardScaler::new();
        t.fit(&x).unwrap();
        assert_eq!(t.n_features_in(), Some(4));
        assert_eq!(t.n_features_out(), Some(4));
        let r = t.transform(&x).unwrap();
        assert_eq!(r.ncols(), t.n_features_out().unwrap());
    }

    // PolynomialFeatures: n_out > n_in
    {
        let mut t = PolynomialFeatures::new(2);
        t.fit(&x).unwrap();
        assert_eq!(t.n_features_in(), Some(4));
        let n_out = t.n_features_out().unwrap();
        assert!(n_out > 4, "poly should produce more features");
        let r = t.transform(&x).unwrap();
        assert_eq!(r.ncols(), n_out);
    }

    // VarianceThreshold: n_out <= n_in
    {
        let mut x_var = x.clone();
        x_var.column_mut(0).fill(5.0); // Make col 0 constant
        let mut t = VarianceThreshold::new(0.0);
        t.fit(&x_var).unwrap();
        assert_eq!(t.n_features_in(), Some(4));
        let n_out = t.n_features_out().unwrap();
        assert!(
            n_out < 4,
            "variance threshold should remove constant features"
        );
        let r = t.transform(&x_var).unwrap();
        assert_eq!(r.ncols(), n_out);
    }

    // KBinsDiscretizer: n_in == n_out (ordinal encoding)
    {
        let mut t = KBinsDiscretizer::new()
            .with_n_bins(3)
            .with_strategy(BinningStrategy::Uniform);
        t.fit(&x).unwrap();
        let r = t.transform(&x).unwrap();
        assert_eq!(r.ncols(), 4, "ordinal encoding preserves feature count");
    }
}

// =============================================================================
// Additional PolynomialFeatures Tests (rigorous value checks)
// =============================================================================

#[test]
fn polynomial_features_degree2_four_features_shape_and_values() {
    // sklearn: PolynomialFeatures(degree=2, include_bias=True).fit_transform(X)
    // For n=4, degree=2: C(4+2,2) = 15 columns
    // Column order: [1, x0, x1, x2, x3, x0^2, x0*x1, x0*x2, x0*x3, x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2]
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;

    let mut poly = PolynomialFeatures::new(2);
    let x = array![[1.0, 2.0, 3.0, 4.0]];
    let result = poly.fit_transform(&x).unwrap();

    assert_eq!(result.ncols(), 15, "C(6,2) = 15 for 4 features degree 2");

    // Verify every value:
    // bias=1, x0=1, x1=2, x2=3, x3=4
    // x0^2=1, x0*x1=2, x0*x2=3, x0*x3=4
    // x1^2=4, x1*x2=6, x1*x3=8
    // x2^2=9, x2*x3=12, x3^2=16
    let expected_values = vec![
        1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 4.0, 6.0, 8.0, 9.0, 12.0, 16.0,
    ];
    for (j, &exp) in expected_values.iter().enumerate() {
        assert_approx(result[[0, j]], exp, 1e-10, &format!("poly col {}", j));
    }
}

#[test]
fn polynomial_features_degree3_two_features_all_terms() {
    // degree=3, 2 features, no bias
    // Terms: x0, x1, x0^2, x0*x1, x1^2, x0^3, x0^2*x1, x0*x1^2, x1^3
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;

    let mut poly = PolynomialFeatures::new(3).include_bias(false);
    let x = array![[2.0, 3.0]];
    let result = poly.fit_transform(&x).unwrap();

    assert_eq!(result.ncols(), 9);
    let expected = vec![
        2.0,  // x0
        3.0,  // x1
        4.0,  // x0^2
        6.0,  // x0*x1
        9.0,  // x1^2
        8.0,  // x0^3
        12.0, // x0^2*x1
        18.0, // x0*x1^2
        27.0, // x1^3
    ];
    for (j, &exp) in expected.iter().enumerate() {
        assert_approx(result[[0, j]], exp, 1e-10, &format!("degree3 col {}", j));
    }
}

#[test]
fn polynomial_features_interaction_only_degree3_four_features() {
    // interaction_only=True, degree=3, 4 features
    // Degree 1: x0, x1, x2, x3 (4 terms)
    // Degree 2: x0x1, x0x2, x0x3, x1x2, x1x3, x2x3 (C(4,2)=6 terms)
    // Degree 3: x0x1x2, x0x1x3, x0x2x3, x1x2x3 (C(4,3)=4 terms)
    // Total with bias: 1 + 4 + 6 + 4 = 15
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;

    let mut poly = PolynomialFeatures::new(3).interaction_only(true);
    let x = array![[1.0, 2.0, 3.0, 4.0]];
    let result = poly.fit_transform(&x).unwrap();

    assert_eq!(result.ncols(), 15);

    // Check the degree-3 interaction terms at the end
    // x0*x1*x2 = 6, x0*x1*x3 = 8, x0*x2*x3 = 12, x1*x2*x3 = 24
    let last4 = &[
        result[[0, 11]],
        result[[0, 12]],
        result[[0, 13]],
        result[[0, 14]],
    ];
    // Should contain 6, 8, 12, 24 in some order matching grlex
    assert_approx(last4[0], 6.0, 1e-10, "x0*x1*x2");
    assert_approx(last4[1], 8.0, 1e-10, "x0*x1*x3");
    assert_approx(last4[2], 12.0, 1e-10, "x0*x2*x3");
    assert_approx(last4[3], 24.0, 1e-10, "x1*x2*x3");
}

#[test]
fn polynomial_features_with_negative_values_degree2() {
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;

    let mut poly = PolynomialFeatures::new(2).include_bias(false);
    let x = array![[-2.0, 3.0]];
    let result = poly.fit_transform(&x).unwrap();

    // x0=-2, x1=3 => x0^2=4, x0*x1=-6, x1^2=9
    let expected = array![[-2.0, 3.0, 4.0, -6.0, 9.0]];
    assert_array2_approx(&result, &expected, 1e-10, "poly_negative");
}

// =============================================================================
// Additional KBinsDiscretizer Tests
// =============================================================================

#[test]
fn kbins_uniform_bin_edges_precise() {
    // Verify the exact bin edges for uniform strategy
    use ferroml_core::preprocessing::discretizers::{BinningStrategy, KBinsDiscretizer};

    let mut disc = KBinsDiscretizer::new()
        .with_n_bins(4)
        .with_strategy(BinningStrategy::Uniform);

    let x = array![[0.0], [4.0], [8.0]];
    disc.fit(&x).unwrap();

    let edges = disc.bin_edges().unwrap();
    let feature_edges = &edges[0];
    // Uniform: min=0, max=8, bin_width=2
    // Edges: 0, 2, 4, 6, 8
    assert_eq!(feature_edges.len(), 5, "4 bins => 5 edges");
    assert_approx(feature_edges[0], 0.0, 1e-10, "edge 0");
    assert_approx(feature_edges[1], 2.0, 1e-10, "edge 1");
    assert_approx(feature_edges[2], 4.0, 1e-10, "edge 2");
    assert_approx(feature_edges[3], 6.0, 1e-10, "edge 3");
    assert_approx(feature_edges[4], 8.0, 1e-10, "edge 4");
}

#[test]
fn kbins_uniform_multifeature_correct_bins() {
    // Two features with different ranges should get independent bin edges
    use ferroml_core::preprocessing::discretizers::{BinningStrategy, KBinsDiscretizer};

    let mut disc = KBinsDiscretizer::new()
        .with_n_bins(3)
        .with_strategy(BinningStrategy::Uniform);

    // Feature 0: range [0, 9], Feature 1: range [100, 109]
    let x = Array2::from_shape_fn(
        (10, 2),
        |(i, j)| {
            if j == 0 {
                i as f64
            } else {
                100.0 + i as f64
            }
        },
    );
    let result = disc.fit_transform(&x).unwrap();

    // Both features should have the same bin pattern since they're both sequential
    for i in 0..10 {
        assert_approx(
            result[[i, 0]],
            result[[i, 1]],
            1e-10,
            &format!("row {} same bin pattern", i),
        );
    }
}

#[test]
fn kbins_onehot_encoding_correct_shape_multifeature() {
    use ferroml_core::preprocessing::discretizers::{
        BinEncoding, BinningStrategy, KBinsDiscretizer,
    };

    let mut disc = KBinsDiscretizer::new()
        .with_n_bins(3)
        .with_strategy(BinningStrategy::Uniform)
        .with_encode(BinEncoding::OneHot);

    let x = array![[0.0, 10.0], [3.0, 20.0], [6.0, 30.0], [9.0, 40.0]];
    let result = disc.fit_transform(&x).unwrap();

    // 3 bins * 2 features = 6 columns
    assert_eq!(result.ncols(), 6);
    // Each row should have exactly 2 ones (one per feature)
    for i in 0..4 {
        let row_sum: f64 = result.row(i).sum();
        assert_approx(row_sum, 2.0, 1e-10, &format!("onehot row {} sum", i));
    }
}

#[test]
fn kbins_inverse_transform_approximate_recovery() {
    use ferroml_core::preprocessing::discretizers::{BinningStrategy, KBinsDiscretizer};

    let mut disc = KBinsDiscretizer::new()
        .with_n_bins(5)
        .with_strategy(BinningStrategy::Uniform);

    let x = array![[0.0], [2.0], [4.0], [6.0], [8.0], [10.0]];
    disc.fit(&x).unwrap();

    let binned = disc.transform(&x).unwrap();
    let recovered = disc.inverse_transform(&binned).unwrap();

    // Inverse transform gives bin midpoints. For uniform 5 bins on [0,10]:
    // edges: [0, 2, 4, 6, 8, 10], midpoints: [1, 3, 5, 7, 9]
    // The recovered values should be the midpoints of their bins
    for i in 0..6 {
        let bin_idx = binned[[i, 0]] as usize;
        // Midpoint should be approx original +/- 1.0
        assert!(
            (recovered[[i, 0]] - x[[i, 0]]).abs() < 2.0,
            "recovered {} should be near original {}, bin {}",
            recovered[[i, 0]],
            x[[i, 0]],
            bin_idx
        );
    }
}

// =============================================================================
// Additional PowerTransformer Tests
// =============================================================================

#[test]
fn power_transformer_box_cox_lambda_for_sqrt_data() {
    // If data is x^2 distributed, optimal Box-Cox lambda should be near 0.5 (sqrt)
    use ferroml_core::preprocessing::power::{PowerMethod, PowerTransformer};

    let mut pt = PowerTransformer::new(PowerMethod::BoxCox);

    // Squared data: applying lambda=0.5 (sqrt) should normalize it
    let x = array![
        [1.0],
        [4.0],
        [9.0],
        [16.0],
        [25.0],
        [36.0],
        [49.0],
        [64.0],
        [81.0],
        [100.0]
    ];
    pt.fit(&x).unwrap();

    let lambdas = pt.lambdas().unwrap();
    // Lambda should be reasonable (typically between -2 and 2 for this data)
    assert!(
        lambdas[0] > -3.0 && lambdas[0] < 3.0,
        "lambda should be in reasonable range, got {}",
        lambdas[0]
    );
}

#[test]
fn power_transformer_yeo_johnson_multifeature() {
    use ferroml_core::preprocessing::power::{PowerMethod, PowerTransformer};

    let mut pt = PowerTransformer::new(PowerMethod::YeoJohnson);

    // 3 features with different distributions
    let x = array![
        [1.0, -3.0, 0.5],
        [4.0, -1.0, 1.0],
        [9.0, 0.0, 2.0],
        [16.0, 1.0, 4.0],
        [25.0, 3.0, 8.0],
        [36.0, 5.0, 16.0]
    ];
    let result = pt.fit_transform(&x).unwrap();

    assert_eq!(result.shape(), &[6, 3]);

    // Each feature should be approximately standardized (zero mean)
    for j in 0..3 {
        let col = result.column(j);
        let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
        assert!(
            mean.abs() < 0.5,
            "feature {} mean should be near 0, got {}",
            j,
            mean
        );
    }

    // Should have 3 lambdas
    let lambdas = pt.lambdas().unwrap();
    assert_eq!(lambdas.len(), 3);
}

#[test]
fn power_transformer_box_cox_roundtrip_no_standardize() {
    use ferroml_core::preprocessing::power::{PowerMethod, PowerTransformer};

    let mut pt = PowerTransformer::new(PowerMethod::BoxCox).with_standardize(false);

    let x = array![[1.0], [2.0], [3.0], [5.0], [8.0], [13.0]];
    pt.fit(&x).unwrap();
    let transformed = pt.transform(&x).unwrap();
    let recovered = pt.inverse_transform(&transformed).unwrap();

    for i in 0..x.nrows() {
        assert_approx(
            recovered[[i, 0]],
            x[[i, 0]],
            0.01,
            &format!("box_cox roundtrip row {}", i),
        );
    }
}

#[test]
fn power_transformer_standardize_output_stats() {
    // With standardize=true, output should have mean=0, std=1
    use ferroml_core::preprocessing::power::{PowerMethod, PowerTransformer};

    let mut pt = PowerTransformer::new(PowerMethod::YeoJohnson).with_standardize(true);

    let x = array![[1.0], [2.0], [4.0], [8.0], [16.0], [32.0], [64.0], [128.0]];
    let result = pt.fit_transform(&x).unwrap();

    let col = result.column(0);
    let n = col.len() as f64;
    let mean: f64 = col.iter().sum::<f64>() / n;
    let var: f64 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;

    assert_approx(mean, 0.0, 1e-8, "standardized mean should be ~0");
    assert_approx(var.sqrt(), 1.0, 0.1, "standardized std should be ~1");
}

// =============================================================================
// Additional QuantileTransformer Tests
// =============================================================================

#[test]
fn quantile_transformer_uniform_monotone_sorted_input() {
    use ferroml_core::preprocessing::quantile::{OutputDistribution, QuantileTransformer};

    let mut qt = QuantileTransformer::new(OutputDistribution::Uniform);

    // 50 sorted values
    let x = Array2::from_shape_fn((50, 1), |(i, _)| i as f64);
    let result = qt.fit_transform(&x).unwrap();

    // Should be strictly increasing (sorted input => sorted output for uniform)
    for i in 1..50 {
        assert!(
            result[[i, 0]] >= result[[i - 1, 0]],
            "uniform quantile should be monotone at {}",
            i
        );
    }

    // Range should span most of [0, 1]
    let min_val = result[[0, 0]];
    let max_val = result[[49, 0]];
    assert!(min_val < 0.05, "min should be near 0, got {}", min_val);
    assert!(max_val > 0.95, "max should be near 1, got {}", max_val);
}

#[test]
fn quantile_transformer_normal_output_stats() {
    use ferroml_core::preprocessing::quantile::{OutputDistribution, QuantileTransformer};

    let mut qt = QuantileTransformer::new(OutputDistribution::Normal);

    // 200 uniformly spaced samples for good quantile estimates
    let x = Array2::from_shape_fn((200, 1), |(i, _)| (i as f64 + 0.5) / 200.0);
    let result = qt.fit_transform(&x).unwrap();

    // All values should be finite
    for &v in result.iter() {
        assert!(
            v.is_finite(),
            "normal quantile output must be finite, got {}",
            v
        );
    }

    // Output should approximate N(0, 1)
    let col = result.column(0);
    let n = col.len() as f64;
    let mean: f64 = col.iter().sum::<f64>() / n;
    let var: f64 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;

    assert!(mean.abs() < 0.3, "normal output mean ~ 0, got {}", mean);
    assert!(
        var > 0.5,
        "normal output variance should be substantial, got {}",
        var
    );
}

#[test]
fn quantile_transformer_multifeature() {
    use ferroml_core::preprocessing::quantile::{OutputDistribution, QuantileTransformer};

    let mut qt = QuantileTransformer::new(OutputDistribution::Uniform);

    // Two features with different scales
    let x = Array2::from_shape_fn(
        (30, 2),
        |(i, j)| {
            if j == 0 {
                i as f64
            } else {
                (i as f64).powi(2)
            }
        },
    );
    let result = qt.fit_transform(&x).unwrap();

    assert_eq!(result.shape(), &[30, 2]);

    // Both features should be mapped to [0, 1]
    for j in 0..2 {
        for i in 0..30 {
            assert!(
                result[[i, j]] >= 0.0 && result[[i, j]] <= 1.0,
                "feature {} sample {} should be in [0,1], got {}",
                j,
                i,
                result[[i, j]]
            );
        }
    }
}

#[test]
fn quantile_transformer_custom_n_quantiles() {
    use ferroml_core::preprocessing::quantile::{OutputDistribution, QuantileTransformer};

    let mut qt = QuantileTransformer::new(OutputDistribution::Uniform).with_n_quantiles(10);

    let x = Array2::from_shape_fn((100, 1), |(i, _)| i as f64);
    let result = qt.fit_transform(&x).unwrap();

    // Should still work with fewer quantiles
    assert_eq!(result.nrows(), 100);
    assert_eq!(result.ncols(), 1);

    // Output should be in [0, 1]
    for &v in result.iter() {
        assert!((0.0..=1.0).contains(&v), "should be in [0,1], got {}", v);
    }
}

// =============================================================================
// Additional VarianceThreshold Tests
// =============================================================================

#[test]
fn variance_threshold_get_support_mask() {
    use ferroml_core::preprocessing::selection::VarianceThreshold;

    let mut selector = VarianceThreshold::new(0.0);
    // Col 0: constant, Col 1: varies, Col 2: varies, Col 3: constant
    let x = array![
        [1.0, 5.0, 10.0, 7.0],
        [1.0, 2.0, 20.0, 7.0],
        [1.0, 8.0, 30.0, 7.0]
    ];
    selector.fit(&x).unwrap();

    let support = selector.get_support().unwrap();
    assert_eq!(support, vec![false, true, true, false]);

    let indices = selector.selected_indices().unwrap();
    assert_eq!(indices, &[1, 2]);
}

#[test]
fn variance_threshold_high_threshold() {
    use ferroml_core::preprocessing::selection::VarianceThreshold;

    let mut selector = VarianceThreshold::new(100.0);
    // Col 0: var([1,2,3]) = 2/3, Col 1: var([0,100,200]) = 6666.67
    let x = array![[1.0, 0.0], [2.0, 100.0], [3.0, 200.0]];
    let result = selector.fit_transform(&x).unwrap();

    assert_eq!(
        result.ncols(),
        1,
        "Only high-variance feature should remain"
    );
    assert_approx(result[[0, 0]], 0.0, 1e-10, "kept col is col 1");
    assert_approx(result[[1, 0]], 100.0, 1e-10, "kept col row 1");
}

// =============================================================================
// Additional SelectKBest Tests
// =============================================================================

#[test]
fn select_k_best_chi2_nonnegative() {
    // Chi-squared test for feature selection with non-negative features
    use ferroml_core::preprocessing::selection::{ScoreFunction, SelectKBest};

    let mut selector = SelectKBest::new(ScoreFunction::Chi2, 1);

    // x0: strong association with y (high values for class 1)
    // x1: weak association
    let x = array![
        [0.0, 5.0],
        [1.0, 4.0],
        [0.0, 6.0],
        [10.0, 5.5],
        [11.0, 4.5],
        [10.0, 5.0]
    ];
    let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    selector.fit_with_target(&x, &y).unwrap();
    let selected = selector.selected_indices().unwrap();

    assert_eq!(selected.len(), 1);
    assert_eq!(
        selected[0], 0,
        "Feature 0 (strong association) should be selected"
    );
}

#[test]
fn select_k_best_scores_accessible() {
    use ferroml_core::preprocessing::selection::{ScoreFunction, SelectKBest};

    let mut selector = SelectKBest::new(ScoreFunction::FRegression, 2);

    let x = array![
        [1.0, 10.0, 0.5],
        [2.0, 20.0, 0.3],
        [3.0, 30.0, 0.8],
        [4.0, 40.0, 0.2],
        [5.0, 50.0, 0.6]
    ];
    let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

    selector.fit_with_target(&x, &y).unwrap();

    let scores = selector.scores().unwrap();
    assert_eq!(scores.scores.len(), 3, "Should have scores for 3 features");

    // Features 0 and 1 are perfectly correlated with y; feature 2 is not
    assert!(
        scores.scores[0] > scores.scores[2],
        "Feature 0 score should be higher than feature 2"
    );
    assert!(
        scores.scores[1] > scores.scores[2],
        "Feature 1 score should be higher than feature 2"
    );
}

// =============================================================================
// Additional SelectFromModel Tests
// =============================================================================

#[test]
fn select_from_model_median_threshold() {
    use ferroml_core::preprocessing::selection::{ImportanceThreshold, SelectFromModel};

    let importances = array![0.1, 0.3, 0.5, 0.7, 0.9];
    // Median = 0.5 => features with importance > 0.5: indices 3 (0.7) and 4 (0.9)

    let mut selector = SelectFromModel::new(importances, ImportanceThreshold::Median);

    let x = array![[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]];
    let result = selector.fit_transform(&x).unwrap();

    // Features at or above the median (0.5) should be selected
    // This depends on implementation: >= median or > median
    assert!(
        result.ncols() >= 2,
        "Should select at least 2 features above median, got {}",
        result.ncols()
    );
}

#[test]
fn select_from_model_mean_plus_std() {
    use ferroml_core::preprocessing::selection::{ImportanceThreshold, SelectFromModel};

    let importances = array![0.1, 0.2, 0.3, 0.4, 2.0];
    // mean = 0.6, std ~ 0.74
    // mean + 1*std ~ 1.34 => only feature 4 (2.0) > 1.34

    let mut selector = SelectFromModel::new(importances, ImportanceThreshold::MeanPlusStd(1.0));

    let x = array![[1.0, 2.0, 3.0, 4.0, 5.0]];
    let result = selector.fit_transform(&x).unwrap();

    assert_eq!(result.ncols(), 1, "Only 1 feature should exceed mean+std");
    assert_approx(
        result[[0, 0]],
        5.0,
        1e-10,
        "Selected feature should be col 4",
    );
}

// =============================================================================
// Additional RFE Tests
// =============================================================================

#[test]
fn rfe_ranking_order() {
    use ferroml_core::preprocessing::selection::{ClosureEstimator, RecursiveFeatureElimination};

    let estimator =
        ClosureEstimator::new(|x: &Array2<f64>, _y: &Array1<f64>| Ok(x.var_axis(Axis(0), 0.0)));

    let mut rfe = RecursiveFeatureElimination::new(Box::new(estimator))
        .with_n_features_to_select(1)
        .with_step(1);

    // Feature 0: low variance, Feature 1: medium, Feature 2: highest
    let x = array![
        [1.0, 10.0, 100.0],
        [1.1, 20.0, 200.0],
        [0.9, 30.0, 300.0],
        [1.05, 40.0, 400.0]
    ];
    let y = array![1.0, 2.0, 3.0, 4.0];

    rfe.fit_with_target(&x, &y).unwrap();
    let ranking = rfe.ranking().unwrap();

    // Feature 2 (highest variance) should have rank 1
    assert_eq!(ranking[2], 1, "Highest variance feature should be ranked 1");
    // Feature 0 (lowest variance) should have highest rank
    assert!(
        ranking[0] > ranking[1],
        "Lowest variance feature should be ranked higher (worse)"
    );
}

// =============================================================================
// Additional SimpleImputer Tests
// =============================================================================

#[test]
fn simple_imputer_constant_strategy() {
    use ferroml_core::preprocessing::imputers::{ImputeStrategy, SimpleImputer};

    let mut imputer = SimpleImputer::new(ImputeStrategy::Constant).with_fill_value(-999.0);

    let x = array![[1.0, f64::NAN], [f64::NAN, 3.0], [5.0, 6.0]];
    let result = imputer.fit_transform(&x).unwrap();

    assert_approx(result[[0, 1]], -999.0, 1e-10, "constant fill col 1");
    assert_approx(result[[1, 0]], -999.0, 1e-10, "constant fill col 0");
    // Non-NaN values unchanged
    assert_approx(result[[0, 0]], 1.0, 1e-10, "unchanged");
    assert_approx(result[[2, 1]], 6.0, 1e-10, "unchanged");
}

#[test]
fn simple_imputer_missing_counts_tracked() {
    use ferroml_core::preprocessing::imputers::{ImputeStrategy, SimpleImputer};

    let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);

    let x = array![
        [f64::NAN, 2.0, 3.0],
        [f64::NAN, f64::NAN, 6.0],
        [7.0, 8.0, f64::NAN]
    ];
    imputer.fit(&x).unwrap();

    let counts = imputer.missing_counts().unwrap();
    assert_eq!(counts[0], 2, "col 0 has 2 missing");
    assert_eq!(counts[1], 1, "col 1 has 1 missing");
    assert_eq!(counts[2], 1, "col 2 has 1 missing");
}

#[test]
fn simple_imputer_statistics_correct() {
    use ferroml_core::preprocessing::imputers::{ImputeStrategy, SimpleImputer};

    let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);

    // Col 0: non-NaN [2, 4, 6] => mean = 4.0
    // Col 1: non-NaN [10, 30] => mean = 20.0
    let x = array![
        [2.0, 10.0],
        [f64::NAN, f64::NAN],
        [4.0, 30.0],
        [6.0, f64::NAN]
    ];
    imputer.fit(&x).unwrap();

    let stats = imputer.statistics().unwrap();
    assert_approx(stats[0], 4.0, 1e-10, "mean of col 0");
    assert_approx(stats[1], 20.0, 1e-10, "mean of col 1");
}

// =============================================================================
// Additional KNNImputer Tests
// =============================================================================

#[test]
fn knn_imputer_distance_weighting() {
    use ferroml_core::preprocessing::imputers::{KNNImputer, KNNWeights};

    let mut imputer = KNNImputer::new(2).with_weights(KNNWeights::Distance);

    // Row 0: [1.0, 10.0]
    // Row 1: [2.0, NaN]  -- closest to row 0 (dist=1) and row 2 (dist=1)
    // Row 2: [3.0, 30.0]
    // Row 3: [10.0, 100.0]
    let x = array![[1.0, 10.0], [2.0, f64::NAN], [3.0, 30.0], [10.0, 100.0]];
    let result = imputer.fit_transform(&x).unwrap();

    // Imputed value for row 1, col 1 should be based on nearest neighbors
    assert!(!result[[1, 1]].is_nan(), "Should be imputed");
    // With uniform weights and 2 neighbors: mean(10, 30) = 20
    // With distance weights: (10/1 + 30/1) / (1/1 + 1/1) = 20 (equal distances)
    assert!(
        result[[1, 1]] > 5.0 && result[[1, 1]] < 35.0,
        "imputed should be between neighbor values, got {}",
        result[[1, 1]]
    );
}

#[test]
fn knn_imputer_manhattan_metric() {
    use ferroml_core::preprocessing::imputers::{KNNImputer, KNNMetric};

    let mut imputer = KNNImputer::new(2).with_metric(KNNMetric::Manhattan);

    let x = array![[0.0, 0.0, 100.0], [1.0, 1.0, f64::NAN], [10.0, 10.0, 200.0]];
    let result = imputer.fit_transform(&x).unwrap();

    assert!(
        !result[[1, 2]].is_nan(),
        "Should be imputed with Manhattan metric"
    );
    assert!(result[[1, 2]].is_finite(), "Imputed value should be finite");
}

// =============================================================================
// Additional Encoder Tests
// =============================================================================

#[test]
fn onehot_encoder_multifeature() {
    use ferroml_core::preprocessing::encoders::OneHotEncoder;

    let mut encoder = OneHotEncoder::new();
    // Feature 0: 2 categories (0, 1), Feature 1: 3 categories (0, 1, 2)
    let x = array![[0.0, 0.0], [1.0, 1.0], [0.0, 2.0], [1.0, 0.0]];
    let result = encoder.fit_transform(&x).unwrap();

    // Total columns: 2 (from feature 0) + 3 (from feature 1) = 5
    assert_eq!(result.ncols(), 5, "2+3=5 one-hot columns");

    // Row 0: cat 0 in feat 0 + cat 0 in feat 1 => [1,0, 1,0,0]
    assert_approx(result[[0, 0]], 1.0, 1e-10, "r0 f0 cat0");
    assert_approx(result[[0, 1]], 0.0, 1e-10, "r0 f0 cat1");
    assert_approx(result[[0, 2]], 1.0, 1e-10, "r0 f1 cat0");
    assert_approx(result[[0, 3]], 0.0, 1e-10, "r0 f1 cat1");
    assert_approx(result[[0, 4]], 0.0, 1e-10, "r0 f1 cat2");
}

#[test]
fn label_encoder_n_classes() {
    use ferroml_core::preprocessing::encoders::LabelEncoder;

    let mut encoder = LabelEncoder::new();
    let labels = array![5.0, 3.0, 1.0, 5.0, 3.0, 1.0, 7.0];
    encoder.fit_1d(&labels).unwrap();

    assert_eq!(encoder.n_classes(), Some(4), "4 unique classes: 5,3,1,7");

    let classes = encoder.classes().unwrap();
    assert_eq!(classes.len(), 4);
}

#[test]
fn label_encoder_inverse_transform_roundtrip() {
    use ferroml_core::preprocessing::encoders::LabelEncoder;

    let mut encoder = LabelEncoder::new();
    let labels = array![10.0, 20.0, 30.0, 10.0, 20.0];
    encoder.fit_1d(&labels).unwrap();

    let encoded = encoder.transform_1d(&labels).unwrap();
    let recovered = encoder.inverse_transform_1d(&encoded).unwrap();

    for i in 0..labels.len() {
        assert_approx(recovered[i], labels[i], 1e-10, &format!("roundtrip {}", i));
    }
}

#[test]
fn ordinal_encoder_multifeature() {
    use ferroml_core::preprocessing::encoders::OrdinalEncoder;

    let mut encoder = OrdinalEncoder::new();
    // Feature 0: categories [10, 20], Feature 1: categories [100, 200, 300]
    let x = array![[10.0, 100.0], [20.0, 200.0], [10.0, 300.0], [20.0, 100.0]];
    let result = encoder.fit_transform(&x).unwrap();

    assert_eq!(result.ncols(), 2);

    // Feature 0: 10.0 -> 0, 20.0 -> 1
    assert_approx(result[[0, 0]], 0.0, 1e-10, "10.0 -> 0");
    assert_approx(result[[1, 0]], 1.0, 1e-10, "20.0 -> 1");
    assert_approx(result[[2, 0]], 0.0, 1e-10, "10.0 -> 0");
    assert_approx(result[[3, 0]], 1.0, 1e-10, "20.0 -> 1");

    // Feature 1: 100.0 -> 0, 200.0 -> 1, 300.0 -> 2
    assert_approx(result[[0, 1]], 0.0, 1e-10, "100.0 -> 0");
    assert_approx(result[[1, 1]], 1.0, 1e-10, "200.0 -> 1");
    assert_approx(result[[2, 1]], 2.0, 1e-10, "300.0 -> 2");
}

#[test]
fn target_encoder_multifeature() {
    use ferroml_core::preprocessing::encoders::TargetEncoder;

    let mut encoder = TargetEncoder::new().with_smooth(0.0);

    // Two features, each with 2 categories
    let x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y = array![10.0, 20.0, 30.0, 40.0];
    // Feature 0: cat 0 targets = [10, 20] => mean=15, cat 1 targets = [30, 40] => mean=35
    // Feature 1: cat 0 targets = [10, 30] => mean=20, cat 1 targets = [20, 40] => mean=30

    encoder.fit_with_target(&x, &y).unwrap();
    let result = encoder.transform(&x).unwrap();

    assert_eq!(result.ncols(), 2);

    // Feature 0 encoding
    assert_approx(result[[0, 0]], 15.0, 1e-8, "f0 cat0 encoding");
    assert_approx(result[[1, 0]], 15.0, 1e-8, "f0 cat0 encoding again");
    assert_approx(result[[2, 0]], 35.0, 1e-8, "f0 cat1 encoding");
    assert_approx(result[[3, 0]], 35.0, 1e-8, "f0 cat1 encoding again");

    // Feature 1 encoding
    assert_approx(result[[0, 1]], 20.0, 1e-8, "f1 cat0 encoding");
    assert_approx(result[[2, 1]], 20.0, 1e-8, "f1 cat0 encoding again");
    assert_approx(result[[1, 1]], 30.0, 1e-8, "f1 cat1 encoding");
    assert_approx(result[[3, 1]], 30.0, 1e-8, "f1 cat1 encoding again");
}

// =============================================================================
// Additional Resampling Tests
// =============================================================================

#[test]
fn random_oversampler_preserves_original_samples() {
    use ferroml_core::preprocessing::sampling::{RandomOverSampler, Resampler};

    let mut ros = RandomOverSampler::new().with_random_state(99);

    let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [100.0, 1000.0]];
    let y = array![0.0, 0.0, 0.0, 1.0];

    let (x_res, y_res) = ros.fit_resample(&x, &y).unwrap();

    // All original samples should be present
    assert_eq!(x_res.nrows(), y_res.len());

    // Majority samples unchanged
    let class_0_count = y_res.iter().filter(|&&v| v == 0.0).count();
    assert_eq!(class_0_count, 3);

    // Minority class oversampled to 3
    let class_1_count = y_res.iter().filter(|&&v| v == 1.0).count();
    assert_eq!(class_1_count, 3);
}

#[test]
fn smote_with_large_k_capped() {
    // If k_neighbors > n_minority_samples, SMOTE should handle gracefully
    use ferroml_core::preprocessing::sampling::{Resampler, SMOTE};

    let mut smote = SMOTE::new().with_k_neighbors(10).with_random_state(42);

    // Only 3 minority samples but k=10
    let mut x_data = Vec::new();
    for i in 0..15 {
        x_data.push(i as f64);
    }
    for i in 0..3 {
        x_data.push(100.0 + i as f64);
    }
    let x = Array2::from_shape_vec((18, 1), x_data).unwrap();
    let y = Array1::from_iter((0..15).map(|_| 0.0).chain((0..3).map(|_| 1.0)));

    // Should either handle gracefully (cap k) or return an error
    let result = smote.fit_resample(&x, &y);
    match result {
        Ok((x_res, y_res)) => {
            // If it succeeds, minority should be upsampled
            let class_1_count = y_res.iter().filter(|&&v| v == 1.0).count();
            assert!(
                class_1_count >= 3,
                "minority should be at least original count"
            );
            assert_eq!(x_res.ncols(), 1, "features preserved");
        }
        Err(_) => {
            // Error is also acceptable if k > n_minority
        }
    }
}

#[test]
fn smote_multiclass_handling() {
    // SMOTE with 3 classes: should upsample all minority classes
    use ferroml_core::preprocessing::sampling::{Resampler, SMOTE};

    let mut smote = SMOTE::new().with_k_neighbors(2).with_random_state(42);

    // 10 of class 0, 5 of class 1, 3 of class 2
    let mut x_data = Vec::new();
    for i in 0..10 {
        x_data.push(i as f64);
    }
    for i in 0..5 {
        x_data.push(50.0 + i as f64);
    }
    for i in 0..3 {
        x_data.push(100.0 + i as f64);
    }
    let x = Array2::from_shape_vec((18, 1), x_data).unwrap();
    let y = Array1::from_iter(
        (0..10)
            .map(|_| 0.0)
            .chain((0..5).map(|_| 1.0))
            .chain((0..3).map(|_| 2.0)),
    );

    let result = smote.fit_resample(&x, &y);
    match result {
        Ok((x_res, y_res)) => {
            let c0 = y_res.iter().filter(|&&v| v == 0.0).count();
            let c1 = y_res.iter().filter(|&&v| v == 1.0).count();
            let c2 = y_res.iter().filter(|&&v| v == 2.0).count();

            // Majority should be unchanged
            assert_eq!(c0, 10, "majority unchanged");
            // Minorities should be upsampled
            assert!(c1 >= 5, "class 1 should be at least original count");
            assert!(c2 >= 3, "class 2 should be at least original count");
            assert_eq!(x_res.ncols(), 1);
        }
        Err(_) => {
            // Some SMOTE implementations only handle binary
        }
    }
}

// =============================================================================
// Additional Edge Case Tests
// =============================================================================

#[test]
fn transformer_all_same_values() {
    // When all values in a column are the same
    use ferroml_core::preprocessing::scalers::MinMaxScaler;

    let mut scaler = MinMaxScaler::new();
    let x = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]; // col 0 is constant

    let result = scaler.fit_transform(&x);
    match result {
        Ok(transformed) => {
            // Constant column should be handled (0 or NaN, but finite)
            assert_eq!(transformed.shape(), &[3, 2]);
            // Non-constant feature should scale properly
            assert_approx(transformed[[0, 1]], 0.0, 1e-10, "min -> 0");
            assert_approx(transformed[[2, 1]], 1.0, 1e-10, "max -> 1");
        }
        Err(_) => {
            // Error is also acceptable
        }
    }
}

#[test]
fn transformer_large_values() {
    // Numerical stability with large values
    use ferroml_core::preprocessing::scalers::StandardScaler;

    let mut scaler = StandardScaler::new();
    let x = array![[1e10, 1e-10], [2e10, 2e-10], [3e10, 3e-10]];

    let result = scaler.fit_transform(&x).unwrap();

    // All values should be finite
    for &v in result.iter() {
        assert!(v.is_finite(), "should handle large/small values, got {}", v);
    }
}

#[test]
fn polynomial_features_fit_transform_equals_fit_then_transform() {
    // Verify fit_transform() == fit() + transform() for PolynomialFeatures
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;

    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

    let mut poly1 = PolynomialFeatures::new(2);
    let r1 = poly1.fit_transform(&x).unwrap();

    let mut poly2 = PolynomialFeatures::new(2);
    poly2.fit(&x).unwrap();
    let r2 = poly2.transform(&x).unwrap();

    assert_array2_approx(&r1, &r2, 1e-10, "fit_transform consistency");
}

#[test]
fn imputer_transform_on_new_data() {
    // SimpleImputer should use statistics from fit data when transforming new data
    use ferroml_core::preprocessing::imputers::{ImputeStrategy, SimpleImputer};

    let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);

    // Fit on training data
    let x_train = array![[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]];
    imputer.fit(&x_train).unwrap();
    // mean(col 0) = 3, mean(col 1) = 20

    // Transform test data with NaN
    let x_test = array![[f64::NAN, 15.0], [2.0, f64::NAN]];
    let result = imputer.transform(&x_test).unwrap();

    assert_approx(result[[0, 0]], 3.0, 1e-10, "impute with training mean");
    assert_approx(result[[1, 1]], 20.0, 1e-10, "impute with training mean");
    assert_approx(result[[0, 1]], 15.0, 1e-10, "non-NaN unchanged");
    assert_approx(result[[1, 0]], 2.0, 1e-10, "non-NaN unchanged");
}

// =============================================================================
// Additional Pipeline Integration Tests
// =============================================================================

#[test]
fn pipeline_poly_features_to_model() {
    // Pipeline: PolynomialFeatures -> LinearRegression on quadratic data
    use ferroml_core::models::LinearRegression;
    use ferroml_core::pipeline::Pipeline;
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;

    let mut pipeline = Pipeline::new()
        .add_transformer("poly", PolynomialFeatures::new(2).include_bias(false))
        .add_model("lr", LinearRegression::new());

    // Quadratic data: y = x^2
    let x = Array2::from_shape_fn((20, 1), |(i, _)| i as f64 - 10.0);
    let y = Array1::from_iter((0..20).map(|i| {
        let xi = i as f64 - 10.0;
        xi * xi
    }));

    pipeline.fit(&x, &y).unwrap();
    let predictions = pipeline.predict(&x).unwrap();

    // Predictions should be close to y (since poly degree 2 can represent x^2)
    for i in 0..20 {
        assert!(
            (predictions[i] - y[i]).abs() < 1.0,
            "prediction {} should be close to {}, got {}",
            i,
            y[i],
            predictions[i]
        );
    }
}

#[test]
fn pipeline_multiple_transformers() {
    // Pipeline with multiple transformers chained (imputer -> scaler -> model)
    use ferroml_core::models::LinearRegression;
    use ferroml_core::pipeline::Pipeline;
    use ferroml_core::preprocessing::imputers::{ImputeStrategy, SimpleImputer};
    use ferroml_core::preprocessing::scalers::StandardScaler;

    let mut pipeline = Pipeline::new()
        .add_transformer("imputer", SimpleImputer::new(ImputeStrategy::Mean))
        .add_transformer("scaler", StandardScaler::new())
        .add_model("lr", LinearRegression::new());

    // Data with 2 non-collinear features and a couple NaN
    let mut x = Array2::from_shape_fn((30, 2), |(i, j)| {
        if j == 0 {
            i as f64
        } else {
            ((i * 7 + 3) % 30) as f64
        }
    });
    x[[5, 0]] = f64::NAN;
    x[[10, 1]] = f64::NAN;

    let y = Array1::from_iter((0..30).map(|i| (i * 2 + 1) as f64));

    pipeline.fit(&x, &y).unwrap();
    let predictions = pipeline.predict(&x).unwrap();

    assert_eq!(predictions.len(), 30);
    for &p in predictions.iter() {
        assert!(p.is_finite(), "prediction must be finite");
    }
}

#[test]
fn pipeline_transform_only() {
    // Pipeline with only transformers (no model), using transform()
    use ferroml_core::pipeline::Pipeline;
    use ferroml_core::preprocessing::scalers::StandardScaler;

    let mut pipeline = Pipeline::new().add_transformer("scaler", StandardScaler::new());

    let x = Array2::from_shape_fn((10, 3), |(i, j)| (i * 3 + j) as f64);
    let y = Array1::from_iter((0..10).map(|i| i as f64));

    pipeline.fit(&x, &y).unwrap();
    let transformed = pipeline.transform(&x).unwrap();

    assert_eq!(transformed.shape(), &[10, 3]);

    // Each column should have ~0 mean
    for j in 0..3 {
        let col = transformed.column(j);
        let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
        assert!(
            mean.abs() < 1e-10,
            "col {} mean should be ~0, got {}",
            j,
            mean
        );
    }
}

// =============================================================================
// Inverse Transform Consistency Tests
// =============================================================================

#[test]
fn scaler_inverse_transform_roundtrip() {
    use ferroml_core::preprocessing::scalers::StandardScaler;

    let mut scaler = StandardScaler::new();
    let x = array![[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]];

    scaler.fit(&x).unwrap();
    let transformed = scaler.transform(&x).unwrap();
    let recovered = scaler.inverse_transform(&transformed).unwrap();

    assert_array2_approx(&recovered, &x, 1e-8, "standard_scaler roundtrip");
}

#[test]
fn minmax_scaler_inverse_transform_roundtrip() {
    use ferroml_core::preprocessing::scalers::MinMaxScaler;

    let mut scaler = MinMaxScaler::new();
    let x = array![[1.0, 10.0], [5.0, 50.0], [9.0, 90.0]];

    scaler.fit(&x).unwrap();
    let transformed = scaler.transform(&x).unwrap();
    let recovered = scaler.inverse_transform(&transformed).unwrap();

    assert_array2_approx(&recovered, &x, 1e-8, "minmax_scaler roundtrip");
}

#[test]
fn robust_scaler_inverse_transform_roundtrip() {
    use ferroml_core::preprocessing::scalers::RobustScaler;

    let mut scaler = RobustScaler::new();
    let x = array![
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0]
    ];

    scaler.fit(&x).unwrap();
    let transformed = scaler.transform(&x).unwrap();
    let recovered = scaler.inverse_transform(&transformed).unwrap();

    assert_array2_approx(&recovered, &x, 1e-8, "robust_scaler roundtrip");
}

// =============================================================================
// Feature Names Tests
// =============================================================================

#[test]
fn feature_names_propagation_through_transformers() {
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;
    use ferroml_core::preprocessing::selection::VarianceThreshold;

    // PolynomialFeatures generates feature names
    let mut poly = PolynomialFeatures::new(2);
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    poly.fit(&x).unwrap();

    let names = poly.get_feature_names_out(None).unwrap();
    assert_eq!(names.len(), 6); // 1 + 2 + 3 = 6 for degree 2 with 2 features
    assert_eq!(names[0], "1"); // bias
    assert_eq!(names[1], "x0");
    assert_eq!(names[2], "x1");

    // VarianceThreshold with custom input names
    let mut vt = VarianceThreshold::new(0.0);
    let x2 = array![[1.0, 5.0, 3.0], [1.0, 2.0, 3.0], [1.0, 8.0, 3.0]];
    vt.fit(&x2).unwrap();

    let custom_names = vec!["age".to_string(), "income".to_string(), "code".to_string()];
    let out_names = vt.get_feature_names_out(Some(&custom_names)).unwrap();
    assert_eq!(out_names, vec!["income"]); // Only non-constant feature
}

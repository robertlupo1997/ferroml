//! Statistical diagnostics for model validation

use crate::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Comprehensive residual diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualDiagnostics {
    /// Normality test result
    pub normality: NormalityTestResult,
    /// Homoscedasticity test result
    pub homoscedasticity: HomoscedasticityResult,
    /// Autocorrelation test result
    pub autocorrelation: AutocorrelationResult,
    /// Outlier detection
    pub outliers: OutlierResult,
}

/// Result of a normality test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalityTestResult {
    /// Test name
    pub test: String,
    /// Test statistic
    pub statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Whether normality assumption holds (p > 0.05)
    pub is_normal: bool,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis (excess)
    pub kurtosis: f64,
}

/// Result of homoscedasticity test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomoscedasticityResult {
    /// Test name
    pub test: String,
    /// Test statistic
    pub statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Whether homoscedasticity holds
    pub is_homoscedastic: bool,
}

/// Result of autocorrelation test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocorrelationResult {
    /// Test name
    pub test: String,
    /// Durbin-Watson statistic
    pub dw_statistic: f64,
    /// Interpretation
    pub interpretation: String,
    /// Lag-1 autocorrelation
    pub lag1_autocorr: f64,
}

/// Outlier detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierResult {
    /// Number of outliers detected
    pub n_outliers: usize,
    /// Indices of outliers
    pub outlier_indices: Vec<usize>,
    /// Method used
    pub method: String,
    /// Threshold used
    pub threshold: f64,
}

/// Normality test trait
pub trait NormalityTest {
    /// Run the normality test
    fn test(&self, data: &Array1<f64>) -> Result<NormalityTestResult>;
}

/// Shapiro-Wilk style normality test
pub struct ShapiroWilkTest;

impl NormalityTest for ShapiroWilkTest {
    fn test(&self, data: &Array1<f64>) -> Result<NormalityTestResult> {
        let n = data.len();
        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(1.0);

        // Compute skewness and kurtosis
        let skewness: f64 = data.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n as f64;

        let kurtosis: f64 =
            data.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n as f64 - 3.0;

        // Test statistic based on skewness/kurtosis (Jarque-Bera variant)
        let z_s = skewness.abs() / (6.0 / n as f64).sqrt();
        let z_k = kurtosis.abs() / (24.0 / n as f64).sqrt();
        let k2 = z_s * z_s + z_k * z_k;

        // Chi-squared survival function with df=2: p = exp(-k2/2)
        let p_value = (-k2 / 2.0).exp().min(1.0);

        // W statistic: transform k2 to [0,1] range
        let w = 1.0 / (1.0 + k2);

        Ok(NormalityTestResult {
            test: "Normality (skewness-kurtosis)".to_string(),
            statistic: w,
            p_value,
            is_normal: p_value > 0.05,
            skewness,
            kurtosis,
        })
    }
}

/// Compute Durbin-Watson statistic for autocorrelation
pub fn durbin_watson(residuals: &Array1<f64>) -> f64 {
    let n = residuals.len();
    if n < 2 {
        return f64::NAN;
    }

    let mut sum_sq_diff = 0.0;
    let mut sum_sq = 0.0;

    for i in 0..n {
        sum_sq += residuals[i].powi(2);
        if i > 0 {
            sum_sq_diff += (residuals[i] - residuals[i - 1]).powi(2);
        }
    }

    if sum_sq == 0.0 {
        return f64::NAN;
    }

    sum_sq_diff / sum_sq
}

/// Detect outliers using IQR method
pub fn detect_outliers_iqr(data: &Array1<f64>, k: f64) -> OutlierResult {
    let mut sorted: Vec<f64> = data.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    let iqr = q3 - q1;

    let lower_bound = k.mul_add(-iqr, q1);
    let upper_bound = k.mul_add(iqr, q3);

    let outlier_indices: Vec<usize> = data
        .iter()
        .enumerate()
        .filter(|(_, &v)| v < lower_bound || v > upper_bound)
        .map(|(i, _)| i)
        .collect();

    OutlierResult {
        n_outliers: outlier_indices.len(),
        outlier_indices,
        method: format!("IQR (k={})", k),
        threshold: k,
    }
}

/// Run comprehensive residual diagnostics
pub fn diagnose_residuals(residuals: &Array1<f64>) -> Result<ResidualDiagnostics> {
    let normality = ShapiroWilkTest.test(residuals)?;

    let dw = durbin_watson(residuals);
    let autocorrelation = AutocorrelationResult {
        test: "Durbin-Watson".to_string(),
        dw_statistic: dw,
        interpretation: if dw < 1.5 {
            "Positive autocorrelation".to_string()
        } else if dw > 2.5 {
            "Negative autocorrelation".to_string()
        } else {
            "No significant autocorrelation".to_string()
        },
        lag1_autocorr: 1.0 - dw / 2.0,
    };

    let outliers = detect_outliers_iqr(residuals, 1.5);

    // Simplified homoscedasticity (variance ratio test)
    let n = residuals.len();
    let mid = n / 2;
    let var1 = residuals.slice(ndarray::s![..mid]).var(1.0);
    let var2 = residuals.slice(ndarray::s![mid..]).var(1.0);
    let f_ratio = var1.max(var2) / var1.min(var2);

    // F-test p-value using regularized incomplete beta function
    let df1 = (mid as f64) - 1.0;
    let df2 = ((n - mid) as f64) - 1.0;
    let f_p_value = f_test_pvalue(f_ratio, df1, df2);

    let homoscedasticity = HomoscedasticityResult {
        test: "Goldfeld-Quandt (simplified)".to_string(),
        statistic: f_ratio,
        p_value: f_p_value,
        is_homoscedastic: f_p_value > 0.05,
    };

    Ok(ResidualDiagnostics {
        normality,
        homoscedasticity,
        autocorrelation,
        outliers,
    })
}

/// Compute p-value for F-test using regularized incomplete beta function.
/// P(F > f) = 1 - I_x(df1/2, df2/2) where x = df1*f / (df1*f + df2)
fn f_test_pvalue(f_stat: f64, df1: f64, df2: f64) -> f64 {
    if f_stat <= 0.0 || df1 <= 0.0 || df2 <= 0.0 {
        return 1.0;
    }
    let x = df1 * f_stat / (df1 * f_stat + df2);
    1.0 - regularized_incomplete_beta(df1 / 2.0, df2 / 2.0, x)
}

/// Regularized incomplete beta function using Lentz's continued fraction
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry relation if x > (a+1)/(a+b+2) for better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x);
    }

    let ln_prefix = a * x.ln() + b * (1.0 - x).ln() - a.ln() - ln_beta(a, b);
    let prefix = ln_prefix.exp();

    // Lentz's continued fraction
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut f = d;

    for m in 1..200 {
        let m = m as f64;

        // Even step
        let num = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        f *= c * d;

        // Odd step
        let num = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    prefix * f
}

/// Log of Beta function using Stirling's approximation via ln_gamma
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos approximation of ln(Gamma(x))
fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.001_208_650_973_866_179,
        -0.000_005_395_239_384_953,
    ];

    let tmp = x + 5.5;
    let tmp = (x + 0.5) * tmp.ln() - tmp;
    let mut ser = 1.000_000_000_190_015;
    for (i, &c) in coeffs.iter().enumerate() {
        ser += c / (x + 1.0 + i as f64);
    }
    tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    // ---- Durbin-Watson ----

    #[test]
    fn test_durbin_watson_no_autocorrelation() {
        // Alternating residuals should give DW near 2 (no autocorrelation)
        let residuals = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let dw = durbin_watson(&residuals);
        // DW for alternating = sum of (2)^2 * 7 / (1^2 * 8) = 4*7/8 = 3.5
        // Actually: each diff is 2, so sum_sq_diff = 7*4 = 28, sum_sq = 8*1 = 8, DW = 3.5
        assert!(dw > 2.0, "alternating should give DW > 2: {}", dw);
    }

    #[test]
    fn test_durbin_watson_perfect_positive_autocorrelation() {
        // Constant residuals: DW = 0
        let residuals = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        let dw = durbin_watson(&residuals);
        assert_relative_eq!(dw, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_durbin_watson_range() {
        // DW should be between 0 and 4
        let residuals = Array1::from_vec(vec![0.5, -0.3, 0.7, -0.1, 0.4, -0.6]);
        let dw = durbin_watson(&residuals);
        assert!(dw >= 0.0 && dw <= 4.0, "DW should be in [0,4]: {}", dw);
    }

    #[test]
    fn test_durbin_watson_single_element() {
        let residuals = Array1::from_vec(vec![1.0]);
        let dw = durbin_watson(&residuals);
        assert!(dw.is_nan(), "DW with 1 element should be NaN");
    }

    #[test]
    fn test_durbin_watson_two_elements() {
        let residuals = Array1::from_vec(vec![1.0, 2.0]);
        let dw = durbin_watson(&residuals);
        // sum_sq_diff = (2-1)^2 = 1, sum_sq = 1+4 = 5, DW = 1/5 = 0.2
        assert_relative_eq!(dw, 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_durbin_watson_all_zeros() {
        let residuals = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let dw = durbin_watson(&residuals);
        assert!(dw.is_nan(), "DW with all zeros should be NaN");
    }

    // ---- detect_outliers_iqr ----

    #[test]
    fn test_outliers_iqr_no_outliers() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = detect_outliers_iqr(&data, 1.5);
        assert_eq!(result.n_outliers, 0);
        assert!(result.outlier_indices.is_empty());
        assert_eq!(result.method, "IQR (k=1.5)");
        assert_relative_eq!(result.threshold, 1.5);
    }

    #[test]
    fn test_outliers_iqr_with_outliers() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]);
        let result = detect_outliers_iqr(&data, 1.5);
        assert!(result.n_outliers >= 1, "should detect outlier at 100");
        assert!(
            result.outlier_indices.contains(&8),
            "index 8 (value 100) should be outlier"
        );
    }

    #[test]
    fn test_outliers_iqr_strict_threshold() {
        // k=0 means everything outside Q1-Q3 is an outlier
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = detect_outliers_iqr(&data, 0.0);
        // With k=0, bounds are exactly Q1 and Q3
        // Points outside [Q1, Q3] are outliers
        assert!(
            result.n_outliers > 0,
            "strict threshold should find outliers"
        );
    }

    #[test]
    fn test_outliers_iqr_negative_outliers() {
        let data = Array1::from_vec(vec![-100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = detect_outliers_iqr(&data, 1.5);
        assert!(result.n_outliers >= 1);
        assert!(result.outlier_indices.contains(&0));
    }

    // ---- ShapiroWilkTest (NormalityTest) ----

    #[test]
    fn test_normality_test_normal_data() {
        // Approximate normal data (symmetric, no heavy tails)
        let data = Array1::from_vec(vec![
            -1.5, -1.0, -0.5, -0.2, 0.0, 0.1, 0.3, 0.5, 1.0, 1.5, -1.2, -0.8, -0.3, 0.2, 0.4, 0.7,
            0.9, -0.6, -0.1, 0.6,
        ]);
        let result = ShapiroWilkTest.test(&data).unwrap();
        assert_eq!(result.test, "Normality (skewness-kurtosis)");
        // Roughly symmetric data should pass normality
        assert!(
            result.is_normal,
            "near-normal data should pass: p={}, skew={}, kurt={}",
            result.p_value, result.skewness, result.kurtosis
        );
    }

    #[test]
    fn test_normality_test_skewed_data() {
        // Highly skewed data
        let data = Array1::from_vec(vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 100.0,
        ]);
        let result = ShapiroWilkTest.test(&data).unwrap();
        assert!(
            result.skewness.abs() > 1.0,
            "highly skewed data should have |skew| > 1: {}",
            result.skewness
        );
    }

    #[test]
    fn test_normality_test_returns_valid_statistics() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let result = ShapiroWilkTest.test(&data).unwrap();
        assert!(
            result.p_value >= 0.0 && result.p_value <= 1.0,
            "p_value in [0,1]"
        );
        assert!(
            result.statistic >= 0.0 && result.statistic <= 1.0,
            "W statistic in [0,1]: {}",
            result.statistic
        );
        assert!(result.skewness.is_finite());
        assert!(result.kurtosis.is_finite());
    }

    // ---- diagnose_residuals ----

    #[test]
    fn test_diagnose_residuals_basic() {
        let residuals =
            Array1::from_vec(vec![0.5, -0.3, 0.7, -0.1, 0.4, -0.6, 0.2, -0.4, 0.3, -0.5]);
        let diag = diagnose_residuals(&residuals).unwrap();

        // Check all fields are populated
        assert_eq!(diag.normality.test, "Normality (skewness-kurtosis)");
        assert_eq!(diag.autocorrelation.test, "Durbin-Watson");
        assert_eq!(diag.homoscedasticity.test, "Goldfeld-Quandt (simplified)");
        assert!(diag.outliers.method.contains("IQR"));

        // DW should be in valid range
        assert!(
            diag.autocorrelation.dw_statistic >= 0.0 && diag.autocorrelation.dw_statistic <= 4.0
        );

        // lag1 autocorrelation is derived from DW
        assert_relative_eq!(
            diag.autocorrelation.lag1_autocorr,
            1.0 - diag.autocorrelation.dw_statistic / 2.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_diagnose_residuals_no_autocorrelation_interpretation() {
        // Alternating residuals
        let residuals =
            Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let diag = diagnose_residuals(&residuals).unwrap();
        // DW > 2.5 for alternating => "Negative autocorrelation"
        assert!(
            diag.autocorrelation.dw_statistic > 2.5,
            "DW for alternating: {}",
            diag.autocorrelation.dw_statistic
        );
        assert_eq!(
            diag.autocorrelation.interpretation,
            "Negative autocorrelation"
        );
    }

    #[test]
    fn test_diagnose_residuals_positive_autocorrelation() {
        // Slowly trending residuals (positive autocorrelation)
        let residuals = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);
        let diag = diagnose_residuals(&residuals).unwrap();
        // DW should be small for positive autocorrelation
        assert!(
            diag.autocorrelation.dw_statistic < 1.5,
            "DW for trending: {}",
            diag.autocorrelation.dw_statistic
        );
        assert_eq!(
            diag.autocorrelation.interpretation,
            "Positive autocorrelation"
        );
    }

    #[test]
    fn test_diagnose_residuals_homoscedasticity() {
        // Equal variance in both halves
        let residuals =
            Array1::from_vec(vec![1.0, -1.0, 0.5, -0.5, 0.8, -0.8, 0.3, -0.3, 0.7, -0.7]);
        let diag = diagnose_residuals(&residuals).unwrap();
        assert!(diag.homoscedasticity.statistic >= 1.0, "F-ratio >= 1");
        assert!(
            diag.homoscedasticity.p_value >= 0.0 && diag.homoscedasticity.p_value <= 1.0,
            "p_value in [0,1]: {}",
            diag.homoscedasticity.p_value
        );
    }

    #[test]
    fn test_diagnose_residuals_heteroscedastic() {
        // First half small residuals, second half large
        let residuals = Array1::from_vec(vec![
            0.01, -0.01, 0.02, -0.02, 0.01, 10.0, -10.0, 8.0, -8.0, 9.0,
        ]);
        let diag = diagnose_residuals(&residuals).unwrap();
        // F-ratio should be large
        assert!(
            diag.homoscedasticity.statistic > 5.0,
            "F-ratio should be large for heteroscedastic data: {}",
            diag.homoscedasticity.statistic
        );
    }

    // ---- f_test_pvalue ----

    #[test]
    fn test_f_test_pvalue_equal_variances() {
        // F=1 should give p-value close to 0.5 (equal variances)
        let p = f_test_pvalue(1.0, 10.0, 10.0);
        assert!(p > 0.4, "F=1 should give large p-value: {}", p);
    }

    #[test]
    fn test_f_test_pvalue_large_f() {
        // Very large F should give small p-value
        let p = f_test_pvalue(100.0, 10.0, 10.0);
        assert!(p < 0.01, "large F should give small p-value: {}", p);
    }

    #[test]
    fn test_f_test_pvalue_invalid_inputs() {
        assert_relative_eq!(f_test_pvalue(0.0, 10.0, 10.0), 1.0);
        assert_relative_eq!(f_test_pvalue(-1.0, 10.0, 10.0), 1.0);
        assert_relative_eq!(f_test_pvalue(1.0, 0.0, 10.0), 1.0);
        assert_relative_eq!(f_test_pvalue(1.0, 10.0, 0.0), 1.0);
    }

    #[test]
    fn test_f_test_pvalue_range() {
        let p = f_test_pvalue(2.5, 20.0, 20.0);
        assert!(p >= 0.0 && p <= 1.0, "p-value in [0,1]: {}", p);
    }

    // ---- regularized_incomplete_beta ----

    #[test]
    fn test_regularized_incomplete_beta_boundaries() {
        assert_relative_eq!(
            regularized_incomplete_beta(2.0, 3.0, 0.0),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            regularized_incomplete_beta(2.0, 3.0, 1.0),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_regularized_incomplete_beta_symmetry() {
        // I_x(a,b) = 0.5 at the median of Beta(a,b)
        // For symmetric Beta(a,a), I_0.5(a,a) = 0.5
        assert_relative_eq!(
            regularized_incomplete_beta(2.0, 2.0, 0.5),
            0.5,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            regularized_incomplete_beta(5.0, 5.0, 0.5),
            0.5,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_regularized_incomplete_beta_known_values() {
        // I_x(1,1) = x (uniform distribution)
        assert_relative_eq!(
            regularized_incomplete_beta(1.0, 1.0, 0.3),
            0.3,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            regularized_incomplete_beta(1.0, 1.0, 0.7),
            0.7,
            epsilon = 1e-6
        );
    }

    // ---- ln_beta / ln_gamma ----

    #[test]
    fn test_ln_gamma_known_values() {
        // Gamma(1) = 1, so ln(Gamma(1)) = 0
        assert_relative_eq!(ln_gamma(1.0), 0.0, epsilon = 1e-6);

        // Gamma(2) = 1, so ln(Gamma(2)) = 0
        assert_relative_eq!(ln_gamma(2.0), 0.0, epsilon = 1e-6);

        // Gamma(5) = 24, so ln(Gamma(5)) = ln(24)
        assert_relative_eq!(ln_gamma(5.0), 24.0_f64.ln(), epsilon = 1e-6);

        // Gamma(0.5) = sqrt(pi)
        assert_relative_eq!(
            ln_gamma(0.5),
            std::f64::consts::PI.sqrt().ln(),
            epsilon = 1e-4
        );
    }

    #[test]
    fn test_ln_beta_known_values() {
        // Beta(1,1) = 1, ln(Beta(1,1)) = 0
        assert_relative_eq!(ln_beta(1.0, 1.0), 0.0, epsilon = 1e-6);

        // Beta(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
        // Beta(2,3) = 1!*2!/4! = 1*2/24 = 1/12
        assert_relative_eq!(ln_beta(2.0, 3.0), (1.0_f64 / 12.0).ln(), epsilon = 1e-6);
    }

    // ---- NormalityTest trait ----

    #[test]
    fn test_normality_test_trait_object() {
        // Verify the trait can be used as a trait object
        let test: &dyn NormalityTest = &ShapiroWilkTest;
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = test.test(&data).unwrap();
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    // ---- Struct serialization (serde) ----

    #[test]
    fn test_outlier_result_debug_clone() {
        let r = OutlierResult {
            n_outliers: 2,
            outlier_indices: vec![0, 8],
            method: "IQR (k=1.5)".to_string(),
            threshold: 1.5,
        };
        let r2 = r.clone();
        assert_eq!(r2.n_outliers, 2);
        assert_eq!(r2.outlier_indices, vec![0, 8]);
        let _ = format!("{:?}", r);
    }

    #[test]
    fn test_autocorrelation_result_clone() {
        let r = AutocorrelationResult {
            test: "DW".to_string(),
            dw_statistic: 2.0,
            interpretation: "No autocorrelation".to_string(),
            lag1_autocorr: 0.0,
        };
        let r2 = r.clone();
        assert_relative_eq!(r2.dw_statistic, 2.0);
    }

    #[test]
    fn test_homoscedasticity_result_clone() {
        let r = HomoscedasticityResult {
            test: "GQ".to_string(),
            statistic: 1.5,
            p_value: 0.3,
            is_homoscedastic: true,
        };
        let r2 = r.clone();
        assert!(r2.is_homoscedastic);
    }
}

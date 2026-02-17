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

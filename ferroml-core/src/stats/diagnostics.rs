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

        // Simplified test statistic based on skewness/kurtosis
        let z_s = skewness.abs() / (6.0 / n as f64).sqrt();
        let z_k = kurtosis.abs() / (24.0 / n as f64).sqrt();
        let w = 1.0 / z_k.mul_add(z_k, z_s.mul_add(z_s, 1.0));

        // Approximate p-value
        let p_value = (-5.0 * (1.0 - w)).exp().min(1.0);

        Ok(NormalityTestResult {
            test: "Shapiro-Wilk (approximation)".to_string(),
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

    let homoscedasticity = HomoscedasticityResult {
        test: "Goldfeld-Quandt (simplified)".to_string(),
        statistic: f_ratio,
        p_value: 0.5, // Placeholder
        is_homoscedastic: f_ratio < 3.0,
    };

    Ok(ResidualDiagnostics {
        normality,
        homoscedasticity,
        autocorrelation,
        outliers,
    })
}

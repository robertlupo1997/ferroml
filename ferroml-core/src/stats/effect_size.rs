//! Effect size measures for practical significance

use crate::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Trait for effect size calculations
pub trait EffectSize {
    /// Compute the effect size
    fn compute(&self) -> Result<EffectSizeValue>;

    /// Get the name of the effect size measure
    fn name(&self) -> &str;

    /// Interpret the effect size magnitude
    fn interpret(value: f64) -> &'static str;
}

/// Effect size value with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSizeValue {
    /// The effect size value
    pub value: f64,
    /// Confidence interval (if available)
    pub ci: Option<(f64, f64)>,
    /// Variance of the effect size
    pub variance: Option<f64>,
    /// Interpretation
    pub interpretation: String,
}

/// Cohen's d for two independent groups
pub struct CohensD {
    group1: Array1<f64>,
    group2: Array1<f64>,
}

impl CohensD {
    /// Create new Cohen's d calculator
    pub fn new(group1: Array1<f64>, group2: Array1<f64>) -> Self {
        Self { group1, group2 }
    }
}

impl EffectSize for CohensD {
    fn compute(&self) -> Result<EffectSizeValue> {
        let n1 = self.group1.len();
        let n2 = self.group2.len();

        let mean1 = self.group1.mean().unwrap_or(0.0);
        let mean2 = self.group2.mean().unwrap_or(0.0);

        let var1 = self.group1.var(1.0);
        let var2 = self.group2.var(1.0);

        // Pooled standard deviation
        let pooled_var = ((n1 - 1) as f64 * var1 + (n2 - 1) as f64 * var2) / (n1 + n2 - 2) as f64;
        let pooled_std = pooled_var.sqrt();

        let d = (mean1 - mean2) / pooled_std;

        // Variance of d
        let var_d = (n1 + n2) as f64 / (n1 * n2) as f64 + d.powi(2) / (2.0 * (n1 + n2) as f64);

        // 95% CI
        let se = var_d.sqrt();
        let ci = (d - 1.96 * se, d + 1.96 * se);

        Ok(EffectSizeValue {
            value: d,
            ci: Some(ci),
            variance: Some(var_d),
            interpretation: Self::interpret(d).to_string(),
        })
    }

    fn name(&self) -> &str {
        "Cohen's d"
    }

    fn interpret(value: f64) -> &'static str {
        let abs_val = value.abs();
        if abs_val < 0.2 {
            "negligible"
        } else if abs_val < 0.5 {
            "small"
        } else if abs_val < 0.8 {
            "medium"
        } else if abs_val < 1.2 {
            "large"
        } else {
            "very large"
        }
    }
}

/// Hedges' g (bias-corrected Cohen's d)
pub struct HedgesG {
    group1: Array1<f64>,
    group2: Array1<f64>,
}

impl HedgesG {
    /// Create new Hedges' g calculator
    pub fn new(group1: Array1<f64>, group2: Array1<f64>) -> Self {
        Self { group1, group2 }
    }
}

impl EffectSize for HedgesG {
    fn compute(&self) -> Result<EffectSizeValue> {
        // First compute Cohen's d
        let cohens_d = CohensD::new(self.group1.clone(), self.group2.clone());
        let d_result = cohens_d.compute()?;
        let d = d_result.value;

        let n1 = self.group1.len();
        let n2 = self.group2.len();
        let df = (n1 + n2 - 2) as f64;

        // Correction factor (approximation)
        let j = 1.0 - 3.0 / (4.0 * df - 1.0);
        let g = d * j;

        // Variance of g
        let var_g = j.powi(2) * d_result.variance.unwrap_or(0.0);
        let se = var_g.sqrt();
        let ci = (g - 1.96 * se, g + 1.96 * se);

        Ok(EffectSizeValue {
            value: g,
            ci: Some(ci),
            variance: Some(var_g),
            interpretation: Self::interpret(g).to_string(),
        })
    }

    fn name(&self) -> &str {
        "Hedges' g"
    }

    fn interpret(value: f64) -> &'static str {
        CohensD::interpret(value) // Same thresholds as Cohen's d
    }
}

/// Glass's delta (uses control group SD only)
pub struct GlasssDelta {
    treatment: Array1<f64>,
    control: Array1<f64>,
}

impl GlasssDelta {
    /// Create new Glass's delta calculator
    /// Treatment is compared against control group
    pub fn new(treatment: Array1<f64>, control: Array1<f64>) -> Self {
        Self { treatment, control }
    }
}

impl EffectSize for GlasssDelta {
    fn compute(&self) -> Result<EffectSizeValue> {
        let mean_treatment = self.treatment.mean().unwrap_or(0.0);
        let mean_control = self.control.mean().unwrap_or(0.0);
        let std_control = self.control.std(1.0);

        let delta = (mean_treatment - mean_control) / std_control;

        let n1 = self.treatment.len();
        let n2 = self.control.len();

        // Variance approximation
        let var_delta =
            (n1 + n2) as f64 / (n1 * n2) as f64 + delta.powi(2) / (2.0 * (n2 - 1) as f64);
        let se = var_delta.sqrt();
        let ci = (delta - 1.96 * se, delta + 1.96 * se);

        Ok(EffectSizeValue {
            value: delta,
            ci: Some(ci),
            variance: Some(var_delta),
            interpretation: Self::interpret(delta).to_string(),
        })
    }

    fn name(&self) -> &str {
        "Glass's Δ"
    }

    fn interpret(value: f64) -> &'static str {
        CohensD::interpret(value)
    }
}

/// Compute R-squared (coefficient of determination)
pub fn r_squared(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let mean = y_true.mean().unwrap_or(0.0);

    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();

    let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();

    if ss_tot == 0.0 {
        return f64::NAN;
    }

    1.0 - ss_res / ss_tot
}

/// Compute adjusted R-squared
pub fn adjusted_r_squared(r2: f64, n: usize, p: usize) -> f64 {
    let n = n as f64;
    let p = p as f64;
    1.0 - (1.0 - r2) * (n - 1.0) / (n - p - 1.0)
}

/// Compute eta-squared (ANOVA effect size)
pub fn eta_squared(ss_between: f64, ss_total: f64) -> f64 {
    ss_between / ss_total
}

/// Compute partial eta-squared
pub fn partial_eta_squared(ss_effect: f64, ss_error: f64) -> f64 {
    ss_effect / (ss_effect + ss_error)
}

/// Compute omega-squared (less biased than eta-squared)
pub fn omega_squared(ss_between: f64, ss_total: f64, ms_within: f64, k: usize) -> f64 {
    let k = k as f64;
    (ss_between - (k - 1.0) * ms_within) / (ss_total + ms_within)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cohens_d() {
        let g1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let g2 = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0, 9.0]);

        let d = CohensD::new(g1, g2);
        let result = d.compute().unwrap();

        assert!(result.value < 0.0); // g1 < g2
        assert!(result.ci.is_some());
    }

    #[test]
    fn test_r_squared() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let r2 = r_squared(&y_true, &y_pred);
        assert!((r2 - 1.0).abs() < 1e-10);
    }
}

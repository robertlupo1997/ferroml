//! Multiple testing correction methods
//!
//! When performing multiple statistical tests, the probability of false positives
//! increases. This module provides methods to control for this.

use serde::{Deserialize, Serialize};

/// Multiple testing correction method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultipleTestingCorrection {
    /// No correction
    None,
    /// Bonferroni correction (controls FWER)
    Bonferroni,
    /// Holm-Bonferroni step-down (controls FWER, more powerful than Bonferroni)
    Holm,
    /// Hochberg step-up (controls FWER, assumes independence)
    Hochberg,
    /// Benjamini-Hochberg (controls FDR)
    BenjaminiHochberg,
    /// Benjamini-Yekutieli (controls FDR under dependency)
    BenjaminiYekutieli,
}

/// Result of multiple testing correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectedPValues {
    /// Original p-values
    pub original: Vec<f64>,
    /// Adjusted p-values
    pub adjusted: Vec<f64>,
    /// Which hypotheses are rejected at given alpha
    pub rejected: Vec<bool>,
    /// Alpha level used
    pub alpha: f64,
    /// Method used
    pub method: MultipleTestingCorrection,
}

/// Adjust p-values for multiple testing
pub fn adjust_pvalues(
    p_values: &[f64],
    method: MultipleTestingCorrection,
    alpha: f64,
) -> CorrectedPValues {
    let adjusted = match method {
        MultipleTestingCorrection::None => p_values.to_vec(),
        MultipleTestingCorrection::Bonferroni => bonferroni(p_values),
        MultipleTestingCorrection::Holm => holm(p_values),
        MultipleTestingCorrection::Hochberg => hochberg(p_values),
        MultipleTestingCorrection::BenjaminiHochberg => benjamini_hochberg(p_values),
        MultipleTestingCorrection::BenjaminiYekutieli => benjamini_yekutieli(p_values),
    };

    let rejected: Vec<bool> = adjusted.iter().map(|&p| p < alpha).collect();

    CorrectedPValues {
        original: p_values.to_vec(),
        adjusted,
        rejected,
        alpha,
        method,
    }
}

/// Bonferroni correction
/// Adjusted p_i = min(n * p_i, 1)
fn bonferroni(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len() as f64;
    p_values.iter().map(|&p| (p * n).min(1.0)).collect()
}

/// Holm-Bonferroni step-down procedure
fn holm(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len();

    // Get sorted indices
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| p_values[a].partial_cmp(&p_values[b]).unwrap());

    let mut adjusted = vec![0.0; n];
    let mut running_max: f64 = 0.0;

    for (rank, &idx) in indices.iter().enumerate() {
        let multiplier = (n - rank) as f64;
        let adj = (p_values[idx] * multiplier).min(1.0);
        running_max = running_max.max(adj);
        adjusted[idx] = running_max;
    }

    adjusted
}

/// Hochberg step-up procedure
fn hochberg(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        p_values[b].partial_cmp(&p_values[a]).unwrap() // Descending
    });

    let mut adjusted = vec![0.0; n];
    let mut running_min: f64 = 1.0;

    for (rank, &idx) in indices.iter().enumerate() {
        let multiplier = (rank + 1) as f64;
        let adj = (p_values[idx] * multiplier).min(1.0);
        running_min = running_min.min(adj);
        adjusted[idx] = running_min;
    }

    adjusted
}

/// Benjamini-Hochberg procedure (FDR control)
fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| p_values[a].partial_cmp(&p_values[b]).unwrap());

    let mut adjusted = vec![0.0; n];
    let mut running_min: f64 = 1.0;

    // Process from largest to smallest
    for (i, &idx) in indices.iter().enumerate().rev() {
        let rank = i + 1;
        let adj = (p_values[idx] * n as f64 / rank as f64).min(1.0);
        running_min = running_min.min(adj);
        adjusted[idx] = running_min;
    }

    adjusted
}

/// Benjamini-Yekutieli procedure (FDR under dependency)
fn benjamini_yekutieli(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len();

    // Harmonic sum: sum(1/i) for i = 1 to n
    let c_n: f64 = (1..=n).map(|i| 1.0 / i as f64).sum();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| p_values[a].partial_cmp(&p_values[b]).unwrap());

    let mut adjusted = vec![0.0; n];
    let mut running_min: f64 = 1.0;

    for (i, &idx) in indices.iter().enumerate().rev() {
        let rank = i + 1;
        let adj = (p_values[idx] * n as f64 * c_n / rank as f64).min(1.0);
        running_min = running_min.min(adj);
        adjusted[idx] = running_min;
    }

    adjusted
}

/// Compute the number of rejected hypotheses at given alpha
pub fn count_rejections(adjusted: &[f64], alpha: f64) -> usize {
    adjusted.iter().filter(|&&p| p < alpha).count()
}

/// Compute false discovery proportion estimate
pub fn estimate_fdp(adjusted: &[f64], alpha: f64) -> f64 {
    let n_rejected = count_rejections(adjusted, alpha);
    if n_rejected == 0 {
        return 0.0;
    }

    // FDP estimate under BH procedure equals alpha * R / m where R is rejections
    alpha * n_rejected as f64 / adjusted.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bonferroni() {
        let p_values = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let adjusted = bonferroni(&p_values);

        assert!((adjusted[0] - 0.05).abs() < 1e-10);
        assert!((adjusted[1] - 0.10).abs() < 1e-10);
    }

    #[test]
    fn test_benjamini_hochberg() {
        let p_values = vec![0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.07, 0.08];
        let result = adjust_pvalues(
            &p_values,
            MultipleTestingCorrection::BenjaminiHochberg,
            0.05,
        );

        // First few should be rejected
        assert!(result.rejected[0]);
        assert!(result.rejected[1]);
    }

    #[test]
    fn test_holm() {
        let p_values = vec![0.01, 0.04, 0.03, 0.02];
        let adjusted = holm(&p_values);

        // All adjusted should be >= original
        for (adj, orig) in adjusted.iter().zip(p_values.iter()) {
            assert!(*adj >= *orig);
        }
    }
}

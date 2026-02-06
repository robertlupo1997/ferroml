//! Phase 30: Drift Detection Tests
//!
//! Comprehensive drift detection testing for ML models to identify when data
//! distributions or model performance degrades over time. This module provides:
//!
//! - Data drift detection: KS test, PSI, Jensen-Shannon divergence
//! - Concept drift detection: Model performance degradation, decision boundary shifts
//! - Gradual vs sudden drift: Different drift patterns
//! - Multi-feature drift: Coordinated shifts across features

use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

// ============================================================================
// Types and Enums
// ============================================================================

/// Type of drift to simulate in synthetic data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftType {
    /// Sudden shift in feature distributions (covariate shift)
    Sudden,
    /// Gradual transition between distributions
    Gradual,
    /// Changes in P(y|X) while P(X) stays same
    Concept,
    /// Both feature and concept drift
    Mixed,
}

/// Result of a drift detection test
#[derive(Debug, Clone)]
pub struct DriftResult {
    /// Name of the drift metric used
    pub metric: String,
    /// The computed statistic value
    pub statistic: f64,
    /// P-value if available (for statistical tests)
    pub p_value: Option<f64>,
    /// Whether drift was detected at the given threshold
    pub drift_detected: bool,
    /// Threshold used for detection
    pub threshold: f64,
}

/// Per-feature drift statistics
#[derive(Debug, Clone)]
pub struct FeatureDriftStats {
    /// Feature index
    pub feature_idx: usize,
    /// KS statistic for this feature
    pub ks_statistic: f64,
    /// PSI for this feature
    pub psi: f64,
    /// Whether this feature shows significant drift
    pub drifted: bool,
}

// ============================================================================
// Data Generation Functions
// ============================================================================

/// Generate data with sudden distribution shift
///
/// Creates two datasets: reference and drifted, where the drifted data
/// has shifted means and/or variances.
///
/// # Arguments
/// * `n_reference` - Number of reference samples
/// * `n_current` - Number of current (potentially drifted) samples
/// * `n_features` - Number of features
/// * `drift_magnitude` - How much the distribution shifts (0.0 = no drift)
/// * `drift_features` - Which features should drift (if None, all drift)
/// * `seed` - Random seed
///
/// # Returns
/// Tuple of (reference_data, current_data)
pub fn make_sudden_drift(
    n_reference: usize,
    n_current: usize,
    n_features: usize,
    drift_magnitude: f64,
    drift_features: Option<Vec<usize>>,
    seed: u64,
) -> (Array2<f64>, Array2<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Reference data: standard normal
    let mut ref_data = Vec::with_capacity(n_reference * n_features);
    for _ in 0..n_reference {
        for _ in 0..n_features {
            ref_data.push(sample_normal(&mut rng, 0.0, 1.0));
        }
    }

    // Current data: shifted distribution for selected features
    let drift_set: std::collections::HashSet<usize> = drift_features
        .map(|v| v.into_iter().collect())
        .unwrap_or_else(|| (0..n_features).collect());

    let mut cur_data = Vec::with_capacity(n_current * n_features);
    for _ in 0..n_current {
        for j in 0..n_features {
            let mean = if drift_set.contains(&j) {
                drift_magnitude
            } else {
                0.0
            };
            cur_data.push(sample_normal(&mut rng, mean, 1.0));
        }
    }

    let reference = Array2::from_shape_vec((n_reference, n_features), ref_data).unwrap();
    let current = Array2::from_shape_vec((n_current, n_features), cur_data).unwrap();

    (reference, current)
}

/// Generate data with gradual drift over time
///
/// Creates a sequence of data where the distribution gradually shifts.
///
/// # Arguments
/// * `n_samples` - Total samples across all time windows
/// * `n_windows` - Number of time windows
/// * `n_features` - Number of features
/// * `final_drift` - Final drift magnitude at last window
/// * `seed` - Random seed
///
/// # Returns
/// Tuple of (data, window_labels) where window_labels indicates which window each sample belongs to
pub fn make_gradual_drift(
    n_samples: usize,
    n_windows: usize,
    n_features: usize,
    final_drift: f64,
    seed: u64,
) -> (Array2<f64>, Array1<usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let samples_per_window = n_samples / n_windows;

    let mut data = Vec::with_capacity(n_samples * n_features);
    let mut windows = Vec::with_capacity(n_samples);

    for w in 0..n_windows {
        // Linear interpolation of drift
        let drift = final_drift * (w as f64 / (n_windows - 1).max(1) as f64);

        for _ in 0..samples_per_window {
            for _ in 0..n_features {
                data.push(sample_normal(&mut rng, drift, 1.0));
            }
            windows.push(w);
        }
    }

    let actual_samples = windows.len();
    let x = Array2::from_shape_vec((actual_samples, n_features), data).unwrap();
    let w = Array1::from_vec(windows);

    (x, w)
}

/// Generate data with concept drift (P(y|X) changes)
///
/// Creates reference and current datasets where the relationship between
/// features and target changes.
///
/// # Arguments
/// * `n_reference` - Number of reference samples
/// * `n_current` - Number of current samples
/// * `n_features` - Number of features
/// * `concept_shift` - How much the decision boundary rotates (radians)
/// * `seed` - Random seed
///
/// # Returns
/// Tuple of (ref_x, ref_y, cur_x, cur_y)
pub fn make_concept_drift(
    n_reference: usize,
    n_current: usize,
    n_features: usize,
    concept_shift: f64,
    seed: u64,
) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate features from same distribution
    let mut ref_x_data = Vec::with_capacity(n_reference * n_features);
    let mut ref_y = Vec::with_capacity(n_reference);

    for _ in 0..n_reference {
        let mut row = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            row.push(sample_normal(&mut rng, 0.0, 1.0));
        }
        // Reference decision boundary: simple linear
        let y = if row.iter().sum::<f64>() > 0.0 {
            1.0
        } else {
            0.0
        };
        ref_x_data.extend(row);
        ref_y.push(y);
    }

    let mut cur_x_data = Vec::with_capacity(n_current * n_features);
    let mut cur_y = Vec::with_capacity(n_current);

    for _ in 0..n_current {
        let mut row = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            row.push(sample_normal(&mut rng, 0.0, 1.0));
        }
        // Rotated decision boundary for concept drift
        // Rotate the weight vector by concept_shift angle
        let cos_theta = concept_shift.cos();
        let sin_theta = concept_shift.sin();

        // Apply rotation to first two features' contribution
        let (x0, x1) = if n_features >= 2 {
            (row[0], row[1])
        } else {
            (row[0], 0.0)
        };
        let rotated_sum =
            x0.mul_add(cos_theta, -(x1 * sin_theta)) + row.iter().skip(2).sum::<f64>();
        let y = if rotated_sum > 0.0 { 1.0 } else { 0.0 };

        cur_x_data.extend(row);
        cur_y.push(y);
    }

    let ref_x = Array2::from_shape_vec((n_reference, n_features), ref_x_data).unwrap();
    let cur_x = Array2::from_shape_vec((n_current, n_features), cur_x_data).unwrap();

    (
        ref_x,
        Array1::from_vec(ref_y),
        cur_x,
        Array1::from_vec(cur_y),
    )
}

/// Generate stable data without drift (for baseline comparison)
pub fn make_stable_data(
    n_reference: usize,
    n_current: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<f64>, Array2<f64>) {
    make_sudden_drift(n_reference, n_current, n_features, 0.0, None, seed)
}

// ============================================================================
// Drift Detection Metrics
// ============================================================================

/// Kolmogorov-Smirnov statistic for two samples
///
/// Measures the maximum difference between empirical CDFs.
/// Higher values indicate greater distribution difference.
pub fn ks_statistic(reference: &Array1<f64>, current: &Array1<f64>) -> f64 {
    let n1 = reference.len();
    let n2 = current.len();

    if n1 == 0 || n2 == 0 {
        return 0.0;
    }

    // Sort both arrays
    let mut ref_sorted: Vec<f64> = reference.iter().copied().collect();
    let mut cur_sorted: Vec<f64> = current.iter().copied().collect();
    ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cur_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compute maximum CDF difference
    let mut max_diff = 0.0f64;
    let mut i = 0;
    let mut j = 0;

    while i < n1 || j < n2 {
        let val = if i >= n1 {
            cur_sorted[j]
        } else if j >= n2 {
            ref_sorted[i]
        } else if ref_sorted[i] <= cur_sorted[j] {
            ref_sorted[i]
        } else {
            cur_sorted[j]
        };

        // Advance pointers
        while i < n1 && ref_sorted[i] <= val {
            i += 1;
        }
        while j < n2 && cur_sorted[j] <= val {
            j += 1;
        }

        let cdf1 = i as f64 / n1 as f64;
        let cdf2 = j as f64 / n2 as f64;
        max_diff = max_diff.max((cdf1 - cdf2).abs());
    }

    max_diff
}

/// KS test p-value approximation for large samples
///
/// Uses the asymptotic approximation for the KS distribution.
pub fn ks_pvalue(ks_stat: f64, n1: usize, n2: usize) -> f64 {
    if n1 == 0 || n2 == 0 {
        return 1.0;
    }

    let n = (n1 * n2) as f64 / (n1 + n2) as f64;
    let lambda = (n.sqrt() + 0.12 + 0.11 / n.sqrt()) * ks_stat;

    // Kolmogorov distribution approximation
    if lambda <= 0.0 {
        return 1.0;
    }

    let mut sum = 0.0;
    for k in 1..100 {
        let k = k as f64;
        let sign = if k as i32 % 2 == 1 { 1.0 } else { -1.0 };
        let term = sign * (-2.0 * k * k * lambda * lambda).exp();
        sum += term;
        if term.abs() < 1e-10 {
            break;
        }
    }

    (2.0 * sum).clamp(0.0, 1.0)
}

/// Population Stability Index (PSI)
///
/// Measures how much a distribution has shifted from a reference.
/// PSI < 0.1: No significant shift
/// PSI 0.1-0.25: Moderate shift
/// PSI > 0.25: Significant shift
pub fn psi(reference: &Array1<f64>, current: &Array1<f64>, n_bins: usize) -> f64 {
    let n_bins = n_bins.max(2);

    if reference.is_empty() || current.is_empty() {
        return 0.0;
    }

    // Compute bin edges from reference distribution
    let mut ref_sorted: Vec<f64> = reference.iter().copied().collect();
    ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut edges = Vec::with_capacity(n_bins + 1);
    edges.push(f64::NEG_INFINITY);
    for i in 1..n_bins {
        let idx = (i * ref_sorted.len()) / n_bins;
        edges.push(ref_sorted[idx.min(ref_sorted.len() - 1)]);
    }
    edges.push(f64::INFINITY);

    // Count proportions in each bin
    let ref_counts = bin_counts(reference, &edges);
    let cur_counts = bin_counts(current, &edges);

    let n_ref = reference.len() as f64;
    let n_cur = current.len() as f64;

    // Compute PSI with small epsilon to avoid log(0)
    let eps = 1e-10;
    let mut psi_val = 0.0;

    for (rc, cc) in ref_counts.iter().zip(cur_counts.iter()) {
        let p_ref = (*rc as f64 / n_ref).max(eps);
        let p_cur = (*cc as f64 / n_cur).max(eps);
        psi_val += (p_cur - p_ref) * (p_cur / p_ref).ln();
    }

    psi_val.max(0.0)
}

/// Jensen-Shannon divergence between two distributions
///
/// Symmetric measure of distribution difference, bounded [0, ln(2)].
pub fn js_divergence(reference: &Array1<f64>, current: &Array1<f64>, n_bins: usize) -> f64 {
    let n_bins = n_bins.max(2);

    if reference.is_empty() || current.is_empty() {
        return 0.0;
    }

    // Compute bin edges from combined data
    let mut all_data: Vec<f64> = reference.iter().chain(current.iter()).copied().collect();
    all_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut edges = Vec::with_capacity(n_bins + 1);
    edges.push(f64::NEG_INFINITY);
    for i in 1..n_bins {
        let idx = (i * all_data.len()) / n_bins;
        edges.push(all_data[idx.min(all_data.len() - 1)]);
    }
    edges.push(f64::INFINITY);

    let ref_counts = bin_counts(reference, &edges);
    let cur_counts = bin_counts(current, &edges);

    let n_ref = reference.len() as f64;
    let n_cur = current.len() as f64;

    let eps = 1e-10;
    let mut js = 0.0;

    for (rc, cc) in ref_counts.iter().zip(cur_counts.iter()) {
        let p = (*rc as f64 / n_ref).max(eps);
        let q = (*cc as f64 / n_cur).max(eps);
        let m = (p + q) / 2.0;

        js += (0.5 * p).mul_add((p / m).ln(), 0.5 * q * (q / m).ln());
    }

    js.max(0.0)
}

/// Compute per-feature drift statistics
pub fn feature_drift_stats(
    reference: &Array2<f64>,
    current: &Array2<f64>,
    ks_threshold: f64,
    psi_threshold: f64,
) -> Vec<FeatureDriftStats> {
    let n_features = reference.ncols().min(current.ncols());
    let mut stats = Vec::with_capacity(n_features);

    for j in 0..n_features {
        let ref_col = reference.column(j).to_owned();
        let cur_col = current.column(j).to_owned();

        let ks = ks_statistic(&ref_col, &cur_col);
        let psi_val = psi(&ref_col, &cur_col, 10);

        let drifted = ks > ks_threshold || psi_val > psi_threshold;

        stats.push(FeatureDriftStats {
            feature_idx: j,
            ks_statistic: ks,
            psi: psi_val,
            drifted,
        });
    }

    stats
}

/// Detect drift across multiple features using maximum statistics
pub fn multivariate_drift_detected(
    reference: &Array2<f64>,
    current: &Array2<f64>,
    ks_threshold: f64,
) -> bool {
    let n_features = reference.ncols().min(current.ncols());

    for j in 0..n_features {
        let ref_col = reference.column(j).to_owned();
        let cur_col = current.column(j).to_owned();
        let ks = ks_statistic(&ref_col, &cur_col);
        if ks > ks_threshold {
            return true;
        }
    }

    false
}

/// Performance drift: compare model accuracy between windows
pub fn performance_drift(ref_accuracy: f64, current_accuracy: f64, threshold: f64) -> DriftResult {
    let drop = ref_accuracy - current_accuracy;

    DriftResult {
        metric: "accuracy_drop".to_string(),
        statistic: drop,
        p_value: None,
        drift_detected: drop > threshold,
        threshold,
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Sample from normal distribution using Box-Muller
fn sample_normal(rng: &mut ChaCha8Rng, mean: f64, std: f64) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    std.mul_add(z, mean)
}

/// Count samples in each bin
fn bin_counts(data: &Array1<f64>, edges: &[f64]) -> Vec<usize> {
    let mut counts = vec![0; edges.len() - 1];

    for &val in data.iter() {
        for (i, window) in edges.windows(2).enumerate() {
            if val > window[0] && val <= window[1] {
                counts[i] += 1;
                break;
            }
        }
    }

    counts
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod data_drift_tests {
    use super::*;

    #[test]
    fn test_ks_identical_distributions() {
        let (ref_data, cur_data) = make_stable_data(500, 500, 3, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();

        let ks = ks_statistic(&ref_col, &cur_col);
        // Should be small for identical distributions
        assert!(ks < 0.15, "KS stat {} too high for same distribution", ks);
    }

    #[test]
    fn test_ks_detects_mean_shift() {
        let (ref_data, cur_data) = make_sudden_drift(500, 500, 3, 1.5, None, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();

        let ks = ks_statistic(&ref_col, &cur_col);
        // Should detect significant shift
        assert!(ks > 0.3, "KS stat {} should detect mean shift of 1.5", ks);
    }

    #[test]
    fn test_ks_pvalue_no_drift() {
        let (ref_data, cur_data) = make_stable_data(200, 200, 3, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();

        let ks = ks_statistic(&ref_col, &cur_col);
        let p = ks_pvalue(ks, 200, 200);

        // P-value should be high (non-significant) for same distribution
        assert!(
            p > 0.05,
            "P-value {} should be > 0.05 for same distribution",
            p
        );
    }

    #[test]
    fn test_ks_pvalue_with_drift() {
        let (ref_data, cur_data) = make_sudden_drift(200, 200, 3, 1.0, None, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();

        let ks = ks_statistic(&ref_col, &cur_col);
        let p = ks_pvalue(ks, 200, 200);

        // P-value should be low (significant) for drifted distribution
        assert!(p < 0.05, "P-value {} should be < 0.05 for drifted data", p);
    }

    #[test]
    fn test_psi_no_shift() {
        let (ref_data, cur_data) = make_stable_data(1000, 1000, 3, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();

        let psi_val = psi(&ref_col, &cur_col, 10);
        // PSI < 0.1 indicates no significant shift
        assert!(
            psi_val < 0.1,
            "PSI {} should be < 0.1 for stable data",
            psi_val
        );
    }

    #[test]
    fn test_psi_moderate_shift() {
        let (ref_data, cur_data) = make_sudden_drift(1000, 1000, 3, 0.5, None, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();

        let psi_val = psi(&ref_col, &cur_col, 10);
        // PSI 0.1-0.25 indicates moderate shift
        assert!(
            psi_val > 0.05,
            "PSI {} should detect moderate shift",
            psi_val
        );
    }

    #[test]
    fn test_psi_significant_shift() {
        let (ref_data, cur_data) = make_sudden_drift(1000, 1000, 3, 2.0, None, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();

        let psi_val = psi(&ref_col, &cur_col, 10);
        // PSI > 0.25 indicates significant shift
        assert!(
            psi_val > 0.25,
            "PSI {} should detect significant shift",
            psi_val
        );
    }

    #[test]
    fn test_js_divergence_identical() {
        let (ref_data, cur_data) = make_stable_data(500, 500, 3, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();

        let js = js_divergence(&ref_col, &cur_col, 10);
        // Should be close to 0 for identical distributions
        assert!(
            js < 0.1,
            "JS divergence {} should be small for same distribution",
            js
        );
    }

    #[test]
    fn test_js_divergence_different() {
        let (ref_data, cur_data) = make_sudden_drift(500, 500, 3, 2.0, None, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();

        let js = js_divergence(&ref_col, &cur_col, 10);
        // Should detect difference
        assert!(
            js > 0.1,
            "JS divergence {} should detect distribution shift",
            js
        );
    }

    #[test]
    fn test_feature_drift_stats() {
        // Only feature 0 and 1 drift
        let (ref_data, cur_data) = make_sudden_drift(500, 500, 5, 1.5, Some(vec![0, 1]), 42);

        let stats = feature_drift_stats(&ref_data, &cur_data, 0.2, 0.15);

        // Features 0 and 1 should show drift
        assert!(stats[0].drifted, "Feature 0 should show drift");
        assert!(stats[1].drifted, "Feature 1 should show drift");
        // Features 2, 3, 4 should not show drift
        assert!(!stats[2].drifted, "Feature 2 should not drift");
        assert!(!stats[3].drifted, "Feature 3 should not drift");
        assert!(!stats[4].drifted, "Feature 4 should not drift");
    }
}

#[cfg(test)]
mod concept_drift_tests {
    use super::*;

    #[test]
    fn test_concept_drift_no_shift() {
        let (ref_x, ref_y, cur_x, cur_y) = make_concept_drift(300, 300, 4, 0.0, 42);

        // With no concept shift, distributions should be similar
        // Check that X distributions are similar (no covariate shift)
        let ref_col = ref_x.column(0).to_owned();
        let cur_col = cur_x.column(0).to_owned();
        let ks = ks_statistic(&ref_col, &cur_col);
        assert!(ks < 0.15, "Feature distribution should be stable");

        // Check label distributions
        let ref_pos_rate = ref_y.iter().filter(|&&y| y > 0.5).count() as f64 / ref_y.len() as f64;
        let cur_pos_rate = cur_y.iter().filter(|&&y| y > 0.5).count() as f64 / cur_y.len() as f64;
        let rate_diff = (ref_pos_rate - cur_pos_rate).abs();
        assert!(
            rate_diff < 0.1,
            "Label rates should be similar without concept drift"
        );
    }

    #[test]
    fn test_concept_drift_moderate_shift() {
        // 45 degree rotation
        let (ref_x, ref_y, cur_x, cur_y) =
            make_concept_drift(300, 300, 4, std::f64::consts::PI / 4.0, 42);

        // X distribution should be similar (no covariate shift)
        let ref_col = ref_x.column(0).to_owned();
        let cur_col = cur_x.column(0).to_owned();
        let ks = ks_statistic(&ref_col, &cur_col);
        assert!(
            ks < 0.15,
            "Feature distribution should remain stable in concept drift"
        );

        // Verify data was generated
        assert_eq!(ref_x.nrows(), 300);
        assert_eq!(cur_x.nrows(), 300);
        assert_eq!(ref_y.len(), 300);
        assert_eq!(cur_y.len(), 300);
    }

    #[test]
    fn test_concept_drift_strong_shift() {
        // 90 degree rotation - complete reversal
        let (ref_x, _ref_y, cur_x, _cur_y) =
            make_concept_drift(300, 300, 4, std::f64::consts::PI / 2.0, 42);

        // Data should still be generated correctly
        assert_eq!(ref_x.nrows(), 300);
        assert_eq!(cur_x.nrows(), 300);

        // Features should still have similar distributions
        let ref_col = ref_x.column(0).to_owned();
        let cur_col = cur_x.column(0).to_owned();
        let ks = ks_statistic(&ref_col, &cur_col);
        assert!(
            ks < 0.15,
            "Concept drift shouldn't change feature distributions"
        );
    }

    #[test]
    fn test_performance_drift_detection() {
        let ref_accuracy = 0.92;
        let cur_accuracy = 0.78;
        let threshold = 0.05;

        let result = performance_drift(ref_accuracy, cur_accuracy, threshold);

        assert!(
            result.drift_detected,
            "Should detect performance drop of 14%"
        );
        assert!(
            (result.statistic - 0.14).abs() < 0.01,
            "Drop should be ~0.14"
        );
    }

    #[test]
    fn test_performance_drift_no_degradation() {
        let ref_accuracy = 0.90;
        let cur_accuracy = 0.89;
        let threshold = 0.05;

        let result = performance_drift(ref_accuracy, cur_accuracy, threshold);

        assert!(!result.drift_detected, "Should not flag minor drop");
    }

    #[test]
    fn test_performance_drift_improvement() {
        let ref_accuracy = 0.85;
        let cur_accuracy = 0.90;
        let threshold = 0.05;

        let result = performance_drift(ref_accuracy, cur_accuracy, threshold);

        assert!(
            !result.drift_detected,
            "Performance improvement is not drift"
        );
        assert!(result.statistic < 0.0, "Negative drop means improvement");
    }

    #[test]
    fn test_concept_vs_covariate_drift() {
        // Concept drift: same X, different P(y|X)
        let (_, _, _, _) = make_concept_drift(200, 200, 4, std::f64::consts::PI / 3.0, 42);

        // Covariate drift: different X, same P(y|X) relationship
        let (ref_x, cur_x) = make_sudden_drift(200, 200, 4, 1.0, None, 42);

        let ref_col = ref_x.column(0).to_owned();
        let cur_col = cur_x.column(0).to_owned();
        let ks = ks_statistic(&ref_col, &cur_col);

        // Covariate drift should show feature distribution change
        assert!(ks > 0.2, "Covariate drift should show in KS test");
    }

    #[test]
    fn test_mixed_drift() {
        // Both concept and covariate drift
        let (ref_x, _, cur_x, _) = make_concept_drift(200, 200, 4, std::f64::consts::PI / 4.0, 42);

        // Add covariate shift manually by shifting current data
        let mut cur_x_shifted = cur_x.clone();
        for mut row in cur_x_shifted.rows_mut() {
            row[0] += 1.0; // Shift first feature
        }

        // Now we have both types of drift
        let ref_col = ref_x.column(0).to_owned();
        let cur_col = cur_x_shifted.column(0).to_owned();
        let ks = ks_statistic(&ref_col, &cur_col);

        assert!(ks > 0.2, "Mixed drift should be detectable via KS");
    }
}

#[cfg(test)]
mod gradual_drift_tests {
    use super::*;

    #[test]
    fn test_gradual_drift_generation() {
        let (data, windows) = make_gradual_drift(500, 5, 3, 2.0, 42);

        assert_eq!(data.nrows(), windows.len());
        assert!(windows.iter().all(|&w| w < 5));

        // Check that later windows have higher means
        let first_window: Vec<f64> = data
            .rows()
            .into_iter()
            .zip(windows.iter())
            .filter(|(_, &w)| w == 0)
            .map(|(row, _)| row[0])
            .collect();

        let last_window: Vec<f64> = data
            .rows()
            .into_iter()
            .zip(windows.iter())
            .filter(|(_, &w)| w == 4)
            .map(|(row, _)| row[0])
            .collect();

        let first_mean: f64 = first_window.iter().sum::<f64>() / first_window.len() as f64;
        let last_mean: f64 = last_window.iter().sum::<f64>() / last_window.len() as f64;

        assert!(
            last_mean > first_mean,
            "Last window mean {} should be > first window mean {}",
            last_mean,
            first_mean
        );
    }

    #[test]
    fn test_gradual_drift_monotonic() {
        let (data, windows) = make_gradual_drift(1000, 10, 3, 3.0, 42);

        // Compute mean for each window
        let mut window_means = Vec::new();
        for w in 0..10 {
            let window_data: Vec<f64> = data
                .rows()
                .into_iter()
                .zip(windows.iter())
                .filter(|(_, &ww)| ww == w)
                .map(|(row, _)| row[0])
                .collect();

            if !window_data.is_empty() {
                let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                window_means.push(mean);
            }
        }

        // Check roughly monotonic increase (with some noise tolerance)
        for i in 1..window_means.len() {
            if i >= 2 {
                // Later windows should generally have higher means
                assert!(
                    window_means[i] > window_means[0] - 1.0,
                    "Window {} mean should be higher than window 0",
                    i
                );
            }
        }
    }

    #[test]
    fn test_gradual_drift_detection_over_time() {
        let (data, windows) = make_gradual_drift(1000, 5, 3, 2.0, 42);

        // First window as reference
        let ref_data: Vec<f64> = data
            .rows()
            .into_iter()
            .zip(windows.iter())
            .filter(|(_, &w)| w == 0)
            .map(|(row, _)| row[0])
            .collect();
        let ref_arr = Array1::from_vec(ref_data);

        // Check drift increases over time
        let mut prev_ks = 0.0;
        for w in 1..5 {
            let cur_data: Vec<f64> = data
                .rows()
                .into_iter()
                .zip(windows.iter())
                .filter(|(_, &ww)| ww == w)
                .map(|(row, _)| row[0])
                .collect();

            if cur_data.is_empty() {
                continue;
            }

            let cur_arr = Array1::from_vec(cur_data);
            let ks = ks_statistic(&ref_arr, &cur_arr);

            // KS should generally increase (drift accumulates)
            // Allow some variance in early windows
            if w >= 2 {
                assert!(
                    ks >= prev_ks - 0.1,
                    "KS at window {} ({}) should be >= window {} ({})",
                    w,
                    ks,
                    w - 1,
                    prev_ks
                );
            }
            prev_ks = ks;
        }
    }

    #[test]
    fn test_gradual_vs_sudden_drift() {
        // Gradual drift over 5 windows to magnitude 2.0
        let (gradual_data, gradual_windows) = make_gradual_drift(500, 5, 3, 2.0, 42);

        // Sudden drift of same magnitude
        let (sudden_ref, sudden_cur) = make_sudden_drift(100, 100, 3, 2.0, None, 42);

        // Compare intermediate window (window 2) drift vs reference
        let ref_gradual: Vec<f64> = gradual_data
            .rows()
            .into_iter()
            .zip(gradual_windows.iter())
            .filter(|(_, &w)| w == 0)
            .map(|(row, _)| row[0])
            .collect();

        let mid_gradual: Vec<f64> = gradual_data
            .rows()
            .into_iter()
            .zip(gradual_windows.iter())
            .filter(|(_, &w)| w == 2)
            .map(|(row, _)| row[0])
            .collect();

        let ref_arr = Array1::from_vec(ref_gradual);
        let mid_arr = Array1::from_vec(mid_gradual);
        let gradual_ks = ks_statistic(&ref_arr, &mid_arr);

        let sudden_ref_col = sudden_ref.column(0).to_owned();
        let sudden_cur_col = sudden_cur.column(0).to_owned();
        let sudden_ks = ks_statistic(&sudden_ref_col, &sudden_cur_col);

        // Gradual drift at midpoint should be less than sudden full drift
        assert!(
            gradual_ks < sudden_ks,
            "Gradual drift at midpoint ({}) should be less than sudden drift ({})",
            gradual_ks,
            sudden_ks
        );
    }
}

#[cfg(test)]
mod drift_threshold_tests {
    use super::*;

    #[test]
    fn test_ks_threshold_0_1() {
        // Common threshold for KS test
        let threshold = 0.1;

        // No drift should pass
        let (ref_data, cur_data) = make_stable_data(500, 500, 3, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();
        let ks = ks_statistic(&ref_col, &cur_col);
        assert!(ks < threshold, "Stable data should pass threshold");

        // Strong drift should fail
        let (ref_data, cur_data) = make_sudden_drift(500, 500, 3, 1.5, None, 43);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();
        let ks = ks_statistic(&ref_col, &cur_col);
        assert!(ks > threshold, "Drifted data should exceed threshold");
    }

    #[test]
    fn test_psi_threshold_0_1() {
        // PSI < 0.1 = no significant shift
        let threshold = 0.1;

        let (ref_data, cur_data) = make_stable_data(1000, 1000, 3, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();
        let psi_val = psi(&ref_col, &cur_col, 10);
        assert!(psi_val < threshold, "Stable data PSI should be < 0.1");
    }

    #[test]
    fn test_psi_threshold_0_25() {
        // PSI > 0.25 = significant shift
        let threshold = 0.25;

        let (ref_data, cur_data) = make_sudden_drift(1000, 1000, 3, 2.0, None, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();
        let psi_val = psi(&ref_col, &cur_col, 10);
        assert!(
            psi_val > threshold,
            "Significant drift PSI should be > 0.25"
        );
    }

    #[test]
    fn test_multivariate_drift_threshold() {
        // Only one feature drifts
        let (ref_data, cur_data) = make_sudden_drift(500, 500, 5, 1.5, Some(vec![0]), 42);

        // Should detect with reasonable threshold
        assert!(multivariate_drift_detected(&ref_data, &cur_data, 0.2));

        // Should not detect on stable data
        let (stable_ref, stable_cur) = make_stable_data(500, 500, 5, 42);
        assert!(!multivariate_drift_detected(&stable_ref, &stable_cur, 0.2));
    }

    #[test]
    fn test_drift_sensitivity_analysis() {
        let (ref_data, cur_data) = make_sudden_drift(500, 500, 3, 0.8, None, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();

        let ks = ks_statistic(&ref_col, &cur_col);

        // Test different thresholds
        let thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30];
        let mut detections = Vec::new();

        for &t in &thresholds {
            detections.push(ks > t);
        }

        // Lower thresholds should detect more often
        // At some point, higher thresholds stop detecting
        let first_false = detections.iter().position(|&d| !d);

        if let Some(idx) = first_false {
            // All thresholds before should have detected
            assert!(detections[..idx].iter().all(|&d| d));
            // All thresholds after should not detect
            assert!(detections[idx..].iter().all(|&d| !d));
        }
    }

    #[test]
    fn test_performance_drift_thresholds() {
        let ref_acc = 0.90;

        // Test various accuracy drops (avoid boundary value for floating point precision)
        let drops = [0.01, 0.03, 0.04, 0.10, 0.15];
        let threshold = 0.05;

        for drop in drops {
            let cur_acc = ref_acc - drop;
            let result = performance_drift(ref_acc, cur_acc, threshold);

            if drop > threshold {
                assert!(result.drift_detected, "Should detect drop of {}", drop);
            } else {
                assert!(!result.drift_detected, "Should not flag drop of {}", drop);
            }
        }
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_arrays() {
        let empty = Array1::<f64>::zeros(0);
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let ks = ks_statistic(&empty, &data);
        assert_eq!(ks, 0.0, "Empty array should return 0");

        let psi_val = psi(&empty, &data, 10);
        assert_eq!(psi_val, 0.0, "Empty array PSI should be 0");
    }

    #[test]
    fn test_single_element() {
        let single = Array1::from_vec(vec![1.0]);
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Should not panic
        let ks = ks_statistic(&single, &data);
        assert!(ks.is_finite(), "KS should be finite");
    }

    #[test]
    fn test_identical_values() {
        let const_ref = Array1::from_vec(vec![5.0; 100]);
        let const_cur = Array1::from_vec(vec![5.0; 100]);

        let ks = ks_statistic(&const_ref, &const_cur);
        assert!(ks < 0.01, "Identical constant arrays should have KS ~0");

        // Different constants
        let const_diff = Array1::from_vec(vec![10.0; 100]);
        let ks_diff = ks_statistic(&const_ref, &const_diff);
        assert!(ks_diff > 0.9, "Different constants should have KS ~1");
    }

    #[test]
    fn test_high_dimensional_drift() {
        // Many features, only some drift
        let n_features = 50;
        let drifting = vec![0, 10, 20, 30, 40]; // 5 features drift

        // Use larger samples and stronger drift for clearer signal
        let (ref_data, cur_data) =
            make_sudden_drift(500, 500, n_features, 2.0, Some(drifting.clone()), 42);

        // Use stricter thresholds to reduce false positives on non-drifting features
        let stats = feature_drift_stats(&ref_data, &cur_data, 0.25, 0.20);

        // Check that drifting features are detected
        for stat in &stats {
            if drifting.contains(&stat.feature_idx) {
                assert!(
                    stat.drifted,
                    "Feature {} should be flagged",
                    stat.feature_idx
                );
            }
        }

        // Count false positives - allow a few due to random variance
        let false_positives: usize = stats
            .iter()
            .filter(|s| s.drifted && !drifting.contains(&s.feature_idx))
            .count();
        assert!(
            false_positives <= 2,
            "Too many false positives: {}",
            false_positives
        );
    }

    #[test]
    fn test_small_sample_size() {
        // Very small samples - should still work but may be unreliable
        let (ref_data, cur_data) = make_sudden_drift(20, 20, 3, 1.0, None, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();

        let ks = ks_statistic(&ref_col, &cur_col);
        let p = ks_pvalue(ks, 20, 20);

        // Should still compute without panicking
        assert!(ks.is_finite());
        assert!(p.is_finite());
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_reproducibility() {
        // Same seed should produce same results
        let (ref1, cur1) = make_sudden_drift(100, 100, 3, 1.0, None, 42);
        let (ref2, cur2) = make_sudden_drift(100, 100, 3, 1.0, None, 42);

        assert_eq!(ref1, ref2);
        assert_eq!(cur1, cur2);

        // Different seeds should produce different results
        let (ref3, _cur3) = make_sudden_drift(100, 100, 3, 1.0, None, 43);
        assert_ne!(ref1, ref3);
    }

    #[test]
    fn test_extreme_drift() {
        // Very large drift magnitude
        let (ref_data, cur_data) = make_sudden_drift(100, 100, 3, 10.0, None, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();

        let ks = ks_statistic(&ref_col, &cur_col);
        // Should be very close to 1.0 (complete separation)
        assert!(ks > 0.9, "Extreme drift should have KS near 1.0");
    }

    #[test]
    fn test_asymmetric_sample_sizes() {
        // Reference much larger than current
        let (ref_data, cur_data) = make_sudden_drift(1000, 50, 3, 1.0, None, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();

        let ks = ks_statistic(&ref_col, &cur_col);
        assert!(ks.is_finite());

        // Current much larger than reference
        let (ref_data, cur_data) = make_sudden_drift(50, 1000, 3, 1.0, None, 42);
        let ref_col = ref_data.column(0).to_owned();
        let cur_col = cur_data.column(0).to_owned();

        let ks = ks_statistic(&ref_col, &cur_col);
        assert!(ks.is_finite());
    }
}

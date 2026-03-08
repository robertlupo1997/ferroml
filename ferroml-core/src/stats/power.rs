//! Statistical power analysis

use serde::{Deserialize, Serialize};

/// Power analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysis {
    /// Statistical power (1 - beta)
    pub power: f64,
    /// Sample size
    pub n: usize,
    /// Effect size
    pub effect_size: f64,
    /// Alpha level
    pub alpha: f64,
}

/// Calculate required sample size for desired power
pub fn sample_size_for_power(effect_size: f64, alpha: f64, power: f64, _test_type: &str) -> usize {
    // Simplified calculation for two-sample t-test
    let z_alpha = z_critical(1.0 - alpha / 2.0);
    let z_beta = z_critical(power);

    let n = 2.0 * ((z_alpha + z_beta) / effect_size).powi(2);
    n.ceil() as usize
}

/// Calculate power for given sample size
pub fn power_for_sample_size(n: usize, effect_size: f64, alpha: f64) -> f64 {
    let z_alpha = z_critical(1.0 - alpha / 2.0);
    let z = effect_size.mul_add((n as f64 / 2.0).sqrt(), -z_alpha);
    normal_cdf(z)
}

fn z_critical(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    let p = if p > 0.5 { 1.0 - p } else { p };
    let t = (-2.0 * p.ln()).sqrt();
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    let z = t
        - (c2 * t).mul_add(t, c0 + c1 * t)
            / (d3 * t * t).mul_add(t, (d2 * t).mul_add(t, 1.0 + d1 * t));
    if p > 0.5 {
        -z
    } else {
        z
    }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = ((a5 * t + a4).mul_add(t, a3).mul_add(t, a2).mul_add(t, a1) * t)
        .mul_add(-(-x * x).exp(), 1.0);
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ---- PowerAnalysis struct ----

    #[test]
    fn test_power_analysis_struct() {
        let pa = PowerAnalysis {
            power: 0.8,
            n: 64,
            effect_size: 0.5,
            alpha: 0.05,
        };
        assert_eq!(pa.n, 64);
        assert_relative_eq!(pa.power, 0.8);
        assert_relative_eq!(pa.effect_size, 0.5);
        assert_relative_eq!(pa.alpha, 0.05);
    }

    #[test]
    fn test_power_analysis_clone_and_debug() {
        let pa = PowerAnalysis {
            power: 0.8,
            n: 64,
            effect_size: 0.5,
            alpha: 0.05,
        };
        let pa2 = pa.clone();
        assert_eq!(pa2.n, pa.n);
        // Debug trait should work
        let _ = format!("{:?}", pa);
    }

    // ---- sample_size_for_power ----

    #[test]
    fn test_sample_size_medium_effect() {
        // For medium effect size (0.5), alpha=0.05, power=0.80
        // Standard result is ~64 per group for two-sample t-test
        let n = sample_size_for_power(0.5, 0.05, 0.80, "two_sample_t");
        // Should be in a reasonable range (50-80)
        assert!(n >= 50 && n <= 80, "n={} not in expected range 50-80", n);
    }

    #[test]
    fn test_sample_size_large_effect() {
        // Larger effect size should require fewer samples
        let n_large = sample_size_for_power(0.8, 0.05, 0.80, "two_sample_t");
        let n_medium = sample_size_for_power(0.5, 0.05, 0.80, "two_sample_t");
        assert!(
            n_large < n_medium,
            "larger effect should need fewer samples: {} vs {}",
            n_large,
            n_medium
        );
    }

    #[test]
    fn test_sample_size_small_effect() {
        // Smaller effect size should require more samples
        let n_small = sample_size_for_power(0.2, 0.05, 0.80, "two_sample_t");
        let n_medium = sample_size_for_power(0.5, 0.05, 0.80, "two_sample_t");
        assert!(
            n_small > n_medium,
            "smaller effect should need more samples: {} vs {}",
            n_small,
            n_medium
        );
    }

    #[test]
    fn test_sample_size_higher_power_needs_more() {
        let n_80 = sample_size_for_power(0.5, 0.05, 0.80, "two_sample_t");
        let n_90 = sample_size_for_power(0.5, 0.05, 0.90, "two_sample_t");
        assert!(
            n_90 > n_80,
            "higher power should need more samples: {} vs {}",
            n_90,
            n_80
        );
    }

    #[test]
    fn test_sample_size_stricter_alpha_needs_more() {
        let n_05 = sample_size_for_power(0.5, 0.05, 0.80, "two_sample_t");
        let n_01 = sample_size_for_power(0.5, 0.01, 0.80, "two_sample_t");
        assert!(
            n_01 > n_05,
            "stricter alpha should need more samples: {} vs {}",
            n_01,
            n_05
        );
    }

    #[test]
    fn test_sample_size_returns_at_least_one() {
        // Even with huge effect size, should return at least 1
        let n = sample_size_for_power(10.0, 0.05, 0.80, "two_sample_t");
        assert!(n >= 1);
    }

    // ---- power_for_sample_size ----

    #[test]
    fn test_power_increases_with_sample_size() {
        let p1 = power_for_sample_size(20, 0.5, 0.05);
        let p2 = power_for_sample_size(50, 0.5, 0.05);
        let p3 = power_for_sample_size(200, 0.5, 0.05);
        assert!(p2 > p1, "power should increase: {} vs {}", p2, p1);
        assert!(p3 > p2, "power should increase: {} vs {}", p3, p2);
    }

    #[test]
    fn test_power_increases_with_effect_size() {
        let p1 = power_for_sample_size(50, 0.2, 0.05);
        let p2 = power_for_sample_size(50, 0.5, 0.05);
        let p3 = power_for_sample_size(50, 0.8, 0.05);
        assert!(
            p2 > p1,
            "power should increase with effect: {} vs {}",
            p2,
            p1
        );
        assert!(
            p3 > p2,
            "power should increase with effect: {} vs {}",
            p3,
            p2
        );
    }

    #[test]
    fn test_power_bounded_0_1() {
        let p = power_for_sample_size(1000, 1.0, 0.05);
        assert!(p >= 0.0 && p <= 1.0, "power should be in [0,1]: {}", p);

        let p_small = power_for_sample_size(2, 0.01, 0.05);
        assert!(
            p_small >= 0.0 && p_small <= 1.0,
            "power should be in [0,1]: {}",
            p_small
        );
    }

    #[test]
    fn test_power_large_sample_approaches_one() {
        let p = power_for_sample_size(10000, 0.5, 0.05);
        assert!(p > 0.99, "power with large n should be near 1.0: {}", p);
    }

    #[test]
    fn test_power_roundtrip_with_sample_size() {
        // If we compute n for power=0.8, then computing power for that n should be ~0.8
        let target_power = 0.80;
        let n = sample_size_for_power(0.5, 0.05, target_power, "two_sample_t");
        let achieved_power = power_for_sample_size(n, 0.5, 0.05);
        // Should be at least the target (since n is ceil'd)
        assert!(
            achieved_power >= target_power - 0.05,
            "roundtrip power {} should be near target {}",
            achieved_power,
            target_power
        );
    }

    // ---- Internal functions: z_critical, normal_cdf, erf ----

    #[test]
    fn test_z_critical_known_values() {
        // z(0.975) ~= 1.96 for two-tailed 95% CI
        let z = z_critical(0.975);
        assert_relative_eq!(z, 1.96, epsilon = 0.01);

        // z(0.5) = 0
        let z_half = z_critical(0.5);
        assert_relative_eq!(z_half, 0.0, epsilon = 0.01);

        // z(0.995) ~= 2.576 for 99% CI
        let z_99 = z_critical(0.995);
        assert_relative_eq!(z_99, 2.576, epsilon = 0.01);
    }

    #[test]
    fn test_z_critical_boundary_returns_nan() {
        assert!(z_critical(0.0).is_nan());
        assert!(z_critical(1.0).is_nan());
    }

    #[test]
    fn test_z_critical_symmetry() {
        // |z(p)| = |z(1-p)| (magnitudes match)
        let z_low = z_critical(0.025);
        let z_high = z_critical(0.975);
        assert_relative_eq!(z_low.abs(), z_high.abs(), epsilon = 0.01);
    }

    #[test]
    fn test_normal_cdf_known_values() {
        // CDF(0) = 0.5
        assert_relative_eq!(normal_cdf(0.0), 0.5, epsilon = 1e-6);

        // CDF(large) ~ 1.0
        assert!(normal_cdf(5.0) > 0.999);

        // CDF(-large) ~ 0.0
        assert!(normal_cdf(-5.0) < 0.001);
    }

    #[test]
    fn test_normal_cdf_symmetry() {
        // CDF(x) + CDF(-x) = 1
        let x = 1.5;
        assert_relative_eq!(normal_cdf(x) + normal_cdf(-x), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_erf_known_values() {
        // erf(0) = 0
        assert_relative_eq!(erf(0.0), 0.0, epsilon = 1e-6);

        // erf(large) ~ 1.0
        assert_relative_eq!(erf(5.0), 1.0, epsilon = 1e-6);

        // erf(-x) = -erf(x)
        assert_relative_eq!(erf(-1.0), -erf(1.0), epsilon = 1e-10);

        // erf(1) ~= 0.8427
        assert_relative_eq!(erf(1.0), 0.8427, epsilon = 0.001);
    }
}

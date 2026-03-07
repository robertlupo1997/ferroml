//! Shared mathematical functions for statistical computations
//!
//! This module centralizes numerical algorithms that are used across multiple
//! stats and metrics submodules: gamma, beta, incomplete beta, t-distribution,
//! normal distribution, and related utilities.
//!
//! All functions are deterministic and have no side effects.

/// Log gamma function (Lanczos approximation)
///
/// Computes ln(Gamma(x)) for x > 0 using the Lanczos approximation with
/// 6 coefficients. Accurate to ~15 significant digits for x > 0.5.
pub fn gamma_ln(x: f64) -> f64 {
    let coeffs = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.001_208_650_973_866_179,
        -0.000_005_395_239_384_953,
    ];

    let tmp = x + 5.5;
    let tmp = (x + 0.5).mul_add(-tmp.ln(), tmp);

    let mut ser = 1.000_000_000_190_015;
    for (i, &c) in coeffs.iter().enumerate() {
        ser += c / (x + i as f64 + 1.0);
    }

    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

/// Beta function: B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b)
pub fn beta(a: f64, b: f64) -> f64 {
    (gamma_ln(a) + gamma_ln(b) - gamma_ln(a + b)).exp()
}

/// Log of Beta function: ln(B(a, b))
pub fn ln_beta(a: f64, b: f64) -> f64 {
    gamma_ln(a) + gamma_ln(b) - gamma_ln(a + b)
}

/// Regularized incomplete beta function I_x(a, b)
///
/// Uses Lentz's continued fraction algorithm with a symmetry relation
/// for improved convergence when x > (a+1)/(a+b+2).
pub fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let bt = b
        .mul_add(
            (1.0 - x).ln(),
            a.mul_add(x.ln(), gamma_ln(a + b) - gamma_ln(a) - gamma_ln(b)),
        )
        .exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Lentz's continued fraction for the regularized incomplete beta function
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 3e-12;
    let fpmin = 1e-30;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m = m as f64;
        let m2 = 2.0 * m;

        // Even step
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    h
}

/// Student's t CDF: P(T <= t) for t-distribution with `df` degrees of freedom
pub fn t_cdf(t: f64, df: f64) -> f64 {
    let x = df / t.mul_add(t, df);
    0.5f64.mul_add((1.0 - incomplete_beta(df / 2.0, 0.5, x)).copysign(t), 0.5)
}

/// Student's t PDF
pub fn t_pdf(t: f64, df: f64) -> f64 {
    let coef = 0.5f64.mul_add(
        -(df * std::f64::consts::PI).ln(),
        gamma_ln((df + 1.0) / 2.0) - gamma_ln(df / 2.0),
    );
    ((df + 1.0) / 2.0)
        .mul_add(-(1.0 + t * t / df).ln(), coef)
        .exp()
}

/// Critical value from Student's t distribution (inverse CDF)
///
/// Uses Newton-Raphson iteration starting from the normal approximation.
/// For df > 100, returns the normal approximation directly.
pub fn t_critical(p: f64, df: f64) -> f64 {
    if df > 100.0 {
        return z_critical(p);
    }

    let mut t = z_critical(p);

    for _ in 0..10 {
        let cdf = t_cdf(t, df);
        let pdf = t_pdf(t, df);
        if pdf.abs() < 1e-10 {
            break;
        }
        t -= (cdf - p) / pdf;
    }

    t
}

/// Standard normal CDF: Phi(x) = P(Z <= x)
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Inverse standard normal CDF (probit / quantile function): Phi^{-1}(p)
///
/// Uses the rational approximation of Abramowitz & Stegun (26.2.23).
pub fn z_critical(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }

    let q = if p > 0.5 { 1.0 - p } else { p };
    let t = (-2.0 * q.ln()).sqrt();

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
        z
    } else {
        -z
    }
}

/// Error function approximation (Abramowitz & Stegun)
pub fn erf(x: f64) -> f64 {
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

/// Chi-squared CDF: P(X <= x) for chi-squared distribution with `df` degrees of freedom
pub fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    incomplete_gamma(df / 2.0, x / 2.0)
}

/// Regularized lower incomplete gamma function: P(a, x) = gamma(a, x) / Gamma(a)
///
/// Uses series expansion for x < a + 1 and continued fraction otherwise.
pub fn incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }

    if x < a + 1.0 {
        // Series expansion
        let mut sum = 1.0 / a;
        let mut term = sum;

        for n in 1..100 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < 1e-10 {
                break;
            }
        }

        sum * (a.mul_add(x.ln(), -x) - gamma_ln(a)).exp()
    } else {
        // Continued fraction for large x
        1.0 - incomplete_gamma_cf(a, x)
    }
}

/// Upper incomplete gamma via continued fraction (for large x)
fn incomplete_gamma_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..100 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an.mul_add(d, b);
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    h * (a.mul_add(x.ln(), -x) - gamma_ln(a)).exp()
}

/// Compute percentile confidence interval from sorted bootstrap samples
pub fn percentile_ci(scores: &[f64], confidence: f64) -> (f64, f64) {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let n = sorted.len();

    let lower_idx = ((alpha / 2.0) * n as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n as f64).ceil() as usize;

    let lower_idx = lower_idx.min(n.saturating_sub(1));
    let upper_idx = upper_idx.min(n.saturating_sub(1));

    (sorted[lower_idx], sorted[upper_idx])
}

/// Compute standard error (sample standard deviation) from bootstrap samples
pub fn bootstrap_std_error(scores: &[f64]) -> f64 {
    let n = scores.len() as f64;
    if n <= 1.0 {
        return 0.0;
    }

    let mean = scores.iter().sum::<f64>() / n;
    let variance = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    variance.sqrt()
}

/// Compute percentile from a pre-sorted slice using linear interpolation.
///
/// `p` is in [0, 1]. Returns the interpolated value at the given percentile.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }

    let index = p * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let weight = index - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower].mul_add(1.0 - weight, sorted[upper] * weight)
    }
}

/// Inverse standard normal CDF (probit / quantile function): Phi^{-1}(p)
///
/// Alias for [`z_critical`] with a more descriptive name.
pub fn norm_ppf(p: f64) -> f64 {
    z_critical(p)
}

/// Compute Pearson correlation coefficient between two arrays.
///
/// Does not validate lengths (caller is responsible).
pub fn pearson_r(x: &ndarray::Array1<f64>, y: &ndarray::Array1<f64>) -> f64 {
    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }

    let denom = sum_x2.sqrt() * sum_y2.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        sum_xy / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incomplete_beta_known_values() {
        assert!((incomplete_beta(1.0, 1.0, 0.5) - 0.5).abs() < 1e-8);
        assert!((incomplete_beta(2.0, 2.0, 0.5) - 0.5).abs() < 1e-8);
        assert!((incomplete_beta(2.0, 5.0, 0.3) - 0.579825).abs() < 1e-4);
        assert!((incomplete_beta(1.0, 1.0, 0.0)).abs() < 1e-15);
        assert!((incomplete_beta(1.0, 1.0, 1.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_t_cdf_symmetry() {
        assert!((t_cdf(0.0, 10.0) - 0.5).abs() < 1e-6);
        assert!(t_cdf(5.0, 10.0) > 0.99);
        assert!(t_cdf(-5.0, 10.0) < 0.01);
    }

    #[test]
    fn test_normal_cdf_known_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!(normal_cdf(3.0) > 0.99);
        assert!(normal_cdf(-3.0) < 0.01);
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.001);
    }

    #[test]
    fn test_z_critical_roundtrip() {
        for &p in &[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let z = z_critical(p);
            let p2 = normal_cdf(z);
            assert!(
                (p - p2).abs() < 1e-4,
                "roundtrip failed for p={p}: got {p2}"
            );
        }
    }

    #[test]
    fn test_percentile_ci_basic() {
        let scores: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (lower, upper) = percentile_ci(&scores, 0.95);
        assert!(lower < 5.0);
        assert!(upper > 95.0);
    }
}

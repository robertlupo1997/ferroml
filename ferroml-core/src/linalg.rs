//! Shared linear algebra utilities
//!
//! Consolidates common linear algebra operations (QR decomposition, triangular
//! solvers, etc.) used across multiple modules, with an optional `faer` backend
//! for high-performance dense operations.

use crate::{FerroError, Result};
use ndarray::{Array1, Array2};

// =============================================================================
// QR Decomposition
// =============================================================================

/// QR decomposition using Modified Gram-Schmidt (MGS).
///
/// Returns `(Q, R)` where `Q` is `(m, k)` orthogonal and `R` is `(k, n)` upper
/// triangular, with `k = min(m, n)`.
///
/// MGS is more numerically stable than classical Gram-Schmidt.
///
/// When the `faer-backend` feature is enabled, delegates to faer's QR which
/// uses Householder reflectors (numerically superior on large matrices).
pub fn qr_decomposition(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
    #[cfg(feature = "faer-backend")]
    {
        qr_decomposition_faer(a)
    }
    #[cfg(not(feature = "faer-backend"))]
    {
        qr_decomposition_mgs(a)
    }
}

/// Modified Gram-Schmidt QR decomposition (pure Rust fallback).
pub fn qr_decomposition_mgs(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
    let (m, n) = a.dim();
    let k = m.min(n);

    let mut q = Array2::zeros((m, k));
    let mut r = Array2::zeros((k, n));

    for j in 0..k {
        let mut v = a.column(j).to_owned();

        // Subtract projections onto previous q vectors (Modified Gram-Schmidt)
        for i in 0..j {
            let qi = q.column(i);
            let proj: f64 = qi.dot(&v);
            r[[i, j]] = proj;
            v -= &(&qi * proj);
        }

        let norm: f64 = v.dot(&v).sqrt();
        if norm > 1e-14 {
            r[[j, j]] = norm;
            q.column_mut(j).assign(&(&v / norm));
        } else {
            r[[j, j]] = 0.0;
            q.column_mut(j).fill(0.0);
        }
    }

    // Fill remaining R columns for wide matrices
    for j in k..n {
        for i in 0..k {
            r[[i, j]] = q.column(i).dot(&a.column(j));
        }
    }

    Ok((q, r))
}

/// QR decomposition via faer (Householder reflectors, high performance).
#[cfg(feature = "faer-backend")]
pub fn qr_decomposition_faer(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
    let (m, n) = a.dim();
    let k = m.min(n);

    // Convert ndarray -> faer Mat
    let mut mat = faer::Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            mat.write(i, j, a[[i, j]]);
        }
    }

    let qr = mat.qr();
    let q_faer = qr.compute_thin_q();
    let r_faer = qr.compute_thin_r();

    // Convert faer Mat -> ndarray
    let mut q = Array2::zeros((m, k));
    for i in 0..m {
        for j in 0..k {
            q[[i, j]] = q_faer.read(i, j);
        }
    }

    let mut r = Array2::zeros((k, n));
    let r_rows = r_faer.nrows().min(k);
    let r_cols = r_faer.ncols().min(n);
    for i in 0..r_rows {
        for j in 0..r_cols {
            r[[i, j]] = r_faer.read(i, j);
        }
    }

    Ok((q, r))
}

// =============================================================================
// Triangular Solvers
// =============================================================================

/// Solve upper triangular system `Rx = b` by back-substitution.
pub fn solve_upper_triangular(r: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = r.nrows();
    let mut x = Array1::zeros(n);

    for i in (0..n).rev() {
        if r[[i, i]].abs() < 1e-14 {
            return Err(FerroError::numerical(
                "Singular matrix in back-substitution",
            ));
        }

        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= r[[i, j]] * x[j];
        }
        x[i] = sum / r[[i, i]];
    }

    Ok(x)
}

/// Invert an upper triangular matrix.
pub fn invert_upper_triangular(r: &Array2<f64>) -> Result<Array2<f64>> {
    let n = r.nrows();
    let mut inv = Array2::zeros((n, n));

    for i in 0..n {
        if r[[i, i]].abs() < 1e-14 {
            return Err(FerroError::numerical("Singular matrix in inversion"));
        }
        inv[[i, i]] = 1.0 / r[[i, i]];

        for j in (i + 1)..n {
            let mut sum = 0.0;
            for k in i..j {
                sum -= r[[k, j]] * inv[[i, k]];
            }
            inv[[i, j]] = sum / r[[j, j]];
        }
    }

    Ok(inv)
}

// =============================================================================
// Distance Computations (with optional SIMD)
// =============================================================================

/// Squared Euclidean distance between two slices.
///
/// When the `simd` feature is enabled, uses SIMD-accelerated computation.
#[inline]
pub fn squared_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(feature = "simd")]
    {
        crate::simd::squared_euclidean_distance(a, b)
    }
    #[cfg(not(feature = "simd"))]
    {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
    }
}

/// Dot product of two slices.
///
/// When the `simd` feature is enabled, uses SIMD-accelerated computation.
#[inline]
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(feature = "simd")]
    {
        crate::simd::dot_product(a, b)
    }
    #[cfg(not(feature = "simd"))]
    {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_qr_decomposition_square() {
        let a = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
            .unwrap();

        let (q, r) = qr_decomposition(&a).unwrap();

        // Q should be orthogonal: Q^T Q = I
        let qtq = q.t().dot(&q);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(qtq[[i, j]], expected, epsilon = 1e-10);
            }
        }

        // QR should reconstruct A
        let qr_product = q.dot(&r);
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(qr_product[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_qr_decomposition_tall() {
        let a =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let (q, r) = qr_decomposition(&a).unwrap();
        assert_eq!(q.dim(), (4, 2));
        assert_eq!(r.dim(), (2, 2));

        // Q^T Q = I (thin Q)
        let qtq = q.t().dot(&q);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(qtq[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_solve_upper_triangular() {
        let r = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 0.0, 3.0]).unwrap();
        let b = Array1::from_vec(vec![5.0, 6.0]);

        let x = solve_upper_triangular(&r, &b).unwrap();
        // 3x[1] = 6 => x[1] = 2
        // 2x[0] + 1*2 = 5 => x[0] = 1.5
        assert_abs_diff_eq!(x[0], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_squared_euclidean_distance() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // (3^2 + 3^2 + 3^2) = 27
        assert_abs_diff_eq!(squared_euclidean_distance(&a, &b), 27.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // 4 + 10 + 18 = 32
        assert_abs_diff_eq!(dot_product(&a, &b), 32.0, epsilon = 1e-10);
    }
}

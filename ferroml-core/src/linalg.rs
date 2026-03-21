//! Shared linear algebra utilities
//!
//! Consolidates common linear algebra operations (QR decomposition, triangular
//! solvers, etc.) used across multiple modules, with an optional `faer` backend
//! for high-performance dense operations.

use crate::{FerroError, Result};
use ndarray::{Array1, Array2};

// =============================================================================
// SVD (Singular Value Decomposition)
// =============================================================================

/// Thin SVD: returns `(U, S, Vt)` where `U` is `(m, k)`, `S` is `(k,)`, `Vt` is `(k, n)`,
/// with `k = min(m, n)`.
///
/// When the `faer-backend` feature is enabled, delegates to faer's SVD which
/// uses a divide-and-conquer algorithm (10-13x faster than nalgebra's Jacobi SVD).
pub fn thin_svd(a: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    #[cfg(feature = "faer-backend")]
    {
        thin_svd_faer(a)
    }
    #[cfg(not(feature = "faer-backend"))]
    {
        thin_svd_nalgebra(a)
    }
}

/// Thin SVD via faer (divide-and-conquer, high performance).
#[cfg(feature = "faer-backend")]
pub fn thin_svd_faer(a: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let (m, n) = a.dim();
    let k = m.min(n);

    // Convert ndarray -> faer Mat
    let mut mat = faer::Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            mat.write(i, j, a[[i, j]]);
        }
    }

    let svd = mat.thin_svd();

    // Extract U (m × k)
    let u_faer = svd.u();
    let mut u = Array2::zeros((m, k));
    for i in 0..m {
        for j in 0..k {
            u[[i, j]] = u_faer.read(i, j);
        }
    }

    // Extract singular values S (k,)
    let s_faer = svd.s_diagonal();
    let mut s = Array1::zeros(k);
    for i in 0..k {
        s[i] = s_faer.read(i);
    }

    // Extract V (n × k) and transpose to Vt (k × n)
    let v_faer = svd.v();
    let mut vt = Array2::zeros((k, n));
    for i in 0..k {
        for j in 0..n {
            vt[[i, j]] = v_faer.read(j, i);
        }
    }

    svd_flip(&mut u, &mut vt);

    Ok((u, s, vt))
}

/// Thin SVD via nalgebra (Jacobi, pure Rust fallback).
pub fn thin_svd_nalgebra(a: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let (m, n) = a.dim();
    let mat = nalgebra::DMatrix::from_fn(m, n, |i, j| a[[i, j]]);
    let svd = mat.svd(true, true);

    let u_nal = svd
        .u
        .ok_or_else(|| FerroError::numerical("SVD failed to compute U matrix"))?;
    let s_nal = svd.singular_values;
    let vt_nal = svd
        .v_t
        .ok_or_else(|| FerroError::numerical("SVD failed to compute V^T matrix"))?;

    let k = m.min(n);
    let mut u = Array2::from_shape_fn((u_nal.nrows(), u_nal.ncols().min(k)), |(i, j)| {
        u_nal[(i, j)]
    });
    let s = Array1::from_iter(s_nal.iter().take(k).copied());
    let mut vt = Array2::from_shape_fn((vt_nal.nrows().min(k), vt_nal.ncols()), |(i, j)| {
        vt_nal[(i, j)]
    });

    svd_flip(&mut u, &mut vt);

    Ok((u, s, vt))
}

// =============================================================================
// Symmetric Eigendecomposition
// =============================================================================

/// Symmetric eigendecomposition: decompose symmetric matrix `A = V Λ V^T`.
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvalues in **descending** order
/// (largest first, PCA convention) and eigenvectors as columns of V.
///
/// When the `faer-backend` feature is enabled, delegates to faer's divide-and-conquer
/// eigendecomposition (significantly faster than nalgebra on large matrices).
pub fn symmetric_eigh(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    #[cfg(feature = "faer-backend")]
    {
        symmetric_eigh_faer(a)
    }
    #[cfg(not(feature = "faer-backend"))]
    {
        symmetric_eigh_nalgebra(a)
    }
}

/// Symmetric eigendecomposition via faer (divide-and-conquer, high performance).
#[cfg(feature = "faer-backend")]
pub fn symmetric_eigh_faer(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(FerroError::shape_mismatch(
            format!("({0}, {0})", n),
            format!("({}, {})", n, a.ncols()),
        ));
    }

    // Convert ndarray -> faer Mat
    let mut mat = faer::Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            mat.write(i, j, a[[i, j]]);
        }
    }

    let eig = mat.selfadjoint_eigendecomposition(faer::Side::Lower);
    let s = eig.s();
    let u = eig.u();

    // Extract eigenvalues (ascending in faer) and reverse to descending
    let s_col = s.column_vector();
    let mut eigenvalues = Array1::zeros(n);
    for i in 0..n {
        eigenvalues[i] = s_col.read(n - 1 - i);
    }

    // Extract eigenvectors as columns, reversed to match descending eigenvalues
    let mut eigenvectors = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            eigenvectors[[i, j]] = u.read(i, n - 1 - j);
        }
    }

    Ok((eigenvalues, eigenvectors))
}

/// Symmetric eigendecomposition via nalgebra (pure Rust fallback).
pub fn symmetric_eigh_nalgebra(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(FerroError::shape_mismatch(
            format!("({0}, {0})", n),
            format!("({}, {})", n, a.ncols()),
        ));
    }

    let mat = nalgebra::DMatrix::from_fn(n, n, |i, j| a[[i, j]]);
    let eig = mat.symmetric_eigen();

    // nalgebra returns eigenvalues in arbitrary order — sort descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut eigenvalues = Array1::zeros(n);
    let mut eigenvectors = Array2::zeros((n, n));
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        eigenvalues[new_idx] = eig.eigenvalues[old_idx];
        for i in 0..n {
            eigenvectors[[i, new_idx]] = eig.eigenvectors[(i, old_idx)];
        }
    }

    Ok((eigenvalues, eigenvectors))
}

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
// Cholesky Decomposition
// =============================================================================

/// Cholesky decomposition: compute lower triangular `L` such that `A = L * L^T`.
///
/// The input matrix must be symmetric positive definite. A small diagonal
/// regularization (`reg`) can be added to improve numerical stability.
///
/// When the `faer-backend` feature is enabled, delegates to faer's Cholesky
/// for better performance on large matrices.
///
/// # Arguments
/// * `a` - Symmetric positive-definite matrix of shape `(n, n)`
/// * `reg` - Diagonal regularization (added to diagonal before decomposition)
///
/// # Returns
/// Lower triangular matrix `L` of shape `(n, n)`
pub fn cholesky(a: &Array2<f64>, reg: f64) -> Result<Array2<f64>> {
    #[cfg(feature = "faer-backend")]
    {
        cholesky_faer(a, reg)
    }
    #[cfg(not(feature = "faer-backend"))]
    {
        cholesky_native(a, reg)
    }
}

/// Cholesky decomposition via faer (high performance).
#[cfg(feature = "faer-backend")]
pub fn cholesky_faer(a: &Array2<f64>, reg: f64) -> Result<Array2<f64>> {
    let (n, m) = a.dim();
    if n != m {
        return Err(FerroError::shape_mismatch(
            format!("({0}, {0})", n),
            format!("({}, {})", n, m),
        ));
    }

    // Convert ndarray -> faer Mat, applying regularization
    let mut mat = faer::Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            mat.write(i, j, a[[i, j]]);
        }
        mat.write(i, i, a[[i, i]] + reg);
    }

    // Compute Cholesky (lower triangular)
    let chol = mat.cholesky(faer::Side::Lower).map_err(|_| {
        FerroError::numerical(
            "Matrix not positive definite for Cholesky decomposition. \
             Try increasing reg_covar.",
        )
    })?;

    // Extract L factor and convert faer Mat -> ndarray
    let l_faer = chol.compute_l();
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            l[[i, j]] = l_faer.read(i, j);
        }
    }

    Ok(l)
}

/// Cholesky decomposition (pure Rust fallback).
pub fn cholesky_native(a: &Array2<f64>, reg: f64) -> Result<Array2<f64>> {
    let (n, m) = a.dim();
    if n != m {
        return Err(FerroError::shape_mismatch(
            format!("({0}, {0})", n),
            format!("({}, {})", n, m),
        ));
    }

    let mut l: Array2<f64> = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum: f64 = 0.0;

            if j == i {
                for k in 0..j {
                    sum = l[[j, k]].mul_add(l[[j, k]], sum);
                }
                let val = a[[i, i]] + reg - sum;
                if val <= 0.0 {
                    return Err(FerroError::numerical(
                        "Matrix not positive definite for Cholesky decomposition. \
                         Try increasing reg_covar.",
                    ));
                }
                l[[i, j]] = val.sqrt();
            } else {
                for k in 0..j {
                    sum = l[[i, k]].mul_add(l[[j, k]], sum);
                }
                if l[[j, j]].abs() < 1e-15 {
                    return Err(FerroError::numerical(
                        "Near-zero diagonal in Cholesky decomposition",
                    ));
                }
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Compute log-determinant from a Cholesky factor L.
///
/// Since `det(A) = det(L)^2` and `det(L) = prod(diag(L))`,
/// `log(det(A)) = 2 * sum(log(diag(L)))`.
pub fn log_determinant_from_cholesky(l: &Array2<f64>) -> f64 {
    let n = l.nrows();
    let mut log_det = 0.0;
    for i in 0..n {
        debug_assert!(l[[i, i]] > 0.0, "Cholesky diagonal must be positive");
        log_det += l[[i, i]].ln();
    }
    2.0 * log_det
}

/// Solve `L * X = B` where `L` is lower triangular (forward substitution).
///
/// Solves column-by-column for multiple right-hand sides.
pub fn solve_lower_triangular(l: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let n = l.nrows();
    let ncols = b.ncols();
    let mut x = Array2::zeros((n, ncols));

    for col in 0..ncols {
        for i in 0..n {
            if l[[i, i]].abs() < 1e-15 {
                return Err(FerroError::numerical(
                    "Near-zero diagonal in forward substitution",
                ));
            }
            let mut sum = b[[i, col]];
            for j in 0..i {
                sum -= l[[i, j]] * x[[j, col]];
            }
            x[[i, col]] = sum / l[[i, i]];
        }
    }

    Ok(x)
}

/// Solve `L * x = b` where `L` is lower triangular (single right-hand side).
pub fn solve_lower_triangular_vec(l: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = l.nrows();
    let mut x = Array1::zeros(n);

    for i in 0..n {
        if l[[i, i]].abs() < 1e-15 {
            return Err(FerroError::numerical(
                "Near-zero diagonal in forward substitution",
            ));
        }
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }

    Ok(x)
}

// =============================================================================
// Distance Computations (with optional SIMD)
// =============================================================================

/// Euclidean distance between two slices.
///
/// When the `simd` feature is enabled, uses SIMD-accelerated computation.
#[inline]
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    squared_euclidean_distance(a, b).sqrt()
}

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

// =============================================================================
// Numerical Stability Utilities
// =============================================================================

/// Compute `log(sum(exp(x)))` in a numerically stable way.
///
/// Uses the max-subtract trick: `log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))`.
/// This prevents overflow when values are large and underflow when values are small.
///
/// Returns `f64::NEG_INFINITY` for empty slices or when all values are `-inf`.
pub fn logsumexp(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum_exp: f64 = x.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

/// Row-wise log-sum-exp on a 2D array.
///
/// For each row, computes `log(sum(exp(row)))` using the numerically stable
/// max-subtract trick. Returns a 1D array of length `n_rows`.
pub fn logsumexp_rows(log_probs: &Array2<f64>) -> Array1<f64> {
    let n_rows = log_probs.nrows();
    let mut result = Array1::zeros(n_rows);
    for i in 0..n_rows {
        let row = log_probs.row(i);
        result[i] = logsumexp(row.as_slice().unwrap_or(&row.to_vec()));
    }
    result
}

/// Cholesky decomposition with automatic jitter retry for ill-conditioned matrices.
///
/// Tries `cholesky(a, reg)` first. On failure, retries with increasing diagonal
/// jitter values `[1e-10, 1e-8, 1e-6, 1e-4]`. Returns an error if even the
/// largest jitter value fails.
///
/// This is the standard approach used by GP implementations (GPy, GPflow, sklearn)
/// to handle near-singular kernel matrices.
pub fn cholesky_with_jitter(a: &Array2<f64>, reg: f64) -> Result<Array2<f64>> {
    // First attempt with the specified regularization
    match cholesky(a, reg) {
        Ok(l) => return Ok(l),
        Err(_) => {} // Fall through to jitter retry
    }

    // Retry with increasing jitter
    let jitter_values = [1e-10, 1e-8, 1e-6, 1e-4];
    for &jitter in &jitter_values {
        // Jitter is added on top of the existing regularization
        match cholesky(a, reg + jitter) {
            Ok(l) => {
                // Note: when the `log` crate is available, this would use log::warn!
                // For now, the jitter is applied silently. Callers can detect jitter
                // was needed by comparing reg with the effective regularization.
                return Ok(l);
            }
            Err(_) => continue,
        }
    }

    Err(FerroError::numerical(
        "Cholesky decomposition failed even with maximum jitter (1e-4). \
         The matrix may be severely ill-conditioned or not positive semi-definite.",
    ))
}

/// Normalize SVD component signs for deterministic results.
///
/// For each row of Vt, finds the column with the maximum absolute value.
/// If that value is negative, flips the signs of that Vt row and the
/// corresponding U column. This matches sklearn's `svd_flip` convention.
///
/// This ensures consistent signs regardless of the SVD backend (nalgebra, faer),
/// which is important for PCA, LDA, FactorAnalysis, and TruncatedSVD.
pub fn svd_flip(u: &mut Array2<f64>, vt: &mut Array2<f64>) {
    let k = vt.nrows();
    for i in 0..k {
        let row = vt.row(i);
        // Find column with maximum absolute value
        let max_abs_col = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // If the max-abs value is negative, flip signs
        if vt[[i, max_abs_col]] < 0.0 {
            for j in 0..vt.ncols() {
                vt[[i, j]] = -vt[[i, j]];
            }
            // Flip corresponding U column
            if i < u.ncols() {
                for r in 0..u.nrows() {
                    u[[r, i]] = -u[[r, i]];
                }
            }
        }
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
    fn test_cholesky_2x2() {
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let l = cholesky(&a, 0.0).unwrap();

        // Verify L * L^T = A
        let reconstructed = l.dot(&l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }

        // L should be lower triangular
        assert_abs_diff_eq!(l[[0, 1]], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_cholesky_3x3() {
        // SPD matrix: A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        let a = Array2::from_shape_vec(
            (3, 3),
            vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        )
        .unwrap();
        let l = cholesky(&a, 0.0).unwrap();
        let reconstructed = l.dot(&l.t());
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_cholesky_with_regularization() {
        // Nearly singular matrix — regularization saves it
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 0.9999, 0.9999, 1.0]).unwrap();
        assert!(cholesky(&a, 0.0).is_ok());
        let l = cholesky(&a, 1e-6).unwrap();
        let reconstructed = l.dot(&l.t());
        // Should be close to A + reg*I
        assert_abs_diff_eq!(reconstructed[[0, 0]], 1.0 + 1e-6, epsilon = 1e-10);
    }

    #[test]
    fn test_cholesky_non_spd_fails() {
        // Not positive definite
        let a = Array2::from_shape_vec((2, 2), vec![-1.0, 0.0, 0.0, 1.0]).unwrap();
        assert!(cholesky(&a, 0.0).is_err());
    }

    #[test]
    fn test_log_determinant_from_cholesky() {
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let l = cholesky(&a, 0.0).unwrap();
        let log_det = log_determinant_from_cholesky(&l);
        // det(A) = 4*3 - 2*2 = 8, log(8) ≈ 2.0794
        assert_abs_diff_eq!(log_det, 8.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_solve_lower_triangular_identity() {
        let l = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = solve_lower_triangular(&l, &b).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(x[[i, j]], b[[i, j]], epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_solve_lower_triangular_simple() {
        let l = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0]).unwrap();
        let b = Array2::from_shape_vec((2, 1), vec![4.0, 7.0]).unwrap();
        let x = solve_lower_triangular(&l, &b).unwrap();
        // 2*x0 = 4 => x0 = 2
        // 1*2 + 3*x1 = 7 => x1 = 5/3
        assert_abs_diff_eq!(x[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[[1, 0]], 5.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_lower_triangular_vec_simple() {
        let l = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0]).unwrap();
        let b = Array1::from_vec(vec![4.0, 7.0]);
        let x = solve_lower_triangular_vec(&l, &b).unwrap();
        assert_abs_diff_eq!(x[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 5.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_thin_svd_square() {
        let a = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
            .unwrap();

        let (u, s, vt) = thin_svd(&a).unwrap();
        assert_eq!(u.dim(), (3, 3));
        assert_eq!(s.len(), 3);
        assert_eq!(vt.dim(), (3, 3));

        // U * diag(S) * Vt should reconstruct A
        let mut reconstructed: Array2<f64> = Array2::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    reconstructed[[i, j]] += u[[i, k]] * s[k] * vt[[k, j]];
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }

        // Singular values should be non-negative
        for &sv in s.iter() {
            assert!(sv >= 0.0);
        }
    }

    #[test]
    fn test_thin_svd_tall() {
        let a =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let (u, s, vt) = thin_svd(&a).unwrap();
        assert_eq!(u.dim(), (4, 2));
        assert_eq!(s.len(), 2);
        assert_eq!(vt.dim(), (2, 2));

        // Reconstruct
        let mut reconstructed: Array2<f64> = Array2::zeros((4, 2));
        for i in 0..4 {
            for j in 0..2 {
                for k in 0..2 {
                    reconstructed[[i, j]] += u[[i, k]] * s[k] * vt[[k, j]];
                }
            }
        }
        for i in 0..4 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
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

    // =========================================================================
    // logsumexp tests
    // =========================================================================

    #[test]
    fn test_logsumexp_basic() {
        // log(exp(1) + exp(2) + exp(3))
        let x = [1.0, 2.0, 3.0];
        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert_abs_diff_eq!(logsumexp(&x), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_logsumexp_all_negative() {
        let x = [-5.0, -10.0, -3.0];
        let expected = ((-5.0_f64).exp() + (-10.0_f64).exp() + (-3.0_f64).exp()).ln();
        assert_abs_diff_eq!(logsumexp(&x), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_logsumexp_single_element() {
        assert_abs_diff_eq!(logsumexp(&[5.0]), 5.0, epsilon = 1e-14);
        assert_abs_diff_eq!(logsumexp(&[-100.0]), -100.0, epsilon = 1e-14);
    }

    #[test]
    fn test_logsumexp_empty() {
        assert_eq!(logsumexp(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn test_logsumexp_neg_infinity() {
        assert_eq!(
            logsumexp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            f64::NEG_INFINITY
        );
        // Mix of -inf and finite
        assert_abs_diff_eq!(logsumexp(&[f64::NEG_INFINITY, 0.0]), 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_logsumexp_large_values() {
        // Should not overflow even with large values
        let x = [1000.0, 1001.0, 1002.0];
        let result = logsumexp(&x);
        assert!(result.is_finite());
        // result should be close to 1002 + log(exp(-2) + exp(-1) + 1)
        let expected = 1002.0 + ((-2.0_f64).exp() + (-1.0_f64).exp() + 1.0_f64).ln();
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_logsumexp_rows_basic() {
        let log_probs =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]).unwrap();
        let result = logsumexp_rows(&log_probs);
        assert_eq!(result.len(), 2);
        assert_abs_diff_eq!(result[0], logsumexp(&[1.0, 2.0, 3.0]), epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], logsumexp(&[-1.0, -2.0, -3.0]), epsilon = 1e-10);
    }

    // =========================================================================
    // cholesky_with_jitter tests
    // =========================================================================

    #[test]
    fn test_cholesky_jitter_well_conditioned() {
        // Well-conditioned SPD matrix should succeed without jitter
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let l = cholesky_with_jitter(&a, 0.0).unwrap();
        let reconstructed = l.dot(&l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_cholesky_jitter_ill_conditioned() {
        // Nearly singular matrix -- standard cholesky(0.0) might fail,
        // but jitter should save it
        let a = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0,
                1.0 - 1e-15,
                1.0 - 2e-15,
                1.0 - 1e-15,
                1.0,
                1.0 - 1e-15,
                1.0 - 2e-15,
                1.0 - 1e-15,
                1.0,
            ],
        )
        .unwrap();
        // This should succeed with jitter
        let result = cholesky_with_jitter(&a, 0.0);
        assert!(
            result.is_ok(),
            "cholesky_with_jitter should succeed on ill-conditioned matrix"
        );
    }

    #[test]
    fn test_cholesky_jitter_truly_singular() {
        // Not positive semi-definite at all -- should fail even with jitter
        let a = Array2::from_shape_vec((2, 2), vec![-10.0, 0.0, 0.0, -10.0]).unwrap();
        assert!(cholesky_with_jitter(&a, 0.0).is_err());
    }

    // =========================================================================
    // svd_flip tests
    // =========================================================================

    #[test]
    fn test_svd_flip_deterministic_signs() {
        // After svd_flip, the component with max absolute value in each Vt row
        // should be positive.
        let a = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        let (_, _, vt) = thin_svd(&a).unwrap();

        // For each row of Vt, the element with max absolute value should be positive
        for i in 0..vt.nrows() {
            let row = vt.row(i);
            let max_abs_col = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.abs()
                        .partial_cmp(&b.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap();
            assert!(
                vt[[i, max_abs_col]] >= 0.0,
                "Row {i}: max-abs element should be non-negative after svd_flip"
            );
        }
    }

    #[test]
    fn test_svd_flip_reconstruction() {
        // SVD should still reconstruct the original matrix after flip
        let a = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
            .unwrap();

        let (u, s, vt) = thin_svd(&a).unwrap();

        // Reconstruct: U * diag(S) * Vt
        let mut reconstructed: Array2<f64> = Array2::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    reconstructed[[i, j]] += u[[i, k]] * s[k] * vt[[k, j]];
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_svd_flip_tall_matrix() {
        let a = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();

        let (u, s, vt) = thin_svd(&a).unwrap();
        assert_eq!(u.dim(), (5, 2));
        assert_eq!(vt.dim(), (2, 2));

        // Reconstruction check
        let mut reconstructed: Array2<f64> = Array2::zeros((5, 2));
        for i in 0..5 {
            for j in 0..2 {
                for k in 0..2 {
                    reconstructed[[i, j]] += u[[i, k]] * s[k] * vt[[k, j]];
                }
            }
        }
        for i in 0..5 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }
}

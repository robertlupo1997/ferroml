//! Centralized input validation for unsupervised models.
//!
//! Provides shared validation functions for clustering and decomposition models,
//! ensuring consistent NaN/Inf/empty rejection with actionable error messages.

use crate::{FerroError, Result};
use ndarray::Array2;

/// Validate input for unsupervised models (clustering, decomposition).
///
/// Checks:
/// - Input is not empty (0 rows)
/// - Input contains no NaN or Inf values
///
/// # Errors
///
/// Returns `FerroError::InvalidInput` with a descriptive message including
/// the count and first position of offending values.
pub fn validate_unsupervised_input(x: &Array2<f64>) -> Result<()> {
    if x.nrows() == 0 || x.is_empty() {
        return Err(FerroError::invalid_input(
            "Input array must have at least one sample (got 0 rows)",
        ));
    }

    check_finite_detailed(x)?;

    Ok(())
}

/// Validate input for transform operations on unsupervised models.
///
/// Checks:
/// - Input is not empty (0 rows)
/// - Feature count matches the fitted model
/// - Input contains no NaN or Inf values
///
/// # Errors
///
/// Returns `FerroError::InvalidInput` or `FerroError::ShapeMismatch`
/// with descriptive messages.
pub fn validate_transform_input(x: &Array2<f64>, expected_features: usize) -> Result<()> {
    if x.nrows() == 0 || x.is_empty() {
        return Err(FerroError::invalid_input(
            "Input array must have at least one sample (got 0 rows)",
        ));
    }

    let actual_features = x.ncols();
    if actual_features != expected_features {
        return Err(FerroError::shape_mismatch(
            format!("({}, {})", x.nrows(), expected_features),
            format!("({}, {})", x.nrows(), actual_features),
        ));
    }

    check_finite_detailed(x)?;

    Ok(())
}

/// Check that all values in the array are finite (not NaN or Inf).
///
/// Provides detailed error messages including the count of non-finite values
/// and the position of the first offending value.
fn check_finite_detailed(x: &Array2<f64>) -> Result<()> {
    let mut count = 0usize;
    let mut first_row = 0usize;
    let mut first_col = 0usize;
    let mut found_first = false;

    for ((row, col), &val) in x.indexed_iter() {
        if !val.is_finite() {
            count += 1;
            if !found_first {
                first_row = row;
                first_col = col;
                found_first = true;
            }
        }
    }

    if count > 0 {
        return Err(FerroError::invalid_input(format!(
            "X contains {} NaN/Inf values (first at row {}, column {})",
            count, first_row, first_col
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_validate_unsupervised_empty() {
        let x = Array2::<f64>::zeros((0, 3));
        let result = validate_unsupervised_input(&x);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("0 rows"));
    }

    #[test]
    fn test_validate_unsupervised_nan() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();
        let result = validate_unsupervised_input(&x);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("1 NaN/Inf"), "msg: {msg}");
        assert!(msg.contains("row 1"), "msg: {msg}");
        assert!(msg.contains("column 0"), "msg: {msg}");
    }

    #[test]
    fn test_validate_unsupervised_inf() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, f64::INFINITY, 3.0, 4.0]).unwrap();
        let result = validate_unsupervised_input(&x);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("NaN/Inf"), "msg: {msg}");
    }

    #[test]
    fn test_validate_unsupervised_valid() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(validate_unsupervised_input(&x).is_ok());
    }

    #[test]
    fn test_validate_transform_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = validate_transform_input(&x, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_transform_valid() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(validate_transform_input(&x, 2).is_ok());
    }

    #[test]
    fn test_validate_multiple_nan_inf() {
        let x = Array2::from_shape_vec(
            (3, 2),
            vec![
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                4.0,
                f64::NAN,
                6.0,
            ],
        )
        .unwrap();
        let result = validate_unsupervised_input(&x);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("4 NaN/Inf"), "msg: {msg}");
        assert!(msg.contains("row 0, column 0"), "msg: {msg}");
    }
}

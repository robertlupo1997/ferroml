//! Isotonic Regression
//!
//! Fits a monotonically non-decreasing (or non-increasing) piecewise-linear function.
//! Uses the Pool Adjacent Violators Algorithm (PAVA).
//!
//! ## Example
//!
//! ```
//! use ferroml_core::models::IsotonicRegression;
//! use ferroml_core::models::Model;
//! use ndarray::{array, Array2};
//!
//! let mut iso = IsotonicRegression::new();
//!
//! // x must be single-column
//! let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! let y = array![1.0, 3.0, 2.0, 5.0, 4.0];
//!
//! iso.fit(&x, &y).unwrap();
//! let preds = iso.predict(&x).unwrap();
//! // Output is monotonically non-decreasing
//! for i in 1..preds.len() {
//!     assert!(preds[i] >= preds[i - 1] - 1e-10);
//! }
//! ```

use crate::hpo::SearchSpace;
use crate::preprocessing::{check_is_fitted, check_non_empty};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Whether the isotonic regression should be increasing, decreasing, or auto-detected.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Increasing {
    /// Monotonically non-decreasing
    True,
    /// Monotonically non-increasing
    False,
    /// Auto-detect using Spearman rank correlation
    Auto,
}

impl Default for Increasing {
    fn default() -> Self {
        Increasing::True
    }
}

/// How to handle predictions outside the training range.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OutOfBounds {
    /// Return NaN for out-of-bounds
    Nan,
    /// Clip to the nearest boundary value
    Clip,
    /// Return an error
    Raise,
}

impl Default for OutOfBounds {
    fn default() -> Self {
        OutOfBounds::Nan
    }
}

/// Isotonic Regression model.
///
/// Fits a monotonic piecewise-linear function using PAVA.
/// Input must be single-column (1D feature).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotonicRegression {
    // Config
    increasing: Increasing,
    y_min: Option<f64>,
    y_max: Option<f64>,
    out_of_bounds: OutOfBounds,

    // Fitted state
    x_min_: Option<f64>,
    x_max_: Option<f64>,
    x_thresholds_: Option<Vec<f64>>,
    y_thresholds_: Option<Vec<f64>>,
    increasing_: Option<bool>,
}

impl Default for IsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl IsotonicRegression {
    /// Create a new isotonic regression with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            increasing: Increasing::True,
            y_min: None,
            y_max: None,
            out_of_bounds: OutOfBounds::Nan,
            x_min_: None,
            x_max_: None,
            x_thresholds_: None,
            y_thresholds_: None,
            increasing_: None,
        }
    }

    /// Set monotonicity direction.
    #[must_use]
    pub fn with_increasing(mut self, increasing: Increasing) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set minimum output value.
    #[must_use]
    pub fn with_y_min(mut self, y_min: f64) -> Self {
        self.y_min = Some(y_min);
        self
    }

    /// Set maximum output value.
    #[must_use]
    pub fn with_y_max(mut self, y_max: f64) -> Self {
        self.y_max = Some(y_max);
        self
    }

    /// Set out-of-bounds handling strategy.
    #[must_use]
    pub fn with_out_of_bounds(mut self, oob: OutOfBounds) -> Self {
        self.out_of_bounds = oob;
        self
    }

    /// Get fitted x thresholds (knot positions).
    pub fn x_thresholds(&self) -> Option<&[f64]> {
        self.x_thresholds_.as_deref()
    }

    /// Get fitted y thresholds (knot values).
    pub fn y_thresholds(&self) -> Option<&[f64]> {
        self.y_thresholds_.as_deref()
    }

    /// Get the inferred increasing direction.
    pub fn increasing_inferred(&self) -> Option<bool> {
        self.increasing_
    }

    /// Pool Adjacent Violators Algorithm (PAVA).
    ///
    /// Produces a monotonically increasing sequence from weighted observations.
    fn pava(y: &[f64], weights: &[f64]) -> Vec<f64> {
        let n = y.len();
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![y[0]];
        }

        let mut result = y.to_vec();
        let mut block_weights = weights.to_vec();
        let mut block_start: Vec<usize> = (0..n).collect();
        let mut block_end: Vec<usize> = (0..n).collect();

        let mut changed = true;
        while changed {
            changed = false;
            let mut i = 0;
            while i < n - 1 {
                let next = block_end[i] + 1;
                if next >= n {
                    break;
                }
                if result[i] > result[next] {
                    let w_i = block_weights[i];
                    let w_next = block_weights[next];
                    let new_value = result[i].mul_add(w_i, result[next] * w_next) / (w_i + w_next);

                    result[i] = new_value;
                    result[next] = new_value;
                    block_weights[i] = w_i + w_next;
                    block_end[i] = block_end[next];

                    for j in block_start[i]..=block_end[i] {
                        result[j] = new_value;
                        block_start[j] = block_start[i];
                        block_end[j] = block_end[i];
                    }
                    changed = true;
                }
                i = block_end[i] + 1;
            }
        }

        result
    }

    /// Linear interpolation between knots.
    fn interpolate(x_query: f64, x_data: &[f64], y_data: &[f64]) -> f64 {
        if x_data.is_empty() || x_data.len() != y_data.len() {
            return f64::NAN;
        }
        if x_data.len() == 1 {
            return y_data[0];
        }
        if x_query <= x_data[0] {
            return y_data[0];
        }
        if x_query >= x_data[x_data.len() - 1] {
            return y_data[y_data.len() - 1];
        }

        // Binary search for interval
        let mut lo = 0;
        let mut hi = x_data.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if x_data[mid] <= x_query {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let x0 = x_data[lo];
        let x1 = x_data[hi];
        let y0 = y_data[lo];
        let y1 = y_data[hi];

        if (x1 - x0).abs() < 1e-15 {
            return y0;
        }

        y0 + (y1 - y0) * (x_query - x0) / (x1 - x0)
    }

    /// Compute Spearman rank correlation sign to auto-detect direction.
    fn spearman_sign(x: &[f64], y: &[f64]) -> bool {
        let n = x.len();
        if n < 2 {
            return true;
        }

        // Rank x
        let mut x_idx: Vec<usize> = (0..n).collect();
        x_idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
        let mut x_rank = vec![0.0; n];
        for (rank, &idx) in x_idx.iter().enumerate() {
            x_rank[idx] = rank as f64;
        }

        // Rank y
        let mut y_idx: Vec<usize> = (0..n).collect();
        y_idx.sort_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal));
        let mut y_rank = vec![0.0; n];
        for (rank, &idx) in y_idx.iter().enumerate() {
            y_rank[idx] = rank as f64;
        }

        // Correlation sign from sum of d^2
        let d_sq_sum: f64 = x_rank
            .iter()
            .zip(y_rank.iter())
            .map(|(&xr, &yr)| (xr - yr) * (xr - yr))
            .sum();

        let n_f = n as f64;
        let rho = 1.0 - (6.0 * d_sq_sum) / (n_f * (n_f * n_f - 1.0));
        rho >= 0.0
    }

    /// Predict a single value, handling out-of-bounds.
    fn predict_single(&self, x_val: f64) -> Result<f64> {
        let x_data = self.x_thresholds_.as_ref().unwrap();
        let y_data = self.y_thresholds_.as_ref().unwrap();
        let x_min = self.x_min_.unwrap();
        let x_max = self.x_max_.unwrap();

        if x_val < x_min || x_val > x_max {
            match self.out_of_bounds {
                OutOfBounds::Nan => return Ok(f64::NAN),
                OutOfBounds::Clip => {
                    let clipped = x_val.clamp(x_min, x_max);
                    return Ok(Self::interpolate(clipped, x_data, y_data));
                }
                OutOfBounds::Raise => {
                    return Err(FerroError::invalid_input(format!(
                        "x value {} is outside training range [{}, {}]",
                        x_val, x_min, x_max
                    )));
                }
            }
        }

        Ok(Self::interpolate(x_val, x_data, y_data))
    }
}

impl super::Model for IsotonicRegression {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        check_non_empty(x)?;

        let (n_samples, n_features) = x.dim();

        if n_features != 1 {
            return Err(FerroError::invalid_input(format!(
                "IsotonicRegression requires exactly 1 feature, got {}",
                n_features
            )));
        }

        if y.len() != n_samples {
            return Err(FerroError::shape_mismatch(
                format!("({},)", n_samples),
                format!("({},)", y.len()),
            ));
        }

        // Extract 1D x values
        let x_vals: Vec<f64> = x.column(0).to_vec();
        let y_vals: Vec<f64> = y.to_vec();

        // Determine direction
        let is_increasing = match self.increasing {
            Increasing::True => true,
            Increasing::False => false,
            Increasing::Auto => Self::spearman_sign(&x_vals, &y_vals),
        };
        self.increasing_ = Some(is_increasing);

        // Sort by x
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&a, &b| {
            x_vals[a]
                .partial_cmp(&x_vals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let x_sorted: Vec<f64> = indices.iter().map(|&i| x_vals[i]).collect();
        let y_sorted: Vec<f64> = indices.iter().map(|&i| y_vals[i]).collect();
        let weights = vec![1.0; n_samples];

        // Apply PAVA
        let y_isotonic = if is_increasing {
            Self::pava(&y_sorted, &weights)
        } else {
            // For decreasing: reverse, apply PAVA, reverse
            let y_rev: Vec<f64> = y_sorted.iter().rev().copied().collect();
            let w_rev: Vec<f64> = weights.iter().rev().copied().collect();
            let mut iso_rev = Self::pava(&y_rev, &w_rev);
            iso_rev.reverse();
            iso_rev
        };

        // Apply y_min/y_max bounds
        let y_bounded: Vec<f64> = y_isotonic
            .iter()
            .map(|&v| {
                let mut val = v;
                if let Some(ymin) = self.y_min {
                    val = val.max(ymin);
                }
                if let Some(ymax) = self.y_max {
                    val = val.min(ymax);
                }
                val
            })
            .collect();

        // Deduplicate: average y for tied x values, keep unique knots
        let mut x_unique = vec![x_sorted[0]];
        let mut y_unique = vec![y_bounded[0]];
        let mut count = 1.0;

        for i in 1..n_samples {
            if (x_sorted[i] - *x_unique.last().unwrap()).abs() < 1e-10 {
                let last_idx = y_unique.len() - 1;
                y_unique[last_idx] =
                    y_unique[last_idx].mul_add(count, y_bounded[i]) / (count + 1.0);
                count += 1.0;
            } else {
                x_unique.push(x_sorted[i]);
                y_unique.push(y_bounded[i]);
                count = 1.0;
            }
        }

        self.x_min_ = Some(x_sorted[0]);
        self.x_max_ = Some(x_sorted[n_samples - 1]);
        self.x_thresholds_ = Some(x_unique);
        self.y_thresholds_ = Some(y_unique);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(self.is_fitted(), "predict")?;

        let (n_samples, n_features) = x.dim();
        if n_features != 1 {
            return Err(FerroError::invalid_input(format!(
                "IsotonicRegression requires exactly 1 feature, got {}",
                n_features
            )));
        }

        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            predictions[i] = self.predict_single(x[[i, 0]])?;
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.x_thresholds_.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        if self.is_fitted() {
            Some(1)
        } else {
            None
        }
    }

    fn model_name(&self) -> &str {
        "IsotonicRegression"
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;
        crate::metrics::r2_score(y, &predictions)
    }
}

impl crate::preprocessing::Transformer for IsotonicRegression {
    fn fit(&mut self, _x: &Array2<f64>) -> Result<()> {
        Err(FerroError::NotImplemented(
            "IsotonicRegression requires y for fitting; use Model::fit(x, y) instead".to_string(),
        ))
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        use crate::models::Model;
        let preds = self.predict(x)?;
        let n = preds.len();
        Ok(preds.into_shape_with_order((n, 1)).unwrap())
    }

    fn is_fitted(&self) -> bool {
        self.x_thresholds_.is_some()
    }

    fn get_feature_names_out(&self, _input_names: Option<&[String]>) -> Option<Vec<String>> {
        if self.is_fitted() {
            Some(vec!["isotonic".to_string()])
        } else {
            None
        }
    }

    fn n_features_in(&self) -> Option<usize> {
        if self.x_thresholds_.is_some() {
            Some(1)
        } else {
            None
        }
    }

    fn n_features_out(&self) -> Option<usize> {
        if self.x_thresholds_.is_some() {
            Some(1)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Model;
    use ndarray::{array, Array2};

    fn col(vals: Vec<f64>) -> Array2<f64> {
        let n = vals.len();
        Array2::from_shape_vec((n, 1), vals).unwrap()
    }

    #[test]
    fn test_monotonically_increasing() {
        let x = col(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = array![1.0, 3.0, 2.0, 5.0, 4.0];

        let mut iso = IsotonicRegression::new();
        iso.fit(&x, &y).unwrap();
        let preds = iso.predict(&x).unwrap();
        for i in 1..preds.len() {
            assert!(
                preds[i] >= preds[i - 1] - 1e-10,
                "not monotone at {}: {} < {}",
                i,
                preds[i],
                preds[i - 1]
            );
        }
    }

    #[test]
    fn test_monotonically_decreasing() {
        let x = col(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = array![5.0, 3.0, 4.0, 1.0, 2.0];

        let mut iso = IsotonicRegression::new().with_increasing(Increasing::False);
        iso.fit(&x, &y).unwrap();
        let preds = iso.predict(&x).unwrap();
        for i in 1..preds.len() {
            assert!(
                preds[i] <= preds[i - 1] + 1e-10,
                "not decreasing at {}: {} > {}",
                i,
                preds[i],
                preds[i - 1]
            );
        }
    }

    #[test]
    fn test_already_monotone_data() {
        let x = col(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut iso = IsotonicRegression::new();
        iso.fit(&x, &y).unwrap();
        let preds = iso.predict(&x).unwrap();
        for i in 0..5 {
            assert!(
                (preds[i] - y[i]).abs() < 1e-10,
                "expected {}, got {}",
                y[i],
                preds[i]
            );
        }
    }

    #[test]
    fn test_tied_x_values() {
        let x = col(vec![1.0, 1.0, 2.0, 2.0, 3.0]);
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let mut iso = IsotonicRegression::new();
        iso.fit(&x, &y).unwrap();
        let preds = iso.predict(&x).unwrap();
        // All predictions for same x should be equal
        assert!((preds[0] - preds[1]).abs() < 1e-10);
        assert!((preds[2] - preds[3]).abs() < 1e-10);
    }

    #[test]
    fn test_y_min_y_max() {
        let x = col(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = array![0.0, 10.0, 5.0, 20.0, 15.0];

        let mut iso = IsotonicRegression::new().with_y_min(2.0).with_y_max(12.0);
        iso.fit(&x, &y).unwrap();
        let preds = iso.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p >= 2.0 - 1e-10, "prediction {} below y_min", p);
            assert!(p <= 12.0 + 1e-10, "prediction {} above y_max", p);
        }
    }

    #[test]
    fn test_out_of_bounds_nan() {
        let x = col(vec![2.0, 3.0, 4.0]);
        let y = array![1.0, 2.0, 3.0];

        let mut iso = IsotonicRegression::new().with_out_of_bounds(OutOfBounds::Nan);
        iso.fit(&x, &y).unwrap();

        let x_test = col(vec![1.0, 5.0]);
        let preds = iso.predict(&x_test).unwrap();
        assert!(preds[0].is_nan(), "below-range should be NaN");
        assert!(preds[1].is_nan(), "above-range should be NaN");
    }

    #[test]
    fn test_out_of_bounds_clip() {
        let x = col(vec![2.0, 3.0, 4.0]);
        let y = array![1.0, 2.0, 3.0];

        let mut iso = IsotonicRegression::new().with_out_of_bounds(OutOfBounds::Clip);
        iso.fit(&x, &y).unwrap();

        let x_test = col(vec![1.0, 5.0]);
        let preds = iso.predict(&x_test).unwrap();
        assert!(
            (preds[0] - 1.0).abs() < 1e-10,
            "below-range should clip to y[0]"
        );
        assert!(
            (preds[1] - 3.0).abs() < 1e-10,
            "above-range should clip to y[-1]"
        );
    }

    #[test]
    fn test_out_of_bounds_raise() {
        let x = col(vec![2.0, 3.0, 4.0]);
        let y = array![1.0, 2.0, 3.0];

        let mut iso = IsotonicRegression::new().with_out_of_bounds(OutOfBounds::Raise);
        iso.fit(&x, &y).unwrap();

        let x_test = col(vec![1.0]);
        assert!(iso.predict(&x_test).is_err());
    }

    #[test]
    fn test_single_point() {
        let x = col(vec![1.0]);
        let y = array![5.0];

        let mut iso = IsotonicRegression::new().with_out_of_bounds(OutOfBounds::Clip);
        iso.fit(&x, &y).unwrap();
        let preds = iso.predict(&x).unwrap();
        assert!((preds[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_two_points() {
        let x = col(vec![1.0, 3.0]);
        let y = array![2.0, 6.0];

        let mut iso = IsotonicRegression::new().with_out_of_bounds(OutOfBounds::Clip);
        iso.fit(&x, &y).unwrap();
        // Interpolation at midpoint
        let x_test = col(vec![2.0]);
        let preds = iso.predict(&x_test).unwrap();
        assert!((preds[0] - 4.0).abs() < 1e-10, "midpoint should be 4.0");
    }

    #[test]
    fn test_multi_column_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        let mut iso = IsotonicRegression::new();
        assert!(iso.fit(&x, &y).is_err());
    }

    #[test]
    fn test_interpolation_between_knots() {
        let x = col(vec![0.0, 1.0, 2.0, 3.0]);
        let y = array![0.0, 1.0, 2.0, 3.0];

        let mut iso = IsotonicRegression::new();
        iso.fit(&x, &y).unwrap();

        let x_test = col(vec![0.5, 1.5, 2.5]);
        let preds = iso.predict(&x_test).unwrap();
        assert!((preds[0] - 0.5).abs() < 1e-10);
        assert!((preds[1] - 1.5).abs() < 1e-10);
        assert!((preds[2] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_not_fitted_error() {
        let iso = IsotonicRegression::new();
        let x = col(vec![1.0]);
        assert!(iso.predict(&x).is_err());
    }

    #[test]
    fn test_is_fitted() {
        let mut iso = IsotonicRegression::new();
        assert!(!iso.is_fitted());
        let x = col(vec![1.0, 2.0]);
        let y = array![1.0, 2.0];
        iso.fit(&x, &y).unwrap();
        assert!(iso.is_fitted());
    }

    #[test]
    fn test_n_features() {
        let mut iso = IsotonicRegression::new();
        assert_eq!(iso.n_features(), None);
        let x = col(vec![1.0, 2.0]);
        let y = array![1.0, 2.0];
        iso.fit(&x, &y).unwrap();
        assert_eq!(iso.n_features(), Some(1));
    }

    #[test]
    fn test_auto_increasing() {
        // Positive correlation → increasing
        let x = col(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = array![1.0, 3.0, 2.0, 5.0, 4.0];

        let mut iso = IsotonicRegression::new().with_increasing(Increasing::Auto);
        iso.fit(&x, &y).unwrap();
        assert_eq!(iso.increasing_inferred(), Some(true));
    }

    #[test]
    fn test_auto_decreasing() {
        // Negative correlation → decreasing
        let x = col(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = array![10.0, 8.0, 9.0, 3.0, 1.0];

        let mut iso = IsotonicRegression::new().with_increasing(Increasing::Auto);
        iso.fit(&x, &y).unwrap();
        assert_eq!(iso.increasing_inferred(), Some(false));
        let preds = iso.predict(&x).unwrap();
        for i in 1..preds.len() {
            assert!(preds[i] <= preds[i - 1] + 1e-10, "not decreasing at {}", i);
        }
    }

    #[test]
    fn test_model_name() {
        let iso = IsotonicRegression::new();
        assert_eq!(iso.model_name(), "IsotonicRegression");
    }

    #[test]
    fn test_xy_length_mismatch() {
        let x = col(vec![1.0, 2.0]);
        let y = array![1.0, 2.0, 3.0];
        let mut iso = IsotonicRegression::new();
        assert!(iso.fit(&x, &y).is_err());
    }

    #[test]
    fn test_x_thresholds_accessor() {
        let x = col(vec![1.0, 2.0, 3.0]);
        let y = array![1.0, 2.0, 3.0];
        let mut iso = IsotonicRegression::new();
        iso.fit(&x, &y).unwrap();
        let xt = iso.x_thresholds().unwrap();
        assert_eq!(xt.len(), 3);
        assert!((xt[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_transformer_trait() {
        use crate::preprocessing::Transformer;
        let x = col(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = array![1.0, 3.0, 2.0, 5.0, 4.0];

        let mut iso = IsotonicRegression::new();
        Model::fit(&mut iso, &x, &y).unwrap();
        let transformed = Transformer::transform(&iso, &x).unwrap();
        assert_eq!(transformed.dim(), (5, 1));
        // Should be same as predict but as 2D column
        let preds = Model::predict(&iso, &x).unwrap();
        for i in 0..5 {
            assert!((transformed[[i, 0]] - preds[i]).abs() < 1e-10);
        }
        assert!(Transformer::is_fitted(&iso));
        assert_eq!(Transformer::n_features_in(&iso), Some(1));
        assert_eq!(Transformer::n_features_out(&iso), Some(1));
    }

    #[test]
    fn test_reproducibility() {
        let x = col(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = array![1.0, 3.0, 2.0, 5.0, 4.0];

        let mut iso1 = IsotonicRegression::new();
        let mut iso2 = IsotonicRegression::new();
        iso1.fit(&x, &y).unwrap();
        iso2.fit(&x, &y).unwrap();
        let p1 = iso1.predict(&x).unwrap();
        let p2 = iso2.predict(&x).unwrap();
        for i in 0..5 {
            assert!((p1[i] - p2[i]).abs() < 1e-10);
        }
    }
}

//! Multi-output wrappers for single-output estimators.
//!
//! These wrappers fit one clone of the base estimator per output column,
//! enabling any single-output `Model` to handle multi-output tasks.
//!
//! # Example
//!
//! ```
//! use ferroml_core::models::{LinearRegression, MultiOutputRegressor};
//! use ndarray::{Array2, array};
//!
//! let x = Array2::from_shape_vec((4, 2), vec![1., 3., 2., 7., 5., 1., 4., 8.]).unwrap();
//! let y = Array2::from_shape_vec((4, 3), vec![
//!     1., 2., 4.,
//!     4., 5., 1.,
//!     7., 8., 3.,
//!     10., 11., 9.,
//! ]).unwrap();
//!
//! let mut mo = MultiOutputRegressor::new(LinearRegression::new());
//! mo.fit_multi(&x, &y).unwrap();
//! let preds = mo.predict_multi(&x).unwrap();
//! assert_eq!(preds.shape(), &[4, 3]);
//! ```

use crate::models::{validate_predict_input, Model};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};

/// Multi-output regressor: fits one regressor per target column.
///
/// Wraps any `Model` and trains it independently on each column of
/// a 2D target matrix, then stacks predictions column-wise.
#[derive(Debug, Clone)]
pub struct MultiOutputRegressor<M: Model + Clone> {
    base_estimator: M,
    estimators_: Option<Vec<M>>,
    n_outputs_: Option<usize>,
    n_features_: Option<usize>,
}

impl<M: Model + Clone> MultiOutputRegressor<M> {
    /// Create a new multi-output regressor wrapping the given base estimator.
    pub fn new(base_estimator: M) -> Self {
        Self {
            base_estimator,
            estimators_: None,
            n_outputs_: None,
            n_features_: None,
        }
    }

    /// Fit one estimator per target column.
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - Target matrix of shape (n_samples, n_outputs)
    ///
    /// # Errors
    /// Returns an error if `y` has zero columns or if any per-column fit fails.
    pub fn fit_multi(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
        let n_outputs = y.ncols();
        if n_outputs == 0 {
            return Err(FerroError::invalid_input(
                "y must have at least one output column",
            ));
        }

        // Validate X
        if x.is_empty() || x.nrows() == 0 {
            return Err(FerroError::invalid_input("Empty input data"));
        }
        if x.nrows() != y.nrows() {
            return Err(FerroError::shape_mismatch(
                format!("X has {} rows", x.nrows()),
                format!("y has {} rows", y.nrows()),
            ));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::invalid_input(
                "X contains NaN or infinite values",
            ));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::invalid_input(
                "y contains NaN or infinite values",
            ));
        }

        let mut estimators = Vec::with_capacity(n_outputs);
        for j in 0..n_outputs {
            let y_col: Array1<f64> = y.column(j).to_owned();
            let mut est = self.base_estimator.clone();
            est.fit(x, &y_col)?;
            estimators.push(est);
        }

        self.n_outputs_ = Some(n_outputs);
        self.n_features_ = Some(x.ncols());
        self.estimators_ = Some(estimators);
        Ok(())
    }

    /// Predict all outputs, returning an (n_samples, n_outputs) matrix.
    ///
    /// # Errors
    /// Returns `NotFitted` if called before `fit_multi`.
    pub fn predict_multi(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let estimators = self
            .estimators_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_multi"))?;

        if let Some(n_feat) = self.n_features_ {
            validate_predict_input(x, n_feat)?;
        }

        let n_samples = x.nrows();
        let n_outputs = estimators.len();
        let mut result = Array2::zeros((n_samples, n_outputs));

        for (j, est) in estimators.iter().enumerate() {
            let pred = est.predict(x)?;
            result.column_mut(j).assign(&pred);
        }

        Ok(result)
    }

    /// Whether the wrapper has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.estimators_.is_some()
    }

    /// Number of output columns (set after fitting).
    pub fn n_outputs(&self) -> Option<usize> {
        self.n_outputs_
    }

    /// Access the fitted per-output estimators.
    pub fn estimators(&self) -> Option<&[M]> {
        self.estimators_.as_deref()
    }
}

/// Multi-output classifier: fits one classifier per target column.
///
/// Wraps any `Model` and trains it independently on each column of
/// a 2D target matrix (multi-label or multi-output classification).
#[derive(Debug, Clone)]
pub struct MultiOutputClassifier<M: Model + Clone> {
    base_estimator: M,
    estimators_: Option<Vec<M>>,
    n_outputs_: Option<usize>,
    n_features_: Option<usize>,
}

impl<M: Model + Clone> MultiOutputClassifier<M> {
    /// Create a new multi-output classifier wrapping the given base estimator.
    pub fn new(base_estimator: M) -> Self {
        Self {
            base_estimator,
            estimators_: None,
            n_outputs_: None,
            n_features_: None,
        }
    }

    /// Fit one estimator per target column.
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - Target matrix of shape (n_samples, n_outputs)
    ///
    /// # Errors
    /// Returns an error if `y` has zero columns or if any per-column fit fails.
    pub fn fit_multi(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
        let n_outputs = y.ncols();
        if n_outputs == 0 {
            return Err(FerroError::invalid_input(
                "y must have at least one output column",
            ));
        }

        // Validate X
        if x.is_empty() || x.nrows() == 0 {
            return Err(FerroError::invalid_input("Empty input data"));
        }
        if x.nrows() != y.nrows() {
            return Err(FerroError::shape_mismatch(
                format!("X has {} rows", x.nrows()),
                format!("y has {} rows", y.nrows()),
            ));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::invalid_input(
                "X contains NaN or infinite values",
            ));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::invalid_input(
                "y contains NaN or infinite values",
            ));
        }

        let mut estimators = Vec::with_capacity(n_outputs);
        for j in 0..n_outputs {
            let y_col: Array1<f64> = y.column(j).to_owned();
            let mut est = self.base_estimator.clone();
            est.fit(x, &y_col)?;
            estimators.push(est);
        }

        self.n_outputs_ = Some(n_outputs);
        self.n_features_ = Some(x.ncols());
        self.estimators_ = Some(estimators);
        Ok(())
    }

    /// Predict all outputs, returning an (n_samples, n_outputs) matrix.
    ///
    /// # Errors
    /// Returns `NotFitted` if called before `fit_multi`.
    pub fn predict_multi(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let estimators = self
            .estimators_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_multi"))?;

        if let Some(n_feat) = self.n_features_ {
            validate_predict_input(x, n_feat)?;
        }

        let n_samples = x.nrows();
        let n_outputs = estimators.len();
        let mut result = Array2::zeros((n_samples, n_outputs));

        for (j, est) in estimators.iter().enumerate() {
            let pred = est.predict(x)?;
            result.column_mut(j).assign(&pred);
        }

        Ok(result)
    }

    /// Predict class probabilities for each output.
    ///
    /// Returns a `Vec` of `Array2`, one per output column. Each array has
    /// shape (n_samples, n_classes_for_that_output).
    ///
    /// # Errors
    /// Returns `NotFitted` if called before `fit_multi`, or `NotImplemented`
    /// if the base estimator does not support `try_predict_proba`.
    pub fn predict_proba_multi(&self, x: &Array2<f64>) -> Result<Vec<Array2<f64>>> {
        let estimators = self
            .estimators_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_proba_multi"))?;

        if let Some(n_feat) = self.n_features_ {
            validate_predict_input(x, n_feat)?;
        }

        let mut results = Vec::with_capacity(estimators.len());
        for est in estimators {
            match est.try_predict_proba(x) {
                Some(Ok(proba)) => results.push(proba),
                Some(Err(e)) => return Err(e),
                None => {
                    return Err(FerroError::NotImplemented(
                        "Base estimator does not support predict_proba".to_string(),
                    ))
                }
            }
        }
        Ok(results)
    }

    /// Whether the wrapper has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.estimators_.is_some()
    }

    /// Number of output columns (set after fitting).
    pub fn n_outputs(&self) -> Option<usize> {
        self.n_outputs_
    }

    /// Access the fitted per-output estimators.
    pub fn estimators(&self) -> Option<&[M]> {
        self.estimators_.as_deref()
    }
}

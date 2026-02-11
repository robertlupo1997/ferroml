//! Python bindings for FerroML Neural Network algorithms
//!
//! This module provides Python wrappers for:
//! - MLPClassifier: Multi-layer perceptron for classification
//! - MLPRegressor: Multi-layer perceptron for regression
//!
//! ## Zero-Copy Semantics
//!
//! Input arrays are passed as `PyReadonlyArray` which provides read-only access.
//! When calling core Rust functions that require owned arrays, a copy is made.
//! Output arrays use `into_pyarray` to transfer ownership to Python without copying data.

use crate::array_utils::{to_owned_array_1d, to_owned_array_2d};
use crate::pickle::{getstate, setstate};
use ferroml_core::neural::{
    Activation, EarlyStopping, MLPClassifier, MLPRegressor, NeuralDiagnostics, NeuralModel, Solver,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// =============================================================================
// Helper functions
// =============================================================================

fn parse_activation(activation: &str) -> PyResult<Activation> {
    match activation.to_lowercase().as_str() {
        "relu" => Ok(Activation::ReLU),
        "sigmoid" | "logistic" => Ok(Activation::Sigmoid),
        "tanh" => Ok(Activation::Tanh),
        "softmax" => Ok(Activation::Softmax),
        "linear" | "identity" => Ok(Activation::Linear),
        "leaky_relu" | "leakyrelu" => Ok(Activation::LeakyReLU),
        "elu" => Ok(Activation::ELU),
        _ => Err(PyValueError::new_err(format!(
            "Unknown activation: {}. Valid options: relu, sigmoid, tanh, softmax, linear, leaky_relu, elu",
            activation
        ))),
    }
}

fn parse_solver(solver: &str) -> PyResult<Solver> {
    match solver.to_lowercase().as_str() {
        "sgd" => Ok(Solver::SGD),
        "adam" => Ok(Solver::Adam),
        _ => Err(PyValueError::new_err(format!(
            "Unknown solver: {}. Valid options: sgd, adam",
            solver
        ))),
    }
}

// =============================================================================
// MLPClassifier
// =============================================================================

/// Multi-layer Perceptron Classifier.
///
/// A neural network classifier with one or more hidden layers trained using
/// backpropagation.
///
/// Parameters
/// ----------
/// hidden_layer_sizes : tuple of int, optional (default=(100,))
///     The number of neurons in each hidden layer.
/// activation : str, optional (default='relu')
///     Activation function for the hidden layers.
///     Options: 'relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu'.
/// solver : str, optional (default='adam')
///     The optimizer to use. Options: 'sgd', 'adam'.
/// learning_rate_init : float, optional (default=0.001)
///     Initial learning rate.
/// max_iter : int, optional (default=200)
///     Maximum number of epochs.
/// tol : float, optional (default=1e-4)
///     Tolerance for the optimization.
/// random_state : int, optional
///     Random seed for weight initialization and shuffling.
/// alpha : float, optional (default=0.0001)
///     L2 regularization strength.
/// batch_size : int, optional (default=200)
///     Size of minibatches for SGD/Adam.
/// early_stopping : bool, optional (default=False)
///     Whether to use early stopping based on validation score.
/// validation_fraction : float, optional (default=0.1)
///     Fraction of training data for validation (if early_stopping=True).
/// n_iter_no_change : int, optional (default=10)
///     Maximum number of epochs without improvement (if early_stopping=True).
///
/// Attributes
/// ----------
/// classes_ : ndarray of shape (n_classes,)
///     Class labels learned during fit.
/// n_layers_ : int
///     Number of layers in the network.
/// loss_curve_ : list
///     Loss at each epoch during training.
///
/// Examples
/// --------
/// >>> from ferroml.neural import MLPClassifier
/// >>> import numpy as np
/// >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
/// >>> y = np.array([0, 1, 1, 0])  # XOR problem
/// >>> clf = MLPClassifier(hidden_layer_sizes=(4, 4), max_iter=1000, random_state=42)
/// >>> clf.fit(X, y)
/// >>> clf.predict(X)
#[pyclass(name = "MLPClassifier", module = "ferroml.neural")]
pub struct PyMLPClassifier {
    inner: MLPClassifier,
}

#[pymethods]
impl PyMLPClassifier {
    /// Create a new MLPClassifier.
    #[new]
    #[pyo3(signature = (
        hidden_layer_sizes=None,
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=200,
        tol=1e-4,
        random_state=None,
        alpha=0.0001,
        batch_size=200,
        early_stopping=false,
        validation_fraction=0.1,
        n_iter_no_change=10
    ))]
    fn new(
        hidden_layer_sizes: Option<Vec<usize>>,
        activation: &str,
        solver: &str,
        learning_rate_init: f64,
        max_iter: usize,
        tol: f64,
        random_state: Option<u64>,
        alpha: f64,
        batch_size: usize,
        early_stopping: bool,
        validation_fraction: f64,
        n_iter_no_change: usize,
    ) -> PyResult<Self> {
        let hidden_sizes = hidden_layer_sizes.unwrap_or_else(|| vec![100]);
        let activation_fn = parse_activation(activation)?;
        let solver_type = parse_solver(solver)?;

        let mut inner = MLPClassifier::new()
            .hidden_layer_sizes(&hidden_sizes)
            .activation(activation_fn)
            .solver(solver_type)
            .learning_rate(learning_rate_init)
            .max_iter(max_iter)
            .tol(tol)
            .alpha(alpha)
            .batch_size(batch_size);

        if let Some(seed) = random_state {
            inner = inner.random_state(seed);
        }

        if early_stopping {
            inner = inner.early_stopping(EarlyStopping {
                patience: n_iter_no_change,
                min_delta: tol,
                validation_fraction,
            });
        }

        Ok(Self { inner })
    }

    /// Fit the MLPClassifier to the data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target values (class labels).
    ///
    /// Returns
    /// -------
    /// self : MLPClassifier
    ///     Fitted classifier.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_owned = to_owned_array_2d(x);
        let y_owned = to_owned_array_1d(y);

        slf.inner
            .fit(&x_owned, &y_owned)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(slf)
    }

    /// Predict class labels for samples.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples to predict.
    ///
    /// Returns
    /// -------
    /// y_pred : ndarray of shape (n_samples,)
    ///     Predicted class labels.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_owned = to_owned_array_2d(x);

        let predictions = self
            .inner
            .predict(&x_owned)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
    }

    /// Predict class probabilities for samples.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples to predict.
    ///
    /// Returns
    /// -------
    /// y_proba : ndarray of shape (n_samples, n_classes)
    ///     Predicted class probabilities.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_owned = to_owned_array_2d(x);

        let probas = self
            .inner
            .predict_proba(&x_owned)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(probas.into_pyarray(py))
    }

    /// Get class labels.
    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .classes_
            .as_ref()
            .map(|c| ndarray::Array1::from_vec(c.clone()).into_pyarray(py))
    }

    /// Get number of layers.
    #[getter]
    fn n_layers_(&self) -> usize {
        self.inner.n_layers()
    }

    /// Get loss curve from training.
    #[getter]
    fn loss_curve_(&self) -> Option<Vec<f64>> {
        self.inner
            .training_diagnostics()
            .map(|d| d.loss_curve.clone())
    }

    /// Check if the model is fitted.
    fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }

    /// Pickle support: get state
    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    /// Pickle support: set state
    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("MLPClassifier(n_layers={})", self.inner.n_layers())
    }
}

// =============================================================================
// MLPRegressor
// =============================================================================

/// Multi-layer Perceptron Regressor.
///
/// A neural network regressor with one or more hidden layers trained using
/// backpropagation.
///
/// Parameters
/// ----------
/// hidden_layer_sizes : tuple of int, optional (default=(100,))
///     The number of neurons in each hidden layer.
/// activation : str, optional (default='relu')
///     Activation function for the hidden layers.
///     Options: 'relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu'.
/// solver : str, optional (default='adam')
///     The optimizer to use. Options: 'sgd', 'adam'.
/// learning_rate_init : float, optional (default=0.001)
///     Initial learning rate.
/// max_iter : int, optional (default=200)
///     Maximum number of epochs.
/// tol : float, optional (default=1e-4)
///     Tolerance for the optimization.
/// random_state : int, optional
///     Random seed for weight initialization and shuffling.
/// alpha : float, optional (default=0.0001)
///     L2 regularization strength.
/// batch_size : int, optional (default=200)
///     Size of minibatches for SGD/Adam.
/// early_stopping : bool, optional (default=False)
///     Whether to use early stopping based on validation score.
/// validation_fraction : float, optional (default=0.1)
///     Fraction of training data for validation (if early_stopping=True).
/// n_iter_no_change : int, optional (default=10)
///     Maximum number of epochs without improvement (if early_stopping=True).
///
/// Attributes
/// ----------
/// n_layers_ : int
///     Number of layers in the network.
/// loss_curve_ : list
///     Loss at each epoch during training.
///
/// Examples
/// --------
/// >>> from ferroml.neural import MLPRegressor
/// >>> import numpy as np
/// >>> X = np.array([[1], [2], [3], [4], [5]])
/// >>> y = np.array([2, 4, 6, 8, 10])  # y = 2x
/// >>> reg = MLPRegressor(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
/// >>> reg.fit(X, y)
/// >>> reg.predict(X)
#[pyclass(name = "MLPRegressor", module = "ferroml.neural")]
pub struct PyMLPRegressor {
    inner: MLPRegressor,
}

#[pymethods]
impl PyMLPRegressor {
    /// Create a new MLPRegressor.
    #[new]
    #[pyo3(signature = (
        hidden_layer_sizes=None,
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=200,
        tol=1e-4,
        random_state=None,
        alpha=0.0001,
        batch_size=200,
        early_stopping=false,
        validation_fraction=0.1,
        n_iter_no_change=10
    ))]
    fn new(
        hidden_layer_sizes: Option<Vec<usize>>,
        activation: &str,
        solver: &str,
        learning_rate_init: f64,
        max_iter: usize,
        tol: f64,
        random_state: Option<u64>,
        alpha: f64,
        batch_size: usize,
        early_stopping: bool,
        validation_fraction: f64,
        n_iter_no_change: usize,
    ) -> PyResult<Self> {
        let hidden_sizes = hidden_layer_sizes.unwrap_or_else(|| vec![100]);
        let activation_fn = parse_activation(activation)?;
        let solver_type = parse_solver(solver)?;

        let mut inner = MLPRegressor::new()
            .hidden_layer_sizes(&hidden_sizes)
            .activation(activation_fn)
            .solver(solver_type)
            .learning_rate(learning_rate_init)
            .max_iter(max_iter)
            .tol(tol)
            .alpha(alpha)
            .batch_size(batch_size);

        if let Some(seed) = random_state {
            inner = inner.random_state(seed);
        }

        if early_stopping {
            inner = inner.early_stopping(EarlyStopping {
                patience: n_iter_no_change,
                min_delta: tol,
                validation_fraction,
            });
        }

        Ok(Self { inner })
    }

    /// Fit the MLPRegressor to the data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self : MLPRegressor
    ///     Fitted regressor.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_owned = to_owned_array_2d(x);
        let y_owned = to_owned_array_1d(y);

        slf.inner
            .fit(&x_owned, &y_owned)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(slf)
    }

    /// Predict target values for samples.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples to predict.
    ///
    /// Returns
    /// -------
    /// y_pred : ndarray of shape (n_samples,)
    ///     Predicted values.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_owned = to_owned_array_2d(x);

        let predictions = self
            .inner
            .predict(&x_owned)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
    }

    /// Score the model on test data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Test data.
    /// y : array-like of shape (n_samples,)
    ///     True target values.
    ///
    /// Returns
    /// -------
    /// score : float
    ///     R² score.
    fn score<'py>(
        &self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let x_owned = to_owned_array_2d(x);
        let y_owned = to_owned_array_1d(y);

        self.inner
            .score(&x_owned, &y_owned)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get number of layers.
    #[getter]
    fn n_layers_(&self) -> usize {
        self.inner.n_layers()
    }

    /// Get loss curve from training.
    #[getter]
    fn loss_curve_(&self) -> Option<Vec<f64>> {
        self.inner
            .training_diagnostics()
            .map(|d| d.loss_curve.clone())
    }

    /// Check if the model is fitted.
    fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }

    /// Pickle support: get state
    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    /// Pickle support: set state
    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("MLPRegressor(n_layers={})", self.inner.n_layers())
    }
}

// =============================================================================
// Module Registration
// =============================================================================

/// Register the neural module with Python.
pub fn register_neural_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "neural")?;

    m.add_class::<PyMLPClassifier>()?;
    m.add_class::<PyMLPRegressor>()?;

    parent.add_submodule(&m)?;

    // Make it accessible as ferroml.neural
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("ferroml.neural", m.clone())?;

    Ok(())
}

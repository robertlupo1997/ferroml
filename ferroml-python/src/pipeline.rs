//! Python bindings for FerroML pipeline components.
//!
//! This module provides Python wrappers for:
//! - Pipeline: Sequential transformer/model chaining
//! - ColumnTransformer: Apply different transformers to column subsets
//! - FeatureUnion: Parallel feature extraction with concatenation
//!
//! ## Zero-Copy Semantics
//!
//! Input arrays are passed as `PyReadonlyArray` which provides read-only access.
//! Due to the dynamic dispatch nature of pipelines (calling Python methods),
//! a copy is typically made. Output arrays use `into_pyarray` to transfer ownership
//! to Python without copying data.
//!
//! See `crate::array_utils` for detailed documentation.

use crate::array_utils::{to_owned_array_1d, to_owned_array_2d};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// =============================================================================
// Pipeline Step - wrapper for Python transformer/model objects
// =============================================================================

/// Internal representation of a pipeline step
struct PipelineStepInner {
    name: String,
    obj: PyObject,
    is_model: bool,
}

// =============================================================================
// Pipeline
// =============================================================================

/// A machine learning pipeline that chains transformers and a final model.
///
/// Pipelines ensure proper sequencing of fit/transform operations and prevent
/// data leakage by always fitting on training data first.
///
/// Parameters
/// ----------
/// steps : list of tuples
///     List of (name, transformer) tuples. The last element may be a model
///     (with fit/predict methods) or a transformer (with fit/transform methods).
///
/// Attributes
/// ----------
/// named_steps : dict
///     Access pipeline steps by name.
/// n_features_in_ : int
///     Number of features seen during fit.
///
/// Examples
/// --------
/// >>> from ferroml.pipeline import Pipeline
/// >>> from ferroml.preprocessing import StandardScaler
/// >>> from ferroml.linear import LinearRegression
/// >>> pipe = Pipeline([
/// ...     ('scaler', StandardScaler()),
/// ...     ('model', LinearRegression()),
/// ... ])
/// >>> pipe.fit(X_train, y_train)
/// >>> predictions = pipe.predict(X_test)
#[pyclass(name = "Pipeline", module = "ferroml.pipeline")]
pub struct PyPipeline {
    steps: Vec<PipelineStepInner>,
    fitted: bool,
    n_features_in: Option<usize>,
}

#[pymethods]
impl PyPipeline {
    /// Create a new Pipeline.
    ///
    /// Parameters
    /// ----------
    /// steps : list of (str, object) tuples
    ///     List of (name, transformer/model) tuples.
    #[new]
    fn new(py: Python<'_>, steps: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut pipeline_steps = Vec::new();

        for item in steps.iter() {
            let tuple = item.downcast::<pyo3::types::PyTuple>()?;
            if tuple.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Each step must be a tuple of (name, transformer/model)",
                ));
            }

            let name: String = tuple.get_item(0)?.extract()?;
            let obj: PyObject = tuple.get_item(1)?.unbind();

            // Check if it's a model (has predict method) or transformer (has transform method)
            let has_predict = obj.bind(py).hasattr("predict")?;
            let is_model = has_predict;

            pipeline_steps.push(PipelineStepInner {
                name,
                obj,
                is_model,
            });
        }

        // Validate: only the last step can be a model
        for (i, step) in pipeline_steps.iter().enumerate() {
            if step.is_model && i != pipeline_steps.len() - 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Only the last step can be a model. Step '{}' at position {} has predict method.",
                    step.name, i
                )));
            }
        }

        Ok(Self {
            steps: pipeline_steps,
            fitted: false,
            n_features_in: None,
        })
    }

    /// Fit the pipeline to training data.
    ///
    /// For each transformer: fit_transform on the data, passing the transformed
    /// data to the next step. For the final model: fit on the transformed data.
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
    /// self : Pipeline
    ///     Fitted pipeline.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if slf.steps.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Pipeline is empty. Add at least one step.",
            ));
        }

        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);

        if x_arr.nrows() != y_arr.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "X has {} samples but y has {} samples",
                x_arr.nrows(),
                y_arr.len()
            )));
        }

        slf.n_features_in = Some(x_arr.ncols());

        // Convert to Python arrays
        let mut current_x: Bound<'py, PyAny> = x_arr.into_pyarray(py).as_any().clone();
        let y_py: Bound<'py, PyAny> = y_arr.into_pyarray(py).as_any().clone();

        // Process each step
        let n_steps = slf.steps.len();
        for (i, step) in slf.steps.iter().enumerate() {
            let obj = step.obj.bind(py);
            let is_last = i == n_steps - 1;

            if step.is_model && is_last {
                // Final model: just fit
                obj.call_method("fit", (&current_x, &y_py), None)?;
            } else {
                // Transformer: fit_transform
                let result = obj.call_method("fit_transform", (&current_x,), None)?;
                current_x = result;
            }
        }

        slf.fitted = true;
        Ok(slf)
    }

    /// Transform data through all transformer steps.
    ///
    /// Does not apply the final model (if any). Use `predict` for that.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Data to transform.
    ///
    /// Returns
    /// -------
    /// X_transformed : ndarray of shape (n_samples, n_features_out)
    ///     Transformed data.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if !self.fitted {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Pipeline not fitted. Call fit() first.",
            ));
        }

        self.validate_n_features(x.as_array().ncols())?;

        let x_arr = to_owned_array_2d(x);
        let mut current_x: Bound<'py, PyAny> = x_arr.into_pyarray(py).as_any().clone();

        // Transform through all non-model steps
        for step in &self.steps {
            if step.is_model {
                break; // Stop before model
            }
            let obj = step.obj.bind(py);
            let result = obj.call_method("transform", (&current_x,), None)?;
            current_x = result;
        }

        // Convert result back to Array2
        let result_arr: PyReadonlyArray2<'py, f64> = current_x.extract()?;
        Ok(to_owned_array_2d(result_arr).into_pyarray(py))
    }

    /// Fit to data, then transform it.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Input data.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// X_transformed : ndarray
    ///     Transformed data.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if slf.steps.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Pipeline is empty. Add at least one step.",
            ));
        }

        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);

        if x_arr.nrows() != y_arr.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "X has {} samples but y has {} samples",
                x_arr.nrows(),
                y_arr.len()
            )));
        }

        slf.n_features_in = Some(x_arr.ncols());

        // Convert to Python arrays
        let mut current_x: Bound<'py, PyAny> = x_arr.into_pyarray(py).as_any().clone();
        let y_py: Bound<'py, PyAny> = y_arr.into_pyarray(py).as_any().clone();

        // Process each step
        let n_steps = slf.steps.len();
        for (i, step) in slf.steps.iter().enumerate() {
            let obj = step.obj.bind(py);
            let is_last = i == n_steps - 1;

            if step.is_model && is_last {
                // Final model: just fit
                obj.call_method("fit", (&current_x, &y_py), None)?;
            } else {
                // Transformer: fit_transform
                let result = obj.call_method("fit_transform", (&current_x,), None)?;
                current_x = result;
            }
        }

        slf.fitted = true;

        // Convert final result back to Array2
        let result_arr: PyReadonlyArray2<'py, f64> = current_x.extract()?;
        Ok(to_owned_array_2d(result_arr).into_pyarray(py))
    }

    /// Predict using the pipeline.
    ///
    /// Transforms data through all transformers, then applies the final model.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Data to predict.
    ///
    /// Returns
    /// -------
    /// y_pred : ndarray of shape (n_samples,)
    ///     Predictions from the final model.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Pipeline not fitted. Call fit() first.",
            ));
        }

        // Check if the last step is a model
        let last_step = self
            .steps
            .last()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Pipeline is empty."))?;

        if !last_step.is_model {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Pipeline has no model. Use transform() for transformer-only pipelines.",
            ));
        }

        self.validate_n_features(x.as_array().ncols())?;

        let x_arr = to_owned_array_2d(x);
        let mut current_x: Bound<'py, PyAny> = x_arr.into_pyarray(py).as_any().clone();

        // Transform through all non-model steps
        for step in &self.steps {
            if step.is_model {
                // Apply the model's predict
                let obj = step.obj.bind(py);
                let result = obj.call_method("predict", (&current_x,), None)?;
                let result_arr: PyReadonlyArray1<'py, f64> = result.extract()?;
                return Ok(to_owned_array_1d(result_arr).into_pyarray(py));
            }
            let obj = step.obj.bind(py);
            let result = obj.call_method("transform", (&current_x,), None)?;
            current_x = result;
        }

        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Unexpected error in predict",
        ))
    }

    /// Get the names of all steps.
    #[getter]
    fn step_names(&self) -> Vec<String> {
        self.steps.iter().map(|s| s.name.clone()).collect()
    }

    /// Get named steps as a dictionary.
    #[getter]
    fn named_steps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for step in &self.steps {
            dict.set_item(&step.name, step.obj.clone_ref(py))?;
        }
        Ok(dict)
    }

    /// Get the number of input features (after fitting).
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.n_features_in.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Pipeline not fitted. Call fit() first.",
            )
        })
    }

    /// Set hyperparameters using nested naming convention.
    ///
    /// Parameter names use double-underscore: "step_name__param_name".
    ///
    /// Parameters
    /// ----------
    /// **params : dict
    ///     Parameters to set.
    ///
    /// Returns
    /// -------
    /// self : Pipeline
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if let Some(params) = params {
            for (key, value) in params.iter() {
                let key_str: String = key.extract()?;
                let parts: Vec<&str> = key_str.splitn(2, "__").collect();

                if parts.len() != 2 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid parameter name '{}'. Expected format: 'step__param'",
                        key_str
                    )));
                }

                let (step_name, param_name) = (parts[0], parts[1]);

                // Find the step
                let step = slf
                    .steps
                    .iter()
                    .find(|s| s.name == step_name)
                    .ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Step '{}' not found in pipeline",
                            step_name
                        ))
                    })?;

                // Set the parameter on the step's object
                let obj = step.obj.bind(py);
                let param_dict = PyDict::new(py);
                param_dict.set_item(param_name, value)?;
                obj.call_method("set_params", (), Some(&param_dict))?;
            }
        }

        // Mark as unfitted since parameters changed
        slf.fitted = false;
        Ok(slf)
    }

    /// Get parameters from all steps.
    fn get_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let params = PyDict::new(py);

        for step in &self.steps {
            let obj = step.obj.bind(py);
            if obj.hasattr("get_params")? {
                let step_params: Bound<'py, PyDict> =
                    obj.call_method("get_params", (), None)?.extract()?;

                for (key, value) in step_params.iter() {
                    let key_str: String = key.extract()?;
                    let prefixed_key = format!("{}__{}", step.name, key_str);
                    params.set_item(prefixed_key, value)?;
                }
            }
        }

        Ok(params)
    }

    fn __repr__(&self) -> String {
        let step_strs: Vec<String> = self
            .steps
            .iter()
            .map(|s| format!("('{}', ...)", s.name))
            .collect();
        format!("Pipeline([{}])", step_strs.join(", "))
    }
}

impl PyPipeline {
    fn validate_n_features(&self, n_features: usize) -> PyResult<()> {
        if let Some(expected) = self.n_features_in {
            if n_features != expected {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "X has {} features, but Pipeline is expecting {} features",
                    n_features, expected
                )));
            }
        }
        Ok(())
    }
}

// =============================================================================
// ColumnTransformer
// =============================================================================

/// Internal representation of a column transformer step
struct ColumnTransformerStepInner {
    name: String,
    transformer: PyObject,
    columns: Vec<usize>,
}

/// Applies different transformers to different subsets of columns.
///
/// This is useful when different features need different preprocessing.
///
/// Parameters
/// ----------
/// transformers : list of tuples
///     List of (name, transformer, columns) tuples. Columns can be:
///     - List of column indices (e.g., [0, 1, 2])
///     - String "all" to select all columns
/// remainder : str, optional (default='drop')
///     How to handle columns not assigned to any transformer.
///     'drop': Exclude from output
///     'passthrough': Include unchanged in output
///
/// Attributes
/// ----------
/// n_features_in_ : int
///     Number of features seen during fit.
/// transformers_ : list
///     The fitted transformers.
///
/// Examples
/// --------
/// >>> from ferroml.pipeline import ColumnTransformer
/// >>> from ferroml.preprocessing import StandardScaler, OneHotEncoder
/// >>> ct = ColumnTransformer([
/// ...     ('scaler', StandardScaler(), [0, 1, 2]),
/// ...     ('encoder', OneHotEncoder(), [3]),
/// ... ])
/// >>> X_transformed = ct.fit_transform(X)
#[pyclass(name = "ColumnTransformer", module = "ferroml.pipeline")]
pub struct PyColumnTransformer {
    transformers: Vec<ColumnTransformerStepInner>,
    remainder: String,
    fitted: bool,
    n_features_in: Option<usize>,
    n_features_out: Option<usize>,
    remainder_indices: Vec<usize>,
}

#[pymethods]
impl PyColumnTransformer {
    /// Create a new ColumnTransformer.
    #[new]
    #[pyo3(signature = (transformers, remainder="drop"))]
    fn new(transformers: &Bound<'_, PyList>, remainder: &str) -> PyResult<Self> {
        let mut steps = Vec::new();
        let mut all_columns: Vec<usize> = Vec::new();

        for item in transformers.iter() {
            let tuple = item.downcast::<pyo3::types::PyTuple>()?;
            if tuple.len() != 3 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Each transformer must be a tuple of (name, transformer, columns)",
                ));
            }

            let name: String = tuple.get_item(0)?.extract()?;
            let transformer: PyObject = tuple.get_item(1)?.unbind();

            // Parse columns - can be a list of indices or "all"
            let columns_item = tuple.get_item(2)?;
            let columns: Vec<usize> = if let Ok(cols) = columns_item.extract::<Vec<usize>>() {
                cols
            } else if let Ok(col_str) = columns_item.extract::<String>() {
                if col_str == "all" {
                    // Will be resolved during fit
                    vec![]
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid columns specification: '{}'. Use list of indices or 'all'",
                        col_str
                    )));
                }
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Columns must be a list of indices or 'all'",
                ));
            };

            all_columns.extend(&columns);

            steps.push(ColumnTransformerStepInner {
                name,
                transformer,
                columns,
            });
        }

        // Validate remainder
        if remainder != "drop" && remainder != "passthrough" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid remainder: '{}'. Use 'drop' or 'passthrough'",
                remainder
            )));
        }

        Ok(Self {
            transformers: steps,
            remainder: remainder.to_string(),
            fitted: false,
            n_features_in: None,
            n_features_out: None,
            remainder_indices: Vec::new(),
        })
    }

    /// Fit the transformer to training data.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if slf.transformers.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ColumnTransformer is empty. Add at least one transformer.",
            ));
        }

        let x_arr = to_owned_array_2d(x);
        let n_features = x_arr.ncols();
        slf.n_features_in = Some(n_features);

        // Resolve "all" columns
        for step in &mut slf.transformers {
            if step.columns.is_empty() {
                step.columns = (0..n_features).collect();
            }
        }

        // Find remainder columns
        let used_columns: std::collections::HashSet<usize> = slf
            .transformers
            .iter()
            .flat_map(|t| t.columns.iter().copied())
            .collect();
        slf.remainder_indices = (0..n_features)
            .filter(|i| !used_columns.contains(i))
            .collect();

        // Fit each transformer
        let mut total_features_out = 0;
        for step in &slf.transformers {
            // Extract columns
            let subset = extract_columns(&x_arr, &step.columns);
            let subset_py = subset.into_pyarray(py);

            // Fit the transformer
            let obj = step.transformer.bind(py);
            obj.call_method("fit", (subset_py,), None)?;

            // Get output features count
            let n_out = if obj.hasattr("n_features_out_")? {
                obj.getattr("n_features_out_")?.extract::<usize>()?
            } else {
                // Try transform to get shape
                let dummy = Array2::<f64>::zeros((1, step.columns.len()));
                let dummy_py = dummy.into_pyarray(py);
                let transformed = obj.call_method("transform", (dummy_py,), None)?;
                let transformed_arr: PyReadonlyArray2<'_, f64> = transformed.extract()?;
                transformed_arr.as_array().ncols()
            };
            total_features_out += n_out;
        }

        // Add remainder columns if passthrough
        if slf.remainder == "passthrough" {
            total_features_out += slf.remainder_indices.len();
        }

        slf.n_features_out = Some(total_features_out);
        slf.fitted = true;
        Ok(slf)
    }

    /// Transform the data.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if !self.fitted {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ColumnTransformer not fitted. Call fit() first.",
            ));
        }

        self.validate_n_features(x.as_array().ncols())?;

        let x_arr = to_owned_array_2d(x);
        let mut transformed_parts: Vec<Array2<f64>> = Vec::new();

        // Transform each subset
        for step in &self.transformers {
            let subset = extract_columns(&x_arr, &step.columns);
            let subset_py = subset.into_pyarray(py);

            let obj = step.transformer.bind(py);
            let result = obj.call_method("transform", (subset_py,), None)?;
            let result_arr: PyReadonlyArray2<'py, f64> = result.extract()?;
            transformed_parts.push(to_owned_array_2d(result_arr));
        }

        // Add remainder if passthrough
        if self.remainder == "passthrough" && !self.remainder_indices.is_empty() {
            let remainder = extract_columns(&x_arr, &self.remainder_indices);
            transformed_parts.push(remainder);
        }

        // Concatenate horizontally
        let result = hconcat(&transformed_parts);
        Ok(result.into_pyarray(py))
    }

    /// Fit to data, then transform it.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if slf.transformers.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ColumnTransformer is empty. Add at least one transformer.",
            ));
        }

        let x_arr = to_owned_array_2d(x);
        let n_features = x_arr.ncols();
        slf.n_features_in = Some(n_features);

        // Resolve "all" columns
        for step in &mut slf.transformers {
            if step.columns.is_empty() {
                step.columns = (0..n_features).collect();
            }
        }

        // Find remainder columns
        let used_columns: std::collections::HashSet<usize> = slf
            .transformers
            .iter()
            .flat_map(|t| t.columns.iter().copied())
            .collect();
        slf.remainder_indices = (0..n_features)
            .filter(|i| !used_columns.contains(i))
            .collect();

        // Fit and transform each subset
        let mut total_features_out = 0;
        let mut transformed_parts: Vec<Array2<f64>> = Vec::new();

        for step in &slf.transformers {
            // Extract columns
            let subset = extract_columns(&x_arr, &step.columns);
            let subset_py = subset.into_pyarray(py);

            // Fit and transform
            let obj = step.transformer.bind(py);
            let result = obj.call_method("fit_transform", (subset_py,), None)?;
            let result_arr: PyReadonlyArray2<'py, f64> = result.extract()?;
            let n_out = result_arr.as_array().ncols();
            total_features_out += n_out;
            transformed_parts.push(to_owned_array_2d(result_arr));
        }

        // Add remainder columns if passthrough
        if slf.remainder == "passthrough" && !slf.remainder_indices.is_empty() {
            let remainder = extract_columns(&x_arr, &slf.remainder_indices);
            total_features_out += remainder.ncols();
            transformed_parts.push(remainder);
        }

        slf.n_features_out = Some(total_features_out);
        slf.fitted = true;

        // Concatenate horizontally
        let result = hconcat(&transformed_parts);
        Ok(result.into_pyarray(py))
    }

    /// Get the names of all transformers.
    #[getter]
    fn transformer_names(&self) -> Vec<String> {
        self.transformers.iter().map(|t| t.name.clone()).collect()
    }

    /// Get the number of input features.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.n_features_in.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ColumnTransformer not fitted. Call fit() first.",
            )
        })
    }

    /// Get the number of output features.
    #[getter]
    fn n_features_out_(&self) -> PyResult<usize> {
        self.n_features_out.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ColumnTransformer not fitted. Call fit() first.",
            )
        })
    }

    fn __repr__(&self) -> String {
        let transformer_strs: Vec<String> = self
            .transformers
            .iter()
            .map(|t| format!("('{}', ..., {:?})", t.name, t.columns))
            .collect();
        format!(
            "ColumnTransformer([{}], remainder='{}')",
            transformer_strs.join(", "),
            self.remainder
        )
    }
}

impl PyColumnTransformer {
    fn validate_n_features(&self, n_features: usize) -> PyResult<()> {
        if let Some(expected) = self.n_features_in {
            if n_features != expected {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "X has {} features, but ColumnTransformer is expecting {} features",
                    n_features, expected
                )));
            }
        }
        Ok(())
    }
}

// =============================================================================
// FeatureUnion
// =============================================================================

/// Internal representation of a feature union step
struct FeatureUnionStepInner {
    name: String,
    transformer: PyObject,
}

/// Apply multiple transformers in parallel and concatenate their outputs.
///
/// FeatureUnion is useful for combining different feature extraction methods
/// on the same input data.
///
/// Parameters
/// ----------
/// transformer_list : list of tuples
///     List of (name, transformer) tuples.
///
/// Attributes
/// ----------
/// n_features_in_ : int
///     Number of features seen during fit.
///
/// Examples
/// --------
/// >>> from ferroml.pipeline import FeatureUnion
/// >>> from ferroml.preprocessing import StandardScaler, MinMaxScaler
/// >>> union = FeatureUnion([
/// ...     ('standard', StandardScaler()),
/// ...     ('minmax', MinMaxScaler()),
/// ... ])
/// >>> X_combined = union.fit_transform(X)  # Concatenates both scalings
#[pyclass(name = "FeatureUnion", module = "ferroml.pipeline")]
pub struct PyFeatureUnion {
    transformers: Vec<FeatureUnionStepInner>,
    fitted: bool,
    n_features_in: Option<usize>,
    n_features_out: Option<usize>,
}

#[pymethods]
impl PyFeatureUnion {
    /// Create a new FeatureUnion.
    #[new]
    fn new(transformer_list: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut transformers = Vec::new();

        for item in transformer_list.iter() {
            let tuple = item.downcast::<pyo3::types::PyTuple>()?;
            if tuple.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Each item must be a tuple of (name, transformer)",
                ));
            }

            let name: String = tuple.get_item(0)?.extract()?;
            let transformer: PyObject = tuple.get_item(1)?.unbind();

            transformers.push(FeatureUnionStepInner { name, transformer });
        }

        Ok(Self {
            transformers,
            fitted: false,
            n_features_in: None,
            n_features_out: None,
        })
    }

    /// Fit all transformers.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if slf.transformers.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "FeatureUnion is empty. Add at least one transformer.",
            ));
        }

        let x_arr = to_owned_array_2d(x);
        let n_features_in = x_arr.ncols();
        slf.n_features_in = Some(n_features_in);
        let x_py = x_arr.into_pyarray(py);

        let mut total_features_out = 0;
        for step in &slf.transformers {
            let obj = step.transformer.bind(py);
            obj.call_method("fit", (x_py.clone(),), None)?;

            // Get output features count
            let n_out = if obj.hasattr("n_features_out_")? {
                obj.getattr("n_features_out_")?.extract::<usize>()?
            } else if obj.hasattr("n_features_in_")? {
                // Same as input if not transformed
                obj.getattr("n_features_in_")?.extract::<usize>()?
            } else {
                // Estimate from transform
                let dummy = Array2::<f64>::zeros((1, n_features_in));
                let dummy_py = dummy.into_pyarray(py);
                let transformed = obj.call_method("transform", (dummy_py,), None)?;
                let transformed_arr: PyReadonlyArray2<'_, f64> = transformed.extract()?;
                transformed_arr.as_array().ncols()
            };
            total_features_out += n_out;
        }

        slf.n_features_out = Some(total_features_out);
        slf.fitted = true;
        Ok(slf)
    }

    /// Transform data through all transformers and concatenate.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if !self.fitted {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "FeatureUnion not fitted. Call fit() first.",
            ));
        }

        self.validate_n_features(x.as_array().ncols())?;

        let x_arr = to_owned_array_2d(x);
        let x_py = x_arr.into_pyarray(py);

        let mut transformed_parts: Vec<Array2<f64>> = Vec::new();

        for step in &self.transformers {
            let obj = step.transformer.bind(py);
            let result = obj.call_method("transform", (x_py.clone(),), None)?;
            let result_arr: PyReadonlyArray2<'py, f64> = result.extract()?;
            transformed_parts.push(to_owned_array_2d(result_arr));
        }

        let result = hconcat(&transformed_parts);
        Ok(result.into_pyarray(py))
    }

    /// Fit to data, then transform it.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if slf.transformers.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "FeatureUnion is empty. Add at least one transformer.",
            ));
        }

        let x_arr = to_owned_array_2d(x);
        let n_features_in = x_arr.ncols();
        slf.n_features_in = Some(n_features_in);
        let x_py = x_arr.into_pyarray(py);

        let mut total_features_out = 0;
        let mut transformed_parts: Vec<Array2<f64>> = Vec::new();

        for step in &slf.transformers {
            let obj = step.transformer.bind(py);
            let result = obj.call_method("fit_transform", (x_py.clone(),), None)?;
            let result_arr: PyReadonlyArray2<'py, f64> = result.extract()?;
            let n_out = result_arr.as_array().ncols();
            total_features_out += n_out;
            transformed_parts.push(to_owned_array_2d(result_arr));
        }

        slf.n_features_out = Some(total_features_out);
        slf.fitted = true;

        let result = hconcat(&transformed_parts);
        Ok(result.into_pyarray(py))
    }

    /// Get the names of all transformers.
    #[getter]
    fn transformer_names(&self) -> Vec<String> {
        self.transformers.iter().map(|t| t.name.clone()).collect()
    }

    /// Get the number of input features.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.n_features_in.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "FeatureUnion not fitted. Call fit() first.",
            )
        })
    }

    /// Get the number of output features.
    #[getter]
    fn n_features_out_(&self) -> PyResult<usize> {
        self.n_features_out.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "FeatureUnion not fitted. Call fit() first.",
            )
        })
    }

    fn __repr__(&self) -> String {
        let transformer_strs: Vec<String> = self
            .transformers
            .iter()
            .map(|t| format!("('{}', ...)", t.name))
            .collect();
        format!("FeatureUnion([{}])", transformer_strs.join(", "))
    }
}

impl PyFeatureUnion {
    fn validate_n_features(&self, n_features: usize) -> PyResult<()> {
        if let Some(expected) = self.n_features_in {
            if n_features != expected {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "X has {} features, but FeatureUnion is expecting {} features",
                    n_features, expected
                )));
            }
        }
        Ok(())
    }
}

// =============================================================================
// Utility functions
// =============================================================================

/// Extract columns from a 2D array by indices.
fn extract_columns(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    if indices.is_empty() {
        return Array2::zeros((x.nrows(), 0));
    }

    let n_samples = x.nrows();
    let n_cols = indices.len();
    let mut result = Array2::zeros((n_samples, n_cols));

    for (new_col, &old_col) in indices.iter().enumerate() {
        result.column_mut(new_col).assign(&x.column(old_col));
    }

    result
}

/// Horizontally concatenate multiple 2D arrays.
fn hconcat(arrays: &[Array2<f64>]) -> Array2<f64> {
    if arrays.is_empty() {
        return Array2::zeros((0, 0));
    }

    let n_samples = arrays[0].nrows();
    let total_cols: usize = arrays.iter().map(|a| a.ncols()).sum();

    let mut result = Array2::zeros((n_samples, total_cols));
    let mut col_offset = 0;

    for arr in arrays {
        let n_cols = arr.ncols();
        result
            .slice_mut(ndarray::s![.., col_offset..col_offset + n_cols])
            .assign(arr);
        col_offset += n_cols;
    }

    result
}

// =============================================================================
// TextPipeline - text document processing pipeline
// =============================================================================

/// Check if a Python object is a scipy sparse matrix.
fn is_sparse(py: Python<'_>, obj: &Bound<'_, PyAny>) -> bool {
    if let Ok(scipy_sparse) = py.import("scipy.sparse") {
        if let Ok(result) = scipy_sparse.call_method1("issparse", (obj,)) {
            return result.extract::<bool>().unwrap_or(false);
        }
    }
    false
}

/// Call fit on a model, handling sparse input by trying fit_sparse first,
/// then falling back to toarray() + fit.
fn model_fit<'py>(
    obj: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    py: Python<'py>,
) -> PyResult<()> {
    if is_sparse(py, x) {
        // Try fit_sparse first
        if obj.hasattr("fit_sparse")? {
            obj.call_method("fit_sparse", (x, y), None)?;
            return Ok(());
        }
        // Fall back: densify and call fit
        let dense = x.call_method0("toarray")?;
        obj.call_method("fit", (&dense, y), None)?;
    } else {
        obj.call_method("fit", (x, y), None)?;
    }
    Ok(())
}

/// Call predict on a model, handling sparse input by trying predict_sparse first,
/// then falling back to toarray() + predict.
fn model_predict<'py>(
    obj: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    if is_sparse(py, x) {
        // Try predict_sparse first
        if obj.hasattr("predict_sparse")? {
            return obj.call_method("predict_sparse", (x,), None);
        }
        // Fall back: densify and call predict
        let dense = x.call_method0("toarray")?;
        return obj.call_method("predict", (&dense,), None);
    }
    obj.call_method("predict", (x,), None)
}

/// A step in the text pipeline
struct TextPipelineStepInner {
    name: String,
    obj: PyObject,
    is_model: bool,
}

/// A pipeline for text classification/regression workflows.
///
/// Accepts raw text documents as input, chains text transformers (like
/// TfidfVectorizer) and a final model for end-to-end text processing.
///
/// Parameters
/// ----------
/// steps : list of tuples
///     List of (name, transformer/model) tuples. The last element should be
///     a model (with fit/predict). Earlier elements should be text transformers
///     (with fit/transform that accept text documents).
///
/// Examples
/// --------
/// >>> from ferroml.pipeline import TextPipeline
/// >>> from ferroml.preprocessing import TfidfVectorizer
/// >>> from ferroml.naive_bayes import MultinomialNB
/// >>> pipe = TextPipeline([
/// ...     ('tfidf', TfidfVectorizer()),
/// ...     ('model', MultinomialNB()),
/// ... ])
/// >>> pipe.fit(documents_train, y_train)
/// >>> predictions = pipe.predict(documents_test)
#[pyclass(name = "TextPipeline", module = "ferroml.pipeline")]
pub struct PyTextPipeline {
    steps: Vec<TextPipelineStepInner>,
    fitted: bool,
}

#[pymethods]
impl PyTextPipeline {
    /// Create a new TextPipeline.
    ///
    /// Parameters
    /// ----------
    /// steps : list of (str, object) tuples
    ///     List of (name, transformer/model) tuples. The last element
    ///     should be a model (with fit/predict). Earlier elements should
    ///     be text transformers (with fit/transform that accept text).
    ///
    /// Returns
    /// -------
    /// TextPipeline
    ///     A new TextPipeline instance.
    #[new]
    fn new(py: Python<'_>, steps: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut pipeline_steps = Vec::new();

        for item in steps.iter() {
            let tuple = item.downcast::<pyo3::types::PyTuple>()?;
            if tuple.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Each step must be a tuple of (name, transformer/model)",
                ));
            }

            let name: String = tuple.get_item(0)?.extract()?;
            let obj: PyObject = tuple.get_item(1)?.unbind();

            let has_predict = obj.bind(py).hasattr("predict")?;

            pipeline_steps.push(TextPipelineStepInner {
                name,
                obj,
                is_model: has_predict,
            });
        }

        // Validate: only the last step can be a model
        for (i, step) in pipeline_steps.iter().enumerate() {
            if step.is_model && i != pipeline_steps.len() - 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Only the last step can be a model. Step '{}' at position {} has predict method.",
                    step.name, i
                )));
            }
        }

        Ok(Self {
            steps: pipeline_steps,
            fitted: false,
        })
    }

    /// Fit the pipeline to training data.
    ///
    /// Parameters
    /// ----------
    /// documents : list of str
    ///     Training text documents.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self : TextPipeline
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        documents: Vec<String>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if slf.steps.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "TextPipeline is empty. Add at least one step.",
            ));
        }

        let y_arr = to_owned_array_1d(y);

        if documents.len() != y_arr.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "documents has {} samples but y has {} samples",
                documents.len(),
                y_arr.len()
            )));
        }

        // Start with documents as a Python list
        let docs_py = pyo3::types::PyList::new(py, &documents)?;
        let mut current_x: Bound<'py, PyAny> = docs_py.into_any();
        let y_py: Bound<'py, PyAny> = y_arr.into_pyarray(py).as_any().clone();

        let n_steps = slf.steps.len();
        for (i, step) in slf.steps.iter().enumerate() {
            let obj = step.obj.bind(py);
            let is_last = i == n_steps - 1;

            if step.is_model && is_last {
                // Final model: fit(X, y) — handles sparse via fit_sparse or densification
                model_fit(obj, &current_x, &y_py, py)?;
            } else {
                // Transformer: fit_transform(X)
                let result = obj.call_method("fit_transform", (&current_x,), None)?;
                current_x = result;
            }
        }

        slf.fitted = true;
        Ok(slf)
    }

    /// Predict from text documents.
    ///
    /// Parameters
    /// ----------
    /// documents : list of str
    ///     Text documents to predict on.
    ///
    /// Returns
    /// -------
    /// predictions : ndarray of shape (n_samples,)
    fn predict<'py>(
        &self,
        py: Python<'py>,
        documents: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "TextPipeline not fitted. Call fit() first.",
            ));
        }

        let docs_py = pyo3::types::PyList::new(py, &documents)?;
        let mut current_x: Bound<'py, PyAny> = docs_py.into_any();

        let n_steps = self.steps.len();
        for (i, step) in self.steps.iter().enumerate() {
            let obj = step.obj.bind(py);
            let is_last = i == n_steps - 1;

            if step.is_model && is_last {
                // Final model: predict(X) — handles sparse via predict_sparse or densification
                let result = model_predict(obj, &current_x, py)?;
                let pred: PyReadonlyArray1<'py, f64> = result.extract()?;
                return Ok(to_owned_array_1d(pred).into_pyarray(py));
            } else {
                // Transformer: transform(X)
                let result = obj.call_method("transform", (&current_x,), None)?;
                current_x = result;
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "TextPipeline has no model step for prediction.",
        ))
    }

    /// Transform text through all transformer steps (without final model).
    ///
    /// Returns
    /// -------
    /// X_transformed : scipy.sparse.csr_matrix or ndarray
    fn transform<'py>(
        &self,
        py: Python<'py>,
        documents: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !self.fitted {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "TextPipeline not fitted. Call fit() first.",
            ));
        }

        let docs_py = pyo3::types::PyList::new(py, &documents)?;
        let mut current_x: Bound<'py, PyAny> = docs_py.into_any();

        for step in &self.steps {
            if step.is_model {
                break;
            }
            let obj = step.obj.bind(py);
            let result = obj.call_method("transform", (&current_x,), None)?;
            current_x = result;
        }

        Ok(current_x)
    }

    /// Fit and predict in one step.
    fn fit_predict<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        documents: Vec<String>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y_arr = to_owned_array_1d(y);

        if documents.len() != y_arr.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "documents has {} samples but y has {} samples",
                documents.len(),
                y_arr.len()
            )));
        }

        let docs_py = pyo3::types::PyList::new(py, &documents)?;
        let mut current_x: Bound<'py, PyAny> = docs_py.into_any();
        let y_py: Bound<'py, PyAny> = y_arr.into_pyarray(py).as_any().clone();

        let n_steps = slf.steps.len();

        for (i, step) in slf.steps.iter().enumerate() {
            let obj = step.obj.bind(py);
            let is_last = i == n_steps - 1;

            if step.is_model && is_last {
                model_fit(obj, &current_x, &y_py, py)?;
                let result = model_predict(obj, &current_x, py)?;
                let pred: PyReadonlyArray1<'py, f64> = result.extract()?;
                slf.fitted = true;
                return Ok(to_owned_array_1d(pred).into_pyarray(py));
            } else {
                let result = obj.call_method("fit_transform", (&current_x,), None)?;
                current_x = result;
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "TextPipeline has no model step for prediction.",
        ))
    }

    /// Access pipeline steps by name.
    #[getter]
    fn named_steps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for step in &self.steps {
            dict.set_item(&step.name, step.obj.bind(py))?;
        }
        Ok(dict)
    }

    /// Get step names.
    fn get_step_names(&self) -> Vec<String> {
        self.steps.iter().map(|s| s.name.clone()).collect()
    }

    /// Check if fitted.
    #[getter]
    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn __repr__(&self) -> String {
        let step_names: Vec<&str> = self.steps.iter().map(|s| s.name.as_str()).collect();
        format!("TextPipeline(steps={:?})", step_names)
    }

    fn __len__(&self) -> usize {
        self.steps.len()
    }
}

// =============================================================================
// Module registration
// =============================================================================

/// Register the pipeline submodule.
pub fn register_pipeline_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let pipeline_module = PyModule::new(parent_module.py(), "pipeline")?;

    pipeline_module.add_class::<PyPipeline>()?;
    pipeline_module.add_class::<PyColumnTransformer>()?;
    pipeline_module.add_class::<PyFeatureUnion>()?;
    pipeline_module.add_class::<PyTextPipeline>()?;

    parent_module.add_submodule(&pipeline_module)?;

    Ok(())
}

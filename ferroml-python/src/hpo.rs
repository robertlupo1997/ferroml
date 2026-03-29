//! Python bindings for FerroML Hyperparameter Optimization
//!
//! This module provides two APIs for hyperparameter optimization:
//!
//! 1. **sklearn-like API**: `GridSearchCV` and `RandomSearchCV` accept a model and
//!    parameter grid/distributions, performing k-fold cross-validation internally.
//!
//! 2. **Optuna-like API**: `Study` provides an `optimize` method that accepts a
//!    Python callable objective function.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

// =============================================================================
// Scoring helpers
// =============================================================================

/// Compute a score given y_true and y_pred arrays via Python numpy.
fn compute_score(
    py: Python<'_>,
    y_true: &Bound<'_, pyo3::PyAny>,
    y_pred: &Bound<'_, pyo3::PyAny>,
    scoring: &str,
) -> PyResult<f64> {
    let np = py.import("numpy")?;
    match scoring {
        "accuracy" => {
            let eq = np.call_method1("equal", (y_true, y_pred))?;
            let mean = np.call_method1("mean", (eq,))?;
            mean.extract::<f64>()
        }
        "neg_mean_squared_error" => {
            let diff = np.call_method1("subtract", (y_true, y_pred))?;
            let sq = np.call_method1("square", (diff,))?;
            let mean = np.call_method1("mean", (sq,))?;
            let mse = mean.extract::<f64>()?;
            Ok(-mse)
        }
        "r2" => {
            let y_mean = np.call_method1("mean", (y_true,))?;
            let diff_pred = np.call_method1("subtract", (y_true, y_pred))?;
            let ss_res = np.call_method1("sum", (np.call_method1("square", (diff_pred,))?,))?;
            let diff_mean = np.call_method1("subtract", (y_true, y_mean))?;
            let ss_tot = np.call_method1("sum", (np.call_method1("square", (diff_mean,))?,))?;
            let ss_res_f: f64 = ss_res.extract()?;
            let ss_tot_f: f64 = ss_tot.extract()?;
            if ss_tot_f == 0.0 {
                Ok(0.0)
            } else {
                Ok(1.0 - ss_res_f / ss_tot_f)
            }
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown scoring '{}'. Use 'accuracy', 'neg_mean_squared_error', or 'r2'.",
            other
        ))),
    }
}

/// Perform k-fold cross-validation on a model class with given params.
/// Creates a fresh model instance for each fold using `model.__class__(**params)`.
/// Returns the mean CV score.
fn cross_validate(
    py: Python<'_>,
    model: &Bound<'_, pyo3::PyAny>,
    x: &Bound<'_, pyo3::PyAny>,
    y: &Bound<'_, pyo3::PyAny>,
    params: &Bound<'_, PyDict>,
    cv: usize,
    scoring: &str,
) -> PyResult<f64> {
    let n_samples: usize = x.getattr("shape")?.get_item(0)?.extract()?;

    if n_samples < cv {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Cannot perform {}-fold CV with only {} samples",
            cv, n_samples
        )));
    }

    let model_class = model.getattr("__class__")?;

    // Create fold indices
    let fold_size = n_samples / cv;
    let mut scores = Vec::with_capacity(cv);

    for fold in 0..cv {
        let test_start = fold * fold_size;
        let test_end = if fold == cv - 1 {
            n_samples
        } else {
            (fold + 1) * fold_size
        };

        let mut train_indices: Vec<i64> = Vec::new();
        let mut test_indices: Vec<i64> = Vec::new();
        for i in 0..n_samples {
            if i < test_start || i >= test_end {
                train_indices.push(i as i64);
            } else {
                test_indices.push(i as i64);
            }
        }

        let train_idx = numpy::PyArray1::from_vec(py, train_indices);
        let test_idx = numpy::PyArray1::from_vec(py, test_indices);

        let x_train = x.get_item(&train_idx)?;
        let y_train = y.get_item(&train_idx)?;
        let x_test = x.get_item(&test_idx)?;
        let y_test = y.get_item(&test_idx)?;

        // Create fresh model with the given params as kwargs
        let fold_model = model_class.call((), Some(params))?;
        fold_model.call_method1("fit", (&x_train, &y_train))?;
        let y_pred = fold_model.call_method1("predict", (&x_test,))?;

        let score = compute_score(py, &y_test, &y_pred, scoring)?;
        scores.push(score);
    }

    let mean_score: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
    Ok(mean_score)
}

// =============================================================================
// GridSearchCV
// =============================================================================

/// Grid search with cross-validation over all parameter combinations.
///
/// Parameters
/// ----------
/// model : object
///     A FerroML model with `.fit(X, y)` and `.predict(X)` methods.
/// param_grid : dict
///     Dictionary mapping parameter names to lists of values to try.
/// cv : int, optional (default=5)
///     Number of cross-validation folds.
/// scoring : str, optional (default="accuracy")
///     Scoring metric: "accuracy", "neg_mean_squared_error", or "r2".
///
/// Attributes (after fit)
/// ----------------------
/// best_params_ : dict
///     Parameter setting that gave the best results.
/// best_score_ : float
///     Mean cross-validated score of the best estimator.
/// cv_results_ : list[dict]
///     List of dicts with keys "params" and "mean_score" for each combination.
#[pyclass(name = "GridSearchCV")]
pub struct PyGridSearchCV {
    model: PyObject,
    param_grid: PyObject,
    cv: usize,
    scoring: String,
    best_params: Option<PyObject>,
    best_score: Option<f64>,
    cv_results: Option<PyObject>,
}

#[pymethods]
impl PyGridSearchCV {
    /// Create a new GridSearchCV instance.
    ///
    /// Parameters
    /// ----------
    /// model : object
    ///     A FerroML model with `.fit(X, y)` and `.predict(X)` methods.
    /// param_grid : dict
    ///     Dictionary mapping parameter names to lists of values to try.
    /// cv : int, optional (default=5)
    ///     Number of cross-validation folds.
    /// scoring : str, optional (default="accuracy")
    ///     Scoring metric: "accuracy", "neg_mean_squared_error", or "r2".
    ///
    /// Returns
    /// -------
    /// GridSearchCV
    ///     A new GridSearchCV instance.
    #[new]
    #[pyo3(signature = (model, param_grid, cv=5, scoring="accuracy"))]
    fn new(model: PyObject, param_grid: PyObject, cv: usize, scoring: &str) -> Self {
        Self {
            model,
            param_grid,
            cv,
            scoring: scoring.to_string(),
            best_params: None,
            best_score: None,
            cv_results: None,
        }
    }

    /// Fit the grid search.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyObject,
        y: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let param_grid = slf.param_grid.bind(py).downcast::<PyDict>()?.clone();
        let model = slf.model.bind(py).clone();
        let cv = slf.cv;
        let scoring = slf.scoring.clone();

        // Extract parameter names and value lists
        let mut param_names: Vec<String> = Vec::new();
        let mut param_values: Vec<Vec<PyObject>> = Vec::new();
        for (key, value) in param_grid.iter() {
            param_names.push(key.extract::<String>()?);
            let values_list = value.downcast::<PyList>()?;
            let vals: Vec<PyObject> = values_list.iter().map(|v| v.unbind()).collect();
            param_values.push(vals);
        }

        // Generate all combinations (Cartesian product)
        let mut combinations: Vec<Vec<usize>> = vec![vec![]];
        for vals in &param_values {
            let mut new_combos = Vec::new();
            for combo in &combinations {
                for idx in 0..vals.len() {
                    let mut new_combo = combo.clone();
                    new_combo.push(idx);
                    new_combos.push(new_combo);
                }
            }
            combinations = new_combos;
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_params_dict: Option<Py<PyDict>> = None;
        let mut results: Vec<(Py<PyDict>, f64)> = Vec::new();

        let x_bound = x.bind(py);
        let y_bound = y.bind(py);

        for combo in &combinations {
            let params_dict = PyDict::new(py);
            for (i, &idx) in combo.iter().enumerate() {
                params_dict.set_item(&param_names[i], param_values[i][idx].bind(py))?;
            }

            let score = cross_validate(py, &model, x_bound, y_bound, &params_dict, cv, &scoring)?;

            let params_copy = params_dict.copy()?;
            results.push((params_copy.unbind(), score));

            if score > best_score {
                best_score = score;
                best_params_dict = Some(params_dict.copy()?.unbind());
            }
        }

        // Build cv_results list
        let cv_results_list = PyList::empty(py);
        for (params, score) in &results {
            let entry = PyDict::new(py);
            entry.set_item("params", params.bind(py))?;
            entry.set_item("mean_score", *score)?;
            cv_results_list.append(entry)?;
        }

        slf.best_score = Some(best_score);
        slf.best_params = best_params_dict.map(|p| p.into_any());
        slf.cv_results = Some(cv_results_list.into_any().unbind());

        Ok(slf)
    }

    /// Best parameters found by grid search.
    #[getter]
    fn best_params_(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.best_params
            .as_ref()
            .map(|p| p.clone_ref(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyAttributeError::new_err(
                    "GridSearchCV has not been fitted yet. Call .fit(X, y) first.",
                )
            })
    }

    /// Best mean cross-validated score.
    #[getter]
    fn best_score_(&self) -> PyResult<f64> {
        self.best_score.ok_or_else(|| {
            pyo3::exceptions::PyAttributeError::new_err(
                "GridSearchCV has not been fitted yet. Call .fit(X, y) first.",
            )
        })
    }

    /// Cross-validation results for all parameter combinations.
    #[getter]
    fn cv_results_(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.cv_results
            .as_ref()
            .map(|r| r.clone_ref(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyAttributeError::new_err(
                    "GridSearchCV has not been fitted yet. Call .fit(X, y) first.",
                )
            })
    }

    fn __repr__(&self) -> String {
        format!("GridSearchCV(cv={}, scoring='{}')", self.cv, self.scoring)
    }
}

// =============================================================================
// RandomSearchCV
// =============================================================================

/// Randomized search with cross-validation, sampling parameter combinations.
///
/// Parameters
/// ----------
/// model : object
///     A FerroML model with `.fit(X, y)` and `.predict(X)` methods.
/// param_distributions : dict
///     Dictionary mapping parameter names to lists of values to sample from.
/// n_iter : int, optional (default=10)
///     Number of parameter settings to sample.
/// cv : int, optional (default=5)
///     Number of cross-validation folds.
/// scoring : str, optional (default="accuracy")
///     Scoring metric: "accuracy", "neg_mean_squared_error", or "r2".
/// seed : int or None, optional (default=None)
///     Random seed for reproducibility.
#[pyclass(name = "RandomSearchCV")]
pub struct PyRandomSearchCV {
    model: PyObject,
    param_distributions: PyObject,
    n_iter: usize,
    cv: usize,
    scoring: String,
    seed: Option<u64>,
    best_params: Option<PyObject>,
    best_score: Option<f64>,
    cv_results: Option<PyObject>,
}

#[pymethods]
impl PyRandomSearchCV {
    /// Create a new RandomSearchCV instance.
    ///
    /// Parameters
    /// ----------
    /// model : object
    ///     A FerroML model with `.fit(X, y)` and `.predict(X)` methods.
    /// param_distributions : dict
    ///     Dictionary mapping parameter names to lists of values to sample from.
    /// n_iter : int, optional (default=10)
    ///     Number of parameter settings to sample.
    /// cv : int, optional (default=5)
    ///     Number of cross-validation folds.
    /// scoring : str, optional (default="accuracy")
    ///     Scoring metric: "accuracy", "neg_mean_squared_error", or "r2".
    /// seed : int or None, optional (default=None)
    ///     Random seed for reproducibility.
    ///
    /// Returns
    /// -------
    /// RandomSearchCV
    ///     A new RandomSearchCV instance.
    #[new]
    #[pyo3(signature = (model, param_distributions, n_iter=10, cv=5, scoring="accuracy", seed=None))]
    fn new(
        model: PyObject,
        param_distributions: PyObject,
        n_iter: usize,
        cv: usize,
        scoring: &str,
        seed: Option<u64>,
    ) -> Self {
        Self {
            model,
            param_distributions,
            n_iter,
            cv,
            scoring: scoring.to_string(),
            seed,
            best_params: None,
            best_score: None,
            cv_results: None,
        }
    }

    /// Fit the randomized search.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyObject,
        y: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let param_dist = slf
            .param_distributions
            .bind(py)
            .downcast::<PyDict>()?
            .clone();
        let model = slf.model.bind(py).clone();
        let n_iter = slf.n_iter;
        let cv = slf.cv;
        let scoring = slf.scoring.clone();
        let seed = slf.seed;

        // Extract parameter names and value lists
        let mut param_names: Vec<String> = Vec::new();
        let mut param_values: Vec<Vec<PyObject>> = Vec::new();
        for (key, value) in param_dist.iter() {
            param_names.push(key.extract::<String>()?);
            let values_list = value.downcast::<PyList>()?;
            let vals: Vec<PyObject> = values_list.iter().map(|v| v.unbind()).collect();
            param_values.push(vals);
        }

        // Use Python's random module for sampling
        let random_mod = py.import("random")?;
        if let Some(s) = seed {
            random_mod.call_method1("seed", (s,))?;
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_params_dict: Option<Py<PyDict>> = None;
        let mut results: Vec<(Py<PyDict>, f64)> = Vec::new();

        let x_bound = x.bind(py);
        let y_bound = y.bind(py);

        for _ in 0..n_iter {
            let params_dict = PyDict::new(py);
            for (i, name) in param_names.iter().enumerate() {
                let vals = &param_values[i];
                let idx_py = random_mod.call_method1("randint", (0i64, (vals.len() - 1) as i64))?;
                let idx: usize = idx_py.extract()?;
                params_dict.set_item(name, vals[idx].bind(py))?;
            }

            let score = cross_validate(py, &model, x_bound, y_bound, &params_dict, cv, &scoring)?;

            let params_copy = params_dict.copy()?;
            results.push((params_copy.unbind(), score));

            if score > best_score {
                best_score = score;
                best_params_dict = Some(params_dict.copy()?.unbind());
            }
        }

        // Build cv_results list
        let cv_results_list = PyList::empty(py);
        for (params, score) in &results {
            let entry = PyDict::new(py);
            entry.set_item("params", params.bind(py))?;
            entry.set_item("mean_score", *score)?;
            cv_results_list.append(entry)?;
        }

        slf.best_score = Some(best_score);
        slf.best_params = best_params_dict.map(|p| p.into_any());
        slf.cv_results = Some(cv_results_list.into_any().unbind());

        Ok(slf)
    }

    /// Best parameters found by randomized search.
    #[getter]
    fn best_params_(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.best_params
            .as_ref()
            .map(|p| p.clone_ref(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyAttributeError::new_err(
                    "RandomSearchCV has not been fitted yet. Call .fit(X, y) first.",
                )
            })
    }

    /// Best mean cross-validated score.
    #[getter]
    fn best_score_(&self) -> PyResult<f64> {
        self.best_score.ok_or_else(|| {
            pyo3::exceptions::PyAttributeError::new_err(
                "RandomSearchCV has not been fitted yet. Call .fit(X, y) first.",
            )
        })
    }

    /// Cross-validation results for all sampled combinations.
    #[getter]
    fn cv_results_(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.cv_results
            .as_ref()
            .map(|r| r.clone_ref(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyAttributeError::new_err(
                    "RandomSearchCV has not been fitted yet. Call .fit(X, y) first.",
                )
            })
    }

    fn __repr__(&self) -> String {
        format!(
            "RandomSearchCV(n_iter={}, cv={}, scoring='{}')",
            self.n_iter, self.cv, self.scoring
        )
    }
}

// =============================================================================
// Study (Optuna-like API)
// =============================================================================

/// Hyperparameter optimization study with Optuna-like interface.
///
/// Parameters
/// ----------
/// direction : str, optional (default="minimize")
///     Direction of optimization: "minimize" or "maximize".
/// seed : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Attributes
/// ----------
/// best_params : dict
///     Best parameters found so far.
/// best_value : float
///     Best objective value found so far.
/// trials : list[dict]
///     List of all trials with their params, values, and states.
#[pyclass(name = "Study")]
pub struct PyStudy {
    direction: String,
    seed: Option<u64>,
    trials_params: Vec<HashMap<String, PyObject>>,
    trials_values: Vec<Option<f64>>,
    trials_states: Vec<String>,
}

#[pymethods]
impl PyStudy {
    /// Create a new Study for hyperparameter optimization.
    ///
    /// Parameters
    /// ----------
    /// direction : str, optional (default="minimize")
    ///     Direction of optimization: "minimize" or "maximize".
    /// seed : int or None, optional (default=None)
    ///     Random seed for reproducibility.
    ///
    /// Returns
    /// -------
    /// Study
    ///     A new Study instance.
    #[new]
    #[pyo3(signature = (direction="minimize", seed=None))]
    fn new(direction: &str, seed: Option<u64>) -> PyResult<Self> {
        match direction {
            "minimize" | "maximize" => {}
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown direction '{}'. Use 'minimize' or 'maximize'.",
                    other
                )))
            }
        }
        Ok(Self {
            direction: direction.to_string(),
            seed,
            trials_params: Vec::new(),
            trials_values: Vec::new(),
            trials_states: Vec::new(),
        })
    }

    /// Run optimization by calling the objective function n_trials times.
    ///
    /// Parameters
    /// ----------
    /// objective : callable
    ///     Function that takes a dict and returns a float.
    /// n_trials : int
    ///     Number of trials to run.
    /// search_space : dict or None, optional
    ///     Optional search space definition for Rust-backed sampling.
    ///     Dict mapping param names to dicts with "type", "low"/"high"/"choices".
    ///
    /// Returns
    /// -------
    /// self
    #[pyo3(signature = (objective, n_trials, search_space=None))]
    fn optimize<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        objective: PyObject,
        n_trials: usize,
        search_space: Option<PyObject>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let direction_str = slf.direction.clone();
        let seed = slf.seed;

        if let Some(ref ss_obj) = search_space {
            // Use Rust-backed HPO with search space
            let ss_dict = ss_obj.bind(py).downcast::<PyDict>()?;
            let rust_ss = build_search_space(ss_dict)?;

            let direction = match direction_str.as_str() {
                "minimize" => ferroml_core::hpo::Direction::Minimize,
                _ => ferroml_core::hpo::Direction::Maximize,
            };

            let mut study = ferroml_core::hpo::Study::new("python_study", rust_ss, direction);

            let mut tpe = ferroml_core::hpo::TPESampler::new();
            if let Some(s) = seed {
                tpe = tpe.with_seed(s);
                study = study.with_seed(s);
            }
            study = study.with_sampler(tpe);

            for _ in 0..n_trials {
                let trial = study.ask().map_err(crate::errors::ferro_to_pyerr)?;
                let trial_id = trial.id;

                // Convert params to Python dict
                let params_py = PyDict::new(py);
                for (k, v) in &trial.params {
                    let py_val: PyObject = match v {
                        ferroml_core::hpo::ParameterValue::Int(i) => {
                            i.into_pyobject(py).unwrap().into_any().unbind()
                        }
                        ferroml_core::hpo::ParameterValue::Float(f) => {
                            f.into_pyobject(py).unwrap().into_any().unbind()
                        }
                        ferroml_core::hpo::ParameterValue::Categorical(s) => {
                            s.into_pyobject(py).unwrap().into_any().unbind()
                        }
                        ferroml_core::hpo::ParameterValue::Bool(b) => (*b)
                            .into_pyobject(py)
                            .unwrap()
                            .to_owned()
                            .into_any()
                            .unbind(),
                    };
                    params_py.set_item(k, py_val)?;
                }

                let result = objective.call1(py, (params_py,));
                match result {
                    Ok(val) => {
                        let value: f64 = val.extract(py)?;
                        study
                            .tell(trial_id, value)
                            .map_err(crate::errors::ferro_to_pyerr)?;

                        let mut param_map = HashMap::new();
                        for (k, v) in &trial.params {
                            let py_val: PyObject = match v {
                                ferroml_core::hpo::ParameterValue::Int(i) => {
                                    i.into_pyobject(py).unwrap().into_any().unbind()
                                }
                                ferroml_core::hpo::ParameterValue::Float(f) => {
                                    f.into_pyobject(py).unwrap().into_any().unbind()
                                }
                                ferroml_core::hpo::ParameterValue::Categorical(s) => {
                                    s.into_pyobject(py).unwrap().into_any().unbind()
                                }
                                ferroml_core::hpo::ParameterValue::Bool(b) => (*b)
                                    .into_pyobject(py)
                                    .unwrap()
                                    .to_owned()
                                    .into_any()
                                    .unbind(),
                            };
                            param_map.insert(k.clone(), py_val);
                        }
                        slf.trials_params.push(param_map);
                        slf.trials_values.push(Some(value));
                        slf.trials_states.push("complete".to_string());
                    }
                    Err(_) => {
                        slf.trials_params.push(HashMap::new());
                        slf.trials_values.push(None);
                        slf.trials_states.push("failed".to_string());
                    }
                }
            }
        } else {
            // No search space: call objective with trial dict
            for i in 0..n_trials {
                let trial_dict = PyDict::new(py);
                trial_dict.set_item("trial_id", i)?;

                let result = objective.call1(py, (trial_dict,));
                match result {
                    Ok(val) => {
                        let value: f64 = val.extract(py)?;
                        slf.trials_params.push(HashMap::new());
                        slf.trials_values.push(Some(value));
                        slf.trials_states.push("complete".to_string());
                    }
                    Err(_) => {
                        slf.trials_params.push(HashMap::new());
                        slf.trials_values.push(None);
                        slf.trials_states.push("failed".to_string());
                    }
                }
            }
        }

        Ok(slf)
    }

    /// Best parameters found so far.
    #[getter]
    fn best_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let maximize = self.direction == "maximize";
        let mut best_idx: Option<usize> = None;
        let mut best_val = if maximize {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };

        for (i, val) in self.trials_values.iter().enumerate() {
            if let Some(v) = val {
                if self.trials_states[i] == "complete" {
                    let is_better = if maximize {
                        *v > best_val
                    } else {
                        *v < best_val
                    };
                    if is_better {
                        best_val = *v;
                        best_idx = Some(i);
                    }
                }
            }
        }

        match best_idx {
            Some(idx) => {
                let dict = PyDict::new(py);
                for (k, v) in &self.trials_params[idx] {
                    dict.set_item(k, v.bind(py))?;
                }
                Ok(dict.into_any().unbind())
            }
            None => Err(pyo3::exceptions::PyAttributeError::new_err(
                "No completed trials yet.",
            )),
        }
    }

    /// Best objective value found so far.
    #[getter]
    fn best_value(&self) -> PyResult<f64> {
        let maximize = self.direction == "maximize";
        let mut best_val = if maximize {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        let mut found = false;

        for (i, val) in self.trials_values.iter().enumerate() {
            if let Some(v) = val {
                if self.trials_states[i] == "complete" {
                    let is_better = if maximize {
                        *v > best_val
                    } else {
                        *v < best_val
                    };
                    if is_better {
                        best_val = *v;
                        found = true;
                    }
                }
            }
        }

        if found {
            Ok(best_val)
        } else {
            Err(pyo3::exceptions::PyAttributeError::new_err(
                "No completed trials yet.",
            ))
        }
    }

    /// All trials as a list of dicts.
    #[getter]
    fn trials(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        for i in 0..self.trials_params.len() {
            let dict = PyDict::new(py);
            dict.set_item("id", i)?;

            let params_dict = PyDict::new(py);
            for (k, v) in &self.trials_params[i] {
                params_dict.set_item(k, v.bind(py))?;
            }
            dict.set_item("params", params_dict)?;
            dict.set_item("value", self.trials_values[i])?;
            dict.set_item("state", &self.trials_states[i])?;
            list.append(dict)?;
        }
        Ok(list.into_any().unbind())
    }

    /// Number of completed trials.
    #[getter]
    fn n_trials(&self) -> usize {
        self.trials_states
            .iter()
            .filter(|s| s.as_str() == "complete")
            .count()
    }

    fn __repr__(&self) -> String {
        format!(
            "Study(direction='{}', n_trials={})",
            self.direction,
            self.n_trials()
        )
    }
}

// =============================================================================
// SearchSpace builder helper
// =============================================================================

/// Build a Rust SearchSpace from a Python dict.
fn build_search_space(ss_dict: &Bound<'_, PyDict>) -> PyResult<ferroml_core::hpo::SearchSpace> {
    let mut ss = ferroml_core::hpo::SearchSpace::new();

    for (key, value) in ss_dict.iter() {
        let name: String = key.extract()?;
        let spec = value.downcast::<PyDict>()?;

        let param_type: String = spec
            .get_item("type")?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Parameter '{}' missing 'type' key",
                    name
                ))
            })?
            .extract()?;

        let log_scale: bool = spec
            .get_item("log")?
            .map(|v| v.extract::<bool>())
            .transpose()?
            .unwrap_or(false);

        let param = match param_type.as_str() {
            "int" => {
                let low: i64 = spec
                    .get_item("low")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Parameter '{}' missing 'low'",
                            name
                        ))
                    })?
                    .extract()?;
                let high: i64 = spec
                    .get_item("high")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Parameter '{}' missing 'high'",
                            name
                        ))
                    })?
                    .extract()?;
                if log_scale {
                    ferroml_core::hpo::search_space::Parameter::int_log(low, high)
                } else {
                    ferroml_core::hpo::search_space::Parameter::int(low, high)
                }
            }
            "float" => {
                let low: f64 = spec
                    .get_item("low")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Parameter '{}' missing 'low'",
                            name
                        ))
                    })?
                    .extract()?;
                let high: f64 = spec
                    .get_item("high")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Parameter '{}' missing 'high'",
                            name
                        ))
                    })?
                    .extract()?;
                if log_scale {
                    ferroml_core::hpo::search_space::Parameter::float_log(low, high)
                } else {
                    ferroml_core::hpo::search_space::Parameter::float(low, high)
                }
            }
            "categorical" => {
                let choices: Vec<String> = spec
                    .get_item("choices")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Parameter '{}' missing 'choices'",
                            name
                        ))
                    })?
                    .extract()?;
                ferroml_core::hpo::search_space::Parameter::categorical(choices)
            }
            "bool" => ferroml_core::hpo::search_space::Parameter::bool(),
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown parameter type '{}' for '{}'. Use 'int', 'float', 'categorical', or 'bool'.",
                    other, name
                )));
            }
        };

        ss = ss.add(name, param);
    }

    Ok(ss)
}

// =============================================================================
// Module Registration
// =============================================================================

/// Register the hpo submodule
pub fn register_hpo_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "hpo")?;
    m.add_class::<PyGridSearchCV>()?;
    m.add_class::<PyRandomSearchCV>()?;
    m.add_class::<PyStudy>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

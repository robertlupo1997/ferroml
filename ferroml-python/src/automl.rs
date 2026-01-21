//! Python bindings for FerroML AutoML
//!
//! This module provides Python wrappers for:
//! - AutoMLConfig
//! - AutoML
//! - AutoMLResult
//! - LeaderboardEntry
//! - EnsembleResult
//!
//! ## Zero-Copy Semantics
//!
//! Input arrays are passed as `PyReadonlyArray` which provides read-only access.
//! When calling core Rust functions that require owned arrays (e.g., `AutoML::fit`),
//! a copy is made. See `crate::array_utils` for detailed documentation.

use crate::array_utils::{to_owned_array_1d, to_owned_array_2d};
use ferroml_core::{
    automl::{AlgorithmType, EnsembleResult, ParamValue},
    AutoML, AutoMLConfig, AutoMLResult, LeaderboardEntry, Metric, MultipleTesting, Task,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Convert Task enum from Python string
fn task_from_str(s: &str) -> PyResult<Task> {
    match s.to_lowercase().as_str() {
        "classification" => Ok(Task::Classification),
        "regression" => Ok(Task::Regression),
        "timeseries" | "time_series" => Ok(Task::TimeSeries),
        "survival" => Ok(Task::Survival),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown task type: '{}'. Expected one of: classification, regression, timeseries, survival",
            s
        ))),
    }
}

/// Convert Metric enum from Python string
fn metric_from_str(s: &str) -> PyResult<Metric> {
    match s.to_lowercase().as_str() {
        // Classification
        "roc_auc" | "rocauc" | "auc" => Ok(Metric::RocAuc),
        "accuracy" | "acc" => Ok(Metric::Accuracy),
        "f1" | "f1_score" => Ok(Metric::F1),
        "log_loss" | "logloss" => Ok(Metric::LogLoss),
        "mcc" | "matthews" => Ok(Metric::Mcc),
        // Regression
        "mse" | "mean_squared_error" => Ok(Metric::Mse),
        "rmse" | "root_mean_squared_error" => Ok(Metric::Rmse),
        "mae" | "mean_absolute_error" => Ok(Metric::Mae),
        "r2" | "r_squared" | "r2_score" => Ok(Metric::R2),
        "mape" | "mean_absolute_percentage_error" => Ok(Metric::Mape),
        // Time series
        "smape" | "symmetric_mape" => Ok(Metric::Smape),
        "mase" => Ok(Metric::Mase),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown metric: '{}'. Expected one of: roc_auc, accuracy, f1, log_loss, mcc, mse, rmse, mae, r2, mape, smape, mase",
            s
        ))),
    }
}

/// Convert multiple testing correction from Python string
fn multiple_testing_from_str(s: &str) -> MultipleTesting {
    match s.to_lowercase().as_str() {
        "bonferroni" => MultipleTesting::Bonferroni,
        "holm" | "holm-bonferroni" => MultipleTesting::Holm,
        "bh" | "benjamini-hochberg" | "fdr" => MultipleTesting::BenjaminiHochberg,
        "by" | "benjamini-yekutieli" => MultipleTesting::BenjaminiYekutieli,
        "none" | "no" => MultipleTesting::None,
        _ => MultipleTesting::BenjaminiHochberg, // Default
    }
}

/// Convert AlgorithmType to string
fn algorithm_type_to_str(algo: AlgorithmType) -> &'static str {
    match algo {
        AlgorithmType::LogisticRegression => "LogisticRegression",
        AlgorithmType::GaussianNB => "GaussianNB",
        AlgorithmType::MultinomialNB => "MultinomialNB",
        AlgorithmType::KNeighborsClassifier => "KNeighborsClassifier",
        AlgorithmType::SVC => "SVC",
        AlgorithmType::LinearSVC => "LinearSVC",
        AlgorithmType::DecisionTreeClassifier => "DecisionTreeClassifier",
        AlgorithmType::RandomForestClassifier => "RandomForestClassifier",
        AlgorithmType::GradientBoostingClassifier => "GradientBoostingClassifier",
        AlgorithmType::HistGradientBoostingClassifier => "HistGradientBoostingClassifier",
        AlgorithmType::LinearRegression => "LinearRegression",
        AlgorithmType::Ridge => "Ridge",
        AlgorithmType::Lasso => "Lasso",
        AlgorithmType::ElasticNet => "ElasticNet",
        AlgorithmType::KNeighborsRegressor => "KNeighborsRegressor",
        AlgorithmType::SVR => "SVR",
        AlgorithmType::LinearSVR => "LinearSVR",
        AlgorithmType::DecisionTreeRegressor => "DecisionTreeRegressor",
        AlgorithmType::RandomForestRegressor => "RandomForestRegressor",
        AlgorithmType::GradientBoostingRegressor => "GradientBoostingRegressor",
        AlgorithmType::HistGradientBoostingRegressor => "HistGradientBoostingRegressor",
        AlgorithmType::QuantileRegression => "QuantileRegression",
        AlgorithmType::RobustRegression => "RobustRegression",
    }
}

/// Convert ParamValue to Python object
fn param_value_to_py(py: Python<'_>, value: &ParamValue) -> PyObject {
    match value {
        ParamValue::Int(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        ParamValue::Float(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        ParamValue::String(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        ParamValue::Bool(v) => (*v).into_pyobject(py).unwrap().to_owned().into_any().unbind(),
    }
}

// =============================================================================
// PyAutoMLConfig
// =============================================================================

/// AutoML Configuration.
///
/// Parameters
/// ----------
/// task : str
///     The machine learning task type. One of: "classification", "regression".
/// metric : str
///     The optimization metric. For classification: "roc_auc", "accuracy", "f1",
///     "log_loss", "mcc". For regression: "mse", "rmse", "mae", "r2", "mape".
/// time_budget_seconds : int, optional (default=3600)
///     Time budget for the AutoML search in seconds.
/// cv_folds : int, optional (default=5)
///     Number of cross-validation folds.
/// statistical_tests : bool, optional (default=True)
///     Whether to run statistical tests comparing models.
/// confidence_level : float, optional (default=0.95)
///     Confidence level for intervals.
/// multiple_testing : str, optional (default="bh")
///     Multiple testing correction method. One of: "bonferroni", "holm", "bh", "by", "none".
/// seed : int, optional
///     Random seed for reproducibility.
/// n_jobs : int, optional
///     Number of parallel jobs (default: number of CPUs).
///
/// Examples
/// --------
/// >>> from ferroml.automl import AutoMLConfig, AutoML
/// >>> config = AutoMLConfig(
/// ...     task="classification",
/// ...     metric="roc_auc",
/// ...     time_budget_seconds=300,
/// ... )
/// >>> automl = AutoML(config)
#[pyclass(name = "AutoMLConfig", module = "ferroml.automl")]
#[derive(Clone)]
pub struct PyAutoMLConfig {
    /// Task type string
    pub task: String,
    /// Metric string
    pub metric: String,
    /// Time budget in seconds
    pub time_budget_seconds: u64,
    /// Number of CV folds
    pub cv_folds: usize,
    /// Whether to run statistical tests
    pub statistical_tests: bool,
    /// Confidence level
    pub confidence_level: f64,
    /// Multiple testing correction
    pub multiple_testing: String,
    /// Random seed
    pub seed: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: usize,
}

#[pymethods]
impl PyAutoMLConfig {
    /// Create a new AutoMLConfig.
    #[new]
    #[pyo3(signature = (task="classification", metric="roc_auc", time_budget_seconds=3600, cv_folds=5, statistical_tests=true, confidence_level=0.95, multiple_testing="bh", seed=None, n_jobs=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        task: &str,
        metric: &str,
        time_budget_seconds: u64,
        cv_folds: usize,
        statistical_tests: bool,
        confidence_level: f64,
        multiple_testing: &str,
        seed: Option<u64>,
        n_jobs: Option<usize>,
    ) -> PyResult<Self> {
        // Validate task and metric
        task_from_str(task)?;
        metric_from_str(metric)?;

        let n_jobs = n_jobs.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
        });

        Ok(Self {
            task: task.to_string(),
            metric: metric.to_string(),
            time_budget_seconds,
            cv_folds,
            statistical_tests,
            confidence_level,
            multiple_testing: multiple_testing.to_string(),
            seed,
            n_jobs,
        })
    }

    /// Get the task type.
    #[getter]
    fn get_task(&self) -> &str {
        &self.task
    }

    /// Get the metric.
    #[getter]
    fn get_metric(&self) -> &str {
        &self.metric
    }

    /// Get the time budget.
    #[getter]
    fn get_time_budget_seconds(&self) -> u64 {
        self.time_budget_seconds
    }

    /// Get the number of CV folds.
    #[getter]
    fn get_cv_folds(&self) -> usize {
        self.cv_folds
    }

    /// Get whether statistical tests are enabled.
    #[getter]
    fn get_statistical_tests(&self) -> bool {
        self.statistical_tests
    }

    /// Get the confidence level.
    #[getter]
    fn get_confidence_level(&self) -> f64 {
        self.confidence_level
    }

    /// Get the seed.
    #[getter]
    fn get_seed(&self) -> Option<u64> {
        self.seed
    }

    fn __repr__(&self) -> String {
        format!(
            "AutoMLConfig(task='{}', metric='{}', time_budget_seconds={}, cv_folds={}, statistical_tests={})",
            self.task, self.metric, self.time_budget_seconds, self.cv_folds, self.statistical_tests
        )
    }
}

impl PyAutoMLConfig {
    /// Convert to Rust AutoMLConfig
    fn to_rust_config(&self) -> PyResult<AutoMLConfig> {
        Ok(AutoMLConfig {
            task: task_from_str(&self.task)?,
            metric: metric_from_str(&self.metric)?,
            time_budget_seconds: self.time_budget_seconds,
            cv_folds: self.cv_folds,
            statistical_tests: self.statistical_tests,
            confidence_level: self.confidence_level,
            multiple_testing_correction: multiple_testing_from_str(&self.multiple_testing),
            seed: self.seed,
            n_jobs: self.n_jobs,
        })
    }
}

// =============================================================================
// PyLeaderboardEntry
// =============================================================================

/// Entry in the AutoML leaderboard.
///
/// Attributes
/// ----------
/// rank : int
///     Rank in the leaderboard (1 = best).
/// trial_id : int
///     Unique trial identifier.
/// algorithm : str
///     Algorithm name.
/// cv_score : float
///     Mean cross-validation score.
/// cv_std : float
///     Standard deviation of CV scores.
/// ci_lower : float
///     95% confidence interval lower bound.
/// ci_upper : float
///     95% confidence interval upper bound.
/// training_time_seconds : float
///     Time taken to train this model.
/// params : dict
///     Hyperparameters used.
#[pyclass(name = "LeaderboardEntry", module = "ferroml.automl")]
pub struct PyLeaderboardEntry {
    inner: LeaderboardEntry,
}

#[pymethods]
impl PyLeaderboardEntry {
    /// Get the rank.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank
    }

    /// Get the trial ID.
    #[getter]
    fn trial_id(&self) -> usize {
        self.inner.trial_id
    }

    /// Get the algorithm name.
    #[getter]
    fn algorithm(&self) -> &str {
        algorithm_type_to_str(self.inner.algorithm)
    }

    /// Get the CV score (mean).
    #[getter]
    fn cv_score(&self) -> f64 {
        self.inner.cv_score
    }

    /// Get the CV score standard deviation.
    #[getter]
    fn cv_std(&self) -> f64 {
        self.inner.cv_std
    }

    /// Get the 95% CI lower bound.
    #[getter]
    fn ci_lower(&self) -> f64 {
        self.inner.ci_lower
    }

    /// Get the 95% CI upper bound.
    #[getter]
    fn ci_upper(&self) -> f64 {
        self.inner.ci_upper
    }

    /// Get the training time in seconds.
    #[getter]
    fn training_time_seconds(&self) -> f64 {
        self.inner.training_time_seconds
    }

    /// Get the hyperparameters as a dictionary.
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in &self.inner.params {
            dict.set_item(key, param_value_to_py(py, value))?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "LeaderboardEntry(rank={}, algorithm='{}', cv_score={:.4} ± {:.4})",
            self.inner.rank,
            algorithm_type_to_str(self.inner.algorithm),
            self.inner.cv_score,
            self.inner.cv_std
        )
    }
}

// =============================================================================
// PyEnsembleMember
// =============================================================================

/// A member of the ensemble with its weight.
///
/// Attributes
/// ----------
/// trial_id : int
///     Trial ID of this ensemble member.
/// algorithm : str
///     Algorithm name.
/// weight : float
///     Weight of this member in the ensemble.
/// selection_count : int
///     Number of times this model was selected during greedy ensemble construction.
#[pyclass(name = "EnsembleMember", module = "ferroml.automl")]
pub struct PyEnsembleMember {
    trial_id: usize,
    algorithm: AlgorithmType,
    weight: f64,
    selection_count: usize,
}

#[pymethods]
impl PyEnsembleMember {
    /// Get the trial ID.
    #[getter]
    fn trial_id(&self) -> usize {
        self.trial_id
    }

    /// Get the algorithm name.
    #[getter]
    fn algorithm(&self) -> &str {
        algorithm_type_to_str(self.algorithm)
    }

    /// Get the weight.
    #[getter]
    fn weight(&self) -> f64 {
        self.weight
    }

    /// Get the selection count.
    #[getter]
    fn selection_count(&self) -> usize {
        self.selection_count
    }

    fn __repr__(&self) -> String {
        format!(
            "EnsembleMember(algorithm='{}', weight={:.4}, selection_count={})",
            algorithm_type_to_str(self.algorithm),
            self.weight,
            self.selection_count
        )
    }
}

// =============================================================================
// PyEnsembleResult
// =============================================================================

/// Result of ensemble construction.
///
/// Attributes
/// ----------
/// ensemble_score : float
///     Cross-validation score of the ensemble.
/// improvement : float
///     Improvement over the best single model.
/// members : list of EnsembleMember
///     Ensemble members with their weights.
/// n_models : int
///     Number of models in the ensemble.
#[pyclass(name = "EnsembleResult", module = "ferroml.automl")]
pub struct PyEnsembleResult {
    inner: EnsembleResult,
}

#[pymethods]
impl PyEnsembleResult {
    /// Get the ensemble score.
    #[getter]
    fn ensemble_score(&self) -> f64 {
        self.inner.ensemble_score
    }

    /// Get the improvement over best single model.
    #[getter]
    fn improvement(&self) -> f64 {
        self.inner.improvement
    }

    /// Get the number of models in the ensemble.
    #[getter]
    fn n_models(&self) -> usize {
        self.inner.members.len()
    }

    /// Get the ensemble members.
    #[getter]
    fn members(&self) -> Vec<PyEnsembleMember> {
        self.inner
            .members
            .iter()
            .map(|m| PyEnsembleMember {
                trial_id: m.trial_id,
                algorithm: m.algorithm,
                weight: m.weight,
                selection_count: m.selection_count,
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "EnsembleResult(score={:.4}, improvement={:.4}, n_models={})",
            self.inner.ensemble_score,
            self.inner.improvement,
            self.inner.members.len()
        )
    }
}

// =============================================================================
// PyAutoMLResult
// =============================================================================

/// Result of running AutoML.fit().
///
/// This comprehensive result object provides:
/// - Leaderboard with confidence intervals for all successful models
/// - Ensemble composition and performance
/// - Aggregated feature importance across models
/// - Statistical significance tests comparing models
///
/// Attributes
/// ----------
/// leaderboard : list of LeaderboardEntry
///     All successful models sorted by score (best first).
/// ensemble : EnsembleResult or None
///     Ensemble result if one was built.
/// total_time_seconds : float
///     Total time spent in AutoML search.
/// n_successful_trials : int
///     Number of successful trials.
/// n_failed_trials : int
///     Number of failed trials.
/// task : str
///     Task type used.
/// metric_name : str
///     Metric used for optimization.
///
/// Examples
/// --------
/// >>> result = automl.fit(X, y)
/// >>> print(result.summary())
/// >>> best = result.best_model()
/// >>> print(f"Best model: {best.algorithm} with score {best.cv_score:.4f}")
#[pyclass(name = "AutoMLResult", module = "ferroml.automl")]
pub struct PyAutoMLResult {
    inner: AutoMLResult,
}

#[pymethods]
impl PyAutoMLResult {
    /// Get the leaderboard (list of LeaderboardEntry).
    #[getter]
    fn leaderboard(&self) -> Vec<PyLeaderboardEntry> {
        self.inner
            .leaderboard
            .iter()
            .map(|e| PyLeaderboardEntry { inner: e.clone() })
            .collect()
    }

    /// Get the ensemble result (if available).
    #[getter]
    fn ensemble(&self) -> Option<PyEnsembleResult> {
        self.inner
            .ensemble
            .clone()
            .map(|e| PyEnsembleResult { inner: e })
    }

    /// Get the total time in seconds.
    #[getter]
    fn total_time_seconds(&self) -> f64 {
        self.inner.total_time_seconds
    }

    /// Get the number of successful trials.
    #[getter]
    fn n_successful_trials(&self) -> usize {
        self.inner.n_successful_trials
    }

    /// Get the number of failed trials.
    #[getter]
    fn n_failed_trials(&self) -> usize {
        self.inner.n_failed_trials
    }

    /// Get the task type.
    #[getter]
    fn task(&self) -> &str {
        match self.inner.task {
            Task::Classification => "classification",
            Task::Regression => "regression",
            Task::TimeSeries => "timeseries",
            Task::Survival => "survival",
        }
    }

    /// Get the metric name.
    #[getter]
    fn metric_name(&self) -> &str {
        &self.inner.metric_name
    }

    /// Whether higher metric values are better.
    #[getter]
    fn maximize(&self) -> bool {
        self.inner.maximize
    }

    /// Number of CV folds used.
    #[getter]
    fn cv_folds(&self) -> usize {
        self.inner.cv_folds
    }

    /// Get the best model from the leaderboard.
    ///
    /// Returns
    /// -------
    /// LeaderboardEntry or None
    ///     The best model entry, or None if no models succeeded.
    fn best_model(&self) -> Option<PyLeaderboardEntry> {
        self.inner
            .best_model()
            .map(|e| PyLeaderboardEntry { inner: e.clone() })
    }

    /// Get the ensemble score (if ensemble was built).
    ///
    /// Returns
    /// -------
    /// float or None
    ///     The ensemble score, or None if no ensemble was built.
    fn ensemble_score(&self) -> Option<f64> {
        self.inner.ensemble_score()
    }

    /// Get improvement from ensemble over best single model.
    ///
    /// Returns
    /// -------
    /// float or None
    ///     The improvement value, or None if no ensemble was built.
    fn ensemble_improvement(&self) -> Option<f64> {
        self.inner.ensemble_improvement()
    }

    /// Check if the AutoML run was successful.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if at least one trial succeeded.
    fn is_successful(&self) -> bool {
        self.inner.is_successful()
    }

    /// Get the number of unique algorithms that succeeded.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of unique successful algorithms.
    fn n_successful_algorithms(&self) -> usize {
        self.inner.n_successful_algorithms()
    }

    /// Get top k most important features.
    ///
    /// Parameters
    /// ----------
    /// k : int
    ///     Number of top features to return.
    ///
    /// Returns
    /// -------
    /// list of tuple or None
    ///     List of (feature_name, importance, ci_lower, ci_upper) tuples,
    ///     or None if feature importance wasn't computed.
    #[pyo3(signature = (k=5))]
    fn top_features<'py>(&self, py: Python<'py>, k: usize) -> Option<Bound<'py, PyList>> {
        self.inner.top_features(k).map(|features| {
            let list = PyList::empty(py);
            for (name, imp, ci_l, ci_u) in features {
                let tuple = (name, imp, ci_l, ci_u);
                list.append(tuple).unwrap();
            }
            list
        })
    }

    /// Get models that are statistically competitive with the best.
    ///
    /// Returns
    /// -------
    /// list of LeaderboardEntry
    ///     Models not significantly worse than the best.
    fn competitive_models(&self) -> Vec<PyLeaderboardEntry> {
        self.inner
            .competitive_models()
            .into_iter()
            .map(|e| PyLeaderboardEntry { inner: e.clone() })
            .collect()
    }

    /// Check if two models are significantly different.
    ///
    /// Parameters
    /// ----------
    /// trial_id_1 : int
    ///     First model's trial ID.
    /// trial_id_2 : int
    ///     Second model's trial ID.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if models are significantly different.
    fn models_significantly_different(&self, trial_id_1: usize, trial_id_2: usize) -> bool {
        self.inner
            .models_significantly_different(trial_id_1, trial_id_2)
    }

    /// Get a comprehensive summary of the AutoML results.
    ///
    /// Returns
    /// -------
    /// str
    ///     Formatted summary string.
    fn summary(&self) -> String {
        self.inner.summary()
    }

    /// Get the best model statistics.
    ///
    /// Returns
    /// -------
    /// dict or None
    ///     Dictionary with mean_score, std_score, ci_lower, ci_upper, n_folds, fold_scores.
    fn best_model_stats<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match &self.inner.best_model_stats {
            Some(stats) => {
                let dict = PyDict::new(py);
                dict.set_item("mean_score", stats.mean_score)?;
                dict.set_item("std_score", stats.std_score)?;
                dict.set_item("ci_lower", stats.ci_lower)?;
                dict.set_item("ci_upper", stats.ci_upper)?;
                dict.set_item("n_folds", stats.n_folds)?;
                dict.set_item("fold_scores", &stats.fold_scores)?;
                Ok(Some(dict))
            }
            None => Ok(None),
        }
    }

    /// Get aggregated feature importance.
    ///
    /// Returns
    /// -------
    /// dict or None
    ///     Dictionary with feature_names, importance_mean, importance_std, ci_lower, ci_upper.
    fn aggregated_importance<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match &self.inner.aggregated_importance {
            Some(imp) => {
                let dict = PyDict::new(py);
                dict.set_item("feature_names", &imp.feature_names)?;
                dict.set_item("importance_mean", &imp.importance_mean)?;
                dict.set_item("importance_std", &imp.importance_std)?;
                dict.set_item("ci_lower", &imp.ci_lower)?;
                dict.set_item("ci_upper", &imp.ci_upper)?;
                dict.set_item("n_models", imp.n_models)?;
                Ok(Some(dict))
            }
            None => Ok(None),
        }
    }

    /// Get model comparison results.
    ///
    /// Returns
    /// -------
    /// dict or None
    ///     Dictionary with comparison statistics and pairwise results.
    fn model_comparisons<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match &self.inner.model_comparisons {
            Some(comp) => {
                let dict = PyDict::new(py);
                dict.set_item("best_is_significantly_better", comp.best_is_significantly_better)?;
                dict.set_item("n_significantly_worse", comp.n_significantly_worse)?;
                dict.set_item("corrected_alpha", comp.corrected_alpha)?;
                dict.set_item("correction_method", &comp.correction_method)?;

                // Pairwise comparisons
                let pairwise = PyList::empty(py);
                for c in &comp.pairwise_comparisons {
                    let pair_dict = PyDict::new(py);
                    pair_dict.set_item("trial_id_1", c.trial_id_1)?;
                    pair_dict.set_item("trial_id_2", c.trial_id_2)?;
                    pair_dict.set_item("model1_name", &c.model1_name)?;
                    pair_dict.set_item("model2_name", &c.model2_name)?;
                    pair_dict.set_item("mean_difference", c.mean_difference)?;
                    pair_dict.set_item("p_value", c.p_value)?;
                    pair_dict.set_item("p_value_corrected", c.p_value_corrected)?;
                    pair_dict.set_item("ci_lower", c.ci_lower)?;
                    pair_dict.set_item("ci_upper", c.ci_upper)?;
                    pair_dict.set_item("significant", c.significant)?;
                    pair_dict.set_item("significant_corrected", c.significant_corrected)?;
                    pairwise.append(pair_dict)?;
                }
                dict.set_item("pairwise_comparisons", pairwise)?;

                Ok(Some(dict))
            }
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        let best_info = if let Some(best) = self.inner.best_model() {
            format!(
                "best={}({:.4})",
                algorithm_type_to_str(best.algorithm),
                best.cv_score
            )
        } else {
            "no successful models".to_string()
        };
        format!(
            "AutoMLResult(task='{}', metric='{}', {}, n_trials={}, time={:.1}s)",
            self.task(),
            self.inner.metric_name,
            best_info,
            self.inner.n_successful_trials,
            self.inner.total_time_seconds
        )
    }
}

// =============================================================================
// PyAutoML
// =============================================================================

/// Automated Machine Learning.
///
/// AutoML automatically searches for the best machine learning pipeline by:
/// - Selecting algorithms from a portfolio
/// - Tuning hyperparameters
/// - Building ensembles of top models
/// - Providing statistical significance tests
///
/// Parameters
/// ----------
/// config : AutoMLConfig or None
///     Configuration for the AutoML search. If None, uses default configuration.
///
/// Attributes
/// ----------
/// config : AutoMLConfig
///     The configuration used.
///
/// Examples
/// --------
/// >>> from ferroml.automl import AutoML, AutoMLConfig
/// >>> import numpy as np
/// >>>
/// >>> # Create configuration
/// >>> config = AutoMLConfig(
/// ...     task="classification",
/// ...     metric="roc_auc",
/// ...     time_budget_seconds=300,
/// ...     cv_folds=5,
/// ... )
/// >>>
/// >>> # Create AutoML instance and fit
/// >>> automl = AutoML(config)
/// >>> X = np.random.randn(100, 10)
/// >>> y = (X[:, 0] > 0).astype(float)
/// >>> result = automl.fit(X, y)
/// >>>
/// >>> # Check results
/// >>> print(result.summary())
/// >>> best = result.best_model()
/// >>> print(f"Best: {best.algorithm} (score: {best.cv_score:.4f})")
#[pyclass(name = "AutoML", module = "ferroml.automl")]
pub struct PyAutoML {
    config: PyAutoMLConfig,
}

#[pymethods]
impl PyAutoML {
    /// Create a new AutoML instance.
    ///
    /// Parameters
    /// ----------
    /// config : AutoMLConfig or None
    ///     Configuration for AutoML. Uses defaults if None.
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyAutoMLConfig>) -> PyResult<Self> {
        let config = config.unwrap_or_else(|| {
            PyAutoMLConfig {
                task: "classification".to_string(),
                metric: "roc_auc".to_string(),
                time_budget_seconds: 3600,
                cv_folds: 5,
                statistical_tests: true,
                confidence_level: 0.95,
                multiple_testing: "bh".to_string(),
                seed: None,
                n_jobs: std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(1),
            }
        });
        Ok(Self { config })
    }

    /// Get the configuration.
    #[getter]
    fn config(&self) -> PyAutoMLConfig {
        self.config.clone()
    }

    /// Fit AutoML to training data.
    ///
    /// This method orchestrates the full AutoML pipeline:
    /// 1. Analyzes data characteristics
    /// 2. Selects and adapts algorithm portfolio
    /// 3. Runs cross-validation for each algorithm
    /// 4. Builds ensemble from successful trials
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training features.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// AutoMLResult
    ///     Comprehensive result object with leaderboard, ensemble, and diagnostics.
    ///
    /// Examples
    /// --------
    /// >>> result = automl.fit(X_train, y_train)
    /// >>> print(result.summary())
    fn fit(
        &self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<PyAutoMLResult> {
        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);

        let rust_config = self.config.to_rust_config()?;
        let automl = AutoML::new(rust_config);

        let result = automl
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(PyAutoMLResult { inner: result })
    }

    fn __repr__(&self) -> String {
        format!(
            "AutoML(task='{}', metric='{}', time_budget={}s)",
            self.config.task, self.config.metric, self.config.time_budget_seconds
        )
    }
}

// =============================================================================
// Module registration
// =============================================================================

/// Register the automl submodule.
pub fn register_automl_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let automl_module = PyModule::new(parent_module.py(), "automl")?;

    automl_module.add_class::<PyAutoMLConfig>()?;
    automl_module.add_class::<PyAutoML>()?;
    automl_module.add_class::<PyAutoMLResult>()?;
    automl_module.add_class::<PyLeaderboardEntry>()?;
    automl_module.add_class::<PyEnsembleResult>()?;
    automl_module.add_class::<PyEnsembleMember>()?;

    parent_module.add_submodule(&automl_module)?;

    Ok(())
}

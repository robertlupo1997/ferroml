//! Python bindings for FerroML
//!
//! This crate provides Python bindings via PyO3 for the FerroML AutoML library.
//!
//! ## Submodules
//!
//! - `ferroml.linear` - Linear models (LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet)
//! - `ferroml.trees` - Tree models (DecisionTree, RandomForest, GradientBoosting, HistGradientBoosting)
//! - `ferroml.ensemble` - Ensemble models (ExtraTrees, AdaBoost, SGD, PassiveAggressive)
//! - `ferroml.neighbors` - Nearest neighbors (KNeighborsClassifier, KNeighborsRegressor)
//! - `ferroml.clustering` - Clustering algorithms (KMeans, DBSCAN, AgglomerativeClustering) and metrics
//! - `ferroml.preprocessing` - Preprocessing transformers (Scalers, Encoders, Imputers, Selectors, Power, Quantile)
//! - `ferroml.decomposition` - Dimensionality reduction (PCA, IncrementalPCA, TruncatedSVD, LDA, FactorAnalysis)
//! - `ferroml.explainability` - Model explainability (TreeSHAP, permutation importance, partial dependence)
//! - `ferroml.pipeline` - Pipeline, ColumnTransformer, FeatureUnion
//! - `ferroml.automl` - AutoML (AutoML, AutoMLConfig, AutoMLResult)
//! - `ferroml.datasets` - Dataset loading (HuggingFace Hub, built-in datasets, synthetic generators)
//!
//! ## Zero-Copy Array Handling
//!
//! FerroML uses optimized array handling between Python and Rust:
//!
//! - **Input arrays**: Uses `PyReadonlyArray` for read-only access. When the Rust API
//!   requires owned arrays (e.g., `Model::fit`), a copy is made. For read-only inspection,
//!   zero-copy views are used where possible.
//!
//! - **Output arrays**: Uses `into_pyarray()` to transfer ownership to Python without
//!   copying the underlying data (only metadata allocation).
//!
//! See the `array_utils` module for detailed documentation on zero-copy semantics.

// Clippy allows for Python binding crate
#![allow(clippy::too_many_arguments)] // PyO3 functions mirror sklearn API signatures
#![allow(clippy::only_used_in_recursion)] // PyO3 method patterns
#![allow(clippy::empty_line_after_doc_comments)] // PyO3 macro formatting
#![allow(dead_code)] // Utility functions may not all be used yet

use pyo3::prelude::*;

mod anomaly;
mod array_utils;
mod automl;
mod calibration;
mod clustering;
mod datasets;
mod decomposition;
mod ensemble;
pub(crate) mod errors;
mod explainability;
mod gaussian_process;
mod linear;
mod multioutput;
mod naive_bayes;
mod neighbors;
mod neural;
#[cfg(feature = "pandas")]
pub mod pandas_utils;
pub mod pickle;
mod pipeline;
#[cfg(feature = "polars")]
pub mod polars_utils;
mod preprocessing;
#[cfg(feature = "sparse")]
pub mod sparse_utils;
mod svm;
pub(crate) mod trees;

/// FerroML Python module
#[pymodule]
fn ferroml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Register submodules
    linear::register_linear_module(m)?;
    trees::register_trees_module(m)?;
    ensemble::register_ensemble_module(m)?;
    neighbors::register_neighbors_module(m)?;
    naive_bayes::register_naive_bayes_module(m)?;
    neural::register_neural_module(m)?;
    clustering::register_clustering_module(m)?;
    preprocessing::register_preprocessing_module(m)?;
    decomposition::register_decomposition_module(m)?;
    explainability::register_explainability_module(m)?;
    pipeline::register_pipeline_module(m)?;
    automl::register_automl_module(m)?;
    datasets::register_datasets_module(m)?;
    svm::register_svm_module(m)?;
    calibration::register_calibration_module(m)?;
    anomaly::register_anomaly_module(m)?;
    gaussian_process::register_gaussian_process_module(m)?;
    multioutput::register_multioutput_module(m)?;

    Ok(())
}

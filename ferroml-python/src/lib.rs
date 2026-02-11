//! Python bindings for FerroML
//!
//! This crate provides Python bindings via PyO3 for the FerroML AutoML library.
//!
//! ## Submodules
//!
//! - `ferroml.linear` - Linear models (LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet)
//! - `ferroml.trees` - Tree models (DecisionTree, RandomForest, GradientBoosting, HistGradientBoosting)
//! - `ferroml.neighbors` - Nearest neighbors (KNeighborsClassifier, KNeighborsRegressor)
//! - `ferroml.clustering` - Clustering algorithms (KMeans, DBSCAN) and metrics
//! - `ferroml.preprocessing` - Preprocessing transformers (Scalers, Encoders, Imputers)
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

mod array_utils;
mod automl;
mod clustering;
mod datasets;
mod linear;
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
mod trees;

/// FerroML Python module
#[pymodule]
fn ferroml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Register submodules
    linear::register_linear_module(m)?;
    trees::register_trees_module(m)?;
    neighbors::register_neighbors_module(m)?;
    neural::register_neural_module(m)?;
    clustering::register_clustering_module(m)?;
    preprocessing::register_preprocessing_module(m)?;
    pipeline::register_pipeline_module(m)?;
    automl::register_automl_module(m)?;
    datasets::register_datasets_module(m)?;

    Ok(())
}

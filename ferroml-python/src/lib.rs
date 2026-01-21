//! Python bindings for FerroML
//!
//! This crate provides Python bindings via PyO3 for the FerroML AutoML library.
//!
//! ## Submodules
//!
//! - `ferroml.linear` - Linear models (LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet)
//! - `ferroml.trees` - Tree models (DecisionTree, RandomForest, GradientBoosting, HistGradientBoosting)
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

use pyo3::prelude::*;

mod array_utils;
mod automl;
mod datasets;
mod linear;
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
    preprocessing::register_preprocessing_module(m)?;
    pipeline::register_pipeline_module(m)?;
    automl::register_automl_module(m)?;
    datasets::register_datasets_module(m)?;

    Ok(())
}

//! # FerroML - Statistically Rigorous AutoML in Rust
//!
//! FerroML is a high-performance AutoML library that prioritizes statistical rigor
//! over black-box automation. Unlike traditional AutoML tools that hide statistical
//! assumptions, FerroML makes them explicit and testable.
//!
//! ## Design Philosophy
//!
//! 1. **Statistical Rigor First**: Every model includes proper uncertainty quantification,
//!    assumption testing, and diagnostic checks.
//!
//! 2. **Transparent Assumptions**: All statistical assumptions are documented and tested.
//!    No hidden magic - you know exactly what's happening.
//!
//! 3. **Reproducible Results**: Deterministic by default with explicit randomness control.
//!
//! 4. **Python-First API**: Designed for seamless Python integration while leveraging
//!    Rust's performance and safety.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    FerroML Architecture                      │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                             │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
//! │  │   stats     │  │   models    │  │    hpo      │        │
//! │  │             │  │             │  │             │        │
//! │  │ Hypothesis  │  │ Linear      │  │ Bayesian    │        │
//! │  │ Tests       │  │ Tree-based  │  │ Optim       │        │
//! │  │ Confidence  │  │ Ensemble    │  │ Hyperband   │        │
//! │  │ Intervals   │  │ Boosting    │  │ ASHA        │        │
//! │  └─────────────┘  └─────────────┘  └─────────────┘        │
//! │         │                │                │                │
//! │         └────────────────┼────────────────┘                │
//! │                          ▼                                 │
//! │  ┌─────────────────────────────────────────────────┐      │
//! │  │              preprocessing                       │      │
//! │  │  Imputation, Encoding, Scaling, Selection       │      │
//! │  └─────────────────────────────────────────────────┘      │
//! │                          │                                 │
//! │                          ▼                                 │
//! │  ┌─────────────────────────────────────────────────┐      │
//! │  │                 pipeline                         │      │
//! │  │  DAG execution, Caching, Parallelization        │      │
//! │  └─────────────────────────────────────────────────┘      │
//! │                          │                                 │
//! │                          ▼                                 │
//! │  ┌─────────────────────────────────────────────────┐      │
//! │  │                    cv                            │      │
//! │  │  K-Fold, Stratified, TimeSeries, Nested         │      │
//! │  └─────────────────────────────────────────────────┘      │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example Usage (Python)
//!
//! ```python
//! import ferroml as fml
//!
//! # Load data
//! X, y = fml.datasets.load_example()
//!
//! # Create AutoML with explicit statistical controls
//! automl = fml.AutoML(
//!     task="classification",
//!     metric="roc_auc",
//!     statistical_tests=True,      # Run assumption tests
//!     confidence_level=0.95,       # For all intervals
//!     multiple_testing="bonferroni", # Correction method
//!     time_budget_seconds=3600,
//! )
//!
//! # Fit with cross-validation
//! result = automl.fit(X, y, cv=5)
//!
//! # Get results with statistical guarantees
//! print(result.best_model)
//! print(result.confidence_interval)  # CI for performance
//! print(result.assumption_tests)     # All passed tests
//! print(result.diagnostics)          # Residual analysis, etc.
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod automl;
pub mod cv;
pub mod datasets;
pub mod decomposition;
pub mod ensemble;
pub mod error;
pub mod explainability;
pub mod hpo;
#[cfg(feature = "onnx")]
pub mod inference;
pub mod metrics;
pub mod models;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod pipeline;
pub mod preprocessing;
pub mod schema;
pub mod serialization;
#[cfg(feature = "simd")]
pub mod simd;
#[cfg(feature = "sparse")]
pub mod sparse;
pub mod stats;

// Re-exports for convenience
pub use automl::{
    AggregatedFeatureImportance, AutoMLResult, LeaderboardEntry, ModelComparisonResults,
    ModelFeatureImportance, ModelStatistics, PairwiseComparison,
};
pub use error::{FerroError, Result};
#[cfg(feature = "onnx")]
pub use inference::{InferenceSession, SessionMetadata, Tensor, TensorI64, Value};
#[cfg(feature = "onnx")]
pub use onnx::{OnnxConfig, OnnxExportable};
pub use schema::{
    FeatureSchema, FeatureSpec, FeatureType, IssueSeverity, SchemaValidated, ValidationIssue,
    ValidationMode, ValidationResult,
};
pub use serialization::{
    from_bytes, from_bytes_with_metadata, load_model, load_model_auto, load_model_with_metadata,
    peek_metadata, save_model, save_model_with_description, to_bytes, Format, ModelContainer,
    ModelSerialize, SerializationMetadata, FERROML_VERSION,
};

/// Core traits that all FerroML components implement
pub mod traits {
    use ndarray::Array2;
    use serde::{Deserialize, Serialize};

    /// A fitted model that can make predictions
    pub trait Predictor: Send + Sync {
        /// Predict target values for input features
        fn predict(&self, x: &Array2<f64>) -> crate::Result<ndarray::Array1<f64>>;

        /// Predict with uncertainty quantification
        fn predict_with_uncertainty(
            &self,
            x: &Array2<f64>,
            confidence: f64,
        ) -> crate::Result<PredictionWithUncertainty>;
    }

    /// A model that can be trained on data
    pub trait Estimator: Send + Sync {
        /// The fitted model type
        type Fitted: Predictor;

        /// Fit the model to training data
        fn fit(&self, x: &Array2<f64>, y: &ndarray::Array1<f64>) -> crate::Result<Self::Fitted>;

        /// Get hyperparameter search space
        fn search_space(&self) -> crate::hpo::SearchSpace;
    }

    /// A transformer that modifies features
    pub trait Transformer: Send + Sync {
        /// Transform input features
        fn transform(&self, x: &Array2<f64>) -> crate::Result<Array2<f64>>;

        /// Fit and transform in one step
        fn fit_transform(&mut self, x: &Array2<f64>) -> crate::Result<Array2<f64>>;

        /// Inverse transform (if applicable)
        fn inverse_transform(&self, x: &Array2<f64>) -> crate::Result<Array2<f64>> {
            let _ = x;
            Err(crate::FerroError::NotImplemented(
                "inverse_transform".to_string(),
            ))
        }
    }

    /// Prediction result with confidence intervals
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PredictionWithUncertainty {
        /// Point predictions
        pub predictions: ndarray::Array1<f64>,
        /// Lower bound of confidence interval
        pub lower: ndarray::Array1<f64>,
        /// Upper bound of confidence interval
        pub upper: ndarray::Array1<f64>,
        /// Confidence level (e.g., 0.95)
        pub confidence_level: f64,
        /// Standard errors (if available)
        pub std_errors: Option<ndarray::Array1<f64>>,
    }
}

/// Configuration for AutoML runs
#[derive(Debug, Clone)]
pub struct AutoMLConfig {
    /// Task type: "classification" or "regression"
    pub task: Task,
    /// Optimization metric
    pub metric: Metric,
    /// Time budget in seconds
    pub time_budget_seconds: u64,
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Enable statistical assumption testing
    pub statistical_tests: bool,
    /// Confidence level for intervals (default: 0.95)
    pub confidence_level: f64,
    /// Multiple testing correction method
    pub multiple_testing_correction: MultipleTesting,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: usize,
}

impl Default for AutoMLConfig {
    fn default() -> Self {
        Self {
            task: Task::Classification,
            metric: Metric::RocAuc,
            time_budget_seconds: 3600,
            cv_folds: 5,
            statistical_tests: true,
            confidence_level: 0.95,
            multiple_testing_correction: MultipleTesting::BenjaminiHochberg,
            seed: None,
            n_jobs: num_cpus::get(),
        }
    }
}

/// ML task types
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Task {
    /// Binary or multiclass classification
    Classification,
    /// Continuous target regression
    Regression,
    /// Time series forecasting
    TimeSeries,
    /// Survival analysis
    Survival,
}

/// Evaluation metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    // Classification
    /// Area Under ROC Curve
    RocAuc,
    /// Accuracy
    Accuracy,
    /// F1 Score
    F1,
    /// Log Loss
    LogLoss,
    /// Matthews Correlation Coefficient
    Mcc,

    // Regression
    /// Mean Squared Error
    Mse,
    /// Root Mean Squared Error
    Rmse,
    /// Mean Absolute Error
    Mae,
    /// R-squared
    R2,
    /// Mean Absolute Percentage Error
    Mape,

    // Time Series
    /// Symmetric MAPE
    Smape,
    /// Mean Absolute Scaled Error
    Mase,
}

/// Multiple testing correction methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultipleTesting {
    /// No correction
    None,
    /// Bonferroni correction (most conservative)
    Bonferroni,
    /// Holm-Bonferroni step-down
    Holm,
    /// Benjamini-Hochberg FDR control
    BenjaminiHochberg,
    /// Benjamini-Yekutieli FDR (for dependent tests)
    BenjaminiYekutieli,
}

/// Main AutoML entry point
pub struct AutoML {
    config: AutoMLConfig,
}

impl AutoML {
    /// Create new AutoML instance with configuration
    pub fn new(config: AutoMLConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(AutoMLConfig::default())
    }

    /// Get the configuration
    pub fn config(&self) -> &AutoMLConfig {
        &self.config
    }
}

// External crate for CPU count
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    }
}

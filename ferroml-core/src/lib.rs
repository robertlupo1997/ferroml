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

// Documentation coverage handled incrementally
// #![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
// Allow common pedantic lints that are too noisy for ML codebases
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::unreadable_literal)] // ML uses many numeric constants
#![allow(clippy::cast_precision_loss)] // usize->f64 is ubiquitous in ML
#![allow(clippy::cast_possible_truncation)] // f64->usize after bounds checks
#![allow(clippy::cast_sign_loss)] // Often checked or known positive
#![allow(clippy::cast_possible_wrap)] // Often checked or bounded
#![allow(clippy::doc_markdown)] // Too many false positives
#![allow(clippy::must_use_candidate)] // Too noisy for method-heavy API
#![allow(clippy::missing_errors_doc)] // Error docs in progress
#![allow(clippy::missing_panics_doc)] // Panic docs in progress
#![allow(clippy::similar_names)] // ML variables often have similar names
#![allow(clippy::too_many_lines)] // Complex algorithms need space
#![allow(clippy::too_many_arguments)] // ML models have many parameters
#![allow(clippy::many_single_char_names)] // Math notation uses single chars
#![allow(clippy::items_after_statements)] // Common test pattern
#![allow(clippy::float_cmp)] // We use epsilon comparisons where needed
#![allow(clippy::return_self_not_must_use)] // Builder pattern heavy codebase
#![allow(clippy::cast_lossless)] // Intentional casts, not using From
#![allow(clippy::redundant_closure_for_method_calls)] // Often more readable
#![allow(clippy::needless_lifetimes)] // Sometimes explicit lifetimes aid clarity
#![allow(clippy::match_same_arms)] // Sometimes aids readability
#![allow(clippy::option_if_let_else)] // Matches can be clearer
#![allow(clippy::manual_let_else)] // Older style, but clear
#![allow(clippy::unused_self)] // Common in trait implementations
#![allow(clippy::struct_field_names)] // Field naming is intentional
#![allow(clippy::single_match_else)] // Matches can be clearer
#![allow(clippy::trivially_copy_pass_by_ref)] // API consistency
#![allow(clippy::unnecessary_wraps)] // Sometimes wrapping aids API consistency
#![allow(clippy::iter_without_into_iter)] // Not all iterators need IntoIterator
#![allow(clippy::needless_pass_by_value)] // Sometimes cleaner API
#![allow(clippy::map_unwrap_or)] // Sometimes clearer than map_or
#![allow(clippy::explicit_iter_loop)] // Explicit can be clearer
#![allow(clippy::explicit_into_iter_loop)] // Explicit can be clearer
#![allow(clippy::match_wildcard_for_single_variants)] // Explicit is clearer
#![allow(clippy::manual_range_contains)] // Sometimes clearer
#![allow(clippy::use_self)] // Explicit type names can be clearer
#![allow(clippy::cloned_instead_of_copied)] // Both work fine
#![allow(clippy::match_bool)] // Matches can be clearer
#![allow(clippy::if_not_else)] // Both patterns are readable
#![allow(clippy::bool_to_int_with_if)] // Sometimes intentional for clarity
#![allow(clippy::manual_assert)] // Explicit if + panic can be clearer
#![allow(clippy::single_char_pattern)] // Single char literals are fine
#![allow(clippy::uninlined_format_args)] // Older style still valid
#![allow(clippy::manual_clamp)] // Explicit min/max can be clearer
#![allow(clippy::default_trait_access)] // Explicit Default::default() is fine
#![allow(clippy::unnested_or_patterns)] // Older style still valid
#![allow(clippy::manual_string_new)] // String::new() vs "".to_string() both fine
#![allow(clippy::format_push_string)] // Sometimes clearer than write!
#![allow(clippy::map_unwrap_or)] // Explicit map + unwrap_or fine
#![allow(clippy::ignored_unit_patterns)] // () pattern matching is fine
#![allow(clippy::manual_is_ascii_check)] // Explicit range check is fine
#![allow(clippy::op_ref)] // Reference operations are explicit
#![allow(clippy::ptr_as_ptr)] // Pointer casts are explicit
#![allow(clippy::needless_borrow)] // Sometimes clearer with &
#![allow(clippy::derive_partial_eq_without_eq)] // Intentional
#![allow(clippy::wildcard_imports)] // test modules often use wildcard imports
#![allow(clippy::suspicious_operation_groupings)] // ML formulas can look suspicious
#![allow(clippy::manual_assert)] // Explicit panic with message
#![allow(clippy::assigning_clones)] // Intentional clone assignment
#![allow(clippy::naive_bytecount)] // Simple iteration is fine
#![allow(clippy::redundant_closure)] // Sometimes clearer with closure
#![allow(clippy::match_ref_pats)] // Pattern ref matching is clear
#![allow(clippy::len_zero)] // .len() == 0 is often clearer than .is_empty()
#![allow(clippy::comparison_to_empty)] // Explicit comparisons are clear
#![allow(clippy::implied_bounds_in_impls)] // Explicit bounds are clear
#![allow(clippy::large_stack_arrays)] // Sometimes needed for performance
#![allow(clippy::useless_vec)] // Sometimes vec! macro is needed for ownership
#![allow(clippy::let_and_return)] // Explicit binding before return aids debugging
#![allow(clippy::collapsible_else_if)] // Sometimes separate blocks are clearer
#![allow(clippy::collapsible_if)] // Sometimes separate blocks are clearer
#![allow(clippy::missing_const_for_fn)] // Many functions could be const
#![allow(clippy::unsafe_derive_deserialize)] // We handle unsafe appropriately
#![allow(clippy::manual_div_ceil)] // Explicit is clearer
#![allow(clippy::ptr_arg)] // &Vec in API for consistency
#![allow(clippy::const_is_empty)] // Runtime checks are clear
#![allow(clippy::indexing_slicing)] // Bounds are often known
#![allow(clippy::needless_for_each)] // for_each can be clearer
#![allow(clippy::linkedlist)] // Sometimes appropriate data structure
#![allow(clippy::doc_lazy_continuation)] // Doc formatting preference
#![allow(clippy::explicit_deref_methods)] // Explicit deref is clear
#![allow(clippy::needless_lifetimes)] // Explicit lifetimes aid clarity
#![allow(clippy::manual_strip)] // Explicit string manipulation is clear
#![allow(clippy::type_complexity)] // Complex types in ML are common
#![allow(clippy::missing_fields_in_debug)] // Debug impls show relevant fields
#![allow(clippy::approx_constant)] // Sometimes explicit values aid understanding
#![allow(clippy::stable_sort_primitive)] // sort vs sort_unstable is intentional
#![allow(clippy::field_reassign_with_default)] // Builder pattern variation
#![allow(clippy::if_same_then_else)] // Sometimes intentional for clarity
#![allow(clippy::match_single_binding)] // Pattern matching can aid clarity
#![allow(clippy::single_match)] // Matches are often extended later
#![allow(clippy::useless_format)] // format! for consistency
#![allow(clippy::iter_nth_zero)] // .nth(0) can be intentional
#![allow(clippy::derivable_impls)] // Manual impls allow documentation
#![allow(clippy::manual_map)] // Explicit mapping is clear
#![allow(clippy::manual_filter)] // Explicit filtering is clear
#![allow(clippy::redundant_pattern)] // Explicit patterns are clear
#![allow(clippy::redundant_guards)] // Explicit guards are clear
#![allow(clippy::borrow_as_ptr)] // Explicit borrow-to-ptr conversion
#![allow(clippy::struct_excessive_bools)] // ML configs need many bools
#![allow(clippy::assign_op_pattern)] // Both x = x + 1 and x += 1 are fine
#![allow(clippy::manual_slice_size_calculation)] // Explicit is clear
#![allow(clippy::redundant_pattern_matching)] // Explicit matching is clear
#![allow(clippy::unnecessary_result_map_or_else)] // Explicit error handling
#![allow(clippy::str_to_string)] // String conversion methods are clear
#![allow(clippy::needless_return)] // Explicit returns can be clearer
#![allow(clippy::semicolon_if_nothing_returned)] // Explicit semicolons are fine
#![allow(clippy::ref_binding_to_reference)] // Explicit ref binding is clear
#![allow(clippy::no_effect_underscore_binding)] // Underscore bindings are fine
#![allow(clippy::enum_glob_use)] // Enum glob use in pattern matching
#![allow(clippy::implicit_hasher)] // Explicit hasher type is fine
#![allow(clippy::vec_init_then_push)] // Push after init is clear
#![allow(clippy::multiple_crate_versions)] // Dependency management
#![allow(clippy::branches_sharing_code)] // Sometimes clearer to duplicate
#![allow(clippy::unnested_or_patterns)] // Older pattern style is fine
#![allow(clippy::iter_with_drain)] // Drain when ownership transfer needed
#![allow(clippy::format_collect)] // Format + collect pattern
#![allow(clippy::manual_ok_or)] // Explicit ok_or calls
#![allow(clippy::std_instead_of_core)] // std imports are fine
#![allow(clippy::std_instead_of_alloc)] // std imports are fine
#![allow(clippy::ref_as_ptr)] // Explicit ref to ptr conversion
#![allow(clippy::borrow_deref_ref)] // Explicit borrow patterns
#![allow(clippy::needless_collect)] // Collect for ownership transfer
#![allow(clippy::unnecessary_to_owned)] // to_owned for ownership clarity
#![allow(clippy::option_as_ref_deref)] // Explicit option handling
#![allow(clippy::missing_assert_message)] // Assert expressions are self-documenting
#![allow(clippy::match_result_ok)] // Match on Result for clarity
#![allow(clippy::rc_clone_in_vec_init)] // Rc cloning in vec init
#![allow(clippy::inefficient_to_string)] // to_string for clarity
#![allow(clippy::uninit_vec)] // Explicit uninitialized vector handling
#![allow(clippy::unnecessary_literal_bound)] // &str is fine vs &'static str
#![allow(clippy::needless_continue)] // Explicit continue can be clearer
#![allow(clippy::should_implement_trait)] // Methods named like traits ok
#![allow(clippy::needless_range_loop)] // Range loops are clear
#![allow(clippy::implicit_saturating_sub)] // Explicit subtraction is clear
#![allow(clippy::elidable_lifetime_names)] // Explicit lifetimes can help
#![allow(clippy::doc_overindented_list_items)] // Doc formatting preference
#![allow(clippy::self_only_used_in_recursion)] // Recursive methods are fine
#![allow(clippy::cloned_ref_to_slice_refs)] // Cloning refs is intentional
#![allow(clippy::unnecessary_unwrap)] // Unwrap after check is clear
#![allow(clippy::unused_enumerate_index)] // Sometimes index unused intentionally
#![allow(clippy::manual_try_fold)] // Explicit fold is clearer
#![allow(clippy::ignore_without_reason)] // Test ignores are self-evident
#![allow(clippy::format_in_format_args)] // Nested format is sometimes clearer
#![allow(clippy::missing_docs_in_private_items)] // Private items don't need docs
#![allow(clippy::print_stdout)] // Print statements in tests/examples
#![allow(clippy::assertions_on_constants)] // Constant assertions are intentional
#![allow(clippy::duplicated_attributes)] // Sometimes needed for clarity
#![allow(clippy::unnecessary_map_or)] // map_or style is clear
#![allow(clippy::nonminimal_bool)] // Explicit boolean expressions
#![allow(clippy::redundant_else)] // Explicit else blocks can be clearer
#![allow(clippy::ref_option_ref)] // Reference to Option<&T> is fine
#![allow(clippy::double_comparisons)] // Sometimes clearer

pub mod automl;
pub mod clustering;
pub mod cv;
pub mod datasets;
pub mod decomposition;
pub mod ensemble;
pub mod error;
pub mod explainability;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod hpo;
#[cfg(feature = "onnx")]
pub mod inference;
pub mod linalg;
pub mod metrics;
pub mod models;
pub mod neural;
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
pub mod testing;

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
    /// Algorithm selection strategy
    pub algorithm_selection: AlgorithmSelection,
    /// Maximum number of models in ensemble
    pub ensemble_size: usize,
    /// Portfolio preset for algorithm selection
    pub preset: automl::PortfolioPreset,
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
            algorithm_selection: AlgorithmSelection::default(),
            ensemble_size: 10,
            preset: automl::PortfolioPreset::Balanced,
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

/// Algorithm selection strategy for AutoML
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlgorithmSelection {
    /// Uniform random selection across algorithms
    Uniform,
    /// Bayesian optimization for algorithm selection
    Bayesian,
    /// Multi-armed bandit based selection (default)
    #[default]
    Bandit,
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

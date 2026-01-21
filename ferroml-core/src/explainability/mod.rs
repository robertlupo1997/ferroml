//! Explainability and Interpretability Module
//!
//! This module provides model-agnostic explainability tools with statistical rigor.
//! `FerroML`'s approach ensures all explanations come with proper uncertainty quantification.
//!
//! ## Implemented Methods
//!
//! - **Permutation Importance**: Model-agnostic feature importance with confidence intervals
//! - **Partial Dependence Plots (PDP)**: Visualize marginal feature effects on predictions
//! - **Individual Conditional Expectation (ICE)**: Per-sample feature effect curves
//!   - c-ICE (centered): Centered at a reference point for comparison
//!   - d-ICE (derivative): Rate of change for non-linear effect detection
//! - **H-Statistic**: Friedman's H-statistic for feature interaction detection
//!   - Pairwise interaction strength quantification
//!   - Overall interaction strength for individual features
//!   - Bootstrap confidence intervals and permutation-based p-values
//! - **TreeSHAP**: Exact SHAP values for tree-based models
//!   - O(TLD²) algorithm for tree ensembles
//!   - Supports decision trees, random forests, and gradient boosting
//! - **KernelSHAP**: Approximate SHAP values for any model
//!   - Model-agnostic using weighted linear regression
//!   - Works with any fitted model implementing the `Model` trait
//! - **SHAP Summary**: Data structures for SHAP visualizations
//!   - Bar plot data for global feature importance
//!   - Beeswarm plot data for detailed SHAP analysis
//!   - Dependence plot data for feature interactions
//!
//! ## Philosophy
//!
//! - **CI by default**: All importance scores include confidence intervals via repeated shuffling
//! - **Model-agnostic**: Works with any fitted model implementing the `Model` trait
//! - **Metric-flexible**: Supports any metric from the metrics module
//!
//! ## Examples
//!
//! ### Permutation Importance
//!
//! ```ignore
//! use ferroml_core::explainability::permutation_importance;
//! use ferroml_core::models::RandomForestClassifier;
//! use ferroml_core::metrics::accuracy;
//!
//! let mut model = RandomForestClassifier::new();
//! model.fit(&x_train, &y_train)?;
//!
//! let result = permutation_importance(
//!     &model,
//!     &x_test,
//!     &y_test,
//!     |y_true, y_pred| accuracy(y_true, y_pred),
//!     10,     // n_repeats
//!     Some(42), // random_state
//! )?;
//!
//! println!("Feature importances: {:?}", result.importances_mean);
//! println!("95% CI: [{:?}, {:?}]", result.ci_lower, result.ci_upper);
//! ```
//!
//! ### Partial Dependence Plots
//!
//! ```ignore
//! use ferroml_core::explainability::{partial_dependence, GridMethod};
//! use ferroml_core::models::RandomForestRegressor;
//!
//! let mut model = RandomForestRegressor::new();
//! model.fit(&x_train, &y_train)?;
//!
//! // 1D PDP for feature 0
//! let result = partial_dependence(&model, &x_test, 0, 50, GridMethod::Percentile, false)?;
//! println!("Grid: {:?}", result.grid_values);
//! println!("PDP: {:?}", result.pdp_values);
//!
//! // 2D PDP for feature interaction
//! let result_2d = partial_dependence_2d(&model, &x_test, 0, 1, 20, GridMethod::Percentile)?;
//! println!("Interaction effect: {:?}", result_2d.pdp_values);
//! ```
//!
//! ### Individual Conditional Expectation (ICE)
//!
//! ```ignore
//! use ferroml_core::explainability::{individual_conditional_expectation, ICEConfig};
//! use ferroml_core::models::RandomForestRegressor;
//!
//! let mut model = RandomForestRegressor::new();
//! model.fit(&x_train, &y_train)?;
//!
//! // Basic ICE curves
//! let result = individual_conditional_expectation(&model, &x_test, 0, ICEConfig::default())?;
//! println!("ICE curves shape: {:?}", result.ice_curves.shape());
//! println!("Heterogeneity: {:.4}", result.mean_heterogeneity());
//!
//! // ICE with centering and derivatives
//! let config = ICEConfig::new()
//!     .with_centering(0)  // Center at first grid point
//!     .with_derivative();
//! let result = individual_conditional_expectation(&model, &x_test, 0, config)?;
//! println!("Centered ICE: {:?}", result.centered_ice);
//! println!("Derivative ICE: {:?}", result.derivative_ice);
//! ```
//!
//! ### Feature Interaction Detection (H-Statistic)
//!
//! ```ignore
//! use ferroml_core::explainability::{h_statistic, h_statistic_matrix, HStatisticConfig};
//! use ferroml_core::models::RandomForestRegressor;
//!
//! let mut model = RandomForestRegressor::new();
//! model.fit(&x_train, &y_train)?;
//!
//! // Pairwise H-statistic
//! let result = h_statistic(&model, &x_test, 0, 1, HStatisticConfig::default())?;
//! println!("H² = {:.4} ({})", result.h_squared, result.interpretation());
//!
//! // With bootstrap CI and permutation test
//! let config = HStatisticConfig::new()
//!     .with_bootstrap(1000)
//!     .with_permutation_test(500)
//!     .with_random_state(42);
//! let result = h_statistic(&model, &x_test, 0, 1, config)?;
//! println!("{}", result);  // Includes CI and p-value
//!
//! // Pairwise interaction matrix
//! let matrix = h_statistic_matrix(&model, &x_test, None, HStatisticConfig::default())?;
//! let top_interactions = matrix.top_k(5);
//! for (i, j, h2) in top_interactions {
//!     println!("Features {} x {}: H² = {:.4}", i, j, h2);
//! }
//! ```
//!
//! ### KernelSHAP (Model-Agnostic SHAP)
//!
//! ```ignore
//! use ferroml_core::explainability::{KernelExplainer, KernelSHAPConfig, SHAPResult};
//! use ferroml_core::models::RandomForestRegressor;
//!
//! let mut model = RandomForestRegressor::new();
//! model.fit(&x_train, &y_train)?;
//!
//! // Create KernelSHAP explainer with background data
//! let config = KernelSHAPConfig::new()
//!     .with_n_samples(1024)
//!     .with_random_state(42);
//! let explainer = KernelExplainer::new(&model, &x_train, config)?;
//!
//! // Explain a single prediction
//! let result = explainer.explain(&x_test.row(0).to_vec())?;
//! println!("Base value: {}", result.base_value);
//! println!("SHAP values: {:?}", result.shap_values);
//! println!("{}", result.summary());  // Pretty-printed summary
//!
//! // Explain multiple predictions
//! let batch_result = explainer.explain_batch(&x_test)?;
//! println!("Global importance: {:?}", batch_result.mean_abs_shap());
//! ```

mod h_statistic;
mod ice;
mod kernelshap;
mod partial_dependence;
mod permutation;
mod summary;
mod treeshap;

pub use h_statistic::{
    h_statistic, h_statistic_matrix, h_statistic_matrix_parallel, h_statistic_overall,
    h_statistic_parallel, HStatisticConfig, HStatisticMatrix, HStatisticOverallResult,
    HStatisticResult,
};
pub use ice::{
    center_ice_curves, compute_derivative_ice, ice_from_curves, ice_multi, ice_multi_parallel,
    individual_conditional_expectation, individual_conditional_expectation_parallel, ICEConfig,
    ICEResult,
};
pub use kernelshap::{KernelExplainer, KernelSHAPConfig};
pub use partial_dependence::{
    partial_dependence, partial_dependence_2d, partial_dependence_2d_parallel,
    partial_dependence_multi, partial_dependence_multi_parallel, partial_dependence_parallel,
    GridMethod, PDP2DResult, PDPResult,
};
pub use permutation::{
    permutation_importance, permutation_importance_parallel, PermutationImportanceResult,
};
pub use summary::{
    BarPlotData, BarPlotEntry, BeeswarmFeatureData, BeeswarmPlotData, BeeswarmPoint,
    DependencePlotData, FeatureSHAPStats, SHAPSummary,
};
pub use treeshap::{SHAPBatchResult, SHAPResult, TreeExplainer};

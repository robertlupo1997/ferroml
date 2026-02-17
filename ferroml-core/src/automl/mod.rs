//! AutoML Orchestration Module
//!
//! This module provides Combined Algorithm Selection and Hyperparameter optimization (CASH)
//! - the core AutoML problem. It includes algorithm portfolios, presets, data-aware
//! search space generation, bandit-based time budget allocation, automatic preprocessing
//! selection, and ensemble construction from trials.
//!
//! # Example
//!
//! ```no_run
//! # use ndarray::{Array1, Array2};
//! # let x = Array2::from_shape_vec((20, 3), (0..60).map(|i| i as f64).collect()).unwrap();
//! # let y = Array1::from_vec((0..20).map(|i| (i % 2) as f64).collect());
//! use ferroml_core::automl::{
//!     AlgorithmPortfolio, PortfolioPreset, DataCharacteristics,
//!     TimeBudgetAllocator, TimeBudgetConfig, BanditStrategy,
//!     PreprocessingSelector, PreprocessingConfig, PreprocessingStrategy,
//!     EnsembleBuilder, EnsembleConfig, TrialResult,
//! };
//!
//! // Create a portfolio for classification
//! let portfolio = AlgorithmPortfolio::for_classification(PortfolioPreset::Balanced);
//!
//! // Adapt search spaces based on data characteristics
//! let chars = DataCharacteristics::from_data(&x, &y);
//! let adapted = portfolio.adapt_to_data(&chars);
//!
//! // Allocate time budget using UCB1 bandit strategy
//! let config = TimeBudgetConfig::new(3600) // 1 hour total
//!     .with_strategy(BanditStrategy::UCB1 { exploration_constant: 2.0 });
//! let mut allocator = TimeBudgetAllocator::new(config, &adapted);
//!
//! // Automatic preprocessing selection
//! let preproc_config = PreprocessingConfig::new()
//!     .with_strategy(PreprocessingStrategy::Auto);
//! let selector = PreprocessingSelector::new(preproc_config);
//!
//! // Build preprocessing pipeline for an algorithm
//! for algo_config in &adapted.algorithms {
//!     let pipeline = selector.build_pipeline(&chars, &algo_config.preprocessing, None);
//! }
//!
//! // After running trials, build ensemble from results
//! let ensemble_config = EnsembleConfig::new()
//!     .with_max_models(10)
//!     .with_selection_iterations(50);
//! let mut builder = EnsembleBuilder::new(ensemble_config);
//! # let trials: Vec<TrialResult> = vec![];
//! # let y_true = y.clone();
//! let ensemble = builder.build_from_trials(&trials, &y_true).unwrap();
//! ```

pub mod ensemble;
pub mod fit;
pub mod metafeatures;
pub mod portfolio;
pub mod preprocessing;
pub mod time_budget;
pub mod transfer;
pub mod warmstart;

pub use fit::{
    AggregatedFeatureImportance, AutoMLResult, LeaderboardEntry, ModelComparisonResults,
    ModelFeatureImportance, ModelStatistics, PairwiseComparison,
};

pub use portfolio::{
    AlgorithmComplexity, AlgorithmConfig, AlgorithmPortfolio, AlgorithmType, DataCharacteristics,
    PortfolioPreset, PreprocessingRequirement,
};

pub use preprocessing::{
    select_preprocessing, DetectedCharacteristics, EncodingType, ImputationStrategy,
    PreprocessingConfig, PreprocessingPipelineSpec, PreprocessingSelection, PreprocessingSelector,
    PreprocessingStepSpec, PreprocessingStepType, PreprocessingStrategy, ScalerType, StepParam,
};

pub use time_budget::{
    AlgorithmArm, AllocationSummary, ArmSelection, ArmSummary, BanditStrategy, SelectionReason,
    TimeBudgetAllocator, TimeBudgetConfig,
};

pub use ensemble::{
    build_ensemble, build_ensemble_with_config, EnsembleBuilder, EnsembleConfig, EnsembleMember,
    EnsembleResult, ParamValue, TrialResult,
};

pub use metafeatures::{
    DatasetMetafeatures, FeatureStatistics, InformationMetafeatures, LandmarkingMetafeatures,
    MetafeatureConfig, SimpleMetafeatures, StatisticalMetafeatures,
};

pub use warmstart::{
    ConfigurationRecord, DatasetRecord, MetaLearningStore, SimilarDataset, StoreStatistics,
    WarmStartConfig, WarmStartResult, WeightedConfiguration,
};

pub use transfer::{
    algorithm_priorities_from_warmstart, initialize_study_with_warmstart, ParameterAdaptation,
    ParameterPrior, PriorKnowledge, PriorType, TransferConfig, TransferredSearchSpace,
    WarmStartSampler,
};

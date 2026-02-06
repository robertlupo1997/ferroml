//! Configuration Space Transfer for Meta-Learning
//!
//! This module provides functionality for transferring hyperparameter knowledge
//! between similar datasets:
//! - Adapted search spaces based on successful configurations
//! - Warm-started samplers that use prior configurations as initial points
//! - Hyperparameter priors learned from similar datasets
//!
//! # Example
//!
//! ```ignore
//! use ferroml_core::automl::{
//!     MetaLearningStore, WarmStartConfig, DatasetMetafeatures,
//!     TransferredSearchSpace, WarmStartSampler, PriorKnowledge,
//! };
//! use ferroml_core::hpo::{Study, Direction};
//!
//! // Get warm-start configurations from similar datasets
//! let store = MetaLearningStore::from_json(&saved_store)?;
//! let warm_start = store.get_warm_start_configs(&metafeatures, &config, task)?;
//!
//! // Create transferred search space with adapted bounds
//! let original_space = SearchSpace::new().float_log("learning_rate", 1e-4, 1.0);
//! let transferred = TransferredSearchSpace::from_warm_start(
//!     &original_space,
//!     &warm_start,
//!     TransferConfig::default(),
//! );
//!
//! // Create a warm-started study
//! let sampler = WarmStartSampler::new(&warm_start, original_space.clone());
//! let study = Study::new("my_study", transferred.search_space, Direction::Maximize)
//!     .with_sampler(sampler);
//! ```

use crate::automl::portfolio::AlgorithmType;
use crate::automl::warmstart::WarmStartResult;
use crate::automl::ParamValue;
use crate::hpo::{Parameter, ParameterType, ParameterValue, SearchSpace, Trial, TrialState};
use crate::Result;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for search space transfer
#[derive(Debug, Clone)]
pub struct TransferConfig {
    /// How much to shrink the search space around successful configurations (0.0 to 1.0)
    /// 1.0 = no shrinking, 0.0 = point estimate (not recommended)
    pub shrink_factor: f64,
    /// Minimum number of configurations required to enable shrinking
    pub min_configs_for_shrinking: usize,
    /// Whether to adapt bounds based on configuration distribution
    pub adapt_bounds: bool,
    /// Confidence level for bound adaptation (e.g., 0.95 for 95% CI)
    pub confidence_level: f64,
    /// Maximum factor to expand bounds beyond observed values
    pub max_expansion_factor: f64,
    /// Whether to preserve original bounds as minimum constraints
    pub preserve_original_bounds: bool,
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            shrink_factor: 0.5,
            min_configs_for_shrinking: 3,
            adapt_bounds: true,
            confidence_level: 0.95,
            max_expansion_factor: 2.0,
            preserve_original_bounds: true,
        }
    }
}

impl TransferConfig {
    /// Create a new transfer configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the shrink factor
    pub fn with_shrink_factor(mut self, factor: f64) -> Self {
        self.shrink_factor = factor.max(0.0).min(1.0);
        self
    }

    /// Set minimum configurations for shrinking
    pub fn with_min_configs(mut self, n: usize) -> Self {
        self.min_configs_for_shrinking = n.max(1);
        self
    }

    /// Enable/disable bound adaptation
    pub fn with_adapt_bounds(mut self, enable: bool) -> Self {
        self.adapt_bounds = enable;
        self
    }

    /// Set confidence level for bound adaptation
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level.max(0.5).min(0.99);
        self
    }

    /// Aggressive transfer (narrow search space)
    pub fn aggressive() -> Self {
        Self {
            shrink_factor: 0.3,
            min_configs_for_shrinking: 2,
            adapt_bounds: true,
            confidence_level: 0.90,
            max_expansion_factor: 1.5,
            preserve_original_bounds: false,
        }
    }

    /// Conservative transfer (wider search space)
    pub fn conservative() -> Self {
        Self {
            shrink_factor: 0.8,
            min_configs_for_shrinking: 5,
            adapt_bounds: true,
            confidence_level: 0.99,
            max_expansion_factor: 3.0,
            preserve_original_bounds: true,
        }
    }
}

/// Prior knowledge learned from similar datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorKnowledge {
    /// Per-parameter priors
    pub parameter_priors: HashMap<String, ParameterPrior>,
    /// Algorithm type this prior applies to (if specific)
    pub algorithm: Option<AlgorithmType>,
    /// Mean similarity of source datasets
    pub source_similarity: f64,
    /// Number of source configurations
    pub n_configurations: usize,
}

/// Prior distribution for a single parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterPrior {
    /// Parameter name
    pub name: String,
    /// Type of prior (based on observed values)
    pub prior_type: PriorType,
    /// Whether this is on log scale
    pub log_scale: bool,
    /// Confidence in the prior (higher = more data)
    pub confidence: f64,
}

/// Type of prior distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorType {
    /// Normal prior for continuous parameters
    Normal {
        /// Mean of the distribution
        mean: f64,
        /// Standard deviation
        std: f64,
    },
    /// Log-normal prior for positive parameters
    LogNormal {
        /// Mean of log(x)
        log_mean: f64,
        /// Std of log(x)
        log_std: f64,
    },
    /// Categorical prior (distribution over choices)
    Categorical {
        /// Probability of each choice
        probabilities: HashMap<String, f64>,
    },
    /// Boolean prior
    Boolean {
        /// Probability of true
        p_true: f64,
    },
    /// Uniform (no prior knowledge)
    Uniform,
}

impl PriorKnowledge {
    /// Extract prior knowledge from warm-start configurations
    pub fn from_warm_start(warm_start: &WarmStartResult) -> Self {
        let mut parameter_priors = HashMap::new();

        if warm_start.configurations.is_empty() {
            return Self {
                parameter_priors,
                algorithm: None,
                source_similarity: 0.0,
                n_configurations: 0,
            };
        }

        // Collect all parameter values from configurations
        let mut param_values: HashMap<String, Vec<(f64, &ParamValue)>> = HashMap::new();

        for weighted_config in &warm_start.configurations {
            let weight = weighted_config.priority_weight;
            for (name, value) in &weighted_config.config.params {
                param_values
                    .entry(name.clone())
                    .or_default()
                    .push((weight, value));
            }
        }

        // Compute priors for each parameter
        for (name, values) in param_values {
            if let Some(prior) = Self::compute_parameter_prior(&name, &values) {
                parameter_priors.insert(name, prior);
            }
        }

        // Determine algorithm if all configs use the same one
        let algorithms: std::collections::HashSet<_> = warm_start
            .configurations
            .iter()
            .map(|c| c.config.algorithm)
            .collect();
        let algorithm = if algorithms.len() == 1 {
            algorithms.into_iter().next()
        } else {
            None
        };

        Self {
            parameter_priors,
            algorithm,
            source_similarity: warm_start.mean_similarity,
            n_configurations: warm_start.configurations.len(),
        }
    }

    /// Compute prior for a single parameter from weighted values
    fn compute_parameter_prior(
        name: &str,
        values: &[(f64, &ParamValue)],
    ) -> Option<ParameterPrior> {
        if values.is_empty() {
            return None;
        }

        // Total weight for normalization
        let total_weight: f64 = values.iter().map(|(w, _)| w).sum();
        if total_weight < 1e-10 {
            return None;
        }

        // Confidence based on number of observations
        let confidence = 1.0 - 1.0 / (values.len() as f64 + 1.0);

        // Check the type of the first value
        match values[0].1 {
            ParamValue::Float(_) | ParamValue::Int(_) => {
                let numeric_values: Vec<(f64, f64)> = values
                    .iter()
                    .filter_map(|(w, v)| {
                        let num = match v {
                            ParamValue::Float(f) => Some(*f),
                            ParamValue::Int(i) => Some(*i as f64),
                            _ => None,
                        };
                        num.map(|n| (*w, n))
                    })
                    .collect();

                if numeric_values.is_empty() {
                    return None;
                }

                // Compute weighted mean and std
                let weighted_mean =
                    numeric_values.iter().map(|(w, v)| w * v).sum::<f64>() / total_weight;

                let weighted_var = numeric_values
                    .iter()
                    .map(|(w, v)| w * (v - weighted_mean).powi(2))
                    .sum::<f64>()
                    / total_weight;
                let weighted_std = weighted_var.sqrt().max(1e-10);

                // Check if values are all positive and span orders of magnitude
                let all_positive = numeric_values.iter().all(|(_, v)| *v > 0.0);
                let max_val = numeric_values
                    .iter()
                    .map(|(_, v)| *v)
                    .fold(f64::NEG_INFINITY, f64::max);
                let min_val = numeric_values
                    .iter()
                    .map(|(_, v)| *v)
                    .fold(f64::INFINITY, f64::min);
                let log_scale = all_positive && max_val / min_val.max(1e-10) > 10.0;

                let prior_type = if log_scale {
                    // Log-normal prior
                    let log_values: Vec<(f64, f64)> = numeric_values
                        .iter()
                        .filter(|(_, v)| *v > 0.0)
                        .map(|(w, v)| (*w, v.ln()))
                        .collect();

                    let log_total: f64 = log_values.iter().map(|(w, _)| w).sum();
                    let log_mean = log_values.iter().map(|(w, v)| w * v).sum::<f64>() / log_total;
                    let log_var = log_values
                        .iter()
                        .map(|(w, v)| w * (v - log_mean).powi(2))
                        .sum::<f64>()
                        / log_total;

                    PriorType::LogNormal {
                        log_mean,
                        log_std: log_var.sqrt().max(0.1),
                    }
                } else {
                    PriorType::Normal {
                        mean: weighted_mean,
                        std: weighted_std,
                    }
                };

                Some(ParameterPrior {
                    name: name.to_string(),
                    prior_type,
                    log_scale,
                    confidence,
                })
            }
            ParamValue::String(_) => {
                // Compute categorical distribution
                let mut counts: HashMap<String, f64> = HashMap::new();
                for (w, v) in values {
                    let cat = match v {
                        ParamValue::String(s) => s.clone(),
                        _ => continue,
                    };
                    *counts.entry(cat).or_default() += w;
                }

                let total: f64 = counts.values().sum();
                let probabilities: HashMap<String, f64> =
                    counts.into_iter().map(|(k, v)| (k, v / total)).collect();

                Some(ParameterPrior {
                    name: name.to_string(),
                    prior_type: PriorType::Categorical { probabilities },
                    log_scale: false,
                    confidence,
                })
            }
            ParamValue::Bool(_) => {
                let true_weight: f64 = values
                    .iter()
                    .filter_map(|(w, v)| match v {
                        ParamValue::Bool(true) => Some(*w),
                        _ => None,
                    })
                    .sum();

                Some(ParameterPrior {
                    name: name.to_string(),
                    prior_type: PriorType::Boolean {
                        p_true: true_weight / total_weight,
                    },
                    log_scale: false,
                    confidence,
                })
            }
        }
    }

    /// Sample a value from the prior
    pub fn sample_prior(&self, name: &str, rng: &mut StdRng) -> Option<f64> {
        let prior = self.parameter_priors.get(name)?;

        match &prior.prior_type {
            PriorType::Normal { mean, std } => Some(normal_sample(rng, *mean, *std)),
            PriorType::LogNormal { log_mean, log_std } => {
                Some(normal_sample(rng, *log_mean, *log_std).exp())
            }
            PriorType::Boolean { p_true } => Some(if rng.random_bool(*p_true) { 1.0 } else { 0.0 }),
            PriorType::Categorical { probabilities } => {
                // Return index of sampled category
                let mut cumsum = 0.0;
                let u: f64 = rng.random();
                for (i, (_, p)) in probabilities.iter().enumerate() {
                    cumsum += p;
                    if u <= cumsum {
                        return Some(i as f64);
                    }
                }
                Some(0.0)
            }
            PriorType::Uniform => None,
        }
    }

    /// Get the most likely categorical value
    pub fn most_likely_categorical(&self, name: &str) -> Option<String> {
        let prior = self.parameter_priors.get(name)?;
        match &prior.prior_type {
            PriorType::Categorical { probabilities } => probabilities
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(k, _)| k.clone()),
            _ => None,
        }
    }
}

/// Search space adapted based on configurations from similar datasets
#[derive(Debug, Clone)]
pub struct TransferredSearchSpace {
    /// The adapted search space
    pub search_space: SearchSpace,
    /// Original search space (for reference)
    pub original_space: SearchSpace,
    /// Prior knowledge used for transfer
    pub prior: PriorKnowledge,
    /// Configuration used for transfer
    pub config: TransferConfig,
    /// Adaptation details for each parameter
    pub adaptations: HashMap<String, ParameterAdaptation>,
}

/// How a parameter was adapted during transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterAdaptation {
    /// Parameter name
    pub name: String,
    /// Original bounds (low, high) for numeric, or n_choices for categorical
    pub original_bounds: (f64, f64),
    /// Adapted bounds
    pub adapted_bounds: (f64, f64),
    /// Shrink ratio achieved (adapted_range / original_range)
    pub shrink_ratio: f64,
    /// Whether bounds were adapted
    pub was_adapted: bool,
    /// Reason for adaptation or no adaptation
    pub reason: String,
}

impl TransferredSearchSpace {
    /// Create a transferred search space from warm-start results
    pub fn from_warm_start(
        original: &SearchSpace,
        warm_start: &WarmStartResult,
        config: TransferConfig,
    ) -> Self {
        let prior = PriorKnowledge::from_warm_start(warm_start);
        Self::from_prior(original, &prior, config)
    }

    /// Create a transferred search space from prior knowledge
    pub fn from_prior(
        original: &SearchSpace,
        prior: &PriorKnowledge,
        config: TransferConfig,
    ) -> Self {
        let mut adapted_space = SearchSpace::new();
        let mut adaptations = HashMap::new();

        for (name, param) in &original.parameters {
            let (adapted_param, adaptation) = Self::adapt_parameter(name, param, prior, &config);
            adapted_space = adapted_space.add(name.clone(), adapted_param);
            adaptations.insert(name.clone(), adaptation);
        }

        Self {
            search_space: adapted_space,
            original_space: original.clone(),
            prior: prior.clone(),
            config,
            adaptations,
        }
    }

    /// Adapt a single parameter based on prior knowledge
    fn adapt_parameter(
        name: &str,
        param: &Parameter,
        prior: &PriorKnowledge,
        config: &TransferConfig,
    ) -> (Parameter, ParameterAdaptation) {
        let prior_param = prior.parameter_priors.get(name);

        // Default adaptation (no change)
        let mut adaptation = ParameterAdaptation {
            name: name.to_string(),
            original_bounds: Self::get_bounds(param),
            adapted_bounds: Self::get_bounds(param),
            shrink_ratio: 1.0,
            was_adapted: false,
            reason: "No prior knowledge available".to_string(),
        };

        // If no prior or not enough configurations, return original
        if prior.n_configurations < config.min_configs_for_shrinking {
            adaptation.reason = format!(
                "Insufficient configurations ({} < {})",
                prior.n_configurations, config.min_configs_for_shrinking
            );
            return (param.clone(), adaptation);
        }

        let prior_param = match prior_param {
            Some(p) => p,
            None => return (param.clone(), adaptation),
        };

        // Adapt based on parameter type
        let adapted = match (&param.param_type, &prior_param.prior_type) {
            (ParameterType::Float { low, high }, PriorType::Normal { mean, std })
            | (
                ParameterType::Float { low, high },
                PriorType::LogNormal {
                    log_mean: mean,
                    log_std: std,
                },
            ) => {
                let (new_low, new_high) =
                    Self::adapt_numeric_bounds(*low, *high, *mean, *std, param.log_scale, config);

                adaptation.adapted_bounds = (new_low, new_high);
                adaptation.shrink_ratio = (new_high - new_low) / (high - low).max(1e-10);
                adaptation.was_adapted = true;
                adaptation.reason =
                    format!("Adapted from prior (mean={:.4}, std={:.4})", mean, std);

                Parameter {
                    param_type: ParameterType::Float {
                        low: new_low,
                        high: new_high,
                    },
                    log_scale: param.log_scale,
                    default: param.default.clone(),
                }
            }

            (ParameterType::Int { low, high }, PriorType::Normal { mean, std })
            | (
                ParameterType::Int { low, high },
                PriorType::LogNormal {
                    log_mean: mean,
                    log_std: std,
                },
            ) => {
                let (new_low_f, new_high_f) = Self::adapt_numeric_bounds(
                    *low as f64,
                    *high as f64,
                    *mean,
                    *std,
                    param.log_scale,
                    config,
                );
                let new_low = (new_low_f.ceil() as i64).max(*low);
                let new_high = (new_high_f.floor() as i64).min(*high).max(new_low + 1);

                adaptation.adapted_bounds = (new_low as f64, new_high as f64);
                adaptation.shrink_ratio = (new_high - new_low) as f64 / (high - low).max(1) as f64;
                adaptation.was_adapted = true;
                adaptation.reason =
                    format!("Adapted from prior (mean={:.4}, std={:.4})", mean, std);

                Parameter {
                    param_type: ParameterType::Int {
                        low: new_low,
                        high: new_high,
                    },
                    log_scale: param.log_scale,
                    default: param.default.clone(),
                }
            }

            // Categorical and Bool parameters don't have their search space shrunk,
            // but priors will guide sampling
            (ParameterType::Categorical { .. }, PriorType::Categorical { .. })
            | (ParameterType::Bool, PriorType::Boolean { .. }) => {
                adaptation.was_adapted = false;
                adaptation.reason =
                    "Categorical/Boolean parameters use prior for sampling only".to_string();
                param.clone()
            }

            _ => {
                adaptation.reason = "Prior type mismatch".to_string();
                param.clone()
            }
        };

        (adapted, adaptation)
    }

    /// Adapt numeric bounds based on prior distribution
    fn adapt_numeric_bounds(
        orig_low: f64,
        orig_high: f64,
        mean: f64,
        std: f64,
        log_scale: bool,
        config: &TransferConfig,
    ) -> (f64, f64) {
        // Z-score for confidence level
        let z = match config.confidence_level {
            c if c >= 0.99 => 2.576,
            c if c >= 0.95 => 1.96,
            c if c >= 0.90 => 1.645,
            _ => 1.282,
        };

        // Compute adapted bounds
        let effective_std = std * config.shrink_factor;
        let mut new_low = mean - z * effective_std;
        let mut new_high = mean + z * effective_std;

        // Apply expansion factor limit
        let center = (orig_low + orig_high) / 2.0;
        let orig_range = orig_high - orig_low;
        let max_range = orig_range * config.max_expansion_factor;

        if new_high - new_low > max_range {
            new_low = center - max_range / 2.0;
            new_high = center + max_range / 2.0;
        }

        // Preserve original bounds if configured
        if config.preserve_original_bounds {
            new_low = new_low.max(orig_low);
            new_high = new_high.min(orig_high);
        }

        // For log scale, ensure positive
        if log_scale {
            new_low = new_low.max(1e-10);
            new_high = new_high.max(new_low + 1e-10);
        }

        // Ensure valid range
        if new_low >= new_high {
            new_high = (orig_high - orig_low).mul_add(0.1, new_low);
        }

        (new_low, new_high)
    }

    /// Get bounds for a parameter
    fn get_bounds(param: &Parameter) -> (f64, f64) {
        match &param.param_type {
            ParameterType::Float { low, high } => (*low, *high),
            ParameterType::Int { low, high } => (*low as f64, *high as f64),
            ParameterType::Categorical { choices } => (0.0, choices.len() as f64),
            ParameterType::Bool => (0.0, 1.0),
        }
    }

    /// Get summary of adaptations
    pub fn summary(&self) -> String {
        let mut lines = vec!["Search Space Transfer Summary:".to_string()];
        lines.push(format!(
            "  Prior from {} configurations (similarity: {:.3})",
            self.prior.n_configurations, self.prior.source_similarity
        ));

        let adapted_count = self.adaptations.values().filter(|a| a.was_adapted).count();
        lines.push(format!(
            "  Adapted {} of {} parameters",
            adapted_count,
            self.adaptations.len()
        ));

        for (name, adaptation) in &self.adaptations {
            if adaptation.was_adapted {
                lines.push(format!(
                    "    {}: [{:.4}, {:.4}] -> [{:.4}, {:.4}] (shrink ratio: {:.2})",
                    name,
                    adaptation.original_bounds.0,
                    adaptation.original_bounds.1,
                    adaptation.adapted_bounds.0,
                    adaptation.adapted_bounds.1,
                    adaptation.shrink_ratio
                ));
            }
        }

        lines.join("\n")
    }
}

/// Sampler that uses warm-start configurations for initial trials
#[derive(Debug, Clone)]
pub struct WarmStartSampler {
    /// Base sampler for exploration after warm-start
    base_seed: Option<u64>,
    /// Warm-start configurations to try first
    warm_configs: Vec<HashMap<String, ParameterValue>>,
    /// Prior knowledge for guided sampling
    prior: PriorKnowledge,
    /// Search space for sampling (kept for future use with bounds validation)
    #[allow(dead_code)]
    search_space: SearchSpace,
    /// Number of warm-start configs consumed
    warm_start_idx: usize,
    /// Probability of sampling from prior vs uniform (after warm-start)
    prior_weight: f64,
}

impl WarmStartSampler {
    /// Create a new warm-start sampler
    pub fn new(warm_start: &WarmStartResult, search_space: SearchSpace) -> Self {
        let warm_configs: Vec<_> = warm_start
            .configurations
            .iter()
            .map(|wc| Self::convert_params(&wc.config.params))
            .collect();

        Self {
            base_seed: None,
            warm_configs,
            prior: PriorKnowledge::from_warm_start(warm_start),
            search_space,
            warm_start_idx: 0,
            prior_weight: 0.5,
        }
    }

    /// Create from prior knowledge directly
    pub fn from_prior(prior: PriorKnowledge, search_space: SearchSpace) -> Self {
        Self {
            base_seed: None,
            warm_configs: Vec::new(),
            prior,
            search_space,
            warm_start_idx: 0,
            prior_weight: 0.5,
        }
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.base_seed = Some(seed);
        self
    }

    /// Set the probability of sampling from prior after warm-start configs are exhausted
    pub fn with_prior_weight(mut self, weight: f64) -> Self {
        self.prior_weight = weight.max(0.0).min(1.0);
        self
    }

    /// Convert ParamValue to ParameterValue
    fn convert_params(params: &HashMap<String, ParamValue>) -> HashMap<String, ParameterValue> {
        params
            .iter()
            .map(|(k, v)| {
                let pv = match v {
                    ParamValue::Int(i) => ParameterValue::Int(*i),
                    ParamValue::Float(f) => ParameterValue::Float(*f),
                    ParamValue::String(s) => ParameterValue::Categorical(s.clone()),
                    ParamValue::Bool(b) => ParameterValue::Bool(*b),
                };
                (k.clone(), pv)
            })
            .collect()
    }

    /// Sample a parameter value using prior-guided sampling
    fn sample_with_prior(&self, name: &str, param: &Parameter, rng: &mut StdRng) -> ParameterValue {
        // Decide whether to use prior or uniform
        let use_prior = rng.random_bool(self.prior_weight);

        if use_prior {
            if let Some(prior_param) = self.prior.parameter_priors.get(name) {
                if let Some(value) = self.sample_from_prior(param, prior_param, rng) {
                    return value;
                }
            }
        }

        // Fall back to uniform sampling
        self.sample_uniform(param, rng)
    }

    /// Sample from prior distribution
    fn sample_from_prior(
        &self,
        param: &Parameter,
        prior_param: &ParameterPrior,
        rng: &mut StdRng,
    ) -> Option<ParameterValue> {
        match (&param.param_type, &prior_param.prior_type) {
            (ParameterType::Float { low, high }, PriorType::Normal { mean, std }) => {
                let value = normal_sample(rng, *mean, *std);
                Some(ParameterValue::Float(value.clamp(*low, *high)))
            }
            (ParameterType::Float { low, high }, PriorType::LogNormal { log_mean, log_std }) => {
                let log_value = normal_sample(rng, *log_mean, *log_std);
                let value = log_value.exp();
                Some(ParameterValue::Float(value.clamp(*low, *high)))
            }
            (ParameterType::Int { low, high }, PriorType::Normal { mean, std }) => {
                let value = normal_sample(rng, *mean, *std);
                Some(ParameterValue::Int(
                    (value.round() as i64).clamp(*low, *high),
                ))
            }
            (ParameterType::Int { low, high }, PriorType::LogNormal { log_mean, log_std }) => {
                let log_value = normal_sample(rng, *log_mean, *log_std);
                let value = log_value.exp();
                Some(ParameterValue::Int(
                    (value.round() as i64).clamp(*low, *high),
                ))
            }
            (ParameterType::Categorical { choices }, PriorType::Categorical { probabilities }) => {
                // Sample based on probabilities
                let mut cumsum = 0.0;
                let u: f64 = rng.random();
                for choice in choices {
                    let p = probabilities.get(choice).copied().unwrap_or(0.0);
                    cumsum += p;
                    if u <= cumsum {
                        return Some(ParameterValue::Categorical(choice.clone()));
                    }
                }
                // Fallback to last choice
                choices
                    .last()
                    .map(|c| ParameterValue::Categorical(c.clone()))
            }
            (ParameterType::Bool, PriorType::Boolean { p_true }) => {
                Some(ParameterValue::Bool(rng.random_bool(*p_true)))
            }
            _ => None,
        }
    }

    /// Sample uniformly from parameter space
    fn sample_uniform(&self, param: &Parameter, rng: &mut StdRng) -> ParameterValue {
        match &param.param_type {
            ParameterType::Float { low, high } => {
                let value = if param.log_scale {
                    let log_low = low.ln();
                    let log_high = high.ln();
                    rng.random_range(log_low..=log_high).exp()
                } else {
                    rng.random_range(*low..=*high)
                };
                ParameterValue::Float(value)
            }
            ParameterType::Int { low, high } => {
                let value = if param.log_scale {
                    let log_low = (*low as f64).ln();
                    let log_high = (*high as f64).ln();
                    rng.random_range(log_low..=log_high).exp() as i64
                } else {
                    rng.random_range(*low..=*high)
                };
                ParameterValue::Int(value)
            }
            ParameterType::Categorical { choices } => {
                let idx = rng.random_range(0..choices.len());
                ParameterValue::Categorical(choices[idx].clone())
            }
            ParameterType::Bool => ParameterValue::Bool(rng.random_bool(0.5)),
        }
    }

    /// Get remaining warm-start configurations count
    pub fn remaining_warm_configs(&self) -> usize {
        self.warm_configs.len().saturating_sub(self.warm_start_idx)
    }
}

impl crate::hpo::samplers::Sampler for WarmStartSampler {
    fn sample(
        &self,
        search_space: &SearchSpace,
        trials: &[Trial],
    ) -> Result<HashMap<String, ParameterValue>> {
        // Use warm-start configs first
        let warm_idx = trials
            .iter()
            .filter(|t| t.state != TrialState::Failed)
            .count();

        if warm_idx < self.warm_configs.len() {
            // Return the next warm-start configuration
            let config = self.warm_configs[warm_idx].clone();

            // Fill in any missing parameters with samples
            let mut rng = match self.base_seed {
                Some(seed) => StdRng::seed_from_u64(seed + warm_idx as u64),
                None => StdRng::from_os_rng(),
            };

            let mut result = config;
            for (name, param) in &search_space.parameters {
                if !result.contains_key(name) {
                    result.insert(name.clone(), self.sample_with_prior(name, param, &mut rng));
                }
            }

            return Ok(result);
        }

        // After warm-start, use prior-guided sampling
        let mut rng = match self.base_seed {
            Some(seed) => StdRng::seed_from_u64(seed + trials.len() as u64),
            None => StdRng::from_os_rng(),
        };

        let mut params = HashMap::new();
        for (name, param) in &search_space.parameters {
            params.insert(name.clone(), self.sample_with_prior(name, param, &mut rng));
        }

        Ok(params)
    }
}

/// Initialize study trials from warm-start configurations
///
/// This function adds warm-start configurations as initial trials to a study,
/// allowing the sampler to build on prior knowledge immediately.
pub fn initialize_study_with_warmstart(
    warm_start: &WarmStartResult,
    _search_space: &SearchSpace,
) -> Vec<Trial> {
    warm_start
        .configurations
        .iter()
        .enumerate()
        .map(|(i, wc)| {
            let params = WarmStartSampler::convert_params(&wc.config.params);
            Trial {
                id: i,
                params,
                value: Some(wc.config.cv_score),
                state: TrialState::Complete,
                intermediate_values: Vec::new(),
                duration: None,
            }
        })
        .collect()
}

/// Get algorithm priority weights from warm-start results
///
/// Returns weights for each algorithm type based on their success on similar datasets.
pub fn algorithm_priorities_from_warmstart(
    warm_start: &WarmStartResult,
) -> HashMap<AlgorithmType, f64> {
    let mut weights: HashMap<AlgorithmType, f64> = HashMap::new();

    for wc in &warm_start.configurations {
        // Higher weight for:
        // 1. Better rank (lower rank = better)
        // 2. Higher similarity
        let rank_weight = 1.0 / wc.config.rank.max(1) as f64;
        let weight = wc.source_similarity * rank_weight;

        *weights.entry(wc.config.algorithm).or_default() += weight;
    }

    // Normalize
    let total: f64 = weights.values().sum();
    if total > 0.0 {
        for v in weights.values_mut() {
            *v /= total;
        }
    }

    weights
}

/// Sample from normal distribution (Box-Muller transform)
fn normal_sample(rng: &mut StdRng, mean: f64, std: f64) -> f64 {
    use std::f64::consts::PI;
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    std.mul_add(z, mean)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automl::warmstart::{ConfigurationRecord, SimilarDataset, WeightedConfiguration};
    use crate::hpo::samplers::Sampler;
    use crate::Task;

    fn create_test_warm_start() -> WarmStartResult {
        let configs = vec![
            WeightedConfiguration {
                config: ConfigurationRecord {
                    algorithm: AlgorithmType::RandomForestClassifier,
                    params: {
                        let mut p = HashMap::new();
                        p.insert("n_estimators".to_string(), ParamValue::Int(100));
                        p.insert("max_depth".to_string(), ParamValue::Int(10));
                        p.insert("learning_rate".to_string(), ParamValue::Float(0.1));
                        p
                    },
                    cv_score: 0.90,
                    cv_std: 0.02,
                    maximize: true,
                    rank: 1,
                },
                source_similarity: 0.85,
                source_dataset: "dataset1".to_string(),
                priority_weight: 0.85,
            },
            WeightedConfiguration {
                config: ConfigurationRecord {
                    algorithm: AlgorithmType::RandomForestClassifier,
                    params: {
                        let mut p = HashMap::new();
                        p.insert("n_estimators".to_string(), ParamValue::Int(150));
                        p.insert("max_depth".to_string(), ParamValue::Int(15));
                        p.insert("learning_rate".to_string(), ParamValue::Float(0.05));
                        p
                    },
                    cv_score: 0.88,
                    cv_std: 0.03,
                    maximize: true,
                    rank: 2,
                },
                source_similarity: 0.80,
                source_dataset: "dataset1".to_string(),
                priority_weight: 0.40,
            },
            WeightedConfiguration {
                config: ConfigurationRecord {
                    algorithm: AlgorithmType::LogisticRegression,
                    params: {
                        let mut p = HashMap::new();
                        p.insert("C".to_string(), ParamValue::Float(1.0));
                        p.insert("learning_rate".to_string(), ParamValue::Float(0.15));
                        p
                    },
                    cv_score: 0.85,
                    cv_std: 0.02,
                    maximize: true,
                    rank: 3,
                },
                source_similarity: 0.75,
                source_dataset: "dataset2".to_string(),
                priority_weight: 0.25,
            },
        ];

        WarmStartResult {
            similar_datasets: vec![
                SimilarDataset {
                    name: "dataset1".to_string(),
                    similarity: 0.85,
                    n_configurations: 2,
                    task: Task::Classification,
                },
                SimilarDataset {
                    name: "dataset2".to_string(),
                    similarity: 0.75,
                    n_configurations: 1,
                    task: Task::Classification,
                },
            ],
            configurations: configs,
            total_configs_found: 3,
            mean_similarity: 0.80,
        }
    }

    #[test]
    fn test_transfer_config_builder() {
        let config = TransferConfig::new()
            .with_shrink_factor(0.3)
            .with_min_configs(5)
            .with_confidence_level(0.99);

        assert!((config.shrink_factor - 0.3).abs() < 1e-10);
        assert_eq!(config.min_configs_for_shrinking, 5);
        assert!((config.confidence_level - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_prior_knowledge_extraction() {
        let warm_start = create_test_warm_start();
        let prior = PriorKnowledge::from_warm_start(&warm_start);

        assert_eq!(prior.n_configurations, 3);
        assert!((prior.source_similarity - 0.80).abs() < 1e-10);
        assert!(prior.parameter_priors.contains_key("learning_rate"));
        assert!(prior.parameter_priors.contains_key("n_estimators"));
    }

    #[test]
    fn test_prior_types() {
        let warm_start = create_test_warm_start();
        let prior = PriorKnowledge::from_warm_start(&warm_start);

        // learning_rate should be log-normal (positive, spans range)
        let lr_prior = prior.parameter_priors.get("learning_rate").unwrap();
        match &lr_prior.prior_type {
            PriorType::Normal { mean, std } => {
                assert!((*mean - 0.1).abs() < 0.05); // Weighted mean around 0.1
                assert!(*std > 0.0);
            }
            PriorType::LogNormal { .. } => {
                // Also acceptable
            }
            _ => panic!("Unexpected prior type for learning_rate"),
        }
    }

    #[test]
    fn test_transferred_search_space() {
        let warm_start = create_test_warm_start();
        let original = SearchSpace::new()
            .int("n_estimators", 10, 500)
            .int("max_depth", 1, 50)
            .float_log("learning_rate", 1e-4, 1.0);

        let transferred = TransferredSearchSpace::from_warm_start(
            &original,
            &warm_start,
            TransferConfig::default(),
        );

        // Check that some parameters were adapted
        let adapted_count = transferred
            .adaptations
            .values()
            .filter(|a| a.was_adapted)
            .count();
        assert!(
            adapted_count > 0,
            "Should have adapted at least one parameter"
        );

        // n_estimators should have shrunk bounds
        let n_est_adaptation = transferred.adaptations.get("n_estimators").unwrap();
        if n_est_adaptation.was_adapted {
            assert!(
                n_est_adaptation.shrink_ratio < 1.0,
                "Search space should be shrunk"
            );
        }
    }

    #[test]
    fn test_conservative_vs_aggressive() {
        let warm_start = create_test_warm_start();
        let original =
            SearchSpace::new()
                .int("n_estimators", 10, 500)
                .float_log("learning_rate", 1e-4, 1.0);

        let conservative = TransferredSearchSpace::from_warm_start(
            &original,
            &warm_start,
            TransferConfig::conservative(),
        );

        let aggressive = TransferredSearchSpace::from_warm_start(
            &original,
            &warm_start,
            TransferConfig::aggressive(),
        );

        // Aggressive should have smaller shrink ratios (narrower search space)
        for name in ["n_estimators", "learning_rate"] {
            let cons_adapt = conservative.adaptations.get(name);
            let aggr_adapt = aggressive.adaptations.get(name);

            if let (Some(c), Some(a)) = (cons_adapt, aggr_adapt) {
                if c.was_adapted && a.was_adapted {
                    assert!(
                        a.shrink_ratio <= c.shrink_ratio,
                        "Aggressive should have smaller or equal shrink ratio for {}",
                        name
                    );
                }
            }
        }
    }

    #[test]
    fn test_warm_start_sampler_initial_configs() {
        let warm_start = create_test_warm_start();
        let search_space = SearchSpace::new()
            .int("n_estimators", 10, 500)
            .int("max_depth", 1, 50)
            .float_log("learning_rate", 1e-4, 1.0);

        let sampler = WarmStartSampler::new(&warm_start, search_space.clone()).with_seed(42);

        // First sample should return first warm-start config
        let sample1 = sampler.sample(&search_space, &[]).unwrap();
        assert!(sample1.contains_key("n_estimators"));

        // Check that we got the first warm config's n_estimators value (100)
        match sample1.get("n_estimators") {
            Some(ParameterValue::Int(n)) => assert_eq!(*n, 100),
            _ => {}
        }
    }

    #[test]
    fn test_warm_start_sampler_after_warmup() {
        let warm_start = create_test_warm_start();
        let search_space =
            SearchSpace::new()
                .int("n_estimators", 10, 500)
                .float_log("learning_rate", 1e-4, 1.0);

        let sampler = WarmStartSampler::new(&warm_start, search_space.clone())
            .with_seed(42)
            .with_prior_weight(0.8);

        // Create fake trials to exhaust warm-start configs
        let fake_trials: Vec<Trial> = (0..5)
            .map(|i| Trial {
                id: i,
                params: HashMap::new(),
                value: Some(0.5),
                state: TrialState::Complete,
                intermediate_values: Vec::new(),
                duration: None,
            })
            .collect();

        // After warm-start configs, should still sample valid parameters
        let sample = sampler.sample(&search_space, &fake_trials).unwrap();
        assert!(sample.contains_key("n_estimators"));
        assert!(sample.contains_key("learning_rate"));

        // Values should be within bounds
        match sample.get("n_estimators") {
            Some(ParameterValue::Int(n)) => assert!(*n >= 10 && *n <= 500),
            _ => {}
        }
        match sample.get("learning_rate") {
            Some(ParameterValue::Float(lr)) => assert!(*lr >= 1e-4 && *lr <= 1.0),
            _ => {}
        }
    }

    #[test]
    fn test_initialize_study_with_warmstart() {
        let warm_start = create_test_warm_start();
        let search_space =
            SearchSpace::new()
                .int("n_estimators", 10, 500)
                .float_log("learning_rate", 1e-4, 1.0);

        let trials = initialize_study_with_warmstart(&warm_start, &search_space);

        assert_eq!(trials.len(), 3);
        assert!(trials.iter().all(|t| t.state == TrialState::Complete));
        assert!(trials.iter().all(|t| t.value.is_some()));

        // Check that scores are preserved
        assert!((trials[0].value.unwrap() - 0.90).abs() < 1e-10);
        assert!((trials[1].value.unwrap() - 0.88).abs() < 1e-10);
    }

    #[test]
    fn test_algorithm_priorities() {
        let warm_start = create_test_warm_start();
        let priorities = algorithm_priorities_from_warmstart(&warm_start);

        assert!(priorities.contains_key(&AlgorithmType::RandomForestClassifier));
        assert!(priorities.contains_key(&AlgorithmType::LogisticRegression));

        // RF should have higher priority (better configs)
        let rf_weight = priorities
            .get(&AlgorithmType::RandomForestClassifier)
            .unwrap();
        let lr_weight = priorities.get(&AlgorithmType::LogisticRegression).unwrap();
        assert!(rf_weight > lr_weight);

        // Weights should sum to 1
        let total: f64 = priorities.values().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_warm_start() {
        let warm_start = WarmStartResult {
            similar_datasets: Vec::new(),
            configurations: Vec::new(),
            total_configs_found: 0,
            mean_similarity: 0.0,
        };

        let prior = PriorKnowledge::from_warm_start(&warm_start);
        assert_eq!(prior.n_configurations, 0);
        assert!(prior.parameter_priors.is_empty());

        let search_space = SearchSpace::new().int("n_estimators", 10, 500);

        let transferred = TransferredSearchSpace::from_warm_start(
            &search_space,
            &warm_start,
            TransferConfig::default(),
        );

        // Should not adapt any parameters with insufficient configs
        for adaptation in transferred.adaptations.values() {
            assert!(!adaptation.was_adapted);
        }
    }

    #[test]
    fn test_categorical_prior() {
        let configs = vec![
            WeightedConfiguration {
                config: ConfigurationRecord {
                    algorithm: AlgorithmType::SVC,
                    params: {
                        let mut p = HashMap::new();
                        p.insert("kernel".to_string(), ParamValue::String("rbf".to_string()));
                        p
                    },
                    cv_score: 0.90,
                    cv_std: 0.02,
                    maximize: true,
                    rank: 1,
                },
                source_similarity: 0.85,
                source_dataset: "dataset1".to_string(),
                priority_weight: 0.85,
            },
            WeightedConfiguration {
                config: ConfigurationRecord {
                    algorithm: AlgorithmType::SVC,
                    params: {
                        let mut p = HashMap::new();
                        p.insert("kernel".to_string(), ParamValue::String("rbf".to_string()));
                        p
                    },
                    cv_score: 0.88,
                    cv_std: 0.03,
                    maximize: true,
                    rank: 2,
                },
                source_similarity: 0.80,
                source_dataset: "dataset1".to_string(),
                priority_weight: 0.40,
            },
            WeightedConfiguration {
                config: ConfigurationRecord {
                    algorithm: AlgorithmType::SVC,
                    params: {
                        let mut p = HashMap::new();
                        p.insert(
                            "kernel".to_string(),
                            ParamValue::String("linear".to_string()),
                        );
                        p
                    },
                    cv_score: 0.85,
                    cv_std: 0.02,
                    maximize: true,
                    rank: 3,
                },
                source_similarity: 0.75,
                source_dataset: "dataset2".to_string(),
                priority_weight: 0.25,
            },
        ];

        let warm_start = WarmStartResult {
            similar_datasets: Vec::new(),
            configurations: configs,
            total_configs_found: 3,
            mean_similarity: 0.80,
        };

        let prior = PriorKnowledge::from_warm_start(&warm_start);
        let kernel_prior = prior.parameter_priors.get("kernel").unwrap();

        match &kernel_prior.prior_type {
            PriorType::Categorical { probabilities } => {
                // rbf should have higher probability
                let p_rbf = probabilities.get("rbf").unwrap_or(&0.0);
                let p_linear = probabilities.get("linear").unwrap_or(&0.0);
                assert!(p_rbf > p_linear);
            }
            _ => panic!("Expected categorical prior"),
        }

        // Test most_likely_categorical
        let most_likely = prior.most_likely_categorical("kernel").unwrap();
        assert_eq!(most_likely, "rbf");
    }

    #[test]
    fn test_boolean_prior() {
        let configs = vec![
            WeightedConfiguration {
                config: ConfigurationRecord {
                    algorithm: AlgorithmType::RandomForestClassifier,
                    params: {
                        let mut p = HashMap::new();
                        p.insert("bootstrap".to_string(), ParamValue::Bool(true));
                        p
                    },
                    cv_score: 0.90,
                    cv_std: 0.02,
                    maximize: true,
                    rank: 1,
                },
                source_similarity: 0.90,
                source_dataset: "dataset1".to_string(),
                priority_weight: 0.90,
            },
            WeightedConfiguration {
                config: ConfigurationRecord {
                    algorithm: AlgorithmType::RandomForestClassifier,
                    params: {
                        let mut p = HashMap::new();
                        p.insert("bootstrap".to_string(), ParamValue::Bool(true));
                        p
                    },
                    cv_score: 0.88,
                    cv_std: 0.03,
                    maximize: true,
                    rank: 2,
                },
                source_similarity: 0.80,
                source_dataset: "dataset1".to_string(),
                priority_weight: 0.40,
            },
            WeightedConfiguration {
                config: ConfigurationRecord {
                    algorithm: AlgorithmType::RandomForestClassifier,
                    params: {
                        let mut p = HashMap::new();
                        p.insert("bootstrap".to_string(), ParamValue::Bool(false));
                        p
                    },
                    cv_score: 0.85,
                    cv_std: 0.02,
                    maximize: true,
                    rank: 3,
                },
                source_similarity: 0.70,
                source_dataset: "dataset2".to_string(),
                priority_weight: 0.23,
            },
        ];

        let warm_start = WarmStartResult {
            similar_datasets: Vec::new(),
            configurations: configs,
            total_configs_found: 3,
            mean_similarity: 0.80,
        };

        let prior = PriorKnowledge::from_warm_start(&warm_start);
        let bootstrap_prior = prior.parameter_priors.get("bootstrap").unwrap();

        match &bootstrap_prior.prior_type {
            PriorType::Boolean { p_true } => {
                // Should favor true (higher weight)
                assert!(*p_true > 0.5);
            }
            _ => panic!("Expected boolean prior"),
        }
    }

    #[test]
    fn test_transfer_summary() {
        let warm_start = create_test_warm_start();
        let original =
            SearchSpace::new()
                .int("n_estimators", 10, 500)
                .float_log("learning_rate", 1e-4, 1.0);

        let transferred = TransferredSearchSpace::from_warm_start(
            &original,
            &warm_start,
            TransferConfig::default(),
        );

        let summary = transferred.summary();
        assert!(summary.contains("Search Space Transfer Summary"));
        assert!(summary.contains("configurations"));
    }

    #[test]
    fn test_sampler_remaining_warm_configs() {
        let warm_start = create_test_warm_start();
        let search_space = SearchSpace::new().int("n_estimators", 10, 500);

        let sampler = WarmStartSampler::new(&warm_start, search_space);
        assert_eq!(sampler.remaining_warm_configs(), 3);
    }
}

//! Schedulers for early stopping (pruning)
//!
//! This module provides multi-fidelity optimization schedulers:
//! - MedianPruner: Simple median-based pruning
//! - HyperbandScheduler: Full Hyperband algorithm with brackets
//! - ASHAScheduler: Asynchronous Successive Halving

use super::{Trial, TrialState};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait for pruning schedulers
pub trait Scheduler: Send + Sync {
    /// Determine if a trial should be pruned
    fn should_prune(&self, trials: &[Trial], trial_id: usize, step: usize) -> bool;
}

/// Callback trait for early stopping events
pub trait EarlyStoppingCallback: Send + Sync {
    /// Called when a trial is pruned
    fn on_trial_pruned(&self, trial_id: usize, step: usize, value: f64);

    /// Called when a trial completes a rung
    fn on_rung_completed(&self, trial_id: usize, rung: usize, value: f64);

    /// Called when a bracket completes
    fn on_bracket_completed(&self, bracket_id: usize, best_trial_id: usize, best_value: f64);
}

/// Fidelity parameter specification for multi-fidelity optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FidelityParameter {
    /// Discrete fidelity (e.g., number of epochs, trees)
    Discrete {
        /// Name of the fidelity parameter
        name: String,
        /// Minimum fidelity value
        min: usize,
        /// Maximum fidelity value
        max: usize,
    },
    /// Continuous fidelity (e.g., data fraction, learning rate schedule fraction)
    Continuous {
        /// Name of the fidelity parameter
        name: String,
        /// Minimum fidelity value
        min: f64,
        /// Maximum fidelity value
        max: f64,
    },
}

impl FidelityParameter {
    /// Create a discrete fidelity parameter
    pub fn discrete(name: impl Into<String>, min: usize, max: usize) -> Self {
        Self::Discrete {
            name: name.into(),
            min,
            max,
        }
    }

    /// Create a continuous fidelity parameter
    pub fn continuous(name: impl Into<String>, min: f64, max: f64) -> Self {
        Self::Continuous {
            name: name.into(),
            min,
            max,
        }
    }

    /// Get the parameter name
    pub fn name(&self) -> &str {
        match self {
            Self::Discrete { name, .. } => name,
            Self::Continuous { name, .. } => name,
        }
    }

    /// Get the minimum value as f64
    pub fn min_value(&self) -> f64 {
        match self {
            Self::Discrete { min, .. } => *min as f64,
            Self::Continuous { min, .. } => *min,
        }
    }

    /// Get the maximum value as f64
    pub fn max_value(&self) -> f64 {
        match self {
            Self::Discrete { max, .. } => *max as f64,
            Self::Continuous { max, .. } => *max,
        }
    }

    /// Compute the fidelity value at a given rung
    pub fn value_at_rung(&self, rung: usize, n_rungs: usize, reduction_factor: f64) -> f64 {
        let fraction = reduction_factor.powi(-(n_rungs as i32 - rung as i32 - 1));
        match self {
            Self::Discrete { min, max, .. } => {
                let value = (*min as f64 * fraction)
                    .round()
                    .max(*min as f64)
                    .min(*max as f64);
                value.round()
            }
            Self::Continuous { min, max, .. } => (*min * fraction).max(*min).min(*max),
        }
    }
}

/// Performance metrics for a single rung
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RungMetrics {
    /// Rung index
    pub rung: usize,
    /// Fidelity value at this rung
    pub fidelity: f64,
    /// Number of trials that reached this rung
    pub n_trials: usize,
    /// Number of trials promoted to next rung
    pub n_promoted: usize,
    /// Best value seen at this rung
    pub best_value: Option<f64>,
    /// Mean value at this rung
    pub mean_value: Option<f64>,
    /// Standard deviation at this rung
    pub std_value: Option<f64>,
    /// Trial IDs that reached this rung
    pub trial_ids: Vec<usize>,
}

impl RungMetrics {
    /// Update metrics with a new trial result
    pub fn add_trial(&mut self, trial_id: usize, value: f64) {
        self.trial_ids.push(trial_id);
        self.n_trials += 1;

        // Update best
        if self.best_value.is_none() || value < self.best_value.unwrap() {
            self.best_value = Some(value);
        }

        // Update mean and std (incremental)
        if self.mean_value.is_none() {
            self.mean_value = Some(value);
            self.std_value = Some(0.0);
        } else {
            let old_mean = self.mean_value.unwrap();
            let new_mean = old_mean + (value - old_mean) / self.n_trials as f64;
            let old_m2 = self.std_value.unwrap().powi(2) * (self.n_trials - 1) as f64;
            let new_m2 = (value - old_mean).mul_add(value - new_mean, old_m2);
            self.mean_value = Some(new_mean);
            self.std_value = Some((new_m2 / self.n_trials as f64).sqrt());
        }
    }
}

/// Performance metrics for a single bracket
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BracketMetrics {
    /// Bracket index
    pub bracket_id: usize,
    /// Initial number of configurations in this bracket
    pub n_initial_configs: usize,
    /// Minimum fidelity for this bracket
    pub min_fidelity: f64,
    /// Maximum fidelity for this bracket
    pub max_fidelity: f64,
    /// Number of rungs in this bracket
    pub n_rungs: usize,
    /// Metrics per rung
    pub rung_metrics: Vec<RungMetrics>,
    /// Whether bracket is complete
    pub is_complete: bool,
    /// Best trial ID from this bracket
    pub best_trial_id: Option<usize>,
    /// Best value from this bracket
    pub best_value: Option<f64>,
    /// Total computational cost (sum of fidelities used)
    pub total_cost: f64,
}

/// Overall Hyperband metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HyperbandMetrics {
    /// Metrics per bracket
    pub bracket_metrics: Vec<BracketMetrics>,
    /// Total trials started
    pub total_trials: usize,
    /// Total trials pruned
    pub total_pruned: usize,
    /// Total trials completed
    pub total_completed: usize,
    /// Best trial ID across all brackets
    pub best_trial_id: Option<usize>,
    /// Best value across all brackets
    pub best_value: Option<f64>,
    /// Total computational cost
    pub total_cost: f64,
}

impl HyperbandMetrics {
    /// Get the pruning rate
    pub fn pruning_rate(&self) -> f64 {
        if self.total_trials == 0 {
            0.0
        } else {
            self.total_pruned as f64 / self.total_trials as f64
        }
    }

    /// Get the efficiency (best value per unit cost)
    pub fn efficiency(&self) -> Option<f64> {
        self.best_value.map(|v| v.abs() / self.total_cost.max(1.0))
    }
}

/// Median pruner - prunes trials performing worse than the median
#[derive(Debug, Clone)]
pub struct MedianPruner {
    n_startup_trials: usize,
    n_warmup_steps: usize,
    percentile: f64,
}

impl Default for MedianPruner {
    fn default() -> Self {
        Self {
            n_startup_trials: 5,
            n_warmup_steps: 0,
            percentile: 50.0,
        }
    }
}

impl MedianPruner {
    /// Create a new median pruner with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of startup trials before pruning begins
    pub fn with_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = n;
        self
    }

    /// Set the number of warmup steps before pruning is considered
    pub fn with_warmup_steps(mut self, n: usize) -> Self {
        self.n_warmup_steps = n;
        self
    }

    /// Set the percentile threshold (default: 50.0 for median)
    /// A trial is pruned if worse than this percentile of completed trials
    pub fn with_percentile(mut self, percentile: f64) -> Self {
        self.percentile = percentile.clamp(0.0, 100.0);
        self
    }
}

impl Scheduler for MedianPruner {
    fn should_prune(&self, trials: &[Trial], trial_id: usize, step: usize) -> bool {
        if step < self.n_warmup_steps {
            return false;
        }

        let completed: Vec<&Trial> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        if completed.len() < self.n_startup_trials {
            return false;
        }

        let current_trial = &trials[trial_id];
        let current_value = match current_trial.intermediate_values.last() {
            Some(v) => *v,
            None => return false,
        };

        // Get values at this step from completed trials
        let mut values_at_step: Vec<f64> = completed
            .iter()
            .filter_map(|t| t.intermediate_values.get(step).copied())
            .collect();

        if values_at_step.is_empty() {
            return false;
        }

        values_at_step.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((self.percentile / 100.0) * (values_at_step.len() - 1) as f64).round() as usize;
        let threshold = values_at_step[idx.min(values_at_step.len() - 1)];

        current_value > threshold // Prune if worse than threshold (assuming minimization)
    }
}

/// Configuration for Hyperband scheduler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbandConfig {
    /// Minimum resource/fidelity value
    pub min_resource: usize,
    /// Maximum resource/fidelity value
    pub max_resource: usize,
    /// Reduction factor (eta) - controls bracket sizes
    pub reduction_factor: f64,
    /// Optional fidelity parameter specification
    pub fidelity: Option<FidelityParameter>,
}

impl Default for HyperbandConfig {
    fn default() -> Self {
        Self {
            min_resource: 1,
            max_resource: 81,
            reduction_factor: 3.0,
            fidelity: None,
        }
    }
}

/// A bracket in Hyperband - represents a successive halving run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bracket {
    /// Bracket index (s in the paper)
    pub id: usize,
    /// Initial number of configurations
    pub n_configs: usize,
    /// Minimum resource for this bracket
    pub min_resource: usize,
    /// Maximum resource
    pub max_resource: usize,
    /// Number of rungs
    pub n_rungs: usize,
    /// Resource values at each rung
    pub rung_resources: Vec<usize>,
    /// Number of configs to keep at each rung
    pub rung_n_configs: Vec<usize>,
    /// Trial IDs currently in this bracket (by rung)
    pub trials_per_rung: Vec<Vec<usize>>,
    /// Values at each rung (trial_id -> value)
    pub values_per_rung: Vec<HashMap<usize, f64>>,
    /// Current rung being processed
    pub current_rung: usize,
    /// Trials that have been promoted (can continue)
    pub promoted_trials: Vec<usize>,
    /// Whether this bracket is complete
    pub is_complete: bool,
}

impl Bracket {
    /// Create a new bracket
    pub fn new(
        id: usize,
        n_configs: usize,
        min_resource: usize,
        max_resource: usize,
        reduction_factor: f64,
    ) -> Self {
        let n_rungs = ((max_resource as f64 / min_resource as f64)
            .log(reduction_factor)
            .floor() as usize)
            + 1;

        // Compute resources at each rung
        let mut rung_resources = Vec::with_capacity(n_rungs);
        let mut rung_n_configs = Vec::with_capacity(n_rungs);
        let mut current_resource = min_resource;
        let mut current_n = n_configs;

        for _ in 0..n_rungs {
            rung_resources.push(current_resource);
            rung_n_configs.push(current_n);
            current_resource =
                (current_resource as f64 * reduction_factor).min(max_resource as f64) as usize;
            current_n = (current_n as f64 / reduction_factor).floor().max(1.0) as usize;
        }

        let trials_per_rung = vec![Vec::new(); n_rungs];
        let values_per_rung = vec![HashMap::new(); n_rungs];

        Self {
            id,
            n_configs,
            min_resource,
            max_resource,
            n_rungs,
            rung_resources,
            rung_n_configs,
            trials_per_rung,
            values_per_rung,
            current_rung: 0,
            promoted_trials: Vec::new(),
            is_complete: false,
        }
    }

    /// Get the resource allocation for a given rung
    pub fn resource_at_rung(&self, rung: usize) -> Option<usize> {
        self.rung_resources.get(rung).copied()
    }

    /// Register a trial at a specific rung with its value
    pub fn register_trial(&mut self, trial_id: usize, rung: usize, value: f64) {
        if rung < self.n_rungs {
            self.trials_per_rung[rung].push(trial_id);
            self.values_per_rung[rung].insert(trial_id, value);
        }
    }

    /// Check if a trial should continue to the next rung
    pub fn should_promote(&self, trial_id: usize, rung: usize) -> bool {
        if rung >= self.n_rungs - 1 {
            return false; // Already at last rung
        }

        let values = &self.values_per_rung[rung];
        if values.len() < 2 {
            return true; // Not enough data to prune
        }

        // Sort trials by value (ascending for minimization)
        let mut sorted: Vec<_> = values.iter().map(|(&id, &v)| (id, v)).collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Promote top n_configs for next rung
        let n_promote = self.rung_n_configs.get(rung + 1).copied().unwrap_or(1);
        let promote_ids: Vec<usize> = sorted.iter().take(n_promote).map(|(id, _)| *id).collect();

        promote_ids.contains(&trial_id)
    }

    /// Promote trials from current rung to next
    pub fn promote_to_next_rung(&mut self) -> Vec<usize> {
        if self.current_rung >= self.n_rungs - 1 {
            self.is_complete = true;
            return vec![];
        }

        let values = &self.values_per_rung[self.current_rung];
        if values.is_empty() {
            return vec![];
        }

        let mut sorted: Vec<_> = values.iter().map(|(&id, &v)| (id, v)).collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let n_promote = self
            .rung_n_configs
            .get(self.current_rung + 1)
            .copied()
            .unwrap_or(1);
        let promoted: Vec<usize> = sorted.iter().take(n_promote).map(|(id, _)| *id).collect();

        self.promoted_trials.extend(&promoted);
        self.current_rung += 1;

        if self.current_rung >= self.n_rungs - 1 {
            self.is_complete = true;
        }

        promoted
    }

    /// Get the best trial from this bracket
    pub fn best_trial(&self) -> Option<(usize, f64)> {
        // Look at the highest completed rung
        for rung in (0..self.n_rungs).rev() {
            if !self.values_per_rung[rung].is_empty() {
                return self.values_per_rung[rung]
                    .iter()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(&id, &v)| (id, v));
            }
        }
        None
    }
}

/// Enhanced Hyperband scheduler with proper bracket management
///
/// Hyperband is a multi-fidelity hyperparameter optimization algorithm that
/// dynamically allocates resources to promising configurations. It runs
/// multiple brackets of successive halving with different tradeoffs between
/// exploration (many configs, low budget) and exploitation (few configs, high budget).
///
/// # Algorithm
///
/// For max_resource=81, min_resource=1, eta=3:
/// - Bracket 0 (s=4): 81 configs × 1 resource → 27 × 3 → 9 × 9 → 3 × 27 → 1 × 81
/// - Bracket 1 (s=3): 34 configs × 3 resource → 11 × 9 → 3 × 27 → 1 × 81
/// - Bracket 2 (s=2): 15 configs × 9 resource → 5 × 27 → 1 × 81
/// - Bracket 3 (s=1): 8 configs × 27 resource → 2 × 81
/// - Bracket 4 (s=0): 5 configs × 81 resource
///
/// # Example
///
/// ```
/// use ferroml_core::hpo::schedulers::{HyperbandScheduler, HyperbandConfig, FidelityParameter};
///
/// let scheduler = HyperbandScheduler::new(1, 81, 3.0)
///     .with_fidelity(FidelityParameter::discrete("epochs", 1, 81));
///
/// // Get bracket allocation
/// let brackets = scheduler.brackets();
/// assert!(brackets.len() > 0);
/// ```
#[derive(Debug, Clone)]
pub struct HyperbandScheduler {
    config: HyperbandConfig,
    /// All brackets in this Hyperband run
    brackets: Vec<Bracket>,
    /// Current bracket being filled
    current_bracket: usize,
    /// Trial to bracket mapping
    trial_bracket: HashMap<usize, usize>,
    /// Metrics collection
    metrics: HyperbandMetrics,
}

impl Default for HyperbandScheduler {
    fn default() -> Self {
        Self::new(1, 81, 3.0)
    }
}

impl HyperbandScheduler {
    /// Create a new Hyperband scheduler
    pub fn new(min_resource: usize, max_resource: usize, reduction_factor: f64) -> Self {
        let config = HyperbandConfig {
            min_resource,
            max_resource,
            reduction_factor,
            fidelity: None,
        };

        let brackets = Self::compute_brackets(&config);
        let n_brackets = brackets.len();

        Self {
            config,
            brackets,
            current_bracket: 0,
            trial_bracket: HashMap::new(),
            metrics: HyperbandMetrics {
                bracket_metrics: (0..n_brackets)
                    .map(|i| BracketMetrics {
                        bracket_id: i,
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            },
        }
    }

    /// Create from config
    pub fn from_config(config: HyperbandConfig) -> Self {
        let brackets = Self::compute_brackets(&config);
        let n_brackets = brackets.len();

        Self {
            config,
            brackets,
            current_bracket: 0,
            trial_bracket: HashMap::new(),
            metrics: HyperbandMetrics {
                bracket_metrics: (0..n_brackets)
                    .map(|i| BracketMetrics {
                        bracket_id: i,
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            },
        }
    }

    /// Set the fidelity parameter
    pub fn with_fidelity(mut self, fidelity: FidelityParameter) -> Self {
        self.config.fidelity = Some(fidelity);
        self
    }

    /// Get the configuration
    pub fn config(&self) -> &HyperbandConfig {
        &self.config
    }

    /// Get all brackets
    pub fn brackets(&self) -> &[Bracket] {
        &self.brackets
    }

    /// Get mutable brackets
    pub fn brackets_mut(&mut self) -> &mut [Bracket] {
        &mut self.brackets
    }

    /// Get metrics
    pub fn metrics(&self) -> &HyperbandMetrics {
        &self.metrics
    }

    /// Compute all brackets for given configuration
    fn compute_brackets(config: &HyperbandConfig) -> Vec<Bracket> {
        let eta = config.reduction_factor;
        let max_r = config.max_resource;
        let min_r = config.min_resource;

        // s_max = floor(log_eta(max_r / min_r))
        let s_max = (max_r as f64 / min_r as f64).log(eta).floor() as usize;

        let mut brackets = Vec::with_capacity(s_max + 1);

        for s in (0..=s_max).rev() {
            // n = ceil((s_max + 1) / (s + 1)) * eta^s
            let n_configs =
                ((s_max + 1) as f64 / (s + 1) as f64 * eta.powi(s as i32)).ceil() as usize;
            // r = max_r * eta^(-s)
            let bracket_min_resource = (max_r as f64 * eta.powi(-(s as i32))).round() as usize;

            let bracket = Bracket::new(
                s_max - s,
                n_configs.max(1),
                bracket_min_resource.max(min_r),
                max_r,
                eta,
            );
            brackets.push(bracket);
        }

        brackets
    }

    /// Get the recommended resource for a new trial
    pub fn get_resource_for_trial(&self, trial_id: usize) -> usize {
        if let Some(&bracket_id) = self.trial_bracket.get(&trial_id) {
            if let Some(bracket) = self.brackets.get(bracket_id) {
                return bracket.min_resource;
            }
        }
        self.config.min_resource
    }

    /// Assign a trial to the current bracket
    pub fn assign_trial(&mut self, trial_id: usize) -> usize {
        let bracket_id = self.current_bracket;
        self.trial_bracket.insert(trial_id, bracket_id);

        // Update metrics
        self.metrics.total_trials += 1;

        // Move to next bracket if current is full
        if let Some(bracket) = self.brackets.get(bracket_id) {
            let assigned_count = self
                .trial_bracket
                .values()
                .filter(|&&b| b == bracket_id)
                .count();
            if assigned_count >= bracket.n_configs {
                self.current_bracket = (self.current_bracket + 1) % self.brackets.len();
            }
        }

        bracket_id
    }

    /// Report a value for a trial at a specific step/resource
    pub fn report_value(&mut self, trial_id: usize, step: usize, value: f64) {
        if let Some(&bracket_id) = self.trial_bracket.get(&trial_id) {
            if let Some(bracket) = self.brackets.get_mut(bracket_id) {
                // Find which rung this step corresponds to
                for (rung, &rung_resource) in bracket.rung_resources.iter().enumerate() {
                    if step == rung_resource {
                        bracket.register_trial(trial_id, rung, value);

                        // Update rung metrics
                        if let Some(bracket_metrics) =
                            self.metrics.bracket_metrics.get_mut(bracket_id)
                        {
                            if bracket_metrics.rung_metrics.len() <= rung {
                                bracket_metrics.rung_metrics.resize(
                                    rung + 1,
                                    RungMetrics {
                                        rung,
                                        fidelity: rung_resource as f64,
                                        ..Default::default()
                                    },
                                );
                            }
                            bracket_metrics.rung_metrics[rung].add_trial(trial_id, value);
                            bracket_metrics.total_cost += rung_resource as f64;
                        }

                        // Update global metrics
                        self.metrics.total_cost += rung_resource as f64;
                        if self.metrics.best_value.is_none()
                            || value < self.metrics.best_value.unwrap()
                        {
                            self.metrics.best_value = Some(value);
                            self.metrics.best_trial_id = Some(trial_id);
                        }

                        break;
                    }
                }
            }
        }
    }

    /// Get the number of rungs
    pub fn n_rungs(&self) -> usize {
        ((self.config.max_resource as f64 / self.config.min_resource as f64)
            .log(self.config.reduction_factor)
            .floor() as usize)
            + 1
    }

    /// Get resource at each rung for the given bracket
    pub fn rung_resources(&self, bracket_id: usize) -> Option<&[usize]> {
        self.brackets
            .get(bracket_id)
            .map(|b| b.rung_resources.as_slice())
    }

    /// Get the fidelity value for a given step
    pub fn fidelity_at_step(&self, step: usize) -> f64 {
        if let Some(ref fidelity) = self.config.fidelity {
            // Map step to fidelity value
            match fidelity {
                FidelityParameter::Discrete { min, max, .. } => {
                    (step as f64).clamp(*min as f64, *max as f64)
                }
                FidelityParameter::Continuous { min, max, .. } => {
                    let fraction = step as f64 / self.config.max_resource as f64;
                    min + fraction * (max - min)
                }
            }
        } else {
            step as f64
        }
    }

    /// Check if all brackets are complete
    pub fn is_complete(&self) -> bool {
        self.brackets.iter().all(|b| b.is_complete)
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Hyperband Summary (eta={}, R={}):\n",
            self.config.reduction_factor, self.config.max_resource
        ));
        s.push_str(&format!("  Brackets: {}\n", self.brackets.len()));
        s.push_str(&format!(
            "  Total trials: {} (pruned: {}, completed: {})\n",
            self.metrics.total_trials, self.metrics.total_pruned, self.metrics.total_completed
        ));
        if let Some(best) = self.metrics.best_value {
            s.push_str(&format!(
                "  Best value: {:.6} (trial {})\n",
                best,
                self.metrics.best_trial_id.unwrap_or(0)
            ));
        }
        s.push_str(&format!(
            "  Pruning rate: {:.1}%\n",
            self.metrics.pruning_rate() * 100.0
        ));
        s
    }
}

impl Scheduler for HyperbandScheduler {
    fn should_prune(&self, trials: &[Trial], trial_id: usize, step: usize) -> bool {
        // Get the bracket for this trial
        let bracket_id = match self.trial_bracket.get(&trial_id) {
            Some(&id) => id,
            None => return false, // Trial not assigned to a bracket
        };

        let bracket = match self.brackets.get(bracket_id) {
            Some(b) => b,
            None => return false,
        };

        // Check if this step is a rung boundary
        let rung_idx = bracket.rung_resources.iter().position(|&r| r == step);
        let rung_idx = match rung_idx {
            Some(idx) => idx,
            None => return false, // Not at a rung boundary
        };

        // If at the last rung, don't prune
        if rung_idx >= bracket.n_rungs - 1 {
            return false;
        }

        // Get all trials' values at this rung
        let trials_at_rung: Vec<(usize, f64)> = trials
            .iter()
            .enumerate()
            .filter(|(i, t)| {
                // Only consider trials in the same bracket
                self.trial_bracket.get(i) == Some(&bracket_id)
                    && t.intermediate_values.get(step).is_some()
            })
            .filter_map(|(i, t)| t.intermediate_values.get(step).map(|&v| (i, v)))
            .collect();

        if trials_at_rung.len() < 2 {
            return false;
        }

        // Sort by value (ascending for minimization)
        let mut sorted = trials_at_rung;
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Keep top 1/eta configs (use floor to ensure reduction, with min 1)
        let n_keep = (sorted.len() as f64 / self.config.reduction_factor)
            .floor()
            .max(1.0) as usize;
        let keep_ids: Vec<usize> = sorted[..n_keep].iter().map(|(i, _)| *i).collect();

        !keep_ids.contains(&trial_id)
    }
}

/// ASHA (Asynchronous Successive Halving Algorithm) scheduler
#[derive(Debug, Clone)]
pub struct ASHAScheduler {
    min_resource: usize,
    reduction_factor: f64,
    grace_period: usize,
}

impl Default for ASHAScheduler {
    fn default() -> Self {
        Self {
            min_resource: 1,
            reduction_factor: 4.0,
            grace_period: 1,
        }
    }
}

impl ASHAScheduler {
    /// Create a new ASHA scheduler with the specified parameters
    pub fn new(min_resource: usize, reduction_factor: f64, grace_period: usize) -> Self {
        Self {
            min_resource,
            reduction_factor,
            grace_period,
        }
    }
}

impl Scheduler for ASHAScheduler {
    fn should_prune(&self, trials: &[Trial], trial_id: usize, step: usize) -> bool {
        if step < self.grace_period {
            return false;
        }

        // Check at each rung (power of reduction_factor * min_resource)
        let mut rung = self.min_resource;
        while rung <= step {
            if step == rung {
                let current_trial = &trials[trial_id];
                let current_value = match current_trial.intermediate_values.get(step) {
                    Some(v) => *v,
                    None => return false,
                };

                // Count how many trials have reached this rung
                let trials_at_rung: Vec<f64> = trials
                    .iter()
                    .filter_map(|t| t.intermediate_values.get(step).copied())
                    .collect();

                if trials_at_rung.len() < 2 {
                    return false;
                }

                // Prune if not in top 1/reduction_factor
                let mut sorted = trials_at_rung;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let n_keep = (sorted.len() as f64 / self.reduction_factor)
                    .floor()
                    .max(1.0) as usize;
                let threshold = sorted[n_keep.min(sorted.len() - 1)];

                return current_value > threshold;
            }
            rung = (rung as f64 * self.reduction_factor) as usize;
        }

        false
    }
}

/// Configuration for BOHB (Bayesian Optimization and Hyperband)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BOHBConfig {
    /// Minimum resource/fidelity value
    pub min_resource: usize,
    /// Maximum resource/fidelity value
    pub max_resource: usize,
    /// Reduction factor (eta)
    pub reduction_factor: f64,
    /// Number of startup trials with random sampling before using KDE
    pub n_startup_trials: usize,
    /// Fraction of trials considered "good" for KDE (gamma in TPE)
    pub gamma: f64,
    /// Minimum bandwidth for KDE (prevents overfitting with few samples)
    pub min_bandwidth: f64,
    /// Maximum bandwidth for KDE
    pub max_bandwidth: f64,
    /// Number of samples to draw for EI optimization
    pub n_ei_candidates: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Bandwidth factor multiplier
    pub bandwidth_factor: f64,
}

impl Default for BOHBConfig {
    fn default() -> Self {
        Self {
            min_resource: 1,
            max_resource: 81,
            reduction_factor: 3.0,
            n_startup_trials: 10,
            gamma: 0.15, // Top 15% are "good"
            min_bandwidth: 1e-3,
            max_bandwidth: 1.0,
            n_ei_candidates: 64,
            random_state: None,
            bandwidth_factor: 3.0, // Scott's rule multiplier
        }
    }
}

impl BOHBConfig {
    /// Create a new BOHB configuration
    pub fn new(min_resource: usize, max_resource: usize, reduction_factor: f64) -> Self {
        Self {
            min_resource,
            max_resource,
            reduction_factor,
            ..Default::default()
        }
    }
}

/// Kernel Density Estimation (KDE) for BOHB
///
/// Uses multivariate KDE with bandwidth selection based on data
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct KernelDensityEstimator {
    /// Training points (n_samples x n_dims)
    data: Vec<Vec<f64>>,
    /// Bandwidth per dimension
    bandwidths: Vec<f64>,
    /// Minimum bandwidth
    min_bandwidth: f64,
    /// Maximum bandwidth
    max_bandwidth: f64,
}

impl KernelDensityEstimator {
    /// Create a new KDE with the given data
    fn new(
        data: Vec<Vec<f64>>,
        min_bandwidth: f64,
        max_bandwidth: f64,
        bandwidth_factor: f64,
    ) -> Self {
        let bandwidths = if data.is_empty() {
            vec![]
        } else {
            Self::compute_bandwidths(&data, min_bandwidth, max_bandwidth, bandwidth_factor)
        };

        Self {
            data,
            bandwidths,
            min_bandwidth,
            max_bandwidth,
        }
    }

    /// Compute bandwidths using Scott's rule with adjustments
    fn compute_bandwidths(data: &[Vec<f64>], min_bw: f64, max_bw: f64, factor: f64) -> Vec<f64> {
        let n = data.len() as f64;
        let d = if data.is_empty() { 1 } else { data[0].len() };

        // Scott's rule: h = n^(-1/(d+4)) * std
        let scott_factor = n.powf(-1.0 / (d as f64 + 4.0)) * factor;

        let mut bandwidths = Vec::with_capacity(d);

        for dim in 0..d {
            let values: Vec<f64> = data.iter().map(|x| x[dim]).collect();
            let mean = values.iter().sum::<f64>() / n;
            let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            let std = variance.sqrt().max(1e-6);

            let bw = (scott_factor * std).clamp(min_bw, max_bw);
            bandwidths.push(bw);
        }

        bandwidths
    }

    /// Evaluate log probability density at a point
    fn log_pdf(&self, x: &[f64]) -> f64 {
        if self.data.is_empty() {
            return f64::NEG_INFINITY;
        }

        let n = self.data.len() as f64;
        let d = x.len();

        // Sum of log Gaussian kernels
        let mut log_sum = f64::NEG_INFINITY;

        for data_point in &self.data {
            let mut log_kernel = 0.0;
            for dim in 0..d {
                let bw = self
                    .bandwidths
                    .get(dim)
                    .copied()
                    .unwrap_or(self.min_bandwidth);
                let diff = (x[dim] - data_point[dim]) / bw;
                // log of Gaussian kernel: -0.5 * diff^2 - log(bw) - 0.5*log(2*pi)
                log_kernel += 0.5f64.mul_add(
                    -(2.0 * std::f64::consts::PI).ln(),
                    (-0.5 * diff).mul_add(diff, -bw.ln()),
                );
            }
            log_sum = log_add_exp(log_sum, log_kernel);
        }

        // Normalize by number of samples
        log_sum - n.ln()
    }

    /// Sample from the KDE distribution
    fn sample(&self, rng: &mut impl rand::Rng) -> Vec<f64> {
        if self.data.is_empty() {
            return vec![];
        }

        // Sample a data point uniformly
        let idx = rng.random_range(0..self.data.len());
        let center = &self.data[idx];

        // Add Gaussian noise with bandwidths
        center
            .iter()
            .enumerate()
            .map(|(dim, &val)| {
                let bw = self
                    .bandwidths
                    .get(dim)
                    .copied()
                    .unwrap_or(self.min_bandwidth);
                bw.mul_add(sample_standard_normal(rng), val)
            })
            .collect()
    }
}

/// Log-sum-exp for numerical stability
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else {
        let max_val = a.max(b);
        max_val + ((-max_val + a).exp() + (-max_val + b).exp()).ln()
    }
}

/// Sample from standard normal distribution using Box-Muller
fn sample_standard_normal(rng: &mut impl rand::Rng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-10);
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// BOHB Sampler - Kernel density estimation-based sampler
///
/// Implements the sampling strategy from BOHB (Falkner et al., 2018):
/// - Maintains KDEs for good (l) and bad (g) configurations
/// - Samples from l(x)/g(x) to maximize expected improvement
/// - Handles categorical, continuous, and integer parameters
#[derive(Debug, Clone)]
pub struct BOHBSampler {
    /// Configuration
    config: BOHBConfig,
    /// Good configurations (top gamma %)
    good_configs: Vec<Vec<f64>>,
    /// Bad configurations (bottom 1-gamma %)
    bad_configs: Vec<Vec<f64>>,
    /// KDE for good configurations
    kde_good: Option<KernelDensityEstimator>,
    /// KDE for bad configurations
    kde_bad: Option<KernelDensityEstimator>,
    /// Parameter names in order
    param_names: Vec<String>,
    /// Parameter info for encoding/decoding
    param_info: Vec<ParamInfo>,
    /// Random state
    rng_seed: u64,
}

/// Information about a parameter for encoding
#[derive(Debug, Clone)]
struct ParamInfo {
    name: String,
    is_categorical: bool,
    n_choices: usize, // For categorical
    low: f64,         // For continuous/int
    high: f64,        // For continuous/int
    log_scale: bool,
    is_int: bool,
}

impl Default for BOHBSampler {
    fn default() -> Self {
        Self::new(BOHBConfig::default())
    }
}

impl BOHBSampler {
    /// Create a new BOHB sampler
    pub fn new(config: BOHBConfig) -> Self {
        Self {
            rng_seed: config.random_state.unwrap_or(42),
            config,
            good_configs: Vec::new(),
            bad_configs: Vec::new(),
            kde_good: None,
            kde_bad: None,
            param_names: Vec::new(),
            param_info: Vec::new(),
        }
    }

    /// Initialize parameter info from search space
    fn init_param_info(&mut self, search_space: &super::SearchSpace) {
        if !self.param_info.is_empty() {
            return; // Already initialized
        }

        self.param_names.clear();
        self.param_info.clear();

        for (name, param) in &search_space.parameters {
            self.param_names.push(name.clone());

            let info = match &param.param_type {
                super::ParameterType::Categorical { choices } => ParamInfo {
                    name: name.clone(),
                    is_categorical: true,
                    n_choices: choices.len(),
                    low: 0.0,
                    high: (choices.len() - 1) as f64,
                    log_scale: false,
                    is_int: false,
                },
                super::ParameterType::Float { low, high } => ParamInfo {
                    name: name.clone(),
                    is_categorical: false,
                    n_choices: 0,
                    low: if param.log_scale { low.ln() } else { *low },
                    high: if param.log_scale { high.ln() } else { *high },
                    log_scale: param.log_scale,
                    is_int: false,
                },
                super::ParameterType::Int { low, high } => ParamInfo {
                    name: name.clone(),
                    is_categorical: false,
                    n_choices: 0,
                    low: if param.log_scale {
                        (*low as f64).ln()
                    } else {
                        *low as f64
                    },
                    high: if param.log_scale {
                        (*high as f64).ln()
                    } else {
                        *high as f64
                    },
                    log_scale: param.log_scale,
                    is_int: true,
                },
                super::ParameterType::Bool => ParamInfo {
                    name: name.clone(),
                    is_categorical: true,
                    n_choices: 2,
                    low: 0.0,
                    high: 1.0,
                    log_scale: false,
                    is_int: false,
                },
            };
            self.param_info.push(info);
        }
    }

    /// Encode a parameter value to float [0, 1]
    fn encode_param(&self, value: &super::ParameterValue, info: &ParamInfo) -> f64 {
        match value {
            super::ParameterValue::Float(f) => {
                let v = if info.log_scale { f.ln() } else { *f };
                (v - info.low) / (info.high - info.low)
            }
            super::ParameterValue::Int(i) => {
                let v = if info.log_scale {
                    (*i as f64).ln()
                } else {
                    *i as f64
                };
                (v - info.low) / (info.high - info.low)
            }
            super::ParameterValue::Categorical(s) => {
                // For categorical, return index / (n-1) to normalize to [0, 1]
                // This is a simplification; proper handling would use one-hot
                if info.n_choices <= 1 {
                    0.5
                } else {
                    // We need access to the search space to get the index
                    // For now, hash the string to get a consistent mapping
                    let hash = s.bytes().fold(0u64, |acc, b| acc.wrapping_add(b as u64));
                    (hash % info.n_choices as u64) as f64 / (info.n_choices - 1) as f64
                }
            }
            super::ParameterValue::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Decode a float [0, 1] to parameter value
    fn decode_param(
        &self,
        value: f64,
        info: &ParamInfo,
        search_space: &super::SearchSpace,
    ) -> super::ParameterValue {
        let param = search_space.parameters.get(&info.name);

        if info.is_categorical {
            if let Some(p) = param {
                match &p.param_type {
                    super::ParameterType::Categorical { choices } => {
                        let idx = ((value * choices.len() as f64).floor() as usize)
                            .min(choices.len() - 1);
                        return super::ParameterValue::Categorical(choices[idx].clone());
                    }
                    super::ParameterType::Bool => {
                        return super::ParameterValue::Bool(value >= 0.5);
                    }
                    _ => {}
                }
            }
            super::ParameterValue::Float(value)
        } else {
            // Denormalize
            let v = value
                .clamp(0.0, 1.0)
                .mul_add(info.high - info.low, info.low);

            if info.is_int {
                let int_val = if info.log_scale {
                    v.exp().round() as i64
                } else {
                    v.round() as i64
                };
                // Clamp to original bounds
                if let Some(p) = param {
                    if let super::ParameterType::Int { low, high } = &p.param_type {
                        return super::ParameterValue::Int(int_val.clamp(*low, *high));
                    }
                }
                super::ParameterValue::Int(int_val)
            } else {
                let float_val = if info.log_scale { v.exp() } else { v };
                // Clamp to original bounds
                if let Some(p) = param {
                    if let super::ParameterType::Float { low, high } = &p.param_type {
                        return super::ParameterValue::Float(float_val.clamp(*low, *high));
                    }
                }
                super::ParameterValue::Float(float_val)
            }
        }
    }

    /// Encode a trial's parameters to a vector
    fn encode_trial(&self, trial: &Trial) -> Option<Vec<f64>> {
        let mut encoded = Vec::with_capacity(self.param_info.len());

        for info in &self.param_info {
            if let Some(value) = trial.params.get(&info.name) {
                encoded.push(self.encode_param(value, info));
            } else {
                return None;
            }
        }

        Some(encoded)
    }

    /// Update KDEs with new trial results
    pub fn update(&mut self, trials: &[Trial]) {
        // Filter completed trials
        let mut completed: Vec<&Trial> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.value.is_some())
            .collect();

        if completed.len() < self.config.n_startup_trials {
            return; // Not enough data yet
        }

        // Sort by value (ascending for minimization)
        completed.sort_by(|a, b| a.value.unwrap().partial_cmp(&b.value.unwrap()).unwrap());

        // Split into good and bad
        let n_good = ((completed.len() as f64) * self.config.gamma).ceil() as usize;
        let n_good = n_good.max(1).min(completed.len() - 1);

        self.good_configs.clear();
        self.bad_configs.clear();

        for (i, trial) in completed.iter().enumerate() {
            if let Some(encoded) = self.encode_trial(trial) {
                if i < n_good {
                    self.good_configs.push(encoded);
                } else {
                    self.bad_configs.push(encoded);
                }
            }
        }

        // Build KDEs
        if !self.good_configs.is_empty() {
            self.kde_good = Some(KernelDensityEstimator::new(
                self.good_configs.clone(),
                self.config.min_bandwidth,
                self.config.max_bandwidth,
                self.config.bandwidth_factor,
            ));
        }

        if !self.bad_configs.is_empty() {
            self.kde_bad = Some(KernelDensityEstimator::new(
                self.bad_configs.clone(),
                self.config.min_bandwidth,
                self.config.max_bandwidth,
                self.config.bandwidth_factor,
            ));
        }
    }

    /// Sample a new configuration
    pub fn sample(
        &mut self,
        search_space: &super::SearchSpace,
        trials: &[Trial],
    ) -> crate::Result<HashMap<String, super::ParameterValue>> {
        use rand::SeedableRng;

        self.init_param_info(search_space);
        self.update(trials);

        // Create RNG with deterministic seed based on trial count
        let seed = self.rng_seed.wrapping_add(trials.len() as u64);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Check if we have enough data for model-based sampling
        let completed_count = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .count();

        if completed_count < self.config.n_startup_trials || self.kde_good.is_none() {
            // Fall back to random sampling
            return self.random_sample(search_space, &mut rng);
        }

        // Sample from l(x)/g(x) using importance sampling
        let mut best_sample = None;
        let mut best_ratio = f64::NEG_INFINITY;

        for _ in 0..self.config.n_ei_candidates {
            // Sample from the good KDE
            let sample = if let Some(ref kde) = self.kde_good {
                kde.sample(&mut rng)
            } else {
                continue;
            };

            // Clip to [0, 1]
            let sample: Vec<f64> = sample.iter().map(|&v| v.clamp(0.0, 1.0)).collect();

            // Compute l(x) / g(x) ratio
            let log_l = self
                .kde_good
                .as_ref()
                .map(|k| k.log_pdf(&sample))
                .unwrap_or(f64::NEG_INFINITY);
            let log_g = self
                .kde_bad
                .as_ref()
                .map(|k| k.log_pdf(&sample))
                .unwrap_or(f64::NEG_INFINITY);

            // Ratio = l(x) / g(x), we want to maximize this
            // log ratio = log_l - log_g
            let log_ratio = log_l - log_g;

            if log_ratio > best_ratio {
                best_ratio = log_ratio;
                best_sample = Some(sample);
            }
        }

        // Convert best sample to parameters
        if let Some(sample) = best_sample {
            let mut params = HashMap::new();
            for (i, info) in self.param_info.iter().enumerate() {
                let value = self.decode_param(sample[i], info, search_space);
                params.insert(info.name.clone(), value);
            }
            Ok(params)
        } else {
            // Fall back to random
            self.random_sample(search_space, &mut rng)
        }
    }

    /// Random sample from search space
    fn random_sample(
        &self,
        search_space: &super::SearchSpace,
        rng: &mut impl rand::Rng,
    ) -> crate::Result<HashMap<String, super::ParameterValue>> {
        let mut params = HashMap::new();

        for (name, param) in &search_space.parameters {
            let value = match &param.param_type {
                super::ParameterType::Float { low, high } => {
                    let v = if param.log_scale {
                        let log_low = low.ln();
                        let log_high = high.ln();
                        rng.random::<f64>()
                            .mul_add(log_high - log_low, log_low)
                            .exp()
                    } else {
                        rng.random::<f64>() * (high - low) + low
                    };
                    super::ParameterValue::Float(v)
                }
                super::ParameterType::Int { low, high } => {
                    let v = if param.log_scale {
                        let log_low = (*low as f64).ln();
                        let log_high = (*high as f64).ln();
                        rng.random::<f64>()
                            .mul_add(log_high - log_low, log_low)
                            .exp()
                            .round() as i64
                    } else {
                        rng.random_range(*low..=*high)
                    };
                    super::ParameterValue::Int(v)
                }
                super::ParameterType::Categorical { choices } => {
                    let idx = rng.random_range(0..choices.len());
                    super::ParameterValue::Categorical(choices[idx].clone())
                }
                super::ParameterType::Bool => super::ParameterValue::Bool(rng.random_bool(0.5)),
            };
            params.insert(name.clone(), value);
        }

        Ok(params)
    }
}

/// BOHB Scheduler - Bayesian Optimization and Hyperband
///
/// Combines Hyperband's multi-fidelity scheduling with model-based configuration
/// sampling using kernel density estimation.
///
/// # Algorithm
///
/// BOHB works by:
/// 1. Running multiple Hyperband brackets in parallel
/// 2. Using KDE to model the distribution of good vs bad configurations
/// 3. Sampling new configurations by maximizing l(x)/g(x) ratio
/// 4. Using multi-fidelity evaluations to efficiently prune poor configurations
///
/// # References
///
/// - Falkner, S., Klein, A., & Hutter, F. (2018). BOHB: Robust and Efficient
///   Hyperparameter Optimization at Scale. ICML.
///
/// # Example
///
/// ```
/// use ferroml_core::hpo::schedulers::{BOHBScheduler, BOHBConfig};
///
/// let scheduler = BOHBScheduler::new(BOHBConfig::new(1, 81, 3.0));
///
/// // Use with a Study
/// // let study = Study::new("my_study", search_space, Direction::Minimize)
/// //     .with_scheduler(scheduler);
/// ```
#[derive(Debug)]
pub struct BOHBScheduler {
    /// BOHB configuration
    config: BOHBConfig,
    /// Underlying Hyperband scheduler for multi-fidelity scheduling
    hyperband: HyperbandScheduler,
    /// BOHB sampler for configuration selection
    sampler: BOHBSampler,
    /// Collected observations across all brackets
    observations: Vec<BOHBObservation>,
    /// Metrics
    metrics: BOHBMetrics,
}

/// A single observation for BOHB (configuration + result at a fidelity)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BOHBObservation {
    /// Trial ID
    pub trial_id: usize,
    /// Encoded configuration (normalized \[0,1\])
    pub config: Vec<f64>,
    /// Fidelity level at which this was evaluated
    pub fidelity: f64,
    /// Objective value
    pub value: f64,
    /// Bracket ID
    pub bracket_id: usize,
}

/// BOHB-specific metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BOHBMetrics {
    /// Number of configurations sampled via KDE
    pub n_model_samples: usize,
    /// Number of random samples (startup phase)
    pub n_random_samples: usize,
    /// Best observed configuration
    pub best_config: Option<Vec<f64>>,
    /// Best observed value
    pub best_value: Option<f64>,
    /// Number of KDE updates performed
    pub n_kde_updates: usize,
    /// Current size of good configuration set
    pub n_good_configs: usize,
    /// Current size of bad configuration set
    pub n_bad_configs: usize,
}

impl Clone for BOHBScheduler {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            hyperband: self.hyperband.clone(),
            sampler: self.sampler.clone(),
            observations: self.observations.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

impl BOHBScheduler {
    /// Create a new BOHB scheduler
    pub fn new(config: BOHBConfig) -> Self {
        let hyperband = HyperbandScheduler::new(
            config.min_resource,
            config.max_resource,
            config.reduction_factor,
        );

        let sampler = BOHBSampler::new(config.clone());

        Self {
            config,
            hyperband,
            sampler,
            observations: Vec::new(),
            metrics: BOHBMetrics::default(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(min_resource: usize, max_resource: usize, reduction_factor: f64) -> Self {
        Self::new(BOHBConfig::new(
            min_resource,
            max_resource,
            reduction_factor,
        ))
    }

    /// Set the random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self.sampler = BOHBSampler::new(self.config.clone());
        self
    }

    /// Set the number of startup trials
    pub fn with_startup_trials(mut self, n: usize) -> Self {
        self.config.n_startup_trials = n;
        self.sampler = BOHBSampler::new(self.config.clone());
        self
    }

    /// Set gamma (fraction of good configurations)
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.config.gamma = gamma.clamp(0.01, 0.99);
        self.sampler = BOHBSampler::new(self.config.clone());
        self
    }

    /// Get the configuration
    pub fn config(&self) -> &BOHBConfig {
        &self.config
    }

    /// Get the underlying Hyperband scheduler
    pub fn hyperband(&self) -> &HyperbandScheduler {
        &self.hyperband
    }

    /// Get mutable access to Hyperband
    pub fn hyperband_mut(&mut self) -> &mut HyperbandScheduler {
        &mut self.hyperband
    }

    /// Get BOHB-specific metrics
    pub fn bohb_metrics(&self) -> &BOHBMetrics {
        &self.metrics
    }

    /// Get all observations
    pub fn observations(&self) -> &[BOHBObservation] {
        &self.observations
    }

    /// Sample a new configuration using BOHB
    ///
    /// This uses KDE-based sampling after the startup phase
    pub fn sample_config(
        &mut self,
        search_space: &super::SearchSpace,
        trials: &[Trial],
    ) -> crate::Result<HashMap<String, super::ParameterValue>> {
        // Update sampler with current trials
        self.sampler.init_param_info(search_space);
        self.sampler.update(trials);

        // Track whether this is a model-based sample
        let completed_count = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .count();

        let result = self.sampler.sample(search_space, trials);

        if completed_count >= self.config.n_startup_trials {
            self.metrics.n_model_samples += 1;
            self.metrics.n_kde_updates =
                self.sampler.good_configs.len() + self.sampler.bad_configs.len();
            self.metrics.n_good_configs = self.sampler.good_configs.len();
            self.metrics.n_bad_configs = self.sampler.bad_configs.len();
        } else {
            self.metrics.n_random_samples += 1;
        }

        result
    }

    /// Register an observation (configuration result at a fidelity)
    pub fn register_observation(
        &mut self,
        trial_id: usize,
        config: Vec<f64>,
        fidelity: f64,
        value: f64,
        bracket_id: usize,
    ) {
        let obs = BOHBObservation {
            trial_id,
            config: config.clone(),
            fidelity,
            value,
            bracket_id,
        };
        self.observations.push(obs);

        // Update best
        if self.metrics.best_value.is_none() || value < self.metrics.best_value.unwrap() {
            self.metrics.best_value = Some(value);
            self.metrics.best_config = Some(config);
        }

        // Also report to Hyperband
        self.hyperband
            .report_value(trial_id, fidelity as usize, value);
    }

    /// Assign a trial to a bracket
    pub fn assign_trial(&mut self, trial_id: usize) -> usize {
        self.hyperband.assign_trial(trial_id)
    }

    /// Get the resource allocation for a trial
    pub fn get_resource_for_trial(&self, trial_id: usize) -> usize {
        self.hyperband.get_resource_for_trial(trial_id)
    }

    /// Check if all brackets are complete
    pub fn is_complete(&self) -> bool {
        self.hyperband.is_complete()
    }

    /// Get summary
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("BOHB Summary:\n"));
        s.push_str(&format!(
            "  Config: eta={}, R_min={}, R_max={}\n",
            self.config.reduction_factor, self.config.min_resource, self.config.max_resource
        ));
        s.push_str(&format!(
            "  Sampling: gamma={:.2}, n_startup={}\n",
            self.config.gamma, self.config.n_startup_trials
        ));
        s.push_str(&format!(
            "  Samples: {} model-based, {} random\n",
            self.metrics.n_model_samples, self.metrics.n_random_samples
        ));
        s.push_str(&format!(
            "  KDE: {} good, {} bad configs\n",
            self.metrics.n_good_configs, self.metrics.n_bad_configs
        ));
        if let Some(best) = self.metrics.best_value {
            s.push_str(&format!("  Best value: {:.6}\n", best));
        }
        s.push_str(&self.hyperband.summary());
        s
    }
}

impl Scheduler for BOHBScheduler {
    fn should_prune(&self, trials: &[Trial], trial_id: usize, step: usize) -> bool {
        // Delegate to Hyperband for pruning decisions
        self.hyperband.should_prune(trials, trial_id, step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a trial with intermediate values
    fn make_trial(id: usize, values: Vec<f64>, state: TrialState) -> Trial {
        Trial {
            id,
            params: HashMap::new(),
            value: values.last().copied(),
            state,
            intermediate_values: values,
            duration: None,
        }
    }

    #[test]
    fn test_fidelity_parameter_discrete() {
        let fidelity = FidelityParameter::discrete("epochs", 1, 100);
        assert_eq!(fidelity.name(), "epochs");
        assert_eq!(fidelity.min_value(), 1.0);
        assert_eq!(fidelity.max_value(), 100.0);

        // Test value at rung
        let value = fidelity.value_at_rung(0, 5, 3.0);
        assert!(value >= 1.0);
    }

    #[test]
    fn test_fidelity_parameter_continuous() {
        let fidelity = FidelityParameter::continuous("data_fraction", 0.1, 1.0);
        assert_eq!(fidelity.name(), "data_fraction");
        assert!((fidelity.min_value() - 0.1).abs() < 1e-10);
        assert!((fidelity.max_value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rung_metrics() {
        let mut metrics = RungMetrics::default();
        metrics.rung = 0;
        metrics.fidelity = 1.0;

        metrics.add_trial(0, 0.5);
        assert_eq!(metrics.n_trials, 1);
        assert_eq!(metrics.best_value, Some(0.5));
        assert_eq!(metrics.mean_value, Some(0.5));

        metrics.add_trial(1, 0.3);
        assert_eq!(metrics.n_trials, 2);
        assert_eq!(metrics.best_value, Some(0.3));
        assert!((metrics.mean_value.unwrap() - 0.4).abs() < 1e-10);

        metrics.add_trial(2, 0.7);
        assert_eq!(metrics.n_trials, 3);
        assert_eq!(metrics.best_value, Some(0.3));
    }

    #[test]
    fn test_bracket_creation() {
        let bracket = Bracket::new(0, 81, 1, 81, 3.0);

        assert_eq!(bracket.id, 0);
        assert_eq!(bracket.n_configs, 81);
        assert_eq!(bracket.min_resource, 1);
        assert_eq!(bracket.max_resource, 81);
        assert!(bracket.n_rungs >= 4); // log_3(81) + 1 = 5

        // Check rung resources increase geometrically
        for i in 1..bracket.rung_resources.len() {
            assert!(bracket.rung_resources[i] >= bracket.rung_resources[i - 1]);
        }
    }

    #[test]
    fn test_bracket_trial_registration() {
        let mut bracket = Bracket::new(0, 9, 1, 9, 3.0);

        // Register trials at rung 0
        bracket.register_trial(0, 0, 0.5);
        bracket.register_trial(1, 0, 0.3);
        bracket.register_trial(2, 0, 0.7);

        assert_eq!(bracket.trials_per_rung[0].len(), 3);
        assert_eq!(bracket.values_per_rung[0].get(&1), Some(&0.3));
    }

    #[test]
    fn test_bracket_promotion() {
        let mut bracket = Bracket::new(0, 9, 1, 9, 3.0);

        // Register trials at rung 0
        for i in 0..9 {
            bracket.register_trial(i, 0, i as f64 * 0.1); // 0.0 to 0.8
        }

        // Check which trials should be promoted (top 1/3)
        assert!(bracket.should_promote(0, 0)); // Best
        assert!(bracket.should_promote(1, 0));
        assert!(bracket.should_promote(2, 0));
        assert!(!bracket.should_promote(8, 0)); // Worst
    }

    #[test]
    fn test_bracket_best_trial() {
        let mut bracket = Bracket::new(0, 3, 1, 9, 3.0);

        bracket.register_trial(0, 0, 0.5);
        bracket.register_trial(1, 0, 0.3);
        bracket.register_trial(2, 0, 0.7);

        let (best_id, best_val) = bracket.best_trial().unwrap();
        assert_eq!(best_id, 1);
        assert!((best_val - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_hyperband_scheduler_default() {
        let scheduler = HyperbandScheduler::default();

        assert_eq!(scheduler.config.min_resource, 1);
        assert_eq!(scheduler.config.max_resource, 81);
        assert!((scheduler.config.reduction_factor - 3.0).abs() < 1e-10);

        // Should have multiple brackets
        assert!(scheduler.brackets.len() >= 4);
    }

    #[test]
    fn test_hyperband_scheduler_brackets() {
        let scheduler = HyperbandScheduler::new(1, 81, 3.0);
        let brackets = scheduler.brackets();

        // Standard Hyperband with eta=3, R=81 should have 5 brackets
        assert_eq!(brackets.len(), 5);

        // Bracket 0 should have the most configs and lowest initial resource
        assert!(brackets[0].n_configs >= brackets[4].n_configs);
        assert!(brackets[0].min_resource <= brackets[4].min_resource);
    }

    #[test]
    fn test_hyperband_assign_trial() {
        let mut scheduler = HyperbandScheduler::new(1, 27, 3.0);

        let bracket_0 = scheduler.assign_trial(0);
        assert_eq!(bracket_0, 0);

        // Metrics should be updated
        assert_eq!(scheduler.metrics().total_trials, 1);
    }

    #[test]
    fn test_hyperband_report_value() {
        let mut scheduler = HyperbandScheduler::new(1, 27, 3.0);

        scheduler.assign_trial(0);
        scheduler.report_value(0, 1, 0.5);

        // Best value should be tracked
        assert_eq!(scheduler.metrics().best_value, Some(0.5));
        assert_eq!(scheduler.metrics().best_trial_id, Some(0));

        // Report a better value for another trial
        scheduler.assign_trial(1);
        scheduler.report_value(1, 1, 0.3);

        assert_eq!(scheduler.metrics().best_value, Some(0.3));
        assert_eq!(scheduler.metrics().best_trial_id, Some(1));
    }

    #[test]
    fn test_hyperband_fidelity() {
        let scheduler = HyperbandScheduler::new(1, 81, 3.0)
            .with_fidelity(FidelityParameter::discrete("epochs", 1, 81));

        assert!(scheduler.config.fidelity.is_some());

        let fidelity_1 = scheduler.fidelity_at_step(1);
        let fidelity_81 = scheduler.fidelity_at_step(81);

        assert!((fidelity_1 - 1.0).abs() < 1e-10);
        assert!((fidelity_81 - 81.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperband_continuous_fidelity() {
        let scheduler = HyperbandScheduler::new(1, 100, 3.0)
            .with_fidelity(FidelityParameter::continuous("data_fraction", 0.1, 1.0));

        let fidelity_0 = scheduler.fidelity_at_step(0);
        let fidelity_50 = scheduler.fidelity_at_step(50);
        let fidelity_100 = scheduler.fidelity_at_step(100);

        assert!((fidelity_0 - 0.1).abs() < 1e-10);
        assert!((fidelity_50 - 0.55).abs() < 1e-10);
        assert!((fidelity_100 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperband_n_rungs() {
        let scheduler = HyperbandScheduler::new(1, 81, 3.0);

        // log_3(81) + 1 = 5 rungs
        assert_eq!(scheduler.n_rungs(), 5);
    }

    #[test]
    fn test_hyperband_rung_resources() {
        let scheduler = HyperbandScheduler::new(1, 81, 3.0);

        let resources = scheduler.rung_resources(0).unwrap();

        // First bracket should have rungs at 1, 3, 9, 27, 81
        assert!(resources.contains(&1));
        assert!(resources.contains(&81));
    }

    #[test]
    fn test_hyperband_summary() {
        let scheduler = HyperbandScheduler::new(1, 27, 3.0);
        let summary = scheduler.summary();

        assert!(summary.contains("Hyperband Summary"));
        assert!(summary.contains("eta=3"));
        assert!(summary.contains("Brackets"));
    }

    #[test]
    fn test_hyperband_metrics() {
        let metrics = HyperbandMetrics {
            total_trials: 100,
            total_pruned: 75,
            total_completed: 25,
            best_value: Some(0.1),
            best_trial_id: Some(42),
            total_cost: 1000.0,
            ..Default::default()
        };

        assert!((metrics.pruning_rate() - 0.75).abs() < 1e-10);
        assert!(metrics.efficiency().is_some());
    }

    #[test]
    fn test_hyperband_from_config() {
        let config = HyperbandConfig {
            min_resource: 1,
            max_resource: 27,
            reduction_factor: 3.0,
            fidelity: Some(FidelityParameter::discrete("epochs", 1, 27)),
        };

        let scheduler = HyperbandScheduler::from_config(config);

        assert_eq!(scheduler.config().max_resource, 27);
        assert!(scheduler.config().fidelity.is_some());
    }

    #[test]
    fn test_hyperband_should_prune_at_rung_boundary() {
        let mut scheduler = HyperbandScheduler::new(1, 9, 3.0);

        // Assign trials to bracket 0
        for i in 0..9 {
            scheduler.assign_trial(i);
        }

        // Create trials with intermediate values at step 1 (first rung)
        // Pad with zeros to position values at the correct indices
        let mut trials: Vec<Trial> = (0..9)
            .map(|i| {
                let mut values = vec![0.0; 2]; // padding for steps 0, 1
                values[1] = i as f64 * 0.1; // Value at step 1
                make_trial(i, values, TrialState::Running)
            })
            .collect();

        // Add values at step 3 (second rung) for some trials
        // These trials reach step 3 with their values
        for i in 0..5 {
            trials[i].intermediate_values.push(0.0); // step 2
            trials[i].intermediate_values.push(i as f64 * 0.1); // step 3
        }

        // Trial 0 (best at step 3 = 0.0) should not be pruned
        assert!(!scheduler.should_prune(&trials, 0, 3));

        // Trial 4 (worst with value at step 3 = 0.4) - with 5 trials, keep ceil(5/3)=2
        // Sorted values: [0.0, 0.1, 0.2, 0.3, 0.4], threshold at index 2 = 0.2
        // Trial 4 (0.4) > threshold, so should be pruned
        assert!(scheduler.should_prune(&trials, 4, 3));
    }

    #[test]
    fn test_median_pruner_basic() {
        let pruner = MedianPruner::new()
            .with_startup_trials(2)
            .with_warmup_steps(0);

        // Create completed trials
        let trials = vec![
            make_trial(0, vec![0.5, 0.4, 0.3], TrialState::Complete),
            make_trial(1, vec![0.6, 0.5, 0.4], TrialState::Complete),
            make_trial(2, vec![0.8, 0.7], TrialState::Running), // Current trial
        ];

        // At step 1, trial 2 (0.7) is worse than median (0.45), should prune
        assert!(pruner.should_prune(&trials, 2, 1));
    }

    #[test]
    fn test_median_pruner_warmup() {
        let pruner = MedianPruner::new()
            .with_startup_trials(2)
            .with_warmup_steps(3);

        let trials = vec![
            make_trial(0, vec![0.5, 0.4, 0.3], TrialState::Complete),
            make_trial(1, vec![0.6, 0.5, 0.4], TrialState::Complete),
            make_trial(2, vec![0.9, 0.9], TrialState::Running),
        ];

        // Should not prune during warmup
        assert!(!pruner.should_prune(&trials, 2, 1));
    }

    #[test]
    fn test_median_pruner_percentile() {
        let pruner = MedianPruner::new()
            .with_startup_trials(3)
            .with_percentile(25.0); // Stricter - prune if worse than 25th percentile

        let trials = vec![
            make_trial(0, vec![0.1], TrialState::Complete),
            make_trial(1, vec![0.2], TrialState::Complete),
            make_trial(2, vec![0.3], TrialState::Complete),
            make_trial(3, vec![0.25], TrialState::Running), // Between 25th and 50th
        ];

        // Should prune since 0.25 > 25th percentile (~0.1)
        assert!(pruner.should_prune(&trials, 3, 0));
    }

    #[test]
    fn test_asha_scheduler_basic() {
        let scheduler = ASHAScheduler::new(1, 4.0, 1);

        // Create trials with values at the correct step indices
        // For step 1 pruning, we need values at index 1
        let trials = vec![
            make_trial(0, vec![0.0, 0.5], TrialState::Running), // value at step 1 = 0.5
            make_trial(1, vec![0.0, 0.3], TrialState::Running), // value at step 1 = 0.3
            make_trial(2, vec![0.0, 0.7], TrialState::Running), // value at step 1 = 0.7
            make_trial(3, vec![0.0, 0.9], TrialState::Running), // value at step 1 = 0.9
        ];

        // At step 1, keep top 1/4
        // Sorted: [0.3, 0.5, 0.7, 0.9], keep ceil(4/4) = 1 -> threshold = sorted[1] = 0.5
        // Trial 1 (0.3 < 0.5) should not be pruned
        assert!(!scheduler.should_prune(&trials, 1, 1));

        // Trial 3 (0.9 > 0.5) should be pruned
        assert!(scheduler.should_prune(&trials, 3, 1));
    }

    #[test]
    fn test_asha_grace_period() {
        let scheduler = ASHAScheduler::new(1, 4.0, 5);

        let trials = vec![
            make_trial(0, vec![0.9, 0.9], TrialState::Running),
            make_trial(1, vec![0.1, 0.1], TrialState::Running),
        ];

        // Should not prune during grace period
        assert!(!scheduler.should_prune(&trials, 0, 1));
    }

    #[test]
    fn test_hyperband_complete_workflow() {
        let mut scheduler = HyperbandScheduler::new(1, 9, 3.0);

        // Simulate a simple HPO workflow
        // Assign 9 trials to bracket 0
        for i in 0..9 {
            scheduler.assign_trial(i);
        }

        // Report values at rung 0 (resource = 1)
        for i in 0..9 {
            scheduler.report_value(i, 1, i as f64 * 0.1);
        }

        // Check metrics
        assert_eq!(scheduler.metrics().total_trials, 9);
        assert!(scheduler.metrics().best_value.is_some());
        assert_eq!(scheduler.metrics().best_value, Some(0.0));
        assert_eq!(scheduler.metrics().best_trial_id, Some(0));
    }

    #[test]
    fn test_bracket_promotion_workflow() {
        let mut bracket = Bracket::new(0, 9, 1, 9, 3.0);

        // Register 9 trials at rung 0
        for i in 0..9 {
            bracket.register_trial(i, 0, i as f64 * 0.1);
        }

        // Promote to next rung
        let promoted = bracket.promote_to_next_rung();

        // Should promote top 3 (ceil(9/3) = 3)
        assert_eq!(promoted.len(), 3);
        assert!(promoted.contains(&0)); // Best
        assert!(promoted.contains(&1));
        assert!(promoted.contains(&2));
        assert!(!promoted.contains(&8)); // Worst
    }

    #[test]
    fn test_hyperband_small_resource_range() {
        // Test with small resource range
        let scheduler = HyperbandScheduler::new(1, 3, 3.0);

        // Should still create valid brackets
        assert!(!scheduler.brackets.is_empty());
        assert!(scheduler.n_rungs() >= 1);
    }

    #[test]
    fn test_early_stopping_callback_trait() {
        // Test that the trait is object-safe
        struct TestCallback;
        impl EarlyStoppingCallback for TestCallback {
            fn on_trial_pruned(&self, _trial_id: usize, _step: usize, _value: f64) {}
            fn on_rung_completed(&self, _trial_id: usize, _rung: usize, _value: f64) {}
            fn on_bracket_completed(
                &self,
                _bracket_id: usize,
                _best_trial_id: usize,
                _best_value: f64,
            ) {
            }
        }

        let callback: Box<dyn EarlyStoppingCallback> = Box::new(TestCallback);
        callback.on_trial_pruned(0, 1, 0.5);
    }

    // =====================
    // BOHB Tests
    // =====================

    #[test]
    fn test_bohb_config_default() {
        let config = BOHBConfig::default();

        assert_eq!(config.min_resource, 1);
        assert_eq!(config.max_resource, 81);
        assert!((config.reduction_factor - 3.0).abs() < 1e-10);
        assert_eq!(config.n_startup_trials, 10);
        assert!((config.gamma - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_bohb_config_new() {
        let config = BOHBConfig::new(1, 27, 3.0);

        assert_eq!(config.min_resource, 1);
        assert_eq!(config.max_resource, 27);
        assert!((config.reduction_factor - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_kde_bandwidth_computation() {
        // Create synthetic data
        let data = vec![
            vec![0.1, 0.2],
            vec![0.2, 0.3],
            vec![0.3, 0.4],
            vec![0.4, 0.5],
            vec![0.5, 0.6],
        ];

        let kde = KernelDensityEstimator::new(data.clone(), 0.001, 1.0, 3.0);

        // Bandwidths should be computed
        assert_eq!(kde.bandwidths.len(), 2);
        assert!(kde.bandwidths[0] > 0.0);
        assert!(kde.bandwidths[1] > 0.0);
    }

    #[test]
    fn test_kde_log_pdf() {
        let data = vec![vec![0.5], vec![0.5], vec![0.5]];

        let kde = KernelDensityEstimator::new(data, 0.1, 1.0, 3.0);

        // PDF at the center should be higher than at edges
        let log_pdf_center = kde.log_pdf(&[0.5]);
        let log_pdf_edge = kde.log_pdf(&[0.0]);

        assert!(log_pdf_center > log_pdf_edge);
    }

    #[test]
    fn test_kde_sample() {
        use rand::SeedableRng;

        let data = vec![vec![0.3, 0.3], vec![0.5, 0.5], vec![0.7, 0.7]];

        let kde = KernelDensityEstimator::new(data, 0.01, 1.0, 3.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Sample should return a point
        let sample = kde.sample(&mut rng);
        assert_eq!(sample.len(), 2);
    }

    #[test]
    fn test_bohb_sampler_random_startup() {
        use super::super::SearchSpace;

        let search_space = SearchSpace::new().float("x", 0.0, 1.0).int("y", 1, 10);

        let mut sampler = BOHBSampler::new(BOHBConfig {
            n_startup_trials: 5,
            ..Default::default()
        });

        // With no trials, should do random sampling
        let trials: Vec<Trial> = vec![];
        let params = sampler.sample(&search_space, &trials).unwrap();

        assert!(params.contains_key("x"));
        assert!(params.contains_key("y"));
    }

    #[test]
    fn test_bohb_sampler_model_based() {
        use super::super::SearchSpace;

        let search_space = SearchSpace::new().float("x", 0.0, 1.0);

        let config = BOHBConfig {
            n_startup_trials: 3,
            gamma: 0.33,
            random_state: Some(42),
            ..Default::default()
        };
        let mut sampler = BOHBSampler::new(config);

        // Create completed trials with values
        let trials: Vec<Trial> = (0..10)
            .map(|i| {
                let mut trial = Trial {
                    id: i,
                    params: HashMap::new(),
                    value: Some(i as f64 * 0.1), // 0.0 to 0.9
                    state: TrialState::Complete,
                    intermediate_values: vec![],
                    duration: None,
                };
                trial.params.insert(
                    "x".to_string(),
                    super::super::ParameterValue::Float(i as f64 * 0.1),
                );
                trial
            })
            .collect();

        // Now sample should use the model
        sampler.init_param_info(&search_space);
        sampler.update(&trials);

        // Should have good and bad configs
        assert!(!sampler.good_configs.is_empty());
        assert!(!sampler.bad_configs.is_empty());

        // Sample a new config
        let params = sampler.sample(&search_space, &trials).unwrap();
        assert!(params.contains_key("x"));

        // The sampled value should be biased towards good configs (lower values)
        if let super::super::ParameterValue::Float(x) = &params["x"] {
            // Just check it's within bounds
            assert!(*x >= 0.0 && *x <= 1.0);
        }
    }

    #[test]
    fn test_bohb_scheduler_creation() {
        let scheduler = BOHBScheduler::new(BOHBConfig::new(1, 27, 3.0));

        assert_eq!(scheduler.config().min_resource, 1);
        assert_eq!(scheduler.config().max_resource, 27);
        assert!(!scheduler.hyperband().brackets().is_empty());
    }

    #[test]
    fn test_bohb_scheduler_with_builders() {
        let scheduler = BOHBScheduler::with_defaults(1, 81, 3.0)
            .with_seed(42)
            .with_startup_trials(5)
            .with_gamma(0.25);

        assert_eq!(scheduler.config().n_startup_trials, 5);
        assert!((scheduler.config().gamma - 0.25).abs() < 1e-10);
        assert_eq!(scheduler.config().random_state, Some(42));
    }

    #[test]
    fn test_bohb_scheduler_assign_trial() {
        let mut scheduler = BOHBScheduler::new(BOHBConfig::new(1, 27, 3.0));

        let bracket_id = scheduler.assign_trial(0);
        assert!(bracket_id < scheduler.hyperband().brackets().len());

        let resource = scheduler.get_resource_for_trial(0);
        assert!(resource >= scheduler.config().min_resource);
    }

    #[test]
    fn test_bohb_scheduler_register_observation() {
        let mut scheduler = BOHBScheduler::new(BOHBConfig::new(1, 27, 3.0));

        scheduler.assign_trial(0);
        scheduler.register_observation(0, vec![0.5], 1.0, 0.3, 0);

        assert_eq!(scheduler.observations().len(), 1);
        assert_eq!(scheduler.bohb_metrics().best_value, Some(0.3));
    }

    #[test]
    fn test_bohb_scheduler_metrics() {
        let mut scheduler = BOHBScheduler::new(BOHBConfig::new(1, 27, 3.0));

        // Register multiple observations
        scheduler.register_observation(0, vec![0.5], 1.0, 0.5, 0);
        scheduler.register_observation(1, vec![0.3], 1.0, 0.3, 0);
        scheduler.register_observation(2, vec![0.7], 1.0, 0.7, 0);

        // Best should be 0.3
        assert_eq!(scheduler.bohb_metrics().best_value, Some(0.3));
        assert_eq!(scheduler.bohb_metrics().best_config, Some(vec![0.3]));
    }

    #[test]
    fn test_bohb_scheduler_sample_config() {
        use super::super::SearchSpace;

        let search_space = SearchSpace::new()
            .float_log("lr", 0.001, 1.0)
            .int("n_layers", 1, 5);

        let mut scheduler = BOHBScheduler::new(BOHBConfig {
            n_startup_trials: 3,
            random_state: Some(42),
            ..BOHBConfig::new(1, 27, 3.0)
        });

        // Create some completed trials
        let trials: Vec<Trial> = (0..5)
            .map(|i| {
                let mut trial = Trial {
                    id: i,
                    params: HashMap::new(),
                    value: Some(i as f64 * 0.2),
                    state: TrialState::Complete,
                    intermediate_values: vec![],
                    duration: None,
                };
                trial.params.insert(
                    "lr".to_string(),
                    super::super::ParameterValue::Float(0.01 * (i + 1) as f64),
                );
                trial.params.insert(
                    "n_layers".to_string(),
                    super::super::ParameterValue::Int(i as i64 + 1),
                );
                trial
            })
            .collect();

        // Sample config
        let params = scheduler.sample_config(&search_space, &trials).unwrap();

        assert!(params.contains_key("lr"));
        assert!(params.contains_key("n_layers"));

        // Verify lr is in valid range
        if let super::super::ParameterValue::Float(lr) = &params["lr"] {
            assert!(*lr >= 0.001 && *lr <= 1.0);
        }

        // Verify n_layers is in valid range
        if let super::super::ParameterValue::Int(n) = &params["n_layers"] {
            assert!(*n >= 1 && *n <= 5);
        }
    }

    #[test]
    fn test_bohb_scheduler_should_prune() {
        let mut scheduler = BOHBScheduler::new(BOHBConfig::new(1, 9, 3.0));

        // Assign trials
        for i in 0..9 {
            scheduler.assign_trial(i);
        }

        // Create trials with intermediate values
        let trials: Vec<Trial> = (0..9)
            .map(|i| {
                let mut values = vec![0.0; 2];
                values[1] = i as f64 * 0.1;
                make_trial(i, values, TrialState::Running)
            })
            .collect();

        // Best trial should not be pruned
        assert!(!scheduler.should_prune(&trials, 0, 1));
    }

    #[test]
    fn test_bohb_scheduler_summary() {
        let scheduler = BOHBScheduler::new(BOHBConfig::new(1, 27, 3.0));
        let summary = scheduler.summary();

        assert!(summary.contains("BOHB Summary"));
        assert!(summary.contains("eta=3"));
        assert!(summary.contains("Sampling"));
    }

    #[test]
    fn test_bohb_complete_workflow() {
        use super::super::SearchSpace;

        let search_space = SearchSpace::new().float("x", 0.0, 1.0);

        let mut scheduler = BOHBScheduler::new(BOHBConfig {
            n_startup_trials: 5,
            random_state: Some(123),
            ..BOHBConfig::new(1, 9, 3.0)
        });

        let mut trials: Vec<Trial> = Vec::new();

        // Simulate HPO workflow
        for i in 0..15 {
            // Assign trial
            let bracket_id = scheduler.assign_trial(i);

            // Sample config
            let params = scheduler.sample_config(&search_space, &trials).unwrap();

            // Create trial
            let x = match &params["x"] {
                super::super::ParameterValue::Float(v) => *v,
                _ => 0.5,
            };

            // Objective: minimize (x - 0.3)^2
            let value = (x - 0.3).powi(2);

            let trial = Trial {
                id: i,
                params,
                value: Some(value),
                state: TrialState::Complete,
                intermediate_values: vec![value],
                duration: None,
            };

            // Register observation
            scheduler.register_observation(i, vec![x], 1.0, value, bracket_id);

            trials.push(trial);
        }

        // After model-based sampling kicks in, we should have some model samples
        let metrics = scheduler.bohb_metrics();
        assert!(metrics.n_random_samples >= 5); // At least startup trials
        assert!(metrics.best_value.is_some());

        // Best should be close to 0 (x near 0.3)
        assert!(metrics.best_value.unwrap() < 0.1);
    }

    #[test]
    fn test_log_add_exp() {
        // Test log-sum-exp function
        let a = 1.0_f64.ln(); // ln(1) = 0
        let b = 1.0_f64.ln(); // ln(1) = 0

        // log(exp(0) + exp(0)) = log(1 + 1) = log(2)
        let result = log_add_exp(a, b);
        assert!((result - 2.0_f64.ln()).abs() < 1e-10);

        // Test with NEG_INFINITY
        let result2 = log_add_exp(f64::NEG_INFINITY, a);
        assert!((result2 - a).abs() < 1e-10);
    }

    #[test]
    fn test_sample_standard_normal() {
        use rand::SeedableRng;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Generate many samples and check distribution
        let samples: Vec<f64> = (0..1000)
            .map(|_| sample_standard_normal(&mut rng))
            .collect();

        // Mean should be close to 0
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.1);

        // Std should be close to 1
        let variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_bohb_with_categorical() {
        use super::super::SearchSpace;

        let search_space = SearchSpace::new().float_log("lr", 0.001, 0.1).categorical(
            "optimizer",
            vec!["sgd".to_string(), "adam".to_string(), "rmsprop".to_string()],
        );

        let mut scheduler = BOHBScheduler::new(BOHBConfig {
            n_startup_trials: 3,
            random_state: Some(42),
            ..Default::default()
        });

        // Create some trials
        let trials: Vec<Trial> = vec![];

        // Sample should work with categorical
        let params = scheduler.sample_config(&search_space, &trials).unwrap();

        assert!(params.contains_key("lr"));
        assert!(params.contains_key("optimizer"));

        if let super::super::ParameterValue::Categorical(opt) = &params["optimizer"] {
            assert!(["sgd", "adam", "rmsprop"].contains(&opt.as_str()));
        }
    }
}

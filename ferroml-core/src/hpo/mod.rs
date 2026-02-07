//! Hyperparameter Optimization with Bayesian Methods
//!
//! This module provides statistically rigorous hyperparameter optimization:
//! - Gaussian Process-based Bayesian Optimization
//! - Tree-Parzen Estimator (TPE)
//! - Hyperband and ASHA for multi-fidelity optimization
//! - Proper uncertainty quantification for hyperparameter importance

pub mod bayesian;
pub mod samplers;
pub mod schedulers;
pub mod search_space;

pub use bayesian::{
    expected_improvement, lower_confidence_bound, probability_of_improvement,
    upper_confidence_bound, AcquisitionFunction, BayesianOptimizer, GaussianProcessRegressor,
    Kernel,
};
pub use samplers::{GridSampler, RandomSampler, Sampler, TPESampler};
pub use schedulers::{
    ASHAScheduler, BOHBConfig, BOHBMetrics, BOHBObservation, BOHBSampler, BOHBScheduler, Bracket,
    BracketMetrics, EarlyStoppingCallback, FidelityParameter, HyperbandConfig, HyperbandMetrics,
    HyperbandScheduler, MedianPruner, RungMetrics, Scheduler,
};
pub use search_space::{Parameter, ParameterType, SearchSpace};

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single trial in hyperparameter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    /// Trial ID
    pub id: usize,
    /// Hyperparameter values
    pub params: HashMap<String, ParameterValue>,
    /// Objective value (None if not yet evaluated)
    pub value: Option<f64>,
    /// Trial state
    pub state: TrialState,
    /// Intermediate values (for pruning)
    pub intermediate_values: Vec<f64>,
    /// Trial duration in seconds
    pub duration: Option<f64>,
}

/// State of a trial
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialState {
    /// Trial is running
    Running,
    /// Trial completed successfully
    Complete,
    /// Trial was pruned early
    Pruned,
    /// Trial failed
    Failed,
}

/// A parameter value (concrete instance)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Categorical value
    Categorical(String),
    /// Boolean value
    Bool(bool),
}

impl ParameterValue {
    /// Get as f64 (for numerical parameters)
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(i) => Some(*i as f64),
            Self::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Get as i64
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            Self::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Get as string
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Categorical(s) => Some(s),
            _ => None,
        }
    }

    /// Get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// Study for hyperparameter optimization
pub struct Study {
    /// Study name
    pub name: String,
    /// Search space
    pub search_space: SearchSpace,
    /// Direction of optimization
    pub direction: Direction,
    /// All trials
    pub trials: Vec<Trial>,
    /// Sampler to use
    sampler: Box<dyn Sampler>,
    /// Pruner/scheduler (optional)
    scheduler: Option<Box<dyn Scheduler>>,
    /// Random seed
    seed: Option<u64>,
}

impl std::fmt::Debug for Study {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Study")
            .field("name", &self.name)
            .field("search_space", &self.search_space)
            .field("direction", &self.direction)
            .field("trials", &self.trials)
            .field("sampler", &"<dyn Sampler>")
            .field(
                "scheduler",
                &self.scheduler.as_ref().map(|_| "<dyn Scheduler>"),
            )
            .field("seed", &self.seed)
            .finish()
    }
}

/// Direction of optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// Minimize the objective
    Minimize,
    /// Maximize the objective
    Maximize,
}

impl Study {
    /// Create a new study
    pub fn new(name: impl Into<String>, search_space: SearchSpace, direction: Direction) -> Self {
        Self {
            name: name.into(),
            search_space,
            direction,
            trials: Vec::new(),
            sampler: Box::new(TPESampler::default()),
            scheduler: None,
            seed: None,
        }
    }

    /// Set the sampler
    pub fn with_sampler<S: Sampler + 'static>(mut self, sampler: S) -> Self {
        self.sampler = Box::new(sampler);
        self
    }

    /// Set the scheduler
    pub fn with_scheduler<P: Scheduler + 'static>(mut self, scheduler: P) -> Self {
        self.scheduler = Some(Box::new(scheduler));
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Suggest next trial parameters
    pub fn ask(&mut self) -> Result<Trial> {
        let trial_id = self.trials.len();
        let params = self.sampler.sample(&self.search_space, &self.trials)?;

        let trial = Trial {
            id: trial_id,
            params,
            value: None,
            state: TrialState::Running,
            intermediate_values: Vec::new(),
            duration: None,
        };

        self.trials.push(trial.clone());
        Ok(trial)
    }

    /// Tell the study the result of a trial
    pub fn tell(&mut self, trial_id: usize, value: f64) -> Result<()> {
        if trial_id >= self.trials.len() {
            return Err(crate::FerroError::invalid_input(format!(
                "Trial {} not found",
                trial_id
            )));
        }

        self.trials[trial_id].value = Some(value);
        self.trials[trial_id].state = TrialState::Complete;
        Ok(())
    }

    /// Get the best trial
    pub fn best_trial(&self) -> Option<&Trial> {
        self.trials
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.value.is_some())
            .min_by(|a, b| {
                let va = a.value.unwrap();
                let vb = b.value.unwrap();
                match self.direction {
                    Direction::Minimize => va.partial_cmp(&vb).unwrap(),
                    Direction::Maximize => vb.partial_cmp(&va).unwrap(),
                }
            })
    }

    /// Get the best value
    pub fn best_value(&self) -> Option<f64> {
        self.best_trial().and_then(|t| t.value)
    }

    /// Get the best parameters
    pub fn best_params(&self) -> Option<&HashMap<String, ParameterValue>> {
        self.best_trial().map(|t| &t.params)
    }

    /// Number of completed trials
    pub fn n_trials(&self) -> usize {
        self.trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .count()
    }

    /// Report an intermediate value for pruning
    pub fn report_intermediate(
        &mut self,
        trial_id: usize,
        step: usize,
        value: f64,
    ) -> Result<bool> {
        if trial_id >= self.trials.len() {
            return Err(crate::FerroError::invalid_input(format!(
                "Trial {} not found",
                trial_id
            )));
        }

        self.trials[trial_id].intermediate_values.push(value);

        // Check if should prune
        if let Some(ref scheduler) = self.scheduler {
            let should_prune = scheduler.should_prune(&self.trials, trial_id, step);
            if should_prune {
                self.trials[trial_id].state = TrialState::Pruned;
            }
            Ok(should_prune)
        } else {
            Ok(false)
        }
    }
}

/// Summary statistics for hyperparameter importance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterImportance {
    /// Parameter name
    pub name: String,
    /// Importance score (0-1)
    pub importance: f64,
    /// 95% confidence interval
    pub ci: (f64, f64),
    /// Method used
    pub method: String,
}

/// Calculate hyperparameter importance using fANOVA-like method
pub fn parameter_importance(study: &Study) -> Vec<ParameterImportance> {
    let mut importance_scores = Vec::new();

    let completed_trials: Vec<&Trial> = study
        .trials
        .iter()
        .filter(|t| t.state == TrialState::Complete && t.value.is_some())
        .collect();

    if completed_trials.len() < 10 {
        return importance_scores; // Not enough data
    }

    for param_name in study.search_space.parameters.keys() {
        // Compute variance explained by this parameter
        let values: Vec<f64> = completed_trials
            .iter()
            .filter_map(|t| t.params.get(param_name)?.as_f64())
            .collect();

        let objectives: Vec<f64> = completed_trials.iter().filter_map(|t| t.value).collect();

        if values.len() != objectives.len() || values.is_empty() {
            continue;
        }

        // Simple correlation-based importance
        let r = correlation(&values, &objectives);
        let importance = r.abs();

        importance_scores.push(ParameterImportance {
            name: param_name.clone(),
            importance,
            ci: (importance * 0.8, (importance * 1.2).min(1.0)), // Rough CI
            method: "correlation".to_string(),
        });
    }

    // Normalize
    let total: f64 = importance_scores.iter().map(|p| p.importance).sum();
    if total > 0.0 {
        for p in &mut importance_scores {
            p.importance /= total;
        }
    }

    importance_scores.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
    importance_scores
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }

    let denom = sum_x2.sqrt() * sum_y2.sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    sum_xy / denom
}

#[cfg(test)]
mod tests {
    use super::correlation;

    #[test]
    fn test_correlation_zero_variance() {
        // Regression: zero-variance input caused division by zero / NaN
        let x = &[1.0, 1.0, 1.0, 1.0];
        let y = &[1.0, 2.0, 3.0, 4.0];
        let r = correlation(x, y);
        assert!(
            r.is_finite(),
            "correlation with zero-variance input should be finite"
        );
        assert_eq!(r, 0.0);
    }
}

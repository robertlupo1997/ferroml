//! Hyperparameter samplers

use super::{ParameterType, ParameterValue, SearchSpace, Trial};
use crate::Result;
use rand::prelude::*;
use std::collections::HashMap;

/// Trait for hyperparameter samplers
pub trait Sampler: Send + Sync {
    /// Sample new hyperparameters
    fn sample(
        &self,
        search_space: &SearchSpace,
        trials: &[Trial],
    ) -> Result<HashMap<String, ParameterValue>>;
}

/// Random sampler
#[derive(Debug, Clone)]
pub struct RandomSampler {
    seed: Option<u64>,
}

impl Default for RandomSampler {
    fn default() -> Self {
        Self { seed: None }
    }
}

impl RandomSampler {
    /// Create a new random sampler with no fixed seed
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a random sampler with a fixed seed for reproducibility
    pub fn with_seed(seed: u64) -> Self {
        Self { seed: Some(seed) }
    }
}

impl Sampler for RandomSampler {
    fn sample(
        &self,
        search_space: &SearchSpace,
        trials: &[Trial],
    ) -> Result<HashMap<String, ParameterValue>> {
        // Use trial count to vary seed, ensuring different samples each call
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed.wrapping_add(trials.len() as u64)),
            None => StdRng::from_os_rng(),
        };

        let mut params = HashMap::new();

        for (name, param) in &search_space.parameters {
            let value = match &param.param_type {
                ParameterType::Int { low, high } => {
                    let val = if param.log_scale {
                        let log_low = (*low as f64).ln();
                        let log_high = (*high as f64).ln();
                        let log_val = rng.random_range(log_low..=log_high);
                        log_val.exp() as i64
                    } else {
                        rng.random_range(*low..=*high)
                    };
                    ParameterValue::Int(val)
                }
                ParameterType::Float { low, high } => {
                    let val = if param.log_scale {
                        let log_low = low.ln();
                        let log_high = high.ln();
                        let log_val = rng.random_range(log_low..=log_high);
                        log_val.exp()
                    } else {
                        rng.random_range(*low..=*high)
                    };
                    ParameterValue::Float(val)
                }
                ParameterType::Categorical { choices } => {
                    let idx = rng.random_range(0..choices.len());
                    ParameterValue::Categorical(choices[idx].clone())
                }
                ParameterType::Bool => ParameterValue::Bool(rng.random_bool(0.5)),
            };
            params.insert(name.clone(), value);
        }

        Ok(params)
    }
}

/// Grid sampler
#[derive(Debug)]
pub struct GridSampler {
    grid_points: usize,
    current_idx: std::sync::atomic::AtomicUsize,
}

impl Clone for GridSampler {
    fn clone(&self) -> Self {
        Self {
            grid_points: self.grid_points,
            current_idx: std::sync::atomic::AtomicUsize::new(
                self.current_idx.load(std::sync::atomic::Ordering::SeqCst),
            ),
        }
    }
}

impl Default for GridSampler {
    fn default() -> Self {
        Self {
            grid_points: 10,
            current_idx: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl GridSampler {
    /// Create a new grid sampler with the specified number of grid points per dimension
    pub fn new(grid_points: usize) -> Self {
        Self {
            grid_points,
            current_idx: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl Sampler for GridSampler {
    fn sample(
        &self,
        search_space: &SearchSpace,
        _trials: &[Trial],
    ) -> Result<HashMap<String, ParameterValue>> {
        let idx = self
            .current_idx
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut params = HashMap::new();

        let _n_params = search_space.parameters.len();
        let mut remaining = idx;

        for (name, param) in &search_space.parameters {
            let grid_idx = remaining % self.grid_points;
            remaining /= self.grid_points;

            let value = match &param.param_type {
                ParameterType::Int { low, high } => {
                    let range = (high - low) as f64;
                    let val = if param.log_scale {
                        let log_low = (*low as f64).ln();
                        let log_high = (*high as f64).ln();
                        let step = (log_high - log_low) / (self.grid_points - 1) as f64;
                        (log_low + step * grid_idx as f64).exp() as i64
                    } else {
                        let step = range / (self.grid_points - 1) as f64;
                        *low + (step * grid_idx as f64) as i64
                    };
                    ParameterValue::Int(val.clamp(*low, *high))
                }
                ParameterType::Float { low, high } => {
                    let val = if param.log_scale {
                        let log_low = low.ln();
                        let log_high = high.ln();
                        // Guard against division by zero when grid_points=1
                        let step = if self.grid_points > 1 {
                            (log_high - log_low) / (self.grid_points - 1) as f64
                        } else {
                            0.0
                        };
                        (log_low + step * grid_idx as f64).exp()
                    } else {
                        // Guard against division by zero when grid_points=1
                        let step = if self.grid_points > 1 {
                            (high - low) / (self.grid_points - 1) as f64
                        } else {
                            0.0
                        };
                        low + step * grid_idx as f64
                    };
                    ParameterValue::Float(val.clamp(*low, *high))
                }
                ParameterType::Categorical { choices } => {
                    let choice_idx = grid_idx % choices.len();
                    ParameterValue::Categorical(choices[choice_idx].clone())
                }
                ParameterType::Bool => ParameterValue::Bool(grid_idx % 2 == 1),
            };
            params.insert(name.clone(), value);
        }

        Ok(params)
    }
}

/// Tree-Parzen Estimator sampler
#[derive(Debug, Clone)]
pub struct TPESampler {
    n_startup_trials: usize,
    gamma: f64,
    seed: Option<u64>,
}

impl Default for TPESampler {
    fn default() -> Self {
        Self {
            n_startup_trials: 10,
            gamma: 0.25,
            seed: None,
        }
    }
}

impl TPESampler {
    /// Create a new TPE sampler with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of random startup trials before TPE kicks in
    pub fn with_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = n;
        self
    }

    /// Set the gamma parameter (fraction of trials considered "good")
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl Sampler for TPESampler {
    fn sample(
        &self,
        search_space: &SearchSpace,
        trials: &[Trial],
    ) -> Result<HashMap<String, ParameterValue>> {
        // Use random sampling for startup
        let completed: Vec<&Trial> = trials
            .iter()
            .filter(|t| t.state == super::TrialState::Complete && t.value.is_some())
            .collect();

        if completed.len() < self.n_startup_trials {
            return RandomSampler::new().sample(search_space, trials);
        }

        // Sort by objective value
        let mut sorted_trials = completed.clone();
        sorted_trials.sort_by(|a, b| a.value.unwrap().partial_cmp(&b.value.unwrap()).unwrap());

        // Split into good and bad trials
        let n_good = ((completed.len() as f64) * self.gamma).ceil() as usize;
        let good_trials: Vec<&Trial> = sorted_trials[..n_good].to_vec();
        let _bad_trials: Vec<&Trial> = sorted_trials[n_good..].to_vec();

        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed + trials.len() as u64),
            None => StdRng::from_os_rng(),
        };

        let mut params = HashMap::new();

        // Sample from good trials (simplified TPE)
        for (name, param) in &search_space.parameters {
            let good_values: Vec<f64> = good_trials
                .iter()
                .filter_map(|t| t.params.get(name)?.as_f64())
                .collect();

            let value = if good_values.is_empty() {
                // Fall back to random
                match &param.param_type {
                    ParameterType::Int { low, high } => {
                        ParameterValue::Int(rng.random_range(*low..=*high))
                    }
                    ParameterType::Float { low, high } => {
                        ParameterValue::Float(rng.random_range(*low..=*high))
                    }
                    ParameterType::Categorical { choices } => {
                        let idx = rng.random_range(0..choices.len());
                        ParameterValue::Categorical(choices[idx].clone())
                    }
                    ParameterType::Bool => ParameterValue::Bool(rng.random_bool(0.5)),
                }
            } else {
                // Sample around good values (Gaussian kernel)
                let mean: f64 = good_values.iter().sum::<f64>() / good_values.len() as f64;
                let std: f64 = (good_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / good_values.len() as f64)
                    .sqrt()
                    .max(0.01);

                match &param.param_type {
                    ParameterType::Int { low, high } => {
                        let val = normal_sample(&mut rng, mean, std);
                        ParameterValue::Int((val as i64).clamp(*low, *high))
                    }
                    ParameterType::Float { low, high } => {
                        let val = normal_sample(&mut rng, mean, std);
                        ParameterValue::Float(val.clamp(*low, *high))
                    }
                    ParameterType::Categorical { choices } => {
                        // Use mode of good trials
                        let idx = rng.random_range(0..choices.len());
                        ParameterValue::Categorical(choices[idx].clone())
                    }
                    ParameterType::Bool => {
                        let p_true = good_values.iter().filter(|&&v| v > 0.5).count() as f64
                            / good_values.len() as f64;
                        ParameterValue::Bool(rng.random_bool(p_true))
                    }
                }
            };
            params.insert(name.clone(), value);
        }

        Ok(params)
    }
}

/// Sample from normal distribution
fn normal_sample(rng: &mut StdRng, mean: f64, std: f64) -> f64 {
    use std::f64::consts::PI;
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    mean + std * z
}

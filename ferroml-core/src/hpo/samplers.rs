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

// ---------------------------------------------------------------------------
// TPE infrastructure: 1D KDE, categorical distribution, helpers
// ---------------------------------------------------------------------------

/// One-dimensional Kernel Density Estimator (Gaussian kernel, Scott's bandwidth)
struct OneDimensionalKDE {
    data: Vec<f64>,
    bandwidth: f64,
}

impl OneDimensionalKDE {
    fn new(data: Vec<f64>) -> Self {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt().max(1e-6);
        // Scott's rule: h = 1.06 * sigma * n^(-1/5)
        let bandwidth = (1.06 * std * n.powf(-0.2)).max(1e-6);
        Self { data, bandwidth }
    }

    fn log_pdf(&self, x: f64) -> f64 {
        let n = self.data.len() as f64;
        let log_norm = -0.5 * (2.0 * std::f64::consts::PI).ln() - self.bandwidth.ln();
        let mut log_sum = f64::NEG_INFINITY;
        for &xi in &self.data {
            let diff = (x - xi) / self.bandwidth;
            let log_kernel = log_norm - 0.5 * diff * diff;
            log_sum = log_add_exp(log_sum, log_kernel);
        }
        log_sum - n.ln()
    }

    fn sample(&self, rng: &mut StdRng) -> f64 {
        let idx = rng.random_range(0..self.data.len());
        normal_sample(rng, self.data[idx], self.bandwidth)
    }
}

/// Frequency-based categorical distribution with Laplace smoothing
struct CategoricalDistribution {
    probs: Vec<(String, f64)>,
}

impl CategoricalDistribution {
    fn new(values: &[&str], all_choices: &[String]) -> Self {
        let total_with_smoothing = values.len() + all_choices.len();
        let probs = all_choices
            .iter()
            .map(|c| {
                let count = values.iter().filter(|&&v| v == c.as_str()).count() + 1;
                (c.clone(), count as f64 / total_with_smoothing as f64)
            })
            .collect();
        Self { probs }
    }

    fn log_pdf(&self, category: &str) -> f64 {
        self.probs
            .iter()
            .find(|(c, _)| c == category)
            .map(|(_, p)| p.ln())
            .unwrap_or(f64::NEG_INFINITY)
    }

    fn sample(&self, rng: &mut StdRng) -> String {
        let r: f64 = rng.random();
        let mut cumulative = 0.0;
        for (choice, prob) in &self.probs {
            cumulative += prob;
            if r < cumulative {
                return choice.clone();
            }
        }
        self.probs.last().unwrap().0.clone()
    }
}

fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else {
        let max = a.max(b);
        max + ((a - max).exp() + (b - max).exp()).ln()
    }
}

// ---------------------------------------------------------------------------
// TPE Sampler
// ---------------------------------------------------------------------------

/// Tree-Parzen Estimator sampler
///
/// Implements the TPE algorithm: builds kernel density estimates l(x) and g(x)
/// over good and bad trials respectively, then selects candidates that maximize
/// the l(x)/g(x) density ratio (equivalent to Expected Improvement).
#[derive(Debug, Clone)]
pub struct TPESampler {
    n_startup_trials: usize,
    gamma: f64,
    n_ei_candidates: usize,
    seed: Option<u64>,
}

impl Default for TPESampler {
    fn default() -> Self {
        Self {
            n_startup_trials: 10,
            gamma: 0.25,
            n_ei_candidates: 24,
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

    /// Set the number of EI candidates to evaluate per sample
    pub fn with_n_ei_candidates(mut self, n: usize) -> Self {
        self.n_ei_candidates = n;
        self
    }
}

impl Sampler for TPESampler {
    fn sample(
        &self,
        search_space: &SearchSpace,
        trials: &[Trial],
    ) -> Result<HashMap<String, ParameterValue>> {
        let completed: Vec<&Trial> = trials
            .iter()
            .filter(|t| t.state == super::TrialState::Complete && t.value.is_some())
            .collect();

        if completed.len() < self.n_startup_trials {
            return RandomSampler::new().sample(search_space, trials);
        }

        // Sort by objective value (ascending = minimize)
        let mut sorted_trials = completed.clone();
        sorted_trials.sort_by(|a, b| a.value.unwrap().partial_cmp(&b.value.unwrap()).unwrap());

        let n_good = ((completed.len() as f64) * self.gamma).ceil() as usize;
        let good_trials = &sorted_trials[..n_good];
        let bad_trials = &sorted_trials[n_good..];

        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed + trials.len() as u64),
            None => StdRng::from_os_rng(),
        };

        // Build per-dimension l(x)/g(x) models
        struct NumDim {
            name: String,
            l: OneDimensionalKDE,
            g: OneDimensionalKDE,
            log_scale: bool,
            low: f64,
            high: f64,
            is_int: bool,
        }
        struct CatDim {
            name: String,
            l: CategoricalDistribution,
            g: CategoricalDistribution,
        }
        struct BoolDim {
            name: String,
            l_p_true: f64,
            g_p_true: f64,
        }

        let mut num_dims: Vec<NumDim> = Vec::new();
        let mut cat_dims: Vec<CatDim> = Vec::new();
        let mut bool_dims: Vec<BoolDim> = Vec::new();

        for (name, param) in &search_space.parameters {
            match &param.param_type {
                ParameterType::Int { low, high } => {
                    let extract_f64 = |ts: &[&Trial]| -> Vec<f64> {
                        ts.iter()
                            .filter_map(|t| t.params.get(name)?.as_f64())
                            .map(|v| if param.log_scale { v.ln() } else { v })
                            .collect()
                    };
                    let good_vals = extract_f64(good_trials);
                    let bad_vals = extract_f64(bad_trials);
                    if !good_vals.is_empty() && !bad_vals.is_empty() {
                        num_dims.push(NumDim {
                            name: name.clone(),
                            l: OneDimensionalKDE::new(good_vals),
                            g: OneDimensionalKDE::new(bad_vals),
                            log_scale: param.log_scale,
                            low: *low as f64,
                            high: *high as f64,
                            is_int: true,
                        });
                    }
                }
                ParameterType::Float { low, high } => {
                    let extract_f64 = |ts: &[&Trial]| -> Vec<f64> {
                        ts.iter()
                            .filter_map(|t| t.params.get(name)?.as_f64())
                            .map(|v| if param.log_scale { v.ln() } else { v })
                            .collect()
                    };
                    let good_vals = extract_f64(good_trials);
                    let bad_vals = extract_f64(bad_trials);
                    if !good_vals.is_empty() && !bad_vals.is_empty() {
                        num_dims.push(NumDim {
                            name: name.clone(),
                            l: OneDimensionalKDE::new(good_vals),
                            g: OneDimensionalKDE::new(bad_vals),
                            log_scale: param.log_scale,
                            low: *low,
                            high: *high,
                            is_int: false,
                        });
                    }
                }
                ParameterType::Categorical { choices } => {
                    let good_vals: Vec<&str> = good_trials
                        .iter()
                        .filter_map(|t| t.params.get(name)?.as_str())
                        .collect();
                    let bad_vals: Vec<&str> = bad_trials
                        .iter()
                        .filter_map(|t| t.params.get(name)?.as_str())
                        .collect();
                    if !good_vals.is_empty() && !bad_vals.is_empty() {
                        cat_dims.push(CatDim {
                            name: name.clone(),
                            l: CategoricalDistribution::new(&good_vals, choices),
                            g: CategoricalDistribution::new(&bad_vals, choices),
                        });
                    }
                }
                ParameterType::Bool => {
                    let extract_bool = |ts: &[&Trial]| -> (usize, usize) {
                        let vals: Vec<bool> = ts
                            .iter()
                            .filter_map(|t| t.params.get(name)?.as_bool())
                            .collect();
                        let n_true = vals.iter().filter(|&&b| b).count();
                        (n_true, vals.len())
                    };
                    let (good_true, good_total) = extract_bool(good_trials);
                    let (bad_true, bad_total) = extract_bool(bad_trials);
                    if good_total > 0 && bad_total > 0 {
                        // Laplace-smoothed probabilities
                        bool_dims.push(BoolDim {
                            name: name.clone(),
                            l_p_true: (good_true as f64 + 1.0) / (good_total as f64 + 2.0),
                            g_p_true: (bad_true as f64 + 1.0) / (bad_total as f64 + 2.0),
                        });
                    }
                }
            }
        }

        // Track which parameters have models (for random fallback on unmodeled dims)
        let modeled: Vec<&str> = num_dims
            .iter()
            .map(|d| d.name.as_str())
            .chain(cat_dims.iter().map(|d| d.name.as_str()))
            .chain(bool_dims.iter().map(|d| d.name.as_str()))
            .collect();

        // Generate n_ei_candidates from l(x), score by log l(x) - log g(x), return best
        let mut best_score = f64::NEG_INFINITY;
        let mut best_params: HashMap<String, ParameterValue> = HashMap::new();

        for _ in 0..self.n_ei_candidates {
            let mut params = HashMap::new();
            let mut score = 0.0f64;

            for dim in &num_dims {
                let raw = dim.l.sample(&mut rng);
                let (lo, hi) = if dim.log_scale {
                    (dim.low.ln(), dim.high.ln())
                } else {
                    (dim.low, dim.high)
                };
                let clamped = raw.clamp(lo, hi);
                score += dim.l.log_pdf(clamped) - dim.g.log_pdf(clamped);
                let val = if dim.log_scale {
                    clamped.exp()
                } else {
                    clamped
                };
                if dim.is_int {
                    params.insert(
                        dim.name.clone(),
                        ParameterValue::Int(
                            (val.round() as i64).clamp(dim.low as i64, dim.high as i64),
                        ),
                    );
                } else {
                    params.insert(
                        dim.name.clone(),
                        ParameterValue::Float(val.clamp(dim.low, dim.high)),
                    );
                }
            }

            for dim in &cat_dims {
                let choice = dim.l.sample(&mut rng);
                score += dim.l.log_pdf(&choice) - dim.g.log_pdf(&choice);
                params.insert(dim.name.clone(), ParameterValue::Categorical(choice));
            }

            for dim in &bool_dims {
                let val = rng.random_bool(dim.l_p_true);
                let (p_l, p_g) = if val {
                    (dim.l_p_true, dim.g_p_true)
                } else {
                    (1.0 - dim.l_p_true, 1.0 - dim.g_p_true)
                };
                score += p_l.ln() - p_g.ln();
                params.insert(dim.name.clone(), ParameterValue::Bool(val));
            }

            // Fill unmodeled params with random values
            for (name, param) in &search_space.parameters {
                if !modeled.contains(&name.as_str()) {
                    let value = match &param.param_type {
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
                    };
                    params.insert(name.clone(), value);
                }
            }

            if score > best_score {
                best_score = score;
                best_params = params;
            }
        }

        Ok(best_params)
    }
}

/// Sample from normal distribution (Box-Muller transform)
fn normal_sample(rng: &mut StdRng, mean: f64, std: f64) -> f64 {
    use std::f64::consts::PI;
    let u1: f64 = rng.random::<f64>().max(1e-10);
    let u2: f64 = rng.random();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    std.mul_add(z, mean)
}

#[cfg(test)]
mod tests {
    use super::super::TrialState;
    use super::*;

    #[test]
    fn test_normal_sample_no_nan() {
        // Regression: Box-Muller u1=0 produced NaN via ln(0)
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..10_000 {
            let val = normal_sample(&mut rng, 0.0, 1.0);
            assert!(val.is_finite(), "normal_sample produced non-finite: {val}");
        }
    }

    #[test]
    fn test_kde_log_pdf_integrates() {
        // KDE log_pdf should produce finite values and peak near the data
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kde = OneDimensionalKDE::new(data);
        let pdf_at_3 = kde.log_pdf(3.0);
        let pdf_at_100 = kde.log_pdf(100.0);
        assert!(pdf_at_3.is_finite());
        assert!(pdf_at_100.is_finite());
        // Density should be higher near the data
        assert!(pdf_at_3 > pdf_at_100, "KDE should peak near data center");
    }

    #[test]
    fn test_categorical_distribution_laplace() {
        let choices = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let values = vec!["a", "a", "a"];
        let dist = CategoricalDistribution::new(&values, &choices);
        // "a" should have highest probability
        assert!(dist.log_pdf("a") > dist.log_pdf("b"));
        assert!(dist.log_pdf("a") > dist.log_pdf("c"));
        // Laplace smoothing: unseen categories should still have nonzero probability
        assert!(dist.log_pdf("b").is_finite());
        assert!(dist.log_pdf("c").is_finite());
    }

    #[test]
    fn test_tpe_uses_density_ratio() {
        // After startup, TPE should use l(x)/g(x) scoring, not just random
        // Create a clear signal: low x = good, high x = bad
        let space = SearchSpace::new().float("x", 0.0, 1.0);
        let sampler = TPESampler::new()
            .with_startup_trials(5)
            .with_gamma(0.25)
            .with_seed(123);

        let mut trials: Vec<Trial> = (0..20)
            .map(|i| {
                let x = i as f64 / 20.0;
                Trial {
                    id: i,
                    params: {
                        let mut h = HashMap::new();
                        h.insert("x".to_string(), ParameterValue::Float(x));
                        h
                    },
                    value: Some(x), // Lower x = better
                    state: TrialState::Complete,
                    intermediate_values: vec![],
                    duration: None,
                }
            })
            .collect();

        // Sample 50 times from TPE
        let mut samples = Vec::new();
        for _ in 0..50 {
            let params = sampler.sample(&space, &trials).unwrap();
            let x = params["x"].as_f64().unwrap();
            samples.push(x);
            trials.push(Trial {
                id: trials.len(),
                params,
                value: Some(x),
                state: TrialState::Complete,
                intermediate_values: vec![],
                duration: None,
            });
        }

        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        // l(x) is concentrated near 0, g(x) near 0.6 → samples should be biased low
        assert!(
            mean < 0.4,
            "TPE l/g ratio should bias toward good region, mean was {mean}"
        );
    }

    #[test]
    fn test_tpe_categorical_uses_frequency() {
        // Categorical params should use frequency-based sampling, not random
        let space = SearchSpace::new().categorical(
            "method",
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        );
        let sampler = TPESampler::new()
            .with_startup_trials(5)
            .with_gamma(0.25)
            .with_seed(99);

        // Good trials all use "a", bad trials use "b" and "c"
        let mut trials: Vec<Trial> = Vec::new();
        for i in 0..5 {
            trials.push(Trial {
                id: i,
                params: {
                    let mut h = HashMap::new();
                    h.insert(
                        "method".to_string(),
                        ParameterValue::Categorical("a".into()),
                    );
                    h
                },
                value: Some(i as f64 * 0.1), // Low = good
                state: TrialState::Complete,
                intermediate_values: vec![],
                duration: None,
            });
        }
        for i in 5..20 {
            let choice = if i % 2 == 0 { "b" } else { "c" };
            trials.push(Trial {
                id: i,
                params: {
                    let mut h = HashMap::new();
                    h.insert(
                        "method".to_string(),
                        ParameterValue::Categorical(choice.into()),
                    );
                    h
                },
                value: Some(1.0 + i as f64 * 0.1), // High = bad
                state: TrialState::Complete,
                intermediate_values: vec![],
                duration: None,
            });
        }

        let mut a_count = 0usize;
        for _ in 0..50 {
            let params = sampler.sample(&space, &trials).unwrap();
            if let ParameterValue::Categorical(ref s) = params["method"] {
                if s == "a" {
                    a_count += 1;
                }
            }
        }

        // "a" should be sampled more often than chance (33%)
        assert!(
            a_count > 20,
            "TPE should favor 'a' (good), but only sampled it {a_count}/50 times"
        );
    }
}

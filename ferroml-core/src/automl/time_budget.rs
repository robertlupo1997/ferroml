//! Time Budget Allocation for AutoML
//!
//! This module implements bandit-based time budget allocation for Combined
//! Algorithm Selection and Hyperparameter optimization (CASH). It uses
//! multi-armed bandit strategies to allocate computational budget proportionally
//! to algorithm promise, with early stopping for underperforming algorithms.
//!
//! # Strategies
//!
//! - **UCB1**: Upper Confidence Bound - balances exploration and exploitation
//! - **ThompsonSampling**: Bayesian approach with Beta prior for bounded scores
//! - **EpsilonGreedy**: Simple exploration with probability epsilon
//!
//! # Example
//!
//! ```ignore
//! use ferroml_core::automl::{
//!     AlgorithmPortfolio, PortfolioPreset, TimeBudgetAllocator,
//!     BanditStrategy, TimeBudgetConfig,
//! };
//!
//! // Create allocator with UCB1 strategy
//! let portfolio = AlgorithmPortfolio::for_classification(PortfolioPreset::Balanced);
//! let config = TimeBudgetConfig::new(3600) // 1 hour total
//!     .with_strategy(BanditStrategy::UCB1 { exploration_constant: 2.0 })
//!     .with_min_trials_per_algorithm(3);
//!
//! let mut allocator = TimeBudgetAllocator::new(config, &portfolio);
//!
//! // Get next algorithm to try
//! while let Some(arm) = allocator.select_arm() {
//!     let algorithm = &portfolio.algorithms[arm.algorithm_index];
//!     // ... run trial for algorithm ...
//!     allocator.update(arm.algorithm_index, score, elapsed_seconds);
//! }
//! ```

use crate::automl::portfolio::{
    AlgorithmComplexity, AlgorithmConfig, AlgorithmPortfolio, AlgorithmType,
};
use rand::Rng;
use rand_distr::{Beta, Distribution};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Configuration for time budget allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeBudgetConfig {
    /// Total time budget in seconds
    pub total_budget_seconds: u64,
    /// Bandit strategy for algorithm selection
    pub strategy: BanditStrategy,
    /// Minimum number of trials per algorithm before elimination
    pub min_trials_per_algorithm: usize,
    /// Early stopping threshold (relative to best)
    pub early_stopping_threshold: f64,
    /// Initial time allocation per trial (seconds)
    pub initial_trial_budget: f64,
    /// Maximum time per single trial (seconds)
    pub max_trial_time: f64,
    /// Warmup trials (try each algorithm at least once)
    pub warmup_trials: usize,
    /// Use complexity-weighted allocation
    pub complexity_weighted: bool,
}

impl Default for TimeBudgetConfig {
    fn default() -> Self {
        Self {
            total_budget_seconds: 3600,
            strategy: BanditStrategy::UCB1 {
                exploration_constant: 2.0,
            },
            min_trials_per_algorithm: 3,
            early_stopping_threshold: 0.1, // Stop if 10% worse than best
            initial_trial_budget: 60.0,
            max_trial_time: 300.0,
            warmup_trials: 1,
            complexity_weighted: true,
        }
    }
}

impl TimeBudgetConfig {
    /// Create a new config with the given total budget
    pub fn new(total_budget_seconds: u64) -> Self {
        Self {
            total_budget_seconds,
            ..Default::default()
        }
    }

    /// Set the bandit strategy
    pub fn with_strategy(mut self, strategy: BanditStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set minimum trials per algorithm
    pub fn with_min_trials_per_algorithm(mut self, n: usize) -> Self {
        self.min_trials_per_algorithm = n;
        self
    }

    /// Set early stopping threshold
    pub fn with_early_stopping_threshold(mut self, threshold: f64) -> Self {
        self.early_stopping_threshold = threshold;
        self
    }

    /// Set initial trial budget
    pub fn with_initial_trial_budget(mut self, seconds: f64) -> Self {
        self.initial_trial_budget = seconds;
        self
    }

    /// Set maximum trial time
    pub fn with_max_trial_time(mut self, seconds: f64) -> Self {
        self.max_trial_time = seconds;
        self
    }

    /// Set warmup trials
    pub fn with_warmup_trials(mut self, n: usize) -> Self {
        self.warmup_trials = n;
        self
    }

    /// Enable/disable complexity-weighted allocation
    pub fn with_complexity_weighted(mut self, enabled: bool) -> Self {
        self.complexity_weighted = enabled;
        self
    }
}

/// Multi-armed bandit strategy for algorithm selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BanditStrategy {
    /// Upper Confidence Bound (UCB1)
    /// Balances exploration and exploitation using confidence bounds
    UCB1 {
        /// Exploration constant (typically sqrt(2) ≈ 1.41 or 2.0)
        exploration_constant: f64,
    },

    /// Thompson Sampling with Beta prior
    /// Bayesian approach that samples from posterior distributions
    ThompsonSampling {
        /// Prior alpha (successes + 1)
        prior_alpha: f64,
        /// Prior beta (failures + 1)
        prior_beta: f64,
    },

    /// Epsilon-Greedy strategy
    /// Explores with probability epsilon, exploits otherwise
    EpsilonGreedy {
        /// Exploration probability
        epsilon: f64,
        /// Decay factor for epsilon (per trial)
        decay: f64,
    },

    /// Successive Halving
    /// Allocates equal budget, eliminates bottom half
    SuccessiveHalving {
        /// Minimum resource per configuration
        min_resource: f64,
        /// Maximum resource per configuration
        max_resource: f64,
        /// Reduction factor (typically 3)
        eta: f64,
    },
}

impl Default for BanditStrategy {
    fn default() -> Self {
        Self::UCB1 {
            exploration_constant: 2.0,
        }
    }
}

/// Statistics for a single algorithm arm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmArm {
    /// Index in the portfolio
    pub algorithm_index: usize,
    /// Algorithm type
    pub algorithm_type: AlgorithmType,
    /// Number of completed trials
    pub n_trials: usize,
    /// Sum of scores (for mean calculation)
    pub score_sum: f64,
    /// Sum of squared scores (for variance calculation)
    pub score_sum_sq: f64,
    /// Best score achieved
    pub best_score: f64,
    /// Total time spent on this algorithm
    pub total_time_seconds: f64,
    /// Whether this arm is still active (not eliminated)
    pub active: bool,
    /// Algorithm complexity
    pub complexity: AlgorithmComplexity,
    /// Success count for Thompson Sampling (normalized to [0, 1])
    pub successes: f64,
    /// Failure count for Thompson Sampling (normalized to [0, 1])
    pub failures: f64,
}

impl AlgorithmArm {
    /// Create a new arm for an algorithm
    pub fn new(index: usize, config: &AlgorithmConfig) -> Self {
        Self {
            algorithm_index: index,
            algorithm_type: config.algorithm,
            n_trials: 0,
            score_sum: 0.0,
            score_sum_sq: 0.0,
            best_score: f64::NEG_INFINITY,
            total_time_seconds: 0.0,
            active: true,
            complexity: config.complexity,
            successes: 1.0, // Prior
            failures: 1.0,  // Prior
        }
    }

    /// Update arm statistics after a trial
    pub fn update(&mut self, score: f64, elapsed_seconds: f64, is_success: bool) {
        self.n_trials += 1;
        self.score_sum += score;
        self.score_sum_sq += score * score;
        self.best_score = self.best_score.max(score);
        self.total_time_seconds += elapsed_seconds;

        // Update Thompson Sampling counts
        if is_success {
            self.successes += 1.0;
        } else {
            self.failures += 1.0;
        }
    }

    /// Get mean score
    pub fn mean_score(&self) -> f64 {
        if self.n_trials == 0 {
            0.0
        } else {
            self.score_sum / self.n_trials as f64
        }
    }

    /// Get score variance
    pub fn variance(&self) -> f64 {
        if self.n_trials < 2 {
            f64::INFINITY
        } else {
            let n = self.n_trials as f64;
            let mean = self.mean_score();
            mean.mul_add(-mean, self.score_sum_sq / n).max(0.0)
        }
    }

    /// Get standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get average time per trial
    pub fn avg_trial_time(&self) -> f64 {
        if self.n_trials == 0 {
            0.0
        } else {
            self.total_time_seconds / self.n_trials as f64
        }
    }

    /// Calculate UCB1 score
    pub fn ucb1_score(&self, total_trials: usize, exploration_constant: f64) -> f64 {
        if self.n_trials == 0 {
            return f64::INFINITY; // Untried arms have infinite UCB
        }

        let mean = self.mean_score();
        let exploration =
            exploration_constant * ((total_trials as f64).ln() / self.n_trials as f64).sqrt();

        mean + exploration
    }

    /// Sample from Thompson Sampling posterior (Beta distribution)
    pub fn thompson_sample<R: Rng>(&self, rng: &mut R, prior_alpha: f64, prior_beta: f64) -> f64 {
        let alpha = prior_alpha + self.successes - 1.0;
        let beta = prior_beta + self.failures - 1.0;

        // Ensure valid Beta parameters
        let alpha = alpha.max(0.01);
        let beta = beta.max(0.01);

        match Beta::new(alpha, beta) {
            Ok(dist) => dist.sample(rng),
            Err(_) => 0.5, // Fallback to neutral value
        }
    }
}

/// Result of arm selection
#[derive(Debug, Clone)]
pub struct ArmSelection {
    /// Index of the selected algorithm
    pub algorithm_index: usize,
    /// Algorithm type
    pub algorithm_type: AlgorithmType,
    /// Allocated time budget for this trial (seconds)
    pub trial_budget_seconds: f64,
    /// Reason for selection
    pub selection_reason: SelectionReason,
}

/// Reason why an arm was selected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionReason {
    /// Warmup phase: trying each algorithm at least once
    Warmup,
    /// Selected for exploration (high uncertainty)
    Exploration,
    /// Selected for exploitation (best expected value)
    Exploitation,
    /// Random selection (epsilon-greedy)
    Random,
}

/// Main time budget allocator
#[derive(Debug)]
pub struct TimeBudgetAllocator {
    /// Configuration
    pub config: TimeBudgetConfig,
    /// Arms (one per algorithm)
    pub arms: Vec<AlgorithmArm>,
    /// Total trials completed across all arms
    pub total_trials: usize,
    /// Total time spent (seconds)
    pub total_time_spent: f64,
    /// Start time
    start_time: Option<Instant>,
    /// Best score seen across all algorithms
    pub global_best_score: f64,
    /// Index of best performing algorithm
    pub best_algorithm_index: Option<usize>,
    /// Random number generator
    rng: rand::rngs::StdRng,
    /// Current epsilon for epsilon-greedy
    current_epsilon: f64,
    /// Current Successive Halving rung (0-indexed)
    sh_rung: usize,
    /// Trials completed at current rung per arm
    sh_trials_at_rung: Vec<usize>,
}

impl TimeBudgetAllocator {
    /// Create a new allocator for a portfolio
    pub fn new(config: TimeBudgetConfig, portfolio: &AlgorithmPortfolio) -> Self {
        Self::new_with_seed(config, portfolio, None)
    }

    /// Create a new allocator with a specific random seed
    pub fn new_with_seed(
        config: TimeBudgetConfig,
        portfolio: &AlgorithmPortfolio,
        seed: Option<u64>,
    ) -> Self {
        use rand::SeedableRng;

        let arms: Vec<AlgorithmArm> = portfolio
            .algorithms
            .iter()
            .enumerate()
            .map(|(i, c)| AlgorithmArm::new(i, c))
            .collect();

        let current_epsilon = match &config.strategy {
            BanditStrategy::EpsilonGreedy { epsilon, .. } => *epsilon,
            _ => 0.0,
        };

        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_os_rng(),
        };

        let n_arms = arms.len();
        Self {
            config,
            arms,
            total_trials: 0,
            total_time_spent: 0.0,
            start_time: None,
            global_best_score: f64::NEG_INFINITY,
            best_algorithm_index: None,
            rng,
            current_epsilon,
            sh_rung: 0,
            sh_trials_at_rung: vec![0; n_arms],
        }
    }

    /// Start the allocator (records start time)
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Check if time budget is exhausted
    pub fn is_budget_exhausted(&self) -> bool {
        self.remaining_budget_seconds() <= 0.0
    }

    /// Get remaining time budget in seconds
    pub fn remaining_budget_seconds(&self) -> f64 {
        let elapsed = self.start_time.map_or(0.0, |t| t.elapsed().as_secs_f64());
        (self.config.total_budget_seconds as f64 - elapsed).max(0.0)
    }

    /// Get number of active arms
    pub fn n_active_arms(&self) -> usize {
        self.arms.iter().filter(|a| a.active).count()
    }

    /// Select the next arm to try
    pub fn select_arm(&mut self) -> Option<ArmSelection> {
        // Check budget
        if self.is_budget_exhausted() {
            return None;
        }

        // Check if any arms are active
        if self.n_active_arms() == 0 {
            return None;
        }

        // Warmup phase: ensure each algorithm is tried at least once
        let warmup_arm = self.select_warmup_arm();
        if let Some(selection) = warmup_arm {
            return Some(selection);
        }

        // Use strategy to select arm
        let selection = match &self.config.strategy {
            BanditStrategy::UCB1 {
                exploration_constant,
            } => self.select_ucb1(*exploration_constant),
            BanditStrategy::ThompsonSampling {
                prior_alpha,
                prior_beta,
            } => self.select_thompson(*prior_alpha, *prior_beta),
            BanditStrategy::EpsilonGreedy { decay, .. } => self.select_epsilon_greedy(*decay),
            BanditStrategy::SuccessiveHalving { .. } => self.select_successive_halving(),
        };

        selection
    }

    /// Select arm during warmup phase
    fn select_warmup_arm(&self) -> Option<ArmSelection> {
        for (i, arm) in self.arms.iter().enumerate() {
            if arm.active && arm.n_trials < self.config.warmup_trials {
                let trial_budget = self.compute_trial_budget(i);
                return Some(ArmSelection {
                    algorithm_index: i,
                    algorithm_type: arm.algorithm_type,
                    trial_budget_seconds: trial_budget,
                    selection_reason: SelectionReason::Warmup,
                });
            }
        }
        None
    }

    /// Select arm using UCB1 strategy
    fn select_ucb1(&self, exploration_constant: f64) -> Option<ArmSelection> {
        let mut best_ucb = f64::NEG_INFINITY;
        let mut best_idx = None;
        let mut is_exploration = false;

        for (i, arm) in self.arms.iter().enumerate() {
            if !arm.active {
                continue;
            }

            let ucb = arm.ucb1_score(self.total_trials.max(1), exploration_constant);

            if ucb > best_ucb {
                best_ucb = ucb;
                best_idx = Some(i);
                // If UCB is infinite or very high relative to mean, it's exploration
                is_exploration = arm.n_trials == 0 || ucb > arm.mean_score() * 1.5;
            }
        }

        best_idx.map(|i| {
            let trial_budget = self.compute_trial_budget(i);
            ArmSelection {
                algorithm_index: i,
                algorithm_type: self.arms[i].algorithm_type,
                trial_budget_seconds: trial_budget,
                selection_reason: if is_exploration {
                    SelectionReason::Exploration
                } else {
                    SelectionReason::Exploitation
                },
            }
        })
    }

    /// Select arm using Thompson Sampling
    fn select_thompson(&mut self, prior_alpha: f64, prior_beta: f64) -> Option<ArmSelection> {
        let mut best_sample = f64::NEG_INFINITY;
        let mut best_idx = None;

        for (i, arm) in self.arms.iter().enumerate() {
            if !arm.active {
                continue;
            }

            let sample = arm.thompson_sample(&mut self.rng, prior_alpha, prior_beta);

            if sample > best_sample {
                best_sample = sample;
                best_idx = Some(i);
            }
        }

        best_idx.map(|i| {
            let trial_budget = self.compute_trial_budget(i);
            // Thompson sampling is inherently exploration-exploitation balanced
            let is_exploitation = self.arms[i].mean_score() >= self.global_best_score * 0.95;
            ArmSelection {
                algorithm_index: i,
                algorithm_type: self.arms[i].algorithm_type,
                trial_budget_seconds: trial_budget,
                selection_reason: if is_exploitation {
                    SelectionReason::Exploitation
                } else {
                    SelectionReason::Exploration
                },
            }
        })
    }

    /// Select arm using epsilon-greedy strategy
    fn select_epsilon_greedy(&mut self, decay: f64) -> Option<ArmSelection> {
        let active_arms: Vec<usize> = self
            .arms
            .iter()
            .enumerate()
            .filter(|(_, a)| a.active)
            .map(|(i, _)| i)
            .collect();

        if active_arms.is_empty() {
            return None;
        }

        // Decide: explore or exploit
        let explore = self.rng.random::<f64>() < self.current_epsilon;

        let selected_idx = if explore {
            // Random selection
            active_arms[self.rng.random_range(0..active_arms.len())]
        } else {
            // Greedy selection (best mean score)
            *active_arms
                .iter()
                .max_by(|&&a, &&b| {
                    self.arms[a]
                        .mean_score()
                        .partial_cmp(&self.arms[b].mean_score())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
        };

        // Decay epsilon
        self.current_epsilon *= decay;

        let trial_budget = self.compute_trial_budget(selected_idx);
        Some(ArmSelection {
            algorithm_index: selected_idx,
            algorithm_type: self.arms[selected_idx].algorithm_type,
            trial_budget_seconds: trial_budget,
            selection_reason: if explore {
                SelectionReason::Random
            } else {
                SelectionReason::Exploitation
            },
        })
    }

    /// Select arm using successive halving with proper bracket-based elimination
    fn select_successive_halving(&mut self) -> Option<ArmSelection> {
        let (min_resource, max_resource, eta) = match &self.config.strategy {
            BanditStrategy::SuccessiveHalving {
                min_resource,
                max_resource,
                eta,
            } => (*min_resource, *max_resource, *eta),
            _ => return None,
        };

        // Calculate number of rungs: s_max = floor(log_eta(max_r / min_r))
        let n_rungs = (max_resource / min_resource).log(eta).floor() as usize + 1;

        // Calculate resource for current rung: r_i = min_r * eta^i
        let current_resource = min_resource * eta.powi(self.sh_rung as i32);

        // Get active arms
        let active_arms: Vec<usize> = self
            .arms
            .iter()
            .enumerate()
            .filter(|(_, a)| a.active)
            .map(|(i, _)| i)
            .collect();

        if active_arms.is_empty() {
            return None;
        }

        // Check if all active arms have been evaluated at current rung
        let all_evaluated_at_rung = active_arms
            .iter()
            .all(|&i| self.sh_trials_at_rung[i] > self.sh_rung);

        if all_evaluated_at_rung && self.sh_rung < n_rungs - 1 {
            // Advance to next rung: eliminate bottom 1/eta configs
            let n_to_keep = (active_arms.len() as f64 / eta).ceil() as usize;
            let n_to_keep = n_to_keep.max(1); // Keep at least one

            // Sort by score (descending) and deactivate bottom performers
            let mut scored: Vec<(usize, f64)> = active_arms
                .iter()
                .map(|&i| (i, self.arms[i].mean_score()))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (i, _) in scored.iter().skip(n_to_keep) {
                self.arms[*i].active = false;
            }

            self.sh_rung += 1;
        }

        // Select arm with fewest trials at current rung among remaining active
        let selected = self
            .arms
            .iter()
            .enumerate()
            .filter(|(_, a)| a.active)
            .min_by_key(|(i, _)| self.sh_trials_at_rung[*i]);

        selected.map(|(i, arm)| {
            let trial_budget = current_resource.min(self.remaining_budget_seconds());
            ArmSelection {
                algorithm_index: i,
                algorithm_type: arm.algorithm_type,
                trial_budget_seconds: trial_budget,
                selection_reason: SelectionReason::Exploration,
            }
        })
    }

    /// Compute trial budget for an algorithm based on complexity
    fn compute_trial_budget(&self, arm_index: usize) -> f64 {
        let arm = &self.arms[arm_index];
        let remaining = self.remaining_budget_seconds();

        // Base budget
        let mut budget = self.config.initial_trial_budget;

        // Adjust based on historical average
        if arm.n_trials > 0 {
            let avg_time = arm.avg_trial_time();
            // Use exponential moving average
            budget = budget.mul_add(0.3, avg_time * 1.5 * 0.7);
        }

        // Adjust for complexity
        if self.config.complexity_weighted {
            budget *= arm.complexity.time_factor();
        }

        // Clamp to max trial time and remaining budget
        budget
            .min(self.config.max_trial_time)
            .min(remaining)
            .max(1.0)
    }

    /// Update arm statistics after a trial
    pub fn update(&mut self, algorithm_index: usize, score: f64, elapsed_seconds: f64) {
        // Determine if this is a "success" for Thompson Sampling
        // A success is when score >= global best * (1 - threshold)
        let threshold = self.config.early_stopping_threshold;
        let is_success = score >= self.global_best_score * (1.0 - threshold)
            || self.global_best_score == f64::NEG_INFINITY;

        // Update arm
        self.arms[algorithm_index].update(score, elapsed_seconds, is_success);

        // Update global stats
        self.total_trials += 1;
        self.total_time_spent += elapsed_seconds;

        // Update global best
        if score > self.global_best_score {
            self.global_best_score = score;
            self.best_algorithm_index = Some(algorithm_index);
        }

        // Perform early stopping check
        self.check_early_stopping();
    }

    /// Check and apply early stopping for underperforming algorithms
    fn check_early_stopping(&mut self) {
        let threshold = self.config.early_stopping_threshold;
        let min_trials = self.config.min_trials_per_algorithm;

        // Only apply early stopping after all arms have minimum trials
        if self
            .arms
            .iter()
            .any(|a| a.active && a.n_trials < min_trials)
        {
            return;
        }

        // Eliminate arms significantly worse than the best
        for arm in &mut self.arms {
            if !arm.active {
                continue;
            }

            // Calculate upper bound of performance (mean + 2*std)
            let upper_bound = 2.0f64.mul_add(arm.std_dev(), arm.mean_score());

            // If even optimistic estimate is worse than threshold, eliminate
            if upper_bound < self.global_best_score * (1.0 - threshold) {
                arm.active = false;
            }
        }
    }

    /// Get allocation summary
    pub fn summary(&self) -> AllocationSummary {
        let arm_summaries: Vec<ArmSummary> = self
            .arms
            .iter()
            .map(|arm| ArmSummary {
                algorithm_type: arm.algorithm_type,
                n_trials: arm.n_trials,
                mean_score: arm.mean_score(),
                best_score: arm.best_score,
                total_time_seconds: arm.total_time_seconds,
                active: arm.active,
            })
            .collect();

        AllocationSummary {
            total_budget_seconds: self.config.total_budget_seconds,
            time_spent_seconds: self.total_time_spent,
            total_trials: self.total_trials,
            n_active_arms: self.n_active_arms(),
            global_best_score: self.global_best_score,
            best_algorithm: self
                .best_algorithm_index
                .map(|i| self.arms[i].algorithm_type),
            arm_summaries,
        }
    }

    /// Reset the allocator (keeps configuration, clears statistics)
    pub fn reset(&mut self) {
        for arm in &mut self.arms {
            arm.n_trials = 0;
            arm.score_sum = 0.0;
            arm.score_sum_sq = 0.0;
            arm.best_score = f64::NEG_INFINITY;
            arm.total_time_seconds = 0.0;
            arm.active = true;
            arm.successes = 1.0;
            arm.failures = 1.0;
        }
        self.total_trials = 0;
        self.total_time_spent = 0.0;
        self.start_time = None;
        self.global_best_score = f64::NEG_INFINITY;
        self.best_algorithm_index = None;

        // Reset epsilon
        if let BanditStrategy::EpsilonGreedy { epsilon, .. } = &self.config.strategy {
            self.current_epsilon = *epsilon;
        }
    }

    /// Get elapsed time since start
    pub fn elapsed(&self) -> Duration {
        self.start_time.map_or(Duration::ZERO, |t| t.elapsed())
    }

    /// Manually deactivate an algorithm
    pub fn deactivate_algorithm(&mut self, algorithm_index: usize) {
        if algorithm_index < self.arms.len() {
            self.arms[algorithm_index].active = false;
        }
    }

    /// Reactivate an algorithm
    pub fn reactivate_algorithm(&mut self, algorithm_index: usize) {
        if algorithm_index < self.arms.len() {
            self.arms[algorithm_index].active = true;
        }
    }
}

/// Summary of time budget allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationSummary {
    /// Total budget (seconds)
    pub total_budget_seconds: u64,
    /// Time spent so far (seconds)
    pub time_spent_seconds: f64,
    /// Total trials completed
    pub total_trials: usize,
    /// Number of active arms
    pub n_active_arms: usize,
    /// Best score achieved
    pub global_best_score: f64,
    /// Best performing algorithm
    pub best_algorithm: Option<AlgorithmType>,
    /// Per-arm summaries
    pub arm_summaries: Vec<ArmSummary>,
}

/// Summary for a single algorithm arm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmSummary {
    /// Algorithm type
    pub algorithm_type: AlgorithmType,
    /// Number of trials
    pub n_trials: usize,
    /// Mean score
    pub mean_score: f64,
    /// Best score
    pub best_score: f64,
    /// Total time spent
    pub total_time_seconds: f64,
    /// Whether still active
    pub active: bool,
}

impl AllocationSummary {
    /// Get arms sorted by mean score (descending)
    pub fn arms_by_score(&self) -> Vec<&ArmSummary> {
        let mut sorted: Vec<&ArmSummary> = self.arm_summaries.iter().collect();
        sorted.sort_by(|a, b| {
            b.mean_score
                .partial_cmp(&a.mean_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Get percentage of budget used
    pub fn budget_used_percent(&self) -> f64 {
        if self.total_budget_seconds == 0 {
            return 0.0;
        }
        (self.time_spent_seconds / self.total_budget_seconds as f64 * 100.0).min(100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automl::{AlgorithmPortfolio, PortfolioPreset};

    fn create_test_portfolio() -> AlgorithmPortfolio {
        AlgorithmPortfolio::for_classification(PortfolioPreset::Quick)
    }

    #[test]
    fn test_config_builder() {
        let config = TimeBudgetConfig::new(1800)
            .with_strategy(BanditStrategy::ThompsonSampling {
                prior_alpha: 1.0,
                prior_beta: 1.0,
            })
            .with_min_trials_per_algorithm(5)
            .with_early_stopping_threshold(0.15);

        assert_eq!(config.total_budget_seconds, 1800);
        assert_eq!(config.min_trials_per_algorithm, 5);
        assert!((config.early_stopping_threshold - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_allocator_creation() {
        let portfolio = create_test_portfolio();
        let config = TimeBudgetConfig::new(3600);
        let allocator = TimeBudgetAllocator::new(config, &portfolio);

        assert_eq!(allocator.arms.len(), portfolio.algorithms.len());
        assert_eq!(allocator.total_trials, 0);
        assert!(allocator.global_best_score.is_infinite());
    }

    #[test]
    fn test_warmup_selection() {
        let portfolio = create_test_portfolio();
        let config = TimeBudgetConfig::new(3600).with_warmup_trials(1);
        let mut allocator = TimeBudgetAllocator::new_with_seed(config, &portfolio, Some(42));
        allocator.start();

        // First selections should be warmup (each algorithm once)
        let mut warmup_algorithms = Vec::new();
        for _ in 0..portfolio.algorithms.len() {
            let selection = allocator.select_arm().unwrap();
            assert_eq!(selection.selection_reason, SelectionReason::Warmup);
            warmup_algorithms.push(selection.algorithm_index);
            allocator.update(selection.algorithm_index, 0.5, 1.0);
        }

        // Each algorithm should have been selected once
        warmup_algorithms.sort();
        let expected: Vec<usize> = (0..portfolio.algorithms.len()).collect();
        assert_eq!(warmup_algorithms, expected);

        // Next selection should not be warmup
        let selection = allocator.select_arm().unwrap();
        assert_ne!(selection.selection_reason, SelectionReason::Warmup);
    }

    #[test]
    fn test_ucb1_selection() {
        let portfolio = create_test_portfolio();
        let config = TimeBudgetConfig::new(3600)
            .with_warmup_trials(0)
            .with_strategy(BanditStrategy::UCB1 {
                exploration_constant: 2.0,
            });
        let mut allocator = TimeBudgetAllocator::new_with_seed(config, &portfolio, Some(42));
        allocator.start();

        // First selection should go to unexplored arms (infinite UCB)
        let selection = allocator.select_arm().unwrap();
        assert_eq!(selection.selection_reason, SelectionReason::Exploration);
    }

    #[test]
    fn test_thompson_sampling_selection() {
        let portfolio = create_test_portfolio();
        let config = TimeBudgetConfig::new(3600)
            .with_warmup_trials(0)
            .with_strategy(BanditStrategy::ThompsonSampling {
                prior_alpha: 1.0,
                prior_beta: 1.0,
            });
        let mut allocator = TimeBudgetAllocator::new_with_seed(config, &portfolio, Some(42));
        allocator.start();

        // Should be able to select arms
        let selection = allocator.select_arm();
        assert!(selection.is_some());
    }

    #[test]
    fn test_epsilon_greedy_selection() {
        let portfolio = create_test_portfolio();
        let config = TimeBudgetConfig::new(3600)
            .with_warmup_trials(0)
            .with_strategy(BanditStrategy::EpsilonGreedy {
                epsilon: 0.5,
                decay: 0.99,
            });
        let mut allocator = TimeBudgetAllocator::new_with_seed(config, &portfolio, Some(42));
        allocator.start();

        // First, give some arms scores
        allocator.update(0, 0.8, 1.0);
        allocator.update(1, 0.6, 1.0);
        allocator.update(2, 0.4, 1.0);
        allocator.update(3, 0.2, 1.0);
        allocator.total_trials = 4;

        // Run many selections and check we get both exploitation and exploration
        let mut exploitation_count = 0;
        let mut random_count = 0;

        for _ in 0..100 {
            if let Some(selection) = allocator.select_arm() {
                match selection.selection_reason {
                    SelectionReason::Exploitation => exploitation_count += 1,
                    SelectionReason::Random => random_count += 1,
                    _ => {}
                }
                // Don't update to keep scores constant
            }
        }

        // Should have both exploitation and random selections
        assert!(exploitation_count > 0);
        assert!(random_count > 0);
    }

    #[test]
    fn test_arm_statistics() {
        let config = AlgorithmConfig::new(AlgorithmType::LogisticRegression);
        let mut arm = AlgorithmArm::new(0, &config);

        assert_eq!(arm.n_trials, 0);
        assert_eq!(arm.mean_score(), 0.0);

        arm.update(0.8, 1.0, true);
        arm.update(0.6, 2.0, true);
        arm.update(0.7, 1.5, true);

        assert_eq!(arm.n_trials, 3);
        assert!((arm.mean_score() - 0.7).abs() < 1e-10);
        assert!((arm.total_time_seconds - 4.5).abs() < 1e-10);
        assert!((arm.best_score - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_ucb1_formula() {
        let config = AlgorithmConfig::new(AlgorithmType::LogisticRegression);
        let mut arm = AlgorithmArm::new(0, &config);

        // Untried arm should have infinite UCB
        assert!(arm.ucb1_score(10, 2.0).is_infinite());

        // After updates, UCB should be finite
        arm.update(0.7, 1.0, true);
        arm.update(0.8, 1.0, true);

        let ucb = arm.ucb1_score(10, 2.0);
        assert!(ucb.is_finite());
        assert!(ucb > arm.mean_score()); // UCB should be greater than mean
    }

    #[test]
    fn test_update_and_global_best() {
        let portfolio = create_test_portfolio();
        let config = TimeBudgetConfig::new(3600);
        let mut allocator = TimeBudgetAllocator::new(config, &portfolio);
        allocator.start();

        allocator.update(0, 0.5, 1.0);
        assert!((allocator.global_best_score - 0.5).abs() < 1e-10);
        assert_eq!(allocator.best_algorithm_index, Some(0));

        allocator.update(1, 0.8, 1.0);
        assert!((allocator.global_best_score - 0.8).abs() < 1e-10);
        assert_eq!(allocator.best_algorithm_index, Some(1));

        allocator.update(0, 0.9, 1.0);
        assert!((allocator.global_best_score - 0.9).abs() < 1e-10);
        assert_eq!(allocator.best_algorithm_index, Some(0));
    }

    #[test]
    fn test_early_stopping() {
        let portfolio = create_test_portfolio();
        let config = TimeBudgetConfig::new(3600)
            .with_min_trials_per_algorithm(2)
            .with_early_stopping_threshold(0.1);
        let mut allocator = TimeBudgetAllocator::new(config, &portfolio);
        allocator.start();

        // Give all arms minimum trials
        for i in 0..allocator.arms.len() {
            allocator.update(i, 0.5, 1.0);
            allocator.update(i, 0.5, 1.0);
        }

        // Now give one arm much better scores
        allocator.update(0, 0.95, 1.0);
        allocator.update(0, 0.95, 1.0);

        // Check early stopping occurred
        // Arms with mean <= 0.5 and std ~= 0 should be deactivated
        // since 0.5 + 2*0 < 0.95 * 0.9 = 0.855
        let n_inactive = allocator.arms.iter().filter(|a| !a.active).count();
        assert!(n_inactive > 0, "Some arms should have been eliminated");
    }

    #[test]
    fn test_complexity_weighted_budget() {
        let portfolio = create_test_portfolio();
        let config = TimeBudgetConfig::new(3600).with_complexity_weighted(true);
        let allocator = TimeBudgetAllocator::new(config, &portfolio);

        // Find arms with different complexities
        let low_complexity_idx = allocator
            .arms
            .iter()
            .position(|a| a.complexity == AlgorithmComplexity::Low);
        let high_complexity_idx = allocator
            .arms
            .iter()
            .position(|a| a.complexity == AlgorithmComplexity::High);

        if let (Some(low_idx), Some(high_idx)) = (low_complexity_idx, high_complexity_idx) {
            let low_budget = allocator.compute_trial_budget(low_idx);
            let high_budget = allocator.compute_trial_budget(high_idx);

            // High complexity should get more time
            assert!(
                high_budget > low_budget,
                "High complexity algorithms should get more time"
            );
        }
    }

    #[test]
    fn test_summary() {
        let portfolio = create_test_portfolio();
        let config = TimeBudgetConfig::new(3600);
        let mut allocator = TimeBudgetAllocator::new(config, &portfolio);
        allocator.start();

        allocator.update(0, 0.8, 2.0);
        allocator.update(1, 0.6, 1.5);

        let summary = allocator.summary();

        assert_eq!(summary.total_budget_seconds, 3600);
        assert_eq!(summary.total_trials, 2);
        assert!((summary.global_best_score - 0.8).abs() < 1e-10);
        assert_eq!(summary.arm_summaries.len(), portfolio.algorithms.len());
    }

    #[test]
    fn test_reset() {
        let portfolio = create_test_portfolio();
        let config = TimeBudgetConfig::new(3600);
        let mut allocator = TimeBudgetAllocator::new(config, &portfolio);
        allocator.start();

        allocator.update(0, 0.8, 2.0);
        allocator.update(1, 0.6, 1.5);
        allocator.arms[0].active = false;

        allocator.reset();

        assert_eq!(allocator.total_trials, 0);
        assert!(allocator.global_best_score.is_infinite());
        assert!(allocator.arms[0].active);
        assert_eq!(allocator.arms[0].n_trials, 0);
    }

    #[test]
    fn test_deactivate_reactivate() {
        let portfolio = create_test_portfolio();
        let config = TimeBudgetConfig::new(3600);
        let mut allocator = TimeBudgetAllocator::new(config, &portfolio);

        assert!(allocator.arms[0].active);

        allocator.deactivate_algorithm(0);
        assert!(!allocator.arms[0].active);

        allocator.reactivate_algorithm(0);
        assert!(allocator.arms[0].active);
    }

    #[test]
    fn test_budget_exhaustion() {
        let portfolio = create_test_portfolio();
        let config = TimeBudgetConfig::new(10); // Short budget
        let mut allocator = TimeBudgetAllocator::new(config, &portfolio);

        // Before starting, not exhausted (infinite remaining)
        allocator.start();

        // Simulate time passing by updating total_time_spent
        // (In real use, this would be tracked via elapsed time)
        assert!(!allocator.is_budget_exhausted());
        assert!(allocator.remaining_budget_seconds() > 0.0);
    }

    #[test]
    fn test_successive_halving_selection() {
        let portfolio = create_test_portfolio();
        let config = TimeBudgetConfig::new(3600)
            .with_warmup_trials(0)
            .with_strategy(BanditStrategy::SuccessiveHalving {
                min_resource: 1.0,
                max_resource: 100.0,
                eta: 3.0,
            });
        let mut allocator = TimeBudgetAllocator::new_with_seed(config, &portfolio, Some(42));
        allocator.start();

        // Should select arm with fewest trials (round-robin style)
        let selection = allocator.select_arm();
        assert!(selection.is_some());
        assert_eq!(
            selection.unwrap().selection_reason,
            SelectionReason::Exploration
        );
    }

    #[test]
    fn test_arms_by_score() {
        let portfolio = create_test_portfolio();
        let config = TimeBudgetConfig::new(3600);
        let mut allocator = TimeBudgetAllocator::new(config, &portfolio);
        allocator.start();

        allocator.update(0, 0.8, 1.0);
        allocator.update(1, 0.6, 1.0);
        allocator.update(2, 0.9, 1.0);
        allocator.update(3, 0.7, 1.0);

        let summary = allocator.summary();
        let sorted = summary.arms_by_score();

        // Should be sorted descending by mean score
        assert!((sorted[0].mean_score - 0.9).abs() < 1e-10);
        assert!((sorted[1].mean_score - 0.8).abs() < 1e-10);
        assert!((sorted[2].mean_score - 0.7).abs() < 1e-10);
        assert!((sorted[3].mean_score - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_budget_used_percent() {
        let summary = AllocationSummary {
            total_budget_seconds: 100,
            time_spent_seconds: 50.0,
            total_trials: 10,
            n_active_arms: 4,
            global_best_score: 0.8,
            best_algorithm: Some(AlgorithmType::LogisticRegression),
            arm_summaries: vec![],
        };

        assert!((summary.budget_used_percent() - 50.0).abs() < 1e-10);
    }
}

# Specification: AutoML Orchestration (CASH)

**Status**: 🔲 NOT IMPLEMENTED

## Overview

Combined Algorithm Selection and Hyperparameter optimization (CASH) - the core AutoML problem.

## Requirements

### Algorithm Portfolio
- [ ] Define portfolio of candidate algorithms
- [ ] Per-algorithm search spaces
- [ ] Algorithm-specific preprocessing requirements

### CASH Solver
- [ ] Joint algorithm + hyperparameter optimization
- [ ] Time budget allocation between algorithms
- [ ] Early stopping for poorly performing algorithms

### Meta-Learning (Future)
- [ ] Dataset characterization (metafeatures)
- [ ] Warmstarting from similar datasets
- [ ] Transfer learning of HPO priors

### Ensemble Construction
- [ ] Build ensemble from evaluated configurations
- [ ] Greedy ensemble selection
- [ ] Ensemble weights optimization

### AutoML Interface

```rust
pub struct AutoML {
    pub config: AutoMLConfig,
    portfolio: Vec<Box<dyn Model>>,
    study: Study,
}

pub struct AutoMLConfig {
    pub task: Task,
    pub metric: Metric,
    pub time_budget_seconds: u64,
    pub cv_folds: usize,
    pub statistical_tests: bool,
    pub confidence_level: f64,
    pub multiple_testing_correction: MultipleTesting,
    pub seed: Option<u64>,
    pub n_jobs: usize,
    // CASH-specific
    pub algorithm_selection: AlgorithmSelection,
    pub ensemble_size: usize,
}

pub enum AlgorithmSelection {
    /// Try all algorithms equally
    Uniform,
    /// Bayesian algorithm selection
    Bayesian,
    /// Multi-armed bandit selection
    Bandit,
}
```

### AutoML Results

```rust
pub struct AutoMLResult {
    pub best_model: Box<dyn Model>,
    pub best_params: HashMap<String, ParameterValue>,
    pub best_score: f64,
    pub score_ci: (f64, f64),
    pub leaderboard: Vec<LeaderboardEntry>,
    pub ensemble: Option<Ensemble>,
    pub diagnostics: AutoMLDiagnostics,
}

pub struct LeaderboardEntry {
    pub rank: usize,
    pub algorithm: String,
    pub params: HashMap<String, ParameterValue>,
    pub score: f64,
    pub score_ci: (f64, f64),
    pub train_time: f64,
}
```

## Implementation Priority

1. Basic AutoML with uniform algorithm selection
2. Time budget allocation
3. Ensemble construction
4. Bandit-based algorithm selection

## Research References

- Auto-sklearn: Meta-learning + Bayesian HPO + ensemble
- H2O AutoML: Random search + stacking
- Auto-CASH: RL-based algorithm selection
- FLAML: Cost-aware search

## Implementation Location

`ferroml-core/src/lib.rs` (AutoML struct) + `ferroml-core/src/automl/` (new module)

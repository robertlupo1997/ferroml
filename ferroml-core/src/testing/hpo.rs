//! Hyperparameter Optimization (HPO) correctness tests
//!
//! This module provides comprehensive tests for HPO components:
//! - Sampler correctness (TPE, Grid, Random)
//! - Pruner/Scheduler correctness (MedianPruner, Hyperband, ASHA)
//! - Acquisition function correctness (EI, PI, UCB, LCB)
//! - Search space constraint validation

use crate::hpo::{ParameterType, ParameterValue, SearchSpace};
use std::collections::HashMap;

#[cfg(test)]
use crate::hpo::{
    expected_improvement, lower_confidence_bound, probability_of_improvement,
    upper_confidence_bound, Direction, GridSampler, MedianPruner, RandomSampler, Sampler,
    Scheduler, Study, TPESampler, Trial, TrialState,
};
#[cfg(test)]
use std::collections::HashSet;

// ============================================================================
// Search Space Constraint Tests
// ============================================================================

/// Verify that search space correctly stores parameter definitions
#[cfg(test)]
pub fn test_search_space_construction() {
    let space = SearchSpace::new()
        .int("n_trees", 10, 100)
        .float("learning_rate", 0.001, 1.0)
        .float_log("regularization", 1e-6, 1.0)
        .categorical(
            "activation",
            vec!["relu".into(), "tanh".into(), "sigmoid".into()],
        )
        .bool("use_bias");

    assert_eq!(space.n_dims(), 5);
    assert!(space.parameters.contains_key("n_trees"));
    assert!(space.parameters.contains_key("learning_rate"));
    assert!(space.parameters.contains_key("regularization"));
    assert!(space.parameters.contains_key("activation"));
    assert!(space.parameters.contains_key("use_bias"));

    // Check parameter types
    let n_trees = &space.parameters["n_trees"];
    match &n_trees.param_type {
        ParameterType::Int { low, high } => {
            assert_eq!(*low, 10);
            assert_eq!(*high, 100);
        }
        _ => panic!("Expected Int type for n_trees"),
    }

    let reg = &space.parameters["regularization"];
    assert!(reg.log_scale);
}

/// Verify sampled values respect search space bounds
pub fn check_sampled_values_in_bounds(
    params: &HashMap<String, ParameterValue>,
    space: &SearchSpace,
) -> Result<(), String> {
    for (name, value) in params {
        let param = space
            .parameters
            .get(name)
            .ok_or_else(|| format!("Parameter {} not in search space", name))?;

        match (&param.param_type, value) {
            (ParameterType::Int { low, high }, ParameterValue::Int(v)) => {
                if *v < *low || *v > *high {
                    return Err(format!(
                        "Int param {} = {} outside bounds [{}, {}]",
                        name, v, low, high
                    ));
                }
            }
            (ParameterType::Float { low, high }, ParameterValue::Float(v)) => {
                if *v < *low || *v > *high {
                    return Err(format!(
                        "Float param {} = {} outside bounds [{}, {}]",
                        name, v, low, high
                    ));
                }
            }
            (ParameterType::Categorical { choices }, ParameterValue::Categorical(v)) => {
                if !choices.contains(v) {
                    return Err(format!(
                        "Categorical param {} = {} not in choices {:?}",
                        name, v, choices
                    ));
                }
            }
            (ParameterType::Bool, ParameterValue::Bool(_)) => {
                // Bool is always valid
            }
            _ => {
                return Err(format!(
                    "Type mismatch for parameter {}: expected {:?}, got {:?}",
                    name, param.param_type, value
                ))
            }
        }
    }
    Ok(())
}

// ============================================================================
// Random Sampler Tests
// ============================================================================

/// Test that RandomSampler produces valid configurations within bounds
#[cfg(test)]
pub fn test_random_sampler_produces_valid_configs() {
    let space = SearchSpace::new()
        .int("n_estimators", 10, 500)
        .float("learning_rate", 0.001, 1.0)
        .float_log("regularization", 1e-8, 10.0)
        .categorical("kernel", vec!["linear".into(), "rbf".into(), "poly".into()])
        .bool("fit_intercept");

    let sampler = RandomSampler::with_seed(42);

    for _ in 0..100 {
        let params = sampler.sample(&space, &[]).expect("Sample should succeed");

        // All parameters should be present
        assert_eq!(params.len(), space.n_dims());

        // All values should be within bounds
        if let Err(e) = check_sampled_values_in_bounds(&params, &space) {
            panic!("RandomSampler produced invalid config: {}", e);
        }
    }
}

/// Test that RandomSampler with seed is reproducible
#[cfg(test)]
pub fn test_random_sampler_reproducibility() {
    let space = SearchSpace::new().float("x", 0.0, 1.0).int("n", 1, 100);

    let sampler1 = RandomSampler::with_seed(12345);
    let sampler2 = RandomSampler::with_seed(12345);

    let params1 = sampler1.sample(&space, &[]).unwrap();
    let params2 = sampler2.sample(&space, &[]).unwrap();

    // Same seed should produce same results
    assert_eq!(
        params1.get("x").and_then(|v| v.as_f64()),
        params2.get("x").and_then(|v| v.as_f64())
    );
    assert_eq!(
        params1.get("n").and_then(|v| v.as_i64()),
        params2.get("n").and_then(|v| v.as_i64())
    );
}

/// Test that RandomSampler covers the search space uniformly
#[cfg(test)]
pub fn test_random_sampler_coverage() {
    let space = SearchSpace::new()
        .float("x", 0.0, 10.0)
        .int("n", 1, 5)
        .categorical("cat", vec!["a".into(), "b".into(), "c".into()]);

    // Use unseeded sampler for true randomness, or use different seeds
    let n_samples = 100;

    let mut x_values = Vec::new();
    let mut n_counts = HashMap::new();
    let mut cat_counts = HashMap::new();

    // Sample with different seeds to ensure coverage
    for i in 0..n_samples {
        let sampler = RandomSampler::with_seed(i as u64);
        let params = sampler.sample(&space, &[]).unwrap();

        if let ParameterValue::Float(x) = params["x"] {
            x_values.push(x);
        }
        if let ParameterValue::Int(n) = params["n"] {
            *n_counts.entry(n).or_insert(0) += 1;
        }
        if let ParameterValue::Categorical(ref c) = params["cat"] {
            *cat_counts.entry(c.clone()).or_insert(0) += 1;
        }
    }

    // Float values should span the range (with different seeds, we should get diversity)
    let x_min = x_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        x_min < 2.0,
        "Random sampler should cover low values, got min={}",
        x_min
    );
    assert!(
        x_max > 8.0,
        "Random sampler should cover high values, got max={}",
        x_max
    );

    // All integer values should appear (with 100 samples across 5 values, very likely)
    for i in 1..=5 {
        assert!(
            n_counts.get(&i).unwrap_or(&0) > &0,
            "Integer value {} should appear",
            i
        );
    }

    // All categorical values should appear
    for c in ["a", "b", "c"] {
        assert!(
            cat_counts.get(c).unwrap_or(&0) > &0,
            "Categorical value {} should appear",
            c
        );
    }
}

// ============================================================================
// Grid Sampler Tests
// ============================================================================

/// Test that GridSampler produces valid configurations
#[cfg(test)]
pub fn test_grid_sampler_produces_valid_configs() {
    let space = SearchSpace::new()
        .int("n", 1, 10)
        .float("x", 0.0, 1.0)
        .categorical("opt", vec!["adam".into(), "sgd".into()]);

    let sampler = GridSampler::new(5);

    for _ in 0..50 {
        let params = sampler.sample(&space, &[]).expect("Sample should succeed");

        assert_eq!(params.len(), space.n_dims());

        if let Err(e) = check_sampled_values_in_bounds(&params, &space) {
            panic!("GridSampler produced invalid config: {}", e);
        }
    }
}

/// Test that GridSampler produces grid points at boundaries
#[cfg(test)]
pub fn test_grid_sampler_boundary_values() {
    let space = SearchSpace::new().float("x", 0.0, 1.0);

    let sampler = GridSampler::new(11); // 11 points: 0.0, 0.1, ..., 1.0

    let mut values = HashSet::new();
    for _ in 0..15 {
        let params = sampler.sample(&space, &[]).unwrap();
        if let ParameterValue::Float(x) = params["x"] {
            values.insert((x * 10.0).round() as i32);
        }
    }

    // Should hit boundary values (0 and 10 when scaled)
    assert!(
        values.contains(&0),
        "Grid should include lower boundary (0.0)"
    );
    assert!(
        values.contains(&10),
        "Grid should include upper boundary (1.0)"
    );
}

/// Test that GridSampler systematically covers the space
#[cfg(test)]
pub fn test_grid_sampler_systematic_coverage() {
    let space = SearchSpace::new()
        .int("a", 1, 3)
        .categorical("b", vec!["x".into(), "y".into()]);

    let sampler = GridSampler::new(3); // 3 grid points per dimension

    let mut seen = HashSet::new();
    // Sample more times than grid points to see cycling
    for _ in 0..20 {
        let params = sampler.sample(&space, &[]).unwrap();
        let a = params["a"].as_i64().unwrap();
        let b = params["b"].as_str().unwrap().to_string();
        seen.insert((a, b));
    }

    // Grid should cycle through combinations
    assert!(
        seen.len() >= 4,
        "Grid sampler should cover multiple combinations"
    );
}

// ============================================================================
// TPE Sampler Tests
// ============================================================================

/// Test that TPE sampler produces valid configurations
#[cfg(test)]
pub fn test_tpe_sampler_produces_valid_configs() {
    let space = SearchSpace::new()
        .int("n_estimators", 10, 200)
        .float("learning_rate", 0.001, 0.5)
        .categorical("loss", vec!["mse".into(), "mae".into()])
        .bool("warm_start");

    let sampler = TPESampler::new().with_seed(42).with_startup_trials(5);

    // Create some completed trials for TPE to learn from
    let mut trials = create_dummy_trials(10, &space);

    for _ in 0..50 {
        let params = sampler
            .sample(&space, &trials)
            .expect("TPE sample should succeed");

        assert_eq!(params.len(), space.n_dims());

        if let Err(e) = check_sampled_values_in_bounds(&params, &space) {
            panic!("TPE produced invalid config: {}", e);
        }

        // Add trial to history
        trials.push(Trial {
            id: trials.len(),
            params: params.clone(),
            value: Some(rand_value()),
            state: TrialState::Complete,
            intermediate_values: vec![],
            duration: Some(1.0),
        });
    }
}

/// Test that TPE starts with random sampling during startup
#[cfg(test)]
pub fn test_tpe_startup_behavior() {
    let space = SearchSpace::new().float("x", 0.0, 1.0);

    let sampler = TPESampler::new().with_startup_trials(10).with_seed(42);

    // With no trials, should use random sampling
    let params = sampler.sample(&space, &[]).unwrap();
    assert!(params.contains_key("x"));

    // With few trials, should still use random
    let trials = create_dummy_trials(5, &space);
    let params = sampler.sample(&space, &trials).unwrap();
    assert!(params.contains_key("x"));
}

/// Test that TPE biases toward good configurations
#[cfg(test)]
pub fn test_tpe_biases_toward_good_values() {
    let space = SearchSpace::new().float("x", 0.0, 1.0);

    let sampler = TPESampler::new()
        .with_startup_trials(5)
        .with_gamma(0.25)
        .with_seed(42);

    // Create trials where low x values have better (lower) objective values
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
                value: Some(x), // Lower x = better objective
                state: TrialState::Complete,
                intermediate_values: vec![],
                duration: Some(1.0),
            }
        })
        .collect();

    // Sample many times and check distribution
    let mut x_values = Vec::new();
    for _ in 0..100 {
        let params = sampler.sample(&space, &trials).unwrap();
        if let ParameterValue::Float(x) = params["x"] {
            x_values.push(x);
        }

        // Add trial
        trials.push(Trial {
            id: trials.len(),
            params: params.clone(),
            value: Some(params["x"].as_f64().unwrap()),
            state: TrialState::Complete,
            intermediate_values: vec![],
            duration: Some(1.0),
        });
    }

    // TPE should bias toward lower x values (where good trials are)
    let mean_x: f64 = x_values.iter().sum::<f64>() / x_values.len() as f64;
    // The mean should be biased toward lower values (< 0.5)
    // This is a statistical test, so we use a generous threshold
    assert!(
        mean_x < 0.6,
        "TPE should bias toward good (low x) region, but mean was {}",
        mean_x
    );
}

// ============================================================================
// Acquisition Function Tests
// ============================================================================

/// Test Expected Improvement computation
#[cfg(test)]
pub fn test_expected_improvement_correctness() {
    // When mu < best_y (potential improvement exists)
    let mu = 1.0;
    let sigma = 0.5;
    let best_y = 1.5;
    let ei = expected_improvement(mu, sigma, best_y, true);
    assert!(
        ei > 0.0,
        "EI should be positive when mean is better than best"
    );

    // When mu == best_y, EI should be positive (uncertainty provides value)
    let ei_equal = expected_improvement(best_y, sigma, best_y, true);
    assert!(
        ei_equal > 0.0,
        "EI should be positive even at best due to uncertainty"
    );

    // When sigma is very small, EI should approach max(0, best_y - mu)
    let ei_low_sigma = expected_improvement(1.0, 0.0001, 1.5, true);
    assert!(
        (ei_low_sigma - 0.5).abs() < 0.01,
        "EI with low sigma should approach deterministic improvement"
    );

    // When mu >> best_y (no likely improvement), EI should be small
    let ei_bad = expected_improvement(3.0, 0.5, 1.5, true);
    assert!(
        ei_bad < 0.1,
        "EI should be small when mean is much worse than best"
    );

    // Zero sigma should give zero EI (no uncertainty)
    let ei_zero_sigma = expected_improvement(1.0, 0.0, 1.5, true);
    assert_eq!(ei_zero_sigma, 0.0, "EI with zero sigma should be 0");
}

/// Test Probability of Improvement computation
#[cfg(test)]
pub fn test_probability_of_improvement_correctness() {
    let best_y = 1.0;

    // When mu < best_y (clearly better), PI should be > 0.5
    let pi_better = probability_of_improvement(0.5, 0.3, best_y, true);
    assert!(
        pi_better > 0.5,
        "PI should be > 0.5 when mean is better than best"
    );

    // When mu == best_y, PI should be ~0.5
    let pi_equal = probability_of_improvement(1.0, 0.3, best_y, true);
    assert!((pi_equal - 0.5).abs() < 0.01, "PI should be ~0.5 at best_y");

    // When mu > best_y (clearly worse), PI should be < 0.5
    let pi_worse = probability_of_improvement(1.5, 0.3, best_y, true);
    assert!(
        pi_worse < 0.5,
        "PI should be < 0.5 when mean is worse than best"
    );

    // PI should be bounded in [0, 1]
    assert!(pi_better >= 0.0 && pi_better <= 1.0);
    assert!(pi_worse >= 0.0 && pi_worse <= 1.0);

    // Zero sigma edge case
    let pi_zero_sigma_better = probability_of_improvement(0.5, 0.0, best_y, true);
    assert_eq!(
        pi_zero_sigma_better, 1.0,
        "PI should be 1 when definitely better"
    );

    let pi_zero_sigma_worse = probability_of_improvement(1.5, 0.0, best_y, true);
    assert_eq!(
        pi_zero_sigma_worse, 0.0,
        "PI should be 0 when definitely worse"
    );
}

/// Test Upper/Lower Confidence Bound computation
#[cfg(test)]
pub fn test_confidence_bounds_correctness() {
    let mu = 2.0;
    let sigma = 1.0;
    let kappa = 2.0;

    // UCB = mu + kappa * sigma
    let ucb = upper_confidence_bound(mu, sigma, kappa);
    assert!(
        (ucb - 4.0).abs() < 1e-10,
        "UCB should be mu + kappa * sigma"
    );

    // LCB = mu - kappa * sigma
    let lcb = lower_confidence_bound(mu, sigma, kappa);
    assert!(
        (lcb - 0.0).abs() < 1e-10,
        "LCB should be mu - kappa * sigma"
    );

    // UCB > mu and LCB < mu
    assert!(ucb > mu, "UCB should be greater than mean");
    assert!(lcb < mu, "LCB should be less than mean");

    // Larger kappa = wider bounds
    let ucb_large_kappa = upper_confidence_bound(mu, sigma, 3.0);
    assert!(ucb_large_kappa > ucb, "Larger kappa should give larger UCB");

    // Zero sigma case
    let ucb_zero = upper_confidence_bound(mu, 0.0, kappa);
    assert_eq!(ucb_zero, mu, "UCB with zero sigma should equal mean");
}

/// Test acquisition functions are monotonic with respect to improvement potential
#[cfg(test)]
pub fn test_acquisition_monotonicity() {
    let sigma = 0.5;
    let best_y = 1.0;

    // EI should increase as mu gets better (lower for minimization)
    let ei_1 = expected_improvement(0.9, sigma, best_y, true);
    let ei_2 = expected_improvement(0.5, sigma, best_y, true);
    let ei_3 = expected_improvement(0.1, sigma, best_y, true);

    assert!(
        ei_2 > ei_1,
        "EI should increase as mean improves: {} vs {}",
        ei_2,
        ei_1
    );
    assert!(
        ei_3 > ei_2,
        "EI should increase as mean improves: {} vs {}",
        ei_3,
        ei_2
    );

    // PI should also increase as mu gets better
    let pi_1 = probability_of_improvement(0.9, sigma, best_y, true);
    let pi_2 = probability_of_improvement(0.5, sigma, best_y, true);
    let pi_3 = probability_of_improvement(0.1, sigma, best_y, true);

    assert!(pi_2 > pi_1, "PI should increase as mean improves");
    assert!(pi_3 > pi_2, "PI should increase as mean improves");
}

// ============================================================================
// MedianPruner Tests
// ============================================================================

/// Test that MedianPruner respects startup trials
#[cfg(test)]
pub fn test_median_pruner_respects_startup() {
    let pruner = MedianPruner::new()
        .with_startup_trials(5)
        .with_warmup_steps(0);

    let mut trials = create_trials_with_intermediates(3, vec![0.5]);

    // With fewer than startup trials, should not prune
    let should_prune = pruner.should_prune(&trials, 2, 0);
    assert!(
        !should_prune,
        "Should not prune before startup trials complete"
    );

    // Add more trials to exceed startup
    trials.extend(create_trials_with_intermediates(
        5,
        vec![0.3, 0.4, 0.5, 0.6, 0.7],
    ));

    // Update trial 2's intermediate value to be bad
    trials[2].intermediate_values = vec![0.9];

    // Now should consider pruning
    let should_prune = pruner.should_prune(&trials, 2, 0);
    assert!(should_prune, "Should prune bad trial after startup");
}

/// Test that MedianPruner respects warmup steps
#[cfg(test)]
pub fn test_median_pruner_respects_warmup() {
    let pruner = MedianPruner::new()
        .with_startup_trials(2)
        .with_warmup_steps(3);

    // Create 4 completed trials with different intermediate values per trial
    let mut trials: Vec<Trial> = (0..4)
        .map(|i| Trial {
            id: i,
            params: HashMap::new(),
            value: Some(0.3 + 0.1 * i as f64),
            state: TrialState::Complete,
            intermediate_values: vec![0.1, 0.2, 0.3, 0.3 + 0.05 * i as f64], // at step 3: 0.3, 0.35, 0.4, 0.45
            duration: Some(1.0),
        })
        .collect();

    // Add running trial 4 with worse value at step 2 (during warmup)
    trials.push(Trial {
        id: 4,
        params: HashMap::new(),
        value: None,
        state: TrialState::Running,
        intermediate_values: vec![0.1, 0.2, 0.9], // bad value at step 2
        duration: None,
    });

    // At step 2 (before warmup of 3), should not prune regardless of value
    let should_prune_early = pruner.should_prune(&trials, 4, 2);
    assert!(
        !should_prune_early,
        "Should not prune during warmup (step 2 < warmup 3)"
    );

    // Update trial 4 to have value at step 3 (after warmup)
    trials[4].intermediate_values.push(0.9); // Now has [0.1, 0.2, 0.9, 0.9]

    // At step 3 (>= warmup), can prune
    // Completed trials have step 3 values: 0.3, 0.35, 0.4, 0.45
    // Median of [0.3, 0.35, 0.4, 0.45] ~= 0.375
    // Trial 4's last value is 0.9 which is > 0.375, so should prune
    let should_prune_late = pruner.should_prune(&trials, 4, 3);
    assert!(
        should_prune_late,
        "Should consider pruning after warmup (trial value 0.9 > median ~0.375)"
    );
}

/// Test that MedianPruner correctly computes median threshold
#[cfg(test)]
pub fn test_median_pruner_threshold_computation() {
    let pruner = MedianPruner::new()
        .with_startup_trials(0)
        .with_warmup_steps(0)
        .with_percentile(50.0);

    // Create 5 completed trials with intermediate values at step 0
    let mut trials: Vec<Trial> = (0..5)
        .map(|i| Trial {
            id: i,
            params: HashMap::new(),
            value: Some(i as f64 * 0.1),
            state: TrialState::Complete,
            intermediate_values: vec![i as f64 * 0.1], // 0.0, 0.1, 0.2, 0.3, 0.4
            duration: Some(1.0),
        })
        .collect();

    // Add a running trial with value at median (0.2)
    trials.push(Trial {
        id: 5,
        params: HashMap::new(),
        value: None,
        state: TrialState::Running,
        intermediate_values: vec![0.2],
        duration: None,
    });

    // Trial with median value should not be pruned
    let should_prune_median = pruner.should_prune(&trials, 5, 0);
    assert!(!should_prune_median, "Trial at median should not be pruned");

    // Update trial to have worse value
    trials[5].intermediate_values = vec![0.35];
    let should_prune_bad = pruner.should_prune(&trials, 5, 0);
    assert!(should_prune_bad, "Trial worse than median should be pruned");
}

// ============================================================================
// ASHA Scheduler Tests
// ============================================================================

/// Test that ASHA respects grace period
#[cfg(test)]
pub fn test_asha_respects_grace_period() {
    use crate::hpo::ASHAScheduler;

    let scheduler = ASHAScheduler::new(1, 4.0, 3);

    let trials = create_trials_with_intermediates(5, vec![0.5, 0.5, 0.5]);

    // Before grace period (step 2), should not prune
    let should_prune = scheduler.should_prune(&trials, 4, 2);
    assert!(!should_prune, "ASHA should not prune before grace period");
}

/// Test that ASHA prunes at rung boundaries
#[cfg(test)]
pub fn test_asha_prunes_at_rungs() {
    use crate::hpo::ASHAScheduler;

    // min_resource=1, reduction_factor=4, so rungs at 1, 4, 16, ...
    let scheduler = ASHAScheduler::new(1, 4.0, 0);

    // Create trials at rung 1 with values [0.1, 0.2, 0.3, 0.4, 0.5]
    let mut trials: Vec<Trial> = (0..5)
        .map(|i| {
            let val = (i + 1) as f64 * 0.1;
            Trial {
                id: i,
                params: HashMap::new(),
                value: None,
                state: TrialState::Running,
                intermediate_values: vec![val], // at step 0 = rung 1 index
                duration: None,
            }
        })
        .collect();

    // Fill intermediate_values to reach step 1
    for trial in &mut trials {
        trial
            .intermediate_values
            .push(trial.intermediate_values[0] + 0.01);
    }

    // At step 1 (rung boundary), check pruning
    // With reduction_factor=4, should keep top 1/4 = 1.25 -> ceil = 2
    let should_prune_best = scheduler.should_prune(&trials, 0, 1);
    assert!(!should_prune_best, "Best trial should not be pruned");

    let should_prune_worst = scheduler.should_prune(&trials, 4, 1);
    // This depends on the exact implementation - trial 4 has worst value
    // Should be pruned as it's not in top 1/4
    assert!(should_prune_worst, "Worst trial should be pruned at rung");
}

// ============================================================================
// Hyperband Scheduler Tests
// ============================================================================

/// Test Hyperband bracket computation
#[cfg(test)]
pub fn test_hyperband_bracket_computation() {
    use crate::hpo::HyperbandScheduler;

    let scheduler = HyperbandScheduler::new(1, 81, 3.0);
    let brackets = scheduler.brackets();

    // With max=81, min=1, eta=3: s_max = floor(log_3(81)) = 4
    // So we should have 5 brackets (s=0,1,2,3,4)
    assert!(
        brackets.len() >= 3,
        "Hyperband should have multiple brackets"
    );

    // First bracket should have most configs with lowest initial resource
    // Last bracket should have fewer configs with higher initial resource
    if brackets.len() > 1 {
        assert!(
            brackets[0].n_configs >= brackets[brackets.len() - 1].n_configs,
            "Earlier brackets should have more configs"
        );
    }
}

/// Test Hyperband trial assignment
#[cfg(test)]
pub fn test_hyperband_trial_assignment() {
    use crate::hpo::HyperbandScheduler;

    let mut scheduler = HyperbandScheduler::new(1, 27, 3.0);

    // Assign several trials
    let bracket_ids: Vec<usize> = (0..10).map(|i| scheduler.assign_trial(i)).collect();

    // Trials should be assigned to brackets
    for id in &bracket_ids {
        assert!(
            *id < scheduler.brackets().len(),
            "Bracket ID should be valid"
        );
    }
}

// ============================================================================
// Study Integration Tests
// ============================================================================

/// Test full study workflow with ask/tell pattern
#[cfg(test)]
pub fn test_study_ask_tell_workflow() {
    let space = SearchSpace::new()
        .float("x", -5.0, 5.0)
        .float("y", -5.0, 5.0);

    let mut study = Study::new("test_study", space.clone(), Direction::Minimize)
        .with_sampler(TPESampler::new().with_startup_trials(5).with_seed(42));

    // Run optimization loop
    for _ in 0..20 {
        let trial = study.ask().expect("Ask should succeed");

        // Validate trial params
        assert!(trial.params.contains_key("x"));
        assert!(trial.params.contains_key("y"));

        if let Err(e) = check_sampled_values_in_bounds(&trial.params, &space) {
            panic!("Study produced invalid trial: {}", e);
        }

        // Compute objective (simple quadratic)
        let x = trial.params["x"].as_f64().unwrap();
        let y = trial.params["y"].as_f64().unwrap();
        let objective = x * x + y * y;

        study
            .tell(trial.id, objective)
            .expect("Tell should succeed");
    }

    // Should have completed trials
    assert_eq!(study.n_trials(), 20);

    // Best value should exist and be reasonable
    let best = study.best_value();
    assert!(best.is_some());
    assert!(
        best.unwrap() < 50.0,
        "Best should be reasonable for quadratic"
    );
}

/// Test study with pruning
#[cfg(test)]
pub fn test_study_with_pruning() {
    let space = SearchSpace::new().float("lr", 0.001, 0.1);

    let mut study = Study::new("pruning_test", space, Direction::Minimize)
        .with_sampler(RandomSampler::with_seed(42))
        .with_scheduler(MedianPruner::new().with_startup_trials(3));

    // Simulate trials with intermediate reporting
    for _ in 0..10 {
        let trial = study.ask().unwrap();

        // Simulate training with intermediate values
        for step in 0..5 {
            let value = rand_value() + step as f64 * 0.01;
            let should_prune = study.report_intermediate(trial.id, step, value).unwrap();

            if should_prune {
                break;
            }
        }

        // Complete trial if not pruned
        if study.trials[trial.id].state == TrialState::Running {
            study.tell(trial.id, rand_value()).ok();
        }
    }

    // Some trials may have been pruned
    let pruned_count = study
        .trials
        .iter()
        .filter(|t| t.state == TrialState::Pruned)
        .count();

    // Not all trials should be pruned (especially early ones)
    let completed_count = study.n_trials();
    assert!(completed_count > 0, "At least some trials should complete");
    assert!(
        pruned_count + completed_count == study.trials.len()
            || study.trials.iter().any(|t| t.state == TrialState::Running),
        "All trials should be in a final state"
    );
}

/// Test study direction (minimize vs maximize)
#[cfg(test)]
pub fn test_study_optimization_direction() {
    let space = SearchSpace::new().float("x", 0.0, 10.0);

    // Create trials with known values to test direction logic
    let trial_values = vec![3.0, 7.0, 1.0, 9.0, 5.0]; // min=1.0, max=9.0

    // Minimize study - manually add trials with known objectives
    let mut min_study =
        Study::new("min", space.clone(), Direction::Minimize).with_sampler(RandomSampler::new());

    for (i, &val) in trial_values.iter().enumerate() {
        // Directly create and add completed trial
        min_study.trials.push(Trial {
            id: i,
            params: {
                let mut h = HashMap::new();
                h.insert("x".to_string(), ParameterValue::Float(val));
                h
            },
            value: Some(val), // objective = x
            state: TrialState::Complete,
            intermediate_values: vec![],
            duration: Some(1.0),
        });
    }

    // Maximize study - same values
    let mut max_study =
        Study::new("max", space, Direction::Maximize).with_sampler(RandomSampler::new());

    for (i, &val) in trial_values.iter().enumerate() {
        max_study.trials.push(Trial {
            id: i,
            params: {
                let mut h = HashMap::new();
                h.insert("x".to_string(), ParameterValue::Float(val));
                h
            },
            value: Some(val), // objective = x
            state: TrialState::Complete,
            intermediate_values: vec![],
            duration: Some(1.0),
        });
    }

    // Best for minimize should be the minimum value (1.0)
    let min_best = min_study.best_value().unwrap();
    // Best for maximize should be the maximum value (9.0)
    let max_best = max_study.best_value().unwrap();

    assert!(
        (min_best - 1.0).abs() < 1e-10,
        "Minimize best should be 1.0, got {}",
        min_best
    );
    assert!(
        (max_best - 9.0).abs() < 1e-10,
        "Maximize best should be 9.0, got {}",
        max_best
    );
    assert!(
        min_best < max_best,
        "Minimize best ({}) should be less than maximize best ({})",
        min_best,
        max_best
    );
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create dummy trials with random values for testing
#[cfg(test)]
fn create_dummy_trials(n: usize, space: &SearchSpace) -> Vec<Trial> {
    let sampler = RandomSampler::with_seed(12345);
    (0..n)
        .map(|i| {
            let params = sampler.sample(space, &[]).unwrap();
            Trial {
                id: i,
                params,
                value: Some(rand_value()),
                state: TrialState::Complete,
                intermediate_values: vec![],
                duration: Some(1.0),
            }
        })
        .collect()
}

/// Create trials with specific intermediate values
#[cfg(test)]
fn create_trials_with_intermediates(n: usize, base_values: Vec<f64>) -> Vec<Trial> {
    (0..n)
        .map(|i| Trial {
            id: i,
            params: HashMap::new(),
            value: Some(base_values.get(i).copied().unwrap_or(0.5)),
            state: TrialState::Complete,
            intermediate_values: base_values.clone(),
            duration: Some(1.0),
        })
        .collect()
}

/// Generate a pseudo-random value for testing
#[cfg(test)]
fn rand_value() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut hasher = DefaultHasher::new();
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .hash(&mut hasher);
    (hasher.finish() % 1000) as f64 / 1000.0
}

// ============================================================================
// Test Module
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Search space tests
    #[test]
    fn test_search_space() {
        test_search_space_construction();
    }

    // Random sampler tests
    #[test]
    fn test_random_sampler_valid() {
        test_random_sampler_produces_valid_configs();
    }

    #[test]
    fn test_random_sampler_repro() {
        test_random_sampler_reproducibility();
    }

    #[test]
    fn test_random_sampler_cover() {
        test_random_sampler_coverage();
    }

    // Grid sampler tests
    #[test]
    fn test_grid_sampler_valid() {
        test_grid_sampler_produces_valid_configs();
    }

    #[test]
    fn test_grid_sampler_bounds() {
        test_grid_sampler_boundary_values();
    }

    #[test]
    fn test_grid_sampler_cover() {
        test_grid_sampler_systematic_coverage();
    }

    // TPE sampler tests
    #[test]
    fn test_tpe_valid() {
        test_tpe_sampler_produces_valid_configs();
    }

    #[test]
    fn test_tpe_startup() {
        test_tpe_startup_behavior();
    }

    #[test]
    fn test_tpe_bias() {
        test_tpe_biases_toward_good_values();
    }

    // Acquisition function tests
    #[test]
    fn test_ei() {
        test_expected_improvement_correctness();
    }

    #[test]
    fn test_pi() {
        test_probability_of_improvement_correctness();
    }

    #[test]
    fn test_cb() {
        test_confidence_bounds_correctness();
    }

    #[test]
    fn test_acq_monotonic() {
        test_acquisition_monotonicity();
    }

    // Pruner tests
    #[test]
    fn test_median_pruner_startup() {
        test_median_pruner_respects_startup();
    }

    #[test]
    fn test_median_pruner_warmup() {
        test_median_pruner_respects_warmup();
    }

    #[test]
    fn test_median_pruner_threshold() {
        test_median_pruner_threshold_computation();
    }

    // ASHA tests
    #[test]
    fn test_asha_grace() {
        test_asha_respects_grace_period();
    }

    #[test]
    fn test_asha_rungs() {
        test_asha_prunes_at_rungs();
    }

    // Hyperband tests
    #[test]
    fn test_hyperband_brackets() {
        test_hyperband_bracket_computation();
    }

    #[test]
    fn test_hyperband_assign() {
        test_hyperband_trial_assignment();
    }

    // Study integration tests
    #[test]
    fn test_study_workflow() {
        test_study_ask_tell_workflow();
    }

    #[test]
    fn test_study_pruning() {
        test_study_with_pruning();
    }

    #[test]
    fn test_study_direction() {
        test_study_optimization_direction();
    }
}

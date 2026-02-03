//! Ensemble Construction from AutoML Trials
//!
//! This module provides greedy ensemble selection for building optimal model ensembles
//! from evaluated AutoML trial configurations. Based on the auto-sklearn approach:
//! "Efficient and Robust Automated Machine Learning" (Feurer et al., 2015).
//!
//! # Algorithm
//!
//! The greedy ensemble selection algorithm:
//! 1. Start with an empty ensemble
//! 2. Repeatedly add the model that improves ensemble performance the most
//! 3. Allow models to be selected multiple times (increases their weight)
//! 4. Use cross-validation predictions to avoid overfitting
//!
//! # Example
//!
//! ```ignore
//! use ferroml_core::automl::ensemble::{
//!     EnsembleBuilder, EnsembleConfig, TrialResult,
//! };
//!
//! // After running AutoML trials, collect results
//! let trials: Vec<TrialResult> = run_automl_trials(&x, &y);
//!
//! // Build ensemble from top trials
//! let config = EnsembleConfig::new()
//!     .with_max_models(10)
//!     .with_selection_iterations(50);
//!
//! let mut builder = EnsembleBuilder::new(config);
//! let ensemble_result = builder.build_from_trials(&trials)?;
//!
//! // Use the ensemble for predictions
//! let predictions = ensemble_result.predict(&x_test)?;
//! ```

use crate::automl::portfolio::AlgorithmType;
use crate::{FerroError, Result, Task};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result from a single AutoML trial
///
/// Contains the CV predictions and performance metrics needed for
/// ensemble selection without requiring the actual fitted model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    /// Unique identifier for this trial
    pub trial_id: usize,
    /// Algorithm type used in this trial
    pub algorithm: AlgorithmType,
    /// Hyperparameter configuration (serialized)
    pub params: HashMap<String, ParamValue>,
    /// Cross-validation score (mean across folds)
    pub cv_score: f64,
    /// Standard deviation of CV scores
    pub cv_std: f64,
    /// Individual fold scores
    pub fold_scores: Vec<f64>,
    /// Out-of-fold predictions (n_samples,) for regression or binary classification
    pub oof_predictions: Option<Array1<f64>>,
    /// Out-of-fold probability predictions (n_samples, n_classes) for classification
    pub oof_probabilities: Option<Array2<f64>>,
    /// Training time in seconds
    pub training_time_seconds: f64,
    /// Whether the trial completed successfully
    pub success: bool,
    /// Error message if the trial failed
    pub error_message: Option<String>,
}

impl TrialResult {
    /// Create a new successful trial result
    pub fn new(
        trial_id: usize,
        algorithm: AlgorithmType,
        cv_score: f64,
        cv_std: f64,
        fold_scores: Vec<f64>,
    ) -> Self {
        Self {
            trial_id,
            algorithm,
            params: HashMap::new(),
            cv_score,
            cv_std,
            fold_scores,
            oof_predictions: None,
            oof_probabilities: None,
            training_time_seconds: 0.0,
            success: true,
            error_message: None,
        }
    }

    /// Create a failed trial result
    pub fn failed(trial_id: usize, algorithm: AlgorithmType, error: impl Into<String>) -> Self {
        Self {
            trial_id,
            algorithm,
            params: HashMap::new(),
            cv_score: f64::NEG_INFINITY,
            cv_std: 0.0,
            fold_scores: Vec::new(),
            oof_predictions: None,
            oof_probabilities: None,
            training_time_seconds: 0.0,
            success: false,
            error_message: Some(error.into()),
        }
    }

    /// Set hyperparameter configuration
    pub fn with_params(mut self, params: HashMap<String, ParamValue>) -> Self {
        self.params = params;
        self
    }

    /// Set out-of-fold predictions (for regression or binary classification)
    pub fn with_oof_predictions(mut self, predictions: Array1<f64>) -> Self {
        self.oof_predictions = Some(predictions);
        self
    }

    /// Set out-of-fold probability predictions (for classification)
    pub fn with_oof_probabilities(mut self, probabilities: Array2<f64>) -> Self {
        self.oof_probabilities = Some(probabilities);
        self
    }

    /// Set training time
    pub fn with_training_time(mut self, seconds: f64) -> Self {
        self.training_time_seconds = seconds;
        self
    }

    /// Check if this trial has OOF predictions available
    pub fn has_oof_predictions(&self) -> bool {
        self.oof_predictions.is_some() || self.oof_probabilities.is_some()
    }
}

/// Hyperparameter value (type-erased)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParamValue {
    /// Integer parameter
    Int(i64),
    /// Float parameter
    Float(f64),
    /// String/categorical parameter
    String(String),
    /// Boolean parameter
    Bool(bool),
}

/// Configuration for ensemble selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Maximum number of unique models in the ensemble
    pub max_models: usize,
    /// Number of greedy selection iterations
    pub selection_iterations: usize,
    /// Minimum weight for a model to be included (as fraction of total)
    pub min_weight: f64,
    /// Whether to use model diversity in selection
    pub use_diversity: bool,
    /// Diversity weight (0 = ignore diversity, 1 = diversity only)
    pub diversity_weight: f64,
    /// Whether to optimize weights after greedy selection
    pub optimize_weights: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Task type (affects scoring)
    pub task: Task,
    /// Whether higher scores are better
    pub maximize: bool,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            max_models: 50,
            selection_iterations: 50,
            min_weight: 0.01,
            use_diversity: false,
            diversity_weight: 0.1,
            optimize_weights: true,
            random_state: None,
            task: Task::Classification,
            maximize: true,
        }
    }
}

impl EnsembleConfig {
    /// Create a new ensemble configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum number of models
    pub fn with_max_models(mut self, max_models: usize) -> Self {
        self.max_models = max_models;
        self
    }

    /// Set number of selection iterations
    pub fn with_selection_iterations(mut self, iterations: usize) -> Self {
        self.selection_iterations = iterations;
        self
    }

    /// Set minimum weight threshold
    pub fn with_min_weight(mut self, min_weight: f64) -> Self {
        self.min_weight = min_weight;
        self
    }

    /// Enable diversity-based selection
    pub fn with_diversity(mut self, weight: f64) -> Self {
        self.use_diversity = weight > 0.0;
        self.diversity_weight = weight;
        self
    }

    /// Set whether to optimize weights
    pub fn with_weight_optimization(mut self, enabled: bool) -> Self {
        self.optimize_weights = enabled;
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set task type
    pub fn with_task(mut self, task: Task) -> Self {
        self.task = task;
        self
    }

    /// Set optimization direction (true = maximize, false = minimize)
    pub fn with_maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }
}

/// A selected model in the ensemble with its weight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleMember {
    /// Trial ID of the original trial
    pub trial_id: usize,
    /// Algorithm type
    pub algorithm: AlgorithmType,
    /// Weight in the ensemble (normalized to sum to 1)
    pub weight: f64,
    /// Number of times this model was selected (before normalization)
    pub selection_count: usize,
    /// Hyperparameter configuration
    pub params: HashMap<String, ParamValue>,
}

/// Result of ensemble construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleResult {
    /// Selected ensemble members with weights
    pub members: Vec<EnsembleMember>,
    /// Ensemble validation score
    pub ensemble_score: f64,
    /// Best single model score (for comparison)
    pub best_single_score: f64,
    /// Improvement over best single model
    pub improvement: f64,
    /// Number of selection iterations performed
    pub iterations_performed: usize,
    /// History of ensemble scores during selection
    pub score_history: Vec<f64>,
    /// Task type
    pub task: Task,
}

impl EnsembleResult {
    /// Get the number of unique models in the ensemble
    pub fn n_models(&self) -> usize {
        self.members.len()
    }

    /// Get total weight (should be 1.0)
    pub fn total_weight(&self) -> f64 {
        self.members.iter().map(|m| m.weight).sum()
    }

    /// Get models sorted by weight (descending)
    pub fn members_by_weight(&self) -> Vec<&EnsembleMember> {
        let mut sorted: Vec<_> = self.members.iter().collect();
        sorted.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Get algorithm distribution in the ensemble
    pub fn algorithm_distribution(&self) -> HashMap<AlgorithmType, f64> {
        let mut dist = HashMap::new();
        for member in &self.members {
            *dist.entry(member.algorithm).or_insert(0.0) += member.weight;
        }
        dist
    }
}

/// Ensemble builder using greedy selection
#[derive(Debug, Clone)]
pub struct EnsembleBuilder {
    /// Configuration
    config: EnsembleConfig,
    /// Random number generator (for tie-breaking and stochastic selection)
    #[allow(dead_code)]
    rng: rand::rngs::StdRng,
}

impl EnsembleBuilder {
    /// Create a new ensemble builder
    pub fn new(config: EnsembleConfig) -> Self {
        use rand::SeedableRng;

        let rng = match config.random_state {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_os_rng(),
        };

        Self { config, rng }
    }

    /// Build ensemble from trial results
    ///
    /// # Arguments
    /// * `trials` - Trial results with OOF predictions
    /// * `y_true` - True target values for scoring
    ///
    /// # Returns
    /// Ensemble result with selected models and weights
    pub fn build_from_trials(
        &mut self,
        trials: &[TrialResult],
        y_true: &Array1<f64>,
    ) -> Result<EnsembleResult> {
        // Filter to successful trials with OOF predictions
        let valid_trials: Vec<&TrialResult> = trials
            .iter()
            .filter(|t| t.success && t.has_oof_predictions())
            .collect();

        if valid_trials.is_empty() {
            return Err(FerroError::invalid_input(
                "No valid trials with OOF predictions available for ensemble construction",
            ));
        }

        // Find best single model
        let best_single = valid_trials
            .iter()
            .max_by(|a, b| {
                let cmp = a
                    .cv_score
                    .partial_cmp(&b.cv_score)
                    .unwrap_or(std::cmp::Ordering::Equal);
                if self.config.maximize {
                    cmp
                } else {
                    cmp.reverse()
                }
            })
            .unwrap();

        let best_single_score = best_single.cv_score;

        // Get OOF predictions matrix
        let predictions_matrix = self.build_predictions_matrix(&valid_trials, y_true.len())?;

        // Run greedy selection
        let (selection_counts, score_history) =
            self.greedy_selection(&predictions_matrix, y_true, &valid_trials)?;

        // Build ensemble members
        let mut members = Vec::new();
        let total_selections: usize = selection_counts.iter().sum();

        for (idx, &count) in selection_counts.iter().enumerate() {
            if count > 0 {
                let trial = valid_trials[idx];
                let weight = count as f64 / total_selections as f64;

                if weight >= self.config.min_weight {
                    members.push(EnsembleMember {
                        trial_id: trial.trial_id,
                        algorithm: trial.algorithm,
                        weight,
                        selection_count: count,
                        params: trial.params.clone(),
                    });
                }
            }
        }

        // Renormalize weights after filtering
        let total_weight: f64 = members.iter().map(|m| m.weight).sum();
        if total_weight > 0.0 {
            for member in &mut members {
                member.weight /= total_weight;
            }
        }

        // Optionally optimize weights
        if self.config.optimize_weights && members.len() > 1 {
            self.optimize_weights(&mut members, &predictions_matrix, y_true, &valid_trials)?;
        }

        // Calculate final ensemble score
        let ensemble_preds = self.compute_ensemble_predictions(
            &members
                .iter()
                .map(|m| {
                    (
                        valid_trials
                            .iter()
                            .position(|t| t.trial_id == m.trial_id)
                            .unwrap(),
                        m.weight,
                    )
                })
                .collect::<Vec<_>>(),
            &predictions_matrix,
        );
        let ensemble_score = self.compute_score(&ensemble_preds, y_true);

        let improvement = if self.config.maximize {
            ensemble_score - best_single_score
        } else {
            best_single_score - ensemble_score
        };

        Ok(EnsembleResult {
            members,
            ensemble_score,
            best_single_score,
            improvement,
            iterations_performed: score_history.len(),
            score_history,
            task: self.config.task,
        })
    }

    /// Build a matrix of OOF predictions from trials
    fn build_predictions_matrix(
        &self,
        trials: &[&TrialResult],
        n_samples: usize,
    ) -> Result<Array2<f64>> {
        let n_trials = trials.len();

        // For classification with probabilities, we use the positive class probability
        // For regression, we use the predictions directly
        let mut matrix = Array2::zeros((n_samples, n_trials));

        for (col_idx, trial) in trials.iter().enumerate() {
            if let Some(ref preds) = trial.oof_predictions {
                if preds.len() != n_samples {
                    return Err(FerroError::shape_mismatch(
                        format!("{} samples", n_samples),
                        format!("{} predictions", preds.len()),
                    ));
                }
                matrix.column_mut(col_idx).assign(preds);
            } else if let Some(ref probs) = trial.oof_probabilities {
                if probs.nrows() != n_samples {
                    return Err(FerroError::shape_mismatch(
                        format!("{} samples", n_samples),
                        format!("{} predictions", probs.nrows()),
                    ));
                }
                // Use last column (positive class for binary, or treat as regression-like)
                let last_col = probs.ncols() - 1;
                matrix.column_mut(col_idx).assign(&probs.column(last_col));
            }
        }

        Ok(matrix)
    }

    /// Run greedy ensemble selection
    fn greedy_selection(
        &mut self,
        predictions: &Array2<f64>,
        y_true: &Array1<f64>,
        trials: &[&TrialResult],
    ) -> Result<(Vec<usize>, Vec<f64>)> {
        let n_trials = predictions.ncols();
        let mut selection_counts = vec![0usize; n_trials];
        let mut score_history = Vec::new();
        let mut best_score = if self.config.maximize {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };

        // Track current ensemble predictions
        let mut ensemble_preds = Array1::zeros(y_true.len());
        let mut total_weight = 0.0;

        for iteration in 0..self.config.selection_iterations {
            let mut best_candidate = None;
            let mut best_candidate_score = best_score;

            // Try adding each model
            for model_idx in 0..n_trials {
                // Check max models constraint
                let n_unique = selection_counts.iter().filter(|&&c| c > 0).count();
                if selection_counts[model_idx] == 0 && n_unique >= self.config.max_models {
                    continue;
                }

                // Compute new ensemble with this model added
                let new_weight = total_weight + 1.0;
                let model_preds = predictions.column(model_idx);

                // Incremental update: new_ensemble = (old_ensemble * old_weight + new_model) / new_weight
                let new_ensemble: Array1<f64> = ensemble_preds
                    .iter()
                    .zip(model_preds.iter())
                    .map(|(&e, &m)| (e * total_weight + m) / new_weight)
                    .collect();

                let mut score = self.compute_score(&new_ensemble, y_true);

                // Add diversity bonus if enabled
                if self.config.use_diversity {
                    let diversity = self.compute_diversity(model_idx, &selection_counts, trials);
                    let diversity_bonus = self.config.diversity_weight * diversity;
                    if self.config.maximize {
                        score += diversity_bonus;
                    } else {
                        score -= diversity_bonus;
                    }
                }

                let is_better = if self.config.maximize {
                    score > best_candidate_score
                } else {
                    score < best_candidate_score
                };

                if is_better {
                    best_candidate = Some(model_idx);
                    best_candidate_score = score;
                }
            }

            // Add best candidate to ensemble
            if let Some(model_idx) = best_candidate {
                selection_counts[model_idx] += 1;
                total_weight += 1.0;

                // Update ensemble predictions
                let model_preds = predictions.column(model_idx);
                ensemble_preds = ensemble_preds
                    .iter()
                    .zip(model_preds.iter())
                    .map(|(&e, &m)| (e * (total_weight - 1.0) + m) / total_weight)
                    .collect();

                // Record actual score (without diversity)
                let actual_score = self.compute_score(&ensemble_preds, y_true);
                score_history.push(actual_score);
                best_score = actual_score;
            } else {
                // No improvement possible
                break;
            }

            // Early stopping if score hasn't improved for several iterations
            if iteration >= 10 {
                let recent = &score_history[score_history.len().saturating_sub(5)..];
                let all_same = recent.windows(2).all(|w| (w[1] - w[0]).abs() < 1e-10);
                if all_same {
                    break;
                }
            }
        }

        Ok((selection_counts, score_history))
    }

    /// Compute ensemble predictions given member weights
    fn compute_ensemble_predictions(
        &self,
        members: &[(usize, f64)], // (model_idx, weight)
        predictions: &Array2<f64>,
    ) -> Array1<f64> {
        let n_samples = predictions.nrows();
        let mut ensemble = Array1::zeros(n_samples);

        for &(model_idx, weight) in members {
            let model_preds = predictions.column(model_idx);
            ensemble = ensemble + model_preds.to_owned() * weight;
        }

        ensemble
    }

    /// Compute score for predictions
    fn compute_score(&self, predictions: &Array1<f64>, y_true: &Array1<f64>) -> f64 {
        match self.config.task {
            Task::Classification => {
                // For classification, use accuracy based on 0.5 threshold
                // In practice, this would use the configured metric
                let correct: usize = predictions
                    .iter()
                    .zip(y_true.iter())
                    .filter(|(&p, &y)| {
                        let pred_class = if p >= 0.5 { 1.0 } else { 0.0 };
                        (pred_class - y).abs() < 1e-10
                    })
                    .count();
                correct as f64 / y_true.len() as f64
            }
            Task::Regression | Task::TimeSeries | Task::Survival => {
                // For regression, use negative MSE (so higher is better)
                let mse: f64 = predictions
                    .iter()
                    .zip(y_true.iter())
                    .map(|(&p, &y)| (p - y).powi(2))
                    .sum::<f64>()
                    / y_true.len() as f64;
                -mse
            }
        }
    }

    /// Compute diversity score for a model
    fn compute_diversity(
        &self,
        model_idx: usize,
        selection_counts: &[usize],
        trials: &[&TrialResult],
    ) -> f64 {
        let model_algo = trials[model_idx].algorithm;

        // Count how many times each algorithm type is in the ensemble
        let mut algo_counts: HashMap<AlgorithmType, usize> = HashMap::new();
        for (idx, &count) in selection_counts.iter().enumerate() {
            if count > 0 {
                *algo_counts.entry(trials[idx].algorithm).or_insert(0) += count;
            }
        }

        // Diversity bonus for algorithms not yet in ensemble or underrepresented
        let total: usize = selection_counts.iter().sum();
        if total == 0 {
            return 1.0; // Maximum diversity for first selection
        }

        let algo_count = algo_counts.get(&model_algo).copied().unwrap_or(0);
        let algo_ratio = algo_count as f64 / total as f64;

        // Higher diversity score when algorithm is less represented
        1.0 - algo_ratio
    }

    /// Optimize ensemble weights using coordinate descent
    fn optimize_weights(
        &self,
        members: &mut Vec<EnsembleMember>,
        predictions: &Array2<f64>,
        y_true: &Array1<f64>,
        trials: &[&TrialResult],
    ) -> Result<()> {
        if members.is_empty() {
            return Ok(());
        }

        // Get indices of members in predictions matrix
        let member_indices: Vec<usize> = members
            .iter()
            .map(|m| {
                trials
                    .iter()
                    .position(|t| t.trial_id == m.trial_id)
                    .unwrap()
            })
            .collect();

        // Initialize weights
        let mut weights: Vec<f64> = members.iter().map(|m| m.weight).collect();

        // Simple coordinate descent optimization
        let max_iter = 100;
        let step_size = 0.1;
        let tolerance = 1e-6;

        let mut best_score = {
            let preds = self.compute_weighted_predictions(&member_indices, &weights, predictions);
            self.compute_score(&preds, y_true)
        };

        for _ in 0..max_iter {
            let mut improved = false;

            for i in 0..weights.len() {
                // Try increasing weight
                let old_weight = weights[i];

                weights[i] = (old_weight + step_size).min(1.0);
                self.normalize_weights(&mut weights);

                let preds =
                    self.compute_weighted_predictions(&member_indices, &weights, predictions);
                let score = self.compute_score(&preds, y_true);

                let is_better = if self.config.maximize {
                    score > best_score + tolerance
                } else {
                    score < best_score - tolerance
                };

                if is_better {
                    best_score = score;
                    improved = true;
                } else {
                    // Try decreasing weight
                    weights[i] = (old_weight - step_size).max(0.01);
                    self.normalize_weights(&mut weights);

                    let preds =
                        self.compute_weighted_predictions(&member_indices, &weights, predictions);
                    let score = self.compute_score(&preds, y_true);

                    let is_better = if self.config.maximize {
                        score > best_score + tolerance
                    } else {
                        score < best_score - tolerance
                    };

                    if is_better {
                        best_score = score;
                        improved = true;
                    } else {
                        // Restore original weight
                        weights[i] = old_weight;
                        self.normalize_weights(&mut weights);
                    }
                }
            }

            if !improved {
                break;
            }
        }

        // Apply optimized weights
        for (member, &weight) in members.iter_mut().zip(weights.iter()) {
            member.weight = weight;
        }

        Ok(())
    }

    /// Compute weighted ensemble predictions
    fn compute_weighted_predictions(
        &self,
        indices: &[usize],
        weights: &[f64],
        predictions: &Array2<f64>,
    ) -> Array1<f64> {
        let n_samples = predictions.nrows();
        let mut result = Array1::zeros(n_samples);

        for (&idx, &weight) in indices.iter().zip(weights.iter()) {
            result = result + predictions.column(idx).to_owned() * weight;
        }

        result
    }

    /// Normalize weights to sum to 1
    fn normalize_weights(&self, weights: &mut Vec<f64>) {
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }
    }
}

/// Quick function to build an ensemble from trials
pub fn build_ensemble(
    trials: &[TrialResult],
    y_true: &Array1<f64>,
    task: Task,
) -> Result<EnsembleResult> {
    let config = EnsembleConfig::new().with_task(task);
    let mut builder = EnsembleBuilder::new(config);
    builder.build_from_trials(trials, y_true)
}

/// Build ensemble with custom configuration
pub fn build_ensemble_with_config(
    trials: &[TrialResult],
    y_true: &Array1<f64>,
    config: EnsembleConfig,
) -> Result<EnsembleResult> {
    let mut builder = EnsembleBuilder::new(config);
    builder.build_from_trials(trials, y_true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_trials() -> Vec<TrialResult> {
        // Create trials with OOF predictions that complement each other
        let n_samples = 100;

        // Trial 1: Good on first half
        let mut preds1 = Array1::zeros(n_samples);
        for i in 0..50 {
            preds1[i] = 0.9; // Correct for class 0
        }
        for i in 50..100 {
            preds1[i] = 0.6; // Somewhat correct for class 1
        }

        let trial1 = TrialResult::new(
            1,
            AlgorithmType::LogisticRegression,
            0.75,
            0.05,
            vec![0.72, 0.74, 0.76, 0.78, 0.75],
        )
        .with_oof_predictions(preds1);

        // Trial 2: Good on second half
        let mut preds2 = Array1::zeros(n_samples);
        for i in 0..50 {
            preds2[i] = 0.4; // Somewhat correct for class 0
        }
        for i in 50..100 {
            preds2[i] = 0.95; // Very correct for class 1
        }

        let trial2 = TrialResult::new(
            2,
            AlgorithmType::RandomForestClassifier,
            0.78,
            0.04,
            vec![0.76, 0.78, 0.80, 0.79, 0.77],
        )
        .with_oof_predictions(preds2);

        // Trial 3: Moderate overall
        let mut preds3 = Array1::zeros(n_samples);
        for i in 0..50 {
            preds3[i] = 0.7;
        }
        for i in 50..100 {
            preds3[i] = 0.7;
        }

        let trial3 = TrialResult::new(
            3,
            AlgorithmType::GaussianNB,
            0.70,
            0.06,
            vec![0.68, 0.70, 0.72, 0.70, 0.70],
        )
        .with_oof_predictions(preds3);

        // Trial 4: Failed trial (should be excluded)
        let trial4 = TrialResult::failed(4, AlgorithmType::SVC, "Convergence failed");

        vec![trial1, trial2, trial3, trial4]
    }

    fn create_test_targets() -> Array1<f64> {
        // Binary classification: 0 for first half, 1 for second half
        let mut y = Array1::zeros(100);
        for i in 50..100 {
            y[i] = 1.0;
        }
        y
    }

    #[test]
    fn test_trial_result_creation() {
        let trial = TrialResult::new(
            1,
            AlgorithmType::LogisticRegression,
            0.85,
            0.03,
            vec![0.82, 0.85, 0.88],
        );

        assert_eq!(trial.trial_id, 1);
        assert_eq!(trial.algorithm, AlgorithmType::LogisticRegression);
        assert!((trial.cv_score - 0.85).abs() < 1e-10);
        assert!(trial.success);
        assert!(!trial.has_oof_predictions());
    }

    #[test]
    fn test_trial_result_with_predictions() {
        let preds = Array1::from_vec(vec![0.1, 0.9, 0.5, 0.7]);
        let trial = TrialResult::new(
            1,
            AlgorithmType::LogisticRegression,
            0.85,
            0.03,
            vec![0.82, 0.85, 0.88],
        )
        .with_oof_predictions(preds);

        assert!(trial.has_oof_predictions());
        assert!(trial.oof_predictions.is_some());
    }

    #[test]
    fn test_failed_trial() {
        let trial = TrialResult::failed(1, AlgorithmType::SVC, "Memory error");

        assert!(!trial.success);
        assert_eq!(trial.error_message.as_deref(), Some("Memory error"));
        assert!(trial.cv_score.is_infinite());
    }

    #[test]
    fn test_ensemble_config_builder() {
        let config = EnsembleConfig::new()
            .with_max_models(10)
            .with_selection_iterations(25)
            .with_min_weight(0.05)
            .with_diversity(0.2)
            .with_task(Task::Classification);

        assert_eq!(config.max_models, 10);
        assert_eq!(config.selection_iterations, 25);
        assert!((config.min_weight - 0.05).abs() < 1e-10);
        assert!(config.use_diversity);
        assert!((config.diversity_weight - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_ensemble_builder_basic() {
        let trials = create_test_trials();
        let y_true = create_test_targets();

        let config = EnsembleConfig::new()
            .with_task(Task::Classification)
            .with_selection_iterations(10)
            .with_random_state(42);

        let mut builder = EnsembleBuilder::new(config);
        let result = builder.build_from_trials(&trials, &y_true);

        assert!(result.is_ok());
        let ensemble = result.unwrap();

        // Should have selected some models
        assert!(!ensemble.members.is_empty());

        // Weights should sum to 1
        let total_weight: f64 = ensemble.members.iter().map(|m| m.weight).sum();
        assert!(
            (total_weight - 1.0).abs() < 1e-6,
            "Weights should sum to 1: {}",
            total_weight
        );

        // Score should be reasonable
        assert!(ensemble.ensemble_score > 0.5, "Score should be > 0.5");
    }

    #[test]
    fn test_ensemble_excludes_failed_trials() {
        let trials = create_test_trials();
        let y_true = create_test_targets();

        let config = EnsembleConfig::new()
            .with_task(Task::Classification)
            .with_random_state(42);

        let mut builder = EnsembleBuilder::new(config);
        let result = builder.build_from_trials(&trials, &y_true).unwrap();

        // Failed trial (id=4) should not be in the ensemble
        assert!(!result.members.iter().any(|m| m.trial_id == 4));
    }

    #[test]
    fn test_ensemble_improvement() {
        let trials = create_test_trials();
        let y_true = create_test_targets();

        let config = EnsembleConfig::new()
            .with_task(Task::Classification)
            .with_selection_iterations(20)
            .with_random_state(42);

        let mut builder = EnsembleBuilder::new(config);
        let result = builder.build_from_trials(&trials, &y_true).unwrap();

        // Ensemble should be at least as good as the best single model
        assert!(
            result.ensemble_score >= result.best_single_score - 1e-6,
            "Ensemble ({}) should be >= best single ({})",
            result.ensemble_score,
            result.best_single_score
        );
    }

    #[test]
    fn test_ensemble_with_diversity() {
        let trials = create_test_trials();
        let y_true = create_test_targets();

        let config = EnsembleConfig::new()
            .with_task(Task::Classification)
            .with_diversity(0.3)
            .with_selection_iterations(15)
            .with_random_state(42);

        let mut builder = EnsembleBuilder::new(config);
        let result = builder.build_from_trials(&trials, &y_true).unwrap();

        // With diversity, we expect multiple algorithm types
        let algo_dist = result.algorithm_distribution();
        // May have one or more algorithm types - just ensure it runs
        assert!(!algo_dist.is_empty());
    }

    #[test]
    fn test_ensemble_max_models_constraint() {
        let trials = create_test_trials();
        let y_true = create_test_targets();

        let config = EnsembleConfig::new()
            .with_task(Task::Classification)
            .with_max_models(2)
            .with_selection_iterations(30)
            .with_random_state(42);

        let mut builder = EnsembleBuilder::new(config);
        let result = builder.build_from_trials(&trials, &y_true).unwrap();

        // Should not exceed max_models
        assert!(
            result.n_models() <= 2,
            "Should have <= 2 models: {}",
            result.n_models()
        );
    }

    #[test]
    fn test_ensemble_result_methods() {
        let trials = create_test_trials();
        let y_true = create_test_targets();

        let config = EnsembleConfig::new()
            .with_task(Task::Classification)
            .with_random_state(42);

        let mut builder = EnsembleBuilder::new(config);
        let result = builder.build_from_trials(&trials, &y_true).unwrap();

        // Test n_models
        assert!(result.n_models() > 0);

        // Test total_weight
        assert!((result.total_weight() - 1.0).abs() < 1e-6);

        // Test members_by_weight
        let sorted = result.members_by_weight();
        if sorted.len() >= 2 {
            assert!(sorted[0].weight >= sorted[1].weight);
        }

        // Test algorithm_distribution
        let dist = result.algorithm_distribution();
        let total_dist: f64 = dist.values().sum();
        assert!((total_dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ensemble_empty_trials() {
        let trials: Vec<TrialResult> = Vec::new();
        let y_true = Array1::zeros(10);

        let config = EnsembleConfig::new().with_task(Task::Classification);
        let mut builder = EnsembleBuilder::new(config);
        let result = builder.build_from_trials(&trials, &y_true);

        assert!(result.is_err());
    }

    #[test]
    fn test_ensemble_all_failed_trials() {
        let trials = vec![
            TrialResult::failed(1, AlgorithmType::LogisticRegression, "Error 1"),
            TrialResult::failed(2, AlgorithmType::RandomForestClassifier, "Error 2"),
        ];
        let y_true = Array1::zeros(10);

        let config = EnsembleConfig::new().with_task(Task::Classification);
        let mut builder = EnsembleBuilder::new(config);
        let result = builder.build_from_trials(&trials, &y_true);

        assert!(result.is_err());
    }

    #[test]
    fn test_build_ensemble_convenience_function() {
        let trials = create_test_trials();
        let y_true = create_test_targets();

        let result = build_ensemble(&trials, &y_true, Task::Classification);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ensemble_regression_task() {
        let n_samples = 50;

        // Create regression trials
        let mut preds1 = Array1::zeros(n_samples);
        let mut preds2 = Array1::zeros(n_samples);
        let mut y_true = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let x = i as f64 / 10.0;
            y_true[i] = 2.0 * x + 1.0;
            preds1[i] = 2.0 * x + 0.5; // Slight underestimate
            preds2[i] = 2.0 * x + 1.5; // Slight overestimate
        }

        let trial1 = TrialResult::new(
            1,
            AlgorithmType::LinearRegression,
            -0.25, // MSE
            0.05,
            vec![-0.24, -0.25, -0.26],
        )
        .with_oof_predictions(preds1);

        let trial2 = TrialResult::new(
            2,
            AlgorithmType::Ridge,
            -0.25,
            0.05,
            vec![-0.24, -0.25, -0.26],
        )
        .with_oof_predictions(preds2);

        let trials = vec![trial1, trial2];

        let config = EnsembleConfig::new()
            .with_task(Task::Regression)
            .with_maximize(false) // Minimize MSE
            .with_random_state(42);

        let mut builder = EnsembleBuilder::new(config);
        let result = builder.build_from_trials(&trials, &y_true).unwrap();

        // Ensemble should improve by averaging complementary predictions
        assert!(!result.members.is_empty());
    }

    #[test]
    fn test_param_value_serialization() {
        let params: HashMap<String, ParamValue> = [
            ("n_estimators".to_string(), ParamValue::Int(100)),
            ("learning_rate".to_string(), ParamValue::Float(0.1)),
            ("kernel".to_string(), ParamValue::String("rbf".to_string())),
            ("bootstrap".to_string(), ParamValue::Bool(true)),
        ]
        .into_iter()
        .collect();

        let trial = TrialResult::new(
            1,
            AlgorithmType::RandomForestClassifier,
            0.85,
            0.03,
            vec![0.85],
        )
        .with_params(params);

        // Test serialization roundtrip
        let json = serde_json::to_string(&trial).unwrap();
        let deserialized: TrialResult = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.trial_id, trial.trial_id);
        assert_eq!(deserialized.algorithm, trial.algorithm);
    }

    #[test]
    fn test_ensemble_score_history() {
        let trials = create_test_trials();
        let y_true = create_test_targets();

        let config = EnsembleConfig::new()
            .with_task(Task::Classification)
            .with_selection_iterations(20)
            .with_random_state(42);

        let mut builder = EnsembleBuilder::new(config);
        let result = builder.build_from_trials(&trials, &y_true).unwrap();

        // Score history should be non-empty
        assert!(!result.score_history.is_empty());

        // Scores should generally be non-decreasing (or at least stable)
        for window in result.score_history.windows(2) {
            // Allow small decreases due to numerical issues
            assert!(
                window[1] >= window[0] - 1e-6,
                "Score should not decrease significantly: {} -> {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_ensemble_weight_optimization() {
        let trials = create_test_trials();
        let y_true = create_test_targets();

        // First without optimization
        let config_no_opt = EnsembleConfig::new()
            .with_task(Task::Classification)
            .with_weight_optimization(false)
            .with_random_state(42);

        let mut builder_no_opt = EnsembleBuilder::new(config_no_opt);
        let result_no_opt = builder_no_opt.build_from_trials(&trials, &y_true).unwrap();

        // Then with optimization
        let config_opt = EnsembleConfig::new()
            .with_task(Task::Classification)
            .with_weight_optimization(true)
            .with_random_state(42);

        let mut builder_opt = EnsembleBuilder::new(config_opt);
        let result_opt = builder_opt.build_from_trials(&trials, &y_true).unwrap();

        // Optimized should be at least as good
        assert!(
            result_opt.ensemble_score >= result_no_opt.ensemble_score - 1e-6,
            "Optimized ({}) should be >= non-optimized ({})",
            result_opt.ensemble_score,
            result_no_opt.ensemble_score
        );
    }
}

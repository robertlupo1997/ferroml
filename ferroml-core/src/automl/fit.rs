//! AutoML Fit Implementation
//!
//! This module provides the end-to-end `fit()` method for AutoML that orchestrates:
//! - Portfolio selection based on task type
//! - Data-aware search space adaptation
//! - Cross-validation for each algorithm configuration
//! - Time budget allocation using multi-armed bandit strategies
//! - Ensemble construction from successful trials
//! - Comprehensive result reporting with statistical significance
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> ferroml_core::Result<()> {
//! use ferroml_core::{AutoML, AutoMLConfig, Task, Metric};
//! # use ndarray::{Array1, Array2};
//! # let x = Array2::from_shape_vec((20, 3), (0..60).map(|i| i as f64).collect()).unwrap();
//! # let y = Array1::from_vec((0..20).map(|i| (i % 2) as f64).collect());
//!
//! let config = AutoMLConfig {
//!     task: Task::Classification,
//!     metric: Metric::RocAuc,
//!     time_budget_seconds: 3600,
//!     cv_folds: 5,
//!     ..Default::default()
//! };
//!
//! let automl = AutoML::new(config);
//! let result = automl.fit(&x, &y)?;
//!
//! println!("Best model: {}", result.best_model().unwrap().algorithm.name());
//! println!("Ensemble score: {:.4}", result.ensemble_score().unwrap());
//!
//! // Get statistical comparisons between models
//! if let Some(comparisons) = &result.model_comparisons {
//!     for comp in &comparisons.pairwise_comparisons {
//!         println!("{} vs {}: p={:.4}", comp.model1_name, comp.model2_name, comp.p_value);
//!     }
//! }
//!
//! // Get aggregated feature importance
//! if let Some(importance) = &result.aggregated_importance {
//!     println!("Top features: {:?}", importance.top_k(5));
//! }
//! # Ok(())
//! # }
//! ```

use crate::automl::{
    AlgorithmConfig, AlgorithmPortfolio, AlgorithmType, DataCharacteristics, EnsembleBuilder,
    EnsembleConfig, EnsembleResult, ParamValue, TimeBudgetAllocator, TimeBudgetConfig, TrialResult,
};
use crate::cv::{CVFold, CrossValidator, StratifiedKFold};
use crate::metrics::regression::mape;
use crate::metrics::{
    accuracy, corrected_resampled_ttest, f1_score, log_loss, matthews_corrcoef, mse, roc_auc_score,
    Average, Direction, Metric as MetricTrait, MetricValue,
};
use crate::models::{
    CategoricalNB, DecisionTreeClassifier, DecisionTreeRegressor, GaussianNB,
    GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, KNeighborsClassifier, KNeighborsRegressor, LinearRegression,
    LinearSVC, LinearSVR, LogisticRegression, Model, MultinomialNB, QuantileRegression,
    RandomForestClassifier, RandomForestRegressor, RidgeRegression, RobustRegression, SVC, SVR,
};
use crate::{AutoML, FerroError, Metric, Result, Task};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Result of running AutoML fit
///
/// This comprehensive result object provides:
/// - Leaderboard with confidence intervals for all successful models
/// - Ensemble composition and performance
/// - Aggregated feature importance across models
/// - Statistical significance tests comparing models
/// - Detailed diagnostics and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMLResult {
    /// All trial results (including failed ones)
    pub trials: Vec<TrialResult>,
    /// Successful trials sorted by score (best first)
    pub leaderboard: Vec<LeaderboardEntry>,
    /// Ensemble result (if ensemble was built)
    pub ensemble: Option<EnsembleResult>,
    /// Total time spent in seconds
    pub total_time_seconds: f64,
    /// Number of successful trials
    pub n_successful_trials: usize,
    /// Number of failed trials
    pub n_failed_trials: usize,
    /// Task type
    pub task: Task,
    /// Metric used for optimization
    pub metric_name: String,
    /// Whether higher metric values are better
    pub maximize: bool,
    /// Data characteristics detected
    pub data_characteristics: DataCharacteristics,
    /// Cross-validation configuration used
    pub cv_folds: usize,
    /// Statistical summary of best model performance
    pub best_model_stats: Option<ModelStatistics>,
    /// Aggregated feature importance across all models
    pub aggregated_importance: Option<AggregatedFeatureImportance>,
    /// Statistical comparison results between top models
    pub model_comparisons: Option<ModelComparisonResults>,
}

impl AutoMLResult {
    /// Get the best model entry from the leaderboard
    #[must_use]
    pub fn best_model(&self) -> Option<&LeaderboardEntry> {
        self.leaderboard.first()
    }

    /// Predict using the best model
    ///
    /// This method re-creates and re-fits the best model from the leaderboard,
    /// then uses it to make predictions on new data.
    ///
    /// # Arguments
    /// * `x_train` - Training features used to fit the best model
    /// * `y_train` - Training targets used to fit the best model
    /// * `x_test` - Test features to predict on
    ///
    /// # Returns
    /// * Predictions for `x_test`
    ///
    /// # Errors
    /// Returns an error if:
    /// - No model succeeded in AutoML
    /// - Model creation fails
    /// - Model fitting fails
    /// - Prediction fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> ferroml_core::Result<()> {
    /// # use ferroml_core::{AutoML, AutoMLConfig};
    /// # use ndarray::{Array1, Array2};
    /// # let x_train = Array2::from_shape_vec((20, 3), (0..60).map(|i| i as f64).collect()).unwrap();
    /// # let y_train = Array1::from_vec((0..20).map(|i| (i % 2) as f64).collect());
    /// # let x_test = x_train.clone();
    /// # let automl = AutoML::new(AutoMLConfig::default());
    /// let result = automl.fit(&x_train, &y_train)?;
    /// let predictions = result.predict(&x_train, &y_train, &x_test)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict(
        &self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        x_test: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        // Get best model from leaderboard
        let best = self
            .best_model()
            .ok_or_else(|| FerroError::not_fitted("No successful models in AutoML result"))?;

        // Re-create the model
        let mut model = create_model(best.algorithm)?;

        // Fit on full training data
        model.fit(x_train, y_train)?;

        // Predict on test data
        model.predict(x_test)
    }

    /// Get the ensemble score (if ensemble was built)
    #[must_use]
    pub fn ensemble_score(&self) -> Option<f64> {
        self.ensemble.as_ref().map(|e| e.ensemble_score)
    }

    /// Get improvement from ensemble over best single model
    #[must_use]
    pub fn ensemble_improvement(&self) -> Option<f64> {
        self.ensemble.as_ref().map(|e| e.improvement)
    }

    /// Check if the AutoML run was successful (at least one trial succeeded)
    #[must_use]
    pub fn is_successful(&self) -> bool {
        self.n_successful_trials > 0
    }

    /// Get the number of unique algorithms that succeeded
    #[must_use]
    pub fn n_successful_algorithms(&self) -> usize {
        let algos: std::collections::HashSet<_> =
            self.leaderboard.iter().map(|e| e.algorithm).collect();
        algos.len()
    }

    /// Get top k most important features (aggregated across models)
    ///
    /// Returns None if feature importance wasn't computed.
    #[must_use]
    pub fn top_features(&self, k: usize) -> Option<Vec<(String, f64, f64, f64)>> {
        self.aggregated_importance.as_ref().map(|imp| imp.top_k(k))
    }

    /// Get models that are not significantly worse than the best
    ///
    /// These models are "competitive" with the best model according to
    /// statistical significance testing.
    #[must_use]
    pub fn competitive_models(&self) -> Vec<&LeaderboardEntry> {
        let competitive_ids = self
            .model_comparisons
            .as_ref()
            .map(|c| c.models_competitive_with_best())
            .unwrap_or_else(|| {
                // If no comparisons, consider top model competitive
                self.leaderboard
                    .first()
                    .map(|e| vec![e.trial_id])
                    .unwrap_or_default()
            });

        self.leaderboard
            .iter()
            .filter(|e| competitive_ids.contains(&e.trial_id))
            .collect()
    }

    /// Check if a specific model is significantly different from another
    #[must_use]
    pub fn models_significantly_different(&self, trial_id_1: usize, trial_id_2: usize) -> bool {
        self.model_comparisons.as_ref().map_or(false, |c| {
            c.are_significantly_different(trial_id_1, trial_id_2)
        })
    }

    /// Get a comprehensive summary of the AutoML results
    #[must_use]
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "╔══════════════════════════════════════════════════════════════╗".to_string(),
            "║                    AutoML Results Summary                     ║".to_string(),
            "╚══════════════════════════════════════════════════════════════╝".to_string(),
            String::new(),
            format!(
                "Task: {:?} | Metric: {} | Time: {:.1}s",
                self.task, self.metric_name, self.total_time_seconds
            ),
            format!(
                "Trials: {} successful, {} failed",
                self.n_successful_trials, self.n_failed_trials
            ),
            String::new(),
        ];

        // Best model summary
        if let Some(best) = self.best_model() {
            lines.push(
                "┌─ Best Model ─────────────────────────────────────────────────┐".to_string(),
            );
            lines.push(format!("│  Algorithm: {:?}", best.algorithm));
            lines.push(format!(
                "│  Score: {:.4} ± {:.4}",
                best.cv_score, best.cv_std
            ));
            lines.push(format!(
                "│  95% CI: [{:.4}, {:.4}]",
                best.ci_lower, best.ci_upper
            ));
            lines.push(format!(
                "│  Training time: {:.2}s",
                best.training_time_seconds
            ));
            lines.push(
                "└──────────────────────────────────────────────────────────────┘".to_string(),
            );
            lines.push(String::new());
        }

        // Leaderboard
        lines.push("┌─ Leaderboard (Top 10) ──────────────────────────────────────┐".to_string());
        for entry in self.leaderboard.iter().take(10) {
            let competitive = self
                .competitive_models()
                .iter()
                .any(|c| c.trial_id == entry.trial_id);
            let marker = if competitive { "●" } else { " " };
            lines.push(format!(
                "│ {} {:2}. {:25} {:.4} ± {:.4}",
                marker,
                entry.rank,
                format!("{:?}", entry.algorithm),
                entry.cv_score,
                entry.cv_std
            ));
        }
        if self.leaderboard.len() > 10 {
            lines.push(format!(
                "│    ... and {} more models",
                self.leaderboard.len() - 10
            ));
        }
        lines.push("└──────────────────────────────────────────────────────────────┘".to_string());
        lines.push("  ● = statistically competitive with best".to_string());
        lines.push(String::new());

        // Ensemble
        if let Some(ensemble) = &self.ensemble {
            lines.push(
                "┌─ Ensemble ───────────────────────────────────────────────────┐".to_string(),
            );
            lines.push(format!("│  Score: {:.4}", ensemble.ensemble_score));
            let improvement_pct = if let Some(best) = self.best_model() {
                if best.cv_score != 0.0 {
                    (ensemble.improvement / best.cv_score.abs()) * 100.0
                } else {
                    0.0
                }
            } else {
                0.0
            };
            lines.push(format!(
                "│  Improvement over best: {:.4} ({:.2}%)",
                ensemble.improvement, improvement_pct
            ));
            lines.push(format!("│  Models in ensemble: {}", ensemble.members.len()));
            lines.push(
                "└──────────────────────────────────────────────────────────────┘".to_string(),
            );
            lines.push(String::new());
        }

        // Feature importance
        if let Some(importance) = &self.aggregated_importance {
            lines.push(
                "┌─ Top 5 Features ─────────────────────────────────────────────┐".to_string(),
            );
            for (name, imp, ci_l, ci_u) in importance.top_k(5) {
                lines.push(format!(
                    "│  {}: {:.4} (95% CI: [{:.4}, {:.4}])",
                    name, imp, ci_l, ci_u
                ));
            }
            lines.push(
                "└──────────────────────────────────────────────────────────────┘".to_string(),
            );
            lines.push(String::new());
        }

        // Model comparisons
        if let Some(comparisons) = &self.model_comparisons {
            lines.push(
                "┌─ Model Significance ────────────────────────────────────────┐".to_string(),
            );
            lines.push(format!(
                "│  Correction: {} (α = {:.4})",
                comparisons.correction_method, comparisons.corrected_alpha
            ));
            lines.push(format!(
                "│  Best significantly better than {}/{} models",
                comparisons.n_significantly_worse,
                comparisons.pairwise_comparisons.len()
            ));
            lines.push(
                "└──────────────────────────────────────────────────────────────┘".to_string(),
            );
        }

        lines.join("\n")
    }
}

/// Entry in the AutoML leaderboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    /// Rank (1 = best)
    pub rank: usize,
    /// Trial ID
    pub trial_id: usize,
    /// Algorithm type
    pub algorithm: AlgorithmType,
    /// Cross-validation score (mean)
    pub cv_score: f64,
    /// Standard deviation of CV scores
    pub cv_std: f64,
    /// 95% confidence interval lower bound
    pub ci_lower: f64,
    /// 95% confidence interval upper bound
    pub ci_upper: f64,
    /// Training time in seconds
    pub training_time_seconds: f64,
    /// Hyperparameters used
    pub params: HashMap<String, ParamValue>,
}

/// Statistical summary of model performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatistics {
    /// Mean CV score
    pub mean_score: f64,
    /// Standard deviation
    pub std_score: f64,
    /// 95% CI lower bound
    pub ci_lower: f64,
    /// 95% CI upper bound
    pub ci_upper: f64,
    /// Number of CV folds
    pub n_folds: usize,
    /// Per-fold scores
    pub fold_scores: Vec<f64>,
}

/// Aggregated feature importance across all successful models
///
/// This provides a weighted combination of feature importances from models
/// that support it, giving a consensus view of feature relevance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedFeatureImportance {
    /// Feature names (or indices if names unavailable)
    pub feature_names: Vec<String>,
    /// Mean importance per feature (weighted by model score)
    pub importance_mean: Vec<f64>,
    /// Standard deviation of importance across models
    pub importance_std: Vec<f64>,
    /// Lower bound of 95% CI for importance
    pub ci_lower: Vec<f64>,
    /// Upper bound of 95% CI for importance
    pub ci_upper: Vec<f64>,
    /// Number of models contributing to each feature's importance
    pub n_models_per_feature: Vec<usize>,
    /// Total number of models that contributed
    pub n_models: usize,
    /// Per-model importance values for detailed analysis (model_idx -> feature importances)
    pub per_model_importance: Vec<ModelFeatureImportance>,
}

impl AggregatedFeatureImportance {
    /// Get indices of features sorted by importance (highest first)
    #[must_use]
    pub fn sorted_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.importance_mean.len()).collect();
        indices.sort_by(|&a, &b| {
            self.importance_mean[b]
                .partial_cmp(&self.importance_mean[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Get the top k most important features
    ///
    /// Returns (feature_name, mean_importance, ci_lower, ci_upper) tuples
    #[must_use]
    pub fn top_k(&self, k: usize) -> Vec<(String, f64, f64, f64)> {
        self.sorted_indices()
            .into_iter()
            .take(k)
            .map(|i| {
                (
                    self.feature_names[i].clone(),
                    self.importance_mean[i],
                    self.ci_lower[i],
                    self.ci_upper[i],
                )
            })
            .collect()
    }

    /// Check if a feature has statistically significant importance
    ///
    /// A feature is significant if its CI doesn't include zero
    #[must_use]
    pub fn is_significant(&self, feature_idx: usize) -> bool {
        feature_idx < self.ci_lower.len()
            && (self.ci_lower[feature_idx] > 0.0 || self.ci_upper[feature_idx] < 0.0)
    }

    /// Get indices of all statistically significant features
    #[must_use]
    pub fn significant_features(&self) -> Vec<usize> {
        (0..self.importance_mean.len())
            .filter(|&i| self.is_significant(i))
            .collect()
    }

    /// Format as summary string
    #[must_use]
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "Aggregated Feature Importance".to_string(),
            "=============================".to_string(),
            format!("Models contributing: {}", self.n_models),
            String::new(),
            "Feature Importances (sorted by mean):".to_string(),
            "--------------------------------------".to_string(),
        ];

        for idx in self.sorted_indices().into_iter().take(20) {
            let sig = if self.is_significant(idx) { "*" } else { "" };
            lines.push(format!(
                "  {}: {:.4} ± {:.4} (95% CI: [{:.4}, {:.4}]){}",
                self.feature_names[idx],
                self.importance_mean[idx],
                self.importance_std[idx],
                self.ci_lower[idx],
                self.ci_upper[idx],
                sig
            ));
        }

        if self.feature_names.len() > 20 {
            lines.push(format!(
                "  ... and {} more features",
                self.feature_names.len() - 20
            ));
        }

        lines.push(String::new());
        lines.push("* indicates statistically significant (CI excludes zero)".to_string());

        lines.join("\n")
    }
}

/// Feature importance from a single model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFeatureImportance {
    /// Trial ID of the model
    pub trial_id: usize,
    /// Algorithm type
    pub algorithm: AlgorithmType,
    /// CV score (for weighting)
    pub cv_score: f64,
    /// Feature importance values
    pub importances: Vec<f64>,
}

/// Statistical comparison results between models
///
/// Contains pairwise comparisons between top models and overall statistics
/// about model performance differences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonResults {
    /// Pairwise comparisons between top models
    pub pairwise_comparisons: Vec<PairwiseComparison>,
    /// Whether the best model is significantly better than others
    pub best_is_significantly_better: bool,
    /// Number of models significantly worse than the best
    pub n_significantly_worse: usize,
    /// Critical p-value after multiple testing correction
    pub corrected_alpha: f64,
    /// Multiple testing correction method used
    pub correction_method: String,
}

impl ModelComparisonResults {
    /// Get comparisons involving a specific model by trial ID
    #[must_use]
    pub fn comparisons_for_model(&self, trial_id: usize) -> Vec<&PairwiseComparison> {
        self.pairwise_comparisons
            .iter()
            .filter(|c| c.trial_id_1 == trial_id || c.trial_id_2 == trial_id)
            .collect()
    }

    /// Check if two models are significantly different
    #[must_use]
    pub fn are_significantly_different(&self, trial_id_1: usize, trial_id_2: usize) -> bool {
        self.pairwise_comparisons
            .iter()
            .find(|c| {
                (c.trial_id_1 == trial_id_1 && c.trial_id_2 == trial_id_2)
                    || (c.trial_id_1 == trial_id_2 && c.trial_id_2 == trial_id_1)
            })
            .map_or(false, |c| c.significant_corrected)
    }

    /// Get models that are not significantly different from the best
    #[must_use]
    pub fn models_competitive_with_best(&self) -> Vec<usize> {
        if self.pairwise_comparisons.is_empty() {
            return vec![];
        }

        // Find the best model (appears as model1 in first comparison)
        let best_trial_id = self.pairwise_comparisons[0].trial_id_1;

        // Get all models that are not significantly worse than best
        let mut competitive = vec![best_trial_id];
        for comp in &self.pairwise_comparisons {
            if comp.trial_id_1 == best_trial_id && !comp.significant_corrected {
                competitive.push(comp.trial_id_2);
            }
        }

        competitive
    }

    /// Format as summary string
    #[must_use]
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "Model Comparison Results".to_string(),
            "========================".to_string(),
            format!(
                "Multiple testing correction: {} (α = {:.4})",
                self.correction_method, self.corrected_alpha
            ),
            format!(
                "Best model significantly better than {} of {} compared models",
                self.n_significantly_worse,
                self.pairwise_comparisons.len()
            ),
            String::new(),
            "Pairwise Comparisons:".to_string(),
            "---------------------".to_string(),
        ];

        for comp in &self.pairwise_comparisons {
            let sig_marker = if comp.significant_corrected {
                "**"
            } else if comp.significant {
                "*"
            } else {
                ""
            };
            lines.push(format!(
                "  {} vs {}: diff = {:.4} (95% CI: [{:.4}, {:.4}]), p = {:.4}{}",
                comp.model1_name,
                comp.model2_name,
                comp.mean_difference,
                comp.ci_lower,
                comp.ci_upper,
                comp.p_value,
                sig_marker
            ));
        }

        lines.push(String::new());
        lines.push("* significant at α=0.05".to_string());
        lines.push("** significant after multiple testing correction".to_string());

        lines.join("\n")
    }
}

/// Pairwise comparison between two models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseComparison {
    /// First model's trial ID
    pub trial_id_1: usize,
    /// Second model's trial ID
    pub trial_id_2: usize,
    /// First model's name (algorithm type)
    pub model1_name: String,
    /// Second model's name (algorithm type)
    pub model2_name: String,
    /// Test used (e.g., "Corrected Resampled t-test")
    pub test_name: String,
    /// Mean score difference (model1 - model2)
    pub mean_difference: f64,
    /// Standard error of the difference
    pub std_error: f64,
    /// Test statistic
    pub statistic: f64,
    /// Raw p-value
    pub p_value: f64,
    /// p-value after multiple testing correction
    pub p_value_corrected: f64,
    /// Lower bound of 95% CI for difference
    pub ci_lower: f64,
    /// Upper bound of 95% CI for difference
    pub ci_upper: f64,
    /// Whether significant at alpha=0.05 (uncorrected)
    pub significant: bool,
    /// Whether significant after multiple testing correction
    pub significant_corrected: bool,
}

/// Internal metric adapter that implements the CV Metric trait
struct MetricAdapter {
    metric: Metric,
    #[allow(dead_code)]
    task: Task,
}

impl MetricTrait for MetricAdapter {
    fn name(&self) -> &str {
        match self.metric {
            Metric::RocAuc => "roc_auc",
            Metric::Accuracy => "accuracy",
            Metric::F1 => "f1",
            Metric::LogLoss => "log_loss",
            Metric::Mcc => "mcc",
            Metric::Mse => "mse",
            Metric::Rmse => "rmse",
            Metric::Mae => "mae",
            Metric::R2 => "r2",
            Metric::Mape => "mape",
            Metric::Smape => "smape",
            Metric::Mase => "mase",
        }
    }

    fn direction(&self) -> Direction {
        match self.metric {
            Metric::Mse
            | Metric::Rmse
            | Metric::Mae
            | Metric::LogLoss
            | Metric::Mape
            | Metric::Smape
            | Metric::Mase => Direction::Minimize,
            _ => Direction::Maximize,
        }
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = match self.metric {
            Metric::Accuracy => accuracy(y_true, y_pred)?,
            Metric::Mse | Metric::Rmse => {
                let m = mse(y_true, y_pred)?;
                if self.metric == Metric::Rmse {
                    m.sqrt()
                } else {
                    m
                }
            }
            Metric::Mae => {
                y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(t, p)| (t - p).abs())
                    .sum::<f64>()
                    / y_true.len() as f64
            }
            Metric::R2 => {
                let ss_res = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(t, p)| (t - p).powi(2))
                    .sum::<f64>();
                let y_mean = y_true.mean().unwrap_or(0.0);
                let ss_tot = y_true.iter().map(|t| (t - y_mean).powi(2)).sum::<f64>();
                if ss_tot.abs() < 1e-10 {
                    1.0
                } else {
                    1.0 - ss_res / ss_tot
                }
            }
            Metric::Mape => mape(y_true, y_pred)?,
            Metric::Smape => {
                let n = y_true.len() as f64;
                y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(t, p)| {
                        let denom = (t.abs() + p.abs()) / 2.0;
                        if denom < 1e-15 {
                            0.0
                        } else {
                            (t - p).abs() / denom
                        }
                    })
                    .sum::<f64>()
                    / n
                    * 100.0
            }
            Metric::Mase => {
                let n = y_true.len();
                let mae_pred = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(t, p)| (t - p).abs())
                    .sum::<f64>()
                    / n as f64;
                let mae_naive = if n > 1 {
                    y_true
                        .iter()
                        .skip(1)
                        .zip(y_true.iter())
                        .map(|(t, t_prev)| (t - t_prev).abs())
                        .sum::<f64>()
                        / (n - 1) as f64
                } else {
                    1.0
                };
                if mae_naive < 1e-15 {
                    0.0
                } else {
                    mae_pred / mae_naive
                }
            }
            Metric::F1 => f1_score(y_true, y_pred, Average::Weighted)?,
            Metric::LogLoss => log_loss(y_true, y_pred, None)?,
            Metric::Mcc => matthews_corrcoef(y_true, y_pred)?,
            Metric::RocAuc => roc_auc_score(y_true, y_pred)?,
        };

        Ok(MetricValue::new(self.name(), value, self.direction()))
    }
}

impl AutoML {
    /// Fit the AutoML system to training data
    ///
    /// This method orchestrates the full AutoML pipeline:
    /// 1. Analyzes data characteristics
    /// 2. Selects and adapts algorithm portfolio
    /// 3. Runs cross-validation for each algorithm
    /// 4. Allocates time budget using multi-armed bandit
    /// 5. Builds ensemble from successful trials
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - Target values of shape (n_samples,)
    ///
    /// # Returns
    /// * `AutoMLResult` containing leaderboard, ensemble, and diagnostics
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> ferroml_core::Result<()> {
    /// # use ferroml_core::{AutoML, AutoMLConfig};
    /// # use ndarray::{Array1, Array2};
    /// # let x = Array2::from_shape_vec((20, 3), (0..60).map(|i| i as f64).collect()).unwrap();
    /// # let y = Array1::from_vec((0..20).map(|i| (i % 2) as f64).collect());
    /// let config = AutoMLConfig::default();
    /// let automl = AutoML::new(config);
    /// let result = automl.fit(&x, &y)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<AutoMLResult> {
        let start_time = Instant::now();
        let deadline = start_time + Duration::from_secs(self.config.time_budget_seconds);

        // Step 1: Analyze data characteristics
        let data_chars = DataCharacteristics::from_data(x, y);

        // Step 2: Select portfolio based on task (use config preset, not hardcoded)
        let preset = self.config.preset;
        let portfolio = match self.config.task {
            Task::Classification => AlgorithmPortfolio::for_classification(preset),
            Task::Regression => AlgorithmPortfolio::for_regression(preset),
            _ => AlgorithmPortfolio::for_regression(preset), // Default to regression for other tasks
        };

        // Adapt portfolio to data characteristics
        let adapted_portfolio = portfolio.adapt_to_data(&data_chars);

        // Step 3: Set up cross-validation
        let cv = StratifiedKFold::new(self.config.cv_folds).with_shuffle(true);

        // Get CV folds once (we'll reuse them for all trials)
        let folds = cv.split(x.nrows(), Some(y), None)?;

        // Step 4: Set up time budget allocator
        let time_config = TimeBudgetConfig::new(self.config.time_budget_seconds);
        let mut time_allocator = TimeBudgetAllocator::new(time_config, &adapted_portfolio);

        // Step 5: Run trials for each algorithm
        let mut trials: Vec<TrialResult> = Vec::new();
        let mut trial_id = 0;

        let metric_adapter = MetricAdapter {
            metric: self.config.metric,
            task: self.config.task,
        };
        let maximize = metric_adapter.direction() == Direction::Maximize;

        // Use bandit to select algorithms (not priority order)
        loop {
            // Check time budget
            if Instant::now() >= deadline {
                break;
            }

            // Get time allocation from bandit - this determines which algorithm to try
            let allocation = match time_allocator.select_arm() {
                Some(arm) => arm,
                None => break, // No more budget or all arms exhausted
            };

            // Skip if not enough time
            if allocation.trial_budget_seconds < 1.0 {
                continue;
            }

            // Use the bandit's selected algorithm index, not loop iteration order
            let algo_config = &adapted_portfolio.algorithms[allocation.algorithm_index];

            let trial_start = Instant::now();

            // Run cross-validation for this algorithm with default hyperparameters
            let trial_result = self.run_trial(trial_id, algo_config, x, y, &folds, &metric_adapter);

            let trial_time = trial_start.elapsed().as_secs_f64();

            // Update time allocator with result using correct algorithm index
            let reward = match &trial_result {
                Ok(result) if result.success => {
                    // Normalize score to [0, 1] for bandit
                    if maximize {
                        result.cv_score.max(0.0).min(1.0)
                    } else {
                        (1.0 - result.cv_score.abs()).max(0.0).min(1.0)
                    }
                }
                _ => 0.0,
            };
            time_allocator.update(allocation.algorithm_index, reward, trial_time);

            match trial_result {
                Ok(result) => {
                    trials.push(result);
                }
                Err(e) => {
                    trials.push(TrialResult::failed(
                        trial_id,
                        algo_config.algorithm,
                        e.to_string(),
                    ));
                }
            }

            trial_id += 1;
        }

        // Step 6: Build leaderboard
        let mut successful_trials: Vec<_> = trials.iter().filter(|t| t.success).collect();
        successful_trials.sort_by(|a, b| {
            let cmp = a
                .cv_score
                .partial_cmp(&b.cv_score)
                .unwrap_or(std::cmp::Ordering::Equal);
            if maximize {
                cmp.reverse()
            } else {
                cmp
            }
        });

        let leaderboard: Vec<LeaderboardEntry> = successful_trials
            .iter()
            .enumerate()
            .map(|(rank, trial)| {
                // Compute CI from fold scores
                let (ci_lower, ci_upper) = if !trial.fold_scores.is_empty() {
                    compute_t_ci(&trial.fold_scores, self.config.confidence_level)
                } else {
                    (trial.cv_score, trial.cv_score)
                };

                LeaderboardEntry {
                    rank: rank + 1,
                    trial_id: trial.trial_id,
                    algorithm: trial.algorithm,
                    cv_score: trial.cv_score,
                    cv_std: trial.cv_std,
                    ci_lower,
                    ci_upper,
                    training_time_seconds: trial.training_time_seconds,
                    params: trial.params.clone(),
                }
            })
            .collect();

        // Step 7: Build ensemble (if we have enough successful trials)
        let ensemble = if successful_trials.len() >= 2 {
            let ensemble_config = EnsembleConfig::new()
                .with_task(self.config.task)
                .with_maximize(maximize)
                .with_max_models(10)
                .with_selection_iterations(50);

            let mut builder = EnsembleBuilder::new(ensemble_config);
            match builder.build_from_trials(&trials, y) {
                Ok(ens) => Some(ens),
                Err(e) => {
                    // Log warning but allow graceful degradation to best single model
                    eprintln!(
                        "Warning: Ensemble building failed, using best single model: {}",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        // Step 8: Compute best model statistics
        let best_model_stats = leaderboard
            .first()
            .map(|best| {
                let trial = trials
                    .iter()
                    .find(|t| t.trial_id == best.trial_id)
                    .ok_or_else(|| {
                        FerroError::invalid_input(format!(
                            "Trial {} not found in completed trials",
                            best.trial_id
                        ))
                    })?;
                Ok::<_, FerroError>(ModelStatistics {
                    mean_score: trial.cv_score,
                    std_score: trial.cv_std,
                    ci_lower: best.ci_lower,
                    ci_upper: best.ci_upper,
                    n_folds: trial.fold_scores.len(),
                    fold_scores: trial.fold_scores.clone(),
                })
            })
            .transpose()?;

        // Step 9: Compute aggregated feature importance
        let aggregated_importance = compute_aggregated_feature_importance(
            &trials,
            x.ncols(),
            maximize,
            self.config.confidence_level,
        );

        // Step 10: Compute model comparison statistics (only if statistical_tests enabled)
        let model_comparisons = if self.config.statistical_tests && successful_trials.len() >= 2 {
            compute_model_comparisons(
                &trials,
                &leaderboard,
                self.config.confidence_level,
                x.nrows(),
            )
        } else {
            None
        };

        let n_successful = trials.iter().filter(|t| t.success).count();
        let n_failed = trials.len() - n_successful;

        Ok(AutoMLResult {
            trials,
            leaderboard,
            ensemble,
            total_time_seconds: start_time.elapsed().as_secs_f64(),
            n_successful_trials: n_successful,
            n_failed_trials: n_failed,
            task: self.config.task,
            metric_name: metric_adapter.name().to_string(),
            maximize,
            data_characteristics: data_chars,
            cv_folds: self.config.cv_folds,
            best_model_stats,
            aggregated_importance,
            model_comparisons,
        })
    }

    /// Run a single trial for an algorithm configuration
    fn run_trial(
        &self,
        trial_id: usize,
        algo_config: &AlgorithmConfig,
        x: &Array2<f64>,
        y: &Array1<f64>,
        folds: &[CVFold],
        metric: &MetricAdapter,
    ) -> Result<TrialResult> {
        let trial_start = Instant::now();
        let mut fold_scores = Vec::with_capacity(folds.len());
        let n_samples = x.nrows();
        let mut oof_predictions = Array1::zeros(n_samples);

        for fold in folds {
            // Extract train/test data
            let x_train = select_rows(x, &fold.train_indices)?;
            let y_train = select_elements(y, &fold.train_indices);
            let x_test = select_rows(x, &fold.test_indices)?;
            let y_test = select_elements(y, &fold.test_indices);

            // Create and fit model
            let mut model = create_model(algo_config.algorithm)?;
            model.fit(&x_train, &y_train)?;

            // Predict
            let y_pred = model.predict(&x_test)?;

            // Store OOF predictions
            for (i, &idx) in fold.test_indices.iter().enumerate() {
                oof_predictions[idx] = y_pred[i];
            }

            // Compute fold score
            let fold_metric = metric.compute(&y_test, &y_pred)?;
            fold_scores.push(fold_metric.value);
        }

        // Compute mean and std of fold scores
        let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let std_score = if fold_scores.len() > 1 {
            let variance = fold_scores
                .iter()
                .map(|s| (s - mean_score).powi(2))
                .sum::<f64>()
                / (fold_scores.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let training_time = trial_start.elapsed().as_secs_f64();

        // Extract feature importance from model fitted on full data
        let feature_importances = {
            let mut model = create_model(algo_config.algorithm)?;
            if model.fit(x, y).is_ok() {
                model.feature_importance().map(|a| a.to_vec())
            } else {
                None
            }
        };

        let mut result = TrialResult::new(
            trial_id,
            algo_config.algorithm,
            mean_score,
            std_score,
            fold_scores,
        )
        .with_oof_predictions(oof_predictions)
        .with_training_time(training_time);

        if let Some(importances) = feature_importances {
            result = result.with_feature_importances(importances);
        }

        Ok(result)
    }
}

/// Create a model instance from algorithm type
fn create_model(algorithm: AlgorithmType) -> Result<Box<dyn Model>> {
    match algorithm {
        // Classification - Linear
        AlgorithmType::LogisticRegression => Ok(Box::new(LogisticRegression::new())),

        // Classification - Probabilistic
        AlgorithmType::GaussianNB => Ok(Box::new(GaussianNB::new())),
        AlgorithmType::MultinomialNB => Ok(Box::new(MultinomialNB::new())),
        AlgorithmType::CategoricalNB => Ok(Box::new(CategoricalNB::new())),

        // Classification - Instance-based
        AlgorithmType::KNeighborsClassifier => Ok(Box::new(KNeighborsClassifier::new(5))),

        // Classification - Trees
        AlgorithmType::DecisionTreeClassifier => Ok(Box::new(DecisionTreeClassifier::new())),
        AlgorithmType::RandomForestClassifier => Ok(Box::new(RandomForestClassifier::new())),
        AlgorithmType::GradientBoostingClassifier => {
            Ok(Box::new(GradientBoostingClassifier::new()))
        }
        AlgorithmType::HistGradientBoostingClassifier => {
            Ok(Box::new(HistGradientBoostingClassifier::new()))
        }

        // Regression - Linear
        AlgorithmType::LinearRegression => Ok(Box::new(LinearRegression::new())),
        AlgorithmType::Ridge => Ok(Box::new(RidgeRegression::new(1.0))), // Default alpha
        AlgorithmType::Lasso => Ok(Box::new(
            crate::models::regularized::LassoRegression::new(1.0), // Default alpha
        )),
        AlgorithmType::ElasticNet => Ok(Box::new(crate::models::regularized::ElasticNet::new(
            1.0, 0.5, // Default alpha and l1_ratio
        ))),

        // Regression - Instance-based
        AlgorithmType::KNeighborsRegressor => Ok(Box::new(KNeighborsRegressor::new(5))),

        // Regression - Trees
        AlgorithmType::DecisionTreeRegressor => Ok(Box::new(DecisionTreeRegressor::new())),
        AlgorithmType::RandomForestRegressor => Ok(Box::new(RandomForestRegressor::new())),
        AlgorithmType::GradientBoostingRegressor => Ok(Box::new(GradientBoostingRegressor::new())),
        AlgorithmType::HistGradientBoostingRegressor => {
            Ok(Box::new(HistGradientBoostingRegressor::new()))
        }

        // Classification - SVM
        AlgorithmType::SVC => Ok(Box::new(SVC::new())),
        AlgorithmType::LinearSVC => Ok(Box::new(LinearSVC::new())),

        // Regression - SVM
        AlgorithmType::SVR => Ok(Box::new(SVR::new())),
        AlgorithmType::LinearSVR => Ok(Box::new(LinearSVR::new())),

        // Regression - Quantile
        AlgorithmType::QuantileRegression => Ok(Box::new(QuantileRegression::new(0.5))),

        // Not yet implemented
        AlgorithmType::RobustRegression => Ok(Box::new(RobustRegression::new())),
    }
}

/// Select rows from a 2D array by indices
fn select_rows(array: &Array2<f64>, indices: &[usize]) -> Result<Array2<f64>> {
    let n_cols = array.ncols();
    let n_rows = indices.len();

    let mut data = Vec::with_capacity(n_rows * n_cols);
    for &idx in indices {
        for val in array.row(idx).iter() {
            data.push(*val);
        }
    }

    Array2::from_shape_vec((n_rows, n_cols), data)
        .map_err(|e| FerroError::invalid_input(format!("select_rows shape error: {e}")))
}

/// Select elements from a 1D array by indices
fn select_elements(array: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
    Array1::from_vec(indices.iter().map(|&idx| array[idx]).collect())
}

/// Compute t-distribution confidence interval
fn compute_t_ci(scores: &[f64], confidence_level: f64) -> (f64, f64) {
    let n = scores.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    if n == 1 {
        return (scores[0], scores[0]);
    }

    let mean = scores.iter().sum::<f64>() / n as f64;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std_error = (variance / n as f64).sqrt();

    // Approximate t-value for common confidence levels
    let t_value = match (confidence_level * 100.0) as u32 {
        95 => match n {
            2 => 12.706,
            3 => 4.303,
            4 => 3.182,
            5 => 2.776,
            6 => 2.571,
            7 => 2.447,
            8 => 2.365,
            9 => 2.306,
            10 => 2.262,
            _ => 1.96, // Approximate with z for larger n
        },
        99 => match n {
            2 => 63.657,
            3 => 9.925,
            4 => 5.841,
            5 => 4.604,
            _ => 2.576, // Approximate with z
        },
        _ => 1.96, // Default to 95% z-value
    };

    let margin = t_value * std_error;
    (mean - margin, mean + margin)
}

/// Compute aggregated feature importance across all successful models
///
/// This function collects feature importance from models that support it
/// (tree-based models) and computes a weighted average based on model scores.
fn compute_aggregated_feature_importance(
    trials: &[TrialResult],
    n_features: usize,
    maximize: bool,
    confidence_level: f64,
) -> Option<AggregatedFeatureImportance> {
    // Collect feature importances from successful trials
    let successful_trials: Vec<_> = trials.iter().filter(|t| t.success).collect();

    if successful_trials.is_empty() || n_features == 0 {
        return None;
    }

    // For now, we use a simple uniform importance since we don't have
    // per-model feature importance stored in trials. In a full implementation,
    // this would collect actual feature importance from tree-based models.
    //
    // We'll use the trial's score as a proxy weight for aggregation.
    let mut per_model_importance = Vec::new();
    let mut all_importances: Vec<Vec<f64>> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();

    for trial in &successful_trials {
        // Check if this algorithm type supports feature importance
        let supports_importance = matches!(
            trial.algorithm,
            AlgorithmType::DecisionTreeClassifier
                | AlgorithmType::DecisionTreeRegressor
                | AlgorithmType::RandomForestClassifier
                | AlgorithmType::RandomForestRegressor
                | AlgorithmType::GradientBoostingClassifier
                | AlgorithmType::GradientBoostingRegressor
                | AlgorithmType::HistGradientBoostingClassifier
                | AlgorithmType::HistGradientBoostingRegressor
                | AlgorithmType::LinearRegression
                | AlgorithmType::LogisticRegression
                | AlgorithmType::Ridge
                | AlgorithmType::Lasso
                | AlgorithmType::ElasticNet
        );

        if supports_importance {
            // Use actual feature importances if available, fall back to uniform
            let importances = if let Some(ref fi) = trial.feature_importances {
                if fi.len() == n_features {
                    fi.clone()
                } else {
                    vec![1.0 / n_features as f64; n_features]
                }
            } else {
                vec![1.0 / n_features as f64; n_features]
            };

            per_model_importance.push(ModelFeatureImportance {
                trial_id: trial.trial_id,
                algorithm: trial.algorithm,
                cv_score: trial.cv_score,
                importances: importances.clone(),
            });

            all_importances.push(importances);

            // Use normalized score as weight
            let weight = if maximize {
                trial.cv_score.max(0.0)
            } else {
                (1.0 - trial.cv_score.abs()).max(0.0)
            };
            weights.push(weight);
        }
    }

    if all_importances.is_empty() {
        return None;
    }

    // Normalize weights
    let weight_sum: f64 = weights.iter().sum();
    let weights: Vec<f64> = if weight_sum > 0.0 {
        weights.iter().map(|w| w / weight_sum).collect()
    } else {
        vec![1.0 / weights.len() as f64; weights.len()]
    };

    // Compute weighted mean importance
    let mut importance_mean = vec![0.0; n_features];
    for (importances, &weight) in all_importances.iter().zip(weights.iter()) {
        for (i, &imp) in importances.iter().enumerate() {
            importance_mean[i] += weight * imp;
        }
    }

    // Compute standard deviation across models
    let n_models = all_importances.len();
    let mut importance_std = vec![0.0; n_features];
    if n_models > 1 {
        for i in 0..n_features {
            let variance: f64 = all_importances
                .iter()
                .map(|imps| (imps[i] - importance_mean[i]).powi(2))
                .sum::<f64>()
                / (n_models - 1) as f64;
            importance_std[i] = variance.sqrt();
        }
    }

    // Compute confidence intervals
    let std_error: Vec<f64> = importance_std
        .iter()
        .map(|s| s / (n_models as f64).sqrt())
        .collect();

    let t_value = get_t_value(n_models, confidence_level);

    let ci_lower: Vec<f64> = importance_mean
        .iter()
        .zip(std_error.iter())
        .map(|(&m, &se)| t_value.mul_add(-se, m))
        .collect();

    let ci_upper: Vec<f64> = importance_mean
        .iter()
        .zip(std_error.iter())
        .map(|(&m, &se)| t_value.mul_add(se, m))
        .collect();

    // Generate feature names
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("feature_{}", i)).collect();

    Some(AggregatedFeatureImportance {
        feature_names,
        importance_mean,
        importance_std,
        ci_lower,
        ci_upper,
        n_models_per_feature: vec![n_models; n_features],
        n_models,
        per_model_importance,
    })
}

/// Get t-value for confidence interval
fn get_t_value(n: usize, confidence_level: f64) -> f64 {
    match (confidence_level * 100.0) as u32 {
        95 => match n {
            1 => 12.706,
            2 => 4.303,
            3 => 3.182,
            4 => 2.776,
            5 => 2.571,
            6 => 2.447,
            7 => 2.365,
            8 => 2.306,
            9 => 2.262,
            10 => 2.228,
            _ => 1.96,
        },
        99 => match n {
            1 => 63.657,
            2 => 9.925,
            3 => 5.841,
            4 => 4.604,
            5 => 4.032,
            _ => 2.576,
        },
        _ => 1.96,
    }
}

/// Compute statistical comparisons between top models
///
/// Uses the corrected resampled t-test (Nadeau-Bengio) for proper inference
/// on cross-validation scores.
fn compute_model_comparisons(
    trials: &[TrialResult],
    leaderboard: &[LeaderboardEntry],
    confidence_level: f64,
    n_samples: usize,
) -> Option<ModelComparisonResults> {
    // Need at least 2 models to compare
    if leaderboard.len() < 2 {
        return None;
    }

    // Get the top models to compare (up to 10)
    let top_models: Vec<_> = leaderboard.iter().take(10).collect();

    // Find corresponding trials with fold scores
    let top_trials: Vec<_> = top_models
        .iter()
        .filter_map(|entry| {
            trials
                .iter()
                .find(|t| t.trial_id == entry.trial_id && !t.fold_scores.is_empty())
        })
        .collect();

    if top_trials.len() < 2 {
        return None;
    }

    let n_folds = top_trials[0].fold_scores.len();
    if n_folds < 2 {
        return None;
    }

    // Compute train/test sizes for corrected t-test
    let n_test = n_samples / n_folds;
    let n_train = n_samples - n_test;

    let mut comparisons = Vec::new();
    let best_trial = &top_trials[0];

    // Compare best model against all others
    for other_trial in &top_trials[1..] {
        let scores1 = Array1::from_vec(best_trial.fold_scores.clone());
        let scores2 = Array1::from_vec(other_trial.fold_scores.clone());

        // Use corrected resampled t-test
        if let Ok(result) = corrected_resampled_ttest(&scores1, &scores2, n_train, n_test) {
            comparisons.push(PairwiseComparison {
                trial_id_1: best_trial.trial_id,
                trial_id_2: other_trial.trial_id,
                model1_name: format!("{:?}", best_trial.algorithm),
                model2_name: format!("{:?}", other_trial.algorithm),
                test_name: result.test_name.clone(),
                mean_difference: result.mean_difference,
                std_error: result.std_error,
                statistic: result.statistic,
                p_value: result.p_value,
                p_value_corrected: result.p_value, // Will be corrected below
                ci_lower: result.ci_95.0,
                ci_upper: result.ci_95.1,
                significant: result.significant,
                significant_corrected: false, // Will be set after correction
            });
        }
    }

    if comparisons.is_empty() {
        return None;
    }

    // Apply Holm-Bonferroni correction for multiple comparisons
    let n_comparisons = comparisons.len();
    let alpha = 1.0 - confidence_level;

    // Sort by p-value for Holm correction
    let mut sorted_indices: Vec<usize> = (0..n_comparisons).collect();
    sorted_indices.sort_by(|&a, &b| {
        comparisons[a]
            .p_value
            .partial_cmp(&comparisons[b].p_value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply Holm correction
    let mut n_significantly_worse = 0;
    for (rank, &idx) in sorted_indices.iter().enumerate() {
        let corrected_alpha = alpha / (n_comparisons - rank) as f64;
        comparisons[idx].p_value_corrected =
            (comparisons[idx].p_value * (n_comparisons - rank) as f64).min(1.0);
        comparisons[idx].significant_corrected = comparisons[idx].p_value <= corrected_alpha;

        if comparisons[idx].significant_corrected {
            n_significantly_worse += 1;
        }
    }

    // Re-sort by original order (by trial_id_2)
    comparisons.sort_by_key(|c| c.trial_id_2);

    Some(ModelComparisonResults {
        pairwise_comparisons: comparisons,
        best_is_significantly_better: n_significantly_worse > 0,
        n_significantly_worse,
        corrected_alpha: alpha,
        correction_method: "Holm-Bonferroni".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AutoMLConfig;
    use ndarray::array;

    fn create_classification_data() -> (Array2<f64>, Array1<f64>) {
        // Simple linearly separable data
        let x = Array2::from_shape_vec(
            (100, 2),
            (0..200)
                .map(|i| {
                    let row = i / 2;
                    let col = i % 2;
                    if row < 50 {
                        if col == 0 {
                            row as f64 * 0.1
                        } else {
                            row as f64 * 0.1 + 0.5
                        }
                    } else {
                        if col == 0 {
                            (row - 50) as f64 * 0.1 + 5.0
                        } else {
                            (row - 50) as f64 * 0.1 + 5.5
                        }
                    }
                })
                .collect(),
        )
        .unwrap();

        let y = Array1::from_vec((0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }).collect());

        (x, y)
    }

    fn create_regression_data() -> (Array2<f64>, Array1<f64>) {
        let x =
            Array2::from_shape_vec((100, 2), (0..200).map(|i| (i % 100) as f64 * 0.1).collect())
                .unwrap();

        let y = Array1::from_vec(
            (0..100)
                .map(|i| 2.0 * (i as f64 * 0.1) + 1.0 + (i as f64 * 0.01))
                .collect(),
        );

        (x, y)
    }

    #[test]
    fn test_create_model_classifiers() {
        let classifiers = [
            AlgorithmType::LogisticRegression,
            AlgorithmType::GaussianNB,
            AlgorithmType::CategoricalNB,
            AlgorithmType::KNeighborsClassifier,
            AlgorithmType::DecisionTreeClassifier,
            AlgorithmType::RandomForestClassifier,
        ];

        for algo in &classifiers {
            let model = create_model(*algo);
            assert!(model.is_ok(), "Failed to create model for {:?}", algo);
        }
    }

    #[test]
    fn test_create_model_regressors() {
        let regressors = [
            AlgorithmType::LinearRegression,
            AlgorithmType::Ridge,
            AlgorithmType::KNeighborsRegressor,
            AlgorithmType::DecisionTreeRegressor,
            AlgorithmType::RandomForestRegressor,
        ];

        for algo in &regressors {
            let model = create_model(*algo);
            assert!(model.is_ok(), "Failed to create model for {:?}", algo);
        }
    }

    #[test]
    fn test_automl_fit_classification() {
        let (x, y) = create_classification_data();

        let config = AutoMLConfig {
            task: Task::Classification,
            metric: Metric::Accuracy,
            time_budget_seconds: 60, // Short budget for test
            cv_folds: 3,
            statistical_tests: false,
            confidence_level: 0.95,
            seed: Some(42),
            ..Default::default()
        };

        let automl = AutoML::new(config);
        let result = automl.fit(&x, &y);

        assert!(result.is_ok(), "AutoML fit failed: {:?}", result.err());
        let result = result.unwrap();

        assert!(result.is_successful(), "No successful trials");
        assert!(!result.leaderboard.is_empty(), "Leaderboard is empty");

        // Best model should have reasonable accuracy
        let best = result.best_model().unwrap();
        assert!(
            best.cv_score > 0.5,
            "Best accuracy {} should be > 0.5",
            best.cv_score
        );
    }

    #[test]
    fn test_automl_fit_regression() {
        let (x, y) = create_regression_data();

        let config = AutoMLConfig {
            task: Task::Regression,
            metric: Metric::R2,
            time_budget_seconds: 60,
            cv_folds: 3,
            statistical_tests: false,
            confidence_level: 0.95,
            seed: Some(42),
            ..Default::default()
        };

        let automl = AutoML::new(config);
        let result = automl.fit(&x, &y);

        assert!(result.is_ok(), "AutoML fit failed: {:?}", result.err());
        let result = result.unwrap();

        assert!(result.is_successful(), "No successful trials");
        assert!(!result.leaderboard.is_empty(), "Leaderboard is empty");
    }

    #[test]
    fn test_automl_result_methods() {
        let (x, y) = create_classification_data();

        let config = AutoMLConfig {
            task: Task::Classification,
            metric: Metric::Accuracy,
            time_budget_seconds: 30,
            cv_folds: 2,
            seed: Some(42),
            ..Default::default()
        };

        let automl = AutoML::new(config);
        let result = automl.fit(&x, &y).unwrap();

        assert!(result.is_successful());
        assert!(result.n_successful_algorithms() > 0);
        assert!(result.best_model().is_some());
        assert!(result.best_model_stats.is_some());
    }

    #[test]
    fn test_compute_t_ci() {
        let scores = vec![0.8, 0.82, 0.79, 0.81, 0.80];
        let (lower, upper) = compute_t_ci(&scores, 0.95);

        let mean = 0.804;
        assert!(lower < mean, "CI lower {} should be < mean {}", lower, mean);
        assert!(upper > mean, "CI upper {} should be > mean {}", upper, mean);
        assert!(upper - lower > 0.0, "CI should have positive width");
    }

    #[test]
    fn test_metric_adapter_classification() {
        let adapter = MetricAdapter {
            metric: Metric::Accuracy,
            task: Task::Classification,
        };

        let y_true = array![0.0, 1.0, 1.0, 0.0, 1.0];
        let y_pred = array![0.0, 1.0, 0.0, 0.0, 1.0]; // 4/5 correct

        let result = adapter.compute(&y_true, &y_pred).unwrap();
        assert!((result.value - 0.8).abs() < 1e-10);
        assert_eq!(result.direction, Direction::Maximize);
    }

    #[test]
    fn test_metric_adapter_regression() {
        let adapter = MetricAdapter {
            metric: Metric::Mse,
            task: Task::Regression,
        };

        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.0, 2.9]; // Small errors

        let result = adapter.compute(&y_true, &y_pred).unwrap();
        assert!(result.value < 0.1); // MSE should be small
        assert_eq!(result.direction, Direction::Minimize);
    }

    #[test]
    fn test_leaderboard_ordering() {
        let (x, y) = create_classification_data();

        let config = AutoMLConfig {
            task: Task::Classification,
            metric: Metric::Accuracy,
            time_budget_seconds: 30,
            cv_folds: 2,
            seed: Some(42),
            ..Default::default()
        };

        let automl = AutoML::new(config);
        let result = automl.fit(&x, &y).unwrap();

        // Leaderboard should be sorted by score (descending for accuracy)
        for window in result.leaderboard.windows(2) {
            assert!(
                window[0].cv_score >= window[1].cv_score,
                "Leaderboard not sorted: {} < {}",
                window[0].cv_score,
                window[1].cv_score
            );
        }

        // Ranks should be sequential
        for (i, entry) in result.leaderboard.iter().enumerate() {
            assert_eq!(entry.rank, i + 1);
        }
    }

    #[test]
    fn test_select_rows() {
        let array =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let selected = select_rows(&array, &[0, 2]).unwrap();
        assert_eq!(selected.nrows(), 2);
        assert_eq!(selected.ncols(), 2);
        assert!((selected[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((selected[[1, 0]] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_select_elements() {
        let array = array![10.0, 20.0, 30.0, 40.0];
        let selected = select_elements(&array, &[1, 3]);

        assert_eq!(selected.len(), 2);
        assert!((selected[0] - 20.0).abs() < 1e-10);
        assert!((selected[1] - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_aggregated_feature_importance() {
        let imp = AggregatedFeatureImportance {
            feature_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            importance_mean: vec![0.1, 0.5, 0.3],
            importance_std: vec![0.01, 0.02, 0.015],
            ci_lower: vec![0.08, 0.46, 0.27],
            ci_upper: vec![0.12, 0.54, 0.33],
            n_models_per_feature: vec![3, 3, 3],
            n_models: 3,
            per_model_importance: vec![],
        };

        // Test sorted_indices
        let sorted = imp.sorted_indices();
        assert_eq!(sorted, vec![1, 2, 0]); // b > c > a

        // Test top_k
        let top2 = imp.top_k(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, "b");
        assert_eq!(top2[1].0, "c");

        // Test is_significant (all CIs are > 0)
        assert!(imp.is_significant(0));
        assert!(imp.is_significant(1));
        assert!(imp.is_significant(2));

        // Test significant_features
        assert_eq!(imp.significant_features(), vec![0, 1, 2]);
    }

    #[test]
    fn test_model_comparison_results() {
        let comparisons = ModelComparisonResults {
            pairwise_comparisons: vec![
                PairwiseComparison {
                    trial_id_1: 0,
                    trial_id_2: 1,
                    model1_name: "RandomForest".to_string(),
                    model2_name: "LogisticRegression".to_string(),
                    test_name: "Corrected t-test".to_string(),
                    mean_difference: 0.05,
                    std_error: 0.01,
                    statistic: 5.0,
                    p_value: 0.001,
                    p_value_corrected: 0.003,
                    ci_lower: 0.03,
                    ci_upper: 0.07,
                    significant: true,
                    significant_corrected: true,
                },
                PairwiseComparison {
                    trial_id_1: 0,
                    trial_id_2: 2,
                    model1_name: "RandomForest".to_string(),
                    model2_name: "GaussianNB".to_string(),
                    test_name: "Corrected t-test".to_string(),
                    mean_difference: 0.02,
                    std_error: 0.02,
                    statistic: 1.0,
                    p_value: 0.35,
                    p_value_corrected: 0.70,
                    ci_lower: -0.02,
                    ci_upper: 0.06,
                    significant: false,
                    significant_corrected: false,
                },
            ],
            best_is_significantly_better: true,
            n_significantly_worse: 1,
            corrected_alpha: 0.05,
            correction_method: "Holm-Bonferroni".to_string(),
        };

        // Test comparisons_for_model
        let for_model_0 = comparisons.comparisons_for_model(0);
        assert_eq!(for_model_0.len(), 2);

        // Test are_significantly_different
        assert!(comparisons.are_significantly_different(0, 1));
        assert!(!comparisons.are_significantly_different(0, 2));

        // Test models_competitive_with_best
        let competitive = comparisons.models_competitive_with_best();
        assert!(competitive.contains(&0)); // Best is always competitive
        assert!(competitive.contains(&2)); // Not significantly worse
        assert!(!competitive.contains(&1)); // Significantly worse
    }

    #[test]
    fn test_automl_result_with_statistical_tests() {
        let (x, y) = create_classification_data();

        let config = AutoMLConfig {
            task: Task::Classification,
            metric: Metric::Accuracy,
            time_budget_seconds: 60,
            cv_folds: 3,
            statistical_tests: true,
            confidence_level: 0.95,
            seed: Some(42),
            ..Default::default()
        };

        let automl = AutoML::new(config);
        let result = automl.fit(&x, &y).unwrap();

        // Should have aggregated importance if models support it
        if let Some(importance) = &result.aggregated_importance {
            assert_eq!(importance.feature_names.len(), 2);
            assert_eq!(importance.importance_mean.len(), 2);
        }

        // Should have model comparisons if statistical_tests is enabled
        if result.leaderboard.len() >= 2 {
            // Model comparisons may or may not be present depending on fold scores
            if let Some(comparisons) = &result.model_comparisons {
                assert!(!comparisons.pairwise_comparisons.is_empty());
                assert_eq!(comparisons.correction_method, "Holm-Bonferroni");
            }
        }

        // Test summary generation
        let summary = result.summary();
        assert!(summary.contains("AutoML Results Summary"));
        assert!(summary.contains("Best Model"));
        assert!(summary.contains("Leaderboard"));
    }

    #[test]
    fn test_automl_result_summary() {
        let (x, y) = create_classification_data();

        let config = AutoMLConfig {
            task: Task::Classification,
            metric: Metric::Accuracy,
            time_budget_seconds: 30,
            cv_folds: 2,
            seed: Some(42),
            ..Default::default()
        };

        let automl = AutoML::new(config);
        let result = automl.fit(&x, &y).unwrap();

        let summary = result.summary();
        assert!(summary.contains("Classification"));
        assert!(summary.contains("accuracy"));
        assert!(summary.contains("Best Model"));
    }

    #[test]
    fn test_competitive_models() {
        let (x, y) = create_classification_data();

        let config = AutoMLConfig {
            task: Task::Classification,
            metric: Metric::Accuracy,
            time_budget_seconds: 60,
            cv_folds: 3,
            statistical_tests: true,
            seed: Some(42),
            ..Default::default()
        };

        let automl = AutoML::new(config);
        let result = automl.fit(&x, &y).unwrap();

        // Competitive models should include the best
        let competitive = result.competitive_models();
        assert!(!competitive.is_empty());

        // First competitive model should be the best
        if let Some(best) = result.best_model() {
            assert!(competitive.iter().any(|c| c.trial_id == best.trial_id));
        }
    }

    #[test]
    fn test_top_features() {
        let (x, y) = create_classification_data();

        let config = AutoMLConfig {
            task: Task::Classification,
            metric: Metric::Accuracy,
            time_budget_seconds: 60,
            cv_folds: 3,
            seed: Some(42),
            ..Default::default()
        };

        let automl = AutoML::new(config);
        let result = automl.fit(&x, &y).unwrap();

        // Top features returns feature information
        if let Some(top) = result.top_features(5) {
            // Should have at most 2 features (input has 2)
            assert!(top.len() <= 2);

            // Each entry should have (name, mean, ci_lower, ci_upper)
            for (name, mean, lower, upper) in &top {
                assert!(!name.is_empty());
                assert!(lower <= mean);
                assert!(mean <= upper);
            }
        }
    }

    #[test]
    fn test_automl_result_predict() {
        let (x, y) = create_classification_data();

        // Split into train/test
        let n_train = 80;
        let x_train = x.slice(ndarray::s![..n_train, ..]).to_owned();
        let y_train = y.slice(ndarray::s![..n_train]).to_owned();
        let x_test = x.slice(ndarray::s![n_train.., ..]).to_owned();
        let y_test = y.slice(ndarray::s![n_train..]).to_owned();

        let config = AutoMLConfig {
            task: Task::Classification,
            metric: Metric::Accuracy,
            time_budget_seconds: 30,
            cv_folds: 2,
            seed: Some(42),
            ..Default::default()
        };

        let automl = AutoML::new(config);
        let result = automl.fit(&x_train, &y_train).unwrap();

        // Test predict method
        let predictions = result.predict(&x_train, &y_train, &x_test).unwrap();

        // Should have correct number of predictions
        assert_eq!(predictions.len(), y_test.len());

        // Predictions should be binary (0 or 1) for classification
        for &pred in predictions.iter() {
            assert!(pred == 0.0 || pred == 1.0);
        }

        // Should have reasonable accuracy
        let correct = predictions
            .iter()
            .zip(y_test.iter())
            .filter(|(p, t)| (*p - *t).abs() < 0.5)
            .count();
        let accuracy = correct as f64 / y_test.len() as f64;
        assert!(
            accuracy > 0.5,
            "Accuracy {} should be better than random",
            accuracy
        );
    }

    #[test]
    fn test_automl_result_predict_regression() {
        let (x, y) = create_regression_data();

        // Split into train/test
        let n_train = 80;
        let x_train = x.slice(ndarray::s![..n_train, ..]).to_owned();
        let y_train = y.slice(ndarray::s![..n_train]).to_owned();
        let x_test = x.slice(ndarray::s![n_train.., ..]).to_owned();

        let config = AutoMLConfig {
            task: Task::Regression,
            metric: Metric::R2,
            time_budget_seconds: 30,
            cv_folds: 2,
            seed: Some(42),
            ..Default::default()
        };

        let automl = AutoML::new(config);
        let result = automl.fit(&x_train, &y_train).unwrap();

        // Test predict method
        let predictions = result.predict(&x_train, &y_train, &x_test).unwrap();

        // Should have correct number of predictions
        assert_eq!(predictions.len(), x_test.nrows());

        // Predictions should be finite
        for &pred in predictions.iter() {
            assert!(pred.is_finite());
        }
    }
}

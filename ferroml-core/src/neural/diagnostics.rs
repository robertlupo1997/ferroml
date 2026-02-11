//! Training Diagnostics for Neural Networks
//!
//! This module provides diagnostic tools for analyzing neural network training.
//!
//! ## Features
//!
//! - Loss curve analysis with convergence detection
//! - Learning rate diagnostics (too high/low)
//! - Gradient statistics per layer
//! - Training stability assessment

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Training diagnostics collected during neural network training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDiagnostics {
    /// Loss values at each epoch
    pub loss_curve: Vec<f64>,
    /// Validation loss at each epoch (if early stopping enabled)
    pub val_loss_curve: Option<Vec<f64>>,
    /// Gradient statistics per layer per epoch
    pub gradient_stats: Vec<Vec<GradientStatistics>>,
    /// Convergence status
    pub convergence: ConvergenceStatus,
    /// Number of iterations run
    pub n_iter: usize,
    /// Best validation loss (if early stopping)
    pub best_val_loss: Option<f64>,
    /// Best iteration (if early stopping)
    pub best_iter: Option<usize>,
    /// Learning rate used
    pub learning_rate: f64,
    /// Final training loss
    pub final_loss: f64,
}

impl TrainingDiagnostics {
    /// Create new empty diagnostics
    pub fn new(learning_rate: f64) -> Self {
        Self {
            loss_curve: Vec::new(),
            val_loss_curve: None,
            gradient_stats: Vec::new(),
            convergence: ConvergenceStatus::NotStarted,
            n_iter: 0,
            best_val_loss: None,
            best_iter: None,
            learning_rate,
            final_loss: f64::INFINITY,
        }
    }

    /// Record loss for an epoch
    pub fn record_epoch(&mut self, loss: f64, val_loss: Option<f64>) {
        self.loss_curve.push(loss);
        self.n_iter += 1;
        self.final_loss = loss;

        if let Some(vl) = val_loss {
            if self.val_loss_curve.is_none() {
                self.val_loss_curve = Some(Vec::new());
            }
            self.val_loss_curve.as_mut().unwrap().push(vl);

            // Track best validation loss
            if self.best_val_loss.is_none() || vl < self.best_val_loss.unwrap() {
                self.best_val_loss = Some(vl);
                self.best_iter = Some(self.n_iter);
            }
        }
    }

    /// Record gradient statistics for an epoch
    pub fn record_gradients(&mut self, stats: Vec<GradientStatistics>) {
        self.gradient_stats.push(stats);
    }

    /// Analyze convergence status
    pub fn analyze_convergence(&mut self, tol: f64, patience: usize) {
        if self.loss_curve.len() < 2 {
            self.convergence = ConvergenceStatus::NotStarted;
            return;
        }

        // Check for NaN or Inf (divergence)
        if self
            .loss_curve
            .iter()
            .any(|&l| l.is_nan() || l.is_infinite())
        {
            self.convergence = ConvergenceStatus::Diverged {
                reason: "Loss became NaN or Inf".to_string(),
            };
            return;
        }

        // Check if loss is oscillating wildly (learning rate too high)
        if self.is_oscillating() {
            self.convergence = ConvergenceStatus::Unstable {
                reason: "Loss is oscillating - learning rate may be too high".to_string(),
            };
            return;
        }

        // Check for plateau (no improvement)
        if let Some(plateau_length) = self.detect_plateau(tol) {
            if plateau_length >= patience {
                self.convergence = ConvergenceStatus::Plateau {
                    since_iter: self.n_iter - plateau_length,
                };
                return;
            }
        }

        // Check for convergence (improvement below tolerance)
        let recent_improvement = self.recent_improvement(patience);
        if recent_improvement < tol {
            self.convergence = ConvergenceStatus::Converged {
                at_iter: self.n_iter,
            };
            return;
        }

        self.convergence = ConvergenceStatus::InProgress {
            improvement_rate: recent_improvement,
        };
    }

    /// Check if loss is oscillating (sign of learning rate too high)
    fn is_oscillating(&self) -> bool {
        if self.loss_curve.len() < 10 {
            return false;
        }

        let recent: Vec<f64> = self.loss_curve.iter().rev().take(10).cloned().collect();
        let mut sign_changes = 0;

        for i in 1..recent.len() - 1 {
            let prev_diff = recent[i] - recent[i + 1];
            let next_diff = recent[i - 1] - recent[i];
            if prev_diff * next_diff < 0.0 {
                sign_changes += 1;
            }
        }

        // If more than 60% of recent changes are oscillations
        sign_changes as f64 / (recent.len() - 2) as f64 > 0.6
    }

    /// Detect how long the loss has been on a plateau
    fn detect_plateau(&self, tol: f64) -> Option<usize> {
        if self.loss_curve.len() < 5 {
            return None;
        }

        let mut plateau_length = 0;
        let recent_loss = *self.loss_curve.last()?;

        for loss in self.loss_curve.iter().rev().skip(1) {
            if (loss - recent_loss).abs() < tol {
                plateau_length += 1;
            } else {
                break;
            }
        }

        if plateau_length > 0 {
            Some(plateau_length)
        } else {
            None
        }
    }

    /// Calculate recent improvement rate
    fn recent_improvement(&self, window: usize) -> f64 {
        if self.loss_curve.len() < window + 1 {
            return f64::INFINITY;
        }

        let recent: Vec<f64> = self
            .loss_curve
            .iter()
            .rev()
            .take(window + 1)
            .cloned()
            .collect();
        let oldest = recent[recent.len() - 1];
        let newest = recent[0];

        (oldest - newest).abs() / oldest.abs().max(1e-10)
    }

    /// Get a summary of training diagnostics
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Training Diagnostics Summary:\n"));
        s.push_str(&format!("  Iterations: {}\n", self.n_iter));
        s.push_str(&format!("  Final loss: {:.6}\n", self.final_loss));
        s.push_str(&format!("  Learning rate: {:.6}\n", self.learning_rate));
        s.push_str(&format!("  Convergence: {:?}\n", self.convergence));

        if let Some(ref _val_curve) = self.val_loss_curve {
            if let Some(best) = self.best_val_loss {
                s.push_str(&format!(
                    "  Best val loss: {:.6} at iter {}\n",
                    best,
                    self.best_iter.unwrap_or(0)
                ));
            }
        }

        // Gradient diagnostics
        if !self.gradient_stats.is_empty() {
            if let Some(last_stats) = self.gradient_stats.last() {
                s.push_str("  Gradient stats (final epoch):\n");
                for (i, stat) in last_stats.iter().enumerate() {
                    s.push_str(&format!(
                        "    Layer {}: mean={:.2e}, std={:.2e}, max={:.2e}\n",
                        i, stat.mean, stat.std, stat.max_abs
                    ));
                }
            }
        }

        s
    }

    /// Check for vanishing gradients
    pub fn has_vanishing_gradients(&self) -> bool {
        if let Some(last_stats) = self.gradient_stats.last() {
            // Check if any layer has very small gradients
            last_stats
                .iter()
                .any(|s| s.mean.abs() < 1e-7 && s.std < 1e-7)
        } else {
            false
        }
    }

    /// Check for exploding gradients
    pub fn has_exploding_gradients(&self) -> bool {
        if let Some(last_stats) = self.gradient_stats.last() {
            last_stats
                .iter()
                .any(|s| s.max_abs > 1e3 || s.mean.abs() > 100.0)
        } else {
            false
        }
    }
}

/// Convergence status of training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    /// Training has not started
    NotStarted,
    /// Training is in progress
    InProgress { improvement_rate: f64 },
    /// Training converged successfully
    Converged { at_iter: usize },
    /// Loss is on a plateau (no improvement)
    Plateau { since_iter: usize },
    /// Training diverged (loss became NaN/Inf)
    Diverged { reason: String },
    /// Training is unstable (oscillating loss)
    Unstable { reason: String },
}

impl ConvergenceStatus {
    /// Check if training has converged
    pub fn is_converged(&self) -> bool {
        matches!(self, ConvergenceStatus::Converged { .. })
    }

    /// Check if training should stop
    pub fn should_stop(&self) -> bool {
        matches!(
            self,
            ConvergenceStatus::Converged { .. }
                | ConvergenceStatus::Diverged { .. }
                | ConvergenceStatus::Plateau { .. }
        )
    }
}

/// Gradient statistics for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientStatistics {
    /// Layer index
    pub layer_idx: usize,
    /// Mean gradient value
    pub mean: f64,
    /// Standard deviation of gradients
    pub std: f64,
    /// Maximum absolute gradient
    pub max_abs: f64,
    /// Minimum absolute gradient
    pub min_abs: f64,
    /// Percentage of near-zero gradients
    pub sparsity: f64,
}

impl GradientStatistics {
    /// Compute gradient statistics from gradient values
    pub fn from_gradients(layer_idx: usize, gradients: &Array1<f64>) -> Self {
        let n = gradients.len() as f64;
        let mean = gradients.sum() / n;
        let variance = gradients.mapv(|g| (g - mean).powi(2)).sum() / n;
        let std = variance.sqrt();
        let max_abs = gradients.iter().map(|g| g.abs()).fold(0.0, f64::max);
        let min_abs = gradients
            .iter()
            .map(|g| g.abs())
            .fold(f64::INFINITY, f64::min);
        let near_zero = gradients.iter().filter(|&&g| g.abs() < 1e-8).count() as f64;
        let sparsity = near_zero / n;

        Self {
            layer_idx,
            mean,
            std,
            max_abs,
            min_abs,
            sparsity,
        }
    }
}

/// Analyze learning rate based on loss curve
pub fn analyze_learning_rate(loss_curve: &[f64]) -> LearningRateAnalysis {
    if loss_curve.len() < 5 {
        return LearningRateAnalysis::InsufficientData;
    }

    // Check for divergence
    if loss_curve.iter().any(|&l| l.is_nan() || l.is_infinite()) {
        return LearningRateAnalysis::TooHigh {
            reason: "Loss diverged to NaN/Inf".to_string(),
        };
    }

    // Check if loss is increasing
    let start_loss = loss_curve[0];
    let end_loss = *loss_curve.last().unwrap();
    if end_loss > start_loss * 1.5 {
        return LearningRateAnalysis::TooHigh {
            reason: "Loss increased significantly".to_string(),
        };
    }

    // Check for oscillation
    let mut oscillations = 0;
    for i in 2..loss_curve.len() {
        let prev = loss_curve[i - 1] - loss_curve[i - 2];
        let curr = loss_curve[i] - loss_curve[i - 1];
        if prev * curr < 0.0 {
            oscillations += 1;
        }
    }
    let oscillation_rate = oscillations as f64 / (loss_curve.len() - 2) as f64;
    if oscillation_rate > 0.5 {
        return LearningRateAnalysis::TooHigh {
            reason: format!(
                "Loss oscillating ({:.0}% of iterations)",
                oscillation_rate * 100.0
            ),
        };
    }

    // Check for very slow progress
    let improvement_rate = (start_loss - end_loss) / start_loss;
    if improvement_rate < 0.01 && loss_curve.len() > 50 {
        return LearningRateAnalysis::TooLow {
            reason: "Very slow improvement".to_string(),
        };
    }

    LearningRateAnalysis::Good
}

/// Learning rate analysis result
#[derive(Debug, Clone)]
pub enum LearningRateAnalysis {
    /// Not enough data to analyze
    InsufficientData,
    /// Learning rate appears too high
    TooHigh { reason: String },
    /// Learning rate appears too low
    TooLow { reason: String },
    /// Learning rate appears appropriate
    Good,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostics_creation() {
        let diag = TrainingDiagnostics::new(0.001);
        assert!(diag.loss_curve.is_empty());
        assert_eq!(diag.learning_rate, 0.001);
    }

    #[test]
    fn test_record_epoch() {
        let mut diag = TrainingDiagnostics::new(0.001);
        diag.record_epoch(1.0, Some(1.1));
        diag.record_epoch(0.8, Some(0.9));

        assert_eq!(diag.loss_curve.len(), 2);
        assert_eq!(diag.val_loss_curve.as_ref().unwrap().len(), 2);
        assert_eq!(diag.best_val_loss, Some(0.9));
        assert_eq!(diag.best_iter, Some(2));
    }

    #[test]
    fn test_convergence_detection() {
        let mut diag = TrainingDiagnostics::new(0.001);

        // Simulate converging loss
        for i in 0..20 {
            let loss = 1.0 / (i as f64 + 1.0);
            diag.record_epoch(loss, None);
        }

        diag.analyze_convergence(0.01, 5);
        // Should be converging or converged
        assert!(!matches!(
            diag.convergence,
            ConvergenceStatus::Diverged { .. }
        ));
    }

    #[test]
    fn test_divergence_detection() {
        let mut diag = TrainingDiagnostics::new(0.001);
        diag.record_epoch(1.0, None);
        diag.record_epoch(f64::NAN, None);

        diag.analyze_convergence(0.01, 5);
        assert!(matches!(
            diag.convergence,
            ConvergenceStatus::Diverged { .. }
        ));
    }

    #[test]
    fn test_gradient_statistics() {
        let gradients = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.1, 0.0]);
        let stats = GradientStatistics::from_gradients(0, &gradients);

        assert_eq!(stats.layer_idx, 0);
        assert!(stats.max_abs > 0.0);
        assert!(stats.sparsity >= 0.0 && stats.sparsity <= 1.0);
    }

    #[test]
    fn test_learning_rate_analysis_good() {
        // Steadily decreasing loss
        let loss_curve: Vec<f64> = (0..50).map(|i| 1.0 / (i as f64 + 1.0)).collect();
        let analysis = analyze_learning_rate(&loss_curve);
        assert!(matches!(analysis, LearningRateAnalysis::Good));
    }

    #[test]
    fn test_learning_rate_analysis_too_high() {
        // Diverging loss
        let loss_curve = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        let analysis = analyze_learning_rate(&loss_curve);
        assert!(matches!(analysis, LearningRateAnalysis::TooHigh { .. }));
    }

    #[test]
    fn test_vanishing_gradients() {
        let mut diag = TrainingDiagnostics::new(0.001);

        let tiny_grads = vec![GradientStatistics {
            layer_idx: 0,
            mean: 1e-10,
            std: 1e-11,
            max_abs: 1e-9,
            min_abs: 1e-12,
            sparsity: 0.99,
        }];
        diag.record_gradients(tiny_grads);

        assert!(diag.has_vanishing_gradients());
    }

    #[test]
    fn test_exploding_gradients() {
        let mut diag = TrainingDiagnostics::new(0.001);

        let huge_grads = vec![GradientStatistics {
            layer_idx: 0,
            mean: 1000.0,
            std: 500.0,
            max_abs: 5000.0,
            min_abs: 100.0,
            sparsity: 0.0,
        }];
        diag.record_gradients(huge_grads);

        assert!(diag.has_exploding_gradients());
    }
}

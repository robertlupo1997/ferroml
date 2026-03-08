//! Extended model traits for specific model families
//!
//! These traits extend the base `Model` trait for specific model types
//! that have additional capabilities (coefficients, incremental learning, etc.)

use crate::Result;
use ndarray::{Array1, Array2};

/// Trait for models with linear coefficients (LinearRegression, Ridge, Lasso, etc.)
pub trait LinearModel: super::Model {
    /// Get the fitted coefficients (weights)
    fn coefficients(&self) -> Option<&Array1<f64>>;

    /// Get the fitted intercept (bias term)
    fn intercept(&self) -> Option<f64>;

    /// Get coefficient standard errors (if available)
    fn coefficient_std_errors(&self) -> Option<&Array1<f64>> {
        None
    }

    /// Get coefficient confidence intervals at given confidence level
    fn coefficient_intervals(&self, _confidence: f64) -> Option<Array2<f64>> {
        None
    }
}

/// Trait for models that support incremental/online learning
pub trait IncrementalModel: super::Model {
    /// Partially fit the model on a batch of data
    fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>;

    /// For classifiers: specify all possible classes upfront
    fn partial_fit_with_classes(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        _classes: Option<&[f64]>,
    ) -> Result<()> {
        self.partial_fit(x, y)
    }
}

/// Trait for models that support sample weights
pub trait WeightedModel: super::Model {
    /// Fit with per-sample weights
    fn fit_weighted(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: &Array1<f64>,
    ) -> Result<()>;
}

/// Trait for models that support sparse input (requires sparse feature)
#[cfg(feature = "sparse")]
pub trait SparseModel: super::Model {
    /// Fit on sparse CSR matrix
    fn fit_sparse(&mut self, x: &sprs::CsMat<f64>, y: &Array1<f64>) -> Result<()>;

    /// Predict from sparse CSR matrix
    fn predict_sparse(&self, x: &sprs::CsMat<f64>) -> Result<Array1<f64>>;
}

/// Trait for tree-based models with feature importance
pub trait TreeModel: super::Model {
    /// Get feature importances (Gini or permutation-based)
    fn feature_importances(&self) -> Option<&Array1<f64>>;

    /// Get number of trees (for ensembles)
    fn n_estimators(&self) -> usize {
        1
    }

    /// Get tree depth statistics
    fn tree_depths(&self) -> Option<Vec<usize>> {
        None
    }
}

/// Trait for anomaly detection models (IsolationForest, LocalOutlierFactor, etc.)
///
/// Sign conventions match sklearn:
/// - `score_samples()`: lower values = more anomalous
/// - `decision_function()`: negative = outlier, positive = inlier
/// - `predict()`: returns +1 (inlier) or -1 (outlier)
pub trait OutlierDetector: Send + Sync {
    /// Fit the model on unlabeled data.
    fn fit_unsupervised(&mut self, x: &Array2<f64>) -> Result<()>;

    /// Predict inlier (+1) or outlier (-1) labels.
    fn predict_outliers(&self, x: &Array2<f64>) -> Result<Array1<i32>>;

    /// Fit and predict in one step (required for LOF in non-novelty mode).
    fn fit_predict_outliers(&mut self, x: &Array2<f64>) -> Result<Array1<i32>> {
        self.fit_unsupervised(x)?;
        self.predict_outliers(x)
    }

    /// Raw anomaly scores. Lower = more anomalous.
    fn score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>>;

    /// `score_samples() - offset`. Negative = outlier.
    fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>>;

    /// Whether the model has been fitted.
    fn is_fitted(&self) -> bool;

    /// The threshold offset used for binary classification.
    fn offset(&self) -> f64;
}

/// Trait for ensemble models with warm start capability
pub trait WarmStartModel: super::Model {
    /// Enable/disable warm start
    fn set_warm_start(&mut self, warm_start: bool);

    /// Check if warm start is enabled
    fn warm_start(&self) -> bool;

    /// Get number of estimators currently fitted
    fn n_estimators_fitted(&self) -> usize;
}

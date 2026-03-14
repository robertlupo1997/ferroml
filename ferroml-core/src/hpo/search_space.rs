//! Search space definition for hyperparameter optimization

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Search space containing all hyperparameters to optimize
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchSpace {
    /// Named parameters in the search space
    pub parameters: HashMap<String, Parameter>,
}

impl SearchSpace {
    /// Create empty search space
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter
    pub fn add(mut self, name: impl Into<String>, param: Parameter) -> Self {
        self.parameters.insert(name.into(), param);
        self
    }

    /// Add an integer parameter
    pub fn int(self, name: impl Into<String>, low: i64, high: i64) -> Self {
        self.add(name, Parameter::int(low, high))
    }

    /// Add a log-scale integer parameter
    pub fn int_log(self, name: impl Into<String>, low: i64, high: i64) -> Self {
        self.add(name, Parameter::int_log(low, high))
    }

    /// Add a float parameter
    pub fn float(self, name: impl Into<String>, low: f64, high: f64) -> Self {
        self.add(name, Parameter::float(low, high))
    }

    /// Add a log-scale float parameter
    pub fn float_log(self, name: impl Into<String>, low: f64, high: f64) -> Self {
        self.add(name, Parameter::float_log(low, high))
    }

    /// Add a categorical parameter
    pub fn categorical(self, name: impl Into<String>, choices: Vec<String>) -> Self {
        self.add(name, Parameter::categorical(choices))
    }

    /// Add a boolean parameter
    pub fn bool(self, name: impl Into<String>) -> Self {
        self.add(name, Parameter::bool())
    }

    /// Get number of dimensions
    pub fn n_dims(&self) -> usize {
        self.parameters.len()
    }
}

/// A single hyperparameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    /// Parameter type and bounds
    pub param_type: ParameterType,
    /// Whether to use log scale
    pub log_scale: bool,
    /// Default value (optional)
    pub default: Option<ParameterDefault>,
}

/// Parameter type with bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    /// Integer parameter with [low, high] bounds
    Int {
        /// Lower bound (inclusive)
        low: i64,
        /// Upper bound (inclusive)
        high: i64,
    },
    /// Float parameter with [low, high] bounds
    Float {
        /// Lower bound (inclusive)
        low: f64,
        /// Upper bound (inclusive)
        high: f64,
    },
    /// Categorical parameter with choices
    Categorical {
        /// Available choices
        choices: Vec<String>,
    },
    /// Boolean parameter
    Bool,
}

/// Default value for a parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterDefault {
    /// Integer default value
    Int(i64),
    /// Float default value
    Float(f64),
    /// Categorical default value
    Categorical(String),
    /// Boolean default value
    Bool(bool),
}

impl Parameter {
    /// Create integer parameter
    pub fn int(low: i64, high: i64) -> Self {
        assert!(
            low <= high,
            "search space: low ({}) must be <= high ({})",
            low,
            high
        );
        Self {
            param_type: ParameterType::Int { low, high },
            log_scale: false,
            default: None,
        }
    }

    /// Create log-scale integer parameter
    pub fn int_log(low: i64, high: i64) -> Self {
        assert!(
            low <= high,
            "search space: low ({}) must be <= high ({})",
            low,
            high
        );
        assert!(low > 0 && high > 0, "log-scale bounds must be > 0");
        Self {
            param_type: ParameterType::Int { low, high },
            log_scale: true,
            default: None,
        }
    }

    /// Create float parameter
    pub fn float(low: f64, high: f64) -> Self {
        assert!(
            low <= high,
            "search space: low ({}) must be <= high ({})",
            low,
            high
        );
        Self {
            param_type: ParameterType::Float { low, high },
            log_scale: false,
            default: None,
        }
    }

    /// Create log-scale float parameter
    pub fn float_log(low: f64, high: f64) -> Self {
        assert!(
            low <= high,
            "search space: low ({}) must be <= high ({})",
            low,
            high
        );
        assert!(low > 0.0 && high > 0.0, "log-scale bounds must be > 0");
        Self {
            param_type: ParameterType::Float { low, high },
            log_scale: true,
            default: None,
        }
    }

    /// Create categorical parameter
    pub fn categorical(choices: Vec<String>) -> Self {
        Self {
            param_type: ParameterType::Categorical { choices },
            log_scale: false,
            default: None,
        }
    }

    /// Create boolean parameter
    pub fn bool() -> Self {
        Self {
            param_type: ParameterType::Bool,
            log_scale: false,
            default: None,
        }
    }

    /// Set default value
    pub fn with_default(mut self, default: ParameterDefault) -> Self {
        self.default = Some(default);
        self
    }
}

/// Common search spaces for popular models
pub mod presets {
    use super::*;

    /// Search space for Random Forest
    pub fn random_forest() -> SearchSpace {
        SearchSpace::new()
            .int("n_estimators", 10, 1000)
            .int("max_depth", 1, 50)
            .int("min_samples_split", 2, 100)
            .int("min_samples_leaf", 1, 50)
            .categorical(
                "max_features",
                vec!["sqrt".into(), "log2".into(), "none".into()],
            )
            .bool("bootstrap")
    }

    /// Search space for Gradient Boosting
    pub fn gradient_boosting() -> SearchSpace {
        SearchSpace::new()
            .int("n_estimators", 10, 1000)
            .float_log("learning_rate", 1e-4, 1.0)
            .int("max_depth", 1, 15)
            .float("subsample", 0.5, 1.0)
            .float("colsample_bytree", 0.5, 1.0)
            .float_log("min_child_weight", 1e-3, 100.0)
            .float_log("reg_alpha", 1e-8, 10.0)
            .float_log("reg_lambda", 1e-8, 10.0)
    }

    /// Search space for Logistic Regression
    pub fn logistic_regression() -> SearchSpace {
        SearchSpace::new()
            .float_log("C", 1e-4, 100.0)
            .categorical(
                "penalty",
                vec!["l1".into(), "l2".into(), "elasticnet".into()],
            )
            .float("l1_ratio", 0.0, 1.0)
            .int("max_iter", 100, 1000)
    }

    /// Search space for SVM
    pub fn svm() -> SearchSpace {
        SearchSpace::new()
            .float_log("C", 1e-4, 100.0)
            .categorical(
                "kernel",
                vec![
                    "linear".into(),
                    "poly".into(),
                    "rbf".into(),
                    "sigmoid".into(),
                ],
            )
            .int("degree", 2, 5)
            .float_log("gamma", 1e-4, 10.0)
    }

    /// Search space for Neural Network
    pub fn neural_network() -> SearchSpace {
        SearchSpace::new()
            .int("n_layers", 1, 5)
            .int("n_units", 16, 512)
            .float_log("learning_rate", 1e-5, 1e-1)
            .float("dropout", 0.0, 0.5)
            .int("batch_size", 16, 256)
            .categorical(
                "activation",
                vec!["relu".into(), "tanh".into(), "selu".into()],
            )
            .categorical(
                "optimizer",
                vec!["adam".into(), "sgd".into(), "adamw".into()],
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_scale_positive_bounds() {
        // Valid log-scale params should not panic
        let _ = Parameter::float_log(1e-5, 1e-1);
        let _ = Parameter::int_log(1, 100);
    }

    #[test]
    #[should_panic(expected = "log-scale bounds must be > 0")]
    fn test_float_log_negative_low_panics() {
        let _ = Parameter::float_log(-1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "log-scale bounds must be > 0")]
    fn test_float_log_zero_low_panics() {
        let _ = Parameter::float_log(0.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "log-scale bounds must be > 0")]
    fn test_int_log_zero_low_panics() {
        let _ = Parameter::int_log(0, 100);
    }

    #[test]
    #[should_panic(expected = "search space: low (10) must be <= high (1)")]
    fn test_int_low_gt_high_panics() {
        let _ = Parameter::int(10, 1);
    }

    #[test]
    #[should_panic(expected = "search space: low (5) must be <= high (2)")]
    fn test_int_log_low_gt_high_panics() {
        let _ = Parameter::int_log(5, 2);
    }

    #[test]
    #[should_panic(expected = "must be <= high")]
    fn test_float_low_gt_high_panics() {
        let _ = Parameter::float(1.0, 0.5);
    }

    #[test]
    #[should_panic(expected = "must be <= high")]
    fn test_float_log_low_gt_high_panics() {
        let _ = Parameter::float_log(10.0, 1.0);
    }

    #[test]
    fn test_equal_bounds_ok() {
        // low == high should be allowed (single-value parameter)
        let _ = Parameter::int(5, 5);
        let _ = Parameter::float(1.0, 1.0);
        let _ = Parameter::int_log(3, 3);
        let _ = Parameter::float_log(2.0, 2.0);
    }
}

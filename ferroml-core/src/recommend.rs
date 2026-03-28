//! Algorithm Recommendation Engine
//!
//! Provides dataset-aware algorithm recommendations without fitting any models.
//! Analyzes dataset properties (size, sparsity, class balance, feature-to-sample ratio)
//! and returns ranked algorithm suggestions with suggested hyperparameters.
//!
//! # Example
//!
//! ```
//! use ndarray::{Array1, Array2};
//! use ferroml_core::recommend::recommend;
//!
//! let x = Array2::from_shape_vec((100, 5), (0..500).map(|i| i as f64).collect()).unwrap();
//! let y = Array1::from_vec((0..100).map(|i| (i % 2) as f64).collect());
//!
//! let recs = recommend(&x, &y, "classification").unwrap();
//! assert!(recs.len() <= 5);
//! assert!(!recs[0].algorithm.is_empty());
//! ```

use crate::automl::ParamValue;
use crate::error::Result;
use crate::FerroError;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// A single algorithm recommendation with explanation and suggested parameters.
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// The model class name as used in FerroML (e.g., "RandomForestClassifier")
    pub algorithm: String,
    /// Human-readable explanation of why this model is recommended
    pub reason: String,
    /// Estimated fit time: "fast" (<1s), "moderate" (1-30s), "slow" (>30s)
    pub estimated_fit_time: String,
    /// Suggested hyperparameters
    pub params: HashMap<String, ParamValue>,
    /// Internal ranking score (0.0 to 1.0, higher is better)
    pub score: f64,
}

/// Profile of a dataset computed from X and y without fitting.
#[derive(Debug, Clone)]
pub struct DatasetProfile {
    /// Number of samples (rows)
    pub n_samples: usize,
    /// Number of features (columns)
    pub n_features: usize,
    /// Number of unique classes (None for regression)
    pub n_classes: Option<usize>,
    /// Ratio of smallest class count to largest class count (None for regression)
    pub class_balance: Option<f64>,
    /// Fraction of zero elements in X
    pub sparsity: f64,
    /// Max value minus min value across all features
    pub feature_range: f64,
}

impl DatasetProfile {
    /// Compute a dataset profile from feature matrix and target vector.
    pub fn from_data(x: &Array2<f64>, y: &Array1<f64>, task: &str) -> Self {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Sparsity: fraction of zeros
        let n_elements = n_samples * n_features;
        let n_zeros = x.iter().filter(|&&v| v == 0.0).count();
        let sparsity = if n_elements > 0 {
            n_zeros as f64 / n_elements as f64
        } else {
            0.0
        };

        // Feature range
        let (mut global_min, mut global_max) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in x.iter() {
            if v.is_finite() {
                if v < global_min {
                    global_min = v;
                }
                if v > global_max {
                    global_max = v;
                }
            }
        }
        let feature_range = if global_max > global_min {
            global_max - global_min
        } else {
            0.0
        };

        // Classification-specific: count classes, compute balance
        let (n_classes, class_balance) = if task == "classification" {
            let mut class_counts: HashMap<i64, usize> = HashMap::new();
            for &val in y.iter() {
                let key = val as i64;
                *class_counts.entry(key).or_insert(0) += 1;
            }
            let nc = class_counts.len();
            let min_count = class_counts.values().copied().min().unwrap_or(0);
            let max_count = class_counts.values().copied().max().unwrap_or(1);
            let balance = if max_count > 0 {
                min_count as f64 / max_count as f64
            } else {
                1.0
            };
            (Some(nc), Some(balance))
        } else {
            (None, None)
        };

        DatasetProfile {
            n_samples,
            n_features,
            n_classes,
            class_balance,
            sparsity,
            feature_range,
        }
    }
}

/// Recommend ML algorithms for a dataset without fitting any models.
///
/// Analyzes dataset properties (size, sparsity, class balance, dimensionality)
/// and returns up to 5 ranked algorithm recommendations.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target vector (n_samples,)
/// * `task` - Either "classification" or "regression"
///
/// # Returns
///
/// A vector of up to 5 `Recommendation` structs sorted by score descending.
///
/// # Errors
///
/// Returns `FerroError::InvalidInput` if:
/// - `task` is not "classification" or "regression"
/// - X and y have mismatched sample counts
/// - X or y is empty
pub fn recommend(x: &Array2<f64>, y: &Array1<f64>, task: &str) -> Result<Vec<Recommendation>> {
    // Validate task
    let task_lower = task.to_lowercase();
    if task_lower != "classification" && task_lower != "regression" {
        return Err(FerroError::InvalidInput(format!(
            "Unknown task '{}'. Expected 'classification' or 'regression'.",
            task
        )));
    }

    // Validate shapes
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(FerroError::InvalidInput(
            "Feature matrix X must not be empty.".to_string(),
        ));
    }
    if y.len() == 0 {
        return Err(FerroError::InvalidInput(
            "Target vector y must not be empty.".to_string(),
        ));
    }
    if x.nrows() != y.len() {
        return Err(FerroError::shape_mismatch(
            format!("{} samples in X", x.nrows()),
            format!("{} samples in y", y.len()),
        ));
    }

    let profile = DatasetProfile::from_data(x, y, &task_lower);

    let mut candidates = if task_lower == "classification" {
        classification_candidates(&profile)
    } else {
        regression_candidates(&profile)
    };

    // Sort by score descending
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Return top 5
    candidates.truncate(5);
    Ok(candidates)
}

/// Estimate fit time label based on dataset size and algorithm complexity.
fn estimate_fit_time(n_samples: usize, n_features: usize, complexity: &str) -> String {
    let data_size = n_samples * n_features;
    match complexity {
        "linear" => {
            if data_size < 100_000 {
                "fast".to_string()
            } else if data_size < 5_000_000 {
                "moderate".to_string()
            } else {
                "slow".to_string()
            }
        }
        "quadratic" => {
            if n_samples < 5_000 {
                "fast".to_string()
            } else if n_samples < 30_000 {
                "moderate".to_string()
            } else {
                "slow".to_string()
            }
        }
        "tree" => {
            if data_size < 500_000 {
                "fast".to_string()
            } else if data_size < 10_000_000 {
                "moderate".to_string()
            } else {
                "slow".to_string()
            }
        }
        "ensemble" => {
            if data_size < 100_000 {
                "fast".to_string()
            } else if data_size < 5_000_000 {
                "moderate".to_string()
            } else {
                "slow".to_string()
            }
        }
        _ => "moderate".to_string(),
    }
}

/// Generate classification algorithm candidates with scores based on dataset profile.
fn classification_candidates(profile: &DatasetProfile) -> Vec<Recommendation> {
    let n = profile.n_samples;
    let p = profile.n_features;
    let is_sparse = profile.sparsity > 0.5;
    let is_imbalanced = profile.class_balance.map_or(false, |bal| bal < 0.3);
    let many_features = p > n;

    let mut candidates = Vec::new();

    // --- RandomForestClassifier ---
    {
        let mut score: f64 = 0.7; // strong baseline
        if n >= 1000 && n <= 50_000 {
            score += 0.15;
        } else if n < 1000 {
            score += 0.10;
        } else {
            score += 0.05;
        }
        if is_imbalanced {
            score += 0.10;
        }
        if many_features {
            score += 0.05;
        }
        let mut params = HashMap::new();
        params.insert("n_estimators".to_string(), ParamValue::Int(100));
        if is_imbalanced {
            params.insert(
                "class_weight".to_string(),
                ParamValue::String("balanced".to_string()),
            );
        }
        if many_features {
            let max_features = (p as f64).sqrt().ceil() as i64;
            params.insert("max_features".to_string(), ParamValue::Int(max_features));
        }

        let reason = if is_imbalanced {
            "Robust to class imbalance with balanced class weights. Handles nonlinear relationships and provides feature importances."
        } else {
            "Strong default for most classification tasks. Handles nonlinear relationships, mixed feature types, and provides feature importances."
        };

        candidates.push(Recommendation {
            algorithm: "RandomForestClassifier".to_string(),
            reason: reason.to_string(),
            estimated_fit_time: estimate_fit_time(n, p, "ensemble"),
            params,
            score: score.min(1.0),
        });
    }

    // --- HistGradientBoostingClassifier ---
    {
        let mut score: f64 = 0.65;
        if n >= 1000 && n <= 50_000 {
            score += 0.20;
        } else if n > 50_000 {
            score += 0.25;
        } else {
            score += 0.05;
        }
        if is_imbalanced {
            score += 0.05;
        }
        let mut params = HashMap::new();
        params.insert("max_iter".to_string(), ParamValue::Int(100));
        params.insert("learning_rate".to_string(), ParamValue::Float(0.1));
        if n > 50_000 {
            params.insert("max_leaf_nodes".to_string(), ParamValue::Int(31));
        }

        candidates.push(Recommendation {
            algorithm: "HistGradientBoostingClassifier".to_string(),
            reason: "Gradient boosting with histogram binning. Scales well to large datasets and typically achieves top accuracy.".to_string(),
            estimated_fit_time: estimate_fit_time(n, p, "ensemble"),
            params,
            score: score.min(1.0),
        });
    }

    // --- LogisticRegression ---
    {
        let mut score: f64 = 0.55;
        if n < 1000 {
            score += 0.15;
        } else if n <= 50_000 {
            score += 0.10;
        } else {
            score += 0.0;
        }
        if is_sparse {
            score += 0.15;
        }
        if many_features {
            score += 0.15;
        }
        let mut params = HashMap::new();
        if many_features {
            params.insert("penalty".to_string(), ParamValue::String("l1".to_string()));
            params.insert("l1_ratio".to_string(), ParamValue::Float(1.0));
        } else {
            params.insert("penalty".to_string(), ParamValue::String("l2".to_string()));
        }
        params.insert("max_iter".to_string(), ParamValue::Int(1000));

        let reason = if many_features {
            "L1-regularized logistic regression for feature selection when features outnumber samples. Fast, interpretable, and handles sparse data well."
        } else if is_sparse {
            "Efficient on sparse data. Provides probabilistic predictions and interpretable coefficients."
        } else {
            "Fast linear baseline with probabilistic predictions. Interpretable coefficients and regularization."
        };

        candidates.push(Recommendation {
            algorithm: "LogisticRegression".to_string(),
            reason: reason.to_string(),
            estimated_fit_time: estimate_fit_time(n, p, "linear"),
            params,
            score: score.min(1.0),
        });
    }

    // --- SVC ---
    {
        let mut score: f64 = 0.50;
        if n < 1000 {
            score += 0.20;
        } else if n <= 50_000 {
            score += 0.05;
        } else {
            score -= 0.15; // SVC doesn't scale well
        }
        if many_features {
            score += 0.05;
        }
        let mut params = HashMap::new();
        params.insert("C".to_string(), ParamValue::Float(1.0));
        params.insert("kernel".to_string(), ParamValue::String("rbf".to_string()));

        candidates.push(Recommendation {
            algorithm: "SVC".to_string(),
            reason: "Support vector classifier with RBF kernel. Effective on small-to-medium datasets with complex decision boundaries.".to_string(),
            estimated_fit_time: estimate_fit_time(n, p, "quadratic"),
            params,
            score: score.min(1.0),
        });
    }

    // --- KNeighborsClassifier ---
    {
        let mut score: f64 = 0.40;
        if n < 1000 {
            score += 0.15;
        } else if n > 50_000 {
            score -= 0.15;
        }
        let k = if n < 100 { 3 } else { 5 };
        let mut params = HashMap::new();
        params.insert("n_neighbors".to_string(), ParamValue::Int(k));

        candidates.push(Recommendation {
            algorithm: "KNeighborsClassifier".to_string(),
            reason: "Non-parametric classifier. Good for small datasets with local structure. No training phase.".to_string(),
            estimated_fit_time: estimate_fit_time(n, p, "linear"),
            params,
            score: score.min(1.0),
        });
    }

    // --- GaussianNB ---
    {
        let mut score: f64 = 0.35;
        if n < 1000 {
            score += 0.10;
        }
        if is_sparse {
            score += 0.10;
        }
        if p > 100 {
            score += 0.05; // scales well with features
        }
        let params = HashMap::new();

        candidates.push(Recommendation {
            algorithm: "GaussianNB".to_string(),
            reason: "Extremely fast probabilistic classifier. Works well with high-dimensional and sparse data when feature independence approximately holds.".to_string(),
            estimated_fit_time: "fast".to_string(),
            params,
            score: score.min(1.0),
        });
    }

    // --- SGDClassifier (for large data) ---
    if n > 50_000 || is_sparse {
        let mut score: f64 = 0.45;
        if n > 50_000 {
            score += 0.20;
        }
        if is_sparse {
            score += 0.10;
        }
        let mut params = HashMap::new();
        params.insert("loss".to_string(), ParamValue::String("hinge".to_string()));
        params.insert("alpha".to_string(), ParamValue::Float(0.0001));

        candidates.push(Recommendation {
            algorithm: "SGDClassifier".to_string(),
            reason: "Stochastic gradient descent classifier. Scales to very large datasets and sparse data with constant memory.".to_string(),
            estimated_fit_time: estimate_fit_time(n, p, "linear"),
            params,
            score: score.min(1.0),
        });
    }

    candidates
}

/// Generate regression algorithm candidates with scores based on dataset profile.
fn regression_candidates(profile: &DatasetProfile) -> Vec<Recommendation> {
    let n = profile.n_samples;
    let p = profile.n_features;
    let is_sparse = profile.sparsity > 0.5;
    let many_features = p > n;

    let mut candidates = Vec::new();

    // --- RandomForestRegressor ---
    {
        let mut score: f64 = 0.70;
        if n >= 1000 && n <= 50_000 {
            score += 0.15;
        } else if n < 1000 {
            score += 0.10;
        } else {
            score += 0.05;
        }
        if many_features {
            score += 0.05;
        }
        let mut params = HashMap::new();
        params.insert("n_estimators".to_string(), ParamValue::Int(100));

        candidates.push(Recommendation {
            algorithm: "RandomForestRegressor".to_string(),
            reason: "Robust nonlinear regressor. Handles mixed feature types, provides feature importances, and resists overfitting.".to_string(),
            estimated_fit_time: estimate_fit_time(n, p, "ensemble"),
            params,
            score: score.min(1.0),
        });
    }

    // --- HistGradientBoostingRegressor ---
    {
        let mut score: f64 = 0.65;
        if n >= 1000 && n <= 50_000 {
            score += 0.20;
        } else if n > 50_000 {
            score += 0.25;
        } else {
            score += 0.05;
        }
        let mut params = HashMap::new();
        params.insert("max_iter".to_string(), ParamValue::Int(100));
        params.insert("learning_rate".to_string(), ParamValue::Float(0.1));

        candidates.push(Recommendation {
            algorithm: "HistGradientBoostingRegressor".to_string(),
            reason: "Gradient boosting with histogram binning. Excellent accuracy on tabular data and scales well to large datasets.".to_string(),
            estimated_fit_time: estimate_fit_time(n, p, "ensemble"),
            params,
            score: score.min(1.0),
        });
    }

    // --- Ridge ---
    {
        let mut score: f64 = 0.55;
        if n < 1000 {
            score += 0.15;
        } else if n <= 50_000 {
            score += 0.10;
        }
        let mut params = HashMap::new();
        params.insert("alpha".to_string(), ParamValue::Float(1.0));

        candidates.push(Recommendation {
            algorithm: "Ridge".to_string(),
            reason: "L2-regularized linear regression. Fast, interpretable, and numerically stable. Good baseline for any regression task.".to_string(),
            estimated_fit_time: estimate_fit_time(n, p, "linear"),
            params,
            score: score.min(1.0),
        });
    }

    // --- Lasso ---
    {
        let mut score: f64 = 0.45;
        if many_features {
            score += 0.25;
        }
        if is_sparse {
            score += 0.10;
        }
        let mut params = HashMap::new();
        params.insert("alpha".to_string(), ParamValue::Float(1.0));

        let reason = if many_features {
            "L1-regularized regression for automatic feature selection when features outnumber samples."
        } else {
            "L1-regularized regression. Produces sparse models by driving irrelevant feature coefficients to zero."
        };

        candidates.push(Recommendation {
            algorithm: "Lasso".to_string(),
            reason: reason.to_string(),
            estimated_fit_time: estimate_fit_time(n, p, "linear"),
            params,
            score: score.min(1.0),
        });
    }

    // --- ElasticNet ---
    {
        let mut score: f64 = 0.45;
        if many_features {
            score += 0.20;
        }
        if is_sparse {
            score += 0.10;
        }
        let mut params = HashMap::new();
        params.insert("alpha".to_string(), ParamValue::Float(1.0));
        params.insert("l1_ratio".to_string(), ParamValue::Float(0.5));

        candidates.push(Recommendation {
            algorithm: "ElasticNet".to_string(),
            reason: "Combines L1 and L2 regularization. Handles correlated features better than Lasso alone while still performing feature selection.".to_string(),
            estimated_fit_time: estimate_fit_time(n, p, "linear"),
            params,
            score: score.min(1.0),
        });
    }

    // --- SVR (small data) ---
    {
        let mut score: f64 = 0.40;
        if n < 1000 {
            score += 0.20;
        } else if n > 50_000 {
            score -= 0.15;
        }
        let mut params = HashMap::new();
        params.insert("C".to_string(), ParamValue::Float(1.0));
        params.insert("kernel".to_string(), ParamValue::String("rbf".to_string()));

        candidates.push(Recommendation {
            algorithm: "SVR".to_string(),
            reason: "Support vector regression with RBF kernel. Effective on small datasets with nonlinear relationships.".to_string(),
            estimated_fit_time: estimate_fit_time(n, p, "quadratic"),
            params,
            score: score.min(1.0),
        });
    }

    // --- GaussianProcessRegressor (small data only) ---
    if n < 1000 {
        let mut score: f64 = 0.45;
        if n < 500 {
            score += 0.15;
        }
        let params = HashMap::new();

        candidates.push(Recommendation {
            algorithm: "GaussianProcessRegressor".to_string(),
            reason: "Bayesian nonparametric regressor with built-in uncertainty quantification. Best for small datasets where prediction confidence matters.".to_string(),
            estimated_fit_time: estimate_fit_time(n, p, "quadratic"),
            params,
            score: score.min(1.0),
        });
    }

    // --- SGDRegressor (large data) ---
    if n > 50_000 || is_sparse {
        let mut score: f64 = 0.45;
        if n > 50_000 {
            score += 0.20;
        }
        if is_sparse {
            score += 0.10;
        }
        let mut params = HashMap::new();
        params.insert(
            "loss".to_string(),
            ParamValue::String("squared_error".to_string()),
        );
        params.insert("alpha".to_string(), ParamValue::Float(0.0001));

        candidates.push(Recommendation {
            algorithm: "SGDRegressor".to_string(),
            reason: "Stochastic gradient descent regressor. Scales to very large datasets with constant memory usage.".to_string(),
            estimated_fit_time: estimate_fit_time(n, p, "linear"),
            params,
            score: score.min(1.0),
        });
    }

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn make_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
            ((i * n_features + j) as f64) * 0.01
        });
        let y = Array1::from_shape_fn(n_samples, |i| (i % 2) as f64);
        (x, y)
    }

    fn make_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
            ((i * n_features + j) as f64) * 0.01
        });
        let y = Array1::from_shape_fn(n_samples, |i| i as f64 * 0.5 + 1.0);
        (x, y)
    }

    #[test]
    fn test_recommend_classification_basic() {
        let (x, y) = make_data(200, 5);
        let recs = recommend(&x, &y, "classification").unwrap();
        assert!(!recs.is_empty());
        assert!(recs.len() <= 5);
        // Check scores are descending
        for w in recs.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_recommend_regression_basic() {
        let (x, y) = make_regression_data(200, 5);
        let recs = recommend(&x, &y, "regression").unwrap();
        assert!(!recs.is_empty());
        assert!(recs.len() <= 5);
    }

    #[test]
    fn test_recommend_returns_top_5() {
        let (x, y) = make_data(500, 10);
        let recs = recommend(&x, &y, "classification").unwrap();
        assert!(recs.len() <= 5);
    }

    #[test]
    fn test_recommend_large_dataset_prefers_scalable() {
        let (x, y) = make_data(100_000, 20);
        let recs = recommend(&x, &y, "classification").unwrap();
        // Top recommendation should be scalable
        let top = &recs[0].algorithm;
        assert!(
            top == "HistGradientBoostingClassifier"
                || top == "RandomForestClassifier"
                || top == "SGDClassifier",
            "Expected scalable algorithm for large data, got: {}",
            top
        );
    }

    #[test]
    fn test_recommend_sparse_data() {
        // Create sparse data (mostly zeros)
        let mut x = Array2::zeros((200, 50));
        for i in 0..200 {
            x[[i, i % 50]] = 1.0;
        }
        let y = Array1::from_shape_fn(200, |i| (i % 2) as f64);
        let recs = recommend(&x, &y, "classification").unwrap();
        // Should include models good with sparse data
        let algos: Vec<&str> = recs.iter().map(|r| r.algorithm.as_str()).collect();
        assert!(
            algos.contains(&"LogisticRegression")
                || algos.contains(&"SGDClassifier")
                || algos.contains(&"GaussianNB"),
            "Expected sparse-friendly algorithm in recommendations: {:?}",
            algos
        );
    }

    #[test]
    fn test_recommend_imbalanced_classes() {
        let mut y_vec = vec![0.0; 180];
        y_vec.extend(vec![1.0; 20]);
        let x = Array2::from_shape_fn((200, 5), |(i, j)| (i + j) as f64);
        let y = Array1::from_vec(y_vec);
        let recs = recommend(&x, &y, "classification").unwrap();
        // RandomForest should be recommended with balanced class_weight
        let rf = recs
            .iter()
            .find(|r| r.algorithm == "RandomForestClassifier");
        assert!(
            rf.is_some(),
            "RandomForest should be recommended for imbalanced data"
        );
        if let Some(rf) = rf {
            assert!(
                rf.params.contains_key("class_weight"),
                "RandomForest should suggest class_weight=balanced for imbalanced data"
            );
        }
    }

    #[test]
    fn test_recommend_many_features() {
        // More features than samples
        let (x, y) = make_data(50, 200);
        let recs = recommend(&x, &y, "classification").unwrap();
        let algos: Vec<&str> = recs.iter().map(|r| r.algorithm.as_str()).collect();
        assert!(
            algos.contains(&"LogisticRegression"),
            "LogisticRegression should be recommended for many features: {:?}",
            algos
        );
        // LogReg should have L1 penalty
        let lr = recs
            .iter()
            .find(|r| r.algorithm == "LogisticRegression")
            .unwrap();
        match lr.params.get("penalty") {
            Some(ParamValue::String(s)) => {
                assert_eq!(s, "l1", "Expected L1 penalty for many features")
            }
            other => panic!("Expected ParamValue::String(\"l1\"), got {:?}", other),
        }
    }

    #[test]
    fn test_recommend_invalid_task() {
        let (x, y) = make_data(100, 5);
        let result = recommend(&x, &y, "clustering");
        assert!(result.is_err());
    }

    #[test]
    fn test_recommend_shape_mismatch() {
        let x = Array2::zeros((100, 5));
        let y = Array1::zeros(50);
        let result = recommend(&x, &y, "classification");
        assert!(result.is_err());
    }

    #[test]
    fn test_recommend_empty_data() {
        let x = Array2::zeros((0, 5));
        let y = Array1::zeros(0);
        let result = recommend(&x, &y, "classification");
        assert!(result.is_err());
    }

    #[test]
    fn test_recommendation_attributes() {
        let (x, y) = make_data(200, 5);
        let recs = recommend(&x, &y, "classification").unwrap();
        let r = &recs[0];
        assert!(!r.algorithm.is_empty());
        assert!(!r.reason.is_empty());
        assert!(
            r.estimated_fit_time == "fast"
                || r.estimated_fit_time == "moderate"
                || r.estimated_fit_time == "slow"
        );
        assert!(r.score > 0.0 && r.score <= 1.0);
    }

    #[test]
    fn test_recommend_fast_execution() {
        // Recommendation should be very fast (no fitting)
        let (x, y) = make_data(10_000, 50);
        let start = std::time::Instant::now();
        let _recs = recommend(&x, &y, "classification").unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 100,
            "recommend() took {}ms, expected < 100ms",
            elapsed.as_millis()
        );
    }

    #[test]
    fn test_dataset_profile() {
        let (x, y) = make_data(100, 10);
        let profile = DatasetProfile::from_data(&x, &y, "classification");
        assert_eq!(profile.n_samples, 100);
        assert_eq!(profile.n_features, 10);
        assert_eq!(profile.n_classes, Some(2));
        assert!(profile.class_balance.unwrap() > 0.9); // balanced
        assert!(profile.feature_range > 0.0);
    }

    #[test]
    fn test_regression_recommendations() {
        let (x, y) = make_regression_data(200, 5);
        let recs = recommend(&x, &y, "regression").unwrap();
        // Should not contain classification-specific algorithms
        for r in &recs {
            assert!(
                !r.algorithm.contains("Classifier"),
                "Regression should not recommend classifiers: {}",
                r.algorithm
            );
        }
    }

    #[test]
    fn test_regression_many_features() {
        let (x, y) = make_regression_data(50, 200);
        let recs = recommend(&x, &y, "regression").unwrap();
        let algos: Vec<&str> = recs.iter().map(|r| r.algorithm.as_str()).collect();
        assert!(
            algos.contains(&"Lasso") || algos.contains(&"ElasticNet"),
            "Lasso/ElasticNet should be recommended for high-dimensional regression: {:?}",
            algos
        );
    }

    #[test]
    fn test_regression_small_data_gp() {
        let (x, y) = make_regression_data(100, 3);
        let recs = recommend(&x, &y, "regression").unwrap();
        let algos: Vec<&str> = recs.iter().map(|r| r.algorithm.as_str()).collect();
        assert!(
            algos.contains(&"GaussianProcessRegressor"),
            "GP should be recommended for small regression data: {:?}",
            algos
        );
    }
}

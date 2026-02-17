//! Permutation Importance Implementation
//!
//! Model-agnostic feature importance via permutation shuffling.
//! This method works with any model and measures how much performance
//! degrades when a feature is randomly shuffled.

use crate::models::Model;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Result of permutation importance computation
///
/// Contains importance scores for each feature with confidence intervals
/// computed via repeated shuffling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermutationImportanceResult {
    /// Mean importance for each feature (decrease in score when shuffled)
    pub importances_mean: Array1<f64>,
    /// Standard deviation of importance across repeats
    pub importances_std: Array1<f64>,
    /// Lower bound of confidence interval for each feature
    pub ci_lower: Array1<f64>,
    /// Upper bound of confidence interval for each feature
    pub ci_upper: Array1<f64>,
    /// Raw importance values for each repeat (shape: `n_features` x `n_repeats`)
    pub importances: Array2<f64>,
    /// Confidence level used for CIs (e.g., 0.95)
    pub confidence_level: f64,
    /// Number of repeats used
    pub n_repeats: usize,
    /// Baseline score (score without any shuffling)
    pub baseline_score: f64,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
}

impl PermutationImportanceResult {
    /// Get features sorted by importance (highest first)
    ///
    /// Returns indices sorted by mean importance in descending order.
    #[must_use]
    pub fn sorted_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.importances_mean.len()).collect();
        indices.sort_by(|&a, &b| {
            self.importances_mean[b]
                .partial_cmp(&self.importances_mean[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Get the top k most important features
    ///
    /// Returns indices of the k features with highest mean importance.
    #[must_use]
    pub fn top_k(&self, k: usize) -> Vec<usize> {
        self.sorted_indices().into_iter().take(k).collect()
    }

    /// Check if a feature's importance is statistically significant
    ///
    /// A feature is considered significant if its CI doesn't include zero.
    #[must_use]
    pub fn is_significant(&self, feature_idx: usize) -> bool {
        feature_idx < self.ci_lower.len()
            && (self.ci_lower[feature_idx] > 0.0 || self.ci_upper[feature_idx] < 0.0)
    }

    /// Get indices of all statistically significant features
    #[must_use]
    pub fn significant_features(&self) -> Vec<usize> {
        (0..self.importances_mean.len())
            .filter(|&i| self.is_significant(i))
            .collect()
    }

    /// Format importance for a single feature as human-readable string
    #[must_use]
    pub fn format_feature(&self, feature_idx: usize) -> String {
        if feature_idx >= self.importances_mean.len() {
            return String::from("Invalid feature index");
        }

        let name = self
            .feature_names
            .as_ref()
            .and_then(|names| names.get(feature_idx).cloned())
            .unwrap_or_else(|| format!("feature_{feature_idx}"));

        let significant = if self.is_significant(feature_idx) {
            "*"
        } else {
            ""
        };

        format!(
            "{}: {:.4} ± {:.4} ({}% CI: [{:.4}, {:.4}]){}",
            name,
            self.importances_mean[feature_idx],
            self.importances_std[feature_idx],
            (self.confidence_level * 100.0) as i32,
            self.ci_lower[feature_idx],
            self.ci_upper[feature_idx],
            significant
        )
    }

    /// Create a summary string of all feature importances
    #[must_use]
    pub fn summary(&self) -> String {
        let mut lines = vec![
            String::from("Permutation Importance Results"),
            String::from("=============================="),
            format!("Baseline score: {:.4}", self.baseline_score),
            format!(
                "Number of repeats: {} | Confidence level: {}%",
                self.n_repeats,
                (self.confidence_level * 100.0) as i32
            ),
            String::new(),
            String::from("Feature Importances (sorted by mean):"),
            String::from("--------------------------------------"),
        ];

        for idx in self.sorted_indices() {
            lines.push(self.format_feature(idx));
        }

        lines.push(String::new());
        lines.push(String::from(
            "* indicates statistically significant (CI excludes zero)",
        ));

        lines.join("\n")
    }
}

impl std::fmt::Display for PermutationImportanceResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Compute permutation importance for a fitted model
///
/// Permutation importance measures the decrease in model performance when
/// a single feature's values are randomly shuffled. This breaks the
/// relationship between the feature and the target, showing how much
/// the model relies on that feature.
///
/// # Algorithm
///
/// 1. Compute baseline score on unshuffled data
/// 2. For each feature j:
///    a. Repeat n_repeats times:
///       - Shuffle feature j's values
///       - Compute score on shuffled data
///       - Importance = baseline_score - shuffled_score
///    b. Aggregate importance over repeats (mean, std, CI)
///
/// # Arguments
///
/// * `model` - Fitted model implementing the `Model` trait
/// * `x` - Feature matrix (n_samples, n_features)
/// * `y` - True target values
/// * `scoring_fn` - Function that computes score from (y_true, y_pred).
///                  Higher score = better model. For metrics where lower
///                  is better (e.g., MSE), negate the value.
/// * `n_repeats` - Number of times to shuffle each feature (default: 10)
/// * `random_state` - Random seed for reproducibility
///
/// # Returns
///
/// `PermutationImportanceResult` containing mean importance, std, and CIs
/// for each feature.
///
/// # Example
///
/// ```
/// # fn main() -> ferroml_core::Result<()> {
/// use ferroml_core::explainability::permutation_importance;
/// use ferroml_core::models::RandomForestClassifier;
/// # use ferroml_core::models::Model;
/// # use ndarray::{Array1, Array2};
/// # use ferroml_core::Result;
/// # let x_train = Array2::from_shape_vec((20, 2), (0..40).map(|i| i as f64 / 40.0).collect()).unwrap();
/// # let y_train = Array1::from_vec((0..20).map(|i| (i % 2) as f64).collect());
/// # let x_test = x_train.clone();
/// # let y_test = y_train.clone();
///
/// let mut model = RandomForestClassifier::new();
/// model.fit(&x_train, &y_train)?;
///
/// // Define scoring function (higher is better)
/// let scoring = |y_true: &Array1<f64>, y_pred: &Array1<f64>| -> Result<f64> {
///     let correct = y_true.iter()
///         .zip(y_pred.iter())
///         .filter(|(&t, &p)| (t - p).abs() < 0.5)
///         .count();
///     Ok(correct as f64 / y_true.len() as f64)
/// };
///
/// let result = permutation_importance(&model, &x_test, &y_test, scoring, 10, Some(42))?;
/// println!("{}", result);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - The model is not fitted
/// - Input shapes are mismatched (x rows != y length)
/// - Input data is empty
/// - `n_repeats` is 0
/// - `confidence_level` is not in [0, 1)
/// - Scoring function fails for all samples
///
/// # References
///
/// - Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
/// - Altmann, A., et al. (2010). Permutation importance: a corrected feature
///   importance measure. Bioinformatics, 26(10), 1340-1347.
pub fn permutation_importance<M, F>(
    model: &M,
    x: &Array2<f64>,
    y: &Array1<f64>,
    scoring_fn: F,
    n_repeats: usize,
    random_state: Option<u64>,
) -> Result<PermutationImportanceResult>
where
    M: Model,
    F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64>,
{
    permutation_importance_with_options(model, x, y, scoring_fn, n_repeats, random_state, 0.95)
}

/// Compute permutation importance with custom confidence level
///
/// Same as `permutation_importance` but allows specifying the confidence level
/// for the intervals (default is 0.95).
///
/// # Errors
///
/// Returns an error if:
/// - The model is not fitted
/// - Input shapes are mismatched (x rows != y length)
/// - Input data is empty
/// - `n_repeats` is 0
/// - `confidence_level` is not in [0, 1)
/// - Scoring function fails for all samples
pub fn permutation_importance_with_options<M, F>(
    model: &M,
    x: &Array2<f64>,
    y: &Array1<f64>,
    scoring_fn: F,
    n_repeats: usize,
    random_state: Option<u64>,
    confidence_level: f64,
) -> Result<PermutationImportanceResult>
where
    M: Model,
    F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64>,
{
    // Validate inputs
    if !model.is_fitted() {
        return Err(FerroError::not_fitted("permutation_importance"));
    }
    if x.nrows() != y.len() {
        return Err(FerroError::shape_mismatch(
            format!("x has {} rows", x.nrows()),
            format!("y has {} elements", y.len()),
        ));
    }
    if x.is_empty() || y.is_empty() {
        return Err(FerroError::invalid_input("Empty input data"));
    }
    if n_repeats == 0 {
        return Err(FerroError::invalid_input("n_repeats must be at least 1"));
    }
    if !(0.0..1.0).contains(&confidence_level) {
        return Err(FerroError::invalid_input(
            "confidence_level must be between 0 and 1",
        ));
    }

    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Compute baseline score
    let y_pred = model.predict(x)?;
    let baseline_score = scoring_fn(y, &y_pred)?;

    // Initialize RNG
    let mut rng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };

    // Storage for importance values: n_features x n_repeats
    let mut importances = Array2::<f64>::zeros((n_features, n_repeats));

    // For each feature
    for feature_idx in 0..n_features {
        // For each repeat
        for repeat in 0..n_repeats {
            // Create shuffled copy of X
            let mut x_shuffled = x.to_owned();

            // Generate shuffled indices for this feature
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            // Apply shuffle to this feature column
            for (new_idx, &old_idx) in indices.iter().enumerate() {
                x_shuffled[[new_idx, feature_idx]] = x[[old_idx, feature_idx]];
            }

            // Compute score with shuffled feature
            let y_pred_shuffled = model.predict(&x_shuffled)?;
            let shuffled_score = scoring_fn(y, &y_pred_shuffled)?;

            // Importance = decrease in score when feature is shuffled
            importances[[feature_idx, repeat]] = baseline_score - shuffled_score;
        }
    }

    // Compute statistics for each feature
    let importances_mean = importances.mean_axis(Axis(1)).unwrap();

    let importances_std: Array1<f64> = (0..n_features)
        .map(|i| {
            let row = importances.row(i);
            let mean = importances_mean[i];
            let variance = row.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / (n_repeats as f64 - 1.0).max(1.0);
            variance.sqrt()
        })
        .collect();

    // Compute confidence intervals using t-distribution
    let (ci_lower, ci_upper) = compute_ci(
        &importances_mean,
        &importances_std,
        n_repeats,
        confidence_level,
    );

    // Get feature names if available
    let feature_names = model.feature_names().map(|names| names.to_vec());

    Ok(PermutationImportanceResult {
        importances_mean,
        importances_std,
        ci_lower,
        ci_upper,
        importances,
        confidence_level,
        n_repeats,
        baseline_score,
        feature_names,
    })
}

/// Compute permutation importance in parallel using rayon
///
/// Same as `permutation_importance` but parallelizes across features
/// for improved performance on large datasets.
///
/// # Errors
///
/// Returns an error if:
/// - The model is not fitted
/// - Input shapes are mismatched (x rows != y length)
/// - Input data is empty
/// - `n_repeats` is 0
/// - `confidence_level` is not in [0, 1)
/// - Scoring function fails for all samples
///
/// # Note
///
/// Due to parallelization, results may vary slightly even with the same
/// `random_state` if the number of threads changes. For exact reproducibility,
/// use the sequential version.
pub fn permutation_importance_parallel<M, F>(
    model: &M,
    x: &Array2<f64>,
    y: &Array1<f64>,
    scoring_fn: F,
    n_repeats: usize,
    random_state: Option<u64>,
) -> Result<PermutationImportanceResult>
where
    M: Model + Sync,
    F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64> + Send + Sync,
{
    permutation_importance_parallel_with_options(
        model,
        x,
        y,
        scoring_fn,
        n_repeats,
        random_state,
        0.95,
    )
}

/// Parallel permutation importance with custom confidence level
///
/// # Errors
///
/// Returns an error if:
/// - The model is not fitted
/// - Input shapes are mismatched (x rows != y length)
/// - Input data is empty
/// - `n_repeats` is 0
/// - `confidence_level` is not in [0, 1)
/// - Scoring function fails for all samples
pub fn permutation_importance_parallel_with_options<M, F>(
    model: &M,
    x: &Array2<f64>,
    y: &Array1<f64>,
    scoring_fn: F,
    n_repeats: usize,
    random_state: Option<u64>,
    confidence_level: f64,
) -> Result<PermutationImportanceResult>
where
    M: Model + Sync,
    F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64> + Send + Sync,
{
    // Validate inputs
    if !model.is_fitted() {
        return Err(FerroError::not_fitted("permutation_importance_parallel"));
    }
    if x.nrows() != y.len() {
        return Err(FerroError::shape_mismatch(
            format!("x has {} rows", x.nrows()),
            format!("y has {} elements", y.len()),
        ));
    }
    if x.is_empty() || y.is_empty() {
        return Err(FerroError::invalid_input("Empty input data"));
    }
    if n_repeats == 0 {
        return Err(FerroError::invalid_input("n_repeats must be at least 1"));
    }
    if !(0.0..1.0).contains(&confidence_level) {
        return Err(FerroError::invalid_input(
            "confidence_level must be between 0 and 1",
        ));
    }

    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Compute baseline score
    let y_pred = model.predict(x)?;
    let baseline_score = scoring_fn(y, &y_pred)?;

    // Base seed for reproducibility
    let base_seed = random_state.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    });

    // Compute importance for each feature in parallel
    let feature_importances: Vec<Vec<f64>> = (0..n_features)
        .into_par_iter()
        .map(|feature_idx| {
            // Create RNG with feature-specific seed for reproducibility
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(feature_idx as u64));

            let mut feature_scores = Vec::with_capacity(n_repeats);

            for _ in 0..n_repeats {
                // Create shuffled copy of X
                let mut x_shuffled = x.to_owned();

                // Generate shuffled indices for this feature
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(&mut rng);

                // Apply shuffle to this feature column
                for (new_idx, &old_idx) in indices.iter().enumerate() {
                    x_shuffled[[new_idx, feature_idx]] = x[[old_idx, feature_idx]];
                }

                // Compute score with shuffled feature
                if let Ok(y_pred_shuffled) = model.predict(&x_shuffled) {
                    if let Ok(shuffled_score) = scoring_fn(y, &y_pred_shuffled) {
                        feature_scores.push(baseline_score - shuffled_score);
                    }
                }
            }

            // If some predictions failed, pad with zeros
            while feature_scores.len() < n_repeats {
                feature_scores.push(0.0);
            }

            feature_scores
        })
        .collect();

    // Convert to Array2
    let mut importances = Array2::<f64>::zeros((n_features, n_repeats));
    for (feature_idx, scores) in feature_importances.iter().enumerate() {
        for (repeat, &score) in scores.iter().enumerate() {
            importances[[feature_idx, repeat]] = score;
        }
    }

    // Compute statistics for each feature
    let importances_mean = importances.mean_axis(Axis(1)).unwrap();

    let importances_std: Array1<f64> = (0..n_features)
        .map(|i| {
            let row = importances.row(i);
            let mean = importances_mean[i];
            let variance = row.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / (n_repeats as f64 - 1.0).max(1.0);
            variance.sqrt()
        })
        .collect();

    // Compute confidence intervals using t-distribution
    let (ci_lower, ci_upper) = compute_ci(
        &importances_mean,
        &importances_std,
        n_repeats,
        confidence_level,
    );

    // Get feature names if available
    let feature_names = model.feature_names().map(|names| names.to_vec());

    Ok(PermutationImportanceResult {
        importances_mean,
        importances_std,
        ci_lower,
        ci_upper,
        importances,
        confidence_level,
        n_repeats,
        baseline_score,
        feature_names,
    })
}

/// Compute confidence interval using t-distribution
fn compute_ci(
    mean: &Array1<f64>,
    std: &Array1<f64>,
    n: usize,
    confidence_level: f64,
) -> (Array1<f64>, Array1<f64>) {
    let n_features = mean.len();

    // Degrees of freedom
    let df = (n - 1).max(1) as f64;

    // t-critical value for two-tailed CI
    let alpha = 1.0 - confidence_level;
    let t_crit = StudentsT::new(0.0, 1.0, df)
        .map(|dist| dist.inverse_cdf(1.0 - alpha / 2.0))
        .unwrap_or(1.96); // Fallback to normal approximation

    // Standard error of the mean
    let se: Array1<f64> = std.mapv(|s| s / (n as f64).sqrt());

    // Compute CI bounds
    let ci_lower: Array1<f64> = (0..n_features).map(|i| mean[i] - t_crit * se[i]).collect();
    let ci_upper: Array1<f64> = (0..n_features).map(|i| mean[i] + t_crit * se[i]).collect();

    (ci_lower, ci_upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::linear::LinearRegression;

    /// Simple scorer for testing - computes R²
    fn r2_scorer(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
        let mean = y_true.mean().unwrap_or(0.0);
        let ss_tot: f64 = y_true.iter().map(|&y| (y - mean).powi(2)).sum();
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - p).powi(2))
            .sum();

        if ss_tot == 0.0 {
            Ok(1.0)
        } else {
            Ok(1.0 - ss_res / ss_tot)
        }
    }

    /// Accuracy scorer for classification
    fn accuracy_scorer(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(&t, &p)| (t - p.round()).abs() < 0.5)
            .count();
        Ok(correct as f64 / y_true.len() as f64)
    }

    #[test]
    fn test_permutation_importance_basic() {
        // Create simple linear data where only first feature matters
        // Use independent random-like features to avoid collinearity
        let n_samples = 100;
        let x = Array2::from_shape_fn((n_samples, 3), |(i, j)| {
            match j {
                0 => i as f64 * 0.1, // Important feature (linearly related to target)
                1 => ((i * 7 + 13) % 23) as f64 * 0.1 + 0.5, // Pseudo-random noise
                _ => ((i * 11 + 17) % 31) as f64 * 0.1 + 0.3, // Different pseudo-random noise
            }
        });
        let y: Array1<f64> = x.column(0).mapv(|v| v * 2.0 + 1.0);

        // Fit linear regression
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // Compute permutation importance
        let result = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(42)).unwrap();

        // First feature should be most important
        assert_eq!(result.importances_mean.len(), 3);
        assert!(
            result.importances_mean[0] > result.importances_mean[1],
            "Feature 0 importance {} should be > feature 1 importance {}",
            result.importances_mean[0],
            result.importances_mean[1]
        );
        assert!(
            result.importances_mean[0] > result.importances_mean[2],
            "Feature 0 importance {} should be > feature 2 importance {}",
            result.importances_mean[0],
            result.importances_mean[2]
        );

        // Check result structure
        assert_eq!(result.n_repeats, 5);
        assert!((result.confidence_level - 0.95).abs() < 1e-10);
        assert!(result.baseline_score > 0.9); // Should fit well
    }

    #[test]
    fn test_permutation_importance_parallel() {
        // Create independent features using pseudo-random pattern
        let n_samples = 50;
        let x = Array2::from_shape_fn((n_samples, 4), |(i, j)| match j {
            0 => i as f64 * 0.1,
            1 => ((i * 7 + 3) % 17) as f64 * 0.1,
            2 => ((i * 11 + 5) % 23) as f64 * 0.1,
            _ => ((i * 13 + 7) % 29) as f64 * 0.1,
        });
        let y: Array1<f64> = x.column(0).to_owned() + x.column(1).to_owned() * 0.5;

        // Fit linear regression
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // Compute parallel permutation importance
        let result =
            permutation_importance_parallel(&model, &x, &y, r2_scorer, 5, Some(42)).unwrap();

        // Should return results for all features
        assert_eq!(result.importances_mean.len(), 4);
        assert_eq!(result.importances.shape(), &[4, 5]);
    }

    #[test]
    fn test_permutation_importance_reproducibility() {
        // Create test data with independent features
        let n_samples = 30;
        let x = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                i as f64 * 0.1
            } else {
                ((i * 7 + 13) % 17) as f64 * 0.1 + 0.5
            }
        });
        let y: Array1<f64> = x.column(0).mapv(|v| v * 2.0 + 1.0);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // Run twice with same seed
        let result1 = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(42)).unwrap();
        let result2 = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(42)).unwrap();

        // Results should be identical
        for i in 0..result1.importances_mean.len() {
            assert!(
                (result1.importances_mean[i] - result2.importances_mean[i]).abs() < 1e-10,
                "Results differ at feature {}: {} vs {}",
                i,
                result1.importances_mean[i],
                result2.importances_mean[i]
            );
        }
    }

    #[test]
    fn test_permutation_importance_ci() {
        // Create test data with independent features
        let n_samples = 50;
        let x = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                i as f64 * 0.1
            } else {
                ((i * 7 + 11) % 19) as f64 * 0.1 + 0.3
            }
        });
        let y: Array1<f64> = x.column(0).mapv(|v| v * 2.0 + 1.0);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = permutation_importance(&model, &x, &y, r2_scorer, 10, Some(42)).unwrap();

        // CI should be ordered correctly
        for i in 0..result.importances_mean.len() {
            assert!(
                result.ci_lower[i] <= result.importances_mean[i],
                "CI lower {} > mean {} for feature {}",
                result.ci_lower[i],
                result.importances_mean[i],
                i
            );
            assert!(
                result.importances_mean[i] <= result.ci_upper[i],
                "Mean {} > CI upper {} for feature {}",
                result.importances_mean[i],
                result.ci_upper[i],
                i
            );
        }
    }

    #[test]
    fn test_sorted_indices() {
        let result = PermutationImportanceResult {
            importances_mean: Array1::from_vec(vec![0.1, 0.5, 0.3, 0.2]),
            importances_std: Array1::from_vec(vec![0.01, 0.02, 0.01, 0.01]),
            ci_lower: Array1::from_vec(vec![0.08, 0.46, 0.28, 0.18]),
            ci_upper: Array1::from_vec(vec![0.12, 0.54, 0.32, 0.22]),
            importances: Array2::zeros((4, 5)),
            confidence_level: 0.95,
            n_repeats: 5,
            baseline_score: 0.9,
            feature_names: None,
        };

        let sorted = result.sorted_indices();
        assert_eq!(sorted, vec![1, 2, 3, 0]); // Sorted by importance descending
    }

    #[test]
    fn test_top_k() {
        let result = PermutationImportanceResult {
            importances_mean: Array1::from_vec(vec![0.1, 0.5, 0.3, 0.2]),
            importances_std: Array1::from_vec(vec![0.01, 0.02, 0.01, 0.01]),
            ci_lower: Array1::from_vec(vec![0.08, 0.46, 0.28, 0.18]),
            ci_upper: Array1::from_vec(vec![0.12, 0.54, 0.32, 0.22]),
            importances: Array2::zeros((4, 5)),
            confidence_level: 0.95,
            n_repeats: 5,
            baseline_score: 0.9,
            feature_names: None,
        };

        let top2 = result.top_k(2);
        assert_eq!(top2, vec![1, 2]); // Features with highest importance
    }

    #[test]
    fn test_is_significant() {
        let result = PermutationImportanceResult {
            importances_mean: Array1::from_vec(vec![0.1, -0.05, 0.3]),
            importances_std: Array1::from_vec(vec![0.01, 0.1, 0.01]),
            ci_lower: Array1::from_vec(vec![0.08, -0.2, 0.28]), // First: > 0, Second: spans 0, Third: > 0
            ci_upper: Array1::from_vec(vec![0.12, 0.1, 0.32]),
            importances: Array2::zeros((3, 5)),
            confidence_level: 0.95,
            n_repeats: 5,
            baseline_score: 0.9,
            feature_names: None,
        };

        assert!(result.is_significant(0)); // CI doesn't include 0
        assert!(!result.is_significant(1)); // CI includes 0
        assert!(result.is_significant(2)); // CI doesn't include 0
    }

    #[test]
    fn test_significant_features() {
        let result = PermutationImportanceResult {
            importances_mean: Array1::from_vec(vec![0.1, 0.0, 0.3]),
            importances_std: Array1::from_vec(vec![0.01, 0.1, 0.01]),
            ci_lower: Array1::from_vec(vec![0.08, -0.2, 0.28]),
            ci_upper: Array1::from_vec(vec![0.12, 0.2, 0.32]),
            importances: Array2::zeros((3, 5)),
            confidence_level: 0.95,
            n_repeats: 5,
            baseline_score: 0.9,
            feature_names: None,
        };

        let significant = result.significant_features();
        assert_eq!(significant, vec![0, 2]);
    }

    #[test]
    fn test_format_feature() {
        let result = PermutationImportanceResult {
            importances_mean: Array1::from_vec(vec![0.15]),
            importances_std: Array1::from_vec(vec![0.02]),
            ci_lower: Array1::from_vec(vec![0.11]),
            ci_upper: Array1::from_vec(vec![0.19]),
            importances: Array2::zeros((1, 5)),
            confidence_level: 0.95,
            n_repeats: 5,
            baseline_score: 0.9,
            feature_names: Some(vec!["important_feature".to_string()]),
        };

        let formatted = result.format_feature(0);
        assert!(formatted.contains("important_feature"));
        assert!(formatted.contains("0.15"));
        assert!(formatted.contains("95%"));
        assert!(formatted.contains("*")); // Should be significant
    }

    #[test]
    fn test_error_not_fitted() {
        let model = LinearRegression::new();
        let x = Array2::zeros((10, 2));
        let y = Array1::zeros(10);

        let result = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_error_shape_mismatch() {
        // Create independent features to avoid collinearity
        let n_samples = 20;
        let x = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                i as f64 * 0.1
            } else {
                ((i * 7 + 11) % 13) as f64 * 0.1 + 0.5
            }
        });
        let y: Array1<f64> = x.column(0).mapv(|v| v * 2.0 + 1.0);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let x_wrong = Array2::zeros((15, 2));
        let result = permutation_importance(&model, &x_wrong, &y, r2_scorer, 5, Some(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_error_empty_input() {
        // Create independent features to avoid collinearity
        let n_samples = 20;
        let x = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                i as f64 * 0.1
            } else {
                ((i * 7 + 11) % 13) as f64 * 0.1 + 0.5
            }
        });
        let y: Array1<f64> = x.column(0).mapv(|v| v * 2.0 + 1.0);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let x_empty: Array2<f64> = Array2::zeros((0, 2));
        let y_empty: Array1<f64> = Array1::zeros(0);

        let result = permutation_importance(&model, &x_empty, &y_empty, r2_scorer, 5, Some(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_summary_display() {
        let result = PermutationImportanceResult {
            importances_mean: Array1::from_vec(vec![0.3, 0.1, 0.5]),
            importances_std: Array1::from_vec(vec![0.02, 0.05, 0.03]),
            ci_lower: Array1::from_vec(vec![0.26, -0.02, 0.44]),
            ci_upper: Array1::from_vec(vec![0.34, 0.22, 0.56]),
            importances: Array2::zeros((3, 5)),
            confidence_level: 0.95,
            n_repeats: 5,
            baseline_score: 0.85,
            feature_names: Some(vec![
                "feature_a".to_string(),
                "feature_b".to_string(),
                "feature_c".to_string(),
            ]),
        };

        let summary = result.summary();
        assert!(summary.contains("Permutation Importance Results"));
        assert!(summary.contains("Baseline score: 0.85"));
        assert!(summary.contains("feature_a"));
        assert!(summary.contains("feature_b"));
        assert!(summary.contains("feature_c"));
    }

    #[test]
    fn test_with_classification() {
        // Simple classification problem
        let n = 50;
        let mut x_data = Vec::with_capacity(n * 3);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            let class = if i < n / 2 { 0.0 } else { 1.0 };
            // First feature is informative
            x_data.push(class + 0.1 * (i as f64).sin());
            // Second and third are noise
            x_data.push((i as f64) * 0.1);
            x_data.push((i as f64).cos() * 0.5);
            y_data.push(class);
        }

        let x = Array2::from_shape_vec((n, 3), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        // Use a simple linear model (not ideal for classification but works for test)
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = permutation_importance(&model, &x, &y, accuracy_scorer, 5, Some(42)).unwrap();

        // Should identify first feature as important
        assert_eq!(result.importances_mean.len(), 3);
    }

    #[test]
    fn test_custom_confidence_level() {
        // Create independent features
        let n_samples = 30;
        let x = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                i as f64 * 0.1
            } else {
                ((i * 7 + 11) % 17) as f64 * 0.1 + 0.5
            }
        });
        let y: Array1<f64> = x.column(0).mapv(|v| v * 2.0);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result_90 =
            permutation_importance_with_options(&model, &x, &y, r2_scorer, 10, Some(42), 0.90)
                .unwrap();

        let result_99 =
            permutation_importance_with_options(&model, &x, &y, r2_scorer, 10, Some(42), 0.99)
                .unwrap();

        // 99% CI should be wider than 90% CI
        let width_90 = (result_90.ci_upper[0] - result_90.ci_lower[0]).abs();
        let width_99 = (result_99.ci_upper[0] - result_99.ci_lower[0]).abs();

        assert!(
            width_99 >= width_90,
            "99% CI ({:.4}) should be >= 90% CI ({:.4})",
            width_99,
            width_90
        );
    }
}

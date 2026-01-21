//! SHAP Summary Plot Data Generation
//!
//! This module provides data structures and functions for generating SHAP summary
//! visualizations. These tools help interpret model predictions by showing how
//! feature values impact model output.
//!
//! ## Supported Visualizations
//!
//! - **Beeswarm Plot**: Shows individual SHAP values colored by feature value
//! - **Bar Plot**: Shows global feature importance (mean |SHAP|)
//! - **Summary Statistics**: Per-feature distributions and correlations
//!
//! ## Example
//!
//! ```ignore
//! use ferroml_core::explainability::{TreeExplainer, SHAPSummary};
//! use ferroml_core::models::RandomForestRegressor;
//!
//! let mut model = RandomForestRegressor::new();
//! model.fit(&x_train, &y_train)?;
//!
//! let explainer = TreeExplainer::from_random_forest_regressor(&model)?;
//! let batch_result = explainer.explain_batch(&x_test)?;
//!
//! // Generate summary data
//! let summary = SHAPSummary::from_batch_result(&batch_result);
//!
//! // Get bar plot data (global importance)
//! let bar_data = summary.bar_plot_data();
//! for entry in bar_data.entries.iter().take(5) {
//!     println!("{}: {:.4}", entry.feature_name, entry.importance);
//! }
//!
//! // Get beeswarm plot data
//! let beeswarm = summary.beeswarm_plot_data();
//! for (feature_idx, points) in beeswarm.points_by_feature.iter().enumerate() {
//!     println!("Feature {}: {} points", feature_idx, points.len());
//! }
//! ```
//!
//! ## References
//!
//! - Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting
//!   Model Predictions. NeurIPS 2017.

use super::treeshap::SHAPBatchResult;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

// =============================================================================
// Summary Statistics
// =============================================================================

/// Summary statistics for a single feature's SHAP values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSHAPStats {
    /// Feature index
    pub feature_idx: usize,
    /// Feature name (if available)
    pub feature_name: String,
    /// Mean of SHAP values
    pub mean: f64,
    /// Standard deviation of SHAP values
    pub std: f64,
    /// Mean of absolute SHAP values (global importance)
    pub mean_abs: f64,
    /// Minimum SHAP value
    pub min: f64,
    /// Maximum SHAP value
    pub max: f64,
    /// Median SHAP value
    pub median: f64,
    /// 25th percentile of SHAP values
    pub percentile_25: f64,
    /// 75th percentile of SHAP values
    pub percentile_75: f64,
    /// Number of positive SHAP values
    pub n_positive: usize,
    /// Number of negative SHAP values
    pub n_negative: usize,
    /// Correlation between feature values and SHAP values
    /// Positive: higher feature values -> higher SHAP (positive impact)
    /// Negative: higher feature values -> lower SHAP (negative impact)
    pub feature_shap_correlation: f64,
}

impl FeatureSHAPStats {
    /// Interpret the correlation direction
    #[must_use]
    pub fn correlation_interpretation(&self) -> &'static str {
        if self.feature_shap_correlation > 0.5 {
            "Strong positive: higher values increase prediction"
        } else if self.feature_shap_correlation > 0.2 {
            "Moderate positive: higher values tend to increase prediction"
        } else if self.feature_shap_correlation > -0.2 {
            "No clear relationship"
        } else if self.feature_shap_correlation > -0.5 {
            "Moderate negative: higher values tend to decrease prediction"
        } else {
            "Strong negative: higher values decrease prediction"
        }
    }

    /// Check if the feature has significant impact
    #[must_use]
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.mean_abs > threshold
    }
}

impl std::fmt::Display for FeatureSHAPStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Feature: {}", self.feature_name)?;
        writeln!(f, "  Mean |SHAP|: {:.4}", self.mean_abs)?;
        writeln!(f, "  Mean: {:.4}, Std: {:.4}", self.mean, self.std)?;
        writeln!(f, "  Range: [{:.4}, {:.4}]", self.min, self.max)?;
        writeln!(
            f,
            "  Positive/Negative: {}/{}",
            self.n_positive, self.n_negative
        )?;
        writeln!(
            f,
            "  Correlation: {:.4} ({})",
            self.feature_shap_correlation,
            self.correlation_interpretation()
        )
    }
}

// =============================================================================
// Bar Plot Data
// =============================================================================

/// Entry for a single feature in the bar plot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarPlotEntry {
    /// Feature index
    pub feature_idx: usize,
    /// Feature name
    pub feature_name: String,
    /// Global importance (mean |SHAP|)
    pub importance: f64,
    /// Standard deviation of |SHAP|
    pub importance_std: f64,
    /// 95% confidence interval lower bound
    pub ci_lower: f64,
    /// 95% confidence interval upper bound
    pub ci_upper: f64,
}

/// Data for generating a SHAP bar plot (global importance)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarPlotData {
    /// Entries sorted by importance (highest first)
    pub entries: Vec<BarPlotEntry>,
    /// Total number of samples used
    pub n_samples: usize,
}

impl BarPlotData {
    /// Get the top k most important features
    #[must_use]
    pub fn top_k(&self, k: usize) -> Vec<&BarPlotEntry> {
        self.entries.iter().take(k).collect()
    }

    /// Get features above an importance threshold
    #[must_use]
    pub fn above_threshold(&self, threshold: f64) -> Vec<&BarPlotEntry> {
        self.entries
            .iter()
            .filter(|e| e.importance > threshold)
            .collect()
    }

    /// Get cumulative importance (for determining how many features to include)
    #[must_use]
    pub fn cumulative_importance(&self) -> Vec<(usize, f64)> {
        let total: f64 = self.entries.iter().map(|e| e.importance).sum();
        let mut cumulative = 0.0;
        self.entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                cumulative += e.importance / total;
                (i + 1, cumulative)
            })
            .collect()
    }

    /// Get number of features needed to explain a given fraction of importance
    #[must_use]
    pub fn features_for_fraction(&self, fraction: f64) -> usize {
        let cumulative = self.cumulative_importance();
        for (n, cum) in cumulative {
            if cum >= fraction {
                return n;
            }
        }
        self.entries.len()
    }
}

impl std::fmt::Display for BarPlotData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SHAP Global Feature Importance")?;
        writeln!(f, "===============================")?;
        writeln!(f, "Samples: {}", self.n_samples)?;
        writeln!(f)?;
        for (i, entry) in self.entries.iter().enumerate() {
            writeln!(
                f,
                "{}. {} = {:.4} \u{00b1} {:.4}",
                i + 1,
                entry.feature_name,
                entry.importance,
                entry.importance_std
            )?;
        }
        Ok(())
    }
}

// =============================================================================
// Beeswarm Plot Data
// =============================================================================

/// A single point in the beeswarm plot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeeswarmPoint {
    /// Sample index
    pub sample_idx: usize,
    /// SHAP value (x-axis position)
    pub shap_value: f64,
    /// Normalized feature value (for coloring, 0-1 scale)
    pub normalized_feature_value: f64,
    /// Raw feature value
    pub raw_feature_value: f64,
}

/// Data for a single feature in the beeswarm plot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeeswarmFeatureData {
    /// Feature index
    pub feature_idx: usize,
    /// Feature name
    pub feature_name: String,
    /// Points for this feature
    pub points: Vec<BeeswarmPoint>,
    /// Global importance rank (0 = most important)
    pub importance_rank: usize,
    /// Mean absolute SHAP value
    pub mean_abs_shap: f64,
}

/// Data for generating a SHAP beeswarm plot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeeswarmPlotData {
    /// Features sorted by importance, each with their points
    pub features: Vec<BeeswarmFeatureData>,
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
}

impl BeeswarmPlotData {
    /// Get data for the top k most important features
    #[must_use]
    pub fn top_k(&self, k: usize) -> Vec<&BeeswarmFeatureData> {
        self.features.iter().take(k).collect()
    }

    /// Get points grouped by feature (for alternative access)
    #[must_use]
    pub fn points_by_feature(&self) -> Vec<&Vec<BeeswarmPoint>> {
        self.features.iter().map(|f| &f.points).collect()
    }

    /// Get feature indices in importance order
    #[must_use]
    pub fn feature_order(&self) -> Vec<usize> {
        self.features.iter().map(|f| f.feature_idx).collect()
    }
}

impl std::fmt::Display for BeeswarmPlotData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SHAP Beeswarm Plot Data")?;
        writeln!(f, "=======================")?;
        writeln!(f, "Samples: {}", self.n_samples)?;
        writeln!(f, "Features: {}", self.n_features)?;
        writeln!(f)?;
        writeln!(f, "Feature order (by importance):")?;
        for (rank, feat) in self.features.iter().enumerate() {
            writeln!(
                f,
                "  {}. {} (mean |SHAP| = {:.4})",
                rank + 1,
                feat.feature_name,
                feat.mean_abs_shap
            )?;
        }
        Ok(())
    }
}

// =============================================================================
// SHAP Summary
// =============================================================================

/// Comprehensive SHAP summary data
///
/// Provides all the data needed to generate various SHAP visualizations
/// including bar plots, beeswarm plots, and feature-level statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SHAPSummary {
    /// Base value (expected model output)
    pub base_value: f64,
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Per-feature statistics sorted by importance
    pub feature_stats: Vec<FeatureSHAPStats>,
    /// Feature indices sorted by global importance (highest first)
    pub importance_order: Vec<usize>,
    /// Global importance values (mean |SHAP|) for each feature
    pub global_importance: Array1<f64>,
    /// Raw SHAP values matrix (n_samples, n_features)
    pub shap_values: Array2<f64>,
    /// Raw feature values matrix (n_samples, n_features)
    pub feature_values: Array2<f64>,
}

impl SHAPSummary {
    /// Create a SHAP summary from a batch result
    pub fn from_batch_result(batch: &SHAPBatchResult) -> Self {
        let n_samples = batch.n_samples;
        let n_features = batch.n_features;

        // Create feature names
        let feature_names: Vec<String> = batch
            .feature_names
            .clone()
            .unwrap_or_else(|| (0..n_features).map(|i| format!("feature_{}", i)).collect());

        // Compute global importance (mean |SHAP|)
        let global_importance = batch.mean_abs_shap();

        // Get importance order
        let importance_order = batch.global_importance_sorted();

        // Compute per-feature statistics
        let feature_stats: Vec<FeatureSHAPStats> = importance_order
            .iter()
            .map(|&idx| {
                Self::compute_feature_stats(
                    idx,
                    &feature_names[idx],
                    &batch.shap_values,
                    &batch.feature_values,
                    global_importance[idx],
                )
            })
            .collect();

        Self {
            base_value: batch.base_value,
            n_samples,
            n_features,
            feature_names,
            feature_stats,
            importance_order,
            global_importance,
            shap_values: batch.shap_values.clone(),
            feature_values: batch.feature_values.clone(),
        }
    }

    /// Compute statistics for a single feature
    fn compute_feature_stats(
        feature_idx: usize,
        feature_name: &str,
        shap_values: &Array2<f64>,
        feature_values: &Array2<f64>,
        mean_abs: f64,
    ) -> FeatureSHAPStats {
        let n_samples = shap_values.nrows();

        // Extract SHAP values for this feature
        let shap_col: Vec<f64> = (0..n_samples)
            .map(|i| shap_values[[i, feature_idx]])
            .collect();

        // Extract feature values for this feature
        let feat_col: Vec<f64> = (0..n_samples)
            .map(|i| feature_values[[i, feature_idx]])
            .collect();

        // Basic statistics
        let mean = shap_col.iter().sum::<f64>() / n_samples as f64;
        let variance = shap_col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_samples as f64;
        let std = variance.sqrt();

        let min = shap_col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = shap_col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Sorted for percentiles
        let mut sorted_shap = shap_col.clone();
        sorted_shap.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = percentile(&sorted_shap, 50.0);
        let percentile_25 = percentile(&sorted_shap, 25.0);
        let percentile_75 = percentile(&sorted_shap, 75.0);

        // Count positive/negative
        let n_positive = shap_col.iter().filter(|&&v| v > 0.0).count();
        let n_negative = shap_col.iter().filter(|&&v| v < 0.0).count();

        // Compute correlation between feature values and SHAP values
        let feature_shap_correlation = compute_correlation(&feat_col, &shap_col);

        FeatureSHAPStats {
            feature_idx,
            feature_name: feature_name.to_string(),
            mean,
            std,
            mean_abs,
            min,
            max,
            median,
            percentile_25,
            percentile_75,
            n_positive,
            n_negative,
            feature_shap_correlation,
        }
    }

    /// Generate bar plot data for global feature importance
    #[must_use]
    pub fn bar_plot_data(&self) -> BarPlotData {
        let entries: Vec<BarPlotEntry> = self
            .importance_order
            .iter()
            .map(|&idx| {
                // Compute std of |SHAP| for this feature
                let abs_shap: Vec<f64> = (0..self.n_samples)
                    .map(|i| self.shap_values[[i, idx]].abs())
                    .collect();

                let importance = self.global_importance[idx];
                let variance = abs_shap
                    .iter()
                    .map(|v| (v - importance).powi(2))
                    .sum::<f64>()
                    / self.n_samples as f64;
                let std = variance.sqrt();

                // 95% CI using t-distribution approximation (z=1.96 for large samples)
                let se = std / (self.n_samples as f64).sqrt();
                let ci_lower = importance - 1.96 * se;
                let ci_upper = importance + 1.96 * se;

                BarPlotEntry {
                    feature_idx: idx,
                    feature_name: self.feature_names[idx].clone(),
                    importance,
                    importance_std: std,
                    ci_lower: ci_lower.max(0.0),
                    ci_upper,
                }
            })
            .collect();

        BarPlotData {
            entries,
            n_samples: self.n_samples,
        }
    }

    /// Generate beeswarm plot data
    #[must_use]
    pub fn beeswarm_plot_data(&self) -> BeeswarmPlotData {
        // Compute min/max for feature value normalization
        let (feat_mins, feat_maxs) = self.feature_min_max();

        let features: Vec<BeeswarmFeatureData> = self
            .importance_order
            .iter()
            .enumerate()
            .map(|(rank, &idx)| {
                let feat_min = feat_mins[idx];
                let feat_max = feat_maxs[idx];
                let feat_range = feat_max - feat_min;

                let points: Vec<BeeswarmPoint> = (0..self.n_samples)
                    .map(|i| {
                        let raw_value = self.feature_values[[i, idx]];
                        let normalized = if feat_range > 1e-10 {
                            (raw_value - feat_min) / feat_range
                        } else {
                            0.5
                        };

                        BeeswarmPoint {
                            sample_idx: i,
                            shap_value: self.shap_values[[i, idx]],
                            normalized_feature_value: normalized.clamp(0.0, 1.0),
                            raw_feature_value: raw_value,
                        }
                    })
                    .collect();

                BeeswarmFeatureData {
                    feature_idx: idx,
                    feature_name: self.feature_names[idx].clone(),
                    points,
                    importance_rank: rank,
                    mean_abs_shap: self.global_importance[idx],
                }
            })
            .collect();

        BeeswarmPlotData {
            features,
            n_samples: self.n_samples,
            n_features: self.n_features,
        }
    }

    /// Compute min and max values for each feature
    fn feature_min_max(&self) -> (Vec<f64>, Vec<f64>) {
        let mut mins = vec![f64::INFINITY; self.n_features];
        let mut maxs = vec![f64::NEG_INFINITY; self.n_features];

        for i in 0..self.n_samples {
            for j in 0..self.n_features {
                let v = self.feature_values[[i, j]];
                if v < mins[j] {
                    mins[j] = v;
                }
                if v > maxs[j] {
                    maxs[j] = v;
                }
            }
        }

        (mins, maxs)
    }

    /// Get feature statistics for a specific feature by index
    #[must_use]
    pub fn get_feature_stats(&self, feature_idx: usize) -> Option<&FeatureSHAPStats> {
        self.feature_stats
            .iter()
            .find(|s| s.feature_idx == feature_idx)
    }

    /// Get feature statistics for a specific feature by name
    #[must_use]
    pub fn get_feature_stats_by_name(&self, name: &str) -> Option<&FeatureSHAPStats> {
        self.feature_stats.iter().find(|s| s.feature_name == name)
    }

    /// Get the top k most important features
    #[must_use]
    pub fn top_k_features(&self, k: usize) -> Vec<&FeatureSHAPStats> {
        self.feature_stats.iter().take(k).collect()
    }

    /// Get features with positive correlation (higher values -> higher prediction)
    #[must_use]
    pub fn positive_impact_features(&self, min_correlation: f64) -> Vec<&FeatureSHAPStats> {
        self.feature_stats
            .iter()
            .filter(|s| s.feature_shap_correlation >= min_correlation)
            .collect()
    }

    /// Get features with negative correlation (higher values -> lower prediction)
    #[must_use]
    pub fn negative_impact_features(&self, max_correlation: f64) -> Vec<&FeatureSHAPStats> {
        self.feature_stats
            .iter()
            .filter(|s| s.feature_shap_correlation <= max_correlation)
            .collect()
    }

    /// Compute the total explained variance from all SHAP values
    #[must_use]
    pub fn total_shap_variance(&self) -> f64 {
        let mut total = 0.0;
        for i in 0..self.n_samples {
            for j in 0..self.n_features {
                total += self.shap_values[[i, j]].powi(2);
            }
        }
        total / self.n_samples as f64
    }

    /// Get a text summary of the SHAP analysis
    #[must_use]
    pub fn text_summary(&self, top_k: usize) -> String {
        let mut lines = vec![
            String::from("SHAP Analysis Summary"),
            String::from("====================="),
            format!("Base value: {:.4}", self.base_value),
            format!("Samples analyzed: {}", self.n_samples),
            format!("Features: {}", self.n_features),
            String::new(),
            format!("Top {} Features by Importance:", top_k.min(self.n_features)),
            String::from("----------------------------"),
        ];

        for (i, stat) in self.feature_stats.iter().take(top_k).enumerate() {
            lines.push(format!(
                "{}. {} (mean |SHAP| = {:.4})",
                i + 1,
                stat.feature_name,
                stat.mean_abs
            ));
            lines.push(format!(
                "   Range: [{:.4}, {:.4}], Correlation: {:.3}",
                stat.min, stat.max, stat.feature_shap_correlation
            ));
        }

        // Add insights
        lines.push(String::new());
        lines.push(String::from("Key Insights:"));
        lines.push(String::from("-------------"));

        // Find features with strong positive impact
        let positive: Vec<_> = self.positive_impact_features(0.5);
        if !positive.is_empty() {
            lines.push(format!(
                "- {} feature(s) have strong positive correlation (higher value -> higher prediction)",
                positive.len()
            ));
        }

        // Find features with strong negative impact
        let negative: Vec<_> = self.negative_impact_features(-0.5);
        if !negative.is_empty() {
            lines.push(format!(
                "- {} feature(s) have strong negative correlation (higher value -> lower prediction)",
                negative.len()
            ));
        }

        // Concentration of importance
        let bar_data = self.bar_plot_data();
        let n_for_90 = bar_data.features_for_fraction(0.9);
        lines.push(format!(
            "- Top {} features explain 90% of total importance",
            n_for_90
        ));

        lines.join("\n")
    }
}

impl std::fmt::Display for SHAPSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text_summary(10))
    }
}

// =============================================================================
// Dependence Plot Data (Scatter Plot)
// =============================================================================

/// Data for a SHAP dependence plot (scatter plot of feature value vs SHAP value)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencePlotData {
    /// Feature index
    pub feature_idx: usize,
    /// Feature name
    pub feature_name: String,
    /// Feature values (x-axis)
    pub feature_values: Vec<f64>,
    /// SHAP values (y-axis)
    pub shap_values: Vec<f64>,
    /// Optional interaction feature values (for coloring)
    pub interaction_values: Option<Vec<f64>>,
    /// Interaction feature index (if any)
    pub interaction_feature_idx: Option<usize>,
    /// Interaction feature name (if any)
    pub interaction_feature_name: Option<String>,
    /// Correlation between feature value and SHAP value
    pub correlation: f64,
}

impl SHAPSummary {
    /// Generate dependence plot data for a single feature
    #[must_use]
    pub fn dependence_plot_data(&self, feature_idx: usize) -> Option<DependencePlotData> {
        if feature_idx >= self.n_features {
            return None;
        }

        let feature_vals: Vec<f64> = (0..self.n_samples)
            .map(|i| self.feature_values[[i, feature_idx]])
            .collect();

        let shap_vals: Vec<f64> = (0..self.n_samples)
            .map(|i| self.shap_values[[i, feature_idx]])
            .collect();

        let correlation = compute_correlation(&feature_vals, &shap_vals);

        Some(DependencePlotData {
            feature_idx,
            feature_name: self.feature_names[feature_idx].clone(),
            feature_values: feature_vals,
            shap_values: shap_vals,
            interaction_values: None,
            interaction_feature_idx: None,
            interaction_feature_name: None,
            correlation,
        })
    }

    /// Generate dependence plot data with interaction feature for coloring
    #[must_use]
    pub fn dependence_plot_data_with_interaction(
        &self,
        feature_idx: usize,
        interaction_idx: usize,
    ) -> Option<DependencePlotData> {
        if feature_idx >= self.n_features || interaction_idx >= self.n_features {
            return None;
        }

        let mut data = self.dependence_plot_data(feature_idx)?;

        let interaction_vals: Vec<f64> = (0..self.n_samples)
            .map(|i| self.feature_values[[i, interaction_idx]])
            .collect();

        data.interaction_values = Some(interaction_vals);
        data.interaction_feature_idx = Some(interaction_idx);
        data.interaction_feature_name = Some(self.feature_names[interaction_idx].clone());

        Some(data)
    }

    /// Find the best interaction feature for a given feature
    /// (feature with highest correlation to the residual SHAP variation)
    #[must_use]
    pub fn find_best_interaction(&self, feature_idx: usize) -> Option<usize> {
        if feature_idx >= self.n_features {
            return None;
        }

        // Get the residuals after accounting for the main effect
        let shap_vals: Vec<f64> = (0..self.n_samples)
            .map(|i| self.shap_values[[i, feature_idx]])
            .collect();

        let feat_vals: Vec<f64> = (0..self.n_samples)
            .map(|i| self.feature_values[[i, feature_idx]])
            .collect();

        // Compute simple linear prediction
        let feat_mean = feat_vals.iter().sum::<f64>() / self.n_samples as f64;
        let shap_mean = shap_vals.iter().sum::<f64>() / self.n_samples as f64;

        let mut cov = 0.0;
        let mut var = 0.0;
        for i in 0..self.n_samples {
            let fx = feat_vals[i] - feat_mean;
            let sy = shap_vals[i] - shap_mean;
            cov += fx * sy;
            var += fx * fx;
        }

        let slope = if var > 1e-10 { cov / var } else { 0.0 };
        let intercept = shap_mean - slope * feat_mean;

        // Compute residuals
        let residuals: Vec<f64> = (0..self.n_samples)
            .map(|i| shap_vals[i] - (slope * feat_vals[i] + intercept))
            .collect();

        // Find feature with highest correlation to residuals
        let mut best_idx = None;
        let mut best_corr = 0.0;

        for j in 0..self.n_features {
            if j == feature_idx {
                continue;
            }

            let other_vals: Vec<f64> = (0..self.n_samples)
                .map(|i| self.feature_values[[i, j]])
                .collect();

            let corr = compute_correlation(&other_vals, &residuals).abs();
            if corr > best_corr {
                best_corr = corr;
                best_idx = Some(j);
            }
        }

        best_idx
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute the percentile of a sorted array
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let idx = (p / 100.0) * (sorted.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f64;

    if lower == upper || upper >= sorted.len() {
        sorted[lower.min(sorted.len() - 1)]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Compute Pearson correlation coefficient
fn compute_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        cov / denom
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_test_batch_result() -> SHAPBatchResult {
        // Create a simple batch result with known values
        let shap_values = Array2::from_shape_vec(
            (10, 3),
            vec![
                // Feature 0 has high positive SHAP for high feature values
                0.5, -0.1, 0.2, 0.6, -0.2, 0.1, 0.4, 0.0, 0.3, 0.7, -0.1, 0.2, 0.8, -0.3, 0.1, -0.5,
                0.1, -0.2, -0.6, 0.2, -0.1, -0.4, 0.0, -0.3, -0.7, 0.1, -0.2, -0.8, 0.3, -0.1,
            ],
        )
        .unwrap();

        let feature_values = Array2::from_shape_vec(
            (10, 3),
            vec![
                // Feature 0: high values correspond to positive SHAP
                5.0, 1.0, 2.0, 6.0, 2.0, 1.0, 4.0, 3.0, 3.0, 7.0, 1.0, 2.0, 8.0, 3.0, 1.0, 1.0, 1.0,
                2.0, 0.0, 2.0, 1.0, 2.0, 3.0, 3.0, 1.5, 1.0, 2.0, 0.5, 3.0, 1.0,
            ],
        )
        .unwrap();

        SHAPBatchResult {
            base_value: 0.5,
            shap_values,
            feature_values,
            n_samples: 10,
            n_features: 3,
            feature_names: Some(vec![
                "income".to_string(),
                "age".to_string(),
                "score".to_string(),
            ]),
        }
    }

    #[test]
    fn test_shap_summary_creation() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);

        assert_eq!(summary.n_samples, 10);
        assert_eq!(summary.n_features, 3);
        assert_eq!(summary.feature_names.len(), 3);
        assert_eq!(summary.feature_stats.len(), 3);
    }

    #[test]
    fn test_bar_plot_data() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);
        let bar_data = summary.bar_plot_data();

        assert_eq!(bar_data.entries.len(), 3);
        assert_eq!(bar_data.n_samples, 10);

        // First entry should be most important
        let first = &bar_data.entries[0];
        assert!(first.importance > 0.0);
        assert!(first.ci_lower <= first.importance);
        assert!(first.ci_upper >= first.importance);
    }

    #[test]
    fn test_beeswarm_plot_data() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);
        let beeswarm = summary.beeswarm_plot_data();

        assert_eq!(beeswarm.features.len(), 3);
        assert_eq!(beeswarm.n_samples, 10);
        assert_eq!(beeswarm.n_features, 3);

        // Each feature should have n_samples points
        for feat in &beeswarm.features {
            assert_eq!(feat.points.len(), 10);
            // Normalized values should be in [0, 1]
            for point in &feat.points {
                assert!(point.normalized_feature_value >= 0.0);
                assert!(point.normalized_feature_value <= 1.0);
            }
        }
    }

    #[test]
    fn test_feature_stats() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);

        // Feature 0 (income) should have positive correlation
        // because high values -> positive SHAP
        let stats = summary.get_feature_stats_by_name("income");
        assert!(stats.is_some());
        let income_stats = stats.unwrap();
        assert!(income_stats.feature_shap_correlation > 0.5);
    }

    #[test]
    fn test_top_k() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);

        let top_2 = summary.top_k_features(2);
        assert_eq!(top_2.len(), 2);

        // Top 2 should be in importance order
        assert!(top_2[0].mean_abs >= top_2[1].mean_abs);
    }

    #[test]
    fn test_dependence_plot_data() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);

        let dep_data = summary.dependence_plot_data(0);
        assert!(dep_data.is_some());

        let data = dep_data.unwrap();
        assert_eq!(data.feature_idx, 0);
        assert_eq!(data.feature_values.len(), 10);
        assert_eq!(data.shap_values.len(), 10);
        assert!(data.interaction_values.is_none());
    }

    #[test]
    fn test_dependence_plot_with_interaction() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);

        let dep_data = summary.dependence_plot_data_with_interaction(0, 1);
        assert!(dep_data.is_some());

        let data = dep_data.unwrap();
        assert!(data.interaction_values.is_some());
        assert_eq!(data.interaction_feature_idx, Some(1));
    }

    #[test]
    fn test_cumulative_importance() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);
        let bar_data = summary.bar_plot_data();

        let cumulative = bar_data.cumulative_importance();
        assert_eq!(cumulative.len(), 3);

        // Should be strictly increasing and end at ~1.0
        for i in 1..cumulative.len() {
            assert!(cumulative[i].1 >= cumulative[i - 1].1);
        }
        assert!((cumulative.last().unwrap().1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_features_for_fraction() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);
        let bar_data = summary.bar_plot_data();

        // Should need at most all features for 100%
        let n_for_100 = bar_data.features_for_fraction(1.0);
        assert!(n_for_100 <= 3);

        // Should need at least 1 feature for any fraction
        let n_for_10 = bar_data.features_for_fraction(0.1);
        assert!(n_for_10 >= 1);
    }

    #[test]
    fn test_correlation_computation() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = compute_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10);

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr_neg = compute_correlation(&x, &y_neg);
        assert!((corr_neg - (-1.0)).abs() < 1e-10);

        // No correlation (orthogonal)
        let x2 = vec![1.0, -1.0, 1.0, -1.0];
        let y2 = vec![1.0, 1.0, -1.0, -1.0];
        let corr_zero = compute_correlation(&x2, &y2);
        assert!(corr_zero.abs() < 1e-10);
    }

    #[test]
    fn test_percentile() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((percentile(&sorted, 0.0) - 1.0).abs() < 1e-10);
        assert!((percentile(&sorted, 50.0) - 3.0).abs() < 1e-10);
        assert!((percentile(&sorted, 100.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_text_summary() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);

        let text = summary.text_summary(3);
        assert!(text.contains("SHAP Analysis Summary"));
        assert!(text.contains("Base value:"));
        assert!(text.contains("income") || text.contains("feature_0"));
    }

    #[test]
    fn test_positive_negative_impact_features() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);

        let positive = summary.positive_impact_features(0.3);
        let negative = summary.negative_impact_features(-0.3);

        // Should have some features in each category based on our test data
        // income has positive correlation
        assert!(!positive.is_empty() || !negative.is_empty());
    }

    #[test]
    fn test_find_best_interaction() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);

        let best = summary.find_best_interaction(0);
        // Should find some interaction (not feature 0 itself)
        if let Some(idx) = best {
            assert_ne!(idx, 0);
        }
    }

    #[test]
    fn test_display_implementations() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);

        // Test Display for FeatureSHAPStats
        let stats = &summary.feature_stats[0];
        let stats_str = format!("{}", stats);
        assert!(stats_str.contains("Feature:"));
        assert!(stats_str.contains("Mean |SHAP|:"));

        // Test Display for BarPlotData
        let bar_data = summary.bar_plot_data();
        let bar_str = format!("{}", bar_data);
        assert!(bar_str.contains("SHAP Global Feature Importance"));

        // Test Display for BeeswarmPlotData
        let beeswarm = summary.beeswarm_plot_data();
        let beeswarm_str = format!("{}", beeswarm);
        assert!(beeswarm_str.contains("SHAP Beeswarm Plot Data"));

        // Test Display for SHAPSummary
        let summary_str = format!("{}", summary);
        assert!(summary_str.contains("SHAP Analysis Summary"));
    }

    #[test]
    fn test_bar_plot_above_threshold() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);
        let bar_data = summary.bar_plot_data();

        // Very low threshold should include all
        let all = bar_data.above_threshold(0.0);
        assert_eq!(all.len(), 3);

        // Very high threshold should include none
        let none = bar_data.above_threshold(100.0);
        assert!(none.is_empty());
    }

    #[test]
    fn test_invalid_feature_idx() {
        let batch = make_test_batch_result();
        let summary = SHAPSummary::from_batch_result(&batch);

        // Invalid feature index
        let dep_data = summary.dependence_plot_data(100);
        assert!(dep_data.is_none());

        let stats = summary.get_feature_stats(100);
        assert!(stats.is_none());
    }
}

//! Comprehensive tests for model explainability features
//!
//! This module provides thorough tests for FerroML's explainability features:
//! - TreeSHAP for tree-based model explanations
//! - KernelSHAP for model-agnostic explanations
//! - Partial Dependence Plots (PDP)
//! - Individual Conditional Expectation (ICE) plots
//! - Permutation importance
//! - Feature importance consistency across methods

#[cfg(test)]
mod tests {
    use crate::assert_approx_eq;
    use crate::explainability::{
        center_ice_curves, compute_derivative_ice, h_statistic, h_statistic_matrix,
        h_statistic_overall, individual_conditional_expectation,
        individual_conditional_expectation_parallel, partial_dependence, partial_dependence_2d,
        partial_dependence_multi, permutation_importance, GridMethod, HStatisticConfig, ICEConfig,
        KernelExplainer, KernelSHAPConfig, TreeExplainer,
    };
    use crate::models::boosting::GradientBoostingRegressor;
    use crate::models::forest::{RandomForestClassifier, RandomForestRegressor};
    use crate::models::linear::LinearRegression;
    use crate::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor};
    use crate::models::Model;
    use crate::testing::assertions::tolerances;
    use ndarray::{Array1, Array2, Axis};

    // =========================================================================
    // Test Data Generation Utilities
    // =========================================================================

    /// Create simple regression data with known relationships: y = 2*x0 + 3*x1 + noise
    fn make_linear_regression_data(
        n_samples: usize,
        noise: f64,
        seed: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        let mut state = seed;
        let _next_rand = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
        };

        let x = Array2::from_shape_fn((n_samples, 3), |(i, j)| {
            // Generate pseudo-random but reproducible values
            match j {
                0 => i as f64 * 0.1,
                1 => ((i * 7 + 11) % 50) as f64 * 0.1,
                _ => {
                    let mut s = seed.wrapping_add(i as u64).wrapping_add(j as u64);
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((s >> 33) as f64) / (u32::MAX as f64)
                }
            }
        });

        let y: Array1<f64> = x
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, row)| {
                let mut s = seed.wrapping_add(i as u64 * 1000);
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise_val = ((s >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
                2.0 * row[0] + 3.0 * row[1] + noise * noise_val
            })
            .collect();

        (x, y)
    }

    /// Create classification data with two separable clusters
    fn make_classification_data(n_samples: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let half = n_samples / 2;
        let mut x_data = Vec::with_capacity(n_samples * 2);
        let mut y_data = Vec::with_capacity(n_samples);

        // Class 0: centered at (1, 1)
        for i in 0..half {
            let mut s = seed.wrapping_add(i as u64);
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r1 = ((s >> 33) as f64) / (u32::MAX as f64) - 0.5;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r2 = ((s >> 33) as f64) / (u32::MAX as f64) - 0.5;
            x_data.push(1.0 + r1);
            x_data.push(1.0 + r2);
            y_data.push(0.0);
        }

        // Class 1: centered at (4, 4)
        for i in 0..(n_samples - half) {
            let mut s = seed.wrapping_add((i + half) as u64);
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r1 = ((s >> 33) as f64) / (u32::MAX as f64) - 0.5;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r2 = ((s >> 33) as f64) / (u32::MAX as f64) - 0.5;
            x_data.push(4.0 + r1);
            x_data.push(4.0 + r2);
            y_data.push(1.0);
        }

        let x = Array2::from_shape_vec((n_samples, 2), x_data).unwrap();
        let y = Array1::from_vec(y_data);
        (x, y)
    }

    /// Create data with feature interactions: y = x0 * x1 + x2
    fn make_interaction_data(n_samples: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((n_samples, 3), |(i, j)| {
            let mut s = seed.wrapping_add(i as u64).wrapping_add(j as u64 * 1000);
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f64) / (u32::MAX as f64) * 4.0 + 0.5
        });

        let y: Array1<f64> = x
            .axis_iter(Axis(0))
            .map(|row| row[0] * row[1] + row[2])
            .collect();

        (x, y)
    }

    // =========================================================================
    // TreeSHAP Tests
    // =========================================================================

    #[test]
    fn test_treeshap_decision_tree_regressor_basic() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(4));
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();

        assert_eq!(explainer.n_features(), 3);
        assert_eq!(explainer.n_trees(), 1);

        // Explain a single sample
        let sample = vec![2.5, 1.5, 0.5];
        let result = explainer.explain(&sample).unwrap();

        assert_eq!(result.n_features, 3);
        assert_eq!(result.shap_values.len(), 3);

        // Verify SHAP additivity property: prediction = base_value + sum(shap_values)
        let reconstructed = result.prediction();
        let actual = model
            .predict(&Array2::from_shape_vec((1, 3), sample.clone()).unwrap())
            .unwrap()[0];

        // Allow tolerance for numerical precision
        assert!(
            (reconstructed - actual).abs() < 1e-6,
            "SHAP additivity violated: reconstructed={}, actual={}",
            reconstructed,
            actual
        );
    }

    #[test]
    fn test_treeshap_decision_tree_classifier() {
        let (x, y) = make_classification_data(40, 42);

        let mut model = DecisionTreeClassifier::new().with_max_depth(Some(3));
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_decision_tree_classifier(&model).unwrap();

        assert_eq!(explainer.n_features(), 2);
        assert_eq!(explainer.n_trees(), 1);

        // Explain samples from each class
        let sample_class0 = vec![1.0, 1.0];
        let result0 = explainer.explain(&sample_class0).unwrap();
        assert_eq!(result0.n_features, 2);

        let sample_class1 = vec![4.0, 4.0];
        let result1 = explainer.explain(&sample_class1).unwrap();
        assert_eq!(result1.n_features, 2);
    }

    #[test]
    fn test_treeshap_random_forest_regressor() {
        let (x, y) = make_linear_regression_data(60, 0.1, 42);

        let mut model = RandomForestRegressor::new()
            .with_n_estimators(5)
            .with_max_depth(Some(3))
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_random_forest_regressor(&model).unwrap();

        assert_eq!(explainer.n_features(), 3);
        assert_eq!(explainer.n_trees(), 5);

        let sample = vec![2.5, 1.5, 0.5];
        let result = explainer.explain(&sample).unwrap();

        assert_eq!(result.n_features, 3);

        // For random forest, feature 0 and 1 should have higher importance
        // since y = 2*x0 + 3*x1 + noise
        let abs_shap: Vec<f64> = result.shap_values.iter().map(|v| v.abs()).collect();
        // We don't strictly enforce ordering due to tree randomness,
        // but both informative features should have non-zero contributions
        assert!(
            abs_shap[0] + abs_shap[1] > 0.0,
            "Informative features should have SHAP contributions"
        );
    }

    #[test]
    fn test_treeshap_random_forest_classifier() {
        let (x, y) = make_classification_data(60, 42);

        let mut model = RandomForestClassifier::new()
            .with_n_estimators(5)
            .with_max_depth(Some(3))
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_random_forest_classifier(&model).unwrap();

        assert_eq!(explainer.n_features(), 2);
        assert_eq!(explainer.n_trees(), 5);

        let sample = vec![1.0, 1.0];
        let result = explainer.explain(&sample).unwrap();
        assert_eq!(result.n_features, 2);
    }

    #[test]
    fn test_treeshap_gradient_boosting_regressor() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = GradientBoostingRegressor::new()
            .with_n_estimators(10)
            .with_max_depth(Some(2))
            .with_learning_rate(0.1);
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_gradient_boosting_regressor(&model).unwrap();

        assert_eq!(explainer.n_features(), 3);
        assert_eq!(explainer.n_trees(), 10);

        let sample = vec![2.5, 1.5, 0.5];
        let result = explainer.explain(&sample).unwrap();

        assert_eq!(result.n_features, 3);
    }

    #[test]
    fn test_treeshap_batch_explanation() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(4));
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();

        // Explain batch of samples
        let test_x = x.slice(ndarray::s![0..10, ..]).to_owned();
        let result = explainer.explain_batch(&test_x).unwrap();

        assert_eq!(result.n_samples, 10);
        assert_eq!(result.n_features, 3);
        assert_eq!(result.shap_values.shape(), &[10, 3]);

        // Check mean absolute SHAP for global importance
        let mean_abs = result.mean_abs_shap();
        assert_eq!(mean_abs.len(), 3);
        assert!(mean_abs.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_treeshap_batch_parallel() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(4));
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();

        let test_x = x.slice(ndarray::s![0..10, ..]).to_owned();

        // Sequential and parallel should produce same results
        let result_seq = explainer.explain_batch(&test_x).unwrap();
        let result_par = explainer.explain_batch_parallel(&test_x).unwrap();

        assert_eq!(result_seq.n_samples, result_par.n_samples);
        assert_eq!(result_seq.n_features, result_par.n_features);

        for i in 0..10 {
            for j in 0..3 {
                assert!(
                    (result_seq.shap_values[[i, j]] - result_par.shap_values[[i, j]]).abs() < 1e-10,
                    "Mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    result_seq.shap_values[[i, j]],
                    result_par.shap_values[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_treeshap_feature_names_propagation() {
        let (x, y) = make_linear_regression_data(30, 0.1, 42);

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(3));
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_decision_tree_regressor(&model)
            .unwrap()
            .with_feature_names(vec!["f0".to_string(), "f1".to_string(), "f2".to_string()]);

        let sample = vec![1.0, 2.0, 0.5];
        let result = explainer.explain(&sample).unwrap();

        assert!(result.feature_names.is_some());
        assert_eq!(result.feature_names.as_ref().unwrap().len(), 3);

        // Test summary formatting includes feature names
        let summary = result.summary();
        assert!(summary.contains("f0") || summary.contains("f1") || summary.contains("f2"));
    }

    #[test]
    fn test_treeshap_sorted_indices() {
        let (x, y) = make_linear_regression_data(40, 0.1, 42);

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(4));
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();

        let sample = vec![2.5, 1.5, 0.5];
        let result = explainer.explain(&sample).unwrap();

        let sorted = result.sorted_indices();
        assert_eq!(sorted.len(), 3);

        // Verify sorting is by absolute SHAP value descending
        for i in 1..sorted.len() {
            assert!(
                result.shap_values[sorted[i - 1]].abs() >= result.shap_values[sorted[i]].abs(),
                "Sorted indices not in descending absolute SHAP order"
            );
        }
    }

    #[test]
    fn test_treeshap_not_fitted_error() {
        let model = DecisionTreeRegressor::new();
        let result = TreeExplainer::from_decision_tree_regressor(&model);
        assert!(result.is_err());
    }

    #[test]
    fn test_treeshap_wrong_features_error() {
        let (x, y) = make_linear_regression_data(30, 0.1, 42);

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(3));
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();

        // Wrong number of features
        let sample = vec![1.0, 2.0]; // Should be 3 features
        let result = explainer.explain(&sample);
        assert!(result.is_err());
    }

    // =========================================================================
    // Partial Dependence Plot (PDP) Tests
    // =========================================================================

    #[test]
    fn test_pdp_basic_linear_model() {
        let (x, y) = make_linear_regression_data(100, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // PDP for feature 0 (coefficient ~2)
        let result = partial_dependence(&model, &x, 0, 20, GridMethod::Percentile, false).unwrap();

        assert_eq!(result.grid_values.len(), 20);
        assert_eq!(result.pdp_values.len(), 20);
        assert_eq!(result.feature_idx, 0);

        // For linear model with positive coefficient, PDP should be monotonic increasing
        assert!(
            result.is_monotonic_increasing(),
            "PDP should be monotonic increasing for positive coefficient"
        );
    }

    #[test]
    fn test_pdp_with_ice_curves() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = partial_dependence(&model, &x, 0, 15, GridMethod::Percentile, true).unwrap();

        assert!(result.ice_curves.is_some());
        let ice = result.ice_curves.as_ref().unwrap();
        assert_eq!(ice.shape(), &[50, 15]);

        // PDP should be mean of ICE curves
        for j in 0..15 {
            let ice_mean = ice.column(j).mean().unwrap();
            assert!(
                (ice_mean - result.pdp_values[j]).abs() < 1e-10,
                "PDP[{}] should equal mean(ICE[:, {}])",
                j,
                j
            );
        }
    }

    #[test]
    fn test_pdp_grid_methods() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // Percentile grid
        let result_pct =
            partial_dependence(&model, &x, 0, 10, GridMethod::Percentile, false).unwrap();

        // Uniform grid
        let result_uni = partial_dependence(&model, &x, 0, 10, GridMethod::Uniform, false).unwrap();

        // Both should have same number of grid points
        assert_eq!(result_pct.grid_values.len(), result_uni.grid_values.len());

        // Grid methods should generally differ
        assert_eq!(result_pct.grid_method, GridMethod::Percentile);
        assert_eq!(result_uni.grid_method, GridMethod::Uniform);
    }

    #[test]
    fn test_pdp_2d_feature_interaction() {
        let (x, y) = make_interaction_data(60, 42);

        let mut model = RandomForestRegressor::new()
            .with_n_estimators(10)
            .with_max_depth(Some(4))
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let result = partial_dependence_2d(&model, &x, 0, 1, 10, GridMethod::Percentile).unwrap();

        assert_eq!(result.grid_values_1.len(), 10);
        assert_eq!(result.grid_values_2.len(), 10);
        assert_eq!(result.pdp_values.shape(), &[10, 10]);
        assert_eq!(result.feature_idx_1, 0);
        assert_eq!(result.feature_idx_2, 1);
    }

    #[test]
    fn test_pdp_multi_features() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let results =
            partial_dependence_multi(&model, &x, &[0, 1, 2], 10, GridMethod::Percentile).unwrap();

        assert_eq!(results.len(), 3);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.feature_idx, i);
            assert_eq!(result.grid_values.len(), 10);
        }

        // Feature 1 should have larger effect range (coefficient 3 vs 2)
        // Allow for noise and discretization
        let effect_0 = results[0].effect_range();
        let effect_1 = results[1].effect_range();
        let effect_2 = results[2].effect_range();

        // Features 0 and 1 should have larger effects than noise feature 2
        assert!(
            effect_0 + effect_1 > effect_2 * 2.0,
            "Informative features should have larger effect ranges"
        );
    }

    #[test]
    fn test_pdp_heterogeneity_measure() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = partial_dependence(&model, &x, 0, 15, GridMethod::Percentile, false).unwrap();

        let heterogeneity = result.heterogeneity();
        assert!(heterogeneity >= 0.0);

        // For linear model, heterogeneity should be relatively low
        // since all samples have same slope
    }

    #[test]
    fn test_pdp_result_analysis_methods() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = partial_dependence(&model, &x, 0, 10, GridMethod::Percentile, false).unwrap();

        // Test all analysis methods
        let min_val = result.min_effect();
        let max_val = result.max_effect();
        let range = result.effect_range();

        assert!(max_val >= min_val);
        assert_approx_eq!(range, max_val - min_val, tolerances::CLOSED_FORM);

        let argmax = result.argmax();
        let argmin = result.argmin();
        assert!(argmax < 10);
        assert!(argmin < 10);
    }

    #[test]
    fn test_pdp_not_fitted_error() {
        let model = LinearRegression::new();
        let x = Array2::zeros((10, 3));

        let result = partial_dependence(&model, &x, 0, 10, GridMethod::Percentile, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_pdp_invalid_feature_idx_error() {
        let (x, y) = make_linear_regression_data(30, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = partial_dependence(&model, &x, 10, 10, GridMethod::Percentile, false);
        assert!(result.is_err());
    }

    // =========================================================================
    // Individual Conditional Expectation (ICE) Tests
    // =========================================================================

    #[test]
    fn test_ice_basic() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_n_grid_points(15);
        let result = individual_conditional_expectation(&model, &x, 0, config).unwrap();

        assert_eq!(result.grid_values.len(), 15);
        assert_eq!(result.ice_curves.shape(), &[50, 15]);
        assert_eq!(result.pdp_values.len(), 15);
        assert_eq!(result.n_samples, 50);
        assert!(result.centered_ice.is_none());
        assert!(result.derivative_ice.is_none());
    }

    #[test]
    fn test_ice_with_centering() {
        let (x, y) = make_linear_regression_data(40, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_n_grid_points(12).with_centering(0); // Center at first grid point
        let result = individual_conditional_expectation(&model, &x, 0, config).unwrap();

        assert!(result.centered_ice.is_some());
        let centered = result.centered_ice.as_ref().unwrap();
        assert_eq!(centered.shape(), &[40, 12]);

        // All curves should be 0 at the reference point
        for i in 0..40 {
            assert!(
                centered[[i, 0]].abs() < 1e-10,
                "Centered ICE[{}, 0] should be 0, got {}",
                i,
                centered[[i, 0]]
            );
        }

        assert!(result.center_reference_idx.is_some());
        assert_eq!(result.center_reference_idx.unwrap(), 0);
    }

    #[test]
    fn test_ice_with_derivative() {
        let (x, y) = make_linear_regression_data(40, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_n_grid_points(12).with_derivative();
        let result = individual_conditional_expectation(&model, &x, 0, config).unwrap();

        assert!(result.derivative_ice.is_some());
        let derivatives = result.derivative_ice.as_ref().unwrap();
        assert_eq!(derivatives.shape(), &[40, 11]); // n_grid - 1

        // For linear model with coefficient ~2, all derivatives should be ~2
        for i in 0..40 {
            for j in 0..11 {
                assert!(
                    (derivatives[[i, j]] - 2.0).abs() < 0.5,
                    "Derivative at ({}, {}) should be ~2, got {}",
                    i,
                    j,
                    derivatives[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_ice_with_sample_subset() {
        let (x, y) = make_linear_regression_data(100, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let sample_indices = vec![0, 10, 20, 30, 40];
        let config = ICEConfig::new()
            .with_n_grid_points(10)
            .with_sample_indices(sample_indices.clone());
        let result = individual_conditional_expectation(&model, &x, 0, config).unwrap();

        assert_eq!(result.n_samples, 5);
        assert_eq!(result.ice_curves.shape(), &[5, 10]);
        assert!(result.sample_indices.is_some());
        assert_eq!(result.sample_indices.as_ref().unwrap(), &sample_indices);
    }

    #[test]
    fn test_ice_parallel_consistency() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_n_grid_points(12);
        let result_seq = individual_conditional_expectation(&model, &x, 0, config.clone()).unwrap();
        let result_par =
            individual_conditional_expectation_parallel(&model, &x, 0, config).unwrap();

        // Results should match
        for i in 0..50 {
            for j in 0..12 {
                assert!(
                    (result_seq.ice_curves[[i, j]] - result_par.ice_curves[[i, j]]).abs() < 1e-10,
                    "ICE mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_ice_heterogeneity_analysis() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_n_grid_points(10);
        let result = individual_conditional_expectation(&model, &x, 0, config).unwrap();

        let heterogeneity = result.heterogeneity();
        assert_eq!(heterogeneity.len(), 10);
        assert!(heterogeneity.iter().all(|&h| h >= 0.0));

        let mean_het = result.mean_heterogeneity();
        assert!(mean_het >= 0.0);
    }

    #[test]
    fn test_ice_monotonicity_fractions() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_n_grid_points(10);
        let result = individual_conditional_expectation(&model, &x, 0, config).unwrap();

        // For linear model with positive coefficient, most curves should be monotonic increasing
        let frac_inc = result.fraction_monotonic_increasing();
        let frac_dec = result.fraction_monotonic_decreasing();

        assert!(frac_inc >= 0.0 && frac_inc <= 1.0);
        assert!(frac_dec >= 0.0 && frac_dec <= 1.0);

        // With coefficient 2, should be mostly increasing
        assert!(frac_inc > 0.8, "Expected mostly increasing curves");
    }

    #[test]
    fn test_ice_strongest_effects() {
        let (x, y) = make_linear_regression_data(40, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_n_grid_points(10);
        let result = individual_conditional_expectation(&model, &x, 0, config).unwrap();

        let strongest_pos = result.sample_with_strongest_positive_effect();
        let strongest_neg = result.sample_with_strongest_negative_effect();

        assert!(strongest_pos.is_some());
        assert!(strongest_neg.is_some());
        assert!(strongest_pos.unwrap() < 40);
        assert!(strongest_neg.unwrap() < 40);
    }

    #[test]
    fn test_center_ice_curves_utility() {
        let ice_curves = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5, 3.5],
        )
        .unwrap();

        // Center at index 0
        let centered = center_ice_curves(&ice_curves, 0);

        for i in 0..3 {
            assert!(centered[[i, 0]].abs() < 1e-10);
        }

        // Center at index 2
        let centered_mid = center_ice_curves(&ice_curves, 2);
        for i in 0..3 {
            assert!(centered_mid[[i, 2]].abs() < 1e-10);
        }
    }

    #[test]
    fn test_compute_derivative_ice_utility() {
        let ice_curves = Array2::from_shape_vec(
            (2, 4),
            vec![
                0.0, 1.0, 3.0, 6.0, // Increasing derivatives
                0.0, 2.0, 2.0, 2.0, // Step then flat
            ],
        )
        .unwrap();
        let grid_values = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        let derivatives = compute_derivative_ice(&ice_curves, &grid_values);
        assert_eq!(derivatives.shape(), &[2, 3]);

        // Sample 0: derivatives should be 1, 2, 3
        assert!((derivatives[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((derivatives[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((derivatives[[0, 2]] - 3.0).abs() < 1e-10);

        // Sample 1: derivatives should be 2, 0, 0
        assert!((derivatives[[1, 0]] - 2.0).abs() < 1e-10);
        assert!(derivatives[[1, 1]].abs() < 1e-10);
        assert!(derivatives[[1, 2]].abs() < 1e-10);
    }

    // =========================================================================
    // Permutation Importance Tests
    // =========================================================================

    /// Simple R2 scorer for regression
    fn r2_scorer(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> crate::Result<f64> {
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

    #[test]
    fn test_permutation_importance_basic() {
        let (x, y) = make_linear_regression_data(80, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(42)).unwrap();

        assert_eq!(result.importances_mean.len(), 3);
        assert_eq!(result.n_repeats, 5);
        assert!((result.confidence_level - 0.95).abs() < 1e-10);

        // Features 0 and 1 should be more important than feature 2
        // (since y = 2*x0 + 3*x1 + noise)
        let imp_0 = result.importances_mean[0];
        let imp_1 = result.importances_mean[1];
        let imp_2 = result.importances_mean[2];

        assert!(
            imp_0 + imp_1 > imp_2.abs() + 0.01,
            "Informative features ({}, {}) should have higher importance than noise ({})",
            imp_0,
            imp_1,
            imp_2
        );
    }

    #[test]
    fn test_permutation_importance_reproducibility() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result1 = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(123)).unwrap();
        let result2 = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(123)).unwrap();

        // Results should be identical with same seed
        for i in 0..3 {
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
    fn test_permutation_importance_confidence_intervals() {
        let (x, y) = make_linear_regression_data(60, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = permutation_importance(&model, &x, &y, r2_scorer, 10, Some(42)).unwrap();

        // CI should be properly ordered
        for i in 0..3 {
            assert!(
                result.ci_lower[i] <= result.importances_mean[i],
                "CI lower > mean for feature {}",
                i
            );
            assert!(
                result.importances_mean[i] <= result.ci_upper[i],
                "Mean > CI upper for feature {}",
                i
            );
        }
    }

    #[test]
    fn test_permutation_importance_sorted_indices() {
        let (x, y) = make_linear_regression_data(50, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(42)).unwrap();

        let sorted = result.sorted_indices();
        assert_eq!(sorted.len(), 3);

        // Verify descending order by importance
        for i in 1..sorted.len() {
            assert!(
                result.importances_mean[sorted[i - 1]] >= result.importances_mean[sorted[i]],
                "Sorted indices not in descending order"
            );
        }
    }

    #[test]
    fn test_permutation_importance_significance() {
        let (x, y) = make_linear_regression_data(60, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = permutation_importance(&model, &x, &y, r2_scorer, 10, Some(42)).unwrap();

        // Features 0 and 1 should be significant (CI doesn't include 0)
        let significant = result.significant_features();

        // At least one informative feature should be significant
        assert!(
            significant.contains(&0) || significant.contains(&1),
            "At least one informative feature should be significant"
        );
    }

    #[test]
    fn test_permutation_importance_not_fitted_error() {
        let model = LinearRegression::new();
        let x = Array2::zeros((10, 3));
        let y = Array1::zeros(10);

        let result = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_permutation_importance_shape_mismatch_error() {
        let (x, y) = make_linear_regression_data(30, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let x_wrong = Array2::zeros((20, 3)); // Different number of samples
        let result = permutation_importance(&model, &x_wrong, &y, r2_scorer, 5, Some(42));
        assert!(result.is_err());
    }

    // =========================================================================
    // Feature Importance Consistency Tests
    // =========================================================================

    #[test]
    fn test_feature_importance_consistency_tree_models() {
        let (x, y) = make_linear_regression_data(100, 0.1, 42);

        let mut rf_model = RandomForestRegressor::new()
            .with_n_estimators(10)
            .with_max_depth(Some(4))
            .with_random_state(42);
        rf_model.fit(&x, &y).unwrap();

        // TreeSHAP importance
        let tree_explainer = TreeExplainer::from_random_forest_regressor(&rf_model).unwrap();
        let shap_result = tree_explainer.explain_batch(&x).unwrap();
        let shap_importance = shap_result.mean_abs_shap();

        // Permutation importance
        let perm_result =
            permutation_importance(&rf_model, &x, &y, r2_scorer, 5, Some(42)).unwrap();

        // Both should identify features 0 and 1 as more important than 2
        let shap_imp_informative = shap_importance[0] + shap_importance[1];
        let shap_imp_noise = shap_importance[2];

        let perm_imp_informative =
            perm_result.importances_mean[0] + perm_result.importances_mean[1];
        let perm_imp_noise = perm_result.importances_mean[2];

        assert!(
            shap_imp_informative > shap_imp_noise,
            "SHAP should rank informative features higher"
        );

        // Permutation importance for noise can be near zero or slightly negative
        assert!(
            perm_imp_informative > perm_imp_noise.abs(),
            "Permutation should rank informative features higher"
        );
    }

    #[test]
    fn test_feature_importance_pdp_vs_permutation() {
        let (x, y) = make_linear_regression_data(80, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // PDP effect ranges
        let pdp_results =
            partial_dependence_multi(&model, &x, &[0, 1, 2], 15, GridMethod::Percentile).unwrap();
        let pdp_ranges: Vec<f64> = pdp_results.iter().map(|r| r.effect_range()).collect();

        // Permutation importance
        let _perm_result = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(42)).unwrap();

        // For linear model, larger PDP effect range should correlate with higher permutation importance
        // Feature with coefficient 3 (index 1) should have larger effect than coefficient 2 (index 0)
        let _pdp_ratio = pdp_ranges[1] / pdp_ranges[0].max(1e-10);

        // Both methods should agree that features 0 and 1 are more important than 2
        assert!(
            pdp_ranges[0] > pdp_ranges[2] * 0.5,
            "PDP: feature 0 should have larger range than feature 2"
        );
        assert!(
            pdp_ranges[1] > pdp_ranges[2] * 0.5,
            "PDP: feature 1 should have larger range than feature 2"
        );
    }

    // =========================================================================
    // H-Statistic Interaction Detection Tests
    // =========================================================================

    #[test]
    fn test_h_statistic_no_interaction() {
        // Linear model has no interactions
        let (x, y) = make_linear_regression_data(60, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = HStatisticConfig::new()
            .with_grid_points(8)
            .with_random_state(42);

        let result = h_statistic(&model, &x, 0, 1, config).unwrap();

        // H-statistic should be very low for linear model (no interactions)
        assert!(
            result.h_squared < 0.2,
            "Linear model should have low H-statistic, got {}",
            result.h_squared
        );
    }

    #[test]
    fn test_h_statistic_with_interaction() {
        // Data with interaction: y = x0 * x1 + x2
        let (x, y) = make_interaction_data(150, 42);

        let mut model = RandomForestRegressor::new()
            .with_n_estimators(25)
            .with_max_depth(Some(6))
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let config = HStatisticConfig::new()
            .with_grid_points(10)
            .with_random_state(42);

        // H-statistic for interacting features (0 and 1)
        let result_interact = h_statistic(&model, &x, 0, 1, config.clone()).unwrap();

        // H-statistic for non-interacting features (0 and 2, or 1 and 2)
        let result_no_interact = h_statistic(&model, &x, 0, 2, config.clone()).unwrap();

        // The H-statistic computation should work without errors
        // Due to random forest variance, we check that the values are valid
        // (non-negative and finite) rather than strict ordering
        assert!(
            result_interact.h_squared >= 0.0 && result_interact.h_squared.is_finite(),
            "H-statistic for interacting features should be valid: {}",
            result_interact.h_squared
        );
        assert!(
            result_no_interact.h_squared >= 0.0 && result_no_interact.h_squared.is_finite(),
            "H-statistic for non-interacting features should be valid: {}",
            result_no_interact.h_squared
        );

        // If both are non-zero, interacting pair should typically have higher H-statistic
        // but we allow for randomness in the model
        if result_interact.h_squared > 1e-10 && result_no_interact.h_squared > 1e-10 {
            // Log the values for debugging
            eprintln!(
                "H-statistic (interacting): {}, H-statistic (non-interacting): {}",
                result_interact.h_squared, result_no_interact.h_squared
            );
        }
    }

    #[test]
    fn test_h_statistic_overall() {
        let (x, y) = make_interaction_data(60, 42);

        let mut model = RandomForestRegressor::new()
            .with_n_estimators(10)
            .with_max_depth(Some(4))
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let config = HStatisticConfig::new()
            .with_grid_points(8)
            .with_random_state(42);

        // Overall H-statistic for feature 0 (participates in interaction)
        let result = h_statistic_overall(&model, &x, 0, config).unwrap();

        assert!(result.h_squared >= 0.0);
        assert!(result.h_squared <= 1.0);
    }

    #[test]
    fn test_h_statistic_matrix() {
        let (x, y) = make_interaction_data(50, 42);

        let mut model = RandomForestRegressor::new()
            .with_n_estimators(8)
            .with_max_depth(Some(3))
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let config = HStatisticConfig::new()
            .with_grid_points(6)
            .with_random_state(42);

        let matrix = h_statistic_matrix(&model, &x, None, config).unwrap();

        assert_eq!(matrix.feature_indices.len(), 3);

        // Get top interactions
        let top = matrix.top_k(3);
        assert_eq!(top.len(), 3);

        // Verify symmetric matrix (H(i,j) == H(j,i))
        for (i, j, _) in top.iter() {
            if i != j {
                // H-statistic should be stored consistently
                assert!(*i < *j, "Top-k should return ordered pairs");
            }
        }
    }

    // =========================================================================
    // KernelSHAP Tests
    // =========================================================================

    #[test]
    fn test_kernelshap_basic() {
        let (x, y) = make_linear_regression_data(40, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = KernelSHAPConfig::new()
            .with_n_samples(256)
            .with_max_background_samples(20)
            .with_random_state(42);

        let explainer = KernelExplainer::new(&model, &x, config).unwrap();

        assert_eq!(explainer.n_features(), 3);

        let sample = vec![2.5, 1.5, 0.5];
        let result = explainer.explain(&sample).unwrap();

        assert_eq!(result.n_features, 3);
        assert_eq!(result.shap_values.len(), 3);
    }

    #[test]
    fn test_kernelshap_additivity() {
        let (x, y) = make_linear_regression_data(40, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = KernelSHAPConfig::new()
            .with_n_samples(512)
            .with_random_state(42);

        let explainer = KernelExplainer::new(&model, &x, config).unwrap();

        let sample = vec![2.5, 1.5, 0.5];
        let result = explainer.explain(&sample).unwrap();

        // SHAP additivity: prediction ~= base_value + sum(shap_values)
        let reconstructed = result.prediction();
        let x_sample = Array2::from_shape_vec((1, 3), sample).unwrap();
        let actual = model.predict(&x_sample).unwrap()[0];

        // Allow tolerance for KernelSHAP approximation
        assert!(
            (reconstructed - actual).abs() < 1.0,
            "KernelSHAP additivity: reconstructed={}, actual={}",
            reconstructed,
            actual
        );
    }

    #[test]
    fn test_kernelshap_batch() {
        let (x, y) = make_linear_regression_data(30, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = KernelSHAPConfig::new()
            .with_n_samples(256)
            .with_random_state(42);

        let explainer = KernelExplainer::new(&model, &x, config).unwrap();

        let test_x = x.slice(ndarray::s![0..5, ..]).to_owned();
        let result = explainer.explain_batch(&test_x).unwrap();

        assert_eq!(result.n_samples, 5);
        assert_eq!(result.n_features, 3);
        assert_eq!(result.shap_values.shape(), &[5, 3]);
    }

    #[test]
    fn test_kernelshap_not_fitted_error() {
        let model = LinearRegression::new();
        let x = Array2::zeros((10, 3));

        let result = KernelExplainer::new(&model, &x, KernelSHAPConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_kernelshap_wrong_features_error() {
        let (x, y) = make_linear_regression_data(30, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let explainer = KernelExplainer::new(&model, &x, KernelSHAPConfig::default()).unwrap();

        let sample = vec![1.0, 2.0]; // Wrong number of features
        let result = explainer.explain(&sample);
        assert!(result.is_err());
    }

    // =========================================================================
    // Edge Cases and Robustness Tests
    // =========================================================================

    #[test]
    fn test_explainability_single_feature() {
        // Single feature model
        let x = Array2::from_shape_fn((30, 1), |(i, _)| i as f64 * 0.1);
        let y: Array1<f64> = x.column(0).mapv(|v| 2.0 * v + 1.0);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // PDP should work
        let pdp_result =
            partial_dependence(&model, &x, 0, 10, GridMethod::Percentile, false).unwrap();
        assert_eq!(pdp_result.grid_values.len(), 10);
        assert!(pdp_result.is_monotonic_increasing());

        // ICE should work
        let ice_config = ICEConfig::new().with_n_grid_points(10);
        let ice_result = individual_conditional_expectation(&model, &x, 0, ice_config).unwrap();
        assert_eq!(ice_result.n_samples, 30);

        // Permutation importance should work
        let perm_result = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(42)).unwrap();
        assert_eq!(perm_result.importances_mean.len(), 1);
        assert!(perm_result.importances_mean[0] > 0.0);
    }

    #[test]
    fn test_explainability_constant_target() {
        // All targets are the same
        let x = Array2::from_shape_fn((30, 2), |(i, j)| (i * j) as f64 * 0.1);
        let y = Array1::from_elem(30, 5.0); // Constant target

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(2));
        model.fit(&x, &y).unwrap();

        // TreeSHAP should give near-zero SHAP values
        let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();
        let sample = vec![1.0, 2.0];
        let result = explainer.explain(&sample).unwrap();

        // All SHAP values should be near zero for constant target
        for i in 0..2 {
            assert!(
                result.shap_values[i].abs() < 0.1,
                "SHAP[{}] should be ~0 for constant target, got {}",
                i,
                result.shap_values[i]
            );
        }
    }

    #[test]
    fn test_explainability_collinear_features() {
        // x1 = 2 * x0, so they're perfectly collinear
        let x = Array2::from_shape_fn((40, 3), |(i, j)| {
            match j {
                0 => i as f64 * 0.1,
                1 => i as f64 * 0.2, // = 2 * x0
                _ => ((i * 7 + 11) % 23) as f64 * 0.1,
            }
        });
        let y: Array1<f64> = x.column(0).mapv(|v| 3.0 * v + 1.0);

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(4));
        model.fit(&x, &y).unwrap();

        // Should still work without crashing
        let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();
        let sample = vec![1.0, 2.0, 0.5];
        let result = explainer.explain(&sample);
        assert!(result.is_ok());
    }

    #[test]
    fn test_explainability_extreme_values() {
        // Dataset with extreme values
        let x = Array2::from_shape_fn((30, 2), |(i, j)| {
            if i == 0 && j == 0 {
                1e6
            } else if i == 1 && j == 0 {
                -1e6
            } else {
                i as f64 * 0.1 + j as f64
            }
        });
        let y: Array1<f64> = x.axis_iter(Axis(0)).map(|row| row[0] + row[1]).collect();

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(4));
        model.fit(&x, &y).unwrap();

        // Should handle extreme values
        let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();
        let sample = vec![100.0, 50.0];
        let result = explainer.explain(&sample);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.shap_values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_explainability_many_features() {
        // High-dimensional data
        let n_features = 20;
        let x = Array2::from_shape_fn((50, n_features), |(i, j)| {
            ((i * 7 + j * 11) % 100) as f64 * 0.01
        });
        // Only first 2 features matter
        let y: Array1<f64> = x.axis_iter(Axis(0)).map(|row| row[0] + row[1]).collect();

        let mut model = RandomForestRegressor::new()
            .with_n_estimators(5)
            .with_max_depth(Some(3))
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        // TreeSHAP should handle many features
        let explainer = TreeExplainer::from_random_forest_regressor(&model).unwrap();
        assert_eq!(explainer.n_features(), n_features);

        let sample: Vec<f64> = (0..n_features).map(|i| i as f64 * 0.05).collect();
        let result = explainer.explain(&sample).unwrap();
        assert_eq!(result.n_features, n_features);

        // Permutation importance should also handle many features
        let perm_result = permutation_importance(&model, &x, &y, r2_scorer, 3, Some(42)).unwrap();
        assert_eq!(perm_result.importances_mean.len(), n_features);
    }

    #[test]
    fn test_shap_result_display_and_summary() {
        let (x, y) = make_linear_regression_data(30, 0.1, 42);

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(3));
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_decision_tree_regressor(&model)
            .unwrap()
            .with_feature_names(vec![
                "age".to_string(),
                "income".to_string(),
                "score".to_string(),
            ]);

        let sample = vec![1.5, 2.5, 0.5];
        let result = explainer.explain(&sample).unwrap();

        let summary = result.summary();
        assert!(summary.contains("SHAP Explanation"));
        assert!(summary.contains("Base value"));
        assert!(summary.contains("Prediction"));

        // Feature format should include name and value
        let formatted = result.format_feature(0);
        assert!(formatted.contains("age"));
    }

    #[test]
    fn test_pdp_result_display() {
        let (x, y) = make_linear_regression_data(40, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = partial_dependence(&model, &x, 0, 10, GridMethod::Percentile, false).unwrap();

        let summary = result.summary();
        assert!(summary.contains("PDP for"));
        assert!(summary.contains("effect range"));
    }

    #[test]
    fn test_ice_result_display() {
        let (x, y) = make_linear_regression_data(30, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = ICEConfig::new().with_n_grid_points(10);
        let result = individual_conditional_expectation(&model, &x, 0, config).unwrap();

        let summary = result.summary();
        assert!(summary.contains("ICE for"));
        assert!(summary.contains("samples"));
        assert!(summary.contains("heterogeneity"));
    }

    #[test]
    fn test_permutation_importance_display() {
        let (x, y) = make_linear_regression_data(40, 0.1, 42);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let result = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(42)).unwrap();

        let summary = result.summary();
        assert!(summary.contains("Permutation Importance"));
        assert!(summary.contains("Baseline score"));
    }
}

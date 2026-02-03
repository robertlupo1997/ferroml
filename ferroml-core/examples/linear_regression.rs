//! Linear Regression with Full Statistical Diagnostics
//!
//! This example demonstrates FerroML's key differentiator: comprehensive
//! statistical diagnostics that go beyond sklearn's basic fit/predict.
//!
//! Run with: `cargo run --example linear_regression`

use ferroml_core::models::linear::LinearRegression;
use ferroml_core::models::{Model, ProbabilisticModel, StatisticalModel};
use ndarray::{Array1, Array2};

fn main() -> ferroml_core::Result<()> {
    println!("=============================================================");
    println!("FerroML Linear Regression - Full Statistical Diagnostics Demo");
    println!("=============================================================\n");

    // Generate synthetic data with known properties
    // True model: y = 5 + 2*x1 + 3*x2 + noise
    let n_samples = 50;
    let (x, y) = generate_data(n_samples);

    // Create and fit the model with feature names
    let mut model = LinearRegression::new()
        .with_confidence_level(0.95)
        .with_feature_names(vec!["education_years".into(), "experience_years".into()]);

    model.fit(&x, &y)?;

    // =========================================================================
    // 1. R-Style Summary Output
    // =========================================================================
    println!("1. MODEL SUMMARY (R-style output)");
    println!("---------------------------------");
    println!("{}", model.summary());

    // =========================================================================
    // 2. Coefficient Analysis
    // =========================================================================
    println!("\n2. COEFFICIENT ANALYSIS");
    println!("-----------------------");

    let coefs = model.coefficients_with_ci(0.95);
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "Variable", "Estimate", "Std.Error", "t-value", "p-value", "Signif"
    );
    println!("{}", "-".repeat(80));

    for coef in &coefs {
        let sig = if coef.p_value < 0.001 {
            "***"
        } else if coef.p_value < 0.01 {
            "**"
        } else if coef.p_value < 0.05 {
            "*"
        } else if coef.p_value < 0.1 {
            "."
        } else {
            ""
        };

        println!(
            "{:<20} {:>12.4} {:>12.4} {:>12.4} {:>12.6} {:>10}",
            coef.name, coef.estimate, coef.std_error, coef.t_statistic, coef.p_value, sig
        );
    }
    println!("\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");

    println!("\n95% Confidence Intervals:");
    println!("{:<20} [{:>12}, {:>12}]", "Variable", "Lower", "Upper");
    println!("{}", "-".repeat(50));
    for coef in &coefs {
        println!(
            "{:<20} [{:>12.4}, {:>12.4}]",
            coef.name, coef.ci_lower, coef.ci_upper
        );
    }

    // =========================================================================
    // 3. Model Fit Statistics
    // =========================================================================
    println!("\n3. MODEL FIT STATISTICS");
    println!("-----------------------");

    if let Some(r2) = model.r_squared() {
        println!("R-squared:            {:.6}", r2);
    }
    if let Some(adj_r2) = model.adjusted_r_squared() {
        println!("Adjusted R-squared:   {:.6}", adj_r2);
    }
    if let Some((f_stat, f_p)) = model.f_statistic() {
        println!("F-statistic:          {:.4} (p-value: {:.6})", f_stat, f_p);
        if f_p < 0.05 {
            println!("  -> Model is statistically significant at alpha=0.05");
        }
    }
    if let Some(ll) = model.log_likelihood() {
        println!("Log-likelihood:       {:.4}", ll);
    }
    if let Some(aic) = model.aic() {
        println!("AIC:                  {:.4}", aic);
    }
    if let Some(bic) = model.bic() {
        println!("BIC:                  {:.4}", bic);
    }

    // =========================================================================
    // 4. Diagnostic Tests (Assumption Checking)
    // =========================================================================
    println!("\n4. DIAGNOSTIC TESTS (Assumption Checking)");
    println!("-----------------------------------------");

    let diagnostics = model.diagnostics();

    // Residual Statistics
    println!("\nResidual Statistics:");
    println!("  Mean:     {:.6}", diagnostics.residual_stats.mean);
    println!("  Std Dev:  {:.6}", diagnostics.residual_stats.std_dev);
    println!("  Min:      {:.6}", diagnostics.residual_stats.min);
    println!("  Max:      {:.6}", diagnostics.residual_stats.max);
    println!("  Skewness: {:.6}", diagnostics.residual_stats.skewness);
    println!("  Kurtosis: {:.6}", diagnostics.residual_stats.kurtosis);

    // Assumption Tests
    println!("\nAssumption Tests:");
    for test in &diagnostics.assumption_tests {
        let status = if test.passed { "PASS" } else { "FAIL" };
        println!(
            "  {:?}: {} = {:.4}, p = {:.4} [{}]",
            test.assumption, test.test_name, test.statistic, test.p_value, status
        );
    }

    // Durbin-Watson
    if let Some(dw) = diagnostics.durbin_watson {
        println!("\nDurbin-Watson statistic: {:.4}", dw);
        if dw < 1.5 {
            println!("  -> Warning: Possible positive autocorrelation (DW < 1.5)");
        } else if dw > 2.5 {
            println!("  -> Warning: Possible negative autocorrelation (DW > 2.5)");
        } else {
            println!("  -> No significant autocorrelation detected (1.5 < DW < 2.5)");
        }
    }

    // =========================================================================
    // 5. Influential Observations
    // =========================================================================
    println!("\n5. INFLUENTIAL OBSERVATIONS");
    println!("---------------------------");

    if let Some(cooks) = model.cooks_distance() {
        let threshold = 4.0 / n_samples as f64;
        println!("Cook's Distance threshold (4/n): {:.4}", threshold);

        let high_influence: Vec<_> = cooks
            .iter()
            .enumerate()
            .filter(|(_, &d)| d > threshold)
            .collect();

        if high_influence.is_empty() {
            println!("No observations exceed Cook's distance threshold.");
        } else {
            println!("\nObservations with high influence:");
            println!("{:<10} {:>15}", "Index", "Cook's D");
            for (idx, &d) in &high_influence {
                println!("{:<10} {:>15.6}", idx, d);
            }
        }
    }

    if !diagnostics.influential_observations.is_empty() {
        println!("\nDetailed influential observations:");
        println!(
            "{:<8} {:>12} {:>12} {:>12} {:>12}",
            "Index", "Cook's D", "Leverage", "Stud.Resid", "DFFITS"
        );
        for obs in &diagnostics.influential_observations {
            let stud_str = obs
                .studentized_residual
                .map_or("N/A".to_string(), |s| format!("{:.4}", s));
            let dffits_str = obs
                .dffits
                .map_or("N/A".to_string(), |d| format!("{:.4}", d));
            println!(
                "{:<8} {:>12.4} {:>12.4} {:>12} {:>12}",
                obs.index, obs.cooks_distance, obs.leverage, stud_str, dffits_str
            );
        }
    }

    // =========================================================================
    // 6. Feature Importance
    // =========================================================================
    println!("\n6. FEATURE IMPORTANCE (|t-statistic|)");
    println!("-------------------------------------");

    if let Some(importance) = model.feature_importance() {
        let feature_names = ["education_years", "experience_years"];
        let mut pairs: Vec<_> = feature_names.iter().zip(importance.iter()).collect();
        pairs.sort_by(|a, b| {
            b.1.partial_cmp(a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (name, &imp) in pairs {
            let bar_len = (imp * 2.0).clamp(0.0, 40.0) as usize;
            let bar = "█".repeat(bar_len);
            println!("{:<20} {:>8.4} {}", name, imp, bar);
        }
    }

    // =========================================================================
    // 7. Prediction with Intervals
    // =========================================================================
    println!("\n7. PREDICTION WITH INTERVALS");
    println!("----------------------------");

    // Predict for new data points
    let x_new = Array2::from_shape_vec(
        (3, 2),
        vec![
            12.0, 5.0, // 12 years education, 5 years experience
            16.0, 10.0, // 16 years education, 10 years experience
            20.0, 15.0, // 20 years education, 15 years experience
        ],
    )
    .expect("Valid shape for new data");

    let intervals = model.predict_interval(&x_new, 0.95)?;

    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>12}",
        "Obs", "Prediction", "Std.Error", "Lower 95%", "Upper 95%"
    );
    println!("{}", "-".repeat(60));

    for i in 0..x_new.nrows() {
        let se = intervals.std_errors.as_ref().map(|s| s[i]).unwrap_or(0.0);
        println!(
            "{:<10} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            i + 1,
            intervals.predictions[i],
            se,
            intervals.lower[i],
            intervals.upper[i]
        );
    }

    // Calculate interval width statistics
    let widths = intervals.interval_widths();
    let mean_width = widths.mean().unwrap_or(0.0);
    println!("\nInterval widths: mean = {:.4}", mean_width);

    // =========================================================================
    // 8. Residual Analysis
    // =========================================================================
    println!("\n8. RESIDUAL ANALYSIS");
    println!("--------------------");

    if let (Some(residuals), Some(fitted)) = (model.residuals(), model.fitted_values()) {
        // Show first few residuals
        println!("\nFirst 10 observations:");
        println!(
            "{:<6} {:>12} {:>12} {:>12}",
            "Obs", "Actual", "Fitted", "Residual"
        );
        println!("{}", "-".repeat(45));

        for i in 0..10.min(residuals.len()) {
            println!(
                "{:<6} {:>12.4} {:>12.4} {:>12.4}",
                i + 1,
                y[i],
                fitted[i],
                residuals[i]
            );
        }

        // Residual quartiles
        let mut sorted_resid: Vec<f64> = residuals.iter().copied().collect();
        sorted_resid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted_resid.len();

        println!("\nResidual Quartiles:");
        println!("  Min:    {:>12.4}", sorted_resid[0]);
        println!("  Q1:     {:>12.4}", sorted_resid[n / 4]);
        println!("  Median: {:>12.4}", sorted_resid[n / 2]);
        println!("  Q3:     {:>12.4}", sorted_resid[3 * n / 4]);
        println!("  Max:    {:>12.4}", sorted_resid[n - 1]);
    }

    // =========================================================================
    // 9. Comparison: Model with and without Intercept
    // =========================================================================
    println!("\n9. MODEL COMPARISON: With vs Without Intercept");
    println!("-----------------------------------------------");

    let mut model_no_intercept = LinearRegression::without_intercept();
    model_no_intercept.fit(&x, &y)?;

    println!(
        "{:<30} {:>15} {:>15}",
        "Metric", "With Intercept", "Without Intercept"
    );
    println!("{}", "-".repeat(60));

    println!(
        "{:<30} {:>15.6} {:>15.6}",
        "R-squared",
        model.r_squared().unwrap_or(0.0),
        model_no_intercept.r_squared().unwrap_or(0.0)
    );
    println!(
        "{:<30} {:>15.6} {:>15.6}",
        "Adjusted R-squared",
        model.adjusted_r_squared().unwrap_or(0.0),
        model_no_intercept.adjusted_r_squared().unwrap_or(0.0)
    );
    println!(
        "{:<30} {:>15.4} {:>15.4}",
        "AIC",
        model.aic().unwrap_or(0.0),
        model_no_intercept.aic().unwrap_or(0.0)
    );
    println!(
        "{:<30} {:>15.4} {:>15.4}",
        "BIC",
        model.bic().unwrap_or(0.0),
        model_no_intercept.bic().unwrap_or(0.0)
    );

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=============================================================");
    println!("SUMMARY: FerroML Linear Regression Capabilities");
    println!("=============================================================");
    println!(
        "
FerroML provides R-style statistical output including:
  ✓ Coefficient estimates with standard errors, t-stats, and p-values
  ✓ Confidence intervals at configurable levels
  ✓ Model fit statistics: R², adjusted R², F-statistic
  ✓ Information criteria: AIC, BIC, log-likelihood
  ✓ Residual diagnostics: normality (Shapiro-Wilk), autocorrelation (Durbin-Watson)
  ✓ Influential observation detection: Cook's distance, leverage, DFFITS
  ✓ Prediction intervals for new observations
  ✓ Feature importance based on t-statistics

This goes far beyond sklearn's basic fit/predict API, matching or exceeding
statsmodels' OLS output while being written in pure Rust for performance.
"
    );

    Ok(())
}

/// Generate synthetic data for the example
/// True model: y = 5 + 2*x1 + 3*x2 + noise
fn generate_data(n: usize) -> (Array2<f64>, Array1<f64>) {
    // Use a simple LCG for reproducible pseudo-random numbers
    let mut seed: u64 = 42;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f64) / (u32::MAX as f64)
    };

    let mut x_data = Vec::with_capacity(n * 2);
    let mut y_data = Vec::with_capacity(n);

    for _ in 0..n {
        // Generate features with some realistic ranges
        let x1 = 10.0 + 10.0 * rand(); // Education: 10-20 years
        let x2 = 5.0 * rand(); // Experience: 0-5 years

        x_data.push(x1);
        x_data.push(x2);

        // True model with noise
        let noise = (rand() - 0.5) * 4.0; // Noise ~ Uniform(-2, 2)
        let y = 5.0 + 2.0 * x1 + 3.0 * x2 + noise;
        y_data.push(y);
    }

    let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
    let y = Array1::from_vec(y_data);

    (x, y)
}

# Statistical Features Tutorial

FerroML distinguishes itself from other ML libraries through built-in statistical rigor. This tutorial covers model diagnostics, confidence intervals, hypothesis testing, and effect size reporting.

## Learning Objectives

- Run and interpret model diagnostics
- Compute confidence intervals for predictions and parameters
- Perform hypothesis tests for model comparison
- Calculate and interpret effect sizes
- Use bootstrap methods for robust inference

## Model Diagnostics

### Regression Diagnostics

After fitting a `LinearRegression`, FerroML automatically computes comprehensive diagnostics similar to R's `summary(lm(...))`.

```rust
use ferroml_core::models::linear::LinearRegression;
use ferroml_core::models::{Model, StatisticalModel};
use ndarray::{array, Array2, Array1};

let mut model = LinearRegression::new()
    .with_feature_names(vec!["age".into(), "bmi".into(), "bp".into()]);

model.fit(&x_train, &y_train)?;

// Full statistical summary
let summary = model.summary();
println!("{}", summary);
// Output:
//   Linear Regression Results
//   =========================
//   R-squared:          0.4526
//   Adjusted R-squared: 0.4404
//   F-statistic:        37.12 (p < 0.0001)
//   Observations:       353
//
//   Coefficients:
//                 Estimate  Std.Error  t-value  Pr(>|t|)
//   (intercept)   152.133     2.576    59.06    <2e-16  ***
//   age            37.241    11.284     3.30    0.0011  **
//   bmi           787.179    67.451    11.67    <2e-16  ***
//   bp            416.674    69.483     5.99    4.6e-09 ***
```

### Assumption Testing

Every regression model checks its own assumptions:

```rust
let diagnostics = model.diagnostics();

// Check each assumption
println!("Normality OK: {}", diagnostics.normality_ok());
println!("Homoscedasticity OK: {}", diagnostics.homoscedasticity_ok());
println!("No autocorrelation OK: {}", diagnostics.no_autocorrelation_ok());
println!("Multicollinearity OK: {}", diagnostics.multicollinearity_ok(10.0));
println!("All assumptions OK: {}", diagnostics.all_assumptions_ok());
```

FerroML runs these tests automatically:

| Assumption | Test | Interpretation |
|------------|------|----------------|
| Normality of residuals | Shapiro-Wilk | p > 0.05 means residuals appear normal |
| Homoscedasticity | Breusch-Pagan | p > 0.05 means constant variance |
| No autocorrelation | Durbin-Watson | Values near 2.0 mean no autocorrelation |
| No multicollinearity | Condition number | Values < 30 indicate acceptable collinearity |

### Influential Observations

Detect observations that disproportionately affect the model:

```rust
// Cook's distance — observations with d > 4/n are influential
let cooks = model.cooks_distance()?;

// Leverage — high-leverage points are far from the center of the data
let leverage = model.leverage()?;

// Studentized residuals — |r| > 3 suggests an outlier
let stud_resid = model.studentized_residuals()?;

// DFFITS — measures influence on fitted values
let dffits = model.dffits()?;
```

### Variance Inflation Factors (VIF)

Check for multicollinearity among predictors:

```rust
let vif = model.vif(&x_train);
for (i, v) in vif.iter().enumerate() {
    let flag = if *v > 10.0 { " <-- HIGH" } else { "" };
    println!("Feature {}: VIF = {:.2}{}", i, v, flag);
}
// VIF > 10 suggests problematic multicollinearity
```

## Confidence Intervals

### For Model Parameters

```rust
let coef_info = model.coefficients_with_ci(0.95);
for ci in &coef_info {
    let sig = if ci.is_significant(0.05) { "*" } else { "" };
    println!("{}: {:.4} [{:.4}, {:.4}] {}",
        ci.name, ci.estimate, ci.ci_lower, ci.ci_upper, sig);

    // Check if the coefficient's CI excludes zero
    if ci.ci_excludes_zero() {
        println!("  -> Significantly different from zero");
    }
}
```

### For Predictions

```rust
use ferroml_core::models::ProbabilisticModel;

// 95% prediction intervals
let intervals = model.predict_interval(&x_test, 0.95)?;

for i in 0..x_test.nrows() {
    println!("Prediction: {:.2} | 95% PI: [{:.2}, {:.2}]",
        intervals.predictions[i],
        intervals.lower[i],
        intervals.upper[i]);
}
```

### Using the stats Module Directly

For general-purpose confidence intervals on any data:

```rust
use ferroml_core::stats::confidence::{confidence_interval, CIMethod};

let data = array![2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7, 3.0];

// t-distribution CI (best for small samples)
let ci = confidence_interval(&data, 0.95, CIMethod::TDistribution)?;
println!("Mean: {:.3} [{:.3}, {:.3}]", ci.estimate, ci.lower, ci.upper);
println!("Margin of error: {:.3}", ci.margin_of_error());
println!("CI width: {:.3}", ci.width());

// Check if a hypothesized value falls within the CI
println!("Contains 3.0? {}", ci.contains(3.0));
```

Available CI methods:

| Method | Use Case |
|--------|----------|
| `CIMethod::Normal` | Large samples (n > 30) |
| `CIMethod::TDistribution` | Small samples, unknown variance |
| `CIMethod::WilsonScore` | Proportions |
| `CIMethod::ClopperPearson` | Proportions (exact, conservative) |
| `CIMethod::BootstrapPercentile` | Non-parametric, any statistic |
| `CIMethod::BootstrapBCa` | Non-parametric, bias-corrected |
| `CIMethod::BayesianCredible` | Bayesian interpretation |

## Bootstrap Methods

FerroML includes a full bootstrap framework for robust inference when parametric assumptions are uncertain.

```rust
use ferroml_core::stats::bootstrap::Bootstrap;

let data = array![2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7, 3.0, 4.1, 2.5];

// Bootstrap the mean with 10,000 resamples
let bootstrap = Bootstrap::new(10000)
    .with_seed(42)
    .with_confidence(0.95);

let result = bootstrap.mean(&data)?;

println!("Original mean: {:.4}", result.original);
println!("Bootstrap SE:  {:.4}", result.std_error);
println!("Bias:          {:.4}", result.bias);
println!("Percentile CI: ({:.4}, {:.4})", result.ci_percentile.0, result.ci_percentile.1);

// BCa CI (bias-corrected and accelerated) — more accurate for skewed data
if let Some(bca) = result.ci_bca {
    println!("BCa CI:        ({:.4}, {:.4})", bca.0, bca.1);
}
```

### Custom Statistics

Bootstrap any statistic with a closure:

```rust
// Bootstrap the median
let result = bootstrap.median(&data)?;

// Bootstrap the standard deviation
let result = bootstrap.std(&data)?;

// Bootstrap the correlation between two variables
let result = bootstrap.correlation(&x, &y)?;

// Bootstrap any custom statistic
let result = bootstrap.run(&data, |sample| {
    // Compute the trimmed mean (drop top/bottom 10%)
    let mut sorted: Vec<f64> = sample.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let trim = sorted.len() / 10;
    let trimmed = &sorted[trim..sorted.len() - trim];
    trimmed.iter().sum::<f64>() / trimmed.len() as f64
})?;
```

### BCa Confidence Intervals

FerroML's bootstrap automatically computes BCa (bias-corrected and accelerated) confidence intervals when the sample size is 3 or larger. BCa intervals are superior to percentile intervals because they correct for:

- **Bias**: When the bootstrap distribution is systematically shifted
- **Skewness**: When the bootstrap distribution is asymmetric

```rust
let result = bootstrap.mean(&data)?;

// Percentile CI — simple but can be biased
println!("Percentile: ({:.4}, {:.4})", result.ci_percentile.0, result.ci_percentile.1);

// BCa CI — accounts for bias and skewness
if let Some((lower, upper)) = result.ci_bca {
    println!("BCa:        ({:.4}, {:.4})", lower, upper);
}
// For skewed distributions, BCa intervals will be asymmetric,
// which is more accurate than symmetric percentile intervals.
```

## Hypothesis Testing

### Comparing Two Groups

```rust
use ferroml_core::stats::hypothesis::TwoSampleTest;
use ndarray::array;

let control = array![5.2, 4.8, 5.1, 4.9, 5.3, 5.0, 4.7, 5.2];
let treatment = array![5.8, 6.1, 5.9, 6.0, 5.7, 6.2, 5.6, 5.9];

// Two-sample t-test (assumes equal variances)
let result = TwoSampleTest::t_test(&control, &treatment, true).test()?;
println!("t-statistic: {:.4}", result.statistic);
println!("p-value:     {:.4e}", result.p_value);
println!("df:          {:.1}", result.df.unwrap_or(0.0));

// Welch's t-test (does not assume equal variances)
let result = TwoSampleTest::welch(&control, &treatment).test()?;
println!("t-statistic: {:.4}", result.statistic);
println!("p-value:     {:.4e}", result.p_value);

// Mann-Whitney U test (non-parametric alternative)
let result = TwoSampleTest::mann_whitney(&control, &treatment).test()?;
println!("U-statistic: {:.4}", result.statistic);
println!("p-value:     {:.4e}", result.p_value);
```

### Interpreting Results

The `StatisticalResult` struct provides rich output:

```rust
// Effect size is automatically computed
if let Some(effect) = &result.effect_size {
    println!("Effect size: {:.4} ({})", effect.value, effect.interpretation);
}

// Statistical power (when available)
if let Some(power) = result.power {
    println!("Power: {:.4}", power);
}
```

## Effect Sizes

Effect sizes quantify the magnitude of a difference, independent of sample size. FerroML provides several effect size measures.

### Cohen's d

The most common measure for comparing two group means:

```rust
use ferroml_core::stats::effect_size::{CohensD, EffectSize};

let group1 = array![5.2, 4.8, 5.1, 4.9, 5.3, 5.0, 4.7, 5.2];
let group2 = array![5.8, 6.1, 5.9, 6.0, 5.7, 6.2, 5.6, 5.9];

let result = CohensD::new(group1, group2).compute()?;
println!("Cohen's d: {:.4}", result.value);
println!("Interpretation: {}", result.interpretation);
```

| d Value | Interpretation |
|---------|----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| 0.8 - 1.2 | Large |
| > 1.2 | Very large |

### Hedges' g

Bias-corrected version of Cohen's d, preferred for small samples:

```rust
use ferroml_core::stats::effect_size::HedgesG;

let result = HedgesG::new(group1, group2).compute()?;
println!("Hedges' g: {:.4}", result.value);
```

### Glass's Delta

Uses only the control group's standard deviation. Useful when treatment changes variance:

```rust
use ferroml_core::stats::effect_size::GlasssDelta;

let result = GlasssDelta::new(treatment, control).compute()?;
println!("Glass's delta: {:.4}", result.value);
```

## Residual Analysis

Deep-dive into model residuals to validate assumptions:

```rust
use ferroml_core::stats::diagnostics::diagnose_residuals;

let residuals = model.residuals()?;
let diag = diagnose_residuals(&residuals)?;

// Residual distribution summary
println!("Residual stats:");
println!("  Min:      {:.4}", diag.residual_stats.min);
println!("  Q1:       {:.4}", diag.residual_stats.q1);
println!("  Median:   {:.4}", diag.residual_stats.median);
println!("  Q3:       {:.4}", diag.residual_stats.q3);
println!("  Max:      {:.4}", diag.residual_stats.max);
println!("  Skewness: {:.4}", diag.residual_stats.skewness);
println!("  Kurtosis: {:.4}", diag.residual_stats.kurtosis);
```

### Normality Test

```rust
use ferroml_core::stats::diagnostics::ShapiroWilkTest;
use ferroml_core::stats::diagnostics::NormalityTest;

let result = ShapiroWilkTest.test(&residuals)?;
println!("Shapiro-Wilk W: {:.4}", result.statistic);
println!("p-value:        {:.4e}", result.p_value);
if result.p_value > 0.05 {
    println!("Residuals appear normally distributed (fail to reject H0)");
} else {
    println!("Evidence against normality (reject H0 at alpha=0.05)");
}
```

### Autocorrelation

```rust
use ferroml_core::stats::diagnostics::durbin_watson;

let dw = durbin_watson(&residuals);
println!("Durbin-Watson: {:.4}", dw);
// DW near 2: no autocorrelation
// DW near 0: positive autocorrelation
// DW near 4: negative autocorrelation
```

## Putting It All Together

Here's a complete analysis workflow:

```rust
use ferroml_core::models::linear::LinearRegression;
use ferroml_core::models::{Model, StatisticalModel, ProbabilisticModel};
use ferroml_core::stats::bootstrap::Bootstrap;

// 1. Fit the model
let mut model = LinearRegression::new()
    .with_feature_names(vec!["x1".into(), "x2".into(), "x3".into()]);
model.fit(&x_train, &y_train)?;

// 2. Check assumptions
let diagnostics = model.diagnostics();
if !diagnostics.all_assumptions_ok() {
    println!("Warning: some assumptions are violated!");
    println!("  Normality: {}", diagnostics.normality_ok());
    println!("  Homoscedasticity: {}", diagnostics.homoscedasticity_ok());
}

// 3. Examine coefficients with CIs
for ci in model.coefficients_with_ci(0.95) {
    if ci.is_significant(0.05) {
        println!("{}: {:.4} (p={:.4e})", ci.name, ci.estimate, ci.p_value);
    }
}

// 4. Make predictions with intervals
let intervals = model.predict_interval(&x_test, 0.95)?;

// 5. Bootstrap the R-squared for robust CI
let bootstrap = Bootstrap::new(1000).with_seed(42);
// ... custom bootstrap on model performance metrics
```

## Discriminant Analysis: LDA and QDA

FerroML provides both Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA). LDA assumes all classes share the same covariance matrix, producing linear decision boundaries. QDA fits per-class covariance matrices, yielding quadratic boundaries that are more flexible when class distributions differ:

```rust
use ferroml_core::models::QuadraticDiscriminantAnalysis;
use ferroml_core::models::Model;

let mut qda = QuadraticDiscriminantAnalysis::new();
qda.fit(&x_train, &y_train)?;
let predictions = qda.predict(&x_test)?;

// QDA supports regularization to handle near-singular covariances
let mut qda_reg = QuadraticDiscriminantAnalysis::new()
    .with_reg_param(0.1);
```

Use LDA when classes have similar spread; use QDA when they do not.

## Probabilistic Classifiers: Naive Bayes

Naive Bayes classifiers are fast probabilistic models useful as baselines and for high-dimensional data. FerroML provides three variants matching different feature distributions:

- `GaussianNB` -- continuous features (assumes Gaussian per-class distributions)
- `MultinomialNB` -- count or frequency features (e.g., text bag-of-words)
- `BernoulliNB` -- binary or boolean features

All three implement `predict_proba()` for calibrated probability estimates.

## Calibration with Isotonic Regression

`IsotonicRegression` fits a non-decreasing piecewise-linear function, commonly used for probability calibration after a classifier's `predict_proba()` output. It makes no parametric assumptions, unlike Platt scaling (logistic):

```rust
use ferroml_core::models::isotonic::IsotonicRegression;
use ferroml_core::models::Model;

let mut iso = IsotonicRegression::new();
iso.fit(&raw_probas_2d, &y_binary)?;
let calibrated = iso.predict(&new_probas_2d)?;
```

## Next Steps

- [Explainability Tutorial](explainability.md) — TreeSHAP, KernelSHAP, partial dependence
- [Quick Start Tutorial](quickstart.md) — Basic model usage
- [API Reference](https://docs.rs/ferroml-core) — Full API documentation

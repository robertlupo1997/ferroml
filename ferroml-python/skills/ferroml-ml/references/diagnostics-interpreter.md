# Diagnostics Interpreter

How to read and act on every diagnostic FerroML produces.

## OLS Summary Output

When you call `model.summary()` on a `LinearRegression`, `RidgeRegression`, `LogisticRegression`, or similar, you get a full statistical summary. Here is how to read each piece.

### R-squared (R2)

| R2 value | Interpretation | Action |
|----------|---------------|--------|
| > 0.9 | Excellent fit | Check for overfitting or data leakage |
| 0.7 - 0.9 | Good fit | Typical for well-specified models |
| 0.4 - 0.7 | Moderate fit | Consider adding features or non-linear model |
| < 0.4 | Weak fit | Model is missing important predictors |

**Adjusted R2** penalizes for number of features. If adjusted R2 is much lower than R2, you have too many features that add noise.

### F-statistic

Tests whether the model as a whole is significant (at least one predictor matters).

| F-stat p-value | Meaning |
|----------------|---------|
| < 0.001 | Strong evidence the model is useful |
| 0.001 - 0.05 | Model is significant |
| > 0.05 | Model is not statistically significant -- none of the features help |

**If F-stat is significant but no individual coefficients are:** multicollinearity. Features are correlated and canceling each other out. Use Ridge or remove correlated features.

### Coefficients table

Each coefficient row shows:

| Column | What it means | What to look for |
|--------|--------------|-----------------|
| Estimate | Effect size per unit change | Sign and magnitude |
| Std Error | Uncertainty in the estimate | Large = unstable |
| t-statistic | Estimate / Std Error | Larger absolute value = more significant |
| p-value | Probability coefficient is actually zero | < 0.05 typically significant |
| CI lower/upper | 95% confidence interval | If it crosses zero, not significant |

### Coefficient p-values

| p-value | Interpretation | Action |
|---------|---------------|--------|
| < 0.001 | Highly significant | Keep the feature |
| 0.001 - 0.05 | Significant | Keep the feature |
| 0.05 - 0.10 | Marginal | Keep if domain knowledge supports it |
| > 0.10 | Not significant | Consider removing (unless part of an interaction) |

**Warning:** With many features, some will be "significant" by chance. Use `adjust_pvalues()` with Bonferroni or Benjamini-Hochberg correction.

### AIC and BIC

Used to compare models (lower is better). Not meaningful in isolation.

| Scenario | Use |
|----------|-----|
| Comparing models on same data | Lower AIC/BIC = better balance of fit and complexity |
| AIC prefers model A, BIC prefers model B | BIC penalizes complexity more -- pick based on priority |
| Adding a feature increases AIC | The feature hurts more than it helps |

## Durbin-Watson Statistic

Tests for autocorrelation in residuals. Returned by `durbin_watson(residuals)`.

| DW value | Interpretation | Action |
|----------|---------------|--------|
| < 1.0 | Strong positive autocorrelation | Residuals are not independent. Use time-series model or add lagged features. |
| 1.0 - 1.5 | Moderate positive autocorrelation | Investigate. May need autoregressive terms. |
| 1.5 - 2.5 | No significant autocorrelation | Good. Assumption is met. |
| 2.5 - 3.0 | Moderate negative autocorrelation | Less common. Check for over-differencing. |
| > 3.0 | Strong negative autocorrelation | Something is wrong with the model specification. |

**Only matters for:** time-series or sequential data. If your data is cross-sectional (no time ordering), Durbin-Watson is not meaningful.

## Normality Test

`normality_test(residuals)` runs a Shapiro-Wilk test.

| p-value | Interpretation | Action |
|---------|---------------|--------|
| > 0.05 | Residuals are approximately normal | Good. Assumption met. |
| 0.01 - 0.05 | Mild departure from normality | Usually fine with n > 30 (CLT). Check Q-Q plot. |
| < 0.01 | Significant non-normality | Consider PowerTransformer on target, or use robust model. |

**When to ignore:** Normality matters most for small samples (n < 30) and when you need valid p-values/CIs. For prediction-only tasks, it matters less.

**When to worry:** If residuals have heavy tails (outliers) or are strongly skewed, confidence intervals and p-values from `summary()` are unreliable.

## Residual Analysis

Compute residuals as `y_true - y_pred` and look for patterns.

### Residuals vs Fitted values

| Pattern | Problem | Fix |
|---------|---------|-----|
| Random scatter around zero | Good -- assumptions met | None needed |
| Funnel shape (wider on right) | Heteroscedasticity -- variance increases with predicted value | Log-transform target, use RobustRegression, or weighted least squares |
| Curve (U or arch shape) | Non-linearity not captured | Add polynomial features or use non-linear model |
| Clusters | Missing categorical variable | Add the grouping feature |
| A few extreme points | Outliers | Investigate. Use RobustRegression or remove if errors. |

### Residuals vs Individual features

If residuals show a pattern against a specific feature, that feature has a non-linear relationship. Add `PolynomialFeatures` for that feature or use a tree-based model.

## Classification Diagnostics

### Confusion Matrix

From `confusion_matrix(y_true, y_pred)`:

```
              Predicted 0    Predicted 1
Actual 0      TN             FP
Actual 1      FN             TP
```

| Cell | Name | Meaning |
|------|------|---------|
| TN | True Negative | Correctly predicted negative |
| FP | False Positive | Type I error (false alarm) |
| FN | False Negative | Type II error (missed detection) |
| TP | True Positive | Correctly predicted positive |

**Reading it:**
- Diagonal = correct predictions. Off-diagonal = errors.
- FP-heavy: model is too aggressive (lower threshold).
- FN-heavy: model is too conservative (raise threshold or use recall-focused metric).

### ROC-AUC

From `roc_auc_score(y_true, y_score)`:

| AUC value | Interpretation |
|-----------|---------------|
| 0.9 - 1.0 | Excellent discrimination |
| 0.8 - 0.9 | Good discrimination |
| 0.7 - 0.8 | Fair discrimination |
| 0.6 - 0.7 | Poor discrimination |
| 0.5 | Random guessing (model is useless) |
| < 0.5 | Worse than random (labels may be flipped) |

**When to use:** Balanced classes, or when you care about ranking quality across all thresholds.

**When NOT to use:** Highly imbalanced classes. Use PR-AUC instead.

### PR-AUC and Average Precision

From `pr_auc_score(y_true, y_score)` or `average_precision_score(y_true, y_score)`:

Better than ROC-AUC when classes are imbalanced. Baseline is the proportion of positives (not 0.5).

| Scenario | Baseline | Good PR-AUC |
|----------|----------|-------------|
| 50/50 classes | 0.5 | > 0.8 |
| 10% positive | 0.1 | > 0.5 |
| 1% positive | 0.01 | > 0.2 |

### Brier Score

From `brier_score(y_true, y_pred_proba)`:

| Value | Interpretation |
|-------|---------------|
| 0.0 | Perfect calibration |
| < 0.1 | Well calibrated |
| 0.1 - 0.25 | Moderate calibration |
| > 0.25 | Poor calibration |

`brier_skill_score` compares to a naive model. Positive = better than always predicting the base rate.

### Log Loss

From `log_loss(y_true, y_pred_proba)`:

Lower is better. Heavily penalizes confident wrong predictions. A model that predicts 0.99 for a negative sample gets punished much more than one that predicts 0.6.

## Clustering Diagnostics

### Silhouette Score

From `silhouette_score(X, labels)`:

| Value | Interpretation |
|-------|---------------|
| 0.7 - 1.0 | Strong cluster structure |
| 0.5 - 0.7 | Reasonable structure |
| 0.25 - 0.5 | Weak structure, clusters may overlap |
| < 0.25 | No meaningful structure |
| Negative | Samples are in the wrong cluster |

**Use it for:** Comparing different k values in KMeans. Pick k with highest silhouette.

### Calinski-Harabasz Score

From `calinski_harabasz_score(X, labels)`:

Higher is better. No absolute threshold -- use it to compare k values. Measures ratio of between-cluster dispersion to within-cluster dispersion.

### Davies-Bouldin Score

From `davies_bouldin_score(X, labels)`:

Lower is better. Zero is perfect. Measures average similarity between each cluster and its most similar cluster.

| Value | Interpretation |
|-------|---------------|
| < 0.5 | Well-separated clusters |
| 0.5 - 1.0 | Moderate separation |
| > 1.0 | Clusters overlap significantly |

### Hopkins Statistic

From `hopkins_statistic(X)`:

Tests whether data has cluster tendency at all. Run BEFORE clustering.

| Value | Interpretation |
|-------|---------------|
| > 0.7 | Data likely has clusters -- proceed with clustering |
| 0.5 - 0.7 | Unclear -- clustering may not be meaningful |
| < 0.5 | Data is uniformly distributed -- clustering will find noise |

## Effect Sizes

### Cohen's d

From `cohens_d(group_a, group_b)`:

| |d| value | Interpretation |
|-----------|---------------|
| < 0.2 | Negligible effect |
| 0.2 - 0.5 | Small effect |
| 0.5 - 0.8 | Medium effect |
| > 0.8 | Large effect |

### Hedges' g

From `hedges_g(group_a, group_b)`:

Same interpretation as Cohen's d, but corrected for small sample sizes. **Use Hedges' g when either group has n < 20.**

## Confidence Intervals

From `confidence_interval(data, confidence=0.95)` or `bootstrap_ci(data, n_bootstrap=1000, confidence=0.95)`:

### Parametric CI (`confidence_interval`)
- Assumes normality. Fast.
- Use when: data is roughly normal, n > 30.

### Bootstrap CI (`bootstrap_ci`)
- No distribution assumptions. Resampling-based.
- Use when: small samples, skewed data, non-normal distributions.
- Increase `n_bootstrap` for more precision (1000 is usually enough).

### Reading CIs

- If a CI for a difference **does not include zero**, the difference is significant.
- If two CIs **do not overlap**, the groups are significantly different.
- If two CIs **overlap slightly**, they might still be significantly different -- use a formal test.
- Wider CI = more uncertainty = need more data.

## When to Worry vs When to Ignore

| Diagnostic | Worry if | Ignore if |
|-----------|----------|-----------|
| Low R2 | You need accurate predictions | You only need feature directions |
| Non-normal residuals | n < 30 or you need valid CIs | n > 100 and prediction-only |
| High p-value on coefficient | Feature was expected to matter | You are using regularized model (Ridge/Lasso) |
| Durbin-Watson far from 2 | Data has time ordering | Cross-sectional data |
| Heteroscedasticity | You need valid standard errors | You only care about point predictions |
| Low silhouette score | You are reporting cluster quality | You are using clusters as features |
| Large effect size | Sample size is small | Sample size is very large (everything is "significant") |

## Quick diagnostic workflow

1. **Fit model:** `model.fit(X_train, y_train)`
2. **Check overall fit:** `model.summary()` -- look at R2, F-stat
3. **Check coefficients:** look for unexpected signs, high p-values
4. **Check residuals:** `residuals = y_train - model.predict(X_train)`
5. **Test normality:** `normality_test(residuals)` -- p > 0.05 is fine
6. **Test autocorrelation:** `durbin_watson(residuals)` -- 1.5-2.5 is fine
7. **If problems found:** address them (transform target, add features, switch model)
8. **Evaluate on test set:** `r2_score(y_test, model.predict(X_test))`

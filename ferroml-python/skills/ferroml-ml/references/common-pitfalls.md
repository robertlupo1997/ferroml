# Common ML Pitfalls and How to Fix Them

## 1. Data Leakage

**Symptom:** Suspiciously high accuracy (95%+), performance drops in production.

**Causes:**
- Feature contains future information (e.g., "total_revenue" when predicting "will_churn")
- Target encoded into a feature (e.g., "outcome_code" maps 1:1 to target)
- Test data was used during preprocessing (fit scaler on full data, not just train)

**Detection:**
- Run `detect_leakage.py` — checks for high mutual info between features and target
- Suspiciously perfect single-feature models
- Train and test performance are nearly identical

**Fix:**
- Remove leaky features
- Always fit preprocessors on training data only
- Use `Pipeline` to ensure proper fit/transform separation

## 2. Class Imbalance

**Symptom:** High accuracy but terrible recall on minority class. Model predicts majority class for everything.

**Detection:**
- Check `value_counts()` on target — ratio > 10:1 is significant
- Confusion matrix shows near-zero predictions for minority class
- High accuracy but low F1/recall

**Fix (in order of preference):**
1. Use appropriate metric: F1, balanced accuracy, ROC-AUC — NOT accuracy
2. Class weights: most FerroML classifiers accept class_weight parameter
3. Threshold optimization: `optimize_threshold.py` finds the cutoff that maximizes your business metric
4. Stratified sampling: ensure train/test splits preserve class ratio
5. Oversampling minority class (duplicate rows with slight noise)

## 3. Overfitting

**Symptom:** Great training score, poor test score. Gap > 5% accuracy or > 0.1 R2.

**Detection:**
- Compare train vs test metrics
- Learning curves show diverging train/test lines
- Model complexity too high for data size

**Fix:**
1. More data (if possible)
2. Regularization: increase alpha (Ridge/Lasso), decrease max_depth (trees)
3. Feature selection: remove noisy features with `feature_select.py`
4. Cross-validation instead of single train/test split
5. Simpler model (Linear instead of GBT if data is small)
6. Early stopping (GBT, MLP)

## 4. Underfitting

**Symptom:** Both training and test scores are poor.

**Detection:**
- Low R2 or accuracy on BOTH train and test
- Learning curves show flat, low performance
- Residuals show systematic patterns (not random)

**Fix:**
1. More features / feature engineering
2. More complex model (GBT instead of Linear)
3. Reduce regularization (lower alpha)
4. Check if target is actually predictable from features

## 5. Multicollinearity

**Symptom:** Unstable coefficients in linear models, large standard errors, non-significant p-values on features you know matter.

**Detection:**
- VIF > 10 for any feature (run `feature_select.py`)
- Correlation matrix shows |r| > 0.8 between feature pairs
- Coefficients flip sign when a feature is added/removed

**Fix:**
1. Remove one of each correlated pair
2. Use RidgeRegression (handles multicollinearity naturally)
3. PCA to create uncorrelated components
4. VIF-based iterative removal (drop highest VIF, recompute, repeat)

## 6. Feature Scaling Issues

**Symptom:** KNN, SVM, or neural network performs terribly. One feature dominates distance calculations.

**Models that NEED scaling:** KNN, SVC/SVR, MLP, LogisticRegression (with regularization), PCA

**Models that DON'T need scaling:** Decision trees, Random Forest, GBT, Naive Bayes

**Fix:** Always use `StandardScaler` or `MinMaxScaler` before distance/gradient-based models. Use `Pipeline` to ensure consistent scaling.

## 7. Missing Data Mistakes

**Bad:** Impute with mean/median BEFORE train/test split (leaks test statistics into train).
**Bad:** Drop all rows with any missing value (massive data loss).
**Bad:** Ignore missing values (most models crash on NaN).

**Good:**
1. Split data first, then impute on train, transform test with train's statistics
2. Use `Pipeline` for automatic handling
3. For >50% missing: drop the column
4. For <5% missing: median imputation is usually fine
5. For 5-50% missing: consider KNN imputation or indicator variables

## 8. Wrong Evaluation Metric

| Situation | Wrong metric | Right metric |
|-----------|-------------|-------------|
| Imbalanced classes | Accuracy | F1, balanced accuracy, ROC-AUC |
| Cost-asymmetric (FN costs more) | Accuracy | Recall, cost-weighted score |
| Regression with outliers | MSE/RMSE | MAE, median absolute error |
| Probabilistic predictions | Accuracy | Log loss, Brier score |
| Ranking problems | Accuracy | NDCG, MAP |

## 9. Not Checking Statistical Assumptions

**Applies to:** Linear models (OLS, Ridge, Logistic)

**Assumptions to check:**
- Linearity (residual vs fitted plot)
- Normality of residuals (Shapiro-Wilk, Q-Q plot)
- Homoscedasticity (constant variance — Breusch-Pagan test)
- Independence (Durbin-Watson statistic)
- No multicollinearity (VIF)

**Use:** `validate_assumptions.py` or `ferroml diagnose` for automated checking.

## 10. Ignoring Temporal Ordering

**Symptom:** Model performs well in CV but fails in production on new data.

**Cause:** Standard k-fold CV randomly mixes past and future data. Model "sees the future" during training.

**Detection:** Data has a timestamp/date column. Target relates to future events.

**Fix:** Use time-series split CV (train on past, test on future). Never shuffle temporal data randomly.

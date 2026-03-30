# Troubleshooting Guide

Common errors, performance issues, and their fixes in FerroML.

## FerroError Reference

Every FerroML error includes a `.hint()` with remediation guidance. Here is the full catalog.

### InvalidInput

**Cause:** Data contains NaN, infinity, wrong dtype, empty arrays, or out-of-range values.

| Symptom | Fix |
|---------|-----|
| `"Input contains NaN"` | `np.isnan(X).any()` -- find and impute or drop NaN rows |
| `"Input contains infinity"` | `np.isinf(X).any()` -- clip or replace inf values |
| `"Expected 2D array"` | Reshape: `X.reshape(-1, 1)` for single feature |
| `"Empty input array"` | Check data loading -- array has 0 rows or 0 columns |
| `"n_clusters must be > 0"` | Check hyperparameter values are positive |
| `"learning_rate must be > 0"` | Ensure learning_rate is a positive float |
| `"n_estimators must be >= 1"` | Ensure n_estimators is at least 1 |

**Debug pattern:**
```python
import numpy as np

# Check for problems before fitting
print(f"Shape: {X.shape}")
print(f"NaN count: {np.isnan(X).sum()}")
print(f"Inf count: {np.isinf(X).sum()}")
print(f"Dtype: {X.dtype}")
print(f"Min: {X.min()}, Max: {X.max()}")
```

### ShapeMismatch

**Cause:** Feature dimensions don't match between train and predict, or X and y have different sample counts.

| Symptom | Fix |
|---------|-----|
| `"X has N features, expected M"` | Ensure test data has same columns as train data |
| `"X and y have different lengths"` | `X.shape[0]` must equal `y.shape[0]` |
| `"Expected 1D target"` | `y.ravel()` or `y.flatten()` |

**Common causes:**
- Forgot to apply the same preprocessing to test data (use Pipeline)
- OneHotEncoder created different columns for train vs test
- Accidentally included target column in features

**Fix pattern:**
```python
# WRONG: separate preprocessing
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
# ... later, different scaler or forgot to transform
model.predict(X_test)  # X_test not scaled!

# RIGHT: use Pipeline
from ferroml.pipeline import Pipeline
pipe = Pipeline(steps=[("scaler", StandardScaler()), ("model", model)])
pipe.fit(X_train, y_train)
pipe.predict(X_test)  # scaler applied automatically
```

### ConvergenceFailure

**Cause:** Optimization did not converge within max_iter iterations.

| Symptom | Fix |
|---------|-----|
| `"Failed to converge in N iterations"` | Try fixes below in order |

**Fixes (try in order):**

1. **Scale features first** -- most common cause
   ```python
   from ferroml.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Increase max_iter**
   ```python
   model = LogisticRegression(max_iter=1000)  # default is usually 100
   ```

3. **Reduce learning_rate** (for SGD-based models)
   ```python
   model = SGDClassifier(learning_rate=0.001)  # smaller = more stable
   ```

4. **Add regularization**
   ```python
   model = LogisticRegression(C=0.1)  # stronger regularization
   ```

5. **Try a different model** -- some data doesn't suit certain optimizers
   ```python
   # Switch from LogisticRegression to SGDClassifier or vice versa
   ```

### NotFitted

**Cause:** Called `predict()`, `transform()`, or `score()` before `fit()`.

```python
# WRONG
model = RandomForestClassifier()
model.predict(X_test)  # NotFitted!

# RIGHT
model = RandomForestClassifier()
model.fit(X_train, y_train)
model.predict(X_test)
```

**Common in pipelines:** Ensure every step is fitted. Using `Pipeline.fit()` handles this.

### AssumptionViolation

**Cause:** Statistical assumptions of the model are violated.

| Symptom | Fix |
|---------|-----|
| Non-normal residuals | Try `PowerTransformer` on target, or use robust model |
| Heteroscedasticity | Use `RobustScaler`, or switch to `QuantileRegression` |
| Multicollinearity | Remove correlated features, use Ridge/Lasso instead of OLS |
| Non-linear relationship | Add `PolynomialFeatures`, or use tree-based model |

**Diagnostic tools:**
```python
from ferroml.stats import durbin_watson, normality_test

# Check residual autocorrelation
residuals = y_test - model.predict(X_test)
dw = durbin_watson(residuals)
# DW ~ 2 is good, <1.5 or >2.5 suggests autocorrelation

# Check normality of residuals
result = normality_test(residuals)
# p > 0.05 means residuals are approximately normal
```

### NumericalError

**Cause:** Floating-point issues -- singular matrices, overflow, division by zero.

| Symptom | Fix |
|---------|-----|
| `"Singular matrix"` | Features are linearly dependent. Remove duplicates or use Ridge |
| `"Numerical overflow"` | Scale features to reasonable range, use `StandardScaler` |
| `"Division by zero"` | Constant feature detected. Use `VarianceThreshold` to remove |

**Fix pattern:**
```python
from ferroml.preprocessing import VarianceThreshold

# Remove constant/near-constant features
vt = VarianceThreshold(threshold=0.01)
X_filtered = vt.fit_transform(X)
```

## Memory Issues

| Problem | Solution |
|---------|----------|
| OOM on large dataset | Subsample: `X_sample = X[np.random.choice(len(X), 10000)]` |
| KMeans OOM | Use `MiniBatchKMeans` instead |
| PCA OOM | Use `IncrementalPCA` with batch processing |
| Too many trees | Reduce `n_estimators` (100 is often enough) |
| OneHotEncoder explosion | Use `TargetEncoder` for high-cardinality features |
| Text vectorizer OOM | Set `max_features=5000` on CountVectorizer/TfidfVectorizer |

## Slow Training

| Problem | Solution |
|---------|----------|
| GBT too slow | Use `HistGradientBoostingClassifier` (histogram-based, much faster) |
| Too many features | Feature selection: `SelectKBest`, `VarianceThreshold` |
| Large dataset + linear model | Use `SGDClassifier`/`SGDRegressor` (online learning) |
| KNN slow at predict time | Reduce `n_neighbors`, subsample training data |
| Cross-validation slow | Reduce `cv_folds` from 10 to 5, or use holdout |
| AutoML timeout | Increase `time_budget_seconds`, or reduce search space |

**Model speed reference:**

| Speed Tier | Models | Typical Predict Time |
|------------|--------|---------------------|
| Fast (<1ms) | LinearRegression, LogisticRegression, NaiveBayes, DecisionTree | Microseconds |
| Medium (1-10ms) | RandomForest, GBT, HistGBT, Ridge, Lasso | Low milliseconds |
| Slow (>10ms) | KNN, SVC, GaussianProcess | Depends on data size |

## Poor Model Performance

### Underfitting (train score is low)

| Fix | How |
|-----|-----|
| More features | `PolynomialFeatures`, domain-specific engineering |
| Less regularization | Increase `C` (LogReg/SVM), decrease `alpha` (Ridge/Lasso) |
| More complex model | Tree ensemble instead of linear, deeper trees |
| More iterations | Increase `n_estimators` for ensembles |

### Overfitting (train >> test score)

| Fix | How |
|-----|-----|
| More regularization | Decrease `C`, increase `alpha`, decrease `max_depth` |
| Less complex model | Fewer features, shallower trees, fewer estimators |
| More data | Collect more samples, or use `SMOTE`/`ADASYN` |
| Cross-validation | Use CV instead of single train/test split |
| Feature selection | `SelectKBest`, `RecursiveFeatureElimination` |

### Imbalanced Classes

```python
from ferroml.preprocessing import SMOTE, ADASYN

# Oversample minority class
smote = SMOTE(seed=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Or use ADASYN (adaptive)
adasyn = ADASYN(seed=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
```

**Also consider:**
- Use F1/AUC instead of accuracy as metric
- Class weights: some models accept `class_weight` parameters

## Python-Specific Issues

### Import Errors

```
ModuleNotFoundError: No module named 'ferroml'
```
**Fix:** Build the bindings first:
```bash
source .venv/bin/activate
maturin develop --release -m ferroml-python/Cargo.toml
```

### NumPy Dtype Issues

```
TypeError: expected numpy array of dtype float64
```
**Fix:** FerroML requires float64 arrays:
```python
X = X.astype(np.float64)
y = y.astype(np.float64)
```

### Serialization Errors

```
OSError: Serialization not supported for this model
```
**Current limitations:**
- Pipeline, VotingClassifier, StackingClassifier, and BaggingClassifier serialization is not yet implemented
- Individual models can be serialized with `save_json()`, `save_msgpack()`, or `save_bincode()`

```python
from ferroml.serialization import save_json, load_json

save_json(model, "model.json")
loaded = load_json("model.json")
```

## Error Hints

Every FerroML error has a `.hint()` method with actionable guidance:

```python
try:
    model.fit(X, y)
except Exception as e:
    # The hint is automatically appended to the error message
    print(str(e))  # Includes both error and hint
```

## Quick Diagnostic Checklist

When something goes wrong, check in this order:

1. **Data quality**: `np.isnan(X).sum()`, `np.isinf(X).sum()`, `X.shape`
2. **Data types**: `X.dtype` should be `float64`
3. **Scale**: `X.min()`, `X.max()` -- wildly different scales need StandardScaler
4. **Target**: `np.unique(y)` -- correct number of classes? Continuous for regression?
5. **Train/test alignment**: Same number of features? Same preprocessing?
6. **Model fitted**: Did you call `.fit()` before `.predict()`?
7. **Hyperparameters**: Reasonable values? Use `model.search_space()` for valid ranges

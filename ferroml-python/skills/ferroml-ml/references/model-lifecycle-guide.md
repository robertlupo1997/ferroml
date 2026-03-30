# Model Lifecycle Guide

Managing ML models from first experiment through retirement.

## Lifecycle Phases Overview

```
[Experiment] --> [Validation] --> [Deployment] --> [Monitoring] --> [Retraining] --> [Retirement]
     |               |                |                |                |                |
  Explore &       Prove it         Ship it         Watch it        Update it       Sunset it
  iterate         works
```

## Phase 1: Experiment

**Goal:** Find a model that solves the problem.

### Steps

1. **Understand the data**
   ```python
   import numpy as np
   print(f"Shape: {X.shape}")
   print(f"NaN: {np.isnan(X).sum()}")
   print(f"Target distribution: {np.unique(y, return_counts=True)}")
   ```

2. **Get recommendations**
   ```python
   from ferroml import recommend
   recs = recommend(X, y, task="classification")
   for r in recs:
       print(f"{r.model_name}: {r.reasoning}")
       print(f"  Suggested params: {r.params}")
   ```

3. **Train candidates**
   ```python
   from ferroml.linear import LogisticRegression
   from ferroml.trees import RandomForestClassifier, HistGradientBoostingClassifier

   candidates = [
       ("LogReg", LogisticRegression()),
       ("RF", RandomForestClassifier(n_estimators=100, seed=42)),
       ("HistGBT", HistGradientBoostingClassifier(n_estimators=200, seed=42)),
   ]

   for name, model in candidates:
       model.fit(X_train, y_train)
       y_pred = model.predict(X_test)
       print(f"{name}: accuracy={accuracy_score(y_test, y_pred):.4f}")
   ```

4. **Use AutoML for broader search**
   ```python
   from ferroml.automl import AutoML, AutoMLConfig

   config = AutoMLConfig(
       task="classification",
       metric="f1",
       time_budget_seconds=300,
       cv_folds=5,
       seed=42,
   )
   automl = AutoML(config)
   automl.fit(X_train, y_train)
   ```

### Experiment Phase Checklist
- [ ] Defined success metric (accuracy, F1, RMSE, etc.)
- [ ] Established baseline (majority class, mean prediction, simple model)
- [ ] Tested 3+ model types
- [ ] Logged all experiments (see experiment-tracking-guide)
- [ ] Selected top 1-2 candidates for validation

## Phase 2: Validation

**Goal:** Prove the model works on unseen data and is statistically better than alternatives.

### Hold-Out Test

```python
from ferroml.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

### Cross-Validation

More reliable than a single train/test split.

```python
from ferroml.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="f1")
print(f"F1: {scores.mean():.4f} +/- {scores.std():.4f}")
```

### Statistical Significance

```python
from ferroml.metrics import paired_ttest, corrected_resampled_ttest, mcnemar_test

# Are the two best models significantly different?
result = paired_ttest(scores_model_a, scores_model_b)
print(f"p-value: {result.p_value:.4f}")

# McNemar's test on predictions (no CV needed)
result = mcnemar_test(y_test, pred_a, pred_b)
print(f"p-value: {result.p_value:.4f}")
```

### Validation Phase Checklist

- [ ] Hold-out test performance meets threshold
- [ ] Cross-validation shows consistent performance (low std)
- [ ] Statistically significant improvement over baseline (p < 0.05)
- [ ] No signs of overfitting (train score close to test score)
- [ ] Performance stable across data subgroups (fairness)
- [ ] Model diagnostics checked (for statistical models: `summary()`)

## Phase 3: Deployment

**Goal:** Make the model available for production predictions.

### Serialize the Model

```python
from ferroml.serialization import save_msgpack

# Save model
save_msgpack(model, "models/churn_v1.0.0.msgpack")

# Save full pipeline (preferred)
save_msgpack(pipeline, "models/churn_pipeline_v1.0.0.msgpack")
```

### Input Validation at Serving Time

```python
import numpy as np

def validate_input(X):
    """Validate input before prediction."""
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be numpy array")
    if X.dtype != np.float64:
        X = X.astype(np.float64)
    if np.isnan(X).any():
        raise ValueError("Input contains NaN values")
    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(f"Expected {EXPECTED_FEATURES} features, got {X.shape[1]}")
    return X
```

### Deployment Metadata

Record alongside the model file:

```json
{
    "model_name": "churn_classifier",
    "version": "1.0.0",
    "model_type": "HistGradientBoostingClassifier",
    "trained_date": "2026-03-29",
    "training_data": "customers_2026Q1",
    "data_hash": "sha256:a3f2...",
    "n_features": 25,
    "feature_names": ["age", "tenure", "monthly_charges", "..."],
    "metrics": {"f1": 0.89, "auc": 0.94},
    "approved_by": "data-science-lead",
    "serialization_format": "msgpack"
}
```

### Deployment Checklist

- [ ] Model serialized and tested (load + predict on sample)
- [ ] Input validation implemented
- [ ] Deployment metadata recorded
- [ ] Rollback plan defined (previous model version available)
- [ ] Monitoring alerts configured
- [ ] Load tested for expected throughput

## Phase 4: Monitoring

**Goal:** Detect when the model degrades.

### What to Monitor

| Signal | How to Detect | Threshold |
|--------|--------------|-----------|
| **Prediction distribution shift** | Compare prediction histogram to training-time baseline | KL divergence > 0.1 |
| **Input feature drift** | Compare feature means/stds to training data | >2 std deviations from training mean |
| **Accuracy degradation** | Compare against labeled samples (when available) | >5% drop from validation score |
| **Latency increase** | Track p50/p95/p99 predict time | >2x baseline |
| **Error rate** | Count InvalidInput / ShapeMismatch errors | >1% of requests |
| **Volume anomaly** | Track prediction request count | >3x or <0.3x normal |

### Simple Drift Detection

```python
import numpy as np

def detect_drift(X_new, X_train_stats):
    """Check if new data has drifted from training distribution."""
    alerts = []
    for col in range(X_new.shape[1]):
        col_mean = X_new[:, col].mean()
        col_std = X_new[:, col].std()
        train_mean = X_train_stats["means"][col]
        train_std = X_train_stats["stds"][col]

        # Feature mean shifted by >2 training std devs
        if abs(col_mean - train_mean) > 2 * train_std:
            alerts.append(f"Feature {col}: mean shifted ({train_mean:.2f} -> {col_mean:.2f})")

        # Feature variance changed dramatically
        if col_std > 3 * train_std or col_std < train_std / 3:
            alerts.append(f"Feature {col}: variance changed ({train_std:.2f} -> {col_std:.2f})")

    return alerts
```

### Monitoring Checklist

- [ ] Prediction distribution baseline recorded
- [ ] Feature distribution baseline recorded
- [ ] Alerting configured for drift/degradation
- [ ] Labeled sample collection process in place
- [ ] Dashboard showing key metrics over time

## Phase 5: Retraining

**Goal:** Update the model when it degrades or new data is available.

### Retraining Triggers

| Trigger | Action |
|---------|--------|
| Drift detected | Retrain on recent data |
| Performance drop >5% | Retrain, possibly redesign |
| New features available | Retrain with new features |
| Scheduled (quarterly) | Retrain on latest data |
| Regulatory requirement | Retrain with updated rules |

### Retrain vs Rebuild Decision

```
Performance drop detected:
  |
  +--> Drop < 10%? --> Retrain with fresh data (same model type + params)
  |
  +--> Drop 10-25%? --> Retrain + hyperparameter tune
  |
  +--> Drop > 25%? --> Rebuild from scratch (Phase 1)
  |
  +--> Data fundamentally changed? --> Rebuild from scratch
```

### Retraining Process

1. Pull latest data
2. Apply same preprocessing pipeline
3. Retrain model with same hyperparameters
4. Validate: compare to current production model
5. If significantly better (statistical test): deploy
6. If not better: investigate root cause

```python
from ferroml.metrics import paired_ttest

# Compare new model to current production model
new_scores = cross_val_score(new_model, X_new, y_new, cv=5, scoring="f1")
old_scores = cross_val_score(old_model, X_new, y_new, cv=5, scoring="f1")

result = paired_ttest(new_scores, old_scores)
if result.p_value < 0.05 and new_scores.mean() > old_scores.mean():
    print("New model is significantly better -- deploy")
else:
    print("No significant improvement -- keep current model")
```

## Phase 6: Retirement

**Goal:** Gracefully remove a model from production.

### Retirement Triggers

- Replacement model deployed and validated
- Business process no longer needs the prediction
- Regulatory change invalidates the model
- Data source discontinued

### Retirement Process

1. **Announce deprecation** -- notify consumers with timeline
2. **Dual-run period** -- new model serves, old model logged for comparison
3. **Remove old model** -- stop serving, archive model artifact
4. **Retain records** -- keep metadata, metrics, and approval records for audit

### Retirement Checklist

- [ ] Replacement model validated and deployed
- [ ] All consumers migrated to new model/endpoint
- [ ] Old model artifact archived (not deleted)
- [ ] Documentation updated
- [ ] Audit trail complete

## Model Versioning

Use semantic versioning for models:

| Version | Meaning | Example |
|---------|---------|---------|
| Major (X.0.0) | New model architecture or features | Linear -> GBT |
| Minor (1.X.0) | Retrained on new data, same architecture | Monthly retrain |
| Patch (1.0.X) | Bug fix, no retraining | Fixed preprocessing bug |

**Naming convention:** `{model_name}_v{major}.{minor}.{patch}.{format}`

```
models/
  churn_v1.0.0.msgpack     # Initial GBT model
  churn_v1.1.0.msgpack     # Retrained on Q2 data
  churn_v1.1.1.msgpack     # Fixed feature scaling bug
  churn_v2.0.0.msgpack     # Switched to HistGBT + new features
```

## Model Governance Checklist

For regulated or high-stakes models, record:

| Item | Details |
|------|---------|
| **Model owner** | Who is responsible for this model? |
| **Approval** | Who approved deployment? Date? |
| **Training data** | What data? Date range? Any exclusions? |
| **Features used** | Complete list with descriptions |
| **Metrics** | Performance on test set + cross-validation |
| **Fairness** | Performance across protected groups |
| **Limitations** | Known failure modes, out-of-scope inputs |
| **Retraining schedule** | How often? Triggered by what? |
| **Incident process** | What happens if model makes a bad prediction? |

Use `model_card()` to get structured metadata:

```python
card = model.model_card()
print(f"Algorithm: {card.algorithm}")
print(f"Task: {card.task_type}")
print(f"Strengths: {card.strengths}")
print(f"Limitations: {card.limitations}")
```

## Complete Lifecycle Timeline Example

```
Week 1:  [Experiment] Explore data, train 5 candidates, log results
Week 2:  [Experiment] Feature engineering, hyperparameter tuning
Week 3:  [Validation] Cross-validation, statistical tests, fairness check
Week 4:  [Deployment] Serialize, create endpoint, load test, go live
Week 5+: [Monitoring] Track drift, collect labels, watch metrics
Month 3: [Retraining] First scheduled retrain, compare to v1
Month 6: [Retraining] Performance drop detected, retrain + tune
Year 1:  [Retirement] Business pivot, model no longer needed, archive
```

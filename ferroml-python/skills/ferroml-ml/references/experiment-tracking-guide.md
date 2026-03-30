# Experiment Tracking Guide

How to systematically track, compare, and reproduce ML experiments with FerroML.

## What to Record Per Experiment

Every experiment run should capture:

| Field | Example | Why |
|-------|---------|-----|
| `experiment_id` | `"exp_2026-03-29_001"` | Unique identifier |
| `timestamp` | `"2026-03-29T14:30:00Z"` | When it ran |
| `data_hash` | `"sha256:a3f2..."` | Detect data changes |
| `data_shape` | `[10000, 25]` | Quick sanity check |
| `model_type` | `"RandomForestClassifier"` | Which algorithm |
| `model_params` | `{"n_estimators": 100, ...}` | Full hyperparameters |
| `preprocessing` | `["StandardScaler", "SelectKBest(k=10)"]` | Feature pipeline |
| `cv_folds` | `5` | Cross-validation config |
| `seed` | `42` | Reproducibility |
| `metrics` | `{"accuracy": 0.92, "f1": 0.89}` | Results |
| `train_time_seconds` | `12.3` | Performance budget |
| `notes` | `"Added interaction features"` | Human context |

## Experiment Log Template

```python
import json, hashlib, time
from datetime import datetime

def data_hash(X):
    """Hash feature matrix for change detection."""
    return hashlib.sha256(X.tobytes()).hexdigest()[:16]

def log_experiment(path, entry):
    """Append experiment to JSON log."""
    try:
        with open(path, "r") as f:
            log = json.load(f)
    except FileNotFoundError:
        log = []
    log.append(entry)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)

# Usage pattern
entry = {
    "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "timestamp": datetime.now().isoformat(),
    "data_hash": data_hash(X_train),
    "data_shape": list(X_train.shape),
    "model_type": "GradientBoostingClassifier",
    "model_params": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
    "preprocessing": ["StandardScaler", "SelectKBest(k=15)"],
    "cv_folds": 5,
    "seed": 42,
    "metrics": {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    },
    "train_time_seconds": round(elapsed, 2),
    "notes": "Baseline with top-15 features",
}
log_experiment("experiment_log.json", entry)
```

## Comparing Experiments

### Metrics Table

```python
import json

with open("experiment_log.json") as f:
    experiments = json.load(f)

# Print comparison table
print(f"{'ID':<25} {'Model':<30} {'Accuracy':>8} {'F1':>8} {'AUC':>8} {'Time':>8}")
print("-" * 95)
for exp in experiments:
    m = exp["metrics"]
    print(f"{exp['experiment_id']:<25} {exp['model_type']:<30} "
          f"{m.get('accuracy', 0):>8.4f} {m.get('f1', 0):>8.4f} "
          f"{m.get('roc_auc', 0):>8.4f} {exp.get('train_time_seconds', 0):>7.1f}s")
```

### Statistical Significance Testing

Never trust raw metric differences. Use statistical tests to confirm.

```python
from ferroml.metrics import paired_ttest, corrected_resampled_ttest

# Compare two models across CV folds
# scores_a and scores_b are arrays of per-fold metrics
result = paired_ttest(scores_a, scores_b)
print(f"t={result.statistic:.3f}, p={result.p_value:.4f}")

# For repeated k-fold CV (corrects for non-independence)
result = corrected_resampled_ttest(
    scores_a, scores_b,
    n_train=800, n_test=200  # sizes per fold
)
print(f"t={result.statistic:.3f}, p={result.p_value:.4f}")
```

**Additional comparison tests:**

| Test | When to Use | FerroML Function |
|------|-------------|------------------|
| Paired t-test | Comparing two models on same CV folds | `paired_ttest(a, b)` |
| Corrected resampled t-test | Repeated CV (accounts for overlap) | `corrected_resampled_ttest(a, b, n_train, n_test)` |
| McNemar's test | Comparing classifiers on same test set | `mcnemar_test(y_true, pred_a, pred_b)` |
| Wilcoxon signed-rank | Non-parametric alternative to paired t | `wilcoxon_test(a, b)` |

## Reproducibility Checklist

1. **Seed everything**: Pass `seed=` to models and `AutoMLConfig`
2. **Record data version**: Use `data_hash()` to detect silent data changes
3. **Pin preprocessing**: Log every transformer and its parameters
4. **Save the model**: Use `save_json()` or `save_msgpack()` for exact reproduction
5. **Record environment**: Python version, FerroML version, OS

```python
import ferroml
entry["ferroml_version"] = ferroml.__version__
```

## Version Management: Re-run vs Compare

| Situation | Action |
|-----------|--------|
| Data changed (new hash) | Re-run all experiments on new data |
| New feature engineering | Run new experiment, compare to baseline |
| Hyperparameter tuning | Run new experiment, compare to previous best |
| Different model type | Run new experiment, compare with statistical test |
| Code bug fix | Re-run affected experiments, update log |
| Same data + same params | No re-run needed, use cached result |

## Decision Framework: Is the New Model Better?

Follow this sequence:

```
1. Is the metric improvement > 0?
   NO  --> Keep current model
   YES --> Continue

2. Is the improvement practically meaningful?
   (e.g., accuracy +0.001 probably doesn't matter)
   NO  --> Keep current model (simpler is better)
   YES --> Continue

3. Is it statistically significant? (p < 0.05)
   NO  --> Collect more data or more CV folds
   YES --> Continue

4. Is the new model acceptable on secondary criteria?
   - Training time within budget?
   - Inference latency acceptable?
   - Model size reasonable?
   - Interpretability sufficient?
   NO  --> Keep current model
   YES --> Adopt new model
```

### Quick Significance Check

```python
from ferroml.metrics import paired_ttest

result = paired_ttest(new_scores, baseline_scores)
improvement = new_scores.mean() - baseline_scores.mean()

if result.p_value < 0.05 and improvement > 0:
    print(f"New model is significantly better (+{improvement:.4f}, p={result.p_value:.4f})")
elif result.p_value >= 0.05:
    print(f"Difference is not statistically significant (p={result.p_value:.4f})")
else:
    print(f"New model is significantly worse ({improvement:.4f}, p={result.p_value:.4f})")
```

## AutoML Experiment Tracking

When using FerroML's AutoML, it handles much of this internally:

```python
from ferroml.automl import AutoML, AutoMLConfig

config = AutoMLConfig(
    task="classification",
    metric="f1",
    time_budget_seconds=120,
    cv_folds=5,
    seed=42,
)
automl = AutoML(config)
automl.fit(X_train, y_train)

# AutoML already explored multiple models
# Log the winner
entry["model_type"] = "AutoML"
entry["model_params"] = {"config": config.__dict__}
entry["metrics"]["f1"] = f1_score(y_test, automl.predict(X_test))
entry["notes"] = "AutoML search, 2-minute budget"
```

# Drift Monitoring Guide

How to detect when your deployed model's world has changed and what to do about it.

## Types of Drift

| Drift type | What changes | How it manifests | Example |
|-----------|-------------|-----------------|---------|
| **Data drift** | Input feature distributions | P(X) changes | Users shift from desktop to mobile (different feature patterns) |
| **Concept drift** | Relationship between features and target | P(Y\|X) changes | Pandemic changes what "normal spending" means for fraud detection |
| **Prediction drift** | Output distribution | P(Y_hat) changes | Model starts predicting "positive" 80% of the time (was 30%) |
| **Label drift** | Target distribution | P(Y) changes | Fraud rate doubles due to new attack vector |

## Detection Methods

### Statistical tests available in FerroML

| Test | Function | What it tests | Best for |
|------|----------|--------------|----------|
| KS test (t-test proxy) | `ttest_ind(ref, current)` | Mean difference between distributions | Continuous features, normal-ish data |
| Welch's t-test | `welch_ttest(ref, current)` | Mean difference (unequal variance) | Continuous features, different sample sizes |
| Mann-Whitney U | `mann_whitney(ref, current)` | Distribution difference (non-parametric) | Any continuous feature, robust to outliers |
| Effect size | `cohens_d(ref, current)` | Magnitude of difference | Understanding practical significance |

### Feature-level drift detection

Compare each feature's distribution between a reference window (training data or recent "good" period) and current production data.

```python
from ferroml.stats import mann_whitney, cohens_d, adjust_pvalues
import numpy as np

def detect_feature_drift(X_reference, X_current, feature_names, alpha=0.05):
    """Check each feature for distribution drift."""
    p_values = []
    results = []

    for i, name in enumerate(feature_names):
        ref = X_reference[:, i]
        cur = X_current[:, i]

        stat, p_value = mann_whitney(ref, cur)
        effect = cohens_d(ref, cur)
        p_values.append(p_value)

        results.append({
            "feature": name,
            "p_value": p_value,
            "effect_size": abs(effect),
            "ref_mean": np.mean(ref),
            "cur_mean": np.mean(cur),
        })

    # Correct for multiple testing
    adjusted = adjust_pvalues(p_values, method="benjamini-hochberg")

    drifted = []
    for i, r in enumerate(results):
        r["adjusted_p"] = adjusted[i]
        r["drifted"] = adjusted[i] < alpha and r["effect_size"] > 0.2
        if r["drifted"]:
            drifted.append(r)

    return drifted, results
```

### Population Stability Index (PSI)

PSI measures how much a distribution has shifted. Compute it manually:

```python
import numpy as np

def compute_psi(reference, current, n_bins=10):
    """Population Stability Index between two distributions."""
    # Create bins from reference data
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    ref_counts = np.histogram(reference, bins=bins)[0] / len(reference)
    cur_counts = np.histogram(current, bins=bins)[0] / len(current)

    # Avoid division by zero
    ref_counts = np.clip(ref_counts, 1e-6, None)
    cur_counts = np.clip(cur_counts, 1e-6, None)

    psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))
    return psi
```

| PSI value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | No significant shift | Continue monitoring |
| 0.1 - 0.25 | Moderate shift | Investigate. May need retraining soon. |
| > 0.25 | Significant shift | Retrain the model. |

### Prediction drift detection

Monitor the distribution of model outputs over time:

```python
from ferroml.stats import mann_whitney

def check_prediction_drift(preds_reference, preds_current):
    """Compare prediction distributions."""
    stat, p_value = mann_whitney(preds_reference, preds_current)

    # Also check proportion shift for classification
    ref_positive_rate = np.mean(preds_reference == 1)
    cur_positive_rate = np.mean(preds_current == 1)
    rate_change = abs(cur_positive_rate - ref_positive_rate)

    return {
        "p_value": p_value,
        "ref_positive_rate": ref_positive_rate,
        "cur_positive_rate": cur_positive_rate,
        "rate_change": rate_change,
        "alert": p_value < 0.01 or rate_change > 0.1,
    }
```

## Monitoring Cadence

| What to check | Frequency | How |
|--------------|-----------|-----|
| Prediction distribution | Daily | Compare today's output distribution to last 30 days |
| Feature means/variances | Daily | Per-feature t-test against training data stats |
| PSI per feature | Weekly | Compare this week's data to training reference |
| Model accuracy (if labels available) | Weekly | Compare to validation baseline |
| Full drift report | Monthly | All features, all tests, effect sizes |
| Feature importance stability | Monthly | Retrain on recent data, compare importances |

## Alerting Thresholds

### Recommended alert levels

| Signal | Yellow (warning) | Red (action needed) |
|--------|-----------------|-------------------|
| PSI on any feature | > 0.1 | > 0.25 |
| Mann-Whitney p-value | < 0.01 on 3+ features | < 0.001 on 5+ features |
| Effect size (Cohen's d) | > 0.5 on any feature | > 0.8 on 3+ features |
| Prediction rate shift | > 5% from baseline | > 15% from baseline |
| Accuracy drop (if labels) | > 2% below baseline | > 5% below baseline |

### Reducing false alarms

1. **Use multiple-testing correction** (`adjust_pvalues`). With 50 features, you expect 2-3 false positives at p=0.05.
2. **Require both statistical significance AND practical significance** (effect size > 0.2).
3. **Use rolling windows** instead of comparing to a fixed reference. Gradual change is normal.
4. **Set different thresholds per feature** based on known variability.

## Feature Drift vs Target Drift

| Type | Detection | What it means |
|------|-----------|--------------|
| **Feature drift only** | Feature distributions change but target relationship is stable | Model may still be accurate. Monitor prediction quality. |
| **Target drift only** | Ground truth distribution changes | Model is learning the wrong base rate. Recalibrate or retrain. |
| **Both feature + target drift** | Both change together | Strongest signal for retraining. The world has changed. |
| **Concept drift** (hardest to detect) | Features and target stable, but P(Y\|X) changes | Need labeled data to detect. Compare recent accuracy to baseline. |

## Retraining Decision Framework

```
Is model accuracy dropping?
├── Yes
│   ├── Feature drift detected? → Retrain on recent data
│   ├── Concept drift suspected? → Retrain with recent data, consider model change
│   └── No drift detected? → Check for data quality issues, label noise
├── No, but drift detected
│   ├── Small drift (PSI < 0.25) → Monitor more closely, no action yet
│   └── Large drift (PSI > 0.25) → Preemptively retrain as insurance
└── No drift, no accuracy drop
    └── Continue monitoring. Retrain on schedule (quarterly).
```

### Retraining strategies

| Strategy | When to use |
|----------|------------|
| **Full retrain** on all historical data | Concept has not changed, just need more data |
| **Sliding window** retrain on last N months | Recent data is more relevant than old data |
| **Incremental update** with `partial_fit()` | SGD, GaussianNB, MiniBatchKMeans -- update without full retrain |
| **Ensemble with recent model** | Blend old model (stability) with new model (adaptation) |

Models supporting incremental updates:
- `SGDClassifier.partial_fit(X_new, y_new)`
- `SGDRegressor.partial_fit(X_new, y_new)`
- `GaussianNB.partial_fit(X_new, y_new)`
- `MiniBatchKMeans.partial_fit(X_new)`
- `PassiveAggressiveClassifier.partial_fit(X_new, y_new)`

## Complete Monitoring Pipeline

```python
import numpy as np
from ferroml.stats import mann_whitney, cohens_d, adjust_pvalues

class DriftMonitor:
    def __init__(self, X_reference, feature_names):
        self.X_ref = X_reference
        self.feature_names = feature_names
        self.ref_means = np.mean(X_reference, axis=0)
        self.ref_stds = np.std(X_reference, axis=0)

    def check(self, X_current, alpha=0.05):
        """Run full drift check. Returns list of drifted features."""
        p_values = []
        effects = []

        for i in range(X_current.shape[1]):
            _, p = mann_whitney(self.X_ref[:, i], X_current[:, i])
            d = abs(cohens_d(self.X_ref[:, i], X_current[:, i]))
            p_values.append(p)
            effects.append(d)

        adjusted = adjust_pvalues(p_values, method="benjamini-hochberg")

        alerts = []
        for i, name in enumerate(self.feature_names):
            if adjusted[i] < alpha and effects[i] > 0.2:
                alerts.append({
                    "feature": name,
                    "p_value": adjusted[i],
                    "effect_size": effects[i],
                    "severity": "red" if effects[i] > 0.8 else "yellow",
                })

        return alerts
```

## What NOT to do

| Mistake | Why it is wrong | Do this instead |
|---------|----------------|-----------------|
| Alert on every p < 0.05 | False positives with many features | Use adjusted p-values + effect size |
| Compare to training data forever | Legitimate gradual changes trigger false alarms | Use sliding reference windows |
| Retrain on every alert | Wastes resources, introduces instability | Require sustained drift + accuracy drop |
| Ignore drift because accuracy is fine | Accuracy may lag -- labels arrive late | Track drift as early warning signal |
| Monitor only predictions | Misses feature-level problems | Monitor features AND predictions |

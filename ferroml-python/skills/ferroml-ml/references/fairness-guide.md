# Fairness and Bias Auditing Guide

How to detect, measure, and mitigate bias in ML models using FerroML.

## Why This Matters

ML models learn patterns from historical data. If that data reflects systemic bias, the model will reproduce and amplify it. This is not just an ethical concern -- it creates legal and business risk.

## Protected Attributes

Features that should not influence decisions (directly or indirectly):

| Attribute | Examples | Common proxies to watch |
|-----------|---------|------------------------|
| Race/ethnicity | Race, skin color, national origin | Zip code, last name, language |
| Gender | Sex, gender identity | First name, job title |
| Age | Date of birth, age | Years of experience, graduation year |
| Religion | Religious affiliation | Neighborhood, holidays taken |
| Disability | Disability status | Medical history, gaps in employment |
| Marital/family | Marital status, pregnancy | Number of dependents |

**Proxy features:** Even if you remove the protected attribute, other features may encode the same information. Zip code correlates with race. First name correlates with gender. Audit for proxy discrimination.

## Fairness Metrics

### Demographic Parity (Statistical Parity)

**Definition:** P(Y_hat=1 | A=0) is approximately equal to P(Y_hat=1 | A=1)

The model approves at equal rates across groups, regardless of the true outcome.

```python
import numpy as np
from ferroml.metrics import confusion_matrix

def demographic_parity(y_pred, protected_attribute):
    """Compare positive prediction rates across groups."""
    groups = np.unique(protected_attribute)
    rates = {}
    for g in groups:
        mask = protected_attribute == g
        rates[g] = np.mean(y_pred[mask] == 1)

    # Disparate impact ratio (should be > 0.8)
    min_rate = min(rates.values())
    max_rate = max(rates.values())
    di_ratio = min_rate / max_rate if max_rate > 0 else 0

    return {
        "group_rates": rates,
        "disparate_impact_ratio": di_ratio,
        "passes_80_percent_rule": di_ratio >= 0.8,
    }
```

**When to use:** Hiring, lending, admissions -- any context where equal access matters.

**Limitation:** Does not account for different base rates. If group A genuinely has higher qualification rates, enforcing demographic parity may reduce accuracy.

### Equalized Odds

**Definition:** TPR and FPR are equal across groups.

The model makes the same types of errors at equal rates for all groups.

```python
import numpy as np

def equalized_odds(y_true, y_pred, protected_attribute):
    """Compare TPR and FPR across groups."""
    groups = np.unique(protected_attribute)
    metrics = {}

    for g in groups:
        mask = protected_attribute == g
        yt = y_true[mask]
        yp = y_pred[mask]

        tp = np.sum((yt == 1) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        metrics[g] = {"tpr": tpr, "fpr": fpr}

    # Check max gap
    tprs = [m["tpr"] for m in metrics.values()]
    fprs = [m["fpr"] for m in metrics.values()]

    return {
        "group_metrics": metrics,
        "tpr_gap": max(tprs) - min(tprs),
        "fpr_gap": max(fprs) - min(fprs),
        "passes": (max(tprs) - min(tprs)) < 0.1 and (max(fprs) - min(fprs)) < 0.1,
    }
```

**When to use:** Criminal justice, medical diagnosis -- contexts where error type matters.

### Calibration Across Groups

**Definition:** Among samples predicted P(Y=1) = p, the actual positive rate is approximately p, for all groups.

```python
import numpy as np

def calibration_by_group(y_true, y_prob, protected_attribute, n_bins=10):
    """Check if predicted probabilities are well-calibrated per group."""
    groups = np.unique(protected_attribute)
    results = {}

    for g in groups:
        mask = protected_attribute == g
        yt = y_true[mask]
        yp = y_prob[mask]

        bin_edges = np.linspace(0, 1, n_bins + 1)
        calibration_errors = []

        for i in range(n_bins):
            bin_mask = (yp >= bin_edges[i]) & (yp < bin_edges[i + 1])
            if np.sum(bin_mask) > 0:
                expected = np.mean(yp[bin_mask])
                actual = np.mean(yt[bin_mask])
                calibration_errors.append(abs(expected - actual))

        results[g] = np.mean(calibration_errors) if calibration_errors else 0

    return results
```

**When to use:** Risk scoring, insurance pricing -- when predicted probabilities are used directly.

## The 80% Rule (Four-Fifths Rule)

From the EEOC Uniform Guidelines on Employee Selection: the selection rate for any protected group should be at least 80% of the rate for the group with the highest rate.

```
Disparate Impact Ratio = (Selection rate of disadvantaged group) / (Selection rate of advantaged group)
```

| DI ratio | Interpretation | Action |
|----------|---------------|--------|
| >= 0.8 | Passes the four-fifths rule | Generally acceptable |
| 0.6 - 0.8 | Marginal | Investigate and document business necessity |
| < 0.6 | Fails the four-fifths rule | Likely discriminatory -- must mitigate |

## Complete Fairness Audit

```python
import numpy as np
from ferroml.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)

def fairness_audit(model, X_test, y_test, protected_attr, group_names=None):
    """Run a complete fairness audit on a model."""
    y_pred = model.predict(X_test)

    # Try to get probabilities
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        has_proba = True
    except Exception:
        y_prob = None
        has_proba = False

    groups = np.unique(protected_attr)
    report = {"overall": {}, "per_group": {}, "fairness": {}}

    # Overall metrics
    report["overall"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    # Per-group metrics
    for g in groups:
        mask = protected_attr == g
        name = group_names[g] if group_names else str(g)
        report["per_group"][name] = {
            "n_samples": int(np.sum(mask)),
            "positive_rate": float(np.mean(y_pred[mask] == 1)),
            "accuracy": accuracy_score(y_test[mask], y_pred[mask]),
            "precision": precision_score(y_test[mask], y_pred[mask]),
            "recall": recall_score(y_test[mask], y_pred[mask]),
            "f1": f1_score(y_test[mask], y_pred[mask]),
        }
        if has_proba:
            report["per_group"][name]["roc_auc"] = roc_auc_score(
                y_test[mask], y_prob[mask]
            )

    # Fairness metrics
    dp = demographic_parity(y_pred, protected_attr)
    eo = equalized_odds(y_test, y_pred, protected_attr)

    report["fairness"] = {
        "demographic_parity": dp,
        "equalized_odds": eo,
    }

    if has_proba:
        report["fairness"]["calibration"] = calibration_by_group(
            y_test, y_prob, protected_attr
        )

    return report
```

## Mitigation Strategies

### Pre-processing: Fix the data

| Method | How | FerroML tool |
|--------|-----|-------------|
| Resampling | Balance groups in training data | `RandomOverSampler`, `RandomUnderSampler`, `SMOTE` |
| Reweighting | Give underrepresented groups higher weight | `class_weight` parameter (where supported) |
| Feature removal | Remove protected attributes and strong proxies | Manual or `SelectKBest` (inspect what correlates with protected attr) |

```python
from ferroml.preprocessing import SMOTE

# Oversample underrepresented group
# First, identify minority group samples
minority_mask = protected_attr_train == minority_group
# Combine group membership with target for resampling
smote = SMOTE(k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### In-processing: Constrain the model

| Method | How |
|--------|-----|
| Regularization toward fairness | Add strong regularization (higher alpha) to reduce reliance on proxy features |
| Adversarial debiasing | Train a second model to predict protected attribute from predictions -- penalize success |
| Feature decorrelation | Use PCA or decorrelation to remove protected-attribute signal from features |

```python
from ferroml.decomposition import PCA
from ferroml.stats import correlation

# Check which features correlate with protected attribute
for i, name in enumerate(feature_names):
    corr = correlation(X_train[:, i], protected_attr_train.astype(float))
    if abs(corr) > 0.3:
        print(f"WARNING: {name} correlates with protected attribute (r={corr:.3f})")
```

### Post-processing: Adjust the outputs

| Method | How |
|--------|-----|
| Group-specific thresholds | Set different classification thresholds per group to equalize TPR/FPR |
| Reject option | Refuse to classify samples near the decision boundary |
| Calibration | Calibrate probabilities separately per group |

```python
import numpy as np

def find_fair_thresholds(y_true, y_prob, protected_attr, target_tpr=0.8):
    """Find per-group thresholds that equalize TPR."""
    groups = np.unique(protected_attr)
    thresholds = {}

    for g in groups:
        mask = protected_attr == g
        yt = y_true[mask]
        yp = y_prob[mask]

        # Binary search for threshold that gives target TPR
        best_thresh = 0.5
        best_gap = float("inf")

        for t in np.linspace(0.01, 0.99, 100):
            preds = (yp >= t).astype(int)
            tp = np.sum((yt == 1) & (preds == 1))
            fn = np.sum((yt == 1) & (preds == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            gap = abs(tpr - target_tpr)
            if gap < best_gap:
                best_gap = gap
                best_thresh = t

        thresholds[g] = best_thresh

    return thresholds
```

## Regulatory Context

### Key regulations

| Regulation | Jurisdiction | Key requirements |
|-----------|-------------|-----------------|
| **EU AI Act** | EU | Risk-based classification. High-risk AI (hiring, credit, law enforcement) requires conformity assessment, documentation, human oversight, bias testing. |
| **ECOA** (Equal Credit Opportunity Act) | US | Cannot discriminate in credit decisions based on race, sex, age, marital status. Requires adverse action notices. |
| **Fair Lending** (HMDA, CRA) | US | Lending decisions must be fair across protected classes. Regulators audit for disparate impact. |
| **Title VII** | US | Employment decisions cannot discriminate. Four-fifths rule for hiring. |
| **GDPR Art. 22** | EU | Right not to be subject to solely automated decisions. Requires human review for significant decisions. |

### Documentation requirements

For regulated models, maintain:

1. **Data documentation:** What data was used, how it was collected, known biases
2. **Fairness metrics:** Results of fairness audit (demographic parity, equalized odds, calibration)
3. **Mitigation steps:** What was done to address identified bias
4. **Business justification:** If disparate impact exists, document the business necessity
5. **Human oversight plan:** How humans review model decisions, especially edge cases
6. **Model card:** Use `ModelClass.model_card()` to generate structured metadata

## Fairness Audit Checklist

Before deploying any model that affects people:

- [ ] Identify all protected attributes relevant to the use case
- [ ] Check for proxy features (correlation > 0.3 with protected attributes)
- [ ] Compute demographic parity ratio (should be >= 0.8)
- [ ] Compute equalized odds (TPR gap < 0.1, FPR gap < 0.1)
- [ ] If using probabilities: check calibration across groups
- [ ] Compute accuracy/F1/AUC per group (no group should be much worse)
- [ ] Apply mitigation if any metric fails
- [ ] Document findings and mitigation steps
- [ ] Plan ongoing monitoring for fairness metrics in production
- [ ] Establish human review process for contested decisions

## What NOT to Do

| Mistake | Why it is wrong |
|---------|----------------|
| Remove protected attributes and assume fairness | Proxy features carry the same signal |
| Optimize only for overall accuracy | Can sacrifice minority group performance |
| Test fairness only at deployment | Drift can introduce bias over time |
| Assume "the data is objective" | Historical data encodes historical discrimination |
| Use a single fairness metric | Different metrics capture different concerns -- use multiple |
| Ignore intersectionality | Bias may affect subgroups (e.g., Black women) more than individual protected classes |

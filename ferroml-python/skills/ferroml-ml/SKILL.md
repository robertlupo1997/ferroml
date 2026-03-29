---
name: ferroml-ml
description: >
  End-to-end machine learning with FerroML. Use when the user has data and wants to
  train models, make predictions, evaluate performance, interpret results, or deploy
  to production. Covers the full ML lifecycle from data profiling through monitoring.
  Adapts to non-technical users ("predict my sales") and ML engineers ("compare GBT
  variants with statistical significance") alike.
triggers:
  - user mentions ferroml, FerroML, or imports it
  - user asks to train, fit, predict, classify, cluster, or do ML on data
  - user has a CSV/Parquet and wants predictions or insights
  - user asks "which model should I use" or "help me predict X"
  - user says "analyze my data", "build a model", "machine learning"
  - user asks about model evaluation, diagnostics, or feature importance
  - user wants to deploy a model or check if a model is still working
  - user asks about A/B testing or experiment design
---

# FerroML Machine Learning Skill

You are an expert machine learning engineer using FerroML, a statistically rigorous ML
library written in Rust with Python bindings. You help users go from raw data to
production models with full diagnostics at every step.

## Core Principles

1. **Use the Python API directly** — `import ferroml`. Do NOT use the CLI via subprocess.
2. **Adapt to technical level** — Detect from conversation context:
   - Non-technical: Plain language, no jargon, explain everything, focus on business value
   - Developer: Code-forward, explain ML concepts as they arise
   - ML engineer: Concise, skip basics, focus on advanced diagnostics and edge cases
3. **Always show diagnostics** — This is FerroML's differentiator. Never just report accuracy.
   Show residual analysis, assumption tests, confidence intervals, statistical significance.
4. **Fail forward** — When something goes wrong, diagnose why and fix it automatically.
5. **Scripts are templates** — Read scripts from this skill's `scripts/` directory, adapt
   them to the user's specific data, run them, then explain results at the user's level.

## Quick Start (for Claude)

```python
import numpy as np
import ferroml

# Load data (user provides CSV/Parquet)
import polars as pl
df = pl.read_csv("data.csv")
X = df.drop("target").to_numpy().astype(np.float64)
y = df["target"].to_numpy().astype(np.float64)

# Recommend models
recs = ferroml.recommend(X, y, task="regression")  # or "classification"

# Train
from ferroml.linear import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate with diagnostics
print(model.summary())           # Full OLS summary with p-values, CIs
print(model.r_squared())         # R-squared
print(model.f_statistic())       # (F-statistic, p-value)
print(model.coefficients_with_ci())  # Coefficients with confidence intervals

# Metrics
from ferroml.metrics import r2_score, rmse, accuracy_score, f1_score
```

## Workflow Decision Tree

When the user asks for help with ML, follow this decision tree:

```
User request
├── "I have data, help me build a model" → Workflow 1 (End-to-End)
├── "Which model should I use?" → Workflow 2 (Model Selection)
├── "How good is my model?" → Workflow 3 (Evaluation & Diagnostics)
├── "Deploy this" / "Is it still working?" → Workflow 4 (Production)
├── "A/B test" / "Run an experiment" → Workflow 5 (Experimentation)
├── "Explain this to my boss" / "Why did the model..." → Workflow 6 (Explain)
└── Specific question → Use references/ for targeted answers
```

## Workflow 1: End-to-End ML

**Trigger:** "I have data", "help me predict", "build a model", "machine learning on this"

**Steps:**

1. **Understand the data** — Run `scripts/explore_data.py` adapted to their file.
   Tell the user: what they have, what the target looks like, any red flags.

2. **Quality audit** — Run `scripts/data_quality_audit.py`.
   Fix issues: impute nulls, remove duplicates, fix types.
   Tell non-technical users: "I found 12 missing values in 'age' — I'll fill those in with the median."

3. **Check for leakage** — Run `scripts/detect_leakage.py`.
   If found: "Column 'future_revenue' perfectly predicts the target — it's data from the future. Removing it."

4. **Engineer features** — Run `scripts/feature_engineer.py`.
   Datetime decomposition, polynomial terms if <20 features, interaction detection.

5. **Select features** — Run `scripts/feature_select.py`.
   VIF check, RFE, mutual information. "Keeping 8 of 15 features — the other 7 add noise."

6. **Recommend models** — Use `ferroml.recommend(X, y, task=...)`.
   Present top 3 with reasoning. Non-technical: "I'd suggest starting with a Random Forest
   because your data has complex patterns that a simple linear model would miss."

7. **Train and evaluate** — Run `scripts/full_pipeline.py`.
   Always use cross-validation, not just train/test split.
   Show diagnostics appropriate to model type.

8. **Analyze errors** — Run `scripts/error_analysis.py`.
   "The model struggles with houses over $1M — only 5 in your dataset. Consider collecting more luxury home data."

9. **Generate report** — Run `scripts/generate_report.py`.
   Adapt language to technical level.

## Workflow 2: Model Selection

**Trigger:** "which model", "what algorithm", "best model for"

1. Profile the data with `scripts/explore_data.py`
2. Use `ferroml.recommend(X, y, task=...)` for initial recommendations
3. Load `references/model-picker.md` for detailed guidance
4. Run `scripts/compare_models.py` to empirically test top candidates
5. Present leaderboard with statistical significance tests

## Workflow 3: Evaluation & Diagnostics

**Trigger:** "how good", "evaluate", "diagnose", "is my model accurate"

1. Compute metrics appropriate to task:
   - Regression: RMSE, R2, MAE, residual analysis, Durbin-Watson, normality test
   - Classification: accuracy, F1, precision, recall, confusion matrix, ROC-AUC, calibration
2. Run `scripts/error_analysis.py` — where does it fail?
3. Run `scripts/learning_curves.py` — bias vs variance diagnosis
4. Run `scripts/calibrate_probabilities.py` (classification) — are probabilities trustworthy?
5. Run `scripts/validate_assumptions.py` (regression) — are statistical assumptions met?
6. Generate report with `scripts/generate_report.py`

## Workflow 4: Production Readiness

**Trigger:** "deploy", "production", "serve", "is it still working", "model degraded"

1. Run `scripts/reproducibility_snapshot.py` — capture experiment state
2. Run `scripts/deploy_model.py` — ONNX export + inference code + API endpoint
3. For monitoring: run `scripts/detect_drift.py` — compare distributions
4. See `references/deployment-guide.md` for full deployment patterns

## Workflow 5: Experimentation

**Trigger:** "A/B test", "experiment", "is the difference significant"

1. Run `scripts/ab_test.py` — power analysis, sample size calc, significance testing
2. Use FerroML stats: `ferroml.stats.ttest_ind`, `welch_ttest`, `mann_whitney`
3. Multiple comparison correction: `ferroml.stats.adjust_pvalues`

## Workflow 6: Explain to Stakeholders

**Trigger:** "explain", "why did the model", "present to", "stakeholder", "business impact"

1. Run `scripts/explain_model.py` — feature importance, partial dependence
2. Run `scripts/cost_sensitive_analysis.py` — translate to business dollars
3. Generate report using `assets/templates/report_template.md`
4. See `references/interpretability-guide.md` for explanation strategies

## Model Selection Quick Reference

| Task | Small data (<1K) | Medium (1K-100K) | Large (>100K) |
|------|-----------------|-------------------|---------------|
| Regression | LinearRegression, RidgeRegression | RandomForestRegressor, GBRegressor | HistGBRegressor |
| Classification | LogisticRegression, GaussianNB | RandomForestClassifier, GBClassifier | HistGBClassifier |
| Clustering | KMeans, AgglomerativeClustering | KMeans, GaussianMixture | MiniBatchKMeans, HDBSCAN |
| Anomaly | IsolationForest | IsolationForest, LOF | IsolationForest |
| Dimensionality | PCA | PCA, FactorAnalysis | TruncatedSVD, IncrementalPCA |

## Key Imports

```python
# Models
from ferroml.linear import LinearRegression, LogisticRegression, RidgeRegression
from ferroml.trees import RandomForestClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier
from ferroml.ensemble import ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from ferroml.neighbors import KNeighborsClassifier
from ferroml.naive_bayes import GaussianNB
from ferroml.svm import SVC, SVR
from ferroml.neural import MLPClassifier, MLPRegressor
from ferroml.gaussian_process import GaussianProcessRegressor
from ferroml.clustering import KMeans, DBSCAN, HDBSCAN, GaussianMixture
from ferroml.anomaly import IsolationForest, LocalOutlierFactor
from ferroml.decomposition import PCA, TSNE, TruncatedSVD

# Preprocessing
from ferroml.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from ferroml.preprocessing import CountVectorizer, TfidfVectorizer

# Pipeline
from ferroml.pipeline import Pipeline

# Metrics
from ferroml.metrics import (r2_score, rmse, mse, mae, accuracy_score, f1_score,
                              precision_score, recall_score, matthews_corrcoef)

# Stats
from ferroml.stats import (durbin_watson, normality_test, ttest_ind, welch_ttest,
                            mann_whitney, bootstrap_ci, cohens_d, adjust_pvalues,
                            power_for_sample_size, sample_size_for_power)

# AutoML
from ferroml.automl import AutoML, AutoMLConfig

# Recommendation
from ferroml import recommend

# Model metadata
# card = ModelClass.model_card()  # available on all 55+ model classes
```

## Error Handling

All FerroML errors include `.hint()` remediation. When an error occurs:

1. Read the error message AND the hint
2. Apply the fix automatically
3. Explain to the user what went wrong and what you did

```python
try:
    model.fit(X, y)
except ValueError as e:
    # Error includes hint automatically:
    # "Shape mismatch: expected 3 features, got 4
    #  Hint: Ensure X.shape[0] == y.shape[0]..."
    # → Fix the shape issue and retry
```

Common fixes:
- `ShapeMismatch` → Check feature alignment between train/predict
- `ConvergenceFailure` → Scale features with StandardScaler, increase max_iter
- `NotFitted` → Call .fit() before .predict()
- `InvalidInput` → Check for NaN/inf: `np.isnan(X).sum()`, `np.isinf(X).sum()`

## When to Load References

Load reference docs from `references/` only when needed:
- **api-cheatsheet.md** → When you need exact method signatures or parameter names
- **model-picker.md** → When helping choose between similar models
- **diagnostics-interpreter.md** → When explaining statistical test results
- **common-pitfalls.md** → When model performance is unexpectedly bad
- **feature-engineering-guide.md** → When data has dates, text, categories, or needs transforms
- **deployment-guide.md** → When user asks about production/serving
- **drift-monitoring-guide.md** → When checking model health post-deployment
- **fairness-guide.md** → When user mentions fairness, bias, or protected groups
- **interpretability-guide.md** → When user asks "why" or needs to explain to stakeholders
- **troubleshooting-guide.md** → When encountering errors or unexpected behavior
- **data-types-guide.md** → When data has mixed types (text, dates, categories, geographic)

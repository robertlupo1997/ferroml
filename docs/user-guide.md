# FerroML User Guide

> **Statistically Rigorous AutoML in Rust**

This guide provides conceptual explanations of FerroML's features and design philosophy. Whether you're a data scientist seeking production-ready ML, a statistician requiring proper inference, or a developer wanting fast, reliable predictions, this guide will help you understand FerroML's approach.

## Table of Contents

1. [Philosophy: Statistical Rigor First](#philosophy-statistical-rigor-first)
2. [Getting Started](#getting-started)
3. [Core Concepts](#core-concepts)
4. [Statistical Foundations](#statistical-foundations)
5. [Models](#models)
6. [Dimensionality Reduction](#dimensionality-reduction)
7. [Clustering](#clustering)
8. [Preprocessing](#preprocessing)
9. [Cross-Validation](#cross-validation)
10. [Hyperparameter Optimization](#hyperparameter-optimization)
11. [Pipelines](#pipelines)
12. [Ensembles](#ensembles)
13. [Explainability](#explainability)
14. [AutoML](#automl)
15. [Deployment](#deployment)
16. [Best Practices](#best-practices)

---

## Philosophy: Statistical Rigor First

### The Problem with Traditional AutoML

Most AutoML tools optimize a single metric (accuracy, AUC) and return a "best model." But this approach hides critical questions:

- **How confident are we in this performance estimate?** A model with 85% accuracy might actually perform anywhere from 80% to 90% in production.
- **Are the assumptions valid?** Linear models assume linearity; tree models assume feature interactions matter. What if these assumptions are wrong?
- **Is the difference meaningful?** Model A has 87% accuracy vs Model B's 86%. Is this a real improvement or random noise?

### FerroML's Answer: Make Statistics Explicit

FerroML treats statistical inference as a first-class citizen:

```
Traditional AutoML:     "Best model: RandomForest, accuracy: 0.87"

FerroML:                "Best model: RandomForest
                         Accuracy: 0.87 (95% CI: [0.84, 0.90])
                         vs. 2nd place LogisticRegression:
                           Difference: 0.02 (p = 0.23, not significant)
                         Assumption tests: All passed
                         Effect size: Cohen's d = 0.15 (small)"
```

This isn't just about adding numbers. It's about answering the question that matters: **"Can I trust this model in production?"**

### The Five Pillars of Statistical Rigor

1. **Uncertainty Quantification**: Every prediction includes confidence intervals. Every metric includes standard errors.

2. **Assumption Testing**: Models test their own assumptions. Linear regression checks for normality, homoscedasticity, and linearity. If assumptions fail, you're warned.

3. **Effect Sizes**: P-values alone are insufficient. FerroML reports effect sizes (Cohen's d, rank-biserial correlation) to answer "how big is this effect?"

4. **Multiple Testing Correction**: When comparing many models, false positives accumulate. FerroML applies Bonferroni, Holm, or Benjamini-Hochberg corrections automatically.

5. **Reproducibility**: Deterministic by default. Every random operation accepts a seed. Results are the same across runs.

---

## Getting Started

### Installation

**Rust (library)**:
```toml
[dependencies]
ferroml-core = "0.1"
```

**Python (bindings)**:
```bash
pip install ferroml
```

### Your First Model

```python
from ferroml.linear import LinearRegression
import numpy as np

# Generate data with known coefficients
np.random.seed(42)
X = np.random.randn(200, 3)
true_coefs = np.array([2.0, -1.0, 0.5])
y = X @ true_coefs + np.random.randn(200) * 0.5  # Add noise

# Fit model
model = LinearRegression()
model.fit(X, y)

# Get statistical summary (R-style output)
print(model.summary())
```

Output:
```
                    OLS Regression Results
==================================================================
R-squared:          0.9412      Adj. R-squared:     0.9403
F-statistic:        1046.3      Prob (F-statistic): 0.0000
No. Observations:   200         Df Residuals:       196

              coef    std err      t      P>|t|    [0.025    0.975]
------------------------------------------------------------------
const        0.0234    0.0355    0.659    0.511   -0.0466    0.0934
x1           1.9823    0.0368   53.861    0.000    1.9097    2.0549
x2          -0.9901    0.0341  -29.036    0.000   -1.0574   -0.9228
x3           0.5112    0.0354   14.441    0.000    0.4414    0.5810
==================================================================

Assumption Tests:
  Normality (Shapiro-Wilk): W=0.994, p=0.612 [PASS]
  Homoscedasticity (Breusch-Pagan): LM=2.34, p=0.505 [PASS]
  Autocorrelation (Durbin-Watson): d=2.01 [PASS]

Influential Observations: 2 (Cook's d > 0.02)
VIF (Variance Inflation Factors): All < 5 [OK]
```

Notice what you get automatically:
- Coefficient confidence intervals
- Hypothesis tests for each coefficient
- Model-level F-test
- Three diagnostic tests for assumptions
- Multicollinearity check (VIF)
- Influential observation detection (Cook's distance)

---

## Core Concepts

### Confidence Intervals

A confidence interval (CI) quantifies uncertainty. A 95% CI means: "If we repeated this experiment 100 times, 95 of those intervals would contain the true value."

FerroML provides CIs for:
- **Model coefficients**: "The true effect of feature X is between 0.8 and 1.2"
- **Performance metrics**: "True AUC is between 0.82 and 0.88"
- **Predictions**: "This customer's predicted spend is $450 ± $50"

**Why this matters**: A model with "accuracy = 0.85" tells you nothing about reliability. A model with "accuracy = 0.85 (95% CI: [0.78, 0.92])" tells you the realistic range of expected performance.

### Effect Sizes

Statistical significance (p-values) answers: "Is there an effect?" Effect sizes answer: "How big is the effect?"

A p-value of 0.001 might correspond to:
- A large effect (Cohen's d = 1.2): Practically important
- A tiny effect (Cohen's d = 0.05): Statistically significant but meaningless

FerroML reports standardized effect sizes:
- **Cohen's d**: For comparing means (small: 0.2, medium: 0.5, large: 0.8)
- **Hedges' g**: Bias-corrected version of Cohen's d
- **Rank-biserial correlation**: For non-parametric comparisons

### Assumption Testing

Every statistical method makes assumptions. Violating assumptions can invalidate results.

**Linear Regression assumes:**
1. **Linearity**: Relationship between X and y is linear
2. **Normality**: Residuals are normally distributed
3. **Homoscedasticity**: Constant variance of residuals
4. **Independence**: No autocorrelation in residuals
5. **No multicollinearity**: Features aren't highly correlated

FerroML tests these automatically:
- Shapiro-Wilk test for normality
- Breusch-Pagan test for homoscedasticity
- Durbin-Watson test for autocorrelation
- VIF for multicollinearity

If assumptions fail, FerroML warns you and suggests alternatives (e.g., robust regression for non-normality).

### Multiple Testing Correction

When you compare 10 models, even with a 5% false positive rate, you expect 0.5 false positives on average. With 100 comparisons, expect 5 spurious "significant" results.

FerroML applies corrections automatically:
- **Bonferroni**: Divide α by number of tests. Conservative but simple.
- **Holm**: Step-down procedure. More powerful than Bonferroni.
- **Benjamini-Hochberg**: Controls false discovery rate. Good for exploration.

---

## Statistical Foundations

### The `stats` Module

FerroML's statistical foundation includes:

**Hypothesis Tests**:
- One-sample, two-sample, and paired t-tests
- Mann-Whitney U test (non-parametric alternative to t-test)
- Wilcoxon signed-rank test
- Chi-squared test

**Confidence Intervals**:
- Normal approximation (large samples)
- Student's t (small samples)
- Bootstrap (distribution-free)

**Effect Sizes**:
- Cohen's d, Hedges' g (for means)
- Rank-biserial correlation (for ranks)
- Glass's delta (when groups have different variances)

**Power Analysis**:
- Calculate required sample size for desired power
- Determine power given sample size and effect size
- Essential for experiment design

**Bootstrap Methods**:
- Percentile bootstrap
- BCa (bias-corrected and accelerated)
- Studentized bootstrap

```rust
use ferroml_core::stats::{TTest, EffectSize, BootstrapCI};

// Compare two groups
let group_a = vec![5.2, 4.8, 5.1, 5.3, 4.9];
let group_b = vec![4.1, 4.3, 4.0, 4.2, 4.4];

// T-test with effect size
let result = TTest::two_sample(&group_a, &group_b).run();
println!("t = {:.3}, p = {:.4}", result.statistic, result.p_value);
println!("Cohen's d = {:.3} ({})", result.effect_size, result.effect_interpretation);
// Output: t = 4.382, p = 0.0023
//         Cohen's d = 2.77 (large)

// Bootstrap CI for mean difference
let ci = BootstrapCI::new(&group_a, &group_b)
    .confidence_level(0.95)
    .n_resamples(10000)
    .compute();
println!("95% CI for difference: [{:.2}, {:.2}]", ci.lower, ci.upper);
```

---

## Models

### Design Philosophy

Every FerroML model provides:
1. **Point predictions**: Standard `predict()` method
2. **Uncertainty quantification**: Prediction intervals via `predict_with_interval()`
3. **Statistical summary**: Coefficient inference via `summary()`
4. **Assumption testing**: Automatic diagnostic checks
5. **Feature importance**: Which features matter most

### Linear Models

**LinearRegression**: Ordinary least squares with full diagnostics

The workhorse of statistical modeling. FerroML's implementation provides:
- QR decomposition for numerical stability
- Coefficient standard errors via $(X^TX)^{-1}\sigma^2$
- Prediction intervals using t-distribution
- Residual diagnostics (normality, heteroscedasticity, autocorrelation)
- Cook's distance for influential observations
- VIF for multicollinearity detection

**LogisticRegression**: Maximum likelihood classification

Binary and multiclass classification with:
- IRLS (Iteratively Reweighted Least Squares) fitting
- Odds ratios with confidence intervals
- Wald and likelihood ratio tests
- ROC-AUC with bootstrap CI
- Pseudo R-squared measures

**Regularized Models**:
- **Ridge**: L2 penalty, shrinks coefficients toward zero
- **Lasso**: L1 penalty, produces sparse solutions
- **ElasticNet**: L1 + L2 combination

Each supports cross-validated hyperparameter selection (RidgeCV, LassoCV, ElasticNetCV).

**Additional Linear Models**:
- **RidgeClassifier**: Ridge regression applied to classification (thresholding the regression output)
- **Perceptron**: Single-layer linear classifier with online learning
- **QuantileRegression**: Predicts conditional quantiles instead of the mean

**Robust Regression**: When assumptions fail

- **Huber**: Downweights outliers smoothly
- **Bisquare (Tukey)**: Zero weight for extreme outliers
- **Hampel**: Three-part weight function
- **Andrew's Wave**: Sine-based redescending estimator

### Tree-Based Models

**DecisionTree**: Interpretable, handles non-linear relationships

FerroML's trees include:
- Gini impurity and entropy criteria
- Cost-complexity pruning (α parameter)
- Feature importance from impurity decrease
- DOT format export for visualization

**RandomForest**: Bootstrap aggregating for variance reduction

- OOB (Out-of-Bag) error estimation (free cross-validation)
- Feature importance with bootstrap confidence intervals
- Parallel tree building via rayon

**GradientBoosting**: Sequential boosting for bias reduction

- Multiple loss functions (squared error, absolute error, Huber)
- Learning rate scheduling (constant, decay)
- Early stopping with validation set
- Stochastic gradient boosting (subsample rows)

**HistGradientBoosting**: LightGBM-style efficient boosting

The most advanced tree algorithm in FerroML:
- **Histogram-based splits**: O(n) instead of O(n log n)
- **Native missing value handling**: No imputation needed
- **Monotonic constraints**: Enforce domain knowledge (e.g., "price must not increase demand")
- **Feature interaction constraints**: Control which features can interact
- **L1/L2 regularization**: Prevent overfitting

```python
from ferroml.trees import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    monotonic_constraints=[1, 0, -1],  # Feature 0: increasing, Feature 2: decreasing
    early_stopping=True,
    validation_fraction=0.1,
)
model.fit(X_train, y_train)
```

### Support Vector Machines

**SVC/SVR**: Kernel-based models for complex boundaries

- Kernels: Linear, RBF, Polynomial, Sigmoid
- Platt scaling for probability calibration
- One-vs-One and One-vs-Rest multiclass
- Epsilon-insensitive loss for regression

**LinearSVC/LinearSVR**: Fast linear versions

- O(n·d) memory vs O(n²) for kernelized
- Coordinate descent optimization
- Feature importance from weight magnitudes

### Naive Bayes

**GaussianNB**: Assumes Gaussian features
**MultinomialNB**: For count data (text classification)
**BernoulliNB**: For binary features
**CategoricalNB**: For discrete categorical features

All support incremental learning via `partial_fit()`.

### K-Nearest Neighbors

**KNeighborsClassifier/Regressor**:
- Distance metrics: Euclidean, Manhattan, Minkowski
- Weighting: Uniform or distance-based
- Spatial indexing: KD-Tree, Ball Tree

**NearestCentroid**: Classifies by nearest class centroid (Rocchio classifier).

### Discriminant Analysis

**QuadraticDiscriminantAnalysis (QDA)**: Quadratic classification boundaries

Unlike LDA (which assumes shared covariance), QDA estimates a separate covariance matrix per class, producing quadratic decision boundaries:
- Per-class mean and covariance estimation
- Regularization parameter to handle ill-conditioned covariance matrices
- `predict_proba()` for calibrated posterior probabilities
- Handles cases where classes have very different covariance structures

### Isotonic Regression

**IsotonicRegression**: Non-parametric monotonic regression

Fits a non-decreasing (or non-increasing) step function using the pool adjacent violators algorithm (PAVA):
- Guarantees monotonicity of predictions
- No parametric assumptions on the functional form
- Commonly used for probability calibration
- Supports both increasing and decreasing constraints

### Anomaly Detection

**IsolationForest**: Tree-based anomaly detection

Isolates anomalies by random recursive partitioning. Anomalies require fewer splits to isolate:
- Configurable contamination parameter (expected proportion of outliers)
- `predict()` returns +1 (inlier) or -1 (outlier)
- `score_samples()` returns anomaly scores (lower = more anomalous)
- Random seed support for reproducibility

**LocalOutlierFactor (LOF)**: Density-based anomaly detection

Compares local density of a point to its neighbors. Points with substantially lower density are outliers:
- Configurable number of neighbors (k)
- Configurable contamination parameter
- `predict()` returns +1 (inlier) or -1 (outlier)
- `score_samples()` returns negative LOF scores

Both anomaly detectors implement the `OutlierDetector` trait:
- `fit(X)`: Learn the data distribution
- `predict(X)`: Classify as inlier (+1) or outlier (-1)
- `score_samples(X)`: Continuous anomaly scores

### Probability Calibration

**TemperatureScalingCalibrator**: Post-hoc probability calibration

Learns a single temperature parameter to calibrate model probabilities via Platt scaling.

---

## Dimensionality Reduction

### PCA (Principal Component Analysis)

Linear dimensionality reduction via singular value decomposition:
- Explained variance ratio for component selection
- `inverse_transform()` for reconstruction
- Efficient for high-dimensional data

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

Non-linear dimensionality reduction for visualization:
- Exact O(N^2) algorithm
- Barnes-Hut O(N log N) approximation for large datasets
- Distance metrics: Euclidean, Manhattan, Cosine
- Initialization: PCA-based or random
- Configurable perplexity (controls local/global structure balance)
- Learning rate and momentum scheduling
- Early exaggeration for better global structure

```python
from ferroml.decomposition import TSNE

tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000)
embedding = tsne.fit_transform(X)
# embedding is (n_samples, 2) — ready for scatter plots
```

t-SNE is best used for visualization (2D or 3D). For feature extraction or preprocessing, prefer PCA.

---

## Clustering

### Algorithms

**KMeans**: Centroid-based clustering with k-means++ initialization and elbow method.

**DBSCAN**: Density-based clustering that discovers arbitrary-shaped clusters and identifies noise points.

**AgglomerativeClustering**: Hierarchical clustering with Ward, complete, single, and average linkage.

**GaussianMixture (GMM)**: Probabilistic soft clustering via Expectation-Maximization

Models data as a mixture of Gaussian distributions:
- Four covariance types: Full, Tied, Diagonal, Spherical
- EM algorithm with convergence detection
- Model selection via BIC (Bayesian Information Criterion) and AIC
- `predict_proba()` for soft cluster assignments
- `sample()` for generating new data from the learned distribution
- Configurable number of components, convergence tolerance, and max iterations

```python
from ferroml.clustering import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type="full", max_iter=100)
gmm.fit(X)
labels = gmm.predict(X)        # Hard assignments
probs = gmm.predict_proba(X)   # Soft assignments
bic = gmm.bic(X)               # Model selection criterion
```

**HDBSCAN**: Hierarchical density-based clustering

An extension of DBSCAN that finds clusters of varying densities and automatically determines the number of clusters:
- Builds a mutual reachability graph and minimum spanning tree
- Extracts clusters using excess-of-mass method on the condensed tree
- Points not belonging to any cluster are labeled as noise (-1)
- Returns cluster membership probabilities
- No need to specify number of clusters (unlike KMeans)
- Configurable `min_cluster_size`, `min_samples`, and `cluster_selection_epsilon`

```python
from ferroml.clustering import HDBSCAN

hdbscan = HDBSCAN(min_cluster_size=5)
hdbscan.fit(X)
labels = hdbscan.labels_          # Cluster labels (-1 for noise)
probs = hdbscan.probabilities_    # Cluster membership probabilities
n = hdbscan.n_clusters_           # Number of clusters found
```

### Clustering Metrics

Silhouette score, Davies-Bouldin index, Calinski-Harabasz index, adjusted Rand index, adjusted mutual information, V-measure, Fowlkes-Mallows index.

---

## Preprocessing

### The Transformer Pattern

All preprocessors follow the `Transformer` trait:
- `fit(X)`: Learn parameters from data
- `transform(X)`: Apply transformation
- `fit_transform(X)`: Both in one call
- `inverse_transform(X)`: Reverse (when possible)

### Scalers

**StandardScaler**: Zero mean, unit variance
$$X_{scaled} = \frac{X - \mu}{\sigma}$$

Essential for:
- Distance-based models (KNN, SVM)
- Gradient descent optimization
- PCA and other decompositions

**MinMaxScaler**: Scale to [0, 1] range
$$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

Use when features must be bounded.

**RobustScaler**: Uses median and IQR
$$X_{scaled} = \frac{X - median}{IQR}$$

Robust to outliers. Use when data has extreme values.

**MaxAbsScaler**: Scale by maximum absolute value
$$X_{scaled} = \frac{X}{|X|_{max}}$$

Preserves sparsity (zeros remain zeros).

### Power Transforms

**PowerTransformer**: Make data more Gaussian

- **Box-Cox**: For positive data, finds optimal λ
- **Yeo-Johnson**: Handles negative values

Tree models don't need this, but linear models and neural networks benefit from normalized distributions.

### Encoders

**OneHotEncoder**: Creates binary columns for each category

Before: `color = ["red", "blue", "red"]`
After: `color_red = [1, 0, 1], color_blue = [0, 1, 0]`

**OrdinalEncoder**: Maps categories to integers

Preserves order for ordinal categories (low, medium, high → 0, 1, 2).

**TargetEncoder**: Uses target mean per category

Reduces dimensionality vs one-hot. Uses cross-validation to prevent leakage.

### Imputation

**SimpleImputer**: Mean, median, mode, or constant

Use median for data with outliers; mode for categorical.

**KNNImputer**: Uses K-nearest neighbors

More sophisticated; considers feature correlations.

### Feature Selection

**VarianceThreshold**: Remove low-variance features

Features with near-zero variance provide no signal.

**SelectKBest**: Keep top K by statistical test

Scoring functions: f_classif (ANOVA), f_regression (F-test), chi2.

**RecursiveFeatureElimination**: Iteratively remove weakest

Fits model, removes least important feature, repeats.

### Resampling (Class Imbalance)

**SMOTE**: Creates synthetic minority samples

Interpolates between minority samples and their neighbors.

**ADASYN**: Adaptive synthetic sampling

More synthetics where minority samples are harder to learn.

**RandomUnderSampler/RandomOverSampler**: Simple resampling

Under: Remove majority samples. Over: Duplicate minority samples.

**SMOTE-Tomek, SMOTE-ENN**: Combined methods

SMOTE + cleaning step for better boundaries.

---

## Cross-Validation

### Why Cross-Validation?

A single train/test split is unreliable:
- Results depend heavily on which samples end up in test
- Variance is high, especially with small datasets

Cross-validation provides:
1. **More reliable estimates**: Average over multiple folds
2. **Confidence intervals**: Quantify uncertainty in performance
3. **Full data utilization**: Every sample is tested once

### CV Strategies

**KFold**: Standard k-fold

Splits data into k folds; each is test set once.

**StratifiedKFold**: Preserves class distribution

Essential for imbalanced classification.

**TimeSeriesSplit**: Respects temporal order

Training always uses past data; never future. Supports:
- Expanding window (all past data)
- Sliding window (fixed window size)
- Gap between train and test

**GroupKFold**: Never splits groups

When samples belong to groups (e.g., multiple measurements per patient), ensures all samples from a group are in same fold.

**NestedCV**: Proper hyperparameter tuning

Outer loop: Evaluate model performance
Inner loop: Tune hyperparameters

Prevents overfitting to validation set. Essential for model selection.

### CV Results with Statistics

```python
from ferroml.cv import cross_val_score

result = cross_val_score(model, X, y, cv=5, metric="accuracy")

print(f"Mean: {result.mean:.3f}")
print(f"Std:  {result.std:.3f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
print(f"Per-fold: {result.scores}")
```

FerroML uses the corrected t-distribution for CI calculation, accounting for the small number of folds.

---

## Hyperparameter Optimization

### The Search Space

Define what to search:

```rust
use ferroml_core::hpo::{SearchSpace, ParameterRange};

let space = SearchSpace::new()
    .add("n_estimators", ParameterRange::int(10, 200))
    .add("max_depth", ParameterRange::int(2, 20))
    .add("learning_rate", ParameterRange::float(0.001, 0.3).log_scale())
    .add("subsample", ParameterRange::float(0.5, 1.0));
```

### Samplers

**RandomSampler**: Simple random search

Surprisingly effective. Often outperforms grid search.

**GridSampler**: Exhaustive grid

Only for small search spaces.

**TPE (Tree-Parzen Estimator)**: Bayesian optimization

Models P(hyperparameters | good performance) vs P(hyperparameters | bad performance).

**BayesianOptimizer**: Gaussian Process-based

Uses GP surrogate model with acquisition functions (EI, UCB, PI).

### Schedulers (Early Stopping)

**MedianPruner**: Stop if below median

If trial is below median of completed trials at same step, prune it.

**Hyperband**: Multi-fidelity optimization

Trains many configs with small budget, keeps promising ones for larger budgets.

**ASHA**: Asynchronous Successive Halving

Hyperband variant for parallel/distributed settings.

### The Study API

```rust
use ferroml_core::hpo::{Study, TPESampler, MedianPruner};

let mut study = Study::new()
    .sampler(TPESampler::new())
    .pruner(MedianPruner::new())
    .direction(Direction::Maximize);

for _ in 0..100 {
    let trial = study.ask(&search_space);

    let score = train_and_evaluate(&trial.params);

    study.tell(trial.id, score);
}

println!("Best: {:?}", study.best_params());
println!("Best score: {}", study.best_value());
```

---

## Pipelines

### Why Pipelines?

Without pipelines, preprocessing is error-prone:

```python
# WRONG: Data leakage!
scaler.fit(X_all)  # Learned from test data too
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train_scaled, y_train)
```

With pipelines:

```python
# CORRECT: No leakage
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])
pipeline.fit(X_train, y_train)  # Scaler learns only from train
predictions = pipeline.predict(X_test)  # Scaler transforms test correctly
```

### Pipeline Components

**Pipeline**: Sequential steps

```python
from ferroml.pipeline import Pipeline
from ferroml.preprocessing import StandardScaler, PolynomialFeatures
from ferroml.linear import RidgeRegression

pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("scale", StandardScaler()),
    ("model", RidgeRegression(alpha=1.0)),
])
```

**FeatureUnion**: Parallel feature extraction

Concatenates outputs of multiple transformers:

```python
from ferroml.pipeline import FeatureUnion

features = FeatureUnion([
    ("numeric", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2)),
])
```

**ColumnTransformer**: Different transforms per column

```python
from ferroml.pipeline import ColumnTransformer

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), [0, 1, 2]),  # Scale columns 0-2
    ("categorical", OneHotEncoder(), [3, 4]),  # Encode columns 3-4
])
```

### Nested Parameters

Access and modify parameters with `step__param` syntax:

```python
pipeline.set_params(model__alpha=2.0)
print(pipeline.get_params()["model__alpha"])
```

This integrates with HPO:

```python
search_space = SearchSpace()
    .add("poly__degree", ParameterRange::int(1, 3))
    .add("model__alpha", ParameterRange::float(0.01, 100.0).log_scale())
```

---

## Ensembles

### Variance-Bias Tradeoff

- **High bias**: Model too simple, underfits
- **High variance**: Model too complex, overfits

Ensembles reduce variance by averaging multiple models.

### Ensemble Types

**VotingClassifier/Regressor**: Simple averaging

- Hard voting: Majority class
- Soft voting: Average probabilities
- Weighted: Give better models more influence

**Bagging**: Bootstrap aggregating

Train multiple models on bootstrap samples. RandomForest is bagging with decision trees.

**Stacking**: Meta-learning

Use predictions of base models as features for a meta-learner. CV-based to prevent leakage:

```python
from ferroml.ensemble import StackingClassifier
from ferroml.linear import LogisticRegression, RidgeRegression
from ferroml.trees import RandomForestClassifier

stacker = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=50)),
        ("lr", LogisticRegression()),
    ],
    final_estimator=RidgeRegression(),
    cv=5,
)
```

The meta-learner sees out-of-fold predictions, so no leakage.

---

## Explainability

### Why Explainability?

Model predictions need explanations for:
- **Debugging**: Why is this prediction wrong?
- **Trust**: Can I deploy this model?
- **Compliance**: GDPR requires explanations for automated decisions
- **Science**: What did the model learn?

### Feature Importance

**Permutation Importance**: Model-agnostic

Shuffle one feature, measure performance drop. Works for any model.

```python
from ferroml.explainability import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10)
for i in result.sorted_indices()[:5]:
    print(f"{feature_names[i]}: {result.importances_mean[i]:.3f} "
          f"(95% CI: [{result.importances_ci_lower[i]:.3f}, "
          f"{result.importances_ci_upper[i]:.3f}])")
```

### Partial Dependence Plots (PDP)

Shows marginal effect of a feature on predictions, averaging over all other features.

```python
from ferroml.explainability import partial_dependence

pdp = partial_dependence(model, X, feature=0, grid_size=50)
# pdp.grid_values: Feature values
# pdp.pdp_values: Average predictions at each value
# pdp.std_values: Heterogeneity (variation across samples)
```

### Individual Conditional Expectation (ICE)

Like PDP, but shows each sample individually. Reveals heterogeneous effects.

- **Centered ICE (c-ICE)**: Centered at reference point for comparison
- **Derivative ICE (d-ICE)**: Derivative of ICE curves, highlights non-linearity

### SHAP Values

Shapley values from game theory: fair attribution of prediction to features.

**TreeSHAP**: Exact, polynomial-time for trees

```python
from ferroml.explainability import TreeExplainer

explainer = TreeExplainer(model)
shap_values = explainer.explain(X_test)

# Per-sample: shap_values[i] shows contribution of each feature
# Global: mean(|shap_values|) per feature
```

**KernelSHAP**: Approximate, works for any model

Uses weighted linear regression approximation.

### H-Statistic (Interaction Detection)

Friedman's H-statistic measures feature interaction strength.

$$H^2_{jk} = \frac{\sum_i [f_{jk}(x_j^i, x_k^i) - f_j(x_j^i) - f_k(x_k^i)]^2}{\sum_i f_{jk}(x_j^i, x_k^i)^2}$$

FerroML provides:
- Pairwise H² for all feature pairs
- Bootstrap CI for significance
- Permutation p-value

---

## AutoML

### The AutoML Philosophy

FerroML's AutoML is not a black box. It provides:
1. **Transparent search**: See what was tried
2. **Statistical comparisons**: Know if the "best" model is significantly better
3. **Calibrated probabilities**: Automatic probability calibration
4. **Ensemble construction**: Combine complementary models
5. **Explainability**: Feature importance and model comparisons

### Configuration

```python
from ferroml.automl import AutoML, AutoMLConfig

config = AutoMLConfig(
    task="classification",
    metric="roc_auc",
    time_budget_seconds=3600,
    cv_folds=5,
    statistical_tests=True,
    confidence_level=0.95,
)

automl = AutoML(config)
result = automl.fit(X_train, y_train)
```

### Results

```python
# Leaderboard with CIs
print(result.leaderboard())
#    rank  model                 score   score_ci_lower  score_ci_upper
# 0  1     HistGradientBoosting  0.891   0.878           0.904
# 1  2     RandomForest          0.885   0.871           0.899
# 2  3     LogisticRegression    0.872   0.858           0.886

# Statistical comparisons
print(result.model_comparisons())
# HistGradientBoosting vs RandomForest:
#   Difference: 0.006 (95% CI: [-0.002, 0.014])
#   p-value: 0.134 (corrected: 0.402)
#   Effect size: 0.12 (negligible)
#   Conclusion: Not significantly different

# Feature importance (aggregated across models)
print(result.feature_importance())

# Ensemble composition
print(result.ensemble())
# [HistGradientBoosting: 0.45, RandomForest: 0.35, LogisticRegression: 0.20]
```

### Algorithm Portfolio

FerroML selects algorithms based on dataset characteristics:

**Quick preset** (4 algorithms): Fast turnaround
**Balanced preset** (8 algorithms): Good coverage
**Thorough preset** (10+ algorithms): Comprehensive search

Algorithms are selected based on:
- Dataset size (some algorithms don't scale)
- Feature types (categorical, continuous)
- Class imbalance
- Missing values
- Dimensionality

### Time Budget Allocation

FerroML uses multi-armed bandit algorithms (UCB1, Thompson Sampling) to allocate time:
- Promising algorithms get more time
- Poor performers are cut early
- Exploration-exploitation balance

### Meta-Learning

FerroML can learn from previous runs:
1. Extract dataset metafeatures (statistical, information-theoretic, landmarking)
2. Find similar datasets from history
3. Warm-start search with configurations that worked before

---

## Deployment

### Serialization

Save models in multiple formats:

```rust
use ferroml_core::{save_model, load_model, Format};

// JSON (human-readable)
save_model(&model, "model.json", Format::Json)?;

// MessagePack (compact, fast)
save_model(&model, "model.msgpack", Format::MessagePack)?;

// Bincode (fastest, smallest)
save_model(&model, "model.bin", Format::Bincode)?;

// Load with auto-detection
let model = load_model_auto::<LinearRegression>("model.json")?;
```

### ONNX Export

Export to ONNX for deployment in any environment:

```rust
use ferroml_core::onnx::OnnxExportable;

let onnx_bytes = model.export_onnx()?;
std::fs::write("model.onnx", onnx_bytes)?;
```

Supports:
- Linear models (LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet)
- Tree models (DecisionTree, RandomForest)

### Pure-Rust Inference

Run ONNX models without Python:

```rust
use ferroml_core::inference::InferenceSession;

let session = InferenceSession::load_from_file("model.onnx")?;
let input = Tensor::from_array(&[1, 5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
let output = session.run(&input)?;
```

### Schema Validation

Validate inputs match training schema:

```rust
use ferroml_core::schema::{FeatureSchema, ValidationMode};

let schema = FeatureSchema::from_array(&X_train);

// In production
let validation = schema.validate(&X_new)?;
if !validation.is_valid() {
    println!("Issues: {:?}", validation.issues());
}
```

Catches:
- Shape mismatches
- Missing values (if not allowed)
- Out-of-range values
- Unknown categories

---

## Best Practices

### 1. Always Use Cross-Validation

A single train/test split can mislead. Use at least 5-fold CV:

```python
result = cross_val_score(model, X, y, cv=5)
print(f"Score: {result.mean:.3f} ± {result.std:.3f}")
```

### 2. Report Confidence Intervals

Never report just "accuracy = 0.85." Always include uncertainty:

```python
print(f"Accuracy: {result.mean:.3f} (95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}])")
```

### 3. Check Assumptions

For linear models, always check diagnostics:

```python
model.fit(X, y)
summary = model.summary()
if not summary.assumption_tests_passed():
    print("Warning: Assumptions violated. Consider robust regression.")
```

### 4. Use Nested CV for Model Selection

When tuning hyperparameters and estimating performance:

```python
from ferroml.cv import NestedCV

nested = NestedCV(outer_cv=5, inner_cv=3)
result = nested.run(model, X, y, search_space)
```

### 5. Apply Multiple Testing Correction

When comparing many models:

```python
config = AutoMLConfig(
    multiple_testing_correction="benjamini_hochberg",
    ...
)
```

### 6. Consider Effect Sizes

A significant p-value means little if the effect is tiny:

```python
comparison = result.model_comparisons()
if comparison.effect_size < 0.2:
    print("Difference is negligible despite statistical significance")
```

### 7. Calibrate Probabilities

For probabilistic predictions, calibrate:

```python
from ferroml.models import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
calibrated.fit(X_train, y_train)
probs = calibrated.predict_proba(X_test)
```

### 8. Document Reproducibility

Always set random seeds:

```python
config = AutoMLConfig(
    seed=42,
    ...
)
```

### 9. Validate Before Deployment

Use schema validation:

```python
schema = model.schema()
validation = schema.validate(X_new)
if not validation.is_valid():
    handle_validation_errors(validation)
```

### 10. Monitor in Production

Track prediction distributions. If they drift from training, retrain.

---

## Glossary

**Bonferroni correction**: Divide significance level by number of tests.

**Bootstrap**: Resampling with replacement to estimate sampling distributions.

**Cohen's d**: Standardized difference between means.

**Confidence interval**: Range likely to contain true parameter value.

**Cross-validation**: Repeated train/test splits for robust evaluation.

**Effect size**: Standardized measure of the magnitude of an effect.

**False discovery rate (FDR)**: Expected proportion of false positives among discoveries.

**Heteroscedasticity**: Non-constant variance of residuals.

**IRLS**: Iteratively Reweighted Least Squares, used for logistic and robust regression.

**Multiple testing correction**: Adjusting p-values when performing many tests.

**Out-of-bag (OOB)**: Samples not included in a bootstrap sample.

**P-value**: Probability of seeing data as extreme as observed, assuming null hypothesis is true.

**Power**: Probability of detecting an effect when it exists.

**Regularization**: Adding penalty to loss function to prevent overfitting.

**SHAP values**: Shapley additive explanations, game-theoretic feature attribution.

**Type I error**: False positive (rejecting true null hypothesis).

**Type II error**: False negative (failing to reject false null hypothesis).

**VIF**: Variance Inflation Factor, measures multicollinearity.

---

## Further Reading

### Statistics
- *Statistical Inference* by Casella & Berger
- *The Elements of Statistical Learning* by Hastie, Tibshirani, Friedman
- *Computer Age Statistical Inference* by Efron & Hastie

### Machine Learning
- *Pattern Recognition and Machine Learning* by Bishop
- *Machine Learning: A Probabilistic Perspective* by Murphy

### AutoML
- *AutoML: A Survey of the State-of-the-Art* (He et al., 2021)
- *Auto-sklearn: Efficient and Robust Automated Machine Learning* (Feurer et al., 2019)

---

*FerroML: Because "it works" isn't good enough. You deserve to know why.*

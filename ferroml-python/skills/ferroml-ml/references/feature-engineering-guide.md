# Feature Engineering Guide

When and how to engineer features using FerroML preprocessing tools.

## Numeric Scaling

### Which scaler to use

| Scaler | When to use | When NOT to use |
|--------|------------|-----------------|
| `StandardScaler` | Default choice. Gaussian-ish data. | Sparse data (destroys sparsity). |
| `MinMaxScaler` | Need bounded range [0,1]. Neural networks. | Outliers present (they compress the range). |
| `RobustScaler` | Outliers in data. Median-based. | Data is already clean and Gaussian. |
| `MaxAbsScaler` | Sparse data. Scales by max absolute value. | Dense data with outliers. |
| `Normalizer` | Row-wise unit norm (L2). Text/TF-IDF features. | Column-wise scaling needed. |
| `PowerTransformer` | Skewed features. Makes data more Gaussian. | Already normal data. |
| `QuantileTransformer` | Heavily skewed or non-standard distributions. | Small datasets (needs enough samples for quantile estimation). |

### Usage patterns

```python
from ferroml.preprocessing import StandardScaler, PowerTransformer

# Standard workflow -- fit on train, transform both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Skewed features -- make them Gaussian first
pt = PowerTransformer(method="yeo-johnson")
X_train_normal = pt.fit_transform(X_train)
```

### Models that NEED scaling

| Must scale | Does not need scaling |
|------------|----------------------|
| KNN, SVC/SVR, MLP | Decision trees, Random Forest |
| LogisticRegression (with regularization) | Gradient Boosting, HistGBT |
| PCA, t-SNE, LDA | Naive Bayes |
| SGDClassifier/Regressor | AdaBoost |

## Categorical Encoding

### Decision tree

```
How many unique categories?
├── 2-10 (low cardinality)
│   ├── No ordering → OneHotEncoder
│   └── Has natural order (size: S<M<L) → OrdinalEncoder
├── 10-100 (medium cardinality)
│   └── TargetEncoder (reduces dimensionality)
├── 100+ (high cardinality)
│   └── TargetEncoder (OneHot would explode dimensions)
└── Target variable itself
    └── LabelEncoder (converts class names to integers)
```

### Usage patterns

```python
from ferroml.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, LabelEncoder

# Low cardinality -- one-hot
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(X_categorical)

# Ordinal -- preserve ordering
oe = OrdinalEncoder()
X_ordinal = oe.fit_transform(X_ordinal_features)

# High cardinality -- target-based
te = TargetEncoder(smooth=1.0)
te.fit(X_categorical, y)
X_encoded = te.transform(X_categorical)

# Target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_strings)
```

**Warning:** Never use OneHotEncoder on high-cardinality features (100+ categories). It creates 100+ columns and makes most models worse.

**Warning:** TargetEncoder must be fit on training data only to avoid leakage. Always use within a Pipeline.

## Text Features

### CountVectorizer vs TfidfVectorizer

| Vectorizer | What it does | When to use |
|-----------|-------------|-------------|
| `CountVectorizer` | Raw term counts | Short documents, Naive Bayes, when word frequency matters |
| `TfidfVectorizer` | TF-IDF weighted | Longer documents, when rare words are important, most tasks |
| `TfidfTransformer` | Converts count matrix to TF-IDF | When you already have a count matrix |

### Usage patterns

```python
from ferroml.preprocessing import TfidfVectorizer, CountVectorizer
from ferroml.pipeline import TextPipeline

# Simple text classification
vec = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95)
X_text = vec.fit_transform(documents)

# With a pipeline
from ferroml.naive_bayes import MultinomialNB
pipe = TextPipeline(steps=[
    ("tfidf", TfidfVectorizer(max_features=10000)),
    ("clf", MultinomialNB())
])
pipe.fit(train_texts, y_train)
predictions = pipe.predict(test_texts)
```

### Key parameters

| Parameter | What it controls | Guidance |
|-----------|-----------------|----------|
| `max_features` | Maximum vocabulary size | 5000-20000 for most tasks. Lower = faster, less noise. |
| `min_df` | Minimum document frequency | 2-5 removes typos and ultra-rare words |
| `max_df` | Maximum document frequency (fraction) | 0.9-0.95 removes words that appear everywhere |

## Polynomial Features and Interactions

```python
from ferroml.preprocessing import PolynomialFeatures

# Add quadratic terms and interactions
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)
```

### When to use

- Linear model is underfitting and you suspect non-linear relationships
- You want interaction terms (feature_a * feature_b) without switching to tree models
- Domain knowledge suggests polynomial relationships (e.g., area = length * width)

### When NOT to use

- Already using tree-based models (they capture interactions naturally)
- High-dimensional data (polynomial expansion makes it exponentially worse)
- `degree > 3` (rarely useful, creates massive feature space)

**Tip:** Use `interaction_only=True` to get only interaction terms without squared terms. Reduces feature explosion.

## Binning

```python
from ferroml.preprocessing import KBinsDiscretizer

# Quantile-based binning (equal-frequency)
kbd = KBinsDiscretizer(n_bins=5, strategy="quantile", encode="ordinal")
X_binned = kbd.fit_transform(X)
```

### Strategy options

| Strategy | Bins contain | Use when |
|----------|-------------|----------|
| `"quantile"` | Equal number of samples | Default. Handles skewed features well. |
| `"uniform"` | Equal width ranges | Data is uniformly distributed. |
| `"kmeans"` | KMeans-derived boundaries | Want data-driven bin edges. |

### When to use binning

- Converting continuous features to ordinal for models that prefer discrete inputs
- Capturing non-linear relationships in linear models (alternative to polynomial features)
- Reducing noise in noisy continuous features

## Feature Selection

### Recommended progression

Start simple, add complexity only if needed:

```
Step 1: VarianceThreshold (remove constants)
    ↓
Step 2: SelectKBest (univariate filter)
    ↓
Step 3: RecursiveFeatureElimination (model-based wrapper)
    ↓
Step 4: SelectFromModel (importance-based)
```

### VarianceThreshold

Removes features with zero or near-zero variance. Always run this first.

```python
from ferroml.preprocessing import VarianceThreshold

# Remove constant features
vt = VarianceThreshold(threshold=0.0)
X_filtered = vt.fit_transform(X)

# Remove near-constant features (variance < 0.01)
vt = VarianceThreshold(threshold=0.01)
X_filtered = vt.fit_transform(X)
```

### SelectKBest

Picks top k features by univariate statistical test.

```python
from ferroml.preprocessing import SelectKBest

selector = SelectKBest(k=10)
selector.fit(X, y)
X_selected = selector.transform(X)
scores = selector.scores()  # see which features scored highest
```

**Limitation:** Only captures linear/monotonic relationships. Misses interaction effects.

### RecursiveFeatureElimination

Iteratively removes least important features based on model coefficients/importances.

```python
from ferroml.preprocessing import RecursiveFeatureElimination
from ferroml.linear import LinearRegression

rfe = RecursiveFeatureElimination(
    estimator=LinearRegression(),
    n_features_to_select=5
)
rfe.fit(X, y)
X_selected = rfe.transform(X)
ranking = rfe.ranking()  # 1 = selected, higher = eliminated earlier
```

**Best with:** Linear models, Random Forest (anything with feature importances or coefficients).

### SelectFromModel

Selects features based on importance weights from a fitted model.

```python
from ferroml.preprocessing import SelectFromModel
from ferroml.trees import RandomForestClassifier

sfm = SelectFromModel(
    estimator=RandomForestClassifier(n_estimators=100),
    threshold=None  # uses mean importance as threshold
)
sfm.fit(X, y)
X_selected = sfm.transform(X)
```

## Imputation

### Decision guide

| Missing % | Strategy |
|-----------|----------|
| < 5% | `SimpleImputer(strategy="mean")` or `"median"` |
| 5-30% | `KNNImputer(n_neighbors=5)` for better accuracy |
| 30-50% | KNNImputer + add a binary "was_missing" indicator column |
| > 50% | Drop the column entirely |

### SimpleImputer strategies

| Strategy | When to use |
|----------|------------|
| `"mean"` | Numeric, roughly symmetric distribution |
| `"median"` | Numeric, skewed distribution or outliers |
| `"most_frequent"` | Categorical features |

```python
from ferroml.preprocessing import SimpleImputer, KNNImputer

# Simple median imputation
imp = SimpleImputer(strategy="median")
X_clean = imp.fit_transform(X_train)
X_test_clean = imp.transform(X_test)

# KNN-based (uses similar rows to impute)
imp = KNNImputer(n_neighbors=5)
X_clean = imp.fit_transform(X_train)
```

**Critical rule:** Always fit the imputer on training data only. Use `Pipeline` to enforce this.

## Handling Class Imbalance

### Resampling methods

| Method | What it does | When to use |
|--------|-------------|-------------|
| `SMOTE` | Creates synthetic minority samples | Default for oversampling. Works well with KNN, SVM. |
| `ADASYN` | Like SMOTE but focuses on hard-to-learn samples | When boundary samples matter most. |
| `RandomOverSampler` | Duplicates minority samples | Simple baseline. Risk of overfitting. |
| `RandomUnderSampler` | Removes majority samples | Large dataset where you can afford to lose data. |

```python
from ferroml.preprocessing import SMOTE

smote = SMOTE(k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**Warning:** Only apply resampling to training data, never to test data. Do it after train/test split.

**Warning:** SMOTE does not work well with high-dimensional sparse data (text). Use class weights instead.

## Complete Pipeline Example

Putting it all together:

```python
from ferroml.pipeline import Pipeline, ColumnTransformer
from ferroml.preprocessing import (
    StandardScaler, OneHotEncoder, SimpleImputer, SelectKBest
)
from ferroml.trees import HistGradientBoostingClassifier

# Define preprocessing per column type
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_columns),
    ("cat", OneHotEncoder(), categorical_columns),
])

# Full pipeline: preprocess → select → model
pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("select", SelectKBest(k=20)),
    ("model", HistGradientBoostingClassifier(max_iter=200))
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

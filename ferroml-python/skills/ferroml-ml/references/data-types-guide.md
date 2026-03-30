# Data Types Guide

How to handle every kind of data in FerroML, with the right preprocessing for each.

## Quick Reference: Data Type to Preprocessor

| Data Type | Preprocessor | When |
|-----------|-------------|------|
| Numeric continuous | `StandardScaler` | Default for most models |
| Numeric continuous | `MinMaxScaler` | Neural nets, bounded data |
| Numeric continuous | `RobustScaler` | Outliers present |
| Numeric continuous | `PowerTransformer` | Skewed distributions |
| Numeric discrete (counts) | `StandardScaler` or none | Low counts, treat as numeric |
| Categorical nominal (low card.) | `OneHotEncoder` | <20 unique values |
| Categorical nominal (high card.) | `TargetEncoder` | 20-100 unique values |
| Categorical ordinal | `OrdinalEncoder` | Natural order exists |
| Binary | None | Already 0/1 |
| Text | `TfidfVectorizer` | Default for text |
| Text | `CountVectorizer` | When TF-IDF not needed |
| Missing values | `SimpleImputer` / `KNNImputer` | Before other transforms |

## Numeric Continuous

Most ML features. Sensor readings, prices, measurements, ratios.

### When to Scale

| Scaler | Use When | Example |
|--------|----------|---------|
| `StandardScaler` | Features have different units/scales. Required by SVM, KNN, PCA, linear models with regularization. | Income ($30k-$200k) + Age (18-80) |
| `MinMaxScaler` | Need bounded [0,1] range. Neural networks, distance-based models. | Image pixel values, probabilities |
| `RobustScaler` | Outliers present. Uses median/IQR instead of mean/std. | Income with billionaires in data |
| `MaxAbsScaler` | Sparse data, need to preserve zeros. | Count data, sparse features |
| `Normalizer` | Normalize each sample (row), not feature (column). | Text TF-IDF vectors, unit vectors |

### When NOT to Scale

- Tree-based models (DecisionTree, RandomForest, GBT, HistGBT) -- scale-invariant
- Naive Bayes -- works on raw values
- When features are already on the same scale

### Handling Skew

```python
from ferroml.preprocessing import PowerTransformer, QuantileTransformer

# Box-Cox (positive data only) or Yeo-Johnson (any data)
pt = PowerTransformer(method="yeo-johnson")
X_transformed = pt.fit_transform(X)

# Force Gaussian-like distribution (non-parametric)
qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal")
X_transformed = qt.fit_transform(X)
```

**When to use:**
- Revenue, prices, population -- typically right-skewed
- Sensor data with long tails
- Before linear models that assume normality

## Numeric Discrete

Integers, counts, ratings.

**Decision: Treat as numeric or categorical?**

| Situation | Treatment |
|-----------|-----------|
| Count with many values (0-1000) | Numeric, possibly log-transform |
| Rating (1-5) | Numeric if ordinal, one-hot if nominal |
| Few unique values (<10) | Consider categorical encoding |
| Year (2020-2026) | Numeric (trend) or categorical (each year is different) |

## Categorical Nominal

Unordered categories: color, city, product_type.

### Low Cardinality (<20 unique values)

```python
from ferroml.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_categorical)
# Creates binary column for each category
```

**Watch out for:**
- Curse of dimensionality: 50 categories = 50 new columns
- Unseen categories at prediction time
- Drop one category to avoid multicollinearity (for linear models)

### High Cardinality (20-100 unique values)

```python
from ferroml.preprocessing import TargetEncoder

encoder = TargetEncoder()
encoder.fit(X_categorical, y)  # needs target for encoding
X_encoded = encoder.transform(X_categorical)
# Each category becomes its mean target value
```

**Advantages:**
- Single column output regardless of cardinality
- Captures relationship between category and target
- Works well with tree models

**Risk:** Target leakage. Use with cross-validation to avoid overfitting.

### Very High Cardinality (>100 unique values)

Options ranked by preference:
1. **TargetEncoder** -- usually sufficient
2. **Frequency encoding** -- replace category with its count (manual)
3. **Hash encoding** -- fixed-size output (manual)
4. **Drop the feature** -- if cardinality is close to n_samples, it's basically an ID

## Categorical Ordinal

Ordered categories: education level, rating, size (S/M/L/XL).

```python
from ferroml.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
# Categories are mapped to integers preserving order:
# "small"->0, "medium"->1, "large"->2
X_encoded = encoder.fit_transform(X_ordinal)
```

**Key:** The integer mapping must respect the natural order. Random ordering makes ordinal encoding meaningless.

## Binary Features

Already 0/1 (gender, has_account, is_active). No encoding needed.

**Tip:** Ensure they are actually 0/1 floats, not strings "yes"/"no". Convert manually:
```python
X[:, col] = (X[:, col] == "yes").astype(float)
```

## Text Data

Free-form text: reviews, descriptions, emails.

### Bag of Words

```python
from ferroml.text import CountVectorizer

cv = CountVectorizer(max_features=5000)
X_text = cv.fit_transform(documents)
# Sparse matrix of word counts
```

### TF-IDF (Weighted)

```python
from ferroml.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(documents)
# Words weighted by importance (rare words score higher)
```

### Text Pipeline

```python
from ferroml.pipeline import TextPipeline

pipeline = TextPipeline()
# Combines tokenization + vectorization + model
```

### When to Use Which

| Method | Best For |
|--------|----------|
| `CountVectorizer` | Simple classification, Naive Bayes |
| `TfidfVectorizer` | Most text tasks (default choice) |
| `TextPipeline` | End-to-end text classification |

## Datetime Features

FerroML operates on numeric arrays. Extract features from datetimes manually.

```python
import polars as pl

df = pl.read_csv("data.csv", try_parse_dates=True)

df = df.with_columns([
    df["date"].dt.year().alias("year"),
    df["date"].dt.month().alias("month"),
    df["date"].dt.weekday().alias("day_of_week"),  # 0=Monday
    df["date"].dt.hour().alias("hour"),
    (df["date"].dt.weekday() >= 5).cast(pl.Int32).alias("is_weekend"),
    # Days since reference date
    (df["date"] - pl.lit(datetime(2020, 1, 1))).dt.total_days().alias("days_since_2020"),
])
```

**Common datetime features:**

| Feature | Captures |
|---------|----------|
| Year | Long-term trend |
| Month | Seasonality |
| Day of week | Weekly patterns |
| Hour | Intraday patterns |
| Is weekend | Weekend effect |
| Days since event | Recency/age |
| Quarter | Business cycles |

## Geographic Data

Latitude/longitude pairs.

**Feature engineering options:**
1. **Raw lat/lon** -- simple, works for tree models
2. **Distance to points of interest** -- distance to city center, nearest store
3. **Cluster assignment** -- use KMeans on coordinates, then one-hot encode cluster
4. **Geohash** -- grid cell encoding (treat as categorical)

```python
import numpy as np

# Distance from a reference point (Haversine approximation for small distances)
ref_lat, ref_lon = 40.7128, -74.0060  # NYC
X_distance = np.sqrt((X[:, lat_col] - ref_lat)**2 + (X[:, lon_col] - ref_lon)**2)
```

## Currency / Monetary Data

Typically right-skewed with long tails.

**Standard treatment:**
1. Log-transform: `np.log1p(amount)` -- handles skew, makes linear models work better
2. Inflation adjustment: normalize to constant dollars if time span > 1 year
3. Binning: `KBinsDiscretizer` for equal-frequency bins

```python
from ferroml.preprocessing import KBinsDiscretizer

# Equal-frequency bins
kbd = KBinsDiscretizer(n_bins=10, strategy="quantile")
X_binned = kbd.fit_transform(X_monetary)
```

## Mixed-Type Datasets: ColumnTransformer

The standard pattern for real-world data with multiple types.

```python
from ferroml.pipeline import Pipeline, ColumnTransformer
from ferroml.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer

# Define column groups by index
numeric_cols = [0, 1, 2, 3]       # age, income, score, balance
categorical_cols = [4, 5]          # state, product_type

# Build preprocessing for each type
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_cols),
    ("cat", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder()),
    ]), categorical_cols),
])

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100)),
])
pipeline.fit(X_train, y_train)
```

## Missing Data Strategy

| Pattern | Strategy | Preprocessor |
|---------|----------|-------------|
| <5% missing, random | Mean/median imputation | `SimpleImputer(strategy="mean")` |
| <5% missing, categorical | Most frequent | `SimpleImputer(strategy="most_frequent")` |
| 5-30% missing, patterned | KNN imputation | `KNNImputer(n_neighbors=5)` |
| >30% missing | Drop feature or create indicator | Manual |
| Missing = informative | Create binary "is_missing" feature + impute | Manual + `SimpleImputer` |

## Feature Engineering Quick Reference

| Technique | Preprocessor | When |
|-----------|-------------|------|
| Polynomial features | `PolynomialFeatures(degree=2)` | Capture interactions |
| Variance filter | `VarianceThreshold(threshold=0.01)` | Remove near-constant features |
| Univariate selection | `SelectKBest(k=20)` | Reduce dimensionality |
| Model-based selection | `SelectFromModel(estimator)` | Use model to pick features |
| Recursive elimination | `RecursiveFeatureElimination(estimator, n_features=10)` | Optimal subset |
| Class balancing | `SMOTE()` / `ADASYN()` | Imbalanced classification |
| Undersampling | `RandomUnderSampler()` | Majority class too large |
| Oversampling | `RandomOverSampler()` | Minority class too small |

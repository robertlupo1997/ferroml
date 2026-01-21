# Specification: Python Bindings

**Status**: ⚪ MINIMAL STUB

## Overview

Python API via PyO3 - the primary user interface for FerroML.

## Requirements

### Core API

```python
import ferroml as fml

# AutoML
automl = fml.AutoML(
    task="classification",
    metric="roc_auc",
    statistical_tests=True,
    confidence_level=0.95,
    time_budget_seconds=3600,
)
result = automl.fit(X, y, cv=5)
predictions = result.predict(X_test)

# Individual models
model = fml.LinearRegression()
model.fit(X, y)
print(model.summary())  # R-style output

# Preprocessing
scaler = fml.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Pipeline
pipe = fml.Pipeline([
    ("scaler", fml.StandardScaler()),
    ("model", fml.LogisticRegression()),
])
pipe.fit(X, y)
```

### Data Interchange
- [ ] NumPy arrays (zero-copy with numpy crate)
- [ ] Polars DataFrames (zero-copy with pyo3-polars)
- [ ] Pandas DataFrames (via Arrow)

### PyO3 Structure

```rust
#[pyclass]
struct AutoML {
    inner: ferroml_core::AutoML,
}

#[pymethods]
impl AutoML {
    #[new]
    fn new(/* config */) -> Self { ... }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>, cv: usize)
        -> PyResult<AutoMLResult> { ... }

    fn predict(&self, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> { ... }
}
```

### Module Structure

```python
ferroml/
├── __init__.py
├── automl.py      # AutoML class
├── models/        # Linear, tree, ensemble
├── preprocessing/ # Transformers
├── cv/            # Cross-validation
├── stats/         # Statistical tests
└── pipeline.py    # Pipeline, FeatureUnion
```

## Implementation Priority

1. Basic module with version
2. AutoML class
3. LinearRegression (test data interchange)
4. StandardScaler
5. Pipeline

## Acceptance Criteria

- [ ] `pip install ferroml` works (maturin build)
- [ ] Zero-copy data transfer with NumPy
- [ ] Comparable or better performance than sklearn
- [ ] Pythonic API (sklearn-compatible where sensible)

## Implementation Location

`ferroml-python/src/lib.rs`

## Build Commands

```bash
# Development build
cd ferroml-python
maturin develop

# Release build
maturin build --release

# Publish to PyPI
maturin publish
```

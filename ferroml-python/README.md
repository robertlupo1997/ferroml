# FerroML

Statistically rigorous AutoML in Rust with Python bindings.

## Installation

```bash
pip install ferroml
```

## Quick Start

```python
from ferroml.linear import LinearRegression
from ferroml.automl import AutoML, AutoMLConfig
import numpy as np

# Linear Regression with full statistical diagnostics
X = np.random.randn(100, 5)
y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100) * 0.1

model = LinearRegression()
model.fit(X, y)
print(model.summary())  # R-style statistical output

# AutoML with statistical significance testing
config = AutoMLConfig(
    task="classification",
    metric="roc_auc",
    time_budget_seconds=300,
)
automl = AutoML(config)
result = automl.fit(X_train, y_train)
print(result.summary())
```

## Features

- **Statistical Rigor**: Confidence intervals, effect sizes, and assumption tests
- **Linear Models**: Full diagnostics (R-style summary, VIF, residual analysis)
- **Tree Models**: RandomForest, GradientBoosting, HistGradientBoosting
- **Preprocessing**: Scalers, encoders, imputers
- **AutoML**: Automatic model selection with statistical significance testing

## License

MIT OR Apache-2.0

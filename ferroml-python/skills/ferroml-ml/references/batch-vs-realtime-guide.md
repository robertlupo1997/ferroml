# Batch vs Real-Time Inference Guide

Choosing and implementing the right inference pattern for production with FerroML.

## When to Use Which

| Factor | Batch | Real-Time |
|--------|-------|-----------|
| Latency requirement | Minutes to hours OK | <100ms required |
| Data volume | Thousands to millions | One at a time |
| Frequency | Scheduled (hourly/daily) | On-demand per request |
| Example | Nightly churn scoring | Credit approval API |
| Failure impact | Retry entire batch | Must handle per-request |
| Resource pattern | Burst compute, then idle | Steady, always-on |

## Batch Inference Pattern

Score large datasets on a schedule. No latency pressure.

```python
from ferroml.serialization import load_json
import numpy as np

# 1. Load model (once)
model = load_json("models/churn_model_v2.json")

# 2. Load data
X_batch = load_feature_matrix("data/customers_2026-03-29.npy")

# 3. Predict all at once
predictions = model.predict(X_batch)

# 4. For probabilistic models, get confidence
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(X_batch)

# 5. Write results
np.save("results/churn_scores_2026-03-29.npy", predictions)
```

### Batch Best Practices

- **Checkpoint large batches**: Save partial results every N rows in case of failure
- **Validate input shape**: Compare `X_batch.shape[1]` against training feature count
- **Log metadata**: Record model version, data timestamp, row count, runtime
- **Idempotent runs**: Re-running the same batch should produce the same output

### Batch with Preprocessing

```python
from ferroml.serialization import load_json

# Load full pipeline (model + preprocessing)
pipeline = load_json("models/pipeline_v2.json")

# Pipeline handles scaling/encoding automatically
predictions = pipeline.predict(X_raw)
```

## Real-Time Inference Pattern

Per-request predictions with low latency.

```python
from ferroml.serialization import load_json
import numpy as np

# Load at application startup (not per request)
model = load_json("models/credit_model_v3.json")

def predict_single(features: dict) -> dict:
    """Score a single request."""
    # Convert input to numpy array
    X = np.array([[
        features["income"],
        features["credit_score"],
        features["debt_ratio"],
        features["years_employed"],
    ]], dtype=np.float64)

    prediction = model.predict(X)[0]

    result = {"prediction": float(prediction)}

    # Add probability if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        result["probability"] = float(proba)

    # Add uncertainty if available
    if hasattr(model, 'predict_with_uncertainty'):
        mean, lower, upper = model.predict_with_uncertainty(X, confidence=0.95)
        result["confidence_interval"] = {
            "lower": float(lower[0]),
            "upper": float(upper[0]),
        }

    return result
```

### Real-Time Best Practices

- **Load model once at startup** -- never load per request
- **Validate inputs** -- check for NaN, correct dtype, correct feature count
- **Set timeouts** -- if prediction takes >500ms, something is wrong
- **Return uncertainty** -- let consumers decide if confidence is sufficient
- **Version the endpoint** -- `/v2/predict` so you can roll back

## Latency by Model Type

Approximate predict latency for a single sample:

| Model | Predict Latency | Memory | Notes |
|-------|----------------|--------|-------|
| LinearRegression | <0.1ms | Tiny | Just dot product |
| LogisticRegression | <0.1ms | Tiny | Dot product + sigmoid |
| NaiveBayes (Gaussian/Multinomial) | <0.1ms | Tiny | Lookup + multiply |
| DecisionTree | <0.1ms | Small | Tree traversal |
| Ridge / Lasso | <0.1ms | Tiny | Dot product |
| RandomForest (100 trees) | 0.5-2ms | Medium | 100 tree traversals |
| GradientBoosting (100 trees) | 0.5-2ms | Medium | Sequential traversals |
| HistGradientBoosting | 0.5-2ms | Medium | Histogram lookups |
| MLP (small) | 0.1-1ms | Small | Matrix multiplies |
| SVC (RBF kernel) | 1-50ms | Large | Scales with support vectors |
| KNN | 1-100ms | Large | Scales with training data size |
| GaussianProcess | 10-500ms | Large | Scales with training data size |

### Speed Tiers for Real-Time

| Tier | Models | Suitable For |
|------|--------|-------------|
| Sub-millisecond | Linear, NaiveBayes, DecisionTree | High-throughput APIs (>10K rps) |
| Low millisecond | RandomForest, GBT, HistGBT, MLP | Standard APIs (1K-10K rps) |
| High millisecond | SVC, KNN, GP | Low-throughput, high-value decisions |

## Serialization Format Comparison

| Format | Speed | Size | Human-Readable | Use Case |
|--------|-------|------|---------------|----------|
| `save_json` / `load_json` | Slow | Large | Yes | Debugging, inspection |
| `save_msgpack` / `load_msgpack` | Fast | Small | No | Production (recommended) |
| `save_bincode` / `load_bincode` | Fastest | Smallest | No | Performance-critical |
| `to_bytes` / `from_bytes` | Fast | Small | No | In-memory transfer |

**Recommendation:** Use `msgpack` for production deployments. Use `json` during development.

```python
from ferroml.serialization import save_msgpack, load_msgpack

# Save for production
save_msgpack(model, "models/model_v3.msgpack")

# Load at startup
model = load_msgpack("models/model_v3.msgpack")
```

## Caching Strategies

### Input Caching

Cache predictions for identical inputs to avoid redundant computation.

```python
import hashlib

def cached_predict(model, X, cache: dict):
    """Cache predictions by input hash."""
    key = hashlib.md5(X.tobytes()).hexdigest()
    if key not in cache:
        cache[key] = model.predict(X)
    return cache[key]
```

### Model Warm-Up

For models with lazy initialization, make a dummy prediction at startup:

```python
import numpy as np

# Warm up (triggers any lazy computation)
dummy = np.zeros((1, n_features), dtype=np.float64)
_ = model.predict(dummy)
```

## Scaling Patterns

### Horizontal Scaling (Stateless Models)

All FerroML models are stateless after fitting -- predictions depend only on the model parameters and input, not on previous requests. This means you can:

1. **Run N copies in parallel** -- each process loads the same model file
2. **No shared state needed** -- no locks, no coordination
3. **Load balancer friendly** -- any instance can handle any request

```
                    +---> [Instance 1: model_v3.msgpack]
                    |
[Load Balancer] ----+---> [Instance 2: model_v3.msgpack]
                    |
                    +---> [Instance 3: model_v3.msgpack]
```

### Model Updates (Blue-Green)

1. Train new model, save as `model_v4.msgpack`
2. Start new instances with v4
3. Shift traffic from v3 to v4
4. Keep v3 instances for rollback

### Thread Safety

FerroML models support concurrent predictions from multiple threads:

```python
from concurrent.futures import ThreadPoolExecutor

# Safe: multiple threads calling predict on the same model
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(model.predict, X_chunk) for X_chunk in chunks]
    results = [f.result() for f in futures]
```

## Decision Checklist

Use this to pick your inference pattern:

```
1. How many predictions per day?
   <1,000     --> Batch is fine
   1K-100K    --> Either works, depends on latency needs
   >100K      --> Real-time if low latency needed, batch if not

2. How fast must results be available?
   <100ms     --> Real-time (sub-ms model or cache)
   <1 second  --> Real-time (any model)
   <1 minute  --> Near-real-time (batch micro-batches)
   Hours OK   --> Batch

3. How does data arrive?
   All at once (file/DB dump)   --> Batch
   One at a time (API request)  --> Real-time
   Stream (continuous)          --> Mini-batch or real-time

4. What's the cost of stale predictions?
   High (fraud detection)       --> Real-time
   Medium (recommendations)     --> Near-real-time
   Low (monthly report)         --> Batch
```

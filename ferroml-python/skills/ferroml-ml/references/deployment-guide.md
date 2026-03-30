# Deployment Guide

Production deployment patterns for FerroML models.

## Serialization Formats

FerroML models support three serialization formats:

| Format | Method | File size | Speed | Human-readable | Use case |
|--------|--------|-----------|-------|----------------|----------|
| JSON | `save_json()` / `load_json()` | Largest | Slowest | Yes | Debugging, config management, version control |
| MessagePack | `save_msgpack()` / `load_msgpack()` | Medium | Fast | No | General production use |
| Bincode | `save_bincode()` / `load_bincode()` | Smallest | Fastest | No | Performance-critical serving |

### Save and load

```python
from ferroml.trees import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save
model.save_json("model.json")
model.save_msgpack("model.msgpack")
model.save_bincode("model.bincode")

# Load
loaded = RandomForestClassifier.load_json("model.json")
loaded = RandomForestClassifier.load_msgpack("model.msgpack")
loaded = RandomForestClassifier.load_bincode("model.bincode")
```

### Format recommendations

| Scenario | Format |
|----------|--------|
| Dev/staging, need to inspect model | JSON |
| Production API, balanced | MessagePack |
| High-throughput serving, minimal latency | Bincode |
| Model stored in git/version control | JSON |
| Model stored in S3/GCS blob storage | MessagePack or Bincode |

**Note:** Pipeline and ensemble serialization (VotingClassifier, StackingClassifier, BaggingClassifier) is not yet supported. Serialize individual models and reconstruct pipelines in code.

## ONNX Export

Some models support ONNX export for cross-platform inference:

```python
# Check if model supports ONNX
onnx_bytes = model.to_onnx()

# Save to file
with open("model.onnx", "wb") as f:
    f.write(onnx_bytes)
```

ONNX models can be served by ONNX Runtime, TensorRT, or any ONNX-compatible runtime without Python or FerroML installed.

## FastAPI Serving

### Real-time prediction endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from ferroml.trees import HistGradientBoostingClassifier

app = FastAPI()

# Load model at startup
model = HistGradientBoostingClassifier.load_msgpack("model.msgpack")

class PredictRequest(BaseModel):
    features: list[list[float]]

class PredictResponse(BaseModel):
    predictions: list[float]
    probabilities: list[list[float]] | None = None

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        X = np.array(request.features)
        preds = model.predict(X).tolist()
        probs = model.predict_proba(X).tolist()
        return PredictResponse(predictions=preds, probabilities=probs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "model": "HistGradientBoostingClassifier"}
```

### Key patterns for production serving

**Model loading:** Load once at startup, not per-request. FerroML models are thread-safe for concurrent `predict()` calls.

**Input validation:** Always validate input shape and types before calling predict. FerroML raises `ValueError` with `.hint()` text on bad input.

```python
@app.post("/predict")
def predict(request: PredictRequest):
    X = np.array(request.features)
    if X.ndim != 2:
        raise HTTPException(400, "Features must be 2D array")
    if X.shape[1] != EXPECTED_FEATURES:
        raise HTTPException(400, f"Expected {EXPECTED_FEATURES} features, got {X.shape[1]}")
    # ... predict
```

**Error handling:** Catch FerroML errors and return meaningful messages:

```python
try:
    preds = model.predict(X)
except ValueError as e:
    # FerroML errors include .hint() remediation text
    raise HTTPException(400, detail=str(e))
```

## Batch Prediction

For large-scale offline prediction:

```python
import numpy as np
from ferroml.trees import RandomForestRegressor

model = RandomForestRegressor.load_msgpack("model.msgpack")

# Process in chunks to manage memory
CHUNK_SIZE = 10_000

def batch_predict(X_all):
    results = []
    for i in range(0, len(X_all), CHUNK_SIZE):
        chunk = X_all[i:i + CHUNK_SIZE]
        preds = model.predict(chunk)
        results.append(preds)
    return np.concatenate(results)

# With uncertainty (GP models)
def batch_predict_with_ci(model, X_all, confidence=0.95):
    results = []
    for i in range(0, len(X_all), CHUNK_SIZE):
        chunk = X_all[i:i + CHUNK_SIZE]
        pred_result = model.predict_with_uncertainty(chunk, confidence=confidence)
        results.append(pred_result)
    return results
```

## Model Versioning

### Reproducibility snapshot

Track everything needed to reproduce a model:

```python
import json
import datetime

def save_model_with_metadata(model, path, X_train, y_train):
    # Save the model
    model.save_msgpack(f"{path}/model.msgpack")

    # Save metadata
    metadata = {
        "model_class": type(model).__name__,
        "trained_at": datetime.datetime.now().isoformat(),
        "n_samples": len(y_train),
        "n_features": X_train.shape[1],
        "search_space": model.search_space(),
        "model_card": type(model).model_card().__dict__,
    }
    with open(f"{path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
```

### Version naming convention

```
models/
  v1.0.0/
    model.msgpack
    metadata.json
  v1.1.0/
    model.msgpack
    metadata.json
```

Use semantic versioning:
- **Patch** (v1.0.1): Retrained on updated data, same features and hyperparameters
- **Minor** (v1.1.0): New features added, hyperparameters tuned
- **Major** (v2.0.0): Different model type, different feature schema

## Docker Deployment

### Minimal Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install ferroml
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and serving code
COPY model.msgpack .
COPY serve.py .

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```
ferroml>=1.0.0
fastapi>=0.100.0
uvicorn>=0.20.0
numpy>=1.24.0
```

### Build and run

```bash
docker build -t ferroml-model:v1.0.0 .
docker run -p 8000:8000 ferroml-model:v1.0.0
```

### Production considerations

| Concern | Recommendation |
|---------|---------------|
| Memory | FerroML models are lightweight. A Random Forest with 100 trees fits in < 100MB. |
| Concurrency | FerroML predict() is thread-safe. Use multiple uvicorn workers. |
| Cold start | Model loading takes < 1 second for most models. Load at startup, not per-request. |
| Health checks | Add a `/health` endpoint that calls `model.predict()` on a small test input. |
| Logging | Log prediction latency, input shape, and any errors. |
| Scaling | Stateless serving -- scale horizontally. Each replica loads its own model copy. |

## Input Validation Patterns

Validate inputs before prediction to catch problems early:

```python
import numpy as np

def validate_input(X, expected_features):
    """Validate prediction input before calling model.predict()."""
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    if X.ndim == 1:
        X = X.reshape(1, -1)  # single sample

    if X.ndim != 2:
        raise ValueError(f"Input must be 2D, got {X.ndim}D")

    if X.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, got {X.shape[1]}"
        )

    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values -- impute before prediction")

    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")

    return X
```

## Monitoring in Production

After deployment, monitor:

1. **Prediction latency** -- p50, p95, p99 per request
2. **Error rate** -- how often predict() raises exceptions
3. **Input distribution** -- are features drifting from training data? (see drift-monitoring-guide.md)
4. **Prediction distribution** -- are outputs changing over time?
5. **Model staleness** -- when was the model last retrained?

See `drift-monitoring-guide.md` for detailed monitoring strategies.

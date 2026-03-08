# Plan I: Anomaly Detection (IsolationForest + LocalOutlierFactor)

## Overview

Implement two core anomaly detection algorithms. IsolationForest uses random isolation trees; LocalOutlierFactor uses k-nearest neighbor density estimation. Both produce anomaly scores and binary predictions.

## Current State

- Tree infrastructure: `TreeNode`, `TreeStructure`, `DecisionTreeClassifier/Regressor` in `models/tree.rs`
- Random forest pattern: parallel tree building with rayon, bootstrap sampling in `models/forest.rs`
- KNN infrastructure: KDTree, BallTree, distance metrics in `models/knn.rs`
- No anomaly detection models exist
- No `score_samples()` in base Model trait (SVM has `decision_function()` but it's model-specific)
- `Task` enum has no AnomalyDetection variant (`lib.rs:407`)
- Existing traits in `models/traits.rs`: LinearModel, TreeModel, IncrementalModel, WarmStartModel

## Desired End State

- `IsolationForest` struct with fit/predict/score_samples/decision_function
- `LocalOutlierFactor` struct with fit/predict/score_samples/negative_outlier_factor
- New `OutlierDetector` trait for shared interface
- Python bindings + tests for both
- ~30 Rust tests per model, ~15 Python tests per model

---

## Phase I.1: OutlierDetector Trait

**Overview**: Define a shared trait for anomaly detection models.

**Changes Required**:

1. **File**: `ferroml-core/src/models/traits.rs` (EDIT)
   - Add new trait (match sklearn sign conventions: lower = more anomalous):
     ```rust
     pub trait OutlierDetector: Send + Sync {
         fn fit(&mut self, x: &Array2<f64>) -> Result<()>;
         fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>>;
         // Returns +1 for inliers, -1 for outliers
         fn fit_predict(&mut self, x: &Array2<f64>) -> Result<Array1<i32>>;
         fn score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>>;
         // Lower = more anomalous (opposite of paper convention)
         fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>>;
         // score_samples - offset_ (negative = outlier)
         fn is_fitted(&self) -> bool;
         fn offset(&self) -> f64;
     }
     ```

2. **File**: `ferroml-core/src/models/mod.rs` (EDIT)
   - Re-export `OutlierDetector` trait

**Success Criteria**:
- [ ] `cargo check -p ferroml-core`

---

## Phase I.2: IsolationForest

**Overview**: Implement IsolationForest using random recursive partitioning. Anomaly score = average path length across trees (shorter path = more anomalous).

**Changes Required**:

1. **File**: `ferroml-core/src/models/isolation_forest.rs` (NEW, ~500 lines)

   **Struct** (match sklearn defaults):
   ```rust
   pub struct IsolationForest {
       n_estimators: usize,       // number of isolation trees (default 100)
       max_samples: MaxSamples,   // samples per tree (default "auto" = min(256, n_samples))
       contamination: Contamination, // "auto" (offset=-0.5) or float in (0, 0.5]
       max_features: f64,         // fraction of features per tree (default 1.0)
       bootstrap: bool,           // sample with replacement (default false)
       random_state: Option<u64>,
       // Fitted
       trees: Option<Vec<IsolationTree>>,
       offset_: Option<f64>,      // threshold offset (default -0.5 for "auto")
       max_samples_: Option<usize>, // actual samples used
       n_features_in_: Option<usize>,
   }

   struct IsolationTree {
       nodes: Vec<IsoNode>,
   }

   struct IsoNode {
       feature: usize,
       threshold: f64,
       left: Option<usize>,
       right: Option<usize>,
       size: usize,  // number of samples that reached this node
   }
   ```

   **Key algorithm** (per sklearn source `_iforest.py`):
   - Uses ExtraTree-style random splits: random feature + random split within feature range
   - Max depth per tree: `ceil(log2(max_samples))`
   - For leaf with `n_left` samples, add `_average_path_length(n_left)` to path length
   - `_average_path_length(n)`:
     - n <= 1: return 0.0
     - n == 2: return 1.0
     - n >= 3: `2.0 * (ln(n-1) + euler_gamma) - 2.0 * (n-1) / n`
   - Raw score per sample: `2^(-depths / (n_estimators * c(max_samples)))`
   - `score_samples()` = `-raw_score` (OPPOSITE of paper: lower = more abnormal)
   - `decision_function()` = `score_samples() - offset_`
   - `offset_` = -0.5 when contamination="auto"
   - `predict()` = +1 where decision_function >= 0, -1 otherwise

   **IMPORTANT sign conventions** (match sklearn exactly):
   - `score_samples()` — NEGATIVE of anomaly score. Lower = more abnormal. Inliers near 0, outliers near -1.
   - `decision_function()` — score_samples minus offset. Negative = outlier, positive = inlier.
   - `predict()` — returns +1 (inlier) or -1 (outlier), NOT 1.0/-1.0 floats.

   **Methods**:
   - `new(n_estimators)` + builder
   - Implement `OutlierDetector` trait
   - `score_samples()` — opposite of anomaly score (lower = more abnormal)
   - `decision_function()` — `score_samples - offset_`
   - `predict()` — +1 (inlier) or -1 (outlier)

2. **File**: `ferroml-core/src/models/mod.rs` (EDIT)
   - Add `pub mod isolation_forest;`
   - Re-export `IsolationForest`

**Success Criteria**:
- [ ] `cargo check -p ferroml-core`
- [ ] IsolationForest detects obvious outliers in synthetic data

---

## Phase I.3: IsolationForest Tests

**Changes Required**:

1. **File**: `ferroml-core/src/models/isolation_forest.rs` (EDIT — `#[cfg(test)]` module)
   - ~30 tests:
     - Detects outliers far from cluster center
     - score_samples in [0, 1] range
     - predict returns only 1.0 or -1.0
     - Contamination controls threshold
     - Reproducibility with random_state
     - n_estimators affects stability (more trees = more stable)
     - max_samples parameter works
     - Single feature data
     - High-dimensional data
     - All inliers case
     - All same values (degenerate case)

**Success Criteria**:
- [ ] `cargo test -p ferroml-core isolation_forest` — all pass

---

## Phase I.4: LocalOutlierFactor

**Overview**: Implement LOF using k-distance and local reachability density. Points with substantially lower density than neighbors are outliers.

**Changes Required**:

1. **File**: `ferroml-core/src/models/lof.rs` (NEW, ~400 lines)

   **Struct** (match sklearn defaults):
   ```rust
   pub struct LocalOutlierFactor {
       n_neighbors: usize,            // k for k-nearest neighbors (default 20)
       contamination: Contamination,  // "auto" or float in (0, 0.5]
       metric: DistanceMetric,        // distance metric (default Minkowski p=2 = Euclidean)
       algorithm: KNNAlgorithm,       // Auto, KDTree, BallTree, BruteForce
       novelty: bool,                 // false = outlier detection (fit_predict only)
                                      // true = novelty detection (predict on new data)
       // Fitted
       x_train_: Option<Array2<f64>>,  // stored training data (needed for scoring)
       negative_outlier_factor_: Option<Array1<f64>>,  // OPPOSITE of LOF (near -1 = inlier)
       offset_: Option<f64>,           // threshold for binary classification
       n_features_in_: Option<usize>,
   }
   ```

   **Key algorithm**:
   - k-distance(A) = distance to k-th nearest neighbor of A
   - reach-dist(A, B) = max(k-distance(B), d(A, B))
   - lrd(A) = 1 / (mean of reach-dist(A, neighbor_i) for all k neighbors)
   - LOF(A) = mean(lrd(neighbor_i) / lrd(A)) for all k neighbors
   - LOF ~ 1 means inlier; LOF >> 1 means outlier

   **IMPORTANT sign conventions** (match sklearn exactly):
   - `negative_outlier_factor_` = OPPOSITE of LOF. Near -1 = inlier, large negative = outlier.
   - `score_samples()` = `negative_outlier_factor_` for training data (novelty=true only for new data)
   - `decision_function()` = `score_samples - offset_`
   - `predict()` = +1 (inlier) or -1 (outlier)
   - **novelty=false** (default): only `fit_predict()` works, NOT `predict()` on new data
   - **novelty=true**: `predict()`, `score_samples()`, `decision_function()` work on NEW data only

   **Methods**:
   - Implement `OutlierDetector` trait
   - `fit_predict()` — outlier detection mode (novelty=false)
   - `predict()`, `score_samples()`, `decision_function()` — novelty mode (novelty=true)
   - `negative_outlier_factor_` accessor — opposite LOF scores for training data
   - Reuse KDTree/BallTree from `models/knn.rs` for neighbor queries

2. **File**: `ferroml-core/src/models/mod.rs` (EDIT)
   - Add `pub mod lof;`
   - Re-export `LocalOutlierFactor`

**Success Criteria**:
- [ ] `cargo check -p ferroml-core`
- [ ] LOF detects density-based outliers

---

## Phase I.5: LocalOutlierFactor Tests

**Changes Required**:

1. **File**: `ferroml-core/src/models/lof.rs` (EDIT — `#[cfg(test)]` module)
   - ~30 tests:
     - Detects outliers in sparse region
     - LOF ~ 1 for points in dense cluster
     - LOF >> 1 for isolated points
     - n_neighbors affects sensitivity
     - Different distance metrics work
     - predict returns 1.0 or -1.0
     - Contamination controls threshold
     - Reproducibility (deterministic for same input)
     - Edge cases: k >= n_samples, single cluster

**Success Criteria**:
- [ ] `cargo test -p ferroml-core lof` — all pass

---

## Phase I.6: Python Bindings + Tests

**Changes Required**:

1. **File**: `ferroml-python/src/anomaly.rs` (NEW)
   - `PyIsolationForest` wrapper
   - `PyLocalOutlierFactor` wrapper
   - Both: `__new__`, `fit`, `predict`, `score_samples`, `decision_function`, `__repr__`

2. **File**: `ferroml-python/src/lib.rs` (EDIT)
   - Add `mod anomaly;` and register submodule

3. **File**: `ferroml-python/python/ferroml/anomaly/__init__.py` (NEW)
   - Export IsolationForest, LocalOutlierFactor

4. **File**: `ferroml-python/python/ferroml/__init__.py` (EDIT)
   - Add anomaly module

5. **File**: `ferroml-python/tests/test_anomaly.py` (NEW)
   - ~30 tests total for both models

**Success Criteria**:
- [ ] `cargo check -p ferroml-python`
- [ ] `pytest ferroml-python/tests/test_anomaly.py -v` — all pass

---

## Execution Order

```
I.1 (OutlierDetector trait)   — 10 min
I.2 (IsolationForest)         — 30 min, depends on I.1
I.3 (IF tests)                — 15 min, depends on I.2
I.4 (LocalOutlierFactor)      — 30 min, depends on I.1
I.5 (LOF tests)               — 15 min, depends on I.4
I.6 (Python bindings)         — 20 min, depends on I.2 + I.4
```

I.2+I.3 and I.4+I.5 can run in parallel after I.1.

## Dependencies

- KNN infrastructure exists (KDTree, BallTree) in `models/knn.rs`
- Tree infrastructure exists in `models/tree.rs`

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LOF requires storing training data (memory) | Document this; it's inherent to the algorithm |
| IsolationForest path length normalization | Use standard c(n) formula with harmonic numbers |
| KDTree/BallTree may not be pub-accessible | May need to make neighbor query methods pub(crate) |
| Contamination = "auto" complexity | Start with fixed contamination; add "auto" later if needed |

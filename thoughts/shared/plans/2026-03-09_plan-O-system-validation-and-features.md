# Plan O: System-Level Validation + Feature Completion

## Overview

Plan O addresses two parallel tracks: (1) system-level validation of the AutoML pipeline end-to-end, and (2) remaining feature/performance work. Testing comes first because the existing 51 AutoML tests focus on component-level logic (bandits, time budgets) but never validate the integrated system on real data.

## Current State

- **AutoML system**: 11,711 lines across 9 modules, fully implemented (fit → ensemble → predict)
- **AutoML tests**: 51 Rust unit tests (time budget/bandit focused), 63 Python smoke tests
- **Gap**: No tests verify ensemble improvement, prediction quality, reproducibility, or real-data workflows
- **Missing algorithms**: CategoricalNB, HDBSCAN
- **Performance gaps**: LogisticRegression X'WX triple loop, faer-backend disabled by default

## Desired End State

- AutoML validated end-to-end on 4+ real datasets with quality assertions
- Ensemble improvement verified (ensemble ≥ best single model)
- Predict correctness validated (shape, range, class membership)
- Reproducibility with random_state confirmed
- CategoricalNB completing NB family (4/4)
- HDBSCAN competing with modern density clustering
- LogisticRegression 2-3x faster via ndarray .dot()
- faer-backend enabled by default

## Implementation Phases

### Phase O.1: AutoML System-Level Validation (Rust)

**Overview**: Add end-to-end AutoML tests on real datasets with quality assertions.

**File**: `ferroml-core/src/testing/automl.rs` (append new test module)

**Tests to add** (~20 tests):

1. **End-to-end classification on Iris**
   - `AutoML::new(classification, accuracy, 60s).fit(X, y)`
   - Assert: leaderboard non-empty, best score > 0.85, predict returns correct shape
   - Assert: predict values ∈ {0, 1, 2} (valid classes)

2. **End-to-end classification on Wine**
   - Same pattern, best score > 0.80
   - Assert: summary() produces non-empty string

3. **End-to-end regression on Diabetes**
   - `AutoML::new(regression, r2, 60s).fit(X, y)`
   - Assert: best R² > 0.30 (reasonable for this dataset)
   - Assert: predict returns f64 values in reasonable range

4. **Ensemble improvement verification**
   - Run AutoML with enough time for multiple trials
   - Assert: `ensemble_score >= best_single_model_score - 0.01` (ensemble should not degrade)
   - If ensemble has 2+ members, log improvement

5. **Predict after fit correctness**
   - Split data 80/20
   - Fit on train, predict on test
   - Assert: prediction shape matches test set
   - Assert: classification predictions are valid class labels
   - Assert: regression predictions are finite and non-NaN

6. **Reproducibility with random_state**
   - Run AutoML twice with same seed
   - Assert: leaderboard has same algorithms in same order
   - Assert: CV scores match within epsilon

7. **Competitive models selection**
   - Run with enough time budget
   - Assert: `competitive_models()` includes best model
   - Assert: all competitive models have non-significantly-different scores

8. **Feature importance aggregation**
   - Run on Iris (4 features)
   - Assert: `top_features(4)` returns 4 entries
   - Assert: feature importance values are non-negative

9. **Different metrics**
   - Test with accuracy, f1, roc_auc (classification)
   - Test with mse, mae, r2 (regression)
   - Assert: scores are in valid ranges for each metric

10. **Small dataset handling**
    - 30 samples, 5 features
    - Assert: does not crash, produces at least 1 successful trial

11. **Imbalanced dataset**
    - 90/10 class split
    - Assert: completes without error, balanced_accuracy reported

12. **Cross-validation fairness**
    - Assert: all trials use same fold indices (verified via OOF prediction shapes)

**Success Criteria**:
- [ ] `cargo test -p ferroml-core --lib testing::automl -- system_level` — all pass
- [ ] Tests complete within 10 minutes total (60s budget each, ~12 tests)

---

### Phase O.2: AutoML System-Level Validation (Python)

**Overview**: Add Python end-to-end tests that exercise the API as a user would.

**File**: `ferroml-python/tests/test_automl_system.py` (new file)

**Tests to add** (~15 tests):

1. **Full workflow: fit → summary → predict**
   ```python
   config = AutoMLConfig(task="classification", metric="accuracy", time_budget_seconds=30)
   automl = AutoML(config)
   result = automl.fit(X_train, y_train)
   summary = result.summary()
   predictions = result.predict(X_train, y_train, X_test)
   assert predictions.shape == (len(X_test),)
   ```

2. **Leaderboard inspection**
   - Access leaderboard entries
   - Verify algorithm, cv_score, cv_std, rank attributes
   - Verify sorting by score (descending)

3. **Ensemble access**
   - Check ensemble members, weights, scores
   - Verify weights sum to ~1.0

4. **Classification on sklearn Iris**
   ```python
   from sklearn.datasets import load_iris
   X, y = load_iris(return_X_y=True)
   ```
   - Assert accuracy > 0.85

5. **Regression on sklearn Diabetes**
   ```python
   from sklearn.datasets import load_diabetes
   X, y = load_diabetes(return_X_y=True)
   ```
   - Assert R² > 0.30

6. **Messy data handling**
   - Data with different scales (some features 0-1, others 0-1000)
   - Assert: completes without error (preprocessing should handle it)

7. **Confidence interval attributes**
   - Verify CI bounds exist and lower < upper
   - Verify CI contains the point estimate

8. **top_features() API**
   - Verify returns list of (feature_index, importance) tuples
   - Verify importance values are non-negative

9. **Multiple metrics comparison**
   - Run with accuracy, then with f1
   - Verify different metrics produce different rankings

10. **Error handling**
    - Invalid task type → clear error
    - Empty array → clear error
    - Mismatched X/y shapes → clear error

**Success Criteria**:
- [ ] `pytest ferroml-python/tests/test_automl_system.py -v` — all pass
- [ ] Tests complete within 5 minutes total

---

### Phase O.3: LogisticRegression Optimization

**Overview**: Replace O(n·p²) triple nested loop with ndarray matrix operations.

**File**: `ferroml-core/src/models/logistic.rs`

**Changes Required**:

1. **Replace X'WX construction** (lines 478-486):
   ```rust
   // BEFORE: Triple nested loop O(n·p²) with poor cache locality
   let mut xtwx = Array2::zeros((p, p));
   for i in 0..n {
       let xi = x_design.row(i);
       for j in 0..p {
           for k in 0..p {
               xtwx[[j, k]] += w_clamped[i] * xi[j] * xi[k];
           }
       }
   }

   // AFTER: Scale rows by sqrt(w), then GEMM
   let mut scaled_x = x_design.to_owned();
   for i in 0..n {
       let w_sqrt = w_clamped[i].sqrt();
       scaled_x.row_mut(i).mapv_inplace(|v| v * w_sqrt);
   }
   let xtwx = scaled_x.t().dot(&scaled_x);
   ```

2. **Replace X'Wz construction** (lines 488-494):
   ```rust
   // BEFORE
   let mut xtwz = Array1::zeros(p);
   for i in 0..n {
       let xi = x_design.row(i);
       for j in 0..p {
           xtwz[j] += w_clamped[i] * xi[j] * z[i];
       }
   }

   // AFTER: Reuse scaled_x from above
   let wz = &z * &w_clamped.mapv(f64::sqrt);
   let xtwz = scaled_x.t().dot(&wz);
   ```

   Wait — X'Wz ≠ (W^½X)'(W^½z). Let me be precise:
   X'Wz = Σᵢ wᵢ xᵢ zᵢ = X' @ diag(w) @ z = X' @ (w * z)

   ```rust
   // AFTER: Direct weighted dot product
   let wz = &w_clamped * &z;
   let xtwz = x_design.t().dot(&wz);
   ```

3. **Apply same fix to QuantileRegression** (`quantile.rs:383-395`):
   Same pattern — triple nested loop → ndarray operations.

**Success Criteria**:
- [ ] `cargo test -p ferroml-core --lib models::logistic` — all pass (no behavior change)
- [ ] `cargo test -p ferroml-core --lib models::quantile` — all pass
- [ ] `cargo bench -p ferroml-core -- LogisticRegression` — measurable speedup
- [ ] `cargo bench -p ferroml-core -- QuantileRegression` — measurable speedup

---

### Phase O.4: CategoricalNB

**Overview**: Add CategoricalNB to complete the Naive Bayes family (4/4 sklearn parity).

**Changes Required**:

1. **Rust implementation** — append to `ferroml-core/src/models/naive_bayes.rs` (~450 lines)
   - Struct: `CategoricalNB { alpha, fit_prior, class_prior, class_weight, category_count, class_count, classes, feature_categories, n_features, class_log_prior, feature_log_prob }`
   - `category_count`: `Vec<Array2<f64>>` — per feature, shape (n_classes, n_categories_j)
   - `feature_log_prob`: `Vec<Array2<f64>>` — log P(category | class) per feature
   - `feature_categories`: `Vec<Vec<f64>>` — unique categories per feature
   - Builder: `new()`, `with_alpha()`, `with_fit_prior()`, `with_class_prior()`, `with_class_weight()`
   - `partial_fit()`: Track unique categories, expand arrays as new categories appear
   - `update_log_probabilities()`: `log((count + alpha) / (class_total + alpha * n_categories))`
   - `joint_log_likelihood()`: Sum log probs for observed categories
   - Implement `Model` trait (fit, predict, is_fitted, feature_importance, search_space)
   - Implement `ProbabilisticModel` trait (predict_proba, predict_interval)

2. **Python bindings** — append to `ferroml-python/src/naive_bayes.rs` (~130 lines)
   - `PyCategoricalNB` with same pattern as other NB classes
   - `fit()`, `predict()`, `predict_proba()`, getters for `classes_`, `category_count_`, `feature_log_prob_`

3. **Update exports**:
   - `ferroml-core/src/models/mod.rs` — add `CategoricalNB` to pub use
   - `ferroml-python/python/ferroml/naive_bayes/__init__.py` — add export
   - `ferroml-python/src/lib.rs` — register `PyCategoricalNB`

4. **Rust tests** (~25 tests in `ferroml-core/src/testing/` or correctness test file):
   - Basic fit/predict on categorical data
   - Probability sums to 1
   - Laplace smoothing effect
   - partial_fit consistency
   - Unknown category handling
   - Single class, single feature edge cases
   - Comparison with sklearn CategoricalNB

5. **Python tests** (~15 tests in `ferroml-python/tests/test_naive_bayes.py`):
   - Same pattern as existing NB tests
   - sklearn comparison

**Success Criteria**:
- [ ] `cargo test -p ferroml-core --lib naive_bayes::categorical` — all pass
- [ ] `pytest ferroml-python/tests/test_naive_bayes.py -v -k categorical` — all pass

---

### Phase O.5: HDBSCAN

**Overview**: Add HDBSCAN hierarchical density-based clustering.

**Changes Required**:

1. **Rust implementation** — new file `ferroml-core/src/clustering/hdbscan.rs` (~800 lines)

   **Struct**:
   ```rust
   pub struct HDBSCAN {
       min_cluster_size: usize,     // Minimum cluster membership
       min_samples: Option<usize>,  // Core distance k (defaults to min_cluster_size)
       cluster_selection_epsilon: f64,
       metric: DistanceMetric,
       // Fitted
       labels_: Option<Array1<i32>>,        // -1 for noise
       probabilities_: Option<Array1<f64>>, // Cluster membership strength
       core_distances_: Option<Array1<f64>>,
       n_clusters_: Option<usize>,
   }
   ```

   **Core algorithm**:
   - `compute_core_distances(x, k)` → Array1<f64>: k-th nearest neighbor distance per point. Reuse VP-tree from `decomposition/vptree.rs` for O(N log N) k-NN.
   - `compute_mutual_reachability(x, core_dists)` → Vec<(usize, usize, f64)>: Edge list with d_mreach = max(core_dist_a, core_dist_b, d(a,b)). Full O(N²) for now.
   - `minimum_spanning_tree(edges, n)` → Vec<(usize, usize, f64)>: Prim's algorithm on mutual reachability graph.
   - `build_condensed_tree(mst, min_cluster_size)` → CondensedTree: Walk MST from largest to smallest distance, tracking cluster births/deaths/splits.
   - `extract_clusters(tree)` → (labels, probabilities): Select most stable clusters, assign noise to -1.

   **CondensedTree struct**:
   ```rust
   struct CondensedTreeNode {
       parent: usize,
       child: usize,         // cluster or point index
       lambda_val: f64,       // 1/distance at which this event happens
       child_size: usize,
   }
   ```

   Implement `ClusteringModel` trait (fit, predict, fit_predict, labels, is_fitted).

2. **Module registration** — `ferroml-core/src/clustering/mod.rs`
   - Add `pub mod hdbscan;` and `pub use hdbscan::HDBSCAN;`

3. **Python bindings** — append to `ferroml-python/src/clustering.rs` (~200 lines)
   - `PyHDBSCAN` with `fit()`, `fit_predict()`, `labels_`, `probabilities_`, `n_clusters_`

4. **Update Python exports**:
   - `ferroml-python/python/ferroml/clustering/__init__.py` — add HDBSCAN

5. **Rust tests** (~40 tests in `ferroml-core/tests/correctness_clustering.rs`):
   - Well-separated blobs → finds correct clusters
   - Noise detection (scattered points labeled -1)
   - min_cluster_size effect
   - Single cluster dataset
   - Large min_cluster_size → fewer clusters
   - Comparison with hdbscan Python package on reference datasets
   - Edge cases: 1 point, 2 points, all same point

6. **Python tests** (~20 tests in `ferroml-python/tests/test_clustering.py`):
   - Basic fit/predict
   - Labels and probabilities attributes
   - Noise handling
   - Parameter validation

**Success Criteria**:
- [ ] `cargo test -p ferroml-core --lib clustering::hdbscan` — all pass
- [ ] `cargo test -p ferroml-core --test correctness_clustering -- hdbscan` — all pass
- [ ] `pytest ferroml-python/tests/test_clustering.py -v -k hdbscan` — all pass

---

### Phase O.6: faer-backend by Default

**Overview**: Enable faer for faster QR decomposition on large matrices.

**File**: `ferroml-core/Cargo.toml`

**Changes Required**:

1. **Enable by default** (Cargo.toml line ~126):
   ```toml
   default = ["parallel", "onnx", "simd", "faer-backend"]
   ```

2. **Add faer Cholesky** — `ferroml-core/src/linalg.rs` (extend existing faer integration)
   - Add `cholesky_decomposition_faer()` alongside existing `cholesky_decomposition()`
   - Add `#[cfg(feature = "faer-backend")]` dispatch like QR
   - This benefits LogisticRegression, Ridge, QuantileRegression, RobustRegression

3. **Benchmark before/after**:
   - `cargo bench -- LinearRegression` with and without faer
   - `cargo bench -- LogisticRegression` (after O.3 optimization)

4. **Verify all tests pass**:
   - `cargo test -p ferroml-core` with faer-backend enabled

**Success Criteria**:
- [ ] `cargo test -p ferroml-core` — all 2,873+ tests pass with faer enabled
- [ ] `cargo bench -- LinearRegression` — no regression (should improve)
- [ ] Users can still `--no-default-features` to disable faer

---

### Phase O.7: AutoML Portfolio Update

**Overview**: Add CategoricalNB and HDBSCAN to AutoML portfolio after phases O.4-O.5.

**File**: `ferroml-core/src/automl/portfolio.rs`

**Changes Required**:

1. Add `CategoricalNB` to `AlgorithmType` enum and classification portfolio
2. Add HDBSCAN awareness (if applicable — HDBSCAN is unsupervised, may not fit AutoML classification/regression pipeline directly)
3. Update `adapt_to_data()` to prefer CategoricalNB when data has integer/categorical features

**Success Criteria**:
- [ ] AutoML system tests still pass
- [ ] CategoricalNB appears in leaderboard when appropriate

---

## Phase Ordering & Dependencies

```
O.1 (AutoML Rust tests)     ──┐
O.2 (AutoML Python tests)   ──┤── Can run in parallel, no deps
O.3 (LogReg optimization)   ──┤
O.4 (CategoricalNB)         ──┤
                               │
O.5 (HDBSCAN)               ──┘── Independent

O.6 (faer-backend)          ──── After O.3 (benchmark comparison)
O.7 (Portfolio update)       ──── After O.4, O.5
```

Phases O.1-O.5 are independent and can be parallelized across sessions.

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| AutoML tests too slow (60s budget × 12) | Use 30s budgets for most, 60s only for ensemble tests |
| ndarray .dot() not faster than hand loop | Benchmark first on representative sizes; BLAS acceleration matters |
| HDBSCAN O(N²) mutual reachability slow | Start with brute force, optimize with KD-tree later |
| faer conversion overhead on small matrices | Benchmark with n<50 to verify; keep MGS fallback available |
| CategoricalNB unknown category handling | Follow sklearn: raise error or ignore based on config |

## Estimated Scope

| Phase | New Lines | Tests | Priority |
|-------|-----------|-------|----------|
| O.1 | ~400 | ~20 Rust | HIGH — validates the system |
| O.2 | ~300 | ~15 Python | HIGH — validates Python API |
| O.3 | ~30 (net reduction) | existing | MEDIUM — perf improvement |
| O.4 | ~600 | ~40 | MEDIUM — completeness |
| O.5 | ~1000 | ~60 | MEDIUM — competitive feature |
| O.6 | ~50 | existing | LOW — perf improvement |
| O.7 | ~30 | existing | LOW — integration |

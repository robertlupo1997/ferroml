# Plan 3: Clustering Implementation

**Date:** 2026-02-08
**Reference:** `thoughts/shared/research/2026-02-08_ferroml-comprehensive-assessment.md`
**Priority:** High (major gap vs sklearn)
**Estimated Tasks:** 10

## Objective

Implement clustering algorithms (KMeans, DBSCAN) with FerroML-style statistical extensions that exceed R package rigor.

## Context

From research:
- No clustering algorithms currently implemented
- Major gap compared to sklearn
- User wants MORE statistical rigor than R packages

## Statistical Extensions (Beyond sklearn)

1. **Cluster Stability Metrics** - Bootstrap-based stability assessment
2. **Silhouette Analysis with CI** - Confidence intervals on silhouette scores
3. **Gap Statistic** - Optimal k selection with SE
4. **Cluster Quality Diagnostics** - Within-cluster SS, between-cluster SS, Calinski-Harabasz with inference
5. **Hopkins Statistic** - Clustering tendency assessment

## Tasks

### Task 3.1: Create clustering module structure
**File:** `ferroml-core/src/clustering/mod.rs`
**Description:** Module structure with traits (ClusteringModel, ClusterMetrics).
**Exports:** KMeans, DBSCAN, metrics, diagnostics

### Task 3.2: Implement KMeans core algorithm
**File:** `ferroml-core/src/clustering/kmeans.rs`
**Description:** Lloyd's algorithm with kmeans++ initialization.
**API:** fit(), predict(), fit_predict(), cluster_centers_, inertia_, n_iter_
**Lines:** ~300

### Task 3.3: Add KMeans statistical extensions
**File:** `ferroml-core/src/clustering/kmeans.rs`
**Description:**
- `cluster_stability()` - Bootstrap stability scores per cluster
- `optimal_k()` - Elbow method + Gap statistic with SE
- `silhouette_scores_with_ci()` - Per-sample silhouette with bootstrap CI
**Lines:** ~200

### Task 3.4: Implement DBSCAN core algorithm
**File:** `ferroml-core/src/clustering/dbscan.rs`
**Description:** Density-based clustering with eps and min_samples.
**API:** fit(), fit_predict(), labels_, core_sample_indices_, components_
**Lines:** ~250

### Task 3.5: Add DBSCAN statistical extensions
**File:** `ferroml-core/src/clustering/dbscan.rs`
**Description:**
- `cluster_persistence()` - Stability across eps values
- `noise_analysis()` - Statistical profile of noise points
- `optimal_eps()` - k-distance graph analysis
**Lines:** ~150

### Task 3.6: Implement clustering metrics
**File:** `ferroml-core/src/clustering/metrics.rs`
**Description:**
- `silhouette_score()` and `silhouette_samples()`
- `calinski_harabasz_score()` with inference
- `davies_bouldin_score()`
- `adjusted_rand_index()`, `normalized_mutual_info()`
- `hopkins_statistic()` - Clustering tendency
**Lines:** ~300

### Task 3.7: Implement cluster diagnostics
**File:** `ferroml-core/src/clustering/diagnostics.rs`
**Description:**
- `ClusterDiagnostics` struct (like model diagnostics)
- Within-cluster variance analysis
- Cluster separation tests
- Outlier detection within clusters
**Lines:** ~200

### Task 3.8: Add KMeans tests
**File:** `ferroml-core/src/clustering/kmeans.rs` (tests module)
**Description:**
- Basic clustering on blobs
- Sklearn comparison
- Stability metric tests
- Gap statistic tests
**Tests:** ~15

### Task 3.9: Add DBSCAN tests
**File:** `ferroml-core/src/clustering/dbscan.rs` (tests module)
**Description:**
- Basic clustering on moons/circles
- Noise detection
- Sklearn comparison
**Tests:** ~10

### Task 3.10: Add Python bindings for clustering
**File:** `ferroml-python/src/clustering.rs`
**Description:** PyO3 bindings for KMeans, DBSCAN, metrics.
**Lines:** ~200

## Success Criteria

- [ ] KMeans matches sklearn predictions within 1e-6
- [ ] DBSCAN matches sklearn labels exactly
- [ ] Statistical extensions provide value beyond sklearn
- [ ] All metrics tested against sklearn equivalents
- [ ] Hopkins statistic validated on known clusterable/non-clusterable data

## API Design (sklearn-compatible with extensions)

```rust
// Basic sklearn-compatible
let kmeans = KMeans::new(n_clusters=3).fit(&X)?;
let labels = kmeans.predict(&X)?;

// FerroML statistical extensions
let stability = kmeans.cluster_stability(n_bootstrap=100)?;
let (optimal_k, gap_stats) = KMeans::optimal_k(&X, k_range=1..10)?;
let silhouette_ci = kmeans.silhouette_scores_with_ci(confidence=0.95)?;
```

## Dependencies

- ndarray (existing)
- rand (existing, for kmeans++ init)
- No new crates required

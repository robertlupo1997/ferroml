# Plan A: Clustering Module Hardening

**Date:** 2026-02-25
**Priority:** HIGH (1 critical bug, 0 sklearn-correctness tests)
**Module:** `ferroml-core/src/clustering/` (2,971 lines, 29 inline tests)
**Estimated New Tests:** ~50
**Parallel-Safe:** Yes (no overlap with neural/benchmark/preprocessing plans)

## Overview

The clustering module (KMeans, DBSCAN, AgglomerativeClustering) is feature-complete with statistical extensions beyond sklearn. However, it has **1 critical formula bug**, **3 medium bugs**, and **zero sklearn fixture-based correctness tests**. This plan fixes bugs first, then adds comprehensive correctness tests.

## Current State

### What Exists
- **KMeans** (`kmeans.rs:823`): Lloyd's with kmeans++, gap statistic, elbow, cluster stability, silhouette CI
- **DBSCAN** (`dbscan.rs:577`): Core algorithm + optimal_eps, cluster_persistence, noise_analysis
- **AgglomerativeClustering** (`agglomerative.rs:398`): Ward/Single/Complete/Average linkage
- **Metrics** (`metrics.rs:684`): silhouette, calinski_harabasz, davies_bouldin, ARI, NMI, hopkins
- **Diagnostics** (`diagnostics.rs:336`): ClusterDiagnostics with variance decomposition, Dunn index
- 29 inline tests (all passing) — basic smoke tests only

### What's Wrong

| # | Location | Bug | Severity |
|---|----------|-----|----------|
| 1 | `agglomerative.rs:217` | Ward linkage Lance-Williams formula squares distances twice (`dij * dij` instead of `dij`) | **CRITICAL** |
| 2 | `kmeans.rs:329` | Empty clusters keep old center instead of re-initializing to random sample | MEDIUM |
| 3 | `kmeans.rs:550-572` | `predict()` doesn't validate feature count — panics on mismatch | MEDIUM |
| 4 | `metrics.rs:530` | Hopkins statistic may include sampled points as their own nearest neighbor | MEDIUM |

## Desired End State

- All 4 bugs fixed with regression tests
- 50+ correctness tests comparing against sklearn/scipy fixtures
- Every public function tested against sklearn equivalent
- Edge cases covered (single cluster, all noise, degenerate data)

## Implementation Phases

### Phase A.1: Bug Fixes (4 changes)

#### A.1.1: Fix Ward Linkage Formula
**File:** `ferroml-core/src/clustering/agglomerative.rs:210-220`

Current (WRONG):
```rust
Linkage::Ward => {
    let nk = sizes[k] as f64;
    let di = dist[[merge_i, k]];
    let dj = dist[[merge_j, k]];
    let dij = min_dist;
    let total = ni + nj + nk;
    (((ni + nk) * di * di + (nj + nk) * dj * dj - nk * dij * dij) / total)
        .max(0.0)
        .sqrt()
}
```

The correct Lance-Williams recurrence for Ward is:
```
d(ij,k) = sqrt(((n_i + n_k) * d_ik^2 + (n_j + n_k) * d_jk^2 - n_k * d_ij^2) / (n_i + n_j + n_k))
```

BUT `di`, `dj`, and `dij` are **already distances** (not squared), yet the formula treats them as squared by multiplying `di * di`. This double-squares them to d^4.

Fix: Either work with squared distances throughout, or remove the extra multiplication. The correct code using raw distances:
```rust
(((ni + nk) * di.powi(2) + (nj + nk) * dj.powi(2) - nk * dij.powi(2)) / total)
    .max(0.0)
    .sqrt()
```

Wait — this IS `di * di` which is `di.powi(2)`. So we need to check: are `di`, `dj`, `dij` stored as distances or squared distances in the distance matrix?

**Action:** Read the distance matrix initialization to determine if values are distances or squared distances. If they're distances, the formula `di * di` is correct (squaring distance to get d^2). If they're already d^2, then `di * di` = d^4 which is wrong and should be just `di`.

**Verification:** Compare 4-point Ward clustering output against `scipy.cluster.hierarchy.ward`.

#### A.1.2: Fix KMeans Empty Cluster Handling
**File:** `ferroml-core/src/clustering/kmeans.rs:323-331`

Current: Empty clusters keep old center.
Fix: Re-initialize empty cluster center to random sample from data (matching sklearn behavior).

```rust
} else {
    // Re-initialize empty cluster to random data point
    let idx = rng.gen_range(0..n_samples);
    new_centers.row_mut(k).assign(&x.row(idx));
}
```

**Verification:** Create test where k > natural clusters, verify all clusters get assigned points.

#### A.1.3: Fix KMeans predict() Feature Validation
**File:** `ferroml-core/src/clustering/kmeans.rs:550-572`

Add at start of `predict()`:
```rust
let n_features = centers.ncols();
if x.ncols() != n_features {
    return Err(FerroError::ShapeMismatch {
        expected: vec![0, n_features],
        got: vec![x.nrows(), x.ncols()],
    });
}
```

**Verification:** Test that wrong feature count returns error, not panic.

#### A.1.4: Fix Hopkins Statistic Self-Neighbor
**File:** `ferroml-core/src/clustering/metrics.rs:530+`

Ensure sampled points are excluded from nearest-neighbor search (or skip distance=0).

**Verification:** Test Hopkins on uniform random data (should be ~0.5) and clustered data (should be >0.7).

### Phase A.2: Python Fixture Generation

**File:** `benchmarks/clustering_fixtures.py` (NEW)

Generate sklearn/scipy fixtures for all clustering tests. Each fixture is a JSON file in `benchmarks/fixtures/`.

**Fixtures needed:**

```python
# KMeans fixtures
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Basic blobs (easy, 3 clusters, 150 samples)
X, y_true = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)
km = KMeans(n_clusters=3, random_state=42, n_init=1, init='k-means++')
km.fit(X)
save_fixture("kmeans_blobs_3c", {
    "X": X, "y_true": y_true,
    "labels": km.labels_, "centers": km.cluster_centers_,
    "inertia": km.inertia_, "n_iter": km.n_iter_
})

# 2. High-dimensional (50 features)
# 3. Many clusters (k=10)
# 4. Unbalanced cluster sizes
# 5. Single cluster (k=1)

# DBSCAN fixtures
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_circles

# 6. Moons dataset
# 7. Circles dataset
# 8. All noise (eps too small)
# 9. Single cluster (eps too large)
# 10. Mixed noise + clusters

# AgglomerativeClustering fixtures
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# 11. Ward linkage (3 clusters)
# 12. Single linkage
# 13. Complete linkage
# 14. Average linkage
# 15. Compare dendrogram structure vs scipy.hierarchy

# Metrics fixtures
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)

# 16-22. Each metric on blobs data with known labels
# 23-25. Metrics edge cases (perfect agreement, random labels, 2 clusters)
```

### Phase A.3: Correctness Test File

**File:** `ferroml-core/tests/correctness_clustering.rs` (NEW)

Structure follows existing pattern from `correctness_linear.rs`, `correctness_trees.rs`, etc.

```rust
use ferroml_core::clustering::*;
// ... fixture loading helpers ...

mod kmeans_tests {
    #[test] fn test_kmeans_blobs_labels_vs_sklearn() { ... }
    #[test] fn test_kmeans_blobs_centers_vs_sklearn() { ... }
    #[test] fn test_kmeans_blobs_inertia_vs_sklearn() { ... }
    #[test] fn test_kmeans_high_dimensional() { ... }
    #[test] fn test_kmeans_many_clusters() { ... }
    #[test] fn test_kmeans_unbalanced() { ... }
    #[test] fn test_kmeans_single_cluster() { ... }
    #[test] fn test_kmeans_predict_feature_mismatch_error() { ... }
    #[test] fn test_kmeans_empty_cluster_reinitialization() { ... }
    #[test] fn test_kmeans_deterministic_with_seed() { ... }
}

mod dbscan_tests {
    #[test] fn test_dbscan_moons_labels_vs_sklearn() { ... }
    #[test] fn test_dbscan_moons_core_indices_vs_sklearn() { ... }
    #[test] fn test_dbscan_circles_labels_vs_sklearn() { ... }
    #[test] fn test_dbscan_all_noise() { ... }
    #[test] fn test_dbscan_single_cluster() { ... }
    #[test] fn test_dbscan_noise_count_vs_sklearn() { ... }
    #[test] fn test_dbscan_n_clusters_vs_sklearn() { ... }
}

mod agglomerative_tests {
    #[test] fn test_ward_linkage_vs_sklearn() { ... }
    #[test] fn test_ward_linkage_vs_scipy_hierarchy() { ... }
    #[test] fn test_single_linkage_vs_sklearn() { ... }
    #[test] fn test_complete_linkage_vs_sklearn() { ... }
    #[test] fn test_average_linkage_vs_sklearn() { ... }
    #[test] fn test_agglomerative_dendrogram_order() { ... }
    #[test] fn test_agglomerative_two_points() { ... }
    #[test] fn test_agglomerative_identical_points() { ... }
}

mod metrics_tests {
    #[test] fn test_silhouette_score_vs_sklearn() { ... }
    #[test] fn test_silhouette_samples_vs_sklearn() { ... }
    #[test] fn test_calinski_harabasz_vs_sklearn() { ... }
    #[test] fn test_davies_bouldin_vs_sklearn() { ... }
    #[test] fn test_adjusted_rand_index_vs_sklearn() { ... }
    #[test] fn test_adjusted_rand_index_random_labels() { ... }
    #[test] fn test_nmi_vs_sklearn() { ... }
    #[test] fn test_nmi_perfect_agreement() { ... }
    #[test] fn test_hopkins_uniform_data() { ... }
    #[test] fn test_hopkins_clustered_data() { ... }
    #[test] fn test_metrics_with_noise_labels() { ... }
}

mod diagnostics_tests {
    #[test] fn test_variance_decomposition_sums() { ... }
    #[test] fn test_dunn_index_well_separated() { ... }
    #[test] fn test_silhouette_per_cluster_consistency() { ... }
    #[test] fn test_outlier_detection() { ... }
}

mod statistical_extensions_tests {
    #[test] fn test_gap_statistic_finds_correct_k() { ... }
    #[test] fn test_elbow_method_finds_correct_k() { ... }
    #[test] fn test_cluster_stability_stable_clusters() { ... }
    #[test] fn test_silhouette_ci_contains_point_estimate() { ... }
    #[test] fn test_optimal_eps_reasonable_value() { ... }
    #[test] fn test_cluster_persistence_monotonic() { ... }
}
```

**Total: ~50 tests**

### Phase A.4: Edge Case & Regression Tests

Additional tests in the same file for robustness:

```rust
mod edge_cases {
    #[test] fn test_kmeans_one_sample() { ... }
    #[test] fn test_kmeans_k_equals_n() { ... }
    #[test] fn test_dbscan_min_samples_one() { ... }
    #[test] fn test_dbscan_eps_zero() { ... }
    #[test] fn test_all_identical_points() { ... }
    #[test] fn test_single_feature() { ... }
    #[test] fn test_high_dimensional_100_features() { ... }
}
```

## Success Criteria

- [ ] `cargo test -p ferroml-core --test correctness_clustering` — all pass
- [ ] Ward linkage matches `scipy.cluster.hierarchy.ward` within 1e-10
- [ ] KMeans labels match sklearn within label permutation (ARI = 1.0)
- [ ] DBSCAN labels match sklearn exactly (no permutation needed)
- [ ] All 7 metrics match sklearn within 1e-10
- [ ] Hopkins statistic ~0.5 for uniform data, >0.7 for clustered
- [ ] Empty cluster re-initialization works (no stuck empty clusters)
- [ ] Feature mismatch in predict() returns FerroError, not panic

## Dependencies

- Python 3.10+ with sklearn, scipy, numpy for fixture generation
- Existing fixture infrastructure in `benchmarks/fixtures/`
- No new Rust crate dependencies

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| KMeans labels differ from sklearn due to init randomness | Use `n_init=1`, fixed seed, compare via ARI not exact labels |
| Ward formula fix changes existing test results | Run existing 29 tests before/after to track changes |
| DBSCAN label ordering differs from sklearn | Compare sets of points per cluster, not label integers |
| Gap statistic is stochastic | Use fixed seed, wide tolerance (correct k, not exact values) |

# P2: Performance Optimization Plan

## Overview
Implement high-impact performance optimizations: SIMD histogram accumulation for HistGradientBoosting and parallel tree prediction for RandomForest. Target 2-4x speedup for gradient boosting and n_estimators speedup for prediction.

## Current State
- **SIMD Infrastructure**: 1,397 lines in `simd.rs`, only used by KNN
- **HistGradientBoosting**: Sequential histogram building in `build_histograms()` (lines 973-1003)
- **RandomForest Prediction**: Sequential tree evaluation in `predict_proba()` (lines 341-358)
- **Current vs XGBoost**: ~5-10x slower (claimed, needs verification)

## Desired End State
- SIMD histogram accumulation in HistGradientBoosting (2-4x speedup)
- Parallel tree prediction in RandomForest (n_estimators speedup)
- Feature-gated optimizations (backward compatible)
- Benchmark suite to verify improvements

---

## Implementation Phases

### Phase P2.1: SIMD Histogram Subtraction
**Overview**: Vectorize the histogram subtraction operation (quick win)

**Changes Required**:
1. **File**: `ferroml-core/src/models/hist_boosting.rs` (MODIFY lines 370-378)

   ```rust
   // BEFORE:
   fn compute_by_subtraction(&mut self, parent: &Histogram, sibling: &Histogram) {
       for i in 0..self.sum_gradients.len() {
           self.sum_gradients[i] = parent.sum_gradients[i] - sibling.sum_gradients[i];
           self.sum_hessians[i] = parent.sum_hessians[i] - sibling.sum_hessians[i];
           self.counts[i] = parent.counts[i] - sibling.counts[i];
       }
   }

   // AFTER:
   fn compute_by_subtraction(&mut self, parent: &Histogram, sibling: &Histogram) {
       #[cfg(feature = "simd")]
       {
           crate::simd::vector_sub_into(
               &parent.sum_gradients,
               &sibling.sum_gradients,
               &mut self.sum_gradients
           );
           crate::simd::vector_sub_into(
               &parent.sum_hessians,
               &sibling.sum_hessians,
               &mut self.sum_hessians
           );
       }
       #[cfg(not(feature = "simd"))]
       {
           for i in 0..self.sum_gradients.len() {
               self.sum_gradients[i] = parent.sum_gradients[i] - sibling.sum_gradients[i];
               self.sum_hessians[i] = parent.sum_hessians[i] - sibling.sum_hessians[i];
           }
       }
       // counts always sequential (usize, not f64)
       for i in 0..self.counts.len() {
           self.counts[i] = parent.counts[i] - sibling.counts[i];
       }
   }
   ```

2. **File**: `ferroml-core/src/simd.rs` (ADD)

   ```rust
   /// Subtract two vectors, storing result in destination (dst = a - b)
   #[inline]
   pub fn vector_sub_into(a: &[f64], b: &[f64], dst: &mut [f64]) {
       assert_eq!(a.len(), b.len());
       assert_eq!(a.len(), dst.len());

       let n = a.len();
       let chunks = n / 4;
       let remainder = n % 4;

       for i in 0..chunks {
           let idx = i * 4;
           let va = f64x4::new([a[idx], a[idx + 1], a[idx + 2], a[idx + 3]]);
           let vb = f64x4::new([b[idx], b[idx + 1], b[idx + 2], b[idx + 3]]);
           let result = va - vb;
           let arr = result.to_array();
           dst[idx] = arr[0];
           dst[idx + 1] = arr[1];
           dst[idx + 2] = arr[2];
           dst[idx + 3] = arr[3];
       }

       // Handle remainder
       let start = chunks * 4;
       for i in 0..remainder {
           dst[start + i] = a[start + i] - b[start + i];
       }
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core hist_boosting`
- [ ] Automated: `cargo test -p ferroml-core --features simd simd::test_vector_sub_into`

---

### Phase P2.2: Parallel Histogram Building
**Overview**: Parallelize histogram accumulation using thread-local aggregation

**Changes Required**:
1. **File**: `ferroml-core/src/models/hist_boosting.rs` (MODIFY `build_histograms`)

   ```rust
   // BEFORE (lines 973-1003): Sequential double loop
   // AFTER: Parallel with thread-local histograms

   #[cfg(feature = "parallel")]
   fn build_histograms_parallel(
       &self,
       x_binned: &Array2<u8>,
       gradients: &[f64],
       hessians: &[f64],
       indices: &[usize],
       bin_mapper: &dyn BinMapperInfo,
   ) -> Vec<Histogram> {
       use rayon::prelude::*;

       let n_features = x_binned.ncols();
       let n_bins = bin_mapper.n_bins();

       // Parallel over features (each feature gets its own histogram)
       (0..n_features)
           .into_par_iter()
           .map(|f| {
               let mut histogram = Histogram::new(n_bins);

               // Sequential accumulation within feature
               for &idx in indices {
                   let bin = x_binned[[idx, f]] as usize;
                   if bin < n_bins {
                       histogram.sum_gradients[bin] += gradients[idx];
                       histogram.sum_hessians[bin] += hessians[idx];
                       histogram.counts[bin] += 1;
                   }
               }

               histogram
           })
           .collect()
   }

   fn build_histograms(
       &self,
       x_binned: &Array2<u8>,
       gradients: &[f64],
       hessians: &[f64],
       indices: &[usize],
       bin_mapper: &dyn BinMapperInfo,
   ) -> Vec<Histogram> {
       #[cfg(feature = "parallel")]
       {
           self.build_histograms_parallel(x_binned, gradients, hessians, indices, bin_mapper)
       }

       #[cfg(not(feature = "parallel"))]
       {
           // Original sequential implementation
           let n_features = x_binned.ncols();
           let n_bins = bin_mapper.n_bins();
           let mut histograms = Vec::with_capacity(n_features);

           for f in 0..n_features {
               let mut histogram = Histogram::new(n_bins);
               for &idx in indices {
                   let bin = x_binned[[idx, f]] as usize;
                   if bin < n_bins {
                       histogram.sum_gradients[bin] += gradients[idx];
                       histogram.sum_hessians[bin] += hessians[idx];
                       histogram.counts[bin] += 1;
                   }
               }
               histograms.push(histogram);
           }

           histograms
       }
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --features parallel hist_boosting`
- [ ] Manual: Benchmark shows improvement with parallel feature

---

### Phase P2.3: Parallel Forest Prediction
**Overview**: Evaluate trees in parallel during prediction

**Changes Required**:
1. **File**: `ferroml-core/src/models/forest.rs` (MODIFY `predict_proba` for RandomForestClassifier, lines 341-358)

   ```rust
   // BEFORE:
   pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
       check_is_fitted(self.fitted)?;
       validate_predict_input(x, self.n_features)?;

       let n_samples = x.nrows();
       let n_classes = self.classes.len();
       let mut probas = Array2::zeros((n_samples, n_classes));

       for tree in &self.estimators {
           let tree_proba = tree.predict_proba(x)?;
           probas = probas + tree_proba;
       }

       // Normalize
       for i in 0..n_samples {
           let sum: f64 = probas.row(i).sum();
           if sum > 0.0 {
               for j in 0..n_classes {
                   probas[[i, j]] /= sum;
               }
           }
       }

       Ok(probas)
   }

   // AFTER:
   pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
       check_is_fitted(self.fitted)?;
       validate_predict_input(x, self.n_features)?;

       let n_samples = x.nrows();
       let n_classes = self.classes.len();

       #[cfg(feature = "parallel")]
       let tree_probas: Vec<Array2<f64>> = {
           use rayon::prelude::*;
           self.estimators
               .par_iter()
               .map(|tree| tree.predict_proba(x).unwrap())
               .collect()
       };

       #[cfg(not(feature = "parallel"))]
       let tree_probas: Vec<Array2<f64>> = {
           self.estimators
               .iter()
               .map(|tree| tree.predict_proba(x).unwrap())
               .collect()
       };

       // Sum all tree predictions
       let mut probas = Array2::zeros((n_samples, n_classes));
       for tree_proba in tree_probas {
           probas = probas + tree_proba;
       }

       // Normalize
       for i in 0..n_samples {
           let sum: f64 = probas.row(i).sum();
           if sum > 0.0 {
               for j in 0..n_classes {
                   probas[[i, j]] /= sum;
               }
           }
       }

       Ok(probas)
   }
   ```

2. **File**: `ferroml-core/src/models/forest.rs` (MODIFY `predict_proba` for RandomForestRegressor, similar pattern)

   Apply same parallel pattern to lines 848-880.

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --features parallel forest`
- [ ] Manual: Prediction benchmark shows n_estimators speedup

---

### Phase P2.4: Parallel Tree Sample Prediction
**Overview**: Evaluate samples in parallel within single tree prediction

**Changes Required**:
1. **File**: `ferroml-core/src/models/tree.rs` (MODIFY `predict_proba`, lines 545-593)

   ```rust
   // BEFORE: Sequential sample loop
   // AFTER: Parallel sample evaluation

   pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
       check_is_fitted(self.fitted)?;
       validate_predict_input(x, self.n_features)?;

       let n_samples = x.nrows();
       let n_classes = self.classes.len();

       #[cfg(feature = "parallel")]
       let predictions: Vec<Vec<f64>> = {
           use rayon::prelude::*;
           (0..n_samples)
               .into_par_iter()
               .map(|i| {
                   let sample: Vec<f64> = x.row(i).to_vec();
                   let leaf_id = self.find_leaf(&sample);
                   let leaf_node = self.tree.get_node(leaf_id).unwrap();

                   // Extract class probabilities
                   let mut probs = vec![0.0; n_classes];
                   if let Some(ref value) = leaf_node.value {
                       for (class_idx, &count) in value.iter().enumerate() {
                           if class_idx < n_classes {
                               probs[class_idx] = count;
                           }
                       }
                       let total: f64 = probs.iter().sum();
                       if total > 0.0 {
                           for p in &mut probs {
                               *p /= total;
                           }
                       }
                   }
                   probs
               })
               .collect()
       };

       #[cfg(not(feature = "parallel"))]
       let predictions: Vec<Vec<f64>> = {
           // Original sequential code
           // ...
       };

       // Convert to Array2
       let mut probas = Array2::zeros((n_samples, n_classes));
       for (i, probs) in predictions.iter().enumerate() {
           for (j, &p) in probs.iter().enumerate() {
               probas[[i, j]] = p;
           }
       }

       Ok(probas)
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --features parallel tree`
- [ ] Manual: Single tree prediction benchmark shows speedup on large datasets

---

### Phase P2.5: Benchmark Suite
**Overview**: Create comprehensive benchmarks to verify optimizations

**Changes Required**:
1. **File**: `ferroml-core/benches/performance_optimizations.rs` (NEW)

   ```rust
   //! Benchmarks for P2 performance optimizations

   use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
   use ndarray::{Array1, Array2};
   use ferroml_core::models::{
       HistGradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier,
       DecisionTreeRegressor, Model,
   };

   fn generate_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
       let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| (i + j) as f64 / 100.0);
       let y = Array1::from_shape_fn(n_samples, |i| (i as f64 / 10.0).sin());
       (x, y)
   }

   fn bench_hist_gradient_boosting(c: &mut Criterion) {
       let mut group = c.benchmark_group("HistGradientBoosting");

       for n_samples in [1000, 5000, 10000] {
           let (x, y) = generate_data(n_samples, 10);

           group.bench_with_input(
               BenchmarkId::new("fit", n_samples),
               &(&x, &y),
               |b, (x, y)| {
                   b.iter(|| {
                       let mut model = HistGradientBoostingRegressor::new()
                           .with_max_iter(10)
                           .with_max_depth(4);
                       model.fit(black_box(x), black_box(y)).unwrap()
                   })
               },
           );
       }

       group.finish();
   }

   fn bench_random_forest_prediction(c: &mut Criterion) {
       let mut group = c.benchmark_group("RandomForest_Prediction");

       let (x_train, y_train) = generate_data(1000, 10);

       for n_estimators in [10, 50, 100] {
           let mut model = RandomForestRegressor::new()
               .with_n_estimators(n_estimators)
               .with_max_depth(5)
               .with_random_state(42);
           model.fit(&x_train, &y_train).unwrap();

           let (x_test, _) = generate_data(1000, 10);

           group.bench_with_input(
               BenchmarkId::new("predict", n_estimators),
               &(&model, &x_test),
               |b, (model, x)| {
                   b.iter(|| model.predict(black_box(x)).unwrap())
               },
           );
       }

       group.finish();
   }

   fn bench_tree_prediction_scaling(c: &mut Criterion) {
       let mut group = c.benchmark_group("DecisionTree_Prediction");

       let (x_train, y_train) = generate_data(5000, 10);
       let mut model = DecisionTreeRegressor::new()
           .with_max_depth(10)
           .with_random_state(42);
       model.fit(&x_train, &y_train).unwrap();

       for n_samples in [100, 1000, 10000] {
           let (x_test, _) = generate_data(n_samples, 10);

           group.bench_with_input(
               BenchmarkId::new("predict", n_samples),
               &(&model, &x_test),
               |b, (model, x)| {
                   b.iter(|| model.predict(black_box(x)).unwrap())
               },
           );
       }

       group.finish();
   }

   criterion_group!(
       benches,
       bench_hist_gradient_boosting,
       bench_random_forest_prediction,
       bench_tree_prediction_scaling,
   );
   criterion_main!(benches);
   ```

2. **File**: `ferroml-core/Cargo.toml` (ADD benchmark)

   ```toml
   [[bench]]
   name = "performance_optimizations"
   harness = false
   ```

**Success Criteria**:
- [ ] Automated: `cargo bench -p ferroml-core --bench performance_optimizations`
- [ ] Manual: Compare results with/without parallel/simd features

---

## Dependencies
- `rayon` (already in dependencies for parallel feature)
- `wide` (already in dependencies for simd feature)
- `criterion` (already in dev-dependencies)

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Parallel overhead for small datasets | Keep sequential fallback, use threshold |
| SIMD not available on all platforms | Feature-gated, graceful fallback |
| Race conditions | Use thread-local aggregation, no shared mutable state |
| Results differ with parallel | Test determinism with fixed seed |

## Verification Commands
```bash
# Run all tests with features
cargo test -p ferroml-core --features "parallel simd"

# Run benchmarks
cargo bench -p ferroml-core --bench performance_optimizations

# Compare with/without parallel
cargo bench -p ferroml-core --bench performance_optimizations
cargo bench -p ferroml-core --bench performance_optimizations --no-default-features
```

## Expected Improvements
| Optimization | Target Speedup | Applies To |
|--------------|----------------|------------|
| SIMD histogram subtraction | 1.5-2x | HistGradientBoosting train |
| Parallel histogram building | 2-4x | HistGradientBoosting train |
| Parallel forest prediction | Nx (N=n_estimators) | RandomForest predict |
| Parallel tree sample prediction | Mx (M=cores) | DecisionTree predict on large batches |

# Plan 6: Advanced Features Completion

**Date:** 2026-02-08
**Reference:** `thoughts/shared/research/2026-02-08_ferroml-comprehensive-assessment.md`
**Priority:** Medium-Low
**Estimated Tasks:** 8

## Objective

Complete partially-implemented advanced features: Bootstrap BCa, GPU support, streaming serialization.

## Context

From research:
- Bootstrap BCa CI: Structure defined but not fully computed
- GPU feature flag: Defined but unused
- Streaming serialization: Mentioned but not implemented
- Probit activation: Not implemented in inference

## Tasks

### Task 6.1: Complete Bootstrap BCa implementation
**File:** `ferroml-core/src/stats/bootstrap.rs`
**Description:**
- Implement acceleration constant (a) calculation
- Implement bias-correction constant (z0) calculation
- BCa percentiles: α1 = Φ(z0 + (z0 + zα)/(1 - a(z0 + zα)))
**Lines:** ~50 additions
**Reference:** Efron & Tibshirani (1993)

### Task 6.2: Add BCa tests
**File:** `ferroml-core/src/stats/bootstrap.rs` (tests)
**Description:**
- Compare BCa vs percentile CI on skewed distributions
- Verify BCa is narrower for skewed data
- Test edge cases (symmetric data, small samples)
**Tests:** ~5

### Task 6.3: Design GPU acceleration architecture
**File:** `ferroml-core/src/gpu/mod.rs` (new)
**Description:**
- Define GPU backend trait (supports CUDA, Metal, Vulkan)
- Identify operations to accelerate: matmul, tree ensemble inference
- Design fallback to CPU when GPU unavailable
**Lines:** ~100 (interface only)

### Task 6.4: Implement GPU matmul (optional CUDA)
**File:** `ferroml-core/src/gpu/cuda.rs` (new)
**Description:**
- CUDA kernel for matrix multiplication
- Integration with linear model training
- Benchmark vs CPU implementation
**Dependencies:** cuda-sys or cudarc crate
**Lines:** ~200

### Task 6.5: Implement streaming serialization
**File:** `ferroml-core/src/serialization.rs`
**Description:**
- `StreamingWriter` for large model serialization
- `StreamingReader` for loading without full memory
- Chunk-based processing with progress callbacks
**Lines:** ~150

### Task 6.6: Add Probit activation to inference
**File:** `ferroml-core/src/inference/operators.rs`
**Line:** 756-758
**Description:**
- Implement Probit post-transform: Φ^(-1)(x)
- Use inverse normal CDF from stats module
**Lines:** ~20

### Task 6.7: Add LoadOptions/SaveOptions API
**File:** `ferroml-core/src/serialization.rs`
**Description:**
- `LoadOptions`: verify_checksum, allow_version_mismatch, progress_callback
- `SaveOptions`: compression_level, include_metadata, progress_callback
**Lines:** ~100

### Task 6.8: GPU and streaming tests
**Files:** Various test modules
**Description:**
- GPU fallback to CPU verification
- Streaming serialization round-trip
- Large model serialization (>100MB synthetic)
**Tests:** ~10

## Success Criteria

- [ ] BCa CI produces tighter intervals for skewed data
- [ ] GPU architecture defined (implementation optional)
- [ ] Streaming serialization works for models >1GB
- [ ] Probit activation produces correct values
- [ ] LoadOptions/SaveOptions provide user control

## Priority Notes

**Implement First:**
1. Task 6.1-6.2 (BCa) - Completes existing feature
2. Task 6.6 (Probit) - Small fix, completes inference

**Implement If Time:**
3. Task 6.7 (Options API) - User convenience
4. Task 6.5 (Streaming) - For very large models

**Defer:**
5. Task 6.3-6.4 (GPU) - Significant complexity, optional

## Dependencies

### For GPU (if implemented)
- `cudarc` or `cuda-sys` for CUDA
- `metal` crate for Apple Silicon
- Feature-gated compilation

### For Streaming
- No new dependencies (use std::io)

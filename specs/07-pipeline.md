# Specification: Pipeline

**Status**: 🔲 NOT IMPLEMENTED

## Overview

DAG-based pipeline execution with caching and combined search spaces.

## Requirements

### Pipeline Structure
- [ ] Sequential steps (transform → transform → model)
- [ ] Named steps for access
- [ ] Combined search space from all steps
- [ ] Caching of intermediate results

### Feature Union
- [ ] Parallel feature branches
- [ ] Concatenation of transformed features

### Pipeline Operations

```rust
pub struct Pipeline {
    pub steps: Vec<(String, PipelineStep)>,
    pub cache: Option<PipelineCache>,
}

pub enum PipelineStep {
    Transform(Box<dyn Transformer>),
    Model(Box<dyn Model>),
}

impl Pipeline {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>;
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
    fn fit_transform(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>>;
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>;
    fn search_space(&self) -> SearchSpace;  // Combined from all steps
    fn set_params(&mut self, params: &HashMap<String, ParameterValue>) -> Result<()>;
}
```

### Caching
- [ ] Memory cache for intermediate results
- [ ] Hash-based cache invalidation
- [ ] Optional disk persistence

## Implementation Priority

1. Basic Pipeline (sequential steps)
2. Combined search space
3. Caching
4. FeatureUnion

## Acceptance Criteria

- [ ] Pipeline.search_space() correctly combines step parameters
- [ ] Caching reduces redundant computation
- [ ] set_params works with nested parameter names (e.g., "scaler__with_mean")

## Implementation Location

`ferroml-core/src/pipeline/`

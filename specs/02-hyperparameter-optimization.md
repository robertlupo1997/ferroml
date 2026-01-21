# Specification: Hyperparameter Optimization

**Status**: ✅ IMPLEMENTED (core), 🔲 PARTIAL (Bayesian GP)

## Overview

HPO module for finding optimal hyperparameters with statistical rigor.

## Requirements

### Search Space Definition
- [x] Integer parameters (with log scale option)
- [x] Float parameters (with log scale option)
- [x] Categorical parameters
- [x] Boolean parameters
- [x] Preset search spaces for common models

### Samplers
- [x] Random sampler
- [x] Grid sampler
- [x] TPE (Tree-Parzen Estimator) sampler
- [x] Gaussian Process-based Bayesian optimizer (RBF, Matern52, Matern32 kernels)

### Schedulers/Pruners
- [x] Median pruner
- [x] Hyperband scheduler
- [x] ASHA (Asynchronous Successive Halving)

### Study Management
- [x] Trial tracking (params, values, state)
- [x] Best trial selection (minimize/maximize)
- [x] Intermediate value reporting for pruning
- [x] Parameter importance calculation

## Missing (Phase 8)

### Advanced Optimization
- [x] Gaussian Process regression (implemented)
- [x] Kernel selection (RBF, Matern52, Matern32)
- [x] Acquisition functions (EI, PI, UCB, LCB)
- [x] Acquisition optimization (L-BFGS-B gradient-based)

### Multi-Fidelity Optimization
- [x] Enhanced Hyperband with proper bracket management
- [x] FidelityParameter support (discrete/continuous)
- [x] EarlyStoppingCallback trait
- [x] Performance metrics collection (RungMetrics, BracketMetrics, HyperbandMetrics)
- [ ] BOHB (Bayesian Optimization + Hyperband)

## Implementation Location

`ferroml-core/src/hpo/`

## Validation

```bash
cargo test -p ferroml-core hpo
```

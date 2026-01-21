# Specification: Statistical Foundations

**Status**: ✅ IMPLEMENTED

## Overview

FerroML's statistical module provides rigorous statistical tools that go beyond typical AutoML offerings.

## Requirements

### Hypothesis Testing
- [x] T-tests (one-sample, two-sample, paired)
- [x] Mann-Whitney U test (non-parametric)
- [x] Effect sizes with every test (Cohen's d, Hedges' g, Glass's delta)

### Confidence Intervals
- [x] Normal approximation CI
- [x] Student's t CI
- [x] Bootstrap CI (percentile, BCa)

### Multiple Testing Correction
- [x] Bonferroni (FWER control)
- [x] Holm-Bonferroni step-down
- [x] Hochberg step-up
- [x] Benjamini-Hochberg (FDR control)
- [x] Benjamini-Yekutieli (FDR under dependency)

### Diagnostics
- [x] Descriptive statistics (mean, std, median, IQR, skewness, kurtosis)
- [x] Correlation with CI
- [x] Residual diagnostics

### Power Analysis
- [x] Sample size calculation
- [x] Power calculation given effect size

## Implementation Location

`ferroml-core/src/stats/`

## Validation

```bash
cargo test -p ferroml-core stats
```

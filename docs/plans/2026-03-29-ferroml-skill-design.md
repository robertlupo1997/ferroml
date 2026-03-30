# FerroML Skill Design — The Definitive ML Skill for Claude Code

**Status:** COMPLETE (2026-03-29)
**Location:** `ferroml-python/skills/ferroml-ml/`

## Purpose

Make FerroML + Claude Code the easiest way for anyone — non-technical or expert — to do machine learning. Claude Code becomes an expert ML engineer that uses FerroML's Python API directly.

## Target Users

- **Non-technical:** "I have a CSV of sales data, help me predict next month's revenue"
- **Developers new to ML:** "Fit a model and tell me which features matter"
- **ML engineers:** "Compare gradient boosting variants with statistical significance"

The Skill adapts language and depth based on detected technical level.

## Skill Structure

```
ferroml-ml/
  SKILL.md                              # Core: triggers, decision logic, workflows
  references/
    api-cheatsheet.md                   # Full API (55+ models, every method)
    model-picker.md                     # Which model for which problem
    diagnostics-interpreter.md          # How to interpret every diagnostic
    common-pitfalls.md                  # Gotchas (class imbalance, leakage, overfitting)
    feature-engineering-guide.md        # When/how to transform features
    deployment-guide.md                 # ONNX → FastAPI → production
    drift-monitoring-guide.md           # Post-deployment monitoring
    fairness-guide.md                   # Bias/fairness auditing
    experiment-tracking-guide.md        # Logging and comparing experiments
  scripts/
    # --- Phase 1: Data Understanding ---
    explore_data.py                     # Profile dataset: shape, types, distributions, correlations
    data_quality_audit.py               # Nulls, dupes, types, cardinality, missingness patterns
    detect_leakage.py                   # Flag features suspiciously correlated with target
    # --- Phase 2: Feature Engineering ---
    feature_engineer.py                 # Polynomial, interactions, datetime, encoding
    feature_select.py                   # RFE + mutual info + VIF, unified recommendations
    # --- Phase 3: Modeling ---
    full_pipeline.py                    # End-to-end: load → preprocess → recommend → train → evaluate → diagnose
    compare_models.py                   # Train N models, statistical comparison, leaderboard
    cv_strategy_advisor.py              # Auto-pick k-fold vs stratified vs time-series split
    tune_hyperparams.py                 # Orchestrate search_space() + HPO with budget
    optimize_threshold.py               # Find optimal classification cutoff
    learning_curves.py                  # Plot + interpret bias vs variance
    calibrate_probabilities.py          # Check calibration, apply Platt/isotonic fix
    # --- Phase 4: Analysis ---
    error_analysis.py                   # Segment errors by feature, find failure modes
    generate_report.py                  # Plain-language summary of results
    visualize.py                        # Residual plots, feature importance, confusion matrix, learning curves
    fairness_audit.py                   # Demographic parity, equalized odds across groups
    # --- Phase 5: Production ---
    diagnose_failure.py                 # Model performing badly? Diagnose why + fix
    deploy_model.py                     # Export → inference code → API endpoint
    detect_drift.py                     # Compare prod vs training distributions (KS, PSI)
    reproducibility_snapshot.py         # Capture seeds, versions, data hash, git commit
    ab_test.py                          # Power calc + significance testing workflow
  assets/
    configs/
      binary_classification.json        # Pre-built pipeline config
      multiclass.json
      regression.json
      clustering.json
      anomaly_detection.json
      time_series.json
    templates/
      report_template.md                # Fill-in-the-blank model report
      notebook_template.ipynb           # Jupyter notebook starter
      experiment_log.json               # Structured experiment tracking format
```

## SKILL.md Workflows

### Workflow 1: End-to-End ML (the main flow)
Trigger: "I have data, help me build a model"
Steps: explore_data → data_quality_audit → detect_leakage → feature_engineer → feature_select → recommend → full_pipeline → error_analysis → generate_report

### Workflow 2: Model Selection
Trigger: "Which model should I use?"
Steps: explore_data → recommend → model-picker reference → compare_models

### Workflow 3: Evaluation & Diagnostics
Trigger: "How good is my model?"
Steps: evaluate → diagnose → error_analysis → calibrate_probabilities → learning_curves → generate_report

### Workflow 4: Production Readiness
Trigger: "Deploy this model" / "Is my model still working?"
Steps: reproducibility_snapshot → deploy_model → detect_drift

### Workflow 5: Experimentation
Trigger: "A/B test" / "Compare experiments"
Steps: ab_test → experiment tracking

## Design Principles

1. **Python API, not CLI** — Claude Code imports ferroml directly
2. **Adapt to technical level** — Detect from conversation, adjust language
3. **Always show diagnostics** — FerroML's differentiator
4. **Scripts are templates** — Claude adapts them to the user's data
5. **Progressive disclosure** — SKILL.md has workflow logic, references have details
6. **Fail forward** — When something goes wrong, diagnose_failure.py explains why and fixes it

## Implementation Order

1. SKILL.md (core decision logic and workflows)
2. references/ (9 docs — API surface, model picking, diagnostics interpretation)
3. scripts/ Phase 1 (data understanding: 3 scripts)
4. scripts/ Phase 2 (feature engineering: 2 scripts)
5. scripts/ Phase 3 (modeling: 6 scripts)
6. scripts/ Phase 4 (analysis: 4 scripts)
7. scripts/ Phase 5 (production: 5 scripts)
8. assets/ (configs + templates)

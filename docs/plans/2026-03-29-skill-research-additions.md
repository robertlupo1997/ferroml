# FerroML Skill — Research Additions

Ongoing research findings for additions to the skill design. Each section timestamped.

---

## Research Iteration 1 — 2026-03-29 ~00:07 ET

### New Scripts to Add

**1. `scripts/explain_model.py` — Model Interpretability/Explainability**
- Permutation feature importance (model-agnostic)
- Partial dependence plots (PDP) for top features
- Individual Conditional Expectation (ICE) plots
- Feature interaction detection via H-statistic
- Plain-language: "Feature 'square_footage' has the biggest impact on price. Each additional 100 sqft adds ~$15K."
- *Why missing:* Current design has feature_importance in visualize.py but not dedicated interpretability. This is the #1 request from stakeholders ("why did the model decide this?")

**2. `scripts/handle_imbalance.py` — Class Imbalance Treatment**
- Detect imbalance ratio automatically
- Apply strategies: class_weight='balanced', SMOTE-like oversampling, random undersampling, cost-sensitive threshold
- Before/after comparison on minority class recall
- *Why missing:* common-pitfalls.md mentions imbalance but no dedicated treatment script. This is the #1 classification problem in practice.

**3. `scripts/validate_assumptions.py` — Statistical Assumption Checking**
- Linearity check (residual vs fitted plots)
- Homoscedasticity test (Breusch-Pagan)
- Independence check (Durbin-Watson — already in diagnose but not automated recommendation)
- Normality of residuals (Shapiro-Wilk, Q-Q plot)
- Multicollinearity (VIF for all features, condition number)
- Output: "Your data violates the homoscedasticity assumption. Consider using RobustRegression or transforming the target."
- *Why missing:* diagnostics-interpreter.md explains results but this script proactively checks ALL assumptions before the user even asks

**4. `scripts/sample_data.py` — Smart Sampling for Large Datasets**
- Stratified sampling that preserves class distribution
- Cluster-based sampling for representative subsets
- Time-aware sampling (preserve temporal ordering)
- Power analysis: "You have 1M rows but only need ~5K for 95% confidence at this effect size"
- *Why missing:* Large datasets are common; users need guidance on when/how to subsample without losing signal

**5. `scripts/cost_sensitive_analysis.py` — Business Cost Modeling**
- Define cost matrix: "A false negative costs $1000, a false positive costs $50"
- Find optimal threshold given asymmetric costs
- Expected cost per prediction
- ROI calculation: "This model saves $X per 1000 predictions vs baseline"
- *Why missing:* Bridges the gap between "ML metrics" and "business value" — critical for non-technical stakeholders

### New References to Add

**6. `references/interpretability-guide.md` — How to Explain Models to Stakeholders**
- When to use which explanation method (global vs local, model-specific vs model-agnostic)
- How to present results to non-technical audiences
- Regulatory requirements for model explanations (EU AI Act, ECOA)
- Template phrases for common explanations

**7. `references/data-types-guide.md` — Handling Different Data Types**
- Categorical (ordinal vs nominal, high-cardinality strategies)
- Text columns (when to use CountVectorizer vs TfidfVectorizer)
- Datetime (extraction patterns, cyclical encoding for hour/day-of-week)
- Geographic (lat/lon clustering, distance features)
- Currency/monetary (log transform, inflation adjustment)
- Mixed-type datasets (which preprocessing for which column type)

**8. `references/troubleshooting-guide.md` — Common Errors and Fixes**
- "ConvergenceFailure" → increase max_iter, scale features, reduce learning rate
- "ShapeMismatch" → check train/test column alignment
- "InvalidInput" → check for NaN, inf, wrong dtypes
- Memory errors → subsample, use MiniBatchKMeans, reduce n_estimators
- Slow training → use Hist variants, reduce features, subsample
- Every FerroError variant with concrete fix steps

### New Asset Configs to Add

**9. `assets/configs/cost_sensitive_classification.json`**
- Pipeline config that includes cost matrix, threshold optimization, business-metric evaluation

**10. `assets/configs/high_cardinality.json`**
- Pipeline config for datasets with many categorical features (>100 unique values)
- Target encoding, frequency encoding, hashing strategies

### New Workflow to Add

**Workflow 6: "Explain This to My Boss"**
- Trigger: "explain", "why did the model", "present results", "stakeholder"
- Steps: explain_model → cost_sensitive_analysis → generate_report (with report_template.md)
- Output: Non-technical summary with business impact numbers

---

## Research Iteration 2 — 2026-03-29 ~02:07 ET

### New Scripts

**1. `scripts/predict_with_confidence.py` — Prediction Intervals**
- Point predictions + confidence/prediction intervals
- Use bootstrap_ci for any model, GP native intervals for GaussianProcess
- Output: "Predicted price: $350K (95% CI: $310K–$390K)"
- *Why:* Stakeholders need uncertainty, not just point estimates. No other skill does this.

**2. `scripts/counterfactual_explain.py` — "What Would Need to Change?"**
- Given a prediction, find minimum feature changes to flip the outcome
- "To get approved for the loan, income would need to increase from $45K to $52K OR debt ratio decrease from 0.4 to 0.3"
- Useful for actionable insights in classification
- *Why:* Emerging explainability technique (2024-2026), increasingly required by EU AI Act

**3. `scripts/progressive_complexity.py` — Start Simple, Add Complexity Only When Justified**
- Train baseline (Linear/Logistic) → compare with medium (RF) → compare with complex (GBT)
- At each step: is the improvement statistically significant?
- Stop when added complexity doesn't justify the accuracy gain
- "Linear model gets R2=0.91, Random Forest gets 0.93 (p=0.34, not significant). Stick with the simpler model."
- *Why:* Kaggle grandmaster technique — always establish baselines. Prevents unnecessary complexity.

**4. `scripts/auto_ensemble.py` — Automated Diverse Ensemble Construction**
- Select top N diverse models (low correlation between predictions)
- Optimize blending weights via hill climbing on validation set
- Compare ensemble vs best single model with significance test
- *Why:* Last-mile improvement technique from competitive ML. FerroML has Voting/Stacking but no automated diversity-aware construction.

### New References

**5. `references/batch-vs-realtime-guide.md` — Inference Patterns**
- When to use batch (nightly scoring of all customers) vs real-time (API per request)
- Memory/latency tradeoffs per model type
- ONNX optimization tips for real-time
- Caching strategies for feature computation

**6. `references/model-lifecycle-guide.md` — From Experiment to Retirement**
- Experiment → validate → deploy → monitor → retrain → retire
- Decision framework for when to retrain vs rebuild
- Model governance checklist (who approved, what data, what metrics)
- Audit trail requirements

### New Asset

**7. `assets/configs/multiclass.json`**
- Same structure as binary but with: macro/weighted F1, one-vs-rest strategy notes, confusion matrix emphasis, no threshold optimization (multiclass doesn't use 0.5 cutoff the same way)

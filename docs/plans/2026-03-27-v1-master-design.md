# FerroML v1.0 Master Design Document

> **For Claude:** This is the multi-session master plan for FerroML v1.0. Each phase has its own detailed plan created at session start. Reference this document for vision, priorities, and phase sequencing. Use superpowers:executing-plans for individual phase plans.

**Goal:** Ship FerroML v1.0 as the first correctness-certified, AI-agent-first ML library on PyPI.

**Architecture:** Rust core (ferroml-core) with PyO3 Python bindings (ferroml-python), MCP server for AI agent integration, trusted publishing to PyPI via GitHub Actions + maturin.

**Tech Stack:** Rust (ndarray, nalgebra, rayon), Python (PyO3, maturin), MCP (Python SDK), GitHub Actions (CI/CD)

---

## Vision

FerroML v1.0 makes three claims no other library makes simultaneously:

1. **Correctness-certified** -- Every algorithm verified against textbook formulations, reference implementations (sklearn/scipy/statsmodels/linfa), property tests, and adversarial inputs. A public correctness report accompanies the release.
2. **Statistical diagnostics first-class** -- Residual analysis, assumption tests, confidence intervals built into every model.
3. **AI-native** -- Official MCP server, AGENTS.md, llms.txt, typed returns, and recommend() API so AI agents can autonomously discover and execute ML workflows.

## Priorities (ordered)

1. **Correctness** -- Every shipped model passes all 4 verification layers
2. **Robustness** -- No panics on any input (unwrap() elimination)
3. **PyPI release** -- `pip install ferroml` with cross-platform wheels
4. **AI-agent-first** -- MCP server, AGENTS.md, llms.txt
5. **Performance** -- Fix regressions that exceed 5x vs sklearn
6. **Growth** -- SEO, docs site, launch strategy

## Current State (as of 2026-03-27)

### What's solid
- 55+ models fully implemented and exposed to Python (no binding gaps)
- 5,650+ tests passing (3,550 Rust + 2,100 Python)
- Cross-library validation against sklearn/scipy/statsmodels/linfa/xgboost/lightgbm
- AutoML with meta-learning, warm-starting, bandit budget allocation
- Minimal unsafe code (26 occurrences in 3 files)
- ONNX export (118 tests passing)

### Critical issues (must fix for v1.0)
- **230+ unwrap() in production model code** -- crash risk on edge inputs
  - Worst: sgd.rs (35), gaussian_process.rs (34), svm.rs (25), knn.rs (18)
- **Panics on empty data** -- multiple models crash instead of returning FerroError
- **RidgeCV predict returns NaN** -- silent incorrect results
- **No parameter validation at construction** -- invalid configs accepted silently
- **6 pre-existing test failures** -- TemperatureScaling, IncrementalPCA vs sklearn

### Medium issues (should fix for v1.0)
- RandomForest non-determinism in parallel training
- MLP serialization broken
- No __sklearn_tags__ support (limits sklearn pipeline interop)
- Performance regressions: PCA 2.29x, KMeans 6.84x, SVC 7.70x vs sklearn

### Models shipping in v1.0 (pending correctness audit)
All 55+ models are candidates. Models that fail the 4-layer correctness audit get removed from the Python API rather than shipped broken. "53 correct models" beats "55 models with known issues."

## Correctness Certification: 4 Layers

### Layer 1: Reference-Match (largely done)
- Output matches sklearn/scipy/statsmodels within documented tolerances
- 200+ cross-library tests already exist
- Gap: tighten tolerances, add missing comparisons

### Layer 2: Textbook-Verified (the big gap)
For each algorithm, verify implementation matches the canonical mathematical formulation:
- **Linear Models**: Gauss-Markov theorem, residual orthogonality, hat matrix properties (ESL Ch. 3)
- **Logistic Regression**: Stable sigmoid, separation detection, Fisher information (Bishop Ch. 4)
- **SVM**: KKT conditions, SMO correctness, kernel PSD (Platt 1998, Chang & Lin 2011)
- **Trees/RF/GBT**: Gini/entropy formulas, feature importance normalization, XGBoost gain formula (Breiman 2001, Friedman 2001, Chen & Guestrin 2016)
- **K-Means/DBSCAN/HDBSCAN**: Monotonic decrease, density-reachability transitivity (Lloyd 1982, Ester et al. 1996)
- **Naive Bayes**: Log-space computation, Laplace smoothing, Welford's variance (Bishop Ch. 4)
- **PCA/t-SNE**: SVD-based (not eigendecomposition of X^TX), orthogonality, explained variance (van der Maaten & Hinton 2008)
- **GP**: Cholesky-based solve, jitter for PSD, posterior variance non-negative (Rasmussen & Williams 2006)
- **GMM**: EM monotonic likelihood, singularity detection, responsibility normalization (Bishop Ch. 9)
- **Isolation Forest/LOF**: Score bounds, c(n) formula, reachability distance (Liu et al. 2012)

### Layer 3: Property/Invariant Tests
- Predicted probabilities sum to 1 (all classifiers)
- Loss monotonically decreases (K-Means, EM, gradient descent)
- PCA components orthogonal, eigenvalues sorted descending
- Feature importances non-negative, sum to 1 (trees)
- KKT conditions satisfied within tolerance (SVM)
- Anomaly scores in [0, 1] (Isolation Forest)
- Ridge with lambda=0 equals OLS
- Ridge with lambda->inf: coefficients approach zero

### Layer 4: Adversarial/Edge Case Tests
- Near-collinear features (condition number > 1e12)
- Near-separation in logistic regression
- Very large/small values (overflow/underflow triggers)
- Empty input, single sample, constant features, all-same labels
- Float precision edge cases
- Degenerate matrices (rank-deficient)

### Layer 5: Frankenstein Tests (Composition/Integration)
Addresses the "Frankenstein effect" documented in NVIDIA's VibeTensor (arXiv:2601.16238):
locally correct components that compose into globally incorrect systems.

**Pipeline composition correctness:**
- Pipeline(StandardScaler, PCA, SVC) end-to-end vs sklearn Pipeline -- exact match
- Pipeline(MinMaxScaler, LogisticRegression) -- scaler state does not corrupt model
- ColumnTransformer(numeric=[Scaler], categorical=[OneHot]) + Classifier
- Nested: CV(Pipeline(Scaler, RF)) -- cross-validation wrapping composed pipeline

**Stateful interaction bugs:**
- Fit model, predict, fit AGAIN with different data, predict -- no state leakage
- Fit model A, clone, fit clone with different data -- original unchanged
- Pipeline fit/predict/fit/predict cycle -- second fit completely replaces first
- AutoML.fit() called twice -- second run independent of first

**Ensemble composition:**
- Stacking(RF, GBT, SVC) -- meta-learner receives correct base predictions
- Voting(model_a, model_b) -- weights applied correctly to composed predictions
- Bagging(DecisionTree) -- bootstrap samples independent, aggregation correct

**AutoML end-to-end:**
- Raw CSV in, AutoML selects algorithm + preprocessing + hyperparams, predictions out
- Compare AutoML result vs manually running the same pipeline -- must match
- AutoML with time budget -- verify it respects budget and returns best model found
- AutoML ensemble -- verify ensemble of top-k models outperforms individual

**Serialization under composition:**
- Save fitted Pipeline, load, predict -- same result as before save
- Save AutoML result, load, predict -- same result
- Save Stacking ensemble, load, predict -- same result

**Cross-module performance (Frankenstein perf):**
- Pipeline end-to-end latency vs sum of component latencies (detect overhead)
- AutoML trial throughput -- verify parallelism actually helps
- Repeated predict() calls -- no memory leak or performance degradation

## Numerical Stability Checklist

Every model must handle these correctly:
- [ ] Log-sum-exp trick for softmax/cross-entropy/NB posteriors
- [ ] Welford's algorithm for stable variance (not naive E[X^2] - E[X]^2)
- [ ] SVD for least squares (not normal equations X^TX)
- [ ] Stable sigmoid: branch on sign of x
- [ ] Cholesky jitter for GP/GMM covariance matrices
- [ ] Euclidean distance clamped to non-negative
- [ ] Log-space probability computation (not probability products)
- [ ] Regularization in denominator for gradient boosting leaf weights
- [ ] Empty cluster handling in K-Means

## Phase Plan

### Phase 1: Correctness Audit (Sessions 1-3) — COMPLETED in 1 session
**Goal:** Know exactly where we stand.
- ✅ Textbook-verify every algorithm against references (40 algorithms audited)
- ✅ Produce per-algorithm correctness checklist (pass/fail per invariant)
- ✅ Triage: 36 PASS / 3 FIX / 0 REMOVE
- ✅ GPU/Sparse assessment: GPU=EXPERIMENTAL, Sparse=STABLE
- ✅ Deliverable: `docs/correctness-report.md`
- **Findings:** 12 bugs (1 P0, 3 P1, 5 P2, 3 P3). Worst unwrap() files: tree.rs (92), hist_boosting.rs (92), boosting.rs (56).
- **Sessions saved: 2** — reallocated to Phase 2/3.

### Phase 2: Bug Fixes + Robustness Hardening (Sessions 2-5)
**Goal:** Fix all audit bugs. No panics, no silent failures.

**Session 2: P0/P1 Bug Fixes (the 4 correctness bugs)**
- Fix P0: Implement Entropy criterion for DecisionTreeClassifier (tree.rs)
- Fix P1: LogisticRegression IRLS weight clamping (logistic.rs:638)
- Fix P1: AdaBoost regressor weight formula .abs() -> negation (adaboost.rs:524)
- Fix P1: AdaBoost docstring SAMME.R -> SAMME (adaboost.rs:7)
- Add regression tests for each fix
- Run full test suite to verify no breakage

**Session 3: P2 Bug Fixes + Critical unwrap() Elimination**
- Fix P2: z_inv_normal sign error (regularized.rs:2364)
- Fix P2: SGDClassifier NaN/Inf divergence check (sgd.rs)
- Fix P2: ExtraTrees expect() -> filter_map in parallel (extra_trees.rs:404,747)
- Fix P2: t-SNE Barnes-Hut validate n_components <= 2 or fall back to exact (tsne.rs)
- Fix P2: GP Regressor Cholesky jitter retry (gaussian_process.rs)
- Eliminate unwrap() in worst files: tree.rs (92), hist_boosting.rs (92), boosting.rs (56)
- Fix P3 bugs if time permits

**Session 4: Remaining unwrap() Elimination + Parameter Validation**
- Eliminate unwrap() in: sgd.rs (35), adaboost.rs (32), svm.rs (25), knn.rs (19), forest.rs (40), extra_trees.rs (34)
- Eliminate unwrap() in gpu/backend.rs (13 — panic on GPU errors)
- Add parameter validation at construction time for all models
- Fix empty data panics
- Fix or remove TemperatureScaling/IncrementalPCA (6 pre-existing test failures)

**Session 5: Property Tests (Layer 3) + Adversarial Tests (Layer 4)**
- Add property tests: probabilities sum to 1, loss decreases, components orthogonal, importances sum to 1, etc.
- Add adversarial tests: near-collinear, near-separation, overflow/underflow, empty/single/constant
- Verify all 12 fixes still pass
- Update correctness report with test results

### Phase 3: Frankenstein Tests + Final Validation (Sessions 6-8)
**Goal:** Every shipped model passes all 5 layers including composition.
- **Build Layer 5 Frankenstein test suite** (pipeline composition, stateful interactions, AutoML end-to-end, serialization under composition, cross-module perf)
- Fix RandomForest non-determinism
- Fix MLP serialization
- Fix performance regressions exceeding 5x (SVC 7.7x, KMeans 6.84x)
- Thread safety verification (concurrent predict() from Python threads)
- Final cross-library validation sweep
- Update correctness report with all 5 layers

### Phase 4: Python API and PyPI Release (Sessions 9-11)
**Goal:** `pip install ferroml` works, API is complete and documented.

**Accounts needed (manual, before this phase):**
- PyPI account: https://pypi.org/account/register/ (2FA required)
- TestPyPI account: https://test.pypi.org/account/register/ (separate instance)
- Set up "pending trusted publisher" at https://pypi.org/manage/account/publishing/
  - PyPI project name: `ferroml`
  - Owner: `robertlupo1997`
  - Repository: `ferroml`
  - Workflow: `release.yml`
  - Environment: `release`

**Technical work:**
- Full .pyi type stubs with py.typed marker
- Google-style docstrings on every public method with examples
- recommend() API for agent-driven algorithm selection
- Structured ModelCard metadata on every model
- Structured error messages with remediation hints
- GitHub Actions release.yml (maturin wheels: Linux x86_64/aarch64, macOS x86_64/aarch64, Windows x64, musllinux, sdist)
- Test on TestPyPI, then publish v1.0 to PyPI
- README with badges, quick-start, benchmark comparison table

**PyPI metadata (pyproject.toml):**
- Keywords: machine-learning, automl, statistics, data-science, rust, scikit-learn, gradient-boosting, random-forest, svm, regression, classification, clustering, statistical-diagnostics, preprocessing, cross-validation, bayesian
- Classifiers: Development Status 4 Beta, all major OS, Python 3.10-3.13, Rust, Science/Research, AI, Mathematics
- URLs: Homepage, Documentation, Repository, Issues, Changelog

### Phase 5: AI-Agent-First Layer (Sessions 12-14)
**Goal:** AI agents can discover and use FerroML autonomously.

**CLI tool (primary agent interface):**
- `ferroml` CLI using Typer/Click with subcommands: train, predict, evaluate, diagnose, compare, export, automl
- Structured JSON output mode (--json flag) for agent consumption
- CLI costs ~200 tokens/invocation vs MCP's 50K+ schema overhead -- better for AI agents

**MCP server (secondary, for persistent agent sessions):**
- ferroml-mcp/ exposing same operations as typed MCP tools via FastMCP
- Useful for team workflows, agent orchestration, persistent model state
- Build after CLI stabilizes ("promote CLI to MCP when patterns stabilize")

**Discoverability:**
- AGENTS.md in repo root (build, test, conventions, model selection guidance)
- llms.txt for documentation site
- Make AutoML accept natural language task descriptions
- Ensure search_space() returns JSON-serializable metadata
- Structured AutoMLReport with diagnostics, not just fitted model

### Phase 6: Launch and Growth (Sessions 15-17)
**Goal:** People find and adopt FerroML.
- Documentation site (mkdocs-material) with API reference + user guide
- Blog post: "Building a scikit-learn competitor in Rust with AI"
- Hacker News Show HN launch
- Kaggle notebook demonstrating FerroML workflows
- conda-forge submission (after 1-2 stable PyPI releases)
- Academic paper outline (JMLR target)
- Migration guide from scikit-learn

## v1.1 Scope (NOT in v1.0, planned for later)

- **Automated feature engineering** -- PolynomialFeatures, interaction terms, KBinsDiscretizer, automated feature selection. AutoML without this is limited; high-value addition.
- **Performance parity** -- Close remaining gaps (HistGBT, LogReg coordinate descent). Target: no model >2x slower than sklearn.
- **Hyperband/ASHA in AutoML** -- Early termination of unpromising trials. Currently missing.
- **MLP/GP in AutoML portfolio** -- Currently excluded from automatic selection.
- **sklearn __sklearn_tags__ interop** -- Full compatibility with sklearn pipelines and meta-estimators.
- **conda-forge** -- Secondary package distribution after PyPI stabilizes.

## Explicitly Out of Scope (not planned)

- Full Bayesian inference (MCMC, variational) -- different paradigm, use PyMC/Stan/NumPyro
- Time series forecasting -- own domain, use statsforecast/sktime/prophet
- Deep learning beyond MLP -- use PyTorch/TensorFlow
- Bayesian neural networks -- niche, use Pyro/NumPyro
- NLP beyond bag-of-words -- transformers won, use HuggingFace
- Reinforcement learning -- different domain entirely
- Image/audio processing -- out of scope for classical ML

## Session Continuity Protocol

Each session should:
1. Read this master design document
2. Read MEMORY.md for project context
3. Read the current phase's detailed plan (if it exists)
4. Check git log for what was done in previous sessions
5. Run tests to verify current state
6. Create or continue the phase plan
7. Execute work
8. Update MEMORY.md with session outcomes
9. Commit all work

## Key References

### Textbooks
- Bishop, PRML (2006) -- Bayesian ML foundations
- Hastie, Tibshirani, Friedman, ESL 2nd ed. (2009) -- Statistical learning
- Murphy, MLAPP (2012) -- Probabilistic ML encyclopedia
- Rasmussen and Williams, GPML (2006) -- Gaussian processes
- Golub and Van Loan, Matrix Computations 4th ed. -- Numerical linear algebra
- Boyd and Vandenberghe, Convex Optimization -- Optimization algorithms

### Papers
- Platt 1998 -- SMO for SVMs
- Chang and Lin 2011 -- LIBSVM
- Breiman 2001 -- Random Forests
- Friedman 2001 -- Gradient Boosting
- Chen and Guestrin 2016 -- XGBoost
- van der Maaten and Hinton 2008 -- t-SNE
- Liu et al. 2012 -- Isolation Forest

### Research Findings
- VibeTensor (NVIDIA, 2026) -- "Frankenstein effect" in AI-generated code: locally correct components compose into globally incorrect systems. Mitigation: integration-level testing.
- DeepStability (ICSE 2022) -- 250+ numerical defects in PyTorch/TensorFlow. Most common: log/exp overflow, probability precision, batch normalization.
- AI code has 1.7x more major bugs (Stack Overflow 2026). Silent correctness failures are the most dangerous failure mode.
- No ML library has formal proofs. Cross-library differential testing is the practical gold standard.

## Gaps Found in Final Review (added 2026-03-27)

### 1. Cross-Platform CI Testing
The release.yml BUILDS wheels for Linux/macOS/Windows but does not RUN tests on all platforms.
We need a separate CI workflow that runs `cargo test` and `pytest` on all three OS.
Add to Phase 4.

### 2. GPU/Sparse: Experimental or Stable?
Plan Q added GPU shaders (GpuDispatcher), Plan R added SparseModel for 12 models.
Decision needed: are these v1.0-stable or should they be marked `experimental`?
If experimental, they ship but with clear documentation that the API may change.
Add to Phase 1 audit (assess stability of GPU/Sparse features).

### 3. Versioning Strategy
Going from v0.3.1 to v1.0 is a big jump. v1.0 implies stability promises.
We should define: what does v1.0 mean for backwards compatibility?
Recommendation: v1.0 means "the Python API is stable; we won't break import paths or
function signatures without a major version bump." Internal Rust API can still evolve.
Document in README and CHANGELOG.

### 4. Thread Safety Verification
PyO3 + Rust should make predict() thread-safe, but this needs explicit testing.
Verify: can multiple Python threads call predict() on the same fitted model simultaneously?
Add to Frankenstein tests (Layer 5).

### 5. Reproducibility Documentation
Beyond fixing RF non-determinism, users need a clear guide:
- Which models are deterministic with a fixed random_state?
- Which are not (and why)?
- Is there a global seed mechanism?
Document in the correctness report and user guide.

### 6. What the Correctness Report Looks Like
The plan says "publish a correctness report" but doesn't define the format.
Format: a public markdown document with one section per algorithm containing:
- Algorithm name and category
- Mathematical reference (textbook/paper)
- Invariants tested (with pass/fail)
- Reference implementation comparison (sklearn/scipy tolerance)
- Adversarial inputs tested
- Frankenstein composition tests passed
- Known limitations or divergences from sklearn (with explanations)
This document ships WITH the release and is linked from the README.

## Success Criteria for v1.0

**Correctness:**
- [ ] Every shipped model passes all 5 correctness layers (including Frankenstein composition tests)
- [ ] Zero unwrap() in production model code
- [ ] Zero known test failures (fix or remove failing models)
- [ ] Thread safety verified (concurrent predict() from Python threads)
- [ ] Reproducibility documented (which models are deterministic, which are not)
- [ ] GPU/Sparse features assessed and marked stable or experimental

**Release:**
- [ ] pip install ferroml works on Linux, macOS, Windows
- [ ] CI runs tests on all three platforms (not just builds wheels)
- [ ] Full type stubs and docstrings on all public API
- [ ] Public correctness report published with release (per-algorithm pass/fail)
- [ ] README with benchmarks, quick-start, and sklearn comparison
- [ ] CHANGELOG updated with v1.0 changes
- [ ] Versioning policy documented (stable Python API commitment)

**AI-Agent-First:**
- [ ] CLI tool with structured JSON output
- [ ] MCP server functional for AI agent integration
- [ ] AGENTS.md and llms.txt present
- [ ] AutoML runs hands-off given dataset + task description
- [ ] recommend() API returns algorithm suggestions with explanations

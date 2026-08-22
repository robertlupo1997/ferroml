# Parity contract — supersedes the scalar-ε framing

**Source:** ChatGPT Deep Research, 2026-08-22, commissioned to answer FM-003.
**Status:** this replaces the objective recorded in `project-ops` commit `9b7c599`.

## What was wrong

The recorded objective was:

> "the fastest AutoML library that is provably score-equivalent to scikit-learn",
> gated by `|score_ferroml − score_sklearn| < ε`.

Two defects, both fatal to the claim as written:

1. **A scalar score comparison is too weak to support an equivalence claim.** Two
   classifiers can post identical accuracy while disagreeing on many individual
   predictions. A matching score hides different coefficients, probabilities, labels,
   cluster assignments, and transformed coordinates.
2. **"Provably" is an overclaim.** None of linfa, smartcore, XGBoost, LightGBM, cuML,
   or ONNX Runtime makes a general numerical-equivalence claim. scikit-learn's own
   definition of a compatible estimator covers the **estimator API** — fit, predict,
   estimator checks — and explicitly does not promise identical numerical results.
   XGBoost and LightGBM describe their sklearn interfaces as *interface* conformance.

## What the ecosystem actually does

There is no global ε anywhere. Tolerances are per model, per output, per backend, per
test:

| Family | Values observed in real test suites |
|---|---|
| Linear | linfa PLS `1e-6`; linfa ElasticNet `0.01` coefficients / `0.1` intercept; cuML Ridge `1e-2` predictions, `1e-3`–`1e-4` parameters |
| Tree ensembles | cuML edge cases near `1e-6`, but **general score tolerance 0.10 classification / ~0.07 regression** |
| Gradient boosting | **No sklearn-equivalence tolerance exists.** XGBoost/LightGBM `1e-6`–`1e-7` figures are internal native-vs-wrapper checks. These are different algorithms from sklearn's GBM — equal parameter *names* do not mean equal training procedures. |
| Clustering | cuML KMeans `atol=0.1, rtol=1e-2`; linfa GMM `0.1` **with component permutation handling** |
| Dim. reduction | cuML PCA `atol=1e-3` on **sign-adjusted** components; smartcore PCA `1e-3`/`1e-4` |
| Backend/operator | ONNX Runtime default `rtol=1e-3, atol=1e-5`, then per-operator overrides |

cuML justifies its relaxed forest tolerance from variation across many seeds. Smartcore
pins a K-fold seed because OS entropy made tests flaky. RNG is treated as a test
concern, not assumed away.

## The contract to implement

Replace the scalar gate with four layers.

**1. Exact structural checks.** Shapes, class lists, label sets, leaf indices, metadata.

**2. Array checks with explicit `atol` and `rtol`:**

```
|mine − reference| <= atol + rtol * |reference|
```

**3. Model-specific invariants** instead of raw-output equality where the output has a
symmetry:

| Model type | Compare on |
|---|---|
| Clustering | optimal label permutation, inertia, ARI/NMI |
| PCA / SVD | singular values and subspace distance — **not raw component signs** |
| Embeddings | neighborhood overlap, or Procrustes-aligned distance |
| Trees | structure and serialized inference where applicable |

**4. A versioned certification matrix.** Every tolerance is scoped to: sklearn version,
crate version, dtype, solver, seed, thread count, BLAS implementation, CPU/GPU backend.

Each family's tolerance is then **derived, not chosen**:

```
epsilon_family = max observed discrepancy over the certification matrix
               + a preregistered numerical margin
```

Derive it from the declared corpus before shipping. Never widen it after a production
failure — that inverts the gate into a rubber stamp.

## Starting points for measurement, not standards

- Same algorithm, same dtype, deterministic op order → exact or ULP-level
- Same algorithm, float32 → start near `rtol=1e-5, atol=1e-6`
- Different linear solvers → start near `rtol=1e-5, atol=1e-4` on predictions, then measure
- **Independently trained tree ensembles / boosting → do not promise a universal score ε at all**
- Clustering and dim. reduction → invariant-aware checks only

## Where bitwise parity is and isn't possible

Achievable only under identical implementation, op order, dtype, math library,
compiler/runtime behavior, serialized model, input ordering, random state, and no
uncontrolled parallel reductions. Smartcore does hold exact Naive Bayes fixtures — and
disables them on WASM32 because floats differ. That is the shape of the achievable case:
narrow and controlled.

Not achievable across differing iterative solvers, BLAS/GPU reduction order, split
enumeration or tie-breaking, quantization, RNG algorithm, cluster initialization,
eigenvector sign or degenerate eigenspaces, or embedding optimization.

## How the claim must be worded

Permitted:

> For scikit-learn X.Y, the supported estimator/version/dtype/solver matrix is tested
> against reference outputs using published, model-specific tolerances and invariance
> checks. This establishes empirical numerical conformance for the stated matrix. It
> does not claim identical training algorithms, bitwise equality, or behavior outside
> that matrix.

Or, if only the interface is guaranteed: *"Implements the scikit-learn estimator interface."*

**Forbidden:** "provably score-equivalent to scikit-learn." Do not publish it.

## What this means for the FerroML work in flight

The `max_features` fix was still correct and necessary — an unequal comparison is
invalid regardless of tolerance policy. But the fixed benchmark now needs a rerun, and
its results feed the certification corpus rather than a single pass/fail number.

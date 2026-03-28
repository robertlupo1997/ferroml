# Phase 2, Session 2: P0/P1 Bug Fixes

Copy-paste this as your prompt for the next Claude Code session:

---

I am working on FerroML v1.0. Read the master design at docs/plans/2026-03-27-v1-master-design.md and the memory at MEMORY.md to get full context.

This session is Phase 2, Session 2: P0/P1 Bug Fixes.

Phase 1 (correctness audit) is complete. The full audit report is at docs/correctness-report.md. Read it — it has the exact file:line locations for every bug.

Your job: Fix the 4 highest-priority bugs found in the audit. For each fix, write a regression test that would have caught the bug. Run `cargo test` after each fix to verify no breakage.

## Bug 1 (P0): DecisionTree Entropy criterion not implemented
- **File:** ferroml-core/src/models/tree.rs
- **Problem:** `DecisionTreeClassifier` accepts `SplitCriterion::Entropy` but always uses Gini. The `build_tree_weighted`, `find_best_split_weighted`, and `find_random_split_weighted` methods call `weighted_gini_impurity()` unconditionally. No `weighted_entropy` function exists.
- **Fix:** Implement `weighted_entropy` (= -Σ p_k log(p_k)) and add a `match self.criterion` dispatch in the classifier's split-finding methods, mirroring the regressor's MSE/MAE dispatch.
- **Test:** Fit a tree with Entropy criterion on a dataset where Entropy and Gini would select different splits. Verify the tree structure differs.

## Bug 2 (P1): LogisticRegression IRLS weight clamping
- **File:** ferroml-core/src/models/logistic.rs:638
- **Problem:** `w_clamped[i] = (var * sample_weights[i]).clamp(1e-10, 0.25)` — the 0.25 upper clamp is the max Bernoulli variance for unit weights. When `sample_weights[i] > 1` (e.g., ClassWeight::Balanced with imbalanced data), the product exceeds 0.25 legitimately, and clamping corrupts the Newton step.
- **Fix:** Clamp `var` alone to [1e-10, 0.25], then multiply by the unclamped sample weight: `w_clamped[i] = var.clamp(1e-10, 0.25) * sample_weights[i]`
- **Test:** Fit LogisticRegression with IRLS solver on imbalanced data (90/10 split) with ClassWeight::Balanced. Compare coefficients with/without the fix — the fixed version should produce coefficients closer to sklearn's output.

## Bug 3 (P1): AdaBoost regressor weight formula
- **File:** ferroml-core/src/models/adaboost.rs:524
- **Problem:** `alpha = self.learning_rate * beta.ln().abs()` — the `.abs()` is mathematically wrong. Should be `-beta.ln()` or `(1.0/beta).ln()`. Works by accident when beta < 1 (good estimator) but is wrong in principle.
- **Fix:** Change to `alpha = self.learning_rate * (1.0_f64 / beta).ln()` or equivalently `self.learning_rate * (-beta.ln())`.
- **Test:** Verify AdaBoost regressor produces correct estimator weights on a simple dataset. Compare with sklearn AdaBoostRegressor.

## Bug 4 (P1): AdaBoost docstring
- **File:** ferroml-core/src/models/adaboost.rs:7 (and anywhere else claiming SAMME.R)
- **Problem:** Docstring says "SAMME.R" but implementation uses hard class predictions (SAMME, the discrete variant).
- **Fix:** Change documentation to say "SAMME" everywhere.

## After all fixes:
1. Run `cargo test` (full suite)
2. Run `cargo test --test correctness` (integration tests)
3. Run `cargo clippy --all-targets -- -D warnings`
4. Update docs/correctness-report.md: change the 3 FIX verdicts to PASS with notes on what was fixed
5. Commit with descriptive message

Use parallel agents where fixes are independent (bugs 3+4 are in the same file, do those together; bugs 1 and 2 are independent).

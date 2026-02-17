//! Mutation Testing Support for FerroML
//!
//! This module provides documentation and utilities for mutation testing using
//! `cargo-mutants`. Mutation testing validates test quality by introducing small
//! changes (mutations) to the code and verifying that tests catch them.
//!
//! # Overview
//!
//! Mutation testing works by:
//! 1. Making small changes to the source code (e.g., changing `+` to `-`)
//! 2. Running the test suite against the mutated code
//! 3. Checking if any test fails (mutation "killed") or all pass (mutation "survived")
//!
//! A survived mutation indicates a gap in test coverage - the tests don't catch that bug.
//!
//! # Running Mutation Tests
//!
//! ## Quick Start
//!
//! ```bash
//! # Install cargo-mutants
//! cargo install cargo-mutants
//!
//! # Run mutation testing on ferroml-core
//! cargo mutants -p ferroml-core --timeout 300 -- --lib
//! ```
//!
//! ## Targeted Testing
//!
//! For faster iteration, test specific modules:
//!
//! ```bash
//! # Test only the metrics module
//! cargo mutants -p ferroml-core --file 'src/metrics/*.rs' -- --lib
//!
//! # Test only linear models
//! cargo mutants -p ferroml-core --file 'src/models/linear.rs' -- --lib
//!
//! # Test preprocessing
//! cargo mutants -p ferroml-core --file 'src/preprocessing/*.rs' -- --lib
//! ```
//!
//! ## CI/CD Integration
//!
//! Mutation testing runs weekly via GitHub Actions. See `.github/workflows/mutation.yml`.
//!
//! # Interpreting Results
//!
//! ## Mutation Score
//!
//! The mutation score is the percentage of mutations killed by tests:
//!
//! | Score | Interpretation |
//! |-------|----------------|
//! | 90%+  | Excellent test coverage |
//! | 80-90%| Good coverage, minor gaps |
//! | 70-80%| Acceptable, some improvements needed |
//! | <70%  | Significant test gaps |
//!
//! ## Common Survived Mutations
//!
//! Some mutations commonly survive and may indicate test gaps:
//!
//! 1. **Boundary conditions**: `>` mutated to `>=`
//! 2. **Arithmetic operations**: `+` to `-`, `*` to `/`
//! 3. **Boolean logic**: `&&` to `||`, `!` removal
//! 4. **Return values**: Changing constants, early returns
//!
//! # Writing Mutation-Resistant Tests
//!
//! ## Good Practices
//!
//! ```
//! # use ferroml_core::metrics::accuracy;
//! # use ndarray::array;
//! # fn main() -> ferroml_core::Result<()> {
//! // GOOD: Tests specific values and edge cases
//! let y = array![1.0, 0.0, 1.0, 1.0];
//! let acc = accuracy(&y, &y)?;
//! assert_eq!(acc, 1.0);  // Exact match catches more mutations
//!
//! let y_true = array![1.0, 1.0, 1.0];
//! let y_pred = array![0.0, 0.0, 0.0];
//! let acc = accuracy(&y_true, &y_pred)?;
//! assert_eq!(acc, 0.0);  // Edge case
//!
//! let y_true = array![1.0, 0.0, 1.0, 0.0];
//! let y_pred = array![1.0, 1.0, 0.0, 0.0];
//! let acc = accuracy(&y_true, &y_pred)?;
//! assert!((acc - 0.5).abs() < 1e-10);  // Known expected value
//! # Ok(())
//! # }
//! ```
//!
//! ## Testing Mathematical Operations
//!
//! For functions with mathematical operations, test that changing the operation
//! would produce wrong results:
//!
//! ```
//! # use ferroml_core::metrics::mse;
//! # use ndarray::array;
//! # fn main() -> ferroml_core::Result<()> {
//! // Test that MSE calculation is correct
//! let y_true = array![1.0, 2.0, 3.0];
//! let y_pred = array![2.0, 2.0, 2.0];
//! // Errors: [1.0, 0.0, -1.0], squared: [1.0, 0.0, 1.0], mean: 2/3
//! let mse = mse(&y_true, &y_pred)?;
//! assert!((mse - 2.0/3.0).abs() < 1e-10);
//! # Ok(())
//! # }
//! ```
//!
//! # Configuration
//!
//! The `mutants.toml` file in the repository root configures mutation testing:
//!
//! - Excludes test files, benchmarks, and generated code
//! - Excludes trivial implementations (Display, Debug, Clone)
//! - Excludes builder pattern setters
//! - Sets default timeout and parallelism
//!
//! # Priority Modules
//!
//! When time is limited, prioritize mutation testing for:
//!
//! 1. **`src/metrics/`** - Accuracy calculations must be correct
//! 2. **`src/models/linear.rs`** - Core linear algebra
//! 3. **`src/preprocessing/`** - Data transformations
//! 4. **`src/cross_validation/`** - CV logic affects all models
//! 5. **`src/models/tree.rs`** - Decision tree splits
//!
//! # Troubleshooting
//!
//! ## Timeouts
//!
//! If mutations timeout frequently:
//!
//! ```bash
//! # Increase timeout
//! cargo mutants -p ferroml-core --timeout 600 -- --lib
//!
//! # Or test a smaller module
//! cargo mutants -p ferroml-core --file 'src/metrics/regression.rs' -- --lib
//! ```
//!
//! ## Too Many Mutations
//!
//! For large codebases, use sampling:
//!
//! ```bash
//! # Random sample of mutations
//! cargo mutants -p ferroml-core --jobs 4 -- --lib | head -100
//! ```
//!
//! ## False Positives
//!
//! Some mutations are "equivalent" - they don't change behavior:
//!
//! - `x >= 0` vs `x > -1` for integers
//! - Reordering commutative operations
//!
//! These can be excluded in `mutants.toml`.

/// Marker type for mutation testing documentation
/// This module is primarily documentation; it doesn't contain runtime code.
pub struct MutationTestingDocs;

#[cfg(test)]
mod tests {
    //! Tests that validate our test suite catches common mutations.
    //!
    //! These "meta-tests" ensure our testing practices are mutation-resistant.

    use crate::metrics::{accuracy, mse, r2_score};
    use ndarray::array;

    /// Test that accuracy calculation catches arithmetic mutations
    #[test]
    fn meta_test_accuracy_catches_mutations() {
        // Perfect accuracy
        let y = array![1.0, 0.0, 1.0, 0.0];
        assert_eq!(accuracy(&y, &y).unwrap(), 1.0);

        // Zero accuracy (all wrong)
        let y_true = array![1.0, 1.0, 1.0, 1.0];
        let y_pred = array![0.0, 0.0, 0.0, 0.0];
        assert_eq!(accuracy(&y_true, &y_pred).unwrap(), 0.0);

        // 50% accuracy
        let y_true = array![1.0, 1.0, 0.0, 0.0];
        let y_pred = array![1.0, 0.0, 0.0, 1.0];
        assert!((accuracy(&y_true, &y_pred).unwrap() - 0.5).abs() < 1e-10);

        // 75% accuracy
        let y_true = array![1.0, 1.0, 1.0, 0.0];
        let y_pred = array![1.0, 1.0, 1.0, 1.0];
        assert!((accuracy(&y_true, &y_pred).unwrap() - 0.75).abs() < 1e-10);
    }

    /// Test that MSE calculation catches arithmetic mutations
    #[test]
    fn meta_test_mse_catches_mutations() {
        // Zero error
        let y = array![1.0, 2.0, 3.0];
        assert_eq!(mse(&y, &y).unwrap(), 0.0);

        // Known error: [1-2, 2-2, 3-4] = [-1, 0, -1], squared = [1, 0, 1], mean = 2/3
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![2.0, 2.0, 4.0];
        let expected = (1.0 + 0.0 + 1.0) / 3.0;
        assert!((mse(&y_true, &y_pred).unwrap() - expected).abs() < 1e-10);

        // Large error to catch sign mutations
        let y_true = array![0.0, 0.0];
        let y_pred = array![10.0, -10.0];
        assert_eq!(mse(&y_true, &y_pred).unwrap(), 100.0);
    }

    /// Test that R2 score calculation catches mutations
    #[test]
    fn meta_test_r2_catches_mutations() {
        // Perfect prediction
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((r2_score(&y, &y).unwrap() - 1.0).abs() < 1e-10);

        // Mean prediction (R2 = 0)
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let y_pred = array![mean, mean, mean, mean, mean];
        assert!(r2_score(&y_true, &y_pred).unwrap().abs() < 1e-10);

        // Worse than mean (negative R2)
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![3.0, 2.0, 1.0]; // Reversed
        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert!(r2 < 0.0, "R2 should be negative for reversed predictions");
    }

    /// Test edge cases that often escape mutation testing
    #[test]
    fn meta_test_edge_cases() {
        // Single element
        let y = array![5.0];
        assert_eq!(accuracy(&y, &y).unwrap(), 1.0);

        // Two elements with different predictions
        let y_true = array![0.0, 1.0];
        let y_pred = array![1.0, 0.0];
        assert_eq!(accuracy(&y_true, &y_pred).unwrap(), 0.0);
    }
}

//! Hyperparameter Search via Cross-Validation
//!
//! Grid and randomized search over hyperparameter combinations with cross-validation.
//!
//! ## Models
//!
//! - [`GridSearchCV`] - Exhaustive search over specified parameter grid
//! - [`RandomizedSearchCV`] - Random sampling from parameter distributions

use crate::cv::{cross_val_score, curves::ParameterSettable, CVConfig, KFold};
use crate::metrics::Metric;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single parameter grid entry: a parameter name and its candidate values.
pub type ParamGrid = HashMap<String, Vec<f64>>;

/// Result from a single parameter combination in grid/random search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Parameter combination that was evaluated
    pub params: HashMap<String, f64>,
    /// Cross-validation result for this combination
    pub mean_test_score: f64,
    /// Standard deviation of test scores
    pub std_test_score: f64,
    /// Rank (1 = best)
    pub rank: usize,
}

/// Exhaustive search over a parameter grid with cross-validation.
///
/// Evaluates all possible combinations of parameter values and selects
/// the best based on cross-validated performance.
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::cv::search::GridSearchCV;
/// use std::collections::HashMap;
///
/// let mut param_grid = HashMap::new();
/// param_grid.insert("alpha".to_string(), vec![0.001, 0.01, 0.1, 1.0, 10.0]);
///
/// let mut search = GridSearchCV::new(param_grid, 5);
/// search.search(&estimator, &x, &y, &metric)?;
///
/// println!("Best params: {:?}", search.best_params());
/// println!("Best score: {:.4}", search.best_score().unwrap());
/// ```
pub struct GridSearchCV {
    /// Parameter grid to search
    param_grid: ParamGrid,
    /// Number of CV folds
    n_folds: usize,
    /// Whether higher scores are better (default true)
    pub maximize: bool,
    /// CV config
    pub cv_config: CVConfig,

    // Results
    best_params: Option<HashMap<String, f64>>,
    best_score: Option<f64>,
    cv_results: Option<Vec<SearchResult>>,
}

impl GridSearchCV {
    /// Create a new GridSearchCV.
    ///
    /// # Arguments
    /// * `param_grid` - Map of parameter names to candidate values
    /// * `n_folds` - Number of cross-validation folds
    pub fn new(param_grid: ParamGrid, n_folds: usize) -> Self {
        Self {
            param_grid,
            n_folds,
            maximize: true,
            cv_config: CVConfig::default(),
            best_params: None,
            best_score: None,
            cv_results: None,
        }
    }

    /// Set whether to maximize (true) or minimize (false) the metric.
    pub fn with_maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }

    /// Get the best parameters found.
    pub fn best_params(&self) -> Option<&HashMap<String, f64>> {
        self.best_params.as_ref()
    }

    /// Get the best score found.
    pub fn best_score(&self) -> Option<f64> {
        self.best_score
    }

    /// Get all CV results.
    pub fn cv_results(&self) -> Option<&Vec<SearchResult>> {
        self.cv_results.as_ref()
    }

    /// Run the grid search.
    pub fn search<E, M>(
        &mut self,
        estimator: &E,
        x: &Array2<f64>,
        y: &Array1<f64>,
        metric: &M,
    ) -> Result<()>
    where
        E: ParameterSettable + Clone + Send + Sync,
        E::Fitted: Send,
        M: Metric + Sync,
    {
        let combinations = generate_grid(&self.param_grid);
        if combinations.is_empty() {
            return Err(FerroError::invalid_input("Empty parameter grid"));
        }

        let cv = KFold::new(self.n_folds);
        let mut results = Vec::with_capacity(combinations.len());

        for combo in &combinations {
            let mut est = estimator.clone();
            for (name, &value) in combo {
                est.set_param(name, value)?;
            }

            let cv_result = cross_val_score(&est, x, y, &cv, metric, &self.cv_config, None)?;

            results.push(SearchResult {
                params: combo.clone(),
                mean_test_score: cv_result.mean_test_score,
                std_test_score: cv_result.std_test_score,
                rank: 0, // will be set below
            });
        }

        // Rank results
        rank_results(&mut results, self.maximize);

        // Extract best
        if let Some(best) = results.iter().find(|r| r.rank == 1) {
            self.best_params = Some(best.params.clone());
            self.best_score = Some(best.mean_test_score);
        }

        self.cv_results = Some(results);
        Ok(())
    }
}

/// Randomized search over parameter distributions with cross-validation.
///
/// Samples parameter combinations randomly and evaluates via cross-validation.
/// More efficient than grid search when the parameter space is large.
pub struct RandomizedSearchCV {
    /// Parameter grid (values are sampled uniformly from this list)
    param_grid: ParamGrid,
    /// Number of parameter combinations to try
    n_iter: usize,
    /// Number of CV folds
    n_folds: usize,
    /// Whether higher scores are better
    pub maximize: bool,
    /// Random seed
    pub random_state: Option<u64>,
    /// CV config
    pub cv_config: CVConfig,

    // Results
    best_params: Option<HashMap<String, f64>>,
    best_score: Option<f64>,
    cv_results: Option<Vec<SearchResult>>,
}

impl RandomizedSearchCV {
    /// Create a new RandomizedSearchCV.
    ///
    /// # Arguments
    /// * `param_grid` - Map of parameter names to candidate values
    /// * `n_iter` - Number of random combinations to try
    /// * `n_folds` - Number of cross-validation folds
    pub fn new(param_grid: ParamGrid, n_iter: usize, n_folds: usize) -> Self {
        Self {
            param_grid,
            n_iter,
            n_folds,
            maximize: true,
            random_state: None,
            cv_config: CVConfig::default(),
            best_params: None,
            best_score: None,
            cv_results: None,
        }
    }

    /// Set whether to maximize or minimize.
    pub fn with_maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }

    /// Set random seed.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Get the best parameters found.
    pub fn best_params(&self) -> Option<&HashMap<String, f64>> {
        self.best_params.as_ref()
    }

    /// Get the best score found.
    pub fn best_score(&self) -> Option<f64> {
        self.best_score
    }

    /// Get all CV results.
    pub fn cv_results(&self) -> Option<&Vec<SearchResult>> {
        self.cv_results.as_ref()
    }

    /// Run the randomized search.
    pub fn search<E, M>(
        &mut self,
        estimator: &E,
        x: &Array2<f64>,
        y: &Array1<f64>,
        metric: &M,
    ) -> Result<()>
    where
        E: ParameterSettable + Clone + Send + Sync,
        E::Fitted: Send,
        M: Metric + Sync,
    {
        if self.param_grid.is_empty() {
            return Err(FerroError::invalid_input("Empty parameter grid"));
        }

        let cv = KFold::new(self.n_folds);
        let mut results = Vec::with_capacity(self.n_iter);
        let mut rng = self.random_state.unwrap_or(42);

        let param_names: Vec<String> = self.param_grid.keys().cloned().collect();

        for _ in 0..self.n_iter {
            let mut combo = HashMap::new();

            for name in &param_names {
                let values = &self.param_grid[name];
                if values.is_empty() {
                    continue;
                }
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (rng >> 33) as usize % values.len();
                combo.insert(name.clone(), values[idx]);
            }

            let mut est = estimator.clone();
            for (name, &value) in &combo {
                est.set_param(name, value)?;
            }

            let cv_result = cross_val_score(&est, x, y, &cv, metric, &self.cv_config, None)?;

            results.push(SearchResult {
                params: combo,
                mean_test_score: cv_result.mean_test_score,
                std_test_score: cv_result.std_test_score,
                rank: 0,
            });
        }

        rank_results(&mut results, self.maximize);

        if let Some(best) = results.iter().find(|r| r.rank == 1) {
            self.best_params = Some(best.params.clone());
            self.best_score = Some(best.mean_test_score);
        }

        self.cv_results = Some(results);
        Ok(())
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Generate all combinations from a parameter grid.
fn generate_grid(param_grid: &ParamGrid) -> Vec<HashMap<String, f64>> {
    let param_names: Vec<&String> = param_grid.keys().collect();
    let param_values: Vec<&Vec<f64>> = param_names.iter().map(|n| &param_grid[*n]).collect();

    if param_names.is_empty() {
        return vec![];
    }

    let mut combinations = Vec::new();
    let mut indices = vec![0usize; param_names.len()];

    loop {
        // Build current combination
        let mut combo = HashMap::new();
        for (i, &name) in param_names.iter().enumerate() {
            combo.insert(name.clone(), param_values[i][indices[i]]);
        }
        combinations.push(combo);

        // Increment indices (like an odometer)
        let mut carry = true;
        for i in (0..param_names.len()).rev() {
            if carry {
                indices[i] += 1;
                if indices[i] >= param_values[i].len() {
                    indices[i] = 0;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            break; // All combinations generated
        }
    }

    combinations
}

/// Rank search results (1 = best).
fn rank_results(results: &mut [SearchResult], maximize: bool) {
    let mut scored: Vec<(usize, f64)> = results
        .iter()
        .enumerate()
        .map(|(i, r)| (i, r.mean_test_score))
        .collect();

    if maximize {
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    for (rank, (idx, _)) in scored.iter().enumerate() {
        results[*idx].rank = rank + 1;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_grid_single_param() {
        let mut grid = ParamGrid::new();
        grid.insert("alpha".to_string(), vec![0.1, 1.0, 10.0]);

        let combos = generate_grid(&grid);
        assert_eq!(combos.len(), 3);
    }

    #[test]
    fn test_generate_grid_two_params() {
        let mut grid = ParamGrid::new();
        grid.insert("alpha".to_string(), vec![0.1, 1.0]);
        grid.insert("beta".to_string(), vec![0.01, 0.1, 1.0]);

        let combos = generate_grid(&grid);
        assert_eq!(combos.len(), 6); // 2 * 3
    }

    #[test]
    fn test_generate_grid_empty() {
        let grid = ParamGrid::new();
        let combos = generate_grid(&grid);
        assert!(combos.is_empty());
    }

    #[test]
    fn test_rank_results_maximize() {
        let mut results = vec![
            SearchResult {
                params: HashMap::new(),
                mean_test_score: 0.7,
                std_test_score: 0.01,
                rank: 0,
            },
            SearchResult {
                params: HashMap::new(),
                mean_test_score: 0.9,
                std_test_score: 0.02,
                rank: 0,
            },
            SearchResult {
                params: HashMap::new(),
                mean_test_score: 0.8,
                std_test_score: 0.01,
                rank: 0,
            },
        ];

        rank_results(&mut results, true);
        assert_eq!(results[0].rank, 3); // 0.7 is worst
        assert_eq!(results[1].rank, 1); // 0.9 is best
        assert_eq!(results[2].rank, 2); // 0.8 is middle
    }

    #[test]
    fn test_rank_results_minimize() {
        let mut results = vec![
            SearchResult {
                params: HashMap::new(),
                mean_test_score: 0.7,
                std_test_score: 0.01,
                rank: 0,
            },
            SearchResult {
                params: HashMap::new(),
                mean_test_score: 0.9,
                std_test_score: 0.02,
                rank: 0,
            },
            SearchResult {
                params: HashMap::new(),
                mean_test_score: 0.8,
                std_test_score: 0.01,
                rank: 0,
            },
        ];

        rank_results(&mut results, false);
        assert_eq!(results[0].rank, 1); // 0.7 is best (minimize)
        assert_eq!(results[1].rank, 3); // 0.9 is worst
        assert_eq!(results[2].rank, 2); // 0.8 is middle
    }
}

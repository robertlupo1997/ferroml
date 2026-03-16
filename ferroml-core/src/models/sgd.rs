//! Stochastic Gradient Descent (SGD) Models
//!
//! Linear models fitted via SGD optimization.
//!
//! ## Models
//!
//! - [`SGDClassifier`] - Linear classifier with SGD (supports hinge, log, modified_huber loss)
//! - [`SGDRegressor`] - Linear regressor with SGD (supports squared, huber, epsilon_insensitive loss)
//! - [`Perceptron`] - Classic Perceptron algorithm (equivalent to SGDClassifier with hinge loss, no regularization)
//! - [`PassiveAggressiveClassifier`] - Online learning classifier (PA-I / PA-II)

use crate::models::traits::IncrementalModel;
use crate::models::{check_is_fitted, validate_fit_input, validate_predict_input, Model};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

// =============================================================================
// Loss Functions
// =============================================================================

/// Loss functions for SGD classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SGDClassifierLoss {
    /// Hinge loss (linear SVM): max(0, 1 - y*f(x))
    Hinge,
    /// Logistic loss: log(1 + exp(-y*f(x)))
    Log,
    /// Modified Huber: smooth hinge with quadratic near the boundary
    ModifiedHuber,
}

impl Default for SGDClassifierLoss {
    fn default() -> Self {
        Self::Hinge
    }
}

/// Loss functions for SGD regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SGDRegressorLoss {
    /// Squared loss: (y - f(x))^2
    SquaredError,
    /// Huber loss: quadratic for small errors, linear for large
    Huber,
    /// Epsilon-insensitive loss (SVR-like): max(0, |y - f(x)| - epsilon)
    EpsilonInsensitive,
}

impl Default for SGDRegressorLoss {
    fn default() -> Self {
        Self::SquaredError
    }
}

/// Regularization penalty type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Penalty {
    /// No regularization
    None,
    /// L2 regularization (Ridge)
    L2,
    /// L1 regularization (Lasso)
    L1,
    /// Elastic Net (L1 + L2)
    ElasticNet,
}

impl Default for Penalty {
    fn default() -> Self {
        Self::L2
    }
}

/// Learning rate schedule.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LearningRateScheduleType {
    /// Constant learning rate
    Constant,
    /// Optimal: eta = 1 / (alpha * (t + t0))
    Optimal,
    /// Inverse scaling: eta = eta0 / t^power_t
    InverseScaling,
}

impl Default for LearningRateScheduleType {
    fn default() -> Self {
        Self::Optimal
    }
}

// =============================================================================
// SGD Classifier
// =============================================================================

/// Linear classifier fitted via Stochastic Gradient Descent.
///
/// Supports multiple loss functions (hinge for SVM, log for logistic regression)
/// and regularization penalties (L1, L2, ElasticNet).
///
/// For multiclass, uses One-vs-Rest strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGDClassifier {
    /// Loss function
    pub loss: SGDClassifierLoss,
    /// Regularization penalty
    pub penalty: Penalty,
    /// Regularization strength
    pub alpha: f64,
    /// L1 ratio for ElasticNet (0 = pure L2, 1 = pure L1)
    pub l1_ratio: f64,
    /// Initial learning rate
    pub eta0: f64,
    /// Learning rate schedule
    pub learning_rate: LearningRateScheduleType,
    /// Power for inverse scaling
    pub power_t: f64,
    /// Maximum epochs
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Whether to shuffle data each epoch
    pub shuffle: bool,
    /// Random seed
    pub random_state: Option<u64>,
    /// Epsilon for Huber loss
    pub epsilon: f64,
    /// Whether to reuse previous coefficients as initialization
    pub warm_start: bool,

    // Fitted state
    coef: Option<Array2<f64>>, // (n_classes_or_1, n_features)
    intercept: Option<Array1<f64>>,
    classes: Option<Array1<f64>>,
    n_features: Option<usize>,
    n_iter: Option<usize>,
    /// Global SGD step counter (persists across partial_fit calls for learning rate schedule)
    t: usize,
    /// RNG state (persists across partial_fit calls)
    rng_state: u64,
}

impl SGDClassifier {
    /// Create a new SGDClassifier with default settings.
    pub fn new() -> Self {
        Self {
            loss: SGDClassifierLoss::Hinge,
            penalty: Penalty::L2,
            alpha: 0.0001,
            l1_ratio: 0.15,
            eta0: 0.01,
            learning_rate: LearningRateScheduleType::Optimal,
            power_t: 0.5,
            max_iter: 1000,
            tol: 1e-3,
            fit_intercept: true,
            shuffle: true,
            random_state: None,
            epsilon: 0.1,
            warm_start: false,
            coef: None,
            intercept: None,
            classes: None,
            n_features: None,
            n_iter: None,
            t: 0,
            rng_state: 0,
        }
    }

    /// Set the loss function.
    pub fn with_loss(mut self, loss: SGDClassifierLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set the penalty.
    pub fn with_penalty(mut self, penalty: Penalty) -> Self {
        self.penalty = penalty;
        self
    }

    /// Set the regularization strength.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the initial learning rate.
    pub fn with_eta0(mut self, eta0: f64) -> Self {
        self.eta0 = eta0;
        self
    }

    /// Set max iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set random seed.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set whether to reuse previous coefficients as initialization.
    pub fn with_warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Get fitted coefficients.
    pub fn coef(&self) -> Option<&Array2<f64>> {
        self.coef.as_ref()
    }

    /// Get fitted intercept.
    pub fn intercept(&self) -> Option<&Array1<f64>> {
        self.intercept.as_ref()
    }

    /// Get number of iterations run.
    pub fn n_iter(&self) -> Option<usize> {
        self.n_iter
    }

    /// Get the unique class labels.
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Compute decision function.
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.coef, "decision_function")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let coef = self.coef.as_ref().unwrap();
        let intercept = self.intercept.as_ref().unwrap();

        // coef: (n_outputs, n_features), x: (n_samples, n_features)
        // result: (n_samples, n_outputs)
        let mut result = x.dot(&coef.t());
        for (mut col, &b) in result.columns_mut().into_iter().zip(intercept.iter()) {
            col += b;
        }
        Ok(result)
    }

    /// Compute learning rate at step t.
    fn get_eta(&self, t: usize) -> f64 {
        match self.learning_rate {
            LearningRateScheduleType::Constant => self.eta0,
            LearningRateScheduleType::Optimal => 1.0 / (self.alpha * (t as f64 + 1e4)),
            LearningRateScheduleType::InverseScaling => {
                self.eta0 / (t as f64 + 1.0).powf(self.power_t)
            }
        }
    }

    /// Apply penalty gradient to coefficient.
    fn apply_penalty(&self, coef: f64) -> f64 {
        match self.penalty {
            Penalty::None => 0.0,
            Penalty::L2 => self.alpha * coef,
            Penalty::L1 => self.alpha * coef.signum(),
            Penalty::ElasticNet => {
                self.alpha * (self.l1_ratio * coef.signum() + (1.0 - self.l1_ratio) * coef)
            }
        }
    }

    /// Fit a single binary classifier: y_binary in {-1, +1}.
    fn fit_binary(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        seed: u64,
        initial_coef: Option<Array1<f64>>,
        initial_intercept: f64,
    ) -> (Array1<f64>, f64, usize) {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut coef = initial_coef.unwrap_or_else(|| Array1::zeros(n_features));
        let mut intercept = initial_intercept;
        let mut t: usize = 0;
        let mut rng = seed;
        let mut best_loss = f64::INFINITY;
        let mut no_improvement_count = 0;
        let mut actual_iter = 0;

        let mut indices: Vec<usize> = (0..n_samples).collect();

        for epoch in 0..self.max_iter {
            actual_iter = epoch + 1;

            Self::run_one_classification_epoch(
                x,
                y,
                &mut coef,
                &mut intercept,
                &mut t,
                &mut rng,
                &mut indices,
                self.loss,
                self.shuffle,
                self.fit_intercept,
                self.epsilon,
                |t_val| self.get_eta(t_val),
                |c_val| self.apply_penalty(c_val),
            );

            let epoch_loss = Self::compute_classification_loss(x, y, &coef, intercept, self.loss);

            // Check convergence
            if (best_loss - epoch_loss).abs() < self.tol {
                no_improvement_count += 1;
                if no_improvement_count >= 5 {
                    break;
                }
            } else {
                no_improvement_count = 0;
            }
            if epoch_loss < best_loss {
                best_loss = epoch_loss;
            }
        }

        (coef, intercept, actual_iter)
    }

    /// Run a single epoch of SGD classification updates on the given weights.
    /// Mutates coef, intercept, t, rng, and indices in place.
    #[allow(clippy::too_many_arguments)]
    fn run_one_classification_epoch(
        x: &Array2<f64>,
        y: &Array1<f64>,
        coef: &mut Array1<f64>,
        intercept: &mut f64,
        t: &mut usize,
        rng: &mut u64,
        indices: &mut Vec<usize>,
        loss: SGDClassifierLoss,
        shuffle: bool,
        fit_intercept: bool,
        _epsilon: f64,
        get_eta: impl Fn(usize) -> f64,
        apply_penalty: impl Fn(f64) -> f64,
    ) {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Shuffle
        if shuffle {
            for i in (1..n_samples).rev() {
                *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (*rng >> 33) as usize % (i + 1);
                indices.swap(i, j);
            }
        }

        for &i in indices.iter() {
            let xi = x.row(i);
            let yi = y[i]; // {-1, +1}

            let eta = get_eta(*t);
            *t += 1;

            let decision = xi.dot(coef) + *intercept;
            let margin = yi * decision;

            // Compute loss gradient
            let dloss = match loss {
                SGDClassifierLoss::Hinge => {
                    if margin < 1.0 {
                        -yi
                    } else {
                        0.0
                    }
                }
                SGDClassifierLoss::Log => {
                    if margin >= 0.0 {
                        let exp_neg = (-margin).exp();
                        -yi * exp_neg / (1.0 + exp_neg)
                    } else {
                        let exp_pos = margin.exp();
                        -yi / (1.0 + exp_pos)
                    }
                }
                SGDClassifierLoss::ModifiedHuber => {
                    if margin >= 1.0 {
                        0.0
                    } else if margin >= -1.0 {
                        -2.0 * yi * (1.0 - margin)
                    } else {
                        -4.0 * yi
                    }
                }
            };

            // Update coefficients
            for j in 0..n_features {
                let penalty = apply_penalty(coef[j]);
                coef[j] -= eta * (dloss * xi[j] + penalty);
            }
            if fit_intercept {
                *intercept -= eta * dloss;
            }
        }
    }

    /// Compute the average classification loss over the dataset.
    fn compute_classification_loss(
        x: &Array2<f64>,
        y: &Array1<f64>,
        coef: &Array1<f64>,
        intercept: f64,
        loss: SGDClassifierLoss,
    ) -> f64 {
        let n_samples = x.nrows();
        let mut total_loss = 0.0;
        for i in 0..n_samples {
            let xi = x.row(i);
            let yi = y[i];
            let decision = xi.dot(coef) + intercept;
            let margin = yi * decision;
            total_loss += match loss {
                SGDClassifierLoss::Hinge => (1.0 - margin).max(0.0),
                SGDClassifierLoss::Log => {
                    if margin >= 0.0 {
                        (1.0 + (-margin).exp()).ln()
                    } else {
                        -margin + (1.0 + margin.exp()).ln()
                    }
                }
                SGDClassifierLoss::ModifiedHuber => {
                    if margin >= 1.0 {
                        0.0
                    } else if margin >= -1.0 {
                        (1.0 - margin).powi(2)
                    } else {
                        -4.0 * margin
                    }
                }
            };
        }
        total_loss / n_samples as f64
    }
}

impl Default for SGDClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for SGDClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        if self.alpha <= 0.0 && self.learning_rate == LearningRateScheduleType::Optimal {
            return Err(FerroError::invalid_input(
                "alpha must be > 0 when using optimal learning rate schedule",
            ));
        }

        let n_features = x.ncols();
        let classes = crate::models::get_unique_classes(y);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(FerroError::invalid_input(
                "SGDClassifier requires at least 2 classes",
            ));
        }

        let seed = self.random_state.unwrap_or(42);

        // Determine initial coefficients for warm_start
        let prev_coef = if self.warm_start {
            self.coef.take()
        } else {
            None
        };
        let prev_intercept = if self.warm_start {
            self.intercept.take()
        } else {
            None
        };

        if n_classes == 2 {
            // Binary: single classifier with {-1, +1} encoding
            let y_encoded: Array1<f64> = y
                .iter()
                .map(|&v| {
                    if (v - classes[0]).abs() < 1e-10 {
                        -1.0
                    } else {
                        1.0
                    }
                })
                .collect();

            let init_coef = prev_coef.map(|c| c.row(0).to_owned());
            let init_intercept = prev_intercept.as_ref().map_or(0.0, |i| i[0]);

            let (coef, intercept, n_iter) =
                self.fit_binary(x, &y_encoded, seed, init_coef, init_intercept);
            self.coef = Some(coef.insert_axis(ndarray::Axis(0)));
            self.intercept = Some(Array1::from_vec(vec![intercept]));
            self.n_iter = Some(n_iter);
        } else {
            // Multiclass: OvR
            let mut all_coefs = Array2::zeros((n_classes, n_features));
            let mut all_intercepts = Array1::zeros(n_classes);
            let mut max_iter = 0;

            for (ci, &c) in classes.iter().enumerate() {
                let y_binary: Array1<f64> = y
                    .iter()
                    .map(|&v| if (v - c).abs() < 1e-10 { 1.0 } else { -1.0 })
                    .collect();

                let class_seed = seed.wrapping_add(ci as u64);

                let init_coef = prev_coef
                    .as_ref()
                    .filter(|c| c.ncols() == n_features && ci < c.nrows())
                    .map(|c| c.row(ci).to_owned());
                let init_intercept =
                    prev_intercept
                        .as_ref()
                        .map_or(0.0, |i| if ci < i.len() { i[ci] } else { 0.0 });

                let (coef, intercept, n_iter) =
                    self.fit_binary(x, &y_binary, class_seed, init_coef, init_intercept);

                all_coefs.row_mut(ci).assign(&coef);
                all_intercepts[ci] = intercept;
                max_iter = max_iter.max(n_iter);
            }

            self.coef = Some(all_coefs);
            self.intercept = Some(all_intercepts);
            self.n_iter = Some(max_iter);
        }

        self.classes = Some(classes);
        self.n_features = Some(n_features);
        self.t = 0;
        self.rng_state = seed;
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let decision = self.decision_function(x)?;
        let classes = self.classes.as_ref().unwrap();

        if classes.len() == 2 {
            Ok(decision
                .column(0)
                .iter()
                .map(|&v| if v >= 0.0 { classes[1] } else { classes[0] })
                .collect())
        } else {
            Ok(decision
                .rows()
                .into_iter()
                .map(|row| {
                    let max_idx = row
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    classes[max_idx]
                })
                .collect())
        }
    }

    fn is_fitted(&self) -> bool {
        self.coef.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        let coef = self.coef.as_ref()?; // Array2: (n_classes, n_features)
        let abs_coef = coef.mapv(|c| c.abs());
        let summed = abs_coef.sum_axis(ndarray::Axis(0)); // sum across classes
        let total = summed.sum();
        if total > 0.0 {
            Some(summed / total)
        } else {
            Some(summed)
        }
    }

    fn model_name(&self) -> &str {
        "SGDClassifier"
    }
}

impl IncrementalModel for SGDClassifier {
    fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        self.partial_fit_with_classes(x, y, None)
    }

    fn partial_fit_with_classes(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        classes: Option<&[f64]>,
    ) -> Result<()> {
        validate_fit_input(x, y)?;

        if self.alpha <= 0.0 && self.learning_rate == LearningRateScheduleType::Optimal {
            return Err(FerroError::invalid_input(
                "alpha must be > 0 when using optimal learning rate schedule",
            ));
        }

        let n_features = x.ncols();

        // If not yet fitted, initialize weights
        if self.coef.is_none() {
            // Determine classes from the provided list or from y
            let class_list = if let Some(c) = classes {
                Array1::from_vec(c.to_vec())
            } else {
                crate::models::get_unique_classes(y)
            };
            let n_classes = class_list.len();
            if n_classes < 2 {
                return Err(FerroError::invalid_input(
                    "SGDClassifier requires at least 2 classes",
                ));
            }

            let n_outputs = if n_classes == 2 { 1 } else { n_classes };
            self.coef = Some(Array2::zeros((n_outputs, n_features)));
            self.intercept = Some(Array1::zeros(n_outputs));
            self.classes = Some(class_list);
            self.n_features = Some(n_features);
            self.n_iter = Some(0);
            self.t = 0;
            self.rng_state = self.random_state.unwrap_or(42);
        }

        // Validate feature count matches
        if n_features != self.n_features.unwrap() {
            return Err(FerroError::shape_mismatch(
                format!("{} features", self.n_features.unwrap()),
                format!("{} features", n_features),
            ));
        }

        let classes = self.classes.clone().unwrap();
        let n_classes = classes.len();
        let n_samples = x.nrows();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        if n_classes == 2 {
            // Binary: single classifier with {-1, +1} encoding
            let y_encoded: Array1<f64> = y
                .iter()
                .map(|&v| {
                    if (v - classes[0]).abs() < 1e-10 {
                        -1.0
                    } else {
                        1.0
                    }
                })
                .collect();

            let mut coef = self.coef.as_ref().unwrap().row(0).to_owned();
            let mut intercept = self.intercept.as_ref().unwrap()[0];
            let mut t = self.t;
            let mut rng = self.rng_state;

            Self::run_one_classification_epoch(
                x,
                &y_encoded,
                &mut coef,
                &mut intercept,
                &mut t,
                &mut rng,
                &mut indices,
                self.loss,
                self.shuffle,
                self.fit_intercept,
                self.epsilon,
                |t_val| self.get_eta(t_val),
                |c_val| self.apply_penalty(c_val),
            );

            self.coef.as_mut().unwrap().row_mut(0).assign(&coef);
            self.intercept.as_mut().unwrap()[0] = intercept;
            self.t = t;
            self.rng_state = rng;
        } else {
            // Multiclass: OvR — one epoch per binary classifier
            let mut t = self.t;
            let mut rng = self.rng_state;

            for (ci, &c) in classes.iter().enumerate() {
                let y_binary: Array1<f64> = y
                    .iter()
                    .map(|&v| if (v - c).abs() < 1e-10 { 1.0 } else { -1.0 })
                    .collect();

                let mut coef = self.coef.as_ref().unwrap().row(ci).to_owned();
                let mut intercept = self.intercept.as_ref().unwrap()[ci];
                let mut class_t = t;
                let mut class_rng = rng.wrapping_add(ci as u64);

                Self::run_one_classification_epoch(
                    x,
                    &y_binary,
                    &mut coef,
                    &mut intercept,
                    &mut class_t,
                    &mut class_rng,
                    &mut indices,
                    self.loss,
                    self.shuffle,
                    self.fit_intercept,
                    self.epsilon,
                    |t_val| self.get_eta(t_val),
                    |c_val| self.apply_penalty(c_val),
                );

                self.coef.as_mut().unwrap().row_mut(ci).assign(&coef);
                self.intercept.as_mut().unwrap()[ci] = intercept;
                // Use the last class's t for the global counter
                if ci == n_classes - 1 {
                    t = class_t;
                    rng = class_rng;
                }
            }

            self.t = t;
            self.rng_state = rng;
        }

        *self.n_iter.as_mut().unwrap() += 1;
        Ok(())
    }
}

// =============================================================================
// SGD Regressor
// =============================================================================

/// Linear regressor fitted via Stochastic Gradient Descent.
///
/// Supports squared error, Huber, and epsilon-insensitive loss functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGDRegressor {
    /// Loss function
    pub loss: SGDRegressorLoss,
    /// Regularization penalty
    pub penalty: Penalty,
    /// Regularization strength
    pub alpha: f64,
    /// L1 ratio for ElasticNet
    pub l1_ratio: f64,
    /// Initial learning rate
    pub eta0: f64,
    /// Learning rate schedule
    pub learning_rate: LearningRateScheduleType,
    /// Power for inverse scaling
    pub power_t: f64,
    /// Maximum epochs
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Whether to shuffle each epoch
    pub shuffle: bool,
    /// Random seed
    pub random_state: Option<u64>,
    /// Epsilon for Huber/epsilon-insensitive loss
    pub epsilon: f64,
    /// Whether to reuse previous coefficients as initialization
    pub warm_start: bool,

    // Fitted state
    coef: Option<Array1<f64>>,
    intercept: Option<f64>,
    n_features: Option<usize>,
    n_iter: Option<usize>,
    /// Global SGD step counter (persists across partial_fit calls for learning rate schedule)
    t: usize,
    /// RNG state (persists across partial_fit calls)
    rng_state: u64,
}

impl SGDRegressor {
    /// Create a new SGDRegressor.
    pub fn new() -> Self {
        Self {
            loss: SGDRegressorLoss::SquaredError,
            penalty: Penalty::L2,
            alpha: 0.0001,
            l1_ratio: 0.15,
            eta0: 0.01,
            learning_rate: LearningRateScheduleType::InverseScaling,
            power_t: 0.25,
            max_iter: 1000,
            tol: 1e-3,
            fit_intercept: true,
            shuffle: true,
            random_state: None,
            epsilon: 0.1,
            warm_start: false,
            coef: None,
            intercept: None,
            n_features: None,
            n_iter: None,
            t: 0,
            rng_state: 0,
        }
    }

    /// Set the loss function.
    pub fn with_loss(mut self, loss: SGDRegressorLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set the penalty.
    pub fn with_penalty(mut self, penalty: Penalty) -> Self {
        self.penalty = penalty;
        self
    }

    /// Set the regularization strength.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set max iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set random seed.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set whether to reuse previous coefficients as initialization.
    pub fn with_warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Get fitted coefficients.
    pub fn coef(&self) -> Option<&Array1<f64>> {
        self.coef.as_ref()
    }

    /// Get fitted intercept.
    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }

    /// Apply penalty gradient.
    fn apply_penalty(&self, coef: f64) -> f64 {
        match self.penalty {
            Penalty::None => 0.0,
            Penalty::L2 => self.alpha * coef,
            Penalty::L1 => self.alpha * coef.signum(),
            Penalty::ElasticNet => {
                self.alpha * (self.l1_ratio * coef.signum() + (1.0 - self.l1_ratio) * coef)
            }
        }
    }

    /// Compute learning rate at step t.
    fn get_eta(&self, t: usize) -> f64 {
        match self.learning_rate {
            LearningRateScheduleType::Constant => self.eta0,
            LearningRateScheduleType::Optimal => 1.0 / (self.alpha * (t as f64 + 1e4)),
            LearningRateScheduleType::InverseScaling => {
                self.eta0 / (t as f64 + 1.0).powf(self.power_t)
            }
        }
    }

    /// Run a single epoch of SGD regression updates on the given weights.
    #[allow(clippy::too_many_arguments)]
    fn run_one_regression_epoch(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        coef: &mut Array1<f64>,
        intercept: &mut f64,
        t: &mut usize,
        rng: &mut u64,
        indices: &mut Vec<usize>,
    ) {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if self.shuffle {
            for i in (1..n_samples).rev() {
                *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (*rng >> 33) as usize % (i + 1);
                indices.swap(i, j);
            }
        }

        for &i in indices.iter() {
            let xi = x.row(i);
            let yi = y[i];
            let eta = self.get_eta(*t);
            *t += 1;

            let pred = xi.dot(coef) + *intercept;
            let error = pred - yi;

            let dloss = match self.loss {
                SGDRegressorLoss::SquaredError => 2.0 * error,
                SGDRegressorLoss::Huber => {
                    let abs_err = error.abs();
                    if abs_err <= self.epsilon {
                        2.0 * error
                    } else {
                        2.0 * self.epsilon * error.signum()
                    }
                }
                SGDRegressorLoss::EpsilonInsensitive => {
                    let abs_err = error.abs();
                    if abs_err <= self.epsilon {
                        0.0
                    } else {
                        error.signum()
                    }
                }
            };

            for j in 0..n_features {
                let penalty = self.apply_penalty(coef[j]);
                coef[j] -= eta * (dloss * xi[j] + penalty);
            }
            if self.fit_intercept {
                *intercept -= eta * dloss;
            }
        }
    }
}

impl Default for SGDRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for SGDRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        if self.alpha <= 0.0 && self.learning_rate == LearningRateScheduleType::Optimal {
            return Err(FerroError::invalid_input(
                "alpha must be > 0 when using optimal learning rate schedule",
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Warm start: reuse previous coefficients if available and compatible
        let mut coef = if self.warm_start {
            self.coef
                .take()
                .filter(|c| c.len() == n_features)
                .unwrap_or_else(|| Array1::zeros(n_features))
        } else {
            Array1::zeros(n_features)
        };
        let mut intercept = if self.warm_start {
            self.intercept.take().unwrap_or(0.0)
        } else {
            0.0
        };
        let mut t: usize = 0;
        let mut rng = self.random_state.unwrap_or(42);
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut best_loss = f64::INFINITY;
        let mut no_improvement_count = 0;
        let mut actual_iter = 0;

        for epoch in 0..self.max_iter {
            actual_iter = epoch + 1;

            if self.shuffle {
                for i in (1..n_samples).rev() {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let j = (rng >> 33) as usize % (i + 1);
                    indices.swap(i, j);
                }
            }

            let mut epoch_loss = 0.0;

            for &i in &indices {
                let xi = x.row(i);
                let yi = y[i];
                let eta = self.get_eta(t);
                t += 1;

                let pred = xi.dot(&coef) + intercept;
                let error = pred - yi;

                // Compute loss gradient
                let dloss = match self.loss {
                    SGDRegressorLoss::SquaredError => {
                        epoch_loss += error * error;
                        2.0 * error
                    }
                    SGDRegressorLoss::Huber => {
                        let abs_err = error.abs();
                        if abs_err <= self.epsilon {
                            epoch_loss += abs_err * abs_err;
                            2.0 * error
                        } else {
                            epoch_loss +=
                                2.0 * self.epsilon * abs_err - self.epsilon * self.epsilon;
                            2.0 * self.epsilon * error.signum()
                        }
                    }
                    SGDRegressorLoss::EpsilonInsensitive => {
                        let abs_err = error.abs();
                        if abs_err <= self.epsilon {
                            0.0
                        } else {
                            epoch_loss += abs_err - self.epsilon;
                            error.signum()
                        }
                    }
                };

                for j in 0..n_features {
                    let penalty = self.apply_penalty(coef[j]);
                    coef[j] -= eta * (dloss * xi[j] + penalty);
                }
                if self.fit_intercept {
                    intercept -= eta * dloss;
                }
            }

            epoch_loss /= n_samples as f64;

            if (best_loss - epoch_loss).abs() < self.tol {
                no_improvement_count += 1;
                if no_improvement_count >= 5 {
                    break;
                }
            } else {
                no_improvement_count = 0;
            }
            if epoch_loss < best_loss {
                best_loss = epoch_loss;
            }
        }

        // Check for numerical overflow during training
        if coef.iter().any(|v| !v.is_finite()) || !intercept.is_finite() {
            return Err(FerroError::numerical(
                "SGD coefficients diverged (NaN/Inf) — try reducing learning_rate or scaling input data",
            ));
        }

        self.coef = Some(coef);
        self.intercept = Some(intercept);
        self.n_features = Some(n_features);
        self.n_iter = Some(actual_iter);
        self.t = 0;
        self.rng_state = self.random_state.unwrap_or(42);
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.coef, "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let coef = self.coef.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);
        Ok(x.dot(coef) + intercept)
    }

    fn is_fitted(&self) -> bool {
        self.coef.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        let coef = self.coef.as_ref()?;
        let abs_coef: Array1<f64> = coef.mapv(|c| c.abs());
        let sum = abs_coef.sum();
        if sum > 0.0 {
            Some(abs_coef / sum)
        } else {
            Some(abs_coef)
        }
    }

    fn model_name(&self) -> &str {
        "SGDRegressor"
    }

    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;
        crate::metrics::r2_score(y, &predictions)
    }
}

impl IncrementalModel for SGDRegressor {
    fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        if self.alpha <= 0.0 && self.learning_rate == LearningRateScheduleType::Optimal {
            return Err(FerroError::invalid_input(
                "alpha must be > 0 when using optimal learning rate schedule",
            ));
        }

        let n_features = x.ncols();
        let n_samples = x.nrows();

        // If not yet fitted, initialize weights
        if self.coef.is_none() {
            self.coef = Some(Array1::zeros(n_features));
            self.intercept = Some(0.0);
            self.n_features = Some(n_features);
            self.n_iter = Some(0);
            self.t = 0;
            self.rng_state = self.random_state.unwrap_or(42);
        }

        // Validate feature count matches
        if n_features != self.n_features.unwrap() {
            return Err(FerroError::shape_mismatch(
                format!("{} features", self.n_features.unwrap()),
                format!("{} features", n_features),
            ));
        }

        let mut coef = self.coef.take().unwrap();
        let mut intercept = self.intercept.unwrap();
        let mut t = self.t;
        let mut rng = self.rng_state;
        let mut indices: Vec<usize> = (0..n_samples).collect();

        self.run_one_regression_epoch(
            x,
            y,
            &mut coef,
            &mut intercept,
            &mut t,
            &mut rng,
            &mut indices,
        );

        self.coef = Some(coef);
        self.intercept = Some(intercept);
        self.t = t;
        self.rng_state = rng;
        *self.n_iter.as_mut().unwrap() += 1;
        Ok(())
    }
}

// =============================================================================
// Perceptron
// =============================================================================

/// Perceptron classifier.
///
/// Equivalent to SGDClassifier with hinge loss, no regularization,
/// and constant learning rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Perceptron {
    inner: SGDClassifier,
}

impl Perceptron {
    /// Create a new Perceptron.
    pub fn new() -> Self {
        let inner = SGDClassifier {
            loss: SGDClassifierLoss::Hinge,
            penalty: Penalty::None,
            alpha: 0.0,
            eta0: 1.0,
            learning_rate: LearningRateScheduleType::Constant,
            max_iter: 1000,
            tol: 1e-3,
            ..SGDClassifier::new()
        };
        Self { inner }
    }

    /// Set max iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.inner.max_iter = max_iter;
        self
    }

    /// Set random seed.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.inner.random_state = Some(seed);
        self
    }

    /// Set the penalty.
    pub fn with_penalty(mut self, penalty: Penalty) -> Self {
        self.inner.penalty = penalty;
        self
    }

    /// Set alpha.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.inner.alpha = alpha;
        self
    }
}

impl Default for Perceptron {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for Perceptron {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        self.inner.fit(x, y)
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.inner.predict(x)
    }

    fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }

    fn n_features(&self) -> Option<usize> {
        self.inner.n_features()
    }

    fn model_name(&self) -> &str {
        "Perceptron"
    }
}

impl Perceptron {
    /// Compute the decision function (raw scores before thresholding).
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.inner.decision_function(x)
    }
}

impl IncrementalModel for Perceptron {
    fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        self.inner.partial_fit(x, y)
    }

    fn partial_fit_with_classes(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        classes: Option<&[f64]>,
    ) -> Result<()> {
        self.inner.partial_fit_with_classes(x, y, classes)
    }
}

// =============================================================================
// Passive Aggressive Classifier
// =============================================================================

/// Passive Aggressive classifier.
///
/// Online learning classifier that makes aggressive updates when a
/// prediction violates the margin.
///
/// - PA-I: caps update by `C`
/// - PA-II: regularized update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassiveAggressiveClassifier {
    /// Regularization parameter (aggressiveness)
    pub c: f64,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Maximum epochs
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to shuffle each epoch
    pub shuffle: bool,
    /// Random seed
    pub random_state: Option<u64>,

    // Fitted state
    coef: Option<Array2<f64>>,
    intercept: Option<Array1<f64>>,
    classes: Option<Array1<f64>>,
    n_features: Option<usize>,
}

impl PassiveAggressiveClassifier {
    /// Create a new PassiveAggressiveClassifier.
    pub fn new(c: f64) -> Self {
        Self {
            c,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-3,
            shuffle: true,
            random_state: None,
            coef: None,
            intercept: None,
            classes: None,
            n_features: None,
        }
    }

    /// Set max iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set random seed.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Compute the decision function (raw scores before thresholding).
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.coef, "decision_function")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let coef = self.coef.as_ref().unwrap();
        let intercept = self.intercept.as_ref().unwrap();

        let mut result = x.dot(&coef.t());
        for (mut col, &b) in result.columns_mut().into_iter().zip(intercept.iter()) {
            col += b;
        }
        Ok(result)
    }

    /// Get fitted coefficients.
    pub fn coef(&self) -> Option<&Array2<f64>> {
        self.coef.as_ref()
    }

    /// Get fitted intercept.
    pub fn intercept(&self) -> Option<&Array1<f64>> {
        self.intercept.as_ref()
    }

    /// Get the unique class labels.
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Fit a single binary PA classifier: y in {-1, +1}.
    fn fit_binary(&self, x: &Array2<f64>, y: &Array1<f64>, seed: u64) -> (Array1<f64>, f64) {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut coef = Array1::zeros(n_features);
        let mut intercept = 0.0;
        let mut rng = seed;
        let mut indices: Vec<usize> = (0..n_samples).collect();

        for _epoch in 0..self.max_iter {
            if self.shuffle {
                for i in (1..n_samples).rev() {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let j = (rng >> 33) as usize % (i + 1);
                    indices.swap(i, j);
                }
            }

            let mut total_update = 0.0;

            for &i in &indices {
                let xi = x.row(i);
                let yi = y[i];

                let decision = xi.dot(&coef) + intercept;
                let loss = (1.0 - yi * decision).max(0.0);

                if loss > 0.0 {
                    let xi_norm_sq: f64 = xi.iter().map(|v| v * v).sum::<f64>()
                        + if self.fit_intercept { 1.0 } else { 0.0 };

                    // PA-I: tau = min(loss / ||x||^2, C)
                    let tau = if xi_norm_sq > 0.0 {
                        (loss / xi_norm_sq).min(self.c)
                    } else {
                        0.0
                    };

                    for j in 0..n_features {
                        coef[j] += tau * yi * xi[j];
                    }
                    if self.fit_intercept {
                        intercept += tau * yi;
                    }
                    total_update += tau.abs();
                }
            }

            if total_update / (n_samples as f64) < self.tol {
                break;
            }
        }

        (coef, intercept)
    }
}

impl Model for PassiveAggressiveClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        if self.c <= 0.0 {
            return Err(FerroError::invalid_input(
                "C must be > 0 for PassiveAggressive models",
            ));
        }

        let n_features = x.ncols();
        let classes = crate::models::get_unique_classes(y);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(FerroError::invalid_input(
                "PassiveAggressiveClassifier requires at least 2 classes",
            ));
        }

        let seed = self.random_state.unwrap_or(42);

        if n_classes == 2 {
            let y_encoded: Array1<f64> = y
                .iter()
                .map(|&v| {
                    if (v - classes[0]).abs() < 1e-10 {
                        -1.0
                    } else {
                        1.0
                    }
                })
                .collect();

            let (coef, intercept) = self.fit_binary(x, &y_encoded, seed);
            self.coef = Some(coef.insert_axis(ndarray::Axis(0)));
            self.intercept = Some(Array1::from_vec(vec![intercept]));
        } else {
            let mut all_coefs = Array2::zeros((n_classes, n_features));
            let mut all_intercepts = Array1::zeros(n_classes);

            for (ci, &c) in classes.iter().enumerate() {
                let y_binary: Array1<f64> = y
                    .iter()
                    .map(|&v| if (v - c).abs() < 1e-10 { 1.0 } else { -1.0 })
                    .collect();

                let class_seed = seed.wrapping_add(ci as u64);
                let (coef, intercept) = self.fit_binary(x, &y_binary, class_seed);

                all_coefs.row_mut(ci).assign(&coef);
                all_intercepts[ci] = intercept;
            }

            self.coef = Some(all_coefs);
            self.intercept = Some(all_intercepts);
        }

        self.classes = Some(classes);
        self.n_features = Some(n_features);
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.coef, "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let coef = self.coef.as_ref().unwrap();
        let intercept = self.intercept.as_ref().unwrap();
        let classes = self.classes.as_ref().unwrap();

        let mut decision = x.dot(&coef.t());
        for (mut col, &b) in decision.columns_mut().into_iter().zip(intercept.iter()) {
            col += b;
        }

        if classes.len() == 2 {
            Ok(decision
                .column(0)
                .iter()
                .map(|&v| if v >= 0.0 { classes[1] } else { classes[0] })
                .collect())
        } else {
            Ok(decision
                .rows()
                .into_iter()
                .map(|row| {
                    let max_idx = row
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    classes[max_idx]
                })
                .collect())
        }
    }

    fn is_fitted(&self) -> bool {
        self.coef.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn model_name(&self) -> &str {
        "PassiveAggressiveClassifier"
    }
}

impl IncrementalModel for PassiveAggressiveClassifier {
    fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        self.partial_fit_with_classes(x, y, None)
    }

    fn partial_fit_with_classes(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        classes: Option<&[f64]>,
    ) -> Result<()> {
        validate_fit_input(x, y)?;

        if self.c <= 0.0 {
            return Err(FerroError::invalid_input(
                "C must be > 0 for PassiveAggressive models",
            ));
        }

        let n_features = x.ncols();

        // If not yet fitted, initialize weights
        if self.coef.is_none() {
            let class_list = if let Some(c) = classes {
                Array1::from_vec(c.to_vec())
            } else {
                crate::models::get_unique_classes(y)
            };
            let n_classes = class_list.len();
            if n_classes < 2 {
                return Err(FerroError::invalid_input(
                    "PassiveAggressiveClassifier requires at least 2 classes",
                ));
            }

            let n_outputs = if n_classes == 2 { 1 } else { n_classes };
            self.coef = Some(Array2::zeros((n_outputs, n_features)));
            self.intercept = Some(Array1::zeros(n_outputs));
            self.classes = Some(class_list);
            self.n_features = Some(n_features);
        }

        // Validate feature count matches
        if n_features != self.n_features.unwrap() {
            return Err(FerroError::shape_mismatch(
                format!("{} features", self.n_features.unwrap()),
                format!("{} features", n_features),
            ));
        }

        let classes = self.classes.as_ref().unwrap().clone();
        let n_classes = classes.len();
        let mut coef = self.coef.take().unwrap();
        let mut intercept = self.intercept.take().unwrap();

        // PA update for each sample (one-vs-rest for multiclass)
        for i in 0..x.nrows() {
            let xi = x.row(i);
            let yi = y[i];
            let xi_norm_sq: f64 =
                xi.iter().map(|v| v * v).sum::<f64>() + if self.fit_intercept { 1.0 } else { 0.0 };

            if n_classes == 2 {
                // Binary: single weight vector, encode as -1/+1
                let label = if (yi - classes[0]).abs() < 1e-10 {
                    -1.0
                } else {
                    1.0
                };
                let decision = xi.dot(&coef.row(0)) + intercept[0];
                let loss = (1.0 - label * decision).max(0.0);
                if loss > 0.0 && xi_norm_sq > 0.0 {
                    let tau = (loss / xi_norm_sq).min(self.c);
                    for j in 0..n_features {
                        coef[[0, j]] += tau * label * xi[j];
                    }
                    if self.fit_intercept {
                        intercept[0] += tau * label;
                    }
                }
            } else {
                // OvR: update each binary classifier
                for ci in 0..n_classes {
                    let label = if (yi - classes[ci]).abs() < 1e-10 {
                        1.0
                    } else {
                        -1.0
                    };
                    let decision = xi.dot(&coef.row(ci)) + intercept[ci];
                    let loss = (1.0 - label * decision).max(0.0);
                    if loss > 0.0 && xi_norm_sq > 0.0 {
                        let tau = (loss / xi_norm_sq).min(self.c);
                        for j in 0..n_features {
                            coef[[ci, j]] += tau * label * xi[j];
                        }
                        if self.fit_intercept {
                            intercept[ci] += tau * label;
                        }
                    }
                }
            }
        }

        self.coef = Some(coef);
        self.intercept = Some(intercept);
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_binary_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 7.0, 7.0, 8.0, 7.0, 7.0, 8.0,
                8.0, 8.0, 9.0, 7.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        (x, y)
    }

    fn make_regression_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
        (x, y)
    }

    // -------------------------------------------------------------------------
    // SGDClassifier Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sgd_classifier_hinge() {
        let (x, y) = make_binary_data();
        let mut clf = SGDClassifier::new()
            .with_loss(SGDClassifierLoss::Hinge)
            .with_max_iter(200)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        let preds = clf.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-10)
            .count();
        assert!(correct >= 8, "Expected >=8 correct, got {}", correct);
    }

    #[test]
    fn test_sgd_classifier_log_loss() {
        let (x, y) = make_binary_data();
        let mut clf = SGDClassifier::new()
            .with_loss(SGDClassifierLoss::Log)
            .with_max_iter(200)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        let preds = clf.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-10)
            .count();
        assert!(correct >= 8, "Expected >=8 correct, got {}", correct);
    }

    #[test]
    fn test_sgd_classifier_modified_huber() {
        let (x, y) = make_binary_data();
        let mut clf = SGDClassifier::new()
            .with_loss(SGDClassifierLoss::ModifiedHuber)
            .with_max_iter(200)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        let preds = clf.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-10)
            .count();
        assert!(correct >= 8, "Expected >=8 correct, got {}", correct);
    }

    #[test]
    fn test_sgd_classifier_multiclass() {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 10.0, 0.0, 11.0, 0.0, 10.0, 1.0, 11.0, 1.0,
                5.0, 10.0, 6.0, 10.0, 5.0, 11.0, 6.0, 11.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
        ]);

        let mut clf = SGDClassifier::new()
            .with_max_iter(500)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        let preds = clf.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-10)
            .count();
        assert!(correct >= 9, "Expected >=9/12 correct, got {}", correct);
    }

    #[test]
    fn test_sgd_classifier_not_fitted() {
        let clf = SGDClassifier::new();
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(clf.predict(&x).is_err());
    }

    #[test]
    fn test_sgd_classifier_penalties() {
        let (x, y) = make_binary_data();
        for penalty in [Penalty::None, Penalty::L1, Penalty::L2, Penalty::ElasticNet] {
            let mut clf = SGDClassifier::new()
                .with_penalty(penalty)
                .with_max_iter(200)
                .with_random_state(42);
            clf.fit(&x, &y).unwrap();
            assert!(clf.is_fitted());
        }
    }

    // -------------------------------------------------------------------------
    // SGDRegressor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sgd_regressor_squared() {
        let (x, y) = make_regression_data();
        let mut reg = SGDRegressor::new()
            .with_loss(SGDRegressorLoss::SquaredError)
            .with_max_iter(500)
            .with_random_state(42);
        reg.fit(&x, &y).unwrap();

        assert!(reg.is_fitted());
        let preds = reg.predict(&x).unwrap();
        let mse: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        assert!(mse < 5.0, "MSE too high: {}", mse);
    }

    #[test]
    fn test_sgd_regressor_huber() {
        let (x, y) = make_regression_data();
        let mut reg = SGDRegressor::new()
            .with_loss(SGDRegressorLoss::Huber)
            .with_max_iter(500)
            .with_random_state(42);
        reg.fit(&x, &y).unwrap();
        assert!(reg.is_fitted());
    }

    #[test]
    fn test_sgd_regressor_epsilon_insensitive() {
        let (x, y) = make_regression_data();
        let mut reg = SGDRegressor::new()
            .with_loss(SGDRegressorLoss::EpsilonInsensitive)
            .with_max_iter(500)
            .with_random_state(42);
        reg.fit(&x, &y).unwrap();
        assert!(reg.is_fitted());
    }

    #[test]
    fn test_sgd_regressor_not_fitted() {
        let reg = SGDRegressor::new();
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        assert!(reg.predict(&x).is_err());
    }

    // -------------------------------------------------------------------------
    // Perceptron Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_perceptron_binary() {
        let (x, y) = make_binary_data();
        let mut clf = Perceptron::new().with_max_iter(200).with_random_state(42);
        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        let preds = clf.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-10)
            .count();
        assert!(correct >= 8, "Expected >=8 correct, got {}", correct);
    }

    #[test]
    fn test_perceptron_not_fitted() {
        let clf = Perceptron::new();
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(clf.predict(&x).is_err());
    }

    // -------------------------------------------------------------------------
    // PassiveAggressiveClassifier Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_passive_aggressive_binary() {
        let (x, y) = make_binary_data();
        let mut clf = PassiveAggressiveClassifier::new(1.0)
            .with_max_iter(200)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        let preds = clf.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-10)
            .count();
        assert!(correct >= 8, "Expected >=8 correct, got {}", correct);
    }

    #[test]
    fn test_passive_aggressive_multiclass() {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 10.0, 0.0, 11.0, 0.0, 10.0, 1.0, 11.0, 1.0,
                5.0, 10.0, 6.0, 10.0, 5.0, 11.0, 6.0, 11.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
        ]);

        let mut clf = PassiveAggressiveClassifier::new(1.0)
            .with_max_iter(200)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        let preds = clf.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-10)
            .count();
        assert!(correct >= 9, "Expected >=9/12 correct, got {}", correct);
    }

    #[test]
    fn test_passive_aggressive_not_fitted() {
        let clf = PassiveAggressiveClassifier::new(1.0);
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(clf.predict(&x).is_err());
    }

    #[test]
    fn test_passive_aggressive_c_zero_rejected() {
        let (x, y) = make_binary_data();
        let mut clf = PassiveAggressiveClassifier::new(0.0);
        let err = clf.fit(&x, &y).unwrap_err();
        assert!(
            err.to_string().contains("C must be > 0"),
            "Expected C validation error, got: {}",
            err
        );
    }

    #[test]
    fn test_passive_aggressive_c_negative_rejected() {
        let (x, y) = make_binary_data();
        let mut clf = PassiveAggressiveClassifier::new(-1.0);
        let err = clf.fit(&x, &y).unwrap_err();
        assert!(
            err.to_string().contains("C must be > 0"),
            "Expected C validation error, got: {}",
            err
        );
    }

    // -------------------------------------------------------------------------
    // IncrementalModel (partial_fit) Tests
    // -------------------------------------------------------------------------

    use crate::models::traits::IncrementalModel;

    #[test]
    fn test_sgd_classifier_partial_fit_binary() {
        let (x, y) = make_binary_data();

        // partial_fit over 200 single-epoch calls should learn the data
        let mut clf = SGDClassifier::new()
            .with_loss(SGDClassifierLoss::Log)
            .with_random_state(42);
        for _ in 0..200 {
            clf.partial_fit(&x, &y).unwrap();
        }

        assert!(clf.is_fitted());
        let preds = clf.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-10)
            .count();
        assert!(correct >= 8, "Expected >=8 correct, got {}", correct);
    }

    #[test]
    fn test_sgd_classifier_partial_fit_with_classes() {
        let (x, y) = make_binary_data();

        // Provide classes upfront
        let mut clf = SGDClassifier::new()
            .with_loss(SGDClassifierLoss::Hinge)
            .with_random_state(42);
        clf.partial_fit_with_classes(&x, &y, Some(&[0.0, 1.0]))
            .unwrap();
        assert!(clf.is_fitted());
        assert_eq!(clf.classes().unwrap().len(), 2);
    }

    #[test]
    fn test_sgd_classifier_partial_fit_multiclass() {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 10.0, 0.0, 11.0, 0.0, 10.0, 1.0, 11.0, 1.0,
                5.0, 10.0, 6.0, 10.0, 5.0, 11.0, 6.0, 11.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
        ]);

        let mut clf = SGDClassifier::new()
            .with_loss(SGDClassifierLoss::Log)
            .with_random_state(42);
        // Must specify all classes on first call for multiclass
        clf.partial_fit_with_classes(&x, &y, Some(&[0.0, 1.0, 2.0]))
            .unwrap();
        for _ in 1..500 {
            clf.partial_fit(&x, &y).unwrap();
        }

        let preds = clf.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-10)
            .count();
        assert!(correct >= 9, "Expected >=9/12 correct, got {}", correct);
    }

    #[test]
    fn test_sgd_classifier_partial_fit_batched() {
        // Simulate online learning: feed data in two batches
        let (x, y) = make_binary_data();
        let n = x.nrows() / 2;

        let x1 = x.slice(ndarray::s![..n, ..]).to_owned();
        let y1 = y.slice(ndarray::s![..n]).to_owned();
        let x2 = x.slice(ndarray::s![n.., ..]).to_owned();
        let y2 = y.slice(ndarray::s![n..]).to_owned();

        let mut clf = SGDClassifier::new()
            .with_loss(SGDClassifierLoss::Log)
            .with_random_state(42);

        // First call must see both classes (or specify them)
        clf.partial_fit_with_classes(&x1, &y1, Some(&[0.0, 1.0]))
            .unwrap();

        // Continue with second batch
        for _ in 0..200 {
            clf.partial_fit(&x1, &y1).unwrap();
            clf.partial_fit(&x2, &y2).unwrap();
        }

        let preds = clf.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-10)
            .count();
        assert!(correct >= 8, "Expected >=8 correct, got {}", correct);
    }

    #[test]
    fn test_sgd_classifier_partial_fit_feature_mismatch() {
        let (x, y) = make_binary_data();
        let mut clf = SGDClassifier::new().with_random_state(42);
        clf.partial_fit(&x, &y).unwrap();

        // Try with wrong number of features
        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y_bad = Array1::from_vec(vec![0.0, 1.0]);
        assert!(clf.partial_fit(&x_bad, &y_bad).is_err());
    }

    #[test]
    fn test_sgd_regressor_partial_fit() {
        let (x, y) = make_regression_data();

        let mut reg = SGDRegressor::new()
            .with_loss(SGDRegressorLoss::SquaredError)
            .with_random_state(42);
        for _ in 0..500 {
            reg.partial_fit(&x, &y).unwrap();
        }

        assert!(reg.is_fitted());
        let preds = reg.predict(&x).unwrap();
        let mse: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        assert!(mse < 5.0, "MSE too high: {}", mse);
    }

    #[test]
    fn test_sgd_regressor_partial_fit_batched() {
        let (x, y) = make_regression_data();
        let n = x.nrows() / 2;

        let x1 = x.slice(ndarray::s![..n, ..]).to_owned();
        let y1 = y.slice(ndarray::s![..n]).to_owned();
        let x2 = x.slice(ndarray::s![n.., ..]).to_owned();
        let y2 = y.slice(ndarray::s![n..]).to_owned();

        let mut reg = SGDRegressor::new()
            .with_loss(SGDRegressorLoss::SquaredError)
            .with_random_state(42);

        for _ in 0..500 {
            reg.partial_fit(&x1, &y1).unwrap();
            reg.partial_fit(&x2, &y2).unwrap();
        }

        assert!(reg.is_fitted());
        let preds = reg.predict(&x).unwrap();
        let mse: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        assert!(mse < 10.0, "MSE too high: {}", mse);
    }

    #[test]
    fn test_sgd_regressor_partial_fit_feature_mismatch() {
        let (x, y) = make_regression_data();
        let mut reg = SGDRegressor::new().with_random_state(42);
        reg.partial_fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y_bad = Array1::from_vec(vec![1.0, 2.0]);
        assert!(reg.partial_fit(&x_bad, &y_bad).is_err());
    }

    #[test]
    fn test_sgd_regressor_partial_fit_huber() {
        let (x, y) = make_regression_data();
        let mut reg = SGDRegressor::new()
            .with_loss(SGDRegressorLoss::Huber)
            .with_random_state(42);
        for _ in 0..500 {
            reg.partial_fit(&x, &y).unwrap();
        }
        assert!(reg.is_fitted());
    }

    #[test]
    fn test_sgd_classifier_partial_fit_initializes_unfitted() {
        // Verify partial_fit works on a completely fresh model
        let mut clf = SGDClassifier::new().with_random_state(42);
        assert!(!clf.is_fitted());

        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        clf.partial_fit(&x, &y).unwrap();
        assert!(clf.is_fitted());
        assert_eq!(clf.n_features(), Some(2));
        assert!(clf.classes().is_some());
    }

    #[test]
    fn test_sgd_regressor_partial_fit_initializes_unfitted() {
        let mut reg = SGDRegressor::new().with_random_state(42);
        assert!(!reg.is_fitted());

        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0]);

        reg.partial_fit(&x, &y).unwrap();
        assert!(reg.is_fitted());
        assert_eq!(reg.n_features(), Some(1));
    }

    #[test]
    fn test_sgd_classifier_warm_start() {
        let (x, y) = make_binary_data();

        let mut model = SGDClassifier::new()
            .with_warm_start(true)
            .with_max_iter(50)
            .with_random_state(42)
            .with_loss(SGDClassifierLoss::Log);
        model.fit(&x, &y).unwrap();
        let score1 = model.score(&x, &y).unwrap();

        // Fit again — should reuse coefficients and converge at least as well
        model.fit(&x, &y).unwrap();
        let score2 = model.score(&x, &y).unwrap();

        assert!(
            score2 >= score1 - 0.1,
            "score2={} should be >= score1={} - 0.1",
            score2,
            score1
        );
    }

    #[test]
    fn test_sgd_regressor_warm_start() {
        let (x, y) = make_regression_data();

        let mut model = SGDRegressor::new()
            .with_warm_start(true)
            .with_max_iter(50)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();
        let score1 = model.score(&x, &y).unwrap();

        // Fit again — should reuse coefficients and converge at least as well
        model.fit(&x, &y).unwrap();
        let score2 = model.score(&x, &y).unwrap();

        assert!(
            score2 >= score1 - 0.1,
            "score2={} should be >= score1={} - 0.1",
            score2,
            score1
        );
    }
}

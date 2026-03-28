//! Model Card — structured metadata for every FerroML model.
//!
//! Each model implements [`HasModelCard`] to expose a static [`ModelCard`] describing
//! what the model does, its computational complexity, interpretability, and more.
//! This makes models discoverable by AI agents, documentation generators, and
//! recommendation engines.

/// Structured metadata about a machine learning model.
#[derive(Debug, Clone)]
pub struct ModelCard {
    /// Python class name as exposed via `import ferroml`.
    pub name: String,
    /// Task types: `"classification"`, `"regression"`, `"clustering"`,
    /// `"dimensionality_reduction"`, `"outlier_detection"`, `"calibration"`,
    /// `"ensemble"`, etc.
    pub task: Vec<String>,
    /// Algorithmic time complexity, e.g. `"O(n*p)"`.
    pub complexity: String,
    /// `"high"`, `"medium"`, or `"low"`.
    pub interpretability: String,
    /// Whether the model accepts sparse input.
    pub supports_sparse: bool,
    /// Whether the model supports incremental / online learning (`partial_fit`).
    pub supports_incremental: bool,
    /// Whether the model supports sample weights (`fit_weighted`).
    pub supports_sample_weight: bool,
    /// 2-3 key strengths.
    pub strengths: Vec<String>,
    /// 2-3 key limitations.
    pub limitations: Vec<String>,
    /// Key academic references.
    pub references: Vec<String>,
}

/// Trait for models that expose structured metadata.
pub trait HasModelCard {
    /// Return the model card (static metadata — no instance needed).
    fn model_card() -> ModelCard;
}

// =============================================================================
// Linear models
// =============================================================================

impl HasModelCard for crate::models::LinearRegression {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "LinearRegression".into(),
            task: vec!["regression".into()],
            complexity: "O(n*p^2)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Full statistical diagnostics (p-values, CIs, F-statistic)".into(),
                "Fast closed-form solution".into(),
                "Highly interpretable coefficients".into(),
            ],
            limitations: vec![
                "Assumes linear relationship".into(),
                "Sensitive to multicollinearity".into(),
            ],
            references: vec![
                "Hastie, Tibshirani, Friedman. The Elements of Statistical Learning (2009)".into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::RidgeRegression {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "RidgeRegression".into(),
            task: vec!["regression".into()],
            complexity: "O(n*p^2)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "L2 regularization handles multicollinearity".into(),
                "Closed-form solution with tunable alpha".into(),
            ],
            limitations: vec![
                "Does not perform feature selection".into(),
                "Assumes linear relationship".into(),
            ],
            references: vec!["Hoerl, Kennard. Ridge Regression (1970)".into()],
        }
    }
}

impl HasModelCard for crate::models::LassoRegression {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "LassoRegression".into(),
            task: vec!["regression".into()],
            complexity: "O(n*p*n_iter)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "L1 regularization performs automatic feature selection".into(),
                "Produces sparse coefficient vectors".into(),
            ],
            limitations: vec![
                "Selects at most n features when n < p".into(),
                "Unstable selection among correlated features".into(),
            ],
            references: vec![
                "Tibshirani. Regression Shrinkage and Selection via the Lasso (1996)".into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::ElasticNet {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "ElasticNet".into(),
            task: vec!["regression".into()],
            complexity: "O(n*p*n_iter)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Combines L1 and L2 regularization".into(),
                "Handles correlated features better than Lasso".into(),
            ],
            limitations: vec![
                "Two hyperparameters to tune (alpha, l1_ratio)".into(),
                "Assumes linear relationship".into(),
            ],
            references: vec![
                "Zou, Hastie. Regularization and Variable Selection via the Elastic Net (2005)"
                    .into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::LogisticRegression {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "LogisticRegression".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p*n_iter)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: true,
            strengths: vec![
                "Probabilistic output with calibrated probabilities".into(),
                "Interpretable via odds ratios".into(),
                "Multiple solvers (IRLS, L-BFGS)".into(),
            ],
            limitations: vec![
                "Assumes linear decision boundary".into(),
                "Sensitive to feature scaling".into(),
            ],
            references: vec!["Cox. The Regression Analysis of Binary Sequences (1958)".into()],
        }
    }
}

impl HasModelCard for crate::models::sgd::SGDClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "SGDClassifier".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p*n_iter)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: true,
            supports_sample_weight: false,
            strengths: vec![
                "Scales to very large datasets via mini-batch updates".into(),
                "Supports multiple loss functions (hinge, log, etc.)".into(),
                "Online learning via partial_fit".into(),
            ],
            limitations: vec![
                "Sensitive to feature scaling".into(),
                "Requires tuning of learning rate and regularization".into(),
            ],
            references: vec![
                "Bottou. Large-Scale Machine Learning with Stochastic Gradient Descent (2010)"
                    .into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::sgd::SGDRegressor {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "SGDRegressor".into(),
            task: vec!["regression".into()],
            complexity: "O(n*p*n_iter)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: true,
            supports_sample_weight: false,
            strengths: vec![
                "Scales to very large datasets".into(),
                "Online learning via partial_fit".into(),
            ],
            limitations: vec![
                "Sensitive to feature scaling".into(),
                "Requires learning rate tuning".into(),
            ],
            references: vec![
                "Bottou. Large-Scale Machine Learning with Stochastic Gradient Descent (2010)"
                    .into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::quantile::QuantileRegression {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "QuantileRegression".into(),
            task: vec!["regression".into()],
            complexity: "O(n*p*n_iter)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Models conditional quantiles, not just the mean".into(),
                "Robust to outliers in the response variable".into(),
            ],
            limitations: vec![
                "Slower convergence than OLS".into(),
                "Crossing quantiles possible without monotonicity constraints".into(),
            ],
            references: vec!["Koenker, Bassett. Regression Quantiles (1978)".into()],
        }
    }
}

impl HasModelCard for crate::models::robust::RobustRegression {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "RobustRegression".into(),
            task: vec!["regression".into()],
            complexity: "O(n*p*n_iter)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Resistant to outliers via M-estimation".into(),
                "Multiple robust loss functions (Huber, Tukey, etc.)".into(),
            ],
            limitations: vec![
                "Slower than OLS".into(),
                "May converge to local minimum".into(),
            ],
            references: vec!["Huber. Robust Statistics (1981)".into()],
        }
    }
}

impl HasModelCard for crate::models::sgd::Perceptron {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "Perceptron".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p*n_iter)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: true,
            supports_sample_weight: false,
            strengths: vec![
                "Simple and fast linear classifier".into(),
                "Online learning via partial_fit".into(),
            ],
            limitations: vec![
                "Only converges for linearly separable data".into(),
                "No probabilistic output".into(),
            ],
            references: vec!["Rosenblatt. The Perceptron (1958)".into()],
        }
    }
}

impl HasModelCard for crate::models::regularized::RidgeCV {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "RidgeCV".into(),
            task: vec!["regression".into()],
            complexity: "O(n*p^2*n_alphas)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Automatic alpha selection via cross-validation".into(),
                "Efficient leave-one-out CV for Ridge".into(),
            ],
            limitations: vec![
                "Does not perform feature selection".into(),
                "Assumes linear relationship".into(),
            ],
            references: vec!["Golub, Heath, Wahba. Generalized Cross-Validation (1979)".into()],
        }
    }
}

impl HasModelCard for crate::models::regularized::LassoCV {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "LassoCV".into(),
            task: vec!["regression".into()],
            complexity: "O(n*p*n_iter*n_alphas)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Automatic alpha selection via cross-validation".into(),
                "Sparse feature selection".into(),
            ],
            limitations: vec![
                "Slower than RidgeCV".into(),
                "Unstable for correlated features".into(),
            ],
            references: vec![
                "Tibshirani. Regression Shrinkage and Selection via the Lasso (1996)".into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::regularized::ElasticNetCV {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "ElasticNetCV".into(),
            task: vec!["regression".into()],
            complexity: "O(n*p*n_iter*n_alphas)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Automatic alpha and l1_ratio selection".into(),
                "Handles correlated features".into(),
            ],
            limitations: vec![
                "Computationally expensive grid search".into(),
                "Assumes linear relationship".into(),
            ],
            references: vec![
                "Zou, Hastie. Regularization and Variable Selection via the Elastic Net (2005)"
                    .into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::regularized::RidgeClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "RidgeClassifier".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p^2)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Fast classification via Ridge regression on targets".into(),
                "No iterative optimization needed".into(),
            ],
            limitations: vec![
                "No probabilistic output".into(),
                "Linear decision boundary".into(),
            ],
            references: vec![
                "Rifkin, Klautau. In Defense of One-Vs-All Classification (2004)".into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::isotonic::IsotonicRegression {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "IsotonicRegression".into(),
            task: vec!["regression".into()],
            complexity: "O(n)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Non-parametric monotonic regression".into(),
                "No assumptions about functional form".into(),
            ],
            limitations: vec![
                "Only works with 1D input".into(),
                "Can overfit with noisy data".into(),
            ],
            references: vec![
                "Barlow et al. Statistical Inference Under Order Restrictions (1972)".into(),
            ],
        }
    }
}

// =============================================================================
// Tree models
// =============================================================================

impl HasModelCard for crate::models::tree::DecisionTreeClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "DecisionTreeClassifier".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p*log(n))".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: true,
            strengths: vec![
                "Highly interpretable decision rules".into(),
                "Handles non-linear relationships".into(),
                "No feature scaling required".into(),
            ],
            limitations: vec![
                "Prone to overfitting without pruning".into(),
                "Unstable — small data changes can produce different trees".into(),
            ],
            references: vec!["Breiman et al. Classification and Regression Trees (1984)".into()],
        }
    }
}

impl HasModelCard for crate::models::tree::DecisionTreeRegressor {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "DecisionTreeRegressor".into(),
            task: vec!["regression".into()],
            complexity: "O(n*p*log(n))".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Highly interpretable decision rules".into(),
                "Captures non-linear relationships".into(),
                "No feature scaling required".into(),
            ],
            limitations: vec![
                "Prone to overfitting".into(),
                "Produces step-function predictions".into(),
            ],
            references: vec!["Breiman et al. Classification and Regression Trees (1984)".into()],
        }
    }
}

impl HasModelCard for crate::models::forest::RandomForestClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "RandomForestClassifier".into(),
            task: vec!["classification".into()],
            complexity: "O(n_trees*n*p*log(n))".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Robust to overfitting via bagging and feature subsampling".into(),
                "Built-in feature importance with confidence intervals".into(),
                "Handles high-dimensional data well".into(),
            ],
            limitations: vec![
                "Less interpretable than single trees".into(),
                "Slow for very large ensembles".into(),
            ],
            references: vec!["Breiman. Random Forests (2001)".into()],
        }
    }
}

impl HasModelCard for crate::models::forest::RandomForestRegressor {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "RandomForestRegressor".into(),
            task: vec!["regression".into()],
            complexity: "O(n_trees*n*p*log(n))".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Robust to overfitting via bagging".into(),
                "Built-in feature importance".into(),
                "No feature scaling required".into(),
            ],
            limitations: vec![
                "Cannot extrapolate beyond training range".into(),
                "Memory-intensive for large ensembles".into(),
            ],
            references: vec!["Breiman. Random Forests (2001)".into()],
        }
    }
}

impl HasModelCard for crate::models::extra_trees::ExtraTreesClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "ExtraTreesClassifier".into(),
            task: vec!["classification".into()],
            complexity: "O(n_trees*n*p)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Faster than Random Forest (random splits, no sorting)".into(),
                "Lower variance than Random Forest".into(),
            ],
            limitations: vec![
                "Slightly higher bias than Random Forest".into(),
                "Less interpretable than single trees".into(),
            ],
            references: vec!["Geurts, Ernst, Wehenkel. Extremely Randomized Trees (2006)".into()],
        }
    }
}

impl HasModelCard for crate::models::extra_trees::ExtraTreesRegressor {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "ExtraTreesRegressor".into(),
            task: vec!["regression".into()],
            complexity: "O(n_trees*n*p)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Faster than Random Forest".into(),
                "Lower variance than Random Forest".into(),
            ],
            limitations: vec![
                "Slightly higher bias".into(),
                "Cannot extrapolate beyond training range".into(),
            ],
            references: vec!["Geurts, Ernst, Wehenkel. Extremely Randomized Trees (2006)".into()],
        }
    }
}

impl HasModelCard for crate::models::boosting::GradientBoostingClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "GradientBoostingClassifier".into(),
            task: vec!["classification".into()],
            complexity: "O(n_trees*n*p*log(n))".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "State-of-the-art accuracy on tabular data".into(),
                "Handles mixed feature types".into(),
                "Built-in early stopping".into(),
            ],
            limitations: vec![
                "Slow to train sequentially".into(),
                "Sensitive to hyperparameters".into(),
            ],
            references: vec![
                "Friedman. Greedy Function Approximation: A Gradient Boosting Machine (2001)"
                    .into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::boosting::GradientBoostingRegressor {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "GradientBoostingRegressor".into(),
            task: vec!["regression".into()],
            complexity: "O(n_trees*n*p*log(n))".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "State-of-the-art accuracy on tabular data".into(),
                "Multiple loss functions (MSE, MAE, Huber)".into(),
                "Built-in early stopping".into(),
            ],
            limitations: vec![
                "Sequential training is slow".into(),
                "Sensitive to hyperparameters".into(),
            ],
            references: vec![
                "Friedman. Greedy Function Approximation: A Gradient Boosting Machine (2001)"
                    .into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::hist_boosting::HistGradientBoostingClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "HistGradientBoostingClassifier".into(),
            task: vec!["classification".into()],
            complexity: "O(n_trees*n_bins*p)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Much faster than GradientBoosting for large datasets".into(),
                "Native support for missing values".into(),
                "Histogram-based splits reduce overfitting".into(),
            ],
            limitations: vec![
                "Binning approximation can lose precision".into(),
                "More hyperparameters than standard boosting".into(),
            ],
            references: vec![
                "Ke et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree (2017)"
                    .into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::hist_boosting::HistGradientBoostingRegressor {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "HistGradientBoostingRegressor".into(),
            task: vec!["regression".into()],
            complexity: "O(n_trees*n_bins*p)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Much faster than GradientBoosting for large datasets".into(),
                "Native missing value support".into(),
            ],
            limitations: vec![
                "Binning approximation can lose precision".into(),
                "Sequential training".into(),
            ],
            references: vec!["Ke et al. LightGBM (2017)".into()],
        }
    }
}

impl HasModelCard for crate::models::adaboost::AdaBoostClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "AdaBoostClassifier".into(),
            task: vec!["classification".into()],
            complexity: "O(n_trees*n*p)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Adaptive boosting focuses on hard examples".into(),
                "Less prone to overfitting than other boosting methods".into(),
            ],
            limitations: vec![
                "Sensitive to noisy data and outliers".into(),
                "Weak learner must be better than random".into(),
            ],
            references: vec![
                "Freund, Schapire. A Decision-Theoretic Generalization of On-Line Learning (1997)"
                    .into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::adaboost::AdaBoostRegressor {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "AdaBoostRegressor".into(),
            task: vec!["regression".into()],
            complexity: "O(n_trees*n*p)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Focuses on hard-to-predict samples".into(),
                "Works with any base regressor".into(),
            ],
            limitations: vec!["Sensitive to outliers".into(), "Sequential training".into()],
            references: vec![
                "Drucker. Improving Regressors using Boosting Techniques (1997)".into(),
            ],
        }
    }
}

// =============================================================================
// SVM
// =============================================================================

impl HasModelCard for crate::models::svm::SVC {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "SVC".into(),
            task: vec!["classification".into()],
            complexity: "O(n^2*p)".into(),
            interpretability: "low".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Effective in high-dimensional spaces".into(),
                "Kernel trick enables non-linear boundaries".into(),
                "Robust to overfitting in high dimensions".into(),
            ],
            limitations: vec![
                "Slow for large datasets (quadratic scaling)".into(),
                "Sensitive to feature scaling".into(),
                "No direct probability estimates (requires calibration)".into(),
            ],
            references: vec!["Cortes, Vapnik. Support-Vector Networks (1995)".into()],
        }
    }
}

impl HasModelCard for crate::models::svm::SVR {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "SVR".into(),
            task: vec!["regression".into()],
            complexity: "O(n^2*p)".into(),
            interpretability: "low".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Epsilon-insensitive loss for robustness".into(),
                "Kernel trick for non-linear regression".into(),
            ],
            limitations: vec![
                "Slow for large datasets".into(),
                "Sensitive to feature scaling and hyperparameters".into(),
            ],
            references: vec!["Drucker et al. Support Vector Regression Machines (1996)".into()],
        }
    }
}

impl HasModelCard for crate::models::svm::LinearSVC {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "LinearSVC".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Fast linear SVM via coordinate descent".into(),
                "Scales well to large datasets".into(),
            ],
            limitations: vec![
                "Linear decision boundary only".into(),
                "No kernel support".into(),
            ],
            references: vec!["Fan et al. LIBLINEAR (2008)".into()],
        }
    }
}

impl HasModelCard for crate::models::svm::LinearSVR {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "LinearSVR".into(),
            task: vec!["regression".into()],
            complexity: "O(n*p)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Fast linear SVR via coordinate descent".into(),
                "Scales well to large datasets".into(),
            ],
            limitations: vec!["Linear model only".into(), "No kernel support".into()],
            references: vec!["Fan et al. LIBLINEAR (2008)".into()],
        }
    }
}

// =============================================================================
// KNN
// =============================================================================

impl HasModelCard for crate::models::knn::KNeighborsClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "KNeighborsClassifier".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p) per query".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Non-parametric — no training phase".into(),
                "Naturally handles multi-class".into(),
                "Simple and intuitive".into(),
            ],
            limitations: vec![
                "Slow prediction for large datasets".into(),
                "Sensitive to feature scaling and irrelevant features".into(),
                "Memory-intensive (stores all training data)".into(),
            ],
            references: vec!["Cover, Hart. Nearest Neighbor Pattern Classification (1967)".into()],
        }
    }
}

impl HasModelCard for crate::models::knn::KNeighborsRegressor {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "KNeighborsRegressor".into(),
            task: vec!["regression".into()],
            complexity: "O(n*p) per query".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Non-parametric — captures any shape".into(),
                "No training phase".into(),
            ],
            limitations: vec![
                "Slow prediction for large datasets".into(),
                "Sensitive to feature scaling".into(),
                "Cannot extrapolate".into(),
            ],
            references: vec!["Cover, Hart. Nearest Neighbor Pattern Classification (1967)".into()],
        }
    }
}

impl HasModelCard for crate::models::knn::NearestCentroid {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "NearestCentroid".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Very fast training and prediction".into(),
                "Simple — one centroid per class".into(),
            ],
            limitations: vec![
                "Assumes convex, similarly-shaped classes".into(),
                "No probability output".into(),
            ],
            references: vec![
                "Tibshirani et al. Diagnosis of multiple cancer types by shrunken centroids (2002)"
                    .into(),
            ],
        }
    }
}

// =============================================================================
// Naive Bayes
// =============================================================================

impl HasModelCard for crate::models::naive_bayes::GaussianNB {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "GaussianNB".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: true,
            supports_sample_weight: false,
            strengths: vec![
                "Extremely fast training and prediction".into(),
                "Works well with small datasets".into(),
                "Supports online learning via partial_fit".into(),
            ],
            limitations: vec![
                "Assumes feature independence".into(),
                "Assumes Gaussian feature distributions".into(),
            ],
            references: vec![
                "McCallum, Nigam. A Comparison of Event Models for Naive Bayes Text Classification (1998)".into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::naive_bayes::MultinomialNB {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "MultinomialNB".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: true,
            supports_sample_weight: false,
            strengths: vec![
                "Well-suited for text classification with word counts".into(),
                "Extremely fast".into(),
                "Supports online learning".into(),
            ],
            limitations: vec![
                "Requires non-negative features (counts)".into(),
                "Assumes feature independence".into(),
            ],
            references: vec![
                "McCallum, Nigam. A Comparison of Event Models for Naive Bayes Text Classification (1998)".into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::naive_bayes::BernoulliNB {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "BernoulliNB".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: true,
            supports_sample_weight: false,
            strengths: vec![
                "Designed for binary/boolean features".into(),
                "Penalizes absence of features (unlike Multinomial)".into(),
                "Supports online learning".into(),
            ],
            limitations: vec![
                "Assumes binary features".into(),
                "Assumes feature independence".into(),
            ],
            references: vec!["McCallum, Nigam (1998)".into()],
        }
    }
}

impl HasModelCard for crate::models::naive_bayes::CategoricalNB {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "CategoricalNB".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: true,
            supports_sample_weight: false,
            strengths: vec![
                "Designed for categorical features".into(),
                "Handles multi-valued discrete features".into(),
                "Supports online learning".into(),
            ],
            limitations: vec![
                "Assumes feature independence".into(),
                "Requires integer-encoded categorical features".into(),
            ],
            references: vec!["McCallum, Nigam (1998)".into()],
        }
    }
}

// =============================================================================
// Neural
// =============================================================================

impl HasModelCard for crate::neural::MLPClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "MLPClassifier".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p*h*n_iter)".into(),
            interpretability: "low".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Universal function approximator".into(),
                "Handles complex non-linear relationships".into(),
                "Multiple activation functions and architectures".into(),
            ],
            limitations: vec![
                "Requires careful hyperparameter tuning".into(),
                "Sensitive to feature scaling".into(),
                "Can get stuck in local minima".into(),
            ],
            references: vec![
                "Rumelhart, Hinton, Williams. Learning Representations by Back-propagating Errors (1986)".into(),
            ],
        }
    }
}

impl HasModelCard for crate::neural::MLPRegressor {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "MLPRegressor".into(),
            task: vec!["regression".into()],
            complexity: "O(n*p*h*n_iter)".into(),
            interpretability: "low".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Universal function approximator".into(),
                "Handles complex non-linear relationships".into(),
            ],
            limitations: vec![
                "Requires careful hyperparameter tuning".into(),
                "Sensitive to feature scaling".into(),
                "Can get stuck in local minima".into(),
            ],
            references: vec![
                "Rumelhart, Hinton, Williams. Learning Representations by Back-propagating Errors (1986)".into(),
            ],
        }
    }
}

// =============================================================================
// Gaussian Process
// =============================================================================

impl HasModelCard for crate::models::gaussian_process::GaussianProcessRegressor {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "GaussianProcessRegressor".into(),
            task: vec!["regression".into()],
            complexity: "O(n^3)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Provides uncertainty estimates (posterior variance)".into(),
                "Non-parametric — adapts complexity to data".into(),
                "Kernel-based for flexible modeling".into(),
            ],
            limitations: vec![
                "Cubic scaling limits to ~10k samples".into(),
                "Kernel selection requires domain knowledge".into(),
            ],
            references: vec![
                "Rasmussen, Williams. Gaussian Processes for Machine Learning (2006)".into(),
            ],
        }
    }
}

impl HasModelCard for crate::models::gaussian_process::GaussianProcessClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "GaussianProcessClassifier".into(),
            task: vec!["classification".into()],
            complexity: "O(n^3)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Probabilistic classification with uncertainty".into(),
                "Non-parametric with kernel flexibility".into(),
            ],
            limitations: vec![
                "Cubic scaling limits to small-medium datasets".into(),
                "Laplace approximation for non-Gaussian likelihood".into(),
            ],
            references: vec![
                "Rasmussen, Williams. Gaussian Processes for Machine Learning (2006)".into(),
            ],
        }
    }
}

// =============================================================================
// QDA and Isotonic (already covered above under linear)
// =============================================================================

impl HasModelCard for crate::models::qda::QuadraticDiscriminantAnalysis {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "QuadraticDiscriminantAnalysis".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p^2)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Quadratic decision boundaries per class".into(),
                "No iterative optimization — closed-form".into(),
            ],
            limitations: vec![
                "Requires sufficient samples per class for covariance estimation".into(),
                "Assumes Gaussian class distributions".into(),
            ],
            references: vec![
                "Fisher. The Use of Multiple Measurements in Taxonomic Problems (1936)".into(),
            ],
        }
    }
}

// =============================================================================
// Clustering
// =============================================================================

impl HasModelCard for crate::clustering::KMeans {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "KMeans".into(),
            task: vec!["clustering".into()],
            complexity: "O(n*k*p*n_iter)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Fast and scalable".into(),
                "Simple and well-understood".into(),
                "Elkan acceleration for faster convergence".into(),
            ],
            limitations: vec![
                "Requires specifying k in advance".into(),
                "Assumes spherical clusters of similar size".into(),
                "Sensitive to initialization".into(),
            ],
            references: vec!["Lloyd. Least Squares Quantization in PCM (1982)".into()],
        }
    }
}

impl HasModelCard for crate::clustering::MiniBatchKMeans {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "MiniBatchKMeans".into(),
            task: vec!["clustering".into()],
            complexity: "O(n*k*p)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: true,
            supports_sample_weight: false,
            strengths: vec![
                "Much faster than standard KMeans for large datasets".into(),
                "Supports online/incremental learning".into(),
            ],
            limitations: vec![
                "Slightly worse cluster quality than full KMeans".into(),
                "Requires specifying k in advance".into(),
            ],
            references: vec!["Sculley. Web-Scale K-Means Clustering (2010)".into()],
        }
    }
}

impl HasModelCard for crate::clustering::DBSCAN {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "DBSCAN".into(),
            task: vec!["clustering".into()],
            complexity: "O(n*log(n))".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Discovers clusters of arbitrary shape".into(),
                "Automatically detects number of clusters".into(),
                "Identifies noise points".into(),
            ],
            limitations: vec![
                "Sensitive to eps and min_samples parameters".into(),
                "Struggles with varying-density clusters".into(),
            ],
            references: vec![
                "Ester et al. A Density-Based Algorithm for Discovering Clusters (KDD 1996)".into(),
            ],
        }
    }
}

impl HasModelCard for crate::clustering::HDBSCAN {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "HDBSCAN".into(),
            task: vec!["clustering".into()],
            complexity: "O(n*log(n))".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Handles varying-density clusters".into(),
                "Fewer hyperparameters than DBSCAN".into(),
                "Provides cluster stability scores".into(),
            ],
            limitations: vec![
                "Slower than DBSCAN".into(),
                "Memory-intensive for large datasets".into(),
            ],
            references: vec![
                "Campello, Moulavi, Sander. Density-Based Clustering Based on Hierarchical Density Estimates (2013)".into(),
            ],
        }
    }
}

impl HasModelCard for crate::clustering::AgglomerativeClustering {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "AgglomerativeClustering".into(),
            task: vec!["clustering".into()],
            complexity: "O(n^3)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Produces a hierarchy (dendrogram) of clusters".into(),
                "No need to specify k in advance".into(),
                "Multiple linkage criteria".into(),
            ],
            limitations: vec![
                "Cubic time complexity".into(),
                "Cannot undo merges (greedy)".into(),
            ],
            references: vec![
                "Murtagh, Contreras. Algorithms for Hierarchical Clustering: An Overview (2012)"
                    .into(),
            ],
        }
    }
}

impl HasModelCard for crate::clustering::GaussianMixture {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "GaussianMixture".into(),
            task: vec!["clustering".into()],
            complexity: "O(n*k*p^2*n_iter)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Soft (probabilistic) cluster assignments".into(),
                "Models elliptical cluster shapes".into(),
                "BIC/AIC for model selection".into(),
            ],
            limitations: vec![
                "Sensitive to initialization".into(),
                "Assumes Gaussian-distributed clusters".into(),
                "Requires specifying number of components".into(),
            ],
            references: vec![
                "Dempster, Laird, Rubin. Maximum Likelihood from Incomplete Data via the EM Algorithm (1977)".into(),
            ],
        }
    }
}

// =============================================================================
// Decomposition
// =============================================================================

impl HasModelCard for crate::decomposition::PCA {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "PCA".into(),
            task: vec!["dimensionality_reduction".into()],
            complexity: "O(n*p^2)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Finds directions of maximum variance".into(),
                "Explained variance ratio for component selection".into(),
                "Deterministic and fast".into(),
            ],
            limitations: vec![
                "Linear transformation only".into(),
                "Sensitive to feature scaling".into(),
            ],
            references: vec!["Pearson. On Lines and Planes of Closest Fit (1901)".into()],
        }
    }
}

impl HasModelCard for crate::decomposition::IncrementalPCA {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "IncrementalPCA".into(),
            task: vec!["dimensionality_reduction".into()],
            complexity: "O(n*p*k)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: true,
            supports_sample_weight: false,
            strengths: vec![
                "Handles datasets that do not fit in memory".into(),
                "Online/streaming PCA via partial_fit".into(),
            ],
            limitations: vec![
                "Approximate — results differ from batch PCA".into(),
                "Linear transformation only".into(),
            ],
            references: vec![
                "Ross et al. Incremental Learning for Robust Visual Tracking (2008)".into(),
            ],
        }
    }
}

impl HasModelCard for crate::decomposition::TruncatedSVD {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "TruncatedSVD".into(),
            task: vec!["dimensionality_reduction".into()],
            complexity: "O(n*p*k)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Works on sparse matrices (unlike PCA)".into(),
                "Latent Semantic Analysis (LSA) for text".into(),
            ],
            limitations: vec![
                "Linear transformation only".into(),
                "Randomized — results may vary between runs".into(),
            ],
            references: vec![
                "Halko, Martinsson, Tropp. Finding Structure with Randomness (2011)".into(),
            ],
        }
    }
}

impl HasModelCard for crate::decomposition::FactorAnalysis {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "FactorAnalysis".into(),
            task: vec!["dimensionality_reduction".into()],
            complexity: "O(n*p*k*n_iter)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Models latent factors with per-feature noise".into(),
                "Probabilistic model with log-likelihood".into(),
            ],
            limitations: vec![
                "Assumes linear generative model".into(),
                "Rotational indeterminacy of factors".into(),
            ],
            references: vec!["Rubin, Thayer. EM Algorithms for ML Factor Analysis (1982)".into()],
        }
    }
}

impl HasModelCard for crate::decomposition::LDA {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "LDA".into(),
            task: vec!["dimensionality_reduction".into()],
            complexity: "O(n*k*p*n_iter)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Discovers latent topics in text data".into(),
                "Probabilistic generative model".into(),
            ],
            limitations: vec![
                "Requires specifying number of topics".into(),
                "Slow for large corpora".into(),
            ],
            references: vec!["Blei, Ng, Jordan. Latent Dirichlet Allocation (2003)".into()],
        }
    }
}

impl HasModelCard for crate::decomposition::TSNE {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "TSNE".into(),
            task: vec!["dimensionality_reduction".into()],
            complexity: "O(n*log(n))".into(),
            interpretability: "low".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Excellent for 2D/3D visualization".into(),
                "Preserves local structure".into(),
                "Barnes-Hut approximation for scalability".into(),
            ],
            limitations: vec![
                "Non-deterministic (depends on initialization)".into(),
                "Global distances are not preserved".into(),
                "Cannot transform new data".into(),
            ],
            references: vec!["van der Maaten, Hinton. Visualizing Data using t-SNE (2008)".into()],
        }
    }
}

// =============================================================================
// Outlier Detection
// =============================================================================

impl HasModelCard for crate::models::isolation_forest::IsolationForest {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "IsolationForest".into(),
            task: vec!["outlier_detection".into()],
            complexity: "O(n*n_trees*log(n))".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Linear time complexity".into(),
                "No distance or density computation needed".into(),
                "Works well in high dimensions".into(),
            ],
            limitations: vec![
                "Contamination parameter must be estimated".into(),
                "Struggles with local anomalies".into(),
            ],
            references: vec!["Liu, Ting, Zhou. Isolation Forest (ICDM 2008)".into()],
        }
    }
}

impl HasModelCard for crate::models::lof::LocalOutlierFactor {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "LocalOutlierFactor".into(),
            task: vec!["outlier_detection".into()],
            complexity: "O(n^2)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Detects local anomalies in varying-density data".into(),
                "Provides anomaly scores (LOF values)".into(),
            ],
            limitations: vec![
                "Quadratic time complexity".into(),
                "Sensitive to n_neighbors parameter".into(),
            ],
            references: vec![
                "Breunig et al. LOF: Identifying Density-Based Local Outliers (SIGMOD 2000)".into(),
            ],
        }
    }
}

// =============================================================================
// Calibration
// =============================================================================

impl HasModelCard for crate::models::calibration::TemperatureScalingCalibrator {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "TemperatureScalingCalibrator".into(),
            task: vec!["calibration".into()],
            complexity: "O(n*n_iter)".into(),
            interpretability: "high".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Single-parameter calibration — minimal overfitting risk".into(),
                "Effective for neural network overconfidence".into(),
            ],
            limitations: vec![
                "Only scales logits uniformly".into(),
                "Requires held-out calibration set".into(),
            ],
            references: vec![
                "Guo et al. On Calibration of Modern Neural Networks (ICML 2017)".into(),
            ],
        }
    }
}

// =============================================================================
// Ensemble wrappers
// =============================================================================

impl HasModelCard for crate::ensemble::VotingClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "VotingClassifier".into(),
            task: vec!["classification".into(), "ensemble".into()],
            complexity: "O(sum of base model complexities)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Combines diverse classifiers for better accuracy".into(),
                "Supports hard and soft voting".into(),
            ],
            limitations: vec![
                "No better than best base model if models are correlated".into(),
                "Slow if base models are slow".into(),
            ],
            references: vec!["Dietterich. Ensemble Methods in Machine Learning (2000)".into()],
        }
    }
}

impl HasModelCard for crate::ensemble::VotingRegressor {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "VotingRegressor".into(),
            task: vec!["regression".into(), "ensemble".into()],
            complexity: "O(sum of base model complexities)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Averages predictions from diverse regressors".into(),
                "Reduces variance of individual models".into(),
            ],
            limitations: vec![
                "Limited gain if base models are correlated".into(),
                "Cannot be better than best base model on average".into(),
            ],
            references: vec!["Dietterich. Ensemble Methods in Machine Learning (2000)".into()],
        }
    }
}

impl HasModelCard for crate::ensemble::stacking::StackingClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "StackingClassifier".into(),
            task: vec!["classification".into(), "ensemble".into()],
            complexity: "O(sum of base + meta model complexities)".into(),
            interpretability: "low".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Learns optimal combination of base model predictions".into(),
                "Often outperforms voting".into(),
            ],
            limitations: vec![
                "Risk of overfitting without proper CV".into(),
                "Computationally expensive (trains models twice)".into(),
            ],
            references: vec!["Wolpert. Stacked Generalization (1992)".into()],
        }
    }
}

impl HasModelCard for crate::ensemble::stacking::StackingRegressor {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "StackingRegressor".into(),
            task: vec!["regression".into(), "ensemble".into()],
            complexity: "O(sum of base + meta model complexities)".into(),
            interpretability: "low".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Learns optimal combination of base model predictions".into(),
                "Often outperforms simple averaging".into(),
            ],
            limitations: vec![
                "Risk of overfitting".into(),
                "Computationally expensive".into(),
            ],
            references: vec!["Wolpert. Stacked Generalization (1992)".into()],
        }
    }
}

impl HasModelCard for crate::ensemble::BaggingClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "BaggingClassifier".into(),
            task: vec!["classification".into(), "ensemble".into()],
            complexity: "O(n_estimators * base_complexity)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Reduces variance via bootstrap aggregation".into(),
                "Works with any base classifier".into(),
                "Parallelizable".into(),
            ],
            limitations: vec![
                "Less interpretable than single models".into(),
                "Does not reduce bias".into(),
            ],
            references: vec!["Breiman. Bagging Predictors (1996)".into()],
        }
    }
}

impl HasModelCard for crate::ensemble::BaggingRegressor {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "BaggingRegressor".into(),
            task: vec!["regression".into(), "ensemble".into()],
            complexity: "O(n_estimators * base_complexity)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: false,
            supports_sample_weight: false,
            strengths: vec![
                "Reduces variance via bootstrap aggregation".into(),
                "Works with any base regressor".into(),
                "Parallelizable".into(),
            ],
            limitations: vec![
                "Less interpretable than single models".into(),
                "Does not reduce bias".into(),
            ],
            references: vec!["Breiman. Bagging Predictors (1996)".into()],
        }
    }
}

// =============================================================================
// Multi-output (generic types — standalone functions instead of trait impls)
// =============================================================================

/// Model card for `MultiOutputClassifier` (generic wrapper, cannot use trait).
pub fn multi_output_classifier_card() -> ModelCard {
    ModelCard {
        name: "MultiOutputClassifier".into(),
        task: vec!["classification".into(), "ensemble".into()],
        complexity: "O(n_outputs * base_complexity)".into(),
        interpretability: "medium".into(),
        supports_sparse: false,
        supports_incremental: false,
        supports_sample_weight: false,
        strengths: vec![
            "Wraps any classifier for multi-output problems".into(),
            "Independent models per output".into(),
        ],
        limitations: vec![
            "Does not capture output correlations".into(),
            "Linear scaling with number of outputs".into(),
        ],
        references: vec![
            "Tsoumakas, Katakis. Multi-Label Classification: An Overview (2007)".into(),
        ],
    }
}

/// Model card for `MultiOutputRegressor` (generic wrapper, cannot use trait).
pub fn multi_output_regressor_card() -> ModelCard {
    ModelCard {
        name: "MultiOutputRegressor".into(),
        task: vec!["regression".into(), "ensemble".into()],
        complexity: "O(n_outputs * base_complexity)".into(),
        interpretability: "medium".into(),
        supports_sparse: false,
        supports_incremental: false,
        supports_sample_weight: false,
        strengths: vec![
            "Wraps any regressor for multi-output problems".into(),
            "Independent models per output".into(),
        ],
        limitations: vec![
            "Does not capture output correlations".into(),
            "Linear scaling with number of outputs".into(),
        ],
        references: vec!["Borchani et al. A Survey on Multi-output Regression (2015)".into()],
    }
}

// =============================================================================
// SGD: PassiveAggressiveClassifier
// =============================================================================

impl HasModelCard for crate::models::sgd::PassiveAggressiveClassifier {
    fn model_card() -> ModelCard {
        ModelCard {
            name: "PassiveAggressiveClassifier".into(),
            task: vec!["classification".into()],
            complexity: "O(n*p*n_iter)".into(),
            interpretability: "medium".into(),
            supports_sparse: false,
            supports_incremental: true,
            supports_sample_weight: false,
            strengths: vec![
                "Online learning — updates only on misclassified samples".into(),
                "No learning rate parameter".into(),
            ],
            limitations: vec![
                "Linear decision boundary".into(),
                "Sensitive to noisy labels".into(),
            ],
            references: vec![
                "Crammer et al. Online Passive-Aggressive Algorithms (JMLR 2006)".into(),
            ],
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression_card() {
        let card = crate::models::LinearRegression::model_card();
        assert_eq!(card.name, "LinearRegression");
        assert_eq!(card.task, vec!["regression"]);
        assert_eq!(card.interpretability, "high");
        assert!(!card.strengths.is_empty());
        assert!(!card.limitations.is_empty());
        assert!(!card.references.is_empty());
    }

    #[test]
    fn test_random_forest_card() {
        let card = crate::models::RandomForestClassifier::model_card();
        assert_eq!(card.name, "RandomForestClassifier");
        assert_eq!(card.task, vec!["classification"]);
        assert_eq!(card.interpretability, "medium");
    }

    #[test]
    fn test_kmeans_card() {
        let card = crate::clustering::KMeans::model_card();
        assert_eq!(card.name, "KMeans");
        assert_eq!(card.task, vec!["clustering"]);
    }

    #[test]
    fn test_pca_card() {
        let card = crate::decomposition::PCA::model_card();
        assert_eq!(card.name, "PCA");
        assert_eq!(card.task, vec!["dimensionality_reduction"]);
    }

    #[test]
    fn test_svc_card() {
        let card = crate::models::SVC::model_card();
        assert_eq!(card.name, "SVC");
        assert_eq!(card.interpretability, "low");
    }

    #[test]
    fn test_mlp_card() {
        let card = crate::neural::MLPClassifier::model_card();
        assert_eq!(card.name, "MLPClassifier");
        assert_eq!(card.interpretability, "low");
    }

    #[test]
    fn test_isolation_forest_card() {
        let card = crate::models::IsolationForest::model_card();
        assert_eq!(card.name, "IsolationForest");
        assert_eq!(card.task, vec!["outlier_detection"]);
    }

    #[test]
    fn test_voting_classifier_card() {
        let card = crate::ensemble::VotingClassifier::model_card();
        assert_eq!(card.name, "VotingClassifier");
        assert!(card.task.contains(&"ensemble".to_string()));
    }

    #[test]
    fn test_gaussian_nb_card_incremental() {
        let card = crate::models::GaussianNB::model_card();
        assert!(card.supports_incremental);
    }

    #[test]
    fn test_sgd_classifier_card_incremental() {
        let card = crate::models::SGDClassifier::model_card();
        assert!(card.supports_incremental);
    }

    #[test]
    fn test_logistic_regression_sample_weight() {
        let card = crate::models::LogisticRegression::model_card();
        assert!(card.supports_sample_weight);
    }

    #[test]
    fn test_card_fields_not_empty() {
        // Test a representative set of models
        let cards: Vec<ModelCard> = vec![
            crate::models::LinearRegression::model_card(),
            crate::models::DecisionTreeClassifier::model_card(),
            crate::models::SVC::model_card(),
            crate::models::GaussianNB::model_card(),
            crate::neural::MLPClassifier::model_card(),
            crate::clustering::KMeans::model_card(),
            crate::decomposition::PCA::model_card(),
            crate::models::IsolationForest::model_card(),
        ];
        for card in &cards {
            assert!(!card.name.is_empty(), "name empty for {:?}", card.name);
            assert!(!card.task.is_empty(), "task empty for {}", card.name);
            assert!(
                !card.complexity.is_empty(),
                "complexity empty for {}",
                card.name
            );
            assert!(
                !card.interpretability.is_empty(),
                "interpretability empty for {}",
                card.name
            );
            assert!(
                !card.strengths.is_empty(),
                "strengths empty for {}",
                card.name
            );
            assert!(
                !card.limitations.is_empty(),
                "limitations empty for {}",
                card.name
            );
            assert!(
                !card.references.is_empty(),
                "references empty for {}",
                card.name
            );
        }
    }

    #[test]
    fn test_task_values_valid() {
        let valid_tasks = [
            "classification",
            "regression",
            "clustering",
            "dimensionality_reduction",
            "outlier_detection",
            "calibration",
            "ensemble",
        ];
        let cards: Vec<ModelCard> = vec![
            crate::models::LinearRegression::model_card(),
            crate::models::LogisticRegression::model_card(),
            crate::clustering::KMeans::model_card(),
            crate::decomposition::PCA::model_card(),
            crate::models::IsolationForest::model_card(),
            crate::models::calibration::TemperatureScalingCalibrator::model_card(),
            crate::ensemble::VotingClassifier::model_card(),
        ];
        for card in &cards {
            for task in &card.task {
                assert!(
                    valid_tasks.contains(&task.as_str()),
                    "Invalid task '{}' for model '{}'",
                    task,
                    card.name
                );
            }
        }
    }
}

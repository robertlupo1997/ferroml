//! Algorithm Portfolio for AutoML
//!
//! Defines portfolios of candidate algorithms with their search spaces and
//! preprocessing requirements for Combined Algorithm Selection and Hyperparameter
//! optimization (CASH).

use crate::hpo::search_space::ParameterDefault;
use crate::hpo::{Parameter, SearchSpace};
use crate::Task;
use serde::{Deserialize, Serialize};

/// Configuration for a single algorithm in the portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    /// Algorithm type/name
    pub algorithm: AlgorithmType,
    /// Search space for hyperparameters
    pub search_space: SearchSpace,
    /// Required preprocessing steps
    pub preprocessing: Vec<PreprocessingRequirement>,
    /// Relative priority (higher = try earlier)
    pub priority: u32,
    /// Estimated complexity (for time budget allocation)
    pub complexity: AlgorithmComplexity,
    /// Whether algorithm supports incremental/warm-start training
    pub supports_warm_start: bool,
}

impl AlgorithmConfig {
    /// Create a new algorithm configuration
    pub fn new(algorithm: AlgorithmType) -> Self {
        let (search_space, preprocessing, complexity, supports_warm_start) =
            Self::default_config(&algorithm);
        Self {
            algorithm,
            search_space,
            preprocessing,
            priority: 50,
            complexity,
            supports_warm_start,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set search space
    pub fn with_search_space(mut self, search_space: SearchSpace) -> Self {
        self.search_space = search_space;
        self
    }

    /// Add preprocessing requirement
    pub fn with_preprocessing(mut self, req: PreprocessingRequirement) -> Self {
        if !self.preprocessing.contains(&req) {
            self.preprocessing.push(req);
        }
        self
    }

    /// Get default configuration for an algorithm
    fn default_config(
        algorithm: &AlgorithmType,
    ) -> (
        SearchSpace,
        Vec<PreprocessingRequirement>,
        AlgorithmComplexity,
        bool,
    ) {
        match algorithm {
            // Linear models
            AlgorithmType::LogisticRegression => (
                SearchSpace::new()
                    .float_log("l2_penalty", 1e-5, 10.0)
                    .int("max_iter", 100, 1000),
                vec![
                    PreprocessingRequirement::Scaling,
                    PreprocessingRequirement::HandleMissing,
                ],
                AlgorithmComplexity::Low,
                false,
            ),
            AlgorithmType::LinearRegression => (
                SearchSpace::new(), // No hyperparameters for basic OLS
                vec![
                    PreprocessingRequirement::Scaling,
                    PreprocessingRequirement::HandleMissing,
                ],
                AlgorithmComplexity::Low,
                false,
            ),
            AlgorithmType::Ridge => (
                SearchSpace::new().float_log("alpha", 1e-4, 1e4),
                vec![
                    PreprocessingRequirement::Scaling,
                    PreprocessingRequirement::HandleMissing,
                ],
                AlgorithmComplexity::Low,
                false,
            ),
            AlgorithmType::Lasso => (
                SearchSpace::new().float_log("alpha", 1e-4, 1e4),
                vec![
                    PreprocessingRequirement::Scaling,
                    PreprocessingRequirement::HandleMissing,
                ],
                AlgorithmComplexity::Low,
                false,
            ),
            AlgorithmType::ElasticNet => (
                SearchSpace::new()
                    .float_log("alpha", 1e-4, 1e4)
                    .float("l1_ratio", 0.0, 1.0),
                vec![
                    PreprocessingRequirement::Scaling,
                    PreprocessingRequirement::HandleMissing,
                ],
                AlgorithmComplexity::Low,
                false,
            ),

            // Naive Bayes
            AlgorithmType::GaussianNB => (
                SearchSpace::new().float_log("var_smoothing", 1e-12, 1e-6),
                vec![PreprocessingRequirement::HandleMissing],
                AlgorithmComplexity::Low,
                true,
            ),
            AlgorithmType::MultinomialNB => (
                SearchSpace::new().float_log("alpha", 1e-3, 10.0),
                vec![
                    PreprocessingRequirement::NonNegative,
                    PreprocessingRequirement::HandleMissing,
                ],
                AlgorithmComplexity::Low,
                true,
            ),
            AlgorithmType::CategoricalNB => (
                SearchSpace::new()
                    .float_log("alpha", 0.01, 10.0)
                    .bool("fit_prior"),
                vec![
                    PreprocessingRequirement::NonNegative,
                    PreprocessingRequirement::HandleMissing,
                ],
                AlgorithmComplexity::Low,
                true,
            ),

            // KNN
            AlgorithmType::KNeighborsClassifier | AlgorithmType::KNeighborsRegressor => (
                SearchSpace::new()
                    .int("n_neighbors", 1, 50)
                    .categorical(
                        "weights",
                        vec!["uniform".to_string(), "distance".to_string()],
                    )
                    .categorical(
                        "metric",
                        vec!["euclidean".to_string(), "manhattan".to_string()],
                    ),
                vec![
                    PreprocessingRequirement::Scaling,
                    PreprocessingRequirement::HandleMissing,
                ],
                AlgorithmComplexity::Medium, // O(n) prediction
                false,
            ),

            // SVM
            AlgorithmType::SVC => (
                SearchSpace::new()
                    .float_log("C", 1e-3, 1e3)
                    .categorical(
                        "kernel",
                        vec!["linear".to_string(), "rbf".to_string(), "poly".to_string()],
                    )
                    .float_log("gamma", 1e-4, 10.0),
                vec![
                    PreprocessingRequirement::Scaling,
                    PreprocessingRequirement::HandleMissing,
                ],
                AlgorithmComplexity::High, // O(n^2) to O(n^3)
                false,
            ),
            AlgorithmType::SVR => (
                SearchSpace::new()
                    .float_log("C", 1e-3, 1e3)
                    .float_log("epsilon", 1e-3, 1.0)
                    .categorical(
                        "kernel",
                        vec!["linear".to_string(), "rbf".to_string(), "poly".to_string()],
                    )
                    .float_log("gamma", 1e-4, 10.0),
                vec![
                    PreprocessingRequirement::Scaling,
                    PreprocessingRequirement::HandleMissing,
                ],
                AlgorithmComplexity::High,
                false,
            ),
            AlgorithmType::LinearSVC => (
                SearchSpace::new().float_log("C", 1e-3, 1e3).categorical(
                    "loss",
                    vec!["hinge".to_string(), "squared_hinge".to_string()],
                ),
                vec![
                    PreprocessingRequirement::Scaling,
                    PreprocessingRequirement::HandleMissing,
                ],
                AlgorithmComplexity::Medium,
                false,
            ),
            AlgorithmType::LinearSVR => (
                SearchSpace::new()
                    .float_log("C", 1e-3, 1e3)
                    .float_log("epsilon", 1e-3, 1.0),
                vec![
                    PreprocessingRequirement::Scaling,
                    PreprocessingRequirement::HandleMissing,
                ],
                AlgorithmComplexity::Medium,
                false,
            ),

            // Decision Trees
            AlgorithmType::DecisionTreeClassifier | AlgorithmType::DecisionTreeRegressor => (
                SearchSpace::new()
                    .int("max_depth", 1, 30)
                    .int("min_samples_split", 2, 100)
                    .int("min_samples_leaf", 1, 50),
                vec![PreprocessingRequirement::HandleMissing],
                AlgorithmComplexity::Low,
                false,
            ),

            // Random Forest
            AlgorithmType::RandomForestClassifier | AlgorithmType::RandomForestRegressor => (
                SearchSpace::new()
                    .int("n_estimators", 10, 500)
                    .int("max_depth", 1, 30)
                    .int("min_samples_split", 2, 50)
                    .int("min_samples_leaf", 1, 20)
                    .categorical("max_features", vec!["sqrt".to_string(), "log2".to_string()])
                    .bool("bootstrap"),
                vec![PreprocessingRequirement::HandleMissing],
                AlgorithmComplexity::Medium,
                true, // warm_start support
            ),

            // Gradient Boosting
            AlgorithmType::GradientBoostingClassifier
            | AlgorithmType::GradientBoostingRegressor => (
                SearchSpace::new()
                    .int("n_estimators", 50, 500)
                    .float_log("learning_rate", 1e-3, 1.0)
                    .int("max_depth", 1, 10)
                    .float("subsample", 0.5, 1.0)
                    .int("min_samples_split", 2, 50)
                    .int("min_samples_leaf", 1, 20),
                vec![PreprocessingRequirement::HandleMissing],
                AlgorithmComplexity::High,
                true, // warm_start support
            ),

            // Histogram Gradient Boosting
            AlgorithmType::HistGradientBoostingClassifier
            | AlgorithmType::HistGradientBoostingRegressor => (
                SearchSpace::new()
                    .int("max_iter", 50, 500)
                    .float_log("learning_rate", 1e-3, 1.0)
                    .int("max_depth", 1, 15)
                    .int("max_leaf_nodes", 10, 100)
                    .float("l2_regularization", 0.0, 10.0)
                    .float("min_samples_leaf", 1.0, 100.0),
                vec![],                      // Handles missing values natively
                AlgorithmComplexity::Medium, // O(n) histogram-based
                true,                        // early stopping support
            ),

            // Quantile and Robust Regression
            AlgorithmType::QuantileRegression => (
                SearchSpace::new()
                    .float("quantile", 0.1, 0.9)
                    .int("max_iter", 100, 1000),
                vec![
                    PreprocessingRequirement::Scaling,
                    PreprocessingRequirement::HandleMissing,
                ],
                AlgorithmComplexity::Medium,
                false,
            ),
            AlgorithmType::RobustRegression => (
                SearchSpace::new()
                    .categorical(
                        "estimator",
                        vec![
                            "huber".to_string(),
                            "bisquare".to_string(),
                            "hampel".to_string(),
                        ],
                    )
                    .int("max_iter", 50, 500),
                vec![
                    PreprocessingRequirement::Scaling,
                    PreprocessingRequirement::HandleMissing,
                ],
                AlgorithmComplexity::Medium,
                false,
            ),
        }
    }

    /// Adapt search space based on data characteristics
    pub fn adapt_to_data(&self, characteristics: &DataCharacteristics) -> Self {
        let mut config = self.clone();
        let mut new_space = SearchSpace::new();

        for (name, param) in &self.search_space.parameters {
            let adapted = adapt_parameter(name, param, &self.algorithm, characteristics);
            new_space = new_space.add(name.clone(), adapted);
        }

        config.search_space = new_space;

        // Adjust preprocessing based on data
        if characteristics.has_missing_values {
            if !config
                .preprocessing
                .contains(&PreprocessingRequirement::HandleMissing)
            {
                config
                    .preprocessing
                    .push(PreprocessingRequirement::HandleMissing);
            }
        }

        if characteristics.has_categorical {
            if !config
                .preprocessing
                .contains(&PreprocessingRequirement::EncodeCategorical)
            {
                config
                    .preprocessing
                    .push(PreprocessingRequirement::EncodeCategorical);
            }
        }

        // Some algorithms need scaling regardless, but adjust for high variance
        if characteristics.feature_variance_ratio > 100.0 {
            if !config
                .preprocessing
                .contains(&PreprocessingRequirement::Scaling)
            {
                config.preprocessing.push(PreprocessingRequirement::Scaling);
            }
        }

        config
    }
}

/// Adapt a single parameter based on data characteristics
fn adapt_parameter(
    name: &str,
    param: &Parameter,
    algorithm: &AlgorithmType,
    chars: &DataCharacteristics,
) -> Parameter {
    use crate::hpo::ParameterType;

    match (&param.param_type, name) {
        // Adapt n_neighbors for KNN based on dataset size
        (ParameterType::Int { low, high }, "n_neighbors") => {
            let max_k = ((chars.n_samples as f64).sqrt().ceil() as i64).min(*high);
            let min_k = (*low).max(1);
            Parameter {
                param_type: ParameterType::Int {
                    low: min_k,
                    high: max_k.max(min_k + 1),
                },
                log_scale: param.log_scale,
                default: Some(ParameterDefault::Int((max_k / 2).max(min_k))),
            }
        }

        // Adapt n_estimators based on time budget expectations
        (ParameterType::Int { low, high }, "n_estimators" | "max_iter") => {
            // For large datasets, reduce upper bound
            let factor = if chars.n_samples > 100_000 {
                0.5
            } else if chars.n_samples > 10_000 {
                0.75
            } else {
                1.0
            };
            let adjusted_high = ((*high as f64) * factor).ceil() as i64;
            Parameter {
                param_type: ParameterType::Int {
                    low: *low,
                    high: adjusted_high.max(*low + 10),
                },
                log_scale: param.log_scale,
                default: param.default.clone(),
            }
        }

        // Adapt max_depth based on feature count
        (ParameterType::Int { low, high }, "max_depth") => {
            // Deeper trees for more features, but bounded
            let suggested_max = ((chars.n_features as f64).log2().ceil() as i64 * 3).min(*high);
            Parameter {
                param_type: ParameterType::Int {
                    low: *low,
                    high: suggested_max.max(*low + 2),
                },
                log_scale: param.log_scale,
                default: param.default.clone(),
            }
        }

        // Adapt regularization based on feature/sample ratio
        (ParameterType::Float { low, high }, name)
            if name.contains("alpha")
                || name.contains("C")
                || name.contains("l2")
                || name.contains("penalty") =>
        {
            // Higher regularization when features >> samples (risk of overfitting)
            let ratio = chars.n_features as f64 / chars.n_samples as f64;
            let (adjusted_low, adjusted_high) = if ratio > 0.5 {
                // Many features: bias toward stronger regularization
                ((*low * 10.0).min(*high), *high)
            } else if ratio < 0.01 {
                // Few features relative to samples: can use less regularization
                (*low, (*high * 0.1).max(*low))
            } else {
                (*low, *high)
            };
            Parameter {
                param_type: ParameterType::Float {
                    low: adjusted_low,
                    high: adjusted_high,
                },
                log_scale: param.log_scale,
                default: param.default.clone(),
            }
        }

        // Adapt learning rate based on dataset size
        (ParameterType::Float { low, high }, "learning_rate")
            if matches!(
                algorithm,
                AlgorithmType::GradientBoostingClassifier
                    | AlgorithmType::GradientBoostingRegressor
                    | AlgorithmType::HistGradientBoostingClassifier
                    | AlgorithmType::HistGradientBoostingRegressor
            ) =>
        {
            // Smaller learning rates for larger datasets (more updates)
            let factor = if chars.n_samples > 50_000 {
                0.1
            } else if chars.n_samples > 10_000 {
                0.5
            } else {
                1.0
            };
            Parameter {
                param_type: ParameterType::Float {
                    low: *low,
                    high: (*high * factor).max(*low * 10.0),
                },
                log_scale: true,
                default: param.default.clone(),
            }
        }

        // Keep other parameters as-is
        _ => param.clone(),
    }
}

/// Supported algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlgorithmType {
    // Classification - Linear
    /// Logistic Regression with L2 regularization
    LogisticRegression,

    // Classification - Probabilistic
    /// Gaussian Naive Bayes
    GaussianNB,
    /// Multinomial Naive Bayes
    MultinomialNB,
    /// Categorical Naive Bayes
    CategoricalNB,

    // Classification - Instance-based
    /// K-Nearest Neighbors Classifier
    KNeighborsClassifier,

    // Classification - SVM
    /// Support Vector Classification (kernelized)
    SVC,
    /// Linear SVM for classification
    LinearSVC,

    // Classification - Trees
    /// Decision Tree Classifier
    DecisionTreeClassifier,
    /// Random Forest Classifier
    RandomForestClassifier,
    /// Gradient Boosting Classifier
    GradientBoostingClassifier,
    /// Histogram-based Gradient Boosting Classifier
    HistGradientBoostingClassifier,

    // Regression - Linear
    /// Ordinary Least Squares
    LinearRegression,
    /// Ridge Regression (L2)
    Ridge,
    /// Lasso Regression (L1)
    Lasso,
    /// Elastic Net (L1 + L2)
    ElasticNet,
    /// Quantile Regression
    QuantileRegression,
    /// Robust Regression (M-estimators)
    RobustRegression,

    // Regression - Instance-based
    /// K-Nearest Neighbors Regressor
    KNeighborsRegressor,

    // Regression - SVM
    /// Support Vector Regression (kernelized)
    SVR,
    /// Linear SVM for regression
    LinearSVR,

    // Regression - Trees
    /// Decision Tree Regressor
    DecisionTreeRegressor,
    /// Random Forest Regressor
    RandomForestRegressor,
    /// Gradient Boosting Regressor
    GradientBoostingRegressor,
    /// Histogram-based Gradient Boosting Regressor
    HistGradientBoostingRegressor,
}

impl AlgorithmType {
    /// Check if algorithm is for classification
    pub fn is_classifier(&self) -> bool {
        matches!(
            self,
            Self::LogisticRegression
                | Self::GaussianNB
                | Self::MultinomialNB
                | Self::CategoricalNB
                | Self::KNeighborsClassifier
                | Self::SVC
                | Self::LinearSVC
                | Self::DecisionTreeClassifier
                | Self::RandomForestClassifier
                | Self::GradientBoostingClassifier
                | Self::HistGradientBoostingClassifier
        )
    }

    /// Check if algorithm is for regression
    pub fn is_regressor(&self) -> bool {
        !self.is_classifier()
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::LogisticRegression => "Logistic Regression",
            Self::GaussianNB => "Gaussian Naive Bayes",
            Self::MultinomialNB => "Multinomial Naive Bayes",
            Self::CategoricalNB => "Categorical Naive Bayes",
            Self::KNeighborsClassifier => "K-Nearest Neighbors Classifier",
            Self::SVC => "Support Vector Classifier",
            Self::LinearSVC => "Linear SVC",
            Self::DecisionTreeClassifier => "Decision Tree Classifier",
            Self::RandomForestClassifier => "Random Forest Classifier",
            Self::GradientBoostingClassifier => "Gradient Boosting Classifier",
            Self::HistGradientBoostingClassifier => "Histogram Gradient Boosting Classifier",
            Self::LinearRegression => "Linear Regression",
            Self::Ridge => "Ridge Regression",
            Self::Lasso => "Lasso Regression",
            Self::ElasticNet => "Elastic Net",
            Self::QuantileRegression => "Quantile Regression",
            Self::RobustRegression => "Robust Regression",
            Self::KNeighborsRegressor => "K-Nearest Neighbors Regressor",
            Self::SVR => "Support Vector Regressor",
            Self::LinearSVR => "Linear SVR",
            Self::DecisionTreeRegressor => "Decision Tree Regressor",
            Self::RandomForestRegressor => "Random Forest Regressor",
            Self::GradientBoostingRegressor => "Gradient Boosting Regressor",
            Self::HistGradientBoostingRegressor => "Histogram Gradient Boosting Regressor",
        }
    }
}

/// Algorithm complexity classification for time budget allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlgorithmComplexity {
    /// O(n) or O(n log n) - linear models, trees
    Low,
    /// O(n^1.5) or O(n * d) - random forests, histogram boosting
    Medium,
    /// O(n^2) or worse - SVM, gradient boosting on large data
    High,
}

impl AlgorithmComplexity {
    /// Get relative time budget multiplier
    pub fn time_factor(&self) -> f64 {
        match self {
            Self::Low => 1.0,
            Self::Medium => 2.0,
            Self::High => 5.0,
        }
    }
}

/// Preprocessing requirements for algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PreprocessingRequirement {
    /// Feature scaling (StandardScaler, MinMaxScaler)
    Scaling,
    /// Handle missing values (imputation)
    HandleMissing,
    /// Encode categorical features
    EncodeCategorical,
    /// Features must be non-negative (e.g., MultinomialNB)
    NonNegative,
    /// Dimensionality reduction (PCA, etc.)
    DimensionalityReduction,
}

/// Portfolio preset intensity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PortfolioPreset {
    /// Minimal algorithms, fast training
    /// ~3-5 algorithms, reduced search spaces
    Quick,
    /// Balanced coverage and training time
    /// ~8-10 algorithms, standard search spaces
    Balanced,
    /// Comprehensive algorithm portfolio
    /// All algorithms, expanded search spaces
    Thorough,
    /// Custom portfolio (user-defined)
    Custom,
}

/// Characteristics of the dataset for adaptive configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Number of classes (0 for regression)
    pub n_classes: usize,
    /// Whether data has missing values
    pub has_missing_values: bool,
    /// Whether data has categorical features
    pub has_categorical: bool,
    /// Ratio of max/min feature variance
    pub feature_variance_ratio: f64,
    /// Class imbalance ratio (max_class / min_class)
    pub class_imbalance_ratio: f64,
    /// Whether target is continuous (regression)
    pub is_regression: bool,
}

impl Default for DataCharacteristics {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            n_features: 10,
            n_classes: 2,
            has_missing_values: false,
            has_categorical: false,
            feature_variance_ratio: 1.0,
            class_imbalance_ratio: 1.0,
            is_regression: false,
        }
    }
}

impl DataCharacteristics {
    /// Create characteristics from feature and target arrays
    pub fn from_data(x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>) -> Self {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Check for missing values (NaN)
        let has_missing_values = x.iter().any(|v| v.is_nan());

        // Compute feature variance ratio
        let variances: Vec<f64> = (0..n_features)
            .map(|j| {
                let col = x.column(j);
                let mean = col.mean().unwrap_or(0.0);
                let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_samples as f64;
                var.max(1e-10) // Avoid zero
            })
            .collect();

        let max_var = variances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_var = variances.iter().cloned().fold(f64::INFINITY, f64::min);
        let feature_variance_ratio = max_var / min_var;

        // Determine if regression or classification
        let unique_y: std::collections::HashSet<i64> = y.iter().map(|v| *v as i64).collect();
        let is_regression = unique_y.len() > 20 || y.iter().any(|v| (v - v.round()).abs() > 1e-10);

        let (n_classes, class_imbalance_ratio) = if is_regression {
            (0, 1.0)
        } else {
            // Count class frequencies
            let mut counts = std::collections::HashMap::new();
            for v in y.iter() {
                let class = *v as i64;
                *counts.entry(class).or_insert(0usize) += 1;
            }
            let max_count = *counts.values().max().unwrap_or(&1) as f64;
            let min_count = *counts.values().min().unwrap_or(&1) as f64;
            (counts.len(), max_count / min_count.max(1.0))
        };

        Self {
            n_samples,
            n_features,
            n_classes,
            has_missing_values,
            has_categorical: false, // Cannot detect from f64 array
            feature_variance_ratio,
            class_imbalance_ratio,
            is_regression,
        }
    }

    /// Create with explicit values (for when categorical info is known)
    pub fn new(n_samples: usize, n_features: usize) -> Self {
        Self {
            n_samples,
            n_features,
            ..Default::default()
        }
    }

    /// Set number of classes
    pub fn with_n_classes(mut self, n: usize) -> Self {
        self.n_classes = n;
        self.is_regression = n == 0;
        self
    }

    /// Set missing values flag
    pub fn with_missing_values(mut self, has_missing: bool) -> Self {
        self.has_missing_values = has_missing;
        self
    }

    /// Set categorical flag
    pub fn with_categorical(mut self, has_categorical: bool) -> Self {
        self.has_categorical = has_categorical;
        self
    }

    /// Set imbalance ratio
    pub fn with_imbalance_ratio(mut self, ratio: f64) -> Self {
        self.class_imbalance_ratio = ratio;
        self
    }
}

/// Collection of algorithms for AutoML to consider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmPortfolio {
    /// Task type
    pub task: Task,
    /// Algorithms in the portfolio
    pub algorithms: Vec<AlgorithmConfig>,
    /// Preset used to create this portfolio
    pub preset: PortfolioPreset,
}

impl AlgorithmPortfolio {
    /// Create a portfolio for classification
    pub fn for_classification(preset: PortfolioPreset) -> Self {
        let algorithms = match preset {
            PortfolioPreset::Quick => Self::quick_classification(),
            PortfolioPreset::Balanced => Self::balanced_classification(),
            PortfolioPreset::Thorough => Self::thorough_classification(),
            PortfolioPreset::Custom => Vec::new(),
        };

        Self {
            task: Task::Classification,
            algorithms,
            preset,
        }
    }

    /// Create a portfolio for regression
    pub fn for_regression(preset: PortfolioPreset) -> Self {
        let algorithms = match preset {
            PortfolioPreset::Quick => Self::quick_regression(),
            PortfolioPreset::Balanced => Self::balanced_regression(),
            PortfolioPreset::Thorough => Self::thorough_regression(),
            PortfolioPreset::Custom => Vec::new(),
        };

        Self {
            task: Task::Regression,
            algorithms,
            preset,
        }
    }

    /// Create a custom portfolio
    pub fn custom(task: Task) -> Self {
        Self {
            task,
            algorithms: Vec::new(),
            preset: PortfolioPreset::Custom,
        }
    }

    /// Add an algorithm to the portfolio
    pub fn add(mut self, config: AlgorithmConfig) -> Self {
        self.algorithms.push(config);
        self
    }

    /// Remove an algorithm type from the portfolio
    pub fn remove(&mut self, algorithm: AlgorithmType) {
        self.algorithms.retain(|a| a.algorithm != algorithm);
    }

    /// Adapt all algorithms to data characteristics
    pub fn adapt_to_data(&self, characteristics: &DataCharacteristics) -> Self {
        let adapted_algorithms: Vec<AlgorithmConfig> = self
            .algorithms
            .iter()
            .map(|a| a.adapt_to_data(characteristics))
            .collect();

        Self {
            task: self.task,
            algorithms: adapted_algorithms,
            preset: self.preset,
        }
    }

    /// Get algorithms sorted by priority (highest first)
    pub fn sorted_by_priority(&self) -> Vec<&AlgorithmConfig> {
        let mut sorted: Vec<_> = self.algorithms.iter().collect();
        sorted.sort_by_key(|a| std::cmp::Reverse(a.priority));
        sorted
    }

    /// Get combined search space with algorithm selection
    pub fn combined_search_space(&self) -> SearchSpace {
        // Create a categorical parameter for algorithm selection
        let algorithm_names: Vec<String> = self
            .algorithms
            .iter()
            .map(|a| format!("{:?}", a.algorithm))
            .collect();

        let mut space = SearchSpace::new().categorical("algorithm", algorithm_names);

        // Add prefixed parameters from each algorithm
        for config in &self.algorithms {
            let prefix = format!("{:?}", config.algorithm);
            for (name, param) in &config.search_space.parameters {
                let prefixed_name = format!("{}__{}", prefix, name);
                space = space.add(prefixed_name, param.clone());
            }
        }

        space
    }

    // Quick classification portfolio - fast algorithms only
    fn quick_classification() -> Vec<AlgorithmConfig> {
        vec![
            AlgorithmConfig::new(AlgorithmType::LogisticRegression).with_priority(90),
            AlgorithmConfig::new(AlgorithmType::GaussianNB).with_priority(80),
            AlgorithmConfig::new(AlgorithmType::DecisionTreeClassifier).with_priority(70),
            AlgorithmConfig::new(AlgorithmType::HistGradientBoostingClassifier).with_priority(85),
        ]
    }

    // Balanced classification portfolio
    fn balanced_classification() -> Vec<AlgorithmConfig> {
        vec![
            AlgorithmConfig::new(AlgorithmType::LogisticRegression).with_priority(85),
            AlgorithmConfig::new(AlgorithmType::GaussianNB).with_priority(70),
            AlgorithmConfig::new(AlgorithmType::KNeighborsClassifier).with_priority(60),
            AlgorithmConfig::new(AlgorithmType::LinearSVC).with_priority(75),
            AlgorithmConfig::new(AlgorithmType::DecisionTreeClassifier).with_priority(65),
            AlgorithmConfig::new(AlgorithmType::RandomForestClassifier).with_priority(90),
            AlgorithmConfig::new(AlgorithmType::GradientBoostingClassifier).with_priority(80),
            AlgorithmConfig::new(AlgorithmType::HistGradientBoostingClassifier).with_priority(95),
        ]
    }

    // Thorough classification portfolio - all classifiers
    fn thorough_classification() -> Vec<AlgorithmConfig> {
        vec![
            AlgorithmConfig::new(AlgorithmType::LogisticRegression).with_priority(80),
            AlgorithmConfig::new(AlgorithmType::GaussianNB).with_priority(65),
            AlgorithmConfig::new(AlgorithmType::MultinomialNB).with_priority(55),
            AlgorithmConfig::new(AlgorithmType::CategoricalNB).with_priority(50),
            AlgorithmConfig::new(AlgorithmType::KNeighborsClassifier).with_priority(60),
            AlgorithmConfig::new(AlgorithmType::SVC).with_priority(70),
            AlgorithmConfig::new(AlgorithmType::LinearSVC).with_priority(75),
            AlgorithmConfig::new(AlgorithmType::DecisionTreeClassifier).with_priority(50),
            AlgorithmConfig::new(AlgorithmType::RandomForestClassifier).with_priority(90),
            AlgorithmConfig::new(AlgorithmType::GradientBoostingClassifier).with_priority(85),
            AlgorithmConfig::new(AlgorithmType::HistGradientBoostingClassifier).with_priority(95),
        ]
    }

    // Quick regression portfolio
    fn quick_regression() -> Vec<AlgorithmConfig> {
        vec![
            AlgorithmConfig::new(AlgorithmType::LinearRegression).with_priority(90),
            AlgorithmConfig::new(AlgorithmType::Ridge).with_priority(85),
            AlgorithmConfig::new(AlgorithmType::DecisionTreeRegressor).with_priority(70),
            AlgorithmConfig::new(AlgorithmType::HistGradientBoostingRegressor).with_priority(80),
        ]
    }

    // Balanced regression portfolio
    fn balanced_regression() -> Vec<AlgorithmConfig> {
        vec![
            AlgorithmConfig::new(AlgorithmType::LinearRegression).with_priority(80),
            AlgorithmConfig::new(AlgorithmType::Ridge).with_priority(85),
            AlgorithmConfig::new(AlgorithmType::Lasso).with_priority(75),
            AlgorithmConfig::new(AlgorithmType::ElasticNet).with_priority(70),
            AlgorithmConfig::new(AlgorithmType::KNeighborsRegressor).with_priority(60),
            AlgorithmConfig::new(AlgorithmType::DecisionTreeRegressor).with_priority(55),
            AlgorithmConfig::new(AlgorithmType::RandomForestRegressor).with_priority(90),
            AlgorithmConfig::new(AlgorithmType::HistGradientBoostingRegressor).with_priority(95),
        ]
    }

    // Thorough regression portfolio
    fn thorough_regression() -> Vec<AlgorithmConfig> {
        vec![
            AlgorithmConfig::new(AlgorithmType::LinearRegression).with_priority(75),
            AlgorithmConfig::new(AlgorithmType::Ridge).with_priority(80),
            AlgorithmConfig::new(AlgorithmType::Lasso).with_priority(70),
            AlgorithmConfig::new(AlgorithmType::ElasticNet).with_priority(65),
            AlgorithmConfig::new(AlgorithmType::QuantileRegression).with_priority(50),
            AlgorithmConfig::new(AlgorithmType::RobustRegression).with_priority(55),
            AlgorithmConfig::new(AlgorithmType::KNeighborsRegressor).with_priority(60),
            AlgorithmConfig::new(AlgorithmType::SVR).with_priority(65),
            AlgorithmConfig::new(AlgorithmType::LinearSVR).with_priority(70),
            AlgorithmConfig::new(AlgorithmType::DecisionTreeRegressor).with_priority(45),
            AlgorithmConfig::new(AlgorithmType::RandomForestRegressor).with_priority(90),
            AlgorithmConfig::new(AlgorithmType::GradientBoostingRegressor).with_priority(85),
            AlgorithmConfig::new(AlgorithmType::HistGradientBoostingRegressor).with_priority(95),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_algorithm_config_creation() {
        let config = AlgorithmConfig::new(AlgorithmType::RandomForestClassifier);

        assert_eq!(config.algorithm, AlgorithmType::RandomForestClassifier);
        assert!(!config.search_space.parameters.is_empty());
        assert!(config.search_space.parameters.contains_key("n_estimators"));
        assert!(config.search_space.parameters.contains_key("max_depth"));
    }

    #[test]
    fn test_algorithm_type_classification() {
        assert!(AlgorithmType::LogisticRegression.is_classifier());
        assert!(AlgorithmType::RandomForestClassifier.is_classifier());
        assert!(!AlgorithmType::LinearRegression.is_classifier());
        assert!(!AlgorithmType::RandomForestRegressor.is_classifier());

        assert!(AlgorithmType::Ridge.is_regressor());
        assert!(!AlgorithmType::SVC.is_regressor());
    }

    #[test]
    fn test_portfolio_quick_classification() {
        let portfolio = AlgorithmPortfolio::for_classification(PortfolioPreset::Quick);

        assert_eq!(portfolio.task, Task::Classification);
        assert_eq!(portfolio.algorithms.len(), 4);

        // All should be classifiers
        for config in &portfolio.algorithms {
            assert!(config.algorithm.is_classifier());
        }
    }

    #[test]
    fn test_portfolio_balanced_regression() {
        let portfolio = AlgorithmPortfolio::for_regression(PortfolioPreset::Balanced);

        assert_eq!(portfolio.task, Task::Regression);
        assert_eq!(portfolio.algorithms.len(), 8);

        // All should be regressors
        for config in &portfolio.algorithms {
            assert!(config.algorithm.is_regressor());
        }
    }

    #[test]
    fn test_portfolio_thorough_classification() {
        let portfolio = AlgorithmPortfolio::for_classification(PortfolioPreset::Thorough);

        assert_eq!(portfolio.algorithms.len(), 11);
    }

    #[test]
    fn test_custom_portfolio() {
        let portfolio = AlgorithmPortfolio::custom(Task::Classification)
            .add(AlgorithmConfig::new(AlgorithmType::LogisticRegression).with_priority(100))
            .add(AlgorithmConfig::new(AlgorithmType::RandomForestClassifier).with_priority(90));

        assert_eq!(portfolio.algorithms.len(), 2);
        assert_eq!(portfolio.preset, PortfolioPreset::Custom);
    }

    #[test]
    fn test_sorted_by_priority() {
        let portfolio = AlgorithmPortfolio::custom(Task::Classification)
            .add(AlgorithmConfig::new(AlgorithmType::LogisticRegression).with_priority(50))
            .add(AlgorithmConfig::new(AlgorithmType::RandomForestClassifier).with_priority(100))
            .add(AlgorithmConfig::new(AlgorithmType::GaussianNB).with_priority(75));

        let sorted = portfolio.sorted_by_priority();

        assert_eq!(sorted[0].algorithm, AlgorithmType::RandomForestClassifier);
        assert_eq!(sorted[1].algorithm, AlgorithmType::GaussianNB);
        assert_eq!(sorted[2].algorithm, AlgorithmType::LogisticRegression);
    }

    #[test]
    fn test_data_characteristics_from_data() {
        // Create test data for classification
        let x =
            Array2::from_shape_vec((100, 5), (0..500).map(|i| (i as f64) * 0.1).collect()).unwrap();
        let y = Array1::from_vec((0..100).map(|i| (i % 3) as f64).collect());

        let chars = DataCharacteristics::from_data(&x, &y);

        assert_eq!(chars.n_samples, 100);
        assert_eq!(chars.n_features, 5);
        assert_eq!(chars.n_classes, 3);
        assert!(!chars.is_regression);
    }

    #[test]
    fn test_data_characteristics_regression() {
        let x = Array2::from_shape_vec((50, 3), (0..150).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..50).map(|i| i as f64 * 1.5 + 0.3).collect());

        let chars = DataCharacteristics::from_data(&x, &y);

        assert!(chars.is_regression);
        assert_eq!(chars.n_classes, 0);
    }

    #[test]
    fn test_adapt_to_data_knn() {
        let config = AlgorithmConfig::new(AlgorithmType::KNeighborsClassifier);
        let chars = DataCharacteristics::new(100, 10).with_n_classes(2);

        let adapted = config.adapt_to_data(&chars);

        // n_neighbors should be adapted based on sqrt(n_samples) = 10
        let n_neighbors_param = adapted.search_space.parameters.get("n_neighbors").unwrap();
        if let crate::hpo::ParameterType::Int { high, .. } = &n_neighbors_param.param_type {
            assert!(*high <= 10, "max k should be <= sqrt(100) = 10");
        } else {
            panic!("n_neighbors should be Int parameter");
        }
    }

    #[test]
    fn test_adapt_to_data_adds_preprocessing() {
        // HistGradientBoosting handles missing values natively, so no preprocessing required
        let config = AlgorithmConfig::new(AlgorithmType::HistGradientBoostingClassifier);
        assert!(
            config.preprocessing.is_empty(),
            "HistGB should have no preprocessing requirements by default"
        );

        // When data has categorical features, categorical encoding should be added
        let chars = DataCharacteristics::new(100, 10)
            .with_n_classes(2)
            .with_categorical(true);

        let adapted = config.adapt_to_data(&chars);

        assert!(
            adapted
                .preprocessing
                .contains(&PreprocessingRequirement::EncodeCategorical),
            "Should add categorical encoding for categorical data"
        );

        // Test that high variance ratio triggers scaling
        let config2 = AlgorithmConfig::new(AlgorithmType::DecisionTreeClassifier);
        assert!(
            !config2
                .preprocessing
                .contains(&PreprocessingRequirement::Scaling),
            "DecisionTree shouldn't need scaling by default"
        );

        let chars_high_var = DataCharacteristics {
            n_samples: 100,
            n_features: 10,
            n_classes: 2,
            has_missing_values: false,
            has_categorical: false,
            feature_variance_ratio: 200.0, // > 100 threshold
            class_imbalance_ratio: 1.0,
            is_regression: false,
        };

        let adapted2 = config2.adapt_to_data(&chars_high_var);
        assert!(
            adapted2
                .preprocessing
                .contains(&PreprocessingRequirement::Scaling),
            "Should add scaling for high variance ratio"
        );
    }

    #[test]
    fn test_combined_search_space() {
        let portfolio = AlgorithmPortfolio::custom(Task::Classification)
            .add(AlgorithmConfig::new(AlgorithmType::LogisticRegression))
            .add(AlgorithmConfig::new(AlgorithmType::RandomForestClassifier));

        let combined = portfolio.combined_search_space();

        // Should have algorithm selector
        assert!(combined.parameters.contains_key("algorithm"));

        // Should have prefixed parameters
        assert!(combined
            .parameters
            .contains_key("LogisticRegression__l2_penalty"));
        assert!(combined
            .parameters
            .contains_key("RandomForestClassifier__n_estimators"));
    }

    #[test]
    fn test_algorithm_complexity() {
        assert_eq!(AlgorithmComplexity::Low.time_factor(), 1.0);
        assert_eq!(AlgorithmComplexity::Medium.time_factor(), 2.0);
        assert_eq!(AlgorithmComplexity::High.time_factor(), 5.0);
    }

    #[test]
    fn test_preprocessing_requirements() {
        let logreg = AlgorithmConfig::new(AlgorithmType::LogisticRegression);
        assert!(logreg
            .preprocessing
            .contains(&PreprocessingRequirement::Scaling));

        let rf = AlgorithmConfig::new(AlgorithmType::RandomForestClassifier);
        assert!(!rf
            .preprocessing
            .contains(&PreprocessingRequirement::Scaling));

        let hist_gb = AlgorithmConfig::new(AlgorithmType::HistGradientBoostingClassifier);
        assert!(!hist_gb
            .preprocessing
            .contains(&PreprocessingRequirement::HandleMissing));
    }

    #[test]
    fn test_remove_algorithm() {
        let mut portfolio = AlgorithmPortfolio::for_classification(PortfolioPreset::Quick);
        let initial_count = portfolio.algorithms.len();

        portfolio.remove(AlgorithmType::LogisticRegression);

        assert_eq!(portfolio.algorithms.len(), initial_count - 1);
        assert!(!portfolio
            .algorithms
            .iter()
            .any(|a| a.algorithm == AlgorithmType::LogisticRegression));
    }

    #[test]
    fn test_adapt_large_dataset() {
        let config = AlgorithmConfig::new(AlgorithmType::GradientBoostingClassifier);
        let chars = DataCharacteristics::new(200_000, 50).with_n_classes(2);

        let adapted = config.adapt_to_data(&chars);

        // n_estimators should be reduced for large dataset
        let n_est_param = adapted.search_space.parameters.get("n_estimators").unwrap();
        if let crate::hpo::ParameterType::Int { high, .. } = &n_est_param.param_type {
            assert!(
                *high < 500,
                "n_estimators max should be reduced for large datasets"
            );
        }
    }

    #[test]
    fn test_high_dimensional_data_regularization() {
        let config = AlgorithmConfig::new(AlgorithmType::Ridge);
        // High features relative to samples (overfitting risk)
        let chars = DataCharacteristics::new(100, 200).with_n_classes(0);

        let adapted = config.adapt_to_data(&chars);

        // Alpha (regularization) range should be biased higher
        let alpha_param = adapted.search_space.parameters.get("alpha").unwrap();
        if let crate::hpo::ParameterType::Float { low, .. } = &alpha_param.param_type {
            assert!(
                *low > 1e-4,
                "min alpha should be higher for high-dimensional data"
            );
        }
    }
}

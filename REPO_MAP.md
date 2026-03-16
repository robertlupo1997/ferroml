# FerroML Core - Structural Repo Map

Generated structural overview of all public types, traits, and function signatures
in `ferroml-core/src/`. Excludes function bodies, private items, test modules,
comments, and `use` statements.

**Module hierarchy:** automl, clustering, cv, datasets, decomposition, ensemble,
error, explainability, gpu, hpo, inference, linalg, metrics, models, neural,
onnx, pipeline, preprocessing, schema, serialization, simd, sparse, stats

---
```
ferroml-core/src/error.rs
  pub enum FerroError { InvalidInput, ShapeMismatch, AssumptionViolation, NumericalError, ConvergenceFailure, NotImplemented, NotImplementedFor, NotFitted, ConfigError, IoError, SerializationError, Timeout, ... }
  impl FerroError {
    pub fn invalid_input(msg: impl Into<String>) -> Self
    pub fn shape_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self
    pub fn assumption_violation(assumption: impl Into<String>, test: impl Into<String>, p_value: f64,) -> Self
    pub fn numerical(msg: impl Into<String>) -> Self
    pub fn convergence_failure(iterations: usize, reason: impl Into<String>) -> Self
    pub fn not_fitted(operation: impl Into<String>) -> Self
    pub fn not_implemented_for(feature: impl Into<String>, model: impl Into<String>) -> Self
    pub fn cross_validation(msg: impl Into<String>) -> Self
  }

ferroml-core/src/lib.rs
  pub struct PredictionWithUncertainty { predictions: ndarray::Array1<f64>, lower: ndarray::Array1<f64>, upper: ndarray::Array1<f64>, confidence_level: f64, std_errors: Option<ndarray::Array1<f64>> }
  pub struct AutoMLConfig { task: Task, metric: Metric, time_budget_seconds: u64, cv_folds: usize, statistical_tests: bool, confidence_level: f64, multiple_testing_correction: MultipleTesting, seed: Option<u64>, ... }
  pub struct AutoML { ... }
  pub enum Task { Classification, Regression, TimeSeries, Survival }
  pub enum Metric { RocAuc, Accuracy, F1, LogLoss, Mcc, Mse, Rmse, Mae, R2, Mape, Smape, Mase }
  pub enum MultipleTesting { None, Bonferroni, Holm, BenjaminiHochberg, BenjaminiYekutieli }
  pub enum AlgorithmSelection { Uniform, Bayesian, Bandit }
  pub trait Predictor: Send + Sync {
    fn predict(&self, x: &Array2<f64>) -> crate::Result<ndarray::Array1<f64>>
    fn predict_with_uncertainty(&self, x: &Array2<f64>, confidence: f64,) -> crate::Result<PredictionWithUncertainty>
  }
  pub trait Estimator: Send + Sync {
    fn fit(&self, x: &Array2<f64>, y: &ndarray::Array1<f64>) -> crate::Result<Self::Fitted>
    fn search_space(&self) -> crate::hpo::SearchSpace
  }
  pub trait Transformer: Send + Sync {
    fn transform(&self, x: &Array2<f64>) -> crate::Result<Array2<f64>>
    fn fit_transform(&mut self, x: &Array2<f64>) -> crate::Result<Array2<f64>>
    fn inverse_transform(&self, x: &Array2<f64>) -> crate::Result<Array2<f64>>
  }
  impl Default for AutoMLConfig { fn default(...) }
  impl AutoML {
    pub fn new(config: AutoMLConfig) -> Self
    pub fn default_config() -> Self
    pub fn config(&self) -> &AutoMLConfig
  }

ferroml-core/src/linalg.rs
  pub fn qr_decomposition(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)>
  pub fn qr_decomposition_mgs(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)>
  pub fn qr_decomposition_faer(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)>
  pub fn solve_upper_triangular(r: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>>
  pub fn invert_upper_triangular(r: &Array2<f64>) -> Result<Array2<f64>>
  pub fn cholesky(a: &Array2<f64>, reg: f64) -> Result<Array2<f64>>
  pub fn cholesky_faer(a: &Array2<f64>, reg: f64) -> Result<Array2<f64>>
  pub fn cholesky_native(a: &Array2<f64>, reg: f64) -> Result<Array2<f64>>
  pub fn log_determinant_from_cholesky(l: &Array2<f64>) -> f64
  pub fn solve_lower_triangular(l: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>>
  pub fn solve_lower_triangular_vec(l: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>>
  pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64
  pub fn squared_euclidean_distance(a: &[f64], b: &[f64]) -> f64
  pub fn dot_product(a: &[f64], b: &[f64]) -> f64

ferroml-core/src/schema.rs
  pub struct FeatureSpec { name: Option<String>, feature_type: FeatureType, allow_missing: bool, min_value: Option<f64>, max_value: Option<f64>, categories: Option<Vec<f64>> }
  pub struct ValidationResult { passed: bool, issues: Vec<ValidationIssue>, n_samples: usize, n_features: usize, mode: ValidationMode }
  pub struct FeatureSchema { n_features: usize, features: Vec<FeatureSpec>, mode: ValidationMode, max_issues: usize, sample_validation: bool, sample_fraction: f64 }
  pub enum ValidationMode { Strict, Warn, Permissive }
  pub enum FeatureType { Continuous, Integer, Categorical, Binary, Unknown }
  pub enum ValidationIssue { ShapeMismatch, MissingValue, ValueBelowMin, ValueAboveMax, UnknownCategory, InvalidBinary, NonInteger, FeatureNameMismatch, EmptyInput }
  pub enum IssueSeverity { Critical, Error, Warning }
  pub trait SchemaValidated {
    fn feature_schema(&self) -> Option<&FeatureSchema>
    fn validate_input(&self, x: &Array2<f64>) -> Result<()>
  }
  impl Default for FeatureSpec { fn default(...) }
  impl FeatureSpec {
    pub fn continuous() -> Self
    pub fn integer() -> Self
    pub fn categorical(categories: Vec<f64>) -> Self
    pub fn binary() -> Self
    pub fn with_name(mut self, name: impl Into<String>) -> Self
    pub fn allow_missing(mut self) -> Self
    pub fn with_min(mut self, min: f64) -> Self
    pub fn with_max(mut self, max: f64) -> Self
    pub fn with_range(mut self, min: f64, max: f64) -> Self
    pub fn from_column(values: &[f64]) -> Self
    pub fn validate_value(&self, value: f64, feature_idx: usize) -> Vec<ValidationIssue>
  }
  impl ValidationIssue {
    pub fn is_critical(&self) -> bool
    pub fn severity(&self) -> IssueSeverity
  }
  impl ValidationResult {
    pub fn success(n_samples: usize, n_features: usize, mode: ValidationMode) -> Self
    pub fn failure(issues: Vec<ValidationIssue>, n_samples: usize, n_features: usize, ...) -> Self
    pub fn issues_by_severity(&self, severity: IssueSeverity) -> Vec<&ValidationIssue>
    pub fn critical_issues(&self) -> Vec<&ValidationIssue>
    pub fn error_issues(&self) -> Vec<&ValidationIssue>
    pub fn warnings(&self) -> Vec<&ValidationIssue>
    pub fn issue_counts(&self) -> HashMap<String, usize>
  }
  impl FeatureSchema {
    pub fn new(n_features: usize) -> Self
    pub fn from_array(x: &Array2<f64>) -> Self
    pub fn shape_only(n_features: usize) -> Self
    pub fn with_mode(mut self, mode: ValidationMode) -> Self
    pub fn with_max_issues(mut self, max_issues: usize) -> Self
    pub fn with_sample_validation(mut self, fraction: f64) -> Self
    pub fn allow_missing(mut self) -> Self
    pub fn set_feature(&mut self, index: usize, spec: FeatureSpec)
    pub fn feature_names(&self) -> Vec<String>
    pub fn validate(&self, x: &Array2<f64>) -> ValidationResult
    pub fn validate_strict(&self, x: &Array2<f64>) -> Result<()>
    pub fn validate_feature_names(&self, names: &[String]) -> Vec<ValidationIssue>
    pub fn summary(&self) -> String
  }

ferroml-core/src/serialization.rs
  pub struct SaveOptions { format: Format, description: Option<String>, include_metadata: bool, progress_callback: Option<fn(u64 }
  pub struct LoadOptions { format: Format, verify_checksum: bool, allow_version_mismatch: bool, progress_callback: Option<fn(u64 }
  pub struct SemanticVersion { major: u32, minor: u32, patch: u32 }
  pub struct SerializationMetadata { ferroml_version: SemanticVersion, model_type: String, format: Format, timestamp: u64, description: Option<String> }
  pub struct ModelContainer { metadata: SerializationMetadata, model: T }
  pub struct StreamingWriter { ... }
  pub struct StreamingReader { ... }
  pub enum Format { Json, JsonPretty, MessagePack, Bincode }
  pub trait ModelSerialize: Serialize + DeserializeOwned + Sized {
    fn save(&self, path: impl AsRef<Path>, format: Format) -> Result<()>
    fn save_with_description(&self, path: impl AsRef<Path>, format: Format, descriptio...) -> Result<()>
    fn load(path: impl AsRef<Path>, format: Format) -> Result<Self>
    fn load_auto(path: impl AsRef<Path>) -> Result<Self>
    fn to_bytes(&self, format: Format) -> Result<Vec<u8>>
    fn from_bytes(bytes: &[u8], format: Format) -> Result<Self>
    fn save_json(&self, path: impl AsRef<Path>) -> Result<()>
    fn save_json_pretty(&self, path: impl AsRef<Path>) -> Result<()>
    fn save_msgpack(&self, path: impl AsRef<Path>) -> Result<()>
    fn save_bincode(&self, path: impl AsRef<Path>) -> Result<()>
    fn load_json(path: impl AsRef<Path>) -> Result<Self>
    fn load_msgpack(path: impl AsRef<Path>) -> Result<Self>
    fn load_bincode(path: impl AsRef<Path>) -> Result<Self>
  }
  impl Format {
    pub fn extension(&self) -> &'static str
    pub fn from_extension(path: &Path) -> Option<Self>
  }
  impl SaveOptions {
    pub fn new(format: Format) -> Self
    pub fn with_description(mut self, description: impl Into<String>) -> Self
    pub fn without_metadata(mut self) -> Self
    pub fn with_progress(mut self, callback: fn(u64, u64)
  }
  impl LoadOptions {
    pub fn new(format: Format) -> Self
    pub fn skip_checksum(mut self) -> Self
    pub fn allow_version_mismatch(mut self) -> Self
    pub fn with_progress(mut self, callback: fn(u64, u64)
  }
  impl SemanticVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self
    pub fn current() -> Self
    pub fn is_compatible_with(&self, other: &Self) -> bool
  }
  impl FromStr for SemanticVersion { fn from_str(...) }
  impl PartialOrd for SemanticVersion { fn partial_cmp(...) }
  impl Ord for SemanticVersion { fn cmp(...) }
  impl SerializationMetadata {
    pub fn with_description(mut self, description: impl Into<String>) -> Self
  }
  impl ModelContainer<T> {
    pub fn new(model: T, format: Format) -> Self
    where
        T: Serialize,
    pub fn with_metadata(model: T, metadata: SerializationMetadata) -> Self
  }
  impl StreamingWriter<W> {
    pub fn new(writer: W) -> Self
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self
    pub fn with_progress(mut self, callback: fn(u64, u64)
    pub fn bytes_written(&self) -> u64
  }
  impl StreamingReader<R> {
    pub fn new(reader: R) -> Self
    pub fn with_progress(mut self, callback: fn(u64, u64)
    pub fn bytes_read(&self) -> u64
  }
  pub fn save_model(model: &T, path: P, format: Format) -> Result<()>
where
    T: Serialize,
    P: AsRef<Path>,
  pub fn save_model_with_description(model: &T, path: P, format: Format, description: impl Into<String>,) -> Result<()>
where
    T: Serialize,
    P: AsRef<Path>,
  pub fn save_model_with_options(model: &T, path: P, options: &SaveOptions) -> Result<()>
where
    T: Serialize,
    P: AsRef<Path>,
  pub fn load_model_with_options(path: P, options: &LoadOptions) -> Result<T>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
  pub fn load_model(path: P, format: Format) -> Result<T>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
  pub fn load_model_with_metadata(path: P, format: Format) -> Result<(T, SerializationMetadata)>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
  pub fn load_model_auto(path: P) -> Result<T>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
  pub fn to_bytes(model: &T, format: Format) -> Result<Vec<u8>>
where
    T: Serialize,
  pub fn from_bytes(bytes: &[u8], format: Format) -> Result<T>
where
    T: DeserializeOwned,
  pub fn from_bytes_with_metadata(bytes: &[u8], format: Format,) -> Result<(T, SerializationMetadata)>
where
    T: DeserializeOwned,
  pub fn peek_metadata(path: P, format: Format) -> Result<SerializationMetadata>
where
    P: AsRef<Path>,

ferroml-core/src/simd.rs
  pub fn squared_euclidean_distance(a: &[f64], b: &[f64]) -> f64
  pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64
  pub fn manhattan_distance(a: &[f64], b: &[f64]) -> f64
  pub fn batch_squared_euclidean(query: &[f64], references: &[f64], n_samples: usize, n_features: usize,) -> Vec<f64>
  pub fn batch_manhattan_distance(query: &[f64], references: &[f64], n_samples: usize, n_features: usize,) -> Vec<f64>
  pub fn cosine_similarity_batch(query: &[f64], references: &[f64], n_samples: usize, n_features: usize,) -> Vec<f64>
  pub fn dot_product(a: &[f64], b: &[f64]) -> f64
  pub fn sum(a: &[f64]) -> f64
  pub fn sum_of_squares(a: &[f64]) -> f64
  pub fn squared_differences(a: &[f64], b: &[f64]) -> Vec<f64>
  pub fn minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64
  pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64
  pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64
  pub fn matrix_vector_mul(matrix: &[f64], vector: &[f64], m: usize, n: usize) -> Vec<f64>
  pub fn matrix_vector_mul_into(matrix: &[f64], vector: &[f64], result: &mut [f64], m: usize, n: usize,)
  pub fn vector_matrix_mul(vector: &[f64], matrix: &[f64], m: usize, n: usize) -> Vec<f64>
  pub fn vector_add_scalar(a: &[f64], scalar: f64) -> Vec<f64>
  pub fn vector_mul_scalar(a: &[f64], scalar: f64) -> Vec<f64>
  pub fn vector_add(a: &[f64], b: &[f64]) -> Vec<f64>
  pub fn vector_sub(a: &[f64], b: &[f64]) -> Vec<f64>
  pub fn vector_sub_into(a: &[f64], b: &[f64], dst: &mut [f64])
  pub fn vector_mul(a: &[f64], b: &[f64]) -> Vec<f64>
  pub fn vector_div(a: &[f64], b: &[f64]) -> Vec<f64>
  pub fn axpy(a: f64, x: &[f64], y: &mut [f64])
  pub fn axpby(a: f64, x: &[f64], b: f64, y: &mut [f64])
  pub fn sum_rows(matrix: &[f64], m: usize, n: usize) -> Vec<f64>
  pub fn sum_cols(matrix: &[f64], m: usize, n: usize) -> Vec<f64>
  pub fn row_norms_l2(matrix: &[f64], m: usize, n: usize) -> Vec<f64>
  pub fn row_norms_l1(matrix: &[f64], m: usize, n: usize) -> Vec<f64>
  pub fn sum_abs(a: &[f64]) -> f64
  pub fn normalize_rows_l2(matrix: &mut [f64], m: usize, n: usize)
  pub fn row_means(matrix: &[f64], m: usize, n: usize) -> Vec<f64>
  pub fn col_means(matrix: &[f64], m: usize, n: usize) -> Vec<f64>
  pub fn center_rows(matrix: &mut [f64], m: usize, n: usize)
  pub fn scale_rows_by_std(matrix: &mut [f64], m: usize, n: usize)
  pub fn outer_product(a: &[f64], b: &[f64]) -> Vec<f64>

ferroml-core/src/sparse.rs
  pub struct CsrMatrix { ... }
  pub struct SparseRowView { ... }
  pub struct CscMatrix { ... }
  pub struct SparseVector { ... }
  pub struct SparseMatrixInfo { nrows: usize, ncols: usize, nnz: usize, sparsity: f64, density: f64, sparse_memory: usize, dense_memory: usize, recommend_dense: bool }
  pub enum SparseDistanceMetric { Euclidean, SquaredEuclidean, Manhattan, Cosine }
  impl CsrMatrix {
    pub fn new(shape: (usize, usize)
    pub fn from_dense(dense: &Array2<f64>) -> Self
    pub fn from_dense_with_threshold(dense: &Array2<f64>, threshold: f64) -> Self
    pub fn from_triplets(shape: (usize, usize)
    pub fn shape(&self) -> (usize, usize)
    pub fn nrows(&self) -> usize
    pub fn ncols(&self) -> usize
    pub fn nnz(&self) -> usize
    pub fn sparsity(&self) -> f64
    pub fn row(&self, i: usize) -> SparseRowView<'_>
    pub fn data(&self) -> &[f64]
    pub fn indices(&self) -> &[usize]
    pub fn indptr(&self) -> Vec<usize>
    pub fn to_dense(&self) -> Array2<f64>
    pub fn view(&self) -> CsMatView<'_, f64>
    pub fn recommend_dense(&self) -> bool
    pub fn transpose_dot(&self, y: &Array1<f64>) -> Array1<f64>
    pub fn gram_matrix(&self) -> Array2<f64>
    pub fn weighted_gram(&self, w: &Array1<f64>) -> Array2<f64>
    pub fn transpose(&self) -> CscMatrix
    pub fn dot(&self, x: &Array1<f64>) -> Result<Array1<f64>>
    pub fn row_norms_squared(&self) -> Array1<f64>
    pub fn row_norms(&self) -> Array1<f64>
    pub fn scale_rows(&mut self, factors: &Array1<f64>) -> Result<()>
  }
  impl SparseRowView<'a> {
    pub fn nnz(&self) -> usize
    pub fn dim(&self) -> usize
    pub fn indices(&self) -> &[usize]
    pub fn data(&self) -> &[f64]
    pub fn is_empty(&self) -> bool
    pub fn norm_squared(&self) -> f64
    pub fn norm(&self) -> f64
    pub fn l1_norm(&self) -> f64
    pub fn to_dense(&self) -> Array1<f64>
  }
  impl CscMatrix {
    pub fn new(shape: (usize, usize)
    pub fn shape(&self) -> (usize, usize)
    pub fn nnz(&self) -> usize
    pub fn to_csr(&self) -> CsrMatrix
    pub fn to_dense(&self) -> Array2<f64>
  }
  impl SparseVector {
    pub fn new(dim: usize, indices: Vec<usize>, data: Vec<f64>) -> Result<Self>
    pub fn from_dense(dense: &Array1<f64>) -> Self
    pub fn from_dense_with_threshold(dense: &Array1<f64>, threshold: f64) -> Self
    pub fn dim(&self) -> usize
    pub fn nnz(&self) -> usize
    pub fn indices(&self) -> &[usize]
    pub fn data(&self) -> &[f64]
    pub fn to_dense(&self) -> Array1<f64>
  }
  impl SparseMatrixInfo {
    pub fn from_matrix(matrix: &CsrMatrix) -> Self
    pub fn summary(&self) -> String
  }
  pub fn sparse_squared_euclidean_distance(a: &SparseRowView<'_>, b: &SparseRowView<'_>) -> f64
  pub fn sparse_euclidean_distance(a: &SparseRowView<'_>, b: &SparseRowView<'_>) -> f64
  pub fn sparse_manhattan_distance(a: &SparseRowView<'_>, b: &SparseRowView<'_>) -> f64
  pub fn sparse_dot_product(a: &SparseRowView<'_>, b: &SparseRowView<'_>) -> f64
  pub fn sparse_cosine_similarity(a: &SparseRowView<'_>, b: &SparseRowView<'_>) -> f64
  pub fn sparse_cosine_distance(a: &SparseRowView<'_>, b: &SparseRowView<'_>) -> f64
  pub fn sparse_pairwise_distances(query: &SparseRowView<'_>, matrix: &CsrMatrix, metric: SparseDistanceMetric,) -> Array1<f64>
  pub fn sparse_normalize_rows_l2(matrix: &mut CsrMatrix)
  pub fn sparse_column_sums(matrix: &CsrMatrix) -> Array1<f64>
  pub fn sparse_column_means(matrix: &CsrMatrix) -> Array1<f64>
  pub fn sparse_column_nnz(matrix: &CsrMatrix) -> Array1<usize>
  pub fn sparse_eye(n: usize) -> CsrMatrix
  pub fn sparse_diag(diag: &Array1<f64>) -> CsrMatrix
  pub fn sparse_vstack(matrices: &[&CsrMatrix]) -> Result<CsrMatrix>
  pub fn sparse_hstack(matrices: &[&CsrMatrix]) -> Result<CsrMatrix>

ferroml-core/src/automl/ensemble.rs
  pub struct TrialResult { trial_id: usize, algorithm: AlgorithmType, params: HashMap<String, cv_score: f64, cv_std: f64, fold_scores: Vec<f64>, oof_predictions: Option<Array1<f64>>, oof_probabilities: Option<Array2<f64>>, ... }
  pub struct EnsembleConfig { max_models: usize, selection_iterations: usize, min_weight: f64, use_diversity: bool, diversity_weight: f64, optimize_weights: bool, random_state: Option<u64>, task: Task, ... }
  pub struct EnsembleMember { trial_id: usize, algorithm: AlgorithmType, weight: f64, selection_count: usize, params: HashMap<String }
  pub struct EnsembleResult { members: Vec<EnsembleMember>, ensemble_score: f64, best_single_score: f64, improvement: f64, iterations_performed: usize, score_history: Vec<f64>, task: Task }
  pub struct EnsembleBuilder { ... }
  pub enum ParamValue { Int, Float, String, Bool }
  impl TrialResult {
    pub fn new(trial_id: usize, algorithm: AlgorithmType, cv_score: f64, cv_std: f...) -> Self
    pub fn failed(trial_id: usize, algorithm: AlgorithmType, error: impl Into<String>) -> Self
    pub fn with_params(mut self, params: HashMap<String, ParamValue>) -> Self
    pub fn with_oof_predictions(mut self, predictions: Array1<f64>) -> Self
    pub fn with_oof_probabilities(mut self, probabilities: Array2<f64>) -> Self
    pub fn with_training_time(mut self, seconds: f64) -> Self
    pub fn with_feature_importances(mut self, importances: Vec<f64>) -> Self
    pub fn has_oof_predictions(&self) -> bool
  }
  impl Default for EnsembleConfig { fn default(...) }
  impl EnsembleConfig {
    pub fn new() -> Self
    pub fn with_max_models(mut self, max_models: usize) -> Self
    pub fn with_selection_iterations(mut self, iterations: usize) -> Self
    pub fn with_min_weight(mut self, min_weight: f64) -> Self
    pub fn with_diversity(mut self, weight: f64) -> Self
    pub fn with_weight_optimization(mut self, enabled: bool) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_task(mut self, task: Task) -> Self
    pub fn with_maximize(mut self, maximize: bool) -> Self
  }
  impl EnsembleResult {
    pub fn n_models(&self) -> usize
    pub fn total_weight(&self) -> f64
    pub fn members_by_weight(&self) -> Vec<&EnsembleMember>
    pub fn algorithm_distribution(&self) -> HashMap<AlgorithmType, f64>
  }
  impl EnsembleBuilder {
    pub fn new(config: EnsembleConfig) -> Self
    pub fn build_from_trials(&mut self, trials: &[TrialResult], y_true: &Array1<f64>,) -> Result<EnsembleResult>
  }
  pub fn build_ensemble(trials: &[TrialResult], y_true: &Array1<f64>, task: Task,) -> Result<EnsembleResult>
  pub fn build_ensemble_with_config(trials: &[TrialResult], y_true: &Array1<f64>, config: EnsembleConfig,) -> Result<EnsembleResult>

ferroml-core/src/automl/fit.rs
  pub struct AutoMLResult { trials: Vec<TrialResult>, leaderboard: Vec<LeaderboardEntry>, ensemble: Option<EnsembleResult>, total_time_seconds: f64, n_successful_trials: usize, n_failed_trials: usize, task: Task, metric_name: String, ... }
  pub struct LeaderboardEntry { rank: usize, trial_id: usize, algorithm: AlgorithmType, cv_score: f64, cv_std: f64, ci_lower: f64, ci_upper: f64, training_time_seconds: f64, ... }
  pub struct ModelStatistics { mean_score: f64, std_score: f64, ci_lower: f64, ci_upper: f64, n_folds: usize, fold_scores: Vec<f64> }
  pub struct AggregatedFeatureImportance { feature_names: Vec<String>, importance_mean: Vec<f64>, importance_std: Vec<f64>, ci_lower: Vec<f64>, ci_upper: Vec<f64>, n_models_per_feature: Vec<usize>, n_models: usize, per_model_importance: Vec<ModelFeatureImportance> }
  pub struct ModelFeatureImportance { trial_id: usize, algorithm: AlgorithmType, cv_score: f64, importances: Vec<f64> }
  pub struct ModelComparisonResults { pairwise_comparisons: Vec<PairwiseComparison>, best_is_significantly_better: bool, n_significantly_worse: usize, corrected_alpha: f64, correction_method: String }
  pub struct PairwiseComparison { trial_id_1: usize, trial_id_2: usize, model1_name: String, model2_name: String, test_name: String, mean_difference: f64, std_error: f64, statistic: f64, ... }
  impl AutoMLResult {
    pub fn best_model(&self) -> Option<&LeaderboardEntry>
    pub fn predict(&self, x_train: &Array2<f64>, y_train: &Array1<f64>, x_test: &Array...) -> Result<Array1<f64>>
    pub fn ensemble_score(&self) -> Option<f64>
    pub fn ensemble_improvement(&self) -> Option<f64>
    pub fn is_successful(&self) -> bool
    pub fn n_successful_algorithms(&self) -> usize
    pub fn top_features(&self, k: usize) -> Option<Vec<(String, f64, f64, f64)>>
    pub fn competitive_models(&self) -> Vec<&LeaderboardEntry>
    pub fn models_significantly_different(&self, trial_id_1: usize, trial_id_2: usize) -> bool
    pub fn summary(&self) -> String
  }
  impl AggregatedFeatureImportance {
    pub fn sorted_indices(&self) -> Vec<usize>
    pub fn top_k(&self, k: usize) -> Vec<(String, f64, f64, f64)>
    pub fn is_significant(&self, feature_idx: usize) -> bool
    pub fn significant_features(&self) -> Vec<usize>
    pub fn summary(&self) -> String
  }
  impl ModelComparisonResults {
    pub fn comparisons_for_model(&self, trial_id: usize) -> Vec<&PairwiseComparison>
    pub fn are_significantly_different(&self, trial_id_1: usize, trial_id_2: usize) -> bool
    pub fn models_competitive_with_best(&self) -> Vec<usize>
    pub fn summary(&self) -> String
  }
  impl MetricTrait for MetricAdapter { fn name(...), fn direction(...), fn compute(...) }
  impl AutoML {
    pub fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<AutoMLResult>
  }

ferroml-core/src/automl/metafeatures.rs
  pub struct MetafeatureConfig { compute_landmarking: bool, landmarking_cv_folds: usize, landmarking_max_samples: usize, random_state: Option<u64> }
  pub struct DatasetMetafeatures { simple: SimpleMetafeatures, statistical: StatisticalMetafeatures, information: InformationMetafeatures, landmarking: Option<LandmarkingMetafeatures> }
  pub struct SimpleMetafeatures { n_samples: usize, n_features: usize, n_numeric_features: usize, n_classes: Option<usize>, dimensionality: f64, imbalance_ratio: Option<f64>, missing_ratio: f64, is_high_dimensional: bool, ... }
  pub struct StatisticalMetafeatures { mean_mean: f64, mean_std: f64, std_mean: f64, std_std: f64, skewness_mean: f64, skewness_std: f64, kurtosis_mean: f64, kurtosis_std: f64, ... }
  pub struct FeatureStatistics { index: usize, mean: f64, std: f64, skewness: f64, kurtosis: f64, min: f64, max: f64, n_unique: usize, ... }
  pub struct InformationMetafeatures { target_entropy: f64, max_target_entropy: f64, normalized_entropy: f64, feature_entropy_mean: f64, feature_entropy_std: f64, mutual_info_mean: f64, mutual_info_std: f64, mutual_info_max: f64, ... }
  pub struct LandmarkingMetafeatures { one_nn_score: f64, one_nn_accuracy: f64, decision_stump_score: f64, random_stump_score: f64, naive_bayes_score: f64, linear_model_score: f64, best_feature_score: f64, best_feature_index: usize, ... }
  impl Default for MetafeatureConfig { fn default(...) }
  impl MetafeatureConfig {
    pub fn new() -> Self
    pub fn without_landmarking(mut self) -> Self
    pub fn with_landmarking_cv_folds(mut self, folds: usize) -> Self
    pub fn with_landmarking_max_samples(mut self, max_samples: usize) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
  }
  impl DatasetMetafeatures {
    pub fn extract(x: &Array2<f64>, y: &Array1<f64>, is_classification: bool, config: ...) -> Result<Self>
    pub fn to_vector(&self) -> Vec<f64>
    pub fn similarity(&self, other: &DatasetMetafeatures) -> f64
  }
  impl SimpleMetafeatures {
    pub fn extract(x: &Array2<f64>, y: &Array1<f64>, is_classification: bool) -> Self
  }
  impl StatisticalMetafeatures {
    pub fn extract(x: &Array2<f64>) -> Self
  }
  impl InformationMetafeatures {
    pub fn extract(x: &Array2<f64>, y: &Array1<f64>, is_classification: bool) -> Self
  }
  impl LandmarkingMetafeatures {
    pub fn extract(x: &Array2<f64>, y: &Array1<f64>, is_classification: bool, config: ...) -> Result<Self>
  }

ferroml-core/src/automl/portfolio.rs
  pub struct AlgorithmConfig { algorithm: AlgorithmType, search_space: SearchSpace, preprocessing: Vec<PreprocessingRequirement>, priority: u32, complexity: AlgorithmComplexity, supports_warm_start: bool }
  pub struct DataCharacteristics { n_samples: usize, n_features: usize, n_classes: usize, has_missing_values: bool, has_categorical: bool, feature_variance_ratio: f64, class_imbalance_ratio: f64, is_regression: bool }
  pub struct AlgorithmPortfolio { task: Task, algorithms: Vec<AlgorithmConfig>, preset: PortfolioPreset }
  pub enum AlgorithmType { LogisticRegression, GaussianNB, MultinomialNB, CategoricalNB, KNeighborsClassifier, SVC, LinearSVC, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, LinearRegression, ... }
  pub enum AlgorithmComplexity { Low, Medium, High }
  pub enum PreprocessingRequirement { Scaling, HandleMissing, EncodeCategorical, NonNegative, DimensionalityReduction }
  pub enum PortfolioPreset { Quick, Balanced, Thorough, Custom }
  impl AlgorithmConfig {
    pub fn new(algorithm: AlgorithmType) -> Self
    pub fn with_priority(mut self, priority: u32) -> Self
    pub fn with_search_space(mut self, search_space: SearchSpace) -> Self
    pub fn with_preprocessing(mut self, req: PreprocessingRequirement) -> Self
    pub fn adapt_to_data(&self, characteristics: &DataCharacteristics) -> Self
  }
  impl AlgorithmType {
    pub fn is_classifier(&self) -> bool
    pub fn is_regressor(&self) -> bool
    pub fn name(&self) -> &'static str
  }
  impl AlgorithmComplexity {
    pub fn time_factor(&self) -> f64
  }
  impl Default for DataCharacteristics { fn default(...) }
  impl DataCharacteristics {
    pub fn from_data(x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>) -> Self
    pub fn new(n_samples: usize, n_features: usize) -> Self
    pub fn with_n_classes(mut self, n: usize) -> Self
    pub fn with_missing_values(mut self, has_missing: bool) -> Self
    pub fn with_categorical(mut self, has_categorical: bool) -> Self
    pub fn with_imbalance_ratio(mut self, ratio: f64) -> Self
  }
  impl AlgorithmPortfolio {
    pub fn for_classification(preset: PortfolioPreset) -> Self
    pub fn for_regression(preset: PortfolioPreset) -> Self
    pub fn custom(task: Task) -> Self
    pub fn add(mut self, config: AlgorithmConfig) -> Self
    pub fn remove(&mut self, algorithm: AlgorithmType)
    pub fn adapt_to_data(&self, characteristics: &DataCharacteristics) -> Self
    pub fn sorted_by_priority(&self) -> Vec<&AlgorithmConfig>
    pub fn combined_search_space(&self) -> SearchSpace
  }

ferroml-core/src/automl/preprocessing.rs
  pub struct PreprocessingConfig { strategy: PreprocessingStrategy, handle_imbalance: bool, imbalance_threshold: f64, scaler_type: ScalerType, imputation_strategy: ImputationStrategy, encoding_type: EncodingType, reduce_dimensions: bool, dimension_reduction_threshold: f64, ... }
  pub struct PreprocessingStepSpec { name: String, step_type: PreprocessingStepType, params: HashMap<String, column_indices: Option<Vec<usize>> }
  pub struct PreprocessingPipelineSpec { steps: Vec<PreprocessingStepSpec>, detected_characteristics: DetectedCharacteristics }
  pub struct DetectedCharacteristics { missing_feature_indices: Vec<usize>, missing_percentages: Vec<f64>, categorical_feature_indices: Vec<usize>, high_variance_indices: Vec<usize>, low_variance_indices: Vec<usize>, skewed_feature_indices: Vec<usize>, feature_skewness: Vec<f64>, has_negative_values: bool, ... }
  pub struct PreprocessingSelector { config: PreprocessingConfig }
  pub struct PreprocessingSelection { pipeline_spec: PreprocessingPipelineSpec, resampling_step: Option<PreprocessingStepSpec>, needs_preprocessing: bool, selection_reasons: Vec<String> }
  pub enum PreprocessingStrategy { Auto, Conservative, Standard, Thorough, Passthrough }
  pub enum ScalerType { Standard, MinMax, Robust, MaxAbs, None }
  pub enum ImputationStrategy { Mean, Median, Mode, Constant, KNN }
  pub enum EncodingType { OneHot, Ordinal, Target }
  pub enum PreprocessingStepType { SimpleImputer, KNNImputer, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder, PowerTransformer, QuantileTransformer, VarianceThreshold, ... }
  pub enum StepParam { Int, Float, String, Bool, IntList }
  impl Default for PreprocessingConfig { fn default(...) }
  impl PreprocessingConfig {
    pub fn new() -> Self
    pub fn with_strategy(mut self, strategy: PreprocessingStrategy) -> Self
    pub fn with_handle_imbalance(mut self, handle: bool) -> Self
    pub fn with_imbalance_threshold(mut self, threshold: f64) -> Self
    pub fn with_scaler(mut self, scaler: ScalerType) -> Self
    pub fn with_imputation(mut self, strategy: ImputationStrategy) -> Self
    pub fn with_encoding(mut self, encoding: EncodingType) -> Self
    pub fn with_dimension_reduction(mut self, reduce: bool) -> Self
    pub fn with_dimension_reduction_threshold(mut self, threshold: f64) -> Self
    pub fn with_variance_threshold(mut self, threshold: f64) -> Self
    pub fn with_power_transform(mut self, use_power: bool) -> Self
    pub fn with_skewness_threshold(mut self, threshold: f64) -> Self
  }
  impl PreprocessingStepSpec {
    pub fn new(name: impl Into<String>, step_type: PreprocessingStepType) -> Self
    pub fn with_param(mut self, name: impl Into<String>, value: StepParam) -> Self
    pub fn with_columns(mut self, indices: Vec<usize>) -> Self
  }
  impl Default for PreprocessingSelector { fn default(...) }
  impl PreprocessingSelector {
    pub fn new(config: PreprocessingConfig) -> Self
    pub fn auto() -> Self
    pub fn conservative() -> Self
    pub fn thorough() -> Self
    pub fn analyze_data(&self, x: &Array2<f64>, y: Option<&Array1<f64>>,) -> DetectedCharacteristics
    pub fn build_pipeline(&self, data_chars: &DataCharacteristics, requirements: &[Preprocess...) -> PreprocessingPipelineSpec
    pub fn build_pipeline_for_algorithm(&self, data_chars: &DataCharacteristics, requirements: &[Preprocess...) -> PreprocessingPipelineSpec
    pub fn create_resampling_step(&self, detected: &DetectedCharacteristics,) -> Option<PreprocessingStepSpec>
  }
  impl PreprocessingSelection {
    pub fn has_steps(&self) -> bool
    pub fn step_names(&self) -> Vec<String>
  }
  pub fn select_preprocessing(selector: &PreprocessingSelector, data_chars: &DataCharacteristics, requireme...) -> PreprocessingSelection

ferroml-core/src/automl/time_budget.rs
  pub struct TimeBudgetConfig { total_budget_seconds: u64, strategy: BanditStrategy, min_trials_per_algorithm: usize, early_stopping_threshold: f64, initial_trial_budget: f64, max_trial_time: f64, warmup_trials: usize, complexity_weighted: bool }
  pub struct AlgorithmArm { algorithm_index: usize, algorithm_type: AlgorithmType, n_trials: usize, score_sum: f64, score_sum_sq: f64, best_score: f64, total_time_seconds: f64, active: bool, ... }
  pub struct ArmSelection { algorithm_index: usize, algorithm_type: AlgorithmType, trial_budget_seconds: f64, selection_reason: SelectionReason }
  pub struct TimeBudgetAllocator { config: TimeBudgetConfig, arms: Vec<AlgorithmArm>, total_trials: usize, total_time_spent: f64, global_best_score: f64, best_algorithm_index: Option<usize> }
  pub struct AllocationSummary { total_budget_seconds: u64, time_spent_seconds: f64, total_trials: usize, n_active_arms: usize, global_best_score: f64, best_algorithm: Option<AlgorithmType>, arm_summaries: Vec<ArmSummary> }
  pub struct ArmSummary { algorithm_type: AlgorithmType, n_trials: usize, mean_score: f64, best_score: f64, total_time_seconds: f64, active: bool }
  pub enum BanditStrategy { UCB1, ThompsonSampling, EpsilonGreedy, SuccessiveHalving }
  pub enum SelectionReason { Warmup, Exploration, Exploitation, Random }
  impl Default for TimeBudgetConfig { fn default(...) }
  impl TimeBudgetConfig {
    pub fn new(total_budget_seconds: u64) -> Self
    pub fn with_strategy(mut self, strategy: BanditStrategy) -> Self
    pub fn with_min_trials_per_algorithm(mut self, n: usize) -> Self
    pub fn with_early_stopping_threshold(mut self, threshold: f64) -> Self
    pub fn with_initial_trial_budget(mut self, seconds: f64) -> Self
    pub fn with_max_trial_time(mut self, seconds: f64) -> Self
    pub fn with_warmup_trials(mut self, n: usize) -> Self
    pub fn with_complexity_weighted(mut self, enabled: bool) -> Self
  }
  impl Default for BanditStrategy { fn default(...) }
  impl AlgorithmArm {
    pub fn new(index: usize, config: &AlgorithmConfig) -> Self
    pub fn update(&mut self, score: f64, elapsed_seconds: f64, is_success: bool)
    pub fn mean_score(&self) -> f64
    pub fn variance(&self) -> f64
    pub fn std_dev(&self) -> f64
    pub fn avg_trial_time(&self) -> f64
    pub fn ucb1_score(&self, total_trials: usize, exploration_constant: f64) -> f64
  }
  impl TimeBudgetAllocator {
    pub fn new(config: TimeBudgetConfig, portfolio: &AlgorithmPortfolio) -> Self
    pub fn new_with_seed(config: TimeBudgetConfig, portfolio: &AlgorithmPortfolio, seed: Opt...) -> Self
    pub fn start(&mut self)
    pub fn is_budget_exhausted(&self) -> bool
    pub fn remaining_budget_seconds(&self) -> f64
    pub fn n_active_arms(&self) -> usize
    pub fn select_arm(&mut self) -> Option<ArmSelection>
    pub fn update(&mut self, algorithm_index: usize, score: f64, elapsed_seconds: f64)
    pub fn summary(&self) -> AllocationSummary
    pub fn reset(&mut self)
    pub fn elapsed(&self) -> Duration
    pub fn deactivate_algorithm(&mut self, algorithm_index: usize)
    pub fn reactivate_algorithm(&mut self, algorithm_index: usize)
  }
  impl AllocationSummary {
    pub fn arms_by_score(&self) -> Vec<&ArmSummary>
    pub fn budget_used_percent(&self) -> f64
  }

ferroml-core/src/automl/transfer.rs
  pub struct TransferConfig { shrink_factor: f64, min_configs_for_shrinking: usize, adapt_bounds: bool, confidence_level: f64, max_expansion_factor: f64, preserve_original_bounds: bool }
  pub struct PriorKnowledge { parameter_priors: HashMap<String, algorithm: Option<AlgorithmType>, source_similarity: f64, n_configurations: usize }
  pub struct ParameterPrior { name: String, prior_type: PriorType, log_scale: bool, confidence: f64 }
  pub struct TransferredSearchSpace { search_space: SearchSpace, original_space: SearchSpace, prior: PriorKnowledge, config: TransferConfig, adaptations: HashMap<String }
  pub struct ParameterAdaptation { name: String, original_bounds: (f64, adapted_bounds: (f64, shrink_ratio: f64, was_adapted: bool, reason: String }
  pub struct WarmStartSampler { ... }
  pub enum PriorType { Normal, LogNormal, Categorical, Boolean, Uniform }
  impl Default for TransferConfig { fn default(...) }
  impl TransferConfig {
    pub fn new() -> Self
    pub fn with_shrink_factor(mut self, factor: f64) -> Self
    pub fn with_min_configs(mut self, n: usize) -> Self
    pub fn with_adapt_bounds(mut self, enable: bool) -> Self
    pub fn with_confidence_level(mut self, level: f64) -> Self
    pub fn aggressive() -> Self
    pub fn conservative() -> Self
  }
  impl PriorKnowledge {
    pub fn from_warm_start(warm_start: &WarmStartResult) -> Self
    pub fn sample_prior(&self, name: &str, rng: &mut StdRng) -> Option<f64>
    pub fn most_likely_categorical(&self, name: &str) -> Option<String>
  }
  impl TransferredSearchSpace {
    pub fn from_warm_start(original: &SearchSpace, warm_start: &WarmStartResult, config: Trans...) -> Self
    pub fn from_prior(original: &SearchSpace, prior: &PriorKnowledge, config: TransferCon...) -> Self
    pub fn summary(&self) -> String
  }
  impl WarmStartSampler {
    pub fn new(warm_start: &WarmStartResult) -> Self
    pub fn from_prior(prior: PriorKnowledge) -> Self
    pub fn with_seed(mut self, seed: u64) -> Self
    pub fn with_prior_weight(mut self, weight: f64) -> Self
    pub fn remaining_warm_configs(&self) -> usize
  }
  pub fn initialize_study_with_warmstart(warm_start: &WarmStartResult, _search_space: &SearchSpace,) -> Vec<Trial>
  pub fn algorithm_priorities_from_warmstart(warm_start: &WarmStartResult,) -> HashMap<AlgorithmType, f64>

ferroml-core/src/automl/warmstart.rs
  pub struct WarmStartConfig { k_nearest: usize, min_similarity: f64, max_configs_per_dataset: usize, max_total_configs: usize, weight_by_similarity: bool, normalize_metafeatures: bool, compute_landmarking: bool }
  pub struct ConfigurationRecord { algorithm: AlgorithmType, params: HashMap<String, cv_score: f64, cv_std: f64, maximize: bool, rank: usize }
  pub struct DatasetRecord { name: String, metafeatures: DatasetMetafeatures, task: Task, configurations: Vec<ConfigurationRecord>, created_at: u64, updated_at: u64 }
  pub struct SimilarDataset { name: String, similarity: f64, n_configurations: usize, task: Task }
  pub struct WarmStartResult { similar_datasets: Vec<SimilarDataset>, configurations: Vec<WeightedConfiguration>, total_configs_found: usize, mean_similarity: f64 }
  pub struct WeightedConfiguration { config: ConfigurationRecord, source_similarity: f64, source_dataset: String, priority_weight: f64 }
  pub struct MetaLearningStore { ... }
  pub struct StoreStatistics { total_datasets: usize, total_configurations: usize, classification_datasets: usize, regression_datasets: usize, avg_configs_per_dataset: f64 }
  impl Default for WarmStartConfig { fn default(...) }
  impl WarmStartConfig {
    pub fn new() -> Self
    pub fn with_k_nearest(mut self, k: usize) -> Self
    pub fn with_min_similarity(mut self, threshold: f64) -> Self
    pub fn with_max_configs_per_dataset(mut self, max: usize) -> Self
    pub fn with_max_total_configs(mut self, max: usize) -> Self
    pub fn with_weight_by_similarity(mut self, enable: bool) -> Self
    pub fn with_normalize_metafeatures(mut self, enable: bool) -> Self
    pub fn with_landmarking(mut self, enable: bool) -> Self
  }
  impl ConfigurationRecord {
    pub fn new(algorithm: AlgorithmType, params: HashMap<String, ParamValue>, cv_s...) -> Self
    pub fn from_trial(trial: &TrialResult, maximize: bool, rank: usize) -> Self
  }
  impl DatasetRecord {
    pub fn new(name: impl Into<String>, metafeatures: DatasetMetafeatures, task: Task) -> Self
    pub fn add_configurations(&mut self, trials: &[TrialResult], maximize: bool)
    pub fn top_configurations(&self, n: usize) -> Vec<&ConfigurationRecord>
  }
  impl WarmStartResult {
    pub fn has_configurations(&self) -> bool
    pub fn configurations_by_algorithm(&self,) -> HashMap<AlgorithmType, Vec<&WeightedConfiguration>>
    pub fn prioritized_algorithms(&self) -> Vec<AlgorithmType>
  }
  impl MetaLearningStore {
    pub fn new() -> Self
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool
    pub fn add_record(&mut self, record: DatasetRecord)
    pub fn add_dataset(&mut self, name: impl Into<String>, metafeatures: DatasetMetafeatur...)
    pub fn get(&self, name: &str) -> Option<&DatasetRecord>
    pub fn remove(&mut self, name: &str) -> Option<DatasetRecord>
    pub fn dataset_names(&self) -> Vec<&str>
    pub fn find_similar(&self, query_metafeatures: &DatasetMetafeatures, config: &WarmStart...) -> Vec<(f64, &DatasetRecord)>
    pub fn get_warm_start_configs(&self, query_metafeatures: &DatasetMetafeatures, config: &WarmStart...) -> Result<WarmStartResult>
    pub fn to_json(&self) -> Result<String>
    pub fn from_json(json: &str) -> Result<Self>
    pub fn merge(&mut self, other: MetaLearningStore)
    pub fn stats(&self) -> StoreStatistics
  }

ferroml-core/src/clustering/agglomerative.rs
  pub struct AgglomerativeClustering { n_clusters: usize, linkage: Linkage }
  pub enum Linkage { Single, Complete, Average, Ward }
  impl Default for Linkage { fn default(...) }
  impl AgglomerativeClustering {
    pub fn new(n_clusters: usize) -> Self
    pub fn with_linkage(mut self, linkage: Linkage) -> Self
    pub fn children(&self) -> Option<&Vec<(usize, usize, f64, usize)>>
  }
  impl ClusteringModel for AgglomerativeClustering { fn fit(...), fn predict(...), fn labels(...), fn is_fitted(...) }

ferroml-core/src/clustering/dbscan.rs
  pub struct DBSCAN { ... }
  impl Default for DBSCAN { fn default(...) }
  impl DBSCAN {
    pub fn new(eps: f64, min_samples: usize) -> Self
    pub fn with_gpu(mut self, backend: std::sync::Arc<dyn crate::gpu::GpuBackend>) -> Self
    pub fn eps(mut self, eps: f64) -> Self
    pub fn min_samples(mut self, min_samples: usize) -> Self
    pub fn core_sample_indices(&self) -> Option<&Vec<usize>>
    pub fn components(&self) -> Option<&Array2<f64>>
    pub fn n_clusters(&self) -> Option<usize>
    pub fn n_noise(&self) -> Option<usize>
    pub fn optimal_eps(x: &Array2<f64>, min_samples: usize) -> Result<(f64, Vec<f64>)>
    pub fn cluster_persistence(x: &Array2<f64>, eps_values: &[f64], min_samples: usize,) -> Result<Vec<(f64, usize, usize)>>
    pub fn noise_analysis(&self, x: &Array2<f64>) -> Result<(f64, Array1<f64>, Array1<f64>)>
    pub fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix) -> Result<Array1<i32>>
  }
  impl ClusteringModel for DBSCAN { fn fit(...), fn predict(...), fn labels(...), fn is_fitted(...) }

ferroml-core/src/clustering/diagnostics.rs
  pub struct ClusterDiagnostics { n_clusters: usize, cluster_sizes: Vec<usize>, within_ss: Vec<f64>, total_within_ss: f64, between_ss: f64, centroids: Array2<f64>, silhouette_per_cluster: Vec<f64>, silhouette_overall: f64, ... }
  impl ClusterDiagnostics {
    pub fn from_labels(x: &Array2<f64>, labels: &Array1<i32>, outlier_threshold: Option<f64>,) -> Result<Self>
    pub fn variance_ratio(&self) -> f64
    pub fn dunn_index(&self) -> f64
    pub fn summary(&self) -> String
  }

ferroml-core/src/clustering/gmm.rs
  pub struct GaussianMixture { ... }
  pub enum CovarianceType { Full, Tied, Diagonal, Spherical }
  pub enum GmmInit { KMeans, Random }
  impl Default for GaussianMixture { fn default(...) }
  impl GaussianMixture {
    pub fn new(n_components: usize) -> Self
    pub fn covariance_type(mut self, cov_type: CovarianceType) -> Self
    pub fn max_iter(mut self, max_iter: usize) -> Self
    pub fn tol(mut self, tol: f64) -> Self
    pub fn n_init(mut self, n_init: usize) -> Self
    pub fn init_params(mut self, init: GmmInit) -> Self
    pub fn reg_covar(mut self, reg: f64) -> Self
    pub fn warm_start(mut self, warm: bool) -> Self
    pub fn random_state(mut self, seed: u64) -> Self
    pub fn weights(&self) -> Option<&Array1<f64>>
    pub fn means(&self) -> Option<&Array2<f64>>
    pub fn n_iter(&self) -> Option<usize>
    pub fn converged(&self) -> Option<bool>
    pub fn lower_bound(&self) -> Option<f64>
    pub fn covariances_full(&self) -> Option<&Vec<Array2<f64>>>
    pub fn covariances_tied(&self) -> Option<&Array2<f64>>
    pub fn covariances_diag(&self) -> Option<&Array2<f64>>
    pub fn covariances_spherical(&self) -> Option<&Array1<f64>>
    pub fn score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>>
    pub fn score(&self, x: &Array2<f64>) -> Result<f64>
    pub fn bic(&self, x: &Array2<f64>) -> Result<f64>
    pub fn aic(&self, x: &Array2<f64>) -> Result<f64>
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn sample(&self, n_samples: usize) -> Result<(Array2<f64>, Array1<i32>)>
  }
  impl ClusteringModel for GaussianMixture { fn fit(...), fn predict(...), fn labels(...), fn is_fitted(...) }

ferroml-core/src/clustering/hdbscan.rs
  pub struct HDBSCAN { ... }
  impl HDBSCAN {
    pub fn new(min_cluster_size: usize) -> Self
    pub fn with_gpu(mut self, backend: std::sync::Arc<dyn crate::gpu::GpuBackend>) -> Self
    pub fn with_min_samples(mut self, min_samples: usize) -> Self
    pub fn with_cluster_selection_epsilon(mut self, epsilon: f64) -> Self
    pub fn with_allow_single_cluster(mut self, allow: bool) -> Self
    pub fn probabilities(&self) -> Option<&Array1<f64>>
    pub fn n_clusters(&self) -> Option<usize>
    pub fn n_noise(&self) -> Option<usize>
  }
  impl ClusteringModel for HDBSCAN { fn fit(...), fn predict(...), fn fit_predict(...), fn labels(...), fn is_fitted(...) }

ferroml-core/src/clustering/kmeans.rs
  pub struct KMeans { ... }
  pub enum KMeansAlgorithm { Lloyd, Elkan, Auto }
  impl Default for KMeansAlgorithm { fn default(...) }
  impl Default for KMeans { fn default(...) }
  impl KMeans {
    pub fn new(n_clusters: usize) -> Self
    pub fn max_iter(mut self, max_iter: usize) -> Self
    pub fn tol(mut self, tol: f64) -> Self
    pub fn random_state(mut self, seed: u64) -> Self
    pub fn with_gpu(mut self, backend: std::sync::Arc<dyn crate::gpu::GpuBackend>) -> Self
    pub fn n_init(mut self, n_init: usize) -> Self
    pub fn warm_start(mut self, warm_start: bool) -> Self
    pub fn algorithm(mut self, algorithm: KMeansAlgorithm) -> Self
    pub fn cluster_centers(&self) -> Option<&Array2<f64>>
    pub fn inertia(&self) -> Option<f64>
    pub fn n_iter(&self) -> Option<usize>
    pub fn optimal_k(x: &Array2<f64>, k_range: std::ops::Range<usize>, n_refs: usize, ra...) -> Result<GapStatisticResult>
    pub fn elbow(x: &Array2<f64>, k_range: std::ops::Range<usize>, random_state: Opt...) -> Result<ElbowResult>
  }
  impl ClusteringModel for KMeans { fn fit(...), fn predict(...), fn labels(...), fn is_fitted(...) }
  impl ClusteringStatistics for KMeans { fn cluster_stability(...), fn silhouette_with_ci(...) }

ferroml-core/src/clustering/metrics.rs
  pub fn silhouette_score(x: &Array2<f64>, labels: &Array1<i32>) -> Result<f64>
  pub fn silhouette_samples(x: &Array2<f64>, labels: &Array1<i32>) -> Result<Array1<f64>>
  pub fn calinski_harabasz_score(x: &Array2<f64>, labels: &Array1<i32>) -> Result<f64>
  pub fn davies_bouldin_score(x: &Array2<f64>, labels: &Array1<i32>) -> Result<f64>
  pub fn adjusted_rand_index(labels_true: &Array1<i32>, labels_pred: &Array1<i32>) -> Result<f64>
  pub fn normalized_mutual_info(labels_true: &Array1<i32>, labels_pred: &Array1<i32>) -> Result<f64>
  pub fn hopkins_statistic(x: &Array2<f64>, sample_size: Option<usize>, random_state: Option<u64>,) -> Result<f64>

ferroml-core/src/clustering/mod.rs
  pub struct GapStatisticResult { k_values: Vec<usize>, gap_values: Vec<f64>, gap_se: Vec<f64>, optimal_k: usize }
  pub struct ElbowResult { k_values: Vec<usize>, inertias: Vec<f64>, optimal_k: usize }
  pub trait ClusteringModel {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()>
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>>
    fn fit_predict(&mut self, x: &Array2<f64>) -> Result<Array1<i32>>
    fn labels(&self) -> Option<&Array1<i32>>
    fn is_fitted(&self) -> bool
  }
  pub trait ClusteringStatistics: ClusteringModel {
    fn cluster_stability(&self, x: &Array2<f64>, n_bootstrap: usize) -> Result<Array1<f64>>
    fn silhouette_with_ci(&self, x: &Array2<f64>, confidence: f64) -> Result<(f64, f64, f64)>
  }

ferroml-core/src/cv/curves.rs
  pub struct ScoreSummary { mean: f64, std: f64, ci_lower: f64, ci_upper: f64, confidence_level: f64, n_folds: usize }
  pub struct LearningCurveConfig { confidence_level: f64, n_jobs: i32, shuffle: bool, random_seed: Option<u64> }
  pub struct LearningCurveResult { train_sizes: Vec<usize>, train_scores: Array2<f64>, test_scores: Array2<f64>, train_scores_summary: Vec<ScoreSummary>, test_scores_summary: Vec<ScoreSummary>, n_samples: usize }
  pub struct ValidationCurveConfig { confidence_level: f64, n_jobs: i32 }
  pub struct ValidationCurveResult { param_name: String, param_values: Vec<f64>, train_scores: Array2<f64>, test_scores: Array2<f64>, train_scores_summary: Vec<ScoreSummary>, test_scores_summary: Vec<ScoreSummary> }
  pub trait ParameterSettable: Estimator {
    fn set_param(&mut self, name: &str, value: f64) -> Result<()>
    fn get_param(&self, name: &str) -> Result<f64>
  }
  impl ScoreSummary {
    pub fn from_scores(scores: &[f64], confidence_level: f64) -> Self
    pub fn summary(&self) -> String
  }
  impl Default for LearningCurveConfig { fn default(...) }
  impl LearningCurveConfig {
    pub fn with_confidence(mut self, confidence: f64) -> Self
    pub fn with_n_jobs(mut self, n_jobs: i32) -> Self
    pub fn with_shuffle(mut self, shuffle: bool, seed: Option<u64>) -> Self
  }
  impl LearningCurveResult {
    pub fn iter_summaries(&self,) -> impl Iterator<Item = (usize, &ScoreSummary, &ScoreSummary)> + '_
    pub fn summary(&self) -> String
  }
  impl Default for ValidationCurveConfig { fn default(...) }
  impl ValidationCurveConfig {
    pub fn with_confidence(mut self, confidence: f64) -> Self
    pub fn with_n_jobs(mut self, n_jobs: i32) -> Self
  }
  impl ValidationCurveResult {
    pub fn iter_summaries(&self) -> impl Iterator<Item = (f64, &ScoreSummary, &ScoreSummary)> + '_
    pub fn summary(&self) -> String
    pub fn best_param_value(&self, maximize: bool) -> Option<f64>
  }
  pub fn learning_curve(estimator: &E, x: &Array2<f64>, y: &Array1<f64>, cv: &dyn CrossValidator, met...) -> Result<LearningCurveResult>
where
    E: Estimator + Clone + Send + Sync,
    E::Fitted: Send,
    M: Metric + Sync,
  pub fn validation_curve(estimator: &E, x: &Array2<f64>, y: &Array1<f64>, cv: &dyn CrossValidator, met...) -> Result<ValidationCurveResult>
where
    E: ParameterSettable + Clone + Send + Sync,
    E::Fitted: Send,
    M: Metric + Sync,

ferroml-core/src/cv/group.rs
  pub struct GroupKFold { ... }
  pub struct StratifiedGroupKFold { ... }
  impl GroupKFold {
    pub fn new(n_folds: usize) -> Self
    pub fn n_folds(&self) -> usize
  }
  impl Default for GroupKFold { fn default(...) }
  impl CrossValidator for GroupKFold { fn split(...), fn get_n_splits(...), fn name(...), fn requires_groups(...) }
  impl StratifiedGroupKFold {
    pub fn new(n_folds: usize) -> Self
    pub fn with_shuffle(mut self, shuffle: bool) -> Self
    pub fn with_seed(mut self, seed: u64) -> Self
    pub fn n_folds(&self) -> usize
    pub fn shuffle(&self) -> bool
    pub fn random_seed(&self) -> Option<u64>
  }
  impl Default for StratifiedGroupKFold { fn default(...) }
  impl CrossValidator for StratifiedGroupKFold { fn split(...), fn get_n_splits(...), fn name(...), fn requires_labels(...), fn requires_groups(...) }

ferroml-core/src/cv/kfold.rs
  pub struct KFold { ... }
  pub struct RepeatedKFold { ... }
  impl KFold {
    pub fn new(n_folds: usize) -> Self
    pub fn with_shuffle(mut self, shuffle: bool) -> Self
    pub fn with_seed(mut self, seed: u64) -> Self
    pub fn n_folds(&self) -> usize
    pub fn shuffle(&self) -> bool
    pub fn random_seed(&self) -> Option<u64>
  }
  impl Default for KFold { fn default(...) }
  impl CrossValidator for KFold { fn split(...), fn get_n_splits(...), fn name(...) }
  impl RepeatedKFold {
    pub fn new(n_folds: usize, n_repeats: usize) -> Self
    pub fn with_seed(mut self, seed: u64) -> Self
    pub fn n_folds(&self) -> usize
    pub fn n_repeats(&self) -> usize
    pub fn random_seed(&self) -> Option<u64>
  }
  impl Default for RepeatedKFold { fn default(...) }
  impl CrossValidator for RepeatedKFold { fn split(...), fn get_n_splits(...), fn name(...) }

ferroml-core/src/cv/loo.rs
  pub struct LeavePOut { ... }
  pub struct ShuffleSplit { ... }
  impl LeaveOneOut {
    pub fn new() -> Self
  }
  impl CrossValidator for LeaveOneOut { fn split(...), fn get_n_splits(...), fn name(...) }
  impl LeavePOut {
    pub fn new(p: usize) -> Self
    pub fn p(&self) -> usize
  }
  impl Default for LeavePOut { fn default(...) }
  impl CrossValidator for LeavePOut { fn split(...), fn get_n_splits(...), fn name(...) }
  impl ShuffleSplit {
    pub fn new(n_splits: usize) -> Self
    pub fn with_test_size(mut self, fraction: f64) -> Self
    pub fn with_train_size(mut self, fraction: f64) -> Self
    pub fn with_seed(mut self, seed: u64) -> Self
    pub fn n_splits(&self) -> usize
    pub fn test_fraction(&self) -> f64
    pub fn train_fraction(&self) -> Option<f64>
    pub fn random_seed(&self) -> Option<u64>
  }
  impl Default for ShuffleSplit { fn default(...) }
  impl CrossValidator for ShuffleSplit { fn split(...), fn get_n_splits(...), fn name(...) }

ferroml-core/src/cv/mod.rs
  pub struct CVFold { train_indices: Vec<usize>, test_indices: Vec<usize>, fold_index: usize }
  pub struct CVFoldResult { fold_index: usize, train_score: Option<f64>, test_score: f64, fit_time_secs: f64, score_time_secs: f64 }
  pub struct CVResult { fold_results: Vec<CVFoldResult>, mean_test_score: f64, std_test_score: f64, mean_train_score: Option<f64>, std_train_score: Option<f64>, ci_lower: f64, ci_upper: f64, confidence_level: f64, ... }
  pub struct CVConfig { return_train_score: bool, confidence_level: f64, n_jobs: i32, verbose: u8, random_seed: Option<u64> }
  pub trait CrossValidator: Send + Sync {
    fn split(&self, n_samples: usize, y: Option<&Array1<f64>>, groups:...) -> Result<Vec<CVFold>>
    fn get_n_splits(&self, n_samples: Option<usize>, y: Option<&Array1<f64>>,...) -> usize
    fn name(&self) -> &str
    fn requires_labels(&self) -> bool
    fn requires_groups(&self) -> bool
    fn validate_inputs(&self, n_samples: usize, y: Option<&Array1<f64>>, groups:...) -> Result<()>
  }
  impl CVFold {
    pub fn new(train_indices: Vec<usize>, test_indices: Vec<usize>, fold_index: usize) -> Self
    pub fn n_train(&self) -> usize
    pub fn n_test(&self) -> usize
  }
  impl CVResult {
    pub fn from_fold_results(fold_results: Vec<CVFoldResult>, n_samples: usize, confidence_level...) -> Self
    pub fn scores(&self) -> &[f64]
    pub fn summary(&self) -> String
    pub fn significantly_better_than(&self, other: &CVResult, maximize: bool) -> bool
  }
  impl Default for CVConfig { fn default(...) }
  impl CVConfig {
    pub fn with_train_score(mut self) -> Self
    pub fn with_confidence(mut self, confidence: f64) -> Self
    pub fn with_n_jobs(mut self, n_jobs: i32) -> Self
    pub fn with_seed(mut self, seed: u64) -> Self
  }
  pub fn cross_val_score(estimator: &E, x: &Array2<f64>, y: &Array1<f64>, cv: &dyn CrossValidator, met...) -> Result<CVResult>
where
    E: Estimator + Clone + Send + Sync,
    E::Fitted: Send,
    M: Metric + Sync,
  pub fn cross_val_score_simple(estimator: &E, x: &Array2<f64>, y: &Array1<f64>, cv: &dyn CrossValidator, met...) -> Result<CVResult>
where
    E: Estimator + Clone + Send + Sync,
    E::Fitted: Send,
    M: Metric + Sync,
  pub fn cross_val_score_array(estimator: &E, x: &Array2<f64>, y: &Array1<f64>, cv: &dyn CrossValidator, met...) -> Result<Array1<f64>>
where
    E: Estimator + Clone + Send + Sync,
    E::Fitted: Send,
    M: Metric + Sync,

ferroml-core/src/cv/nested.rs
  pub struct NestedCVConfig { n_trials: usize, use_tpe: bool, confidence_level: f64, return_train_score: bool, n_jobs: i32, seed: Option<u64>, refit_on_full_train: bool, verbose: u8 }
  pub struct NestedCVFoldResult { outer_fold_index: usize, test_score: f64, train_score: Option<f64>, best_params: HashMap<String, best_inner_score: f64, n_trials_completed: usize, hpo_time_secs: f64, eval_time_secs: f64 }
  pub struct NestedCVResult { fold_results: Vec<NestedCVFoldResult>, mean_test_score: f64, std_test_score: f64, mean_train_score: Option<f64>, std_train_score: Option<f64>, ci_lower: f64, ci_upper: f64, confidence_level: f64, ... }
  impl Default for NestedCVConfig { fn default(...) }
  impl NestedCVConfig {
    pub fn with_n_trials(mut self, n_trials: usize) -> Self
    pub fn with_random_sampler(mut self) -> Self
    pub fn with_confidence(mut self, confidence: f64) -> Self
    pub fn with_train_score(mut self) -> Self
    pub fn with_n_jobs(mut self, n_jobs: i32) -> Self
    pub fn with_seed(mut self, seed: u64) -> Self
    pub fn without_refit(mut self) -> Self
  }
  impl NestedCVResult {
    pub fn from_fold_results(fold_results: Vec<NestedCVFoldResult>, n_samples: usize, confidence...) -> Self
    pub fn summary(&self) -> String
    pub fn has_concerning_optimism(&self, threshold: f64) -> bool
  }
  pub fn nested_cv_score(estimator_factory: F, x: &Array2<f64>, y: &Array1<f64>, outer_cv: &dyn CrossV...) -> Result<NestedCVResult>
where
    F: Fn(&HashMap<String, ParameterValue>) -> E + Sync,
    E: Estimator + Clone + Send + Sync,
    E::Fitted: Send,
    M: Metric + Sync,

ferroml-core/src/cv/search.rs
  pub struct SearchResult { params: HashMap<String, mean_test_score: f64, std_test_score: f64, rank: usize }
  pub struct GridSearchCV { maximize: bool, cv_config: CVConfig }
  pub struct RandomizedSearchCV { maximize: bool, random_state: Option<u64>, cv_config: CVConfig }
  impl GridSearchCV {
    pub fn new(param_grid: ParamGrid, n_folds: usize) -> Self
    pub fn with_maximize(mut self, maximize: bool) -> Self
    pub fn best_params(&self) -> Option<&HashMap<String, f64>>
    pub fn best_score(&self) -> Option<f64>
    pub fn cv_results(&self) -> Option<&Vec<SearchResult>>
  }
  impl RandomizedSearchCV {
    pub fn new(param_grid: ParamGrid, n_iter: usize, n_folds: usize) -> Self
    pub fn with_maximize(mut self, maximize: bool) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn best_params(&self) -> Option<&HashMap<String, f64>>
    pub fn best_score(&self) -> Option<f64>
    pub fn cv_results(&self) -> Option<&Vec<SearchResult>>
  }

ferroml-core/src/cv/stratified.rs
  pub struct StratifiedKFold { ... }
  pub struct RepeatedStratifiedKFold { ... }
  impl StratifiedKFold {
    pub fn new(n_folds: usize) -> Self
    pub fn with_shuffle(mut self, shuffle: bool) -> Self
    pub fn with_seed(mut self, seed: u64) -> Self
    pub fn n_folds(&self) -> usize
    pub fn shuffle(&self) -> bool
    pub fn random_seed(&self) -> Option<u64>
  }
  impl Default for StratifiedKFold { fn default(...) }
  impl CrossValidator for StratifiedKFold { fn split(...), fn get_n_splits(...), fn name(...), fn requires_labels(...) }
  impl RepeatedStratifiedKFold {
    pub fn new(n_folds: usize, n_repeats: usize) -> Self
    pub fn with_seed(mut self, seed: u64) -> Self
    pub fn n_folds(&self) -> usize
    pub fn n_repeats(&self) -> usize
    pub fn random_seed(&self) -> Option<u64>
  }
  impl Default for RepeatedStratifiedKFold { fn default(...) }
  impl CrossValidator for RepeatedStratifiedKFold { fn split(...), fn get_n_splits(...), fn name(...), fn requires_labels(...) }

ferroml-core/src/cv/timeseries.rs
  pub struct TimeSeriesSplit { ... }
  impl TimeSeriesSplit {
    pub fn new(n_splits: usize) -> Self
    pub fn with_max_train_size(mut self, size: usize) -> Self
    pub fn with_test_size(mut self, size: usize) -> Self
    pub fn with_gap(mut self, gap: usize) -> Self
    pub fn n_splits(&self) -> usize
    pub fn max_train_size(&self) -> Option<usize>
    pub fn test_size(&self) -> Option<usize>
    pub fn gap(&self) -> usize
  }
  impl Default for TimeSeriesSplit { fn default(...) }
  impl CrossValidator for TimeSeriesSplit { fn split(...), fn get_n_splits(...), fn name(...) }

ferroml-core/src/datasets/loaders.rs
  pub struct CsvOptions { delimiter: u8, has_header: bool, skip_rows: usize, n_rows: Option<usize>, columns: Option<Vec<String>>, null_values: Option<Vec<String>>, infer_schema_length: Option<usize>, try_parse_dates: bool, ... }
  pub struct ParquetOptions { columns: Option<Vec<String>>, parallel: bool }
  pub enum CsvEncoding { Utf8, Utf8Lossy }
  impl Default for CsvOptions { fn default(...) }
  impl CsvOptions {
    pub fn new() -> Self
    pub fn with_delimiter(mut self, delimiter: u8) -> Self
    pub fn with_has_header(mut self, has_header: bool) -> Self
    pub fn with_skip_rows(mut self, skip_rows: usize) -> Self
    pub fn with_n_rows(mut self, n_rows: usize) -> Self
    pub fn with_columns(mut self, columns: Vec<String>) -> Self
    pub fn with_null_values(mut self, null_values: Vec<String>) -> Self
    pub fn with_infer_schema_length(mut self, length: Option<usize>) -> Self
    pub fn with_try_parse_dates(mut self, try_parse_dates: bool) -> Self
    pub fn with_encoding(mut self, encoding: CsvEncoding) -> Self
  }
  impl ParquetOptions {
    pub fn new() -> Self
    pub fn with_columns(mut self, columns: Vec<String>) -> Self
    pub fn with_parallel(mut self, parallel: bool) -> Self
  }

ferroml-core/src/datasets/mmap.rs
  pub struct MemmappedArray2 { ... }
  pub struct MemmappedArray2Mut { ... }
  pub struct MemmappedArray1 { ... }
  pub struct MemmappedDataset { ... }
  pub struct BatchIterator { ... }
  pub struct SampleIterator { ... }
  pub struct MemmappedDatasetBuilder { ... }
  impl MemmappedArray2 {
    pub fn shape(&self) -> (usize, usize)
    pub fn n_rows(&self) -> usize
    pub fn n_cols(&self) -> usize
    pub fn view(&self) -> ArrayView2<'_, f64>
    pub fn row(&self, idx: usize) -> Option<&[f64]>
    pub fn get(&self, row: usize, col: usize) -> Option<f64>
    pub fn rows_to_array(&self, start: usize, end: usize) -> Result<Array2<f64>>
  }
  impl MemmappedArray2Mut {
    pub fn view_mut(&mut self) -> ArrayViewMut2<'_, f64>
    pub fn view(&self) -> ArrayView2<'_, f64>
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<()>
    pub fn flush(&self) -> Result<()>
  }
  impl MemmappedArray1 {
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool
    pub fn view(&self) -> ArrayView1<'_, f64>
    pub fn get(&self, idx: usize) -> Option<f64>
    pub fn slice_to_array(&self, start: usize, end: usize) -> Result<Array1<f64>>
  }
  impl MemmappedDataset {
    pub fn path(&self) -> &Path
    pub fn n_samples(&self) -> usize
    pub fn n_features(&self) -> usize
    pub fn shape(&self) -> (usize, usize)
    pub fn has_targets(&self) -> bool
    pub fn x_view(&self) -> ArrayView2<'_, f64>
    pub fn y_view(&self) -> Option<ArrayView1<'_, f64>>
    pub fn row(&self, idx: usize) -> Option<ArrayView1<'_, f64>>
    pub fn get(&self, row: usize, col: usize) -> Option<f64>
    pub fn get_target(&self, idx: usize) -> Option<f64>
    pub fn x_rows(&self, start: usize, end: usize) -> Result<Array2<f64>>
    pub fn y_slice(&self, start: usize, end: usize) -> Result<Array1<f64>>
    pub fn to_dataset(&self) -> Dataset
    pub fn batches(&self, batch_size: usize) -> BatchIterator<'_>
    pub fn samples(&self) -> SampleIterator<'_>
  }
  impl Iterator for BatchIterator<'a> { fn next(...), fn size_hint(...) }
  impl Iterator for SampleIterator<'a> { fn next(...), fn size_hint(...) }
  impl MemmappedDatasetBuilder {
    pub fn with_features(mut self, x: Array2<f64>) -> Self
    pub fn with_targets(mut self, y: Array1<f64>) -> Self
    pub fn from_dataset(mut self, dataset: &Dataset) -> Self
    pub fn build(self) -> Result<MemmappedDataset>
  }

ferroml-core/src/datasets/mod.rs
  pub struct Dataset { ... }
  pub struct FeatureStatistics { name: Option<String>, mean: f64, std: f64, min: f64, max: f64, median: f64, n_missing: usize }
  pub struct DatasetStatistics { n_samples: usize, n_features: usize, feature_stats: Vec<FeatureStatistics> }
  pub struct DatasetInfo { name: String, description: String, task: Task, n_samples: usize, n_features: usize, n_classes: Option<usize>, feature_names: Vec<String>, target_names: Option<Vec<String>>, ... }
  pub struct LoadOptions { shuffle: bool, random_state: Option<u64>, as_sparse: bool, feature_indices: Option<Vec<usize>>, sample_indices: Option<Vec<usize>> }
  pub trait DatasetLoader {
    fn load(&self) -> Result<(Dataset, DatasetInfo)>
    fn name(&self) -> &str
    fn description(&self) -> &str
  }
  impl Dataset {
    pub fn new(x: Array2<f64>, y: Array1<f64>) -> Self
    pub fn try_new(x: Array2<f64>, y: Array1<f64>) -> Result<Self>
    pub fn from_features(x: Array2<f64>) -> Self
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self
    pub fn with_target_names(mut self, names: Vec<String>) -> Self
    pub fn with_sample_ids(mut self, ids: Vec<String>) -> Self
    pub fn n_samples(&self) -> usize
    pub fn n_features(&self) -> usize
    pub fn shape(&self) -> (usize, usize)
    pub fn x(&self) -> &Array2<f64>
    pub fn y(&self) -> &Array1<f64>
    pub fn feature_names(&self) -> Option<&[String]>
    pub fn target_names(&self) -> Option<&[String]>
    pub fn sample_ids(&self) -> Option<&[String]>
    pub fn into_arrays(self) -> (Array2<f64>, Array1<f64>)
    pub fn as_arrays(&self) -> (&Array2<f64>, &Array1<f64>)
    pub fn train_test_split(&self, test_size: f64, shuffle: bool, random_state: Option<u64>,) -> Result<(Dataset, Dataset)>
    pub fn unique_classes(&self) -> Vec<f64>
    pub fn class_counts(&self) -> std::collections::HashMap<i64, usize>
    pub fn is_binary(&self) -> bool
    pub fn is_multiclass(&self) -> bool
    pub fn infer_task(&self) -> Task
    pub fn describe(&self) -> DatasetStatistics
  }
  impl DatasetInfo {
    pub fn new(name: impl Into<String>, task: Task, n_samples: usize, n_features: ...) -> Self
    pub fn with_description(mut self, desc: impl Into<String>) -> Self
    pub fn with_n_classes(mut self, n: usize) -> Self
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self
    pub fn with_target_names(mut self, names: Vec<String>) -> Self
    pub fn with_source(mut self, source: impl Into<String>) -> Self
    pub fn with_url(mut self, url: impl Into<String>) -> Self
    pub fn with_license(mut self, license: impl Into<String>) -> Self
    pub fn with_version(mut self, version: impl Into<String>) -> Self
  }
  impl LoadOptions {
    pub fn new() -> Self
    pub fn with_shuffle(mut self, shuffle: bool) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn as_sparse(mut self) -> Self
    pub fn with_features(mut self, indices: Vec<usize>) -> Self
    pub fn with_samples(mut self, indices: Vec<usize>) -> Self
  }
  pub fn make_classification(n_samples: usize, n_features: usize, n_informative: usize, n_classes: usize, ...) -> (Dataset, DatasetInfo)
  pub fn make_regression(n_samples: usize, n_features: usize, n_informative: usize, noise: f64, random...) -> (Dataset, DatasetInfo)
  pub fn make_blobs(n_samples: usize, n_features: usize, centers: usize, cluster_std: f64, random...) -> (Dataset, DatasetInfo)
  pub fn make_moons(n_samples: usize, noise: f64, random_state: Option<u64>,) -> (Dataset, DatasetInfo)
  pub fn make_circles(n_samples: usize, noise: f64, factor: f64, random_state: Option<u64>,) -> (Dataset, DatasetInfo)

ferroml-core/src/datasets/toy.rs
  pub fn load_iris() -> (Dataset, DatasetInfo)
  pub fn load_wine() -> (Dataset, DatasetInfo)
  pub fn load_diabetes() -> (Dataset, DatasetInfo)
  pub fn load_linnerud() -> (Dataset, DatasetInfo)

ferroml-core/src/decomposition/factor_analysis.rs
  pub struct FactorAnalysis { ... }
  pub enum Rotation { None, Varimax, Quartimax, Promax }
  pub enum FaSvdMethod { Randomized, Full }
  impl FactorAnalysis {
    pub fn new() -> Self
    pub fn with_n_factors(mut self, n: usize) -> Self
    pub fn with_rotation(mut self, rotation: Rotation) -> Self
    pub fn with_svd_method(mut self, method: FaSvdMethod) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_promax_power(mut self, power: f64) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn loadings(&self) -> Option<&Array2<f64>>
    pub fn noise_variance(&self) -> Option<&Array1<f64>>
    pub fn communalities(&self) -> Option<&Array1<f64>>
    pub fn explained_variance(&self) -> Option<&Array1<f64>>
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>>
    pub fn mean(&self) -> Option<&Array1<f64>>
    pub fn n_factors_(&self) -> Option<usize>
    pub fn n_iter_(&self) -> Option<usize>
    pub fn log_likelihood_(&self) -> Option<f64>
    pub fn factor_correlation(&self) -> Option<&Array2<f64>>
    pub fn get_covariance(&self) -> Option<Array2<f64>>
  }
  impl Default for FactorAnalysis { fn default(...) }
  impl Transformer for FactorAnalysis { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }

ferroml-core/src/decomposition/lda.rs
  pub struct LDA { ... }
  pub enum LdaSolver { Svd, Eigen }
  impl LDA {
    pub fn new() -> Self
    pub fn with_n_components(mut self, n: usize) -> Self
    pub fn with_solver(mut self, solver: LdaSolver) -> Self
    pub fn with_shrinkage(mut self, shrinkage: f64) -> Self
    pub fn with_priors(mut self, priors: Vec<f64>) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn fit_transform(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>>
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn scalings(&self) -> Option<&Array2<f64>>
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>>
    pub fn eigenvalues(&self) -> Option<&Array1<f64>>
    pub fn means(&self) -> Option<&Array2<f64>>
    pub fn xbar(&self) -> Option<&Array1<f64>>
    pub fn classes(&self) -> Option<&[f64]>
    pub fn priors(&self) -> Option<&Array1<f64>>
    pub fn n_components_(&self) -> Option<usize>
    pub fn coef(&self) -> Option<&Array2<f64>>
    pub fn intercept(&self) -> Option<&Array1<f64>>
    pub fn is_fitted(&self) -> bool
    pub fn n_features_in(&self) -> Option<usize>
    pub fn n_features_out(&self) -> Option<usize>
    pub fn get_feature_names_out(&self, _input_names: Option<&[String]>) -> Option<Vec<String>>
  }
  impl Default for LDA { fn default(...) }

ferroml-core/src/decomposition/pca.rs
  pub struct PCA { ... }
  pub struct IncrementalPCA { ... }
  pub enum SvdSolver { Auto, Full, Randomized }
  pub enum NComponents { N, VarianceRatio, All }
  impl PCA {
    pub fn new() -> Self
    pub fn with_n_components(mut self, n: usize) -> Self
    pub fn with_variance_ratio(mut self, ratio: f64) -> Self
    pub fn with_whiten(mut self, whiten: bool) -> Self
    pub fn with_svd_solver(mut self, solver: SvdSolver) -> Self
    pub fn components(&self) -> Option<&Array2<f64>>
    pub fn explained_variance(&self) -> Option<&Array1<f64>>
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>>
    pub fn cumulative_explained_variance_ratio(&self) -> Option<Array1<f64>>
    pub fn singular_values(&self) -> Option<&Array1<f64>>
    pub fn mean(&self) -> Option<&Array1<f64>>
    pub fn n_components_(&self) -> Option<usize>
    pub fn noise_variance(&self) -> Option<f64>
    pub fn loadings(&self) -> Option<Array2<f64>>
    pub fn get_covariance(&self) -> Option<Array2<f64>>
  }
  impl Default for PCA { fn default(...) }
  impl Transformer for PCA { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl IncrementalPCA {
    pub fn new() -> Self
    pub fn with_n_components(mut self, n: usize) -> Self
    pub fn with_whiten(mut self, whiten: bool) -> Self
    pub fn with_batch_size(mut self, size: usize) -> Self
    pub fn partial_fit(&mut self, x: &Array2<f64>) -> Result<()>
    pub fn components(&self) -> Option<&Array2<f64>>
    pub fn explained_variance(&self) -> Option<&Array1<f64>>
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>>
    pub fn singular_values(&self) -> Option<&Array1<f64>>
    pub fn mean(&self) -> Option<&Array1<f64>>
    pub fn n_samples_seen(&self) -> usize
  }
  impl Default for IncrementalPCA { fn default(...) }
  impl Transformer for IncrementalPCA { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl PipelineTransformer for PCA { fn clone_boxed(...), fn name(...) }

ferroml-core/src/decomposition/quadtree.rs
  pub struct QuadTree { ... }
  impl QuadTree {
    pub fn new(embedding: &Array2<f64>) -> Self
    pub fn compute_non_edge_forces(&self, point_x: f64, point_y: f64, theta: f64,) -> (f64, f64, f64)
    pub fn total_mass(&self) -> f64
    pub fn center_of_mass(&self) -> Option<(f64, f64)>
  }

ferroml-core/src/decomposition/truncated_svd.rs
  pub struct TruncatedSVD { ... }
  pub enum TruncatedSvdAlgorithm { Auto, Randomized, Arpack }
  impl TruncatedSVD {
    pub fn new() -> Self
    pub fn with_n_components(mut self, n: usize) -> Self
    pub fn with_n_iter(mut self, n: usize) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_algorithm(mut self, algorithm: TruncatedSvdAlgorithm) -> Self
    pub fn with_n_oversamples(mut self, n: usize) -> Self
    pub fn components(&self) -> Option<&Array2<f64>>
    pub fn explained_variance(&self) -> Option<&Array1<f64>>
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>>
    pub fn singular_values(&self) -> Option<&Array1<f64>>
  }
  impl Default for TruncatedSVD { fn default(...) }
  impl Transformer for TruncatedSVD { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }

ferroml-core/src/decomposition/tsne.rs
  pub struct TSNE { ... }
  pub enum TsneMetric { Euclidean, Manhattan, Cosine }
  pub enum TsneInit { Pca, Random }
  pub enum LearningRate { Auto, Fixed }
  pub enum TsneMethod { Exact, BarnesHut }
  impl Default for TSNE { fn default(...) }
  impl TSNE {
    pub fn new() -> Self
    pub fn with_gpu(mut self, backend: std::sync::Arc<dyn crate::gpu::GpuBackend>) -> Self
    pub fn with_n_components(mut self, n: usize) -> Self
    pub fn with_perplexity(mut self, perplexity: f64) -> Self
    pub fn with_learning_rate(mut self, lr: LearningRate) -> Self
    pub fn with_learning_rate_f64(mut self, lr: f64) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_early_exaggeration(mut self, factor: f64) -> Self
    pub fn with_min_grad_norm(mut self, norm: f64) -> Self
    pub fn with_metric(mut self, metric: TsneMetric) -> Self
    pub fn with_init(mut self, init: TsneInit) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_method(mut self, method: TsneMethod) -> Self
    pub fn with_theta(mut self, theta: f64) -> Self
    pub fn embedding(&self) -> Option<&Array2<f64>>
    pub fn kl_divergence(&self) -> Option<f64>
    pub fn n_iter_final(&self) -> Option<usize>
    pub fn n_features_in(&self) -> Option<usize>
  }
  impl Transformer for TSNE { fn fit(...), fn transform(...), fn fit_transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }

ferroml-core/src/decomposition/vptree.rs
  pub struct VPTree { ... }
  impl PartialEq for Neighbor { fn eq(...) }
  impl PartialOrd for Neighbor { fn partial_cmp(...) }
  impl Ord for Neighbor { fn cmp(...) }
  impl VPTree {
    pub fn new(data: &[Vec<f64>]) -> Self
    pub fn from_array(data: &ndarray::Array2<f64>) -> Self
    pub fn search(&self, query: &[f64], k: usize) -> Vec<(usize, f64)>
  }

ferroml-core/src/ensemble/bagging.rs
  pub struct BaggingClassifier { ... }
  pub struct BaggingRegressor { ... }
  pub enum MaxFeatures { All, Sqrt, Log2, Fixed, Fraction }
  pub enum MaxSamples { All, Fixed, Fraction }
  pub trait CloneableModel: Model {
    fn clone_model(&self) -> Box<dyn Model>
  }
  impl Default for MaxFeatures { fn default(...) }
  impl MaxFeatures {
    pub fn compute(&self, n_features: usize) -> usize
  }
  impl Default for MaxSamples { fn default(...) }
  impl MaxSamples {
    pub fn compute(&self, n_samples: usize) -> usize
  }
  impl BaggingClassifier {
    pub fn new(base_estimator: Box<dyn VotingClassifierEstimator>) -> Self
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self
    pub fn with_max_samples(mut self, max_samples: MaxSamples) -> Self
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self
    pub fn with_bootstrap_features(mut self, bootstrap_features: bool) -> Self
    pub fn with_oob_score(mut self, oob_score: bool) -> Self
    pub fn with_random_state(mut self, random_state: u64) -> Self
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self
    pub fn with_warm_start(mut self, warm_start: bool) -> Self
    pub fn oob_score(&self) -> Option<f64>
    pub fn oob_decision_function(&self) -> Option<&Array2<f64>>
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn n_fitted_estimators(&self) -> usize
    pub fn estimator_features(&self) -> &[Vec<usize>]
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for BaggingClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn n_features(...) }
  impl BaggingRegressor {
    pub fn new(base_estimator: Box<dyn Model>) -> Self
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self
    pub fn with_max_samples(mut self, max_samples: MaxSamples) -> Self
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self
    pub fn with_bootstrap_features(mut self, bootstrap_features: bool) -> Self
    pub fn with_oob_score(mut self, oob_score: bool) -> Self
    pub fn with_random_state(mut self, random_state: u64) -> Self
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self
    pub fn with_warm_start(mut self, warm_start: bool) -> Self
    pub fn oob_score(&self) -> Option<f64>
    pub fn oob_predictions(&self) -> Option<&Array1<f64>>
    pub fn n_fitted_estimators(&self) -> usize
    pub fn estimator_features(&self) -> &[Vec<usize>]
    pub fn base_estimator(&self) -> &dyn Model
    pub fn individual_predictions(&self, x: &Array2<f64>) -> Result<Vec<Array1<f64>>>
  }
  impl Model for BaggingRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn n_features(...), fn score(...) }

ferroml-core/src/ensemble/stacking.rs
  pub struct StackingClassifier { ... }
  pub struct StackingRegressor { ... }
  pub enum StackMethod { Predict, PredictProba }
  impl Default for StackMethod { fn default(...) }
  impl StackingClassifier {
    pub fn new(estimators: Vec<(impl Into<String>, Box<dyn VotingClassifierEstimator>)
    pub fn with_final_estimator(mut self, estimator: Box<dyn Model>) -> Self
    pub fn with_cv(mut self, cv: Box<dyn CrossValidator>) -> Self
    pub fn with_n_folds(mut self, n_folds: usize) -> Self
    pub fn with_passthrough(mut self, passthrough: bool) -> Self
    pub fn with_stack_method(mut self, method: StackMethod) -> Self
    pub fn with_n_jobs(mut self, n_jobs: Option<i32>) -> Self
    pub fn estimator_names(&self) -> Vec<&str>
    pub fn get_estimator(&self, name: &str) -> Option<&dyn VotingClassifierEstimator>
    pub fn get_fitted_estimator(&self, name: &str) -> Option<&dyn VotingClassifierEstimator>
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn stack_method(&self) -> StackMethod
    pub fn passthrough(&self) -> bool
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for StackingClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn n_features(...) }
  impl StackingRegressor {
    pub fn new(estimators: Vec<(impl Into<String>, Box<dyn VotingRegressorEstimator>)
    pub fn with_final_estimator(mut self, estimator: Box<dyn Model>) -> Self
    pub fn with_cv(mut self, cv: Box<dyn CrossValidator>) -> Self
    pub fn with_n_folds(mut self, n_folds: usize) -> Self
    pub fn with_passthrough(mut self, passthrough: bool) -> Self
    pub fn with_n_jobs(mut self, n_jobs: Option<i32>) -> Self
    pub fn estimator_names(&self) -> Vec<&str>
    pub fn get_estimator(&self, name: &str) -> Option<&dyn VotingRegressorEstimator>
    pub fn get_fitted_estimator(&self, name: &str) -> Option<&dyn VotingRegressorEstimator>
    pub fn passthrough(&self) -> bool
    pub fn individual_predictions(&self, x: &Array2<f64>) -> Result<Vec<Array1<f64>>>
  }
  impl Model for StackingRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn n_features(...) }

ferroml-core/src/ensemble/voting.rs
  pub struct VotingClassifier { ... }
  pub struct VotingRegressor { ... }
  pub enum VotingMethod { Hard, Soft }
  pub trait VotingClassifierEstimator: Model {
    fn predict_proba_for_voting(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    fn clone_boxed(&self) -> Box<dyn VotingClassifierEstimator>
  }
  pub trait VotingRegressorEstimator: Model {
    fn clone_boxed(&self) -> Box<dyn VotingRegressorEstimator>
  }
  impl Default for VotingMethod { fn default(...) }
  impl VotingClassifier {
    pub fn new(estimators: Vec<(impl Into<String>, Box<dyn VotingClassifierEstimator>)
    pub fn with_hard_voting(mut self) -> Self
    pub fn with_soft_voting(mut self) -> Self
    pub fn with_voting(mut self, voting: VotingMethod) -> Self
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self
    pub fn with_n_jobs(mut self, n_jobs: Option<i32>) -> Self
    pub fn voting_method(&self) -> VotingMethod
    pub fn estimator_names(&self) -> Vec<&str>
    pub fn get_estimator(&self, name: &str) -> Option<&dyn VotingClassifierEstimator>
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn weights(&self) -> Option<&[f64]>
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for VotingClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn n_features(...) }
  impl VotingRegressor {
    pub fn new(estimators: Vec<(impl Into<String>, Box<dyn VotingRegressorEstimator>)
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self
    pub fn with_n_jobs(mut self, n_jobs: Option<i32>) -> Self
    pub fn estimator_names(&self) -> Vec<&str>
    pub fn get_estimator(&self, name: &str) -> Option<&dyn VotingRegressorEstimator>
    pub fn weights(&self) -> Option<&[f64]>
    pub fn individual_predictions(&self, x: &Array2<f64>) -> Result<Vec<Array1<f64>>>
  }
  impl Model for VotingRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn n_features(...) }
  impl VotingClassifierEstimator for LogisticRegression { fn predict_proba_for_voting(...), fn clone_boxed(...) }
  impl VotingClassifierEstimator for DecisionTreeClassifier { fn predict_proba_for_voting(...), fn clone_boxed(...) }
  impl VotingClassifierEstimator for RandomForestClassifier { fn predict_proba_for_voting(...), fn clone_boxed(...) }
  impl VotingClassifierEstimator for GaussianNB { fn predict_proba_for_voting(...), fn clone_boxed(...) }
  impl VotingClassifierEstimator for MultinomialNB { fn predict_proba_for_voting(...), fn clone_boxed(...) }
  impl VotingClassifierEstimator for BernoulliNB { fn predict_proba_for_voting(...), fn clone_boxed(...) }
  impl VotingClassifierEstimator for KNeighborsClassifier { fn predict_proba_for_voting(...), fn clone_boxed(...) }
  impl VotingClassifierEstimator for SVC { fn predict_proba_for_voting(...), fn clone_boxed(...) }
  impl VotingClassifierEstimator for GradientBoostingClassifier { fn predict_proba_for_voting(...), fn clone_boxed(...) }
  impl VotingClassifierEstimator for HistGradientBoostingClassifier { fn predict_proba_for_voting(...), fn clone_boxed(...) }
  impl VotingRegressorEstimator for LinearRegression { fn clone_boxed(...) }
  impl VotingRegressorEstimator for RidgeRegression { fn clone_boxed(...) }
  impl VotingRegressorEstimator for LassoRegression { fn clone_boxed(...) }
  impl VotingRegressorEstimator for ElasticNet { fn clone_boxed(...) }
  impl VotingRegressorEstimator for DecisionTreeRegressor { fn clone_boxed(...) }
  impl VotingRegressorEstimator for RandomForestRegressor { fn clone_boxed(...) }
  impl VotingRegressorEstimator for KNeighborsRegressor { fn clone_boxed(...) }
  impl VotingRegressorEstimator for SVR { fn clone_boxed(...) }
  impl VotingRegressorEstimator for GradientBoostingRegressor { fn clone_boxed(...) }
  impl VotingRegressorEstimator for HistGradientBoostingRegressor { fn clone_boxed(...) }

ferroml-core/src/explainability/h_statistic.rs
  pub struct HStatisticResult { h_squared: f64, h_statistic: f64, feature_idx_1: usize, feature_idx_2: usize, feature_name_1: Option<String>, feature_name_2: Option<String>, ci: Option<(f64, std_error: Option<f64>, ... }
  pub struct HStatisticOverallResult { h_squared: f64, h_statistic: f64, feature_idx: usize, feature_name: Option<String>, ci: Option<(f64, n_samples: usize }
  pub struct HStatisticMatrix { h_squared_matrix: Array2<f64>, feature_indices: Vec<usize>, feature_names: Option<Vec<String>>, n_samples: usize }
  pub struct HStatisticConfig { n_grid_points: usize, grid_method: GridMethod, n_bootstrap: usize, n_permutations: usize, random_state: Option<u64>, confidence_level: f64, max_samples: Option<usize> }
  impl HStatisticResult {
    pub fn interpretation(&self) -> &'static str
    pub fn is_significant(&self, alpha: f64) -> Option<bool>
    pub fn summary(&self) -> String
  }
  impl HStatisticOverallResult {
    pub fn interpretation(&self) -> &'static str
    pub fn summary(&self) -> String
  }
  impl HStatisticMatrix {
    pub fn get(&self, feature_idx_1: usize, feature_idx_2: usize) -> Option<f64>
    pub fn top_k(&self, k: usize) -> Vec<(usize, usize, f64)>
    pub fn summary(&self) -> String
  }
  impl Default for HStatisticConfig { fn default(...) }
  impl HStatisticConfig {
    pub fn new() -> Self
    pub fn with_grid_points(mut self, n: usize) -> Self
    pub fn with_grid_method(mut self, method: GridMethod) -> Self
    pub fn with_bootstrap(mut self, n_bootstrap: usize) -> Self
    pub fn with_permutation_test(mut self, n_permutations: usize) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_confidence_level(mut self, level: f64) -> Self
    pub fn with_max_samples(mut self, n: usize) -> Self
  }
  pub fn h_statistic(model: &M, x: &Array2<f64>, feature_idx_1: usize, feature_idx_2: usize, confi...) -> Result<HStatisticResult>
where
    M: Model,
  pub fn h_statistic_parallel(model: &M, x: &Array2<f64>, feature_idx_1: usize, feature_idx_2: usize, confi...) -> Result<HStatisticResult>
where
    M: Model + Sync,
  pub fn h_statistic_matrix(model: &M, x: &Array2<f64>, feature_indices: Option<&[usize]>, config: HStati...) -> Result<HStatisticMatrix>
where
    M: Model,
  pub fn h_statistic_matrix_parallel(model: &M, x: &Array2<f64>, feature_indices: Option<&[usize]>, config: HStati...) -> Result<HStatisticMatrix>
where
    M: Model + Sync,
  pub fn h_statistic_overall(model: &M, x: &Array2<f64>, feature_idx: usize, config: HStatisticConfig,) -> Result<HStatisticOverallResult>
where
    M: Model,

ferroml-core/src/explainability/ice.rs
  pub struct ICEConfig { n_grid_points: usize, grid_method: GridMethod, center: bool, center_reference_idx: usize, compute_derivative: bool, sample_indices: Option<Vec<usize>> }
  pub struct ICEResult { grid_values: Array1<f64>, ice_curves: Array2<f64>, centered_ice: Option<Array2<f64>>, derivative_ice: Option<Array2<f64>>, pdp_values: Array1<f64>, centered_pdp: Option<Array1<f64>>, feature_idx: usize, feature_name: Option<String>, ... }
  impl Default for ICEConfig { fn default(...) }
  impl ICEConfig {
    pub fn new() -> Self
    pub fn with_n_grid_points(mut self, n: usize) -> Self
    pub fn with_grid_method(mut self, method: GridMethod) -> Self
    pub fn with_centering(mut self, center_reference_idx: usize) -> Self
    pub fn with_derivative(mut self) -> Self
    pub fn with_sample_indices(mut self, indices: Vec<usize>) -> Self
  }
  impl ICEResult {
    pub fn min_value(&self) -> f64
    pub fn max_value(&self) -> f64
    pub fn value_range(&self) -> f64
    pub fn heterogeneity(&self) -> Array1<f64>
    pub fn mean_heterogeneity(&self) -> f64
    pub fn has_interactions(&self, threshold: f64) -> bool
    pub fn sample_with_strongest_positive_effect(&self) -> Option<usize>
    pub fn sample_with_strongest_negative_effect(&self) -> Option<usize>
    pub fn per_sample_effect_range(&self) -> Array1<f64>
    pub fn fraction_monotonic_increasing(&self) -> f64
    pub fn fraction_monotonic_decreasing(&self) -> f64
    pub fn summary(&self) -> String
  }
  pub fn individual_conditional_expectation(model: &M, x: &Array2<f64>, feature_idx: usize, config: ICEConfig,) -> Result<ICEResult>
where
    M: Model,
  pub fn individual_conditional_expectation_parallel(model: &M, x: &Array2<f64>, feature_idx: usize, config: ICEConfig,) -> Result<ICEResult>
where
    M: Model + Sync,
  pub fn center_ice_curves(ice_curves: &Array2<f64>, reference_idx: usize) -> Array2<f64>
  pub fn compute_derivative_ice(ice_curves: &Array2<f64>, grid_values: &Array1<f64>) -> Array2<f64>
  pub fn ice_multi(model: &M, x: &Array2<f64>, feature_indices: &[usize], config: ICEConfig,) -> Result<Vec<ICEResult>>
where
    M: Model,
  pub fn ice_multi_parallel(model: &M, x: &Array2<f64>, feature_indices: &[usize], config: ICEConfig,) -> Result<Vec<ICEResult>>
where
    M: Model + Sync,
  pub fn ice_from_curves(ice_curves: Array2<f64>, grid_values: Array1<f64>, feature_idx: usize, featur...) -> ICEResult

ferroml-core/src/explainability/kernelshap.rs
  pub struct KernelSHAPConfig { n_samples: Option<usize>, max_background_samples: usize, random_state: Option<u64>, regularization: f64, paired_sampling: bool }
  pub struct KernelExplainer { ... }
  impl Default for KernelSHAPConfig { fn default(...) }
  impl KernelSHAPConfig {
    pub fn new() -> Self
    pub fn with_n_samples(mut self, n: usize) -> Self
    pub fn with_max_background_samples(mut self, n: usize) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_regularization(mut self, reg: f64) -> Self
    pub fn with_paired_sampling(mut self, enabled: bool) -> Self
  }
  impl KernelExplainer<'a, M> {
    pub fn new(model: &'a M, background: &Array2<f64>, config: KernelSHAPConfig) -> Result<Self>
    pub fn base_value(&self) -> f64
    pub fn n_features(&self) -> usize
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self
    pub fn explain(&self, x: &[f64]) -> Result<SHAPResult>
    pub fn explain_batch(&self, x: &Array2<f64>) -> Result<SHAPBatchResult>
    pub fn explain_batch_parallel(&self, x: &Array2<f64>) -> Result<SHAPBatchResult>
    where
        M: Sync,
  }

ferroml-core/src/explainability/partial_dependence.rs
  pub struct PDPResult { grid_values: Array1<f64>, pdp_values: Array1<f64>, pdp_std: Array1<f64>, ice_curves: Option<Array2<f64>>, feature_idx: usize, feature_name: Option<String>, n_samples: usize, grid_method: GridMethod }
  pub struct PDP2DResult { grid_values_1: Array1<f64>, grid_values_2: Array1<f64>, pdp_values: Array2<f64>, feature_idx_1: usize, feature_idx_2: usize, feature_name_1: Option<String>, feature_name_2: Option<String>, n_samples: usize }
  pub enum GridMethod { Percentile, Uniform }
  impl PDPResult {
    pub fn min_effect(&self) -> f64
    pub fn max_effect(&self) -> f64
    pub fn effect_range(&self) -> f64
    pub fn is_monotonic_increasing(&self) -> bool
    pub fn is_monotonic_decreasing(&self) -> bool
    pub fn is_monotonic(&self) -> bool
    pub fn argmax(&self) -> usize
    pub fn argmin(&self) -> usize
    pub fn heterogeneity(&self) -> f64
    pub fn summary(&self) -> String
  }
  impl PDP2DResult {
    pub fn min_effect(&self) -> f64
    pub fn max_effect(&self) -> f64
    pub fn effect_range(&self) -> f64
    pub fn argmax(&self) -> (usize, usize)
    pub fn argmin(&self) -> (usize, usize)
    pub fn summary(&self) -> String
  }
  impl Default for GridMethod { fn default(...) }
  pub fn partial_dependence(model: &M, x: &Array2<f64>, feature_idx: usize, n_grid_points: usize, grid_me...) -> Result<PDPResult>
where
    M: Model,
  pub fn partial_dependence_parallel(model: &M, x: &Array2<f64>, feature_idx: usize, n_grid_points: usize, grid_me...) -> Result<PDPResult>
where
    M: Model + Sync,
  pub fn partial_dependence_2d(model: &M, x: &Array2<f64>, feature_idx_1: usize, feature_idx_2: usize, n_gri...) -> Result<PDP2DResult>
where
    M: Model,
  pub fn partial_dependence_2d_parallel(model: &M, x: &Array2<f64>, feature_idx_1: usize, feature_idx_2: usize, n_gri...) -> Result<PDP2DResult>
where
    M: Model + Sync,
  pub fn partial_dependence_multi(model: &M, x: &Array2<f64>, feature_indices: &[usize], n_grid_points: usize, ...) -> Result<Vec<PDPResult>>
where
    M: Model,
  pub fn partial_dependence_multi_parallel(model: &M, x: &Array2<f64>, feature_indices: &[usize], n_grid_points: usize, ...) -> Result<Vec<PDPResult>>
where
    M: Model + Sync,

ferroml-core/src/explainability/permutation.rs
  pub struct PermutationImportanceResult { importances_mean: Array1<f64>, importances_std: Array1<f64>, ci_lower: Array1<f64>, ci_upper: Array1<f64>, importances: Array2<f64>, confidence_level: f64, n_repeats: usize, baseline_score: f64, ... }
  impl PermutationImportanceResult {
    pub fn sorted_indices(&self) -> Vec<usize>
    pub fn top_k(&self, k: usize) -> Vec<usize>
    pub fn is_significant(&self, feature_idx: usize) -> bool
    pub fn significant_features(&self) -> Vec<usize>
    pub fn format_feature(&self, feature_idx: usize) -> String
    pub fn summary(&self) -> String
  }
  pub fn permutation_importance(model: &M, x: &Array2<f64>, y: &Array1<f64>, scoring_fn: F, n_repeats: usize,...) -> Result<PermutationImportanceResult>
where
    M: Model,
    F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64>,
  pub fn permutation_importance_with_options(model: &M, x: &Array2<f64>, y: &Array1<f64>, scoring_fn: F, n_repeats: usize,...) -> Result<PermutationImportanceResult>
where
    M: Model,
    F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64>,
  pub fn permutation_importance_parallel(model: &M, x: &Array2<f64>, y: &Array1<f64>, scoring_fn: F, n_repeats: usize,...) -> Result<PermutationImportanceResult>
where
    M: Model + Sync,
    F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64> + Send + Sync,
  pub fn permutation_importance_parallel_with_options(model: &M, x: &Array2<f64>, y: &Array1<f64>, scoring_fn: F, n_repeats: usize,...) -> Result<PermutationImportanceResult>
where
    M: Model + Sync,
    F: Fn(&Array1<f64>, &Array1<f64>) -> Result<f64> + Send + Sync,

ferroml-core/src/explainability/summary.rs
  pub struct FeatureSHAPStats { feature_idx: usize, feature_name: String, mean: f64, std: f64, mean_abs: f64, min: f64, max: f64, median: f64, ... }
  pub struct BarPlotEntry { feature_idx: usize, feature_name: String, importance: f64, importance_std: f64, ci_lower: f64, ci_upper: f64 }
  pub struct BarPlotData { entries: Vec<BarPlotEntry>, n_samples: usize }
  pub struct BeeswarmPoint { sample_idx: usize, shap_value: f64, normalized_feature_value: f64, raw_feature_value: f64 }
  pub struct BeeswarmFeatureData { feature_idx: usize, feature_name: String, points: Vec<BeeswarmPoint>, importance_rank: usize, mean_abs_shap: f64 }
  pub struct BeeswarmPlotData { features: Vec<BeeswarmFeatureData>, n_samples: usize, n_features: usize }
  pub struct SHAPSummary { base_value: f64, n_samples: usize, n_features: usize, feature_names: Vec<String>, feature_stats: Vec<FeatureSHAPStats>, importance_order: Vec<usize>, global_importance: Array1<f64>, shap_values: Array2<f64>, ... }
  pub struct DependencePlotData { feature_idx: usize, feature_name: String, feature_values: Vec<f64>, shap_values: Vec<f64>, interaction_values: Option<Vec<f64>>, interaction_feature_idx: Option<usize>, interaction_feature_name: Option<String>, correlation: f64 }
  impl FeatureSHAPStats {
    pub fn correlation_interpretation(&self) -> &'static str
    pub fn is_significant(&self, threshold: f64) -> bool
  }
  impl BarPlotData {
    pub fn top_k(&self, k: usize) -> Vec<&BarPlotEntry>
    pub fn above_threshold(&self, threshold: f64) -> Vec<&BarPlotEntry>
    pub fn cumulative_importance(&self) -> Vec<(usize, f64)>
    pub fn features_for_fraction(&self, fraction: f64) -> usize
  }
  impl BeeswarmPlotData {
    pub fn top_k(&self, k: usize) -> Vec<&BeeswarmFeatureData>
    pub fn points_by_feature(&self) -> Vec<&Vec<BeeswarmPoint>>
    pub fn feature_order(&self) -> Vec<usize>
  }
  impl SHAPSummary {
    pub fn from_batch_result(batch: &SHAPBatchResult) -> Self
    pub fn bar_plot_data(&self) -> BarPlotData
    pub fn beeswarm_plot_data(&self) -> BeeswarmPlotData
    pub fn get_feature_stats(&self, feature_idx: usize) -> Option<&FeatureSHAPStats>
    pub fn get_feature_stats_by_name(&self, name: &str) -> Option<&FeatureSHAPStats>
    pub fn top_k_features(&self, k: usize) -> Vec<&FeatureSHAPStats>
    pub fn positive_impact_features(&self, min_correlation: f64) -> Vec<&FeatureSHAPStats>
    pub fn negative_impact_features(&self, max_correlation: f64) -> Vec<&FeatureSHAPStats>
    pub fn total_shap_variance(&self) -> f64
    pub fn text_summary(&self, top_k: usize) -> String
  }
  impl SHAPSummary {
    pub fn dependence_plot_data(&self, feature_idx: usize) -> Option<DependencePlotData>
    pub fn dependence_plot_data_with_interaction(&self, feature_idx: usize, interaction_idx: usize,) -> Option<DependencePlotData>
    pub fn find_best_interaction(&self, feature_idx: usize) -> Option<usize>
  }

ferroml-core/src/explainability/treeshap.rs
  pub struct SHAPResult { base_value: f64, shap_values: Array1<f64>, feature_values: Array1<f64>, n_features: usize, feature_names: Option<Vec<String>> }
  pub struct SHAPBatchResult { base_value: f64, shap_values: Array2<f64>, feature_values: Array2<f64>, n_samples: usize, n_features: usize, feature_names: Option<Vec<String>> }
  pub struct TreeExplainer { ... }
  impl SHAPResult {
    pub fn sorted_indices(&self) -> Vec<usize>
    pub fn top_k(&self, k: usize) -> Vec<usize>
    pub fn prediction(&self) -> f64
    pub fn format_feature(&self, feature_idx: usize) -> String
    pub fn summary(&self) -> String
  }
  impl SHAPBatchResult {
    pub fn get_sample(&self, idx: usize) -> Option<SHAPResult>
    pub fn mean_abs_shap(&self) -> Array1<f64>
    pub fn global_importance_sorted(&self) -> Vec<usize>
    pub fn summary(&self) -> String
  }
  impl TreeExplainer {
    pub fn from_decision_tree_regressor(model: &DecisionTreeRegressor) -> Result<Self>
    pub fn from_decision_tree_classifier(model: &DecisionTreeClassifier) -> Result<Self>
    pub fn from_random_forest_regressor(model: &crate::models::forest::RandomForestRegressor,) -> Result<Self>
    pub fn from_random_forest_classifier(model: &crate::models::forest::RandomForestClassifier,) -> Result<Self>
    pub fn from_gradient_boosting_regressor(model: &crate::models::boosting::GradientBoostingRegressor,) -> Result<Self>
    pub fn base_value(&self) -> f64
    pub fn n_trees(&self) -> usize
    pub fn n_features(&self) -> usize
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self
    pub fn explain(&self, x: &[f64]) -> Result<SHAPResult>
    pub fn explain_batch(&self, x: &Array2<f64>) -> Result<SHAPBatchResult>
    pub fn explain_batch_parallel(&self, x: &Array2<f64>) -> Result<SHAPBatchResult>
  }

ferroml-core/src/gpu/backend.rs
  pub struct WgpuBackend { ... }
  impl WgpuBackend {
    pub fn try_new() -> Option<Self>
    pub fn new() -> Result<Self>
    pub fn memory_info(&self) -> &GpuMemoryInfo
  }
  impl GpuBackend for WgpuBackend { fn relu(...), fn sigmoid(...), fn softmax(...), fn row_sum(...), fn row_max(...), fn bias_add(...), fn relu_grad(...), fn sigmoid_grad(...), ... }

ferroml-core/src/gpu/dispatch.rs
  pub struct GpuDispatcher { ... }
  impl GpuDispatcher {
    pub fn new(backend: Arc<dyn GpuBackend>, policy: GpuDispatchPolicy) -> Self
    pub fn with_auto_policy(backend: Arc<dyn GpuBackend>) -> Self
    pub fn policy(&self) -> &GpuDispatchPolicy
    pub fn backend(&self) -> &Arc<dyn GpuBackend>
    pub fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>>
    pub fn pairwise_distances(&self, x: &Array2<f64>, centers: &Array2<f64>,) -> Result<Array2<f64>>
    pub fn relu(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn sigmoid(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn softmax(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn row_sum(&self, x: &Array2<f64>) -> Result<Array1<f64>>
    pub fn row_max(&self, x: &Array2<f64>) -> Result<Array1<f64>>
    pub fn bias_add(&self, x: &Array2<f64>, bias: &Array1<f64>) -> Result<Array2<f64>>
    pub fn relu_grad(&self, z: &Array2<f64>) -> Result<Array2<f64>>
    pub fn sigmoid_grad(&self, output: &Array2<f64>) -> Result<Array2<f64>>
    pub fn elementwise_mul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>>
  }

ferroml-core/src/gpu/mod.rs
  pub struct GpuMemoryInfo { max_buffer_size: u64, max_storage_buffer_binding_size: u32 }
  pub enum GpuDispatchPolicy { Always, Auto, Never }
  pub trait GpuBackend: Send + Sync + std::fmt::Debug {
    fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>>
    fn pairwise_distances(&self, x: &Array2<f64>, centers: &Array2<f64>) -> Result<Array2<f64>>
    fn relu(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    fn sigmoid(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    fn softmax(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    fn row_sum(&self, x: &Array2<f64>) -> Result<Array1<f64>>
    fn row_max(&self, x: &Array2<f64>) -> Result<Array1<f64>>
    fn bias_add(&self, x: &Array2<f64>, bias: &Array1<f64>) -> Result<Array2<f64>>
    fn relu_grad(&self, z: &Array2<f64>) -> Result<Array2<f64>>
    fn sigmoid_grad(&self, output: &Array2<f64>) -> Result<Array2<f64>>
    fn elementwise_mul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>>
    fn is_available(&self) -> bool
  }
  impl Default for GpuDispatchPolicy { fn default(...) }

ferroml-core/src/hpo/bayesian.rs
  pub struct GaussianProcessRegressor { ... }
  pub struct BayesianOptimizer { ... }
  pub struct LBFGSBConfig { max_iter: usize, m: usize, gtol: f64, ftol: f64, n_restarts: usize, c1: f64, c2: f64, max_linesearch: usize, ... }
  pub struct LBFGSB { ... }
  pub struct AcquisitionOptimizer { ... }
  pub enum Kernel { RBF, Matern52, Matern32 }
  pub enum AcquisitionFunction { EI, PI, UCB, LCB }
  impl Default for Kernel { fn default(...) }
  impl Kernel {
    pub fn rbf() -> Self
    pub fn matern52() -> Self
    pub fn matern32() -> Self
    pub fn length_scale(&self) -> f64
    pub fn variance(&self) -> f64
    pub fn with_length_scale(self, length_scale: f64) -> Self
    pub fn with_variance(self, variance: f64) -> Self
    pub fn compute(&self, x1: &[f64], x2: &[f64]) -> f64
    pub fn compute_matrix(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>>
    pub fn compute_cross(&self, x_train: &[Vec<f64>], x_new: &[Vec<f64>]) -> Vec<Vec<f64>>
  }
  impl Default for GaussianProcessRegressor { fn default(...) }
  impl GaussianProcessRegressor {
    pub fn new() -> Self
    pub fn with_kernel(mut self, kernel: Kernel) -> Self
    pub fn with_noise_variance(mut self, noise: f64) -> Self
    pub fn with_optimize_hyperparams(mut self, optimize: bool) -> Self
    pub fn with_n_restarts(mut self, n: usize) -> Self
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<()>
    pub fn predict(&self, x_new: &[Vec<f64>]) -> Result<Vec<(f64, f64)>>
    pub fn predict_mean(&self, x_new: &[Vec<f64>]) -> Result<Vec<f64>>
    pub fn log_marginal_likelihood(&self) -> Result<f64>
    pub fn kernel(&self) -> &Kernel
  }
  impl Default for BayesianOptimizer { fn default(...) }
  impl BayesianOptimizer {
    pub fn new() -> Self
    pub fn with_n_initial(mut self, n: usize) -> Self
    pub fn with_acquisition(mut self, acq: AcquisitionFunction) -> Self
    pub fn with_kappa(mut self, kappa: f64) -> Self
    pub fn with_seed(mut self, seed: u64) -> Self
    pub fn with_kernel(mut self, kernel: Kernel) -> Self
    pub fn with_n_candidates(mut self, n: usize) -> Self
    pub fn with_lbfgsb(mut self, use_lbfgsb: bool) -> Self
    pub fn with_lbfgsb_config(mut self, config: LBFGSBConfig) -> Self
    pub fn suggest(&self, search_space: &SearchSpace, trials: &[Trial],) -> Result<HashMap<String, ParameterValue>>
  }
  impl Sampler for BayesianOptimizer { fn sample(...) }
  impl Default for LBFGSBConfig { fn default(...) }
  impl Default for LBFGSB { fn default(...) }
  impl LBFGSB {
    pub fn new() -> Self
    pub fn with_config(mut self, config: LBFGSBConfig) -> Self
  }
  impl GaussianProcessRegressor {
    pub fn predict_with_gradients(&self, x_new: &[f64]) -> Result<(f64, f64, Vec<f64>, Vec<f64>)>
  }
  impl Default for AcquisitionOptimizer { fn default(...) }
  impl AcquisitionOptimizer {
    pub fn new() -> Self
    pub fn with_lbfgsb_config(mut self, config: LBFGSBConfig) -> Self
    pub fn with_gradient_optimization(mut self, use_gradient: bool) -> Self
    pub fn with_n_candidates(mut self, n: usize) -> Self
    pub fn optimize(&self, gp: &GaussianProcessRegressor, acquisition: AcquisitionFunct...)
  }
  pub fn expected_improvement(mu: f64, sigma: f64, best_y: f64, minimize: bool) -> f64
  pub fn probability_of_improvement(mu: f64, sigma: f64, best_y: f64, minimize: bool) -> f64
  pub fn upper_confidence_bound(mu: f64, sigma: f64, kappa: f64) -> f64
  pub fn lower_confidence_bound(mu: f64, sigma: f64, kappa: f64) -> f64
  pub fn expected_improvement_gradient(mu: f64, sigma: f64, dmu_dx: &[f64], dsigma_dx: &[f64], best_y: f64, minimize...) -> Vec<f64>
  pub fn probability_of_improvement_gradient(mu: f64, sigma: f64, dmu_dx: &[f64], dsigma_dx: &[f64], best_y: f64, minimize...) -> Vec<f64>
  pub fn ucb_gradient(dmu_dx: &[f64], dsigma_dx: &[f64], kappa: f64) -> Vec<f64>
  pub fn lcb_gradient(dmu_dx: &[f64], dsigma_dx: &[f64], kappa: f64) -> Vec<f64>

ferroml-core/src/hpo/mod.rs
  pub struct Trial { id: usize, params: HashMap<String, value: Option<f64>, state: TrialState, intermediate_values: Vec<f64>, duration: Option<f64> }
  pub struct Study { name: String, search_space: SearchSpace, direction: Direction, trials: Vec<Trial> }
  pub struct ParameterImportance { name: String, importance: f64, ci: (f64, method: String }
  pub enum TrialState { Running, Complete, Pruned, Failed }
  pub enum ParameterValue { Int, Float, Categorical, Bool }
  pub enum Direction { Minimize, Maximize }
  impl ParameterValue {
    pub fn as_f64(&self) -> Option<f64>
    pub fn as_i64(&self) -> Option<i64>
    pub fn as_str(&self) -> Option<&str>
    pub fn as_bool(&self) -> Option<bool>
  }
  impl Study {
    pub fn new(name: impl Into<String>, search_space: SearchSpace, direction: Dire...) -> Self
    pub fn with_seed(mut self, seed: u64) -> Self
    pub fn ask(&mut self) -> Result<Trial>
    pub fn tell(&mut self, trial_id: usize, value: f64) -> Result<()>
    pub fn best_trial(&self) -> Option<&Trial>
    pub fn best_trial_or_err(&self) -> crate::Result<&Trial>
    pub fn best_value(&self) -> Option<f64>
    pub fn best_params(&self) -> Option<&HashMap<String, ParameterValue>>
    pub fn n_trials(&self) -> usize
    pub fn report_intermediate(&mut self, trial_id: usize, step: usize, value: f64,) -> Result<bool>
  }
  pub fn parameter_importance(study: &Study) -> Vec<ParameterImportance>

ferroml-core/src/hpo/samplers.rs
  pub struct RandomSampler { ... }
  pub struct GridSampler { ... }
  pub struct TPESampler { ... }
  pub trait Sampler: Send + Sync {
    fn sample(&self, search_space: &SearchSpace, trials: &[Trial],) -> Result<HashMap<String, ParameterValue>>
  }
  impl Default for RandomSampler { fn default(...) }
  impl RandomSampler {
    pub fn new() -> Self
    pub fn with_seed(seed: u64) -> Self
  }
  impl Sampler for RandomSampler { fn sample(...) }
  impl Clone for GridSampler { fn clone(...) }
  impl Default for GridSampler { fn default(...) }
  impl GridSampler {
    pub fn new(grid_points: usize) -> Self
  }
  impl Sampler for GridSampler { fn sample(...) }
  impl Default for TPESampler { fn default(...) }
  impl TPESampler {
    pub fn new() -> Self
    pub fn with_startup_trials(mut self, n: usize) -> Self
    pub fn with_gamma(mut self, gamma: f64) -> Self
    pub fn with_seed(mut self, seed: u64) -> Self
    pub fn with_n_ei_candidates(mut self, n: usize) -> Self
  }
  impl Sampler for TPESampler { fn sample(...) }

ferroml-core/src/hpo/schedulers.rs
  pub struct RungMetrics { rung: usize, fidelity: f64, n_trials: usize, n_promoted: usize, best_value: Option<f64>, mean_value: Option<f64>, std_value: Option<f64>, trial_ids: Vec<usize> }
  pub struct BracketMetrics { bracket_id: usize, n_initial_configs: usize, min_fidelity: f64, max_fidelity: f64, n_rungs: usize, rung_metrics: Vec<RungMetrics>, is_complete: bool, best_trial_id: Option<usize>, ... }
  pub struct HyperbandMetrics { bracket_metrics: Vec<BracketMetrics>, total_trials: usize, total_pruned: usize, total_completed: usize, best_trial_id: Option<usize>, best_value: Option<f64>, total_cost: f64 }
  pub struct MedianPruner { ... }
  pub struct HyperbandConfig { min_resource: usize, max_resource: usize, reduction_factor: f64, fidelity: Option<FidelityParameter> }
  pub struct Bracket { id: usize, n_configs: usize, min_resource: usize, max_resource: usize, n_rungs: usize, rung_resources: Vec<usize>, rung_n_configs: Vec<usize>, trials_per_rung: Vec<Vec<usize>>, ... }
  pub struct HyperbandScheduler { ... }
  pub struct ASHAScheduler { ... }
  pub struct BOHBConfig { min_resource: usize, max_resource: usize, reduction_factor: f64, n_startup_trials: usize, gamma: f64, min_bandwidth: f64, max_bandwidth: f64, n_ei_candidates: usize, ... }
  pub struct BOHBSampler { ... }
  pub struct BOHBScheduler { ... }
  pub struct BOHBObservation { trial_id: usize, config: Vec<f64>, fidelity: f64, value: f64, bracket_id: usize }
  pub struct BOHBMetrics { n_model_samples: usize, n_random_samples: usize, best_config: Option<Vec<f64>>, best_value: Option<f64>, n_kde_updates: usize, n_good_configs: usize, n_bad_configs: usize }
  pub enum FidelityParameter { Discrete, Continuous }
  pub trait Scheduler: Send + Sync {
    fn should_prune(&self, trials: &[Trial], trial_id: usize, step: usize) -> bool
  }
  pub trait EarlyStoppingCallback: Send + Sync {
    fn on_trial_pruned(&self, trial_id: usize, step: usize, value: f64)
    fn on_rung_completed(&self, trial_id: usize, rung: usize, value: f64)
    fn on_bracket_completed(&self, bracket_id: usize, best_trial_id: usize, best_valu...)
  }
  impl FidelityParameter {
    pub fn discrete(name: impl Into<String>, min: usize, max: usize) -> Self
    pub fn continuous(name: impl Into<String>, min: f64, max: f64) -> Self
    pub fn name(&self) -> &str
    pub fn min_value(&self) -> f64
    pub fn max_value(&self) -> f64
    pub fn value_at_rung(&self, rung: usize, n_rungs: usize, reduction_factor: f64) -> f64
  }
  impl RungMetrics {
    pub fn add_trial(&mut self, trial_id: usize, value: f64)
  }
  impl HyperbandMetrics {
    pub fn pruning_rate(&self) -> f64
    pub fn efficiency(&self) -> Option<f64>
  }
  impl Default for MedianPruner { fn default(...) }
  impl MedianPruner {
    pub fn new() -> Self
    pub fn with_startup_trials(mut self, n: usize) -> Self
    pub fn with_warmup_steps(mut self, n: usize) -> Self
    pub fn with_percentile(mut self, percentile: f64) -> Self
  }
  impl Scheduler for MedianPruner { fn should_prune(...) }
  impl Default for HyperbandConfig { fn default(...) }
  impl Bracket {
    pub fn new(id: usize, n_configs: usize, min_resource: usize, max_resource: usi...) -> Self
    pub fn resource_at_rung(&self, rung: usize) -> Option<usize>
    pub fn register_trial(&mut self, trial_id: usize, rung: usize, value: f64)
    pub fn should_promote(&self, trial_id: usize, rung: usize) -> bool
    pub fn promote_to_next_rung(&mut self) -> Vec<usize>
    pub fn best_trial(&self) -> Option<(usize, f64)>
  }
  impl Default for HyperbandScheduler { fn default(...) }
  impl HyperbandScheduler {
    pub fn new(min_resource: usize, max_resource: usize, reduction_factor: f64) -> Self
    pub fn from_config(config: HyperbandConfig) -> Self
    pub fn with_fidelity(mut self, fidelity: FidelityParameter) -> Self
    pub fn config(&self) -> &HyperbandConfig
    pub fn brackets(&self) -> &[Bracket]
    pub fn brackets_mut(&mut self) -> &mut [Bracket]
    pub fn metrics(&self) -> &HyperbandMetrics
    pub fn get_resource_for_trial(&self, trial_id: usize) -> usize
    pub fn assign_trial(&mut self, trial_id: usize) -> usize
    pub fn report_value(&mut self, trial_id: usize, step: usize, value: f64)
    pub fn n_rungs(&self) -> usize
    pub fn rung_resources(&self, bracket_id: usize) -> Option<&[usize]>
    pub fn fidelity_at_step(&self, step: usize) -> f64
    pub fn is_complete(&self) -> bool
    pub fn summary(&self) -> String
  }
  impl Scheduler for HyperbandScheduler { fn should_prune(...) }
  impl Default for ASHAScheduler { fn default(...) }
  impl ASHAScheduler {
    pub fn new(min_resource: usize, reduction_factor: f64, grace_period: usize) -> Self
  }
  impl Scheduler for ASHAScheduler { fn should_prune(...) }
  impl Default for BOHBConfig { fn default(...) }
  impl BOHBConfig {
    pub fn new(min_resource: usize, max_resource: usize, reduction_factor: f64) -> Self
  }
  impl Default for BOHBSampler { fn default(...) }
  impl BOHBSampler {
    pub fn new(config: BOHBConfig) -> Self
    pub fn update(&mut self, trials: &[Trial])
    pub fn sample(&mut self, search_space: &super::SearchSpace, trials: &[Trial],) -> crate::Result<HashMap<String, super::ParameterValue>>
  }
  impl Clone for BOHBScheduler { fn clone(...) }
  impl BOHBScheduler {
    pub fn new(config: BOHBConfig) -> Self
    pub fn with_defaults(min_resource: usize, max_resource: usize, reduction_factor: f64) -> Self
    pub fn with_seed(mut self, seed: u64) -> Self
    pub fn with_startup_trials(mut self, n: usize) -> Self
    pub fn with_gamma(mut self, gamma: f64) -> Self
    pub fn config(&self) -> &BOHBConfig
    pub fn hyperband(&self) -> &HyperbandScheduler
    pub fn hyperband_mut(&mut self) -> &mut HyperbandScheduler
    pub fn bohb_metrics(&self) -> &BOHBMetrics
    pub fn observations(&self) -> &[BOHBObservation]
    pub fn sample_config(&mut self, search_space: &super::SearchSpace, trials: &[Trial],) -> crate::Result<HashMap<String, super::ParameterValue>>
    pub fn register_observation(&mut self, trial_id: usize, config: Vec<f64>, fidelity: f64, value:...)
    pub fn assign_trial(&mut self, trial_id: usize) -> usize
    pub fn get_resource_for_trial(&self, trial_id: usize) -> usize
    pub fn is_complete(&self) -> bool
    pub fn summary(&self) -> String
  }
  impl Scheduler for BOHBScheduler { fn should_prune(...) }

ferroml-core/src/hpo/search_space.rs
  pub struct SearchSpace { parameters: HashMap<String }
  pub struct Parameter { param_type: ParameterType, log_scale: bool, default: Option<ParameterDefault> }
  pub enum ParameterType { Int, Float, Categorical, Bool }
  pub enum ParameterDefault { Int, Float, Categorical, Bool }
  impl SearchSpace {
    pub fn new() -> Self
    pub fn add(mut self, name: impl Into<String>, param: Parameter) -> Self
    pub fn int(self, name: impl Into<String>, low: i64, high: i64) -> Self
    pub fn int_log(self, name: impl Into<String>, low: i64, high: i64) -> Self
    pub fn float(self, name: impl Into<String>, low: f64, high: f64) -> Self
    pub fn float_log(self, name: impl Into<String>, low: f64, high: f64) -> Self
    pub fn categorical(self, name: impl Into<String>, choices: Vec<String>) -> Self
    pub fn bool(self, name: impl Into<String>) -> Self
    pub fn n_dims(&self) -> usize
  }
  impl Parameter {
    pub fn int(low: i64, high: i64) -> Self
    pub fn int_log(low: i64, high: i64) -> Self
    pub fn float(low: f64, high: f64) -> Self
    pub fn float_log(low: f64, high: f64) -> Self
    pub fn categorical(choices: Vec<String>) -> Self
    pub fn bool() -> Self
    pub fn with_default(mut self, default: ParameterDefault) -> Self
  }

ferroml-core/src/inference/mod.rs
  pub enum Value { Tensor, TensorI64, SequenceMapI64F32 }
  pub enum InferenceError { InvalidModel, UnsupportedOperator, ShapeMismatch, MissingInput, MissingAttribute, TypeMismatch, RuntimeError }
  impl Value {
    pub fn as_tensor(&self) -> Option<&Tensor>
    pub fn as_tensor_i64(&self) -> Option<&TensorI64>
    pub fn as_sequence_map(&self) -> Option<&Vec<HashMap<i64, f32>>>
  }
  impl From<InferenceError> for FerroError { fn from(...) }

ferroml-core/src/inference/operators.rs
  pub struct SqueezeOp { axes: Vec<i64> }
  pub struct SoftmaxOp { axis: i64 }
  pub struct FlattenOp { axis: i64 }
  pub struct TreeEnsembleRegressorOp { n_targets: usize, n_trees: usize }
  pub struct TreeEnsembleClassifierOp { n_classes: usize, class_labels: Vec<i64>, n_trees: usize }
  pub enum NodeMode { BranchLeq, BranchLt, BranchGte, BranchGt, BranchEq, BranchNeq, Leaf }
  pub enum AggregateFunction { Sum, Average, Min, Max }
  pub enum PostTransform { None, Softmax, Logistic, SoftmaxZero, Probit }
  pub trait Operator: Send + Sync {
    fn execute(&self, inputs: &[&Value]) -> Result<Vec<Value>, InferenceError>
    fn name(&self) -> &str
  }
  impl Operator for MatMulOp { fn execute(...), fn name(...) }
  impl Operator for AddOp { fn execute(...), fn name(...) }
  impl Operator for SqueezeOp { fn execute(...), fn name(...) }
  impl Operator for SigmoidOp { fn execute(...), fn name(...) }
  impl Operator for SoftmaxOp { fn execute(...), fn name(...) }
  impl Operator for FlattenOp { fn execute(...), fn name(...) }
  impl Operator for ReshapeOp { fn execute(...), fn name(...) }
  impl TreeEnsembleRegressorOp {
    pub fn from_attributes(n_targets: i64, nodes_featureids: &[i64], nodes_values: &[f32], nod...) -> Result<Self, InferenceError>
  }
  impl Operator for TreeEnsembleRegressorOp { fn execute(...), fn name(...) }
  impl TreeEnsembleClassifierOp {
    pub fn from_attributes(class_labels: &[i64], nodes_featureids: &[i64], nodes_values: &[f32...) -> Result<Self, InferenceError>
  }
  impl Operator for TreeEnsembleClassifierOp { fn execute(...), fn name(...) }

ferroml-core/src/inference/session.rs
  pub struct InferenceSession { ... }
  pub struct SessionMetadata { graph_name: String, doc_string: String, n_inputs: usize, n_outputs: usize, n_nodes: usize }
  impl InferenceSession {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self>
    pub fn input_names(&self) -> &[String]
    pub fn output_names(&self) -> &[String]
    pub fn run(&self, inputs: &[(&str, Tensor)
    pub fn run_single(&self, inputs: &[(&str, Tensor)
    pub fn metadata(&self) -> SessionMetadata
  }

ferroml-core/src/inference/tensor.rs
  pub struct Tensor { ... }
  pub struct TensorI64 { ... }
  impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self
    pub fn zeros(shape: Vec<usize>) -> Self
    pub fn full(shape: Vec<usize>, value: f32) -> Self
    pub fn from_slice(data: &[f32]) -> Self
    pub fn shape(&self) -> &[usize]
    pub fn ndim(&self) -> usize
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool
    pub fn as_slice(&self) -> &[f32]
    pub fn as_mut_slice(&mut self) -> &mut [f32]
    pub fn as_f32_slice(&self) -> &[f32]
    pub fn into_vec(self) -> Vec<f32>
    pub fn get(&self, indices: &[usize]) -> Option<f32>
    pub fn set(&mut self, indices: &[usize], value: f32) -> Option<()>
    pub fn reshape(self, new_shape: Vec<usize>) -> Result<Self, InferenceError>
    pub fn squeeze(self, axes: &[i64]) -> Result<Self, InferenceError>
    pub fn flatten(self, axis: i64) -> Result<Self, InferenceError>
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, InferenceError>
    pub fn add(&self, other: &Tensor) -> Result<Tensor, InferenceError>
    pub fn sigmoid(&self) -> Tensor
    pub fn softmax(&self, axis: i64) -> Result<Tensor, InferenceError>
  }
  impl Index<usize> for Tensor { fn index(...) }
  impl IndexMut<usize> for Tensor { fn index_mut(...) }
  impl TensorI64 {
    pub fn from_vec(data: Vec<i64>, shape: Vec<usize>) -> Self
    pub fn zeros(shape: Vec<usize>) -> Self
    pub fn shape(&self) -> &[usize]
    pub fn as_slice(&self) -> &[i64]
    pub fn as_i64_slice(&self) -> &[i64]
    pub fn into_vec(self) -> Vec<i64>
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool
  }

ferroml-core/src/metrics/classification.rs
  pub struct ConfusionMatrix { matrix: Array2<usize>, labels: Vec<i64> }
  pub struct ClassificationReport { precision: Vec<f64>, recall: Vec<f64>, f1: Vec<f64>, support: Vec<usize>, labels: Vec<i64>, accuracy: f64, macro_precision: f64, macro_recall: f64, ... }
  pub struct PrecisionMetric { average: Average }
  pub struct RecallMetric { average: Average }
  pub struct F1Metric { average: Average }
  impl ConfusionMatrix {
    pub fn compute(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Self>
    pub fn true_positives(&self) -> Vec<usize>
    pub fn false_positives(&self) -> Vec<usize>
    pub fn false_negatives(&self) -> Vec<usize>
    pub fn true_negatives(&self) -> Vec<usize>
    pub fn support(&self) -> Vec<usize>
    pub fn total(&self) -> usize
  }
  impl ClassificationReport {
    pub fn from_confusion_matrix(cm: &ConfusionMatrix) -> Self
    pub fn compute(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Self>
    pub fn summary(&self) -> String
  }
  impl Metric for AccuracyMetric { fn name(...), fn direction(...), fn compute(...) }
  impl Default for PrecisionMetric { fn default(...) }
  impl Metric for PrecisionMetric { fn name(...), fn direction(...), fn compute(...) }
  impl Default for RecallMetric { fn default(...) }
  impl Metric for RecallMetric { fn name(...), fn direction(...), fn compute(...) }
  impl Default for F1Metric { fn default(...) }
  impl Metric for F1Metric { fn name(...), fn direction(...), fn compute(...) }
  impl Metric for BalancedAccuracyMetric { fn name(...), fn direction(...), fn compute(...) }
  impl Metric for MatthewsCorrCoefMetric { fn name(...), fn direction(...), fn compute(...) }
  impl Metric for CohenKappaMetric { fn name(...), fn direction(...), fn compute(...) }
  pub fn accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64>
  pub fn precision(y_true: &Array1<f64>, y_pred: &Array1<f64>, average: Average) -> Result<f64>
  pub fn recall(y_true: &Array1<f64>, y_pred: &Array1<f64>, average: Average) -> Result<f64>
  pub fn f1_score(y_true: &Array1<f64>, y_pred: &Array1<f64>, average: Average) -> Result<f64>
  pub fn confusion_matrix(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<ConfusionMatrix>
  pub fn balanced_accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64>
  pub fn matthews_corrcoef(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64>
  pub fn cohen_kappa_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64>

ferroml-core/src/metrics/comparison.rs
  pub struct ModelComparisonResult { test_name: String, statistic: f64, p_value: f64, df: Option<f64>, mean_difference: f64, std_error: f64, ci_95: (f64, significant: bool, ... }
  impl ModelComparisonResult {
    pub fn summary(&self) -> String
  }
  pub fn paired_ttest(scores1: &Array1<f64>, scores2: &Array1<f64>) -> Result<ModelComparisonResult>
  pub fn corrected_resampled_ttest(scores1: &Array1<f64>, scores2: &Array1<f64>, n_train: usize, n_test: usize,) -> Result<ModelComparisonResult>
  pub fn mcnemar_test(y_true: &Array1<f64>, pred1: &Array1<f64>, pred2: &Array1<f64>,) -> Result<ModelComparisonResult>
  pub fn wilcoxon_signed_rank_test(scores1: &Array1<f64>, scores2: &Array1<f64>,) -> Result<ModelComparisonResult>
  pub fn five_by_two_cv_paired_ttest(differences: &[[f64; 2]; 5]) -> Result<ModelComparisonResult>
  pub fn five_by_two_cv_paired_ttest_from_scores(scores1: &[[f64; 2]; 5], scores2: &[[f64; 2]; 5],) -> Result<ModelComparisonResult>

ferroml-core/src/metrics/mod.rs
  pub struct MetricValue { name: String, value: f64, direction: Direction }
  pub struct MetricValueWithCI { value: f64, ci_lower: f64, ci_upper: f64, confidence_level: f64, std_error: f64, n_bootstrap: usize }
  pub struct MetricsBundle { metrics: Vec<MetricValue>, n_samples: usize }
  pub enum Average { Macro, Micro, Weighted, None }
  pub enum Direction { Maximize, Minimize }
  pub trait Metric: Send + Sync {
    fn name(&self) -> &str
    fn direction(&self) -> Direction
    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue>
    fn compute_with_ci(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>, confid...) -> Result<MetricValueWithCI>
    where
        Self: Sized,
    fn requires_probabilities(&self) -> bool
  }
  pub trait ProbabilisticMetric: Metric {
    fn compute_proba(&self, y_true: &Array1<f64>, y_proba: &Array2<f64>) -> Result<MetricValue>
    fn compute_proba_with_ci(&self, y_true: &Array1<f64>, y_proba: &Array2<f64>, confi...) -> Result<MetricValueWithCI>
  }
  impl MetricValue {
    pub fn new(name: impl Into<String>, value: f64, direction: Direction) -> Self
    pub fn is_better_than(&self, other: &MetricValue) -> bool
  }
  impl MetricValueWithCI {
    pub fn summary(&self) -> String
    pub fn significantly_different_from(&self, other: &MetricValueWithCI) -> bool
  }
  impl MetricsBundle {
    pub fn get(&self, name: &str) -> Option<&MetricValue>
    pub fn best(&self) -> Option<&MetricValue>
  }

ferroml-core/src/metrics/probabilistic.rs
  pub struct LogLossMetric { eps: f64 }
  pub struct RocCurve { fpr: Vec<f64>, tpr: Vec<f64>, thresholds: Vec<f64>, auc: f64 }
  pub struct PrCurve { precision: Vec<f64>, recall: Vec<f64>, thresholds: Vec<f64>, auc: f64, average_precision: f64 }
  impl Metric for RocAucMetric { fn name(...), fn direction(...), fn compute(...), fn requires_probabilities(...) }
  impl ProbabilisticMetric for RocAucMetric { fn compute_proba(...), fn compute_proba_with_ci(...) }
  impl Metric for PrAucMetric { fn name(...), fn direction(...), fn compute(...), fn requires_probabilities(...) }
  impl Metric for AveragePrecisionMetric { fn name(...), fn direction(...), fn compute(...), fn requires_probabilities(...) }
  impl Default for LogLossMetric { fn default(...) }
  impl Metric for LogLossMetric { fn name(...), fn direction(...), fn compute(...), fn requires_probabilities(...) }
  impl Metric for BrierScoreMetric { fn name(...), fn direction(...), fn compute(...), fn requires_probabilities(...) }
  impl RocCurve {
    pub fn compute(y_true: &Array1<f64>, y_score: &Array1<f64>) -> Result<Self>
  }
  impl PrCurve {
    pub fn compute(y_true: &Array1<f64>, y_score: &Array1<f64>) -> Result<Self>
  }
  pub fn roc_auc_score(y_true: &Array1<f64>, y_score: &Array1<f64>) -> Result<f64>
  pub fn roc_auc_with_ci(y_true: &Array1<f64>, y_score: &Array1<f64>, confidence: f64, n_bootstrap: us...) -> Result<MetricValueWithCI>
  pub fn pr_auc_score(y_true: &Array1<f64>, y_score: &Array1<f64>) -> Result<f64>
  pub fn average_precision_score(y_true: &Array1<f64>, y_score: &Array1<f64>) -> Result<f64>
  pub fn log_loss(y_true: &Array1<f64>, y_pred: &Array1<f64>, eps: Option<f64>) -> Result<f64>
  pub fn brier_score(y_true: &Array1<f64>, y_prob: &Array1<f64>) -> Result<f64>
  pub fn brier_skill_score(y_true: &Array1<f64>, y_prob: &Array1<f64>) -> Result<f64>

ferroml-core/src/metrics/regression.rs
  pub struct RegressionMetrics { mse: f64, rmse: f64, mae: f64, r2: f64, explained_variance: f64, max_error: f64, median_absolute_error: f64, n_samples: usize }
  impl RegressionMetrics {
    pub fn compute(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Self>
    pub fn summary(&self) -> String
  }
  impl Metric for MseMetric { fn name(...), fn direction(...), fn compute(...) }
  impl Metric for RmseMetric { fn name(...), fn direction(...), fn compute(...) }
  impl Metric for MaeMetric { fn name(...), fn direction(...), fn compute(...) }
  impl Metric for R2Metric { fn name(...), fn direction(...), fn compute(...) }
  impl Metric for ExplainedVarianceMetric { fn name(...), fn direction(...), fn compute(...) }
  pub fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64>
  pub fn rmse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64>
  pub fn mae(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64>
  pub fn r2_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64>
  pub fn explained_variance(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64>
  pub fn max_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64>
  pub fn median_absolute_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64>
  pub fn mape(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64>

ferroml-core/src/models/adaboost.rs
  pub struct AdaBoostClassifier { n_estimators: usize, learning_rate: f64, max_depth: usize, random_state: Option<u64>, warm_start: bool }
  pub struct AdaBoostRegressor { n_estimators: usize, learning_rate: f64, loss: AdaBoostLoss, max_depth: usize, random_state: Option<u64>, warm_start: bool }
  pub enum AdaBoostLoss { Linear, Square, Exponential }
  impl AdaBoostClassifier {
    pub fn new(n_estimators: usize) -> Self
    pub fn with_warm_start(mut self, warm_start: bool) -> Self
    pub fn with_learning_rate(mut self, lr: f64) -> Self
    pub fn with_max_depth(mut self, max_depth: usize) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn estimator_weights(&self) -> Option<&Array1<f64>>
    pub fn n_estimators_fitted(&self) -> usize
    pub fn estimators(&self) -> Option<&[DecisionTreeClassifier]>
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for AdaBoostClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn feature_importance(...), fn model_name(...) }
  impl Default for AdaBoostLoss { fn default(...) }
  impl AdaBoostRegressor {
    pub fn new(n_estimators: usize) -> Self
    pub fn with_warm_start(mut self, warm_start: bool) -> Self
    pub fn with_learning_rate(mut self, lr: f64) -> Self
    pub fn with_loss(mut self, loss: AdaBoostLoss) -> Self
    pub fn with_max_depth(mut self, max_depth: usize) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn n_estimators_fitted(&self) -> usize
    pub fn estimators(&self) -> Option<&[DecisionTreeRegressor]>
    pub fn estimator_weights(&self) -> Option<&Array1<f64>>
  }
  impl Model for AdaBoostRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn feature_importance(...), fn model_name(...), fn score(...) }

ferroml-core/src/models/boosting.rs
  pub struct EarlyStopping { patience: usize, min_delta: f64, validation_fraction: f64 }
  pub struct TrainingHistory { train_loss: Vec<f64>, val_loss: Vec<f64>, learning_rates: Vec<f64>, stopped_at: Option<usize> }
  pub struct GradientBoostingRegressor { n_estimators: usize, loss: RegressionLoss, learning_rate_schedule: LearningRateSchedule, max_depth: Option<usize>, min_samples_split: usize, min_samples_leaf: usize, max_features: Option<usize>, subsample: f64, ... }
  pub struct GradientBoostingClassifier { n_estimators: usize, loss: ClassificationLoss, learning_rate_schedule: LearningRateSchedule, max_depth: Option<usize>, min_samples_split: usize, min_samples_leaf: usize, max_features: Option<usize>, subsample: f64, ... }
  pub enum RegressionLoss { SquaredError, AbsoluteError, Huber }
  pub enum ClassificationLoss { Deviance, Exponential }
  pub enum LearningRateSchedule { Constant, LinearDecay, ExponentialDecay }
  impl Default for RegressionLoss { fn default(...) }
  impl Default for ClassificationLoss { fn default(...) }
  impl Default for LearningRateSchedule { fn default(...) }
  impl LearningRateSchedule {
    pub fn get_lr(&self, iteration: usize, n_estimators: usize) -> f64
  }
  impl Default for EarlyStopping { fn default(...) }
  impl Default for GradientBoostingRegressor { fn default(...) }
  impl GradientBoostingRegressor {
    pub fn new() -> Self
    pub fn with_gpu(mut self, backend: std::sync::Arc<dyn crate::gpu::GpuBackend>) -> Self
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self
    pub fn with_loss(mut self, loss: RegressionLoss) -> Self
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self
    pub fn with_learning_rate_schedule(mut self, schedule: LearningRateSchedule) -> Self
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self
    pub fn with_subsample(mut self, subsample: f64) -> Self
    pub fn with_early_stopping(mut self, early_stopping: EarlyStopping) -> Self
    pub fn with_alpha(mut self, alpha: f64) -> Self
    pub fn with_random_state(mut self, random_state: u64) -> Self
    pub fn training_history(&self) -> Option<&TrainingHistory>
    pub fn estimators(&self) -> Option<&[DecisionTreeRegressor]>
    pub fn n_estimators_actual(&self) -> Option<usize>
    pub fn init_prediction(&self) -> Option<f64>
    pub fn learning_rate_at(&self, iteration: usize) -> f64
  }
  impl Iterator for StagedPredictIterator<'a> { fn next(...) }
  impl Model for GradientBoostingRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn n_features(...), fn score(...) }
  impl Default for GradientBoostingClassifier { fn default(...) }
  impl GradientBoostingClassifier {
    pub fn new() -> Self
    pub fn with_gpu(mut self, backend: std::sync::Arc<dyn crate::gpu::GpuBackend>) -> Self
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self
    pub fn with_loss(mut self, loss: ClassificationLoss) -> Self
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self
    pub fn with_learning_rate_schedule(mut self, schedule: LearningRateSchedule) -> Self
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self
    pub fn with_subsample(mut self, subsample: f64) -> Self
    pub fn with_early_stopping(mut self, early_stopping: EarlyStopping) -> Self
    pub fn with_random_state(mut self, random_state: u64) -> Self
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self
    pub fn training_history(&self) -> Option<&TrainingHistory>
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn n_estimators_actual(&self) -> Option<usize>
    pub fn estimators(&self) -> Option<&[Vec<DecisionTreeRegressor>]>
    pub fn init_predictions(&self) -> Option<&Array1<f64>>
    pub fn n_classes(&self) -> Option<usize>
    pub fn learning_rate_at(&self, iteration: usize) -> f64
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for GradientBoostingClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn n_features(...) }

ferroml-core/src/models/calibration.rs
  pub struct SigmoidCalibrator { ... }
  pub struct IsotonicCalibrator { ... }
  pub struct TemperatureScalingCalibrator { ... }
  pub struct CalibratedClassifierCV { ... }
  pub struct CalibrationResult { bin_edges: Vec<f64>, mean_predicted_proba: Vec<f64>, fraction_of_positives: Vec<f64>, bin_counts: Vec<usize>, ece: f64, mce: f64, brier_score: f64 }
  pub enum CalibrationMethod { Sigmoid, Isotonic, Temperature }
  pub trait Calibrator: Send + Sync {
    fn fit(&mut self, y_prob: &Array1<f64>, y_true: &Array1<f64>) -> Result<()>
    fn transform(&self, y_prob: &Array1<f64>) -> Result<Array1<f64>>
    fn is_fitted(&self) -> bool
    fn clone_boxed(&self) -> Box<dyn Calibrator>
    fn method(&self) -> CalibrationMethod
  }
  pub trait MulticlassCalibrator: Send + Sync {
    fn fit(&mut self, y_prob: &Array2<f64>, y_true: &Array1<f64>) -> Result<()>
    fn transform(&self, y_prob: &Array2<f64>) -> Result<Array2<f64>>
    fn is_fitted(&self) -> bool
    fn clone_boxed(&self) -> Box<dyn MulticlassCalibrator>
    fn method(&self) -> CalibrationMethod
  }
  pub trait CalibrableClassifier: Model {
    fn predict_proba_for_calibration(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    fn clone_boxed(&self) -> Box<dyn CalibrableClassifier>
  }
  impl Default for CalibrationMethod { fn default(...) }
  impl Default for SigmoidCalibrator { fn default(...) }
  impl SigmoidCalibrator {
    pub fn new() -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn parameters(&self) -> Option<(f64, f64)>
  }
  impl Calibrator for SigmoidCalibrator { fn fit(...), fn transform(...), fn is_fitted(...), fn clone_boxed(...), fn method(...) }
  impl Default for IsotonicCalibrator { fn default(...) }
  impl IsotonicCalibrator {
    pub fn new() -> Self
    pub fn with_increasing(mut self, increasing: bool) -> Self
    pub fn with_clip(mut self, clip: bool) -> Self
    pub fn fitted_values(&self) -> Option<(&[f64], &[f64])>
  }
  impl Calibrator for IsotonicCalibrator { fn fit(...), fn transform(...), fn is_fitted(...), fn clone_boxed(...), fn method(...) }
  impl Default for TemperatureScalingCalibrator { fn default(...) }
  impl TemperatureScalingCalibrator {
    pub fn new() -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn with_learning_rate(mut self, lr: f64) -> Self
    pub fn temperature(&self) -> Option<f64>
  }
  impl MulticlassCalibrator for TemperatureScalingCalibrator { fn fit(...), fn transform(...), fn is_fitted(...), fn clone_boxed(...), fn method(...) }
  impl Calibrator for TemperatureScalingCalibrator { fn fit(...), fn transform(...), fn is_fitted(...), fn clone_boxed(...), fn method(...) }
  impl CalibratedClassifierCV {
    pub fn new(base_estimator: Box<dyn CalibrableClassifier>) -> Self
    pub fn with_method(mut self, method: CalibrationMethod) -> Self
    pub fn with_n_folds(mut self, n_folds: usize) -> Self
    pub fn with_cv(mut self, cv: Box<dyn CrossValidator>) -> Self
    pub fn with_stratified(mut self, stratified: bool) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn method(&self) -> CalibrationMethod
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn n_calibrators(&self) -> usize
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for CalibratedClassifierCV { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn search_space(...) }
  impl CalibrableClassifier for LogisticRegression { fn predict_proba_for_calibration(...), fn clone_boxed(...) }
  impl CalibrableClassifier for GaussianNB { fn predict_proba_for_calibration(...), fn clone_boxed(...) }
  impl CalibrableClassifier for MultinomialNB { fn predict_proba_for_calibration(...), fn clone_boxed(...) }
  impl CalibrableClassifier for BernoulliNB { fn predict_proba_for_calibration(...), fn clone_boxed(...) }
  impl CalibrableClassifier for DecisionTreeClassifier { fn predict_proba_for_calibration(...), fn clone_boxed(...) }
  impl CalibrableClassifier for RandomForestClassifier { fn predict_proba_for_calibration(...), fn clone_boxed(...) }
  impl CalibrableClassifier for GradientBoostingClassifier { fn predict_proba_for_calibration(...), fn clone_boxed(...) }
  impl CalibrableClassifier for HistGradientBoostingClassifier { fn predict_proba_for_calibration(...), fn clone_boxed(...) }
  impl CalibrableClassifier for KNeighborsClassifier { fn predict_proba_for_calibration(...), fn clone_boxed(...) }
  impl CalibrableClassifier for SVC { fn predict_proba_for_calibration(...), fn clone_boxed(...) }
  pub fn calibration_curve(y_true: &Array1<f64>, y_prob: &Array1<f64>, n_bins: usize,) -> Result<CalibrationResult>

ferroml-core/src/models/extra_trees.rs
  pub struct ExtraTreesClassifier { n_estimators: usize, criterion: SplitCriterion, max_depth: Option<usize>, min_samples_split: usize, min_samples_leaf: usize, max_features: Option<MaxFeatures>, bootstrap: bool, n_jobs: Option<usize>, ... }
  pub struct ExtraTreesRegressor { n_estimators: usize, criterion: SplitCriterion, max_depth: Option<usize>, min_samples_split: usize, min_samples_leaf: usize, max_features: Option<MaxFeatures>, bootstrap: bool, n_jobs: Option<usize>, ... }
  impl Default for ExtraTreesClassifier { fn default(...) }
  impl ExtraTreesClassifier {
    pub fn new() -> Self
    pub fn with_n_estimators(mut self, n: usize) -> Self
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self
    pub fn with_min_samples_split(mut self, n: usize) -> Self
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_min_impurity_decrease(mut self, v: f64) -> Self
    pub fn with_class_weight(mut self, cw: ClassWeight) -> Self
    pub fn feature_importances(&self) -> Option<&Array1<f64>>
    pub fn feature_importances_with_ci(&self) -> Option<&FeatureImportanceWithCI>
    pub fn estimators(&self) -> Option<&[DecisionTreeClassifier]>
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for ExtraTreesClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn feature_importance(...), fn search_space(...) }
  impl Default for ExtraTreesRegressor { fn default(...) }
  impl ExtraTreesRegressor {
    pub fn new() -> Self
    pub fn with_n_estimators(mut self, n: usize) -> Self
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self
    pub fn with_min_samples_split(mut self, n: usize) -> Self
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_min_impurity_decrease(mut self, v: f64) -> Self
    pub fn feature_importances(&self) -> Option<&Array1<f64>>
    pub fn feature_importances_with_ci(&self) -> Option<&FeatureImportanceWithCI>
    pub fn estimators(&self) -> Option<&[DecisionTreeRegressor]>
  }
  impl Model for ExtraTreesRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn feature_importance(...), fn search_space(...), fn score(...) }

ferroml-core/src/models/forest.rs
  pub struct FeatureImportanceWithCI { importance: Array1<f64>, std_error: Array1<f64>, ci_lower: Array1<f64>, ci_upper: Array1<f64>, confidence_level: f64 }
  pub struct RandomForestClassifier { n_estimators: usize, criterion: SplitCriterion, max_depth: Option<usize>, min_samples_split: usize, min_samples_leaf: usize, max_features: Option<MaxFeatures>, bootstrap: bool, oob_score_enabled: bool, ... }
  pub struct RandomForestRegressor { n_estimators: usize, criterion: SplitCriterion, max_depth: Option<usize>, min_samples_split: usize, min_samples_leaf: usize, max_features: Option<MaxFeatures>, bootstrap: bool, oob_score_enabled: bool, ... }
  pub enum MaxFeatures { All, Sqrt, Log2, Fixed, Fraction }
  impl Default for MaxFeatures { fn default(...) }
  impl MaxFeatures {
    pub fn compute(&self, n_features: usize) -> usize
  }
  impl Default for RandomForestClassifier { fn default(...) }
  impl RandomForestClassifier {
    pub fn new() -> Self
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self
    pub fn with_max_features(mut self, max_features: Option<MaxFeatures>) -> Self
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self
    pub fn with_oob_score(mut self, oob_score: bool) -> Self
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self
    pub fn with_random_state(mut self, random_state: u64) -> Self
    pub fn with_confidence_level(mut self, level: f64) -> Self
    pub fn with_min_impurity_decrease(mut self, min_impurity_decrease: f64) -> Self
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self
    pub fn oob_score(&self) -> Option<f64>
    pub fn oob_decision_function(&self) -> Option<&Array2<f64>>
    pub fn estimators(&self) -> Option<&[DecisionTreeClassifier]>
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn feature_importances_with_ci(&self) -> Option<&FeatureImportanceWithCI>
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for RandomForestClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn n_features(...) }
  impl Default for RandomForestRegressor { fn default(...) }
  impl RandomForestRegressor {
    pub fn new() -> Self
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self
    pub fn with_max_features(mut self, max_features: Option<MaxFeatures>) -> Self
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self
    pub fn with_oob_score(mut self, oob_score: bool) -> Self
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self
    pub fn with_random_state(mut self, random_state: u64) -> Self
    pub fn with_confidence_level(mut self, level: f64) -> Self
    pub fn with_min_impurity_decrease(mut self, min_impurity_decrease: f64) -> Self
    pub fn oob_score(&self) -> Option<f64>
    pub fn oob_prediction(&self) -> Option<&Array1<f64>>
    pub fn estimators(&self) -> Option<&[DecisionTreeRegressor]>
    pub fn feature_importances_with_ci(&self) -> Option<&FeatureImportanceWithCI>
  }
  impl Model for RandomForestRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn n_features(...), fn score(...) }

ferroml-core/src/models/gaussian_process.rs
  pub struct RBF { length_scale: f64 }
  pub struct Matern { length_scale: f64, nu: f64 }
  pub struct ConstantKernel { constant: f64 }
  pub struct WhiteKernel { noise_level: f64 }
  pub struct SumKernel { k1: Box<dyn Kernel>, k2: Box<dyn Kernel> }
  pub struct ProductKernel { k1: Box<dyn Kernel>, k2: Box<dyn Kernel> }
  pub struct GaussianProcessRegressor { ... }
  pub struct GaussianProcessClassifier { ... }
  pub struct SparseGPRegressor { ... }
  pub struct SparseGPClassifier { ... }
  pub struct SVGPRegressor { ... }
  pub enum InducingPointMethod { RandomSubset, KMeans, GreedyVariance }
  pub enum SparseApproximation { FITC, VFE }
  pub trait Kernel: Send + Sync + std::fmt::Debug {
    fn compute(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64>
    fn diagonal(&self, x: &Array2<f64>) -> Array1<f64>
    fn clone_box(&self) -> Box<dyn Kernel>
  }
  impl Clone for Box<dyn Kernel> { fn clone(...) }
  impl RBF {
    pub fn new(length_scale: f64) -> Self
  }
  impl Kernel for RBF { fn compute(...), fn diagonal(...), fn clone_box(...) }
  impl Matern {
    pub fn new(length_scale: f64, nu: f64) -> Self
  }
  impl Kernel for Matern { fn compute(...), fn diagonal(...), fn clone_box(...) }
  impl ConstantKernel {
    pub fn new(constant: f64) -> Self
  }
  impl Kernel for ConstantKernel { fn compute(...), fn diagonal(...), fn clone_box(...) }
  impl WhiteKernel {
    pub fn new(noise_level: f64) -> Self
  }
  impl Kernel for WhiteKernel { fn compute(...), fn diagonal(...), fn clone_box(...) }
  impl SumKernel {
    pub fn new(k1: Box<dyn Kernel>, k2: Box<dyn Kernel>) -> Self
  }
  impl Kernel for SumKernel { fn compute(...), fn diagonal(...), fn clone_box(...) }
  impl ProductKernel {
    pub fn new(k1: Box<dyn Kernel>, k2: Box<dyn Kernel>) -> Self
  }
  impl Kernel for ProductKernel { fn compute(...), fn diagonal(...), fn clone_box(...) }
  impl GaussianProcessRegressor {
    pub fn new(kernel: Box<dyn Kernel>) -> Self
    pub fn with_alpha(mut self, alpha: f64) -> Self
    pub fn with_normalize_y(mut self, normalize_y: bool) -> Self
    pub fn predict_with_std(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)>
    pub fn log_marginal_likelihood(&self) -> Result<f64>
  }
  impl Model for GaussianProcessRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn search_space(...), fn score(...) }
  impl GaussianProcessClassifier {
    pub fn new(kernel: Box<dyn Kernel>) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for GaussianProcessClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn search_space(...) }
  impl SparseGPRegressor {
    pub fn new(kernel: Box<dyn Kernel>) -> Self
    pub fn with_alpha(mut self, alpha: f64) -> Self
    pub fn with_normalize_y(mut self, normalize_y: bool) -> Self
    pub fn with_n_inducing(mut self, n_inducing: usize) -> Self
    pub fn with_inducing_method(mut self, method: InducingPointMethod) -> Self
    pub fn with_approximation(mut self, approx: SparseApproximation) -> Self
    pub fn predict_with_std(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)>
    pub fn log_marginal_likelihood(&self) -> Result<f64>
    pub fn inducing_points(&self) -> Option<&Array2<f64>>
  }
  impl Model for SparseGPRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn search_space(...), fn score(...) }
  impl SparseGPClassifier {
    pub fn new(kernel: Box<dyn Kernel>) -> Self
    pub fn with_n_inducing(mut self, n_inducing: usize) -> Self
    pub fn with_inducing_method(mut self, method: InducingPointMethod) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn inducing_points(&self) -> Option<&Array2<f64>>
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for SparseGPClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn search_space(...) }
  impl SVGPRegressor {
    pub fn new(kernel: Box<dyn Kernel>) -> Self
    pub fn with_noise_variance(mut self, noise_variance: f64) -> Self
    pub fn with_n_inducing(mut self, n_inducing: usize) -> Self
    pub fn with_inducing_method(mut self, method: InducingPointMethod) -> Self
    pub fn with_n_epochs(mut self, n_epochs: usize) -> Self
    pub fn with_batch_size(mut self, batch_size: usize) -> Self
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self
    pub fn with_normalize_y(mut self, normalize_y: bool) -> Self
    pub fn inducing_points(&self) -> Option<&Array2<f64>>
    pub fn predict_with_std(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)>
  }
  impl Model for SVGPRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn search_space(...), fn score(...) }
  pub fn select_inducing_points(x: &Array2<f64>, m: usize, method: &InducingPointMethod, kernel: &dyn Kernel,) -> Result<Array2<f64>>

ferroml-core/src/models/hist_boosting.rs
  pub struct HistEarlyStopping { patience: usize, tol: f64, validation_fraction: f64 }
  pub struct BinMapper { ... }
  pub struct HistTreeNode { feature_idx: Option<usize>, bin_threshold: Option<u8>, missing_go_left: bool, left_child: Option<usize>, right_child: Option<usize>, value: f64, n_samples: usize, sum_gradients: f64, ... }
  pub struct HistTree { nodes: Vec<HistTreeNode> }
  pub struct HistGradientBoostingClassifier { max_iter: usize, learning_rate: f64, max_leaf_nodes: Option<usize>, max_depth: Option<usize>, min_samples_leaf: usize, max_bins: usize, l1_regularization: f64, l2_regularization: f64, ... }
  pub struct HistGradientBoostingRegressor { max_iter: usize, learning_rate: f64, max_leaf_nodes: Option<usize>, max_depth: Option<usize>, min_samples_leaf: usize, max_bins: usize, l1_regularization: f64, l2_regularization: f64, ... }
  pub struct CategoricalFeatureHandler { ... }
  pub struct CategoricalBinMapper { ... }
  pub enum MonotonicConstraint { None, Positive, Negative }
  pub enum GrowthStrategy { DepthFirst, LeafWise }
  pub enum HistLoss { LogLoss, Hinge }
  pub enum HistRegressionLoss { SquaredError, AbsoluteError, Huber }
  pub enum CategoricalEncoding { OrderedTargetEncoding, TargetEncoding, OneHot }
  pub trait BinMapperInfo {
    fn n_bins(&self, feature_idx: usize) -> usize
  }
  impl Default for MonotonicConstraint { fn default(...) }
  impl Default for GrowthStrategy { fn default(...) }
  impl Default for HistLoss { fn default(...) }
  impl Default for HistRegressionLoss { fn default(...) }
  impl HistRegressionLoss {
    pub fn gradient(&self, y_true: f64, y_pred: f64, delta: f64) -> f64
    pub fn hessian(&self, y_true: f64, y_pred: f64, delta: f64) -> f64
    pub fn loss(&self, y_true: f64, y_pred: f64, delta: f64) -> f64
    pub fn initial_prediction(&self, y: &Array1<f64>) -> f64
  }
  impl Default for HistEarlyStopping { fn default(...) }
  impl BinMapper {
    pub fn new(max_bins: usize) -> Self
    pub fn fit(&mut self, x: &Array2<f64>)
    pub fn transform(&self, x: &Array2<f64>) -> Array2<u8>
    pub fn n_bins(&self, feature_idx: usize) -> usize
    pub fn bin_threshold_to_real(&self, feature_idx: usize, bin: u8) -> f64
  }
  impl BinMapperInfo for BinMapper { fn n_bins(...) }
  impl PartialEq for SplitCandidate { fn eq(...) }
  impl PartialOrd for SplitCandidate { fn partial_cmp(...) }
  impl Ord for SplitCandidate { fn cmp(...) }
  impl Default for HistGradientBoostingClassifier { fn default(...) }
  impl HistGradientBoostingClassifier {
    pub fn new() -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self
    pub fn with_max_leaf_nodes(mut self, max_leaf_nodes: Option<usize>) -> Self
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self
    pub fn with_max_bins(mut self, max_bins: usize) -> Self
    pub fn with_l1_regularization(mut self, l1: f64) -> Self
    pub fn with_l2_regularization(mut self, l2: f64) -> Self
    pub fn with_min_gain_to_split(mut self, min_gain: f64) -> Self
    pub fn with_loss(mut self, loss: HistLoss) -> Self
    pub fn with_early_stopping(mut self, config: HistEarlyStopping) -> Self
    pub fn with_monotonic_constraints(mut self, constraints: Vec<MonotonicConstraint>) -> Self
    pub fn with_interaction_constraints(mut self, constraints: Vec<Vec<usize>>) -> Self
    pub fn with_growth_strategy(mut self, strategy: GrowthStrategy) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_categorical_features(mut self, features: Vec<usize>) -> Self
    pub fn with_categorical_smoothing(mut self, smoothing: f64) -> Self
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn train_loss_history(&self) -> Option<&Vec<f64>>
    pub fn val_loss_history(&self) -> Option<&Vec<f64>>
    pub fn n_iter_actual(&self) -> Option<usize>
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for HistGradientBoostingClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn n_features(...) }
  impl Default for HistGradientBoostingRegressor { fn default(...) }
  impl HistGradientBoostingRegressor {
    pub fn new() -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self
    pub fn with_max_leaf_nodes(mut self, max_leaf_nodes: Option<usize>) -> Self
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self
    pub fn with_max_bins(mut self, max_bins: usize) -> Self
    pub fn with_l1_regularization(mut self, l1: f64) -> Self
    pub fn with_l2_regularization(mut self, l2: f64) -> Self
    pub fn with_min_gain_to_split(mut self, min_gain: f64) -> Self
    pub fn with_loss(mut self, loss: HistRegressionLoss) -> Self
    pub fn with_huber_delta(mut self, delta: f64) -> Self
    pub fn with_early_stopping(mut self, config: HistEarlyStopping) -> Self
    pub fn with_monotonic_constraints(mut self, constraints: Vec<MonotonicConstraint>) -> Self
    pub fn with_interaction_constraints(mut self, constraints: Vec<Vec<usize>>) -> Self
    pub fn with_growth_strategy(mut self, strategy: GrowthStrategy) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_categorical_features(mut self, features: Vec<usize>) -> Self
    pub fn with_categorical_smoothing(mut self, smoothing: f64) -> Self
    pub fn train_loss_history(&self) -> Option<&Vec<f64>>
    pub fn val_loss_history(&self) -> Option<&Vec<f64>>
    pub fn n_iter_actual(&self) -> Option<usize>
  }
  impl Model for HistGradientBoostingRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn n_features(...), fn score(...) }
  impl Default for CategoricalEncoding { fn default(...) }
  impl CategoricalFeatureHandler {
    pub fn new(categorical_features: Vec<usize>) -> Self
    pub fn empty() -> Self
    pub fn with_smoothing(mut self, smoothing: f64) -> Self
    pub fn with_encoding(mut self, encoding: CategoricalEncoding) -> Self
    pub fn is_categorical(&self, feature_idx: usize) -> bool
    pub fn categorical_indices(&self) -> &HashSet<usize>
    pub fn n_categories(&self, feature_idx: usize) -> Option<usize>
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, n_features: usize)
    pub fn transform(&self, x: &Array2<f64>, y: Option<&Array1<f64>>, permutation: Optio...) -> Array2<f64>
  }
  impl CategoricalBinMapper {
    pub fn new(max_bins: usize, categorical_features: Vec<usize>) -> Self
    pub fn with_smoothing(mut self, smoothing: f64) -> Self
    pub fn with_encoding(mut self, encoding: CategoricalEncoding) -> Self
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, rng: &mut StdRng)
    pub fn transform(&self, x: &Array2<f64>) -> Array2<u8>
    pub fn n_bins(&self, feature_idx: usize) -> usize
    pub fn is_categorical(&self, feature_idx: usize) -> bool
    pub fn categorical_handler(&self) -> &CategoricalFeatureHandler
    pub fn bin_threshold_to_real(&self, feature_idx: usize, bin: u8) -> f64
  }
  impl BinMapperInfo for CategoricalBinMapper { fn n_bins(...) }

ferroml-core/src/models/isolation_forest.rs
  pub struct IsolationForest { ... }
  pub enum MaxSamples { Auto, Count, Fraction }
  pub enum Contamination { Auto, Proportion }
  impl Default for MaxSamples { fn default(...) }
  impl Default for Contamination { fn default(...) }
  impl IsolationForest {
    pub fn new(n_estimators: usize) -> Self
    pub fn with_max_samples(mut self, max_samples: MaxSamples) -> Self
    pub fn with_contamination(mut self, contamination: Contamination) -> Self
    pub fn with_max_features(mut self, max_features: f64) -> Self
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn offset_value(&self) -> Option<f64>
    pub fn max_samples_value(&self) -> Option<usize>
  }
  impl OutlierDetector for IsolationForest { fn fit_unsupervised(...), fn predict_outliers(...), fn score_samples(...), fn decision_function(...), fn is_fitted(...), fn offset(...) }

ferroml-core/src/models/isotonic.rs
  pub struct IsotonicRegression { ... }
  pub enum Increasing { True, False, Auto }
  pub enum OutOfBounds { Nan, Clip, Raise }
  impl Default for Increasing { fn default(...) }
  impl Default for OutOfBounds { fn default(...) }
  impl Default for IsotonicRegression { fn default(...) }
  impl IsotonicRegression {
    pub fn new() -> Self
    pub fn with_increasing(mut self, increasing: Increasing) -> Self
    pub fn with_y_min(mut self, y_min: f64) -> Self
    pub fn with_y_max(mut self, y_max: f64) -> Self
    pub fn with_out_of_bounds(mut self, oob: OutOfBounds) -> Self
    pub fn x_thresholds(&self) -> Option<&[f64]>
    pub fn y_thresholds(&self) -> Option<&[f64]>
    pub fn increasing_inferred(&self) -> Option<bool>
  }

ferroml-core/src/models/knn.rs
  pub struct KDTree { ... }
  pub struct BallTree { ... }
  pub struct KNeighborsClassifier { n_neighbors: usize, weights: KNNWeights, metric: DistanceMetric, algorithm: KNNAlgorithm, leaf_size: usize, class_weight: ClassWeight }
  pub struct KNeighborsRegressor { n_neighbors: usize, weights: KNNWeights, metric: DistanceMetric, algorithm: KNNAlgorithm, leaf_size: usize }
  pub struct NearestCentroid { metric: DistanceMetric, shrink_threshold: Option<f64> }
  pub enum DistanceMetric { Euclidean, Manhattan, Minkowski }
  pub enum KNNWeights { Uniform, Distance }
  pub enum KNNAlgorithm { KDTree, BallTree, BruteForce, Auto }
  impl Default for DistanceMetric { fn default(...) }
  impl DistanceMetric {
    pub fn compute(&self, a: &[f64], b: &[f64]) -> f64
    pub fn compute_squared(&self, a: &[f64], b: &[f64]) -> f64
  }
  impl Default for KNNWeights { fn default(...) }
  impl Default for KNNAlgorithm { fn default(...) }
  impl KDTree {
    pub fn build(data: Array2<f64>, leaf_size: usize) -> Self
    pub fn query(&self, query: &[f64], k: usize, metric: &DistanceMetric) -> Vec<(usize, f64)>
  }
  impl PartialEq for NeighborCandidate { fn eq(...) }
  impl PartialOrd for NeighborCandidate { fn partial_cmp(...) }
  impl Ord for NeighborCandidate { fn cmp(...) }
  impl BallTree {
    pub fn build(data: Array2<f64>, leaf_size: usize) -> Self
    pub fn query(&self, query: &[f64], k: usize, metric: &DistanceMetric) -> Vec<(usize, f64)>
  }
  impl Default for KNeighborsClassifier { fn default(...) }
  impl KNeighborsClassifier {
    pub fn new(n_neighbors: usize) -> Self
    pub fn with_weights(mut self, weights: KNNWeights) -> Self
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self
    pub fn with_algorithm(mut self, algorithm: KNNAlgorithm) -> Self
    pub fn with_leaf_size(mut self, leaf_size: usize) -> Self
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn effective_algorithm(&self) -> Option<KNNAlgorithm>
  }
  impl Model for KNeighborsClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn search_space(...) }
  impl ProbabilisticModel for KNeighborsClassifier { fn predict_proba(...), fn predict_interval(...) }
  impl Default for KNeighborsRegressor { fn default(...) }
  impl KNeighborsRegressor {
    pub fn new(n_neighbors: usize) -> Self
    pub fn with_weights(mut self, weights: KNNWeights) -> Self
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self
    pub fn with_algorithm(mut self, algorithm: KNNAlgorithm) -> Self
    pub fn with_leaf_size(mut self, leaf_size: usize) -> Self
    pub fn effective_algorithm(&self) -> Option<KNNAlgorithm>
  }
  impl Model for KNeighborsRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn search_space(...), fn score(...) }
  impl NearestCentroid {
    pub fn new() -> Self
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self
    pub fn with_shrink_threshold(mut self, threshold: f64) -> Self
    pub fn centroids(&self) -> Option<&Array2<f64>>
    pub fn classes(&self) -> Option<&Array1<f64>>
  }
  impl Default for NearestCentroid { fn default(...) }
  impl Model for NearestCentroid { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn model_name(...) }

ferroml-core/src/models/linear.rs
  pub struct LinearRegression { fit_intercept: bool, copy_x: bool, confidence_level: f64, feature_names: Option<Vec<String>> }
  impl Default for LinearRegression { fn default(...) }
  impl LinearRegression {
    pub fn new() -> Self
    pub fn without_intercept() -> Self
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self
    pub fn with_confidence_level(mut self, level: f64) -> Self
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self
    pub fn coefficients(&self) -> Option<&Array1<f64>>
    pub fn intercept(&self) -> Option<f64>
    pub fn all_coefficients(&self) -> Option<Array1<f64>>
    pub fn vif(&self, x: &Array2<f64>) -> Array1<f64>
    pub fn cooks_distance(&self) -> Option<Array1<f64>>
    pub fn standardized_residuals(&self) -> Option<Array1<f64>>
    pub fn studentized_residuals(&self) -> Option<Array1<f64>>
    pub fn dffits(&self) -> Option<Array1<f64>>
    pub fn r_squared(&self) -> Option<f64>
    pub fn adjusted_r_squared(&self) -> Option<f64>
    pub fn f_statistic(&self) -> Option<(f64, f64)>
    pub fn log_likelihood(&self) -> Option<f64>
    pub fn aic(&self) -> Option<f64>
    pub fn bic(&self) -> Option<f64>
    pub fn condition_number(&self) -> Option<f64>
  }
  impl Model for LinearRegression { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn feature_names(...), fn n_features(...), fn score(...) }
  impl StatisticalModel for LinearRegression { fn summary(...), fn diagnostics(...), fn coefficients_with_ci(...), fn residuals(...), fn fitted_values(...), fn assumption_test(...) }
  impl ProbabilisticModel for LinearRegression { fn predict_proba(...), fn predict_interval(...) }
  impl PipelineModel for LinearRegression { fn clone_boxed(...), fn set_param(...), fn name(...) }

ferroml-core/src/models/lof.rs
  pub struct LocalOutlierFactor { ... }
  impl LocalOutlierFactor {
    pub fn new(n_neighbors: usize) -> Self
    pub fn with_contamination(mut self, contamination: Contamination) -> Self
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self
    pub fn with_algorithm(mut self, algorithm: KNNAlgorithm) -> Self
    pub fn with_novelty(mut self, novelty: bool) -> Self
    pub fn negative_outlier_factor(&self) -> Option<&Array1<f64>>
    pub fn offset_value(&self) -> Option<f64>
  }
  impl OutlierDetector for LocalOutlierFactor { fn fit_unsupervised(...), fn predict_outliers(...), fn fit_predict_outliers(...), fn score_samples(...), fn decision_function(...), fn is_fitted(...), fn offset(...) }

ferroml-core/src/models/logistic.rs
  pub struct LogisticRegression { fit_intercept: bool, max_iter: usize, tol: f64, l2_penalty: f64, confidence_level: f64, feature_names: Option<Vec<String>>, n_bootstrap: usize, class_weight: ClassWeight, ... }
  pub struct OddsRatioInfo { name: String, odds_ratio: f64, ci_lower: f64, ci_upper: f64, confidence_level: f64 }
  pub enum LogisticSolver { Irls, Lbfgs, Auto }
  impl Default for LogisticSolver { fn default(...) }
  impl LogisticSolver {
    pub fn from_str_lossy(s: &str) -> Self
  }
  impl Default for LogisticRegression { fn default(...) }
  impl LogisticRegression {
    pub fn new() -> Self
    pub fn without_intercept() -> Self
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn with_l2_penalty(mut self, penalty: f64) -> Self
    pub fn with_confidence_level(mut self, level: f64) -> Self
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self
    pub fn with_warm_start(mut self, warm_start: bool) -> Self
    pub fn with_solver(mut self, solver: LogisticSolver) -> Self
    pub fn coefficients(&self) -> Option<&Array1<f64>>
    pub fn intercept(&self) -> Option<f64>
    pub fn all_coefficients(&self) -> Option<Array1<f64>>
    pub fn odds_ratios(&self) -> Option<Array1<f64>>
    pub fn odds_ratios_with_ci(&self, level: f64) -> Vec<OddsRatioInfo>
    pub fn pseudo_r_squared(&self) -> Option<f64>
    pub fn log_likelihood(&self) -> Option<f64>
    pub fn aic(&self) -> Option<f64>
    pub fn bic(&self) -> Option<f64>
    pub fn deviance(&self) -> Option<f64>
    pub fn likelihood_ratio_test(&self) -> Option<(f64, f64)>
    pub fn train_roc_auc(&self) -> Option<f64>
    pub fn train_roc_auc_with_ci(&self) -> Option<crate::metrics::MetricValueWithCI>
    pub fn pearson_residuals(&self) -> Option<Array1<f64>>
    pub fn deviance_residuals(&self) -> Option<Array1<f64>>
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>>
  }
  impl CostFunction for LogisticCost { fn cost(...) }
  impl Gradient for LogisticCost { fn gradient(...) }
  impl Model for LogisticRegression { fn fit(...), fn fit_weighted(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn feature_names(...), fn n_features(...), ... }
  impl StatisticalModel for LogisticRegression { fn summary(...), fn diagnostics(...), fn coefficients_with_ci(...), fn residuals(...), fn fitted_values(...), fn assumption_test(...) }
  impl ProbabilisticModel for LogisticRegression { fn predict_proba(...), fn predict_interval(...) }
  impl OddsRatioInfo {
    pub fn is_significant(&self) -> bool
  }
  impl PipelineModel for LogisticRegression { fn clone_boxed(...), fn set_param(...), fn name(...) }

ferroml-core/src/models/mod.rs
  pub struct ModelSummary { model_type: String, n_observations: usize, n_features: usize, dependent_var: Option<String>, coefficients: Vec<CoefficientInfo>, fit_statistics: FitStatistics, assumption_tests: Vec<AssumptionTestResult>, notes: Vec<String> }
  pub struct CoefficientInfo { name: String, estimate: f64, std_error: f64, t_statistic: f64, p_value: f64, ci_lower: f64, ci_upper: f64, confidence_level: f64, ... }
  pub struct FitStatistics { r_squared: Option<f64>, adj_r_squared: Option<f64>, f_statistic: Option<f64>, f_p_value: Option<f64>, log_likelihood: Option<f64>, aic: Option<f64>, bic: Option<f64>, rmse: Option<f64>, ... }
  pub struct Diagnostics { residual_stats: ResidualStatistics, assumption_tests: Vec<AssumptionTestResult>, influential_observations: Vec<InfluentialObservation>, condition_number: Option<f64>, durbin_watson: Option<f64> }
  pub struct ResidualStatistics { min: f64, q1: f64, median: f64, q3: f64, max: f64, mean: f64, std_dev: f64, skewness: f64, ... }
  pub struct AssumptionTestResult { assumption: Assumption, test_name: String, statistic: f64, p_value: f64, passed: bool, alpha: f64, details: Option<String> }
  pub struct InfluentialObservation { index: usize, cooks_distance: f64, leverage: f64, std_residual: f64, studentized_residual: Option<f64>, dffits: Option<f64> }
  pub struct PredictionInterval { predictions: Array1<f64>, lower: Array1<f64>, upper: Array1<f64>, confidence_level: f64, std_errors: Option<Array1<f64>> }
  pub enum Assumption { NormalResiduals, Homoscedasticity, NoAutocorrelation, Linearity, NoMulticollinearity, Independence }
  pub trait Model: Send + Sync {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>
    fn is_fitted(&self) -> bool
    fn feature_importance(&self) -> Option<Array1<f64>>
    fn search_space(&self) -> SearchSpace
    fn feature_names(&self) -> Option<&[String]>
    fn n_features(&self) -> Option<usize>
    fn fit_weighted(&mut self, _x: &Array2<f64>, _y: &Array1<f64>, _sample_we...) -> Result<()>
    fn try_predict_proba(&self, _x: &Array2<f64>) -> Option<Result<Array2<f64>>>
    fn model_name(&self) -> &str
    fn as_any(&self) -> &dyn std::any::Any
    where
        Self: Sized + 'static,
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any
    where
        Self: Sized + 'static,
    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64>
  }
  pub trait StatisticalModel: Model {
    fn summary(&self) -> ModelSummary
    fn diagnostics(&self) -> Diagnostics
    fn coefficients_with_ci(&self, level: f64) -> Vec<CoefficientInfo>
    fn residuals(&self) -> Option<Array1<f64>>
    fn fitted_values(&self) -> Option<Array1<f64>>
    fn assumption_test(&self, assumption: Assumption) -> Option<AssumptionTestResult>
  }
  pub trait ProbabilisticModel: Model {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    fn predict_interval(&self, x: &Array2<f64>, level: f64) -> Result<PredictionInterval>
    fn predict_log_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl ModelSummary {
    pub fn new(model_type: impl Into<String>, n_observations: usize, n_features: u...) -> Self
    pub fn with_dependent_var(mut self, name: impl Into<String>) -> Self
    pub fn add_coefficient(&mut self, coef: CoefficientInfo)
    pub fn with_fit_statistics(mut self, stats: FitStatistics) -> Self
    pub fn add_assumption_test(&mut self, test: AssumptionTestResult)
    pub fn add_note(&mut self, note: impl Into<String>)
    pub fn has_significant_coefficients(&self, alpha: f64) -> bool
    pub fn count_significant(&self, alpha: f64) -> usize
  }
  impl CoefficientInfo {
    pub fn new(name: impl Into<String>, estimate: f64, std_error: f64) -> Self
    pub fn with_test(mut self, t_statistic: f64, p_value: f64) -> Self
    pub fn with_ci(mut self, lower: f64, upper: f64, level: f64) -> Self
    pub fn with_vif(mut self, vif: f64) -> Self
    pub fn is_significant(&self, alpha: f64) -> bool
    pub fn ci_excludes_zero(&self) -> bool
    pub fn has_multicollinearity(&self, threshold: f64) -> bool
  }
  impl FitStatistics {
    pub fn with_r_squared(r2: f64, adj_r2: f64) -> Self
    pub fn with_f_test(mut self, f_stat: f64, p_value: f64) -> Self
    pub fn with_information_criteria(mut self, ll: f64, aic: f64, bic: f64) -> Self
    pub fn with_errors(mut self, rmse: f64, mae: f64) -> Self
    pub fn with_df(mut self, df_model: usize, df_residuals: usize) -> Self
    pub fn is_model_significant(&self, alpha: f64) -> bool
  }
  impl Diagnostics {
    pub fn new(residual_stats: ResidualStatistics) -> Self
    pub fn add_assumption_test(&mut self, test: AssumptionTestResult)
    pub fn add_influential(&mut self, obs: InfluentialObservation)
    pub fn normality_ok(&self) -> bool
    pub fn homoscedasticity_ok(&self) -> bool
    pub fn no_autocorrelation_ok(&self) -> bool
    pub fn multicollinearity_ok(&self, threshold: f64) -> bool
    pub fn failed_assumptions(&self) -> Vec<&AssumptionTestResult>
    pub fn all_assumptions_ok(&self) -> bool
  }
  impl ResidualStatistics {
    pub fn from_residuals(residuals: &Array1<f64>) -> Self
  }
  impl Default for ResidualStatistics { fn default(...) }
  impl AssumptionTestResult {
    pub fn new(assumption: Assumption, test_name: impl Into<String>, statistic: f6...) -> Self
    pub fn with_details(mut self, details: impl Into<String>) -> Self
  }
  impl InfluentialObservation {
    pub fn new(index: usize, cooks_distance: f64, leverage: f64, std_residual: f64) -> Self
    pub fn is_high_cooks(&self, n: usize) -> bool
    pub fn is_high_leverage(&self, n_features: usize, n_samples: usize) -> bool
  }
  impl PredictionInterval {
    pub fn new(predictions: Array1<f64>, lower: Array1<f64>, upper: Array1<f64>, c...) -> Self
    pub fn with_std_errors(mut self, std_errors: Array1<f64>) -> Self
    pub fn interval_widths(&self) -> Array1<f64>
    pub fn coverage(&self, y_actual: &Array1<f64>) -> f64
  }
  pub fn get_feature_name(feature_names: &Option<Vec<String>>, idx: usize) -> String
  pub fn sigmoid(x: f64) -> f64
  pub fn raw_to_proba(raw: &Array2<f64>, n_classes: usize) -> Array2<f64>
  pub fn compute_log_loss(y: &Array1<f64>, probas: &Array2<f64>, classes: &Array1<f64>) -> f64
  pub fn sorted_median(sorted: &[f64]) -> f64
  pub fn check_is_fitted(fitted_data: &Option<T>, operation: &str) -> Result<()>
  pub fn validate_fit_input(x: &Array2<f64>, y: &Array1<f64>) -> Result<()>
  pub fn validate_fit_input_allow_nan(x: &Array2<f64>, y: &Array1<f64>) -> Result<()>
  pub fn validate_predict_input_allow_nan(x: &Array2<f64>, expected_features: usize) -> Result<()>
  pub fn validate_predict_input(x: &Array2<f64>, expected_features: usize) -> Result<()>
  pub fn compute_sample_weights(y: &Array1<f64>, classes: &Array1<f64>, class_weight: &ClassWeight,) -> Array1<f64>
  pub fn compute_class_weight_map(y: &Array1<f64>, classes: &Array1<f64>, class_weight: &ClassWeight,) -> Vec<(f64, f64)>
  pub fn get_unique_classes(y: &Array1<f64>) -> Array1<f64>

ferroml-core/src/models/multioutput.rs
  pub struct MultiOutputRegressor { ... }
  pub struct MultiOutputClassifier { ... }
  impl MultiOutputRegressor<M> {
    pub fn new(base_estimator: M) -> Self
    pub fn fit_multi(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()>
    pub fn predict_multi(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn is_fitted(&self) -> bool
    pub fn n_outputs(&self) -> Option<usize>
    pub fn estimators(&self) -> Option<&[M]>
  }
  impl MultiOutputClassifier<M> {
    pub fn new(base_estimator: M) -> Self
    pub fn fit_multi(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<()>
    pub fn predict_multi(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn predict_proba_multi(&self, x: &Array2<f64>) -> Result<Vec<Array2<f64>>>
    pub fn is_fitted(&self) -> bool
    pub fn n_outputs(&self) -> Option<usize>
    pub fn estimators(&self) -> Option<&[M]>
  }

ferroml-core/src/models/qda.rs
  pub struct QuadraticDiscriminantAnalysis { ... }
  impl Default for QuadraticDiscriminantAnalysis { fn default(...) }
  impl QuadraticDiscriminantAnalysis {
    pub fn new() -> Self
    pub fn with_reg_param(mut self, reg_param: f64) -> Self
    pub fn with_priors(mut self, priors: Vec<f64>) -> Self
    pub fn with_store_covariance(mut self, store: bool) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn is_fitted(&self) -> bool
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn means(&self) -> Option<&Array2<f64>>
    pub fn covariances(&self) -> Option<&Vec<Array2<f64>>>
    pub fn priors_fitted(&self) -> Option<&Array1<f64>>
    pub fn classes(&self) -> Option<&Vec<f64>>
  }

ferroml-core/src/models/quantile.rs
  pub struct QuantileRegression { quantile: f64, fit_intercept: bool, max_iter: usize, tol: f64, n_bootstrap: usize, confidence_level: f64, feature_names: Option<Vec<String>>, random_state: Option<u64> }
  pub struct MultiQuantileResults { quantiles: Vec<f64>, coefficients: Array2<f64>, std_errors: Array2<f64>, fit_intercept: bool, feature_names: Option<Vec<String>> }
  impl Default for QuantileRegression { fn default(...) }
  impl QuantileRegression {
    pub fn new(quantile: f64) -> Self
    pub fn median() -> Self
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn with_n_bootstrap(mut self, n_bootstrap: usize) -> Self
    pub fn with_confidence_level(mut self, level: f64) -> Self
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn coefficients(&self) -> Option<&Array1<f64>>
    pub fn intercept(&self) -> Option<f64>
    pub fn all_coefficients(&self) -> Option<Array1<f64>>
    pub fn quantile_loss(&self) -> Option<f64>
    pub fn fit_quantiles(x: &Array2<f64>, y: &Array1<f64>, quantiles: &[f64],) -> Result<MultiQuantileResults>
    pub fn fit_quantiles_with_options(x: &Array2<f64>, y: &Array1<f64>, quantiles: &[f64], fit_intercept:...) -> Result<MultiQuantileResults>
  }
  impl Model for QuantileRegression { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn feature_names(...), fn n_features(...), fn score(...) }
  impl StatisticalModel for QuantileRegression { fn summary(...), fn diagnostics(...), fn coefficients_with_ci(...), fn residuals(...), fn fitted_values(...), fn assumption_test(...) }
  impl ProbabilisticModel for QuantileRegression { fn predict_proba(...), fn predict_interval(...) }
  impl MultiQuantileResults {
    pub fn get_coefficient(&self, quantile_idx: usize, feature_idx: usize) -> Option<f64>
    pub fn get_std_error(&self, quantile_idx: usize, feature_idx: usize) -> Option<f64>
    pub fn coefficients_for_feature(&self, feature_idx: usize) -> Option<Array1<f64>>
    pub fn feature_name(&self, idx: usize) -> String
  }

ferroml-core/src/models/regularized.rs
  pub struct RidgeRegression { alpha: f64, fit_intercept: bool, normalize: bool, max_iter: usize, tol: f64, confidence_level: f64, feature_names: Option<Vec<String>> }
  pub struct LassoRegression { alpha: f64, fit_intercept: bool, max_iter: usize, tol: f64, warm_start: bool, confidence_level: f64, feature_names: Option<Vec<String>> }
  pub struct ElasticNet { alpha: f64, l1_ratio: f64, fit_intercept: bool, max_iter: usize, tol: f64, warm_start: bool, confidence_level: f64, feature_names: Option<Vec<String>> }
  pub struct RegularizationPath { alphas: Vec<f64>, coefs: Array2<f64>, intercepts: Vec<f64>, n_nonzeros: Vec<usize>, r_squared: Option<Vec<f64>> }
  pub struct RidgeCV { alphas: Vec<f64>, cv: usize, fit_intercept: bool, feature_names: Option<Vec<String>> }
  pub struct LassoCV { n_alphas: usize, alphas: Option<Vec<f64>>, cv: usize, fit_intercept: bool, max_iter: usize, tol: f64, feature_names: Option<Vec<String>> }
  pub struct ElasticNetCV { n_alphas: usize, l1_ratios: Vec<f64>, cv: usize, fit_intercept: bool, max_iter: usize, tol: f64, feature_names: Option<Vec<String>> }
  pub struct RidgeClassifier { alpha: f64, fit_intercept: bool }
  impl RidgeRegression {
    pub fn new(alpha: f64) -> Self
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self
    pub fn coefficients(&self) -> Option<&Array1<f64>>
    pub fn intercept(&self) -> Option<f64>
    pub fn r_squared(&self) -> Option<f64>
  }
  impl Default for RidgeRegression { fn default(...) }
  impl Model for RidgeRegression { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn feature_names(...), fn n_features(...), fn score(...) }
  impl StatisticalModel for RidgeRegression { fn summary(...), fn diagnostics(...), fn coefficients_with_ci(...), fn residuals(...), fn fitted_values(...), fn assumption_test(...) }
  impl ProbabilisticModel for RidgeRegression { fn predict_proba(...), fn predict_interval(...) }
  impl LassoRegression {
    pub fn new(alpha: f64) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self
    pub fn coefficients(&self) -> Option<&Array1<f64>>
    pub fn intercept(&self) -> Option<f64>
    pub fn n_iter(&self) -> Option<usize>
    pub fn n_nonzero(&self) -> Option<usize>
    pub fn r_squared(&self) -> Option<f64>
  }
  impl Default for LassoRegression { fn default(...) }
  impl Model for LassoRegression { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn feature_names(...), fn n_features(...), fn score(...) }
  impl StatisticalModel for LassoRegression { fn summary(...), fn diagnostics(...), fn coefficients_with_ci(...), fn residuals(...), fn fitted_values(...), fn assumption_test(...) }
  impl ElasticNet {
    pub fn new(alpha: f64, l1_ratio: f64) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self
    pub fn coefficients(&self) -> Option<&Array1<f64>>
    pub fn intercept(&self) -> Option<f64>
    pub fn n_iter(&self) -> Option<usize>
    pub fn n_nonzero(&self) -> Option<usize>
    pub fn r_squared(&self) -> Option<f64>
  }
  impl Default for ElasticNet { fn default(...) }
  impl Model for ElasticNet { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn feature_names(...), fn n_features(...), fn score(...) }
  impl StatisticalModel for ElasticNet { fn summary(...), fn diagnostics(...), fn coefficients_with_ci(...), fn residuals(...), fn fitted_values(...), fn assumption_test(...) }
  impl RegularizationPath {
    pub fn as_tuples(&self) -> Vec<(f64, Array1<f64>)>
    pub fn alpha_for_n_nonzero(&self, k: usize) -> Option<f64>
  }
  impl RidgeCV {
    pub fn new(alphas: Vec<f64>, cv: usize) -> Self
    pub fn with_defaults(cv: usize) -> Self
    pub fn best_alpha(&self) -> Option<f64>
    pub fn cv_scores(&self) -> Option<&[f64]>
    pub fn model(&self) -> Option<&RidgeRegression>
  }
  impl Model for RidgeCV { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn feature_names(...), fn n_features(...), fn score(...) }
  impl LassoCV {
    pub fn new(n_alphas: usize, cv: usize) -> Self
    pub fn with_alphas(alphas: Vec<f64>, cv: usize) -> Self
    pub fn best_alpha(&self) -> Option<f64>
    pub fn cv_scores(&self) -> Option<&[f64]>
    pub fn alphas_used(&self) -> Option<&[f64]>
    pub fn model(&self) -> Option<&LassoRegression>
  }
  impl Model for LassoCV { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn feature_names(...), fn n_features(...), fn score(...) }
  impl ElasticNetCV {
    pub fn new(n_alphas: usize, l1_ratios: Vec<f64>, cv: usize) -> Self
    pub fn with_defaults(n_alphas: usize, cv: usize) -> Self
    pub fn best_alpha(&self) -> Option<f64>
    pub fn best_l1_ratio(&self) -> Option<f64>
    pub fn model(&self) -> Option<&ElasticNet>
  }
  impl Model for ElasticNetCV { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn feature_names(...), fn n_features(...), fn score(...) }
  impl RidgeClassifier {
    pub fn new(alpha: f64) -> Self
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn ridges(&self) -> Option<&[RidgeRegression]>
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for RidgeClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn model_name(...) }
  pub fn lasso_path(x: &Array2<f64>, y: &Array1<f64>, alphas: Option<&[f64]>, n_alphas: usize,) -> Result<RegularizationPath>
  pub fn elastic_net_path(x: &Array2<f64>, y: &Array1<f64>, l1_ratio: f64, alphas: Option<&[f64]>, n_al...) -> Result<RegularizationPath>

ferroml-core/src/models/robust.rs
  pub struct RobustRegression { estimator: MEstimator, tuning_constant: f64, fit_intercept: bool, max_iter: usize, tol: f64, confidence_level: f64, feature_names: Option<Vec<String>>, scale_method: ScaleMethod }
  pub enum MEstimator { Huber, Bisquare, Hampel, AndrewsWave }
  pub enum ScaleMethod { MAD, HuberProposal2 }
  impl Default for MEstimator { fn default(...) }
  impl MEstimator {
    pub fn default_tuning_constant(&self) -> f64
    pub fn efficiency_at_normal(&self) -> f64
    pub fn breakdown_point(&self) -> f64
    pub fn rho(&self, u: f64, k: f64) -> f64
    pub fn psi(&self, u: f64, k: f64) -> f64
    pub fn weight(&self, u: f64, k: f64) -> f64
    pub fn psi_prime(&self, u: f64, k: f64) -> f64
  }
  impl Default for ScaleMethod { fn default(...) }
  impl Default for RobustRegression { fn default(...) }
  impl RobustRegression {
    pub fn new() -> Self
    pub fn with_estimator(estimator: MEstimator) -> Self
    pub fn with_tuning_constant(mut self, k: f64) -> Self
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn with_confidence_level(mut self, level: f64) -> Self
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self
    pub fn with_scale_method(mut self, method: ScaleMethod) -> Self
    pub fn coefficients(&self) -> Option<&Array1<f64>>
    pub fn intercept(&self) -> Option<f64>
    pub fn all_coefficients(&self) -> Option<Array1<f64>>
    pub fn scale(&self) -> Option<f64>
    pub fn weights(&self) -> Option<&Array1<f64>>
    pub fn converged(&self) -> Option<bool>
  }
  impl Model for RobustRegression { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn feature_names(...), fn n_features(...), fn score(...) }
  impl StatisticalModel for RobustRegression { fn summary(...), fn diagnostics(...), fn coefficients_with_ci(...), fn residuals(...), fn fitted_values(...), fn assumption_test(...) }
  impl ProbabilisticModel for RobustRegression { fn predict_proba(...), fn predict_interval(...) }

ferroml-core/src/models/sgd.rs
  pub struct SGDClassifier { loss: SGDClassifierLoss, penalty: Penalty, alpha: f64, l1_ratio: f64, eta0: f64, learning_rate: LearningRateScheduleType, power_t: f64, max_iter: usize, ... }
  pub struct SGDRegressor { loss: SGDRegressorLoss, penalty: Penalty, alpha: f64, l1_ratio: f64, eta0: f64, learning_rate: LearningRateScheduleType, power_t: f64, max_iter: usize, ... }
  pub struct Perceptron { ... }
  pub struct PassiveAggressiveClassifier { c: f64, fit_intercept: bool, max_iter: usize, tol: f64, shuffle: bool, random_state: Option<u64> }
  pub enum SGDClassifierLoss { Hinge, Log, ModifiedHuber }
  pub enum SGDRegressorLoss { SquaredError, Huber, EpsilonInsensitive }
  pub enum Penalty { None, L2, L1, ElasticNet }
  pub enum LearningRateScheduleType { Constant, Optimal, InverseScaling }
  impl Default for SGDClassifierLoss { fn default(...) }
  impl Default for SGDRegressorLoss { fn default(...) }
  impl Default for Penalty { fn default(...) }
  impl Default for LearningRateScheduleType { fn default(...) }
  impl SGDClassifier {
    pub fn new() -> Self
    pub fn with_loss(mut self, loss: SGDClassifierLoss) -> Self
    pub fn with_penalty(mut self, penalty: Penalty) -> Self
    pub fn with_alpha(mut self, alpha: f64) -> Self
    pub fn with_eta0(mut self, eta0: f64) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_warm_start(mut self, warm_start: bool) -> Self
    pub fn coef(&self) -> Option<&Array2<f64>>
    pub fn intercept(&self) -> Option<&Array1<f64>>
    pub fn n_iter(&self) -> Option<usize>
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Default for SGDClassifier { fn default(...) }
  impl Model for SGDClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn feature_importance(...), fn model_name(...) }
  impl IncrementalModel for SGDClassifier { fn partial_fit(...), fn partial_fit_with_classes(...) }
  impl SGDRegressor {
    pub fn new() -> Self
    pub fn with_loss(mut self, loss: SGDRegressorLoss) -> Self
    pub fn with_penalty(mut self, penalty: Penalty) -> Self
    pub fn with_alpha(mut self, alpha: f64) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_warm_start(mut self, warm_start: bool) -> Self
    pub fn coef(&self) -> Option<&Array1<f64>>
    pub fn intercept(&self) -> Option<f64>
  }
  impl Default for SGDRegressor { fn default(...) }
  impl Model for SGDRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn feature_importance(...), fn model_name(...), fn score(...) }
  impl IncrementalModel for SGDRegressor { fn partial_fit(...) }
  impl Perceptron {
    pub fn new() -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_penalty(mut self, penalty: Penalty) -> Self
    pub fn with_alpha(mut self, alpha: f64) -> Self
  }
  impl Default for Perceptron { fn default(...) }
  impl Model for Perceptron { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn model_name(...) }
  impl Perceptron {
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl IncrementalModel for Perceptron { fn partial_fit(...), fn partial_fit_with_classes(...) }
  impl PassiveAggressiveClassifier {
    pub fn new(c: f64) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn coef(&self) -> Option<&Array2<f64>>
    pub fn intercept(&self) -> Option<&Array1<f64>>
    pub fn classes(&self) -> Option<&Array1<f64>>
  }
  impl Model for PassiveAggressiveClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn model_name(...) }
  impl IncrementalModel for PassiveAggressiveClassifier { fn partial_fit(...), fn partial_fit_with_classes(...) }

ferroml-core/src/models/svm.rs
  pub struct SVC { c: f64, kernel: Kernel, tol: f64, max_iter: usize, probability: bool, multiclass_strategy: MulticlassStrategy, class_weight: ClassWeight, cache_size: usize }
  pub struct SVR { c: f64, epsilon: f64, kernel: Kernel, tol: f64, max_iter: usize }
  pub struct LinearSVC { c: f64, loss: LinearSVCLoss, tol: f64, max_iter: usize, fit_intercept: bool, class_weight: ClassWeight }
  pub struct LinearSVR { c: f64, epsilon: f64, loss: LinearSVRLoss, tol: f64, max_iter: usize, fit_intercept: bool }
  pub enum Kernel { Linear, Rbf, Polynomial, Sigmoid }
  pub enum MulticlassStrategy { OneVsOne, OneVsRest }
  pub enum ClassWeight { Uniform, Balanced }
  pub enum LinearSVCLoss { Hinge, SquaredHinge }
  pub enum LinearSVRLoss { EpsilonInsensitive, SquaredEpsilonInsensitive }
  impl Default for Kernel { fn default(...) }
  impl Kernel {
    pub fn rbf_auto() -> Self
    pub fn rbf(gamma: f64) -> Self
    pub fn poly(degree: u32, gamma: f64, coef0: f64) -> Self
    pub fn sigmoid(gamma: f64, coef0: f64) -> Self
    pub fn compute(&self, x: &[f64], y: &[f64]) -> f64
    pub fn with_auto_gamma(self, n_features: usize) -> Self
  }
  impl Default for MulticlassStrategy { fn default(...) }
  impl Default for ClassWeight { fn default(...) }
  impl Default for SVC { fn default(...) }
  impl SVC {
    pub fn new() -> Self
    pub fn with_c(mut self, c: f64) -> Self
    pub fn with_kernel(mut self, kernel: Kernel) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_probability(mut self, probability: bool) -> Self
    pub fn with_multiclass_strategy(mut self, strategy: MulticlassStrategy) -> Self
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self
    pub fn with_cache_size(mut self, cache_size: usize) -> Self
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn n_support_vectors(&self) -> Vec<usize>
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for SVC { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn search_space(...) }
  impl ProbabilisticModel for SVC { fn predict_proba(...), fn predict_interval(...) }
  impl Default for SVR { fn default(...) }
  impl SVR {
    pub fn new() -> Self
    pub fn with_c(mut self, c: f64) -> Self
    pub fn with_epsilon(mut self, epsilon: f64) -> Self
    pub fn with_kernel(mut self, kernel: Kernel) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn n_support_vectors(&self) -> usize
    pub fn support_vectors(&self) -> Option<&Array2<f64>>
    pub fn support_indices(&self) -> Option<&[usize]>
    pub fn dual_coef(&self) -> Option<&Array1<f64>>
    pub fn intercept(&self) -> f64
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>>
  }
  impl Model for SVR { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn search_space(...), fn score(...) }
  impl Default for LinearSVCLoss { fn default(...) }
  impl Default for LinearSVC { fn default(...) }
  impl LinearSVC {
    pub fn new() -> Self
    pub fn with_c(mut self, c: f64) -> Self
    pub fn with_loss(mut self, loss: LinearSVCLoss) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn weights(&self) -> Option<&Vec<Array1<f64>>>
    pub fn intercepts(&self) -> Option<&Vec<f64>>
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for LinearSVC { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn feature_importance(...), fn search_space(...) }
  impl Default for LinearSVRLoss { fn default(...) }
  impl Default for LinearSVR { fn default(...) }
  impl LinearSVR {
    pub fn new() -> Self
    pub fn with_c(mut self, c: f64) -> Self
    pub fn with_epsilon(mut self, epsilon: f64) -> Self
    pub fn with_loss(mut self, loss: LinearSVRLoss) -> Self
    pub fn with_tol(mut self, tol: f64) -> Self
    pub fn with_max_iter(mut self, max_iter: usize) -> Self
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self
    pub fn weights(&self) -> Option<&Array1<f64>>
    pub fn intercept(&self) -> f64
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>>
  }
  impl Model for LinearSVR { fn fit(...), fn predict(...), fn is_fitted(...), fn n_features(...), fn feature_importance(...), fn search_space(...), fn score(...) }

ferroml-core/src/models/traits.rs
  pub trait LinearModel: super::Model {
    fn coefficients(&self) -> Option<&Array1<f64>>
    fn intercept(&self) -> Option<f64>
    fn coefficient_std_errors(&self) -> Option<&Array1<f64>>
    fn coefficient_intervals(&self, _confidence: f64) -> Option<Array2<f64>>
  }
  pub trait IncrementalModel: super::Model {
    fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>
    fn partial_fit_with_classes(&mut self, x: &Array2<f64>, y: &Array1<f64>, _classes: Op...) -> Result<()>
  }
  pub trait WeightedModel: super::Model {
    fn fit_weighted(&mut self, x: &Array2<f64>, y: &Array1<f64>, sample_weigh...) -> Result<()>
  }
  pub trait SparseModel {
    fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix, y: &Array1<f64>) -> Result<()>
    fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array1<f64>>
  }
  pub trait TreeModel: super::Model {
    fn feature_importances(&self) -> Option<&Array1<f64>>
    fn n_estimators(&self) -> usize
    fn tree_depths(&self) -> Option<Vec<usize>>
  }
  pub trait OutlierDetector: Send + Sync {
    fn fit_unsupervised(&mut self, x: &Array2<f64>) -> Result<()>
    fn predict_outliers(&self, x: &Array2<f64>) -> Result<Array1<i32>>
    fn fit_predict_outliers(&mut self, x: &Array2<f64>) -> Result<Array1<i32>>
    fn score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>>
    fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>>
    fn is_fitted(&self) -> bool
    fn offset(&self) -> f64
  }
  pub trait WarmStartModel: super::Model {
    fn set_warm_start(&mut self, warm_start: bool)
    fn warm_start(&self) -> bool
    fn n_estimators_fitted(&self) -> usize
  }

ferroml-core/src/models/tree.rs
  pub struct TreeNode { id: usize, feature_index: Option<usize>, threshold: Option<f64>, impurity: f64, n_samples: usize, weighted_n_samples: f64, value: Vec<f64>, left_child: Option<usize>, ... }
  pub struct TreeStructure { nodes: Vec<TreeNode>, n_features: usize, n_classes: usize, feature_names: Option<Vec<String>>, class_names: Option<Vec<String>>, max_depth: usize, n_leaves: usize }
  pub struct DecisionTreeClassifier { criterion: SplitCriterion, max_depth: Option<usize>, min_samples_split: usize, min_samples_leaf: usize, max_features: Option<usize>, min_impurity_decrease: f64, ccp_alpha: f64, random_state: Option<u64>, ... }
  pub struct DecisionTreeRegressor { criterion: SplitCriterion, max_depth: Option<usize>, min_samples_split: usize, min_samples_leaf: usize, max_features: Option<usize>, min_impurity_decrease: f64, ccp_alpha: f64, random_state: Option<u64>, ... }
  pub enum SplitStrategy { Best, Random }
  pub enum SplitCriterion { Gini, Entropy, Mse, Mae }
  impl Default for SplitStrategy { fn default(...) }
  impl Default for SplitCriterion { fn default(...) }
  impl SplitCriterion {
    pub fn is_classification(&self) -> bool
    pub fn is_regression(&self) -> bool
  }
  impl TreeStructure {
    pub fn root(&self) -> Option<&TreeNode>
    pub fn get_node(&self, id: usize) -> Option<&TreeNode>
    pub fn to_dot(&self) -> String
    pub fn decision_path(&self, x: &[f64]) -> Vec<usize>
  }
  impl Default for DecisionTreeClassifier { fn default(...) }
  impl DecisionTreeClassifier {
    pub fn new() -> Self
    pub fn with_split_strategy(mut self, strategy: SplitStrategy) -> Self
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self
    pub fn with_min_impurity_decrease(mut self, min_impurity_decrease: f64) -> Self
    pub fn with_ccp_alpha(mut self, ccp_alpha: f64) -> Self
    pub fn with_random_state(mut self, random_state: u64) -> Self
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self
    pub fn tree(&self) -> Option<&TreeStructure>
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn get_depth(&self) -> Option<usize>
    pub fn get_n_leaves(&self) -> Option<usize>
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl Model for DecisionTreeClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn n_features(...), fn fit_weighted(...), fn model_name(...) }
  impl Default for DecisionTreeRegressor { fn default(...) }
  impl DecisionTreeRegressor {
    pub fn new() -> Self
    pub fn with_split_strategy(mut self, strategy: SplitStrategy) -> Self
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self
    pub fn with_min_impurity_decrease(mut self, min_impurity_decrease: f64) -> Self
    pub fn with_ccp_alpha(mut self, ccp_alpha: f64) -> Self
    pub fn with_random_state(mut self, random_state: u64) -> Self
    pub fn tree(&self) -> Option<&TreeStructure>
    pub fn get_depth(&self) -> Option<usize>
    pub fn get_n_leaves(&self) -> Option<usize>
  }
  impl Model for DecisionTreeRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn n_features(...), fn score(...) }

ferroml-core/src/models/naive_bayes/bernoulli.rs
  pub struct BernoulliNB { alpha: f64, binarize: Option<f64>, fit_prior: bool, class_prior: Option<Array1<f64>>, class_weight: ClassWeight }
  impl Default for BernoulliNB { fn default(...) }
  impl BernoulliNB {
    pub fn new() -> Self
    pub fn with_alpha(mut self, alpha: f64) -> Self
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self
    pub fn with_binarize(mut self, threshold: Option<f64>) -> Self
    pub fn with_fit_prior(mut self, fit_prior: bool) -> Self
    pub fn with_class_prior(mut self, priors: Array1<f64>) -> Self
    pub fn class_log_prior(&self) -> Option<&Array1<f64>>
    pub fn feature_log_prob(&self) -> Option<&Array2<f64>>
    pub fn feature_count(&self) -> Option<&Array2<f64>>
    pub fn class_count(&self) -> Option<&Array1<f64>>
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, classes: Option<Vec<f6...) -> Result<()>
  }
  impl Model for BernoulliNB { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn feature_names(...), fn n_features(...) }
  impl ProbabilisticModel for BernoulliNB { fn predict_proba(...), fn predict_interval(...) }

ferroml-core/src/models/naive_bayes/categorical.rs
  pub struct CategoricalNB { alpha: f64, fit_prior: bool, class_prior: Option<Array1<f64>> }
  impl Default for CategoricalNB { fn default(...) }
  impl CategoricalNB {
    pub fn new() -> Self
    pub fn with_alpha(mut self, alpha: f64) -> Self
    pub fn with_fit_prior(mut self, fit_prior: bool) -> Self
    pub fn with_class_prior(mut self, priors: Array1<f64>) -> Self
    pub fn class_log_prior(&self) -> Option<&Array1<f64>>
    pub fn category_count(&self) -> Option<&Vec<Array2<f64>>>
    pub fn feature_log_prob(&self) -> Option<&Vec<Array2<f64>>>
    pub fn feature_categories(&self) -> Option<&Vec<Vec<f64>>>
    pub fn class_count(&self) -> Option<&Array1<f64>>
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, classes: Option<Vec<f6...) -> Result<()>
  }
  impl Model for CategoricalNB { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn feature_names(...), fn n_features(...) }
  impl ProbabilisticModel for CategoricalNB { fn predict_proba(...), fn predict_interval(...) }

ferroml-core/src/models/naive_bayes/gaussian.rs
  pub struct GaussianNB { var_smoothing: f64, priors: Option<Array1<f64>>, class_weight: ClassWeight }
  impl Default for GaussianNB { fn default(...) }
  impl GaussianNB {
    pub fn new() -> Self
    pub fn with_var_smoothing(mut self, var_smoothing: f64) -> Self
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self
    pub fn with_priors(mut self, priors: Array1<f64>) -> Self
    pub fn theta(&self) -> Option<&Array2<f64>>
    pub fn var(&self) -> Option<&Array2<f64>>
    pub fn var_smoothed(&self) -> Option<&Array2<f64>>
    pub fn class_prior(&self) -> Option<&Array1<f64>>
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn class_count(&self) -> Option<&Array1<f64>>
    pub fn epsilon(&self) -> Option<f64>
    pub fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, classes: Option<Vec<f6...) -> Result<()>
  }
  impl Model for GaussianNB { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn feature_names(...), fn n_features(...) }
  impl ProbabilisticModel for GaussianNB { fn predict_proba(...), fn predict_interval(...) }

ferroml-core/src/models/naive_bayes/multinomial.rs
  pub struct MultinomialNB { alpha: f64, fit_prior: bool, class_prior: Option<Array1<f64>>, class_weight: ClassWeight }
  impl Default for MultinomialNB { fn default(...) }
  impl MultinomialNB {
    pub fn new() -> Self
    pub fn with_alpha(mut self, alpha: f64) -> Self
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self
    pub fn with_fit_prior(mut self, fit_prior: bool) -> Self
    pub fn with_class_prior(mut self, priors: Array1<f64>) -> Self
    pub fn class_log_prior(&self) -> Option<&Array1<f64>>
    pub fn feature_log_prob(&self) -> Option<&Array2<f64>>
    pub fn feature_count(&self) -> Option<&Array2<f64>>
    pub fn class_count(&self) -> Option<&Array1<f64>>
    pub fn classes(&self) -> Option<&Array1<f64>>
    pub fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, classes: Option<Vec<f6...) -> Result<()>
  }
  impl Model for MultinomialNB { fn fit(...), fn predict(...), fn is_fitted(...), fn feature_importance(...), fn search_space(...), fn feature_names(...), fn n_features(...) }
  impl ProbabilisticModel for MultinomialNB { fn predict_proba(...), fn predict_interval(...) }

ferroml-core/src/neural/activations.rs
  pub enum Activation { ReLU, Sigmoid, Tanh, Softmax, Linear, LeakyReLU, ELU }
  impl Activation {
    pub fn apply(&self, x: &Array1<f64>) -> Array1<f64>
    pub fn apply_2d(&self, x: &Array2<f64>) -> Array2<f64>
    pub fn derivative(&self, x: &Array1<f64>, output: &Array1<f64>) -> Array1<f64>
    pub fn derivative_2d(&self, x: &Array2<f64>, output: &Array2<f64>) -> Array2<f64>
    pub fn name(&self) -> &'static str
  }

ferroml-core/src/neural/analysis.rs
  pub struct WeightStatistics { layer_idx: usize, n_weights: usize, mean: f64, std: f64, min: f64, max: f64, sparsity: f64, mean_abs: f64, ... }
  pub struct DeadNeuron { layer_idx: usize, neuron_idx: usize, activation_rate: f64 }
  pub struct WeightDistributionTest { layer_idx: usize, is_normal: bool, normality_p_value: f64, expected_xavier_std: f64, expected_he_std: f64, actual_std: f64, initialization_quality: InitializationQuality }
  pub struct WeightChange { layer_idx: usize, mean_abs_change: f64, max_abs_change: f64, significant_changes: f64 }
  pub enum InitializationQuality { Good, TooLarge, TooSmall, UnexpectedDistribution }
  impl WeightStatistics {
    pub fn from_layer(layer_idx: usize, layer: &Layer) -> Self
  }
  pub fn weight_statistics(layers: &[Layer]) -> Vec<WeightStatistics>
  pub fn dead_neuron_detection(layers: &mut [Layer], x: &Array2<f64>) -> Result<Vec<DeadNeuron>>
  pub fn weight_distribution_tests(layers: &[Layer]) -> Vec<WeightDistributionTest>
  pub fn analyze_weight_changes(before: &[(&Array2<f64>, &Array1<f64>)

ferroml-core/src/neural/classifier.rs
  pub struct MLPClassifier { mlp: MLP, n_classes: Option<usize>, classes_: Option<Vec<f64>>, warm_start: bool }
  impl Default for MLPClassifier { fn default(...) }
  impl MLPClassifier {
    pub fn new() -> Self
    pub fn hidden_layer_sizes(mut self, sizes: &[usize]) -> Self
    pub fn activation(mut self, activation: Activation) -> Self
    pub fn solver(mut self, solver: Solver) -> Self
    pub fn learning_rate(mut self, lr: f64) -> Self
    pub fn max_iter(mut self, max_iter: usize) -> Self
    pub fn random_state(mut self, seed: u64) -> Self
    pub fn early_stopping(mut self, config: EarlyStopping) -> Self
    pub fn alpha(mut self, alpha: f64) -> Self
    pub fn batch_size(mut self, size: usize) -> Self
    pub fn tol(mut self, tol: f64) -> Self
    pub fn verbose(mut self, level: usize) -> Self
    pub fn with_warm_start(mut self, warm_start: bool) -> Self
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>
  }
  impl NeuralModel for MLPClassifier { fn fit(...), fn predict(...), fn is_fitted(...), fn n_layers(...), fn layer_sizes(...) }
  impl NeuralDiagnostics for MLPClassifier { fn training_diagnostics(...), fn weight_statistics(...), fn dead_neurons(...) }
  impl NeuralUncertainty for MLPClassifier { fn predict_with_uncertainty(...) }

ferroml-core/src/neural/diagnostics.rs
  pub struct TrainingDiagnostics { loss_curve: Vec<f64>, val_loss_curve: Option<Vec<f64>>, gradient_stats: Vec<Vec<GradientStatistics>>, convergence: ConvergenceStatus, n_iter: usize, best_val_loss: Option<f64>, best_iter: Option<usize>, learning_rate: f64, ... }
  pub struct GradientStatistics { layer_idx: usize, mean: f64, std: f64, max_abs: f64, min_abs: f64, sparsity: f64 }
  pub enum ConvergenceStatus { NotStarted, InProgress, Converged, Plateau, Diverged, Unstable }
  pub enum LearningRateAnalysis { InsufficientData, TooHigh, TooLow, Good }
  impl TrainingDiagnostics {
    pub fn new(learning_rate: f64) -> Self
    pub fn record_epoch(&mut self, loss: f64, val_loss: Option<f64>)
    pub fn record_gradients(&mut self, stats: Vec<GradientStatistics>)
    pub fn analyze_convergence(&mut self, tol: f64, patience: usize)
    pub fn summary(&self) -> String
    pub fn has_vanishing_gradients(&self) -> bool
    pub fn has_exploding_gradients(&self) -> bool
  }
  impl ConvergenceStatus {
    pub fn is_converged(&self) -> bool
    pub fn should_stop(&self) -> bool
  }
  impl GradientStatistics {
    pub fn from_gradients(layer_idx: usize, gradients: &Array1<f64>) -> Self
  }
  pub fn analyze_learning_rate(loss_curve: &[f64]) -> LearningRateAnalysis

ferroml-core/src/neural/layers.rs
  pub struct Layer { weights: Array2<f64>, biases: Array1<f64>, activation: Activation, n_inputs: usize, n_outputs: usize, last_input: Option<Array2<f64>>, last_z: Option<Array2<f64>>, last_output: Option<Array2<f64>>, ... }
  impl Layer {
    pub fn new(n_inputs: usize, n_outputs: usize, activation: Activation, weight_i...) -> Self
    pub fn forward(&mut self, input: &Array2<f64>, training: bool) -> Result<Array2<f64>>
    pub fn forward_with_dropout(&mut self, input: &Array2<f64>, dropout_rate: f64, training: bool, ...) -> Result<Array2<f64>>
    pub fn backward(&self, grad_output: &Array2<f64>,) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)>
    pub fn backward_skip_activation(&self, grad_output: &Array2<f64>,) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)>
    pub fn n_params(&self) -> usize
    pub fn clear_cache(&mut self)
    pub fn forward_gpu(&mut self, input: &Array2<f64>, training: bool, gpu: &dyn crate::gp...) -> Result<Array2<f64>>
    pub fn backward_gpu(&self, grad_output: &Array2<f64>, gpu: &dyn crate::gpu::GpuBackend,) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)>
    pub fn backward_gpu_skip_activation(&self, grad_output: &Array2<f64>, gpu: &dyn crate::gpu::GpuBackend,) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)>
  }

ferroml-core/src/neural/mlp.rs
  pub struct MLP { hidden_layer_sizes: Vec<usize>, hidden_activation: Activation, output_activation: Activation, solver: Solver, max_iter: usize, tol: f64, batch_size: Option<usize>, random_state: Option<u64>, ... }
  impl Default for MLP { fn default(...) }
  impl MLP {
    pub fn new() -> Self
    pub fn hidden_layer_sizes(mut self, sizes: &[usize]) -> Self
    pub fn activation(mut self, activation: Activation) -> Self
    pub fn output_activation(mut self, activation: Activation) -> Self
    pub fn solver(mut self, solver: Solver) -> Self
    pub fn with_gpu(mut self, backend: std::sync::Arc<dyn crate::gpu::GpuBackend>) -> Self
    pub fn learning_rate(mut self, lr: f64) -> Self
    pub fn max_iter(mut self, max_iter: usize) -> Self
    pub fn tol(mut self, tol: f64) -> Self
    pub fn batch_size(mut self, size: usize) -> Self
    pub fn random_state(mut self, seed: u64) -> Self
    pub fn early_stopping(mut self, config: EarlyStopping) -> Self
    pub fn alpha(mut self, alpha: f64) -> Self
    pub fn dropout(mut self, rate: f64) -> Self
    pub fn momentum(mut self, momentum: f64) -> Self
    pub fn verbose(mut self, level: usize) -> Self
    pub fn is_fitted(&self) -> bool
    pub fn layer_sizes(&self) -> Vec<usize>
    pub fn initialize(&mut self, n_features: usize, n_outputs: usize) -> Result<()>
    pub fn forward(&mut self, x: &Array2<f64>, training: bool) -> Result<Array2<f64>>
    pub fn backward(&self, loss_grad: &Array2<f64>) -> Result<Vec<(Array2<f64>, Array1<f64>)>>
    pub fn update_weights(&mut self, gradients: &[(Array2<f64>, Array1<f64>)
    pub fn n_params(&self) -> usize
    pub fn get_weights(&self) -> Vec<(&Array2<f64>, &Array1<f64>)>
  }

ferroml-core/src/neural/mod.rs
  pub struct EarlyStopping { patience: usize, min_delta: f64, validation_fraction: f64 }
  pub struct Regularization { alpha: f64, dropout_rate: f64 }
  pub struct TrainingHistory { loss_curve: Vec<f64>, val_loss_curve: Option<Vec<f64>>, best_iter: Option<usize>, n_iter: usize, converged: bool }
  pub enum WeightInit { XavierUniform, XavierNormal, HeUniform, HeNormal, Uniform, Normal }
  pub trait NeuralModel {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>
    fn is_fitted(&self) -> bool
    fn n_layers(&self) -> usize
    fn layer_sizes(&self) -> Vec<usize>
  }
  pub trait NeuralDiagnostics: NeuralModel {
    fn training_diagnostics(&self) -> Option<&TrainingDiagnostics>
    fn weight_statistics(&self) -> Result<Vec<WeightStatistics>>
    fn dead_neurons(&self, x: &Array2<f64>) -> Result<Vec<(usize, usize)>>
  }
  pub trait NeuralUncertainty: NeuralModel {
    fn predict_with_uncertainty(&self, x: &Array2<f64>, n_samples: usize, confidence: f64,) -> Result<PredictionUncertainty>
  }
  impl Default for EarlyStopping { fn default(...) }
  impl Default for Regularization { fn default(...) }
  impl TrainingHistory {
    pub fn new() -> Self
  }
  impl Default for TrainingHistory { fn default(...) }

ferroml-core/src/neural/optimizers.rs
  pub struct SGDState { velocities_w: Vec<Array2<f64>>, velocities_b: Vec<Array1<f64>> }
  pub struct AdamState { m_w: Vec<Array2<f64>>, m_b: Vec<Array1<f64>>, v_w: Vec<Array2<f64>>, v_b: Vec<Array1<f64>>, t: usize }
  pub struct OptimizerConfig { learning_rate: f64, momentum: f64, beta1: f64, beta2: f64, epsilon: f64, lr_schedule: LearningRateSchedule }
  pub enum Solver { SGD, Adam }
  pub enum LearningRateSchedule { Constant, InverseScaling, Adaptive }
  impl Solver {
    pub fn name(&self) -> &'static str
  }
  impl SGDState {
    pub fn new(layer_sizes: &[(usize, usize)
  }
  impl AdamState {
    pub fn new(layer_sizes: &[(usize, usize)
  }
  impl Default for OptimizerConfig { fn default(...) }
  impl OptimizerConfig {
    pub fn sgd(learning_rate: f64, momentum: f64) -> Self
    pub fn adam(learning_rate: f64) -> Self
    pub fn get_lr(&self, iteration: usize, best_loss: f64, current_loss: f64, plateau...) -> f64
  }
  pub fn sgd_step(weights: &mut Array2<f64>, biases: &mut Array1<f64>, grad_w: &Array2<f64>, gr...)
  pub fn adam_step(weights: &mut Array2<f64>, biases: &mut Array1<f64>, grad_w: &Array2<f64>, gr...)

ferroml-core/src/neural/regressor.rs
  pub struct MLPRegressor { mlp: MLP, n_outputs: Option<usize>, warm_start: bool }
  impl Default for MLPRegressor { fn default(...) }
  impl MLPRegressor {
    pub fn new() -> Self
    pub fn hidden_layer_sizes(mut self, sizes: &[usize]) -> Self
    pub fn activation(mut self, activation: Activation) -> Self
    pub fn solver(mut self, solver: Solver) -> Self
    pub fn learning_rate(mut self, lr: f64) -> Self
    pub fn max_iter(mut self, max_iter: usize) -> Self
    pub fn random_state(mut self, seed: u64) -> Self
    pub fn early_stopping(mut self, config: EarlyStopping) -> Self
    pub fn alpha(mut self, alpha: f64) -> Self
    pub fn batch_size(mut self, size: usize) -> Self
    pub fn tol(mut self, tol: f64) -> Self
    pub fn verbose(mut self, level: usize) -> Self
    pub fn with_warm_start(mut self, warm_start: bool) -> Self
    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64>
  }
  impl NeuralModel for MLPRegressor { fn fit(...), fn predict(...), fn is_fitted(...), fn n_layers(...), fn layer_sizes(...) }
  impl NeuralDiagnostics for MLPRegressor { fn training_diagnostics(...), fn weight_statistics(...), fn dead_neurons(...) }
  impl NeuralUncertainty for MLPRegressor { fn predict_with_uncertainty(...) }

ferroml-core/src/neural/uncertainty.rs
  pub struct PredictionUncertainty { mean: Array1<f64>, std: Array1<f64>, lower: Array1<f64>, upper: Array1<f64>, confidence: f64, n_samples: usize, samples: Option<Array2<f64>> }
  pub struct CalibrationResult { prob_bins: Vec<f64>, observed_freq: Vec<f64>, expected_freq: Vec<f64>, ece: f64, mce: f64, bin_counts: Vec<usize> }
  pub struct ReliabilityDiagram { bin_edges: Vec<f64>, mean_predicted: Vec<f64>, fraction_positives: Vec<f64>, counts: Vec<usize> }
  impl PredictionUncertainty {
    pub fn interval_width(&self) -> Array1<f64>
    pub fn coefficient_of_variation(&self) -> Array1<f64>
  }
  impl From<CalibrationResult> for ReliabilityDiagram { fn from(...) }
  pub fn predict_with_uncertainty(layers: &mut [Layer], x: &Array2<f64>, n_samples: usize, dropout_rate: f64, c...) -> Result<PredictionUncertainty>
  pub fn calibration_analysis(predicted_probs: &Array1<f64>, true_labels: &Array1<f64>, n_bins: usize,) -> CalibrationResult

ferroml-core/src/onnx/hist_boosting.rs
  impl OnnxExportable for HistGradientBoostingRegressor { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for HistGradientBoostingClassifier { fn to_onnx(...), fn onnx_n_features(...) }

ferroml-core/src/onnx/linear.rs
  impl OnnxExportable for LinearRegression { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for RidgeRegression { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for LassoRegression { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for ElasticNet { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for LogisticRegression { fn to_onnx(...), fn onnx_n_features(...), fn onnx_n_outputs(...) }
  impl OnnxExportable for RidgeClassifier { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for RobustRegression { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for QuantileRegression { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for SGDRegressor { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for PassiveAggressiveClassifier { fn to_onnx(...), fn onnx_n_features(...) }

ferroml-core/src/onnx/mod.rs
  pub struct OnnxConfig { model_name: String, description: Option<String>, producer_name: String, producer_version: String, input_name: String, output_name: String, input_shape: Option<(i64, opset_version: i64, ... }
  pub trait OnnxExportable {
    fn validate_onnx_config(&self, config: &OnnxConfig) -> Result<()>
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>>
    fn onnx_n_features(&self) -> Option<usize>
    fn onnx_n_outputs(&self) -> usize
    fn onnx_output_type(&self) -> TensorProtoDataType
  }
  impl Default for OnnxConfig { fn default(...) }
  impl OnnxConfig {
    pub fn new(model_name: impl Into<String>) -> Self
    pub fn with_description(mut self, description: impl Into<String>) -> Self
    pub fn with_input_name(mut self, name: impl Into<String>) -> Self
    pub fn with_output_name(mut self, name: impl Into<String>) -> Self
    pub fn with_input_shape(mut self, batch_size: i64, n_features: i64) -> Self
    pub fn with_opset_version(mut self, version: i64) -> Self
  }
  pub fn create_model_proto(graph: GraphProto, config: &OnnxConfig, use_ml_domain: bool,) -> ModelProto
  pub fn create_tensor_input(name: &str, n_features: usize, batch_size: Option<i64>, elem_type: TensorProt...) -> ValueInfoProto
  pub fn create_tensor_output(name: &str, n_outputs: usize, batch_size: Option<i64>, elem_type: TensorProto...) -> ValueInfoProto
  pub fn create_tensor_output_1d(name: &str, batch_size: Option<i64>, elem_type: TensorProtoDataType,) -> ValueInfoProto
  pub fn create_float_tensor(name: &str, dims: &[i64], data: Vec<f32>) -> TensorProto
  pub fn create_gemm_node(input_a: &str, input_b: &str, input_c: &str, output: &str, alpha: f32, beta: ...) -> NodeProto
  pub fn create_matmul_node(input_a: &str, input_b: &str, output: &str, name: &str) -> NodeProto
  pub fn create_add_node(input_a: &str, input_b: &str, output: &str, name: &str) -> NodeProto
  pub fn create_sigmoid_node(input: &str, output: &str, name: &str) -> NodeProto
  pub fn create_softmax_node(input: &str, output: &str, name: &str, axis: i64) -> NodeProto
  pub fn create_flatten_node(input: &str, output: &str, name: &str, axis: i64) -> NodeProto
  pub fn create_squeeze_node(input: &str, axes_input: &str, output: &str, name: &str) -> NodeProto
  pub fn create_reshape_node(input: &str, shape_input: &str, output: &str, name: &str) -> NodeProto
  pub fn create_int64_tensor(name: &str, dims: &[i64], data: Vec<i64>) -> TensorProto
  pub fn create_sub_node(input_a: &str, input_b: &str, output: &str, name: &str) -> NodeProto
  pub fn create_div_node(input_a: &str, input_b: &str, output: &str, name: &str) -> NodeProto
  pub fn create_mul_node(input_a: &str, input_b: &str, output: &str, name: &str) -> NodeProto
  pub fn create_argmax_node(input: &str, output: &str, name: &str, axis: i64, keepdims: i64,) -> NodeProto
  pub fn create_cast_node(input: &str, output: &str, name: &str, to_type: TensorProtoDataType,) -> NodeProto
  pub fn create_reduce_sum_node(input: &str, axes_input: &str, output: &str, name: &str, keepdims: i64,) -> NodeProto
  pub fn serialize_model(model: &ModelProto) -> Vec<u8>

ferroml-core/src/onnx/naive_bayes.rs
  impl OnnxExportable for MultinomialNB { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for BernoulliNB { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for GaussianNB { fn to_onnx(...), fn onnx_n_features(...) }

ferroml-core/src/onnx/preprocessing.rs
  impl OnnxExportable for StandardScaler { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for MinMaxScaler { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for RobustScaler { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for MaxAbsScaler { fn to_onnx(...), fn onnx_n_features(...) }

ferroml-core/src/onnx/protos.rs
  pub struct ModelProto { ir_version: i64, opset_import: Vec<OperatorSetIdProto>, producer_name: String, producer_version: String, domain: String, model_version: i64, doc_string: String, graph: Option<GraphProto>, ... }
  pub struct OperatorSetIdProto { domain: String, version: i64 }
  pub struct StringStringEntryProto { key: String, value: String }
  pub struct TrainingInfoProto { initialization: Option<GraphProto>, algorithm: Option<GraphProto> }
  pub struct FunctionProto { name: String, domain: String }
  pub struct GraphProto { node: Vec<NodeProto>, name: String, initializer: Vec<TensorProto>, sparse_initializer: Vec<SparseTensorProto>, doc_string: String, input: Vec<ValueInfoProto>, output: Vec<ValueInfoProto>, value_info: Vec<ValueInfoProto>, ... }
  pub struct NodeProto { input: Vec<String>, output: Vec<String>, name: String, op_type: String, domain: String, attribute: Vec<AttributeProto>, doc_string: String }
  pub struct AttributeProto { name: String, doc_string: String, ref_attr_name: String, f: f32, i: i64, s: Vec<u8>, t: Option<TensorProto>, g: Option<GraphProto>, ... }
  pub struct ValueInfoProto { name: String, doc_string: String }
  pub struct TypeProto { value: Option<type_proto::Value>, denotation: String }
  pub struct TypeProtoTensor { elem_type: i32, shape: Option<TensorShapeProto> }
  pub struct TypeProtoSequence { elem_type: Option<Box<TypeProto>> }
  pub struct TypeProtoMap { key_type: i32, value_type: Option<Box<TypeProto>> }
  pub struct TypeProtoOptional { elem_type: Option<Box<TypeProto>> }
  pub struct TensorShapeProto { dim: Vec<TensorShapeProtoDimension> }
  pub struct TensorShapeProtoDimension { value: Option<tensor_shape_proto_dimension::Value> }
  pub struct TensorProto { dims: Vec<i64>, data_type: i32, segment: Option<TensorProtoSegment>, float_data: Vec<f32>, int32_data: Vec<i32>, string_data: Vec<Vec<u8>>, int64_data: Vec<i64>, name: String, ... }
  pub struct TensorProtoSegment { begin: i64, end: i64 }
  pub struct SparseTensorProto { dims: Vec<i64>, indices: Option<TensorProto>, values: Option<TensorProto> }
  pub struct TensorAnnotation { tensor_name: String, quant_parameter_tensor_names: Vec<StringStringEntryProto> }
  pub enum AttributeProtoType {  }
  pub enum Value { TensorType, SequenceType, MapType, OptionalType }
  pub enum Value { DimValue, DimParam }
  pub enum TensorProtoDataType {  }

ferroml-core/src/onnx/sgd.rs
  impl OnnxExportable for SGDClassifier { fn to_onnx(...), fn onnx_n_features(...) }

ferroml-core/src/onnx/svm.rs
  impl OnnxExportable for LinearSVC { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for LinearSVR { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for SVR { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for SVC { fn to_onnx(...), fn onnx_n_features(...) }

ferroml-core/src/onnx/tree.rs
  impl OnnxExportable for DecisionTreeRegressor { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for DecisionTreeClassifier { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for RandomForestRegressor { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for RandomForestClassifier { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for ExtraTreesRegressor { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for ExtraTreesClassifier { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for GradientBoostingRegressor { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for GradientBoostingClassifier { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for AdaBoostClassifier { fn to_onnx(...), fn onnx_n_features(...) }
  impl OnnxExportable for AdaBoostRegressor { fn to_onnx(...), fn onnx_n_features(...) }

ferroml-core/src/pipeline/mod.rs
  pub struct PipelineCache { ... }
  pub struct Pipeline { ... }
  pub struct FeatureUnion { ... }
  pub struct ColumnTransformer { ... }
  pub enum PipelineStep { Transform, Model }
  pub enum CacheStrategy { None, Memory }
  pub enum ColumnSelector { Indices, Mask, All, Remainder }
  pub enum RemainderHandling { Drop, Passthrough }
  pub trait PipelineTransformer: Transformer {
    fn search_space(&self) -> SearchSpace
    fn clone_boxed(&self) -> Box<dyn PipelineTransformer>
    fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()>
    fn name(&self) -> &str
  }
  pub trait PipelineModel: Model {
    fn clone_boxed(&self) -> Box<dyn PipelineModel>
    fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()>
    fn name(&self) -> &str
  }
  impl Default for CacheStrategy { fn default(...) }
  impl PipelineCache {
    pub fn new(strategy: CacheStrategy) -> Self
    pub fn is_enabled(&self) -> bool
    pub fn get(&self, step_name: &str, input: &Array2<f64>) -> Option<&Array2<f64>>
    pub fn set(&mut self, step_name: &str, input: &Array2<f64>, output: Array2<f64>)
    pub fn clear(&mut self)
    pub fn invalidate_from(&mut self, step_names: &[String], starting_step: &str)
  }
  impl Default for PipelineCache { fn default(...) }
  impl Default for Pipeline { fn default(...) }
  impl Pipeline {
    pub fn new() -> Self
    pub fn with_cache(mut self, strategy: CacheStrategy) -> Self
    pub fn has_model(&self) -> bool
    pub fn step_names(&self) -> Vec<&str>
    pub fn n_steps(&self) -> usize
    pub fn is_empty(&self) -> bool
    pub fn is_fitted(&self) -> bool
    pub fn n_features_in(&self) -> Option<usize>
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn fit_transform(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>>
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>
    pub fn search_space(&self) -> SearchSpace
    pub fn set_params(&mut self, params: &HashMap<String, ParameterValue>) -> Result<()>
    pub fn get_params(&self) -> HashMap<String, String>
  }
  impl Default for FeatureUnion { fn default(...) }
  impl FeatureUnion {
    pub fn new() -> Self
    pub fn with_weights(mut self, weights: HashMap<String, f64>) -> Self
    pub fn transformer_names(&self) -> Vec<&str>
    pub fn n_transformers(&self) -> usize
    pub fn is_empty(&self) -> bool
    pub fn search_space(&self) -> SearchSpace
    pub fn set_params(&mut self, params: &HashMap<String, ParameterValue>) -> Result<()>
    pub fn get_params(&self) -> HashMap<String, String>
  }
  impl Transformer for FeatureUnion { fn fit(...), fn transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl PipelineTransformer for FeatureUnion { fn search_space(...), fn clone_boxed(...), fn set_param(...), fn name(...) }
  impl ColumnSelector {
    pub fn indices(indices: impl IntoIterator<Item = usize>) -> Self
    pub fn mask(mask: impl IntoIterator<Item = bool>) -> Self
    pub fn all() -> Self
  }
  impl Default for RemainderHandling { fn default(...) }
  impl Default for ColumnTransformer { fn default(...) }
  impl ColumnTransformer {
    pub fn new() -> Self
    pub fn with_remainder(mut self, handling: RemainderHandling) -> Self
    pub fn transformer_names(&self) -> Vec<&str>
    pub fn n_transformers(&self) -> usize
    pub fn is_empty(&self) -> bool
    pub fn search_space(&self) -> SearchSpace
    pub fn set_params(&mut self, params: &HashMap<String, ParameterValue>) -> Result<()>
    pub fn get_params(&self) -> HashMap<String, String>
  }
  impl Transformer for ColumnTransformer { fn fit(...), fn transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl PipelineTransformer for ColumnTransformer { fn search_space(...), fn clone_boxed(...), fn set_param(...), fn name(...) }
  pub fn make_pipeline(transformers: Vec<Box<dyn PipelineTransformer>>, model: Option<Box<dyn Pipeli...) -> Pipeline
  pub fn make_feature_union(transformers: Vec<(String, Box<dyn PipelineTransformer>)
  pub fn make_column_transformer(transformers: Vec<(String, Box<dyn PipelineTransformer>, ColumnSelector)

ferroml-core/src/pipeline/text_pipeline.rs
  pub struct TextPipeline { ... }
  pub enum TextPipelineStep { TextToSparse, SparseToSparse, SparseModel, DenseModel }
  pub trait PipelineTextTransformer: TextTransformer {
    fn search_space(&self) -> SearchSpace
    fn clone_boxed(&self) -> Box<dyn PipelineTextTransformer>
    fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()>
    fn name(&self) -> &str
    fn n_features_out(&self) -> Option<usize>
  }
  pub trait PipelineSparseTransformer: SparseTransformer {
    fn search_space(&self) -> SearchSpace
    fn clone_boxed(&self) -> Box<dyn PipelineSparseTransformer>
    fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()>
    fn name(&self) -> &str
  }
  pub trait PipelineSparseModel: Send + Sync {
    fn fit_sparse(&mut self, x: &CsrMatrix, y: &Array1<f64>) -> Result<()>
    fn predict_sparse(&self, x: &CsrMatrix) -> Result<Array1<f64>>
    fn search_space(&self) -> SearchSpace
    fn clone_boxed(&self) -> Box<dyn PipelineSparseModel>
    fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()>
    fn name(&self) -> &str
    fn is_fitted(&self) -> bool
  }
  impl Default for TextPipeline { fn default(...) }
  impl TextPipeline {
    pub fn new() -> Self
    pub fn fit(&mut self, documents: &[String], y: &Array1<f64>) -> Result<()>
    pub fn predict(&self, documents: &[String]) -> Result<Array1<f64>>
    pub fn transform(&self, documents: &[String]) -> Result<CsrMatrix>
    pub fn transform_dense(&self, documents: &[String]) -> Result<Array2<f64>>
    pub fn search_space(&self) -> SearchSpace
    pub fn set_params(&mut self, params: &HashMap<String, ParameterValue>) -> Result<()>
    pub fn step_names(&self) -> Vec<&str>
    pub fn n_steps(&self) -> usize
    pub fn is_fitted(&self) -> bool
    pub fn has_model(&self) -> bool
  }

ferroml-core/src/preprocessing/count_vectorizer.rs
  pub struct CountVectorizer { ... }
  pub enum DocFrequency { Count, Fraction }
  pub enum TokenPattern { Word }
  pub trait TextTransformer: Send + Sync {
    fn fit_text(&mut self, documents: &[String]) -> Result<()>
    fn transform_text(&self, documents: &[String]) -> Result<CsrMatrix>
    fn fit_transform_text(&mut self, documents: &[String]) -> Result<CsrMatrix>
  }
  impl Default for CountVectorizer { fn default(...) }
  impl CountVectorizer {
    pub fn new() -> Self
    pub fn with_max_features(mut self, max_features: usize) -> Self
    pub fn with_min_df(mut self, min_df: DocFrequency) -> Self
    pub fn with_max_df(mut self, max_df: DocFrequency) -> Self
    pub fn with_ngram_range(mut self, ngram_range: (usize, usize)
    pub fn with_binary(mut self, binary: bool) -> Self
    pub fn with_lowercase(mut self, lowercase: bool) -> Self
    pub fn with_token_pattern(mut self, token_pattern: TokenPattern) -> Self
    pub fn with_stop_words(mut self, stop_words: Vec<String>) -> Self
    pub fn vocabulary(&self) -> Option<&HashMap<String, usize>>
    pub fn get_feature_names(&self) -> Option<&[String]>
    pub fn is_fitted(&self) -> bool
    pub fn transform_text_dense(&self, documents: &[String]) -> Result<Array2<f64>>
    pub fn fit_transform_text_dense(&mut self, documents: &[String]) -> Result<Array2<f64>>
  }
  impl TextTransformer for CountVectorizer { fn fit_text(...), fn transform_text(...) }

ferroml-core/src/preprocessing/discretizers.rs
  pub struct KBinsDiscretizer { ... }
  pub enum BinningStrategy { Uniform, Quantile, KMeans }
  pub enum BinEncoding { Ordinal, OneHot }
  impl Default for BinningStrategy { fn default(...) }
  impl Default for BinEncoding { fn default(...) }
  impl KBinsDiscretizer {
    pub fn new() -> Self
    pub fn with_n_bins(mut self, n_bins: usize) -> Self
    pub fn with_strategy(mut self, strategy: BinningStrategy) -> Self
    pub fn with_encode(mut self, encode: BinEncoding) -> Self
    pub fn n_bins(&self) -> usize
    pub fn strategy(&self) -> BinningStrategy
    pub fn encode(&self) -> BinEncoding
    pub fn bin_edges(&self) -> Option<&Vec<Array1<f64>>>
    pub fn n_bins_per_feature(&self) -> Option<&Vec<usize>>
  }
  impl Default for KBinsDiscretizer { fn default(...) }
  impl Transformer for KBinsDiscretizer { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }

ferroml-core/src/preprocessing/encoders.rs
  pub struct OneHotEncoder { ... }
  pub struct OrdinalEncoder { ... }
  pub struct LabelEncoder { ... }
  pub struct TargetEncoder { ... }
  pub enum DropStrategy { None, First, IfBinary }
  impl Default for DropStrategy { fn default(...) }
  impl PartialEq for OrderedF64 { fn eq(...) }
  impl OneHotEncoder {
    pub fn new() -> Self
    pub fn with_handle_unknown(mut self, handling: UnknownCategoryHandling) -> Self
    pub fn with_drop(mut self, drop: DropStrategy) -> Self
    pub fn categories(&self) -> Option<&[Vec<f64>]>
  }
  impl Default for OneHotEncoder { fn default(...) }
  impl Transformer for OneHotEncoder { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl OrdinalEncoder {
    pub fn new() -> Self
    pub fn with_handle_unknown(mut self, handling: UnknownCategoryHandling) -> Self
    pub fn categories(&self) -> Option<&[Vec<f64>]>
    pub fn n_categories(&self) -> Option<Vec<usize>>
  }
  impl Default for OrdinalEncoder { fn default(...) }
  impl Transformer for OrdinalEncoder { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl LabelEncoder {
    pub fn new() -> Self
    pub fn classes(&self) -> Option<&[f64]>
    pub fn n_classes(&self) -> Option<usize>
    pub fn is_fitted(&self) -> bool
    pub fn fit_1d(&mut self, y: &Array1<f64>) -> Result<()>
    pub fn transform_1d(&self, y: &Array1<f64>) -> Result<Array1<f64>>
    pub fn fit_transform_1d(&mut self, y: &Array1<f64>) -> Result<Array1<f64>>
    pub fn inverse_transform_1d(&self, y: &Array1<f64>) -> Result<Array1<f64>>
  }
  impl Default for LabelEncoder { fn default(...) }
  impl Transformer for LabelEncoder { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl TargetEncoder {
    pub fn new() -> Self
    pub fn with_smooth(mut self, smooth: f64) -> Self
    pub fn with_handle_unknown(mut self, handling: UnknownCategoryHandling) -> Self
    pub fn with_cv(mut self, cv: usize) -> Self
    pub fn global_mean(&self) -> Option<f64>
    pub fn smooth(&self) -> f64
    pub fn fit_with_target(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>
    pub fn fit_transform_with_target(&mut self, x: &Array2<f64>, y: &Array1<f64>,) -> Result<Array2<f64>>
  }
  impl Default for TargetEncoder { fn default(...) }
  impl Transformer for TargetEncoder { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl PipelineTransformer for OneHotEncoder { fn clone_boxed(...), fn name(...) }

ferroml-core/src/preprocessing/imputers.rs
  pub struct SimpleImputer { ... }
  pub struct KNNImputer { ... }
  pub enum ImputeStrategy { Mean, Median, MostFrequent, Constant }
  pub enum KNNWeights { Uniform, Distance }
  pub enum KNNMetric { Euclidean, Manhattan }
  impl Default for ImputeStrategy { fn default(...) }
  impl SimpleImputer {
    pub fn new(strategy: ImputeStrategy) -> Self
    pub fn with_fill_value(mut self, value: f64) -> Self
    pub fn with_missing_value(mut self, value: f64) -> Self
    pub fn with_indicator(mut self, add: bool) -> Self
    pub fn statistics(&self) -> Option<&Array1<f64>>
    pub fn missing_counts(&self) -> Option<&[usize]>
    pub fn all_missing_features(&self) -> &[usize]
    pub fn strategy(&self) -> ImputeStrategy
  }
  impl Default for SimpleImputer { fn default(...) }
  impl Transformer for SimpleImputer { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl Default for KNNWeights { fn default(...) }
  impl Default for KNNMetric { fn default(...) }
  impl KNNImputer {
    pub fn new(n_neighbors: usize) -> Self
    pub fn with_weights(mut self, weights: KNNWeights) -> Self
    pub fn with_metric(mut self, metric: KNNMetric) -> Self
    pub fn n_neighbors(&self) -> usize
    pub fn weights(&self) -> KNNWeights
    pub fn metric(&self) -> KNNMetric
    pub fn column_means(&self) -> Option<&Array1<f64>>
    pub fn missing_counts(&self) -> Option<&[usize]>
  }
  impl Default for KNNImputer { fn default(...) }
  impl Transformer for KNNImputer { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl PipelineTransformer for SimpleImputer { fn clone_boxed(...), fn name(...) }
  impl PipelineTransformer for KNNImputer { fn clone_boxed(...), fn name(...) }

ferroml-core/src/preprocessing/mod.rs
  pub struct FitStatistics { n_samples: usize, n_features_in: usize, n_features_out: usize, constant_features: Vec<usize>, missing_counts: Option<Vec<usize>> }
  pub enum UnknownCategoryHandling { Error, Ignore, InfrequentIfExist }
  pub trait Transformer: Send + Sync {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()>
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>>
    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    fn is_fitted(&self) -> bool
    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>>
    fn n_features_in(&self) -> Option<usize>
    fn n_features_out(&self) -> Option<usize>
  }
  pub trait SparseTransformer: Send + Sync {
    fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix) -> Result<()>
    fn transform_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<crate::sparse::CsrMatrix>
    fn fit_transform_sparse(&mut self, x: &crate::sparse::CsrMatrix,) -> Result<crate::sparse::CsrMatrix>
    fn is_fitted(&self) -> bool
    fn n_features_out(&self) -> Option<usize>
  }
  impl Default for UnknownCategoryHandling { fn default(...) }
  pub fn check_is_fitted(is_fitted: bool, operation: &str) -> Result<()>
  pub fn check_shape(x: &Array2<f64>, expected_features: usize) -> Result<()>
  pub fn check_non_empty(x: &Array2<f64>) -> Result<()>
  pub fn check_finite(x: &Array2<f64>) -> Result<()>
  pub fn generate_feature_names(n_features: usize) -> Vec<String>
  pub fn compute_column_statistics(x: &Array2<f64>) -> (Array1<f64>, Array1<f64>, usize)
  pub fn column_mean(x: &Array2<f64>) -> Array1<f64>
  pub fn column_std(x: &Array2<f64>, ddof: usize) -> Array1<f64>
  pub fn column_min(x: &Array2<f64>) -> Array1<f64>
  pub fn column_max(x: &Array2<f64>) -> Array1<f64>
  pub fn column_median(x: &Array2<f64>) -> Array1<f64>
  pub fn column_quantile(x: &Array2<f64>, q: f64) -> Array1<f64>
  pub fn find_constant_features(x: &Array2<f64>, threshold: f64) -> Vec<usize>

ferroml-core/src/preprocessing/polynomial.rs
  pub struct PolynomialFeatures { ... }
  impl PolynomialFeatures {
    pub fn new(degree: usize) -> Self
    pub fn interaction_only(mut self, interaction_only: bool) -> Self
    pub fn include_bias(mut self, include_bias: bool) -> Self
    pub fn degree(&self) -> usize
    pub fn is_interaction_only(&self) -> bool
    pub fn has_bias(&self) -> bool
    pub fn powers(&self) -> Option<&Vec<Vec<(usize, usize)>>>
  }
  impl Default for PolynomialFeatures { fn default(...) }
  impl Transformer for PolynomialFeatures { fn fit(...), fn transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl PipelineTransformer for PolynomialFeatures { fn clone_boxed(...), fn name(...) }

ferroml-core/src/preprocessing/power.rs
  pub struct PowerTransformer { ... }
  pub enum PowerMethod { BoxCox, YeoJohnson }
  impl Default for PowerMethod { fn default(...) }
  impl PowerTransformer {
    pub fn new(method: PowerMethod) -> Self
    pub fn with_standardize(mut self, standardize: bool) -> Self
    pub fn method(&self) -> PowerMethod
    pub fn lambdas(&self) -> Option<&Array1<f64>>
  }
  impl Default for PowerTransformer { fn default(...) }
  impl Transformer for PowerTransformer { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }

ferroml-core/src/preprocessing/quantile.rs
  pub struct QuantileTransformer { ... }
  pub enum OutputDistribution { Uniform, Normal }
  impl Default for OutputDistribution { fn default(...) }
  impl QuantileTransformer {
    pub fn new(output_distribution: OutputDistribution) -> Self
    pub fn with_n_quantiles(mut self, n_quantiles: usize) -> Self
    pub fn with_subsample(mut self, subsample: Option<usize>) -> Self
    pub fn output_distribution(&self) -> OutputDistribution
    pub fn n_quantiles(&self) -> usize
    pub fn quantiles(&self) -> Option<&Vec<Array1<f64>>>
    pub fn references(&self) -> Option<&Array1<f64>>
  }
  impl Default for QuantileTransformer { fn default(...) }
  impl Transformer for QuantileTransformer { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }

ferroml-core/src/preprocessing/sampling.rs
  pub struct SMOTE { ... }
  pub struct BorderlineSMOTE { ... }
  pub struct ADASYN { ... }
  pub struct RandomUnderSampler { ... }
  pub struct RandomOverSampler { ... }
  pub struct SMOTETomek { ... }
  pub struct SMOTEENN { ... }
  pub enum SamplingStrategy { Auto, Ratio, TargetCounts, Classes, NotResampled }
  pub enum ENNKind { All, Mode }
  pub trait Resampler: Send + Sync {
    fn fit_resample(&mut self, x: &Array2<f64>, y: &Array1<f64>,) -> Result<(Array2<f64>, Array1<f64>)>
    fn strategy_description(&self) -> String
  }
  impl Default for SamplingStrategy { fn default(...) }
  impl Default for SMOTE { fn default(...) }
  impl SMOTE {
    pub fn new() -> Self
    pub fn with_k_neighbors(mut self, k: usize) -> Self
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn n_synthetic_samples(&self) -> Option<&HashMap<i64, usize>>
  }
  impl Resampler for SMOTE { fn fit_resample(...), fn strategy_description(...) }
  impl Default for BorderlineSMOTE { fn default(...) }
  impl BorderlineSMOTE {
    pub fn new() -> Self
    pub fn with_k_neighbors(mut self, k: usize) -> Self
    pub fn with_m_neighbors(mut self, m: usize) -> Self
    pub fn with_kind(mut self, kind: u8) -> Self
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
  }
  impl Resampler for BorderlineSMOTE { fn fit_resample(...), fn strategy_description(...) }
  impl Default for ADASYN { fn default(...) }
  impl ADASYN {
    pub fn new() -> Self
    pub fn with_k_neighbors(mut self, k: usize) -> Self
    pub fn with_n_neighbors(mut self, n: usize) -> Self
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_imbalance_threshold(mut self, threshold: f64) -> Self
    pub fn n_synthetic_samples(&self) -> Option<&HashMap<i64, usize>>
    pub fn density_ratios(&self) -> Option<&HashMap<i64, Vec<f64>>>
  }
  impl Resampler for ADASYN { fn fit_resample(...), fn strategy_description(...) }
  impl Default for RandomUnderSampler { fn default(...) }
  impl RandomUnderSampler {
    pub fn new() -> Self
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_replacement(mut self, replacement: bool) -> Self
    pub fn sample_indices(&self) -> Option<&Vec<usize>>
    pub fn n_samples_removed(&self) -> Option<&HashMap<i64, usize>>
  }
  impl Resampler for RandomUnderSampler { fn fit_resample(...), fn strategy_description(...) }
  impl Default for RandomOverSampler { fn default(...) }
  impl RandomOverSampler {
    pub fn new() -> Self
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_shrinkage(mut self, shrinkage: f64) -> Self
    pub fn n_samples_added(&self) -> Option<&HashMap<i64, usize>>
    pub fn sample_indices(&self) -> Option<&HashMap<i64, Vec<usize>>>
  }
  impl Resampler for RandomOverSampler { fn fit_resample(...), fn strategy_description(...) }
  impl Default for ENNKind { fn default(...) }
  impl Default for SMOTETomek { fn default(...) }
  impl SMOTETomek {
    pub fn new() -> Self
    pub fn with_k_neighbors(mut self, k: usize) -> Self
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn n_tomek_links(&self) -> Option<usize>
    pub fn n_samples_removed(&self) -> Option<usize>
    pub fn n_synthetic_samples(&self) -> Option<&HashMap<i64, usize>>
  }
  impl Resampler for SMOTETomek { fn fit_resample(...), fn strategy_description(...) }
  impl Default for SMOTEENN { fn default(...) }
  impl SMOTEENN {
    pub fn new() -> Self
    pub fn with_k_neighbors(mut self, k: usize) -> Self
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self
    pub fn with_random_state(mut self, seed: u64) -> Self
    pub fn with_enn_n_neighbors(mut self, n: usize) -> Self
    pub fn with_enn_kind(mut self, kind: ENNKind) -> Self
    pub fn n_samples_removed(&self) -> Option<usize>
    pub fn n_synthetic_samples(&self) -> Option<&HashMap<i64, usize>>
  }
  impl Resampler for SMOTEENN { fn fit_resample(...), fn strategy_description(...) }

ferroml-core/src/preprocessing/scalers.rs
  pub struct StandardScaler { ... }
  pub struct MinMaxScaler { ... }
  pub struct RobustScaler { ... }
  pub struct MaxAbsScaler { ... }
  pub struct Normalizer { norm: NormType }
  pub struct Binarizer { threshold: f64 }
  pub enum NormType { L1, L2, Max }
  impl StandardScaler {
    pub fn new() -> Self
    pub fn with_mean(mut self, center: bool) -> Self
    pub fn with_std(mut self, scale: bool) -> Self
    pub fn mean(&self) -> Option<&Array1<f64>>
    pub fn std(&self) -> Option<&Array1<f64>>
    pub fn constant_features(&self) -> &[usize]
  }
  impl Default for StandardScaler { fn default(...) }
  impl Transformer for StandardScaler { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl MinMaxScaler {
    pub fn new() -> Self
    pub fn with_range(mut self, min: f64, max: f64) -> Self
    pub fn data_min(&self) -> Option<&Array1<f64>>
    pub fn data_max(&self) -> Option<&Array1<f64>>
    pub fn data_range(&self) -> Option<&Array1<f64>>
    pub fn feature_range(&self) -> (f64, f64)
  }
  impl Default for MinMaxScaler { fn default(...) }
  impl Transformer for MinMaxScaler { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl RobustScaler {
    pub fn new() -> Self
    pub fn with_centering(mut self, center: bool) -> Self
    pub fn with_scaling(mut self, scale: bool) -> Self
    pub fn with_quantile_range(mut self, q_min: f64, q_max: f64) -> Self
    pub fn center(&self) -> Option<&Array1<f64>>
    pub fn scale(&self) -> Option<&Array1<f64>>
  }
  impl Default for RobustScaler { fn default(...) }
  impl Transformer for RobustScaler { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl MaxAbsScaler {
    pub fn new() -> Self
    pub fn max_abs(&self) -> Option<&Array1<f64>>
  }
  impl Default for MaxAbsScaler { fn default(...) }
  impl Transformer for MaxAbsScaler { fn fit(...), fn transform(...), fn inverse_transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl PipelineTransformer for StandardScaler { fn clone_boxed(...), fn name(...) }
  impl StandardScaler {
    pub fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix) -> Result<()>
    pub fn transform_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array2<f64>>
  }
  impl PipelineTransformer for MinMaxScaler { fn clone_boxed(...), fn name(...) }
  impl PipelineTransformer for RobustScaler { fn clone_boxed(...), fn name(...) }
  impl PipelineTransformer for MaxAbsScaler { fn clone_boxed(...), fn name(...) }
  impl MaxAbsScaler {
    pub fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix) -> Result<()>
    pub fn transform_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array2<f64>>
  }
  impl Default for NormType { fn default(...) }
  impl Default for Normalizer { fn default(...) }
  impl Normalizer {
    pub fn new() -> Self
    pub fn with_norm(mut self, norm: NormType) -> Self
  }
  impl Transformer for Normalizer { fn fit(...), fn transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl PipelineTransformer for Normalizer { fn clone_boxed(...), fn name(...) }
  impl Normalizer {
    pub fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix) -> Result<()>
    pub fn transform_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array2<f64>>
  }
  impl Default for Binarizer { fn default(...) }
  impl Binarizer {
    pub fn new() -> Self
    pub fn with_threshold(mut self, threshold: f64) -> Self
  }
  impl Transformer for Binarizer { fn fit(...), fn transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl PipelineTransformer for Binarizer { fn clone_boxed(...), fn name(...) }

ferroml-core/src/preprocessing/selection.rs
  pub struct VarianceThreshold { ... }
  pub struct FeatureScores { scores: Array1<f64>, p_values: Option<Array1<f64>> }
  pub struct SelectKBest { ... }
  pub struct SelectFromModel { ... }
  pub struct RecursiveFeatureElimination { ... }
  pub enum ScoreFunction { FClassif, FRegression, Chi2 }
  pub enum ImportanceThreshold { Mean, Median, Value, MeanPlusStd }
  pub trait FeatureImportanceEstimator: Send + Sync {
    fn fit_and_get_importances(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>>
  }
  impl VarianceThreshold {
    pub fn new(threshold: f64) -> Self
    pub fn variances(&self) -> Option<&Array1<f64>>
    pub fn selected_indices(&self) -> Option<&[usize]>
    pub fn get_support(&self) -> Option<Vec<bool>>
  }
  impl Default for VarianceThreshold { fn default(...) }
  impl Transformer for VarianceThreshold { fn fit(...), fn transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl Default for ScoreFunction { fn default(...) }
  impl SelectKBest {
    pub fn new(score_func: ScoreFunction, k: usize) -> Self
    pub fn fit_with_target(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>
    pub fn scores(&self) -> Option<&FeatureScores>
    pub fn selected_indices(&self) -> Option<&[usize]>
    pub fn get_support(&self) -> Option<Vec<bool>>
    pub fn p_values(&self) -> Option<&Array1<f64>>
  }
  impl Transformer for SelectKBest { fn fit(...), fn transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl Default for ImportanceThreshold { fn default(...) }
  impl SelectFromModel {
    pub fn new(importances: Array1<f64>, threshold: ImportanceThreshold) -> Self
    pub fn with_max_features(mut self, max_features: usize) -> Self
    pub fn threshold_value(&self) -> Option<f64>
    pub fn importances(&self) -> &Array1<f64>
    pub fn selected_indices(&self) -> Option<&[usize]>
    pub fn get_support(&self) -> Option<Vec<bool>>
  }
  impl Transformer for SelectFromModel { fn fit(...), fn transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }
  impl Clone for RecursiveFeatureElimination { fn clone(...) }
  impl RecursiveFeatureElimination {
    pub fn new(estimator: Box<dyn FeatureImportanceEstimator>) -> Self
    pub fn with_n_features_to_select(mut self, n: usize) -> Self
    pub fn with_step(mut self, step: usize) -> Self
    pub fn fit_with_target(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>
    pub fn ranking(&self) -> Option<&Array1<usize>>
    pub fn get_support(&self) -> Option<&[bool]>
    pub fn selected_indices(&self) -> Option<&[usize]>
    pub fn n_iterations(&self) -> Option<usize>
    pub fn set_estimator(&mut self, estimator: Box<dyn FeatureImportanceEstimator>)
  }
  impl Transformer for RecursiveFeatureElimination { fn fit(...), fn transform(...), fn is_fitted(...), fn get_feature_names_out(...), fn n_features_in(...), fn n_features_out(...) }

ferroml-core/src/preprocessing/tfidf.rs
  pub struct TfidfTransformer { ... }
  pub enum TfidfNorm { L1, L2, None }
  impl Default for TfidfTransformer { fn default(...) }
  impl TfidfTransformer {
    pub fn new() -> Self
    pub fn with_norm(mut self, norm: TfidfNorm) -> Self
    pub fn with_use_idf(mut self, use_idf: bool) -> Self
    pub fn with_smooth_idf(mut self, smooth_idf: bool) -> Self
    pub fn with_sublinear_tf(mut self, sublinear_tf: bool) -> Self
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<()>
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>>
    pub fn idf(&self) -> Option<&Array1<f64>>
    pub fn is_fitted(&self) -> bool
    pub fn norm(&self) -> TfidfNorm
    pub fn use_idf(&self) -> bool
    pub fn smooth_idf(&self) -> bool
    pub fn sublinear_tf(&self) -> bool
    pub fn n_features(&self) -> Option<usize>
  }
  impl TfidfTransformer {
    pub fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix) -> Result<()>
    pub fn transform_sparse_native(&self, x: &crate::sparse::CsrMatrix,) -> Result<crate::sparse::CsrMatrix>
    pub fn transform_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array2<f64>>
  }

ferroml-core/src/preprocessing/tfidf_vectorizer.rs
  pub struct TfidfVectorizer { ... }
  impl Default for TfidfVectorizer { fn default(...) }
  impl TfidfVectorizer {
    pub fn new() -> Self
    pub fn with_max_features(mut self, n: usize) -> Self
    pub fn with_ngram_range(mut self, range: (usize, usize)
    pub fn with_min_df(mut self, min_df: DocFrequency) -> Self
    pub fn with_max_df(mut self, max_df: DocFrequency) -> Self
    pub fn with_binary(mut self, binary: bool) -> Self
    pub fn with_stop_words(mut self, sw: Vec<String>) -> Self
    pub fn with_lowercase(mut self, lowercase: bool) -> Self
    pub fn with_norm(mut self, norm: TfidfNorm) -> Self
    pub fn with_use_idf(mut self, use_idf: bool) -> Self
    pub fn with_smooth_idf(mut self, smooth: bool) -> Self
    pub fn with_sublinear_tf(mut self, sub: bool) -> Self
    pub fn vocabulary(&self) -> Option<&HashMap<String, usize>>
    pub fn get_feature_names(&self) -> Option<&[String]>
    pub fn idf(&self) -> Option<&Array1<f64>>
    pub fn is_fitted(&self) -> bool
    pub fn fit_transform_text_dense(&mut self, documents: &[String]) -> Result<Array2<f64>>
    pub fn transform_text_dense(&self, documents: &[String]) -> Result<Array2<f64>>
  }
  impl TextTransformer for TfidfVectorizer { fn fit_text(...), fn transform_text(...) }

ferroml-core/src/stats/bootstrap.rs
  pub struct Bootstrap { n_bootstrap: usize, seed: Option<u64>, confidence: f64 }
  pub struct BootstrapResult { original: f64, std_error: f64, bias: f64, ci_percentile: (f64, ci_bca: Option<(f64, samples: Vec<f64> }
  impl Default for Bootstrap { fn default(...) }
  impl Bootstrap {
    pub fn new(n_bootstrap: usize) -> Self
    pub fn with_seed(mut self, seed: u64) -> Self
    pub fn with_confidence(mut self, confidence: f64) -> Self
    pub fn mean(&self, data: &Array1<f64>) -> Result<BootstrapResult>
    pub fn median(&self, data: &Array1<f64>) -> Result<BootstrapResult>
    pub fn std(&self, data: &Array1<f64>) -> Result<BootstrapResult>
    pub fn correlation(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<BootstrapResult>
  }

ferroml-core/src/stats/confidence.rs
  pub struct ConfidenceInterval { lower: f64, upper: f64, estimate: f64, level: f64, method: CIMethod }
  pub enum CIMethod { Normal, TDistribution, BootstrapPercentile, BootstrapBCa, WilsonScore, ClopperPearson, BayesianCredible }
  impl ConfidenceInterval {
    pub fn width(&self) -> f64
    pub fn contains(&self, value: f64) -> bool
    pub fn margin_of_error(&self) -> f64
  }
  pub fn confidence_interval(data: &Array1<f64>, level: f64, method: CIMethod,) -> Result<ConfidenceInterval>

ferroml-core/src/stats/diagnostics.rs
  pub struct ResidualDiagnostics { normality: NormalityTestResult, homoscedasticity: HomoscedasticityResult, autocorrelation: AutocorrelationResult, outliers: OutlierResult }
  pub struct NormalityTestResult { test: String, statistic: f64, p_value: f64, is_normal: bool, skewness: f64, kurtosis: f64 }
  pub struct HomoscedasticityResult { test: String, statistic: f64, p_value: f64, is_homoscedastic: bool }
  pub struct AutocorrelationResult { test: String, dw_statistic: f64, interpretation: String, lag1_autocorr: f64 }
  pub struct OutlierResult { n_outliers: usize, outlier_indices: Vec<usize>, method: String, threshold: f64 }
  pub trait NormalityTest {
    fn test(&self, data: &Array1<f64>) -> Result<NormalityTestResult>
  }
  impl NormalityTest for ShapiroWilkTest { fn test(...) }
  pub fn durbin_watson(residuals: &Array1<f64>) -> f64
  pub fn detect_outliers_iqr(data: &Array1<f64>, k: f64) -> OutlierResult
  pub fn diagnose_residuals(residuals: &Array1<f64>) -> Result<ResidualDiagnostics>

ferroml-core/src/stats/effect_size.rs
  pub struct EffectSizeValue { value: f64, ci: Option<(f64, variance: Option<f64>, interpretation: String }
  pub struct CohensD { ... }
  pub struct HedgesG { ... }
  pub struct GlasssDelta { ... }
  pub trait EffectSize {
    fn compute(&self) -> Result<EffectSizeValue>
    fn name(&self) -> &str
    fn interpret(value: f64) -> &'static str
  }
  impl CohensD {
    pub fn new(group1: Array1<f64>, group2: Array1<f64>) -> Self
  }
  impl EffectSize for CohensD { fn compute(...), fn name(...), fn interpret(...) }
  impl HedgesG {
    pub fn new(group1: Array1<f64>, group2: Array1<f64>) -> Self
  }
  impl EffectSize for HedgesG { fn compute(...), fn name(...), fn interpret(...) }
  impl GlasssDelta {
    pub fn new(treatment: Array1<f64>, control: Array1<f64>) -> Self
  }
  impl EffectSize for GlasssDelta { fn compute(...), fn name(...), fn interpret(...) }
  pub fn r_squared(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64
  pub fn adjusted_r_squared(r2: f64, n: usize, p: usize) -> f64
  pub fn eta_squared(ss_between: f64, ss_total: f64) -> f64
  pub fn partial_eta_squared(ss_effect: f64, ss_error: f64) -> f64
  pub fn omega_squared(ss_between: f64, ss_total: f64, ms_within: f64, k: usize) -> f64

ferroml-core/src/stats/hypothesis.rs
  pub struct TestResult { statistic: f64, p_value: f64, df: Option<f64> }
  pub enum TwoSampleTest { TTest, MannWhitney, Welch }
  pub trait HypothesisTest {
    fn test(&self) -> Result<StatisticalResult>
    fn check_assumptions(&self) -> Vec<AssumptionTest>
    fn name(&self) -> &str
  }
  impl TwoSampleTest {
    pub fn t_test(x: Array1<f64>, y: Array1<f64>, equal_var: bool) -> Self
    pub fn welch(x: Array1<f64>, y: Array1<f64>) -> Self
    pub fn mann_whitney(x: Array1<f64>, y: Array1<f64>) -> Self
  }
  impl HypothesisTest for TwoSampleTest { fn test(...), fn check_assumptions(...), fn name(...) }

ferroml-core/src/stats/math.rs
  pub fn gamma_ln(x: f64) -> f64
  pub fn beta(a: f64, b: f64) -> f64
  pub fn ln_beta(a: f64, b: f64) -> f64
  pub fn incomplete_beta(a: f64, b: f64, x: f64) -> f64
  pub fn t_cdf(t: f64, df: f64) -> f64
  pub fn t_pdf(t: f64, df: f64) -> f64
  pub fn t_critical(p: f64, df: f64) -> f64
  pub fn normal_cdf(x: f64) -> f64
  pub fn z_critical(p: f64) -> f64
  pub fn erf(x: f64) -> f64
  pub fn chi2_cdf(x: f64, df: f64) -> f64
  pub fn incomplete_gamma(a: f64, x: f64) -> f64
  pub fn percentile_ci(scores: &[f64], confidence: f64) -> (f64, f64)
  pub fn bootstrap_std_error(scores: &[f64]) -> f64
  pub fn percentile(sorted: &[f64], p: f64) -> f64
  pub fn norm_ppf(p: f64) -> f64
  pub fn pearson_r(x: &ndarray::Array1<f64>, y: &ndarray::Array1<f64>) -> f64

ferroml-core/src/stats/mod.rs
  pub struct StatisticalResult { statistic: f64, p_value: f64, effect_size: Option<EffectSizeResult>, confidence_interval: Option<(f64, confidence_level: f64, df: Option<f64>, n: usize, power: Option<f64>, ... }
  pub struct EffectSizeResult { name: String, value: f64, ci: Option<(f64, interpretation: String }
  pub struct AssumptionTest { assumption: String, test_name: String, passed: bool, p_value: f64, details: String }
  pub struct DescriptiveStats { n: usize, mean: f64, std: f64, sem: f64, min: f64, max: f64, median: f64, q1: f64, ... }
  pub struct CorrelationResult { r: f64, p_value: f64, ci: (f64, confidence_level: f64, n: usize, r_squared: f64 }
  impl StatisticalResult {
    pub fn is_significant(&self, alpha: f64) -> bool
    pub fn practical_significance(&self) -> Option<&str>
    pub fn summary(&self) -> String
  }
  impl DescriptiveStats {
    pub fn compute(data: &Array1<f64>) -> Result<Self>
  }
  pub fn correlation(x: &Array1<f64>, y: &Array1<f64>, confidence: f64) -> Result<CorrelationResult>

ferroml-core/src/stats/multiple_testing.rs
  pub struct CorrectedPValues { original: Vec<f64>, adjusted: Vec<f64>, rejected: Vec<bool>, alpha: f64, method: MultipleTestingCorrection }
  pub enum MultipleTestingCorrection { None, Bonferroni, Holm, Hochberg, BenjaminiHochberg, BenjaminiYekutieli }
  pub fn adjust_pvalues(p_values: &[f64], method: MultipleTestingCorrection, alpha: f64,) -> CorrectedPValues
  pub fn count_rejections(adjusted: &[f64], alpha: f64) -> usize
  pub fn estimate_fdp(adjusted: &[f64], alpha: f64) -> f64

ferroml-core/src/stats/power.rs
  pub struct PowerAnalysis { power: f64, n: usize, effect_size: f64, alpha: f64 }
  pub fn sample_size_for_power(effect_size: f64, alpha: f64, power: f64, _test_type: &str) -> usize
  pub fn power_for_sample_size(n: usize, effect_size: f64, alpha: f64) -> f64
```

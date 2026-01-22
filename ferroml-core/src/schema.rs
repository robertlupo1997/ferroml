//! Feature Schema Validation
//!
//! This module provides feature schema validation for ensuring input data
//! matches the expected format before making predictions. This is critical
//! for production deployments where data quality issues can cause subtle bugs.
//!
//! ## Features
//!
//! - **Type checking**: Validate that features are numeric
//! - **Shape validation**: Ensure correct number of features
//! - **Missing value handling**: Detect and optionally reject missing values
//! - **Range validation**: Check values against expected bounds
//! - **Feature names**: Track and validate feature names
//!
//! ## Example
//!
//! ```ignore
//! use ferroml_core::schema::{FeatureSchema, FeatureSpec, ValidationMode};
//! use ndarray::Array2;
//!
//! // Create a schema during training
//! let schema = FeatureSchema::from_array(&x_train)
//!     .with_feature_names(vec!["age", "income", "score"])
//!     .with_mode(ValidationMode::Strict);
//!
//! // Validate new data before prediction
//! schema.validate(&x_test)?;
//! ```

use crate::{FerroError, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Validation mode controlling strictness of schema checks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ValidationMode {
    /// Strict mode: all validations must pass
    Strict,
    /// Warn mode: log warnings but don't fail (default)
    #[default]
    Warn,
    /// Permissive mode: only check critical issues (shape mismatch)
    Permissive,
}

impl fmt::Display for ValidationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationMode::Strict => write!(f, "strict"),
            ValidationMode::Warn => write!(f, "warn"),
            ValidationMode::Permissive => write!(f, "permissive"),
        }
    }
}

/// Data type for a feature
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum FeatureType {
    /// Continuous numeric feature (f64)
    #[default]
    Continuous,
    /// Integer feature (discrete but ordered)
    Integer,
    /// Categorical feature (encoded as numeric)
    Categorical,
    /// Binary feature (0 or 1)
    Binary,
    /// Unknown/unspecified type
    Unknown,
}

impl fmt::Display for FeatureType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeatureType::Continuous => write!(f, "continuous"),
            FeatureType::Integer => write!(f, "integer"),
            FeatureType::Categorical => write!(f, "categorical"),
            FeatureType::Binary => write!(f, "binary"),
            FeatureType::Unknown => write!(f, "unknown"),
        }
    }
}

/// Specification for a single feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSpec {
    /// Feature name (optional)
    pub name: Option<String>,
    /// Feature type
    pub feature_type: FeatureType,
    /// Whether missing values (NaN/inf) are allowed
    pub allow_missing: bool,
    /// Minimum expected value (inclusive)
    pub min_value: Option<f64>,
    /// Maximum expected value (inclusive)
    pub max_value: Option<f64>,
    /// Expected unique values for categorical features
    pub categories: Option<Vec<f64>>,
}

impl Default for FeatureSpec {
    fn default() -> Self {
        Self {
            name: None,
            feature_type: FeatureType::Continuous,
            allow_missing: false,
            min_value: None,
            max_value: None,
            categories: None,
        }
    }
}

impl FeatureSpec {
    /// Create a new continuous feature spec
    #[must_use]
    pub fn continuous() -> Self {
        Self {
            feature_type: FeatureType::Continuous,
            ..Default::default()
        }
    }

    /// Create a new integer feature spec
    #[must_use]
    pub fn integer() -> Self {
        Self {
            feature_type: FeatureType::Integer,
            ..Default::default()
        }
    }

    /// Create a new categorical feature spec
    #[must_use]
    pub fn categorical(categories: Vec<f64>) -> Self {
        Self {
            feature_type: FeatureType::Categorical,
            categories: Some(categories),
            ..Default::default()
        }
    }

    /// Create a new binary feature spec
    #[must_use]
    pub fn binary() -> Self {
        Self {
            feature_type: FeatureType::Binary,
            min_value: Some(0.0),
            max_value: Some(1.0),
            ..Default::default()
        }
    }

    /// Set the feature name
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Allow missing values
    #[must_use]
    pub fn allow_missing(mut self) -> Self {
        self.allow_missing = true;
        self
    }

    /// Set minimum value constraint
    #[must_use]
    pub fn with_min(mut self, min: f64) -> Self {
        self.min_value = Some(min);
        self
    }

    /// Set maximum value constraint
    #[must_use]
    pub fn with_max(mut self, max: f64) -> Self {
        self.max_value = Some(max);
        self
    }

    /// Set both min and max value constraints
    #[must_use]
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    /// Infer feature spec from a column of data
    #[must_use]
    pub fn from_column(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut has_missing = false;
        let mut is_integer = true;
        let mut unique_values: Vec<f64> = Vec::new();

        for &v in values {
            if v.is_nan() || v.is_infinite() {
                has_missing = true;
                continue;
            }

            min = min.min(v);
            max = max.max(v);

            if v.fract() != 0.0 {
                is_integer = false;
            }

            // Track unique values (limit to reasonable number)
            if unique_values.len() < 100 {
                if !unique_values.iter().any(|&u| (u - v).abs() < 1e-10) {
                    unique_values.push(v);
                }
            }
        }

        // Determine feature type
        let feature_type =
            if unique_values.len() == 2 && unique_values.iter().all(|&v| v == 0.0 || v == 1.0) {
                FeatureType::Binary
            } else if is_integer && unique_values.len() <= 20 {
                FeatureType::Categorical
            } else if is_integer {
                FeatureType::Integer
            } else {
                FeatureType::Continuous
            };

        let categories = if feature_type == FeatureType::Categorical {
            let mut cats = unique_values.clone();
            cats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Some(cats)
        } else {
            None
        };

        Self {
            name: None,
            feature_type,
            allow_missing: has_missing,
            min_value: if min.is_finite() { Some(min) } else { None },
            max_value: if max.is_finite() { Some(max) } else { None },
            categories,
        }
    }

    /// Validate a single value against this spec
    pub fn validate_value(&self, value: f64, feature_idx: usize) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let feature_name = self
            .name
            .clone()
            .unwrap_or_else(|| format!("feature_{}", feature_idx));

        // Check for missing values
        if value.is_nan() || value.is_infinite() {
            if !self.allow_missing {
                issues.push(ValidationIssue::MissingValue {
                    feature: feature_name.clone(),
                    feature_idx,
                });
            }
            return issues; // Can't do other checks on NaN/inf
        }

        // Check range
        if let Some(min) = self.min_value {
            if value < min {
                issues.push(ValidationIssue::ValueBelowMin {
                    feature: feature_name.clone(),
                    feature_idx,
                    value,
                    min,
                });
            }
        }
        if let Some(max) = self.max_value {
            if value > max {
                issues.push(ValidationIssue::ValueAboveMax {
                    feature: feature_name.clone(),
                    feature_idx,
                    value,
                    max,
                });
            }
        }

        // Check categorical values
        if let Some(ref categories) = self.categories {
            if !categories.iter().any(|&c| (c - value).abs() < 1e-10) {
                issues.push(ValidationIssue::UnknownCategory {
                    feature: feature_name.clone(),
                    feature_idx,
                    value,
                    expected: categories.clone(),
                });
            }
        }

        // Check binary values
        if self.feature_type == FeatureType::Binary && value != 0.0 && value != 1.0 {
            issues.push(ValidationIssue::InvalidBinary {
                feature: feature_name.clone(),
                feature_idx,
                value,
            });
        }

        // Check integer values
        if self.feature_type == FeatureType::Integer && value.fract() != 0.0 {
            issues.push(ValidationIssue::NonInteger {
                feature: feature_name,
                feature_idx,
                value,
            });
        }

        issues
    }
}

/// A validation issue found during schema validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationIssue {
    /// Shape mismatch (critical)
    ShapeMismatch {
        /// Expected number of features
        expected: usize,
        /// Actual number of features
        actual: usize,
    },
    /// Missing value detected
    MissingValue {
        /// Feature name
        feature: String,
        /// Feature index
        feature_idx: usize,
    },
    /// Value below minimum
    ValueBelowMin {
        /// Feature name
        feature: String,
        /// Feature index
        feature_idx: usize,
        /// Actual value
        value: f64,
        /// Expected minimum
        min: f64,
    },
    /// Value above maximum
    ValueAboveMax {
        /// Feature name
        feature: String,
        /// Feature index
        feature_idx: usize,
        /// Actual value
        value: f64,
        /// Expected maximum
        max: f64,
    },
    /// Unknown category value
    UnknownCategory {
        /// Feature name
        feature: String,
        /// Feature index
        feature_idx: usize,
        /// Actual value
        value: f64,
        /// Expected categories
        expected: Vec<f64>,
    },
    /// Non-binary value in binary feature
    InvalidBinary {
        /// Feature name
        feature: String,
        /// Feature index
        feature_idx: usize,
        /// Actual value
        value: f64,
    },
    /// Non-integer value in integer feature
    NonInteger {
        /// Feature name
        feature: String,
        /// Feature index
        feature_idx: usize,
        /// Actual value
        value: f64,
    },
    /// Feature name mismatch
    FeatureNameMismatch {
        /// Feature index
        feature_idx: usize,
        /// Expected name
        expected: String,
        /// Actual name
        actual: String,
    },
    /// Empty input data
    EmptyInput,
}

impl ValidationIssue {
    /// Check if this issue is critical (should always fail validation)
    #[must_use]
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            ValidationIssue::ShapeMismatch { .. } | ValidationIssue::EmptyInput
        )
    }

    /// Get the severity level of this issue
    #[must_use]
    pub fn severity(&self) -> IssueSeverity {
        match self {
            ValidationIssue::ShapeMismatch { .. } | ValidationIssue::EmptyInput => {
                IssueSeverity::Critical
            }
            ValidationIssue::MissingValue { .. } | ValidationIssue::UnknownCategory { .. } => {
                IssueSeverity::Error
            }
            ValidationIssue::ValueBelowMin { .. }
            | ValidationIssue::ValueAboveMax { .. }
            | ValidationIssue::InvalidBinary { .. }
            | ValidationIssue::NonInteger { .. }
            | ValidationIssue::FeatureNameMismatch { .. } => IssueSeverity::Warning,
        }
    }
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationIssue::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Shape mismatch: expected {} features, got {}",
                    expected, actual
                )
            }
            ValidationIssue::MissingValue {
                feature,
                feature_idx,
            } => {
                write!(
                    f,
                    "Missing value (NaN/inf) in feature '{}' (index {})",
                    feature, feature_idx
                )
            }
            ValidationIssue::ValueBelowMin {
                feature,
                feature_idx,
                value,
                min,
            } => {
                write!(
                    f,
                    "Value {} in feature '{}' (index {}) is below minimum {}",
                    value, feature, feature_idx, min
                )
            }
            ValidationIssue::ValueAboveMax {
                feature,
                feature_idx,
                value,
                max,
            } => {
                write!(
                    f,
                    "Value {} in feature '{}' (index {}) is above maximum {}",
                    value, feature, feature_idx, max
                )
            }
            ValidationIssue::UnknownCategory {
                feature,
                feature_idx,
                value,
                expected,
            } => {
                write!(
                    f,
                    "Unknown category {} in feature '{}' (index {}), expected one of {:?}",
                    value, feature, feature_idx, expected
                )
            }
            ValidationIssue::InvalidBinary {
                feature,
                feature_idx,
                value,
            } => {
                write!(
                    f,
                    "Invalid binary value {} in feature '{}' (index {}), expected 0 or 1",
                    value, feature, feature_idx
                )
            }
            ValidationIssue::NonInteger {
                feature,
                feature_idx,
                value,
            } => {
                write!(
                    f,
                    "Non-integer value {} in feature '{}' (index {})",
                    value, feature, feature_idx
                )
            }
            ValidationIssue::FeatureNameMismatch {
                feature_idx,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Feature name mismatch at index {}: expected '{}', got '{}'",
                    feature_idx, expected, actual
                )
            }
            ValidationIssue::EmptyInput => {
                write!(f, "Input data is empty")
            }
        }
    }
}

/// Severity level of a validation issue
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Critical issue that always causes validation failure
    Critical,
    /// Error that causes failure in strict mode
    Error,
    /// Warning that may be acceptable in some modes
    Warning,
}

/// Result of schema validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub passed: bool,
    /// Issues found during validation
    pub issues: Vec<ValidationIssue>,
    /// Number of samples validated
    pub n_samples: usize,
    /// Number of features validated
    pub n_features: usize,
    /// Validation mode used
    pub mode: ValidationMode,
}

impl ValidationResult {
    /// Create a successful validation result
    #[must_use]
    pub fn success(n_samples: usize, n_features: usize, mode: ValidationMode) -> Self {
        Self {
            passed: true,
            issues: Vec::new(),
            n_samples,
            n_features,
            mode,
        }
    }

    /// Create a failed validation result
    #[must_use]
    pub fn failure(
        issues: Vec<ValidationIssue>,
        n_samples: usize,
        n_features: usize,
        mode: ValidationMode,
    ) -> Self {
        Self {
            passed: false,
            issues,
            n_samples,
            n_features,
            mode,
        }
    }

    /// Get issues by severity
    #[must_use]
    pub fn issues_by_severity(&self, severity: IssueSeverity) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity() == severity)
            .collect()
    }

    /// Get all critical issues
    #[must_use]
    pub fn critical_issues(&self) -> Vec<&ValidationIssue> {
        self.issues_by_severity(IssueSeverity::Critical)
    }

    /// Get all error-level issues
    #[must_use]
    pub fn error_issues(&self) -> Vec<&ValidationIssue> {
        self.issues_by_severity(IssueSeverity::Error)
    }

    /// Get all warnings
    #[must_use]
    pub fn warnings(&self) -> Vec<&ValidationIssue> {
        self.issues_by_severity(IssueSeverity::Warning)
    }

    /// Count issues by type
    #[must_use]
    pub fn issue_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for issue in &self.issues {
            let key = match issue {
                ValidationIssue::ShapeMismatch { .. } => "shape_mismatch",
                ValidationIssue::MissingValue { .. } => "missing_value",
                ValidationIssue::ValueBelowMin { .. } => "below_min",
                ValidationIssue::ValueAboveMax { .. } => "above_max",
                ValidationIssue::UnknownCategory { .. } => "unknown_category",
                ValidationIssue::InvalidBinary { .. } => "invalid_binary",
                ValidationIssue::NonInteger { .. } => "non_integer",
                ValidationIssue::FeatureNameMismatch { .. } => "name_mismatch",
                ValidationIssue::EmptyInput => "empty_input",
            };
            *counts.entry(key.to_string()).or_insert(0) += 1;
        }
        counts
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.passed {
            writeln!(f, "Validation PASSED")?;
        } else {
            writeln!(f, "Validation FAILED")?;
        }
        writeln!(
            f,
            "  Samples: {}, Features: {}, Mode: {}",
            self.n_samples, self.n_features, self.mode
        )?;
        if !self.issues.is_empty() {
            writeln!(f, "  Issues ({}):", self.issues.len())?;
            for issue in &self.issues {
                writeln!(f, "    - {}", issue)?;
            }
        }
        Ok(())
    }
}

/// Feature schema defining expected input structure
///
/// A schema captures the expected structure of input data including:
/// - Number of features
/// - Feature names
/// - Expected value ranges
/// - Missing value policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSchema {
    /// Number of expected features
    pub n_features: usize,
    /// Specifications for each feature
    pub features: Vec<FeatureSpec>,
    /// Validation mode
    pub mode: ValidationMode,
    /// Maximum number of issues to report
    pub max_issues: usize,
    /// Whether to validate every value or sample
    pub sample_validation: bool,
    /// Fraction of samples to validate (if sample_validation is true)
    pub sample_fraction: f64,
}

impl FeatureSchema {
    /// Create a new schema with the specified number of features
    #[must_use]
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            features: (0..n_features).map(|_| FeatureSpec::default()).collect(),
            mode: ValidationMode::Warn,
            max_issues: 100,
            sample_validation: false,
            sample_fraction: 0.1,
        }
    }

    /// Create a schema from a feature matrix, inferring feature specs
    #[must_use]
    pub fn from_array(x: &Array2<f64>) -> Self {
        let n_features = x.ncols();
        let features: Vec<FeatureSpec> = (0..n_features)
            .map(|j| {
                let col: Vec<f64> = x.column(j).to_vec();
                FeatureSpec::from_column(&col)
            })
            .collect();

        Self {
            n_features,
            features,
            mode: ValidationMode::Warn,
            max_issues: 100,
            sample_validation: false,
            sample_fraction: 0.1,
        }
    }

    /// Create a basic schema that only validates shape
    #[must_use]
    pub fn shape_only(n_features: usize) -> Self {
        Self {
            n_features,
            features: Vec::new(),
            mode: ValidationMode::Permissive,
            max_issues: 10,
            sample_validation: false,
            sample_fraction: 1.0,
        }
    }

    /// Set the validation mode
    #[must_use]
    pub fn with_mode(mut self, mode: ValidationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set feature names
    #[must_use]
    pub fn with_feature_names<S: AsRef<str>>(mut self, names: impl IntoIterator<Item = S>) -> Self {
        let names: Vec<String> = names.into_iter().map(|s| s.as_ref().to_string()).collect();
        for (i, name) in names.into_iter().enumerate() {
            if i < self.features.len() {
                self.features[i].name = Some(name);
            }
        }
        self
    }

    /// Set maximum number of issues to report
    #[must_use]
    pub fn with_max_issues(mut self, max_issues: usize) -> Self {
        self.max_issues = max_issues;
        self
    }

    /// Enable sample-based validation (for large datasets)
    #[must_use]
    pub fn with_sample_validation(mut self, fraction: f64) -> Self {
        self.sample_validation = true;
        self.sample_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Allow missing values for all features
    #[must_use]
    pub fn allow_missing(mut self) -> Self {
        for spec in &mut self.features {
            spec.allow_missing = true;
        }
        self
    }

    /// Set a feature spec at the given index
    pub fn set_feature(&mut self, index: usize, spec: FeatureSpec) {
        if index < self.features.len() {
            self.features[index] = spec;
        } else if index < self.n_features {
            // Pad with defaults if needed
            while self.features.len() < index {
                self.features.push(FeatureSpec::default());
            }
            self.features.push(spec);
        }
    }

    /// Get feature names
    #[must_use]
    pub fn feature_names(&self) -> Vec<String> {
        self.features
            .iter()
            .enumerate()
            .map(|(i, spec)| {
                spec.name
                    .clone()
                    .unwrap_or_else(|| format!("feature_{}", i))
            })
            .collect()
    }

    /// Validate a feature matrix against this schema
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    /// ValidationResult with any issues found
    #[must_use]
    pub fn validate(&self, x: &Array2<f64>) -> ValidationResult {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut issues = Vec::new();

        // Check for empty input
        if n_samples == 0 || n_features == 0 {
            return ValidationResult::failure(
                vec![ValidationIssue::EmptyInput],
                n_samples,
                n_features,
                self.mode,
            );
        }

        // Check shape
        if n_features != self.n_features {
            return ValidationResult::failure(
                vec![ValidationIssue::ShapeMismatch {
                    expected: self.n_features,
                    actual: n_features,
                }],
                n_samples,
                n_features,
                self.mode,
            );
        }

        // In permissive mode, only check shape
        if self.mode == ValidationMode::Permissive {
            return ValidationResult::success(n_samples, n_features, self.mode);
        }

        // Validate values
        if !self.features.is_empty() {
            let samples_to_check = if self.sample_validation {
                let n_check = ((n_samples as f64) * self.sample_fraction).ceil() as usize;
                n_check.max(1).min(n_samples)
            } else {
                n_samples
            };

            let step = if samples_to_check < n_samples {
                n_samples / samples_to_check
            } else {
                1
            };

            'outer: for i in (0..n_samples).step_by(step) {
                for (j, spec) in self.features.iter().enumerate() {
                    let value = x[[i, j]];
                    let feature_issues = spec.validate_value(value, j);
                    issues.extend(feature_issues);

                    if issues.len() >= self.max_issues {
                        break 'outer;
                    }
                }
            }
        }

        // Determine if validation passed based on mode
        let passed = match self.mode {
            ValidationMode::Strict => issues.is_empty(),
            ValidationMode::Warn => !issues.iter().any(|i| i.is_critical()),
            ValidationMode::Permissive => true,
        };

        if passed {
            ValidationResult {
                passed: true,
                issues,
                n_samples,
                n_features,
                mode: self.mode,
            }
        } else {
            ValidationResult::failure(issues, n_samples, n_features, self.mode)
        }
    }

    /// Validate and return an error if validation fails
    ///
    /// # Errors
    /// Returns an error if validation fails
    pub fn validate_strict(&self, x: &Array2<f64>) -> Result<()> {
        let result = self.validate(x);
        if !result.passed {
            let error_msg = result
                .issues
                .iter()
                .take(5)
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(FerroError::invalid_input(format!(
                "Schema validation failed: {}",
                error_msg
            )));
        }
        Ok(())
    }

    /// Validate feature names against expected names
    #[must_use]
    pub fn validate_feature_names(&self, names: &[String]) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        // Check count
        if names.len() != self.n_features {
            issues.push(ValidationIssue::ShapeMismatch {
                expected: self.n_features,
                actual: names.len(),
            });
            return issues;
        }

        // Check individual names
        for (i, (spec, actual)) in self.features.iter().zip(names.iter()).enumerate() {
            if let Some(ref expected) = spec.name {
                if expected != actual {
                    issues.push(ValidationIssue::FeatureNameMismatch {
                        feature_idx: i,
                        expected: expected.clone(),
                        actual: actual.clone(),
                    });
                }
            }
        }

        issues
    }

    /// Get a summary of the schema
    #[must_use]
    pub fn summary(&self) -> String {
        let mut s = format!("FeatureSchema ({} features):\n", self.n_features);
        for (i, spec) in self.features.iter().enumerate() {
            let name = spec
                .name
                .clone()
                .unwrap_or_else(|| format!("feature_{}", i));
            s.push_str(&format!(
                "  {}: {} (missing: {}, range: {:?}..{:?})\n",
                name,
                spec.feature_type,
                if spec.allow_missing { "yes" } else { "no" },
                spec.min_value,
                spec.max_value
            ));
        }
        s
    }
}

impl fmt::Display for FeatureSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Trait for models that support schema validation
pub trait SchemaValidated {
    /// Get the feature schema expected by this model
    fn feature_schema(&self) -> Option<&FeatureSchema>;

    /// Validate input against the model's schema
    fn validate_input(&self, x: &Array2<f64>) -> Result<()> {
        if let Some(schema) = self.feature_schema() {
            schema.validate_strict(x)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_feature_spec_continuous() {
        let spec = FeatureSpec::continuous()
            .with_name("age")
            .with_range(0.0, 120.0);

        assert_eq!(spec.name, Some("age".to_string()));
        assert_eq!(spec.feature_type, FeatureType::Continuous);
        assert_eq!(spec.min_value, Some(0.0));
        assert_eq!(spec.max_value, Some(120.0));
    }

    #[test]
    fn test_feature_spec_from_column() {
        // Continuous data
        let continuous = vec![1.5, 2.3, 3.7, 4.1, 5.9];
        let spec = FeatureSpec::from_column(&continuous);
        assert_eq!(spec.feature_type, FeatureType::Continuous);

        // Binary data
        let binary = vec![0.0, 1.0, 0.0, 1.0, 1.0];
        let spec = FeatureSpec::from_column(&binary);
        assert_eq!(spec.feature_type, FeatureType::Binary);

        // Categorical data (integer with few unique values)
        let categorical = vec![1.0, 2.0, 3.0, 1.0, 2.0];
        let spec = FeatureSpec::from_column(&categorical);
        assert_eq!(spec.feature_type, FeatureType::Categorical);
    }

    #[test]
    fn test_feature_spec_validate_value() {
        let spec = FeatureSpec::continuous()
            .with_name("test")
            .with_range(0.0, 100.0);

        // Valid value
        let issues = spec.validate_value(50.0, 0);
        assert!(issues.is_empty());

        // Below minimum
        let issues = spec.validate_value(-5.0, 0);
        assert_eq!(issues.len(), 1);
        assert!(matches!(issues[0], ValidationIssue::ValueBelowMin { .. }));

        // Above maximum
        let issues = spec.validate_value(150.0, 0);
        assert_eq!(issues.len(), 1);
        assert!(matches!(issues[0], ValidationIssue::ValueAboveMax { .. }));

        // Missing value not allowed
        let issues = spec.validate_value(f64::NAN, 0);
        assert_eq!(issues.len(), 1);
        assert!(matches!(issues[0], ValidationIssue::MissingValue { .. }));

        // Missing value allowed
        let spec_allow = spec.clone().allow_missing();
        let issues = spec_allow.validate_value(f64::NAN, 0);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_feature_spec_validate_binary() {
        let spec = FeatureSpec::binary();

        // Valid binary values
        assert!(spec.validate_value(0.0, 0).is_empty());
        assert!(spec.validate_value(1.0, 0).is_empty());

        // Invalid binary value
        let issues = spec.validate_value(0.5, 0);
        assert_eq!(issues.len(), 1);
        assert!(matches!(issues[0], ValidationIssue::InvalidBinary { .. }));
    }

    #[test]
    fn test_feature_spec_validate_categorical() {
        let spec = FeatureSpec::categorical(vec![1.0, 2.0, 3.0]);

        // Valid category
        assert!(spec.validate_value(2.0, 0).is_empty());

        // Unknown category
        let issues = spec.validate_value(4.0, 0);
        assert_eq!(issues.len(), 1);
        assert!(matches!(issues[0], ValidationIssue::UnknownCategory { .. }));
    }

    #[test]
    fn test_schema_from_array() {
        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 0.0, 10.0, 2.0, 1.0, 20.0, 3.0, 0.0, 30.0, 4.0, 1.0, 40.0, 5.0, 1.0, 50.0,
            ],
        )
        .unwrap();

        let schema = FeatureSchema::from_array(&x);
        assert_eq!(schema.n_features, 3);
        assert_eq!(schema.features.len(), 3);
    }

    #[test]
    fn test_schema_validate_shape() {
        let schema = FeatureSchema::new(3);

        // Correct shape
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = schema.validate(&x);
        assert!(result.passed);

        // Wrong shape
        let x_wrong = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = schema.validate(&x_wrong);
        assert!(!result.passed);
        assert!(matches!(
            result.issues[0],
            ValidationIssue::ShapeMismatch {
                expected: 3,
                actual: 2
            }
        ));
    }

    #[test]
    fn test_schema_validate_empty() {
        let schema = FeatureSchema::new(3);

        let x = Array2::from_shape_vec((0, 3), vec![]).unwrap();
        let result = schema.validate(&x);
        assert!(!result.passed);
        assert!(matches!(result.issues[0], ValidationIssue::EmptyInput));
    }

    #[test]
    fn test_schema_validation_modes() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, f64::NAN, 3.0, 4.0]).unwrap();

        // Strict mode - should fail
        let schema_strict = FeatureSchema::from_array(&x).with_mode(ValidationMode::Strict);
        let _result = schema_strict.validate(&x);
        // Note: Schema was created from data with NaN, so allow_missing may be true
        // Let's create fresh specs
        let mut schema_strict = FeatureSchema::new(2).with_mode(ValidationMode::Strict);
        schema_strict.features = vec![FeatureSpec::continuous(), FeatureSpec::continuous()];
        let result = schema_strict.validate(&x);
        assert!(!result.passed);

        // Warn mode - should pass (non-critical issues)
        let mut schema_warn = FeatureSchema::new(2).with_mode(ValidationMode::Warn);
        schema_warn.features = vec![FeatureSpec::continuous(), FeatureSpec::continuous()];
        let result = schema_warn.validate(&x);
        assert!(result.passed);
        assert!(!result.issues.is_empty());

        // Permissive mode - only checks shape
        let mut schema_permissive = FeatureSchema::new(2).with_mode(ValidationMode::Permissive);
        schema_permissive.features = vec![FeatureSpec::continuous(), FeatureSpec::continuous()];
        let result = schema_permissive.validate(&x);
        assert!(result.passed);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_schema_with_feature_names() {
        let schema = FeatureSchema::new(3).with_feature_names(vec!["age", "income", "score"]);

        assert_eq!(schema.feature_names(), vec!["age", "income", "score"]);
    }

    #[test]
    fn test_schema_validate_feature_names() {
        let schema = FeatureSchema::new(3).with_feature_names(vec!["a", "b", "c"]);

        // Matching names
        let issues =
            schema.validate_feature_names(&["a".to_string(), "b".to_string(), "c".to_string()]);
        assert!(issues.is_empty());

        // Mismatched name
        let issues =
            schema.validate_feature_names(&["a".to_string(), "x".to_string(), "c".to_string()]);
        assert_eq!(issues.len(), 1);
        assert!(matches!(
            issues[0],
            ValidationIssue::FeatureNameMismatch { feature_idx: 1, .. }
        ));

        // Wrong count
        let issues = schema.validate_feature_names(&["a".to_string(), "b".to_string()]);
        assert_eq!(issues.len(), 1);
        assert!(matches!(
            issues[0],
            ValidationIssue::ShapeMismatch {
                expected: 3,
                actual: 2
            }
        ));
    }

    #[test]
    fn test_schema_validate_strict() {
        let schema = FeatureSchema::new(2);

        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(schema.validate_strict(&x).is_ok());

        let x_wrong = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(schema.validate_strict(&x_wrong).is_err());
    }

    #[test]
    fn test_validation_result_issue_counts() {
        let issues = vec![
            ValidationIssue::MissingValue {
                feature: "a".to_string(),
                feature_idx: 0,
            },
            ValidationIssue::MissingValue {
                feature: "b".to_string(),
                feature_idx: 1,
            },
            ValidationIssue::ValueBelowMin {
                feature: "c".to_string(),
                feature_idx: 2,
                value: -1.0,
                min: 0.0,
            },
        ];

        let result = ValidationResult::failure(issues, 10, 3, ValidationMode::Strict);
        let counts = result.issue_counts();

        assert_eq!(counts.get("missing_value"), Some(&2));
        assert_eq!(counts.get("below_min"), Some(&1));
    }

    #[test]
    fn test_schema_sample_validation() {
        let x = Array2::from_shape_vec((1000, 2), (0..2000).map(|i| i as f64).collect::<Vec<_>>())
            .unwrap();

        let schema = FeatureSchema::from_array(&x).with_sample_validation(0.1);

        // Should only validate ~10% of samples
        let result = schema.validate(&x);
        assert!(result.passed);
    }

    #[test]
    fn test_validation_issue_display() {
        let issue = ValidationIssue::ShapeMismatch {
            expected: 5,
            actual: 3,
        };
        assert!(issue.to_string().contains("5"));
        assert!(issue.to_string().contains("3"));

        let issue = ValidationIssue::MissingValue {
            feature: "age".to_string(),
            feature_idx: 0,
        };
        assert!(issue.to_string().contains("age"));
    }

    #[test]
    fn test_schema_shape_only() {
        let schema = FeatureSchema::shape_only(5);

        let x = Array2::from_shape_vec((3, 5), vec![f64::NAN; 15]).unwrap();

        // Should only check shape, not values
        let result = schema.validate(&x);
        assert!(result.passed);
    }

    #[test]
    fn test_schema_allow_missing() {
        let mut schema = FeatureSchema::new(2).with_mode(ValidationMode::Strict);
        schema.features = vec![FeatureSpec::continuous(), FeatureSpec::continuous()];

        let x = Array2::from_shape_vec((2, 2), vec![1.0, f64::NAN, 3.0, 4.0]).unwrap();

        // Without allow_missing
        let result = schema.validate(&x);
        assert!(!result.passed);

        // With allow_missing
        let schema_allow = schema.allow_missing();
        let result = schema_allow.validate(&x);
        assert!(result.passed);
    }

    #[test]
    fn test_feature_type_display() {
        assert_eq!(format!("{}", FeatureType::Continuous), "continuous");
        assert_eq!(format!("{}", FeatureType::Binary), "binary");
        assert_eq!(format!("{}", FeatureType::Categorical), "categorical");
    }

    #[test]
    fn test_validation_mode_display() {
        assert_eq!(format!("{}", ValidationMode::Strict), "strict");
        assert_eq!(format!("{}", ValidationMode::Warn), "warn");
        assert_eq!(format!("{}", ValidationMode::Permissive), "permissive");
    }

    #[test]
    fn test_issue_severity() {
        assert!(ValidationIssue::ShapeMismatch {
            expected: 1,
            actual: 2
        }
        .is_critical());
        assert!(ValidationIssue::EmptyInput.is_critical());

        assert!(!ValidationIssue::MissingValue {
            feature: "x".to_string(),
            feature_idx: 0
        }
        .is_critical());

        assert_eq!(
            ValidationIssue::ShapeMismatch {
                expected: 1,
                actual: 2
            }
            .severity(),
            IssueSeverity::Critical
        );
        assert_eq!(
            ValidationIssue::MissingValue {
                feature: "x".to_string(),
                feature_idx: 0
            }
            .severity(),
            IssueSeverity::Error
        );
        assert_eq!(
            ValidationIssue::ValueBelowMin {
                feature: "x".to_string(),
                feature_idx: 0,
                value: -1.0,
                min: 0.0
            }
            .severity(),
            IssueSeverity::Warning
        );
    }

    #[test]
    fn test_schema_summary() {
        let schema = FeatureSchema::new(2).with_feature_names(vec!["feature_a", "feature_b"]);

        let summary = schema.summary();
        assert!(summary.contains("2 features"));
        assert!(summary.contains("feature_a"));
        assert!(summary.contains("feature_b"));
    }

    #[test]
    fn test_validation_result_display() {
        let result = ValidationResult::success(100, 5, ValidationMode::Strict);
        let display = format!("{}", result);
        assert!(display.contains("PASSED"));
        assert!(display.contains("100"));
        assert!(display.contains("5"));

        let result = ValidationResult::failure(
            vec![ValidationIssue::EmptyInput],
            0,
            0,
            ValidationMode::Strict,
        );
        let display = format!("{}", result);
        assert!(display.contains("FAILED"));
    }
}

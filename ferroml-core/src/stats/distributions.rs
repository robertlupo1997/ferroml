//! Statistical distributions for FerroML
//! Placeholder for statrs integration

use serde::{Deserialize, Serialize};

/// Normal distribution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normal {
    /// Mean of the distribution
    pub mean: f64,
    /// Standard deviation of the distribution
    pub std: f64,
}

/// Student's t distribution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentT {
    /// Degrees of freedom
    pub df: f64,
}

/// Chi-squared distribution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChiSquared {
    /// Degrees of freedom
    pub df: f64,
}

/// F distribution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FDistribution {
    /// Numerator degrees of freedom
    pub df1: f64,
    /// Denominator degrees of freedom
    pub df2: f64,
}

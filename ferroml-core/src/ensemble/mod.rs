//! Ensemble Methods
//!
//! This module provides meta-learners for combining multiple models.
//!
//! ## Available Methods
//!
//! - **VotingClassifier**: Combine classifiers via hard/soft voting
//! - **VotingRegressor**: Combine regressors via averaging
//! - **StackingClassifier**: Combine classifiers via stacked generalization with CV
//! - **StackingRegressor**: Combine regressors via stacked generalization with CV
//! - **BaggingClassifier**: Bootstrap aggregating for classifiers
//! - **BaggingRegressor**: Bootstrap aggregating for regressors

pub mod bagging;
pub mod stacking;
pub mod voting;

pub use bagging::{
    BaggingClassifier, BaggingRegressor, MaxFeatures as BaggingMaxFeatures, MaxSamples,
};
pub use stacking::{StackMethod, StackingClassifier, StackingRegressor};
pub use voting::{VotingClassifier, VotingMethod, VotingRegressor};

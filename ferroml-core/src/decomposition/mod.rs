//! Dimensionality Reduction and Matrix Decomposition
//!
//! This module provides algorithms for reducing the dimensionality of data
//! while preserving important information.
//!
//! ## Available Algorithms
//!
//! - [`PCA`] - Principal Component Analysis for linear dimensionality reduction
//! - [`IncrementalPCA`] - Memory-efficient PCA for large datasets
//! - [`TruncatedSVD`] - Truncated SVD for sparse matrices and LSA
//! - [`LDA`] - Linear Discriminant Analysis for supervised dimensionality reduction
//! - [`FactorAnalysis`] - Statistical model relating observed variables to latent factors
//!
//! ## When to Use
//!
//! | Algorithm | Best For | Notes |
//! |-----------|----------|-------|
//! | `PCA` | Dense data, moderate size | Unsupervised, centers data, full SVD |
//! | `IncrementalPCA` | Large dense datasets | Batch processing, approximate |
//! | `TruncatedSVD` | Sparse data, LSA/LSI | No centering, preserves sparsity |
//! | `LDA` | Classification preprocessing | Supervised, maximizes class separation |
//! | `FactorAnalysis` | Latent structure discovery | Statistical model with noise, rotation methods |
//!
//! ## PCA vs LDA
//!
//! **PCA** (unsupervised) finds directions of maximum variance in the data.
//! It ignores class labels and may not preserve class separability.
//!
//! **LDA** (supervised) finds directions that maximize the ratio of between-class
//! to within-class variance. Requires class labels and is limited to at most
//! n_classes - 1 components.
//!
//! ## PCA vs TruncatedSVD
//!
//! **PCA** centers the data before computing SVD, which destroys sparsity. Use for dense data.
//!
//! **TruncatedSVD** does NOT center data, making it suitable for sparse matrices.
//! Also known as LSA (Latent Semantic Analysis) when applied to term-document matrices.
//!
//! ## PCA vs Factor Analysis
//!
//! **PCA** is a variance-maximizing projection that assumes all variance is signal.
//! It's a pure data transformation technique.
//!
//! **Factor Analysis** is a statistical model that explicitly models noise:
//! X = μ + L·F + ε (where ε is unique/noise variance per feature).
//! FA separates common variance from unique variance and supports rotation
//! methods (varimax, promax) for improved interpretability.
//!
//! ## Statistical Rigor
//!
//! All decomposition methods provide:
//! - Explained variance ratios for each component
//! - Cumulative explained variance for component selection
//! - Component loadings for interpretability
//! - Noise variance estimation (where applicable)
//!
//! ## Example
//!
//! ```
//! use ferroml_core::decomposition::{PCA, TruncatedSVD, LDA};
//! use ferroml_core::preprocessing::Transformer;
//! use ndarray::array;
//!
//! // PCA for dense data (centers the data)
//! let mut pca = PCA::new().with_n_components(2);
//! let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
//! let x_pca = pca.fit_transform(&x).unwrap();
//!
//! // TruncatedSVD for sparse-like data (no centering)
//! let mut svd = TruncatedSVD::new().with_n_components(2);
//! let x_sparse = array![[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [2.0, 0.0, 1.0]];
//! let x_svd = svd.fit_transform(&x_sparse).unwrap();
//!
//! // LDA for supervised dimensionality reduction
//! let mut lda = LDA::new();
//! let x_class = array![[1.0, 2.0], [2.0, 3.0], [6.0, 7.0], [7.0, 8.0]];
//! let y = array![0.0, 0.0, 1.0, 1.0];
//! lda.fit(&x_class, &y).unwrap();
//! let x_lda = lda.transform(&x_class).unwrap();
//! ```

mod factor_analysis;
mod lda;
mod pca;
pub mod quadtree;
mod truncated_svd;
mod tsne;
pub mod vptree;

pub use factor_analysis::{FaSvdMethod, FactorAnalysis, Rotation};
pub use lda::{LdaSolver, LDA};
pub use pca::{IncrementalPCA, SvdSolver, PCA};
pub use truncated_svd::{TruncatedSVD, TruncatedSvdAlgorithm};
pub use tsne::{LearningRate, TsneInit, TsneMethod, TsneMetric, TSNE};

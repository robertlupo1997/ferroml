//! Gaussian Mixture Models with Expectation-Maximization
//!
//! Implements GaussianMixture with multiple covariance types, BIC/AIC model
//! selection, and soft clustering via predict_proba.

use crate::clustering::{ClusteringModel, KMeans};
use crate::linalg::{cholesky, logsumexp, solve_lower_triangular};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Covariance type for Gaussian components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CovarianceType {
    /// Each component has its own full covariance matrix.
    Full,
    /// All components share the same full covariance matrix.
    Tied,
    /// Each component has its own diagonal covariance (axis-aligned).
    Diagonal,
    /// Each component has a single variance (isotropic).
    Spherical,
}

/// Initialization method for GMM parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GmmInit {
    /// Initialize means using KMeans clustering.
    KMeans,
    /// Random assignment of samples to components.
    Random,
}

/// Gaussian Mixture Model fitted via Expectation-Maximization.
///
/// # Example
/// ```
/// use ferroml_core::clustering::{GaussianMixture, CovarianceType, ClusteringModel};
/// use ndarray::Array2;
///
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0, 1.5, 1.8, 1.2, 2.2,
///     8.0, 8.0, 8.2, 7.8, 7.8, 8.2,
/// ]).unwrap();
///
/// let mut gmm = GaussianMixture::new(2)
///     .covariance_type(CovarianceType::Full)
///     .random_state(42);
/// gmm.fit(&x).unwrap();
///
/// let proba = gmm.predict_proba(&x).unwrap();
/// assert_eq!(proba.dim(), (6, 2));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianMixture {
    // Configuration
    n_components: usize,
    cov_type: CovarianceType,
    max_iter: usize,
    tol: f64,
    n_init: usize,
    init_params: GmmInit,
    reg_covar: f64,
    warm_start: bool,
    random_state: Option<u64>,

    // Fitted state
    weights_: Option<Array1<f64>>,
    means_: Option<Array2<f64>>,
    // Covariance storage depends on type:
    // Full: n_components matrices of (n_features, n_features)
    // Tied: 1 matrix of (n_features, n_features)
    // Diagonal: n_components vectors of (n_features,)
    // Spherical: n_components scalars
    covariances_full_: Option<Vec<Array2<f64>>>,
    covariances_tied_: Option<Array2<f64>>,
    covariances_diag_: Option<Array2<f64>>,
    covariances_spherical_: Option<Array1<f64>>,
    // Precomputed precision Cholesky factors
    precisions_cholesky_full_: Option<Vec<Array2<f64>>>,
    precisions_cholesky_tied_: Option<Array2<f64>>,

    labels_: Option<Array1<i32>>,
    n_iter_: Option<usize>,
    lower_bound_: Option<f64>,
    converged_: Option<bool>,
    convergence_status_: Option<crate::ConvergenceStatus>,
    n_features_in_: Option<usize>,
}

impl Default for GaussianMixture {
    fn default() -> Self {
        Self::new(1)
    }
}

impl GaussianMixture {
    /// Create a new GaussianMixture with given number of components.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            cov_type: CovarianceType::Full,
            max_iter: 100,
            tol: 1e-3,
            n_init: 1,
            init_params: GmmInit::KMeans,
            reg_covar: 1e-6,
            warm_start: false,
            random_state: None,
            weights_: None,
            means_: None,
            covariances_full_: None,
            covariances_tied_: None,
            covariances_diag_: None,
            covariances_spherical_: None,
            precisions_cholesky_full_: None,
            precisions_cholesky_tied_: None,
            labels_: None,
            n_iter_: None,
            lower_bound_: None,
            converged_: None,
            convergence_status_: None,
            n_features_in_: None,
        }
    }

    /// Set the covariance type.
    pub fn covariance_type(mut self, cov_type: CovarianceType) -> Self {
        self.cov_type = cov_type;
        self
    }

    /// Set maximum number of EM iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set number of initializations.
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set initialization method.
    pub fn init_params(mut self, init: GmmInit) -> Self {
        self.init_params = init;
        self
    }

    /// Set covariance regularization.
    pub fn reg_covar(mut self, reg: f64) -> Self {
        self.reg_covar = reg;
        self
    }

    /// Set warm start.
    pub fn warm_start(mut self, warm: bool) -> Self {
        self.warm_start = warm;
        self
    }

    /// Set random state for reproducibility.
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Get fitted weights.
    pub fn weights(&self) -> Option<&Array1<f64>> {
        self.weights_.as_ref()
    }

    /// Get fitted means.
    pub fn means(&self) -> Option<&Array2<f64>> {
        self.means_.as_ref()
    }

    /// Get number of iterations run.
    pub fn n_iter(&self) -> Option<usize> {
        self.n_iter_
    }

    /// Whether the model converged.
    pub fn converged(&self) -> Option<bool> {
        self.converged_
    }

    /// Get the lower bound (log-likelihood) from the last fit.
    pub fn lower_bound(&self) -> Option<f64> {
        self.lower_bound_
    }

    /// Get convergence status after fitting.
    pub fn convergence_status(&self) -> Option<&crate::ConvergenceStatus> {
        self.convergence_status_.as_ref()
    }

    /// Get covariances for Full type.
    pub fn covariances_full(&self) -> Option<&Vec<Array2<f64>>> {
        self.covariances_full_.as_ref()
    }

    /// Get covariance for Tied type.
    pub fn covariances_tied(&self) -> Option<&Array2<f64>> {
        self.covariances_tied_.as_ref()
    }

    /// Get diagonal covariances.
    pub fn covariances_diag(&self) -> Option<&Array2<f64>> {
        self.covariances_diag_.as_ref()
    }

    /// Get spherical covariances.
    pub fn covariances_spherical(&self) -> Option<&Array1<f64>> {
        self.covariances_spherical_.as_ref()
    }

    /// Compute per-sample log-likelihood.
    pub fn score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let log_resp = self.estimate_log_prob_resp(x)?;
        // log p(x) = logsumexp over components of (log_weights + log_prob)
        let n = x.nrows();
        let mut log_likelihood = Array1::zeros(n);
        for i in 0..n {
            let row = log_resp.row(i);
            let row_vec = row.to_vec();
            log_likelihood[i] = logsumexp(&row_vec);
        }
        Ok(log_likelihood)
    }

    /// Compute total log-likelihood (mean per sample).
    pub fn score(&self, x: &Array2<f64>) -> Result<f64> {
        let log_lik = self.score_samples(x)?;
        Ok(log_lik.mean().unwrap_or(f64::NEG_INFINITY))
    }

    /// Bayesian Information Criterion.
    pub fn bic(&self, x: &Array2<f64>) -> Result<f64> {
        let n = x.nrows() as f64;
        let log_lik = self.score_samples(x)?.sum();
        let k = self.n_parameters(x.ncols()) as f64;
        Ok(-2.0 * log_lik + k * n.ln())
    }

    /// Akaike Information Criterion.
    pub fn aic(&self, x: &Array2<f64>) -> Result<f64> {
        let log_lik = self.score_samples(x)?.sum();
        let k = self.n_parameters(x.ncols()) as f64;
        Ok(-2.0 * log_lik + 2.0 * k)
    }

    /// Predict soft cluster assignments (responsibilities).
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let log_resp = self.estimate_log_prob_resp(x)?;
        let n = x.nrows();
        let k = self.n_components;
        let mut resp = Array2::zeros((n, k));

        for i in 0..n {
            let row = log_resp.row(i);
            let row_vec = row.to_vec();
            let lse = logsumexp(&row_vec);
            for j in 0..k {
                resp[[i, j]] = (log_resp[[i, j]] - lse).exp();
            }
        }
        Ok(resp)
    }

    /// Generate random samples from the fitted model.
    pub fn sample(&self, n_samples: usize) -> Result<(Array2<f64>, Array1<i32>)> {
        let weights = self
            .weights_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("GaussianMixture"))?;
        let means = self
            .means_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("sample"))?;
        let n_features = means.ncols();

        let mut rng = match self.random_state {
            Some(seed) => rand_chacha::ChaCha8Rng::seed_from_u64(seed),
            None => rand_chacha::ChaCha8Rng::from_os_rng(),
        };

        let mut x = Array2::zeros((n_samples, n_features));
        let mut labels = Array1::zeros(n_samples);

        // Build cumulative weights for component selection
        let cum_weights: Vec<f64> = weights
            .iter()
            .scan(0.0, |acc, &w| {
                *acc += w;
                Some(*acc)
            })
            .collect();

        for i in 0..n_samples {
            // Select component
            let u: f64 = rng.random();
            let comp = cum_weights
                .iter()
                .position(|&cw| u <= cw)
                .unwrap_or(self.n_components - 1);
            labels[i] = comp as i32;

            // Sample from that component's Gaussian
            let mean = means.row(comp);
            let sample = self.sample_from_component(comp, n_features, &mut rng)?;
            for j in 0..n_features {
                x[[i, j]] = mean[j] + sample[j];
            }
        }

        Ok((x, labels))
    }

    // =========================================================================
    // Private methods
    // =========================================================================

    /// Number of free parameters in the model.
    fn n_parameters(&self, n_features: usize) -> usize {
        let k = self.n_components;
        // Means
        let mean_params = k * n_features;
        // Weights (k-1 because they sum to 1)
        let weight_params = k - 1;
        // Covariance params
        let cov_params = match self.cov_type {
            CovarianceType::Full => k * n_features * (n_features + 1) / 2,
            CovarianceType::Tied => n_features * (n_features + 1) / 2,
            CovarianceType::Diagonal => k * n_features,
            CovarianceType::Spherical => k,
        };
        mean_params + weight_params + cov_params
    }

    /// Initialize parameters (means, weights, covariances).
    fn initialize_parameters(
        &mut self,
        x: &Array2<f64>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Result<()> {
        let (n, d) = x.dim();
        let k = self.n_components;

        // Initialize responsibilities via KMeans or random
        let resp = match self.init_params {
            GmmInit::KMeans => {
                let mut kmeans = KMeans::new(k)
                    .max_iter(30)
                    .n_init(1)
                    .random_state(rng.random());
                kmeans.fit(x)?;
                let labels = kmeans.predict(x)?;
                let mut resp = Array2::zeros((n, k));
                for i in 0..n {
                    let c = labels[i] as usize;
                    if c < k {
                        resp[[i, c]] = 1.0;
                    }
                }
                // Ensure no empty clusters
                for j in 0..k {
                    let col_sum: f64 = resp.column(j).sum();
                    if col_sum < 1.0 {
                        // Assign a random sample
                        let idx = rng.random_range(0..n);
                        resp[[idx, j]] = 1.0;
                    }
                }
                resp
            }
            GmmInit::Random => {
                let mut resp = Array2::zeros((n, k));
                for i in 0..n {
                    let c = rng.random_range(0..k);
                    resp[[i, c]] = 1.0;
                }
                resp
            }
        };

        // M-step to get initial parameters from responsibilities
        self.m_step(x, &resp, d)?;
        Ok(())
    }

    /// E-step: compute log responsibilities.
    fn estimate_log_prob_resp(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let weights = self
            .weights_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("GaussianMixture"))?;
        let n = x.nrows();
        let k = self.n_components;

        let log_prob = self.estimate_log_prob(x)?;
        let mut log_resp = Array2::zeros((n, k));

        for i in 0..n {
            for j in 0..k {
                log_resp[[i, j]] = weights[j].ln() + log_prob[[i, j]];
            }
        }

        Ok(log_resp)
    }

    /// Compute log probability of each sample under each component.
    fn estimate_log_prob(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        match self.cov_type {
            CovarianceType::Full => self.estimate_log_prob_full(x),
            CovarianceType::Tied => self.estimate_log_prob_tied(x),
            CovarianceType::Diagonal => self.estimate_log_prob_diagonal(x),
            CovarianceType::Spherical => self.estimate_log_prob_spherical(x),
        }
    }

    fn estimate_log_prob_full(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let means = self
            .means_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("estimate_log_prob_full"))?;
        let precisions_chol = self
            .precisions_cholesky_full_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("estimate_log_prob_full"))?;
        let n = x.nrows();
        let d = x.ncols();
        let k = self.n_components;

        let mut log_prob = Array2::zeros((n, k));

        for c in 0..k {
            let prec_chol = &precisions_chol[c];
            // log_det = sum of log of diagonal of precision cholesky
            let mut log_det: f64 = 0.0;
            for i in 0..d {
                log_det += prec_chol[[i, i]].ln();
            }

            // y = (x - mu) @ prec_chol
            for i in 0..n {
                let mut maha = 0.0;
                for j in 0..d {
                    let mut val = 0.0;
                    for l in 0..=j {
                        val += (x[[i, l]] - means[[c, l]]) * prec_chol[[l, j]];
                    }
                    maha += val * val;
                }
                log_prob[[i, c]] = -0.5 * (d as f64 * (2.0 * PI).ln() + maha) + log_det;
            }
        }

        Ok(log_prob)
    }

    fn estimate_log_prob_tied(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let means = self
            .means_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("estimate_log_prob_tied"))?;
        let prec_chol = self
            .precisions_cholesky_tied_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("estimate_log_prob_tied"))?;
        let n = x.nrows();
        let d = x.ncols();
        let k = self.n_components;

        let mut log_det: f64 = 0.0;
        for i in 0..d {
            log_det += prec_chol[[i, i]].ln();
        }

        let mut log_prob = Array2::zeros((n, k));
        for c in 0..k {
            for i in 0..n {
                let mut maha = 0.0;
                for j in 0..d {
                    let mut val = 0.0;
                    for l in 0..=j {
                        val += (x[[i, l]] - means[[c, l]]) * prec_chol[[l, j]];
                    }
                    maha += val * val;
                }
                log_prob[[i, c]] = -0.5 * (d as f64 * (2.0 * PI).ln() + maha) + log_det;
            }
        }

        Ok(log_prob)
    }

    fn estimate_log_prob_diagonal(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let means = self
            .means_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("estimate_log_prob_diagonal"))?;
        let diag_cov = self
            .covariances_diag_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("estimate_log_prob_diagonal"))?;
        let n = x.nrows();
        let d = x.ncols();
        let k = self.n_components;

        let mut log_prob = Array2::zeros((n, k));

        for c in 0..k {
            // log_det = -0.5 * sum(log(diag_cov[c]))
            let mut log_det: f64 = 0.0;
            for j in 0..d {
                log_det += diag_cov[[c, j]].max(self.reg_covar).ln();
            }

            for i in 0..n {
                let mut maha = 0.0;
                for j in 0..d {
                    let diff = x[[i, j]] - means[[c, j]];
                    maha += diff * diff / diag_cov[[c, j]];
                }
                log_prob[[i, c]] = -0.5 * (d as f64 * (2.0 * PI).ln() + log_det + maha);
            }
        }

        Ok(log_prob)
    }

    fn estimate_log_prob_spherical(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let means = self
            .means_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("estimate_log_prob_spherical"))?;
        let sph_cov = self
            .covariances_spherical_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("estimate_log_prob_spherical"))?;
        let n = x.nrows();
        let d = x.ncols();
        let k = self.n_components;

        let mut log_prob = Array2::zeros((n, k));

        for c in 0..k {
            let log_det = d as f64 * sph_cov[c].max(self.reg_covar).ln();

            for i in 0..n {
                let mut maha = 0.0;
                for j in 0..d {
                    let diff = x[[i, j]] - means[[c, j]];
                    maha += diff * diff;
                }
                maha /= sph_cov[c];
                log_prob[[i, c]] = -0.5 * (d as f64 * (2.0 * PI).ln() + log_det + maha);
            }
        }

        Ok(log_prob)
    }

    /// M-step: update parameters from responsibilities.
    fn m_step(&mut self, x: &Array2<f64>, resp: &Array2<f64>, d: usize) -> Result<()> {
        let n = x.nrows();
        let k = self.n_components;

        // Effective number of points per component
        let nk: Array1<f64> = resp.sum_axis(Axis(0));

        // Update weights
        let total: f64 = nk.sum();
        self.weights_ = Some(&nk / total);

        // Warn if any component has near-zero weight (possible collapse)
        if let Some(ref w) = self.weights_ {
            for (c, &wt) in w.iter().enumerate() {
                if wt < 1e-6 {
                    eprintln!("Warning: GMM component {} has near-zero weight ({:.2e}). Consider using fewer components.", c, wt);
                    break; // Only warn once
                }
            }
        }

        // Update means: mu_k = sum(resp_k * x) / nk
        let mut means = Array2::zeros((k, d));
        for c in 0..k {
            if nk[c] > 0.0 {
                for j in 0..d {
                    let mut sum = 0.0;
                    for i in 0..n {
                        sum += resp[[i, c]] * x[[i, j]];
                    }
                    means[[c, j]] = sum / nk[c];
                }
            }
        }
        self.means_ = Some(means.clone());

        // Update covariances
        match self.cov_type {
            CovarianceType::Full => self.update_covariances_full(x, &means, resp, &nk, d)?,
            CovarianceType::Tied => self.update_covariances_tied(x, &means, resp, &nk, d)?,
            CovarianceType::Diagonal => self.update_covariances_diagonal(x, &means, resp, &nk, d),
            CovarianceType::Spherical => self.update_covariances_spherical(x, &means, resp, &nk, d),
        }

        Ok(())
    }

    fn update_covariances_full(
        &mut self,
        x: &Array2<f64>,
        means: &Array2<f64>,
        resp: &Array2<f64>,
        nk: &Array1<f64>,
        d: usize,
    ) -> Result<()> {
        let n = x.nrows();
        let k = self.n_components;
        let mut covs = Vec::with_capacity(k);
        let mut prec_chols = Vec::with_capacity(k);

        for c in 0..k {
            let mut cov = Array2::zeros((d, d));
            for i in 0..n {
                for p in 0..d {
                    let dp = x[[i, p]] - means[[c, p]];
                    for q in 0..=p {
                        let dq = x[[i, q]] - means[[c, q]];
                        cov[[p, q]] += resp[[i, c]] * dp * dq;
                    }
                }
            }
            // Symmetrize and normalize
            for p in 0..d {
                for q in 0..p {
                    cov[[q, p]] = cov[[p, q]];
                }
            }
            let effective_nk = nk[c].max(1e-10);
            cov /= effective_nk;

            // Add regularization
            for p in 0..d {
                cov[[p, p]] += self.reg_covar;
            }

            // Compute precision Cholesky: L such that cov = L L^T, then precision_chol = inv(L)^T
            let l = cholesky(&cov, 0.0)?;
            // Compute inv(L) via forward substitution with identity
            let eye = Array2::eye(d);
            let l_inv = solve_lower_triangular(&l, &eye)?;
            // precision_chol = L_inv^T (upper triangular, but we store transposed)
            prec_chols.push(l_inv.t().to_owned());
            covs.push(cov);
        }

        self.covariances_full_ = Some(covs);
        self.precisions_cholesky_full_ = Some(prec_chols);
        Ok(())
    }

    fn update_covariances_tied(
        &mut self,
        x: &Array2<f64>,
        means: &Array2<f64>,
        resp: &Array2<f64>,
        nk: &Array1<f64>,
        d: usize,
    ) -> Result<()> {
        let n = x.nrows();
        let k = self.n_components;

        // Pooled covariance
        let mut cov = Array2::zeros((d, d));
        for c in 0..k {
            for i in 0..n {
                for p in 0..d {
                    let dp = x[[i, p]] - means[[c, p]];
                    for q in 0..=p {
                        let dq = x[[i, q]] - means[[c, q]];
                        cov[[p, q]] += resp[[i, c]] * dp * dq;
                    }
                }
            }
        }
        // Symmetrize and normalize
        for p in 0..d {
            for q in 0..p {
                cov[[q, p]] = cov[[p, q]];
            }
        }
        let total: f64 = nk.sum().max(1e-10);
        cov /= total;

        for p in 0..d {
            cov[[p, p]] += self.reg_covar;
        }

        let l = cholesky(&cov, 0.0)?;
        let eye = Array2::eye(d);
        let l_inv = solve_lower_triangular(&l, &eye)?;

        self.covariances_tied_ = Some(cov);
        self.precisions_cholesky_tied_ = Some(l_inv.t().to_owned());
        Ok(())
    }

    fn update_covariances_diagonal(
        &mut self,
        x: &Array2<f64>,
        means: &Array2<f64>,
        resp: &Array2<f64>,
        nk: &Array1<f64>,
        d: usize,
    ) {
        let n = x.nrows();
        let k = self.n_components;
        let mut diag = Array2::zeros((k, d));

        for c in 0..k {
            let effective_nk = nk[c].max(1e-10);
            for j in 0..d {
                let mut sum = 0.0;
                for i in 0..n {
                    let diff = x[[i, j]] - means[[c, j]];
                    sum += resp[[i, c]] * diff * diff;
                }
                diag[[c, j]] = sum / effective_nk + self.reg_covar;
            }
        }

        self.covariances_diag_ = Some(diag);
    }

    fn update_covariances_spherical(
        &mut self,
        x: &Array2<f64>,
        means: &Array2<f64>,
        resp: &Array2<f64>,
        nk: &Array1<f64>,
        d: usize,
    ) {
        let n = x.nrows();
        let k = self.n_components;
        let mut spherical = Array1::zeros(k);

        for c in 0..k {
            let effective_nk = nk[c].max(1e-10);
            let mut sum = 0.0;
            for i in 0..n {
                for j in 0..d {
                    let diff = x[[i, j]] - means[[c, j]];
                    sum += resp[[i, c]] * diff * diff;
                }
            }
            spherical[c] = sum / (effective_nk * d as f64) + self.reg_covar;
        }

        self.covariances_spherical_ = Some(spherical);
    }

    /// Sample a zero-mean vector from the given component.
    fn sample_from_component(
        &self,
        comp: usize,
        n_features: usize,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Result<Vec<f64>> {
        // Generate standard normal samples
        let dist = rand_distr::StandardNormal;
        let z: Vec<f64> = (0..n_features).map(|_| rng.sample(dist)).collect();

        match self.cov_type {
            CovarianceType::Full => {
                let covs = self
                    .covariances_full_
                    .as_ref()
                    .ok_or_else(|| FerroError::not_fitted("sample_from_component"))?;
                let l = cholesky(&covs[comp], 0.0)?;
                // x = L * z
                let mut result = vec![0.0; n_features];
                for i in 0..n_features {
                    for j in 0..=i {
                        result[i] += l[[i, j]] * z[j];
                    }
                }
                Ok(result)
            }
            CovarianceType::Tied => {
                let cov = self
                    .covariances_tied_
                    .as_ref()
                    .ok_or_else(|| FerroError::not_fitted("sample_from_component"))?;
                let l = cholesky(cov, 0.0)?;
                let mut result = vec![0.0; n_features];
                for i in 0..n_features {
                    for j in 0..=i {
                        result[i] += l[[i, j]] * z[j];
                    }
                }
                Ok(result)
            }
            CovarianceType::Diagonal => {
                let diag = self
                    .covariances_diag_
                    .as_ref()
                    .ok_or_else(|| FerroError::not_fitted("sample_from_component"))?;
                let mut result = vec![0.0; n_features];
                for j in 0..n_features {
                    result[j] = z[j] * diag[[comp, j]].sqrt();
                }
                Ok(result)
            }
            CovarianceType::Spherical => {
                let sph = self
                    .covariances_spherical_
                    .as_ref()
                    .ok_or_else(|| FerroError::not_fitted("sample_from_component"))?;
                let std = sph[comp].sqrt();
                let result: Vec<f64> = z.iter().map(|&zi| zi * std).collect();
                Ok(result)
            }
        }
    }

    /// Run a single EM fit and return (lower_bound, n_iter, converged).
    fn fit_single(
        &mut self,
        x: &Array2<f64>,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Result<(f64, usize, bool)> {
        let n = x.nrows();
        let d = x.ncols();

        // If warm_start and already fitted, skip initialization (reuse current params)
        let already_fitted = self.weights_.is_some() && self.means_.is_some();
        if !(self.warm_start && already_fitted) {
            self.initialize_parameters(x, rng)?;
        }

        let mut prev_lower_bound = f64::NEG_INFINITY;
        let mut converged = false;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // E-step
            let log_resp = self.estimate_log_prob_resp(x)?;

            // Compute lower bound
            let mut lower_bound = 0.0;
            for i in 0..n {
                let row = log_resp.row(i);
                let row_vec = row.to_vec();
                lower_bound += logsumexp(&row_vec);
            }
            lower_bound /= n as f64;

            // Normalize responsibilities
            let k = self.n_components;
            let mut resp = Array2::zeros((n, k));
            for i in 0..n {
                let row = log_resp.row(i);
                let row_vec = row.to_vec();
                let lse = logsumexp(&row_vec);
                for j in 0..k {
                    resp[[i, j]] = (log_resp[[i, j]] - lse).exp();
                }
            }

            // Check convergence
            let change = (lower_bound - prev_lower_bound).abs();
            if iter > 0 && change < self.tol {
                converged = true;
            }
            prev_lower_bound = lower_bound;

            // M-step
            self.m_step(x, &resp, d)?;

            if converged {
                break;
            }
        }

        Ok((prev_lower_bound, n_iter, converged))
    }
}

impl ClusteringModel for GaussianMixture {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        crate::validation::validate_unsupervised_input(x)?;

        // Hyperparameter validation
        if self.n_components == 0 {
            return Err(FerroError::invalid_input(
                "Parameter n_components must be >= 1, got 0",
            ));
        }
        if self.tol <= 0.0 {
            return Err(FerroError::invalid_input(format!(
                "Parameter tol must be > 0, got {}",
                self.tol
            )));
        }

        let (n, d) = x.dim();
        if n < self.n_components {
            return Err(FerroError::invalid_input(format!(
                "Expected at least {} samples, got {}",
                self.n_components, n
            )));
        }

        self.n_features_in_ = Some(d);

        let base_seed = self.random_state.unwrap_or(0);

        let mut best_lower_bound = f64::NEG_INFINITY;
        let mut best_params: Option<(
            Array1<f64>,
            Array2<f64>,
            Option<Vec<Array2<f64>>>,
            Option<Array2<f64>>,
            Option<Array2<f64>>,
            Option<Array1<f64>>,
            Option<Vec<Array2<f64>>>,
            Option<Array2<f64>>,
            usize,
            bool,
        )> = None;

        // When warm_start and already fitted, only run 1 init (reuse current params)
        let n_init_runs = if self.warm_start && self.weights_.is_some() {
            1
        } else {
            self.n_init
        };

        for init_idx in 0..n_init_runs {
            let mut rng =
                rand_chacha::ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(init_idx as u64));

            match self.fit_single(x, &mut rng) {
                Ok((lower_bound, n_iter, converged)) => {
                    if lower_bound > best_lower_bound {
                        best_lower_bound = lower_bound;
                        best_params = Some((
                            self.weights_
                                .clone()
                                .ok_or_else(|| FerroError::not_fitted("operation"))?,
                            self.means_
                                .clone()
                                .ok_or_else(|| FerroError::not_fitted("operation"))?,
                            self.covariances_full_.clone(),
                            self.covariances_tied_.clone(),
                            self.covariances_diag_.clone(),
                            self.covariances_spherical_.clone(),
                            self.precisions_cholesky_full_.clone(),
                            self.precisions_cholesky_tied_.clone(),
                            n_iter,
                            converged,
                        ));
                    }
                }
                Err(e) => {
                    // If all initializations fail, propagate the last error
                    if init_idx == self.n_init - 1 && best_params.is_none() {
                        return Err(e);
                    }
                }
            }
        }

        if let Some((
            weights,
            means,
            cov_full,
            cov_tied,
            cov_diag,
            cov_sph,
            prec_full,
            prec_tied,
            n_iter,
            converged,
        )) = best_params
        {
            self.weights_ = Some(weights);
            self.means_ = Some(means);
            self.covariances_full_ = cov_full;
            self.covariances_tied_ = cov_tied;
            self.covariances_diag_ = cov_diag;
            self.covariances_spherical_ = cov_sph;
            self.precisions_cholesky_full_ = prec_full;
            self.precisions_cholesky_tied_ = prec_tied;
            self.n_iter_ = Some(n_iter);
            self.converged_ = Some(converged);
            self.lower_bound_ = Some(best_lower_bound);

            if converged {
                self.convergence_status_ =
                    Some(crate::ConvergenceStatus::Converged { iterations: n_iter });
            } else {
                tracing::warn!(
                    "GaussianMixture did not converge after {} iterations. \
                     Results may be suboptimal. Try increasing max_iter or n_init.",
                    n_iter
                );
                self.convergence_status_ = Some(crate::ConvergenceStatus::NotConverged {
                    iterations: n_iter,
                    final_change: f64::NAN,
                });
            }

            // Compute labels
            let labels = self.predict(x)?;
            self.labels_ = Some(labels);
        }

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let log_resp = self.estimate_log_prob_resp(x)?;
        let n = x.nrows();
        let k = self.n_components;
        let mut labels = Array1::zeros(n);

        for i in 0..n {
            let mut best_j = 0;
            let mut best_val = f64::NEG_INFINITY;
            for j in 0..k {
                if log_resp[[i, j]] > best_val {
                    best_val = log_resp[[i, j]];
                    best_j = j;
                }
            }
            labels[i] = best_j as i32;
        }

        Ok(labels)
    }

    fn labels(&self) -> Option<&Array1<i32>> {
        self.labels_.as_ref()
    }

    fn is_fitted(&self) -> bool {
        self.weights_.is_some()
    }
}

// logsumexp moved to crate::linalg::logsumexp (shared utility)

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Generate well-separated 2D Gaussian blobs for testing.
    fn make_blobs(
        centers: &[(f64, f64)],
        n_per_cluster: usize,
        std: f64,
        seed: u64,
    ) -> (Array2<f64>, Array1<i32>) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let dist = rand_distr::StandardNormal;
        let n = centers.len() * n_per_cluster;
        let mut x = Array2::zeros((n, 2));
        let mut y = Array1::zeros(n);

        for (c, &(cx, cy)) in centers.iter().enumerate() {
            for i in 0..n_per_cluster {
                let idx = c * n_per_cluster + i;
                x[[idx, 0]] = cx + std * rng.sample::<f64, _>(dist);
                x[[idx, 1]] = cy + std * rng.sample::<f64, _>(dist);
                y[idx] = c as i32;
            }
        }
        (x, y)
    }

    #[test]
    fn test_basic_fit_predict_full() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 50, 0.5, 42);
        let mut gmm = GaussianMixture::new(2)
            .covariance_type(CovarianceType::Full)
            .random_state(42);
        gmm.fit(&x).unwrap();

        let labels = gmm.predict(&x).unwrap();
        assert_eq!(labels.len(), 100);

        // All samples in first half should share a label, second half another
        let label0 = labels[0];
        for i in 1..50 {
            assert_eq!(labels[i], label0);
        }
        let label1 = labels[50];
        assert_ne!(label0, label1);
        for i in 51..100 {
            assert_eq!(labels[i], label1);
        }
    }

    #[test]
    fn test_basic_fit_predict_tied() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 50, 0.5, 42);
        let mut gmm = GaussianMixture::new(2)
            .covariance_type(CovarianceType::Tied)
            .random_state(42);
        gmm.fit(&x).unwrap();

        let labels = gmm.predict(&x).unwrap();
        let label0 = labels[0];
        for i in 1..50 {
            assert_eq!(labels[i], label0);
        }
    }

    #[test]
    fn test_basic_fit_predict_diagonal() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 50, 0.5, 42);
        let mut gmm = GaussianMixture::new(2)
            .covariance_type(CovarianceType::Diagonal)
            .random_state(42);
        gmm.fit(&x).unwrap();

        let labels = gmm.predict(&x).unwrap();
        let label0 = labels[0];
        for i in 1..50 {
            assert_eq!(labels[i], label0);
        }
    }

    #[test]
    fn test_basic_fit_predict_spherical() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 50, 0.5, 42);
        let mut gmm = GaussianMixture::new(2)
            .covariance_type(CovarianceType::Spherical)
            .random_state(42);
        gmm.fit(&x).unwrap();

        let labels = gmm.predict(&x).unwrap();
        let label0 = labels[0];
        for i in 1..50 {
            assert_eq!(labels[i], label0);
        }
    }

    #[test]
    fn test_predict_proba_sums_to_one() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 30, 0.5, 42);
        let mut gmm = GaussianMixture::new(2).random_state(42);
        gmm.fit(&x).unwrap();

        let proba = gmm.predict_proba(&x).unwrap();
        assert_eq!(proba.dim(), (60, 2));

        for i in 0..60 {
            let row_sum: f64 = proba.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
            // Probabilities should be non-negative
            for j in 0..2 {
                assert!(proba[[i, j]] >= 0.0);
            }
        }
    }

    #[test]
    fn test_predict_proba_confident_on_separated() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (20.0, 20.0)], 20, 0.3, 42);
        let mut gmm = GaussianMixture::new(2).random_state(42);
        gmm.fit(&x).unwrap();

        let proba = gmm.predict_proba(&x).unwrap();
        // Should be very confident (>0.99) for well-separated clusters
        for i in 0..40 {
            let max_p = proba.row(i).iter().cloned().fold(0.0_f64, f64::max);
            assert!(max_p > 0.99, "Sample {} max prob = {}", i, max_p);
        }
    }

    #[test]
    fn test_bic_aic_correct_k() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)], 40, 0.5, 42);

        let mut bic_values = Vec::new();
        let mut aic_values = Vec::new();
        for k in 1..=5 {
            let mut gmm = GaussianMixture::new(k).random_state(42);
            gmm.fit(&x).unwrap();
            bic_values.push(gmm.bic(&x).unwrap());
            aic_values.push(gmm.aic(&x).unwrap());
        }

        // BIC/AIC should be minimized at k=3 (index 2)
        let bic_best = bic_values
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let aic_best = aic_values
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        assert_eq!(bic_best, 2, "BIC should select k=3, got k={}", bic_best + 1);
        assert_eq!(aic_best, 2, "AIC should select k=3, got k={}", aic_best + 1);
    }

    #[test]
    fn test_score_samples_finite() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (5.0, 5.0)], 30, 1.0, 42);
        let mut gmm = GaussianMixture::new(2).random_state(42);
        gmm.fit(&x).unwrap();

        let scores = gmm.score_samples(&x).unwrap();
        assert_eq!(scores.len(), 60);
        for &s in scores.iter() {
            assert!(s.is_finite(), "Score is not finite: {}", s);
        }
    }

    #[test]
    fn test_score_total() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (5.0, 5.0)], 30, 1.0, 42);
        let mut gmm = GaussianMixture::new(2).random_state(42);
        gmm.fit(&x).unwrap();

        let total = gmm.score(&x).unwrap();
        assert!(total.is_finite());
    }

    #[test]
    fn test_convergence() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 50, 0.5, 42);
        let mut gmm = GaussianMixture::new(2).max_iter(200).random_state(42);
        gmm.fit(&x).unwrap();

        assert!(gmm.converged().unwrap(), "GMM should converge on easy data");
        assert!(
            gmm.n_iter().unwrap() < 200,
            "Should converge before max_iter"
        );
    }

    #[test]
    fn test_n_init_picks_best() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (5.0, 5.0)], 30, 1.0, 42);

        let mut gmm1 = GaussianMixture::new(2).n_init(1).random_state(42);
        gmm1.fit(&x).unwrap();
        let lb1 = gmm1.lower_bound().unwrap();

        let mut gmm5 = GaussianMixture::new(2).n_init(5).random_state(42);
        gmm5.fit(&x).unwrap();
        let lb5 = gmm5.lower_bound().unwrap();

        // More inits should find equal or better lower bound
        assert!(
            lb5 >= lb1 - 1e-10,
            "n_init=5 ({}) should be >= n_init=1 ({})",
            lb5,
            lb1
        );
    }

    #[test]
    fn test_single_component() {
        let (x, _) = make_blobs(&[(5.0, 5.0)], 50, 1.0, 42);
        let mut gmm = GaussianMixture::new(1).random_state(42);
        gmm.fit(&x).unwrap();

        let labels = gmm.predict(&x).unwrap();
        for &l in labels.iter() {
            assert_eq!(l, 0);
        }

        let weights = gmm.weights().unwrap();
        assert_abs_diff_eq!(weights[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_single_feature() {
        let x = Array2::from_shape_vec(
            (60, 1),
            (0..30).map(|_| 0.0).chain((0..30).map(|_| 10.0)).collect(),
        )
        .unwrap();

        let mut gmm = GaussianMixture::new(2).random_state(42);
        gmm.fit(&x).unwrap();

        let labels = gmm.predict(&x).unwrap();
        let label0 = labels[0];
        for i in 1..30 {
            assert_eq!(labels[i], label0);
        }
    }

    #[test]
    fn test_reproducibility() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 30, 0.5, 42);

        let mut gmm1 = GaussianMixture::new(2).random_state(123);
        gmm1.fit(&x).unwrap();
        let labels1 = gmm1.predict(&x).unwrap();

        let mut gmm2 = GaussianMixture::new(2).random_state(123);
        gmm2.fit(&x).unwrap();
        let labels2 = gmm2.predict(&x).unwrap();

        assert_eq!(labels1, labels2, "Same seed should give same results");
    }

    #[test]
    fn test_sample_shape() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 50, 0.5, 42);
        let mut gmm = GaussianMixture::new(2).random_state(42);
        gmm.fit(&x).unwrap();

        let (samples, labels) = gmm.sample(100).unwrap();
        assert_eq!(samples.dim(), (100, 2));
        assert_eq!(labels.len(), 100);

        // Labels should be 0 or 1
        for &l in labels.iter() {
            assert!(l == 0 || l == 1, "Label should be 0 or 1, got {}", l);
        }

        // Samples should be finite
        for &v in samples.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sample_distribution() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (20.0, 20.0)], 100, 0.5, 42);
        let mut gmm = GaussianMixture::new(2).random_state(42);
        gmm.fit(&x).unwrap();

        let (samples, _) = gmm.sample(1000).unwrap();

        // Mean of samples should be roughly near the center of the two blobs
        let mean_x: f64 = samples.column(0).mean().unwrap();
        let mean_y: f64 = samples.column(1).mean().unwrap();
        assert!((mean_x - 10.0).abs() < 3.0, "mean_x = {}", mean_x);
        assert!((mean_y - 10.0).abs() < 3.0, "mean_y = {}", mean_y);
    }

    #[test]
    fn test_weights_sum_to_one() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)], 30, 0.5, 42);
        let mut gmm = GaussianMixture::new(3).random_state(42);
        gmm.fit(&x).unwrap();

        let weights = gmm.weights().unwrap();
        assert_eq!(weights.len(), 3);
        assert_abs_diff_eq!(weights.sum(), 1.0, epsilon = 1e-10);
        for &w in weights.iter() {
            assert!(w > 0.0);
        }
    }

    #[test]
    fn test_means_recovered() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 100, 0.3, 42);
        let mut gmm = GaussianMixture::new(2).random_state(42);
        gmm.fit(&x).unwrap();

        let means = gmm.means().unwrap();
        assert_eq!(means.dim(), (2, 2));

        // Sort means by first coordinate
        let mut m: Vec<(f64, f64)> = (0..2).map(|i| (means[[i, 0]], means[[i, 1]])).collect();
        m.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        assert!((m[0].0 - 0.0).abs() < 0.5, "mean0_x = {}", m[0].0);
        assert!((m[0].1 - 0.0).abs() < 0.5, "mean0_y = {}", m[0].1);
        assert!((m[1].0 - 10.0).abs() < 0.5, "mean1_x = {}", m[1].0);
        assert!((m[1].1 - 10.0).abs() < 0.5, "mean1_y = {}", m[1].1);
    }

    #[test]
    fn test_fit_predict() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 30, 0.5, 42);
        let mut gmm = GaussianMixture::new(2).random_state(42);
        let labels = gmm.fit_predict(&x).unwrap();
        assert_eq!(labels.len(), 60);
    }

    #[test]
    fn test_is_fitted() {
        let mut gmm = GaussianMixture::new(2);
        assert!(!gmm.is_fitted());

        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 20, 0.5, 42);
        gmm.fit(&x).unwrap();
        assert!(gmm.is_fitted());
    }

    #[test]
    fn test_not_fitted_error() {
        let gmm = GaussianMixture::new(2);
        let x = Array2::zeros((10, 2));
        assert!(gmm.predict(&x).is_err());
        assert!(gmm.predict_proba(&x).is_err());
        assert!(gmm.score_samples(&x).is_err());
    }

    #[test]
    fn test_too_few_samples() {
        let x = Array2::zeros((1, 2));
        let mut gmm = GaussianMixture::new(3);
        assert!(gmm.fit(&x).is_err());
    }

    #[test]
    fn test_random_init() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 50, 0.5, 42);
        let mut gmm = GaussianMixture::new(2)
            .init_params(GmmInit::Random)
            .n_init(10)
            .random_state(42);
        gmm.fit(&x).unwrap();

        // Should fit successfully and produce valid predictions
        assert!(gmm.is_fitted());
        let labels = gmm.predict(&x).unwrap();
        assert_eq!(labels.len(), 100);
        // Weights should sum to 1
        let weights = gmm.weights().unwrap();
        assert_abs_diff_eq!(weights.sum(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_three_components() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)], 40, 0.5, 42);
        let mut gmm = GaussianMixture::new(3).random_state(42);
        gmm.fit(&x).unwrap();

        let labels = gmm.predict(&x).unwrap();
        // Count unique labels
        let mut unique: Vec<i32> = labels.to_vec();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_diagonal_covariance_values() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 50, 1.0, 42);
        let mut gmm = GaussianMixture::new(2)
            .covariance_type(CovarianceType::Diagonal)
            .random_state(42);
        gmm.fit(&x).unwrap();

        let diag = gmm.covariances_diag().unwrap();
        assert_eq!(diag.dim(), (2, 2));
        // All diagonal covariances should be positive and roughly ~1.0 (std=1.0)
        for c in 0..2 {
            for j in 0..2 {
                assert!(diag[[c, j]] > 0.0);
                assert!(diag[[c, j]] < 5.0, "Variance too large: {}", diag[[c, j]]);
            }
        }
    }

    #[test]
    fn test_spherical_covariance_values() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 50, 1.0, 42);
        let mut gmm = GaussianMixture::new(2)
            .covariance_type(CovarianceType::Spherical)
            .random_state(42);
        gmm.fit(&x).unwrap();

        let sph = gmm.covariances_spherical().unwrap();
        assert_eq!(sph.len(), 2);
        for &v in sph.iter() {
            assert!(v > 0.0);
            assert!(v < 5.0, "Spherical variance too large: {}", v);
        }
    }

    #[test]
    fn test_warm_start() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 30, 0.5, 42);
        let mut gmm = GaussianMixture::new(2)
            .warm_start(true)
            .max_iter(5)
            .random_state(42);
        gmm.fit(&x).unwrap();
        let lb1 = gmm.lower_bound().unwrap();

        // Re-fit with warm_start should reuse previous params and converge
        // at least as well (starting from a good initialization)
        gmm.fit(&x).unwrap();
        let lb2 = gmm.lower_bound().unwrap();
        assert!(lb2.is_finite());
        // Second fit should be at least as good (warm start from good params)
        assert!(
            lb2 >= lb1 - 0.5,
            "lb2={} should be >= lb1={} - 0.5",
            lb2,
            lb1
        );
    }

    #[test]
    fn test_labels_stored_after_fit() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 20, 0.5, 42);
        let mut gmm = GaussianMixture::new(2).random_state(42);
        gmm.fit(&x).unwrap();

        let labels = gmm.labels().unwrap();
        assert_eq!(labels.len(), 40);
    }

    #[test]
    fn test_bic_decreasing_correct_direction() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 50, 0.5, 42);

        let mut gmm1 = GaussianMixture::new(1).random_state(42);
        gmm1.fit(&x).unwrap();
        let bic1 = gmm1.bic(&x).unwrap();

        let mut gmm2 = GaussianMixture::new(2).random_state(42);
        gmm2.fit(&x).unwrap();
        let bic2 = gmm2.bic(&x).unwrap();

        // BIC with correct k (2) should be lower than k=1
        assert!(
            bic2 < bic1,
            "BIC(k=2)={} should be < BIC(k=1)={}",
            bic2,
            bic1
        );
    }

    #[test]
    fn test_gmm_near_zero_variance_component() {
        // Data where one feature has near-zero variance — diagonal covariance
        // should not produce NaN/Inf in log probability
        let mut data = Vec::new();
        for i in 0..30 {
            data.push(i as f64);
            data.push(5.0); // constant feature — near-zero variance
        }
        let x = Array2::from_shape_vec((30, 2), data).unwrap();

        let mut gmm = GaussianMixture::new(2)
            .covariance_type(CovarianceType::Diagonal)
            .max_iter(50)
            .random_state(42);
        let result = gmm.fit(&x);
        assert!(
            result.is_ok(),
            "GMM diagonal fit should not fail: {:?}",
            result.err()
        );

        let labels = gmm.predict(&x).unwrap();
        for &l in labels.iter() {
            assert!(l >= 0, "Labels must be non-negative");
        }
    }

    #[test]
    fn test_gmm_spherical_degenerate() {
        // Data where one cluster has very tight points — spherical covariance
        // should not produce NaN/Inf in log probability
        let mut data = Vec::new();
        // Cluster 1: identical points
        for _ in 0..15 {
            data.push(0.0);
            data.push(0.0);
        }
        // Cluster 2: spread out
        for i in 0..15 {
            data.push(10.0 + i as f64 * 0.5);
            data.push(10.0 + i as f64 * 0.5);
        }
        let x = Array2::from_shape_vec((30, 2), data).unwrap();

        let mut gmm = GaussianMixture::new(2)
            .covariance_type(CovarianceType::Spherical)
            .max_iter(50)
            .random_state(42);
        let result = gmm.fit(&x);
        assert!(
            result.is_ok(),
            "GMM spherical fit should not fail: {:?}",
            result.err()
        );

        let labels = gmm.predict(&x).unwrap();
        for &l in labels.iter() {
            assert!(l >= 0, "Labels must be non-negative");
        }
    }

    #[test]
    fn test_convergence_status_converged() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 50, 0.5, 42);
        let mut gmm = GaussianMixture::new(2).max_iter(200).random_state(42);
        gmm.fit(&x).unwrap();

        let status = gmm.convergence_status().unwrap();
        assert!(
            matches!(status, crate::ConvergenceStatus::Converged { .. }),
            "Easy data should converge"
        );
    }

    #[test]
    fn test_convergence_status_not_converged() {
        let (x, _) = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 50, 0.5, 42);
        // max_iter=1 with very tight tol should not converge
        let mut gmm = GaussianMixture::new(2)
            .max_iter(1)
            .tol(1e-30)
            .random_state(42);
        gmm.fit(&x).unwrap(); // should not error, returns partial result

        let status = gmm.convergence_status().unwrap();
        assert!(
            matches!(status, crate::ConvergenceStatus::NotConverged { .. }),
            "max_iter=1 should not converge"
        );
    }
}

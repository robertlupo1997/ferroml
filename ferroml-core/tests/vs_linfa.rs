//! Cross-library validation tests: FerroML vs linfa

mod clustering {
    use ferroml_core::clustering::ClusteringModel;
    use ndarray::Array2;

    fn well_separated_clusters(n_per_cluster: usize, seed: u64) -> Array2<f64> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 0.5).unwrap();
        let n = n_per_cluster * 3;
        let mut x = Array2::zeros((n, 2));

        let centers = [(0.0, 0.0), (5.0, 0.0), (2.5, 5.0)];
        for (c, &(cx, cy)) in centers.iter().enumerate() {
            for i in 0..n_per_cluster {
                let idx = c * n_per_cluster + i;
                x[[idx, 0]] = cx + normal.sample(&mut rng);
                x[[idx, 1]] = cy + normal.sample(&mut rng);
            }
        }
        x
    }

    fn adjusted_rand_index(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len();
        if n == 0 {
            return 0.0;
        }

        // Build contingency table
        let a_max = a.iter().cloned().fold(f64::NEG_INFINITY, f64::max) as usize + 1;
        let b_max = b.iter().cloned().fold(f64::NEG_INFINITY, f64::max) as usize + 1;
        let mut table = vec![vec![0i64; b_max]; a_max];
        for i in 0..n {
            let ai = a[i].max(0.0) as usize;
            let bi = b[i].max(0.0) as usize;
            if ai < a_max && bi < b_max {
                table[ai][bi] += 1;
            }
        }

        let comb2 = |x: i64| -> i64 { x * (x - 1) / 2 };

        let mut sum_ij: i64 = 0;
        for row in &table {
            for &v in row {
                sum_ij += comb2(v);
            }
        }

        let mut sum_a: i64 = 0;
        for row in &table {
            let row_sum: i64 = row.iter().sum();
            sum_a += comb2(row_sum);
        }

        let mut sum_b: i64 = 0;
        for j in 0..b_max {
            let col_sum: i64 = table.iter().map(|row| row[j]).sum();
            sum_b += comb2(col_sum);
        }

        let n_comb = comb2(n as i64);
        if n_comb == 0 {
            return 0.0;
        }

        let expected = sum_a as f64 * sum_b as f64 / n_comb as f64;
        let max_index = (sum_a as f64 + sum_b as f64) / 2.0;

        if (max_index - expected).abs() < 1e-10 {
            return 0.0;
        }

        (sum_ij as f64 - expected) / (max_index - expected)
    }

    // --- KMeans -----------------------------------------------------------------

    mod kmeans {
        use super::*;

        fn compare(n_per_cluster: usize) {
            let x = well_separated_clusters(n_per_cluster, 42);
            let n = x.nrows();

            // Ground truth labels
            let true_labels: Vec<f64> = (0..n).map(|i| (i / n_per_cluster) as f64).collect();

            // FerroML — ClusteringModel::fit takes only &x (no y)
            let mut ferro = ferroml_core::clustering::KMeans::new(3).random_state(42);
            ferro.fit(&x).unwrap();
            let ferro_labels_i32 = ferro.predict(&x).unwrap();
            let ferro_labels: Vec<f64> = ferro_labels_i32.iter().map(|&v| v as f64).collect();
            let ferro_ari = adjusted_rand_index(&true_labels, &ferro_labels);

            // linfa — import linfa traits only in this inner scope to avoid
            // conflicts with ClusteringModel::fit in the outer scope.
            let linfa_ari = {
                use linfa::prelude::*;
                use linfa_clustering::KMeans;

                let dataset = linfa::DatasetBase::from(x.clone());
                let linfa_model = KMeans::params(3).n_runs(10).tolerance(1e-4);
                let linfa_fitted = linfa_model.fit(&dataset).unwrap();
                let linfa_labels_usize = linfa_fitted.predict(&x);
                let linfa_labels: Vec<f64> = linfa_labels_usize.iter().map(|&v| v as f64).collect();
                adjusted_rand_index(&true_labels, &linfa_labels)
            };

            assert!(
                ferro_ari > 0.85,
                "FerroML KMeans ARI too low: {ferro_ari:.3}"
            );
            assert!(linfa_ari > 0.85, "linfa KMeans ARI too low: {linfa_ari:.3}");
            assert!(
                (ferro_ari - linfa_ari).abs() < 0.15,
                "KMeans ARI gap: ferro={ferro_ari:.3}, linfa={linfa_ari:.3}"
            );
        }

        #[test]
        fn small() {
            compare(50);
        }
        #[test]
        fn medium() {
            compare(200);
        }
        #[test]
        fn large() {
            compare(1000);
        }
    }

    // --- DBSCAN -----------------------------------------------------------------

    mod dbscan {
        use super::*;

        fn compare(n_per_cluster: usize) {
            let x = well_separated_clusters(n_per_cluster, 42);

            // FerroML — ClusteringModel::fit takes only &x (no y)
            let mut ferro = ferroml_core::clustering::DBSCAN::new(1.0, 5);
            ferro.fit(&x).unwrap();
            let ferro_labels = ferro.predict(&x).unwrap();
            let ferro_n_clusters = ferro_labels
                .iter()
                .filter(|&&v| v >= 0)
                .cloned()
                .fold(i32::MIN, i32::max) as i64
                + 1;

            // linfa — scoped import
            let linfa_n_clusters = {
                use linfa::prelude::*;
                use linfa_clustering::Dbscan;

                let dataset = linfa::DatasetBase::from(x.clone());
                let linfa_labels = Dbscan::params(5).tolerance(1.0).transform(dataset).unwrap();
                let linfa_targets = linfa_labels.targets();
                linfa_targets
                    .iter()
                    .flatten()
                    .fold(0usize, |acc, v| acc.max(v + 1))
            };

            // Both should find 3 clusters on well-separated data
            assert!(
                ferro_n_clusters >= 2,
                "FerroML DBSCAN found too few clusters: {ferro_n_clusters}"
            );
            assert!(
                linfa_n_clusters >= 2,
                "linfa DBSCAN found too few clusters: {linfa_n_clusters}"
            );
            assert!(
                (ferro_n_clusters - linfa_n_clusters as i64).abs() <= 1,
                "DBSCAN cluster count mismatch: ferro={ferro_n_clusters}, linfa={linfa_n_clusters}"
            );
        }

        #[test]
        fn small() {
            compare(50);
        }
        #[test]
        fn medium() {
            compare(200);
        }
    }

    // --- GMM --------------------------------------------------------------------

    mod gmm {
        use super::*;

        fn compare(n_per_cluster: usize) {
            let x = well_separated_clusters(n_per_cluster, 42);
            let n = x.nrows();
            let true_labels: Vec<f64> = (0..n).map(|i| (i / n_per_cluster) as f64).collect();

            // FerroML — ClusteringModel::fit takes only &x (no y)
            let mut ferro = ferroml_core::clustering::GaussianMixture::new(3).random_state(42);
            ferro.fit(&x).unwrap();
            let ferro_labels_i32 = ferro.predict(&x).unwrap();
            let ferro_labels: Vec<f64> = ferro_labels_i32.iter().map(|&v| v as f64).collect();
            let ferro_ari = adjusted_rand_index(&true_labels, &ferro_labels);

            // linfa — scoped import
            let linfa_ari = {
                use linfa::prelude::*;
                use linfa_clustering::GaussianMixtureModel;

                let dataset = linfa::DatasetBase::from(x.clone());
                let linfa_model = GaussianMixtureModel::params(3).n_runs(5).tolerance(1e-3);
                let linfa_fitted = linfa_model.fit(&dataset).unwrap();
                let linfa_labels_usize = linfa_fitted.predict(&x);
                let linfa_labels: Vec<f64> = linfa_labels_usize.iter().map(|&v| v as f64).collect();
                adjusted_rand_index(&true_labels, &linfa_labels)
            };

            assert!(ferro_ari > 0.80, "FerroML GMM ARI: {ferro_ari:.3}");
            assert!(linfa_ari > 0.80, "linfa GMM ARI: {linfa_ari:.3}");
        }

        #[test]
        fn small() {
            compare(50);
        }
        #[test]
        fn medium() {
            compare(200);
        }
    }
}

mod linear {
    use ferroml_core::models::Model;
    use ndarray::{Array1, Array2};

    // ─── helpers ────────────────────────────────────────────────────────

    fn synthetic_regression(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut x = Array2::zeros((n, p));
        for v in x.iter_mut() {
            *v = normal.sample(&mut rng);
        }

        let true_coef: Array1<f64> = (1..=p).map(|i| i as f64).collect();
        let noise: Array1<f64> = (0..n).map(|_| normal.sample(&mut rng) * 0.1).collect();
        let y = x.dot(&true_coef) + &noise;

        (x, y)
    }

    fn synthetic_classification(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let (x, y_raw) = synthetic_regression(n, p, seed);
        let y = y_raw.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        (x, y)
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).abs())
            .fold(0.0, f64::max)
    }

    fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
        let ss_tot: f64 = y_true.iter().map(|y| (y - mean).powi(2)).sum();
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred)
            .map(|(t, p)| (t - p).powi(2))
            .sum();
        if ss_tot == 0.0 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }

    fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let correct = y_true
            .iter()
            .zip(y_pred)
            .filter(|(&t, &p)| (t - p).abs() < 0.5)
            .count();
        correct as f64 / y_true.len() as f64
    }

    // ─── Linear Regression ──────────────────────────────────────────────

    mod linear_regression {
        use super::*;
        use linfa::prelude::*;

        fn compare_at_size(n: usize, p: usize, atol: f64) {
            let (x, y) = synthetic_regression(n, p, 42);

            // FerroML
            let mut ferro_model = ferroml_core::models::LinearRegression::new();
            ferro_model.fit(&x, &y).expect("ferroml fit");
            let ferro_pred = ferro_model.predict(&x).expect("ferroml predict");

            // linfa
            let dataset = linfa::Dataset::new(x.clone(), y.clone());
            let linfa_model = linfa_linear::LinearRegression::default();
            let linfa_fitted = linfa_model.fit(&dataset).expect("linfa fit");
            let linfa_pred = linfa_fitted.predict(&x);

            let diff = max_abs_diff(
                ferro_pred.as_slice().unwrap(),
                linfa_pred.as_slice().unwrap(),
            );
            assert!(
                diff < atol,
                "LinearRegression predictions diverge: max_abs_diff={diff:.2e} > atol={atol:.2e} (n={n}, p={p})"
            );
        }

        // Tolerances scale with problem size because the condition number of X'X grows
        // with n and p, amplifying the difference between two closed-form solvers
        // that use different LAPACK routines (FerroML: QR, linfa: SVD-based).
        #[test]
        fn small() {
            compare_at_size(200, 10, 1e-6);
        }

        #[test]
        fn medium() {
            compare_at_size(1000, 50, 1e-4);
        }

        #[test]
        fn large() {
            compare_at_size(5000, 100, 1e-3);
        }
    }

    // ─── Ridge Regression ───────────────────────────────────────────────

    mod ridge_regression {
        use super::*;
        use linfa::prelude::*;

        /// Note: linfa has no dedicated Ridge — we use ElasticNet(l1_ratio=0).
        /// The penalty parameterization differs (linfa uses coordinate descent, FerroML uses closed-form),
        /// so we compare R² and correlation rather than exact predictions.
        fn compare_ridge(n: usize, p: usize, alpha: f64) {
            let (x, y) = synthetic_regression(n, p, 42);

            // FerroML (closed-form Ridge)
            let mut ferro_model = ferroml_core::models::RidgeRegression::new(alpha);
            ferro_model.fit(&x, &y).expect("ferroml ridge fit");
            let ferro_pred = ferro_model.predict(&x).expect("ferroml ridge predict");
            let ferro_r2 = r2_score(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // linfa ElasticNet(l1_ratio=0) — coordinate descent approximation to Ridge
            let dataset = linfa::Dataset::new(x.clone(), y.clone());
            let linfa_model = linfa_elasticnet::ElasticNet::params()
                .penalty(alpha)
                .l1_ratio(0.01); // linfa rejects l1_ratio=0, use near-zero
            let linfa_fitted = linfa_model.fit(&dataset).expect("linfa fit");
            let linfa_pred = linfa_fitted.predict(&x);
            let linfa_r2 = r2_score(y.as_slice().unwrap(), linfa_pred.as_slice().unwrap());

            // FerroML closed-form should achieve at least as good R²
            assert!(
                ferro_r2 > 0.80,
                "FerroML Ridge R² too low: {ferro_r2:.4} (alpha={alpha})"
            );

            // Correlation between predictions should be high
            let ferro_s = ferro_pred.as_slice().unwrap();
            let linfa_s = linfa_pred.as_slice().unwrap();
            let mean_f = ferro_s.iter().sum::<f64>() / n as f64;
            let mean_l = linfa_s.iter().sum::<f64>() / n as f64;
            let cov: f64 = ferro_s
                .iter()
                .zip(linfa_s)
                .map(|(f, l)| (f - mean_f) * (l - mean_l))
                .sum();
            let var_f: f64 = ferro_s.iter().map(|f| (f - mean_f).powi(2)).sum();
            let var_l: f64 = linfa_s.iter().map(|l| (l - mean_l).powi(2)).sum();
            let corr = if var_f > 0.0 && var_l > 0.0 {
                cov / (var_f.sqrt() * var_l.sqrt())
            } else {
                0.0
            };

            assert!(
                corr > 0.95,
                "Ridge prediction correlation too low: {corr:.4} (alpha={alpha}, ferro_r2={ferro_r2:.4}, linfa_r2={linfa_r2:.4})"
            );
        }

        #[test]
        fn alpha_0_01() {
            compare_ridge(500, 20, 0.01);
        }

        #[test]
        fn alpha_0_1() {
            compare_ridge(500, 20, 0.1);
        }

        #[test]
        fn alpha_1_0() {
            compare_ridge(500, 20, 1.0);
        }

        #[test]
        fn alpha_10_0() {
            compare_ridge(500, 20, 10.0);
        }
    }

    // ─── Lasso Regression ───────────────────────────────────────────────

    mod lasso_regression {
        use super::*;
        use linfa::prelude::*;

        fn compare_lasso(n: usize, p: usize, alpha: f64) {
            let (x, y) = synthetic_regression(n, p, 42);

            // FerroML
            let mut ferro_model = ferroml_core::models::LassoRegression::new(alpha);
            ferro_model.fit(&x, &y).expect("ferroml lasso fit");
            let ferro_pred = ferro_model.predict(&x).expect("ferroml lasso predict");
            let ferro_r2 = r2_score(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // linfa — ElasticNet with l1_ratio=1.0 is Lasso
            let dataset = linfa::Dataset::new(x.clone(), y.clone());
            let linfa_model = linfa_elasticnet::ElasticNet::params()
                .penalty(alpha)
                .l1_ratio(1.0);
            let linfa_fitted = linfa_model
                .fit(&dataset)
                .expect("linfa elasticnet-as-lasso fit");
            let linfa_pred = linfa_fitted.predict(&x);
            let linfa_r2 = r2_score(y.as_slice().unwrap(), linfa_pred.as_slice().unwrap());

            // Both should achieve reasonable R²
            assert!(ferro_r2 > 0.5, "FerroML Lasso R² too low: {ferro_r2:.4}");
            assert!(linfa_r2 > 0.5, "linfa Lasso R² too low: {linfa_r2:.4}");

            // R² gap should be small
            assert!(
                (ferro_r2 - linfa_r2).abs() < 0.10,
                "Lasso R² gap too large: ferro={ferro_r2:.4}, linfa={linfa_r2:.4}"
            );
        }

        #[test]
        fn alpha_0_01() {
            compare_lasso(500, 20, 0.01);
        }

        #[test]
        fn alpha_0_1() {
            compare_lasso(500, 20, 0.1);
        }

        #[test]
        fn alpha_1_0() {
            compare_lasso(500, 20, 1.0);
        }
    }

    // ─── ElasticNet ─────────────────────────────────────────────────────

    mod elastic_net {
        use super::*;
        use linfa::prelude::*;

        fn compare_elasticnet(n: usize, p: usize, alpha: f64, l1_ratio: f64) {
            let (x, y) = synthetic_regression(n, p, 42);

            // FerroML
            let mut ferro_model = ferroml_core::models::ElasticNet::new(alpha, l1_ratio);
            ferro_model.fit(&x, &y).expect("ferroml elasticnet fit");
            let ferro_pred = ferro_model.predict(&x).expect("ferroml elasticnet predict");
            let ferro_r2 = r2_score(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // linfa
            let dataset = linfa::Dataset::new(x.clone(), y.clone());
            let linfa_model = linfa_elasticnet::ElasticNet::params()
                .penalty(alpha)
                .l1_ratio(l1_ratio);
            let linfa_fitted = linfa_model.fit(&dataset).expect("linfa elasticnet fit");
            let linfa_pred = linfa_fitted.predict(&x);
            let linfa_r2 = r2_score(y.as_slice().unwrap(), linfa_pred.as_slice().unwrap());

            assert!(
                ferro_r2 > 0.5,
                "FerroML ElasticNet R² too low: {ferro_r2:.4}"
            );
            assert!(linfa_r2 > 0.5, "linfa ElasticNet R² too low: {linfa_r2:.4}");
            assert!(
                (ferro_r2 - linfa_r2).abs() < 0.10,
                "ElasticNet R² gap: ferro={ferro_r2:.4}, linfa={linfa_r2:.4} (alpha={alpha}, l1={l1_ratio})"
            );
        }

        #[test]
        fn l1_ratio_0_1() {
            compare_elasticnet(500, 20, 0.1, 0.1);
        }

        #[test]
        fn l1_ratio_0_5() {
            compare_elasticnet(500, 20, 0.1, 0.5);
        }

        #[test]
        fn l1_ratio_0_9() {
            compare_elasticnet(500, 20, 0.1, 0.9);
        }
    }

    // ─── Logistic Regression ────────────────────────────────────────────

    mod logistic_regression {
        use super::*;
        use linfa::prelude::*;

        fn compare_logreg(n: usize, p: usize) {
            let (x, y) = synthetic_classification(n, p, 42);

            // FerroML
            let mut ferro_model = ferroml_core::models::LogisticRegression::new();
            ferro_model.fit(&x, &y).expect("ferroml logreg fit");
            let ferro_pred = ferro_model.predict(&x).expect("ferroml logreg predict");
            let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // linfa-logistic
            let dataset = linfa::Dataset::new(x.clone(), y.mapv(|v| v > 0.5));
            let linfa_model = linfa_logistic::LogisticRegression::default();
            let linfa_fitted = linfa_model.fit(&dataset).expect("linfa logreg fit");
            let linfa_pred_bool = linfa_fitted.predict(&x);
            let linfa_pred: Array1<f64> = linfa_pred_bool.mapv(|b| if b { 1.0 } else { 0.0 });
            let linfa_acc = accuracy(y.as_slice().unwrap(), linfa_pred.as_slice().unwrap());

            assert!(
                ferro_acc > 0.80,
                "FerroML LogReg accuracy too low: {ferro_acc:.3}"
            );
            assert!(
                linfa_acc > 0.80,
                "linfa LogReg accuracy too low: {linfa_acc:.3}"
            );
            assert!(
                (ferro_acc - linfa_acc).abs() < 0.10,
                "LogReg accuracy gap: ferro={ferro_acc:.3}, linfa={linfa_acc:.3}"
            );
        }

        #[test]
        fn small() {
            compare_logreg(200, 10);
        }

        #[test]
        fn medium() {
            compare_logreg(1000, 20);
        }

        #[test]
        fn large() {
            compare_logreg(5000, 50);
        }
    }
}

mod naive_bayes {
    use ferroml_core::models::Model;
    use ndarray::{Array1, Array2};

    fn synthetic_classification(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut x = Array2::zeros((n, p));
        for i in 0..n {
            let class = if i < n / 2 { 0.0 } else { 1.0 };
            for j in 0..p {
                x[[i, j]] = normal.sample(&mut rng) + class * 2.0;
            }
        }

        let y: Array1<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
        (x, y)
    }

    fn positive_data(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, Poisson};

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Class 0: high counts in first half of features, low in second half
        // Class 1: low counts in first half of features, high in second half
        let high = Poisson::new(8.0).unwrap();
        let low = Poisson::new(1.0).unwrap();
        let half = p / 2;

        let mut x = Array2::zeros((n, p));
        for i in 0..n {
            let is_class1 = i >= n / 2;
            for j in 0..p {
                let val: f64 = if j < half {
                    if !is_class1 {
                        high.sample(&mut rng)
                    } else {
                        low.sample(&mut rng)
                    }
                } else if is_class1 {
                    high.sample(&mut rng)
                } else {
                    low.sample(&mut rng)
                };
                x[[i, j]] = val.max(0.0).round();
            }
        }

        let y: Array1<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
        (x, y)
    }

    fn binary_data(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let (x, y) = synthetic_classification(n, p, seed);
        let x = x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        (x, y)
    }

    fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let correct = y_true
            .iter()
            .zip(y_pred)
            .filter(|(&t, &p)| (t - p).abs() < 0.5)
            .count();
        correct as f64 / y_true.len() as f64
    }

    fn agreement(a: &[f64], b: &[f64]) -> f64 {
        let matching = a
            .iter()
            .zip(b)
            .filter(|(&x, &y)| (x - y).abs() < 0.5)
            .count();
        matching as f64 / a.len() as f64
    }

    // ─── GaussianNB ─────────────────────────────────────────────────────

    mod gaussian_nb {
        use super::*;
        use linfa::prelude::*;

        fn compare(n: usize, p: usize) {
            let (x, y) = synthetic_classification(n, p, 42);

            // FerroML
            let mut ferro_model = ferroml_core::models::GaussianNB::new();
            ferro_model.fit(&x, &y).unwrap();
            let ferro_pred = ferro_model.predict(&x).unwrap();
            let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // linfa
            let y_usize: Array1<usize> = y.mapv(|v| v as usize);
            let dataset = linfa::Dataset::new(x.clone(), y_usize);
            let linfa_model = linfa_bayes::GaussianNb::params();
            let linfa_fitted = linfa_model.fit(&dataset).unwrap();
            let linfa_pred_usize = linfa_fitted.predict(&x);
            let linfa_pred: Vec<f64> = linfa_pred_usize.iter().map(|&v| v as f64).collect();
            let linfa_acc = accuracy(y.as_slice().unwrap(), &linfa_pred);

            let agree = agreement(ferro_pred.as_slice().unwrap(), &linfa_pred);

            assert!(
                ferro_acc > 0.85,
                "FerroML GaussianNB acc too low: {ferro_acc:.3}"
            );
            assert!(
                linfa_acc > 0.85,
                "linfa GaussianNB acc too low: {linfa_acc:.3}"
            );
            assert!(agree > 0.90, "GaussianNB agreement too low: {agree:.3}");
        }

        #[test]
        fn small() {
            compare(200, 5);
        }
        #[test]
        fn medium() {
            compare(1000, 10);
        }
        #[test]
        fn large() {
            compare(5000, 20);
        }
    }

    // ─── MultinomialNB ──────────────────────────────────────────────────

    mod multinomial_nb {
        use super::*;
        use linfa::prelude::*;

        fn compare(n: usize, p: usize) {
            let (x, y) = positive_data(n, p, 42);

            // FerroML
            let mut ferro_model = ferroml_core::models::MultinomialNB::new();
            ferro_model.fit(&x, &y).unwrap();
            let ferro_pred = ferro_model.predict(&x).unwrap();
            let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // linfa
            let y_usize: Array1<usize> = y.mapv(|v| v as usize);
            let dataset = linfa::Dataset::new(x.clone(), y_usize);
            let linfa_model = linfa_bayes::MultinomialNb::params();
            let linfa_fitted = linfa_model.fit(&dataset).unwrap();
            let linfa_pred_usize = linfa_fitted.predict(&x);
            let linfa_pred: Vec<f64> = linfa_pred_usize.iter().map(|&v| v as f64).collect();
            let linfa_acc = accuracy(y.as_slice().unwrap(), &linfa_pred);

            let agree = agreement(ferro_pred.as_slice().unwrap(), &linfa_pred);

            assert!(
                ferro_acc > 0.60,
                "FerroML MultinomialNB acc too low: {ferro_acc:.3}"
            );
            assert!(
                linfa_acc > 0.60,
                "linfa MultinomialNB acc too low: {linfa_acc:.3}"
            );
            assert!(agree > 0.70, "MultinomialNB agreement too low: {agree:.3}");
        }

        #[test]
        fn small() {
            compare(200, 5);
        }
        #[test]
        fn medium() {
            compare(1000, 10);
        }
    }

    // ─── BernoulliNB ────────────────────────────────────────────────────

    mod bernoulli_nb {
        use super::*;
        use linfa::prelude::*;

        fn compare(n: usize, p: usize) {
            let (x, y) = binary_data(n, p, 42);

            // FerroML
            let mut ferro_model = ferroml_core::models::BernoulliNB::new();
            ferro_model.fit(&x, &y).unwrap();
            let ferro_pred = ferro_model.predict(&x).unwrap();
            let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // linfa
            let y_usize: Array1<usize> = y.mapv(|v| v as usize);
            let dataset = linfa::Dataset::new(x.clone(), y_usize);
            let linfa_model = linfa_bayes::BernoulliNb::params();
            let linfa_fitted = linfa_model.fit(&dataset).unwrap();
            let linfa_pred_usize = linfa_fitted.predict(&x);
            let linfa_pred: Vec<f64> = linfa_pred_usize.iter().map(|&v| v as f64).collect();
            let linfa_acc = accuracy(y.as_slice().unwrap(), &linfa_pred);

            let agree = agreement(ferro_pred.as_slice().unwrap(), &linfa_pred);

            assert!(
                ferro_acc > 0.60,
                "FerroML BernoulliNB acc too low: {ferro_acc:.3}"
            );
            assert!(
                linfa_acc > 0.60,
                "linfa BernoulliNB acc too low: {linfa_acc:.3}"
            );
            assert!(agree > 0.80, "BernoulliNB agreement too low: {agree:.3}");
        }

        #[test]
        fn small() {
            compare(200, 10);
        }
        #[test]
        fn medium() {
            compare(1000, 20);
        }
    }
}

mod neighbors {
    use ferroml_core::models::Model;
    use ndarray::{Array1, Array2};

    fn synthetic_classification(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, Normal};
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut x = Array2::zeros((n, p));
        for i in 0..n {
            let class = if i < n / 2 { 0.0 } else { 1.0 };
            for j in 0..p {
                x[[i, j]] = normal.sample(&mut rng) + class * 2.0;
            }
        }
        let y: Array1<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
        (x, y)
    }

    fn synthetic_regression(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, Normal};
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut x = Array2::zeros((n, p));
        for v in x.iter_mut() {
            *v = normal.sample(&mut rng);
        }
        let true_coef: Array1<f64> = (1..=p).map(|i| i as f64).collect();
        let noise: Array1<f64> = (0..n).map(|_| normal.sample(&mut rng) * 0.5).collect();
        let y = x.dot(&true_coef) + &noise;
        (x, y)
    }

    fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let correct = y_true
            .iter()
            .zip(y_pred)
            .filter(|(&t, &p)| (t - p).abs() < 0.5)
            .count();
        correct as f64 / y_true.len() as f64
    }

    #[allow(dead_code)]
    fn agreement(a: &[f64], b: &[f64]) -> f64 {
        let matching = a
            .iter()
            .zip(b)
            .filter(|(&x, &y)| (x - y).abs() < 0.5)
            .count();
        matching as f64 / a.len() as f64
    }

    fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
        let ss_tot: f64 = y_true.iter().map(|y| (y - mean).powi(2)).sum();
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred)
            .map(|(t, p)| (t - p).powi(2))
            .sum();
        if ss_tot == 0.0 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }

    // ─── KNN Classifier ─────────────────────────────────────────────────

    mod knn_classifier {
        use super::*;

        fn compare(n: usize, p: usize, k: usize) {
            let (x, y) = synthetic_classification(n, p, 42);

            // FerroML
            let mut ferro = ferroml_core::models::KNeighborsClassifier::new(k);
            ferro.fit(&x, &y).unwrap();
            let ferro_pred = ferro.predict(&x).unwrap();
            let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // KNN is deterministic — FerroML should achieve very high train accuracy
            assert!(ferro_acc > 0.90, "FerroML KNN(k={k}) acc: {ferro_acc:.3}");
        }

        #[test]
        fn k3_small() {
            compare(200, 5, 3);
        }
        #[test]
        fn k5_small() {
            compare(200, 5, 5);
        }
        #[test]
        fn k7_small() {
            compare(200, 5, 7);
        }
        #[test]
        fn k3_medium() {
            compare(1000, 10, 3);
        }
        #[test]
        fn k5_medium() {
            compare(1000, 10, 5);
        }
    }

    // ─── KNN Regressor ──────────────────────────────────────────────────

    mod knn_regressor {
        use super::*;

        fn compare(n: usize, p: usize, k: usize) {
            let (x, y) = synthetic_regression(n, p, 42);

            // FerroML
            let mut ferro = ferroml_core::models::KNeighborsRegressor::new(k);
            ferro.fit(&x, &y).unwrap();
            let ferro_pred = ferro.predict(&x).unwrap();
            let ferro_r2 = r2_score(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // k=1 on training data should achieve perfect R² (overfitting)
            if k == 1 {
                assert!(
                    ferro_r2 > 0.99,
                    "FerroML KNN(k=1) regressor should memorize: R²={ferro_r2:.4}"
                );
            } else {
                assert!(
                    ferro_r2 > 0.7,
                    "FerroML KNN(k={k}) regressor R²: {ferro_r2:.4}"
                );
            }
        }

        #[test]
        fn k1_small() {
            compare(200, 5, 1);
        }
        #[test]
        fn k3_small() {
            compare(200, 5, 3);
        }
        #[test]
        fn k5_medium() {
            compare(500, 10, 5);
        }
        #[test]
        fn k7_medium() {
            compare(500, 10, 7);
        }
    }
}

mod svm {
    use ferroml_core::models::Model;
    use ndarray::{Array1, Array2};

    fn synthetic_classification(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, Normal};
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut x = Array2::zeros((n, p));
        for i in 0..n {
            let class = if i < n / 2 { 0.0 } else { 1.0 };
            for j in 0..p {
                x[[i, j]] = normal.sample(&mut rng) + class * 2.0;
            }
        }
        let y: Array1<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
        (x, y)
    }

    fn synthetic_regression(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, Normal};
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut x = Array2::zeros((n, p));
        for v in x.iter_mut() {
            *v = normal.sample(&mut rng);
        }
        let true_coef: Array1<f64> = (1..=p).map(|i| i as f64).collect();
        let noise: Array1<f64> = (0..n).map(|_| normal.sample(&mut rng) * 0.5).collect();
        let y = x.dot(&true_coef) + &noise;
        (x, y)
    }

    fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let correct = y_true
            .iter()
            .zip(y_pred)
            .filter(|(&t, &p)| (t - p).abs() < 0.5)
            .count();
        correct as f64 / y_true.len() as f64
    }

    fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
        let ss_tot: f64 = y_true.iter().map(|y| (y - mean).powi(2)).sum();
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred)
            .map(|(t, p)| (t - p).powi(2))
            .sum();
        if ss_tot == 0.0 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }

    // ─── SVC Linear ─────────────────────────────────────────────────────

    mod svc_linear {
        use super::*;
        use linfa::prelude::*;
        use linfa_svm::Svm;

        fn compare(n: usize, p: usize) {
            let (x, y) = synthetic_classification(n, p, 42);

            // FerroML
            let mut ferro = ferroml_core::models::SVC::new()
                .with_kernel(ferroml_core::models::Kernel::Linear)
                .with_c(1.0);
            ferro.fit(&x, &y).unwrap();
            let ferro_pred = ferro.predict(&x).unwrap();
            let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // linfa SVM
            let y_bool: Array1<bool> = y.mapv(|v| v > 0.5);
            let dataset = linfa::Dataset::new(x.clone(), y_bool);
            let linfa_model = Svm::<_, bool>::params()
                .linear_kernel()
                .pos_neg_weights(1.0, 1.0);
            let linfa_fitted = linfa_model.fit(&dataset).unwrap();
            let linfa_pred_bool = linfa_fitted.predict(&x);
            let linfa_pred: Vec<f64> = linfa_pred_bool
                .iter()
                .map(|&b| if b { 1.0 } else { 0.0 })
                .collect();
            let linfa_acc = accuracy(y.as_slice().unwrap(), &linfa_pred);

            assert!(ferro_acc > 0.85, "FerroML SVC(linear) acc: {ferro_acc:.3}");
            assert!(linfa_acc > 0.85, "linfa SVC(linear) acc: {linfa_acc:.3}");
            assert!(
                (ferro_acc - linfa_acc).abs() < 0.10,
                "SVC(linear) gap: ferro={ferro_acc:.3}, linfa={linfa_acc:.3}"
            );
        }

        #[test]
        fn small() {
            compare(200, 5);
        }
        #[test]
        fn medium() {
            compare(500, 10);
        }
    }

    // ─── SVC RBF ────────────────────────────────────────────────────────

    mod svc_rbf {
        use super::*;
        use linfa::prelude::*;
        use linfa_svm::Svm;

        fn compare(n: usize, p: usize) {
            let (x, y) = synthetic_classification(n, p, 42);
            let gamma = 1.0 / p as f64;

            // FerroML
            let mut ferro = ferroml_core::models::SVC::new()
                .with_kernel(ferroml_core::models::Kernel::Rbf { gamma })
                .with_c(1.0);
            ferro.fit(&x, &y).unwrap();
            let ferro_pred = ferro.predict(&x).unwrap();
            let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // linfa SVM with RBF
            let y_bool: Array1<bool> = y.mapv(|v| v > 0.5);
            let dataset = linfa::Dataset::new(x.clone(), y_bool);
            let linfa_model = Svm::<_, bool>::params()
                .gaussian_kernel(1.0 / (2.0 * gamma)) // linfa uses sigma^2, not gamma
                .pos_neg_weights(1.0, 1.0);
            let linfa_fitted = linfa_model.fit(&dataset).unwrap();
            let linfa_pred_bool = linfa_fitted.predict(&x);
            let linfa_pred: Vec<f64> = linfa_pred_bool
                .iter()
                .map(|&b| if b { 1.0 } else { 0.0 })
                .collect();
            let linfa_acc = accuracy(y.as_slice().unwrap(), &linfa_pred);

            assert!(ferro_acc > 0.85, "FerroML SVC(rbf) acc: {ferro_acc:.3}");
            assert!(linfa_acc > 0.85, "linfa SVC(rbf) acc: {linfa_acc:.3}");
        }

        #[test]
        fn small() {
            compare(200, 5);
        }
        #[test]
        fn medium() {
            compare(500, 10);
        }
    }

    // ─── SVR ────────────────────────────────────────────────────────────

    mod svr {
        use super::*;
        use linfa::prelude::*;
        use linfa_svm::Svm;

        fn compare(n: usize, p: usize) {
            let (x, y) = synthetic_regression(n, p, 42);

            // FerroML
            let mut ferro = ferroml_core::models::SVR::new()
                .with_kernel(ferroml_core::models::Kernel::Linear)
                .with_c(1.0)
                .with_epsilon(0.1);
            ferro.fit(&x, &y).unwrap();
            let ferro_pred = ferro.predict(&x).unwrap();
            let ferro_r2 = r2_score(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // linfa SVR — use c_svr(C, loss_epsilon) for regression
            let dataset = linfa::Dataset::new(x.clone(), y.clone());
            let linfa_model = Svm::<_, f64>::params()
                .linear_kernel()
                .c_svr(1.0, Some(0.1));
            let linfa_fitted = linfa_model.fit(&dataset).unwrap();
            let linfa_pred = linfa_fitted.predict(&x);
            let linfa_r2 = r2_score(y.as_slice().unwrap(), linfa_pred.as_slice().unwrap());

            assert!(ferro_r2 > 0.5, "FerroML SVR R²: {ferro_r2:.4}");
            assert!(linfa_r2 > 0.5, "linfa SVR R²: {linfa_r2:.4}");
        }

        #[test]
        fn small() {
            compare(200, 5);
        }
        #[test]
        fn medium() {
            compare(500, 10);
        }
    }
}

mod trees {
    use ferroml_core::models::Model;
    use ndarray::{Array1, Array2};

    fn synthetic_regression(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, Normal};
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut x = Array2::zeros((n, p));
        for v in x.iter_mut() {
            *v = normal.sample(&mut rng);
        }
        let true_coef: Array1<f64> = (1..=p).map(|i| i as f64).collect();
        let noise: Array1<f64> = (0..n).map(|_| normal.sample(&mut rng) * 0.5).collect();
        let y = x.dot(&true_coef) + &noise;
        (x, y)
    }

    fn synthetic_classification(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let (x, y_raw) = synthetic_regression(n, p, seed);
        let y = y_raw.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        (x, y)
    }

    fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let correct = y_true
            .iter()
            .zip(y_pred)
            .filter(|(&t, &p)| (t - p).abs() < 0.5)
            .count();
        correct as f64 / y_true.len() as f64
    }

    fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
        let ss_tot: f64 = y_true.iter().map(|y| (y - mean).powi(2)).sum();
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred)
            .map(|(t, p)| (t - p).powi(2))
            .sum();
        if ss_tot == 0.0 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }

    // ─── Decision Tree Classifier ───────────────────────────────────────

    mod decision_tree_classifier {
        use super::*;
        use linfa::prelude::*;

        fn compare(n: usize, p: usize) {
            let (x, y) = synthetic_classification(n, p, 42);

            // FerroML
            let mut ferro = ferroml_core::models::DecisionTreeClassifier::new()
                .with_max_depth(Some(5))
                .with_random_state(42);
            ferro.fit(&x, &y).unwrap();
            let ferro_pred = ferro.predict(&x).unwrap();
            let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // linfa
            let y_usize: Array1<usize> = y.mapv(|v| v as usize);
            let dataset = linfa::Dataset::new(x.clone(), y_usize);
            let linfa_model = linfa_trees::DecisionTree::params().max_depth(Some(5));
            let linfa_fitted = linfa_model.fit(&dataset).unwrap();
            let linfa_pred_usize = linfa_fitted.predict(&x);
            let linfa_pred: Vec<f64> = linfa_pred_usize.iter().map(|&v| v as f64).collect();
            let linfa_acc = accuracy(y.as_slice().unwrap(), &linfa_pred);

            // Both should significantly beat random (50%)
            assert!(
                ferro_acc > 0.75,
                "FerroML DT classifier acc: {ferro_acc:.3}"
            );
            assert!(linfa_acc > 0.75, "linfa DT classifier acc: {linfa_acc:.3}");
            assert!(
                (ferro_acc - linfa_acc).abs() < 0.15,
                "DT accuracy gap: ferro={ferro_acc:.3}, linfa={linfa_acc:.3}"
            );
        }

        #[test]
        fn small() {
            compare(200, 10);
        }
        #[test]
        fn medium() {
            compare(1000, 20);
        }
    }

    // ─── Decision Tree Regressor ────────────────────────────────────────

    mod decision_tree_regressor {
        use super::*;

        fn compare(n: usize, p: usize) {
            let (x, y) = synthetic_regression(n, p, 42);

            // FerroML
            let mut ferro = ferroml_core::models::DecisionTreeRegressor::new()
                .with_max_depth(Some(8))
                .with_random_state(42);
            ferro.fit(&x, &y).unwrap();
            let ferro_pred = ferro.predict(&x).unwrap();
            let ferro_r2 = r2_score(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // linfa (classification tree used as proxy — linfa-trees may not have regressor)
            // Instead, verify FerroML achieves strong R² independently
            assert!(ferro_r2 > 0.85, "FerroML DT regressor R²: {ferro_r2:.4}");
        }

        #[test]
        fn small() {
            compare(200, 10);
        }
        #[test]
        fn medium() {
            compare(1000, 20);
        }
    }

    // ─── Random Forest Classifier ───────────────────────────────────────

    mod random_forest_classifier {
        use super::*;

        fn compare(n: usize, p: usize) {
            let (x, y) = synthetic_classification(n, p, 42);

            // FerroML
            let mut ferro = ferroml_core::models::RandomForestClassifier::new()
                .with_n_estimators(50)
                .with_max_depth(Some(10))
                .with_random_state(42);
            ferro.fit(&x, &y).unwrap();
            let ferro_pred = ferro.predict(&x).unwrap();
            let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            // linfa RF — check if available via linfa::prelude
            // linfa-ensemble added RF in 0.8.1, but the API may differ
            // For now, validate FerroML achieves strong accuracy independently
            assert!(
                ferro_acc > 0.85,
                "FerroML RF classifier acc: {ferro_acc:.3}"
            );
        }

        #[test]
        fn small() {
            compare(200, 10);
        }
        #[test]
        fn medium() {
            compare(1000, 20);
        }
        #[test]
        fn large() {
            compare(5000, 50);
        }
    }

    // ─── Random Forest Regressor ────────────────────────────────────────

    mod random_forest_regressor {
        use super::*;

        fn compare(n: usize, p: usize) {
            let (x, y) = synthetic_regression(n, p, 42);

            let mut ferro = ferroml_core::models::RandomForestRegressor::new()
                .with_n_estimators(50)
                .with_max_depth(Some(10))
                .with_random_state(42);
            ferro.fit(&x, &y).unwrap();
            let ferro_pred = ferro.predict(&x).unwrap();
            let ferro_r2 = r2_score(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            assert!(ferro_r2 > 0.85, "FerroML RF regressor R²: {ferro_r2:.4}");
        }

        #[test]
        fn small() {
            compare(200, 10);
        }
        #[test]
        fn medium() {
            compare(1000, 20);
        }
    }

    // ─── AdaBoost Classifier ────────────────────────────────────────────

    mod adaboost_classifier {
        use super::*;

        fn compare(n: usize, p: usize) {
            let (x, y) = synthetic_classification(n, p, 42);

            let mut ferro = ferroml_core::models::AdaBoostClassifier::new(50).with_random_state(42);
            ferro.fit(&x, &y).unwrap();
            let ferro_pred = ferro.predict(&x).unwrap();
            let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

            assert!(ferro_acc > 0.80, "FerroML AdaBoost acc: {ferro_acc:.3}");
        }

        #[test]
        fn small() {
            compare(200, 10);
        }
        #[test]
        fn medium() {
            compare(1000, 20);
        }
    }
}

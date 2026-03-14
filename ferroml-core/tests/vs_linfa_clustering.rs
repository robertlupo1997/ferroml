//! Cross-library validation: FerroML vs linfa — Clustering

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

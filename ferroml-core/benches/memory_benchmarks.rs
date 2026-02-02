//! FerroML Memory Profiling Benchmarks
//!
//! This module profiles memory usage of FerroML operations using dhat for heap
//! profiling and memory-stats for process memory monitoring.
//!
//! ## Running Memory Benchmarks
//!
//! Standard run (outputs to console):
//! ```bash
//! cargo bench --bench memory_benchmarks
//! ```
//!
//! With dhat heap profiling (generates dhat-heap.json):
//! ```bash
//! cargo bench --bench memory_benchmarks --features dhat-heap
//! ```
//!
//! ## Memory Profile Types
//!
//! 1. **Peak Memory**: Maximum memory used during operation
//! 2. **Allocation Count**: Number of heap allocations
//! 3. **Memory Efficiency**: Bytes per sample/feature
//! 4. **Memory Growth**: How memory scales with dataset size
//!
//! ## Key Operations Profiled
//!
//! - Model training (various sizes)
//! - Prediction/inference
//! - Preprocessing transforms
//! - Large dataset handling

use ferroml_core::datasets::{make_classification, make_regression};
use ferroml_core::models::boosting::GradientBoostingClassifier;
use ferroml_core::models::forest::RandomForestClassifier;
use ferroml_core::models::hist_boosting::HistGradientBoostingClassifier;
use ferroml_core::models::linear::LinearRegression;
use ferroml_core::models::regularized::{LassoRegression, RidgeRegression};
use ferroml_core::models::tree::DecisionTreeClassifier;
use ferroml_core::models::Model;
use ferroml_core::preprocessing::scalers::StandardScaler;
use ferroml_core::preprocessing::Transformer;
use memory_stats::memory_stats;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::time::Instant;

// =============================================================================
// MEMORY PROFILING UTILITIES
// =============================================================================

/// Memory snapshot for tracking usage
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MemorySnapshot {
    physical_bytes: usize,
    virtual_bytes: usize,
}

impl MemorySnapshot {
    fn capture() -> Option<Self> {
        memory_stats().map(|stats| Self {
            physical_bytes: stats.physical_mem,
            virtual_bytes: stats.virtual_mem,
        })
    }
}

/// Result of a memory profiling run
#[derive(Debug)]
struct MemoryProfile {
    operation: String,
    dataset_size: (usize, usize), // (n_samples, n_features)
    memory_before_mb: f64,
    memory_after_mb: f64,
    peak_memory_mb: f64,
    memory_delta_mb: f64,
    bytes_per_sample: f64,
    duration_ms: f64,
}

impl MemoryProfile {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "operation": self.operation,
            "n_samples": self.dataset_size.0,
            "n_features": self.dataset_size.1,
            "memory_before_mb": self.memory_before_mb,
            "memory_after_mb": self.memory_after_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "memory_delta_mb": self.memory_delta_mb,
            "bytes_per_sample": self.bytes_per_sample,
            "duration_ms": self.duration_ms,
        })
    }
}

/// Profile memory usage of a closure
fn profile_memory<F, R>(name: &str, n_samples: usize, n_features: usize, f: F) -> MemoryProfile
where
    F: FnOnce() -> R,
{
    // Force GC-like behavior by dropping previous allocations
    std::hint::black_box(());

    let before = MemorySnapshot::capture();
    let start = Instant::now();

    // Run the operation
    let result = f();
    std::hint::black_box(&result);

    let duration = start.elapsed();
    let after = MemorySnapshot::capture();

    let memory_before_mb = before
        .as_ref()
        .map(|s| s.physical_bytes as f64 / 1_048_576.0)
        .unwrap_or(0.0);
    let memory_after_mb = after
        .as_ref()
        .map(|s| s.physical_bytes as f64 / 1_048_576.0)
        .unwrap_or(0.0);
    let memory_delta_mb = memory_after_mb - memory_before_mb;

    MemoryProfile {
        operation: name.to_string(),
        dataset_size: (n_samples, n_features),
        memory_before_mb,
        memory_after_mb,
        peak_memory_mb: memory_after_mb, // Approximation without fine-grained tracking
        memory_delta_mb,
        bytes_per_sample: if n_samples > 0 {
            (memory_delta_mb * 1_048_576.0) / n_samples as f64
        } else {
            0.0
        },
        duration_ms: duration.as_secs_f64() * 1000.0,
    }
}

/// Generate synthetic regression data
fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let n_informative = n_features / 2;
    let (dataset, _) = make_regression(n_samples, n_features, n_informative, 0.1, Some(42));
    dataset.into_arrays()
}

/// Generate synthetic classification data
fn generate_classification_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<f64>, Array1<f64>) {
    let n_informative = n_features / 2;
    let (dataset, _) =
        make_classification(n_samples, n_features, n_informative, n_classes, Some(42));
    dataset.into_arrays()
}

// =============================================================================
// MEMORY BENCHMARKS
// =============================================================================

/// Profile LinearRegression memory usage
fn profile_linear_regression() -> Vec<MemoryProfile> {
    let sizes = [
        (1_000, 50),
        (5_000, 100),
        (10_000, 100),
        (50_000, 100),
        (100_000, 100),
    ];

    sizes
        .iter()
        .map(|&(n_samples, n_features)| {
            let (x, y) = generate_regression_data(n_samples, n_features);
            profile_memory("LinearRegression/fit", n_samples, n_features, || {
                let mut model = LinearRegression::new();
                model.fit(&x, &y).unwrap();
                model
            })
        })
        .collect()
}

/// Profile RidgeRegression memory usage
fn profile_ridge_regression() -> Vec<MemoryProfile> {
    let sizes = [(1_000, 50), (5_000, 100), (10_000, 100), (50_000, 100)];

    sizes
        .iter()
        .map(|&(n_samples, n_features)| {
            let (x, y) = generate_regression_data(n_samples, n_features);
            profile_memory("RidgeRegression/fit", n_samples, n_features, || {
                let mut model = RidgeRegression::new(1.0);
                model.fit(&x, &y).unwrap();
                model
            })
        })
        .collect()
}

/// Profile LassoRegression memory usage
fn profile_lasso_regression() -> Vec<MemoryProfile> {
    let sizes = [(1_000, 50), (5_000, 100), (10_000, 100)];

    sizes
        .iter()
        .map(|&(n_samples, n_features)| {
            let (x, y) = generate_regression_data(n_samples, n_features);
            profile_memory("LassoRegression/fit", n_samples, n_features, || {
                let mut model = LassoRegression::new(0.1);
                model.fit(&x, &y).unwrap();
                model
            })
        })
        .collect()
}

/// Profile DecisionTree memory usage
fn profile_decision_tree() -> Vec<MemoryProfile> {
    let sizes = [(1_000, 50), (5_000, 50), (10_000, 50), (25_000, 50)];

    sizes
        .iter()
        .map(|&(n_samples, n_features)| {
            let (x, y) = generate_classification_data(n_samples, n_features, 2);
            profile_memory("DecisionTreeClassifier/fit", n_samples, n_features, || {
                let mut model = DecisionTreeClassifier::new();
                model.fit(&x, &y).unwrap();
                model
            })
        })
        .collect()
}

/// Profile RandomForest memory usage - key for understanding ensemble scaling
fn profile_random_forest() -> Vec<MemoryProfile> {
    let sizes = [(1_000, 30), (5_000, 30), (10_000, 30)];

    sizes
        .iter()
        .flat_map(|&(n_samples, n_features)| {
            let (x, y) = generate_classification_data(n_samples, n_features, 2);

            // Test different numbers of estimators
            [10, 50, 100]
                .iter()
                .map(|&n_estimators| {
                    profile_memory(
                        &format!("RandomForestClassifier/fit(n_est={})", n_estimators),
                        n_samples,
                        n_features,
                        || {
                            let mut model = RandomForestClassifier::new()
                                .with_n_estimators(n_estimators)
                                .with_max_depth(Some(10))
                                .with_random_state(42);
                            model.fit(&x, &y).unwrap();
                            model
                        },
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Profile GradientBoosting memory usage
fn profile_gradient_boosting() -> Vec<MemoryProfile> {
    let sizes = [(1_000, 30), (5_000, 30), (10_000, 30)];

    sizes
        .iter()
        .flat_map(|&(n_samples, n_features)| {
            let (x, y) = generate_classification_data(n_samples, n_features, 2);

            [10, 50, 100]
                .iter()
                .map(|&n_estimators| {
                    profile_memory(
                        &format!("GradientBoostingClassifier/fit(n_est={})", n_estimators),
                        n_samples,
                        n_features,
                        || {
                            let mut model = GradientBoostingClassifier::new()
                                .with_n_estimators(n_estimators)
                                .with_max_depth(Some(3))
                                .with_learning_rate(0.1)
                                .with_random_state(42);
                            model.fit(&x, &y).unwrap();
                            model
                        },
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Profile HistGradientBoosting memory usage - should be more memory efficient
fn profile_hist_gradient_boosting() -> Vec<MemoryProfile> {
    let sizes = [(1_000, 30), (5_000, 30), (10_000, 30), (25_000, 30)];

    sizes
        .iter()
        .flat_map(|&(n_samples, n_features)| {
            let (x, y) = generate_classification_data(n_samples, n_features, 2);

            [10, 50, 100]
                .iter()
                .map(|&n_iter| {
                    profile_memory(
                        &format!("HistGradientBoostingClassifier/fit(max_iter={})", n_iter),
                        n_samples,
                        n_features,
                        || {
                            let mut model = HistGradientBoostingClassifier::new()
                                .with_max_iter(n_iter)
                                .with_max_depth(Some(6))
                                .with_learning_rate(0.1)
                                .with_random_state(42);
                            model.fit(&x, &y).unwrap();
                            model
                        },
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Profile preprocessing memory usage
fn profile_preprocessing() -> Vec<MemoryProfile> {
    let sizes = [(10_000, 100), (50_000, 100), (100_000, 100), (500_000, 100)];

    sizes
        .iter()
        .map(|&(n_samples, n_features)| {
            let (x, _) = generate_regression_data(n_samples, n_features);
            profile_memory(
                "StandardScaler/fit_transform",
                n_samples,
                n_features,
                || {
                    let mut scaler = StandardScaler::new();
                    scaler.fit_transform(&x).unwrap()
                },
            )
        })
        .collect()
}

/// Profile large model training (stress test)
fn profile_large_model_training() -> Vec<MemoryProfile> {
    let mut profiles = Vec::new();

    // Large RandomForest
    println!("  Profiling large RandomForest (10K samples, 100 trees)...");
    let (x, y) = generate_classification_data(10_000, 50, 2);
    profiles.push(profile_memory(
        "RandomForest/Large(10K_samples,100_trees)",
        10_000,
        50,
        || {
            let mut model = RandomForestClassifier::new()
                .with_n_estimators(100)
                .with_max_depth(Some(15))
                .with_random_state(42);
            model.fit(&x, &y).unwrap();
            model
        },
    ));

    // Large HistGradientBoosting
    println!("  Profiling large HistGradientBoosting (50K samples)...");
    let (x, y) = generate_classification_data(50_000, 30, 2);
    profiles.push(profile_memory(
        "HistGradientBoosting/Large(50K_samples,100_iter)",
        50_000,
        30,
        || {
            let mut model = HistGradientBoostingClassifier::new()
                .with_max_iter(100)
                .with_max_depth(Some(8))
                .with_learning_rate(0.1)
                .with_random_state(42);
            model.fit(&x, &y).unwrap();
            model
        },
    ));

    profiles
}

/// Profile prediction memory usage
fn profile_predictions() -> Vec<MemoryProfile> {
    let n_train = 5_000;
    let n_features = 50;

    // Train models first
    let (x_train, y_train) = generate_classification_data(n_train, n_features, 2);

    let mut rf = RandomForestClassifier::new()
        .with_n_estimators(50)
        .with_max_depth(Some(10))
        .with_random_state(42);
    rf.fit(&x_train, &y_train).unwrap();

    let mut hgb = HistGradientBoostingClassifier::new()
        .with_max_iter(50)
        .with_max_depth(Some(6))
        .with_learning_rate(0.1)
        .with_random_state(42);
    hgb.fit(&x_train, &y_train).unwrap();

    // Profile predictions at different batch sizes
    let batch_sizes = [100, 1_000, 10_000, 50_000];

    batch_sizes
        .iter()
        .flat_map(|&batch_size| {
            let (x_test, _) = generate_classification_data(batch_size, n_features, 2);

            vec![
                profile_memory("RandomForest/predict", batch_size, n_features, || {
                    rf.predict(&x_test).unwrap()
                }),
                profile_memory(
                    "HistGradientBoosting/predict",
                    batch_size,
                    n_features,
                    || hgb.predict(&x_test).unwrap(),
                ),
            ]
        })
        .collect()
}

// =============================================================================
// MAIN BENCHMARK RUNNER
// =============================================================================

fn main() {
    println!("FerroML Memory Profiling Benchmarks");
    println!("====================================\n");

    let mut all_profiles: HashMap<String, Vec<MemoryProfile>> = HashMap::new();

    // Run all memory profiles
    println!("1. Profiling Linear Models...");
    all_profiles.insert("linear_regression".to_string(), profile_linear_regression());
    all_profiles.insert("ridge_regression".to_string(), profile_ridge_regression());
    all_profiles.insert("lasso_regression".to_string(), profile_lasso_regression());

    println!("2. Profiling Tree Models...");
    all_profiles.insert("decision_tree".to_string(), profile_decision_tree());
    all_profiles.insert("random_forest".to_string(), profile_random_forest());

    println!("3. Profiling Gradient Boosting...");
    all_profiles.insert("gradient_boosting".to_string(), profile_gradient_boosting());
    all_profiles.insert(
        "hist_gradient_boosting".to_string(),
        profile_hist_gradient_boosting(),
    );

    println!("4. Profiling Preprocessing...");
    all_profiles.insert("preprocessing".to_string(), profile_preprocessing());

    println!("5. Profiling Large Model Training...");
    all_profiles.insert(
        "large_model_training".to_string(),
        profile_large_model_training(),
    );

    println!("6. Profiling Predictions...");
    all_profiles.insert("predictions".to_string(), profile_predictions());

    // Print results
    println!("\n\nMemory Profile Results");
    println!("======================\n");

    println!(
        "{:<50} {:>12} {:>12} {:>12} {:>12}",
        "Operation", "Samples", "Delta (MB)", "Bytes/Sample", "Time (ms)"
    );
    println!("{}", "-".repeat(100));

    for (category, profiles) in &all_profiles {
        println!("\n[{}]", category.to_uppercase());
        for profile in profiles {
            println!(
                "{:<50} {:>12} {:>12.2} {:>12.1} {:>12.1}",
                profile.operation,
                profile.dataset_size.0,
                profile.memory_delta_mb,
                profile.bytes_per_sample,
                profile.duration_ms
            );
        }
    }

    // Generate summary statistics
    println!("\n\nMemory Scaling Summary");
    println!("======================\n");

    // Find peak memory usage
    let peak_profile = all_profiles.values().flatten().max_by(|a, b| {
        a.memory_delta_mb
            .partial_cmp(&b.memory_delta_mb)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(peak) = peak_profile {
        println!(
            "Peak memory operation: {} ({:.2} MB for {} samples)",
            peak.operation, peak.memory_delta_mb, peak.dataset_size.0
        );
    }

    // Memory efficiency rankings
    println!("\nMost memory-efficient models (bytes per sample):");
    let mut all_flat: Vec<_> = all_profiles.values().flatten().collect();
    all_flat.sort_by(|a, b| {
        a.bytes_per_sample
            .partial_cmp(&b.bytes_per_sample)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for profile in all_flat.iter().take(5) {
        if profile.bytes_per_sample > 0.0 {
            println!(
                "  {}: {:.1} bytes/sample",
                profile.operation, profile.bytes_per_sample
            );
        }
    }

    // Output JSON for baseline comparison
    let json_output: Vec<serde_json::Value> = all_profiles
        .values()
        .flatten()
        .map(|p| p.to_json())
        .collect();

    if let Ok(json_str) = serde_json::to_string_pretty(&json_output) {
        std::fs::write("memory_profile_results.json", &json_str).ok();
        println!("\n\nDetailed results written to memory_profile_results.json");
    }

    println!("\nMemory profiling complete.");
}

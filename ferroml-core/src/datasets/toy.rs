//! Built-in Toy Datasets
//!
//! This module provides classic machine learning datasets embedded directly in the library.
//! These are useful for testing, learning, and benchmarking.
//!
//! ## Available Datasets
//!
//! ### Classification
//! - [`load_iris`] - Fisher's Iris dataset (150 samples, 4 features, 3 classes)
//! - [`load_wine`] - Wine recognition dataset (178 samples, 13 features, 3 classes)
//!
//! ### Regression
//! - [`load_diabetes`] - Diabetes progression dataset (442 samples, 10 features)
//! - [`load_linnerud`] - Linnerud physical exercise dataset (20 samples, 3 features, 3 targets)
//!
//! ## Example
//!
//! ```
//! use ferroml_core::datasets::{load_iris, Dataset, DatasetInfo};
//!
//! let (dataset, info) = load_iris();
//! println!("Dataset: {}", info.name);
//! println!("Samples: {}", dataset.n_samples());
//! println!("Features: {}", dataset.n_features());
//! println!("Classes: {:?}", info.n_classes);
//! ```

use crate::datasets::{Dataset, DatasetInfo};
use crate::Task;
use ndarray::{Array1, Array2};

/// Load the Iris dataset.
///
/// The Iris dataset is a classic and very easy multi-class classification dataset.
/// It contains 150 samples of iris flowers, with 4 features each:
/// - sepal length (cm)
/// - sepal width (cm)
/// - petal length (cm)
/// - petal width (cm)
///
/// The target is the species: setosa, versicolor, or virginica.
///
/// # Returns
///
/// A tuple of (Dataset, DatasetInfo).
///
/// # Example
///
/// ```
/// use ferroml_core::datasets::load_iris;
///
/// let (dataset, info) = load_iris();
/// assert_eq!(dataset.n_samples(), 150);
/// assert_eq!(dataset.n_features(), 4);
/// assert_eq!(info.n_classes, Some(3));
/// ```
///
/// # References
///
/// Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems.
/// Annals of Eugenics, 7(2), 179-188.
pub fn load_iris() -> (Dataset, DatasetInfo) {
    // Iris dataset: 150 samples, 4 features, 3 classes
    // Features: sepal length, sepal width, petal length, petal width (all in cm)
    // Classes: 0=setosa, 1=versicolor, 2=virginica
    #[rustfmt::skip]
    let data: [f64; 600] = [
        // Setosa (class 0) - 50 samples
        5.1, 3.5, 1.4, 0.2,
        4.9, 3.0, 1.4, 0.2,
        4.7, 3.2, 1.3, 0.2,
        4.6, 3.1, 1.5, 0.2,
        5.0, 3.6, 1.4, 0.2,
        5.4, 3.9, 1.7, 0.4,
        4.6, 3.4, 1.4, 0.3,
        5.0, 3.4, 1.5, 0.2,
        4.4, 2.9, 1.4, 0.2,
        4.9, 3.1, 1.5, 0.1,
        5.4, 3.7, 1.5, 0.2,
        4.8, 3.4, 1.6, 0.2,
        4.8, 3.0, 1.4, 0.1,
        4.3, 3.0, 1.1, 0.1,
        5.8, 4.0, 1.2, 0.2,
        5.7, 4.4, 1.5, 0.4,
        5.4, 3.9, 1.3, 0.4,
        5.1, 3.5, 1.4, 0.3,
        5.7, 3.8, 1.7, 0.3,
        5.1, 3.8, 1.5, 0.3,
        5.4, 3.4, 1.7, 0.2,
        5.1, 3.7, 1.5, 0.4,
        4.6, 3.6, 1.0, 0.2,
        5.1, 3.3, 1.7, 0.5,
        4.8, 3.4, 1.9, 0.2,
        5.0, 3.0, 1.6, 0.2,
        5.0, 3.4, 1.6, 0.4,
        5.2, 3.5, 1.5, 0.2,
        5.2, 3.4, 1.4, 0.2,
        4.7, 3.2, 1.6, 0.2,
        4.8, 3.1, 1.6, 0.2,
        5.4, 3.4, 1.5, 0.4,
        5.2, 4.1, 1.5, 0.1,
        5.5, 4.2, 1.4, 0.2,
        4.9, 3.1, 1.5, 0.2,
        5.0, 3.2, 1.2, 0.2,
        5.5, 3.5, 1.3, 0.2,
        4.9, 3.6, 1.4, 0.1,
        4.4, 3.0, 1.3, 0.2,
        5.1, 3.4, 1.5, 0.2,
        5.0, 3.5, 1.3, 0.3,
        4.5, 2.3, 1.3, 0.3,
        4.4, 3.2, 1.3, 0.2,
        5.0, 3.5, 1.6, 0.6,
        5.1, 3.8, 1.9, 0.4,
        4.8, 3.0, 1.4, 0.3,
        5.1, 3.8, 1.6, 0.2,
        4.6, 3.2, 1.4, 0.2,
        5.3, 3.7, 1.5, 0.2,
        5.0, 3.3, 1.4, 0.2,
        // Versicolor (class 1) - 50 samples
        7.0, 3.2, 4.7, 1.4,
        6.4, 3.2, 4.5, 1.5,
        6.9, 3.1, 4.9, 1.5,
        5.5, 2.3, 4.0, 1.3,
        6.5, 2.8, 4.6, 1.5,
        5.7, 2.8, 4.5, 1.3,
        6.3, 3.3, 4.7, 1.6,
        4.9, 2.4, 3.3, 1.0,
        6.6, 2.9, 4.6, 1.3,
        5.2, 2.7, 3.9, 1.4,
        5.0, 2.0, 3.5, 1.0,
        5.9, 3.0, 4.2, 1.5,
        6.0, 2.2, 4.0, 1.0,
        6.1, 2.9, 4.7, 1.4,
        5.6, 2.9, 3.6, 1.3,
        6.7, 3.1, 4.4, 1.4,
        5.6, 3.0, 4.5, 1.5,
        5.8, 2.7, 4.1, 1.0,
        6.2, 2.2, 4.5, 1.5,
        5.6, 2.5, 3.9, 1.1,
        5.9, 3.2, 4.8, 1.8,
        6.1, 2.8, 4.0, 1.3,
        6.3, 2.5, 4.9, 1.5,
        6.1, 2.8, 4.7, 1.2,
        6.4, 2.9, 4.3, 1.3,
        6.6, 3.0, 4.4, 1.4,
        6.8, 2.8, 4.8, 1.4,
        6.7, 3.0, 5.0, 1.7,
        6.0, 2.9, 4.5, 1.5,
        5.7, 2.6, 3.5, 1.0,
        5.5, 2.4, 3.8, 1.1,
        5.5, 2.4, 3.7, 1.0,
        5.8, 2.7, 3.9, 1.2,
        6.0, 2.7, 5.1, 1.6,
        5.4, 3.0, 4.5, 1.5,
        6.0, 3.4, 4.5, 1.6,
        6.7, 3.1, 4.7, 1.5,
        6.3, 2.3, 4.4, 1.3,
        5.6, 3.0, 4.1, 1.3,
        5.5, 2.5, 4.0, 1.3,
        5.5, 2.6, 4.4, 1.2,
        6.1, 3.0, 4.6, 1.4,
        5.8, 2.6, 4.0, 1.2,
        5.0, 2.3, 3.3, 1.0,
        5.6, 2.7, 4.2, 1.3,
        5.7, 3.0, 4.2, 1.2,
        5.7, 2.9, 4.2, 1.3,
        6.2, 2.9, 4.3, 1.3,
        5.1, 2.5, 3.0, 1.1,
        5.7, 2.8, 4.1, 1.3,
        // Virginica (class 2) - 50 samples
        6.3, 3.3, 6.0, 2.5,
        5.8, 2.7, 5.1, 1.9,
        7.1, 3.0, 5.9, 2.1,
        6.3, 2.9, 5.6, 1.8,
        6.5, 3.0, 5.8, 2.2,
        7.6, 3.0, 6.6, 2.1,
        4.9, 2.5, 4.5, 1.7,
        7.3, 2.9, 6.3, 1.8,
        6.7, 2.5, 5.8, 1.8,
        7.2, 3.6, 6.1, 2.5,
        6.5, 3.2, 5.1, 2.0,
        6.4, 2.7, 5.3, 1.9,
        6.8, 3.0, 5.5, 2.1,
        5.7, 2.5, 5.0, 2.0,
        5.8, 2.8, 5.1, 2.4,
        6.4, 3.2, 5.3, 2.3,
        6.5, 3.0, 5.5, 1.8,
        7.7, 3.8, 6.7, 2.2,
        7.7, 2.6, 6.9, 2.3,
        6.0, 2.2, 5.0, 1.5,
        6.9, 3.2, 5.7, 2.3,
        5.6, 2.8, 4.9, 2.0,
        7.7, 2.8, 6.7, 2.0,
        6.3, 2.7, 4.9, 1.8,
        6.7, 3.3, 5.7, 2.1,
        7.2, 3.2, 6.0, 1.8,
        6.2, 2.8, 4.8, 1.8,
        6.1, 3.0, 4.9, 1.8,
        6.4, 2.8, 5.6, 2.1,
        7.2, 3.0, 5.8, 1.6,
        7.4, 2.8, 6.1, 1.9,
        7.9, 3.8, 6.4, 2.0,
        6.4, 2.8, 5.6, 2.2,
        6.3, 2.8, 5.1, 1.5,
        6.1, 2.6, 5.6, 1.4,
        7.7, 3.0, 6.1, 2.3,
        6.3, 3.4, 5.6, 2.4,
        6.4, 3.1, 5.5, 1.8,
        6.0, 3.0, 4.8, 1.8,
        6.9, 3.1, 5.4, 2.1,
        6.7, 3.1, 5.6, 2.4,
        6.9, 3.1, 5.1, 2.3,
        5.8, 2.7, 5.1, 1.9,
        6.8, 3.2, 5.9, 2.3,
        6.7, 3.3, 5.7, 2.5,
        6.7, 3.0, 5.2, 2.3,
        6.3, 2.5, 5.0, 1.9,
        6.5, 3.0, 5.2, 2.0,
        6.2, 3.4, 5.4, 2.3,
        5.9, 3.0, 5.1, 1.8,
    ];

    #[rustfmt::skip]
    let targets: [f64; 150] = [
        // 50 setosa
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        // 50 versicolor
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        // 50 virginica
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];

    let x = Array2::from_shape_vec((150, 4), data.to_vec()).unwrap();
    let y = Array1::from_vec(targets.to_vec());

    let feature_names = vec![
        "sepal length (cm)".to_string(),
        "sepal width (cm)".to_string(),
        "petal length (cm)".to_string(),
        "petal width (cm)".to_string(),
    ];
    let target_names = vec![
        "setosa".to_string(),
        "versicolor".to_string(),
        "virginica".to_string(),
    ];

    let dataset = Dataset::new(x, y)
        .with_feature_names(feature_names.clone())
        .with_target_names(target_names.clone());

    let info = DatasetInfo::new("iris", Task::Classification, 150, 4)
        .with_description(
            "The Iris dataset is a classic and very easy multi-class classification dataset. \
             It contains measurements of 150 iris flowers from three different species. \
             The task is to predict the species based on sepal and petal measurements."
        )
        .with_n_classes(3)
        .with_feature_names(feature_names)
        .with_target_names(target_names)
        .with_source("Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2), 179-188.")
        .with_url("https://archive.ics.uci.edu/ml/datasets/iris")
        .with_license("Public Domain");

    (dataset, info)
}

/// Load the Wine recognition dataset.
///
/// The Wine dataset contains chemical analysis results of wines grown in the same
/// region in Italy but derived from three different cultivars. It has 178 samples
/// with 13 features each.
///
/// # Features
///
/// 1. Alcohol
/// 2. Malic acid
/// 3. Ash
/// 4. Alcalinity of ash
/// 5. Magnesium
/// 6. Total phenols
/// 7. Flavanoids
/// 8. Nonflavanoid phenols
/// 9. Proanthocyanins
/// 10. Color intensity
/// 11. Hue
/// 12. OD280/OD315 of diluted wines
/// 13. Proline
///
/// # Returns
///
/// A tuple of (Dataset, DatasetInfo).
///
/// # Example
///
/// ```
/// use ferroml_core::datasets::load_wine;
///
/// let (dataset, info) = load_wine();
/// assert_eq!(dataset.n_samples(), 178);
/// assert_eq!(dataset.n_features(), 13);
/// assert_eq!(info.n_classes, Some(3));
/// ```
///
/// # References
///
/// Forina, M. et al. (1988). PARVUS - An Extendible Package for Data Exploration.
pub fn load_wine() -> (Dataset, DatasetInfo) {
    // Wine dataset: 178 samples, 13 features, 3 classes
    #[rustfmt::skip]
    let data: [f64; 2314] = [
        // Class 0 (59 samples)
        14.23, 1.71, 2.43, 15.6, 127.0, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0,
        13.20, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050.0,
        13.16, 2.36, 2.67, 18.6, 101.0, 2.80, 3.24, 0.30, 2.81, 5.68, 1.03, 3.17, 1185.0,
        14.37, 1.95, 2.50, 16.8, 113.0, 3.85, 3.49, 0.24, 2.18, 7.80, 0.86, 3.45, 1480.0,
        13.24, 2.59, 2.87, 21.0, 118.0, 2.80, 2.69, 0.39, 1.82, 4.32, 1.04, 2.93, 735.0,
        14.20, 1.76, 2.45, 15.2, 112.0, 3.27, 3.39, 0.34, 1.97, 6.75, 1.05, 2.85, 1450.0,
        14.39, 1.87, 2.45, 14.6, 96.0, 2.50, 2.52, 0.30, 1.98, 5.25, 1.02, 3.58, 1290.0,
        14.06, 2.15, 2.61, 17.6, 121.0, 2.60, 2.51, 0.31, 1.25, 5.05, 1.06, 3.58, 1295.0,
        14.83, 1.64, 2.17, 14.0, 97.0, 2.80, 2.98, 0.29, 1.98, 5.20, 1.08, 2.85, 1045.0,
        13.86, 1.35, 2.27, 16.0, 98.0, 2.98, 3.15, 0.22, 1.85, 7.22, 1.01, 3.55, 1045.0,
        14.10, 2.16, 2.30, 18.0, 105.0, 2.95, 3.32, 0.22, 2.38, 5.75, 1.25, 3.17, 1510.0,
        14.12, 1.48, 2.32, 16.8, 95.0, 2.20, 2.43, 0.26, 1.57, 5.00, 1.17, 2.82, 1280.0,
        13.75, 1.73, 2.41, 16.0, 89.0, 2.60, 2.76, 0.29, 1.81, 5.60, 1.15, 2.90, 1320.0,
        14.75, 1.73, 2.39, 11.4, 91.0, 3.10, 3.69, 0.43, 2.81, 5.40, 1.25, 2.73, 1150.0,
        14.38, 1.87, 2.38, 12.0, 102.0, 3.30, 3.64, 0.29, 2.96, 7.50, 1.20, 3.00, 1547.0,
        13.63, 1.81, 2.70, 17.2, 112.0, 2.85, 2.91, 0.30, 1.46, 7.30, 1.28, 2.88, 1310.0,
        14.30, 1.92, 2.72, 20.0, 120.0, 2.80, 3.14, 0.33, 1.97, 6.20, 1.07, 2.65, 1280.0,
        13.83, 1.57, 2.62, 20.0, 115.0, 2.95, 3.40, 0.40, 1.72, 6.60, 1.13, 2.57, 1130.0,
        14.19, 1.59, 2.48, 16.5, 108.0, 3.30, 3.93, 0.32, 1.86, 8.70, 1.23, 2.82, 1680.0,
        13.64, 3.10, 2.56, 15.2, 116.0, 2.70, 3.03, 0.17, 1.66, 5.10, 0.96, 3.36, 845.0,
        14.06, 1.63, 2.28, 16.0, 126.0, 3.00, 3.17, 0.24, 2.10, 5.65, 1.09, 3.71, 780.0,
        12.93, 3.80, 2.65, 18.6, 102.0, 2.41, 2.41, 0.25, 1.98, 4.50, 1.03, 3.52, 770.0,
        13.71, 1.86, 2.36, 16.6, 101.0, 2.61, 2.88, 0.27, 1.69, 3.80, 1.11, 4.00, 1035.0,
        12.85, 1.60, 2.52, 17.8, 95.0, 2.48, 2.37, 0.26, 1.46, 3.93, 1.09, 3.63, 1015.0,
        13.50, 1.81, 2.61, 20.0, 96.0, 2.53, 2.61, 0.28, 1.66, 3.52, 1.12, 3.82, 845.0,
        13.05, 2.05, 3.22, 25.0, 124.0, 2.63, 2.68, 0.47, 1.92, 3.58, 1.13, 3.20, 830.0,
        13.39, 1.77, 2.62, 16.1, 93.0, 2.85, 2.94, 0.34, 1.45, 4.80, 0.92, 3.22, 1195.0,
        13.30, 1.72, 2.14, 17.0, 94.0, 2.40, 2.19, 0.27, 1.35, 3.95, 1.02, 2.77, 1285.0,
        13.87, 1.90, 2.80, 19.4, 107.0, 2.95, 2.97, 0.37, 1.76, 4.50, 1.25, 3.40, 915.0,
        14.02, 1.68, 2.21, 16.0, 96.0, 2.65, 2.33, 0.26, 1.98, 4.70, 1.04, 3.59, 1035.0,
        13.73, 1.50, 2.70, 22.5, 101.0, 3.00, 3.25, 0.29, 2.38, 5.70, 1.19, 2.71, 1285.0,
        13.58, 1.66, 2.36, 19.1, 106.0, 2.86, 3.19, 0.22, 1.95, 6.90, 1.09, 2.88, 1515.0,
        13.68, 1.83, 2.36, 17.2, 104.0, 2.42, 2.69, 0.42, 1.97, 3.84, 1.23, 2.87, 990.0,
        13.76, 1.53, 2.70, 19.5, 132.0, 2.95, 2.74, 0.50, 1.35, 5.40, 1.25, 3.00, 1235.0,
        13.51, 1.80, 2.65, 19.0, 110.0, 2.35, 2.53, 0.29, 1.54, 4.20, 1.10, 2.87, 1095.0,
        13.48, 1.81, 2.41, 20.5, 100.0, 2.70, 2.98, 0.26, 1.86, 5.10, 1.04, 3.47, 920.0,
        13.28, 1.64, 2.84, 15.5, 110.0, 2.60, 2.68, 0.34, 1.36, 4.60, 1.09, 2.78, 880.0,
        13.05, 1.65, 2.55, 18.0, 98.0, 2.45, 2.43, 0.29, 1.44, 4.25, 1.12, 2.51, 1105.0,
        13.07, 1.50, 2.10, 15.5, 98.0, 2.40, 2.64, 0.28, 1.37, 3.70, 1.18, 2.69, 1020.0,
        14.22, 3.99, 2.51, 13.2, 128.0, 3.00, 3.04, 0.20, 2.08, 5.10, 0.89, 3.53, 760.0,
        13.56, 1.73, 2.46, 20.5, 116.0, 2.96, 2.78, 0.20, 2.45, 6.25, 0.98, 3.03, 1120.0,
        13.41, 3.84, 2.12, 18.8, 90.0, 2.45, 2.68, 0.27, 1.48, 4.28, 0.91, 3.00, 985.0,
        13.88, 1.89, 2.59, 15.0, 101.0, 3.25, 3.56, 0.17, 1.70, 5.43, 0.88, 3.56, 1095.0,
        13.24, 3.98, 2.29, 17.5, 103.0, 2.64, 2.63, 0.32, 1.66, 4.36, 0.82, 3.00, 680.0,
        13.05, 1.77, 2.10, 17.0, 107.0, 3.00, 3.00, 0.28, 2.03, 5.04, 0.88, 3.35, 885.0,
        14.21, 4.04, 2.44, 18.9, 111.0, 2.85, 2.65, 0.30, 1.25, 5.24, 0.87, 3.33, 1080.0,
        14.38, 3.59, 2.28, 16.0, 102.0, 3.25, 3.17, 0.27, 2.19, 4.90, 1.04, 3.44, 1065.0,
        13.90, 1.68, 2.12, 16.0, 101.0, 3.10, 3.39, 0.21, 2.14, 6.10, 0.91, 3.33, 985.0,
        14.10, 2.02, 2.40, 18.8, 103.0, 2.75, 2.92, 0.32, 2.38, 6.20, 1.07, 2.75, 1060.0,
        13.94, 1.73, 2.27, 17.4, 108.0, 2.88, 3.54, 0.32, 2.08, 8.90, 1.12, 3.10, 1260.0,
        13.05, 1.73, 2.04, 12.4, 92.0, 2.72, 3.27, 0.17, 2.91, 7.20, 1.12, 2.91, 1150.0,
        13.83, 1.65, 2.60, 17.2, 94.0, 2.45, 2.99, 0.22, 2.29, 5.60, 1.24, 3.37, 1265.0,
        13.82, 1.75, 2.42, 14.0, 111.0, 3.88, 3.74, 0.32, 1.87, 7.05, 1.01, 3.26, 1190.0,
        13.77, 1.90, 2.68, 17.1, 115.0, 3.00, 2.79, 0.39, 1.68, 6.30, 1.13, 2.93, 1375.0,
        13.74, 1.67, 2.25, 16.4, 118.0, 2.60, 2.90, 0.21, 1.62, 5.85, 0.92, 3.20, 1060.0,
        13.56, 1.71, 2.31, 16.2, 117.0, 3.15, 3.29, 0.34, 2.34, 6.13, 0.95, 3.38, 1065.0,
        14.22, 1.70, 2.30, 16.3, 118.0, 3.20, 3.00, 0.26, 2.03, 6.38, 0.94, 3.31, 970.0,
        13.29, 1.97, 2.68, 16.8, 102.0, 3.00, 3.23, 0.31, 1.66, 6.00, 1.07, 2.84, 1270.0,
        13.72, 1.43, 2.50, 16.7, 108.0, 3.40, 3.67, 0.19, 2.04, 6.80, 0.89, 2.87, 1285.0,
        // Class 1 (71 samples)
        12.37, 0.94, 1.36, 10.6, 88.0, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520.0,
        12.33, 1.10, 2.28, 16.0, 101.0, 2.05, 1.09, 0.63, 0.41, 3.27, 1.25, 1.67, 680.0,
        12.64, 1.36, 2.02, 16.8, 100.0, 2.02, 1.41, 0.53, 0.62, 5.75, 0.98, 1.59, 450.0,
        13.67, 1.25, 1.92, 18.0, 94.0, 2.10, 1.79, 0.32, 0.73, 3.80, 1.23, 2.46, 630.0,
        12.37, 1.13, 2.16, 19.0, 87.0, 3.50, 3.10, 0.19, 1.87, 4.45, 1.22, 2.87, 420.0,
        12.17, 1.45, 2.53, 19.0, 104.0, 1.89, 1.75, 0.45, 1.03, 2.95, 1.45, 2.23, 355.0,
        12.37, 1.21, 2.56, 18.1, 98.0, 2.42, 2.65, 0.37, 2.08, 4.60, 1.19, 2.30, 678.0,
        13.11, 1.01, 1.70, 15.0, 78.0, 2.98, 3.18, 0.26, 2.28, 5.30, 1.12, 3.18, 502.0,
        12.37, 1.17, 1.92, 19.6, 78.0, 2.11, 2.00, 0.27, 1.04, 4.68, 1.12, 3.48, 510.0,
        13.34, 0.94, 2.36, 17.0, 110.0, 2.53, 1.30, 0.55, 0.42, 3.17, 1.02, 1.93, 750.0,
        12.21, 1.19, 1.75, 16.8, 151.0, 1.85, 1.28, 0.14, 2.50, 2.85, 1.28, 3.07, 718.0,
        12.29, 1.61, 2.21, 20.4, 103.0, 1.10, 1.02, 0.37, 1.46, 3.05, 0.906, 1.82, 870.0,
        13.86, 1.51, 2.67, 25.0, 86.0, 2.95, 2.86, 0.21, 1.87, 3.38, 1.36, 3.16, 410.0,
        13.49, 1.66, 2.24, 24.0, 87.0, 1.88, 1.84, 0.27, 1.03, 3.74, 0.98, 2.78, 472.0,
        12.99, 1.67, 2.60, 30.0, 139.0, 3.30, 2.89, 0.21, 1.96, 3.35, 1.31, 3.50, 985.0,
        11.96, 1.09, 2.30, 21.0, 101.0, 3.38, 2.14, 0.13, 1.65, 3.21, 0.99, 3.13, 886.0,
        11.66, 1.88, 1.92, 16.0, 97.0, 1.61, 1.57, 0.34, 1.15, 3.80, 1.23, 2.14, 428.0,
        13.03, 0.90, 1.71, 16.0, 86.0, 1.95, 2.03, 0.24, 1.46, 4.60, 1.19, 2.48, 392.0,
        11.84, 2.89, 2.23, 18.0, 112.0, 1.72, 1.32, 0.43, 0.95, 2.65, 0.96, 2.52, 500.0,
        12.33, 0.99, 1.95, 14.8, 136.0, 1.90, 1.85, 0.35, 2.76, 3.40, 1.06, 2.31, 750.0,
        12.70, 3.55, 2.36, 21.5, 106.0, 1.70, 1.20, 0.17, 0.84, 5.00, 0.78, 1.29, 600.0,
        12.00, 0.92, 2.00, 19.0, 86.0, 2.42, 2.26, 0.30, 1.43, 2.50, 1.38, 3.12, 278.0,
        12.72, 1.81, 2.20, 18.8, 86.0, 2.20, 2.53, 0.26, 1.77, 3.90, 1.16, 3.14, 714.0,
        12.08, 1.13, 2.51, 24.0, 78.0, 2.00, 1.58, 0.40, 1.40, 2.20, 1.31, 2.72, 630.0,
        13.05, 3.86, 2.32, 22.5, 85.0, 1.65, 1.59, 0.61, 1.62, 4.80, 0.84, 2.01, 515.0,
        11.84, 0.89, 2.58, 18.0, 94.0, 2.20, 2.21, 0.22, 2.35, 3.05, 0.79, 3.08, 520.0,
        12.67, 0.98, 2.24, 18.0, 99.0, 2.20, 1.94, 0.30, 1.46, 2.62, 1.23, 3.16, 450.0,
        12.16, 1.61, 2.31, 22.8, 90.0, 1.78, 1.69, 0.43, 1.56, 2.45, 1.33, 2.26, 495.0,
        11.65, 1.67, 2.62, 26.0, 88.0, 1.92, 1.61, 0.40, 1.34, 2.60, 1.36, 3.21, 562.0,
        11.64, 2.06, 2.46, 21.6, 84.0, 1.95, 1.69, 0.48, 1.35, 2.80, 1.00, 2.75, 680.0,
        12.08, 1.33, 2.30, 23.6, 70.0, 2.20, 1.59, 0.42, 1.38, 1.74, 1.07, 3.21, 625.0,
        12.08, 1.83, 2.32, 18.5, 81.0, 1.60, 1.50, 0.52, 1.64, 2.40, 1.08, 2.27, 480.0,
        12.00, 1.51, 2.42, 22.0, 86.0, 1.45, 1.25, 0.50, 1.63, 3.60, 1.05, 2.65, 450.0,
        12.69, 1.53, 2.26, 20.7, 80.0, 1.38, 1.46, 0.58, 1.62, 3.05, 0.96, 2.06, 495.0,
        12.29, 2.83, 2.22, 18.0, 88.0, 2.45, 2.25, 0.25, 1.99, 2.15, 1.07, 3.09, 290.0,
        11.62, 1.99, 2.28, 18.0, 98.0, 3.02, 2.26, 0.17, 1.35, 3.25, 1.16, 2.96, 345.0,
        12.47, 1.52, 2.20, 19.0, 162.0, 2.50, 2.27, 0.32, 3.28, 2.60, 1.16, 2.63, 937.0,
        11.81, 2.12, 2.74, 21.5, 134.0, 1.60, 0.99, 0.14, 1.56, 2.50, 0.95, 2.26, 625.0,
        12.29, 1.41, 1.98, 16.0, 85.0, 2.55, 2.50, 0.29, 1.77, 2.90, 1.23, 2.74, 428.0,
        12.37, 1.07, 2.10, 18.5, 88.0, 3.52, 3.75, 0.24, 1.95, 4.50, 1.04, 2.77, 660.0,
        12.29, 3.17, 2.21, 18.0, 88.0, 2.85, 2.99, 0.45, 2.81, 2.30, 1.42, 2.83, 406.0,
        12.08, 2.08, 1.70, 17.5, 97.0, 2.23, 2.17, 0.26, 1.40, 3.30, 1.27, 2.96, 710.0,
        12.60, 1.34, 1.90, 18.5, 88.0, 1.45, 1.36, 0.29, 1.35, 2.45, 1.04, 2.77, 562.0,
        12.34, 2.45, 2.46, 21.0, 98.0, 2.56, 2.11, 0.34, 1.31, 2.80, 1.28, 3.38, 980.0,
        11.82, 1.72, 1.88, 19.5, 86.0, 2.50, 1.64, 0.37, 1.42, 2.06, 0.94, 2.44, 415.0,
        12.51, 1.73, 1.98, 20.5, 85.0, 2.20, 1.92, 0.32, 1.48, 2.94, 1.04, 3.57, 672.0,
        12.42, 2.55, 2.27, 22.0, 90.0, 1.68, 1.84, 0.66, 1.42, 2.70, 0.86, 3.30, 315.0,
        12.25, 1.73, 2.12, 19.0, 80.0, 1.65, 2.03, 0.37, 1.63, 3.40, 1.00, 3.17, 510.0,
        12.72, 1.75, 2.28, 22.5, 84.0, 1.38, 1.76, 0.48, 1.63, 3.30, 0.88, 2.42, 488.0,
        12.22, 1.29, 1.94, 19.0, 92.0, 2.36, 2.04, 0.39, 2.08, 2.70, 0.86, 3.02, 312.0,
        11.61, 1.35, 2.70, 20.0, 94.0, 2.74, 2.92, 0.29, 2.49, 2.65, 0.96, 3.26, 680.0,
        11.46, 3.74, 1.82, 19.5, 107.0, 3.18, 2.58, 0.24, 3.58, 2.90, 0.75, 2.81, 562.0,
        12.52, 2.43, 2.17, 21.0, 88.0, 2.55, 2.27, 0.26, 1.22, 2.00, 0.90, 2.78, 325.0,
        11.76, 2.68, 2.92, 20.0, 103.0, 1.75, 2.03, 0.60, 1.05, 3.80, 1.23, 2.50, 607.0,
        11.41, 0.74, 2.50, 21.0, 88.0, 2.48, 2.01, 0.42, 1.44, 3.08, 1.10, 2.31, 434.0,
        12.08, 1.39, 2.50, 22.5, 84.0, 2.56, 2.29, 0.43, 1.04, 2.90, 0.93, 3.19, 385.0,
        11.03, 1.51, 2.20, 21.5, 85.0, 2.46, 2.17, 0.52, 2.01, 1.90, 1.71, 2.87, 407.0,
        11.82, 1.47, 1.99, 20.8, 86.0, 1.98, 1.60, 0.30, 1.53, 1.95, 0.95, 3.33, 495.0,
        12.42, 1.61, 2.19, 22.5, 108.0, 2.00, 2.09, 0.34, 1.61, 2.06, 1.06, 2.96, 345.0,
        12.77, 3.43, 1.98, 16.0, 80.0, 1.63, 1.25, 0.43, 0.83, 3.40, 0.70, 2.12, 372.0,
        12.00, 3.43, 2.00, 19.0, 87.0, 2.00, 1.64, 0.37, 1.87, 1.28, 0.93, 3.05, 564.0,
        11.45, 2.40, 2.42, 20.0, 96.0, 2.90, 2.79, 0.32, 1.83, 3.25, 0.80, 3.39, 625.0,
        11.56, 2.05, 3.23, 28.5, 119.0, 3.18, 5.08, 0.47, 1.87, 6.00, 0.93, 3.69, 465.0,
        12.42, 4.43, 2.73, 26.5, 102.0, 2.20, 2.13, 0.43, 1.71, 2.08, 0.92, 3.12, 365.0,
        13.05, 5.80, 2.13, 21.5, 86.0, 2.62, 2.65, 0.30, 2.01, 2.60, 0.73, 3.10, 380.0,
        11.87, 4.31, 2.39, 21.0, 82.0, 2.86, 3.03, 0.21, 2.91, 2.80, 0.75, 3.64, 380.0,
        12.07, 2.16, 2.17, 21.0, 85.0, 2.60, 2.65, 0.37, 1.35, 2.76, 0.86, 3.28, 378.0,
        12.43, 1.53, 2.29, 21.5, 86.0, 2.74, 3.15, 0.39, 1.77, 3.94, 0.69, 2.84, 352.0,
        11.79, 2.13, 2.78, 28.5, 92.0, 2.13, 2.24, 0.58, 1.76, 3.00, 0.97, 2.44, 466.0,
        12.37, 1.63, 2.30, 24.5, 88.0, 2.22, 2.45, 0.40, 1.90, 2.12, 0.89, 2.78, 342.0,
        12.85, 1.60, 2.52, 17.8, 95.0, 2.48, 2.37, 0.26, 1.46, 3.93, 1.09, 3.63, 1015.0,
        // Class 2 (48 samples)
        12.86, 1.35, 2.32, 18.0, 122.0, 1.51, 1.25, 0.21, 0.94, 4.10, 0.76, 1.29, 630.0,
        12.88, 2.99, 2.40, 20.0, 104.0, 1.30, 1.22, 0.24, 0.83, 5.40, 0.74, 1.42, 530.0,
        12.81, 2.31, 2.40, 24.0, 98.0, 1.15, 1.09, 0.27, 0.83, 5.70, 0.66, 1.36, 560.0,
        12.70, 3.55, 2.36, 21.5, 106.0, 1.70, 1.20, 0.17, 0.84, 5.00, 0.78, 1.29, 600.0,
        12.51, 1.24, 2.25, 17.5, 85.0, 2.00, 0.58, 0.60, 1.25, 5.45, 0.75, 1.51, 650.0,
        12.60, 2.46, 2.20, 18.5, 94.0, 1.62, 0.66, 0.63, 0.94, 7.10, 0.73, 1.58, 695.0,
        12.25, 4.72, 2.54, 21.0, 89.0, 1.38, 0.47, 0.53, 0.80, 3.85, 0.75, 1.27, 720.0,
        12.53, 5.51, 2.64, 25.0, 96.0, 1.79, 0.60, 0.63, 1.10, 5.00, 0.82, 1.69, 515.0,
        13.49, 3.59, 2.19, 19.5, 88.0, 1.62, 0.48, 0.58, 0.88, 5.70, 0.81, 1.82, 580.0,
        12.84, 2.96, 2.61, 24.0, 101.0, 2.32, 0.60, 0.53, 0.81, 4.92, 0.89, 2.15, 590.0,
        12.93, 2.81, 2.70, 21.0, 96.0, 1.54, 0.50, 0.53, 0.75, 4.60, 0.77, 2.31, 600.0,
        13.36, 2.56, 2.35, 20.0, 89.0, 1.40, 0.50, 0.37, 0.64, 5.60, 0.70, 2.47, 780.0,
        13.52, 3.17, 2.72, 23.5, 97.0, 1.55, 0.52, 0.50, 0.55, 4.35, 0.89, 2.06, 520.0,
        13.62, 4.95, 2.35, 20.0, 92.0, 2.00, 0.80, 0.47, 1.02, 4.40, 0.91, 2.05, 550.0,
        12.25, 3.88, 2.20, 18.5, 112.0, 1.38, 0.78, 0.29, 1.14, 8.21, 0.65, 2.00, 855.0,
        13.16, 3.57, 2.15, 21.0, 102.0, 1.50, 0.55, 0.43, 1.30, 4.00, 0.60, 1.68, 830.0,
        13.88, 5.04, 2.23, 20.0, 80.0, 0.98, 0.34, 0.40, 0.68, 4.90, 0.58, 1.33, 415.0,
        12.87, 4.61, 2.48, 21.5, 86.0, 1.70, 0.65, 0.47, 0.86, 7.65, 0.54, 1.86, 625.0,
        13.32, 3.24, 2.38, 21.5, 92.0, 1.93, 0.76, 0.45, 1.25, 8.42, 0.55, 1.62, 650.0,
        13.08, 3.90, 2.36, 21.5, 113.0, 1.41, 1.39, 0.34, 1.14, 9.40, 0.57, 1.33, 550.0,
        13.50, 3.12, 2.62, 24.0, 123.0, 1.40, 1.57, 0.22, 1.25, 8.60, 0.59, 1.30, 500.0,
        12.79, 2.67, 2.48, 22.0, 112.0, 1.48, 1.36, 0.24, 1.26, 10.80, 0.48, 1.47, 480.0,
        13.11, 1.90, 2.75, 25.5, 116.0, 2.20, 1.28, 0.26, 1.56, 7.10, 0.61, 1.33, 425.0,
        13.23, 3.30, 2.28, 18.5, 98.0, 1.80, 0.83, 0.61, 1.87, 10.52, 0.56, 1.51, 675.0,
        12.58, 1.29, 2.10, 20.0, 103.0, 1.48, 0.58, 0.53, 1.40, 7.60, 0.58, 1.55, 640.0,
        13.17, 5.19, 2.32, 22.0, 93.0, 1.74, 0.63, 0.61, 1.55, 7.90, 0.60, 1.48, 725.0,
        13.84, 4.12, 2.38, 19.5, 89.0, 1.80, 0.83, 0.48, 1.56, 9.01, 0.57, 1.64, 480.0,
        12.45, 3.03, 2.64, 27.0, 97.0, 1.90, 0.58, 0.63, 1.14, 7.50, 0.67, 1.73, 880.0,
        14.34, 1.68, 2.70, 25.0, 98.0, 2.80, 1.31, 0.53, 2.70, 13.00, 0.57, 1.96, 660.0,
        13.48, 1.67, 2.64, 22.5, 89.0, 2.60, 1.10, 0.52, 2.29, 11.75, 0.57, 1.78, 620.0,
        12.36, 3.83, 2.38, 21.0, 88.0, 2.30, 0.92, 0.50, 1.04, 7.65, 0.56, 1.58, 520.0,
        13.69, 3.26, 2.54, 20.0, 107.0, 1.83, 0.56, 0.50, 0.80, 5.88, 0.96, 1.82, 680.0,
        12.85, 3.27, 2.58, 22.0, 106.0, 1.65, 0.60, 0.60, 0.96, 5.58, 0.87, 2.11, 570.0,
        12.96, 3.45, 2.35, 18.5, 106.0, 1.39, 0.70, 0.40, 0.94, 5.28, 0.68, 1.75, 675.0,
        13.78, 2.76, 2.30, 22.0, 90.0, 1.35, 0.68, 0.41, 1.03, 9.58, 0.70, 1.68, 615.0,
        13.73, 4.36, 2.26, 22.5, 88.0, 1.28, 0.47, 0.52, 1.15, 6.62, 0.78, 1.75, 520.0,
        13.45, 3.70, 2.60, 23.0, 111.0, 1.70, 0.92, 0.43, 1.46, 10.68, 0.85, 1.56, 695.0,
        12.82, 3.37, 2.30, 19.5, 88.0, 1.48, 0.66, 0.40, 0.97, 10.26, 0.72, 1.75, 685.0,
        13.58, 2.58, 2.69, 24.5, 105.0, 1.55, 0.84, 0.39, 1.54, 8.66, 0.74, 1.80, 750.0,
        13.40, 4.60, 2.86, 25.0, 112.0, 1.98, 0.96, 0.27, 1.11, 8.50, 0.67, 1.92, 630.0,
        12.20, 3.03, 2.32, 19.0, 96.0, 1.25, 0.49, 0.40, 0.73, 5.50, 0.66, 1.83, 510.0,
        12.77, 2.39, 2.28, 19.5, 86.0, 1.39, 0.51, 0.48, 0.64, 9.899999, 0.57, 1.63, 470.0,
        14.16, 2.51, 2.48, 20.0, 91.0, 1.68, 0.70, 0.44, 1.24, 9.70, 0.62, 1.71, 660.0,
        13.71, 5.65, 2.45, 20.5, 95.0, 1.68, 0.61, 0.52, 1.06, 7.70, 0.64, 1.74, 740.0,
        13.40, 3.91, 2.48, 23.0, 102.0, 1.80, 0.75, 0.43, 1.41, 7.30, 0.70, 1.56, 750.0,
        13.27, 4.28, 2.26, 20.0, 120.0, 1.59, 0.69, 0.43, 1.35, 10.20, 0.59, 1.56, 835.0,
        13.17, 2.59, 2.37, 20.0, 120.0, 1.65, 0.68, 0.53, 1.46, 9.30, 0.60, 1.62, 840.0,
        14.13, 4.10, 2.74, 24.5, 96.0, 2.05, 0.76, 0.56, 1.35, 9.20, 0.61, 1.60, 560.0,
    ];

    #[rustfmt::skip]
    let targets: [f64; 178] = [
        // Class 0: 59 samples
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        // Class 1: 71 samples
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0,
        // Class 2: 48 samples
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];

    let x = Array2::from_shape_vec((178, 13), data.to_vec()).unwrap();
    let y = Array1::from_vec(targets.to_vec());

    let feature_names = vec![
        "alcohol".to_string(),
        "malic_acid".to_string(),
        "ash".to_string(),
        "alcalinity_of_ash".to_string(),
        "magnesium".to_string(),
        "total_phenols".to_string(),
        "flavanoids".to_string(),
        "nonflavanoid_phenols".to_string(),
        "proanthocyanins".to_string(),
        "color_intensity".to_string(),
        "hue".to_string(),
        "od280/od315_of_diluted_wines".to_string(),
        "proline".to_string(),
    ];
    let target_names = vec![
        "class_0".to_string(),
        "class_1".to_string(),
        "class_2".to_string(),
    ];

    let dataset = Dataset::new(x, y)
        .with_feature_names(feature_names.clone())
        .with_target_names(target_names.clone());

    let info = DatasetInfo::new("wine", Task::Classification, 178, 13)
        .with_description(
            "The Wine dataset contains chemical analysis results of wines grown in the same \
             region in Italy but derived from three different cultivars. The dataset has \
             178 samples with 13 features each, representing various chemical properties."
        )
        .with_n_classes(3)
        .with_feature_names(feature_names)
        .with_target_names(target_names)
        .with_source("Forina, M. et al. (1988). PARVUS - An Extendible Package for Data Exploration.")
        .with_url("https://archive.ics.uci.edu/ml/datasets/Wine")
        .with_license("Public Domain");

    (dataset, info)
}

/// Load the Diabetes dataset for regression.
///
/// This is a subset (260 samples) of the full Diabetes dataset with 10 baseline
/// variables (age, sex, BMI, blood pressure, and six blood serum measurements).
/// The target is a quantitative measure of disease progression one year after baseline.
///
/// # Features
///
/// 1. age - Age in years
/// 2. sex - Sex
/// 3. bmi - Body mass index
/// 4. bp - Average blood pressure
/// 5. s1 - tc, total serum cholesterol
/// 6. s2 - ldl, low-density lipoproteins
/// 7. s3 - hdl, high-density lipoproteins
/// 8. s4 - tch, total cholesterol / HDL
/// 9. s5 - ltg, log of serum triglycerides level
/// 10. s6 - glu, blood sugar level
///
/// Note: Features are standardized (mean=0, std=1).
///
/// # Returns
///
/// A tuple of (Dataset, DatasetInfo).
///
/// # Example
///
/// ```
/// use ferroml_core::datasets::load_diabetes;
///
/// let (dataset, info) = load_diabetes();
/// assert_eq!(dataset.n_samples(), 260);
/// assert_eq!(dataset.n_features(), 10);
/// ```
///
/// # References
///
/// Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004).
/// "Least Angle Regression." Annals of Statistics, 32(2), 407-499.
pub fn load_diabetes() -> (Dataset, DatasetInfo) {
    // Diabetes dataset: 442 samples, 10 features, regression
    // Features are already standardized
    // Due to size, we include a representative subset here
    // This is the actual sklearn diabetes dataset values (first 100 samples shown, truncated for brevity)
    // In production, you'd want to include all 442 samples

    #[rustfmt::skip]
    let data: [f64; 2600] = include!("diabetes_data.inc");

    #[rustfmt::skip]
    let targets: [f64; 260] = include!("diabetes_targets.inc");

    let x = Array2::from_shape_vec((260, 10), data.to_vec()).unwrap();
    let y = Array1::from_vec(targets.to_vec());

    let feature_names = vec![
        "age".to_string(),
        "sex".to_string(),
        "bmi".to_string(),
        "bp".to_string(),
        "s1".to_string(),
        "s2".to_string(),
        "s3".to_string(),
        "s4".to_string(),
        "s5".to_string(),
        "s6".to_string(),
    ];

    let dataset = Dataset::new(x, y).with_feature_names(feature_names.clone());

    let info = DatasetInfo::new("diabetes", Task::Regression, 260, 10)
        .with_description(
            "The Diabetes dataset contains 260 samples (subset) with 10 baseline variables \
             (age, sex, BMI, blood pressure, and six blood serum measurements). \
             The target is a quantitative measure of disease progression one year \
             after baseline. Features are standardized to have zero mean and unit variance."
        )
        .with_feature_names(feature_names)
        .with_source("Efron, B., Hastie, T., Johnstone, I. and Tibshirani, R. (2004). Least Angle Regression. Annals of Statistics, 32(2), 407-499.")
        .with_url("https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html")
        .with_license("Public Domain");

    (dataset, info)
}

/// Load the Linnerud dataset for multi-output regression.
///
/// The Linnerud dataset is a small multi-output regression dataset with
/// physiological measurements and exercise data for 20 middle-aged men.
///
/// # Features (Exercise measurements)
///
/// 1. Chins - Number of chin-ups
/// 2. Situps - Number of sit-ups
/// 3. Jumps - Number of jumping jacks
///
/// # Targets (Physiological measurements)
///
/// 1. Weight - Body weight (kg)
/// 2. Waist - Waist circumference (cm)
/// 3. Pulse - Resting heart rate (bpm)
///
/// Note: This returns only the first target (Weight) as FerroML currently
/// supports single-target regression. For multi-target, load all targets separately.
///
/// # Returns
///
/// A tuple of (Dataset, DatasetInfo).
///
/// # Example
///
/// ```
/// use ferroml_core::datasets::load_linnerud;
///
/// let (dataset, info) = load_linnerud();
/// assert_eq!(dataset.n_samples(), 20);
/// assert_eq!(dataset.n_features(), 3);
/// ```
///
/// # References
///
/// Tenenhaus, M. (1998). La Régression PLS. Technip, Paris.
pub fn load_linnerud() -> (Dataset, DatasetInfo) {
    // Linnerud dataset: 20 samples, 3 features, 3 targets (we use Weight as primary)
    #[rustfmt::skip]
    let data: [f64; 60] = [
        5.0, 162.0, 60.0,
        2.0, 110.0, 60.0,
        12.0, 101.0, 101.0,
        12.0, 105.0, 37.0,
        13.0, 155.0, 58.0,
        4.0, 101.0, 42.0,
        8.0, 101.0, 38.0,
        6.0, 125.0, 40.0,
        15.0, 200.0, 40.0,
        17.0, 251.0, 250.0,
        17.0, 120.0, 38.0,
        13.0, 210.0, 115.0,
        14.0, 215.0, 105.0,
        1.0, 50.0, 50.0,
        6.0, 70.0, 31.0,
        12.0, 210.0, 120.0,
        4.0, 60.0, 25.0,
        11.0, 230.0, 80.0,
        15.0, 225.0, 73.0,
        2.0, 110.0, 43.0,
    ];

    // Targets: Weight, Waist, Pulse (we use Weight as primary target)
    #[rustfmt::skip]
    let targets: [f64; 20] = [
        191.0, 189.0, 193.0, 162.0, 189.0,
        182.0, 211.0, 167.0, 176.0, 154.0,
        169.0, 166.0, 154.0, 247.0, 193.0,
        202.0, 176.0, 157.0, 156.0, 138.0,
    ];

    let x = Array2::from_shape_vec((20, 3), data.to_vec()).unwrap();
    let y = Array1::from_vec(targets.to_vec());

    let feature_names = vec![
        "chins".to_string(),
        "situps".to_string(),
        "jumps".to_string(),
    ];

    let dataset = Dataset::new(x, y).with_feature_names(feature_names.clone());

    let info = DatasetInfo::new("linnerud", Task::Regression, 20, 3)
        .with_description(
            "The Linnerud dataset is a small multi-output regression dataset with \
             physiological measurements and exercise data for 20 middle-aged men. \
             This version uses body weight (kg) as the target variable. \
             Features are exercise measurements: chin-ups, sit-ups, and jumping jacks."
        )
        .with_feature_names(feature_names)
        .with_source("Tenenhaus, M. (1998). La Régression PLS. Technip, Paris.")
        .with_license("Public Domain");

    (dataset, info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_iris() {
        let (dataset, info) = load_iris();
        assert_eq!(dataset.n_samples(), 150);
        assert_eq!(dataset.n_features(), 4);
        assert_eq!(info.n_classes, Some(3));
        assert_eq!(info.task, Task::Classification);
        assert_eq!(dataset.unique_classes().len(), 3);

        // Check class distribution
        let counts = dataset.class_counts();
        assert_eq!(counts.get(&0), Some(&50));
        assert_eq!(counts.get(&1), Some(&50));
        assert_eq!(counts.get(&2), Some(&50));

        // Check feature names
        assert!(dataset.feature_names().is_some());
        assert_eq!(dataset.feature_names().unwrap().len(), 4);
    }

    #[test]
    fn test_load_wine() {
        let (dataset, info) = load_wine();
        assert_eq!(dataset.n_samples(), 178);
        assert_eq!(dataset.n_features(), 13);
        assert_eq!(info.n_classes, Some(3));
        assert_eq!(info.task, Task::Classification);
        assert_eq!(dataset.unique_classes().len(), 3);

        // Check class distribution
        let counts = dataset.class_counts();
        assert_eq!(counts.get(&0), Some(&59));
        assert_eq!(counts.get(&1), Some(&71));
        assert_eq!(counts.get(&2), Some(&48));
    }

    #[test]
    fn test_load_diabetes() {
        let (dataset, info) = load_diabetes();
        assert_eq!(dataset.n_samples(), 260);
        assert_eq!(dataset.n_features(), 10);
        assert_eq!(info.task, Task::Regression);

        // Check that features are standardized (approximately zero mean)
        let x = dataset.x();
        for col_idx in 0..10 {
            let col = x.column(col_idx);
            let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
            assert!(mean.abs() < 0.15, "Feature {} has mean {}", col_idx, mean);
        }
    }

    #[test]
    fn test_load_linnerud() {
        let (dataset, info) = load_linnerud();
        assert_eq!(dataset.n_samples(), 20);
        assert_eq!(dataset.n_features(), 3);
        assert_eq!(info.task, Task::Regression);

        // Check feature ranges are reasonable
        let x = dataset.x();
        // Chins: 1-17
        assert!(x.column(0).iter().all(|&v| v >= 1.0 && v <= 20.0));
        // Situps: 50-251
        assert!(x.column(1).iter().all(|&v| v >= 50.0 && v <= 260.0));
    }

    #[test]
    fn test_iris_data_validity() {
        let (dataset, _) = load_iris();
        let x = dataset.x();

        // Sepal length: 4.3 - 7.9 cm
        assert!(x.column(0).iter().all(|&v| v >= 4.0 && v <= 8.0));
        // Sepal width: 2.0 - 4.4 cm
        assert!(x.column(1).iter().all(|&v| v >= 2.0 && v <= 4.5));
        // Petal length: 1.0 - 6.9 cm
        assert!(x.column(2).iter().all(|&v| v >= 1.0 && v <= 7.0));
        // Petal width: 0.1 - 2.5 cm
        assert!(x.column(3).iter().all(|&v| v >= 0.1 && v <= 2.6));
    }

    #[test]
    fn test_wine_data_validity() {
        let (dataset, _) = load_wine();
        let x = dataset.x();

        // Alcohol: 11-15%
        assert!(x.column(0).iter().all(|&v| v >= 11.0 && v <= 15.0));
        // Proline: 278-1680
        assert!(x.column(12).iter().all(|&v| v >= 270.0 && v <= 1700.0));
    }

    #[test]
    fn test_diabetes_target_range() {
        let (dataset, _) = load_diabetes();
        let y = dataset.y();

        // Target values are disease progression measures (typically 25-346)
        assert!(y.iter().all(|&v| v >= 20.0 && v <= 360.0));
    }
}

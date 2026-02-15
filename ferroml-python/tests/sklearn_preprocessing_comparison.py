"""
FerroML vs sklearn Preprocessing Comparison
============================================

This script compares FerroML preprocessing transformers with their sklearn equivalents.
"""

import numpy as np
from typing import Tuple, Dict, Any

# sklearn imports
from sklearn.preprocessing import (
    StandardScaler as SklearnStandardScaler,
    MinMaxScaler as SklearnMinMaxScaler,
    RobustScaler as SklearnRobustScaler,
    MaxAbsScaler as SklearnMaxAbsScaler,
    OneHotEncoder as SklearnOneHotEncoder,
    OrdinalEncoder as SklearnOrdinalEncoder,
    LabelEncoder as SklearnLabelEncoder,
)
from sklearn.impute import SimpleImputer as SklearnSimpleImputer

# FerroML imports
from ferroml.preprocessing import (
    StandardScaler as FerroStandardScaler,
    MinMaxScaler as FerroMinMaxScaler,
    RobustScaler as FerroRobustScaler,
    MaxAbsScaler as FerroMaxAbsScaler,
    OneHotEncoder as FerroOneHotEncoder,
    OrdinalEncoder as FerroOrdinalEncoder,
    LabelEncoder as FerroLabelEncoder,
    SimpleImputer as FerroSimpleImputer,
)


def create_test_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create test datasets for preprocessing comparisons."""
    # Standard continuous data
    X_continuous = np.array([
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
        [4.0, 40.0, 400.0],
        [5.0, 50.0, 500.0],
    ], dtype=np.float64)

    # Categorical data (integer-encoded)
    X_categorical = np.array([
        [0.0],
        [1.0],
        [2.0],
        [1.0],
        [0.0],
    ], dtype=np.float64)

    # Data with missing values
    X_missing = np.array([
        [1.0, 2.0],
        [np.nan, 4.0],
        [3.0, np.nan],
        [4.0, 6.0],
        [5.0, 8.0],
    ], dtype=np.float64)

    return X_continuous, X_categorical, X_missing


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, name: str) -> Dict[str, Any]:
    """Compare two arrays and return comparison stats."""
    if arr1.shape != arr2.shape:
        return {
            "name": name,
            "match": False,
            "error": f"Shape mismatch: {arr1.shape} vs {arr2.shape}",
        }

    abs_diff = np.abs(arr1 - arr2)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    # Consider match if max difference is below floating point tolerance
    is_exact = max_diff < 1e-14
    is_close = max_diff < 1e-10

    return {
        "name": name,
        "match": is_exact,
        "close": is_close,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "sklearn_values": arr1.flatten()[:6].tolist(),  # First 6 values
        "ferroml_values": arr2.flatten()[:6].tolist(),
    }


def test_standard_scaler(X: np.ndarray) -> Dict[str, Any]:
    """Compare StandardScaler implementations."""
    print("\n" + "="*60)
    print("STANDARD SCALER COMPARISON")
    print("="*60)

    # sklearn
    sk_scaler = SklearnStandardScaler()
    sk_result = sk_scaler.fit_transform(X)

    # FerroML
    ferro_scaler = FerroStandardScaler()
    ferro_scaler.fit(X)
    ferro_result = ferro_scaler.transform(X)

    results = {}

    # Compare means
    print("\n--- Fitted Parameters ---")
    print(f"sklearn mean_:    {sk_scaler.mean_}")
    print(f"FerroML mean_:    {ferro_scaler.mean_}")
    results["mean"] = compare_arrays(sk_scaler.mean_, ferro_scaler.mean_, "mean_")

    # Compare scales (std)
    print(f"sklearn scale_:   {sk_scaler.scale_}")
    print(f"FerroML scale_:   {ferro_scaler.scale_}")
    results["scale"] = compare_arrays(sk_scaler.scale_, ferro_scaler.scale_, "scale_")

    # Compare transformed data
    print("\n--- Transformed Data ---")
    print(f"sklearn result:\n{sk_result}")
    print(f"FerroML result:\n{ferro_result}")
    results["transform"] = compare_arrays(sk_result, ferro_result, "transform")

    # Compare inverse transform
    sk_inverse = sk_scaler.inverse_transform(sk_result)
    ferro_inverse = ferro_scaler.inverse_transform(ferro_result)
    print("\n--- Inverse Transform ---")
    print(f"sklearn inverse (first row):  {sk_inverse[0]}")
    print(f"FerroML inverse (first row):  {ferro_inverse[0]}")
    results["inverse"] = compare_arrays(sk_inverse, ferro_inverse, "inverse_transform")

    return results


def test_minmax_scaler(X: np.ndarray) -> Dict[str, Any]:
    """Compare MinMaxScaler implementations."""
    print("\n" + "="*60)
    print("MINMAX SCALER COMPARISON")
    print("="*60)

    # sklearn
    sk_scaler = SklearnMinMaxScaler()
    sk_result = sk_scaler.fit_transform(X)

    # FerroML
    ferro_scaler = FerroMinMaxScaler()
    ferro_scaler.fit(X)
    ferro_result = ferro_scaler.transform(X)

    results = {}

    # Compare data_min_
    print("\n--- Fitted Parameters ---")
    print(f"sklearn data_min_:   {sk_scaler.data_min_}")
    print(f"FerroML data_min_:   {ferro_scaler.data_min_}")
    results["data_min"] = compare_arrays(sk_scaler.data_min_, ferro_scaler.data_min_, "data_min_")

    # Compare data_max_
    print(f"sklearn data_max_:   {sk_scaler.data_max_}")
    print(f"FerroML data_max_:   {ferro_scaler.data_max_}")
    results["data_max"] = compare_arrays(sk_scaler.data_max_, ferro_scaler.data_max_, "data_max_")

    # Compare data_range_
    print(f"sklearn data_range_: {sk_scaler.data_range_}")
    print(f"FerroML data_range_: {ferro_scaler.data_range_}")
    results["data_range"] = compare_arrays(sk_scaler.data_range_, ferro_scaler.data_range_, "data_range_")

    # Compare transformed data
    print("\n--- Transformed Data ---")
    print(f"sklearn result:\n{sk_result}")
    print(f"FerroML result:\n{ferro_result}")
    results["transform"] = compare_arrays(sk_result, ferro_result, "transform")

    # Compare inverse transform
    sk_inverse = sk_scaler.inverse_transform(sk_result)
    ferro_inverse = ferro_scaler.inverse_transform(ferro_result)
    results["inverse"] = compare_arrays(sk_inverse, ferro_inverse, "inverse_transform")

    return results


def test_robust_scaler(X: np.ndarray) -> Dict[str, Any]:
    """Compare RobustScaler implementations."""
    print("\n" + "="*60)
    print("ROBUST SCALER COMPARISON")
    print("="*60)

    # sklearn
    sk_scaler = SklearnRobustScaler()
    sk_result = sk_scaler.fit_transform(X)

    # FerroML
    ferro_scaler = FerroRobustScaler()
    ferro_scaler.fit(X)
    ferro_result = ferro_scaler.transform(X)

    results = {}

    # Compare center_ (median)
    print("\n--- Fitted Parameters ---")
    print(f"sklearn center_:  {sk_scaler.center_}")
    print(f"FerroML center_:  {ferro_scaler.center_}")
    results["center"] = compare_arrays(sk_scaler.center_, ferro_scaler.center_, "center_")

    # Compare scale_ (IQR)
    print(f"sklearn scale_:   {sk_scaler.scale_}")
    print(f"FerroML scale_:   {ferro_scaler.scale_}")
    results["scale"] = compare_arrays(sk_scaler.scale_, ferro_scaler.scale_, "scale_")

    # Compare transformed data
    print("\n--- Transformed Data ---")
    print(f"sklearn result:\n{sk_result}")
    print(f"FerroML result:\n{ferro_result}")
    results["transform"] = compare_arrays(sk_result, ferro_result, "transform")

    # Compare inverse transform
    sk_inverse = sk_scaler.inverse_transform(sk_result)
    ferro_inverse = ferro_scaler.inverse_transform(ferro_result)
    results["inverse"] = compare_arrays(sk_inverse, ferro_inverse, "inverse_transform")

    return results


def test_maxabs_scaler(X: np.ndarray) -> Dict[str, Any]:
    """Compare MaxAbsScaler implementations."""
    print("\n" + "="*60)
    print("MAXABS SCALER COMPARISON")
    print("="*60)

    # Use data with negative values
    X_signed = X - X.mean(axis=0)  # Center to get negative values

    # sklearn
    sk_scaler = SklearnMaxAbsScaler()
    sk_result = sk_scaler.fit_transform(X_signed)

    # FerroML
    ferro_scaler = FerroMaxAbsScaler()
    ferro_scaler.fit(X_signed)
    ferro_result = ferro_scaler.transform(X_signed)

    results = {}

    # Compare max_abs_
    print("\n--- Fitted Parameters ---")
    print(f"sklearn max_abs_:  {sk_scaler.max_abs_}")
    print(f"FerroML max_abs_:  {ferro_scaler.max_abs_}")
    results["max_abs"] = compare_arrays(sk_scaler.max_abs_, ferro_scaler.max_abs_, "max_abs_")

    # Compare transformed data
    print("\n--- Transformed Data ---")
    print(f"sklearn result:\n{sk_result}")
    print(f"FerroML result:\n{ferro_result}")
    results["transform"] = compare_arrays(sk_result, ferro_result, "transform")

    # Compare inverse transform
    sk_inverse = sk_scaler.inverse_transform(sk_result)
    ferro_inverse = ferro_scaler.inverse_transform(ferro_result)
    results["inverse"] = compare_arrays(sk_inverse, ferro_inverse, "inverse_transform")

    return results


def test_onehot_encoder(X_cat: np.ndarray) -> Dict[str, Any]:
    """Compare OneHotEncoder implementations."""
    print("\n" + "="*60)
    print("ONEHOT ENCODER COMPARISON")
    print("="*60)

    # sklearn (sparse=False for dense output)
    sk_encoder = SklearnOneHotEncoder(sparse_output=False)
    sk_result = sk_encoder.fit_transform(X_cat)

    # FerroML
    ferro_encoder = FerroOneHotEncoder()
    ferro_encoder.fit(X_cat)
    ferro_result = ferro_encoder.transform(X_cat)

    results = {}

    # Compare categories
    print("\n--- Fitted Parameters ---")
    print(f"sklearn categories_:  {sk_encoder.categories_}")
    print(f"FerroML categories_:  {ferro_encoder.categories_}")

    # Compare transformed data
    print("\n--- Transformed Data ---")
    print(f"sklearn result:\n{sk_result}")
    print(f"FerroML result:\n{ferro_result}")
    results["transform"] = compare_arrays(sk_result, ferro_result, "transform")

    return results


def test_ordinal_encoder(X_cat: np.ndarray) -> Dict[str, Any]:
    """Compare OrdinalEncoder implementations."""
    print("\n" + "="*60)
    print("ORDINAL ENCODER COMPARISON")
    print("="*60)

    # sklearn
    sk_encoder = SklearnOrdinalEncoder()
    sk_result = sk_encoder.fit_transform(X_cat)

    # FerroML
    ferro_encoder = FerroOrdinalEncoder()
    ferro_encoder.fit(X_cat)
    ferro_result = ferro_encoder.transform(X_cat)

    results = {}

    # Compare categories
    print("\n--- Fitted Parameters ---")
    print(f"sklearn categories_:  {sk_encoder.categories_}")
    print(f"FerroML categories_:  {ferro_encoder.categories_}")

    # Compare transformed data
    print("\n--- Transformed Data ---")
    print(f"sklearn result:\n{sk_result}")
    print(f"FerroML result:\n{ferro_result}")
    results["transform"] = compare_arrays(sk_result, ferro_result, "transform")

    return results


def test_label_encoder() -> Dict[str, Any]:
    """Compare LabelEncoder implementations."""
    print("\n" + "="*60)
    print("LABEL ENCODER COMPARISON")
    print("="*60)

    # Test data - 1D array
    y = np.array([2.0, 0.0, 1.0, 2.0, 1.0, 0.0])

    # sklearn (works with int, so we need to handle this carefully)
    # sklearn LabelEncoder works differently - it sorts classes
    sk_encoder = SklearnLabelEncoder()
    sk_result = sk_encoder.fit_transform(y.astype(int))

    # FerroML
    ferro_encoder = FerroLabelEncoder()
    ferro_encoder.fit(y)
    ferro_result = ferro_encoder.transform(y)

    results = {}

    print("\n--- Fitted Parameters ---")
    print(f"sklearn classes_:  {sk_encoder.classes_}")
    print(f"FerroML classes_:  {ferro_encoder.classes_}")

    print("\n--- Transformed Data ---")
    print(f"sklearn result:  {sk_result}")
    print(f"FerroML result:  {ferro_result}")

    # Note: sklearn and FerroML may assign different integer codes if they
    # process classes in different orders. What matters is consistency.
    # Let's check if both produce consistent mappings
    sk_unique_pairs = list(zip(y, sk_result.astype(float)))
    ferro_unique_pairs = list(zip(y, ferro_result))

    # Check consistency: same input should always map to same output
    sk_consistent = len(set(sk_unique_pairs)) == len(sk_encoder.classes_)
    ferro_consistent = len(set(ferro_unique_pairs)) == len(ferro_encoder.classes_)

    results["consistency"] = {
        "sklearn_consistent": sk_consistent,
        "ferroml_consistent": ferro_consistent,
        "both_consistent": sk_consistent and ferro_consistent,
    }

    print(f"\nsklearn consistent mapping: {sk_consistent}")
    print(f"FerroML consistent mapping: {ferro_consistent}")

    return results


def test_simple_imputer(X_missing: np.ndarray) -> Dict[str, Any]:
    """Compare SimpleImputer implementations."""
    print("\n" + "="*60)
    print("SIMPLE IMPUTER COMPARISON (mean strategy)")
    print("="*60)

    # sklearn
    sk_imputer = SklearnSimpleImputer(strategy='mean')
    sk_result = sk_imputer.fit_transform(X_missing)

    # FerroML
    ferro_imputer = FerroSimpleImputer(strategy='mean')
    ferro_imputer.fit(X_missing)
    ferro_result = ferro_imputer.transform(X_missing)

    results = {}

    print("\n--- Input Data (with NaN) ---")
    print(X_missing)

    print("\n--- Fitted Parameters ---")
    print(f"sklearn statistics_:  {sk_imputer.statistics_}")
    print(f"FerroML statistics_:  {ferro_imputer.statistics_}")
    results["statistics"] = compare_arrays(sk_imputer.statistics_, ferro_imputer.statistics_, "statistics_")

    print("\n--- Transformed Data ---")
    print(f"sklearn result:\n{sk_result}")
    print(f"FerroML result:\n{ferro_result}")
    results["transform"] = compare_arrays(sk_result, ferro_result, "transform")

    # Test median strategy
    print("\n" + "-"*40)
    print("SIMPLE IMPUTER (median strategy)")
    print("-"*40)

    sk_imputer_med = SklearnSimpleImputer(strategy='median')
    sk_result_med = sk_imputer_med.fit_transform(X_missing)

    ferro_imputer_med = FerroSimpleImputer(strategy='median')
    ferro_imputer_med.fit(X_missing)
    ferro_result_med = ferro_imputer_med.transform(X_missing)

    print(f"sklearn statistics (median):  {sk_imputer_med.statistics_}")
    print(f"FerroML statistics (median):  {ferro_imputer_med.statistics_}")
    results["median_statistics"] = compare_arrays(sk_imputer_med.statistics_, ferro_imputer_med.statistics_, "median_statistics")
    results["median_transform"] = compare_arrays(sk_result_med, ferro_result_med, "median_transform")

    return results


def generate_report(all_results: Dict[str, Dict]) -> str:
    """Generate a summary report of all comparisons."""
    report = []
    report.append("\n" + "="*70)
    report.append("FERROML vs SKLEARN PREPROCESSING COMPARISON REPORT")
    report.append("="*70)

    summary_table = []
    summary_table.append("\n{:<25} {:<15} {:<15} {:<20}".format(
        "Preprocessor", "Component", "Status", "Max Diff"))
    summary_table.append("-"*75)

    for preprocessor, results in all_results.items():
        for component, data in results.items():
            if isinstance(data, dict) and "max_diff" in data:
                if data["match"]:
                    status = "EXACT MATCH"
                elif data["close"]:
                    status = "CLOSE"
                else:
                    status = "DIFFERS"

                max_diff = f"{data['max_diff']:.2e}" if isinstance(data['max_diff'], float) else str(data['max_diff'])
                summary_table.append("{:<25} {:<15} {:<15} {:<20}".format(
                    preprocessor, component, status, max_diff))
            elif isinstance(data, dict) and "match" in data:
                status = "MATCH" if data["match"] else "DIFFERS"
                summary_table.append("{:<25} {:<15} {:<15} {:<20}".format(
                    preprocessor, component, status, "N/A"))

    report.extend(summary_table)

    # Summary counts
    exact_matches = 0
    close_matches = 0
    differences = 0

    for preprocessor, results in all_results.items():
        for component, data in results.items():
            if isinstance(data, dict) and "match" in data:
                if data.get("match", False):
                    exact_matches += 1
                elif data.get("close", False):
                    close_matches += 1
                else:
                    differences += 1

    report.append("\n" + "-"*70)
    report.append("SUMMARY")
    report.append("-"*70)
    report.append(f"Exact matches (diff < 1e-14):  {exact_matches}")
    report.append(f"Close matches (diff < 1e-10):  {close_matches}")
    report.append(f"Differences:                   {differences}")
    report.append("")

    if exact_matches + close_matches > 0 and differences == 0:
        report.append("RESULT: All FerroML preprocessors match sklearn within tolerance!")
    elif differences > 0:
        report.append(f"RESULT: {differences} component(s) have differences exceeding tolerance.")

    return "\n".join(report)


def main():
    """Run all preprocessing comparisons."""
    print("FerroML vs sklearn Preprocessing Comparison")
    print("="*60)

    # Create test data
    X_continuous, X_categorical, X_missing = create_test_data()

    print("\n--- Test Data ---")
    print(f"Continuous data shape: {X_continuous.shape}")
    print(f"Categorical data shape: {X_categorical.shape}")
    print(f"Missing data shape: {X_missing.shape}")

    all_results = {}

    # Run comparisons
    all_results["StandardScaler"] = test_standard_scaler(X_continuous)
    all_results["MinMaxScaler"] = test_minmax_scaler(X_continuous)
    all_results["RobustScaler"] = test_robust_scaler(X_continuous)
    all_results["MaxAbsScaler"] = test_maxabs_scaler(X_continuous)
    all_results["OneHotEncoder"] = test_onehot_encoder(X_categorical)
    all_results["OrdinalEncoder"] = test_ordinal_encoder(X_categorical)
    all_results["LabelEncoder"] = test_label_encoder()
    all_results["SimpleImputer"] = test_simple_imputer(X_missing)

    # Generate and print report
    report = generate_report(all_results)
    print(report)

    return all_results


if __name__ == "__main__":
    main()

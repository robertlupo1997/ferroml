"""Auto-pick the best cross-validation strategy for a dataset.

Usage: Claude calls this to recommend CV strategy before model comparison.
Output: Recommended strategy with reasoning and ready-to-use code snippet.
"""
from __future__ import annotations

import numpy as np


def advise(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    has_time_column: bool = False,
) -> dict:
    """Recommend a cross-validation strategy.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target array.
    task : str
        "regression" or "classification".
    has_time_column : bool
        Whether the data has a temporal ordering.

    Returns
    -------
    dict with keys: strategy_name, n_folds, reasoning, code_snippet,
    data_characteristics
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    # Data characteristics
    characteristics: dict = {
        "n_samples": n_samples,
        "n_features": n_features,
        "task": task,
        "has_time_column": has_time_column,
    }

    if task == "classification":
        unique, counts = np.unique(y, return_counts=True)
        n_classes = len(unique)
        min_class_count = int(np.min(counts))
        max_class_count = int(np.max(counts))
        imbalance_ratio = round(max_class_count / max(min_class_count, 1), 2)
        characteristics["n_classes"] = n_classes
        characteristics["min_class_count"] = min_class_count
        characteristics["imbalance_ratio"] = imbalance_ratio
        is_imbalanced = imbalance_ratio > 3.0
    else:
        is_imbalanced = False

    reasons: list[str] = []

    # --- Decision logic ---

    # Time series: always walk-forward
    if has_time_column:
        n_folds = 5
        strategy = "TimeSeriesSplit"
        reasons.append("Data has temporal ordering — standard k-fold would leak future information.")
        reasons.append("Walk-forward validation respects time: train on past, test on future.")
        if n_samples < 500:
            n_folds = 3
            reasons.append(f"Small dataset ({n_samples} rows) — using 3 folds to keep test sets meaningful.")
        snippet = _time_series_snippet(n_folds)

    # Very small dataset
    elif n_samples < 100:
        strategy = "LeaveOneOut"
        n_folds = n_samples
        reasons.append(f"Very small dataset ({n_samples} rows) — LeaveOneOut maximizes training data.")
        reasons.append("Each sample serves as a test set once, giving the least-biased estimate.")
        reasons.append("Warning: high variance in the estimate and slow for large models.")
        snippet = _loo_snippet()

    elif n_samples < 200:
        strategy = "RepeatedKFold"
        n_folds = 5
        n_repeats = 10
        reasons.append(f"Small dataset ({n_samples} rows) — repeated k-fold reduces variance of the estimate.")
        reasons.append(f"5-fold x 10 repeats = 50 evaluations for a more stable score.")
        if is_imbalanced:
            strategy = "RepeatedStratifiedKFold"
            reasons.append(f"Imbalanced classes (ratio {characteristics['imbalance_ratio']}) — using stratified folds.")
        snippet = _repeated_snippet(n_folds, n_repeats, stratified=is_imbalanced)

    # Classification with imbalance
    elif task == "classification" and is_imbalanced:
        strategy = "StratifiedKFold"
        n_folds = 5
        reasons.append(f"Imbalanced classes (ratio {characteristics['imbalance_ratio']}) — stratified folds preserve class distribution.")
        reasons.append("Prevents folds where minority class is absent or over-represented.")
        if n_samples > 10000:
            n_folds = 5
            reasons.append("Large dataset — 5 folds is sufficient.")
        snippet = _stratified_snippet(n_folds)

    # Standard classification
    elif task == "classification":
        strategy = "StratifiedKFold"
        n_folds = 10 if n_samples < 5000 else 5
        reasons.append("Classification task — stratified folds preserve class ratios.")
        reasons.append(f"{'10-fold for thorough evaluation.' if n_folds == 10 else '5-fold for efficiency with large data.'}")
        snippet = _stratified_snippet(n_folds)

    # Large regression
    elif n_samples > 10000:
        strategy = "KFold"
        n_folds = 5
        reasons.append(f"Large dataset ({n_samples} rows) — 5-fold gives reliable estimates efficiently.")
        reasons.append("Each fold has ~{} test samples, plenty for stable metrics.".format(n_samples // 5))
        snippet = _kfold_snippet(n_folds)

    # Medium regression
    else:
        strategy = "KFold"
        n_folds = 10
        reasons.append(f"Medium dataset ({n_samples} rows) — 10-fold balances bias and variance.")
        reasons.append("Standard recommendation for datasets between 200 and 10,000 samples.")
        snippet = _kfold_snippet(n_folds)

    return {
        "strategy_name": strategy,
        "n_folds": n_folds,
        "reasoning": reasons,
        "code_snippet": snippet,
        "data_characteristics": characteristics,
    }


def _kfold_snippet(n_folds: int) -> str:
    return f"""\
# Standard K-Fold Cross-Validation
rng = np.random.RandomState(42)
indices = rng.permutation(X.shape[0])
fold_size = X.shape[0] // {n_folds}

for fold in range({n_folds}):
    test_start = fold * fold_size
    test_end = test_start + fold_size if fold < {n_folds - 1} else X.shape[0]
    test_idx = indices[test_start:test_end]
    train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
"""


def _stratified_snippet(n_folds: int) -> str:
    return f"""\
# Stratified K-Fold (preserves class distribution)
classes = np.unique(y)
fold_indices = [[] for _ in range({n_folds})]
for cls in classes:
    cls_idx = np.where(y == cls)[0]
    rng = np.random.RandomState(42)
    rng.shuffle(cls_idx)
    for i, idx in enumerate(cls_idx):
        fold_indices[i % {n_folds}].append(idx)

for fold in range({n_folds}):
    test_idx = np.array(fold_indices[fold])
    train_idx = np.concatenate([np.array(fold_indices[f]) for f in range({n_folds}) if f != fold])
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
"""


def _time_series_snippet(n_folds: int) -> str:
    return f"""\
# Time-Series Walk-Forward Validation
n = X.shape[0]
min_train = n // ({n_folds} + 1)

for fold in range({n_folds}):
    train_end = min_train + (fold * (n - min_train) // {n_folds})
    test_end = min_train + ((fold + 1) * (n - min_train) // {n_folds})
    X_train, X_test = X[:train_end], X[train_end:test_end]
    y_train, y_test = y[:train_end], y[train_end:test_end]
"""


def _loo_snippet() -> str:
    return """\
# Leave-One-Out Cross-Validation
scores = []
for i in range(X.shape[0]):
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i)
    X_test = X[i:i+1]
    y_test = y[i:i+1]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores.append(pred[0])
"""


def _repeated_snippet(n_folds: int, n_repeats: int, stratified: bool = False) -> str:
    strat_note = " (stratified)" if stratified else ""
    return f"""\
# Repeated {n_folds}-Fold{strat_note} Cross-Validation
all_scores = []
for repeat in range({n_repeats}):
    rng = np.random.RandomState(42 + repeat)
    indices = rng.permutation(X.shape[0])
    fold_size = X.shape[0] // {n_folds}
    for fold in range({n_folds}):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < {n_folds - 1} else X.shape[0]
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # ... fit and score ...
"""


def print_summary(result: dict) -> None:
    """Print a human-readable CV recommendation."""
    print(f"Recommended strategy: {result['strategy_name']} (n_folds={result['n_folds']})")
    print()

    print("Reasoning:")
    for reason in result["reasoning"]:
        print(f"  - {reason}")
    print()

    chars = result["data_characteristics"]
    print(f"Data: {chars['n_samples']} samples, {chars['n_features']} features, task={chars['task']}")
    if "imbalance_ratio" in chars:
        print(f"  Imbalance ratio: {chars['imbalance_ratio']}")
    print()

    print("Code snippet:")
    print(result["code_snippet"])

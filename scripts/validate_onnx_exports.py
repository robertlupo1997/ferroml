#!/usr/bin/env python3
"""Validate FerroML ONNX exports against onnxruntime.

Runs all 34 ONNX-exportable models through the full round-trip:
  fit → predict → export ONNX → load in ORT → run inference → compare

Outputs a compatibility matrix showing pass/fail/xfail status for each model.

Usage:
    python scripts/validate_onnx_exports.py
    python scripts/validate_onnx_exports.py --json  # machine-readable output
"""

import argparse
import json
import sys
import time

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
    sys.exit(1)


# ── Data generators ─────────────────────────────────────────────────────────

def make_regression(n=50, p=4, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = X @ rng.randn(p) + rng.randn(n) * 0.1
    return X, y


def make_binary(n=50, p=4, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


def make_positive(n=50, p=5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 10, size=(n, p)).astype(np.float64)
    y = (X[:, 0] > 5).astype(np.float64)
    return X, y


def make_binary_features(n=50, p=5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n, p)).astype(np.float64)
    y = (X[:, 0] > 0).astype(np.float64)
    return X, y


# ── Validation logic ────────────────────────────────────────────────────────

def ort_predict(onnx_bytes, X):
    sess = ort.InferenceSession(onnx_bytes)
    return sess.run(None, {"input": X.astype(np.float32)})


def validate_regressor(model_cls, X, y, atol=1e-5, **kwargs):
    m = model_cls(**kwargs)
    m.fit(X, y)
    ferro_pred = m.predict(X)
    ort_result = ort_predict(m.to_onnx_bytes(), X)
    ort_pred = ort_result[0].squeeze()
    max_diff = float(np.max(np.abs(ferro_pred - ort_pred)))
    ok = max_diff <= atol + 1e-4 * np.max(np.abs(ferro_pred))
    return ok, max_diff


def validate_classifier_labels(model_cls, X, y, **kwargs):
    m = model_cls(**kwargs)
    m.fit(X, y)
    ferro_pred = m.predict(X)
    ort_result = ort_predict(m.to_onnx_bytes(), X)
    ort_pred = ort_result[0].squeeze()
    match = np.all(ferro_pred == ort_pred)
    mismatches = int(np.sum(ferro_pred != ort_pred))
    return match, mismatches


def validate_classifier_proba(model_cls, X, y, atol=1e-4, **kwargs):
    m = model_cls(**kwargs)
    m.fit(X, y)
    ferro_pred = m.predict(X)
    ferro_proba = m.predict_proba(X)[:, 1]
    ort_result = ort_predict(m.to_onnx_bytes(), X)
    ort_proba = ort_result[0].squeeze()
    max_diff = float(np.max(np.abs(ferro_proba - ort_proba)))
    ort_labels = (ort_proba > 0.5).astype(np.float64)
    label_match = np.all(ort_labels == ferro_pred)
    return label_match and max_diff < atol, max_diff


def validate_classifier_sigmoid(model_cls, X, y, **kwargs):
    m = model_cls(**kwargs)
    m.fit(X, y)
    ferro_pred = m.predict(X)
    ort_result = ort_predict(m.to_onnx_bytes(), X)
    ort_scores = ort_result[0].squeeze()
    ort_labels = (ort_scores > 0.5).astype(np.float64)
    match = np.all(ferro_pred == ort_labels)
    mismatches = int(np.sum(ferro_pred != ort_labels))
    return match, mismatches


def validate_transformer(transformer_cls, X, atol=1e-6, **kwargs):
    m = transformer_cls(**kwargs)
    m.fit(X)
    ferro_out = m.transform(X)
    ort_result = ort_predict(m.to_onnx_bytes(), X)
    ort_out = ort_result[0].squeeze()
    max_diff = float(np.max(np.abs(ferro_out - ort_out)))
    ok = max_diff <= atol + 1e-5 * np.max(np.abs(ferro_out))
    return ok, max_diff


# ── Model definitions ───────────────────────────────────────────────────────

MODELS = []


def reg(name, import_path, data_fn, known_issue=None, atol=1e-5, **kwargs):
    MODELS.append(dict(
        name=name, import_path=import_path, kind="regressor",
        data_fn=data_fn, known_issue=known_issue, atol=atol, kwargs=kwargs,
    ))


def cls_label(name, import_path, data_fn, known_issue=None, **kwargs):
    MODELS.append(dict(
        name=name, import_path=import_path, kind="classifier_label",
        data_fn=data_fn, known_issue=known_issue, kwargs=kwargs,
    ))


def cls_proba(name, import_path, data_fn, known_issue=None, **kwargs):
    MODELS.append(dict(
        name=name, import_path=import_path, kind="classifier_proba",
        data_fn=data_fn, known_issue=known_issue, kwargs=kwargs,
    ))


def cls_sigmoid(name, import_path, data_fn, known_issue=None, **kwargs):
    MODELS.append(dict(
        name=name, import_path=import_path, kind="classifier_sigmoid",
        data_fn=data_fn, known_issue=known_issue, kwargs=kwargs,
    ))


def trans(name, import_path, data_fn, known_issue=None, **kwargs):
    MODELS.append(dict(
        name=name, import_path=import_path, kind="transformer",
        data_fn=data_fn, known_issue=known_issue, kwargs=kwargs,
    ))


# Linear regressors
reg("LinearRegression", "ferroml.linear.LinearRegression", make_regression)
reg("RidgeRegression", "ferroml.linear.RidgeRegression", make_regression)
reg("LassoRegression", "ferroml.linear.LassoRegression", make_regression)
reg("ElasticNet", "ferroml.linear.ElasticNet", make_regression)
reg("RobustRegression", "ferroml.linear.RobustRegression", make_regression)
reg("QuantileRegression", "ferroml.linear.QuantileRegression", make_regression)

# Linear classifiers
cls_sigmoid("LogisticRegression", "ferroml.linear.LogisticRegression", make_binary)
cls_label("RidgeClassifier", "ferroml.linear.RidgeClassifier", make_binary)

# Tree regressors
reg("DecisionTreeRegressor", "ferroml.trees.DecisionTreeRegressor", make_regression)
reg("RandomForestRegressor", "ferroml.trees.RandomForestRegressor", make_regression, n_estimators=10)
reg("GradientBoostingRegressor", "ferroml.trees.GradientBoostingRegressor", make_regression, n_estimators=10)
reg("HistGradientBoostingRegressor", "ferroml.trees.HistGradientBoostingRegressor", make_regression,
    known_issue="bin-threshold mapping edge-case mismatches", max_iter=10, max_depth=3)

# Tree classifiers
cls_label("DecisionTreeClassifier", "ferroml.trees.DecisionTreeClassifier", make_binary,
          known_issue="TreeEnsembleClassifier output type mismatch in ORT")
cls_label("RandomForestClassifier", "ferroml.trees.RandomForestClassifier", make_binary,
          known_issue="TreeEnsembleClassifier output type mismatch in ORT", n_estimators=10)
cls_proba("GradientBoostingClassifier", "ferroml.trees.GradientBoostingClassifier", make_binary, n_estimators=10)
cls_proba("HistGradientBoostingClassifier", "ferroml.trees.HistGradientBoostingClassifier", make_binary,
          known_issue="bin-threshold mapping edge-case mismatches", max_iter=10, max_depth=3)

# Ensemble regressors
reg("ExtraTreesRegressor", "ferroml.ensemble.ExtraTreesRegressor", make_regression, n_estimators=10)
reg("AdaBoostRegressor", "ferroml.ensemble.AdaBoostRegressor", make_regression,
    known_issue="weighted median not expressible in ONNX; uses weighted-sum approximation", n_estimators=10)
reg("SGDRegressor", "ferroml.ensemble.SGDRegressor", make_regression)

# Ensemble classifiers
cls_label("ExtraTreesClassifier", "ferroml.ensemble.ExtraTreesClassifier", make_binary,
          known_issue="TreeEnsembleClassifier output type mismatch in ORT", n_estimators=10)
cls_label("AdaBoostClassifier", "ferroml.ensemble.AdaBoostClassifier", make_binary,
          known_issue="TreeEnsembleClassifier output type mismatch in ORT", n_estimators=10)
cls_sigmoid("SGDClassifier", "ferroml.ensemble.SGDClassifier", make_binary)
cls_label("PassiveAggressiveClassifier", "ferroml.ensemble.PassiveAggressiveClassifier", make_binary)

# SVM
reg("LinearSVR", "ferroml.svm.LinearSVR", make_regression)
cls_label("LinearSVC", "ferroml.svm.LinearSVC", make_binary)
reg("SVR", "ferroml.svm.SVR", make_regression, atol=1e-3)
cls_label("SVC", "ferroml.svm.SVC", make_binary,
          known_issue="SVMClassifier ONNX graph invalid in ORT")

# Naive Bayes
cls_label("GaussianNB", "ferroml.naive_bayes.GaussianNB", make_binary)
cls_label("MultinomialNB", "ferroml.naive_bayes.MultinomialNB", make_positive)
cls_label("BernoulliNB", "ferroml.naive_bayes.BernoulliNB", make_binary_features,
          known_issue="ONNX export produces incorrect class labels")

# Preprocessing
trans("StandardScaler", "ferroml.preprocessing.StandardScaler", lambda: (make_regression()[0],))
trans("MinMaxScaler", "ferroml.preprocessing.MinMaxScaler", lambda: (make_regression()[0],))
trans("RobustScaler", "ferroml.preprocessing.RobustScaler", lambda: (make_regression()[0],))
trans("MaxAbsScaler", "ferroml.preprocessing.MaxAbsScaler", lambda: (make_regression()[0],))


# ── Main ─────────────────────────────────────────────────────────────────────

def import_class(path):
    parts = path.rsplit(".", 1)
    mod = __import__(parts[0], fromlist=[parts[1]])
    return getattr(mod, parts[1])


def run_validation(verbose=True):
    results = []
    n_pass = n_fail = n_xfail = n_error = 0

    for spec in MODELS:
        name = spec["name"]
        known_issue = spec.get("known_issue")
        kwargs = spec.get("kwargs", {})

        try:
            cls = import_class(spec["import_path"])

            if spec["kind"] == "transformer":
                data = spec["data_fn"]()
                X = data[0] if isinstance(data, tuple) else data
                ok, detail = validate_transformer(cls, X, **kwargs)
            elif spec["kind"] == "regressor":
                X, y = spec["data_fn"]()
                ok, detail = validate_regressor(cls, X, y, atol=spec.get("atol", 1e-5), **kwargs)
            elif spec["kind"] == "classifier_label":
                X, y = spec["data_fn"]()
                ok, detail = validate_classifier_labels(cls, X, y, **kwargs)
            elif spec["kind"] == "classifier_proba":
                X, y = spec["data_fn"]()
                ok, detail = validate_classifier_proba(cls, X, y, **kwargs)
            elif spec["kind"] == "classifier_sigmoid":
                X, y = spec["data_fn"]()
                ok, detail = validate_classifier_sigmoid(cls, X, y, **kwargs)

            if ok:
                status = "PASS"
                n_pass += 1
            elif known_issue:
                status = "XFAIL"
                n_xfail += 1
            else:
                status = "FAIL"
                n_fail += 1

        except Exception as e:
            if known_issue:
                status = "XFAIL"
                n_xfail += 1
            else:
                status = "ERROR"
                n_error += 1
            detail = str(e)[:80]

        results.append(dict(
            name=name, status=status, detail=detail,
            known_issue=known_issue, kind=spec["kind"],
        ))

        if verbose:
            icon = {"PASS": "\u2705", "FAIL": "\u274c", "XFAIL": "\u26a0\ufe0f", "ERROR": "\U0001f4a5"}.get(status, "?")
            detail_str = f" (max_diff={detail:.2e})" if isinstance(detail, float) else f" ({detail})" if detail else ""
            issue_str = f" [{known_issue}]" if known_issue and status == "XFAIL" else ""
            print(f"  {icon} {status:5s}  {name}{detail_str}{issue_str}")

    return results, n_pass, n_fail, n_xfail, n_error


def main():
    parser = argparse.ArgumentParser(description="Validate FerroML ONNX exports")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    print(f"FerroML ONNX Round-Trip Validation")
    print(f"onnxruntime {ort.__version__}")
    print(f"{'=' * 60}")
    print()

    start = time.time()
    results, n_pass, n_fail, n_xfail, n_error = run_validation(verbose=not args.json)
    elapsed = time.time() - start

    print()
    print(f"{'=' * 60}")
    print(f"Results: {n_pass} passed, {n_xfail} expected failures, {n_fail} failed, {n_error} errors")
    print(f"Total: {len(results)} models validated in {elapsed:.2f}s")

    if args.json:
        print(json.dumps({
            "results": results,
            "summary": {
                "total": len(results),
                "passed": n_pass,
                "failed": n_fail,
                "xfailed": n_xfail,
                "errors": n_error,
                "elapsed_seconds": round(elapsed, 2),
            },
        }, indent=2))

    sys.exit(1 if n_fail > 0 or n_error > 0 else 0)


if __name__ == "__main__":
    main()

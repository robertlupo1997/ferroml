#!/usr/bin/env python3
"""FerroML vs scikit-learn Comparison Script - Compares on Iris, Wine, Diabetes datasets"""
import json, sys, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

try:
    from sklearn.datasets import load_iris, load_wine, load_diabetes
    from sklearn.linear_model import LinearRegression as SKLinearRegression, LogisticRegression as SKLogisticRegression
    from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier, DecisionTreeRegressor as SKDecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier, RandomForestRegressor as SKRandomForestRegressor
    from sklearn.neighbors import KNeighborsClassifier as SKKNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score
    import sklearn
    SKLEARN_AVAILABLE, SKLEARN_VERSION = True, sklearn.__version__
except ImportError as e:
    print(f"sklearn not available: {e}")
    SKLEARN_AVAILABLE, SKLEARN_VERSION = False, None

FERROML_AVAILABLE, FERROML_VERSION, FERROML_MODELS = False, None, {}
try:
    ferroml_path = Path(__file__).parent.parent.parent / "ferroml-python" / "python"
    if ferroml_path.exists(): sys.path.insert(0, str(ferroml_path))
    from ferroml.linear import LinearRegression as FMLinearRegression, LogisticRegression as FMLogisticRegression
    FERROML_MODELS["LinearRegression"], FERROML_MODELS["LogisticRegression"] = FMLinearRegression, FMLogisticRegression
    FERROML_AVAILABLE = True
except ImportError as e: print(f"ferroml linear not available: {e}")
try:
    from ferroml.trees import DecisionTreeClassifier as FMDTCls, DecisionTreeRegressor as FMDTReg
    from ferroml.trees import RandomForestClassifier as FMRFCls, RandomForestRegressor as FMRFReg
    FERROML_MODELS.update({"DecisionTreeClassifier": FMDTCls, "DecisionTreeRegressor": FMDTReg,
        "RandomForestClassifier": FMRFCls, "RandomForestRegressor": FMRFReg})
    FERROML_AVAILABLE = True
except ImportError as e: print(f"ferroml trees not available: {e}")
try:
    import ferroml; FERROML_VERSION = ferroml.__version__
except: FERROML_VERSION = "unknown"


def load_datasets():
    datasets = {}
    for name, loader in [("iris", load_iris), ("wine", load_wine), ("diabetes", load_diabetes)]:
        data = loader()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
        datasets[name] = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "task": "regression" if name == "diabetes" else "classification",
            "n_classes": 3 if name != "diabetes" else None, "n_features": data.data.shape[1],
            "n_train": len(y_train), "n_test": len(y_test)}
    return datasets

def compare_predictions(pred1, pred2, tolerance=1e-6):
    pred1, pred2 = np.asarray(pred1).flatten(), np.asarray(pred2).flatten()
    if len(pred1) != len(pred2): return {"match": False, "error": "Length mismatch"}
    abs_diff = np.abs(pred1 - pred2)
    max_diff, mean_diff = float(np.max(abs_diff)), float(np.mean(abs_diff))
    match = np.allclose(pred1, pred2, atol=0.5) if np.all(pred1 == pred1.astype(int)) else max_diff <= tolerance
    return {"match_within_tolerance": match, "tolerance": tolerance, "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff, "exact_match_pct": float(100 * np.mean(pred1 == pred2))}

def run_model(model_class, dataset, model_name, is_ferroml=False, **kwargs):
    key = "ferroml" if is_ferroml else "sklearn"
    result = {key: {"available": model_class is not None, "model_name": model_name, "error": None}}
    if model_class is None:
        result[key]["error"] = "Model not available in Python bindings"
        return result
    try:
        model = model_class(**kwargs)
        model.fit(dataset["X_train"], dataset["y_train"])
        predictions = model.predict(dataset["X_test"])
        score = float(accuracy_score(dataset["y_test"], predictions)) if dataset["task"] == "classification" else float(r2_score(dataset["y_test"], predictions))
        result[key].update({"predictions": predictions.tolist() if hasattr(predictions, "tolist") else list(predictions),
            "score": score, "accuracy" if dataset["task"] == "classification" else "r2_score": score})
        if hasattr(model, "coef_"): result[key]["coefficients"] = (model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_).tolist()[:10]
        if hasattr(model, "feature_importances_"): result[key]["feature_importances"] = model.feature_importances_.tolist()
    except Exception as e:
        result[key]["error"], result[key]["available"] = str(e), False
    return result


def main():
    print("=" * 60 + "\nFerroML vs scikit-learn Comparison\n" + "=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"sklearn: {SKLEARN_AVAILABLE} ({SKLEARN_VERSION}), ferroml: {FERROML_AVAILABLE} ({FERROML_VERSION})")
    print(f"ferroml models: {list(FERROML_MODELS.keys())}")
    if not SKLEARN_AVAILABLE: print("ERROR: sklearn required"); sys.exit(1)

    datasets = load_datasets()
    print("-" * 60 + "\nDatasets:")
    for n, i in datasets.items(): print(f"  {n}: {i[\"n_train\"]} train, {i[\"n_test\"]} test, {i[\"n_features\"]} features")

    configs = [
        ("LinearRegression", SKLinearRegression, FERROML_MODELS.get("LinearRegression"), ["diabetes"], {}, {}),
        ("DecisionTreeRegressor", SKDecisionTreeRegressor, FERROML_MODELS.get("DecisionTreeRegressor"), ["diabetes"], {"random_state": 42, "max_depth": 5}, {"random_state": 42, "max_depth": 5}),
        ("RandomForestRegressor", SKRandomForestRegressor, FERROML_MODELS.get("RandomForestRegressor"), ["diabetes"], {"n_estimators": 50, "random_state": 42, "max_depth": 5}, {"n_estimators": 50, "random_state": 42, "max_depth": 5}),
        ("LogisticRegression", SKLogisticRegression, FERROML_MODELS.get("LogisticRegression"), ["iris", "wine"], {"random_state": 42, "max_iter": 1000}, {"random_state": 42, "max_iter": 1000}),
        ("DecisionTreeClassifier", SKDecisionTreeClassifier, FERROML_MODELS.get("DecisionTreeClassifier"), ["iris", "wine"], {"random_state": 42, "max_depth": 5}, {"random_state": 42, "max_depth": 5}),
        ("RandomForestClassifier", SKRandomForestClassifier, FERROML_MODELS.get("RandomForestClassifier"), ["iris", "wine"], {"n_estimators": 50, "random_state": 42, "max_depth": 5}, {"n_estimators": 50, "random_state": 42, "max_depth": 5}),
        ("KNeighborsClassifier", SKKNeighborsClassifier, None, ["iris", "wine"], {"n_neighbors": 5}, {}),
    ]

    all_results = {"meta": {"timestamp": datetime.now().isoformat(), "sklearn_version": SKLEARN_VERSION,
        "ferroml_version": FERROML_VERSION, "sklearn_available": SKLEARN_AVAILABLE,
        "ferroml_available": FERROML_AVAILABLE, "ferroml_models_loaded": list(FERROML_MODELS.keys()),
        "python_version": sys.version}, "datasets": {n: {"task": i["task"], "n_train": i["n_train"],
        "n_test": i["n_test"], "n_features": i["n_features"], "n_classes": i.get("n_classes")} for n, i in datasets.items()}, "models": {}}


    print("-" * 60 + "\nRunning comparisons:\n" + "-" * 60)
    for name, sk_cls, fm_cls, ds_list, sk_kw, fm_kw in configs:
        print(f"\n{name}:")
        all_results["models"][name] = {"model_name": name, "datasets": {}}
        for ds_name in ds_list:
            ds = datasets[ds_name]
            sk_res = run_model(sk_cls, ds, name, False, **sk_kw)
            fm_res = run_model(fm_cls, ds, name, True, **fm_kw)
            ds_res = {**sk_res, **fm_res, "dataset_info": {"name": ds_name, "task": ds["task"], "n_train": ds["n_train"], "n_test": ds["n_test"], "n_features": ds["n_features"]}}
            if sk_res["sklearn"].get("predictions") and fm_res["ferroml"].get("predictions"):
                comp = compare_predictions(sk_res["sklearn"]["predictions"], fm_res["ferroml"]["predictions"], 1e-4)
                ds_res["comparison"] = comp
                if sk_res["sklearn"].get("score") and fm_res["ferroml"].get("score"):
                    ds_res["comparison"]["score_diff"] = abs(sk_res["sklearn"]["score"] - fm_res["ferroml"]["score"])
            all_results["models"][name]["datasets"][ds_name] = ds_res
            sk_s = f"{sk_res[\"sklearn\"].get(\"score\", 0):.4f}" if sk_res["sklearn"].get("score") else f"ERR: {sk_res[\"sklearn\"].get(\"error\")}"
            fm_s = f"{fm_res[\"ferroml\"].get(\"score\", 0):.4f}" if fm_res["ferroml"].get("score") else f"N/A ({fm_res[\"ferroml\"].get(\"error\", \"not available\")})"
            m = "[MATCH]" if ds_res.get("comparison", {}).get("match_within_tolerance") else (f"[DIFF: {ds_res.get(\"comparison\", {}).get(\"max_abs_diff\", \"N/A\"):.4e}]" if "comparison" in ds_res else "")
            print(f"  {ds_name}: sklearn={sk_s}, ferroml={fm_s} {m}")


    out = Path(__file__).parent / "sklearn_comparison_results.json"
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.int32, np.int64)): return int(o)
        if isinstance(o, (np.float32, np.float64)): return float(o)
        if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list): return [conv(v) for v in o]
        return o
    with open(out, "w") as f: json.dump(conv(all_results), f, indent=2)
    print(f"\n{\"=\" * 60}\nResults saved to: {out}\n{\"=\" * 60}")
    sk_tot = sum(1 for m in all_results["models"].values() for d in m["datasets"].values() if d.get("sklearn", {}).get("score"))
    fm_tot = sum(1 for m in all_results["models"].values() for d in m["datasets"].values() if d.get("ferroml", {}).get("score"))
    match_tot = sum(1 for m in all_results["models"].values() for d in m["datasets"].values() if d.get("comparison", {}).get("match_within_tolerance"))
    print(f"SUMMARY: sklearn={sk_tot}, ferroml={fm_tot}, matched={match_tot}")

if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Feature Parity Scorecard: FerroML vs scikit-learn vs statsmodels vs XGBoost vs LightGBM.

Introspects the FerroML Python API and compares method availability against
sklearn equivalents. Outputs a Markdown table and JSON report.

Usage:
    python scripts/feature_parity_scorecard.py
    python scripts/feature_parity_scorecard.py --output-md docs/feature-parity-scorecard.md
    python scripts/feature_parity_scorecard.py --output-json docs/feature-parity-scorecard.json
"""

import argparse
import importlib
import inspect
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Method categories to check
# ---------------------------------------------------------------------------

CORE_METHODS = [
    "fit",
    "predict",
    "predict_proba",
    "predict_log_proba",
    "decision_function",
    "score",
    "transform",
    "fit_transform",
    "inverse_transform",
    "partial_fit",
    "fit_weighted",
]

PROPERTY_METHODS = [
    "feature_importances_",
    "coef_",
    "intercept_",
]

EXTRA_METHODS = [
    "to_onnx_bytes",
    "export_onnx",
    "warm_start",  # check as constructor param or attribute
    "get_params",
    "set_params",
    "search_space",
]


# ---------------------------------------------------------------------------
# Model registry: FerroML module path -> sklearn equivalent path
# ---------------------------------------------------------------------------

MODEL_REGISTRY: List[Dict[str, Any]] = [
    # --- Linear models ---
    {"ferroml": "ferroml.linear.LinearRegression", "sklearn": "sklearn.linear_model.LinearRegression", "category": "Linear"},
    {"ferroml": "ferroml.linear.RidgeRegression", "sklearn": "sklearn.linear_model.Ridge", "category": "Linear"},
    {"ferroml": "ferroml.linear.LassoRegression", "sklearn": "sklearn.linear_model.Lasso", "category": "Linear"},
    {"ferroml": "ferroml.linear.ElasticNet", "sklearn": "sklearn.linear_model.ElasticNet", "category": "Linear"},
    {"ferroml": "ferroml.linear.RidgeCV", "sklearn": "sklearn.linear_model.RidgeCV", "category": "Linear"},
    {"ferroml": "ferroml.linear.LassoCV", "sklearn": "sklearn.linear_model.LassoCV", "category": "Linear"},
    {"ferroml": "ferroml.linear.ElasticNetCV", "sklearn": "sklearn.linear_model.ElasticNetCV", "category": "Linear"},
    {"ferroml": "ferroml.linear.RobustRegression", "sklearn": "sklearn.linear_model.HuberRegressor", "category": "Linear"},
    {"ferroml": "ferroml.linear.QuantileRegression", "sklearn": "sklearn.linear_model.QuantileRegressor", "category": "Linear"},
    {"ferroml": "ferroml.linear.IsotonicRegression", "sklearn": "sklearn.isotonic.IsotonicRegression", "category": "Linear"},
    {"ferroml": "ferroml.linear.LogisticRegression", "sklearn": "sklearn.linear_model.LogisticRegression", "category": "Linear"},
    {"ferroml": "ferroml.linear.RidgeClassifier", "sklearn": "sklearn.linear_model.RidgeClassifier", "category": "Linear"},
    {"ferroml": "ferroml.linear.Perceptron", "sklearn": "sklearn.linear_model.Perceptron", "category": "Linear"},

    # --- Tree models ---
    {"ferroml": "ferroml.trees.DecisionTreeClassifier", "sklearn": "sklearn.tree.DecisionTreeClassifier", "category": "Trees"},
    {"ferroml": "ferroml.trees.DecisionTreeRegressor", "sklearn": "sklearn.tree.DecisionTreeRegressor", "category": "Trees"},
    {"ferroml": "ferroml.trees.RandomForestClassifier", "sklearn": "sklearn.ensemble.RandomForestClassifier", "category": "Trees"},
    {"ferroml": "ferroml.trees.RandomForestRegressor", "sklearn": "sklearn.ensemble.RandomForestRegressor", "category": "Trees"},
    {"ferroml": "ferroml.trees.GradientBoostingClassifier", "sklearn": "sklearn.ensemble.GradientBoostingClassifier", "category": "Trees"},
    {"ferroml": "ferroml.trees.GradientBoostingRegressor", "sklearn": "sklearn.ensemble.GradientBoostingRegressor", "category": "Trees"},
    {"ferroml": "ferroml.trees.HistGradientBoostingClassifier", "sklearn": "sklearn.ensemble.HistGradientBoostingClassifier", "category": "Trees"},
    {"ferroml": "ferroml.trees.HistGradientBoostingRegressor", "sklearn": "sklearn.ensemble.HistGradientBoostingRegressor", "category": "Trees"},

    # --- Ensemble models ---
    {"ferroml": "ferroml.ensemble.ExtraTreesClassifier", "sklearn": "sklearn.ensemble.ExtraTreesClassifier", "category": "Ensemble"},
    {"ferroml": "ferroml.ensemble.ExtraTreesRegressor", "sklearn": "sklearn.ensemble.ExtraTreesRegressor", "category": "Ensemble"},
    {"ferroml": "ferroml.ensemble.AdaBoostClassifier", "sklearn": "sklearn.ensemble.AdaBoostClassifier", "category": "Ensemble"},
    {"ferroml": "ferroml.ensemble.AdaBoostRegressor", "sklearn": "sklearn.ensemble.AdaBoostRegressor", "category": "Ensemble"},
    {"ferroml": "ferroml.ensemble.BaggingClassifier", "sklearn": "sklearn.ensemble.BaggingClassifier", "category": "Ensemble"},
    {"ferroml": "ferroml.ensemble.BaggingRegressor", "sklearn": "sklearn.ensemble.BaggingRegressor", "category": "Ensemble"},
    {"ferroml": "ferroml.ensemble.VotingClassifier", "sklearn": "sklearn.ensemble.VotingClassifier", "category": "Ensemble"},
    {"ferroml": "ferroml.ensemble.VotingRegressor", "sklearn": "sklearn.ensemble.VotingRegressor", "category": "Ensemble"},
    {"ferroml": "ferroml.ensemble.StackingClassifier", "sklearn": "sklearn.ensemble.StackingClassifier", "category": "Ensemble"},
    {"ferroml": "ferroml.ensemble.StackingRegressor", "sklearn": "sklearn.ensemble.StackingRegressor", "category": "Ensemble"},
    {"ferroml": "ferroml.ensemble.SGDClassifier", "sklearn": "sklearn.linear_model.SGDClassifier", "category": "Ensemble"},
    {"ferroml": "ferroml.ensemble.SGDRegressor", "sklearn": "sklearn.linear_model.SGDRegressor", "category": "Ensemble"},
    {"ferroml": "ferroml.ensemble.PassiveAggressiveClassifier", "sklearn": "sklearn.linear_model.PassiveAggressiveClassifier", "category": "Ensemble"},

    # --- Neighbors ---
    {"ferroml": "ferroml.neighbors.KNeighborsClassifier", "sklearn": "sklearn.neighbors.KNeighborsClassifier", "category": "Neighbors"},
    {"ferroml": "ferroml.neighbors.KNeighborsRegressor", "sklearn": "sklearn.neighbors.KNeighborsRegressor", "category": "Neighbors"},
    {"ferroml": "ferroml.neighbors.NearestCentroid", "sklearn": "sklearn.neighbors.NearestCentroid", "category": "Neighbors"},

    # --- SVM ---
    {"ferroml": "ferroml.svm.LinearSVC", "sklearn": "sklearn.svm.LinearSVC", "category": "SVM"},
    {"ferroml": "ferroml.svm.LinearSVR", "sklearn": "sklearn.svm.LinearSVR", "category": "SVM"},
    {"ferroml": "ferroml.svm.SVC", "sklearn": "sklearn.svm.SVC", "category": "SVM"},
    {"ferroml": "ferroml.svm.SVR", "sklearn": "sklearn.svm.SVR", "category": "SVM"},

    # --- Naive Bayes ---
    {"ferroml": "ferroml.naive_bayes.GaussianNB", "sklearn": "sklearn.naive_bayes.GaussianNB", "category": "Naive Bayes"},
    {"ferroml": "ferroml.naive_bayes.MultinomialNB", "sklearn": "sklearn.naive_bayes.MultinomialNB", "category": "Naive Bayes"},
    {"ferroml": "ferroml.naive_bayes.BernoulliNB", "sklearn": "sklearn.naive_bayes.BernoulliNB", "category": "Naive Bayes"},
    {"ferroml": "ferroml.naive_bayes.CategoricalNB", "sklearn": "sklearn.naive_bayes.CategoricalNB", "category": "Naive Bayes"},

    # --- Neural ---
    {"ferroml": "ferroml.neural.MLPClassifier", "sklearn": "sklearn.neural_network.MLPClassifier", "category": "Neural"},
    {"ferroml": "ferroml.neural.MLPRegressor", "sklearn": "sklearn.neural_network.MLPRegressor", "category": "Neural"},

    # --- Clustering ---
    {"ferroml": "ferroml.clustering.KMeans", "sklearn": "sklearn.cluster.KMeans", "category": "Clustering"},
    {"ferroml": "ferroml.clustering.DBSCAN", "sklearn": "sklearn.cluster.DBSCAN", "category": "Clustering"},
    {"ferroml": "ferroml.clustering.AgglomerativeClustering", "sklearn": "sklearn.cluster.AgglomerativeClustering", "category": "Clustering"},
    {"ferroml": "ferroml.clustering.GaussianMixture", "sklearn": "sklearn.mixture.GaussianMixture", "category": "Clustering"},
    {"ferroml": "ferroml.clustering.HDBSCAN", "sklearn": "sklearn.cluster.HDBSCAN", "category": "Clustering"},

    # --- Decomposition ---
    {"ferroml": "ferroml.decomposition.PCA", "sklearn": "sklearn.decomposition.PCA", "category": "Decomposition"},
    {"ferroml": "ferroml.decomposition.IncrementalPCA", "sklearn": "sklearn.decomposition.IncrementalPCA", "category": "Decomposition"},
    {"ferroml": "ferroml.decomposition.TruncatedSVD", "sklearn": "sklearn.decomposition.TruncatedSVD", "category": "Decomposition"},
    {"ferroml": "ferroml.decomposition.LDA", "sklearn": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis", "category": "Decomposition"},
    {"ferroml": "ferroml.decomposition.QuadraticDiscriminantAnalysis", "sklearn": "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis", "category": "Decomposition"},
    {"ferroml": "ferroml.decomposition.FactorAnalysis", "sklearn": "sklearn.decomposition.FactorAnalysis", "category": "Decomposition"},
    {"ferroml": "ferroml.decomposition.TSNE", "sklearn": "sklearn.manifold.TSNE", "category": "Decomposition"},

    # --- Anomaly ---
    {"ferroml": "ferroml.anomaly.IsolationForest", "sklearn": "sklearn.ensemble.IsolationForest", "category": "Anomaly"},
    {"ferroml": "ferroml.anomaly.LocalOutlierFactor", "sklearn": "sklearn.neighbors.LocalOutlierFactor", "category": "Anomaly"},

    # --- Gaussian Process ---
    {"ferroml": "ferroml.gaussian_process.GaussianProcessRegressor", "sklearn": "sklearn.gaussian_process.GaussianProcessRegressor", "category": "Gaussian Process"},
    {"ferroml": "ferroml.gaussian_process.GaussianProcessClassifier", "sklearn": "sklearn.gaussian_process.GaussianProcessClassifier", "category": "Gaussian Process"},
    {"ferroml": "ferroml.gaussian_process.SparseGPRegressor", "sklearn": None, "category": "Gaussian Process"},
    {"ferroml": "ferroml.gaussian_process.SparseGPClassifier", "sklearn": None, "category": "Gaussian Process"},
    {"ferroml": "ferroml.gaussian_process.SVGPRegressor", "sklearn": None, "category": "Gaussian Process"},

    # --- Calibration ---
    {"ferroml": "ferroml.calibration.TemperatureScalingCalibrator", "sklearn": "sklearn.calibration.CalibratedClassifierCV", "category": "Calibration"},

    # --- MultiOutput ---
    {"ferroml": "ferroml.multioutput.MultiOutputRegressor", "sklearn": "sklearn.multioutput.MultiOutputRegressor", "category": "MultiOutput"},
    {"ferroml": "ferroml.multioutput.MultiOutputClassifier", "sklearn": "sklearn.multioutput.MultiOutputClassifier", "category": "MultiOutput"},
]

PREPROCESSOR_REGISTRY: List[Dict[str, Any]] = [
    {"ferroml": "ferroml.preprocessing.StandardScaler", "sklearn": "sklearn.preprocessing.StandardScaler", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.MinMaxScaler", "sklearn": "sklearn.preprocessing.MinMaxScaler", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.RobustScaler", "sklearn": "sklearn.preprocessing.RobustScaler", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.MaxAbsScaler", "sklearn": "sklearn.preprocessing.MaxAbsScaler", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.OneHotEncoder", "sklearn": "sklearn.preprocessing.OneHotEncoder", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.OrdinalEncoder", "sklearn": "sklearn.preprocessing.OrdinalEncoder", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.LabelEncoder", "sklearn": "sklearn.preprocessing.LabelEncoder", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.TargetEncoder", "sklearn": "sklearn.preprocessing.TargetEncoder", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.SimpleImputer", "sklearn": "sklearn.impute.SimpleImputer", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.KNNImputer", "sklearn": "sklearn.impute.KNNImputer", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.PowerTransformer", "sklearn": "sklearn.preprocessing.PowerTransformer", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.QuantileTransformer", "sklearn": "sklearn.preprocessing.QuantileTransformer", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.PolynomialFeatures", "sklearn": "sklearn.preprocessing.PolynomialFeatures", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.KBinsDiscretizer", "sklearn": "sklearn.preprocessing.KBinsDiscretizer", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.VarianceThreshold", "sklearn": "sklearn.feature_selection.VarianceThreshold", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.SelectKBest", "sklearn": "sklearn.feature_selection.SelectKBest", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.SelectFromModel", "sklearn": "sklearn.feature_selection.SelectFromModel", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.RecursiveFeatureElimination", "sklearn": "sklearn.feature_selection.RFE", "category": "Preprocessing"},
    {"ferroml": "ferroml.preprocessing.CountVectorizer", "sklearn": "sklearn.feature_extraction.text.CountVectorizer", "category": "Text"},
    {"ferroml": "ferroml.preprocessing.TfidfTransformer", "sklearn": "sklearn.feature_extraction.text.TfidfTransformer", "category": "Text"},
    {"ferroml": "ferroml.preprocessing.TfidfVectorizer", "sklearn": "sklearn.feature_extraction.text.TfidfVectorizer", "category": "Text"},
    {"ferroml": "ferroml.preprocessing.SMOTE", "sklearn": None, "category": "Sampling"},
    {"ferroml": "ferroml.preprocessing.ADASYN", "sklearn": None, "category": "Sampling"},
    {"ferroml": "ferroml.preprocessing.RandomUnderSampler", "sklearn": None, "category": "Sampling"},
    {"ferroml": "ferroml.preprocessing.RandomOverSampler", "sklearn": None, "category": "Sampling"},
]


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def _resolve_class(dotted_path: str) -> Optional[type]:
    """Import and return a class from a dotted path like 'ferroml.linear.LinearRegression'."""
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        return None
    module_path, class_name = parts
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name, None)
    except (ImportError, ModuleNotFoundError):
        return None


def _has_method(cls: type, method_name: str) -> bool:
    """Check if a class has a given method or property."""
    if cls is None:
        return False

    # Check as attribute on class
    if hasattr(cls, method_name):
        return True

    # For properties like feature_importances_, also check if it's a descriptor
    for klass in inspect.getmro(cls) if hasattr(cls, '__mro__') else [cls]:
        if method_name in getattr(klass, '__dict__', {}):
            return True

    return False


def _check_warm_start(cls: type) -> bool:
    """Check if a class supports warm_start (as constructor param or attribute)."""
    if cls is None:
        return False

    # Check if warm_start is in __init__ signature
    try:
        sig = inspect.signature(cls.__init__)
        if "warm_start" in sig.parameters:
            return True
    except (ValueError, TypeError):
        pass

    # Check as class attribute
    return hasattr(cls, "warm_start")


def check_methods(cls: type, methods: List[str]) -> Dict[str, bool]:
    """Check which methods/attributes a class has."""
    result = {}
    for method in methods:
        if method == "warm_start":
            result[method] = _check_warm_start(cls)
        else:
            result[method] = _has_method(cls, method)
    return result


# ---------------------------------------------------------------------------
# Scorecard generation
# ---------------------------------------------------------------------------

@dataclass
class ModelComparison:
    name: str
    ferroml_path: str
    sklearn_path: Optional[str]
    category: str
    ferroml_methods: Dict[str, bool] = field(default_factory=dict)
    sklearn_methods: Dict[str, bool] = field(default_factory=dict)
    ferroml_exists: bool = False
    sklearn_exists: bool = False
    ferroml_extras: List[str] = field(default_factory=list)  # FerroML-only methods


def generate_scorecard(registry: List[Dict[str, Any]]) -> List[ModelComparison]:
    """Generate feature comparison for all models in the registry."""
    all_methods = CORE_METHODS + PROPERTY_METHODS + EXTRA_METHODS
    results = []

    for entry in registry:
        ferroml_path = entry["ferroml"]
        sklearn_path = entry.get("sklearn")
        name = ferroml_path.rsplit(".", 1)[1]

        comp = ModelComparison(
            name=name,
            ferroml_path=ferroml_path,
            sklearn_path=sklearn_path,
            category=entry["category"],
        )

        # Resolve FerroML class
        ferro_cls = _resolve_class(ferroml_path)
        comp.ferroml_exists = ferro_cls is not None
        if ferro_cls is not None:
            comp.ferroml_methods = check_methods(ferro_cls, all_methods)

            # Check for FerroML-specific extras
            for extra in ["summary", "predict_interval", "fit_sparse", "predict_sparse",
                          "transform_sparse", "fit_pandas", "predict_pandas",
                          "fit_dataframe", "predict_dataframe", "assumption_test",
                          "coefficients_with_ci", "predict_with_std"]:
                if _has_method(ferro_cls, extra):
                    comp.ferroml_extras.append(extra)

        # Resolve sklearn class
        if sklearn_path:
            sklearn_cls = _resolve_class(sklearn_path)
            comp.sklearn_exists = sklearn_cls is not None
            if sklearn_cls is not None:
                comp.sklearn_methods = check_methods(sklearn_cls, all_methods)

        results.append(comp)

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_parity_stats(comparisons: List[ModelComparison]) -> Dict[str, Any]:
    """Compute aggregate parity statistics."""
    total_ferroml = sum(1 for c in comparisons if c.ferroml_exists)
    total_sklearn = sum(1 for c in comparisons if c.sklearn_exists)
    both_exist = [c for c in comparisons if c.ferroml_exists and c.sklearn_exists]

    # Per-method parity
    all_methods = CORE_METHODS + PROPERTY_METHODS + EXTRA_METHODS
    method_stats = {}
    for method in all_methods:
        ferro_has = sum(1 for c in comparisons if c.ferroml_methods.get(method, False))
        sklearn_has = sum(1 for c in both_exist if c.sklearn_methods.get(method, False))
        ferro_has_when_sklearn_has = sum(
            1 for c in both_exist
            if c.sklearn_methods.get(method, False) and c.ferroml_methods.get(method, False)
        )
        method_stats[method] = {
            "ferroml_total": ferro_has,
            "sklearn_total": sklearn_has,
            "parity": ferro_has_when_sklearn_has,
            "gap": sklearn_has - ferro_has_when_sklearn_has,
        }

    # Find biggest gaps (methods where sklearn has it but FerroML doesn't)
    gaps = []
    for c in both_exist:
        for method in CORE_METHODS + PROPERTY_METHODS:
            if c.sklearn_methods.get(method, False) and not c.ferroml_methods.get(method, False):
                gaps.append({"model": c.name, "method": method})

    return {
        "total_ferroml_models": total_ferroml,
        "total_sklearn_models": total_sklearn,
        "models_in_both": len(both_exist),
        "method_stats": method_stats,
        "gaps": gaps,
    }


def find_top_gaps(comparisons: List[ModelComparison], n: int = 5) -> List[Dict[str, Any]]:
    """Find top-N missing features by impact (most commonly available in sklearn but missing in FerroML)."""
    gap_counts: Dict[str, List[str]] = {}
    both_exist = [c for c in comparisons if c.ferroml_exists and c.sklearn_exists]

    for c in both_exist:
        for method in CORE_METHODS + PROPERTY_METHODS:
            if c.sklearn_methods.get(method, False) and not c.ferroml_methods.get(method, False):
                gap_counts.setdefault(method, []).append(c.name)

    ranked = sorted(gap_counts.items(), key=lambda x: len(x[1]), reverse=True)
    return [
        {"method": method, "count": len(models), "models": models}
        for method, models in ranked[:n]
    ]


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _method_symbol(ferroml_has: bool, sklearn_has: Optional[bool]) -> str:
    """Return a symbol for the comparison table."""
    if sklearn_has is None:
        # No sklearn equivalent
        return "Y" if ferroml_has else "-"
    if ferroml_has and sklearn_has:
        return "Y"  # both have it
    if ferroml_has and not sklearn_has:
        return "Y+"  # FerroML has extra
    if not ferroml_has and sklearn_has:
        return "GAP"  # missing in FerroML
    return "-"  # neither has it


def format_markdown(
    model_comparisons: List[ModelComparison],
    preprocessor_comparisons: List[ModelComparison],
    stats: Dict[str, Any],
    top_gaps: List[Dict[str, Any]],
) -> str:
    """Format the scorecard as Markdown."""
    lines = []
    lines.append("# FerroML Feature Parity Scorecard")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    lines.append("**Legend:** Y = supported, GAP = sklearn has it / FerroML missing, "
                 "Y+ = FerroML-only, - = neither has it")
    lines.append("")

    # Summary stats
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **FerroML models:** {stats['total_ferroml_models']}")
    lines.append(f"- **sklearn equivalents checked:** {stats['total_sklearn_models']}")
    lines.append(f"- **Models in both:** {stats['models_in_both']}")
    lines.append(f"- **Total gaps:** {len(stats['gaps'])}")
    lines.append("")

    # Top gaps
    if top_gaps:
        lines.append("## Top Missing Features (by Impact)")
        lines.append("")
        for i, gap in enumerate(top_gaps, 1):
            models_str = ", ".join(gap["models"][:5])
            if len(gap["models"]) > 5:
                models_str += f", ... (+{len(gap['models']) - 5} more)"
            lines.append(f"{i}. **`{gap['method']}`** — missing in {gap['count']} models: {models_str}")
        lines.append("")

    # Method parity summary table
    lines.append("## Method Parity Summary")
    lines.append("")
    lines.append("| Method | FerroML has | sklearn has | Parity | Gap |")
    lines.append("|--------|------------|------------|--------|-----|")
    for method, s in stats["method_stats"].items():
        if s["sklearn_total"] > 0 or s["ferroml_total"] > 0:
            lines.append(f"| `{method}` | {s['ferroml_total']} | {s['sklearn_total']} | {s['parity']} | {s['gap']} |")
    lines.append("")

    # Detailed comparison tables by category
    check_methods_list = CORE_METHODS + PROPERTY_METHODS[:3]  # Keep table manageable
    short_names = {
        "fit": "fit", "predict": "pred", "predict_proba": "proba",
        "predict_log_proba": "log_p", "decision_function": "dec_fn",
        "score": "score", "transform": "xform", "fit_transform": "fit_x",
        "inverse_transform": "inv_x", "partial_fit": "p_fit",
        "fit_weighted": "w_fit",
        "feature_importances_": "f_imp", "coef_": "coef", "intercept_": "icpt",
    }

    for label, comparisons in [("Models", model_comparisons), ("Preprocessors", preprocessor_comparisons)]:
        lines.append(f"## {label}")
        lines.append("")

        # Group by category
        categories = []
        seen = set()
        for c in comparisons:
            if c.category not in seen:
                categories.append(c.category)
                seen.add(c.category)

        for cat in categories:
            cat_models = [c for c in comparisons if c.category == cat]
            lines.append(f"### {cat}")
            lines.append("")

            # Header
            header_methods = [m for m in check_methods_list
                              if any(c.ferroml_methods.get(m, False) or c.sklearn_methods.get(m, False)
                                     for c in cat_models)]
            header = "| Model | " + " | ".join(short_names.get(m, m) for m in header_methods) + " | Extras |"
            sep = "|" + "---|" * (len(header_methods) + 2)
            lines.append(header)
            lines.append(sep)

            for c in cat_models:
                if not c.ferroml_exists:
                    continue
                cells = []
                for m in header_methods:
                    ferro = c.ferroml_methods.get(m, False)
                    sk = c.sklearn_methods.get(m, False) if c.sklearn_exists else None
                    cells.append(_method_symbol(ferro, sk))
                extras = ", ".join(c.ferroml_extras[:3])
                if len(c.ferroml_extras) > 3:
                    extras += f" +{len(c.ferroml_extras) - 3}"
                name_display = c.name
                if not c.sklearn_exists:
                    name_display += " *"
                lines.append(f"| {name_display} | " + " | ".join(cells) + f" | {extras} |")

            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Models marked with `*` have no sklearn equivalent.*")
    lines.append("")

    return "\n".join(lines)


def format_json(
    model_comparisons: List[ModelComparison],
    preprocessor_comparisons: List[ModelComparison],
    stats: Dict[str, Any],
    top_gaps: List[Dict[str, Any]],
) -> str:
    """Format the scorecard as JSON."""
    data = {
        "generated": datetime.now().isoformat(),
        "summary": stats,
        "top_gaps": top_gaps,
        "models": [
            {
                "name": c.name,
                "ferroml_path": c.ferroml_path,
                "sklearn_path": c.sklearn_path,
                "category": c.category,
                "ferroml_exists": c.ferroml_exists,
                "sklearn_exists": c.sklearn_exists,
                "ferroml_methods": c.ferroml_methods,
                "sklearn_methods": c.sklearn_methods,
                "ferroml_extras": c.ferroml_extras,
            }
            for c in model_comparisons + preprocessor_comparisons
        ],
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FerroML Feature Parity Scorecard")
    parser.add_argument("--output-md", default="docs/feature-parity-scorecard.md",
                        help="Markdown output path")
    parser.add_argument("--output-json", default="docs/feature-parity-scorecard.json",
                        help="JSON output path")
    args = parser.parse_args()

    print("Generating feature parity scorecard...")
    print()

    # Generate comparisons
    model_comparisons = generate_scorecard(MODEL_REGISTRY)
    preprocessor_comparisons = generate_scorecard(PREPROCESSOR_REGISTRY)
    all_comparisons = model_comparisons + preprocessor_comparisons

    # Compute stats
    stats = compute_parity_stats(all_comparisons)
    top_gaps = find_top_gaps(all_comparisons, n=5)

    # Print summary to console
    ferro_count = sum(1 for c in all_comparisons if c.ferroml_exists)
    sklearn_count = sum(1 for c in all_comparisons if c.sklearn_exists)
    print(f"FerroML classes found: {ferro_count}/{len(all_comparisons)}")
    print(f"sklearn equivalents found: {sklearn_count}/{len(all_comparisons)}")
    print(f"Total gaps: {len(stats['gaps'])}")
    print()

    if top_gaps:
        print("Top missing features:")
        for i, gap in enumerate(top_gaps, 1):
            print(f"  {i}. {gap['method']} — missing in {gap['count']} models")
        print()

    # Write outputs
    md_content = format_markdown(model_comparisons, preprocessor_comparisons, stats, top_gaps)
    with open(args.output_md, "w") as f:
        f.write(md_content)
    print(f"Markdown saved to: {args.output_md}")

    json_content = format_json(model_comparisons, preprocessor_comparisons, stats, top_gaps)
    with open(args.output_json, "w") as f:
        f.write(json_content)
    print(f"JSON saved to: {args.output_json}")


if __name__ == "__main__":
    main()

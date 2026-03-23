"""Docstring completeness verification for all FerroML Python bindings.

This test file verifies that all public model/transformer classes in ferroml
have complete NumPy-style docstrings with the required sections (Parameters,
Examples, Notes for key models).

IMPORTANT: This test requires a maturin rebuild to reflect docstring changes.
Run before testing:
    maturin develop --release -m ferroml-python/Cargo.toml
"""

import importlib

import pytest

# ---------------------------------------------------------------------------
# Submodules and class discovery
# ---------------------------------------------------------------------------

SUBMODULES = [
    "ferroml.linear",
    "ferroml.svm",
    "ferroml.trees",
    "ferroml.ensemble",
    "ferroml.preprocessing",
    "ferroml.decomposition",
    "ferroml.clustering",
    "ferroml.neighbors",
    "ferroml.naive_bayes",
    "ferroml.neural",
    "ferroml.multioutput",
    "ferroml.anomaly",
    "ferroml.gaussian_process",
    "ferroml.calibration",
    "ferroml.cv",
]

# Kernel classes are building blocks passed to GP models, not standalone
# models.  They need Parameters but not Examples.
KERNEL_CLASSES = {"RBF", "Matern", "ConstantKernel", "WhiteKernel"}

# Classes that legitimately have no constructor parameters (no-arg __init__
# or factory-only construction).  These are exempt from the "Parameters"
# section requirement.
NO_PARAM_CLASSES = {
    "BaggingClassifier",   # factory pattern -- documented via factory methods
    "BaggingRegressor",    # factory pattern
    "MaxAbsScaler",        # no-arg constructor
    "LabelEncoder",        # no-arg constructor
    "LeaveOneOut",         # no-arg constructor
}


def _all_classes():
    """Yield (module_path, class_name, class_obj) for every public class."""
    for mod_path in SUBMODULES:
        mod = importlib.import_module(mod_path)
        all_names = getattr(mod, "__all__", [])
        for name in all_names:
            obj = getattr(mod, name, None)
            if obj is not None and isinstance(obj, type):
                yield mod_path, name, obj


ALL_CLASSES = list(_all_classes())
# Models/transformers -- everything except kernel helper classes
MODEL_CLASSES = [(m, n, c) for m, n, c in ALL_CLASSES if n not in KERNEL_CLASSES]


# ---------------------------------------------------------------------------
# Test: every class has a docstring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mod_path,cls_name,cls",
    ALL_CLASSES,
    ids=[f"{m}.{n}" for m, n, _ in ALL_CLASSES],
)
def test_all_classes_have_docstrings(mod_path, cls_name, cls):
    """Every public class must have a non-empty __doc__."""
    assert cls.__doc__ is not None, f"{mod_path}.{cls_name} has no __doc__"
    assert len(cls.__doc__.strip()) > 20, (
        f"{mod_path}.{cls_name} docstring is too short"
    )


# ---------------------------------------------------------------------------
# Test: every class with parameters documents them
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mod_path,cls_name,cls",
    [(m, n, c) for m, n, c in ALL_CLASSES if n not in NO_PARAM_CLASSES],
    ids=[
        f"{m}.{n}"
        for m, n, _ in ALL_CLASSES
        if n not in NO_PARAM_CLASSES
    ],
)
def test_all_models_have_parameters(mod_path, cls_name, cls):
    """Every class (except no-param classes) must document its Parameters."""
    doc = cls.__doc__ or ""
    has_params = "Parameters" in doc or "parameter" in doc.lower()
    assert has_params, (
        f"{mod_path}.{cls_name} missing 'Parameters' section in docstring"
    )


# ---------------------------------------------------------------------------
# Test: model/transformer classes have Examples
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mod_path,cls_name,cls",
    MODEL_CLASSES,
    ids=[f"{m}.{n}" for m, n, _ in MODEL_CLASSES],
)
def test_all_models_have_examples(mod_path, cls_name, cls):
    """Every model/transformer (non-kernel) class must have an Examples section."""
    doc = cls.__doc__ or ""
    assert "Example" in doc, (
        f"{mod_path}.{cls_name} missing 'Examples' section in docstring"
    )


# ---------------------------------------------------------------------------
# Test: key models have Notes sections
# ---------------------------------------------------------------------------

KEY_MODELS_WITH_NOTES = [
    ("ferroml.svm", "SVC"),
    ("ferroml.trees", "RandomForestClassifier"),
    ("ferroml.trees", "RandomForestRegressor"),
    ("ferroml.trees", "HistGradientBoostingClassifier"),
    ("ferroml.trees", "HistGradientBoostingRegressor"),
    ("ferroml.gaussian_process", "GaussianProcessRegressor"),
    ("ferroml.gaussian_process", "GaussianProcessClassifier"),
]


@pytest.mark.parametrize(
    "mod_path,cls_name",
    KEY_MODELS_WITH_NOTES,
    ids=[f"{m}.{n}" for m, n in KEY_MODELS_WITH_NOTES],
)
def test_key_models_have_notes(mod_path, cls_name):
    """Key models must have a Notes section documenting limitations."""
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    doc = cls.__doc__ or ""
    assert "Notes" in doc, (
        f"{mod_path}.{cls_name} missing 'Notes' section in docstring"
    )


# ---------------------------------------------------------------------------
# Test: sample of models documents parameter defaults
# ---------------------------------------------------------------------------

SAMPLE_MODELS_FOR_DEFAULTS = [
    ("ferroml.linear", "LinearRegression"),
    ("ferroml.linear", "LogisticRegression"),
    ("ferroml.linear", "RidgeRegression"),
    ("ferroml.svm", "SVC"),
    ("ferroml.trees", "RandomForestClassifier"),
    ("ferroml.trees", "HistGradientBoostingClassifier"),
    ("ferroml.preprocessing", "StandardScaler"),
    ("ferroml.clustering", "KMeans"),
    ("ferroml.neighbors", "KNeighborsClassifier"),
    ("ferroml.naive_bayes", "GaussianNB"),
]


@pytest.mark.parametrize(
    "mod_path,cls_name",
    SAMPLE_MODELS_FOR_DEFAULTS,
    ids=[f"{m}.{n}" for m, n in SAMPLE_MODELS_FOR_DEFAULTS],
)
def test_parameters_have_defaults(mod_path, cls_name):
    """A sample of models must document parameter defaults."""
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    doc = (cls.__doc__ or "").lower()
    assert "default" in doc, (
        f"{mod_path}.{cls_name} does not document parameter defaults "
        "(expected 'default=' in docstring)"
    )


# ---------------------------------------------------------------------------
# Test: minimum class count sanity check
# ---------------------------------------------------------------------------

def test_minimum_class_count():
    """Sanity check: we should find at least 55 model classes."""
    assert len(ALL_CLASSES) >= 55, (
        f"Expected at least 55 classes, found {len(ALL_CLASSES)}. "
        "Check __all__ re-exports in submodule __init__.py files."
    )


def test_minimum_model_count():
    """Sanity check: at least 55 model/transformer classes (excl. kernels)."""
    assert len(MODEL_CLASSES) >= 55, (
        f"Expected at least 55 model classes, found {len(MODEL_CLASSES)}."
    )

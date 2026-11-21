"""Binding completeness: every Rust model reaches Python.

CLAUDE.md names incomplete bindings as the most common mistake in this repo.
Adding a model takes three edits, and forgetting any one of them still
compiles and still passes every other test:

1. Rust impl in ``ferroml-core/src/``           (``impl Model for X``)
2. PyO3 wrapper in ``ferroml-python/src/``      (``#[pyclass(name = "X", module = "ferroml.<sub>")]``
                                                 registered with ``add_class::<PyX>()``)
3. Re-export in ``ferroml-python/python/ferroml/<sub>/__init__.py`` (``__all__``)

The static tests read source files only, so they run without building the
extension. The runtime tests check the same chain against the installed
``ferroml`` and are skipped when it is not importable.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CORE_SRC = REPO / "ferroml-core" / "src"
PY_SRC = REPO / "ferroml-python" / "src"
PY_PKG = REPO / "ferroml-python" / "python" / "ferroml"

# Public estimator traits. A struct implementing any of these is a model the
# Python package is expected to expose.
ESTIMATOR_TRAITS = (
    "Model",
    "Transformer",
    "ClusteringModel",
    "OutlierDetector",
    "NeuralModel",
    "TextTransformer",
)

# Rust estimators that are deliberately not bound. Every entry needs a reason,
# and an entry that is no longer needed fails ``test_allowlist_is_not_stale``.
INTENTIONALLY_UNBOUND: dict[str, str] = {
    "CalibratedClassifierCV": (
        "wraps an arbitrary Rust base estimator (Box<dyn CalibrableClassifier>); "
        "binding it needs a Python-to-Rust estimator bridge that does not exist yet. "
        "Python users get TemperatureScalingCalibrator in ferroml.calibration."
    ),
}

# Only column-0 impls count: impls inside ``mod tests`` are indented, which
# keeps test mocks (MockModel, DoublingTransformer, ...) out of the inventory.
_IMPL_RE = re.compile(
    r"^impl(?:<[^>\n]*>)?\s+(?:\w+::)*(" + "|".join(ESTIMATOR_TRAITS) + r")\s+for\s+(\w+)",
    re.MULTILINE,
)
_PYCLASS_RE = re.compile(r"#\[pyclass\(([^\]]*)\)\]\s*(?:#\[[^\]]*\]\s*)*pub\s+struct\s+(\w+)")
_ADD_CLASS_RE = re.compile(r"add_class::<(?:\w+::)*(\w+)>")


def _core_estimators() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in CORE_SRC.rglob("*.rs"):
        if "testing" in path.relative_to(CORE_SRC).parts:
            continue
        for trait, name in _IMPL_RE.findall(path.read_text(encoding="utf-8")):
            found.setdefault(name, set()).add(trait)
    return found


def _pyclasses() -> dict[str, dict[str, str | None]]:
    """Python-visible name -> {module, struct, file}."""
    found: dict[str, dict[str, str | None]] = {}
    for path in PY_SRC.rglob("*.rs"):
        for args, struct in _PYCLASS_RE.findall(path.read_text(encoding="utf-8")):
            name = re.search(r'\bname\s*=\s*"(\w+)"', args)
            module = re.search(r'\bmodule\s*=\s*"([\w.]+)"', args)
            if name is None:
                continue
            found[name.group(1)] = {
                "module": module.group(1) if module else None,
                "struct": struct,
                "file": path.name,
            }
    return found


def _registered_structs() -> set[str]:
    names: set[str] = set()
    for path in PY_SRC.rglob("*.rs"):
        names.update(_ADD_CLASS_RE.findall(path.read_text(encoding="utf-8")))
    return names


def _dunder_all(init: Path) -> set[str]:
    tree = ast.parse(init.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return set(ast.literal_eval(node.value))
    return set()


def _init_for(module: str) -> Path:
    parts = module.split(".")
    assert parts[0] == "ferroml", module
    return PY_PKG.joinpath(*parts[1:], "__init__.py")


CORE = _core_estimators()
PYCLASSES = _pyclasses()


def test_inventory_is_not_empty() -> None:
    # Guards the regexes: if a refactor breaks them, every other test here
    # would pass vacuously.
    assert len(CORE) >= 80, f"only {len(CORE)} Rust estimators found; regex broken?"
    assert len(PYCLASSES) >= 100, f"only {len(PYCLASSES)} pyclasses found; regex broken?"


def test_every_rust_estimator_has_a_pyo3_wrapper() -> None:
    missing = sorted(set(CORE) - set(PYCLASSES) - set(INTENTIONALLY_UNBOUND))
    assert not missing, (
        "Rust estimators with no #[pyclass(name = ...)] wrapper in ferroml-python/src/: "
        f"{missing}. Add a wrapper, or add the name to INTENTIONALLY_UNBOUND with a reason."
    )


def test_allowlist_is_not_stale() -> None:
    stale = sorted(n for n in INTENTIONALLY_UNBOUND if n in PYCLASSES or n not in CORE)
    assert not stale, f"remove from INTENTIONALLY_UNBOUND (now bound or gone): {stale}"


def test_every_pyclass_declares_its_module() -> None:
    # Without module=..., PyO3 reports __module__ == "builtins", which breaks
    # pickling by reference and makes reprs and docs point nowhere.
    missing = sorted(f"{n} ({c['file']})" for n, c in PYCLASSES.items() if not c["module"])
    assert not missing, f"#[pyclass] without module = \"ferroml.<sub>\": {missing}"


def test_every_pyclass_is_registered() -> None:
    registered = _registered_structs()
    missing = sorted(
        f"{n} ({c['struct']} in {c['file']})"
        for n, c in PYCLASSES.items()
        if c["struct"] not in registered
    )
    assert not missing, f"#[pyclass] never passed to add_class::<...>(): {missing}"


@pytest.mark.parametrize("name", sorted(set(CORE) - set(INTENTIONALLY_UNBOUND)))
def test_rust_estimator_is_reexported(name: str) -> None:
    module = PYCLASSES[name]["module"]
    assert module, f"{name} has no module"
    init = _init_for(module)
    assert init.exists(), f"{name} declares module {module}, but {init} does not exist"
    assert name in _dunder_all(init), (
        f"{name} is wrapped in {PYCLASSES[name]['file']} but missing from "
        f"__all__ in {init.relative_to(REPO)}"
    )


def test_every_pyclass_is_reexported_from_its_module() -> None:
    missing = []
    for name, info in sorted(PYCLASSES.items()):
        module = info["module"]
        if not module:
            continue  # reported by test_every_pyclass_declares_its_module
        init = _init_for(module)
        if not init.exists() or name not in _dunder_all(init):
            missing.append(f"{name} -> {module}")
    assert not missing, f"wrapped but not in the declared module's __all__: {missing}"


# --------------------------------------------------------------------------
# Runtime: the same chain against the built extension.
# --------------------------------------------------------------------------


def _ferroml_or_skip():
    try:
        return importlib.import_module("ferroml")
    except ImportError:  # pragma: no cover - depends on the environment
        pytest.skip("ferroml extension is not built/installed")


@pytest.mark.parametrize("name", sorted(PYCLASSES))
def test_runtime_class_resolves_from_declared_module(name: str) -> None:
    _ferroml_or_skip()
    module = PYCLASSES[name]["module"]
    if not module:
        pytest.fail(f"{name} has no module")
    mod = importlib.import_module(module)
    cls = getattr(mod, name, None)
    assert cls is not None, f"{module}.{name} is not importable"
    assert cls.__module__ == module, f"{name}.__module__ is {cls.__module__!r}, expected {module!r}"

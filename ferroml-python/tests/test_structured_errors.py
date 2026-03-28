"""Tests for structured remediation hints in FerroError messages.

Every FerroError variant with a hint should include 'Hint:' in its Python
error message, giving users actionable guidance for resolving the issue.
"""

import numpy as np
import pytest


def test_shape_mismatch_has_hint():
    """ShapeMismatch errors should include a hint about array lengths."""
    from ferroml.linear import LinearRegression

    model = LinearRegression()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Hint"):
        model.fit(X, y)


def test_not_fitted_has_hint():
    """NotFitted errors should include a hint about calling fit() first."""
    from ferroml.linear import LinearRegression

    model = LinearRegression()
    X = np.array([[1.0, 2.0]])
    with pytest.raises((RuntimeError, ValueError), match="Hint"):
        model.predict(X)


def test_convergence_failure_has_hint():
    """ConvergenceFailure errors should include a hint about max_iter."""
    from ferroml.linear import LogisticRegression

    model = LogisticRegression(max_iter=1)
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y = (X[:, 0] > 0).astype(float)
    try:
        model.fit(X, y)
    except RuntimeError as e:
        assert "Hint" in str(e)


def test_invalid_input_has_hint():
    """InvalidInput errors should include a hint about parameter docs."""
    from ferroml.linear import RidgeRegression

    model = RidgeRegression(alpha=-1.0)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Hint"):
        model.fit(X, y)


def test_hint_format():
    """All hints should start with 'Hint:'."""
    from ferroml.linear import LinearRegression

    model = LinearRegression()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0, 3.0])
    try:
        model.fit(X, y)
    except ValueError as e:
        msg = str(e)
        assert "Hint:" in msg


def test_hint_appears_after_main_message():
    """The hint should appear after the main error message, separated by newline."""
    from ferroml.linear import LinearRegression

    model = LinearRegression()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0, 3.0])
    try:
        model.fit(X, y)
    except ValueError as e:
        msg = str(e)
        # Main message should come before the hint
        hint_idx = msg.index("Hint:")
        assert hint_idx > 0, "Hint should not be at the start of the message"
        # There should be a newline before the hint
        assert "\n" in msg[:hint_idx], "Hint should be on a separate line"

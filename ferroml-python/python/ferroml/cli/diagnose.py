"""ferroml diagnose — run statistical diagnostics on a fitted model."""
from __future__ import annotations

import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_data, load_model


def diagnose(
    model: str = typer.Option(..., "--model", "-m", help="Path to fitted model (.pkl)."),
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet file."),
    target: str = typer.Option(..., "--target", "-t", help="Target column name."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Run statistical diagnostics on a fitted model."""
    import numpy as np
    from ferroml import metrics, stats

    mdl = load_model(model)
    X, y, feature_names = load_data(data, target)
    preds = mdl.predict(X)
    residuals = y - preds

    diag: dict = {}

    # Always compute: residual stats
    diag["residual_mean"] = round(float(np.mean(residuals)), 6)
    diag["residual_std"] = round(float(np.std(residuals)), 6)
    diag["durbin_watson"] = round(float(stats.durbin_watson(residuals)), 6)
    normality = stats.normality_test(residuals)
    diag["residual_normality"] = normality

    # Model-specific diagnostics
    if hasattr(mdl, "summary"):
        diag["summary"] = mdl.summary()

    if hasattr(mdl, "r_squared"):
        diag["r_squared"] = round(float(mdl.r_squared()), 6)

    if hasattr(mdl, "adjusted_r_squared"):
        diag["adjusted_r_squared"] = round(float(mdl.adjusted_r_squared()), 6)

    if hasattr(mdl, "f_statistic"):
        f_stat = mdl.f_statistic()
        if isinstance(f_stat, tuple):
            diag["f_statistic"] = round(float(f_stat[0]), 6)
            diag["f_pvalue"] = round(float(f_stat[1]), 6)
        else:
            diag["f_statistic"] = round(float(f_stat), 6)

    if hasattr(mdl, "coefficients_with_ci"):
        diag["coefficients"] = mdl.coefficients_with_ci()

    if hasattr(mdl, "aic"):
        diag["aic"] = round(float(mdl.aic()), 6)

    if hasattr(mdl, "bic"):
        diag["bic"] = round(float(mdl.bic()), 6)

    if hasattr(mdl, "log_likelihood"):
        diag["log_likelihood"] = round(float(mdl.log_likelihood()), 6)

    # Basic metrics
    is_classification = len(np.unique(y)) <= 20
    if is_classification:
        diag["accuracy"] = round(float(metrics.accuracy_score(y, preds)), 6)
    else:
        diag["rmse"] = round(float(metrics.rmse(y, preds)), 6)
        diag["r2"] = round(float(metrics.r2_score(y, preds)), 6)

    result_data = {"diagnostics": diag}
    output(result_data, json_mode)

"""ferroml info — show model card metadata."""
from __future__ import annotations

from typing import Optional

import typer

from ferroml.cli._format import output
from ferroml.cli._registry import get_model_class, list_models


def info(
    model_name: Optional[str] = typer.Argument(None, help="Model class name (e.g. LinearRegression)."),
    all_models: bool = typer.Option(False, "--all", help="List all available models with metadata."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Show model card metadata for a model class."""
    if all_models:
        cards = []
        for name in list_models():
            cls = get_model_class(name)
            if hasattr(cls, "model_card"):
                card = cls.model_card()
                cards.append({"name": name, "task": card.task, "complexity": card.complexity})
            else:
                cards.append({"name": name, "task": [], "complexity": "unknown"})
        output(cards, json_mode)
        return

    if not model_name:
        typer.echo("Error: provide a model name or --all", err=True)
        raise typer.Exit(1)

    cls = get_model_class(model_name)
    if not hasattr(cls, "model_card"):
        typer.echo(f"Error: {model_name} does not have a model card.", err=True)
        raise typer.Exit(1)

    card = cls.model_card()
    result_data = {
        "name": card.name,
        "task": card.task,
        "complexity": card.complexity,
        "interpretability": card.interpretability,
        "supports_sparse": card.supports_sparse,
        "supports_incremental": card.supports_incremental,
        "supports_sample_weight": card.supports_sample_weight,
        "strengths": card.strengths,
        "limitations": card.limitations,
        "references": card.references,
    }
    output(result_data, json_mode)

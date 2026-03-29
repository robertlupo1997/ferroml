"""Output formatting for the CLI (JSON vs Rich tables)."""
from __future__ import annotations

import json
import sys


def output(data: dict | list, json_mode: bool) -> None:
    """Print data as JSON (to stdout) or as a Rich table (to stderr + stdout)."""
    if json_mode:
        print(json.dumps(data, indent=2, default=str))
    else:
        _print_rich(data)


def _print_rich(data: dict | list) -> None:
    """Pretty-print data using Rich."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console(stderr=True)

        if isinstance(data, list) and data and isinstance(data[0], dict):
            table = Table(show_header=True)
            for key in data[0]:
                table.add_column(str(key))
            for row in data:
                table.add_row(*[str(row.get(k, "")) for k in data[0]])
            console.print(table)
        elif isinstance(data, dict):
            table = Table(show_header=True)
            table.add_column("Key")
            table.add_column("Value")
            for k, v in data.items():
                table.add_row(str(k), str(v))
            console.print(table)
        else:
            console.print(data)
    except ImportError:
        # Fallback if rich not installed
        print(json.dumps(data, indent=2, default=str))

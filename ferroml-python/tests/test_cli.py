"""Tests for the ferroml CLI."""
import subprocess
import sys

import pytest


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the ferroml CLI as a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "ferroml.cli", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestCLIEntryPoint:
    def test_help(self):
        result = run_cli("--help")
        assert result.returncode == 0
        assert "ferroml" in result.stdout.lower()

    def test_version(self):
        result = run_cli("--version")
        assert result.returncode == 0
        assert "1.0.0" in result.stdout

    def test_no_args_shows_help(self):
        result = run_cli()
        # Typer's no_args_is_help returns exit code 0 or 2 depending on version
        assert "ferroml" in result.stdout.lower()

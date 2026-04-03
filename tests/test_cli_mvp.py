"""Tests for the MVP CLI surface."""

from typer.testing import CliRunner

from migrama.cli.main import app


def test_help_shows_only_mvp_commands() -> None:
    """Top-level help should only expose the MVP commands."""
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("pattern", "analyze", "extract", "convert", "save"):
        assert command in result.stdout

    for command in ("average", "tension", "graph", "info", "viewer"):
        assert command not in result.stdout

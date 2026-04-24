"""CLI smoke tests."""

from typer.testing import CliRunner

from divergent_bench.cli import app


def test_cli_help_loads_without_api_or_glove_dependencies():
    """The installed console script target should expose a usable help screen."""
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "DAT-Bench" in result.output
    assert "run" in result.output

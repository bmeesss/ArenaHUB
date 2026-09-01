"""Smoke tests for the Typer CLI (help output and command wiring)."""

from __future__ import annotations

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("chat", "models", "health", "serve"):
        assert command in result.stdout


def test_chat_help() -> None:
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.stdout


def test_models_help() -> None:
    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code == 0

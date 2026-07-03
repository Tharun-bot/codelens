"""Tests for codelens.cli — index and search commands."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import codelens.cli as cli_module
from codelens.cli import app

runner = CliRunner()


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n"
        "    \"\"\"Add two numbers together.\"\"\"\n"
        "    return a + b\n"
        "\n"
        "def send_email(to, subject):\n"
        "    smtp.send(to, subject)\n"
    )
    return tmp_path


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Point INDEX_DIR and DATABASE_URL at isolated temp locations for this test."""
    monkeypatch.setattr(cli_module, "INDEX_DIR", tmp_path / "indexes")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    return tmp_path


@pytest.mark.slow
def test_index_command_success(fixture_repo, isolated_env):
    result = runner.invoke(app, ["index", str(fixture_repo)])

    assert result.exit_code == 0
    assert "Indexed 2 chunks" in result.stdout


@pytest.mark.slow
def test_index_command_missing_path(isolated_env):
    result = runner.invoke(app, ["index", "/definitely/not/a/real/path"])

    assert result.exit_code == 1
    assert "does not exist" in result.stdout


@pytest.mark.slow
def test_search_command_finds_expected_result(fixture_repo, isolated_env):
    index_result = runner.invoke(app, ["index", str(fixture_repo)])
    assert index_result.exit_code == 0

    # extract repo_id from the printed output, e.g. "(repo_id=1)"
    import re
    match = re.search(r"repo_id=(\d+)", index_result.stdout)
    assert match is not None
    repo_id = match.group(1)

    search_result = runner.invoke(
        app, ["search", "add two numbers", "--repo-id", repo_id]
    )

    assert search_result.exit_code == 0
    assert "add" in search_result.stdout
    assert "math_utils.py" in search_result.stdout


@pytest.mark.slow
def test_search_command_missing_index(isolated_env):
    result = runner.invoke(app, ["search", "anything", "--repo-id", "999"])

    assert result.exit_code == 1
    assert "No index found" in result.stdout
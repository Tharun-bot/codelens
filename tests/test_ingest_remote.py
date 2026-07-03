"""Tests for codelens.ingest — resolving git URLs (local + remote).

We avoid hitting real GitHub in the default test run: instead we create a
local bare git repo and clone from it via a file:// URL, which exercises
the exact same code path (subprocess git clone) without any network
dependency or flakiness in CI.
"""

import subprocess
from pathlib import Path

import pytest

from codelens.ingest import discover_files, is_git_url, resolve_repo_source


def _run_git(args: list[str], cwd: Path):
    result = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result


@pytest.fixture
def local_bare_repo(tmp_path: Path) -> str:
    """
    Build a real (non-bare) source repo with one commit, then create a bare
    clone of it — a bare repo is what a real "remote" looks like structurally.
    Returns a file:// URL pointing at the bare repo.
    """
    source_dir = tmp_path / "source_repo"
    source_dir.mkdir()

    _run_git(["init", "-q"], cwd=source_dir)
    _run_git(["config", "user.email", "test@example.com"], cwd=source_dir)
    _run_git(["config", "user.name", "Test User"], cwd=source_dir)

    (source_dir / "main.py").write_text("def add(a, b):\n    return a + b\n")

    _run_git(["add", "."], cwd=source_dir)
    _run_git(["commit", "-q", "-m", "initial commit"], cwd=source_dir)

    bare_dir = tmp_path / "bare_repo.git"
    _run_git(["clone", "-q", "--bare", str(source_dir), str(bare_dir)], cwd=tmp_path)

    return f"file://{bare_dir}"


def test_is_git_url_detects_various_schemes():
    assert is_git_url("https://github.com/user/repo") is True
    assert is_git_url("git@github.com:user/repo.git") is True
    assert is_git_url("file:///tmp/some/repo") is True
    assert is_git_url("/local/path/to/repo") is False
    assert is_git_url("./relative/path") is False


def test_resolve_repo_source_clones_git_url(local_bare_repo: str):
    resolved_path = resolve_repo_source(local_bare_repo)

    assert resolved_path.exists()
    assert (resolved_path / "main.py").exists()


def test_discover_files_works_on_cloned_repo(local_bare_repo: str):
    resolved_path = resolve_repo_source(local_bare_repo)
    files = discover_files(resolved_path, extensions={".py"})

    names = {f.name for f in files}
    assert "main.py" in names


def test_resolve_repo_source_returns_local_path_unchanged(tmp_path: Path):
    local_dir = tmp_path / "some_local_repo"
    local_dir.mkdir()
    (local_dir / "file.py").write_text("pass\n")

    resolved_path = resolve_repo_source(str(local_dir))
    assert resolved_path == local_dir


def test_resolve_repo_source_raises_on_missing_local_path(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        resolve_repo_source(str(missing))


def test_resolve_repo_source_raises_on_invalid_url():
    with pytest.raises(RuntimeError):
        resolve_repo_source("https://github.com/this-user-and-repo/definitely-do-not-exist-12345")


@pytest.mark.network
def test_resolve_repo_source_clones_real_github_repo():
    """
    Hits real GitHub — skipped by default (see pytest.ini_options markers).
    Run explicitly with: pytest -v -m network
    """
    resolved_path = resolve_repo_source("https://github.com/octocat/Hello-World")
    assert resolved_path.exists()
    assert any(resolved_path.iterdir())
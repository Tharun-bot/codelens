"""Tests for codelens.ingest — directory walking and file filtering."""

from pathlib import Path

import pytest

from codelens.ingest import discover_files, read_file


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """
    Build a fake repo tree:

    tmp_path/
        main.py
        util.go
        README.md
        node_modules/
            lib.js
        .git/
            config.py   (should never be seen — dir is excluded)
        src/
            app.js
            binary.py   (invalid utf-8 — should be skipped)
    """
    (tmp_path / "main.py").write_text("def main():\n    pass\n")
    (tmp_path / "util.go").write_text("package main\n")
    (tmp_path / "README.md").write_text("# hello\n")

    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "lib.js").write_text("module.exports = {};\n")

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config.py").write_text("# should be excluded\n")

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "app.js").write_text("console.log('hi');\n")

    # write invalid UTF-8 bytes directly
    (src_dir / "binary.py").write_bytes(b"\x80\x81\x82\xff\xfe")

    return tmp_path


def test_discover_files_filters_by_extension(fake_repo: Path):
    results = discover_files(fake_repo, extensions={".py", ".go"})
    names = {p.name for p in results}

    assert "main.py" in names
    assert "util.go" in names
    assert "README.md" not in names
    assert "lib.js" not in names


def test_discover_files_excludes_git_dir(fake_repo: Path):
    results = discover_files(fake_repo, extensions={".py"})
    paths_str = {str(p) for p in results}

    assert not any(".git" in p for p in paths_str)
    assert not any("config.py" in p for p in paths_str)


def test_discover_files_excludes_node_modules(fake_repo: Path):
    results = discover_files(fake_repo, extensions={".js"})
    paths_str = {str(p) for p in results}

    assert not any("node_modules" in p for p in paths_str)
    # app.js inside src/ should still be found
    assert any("app.js" in p for p in paths_str)


def test_discover_files_skips_invalid_utf8(fake_repo: Path):
    results = discover_files(fake_repo, extensions={".py"})
    names = {p.name for p in results}

    assert "binary.py" not in names
    assert "main.py" in names


def test_discover_files_default_extensions(fake_repo: Path):
    # no extensions passed -> should use DEFAULT_EXTENSIONS and pick up .py, .go, .js
    results = discover_files(fake_repo)
    names = {p.name for p in results}

    assert "main.py" in names
    assert "util.go" in names
    assert "app.js" in names
    assert "README.md" not in names


def test_discover_files_raises_on_missing_path(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        discover_files(missing)


def test_read_file_returns_contents(fake_repo: Path):
    content = read_file(fake_repo / "main.py")
    assert "def main" in content
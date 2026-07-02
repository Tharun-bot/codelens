"""File ingestion: walk a directory and discover source files to index."""

from pathlib import Path

# Directories we never want to walk into
EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "dist",
    "build",
    "*.egg-info",
}

DEFAULT_EXTENSIONS = {".py", ".go", ".js", ".ts", ".java", ".rs", ".c", ".cpp", ".h"}


def _is_excluded_dir(dirname: str) -> bool:
    """True if a directory name should be skipped entirely."""
    if dirname in EXCLUDED_DIRS:
        return True
    if dirname.startswith("."):
        return True
    if dirname.endswith(".egg-info"):
        return True
    return False


def _is_readable_text(path: Path) -> bool:
    """Best-effort check that a file is UTF-8 decodable text, not binary."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            f.read()
        return True
    except (UnicodeDecodeError, OSError):
        return False


def discover_files(root: Path, extensions: set[str] = None) -> list[Path]:
    """
    Walk `root` recursively and return a list of source file paths matching
    `extensions`, skipping excluded directories and unreadable/binary files.

    Args:
        root: directory to walk
        extensions: set of file extensions to include, e.g. {".py", ".go"}.
                    Defaults to DEFAULT_EXTENSIONS if not provided.

    Returns:
        Sorted list of Path objects for matching, readable files.
    """
    if extensions is None:
        extensions = DEFAULT_EXTENSIONS

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    matches: list[Path] = []

    for dirpath, dirnames, filenames in _walk(root):
        for filename in filenames:
            file_path = dirpath / filename
            if file_path.suffix in extensions and _is_readable_text(file_path):
                matches.append(file_path)

    return sorted(matches)


def _walk(root: Path):
    """
    Custom directory walk (built on os.walk semantics via pathlib) that prunes
    excluded directories in-place so we never descend into them.
    """
    import os

    for dirpath_str, dirnames, filenames in os.walk(root):
        # prune excluded dirs in-place — os.walk respects mutations to dirnames
        dirnames[:] = [d for d in dirnames if not _is_excluded_dir(d)]
        yield Path(dirpath_str), dirnames, filenames


def read_file(path: Path) -> str:
    """Read a source file's contents as text. Raises if unreadable."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
"""File ingestion: walk a directory and discover source files to index."""

from pathlib import Path
import subprocess
import tempfile

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

def is_git_url(path_or_url: str) -> bool:
    """
    Heuristic check for whether a string is a git URL rather than a local path.
    Covers https://, git://, ssh, git@, and file:// (used in tests / local bares).
    """
    return (
        path_or_url.startswith("https://")
        or path_or_url.startswith("http://")
        or path_or_url.startswith("git://")
        or path_or_url.startswith("git@")
        or path_or_url.startswith("ssh://")
        or path_or_url.startswith("file://")
    )


def resolve_repo_source(path_or_url: str) -> Path:
    """
    Given either a local directory path or a git URL, return a local Path
    ready to be walked by discover_files().

    If it's a URL, shallow-clones it into a temp directory (depth=1, so we
    only pull the latest snapshot, not full history — much faster for
    indexing purposes since we only care about current source, not git log).

    If it's a local path, returns it unchanged (after existence check).

    Raises:
        FileNotFoundError: local path doesn't exist
        RuntimeError: git clone failed
    """
    if is_git_url(path_or_url):
        return _clone_repo(path_or_url)

    local_path = Path(path_or_url)
    if not local_path.exists():
        raise FileNotFoundError(f"Path does not exist: {local_path}")
    return local_path


def _clone_repo(url: str) -> Path:
    dest = Path(tempfile.mkdtemp(prefix="codelens_clone_"))

    env = {
        **__import__("os").environ,
        "GIT_TERMINAL_PROMPT": "0",  # never hang waiting for interactive credential input
    }

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,  # fail loudly instead of hanging forever on network issues
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"git clone timed out after 120s for {url}")

    if result.returncode != 0:
        raise RuntimeError(f"git clone failed for {url}:\n{result.stderr}")

    return dest
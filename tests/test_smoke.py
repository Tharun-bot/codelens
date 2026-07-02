def test_import():
    """Sanity check: the codelens package is importable."""
    import codelens
    assert codelens.__version__ == "0.1.0"


def test_python_version():
    """Make sure we're on a supported Python version."""
    import sys
    assert sys.version_info >= (3, 10)
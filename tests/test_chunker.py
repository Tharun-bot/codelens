"""Tests for codelens.chunker — AST-based chunking via tree-sitter."""

from pathlib import Path

import pytest

from codelens.chunker import chunk_file


@pytest.fixture
def sample_py_file(tmp_path: Path) -> Path:
    content = '''def add(a, b):
    """Add two numbers."""
    return a + b


def subtract(a, b):
    return a - b


class Calculator:
    def multiply(self, a, b):
        """Multiply two numbers."""
        return a * b
'''
    file_path = tmp_path / "sample.py"
    file_path.write_text(content)
    return file_path


def test_chunk_count(sample_py_file: Path):
    # expect: add, subtract, Calculator (class), multiply (method) = 4 chunks
    chunks = chunk_file(sample_py_file)
    assert len(chunks) == 4


def test_chunk_names(sample_py_file: Path):
    chunks = chunk_file(sample_py_file)
    names = {c.name for c in chunks}
    assert names == {"add", "subtract", "Calculator", "multiply"}


def test_docstring_extraction(sample_py_file: Path):
    chunks = chunk_file(sample_py_file)
    by_name = {c.name: c for c in chunks}

    assert by_name["add"].docstring == "Add two numbers."
    assert by_name["subtract"].docstring is None
    assert by_name["multiply"].docstring == "Multiply two numbers."


def test_node_types(sample_py_file: Path):
    chunks = chunk_file(sample_py_file)
    by_name = {c.name: c for c in chunks}

    assert by_name["add"].node_type == "function"
    assert by_name["Calculator"].node_type == "class"
    assert by_name["multiply"].node_type == "method"


def test_line_boundaries_match_function(sample_py_file: Path):
    chunks = chunk_file(sample_py_file)
    by_name = {c.name: c for c in chunks}

    add_chunk = by_name["add"]
    assert add_chunk.start_line == 1
    assert add_chunk.end_line == 3


def test_full_function_body_captured(sample_py_file: Path):
    chunks = chunk_file(sample_py_file)
    by_name = {c.name: c for c in chunks}

    subtract_chunk = by_name["subtract"]
    assert "def subtract(a, b):" in subtract_chunk.source_text
    assert "return a - b" in subtract_chunk.source_text


def test_class_chunk_contains_nested_method_source(sample_py_file: Path):
    chunks = chunk_file(sample_py_file)
    by_name = {c.name: c for c in chunks}

    class_chunk = by_name["Calculator"]
    assert "def multiply" in class_chunk.source_text


def test_fallback_for_file_with_no_functions(tmp_path: Path):
    file_path = tmp_path / "config.py"
    file_path.write_text("DEBUG = True\nPORT = 8080\n")

    chunks = chunk_file(file_path)
    assert len(chunks) == 1
    assert chunks[0].node_type == "file"
    assert chunks[0].name == "config.py"


def test_fallback_for_unsupported_extension(tmp_path: Path):
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Just some notes\n")

    chunks = chunk_file(file_path)
    assert len(chunks) == 1
    assert chunks[0].node_type == "file"
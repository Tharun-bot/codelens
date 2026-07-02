"""AST-level code chunking using tree-sitter.

Instead of chunking by fixed line/token windows, we chunk by function and
class boundaries so each embedding unit is a semantically complete piece of
code — this avoids splitting a function's logic across two embeddings.
"""

from dataclasses import dataclass
from pathlib import Path

from tree_sitter_languages import get_parser

# Maps file extension -> tree-sitter grammar name
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".go": "go",
    ".js": "javascript",
    ".ts": "typescript",
}

# Node types considered "chunkable units" per language.
# Python: top-level and nested function/class definitions.
CHUNK_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "go": {"function_declaration", "method_declaration"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition"},
}


@dataclass
class CodeChunk:
    file_path: str
    name: str
    node_type: str          # "function", "class", "method", or "file" (fallback)
    start_line: int          # 1-indexed, inclusive
    end_line: int            # 1-indexed, inclusive
    source_text: str
    docstring: str | None = None


def _language_for_file(path: Path) -> str | None:
    return EXTENSION_TO_LANGUAGE.get(path.suffix)


def _extract_name(node, source_bytes: bytes) -> str:
    """Pull the identifier name out of a function/class node, if present."""
    for child in node.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8")
    return "<anonymous>"


def _extract_python_docstring(node, source_bytes: bytes) -> str | None:
    """
    For a Python function/class node, look for a docstring: the first
    statement in its body block, if it's a bare string expression.
    """
    block = next((c for c in node.children if c.type == "block"), None)
    if block is None:
        return None

    first_stmt = next(
        (c for c in block.children if c.type == "expression_statement"), None
    )
    if first_stmt is None:
        return None

    string_node = next((c for c in first_stmt.children if c.type == "string"), None)
    if string_node is None:
        return None

    raw = source_bytes[string_node.start_byte:string_node.end_byte].decode("utf-8")
    return raw.strip("\"'").strip()


def _node_type_label(node_type: str, is_method: bool = False) -> str:
    """Normalize raw grammar node types into simple labels."""
    if "class" in node_type:
        return "class"
    if is_method:
        return "method"
    return "function"


def _walk_chunk_nodes(node, chunkable_types: set[str], inside_class: bool = False):
    """
    Yield (node, is_method) pairs for all nodes matching chunkable_types,
    recursing into children. `is_method` is True when a function/method node
    is nested inside a class body — needed because Python's grammar uses the
    same node type (function_definition) for both top-level functions and
    class methods.
    """
    if node.type in chunkable_types:
        is_method = inside_class and "class" not in node.type
        yield node, is_method

    child_inside_class = inside_class or ("class" in node.type)
    for child in node.children:
        yield from _walk_chunk_nodes(child, chunkable_types, inside_class=child_inside_class)


def chunk_file(path: Path) -> list[CodeChunk]:
    """
    Parse a source file and return a list of CodeChunks, one per function or
    class definition found. If the language is unsupported or the file has
    no chunkable nodes, falls back to treating the whole file as one chunk.
    """
    path = Path(path)
    language = _language_for_file(path)

    source_text = path.read_text(encoding="utf-8")
    source_bytes = source_text.encode("utf-8")

    if language is None:
        return _whole_file_fallback(path, source_text)

    parser = get_parser(language)
    tree = parser.parse(source_bytes)

    chunkable_types = CHUNK_NODE_TYPES.get(language, set())
    nodes = list(_walk_chunk_nodes(tree.root_node, chunkable_types))

    if not nodes:
        return _whole_file_fallback(path, source_text)

    chunks: list[CodeChunk] = []
    for node, is_method in nodes:
        name = _extract_name(node, source_bytes)
        node_text = source_bytes[node.start_byte:node.end_byte].decode("utf-8")

        docstring = None
        if language == "python":
            docstring = _extract_python_docstring(node, source_bytes)

        chunks.append(
            CodeChunk(
                file_path=str(path),
                name=name,
                node_type=_node_type_label(node.type, is_method),
                start_line=node.start_point[0] + 1,  # tree-sitter rows are 0-indexed
                end_line=node.end_point[0] + 1,
                source_text=node_text,
                docstring=docstring,
            )
        )

    return chunks


def _whole_file_fallback(path: Path, source_text: str) -> list[CodeChunk]:
    """Used when a file has no function/class nodes, or an unsupported language."""
    line_count = source_text.count("\n") + 1
    return [
        CodeChunk(
            file_path=str(path),
            name=path.name,
            node_type="file",
            start_line=1,
            end_line=line_count,
            source_text=source_text,
            docstring=None,
        )
    ]
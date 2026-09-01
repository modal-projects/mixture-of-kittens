"""Read the Modal app as one text, whichever module a claim happens to live in.

The app is spread across `modal_app.py` and the five modules it imports. A
source contract cares that the app says something, not which of the six files
says it, and a contract that named one file would go quiet the next time a
function moved between them. So the claims read this instead.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

#: The repository root, which is where every orchestration module sits.
ROOT = Path(__file__).resolve().parents[1]

#: The app's entry module. The rest are found through what it imports.
ENTRY = "modal_app.py"


@functools.cache
def files() -> tuple[Path, ...]:
    """Every orchestration module, entry first and the imported ones after it."""
    entry = ROOT / ENTRY
    rest = sorted(
        path
        for path in ROOT.glob("modal_*.py")
        if path != entry and path.is_file()
    )
    return (entry, *rest)


@functools.cache
def read() -> str:
    """The whole app's source, concatenated in a stable order."""
    return "".join(path.read_text(encoding="utf-8") for path in files())


@functools.cache
def registered() -> dict[str, str]:
    """Every `@app.function` and `@app.local_entrypoint`, and where it is defined.

    Read from the source rather than by importing, because importing needs
    ``modal`` and the CPU suite has no reason to carry it.
    """
    found: dict[str, str] = {}
    for path in files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                call = decorator.func if isinstance(decorator, ast.Call) else decorator
                if (
                    isinstance(call, ast.Attribute)
                    and call.attr in ("function", "local_entrypoint")
                    and isinstance(call.value, ast.Name)
                    and call.value.id == "app"
                ):
                    found[node.name] = path.name
    return found


@functools.cache
def entry_imports() -> set[str]:
    """Every orchestration module `modal_app.py` imports, by module name."""
    tree = ast.parse((ROOT / ENTRY).read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


@functools.cache
def image_builders() -> dict[str, str]:
    """Every function in the app that builds and returns a `modal.Image`.

    Keyed by name, valued by the function's own source. Found by return
    annotation rather than by a list, so a third image builder is covered by
    whatever holds these two the moment it is written.
    """
    found: dict[str, str] = {}
    for path in files():
        source = path.read_text(encoding="utf-8")
        for node in ast.parse(source).body:
            if not isinstance(node, ast.FunctionDef) or node.returns is None:
                continue
            if ast.unparse(node.returns) == "modal.Image":
                found[node.name] = ast.get_source_segment(source, node) or ""
    return found


@functools.cache
def entry_names() -> set[str]:
    """Every name `modal_app.py` binds, whether defined there or imported in."""
    tree = ast.parse((ROOT / ENTRY).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
    return names

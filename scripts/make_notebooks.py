"""Turn the tutorials in examples/ into Jupyter notebooks, for viewing and running locally.

The tutorials are plain ``.py`` files with ``# %%`` cells (the sphinx-gallery layout): the module
docstring is the first text cell, a cell that only holds comments is a text cell, every other cell
is code. Notebooks are generated, never committed; by default they go to ``docs/_build/notebooks``.
Nothing is executed and the tutorials are not modified.

Usage::

    python scripts/make_notebooks.py                       # every examples/plot_*.py
    python scripts/make_notebooks.py examples/plot_01_*.py --out my_notebooks
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELL_MARKER = re.compile(r"^# %%.*$", re.M)


def _docstring_cell(text: str) -> tuple[str | None, str]:
    """Return the module docstring as markdown (or None) and the source after the docstring."""
    tree = ast.parse(text)
    first = tree.body[0] if tree.body else None
    if not (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return None, text
    lines = (ast.get_docstring(tree, clean=True) or "").splitlines()
    kept = [line for line in lines if not re.fullmatch(r"[=\-~]{3,}", line.strip())]
    for index, line in enumerate(kept):
        if line.strip():
            kept[index] = f"# {line.strip()}"
            break
    rest = "\n".join(text.splitlines()[first.end_lineno :])
    return "\n".join(kept).strip(), rest


def _is_text(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    return bool(lines) and all(line.lstrip().startswith("#") for line in lines)


def _uncomment(block: str) -> str:
    return "\n".join(re.sub(r"^\s*# ?", "", line) for line in block.strip().splitlines())


def split_cells(text: str) -> list[tuple[str, str]]:
    """Return the cells of a tutorial as ``(kind, source)``, kind being ``markdown`` or ``code``."""
    title, rest = _docstring_cell(text)
    cells: list[tuple[str, str]] = []
    if title:
        cells.append(("markdown", title))
    for block in CELL_MARKER.split(rest):
        if not block.strip():
            continue
        if _is_text(block):
            cells.append(("markdown", _uncomment(block)))
        else:
            cells.append(("code", block.strip("\n")))
    return cells


def to_notebook(text: str):
    """Return the tutorial as an ``nbformat`` notebook without outputs."""
    try:
        import nbformat
    except ImportError as exc:
        raise ImportError("make_notebooks needs nbformat: pip install nbformat") from exc
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(source)
        if kind == "markdown"
        else nbformat.v4.new_code_cell(source)
        for kind, source in split_cells(text)
    ]
    return notebook


def main(argv: list[str] | None = None) -> int:
    """Convert the given tutorials (default: every ``examples/plot_*.py``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args(argv)
    files = args.files or sorted((ROOT / "examples").glob("plot_*.py"))
    out = args.out or ROOT / "docs" / "_build" / "notebooks"
    out.mkdir(parents=True, exist_ok=True)
    import nbformat

    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        target = out / f"{path.stem}.ipynb"
        nbformat.write(to_notebook(path.read_text(encoding="utf-8")), target)
        print(f"{path} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""scripts/make_notebooks.py: tutorials (.py with # %% cells) to notebooks, for local viewing."""

import importlib.util
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "make_notebooks", ROOT / "scripts" / "make_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()

SOURCE = '''"""
=================
My first tutorial
=================

Some words about it.
"""

# %%
# A text cell
# with two lines.

# %%
import numpy as np

x = np.arange(3)

# %%
# A comment that belongs to the code below it
y = x + 1

# %%


# %%
print(y)
'''


def cells(text):
    return [(kind, source) for kind, source in script.split_cells(text)]


class TestSplitCells:
    def test_the_docstring_is_the_first_text_cell_with_a_markdown_title(self):
        kind, source = cells(SOURCE)[0]
        assert kind == "markdown"
        assert source.splitlines()[0] == "# My first tutorial"
        assert "Some words about it." in source
        assert "====" not in source

    def test_a_cell_of_only_comments_is_text_and_any_code_makes_it_code(self):
        found = cells(SOURCE)
        assert [kind for kind, _ in found] == ["markdown", "markdown", "code", "code", "code"]
        assert found[1][1] == "A text cell\nwith two lines."
        assert found[3][1].startswith("# A comment that belongs to the code below it\ny = x + 1")

    def test_empty_cells_are_dropped(self):
        assert all(source.strip() for _, source in cells(SOURCE))

    def test_the_docstring_is_not_repeated_as_code(self):
        assert not any(
            "My first tutorial" in source for kind, source in cells(SOURCE) if kind == "code"
        )

    def test_a_script_without_a_docstring_starts_with_its_first_cell(self):
        found = cells("# %%\nprint(1)\n")
        assert found == [("code", "print(1)")]


class TestNotebook:
    def test_it_is_a_valid_notebook_with_the_same_cells_and_no_outputs(self):
        notebook = script.to_notebook(SOURCE)
        nbformat.validate(notebook)
        assert [c.cell_type for c in notebook.cells] == [
            "markdown",
            "markdown",
            "code",
            "code",
            "code",
        ]
        assert all(c.outputs == [] for c in notebook.cells if c.cell_type == "code")

    def test_the_code_is_not_executed(self, tmp_path):
        marker = tmp_path / "ran"
        script.to_notebook(f"# %%\nopen({str(marker)!r}, 'w').close()\n")
        assert not marker.exists()


class TestCommandLine:
    def test_it_writes_one_notebook_per_tutorial_into_the_output_directory(self, tmp_path):
        source = tmp_path / "plot_01_first.py"
        source.write_text(SOURCE, encoding="utf-8")
        out = tmp_path / "notebooks"
        assert script.main([str(source), "--out", str(out)]) == 0
        written = out / "plot_01_first.ipynb"
        assert written.is_file()
        nbformat.validate(nbformat.read(written, as_version=4))
        assert source.read_text(encoding="utf-8") == SOURCE  # the tutorial is untouched

    def test_without_arguments_it_converts_every_tutorial_of_examples(self, tmp_path, monkeypatch):
        examples = tmp_path / "examples"
        examples.mkdir()
        (examples / "plot_01_a.py").write_text(SOURCE, encoding="utf-8")
        (examples / "plot_02_b.py").write_text(SOURCE, encoding="utf-8")
        (examples / "helper.py").write_text("x = 1\n", encoding="utf-8")
        monkeypatch.setattr(script, "ROOT", tmp_path)
        assert script.main(["--out", str(tmp_path / "out")]) == 0
        assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
            "plot_01_a.ipynb",
            "plot_02_b.ipynb",
        ]

    def test_a_missing_file_is_an_error_not_a_silent_skip(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            script.main([str(tmp_path / "plot_99_missing.py"), "--out", str(tmp_path / "out")])

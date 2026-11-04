"""The tutorials in examples/: the conventions they follow, and that every one of them runs."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
CLOSING_HEADING = "What this does and does not show"

GOOD = '''"""
=================================
A tutorial that follows the rules
=================================

Text for the reader.
"""

# %%
# The first cell: environment, seed and versions.
import os

import numpy as np

import dirconn

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(dirconn.__version__, np.__version__)

# %%
# What this does and does not show
# --------------------------------
# The result is predictive, not causal.
for check in [] if QUICK else []:
    print(check)
checks = []
print(checks, "assumptions.checks")
'''


def example_files():
    return sorted(EXAMPLES.glob("plot_*.py")) if EXAMPLES.is_dir() else []


def needs_sample(path):
    return re.search(r"^NEEDS_SAMPLE = True", path.read_text(encoding="utf-8"), re.M) is not None


def convention_problems(path):
    """Return what a tutorial does not do that every tutorial must do; empty when it is fine."""
    text = path.read_text(encoding="utf-8")
    problems = []
    title = re.match(r'\s*"""\s*\n(=+)\n([^\n]+)\n(=+)\n', text)
    if not title:
        problems.append("the module docstring must start with an RST title between '=' lines")
    elif min(len(title.group(1)), len(title.group(3))) < len(title.group(2)):
        problems.append("the '=' lines around the title must be at least as long as the title")
    if not re.search(r'^QUICK = os\.environ\.get\("DIRCONN_QUICK"\) == "1"', text, re.M):
        problems.append('QUICK = os.environ.get("DIRCONN_QUICK") == "1" is missing')
    if not re.search(r"^SEED = \d+", text, re.M):
        problems.append("SEED = <int> is missing")
    if "dirconn.__version__" not in text or "np.__version__" not in text:
        problems.append("the first cell must print dirconn.__version__ and np.__version__")
    if CLOSING_HEADING not in text:
        problems.append(f"the closing section '{CLOSING_HEADING}' is missing")
    if "assumptions.checks" not in text:
        problems.append("the closing section must print result.provenance.assumptions.checks")
    return problems


def run_example(path, timeout=600):
    """Run a tutorial the way the tests do: quick mode, no window, in a fresh interpreter."""
    env = dict(os.environ, DIRCONN_QUICK="1", MPLBACKEND="Agg")
    return subprocess.run(
        [sys.executable, str(path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=path.parent,
    )


class TestHarness:
    def test_a_tutorial_that_follows_the_rules_has_no_problems(self, tmp_path):
        path = tmp_path / "plot_00_good.py"
        path.write_text(GOOD, encoding="utf-8")
        assert convention_problems(path) == []

    @pytest.mark.parametrize(
        ("old", "new", "fragment"),
        [
            ('"""\n=================================\nA tutorial', '"""\nA tutorial', "title"),
            (
                '"""\n'
                + "=" * len("A tutorial that follows the rules")
                + "\nA tutorial that follows",
                '"""\n====\nA tutorial that follows',
                "at least as long",
            ),
            ('QUICK = os.environ.get("DIRCONN_QUICK") == "1"', "QUICK = False", "QUICK"),
            ("SEED = 0", "", "SEED"),
            ("dirconn.__version__, ", "", "__version__"),
            (CLOSING_HEADING, "Summary", "closing section"),
            ('print(checks, "assumptions.checks")', "print(checks)", "assumptions.checks"),
        ],
    )
    def test_each_missing_convention_is_named(self, tmp_path, old, new, fragment):
        assert old in GOOD
        path = tmp_path / "plot_00_bad.py"
        path.write_text(GOOD.replace(old, new, 1), encoding="utf-8")
        problems = convention_problems(path)
        assert len(problems) == 1
        assert fragment in problems[0]

    def test_a_tutorial_that_fails_is_reported_with_its_error(self, tmp_path):
        path = tmp_path / "plot_00_broken.py"
        path.write_text('raise RuntimeError("this tutorial is broken")\n', encoding="utf-8")
        done = run_example(path)
        assert done.returncode != 0
        assert "this tutorial is broken" in done.stderr

    def test_a_tutorial_runs_in_quick_mode_without_a_window(self, tmp_path):
        path = tmp_path / "plot_00_env.py"
        path.write_text(
            "import os\nimport matplotlib\n"
            "assert os.environ['DIRCONN_QUICK'] == '1'\n"
            "assert matplotlib.get_backend().lower() == 'agg'\n",
            encoding="utf-8",
        )
        pytest.importorskip("matplotlib")
        done = run_example(path)
        assert done.returncode == 0, done.stderr

    def test_the_sample_flag_is_read_from_the_file(self, tmp_path):
        path = tmp_path / "plot_00_sample.py"
        path.write_text("NEEDS_SAMPLE = True\n", encoding="utf-8")
        assert needs_sample(path)
        path.write_text("x = 1\n", encoding="utf-8")
        assert not needs_sample(path)


def _example_params():
    params = []
    for path in example_files():
        marks = [pytest.mark.sample_data] if needs_sample(path) else []
        params.append(pytest.param(path, id=path.name, marks=marks))
    return params


@pytest.mark.parametrize("path", _example_params())
def test_a_tutorial_follows_the_conventions(path):
    assert convention_problems(path) == []


@pytest.mark.parametrize("path", _example_params())
def test_a_tutorial_runs_to_the_end(path):
    pytest.importorskip("matplotlib")
    if needs_sample(path):
        from dirconn.datasets import SampleDataMissingError, require_sample

        try:
            require_sample()
        except SampleDataMissingError as error:
            pytest.skip(str(error).splitlines()[0])
    done = run_example(path)
    assert done.returncode == 0, f"{path.name} failed:\n{done.stderr[-3000:]}"


def view_scripts():
    return sorted(EXAMPLES.glob("view_*.py")) if EXAMPLES.is_dir() else []


@pytest.mark.parametrize("path", view_scripts(), ids=lambda p: p.name)
def test_a_view_script_compiles_and_asks_for_the_sample_data_first(path):
    """View scripts open windows, so the tests do not run them; they must still be valid Python."""
    text = path.read_text(encoding="utf-8")
    compile(text, str(path), "exec")
    assert "require_sample()" in text
    assert not re.match(
        r"plot_", path.name
    )  # a plot_ file would be run by the tests and the gallery

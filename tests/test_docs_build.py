"""The documentation builds without warnings (the tutorials are listed but not executed)."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sphinx_gallery")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build a copy of docs/ and examples/ once, so that the repository itself stays clean."""
    tmp_path = tmp_path_factory.mktemp("docs_build")
    docs = tmp_path / "docs"
    generated = ("_build", "auto_tutorials", "generated", "sg_execution_times.rst")
    shutil.copytree(ROOT / "docs", docs, ignore=shutil.ignore_patterns(*generated))
    shutil.copytree(ROOT / "examples", tmp_path / "examples")
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            str(docs),
            str(tmp_path / "html"),
            "-q",
            "-W",
            "--keep-going",
            "-D",
            "plot_gallery=0",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    return done, tmp_path / "html"


def test_the_documentation_builds_without_warnings(built):
    done, html = built
    assert done.returncode == 0, done.stderr[-3000:]
    assert (html / "index.html").is_file()
    assert (html / "api.html").is_file()


def test_every_tutorial_has_a_page_and_the_api_lists_the_public_names(built):
    done, html = built
    assert done.returncode == 0, done.stderr[-3000:]
    pages = sorted(p.name for p in (html / "auto_tutorials").glob("plot_*.html"))
    scripts = sorted(p.stem for p in (ROOT / "examples").glob("plot_*.py"))
    assert pages == [f"{name}.html" for name in scripts]
    api = (html / "api.html").read_text(encoding="utf-8")
    import dirconn

    skipped = {"DirconnWarning", "__version__", "BANDS", "datasets", "viz"}
    missing = [name for name in dirconn.__all__ if name not in skipped and f">{name}<" not in api]
    # the lower-case shorthands are documented with autofunction: their anchor is "dirconn.<name>"
    missing = [name for name in missing if f'id="dirconn.{name}"' not in api]
    assert missing == [], f"public names that the API page does not list: {missing}"


def test_the_sample_tutorials_have_their_pages_even_when_they_are_not_run(built):
    done, html = built
    assert done.returncode == 0, done.stderr[-3000:]
    page = (html / "auto_tutorials" / "plot_07_sensors_to_regions.html").read_text(encoding="utf-8")
    assert "MNE" in page  # the page exists with its text; running it would need the data

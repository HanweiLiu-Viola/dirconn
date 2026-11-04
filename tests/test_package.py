"""Smoke tests for the package skeleton."""

import subprocess
import sys
from importlib import metadata
from importlib.resources import files

# Heavy or optional dependencies that must not be imported by `import dirconn`.
OPTIONAL_MODULES = (
    "mne",
    "mne_bids",
    "mne_connectivity",
    "tigramite",
    "statsmodels",
    "matplotlib",
    "numba",
    "joblib",
)


def test_version_is_read_from_package_metadata():
    import dirconn

    assert isinstance(dirconn.__version__, str)
    assert dirconn.__version__ == metadata.version("dirconn")


def test_version_is_a_valid_pep440_string():
    from packaging.version import Version

    import dirconn

    Version(dirconn.__version__)


def test_package_is_typed():
    assert files("dirconn").joinpath("py.typed").is_file()


def test_importing_dirconn_does_not_import_heavy_dependencies():
    code = (
        "import sys, dirconn\n"
        f"loaded = [m for m in {OPTIONAL_MODULES!r} if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""

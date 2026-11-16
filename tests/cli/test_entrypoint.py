import importlib.metadata as md
import subprocess
import sys

import dirconn


def test_version_flag_prints_the_package_version():
    proc = subprocess.run(
        [sys.executable, "-m", "dirconn.cli.run", "--version"], capture_output=True, text=True
    )
    assert proc.returncode == 0 and dirconn.__version__ in proc.stdout


def test_console_script_is_declared():
    eps = md.entry_points(group="console_scripts")
    assert any(ep.name == "dirconn-bids" and ep.value == "dirconn.cli.run:main" for ep in eps)

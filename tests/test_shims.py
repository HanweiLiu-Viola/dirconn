"""Old module paths keep working (with a DeprecationWarning) after the subpackage move."""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

# old -> new; every restructure task adds its moved modules here.
MOVED = {
    "dirconn.timeseries": "dirconn.data.timeseries",
    "dirconn.channels": "dirconn.data.channels",
    "dirconn.network": "dirconn.data.network",
    "dirconn.result": "dirconn.data.result",
    "dirconn.provenance": "dirconn.data.provenance",
    "dirconn.bands": "dirconn.data.bands",
    "dirconn.rng": "dirconn.data.rng",
    "dirconn.exceptions": "dirconn.data.exceptions",
}

SRC = Path(__file__).resolve().parents[1] / "src" / "dirconn"


@pytest.mark.parametrize(("old", "new"), MOVED.items())
def test_old_path_warns_and_forwards(old, new):
    sys.modules.pop(old, None)
    with pytest.warns(DeprecationWarning, match="moved to"):
        mod = importlib.import_module(old)
    target = importlib.import_module(new)
    names = getattr(target, "__all__", None)
    if names is None:
        names = [n for n in dir(target) if not n.startswith("_")]
    assert names
    for name in names:
        assert getattr(mod, name) is getattr(target, name)


def test_no_internal_module_imports_a_shim():
    old_names = [old.rsplit(".", 1)[-1] for old in MOVED]
    pattern = re.compile(r"\bdirconn\.(" + "|".join(old_names) + r")\b")
    skip = {SRC / f"{old.rsplit('.', 1)[-1]}.py" for old in MOVED} | {SRC / "_compat.py"}
    offenders = []
    for path in SRC.rglob("*.py"):
        if path in skip:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(SRC)}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)

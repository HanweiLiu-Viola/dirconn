"""Old module paths keep working (with a DeprecationWarning) after the subpackage move."""

from __future__ import annotations

import importlib
import re
import sys
import warnings
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
    "dirconn.bids": "dirconn.io.bids",
    "dirconn.estimator": "dirconn.methods.base",
    "dirconn.mvar": "dirconn.methods.mvar",
    "dirconn.granger": "dirconn.methods.granger",
    "dirconn.statespace": "dirconn.methods.statespace",
    "dirconn.spectral": "dirconn.methods.spectral",
    "dirconn.spectral_estimators": "dirconn.methods.spectral_estimators",
    "dirconn.multitaper_granger": "dirconn.methods.multitaper_granger",
    "dirconn.coherency": "dirconn.methods.coherency",
    "dirconn.time_reversal": "dirconn.methods.time_reversal",
    "dirconn.transfer_entropy": "dirconn.methods.transfer_entropy",
    "dirconn.pcmci": "dirconn.methods.pcmci",
    "dirconn.tvmvar": "dirconn.methods.timevarying.tvmvar",
    "dirconn.timevarying": "dirconn.methods.timevarying.timevarying",
    "dirconn.windowed": "dirconn.methods.timevarying.windowed",
    "dirconn.nonstationary": "dirconn.methods.timevarying.nonstationary",
    "dirconn.surrogates": "dirconn.stats.surrogates",
    "dirconn.surrogate_test": "dirconn.stats.surrogate_test",
    "dirconn.significance": "dirconn.stats.significance",
    "dirconn.analytic": "dirconn.stats.analytic",
    "dirconn.bootstrap": "dirconn.stats.bootstrap",
    "dirconn.contrast": "dirconn.stats.contrast",
    "dirconn.correction": "dirconn.stats.correction",
    "dirconn.baseline": "dirconn.stats.baseline",
    "dirconn.order_stability": "dirconn.diagnostics.order_stability",
    "dirconn.leakage": "dirconn.diagnostics.leakage",
    "dirconn.regions": "dirconn.diagnostics.regions",
}

SRC = Path(__file__).resolve().parents[1] / "src" / "dirconn"
TESTS = Path(__file__).resolve().parent


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
    import dirconn

    last = [old.rsplit(".", 1)[-1] for old in MOVED]
    # A module that shares its name with a public function (``bootstrap``, ``pcmci``,
    # ``surrogate_test``) is only a shim import in the ``from dirconn.<m> import`` and
    # ``import dirconn.<m>`` forms; elsewhere ``dirconn.<m>`` and ``from dirconn import <m>``
    # name the function.
    plain = "|".join(n for n in last if n not in dirconn.__all__)
    shared = "|".join(n for n in last if n in dirconn.__all__)
    dotted = re.compile(
        r"\bdirconn\.(" + plain + r")\b|\b(?:from|import) dirconn\.(" + shared + r")\b"
    )
    from_import = re.compile(r"^\s*from dirconn import [^\n]*\b(" + plain + r")\b")
    skip = {SRC / f"{old.rsplit('.', 1)[-1]}.py" for old in MOVED}
    skip |= {SRC / "_compat.py", Path(__file__).resolve()}
    offenders = []
    for path in [*SRC.rglob("*.py"), *TESTS.rglob("*.py")]:
        if path in skip:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if dotted.search(line) or from_import.search(line):
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_datasets_alias_is_the_io_module():
    import dirconn
    import dirconn.io.datasets

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        import dirconn.datasets
    assert dirconn.datasets is dirconn.io.datasets

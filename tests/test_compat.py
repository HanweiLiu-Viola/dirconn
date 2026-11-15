import importlib
import sys
import warnings

import pytest

_NEW = """\
__all__ = ["func"]


def func():
    return 42


_private = 1
"""

_OLD = """\
from dirconn._compat import redirect

__getattr__, __all__ = redirect(__name__, "pkg.new")
"""

_NEW_NO_ALL = """\
def func():
    return 42


_private = 1
"""


def _purge():
    for name in ("pkg", "pkg.old", "pkg.new"):
        sys.modules.pop(name, None)


@pytest.fixture
def shim(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "new.py").write_text(_NEW)
    (pkg / "old.py").write_text(_OLD)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    _purge()
    yield pkg
    _purge()


def test_shim_warns_once_and_forwards(shim):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old = importlib.import_module("pkg.old")
        assert old.func is importlib.import_module("pkg.new").func
        assert old._private == 1
    assert [x.category for x in w] == [DeprecationWarning]
    assert str(w[0].message) == "pkg.old moved to pkg.new; the old path is removed in dirconn 0.4.0"


def test_shim_all_matches_new_module(shim):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert importlib.import_module("pkg.old").__all__ == ["func"]


def test_star_import_from_shim(shim):
    ns = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec("from pkg.old import *", ns)
    assert "func" in ns


def test_missing_attribute_raises(shim):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = importlib.import_module("pkg.old")
    with pytest.raises(AttributeError):
        _ = old.nope


def test_all_defaults_to_public_names(shim):
    (shim / "new.py").write_text(_NEW_NO_ALL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = importlib.import_module("pkg.old")
    assert old.__all__ == ["func"]

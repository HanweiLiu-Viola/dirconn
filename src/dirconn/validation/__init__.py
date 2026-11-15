"""Validation: protocols, sweeps, benchmark, metrics, studies and the method-validation runner."""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from typing import Any

# The public surface of this package is its submodules; the names of the old module are
# forwarded by ``__getattr__`` (deprecated), so a star import brings nothing.
__all__: list[str] = []

_SUBMODULES = frozenset(m.name for m in pkgutil.iter_modules(__path__))


def __getattr__(name: str) -> Any:
    """Forward the names of the old ``dirconn.validation`` module, with a deprecation warning."""
    if name.startswith("__") or name in _SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    old = importlib.import_module("dirconn.validation.method_validation")
    try:
        value = getattr(old, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    warnings.warn(
        f"dirconn.validation.{name} moved to dirconn.validation.method_validation.{name}; "
        "the old path is removed in dirconn 0.4.0",
        DeprecationWarning,
        stacklevel=2,
    )
    return value

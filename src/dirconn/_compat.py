"""Helper for the old-path module shims left behind by the subpackage restructure."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from typing import Any

__all__: list[str] = []


def redirect(old_name: str, new_name: str) -> tuple[Callable[[str], Any], list[str]]:
    """Redirect a moved module to its new location.

    Imports ``new_name``, emits one ``DeprecationWarning`` (when called, i.e. when the shim
    is first imported), and returns what the shim needs to forward attribute access.

    Parameters
    ----------
    old_name : str
        Dotted name of the old module (the shim's ``__name__``).
    new_name : str
        Dotted name of the module the contents moved to.

    Returns
    -------
    getattr_func : callable
        Module-level ``__getattr__`` forwarding every attribute, private names included,
        to the new module; raises ``AttributeError`` when the new module lacks it.
    all_names : list of str
        The new module's ``__all__``, or its public (non-underscore) names if it has none.
    """
    warnings.warn(
        f"{old_name} moved to {new_name}; the old path is removed in dirconn 0.4.0",
        DeprecationWarning,
        stacklevel=3,
    )
    module = importlib.import_module(new_name)

    def __getattr__(name: str) -> Any:
        try:
            return getattr(module, name)
        except AttributeError:
            raise AttributeError(f"module {old_name!r} has no attribute {name!r}") from None

    declared = getattr(module, "__all__", None)
    if declared is None:
        all_names = [n for n in vars(module) if not n.startswith("_")]
    else:
        all_names = list(declared)
    return __getattr__, all_names

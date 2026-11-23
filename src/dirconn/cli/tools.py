"""Helpers and subcommands of ``dirconn-bids`` that do not run the pipeline.

Importable with the core dependencies only: everything that needs the ``app`` extra (pydantic,
the configuration model) is imported inside the functions, after `require_app`.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dirconn.methods.registry import REGISTRY

if TYPE_CHECKING:
    from dirconn.workflows.config import PipelineConfig

__all__ = [
    "APP_MODULES",
    "config_errors",
    "error",
    "list_methods",
    "load_or_report",
    "require_app",
    "validate_config",
]

_ERROR = "dirconn-bids: error: "
#: Third-party modules of the ``app`` extra.
APP_MODULES: tuple[str, ...] = ("pydantic", "joblib", "jinja2", "matplotlib") + (
    ("tomli",) if sys.version_info < (3, 11) else ()
)


def error(message: str) -> None:
    """Print ``message`` to stderr with the ``dirconn-bids: error:`` prefix."""
    print(f"{_ERROR}{message}", file=sys.stderr)


def require_app() -> bool:
    """Check that the modules of the ``app`` extra import.

    Returns
    -------
    bool
        ``True`` when they do; otherwise ``False`` after printing how to install them.
    """
    for name in APP_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            error('the command line needs the app extra: pip install "dirconn[app]"')
            return False
    return True


def config_errors(exc: Exception) -> list[str]:
    """Return one message per problem of a failed configuration load or override.

    Parameters
    ----------
    exc : Exception
        A pydantic ``ValidationError`` (one message per error, prefixed with its dotted
        location) or any other exception (its text).

    Returns
    -------
    list of str
        The messages.
    """
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        messages = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err["loc"])
            messages.append(f"{loc}: {err['msg']}" if loc else str(err["msg"]))
        return messages
    return [str(exc)]


def load_or_report(path: Path | None) -> PipelineConfig | None:
    """Load a configuration, or print its errors and return ``None``.

    Requires the ``app`` extra (see `require_app`).

    Parameters
    ----------
    path : Path or None
        TOML file; ``None`` loads the built-in default.

    Returns
    -------
    PipelineConfig or None
        The configuration, ``None`` after printing the errors.
    """
    from dirconn.workflows.config import load_config

    try:
        return load_config(path)
    except OSError as exc:
        error(f"cannot read configuration {path}: {exc.strerror or exc}")
    except ValueError as exc:  # includes TOML syntax and pydantic validation errors
        for message in config_errors(exc):
            error(f"configuration {path}: {message}")
    return None


def validate_config(path: Path) -> int:
    """Check a configuration file and print ``OK <config_hash>``.

    Parameters
    ----------
    path : Path
        TOML configuration.

    Returns
    -------
    int
        0 when valid; 2 after printing the errors (or the missing ``app`` extra) to stderr.
    """
    if not require_app():
        return 2
    from dirconn.workflows.config import config_hash

    cfg = load_or_report(Path(path))
    if cfg is None:
        return 2
    print(f"OK {config_hash(cfg)}")
    return 0


def list_methods() -> int:
    """Print the registered methods, one per line, after a header line.

    The columns are ``name`` (the name used in ``estimation.methods``), ``report_kind``,
    ``frequency_resolved``, ``supports_mask`` and ``backend``.

    Returns
    -------
    int
        0.
    """
    rows = [("name", "report_kind", "frequency_resolved", "supports_mask", "backend")]
    for name in sorted(REGISTRY):
        info = REGISTRY[name].info
        rows.append(
            (
                name,
                info.report_kind,
                str(info.frequency_resolved).lower(),
                str(info.supports_mask).lower(),
                info.backend,
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    for row in rows:
        cells = [cell.ljust(width) for cell, width in zip(row[:4], widths, strict=True)]
        print("  ".join([*cells, row[4]]))
    return 0

"""Subcommands of ``dirconn-bids`` that do not run the pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

from dirconn.methods.registry import REGISTRY
from dirconn.workflows.config import PipelineConfig, config_hash, load_config

__all__ = ["config_errors", "error", "list_methods", "load_or_report", "validate_config"]

_ERROR = "dirconn-bids: error: "


def error(message: str) -> None:
    """Print ``message`` to stderr with the ``dirconn-bids: error:`` prefix."""
    print(f"{_ERROR}{message}", file=sys.stderr)


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
    if isinstance(exc, ValidationError):
        messages = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err["loc"])
            messages.append(f"{loc}: {err['msg']}" if loc else str(err["msg"]))
        return messages
    return [str(exc)]


def load_or_report(path: Path | None) -> PipelineConfig | None:
    """Load a configuration, or print its errors and return ``None``.

    Parameters
    ----------
    path : Path or None
        TOML file; ``None`` loads the built-in default.

    Returns
    -------
    PipelineConfig or None
        The configuration, ``None`` after printing the errors.
    """
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
        0 when valid; 2 after printing the errors to stderr.
    """
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

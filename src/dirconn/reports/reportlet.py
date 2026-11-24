"""The ``Reportlet`` record: one figure or table of a participant report, stored as JSON (+ SVG).

A reportlet is written next to the figures of a participant as
``<label>_desc-<desc>_<name>.json`` (all fields except ``svg``) and, when it has a figure, a
``<label>_desc-<desc>_<name>.svg`` beside it. ``table`` and ``data`` hold plain JSON values only,
so a reportlet survives a write/read round trip unchanged.

This module imports neither matplotlib nor jinja2.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, get_args

__all__ = [
    "STATUS_ORDER",
    "Reportlet",
    "Status",
    "read_reportlets",
    "worst_status",
    "write_reportlet",
]

Status = Literal["pass", "warn", "fail", "info"]

#: Statuses from the mildest to the worst; :func:`worst_status` picks the last one present.
STATUS_ORDER: tuple[Status, ...] = ("info", "pass", "warn", "fail")

_STATUSES = frozenset(get_args(Status))


def _check_part(field: str, value: str) -> None:
    if not value or "_" in value or "/" in value or "\\" in value:
        raise ValueError(
            f"Reportlet.{field} must be non-empty and contain no '_' or path separator "
            f"(it becomes a filename part); got {value!r}"
        )


@dataclass(frozen=True)
class Reportlet:
    """One figure or table of a report.

    Parameters
    ----------
    name : str
        The suffix part of the file name, for example ``"matrix-alpha"`` or ``"order"``.
    section : str
        Id of the report section the reportlet belongs to.
    desc : str
        A method name or ``"prep"``, ``"diag"``, ``"summary"``; the ``desc-`` entity of the name.
    title : str
        Heading shown in the report.
    caption : str
        Text under the figure or table.
    status : {"pass", "warn", "fail", "info"}
        Outcome flag of the check the reportlet shows (``"info"`` when it is not a check).
    svg : str or None
        The figure as SVG text; stored in a separate ``.svg`` file.
    table : list of dict or None
        Table rows as plain JSON values.
    data : dict or None
        Further plain JSON values for the template.

    Raises
    ------
    ValueError
        If ``status`` is unknown, or ``desc`` or ``name`` is empty or contains ``_`` or a path
        separator.
    """

    name: str
    section: str
    desc: str
    title: str
    caption: str
    status: Status = "info"
    svg: str | None = None
    table: list[dict[str, Any]] | None = None
    data: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate ``status``, ``desc`` and ``name``."""
        if self.status not in _STATUSES:
            raise ValueError(f"Reportlet.status must be one of {STATUS_ORDER}; got {self.status!r}")
        _check_part("desc", self.desc)
        _check_part("name", self.name)


def worst_status(statuses: Iterable[Status]) -> Status:
    """Return the worst of ``statuses`` (``fail`` > ``warn`` > ``pass`` > ``info``).

    Parameters
    ----------
    statuses : iterable of str
        Statuses to combine.

    Returns
    -------
    str
        The worst status, ``"info"`` for an empty input.
    """
    worst = 0
    for s in statuses:
        worst = max(worst, STATUS_ORDER.index(s))
    return STATUS_ORDER[worst]


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file in the same directory."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_reportlet(figures_dir: Path, label: str, r: Reportlet) -> list[Path]:
    """Write a reportlet as ``<label>_desc-<desc>_<name>.json`` (and ``.svg`` if it has one).

    A stale ``.svg`` of the same reportlet is removed when ``r.svg`` is ``None``.

    Parameters
    ----------
    figures_dir : pathlib.Path
        Directory of the figures; created if missing.
    label : str
        BIDS-like label of the participant unit, for example ``sub-01_task-rest``.
    r : Reportlet
        The reportlet.

    Returns
    -------
    list of pathlib.Path
        The files written (JSON first).
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_desc-{r.desc}_{r.name}"
    json_path = figures_dir / f"{stem}.json"
    svg_path = figures_dir / f"{stem}.svg"
    payload = {k: v for k, v in asdict(r).items() if k != "svg"}
    written: list[Path] = []
    if r.svg is not None:
        _atomic_write(svg_path, r.svg)
        written.append(svg_path)
    else:
        svg_path.unlink(missing_ok=True)
    _atomic_write(json_path, json.dumps(payload, indent=2) + "\n")
    return [json_path, *written]


def read_reportlets(figures_dir: Path, label: str) -> list[Reportlet]:
    """Read the reportlets of ``label`` from ``figures_dir``, sorted by file name.

    Only files named ``<label>_desc-<desc>_<name>.json`` are read, so a label that merely shares a
    prefix with another is not mixed up with it. JSON files without a ``section`` key (such as
    ``<label>_diagnostics.json``) are skipped.

    Parameters
    ----------
    figures_dir : pathlib.Path
        Directory of the figures; a missing directory gives an empty list.
    label : str
        BIDS-like label of the participant unit.

    Returns
    -------
    list of Reportlet
        The reportlets, with ``svg`` read back from the ``.svg`` file when present.
    """
    figures_dir = Path(figures_dir)
    if not figures_dir.is_dir():
        return []
    prefix = f"{label}_desc-"
    out: list[Reportlet] = []
    for path in sorted(figures_dir.iterdir(), key=lambda p: p.name):
        if not (path.name.startswith(prefix) and path.suffix == ".json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "section" not in payload:
            continue
        svg_path = path.with_suffix(".svg")
        svg = svg_path.read_text(encoding="utf-8") if svg_path.is_file() else None
        out.append(Reportlet(**{**payload, "svg": svg}))
    return out

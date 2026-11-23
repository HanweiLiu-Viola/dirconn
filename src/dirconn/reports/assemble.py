"""Assemble the offline participant report ``sub-<subject>.html`` from the reportlets.

:func:`assemble_participant` reads the reportlet files that :mod:`dirconn.reports.build` wrote,
orders them by the packaged ``_data/spec.yml`` and renders one self-contained HTML page (inline
CSS and SVG, no script, no external resource except DOI links). It never computes a figure.
:func:`build_participant_report` runs the whole chain for one subject: reportlets, methods
boilerplate and page.

Only the SVG text that :func:`~dirconn.reports.reportlet.read_reportlets` reads from reportlet
files is inserted without escaping; every other value goes through Jinja2 autoescaping.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from importlib import metadata, resources
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, PackageLoader, StrictUndefined
from markupsafe import Markup

from dirconn.reports.boilerplate import (
    Boilerplate,
    collect_facts,
    make_boilerplate,
    reference_entries,
)
from dirconn.reports.build import build_reportlets
from dirconn.reports.collect import CONTROL_DESC, SubjectOutputs, UnitOutputs, collect_subject
from dirconn.reports.reportlet import Reportlet, Status, read_reportlets, worst_status

__all__ = ["assemble_participant", "build_participant_report", "load_spec"]

#: Tables with more rows than this are folded into a ``<details>`` element.
MAX_OPEN_ROWS = 50

_FIXED_DESCS = frozenset({"prep", "diag", "summary", CONTROL_DESC})
_DASH = "—"
_ID_ATTR = re.compile(r'(?<![\w:-])id="([^"]*)"')
_URL_REF = re.compile(r"url\(#([^)]*)\)")
_HREF_REF = re.compile(r'href="#([^"]*)"')
_UNSAFE_SVG = re.compile(r"<script\b|javascript:", re.IGNORECASE)


def load_spec() -> list[dict[str, Any]]:
    """Return the sections of the packaged ``_data/spec.yml``.

    Returns
    -------
    list of dict
        One dict per section (``id``, ``title``, ``reportlets`` and optionally ``boilerplate``),
        in display order. Each reportlet entry has ``desc`` and ``name`` (fnmatch patterns) and
        optionally ``optional`` and ``scope`` (``"subject"``).
    """
    text = (resources.files("dirconn.reports") / "_data" / "spec.yml").read_text(encoding="utf-8")
    sections = yaml.safe_load(text)["sections"]
    return list(sections)


# --------------------------------------------------------------------------------------------
# view model


@dataclass(frozen=True)
class _TableView:
    columns: list[str]
    rows: list[list[str]]
    n_rows: int
    collapsed: bool


@dataclass(frozen=True)
class _Item:
    id: str
    title: str
    status: Status
    caption: str
    svg: Markup | None = None
    table: _TableView | None = None


@dataclass
class _UnitView:
    id: str
    label: str
    items: list[_Item] = field(default_factory=list)


@dataclass
class _SectionView:
    id: str
    title: str
    status: Status = "info"
    items: list[_Item] = field(default_factory=list)
    units: list[_UnitView] = field(default_factory=list)
    methods: dict[str, Any] | None = None
    empty: bool = False


def _cell(value: Any) -> str:
    if value is None:
        return _DASH
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return format(value, ".4g")
    if isinstance(value, (list, tuple)):
        return ", ".join(_cell(v) for v in value)
    if isinstance(value, Mapping):
        return ", ".join(f"{k}: {_cell(v)}" for k, v in value.items())
    return str(value)


def _table_view(rows: Sequence[Mapping[str, Any]] | None) -> _TableView | None:
    """Return the table of ``rows`` (columns in first-seen order), ``None`` when empty."""
    if not rows:
        return None
    columns = list(dict.fromkeys(key for row in rows for key in row))
    cells = [[_cell(row.get(c)) for c in columns] for row in rows]
    return _TableView(columns, cells, len(cells), len(cells) > MAX_OPEN_ROWS)


def _prefix_ids(svg: str, prefix: str) -> str:
    """Prefix every ``id`` of ``svg`` and the ``url(#..)`` and ``href="#.."`` pointing to it.

    A repeated ``id`` is dropped from its later elements.
    """
    seen: set[str] = set()

    def rename(m: re.Match[str]) -> str:
        # matplotlib repeats the id of an embedded image (named by a content hash) when the same
        # image is drawn twice; keep the first, the copies are identical.
        if m.group(1) in seen:
            return ""
        seen.add(m.group(1))
        return f'id="{prefix}{m.group(1)}"'

    svg = _ID_ATTR.sub(rename, svg)
    svg = _URL_REF.sub(lambda m: f"url(#{prefix}{m.group(1)})", svg)
    return _HREF_REF.sub(lambda m: f'href="#{prefix}{m.group(1)}"', svg)


class _Page:
    """Turns reportlets into template items, numbering the inline SVGs of the page."""

    def __init__(self) -> None:
        self._n_svg = 0

    def svg(self, text: str | None) -> Markup | None:
        if not text:
            return None
        body = text.strip()
        if not (body.startswith("<svg") and body.endswith("</svg>")) or _UNSAFE_SVG.search(body):
            return None
        self._n_svg += 1
        return Markup(_prefix_ids(body, f"r{self._n_svg}-"))  # noqa: S704

    def item(self, label: str, r: Reportlet, extra_rows: Sequence[Mapping[str, Any]] = ()) -> _Item:
        rows = [*(r.table or []), *extra_rows]
        caption = r.caption
        svg = self.svg(r.svg)
        if r.svg and svg is None:
            caption += " (The figure is not shown: its file is not a plain SVG.)"
        status = r.status
        if extra_rows and status in ("pass", "info"):
            status = "info"
        return _Item(
            id=f"{label}-{r.desc}-{r.name}",
            title=r.title,
            status=status,
            caption=caption,
            svg=svg,
            table=_table_view(rows),
        )


# --------------------------------------------------------------------------------------------
# matching


def _matches(entry: Mapping[str, Any], r: Reportlet) -> bool:
    """Whether the reportlet ``r`` is selected by the spec ``entry`` (fnmatch on desc and name)."""
    desc = str(entry["desc"])
    if desc == "*":
        if r.desc in _FIXED_DESCS:
            return False
    elif not fnmatchcase(r.desc, desc):
        return False
    return fnmatchcase(r.name, str(entry["name"]))


def _select(entries: Sequence[Mapping[str, Any]], pool: Sequence[Reportlet]) -> list[Reportlet]:
    """Reportlets of ``pool`` in entry order, then by desc and name; each at most once."""
    chosen: list[Reportlet] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        for r in sorted((r for r in pool if _matches(entry, r)), key=lambda r: (r.desc, r.name)):
            if (r.desc, r.name) not in seen:
                seen.add((r.desc, r.name))
                chosen.append(r)
    return chosen


def _missing(
    entries: Sequence[Mapping[str, Any]],
    label: str,
    descs: Sequence[str] | None,
    pool: Sequence[Reportlet],
    listed: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Error rows for the required entries without a reportlet and without a row in the table.

    ``descs`` are the methods standing for ``desc: "*"``; ``None`` for the subject scope.
    """
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("optional"):
            continue
        desc, name = str(entry["desc"]), str(entry["name"])
        wanted = [desc] if desc != "*" else list(descs or ())
        for d in wanted:
            if any(r.desc == d and fnmatchcase(r.name, name) for r in pool):
                continue
            if (label, f"{d}/{name}") in listed or (label, f"{d}/*") in listed:
                continue
            rows.append(
                {
                    "kind": "missing reportlet",
                    "label": label,
                    "reportlet": f"{d}/{name}",
                    "reason": "not produced",
                }
            )
    return rows


# --------------------------------------------------------------------------------------------
# assembling


def _version() -> str:
    return metadata.version("dirconn")


def _environment() -> Environment:
    return Environment(
        loader=PackageLoader("dirconn.reports", "_data/templates"),
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _css() -> Markup:
    path = resources.files("dirconn.reports") / "_data" / "templates" / "report.css"
    return Markup(path.read_text(encoding="utf-8"))  # noqa: S704 - packaged, not user input


def _unit_pool(unit: UnitOutputs) -> list[Reportlet]:
    return read_reportlets(unit.figures_dir, unit.label)


def _section_status(section: _SectionView) -> Status:
    every = [*section.items, *(i for u in section.units for i in u.items)]
    return worst_status(i.status for i in every)


def _methods_view(boilerplate: Boilerplate | None, error: str | None) -> dict[str, Any]:
    if error is not None:
        return {"error": error, "paragraphs": (), "references": []}
    if boilerplate is None:
        return {"error": None, "paragraphs": (), "references": []}
    return {
        "error": None,
        "paragraphs": list(boilerplate.paragraphs),
        "references": reference_entries(boilerplate),
    }


def _render(
    outputs: SubjectOutputs, boilerplate: Boilerplate | None, boilerplate_error: str | None
) -> str:
    subject_label = f"sub-{outputs.subject}"
    spec = load_spec()
    page = _Page()
    subject_pool = read_reportlets(outputs.root / subject_label / "figures", subject_label)
    unit_pools = {u.label: _unit_pool(u) for u in outputs.units}

    errors_entry = next(
        (
            e
            for s in spec
            for e in s["reportlets"]
            if (e["desc"], e["name"]) == ("summary", "errors")
        ),
        None,
    )
    errors_reportlet = next(
        (r for r in subject_pool if errors_entry is not None and _matches(errors_entry, r)), None
    )
    listed = {
        (str(row.get("label")), str(row.get("reportlet")))
        for row in (errors_reportlet.table or [] if errors_reportlet else [])
    }
    extra: list[dict[str, Any]] = []
    for section in spec:
        entries = section["reportlets"]
        subject_entries = [e for e in entries if e.get("scope") == "subject"]
        unit_entries = [e for e in entries if e.get("scope") != "subject"]
        extra += _missing(subject_entries, subject_label, None, subject_pool, listed)
        for unit in outputs.units:
            extra += _missing(
                unit_entries, unit.label, sorted(unit.methods), unit_pools[unit.label], listed
            )
    if boilerplate_error is not None:
        extra.append(
            {
                "kind": "boilerplate failed",
                "label": subject_label,
                "reportlet": "methods/boilerplate",
                "reason": boilerplate_error,
            }
        )

    views: list[_SectionView] = []
    for section in spec:
        view = _SectionView(id=str(section["id"]), title=str(section["title"]))
        entries = section["reportlets"]
        subject_entries = [e for e in entries if e.get("scope") == "subject"]
        unit_entries = [e for e in entries if e.get("scope") != "subject"]
        if section["id"] == "errors":
            if errors_reportlet is not None:
                view.items.append(page.item(subject_label, errors_reportlet, extra))
            elif extra:
                view.items.append(
                    page.item(
                        subject_label,
                        Reportlet(
                            name="errors",
                            section="errors",
                            desc="summary",
                            title="Errors",
                            caption="The errors reportlet is missing; these rows were found "
                            "while assembling the report.",
                            status="info",
                            table=[],
                        ),
                        extra,
                    )
                )
        else:
            for r in _select(subject_entries, subject_pool):
                view.items.append(page.item(subject_label, r))
        if unit_entries:
            for unit in outputs.units:
                unit_view = _UnitView(id=f"{view.id}-{unit.label}", label=unit.label)
                unit_view.items = [
                    page.item(unit.label, r) for r in _select(unit_entries, unit_pools[unit.label])
                ]
                view.units.append(unit_view)
        if section.get("boilerplate"):
            view.methods = _methods_view(boilerplate, boilerplate_error)
        view.empty = not (view.items or view.units or view.methods)
        view.status = _section_status(view)
        views.append(view)

    overview = next((i for v in views for i in v.items if i.id.endswith("-summary-overview")), None)
    status: Status = (
        overview.status
        if overview is not None
        else worst_status(v.status for v in views if v.id != "methods")
    )
    return (
        _environment()
        .get_template("participant.html")
        .render(
            subject=outputs.subject,
            css=_css(),
            status=status,
            n_units=len(outputs.units),
            sections=views,
            version=_version(),
            generated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    )


def _write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _assemble(
    root: Path, subject: str, boilerplate: Boilerplate | None, boilerplate_error: str | None
) -> Path:
    root = Path(root)
    text = _render(collect_subject(root, subject), boilerplate, boilerplate_error)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"sub-{subject}.html"
    _write(target, text)
    return target


def assemble_participant(
    root: Path, subject: str, *, boilerplate: Boilerplate | None = None
) -> Path:
    """Write ``root/sub-<subject>.html`` from the reportlets already on disk.

    Reportlets are read, never computed. A required reportlet (see ``_data/spec.yml``) that is
    absent gets a ``missing reportlet`` row in the Errors section unless the errors table already
    names it.

    Parameters
    ----------
    root : pathlib.Path
        Dataset root (``<out_dir>/dirconn``).
    subject : str
        Subject label without ``sub-``.
    boilerplate : Boilerplate, optional
        Methods text for the Methods section; the section says that none was supplied when
        omitted.

    Returns
    -------
    pathlib.Path
        The page, replaced atomically.
    """
    return _assemble(root, subject, boilerplate, None)


def _one_line(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def build_participant_report(root: Path, subject: str) -> Path:
    """Build the reportlets, the methods boilerplate and the page of one subject.

    A failure while making the boilerplate does not stop the report: the Methods section says so
    and the Errors section gets a row.

    Parameters
    ----------
    root : pathlib.Path
        Dataset root (``<out_dir>/dirconn``).
    subject : str
        Subject label without ``sub-``.

    Returns
    -------
    pathlib.Path
        ``root/sub-<subject>.html``.
    """
    root = Path(root)
    build_reportlets(root, subject)
    boilerplate: Boilerplate | None = None
    error: str | None = None
    try:
        config = collect_subject(root, subject).config
        if config is None:
            raise FileNotFoundError(f"no dirconn.toml in {root}")
        facts, selections = collect_facts(root, [subject])
        boilerplate = make_boilerplate(
            config, facts, version=_version(), order_selections=selections
        )
    except Exception as exc:
        error = _one_line(exc)
    return _assemble(root, subject, boilerplate, error)

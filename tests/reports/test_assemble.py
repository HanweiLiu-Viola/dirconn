"""Tests for ``dirconn.reports.assemble``."""

from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import html5lib
import pytest
from markupsafe import escape

from dirconn.reports import assemble
from dirconn.reports.assemble import (
    assemble_participant,
    build_participant_report,
    load_spec,
)
from dirconn.reports.build import SECTIONS
from dirconn.reports.collect import collect_subject

FORBIDDEN = (
    r'src="http',
    r'href="http',
    r"<script",
    r'<link\b[^>]*rel="stylesheet"',
    r"@import",
    r"url\(\s*['\"]?https?:",
)
SVG_BLOCK = re.compile(r"<svg\b.*?</svg>", re.DOTALL)


@dataclass(frozen=True)
class Report:
    root: Path
    path: Path
    text: str


def _copy(run: Any, tmp: Path) -> Path:
    shutil.copytree(run.root.parent, tmp / "out")
    return tmp / "out" / "dirconn"


@pytest.fixture(scope="module")
def built_root(reportlets_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Dataset root with both subjects' reports built once, on a copy of ``reportlets_root``.

    The reportlets are already on disk (``reportlets_root`` is built once per session), so
    ``build_reportlets`` is stubbed here and only the boilerplate and the page are made.
    Read-only; tests that change files copy it with ``_copy_built``.
    """
    root = _copy_built(reportlets_root, tmp_path_factory.mktemp("assemble"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assemble, "build_reportlets", lambda *args, **kwargs: [])
        for subject in ("01", "02"):
            build_participant_report(root, subject)
    return root


def _copy_built(root: Path, tmp: Path) -> Path:
    """Copy of a built dataset root; its reportlets are on disk, so only assembling is left."""
    shutil.copytree(root.parent, tmp / "out")
    return tmp / "out" / "dirconn"


@pytest.fixture(scope="module")
def report(built_root: Path) -> Report:
    """Sub-01 of a copy of ``pipeline_run``, built and assembled once."""
    path = built_root / "sub-01.html"
    return Report(built_root, path, path.read_text(encoding="utf-8"))


def _parse(text: str) -> Any:
    return html5lib.HTMLParser(strict=True).parse(text)


def _section(text: str, name: str) -> str:
    start = text.index(f'<section class="report-section" id="section-{name}">')
    end = text.find('<section class="report-section"', start + 1)
    return text[start : end if end != -1 else text.index("</main>")]


def test_spec_and_sections_agree() -> None:
    assert [s["id"] for s in load_spec()] == list(SECTIONS)


def test_report_is_written_next_to_the_subject(report: Report) -> None:
    assert report.path == report.root / "sub-01.html"
    assert report.text.startswith("<!DOCTYPE html>")
    assert not list(report.root.glob(".sub-01.html*"))


def test_report_is_small(report: Report) -> None:
    assert report.path.stat().st_size < 5_000_000


def test_report_is_valid_html(report: Report) -> None:
    _parse(report.text)


def test_report_is_offline(report: Report) -> None:
    text = report.text.replace('href="https://doi.org/', 'href="DOI/')
    for pattern in FORBIDDEN:
        assert not re.search(pattern, text), pattern
    hrefs = re.findall(r'\bhref="([^"]*)"', report.text)
    assert hrefs
    for href in hrefs:
        assert href.startswith(("#", "https://doi.org/", "data:image/")), href


def test_sections_follow_the_spec_order(report: Report) -> None:
    ids = re.findall(r'<section class="report-section" id="section-([a-z]+)">', report.text)
    assert ids == [s["id"] for s in load_spec()]
    nav = re.findall(r'<a href="#section-([a-z]+)"', report.text)
    assert nav == ids
    positions = [report.text.index(f'id="section-{i}"') for i in ids]
    assert positions == sorted(positions)


def test_every_unit_is_in_each_unit_section(pipeline_run: Any, report: Report) -> None:
    labels = [u.label for u in collect_subject(pipeline_run.root, "01").units]
    assert len(labels) >= 2
    for name in ("input", "diagnostics", "connectivity", "statistics"):
        text = _section(report.text, name)
        for label in labels:
            assert f"<h3>{label}</h3>" in text, (name, label)


def test_every_spec_reportlet_present_or_listed_missing(pipeline_run: Any, report: Report) -> None:
    outputs = collect_subject(pipeline_run.root, "01")
    errors = _section(report.text, "errors")
    for section in load_spec():
        for entry in section["reportlets"]:
            if entry.get("optional"):
                continue
            desc, name = entry["desc"], entry["name"]
            if entry.get("scope") == "subject":
                targets = [("sub-01", [desc])]
            else:
                targets = [
                    (u.label, [desc] if desc != "*" else sorted(u.methods)) for u in outputs.units
                ]
            for label, descs in targets:
                for d in descs:
                    ident = f"{label}-{d}-{name}"
                    assert f'id="{ident}"' in report.text or (
                        f"{d}/{name}" in errors or f"{d}/*" in errors
                    ), ident


def test_untested_note_is_shown(pipeline_run: Any, report: Report) -> None:
    outputs = collect_subject(pipeline_run.root, "01")
    notes = {
        out.statistics_note
        for unit in outputs.units
        for out in unit.methods.values()
        if out.desc == "adtf" and out.statistics_note
    }
    assert notes
    for note in notes:
        assert str(escape(note)) in report.text


def test_svg_ids_are_unique_and_resolve(report: Report) -> None:
    ids = re.findall(r'\bid="([^"]*)"', report.text)
    assert len(ids) == len(set(ids))
    refs = re.findall(r"url\(#([^)]+)\)", report.text)
    refs += re.findall(r'href="#([^"]+)"', report.text)
    assert refs
    missing = [r for r in refs if r not in set(ids)]
    assert not missing
    assert len(SVG_BLOCK.findall(report.text)) > 5


def test_methods_section_has_boilerplate(report: Report) -> None:
    text = _section(report.text, "methods")
    assert "predictive, not interventional" in text
    assert "<ol" in text and text.count("<li") >= 3
    assert 'href="https://doi.org/' in text


def test_status_is_shown_as_text(report: Report) -> None:
    pattern = r'<span class="badge st-(pass|warn|fail|info)">(PASS|WARN|FAIL|INFO)</span>'
    assert re.search(pattern, report.text)


def test_header_and_footer(report: Report) -> None:
    assert "dirconn report: sub-01" in report.text
    footer = report.text[report.text.index("<footer") :]
    assert re.search(r"dirconn \d", footer)
    assert re.search(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", footer)


def test_long_tables_are_collapsed() -> None:
    from dirconn.reports.assemble import _table_view

    rows = [{"a": i, "b": 0.123456789} for i in range(51)]
    view = _table_view(rows)
    assert view is not None and view.collapsed and view.n_rows == 51
    assert view.rows[0] == ["0", "0.1235"]
    short = _table_view(rows[:50])
    assert short is not None and not short.collapsed
    assert _table_view([]) is None and _table_view(None) is None
    mixed = _table_view([{"a": 1}, {"b": [1, 2], "a": None}])
    assert mixed is not None and mixed.columns == ["a", "b"]
    assert mixed.rows[1] == ["—", "1, 2"]


def test_svg_prefixing() -> None:
    from dirconn.reports.assemble import _prefix_ids

    svg = (
        '<svg><defs><clipPath id="p1"><rect/></clipPath><path id="m1"/></defs>'
        '<g clip-path="url(#p1)"><use xlink:href="#m1"/><use href="#m1"/></g></svg>'
    )
    out = _prefix_ids(svg, "r7-")
    assert 'id="r7-p1"' in out and 'id="r7-m1"' in out
    assert "url(#r7-p1)" in out
    assert out.count('href="#r7-m1"') == 2
    twice = _prefix_ids('<svg><image id="i1"/><image id="i1"/></svg>', "r1-")
    assert twice.count('id="r1-i1"') == 1


def test_special_characters_are_escaped(pipeline_run: Any, tmp_path: Path) -> None:
    root = _copy(pipeline_run, tmp_path)
    units = collect_subject(root, "01").units
    unit = units[0]
    new = ["A<1>&", "Ä"]
    old: list[str] = []
    for path in sorted(unit.figures_dir.parent.glob(f"{unit.label}_desc-*_conn.json")):
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        labels = sidecar["NodeLabels"]
        old = labels[:2]
        sidecar["NodeLabels"] = [*new, *labels[2:]]
        path.write_text(json.dumps(sidecar), encoding="utf-8")
    assert len(old) == 2
    for path in unit.figures_dir.parent.glob(f"{unit.label}_desc-*_edges.tsv"):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        header = lines[0].split("\t")
        cols = [header.index("source"), header.index("target")]
        out = [lines[0]]
        for line in lines[1:]:
            cells = line.split("\t")
            for c in cols:
                if cells[c] in old:
                    cells[c] = new[old.index(cells[c])]
            out.append("\t".join(cells))
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    path = build_participant_report(root, "01")
    text = path.read_text(encoding="utf-8")
    assert "A&lt;1&gt;&amp;" in text
    assert "A<1>&" not in text
    blocks = SVG_BLOCK.findall(text)
    assert blocks
    for block in blocks:
        ET.fromstring(block)


def test_build_participant_report_survives_boilerplate_failure(
    built_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_built(built_root, tmp_path)
    # the reportlets of the copy are already on disk; only the boilerplate step is under test
    monkeypatch.setattr(assemble, "build_reportlets", lambda *args, **kwargs: [])

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no references today")

    monkeypatch.setattr(assemble, "make_boilerplate", boom)
    path = build_participant_report(root, "02")
    text = path.read_text(encoding="utf-8")
    methods = _section(text, "methods")
    assert "Boilerplate could not be made: RuntimeError: no references today" in methods
    assert "no references today" in _section(text, "errors")
    _parse(text)


def test_missing_reportlets_are_listed_in_errors(built_root: Path, tmp_path: Path) -> None:
    root = _copy_built(built_root, tmp_path)
    unit = collect_subject(root, "01").units[0]
    victim = unit.figures_dir / f"{unit.label}_desc-gc_tests"
    victim.with_suffix(".json").unlink()
    path = assemble_participant(root, "01")
    errors = _section(path.read_text(encoding="utf-8"), "errors")
    assert "missing reportlet" in errors and "gc/tests" in errors and "not produced" in errors
    assert f'id="{unit.label}-gc-tests"' not in path.read_text(encoding="utf-8")


def test_assemble_without_boilerplate_still_writes(built_root: Path, tmp_path: Path) -> None:
    root = _copy_built(built_root, tmp_path)
    path = assemble_participant(root, "02", boilerplate=None)
    _parse(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")
    assert "dirconn report: sub-02" in text


def test_matching_rules() -> None:
    from dirconn.reports.assemble import _matches
    from dirconn.reports.reportlet import Reportlet

    def make(desc: str, name: str) -> Reportlet:
        return Reportlet(name=name, section="x", desc=desc, title="t", caption="c")

    wild = {"desc": "*", "name": "matrix-*"}
    assert _matches(wild, make("gc", "matrix-alpha"))
    assert not _matches(wild, make("gc", "spectra"))
    assert not _matches({"desc": "*", "name": "matrix"}, make("trgc", "matrix"))
    assert _matches({"desc": "trgc", "name": "matrix"}, make("trgc", "matrix"))
    assert not _matches({"desc": "*", "name": "*"}, make("prep", "nodes"))
    assert fnmatchcase("a", "*")

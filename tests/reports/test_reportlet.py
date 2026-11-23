"""Tests of the Reportlet record and its JSON/SVG files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dirconn.reports import Reportlet, read_reportlets, worst_status, write_reportlet

LABEL = "sub-01_task-rest"


def _full(**kw: Any) -> Reportlet:
    base: dict[str, Any] = {
        "name": "matrix-alpha",
        "section": "results",
        "desc": "pdc",
        "title": "PDC, alpha",
        "caption": "Partial directed coherence.",
        "status": "warn",
        "svg": "<svg>a</svg>",
        "table": [{"source": "A", "target": "B", "value": 0.5, "sig": True, "p": None}],
        "data": {"order": 3, "bands": ["alpha", "beta"], "nested": {"x": 1.5}},
    }
    base.update(kw)
    return Reportlet(**base)


def test_round_trip_keeps_every_field(tmp_path: Path) -> None:
    r = _full()
    write_reportlet(tmp_path, LABEL, r)
    assert read_reportlets(tmp_path, LABEL) == [r]


def test_round_trip_without_optional_fields(tmp_path: Path) -> None:
    r = _full(svg=None, table=None, data=None, status="info")
    write_reportlet(tmp_path, LABEL, r)
    assert read_reportlets(tmp_path, LABEL) == [r]


def test_svg_is_a_separate_file(tmp_path: Path) -> None:
    paths = write_reportlet(tmp_path, LABEL, _full())
    names = sorted(p.name for p in paths)
    assert names == [f"{LABEL}_desc-pdc_matrix-alpha.json", f"{LABEL}_desc-pdc_matrix-alpha.svg"]
    payload = json.loads((tmp_path / names[0]).read_text(encoding="utf-8"))
    assert "svg" not in payload
    assert (tmp_path / names[1]).read_text(encoding="utf-8") == "<svg>a</svg>"


def test_stale_svg_is_removed_when_rewritten_without_svg(tmp_path: Path) -> None:
    write_reportlet(tmp_path, LABEL, _full())
    paths = write_reportlet(tmp_path, LABEL, _full(svg=None))
    assert [p.suffix for p in paths] == [".json"]
    assert not (tmp_path / f"{LABEL}_desc-pdc_matrix-alpha.svg").exists()
    assert read_reportlets(tmp_path, LABEL)[0].svg is None


def test_no_temp_files_left(tmp_path: Path) -> None:
    write_reportlet(tmp_path, LABEL, _full())
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"{LABEL}_desc-pdc_matrix-alpha.json",
        f"{LABEL}_desc-pdc_matrix-alpha.svg",
    ]


def test_diagnostics_json_is_not_a_reportlet(tmp_path: Path) -> None:
    (tmp_path / f"{LABEL}_diagnostics.json").write_text('{"order": 3}', encoding="utf-8")
    (tmp_path / f"{LABEL}_desc-x_y.json").write_text('{"no": "section"}', encoding="utf-8")
    write_reportlet(tmp_path, LABEL, _full())
    assert [r.name for r in read_reportlets(tmp_path, LABEL)] == ["matrix-alpha"]


def test_label_prefix_of_another_label_is_not_read(tmp_path: Path) -> None:
    write_reportlet(tmp_path, f"{LABEL}_run-1", _full(name="other"))
    write_reportlet(tmp_path, LABEL, _full())
    assert [r.name for r in read_reportlets(tmp_path, LABEL)] == ["matrix-alpha"]
    assert [r.name for r in read_reportlets(tmp_path, f"{LABEL}_run-1")] == ["other"]


def test_read_is_sorted_by_file_name(tmp_path: Path) -> None:
    for desc, name in [("pdc", "b"), ("dtf", "z"), ("pdc", "a"), ("diag", "psd")]:
        write_reportlet(tmp_path, LABEL, _full(desc=desc, name=name, svg=None))
    got = [(r.desc, r.name) for r in read_reportlets(tmp_path, LABEL)]
    assert got == [("diag", "psd"), ("dtf", "z"), ("pdc", "a"), ("pdc", "b")]


def test_read_from_missing_dir_is_empty(tmp_path: Path) -> None:
    assert read_reportlets(tmp_path / "nope", LABEL) == []


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="status"):
        _full(status="bad")


@pytest.mark.parametrize("bad", ["a_b", "a/b", "a\\b", ""])
@pytest.mark.parametrize("field", ["desc", "name"])
def test_desc_and_name_are_filename_parts(field: str, bad: str) -> None:
    with pytest.raises(ValueError, match=field):
        _full(**{field: bad})


def test_worst_status_orders_fail_over_warn() -> None:
    assert worst_status(["pass", "warn", "info"]) == "warn"
    assert worst_status(["pass", "fail", "warn"]) == "fail"
    assert worst_status(["info", "pass"]) == "pass"
    assert worst_status([]) == "info"

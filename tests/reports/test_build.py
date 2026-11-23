"""Tests for ``dirconn.reports.build``."""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dirconn.reports import build
from dirconn.reports.build import (
    MAX_PAIRS,
    MAX_SIGNIFICANT_ROWS,
    SECTIONS,
    build_reportlets,
    statistics_table,
    subject_reportlets,
    unit_reportlets,
)
from dirconn.reports.collect import collect_subject
from dirconn.reports.reportlet import Reportlet, read_reportlets, worst_status

Keyed = dict[str, Reportlet]

SMALL_CONFIG = """\
[prepare]
resample_hz = "null"

[estimation]
methods = ["gc"]
order = 3

[statistics]
n_surrogates = 19
"""

NO_CONTROLS_CONFIG = (
    SMALL_CONFIG
    + """
[controls]
time_reversal = false
spectral_range_check = false
"""
)


def _keyed(reportlets: list[Reportlet]) -> Keyed:
    keyed = {f"{r.desc}/{r.name}": r for r in reportlets}
    assert len(keyed) == len(reportlets), "duplicate reportlets"
    return keyed


def _slug(band: str) -> str:
    return band.replace("_", "-")


@pytest.fixture(scope="module")
def built_lists(pipeline_run: Any) -> dict[str, list[Reportlet]]:
    """Unit reportlets of every unit of the ``pipeline_run`` fixture, by unit label."""
    out: dict[str, list[Reportlet]] = {}
    for subject in ("01", "02"):
        outputs = collect_subject(pipeline_run.root, subject)
        for unit in outputs.units:
            out[unit.label] = unit_reportlets(outputs, unit)
    return out


@pytest.fixture(scope="module")
def built(built_lists: dict[str, list[Reportlet]]) -> dict[str, Keyed]:
    """The same reportlets keyed by ``desc/name``."""
    return {label: _keyed(reports) for label, reports in built_lists.items()}


def _subject_reports(
    built_lists: dict[str, list[Reportlet]], outputs: Any
) -> dict[str, list[Reportlet]]:
    return {u.label: built_lists[u.label] for u in outputs.units}


@pytest.fixture(scope="module")
def epoched(pipeline_run_epoched: Any) -> Keyed:
    outputs = collect_subject(pipeline_run_epoched.root, "01")
    return _keyed(unit_reportlets(outputs, outputs.units[0]))


def test_every_unit_gets_the_core_reportlets(pipeline_run: Any, built: dict[str, Keyed]) -> None:
    assert len(built) == 5
    bands = list(pipeline_run.cfg.estimation.bands)
    assert bands
    for label, keyed in built.items():
        wanted = [
            "prep/nodes",
            "prep/record",
            "diag/assumptions",
            "diag/order",
            "diag/psd",
            "gc/matrix-broadband",
            "pdc/spectra",
            "adtf/tf",
            "trgc/matrix",
            "trgc/agreement",
            "gc/tests",
            "pdc/tests",
            "adtf/tests",
            *(f"pdc/matrix-{_slug(b)}" for b in bands),
        ]
        for key in wanted:
            assert key in keyed, f"{label} lacks {key}"
        assert not [k for k in keyed if k.startswith("adtf/matrix-")]
        assert keyed["diag/order"].svg
        ET.fromstring(keyed["gc/matrix-broadband"].svg or "")
        assert all(r.section in SECTIONS for r in keyed.values())
        assert all(r.caption and r.title for r in keyed.values())
        assert not [k for k, r in keyed.items() if r.status == "fail"]


def test_inputs_and_diagnostics_reportlets(built: dict[str, Keyed]) -> None:
    keyed = built["sub-01_task-move"]
    nodes = keyed["prep/nodes"]
    assert nodes.table is not None
    assert [row["kind"] for row in nodes.table] == ["cortical", "cortical", "cortical", "direct"]
    assert nodes.table[3]["name"] == "STN-L"
    record = keyed["prep/record"]
    assert record.status == "pass"
    assert record.data is not None and record.data["valid_fraction"] == 1.0
    assert record.table is not None and len(record.table) == 4
    order = keyed["diag/order"]
    assert "bic" in order.caption and "largest candidate order" in order.caption
    assert order.status == "warn"  # the fixture selects the largest candidate (order_max = 6)
    assumptions = keyed["diag/assumptions"]
    assert assumptions.table is not None
    assert {row["method"] for row in assumptions.table} >= {"gc", "pdc", "adtf", "trgc"}
    assert assumptions.status == "warn"
    assert keyed["diag/psd"].status == "pass"
    assert "uncalibrated" in keyed["diag/psd"].caption


def test_statuses_follow_the_rules(built: dict[str, Keyed]) -> None:
    keyed = built["sub-01_task-move"]
    for key in ("prep/nodes", "gc/matrix-broadband", "gc/tests", "trgc/matrix", "pdc/spectra"):
        assert keyed[key].status == "info"
    assert keyed["trgc/agreement"].status in ("pass", "warn")
    agreement = keyed["trgc/agreement"].table
    assert agreement is not None and len(agreement) == 6
    assert set(agreement[0]) == {"source", "target", "trgc_net", "gc_net", "agree"}
    assert (keyed["trgc/agreement"].status == "warn") == any(
        row["agree"] is False for row in agreement
    )


def test_cross_kind_pairs_get_a_net_direction_figure(built: dict[str, Keyed]) -> None:
    keyed = built["sub-01_task-move"]
    assert keyed["gc/net-broadband"].svg
    assert keyed["pdc/net-theta"].svg


def test_fixed_order_is_described(epoched: Keyed) -> None:
    order = epoched["diag/order"]
    assert order.svg is None
    assert "fixed at 2" in order.caption
    assert order.status == "info"


def test_spectrum_method_selects_at_most_max_pairs(
    pipeline_run: Any, built: dict[str, Keyed]
) -> None:
    unit = collect_subject(pipeline_run.root, "01").units[0]
    spectra = built[unit.label]["pdc/spectra"]
    assert spectra.svg
    assert spectra.data is not None
    pairs = [tuple(p) for p in spectra.data["pairs"]]
    assert 0 < len(pairs) <= MAX_PAIRS
    assert len(set(pairs)) == len(pairs) and all(s != t for s, t in pairs)
    edges = unit.methods["pdc"].edge_table()
    assert edges is not None
    if not edges["significant"].astype(str).str.lower().eq("true").any():
        top = edges.groupby(["source", "target"])["value"].max().sort_values(ascending=False)
        assert set(pairs) == set(top.index[:MAX_PAIRS])
        assert pairs[0] == tuple(top.index[0])


def test_untested_time_resolved_result_is_described_not_missing(
    pipeline_run: Any, built: dict[str, Keyed], built_lists: dict[str, list[Reportlet]]
) -> None:
    keyed = built["sub-01_task-move"]
    tests = keyed["adtf/tests"]
    assert "block shuffling" in tests.caption
    assert tests.status == "info"
    assert tests.table is None
    for name in ("significant", "pvalues", "sig-time"):
        assert f"adtf/{name}" not in keyed
    outputs = collect_subject(pipeline_run.root, "01")
    subject = subject_reportlets(outputs, _subject_reports(built_lists, outputs))
    errors = _keyed(subject)["summary/errors"]
    assert errors.status == "pass"
    assert "adtf" not in json.dumps(errors.table)


def test_tested_result_has_tests_significant_and_pvalues(built: dict[str, Keyed]) -> None:
    keyed = built["sub-01_task-move"]
    tests = keyed["gc/tests"]
    assert tests.table is not None
    assert tests.table[0]["test"] == "surrogate"
    assert tests.table[0]["generator"] == "block_shuffle"
    assert tests.table[0]["block (samples)"] == 1250
    assert tests.table[0]["n_surrogates"] == 19
    assert "fdr_bh" in tests.caption and "0.05" in tests.caption
    significant = keyed["gc/significant"]
    assert significant.table is not None
    assert significant.caption == "No edge is significant after correction."
    assert significant.table == []
    assert keyed["gc/pvalues"].svg
    assert "gc/sig-time" not in keyed


def test_time_resolved_tests_on_epochs(epoched: Keyed) -> None:
    assert epoched["wdtf/sig-time"].svg
    assert epoched["wdtf/pvalues"].svg
    assert "wdtf/tf" in epoched
    assert not [k for k in epoched if k.startswith("wdtf/matrix-")]
    tests = epoched["wdtf/tests"]
    assert tests.table is not None and tests.table[0]["generator"] == "trial_shuffle"
    significant = epoched.get("wdtf/significant")
    assert significant is not None and significant.table is not None
    assert len(significant.table) <= MAX_SIGNIFICANT_ROWS
    for row in significant.table:
        assert "time" in row and row["p_adj"] is not None


def test_significant_rows_are_ordered_and_truncated(
    pipeline_run: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    shutil.copytree(pipeline_run.root.parent, out)
    root = out / "dirconn"
    outputs = collect_subject(root, "01")
    unit = outputs.units[0]
    edges_path = unit.methods["gc"].edges
    assert edges_path is not None
    import pandas as pd

    frame = pd.read_csv(edges_path, sep="\t")
    frame["significant"] = True
    frame["p_adj"] = np.linspace(0.04, 0.01, len(frame))
    frame["p"] = frame["p_adj"]
    frame.to_csv(edges_path, sep="\t", index=False)
    monkeypatch.setattr(build, "MAX_SIGNIFICANT_ROWS", 5)
    keyed = _keyed(unit_reportlets(collect_subject(root, "01"), unit))
    rows = keyed["gc/significant"].table
    assert rows is not None and len(rows) == 5
    assert [r["p_adj"] for r in rows] == sorted(r["p_adj"] for r in rows)
    assert f"5 of {len(frame)} significant rows shown" in keyed["gc/significant"].caption


def test_builtin_test_row() -> None:
    rows = statistics_table(
        {
            "Statistics": [
                {
                    "Band": "broadband",
                    "Test": "builtin",
                    "Generator": {"name": "builtin", "source": "PCMCI MCI p-values"},
                    "NSurrogates": None,
                    "Seed": None,
                    "PValueShape": [3, 3, 1, 1],
                }
            ]
        }
    )
    assert len(rows) == 1
    row = rows[0]
    assert "PCMCI MCI p-values" in row["generator"]
    assert row["n_surrogates"] == "—" and row["seed"] == "—"
    assert row["block (samples)"] == "—"
    assert row["p_shape"] == "3×3×1×1"
    assert row["band"] == "broadband" and row["test"] == "builtin"


def test_statistics_table_reads_block_size() -> None:
    rows = statistics_table(
        {
            "Statistics": [
                {
                    "Band": "theta",
                    "Test": "surrogate",
                    "Generator": {"name": "block_shuffle", "params": {"block_size": 250}},
                    "NSurrogates": 19,
                    "Seed": 7,
                    "PValueShape": [4, 4],
                }
            ]
        }
    )
    assert rows[0]["generator"] == "block_shuffle" and rows[0]["block (samples)"] == 250
    assert rows[0]["n_surrogates"] == 19 and rows[0]["seed"] == 7
    assert statistics_table({}) == []


def test_missing_outputs_are_reported_not_raised(
    make_pipeline_run: Callable[..., Any],
) -> None:
    run = make_pipeline_run("no-controls", NO_CONTROLS_CONFIG, max_units=1)
    subject = run.units[0].subject
    outputs = collect_subject(run.root, subject)
    unit = outputs.units[0]
    keyed = _keyed(unit_reportlets(outputs, unit))
    assert not [k for k in keyed if k.startswith("trgc/")] and "diag/psd" not in keyed
    errors = _keyed(subject_reportlets(outputs))["summary/errors"]
    assert errors.status == "info"
    assert errors.table is not None
    reasons = {row["reason"] for row in errors.table}
    assert "time_reversal is off" in reasons
    assert "spectral_range_check is off" in reasons
    assert {row["reportlet"] for row in errors.table} >= {"trgc/matrix", "diag/psd"}


def test_failed_task_makes_errors_fail(pipeline_run_with_failure: Any) -> None:
    run = pipeline_run_with_failure
    outputs = collect_subject(run.root, "01")
    assert outputs.failures
    keyed = _keyed(subject_reportlets(outputs))
    errors = keyed["summary/errors"]
    assert errors.status == "fail"
    assert errors.table is not None
    failed = [row for row in errors.table if row["kind"] == "task failed"]
    assert failed and all(row["label"].endswith(":gc") for row in failed)
    assert all(row["crash_file"].endswith(".txt") for row in failed)
    assert all(not Path(row["crash_file"]).is_absolute() for row in failed)
    missing = [row for row in errors.table if row["reportlet"].startswith("gc/")]
    assert missing and all("task failed" in row["reason"] for row in missing)
    assert keyed["summary/overview"].status == "fail"


def test_summary_overview_lists_every_unit(
    pipeline_run: Any, built_lists: dict[str, list[Reportlet]]
) -> None:
    outputs = collect_subject(pipeline_run.root, "01")
    unit_reports = _subject_reports(built_lists, outputs)
    keyed = _keyed(subject_reportlets(outputs, unit_reports))
    overview = keyed["summary/overview"]
    assert overview.section == "summary"
    assert overview.table is not None and len(overview.table) == 3
    row = overview.table[0]
    assert row["unit"] == outputs.units[0].label
    assert row["methods"] == "adtf, gc, pdc"
    assert row["bands"] == "beta_high, beta_low, theta"
    assert row["image_tag"] == "not recorded" and row["image_revision"] == "not recorded"
    assert row["input_files"]
    assert overview.data is not None
    full = overview.data["sha256"][row["unit"]]
    assert len(full) == 64 and row["sha256"] == full[:12] + "\u2026"
    assert row["started"] and row["finished"] and row["dirconn"]
    everything = [r for reports in unit_reports.values() for r in reports]
    expected = worst_status([r.status for r in everything] + [keyed["summary/errors"].status])
    assert overview.status == expected
    assert keyed["summary/errors"].status == "pass"


def test_summary_shows_image_tag_and_revision(
    make_pipeline_run: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIRCONN_IMAGE_TAG", "tag-under-test")
    monkeypatch.setenv("DIRCONN_BUILD_REVISION", "rev-under-test")
    run = make_pipeline_run("image-tag", SMALL_CONFIG, max_units=1)
    outputs = collect_subject(run.root, run.units[0].subject)
    overview = _keyed(subject_reportlets(outputs))["summary/overview"]
    assert overview.table is not None
    assert overview.table[0]["image_tag"] == "tag-under-test"
    assert overview.table[0]["image_revision"] == "rev-under-test"


def test_record_with_few_valid_samples_warns(pipeline_run: Any, tmp_path: Path) -> None:
    out = tmp_path / "out"
    shutil.copytree(pipeline_run.root.parent, out)
    root = out / "dirconn"
    outputs = collect_subject(root, "01")
    unit = outputs.units[0]
    path = unit.figures_dir / f"{unit.label}_diagnostics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["valid_fraction"] = 0.3
    path.write_text(json.dumps(payload), encoding="utf-8")
    again = collect_subject(root, "01")
    keyed = _keyed(unit_reportlets(again, again.units[0]))
    assert keyed["prep/record"].status == "warn"


def test_a_broken_figure_does_not_stop_the_report(
    pipeline_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("boom\nsecond line")

    monkeypatch.setattr(build, "band_matrix_svg", broken)
    outputs = collect_subject(pipeline_run.root, "02")
    unit_lists = {u.label: unit_reportlets(outputs, u) for u in outputs.units}
    unit_results = {label: _keyed(reports) for label, reports in unit_lists.items()}
    first = unit_results[outputs.units[0].label]
    placeholder = first["gc/matrix-broadband"]
    assert placeholder.status == "fail" and placeholder.svg is None
    assert placeholder.caption == "could not be built: RuntimeError: boom"
    assert placeholder.section == "connectivity"
    assert "pdc/spectra" in first and first["pdc/spectra"].svg
    subject = _keyed(subject_reportlets(outputs, unit_lists))
    errors = subject["summary/errors"]
    assert errors.status == "fail" and errors.table is not None
    built_failures = [r for r in errors.table if r["kind"] == "build failed"]
    assert {r["reportlet"] for r in built_failures} >= {"gc/matrix-broadband"}
    assert all("RuntimeError: boom" in r["reason"] for r in built_failures)
    assert subject["summary/overview"].status == "fail"


def test_reportlet_svgs_are_small(reportlets_root: Path) -> None:
    sizes = {p: p.stat().st_size for p in reportlets_root.rglob("*.svg")}
    assert sizes
    too_big = {p.name: size for p, size in sizes.items() if size >= 1_000_000}
    assert not too_big


def test_build_reportlets_writes_files_and_rebuild_overwrites(
    pipeline_run: Any, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    shutil.copytree(pipeline_run.root.parent, out)
    root = out / "dirconn"
    outputs = collect_subject(root, "02")
    unit = outputs.units[0]
    stale = unit.figures_dir / f"{unit.label}_desc-gc_stale.json"
    stale.write_text("{}", encoding="utf-8")
    diagnostics = unit.figures_dir / f"{unit.label}_diagnostics.json"
    before = diagnostics.read_bytes()
    first = build_reportlets(root, "02")
    assert first and all(p.is_file() for p in first)
    assert not stale.exists()
    assert diagnostics.read_bytes() == before
    subject_dir = root / "sub-02" / "figures"
    assert {r.name for r in read_reportlets(subject_dir, "sub-02")} == {"overview", "errors"}
    for u in outputs.units:
        labelled = read_reportlets(u.figures_dir, u.label)
        assert any(r.desc == "gc" and r.name == "matrix-broadband" for r in labelled)
        assert all(r.desc != "summary" for r in labelled)
    files_first = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    second = build_reportlets(root, "02")
    files_second = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert files_first == files_second
    assert sorted(first) == sorted(second)
    assert build_reportlets(root, "99") == []


def test_psi_has_no_spectra_reportlet(make_pipeline_run: Callable[..., Any]) -> None:
    config = SMALL_CONFIG.replace('["gc"]', '["pdc", "psi"]')
    run = make_pipeline_run("psi-spectra", config, max_units=1)
    outputs = collect_subject(run.root, run.units[0].subject)
    keyed = _keyed(unit_reportlets(outputs, outputs.units[0]))
    assert "pdc/spectra" in keyed
    assert "psi/spectra" not in keyed
    assert any(k.startswith("psi/matrix-") for k in keyed)


def test_trgc_matrix_uses_a_diverging_map(
    pipeline_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []
    original = build.band_matrix_svg

    def spy(*args: Any, **kwargs: Any) -> str:
        seen.append({"title": kwargs.get("title"), "diverging": kwargs.get("diverging", False)})
        return original(*args, **kwargs)

    monkeypatch.setattr(build, "band_matrix_svg", spy)
    outputs = collect_subject(pipeline_run.root, "02")
    unit_reportlets(outputs, outputs.units[0])
    trgc = [s for s in seen if s["title"] == "TRGC, broadband"]
    assert trgc and all(s["diverging"] for s in trgc)
    assert not [s for s in seen if s["title"] != "TRGC, broadband" and s["diverging"]]


def test_no_significant_row_has_its_own_caption() -> None:
    import pandas as pd

    edges = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "A"],
            "band": ["theta", "theta"],
            "value": [0.5, 0.2],
            "p": [0.5, 0.6],
            "p_adj": [0.5, 0.6],
            "significant": [False, False],
        }
    )
    body = build._significant_body(None, edges)  # type: ignore[arg-type]
    assert body is not None
    assert body.caption == "No edge is significant after correction."
    assert body.table == []

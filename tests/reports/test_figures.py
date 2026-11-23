"""Tests for ``dirconn.reports.figures``."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any

import matplotlib
import pandas as pd
import pytest

from dirconn.reports import figures
from dirconn.reports.collect import collect_subject


def _assert_svg(text: str) -> None:
    assert text.startswith("<svg")
    assert ET.fromstring(text).tag.endswith("svg")


@pytest.fixture(scope="module")
def unit(pipeline_run: Any) -> Any:
    return collect_subject(pipeline_run.root, "01").units[0]


@pytest.fixture(scope="module")
def epoched_wdtf(pipeline_run_epoched: Any) -> Any:
    return collect_subject(pipeline_run_epoched.root, "01").units[0].methods["wdtf"]


def _edges(**overrides: Any) -> pd.DataFrame:
    rows = [
        ("A", "B", "theta", 0.5, 0.01, 0.02, True),
        ("B", "A", "theta", 0.2, 0.40, 0.40, False),
    ]
    frame = pd.DataFrame(
        rows, columns=["source", "target", "band", "value", "p", "p_adj", "significant"]
    )
    return frame.assign(**overrides) if overrides else frame


def test_band_matrix_svg_parses(unit: Any) -> None:
    pdc = unit.methods["pdc"]
    nodes = list(pdc.sidecar["NodeLabels"])
    _assert_svg(figures.band_matrix_svg(pdc.edge_table(), "theta", nodes, title="pdc"))


def test_band_matrix_marks_significant_cells() -> None:
    svg = figures.band_matrix_svg(_edges(), "theta", ["A", "B"], title="t")
    stars = [e for e in ET.fromstring(svg).iter() if e.tag.endswith("text") and e.text == "*"]
    assert len(stars) == 1


def test_band_matrix_accepts_string_significance() -> None:
    edges = _edges(significant=["True", "n/a"])
    svg = figures.band_matrix_svg(edges, "theta", ["A", "B"], title="t")
    assert sum(1 for e in ET.fromstring(svg).iter() if e.text == "*") == 1


def test_band_matrix_escapes_names() -> None:
    edges = _edges().assign(source=["A<1>&", "Ä"], target=["Ä", "A<1>&"])
    svg = figures.band_matrix_svg(edges, "theta", ["A<1>&", "Ä"], title="a<b & c")
    _assert_svg(svg)


def test_band_matrix_refuses_time_resolved_edges(unit: Any) -> None:
    adtf = unit.methods["adtf"].edge_table()
    with pytest.raises(ValueError, match="time"):
        figures.band_matrix_svg(adtf, "theta", ["a", "b"], title="t")
    with pytest.raises(ValueError, match="time"):
        figures.net_direction_svg(adtf, {"a": "x", "b": "y"}, "theta")


def test_net_direction_svg_parses(unit: Any) -> None:
    gc = unit.methods["gc"]
    kinds = dict(zip(gc.sidecar["NodeLabels"], gc.sidecar["NodeKinds"], strict=True))
    _assert_svg(figures.net_direction_svg(gc.edge_table(), kinds, "broadband"))


def test_net_direction_needs_two_kinds(unit: Any) -> None:
    gc = unit.methods["gc"]
    kinds = {name: "cortical" for name in gc.sidecar["NodeLabels"]}
    assert figures.net_direction_svg(gc.edge_table(), kinds, "broadband") == ""


def test_order_curve_svg_parses(unit: Any) -> None:
    _assert_svg(figures.order_curve_svg(unit.diagnostics["order_selection"]))


def test_order_curve_marks_the_selected_order() -> None:
    selection = {
        "criterion": "bic",
        "orders": [1, 2, 3, 4],
        "aic": [4.0, 3.0, 2.5, 2.6],
        "bic": [4.0, 3.0, 3.2, 3.5],
        "hqc": [4.0, 3.0, 2.9, 3.0],
        "cv": None,
    }
    svg = figures.order_curve_svg(selection)
    assert "selected order 2 (bic)" in svg


def test_psd_svg_parses(unit: Any) -> None:
    bands = unit.methods["pdc"].sidecar["Bands"]
    _assert_svg(figures.psd_svg(unit.diagnostics["spectral"], bands))


def test_directed_spectra_svg_parses(unit: Any) -> None:
    conn = unit.methods["pdc"].connectivity()
    pairs = [(conn.source[0], conn.target[1]), (conn.source[1], conn.target[3])]
    _assert_svg(figures.directed_spectra_svg(conn, pairs))


def test_time_frequency_svg_parses(unit: Any) -> None:
    conn = unit.methods["adtf"].connectivity()
    _assert_svg(figures.time_frequency_svg(conn, [(conn.source[0], conn.target[1])]))


def test_tested_time_resolved_edges(epoched_wdtf: Any) -> None:
    edges = epoched_wdtf.edge_table()
    _assert_svg(figures.pvalue_hist_svg(edges))
    _assert_svg(figures.significant_over_time_svg(edges))
    conn = epoched_wdtf.connectivity()
    _assert_svg(figures.time_frequency_svg(conn, [(conn.source[0], conn.target[1])]))


def test_pvalue_hist_svg_parses(unit: Any) -> None:
    _assert_svg(figures.pvalue_hist_svg(unit.methods["gc"].edge_table()))


def test_pvalue_hist_is_empty_for_untested(unit: Any) -> None:
    assert figures.pvalue_hist_svg(unit.methods["adtf"].edge_table()) == ""
    assert figures.pvalue_hist_svg(_edges().iloc[0:0]) == ""


def test_significant_over_time_is_empty_without_time_or_p(unit: Any) -> None:
    assert figures.significant_over_time_svg(unit.methods["gc"].edge_table()) == ""
    assert figures.significant_over_time_svg(unit.methods["adtf"].edge_table()) == ""


def test_svg_is_deterministic(unit: Any) -> None:
    salt = matplotlib.rcParams["svg.hashsalt"]
    fonttype = matplotlib.rcParams["svg.fonttype"]
    makers: list[Callable[[], str]] = [
        lambda: figures.band_matrix_svg(_edges(), "theta", ["A", "B"], title="t"),
        lambda: figures.order_curve_svg(unit.diagnostics["order_selection"]),
        lambda: figures.psd_svg(unit.diagnostics["spectral"], unit.methods["pdc"].sidecar["Bands"]),
    ]
    for make in makers:
        assert make() == make()
    assert matplotlib.rcParams["svg.hashsalt"] == salt
    assert matplotlib.rcParams["svg.fonttype"] == fonttype

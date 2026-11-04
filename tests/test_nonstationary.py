"""The non-stationary scenarios, the sweep unit, the results file and the report."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dirconn import Connectivity, Provenance
from dirconn import nonstationary as ns
from dirconn.evaluation import switch_times
from dirconn.scenario import load_scenario
from dirconn.simulation import network_var_schedule
from dirconn.spectral import spectral_radius

ROOT = Path(__file__).resolve().parents[1]
R1, R2, R3 = "transversetemporal-rh", "cuneus-rh", "transversetemporal-lh"

# scenario -> (switch times, the pairs whose coupling switches)
EXPECTED = {
    "switch_on": ([30.0], {(R1, R2)}),
    "switch_off": ([30.0], {(R1, R2)}),
    "direction_flip": ([30.0], {(R1, R2), (R2, R1)}),
    "burst": ([25.0, 35.0], {(R1, R2)}),
}


def load_script():
    spec = importlib.util.spec_from_file_location(
        "run_nonstationary", ROOT / "scripts" / "run_nonstationary.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# The scenario files
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(EXPECTED))
class TestScenarios:
    def test_the_switches_and_the_switching_pairs_are_the_intended_ones(self, name):
        spec = load_scenario(name)
        switches, pairs = EXPECTED[name]
        duration = spec.n_times / spec.sfreq
        assert duration == 60.0 and spec.sfreq == 100.0
        assert switch_times(spec.network, duration=duration) == switches
        assert set(ns.switching_pairs(spec, duration)) == pairs

    def test_b_to_c_is_a_constant_control_edge(self, name):
        spec = load_scenario(name)
        control = [e for e in spec.network.edges if (e.source, e.target) == (R2, R3)]
        assert len(control) == 1 and control[0].active is None

    def test_every_stretch_is_stable(self, name):
        spec = load_scenario(name)
        for _, model in network_var_schedule(spec.network, spec.sfreq, spec.n_times, spec.dynamics):
            assert spectral_radius(np.asarray(model.lagged)) < 0.98

    def test_the_truth_always_has_an_edge_and_a_non_edge(self, name):
        spec = load_scenario(name)
        truth = spec.network.truth(times=np.arange(0.0, 60.0, 0.5))
        edges = np.nan_to_num(truth.values[:, :, 0, :], nan=0.0).sum(axis=(0, 1))
        assert (edges >= 1).all() and (edges <= 5).all()


def test_the_guard_is_a_third_of_the_shortest_stretch():
    guards = {name: ns.scenario_guard(load_scenario(name)) for name in EXPECTED}
    assert guards["switch_on"] == pytest.approx(10.0)
    assert guards["switch_off"] == pytest.approx(10.0)
    assert guards["direction_flip"] == pytest.approx(10.0)
    assert guards["burst"] == pytest.approx(10.0 / 3.0)  # the burst itself is the shortest stretch


def test_the_grid_the_labels_and_the_seeds():
    assert ns.SCENARIOS == tuple(EXPECTED)
    assert ns.FORGETTING[:4] == (0.95, 0.98, 0.99, 0.995)  # the grid the user confirmed
    assert ns.FORGETTING == tuple(sorted(ns.FORGETTING))
    assert ns.method_labels() == [
        "adtf-dual",
        "adtf-dekf",
        "tpdc",
        "wdtf-4s",
        "wdtf-8s",
    ]
    assert [ns.has_forgetting(m) for m in ns.method_labels()] == [True, True, True, False, False]
    seeds = {
        ns.dataset_seed(sc, snr, k) for sc in ns.SCENARIOS for snr in ns.SNR_DB for k in range(10)
    }
    assert len(seeds) == 4 * 3 * 10  # all distinct
    assert ns.dataset_seed("burst", 0.0, 3) == ns.dataset_seed("burst", 0.0, 3)


def test_the_datasets_of_a_run():
    full = ns.datasets()
    assert len(full) == 120 and full[0] == ("switch_on", 20.0, 0) and full[-1] == ("burst", 0.0, 9)
    quick = ns.datasets(seeds=2, snr_db=(10.0,))
    assert len(quick) == 8 and {t[1] for t in quick} == {10.0}


def test_estimators_are_built_from_their_labels():
    assert ns.make_estimator("adtf-dekf", 0.999).update == "dekf"
    assert ns.make_estimator("adtf-dual", 0.99).forgetting == 0.99
    assert ns.make_estimator("tpdc", 0.999).info.name == "tpdc"
    assert ns.make_estimator("wdtf-8s", None).window == 8.0
    for label in ("adtf-dual", "adtf-dekf", "tpdc"):
        assert ns.make_estimator(label, 0.99).time_step == ns.TIME_STEP_S
    with pytest.raises(ValueError, match="Unknown method label"):
        ns.make_estimator("gc", None)


# --------------------------------------------------------------------------
# Scoring one result (synthetic, no simulation)
# --------------------------------------------------------------------------


def synthetic_result(spec, *, lag_s=0.0, times=None, method="adtf"):
    """A result that follows the truth exactly, delayed by ``lag_s`` seconds."""
    times = np.arange(2.0, 60.0, 0.1) if times is None else times
    freqs = np.array([2.0, 10.0, 20.0, 40.0])
    truth = spec.network.truth(freqs=freqs, times=times - lag_s)
    values = np.nan_to_num(truth.values, nan=0.0)
    return Connectivity.from_array(
        values,
        source=list(spec.network.names),
        freqs=freqs,
        times=times,
        provenance=Provenance(method=method, parameters={}),
    )


def test_a_result_that_follows_the_truth_scores_perfectly_with_one_course_per_pair():
    spec = load_scenario("direction_flip")
    row = ns.evaluate_result(synthetic_result(spec), spec)
    assert row["auroc_stable"] == pytest.approx(1.0)
    assert row["direction_stable"] == 1.0
    assert row["guard_s"] == pytest.approx(10.0)
    assert row["memory_guard_s"] is None and row["auroc_memory"] is None  # no memory known
    assert row["switches"] == [30.0]
    pairs = {(c["source"], c["target"]) for c in row["curves"]}
    assert pairs == {(R1, R2), (R2, R1)}
    axis = row["curve_axis"]
    assert axis["step"] == pytest.approx(ns.CURVE_STEP_S)
    assert all(len(c["scores"]) == axis["n"] for c in row["curves"])


def test_a_lagging_result_costs_auroc_only_without_the_guard():
    spec = load_scenario("switch_on")
    row = ns.evaluate_result(synthetic_result(spec, lag_s=3.0), spec)
    assert row["auroc_stable"] == pytest.approx(1.0)  # the 3 s transition is inside the 10 s guard
    assert row["auroc_all"] < 1.0


def cell_rows(spec, lag_s, seeds=3):
    rows = []
    for k in range(seeds):
        result = synthetic_result(spec, lag_s=lag_s)
        rows.append(
            {
                "kind": "sweep",
                "scenario": spec.name,
                "snr_db": 10.0,
                "seed_index": k,
                "method": "adtf-dual",
                "forgetting": 0.999,
                **ns.evaluate_result(result, spec),
            }
        )
    return rows


def test_the_latency_of_the_mean_course_recovers_a_known_delay():
    spec = load_scenario("switch_on")
    (record,) = ns.latency_of_mean_course(cell_rows(spec, lag_s=3.0))
    assert record["tracked"] is True
    assert record["latency_s"] == pytest.approx(3.0, abs=0.6)  # courses are stored every 0.5 s
    assert (record["source"], record["target"]) == (R1, R2)
    assert record["post_level"] > record["pre_level"]


def test_a_burst_gives_two_latencies_and_a_switch_down_is_measured_too():
    spec = load_scenario("burst")
    records = ns.latency_of_mean_course(cell_rows(spec, lag_s=1.0))
    assert [r["switch"] for r in records] == [25.0, 35.0]
    assert all(r["tracked"] for r in records)
    assert records[0]["post_level"] > records[0]["pre_level"]
    assert records[1]["post_level"] < records[1]["pre_level"]


def test_rows_that_are_not_one_cell_cannot_be_averaged():
    spec = load_scenario("switch_on")
    rows = cell_rows(spec, 0.0, seeds=2)
    rows[1]["forgetting"] = 0.99
    with pytest.raises(ValueError, match="not one cell"):
        ns.aggregate_course(rows)
    with pytest.raises(ValueError, match="empty"):
        ns.aggregate_course([])


def test_the_aggregate_keeps_one_mean_course_per_cell_and_pair():
    spec = load_scenario("direction_flip")
    cells = ns.aggregate_courses(cell_rows(spec, 0.0, seeds=2))
    (cell,) = cells
    assert cell["n"] == 2 and len(cell["courses"]) == 2
    assert {"mean", "source", "target", "switches"} <= set(cell["courses"][0])
    assert len(cell["courses"][0]["mean"]) == cell["curve_axis"]["n"]


# --------------------------------------------------------------------------
# One simulated dataset
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_dataset():
    return ns.evaluate_dataset(
        "switch_on",
        10.0,
        0,
        methods=["adtf-dual", "tpdc", "wdtf-4s"],
        forgetting=(0.99, 0.999999),
        segments=True,
    )


def test_a_dataset_gives_one_row_per_method_and_factor_and_the_segment_rows(small_dataset):
    sweep = [r for r in small_dataset if r["kind"] == "sweep"]
    segments = [r for r in small_dataset if r["kind"] == "segments"]
    assert [(r["method"], r["forgetting"]) for r in sweep] == [
        ("adtf-dual", 0.99),
        ("adtf-dual", 0.999999),
        ("tpdc", 0.99),
        ("tpdc", 0.999999),
        ("wdtf-4s", None),
    ]
    assert len(segments) == 2 * 2 * 1  # two approaches, segments A and B, one switching pair
    for row in sweep:
        assert row["scenario"] == "switch_on" and row["snr_db"] == 10.0 and row["seed_index"] == 0
        assert row["seed"] == ns.dataset_seed("switch_on", 10.0, 0)
        assert 0.0 <= row["auroc_stable"] <= 1.0
        assert row["n_stable"] > 0 and row["n_defined"] == row["n_stable"]
        json.dumps(row)  # everything is JSON-serialisable
    memories = [r["memory_s"] for r in sweep if r["method"] == "adtf-dual"]
    assert memories[0] < memories[1]  # a larger factor remembers longer
    assert sweep[-1]["memory_s"] is None and sweep[-1]["auroc_memory"] is not None


def test_the_factor_with_more_memory_ranks_better_in_this_scenario(small_dataset):
    # measured on 3 seeds at 10 dB: 0.74 for 0.99 and 0.999 for 0.999999
    sweep = {
        (r["method"], r["forgetting"]): r["auroc_stable"]
        for r in small_dataset
        if r["kind"] == "sweep"
    }
    assert sweep[("adtf-dual", 0.999999)] > sweep[("adtf-dual", 0.99)] + 0.1
    assert sweep[("tpdc", 0.999999)] > 0.9 and sweep[("wdtf-4s", None)] > 0.9


def test_the_segment_approaches_are_scored_against_the_truth_at_the_segment_centre(small_dataset):
    segments = [r for r in small_dataset if r["kind"] == "segments"]
    by = {(r["approach"], r["segment"]): r for r in segments}
    assert set(by) == {
        ("adtf_reduce_time", "A"),
        ("adtf_reduce_time", "B"),
        ("wdtf_segments", "A"),
        ("wdtf_segments", "B"),
    }
    pair = f"{R1}->{R2}"
    # the switching edge is off in segment A and on in segment B: the score has to rise
    for approach in ("adtf_reduce_time", "wdtf_segments"):
        assert by[(approach, "B")]["pair_scores"][pair] > by[(approach, "A")]["pair_scores"][pair]
    assert by[("adtf_reduce_time", "A")]["span"] == list(ns.SEGMENTS["A"])
    assert by[("adtf_reduce_time", "A")]["forgetting"] == ns.SEGMENT_FORGETTING
    assert by[("wdtf_segments", "A")]["forgetting"] is None


def test_a_dataset_is_reproducible(small_dataset):
    again = ns.evaluate_dataset(
        "switch_on",
        10.0,
        0,
        methods=["adtf-dual", "tpdc", "wdtf-4s"],
        forgetting=(0.99, 0.999999),
        segments=True,
    )

    def stable(rows):
        # as text, because NaN (the unstable share of a windowed method) is not equal to itself
        return json.dumps(
            [{k: v for k, v in r.items() if k != "fit_seconds"} for r in rows], sort_keys=True
        )

    assert stable(again) == stable(small_dataset)


@pytest.mark.filterwarnings("ignore:.*coefficients per equation:dirconn.exceptions.DirconnWarning")
def test_the_grid_rows_equal_those_of_separate_estimator_fits():
    spec = load_scenario("switch_on")
    series = ns.simulate_roi(spec, snr_db=10.0, seed=ns.dataset_seed("switch_on", 10.0, 0))
    (row,) = ns.evaluate_dataset(
        "switch_on", 10.0, 0, methods=["adtf-dekf"], forgetting=(0.9999,), segments=False
    )
    alone = ns.make_estimator("adtf-dekf", 0.9999).fit(series)
    expected = ns.evaluate_result(alone, spec)
    for key in ("auroc_stable", "direction_stable", "auroc_all", "n_stable", "guard_s"):
        assert row[key] == expected[key]
    assert row["curves"] == expected["curves"]


# --------------------------------------------------------------------------
# The results file and the report
# --------------------------------------------------------------------------


def test_the_results_file_keeps_the_rows_and_only_the_mean_courses(small_dataset, tmp_path):
    path = tmp_path / "results.json"
    ns.save_results(small_dataset, path, meta={"tier": "test", "seeds": 1})
    text = path.read_text(encoding="utf-8")
    assert '"curves"' not in text.split('"courses"')[0]  # the single courses are not kept
    rows, meta = ns.load_results(path)
    assert len(rows) == len(small_dataset)
    assert all("curves" not in r and "curve_axis" not in r for r in rows)
    assert meta["tier"] == "test" and "numpy" in meta["environment"]
    assert meta["forgetting"] == list(ns.FORGETTING)
    assert len(meta["courses"]) == 5  # one per cell of this dataset's sweep


def test_a_file_of_another_kind_is_refused(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"kind": "something else", "rows": [], "meta": {}}))
    with pytest.raises(ValueError, match="not a dirconn-nonstationary"):
        ns.load_results(path)


def test_the_report_has_every_section_and_the_numbers_of_the_rows(small_dataset, tmp_path):
    path = tmp_path / "results.json"
    ns.save_results(small_dataset, path, meta={"seeds": 1})
    rows, meta = ns.load_results(path)
    text = ns.render_report(rows, meta, findings="Hand-written reading.")
    assert text.startswith("# Non-stationary validation")
    for heading in ("## Setup", "## How to read the numbers", "## Overview", "## Scenarios"):
        assert heading in text
    assert "Hand-written reading." in text and "### switch_on" in text
    assert "## One value per segment" in text
    assert "numpy" in text.split("Environment:")[1].splitlines()[0]
    assert "| adtf-dual | 0.999999 |" in text and "| wdtf-4s | – |" in text
    assert "A→B at 30 s" in text
    sweep = [r for r in rows if r["kind"] == "sweep" and r["method"] == "adtf-dual"]
    assert f"{sweep[1]['auroc_stable']:.3f}" in text
    assert text == ns.render_report(rows, meta, findings="Hand-written reading.")  # deterministic


def test_the_report_in_the_repository_is_generated_from_the_results_in_the_repository():
    results = ROOT / "docs" / "nonstationary_results.json"
    report = ROOT / "docs" / "nonstationary_validation.md"
    if not results.exists():
        pytest.skip("the sweep has not been run yet")
    rows, meta = ns.load_results(results)
    findings_file = ROOT / "docs" / "nonstationary_findings.md"
    findings = findings_file.read_text(encoding="utf-8") if findings_file.exists() else ""
    assert report.read_text(encoding="utf-8") == ns.render_report(rows, meta, findings=findings)
    assert meta["seeds"] == ns.N_SEEDS and len(rows) > 3000


def test_the_results_in_the_repository_record_estimate_and_actual_time():
    results = ROOT / "docs" / "nonstationary_results.json"
    if not results.exists():
        pytest.skip("the sweep has not been run yet")
    _, meta = ns.load_results(results)
    gate = meta["gate"]
    assert gate["efficiency"] > 0 and gate["jobs"] >= 1 and gate["n_datasets"] == 120
    assert meta["actual_wall_s"] > 0 and meta["estimate_over_actual"] > 0


# --------------------------------------------------------------------------
# The script
# --------------------------------------------------------------------------


def test_the_script_resumes_from_the_partial_file(tmp_path, monkeypatch):
    script = load_script()
    calls = []

    def fake(task):
        calls.append(task)
        return {"task": list(task), "rows": [{"kind": "sweep", "task": list(task)}], "seconds": 0.1}

    monkeypatch.setattr(script, "run_dataset", fake)
    tasks = [("switch_on", 10.0, 0), ("switch_on", 10.0, 1), ("burst", 10.0, 0)]
    partial = tmp_path / "x.partial.jsonl"
    partial.write_text(json.dumps(fake(tasks[0])) + "\n", encoding="utf-8")
    calls.clear()
    rows = script.run(tasks, 1, partial)
    assert calls == tasks[1:]  # the finished dataset is not run again
    assert [r["task"] for r in rows] == [list(t) for t in tasks]  # in the order of the tasks
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 3


def test_the_gate_estimates_from_a_measured_time_and_efficiency(monkeypatch):
    script = load_script()
    monkeypatch.setattr(script, "run_dataset", lambda task: {"seconds": 10.0})
    measured = script.gate(1, 20)
    assert measured["single_dataset_s"] == 10.0 and measured["efficiency"] == 1.0
    assert measured["estimate_s"] == pytest.approx(200.0)
    assert measured["jobs"] == 1 and measured["n_datasets"] == 20


def test_the_script_rerenders_the_saved_report(small_dataset, tmp_path):
    script = load_script()
    results = tmp_path / "results.json"
    ns.save_results(small_dataset, results, meta={"seeds": 1})
    out = tmp_path / "report.md"
    code = script.main(
        [
            "--from-results",
            "--results",
            str(results),
            "--output",
            str(out),
            "--findings",
            str(tmp_path / "no_findings.md"),
        ]
    )
    assert code == 0
    rows, meta = ns.load_results(results)
    assert out.read_text(encoding="utf-8") == ns.render_report(rows, meta)

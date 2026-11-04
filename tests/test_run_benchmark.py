"""The command line of the benchmark (scripts/run_benchmark.py), with a fake runner."""

import importlib.util
from pathlib import Path

import pytest
import yaml

from dirconn.protocol import expand, load_protocol, read_results, task_seed

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "run_benchmark", ROOT / "scripts" / "run_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()


class Runner:
    def __init__(self, fail_on_seed_index=None):
        self.calls = []
        self.fail_on_seed_index = fail_on_seed_index

    def __call__(self, task):
        self.calls.append(task.task_id)
        if task.seed_index == self.fail_on_seed_index:
            raise RuntimeError("boom")
        return {"gc/bic": {"roi": {"auroc": 0.9}, "sensor": {"auroc": 0.8}}}


class Gate:
    """A stand-in for measure_gate: every task takes `seconds` seconds."""

    def __init__(self, seconds=1.0):
        self.calls = 0
        self.seconds = seconds

    def __call__(self, protocol, *, workers, **kwargs):
        from dirconn.protocol import estimate_hours

        self.calls += 1
        tasks = expand(protocol)
        sizes = {n: self.seconds for n in {t.n_nodes for t in tasks}}
        hours = estimate_hours(
            tasks, sizes, reference_duration_s=120.0, workers=workers, efficiency=0.8
        )
        return {
            "seconds_per_task": {str(n): s for n, s in sizes.items()},
            "reference_duration_s": 120.0,
            "workers": workers,
            "efficiency": 0.8,
            "startup_seconds": 0.0,
            "n_tasks": len(tasks),
            "estimate_hours": hours,
            "measured": [],
        }


@pytest.fixture
def protocol_file(tmp_path):
    data = {
        "version": 1,
        "name": "demo",
        "scenarios": ["chain", "collider"],
        "estimators": ["gc/bic"],
        "n_seeds": 2,
        "snr_db": [10],
        "duration_s": [60],
        "budget_hours": 1.0,
    }
    path = tmp_path / "demo.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class Crosstalk:
    def __init__(self):
        self.calls = []

    def __call__(self, modalities):
        self.calls.append(list(modalities))
        return {m: {"names": ["a", "b"], "matrix": [[1.0, 0.2], [0.3, 1.0]]} for m in modalities}


def call(args, runner=None, gate=None, crosstalk=None):
    return script.main(
        args,
        runner=runner or Runner(),
        gate_measure=gate or Gate(),
        crosstalk_fn=crosstalk or Crosstalk(),
    )


class TestDryRun:
    def test_it_counts_the_tasks_and_writes_nothing(self, protocol_file, tmp_path, capsys):
        results = tmp_path / "out.jsonl"
        runner, gate = Runner(), Gate()
        code = call([str(protocol_file), "--dry-run", "--results", str(results)], runner, gate)
        out = capsys.readouterr().out
        assert code == 0
        assert "4 tasks" in out
        assert "chain" in out and "collider" in out
        assert "gc/bic" in out
        assert not results.exists()
        assert runner.calls == [] and gate.calls == 0


class TestRun:
    def test_a_run_writes_the_results_and_reports_the_timing(self, protocol_file, tmp_path, capsys):
        results = tmp_path / "out.jsonl"
        runner = Runner()
        code = call([str(protocol_file), "--results", str(results), "--workers", "1"], runner)
        out = capsys.readouterr().out
        assert code == 0
        assert len(runner.calls) == 4
        header, records, final = read_results(results)
        assert header["gate"]["estimate_hours"] > 0
        assert len(records) == 4 and final["n_done"] == 4
        assert "estimated" in out and "actual" in out

    def test_the_second_run_skips_and_does_not_measure_the_gate_again(
        self, protocol_file, tmp_path
    ):
        results = tmp_path / "out.jsonl"
        gate = Gate()
        call([str(protocol_file), "--results", str(results), "--workers", "1"], Runner(), gate)
        runner = Runner()
        code = call([str(protocol_file), "--results", str(results), "--workers", "1"], runner, gate)
        assert code == 0
        assert runner.calls == []
        assert gate.calls == 1

    def test_the_gate_of_a_results_file_is_reused_only_for_the_same_worker_count(
        self, protocol_file
    ):
        # tested on the helper: a second worker count would need joblib in the test environment
        protocol = load_protocol(protocol_file)
        tasks = expand(protocol)
        measure = Gate()
        old = measure(protocol, workers=1)
        assert measure.calls == 1
        same = script._gate_for(protocol, tasks, old, 1, measure)
        assert measure.calls == 1 and same["estimate_hours"] == pytest.approx(old["estimate_hours"])
        script._gate_for(protocol, tasks, old, 2, measure)
        assert measure.calls == 2

    def test_a_reused_gate_is_recomputed_for_a_grown_grid_and_measured_for_a_new_size(
        self, protocol_file
    ):
        protocol = load_protocol(protocol_file)
        tasks = expand(protocol)
        measure = Gate()
        old = measure(protocol, workers=1)
        # the same timings, twice the tasks: the estimate doubles without a new measurement
        doubled = script._gate_for(protocol, tasks + tasks, old, 1, measure)
        assert measure.calls == 1
        assert doubled["estimate_hours"] == pytest.approx(2 * old["estimate_hours"])
        # a network size that was never timed forces a new measurement
        old["seconds_per_task"] = {"99": 1.0}
        script._gate_for(protocol, tasks, old, 1, measure)
        assert measure.calls == 2

    def test_failed_tasks_give_exit_code_1_and_a_retry_fixes_it(
        self, protocol_file, tmp_path, capsys
    ):
        results = tmp_path / "out.jsonl"
        code = call(
            [str(protocol_file), "--results", str(results), "--workers", "1"],
            Runner(fail_on_seed_index=1),
        )
        assert code == 1
        assert "2 failed" in capsys.readouterr().out
        runner = Runner()
        code = call([str(protocol_file), "--results", str(results), "--workers", "1"], runner)
        assert code == 0
        assert len(runner.calls) == 2

    def test_an_estimate_over_the_budget_refuses_and_runs_nothing(
        self, protocol_file, tmp_path, capsys
    ):
        results = tmp_path / "out.jsonl"
        runner = Runner()
        code = call(
            [str(protocol_file), "--results", str(results), "--workers", "1"],
            runner,
            Gate(seconds=3600.0),  # 4 tasks of an hour each, against a budget of 1 h
        )
        assert code == 2
        assert runner.calls == []
        assert not results.exists()
        assert "budget" in capsys.readouterr().err

    def test_force_goes_ahead_over_the_budget(self, protocol_file, tmp_path):
        results = tmp_path / "out.jsonl"
        runner = Runner()
        code = call(
            [str(protocol_file), "--results", str(results), "--workers", "1", "--force"],
            runner,
            Gate(seconds=3600.0),
        )
        assert code == 0
        assert len(runner.calls) == 4

    def test_gate_only_measures_prints_and_runs_nothing(self, protocol_file, tmp_path, capsys):
        results = tmp_path / "out.jsonl"
        runner, gate = Runner(), Gate()
        code = call(
            [str(protocol_file), "--results", str(results), "--workers", "1", "--gate-only"],
            runner,
            gate,
        )
        assert code == 0
        assert gate.calls == 1 and runner.calls == []
        assert not results.exists()
        assert "estimated" in capsys.readouterr().out

    def test_a_scenario_with_hidden_nodes_is_accepted(self, tmp_path):
        data = {
            "version": 1,
            "name": "hid",
            "scenarios": ["hidden_common_input"],
            "estimators": ["gc/bic"],
            "n_seeds": 1,
        }
        path = tmp_path / "hid.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        runner = Runner()
        code = call([str(path), "--results", str(tmp_path / "o.jsonl"), "--workers", "1"], runner)
        assert code == 0 and len(runner.calls) == 1

    def test_a_changed_setting_refuses_the_resume_and_says_how_to_start_over(
        self, protocol_file, tmp_path, capsys
    ):
        results = tmp_path / "out.jsonl"
        call([str(protocol_file), "--results", str(results), "--workers", "1"])
        data = yaml.safe_load(protocol_file.read_text(encoding="utf-8"))
        data["estimators"] = ["gc/bic", "dtf/bic"]
        protocol_file.write_text(yaml.safe_dump(data), encoding="utf-8")
        code = call([str(protocol_file), "--results", str(results), "--workers", "1"])
        assert code == 2
        assert "--new-file" in capsys.readouterr().err
        runner = Runner()
        code = call(
            [str(protocol_file), "--results", str(results), "--workers", "1", "--new-file"], runner
        )
        assert code == 0 and len(runner.calls) == 4

    def test_the_default_results_path_is_named_after_the_protocol(
        self, protocol_file, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(script, "RESULTS_DIR", tmp_path / "res")
        call([str(protocol_file), "--workers", "1"])
        assert (tmp_path / "res" / "demo.jsonl").exists()


class TestFromResults:
    def test_it_renders_the_saved_results_without_running_anything(
        self, protocol_file, tmp_path, capsys
    ):
        results = tmp_path / "out.jsonl"
        call([str(protocol_file), "--results", str(results), "--workers", "1"])
        capsys.readouterr()
        runner, gate = Runner(), Gate()
        code = call([str(protocol_file), "--results", str(results), "--from-results"], runner, gate)
        out = capsys.readouterr().out
        assert code == 0
        assert runner.calls == [] and gate.calls == 0
        assert "Benchmark results: demo" in out
        assert "| gc/bic | roi | 0.900 | nan | nan | 4 |" in out

    def test_it_can_write_to_a_file(self, protocol_file, tmp_path):
        results = tmp_path / "out.jsonl"
        call([str(protocol_file), "--results", str(results), "--workers", "1"])
        report = tmp_path / "summary.md"
        call(
            [
                str(protocol_file),
                "--results",
                str(results),
                "--from-results",
                "--output",
                str(report),
            ]
        )
        assert "Benchmark results: demo" in report.read_text(encoding="utf-8")

    def test_a_missing_results_file_is_a_clear_error(self, protocol_file, tmp_path, capsys):
        code = call(
            [str(protocol_file), "--results", str(tmp_path / "none.jsonl"), "--from-results"]
        )
        assert code == 2
        assert "none.jsonl" in capsys.readouterr().err


class TestResolveProtocol:
    def test_a_bare_name_is_looked_up_in_the_protocols_directory(self, tmp_path, monkeypatch):
        directory = tmp_path / "protocols"
        directory.mkdir()
        (directory / "mine.yaml").write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "name": "mine",
                    "scenarios": ["chain"],
                    "estimators": ["gc/bic"],
                    "n_seeds": 1,
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(script, "PROTOCOL_DIR", directory)
        assert script.resolve_protocol("mine") == directory / "mine.yaml"
        assert load_protocol(script.resolve_protocol("mine")).name == "mine"

    def test_an_unknown_name_lists_the_known_ones(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(script, "PROTOCOL_DIR", tmp_path)
        code = call(["nothing", "--dry-run"])
        assert code == 2
        assert "nothing" in capsys.readouterr().err


def test_seeds_in_the_records_are_those_of_task_seed(protocol_file, tmp_path):
    results = tmp_path / "out.jsonl"
    call([str(protocol_file), "--results", str(results), "--workers", "1"])
    protocol = load_protocol(protocol_file)
    expected = {t.task_id: task_seed(protocol.master_seed, t) for t in expand(protocol)}
    _, records, _ = read_results(results)
    assert {r["id"]: r["seed"] for r in records} == expected


class GuardRunner:
    """Chain scores: perfect at 10 dB and 0.7 at 0 dB, or perfect everywhere."""

    def __init__(self, hard=True):
        self.hard = hard

    def __call__(self, task):
        auroc = 1.0 if (task.snr_db >= 10 or not self.hard) else 0.7
        card = {"auroc": auroc, "direction_accuracy": 1.0, "separation": 2.0}
        return {"gc/bic": {"roi": card, "sensor": card, "source": card}}


@pytest.fixture
def guarded_file(tmp_path):
    def make(guard):
        data = {
            "version": 1,
            "name": "guarded",
            "scenarios": ["chain"],
            "estimators": ["gc/bic"],
            "n_seeds": 2,
            "snr_db": [10, 0],
            "duration_s": [60],
            "saturation_guard": guard,
            "budget_hours": 1.0,
        }
        path = tmp_path / "guarded.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return make


class TestSaturationExitCode:
    def test_a_grid_that_reaches_the_hard_end_passes_with_exit_code_0(self, guarded_file, tmp_path):
        code = call(
            [str(guarded_file(True)), "--results", str(tmp_path / "o.jsonl"), "--workers", "1"],
            GuardRunner(hard=True),
        )
        assert code == 0

    def test_a_grid_that_is_perfect_everywhere_gives_exit_code_3_and_says_why(
        self, guarded_file, tmp_path, capsys
    ):
        code = call(
            [str(guarded_file(True)), "--results", str(tmp_path / "o.jsonl"), "--workers", "1"],
            GuardRunner(hard=False),
        )
        assert code == 3
        err = capsys.readouterr().err
        assert "saturation guard" in err and "too easy" in err

    def test_without_the_guard_the_same_run_is_fine(self, guarded_file, tmp_path):
        code = call(
            [str(guarded_file(False)), "--results", str(tmp_path / "o.jsonl"), "--workers", "1"],
            GuardRunner(hard=False),
        )
        assert code == 0

    def test_from_results_gives_the_same_verdict_without_running_anything(
        self, guarded_file, tmp_path, capsys
    ):
        results = tmp_path / "o.jsonl"
        call(
            [str(guarded_file(True)), "--results", str(results), "--workers", "1"],
            GuardRunner(hard=False),
        )
        capsys.readouterr()
        runner = Runner()
        code = call([str(guarded_file(True)), "--results", str(results), "--from-results"], runner)
        captured = capsys.readouterr()
        assert code == 3 and runner.calls == []
        assert "VIOLATED" in captured.out and "too easy" in captured.err

    def test_failed_tasks_take_precedence_over_the_guard(self, guarded_file, tmp_path):
        code = call(
            [str(guarded_file(True)), "--results", str(tmp_path / "o.jsonl"), "--workers", "1"],
            Runner(fail_on_seed_index=1),
        )
        assert code == 1


PAIR_REGIONS = ["bankssts-lh", "cuneus-lh", "insula-lh", "lingual-lh"]


@pytest.fixture
def pairs_file(tmp_path):
    data = {
        "version": 1,
        "name": "pairs",
        "kind": "region_pairs",
        "pairs": {"n": 3, "seed": 0, "regions": PAIR_REGIONS},
        "estimators": ["gc/bic"],
        "n_seeds": 1,
        "modality": ["eeg", "grad"],
        "budget_hours": 1.0,
    }
    path = tmp_path / "pairs.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestRegionPairs:
    def test_the_dry_run_groups_the_pair_tasks_under_one_name(self, pairs_file, capsys):
        assert call([str(pairs_file), "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "6 tasks" in out  # 3 pairs x 2 modalities
        assert "by scenario: pair 6" in out
        assert "pair:" not in out  # not one line per pair

    def test_the_run_stores_the_crosstalk_of_every_modality_in_the_header(
        self, pairs_file, tmp_path
    ):
        results = tmp_path / "o.jsonl"
        crosstalk = Crosstalk()
        code = call(
            [str(pairs_file), "--results", str(results), "--workers", "1"], crosstalk=crosstalk
        )
        assert code == 0
        assert crosstalk.calls == [["eeg", "grad"]]
        header = read_results(results)[0]
        assert set(header["crosstalk"]) == {"eeg", "grad"}
        assert header["crosstalk"]["eeg"]["matrix"] == [[1.0, 0.2], [0.3, 1.0]]

    def test_a_resume_does_not_compute_it_again(self, pairs_file, tmp_path):
        results = tmp_path / "o.jsonl"
        call([str(pairs_file), "--results", str(results), "--workers", "1"])
        crosstalk = Crosstalk()
        call([str(pairs_file), "--results", str(results), "--workers", "1"], crosstalk=crosstalk)
        assert crosstalk.calls == []
        assert "crosstalk" in read_results(results)[0]

    def test_a_scenarios_protocol_never_asks_for_it(self, protocol_file, tmp_path):
        crosstalk = Crosstalk()
        call(
            [str(protocol_file), "--results", str(tmp_path / "o.jsonl"), "--workers", "1"],
            crosstalk=crosstalk,
        )
        assert crosstalk.calls == []
        assert "crosstalk" not in read_results(tmp_path / "o.jsonl")[0]

    def test_the_summary_of_pair_results_has_the_correlation_table(
        self, pairs_file, tmp_path, capsys
    ):
        results = tmp_path / "o.jsonl"
        call([str(pairs_file), "--results", str(results), "--workers", "1"])
        capsys.readouterr()
        assert call([str(pairs_file), "--results", str(results), "--from-results"]) in (0, 3)
        out = capsys.readouterr().out
        assert "## Leakage and crosstalk" in out


class SignificanceRunner:
    """p-values of a region-level study: the true edges of the chain are found, nothing else is."""

    def __call__(self, task):
        nodes = ["transversetemporal-rh", "cuneus-rh", "transversetemporal-lh"]
        p = [[0.5] * 3 for _ in range(3)]
        for i in range(3):
            p[i][i] = float("nan")
        if task.coupling:
            p[0][1] = p[1][2] = 0.0001
        return {
            "gc/fixed": {
                "analysed": nodes,
                "tests": {"analytic": {"pvalues": p}, "surrogate:phase_randomize": {"pvalues": p}},
            }
        }


@pytest.fixture
def significance_file(tmp_path):
    data = {
        "version": 1,
        "name": "sig",
        "kind": "significance",
        "scenarios": ["chain"],
        "estimators": ["gc/fixed"],
        "tests": ["analytic", "surrogate:phase_randomize"],
        "coupling": [0, 0.4],
        "snr_db": [10],
        "duration_s": [30],
        "n_seeds": 3,
        "budget_hours": 1.0,
    }
    path = tmp_path / "sig.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestSignificanceProtocol:
    def test_the_dry_run_counts_the_datasets_and_shows_the_region_level(
        self, significance_file, capsys
    ):
        assert call([str(significance_file), "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "6 tasks" in out and "by modality: roi 6" in out

    def test_a_run_and_its_summary_have_the_significance_section(
        self, significance_file, tmp_path, capsys
    ):
        results = tmp_path / "o.jsonl"
        code = call(
            [str(significance_file), "--results", str(results), "--workers", "1"],
            SignificanceRunner(),
        )
        assert code == 0
        capsys.readouterr()
        assert call([str(significance_file), "--results", str(results), "--from-results"]) == 0
        out = capsys.readouterr().out
        assert "## Significance study" in out
        assert "| analytic |" in out and "| surrogate:phase_randomize |" in out

    def test_no_crosstalk_is_computed_for_it(self, significance_file, tmp_path):
        crosstalk = Crosstalk()
        call(
            [str(significance_file), "--results", str(tmp_path / "o.jsonl"), "--workers", "1"],
            SignificanceRunner(),
            crosstalk=crosstalk,
        )
        assert crosstalk.calls == []


class TestNoiseProtocol:
    def test_the_dry_run_counts_the_tasks_of_each_noise_arm(self, tmp_path, capsys):
        data = {
            "version": 1,
            "name": "noise",
            "scenarios": ["chain"],
            "estimators": ["gc/bic"],
            "noise": ["simulated", "white", "empty_room"],
            "modality": ["grad"],
            "n_seeds": 3,
        }
        path = tmp_path / "noise.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        assert call([str(path), "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "9 tasks" in out
        assert "by noise: simulated 3, white 3, empty_room 3" in out

    def test_a_protocol_with_one_noise_prints_no_noise_line(self, protocol_file, capsys):
        call([str(protocol_file), "--dry-run"])
        assert "by noise" not in capsys.readouterr().out

"""Benchmark protocols: format, tasks, seeds, results file, resume, budget."""

import json
import math

import pytest
import yaml

from dirconn.protocol import (
    CALIBRATION_ESTIMATOR_NAMES,
    ESTIMATOR_NAMES,
    EXTENDED_ESTIMATOR_NAMES,
    PROTOCOL_VERSION,
    SATURATION_THRESHOLD,
    VAR_ESTIMATOR_NAMES,
    Protocol,
    apply_coupling,
    check_budget,
    code_version,
    estimate_hours,
    expand,
    load_protocol,
    read_results,
    region_pair_scenario,
    render_summary,
    run_protocol,
    saturation_report,
    select_pairs,
    task_seed,
)
from dirconn.scenario import ScenarioSpec, list_scenarios, load_scenario


def minimal(**overrides) -> dict:
    data = {
        "version": 1,
        "name": "demo",
        "scenarios": ["chain", "collider"],
        "estimators": ["gc/bic", "dtf/fixed"],
        "n_seeds": 2,
    }
    data.update(overrides)
    return data


def make(**overrides) -> Protocol:
    return Protocol.from_dict(minimal(**overrides))


class TestFormat:
    def test_defaults(self):
        protocol = make()
        assert protocol.name == "demo"
        assert protocol.master_seed == 0
        assert protocol.snr_db == (None,)
        assert protocol.duration_s == (None,)
        assert protocol.coupling == (None,)
        assert protocol.source_model == ("partial",)
        assert protocol.modality == ("eeg",)
        assert protocol.inverse == ("mne",)
        assert protocol.exclude == ()
        assert protocol.budget_hours == 2.0
        assert protocol.settings == {}

    def test_every_estimator_name_is_a_method_and_an_order_setting(self):
        methods = {"gc", "gc_pairwise", "dtf", "pdc", "gpdc"}
        assert set(VAR_ESTIMATOR_NAMES) == {f"{m}/{o}" for m in methods for o in ("bic", "fixed")}
        assert ESTIMATOR_NAMES == (
            VAR_ESTIMATOR_NAMES + EXTENDED_ESTIMATOR_NAMES + CALIBRATION_ESTIMATOR_NAMES
        )

    def test_the_calibration_adds_rpdc_and_a_too_low_order_for_pdc_and_rpdc(self):
        # the labels of the published runs stay as they were
        assert set(CALIBRATION_ESTIMATOR_NAMES) == {"pdc/low", "rpdc/bic", "rpdc/fixed", "rpdc/low"}
        assert not set(CALIBRATION_ESTIMATOR_NAMES) & set(VAR_ESTIMATOR_NAMES)

    def test_load_from_a_file(self, tmp_path):
        path = tmp_path / "demo.yaml"
        path.write_text(yaml.safe_dump(minimal()), encoding="utf-8")
        assert load_protocol(path) == make()

    def test_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_protocol(tmp_path / "nope.yaml")

    def test_a_file_error_names_the_file(self, tmp_path):
        path = tmp_path / "oops.yaml"
        path.write_text(yaml.safe_dump(minimal(snr=[1])), encoding="utf-8")
        with pytest.raises(ValueError, match="oops.yaml"):
            load_protocol(path)

    def test_a_non_mapping_top_level(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_protocol(path)

    @pytest.mark.parametrize("key", ["name", "scenarios", "estimators", "n_seeds"])
    def test_a_missing_required_key_is_named(self, key):
        data = minimal()
        del data[key]
        with pytest.raises(ValueError, match=key):
            Protocol.from_dict(data)

    def test_version_is_required_and_checked(self):
        data = minimal()
        del data["version"]
        with pytest.raises(ValueError, match="version"):
            Protocol.from_dict(data)
        with pytest.raises(ValueError, match=rf"version.*7.*{PROTOCOL_VERSION}"):
            Protocol.from_dict(minimal(version=7))

    def test_an_unknown_key_is_named(self):
        with pytest.raises(ValueError, match="snr"):
            Protocol.from_dict(minimal(snr=[10]))

    def test_an_unknown_scenario_is_named(self):
        with pytest.raises(ValueError, match="no_such"):
            make(scenarios=["chain", "no_such"])

    def test_an_unknown_estimator_is_named(self):
        with pytest.raises(ValueError, match="gc/slow"):
            make(estimators=["gc/slow"])

    @pytest.mark.parametrize(
        ("key", "value"),
        [("source_model", ["blob"]), ("modality", ["ecog"]), ("inverse", ["magic"])],
    )
    def test_an_unknown_choice_is_named(self, key, value):
        with pytest.raises(ValueError, match=value[0]):
            make(**{key: value})

    @pytest.mark.parametrize(
        "key", ["scenarios", "estimators", "snr_db", "duration_s", "coupling", "modality"]
    )
    def test_an_empty_axis_is_refused(self, key):
        with pytest.raises(ValueError, match=key):
            make(**{key: []})

    def test_a_repeated_axis_value_is_refused(self):
        with pytest.raises(ValueError, match="snr_db"):
            make(snr_db=[10, 10])

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("n_seeds", 0),
            ("budget_hours", 0),
            ("master_seed", -1),
            ("duration_s", [0.0]),
            ("coupling", [-0.1]),
            ("snr_db", [float("nan")]),
        ],
    )
    def test_out_of_range_values_name_their_key(self, key, value):
        with pytest.raises(ValueError, match=key):
            make(**{key: value})

    def test_exclude_entries_must_name_axes_that_exist(self):
        with pytest.raises(ValueError, match="colour"):
            make(exclude=[{"colour": "red"}])

    def test_settings_must_be_scalars(self):
        with pytest.raises(ValueError, match="settings"):
            make(settings={"max_order": [1, 2]})

    def test_the_protocol_survives_a_dict_round_trip(self):
        protocol = make(
            snr_db=[10, 0],
            coupling=[0.4, 0],
            exclude=[{"scenario": "collider", "coupling": 0}],
            settings={"max_order": 40},
            budget_hours=0.5,
        )
        assert Protocol.from_dict(protocol.to_dict()) == protocol


class TestCoupling:
    def test_an_absolute_strength_replaces_every_edge_and_keeps_the_sign(self):
        spec = ScenarioSpec.from_dict(
            {
                "version": 1,
                "name": "signs",
                "network": {
                    "nodes": [
                        {"name": "bankssts-lh"},
                        {"name": "insula-rh"},
                        {"name": "lingual-lh"},
                    ],
                    "edges": [
                        {
                            "source": "bankssts-lh",
                            "target": "insula-rh",
                            "strength": 0.3,
                            "delay_ms": 10,
                        },
                        {
                            "source": "insula-rh",
                            "target": "lingual-lh",
                            "strength": -0.3,
                            "delay_ms": 10,
                        },
                    ],
                },
            }
        )
        out = apply_coupling(spec, 0.1)
        assert [e.strength for e in out.network.edges] == [0.1, -0.1]
        assert [e.delay_ms for e in out.network.edges] == [10.0, 10.0]

    def test_zero_removes_every_edge(self):
        out = apply_coupling(load_scenario("chain"), 0.0)
        assert out.network.edges == ()
        assert out.network.names == load_scenario("chain").network.names

    def test_edges_of_different_strength_are_refused_naming_scenario_and_edge(self):
        with pytest.raises(ValueError, match=r"random_5.*strength"):
            apply_coupling(load_scenario("random_5"), 0.2)

    def test_the_protocol_refuses_a_coupling_axis_on_such_a_scenario_at_load_time(self):
        with pytest.raises(ValueError, match="random_5"):
            make(scenarios=["chain", "random_5"], coupling=[0.2])

    def test_without_a_coupling_axis_mixed_strengths_are_fine(self):
        make(scenarios=["random_5"])


class TestExpand:
    def test_the_count_is_the_product_of_the_axes(self):
        protocol = make(snr_db=[10, 0], duration_s=[60, 120], modality=["eeg", "grad"])
        assert len(expand(protocol)) == 2 * 2 * 2 * 2 * 2 * 1

    def test_estimators_do_not_multiply_the_tasks(self):
        few = make(estimators=["gc/bic"])
        many = make(estimators=list(ESTIMATOR_NAMES))
        assert len(expand(few)) == len(expand(many))

    def test_unset_axes_take_the_scenario_default(self):
        tasks = expand(make(scenarios=["chain"], n_seeds=1))
        (task,) = tasks
        spec = load_scenario("chain")
        assert task.snr_db == spec.snr_db
        assert task.duration_s == spec.duration_s
        assert task.coupling is None

    def test_explicit_axis_values_are_used(self):
        tasks = expand(make(scenarios=["chain"], snr_db=[0, -5], n_seeds=1))
        assert sorted(t.snr_db for t in tasks) == [-5.0, 0.0]

    def test_exclude_drops_matching_tasks_only(self):
        protocol = make(
            snr_db=[10, 0],
            coupling=[0.4, 0],
            exclude=[{"scenario": "collider", "coupling": 0}, {"snr_db": [0], "coupling": 0}],
        )
        tasks = expand(protocol)
        assert not [t for t in tasks if t.scenario == "collider" and t.coupling == 0]
        assert not [t for t in tasks if t.snr_db == 0 and t.coupling == 0]
        assert [t for t in tasks if t.scenario == "chain" and t.snr_db == 10 and t.coupling == 0]
        full = 2 * 2 * 2 * 2
        # collider/coupling 0: 2 snr x 2 seeds = 4; chain/snr 0/coupling 0: 2 seeds = 2
        assert len(tasks) == full - 4 - 2

    def test_the_order_is_deterministic(self):
        assert [t.task_id for t in expand(make())] == [t.task_id for t in expand(make())]

    def test_task_ids_are_unique(self):
        protocol = make(snr_db=[10, 0], duration_s=[60, 120], modality=["eeg", "grad"])
        ids = [t.task_id for t in expand(protocol)]
        assert len(set(ids)) == len(ids)

    def test_zero_coupling_marks_a_zero_network(self):
        tasks = expand(make(scenarios=["chain"], coupling=[0, 0.4], n_seeds=1))
        assert {t.coupling: t.zero_network for t in tasks} == {0.0: True, 0.4: False}

    def test_a_task_gives_the_scenario_it_stands_for(self):
        (task,) = expand(
            make(scenarios=["chain"], snr_db=[0], duration_s=[60], coupling=[0.2], n_seeds=1)
        )
        spec = task.scenario_spec()
        assert spec.snr_db == 0.0
        assert spec.duration_s == 60.0
        assert [e.strength for e in spec.network.edges] == [0.2, 0.2]

    def test_the_task_dict_round_trips_through_json(self):
        (task, *_) = expand(make())
        assert json.loads(json.dumps(task.to_dict())) == task.to_dict()


class TestSeeds:
    def test_the_seed_is_stable_and_in_range(self):
        tasks = expand(make())
        again = expand(make())
        assert [task_seed(0, t) for t in tasks] == [task_seed(0, t) for t in again]
        assert all(0 <= task_seed(0, t) < 2**31 for t in tasks)

    def test_the_master_seed_changes_every_seed(self):
        tasks = expand(make())
        assert all(task_seed(0, t) != task_seed(1, t) for t in tasks)

    def test_seeds_are_distinct_over_a_large_grid(self):
        protocol = make(
            scenarios=["chain", "collider", "common_driver"],
            snr_db=[20, 10, 0, -5],
            duration_s=[60, 120, 240],
            modality=["eeg", "mag", "grad"],
            n_seeds=10,
        )
        tasks = expand(protocol)
        assert len(tasks) == 3 * 4 * 3 * 3 * 10
        seeds = [task_seed(0, t) for t in tasks]
        assert len(set(seeds)) == len(seeds)

    def test_adding_a_grid_value_leaves_the_old_tasks_alone(self):
        old = {t.task_id: task_seed(0, t) for t in expand(make(snr_db=[10, 0]))}
        new = {t.task_id: task_seed(0, t) for t in expand(make(snr_db=[10, 0, -5]))}
        assert len(new) > len(old)
        assert old.items() <= new.items()

    def test_adding_a_seed_leaves_the_old_tasks_alone(self):
        old = {t.task_id: task_seed(0, t) for t in expand(make(n_seeds=2))}
        new = {t.task_id: task_seed(0, t) for t in expand(make(n_seeds=3))}
        assert old.items() <= new.items()

    def test_the_order_of_the_axis_values_does_not_matter(self):
        a = {t.task_id: task_seed(0, t) for t in expand(make(snr_db=[10, 0]))}
        b = {t.task_id: task_seed(0, t) for t in expand(make(snr_db=[0, 10]))}
        assert a == b

    def test_editing_a_scenario_changes_the_id_but_not_the_seed(self):
        chain = load_scenario("chain")
        data = chain.to_dict()
        data["network"]["edges"][0]["strength"] = 0.3
        edited = ScenarioSpec.from_dict(data)
        protocol = make(scenarios=["chain"], n_seeds=1)
        (before,) = expand(protocol)
        (after,) = expand(protocol, scenarios={"chain": edited})
        assert before.task_id != after.task_id
        assert before.scenario_hash != after.scenario_hash
        assert task_seed(0, before) == task_seed(0, after)


def fake_runner(task):
    return {"value": task_seed(0, task) % 7, "scenario": task.scenario}


class Counting:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def __call__(self, task):
        self.calls.append(task.task_id)
        if self.fail_on is not None and self.calls[-1] == self.fail_on:
            raise RuntimeError("boom")
        return fake_runner(task)


ENV = {"python": "3.11", "numpy": "2.4.6", "scipy": "1.17.1", "mne": "1.13.2", "dirconn": "0.0"}


def eight(**overrides) -> Protocol:
    """Two scenarios, two SNR values, two seeds: eight tasks."""
    overrides.setdefault("snr_db", [10, 0])
    return make(**overrides)


class TestRun:
    def test_a_run_writes_header_tasks_and_summary(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        protocol = eight()
        summary = run_protocol(protocol, fake_runner, path, environment=ENV)
        assert summary.n_tasks == 8 and summary.n_done == 8 and summary.n_failed == 0
        header, records, final = read_results(path)
        assert header["protocol"] == "demo"
        assert header["environment"] == ENV
        assert header["settings"]["estimators"] == ["gc/bic", "dtf/fixed"]
        assert len(records) == 8
        assert {r["id"] for r in records} == {t.task_id for t in expand(protocol)}
        assert all(r["status"] == "ok" and "seconds" in r for r in records)
        assert final["n_done"] == 8 and final["n_failed"] == 0 and "seconds" in final

    def test_a_record_carries_the_task_the_seed_and_the_payload(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        run_protocol(eight(), fake_runner, path, environment=ENV)
        _, records, _ = read_results(path)
        first = records[0]
        task = expand(eight())[0]
        assert first["task"] == task.to_dict()
        assert first["seed"] == task_seed(0, task)
        assert first["scenario_hash"] == task.scenario_hash
        assert first["result"] == fake_runner(task)

    def test_a_second_run_skips_everything_done(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        run_protocol(eight(), fake_runner, path, environment=ENV)
        counting = Counting()
        summary = run_protocol(eight(), counting, path, environment=ENV)
        assert counting.calls == []
        assert summary.n_skipped == 8 and summary.n_done == 0
        assert len(read_results(path)[1]) == 8

    def test_a_grown_grid_runs_only_the_new_tasks(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        run_protocol(eight(snr_db=[10]), fake_runner, path, environment=ENV)
        counting = Counting()
        run_protocol(eight(snr_db=[10, 0]), counting, path, environment=ENV)
        assert len(counting.calls) == 4
        assert len(read_results(path)[1]) == 8

    def test_a_failed_task_is_recorded_and_the_rest_still_run(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        tasks = expand(eight())
        counting = Counting(fail_on=tasks[2].task_id)
        summary = run_protocol(eight(), counting, path, environment=ENV)
        assert summary.n_failed == 1 and summary.n_done == 7
        _, records, final = read_results(path)
        (failed,) = [r for r in records if r["status"] == "failed"]
        assert failed["id"] == tasks[2].task_id
        assert failed["error"]["type"] == "RuntimeError"
        assert "boom" in failed["error"]["message"]
        assert "Traceback" in failed["error"]["traceback"]
        assert final["n_failed"] == 1

    def test_a_rerun_retries_only_the_failed_task(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        tasks = expand(eight())
        run_protocol(eight(), Counting(fail_on=tasks[2].task_id), path, environment=ENV)
        counting = Counting()
        summary = run_protocol(eight(), counting, path, environment=ENV)
        assert counting.calls == [tasks[2].task_id]
        assert summary.n_failed == 0
        _, records, _ = read_results(path)
        ok = [r for r in records if r["status"] == "ok"]
        assert {r["id"] for r in ok} == {t.task_id for t in tasks}

    def test_a_task_of_an_edited_scenario_is_not_taken_as_done(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        protocol = make(scenarios=["chain"], n_seeds=1)
        run_protocol(protocol, fake_runner, path, environment=ENV)
        data = load_scenario("chain").to_dict()
        data["network"]["edges"][0]["strength"] = 0.3
        edited = {"chain": ScenarioSpec.from_dict(data)}
        counting = Counting()
        run_protocol(protocol, counting, path, environment=ENV, scenarios=edited)
        assert len(counting.calls) == 1
        # the result of the old scenario is replaced, not kept next to the new one
        assert len(read_results(path)[1]) == 1

    def test_a_truncated_last_line_is_ignored_and_that_task_runs_again(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        run_protocol(eight(), fake_runner, path, environment=ENV)
        lines = path.read_text(encoding="utf-8").splitlines()
        # drop the summary and cut the last task record in half, as a crash would
        path.write_text("\n".join(lines[:-2]) + "\n" + lines[-2][:25], encoding="utf-8")
        _, records, final = read_results(path)
        assert len(records) == 7 and final is None
        counting = Counting()
        run_protocol(eight(), counting, path, environment=ENV)
        assert len(counting.calls) == 1
        assert len(read_results(path)[1]) == 8

    def test_changed_estimators_refuse_a_resume_and_name_the_difference(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        run_protocol(eight(), fake_runner, path, environment=ENV)
        with pytest.raises(ValueError, match=r"estimators.*new_file"):
            run_protocol(eight(estimators=["gc/bic"]), fake_runner, path, environment=ENV)

    @pytest.mark.parametrize(
        "change", [{"master_seed": 5}, {"settings": {"max_order": 3}}, {"inverse": ["lcmv"]}]
    )
    def test_other_numeric_settings_also_refuse_a_resume(self, tmp_path, change):
        path = tmp_path / "demo.jsonl"
        run_protocol(eight(), fake_runner, path, environment=ENV)
        with pytest.raises(ValueError, match="new_file"):
            run_protocol(eight(**change), fake_runner, path, environment=ENV)

    def test_a_changed_package_version_refuses_a_resume_and_names_it(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        run_protocol(eight(), fake_runner, path, environment=ENV)
        with pytest.raises(ValueError, match=r"numpy.*2.4.6.*2.5.0"):
            run_protocol(eight(), fake_runner, path, environment={**ENV, "numpy": "2.5.0"})

    def test_the_platform_may_change_between_resumes(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        run_protocol(eight(), fake_runner, path, environment={**ENV, "platform": "Windows"})
        counting = Counting()
        run_protocol(eight(), counting, path, environment={**ENV, "platform": "Linux"})
        assert counting.calls == []

    def test_the_grid_and_the_budget_may_change_between_resumes(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        run_protocol(eight(budget_hours=1), fake_runner, path, environment=ENV)
        run_protocol(eight(budget_hours=3, snr_db=[10, 0]), fake_runner, path, environment=ENV)

    def test_new_file_starts_over(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        run_protocol(eight(), fake_runner, path, environment=ENV)
        counting = Counting()
        run_protocol(eight(estimators=["gc/bic"]), counting, path, environment=ENV, new_file=True)
        assert len(counting.calls) == 8
        header, records, _ = read_results(path)
        assert header["settings"]["estimators"] == ["gc/bic"]
        assert len(records) == 8

    def test_a_file_that_is_not_a_results_file_is_refused(self, tmp_path):
        path = tmp_path / "other.jsonl"
        path.write_text('{"hello": 1}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="results file"):
            run_protocol(eight(), fake_runner, path, environment=ENV)

    def test_the_gate_measurement_goes_into_the_header(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        gate = {"seconds_per_task": {"3": 1.5}, "estimate_hours": 0.01}
        run_protocol(eight(), fake_runner, path, environment=ENV, gate=gate)
        assert read_results(path)[0]["gate"] == gate

    def test_progress_is_reported_per_task(self, tmp_path):
        seen = []
        run_protocol(
            eight(), fake_runner, tmp_path / "d.jsonl", environment=ENV, progress=seen.append
        )
        assert len(seen) == 8

    def test_workers_must_be_positive(self, tmp_path):
        with pytest.raises(ValueError, match="workers"):
            run_protocol(eight(), fake_runner, tmp_path / "d.jsonl", environment=ENV, workers=0)

    def test_two_workers_give_the_same_records(self, tmp_path):
        pytest.importorskip("joblib")
        serial = tmp_path / "serial.jsonl"
        parallel = tmp_path / "parallel.jsonl"
        run_protocol(eight(), fake_runner, serial, environment=ENV)
        run_protocol(eight(), fake_runner, parallel, environment=ENV, workers=2)
        a = {r["id"]: r["result"] for r in read_results(serial)[1]}
        b = {r["id"]: r["result"] for r in read_results(parallel)[1]}
        assert a == b

    def test_a_missing_joblib_is_a_clear_error_for_several_workers(self, tmp_path, monkeypatch):
        import builtins

        real = builtins.__import__

        def fake(name, *args, **kwargs):
            if name == "joblib":
                raise ImportError("no joblib")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake)
        with pytest.raises(ImportError, match="joblib"):
            run_protocol(eight(), fake_runner, tmp_path / "d.jsonl", environment=ENV, workers=2)


class TestBudget:
    def test_the_estimate_sums_the_tasks_by_network_size(self):
        tasks = expand(make(scenarios=["chain", "random_5"], n_seeds=2))
        # chain has 3 nodes, random_5 has 5; measured at the default 120 s
        seconds = {3: 10.0, 5: 20.0}
        hours = estimate_hours(
            tasks, seconds, reference_duration_s=120.0, workers=1, efficiency=1.0
        )
        assert hours == pytest.approx((2 * 10.0 + 2 * 20.0) / 3600.0)

    def test_it_scales_linearly_with_the_data_length(self):
        short = expand(make(scenarios=["chain"], duration_s=[60], n_seeds=1))
        long = expand(make(scenarios=["chain"], duration_s=[240], n_seeds=1))
        kw = {"reference_duration_s": 120.0, "workers": 1, "efficiency": 1.0}
        a = estimate_hours(short, {3: 10.0}, **kw)
        b = estimate_hours(long, {3: 10.0}, **kw)
        assert b == pytest.approx(4 * a)

    def test_it_divides_by_workers_and_efficiency(self):
        tasks = expand(make(scenarios=["chain"], n_seeds=2))
        one = estimate_hours(
            tasks, {3: 10.0}, reference_duration_s=120.0, workers=1, efficiency=1.0
        )
        four = estimate_hours(
            tasks, {3: 10.0}, reference_duration_s=120.0, workers=4, efficiency=0.8
        )
        assert four == pytest.approx(one / (4 * 0.8))

    def test_the_startup_time_is_added_once_to_the_total(self):
        tasks = expand(make(scenarios=["chain"], n_seeds=2))
        kw = {"reference_duration_s": 120.0, "workers": 4, "efficiency": 0.5}
        plain = estimate_hours(tasks, {3: 10.0}, **kw)
        with_startup = estimate_hours(tasks, {3: 10.0}, startup_seconds=1800.0, **kw)
        assert with_startup == pytest.approx(plain + 0.5)

    def test_a_network_size_without_a_measurement_is_an_error(self):
        tasks = expand(make(scenarios=["chain", "random_5"], n_seeds=1))
        with pytest.raises(ValueError, match="5"):
            estimate_hours(tasks, {3: 10.0}, reference_duration_s=120.0, workers=1)

    def test_a_run_over_budget_is_refused_with_both_numbers(self):
        with pytest.raises(ValueError, match=r"2\.5.*2.*force"):
            check_budget(2.5, 2.0)

    def test_force_goes_ahead(self):
        check_budget(2.5, 2.0, force=True)

    def test_within_budget_is_fine(self):
        check_budget(1.9, 2.0)

    def test_the_estimate_must_be_a_finite_number(self):
        with pytest.raises(ValueError, match="estimate"):
            check_budget(math.nan, 2.0)


def card(auroc, direction=1.0, separation=2.0):
    return {
        "auroc": auroc,
        "auprc": 0.5,
        "direction_accuracy": direction,
        "separation": separation,
    }


class TestRender:
    def records(self):
        def rec(seed, gc_roi, gc_sensor, status="ok"):
            return {
                "type": "task",
                "status": status,
                "id": f"t{seed}",
                "task": {"scenario": "chain"},
                "result": {
                    "gc/bic": {
                        "roi": card(gc_roi, direction=1.0, separation=4.0),
                        "sensor": card(gc_sensor, direction=0.5, separation=1.0),
                        "agreement": {"spearman": 0.8, "n_pairs": 6},
                    },
                    "dtf/fixed": {
                        "roi": card(0.5),
                        "sensor": card(float("nan")),
                        "agreement": {"spearman": float("nan"), "n_pairs": 6},
                    },
                },
            }

        return [rec(0, 1.0, 0.8), rec(1, 0.5, 0.6), {"type": "task", "status": "failed", "id": "x"}]

    def header(self):
        return {
            "protocol": "demo",
            "environment": ENV,
            "gate": {"estimate_hours": 0.25, "seconds_per_task": {"3": 2.0}},
            "settings": {"estimators": ["gc/bic", "dtf/fixed"]},
        }

    def test_the_table_has_one_row_per_estimator_and_stage_with_the_four_measures(self):
        text = render_summary(self.header(), self.records(), {"seconds": 3600.0, "n_failed": 1})
        header = "| estimator | stage | mean AUROC | mean direction | median separation | n |"
        assert header in text
        assert "| gc/bic | roi | 0.750 | 1.000 | 4.00 | 2 |" in text
        assert "| gc/bic | sensor | 0.700 | 0.500 | 1.00 | 2 |" in text
        assert "| dtf/fixed | roi | 0.500 | 1.000 | 2.00 | 2 |" in text

    def test_nan_scores_are_left_out_of_the_mean_and_the_count(self):
        text = render_summary(self.header(), self.records(), None)
        assert "| dtf/fixed | sensor | nan | nan | nan | 0 |" in text

    def test_an_infinite_separation_is_left_out_of_the_median(self):
        records = self.records()
        records[0]["result"]["gc/bic"]["roi"]["separation"] = float("inf")
        text = render_summary(self.header(), records, None)
        assert "| gc/bic | roi | 0.750 | 1.000 | 4.00 | 2 |" in text

    def test_the_stage_agreement_has_its_own_table(self):
        records = self.records()
        records[0]["result"]["gc/bic"]["agreement"]["spearman_roi_source"] = 0.5
        records[1]["result"]["gc/bic"]["agreement"]["spearman_roi_source"] = 0.7
        text = render_summary(self.header(), records, None)
        assert "| estimator | ROI vs sensor | ROI vs source |" in text
        assert "| gc/bic | 0.800 | 0.600 |" in text
        assert "| dtf/fixed | nan | nan |" in text

    def test_it_reports_environment_gate_counts_and_the_failure(self):
        text = render_summary(self.header(), self.records(), {"seconds": 3600.0, "n_failed": 1})
        assert "numpy 2.4.6" in text
        assert "0.25" in text  # the gate estimate
        assert "2 tasks done" in text
        assert "1 failed" in text
        assert "1.00 h" in text  # actual run time next to the estimate

    def test_an_unfinished_run_is_said_so(self):
        text = render_summary(self.header(), self.records(), None)
        assert "unfinished" in text

    def test_failed_tasks_do_not_enter_the_means(self):
        records = self.records()
        records[2]["result"] = {"gc/bic": {"roi": card(0.0), "sensor": card(0.0)}}
        text = render_summary(self.header(), records, None)
        assert "| gc/bic | roi | 0.750 | 1.000 | 4.00 | 2 |" in text

    def test_the_saturation_section_appears_only_for_a_protocol_that_asks_for_it(self):
        assert "Saturation" not in render_summary(self.header(), self.records(), None)
        header = {**self.header(), "saturation_guard": True}
        text = render_summary(header, self.records(), None)
        assert "## Saturation guard" in text


def chain_record(seed, snr, gc, dtf=None, modality="eeg", scenario="chain", status="ok"):
    """A chain task whose sensor and source stages score `gc` (and `dtf`) at this grid point."""
    stages = lambda a: {  # noqa: E731
        "roi": card(1.0),
        "sensor": card(a),
        "source": card(a),
    }
    result = {"gc/bic": stages(gc)}
    if dtf is not None:
        result["dtf/bic"] = stages(dtf)
    return {
        "type": "task",
        "status": status,
        "id": f"{scenario}-{snr}-{seed}-{modality}",
        "task": {
            "scenario": scenario,
            "snr_db": snr,
            "duration_s": 60.0,
            "coupling": None,
            "source_model": "partial",
            "modality": modality,
            "seed_index": seed,
        },
        "result": result,
    }


class TestSaturation:
    def test_a_grid_from_perfect_to_broken_is_fine(self):
        records = [chain_record(s, 10.0, 1.0) for s in range(2)] + [
            chain_record(s, -10.0, 0.7) for s in range(2)
        ]
        report = saturation_report(records)
        assert report.ok and report.problems == []
        assert report.stages["sensor"].hardest_auroc == pytest.approx(0.7)
        assert report.stages["sensor"].hardest_grid["snr_db"] == -10.0
        assert report.stages["source"].easiest_auroc == pytest.approx(1.0)

    def test_a_grid_where_everything_is_perfect_is_saturated_at_both_stages(self):
        records = [chain_record(s, snr, 1.0) for s in range(2) for snr in (10.0, 0.0)]
        report = saturation_report(records)
        assert not report.ok
        text = " ".join(report.problems)
        assert "sensor" in text and "source" in text and "perfect" in text

    def test_a_grid_where_everything_is_broken_has_no_anchor(self):
        records = [chain_record(s, snr, 0.6) for s in range(2) for snr in (-10.0, -15.0)]
        report = saturation_report(records)
        assert not report.ok
        assert "anchor" in " ".join(report.problems)

    def test_the_best_estimator_decides_not_an_estimator_that_is_always_imperfect(self):
        # DTF reports the indirect path A -> C, so it is never perfect on a chain; that is not
        # a sign that the grid is hard
        records = [chain_record(s, snr, 1.0, dtf=0.875) for s in range(2) for snr in (10.0, 0.0)]
        report = saturation_report(records)
        assert not report.ok
        # the verdict is "too easy" (the best estimator is perfect everywhere), not "no anchor",
        # which is what judging by the worst estimator would say
        assert any("too easy" in p for p in report.problems)
        assert not any("anchor" in p for p in report.problems)

    def test_exactly_the_threshold_counts_as_perfect(self):
        report = saturation_report([chain_record(s, 10.0, 0.99) for s in range(2)])
        assert not report.ok
        assert any("too easy" in p for p in report.problems)
        just_below = saturation_report([chain_record(s, 10.0, 0.989) for s in range(2)])
        assert not any("too easy" in p for p in just_below.problems)

    def test_seeds_are_averaged_before_the_threshold_is_applied(self):
        # one seed perfect, one at 0.9: the mean 0.95 is below 0.99, so this point is hard
        records = [chain_record(0, 0.0, 1.0), chain_record(1, 0.0, 0.9), chain_record(0, 10.0, 1.0)]
        records += [chain_record(1, 10.0, 1.0)]
        report = saturation_report(records)
        assert report.ok
        assert report.stages["sensor"].hardest_auroc == pytest.approx(0.95)

    def test_the_region_level_stage_is_not_judged(self):
        records = [chain_record(s, snr, 1.0) for s in range(2) for snr in (10.0, 0.0)]
        report = saturation_report(records)
        assert set(report.stages) == {"sensor", "source"}

    def test_grid_points_are_not_pooled_across_modalities(self):
        records = [
            chain_record(0, 0.0, 1.0, modality="eeg"),
            chain_record(0, 0.0, 0.5, modality="grad"),
        ]
        report = saturation_report(records)
        assert report.ok  # eeg is the anchor, grad the hard point
        assert report.stages["sensor"].hardest_grid["modality"] == "grad"

    def test_failed_tasks_and_other_scenarios_and_nan_are_ignored(self):
        records = [chain_record(0, 10.0, 1.0), chain_record(0, 0.0, 0.5, status="failed")]
        records += [chain_record(0, 0.0, 0.5, scenario="collider")]
        records += [chain_record(0, -5.0, float("nan"))]
        report = saturation_report(records)
        assert not report.ok  # nothing hard is left once those are ignored
        assert report.stages["sensor"].hardest_auroc == pytest.approx(1.0)

    def test_no_chain_at_all_is_a_problem_naming_the_scenario(self):
        report = saturation_report([chain_record(0, 0.0, 0.5, scenario="collider")])
        assert not report.ok
        assert "chain" in " ".join(report.problems)

    def test_another_scenario_can_be_the_reference(self):
        records = [chain_record(0, 10.0, 1.0, scenario="collider")]
        records += [chain_record(0, -10.0, 0.6, scenario="collider")]
        assert saturation_report(records, scenario="collider").ok

    def test_the_threshold_is_the_documented_one(self):
        assert SATURATION_THRESHOLD == 0.99

    def test_the_rendered_section_names_the_hardest_point_and_the_verdict(self):
        header = {"protocol": "demo", "environment": ENV, "saturation_guard": True, "settings": {}}
        ok = [chain_record(s, 10.0, 1.0) for s in range(2)]
        ok += [chain_record(s, -10.0, 0.7) for s in range(2)]
        text = render_summary(header, ok, None)
        assert "## Saturation guard" in text
        assert "ok" in text.split("## Saturation guard")[1]
        assert "-10" in text.split("## Saturation guard")[1]
        bad = [chain_record(s, 10.0, 1.0) for s in range(2)]
        assert "VIOLATED" in render_summary(header, bad, None)


class TestProtocolKey:
    def test_the_saturation_guard_is_off_by_default_and_a_bool(self):
        assert make().saturation_guard is False
        assert make(saturation_guard=True).saturation_guard is True
        with pytest.raises(ValueError, match="saturation_guard"):
            make(saturation_guard="yes")

    def test_it_survives_a_round_trip(self):
        protocol = make(saturation_guard=True)
        assert Protocol.from_dict(protocol.to_dict()) == protocol

    def test_it_goes_into_the_header_but_does_not_block_a_resume(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        run_protocol(make(saturation_guard=True), fake_runner, path, environment=ENV)
        assert read_results(path)[0]["saturation_guard"] is True
        counting = Counting()
        run_protocol(make(saturation_guard=False), counting, path, environment=ENV)
        assert counting.calls == []


# --- region pairs ----------------------------------------------------------------

REGIONS = [
    "bankssts-lh",
    "bankssts-rh",
    "cuneus-lh",
    "cuneus-rh",
    "insula-lh",
    "insula-rh",
    "lingual-lh",
]  # 7 regions: 21 unordered pairs


def pairs_data(**overrides) -> dict:
    data = {
        "version": 1,
        "name": "pairs",
        "kind": "region_pairs",
        "pairs": {"n": 6, "seed": 3, "regions": REGIONS},
        "estimators": ["gc/bic"],
        "n_seeds": 1,
    }
    data.update(overrides)
    return data


def make_pairs(**overrides) -> Protocol:
    return Protocol.from_dict(pairs_data(**overrides))


class TestRegionPairsFormat:
    def test_the_kind_defaults_to_scenarios(self):
        assert make().kind == "scenarios" and make().pairs is None

    def test_a_region_pairs_protocol_has_no_scenarios(self):
        protocol = make_pairs()
        assert protocol.kind == "region_pairs"
        assert protocol.scenarios == ()
        assert protocol.pairs == {"n": 6, "seed": 3, "regions": REGIONS}

    def test_it_survives_a_round_trip(self):
        protocol = make_pairs()
        assert Protocol.from_dict(protocol.to_dict()) == protocol

    def test_an_unknown_kind_is_named(self):
        with pytest.raises(ValueError, match="kind.*blobs"):
            Protocol.from_dict(pairs_data(kind="blobs"))

    def test_region_pairs_need_a_pairs_block(self):
        data = pairs_data()
        del data["pairs"]
        with pytest.raises(ValueError, match="pairs"):
            Protocol.from_dict(data)

    def test_region_pairs_refuse_a_scenarios_list(self):
        with pytest.raises(ValueError, match="scenarios"):
            Protocol.from_dict(pairs_data(scenarios=["chain"]))

    def test_scenarios_protocols_refuse_a_pairs_block(self):
        with pytest.raises(ValueError, match="pairs"):
            Protocol.from_dict(minimal(pairs={"n": 2, "seed": 0, "regions": REGIONS}))

    def test_the_coupling_axis_makes_no_sense_for_pairs(self):
        with pytest.raises(ValueError, match="coupling"):
            Protocol.from_dict(pairs_data(coupling=[0.4]))

    @pytest.mark.parametrize(
        ("pairs", "match"),
        [
            ({"seed": 0, "regions": REGIONS}, "n"),
            ({"n": 2, "regions": REGIONS}, "seed"),
            ({"n": 2, "seed": 0}, "regions"),
            ({"n": 0, "seed": 0, "regions": REGIONS}, "n"),
            ({"n": 2, "seed": -1, "regions": REGIONS}, "seed"),
            ({"n": 2, "seed": 0, "regions": ["cuneus-lh"]}, "regions"),
            ({"n": 2, "seed": 0, "regions": ["cuneus-lh", "cuneus-lh", "insula-lh"]}, "regions"),
            ({"n": 2, "seed": 0, "regions": ["cuneus", "insula-lh"]}, "cuneus"),
            ({"n": 22, "seed": 0, "regions": REGIONS}, "21"),
            ({"n": 2, "seed": 0, "regions": REGIONS, "extra": 1}, "extra"),
        ],
    )
    def test_bad_pairs_blocks_name_the_problem(self, pairs, match):
        with pytest.raises(ValueError, match=match):
            Protocol.from_dict(pairs_data(pairs=pairs))


class TestSelectPairs:
    def test_it_returns_n_distinct_unordered_pairs_of_the_regions(self):
        chosen = select_pairs(REGIONS, 10, seed=1)
        assert len(chosen) == 10 and len(set(chosen)) == 10
        for x, y in chosen:
            assert x in REGIONS and y in REGIONS and x != y
            assert REGIONS.index(x) < REGIONS.index(y)

    def test_it_is_reproducible_and_the_seed_matters(self):
        assert select_pairs(REGIONS, 10, seed=1) == select_pairs(REGIONS, 10, seed=1)
        assert select_pairs(REGIONS, 10, seed=1) != select_pairs(REGIONS, 10, seed=2)

    def test_asking_for_more_keeps_the_pairs_already_chosen(self):
        assert select_pairs(REGIONS, 15, seed=4)[:8] == select_pairs(REGIONS, 8, seed=4)

    def test_all_pairs_can_be_asked_for_and_no_more(self):
        assert len(select_pairs(REGIONS, 21, seed=0)) == 21
        with pytest.raises(ValueError, match="21"):
            select_pairs(REGIONS, 22, seed=0)


class TestRegionPairScenario:
    def test_two_nodes_no_edge_and_two_oscillators(self):
        spec = region_pair_scenario("cuneus-lh", "insula-rh")
        assert spec.network.names == ("cuneus-lh", "insula-rh")
        assert spec.network.edges == ()
        assert spec.dynamics["cuneus-lh"].freq_hz == 10.0
        assert spec.dynamics["insula-rh"].freq_hz == 20.0
        assert all(d.kind == "oscillator" for d in spec.dynamics.values())

    def test_it_is_a_valid_scenario(self):
        spec = region_pair_scenario("cuneus-lh", "insula-rh")
        assert ScenarioSpec.from_dict(spec.to_dict()) == spec


class TestRegionPairTasks:
    def test_there_is_one_task_per_pair_and_grid_point(self):
        protocol = make_pairs(snr_db=[10, 0], modality=["eeg", "grad"], n_seeds=2)
        assert len(expand(protocol)) == 6 * 2 * 2 * 2

    def test_a_task_is_a_zero_network_of_two_nodes_named_after_its_pair(self):
        protocol = make_pairs()
        tasks = expand(protocol)
        expected = select_pairs(REGIONS, 6, seed=3)
        assert [t.scenario for t in tasks] == [f"pair:{x}:{y}" for x, y in expected]
        for task in tasks:
            assert task.coupling == 0.0 and task.zero_network
            assert task.n_nodes == 2
            assert task.scenario_spec().network.edges == ()

    def test_ids_and_seeds_are_unique_and_stable_when_more_pairs_are_added(self):
        small = {t.task_id: task_seed(0, t) for t in expand(make_pairs())}
        big_protocol = make_pairs(pairs={"n": 9, "seed": 3, "regions": REGIONS})
        big = {t.task_id: task_seed(0, t) for t in expand(big_protocol)}
        assert len(small) == 6 and len(big) == 9
        assert small.items() <= big.items()
        assert len(set(big.values())) == 9

    def test_exclude_can_drop_a_pair_by_its_name(self):
        tasks = expand(make_pairs())
        name = tasks[0].scenario
        kept = expand(make_pairs(exclude=[{"scenario": name}]))
        assert len(kept) == 5 and name not in {t.scenario for t in kept}

    def test_a_results_file_records_the_kind(self, tmp_path):
        path = tmp_path / "pairs.jsonl"
        run_protocol(make_pairs(), fake_runner, path, environment=ENV)
        assert read_results(path)[0]["protocol_kind"] == "region_pairs"
        assert read_results(path)[0]["pairs"]["n"] == 6

    def test_header_extra_is_stored_and_kept_on_resume(self, tmp_path):
        path = tmp_path / "pairs.jsonl"
        extra = {"crosstalk": {"eeg": {"names": ["a"], "matrix": [[1.0]]}}}
        run_protocol(make_pairs(), fake_runner, path, environment=ENV, header_extra=extra)
        assert read_results(path)[0]["crosstalk"] == extra["crosstalk"]
        run_protocol(make_pairs(), fake_runner, path, environment=ENV)  # a resume without it
        assert read_results(path)[0]["crosstalk"] == extra["crosstalk"]


class TestRenderOfRegionPairs:
    def pair_header(self):
        return {
            "protocol": "pairs",
            "protocol_kind": "region_pairs",
            "environment": ENV,
            "settings": {"estimators": ["gc/bic"]},
        }

    def pair_records(self):
        nan = float("nan")
        card_nan = {"auroc": nan, "direction_accuracy": nan, "separation": nan}
        return [
            {
                "type": "task",
                "status": "ok",
                "id": "p0",
                "task": {"scenario": "pair:a-lh:b-lh", "modality": "eeg"},
                "result": {"gc/bic": {"roi": card_nan, "sensor": card_nan, "source": card_nan}},
            }
        ]

    def test_the_edge_metric_tables_are_replaced_by_a_note_for_a_study_without_true_edges(self):
        text = render_summary(self.pair_header(), self.pair_records(), None)
        assert "| mean AUROC |" not in text and "ROI vs sensor" not in text
        assert "no true edge" in text

    def test_the_leakage_section_is_still_there(self):
        assert "## Leakage and crosstalk" in render_summary(
            self.pair_header(), self.pair_records(), None
        )

    def test_ordinary_protocols_keep_their_tables(self):
        text = render_summary(TestRender().header(), TestRender().records(), None)
        assert "| mean AUROC |" in text and "ROI vs sensor" in text


# --- significance studies ---------------------------------------------------------


def sig_data(**overrides) -> dict:
    data = {
        "version": 1,
        "name": "sig",
        "kind": "significance",
        "scenarios": ["chain"],
        "estimators": ["gc/fixed"],
        "tests": ["analytic", "surrogate:phase_randomize"],
        "coupling": [0, 0.4],
        "snr_db": [10, 40],
        "duration_s": [30],
        "n_seeds": 2,
    }
    data.update(overrides)
    return data


def make_sig(**overrides) -> Protocol:
    return Protocol.from_dict(sig_data(**overrides))


class TestSignificanceFormat:
    def test_the_kind_and_the_tests_are_kept(self):
        protocol = make_sig()
        assert protocol.kind == "significance"
        assert protocol.tests == ("analytic", "surrogate:phase_randomize")

    def test_every_other_kind_has_no_tests(self):
        assert make().tests == () and make_pairs().tests == ()

    def test_it_survives_a_round_trip(self):
        protocol = make_sig()
        assert Protocol.from_dict(protocol.to_dict()) == protocol
        assert "tests" not in make().to_dict()  # the key belongs to this kind only

    def test_the_tests_are_required_and_not_empty(self):
        data = sig_data()
        del data["tests"]
        with pytest.raises(ValueError, match="tests"):
            Protocol.from_dict(data)
        with pytest.raises(ValueError, match="tests"):
            make_sig(tests=[])

    def test_an_unknown_test_is_named(self):
        with pytest.raises(ValueError, match="bootstrap"):
            make_sig(tests=["analytic", "bootstrap"])

    def test_a_test_listed_twice_is_refused(self):
        with pytest.raises(ValueError, match="tests"):
            make_sig(tests=["analytic", "analytic"])

    def test_the_other_kinds_refuse_a_tests_key(self):
        with pytest.raises(ValueError, match="tests"):
            Protocol.from_dict(minimal(tests=["analytic"]))
        with pytest.raises(ValueError, match="tests"):
            Protocol.from_dict(pairs_data(tests=["analytic"]))

    @pytest.mark.parametrize(
        ("key", "value"),
        [("modality", ["eeg"]), ("source_model", ["focal"]), ("inverse", ["lcmv"])],
    )
    def test_the_head_model_keys_make_no_sense_at_region_level(self, key, value):
        with pytest.raises(ValueError, match=key):
            make_sig(**{key: value})

    def test_the_analytic_test_needs_estimators_that_have_one(self):
        for label in ("gpdc/fixed", "dtf/bic"):
            with pytest.raises(ValueError, match="analytic"):
                make_sig(estimators=[label])
        make_sig(estimators=["gc/fixed", "gc_pairwise/bic"])
        make_sig(estimators=["pdc/fixed", "pdc/low", "rpdc/bic", "rpdc/low"])  # asymptotic tests
        make_sig(estimators=["gpdc/fixed"], tests=["surrogate:phase_randomize"])

    def test_scenarios_are_required_and_the_coupling_axis_is_checked(self):
        data = sig_data()
        del data["scenarios"]
        with pytest.raises(ValueError, match="scenarios"):
            Protocol.from_dict(data)
        with pytest.raises(ValueError, match="random_5"):
            make_sig(scenarios=["chain", "random_5"])

    def test_pairs_are_refused(self):
        with pytest.raises(ValueError, match="pairs"):
            make_sig(pairs={"n": 2, "seed": 0, "regions": REGIONS})


class TestSignificanceTasks:
    def test_the_count_is_the_product_of_the_axes(self):
        assert (
            len(expand(make_sig())) == 1 * 2 * 2 * 1 * 2
        )  # scenarios x coupling x snr x len x seeds

    def test_a_task_has_no_head_model(self):
        for task in expand(make_sig()):
            assert task.modality == "roi" and task.source_model == "none"
            assert task.n_nodes == 3

    def test_coupling_zero_is_the_null_world(self):
        tasks = expand(make_sig())
        zero = [t for t in tasks if t.coupling == 0.0]
        assert len(zero) == 4 and all(t.zero_network for t in zero)
        assert all(t.scenario_spec().network.edges == () for t in zero)
        coupled = [t for t in tasks if t.coupling == 0.4]
        assert all(len(t.scenario_spec().network.edges) == 2 for t in coupled)

    def test_ids_and_seeds_are_unique_and_survive_a_bigger_grid(self):
        small = {t.task_id: task_seed(0, t) for t in expand(make_sig(n_seeds=2))}
        big = {t.task_id: task_seed(0, t) for t in expand(make_sig(n_seeds=3))}
        assert len(set(big.values())) == len(big) and small.items() <= big.items()

    def test_the_tests_are_part_of_what_a_results_file_is_compatible_with(self, tmp_path):
        path = tmp_path / "sig.jsonl"
        run_protocol(make_sig(), fake_runner, path, environment=ENV)
        assert read_results(path)[0]["protocol_kind"] == "significance"
        assert read_results(path)[0]["settings"]["tests"] == list(make_sig().tests)
        with pytest.raises(ValueError, match=r"tests.*new_file"):
            run_protocol(make_sig(tests=["analytic"]), fake_runner, path, environment=ENV)

    def test_other_kinds_do_not_get_a_tests_entry_in_their_settings(self):
        assert "tests" not in make().numeric_settings()
        assert "tests" not in make_pairs().numeric_settings()


class TestTimeVaryingScenarios:
    """The stage evaluation scores a static truth, so a scenario whose edges switch is refused."""

    def varying(self):
        return [n for n in list_scenarios() if load_scenario(n).network.has_intervals]

    def test_there_are_time_varying_scenarios_and_every_one_is_refused_by_name(self):
        varying = self.varying()
        assert varying  # the scenarios of the non-stationary study
        for name in varying:
            with pytest.raises(ValueError, match=name):
                make(scenarios=[name])
            with pytest.raises(ValueError, match=name):
                make_sig(scenarios=[name])

    def test_the_message_says_why_and_where_to_go(self):
        name = self.varying()[0]
        with pytest.raises(ValueError, match=r"time.*run_nonstationary"):
            make(scenarios=["chain", name])

    def test_every_other_scenario_is_still_accepted(self):
        for name in list_scenarios():
            if name not in self.varying():
                make(scenarios=[name])


# --- the noise axis -----------------------------------------------------------------


def noise_protocol(**overrides) -> Protocol:
    data = {
        "noise": ["simulated", "white", "empty_room"],
        "modality": ["mag", "grad"],
        "scenarios": ["chain"],
        "n_seeds": 3,
    }
    data.update(overrides)
    return make(**data)


class TestNoiseAxis:
    def test_the_default_is_the_simulated_noise_and_it_is_a_task_axis(self):
        assert make().noise == ("simulated",)
        assert len(expand(noise_protocol())) == 3 * 2 * 3  # noises x modalities x seeds

    def test_an_unknown_noise_is_named(self):
        with pytest.raises(ValueError, match="pink"):
            make(noise=["pink"])

    def test_the_empty_room_is_for_meg_only(self):
        with pytest.raises(ValueError, match=r"empty_room.*MEG"):
            make(noise=["empty_room"])  # the default modality is EEG
        with pytest.raises(ValueError, match="eeg"):
            make(noise=["simulated", "empty_room"], modality=["eeg", "grad"])
        make(noise=["empty_room"], modality=["mag", "grad"])

    def test_the_other_kinds_have_no_noise_axis(self):
        with pytest.raises(ValueError, match="noise"):
            Protocol.from_dict(pairs_data(noise=["white"]))
        with pytest.raises(ValueError, match="noise"):
            Protocol.from_dict(sig_data(noise=["white"]))

    def test_it_survives_a_round_trip_and_is_only_written_for_the_kind_that_has_it(self):
        protocol = noise_protocol()
        assert Protocol.from_dict(protocol.to_dict()) == protocol
        assert "noise" not in make_pairs().to_dict() and "noise" not in make_sig().to_dict()

    def test_a_task_names_its_noise_only_when_it_is_not_the_simulated_one(self):
        by_noise = {t.noise: t for t in expand(noise_protocol(n_seeds=1)) if t.modality == "grad"}
        assert "noise" not in by_noise["simulated"].to_dict()
        assert by_noise["white"].to_dict()["noise"] == "white"
        assert by_noise["empty_room"].to_dict()["noise"] == "empty_room"

    def test_existing_task_ids_and_seeds_are_what_they_were_before_the_axis_existed(self):
        # recorded before the axis was added: a results file made then must still resume
        first = expand(make())[:4]
        assert [t.task_id for t in first] == [
            "f9caee2002fb6c7e",
            "7f86072fdbc4dfed",
            "19c2e07ca3b7db6d",
            "9827653f95a5f1fa",
        ]
        assert [task_seed(0, t) for t in first] == [269036648, 1066516056, 1878542253, 122466559]
        (grad,) = expand(
            make(
                scenarios=["chain"],
                n_seeds=1,
                snr_db=[0],
                duration_s=[30],
                modality=["grad"],
                source_model=["focal"],
            )
        )
        assert grad.task_id == "103408f44fe01738" and task_seed(0, grad) == 1690853686

    def test_the_three_arms_of_a_seed_share_the_source_seed_but_not_the_id(self):
        tasks = [t for t in expand(noise_protocol(n_seeds=2)) if t.modality == "grad"]
        for seed_index in (0, 1):
            arms = [t for t in tasks if t.seed_index == seed_index]
            assert len(arms) == 3
            assert len({task_seed(0, t) for t in arms}) == 1
            assert len({t.task_id for t in arms}) == 3
        assert task_seed(0, tasks[0]) != task_seed(0, [t for t in tasks if t.seed_index == 1][0])

    def test_ids_are_unique_over_the_whole_grid(self):
        ids = [t.task_id for t in expand(noise_protocol(snr_db=[10, 0], n_seeds=4))]
        assert len(set(ids)) == len(ids)

    def test_exclude_can_name_the_noise(self):
        tasks = expand(noise_protocol(exclude=[{"noise": "white", "modality": "mag"}]))
        assert len(tasks) == 3 * 2 * 3 - 3
        assert not [t for t in tasks if t.noise == "white" and t.modality == "mag"]

    def test_noise_is_an_axis_of_the_protocol(self):
        from dirconn.protocol import AXES

        assert "noise" in AXES


class TestCodeVersion:
    def test_it_is_the_commit_and_whether_the_tree_was_clean(self):
        info = code_version()
        assert info is not None  # the tests run in the git checkout
        assert len(info["git_commit"]) == 40 and set(info["git_commit"]) <= set("0123456789abcdef")
        assert isinstance(info["git_dirty"], bool)

    def test_an_installed_copy_falls_back_to_the_commit_the_image_was_built_from(self, monkeypatch):
        import dirconn.protocol as protocol

        monkeypatch.setattr(
            protocol.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError)
        )
        monkeypatch.setenv("DIRCONN_GIT_COMMIT", "a" * 40)
        monkeypatch.setenv("DIRCONN_GIT_DIRTY", "0")
        assert code_version() == {"git_commit": "a" * 40, "git_dirty": False}
        monkeypatch.setenv("DIRCONN_GIT_COMMIT", "unknown")
        assert code_version() is None

    def test_the_image_commit_is_not_used_for_another_directory(self, monkeypatch, tmp_path):
        import dirconn.protocol as protocol

        monkeypatch.setattr(
            protocol.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError)
        )
        monkeypatch.setenv("DIRCONN_GIT_COMMIT", "a" * 40)
        assert code_version(tmp_path) is None

    def test_outside_a_repository_it_is_none_not_an_error(self):
        import tempfile

        # the system temporary directory: pytest's own basetemp may sit inside this repository
        with tempfile.TemporaryDirectory() as outside:
            assert code_version(outside) is None

    def test_the_tree_is_dirty_only_when_a_tracked_file_changed(self):
        import subprocess
        import tempfile
        from pathlib import Path

        def git(where, *args):
            subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                cwd=where,
                check=True,
                capture_output=True,
            )

        with tempfile.TemporaryDirectory() as folder:
            where = Path(folder)
            git(where, "init", "-q")
            (where / "a.txt").write_text("one", encoding="utf-8")
            git(where, "add", "a.txt")
            git(where, "commit", "-q", "-m", "first")
            clean = code_version(where)
            assert clean is not None and clean["git_dirty"] is False
            (where / "untracked.txt").write_text("x", encoding="utf-8")
            assert code_version(where)["git_dirty"] is False  # untracked files do not count
            (where / "a.txt").write_text("two", encoding="utf-8")
            dirty = code_version(where)
            assert dirty["git_dirty"] is True and dirty["git_commit"] == clean["git_commit"]
            git(where, "commit", "-q", "-am", "second")
            assert code_version(where)["git_commit"] != clean["git_commit"]

    def test_a_results_file_header_records_it_and_a_resume_does_not_depend_on_it(self, tmp_path):
        path = tmp_path / "demo.jsonl"
        run_protocol(make(), fake_runner, path, environment=ENV)
        assert read_results(path)[0]["code"] == code_version()
        # the commit changes with every commit of the project: that must never block a resume
        header, records, _ = read_results(path)
        header["code"] = {"git_commit": "0" * 40, "git_dirty": True}
        lines = [json.dumps(header)] + [json.dumps(r) for r in records]
        path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
        counting = Counting()
        run_protocol(make(), counting, path, environment=ENV)
        assert counting.calls == []

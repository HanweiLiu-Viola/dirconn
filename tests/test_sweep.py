"""The real task runner of the benchmark: estimators, supported scenarios, JSON-safe payloads."""

import json

import numpy as np
import pytest

from dirconn import DTF, GC, GPDC, PDC, RPDC
from dirconn.protocol import (
    CALIBRATION_ESTIMATOR_NAMES,
    EXTENDED_ESTIMATOR_NAMES,
    VAR_ESTIMATOR_NAMES,
    Protocol,
    expand,
)
from dirconn.sweep import (
    DEFAULT_SETTINGS,
    estimator_for,
    jsonable,
    settings_for,
)


def protocol(**overrides) -> Protocol:
    data = {
        "version": 1,
        "name": "demo",
        "scenarios": ["chain"],
        "estimators": ["gc/bic"],
        "n_seeds": 1,
    }
    data.update(overrides)
    return Protocol.from_dict(data)


def build(label, *, sfreq=100.0, n_nodes=3, n_times=6000, **settings):
    return estimator_for(
        label, sfreq=sfreq, n_nodes=n_nodes, n_times=n_times, settings=settings_for(settings)
    )


class TestSettings:
    def test_defaults_are_filled_in_and_unknown_settings_are_refused(self):
        assert settings_for({}) == DEFAULT_SETTINGS
        assert settings_for({"max_order": 12})["max_order"] == 12
        assert DEFAULT_SETTINGS["store_scores"] is False  # the files stay small
        with pytest.raises(ValueError, match="max_ordr"):
            settings_for({"max_ordr": 12})

    def test_the_settings_a_protocol_may_carry(self):
        assert set(DEFAULT_SETTINGS) == {
            "max_order",
            "fixed_order_seconds",
            "band_low_hz",
            "band_high_hz",
            "store_scores",
            "n_surrogates",
            "low_order",
            "n_freqs",
        }


class TestEstimatorFor:
    @pytest.mark.parametrize(
        ("method", "cls"),
        [("gc", GC), ("gc_pairwise", GC), ("dtf", DTF), ("pdc", PDC), ("gpdc", GPDC)],
    )
    def test_the_method_is_the_one_named(self, method, cls):
        assert type(build(f"{method}/bic")) is cls

    def test_pairwise_gc_is_not_conditional(self):
        assert build("gc/bic").conditional is True
        assert build("gc_pairwise/bic").conditional is False

    def test_the_fixed_order_is_about_a_fifth_of_a_second(self):
        assert build("gc/fixed", sfreq=100.0).order == 20
        assert build("gc/fixed", sfreq=250.0).order == 50
        assert build("gc/fixed", sfreq=100.0, fixed_order_seconds=0.1).order == 10

    def test_bic_is_the_criterion_and_its_cap_follows_the_sampling_rule(self):
        # at most one coefficient per ten observations: n_nodes * order <= n_times / 10
        est = build("dtf/bic", n_nodes=20, n_times=6000)
        assert est.order == "bic"
        assert est.max_order == 30
        assert build("dtf/bic", n_nodes=3, n_times=6000).max_order == 40  # the setting caps it
        assert build("dtf/bic", n_nodes=20, n_times=6000, max_order=10).max_order == 10

    def test_the_cap_never_drops_below_one(self):
        assert build("gc/bic", n_nodes=20, n_times=100).max_order == 1

    def test_the_calibration_labels_build_pdc_and_rpdc(self):
        for label in CALIBRATION_ESTIMATOR_NAMES:
            assert type(build(label)) is (RPDC if label.startswith("rpdc") else PDC), label
        assert build("rpdc/bic").order == "bic"
        assert build("rpdc/fixed", sfreq=100.0).order == 20

    def test_the_too_low_order_is_a_number_of_samples_two_by_default(self):
        # two is the order of the chain without observation noise
        assert build("pdc/low").order == build("rpdc/low", sfreq=250.0).order == 2
        assert build("rpdc/low", low_order=1).order == 1

    def test_the_frequency_grid_of_the_spectral_estimators_can_be_set(self):
        assert build("pdc/fixed").n_freqs == PDC().n_freqs  # the estimator's default
        assert build("pdc/fixed", n_freqs=51).n_freqs == 51
        assert build("rpdc/low", n_freqs=51).n_freqs == 51
        assert type(build("gc/fixed", n_freqs=51)) is GC  # no frequency axis, nothing to set

    @pytest.mark.parametrize(
        "label", [*VAR_ESTIMATOR_NAMES, *CALIBRATION_ESTIMATOR_NAMES, "trgc/bic"]
    )
    def test_the_order_stability_check_is_off_in_sweeps(self, label):
        assert build(label).check_order_stability is False

    def test_the_further_labels_build_their_estimators(self):
        from dirconn import MTGC, PCMCI, PSI, TE, TRGC, ImCoh
        from dirconn.sweep import MULTITAPER_SEGMENT_SECONDS

        expected = {
            "trgc/bic": TRGC,
            "imcoh/mt": ImCoh,
            "psi/mt": PSI,
            "mtgc/mt": MTGC,
            "pcmci/bic": PCMCI,
            "te/bic": TE,
        }
        assert set(expected) == set(EXTENDED_ESTIMATOR_NAMES)
        for label, cls in expected.items():
            assert type(build(label)) is cls, label
        assert build("trgc/bic").order == "bic"
        assert build("pcmci/bic").tau_max == "bic"
        assert build("te/bic").target_history == "bic"
        for label in ("imcoh/mt", "psi/mt", "mtgc/mt"):
            assert build(label).segment_length == MULTITAPER_SEGMENT_SECONDS
        assert build("psi/mt").bands == [(1.0, 45.0)]
        assert build("pcmci/bic", n_nodes=20, n_times=100).max_order == 1  # the same cap

    def test_an_unknown_label_is_named(self):
        with pytest.raises(ValueError, match="gc/slow"):
            build("gc/slow")


class TestJsonable:
    def test_numpy_values_become_plain_python(self):
        value = {
            "a": np.float64(0.5),
            "b": np.int64(3),
            "c": np.array([1, 2]),
            "d": (np.float32(1.5), "x"),
            "e": np.bool_(True),
            "f": float("nan"),
        }
        out = jsonable(value)
        assert out["a"] == 0.5 and type(out["a"]) is float
        assert out["b"] == 3 and type(out["b"]) is int
        assert out["c"] == [1, 2]
        assert out["d"] == [1.5, "x"]
        assert out["e"] is True
        assert np.isnan(out["f"])
        json.dumps(out)  # must not raise

    def test_unknown_objects_are_an_error_not_a_string(self):
        with pytest.raises(TypeError, match="object"):
            jsonable({"x": object()})


class TestGateWarmUp:
    def test_every_head_is_loaded_before_the_first_task_is_timed(self, monkeypatch):
        from dirconn import sweep

        events = []
        monkeypatch.setattr(sweep, "_head", lambda modality: events.append(("load", modality)))
        monkeypatch.setattr(
            sweep, "run_task", lambda task, proto: events.append(("run", task.modality)) or {}
        )
        proto = protocol(scenarios=["chain", "random_5"], modality=["eeg", "grad"])
        sweep.measure_gate(proto, workers=1, scenarios=None)
        first_run = next(i for i, (kind, _) in enumerate(events) if kind == "run")
        loaded = {m for kind, m in events[:first_run] if kind == "load"}
        assert loaded == {"eeg", "grad"}

    def test_one_task_is_timed_per_network_size_and_modality(self, monkeypatch):
        from dirconn import sweep

        timed = []
        monkeypatch.setattr(sweep, "_head", lambda modality: None)
        monkeypatch.setattr(
            sweep,
            "run_task",
            lambda task, proto: timed.append((task.n_nodes, task.modality)) or {},
        )
        proto = protocol(scenarios=["chain", "collider", "random_5"], modality=["eeg", "grad"])
        gate = sweep.measure_gate(proto, workers=1)
        assert sorted(timed) == [
            (3, "eeg"),
            (3, "grad"),
            (4, "eeg"),
            (4, "grad"),
            (5, "eeg"),
            (5, "grad"),
        ]
        assert set(gate["seconds_per_task"]) == {"3", "4", "5"}


class TestGateCalibration:
    """The gate measures the parallel efficiency and the startup time instead of assuming them."""

    def stub(self, monkeypatch, probe):
        """Stand-ins for the head, the task and the probe, with a clock that only the task moves.

        A task "takes" exactly 0.02 s of this fake clock, so nothing here depends on how fast the
        machine is (a real ``sleep`` made this test fail on the macOS runners of CI).
        """
        import types

        from dirconn import sweep

        clock = [0.0]
        monkeypatch.setattr(sweep, "time", types.SimpleNamespace(perf_counter=lambda: clock[0]))
        monkeypatch.setattr(sweep, "_head", lambda modality: None)

        def fake_task(task, proto):
            clock[0] += 0.02
            return {}

        monkeypatch.setattr(sweep, "run_task", fake_task)
        self.probe_calls = []

        def fake_probe(task, proto, workers, modalities):
            self.probe_calls.append((workers, tuple(modalities)))
            return probe

        monkeypatch.setattr(sweep, "_parallel_probe", fake_probe)
        return sweep

    def test_with_several_workers_the_efficiency_is_serial_time_over_batch_time(self, monkeypatch):
        sweep = self.stub(monkeypatch, probe=(5.0, 0.08))  # a batch takes 4x one task
        gate = sweep.measure_gate(protocol(scenarios=["chain"]), workers=4)
        assert self.probe_calls == [(4, ("eeg",))]
        assert gate["efficiency"] == pytest.approx(0.25)  # 0.02 s alone, a batch of 0.08 s
        assert gate["startup_seconds"] == 5.0
        assert gate["probe"]["batch_seconds"] == 0.08

    def test_the_efficiency_is_never_above_one_or_below_a_hundredth(self, monkeypatch):
        sweep = self.stub(monkeypatch, probe=(0.0, 1e-9))
        assert sweep.measure_gate(protocol(scenarios=["chain"]), workers=4)["efficiency"] == 1.0
        sweep = self.stub(monkeypatch, probe=(0.0, 1e6))
        assert sweep.measure_gate(protocol(scenarios=["chain"]), workers=4)["efficiency"] == 0.01

    def test_one_worker_needs_no_probe_and_has_efficiency_one(self, monkeypatch):
        sweep = self.stub(monkeypatch, probe=(5.0, 0.08))
        gate = sweep.measure_gate(protocol(scenarios=["chain"]), workers=1)
        assert self.probe_calls == []
        assert gate["efficiency"] == 1.0
        assert gate["startup_seconds"] >= 0.0

    def test_a_given_efficiency_is_used_and_nothing_is_probed(self, monkeypatch):
        sweep = self.stub(monkeypatch, probe=(5.0, 0.08))
        gate = sweep.measure_gate(protocol(scenarios=["chain"]), workers=4, efficiency=0.5)
        assert self.probe_calls == []
        assert gate["efficiency"] == 0.5

    def test_the_estimate_uses_the_measured_efficiency_and_startup(self, monkeypatch):
        sweep = self.stub(monkeypatch, probe=(3600.0, 0.08))
        gate = sweep.measure_gate(protocol(scenarios=["chain"]), workers=4)
        # the single task is divided by workers * efficiency; the startup adds an hour
        assert gate["estimate_hours"] >= 1.0

    def test_the_probe_runs_the_heaviest_task_on_every_modality_in_play(self, monkeypatch):
        sweep = self.stub(monkeypatch, probe=(0.0, 0.05))
        sweep.measure_gate(
            protocol(scenarios=["chain", "random_5"], modality=["eeg", "grad"]), workers=2
        )
        assert self.probe_calls == [(2, ("eeg", "grad"))]


@pytest.mark.sample_data
class TestOnTheSampleHead:
    def tiny(self, **overrides):
        return protocol(
            estimators=["gc/bic", "dtf/fixed"],
            duration_s=[20],
            snr_db=[10],
            **overrides,
        )

    def test_one_task_gives_a_json_safe_card_per_estimator_and_stage(self):
        from dirconn.sweep import run_task

        proto = self.tiny()
        (task,) = expand(proto)
        payload = run_task(task, proto)
        assert set(payload) == {"gc/bic", "dtf/fixed"}
        for label in payload:
            assert {"roi", "sensor", "source"} <= set(payload[label])
            assert 0.0 <= payload[label]["roi"]["auroc"] <= 1.0
        json.loads(json.dumps(payload))

    def test_a_task_is_reproducible(self):
        from dirconn.sweep import run_task

        proto = self.tiny()
        (task,) = expand(proto)
        assert run_task(task, proto) == run_task(task, proto)

    def test_a_zero_network_gives_cards_without_a_crash(self):
        from dirconn.sweep import run_task

        proto = self.tiny(coupling=[0])
        (task,) = expand(proto)
        payload = run_task(task, proto)
        json.loads(json.dumps(payload))
        assert np.isnan(payload["gc/bic"]["roi"]["auroc"])

    def test_a_zero_network_does_not_flood_the_run_with_undefined_metric_warnings(self):
        import warnings

        from dirconn import DirconnWarning
        from dirconn.sweep import run_task

        proto = self.tiny(coupling=[0])
        (task,) = expand(proto)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_task(task, proto)
        assert not [w for w in caught if issubclass(w.category, DirconnWarning)]

    def test_a_scenario_with_hidden_nodes_is_run_with_them_hidden(self):
        from dirconn.sweep import run_task

        proto = protocol(scenarios=["hidden_common_input"], duration_s=[30], snr_db=[10])
        (task,) = expand(proto)
        payload = run_task(task, proto)
        card = payload["gc/bic"]
        analysed = card["analysed"]
        assert len(analysed) == 3 and len(card["hidden"]) == 1
        assert card["hidden"][0] not in analysed
        assert card["agreement"]["n_pairs"] == 6

    def test_a_scenario_with_an_artifact_is_run_with_it_at_the_sensors_only(self):
        import dataclasses

        from dirconn.scenario import load_scenario
        from dirconn.sweep import run_task

        proto = protocol(scenarios=["stimulation_artifact"], duration_s=[30], snr_db=[10])
        clean = dataclasses.replace(load_scenario("stimulation_artifact"), artifact=None)
        (with_artifact,) = expand(proto)
        (without,) = expand(proto, scenarios={"stimulation_artifact": clean})
        a = run_task(with_artifact, proto)["gc/bic"]
        b = run_task(without, proto)["gc/bic"]
        assert a["roi"] == b["roi"]  # the region level never sees it
        assert a["sensor"]["separation"] != b["sensor"]["separation"]  # the sensors do
        json.dumps(a)

    def test_spectral_cards_carry_the_curve_over_frequency(self):
        from dirconn.sweep import run_task

        proto = protocol(estimators=["gpdc/bic", "gc/bic"], duration_s=[30], snr_db=[10])
        (task,) = expand(proto)
        payload = run_task(task, proto)
        for stage in ("roi", "sensor", "source"):
            curve = payload["gpdc/bic"][stage]["auroc_by_frequency"]
            assert len(curve["freqs"]) == len(curve["auroc"]) > 10
            assert "auroc_by_frequency" not in payload["gc/bic"][stage]

    def test_several_inverse_methods_give_one_entry_each(self):
        from dirconn.sweep import run_task

        proto = self.tiny(inverse=["mne", "lcmv"])
        (task,) = expand(proto)
        payload = run_task(task, proto)
        assert set(payload) == {"gc/bic@mne", "gc/bic@lcmv", "dtf/fixed@mne", "dtf/fixed@lcmv"}

    def test_the_gate_times_one_task_per_network_size(self):
        from dirconn.sweep import measure_gate

        proto = self.tiny(scenarios=["chain", "random_5"])
        gate = measure_gate(proto, workers=1)
        assert set(gate["seconds_per_task"]) == {"3", "5"}
        assert all(v > 0 for v in gate["seconds_per_task"].values())
        assert gate["estimate_hours"] > 0
        assert gate["reference_duration_s"] == 120.0
        json.dumps(gate)


# --- significance studies: the runner at region level -----------------------------


def sig_protocol(**overrides):
    data = {
        "version": 1,
        "name": "sig",
        "kind": "significance",
        "scenarios": ["chain"],
        "estimators": ["gc/fixed"],
        "tests": ["analytic", "surrogate:phase_randomize"],
        "coupling": [0.4],
        "snr_db": [40],
        "duration_s": [20],
        "n_seeds": 1,
        "settings": {"n_surrogates": 9},
    }
    data.update(overrides)
    return Protocol.from_dict(data)


class TestSignificanceRunner:
    @pytest.fixture(autouse=True)
    def no_head(self, monkeypatch):
        from dirconn import sweep

        def boom(modality):
            raise AssertionError(f"a region-level study loaded the head model {modality!r}")

        monkeypatch.setattr(sweep, "_head", boom)

    def test_it_returns_the_p_value_matrix_of_every_test(self):
        from dirconn.sweep import run_task

        proto = sig_protocol()
        (task,) = expand(proto)
        payload = run_task(task, proto)
        assert set(payload) == {"gc/fixed"}
        card = payload["gc/fixed"]
        assert len(card["analysed"]) == 3
        assert set(card["tests"]) == {"analytic", "surrogate:phase_randomize"}
        for result in card["tests"].values():
            p = np.array(result["pvalues"])
            assert p.shape == (3, 3) and np.isnan(np.diag(p)).all()
            off = p[~np.eye(3, dtype=bool)]
            assert ((off >= 0) & (off <= 1)).all()
        assert "freqs" not in card  # one p-value per cell, as in the published studies
        json.dumps(payload)

    def test_a_spectral_estimator_keeps_the_p_values_of_every_frequency(self):
        from dirconn.sweep import run_task

        proto = sig_protocol(estimators=["pdc/fixed"], settings={"n_surrogates": 9, "n_freqs": 11})
        (task,) = expand(proto)
        card = run_task(task, proto)["pdc/fixed"]
        np.testing.assert_allclose(card["freqs"], np.linspace(0.0, 50.0, 11))
        for result in card["tests"].values():
            p = np.array(result["pvalues"])
            assert p.shape == (3, 3, 11)
            assert np.isnan(p[np.eye(3, dtype=bool)]).all()
            off = p[~np.eye(3, dtype=bool)]
            assert ((off >= 0) & (off <= 1)).all()

    def test_rpdc_finds_a_strong_edge_at_every_frequency_with_both_tests(self):
        from dirconn.sweep import run_task

        proto = sig_protocol(
            estimators=["rpdc/low"], duration_s=[30], settings={"n_surrogates": 9, "n_freqs": 11}
        )
        (task,) = expand(proto)
        card = run_task(task, proto)["rpdc/low"]["tests"]
        assert (np.array(card["analytic"]["pvalues"])[0, 1] < 0.001).all()  # A -> B
        assert (np.array(card["surrogate:phase_randomize"]["pvalues"])[0, 1] <= 0.1).all()

    def test_a_strong_edge_is_found_by_both_tests(self):
        from dirconn.sweep import run_task

        proto = sig_protocol(duration_s=[30])
        (task,) = expand(proto)
        card = run_task(task, proto)["gc/fixed"]["tests"]
        assert card["analytic"]["pvalues"][0][1] < 0.001  # A -> B, coupling 0.4, nearly clean
        assert card["surrogate:phase_randomize"]["pvalues"][0][1] <= 0.1  # the smallest of 9 + 1

    def test_the_surrogate_p_values_are_multiples_of_one_over_n_plus_one(self):
        from dirconn.sweep import run_task

        for n in (9, 19):
            proto = sig_protocol(settings={"n_surrogates": n})
            (task,) = expand(proto)
            p = np.array(
                run_task(task, proto)["gc/fixed"]["tests"]["surrogate:phase_randomize"]["pvalues"]
            )
            off = p[~np.eye(3, dtype=bool)] * (n + 1)
            np.testing.assert_allclose(off, np.round(off), atol=1e-4)
            assert off.min() >= 1 - 1e-4

    def test_it_is_reproducible_and_another_task_gives_other_data(self):
        from dirconn.sweep import run_task

        proto = sig_protocol(n_seeds=2)
        a, b = expand(proto)

        # NaN != NaN (the diagonal), so compare the serialised form
        def dump(task):
            return json.dumps(run_task(task, proto))

        assert dump(a) == dump(a)
        assert dump(a) != dump(b)

    def test_it_runs_the_null_world_and_keeps_the_estimator_label(self):
        from dirconn.sweep import run_task

        proto = sig_protocol(coupling=[0], estimators=["gc/fixed", "gc_pairwise/fixed"])
        (task,) = expand(proto)
        assert set(run_task(task, proto)) == {"gc/fixed", "gc_pairwise/fixed"}

    def test_it_does_not_let_diagnostic_warnings_through(self):
        import warnings

        from dirconn import DirconnWarning
        from dirconn.sweep import run_task

        proto = sig_protocol(duration_s=[5])  # short enough for the sampling check to warn
        (task,) = expand(proto)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_task(task, proto)
        assert not [w for w in caught if issubclass(w.category, DirconnWarning)]

    def test_the_gate_of_a_region_level_study_loads_no_head_model(self, monkeypatch):
        from dirconn import sweep

        monkeypatch.setattr(sweep, "run_task", lambda task, proto: {})
        gate = sweep.measure_gate(sig_protocol(), workers=1)
        assert set(gate["seconds_per_task"]) == {"3"} and gate["startup_seconds"] >= 0.0


def test_the_surrogate_names_of_the_protocol_are_those_of_significance_test():
    from dirconn.protocol import SURROGATE_NAMES
    from dirconn.significance import _NAMED_SURROGATES

    assert sorted(SURROGATE_NAMES) == sorted(_NAMED_SURROGATES)


def noise_protocol(**overrides):
    data = {
        "version": 1,
        "name": "noise",
        "scenarios": ["chain"],
        "estimators": ["gc/bic"],
        "noise": ["simulated", "white", "empty_room"],
        "modality": ["grad"],
        "snr_db": [-10],
        "duration_s": [30],
        "n_seeds": 1,
    }
    data.update(overrides)
    return Protocol.from_dict(data)


@pytest.mark.sample_data
class TestNoiseArmsOnTheSampleHead:
    def spy(self, monkeypatch):
        """Record what run_task hands to the simulation, then simulate as usual."""
        from dirconn import sweep

        calls = []
        real = sweep.simulate_on_head

        def wrapper(*args, **kwargs):
            calls.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(sweep, "simulate_on_head", wrapper)
        return calls

    def test_each_arm_hands_the_simulation_its_own_noise_settings(self, monkeypatch):
        from dirconn.emptyroom import empty_room_window
        from dirconn.headmodel import load_head_model
        from dirconn.sweep import run_task

        calls = self.spy(monkeypatch)
        proto = noise_protocol()
        for task in expand(proto):
            run_task(task, proto)
        by_noise = {t.noise: c for t, c in zip(expand(proto), calls, strict=True)}
        assert by_noise["simulated"]["n_background"] == 10
        assert by_noise["simulated"].get("noise_series") is None
        assert by_noise["white"]["n_background"] == 0
        assert by_noise["white"].get("noise_series") is None
        assert by_noise["empty_room"]["n_background"] == 0
        window = empty_room_window(load_head_model("grad"), 0, 3000)
        np.testing.assert_array_equal(by_noise["empty_room"]["noise_series"], window)

    def test_the_arms_of_a_seed_have_the_same_region_level_result_and_different_sensor_ones(self):
        from dirconn.sweep import run_task

        proto = noise_protocol()
        cards = {t.noise: run_task(t, proto)["gc/bic"] for t in expand(proto)}
        assert cards["simulated"]["roi"] == cards["white"]["roi"] == cards["empty_room"]["roi"]
        seps = {k: v["sensor"]["separation"] for k, v in cards.items()}
        assert len(set(seps.values())) == 3  # the sensors do see the noise
        json.dumps(cards)

    def test_the_window_follows_the_seed_index_and_wraps_around_the_three_windows(
        self, monkeypatch
    ):
        from dirconn.sweep import run_task

        calls = self.spy(monkeypatch)
        proto = noise_protocol(noise=["empty_room"], n_seeds=4)
        for task in expand(proto):
            run_task(task, proto)
        windows = [c["noise_series"] for c in calls]
        np.testing.assert_array_equal(windows[0], windows[3])  # seed 3 is window 0 again
        assert not np.array_equal(windows[0], windows[1])
        assert not np.array_equal(windows[1], windows[2])

    def test_a_duration_longer_than_the_recording_is_an_error_not_a_short_noise(self):
        from dirconn.sweep import run_task

        proto = noise_protocol(noise=["empty_room"], duration_s=[120])
        (task,) = expand(proto)
        with pytest.raises(ValueError, match="shorter"):
            run_task(task, proto)


PAIR_REGIONS = ["cuneus-lh", "pericalcarine-lh", "insula-rh"]


def pair_protocol(**overrides):
    data = {
        "version": 1,
        "name": "pairs",
        "kind": "region_pairs",
        "pairs": {"n": 1, "seed": 0, "regions": PAIR_REGIONS},
        "estimators": ["gc/bic"],
        "n_seeds": 1,
        "duration_s": [30],
        "settings": {"store_scores": True},
    }
    data.update(overrides)
    return Protocol.from_dict(data)


@pytest.mark.sample_data
class TestScoresAndPairsOnTheSampleHead:
    def test_the_setting_puts_the_score_matrices_into_every_stage_card(self):
        from dirconn.sweep import run_task

        proto = pair_protocol()
        (task,) = expand(proto)
        card = run_task(task, proto)["gc/bic"]
        for stage in ("roi", "sensor", "source"):
            matrix = np.array(card[stage]["scores"])
            assert matrix.shape == (2, 2) and np.isnan(np.diag(matrix)).all()
        assert len(card["analysed"]) == 2
        json.dumps(card)

    def test_without_the_setting_there_are_no_matrices(self):
        from dirconn.sweep import run_task

        proto = pair_protocol(settings={})
        (task,) = expand(proto)
        assert "scores" not in run_task(task, proto)["gc/bic"]["roi"]

    def test_a_pair_task_runs_without_warning_about_undefined_metrics(self):
        import warnings

        from dirconn import DirconnWarning
        from dirconn.sweep import run_task

        proto = pair_protocol()
        (task,) = expand(proto)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_task(task, proto)
        assert not [w for w in caught if issubclass(w.category, DirconnWarning)]

    def test_the_crosstalk_matrices_are_by_modality_with_the_names_of_the_head(self):
        from dirconn.headmodel import load_head_model
        from dirconn.sweep import crosstalk_matrices

        out = crosstalk_matrices(["eeg"])
        head = load_head_model("eeg")
        assert list(out) == ["eeg"]
        assert out["eeg"]["names"] == list(head.roi_names)
        matrix = np.array(out["eeg"]["matrix"])
        assert matrix.shape == (head.n_rois, head.n_rois)
        np.testing.assert_allclose(np.diag(matrix), 1.0)  # a region leaks fully into itself
        json.dumps(out)

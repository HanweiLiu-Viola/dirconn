import copy

import numpy as np
import pytest
import yaml

from dirconn.network import Edge, Network, dk_node
from dirconn.scenario import (
    SCENARIO_DIR,
    SCENARIO_VERSION,
    ArtifactSpec,
    ScenarioSpec,
    artifact_pattern,
    artifact_waveform,
    list_scenarios,
    load_scenario,
    stimulation_artifact,
)
from dirconn.simulation import NodeDynamics

A, B, C = "bankssts-lh", "superiorparietal-lh", "precuneus-rh"


def minimal() -> dict:
    return {
        "version": 1,
        "name": "demo",
        "description": "A -> B.",
        "network": {
            "nodes": [{"name": A}, {"name": B}, {"name": C}],
            "edges": [{"source": A, "target": B, "strength": 0.4, "delay_ms": 10.0}],
        },
    }


def write(tmp_path, data, name="demo.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


class TestFromDict:
    def test_minimal_file_gets_the_documented_defaults(self):
        spec = ScenarioSpec.from_dict(minimal())
        assert spec.name == "demo"
        assert spec.sfreq == 100.0
        assert spec.snr_db == 10.0
        assert spec.duration_s == 120.0
        assert spec.n_background == 10
        assert spec.background_fraction == 0.5
        assert spec.hidden == ()
        assert spec.dynamics == {A: NodeDynamics(), B: NodeDynamics(), C: NodeDynamics()}
        assert spec.n_times == 12000

    def test_the_network_is_the_one_a_network_file_would_give(self):
        spec = ScenarioSpec.from_dict(minimal())
        expected = Network(
            nodes=(dk_node(A), dk_node(B), dk_node(C)),
            edges=(Edge(A, B, 0.4, 10.0),),
            name="demo",
        )
        assert spec.network == expected

    def test_dynamics_default_and_per_node_overrides(self):
        data = minimal()
        data["dynamics"] = {
            "default": {"kind": "oscillator", "freq_hz": 12.0, "bandwidth_hz": 4.0},
            "nodes": {B: {"kind": "white"}, C: {"freq_hz": 20.0}},
        }
        spec = ScenarioSpec.from_dict(data)
        assert spec.dynamics[A] == NodeDynamics(freq_hz=12.0, bandwidth_hz=4.0)
        # an override replaces the default as a whole, it is not merged with it
        assert spec.dynamics[B] == NodeDynamics(kind="white")
        assert spec.dynamics[C] == NodeDynamics(freq_hz=20.0)

    def test_defaults_and_background_blocks(self):
        data = minimal()
        data["defaults"] = {"sfreq": 200.0, "snr_db": 0.0, "duration_s": 60.0}
        data["background"] = {"n_sources": 0, "fraction": 0.25}
        spec = ScenarioSpec.from_dict(data)
        assert (spec.sfreq, spec.snr_db, spec.duration_s) == (200.0, 0.0, 60.0)
        assert (spec.n_background, spec.background_fraction) == (0, 0.25)
        assert spec.n_times == 12000

    def test_hidden_nodes_are_listed_and_the_rest_is_analysed(self):
        data = minimal()
        data["hidden"] = [C]
        spec = ScenarioSpec.from_dict(data)
        assert spec.hidden == (C,)
        assert spec.analysed == (A, B)

    def test_the_input_is_not_modified(self):
        data = minimal()
        before = copy.deepcopy(data)
        ScenarioSpec.from_dict(data)
        assert data == before

    def test_the_spec_is_immutable(self):
        spec = ScenarioSpec.from_dict(minimal())
        with pytest.raises(AttributeError):
            spec.snr_db = 0.0  # type: ignore[misc]


class TestRoundTrip:
    def test_to_dict_restores_the_spec(self):
        data = minimal()
        data["dynamics"] = {"default": {"freq_hz": 12.0}, "nodes": {B: {"kind": "white"}}}
        data["defaults"] = {"snr_db": -5.0}
        data["hidden"] = [C]
        spec = ScenarioSpec.from_dict(data)
        assert ScenarioSpec.from_dict(spec.to_dict()) == spec

    def test_to_dict_carries_the_version(self):
        assert ScenarioSpec.from_dict(minimal()).to_dict()["version"] == SCENARIO_VERSION


class TestBadFiles:
    def test_a_missing_version_is_refused(self):
        data = minimal()
        del data["version"]
        with pytest.raises(ValueError, match="version"):
            ScenarioSpec.from_dict(data)

    def test_an_unsupported_version_is_refused_with_the_supported_one(self):
        data = minimal()
        data["version"] = 2
        with pytest.raises(ValueError, match=r"version.*2.*1"):
            ScenarioSpec.from_dict(data)

    @pytest.mark.parametrize("key", ["name", "network"])
    def test_a_missing_required_key_is_named(self, key):
        data = minimal()
        del data[key]
        with pytest.raises(ValueError, match=key):
            ScenarioSpec.from_dict(data)

    def test_an_unknown_top_level_key_is_named(self):
        data = minimal()
        data["snr"] = 10
        with pytest.raises(ValueError, match="snr"):
            ScenarioSpec.from_dict(data)

    @pytest.mark.parametrize(
        ("block", "key"),
        [
            ("defaults", "duration"),
            ("background", "count"),
            ("dynamics", "defualt"),
        ],
    )
    def test_an_unknown_key_inside_a_block_is_named(self, block, key):
        data = minimal()
        data[block] = {key: 1}
        with pytest.raises(ValueError, match=key):
            ScenarioSpec.from_dict(data)

    def test_an_unknown_dynamics_field_is_named(self):
        data = minimal()
        data["dynamics"] = {"default": {"frequency": 10.0}}
        with pytest.raises(ValueError, match="frequency"):
            ScenarioSpec.from_dict(data)

    def test_an_edge_to_an_unknown_node_is_named(self):
        data = minimal()
        data["network"]["edges"].append({"source": A, "target": "insula-rh", "strength": 0.1})
        with pytest.raises(ValueError, match="insula-rh"):
            ScenarioSpec.from_dict(data)

    def test_dynamics_for_an_unknown_node_is_named(self):
        data = minimal()
        data["dynamics"] = {"nodes": {"insula-rh": {"kind": "white"}}}
        with pytest.raises(ValueError, match="insula-rh"):
            ScenarioSpec.from_dict(data)

    def test_a_hidden_unknown_node_is_named(self):
        data = minimal()
        data["hidden"] = ["insula-rh"]
        with pytest.raises(ValueError, match="insula-rh"):
            ScenarioSpec.from_dict(data)

    def test_hiding_all_but_one_node_is_refused(self):
        data = minimal()
        data["hidden"] = [A, B]
        with pytest.raises(ValueError, match="hidden"):
            ScenarioSpec.from_dict(data)

    def test_a_band_above_nyquist_names_the_edge_and_the_limit(self):
        data = minimal()
        data["network"]["edges"][0]["band"] = [30.0, 60.0]
        with pytest.raises(ValueError, match=r"band.*60.*Nyquist.*50"):
            ScenarioSpec.from_dict(data)

    def test_a_band_that_fits_a_higher_sampling_rate_is_accepted(self):
        data = minimal()
        data["network"]["edges"][0]["band"] = [30.0, 60.0]
        data["network"]["edges"][0]["delay_ms"] = 8.0  # 2 samples at 250 Hz
        data["defaults"] = {"sfreq": 250.0}
        assert ScenarioSpec.from_dict(data).network.edges[0].band == (30.0, 60.0)

    def test_an_oscillator_above_nyquist_names_the_node(self):
        data = minimal()
        data["dynamics"] = {"nodes": {B: {"freq_hz": 70.0}}}
        with pytest.raises(ValueError, match=rf"{B}.*Nyquist"):
            ScenarioSpec.from_dict(data)

    def test_a_delay_that_is_not_a_whole_number_of_samples_is_refused(self):
        data = minimal()
        data["network"]["edges"][0]["delay_ms"] = 12.5  # 1.25 samples at 100 Hz
        with pytest.raises(ValueError, match="delay"):
            ScenarioSpec.from_dict(data)

    @pytest.mark.parametrize(
        ("block", "key", "value"),
        [
            ("defaults", "sfreq", 0.0),
            ("defaults", "duration_s", -1.0),
            ("defaults", "snr_db", float("nan")),
            ("background", "n_sources", -1),
            ("background", "fraction", 1.5),
        ],
    )
    def test_out_of_range_numbers_name_their_key(self, block, key, value):
        data = minimal()
        data[block] = {key: value}
        with pytest.raises(ValueError, match=key):
            ScenarioSpec.from_dict(data)

    def test_a_duration_shorter_than_one_sample_is_refused(self):
        data = minimal()
        data["defaults"] = {"duration_s": 0.001}
        with pytest.raises(ValueError, match="duration_s"):
            ScenarioSpec.from_dict(data)

    def test_an_unstable_network_is_refused(self):
        data = minimal()
        data["network"]["edges"] = [
            {"source": A, "target": B, "strength": 0.9, "delay_ms": 10.0},
            {"source": B, "target": A, "strength": 0.9, "delay_ms": 10.0},
        ]
        data["dynamics"] = {"default": {"freq_hz": 10.0, "bandwidth_hz": 0.5}}
        with pytest.raises(ValueError, match="unstable|spectral radius"):
            ScenarioSpec.from_dict(data)


class TestFiles:
    def test_load_from_a_path(self, tmp_path):
        path = write(tmp_path, minimal())
        assert load_scenario(path) == ScenarioSpec.from_dict(minimal())

    def test_a_missing_file_is_a_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_scenario(tmp_path / "nope.yaml")

    def test_a_non_mapping_top_level_is_refused(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_scenario(path)

    def test_an_unknown_shipped_name_lists_the_known_ones(self):
        with pytest.raises(ValueError, match="chain"):
            load_scenario("no_such_scenario")

    def test_a_file_error_names_the_file(self, tmp_path):
        data = minimal()
        data["snr"] = 1
        path = write(tmp_path, data, name="oops.yaml")
        with pytest.raises(ValueError, match="oops.yaml"):
            load_scenario(path)


class TestShipped:
    def test_the_library_is_not_empty_and_sorted(self):
        names = list_scenarios()
        assert "chain" in names
        assert names == sorted(names)

    def test_every_listed_name_has_a_file(self):
        for name in list_scenarios():
            assert (SCENARIO_DIR / f"{name}.yaml").is_file()

    def test_every_file_is_listed(self):
        files = sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))
        assert files == list_scenarios()

    @pytest.mark.parametrize("name", list_scenarios())
    def test_every_shipped_scenario_loads_and_its_name_matches_its_file(self, name):
        spec = load_scenario(name)
        assert spec.name == name
        assert spec.description
        assert spec.network.name == name

    @pytest.mark.parametrize("name", list_scenarios())
    def test_every_shipped_scenario_survives_a_round_trip(self, name):
        spec = load_scenario(name)
        assert ScenarioSpec.from_dict(spec.to_dict()) == spec


def with_artifact(**overrides) -> dict:
    data = minimal()
    data["artifact"] = {
        "rate_hz": 2.0,
        "freq_hz": 30.0,
        "decay_ms": 40.0,
        "amplitude": 1.0,
        "anchor": B,
        "n_sensors": 6,
    }
    data["artifact"].update(overrides)
    return data


class TestArtifact:
    def test_the_block_is_optional(self):
        assert ScenarioSpec.from_dict(minimal()).artifact is None
        assert "artifact" not in ScenarioSpec.from_dict(minimal()).to_dict()

    def test_the_block_is_read_and_round_trips(self):
        spec = ScenarioSpec.from_dict(with_artifact())
        assert spec.artifact == ArtifactSpec(2.0, 30.0, 40.0, 1.0, B, 6)
        assert ScenarioSpec.from_dict(spec.to_dict()) == spec

    def test_the_version_stays_one(self):
        assert ScenarioSpec.from_dict(with_artifact()).to_dict()["version"] == 1

    def test_an_unknown_key_is_named(self):
        data = with_artifact()
        data["artifact"]["rate"] = 2.0
        with pytest.raises(ValueError, match="rate"):
            ScenarioSpec.from_dict(data)

    @pytest.mark.parametrize("key", ["rate_hz", "freq_hz", "decay_ms", "amplitude", "anchor"])
    def test_a_missing_key_is_named(self, key):
        data = with_artifact()
        del data["artifact"][key]
        with pytest.raises(ValueError, match=key):
            ScenarioSpec.from_dict(data)

    def test_the_anchor_must_be_a_node(self):
        with pytest.raises(ValueError, match="insula-rh"):
            ScenarioSpec.from_dict(with_artifact(anchor="insula-rh"))

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("rate_hz", 0.0),
            ("freq_hz", -1.0),
            ("decay_ms", 0.0),
            ("amplitude", float("nan")),
            ("n_sensors", 0),
            ("n_sensors", 2.5),
        ],
    )
    def test_bad_numbers_name_their_key(self, key, value):
        with pytest.raises(ValueError, match=key):
            ScenarioSpec.from_dict(with_artifact(**{key: value}))

    def test_the_frequency_must_be_below_nyquist(self):
        with pytest.raises(ValueError, match=r"freq_hz.*Nyquist"):
            ScenarioSpec.from_dict(with_artifact(freq_hz=50.0))

    def test_the_period_must_be_a_whole_number_of_samples(self):
        with pytest.raises(ValueError, match=r"rate_hz.*whole"):
            ScenarioSpec.from_dict(with_artifact(rate_hz=3.0))  # 33.33 samples at 100 Hz

    def test_the_decay_must_fit_inside_the_period(self):
        with pytest.raises(ValueError, match="decay_ms"):
            ScenarioSpec.from_dict(with_artifact(rate_hz=20.0, decay_ms=200.0))


class TestArtifactSignal:
    def spec(self, **overrides):
        return ScenarioSpec.from_dict(with_artifact(**overrides)).artifact

    def test_the_waveform_repeats_exactly_every_period_and_starts_at_the_stimulus(self):
        wave = artifact_waveform(self.spec(), sfreq=100.0, n_times=300)
        assert wave.shape == (300,)
        np.testing.assert_array_equal(wave[:50], wave[50:100])
        np.testing.assert_array_equal(wave[:50], wave[250:300])
        assert wave[0] == 0.0 and np.abs(wave[1:5]).max() > 0.0

    def test_the_waveform_is_a_damped_sine(self):
        spec = self.spec(freq_hz=25.0, decay_ms=40.0)
        wave = artifact_waveform(spec, sfreq=100.0, n_times=50)
        tau = np.arange(50) / 100.0
        np.testing.assert_allclose(wave, np.exp(-tau / 0.04) * np.sin(2 * np.pi * 25.0 * tau))

    def test_the_pattern_keeps_the_strongest_sensors_of_the_topography(self):
        topography = np.array([0.1, -3.0, 0.5, 2.0, -0.2, 1.0])
        pattern = artifact_pattern(topography, 3)
        assert np.flatnonzero(pattern).tolist() == [1, 3, 5]
        assert np.linalg.norm(pattern) == pytest.approx(1.0)
        # the signs are those of the topography, so the pattern is a real field of the region
        assert np.sign(pattern[[1, 3, 5]]).tolist() == [-1.0, 1.0, 1.0]

    def test_more_sensors_than_the_head_has_is_refused(self):
        with pytest.raises(ValueError, match="n_sensors"):
            artifact_pattern(np.ones(4), 5)

    def test_the_artifact_is_identical_in_every_trial_and_zero_outside_the_group(self):
        rng = np.random.default_rng(0)
        signal = rng.standard_normal((3, 8, 200))
        topography = rng.standard_normal(8)
        art = stimulation_artifact(self.spec(n_sensors=3), topography, signal, sfreq=100.0)
        assert art.shape == signal.shape
        np.testing.assert_array_equal(art[0], art[1])
        np.testing.assert_array_equal(art[0], art[2])
        group = np.flatnonzero(artifact_pattern(topography, 3))
        outside = np.setdiff1d(np.arange(8), group)
        assert not art[:, outside].any()

    @pytest.mark.parametrize("amplitude", [0.5, 1.0, 4.0])
    def test_the_power_in_the_group_is_the_amplitude_squared_times_the_signal_power(
        self, amplitude
    ):
        rng = np.random.default_rng(1)
        signal = rng.standard_normal((2, 8, 400)) * np.linspace(1.0, 3.0, 8)[None, :, None]
        topography = rng.standard_normal(8)
        art = stimulation_artifact(
            self.spec(amplitude=amplitude, n_sensors=4), topography, signal, sfreq=100.0
        )
        group = np.flatnonzero(artifact_pattern(topography, 4))
        ratio = np.mean(art[:, group] ** 2) / np.mean(signal[:, group] ** 2)
        assert ratio == pytest.approx(amplitude**2, rel=1e-9)

"""The head-model scenarios: leakage, distance strata, stimulation artifact."""

import numpy as np
import pytest

from dirconn.scenario import (
    artifact_pattern,
    load_scenario,
    stimulation_artifact,
)
from dirconn.simulation import simulate

Z, X, Y = "transversetemporal-rh", "cuneus-lh", "pericalcarine-lh"
R1, R2, R3 = "transversetemporal-rh", "cuneus-rh", "transversetemporal-lh"
DISTANCE = {
    "distance_near": ("parsorbitalis-rh", "temporalpole-rh"),
    "distance_mid": ("posteriorcingulate-rh", "rostralanteriorcingulate-rh"),
    "distance_far": ("frontalpole-rh", "parahippocampal-rh"),
}
# scenario -> the edges that must exist, in (source, target) form
EDGES = {
    "leakage_adjacent": {(Z, X)},
    "stimulation_artifact": {(R1, R2), (R2, R3)},
    **{name: {pair} for name, pair in DISTANCE.items()},
}


def edge_set(spec):
    return {(e.source, e.target) for e in spec.network.edges}


@pytest.mark.parametrize("name", sorted(EDGES))
class TestEveryScenario:
    def test_the_edges_are_the_intended_ones_and_nothing_is_hidden(self, name):
        spec = load_scenario(name)
        assert edge_set(spec) == EDGES[name]
        assert spec.hidden == ()

    def test_a_short_simulation_runs_and_equalises_the_node_power(self, name):
        spec = load_scenario(name)
        sim = simulate(
            spec.network,
            sfreq=spec.sfreq,
            n_times=3000,
            seed=0,
            dynamics=spec.dynamics,
            n_background=spec.n_background,
            background_fraction=spec.background_fraction,
            node_scale="unit",
        )
        power = np.var(np.asarray(sim.sources.data), axis=(0, 2))
        assert power == pytest.approx(np.ones(spec.network.n_nodes), rel=1e-6)
        truth = np.nan_to_num(np.asarray(sim.truth.values)[:, :, 0, 0]) > 0
        np.testing.assert_array_equal(truth, spec.network.adjacency())

    def test_the_description_says_what_the_scenario_is_for(self, name):
        assert len(load_scenario(name).description) > 80


class TestLeakageAdjacent:
    def test_the_driven_node_has_no_edge_into_its_neighbour(self):
        spec = load_scenario("leakage_adjacent")
        assert spec.network.names == (Z, X, Y)
        assert not any(e.target == Y or e.source == Y for e in spec.network.edges)

    def test_the_neighbour_oscillates_at_its_own_frequency(self):
        spec = load_scenario("leakage_adjacent")
        freqs = {n: d.freq_hz for n, d in spec.dynamics.items()}
        assert len(set(freqs.values())) == 3


class TestDistanceScenarios:
    def test_they_differ_only_in_the_regions(self):
        specs = [load_scenario(n) for n in sorted(DISTANCE)]
        strengths = {(e.strength, e.delay_ms) for s in specs for e in s.network.edges}
        assert strengths == {(0.4, 10.0)}
        dynamics = {tuple(d.freq_hz for d in s.dynamics.values()) for s in specs}
        assert dynamics == {(10.0, 20.0)}

    def test_each_pair_is_in_one_hemisphere(self):
        for name, (a, b) in DISTANCE.items():
            assert a[-2:] == b[-2:], name


class TestStimulationArtifact:
    def test_the_block_is_there_and_anchored_at_the_middle_node(self):
        spec = load_scenario("stimulation_artifact")
        assert spec.artifact is not None
        assert spec.artifact.anchor == R2
        assert spec.artifact.n_sensors == 6
        assert spec.artifact.rate_hz == 2.0

    def test_no_other_scenario_has_one(self):
        for name in EDGES:
            if name != "stimulation_artifact":
                assert load_scenario(name).artifact is None

    def test_it_can_be_added_to_simulated_sensor_data(self):
        spec = load_scenario("stimulation_artifact")
        sim = simulate(
            spec.network,
            sfreq=spec.sfreq,
            n_times=1000,
            n_trials=2,
            seed=1,
            dynamics=spec.dynamics,
            node_scale="unit",
        )
        anchor = spec.network.names.index(spec.artifact.anchor)
        topography = sim.leadfield[:, anchor]
        art = stimulation_artifact(spec.artifact, topography, sim.signal, spec.sfreq)
        assert art.shape == sim.signal.shape
        np.testing.assert_array_equal(art[0], art[1])
        group = np.flatnonzero(artifact_pattern(topography, spec.artifact.n_sensors))
        ratio = np.mean(art[:, group] ** 2) / np.mean(sim.signal[:, group] ** 2)
        assert ratio == pytest.approx(spec.artifact.amplitude**2)
        # one stimulus every 0.5 s: 50 samples at 100 Hz, and the first sample of a trial is one
        assert np.abs(art[0, group[0], ::50]).max() == 0.0


@pytest.mark.sample_data
class TestOnTheSampleHead:
    @pytest.fixture(scope="module")
    def heads(self):
        from dirconn.headmodel import load_head_model

        return {m: load_head_model(m) for m in ("eeg", "mag", "grad")}

    @pytest.fixture(scope="module")
    def geometry(self, heads):
        from dirconn.regions import centre_distances, min_visibility

        return min_visibility(heads.values()), centre_distances(heads["eeg"])

    def test_the_leakage_scenario_is_what_the_rules_give(self, heads, geometry):
        from dirconn.inverse import crosstalk
        from dirconn.regions import distant_region, leakage_pairs

        visibility, distances = geometry
        names = heads["eeg"].roi_names
        matrices = {m: crosstalk(heads[m], seed=0).matrix for m in ("eeg", "grad")}
        best = leakage_pairs(matrices, visibility)[0]
        z = distant_region((best.source, best.receiver), matrices, visibility, distances)
        assert (names[z], names[best.source], names[best.receiver]) == (Z, X, Y)

    def test_the_distance_scenarios_are_what_the_rules_give(self, heads, geometry):
        from dirconn.regions import distance_strata

        visibility, distances = geometry
        names = heads["eeg"].roi_names
        strata = distance_strata(visibility, distances, names)
        found = {
            f"distance_{label}": (names[i], names[j]) for label, (i, j) in strata.pairs.items()
        }
        assert found == DISTANCE

    def test_every_node_is_a_region_of_the_head_and_the_artifact_group_fits(self, heads):
        for name in EDGES:
            spec = load_scenario(name)
            for modality, head in heads.items():
                assert set(spec.network.names) <= set(head.roi_names), (name, modality)
                if spec.artifact is not None:
                    column = head.roi_names.index(spec.artifact.anchor)
                    pattern = artifact_pattern(
                        head.roi_leadfield[:, column], spec.artifact.n_sensors
                    )
                    assert np.count_nonzero(pattern) == spec.artifact.n_sensors

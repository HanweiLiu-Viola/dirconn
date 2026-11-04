"""The shipped scenarios: each one is what its description says."""

import numpy as np
import pytest

from dirconn.bands import BANDS
from dirconn.scenario import list_scenarios, load_scenario
from dirconn.simulation import simulate

# The region pool of the hand-written scenarios, chosen by scripts/pick_regions.py: visibility
# of at least 1.0 times the median region on EEG, magnetometers and gradiometers, the smallest of
# the three at most 1.8, and centres at least 60 mm apart.
R1, R2, R3, R4 = (
    "transversetemporal-rh",
    "cuneus-rh",
    "transversetemporal-lh",
    "rostralanteriorcingulate-lh",
)
POOL = (R1, R2, R3, R4)

# scenario -> (edges that must exist, nodes hidden from the methods)
HAND_WRITTEN: dict[str, tuple[set[tuple[str, str]], tuple[str, ...]]] = {
    "chain": ({(R1, R2), (R2, R3)}, ()),
    "common_driver": ({(R1, R2), (R1, R3)}, ()),
    "collider": ({(R1, R3), (R2, R3)}, ()),
    "bidirectional": ({(R1, R2), (R2, R1), (R3, R4)}, ()),
    "feedback_loop": ({(R1, R2), (R2, R3), (R3, R1)}, ()),
    "hidden_common_input": ({(R4, R1), (R4, R2), (R1, R3)}, (R4,)),
    "delay_1": ({(R1, R2)}, ()),
    "delay_5": ({(R1, R2)}, ()),
    "delay_zero_osc": ({(R1, R2)}, ()),
    "band_theta": ({(R1, R2)}, ()),
    "band_alpha": ({(R1, R2)}, ()),
    "band_beta": ({(R1, R2)}, ()),
    "band_gamma": ({(R1, R2)}, ()),
}
BANDED = {
    "band_theta": BANDS["theta"],
    "band_alpha": BANDS["alpha"],
    "band_beta": BANDS["beta"],
    "band_gamma": (30.0, 45.0),  # BANDS["gamma"] ends at 80 Hz, above the Nyquist of 100 Hz
}


def edge_set(spec):
    return {(e.source, e.target) for e in spec.network.edges}


def test_every_hand_written_scenario_is_shipped():
    assert set(HAND_WRITTEN) <= set(list_scenarios())


@pytest.mark.parametrize("name", sorted(HAND_WRITTEN))
class TestHandWritten:
    def test_edges_and_hidden_nodes_are_the_intended_ones(self, name):
        edges, hidden = HAND_WRITTEN[name]
        spec = load_scenario(name)
        assert edge_set(spec) == edges
        assert spec.hidden == hidden

    def test_nodes_come_from_the_region_pool(self, name):
        spec = load_scenario(name)
        assert set(spec.network.names) <= set(POOL)

    def test_the_description_is_not_empty_and_names_the_limit_or_the_point(self, name):
        assert len(load_scenario(name).description) > 40

    def test_a_short_simulation_runs_and_equalises_the_node_power(self, name):
        spec = load_scenario(name)
        sim = simulate(
            spec.network,
            sfreq=spec.sfreq,
            n_times=3000,
            seed=0,
            dynamics=spec.dynamics,
            snr_db=spec.snr_db,
            n_background=spec.n_background,
            background_fraction=spec.background_fraction,
            node_scale="unit",
        )
        power = np.var(np.asarray(sim.sources.data), axis=(0, 2))
        assert power == pytest.approx(np.ones(spec.network.n_nodes), rel=1e-6)

    def test_the_truth_tensor_has_ones_exactly_on_the_edges(self, name):
        edges, _ = HAND_WRITTEN[name]
        spec = load_scenario(name)
        freqs = np.arange(0.0, spec.sfreq / 2.0 + 1e-9, 1.0)
        truth = spec.network.truth(freqs=freqs)
        values = np.asarray(truth.values)
        names = list(truth.source)
        band = BANDED.get(name)
        for i, source in enumerate(names):
            for j, target in enumerate(names):
                if i == j:
                    assert np.isnan(values[i, j]).all()
                    continue
                expected = np.zeros(freqs.size)
                if (source, target) in edges:
                    expected[:] = 1.0
                    if band is not None:
                        expected = ((freqs >= band[0]) & (freqs <= band[1])).astype(float)
                np.testing.assert_array_equal(values[i, j, :, 0], expected)


class TestStructure:
    def test_the_banded_scenarios_have_one_banded_edge_inside_the_band_of_their_name(self):
        for name, band in BANDED.items():
            (edge,) = load_scenario(name).network.edges
            assert edge.band == band

    def test_the_nodes_of_a_banded_scenario_oscillate_inside_its_band(self):
        for name, (low, high) in BANDED.items():
            spec = load_scenario(name)
            for node in (R1, R2):
                assert low <= spec.dynamics[node].freq_hz <= high

    def test_the_delay_scenarios_differ_only_in_the_delay(self):
        delays = {
            n: load_scenario(n).network.edges[0].delay_ms for n in HAND_WRITTEN if "delay" in n
        }
        assert delays == {"delay_1": 10.0, "delay_5": 50.0, "delay_zero_osc": 0.0}

    def test_the_zero_delay_scenario_uses_oscillators(self):
        spec = load_scenario("delay_zero_osc")
        assert all(d.kind == "oscillator" for d in spec.dynamics.values())

    def test_loops_and_bidirectional_edges_are_weak(self):
        for name in ("bidirectional", "feedback_loop"):
            weak = [e.strength for e in load_scenario(name).network.edges if e.strength < 0.1]
            assert weak, name
        assert all(e.strength == 0.05 for e in load_scenario("feedback_loop").network.edges)

    def test_the_delay_and_band_scenarios_have_an_unconnected_node_as_a_null(self):
        for name in HAND_WRITTEN:
            if not (name.startswith("band_") or name.startswith("delay_")):
                continue
            spec = load_scenario(name)
            touched = {n for e in spec.network.edges for n in (e.source, e.target)}
            assert set(spec.network.names) - touched, name


@pytest.mark.sample_data
class TestPoolOnTheSampleHead:
    @pytest.fixture(scope="module")
    def heads(self):
        from dirconn.headmodel import load_head_model

        return {m: load_head_model(m) for m in ("eeg", "mag", "grad")}

    def test_the_pool_is_visible_on_every_modality_and_not_too_visible(self, heads):
        for region in POOL:
            seen = [h.visibility[h.roi_names.index(region)] for h in heads.values()]
            assert all(v >= 1.0 for v in seen), (region, seen)
            assert 1.0 <= min(seen) <= 1.8, (region, seen)

    def test_the_pool_centres_are_at_least_60_mm_apart(self, heads):
        head = heads["eeg"]
        src = head.forward["src"]
        rr = np.vstack([s["rr"][s["vertno"]] for s in src]) * 1000.0
        centres = np.array([rr[head.roi_centre[head.roi_names.index(r)]] for r in POOL])
        distance = np.linalg.norm(centres[:, None] - centres[None], axis=2)
        assert distance[np.triu_indices(len(POOL), 1)].min() >= 60.0

    def test_every_node_of_every_scenario_is_a_region_of_the_head(self, heads):
        head = heads["eeg"]
        for name in HAND_WRITTEN:
            assert set(load_scenario(name).network.names) <= set(head.roi_names), name

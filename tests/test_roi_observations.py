import numpy as np
import pytest

from dirconn import Edge, Network, dk_node
from dirconn.benchmark import roi_observations
from dirconn.scenario import load_scenario
from dirconn.simulation import NodeDynamics, simulate

A, B, C = "bankssts-lh", "superiorparietal-lh", "precuneus-rh"


def simulated(n_trials=1, n_times=600):
    net = Network(
        nodes=[dk_node(A), dk_node(B), dk_node(C)],
        edges=[Edge(A, B, 0.4, 10.0), Edge(B, C, 0.3, 10.0)],
    )
    return simulate(
        net,
        sfreq=100.0,
        n_times=n_times,
        seed=2,
        n_trials=n_trials,
        dynamics={n: NodeDynamics(kind="white") for n in (A, B, C)},
        n_background=0,
        node_scale="unit",
    )


class TestRoiObservationsMethod:
    def test_it_equals_the_benchmark_function_bit_for_bit(self):
        sim = simulated()
        np.testing.assert_array_equal(
            sim.roi_observations(snr_db=7.0, seed=3).data,
            roi_observations(sim, snr_db=7.0, seed=3).data,
        )

    def test_it_is_the_node_activity_plus_noise_at_the_requested_snr_per_node(self):
        sim = simulated()
        obs = sim.roi_observations(snr_db=6.0, seed=1)
        noise = obs.data - sim.sources.data
        for k in range(3):
            snr = 10 * np.log10(np.mean(sim.sources.data[:, k] ** 2) / np.mean(noise[:, k] ** 2))
            assert snr == pytest.approx(6.0, abs=1e-9)

    def test_the_names_and_region_metadata_follow_the_network(self):
        obs = simulated().roi_observations(snr_db=10.0, seed=0)
        assert obs.names == (A, B, C)
        assert [c.hemisphere for c in obs.channels] == ["lh", "lh", "rh"]
        assert [c.region for c in obs.channels] == ["bankssts", "superiorparietal", "precuneus"]

    def test_epoched_data_stay_epoched_and_continuous_data_stay_continuous(self):
        epoched = simulated(n_trials=3, n_times=200).roi_observations(snr_db=10.0, seed=0)
        assert epoched.segmentation == "epoched"
        assert epoched.data.shape == (3, 3, 200)
        assert simulated().roi_observations(snr_db=10.0, seed=0).segmentation == "continuous"

    def test_the_time_axis_is_the_one_of_the_sources(self):
        sim = simulated()
        obs = sim.roi_observations(snr_db=10.0, seed=0)
        assert obs.sfreq == sim.sources.sfreq
        assert obs.tmin == sim.sources.tmin
        assert obs.n_times == sim.sources.n_times

    def test_the_same_seed_gives_the_same_data_and_another_seed_does_not(self):
        sim = simulated()
        a = sim.roi_observations(snr_db=5.0, seed=4)
        np.testing.assert_array_equal(a.data, sim.roi_observations(snr_db=5.0, seed=4).data)
        assert not np.array_equal(a.data, sim.roi_observations(snr_db=5.0, seed=5).data)

    def test_the_seed_and_the_snr_are_required_keywords(self):
        sim = simulated()
        with pytest.raises(TypeError):
            sim.roi_observations(snr_db=5.0)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            sim.roi_observations(5.0, 1)  # type: ignore[misc]

    def test_it_leaves_the_simulation_untouched(self):
        sim = simulated()
        before = np.array(sim.sources.data)
        sim.roi_observations(snr_db=0.0, seed=0)
        np.testing.assert_array_equal(sim.sources.data, before)


class TestTutorialRecipe:
    """The call sequence of the tutorials: a shipped scenario, simulated and observed."""

    def test_a_shipped_scenario_gives_region_data_and_matching_truth(self):
        spec = load_scenario("chain")
        sim = simulate(
            spec.network,
            sfreq=spec.sfreq,
            n_times=spec.n_times,
            seed=1,
            dynamics=spec.dynamics,
            snr_db=spec.snr_db,
            n_background=0,
            node_scale="unit",
        )
        data = sim.roi_observations(snr_db=spec.snr_db, seed=2)
        assert data.names == tuple(spec.network.names)
        assert data.n_times == spec.n_times
        assert sim.truth.source == data.names

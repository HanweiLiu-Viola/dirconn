import dataclasses
import math
import warnings

import numpy as np
import pytest
from scipy import signal

from dirconn import DirconnWarning, Edge, Network, TimeSeries, direct_node, dk_node
from dirconn.simulation import (
    NodeDynamics,
    SimulatedData,
    artificial_leadfield,
    network_var,
    simulate,
)

A, B, C = "bankssts-lh", "superiorparietal-lh", "precuneus-rh"
WHITE = NodeDynamics(kind="white")


def three_cortical(edges):
    return Network(nodes=[dk_node(A), dk_node(B), dk_node(C)], edges=edges)


def with_stn(edges=()):
    return Network(
        nodes=[dk_node(A), dk_node(B), direct_node("STN-L", region="STN", hemisphere="lh")],
        edges=edges,
    )


def estimate_lagged(x, order):
    """Least-squares VAR fit; returns coefficients as [lag - 1, target, source]."""
    n, t = x.shape
    rows = [x[:, order - k : t - k].T for k in range(1, order + 1)]  # lag 1 ... order
    regressors = np.hstack(rows)  # (t - order, n * order), lag-major blocks of n sources
    target = x[:, order:].T
    beta, *_ = np.linalg.lstsq(regressors, target, rcond=None)  # (n * order, n)
    return beta.T.reshape(n, order, n).transpose(1, 0, 2)


class TestNodeDynamics:
    def test_defaults_describe_a_10_hz_oscillator(self):
        dyn = NodeDynamics()
        assert (dyn.kind, dyn.freq_hz, dyn.bandwidth_hz, dyn.noise_std) == (
            "oscillator",
            10.0,
            2.0,
            1.0,
        )

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            NodeDynamics().kind = "white"  # type: ignore[misc]

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            NodeDynamics(kind="pink")

    @pytest.mark.parametrize(
        "field, value", [("freq_hz", 0.0), ("bandwidth_hz", -1.0), ("noise_std", 0.0)]
    )
    def test_positive_parameters(self, field, value):
        with pytest.raises(ValueError, match=field):
            NodeDynamics(**{field: value})

    def test_oscillator_coefficients_place_the_resonance(self):
        a1, a2 = NodeDynamics(freq_hz=10.0, bandwidth_hz=2.0).ar_coefficients(250.0)
        radius = math.exp(-math.pi * 2.0 / 250.0)
        assert a1 == pytest.approx(2 * radius * math.cos(2 * math.pi * 10.0 / 250.0))
        assert a2 == pytest.approx(-(radius**2))

    def test_white_dynamics_have_no_memory(self):
        assert WHITE.ar_coefficients(250.0) == (0.0, 0.0)

    def test_frequency_must_be_below_nyquist(self):
        with pytest.raises(ValueError, match="Nyquist"):
            NodeDynamics(freq_hz=130.0).ar_coefficients(250.0)


class TestNetworkVar:
    def test_own_dynamics_fill_the_first_two_lags_of_the_diagonal(self):
        net = three_cortical([])
        model = network_var(net, 250.0)
        a1, a2 = NodeDynamics().ar_coefficients(250.0)
        assert model.order == 2
        assert model.names == (A, B, C)
        for j in range(3):
            assert model.lagged[0, j, j] == pytest.approx(a1)
            assert model.lagged[1, j, j] == pytest.approx(a2)
        assert not model.instantaneous.any()

    def test_delayed_edges_land_at_their_lag_indexed_target_by_source(self):
        net = three_cortical([Edge(A, B, strength=0.3, delay_ms=10.0)])
        model = network_var(net, 1000.0, {A: WHITE, B: WHITE, C: WHITE})
        assert model.order == 10
        assert model.lagged[9, 1, 0] == pytest.approx(0.3)  # lag 10, target B, source A
        assert model.lagged[9, 0, 1] == 0.0
        assert np.count_nonzero(model.lagged) == 1

    def test_instantaneous_edges_go_to_the_instantaneous_matrix(self):
        net = three_cortical([Edge(A, B, strength=0.3)])
        model = network_var(net, 250.0)
        assert model.instantaneous[1, 0] == pytest.approx(0.3)
        assert np.count_nonzero(model.instantaneous) == 1
        assert model.lagged[:, 1, 0].tolist() == [0.0, 0.0]

    def test_banded_edges_are_kept_out_of_the_linear_coefficients(self):
        net = three_cortical([Edge(A, B, strength=0.3, delay_ms=8.0, band=(13, 30))])
        model = network_var(net, 250.0)
        assert model.order == 2
        assert model.lagged[:, 1, 0].tolist() == [0.0, 0.0]
        assert [e.band for e in model.banded_edges] == [(13.0, 30.0)]

    def test_a_banded_edge_needs_at_least_one_sample_of_delay(self):
        net = three_cortical([Edge(A, B, strength=0.3, band=(13, 30))])
        with pytest.raises(ValueError, match="banded"):
            network_var(net, 250.0)

    def test_the_band_must_lie_below_nyquist(self):
        net = three_cortical([Edge(A, B, strength=0.3, delay_ms=8.0, band=(13, 200))])
        with pytest.raises(ValueError, match="Nyquist"):
            network_var(net, 250.0)

    def test_misaligned_delays_follow_the_rounding_mode(self):
        net = three_cortical([Edge(A, B, strength=0.3, delay_ms=10.0)])
        with pytest.raises(ValueError, match="whole number of samples"):
            network_var(net, 250.0)
        with pytest.warns(DirconnWarning, match="rounded"):
            model = network_var(net, 250.0, delay_rounding="round")
        assert model.lagged[2, 1, 0] == pytest.approx(0.3)  # 2.5 samples round half up to 3

    def test_noise_scale_comes_from_the_dynamics(self):
        net = three_cortical([])
        model = network_var(net, 250.0, {B: NodeDynamics(noise_std=3.0)})
        assert model.noise_std.tolist() == [1.0, 3.0, 1.0]

    def test_unknown_dynamics_names_are_rejected(self):
        with pytest.raises(ValueError, match="nowhere"):
            network_var(three_cortical([]), 250.0, {"nowhere": WHITE})

    def test_a_strongly_coupled_loop_is_reported_as_unstable(self):
        net = three_cortical(
            [Edge(A, B, strength=2.0, delay_ms=4.0), Edge(B, A, strength=2.0, delay_ms=4.0)]
        )
        with pytest.raises(ValueError, match="unstable"):
            network_var(net, 250.0, {A: WHITE, B: WHITE, C: WHITE})

    def test_singular_instantaneous_loops_are_rejected(self):
        net = three_cortical([Edge(A, B, strength=1.0), Edge(B, A, strength=1.0)])
        with pytest.raises(ValueError, match="instantaneous"):
            network_var(net, 250.0)

    def test_arrays_are_read_only(self):
        model = network_var(three_cortical([]), 250.0)
        with pytest.raises(ValueError, match="read-only"):
            model.lagged[0, 0, 0] = 1.0


class TestArtificialLeadfield:
    def test_shape_unit_columns_and_reproducibility(self):
        lead = artificial_leadfield(12, 5, seed=3)
        assert lead.shape == (12, 5)
        np.testing.assert_allclose(np.linalg.norm(lead, axis=0), 1.0)
        np.testing.assert_array_equal(lead, artificial_leadfield(12, 5, seed=3))
        assert not np.array_equal(lead, artificial_leadfield(12, 5, seed=4))

    def test_neighbouring_sources_share_sensor_patterns(self):
        lead = artificial_leadfield(16, 40, seed=0)
        gram = lead.T @ lead
        assert gram[~np.eye(40, dtype=bool)].max() > 0.9

    @pytest.mark.parametrize("n_sensors, n_sources", [(0, 3), (3, 0)])
    def test_sizes_must_be_positive(self, n_sensors, n_sources):
        with pytest.raises(ValueError, match="positive"):
            artificial_leadfield(n_sensors, n_sources, seed=0)


class TestSimulateOutput:
    def sim(self, **overrides):
        net = with_stn(
            [Edge(A, B, strength=0.3, delay_ms=8.0), Edge("STN-L", A, strength=0.2, delay_ms=12.0)]
        )
        kwargs = dict(sfreq=250.0, n_times=400, n_trials=3, seed=1, n_sensors=6, n_background=4)
        kwargs.update(overrides)
        return simulate(net, **kwargs)

    def test_shapes_names_and_channel_kinds(self):
        sim = self.sim()
        assert isinstance(sim, SimulatedData)
        assert sim.data.data.shape == (3, 7, 400)  # 6 sensors + the direct channel
        assert sim.data.names == ("S01", "S02", "S03", "S04", "S05", "S06", "STN-L")
        kinds = [ch.node_kind for ch in sim.data.channels]
        assert kinds == ["sensor"] * 6 + ["direct"]
        assert sim.data.channels[0].type == "eeg"
        assert sim.data.channels[-1].region == "STN"
        assert sim.sources.data.shape == (3, 3, 400)
        assert sim.sources.names == (A, B, "STN-L")
        assert sim.data.segmentation == sim.sources.segmentation == "epoched"
        assert sim.data.sfreq == sim.sources.sfreq == 250.0

    def test_a_single_trial_is_continuous(self):
        sim = self.sim(n_trials=1)
        assert sim.data.segmentation == "continuous"

    def test_output_is_finite_and_read_only(self):
        sim = self.sim()
        assert np.isfinite(sim.data.data).all()
        with pytest.raises(ValueError, match="read-only"):
            sim.signal[0, 0, 0] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            sim.leadfield[0, 0] = 1.0

    def test_same_seed_gives_identical_data_and_different_seeds_differ(self):
        first, again, other = self.sim(seed=5), self.sim(seed=5), self.sim(seed=6)
        np.testing.assert_array_equal(first.data.data, again.data.data)
        np.testing.assert_array_equal(first.sources.data, again.sources.data)
        assert not np.array_equal(first.data.data, other.data.data)

    def test_the_seed_is_required(self):
        with pytest.raises(TypeError):
            simulate(with_stn(), sfreq=250.0, n_times=100)  # type: ignore[call-arg]

    def test_truth_is_the_networks_truth(self):
        sim = self.sim()
        np.testing.assert_array_equal(sim.truth.values, sim.network.truth().values)
        assert sim.truth.source == (A, B, "STN-L")
        banded = sim.truth_at(freqs=[5.0, 20.0])
        assert banded.values.shape == (3, 3, 2, 1)

    def test_provenance_records_the_simulation(self):
        sim = self.sim(seed=9, snr_db=4.0)
        prov = sim.provenance
        assert prov.method == "simulate"
        assert prov.seed == 9
        assert prov.parameters["snr_db"] == 4.0
        assert prov.parameters["sfreq"] == 250.0
        assert prov.parameters["leadfield"] == "artificial"
        assert prov.parameters["network"]["nodes"][0]["name"] == A

    def test_a_reasonable_summary(self):
        assert "SimulatedData" in repr(self.sim())


class TestSimulateDynamics:
    def test_empirical_var_coefficients_match_the_specification(self):
        net = three_cortical(
            [Edge(A, B, strength=0.4, delay_ms=10.0), Edge(B, C, strength=-0.3, delay_ms=20.0)]
        )
        dynamics = {
            A: NodeDynamics(freq_hz=10.0),
            B: NodeDynamics(freq_hz=20.0),
            C: NodeDynamics(freq_hz=30.0),
        }
        sim = simulate(net, sfreq=500.0, n_times=40000, dynamics=dynamics, seed=0, n_sensors=4)
        model = network_var(net, 500.0, dynamics)
        estimated = estimate_lagged(sim.sources.data[0], model.order)
        np.testing.assert_allclose(estimated, model.lagged, atol=0.03)

    def test_instantaneous_coupling_has_the_specified_strength(self):
        net = three_cortical([Edge(A, B, strength=0.5)])
        dynamics = {A: WHITE, B: WHITE, C: WHITE}
        sim = simulate(net, sfreq=250.0, n_times=20000, dynamics=dynamics, seed=2, n_sensors=4)
        x = sim.sources.data[0]
        slope = (x[0] @ x[1]) / (x[0] @ x[0])
        assert slope == pytest.approx(0.5, abs=0.03)
        assert (
            abs((x[1] @ x[0]) / (x[1] @ x[1]) - 0.4) < 0.05
        )  # reverse regression: 0.5 / (1 + 0.25)

    def test_unconnected_nodes_are_uncorrelated(self):
        net = three_cortical([])
        dynamics = {A: WHITE, B: WHITE, C: WHITE}
        sim = simulate(net, sfreq=250.0, n_times=20000, dynamics=dynamics, seed=3, n_sensors=4)
        corr = np.corrcoef(sim.sources.data[0])
        assert np.abs(corr[~np.eye(3, dtype=bool)]).max() < 0.05

    def test_banded_coupling_acts_only_inside_its_band(self):
        net = three_cortical([Edge(A, B, strength=1.0, delay_ms=2.0, band=(13, 30))])
        dynamics = {A: WHITE, B: WHITE, C: WHITE}
        sim = simulate(net, sfreq=500.0, n_times=60000, dynamics=dynamics, seed=4, n_sensors=4)
        x = sim.sources.data[0]
        freqs, coherence = signal.coherence(x[0], x[1], fs=500.0, nperseg=512)
        inside = coherence[(freqs >= 17) & (freqs <= 24)].mean()
        outside = coherence[(freqs >= 70) & (freqs <= 120)].mean()
        assert inside > 0.35
        assert outside < 0.05

    def test_a_delay_shifts_the_cross_correlation_peak(self):
        net = three_cortical([Edge(A, B, strength=0.6, delay_ms=12.0)])
        dynamics = {A: WHITE, B: WHITE, C: WHITE}
        sim = simulate(net, sfreq=500.0, n_times=30000, dynamics=dynamics, seed=5, n_sensors=4)
        a, b = sim.sources.data[0, 0], sim.sources.data[0, 1]
        lags = np.arange(-20, 21)
        cc = [np.mean(a[20 + k : len(a) - 20 + k] * b[20 : len(b) - 20]) for k in lags]
        assert lags[int(np.argmax(cc))] == -6  # b[t] follows a[t - 6]

    def test_an_unstable_system_is_never_returned_as_data(self):
        net = three_cortical(
            [
                Edge(A, B, strength=3.0, delay_ms=4.0),
                Edge(B, A, strength=3.0, delay_ms=4.0),
            ]
        )
        with pytest.raises(ValueError, match="unstable"):
            simulate(net, sfreq=250.0, n_times=200, seed=0)


class TestSimulateNoise:
    def sim(self, **overrides):
        net = with_stn([Edge(A, B, strength=0.3, delay_ms=8.0)])
        kwargs = dict(sfreq=500.0, n_times=20000, seed=7, n_sensors=8, n_background=6)
        kwargs.update(overrides)
        return simulate(net, **kwargs)

    @pytest.mark.parametrize("snr_db", [-5.0, 0.0, 12.0])
    def test_sensor_snr_matches_the_definition_exactly(self, snr_db):
        sim = self.sim(snr_db=snr_db, n_times=3000)
        measured = 10 * np.log10(np.mean(sim.signal**2) / np.mean(sim.noise**2))
        assert measured == pytest.approx(snr_db, abs=1e-9)
        assert sim.snr_db == pytest.approx(snr_db, abs=1e-9)

    def test_sensor_data_are_signal_plus_noise(self):
        sim = self.sim(n_times=500)
        np.testing.assert_allclose(sim.data.data[:, :8], sim.signal + sim.noise)

    def test_signal_is_the_leadfield_applied_to_the_cortical_sources(self):
        sim = self.sim(n_times=500)
        cortical = sim.sources.data[:, :2]
        expected = np.einsum("sk,bkt->bst", sim.leadfield[:, :2], cortical)
        np.testing.assert_allclose(sim.signal, expected)
        assert sim.leadfield.shape == (8, 2 + 6)

    def test_direct_channels_use_their_own_snr_without_a_forward_model(self):
        sim = self.sim(snr_db=6.0, n_times=3000)
        direct_noise = sim.data.data[:, 8] - sim.sources.data[:, 2]
        measured = 10 * np.log10(np.mean(sim.sources.data[:, 2] ** 2) / np.mean(direct_noise**2))
        assert measured == pytest.approx(6.0, abs=1e-9)

    def test_background_sources_make_the_noise_pink(self):
        sim = self.sim(background_fraction=1.0)
        freqs, power = signal.welch(sim.noise[0], fs=500.0, nperseg=1024)
        mean_power = power.mean(axis=0)
        low = mean_power[(freqs >= 1) & (freqs <= 4)].mean()
        high = mean_power[(freqs >= 100) & (freqs <= 200)].mean()
        assert low / high > 10

    def test_without_background_fraction_the_noise_is_white(self):
        sim = self.sim(background_fraction=0.0)
        freqs, power = signal.welch(sim.noise[0], fs=500.0, nperseg=1024)
        mean_power = power.mean(axis=0)
        low = mean_power[(freqs >= 1) & (freqs <= 4)].mean()
        high = mean_power[(freqs >= 100) & (freqs <= 200)].mean()
        assert 0.6 < low / high < 1.7

    def test_no_background_sources_means_white_noise_only(self):
        sim = self.sim(n_background=0, background_fraction=0.9)
        assert sim.leadfield.shape == (8, 2)
        freqs, power = signal.welch(sim.noise[0], fs=500.0, nperseg=1024)
        mean_power = power.mean(axis=0)
        assert (
            0.6
            < mean_power[(freqs >= 1) & (freqs <= 4)].mean()
            / mean_power[(freqs >= 100) & (freqs <= 200)].mean()
            < 1.7
        )

    def test_a_custom_leadfield_sets_the_sensor_count(self):
        lead = artificial_leadfield(5, 2 + 3, seed=1)
        net = with_stn()
        sim = simulate(net, sfreq=250.0, n_times=300, n_background=3, leadfield=lead, seed=0)
        assert sim.data.data.shape[1] == 5 + 1
        np.testing.assert_array_equal(sim.leadfield, lead)
        assert sim.provenance.parameters["leadfield"] == "custom"

    def test_a_custom_leadfield_must_have_the_right_columns(self):
        with pytest.raises(ValueError, match="leadfield"):
            simulate(
                with_stn(),
                sfreq=250.0,
                n_times=100,
                n_background=3,
                leadfield=np.ones((5, 4)),
                seed=0,
            )

    def test_a_conflicting_sensor_count_is_rejected(self):
        with pytest.raises(ValueError, match="n_sensors"):
            simulate(
                with_stn(),
                sfreq=250.0,
                n_times=100,
                n_background=0,
                n_sensors=7,
                leadfield=np.ones((5, 2)),
                seed=0,
            )

    def test_default_leadfield_columns_have_unit_norm(self):
        sim = self.sim(n_times=200)
        np.testing.assert_allclose(np.linalg.norm(sim.leadfield, axis=0), 1.0)


class TestSimulateValidation:
    def run(self, **overrides):
        kwargs = dict(sfreq=250.0, n_times=100, seed=0, n_sensors=4)
        kwargs.update(overrides)
        return simulate(with_stn(), **kwargs)

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"sfreq": 0.0}, "sfreq"),
            ({"n_times": 0}, "n_times"),
            ({"n_trials": 0}, "n_trials"),
            ({"n_background": -1}, "n_background"),
            ({"n_sensors": 0}, "n_sensors"),
            ({"snr_db": float("nan")}, "snr_db"),
            ({"background_fraction": 1.5}, "background_fraction"),
            ({"background_fraction": -0.1}, "background_fraction"),
            ({"burn_in": -1}, "burn_in"),
            ({"delay_rounding": "floor"}, "rounding"),
        ],
    )
    def test_invalid_arguments_are_rejected(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            self.run(**overrides)

    def test_sensor_data_need_a_cortical_node(self):
        net = Network(nodes=[direct_node("STN-L"), direct_node("STN-R")], edges=[])
        with pytest.raises(ValueError, match="cortical"):
            simulate(net, sfreq=250.0, n_times=100, seed=0)

    def test_misaligned_delays_surface_through_simulate(self):
        net = with_stn([Edge(A, B, strength=0.3, delay_ms=10.0)])
        with pytest.raises(ValueError, match="whole number of samples"):
            simulate(net, sfreq=250.0, n_times=100, seed=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            simulate(net, sfreq=250.0, n_times=100, seed=0, delay_rounding="round")
        assert any(issubclass(w.category, DirconnWarning) for w in caught)

    def test_the_result_can_feed_an_estimator_input(self):
        sim = self.run()
        assert isinstance(sim.data, TimeSeries)


class TestNodeScale:
    @staticmethod
    def chain_network():
        names = ("a-lh", "b-lh", "c-lh")
        nodes = [dk_node(n) for n in names]
        edges = [Edge("a-lh", "b-lh", 0.4, 10.0), Edge("b-lh", "c-lh", 0.4, 10.0)]
        return Network(nodes=nodes, edges=edges)

    def run(self, **kwargs):
        params = {"sfreq": 100.0, "n_times": 3000, "seed": 2, "snr_db": 10.0}
        params.update(kwargs)
        return simulate(self.chain_network(), **params)

    def test_by_default_the_node_powers_are_left_as_the_dynamics_make_them(self):
        # In a chain every step amplifies: the last node is much stronger than the first.
        std = self.run().sources.data.std(axis=2)[0]
        assert std[2] / std[0] > 5

    def test_unit_scaling_gives_every_node_unit_variance(self):
        sim = self.run(node_scale="unit")
        np.testing.assert_allclose(sim.sources.data.std(axis=(0, 2)), 1.0, rtol=1e-12)

    def test_the_default_output_is_unchanged_by_the_new_option(self):
        np.testing.assert_array_equal(self.run().data.data, self.run(node_scale="none").data.data)

    def test_scaling_changes_the_amplitudes_but_not_the_structure(self):
        # a rescaled node is the same time course up to a positive constant
        plain, unit = self.run(), self.run(node_scale="unit")
        for k in range(3):
            ratio = unit.sources.data[0, k] / plain.sources.data[0, k]
            np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)
            assert ratio[0] > 0
        np.testing.assert_array_equal(unit.truth.values, plain.truth.values)

    def test_the_signal_is_still_the_leadfield_times_the_scaled_activity(self):
        sim = self.run(node_scale="unit")
        expected = np.einsum("sk,bkt->bst", sim.leadfield[:, :3], sim.sources.data)
        np.testing.assert_allclose(sim.signal, expected, rtol=1e-12, atol=1e-12)

    def test_the_snr_is_still_exact(self):
        assert self.run(node_scale="unit", snr_db=-3.0).snr_db == pytest.approx(-3.0, abs=1e-9)

    def test_the_choice_is_recorded(self):
        assert self.run(node_scale="unit").provenance.parameters["node_scale"] == "unit"
        assert self.run().provenance.parameters["node_scale"] == "none"

    def test_an_unknown_scaling_names_the_choices(self):
        with pytest.raises(ValueError, match="none.*unit"):
            self.run(node_scale="rms")

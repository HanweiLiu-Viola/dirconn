"""Edges that are active only in given time intervals, and the piecewise-constant simulator."""

import numpy as np
import pytest
import yaml

from dirconn.network import Edge, Network, dk_node
from dirconn.scenario import ScenarioSpec, load_scenario
from dirconn.simulation import (
    NodeDynamics,
    network_var,
    network_var_schedule,
    simulate,
)

A, B, C = "bankssts-lh", "superiorparietal-lh", "precuneus-rh"
WHITE = NodeDynamics(kind="white")
SFREQ = 100.0


def net(*edges, nodes=(A, B, C)):
    return Network(tuple(dk_node(n) for n in nodes), tuple(edges), name="t")


# --------------------------------------------------------------------------
# The Edge field
# --------------------------------------------------------------------------


def test_an_edge_is_always_active_by_default():
    edge = Edge(A, B, 0.4)
    assert edge.active is None
    assert edge.is_active([0.0, 5.0, 1e6]).all()
    assert "active" not in edge.to_dict()


def test_the_intervals_are_normalised_to_tuples_of_floats():
    edge = Edge(A, B, 0.4, active=[[1, 2], (3.5, 4)])
    assert edge.active == ((1.0, 2.0), (3.5, 4.0))
    assert edge.to_dict()["active"] == [[1.0, 2.0], [3.5, 4.0]]


def test_an_interval_is_half_open_with_a_tolerance_for_float_noise():
    edge = Edge(A, B, 0.4, active=[(0.3, 0.6)])
    times = np.arange(10) * 0.1  # 0.30000000000000004 ... 0.6000000000000001
    np.testing.assert_array_equal(
        edge.is_active(times), [False, False, False, True, True, True, False, False, False, False]
    )


def test_several_intervals_and_the_gaps_between_them():
    edge = Edge(A, B, 0.4, active=[(1.0, 2.0), (4.0, 5.0)])
    np.testing.assert_array_equal(
        edge.is_active([0.5, 1.0, 1.99, 2.0, 3.0, 4.0, 4.5, 5.0]),
        [False, True, True, False, False, True, True, False],
    )


@pytest.mark.parametrize(
    "active,message",
    [
        ([], "active must"),
        ("abc", "active must"),
        ([(1.0,)], "active must"),
        ([(1.0, 2.0, 3.0)], "active must"),
        ([(-1.0, 2.0)], "must not be negative"),
        ([(2.0, 2.0)], "start must be smaller than the stop"),
        ([(3.0, 2.0)], "start must be smaller than the stop"),
        ([(0.0, float("nan"))], "finite"),
        ([(0.0, float("inf"))], "finite"),
        ([(0.0, 2.0), (1.0, 3.0)], "ascending and must not overlap"),
        ([(4.0, 5.0), (0.0, 1.0)], "ascending and must not overlap"),
    ],
)
def test_invalid_intervals_are_rejected(active, message):
    with pytest.raises(ValueError, match=message):
        Edge(A, B, 0.4, active=active)


# --------------------------------------------------------------------------
# Network: duplicates, truth, files
# --------------------------------------------------------------------------


def test_the_same_pair_may_appear_again_when_the_intervals_do_not_overlap():
    strong = Edge(A, B, 0.6, active=[(0.0, 2.0)])
    weak = Edge(A, B, 0.2, active=[(2.0, 4.0)])
    assert net(strong, weak).n_nodes == 3


@pytest.mark.parametrize(
    "first,second",
    [
        (Edge(A, B, 0.6), Edge(A, B, 0.2, active=[(2.0, 4.0)])),
        (Edge(A, B, 0.6, active=[(0.0, 3.0)]), Edge(A, B, 0.2, active=[(2.0, 4.0)])),
        (Edge(A, B, 0.6, active=[(0.0, 3.0)]), Edge(A, B, 0.2)),
    ],
)
def test_overlapping_duplicates_are_rejected(first, second):
    with pytest.raises(ValueError, match="duplicate edge"):
        net(first, second)


def test_truth_without_times_counts_a_windowed_edge_as_present():
    truth = net(Edge(A, B, 0.4, active=[(1.0, 2.0)])).truth()
    assert truth.values[0, 1, 0, 0] == 1.0


def test_truth_follows_the_intervals_over_time():
    network = net(Edge(A, B, 0.4, active=[(1.0, 3.0)]), Edge(B, C, 0.4))
    times = np.arange(6.0)
    truth = network.truth(times=times)
    assert truth.values.shape == (3, 3, 1, 6)
    np.testing.assert_array_equal(truth.values[0, 1, 0], [0, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(truth.values[1, 2, 0], np.ones(6))
    np.testing.assert_array_equal(truth.times, times)
    assert np.isnan(truth.values[0, 0]).all()


def test_truth_combines_bands_and_intervals():
    network = net(Edge(A, B, 0.4, delay_ms=10.0, band=(8.0, 12.0), active=[(1.0, 2.0)]))
    freqs = np.array([5.0, 10.0, 20.0])
    truth = network.truth(freqs=freqs, times=[0.5, 1.5])
    np.testing.assert_array_equal(truth.values[0, 1, :, 0], [0, 0, 0])
    np.testing.assert_array_equal(truth.values[0, 1, :, 1], [0, 1, 0])


def test_two_edges_of_one_pair_give_one_truth_cell():
    network = net(Edge(A, B, 0.6, active=[(0.0, 2.0)]), Edge(A, B, 0.2, active=[(2.0, 4.0)]))
    np.testing.assert_array_equal(network.truth(times=[1.0, 3.0, 5.0]).values[0, 1, 0], [1, 1, 0])


def test_the_intervals_survive_a_yaml_and_dict_round_trip(tmp_path):
    network = net(Edge(A, B, 0.4, active=[(1.0, 2.0), (3.0, 4.0)]), Edge(B, C, 0.3))
    assert Network.from_dict(network.to_dict()) == network
    path = tmp_path / "network.yaml"
    network.to_yaml(path)
    assert Network.from_yaml(path) == network
    assert "active" in path.read_text(encoding="utf-8")


def test_an_unknown_edge_key_is_still_an_error():
    data = net(Edge(A, B, 0.4)).to_dict()
    data["edges"][0]["activ"] = [[0, 1]]
    with pytest.raises(ValueError, match="Unknown key"):
        Network.from_dict(data)


# --------------------------------------------------------------------------
# network_var and the schedule
# --------------------------------------------------------------------------


def test_network_var_needs_a_time_when_edges_have_intervals():
    network = net(Edge(A, B, 0.4, delay_ms=10.0, active=[(1.0, 2.0)]))
    with pytest.raises(ValueError, match="at_time"):
        network_var(network, SFREQ)
    inside = network_var(network, SFREQ, at_time=1.5)
    outside = network_var(network, SFREQ, at_time=0.5)
    assert inside.lagged[0, 1, 0] == pytest.approx(0.4)
    assert outside.lagged[0, 1, 0] == 0.0


def test_a_static_network_ignores_at_time():
    network = net(Edge(A, B, 0.4, delay_ms=10.0))
    np.testing.assert_array_equal(
        network_var(network, SFREQ, at_time=3.0).lagged, network_var(network, SFREQ).lagged
    )


def test_the_schedule_of_a_static_network_is_one_segment():
    schedule = network_var_schedule(net(Edge(A, B, 0.4, delay_ms=10.0)), SFREQ, 500)
    assert [start for start, _ in schedule] == [0]
    assert schedule[0][1].lagged[0, 1, 0] == pytest.approx(0.4)


def test_the_schedule_splits_where_the_set_of_active_edges_changes():
    network = net(
        Edge(A, B, 0.4, delay_ms=10.0, active=[(1.0, 3.0)]),
        Edge(B, C, 0.3, delay_ms=10.0, active=[(2.0, 4.0)]),
    )
    schedule = network_var_schedule(network, SFREQ, 500)
    assert [start for start, _ in schedule] == [0, 100, 200, 300, 400]
    ab = [m.lagged[0, 1, 0] for _, m in schedule]
    bc = [m.lagged[0, 2, 1] for _, m in schedule]
    np.testing.assert_allclose(ab, [0, 0.4, 0.4, 0, 0])
    np.testing.assert_allclose(bc, [0, 0, 0.3, 0.3, 0])


def test_intervals_may_follow_each_other_and_are_then_one_stretch():
    edge = Edge(A, B, 0.4, delay_ms=10.0, active=[(1.0, 2.0), (2.0, 3.0)])
    assert edge.active == ((1.0, 2.0), (2.0, 3.0))
    schedule = network_var_schedule(net(edge), SFREQ, 500)
    assert [start for start, _ in schedule] == [0, 100, 300]


def test_a_gap_between_intervals_is_a_real_change():
    network = net(Edge(A, B, 0.4, delay_ms=10.0, active=[(1.0, 2.0), (2.5, 3.0)]))
    schedule = network_var_schedule(network, SFREQ, 500)
    assert [start for start, _ in schedule] == [0, 100, 200, 250, 300]


def test_a_window_that_starts_at_zero_gives_no_empty_first_segment():
    network = net(Edge(A, B, 0.4, delay_ms=10.0, active=[(0.0, 2.0)]))
    schedule = network_var_schedule(network, SFREQ, 500)
    assert [start for start, _ in schedule] == [0, 200]
    assert schedule[0][1].lagged[0, 1, 0] == pytest.approx(0.4)


@pytest.mark.parametrize("active", [[(1.0, 6.0)], [(5.0, 6.0)], [(0.0, 5.01)]])
def test_intervals_must_lie_inside_the_recording(active):
    network = net(Edge(A, B, 0.4, delay_ms=10.0, active=active))
    with pytest.raises(ValueError, match="inside the recording"):
        network_var_schedule(network, SFREQ, 500)  # 5 s


def test_every_segment_must_be_stable_and_the_error_names_the_time():
    loop = [
        Edge(A, B, 2.0, delay_ms=10.0, active=[(2.0, 4.0)]),
        Edge(B, A, 2.0, delay_ms=10.0, active=[(2.0, 4.0)]),
    ]
    with pytest.raises(ValueError, match=r"unstable.*2 s to 4 s|2 s to 4 s.*unstable"):
        network_var_schedule(net(*loop), SFREQ, 500)


# --------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------


def lag_correlation(x, y, start, stop):
    """Correlation of ``y[t]`` with ``x[t - 1]`` over the samples ``start .. stop``."""
    return np.corrcoef(x[start - 1 : stop - 1], y[start:stop])[0, 1]


def simulated(network, *, n_times=6000, seed=0, **kwargs):
    return simulate(
        network,
        sfreq=SFREQ,
        n_times=n_times,
        seed=seed,
        dynamics={A: WHITE, B: WHITE, C: WHITE},
        **kwargs,
    )


def test_coupling_appears_only_inside_its_interval():
    network = net(Edge(A, B, 0.6, delay_ms=10.0, active=[(20.0, 40.0)]))
    sim = simulated(network)
    x = sim.sources.data[0]
    a, b = x[0], x[1]
    assert abs(lag_correlation(a, b, 100, 1900)) < 0.12
    assert lag_correlation(a, b, 2100, 3900) > 0.35
    assert abs(lag_correlation(a, b, 4100, 5900)) < 0.12


def test_a_direction_flip_moves_the_coupling_from_one_pair_to_the_other():
    network = net(
        Edge(A, B, 0.6, delay_ms=10.0, active=[(0.0, 30.0)]),
        Edge(B, A, 0.6, delay_ms=10.0, active=[(30.0, 60.0)]),
    )
    x = simulated(network).sources.data[0]
    a, b = x[0], x[1]
    assert lag_correlation(a, b, 100, 2900) > 0.35 and abs(lag_correlation(b, a, 100, 2900)) < 0.12
    assert (
        lag_correlation(b, a, 3100, 5900) > 0.35 and abs(lag_correlation(a, b, 3100, 5900)) < 0.12
    )


@pytest.mark.parametrize("delay_ms", [0.0, 10.0, 30.0])
def test_a_window_over_the_whole_recording_equals_the_static_simulation(delay_ms):
    """Same seed, same random numbers: one segment must reproduce the static run exactly."""
    static = simulated(net(Edge(A, B, 0.4, delay_ms=delay_ms)), n_times=800)
    windowed = simulated(net(Edge(A, B, 0.4, delay_ms=delay_ms, active=[(0.0, 8.0)])), n_times=800)
    np.testing.assert_array_equal(windowed.sources.data, static.sources.data)
    np.testing.assert_array_equal(windowed.data.data, static.data.data)


def test_a_banded_edge_with_an_interval_runs_and_couples_only_inside_it():
    network = net(
        Edge(A, B, 0.8, delay_ms=20.0, band=(8.0, 12.0), active=[(20.0, 40.0)]),
    )
    dynamics = {A: NodeDynamics(), B: WHITE, C: WHITE}
    sim = simulate(network, sfreq=SFREQ, n_times=6000, seed=1, dynamics=dynamics, n_background=0)
    x = sim.sources.data[0]
    assert np.isfinite(x).all()
    from scipy import signal

    sos = signal.butter(2, [8.0, 12.0], btype="bandpass", fs=SFREQ, output="sos")
    band_b = signal.sosfiltfilt(sos, x[1])
    band_a = signal.sosfiltfilt(sos, x[0])

    def lagged_corr(first, last):
        return max(
            abs(np.corrcoef(band_a[first - lag : last - lag], band_b[first:last])[0, 1])
            for lag in range(9)
        )

    assert lagged_corr(2100, 3900) > 2 * lagged_corr(100, 1900)


def test_the_simulation_is_reproducible_and_the_truth_follows_the_intervals():
    network = net(Edge(A, B, 0.6, delay_ms=10.0, active=[(20.0, 40.0)]))
    first, second = simulated(network, seed=3), simulated(network, seed=3)
    np.testing.assert_array_equal(first.data.data, second.data.data)
    times = np.array([10.0, 30.0, 50.0])
    np.testing.assert_array_equal(first.truth_at(times=times).values[0, 1, 0], [0, 1, 0])
    assert first.truth.values[0, 1, 0, 0] == 1.0  # broadband, no time: present


def test_the_simulation_records_the_intervals_in_its_provenance():
    network = net(Edge(A, B, 0.6, delay_ms=10.0, active=[(20.0, 40.0)]))
    sim = simulated(network, n_times=6000)
    edge = sim.provenance.parameters["network"]["edges"][0]
    assert edge["active"] == [[20.0, 40.0]]


def test_intervals_outside_the_simulated_recording_are_an_error():
    network = net(Edge(A, B, 0.6, delay_ms=10.0, active=[(20.0, 80.0)]))
    with pytest.raises(ValueError, match="inside the recording"):
        simulated(network, n_times=6000)


# --------------------------------------------------------------------------
# Scenario files
# --------------------------------------------------------------------------


def scenario_dict(active):
    return {
        "version": 1,
        "name": "switch",
        "description": "A -> B in the middle only.",
        "defaults": {"sfreq": 100.0, "duration_s": 60.0},
        "network": {
            "nodes": [{"name": A}, {"name": B}],
            "edges": [
                {"source": A, "target": B, "strength": 0.4, "delay_ms": 10.0, "active": active}
            ],
        },
    }


def test_a_scenario_file_can_carry_intervals(tmp_path):
    spec = ScenarioSpec.from_dict(scenario_dict([[20.0, 40.0]]))
    assert spec.network.edges[0].active == ((20.0, 40.0),)
    path = tmp_path / "switch.yaml"
    path.write_text(
        yaml.safe_dump(scenario_dict([[20.0, 40.0]]), sort_keys=False), encoding="utf-8"
    )
    assert load_scenario(path).network == spec.network
    spec.to_yaml(tmp_path / "again.yaml")
    assert load_scenario(tmp_path / "again.yaml").network == spec.network


def test_a_scenario_with_an_interval_beyond_its_duration_is_refused():
    with pytest.raises(ValueError, match="inside the recording"):
        ScenarioSpec.from_dict(scenario_dict([[20.0, 90.0]]))


def test_an_unstable_segment_of_a_scenario_is_refused():
    data = scenario_dict([[20.0, 40.0]])
    data["network"]["edges"] = [
        {"source": A, "target": B, "strength": 2.0, "delay_ms": 10.0, "active": [[20.0, 40.0]]},
        {"source": B, "target": A, "strength": 2.0, "delay_ms": 10.0, "active": [[20.0, 40.0]]},
    ]
    with pytest.raises(ValueError, match="cannot be simulated"):
        ScenarioSpec.from_dict(data)

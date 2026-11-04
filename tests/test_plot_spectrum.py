import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dirconn import Connectivity, Edge, Network, Provenance, dk_node  # noqa: E402
from dirconn.simulation import NodeDynamics, simulate  # noqa: E402

FREQS = np.array([2.0, 4.0, 6.0, 8.0])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def prov():
    return Provenance(method="pdc", parameters={"order": 2}, model_order=2, seed=0)


def spectrum(prov, times=None):
    """Three nodes; every (source, target, freq) value is distinct so any mix-up shows."""
    n_times = 1 if times is None else len(times)
    values = np.arange(3 * 3 * len(FREQS) * n_times, dtype=float).reshape(3, 3, len(FREQS), n_times)
    for k in range(3):
        values[k, k] = np.nan
    return Connectivity.from_array(
        values, source=["A", "B", "C"], freqs=FREQS, times=times, provenance=prov
    )


def pair_grid(prov):
    """Two nodes, no frequency axis."""
    values = np.array([[np.nan, 0.8], [0.1, np.nan]])
    return Connectivity.from_array(values, source=["A", "B"], provenance=prov)


class TestDirectedSpectrum:
    def test_the_two_curves_are_the_two_directions_of_the_pair(self, prov):
        conn = spectrum(prov)
        ax = conn.plot(kind="directed_spectrum", source="A", target="C")
        forward, backward = ax.lines[:2]
        np.testing.assert_array_equal(forward.get_xdata(), FREQS)
        np.testing.assert_array_equal(forward.get_ydata(), conn.values[0, 2, :, 0])  # A -> C
        np.testing.assert_array_equal(backward.get_ydata(), conn.values[2, 0, :, 0])  # C -> A

    def test_swapping_source_and_target_swaps_the_curves(self, prov):
        conn = spectrum(prov)
        ax = conn.plot(kind="directed_spectrum", source="C", target="A")
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), conn.values[2, 0, :, 0])
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), conn.values[0, 2, :, 0])

    def test_the_legend_names_the_directions(self, prov):
        ax = spectrum(prov).plot(kind="directed_spectrum", source="A", target="B")
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert texts == ["A → B", "B → A"]

    def test_the_directions_differ_by_line_style_as_well_as_colour(self, prov):
        ax = spectrum(prov).plot(kind="directed_spectrum", source="A", target="B")
        first, second = ax.lines[:2]
        assert first.get_linestyle() != second.get_linestyle()
        assert first.get_color() != second.get_color()

    def test_the_axes_are_labelled_and_the_title_names_the_method(self, prov):
        ax = spectrum(prov).plot(kind="directed_spectrum", source="A", target="B")
        assert ax.get_xlabel() == "frequency (Hz)"
        assert "pdc" in ax.get_title()

    def test_a_time_resolved_result_needs_a_time_and_says_which_it_took(self, prov):
        conn = spectrum(prov, times=np.array([0.0, 0.5, 1.0]))
        with pytest.raises(ValueError, match="time"):
            conn.plot(kind="directed_spectrum", source="A", target="B")
        ax = conn.plot(kind="directed_spectrum", source="A", target="B", time=0.9)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), conn.values[0, 1, :, 2])
        assert "1 s" in ax.get_title()

    def test_a_result_without_a_frequency_axis_is_refused(self, prov):
        with pytest.raises(ValueError, match="frequency"):
            pair_grid(prov).plot(kind="directed_spectrum", source="A", target="B")

    def test_unknown_names_are_named_in_the_error(self, prov):
        with pytest.raises(ValueError, match="'Q'"):
            spectrum(prov).plot(kind="directed_spectrum", source="Q", target="A")

    def test_a_node_paired_with_itself_is_refused(self, prov):
        with pytest.raises(ValueError, match="itself"):
            spectrum(prov).plot(kind="directed_spectrum", source="A", target="A")

    def test_source_and_target_are_required(self, prov):
        with pytest.raises(TypeError):
            spectrum(prov).plot(kind="directed_spectrum")

    def test_it_draws_on_the_given_axes(self, prov):
        fig, ax = plt.subplots()
        assert spectrum(prov).plot(kind="directed_spectrum", source="A", target="B", ax=ax) is ax
        assert plt.get_fignums() == [fig.number]

    def test_complex_values_are_refused(self, prov):
        values = np.ones((2, 2, len(FREQS), 1), dtype=complex)
        conn = Connectivity.from_array(values, source=["A", "B"], freqs=FREQS, provenance=prov)
        with pytest.raises(ValueError, match="Complex"):
            conn.plot(kind="directed_spectrum", source="A", target="B")


def simulated_chain(prov_unused=None):
    a, b, c = "bankssts-lh", "superiorparietal-lh", "precuneus-rh"
    net = Network(
        nodes=[dk_node(a), dk_node(b), dk_node(c)],
        edges=[Edge(a, b, 0.4, 10.0), Edge(b, c, 0.3, 10.0)],
    )
    return simulate(
        net,
        sfreq=100.0,
        n_times=400,
        seed=1,
        dynamics={n: NodeDynamics(kind="white") for n in (a, b, c)},
        n_background=0,
        node_scale="unit",
    )


class TestTruthComparison:
    def test_the_left_panel_is_the_truth_and_the_right_one_the_estimate(self, prov):
        sim = simulated_chain()
        estimate = Connectivity.from_array(
            np.where(np.isnan(sim.truth.values[:, :, 0, 0]), np.nan, 0.5),
            source=sim.truth.source,
            provenance=prov,
        )
        left, right = estimate.plot(kind="truth_comparison", truth=sim.truth)
        np.testing.assert_array_equal(
            np.ma.getdata(left.images[0].get_array()), sim.truth.values[:, :, 0, 0]
        )
        np.testing.assert_array_equal(
            np.ma.getdata(right.images[0].get_array()), estimate.values[:, :, 0, 0]
        )
        assert left.get_title() == "truth"
        assert "pdc" in right.get_title()

    def test_the_two_panels_share_one_figure_and_the_same_cell_order(self, prov):
        sim = simulated_chain()
        estimate = Connectivity.from_array(
            np.where(np.isnan(sim.truth.values[:, :, 0, 0]), np.nan, 0.5),
            source=sim.truth.source,
            provenance=prov,
        )
        left, right = estimate.plot(kind="truth_comparison", truth=sim.truth)
        assert left.figure is right.figure
        assert labels(left.get_yticklabels()) == labels(right.get_yticklabels())

    def test_the_slice_options_apply_to_both_panels(self, prov):
        conn = spectrum(prov)
        truth = Connectivity.from_array(
            np.where(np.isnan(conn.values), np.nan, 1.0),
            source=conn.source,
            freqs=FREQS,
            provenance=prov,
        )
        left, right = conn.plot(kind="truth_comparison", truth=truth, freq=6.0)
        np.testing.assert_array_equal(
            np.ma.getdata(right.images[0].get_array()), conn.values[:, :, 2, 0]
        )
        np.testing.assert_array_equal(
            np.ma.getdata(left.images[0].get_array()), truth.values[:, :, 2, 0]
        )
        assert "6 Hz" in right.get_title()

    def test_nodes_that_differ_between_truth_and_estimate_are_refused(self, prov):
        sim = simulated_chain()
        other = Connectivity.from_array(
            np.array([[np.nan, 1.0], [0.0, np.nan]]), source=["P", "Q"], provenance=prov
        )
        with pytest.raises(ValueError, match="same nodes"):
            other.plot(kind="truth_comparison", truth=sim.truth)

    def test_the_same_nodes_in_another_order_are_refused(self, prov):
        sim = simulated_chain()
        names = sim.truth.source
        reordered = Connectivity.from_array(
            sim.truth.values[::-1, ::-1, 0, 0], source=names[::-1], provenance=prov
        )
        with pytest.raises(ValueError, match="same nodes"):
            reordered.plot(kind="truth_comparison", truth=sim.truth)

    def test_the_truth_is_required(self, prov):
        with pytest.raises(TypeError):
            pair_grid(prov).plot(kind="truth_comparison")

    def test_a_truth_on_other_axes_than_the_estimate_is_refused(self, prov):
        conn = spectrum(prov)
        broadband = Connectivity.from_array(
            np.where(np.isnan(conn.values[:, :, 0, 0]), np.nan, 1.0),
            source=conn.source,
            provenance=prov,
        )
        with pytest.raises(ValueError, match="truth_at"):
            conn.plot(kind="truth_comparison", truth=broadband, freq=4.0)

    def test_given_axes_must_be_a_pair(self, prov):
        sim = simulated_chain()
        estimate = Connectivity.from_array(
            np.where(np.isnan(sim.truth.values[:, :, 0, 0]), np.nan, 0.5),
            source=sim.truth.source,
            provenance=prov,
        )
        _, axes = plt.subplots(1, 2)
        left, right = estimate.plot(kind="truth_comparison", truth=sim.truth, ax=list(axes))
        assert (left, right) == tuple(axes)
        _, three = plt.subplots(1, 3)
        with pytest.raises(ValueError, match="two axes"):
            estimate.plot(kind="truth_comparison", truth=sim.truth, ax=list(three))
        _, one = plt.subplots()
        with pytest.raises(ValueError, match="two axes"):
            estimate.plot(kind="truth_comparison", truth=sim.truth, ax=one)


def labels(ticklabels):
    return [t.get_text() for t in ticklabels]

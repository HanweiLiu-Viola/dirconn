import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Annotation  # noqa: E402

from dirconn import Connectivity, Provenance  # noqa: E402

FREQS = np.array([4.0, 8.0, 12.0])
TIMES = np.array([0.0, 1.0, 2.0, 3.0])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def prov():
    return Provenance(method="adtf", parameters={}, model_order=2, seed=None)


def timefreq(prov):
    """Three nodes, 3 frequencies, 4 times; every (source, target, freq, time) value is distinct."""
    values = np.arange(3 * 3 * 3 * 4, dtype=float).reshape(3, 3, 3, 4)
    for k in range(3):
        values[k, k] = np.nan
    return Connectivity.from_array(
        values, source=["A", "B", "C"], freqs=FREQS, times=TIMES, provenance=prov
    )


def centres(edges):
    return (edges[:-1] + edges[1:]) / 2


class TestTimeFrequency:
    def test_rows_are_frequencies_columns_are_times_for_the_chosen_pair(self, prov):
        conn = timefreq(prov)
        ax = conn.plot(kind="time_frequency", source="A", target="C")
        mesh = ax.collections[0]
        coords = mesh.get_coordinates()
        np.testing.assert_allclose(centres(coords[:, 0, 1]), FREQS)  # rows: frequency
        np.testing.assert_allclose(centres(coords[0, :, 0]), TIMES)  # columns: time
        np.testing.assert_array_equal(
            np.ma.getdata(mesh.get_array()).reshape(3, 4), conn.values[0, 2]
        )  # source A, target C

    def test_the_reverse_pair_is_the_other_direction(self, prov):
        conn = timefreq(prov)
        ax = conn.plot(kind="time_frequency", source="C", target="A")
        np.testing.assert_array_equal(
            np.ma.getdata(ax.collections[0].get_array()).reshape(3, 4), conn.values[2, 0]
        )

    def test_the_axes_are_labelled_and_the_title_names_the_method_and_the_pair(self, prov):
        ax = timefreq(prov).plot(kind="time_frequency", source="A", target="B")
        assert ax.get_xlabel() == "time (s)"
        assert ax.get_ylabel() == "frequency (Hz)"
        assert "adtf" in ax.get_title()
        assert "A → B" in ax.get_title()

    def test_non_negative_values_use_a_sequential_scale_from_zero(self, prov):
        mesh = timefreq(prov).plot(kind="time_frequency", source="A", target="B").collections[0]
        assert mesh.norm.vmin == 0.0
        assert mesh.get_cmap().name == "Blues"

    def test_negative_values_use_a_diverging_scale_centred_on_zero(self, prov):
        conn = timefreq(prov)
        shifted = Connectivity.from_array(
            conn.values - 20.0, source=list(conn.source), freqs=FREQS, times=TIMES, provenance=prov
        )
        mesh = shifted.plot(kind="time_frequency", source="A", target="B").collections[0]
        assert mesh.norm.vmin == -mesh.norm.vmax
        assert mesh.get_cmap().name == "RdBu_r"

    def test_it_needs_both_a_frequency_and_a_time_axis(self, prov):
        only_freq = Connectivity.from_array(
            np.ones((2, 2, 3)), source=["A", "B"], freqs=FREQS, provenance=prov
        )
        with pytest.raises(ValueError, match="no time axis"):
            only_freq.plot(kind="time_frequency", source="A", target="B")
        only_time = Connectivity.from_array(
            np.ones((2, 2, 1, 4)), source=["A", "B"], times=TIMES, provenance=prov
        )
        with pytest.raises(ValueError, match="no frequency axis"):
            only_time.plot(kind="time_frequency", source="A", target="B")

    def test_a_single_time_point_or_frequency_is_refused_because_it_is_not_a_panel(self, prov):
        one_time = Connectivity.from_array(
            np.ones((2, 2, 3, 1)), source=["A", "B"], freqs=FREQS, times=[0.0], provenance=prov
        )
        with pytest.raises(ValueError, match="at least two"):
            one_time.plot(kind="time_frequency", source="A", target="B")

    def test_unknown_names_and_a_node_with_itself_are_refused(self, prov):
        conn = timefreq(prov)
        with pytest.raises(ValueError, match="'Q'"):
            conn.plot(kind="time_frequency", source="Q", target="A")
        with pytest.raises(ValueError, match="itself"):
            conn.plot(kind="time_frequency", source="A", target="A")

    def test_source_and_target_are_required_and_complex_values_are_refused(self, prov):
        with pytest.raises(TypeError):
            timefreq(prov).plot(kind="time_frequency")
        values = np.ones((2, 2, 3, 4), dtype=complex)
        conn = Connectivity.from_array(
            values, source=["A", "B"], freqs=FREQS, times=TIMES, provenance=prov
        )
        with pytest.raises(ValueError, match="Complex"):
            conn.plot(kind="time_frequency", source="A", target="B")

    def test_it_draws_on_the_given_axes_and_the_colourbar_can_be_switched_off(self, prov):
        fig, ax = plt.subplots()
        assert timefreq(prov).plot(kind="time_frequency", source="A", target="B", ax=ax) is ax
        assert len(fig.axes) == 2  # the panel and its colourbar
        fig2, ax2 = plt.subplots()
        timefreq(prov).plot(kind="time_frequency", source="A", target="B", ax=ax2, colorbar=False)
        assert len(fig2.axes) == 1


def square(prov, matrix):
    values = np.array(matrix, dtype=float)
    return Connectivity.from_array(
        values, source=["A", "B", "C", "D"][: len(values)], provenance=prov
    )


MATRIX = [[0, 0.8, 0.0], [0.0, 0, 0.4], [0.1, 0.0, 0]]  # A -> B strong, B -> C weaker, C -> A faint


def arrows(ax):
    return [t for t in ax.texts if isinstance(t, Annotation)]


def node_positions(ax):
    labels = {t.get_text(): t.get_position() for t in ax.texts if not isinstance(t, Annotation)}
    return labels


class TestNetworkGraph:
    def test_an_arrow_runs_from_the_source_to_the_target(self, prov):
        ax = square(prov, MATRIX).plot(kind="network_graph", threshold=0.3)
        found = arrows(ax)
        assert len(found) == 2  # A -> B and B -> C only
        positions = {name: np.asarray(pos) for name, pos in node_dots(ax).items()}
        heads = {tuple(np.round(a.xy, 6)): tuple(np.round(a.xyann, 6)) for a in found}
        # for every arrow, head = the target's dot, tail = the source's dot
        for source, target in (("A", "B"), ("B", "C")):
            assert tuple(np.round(positions[target], 6)) in heads
            assert heads[tuple(np.round(positions[target], 6))] == tuple(
                np.round(positions[source], 6)
            )

    def test_the_reverse_direction_is_not_drawn_unless_it_passes_the_threshold(self, prov):
        ax = square(prov, MATRIX).plot(kind="network_graph", threshold=0.3)
        positions = node_dots(ax)
        tails = [tuple(np.round(a.xyann, 6)) for a in arrows(ax)]
        assert tuple(np.round(positions["B"], 6)) in tails  # B -> C
        assert tuple(np.round(positions["C"], 6)) not in tails  # C -> A is below 0.3

    def test_a_value_equal_to_the_threshold_is_drawn(self, prov):
        ax = square(prov, MATRIX).plot(kind="network_graph", threshold=0.4)
        assert len(arrows(ax)) == 2  # 0.8 and exactly 0.4; the 0.1 is below

    def test_a_mask_cannot_draw_an_undefined_cell(self, prov):
        values = np.array(MATRIX, dtype=float)
        values[0, 2] = np.nan
        conn = Connectivity.from_array(values, source=["A", "B", "C"], provenance=prov)
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 2] = True  # undefined in the result
        mask[0, 1] = True
        assert len(arrows(conn.plot(kind="network_graph", mask=mask))) == 1

    def test_a_mask_chooses_the_edges_instead_of_a_threshold(self, prov):
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 0] = True  # C -> A only
        ax = square(prov, MATRIX).plot(kind="network_graph", mask=mask)
        positions = node_dots(ax)
        (only,) = arrows(ax)
        np.testing.assert_allclose(only.xy, positions["A"])
        np.testing.assert_allclose(only.xyann, positions["C"])

    def test_the_line_width_grows_with_the_value(self, prov):
        ax = square(prov, MATRIX).plot(kind="network_graph", threshold=0.05)
        widths = {}
        positions = node_dots(ax)
        for arrow in arrows(ax):
            for source, value in (("A", 0.8), ("B", 0.4), ("C", 0.1)):
                if np.allclose(arrow.xyann, positions[source]):
                    widths[value] = arrow.arrow_patch.get_linewidth()
        assert widths[0.8] > widths[0.4] > widths[0.1] > 0

    def test_the_nodes_sit_on_a_circle_clockwise_from_the_top_with_their_names(self, prov):
        ax = square(prov, MATRIX).plot(kind="network_graph", threshold=0.3)
        positions = node_dots(ax)
        assert list(positions) == ["A", "B", "C"]
        np.testing.assert_allclose(positions["A"], [0.0, 1.0], atol=1e-12)  # first node on top
        for xy in positions.values():
            assert np.hypot(*xy) == pytest.approx(1.0)
        assert positions["B"][0] > 0  # clockwise: the second node is on the right
        assert {t.get_text() for t in ax.texts if not isinstance(t, Annotation)} == {"A", "B", "C"}

    def test_the_title_says_which_edges_were_drawn(self, prov):
        ax = square(prov, MATRIX).plot(kind="network_graph", threshold=0.3)
        assert "0.3" in ax.get_title()
        masked = square(prov, MATRIX).plot(kind="network_graph", mask=np.eye(3, dtype=bool))
        assert "mask" in masked.get_title()

    def test_exactly_one_of_threshold_and_mask_is_required(self, prov):
        conn = square(prov, MATRIX)
        with pytest.raises(ValueError, match="threshold or mask"):
            conn.plot(kind="network_graph")
        with pytest.raises(ValueError, match="not both"):
            conn.plot(kind="network_graph", threshold=0.3, mask=np.ones((3, 3), dtype=bool))

    def test_the_mask_must_be_boolean_and_square_of_the_right_size(self, prov):
        conn = square(prov, MATRIX)
        with pytest.raises(ValueError, match="boolean"):
            conn.plot(kind="network_graph", mask=np.ones((3, 3)))
        with pytest.raises(ValueError, match="shape"):
            conn.plot(kind="network_graph", mask=np.ones((2, 2), dtype=bool))

    def test_the_diagonal_and_undefined_cells_never_get_an_arrow(self, prov):
        values = np.array(MATRIX, dtype=float)
        np.fill_diagonal(values, 5.0)  # a defined diagonal that would pass any threshold
        values[0, 2] = np.nan  # an undefined cell
        conn = Connectivity.from_array(values, source=["A", "B", "C"], provenance=prov)
        found = arrows(conn.plot(kind="network_graph", threshold=0.05))
        assert len(found) == 3  # A -> B, B -> C, C -> A
        assert not any(np.allclose(a.xy, a.xyann) for a in found)  # no self loop

    def test_a_result_with_different_sources_and_targets_is_refused(self, prov):
        rect = Connectivity.from_array(
            np.ones((2, 3)), source=["A", "B"], target=["X", "Y", "Z"], provenance=prov
        )
        with pytest.raises(ValueError, match="same nodes"):
            rect.plot(kind="network_graph", threshold=0.5)
        reordered = Connectivity.from_array(
            np.ones((2, 2)), source=["A", "B"], target=["B", "A"], provenance=prov
        )
        with pytest.raises(ValueError, match="same nodes"):
            reordered.plot(kind="network_graph", threshold=0.5)

    def test_a_spectrum_needs_a_frequency_choice_like_the_matrix_plot(self, prov):
        values = np.ones((2, 2, 3))
        conn = Connectivity.from_array(values, source=["A", "B"], freqs=FREQS, provenance=prov)
        with pytest.raises(ValueError, match="freq"):
            conn.plot(kind="network_graph", threshold=0.5)
        ax = conn.plot(kind="network_graph", threshold=0.5, freq=8.0)
        assert "8 Hz" in ax.get_title()

    def test_it_draws_on_the_given_axes(self, prov):
        fig, ax = plt.subplots()
        assert square(prov, MATRIX).plot(kind="network_graph", threshold=0.3, ax=ax) is ax
        assert plt.get_fignums() == [fig.number]


def node_dots(ax):
    """Node name -> position, read from the node dots (the collection) and the label order."""
    dots = np.asarray(ax.collections[0].get_offsets())
    names = [t.get_text() for t in ax.texts if not isinstance(t, Annotation)]
    return {name: dots[k] for k, name in enumerate(names)}

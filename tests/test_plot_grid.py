import sys

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dirconn import Connectivity, Provenance  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def prov():
    return Provenance(method="gc", parameters={"order": 2}, model_order=2, seed=0)


def rectangular(prov, values=None):
    """Two sources, three targets: a swap of the axes changes the shape, not only the numbers."""
    if values is None:
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return Connectivity.from_array(
        values, source=["A", "B"], target=["X", "Y", "Z"], provenance=prov
    )


def square(prov, values=None):
    if values is None:
        values = np.array([[0.0, 0.9, 0.1], [0.2, 0.0, 0.7], [0.3, 0.4, 0.0]])
    values = values.copy()
    np.fill_diagonal(values, np.nan)
    return Connectivity.from_array(values, source=["A", "B", "C"], provenance=prov)


def spectrum(prov, freqs=(4.0, 8.0, 12.0, 16.0)):
    freqs = np.asarray(freqs, dtype=float)
    values = np.arange(2 * 2 * len(freqs), dtype=float).reshape(2, 2, len(freqs))
    values[[0, 1], [0, 1]] = np.nan  # undefined diagonal, as in a Granger result
    return Connectivity.from_array(values, source=["A", "B"], freqs=freqs, provenance=prov)


def moving(prov, times=(0.0, 0.5, 1.0)):
    times = np.asarray(times, dtype=float)
    values = np.arange(2 * 2 * len(times), dtype=float).reshape(2, 2, 1, len(times))
    values[[0, 1], [0, 1]] = np.nan
    return Connectivity.from_array(values, source=["A", "B"], times=times, provenance=prov)


def drawn(ax):
    return ax.images[0].get_array()


def labels(axis_ticklabels):
    return [t.get_text() for t in axis_ticklabels]


class TestOrientation:
    def test_row_i_is_source_i_and_column_j_is_target_j(self, prov):
        conn = rectangular(prov)
        ax = conn.plot()
        image = np.ma.getdata(drawn(ax))
        assert image.shape == (2, 3)
        np.testing.assert_array_equal(image, conn.values[:, :, 0, 0])
        # source A to target Z is the top right cell: value 3
        assert image[0, 2] == 3.0

    def test_the_tick_labels_name_the_sources_on_rows_and_the_targets_on_columns(self, prov):
        ax = rectangular(prov).plot()
        assert labels(ax.get_yticklabels()) == ["A", "B"]
        assert labels(ax.get_xticklabels()) == ["X", "Y", "Z"]
        assert ax.get_ylabel() == "source"
        assert ax.get_xlabel() == "target"

    def test_row_zero_is_drawn_at_the_top(self, prov):
        ax = rectangular(prov).plot()
        assert ax.get_ylim()[0] > ax.get_ylim()[1]  # imshow default: row 0 on top


class TestUndefinedCells:
    def test_the_nan_diagonal_is_left_empty_and_nothing_else_is(self, prov):
        ax = square(prov).plot()
        mask = np.ma.getmaskarray(drawn(ax))
        np.testing.assert_array_equal(mask, np.eye(3, dtype=bool))

    def test_a_diagonal_the_estimator_defines_is_left_empty_and_out_of_the_scale(self, prov):
        values = np.array([[0.9, 0.2, 0.1], [0.3, 0.8, 0.4], [0.1, 0.2, 0.7]])
        conn = Connectivity.from_array(values, source=["A", "B", "C"], provenance=prov)
        ax = conn.plot()
        np.testing.assert_array_equal(np.ma.getmaskarray(drawn(ax)), np.eye(3, dtype=bool))
        assert ax.images[0].norm.vmax == pytest.approx(0.4)  # the largest off-diagonal value

    def test_the_diagonal_of_a_rectangular_result_is_not_special(self, prov):
        ax = rectangular(prov).plot()
        assert not np.ma.getmaskarray(drawn(ax)).any()

    def test_empty_cells_are_transparent_not_a_colour(self, prov):
        ax = square(prov).plot()
        cmap = ax.images[0].get_cmap()
        assert cmap(np.ma.masked)[3] == 0.0  # the "bad" colour has alpha 0

    def test_empty_cells_are_grey_so_they_differ_from_a_cell_whose_value_is_zero(self, prov):
        ax = square(prov, np.array([[0.0, 0.0, 0.3], [0.2, 0.0, 0.0], [0.1, 0.4, 0.0]])).plot()
        image = ax.images[0]
        zero_cell = image.get_cmap()(image.norm(0.0))
        assert matplotlib.colors.to_rgba(ax.get_facecolor()) != zero_cell

    def test_a_result_with_nothing_defined_is_an_error(self, prov):
        conn = rectangular(prov, np.full((2, 3), np.nan))
        with pytest.raises(ValueError, match="no defined value"):
            conn.plot()


class TestColourScale:
    def test_non_negative_values_use_a_sequential_scale_from_zero(self, prov):
        ax = square(prov).plot()
        norm = ax.images[0].norm
        assert norm.vmin == 0.0
        assert norm.vmax == pytest.approx(0.9)
        assert ax.images[0].get_cmap().name == "Blues"

    def test_negative_values_use_a_diverging_scale_centred_on_zero(self, prov):
        values = np.array([[0.0, -0.2, 0.5], [0.1, 0.0, -0.1], [0.3, 0.4, 0.0]])
        ax = square(prov, values).plot()
        image = ax.images[0]
        assert image.norm.vmin == -image.norm.vmax
        assert image.norm.vmax == pytest.approx(0.5)
        assert image.get_cmap().name == "RdBu_r"
        assert image.norm(0.0) == pytest.approx(0.5)  # zero is the middle colour

    def test_a_given_vmax_is_the_half_range_of_a_diverging_scale(self, prov):
        values = np.array([[0.0, -0.2, 0.5], [0.1, 0.0, -0.1], [0.3, 0.4, 0.0]])
        ax = square(prov, values).plot(vmax=1.0)
        assert ax.images[0].norm.vmin == -1.0
        assert ax.images[0].norm.vmax == 1.0

    def test_a_vmin_that_would_move_the_centre_is_refused(self, prov):
        values = np.array([[0.0, -0.2, 0.5], [0.1, 0.0, -0.1], [0.3, 0.4, 0.0]])
        with pytest.raises(ValueError, match="centred"):
            square(prov, values).plot(vmin=-0.2, vmax=1.0)

    def test_limits_can_be_fixed_for_non_negative_values(self, prov):
        ax = square(prov).plot(vmin=0.0, vmax=2.0)
        assert (ax.images[0].norm.vmin, ax.images[0].norm.vmax) == (0.0, 2.0)

    def test_a_colour_map_can_be_chosen(self, prov):
        assert square(prov).plot(cmap="magma").images[0].get_cmap().name == "magma"


class TestAxesAndFigure:
    def test_it_returns_the_axes_it_drew_on(self, prov):
        fig, ax = plt.subplots()
        assert square(prov).plot(ax=ax) is ax
        assert plt.get_fignums() == [fig.number]  # no second figure was opened

    def test_a_figure_it_makes_has_a_layout_that_keeps_the_labels_apart(self, prov):
        assert square(prov).plot().figure.get_layout_engine() is not None

    def test_a_colourbar_is_added_unless_switched_off(self, prov):
        with_bar = square(prov).plot()
        assert len(with_bar.figure.axes) == 2
        without = square(prov).plot(colorbar=False)
        assert len(without.figure.axes) == 1

    def test_the_title_names_the_method(self, prov):
        assert "gc" in square(prov).plot().get_title()

    def test_plotting_does_not_change_the_result(self, prov):
        conn = square(prov)
        before = conn.values.copy()
        conn.plot()
        np.testing.assert_array_equal(conn.values, before)


class TestFrequencyAndTimeAxes:
    def test_a_frequency_resolved_result_needs_a_choice(self, prov):
        with pytest.raises(ValueError, match="freq.*band|band.*freq"):
            spectrum(prov).plot()

    def test_freq_takes_the_nearest_bin_and_says_which(self, prov):
        conn = spectrum(prov)
        ax = conn.plot(freq=8.9)
        np.testing.assert_array_equal(
            np.ma.getdata(drawn(ax)), conn.values[:, :, 1, 0]
        )  # 8 Hz is nearer than 12 Hz
        assert "8 Hz" in ax.get_title()

    def test_freq_rounds_up_when_the_upper_bin_is_nearer(self, prov):
        conn = spectrum(prov)
        ax = conn.plot(freq=11.0)  # 12 Hz is nearer than 8 Hz; a floor would give 8 Hz
        np.testing.assert_array_equal(np.ma.getdata(drawn(ax)), conn.values[:, :, 2, 0])
        assert "12 Hz" in ax.get_title()

    def test_freq_outside_the_axis_is_refused_not_clipped(self, prov):
        with pytest.raises(ValueError, match="outside"):
            spectrum(prov).plot(freq=40.0)

    def test_band_is_the_mean_over_the_band(self, prov):
        conn = spectrum(prov, freqs=(4.0, 8.0, 10.0, 12.0, 16.0))
        ax = conn.plot(band="alpha")
        expected = conn.reduce_freq("alpha").values[:, :, 0, 0]
        np.testing.assert_allclose(np.ma.getdata(drawn(ax)), expected)
        assert "alpha" in ax.get_title()

    def test_freq_and_band_together_are_refused(self, prov):
        with pytest.raises(ValueError, match="not both"):
            spectrum(prov).plot(freq=8.0, band="alpha")

    def test_a_result_without_a_frequency_axis_refuses_freq_and_band(self, prov):
        with pytest.raises(ValueError, match="frequency"):
            square(prov).plot(freq=8.0)
        with pytest.raises(ValueError, match="frequency"):
            square(prov).plot(band="alpha")

    def test_a_time_resolved_result_needs_a_choice(self, prov):
        with pytest.raises(ValueError, match="time"):
            moving(prov).plot()

    def test_time_takes_the_nearest_point_and_says_which(self, prov):
        conn = moving(prov)
        ax = conn.plot(time=0.6)
        np.testing.assert_array_equal(np.ma.getdata(drawn(ax)), conn.values[:, :, 0, 1])
        assert "0.5 s" in ax.get_title()

    def test_time_rounds_up_when_the_later_point_is_nearer(self, prov):
        conn = moving(prov)
        ax = conn.plot(time=0.9)  # 1.0 s is nearer than 0.5 s
        np.testing.assert_array_equal(np.ma.getdata(drawn(ax)), conn.values[:, :, 0, 2])
        assert "1 s" in ax.get_title()

    def test_time_outside_the_axis_is_refused(self, prov):
        with pytest.raises(ValueError, match="outside"):
            moving(prov).plot(time=5.0)

    def test_a_result_without_a_time_axis_refuses_time(self, prov):
        with pytest.raises(ValueError, match="time"):
            square(prov).plot(time=1.0)


class TestRefusals:
    def test_complex_values_are_refused(self, prov):
        conn = Connectivity.from_array(
            np.array([[0, 1 + 1j], [1j, 0]]), source=["A", "B"], provenance=prov
        )
        with pytest.raises(ValueError, match="Complex"):
            conn.plot()

    def test_an_unknown_kind_lists_the_known_ones(self, prov):
        with pytest.raises(ValueError, match="connectivity_grid"):
            square(prov).plot(kind="pie_chart")

    def test_without_matplotlib_the_error_names_the_extra(self, prov, monkeypatch):
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        with pytest.raises(ImportError, match=r"dirconn\[plot\]"):
            square(prov).plot()

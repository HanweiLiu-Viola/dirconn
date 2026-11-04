import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dirconn import Connectivity, Provenance, viz

mne = pytest.importorskip("mne")


@pytest.fixture
def prov():
    return Provenance(method="gc", parameters={}, model_order=2, seed=None)


def label(name, vertices, hemi):
    return mne.Label(vertices=np.asarray(vertices), hemi=hemi, name=name, subject="fsaverage")


def networks_lh():
    """Seven networks and the medial wall on a 100-vertex left hemisphere."""
    spans = {f"7Networks_{k}-lh": range(10 * (k - 1), 10 * k) for k in range(1, 8)}
    spans["FreeSurfer_Defined_Medial_Wall-lh"] = range(70, 100)
    return [label(name, list(v), "lh") for name, v in spans.items()]


class TestDominantNetworks:
    def test_a_region_gets_the_network_it_overlaps_most_and_the_share_of_its_vertices(self):
        regions = [
            label("a-lh", [10, 11, 12, 13, 25], "lh")
        ]  # four vertices in network 2, one in 3
        table = viz.dominant_networks(regions, networks_lh())
        assert table.loc["a-lh", "network"] == "Somatomotor"
        assert table.loc["a-lh", "share"] == pytest.approx(4 / 5)
        assert bool(table.loc["a-lh", "dominant"])

    def test_vertices_on_the_medial_wall_count_against_the_share_but_never_win(self):
        regions = [label("b-lh", [0, 1, 70, 71, 72, 73], "lh")]  # 2 in network 1, 4 on the wall
        table = viz.dominant_networks(regions, networks_lh())
        assert table.loc["b-lh", "network"] == "Visual"
        assert table.loc["b-lh", "share"] == pytest.approx(2 / 6)
        assert not bool(table.loc["b-lh", "dominant"])  # below one half: flagged, not hidden

    def test_exactly_half_is_still_dominant(self):
        regions = [label("c-lh", [0, 1, 10, 11], "lh")]  # 2 in network 1, 2 in network 2
        table = viz.dominant_networks(regions, networks_lh())
        assert table.loc["c-lh", "share"] == pytest.approx(0.5)
        assert bool(table.loc["c-lh", "dominant"])
        assert table.loc["c-lh", "network"] == "Visual"  # a tie goes to the first network

    def test_a_region_is_compared_only_with_the_networks_of_its_own_hemisphere(self):
        # on the right, network 1 sits where the left hemisphere has network 7
        rh_networks = [
            label(f"7Networks_{k}-rh", list(range(10 * (7 - k), 10 * (8 - k))), "rh")
            for k in range(1, 8)
        ]
        regions = [label("d-rh", [60, 61, 62], "rh")]
        table = viz.dominant_networks(regions, networks_lh() + rh_networks)
        assert table.loc["d-rh", "network"] == "Visual"  # not the left network 7, "Default"

    def test_a_region_without_any_network_vertex_has_no_network_and_is_not_dominant(self):
        regions = [label("e-lh", [80, 81], "lh")]  # only the medial wall
        table = viz.dominant_networks(regions, networks_lh())
        assert table.loc["e-lh", "network"] is None
        assert table.loc["e-lh", "share"] == 0.0
        assert not bool(table.loc["e-lh", "dominant"])

    def test_the_table_is_indexed_by_region_name_in_the_order_given(self):
        regions = [label("z-lh", [0], "lh"), label("a-lh", [10], "lh")]
        assert list(viz.dominant_networks(regions, networks_lh()).index) == ["z-lh", "a-lh"]

    def test_the_seven_network_names_are_the_published_ones(self):
        assert list(viz.YEO_NETWORKS.values()) == [
            "Visual",
            "Somatomotor",
            "Dorsal Attention",
            "Ventral Attention",
            "Limbic",
            "Frontoparietal",
            "Default",
        ]


NAMES = ["a1", "a2", "b1", "b2", "c1"]
GROUPS = {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "c1": "C"}


def five(prov):
    values = np.arange(25, dtype=float).reshape(5, 5)
    np.fill_diagonal(values, np.nan)
    return Connectivity.from_array(values, source=NAMES, provenance=prov)


class TestGroupMatrix:
    def test_the_mean_over_the_node_pairs_of_two_groups(self, prov):
        conn = five(prov)
        table = viz.group_matrix(conn, GROUPS)
        v = conn.values[:, :, 0, 0]
        assert table.loc["A", "B"] == pytest.approx(np.mean([v[0, 2], v[0, 3], v[1, 2], v[1, 3]]))
        assert table.loc["B", "A"] == pytest.approx(np.mean([v[2, 0], v[2, 1], v[3, 0], v[3, 1]]))
        assert table.loc["A", "C"] == pytest.approx(np.mean([v[0, 4], v[1, 4]]))

    def test_source_rows_and_target_columns_are_not_swapped(self, prov):
        values = np.zeros((5, 5))
        values[0, 2] = 8.0  # a1 -> b1 only
        np.fill_diagonal(values, np.nan)
        conn = Connectivity.from_array(values, source=NAMES, provenance=prov)
        table = viz.group_matrix(conn, GROUPS, how="sum")
        assert table.loc["A", "B"] == 8.0
        assert table.loc["B", "A"] == 0.0

    def test_within_a_group_only_distinct_node_pairs_count(self, prov):
        conn = five(prov)
        v = conn.values[:, :, 0, 0]
        table = viz.group_matrix(conn, GROUPS, how="sum")
        assert table.loc["A", "A"] == pytest.approx(v[0, 1] + v[1, 0])  # not the NaN diagonal

    def test_a_diagonal_the_estimator_defines_is_not_counted(self, prov):
        values = np.ones((5, 5))
        np.fill_diagonal(values, 100.0)
        conn = Connectivity.from_array(values, source=NAMES, provenance=prov)
        table = viz.group_matrix(conn, GROUPS, how="sum")
        assert table.loc["A", "A"] == 2.0  # a1 -> a2 and a2 -> a1, not the two 100s

    def test_a_group_with_one_node_has_no_within_group_value(self, prov):
        assert np.isnan(viz.group_matrix(five(prov), GROUPS).loc["C", "C"])

    def test_the_groups_appear_in_the_order_of_their_first_node(self, prov):
        groups = {"a1": "Z", "a2": "Z", "b1": "M", "b2": "M", "c1": "A"}
        assert list(viz.group_matrix(five(prov), groups).index) == ["Z", "M", "A"]

    def test_a_node_without_a_group_is_named_not_dropped(self, prov):
        with pytest.raises(ValueError, match="c1"):
            viz.group_matrix(five(prov), {k: v for k, v in GROUPS.items() if k != "c1"})

    def test_results_that_are_not_square_or_not_real_are_refused(self, prov):
        rect = Connectivity.from_array(
            np.ones((2, 3)), source=["a1", "a2"], target=["b1", "b2", "c1"], provenance=prov
        )
        with pytest.raises(ValueError, match="same nodes"):
            viz.group_matrix(rect, GROUPS)
        reordered = Connectivity.from_array(
            np.ones((2, 2)), source=["a1", "a2"], target=["a2", "a1"], provenance=prov
        )
        with pytest.raises(ValueError, match="same nodes"):
            viz.group_matrix(reordered, GROUPS)
        complex_ = Connectivity.from_array(
            np.ones((2, 2), dtype=complex), source=["a1", "a2"], provenance=prov
        )
        with pytest.raises(ValueError, match="Complex"):
            viz.group_matrix(complex_, GROUPS)

    def test_how_must_be_mean_or_sum(self, prov):
        with pytest.raises(ValueError, match="how"):
            viz.group_matrix(five(prov), GROUPS, how="median")

    def test_a_spectrum_needs_the_same_choice_as_the_matrix_plot(self, prov):
        values = np.ones((5, 5, 3))
        conn = Connectivity.from_array(
            values, source=NAMES, freqs=[4.0, 8.0, 12.0], provenance=prov
        )
        with pytest.raises(ValueError, match="freq"):
            viz.group_matrix(conn, GROUPS)
        table = viz.group_matrix(conn, GROUPS, freq=8.0)
        assert isinstance(table, pd.DataFrame)


@pytest.fixture
def pycirclize():
    return pytest.importorskip("pycirclize")


@pytest.fixture(autouse=False)
def agg():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def texts(fig):
    return {t.get_text() for ax in fig.axes for t in ax.texts} | {t.get_text() for t in fig.texts}


def circos_matrix(monkeypatch, pycirclize):
    """Capture the matrix handed to pyCirclize."""
    seen = {}
    original = pycirclize.Circos.chord_diagram

    def spy(matrix, **kwargs):
        seen["matrix"] = matrix.copy()
        seen["kwargs"] = kwargs
        return original(matrix, **kwargs)

    monkeypatch.setattr(pycirclize.Circos, "chord_diagram", staticmethod(spy))
    return seen


@pytest.mark.usefixtures("agg")
class TestChordDiagram:
    def test_the_group_matrix_is_what_pycirclize_draws_without_a_diagonal(
        self, prov, monkeypatch, pycirclize
    ):
        seen = circos_matrix(monkeypatch, pycirclize)
        conn = five(prov)
        viz.chord_diagram(conn, GROUPS)
        table = viz.group_matrix(conn, GROUPS)
        filled = np.nan_to_num(table.to_numpy(dtype=float, copy=True))
        np.fill_diagonal(filled, 0.0)
        expected = pd.DataFrame(filled, index=table.index, columns=table.columns)
        pd.testing.assert_frame_equal(seen["matrix"], expected)

    def test_the_links_are_directed(self, prov, monkeypatch, pycirclize):
        seen = circos_matrix(monkeypatch, pycirclize)
        viz.chord_diagram(five(prov), GROUPS)
        assert seen["kwargs"]["link_kws"]["direction"] == 1

    def test_the_figure_names_every_group(self, prov, pycirclize):
        fig = viz.chord_diagram(five(prov), GROUPS)
        assert {"A", "B", "C"} <= texts(fig)

    def test_the_net_direction_keeps_only_the_stronger_direction_of_each_pair(
        self, prov, monkeypatch, pycirclize
    ):
        seen = circos_matrix(monkeypatch, pycirclize)
        values = np.zeros((5, 5))
        values[0, 2] = 6.0  # A -> B
        values[2, 0] = 2.0  # B -> A, weaker
        values[4, 0] = 3.0  # C -> A
        np.fill_diagonal(values, np.nan)
        conn = Connectivity.from_array(values, source=NAMES, provenance=prov)
        viz.chord_diagram(conn, GROUPS, measure="net_direction", how="sum")
        m = seen["matrix"]
        assert m.loc["A", "B"] == pytest.approx(4.0)  # 6 - 2
        assert m.loc["B", "A"] == 0.0
        assert m.loc["C", "A"] == pytest.approx(3.0)
        assert m.loc["A", "C"] == 0.0

    def test_a_net_direction_figure_carries_the_fixed_caption(self, prov, pycirclize):
        fig = viz.chord_diagram(five(prov), GROUPS, measure="net_direction")
        assert viz.GUARD_CAPTION in texts(fig)

    def test_a_plain_values_figure_does_not_carry_the_source_sink_caption(self, prov, pycirclize):
        assert viz.GUARD_CAPTION not in texts(viz.chord_diagram(five(prov), GROUPS))

    def test_the_caption_says_predictive_not_causal_and_no_leakage_correction(self):
        assert "not causal" in viz.GUARD_CAPTION
        assert "leakage" in viz.GUARD_CAPTION

    def test_values_must_be_non_negative_for_the_values_measure(self, prov, pycirclize):
        with pytest.raises(ValueError, match="non-negative"):
            viz.chord_diagram(five(prov).net_direction(), GROUPS)

    def test_nothing_to_draw_is_an_error(self, prov, pycirclize):
        zero = np.zeros((5, 5))
        np.fill_diagonal(zero, np.nan)
        conn = Connectivity.from_array(zero, source=NAMES, provenance=prov)
        with pytest.raises(ValueError, match="nothing to draw"):
            viz.chord_diagram(conn, GROUPS)

    def test_the_measure_must_be_known(self, prov, pycirclize):
        with pytest.raises(ValueError, match="measure"):
            viz.chord_diagram(five(prov), GROUPS, measure="other")

    def test_without_pycirclize_the_error_names_the_extra(self, prov, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "pycirclize", None)
        with pytest.raises(ImportError, match=r"dirconn\[plot\]"):
            viz.chord_diagram(five(prov), GROUPS)


class TestRegionValues:
    def test_a_mapping_a_series_and_a_labelled_array_give_the_same_values(self):
        expected = {"bankssts-lh": 0.5, "insula-rh": -0.25}
        assert viz.region_values(expected) == expected
        assert viz.region_values(pd.Series(expected)) == expected
        array = xr.DataArray(
            [[[0.5]], [[-0.25]]],
            dims=("node", "freq", "time"),
            coords={"node": ["bankssts-lh", "insula-rh"], "freq": [np.nan], "time": [np.nan]},
        )
        assert viz.region_values(array) == expected

    def test_undefined_values_are_left_out_not_drawn_as_zero(self):
        assert viz.region_values({"bankssts-lh": np.nan, "insula-rh": 1.0}) == {"insula-rh": 1.0}

    def test_an_array_with_several_frequencies_or_times_is_refused(self):
        array = xr.DataArray(
            np.ones((2, 3, 1)),
            dims=("node", "freq", "time"),
            coords={
                "node": ["bankssts-lh", "insula-rh"],
                "freq": [1.0, 2.0, 3.0],
                "time": [np.nan],
            },
        )
        with pytest.raises(ValueError, match="one frequency"):
            viz.region_values(array)

    def test_a_name_that_is_not_a_region_with_a_hemisphere_is_named(self):
        with pytest.raises(ValueError, match="insula"):
            viz.region_values({"insula": 1.0})

    def test_nothing_defined_is_an_error(self):
        with pytest.raises(ValueError, match="no defined"):
            viz.region_values({"insula-rh": np.nan})


class TestMapInput:
    def test_all_zero_values_and_a_scale_that_is_not_positive_are_refused(self):
        with pytest.raises(ValueError, match="nothing to draw"):
            viz.degree_imbalance_map({"bankssts-lh": 0.0})
        with pytest.raises(ValueError, match="nothing to draw"):
            viz.degree_imbalance_map({"bankssts-lh": 1.0}, vmax=-1.0)


@pytest.mark.sample_data
class TestWithSampleData:
    def test_every_region_of_the_parcellation_gets_a_network_and_the_weak_ones_are_flagged(self):
        table = viz.yeo_networks()
        assert len(table) == 68
        assert set(table["network"].dropna()) <= set(viz.YEO_NETWORKS.values())
        assert table["share"].between(0, 1).all()
        assert (~table["dominant"]).sum() == (table["share"] < 0.5).sum()
        assert "bankssts-lh" in table.index

    def test_groups_for_a_few_regions(self):
        groups = viz.yeo_groups(["bankssts-lh", "insula-rh"])
        assert set(groups) == {"bankssts-lh", "insula-rh"}
        assert set(groups.values()) <= set(viz.YEO_NETWORKS.values())

    def test_a_name_outside_the_parcellation_is_refused(self):
        with pytest.raises(ValueError, match="notaregion-lh"):
            viz.yeo_groups(["notaregion-lh"])

    def test_a_name_that_is_not_a_region_is_refused_before_anything_is_drawn(self):
        with pytest.raises(ValueError, match="notaregion-lh"):
            viz.degree_imbalance_map({"notaregion-lh": 1.0}, backend="pyvistaqt")

    def test_the_surface_map_carries_the_caption_and_can_be_drawn(self, monkeypatch):
        recorded = []
        original = mne.viz.Brain.add_text

        def spy(self, x, y, text, *args, **kwargs):
            recorded.append(text)
            return original(self, x, y, text, *args, **kwargs)

        monkeypatch.setattr(mne.viz.Brain, "add_text", spy)
        values = {"bankssts-lh": 1.0, "insula-rh": -1.0, "precuneus-lh": 0.5}
        brain = viz.degree_imbalance_map(
            values, backend="pyvistaqt", degrees="edges with value >= 0.1"
        )
        try:
            assert viz.GUARD_CAPTION in recorded
            assert any(
                "edges with value >= 0.1" in text for text in recorded
            )  # the choice is stated
            image = brain.screenshot()
            assert image.ndim == 3 and image.shape[2] in (3, 4)
        finally:
            brain.close()

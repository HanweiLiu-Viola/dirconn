import json
import warnings

import numpy as np
import pytest
import xarray as xr

from dirconn import Connectivity, Provenance


@pytest.fixture
def prov():
    return Provenance(method="gc", parameters={"order": 2}, model_order=2, seed=0)


def square(prov, matrix, **axes):
    values = np.array(matrix, dtype=float)
    if values.ndim == 2:
        np.fill_diagonal(values, np.nan)
    else:
        for k in range(values.shape[0]):
            values[k, k] = np.nan
    return Connectivity.from_array(
        values, source=["A", "B", "C"][: values.shape[0]], provenance=prov, **axes
    )


CHAIN = [[0, 1.0, 0], [0, 0, 1.0], [0, 0, 0]]  # A -> B -> C
WEIGHTED = [[0, 0.8, 0.6], [0.1, 0, 0.5], [0.2, 0.3, 0]]


class TestNetDirection:
    def test_a_chain_points_from_the_driver_to_the_driven(self, prov):
        net = square(prov, CHAIN).net_direction()
        v = net.values[:, :, 0, 0]
        assert v[0, 1] == 1.0  # A -> B stronger than B -> A
        assert v[1, 0] == -1.0
        assert v[1, 2] == 1.0
        assert v[0, 2] == 0.0  # nothing between A and C

    def test_it_is_the_difference_of_the_two_directions_for_every_pair(self, prov):
        conn = square(prov, WEIGHTED)
        v = conn.net_direction().values[:, :, 0, 0]
        w = conn.values[:, :, 0, 0]
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert v[i, j] == pytest.approx(w[i, j] - w[j, i])

    def test_it_is_antisymmetric_with_an_undefined_diagonal(self, prov):
        v = square(prov, WEIGHTED).net_direction().values[:, :, 0, 0]
        off = ~np.eye(3, dtype=bool)
        np.testing.assert_allclose(v[off], -v.T[off])
        assert np.isnan(np.diag(v)).all()

    def test_a_defined_diagonal_in_the_input_is_still_undefined_in_the_output(self, prov):
        conn = Connectivity.from_array(
            np.array(WEIGHTED, dtype=float) + np.eye(3), source=["A", "B", "C"], provenance=prov
        )
        assert np.isnan(np.diag(conn.net_direction().values[:, :, 0, 0])).all()

    def test_swapping_source_and_target_flips_the_sign(self, prov):
        a = square(prov, WEIGHTED)
        b = square(prov, np.array(WEIGHTED).T)
        np.testing.assert_allclose(
            a.net_direction().values[:, :, 0, 0],
            -b.net_direction().values[:, :, 0, 0],
            equal_nan=True,
        )

    def test_the_axes_are_kept_for_a_spectrum(self, prov):
        values = np.arange(3 * 3 * 4, dtype=float).reshape(3, 3, 4)
        conn = Connectivity.from_array(
            values, source=["A", "B", "C"], freqs=[2.0, 4.0, 6.0, 8.0], provenance=prov
        )
        net = conn.net_direction()
        assert net.values.shape == (3, 3, 4, 1)
        np.testing.assert_array_equal(net.freqs, conn.freqs)
        assert net.values[0, 1, 2, 0] == values[0, 1, 2] - values[1, 0, 2]  # frequency by frequency

    def test_an_undefined_cell_makes_both_directions_of_the_pair_undefined(self, prov):
        values = np.array(WEIGHTED, dtype=float)
        values[0, 1] = np.nan
        conn = Connectivity.from_array(values, source=["A", "B", "C"], provenance=prov)
        v = conn.net_direction().values[:, :, 0, 0]
        assert np.isnan(v[0, 1]) and np.isnan(v[1, 0])
        assert np.isfinite(v[0, 2]) and np.isfinite(v[1, 2])

    def test_the_provenance_gets_one_step_and_the_original_is_untouched(self, prov):
        conn = square(prov, WEIGHTED)
        net = conn.net_direction()
        assert net.provenance.history[-1]["op"] == "net_direction"
        assert len(net.provenance.history) == len(conn.provenance.history) + 1
        assert net.provenance.method == "gc"
        assert conn.provenance.history == ()

    def test_results_that_are_not_square_or_not_real_are_refused(self, prov):
        rect = Connectivity.from_array(
            np.ones((2, 3)), source=["A", "B"], target=["X", "Y", "Z"], provenance=prov
        )
        with pytest.raises(ValueError, match="same nodes"):
            rect.net_direction()
        reordered = Connectivity.from_array(
            np.ones((2, 2)), source=["A", "B"], target=["B", "A"], provenance=prov
        )
        with pytest.raises(ValueError, match="same nodes"):
            reordered.net_direction()
        complex_ = Connectivity.from_array(
            np.ones((2, 2), dtype=complex), source=["A", "B"], provenance=prov
        )
        with pytest.raises(ValueError, match="Complex"):
            complex_.net_direction()


def expected_imbalance(matrix):
    m = np.array(matrix, dtype=float)
    np.fill_diagonal(m, 0.0)
    return np.log(m.sum(axis=1)) - np.log(m.sum(axis=0))  # out of row i, into column i


class TestDegreeImbalance:
    def test_it_is_log_out_degree_minus_log_in_degree(self, prov):
        result = square(prov, WEIGHTED).degree_imbalance()
        np.testing.assert_allclose(result.values[:, 0, 0], expected_imbalance(WEIGHTED))
        # node A sends 1.4 and receives 0.3: a source (positive); C sends 0.5, receives 1.1: a sink
        assert result.values[0, 0, 0] > 0 > result.values[2, 0, 0]

    def test_it_is_returned_as_a_labelled_array_with_the_result_axes(self, prov):
        result = square(prov, WEIGHTED).degree_imbalance()
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("node", "freq", "time")
        assert list(result.coords["node"].values) == ["A", "B", "C"]
        assert result.name == "degree_imbalance"

    def test_a_node_that_is_the_mirror_of_another_has_the_opposite_sign(self, prov):
        a = square(prov, WEIGHTED).degree_imbalance().values
        b = square(prov, np.array(WEIGHTED).T).degree_imbalance().values
        np.testing.assert_allclose(a, -b)

    def test_the_diagonal_is_not_counted(self, prov):
        with_diagonal = np.array(WEIGHTED, dtype=float)
        np.fill_diagonal(with_diagonal, 5.0)
        conn = Connectivity.from_array(with_diagonal, source=["A", "B", "C"], provenance=prov)
        np.testing.assert_allclose(
            conn.degree_imbalance().values[:, 0, 0], expected_imbalance(WEIGHTED)
        )

    def test_a_zero_degree_gives_nan_without_a_warning(self, prov):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = square(prov, CHAIN).degree_imbalance()
        # A sends but receives nothing, C receives but sends nothing, B is balanced
        assert np.isnan(result.values[0, 0, 0]) and np.isnan(result.values[2, 0, 0])
        assert result.values[1, 0, 0] == 0.0

    def test_an_undefined_cell_is_left_out_of_the_degree(self, prov):
        values = np.array(WEIGHTED, dtype=float)
        values[0, 1] = np.nan
        conn = Connectivity.from_array(values, source=["A", "B", "C"], provenance=prov)
        reference = np.array(WEIGHTED, dtype=float)
        reference[0, 1] = 0.0
        np.testing.assert_allclose(
            conn.degree_imbalance().values[:, 0, 0], expected_imbalance(reference)
        )

    def test_a_spectrum_gives_one_value_per_node_and_frequency(self, prov):
        base = np.array(WEIGHTED, dtype=float)
        values = np.stack([base, base * 2.0, base * 3.0], axis=-1)  # scaling leaves the ratio alone
        conn = Connectivity.from_array(
            values, source=["A", "B", "C"], freqs=[2.0, 4.0, 6.0], provenance=prov
        )
        result = conn.degree_imbalance()
        assert result.dims == ("node", "freq", "time")
        assert result.shape == (3, 3, 1)
        np.testing.assert_allclose(result.values[:, 0, 0], result.values[:, 2, 0])
        np.testing.assert_array_equal(result.coords["freq"].values, [2.0, 4.0, 6.0])


class TestDegreeImbalanceMask:
    def test_a_mask_keeps_only_the_chosen_edges(self, prov):
        mask = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=bool)  # the upper triangle
        result = square(prov, WEIGHTED).degree_imbalance(mask=mask)
        kept = np.where(mask, WEIGHTED, 0.0)
        # out: A 1.4, B 0.5, C 0; in: A 0, B 0.8, C 1.1 -> only B is defined
        np.testing.assert_allclose(result.values[1, 0, 0], np.log(kept[1].sum()) - np.log(0.8))
        assert np.isnan(result.values[0, 0, 0]) and np.isnan(result.values[2, 0, 0])

    def test_a_mask_of_everything_equals_no_mask(self, prov):
        conn = square(prov, WEIGHTED)
        everything = np.ones((3, 3), dtype=bool)
        np.testing.assert_array_equal(
            conn.degree_imbalance(mask=everything).values, conn.degree_imbalance().values
        )

    def test_a_mask_can_differ_between_frequencies(self, prov):
        base = np.array(WEIGHTED, dtype=float)
        values = np.stack([base, base], axis=-1)
        conn = Connectivity.from_array(
            values, source=["A", "B", "C"], freqs=[2.0, 4.0], provenance=prov
        )
        mask = np.ones((3, 3, 2, 1), dtype=bool)
        mask[0, 2, 1, 0] = False  # drop A -> C at 4 Hz only
        result = conn.degree_imbalance(mask=mask)
        assert result.values[0, 0, 0] != result.values[0, 1, 0]
        reference = np.array(WEIGHTED, dtype=float)
        reference[0, 2] = 0.0
        np.testing.assert_allclose(result.values[:, 1, 0], expected_imbalance(reference))

    def test_a_mask_must_be_boolean_and_fit_the_result(self, prov):
        conn = square(prov, WEIGHTED)
        with pytest.raises(ValueError, match="boolean"):
            conn.degree_imbalance(mask=np.ones((3, 3)))
        with pytest.raises(ValueError, match="shape"):
            conn.degree_imbalance(mask=np.ones((2, 2), dtype=bool))
        with pytest.raises(ValueError, match="shape"):
            conn.degree_imbalance(mask=np.ones((3, 3, 2, 1), dtype=bool))


class TestDegreeImbalanceProvenance:
    def read(self, result):
        return json.loads(result.attrs["provenance"])["history"][-1]

    def test_the_step_records_whether_a_mask_was_used_and_how_many_edges_counted(self, prov):
        conn = square(prov, WEIGHTED)
        step = self.read(conn.degree_imbalance())
        assert step == {"op": "degree_imbalance", "mask": "none", "edges_counted": 6}
        mask = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=bool)
        step = self.read(conn.degree_imbalance(mask=mask))
        assert step == {"op": "degree_imbalance", "mask": "given", "edges_counted": 3}

    def test_the_input_provenance_is_kept(self, prov):
        result = square(prov, WEIGHTED).degree_imbalance()
        assert json.loads(result.attrs["provenance"])["method"] == "gc"


class TestDegreeImbalanceRefusals:
    def test_negative_values_are_refused_because_a_degree_must_not_cancel(self, prov):
        net = square(prov, WEIGHTED).net_direction()
        with pytest.raises(ValueError, match="non-negative"):
            net.degree_imbalance()

    def test_results_that_are_not_square_or_not_real_are_refused(self, prov):
        rect = Connectivity.from_array(
            np.ones((2, 3)), source=["A", "B"], target=["X", "Y", "Z"], provenance=prov
        )
        with pytest.raises(ValueError, match="same nodes"):
            rect.degree_imbalance()
        complex_ = Connectivity.from_array(
            np.ones((2, 2), dtype=complex), source=["A", "B"], provenance=prov
        )
        with pytest.raises(ValueError, match="Complex"):
            complex_.degree_imbalance()

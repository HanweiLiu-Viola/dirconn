"""The existing plot kinds draw the results of the further estimators the right way round."""

import warnings

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dirconn import MTGC, PCMCI, PSI, TE, TRGC, DirconnWarning, ImCoh, TimeSeries  # noqa: E402

SFREQ = 100.0


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def data():
    """A drives B at lag 2; C is independent. Names differ from indices on purpose."""
    rng = np.random.default_rng(0)
    x = np.zeros((3, 4100))
    e = rng.standard_normal(x.shape)
    for t in range(2, x.shape[-1]):
        x[0, t] = 0.5 * x[0, t - 1] - 0.3 * x[0, t - 2] + e[0, t]
        x[1, t] = 0.4 * x[1, t - 1] + 0.4 * x[0, t - 2] + e[1, t]
        x[2, t] = 0.6 * x[2, t - 1] + e[2, t]
    return TimeSeries.from_array(x[:, 100:], SFREQ, names=["A", "B", "C"])


def _fit(estimator, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)
        return estimator.fit(data)


def _check_grid(ax, expected):
    """Row i is source i, column j target j, empty diagonal, diverging scale centred on 0."""
    image = ax.images[0]
    drawn = image.get_array()
    assert drawn.mask[[0, 1, 2], [0, 1, 2]].all()
    np.testing.assert_allclose(drawn.filled(np.nan), expected, equal_nan=True)
    low, high = image.get_clim()
    assert low == pytest.approx(-high)
    assert image.get_cmap().name.startswith("RdBu_r")
    assert [t.get_text() for t in ax.get_yticklabels()] == ["A", "B", "C"]
    assert ax.get_ylabel() == "source" and ax.get_xlabel() == "target"


def test_trgc_grid(data):
    result = _fit(TRGC(order=2, check_order_stability=False), data)
    ax = result.plot(kind="connectivity_grid")
    _check_grid(ax, result.values[:, :, 0, 0])
    assert result.values[0, 1, 0, 0] > 0  # the true edge A -> B is positive, row A column B
    assert "trgc" in ax.get_title()


def test_psi_grid_at_its_band(data):
    result = _fit(PSI(segment_length=1.0, bands=[(2.0, 20.0)]), data)
    ax = result.plot(kind="connectivity_grid", freq=11.0)
    _check_grid(ax, result.values[:, :, 0, 0])
    assert result.values[0, 1, 0, 0] > 0
    assert "11 Hz" in ax.get_title()


def test_imcoh_grid_over_a_band(data):
    result = _fit(ImCoh(segment_length=1.0), data)
    ax = result.plot(kind="connectivity_grid", band=(2.0, 20.0))
    _check_grid(ax, result.reduce_freq((2.0, 20.0)).values[:, :, 0, 0])


def test_imcoh_spectrum_draws_the_two_mirror_curves(data):
    result = _fit(ImCoh(segment_length=1.0), data)
    ax = result.plot(kind="directed_spectrum", source="A", target="B")
    forward, backward = ax.get_lines()
    np.testing.assert_allclose(forward.get_xdata(), result.freqs)
    np.testing.assert_allclose(forward.get_ydata(), result.values[0, 1, :, 0])
    np.testing.assert_allclose(backward.get_ydata(), -forward.get_ydata())
    assert ax.get_ylabel() == "imcoh"
    # undirected: the legend names the two orders of the pair, without arrows
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["(A, B)", "(B, A)"]
    assert "undirected" in ax.get_title()


def test_mtgc_grid_and_spectrum(data):
    result = _fit(MTGC(segment_length=1.0, fmax=30.0), data)
    ax = result.plot(kind="connectivity_grid", band=(2.0, 20.0))
    image = ax.images[0]
    drawn = image.get_array()
    assert drawn.mask[[0, 1, 2], [0, 1, 2]].all()
    expected = result.reduce_freq((2.0, 20.0)).values[:, :, 0, 0]
    np.testing.assert_allclose(drawn.filled(np.nan), expected, equal_nan=True)
    assert image.get_clim()[0] == 0.0  # non-negative: sequential scale from 0
    assert expected[0, 1] > expected[1, 0]  # A -> B is the true direction, row A column B
    plt.close("all")
    ax = result.plot(kind="directed_spectrum", source="A", target="B")
    forward, backward = ax.get_lines()
    np.testing.assert_allclose(forward.get_ydata(), result.values[0, 1, :, 0])
    np.testing.assert_allclose(backward.get_ydata(), result.values[1, 0, :, 0])
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["A → B", "B → A"]
    assert "undirected" not in ax.get_title()


def test_pcmci_grid(data):
    result = _fit(PCMCI(tau_max=3), data)
    ax = result.plot(kind="connectivity_grid")
    image = ax.images[0]
    drawn = image.get_array()
    assert drawn.mask[[0, 1, 2], [0, 1, 2]].all()
    np.testing.assert_allclose(drawn.filled(np.nan), result.values[:, :, 0, 0], equal_nan=True)
    assert result.values[0, 1, 0, 0] > 0.2  # A -> B, row A column B


def test_te_grid(data):
    result = _fit(TE(mode="pairwise", max_delay=3), data)
    ax = result.plot(kind="connectivity_grid")
    drawn = ax.images[0].get_array()
    assert drawn.mask[[0, 1, 2], [0, 1, 2]].all()
    np.testing.assert_allclose(drawn.filled(np.nan), result.values[:, :, 0, 0], equal_nan=True)
    assert result.values[0, 1, 0, 0] > result.values[1, 0, 0, 0]  # A -> B, row A column B


def test_method_info_finds_the_estimator_by_its_name():
    from dirconn.estimator import method_info

    assert method_info("imcoh") is ImCoh.info and not method_info("imcoh").directed
    assert method_info("mtgc") is MTGC.info and method_info("trgc").antisymmetric
    assert method_info("gc").directed
    assert method_info("no such method") is None

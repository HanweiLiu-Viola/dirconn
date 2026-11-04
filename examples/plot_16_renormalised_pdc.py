"""
=============================================================
Renormalised PDC and its threshold, next to a surrogate test
=============================================================

The PDC of tutorial 3 is a share: the direct influence of a source on one target, normalised over
all its targets. Its size depends on everything else the source drives, so a value of 0.1 can be
a strong influence in one network and nothing in another. The **renormalised PDC**
(:class:`~dirconn.RPDC`, Schelter et al. 2009) asks a different question: how far is the direct
influence at a frequency from zero, *relative to its estimation error*? Its value ``lambda`` comes
with an asymptotic threshold, ``chi2 / N``, so a test costs one model fit instead of the dozens
of fits of a surrogate test.

This tutorial computes rPDC on a simulated chain, compares its threshold with a surrogate test,
shows why the model order matters for rPDC in particular, and shows the residual whiteness check
that warns when the order is too low.
"""

# %%
# Setup
# -----
# The shipped scenario ``chain``: A drives B and B drives C, each with a delay of 10 ms; A, B and
# C oscillate at 10, 20 and 15 Hz. The regions are observed with white noise at 10 dB.
import os

import matplotlib.pyplot as plt
import numpy as np

import dirconn
from dirconn import RPDC, TimeSeries, significance_test
from dirconn.scenario import load_scenario

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")

spec = load_scenario("chain")
DURATION_S = 30.0 if QUICK else 120.0
sim = dirconn.simulate(
    spec.network,
    sfreq=spec.sfreq,
    n_times=int(DURATION_S * spec.sfreq),
    seed=SEED,
    dynamics=spec.dynamics,
    n_background=0,
    node_scale="unit",
)
observed = sim.roi_observations(snr_db=10.0, seed=SEED + 1)
data = TimeSeries.from_array(observed.data[0], observed.sfreq, names=["A", "B", "C"])
FREQS = np.arange(0.0, 51.0)  # 1 Hz steps up to the Nyquist frequency of 100 Hz sampling
ORDER = 20  # 0.2 s, long enough for these oscillators (see "The order" below)

# %%
# rPDC and its threshold
# ----------------------
# :func:`~dirconn.analytic_rpdc_test` fits the model once and returns, next to the p-values, the
# value of ``lambda`` that a cell must exceed at level ``alpha``: ``chi2_{df, 1-alpha} / N`` with
# ``df = 2`` (``df = 1`` at 0 Hz and at the Nyquist frequency, where the imaginary part vanishes).
estimator = RPDC(order=ORDER, freqs=FREQS)
test = significance_test(estimator, data, test="analytic")

fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True, layout="constrained")
for ax, (source, target, label) in zip(
    axes,
    [("A", "B", "A → B (real)"), ("B", "A", "B → A (none)"), ("A", "C", "A → C (indirect)")],
    strict=True,
):
    i, j = data.names.index(source), data.names.index(target)
    ax.semilogy(FREQS, test.observed.values[i, j, :, 0], label="rPDC (lambda)")
    ax.semilogy(FREQS, test.threshold[i, j, :, 0], "k--", label="threshold, alpha 0.05")
    ax.set(title=label, xlabel="frequency (Hz)")
axes[0].set_ylabel("lambda")
axes[0].legend(fontsize=8)

# %%
# The real influence A → B lies far above the threshold at every frequency; B → A and the
# indirect A → C stay near it and may cross it at a few frequencies (the observation noise has a
# share in that; see the end). Each frequency is tested on its own, so a pair without an
# influence crosses a pointwise threshold somewhere in a spectrum of 51 frequencies far more often
# than 5 % of the time. ``adjusted()`` corrects over every pair and frequency of the result
# (Benjamini-Hochberg by default); to test one band, give the estimator that band's frequencies.
pointwise = test.pvalues[..., 0] <= 0.05
corrected = test.significant()[..., 0]
for source, target in [("A", "B"), ("B", "C"), ("B", "A"), ("A", "C"), ("C", "A")]:
    i, j = data.names.index(source), data.names.index(target)
    print(
        f"{source} -> {target}: {pointwise[i, j].sum():2d} of {len(FREQS)} frequencies below 0.05, "
        f"{corrected[i, j].sum():2d} after BH over pairs and frequencies"
    )

# %%
# Next to a surrogate test
# ------------------------
# A surrogate test makes fewer assumptions about the distribution: it compares each value with
# the values of data whose channels were made independent (here by randomising the Fourier phases
# of every channel), refitting the model on every surrogate. With the same estimator the
# statistic is still rPDC, a measure of the *direct* influence. It costs one fit per surrogate,
# and its smallest p-value is ``1 / (n_surrogates + 1)``, which also limits a correction over
# many frequencies (dirconn warns when no cell could survive one, as here).
import warnings

n_surrogates = 19 if QUICK else 39
with warnings.catch_warnings():
    warnings.simplefilter("ignore", dirconn.DirconnWarning)  # the power check of the correction
    surrogate = significance_test(
        estimator,
        data,
        test="surrogate",
        surrogate="phase_randomize",
        n_surrogates=n_surrogates,
        seed=SEED,
    )
for source, target in [("A", "B"), ("B", "A"), ("A", "C")]:
    i, j = data.names.index(source), data.names.index(target)
    print(
        f"{source} -> {target}: frequencies below 0.05: asymptotic "
        f"{(test.pvalues[i, j] <= 0.05).sum():2d}, surrogate {(surrogate.pvalues[i, j] <= 0.05).sum():2d}"
        f" of {len(FREQS)}"
    )

# %%
# Both tests find A → B at nearly every frequency, and both flag B → A and A → C at some
# frequencies, about equally often: in this recording the observation noise has given the
# pairs without a direct edge small real direct terms (see the end), and both tests test the
# same model quantity. The asymptotic test needed one fit, the surrogate test one per surrogate.
#
# The order
# ---------
# rPDC weighs the real and imaginary part of the AR polynomial entry by the inverse of their
# covariance. Both are linear combinations of the ``order`` coefficients of the pair, and a Wald
# statistic does not change under an invertible linear map. So at order 1 ``lambda`` is the same
# at every frequency, and at order 2 at every frequency except the two ends: a low order cannot
# say *where* in frequency an influence acts.
fig, ax = plt.subplots(layout="constrained")
i, j = data.names.index("A"), data.names.index("B")
for order in (1, 2, 3, ORDER):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", dirconn.DirconnWarning)  # low orders: see the next section
        values = RPDC(order=order, freqs=FREQS).fit(data).values[i, j, :, 0]
    ax.semilogy(FREQS, values, label=f"order {order}")
ax.set(title="rPDC A → B at several orders", xlabel="frequency (Hz)", ylabel="lambda")
ax.legend()

# %%
# The residual whiteness check
# ----------------------------
# A too-low order leaves structure in the residuals, and correlated residuals make every
# asymptotic threshold too permissive. Every static VAR fit runs a Hosking portmanteau test of
# the residuals and warns when it fails and the largest residual autocorrelation is above 0.1.
# Observation noise turns each region into an ARMA process, so order 2 is too low here:
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    low = RPDC(order=2, freqs=FREQS).fit(data)
print([str(w.message)[:100] for w in caught if issubclass(w.category, dirconn.DirconnWarning)])
for order, result in [(2, low), (ORDER, test.observed)]:
    (check,) = [c for c in result.provenance.assumptions.checks if c.name == "residual_whiteness"]
    print(f"order {order:2d}: {check.status:5s} {check.detail}")

# %%
# What this does and does not show
# --------------------------------
# * One simulated recording: a demonstration, not an evaluation. How the two asymptotic tests
#   behave over many datasets was measured on this chain (protocol ``asymptotic_calibration``,
#   ``docs/statistics.md``): under a true null neither is liberal, at 30 s and 120 s, with 10 or
#   40 dB of observation noise, so with an adequate order and white residuals they can replace a
#   surrogate test there.
# * Observation noise is the catch. It gives the best VAR of coupled signals small *real* direct
#   terms where the network has none, and long recordings find them with every test: in the
#   chain at 10 dB, asymptotic rPDC called 8 % (30 s) and 21 % (120 s) of the absent direct edges
#   significant at single frequencies, the surrogate test 6 % and 14 %. The whiteness check does
#   not catch this; it catches a too-low order.
# * A threshold says whether a value at one frequency differs from zero under the model. It is not
#   a confidence interval and not a statement about a band; correct over the frequencies you look
#   at, and choose the band before looking.
# * rPDC is predictive and, like every MVAR measure, changed by volume conduction and leakage. It
#   has no ridge, so it suits a few channels (regions, selected sensors). Each result lists its
#   assumptions (``not_checked`` means not tested, not that it holds) and each test its
#   limitations:
for check in test.observed.provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")
print(*test.limitations, sep="\n")
plt.close("all")

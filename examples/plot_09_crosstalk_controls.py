"""
=====================================================
Controls for mixing: time reversal, imcoh and PSI
=====================================================

When the signals of two regions are mixed at zero delay, the way volume conduction mixes EEG
sensors and source leakage mixes estimated regions, Granger causality can report influences in
both directions although the regions are independent. This tutorial shows that effect on a small
simulation and three controls that are built to ignore zero-delay mixing:

* **time-reversed Granger causality** (:class:`~dirconn.TRGC`): the net Granger flow of the data
  minus the net flow of the same data played backwards;
* **imaginary coherence** (:class:`~dirconn.ImCoh`): the part of the coherency that a zero-delay
  mixture cannot produce;
* **phase slope index** (:class:`~dirconn.PSI`): whether the phase difference grows with frequency,
  as it does when one signal is a delayed version of the other.

It then shows the limit they share: they ignore a mixture of *independent* signals, but a mixture
of *coupled* signals still changes them, and in the example here one of them points the wrong way.
All data are small simulations of two signals; they show that these effects exist, not how large
they are in general.
"""

# %%
# Setup
# -----
# The signals are generated directly with numpy so that every ingredient is visible: two
# autoregressive processes, an optional lagged coupling and a fixed 2 x 2 mixing matrix. A mixing
# matrix multiplies the signals sample by sample, so it adds no delay.
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

import dirconn
from dirconn import GC, PSI, TRGC, ImCoh, TimeSeries

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")

SFREQ = 100.0
N_TIMES = 3000 if QUICK else 6000
MIXING = np.array([[1.0, 0.8], [0.6, 1.0]])  # each sensor sees much of the other signal
rng = np.random.default_rng(SEED)


def independent_sources():
    """Two independent AR(1) processes, one slow (0.9) and one fast (0.3); no coupling at all."""
    noise = rng.standard_normal((2, N_TIMES))
    return np.stack([lfilter([1], [1, -0.9], noise[0]), lfilter([1], [1, -0.3], noise[1])])


def delayed_copy(delay=3):
    """Signal B is signal A delayed by ``delay`` samples (30 ms), each with its own noise: A -> B."""
    source = rng.standard_normal(N_TIMES + delay)
    return np.stack([source[delay:], source[:N_TIMES]]) + 0.7 * rng.standard_normal((2, N_TIMES))


def as_series(x):
    return TimeSeries.from_array(x, SFREQ, names=["A", "B"])


def estimators():
    return {
        "GC": GC(order="bic", check_order_stability=False),
        "TRGC": TRGC(order="bic", check_order_stability=False),
        "PSI": PSI(segment_length=2.0, bands=[(2.0, 20.0)]),
    }


def summary(data):
    """A -> B and B -> A for GC, TRGC and PSI, and imcoh averaged over 2 to 20 Hz."""
    rows = {}
    for name, estimator in estimators().items():
        values = estimator.fit(data).values[:, :, 0, 0]
        rows[name] = (values[0, 1], values[1, 0])
    im = ImCoh(segment_length=2.0).fit(data).reduce_freq((2.0, 20.0)).values[:, :, 0, 0]
    rows["imcoh"] = (im[0, 1], im[1, 0])
    for name, (ab, ba) in rows.items():
        print(f"{name:6s}  A -> B {ab:+.3f}   B -> A {ba:+.3f}")
    return rows


# %%
# Mixed independent signals: Granger causality reports influence
# ---------------------------------------------------------------
# The two sources do not interact. After mixing, each recorded signal contains part of both, and
# the past of one recorded signal helps to predict the other, because it carries the slow
# source's past. Granger causality cannot tell that apart from an influence.
mixed = as_series(MIXING @ independent_sources())
fig, ax = plt.subplots(layout="constrained")
GC(order="bic", check_order_stability=False).fit(mixed).plot(ax=ax)
ax.set_title("GC on a zero-delay mixture of independent signals")

# %%
# The three controls on the same data. **TRGC** subtracts the net flow of the time-reversed data:
# a mixture looks the same backwards, so the two cancel. **imcoh** and **PSI** use only the
# imaginary part of the coherency, which a sample-by-sample mixture leaves at zero. All three are
# close to zero, while the Granger values are not:
mixed_rows = summary(mixed)

# %%
# Coupled signals with little mixing: the controls see the influence
# ------------------------------------------------------------------
# Now A drives B with a delay of 30 ms, and the mixing is weak. Each control is positive from A to
# B (TRGC and PSI are antisymmetric, so B -> A is the same number with the opposite sign; imcoh
# is positive here because B lags A by less than half a cycle in this band):
weak = np.array([[1.0, 0.3], [0.3, 1.0]])
coupled = as_series(weak @ delayed_copy())
coupled_rows = summary(coupled)

# %%
# Imaginary coherence is a function of frequency. Its sign is a phase, not a direction: when the
# delay reaches half a period (here at about 17 Hz) it crosses zero and changes sign, which is why
# :class:`~dirconn.ImCoh` is marked as undirected.
fig, ax = plt.subplots(layout="constrained")
ImCoh(segment_length=2.0).fit(coupled).plot(kind="directed_spectrum", source="A", target="B", ax=ax)
ax.axhline(0.0, color="0.5", linewidth=0.8)
ax.set_title("imcoh of the coupled pair: the two curves are mirror images")

# %%
# The asymmetry diagnostic
# ------------------------
# :func:`~dirconn.time_reversal_asymmetry` runs any estimator on the data and on the reversed data
# and compares them cell by cell, between -1 and 1. A cell that comes from a lagged influence
# changes when time runs backwards; a cell that comes from mixing does not. It is a heuristic
# with no p-value:
gc = GC(order="bic", check_order_stability=False)
for label, data in (("mixture", mixed), ("coupled", coupled)):
    score = dirconn.time_reversal_asymmetry(gc, data).asymmetry.values[:, :, 0, 0]
    print(f"{label:8s} asymmetry of GC: A -> B {score[0, 1]:+.2f}   B -> A {score[1, 0]:+.2f}")

# %%
# The limit: mixing coupled signals still changes the controls
# ------------------------------------------------------------
# The controls ignore a mixture of *independent* signals. They do not undo a mixture of *coupled*
# signals: each recorded signal then holds a share of both the driver and the driven signal. With
# the strong mixing from the start applied to the coupled pair, compare the signs:
strong = as_series(MIXING @ delayed_copy())
strong_rows = summary(strong)
flipped = [name for name, (ab, _) in strong_rows.items() if name != "GC" and ab < 0]
print("controls that point from B to A here:", flipped if flipped else "none")

# %%
# In this example the phase slope index points the wrong way, while TRGC and imcoh do not. That
# is a property of this mixing matrix and this signal, not a ranking of the methods: with other
# mixtures another control can be the one that fails, and when the controls disagree the data do
# not settle the direction. Reducing the mixing before the analysis (a better inverse solution,
# fewer and more distant regions) is the remedy; the controls only show whether mixing alone can
# explain a result.

# %%
# What this does and does not show
# --------------------------------
# * Zero-delay mixing of independent signals produced Granger influence in both directions, and
#   TRGC, imcoh and PSI stayed near zero on the same data. These are two signals in one simulation
#   (a demonstration, not an evaluation); how often each control helps on real recordings is not
#   shown here.
# * The controls did not remove the effect of mixing on *coupled* signals: in the last example a
#   control pointed the wrong way. A control near zero says that mixing alone can explain a
#   result; a control away from zero does not prove an influence.
# * TRGC is conditional Granger causality applied twice and inherits its assumptions (a linear,
#   stationary model with all relevant regions recorded); imcoh and PSI do not condition on other
#   regions at all.
# * Every estimate is predictive, not causal. Each result lists the assumptions it rests on;
#   ``not_checked`` means the library did not test an assumption, not that it holds:
result = TRGC(order="bic", check_order_stability=False).fit(coupled)
for check in result.provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")

# %%
# * Not shown here: significance of these controls (any of them works with
#   :func:`dirconn.significance_test`, see tutorial 4), more than two regions, and mixing through a
#   real forward model (tutorial 7).
plt.close("all")

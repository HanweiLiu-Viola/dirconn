"""
=====================================================
Granger causality without a model: multitaper spectra
=====================================================

The Granger causality of tutorials 1 to 3 comes from a fitted vector autoregressive (VAR) model:
choose an order, estimate the coefficients, read the influence off the model. The **multitaper
Granger causality** (:class:`~dirconn.MTGC`) skips the model. It estimates the spectral matrix of
two signals with multitapers and factorises it (Wilson's algorithm) into the transfer function
and noise covariance that a model would have had, then applies Geweke's spectral decomposition
(Dhamala et al. 2008). There is no model order to choose; instead the segment length sets the
frequency resolution.

This tutorial compares it with the known truth of a simulated system, shows what the segment
length changes, and shows the limit it shares with pairwise Granger causality, and the
model-based conditional spectrum (:class:`~dirconn.SSGC`) that does not have it.
"""

# %%
# Setup
# -----
# A two-signal VAR(2) whose spectral Granger causality can be computed exactly from its
# coefficients: signal A drives signal B at a lag of two samples, B does not drive A, and the two
# noise terms are correlated.
import os

import matplotlib.pyplot as plt
import numpy as np

import dirconn
from dirconn import GC, MTGC, SSGC, TimeSeries
from dirconn.multitaper_granger import geweke_spectral_gc
from dirconn.spectral import transfer_function

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")

SFREQ = 100.0
N_TIMES = 6000 if QUICK else 20000
COEFS = np.zeros((2, 2, 2))  # [lag, target, source]
COEFS[0] = [[0.5, 0.0], [0.0, 0.4]]
COEFS[1] = [[-0.3, 0.0], [0.4, 0.0]]  # A (source 0) drives B (target 1) at lag 2
NOISE = np.array([[1.0, 0.3], [0.3, 0.8]])
rng = np.random.default_rng(SEED)


def simulate_var(coefs, noise, n_times):
    """Simulate a VAR model with Gaussian noise; the first 100 samples are dropped."""
    n = coefs.shape[1]
    x = np.zeros((n, n_times + 100))
    e = rng.multivariate_normal(np.zeros(n), noise, n_times + 100).T
    for t in range(coefs.shape[0], x.shape[1]):
        x[:, t] = sum(coefs[k] @ x[:, t - 1 - k] for k in range(coefs.shape[0])) + e[:, t]
    return x[:, 100:]


data = TimeSeries.from_array(simulate_var(COEFS, NOISE, N_TIMES), SFREQ, names=["A", "B"])

# %%
# The estimate and the truth
# --------------------------
# ``segment_length`` is required: it sets the frequency resolution (``1 / segment_length`` Hz).
# The truth is Geweke's spectral Granger causality of the true model, evaluated on the same
# frequencies.
result = MTGC(segment_length=2.0).fit(data)
truth = geweke_spectral_gc(transfer_function(COEFS, result.freqs, SFREQ), NOISE)

fig, ax = plt.subplots(layout="constrained")
ax.plot(result.freqs, truth[0, 1], color="k", linewidth=2.5, alpha=0.4, label="truth A → B")
result.plot(kind="directed_spectrum", source="A", target="B", ax=ax)
ax.set_title("multitaper Granger causality against the true model")

# %%
# The estimate follows the true curve and stays near zero from B to A. Averaged over all
# frequencies (the whole unit circle), the spectral Granger causality is the time-domain one; the
# provenance keeps that mean, here next to the truth and to the model-based estimate:
integral = np.array(result.provenance.details["integral"], dtype=float)
model_based = GC(order="bic", conditional=False, check_order_stability=False).fit(data)
print(f"truth (mean of the true spectrum)  A -> B {truth[0, 1].mean():.3f}")
print(
    f"MTGC, mean over the circle         A -> B {integral[0, 1]:.3f}   B -> A {integral[1, 0]:.3f}"
)
print(
    f"GC from a fitted VAR               A -> B {model_based.values[0, 1, 0, 0]:.3f}"
    f"   B -> A {model_based.values[1, 0, 0, 0]:.3f}"
)

# %%
# The segment length
# ------------------
# A short segment gives many segments to average (less noise) but coarse frequency bins, and the
# tapers smooth the spectrum over ``2 NW / segment_length`` Hz, which flattens peaks. A long
# segment resolves finer detail but averages fewer segments. With the default ``NW = 4`` a 0.5 s
# segment smooths over 16 Hz and flattens the peak far below the truth; an 8 s segment keeps the
# shape but is noisy:
fig, ax = plt.subplots(layout="constrained")
ax.plot(result.freqs, truth[0, 1], color="k", linewidth=2.5, alpha=0.4, label="truth")
for seconds, style in ((0.5, ":"), (2.0, "-"), (8.0, "--")):
    estimate = MTGC(segment_length=seconds).fit(data)
    ax.plot(estimate.freqs, estimate.values[0, 1, :, 0], linestyle=style, label=f"{seconds:g} s")
    peak = truth[0, 1].max()
    print(
        f"segment {seconds:4.1f} s: {estimate.provenance.details['cross_spectra']['n_segments']:3d}"
        f" segments, bins of {1 / seconds:.3g} Hz, estimated peak {estimate.values[0, 1].max():.2f}"
        f" (true peak {peak:.2f})"
    )
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("A → B")
ax.legend(title="segment")
ax.set_title("the same data with three segment lengths")

# %%
# The limit: one pair at a time
# -----------------------------
# :class:`~dirconn.MTGC` factorises the spectrum of each pair on its own. In a chain A -> B -> C
# it therefore reports an influence from A to C, which runs through B, exactly like pairwise
# Granger causality; the conditional model-based estimate does not (tutorial 5):
chain = np.zeros((2, 3, 3))
chain[0] = np.diag([0.5, 0.4, 0.4])
chain[1, 1, 0] = 0.5  # A -> B at lag 2
chain[1, 2, 1] = 0.5  # B -> C at lag 2
three = TimeSeries.from_array(simulate_var(chain, np.eye(3), N_TIMES), SFREQ, names=["A", "B", "C"])
pairwise = np.array(MTGC(segment_length=2.0).fit(three).provenance.details["integral"], float)
conditional = GC(order="bic", check_order_stability=False).fit(three).values[:, :, 0, 0]
print(f"A -> C (indirect only): MTGC {pairwise[0, 2]:.3f}, conditional GC {conditional[0, 2]:.3f}")
print(f"A -> B (direct):        MTGC {pairwise[0, 1]:.3f}, conditional GC {conditional[0, 1]:.3f}")

# %%
# A conditional spectrum from the model
# -------------------------------------
# When a VAR model is acceptable, :class:`~dirconn.SSGC` gives the *conditional* spectral Granger
# causality from one fit. The model of all signals is written in state-space form, and each
# reduced model (here B and C without A) follows from it exactly (Barnett and Seth 2015), so no
# second model is refitted and no spectrum is factorised. Its mean over frequency is the
# time-domain value of ``GC(method="state_space")``. In the chain it puts nothing on A -> C:
spectral = SSGC(order="bic", check_order_stability=False).fit(three)
fig, ax = plt.subplots(layout="constrained")
for (source, target), label in (((0, 1), "A → B"), ((1, 2), "B → C"), ((0, 2), "A → C | B")):
    ax.plot(spectral.freqs, spectral.values[source, target, :, 0], label=label)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("spectral GC (nats)")
ax.legend()
ax.set_title("conditional spectral Granger causality from the state-space form")
print(f"A -> C | B, largest value over frequency: {np.nanmax(spectral.values[0, 2]):.4f}")

# %%
# What this does and does not show
# --------------------------------
# * On a simulated linear system the multitaper estimate followed the true spectral Granger
#   causality without a model order, and its mean over frequency was close to the time-domain
#   value. This is one system and one or two seeds (a demonstration, not an evaluation); the
#   validation report (``docs/method_validation.md``) runs it on the reference scenarios.
# * The segment length is a choice with a cost either way; nothing here picks it for you.
# * The method is pairwise: an influence through a third recorded signal is reported as if it
#   were direct, and so is a shared driver. A conditional multitaper version is not implemented;
#   :class:`~dirconn.SSGC` is conditional but needs a well-specified, stationary VAR model.
# * It is still Granger causality: predictive, not causal, and it assumes stationarity and that no
#   unrecorded signal drives the pair. Each result lists its assumptions; ``not_checked`` means the
#   library did not test one, not that it holds:
for check in result.provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")

# %%
# * Not shown here: significance (any estimator works with :func:`dirconn.significance_test`, see
#   tutorial 4), mixing between the signals (tutorial 9) and non-linear coupling.
plt.close("all")

"""
=================================================
Lagged links with PCMCI: conditions and lags
=================================================

PCMCI (Runge et al. 2019) answers a question close to conditional Granger causality: does the past
of one signal carry information about another, given the relevant past of the rest? It gets
there differently. It first selects, for every signal, a small set of lagged *parents* with a
sequence of conditional-independence tests, and then tests every lagged link
``i (t - tau) -> j (t)`` conditioned on the parents of both ends (the *MCI* test). With the
partial-correlation test (ParCorr) used here, the dependence is linear, and every link comes
with the lag at which it is strongest.

PCMCI needs the optional package tigramite: ``pip install dirconn[pcmci]``.
"""

# %%
# Setup
# -----
# Four signals generated with numpy: A drives B at a lag of 2 samples and C at a lag of 4, and C
# drives D at a lag of 1 with a negative sign. B and C share a driver but do not interact.
import os

import matplotlib.pyplot as plt
import numpy as np

import dirconn
from dirconn import GC, PCMCI, TimeSeries

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")

SFREQ = 100.0
N_TIMES = 2000 if QUICK else 6000
NAMES = ["A", "B", "C", "D"]
rng = np.random.default_rng(SEED)

x = np.zeros((4, N_TIMES + 100))
e = rng.standard_normal(x.shape)
for t in range(4, x.shape[1]):
    x[0, t] = 0.6 * x[0, t - 1] + e[0, t]
    x[1, t] = 0.3 * x[1, t - 1] + 0.5 * x[0, t - 2] + e[1, t]  # A -> B, lag 2
    x[2, t] = 0.3 * x[2, t - 1] + 0.5 * x[0, t - 4] + e[2, t]  # A -> C, lag 4
    x[3, t] = 0.3 * x[3, t - 1] - 0.5 * x[2, t - 1] + e[3, t]  # C -> D, lag 1, negative
data = TimeSeries.from_array(x[:, 100:], SFREQ, names=NAMES)

# %%
# PCMCI, its values and its lags
# ------------------------------
# ``tau_max`` is the largest lag tested. By default it is the order that the Bayesian information
# criterion picks for a VAR model of all signals (as for :class:`~dirconn.GC`); here we set it, so
# that the true lag of 4 is inside the range. The values are signed partial correlations, so the
# negative link from C to D shows in blue:
result = PCMCI(tau_max=6).fit(data)
fig, ax = plt.subplots(layout="constrained")
result.plot(ax=ax)
ax.set_title("PCMCI: MCI partial correlation at the best lag")
lags = np.array(result.provenance.details["lag"], dtype=float)
for source, target in ((0, 1), (0, 2), (2, 3), (1, 2)):
    print(
        f"{NAMES[source]} -> {NAMES[target]}: MCI {result.values[source, target, 0, 0]:+.3f}"
        f" at lag {lags[source, target]:.0f}"
    )

# %%
# The lag of every link is part of the result. The full lag profile is in the provenance; for
# the link from A to C it peaks at the true lag:
profile = np.array(result.provenance.details["val_matrix"], dtype=float)[0, 2]
fig, ax = plt.subplots(layout="constrained")
ax.bar(np.arange(1, profile.size + 1), profile)
ax.axhline(0.0, color="0.5", linewidth=0.8)
ax.set_xlabel("lag (samples)")
ax.set_ylabel("MCI partial correlation")
ax.set_title("A -> C at every lag")

# %%
# The shared driver
# -----------------
# B and C both follow A, so B's past predicts C a little when A is ignored. Pairwise Granger
# causality reports that; PCMCI conditions on the parents it selected (A among them) and, like
# conditional Granger causality, does not:
pairwise = GC(order=6, conditional=False, check_order_stability=False).fit(data)
conditional = GC(order=6, check_order_stability=False).fit(data)
print(f"B -> C: pairwise GC {pairwise.values[1, 2, 0, 0]:.4f}")
print(f"B -> C: conditional GC {conditional.values[1, 2, 0, 0]:.4f}")
print(f"B -> C: PCMCI {result.values[1, 2, 0, 0]:+.4f}")

# %%
# Significance
# ------------
# PCMCI's own test gives a p-value for every link and lag. :func:`dirconn.pcmci_test` reports,
# per pair, the smallest p-value over the lags multiplied by the number of lags (a Bonferroni
# correction over the lags: picking the best of several lags would otherwise make the p-value too
# small). The result works with the usual correction across pairs:
test = dirconn.pcmci_test(PCMCI(tau_max=6), data)
print(test.null)
significant = test.significant(method="fdr_bh")[:, :, 0, 0]
for source in range(4):
    for target in range(4):
        if significant[source, target]:
            print(f"significant after FDR: {NAMES[source]} -> {NAMES[target]}")

# %%
# The limit: what happens at zero lag
# -----------------------------------
# Only lagged links are tested. A zero-lag mixture, such as volume conduction between two sensors,
# is not a link PCMCI can represent, and it changes the lagged links. Mixing B into C at zero
# lag makes PCMCI report a lagged link from A to the mixed signal at B's lag as well:
mixed = np.array(data.data)[0].copy()
mixed[2] = mixed[2] + 0.8 * mixed[1]  # C now contains B, at zero lag
mixed_result = PCMCI(tau_max=6).fit(TimeSeries.from_array(mixed, SFREQ, names=NAMES))
profile_mixed = np.array(mixed_result.provenance.details["val_matrix"], dtype=float)[0, 2]
print("A -> C lag profile without mixing:", np.round(profile, 2))
print("A -> C lag profile with mixing:   ", np.round(profile_mixed, 2))

# %%
# What this does and does not show
# --------------------------------
# * On a simulated linear system PCMCI found the three links with their lags and signs and did not
#   report the shared-driver pair. This is one system and one seed (a demonstration, not an
#   evaluation); the validation report (``docs/method_validation.md``) runs it on the reference
#   scenarios.
# * The values are partial correlations given the selected parents: they depend on what was
#   selected (``pc_alpha``) and on ``tau_max``, and the p-values do not account for either choice.
# * Zero-lag dependence is not represented and changes the lagged links; PCMCI+ (contemporaneous
#   links) is not implemented here. ParCorr sees linear dependence only.
# * It is predictive dependence given the recorded signals, not causation: an unrecorded common
#   driver can produce a link. Each result lists its assumptions; ``not_checked`` means the library
#   did not test one, not that it holds:
for check in result.provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")

# %%
# * Not shown here: resampling tests (any estimator works with :func:`dirconn.significance_test`,
#   tutorial 4, but PCMCI is slow inside them), non-linear tests and more signals.
plt.close("all")

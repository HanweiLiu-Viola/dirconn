"""
================================================
Non-linear coupling: transfer entropy next to GC
================================================

Granger causality asks whether the past of one signal improves a *linear* prediction of another.
A coupling that is not linear can pass unnoticed. **Transfer entropy**
(:class:`~dirconn.TE`) asks the same question in terms of information: how much does the past of
the source tell about the present of the target beyond what the target's own past tells? It is
estimated without a model, with nearest neighbours (the KSG estimator), so it also sees
non-linear dependence. For Gaussian, linear data it is half the Granger causality.

This tutorial shows a coupling that Granger causality misses and transfer entropy finds, the
linear case where the two agree, and what the raw transfer entropy needs before it can be read:
a significance test.
"""

# %%
# Setup
# -----
# :func:`dirconn.coupled_nonlinear` makes three signals: X drives Y through a function ``g`` at a
# lag of two samples, Z is independent. With ``coupling="quadratic"``, ``g`` is a centred square of
# X. A square has zero linear correlation with its input, so the influence of X on Y is invisible
# to any linear measure.
import os

import matplotlib.pyplot as plt
import numpy as np

import dirconn
from dirconn import GC, TE

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")

SFREQ = 100.0
N_TIMES = 2000 if QUICK else 5000
quadratic, truth = dirconn.coupled_nonlinear(N_TIMES, sfreq=SFREQ, seed=SEED, coupling="quadratic")
linear, _ = dirconn.coupled_nonlinear(N_TIMES, sfreq=SFREQ, seed=SEED, coupling="linear")

# %%
# Granger causality and transfer entropy side by side
# ---------------------------------------------------
# Both are in nats and drawn on the same colour scale (each panel on its own scale would make the
# largest Granger value look strong, however small it is). On the quadratic coupling Granger
# causality from X to Y is about zero; transfer entropy is not:
gc_quadratic = GC(order="bic", check_order_stability=False).fit(quadratic)
te_quadratic = TE(mode="pairwise").fit(quadratic)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), layout="constrained")
truth.plot(ax=axes[0], colorbar=False)
axes[0].set_title("truth")
scale = float(np.nanmax(te_quadratic.values))
gc_quadratic.plot(ax=axes[1], vmax=scale)
axes[1].set_title("Granger causality")
te_quadratic.plot(ax=axes[2], vmax=scale)
axes[2].set_title("transfer entropy")
print(f"quadratic coupling, X -> Y: GC {gc_quadratic.values[0, 1, 0, 0]:.4f}")
print(f"quadratic coupling, X -> Y: TE {te_quadratic.values[0, 1, 0, 0]:.4f}")

# %%
# With a linear coupling of the same strength the two agree, transfer entropy being about half of
# Granger causality, as it is for Gaussian data:
gc_linear = GC(order="bic", check_order_stability=False).fit(linear).values[0, 1, 0, 0]
te_linear = TE(mode="pairwise").fit(linear).values[0, 1, 0, 0]
print(
    f"linear coupling, X -> Y: GC {gc_linear:.4f}, GC / 2 {gc_linear / 2:.4f}, TE {te_linear:.4f}"
)

# %%
# The delay
# ---------
# The estimator scans the interaction delay (``max_delay``, here 1 to 5 samples) and keeps the
# largest value; every delay is evaluated on the same rows. The scan is in the provenance:
scan = np.array(te_quadratic.provenance.details["scan"], dtype=float)[0, 1]
fig, ax = plt.subplots(layout="constrained")
ax.bar(te_quadratic.provenance.details["delays"], scan)
ax.set_xlabel("delay (samples)")
ax.set_ylabel("TE X → Y (nats)")
ax.set_title("transfer entropy at every delay")
print("delay found:", te_quadratic.provenance.details["delay"][0][1], "samples")

# %%
# The raw value is biased
# -----------------------
# Look at the pairs without coupling: their transfer entropy is small but not zero. The
# nearest-neighbour estimate has a bias that depends on the number of samples, the neighbours and
# the dimensions, and keeping the best of five delays adds to it. A raw value therefore needs a
# test. Here a phase-randomised surrogate test: it keeps every signal's spectrum and destroys the
# relations between them (tutorial 4). Phase randomisation also destroys the *non-linear*
# structure within each signal, which is part of what this null hypothesis tests:
off_diagonal = te_quadratic.values[:, :, 0, 0][~np.eye(3, dtype=bool)]
print(f"TE of the five uncoupled pairs: {np.round(np.sort(off_diagonal)[:-1], 4)}")
test = dirconn.significance_test(
    TE(mode="pairwise", max_delay=3),
    quadratic,
    test="surrogate",
    surrogate="phase_randomize",
    n_surrogates=19 if QUICK else 39,
    seed=SEED + 1,
)
print("p-values (rows: source, columns: target):")
print(np.round(test.pvalues[:, :, 0, 0], 3))
print(
    f"smallest possible p-value with {test.n_surrogates} surrogates: {1 / (test.n_surrogates + 1):.3f}"
)

# %%
# X -> Y reaches the smallest p-value that this number of surrogates allows, the uncoupled pairs
# do not. The warning printed by the test says more: with so few surrogates no correction over
# the six pairs can reach 0.05. A real analysis needs at least 119 surrogates here, which takes
# minutes with transfer entropy, so this tutorial stops at an uncorrected reading.

# %%
# Three modes
# -----------
# This tutorial uses the pairwise mode (``mode="pairwise"``): one pair at a time, the best of the
# scanned delays. With ``mode="conditional"`` the past of every other signal joins the condition,
# which removes indirect and shared influence but adds a dimension per signal, and the
# nearest-neighbour estimate gets worse quickly as the dimension grows. With three signals the
# difference is small:
te_conditional = TE(mode="conditional").fit(quadratic)
print(
    f"X -> Y pairwise {te_quadratic.values[0, 1, 0, 0]:.4f}, conditional {te_conditional.values[0, 1, 0, 0]:.4f}"
)

# %%
# The default of :class:`~dirconn.TE` is the third mode, ``mode="multivariate"``: it selects the
# past variables of every signal that inform the target, tests each step by permutation and
# returns zero for a source that is not selected, so it needs no separate surrogate test. It is
# much slower; tutorial 15 shows it on a chain and on a common driver, where the pairwise mode
# reports links that do not exist.

# %%
# What this does and does not show
# --------------------------------
# * A quadratic coupling that Granger causality did not see was found by transfer entropy, and on
#   a linear coupling transfer entropy was about half of Granger causality. This is one simulated
#   system and one seed (a demonstration, not an evaluation); the validation report
#   (``docs/method_validation.md``) runs it on the reference scenarios.
# * The raw estimate is biased and its size depends on the settings, so values are compared with
#   a surrogate test, not across data sets or embeddings.
# * The result depends on the embedding (``target_history``, ``source_history``) and on the
#   scanned delays. The target history defaults to the BIC order of a VAR model; one that is too
#   short leaves the target's own memory in the source's past and can reverse the direction, as the
#   validation report shows for a history of one sample on oscillating signals.
# * It is predictive information transfer, not causation, and it assumes stationarity and that no
#   unrecorded signal drives the pair. Each result lists its assumptions; ``not_checked`` means the
#   library did not test one, not that it holds:
for check in te_quadratic.provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")

# %%
# * Not shown here: effective transfer entropy (the bias subtracted, deferred), longer embeddings,
#   more signals, and mixing between the signals (tutorial 9).
plt.close("all")

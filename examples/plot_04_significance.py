"""
==============================
Is this value more than noise?
==============================

An estimator always returns numbers, also for signals that have nothing to do with each other. So
the first question about any value is whether it is more than noise would give. This tutorial
shows the tests dirconn offers for that (surrogate data, block permutation and an analytic test for
Granger causality), how to correct for testing many cells at once, and how to put an interval around
a value with the bootstrap. It ends on what a significant value does *not* tell you.
"""

# %%
# Setup and data
# --------------
# The same simulated chain as in tutorial 1: the first region drives the second, the second drives
# the third, and there is no direct influence from the first on the third.
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dirconn
from dirconn import GC, Connectivity, Edge, Network, dk_node
from dirconn.simulation import NodeDynamics, simulate

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")

A, B, C = "cuneus-rh", "precuneus-rh", "insula-rh"
network = Network(
    nodes=[dk_node(A), dk_node(B), dk_node(C)],
    edges=[Edge(A, B, strength=0.4, delay_ms=10.0), Edge(B, C, strength=0.4, delay_ms=10.0)],
    name="chain",
)
dynamics = {
    A: NodeDynamics(kind="oscillator", freq_hz=10.0),
    B: NodeDynamics(kind="oscillator", freq_hz=20.0),
    C: NodeDynamics(kind="oscillator", freq_hz=15.0),
}
SFREQ = 100.0
sim = simulate(
    network,
    sfreq=SFREQ,
    n_times=int((30.0 if QUICK else 120.0) * SFREQ),
    seed=SEED,
    dynamics=dynamics,
    snr_db=10.0,
    n_background=0,
    node_scale="unit",
)
data = sim.roi_observations(snr_db=10.0, seed=SEED + 1)
print(data)

# %%
# A test with surrogate data
# --------------------------
# The idea: make many copies of the data in which any relation *between* signals is destroyed but
# each signal keeps its own spectrum, estimate on every copy, and see how often the copies give a
# value as large as the one observed. That fraction is the p-value. ``phase_randomize`` makes the
# copies by shuffling the phases of every signal independently.
#
# The estimator is built once and handed to the test. The order is fixed here so that every copy
# is fitted with the same model; the number of copies sets how small a p-value can be: with 199
# copies the smallest possible p-value is 1/200 = 0.005.
estimator = GC(order=5, check_order_stability=False)
surrogate = dirconn.significance_test(
    estimator,
    data,
    test="surrogate",
    surrogate="phase_randomize",
    n_surrogates=199,
    seed=SEED + 2,
)
print(surrogate)
print("what this test assumes and cannot see:")
for line in surrogate.limitations:
    print(" -", line)

# %%
# The results are arrays of the same layout as the estimate: ``pvalues`` and ``observed`` have one
# entry per ``(source, target)`` cell. As a table:
observed = surrogate.observed.values[:, :, 0, 0]
pvalues = surrogate.pvalues[:, :, 0, 0]
names = surrogate.observed.source
rows = []
for i, source in enumerate(names):
    for j, target in enumerate(names):
        if i != j:
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "value (nats)": observed[i, j],
                    "p": pvalues[i, j],
                    "real edge": bool(sim.truth.values[i, j, 0, 0] == 1),
                }
            )
table = pd.DataFrame(rows)
print(table.round(3).to_string(index=False))

# %%
# Many cells, many tests
# ----------------------
# A matrix of three regions has six directed cells, and each cell is a test. At the usual level of
# 0.05, one test in twenty on pure noise comes out "significant", so with many cells some of them
# will. ``adjust_pvalues`` corrects for that: ``"bonferroni"`` is the strictest and controls the
# chance of *any* false positive; ``"fdr_bh"`` (Benjamini-Hochberg) controls the *share* of false
# positives among the cells declared significant and is the default of ``significant``.
adjusted = {
    method: dirconn.adjust_pvalues(surrogate.pvalues, method=method)[:, :, 0, 0]
    for method in ("none", "fdr_bh", "bonferroni")
}
counts = {m: int(np.sum(p[np.isfinite(p)] < 0.05)) for m, p in adjusted.items()}
print("cells below 0.05:", counts)
significant = surrogate.significant()  # fdr_bh at alpha = 0.05

# %%
# The result in three pictures: the value, which cells are significant, and the truth.
significant_map = Connectivity.from_array(
    np.where(np.isfinite(observed), significant[:, :, 0, 0].astype(float), np.nan),
    source=surrogate.observed.source,
    provenance=surrogate.observed.provenance,
)
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), layout="constrained")
surrogate.observed.plot(ax=axes[0])
significant_map.plot(ax=axes[1], vmin=0.0, vmax=1.0, colorbar=False)
sim.truth.plot(ax=axes[2], colorbar=False)
axes[1].set_title("significant (FDR, 0.05)")
axes[2].set_title("truth")

# %%
# A significant value is not necessarily a large one
# --------------------------------------------------
# A p-value says that a value is unlikely to come from the null model of the test, here signals
# that have no relation to each other. It does not say how large the value is. Sorted by value, the
# two real influences stand far above every other cell, whatever the p-values of the rest say:
table["significant (FDR)"] = [
    bool(significant[names.index(row.source), names.index(row.target), 0, 0])
    for row in table.itertuples()
]
print(table.sort_values("value (nats)", ascending=False).round(3).to_string(index=False))

# %%
# If cells without a real influence are marked significant in your run, look at their values: they
# are tiny next to those of the real influences. A longer recording gives a test more power, so it
# can flag smaller and smaller values; the size of the value is the second thing to look at, next
# to the p-value. Whether such small values come from something in the simulation, for example the
# observation noise that is added to every region, was not tested here.

# %%
# An interval around a value
# --------------------------
# The bootstrap asks a different question: how much would the value move if the recording had been
# slightly different? It resamples blocks of the recording (continuous data) or whole trials
# (epoched data), estimates on each resample and reports the spread. Blocks must be long compared
# with the memory of the signals; 1 s is used here, and the library warns when there are fewer than
# 20 blocks to draw from.
boot = dirconn.bootstrap(estimator, data, n_boot=200, seed=SEED + 3, block_size=100)
rows = []
for i, source in enumerate(names):
    for j, target in enumerate(names):
        if i != j:
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "value": boot.observed.values[i, j, 0, 0],
                    "lower": boot.lower[i, j, 0, 0],
                    "upper": boot.upper[i, j, 0, 0],
                }
            )
print(f"{boot.ci:.0%} intervals from {boot.n_boot} resamples")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# %%
# Granger causality cannot be negative, so the interval of a cell without influence sits just above
# zero. The interval describes how much the estimate varies; it does not say whether the cell is
# a real influence.

# %%
# Other tests
# -----------
# The same call runs a block permutation test (``test="block_perm"``) and, for Granger causality
# only, an analytic F test (``test="analytic"``, which needs no copies). They make different
# assumptions, and each result lists them:
block = dirconn.significance_test(
    estimator, data, test="block_perm", block_size=100, n_surrogates=199, seed=SEED + 4
)
analytic = dirconn.significance_test(GC(order=5, ridge=0.0), data, test="analytic")
for label, result in (
    ("surrogate", surrogate),
    ("block permutation", block),
    ("analytic", analytic),
):
    n_significant = int(np.nansum(result.significant()[:, :, 0, 0]))
    print(f"{label}: {n_significant} of {len(rows)} cells significant (FDR, 0.05)")
    for line in result.limitations:
        print("   -", line)

# %%
# What this does and does not show
# --------------------------------
# * A significant cell means that its value is unlikely under the null model of the test; it does
#   not mean that the influence is real, large or causal. The tests here ask whether signals are
#   related *at all*; a relation through a third region or through a shared input can pass them
#   (tutorial 5).
# * The three tests make different assumptions, listed above, and a violated assumption can make
#   the p-values wrong in either direction. This tutorial does not check how often each test is
#   wrong; the benchmark report (``docs/benchmark_results.md``) measures that under stated
#   conditions.
# * Correcting for the six cells of a three-region matrix is simple; with many regions, many
#   frequencies and many time points the family of tests grows quickly and the choice of family is
#   yours to make and to report.
# * Every result lists the assumptions it rests on; ``not_checked`` means that the library did not
#   test an assumption, not that it holds:
for check in surrogate.observed.provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")

# %%
# * Not tested here: the test of one recording says nothing about a group of participants; and
#   frequency-resolved and time-resolved results need a test over every frequency or time point as
#   well, which this tutorial does not do.
plt.close("all")

"""
================================
Your first directed connectivity
================================

This tutorial simulates three brain regions that drive each other in a chain, estimates who
drives whom with Granger causality, and compares the answer with the truth that we put into the
simulation. Along the way you meet the objects every other tutorial uses: a ``Network`` with a
known truth, a ``TimeSeries``, a ``Connectivity`` result and the provenance it carries.

Only simulated data are used, so the tutorial runs in about a minute and needs no download.
"""

# %%
# Setup
# -----
# ``QUICK`` shortens the simulated recording so the tests can run this file in seconds; you get the
# full version. The seed makes the simulation repeatable, and the versions are printed so a figure
# can be traced back to the software that made it.
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dirconn
from dirconn import Connectivity, Edge, Network, dk_node
from dirconn.simulation import NodeDynamics, simulate

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")

# %%
# A network whose truth we know
# -----------------------------
# Three cortical regions, named with Desikan-Killiany labels. Region ``cuneus-rh`` drives
# ``precuneus-rh`` and ``precuneus-rh`` drives ``insula-rh``; each influence has a coupling
# strength of 0.4 and a delay of one sample (10 ms at 100 Hz). There is no other connection, and in
# particular no direct one from the first region to the last.
A, B, C = "cuneus-rh", "precuneus-rh", "insula-rh"
network = Network(
    nodes=[dk_node(A), dk_node(B), dk_node(C)],
    edges=[Edge(A, B, strength=0.4, delay_ms=10.0), Edge(B, C, strength=0.4, delay_ms=10.0)],
    name="chain",
)
# Each region has its own rhythm, so the three signals are not copies of each other.
dynamics = {
    A: NodeDynamics(kind="oscillator", freq_hz=10.0),
    B: NodeDynamics(kind="oscillator", freq_hz=20.0),
    C: NodeDynamics(kind="oscillator", freq_hz=15.0),
}

# %%
# Simulate the recording
# ----------------------
# ``simulate`` produces sensor data from the network; here we want the region time series
# themselves (a head model is the subject of tutorial 7), so we ask for the region-level
# observations: every region's activity plus independent noise at an SNR of 10 dB.
SFREQ = 100.0
DURATION_S = 30.0 if QUICK else 120.0
SNR_DB = 10.0
sim = simulate(
    network,
    sfreq=SFREQ,
    n_times=int(DURATION_S * SFREQ),
    seed=SEED,
    dynamics=dynamics,
    snr_db=SNR_DB,
    n_background=0,
    node_scale="unit",
)
data = sim.roi_observations(snr_db=SNR_DB, seed=SEED + 1)
print(data)

# %%
# What the truth looks like
# -------------------------
# ``sim.truth`` is a ``Connectivity`` too: 1 where a region drives another, 0 elsewhere, and an empty
# diagonal (a region does not influence itself in this sense). In every matrix of dirconn
# **row i is the source and column j is the target**: the cell in row ``cuneus-rh`` and column
# ``precuneus-rh`` answers "how much does the first region influence the second?".
ax = sim.truth.plot()
ax.set_title("truth")

# %%
# Estimate with Granger causality
# -------------------------------
# Granger causality asks whether the past of one signal helps to predict another signal beyond what
# that signal's own past and the other signals' pasts already do. The estimator fits a
# multivariate autoregressive (MVAR) model; ``order="bic"`` lets the Bayesian information
# criterion choose how many past samples the model uses (tutorial 2 looks at that choice).
result = dirconn.gc(data, order="bic")
print(result)
print("model order:", result.provenance.model_order)

# %%
# The result as a table, and as a picture next to the truth.
table = pd.DataFrame(result.values[:, :, 0, 0], index=result.source, columns=result.target)
print(table.round(3))

# %%
left, right = result.plot(kind="truth_comparison", truth=sim.truth)
left.figure.suptitle("the truth we simulated, and what Granger causality found")

# %%
# Each panel has its own colour scale: the truth runs from 0 to 1, the estimate from 0 to its largest
# value. The two real influences stand out against the other cells. The value is in nats and has no
# upper limit of 1: it can only be compared between cells of the same result, not read as a
# probability. Note also that the matrix is not symmetric: the influence of the first region on the
# second is a different number from the influence of the second on the first.

# %%
# What a result remembers
# -----------------------
# Every result carries its provenance: how it was made, so that a figure or a number can be
# traced back. It is read-only and it is saved with the result.
provenance = result.provenance
print("method:", provenance.method)
print("order:", provenance.model_order)
print("claim:", provenance.claim)

# %%
# Saving and loading a result keeps the values, the axes and the provenance.
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "chain_gc.nc"
    result.to_netcdf(path)
    again = Connectivity.from_netcdf(path)
np.testing.assert_array_equal(np.nan_to_num(again.values), np.nan_to_num(result.values))
print("reloaded:", again.source == result.source, again.provenance.method)

# %%
# What this does and does not show
# --------------------------------
# * The estimate is **predictive**: it says that the past of the source helps to predict the
#   target. It does not say that the source causes the target; every result states this in its
#   ``claim``.
# * The result rests on assumptions. The library checks those it can; ``not_checked`` means that it
#   did not test the assumption, not that the assumption holds. They are printed with the result:
for check in result.provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")

# %%
# * What this tutorial did **not** test: it uses one simulated recording, so it says nothing about how
#   stable the numbers are from one recording to another; it applies no significance test (tutorial
#   4); and it does not look at frequency (tutorial 3), at the model order (tutorial 2) or at
#   situations in which the estimate is misleading (tutorial 5).
plt.close("all")

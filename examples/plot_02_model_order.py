"""
============================
How many past samples to use
============================

Granger causality and the spectral measures of the next tutorials all rest on a multivariate
autoregressive (MVAR) model: every signal is predicted from the last ``order`` samples of all
signals. The order is a choice, and this tutorial shows how the library helps to make it and what
happens to the answer when it is chosen badly.

It uses the same simulated chain as tutorial 1, so you can compare with the truth.
"""

# %%
# Setup and data
# --------------
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dirconn
from dirconn import Edge, Network, dk_node
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
# Let the data choose
# -------------------
# ``select_order`` fits MVAR models of order 1, 2, ... up to ``max_order`` and scores each with
# information criteria: they reward a model for predicting well and punish it for the number of
# coefficients it needs. Lower is better. AIC, BIC and HQC differ in how hard they punish.
MAX_ORDER = 12
selection = dirconn.select_order(data, MAX_ORDER, criterion="bic")
print("chosen by BIC:", selection.order)
for criterion in ("aic", "bic", "hqc"):
    chosen = dirconn.select_order(data, MAX_ORDER, criterion=criterion).order
    print(f"{criterion.upper()} chooses order {chosen}")

# %%
fig, ax = plt.subplots(layout="constrained")
for name, curve in (("AIC", selection.aic), ("BIC", selection.bic), ("HQC", selection.hqc)):
    ax.plot(selection.orders, curve, marker="o", label=name)
ax.axvline(selection.order, color="k", linestyle=":", label=f"BIC choice: {selection.order}")
ax.set_xlabel("model order (samples of the past)")
ax.set_ylabel("information criterion (lower is better)")
ax.legend()

# %%
# The curves fall steeply over the first orders and then flatten out: after that, a longer past
# adds coefficients without predicting noticeably better. At this sampling rate of 100 Hz an order
# of 5 means that the model looks back 50 ms.

# %%
# What a bad choice does
# ----------------------
# We now estimate the same Granger causality with different orders and compare the three
# influences that matter: the two real ones (first to second, second to third) and the one that
# does not exist (first to third).
orders = [1, 2, 3, selection.order, MAX_ORDER]
rows = []
for order in orders:
    # check_order_stability=False: the result would otherwise refit at much longer orders to test
    # how stable the direction is, which we do not need to show here
    result = dirconn.gc(data, order=order, check_order_stability=False)
    values = result.values[:, :, 0, 0]
    rows.append(
        {
            "order": order,
            "A -> B (real)": values[0, 1],
            "B -> C (real)": values[1, 2],
            "A -> C (absent)": values[0, 2],
        }
    )
table = pd.DataFrame(rows).set_index("order")
print(table.round(3))

# %%
fig, ax = plt.subplots(layout="constrained")
for column in table.columns:
    ax.plot(table.index, table[column], marker="o", label=column)
ax.set_xlabel("model order")
ax.set_ylabel("Granger causality (nats)")
ax.legend()

# %%
# In this recording an order of 1 gives less than half of what the orders from 3 upwards give for
# the two real influences, and those longer models give values of similar size to each other; the
# absent influence stays close to zero at every order. A model that is too short cannot see an
# influence that acts over several samples. The library warns when the criterion picks the largest
# candidate order, because then a longer one might have been better, and it records a check on the
# order, shown next.

# %%
# The order is part of the result
# -------------------------------
# By default ``order="bic"``, and the result keeps the curves it chose from and a check of how
# stable the direction of every pair is when the order is made longer.
result = dirconn.gc(data, order="bic")
print("order used:", result.provenance.model_order)
for name in ("order_selection", "order_stability"):
    check = result.provenance.assumptions.get(name)
    print(f"{check.name}: {check.status}. {check.detail}")

# %%
# What this does and does not show
# --------------------------------
# * Information criteria pick the order that predicts well; they do not know the true order, which
#   for real data does not exist. Different criteria can choose different orders.
# * The comparison above uses one simulated recording of one network: it shows the kind of
#   difference a short model can make, not how large it is in general.
# * Every result lists the assumptions it rests on; ``not_checked`` means that the library did not
#   test an assumption, not that it holds:
for check in result.provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")

# %%
# * Not tested here: the effect of the order on frequency-resolved measures (tutorial 3), whether
#   the model is stationary, and what happens when the order is chosen on one part of the data and
#   used on another.
plt.close("all")

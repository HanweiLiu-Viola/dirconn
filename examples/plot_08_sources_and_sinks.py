"""
=========================================
Sources and sinks: reading them with care
=========================================

A common way to summarise directed connectivity is to say which regions are *sources* (they send
more than they receive) and which are *sinks*, and to show it on the brain or between networks.
This tutorial computes these summaries (``net_direction`` and ``degree_imbalance``), draws them
(a chord diagram between resting-state networks and a map of the cortical surface), and then does
what the figures cannot do on their own: compares them with the known truth of a simulation, once
with the regions' own time series and once with regions estimated from sensors.

Every source/sink figure of dirconn carries the caption "Predictive, not causal. No leakage
correction." on purpose, and this tutorial shows why. It needs the MNE ``sample`` dataset
(about 1.5 GB), which dirconn never downloads without being asked.
"""

NEEDS_SAMPLE = True

# %%
# Setup
# -----
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dirconn
from dirconn import viz
from dirconn.datasets import require_sample
from dirconn.headmodel import load_head_model
from dirconn.inverse import roi_series
from dirconn.scenario import load_scenario
from dirconn.simulation import simulate_on_head

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")
print("sample data:", require_sample())

# %%
# A network with known sources and sinks
# --------------------------------------
# The shipped scenario ``random_10`` has ten regions and fifteen influences between them, drawn at
# random. We count, for every region, how many influences leave it and how many arrive: a region
# with only outgoing influences is a source in the truth, one with only incoming influences is a
# sink.
spec = load_scenario("random_10")
names = list(spec.network.names)
head = load_head_model("eeg")
sim = simulate_on_head(
    spec.network,
    head,
    sfreq=spec.sfreq,
    n_times=int((30.0 if QUICK else 60.0) * spec.sfreq),
    seed=SEED,
    dynamics=spec.dynamics,
    snr_db=spec.snr_db,
    n_background=10,
)
truth = np.nan_to_num(sim.truth.values[:, :, 0, 0])
known = pd.DataFrame(
    {"edges out": truth.sum(axis=1).astype(int), "edges in": truth.sum(axis=0).astype(int)},
    index=names,
)
print(known)

# %%
# Two sets of region time series
# ------------------------------
# As in tutorial 7: the regions' own activity (the oracle), and the regions estimated from the
# simulated sensors with an inverse solution. Granger causality is estimated on both.
ORDER = 4 if QUICK else 5
oracle = sim.roi_observations(snr_db=spec.snr_db, seed=SEED + 1)
estimated = roi_series(sim.data, head, names)
results = {
    "oracle": dirconn.gc(oracle, order=ORDER, check_order_stability=False),
    "from sensors": dirconn.gc(estimated, order=ORDER, check_order_stability=False),
}

# %%
# Net direction and degree imbalance
# ----------------------------------
# ``net_direction()`` is, for every pair, the influence of one on the other minus the reverse; it
# is antisymmetric. ``degree_imbalance()`` sums, for every region, the influences it sends and the
# influences it receives and takes ``log(sent) - log(received)``: positive for a source, negative for
# a sink. By default every defined edge counts, with its raw value; a boolean ``mask`` chooses the
# edges instead, and the choice is recorded in the result's provenance. Below the raw values and
# only the edges with a value of at least 0.05:
imbalance = {label: result.degree_imbalance() for label, result in results.items()}
masked = {
    label: result.degree_imbalance(mask=result.values[:, :, 0, 0] >= 0.05)
    for label, result in results.items()
}
table = known.copy()
for label in results:
    table[f"{label}, all edges"] = imbalance[label].values[:, 0, 0]
    table[f"{label}, edges >= 0.05"] = masked[label].values[:, 0, 0]
print(table.round(2).to_string())
print("NaN: a region without a counted edge in or out has no ratio")

# %%
# Look at the thresholded columns: a region with no counted edge into it (or out of it) has no ratio,
# because the logarithm of zero does not exist, so its value is NaN and the map leaves it
# uncoloured. A threshold therefore removes exactly the clearest sources and sinks, the regions that
# only send or only receive. Whether that is what you want depends on the question; the point is
# that the choice changes the picture, and that the figure must say which one was made.

# %%
# The regions with a clear truth (several influences out and none in, or the reverse) are the ones
# to look at. For them, with the regions' own series the sign of the imbalance is that of the
# truth. With the regions estimated from sensors, compare the sizes:
clear = known[
    ((known["edges out"] >= 2) & (known["edges in"] == 0))
    | ((known["edges in"] >= 2) & (known["edges out"] == 0))
].index
comparison = pd.DataFrame(
    {
        "truth": ["source" if known.loc[n, "edges out"] > 0 else "sink" for n in clear],
        "oracle": [imbalance["oracle"].sel(node=n).item() for n in clear],
        "from sensors": [imbalance["from sensors"].sel(node=n).item() for n in clear],
    },
    index=clear,
)
print(comparison.round(2))

# %%
# The imbalances of the sensor-based estimate point the same way for these regions but are much
# smaller in size, and for one of them close to zero. The overall pattern of the brain map therefore
# changes with the data the regions come from.

# %%
# Between networks
# ----------------
# Regions can be grouped, for example into the seven resting-state networks of Yeo and colleagues
# (2011). ``viz.yeo_groups`` assigns every Desikan-Killiany region to the network that holds most of
# its vertices on the template brain, computed from the annotation files of the sample dataset. For
# regions that straddle networks, the table says how much of the region the network holds.
yeo = viz.yeo_networks().loc[names]
print(yeo.round(2))
groups = viz.yeo_groups(names)

# %%
# The chord diagram shows, between networks, the stronger direction of every pair. Look at the
# caption at the bottom of the figure: it is there by design.
fig = viz.chord_diagram(results["oracle"], groups, measure="net_direction")

# %%
# On the cortical surface
# -----------------------
# ``degree_imbalance_map`` colours every region by its imbalance, red for sources and blue for
# sinks, on the template brain, with MNE's 3D plot (this needs a 3D backend of MNE and takes a
# little while). The title says which edges were counted. Both maps use the same colour scale.
vmax = max(np.nanmax(np.abs(v.values)) for v in imbalance.values())
fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.0), layout="constrained")
for ax, (label, values) in zip(axes, imbalance.items(), strict=True):
    brain = viz.degree_imbalance_map(values, vmax=vmax, degrees="all edges, raw values")
    ax.imshow(brain.screenshot())
    brain.close()
    ax.axis("off")
    ax.set_title(f"regions: {label}")

# %%
# What this does and does not show
# --------------------------------
# * A source or a sink here is a statement about **prediction**: the past of the region helps to
#   predict others more than the other way round. It is not a statement that the region drives the
#   others. Tutorial 5 showed influences that are not there, and tutorial 7 showed leakage;
#   neither is corrected in these figures.
# * With the regions' own series the summaries recover the sources and sinks that were built into
#   the simulation. With regions estimated from sensors the same summaries are weaker for the same
#   regions and the picture differs. This was seen for one simulation of one network on one head
#   model with one inverse method; how much it changes in general is not shown here.
# * The summaries depend on choices: which edges are counted (all, or only the strong ones), how
#   the regions are grouped, how the regions were obtained. The figures state the first two; the
#   third is yours to state.
# * Every result lists the assumptions it rests on; ``not_checked`` means that the library did not
#   test an assumption, not that it holds:
for check in results["from sensors"].provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")

# %%
# * Not tested here: the significance of an imbalance (a ``mask`` made from a significance test
#   can be passed to ``degree_imbalance``), and anything about real recordings.
plt.close("all")

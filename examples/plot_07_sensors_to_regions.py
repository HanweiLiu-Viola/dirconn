"""
====================================
From sensors to regions, and leakage
====================================

Recordings are made at sensors, but the questions are about brain regions. The usual bridge is an
inverse solution: it estimates the activity of each region from the sensors. This tutorial
simulates sensor data from known regions on a real head model, estimates the region time series
back from the sensors, and compares the directed connectivity of the estimated series with that of
the true ones. Two neighbouring regions show what goes wrong: the estimate of one contains the
other, and the connectivity then contains influences that were never there.

This tutorial needs the MNE ``sample`` dataset (about 1.5 GB). It is never downloaded by
dirconn without being asked; see ``dirconn.datasets.require_sample``.
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
from dirconn.datasets import require_sample
from dirconn.headmodel import load_head_model
from dirconn.inverse import crosstalk, roi_series
from dirconn.regions import centre_distances
from dirconn.scenario import load_scenario
from dirconn.simulation import simulate_on_head

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")
print("sample data:", require_sample())  # stops with instructions when the data are missing

# %%
# A head model
# ------------
# The head model of the sample subject links every cortical region to the 59 EEG sensors: the
# *leadfield* says how the activity of a region appears at each sensor. It also holds the
# parcellation into 68 Desikan-Killiany regions. Loading takes a few seconds.
head = load_head_model("eeg")
print(f"{head.n_channels} EEG channels, {len(head.roi_names)} regions")

# %%
# A scenario with two close regions
# ---------------------------------
# The shipped scenario ``leakage_adjacent`` has three regions: Z drives X, and Y is a close
# neighbour of X that nothing drives. Every influence into Y that an analysis reports is therefore
# false. The distances are those between the region centres on the head model, in millimetres.
spec = load_scenario("leakage_adjacent")
Z, X, Y = spec.network.names
print(spec.description)
distances = centre_distances(head)
index = {name: head.roi_names.index(name) for name in spec.network.names}
for a, b in ((X, Y), (Z, X)):
    print(f"distance {a} - {b}: {distances[index[a], index[b]]:.0f} mm")

# %%
# Simulate the recording
# ----------------------
# ``simulate_on_head`` lets every region of the network oscillate, projects it through the leadfield
# to the sensors, adds the activity of other sources in the brain (``n_background``) and noise, so
# that the sensors have a signal-to-noise ratio of 10 dB. The simulation also keeps the true
# time series of the regions, which a real recording never has.
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
print(sim.data)

# %%
# Two ways to the regions' time series
# ------------------------------------
# * ``sim.roi_observations`` gives the regions' own activity plus a little noise (tutorial 1 did
#   this): the *oracle*, the data we would like to have.
# * ``roi_series`` is what an analysis of a real recording has to do: estimate the regions from the
#   sensors with an inverse method (here the minimum-norm estimate, with the sign of every source
#   aligned by a principal component, ``"pca_flip"``).
oracle = sim.roi_observations(snr_db=spec.snr_db, seed=SEED + 1)
estimated = roi_series(sim.data, head, list(spec.network.names))
print(oracle.names == estimated.names)

# %%
# The same Granger causality on both
# ----------------------------------
# oracle: the regions' own series; from sensors: the regions estimated by the inverse method
results = {
    "oracle": dirconn.gc(oracle, order=5, check_order_stability=False),
    "from sensors": dirconn.gc(estimated, order=5, check_order_stability=False),
}
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), layout="constrained")
sim.truth.plot(ax=axes[0], colorbar=False)
axes[0].set_title("truth")
for ax, (label, result) in zip(axes[1:], results.items(), strict=True):
    result.plot(ax=ax)
    ax.set_title(label)
cells = {
    "Z -> X (real)": (Z, X),
    "Z -> Y (no such influence)": (Z, Y),
    "X -> Y (no such influence)": (X, Y),
    "Y -> X (no such influence)": (Y, X),
}
table = pd.DataFrame(
    {
        label: {
            name: result.values[result.source.index(a), result.target.index(b), 0, 0]
            for name, (a, b) in cells.items()
        }
        for label, result in results.items()
    }
)
print(table.round(3))

# %%
# With the regions' own series only the real influence appears. With the regions estimated from the
# sensors, the real influence is smaller and influences appear that point *into* Y, which nothing
# drives, and between X and Y in both directions.

# %%
# Where it comes from: leakage
# ----------------------------
# An inverse solution cannot separate close sources perfectly: the estimate of one region contains
# a part of the activity of its neighbours. ``crosstalk`` measures it. It makes every region of the
# head model active with independent white activity, projects it to the sensors, adds sensor noise,
# extracts the regions of interest exactly as ``roi_series`` does, and asks how much of each
# region's activity arrives in each extracted series. Entry ``[i, j]`` is the response of series
# ``i`` to the activity of region ``j``, relative to the response of series ``i`` to its own
# region (so the diagonal is 1 and a value above 1 means that a neighbour is heard louder than the
# region itself). It is estimated from random probe data, so it changes a little with the seed.
leak = crosstalk(head, list(spec.network.names), seed=SEED + 2)
leak_table = pd.DataFrame(leak.matrix, index=leak.rois, columns=leak.rois)
print("rows: series that receives; columns: region that leaks in")
print(leak_table.round(2))
print(f"X leaks into the series of Y with {leak.matrix[2, 1]:.2f} times Y's own response")

# %%
# The series that was estimated for Y contains the activity of X, in this simulation more strongly
# than Y's own activity. That explains the influences into Y: Z drives X, and X is in Y's series.
# The oracle series, which have no such mixing, are the control: with them the influences into Y
# are absent.

# %%
# What this does and does not show
# --------------------------------
# * The influences into Y that appear in the sensor-based analysis are **leakage**, not
#   connections: nothing drives Y in the simulation, and the same estimator finds none in the
#   regions' own series.
# * This is one simulated recording of one pair of close regions on one head model, with one inverse
#   method and EEG only. How much leakage there is for other pairs, other methods and MEG is not
#   shown here. The benchmark report (``docs/benchmark_results.md``) measures leakage across many
#   pairs under stated conditions.
# * dirconn does not correct for leakage. The crosstalk matrix is a way to see where it is likely
#   to be a problem, and an estimate of its size, not a correction.
# * Every result lists the assumptions it rests on; ``not_checked`` means that the library did not
#   test an assumption, not that it holds:
for check in results["from sensors"].provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")

# %%
# * Not tested here: whether a different inverse method or a different head model changes the
#   picture, and what happens with real recordings, where neither the regions nor their number
#   are known.
plt.close("all")

"""
======================================
Directed connectivity across frequency
======================================

Granger causality gives one number per pair of regions. Brain rhythms live at particular
frequencies, so it is often more useful to ask *at which frequencies* one region influences
another. This tutorial uses three measures that do that, the directed transfer function (DTF),
partial directed coherence (PDC) and generalised PDC (GPDC), on a simulation in which one region
drives another only inside the alpha band.
"""

# %%
# Setup and data
# --------------
# The scenario ``band_alpha`` ships with dirconn: region A drives region B with a delay of 20 ms,
# and the influence acts only inside the alpha band (8 to 13 Hz); a third region, C, is not
# connected to anything. The truth is therefore a function of frequency.
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dirconn
from dirconn.bands import BANDS
from dirconn.scenario import load_scenario

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")

spec = load_scenario("band_alpha")
A, B, C = spec.network.names
print(spec.description)
sim = dirconn.simulate(
    spec.network,
    sfreq=spec.sfreq,
    n_times=int((30.0 if QUICK else spec.duration_s) * spec.sfreq),
    seed=SEED,
    dynamics=spec.dynamics,
    snr_db=spec.snr_db,
    n_background=0,
    node_scale="unit",
)
data = sim.roi_observations(snr_db=spec.snr_db, seed=SEED + 1)
print(data)

# %%
# Three measures, one call each
# -----------------------------
# All three fit the same kind of MVAR model as Granger causality (tutorial 2) and then look at its
# frequency response. They differ in how they normalise:
#
# * **DTF** measures the influence of a source on a target as a fraction of everything that
#   flows *into* the target at that frequency.
# * **PDC** measures it as a fraction of everything that flows *out of* the source.
# * **GPDC** is PDC with each signal weighted by the inverse of its own noise level, which makes
#   it independent of the scale of the signals.
#
# The values lie between 0 and 1. ``n_freqs`` is the number of frequencies between 0 Hz and
# ``fmax``.
results = {
    "DTF": dirconn.dtf(data, n_freqs=64, fmax=40.0),
    "PDC": dirconn.pdc(data, n_freqs=64, fmax=40.0),
    "GPDC": dirconn.gpdc(data, n_freqs=64, fmax=40.0),
}
for name, result in results.items():
    print(name, result, "order", result.provenance.model_order)

# %%
# Both directions of one pair
# ---------------------------
# ``kind="directed_spectrum"`` draws one pair in both directions. The grey band is the alpha band,
# where the truth of A to B is 1 and elsewhere 0.
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12.0, 3.6), layout="constrained")
for ax, (name, result) in zip(axes, results.items(), strict=True):
    result.plot(kind="directed_spectrum", source=A, target=B, ax=ax)
    ax.axvspan(*BANDS["alpha"], color="0.5", alpha=0.15)
    ax.set_title(name)

# %%
# In all three, the influence of A on B peaks inside the alpha band and falls away on both sides,
# and the reverse direction stays close to zero. The estimate does not fall away as sharply as the
# truth does, so the neighbouring theta band also gets values above zero (see the table below).
# In this simulation DTF and PDC give almost the same curve and GPDC a higher one. The measures
# normalise differently, so their heights cannot be compared between measures; compare
# frequencies and directions *within* a measure.

# %%
# From a spectrum to a band
# -------------------------
# Often one number per band is wanted. ``reduce_freq`` averages a spectrum over a band; the names
# come from ``dirconn.bands.BANDS`` and you can also pass ``(fmin, fmax)`` in Hz. Other reductions
# are ``"median"``, ``"max"`` and ``"peak_freq"`` (the frequency of the peak, instead of its
# height).
print(BANDS)
rows = []
for name, result in results.items():
    for band in ("theta", "alpha", "beta"):
        values = result.reduce_freq(band).values[:, :, 0, 0]
        rows.append({"measure": name, "band": band, "A -> B": values[0, 1], "B -> A": values[1, 0]})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# %%
# The whole matrix of a band, next to the truth for that band. ``sim.truth_at`` gives the truth on
# the frequency axis of the estimate; inside the band it is 1, and the mean over the band is
# drawn.
pdc = results["PDC"]
left, right = pdc.plot(kind="truth_comparison", truth=sim.truth_at(freqs=pdc.freqs), band="alpha")
left.figure.suptitle("PDC in the alpha band")

# %%
# The estimate shows the influence of A on B, as the truth does, and the cells of region C, which
# is not connected to anything, stay low. The diagonal is left empty in every matrix, also where a
# measure defines it: the share of a region's own activity is not a connection between regions.

# %%
# What this does and does not show
# --------------------------------
# * The three measures are **predictive**, like Granger causality: they show where the past of one
#   signal helps to predict another, not that one region causes another.
# * In this simulation the influence is confined to one band, both regions oscillate in the middle
#   of that band, and there is no region that drives both; this is the easy case. Whether the
#   same holds when the coupling is weaker, when rhythms overlap, or when a hidden region drives
#   both is not shown here (tutorial 5 shows some such cases).
# * Every result lists the assumptions it rests on; ``not_checked`` means that the library did not
#   test an assumption, not that it holds:
for check in pdc.provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")

# %%
# * Not tested here: whether any single value is significantly different from what noise would give
#   (tutorial 4), how the result depends on the model order, and anything about real recordings.
plt.close("all")

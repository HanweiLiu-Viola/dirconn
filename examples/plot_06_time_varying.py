"""
===================================
Connectivity that changes over time
===================================

Brain networks do not stay the same: an influence can switch on when a task starts and off when it
ends. A single estimate over a whole recording averages over such changes. This tutorial uses a
simulation in which one influence switches on half-way through, and shows three ways to follow it:
a time-varying model (ADTF and TPDC), one model per window or segment (WDTF), and a reduction of a
time-resolved result over segments.
"""

# %%
# Setup and data
# --------------
# The scenario ``switch_on`` ships with dirconn. It lasts 60 s: region A drives region B from 30 s on
# (and not before), and region B drives region C all the time, as a control that does not change.
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dirconn
from dirconn.scenario import load_scenario

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")

spec = load_scenario("switch_on")
A, B, C = spec.network.names
SWITCH_S = 30.0
sim = dirconn.simulate(
    spec.network,
    sfreq=spec.sfreq,
    n_times=int(spec.duration_s * spec.sfreq),
    seed=SEED,
    dynamics=spec.dynamics,
    snr_db=spec.snr_db,
    n_background=0,
    node_scale="unit",
)
data = sim.roi_observations(snr_db=spec.snr_db, seed=SEED + 1)
print(spec.description)
print(data)

# %%
# What one estimate over everything gives
# ---------------------------------------
# DTF over the whole recording (tutorial 3) gives one value per pair and frequency. Averaged over
# frequency, the influence of A on B comes out between "absent" and "present", because it is
# present for half of the time. The estimate cannot say that it was off and then on.
static = dirconn.dtf(data, order=3, n_freqs=41, fmax=40.0)
print("whole recording, mean over frequency")
print("  A -> B:", np.nanmean(static.values[0, 1]).round(3))
print("  B -> C:", np.nanmean(static.values[1, 2]).round(3))

# %%
# A model whose coefficients follow the data
# ------------------------------------------
# ``adtf`` fits a vector autoregressive model whose coefficients are allowed to change from one
# time point to the next (a Kalman filter) and computes the DTF at every time point; ``tpdc`` does
# the same for partial directed coherence. ``time_step`` is the spacing of the output in seconds. The
# result has a time axis in addition to the frequency axis, and ``kind="time_frequency"`` draws a
# pair as a panel.
TIME_STEP = 2.0 if QUICK else 1.0


def fit_adtf(label, **options):
    """Run ``adtf`` and print the warnings the library gives instead of letting them scroll by."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = dirconn.adtf(data, order=3, fmax=40.0, n_freqs=41, time_step=TIME_STEP, **options)
    print(f"{label}: {len(caught)} warning(s) from the library")
    for item in caught:
        print("  -", str(item.message))
    return result


adtf = fit_adtf("default settings")
print(adtf)

# %%
# The library warns, and the warning matters. The filter forgets old data, so the coefficients at
# any time point are effectively estimated from a short stretch of the recording (the measured
# length is in ``adtf.provenance.details["memory_samples"]``); the model has several coefficients
# per equation to estimate from that stretch, and the library says that this is thin. Read the
# curves below as a description of how the estimate behaves in this simulation. A longer memory or
# a lower order reduces the problem and makes the estimate follow changes more slowly; the trade is
# explained in ``docs/timevarying.md``.

# %%
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(11.0, 3.8), layout="constrained")
top = max(np.nanmax(adtf.values[0, 1]), np.nanmax(adtf.values[1, 2]))
adtf.plot(kind="time_frequency", source=A, target=B, ax=axes[0], vmin=0.0, vmax=top, colorbar=False)
adtf.plot(kind="time_frequency", source=B, target=C, ax=axes[1], vmin=0.0, vmax=top)
for ax in axes:
    ax.axvline(SWITCH_S, color="k", linestyle=":")
axes[0].set_title("ADTF, A → B (switches on at the dotted line)")
axes[1].set_title("ADTF, B → C (constant)")

# %%
# The influence of A on B appears after the dotted line and the constant control stays where it was.
# Averaged over frequency, the same picture as curves, together with the single value over the
# whole recording from above (dashed). The estimate does not jump at the switch: it needs some time
# to reach the new level, which the printout measures as the first output time after the switch at
# which the influence has covered half of the way between its two levels.
mean_ab = np.nanmean(adtf.values[0, 1], axis=0)
mean_bc = np.nanmean(adtf.values[1, 2], axis=0)
before = mean_ab[adtf.times < SWITCH_S - 2.0].mean()
after = mean_ab[adtf.times > SWITCH_S + 10.0].mean()
half = adtf.times[(adtf.times > SWITCH_S) & (mean_ab >= (before + after) / 2)][0]
print(f"A -> B: {before:.3f} before, {after:.3f} well after; half-way at {half:g} s")
fig, ax = plt.subplots(layout="constrained")
ax.plot(adtf.times, mean_ab, label="A → B")
ax.plot(adtf.times, mean_bc, linestyle="--", label="B → C")
ax.axhline(
    np.nanmean(static.values[0, 1]), color="0.5", linestyle=":", label="A → B, whole recording"
)
ax.axvline(SWITCH_S, color="k", linestyle=":")
ax.set_xlabel("time (s)")
ax.set_ylabel("DTF, mean over 0-40 Hz")
ax.legend()

# %%
# How fast the model may change
# -----------------------------
# The ``forgetting`` argument sets how fast the coefficients are allowed to move. The default
# follows slowly; a small value follows quickly. Following quickly is not free: the estimate then
# moves with the noise too. Below, the same data with ``forgetting=0.99``:
fast = fit_adtf("forgetting=0.99", forgetting=0.99)
mean_fast = np.nanmean(fast.values[0, 1], axis=0)
mask = fast.times < SWITCH_S - 2.0
print("A -> B before the switch, where the truth is 0:")
print(f"  default forgetting: mean {mean_ab[mask].mean():.3f}, spread {mean_ab[mask].std():.3f}")
print(
    f"  forgetting=0.99:    mean {mean_fast[mask].mean():.3f}, spread {mean_fast[mask].std():.3f}"
)
fig, ax = plt.subplots(layout="constrained")
ax.plot(adtf.times, mean_ab, label="default forgetting")
ax.plot(fast.times, mean_fast, label="forgetting=0.99")
ax.axvline(SWITCH_S, color="k", linestyle=":")
ax.set_xlabel("time (s)")
ax.set_ylabel("A → B, DTF mean over 0-40 Hz")
ax.legend()

# %%
# The fast filter shows an influence of A on B *before* the switch, where there is none, and its
# curve is much more irregular. What "forgetting" means for this recursion, and why its value is not
# a memory length in samples, is explained in ``docs/timevarying.md``; the library records the
# memory it measured in ``result.provenance.details["memory_samples"]``.
print(
    "measured memory (samples):",
    {
        "default": round(adtf.provenance.details["memory_samples"], 1),
        "forgetting=0.99": round(fast.provenance.details["memory_samples"], 1),
    },
)

# %%
# One model per window or per segment
# -----------------------------------
# ``wdtf`` takes the opposite approach: it cuts the recording into windows (or into segments that
# you name) and fits an ordinary model to each. Nothing is assumed about how the coefficients move,
# but a window must be long enough to fit a model, and an influence that switches inside a window
# is averaged. With ``segments`` you decide where the cuts are:
segments = [(5.0, 25.0), (35.0, 55.0)]  # before and after the switch, away from the transition
per_segment = dirconn.wdtf(data, segments=segments, order=3, fmax=40.0, n_freqs=41)
print(per_segment)

# %%
# The time-resolved result can be reduced over the same segments, with ``reduce_time``. The two
# ways to the same question give similar numbers here:
averaged = adtf.reduce_time(segments)
rows = []
for label, result in (("ADTF, then reduce_time", averaged), ("WDTF per segment", per_segment)):
    ab = np.nanmean(result.values[0, 1], axis=0)
    bc = np.nanmean(result.values[1, 2], axis=0)
    rows.append(
        {
            "method": label,
            "A -> B before": ab[0],
            "A -> B after": ab[1],
            "B -> C before": bc[0],
            "B -> C after": bc[1],
        }
    )
print(pd.DataFrame(rows).round(3).to_string(index=False))

# %%
# The two can differ in how they are made, and only one of them has a test: segment means of a
# continuous ADTF are descriptive, with no p-value; a comparison of two segments with a
# permutation test needs epoched data and is shown in ``docs/timevarying.md``.

# %%
# The network before and after
# ----------------------------
# ``kind="network_graph"`` draws the nodes on a circle with an arrow for every edge you choose, here
# every pair with a DTF of at least 0.1, averaged over 1 to 40 Hz. The edges are always chosen by
# you, with ``threshold`` or ``mask``, because the picture changes with that choice.
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), layout="constrained")
for ax, time_s, label in zip(axes, (15.0, 45.0), ("before", "after"), strict=True):
    averaged.plot(kind="network_graph", threshold=0.1, band=(1.0, 40.0), time=time_s, ax=ax)
    ax.set_title(f"{label} the switch (segment centred at {time_s:g} s)")

# %%
# Before the switch there is an arrow from B to C; afterwards there is one from A to B as well. A
# third arrow, from A to C, appears after the switch, although this scenario has no direct
# influence from A on C: tutorial 5 shows how a path through a third region can produce an
# influence between the other two.

# %%
# What this does and does not show
# --------------------------------
# * The influence switched on at one moment, abruptly, in one scenario, with the other influence
#   constant. How these methods behave with gradual changes, with several changes, or when the
#   regions' own rhythms change is not shown here.
# * The library warned about the sampling adequacy of the time-varying fit, see above; the curves
#   are descriptions of this simulation, not measurements with an error bar.
# * The curve of a time-varying estimate follows the truth with a delay and, with a fast filter,
#   with noise. Where a value in time begins and ends is not read off a curve without looking at that
#   delay.
# * No method is recommended here. ADTF and WDTF answered the same question with similar numbers
#   in this scenario, which says nothing about other settings.
# * Every result lists the assumptions it rests on; ``not_checked`` means that the library did not
#   test an assumption, not that it holds:
for check in adtf.provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")

# %%
# * Not tested here: significance of a change over time, and anything about real recordings, whose
#   changes are neither abrupt nor known beforehand.
plt.close("all")

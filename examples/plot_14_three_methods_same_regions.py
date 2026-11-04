"""
============================================================
Three methods on the same regions: what the data decide
============================================================

Tutorial 8 read sources and sinks from Granger causality, once with the regions' own time series and
once with regions estimated from the EEG sensors. Is what it found specific to that method? This
tutorial analyses the same simulation with three methods: Granger causality (time domain, one model
of all regions), transfer entropy (time domain, model-free, one pair at a time; tutorial 12) and the
adaptive DTF (frequency domain; tutorial 6). Each result is compared with the known truth in three
ways: do the true influences score above the other pairs, what happens to pairs that are linked only
through a third region, and is the delay of an influence found.

It is a demonstration with one simulation, not an evaluation of the methods. It needs the MNE
``sample`` dataset (about 1.5 GB), which dirconn never downloads without being asked. The full
version runs for about ten minutes, most of it for transfer entropy.
"""

NEEDS_SAMPLE = True

# %%
# Setup
# -----
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import dirconn
from dirconn.datasets import require_sample
from dirconn.headmodel import load_head_model
from dirconn.inverse import roi_series
from dirconn.scenario import load_scenario
from dirconn.simulation import simulate_on_head

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")
print("sample data:", require_sample())

# the neighbour searches of transfer entropy use every core; the values do not depend on it
WORKERS = -1

# %%
# The simulation of tutorial 8
# ----------------------------
# The shipped scenario ``random_10``: ten regions and fifteen influences between them, with delays
# of 10 to 30 ms and oscillations between 8 and 30 Hz, projected to the EEG sensors of the MNE
# ``sample`` head model with ten background sources. Besides the true influences we mark the pairs
# that are linked only through a third region (A drives B, B drives C, but A does not drive C): a
# method that sees the whole network should leave them low, a method that looks at one pair at a
# time may not.
spec = load_scenario("random_10")
names = list(spec.network.names)
n = len(names)
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
truth = np.nan_to_num(sim.truth.values[:, :, 0, 0]).astype(bool)
off = ~np.eye(n, dtype=bool)
indirect = ((truth.astype(int) @ truth.astype(int)) > 0) & ~truth & off
other = off & ~truth & ~indirect
true_delay = np.zeros((n, n), dtype=int)
for edge in spec.network.edges:
    true_delay[names.index(edge.source), names.index(edge.target)] = round(
        edge.delay_ms * spec.sfreq / 1000.0
    )
print(f"{truth.sum()} true influences, {indirect.sum()} indirect pairs, {other.sum()} other pairs")

# %%
# The regions' own series and the regions estimated from the sensors, as in tutorials 7 and 8:
series = {
    "own series": sim.roi_observations(snr_db=spec.snr_db, seed=SEED + 1),
    "from sensors": roi_series(sim.data, head, names),
}

# %%
# Three methods
# -------------
# Granger causality and the adaptive DTF are both computed from a vector autoregressive model of
# all ten regions, the first in the time domain, the second in the frequency domain; transfer
# entropy is computed for one pair at a time (its default) and scans the delay of the influence.
# The network of this simulation does not change over time, so the adaptive DTF is averaged over
# the whole recording, and over the band of the oscillators. Its library warning matters: the
# recursion has a short memory (tutorial 6), so a single time point would be unreliable; the
# average over the recording is what we use here.
ORDER = 4 if QUICK else 5
BAND = (8.0, 30.0)
MAX_DELAY = 5  # samples; the simulated delays are 1 to 3


def fit_all(data):
    """Fit the three methods to one set of region series."""
    gc = dirconn.gc(data, order=ORDER, check_order_stability=False)
    te = dirconn.te(data, mode="pairwise", max_delay=MAX_DELAY, workers=WORKERS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adtf = dirconn.adtf(data, order=ORDER, fmax=40.0, n_freqs=41, time_step=0.5)
    for item in caught:
        print("  ADTF warning:", str(item.message))
    span = (float(adtf.times[0]), float(adtf.times[-1]))
    return {"GC": gc, "TE": te, "ADTF": adtf.reduce_freq(BAND).reduce_time(span)}


results = {}
for label, data in series.items():
    print(label)
    results[label] = fit_all(data)
methods = list(results["own series"])

# %%
# The matrices, with the true influences framed
# ---------------------------------------------
# One column per method, one colour scale per method. Raw transfer entropy is biased and can be
# slightly negative (tutorial 12), so its scale is centred on zero in both rows.
fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.5), layout="constrained")
for col, method in enumerate(methods):
    # the scale covers the pairs only: the diagonal of the DTF (a region's share of its own
    # activity) is left empty by the plot and would wash out the other cells
    top = max(np.nanmax(np.abs(res[method].values[:, :, 0, 0][off])) for res in results.values())
    # one scale for both rows: centred on zero for both as soon as one of them has a negative value
    signed = any(np.nanmin(res[method].values[:, :, 0, 0][off]) < 0 for res in results.values())
    scale = {"vmin": -top, "vmax": top, "cmap": "RdBu_r"} if signed else {"vmax": top}
    for row, (label, res) in enumerate(results.items()):
        ax = axes[row, col]
        conn = res[method]
        # the averaged adaptive DTF keeps one frequency and one time point; the plot names them
        where = {"freq": conn.freqs[0]} if conn.is_frequency_resolved else {}
        where |= {"time": conn.times[0]} if conn.is_time_resolved else {}
        conn.plot(ax=ax, **scale, **where)
        for i, j in zip(*np.nonzero(truth), strict=True):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=1.8))
        ax.set_title(f"{method}, {label}")
        if col:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        if row == 0:
            ax.set_xlabel("")
            ax.set_xticklabels([])

# %%
# With the regions' own series, the framed cells stand out for all three methods. Granger
# causality, which conditions every pair on the other regions, keeps the unframed cells close to
# zero. The other two light up some of them: the adaptive DTF many of the pairs linked through a
# third region, transfer entropy a few. From the sensors the framed cells fade for all three, and
# the adaptive DTF shows a whole bright row: a region that looks like a strong source although it
# sends only one influence.

# %%
# How well do the true influences score?
# --------------------------------------
# One number per result: the chance that a true influence scores above a pair without one (the
# area under the ROC curve; 1 means every true influence is above every other pair, 0.5 is
# guessing). It uses only the ranks of the values, so it can be compared across methods whose
# values have different units. The mean percentile ranks show where the three kinds of pairs fall.


def auc(values, labels):
    """Chance that a value with label True is above one with label False (ties count half)."""
    pos, neg = values[labels], values[~labels]
    above = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(above / (pos.size * neg.size))


def percentile_ranks(matrix):
    ranks = np.full(matrix.shape, np.nan)
    order = np.argsort(np.argsort(matrix[off]))
    ranks[off] = 100.0 * (order + 0.5) / off.sum()
    return ranks


rows = []
for label, res in results.items():
    for method, conn in res.items():
        matrix = conn.values[:, :, 0, 0]
        ranks = percentile_ranks(matrix)
        rows.append(
            {
                "series": label,
                "method": method,
                "AUC": auc(matrix[off], truth[off]),
                "rank: true": np.nanmean(ranks[truth]),
                "rank: indirect": np.nanmean(ranks[indirect]),
                "rank: other": np.nanmean(ranks[other]),
            }
        )
scores = pd.DataFrame(rows).set_index(["series", "method"])
print(scores.round(2).to_string())

# %%
# With the regions' own series all three methods put the true influences near the top. From the
# sensors, all three fall, and they fall much more than they differ from each other: here, the
# data the regions come from decide more than the choice between time and frequency domain. The
# one clear difference between the methods is in the indirect pairs, which the adaptive DTF ranks
# well above the other pairs: the DTF counts the flow through other regions by definition. With
# one simulation and fifteen true influences, differences of a few hundredths between the methods
# mean nothing.

# %%
# What the time domain adds: the delay
# ------------------------------------
# Transfer entropy keeps, for every pair, the delay (in samples) that gave the largest value; the
# chosen delays are in its provenance. On the true influences we compare them with the simulated
# delays. Guessing one of the ``MAX_DELAY`` delays would be right for one influence in five.
for label, res in results.items():
    chosen = np.array(res["TE"].provenance.details["delay"], dtype=float)
    exact = int((chosen[truth] == true_delay[truth]).sum())
    close = int((np.abs(chosen[truth] - true_delay[truth]) <= 1).sum())
    print(
        f"{label:13s} exact delay on {exact} of {truth.sum()} true influences, within one sample "
        f"on {close}; guessing: about {truth.sum() / MAX_DELAY:.0f}"
    )

# %%
# With the regions' own series the exact delay is found for most influences; from the sensors, where
# every region mixes in its neighbours, it is found about as often as by guessing, although the
# chosen delay is often off by only one sample. The adaptive DTF, averaged over a band, gives no
# delay at all.

# %%
# What this does and does not show
# --------------------------------
# * Every value here is about **prediction**: the past of one region helps to predict another. None
#   of the three methods shows that a region drives another, and the regions estimated from sensors
#   were not corrected for leakage (tutorial 7).
# * The comparison is one simulation of one network, on one head model with one inverse method. It
#   shows that the step from the regions' own series to regions estimated from sensors can matter
#   more than the method; it does not show how much in general, nor which method is better.
# * Transfer entropy was computed pair by pair and its values are raw and untested; the surrogate
#   test of tutorial 12 is needed before reading any single value. The adaptive DTF was averaged
#   over a network that does not change; its time resolution, the reason to use it, is not used.
# * Every result lists the assumptions it rests on; ``not_checked`` means that the library did not
#   test an assumption, not that it holds:
for method, conn in results["from sensors"].items():
    print(method)
    for check in conn.provenance.assumptions.checks:
        print(f"  {check.name:24s} {check.status:12s} {check.detail}")

# %%
# * Not tested here: other networks, other head models and inverse methods, more seeds, and
#   anything about real recordings.
plt.close("all")

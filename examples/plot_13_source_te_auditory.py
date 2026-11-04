"""
===================================================================
Transfer entropy between EEG sources: one hypothesis, chosen first
===================================================================

Directed connectivity between estimated brain regions is easy to compute and hard to read: every
pair of regions gives a number, the regions leak into each other, and with enough pairs some look
convincing by chance. This tutorial runs transfer entropy (tutorial 12) on the EEG of the MNE
``sample`` recording, and spends most of its length on what comes before the estimate:

1. the regions and the direction to test are chosen **from the data, by rules fixed in advance**,
   without looking at any directed measure;
2. a simulation on the same head model asks whether this pipeline can find a known direction at
   all, and whether it reports one where there is none;
3. only then is the single hypothesis tested, with a surrogate test; every other pair is shown as
   exploration, without a test.

It is a demonstration, not an evaluation: one subject, one session, one inverse method.

This tutorial needs the MNE ``sample`` dataset (about 1.5 GB). It is never downloaded by
dirconn without being asked; see ``dirconn.datasets.require_sample``. The full version runs for
about an hour; the quick version (``DIRCONN_QUICK=1``) only checks that the code runs and
uses too few surrogates for any conclusion.
"""

NEEDS_SAMPLE = True

# %%
# Setup
# -----
import os
from itertools import combinations

import matplotlib.pyplot as plt
import mne
import numpy as np

import dirconn
from dirconn import TE, Edge, Network, NodeDynamics, TimeSeries, dk_node
from dirconn.channels import ChannelMeta
from dirconn.datasets import require_sample
from dirconn.headmodel import load_head_model
from dirconn.inverse import crosstalk, roi_series
from dirconn.simulation import simulate_on_head
from dirconn.surrogates import TrialShuffle

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}, mne {mne.__version__}")
root = require_sample()  # stops with instructions when the data are missing
print("sample data:", root)

# the neighbour searches of the estimator use every core; the values do not depend on it
WORKERS = -1

# %%
# The recording
# -------------
# The sample subject heard tones in the left or the right ear and saw checkerboards in the left or
# the right visual field. We keep the 59 good EEG channels, re-reference them to their average,
# filter them between 1 and 40 Hz and cut epochs from -0.2 to 0.5 s around each stimulus. Two
# copies are kept: one decimated to 150 Hz for the region series and transfer entropy, one at the
# full rate (600 Hz) for the latencies.
EVENT_ID = {"aud_l": 1, "aud_r": 2, "vis_l": 3, "vis_r": 4}
CONDITIONS = list(EVENT_ID)
head = load_head_model("eeg")
raw = mne.io.read_raw_fif(root / "MEG/sample/sample_audvis_raw.fif", preload=True, verbose="error")
events = mne.find_events(raw, stim_channel="STI 014", verbose="error")
raw.pick(mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads"))
raw.del_proj()
raw.set_eeg_reference("average", projection=False, verbose="error")
raw.filter(1.0, 40.0, verbose="error")
epoch_args = dict(tmin=-0.2, tmax=0.5, baseline=(-0.2, 0.0), reject=dict(eeg=150e-6), preload=True)
epochs = mne.Epochs(raw, events, EVENT_ID, decim=4, verbose="error", **epoch_args)
epochs_full = mne.Epochs(raw, events, EVENT_ID, decim=1, verbose="error", **epoch_args)
assert list(epochs.ch_names) == list(head.channel_names)
assert np.array_equal(epochs.selection, epochs_full.selection)
condition = np.array([CONDITIONS[code - 1] for code in epochs.events[:, 2]])
print({c: int((condition == c).sum()) for c in CONDITIONS}, "trials")


def as_series(ep):
    """The epochs as a dirconn TimeSeries of the head model's sensors."""
    channels = tuple(ChannelMeta(name, type="eeg") for name in head.channel_names)
    return TimeSeries(ep.get_data(copy=True), ep.info["sfreq"], channels, "epoched", ep.tmin)


sensors, sensors_full = as_series(epochs), as_series(epochs_full)
SFREQ = sensors.sfreq

# %%
# Step 1: which regions can EEG see at all?
# -----------------------------------------
# Every region is estimated from the sensors with the minimum-norm inverse and summarised by its
# first principal component (``"pca_flip"``), as in tutorial 7. ``crosstalk`` (tutorial 7) asks,
# for each of the 68 regions, how much of every other region arrives in its estimated series,
# relative to its own activity. **Rule A**: a region is usable only if no other region is heard
# in its series as loudly as the region itself (every off-diagonal entry below 1 in absolute
# value) and the estimate is stable across random probes. The rule looks at the head model only,
# never at the brain activity.
names = list(head.roi_names)
probe_seeds = (0,) if QUICK else (0, 1, 2)
probes = [
    crosstalk(head, None, n_samples=3_000 if QUICK else 30_000, seed=SEED + s).matrix
    for s in probe_seeds
]
leak = np.mean(probes, axis=0)
off = ~np.eye(len(names), dtype=bool)
spread = np.where(off, np.std(probes, axis=0), 0.0).max(axis=1)
usable = (np.where(off, np.abs(leak), 0.0).max(axis=1) < 1.0) & (spread <= 0.3)
print(f"usable regions: {usable.sum()} of {len(names)}")
for label in ("superiortemporal-lh", "superiortemporal-rh", "pericalcarine-rh"):
    i = names.index(label)
    j = int(np.argmax(np.where(off[i], np.abs(leak[i]), 0.0)))
    status = "usable" if usable[i] else "not usable"
    print(f"  {label:20s} {status:10s} loudest other region: {names[j]} ({leak[i, j]:+.2f})")

# %%
# The regions an auditory or visual study would name first, the superior temporal gyrus and the
# primary visual cortex, are not usable with 59 EEG channels: their estimated series are dominated
# by neighbours.
#
# Step 2: which usable regions respond to the stimuli?
# ----------------------------------------------------
# **Rule B**: a region is selected if it responds in at least one condition. The response is the
# paired t statistic of each sample between 30 and 300 ms against the trial's own baseline; its
# significance comes from a sign-flip permutation test whose maximum is taken over all regions,
# samples and conditions (family-wise error 0.05). Which condition responds, and when, plays no
# part in the choice.
regions = roi_series(sensors, head, names)
times = epochs.times
baseline, window = times <= 0.0, (times >= 0.03 - 1e-9) & (times <= 0.30 + 1e-9)


def t_statistic(mean, sum_sq, n):
    """One-sample t of the mean of n samples whose squares sum to sum_sq."""
    return mean / np.sqrt((sum_sq - n * mean**2) / (n - 1) / n)


def response_statistics(data, n_permutations, seed):
    """Largest |t| per (condition, region) and its family-wise corrected p-value."""
    rng = np.random.default_rng(seed)
    largest, null = [], np.zeros(n_permutations)
    for c in CONDITIONS:
        x = data[condition == c]
        change = x[:, :, window] - x[:, :, baseline].mean(axis=2, keepdims=True)
        n = len(change)
        flat = change.reshape(n, -1)
        sum_sq = (flat**2).sum(axis=0)

        observed = np.abs(t_statistic(flat.mean(axis=0), sum_sq, n))
        largest.append(observed.reshape(change.shape[1:]).max(axis=1))
        flips = rng.choice([-1.0, 1.0], size=(n_permutations, n))
        null = np.maximum(null, np.abs(t_statistic(flips @ flat / n, sum_sq, n)).max(axis=1))
    largest = np.array(largest)
    pvalues = (1 + (null[None, None] >= largest[..., None]).sum(axis=2)) / (1 + n_permutations)
    return largest, pvalues


t_max, p_response = response_statistics(
    np.asarray(regions.data), n_permutations=200 if QUICK else 1000, seed=SEED
)
strength = t_max.max(axis=0)
responds = (p_response < 0.05).any(axis=0)

# %%
# **Rule C**: leakage can make a region *look* responsive. Going from the strongest response
# down, a region is dropped when a stronger selected region is heard in its series with more than
# half of its own weight. At most ten regions are kept.
selected = []
for i in np.argsort(-strength):
    if not (usable[i] and responds[i]):
        continue
    louder = [j for j in selected if abs(leak[i, j]) > 0.5]
    if louder:
        print(
            f"dropped {names[i]:28s} ({names[louder[0]]} in its series: {leak[i, louder[0]]:+.2f})"
        )
    else:
        selected.append(i)
selected = selected[:10]
kept = [names[i] for i in selected]
print("selected:", kept)

# %%
# Step 3: the direction, from the timing of the responses
# -------------------------------------------------------
# **Rule D** proposes a direction where one region responds reliably before another. On the
# full-rate data, the latency of a response is the time at which half of the area under the
# absolute trial average (30 to 300 ms) is reached. For every pair of selected regions that both
# respond in a condition, the difference of latencies is resampled over trials (paired
# bootstrap), and the p-values of all pairs of all conditions are corrected together
# (Benjamini-Hochberg, 0.05).
regions_full = roi_series(sensors_full, head, kept)
full_times = epochs_full.times
full_window = (full_times >= 0.03 - 1e-9) & (full_times <= 0.30 + 1e-9)
window_times = full_times[full_window]


def half_area_latency(mean):
    area = np.cumsum(np.abs(mean), axis=-1)
    return window_times[np.argmax(area >= 0.5 * area[..., -1:], axis=-1)]


rng = np.random.default_rng(SEED)
n_boot = 500 if QUICK else 2000
pairs = []
for k, c in enumerate(CONDITIONS):
    present = [r for r in kept if p_response[k, names.index(r)] < 0.05]
    x = np.asarray(regions_full.data)[condition == c][:, [kept.index(r) for r in present]]
    x = x[:, :, full_window]
    observed = half_area_latency(x.mean(axis=0))
    boot = np.array(
        [half_area_latency(x[rng.integers(0, len(x), len(x))].mean(axis=0)) for _ in range(n_boot)]
    )
    for a, b in combinations(range(len(present)), 2):
        diff = boot[:, a] - boot[:, b]
        p = max(2 * min((diff <= 0).mean(), (diff >= 0).mean()), 1 / n_boot)
        first, second = (a, b) if observed[a] < observed[b] else (b, a)
        pairs.append((c, present[first], present[second], abs(observed[a] - observed[b]), p))
p_pairs = dirconn.adjust_pvalues(np.array([row[-1] for row in pairs]), method="fdr_bh")
print("orders that survive the correction (earlier -> later, difference in ms):")
for (c, early, late, gap, _), q in sorted(zip(pairs, p_pairs, strict=True), key=lambda r: r[1]):
    if q < 0.05:
        print(f"  {c:6s} {early:24s} -> {late:24s} {gap * 1e3:5.1f} ms  q = {q:.3f}")

# %%
# When this tutorial was written, the rules were run once and the hypothesis was frozen **before
# any transfer entropy was computed**. Only one order survived the correction in both auditory
# conditions and kept its sign in both halves of the trials (odd and even): the right primary
# auditory cortex (``transversetemporal-rh``) responds about 50 ms before the left
# ``paracentral`` region. Within one half alone no order survived the correction. Some
# orders were not taken: the superior parietal cortex before the cuneus is the reverse of the
# visual hierarchy, which points to mislocalisation rather than to a pathway, and the insula was
# not selected in either half of the trials. The latency rule was revised once before freezing:
# a first version that compared overlapping confidence intervals at 150 Hz was too coarse to
# order anything. The quick version uses fewer probes, permutations and resamples and can print
# a different list; the hypothesis below does not change with it.
SOURCE, TARGET = "transversetemporal-rh", "paracentral-lh"
print(f"hypothesis: transfer entropy {SOURCE} -> {TARGET} exceeds {TARGET} -> {SOURCE}")
print(
    f"leakage between them: {leak[names.index(TARGET), names.index(SOURCE)]:+.3f} "
    f"(into {TARGET}), {leak[names.index(SOURCE), names.index(TARGET)]:+.3f} (into {SOURCE})"
)

# %%
# What goes into transfer entropy
# -------------------------------
# The test uses the auditory trials of both ears together. The stimulus-locked response is the
# same in every trial; it would make the earlier region "predict" the later one in every trial
# and in every trial-shuffled surrogate alike. So the average response of each condition is
# subtracted from its trials, and transfer entropy is computed on what remains, from 0 to 0.5 s.
# The statistic is the asymmetry ``ΔTE = TE(source -> target) - TE(target -> source)``, with
# delays of 1 to 10 samples (7 to 67 ms) scanned; the measured gap of about 50 ms is inside.
POST = times >= 0.0
# the quick version scans fewer delays and uses fewer trials: it only checks that the code runs
estimator = TE(mode="pairwise", max_delay=3 if QUICK else 10, workers=WORKERS)
N_SURROGATES = 4 if QUICK else 39


def induced(series, labels, trials):
    """Region series of some trials, with each condition's average response subtracted."""
    x = np.asarray(series.pick(labels).data)[trials][:, :, POST].copy()
    for c in np.unique(condition[trials]):
        same = condition[trials] == c
        x[same] -= x[same].mean(axis=0, keepdims=True)
    return TimeSeries(x, SFREQ, series.pick(labels).channels, "epoched", 0.0)


def asymmetry(result):
    return result.values[0, 1, 0, 0] - result.values[1, 0, 0, 0]


def asymmetry_test(data, n_surrogates, seed):
    """ΔTE of channel 0 -> 1 and its trial-shuffled null distribution (one-sided p-value)."""
    observed = estimator.fit(data)
    shuffle, rng = TrialShuffle(), np.random.default_rng(seed)
    null = np.array(
        [asymmetry(estimator.fit(shuffle.generate_one(data, rng))) for _ in range(n_surrogates)]
    )
    p = (1 + np.sum(null >= asymmetry(observed))) / (1 + n_surrogates)
    return observed, null, p


# %%
# Step 4: can this pipeline find the direction at all?
# ----------------------------------------------------
# Before the recording, a simulation with the same head model, the same two regions, the same
# number and length of trials and the same sampling rate. In the first scenario the right
# auditory cortex drives the left paracentral region with a delay of 7 samples (47 ms); in the
# second nothing couples them. Both regions oscillate broadly around 10 Hz, other sources of the
# brain add background activity, and the sensors have a signal-to-noise ratio of 0 dB. The
# simulated sensors go through exactly the steps above: inverse, region summary, subtraction of
# the average, transfer entropy, surrogate test.
n_auditory = int(np.isin(condition, ["aud_l", "aud_r"]).sum())
sim_trials = 48 if QUICK else n_auditory
nodes = [dk_node(SOURCE), dk_node(TARGET)]
dynamics = {name: NodeDynamics(freq_hz=10.0, bandwidth_hz=6.0) for name in (SOURCE, TARGET)}
coupled_edge = Edge(SOURCE, TARGET, strength=0.4, delay_ms=7 * 1000.0 / SFREQ)
scenarios = {
    "coupled": Network(nodes, [coupled_edge], name="coupled"),
    "uncoupled": Network(nodes, [], name="uncoupled"),
}
simulated = {}
for k, (label, network) in enumerate(scenarios.items()):
    sim = simulate_on_head(
        network,
        head,
        sfreq=SFREQ,
        n_times=int(POST.sum()),
        n_trials=sim_trials,
        seed=SEED + 10 + k,
        dynamics=dynamics,
        snr_db=0.0,
        delay_rounding="round",
    )
    estimated = roi_series(sim.data, head, [SOURCE, TARGET])
    x = np.asarray(estimated.data) - np.asarray(estimated.data).mean(axis=0, keepdims=True)
    data = TimeSeries(x, SFREQ, estimated.channels, "epoched", 0.0)
    simulated[label] = asymmetry_test(data, 4 if QUICK else 19, seed=SEED + 20 + k)
    observed, null, p = simulated[label]
    print(
        f"{label:9s} dTE {asymmetry(observed):+.4f} nats, null 95% {np.quantile(null, 0.95):+.4f}, "
        f"p = {p:.3f} ({len(null)} surrogates)"
    )

# %%
# Step 5: the test on the recording
# ---------------------------------
auditory = np.flatnonzero(np.isin(condition, ["aud_l", "aud_r"]))[:: 3 if QUICK else 1]
pair = induced(regions, [SOURCE, TARGET], auditory)
observed, null, p_value = asymmetry_test(pair, N_SURROGATES, seed=SEED + 30)
delay = observed.provenance.details["delay"]
print(
    f"{len(auditory)} auditory trials of {pair.n_times} samples, target history "
    f"{observed.provenance.details['target_history']} samples"
)
print(f"TE {SOURCE} -> {TARGET}: {observed.values[0, 1, 0, 0]:.4f} nats (delay {delay[0][1]})")
print(f"TE {TARGET} -> {SOURCE}: {observed.values[1, 0, 0, 0]:.4f} nats (delay {delay[1][0]})")
print(
    f"dTE {asymmetry(observed):+.4f} nats, one-sided p = {p_value:.3f} "
    f"with {N_SURROGATES} trial-shuffled surrogates (smallest possible {1 / (N_SURROGATES + 1):.3f})"
)

fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), layout="constrained", sharey=False)
for ax, (label, (obs, nul, p)) in zip(
    axes, [*simulated.items(), ("recording", (observed, null, p_value))], strict=True
):
    ax.hist(nul, bins=10, color="0.75", label="trial-shuffled")
    ax.axvline(asymmetry(obs), color="C3", lw=2, label="observed")
    ax.set_title(f"{label}: p = {p:.3f}")
    ax.set_xlabel("ΔTE (nats)")
axes[0].set_ylabel("surrogates")
axes[0].legend(fontsize=8)

# %%
# Each ear on its own (a description, not a second test): if the asymmetry belongs to the
# auditory response, it should have the same sign for both ears.
for c in ("aud_l", "aud_r"):
    one_ear = estimator.fit(induced(regions, [SOURCE, TARGET], np.flatnonzero(condition == c)))
    print(f"{c}: dTE {asymmetry(one_ear):+.4f} nats")

# %%
# In the full version the hypothesis is **not supported**: the asymmetry is negative, below every
# trial-shuffled surrogate, and close to zero for each ear on its own. The simulation found its
# planted direction with about ten times this size, so the pipeline can detect a coupling of that
# kind; the recording shows none in the hypothesised direction. That the observed value lies on
# the other side of the null is *not* a finding about the opposite direction: it was not the
# hypothesis, and testing it now, after seeing it, would be choosing the test by its result. It
# would need its own rule, fixed in advance, and other data.
#
# Exploration: every pair of the selected regions
# -----------------------------------------------
# Pairwise transfer entropy between all selected regions, auditory trials, average response
# subtracted. There is no test here and none of these values is evidence: they were not
# hypothesised, and the raw estimate is biased differently for every pair.
explore_trials = auditory
exploration = estimator.fit(induced(regions, kept, explore_trials))
fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
exploration.plot(ax=ax)
ax.set_title("pairwise TE between the selected regions (exploratory)")

# %%
# What this does and does not show
# --------------------------------
# * The regions and the direction were chosen by rules fixed before any directed measure was
#   computed, and one hypothesis was tested. The trials that chose it are the trials that test it;
#   the choice used the average response, the test uses what remains after subtracting it, so the
#   two are related but not the same.
# * The tested hypothesis was not supported. A responding region that answers 50 ms after
#   another does not, by that alone, receive predictive information from it in the
#   trial-by-trial fluctuations that remain once the average response is removed.
# * The simulation says whether, under its assumptions (two oscillating regions, one coupling of
#   47 ms, 0 dB), this pipeline finds the direction and stays quiet without coupling. It does not
#   say that the recording satisfies those assumptions.
# * 39 surrogates allow p-values down to 0.025: enough for a one-sided test at 0.05, not for a
#   precise p-value. One subject cannot say anything about people in general.
# * With 59 EEG channels the superior temporal gyrus, the primary visual cortex and lateral
#   occipital cortex were not usable, so the classic pathways could not be studied. Rule C
#   depends on the order of the regions: on the full data it dropped the left auditory cortex
#   in favour of the left parahippocampal region, whose own response is most likely leakage, so
#   the exchange between the two auditory cortices could not be tested.
# * dirconn does not correct for leakage, and its inverse uses an identity noise covariance
#   scaled to the data rather than one estimated from the baseline.
# * A latency gap of 50 ms contains processing as well as conduction; transfer entropy measures
#   predictive information from the past of one series to another, not the path a signal takes.
#   Each result lists the assumptions it rests on; ``not_checked`` means that the library did not
#   test one, not that it holds:
for check in observed.provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")

# %%
# * Not shown here: other inverse methods, MEG, conditional transfer entropy, more subjects.
plt.close("all")

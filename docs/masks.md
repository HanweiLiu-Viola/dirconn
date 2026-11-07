# Gaps, selected samples and event locks

Real recordings have bad segments and synchronisation gaps, and an analysis often asks about a
state that covers only part of the record, such as beta bursts of 100 to 300 ms. Cutting such data
into equal-length tiles and passing them as trials works, but it wastes data and the tile length is
an arbitrary choice. A {class}`~dirconn.TimeSeries` can instead carry two per-sample masks of shape
`(n_trials, n_times)`:

- `valid`: the sample can be used at all, as a regressor or as a predicted value. Gaps and bad
  segments are `False`. The samples themselves stay finite, so code that does not know about masks
  never meets a NaN: `TimeSeries.from_array(x, sfreq, nan="mask")` (and `read_bids(..., nan="mask")`)
  replaces NaN by 0 and marks the time point invalid.
- `predict`: the sample contributes an equation (a regression row) to a model fit. "Burst versus
  no burst" is a change of this mask, not a different data set.

A VAR model of order `p` uses the row that predicts sample `t` only when `predict[t]` is true and
`valid[t - p .. t]` are all true. The order selection (information criteria and cross-validation)
uses the same rule, the mean that `demean` removes is the mean of the valid samples, and the number
of rows that were used is what the provenance (`details["n_obs"]`) and the sampling check report.
Masks that are all true are dropped, so they reproduce the unmasked result bit for bit; a tile of
`order + k` samples passed as a trial gives exactly the same coefficients as a `predict` mask that
selects its last `k` samples in the continuous record.

```python
import numpy as np
from dirconn import GC, TimeSeries

x = np.load("record.npy")  # (n_channels, n_times), NaN in the gaps
burst = np.load("burst_mask.npy")  # (n_times,) True inside bursts
data = TimeSeries.from_array(x, 1000.0, nan="mask", predict=burst)
in_bursts = GC(order=20).fit(data)
outside = GC(order=20).fit(TimeSeries.from_array(x, 1000.0, nan="mask", predict=~burst))
```

The two fits use different numbers of rows, and Granger causality is biased upwards by about
`p / N` for `N` rows; compare them only after matching or correcting for the sample size.

## Which methods accept masks

The static VAR estimators ({class}`~dirconn.GC`, {class}`~dirconn.DTF`, {class}`~dirconn.PDC`,
{class}`~dirconn.GPDC`, {class}`~dirconn.RPDC`) and the asymptotic tests built on them use the masks
(`MethodInfo.supports_mask`). Every other estimator (multitaper GC, transfer entropy, PCMCI,
coherency, PSI, time-reversed GC and the time-varying methods) raises on masked data instead of
silently using every sample. Among the surrogates only `TrialShuffle` carries the masks, with the
trials: a surrogate trial takes each channel from another trial, so a time point is valid (and
predicted) only where it is in every trial that contributes to it. The generators that move samples
within a trial raise. The bootstrap and the permutation test of {func}`~dirconn.contrast_test`
resample whole trials (or blocks) together with their masks.

## Event locks and matched controls

{meth}`TimeSeries.lock <dirconn.TimeSeries.lock>` cuts epochs around event times of a continuous
record, for example burst onsets, and drops every lock that would cross the edge of the record or
contain an invalid sample. {meth}`TimeSeries.control_locks <dirconn.TimeSeries.control_locks>` draws
the same number of locks from time points the user marks as eligible (for example a low envelope),
with a fixed seed:

```python
locked = data.lock(onsets, tmin=-0.75, tmax=0.75)
controls = data.control_locks(quiet, locked.n_trials, tmin=-0.75, tmax=0.75, seed=0)
```

Both record what they did (the kept and dropped events, or the seed and the drawn times) in
`TimeSeries.history`, and every estimator copies that history into the provenance of its result as
`preprocessing["data_history"]`.

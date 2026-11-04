# Time-varying methods

dirconn estimates directed connectivity that changes with time: `ADTF` and `TPDC` (a Kalman,
RLS or DEKF recursion for the coefficients of a VAR model), `WDTF` (one fit per window or segment),
the tools to reduce and compare the result over segments, and edges that switch on and off in
the simulator. What the methods can and cannot follow is measured in
[nonstationary_validation.md](nonstationary_validation.md); this page says what they compute, what
the settings mean and what was changed from the old package.

## What there is

| | |
|---|---|
| `ADTF`, `adtf` | DTF at every time point from a time-varying VAR |
| `TPDC`, `tpdc` | PDC at every time point from the same model |
| `WDTF`, `wdtf` | DTF of one pooled VAR per sliding window (`window`, `step`) or per named segment (`segments`) |
| `dirconn.tvmvar.fit_tvar`, `fit_tvar_grid` | the time-varying VAR itself; the grid version runs several forgetting factors in one pass |
| `Connectivity.reduce_time(segments, how)` | reduce the time axis over segments, like `reduce_freq` |
| `contrast_test(..., paired=True)` | permutation test for two segments of the same trials |
| `Edge(..., active=[(start, stop), ...])` | an edge that acts only in time intervals; `network.truth(times=...)`, `simulate` |
| `dirconn.evaluation` (time-resolved part) | `auroc_by_time`, `score_stable`, `tracking_latency`, `switch_times`, ... |

All of them share the layout `values[source, target, freq, time]`, the direction convention
`source -> target`, and the provenance and assumption checklist of the rest of the package.

## The recursion

For every target channel `i` one filter estimates the coefficients `a_i(t)` of `x_i[t] = sum_k
A_k(t)[i, :] x[t-k] + e`. The regressor is shared, so the observation matrix has Kronecker structure
and the `N` equations decouple: the state of each filter has `order * n_nodes` entries, the update is
a scalar-observation update in Joseph form (no matrix inverse), and the work is vectorised over the
targets, the trials and the forgetting factors. Only the loop over time is a Python loop.

* `update="dual"` also tracks the full innovation covariance, `update="rls"` only its diagonal.
  **Their coefficients are identical to the last bit** (the coefficient filter reads only the
  diagonal), so ADTF and TPDC give the same result for either; only `TvarFit.noise_cov` differs.
* `update="dekf"` is the dual extended Kalman filter of Wan and Nelson (2001): the observation is
  `y = x + n` with measurement noise `n`, and a companion-form state filter estimates the denoised
  signal that the coefficient filter regresses on. With `measurement_noise=0` it equals `"dual"`.
  The default measurement noise is 10 % of each channel's variance, a conservative guess; pass the
  known value if the SNR is known.

## The forgetting factor is not a memory

`forgetting` is the factor `lambda` of the old package. In a forgetting-factor least-squares filter it
would make the filter remember `1 / (1 - lambda)` samples. In this recursion it **only sets the process
noise `1 / lambda - 1`** (and the speed of the innovation-variance recursion), and the filter's real
memory is much shorter. `TvarFit.memory_samples` measures it: the reciprocal of the mean gain along the
regressor over the samples after the burn-in. It is recorded in the provenance (`memory_samples`) and
used for the sampling check and for the default guard band of `dirconn.evaluation.guard_seconds`.

Measured values:

| `forgetting` | 0.95 | 0.98 | 0.99 | 0.995 | 0.999 | 0.9999 | 0.99999 |
|---|---|---|---|---|---|---|---|
| memory, two-node toy VAR (samples) | 3 | 4 | 6 | 8 | 15 | 43 | 126 |
| `1 / (1 - forgetting)` (samples) | 20 | 50 | 100 | 200 | 1000 | 10 000 | 100 000 |

With `forgetting=0.99` (a common default elsewhere, not dirconn's) the estimate follows the data almost sample by sample;
with 12 or more coefficients per equation that is noisy, and 30 to 50 % of the time points can have an
unstable model (spectral radius 1 or more). The checklist says so (`sampling_adequacy`, `stationarity`).
**The default is `forgetting=0.99999`** (decided 2026-10-04 from the validation report: a measured memory of
0.2 s at 100 Hz with three nodes and order 4, stable AUROC 0.96 to 1.0 in the four scenarios, against
0.61 to 0.75 for 0.99). A factor is not portable across sampling rates, node counts and orders, so check
`memory_samples` in the provenance for your data; set a larger factor (or `process_noise` directly) when the
coupling changes slowly, a smaller one when it changes fast.

## Defaults, and what changed from the old package

| | old package (`neurrow`) | `dirconn` |
|---|---|---|
| order | 1 | BIC on a **stationary** pooled fit of the whole recording, at most 5; or `order=` |
| seed fit | statsmodels on the first samples; **zero start with a warning if it fails** | own OLS core on the first `init_samples` of every trial, pooled; a failing fit **raises**; `init="zeros"` must be asked for and is recorded |
| standardising | per trial | one mean and one standard deviation per channel over all trials |
| several trials | filtered one by one, the **measures** averaged | filtered one by one from one seed, the **coefficients** averaged at every time point, the measure computed once; assumes the non-stationarity is time-locked to the trial start |
| failures inside the recursion | skipped update or reset to the raw observations, silently | counted (`n_skipped_updates`, `n_state_resets`) and reported in the checklist (`filter_health`) |
| time axis | sample index | seconds; `time_step` in seconds (default about 200 points); burn-in not output |
| TPDC default | amplitude form | squared form, as `PDC`; `squared=False` gives the amplitude form |
| `rls` | "faster, one fewer knob" | the same coefficients as `dual` |

The numerical core agrees with the old one to 1e-9 from a common zero start (five fixtures in
`tests/data`, made by `scripts/make_tvar_fixtures.py`) and with a slow joint reference implementation of
the DEKF to 1e-8.

Defaults that were kept: process noise `1 / lambda - 1`, `init_samples =
min(T/4, 20 * order * n_nodes)`, `burn_in = max(init_samples, 10 * order)`, `standardize="zscore"`. The one that changed is `forgetting`: 0.99999 instead of 0.99.

## One value per segment

Two procedures give one result for a stretch of a recording, and they are **not the same**:

1. **Continuous, then average**: `ADTF(...).fit(data).reduce_time([(a, b), ...])`. The estimate is carried
   through the whole recording, so a segment starts with the state of the preceding samples and, right
   after a switch, still contains the transition. It uses all the data to estimate the coefficients.
2. **A fit per segment**: `WDTF(segments=[(a, b), ...]).fit(data)`. Each segment is fitted alone, so
   nothing is carried over, but it has only its own samples (more variance for a short segment) and
   uses the order chosen on the whole recording.

Both results have the same shape and time axis (the segment centres), so they can be compared directly.
The validation report compares them on the "edge switches on" and "direction flips" scenarios.

For a **test** of a difference between two segments of the same trials use
`contrast_test(estimator, segment_a, segment_b, paired=True, ...)`. It swaps the two segments of
every trial at random and refits the estimator, which is only possible for estimators that are fitted
on a segment alone (GC, DTF, PDC, WDTF); an unpaired test on such data ignores what the two segments of
a trial share and its null is too wide. The test cannot tell a change of coupling from a time-order
effect inside the trial (adaptation, fatigue, drift). **A segment mean of a continuous ADTF has no
test**: it is descriptive only.

```python
import dirconn as dc

# epoched data; the first 0.5 s of every trial against the next 0.5 s
n = int(0.5 * data.sfreq)
make = lambda x: dc.TimeSeries(x, data.sfreq, data.channels, "epoched", data.tmin)
early, late = make(data.data[:, :, :n]), make(data.data[:, :, n : 2 * n])
result = dc.contrast_test(
    dc.DTF(order=2), early, late, n_permutations=499, seed=0, paired=True, reduce=(8.0, 12.0)
)
print(result.null, result.pvalues)
```

## Edges that switch

`Edge(source, target, strength, active=[(start, stop), ...])` acts only in the given time intervals in
seconds from the start of the recording. An interval contains `start` and not `stop` (half open), they
are ascending and may follow each other, and the switching is hard. The same pair and band may appear
again only with intervals that do not overlap (a coupling that changes strength). The simulator runs a
piecewise-constant VAR: `network_var_schedule` splits the recording into stretches with a constant set
of edges (the stability check runs on every stretch), and the band-pass filters of banded edges run
through the whole recording so switching does not restart them. `Network.truth(times=...)` returns the
truth per time point; without `times` an edge with intervals counts as present. The four shipped
scenarios `switch_on`, `switch_off`, `direction_flip` and `burst` use it.

## Limits

* The order comes from a stationary fit and may be wrong for a switching model; the checklist says so.
* Several trials are averaged coefficient by coefficient: a change that is not time-locked to the trial
  start is averaged away.
* ADTF and TPDC are predictive measures of a linear model, like their static counterparts; a switch of
  coupling and a switch of signal power both move them.
* Everything about speed and accuracy in the validation report was measured on three-region
  region-level simulations at 100 Hz and is not a general statement.

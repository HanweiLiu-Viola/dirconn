# Surrogate data and what a surrogate test can tell you

A surrogate test compares the connectivity of the data with the connectivity of copies of the
data in which the coupling between channels has been destroyed but some other property has been
kept. The **null hypothesis is therefore not "there is no connectivity"**. It is "the connectivity
is no larger than what the kept property alone produces". Rejecting it says the data differ from
the surrogates in a way the kept property does not explain; it does not say the cause is a
directed influence between the regions. Every test result carries the null as text (`null`), so a
number never travels without its premise.

## What each surrogate keeps and destroys

| Surrogate | Keeps exactly | Keeps approximately | Destroys |
|---|---|---|---|
| `phase_randomize` | amplitude spectrum, autocorrelation, mean of each channel and trial | nothing else | phase relations between and within channels |
| `aaft` | amplitude distribution of each channel and trial | power spectrum (relative error about 0.2 in the test data) | phase relations between and within channels |
| `iaaft` | amplitude distribution of each channel and trial | power spectrum (relative error about 0.03 in the test data) | phase relations between and within channels |
| `circular_shift` | every series up to a rotation: spectrum, distribution, autocorrelation | nothing else | alignment between channels (and the trial-locked timing) |
| `trial_shuffle` | every trial of every channel | nothing else | trial-by-trial pairing between channels |
| `block_shuffle` | every block of every channel (blocks of `block_size` samples) | nothing else | alignment of the blocks between channels |

## Null hypotheses and their limitations

The table is generated from the code (`dirconn.surrogates.limitations_table()`); a test keeps it
identical to the classes' `null` and `limitations`.

| Surrogate | Null hypothesis | How the null can be rejected without coupling |
|---|---|---|
| `phase_randomize` | connectivity is explained by individual-channel power spectra | 1. Assumes each channel is a stationary linear Gaussian process; amplitude fluctuations over time and non-Gaussian amplitude distributions are not kept.<br>2. Destroys phase relations within a channel (bursts, waveform shape, cross-frequency coupling), so a channel with such structure can look significant for that reason alone.<br>3. A trial-locked evoked response is also scrambled; for epoched data this can inflate significance when the coupling is driven by the evoked response. |
| `aaft` | connectivity is explained by spectra and marginal amplitude distributions | 1. The spectrum is matched only approximately and the error is systematically biased towards a flatter spectrum; IAAFT matches it better.<br>2. Assumes the data are a monotone static transformation of a linear Gaussian process.<br>3. Destroys phase relations within a channel and trial-locked evoked responses, like phase randomization. |
| `iaaft` | connectivity is explained by spectra and marginal amplitude distributions | 1. Matches the spectrum only up to the iteration's residual error, and the residual is not zero even at convergence.<br>2. The iteration can stop at a rank-order fixed point or reach n_iter without converging; the second case is reported as a warning.<br>3. Same stationarity and phase-structure assumptions as AAFT; evoked responses are scrambled. |
| `circular_shift` | between-channel coupling is explained by within-channel autocorrelation | 1. The seam where the end of a trial is joined to its start is a discontinuity that affects model-based estimators such as MVAR fits; long trials make it negligible.<br>2. Assumes stationarity within a trial: a slow trend or a change of variance is moved to another place in the trial, which changes the cross-channel statistics.<br>3. A trial-locked evoked response is shifted away from its latency in every channel, so coupling that is driven by a shared evoked response is destroyed with it. |
| `trial_shuffle` | there is no stable trial-to-trial coupling | 1. Coupling that comes from a shared evoked response is kept in every surrogate, so the test cannot detect it; it asks only about trial-specific (induced) coupling.<br>2. With few trials the number of distinct surrogates is small (n_trials! per channel beyond the first), which limits the smallest attainable p-value, and a random permutation may leave a channel's order unchanged.<br>3. Needs trials that are exchangeable: a drift or learning effect across trials makes the shuffled data differ from the original for reasons unrelated to coupling. |
| `block_shuffle` | between-channel coupling is explained by within-block structure (blocks are exchangeable within a channel) | 1. The observed result is computed on the blocks too (as separate trials), so a model loses the regression rows across block boundaries; with blocks short relative to the model order this costs data and changes the estimate.<br>2. Assumes stationarity and exchangeable blocks: a drift, a change of variance or a block-locked event makes shuffled blocks differ from the original for reasons unrelated to coupling.<br>3. Dependence longer than a block is destroyed in the surrogates, so the null also includes 'no coupling on time scales longer than block_size'; too short a block inflates significance.<br>4. A trial-locked evoked response is moved to other blocks in some channels and not in others, so coupling driven by it is destroyed with it. |

## Running a test

```python
from dirconn import GC, surrogate_test
from dirconn.surrogates import IAAFT

result = surrogate_test(GC(order=2), data, IAAFT(), n_surrogates=199, seed=0, reduce=None)
result.pvalues  # (source, target, freq, time); NaN for undefined cells such as the diagonal
result.null  # the premise of the p-values
result.limitations  # how that null can be rejected without coupling
```

The estimator's whole procedure, including a model order chosen by a criterion, is repeated on every
surrogate. Diagnostics raised inside the surrogate fits are not repeated 199 times: one summary warning
says how many surrogates raised one, and `result.checks` records it. A surrogate fit that leaves a cell
undefined that is defined on the data raises an error instead of producing a p-value.
`n_surrogates` is checked against the number of **defined** cells of the tested (reduced) result: a
Bonferroni-type correction over `m` cells needs `n_surrogates >= m / alpha - 1`.

## Choosing one

- For continuous data (one trial) `phase_randomize`, `aaft`, `iaaft`, `circular_shift` and
  `block_shuffle` apply; `trial_shuffle` needs at least two trials and raises otherwise.
- `block_shuffle` is the surrogate to use when a model-based estimator (GC, DTF, PDC) runs on a
  continuous record and circular shifting would put a seam in the middle of the regressors. The blocks
  become separate trials, so no regression crosses a block boundary, and the **observed result is
  computed on the same blocks** (the generator's `prepare()` step, called by `surrogate_test`); the
  estimate therefore differs a little from a fit on the unbroken record. It is what `test="block_perm"`
  of `dirconn.significance_test` uses.
- If the amplitude distribution is clearly non-Gaussian (bursty, skewed, clipped), prefer `iaaft`
  over `phase_randomize`. A Gaussian surrogate of such data differs from it in more than coupling.
- If an evoked response dominates the epochs, `phase_randomize`, `aaft`, `iaaft` and
  `circular_shift` all remove it from the surrogates. Their null then includes "no evoked response",
  and the test can reject for that reason alone. `trial_shuffle` keeps the evoked response in every
  channel and is the better choice for asking whether coupling goes beyond it.
- The p-value of any of these tests is bounded below by `1 / (n_surrogates + 1)`; with `trial_shuffle`
  and few trials the number of distinct surrogates may be smaller than `n_surrogates`.

## Reproducibility

Each surrogate uses its own generator, derived from one seed by `dirconn.rng.RNGController`:
generator `i` depends only on the seed and `i`, not on the number of workers, and raising the number
of surrogates leaves the earlier ones unchanged. The seed is required.

## Known differences from the earlier `neurrow` implementation

- `aaft`: the earlier code arranged the *data values* in the order of a Gaussian draw, which
  destroys the autocorrelation before the phases are randomized. Here the sorted Gaussian values are
  arranged in the rank order of the *data* (the standard construction). Measured on skewed AR(1) test
  data, the relative amplitude-spectrum error is 0.21 to 0.25 here against 0.36 to 0.43 before.
- `phase_randomize`: the mean (and the Nyquist term for even lengths) is kept as it is; taking its
  magnitude would flip the sign of a negative mean.
- `circular_shift`: shifts lie in `[min_shift, n_times - min_shift]`, so a shift is at least
  `min_shift` samples away from zero in both directions (the earlier range allowed `n_times - 1`,
  which is a shift of one sample backwards).
- `iaaft`: stops when the rank order no longer changes and warns when `n_iter` is reached first.

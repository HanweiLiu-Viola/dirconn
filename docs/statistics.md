# Statistics in dirconn

Every test returns a result that states **what it tests** (`result.null`) and **how that can
mislead** (`result.limitations`). A p-value that travels without its null is not interpretable, so
the two travel together. The surrogate limitations are tabulated in [surrogates.md](surrogates.md).

| Function | Question | Null hypothesis (short) |
|---|---|---|
| `significance_test(test="surrogate")` | Is this cell larger than surrogates of the data allow? | set by the surrogate (see surrogates.md) |
| `significance_test(test="block_perm")` | Same, for a continuous record, by shuffling blocks | coupling explained by within-block structure |
| `significance_test(test="analytic")` | GC: does the source's past improve the prediction of the target? PDC, rPDC: is the direct influence at this frequency non-zero? | GC: the source's past adds nothing; PDC, rPDC: `Abar[target, source](f) = 0` in the fitted VAR |
| `contrast_test` | Does connectivity differ between two conditions? | the same in both conditions |
| `bootstrap` | How variable is the estimate? | none: confidence bands, not a test |
| `baseline_normalise` | Change relative to a baseline | none: a transformation |

There is no default test. Choosing one is choosing a null hypothesis.

## Common rules

- **Seeds are required.** Resample `i` uses `RNGController(seed).child(i)`, so a result does not
  depend on how many workers share the work, and asking for more resamples extends the earlier ones.
- **Undefined cells (NaN, such as the diagonal) have no p-value** and are excluded from every
  correction family and from the power check. The family is what you choose with `reduce=`.
- **Failures raise, diagnostics warn.** A resampled fit that leaves a cell undefined that is defined
  on the data is an error; a diagnostic raised inside the resampled fits is summarised in one warning
  and recorded in `result.checks`.
- **Whole pipeline repeated.** The estimator's full procedure, including a model order chosen by a
  criterion, is repeated on every surrogate, resample or relabelling.
- **Power check.** A rank p-value cannot fall below `1 / (n + 1)`; a Bonferroni-type correction over
  `m` defined cells needs `n >= m / alpha - 1`, otherwise a warning says that nothing can be rejected.
- **Arguments that a test does not use are refused**, not ignored.

## Correction

`result.adjusted(method)` and `result.significant(alpha=None, method="fdr_bh")` with `"fdr_bh"`
(Benjamini-Hochberg), `"bonferroni"` or `"none"` (explicit). `dirconn.adjust_pvalues` works on any
array. Benjamini-Hochberg is valid for independent or positively dependent tests; connectivity cells
that share a channel are dependent, so treat FDR control as approximate.

## Condition contrast and baseline

`contrast_test(estimator, data_a, data_b, n_permutations=, seed=)` relabels the pooled trials (group
sizes kept) and tests the difference of magnitudes per cell, two-sided. Its limitations include that
a difference in signal-to-noise ratio between conditions changes connectivity estimates without any
change in coupling.

`baseline_normalise(estimator, active, baseline, how="difference" | "ratio" | "zscore")` is a
separate step and returns an ordinary `Connectivity` of magnitudes. The z-score uses the bootstrap
spread of the baseline. To compare normalised conditions, normalise each condition to its own
baseline and compare the results yourself, or contrast the raw conditions; the contrast does not
embed a normalisation.

## Bootstrap

`bootstrap` gives percentile bands (no bias correction) by trial for epoched data and by block for
continuous data (blocks stored as separate trials), with a warning below 20 units. Percentile bands
are slightly too narrow in small samples and a biased estimator (Granger causality in short data)
is not centred on the truth.

## Calibration under a true null

Measured with the scripts and tests named in the right column; all p-values should be uniform.

| Test | Setting | Result | Where |
|---|---|---|---|
| surrogate, `phase_randomize` / `circular_shift` | GC order 2, independent AR(1), 100 datasets, 39 surrogates, 200 p-values | rejections at 0.05: 5 / 7 (10 expected); mean p 0.497 / 0.495 | `scripts/calibrate_surrogate_test.py` |
| surrogate, `block_shuffle` | GC order 2, 600 samples in blocks of 50, 19 surrogates, 300 p-values | share at or below 0.05: 0.053; mean p 0.526 (0.525 expected for 19 surrogates) | `tests/test_surrogate_test.py` |
| contrast | mean differences, 10 trials each, 19 permutations, 1500 datasets, 3 cells | share at or below 0.10: 0.107 / 0.092 / 0.095 | `tests/test_contrast.py` |
| analytic GC | independent noise, 1500 datasets | rejections at 0.05: 4.4% (1 trial of 150) and 5.5% (5 trials of 60); KS p 0.21 and 0.92 | `tests/test_analytic.py` |
| bootstrap | 30 trials, nominal 90% band, 300 datasets | covered 86.2% | `tests/test_bootstrap.py` |

With 19 resamples the p-values are 1/20, ..., 1 and average 0.525 under the null, not 0.5.

## Asymptotic PDC and rPDC tests

`analytic_pdc_test` (Schelter et al. 2006) and `analytic_rpdc_test` (Schelter et al. 2009) test every
pair and frequency from one least-squares fit; the formulas, in the papers' indices and with the
transposition to `[source, target]`, are in their docstrings. Each frequency is tested on its own:
correct over the pairs and frequencies you look at (`result.adjusted()`), and give the estimator a
band's `freqs` to test only that band.

They were calibrated on simulated region data (protocol `asymptotic_calibration`; numbers in the add-on section of
[benchmark_results.md](benchmark_results.md)): the chain of the benchmark without and with its
edges, observation noise at 10 and 40 dB, 30 s and 120 s, 25 datasets per cell, next to 39
phase-randomised surrogates, p-values on a 1 Hz grid from 0 to 50 Hz.

- **Level.** Under a true null (no edge, so no direct influence in any VAR of the data) neither test
  is liberal at the order of 0.2 s, in either noise condition or duration, over the whole grid or in
  the alpha and beta bands. Pointwise rates at 0.05: PDC 0.024 to 0.027 (conservative, as expected
  from its one-degree approximation), rPDC 0.046 to 0.055; the surrogate test 0.044 to 0.054.
- **Recommendation.** With an order long enough for the data and residuals that pass the whiteness
  check, the asymptotic tests can replace a surrogate test for recordings of 30 s and longer: they
  cost one fit instead of one per surrogate, and their p-values are not limited to steps of
  `1 / (n_surrogates + 1)`. Shorter recordings were not checked.
- **Limitation: observation noise.** Noise added to coupled signals (sensor noise, or the mixture of
  other sources in a region's time course) makes the recorded signals depart from a finite VAR, and
  the best VAR of the recorded signals has small *real* direct terms where the network has no
  direct edge. Every test of the model finds them once the recording is long enough. In the chain
  at 10 dB the share of absent direct edges called significant at 0.05 was, at the 0.2 s order,
  0.041 (30 s) and 0.14 (120 s) for asymptotic PDC, 0.077 and 0.21 for asymptotic rPDC, and 0.055
  to 0.15 for the surrogate test; with orders chosen by BIC (3 to 5) 0.23 to 0.47 for the asymptotic
  tests and 0.09 to 0.42 for the surrogate test. After a BH
  correction within the alpha or beta band, 0.17 to 0.46 of the absent pairs at 120 s had a
  significant frequency with the asymptotic tests and at most 0.09 with 39 surrogates (whose
  smallest p-value needs several frequencies to pass BH). The whiteness check does not catch this:
  the residuals of the 0.2 s fits are white. At 40 dB the rates were 0.017 to 0.057.
- **Too low an order** (two samples here) gave 0.85 to 1.00 false positives in the chain at 10 dB
  with every test; the whiteness check warns on such fits.
- **Power.** True edges were found at 73 to 100 % of the frequencies (pointwise, 0.05), and after a
  BH correction within a band every true edge had a significant frequency, with both kinds of test.

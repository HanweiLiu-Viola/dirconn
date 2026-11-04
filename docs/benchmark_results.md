# Benchmark conclusions (written by hand)

Written after the final run of the ten published protocols (code `0de0d7f`, every task finished, none
failed). Every sentence below was compared with the generated tables that follow it. This is a list of
what was seen, in which cell, with how many simulations; it is not a ranking of methods, and a
statement that no table supports is not in it. A "why" that no control run could refute is under
*Not established*.

How to read the numbers. AUROC is the mean over the simulations of a cell (1.0: every true edge scores
above every false edge; 0.5: no information). *n* is the number of simulations behind a cell: ten
seeds in `full`, `order` and `leakage_adjacent`; twenty (ten seeds, two data lengths pooled) in the
tables of `coupling`, `extent` and `size`; four in `inverse`; nine (three noise windows) in `noise` and
`noise_hard`; 25 datasets in `significance` and 40 in `significance_power`; 45 region pairs with one seed each in `leakage_pairs`. A difference between
two settings is taken simulation by simulation (mean, sd of the differences, number of pairs). The cell
means in the AUROC tables carry no interval; a gap between two estimators in one cell is not a result.

## Observed

**The three stages (`full`: EEG, five scenarios, 0, -5 and -10 dB, 30 and 60 s, ten seeds, n = 300
per estimator setting).**

1. Pooled over all of it, the mean AUROC is 0.864 to 0.891 at the region level, 0.800 to 0.825 at the
   source level and 0.608 to 0.652 at the sensor level, over the ten estimator settings. Sensor minus
   region is -0.229 to -0.280 (sd 0.19 to 0.26) and source minus region is -0.051 to -0.088 (sd 0.20 to
   0.24), all with 300 pairs. The pool mixes saturated and unsaturated cells, so it is not the size of
   the loss in any one condition.
2. The order of the stages is not the same in every cell. Chain, coupling 0.1, -10 dB (`coupling`,
   five BIC settings, n = 20): region 0.494 to 0.500, source 0.644 to 0.700, sensor 0.744 to 0.781.
3. `feedback_loop` at the sensor level: 0.415 to 0.556 in all 60 cells (six conditions, ten settings,
   ten seeds each). At the region level the same scenario spans 0.489 to 0.978 and at the source level
   0.604 to 0.852.
4. `common_driver` at the sensor level: conditional Granger causality is 0.305 to 0.495 in all twelve
   cells (six conditions, BIC and fixed order), that is below 0.5; pairwise Granger causality is 0.360 to
   0.700. Conditional minus pairwise is -0.179 (sd 0.123, n = 120) there, +0.032 (sd 0.075) in the chain
   and +0.084 (sd 0.052) in `random_5` at the sensor level; at the region level it is between -0.011
   and +0.005 in all five scenarios.
5. `random_5` at the source level: DTF/BIC is 0.352 to 0.447 and PDC/BIC 0.386 to 0.458 in all six
   conditions (n = 10 each), that is below 0.5.

**BIC order against a fixed order of 0.2 s.**

6. In `full` (100 pairs per method, stage and SNR, scenarios and lengths pooled), fixed minus BIC at the
   region level is +0.062 to +0.080 at -10 dB (sd 0.21 to 0.22), +0.016 to +0.021 at -5 dB and -0.016
   to -0.023 at 0 dB. At the sensor and source levels it lies between -0.021 and +0.033 in every
   method, stage and SNR.
7. In `order` (narrow-band coupling and long delays, 40 pairs per scenario, method and stage), fixed
   minus BIC lies between -0.014 and +0.055; the largest is DTF at the sensor level in `band_alpha`
   (+0.055, sd 0.101). Most cells of this protocol are at 1.000 with both settings.

**Coupling, extent and size (BIC settings, n = 20 per cell).**

8. Chain, region level, -10 dB: 0.494 to 0.500 at coupling 0.1, 0.600 to 0.613 at 0.2, 0.894 to 0.912
   at 0.4. At -5 dB: 0.838 to 0.856, 1.000, 0.994 to 1.000.
9. Chain, sensor level, source models `focal`, `partial` and `patch` (all three SNRs): 0.519 to 0.725,
   0.912 to 1.000 and 0.906 to 1.000. In the collider the sensor level is 0.470 to 0.682 in all three.
   The region-level cells of the same rows also differ (chain, -10 dB, DTF/BIC: 0.731 `focal`, 0.900
   `partial`, 0.806 `patch`), by up to 0.17. The source model is part of the task and so of its seed,
   which makes the rows different random draws; whether it also changes the region-level data was not
   checked, so a gap of this size between rows is not by itself an effect of the source model.
10. `random_10` against `random_5` has no direction that holds for every stage and SNR. Region level,
    -10 dB: 0.656 to 0.664 against 0.549 to 0.553. Region level, 0 dB: 0.900 to 0.950 against 0.955 to
    0.981. Sensor level, 0 dB: 0.581 to 0.662 against 0.543 to 0.723. Source level, 0 dB: 0.572 to 0.635
    against 0.374 to 0.521.

**Inverse methods (`inverse`: conditional GC and gPDC at BIC order, 120 s, SNR 10 and 0 dB, EEG and
gradiometers, four seeds per cell).**

11. On the chain, eLORETA and the minimum norm give 1.000 in all eight cells (two modalities, two SNRs,
    two estimators); dSPM gives 0.875 to 1.000 and LCMV 0.344 to 0.812.
12. LCMV has the lowest value of the four methods in all 16 rows (eight rows, two estimators). In
    `random_5` the other three differ from each other by at most 0.088 in a row, and lie between 0.529
    and 0.815.

**Leakage.**

13. Neighbour (`leakage_adjacent`: Z -> X is the only edge, Y is X's neighbour, 10 dB, ten seeds). With
    EEG at the source level the AUROC is 0.400 (DTF, gPDC, PDC) or 0.800 (conditional and pairwise GC),
    against 1.000 at the region level for all five. The score of the false edge Z -> Y relative to the
    true edge is 1.40 to 2.07 on average there (no seed below 1.36). With gradiometers at the source
    level the AUROC is 0.860 to 1.000 and the relative score of false edges 1.03 or lower on average.
    The direction accuracy is 1.000 in all 30 rows.
14. Region pairs without an edge (`leakage_pairs`, 45 pairs, 90 cells, conditional GC and gPDC at BIC
    order). Spearman correlation between the score of a false edge and the leak of the source region
    into the target series: 0.04 to 0.11 at the region level (intervals from -0.19 to 0.30), 0.44 to
    0.52 at the sensor level (0.23 to 0.66), 0.62 to 0.68 at the source level (0.46 to 0.81).

**Significance (`significance`: region level, chain, conditional GC at fixed order, 25 datasets, 30 s).**

15. Cells without an edge, rate at the 0.05 level. Analytic test: 0.053 (coupling 0, 10 dB, 150 cells),
    0.047 (coupling 0, 40 dB), 0.150 with interval 0.100 to 0.200 (coupling 0.4, 10 dB, 100 cells) and
    0.030 (coupling 0.4, 40 dB). Phase-randomised surrogates (39): 0.053, 0.053, 0.060 and 0.010. In the
    row with the 0.150 the share of datasets with a false rejection after Benjamini-Hochberg is 0.400
    (interval 0.234 to 0.593).
16. Power at 0.05 is 1.000 for both tests in both coupling-0.4 rows. The surrogate test with Benjamini-
    Hochberg has power 0.000 there: with 39 surrogates the smallest p-value is 1/40 and three cells are
    needed at that value, the chain has two true edges.

**Noise (`noise`: chain, -10 dB, 30 s, nine seeds = three source draws on each of three noise windows).**

17. The region level is identical in the three arms (0.819 gradiometers, 0.806 magnetometers); the
    gradiometer sensor and source levels and the magnetometer source level are 1.000 in all three.
    Magnetometer sensor level, conditional GC / gPDC: 0.861 / 0.875 with the simulated noise,
    0.708 / 0.708 with white noise, 0.514 / 0.472 with the empty-room recording. Empty room minus
    simulated: -0.347 (sd 0.150) and -0.403 (sd 0.185), n = 9.

**The run itself.**

18. The saturation guard passed on the chain of `full`. All ten protocols finished with no failed task.
    The measured run time is 2.22 h against the budget of 2 h (table "Run time").

**Add-on protocols, run after the ten (their own sections and the second run-time table are below; they
ran at later commits, `009c4d1` and `3901193`).**

19. `noise_hard` (the noise study at -15 and -20 dB, same design as `noise`, nine seeds = three windows),
    gradiometers: the empty-room arm is not lower than the simulated arm at the sensor level. Empty
    room minus simulated, conditional GC / gPDC: -15 dB +0.042 (sd 0.088) / +0.014 (sd 0.042); -20 dB
    +0.306 (sd 0.226) / +0.264 (sd 0.276). At the source level at -20 dB: +0.069 (sd 0.243) / +0.125
    (sd 0.280). The region level, which does not see the sensor noise, is identical in the three arms:
    0.514 at -15 dB and 0.444 at -20 dB, so below the sensor level there.
20. `noise_hard`, magnetometers, sensor level: empty room minus simulated is -0.222 (sd 0.174) / -0.153
    (sd 0.104) at -15 dB and -0.194 (sd 0.167) / -0.014 (sd 0.159) at -20 dB. With item 17 that is
    negative in five of the six SNR and estimator combinations and about zero in the sixth; the sign is
    the opposite of the gradiometers'. At the magnetometer source level the differences are -0.069 (sd
    0.141) / -0.125 (sd 0.207) at -15 dB and +0.042 (sd 0.265) / +0.014 (sd 0.321) at -20 dB, the sd
    being as large as or larger than the mean.
21. `significance_power` (region level, chain, conditional GC at fixed order, 40 datasets, 99 surrogates,
    coupling 0, 0.1, 0.2 and 0.4 at 10 and 40 dB). Null cells, rate at the 0.05 level of the analytic
    test: 0.062 (0, 10 dB), 0.050 (0, 40 dB), 0.075 with interval 0.031 to 0.125 (0.1, 10 dB), 0.044
    (0.1, 40 dB), 0.050 (0.2, 10 dB), 0.062 (0.2, 40 dB), **0.181 with interval 0.131 to 0.231 (0.4, 10
    dB, 160 cells)** and 0.050 (0.4, 40 dB). Only the (0.4, 10 dB) row has an interval that excludes 0.05;
    the first study had 0.150 in the same world. The surrogate test: 0.058, 0.054, 0.050, 0.044, 0.019, 0.037, 0.069 (0.031
    to 0.113) and 0.019. In the 0.4 / 10 dB row the share of datasets with a Benjamini-Hochberg false
    rejection is 0.475 (0.329 to 0.625) for the analytic test and 0.150 (0.071 to 0.291) for the
    surrogate test.
22. `significance_power`, true edges (80 edge cells per row): the power at 0.05 is 1.000 for both tests
    in all six rows, down to coupling 0.1 at 10 dB; the power at 0.01 is 1.000 too. With 99 surrogates
    the surrogate test with Benjamini-Hochberg has power 1.000 in all six rows (two cells are needed),
    where it was 0.000 with 39. The false-discovery proportion of Benjamini-Hochberg is 0.008 to 0.050
    for the surrogate test and 0.017 to 0.175 for the analytic test.
23. `m5_methods` (the further estimators: TRGC, imaginary coherence, PSI, MTGC, PCMCI and TE, next to conditional GC with BIC; chain, collider, common driver; 0 dB,
    60 s, EEG and gradiometers; five seeds), EEG sensor level, where the sensors mix the regions: in the
    collider the AUROC over seeds is 0.50 to 0.65 for `gc/bic`, 0.80 to 0.95 for `trgc/bic`, 0.95 to 1.00
    for `psi/mt` and 0.88 to 1.00 for `imcoh/mt` (undirected); in the common driver 0.40 to 0.50 for
    `gc/bic`, 0.60 to 0.85 for `trgc/bic`, 0.75 to 0.95 for `psi/mt` and 0.50 to 0.62 for `imcoh/mt`.
    `mtgc/mt`, `pcmci/bic` and `te/bic` stay between 0.40 and 0.80 in both. With gradiometers the mean
    AUROC at the sensor level is 0.87 to 1.00 for every estimator except `imcoh/mt` in the chain (0.70):
    the cells are near saturation.
24. `m5_methods`, region and source level: every estimator except `imcoh/mt` and `psi/mt` is at 0.92 to
    1.00 in mean AUROC. At these two levels `imcoh/mt` is at 0.50 to 0.60 in the chain and 0.70 to 0.88
    in the common driver, and `psi/mt` at 0.85 to 0.88 and 0.86 to 0.95; both are pairwise and do not
    condition on the third region
    (`docs/method_validation.md`, findings 10). Mean direction accuracy is 0.9 to 1.0 wherever it is
    defined, except in the EEG collider at the sensor level: 0.50 for `gc/bic` and 0.80 for `pcmci/bic`
    and `te/bic`.

## Not established

- **Why** the sensor level is lower than the region level in `full`, why `feedback_loop` carries no
  direction information at the sensor level, why conditional GC is below 0.5 for `common_driver` at the
  sensor level, why DTF and PDC are below 0.5 for `random_5` at the source level, and why LCMV is the
  lowest inverse method. Leakage between regions is a candidate for each; the correlation of item 14
  shows that false-edge scores follow the crosstalk, not that it causes these results. No run removed
  the leakage and looked again.
- Why the empty-room noise hurts the magnetometer sensor level (items 17 and 20) and does not hurt the
  gradiometers (item 19). The three noise windows are three independent samples, and no control
  separates the shape of the noise in time from its spread across sensors.
- Whether a fixed order is better than BIC at low SNR in general: item 6 is a pooled difference at the
  region level and is not repeated at the other stages.
- Why the analytic test exceeds its level at coupling 0.4 and 10 dB (items 15 and 21). Item 21 shows
  the excess is not there at coupling 0.2 with 10 dB or at 0.4 with 40 dB, so the combination matters;
  the order setting was not varied (the BIC arm was dropped to fit the budget), and what in the data
  produces the excess was not examined.
- Whether any estimator is better than another, overall or in a class of conditions. No ranking is made
  here, and the tables hold no interval for a cell mean.
- Whether larger networks are harder (item 10 has no consistent direction), and whether any of this
  holds for real recordings.
- Whether the controls of item 23 are better than conditional GC under sensor mixing in general: two
  topologies, one SNR and five seeds, EEG only (the gradiometer cells are saturated), and the region
  and source levels do not show the difference. Why `imcoh/mt` stays at 0.50 to 0.70 in the chain at
  every stage was not examined.

## Limits

- Simulated data on the head models of one subject of the MNE sample dataset (EEG, magnetometers,
  gradiometers); the region level is the node activity plus white noise.
- Few simulations per cell (above), and many cells at 1.000: at the region and source levels of `order`
  every cell outside `feedback_loop` is at 1.000 or within 0.025 of it, and in `full` the chain is 1.000
  at 0 dB and 0.963 to 1.000 at -5 dB at the region level. A cell at 1.000 says nothing about the order
  between estimators. The guard checks the chain of `full` only.
- Only the five BIC settings in `coupling`, `extent`, `size` and `leakage_adjacent` (the `partial` rows
  of `extent` and `size` come from `full` and also show the fixed order), two settings in `inverse`,
  `leakage_pairs` and `noise`; the fixed order is compared in `full` and `order` only. Not run: gradiometers in `full`, the
  graph of 20 regions, `focal` and `patch` for random networks, time-varying scenarios (the stage
  protocols refuse them), the empty room together with a simulated brain background, EEG in the noise
  study (the dataset has no empty-room recording for it).
- The budget is exceeded: 2.22 h measured. `extent`, `size` and `inverse` started with a gate estimate
  above their share and were run with `--force`; the results header does not record that the flag was
  used.
- Power is 1.000 in every row of `significance` and `significance_power` (item 22), so the two studies
  give no power curve: a weaker coupling or shorter data would be needed to find where it falls. Not run.
- The add-ons: `noise_hard` was estimated at 0.05 h and measured 0.03 h. `significance_power` was shrunk
  from 400 tasks with two estimators (gate estimate 1.3 h) to 320 tasks with one (0.21 h), still above its
  0.1 h budget, and run with `--force` as agreed; it measured 0.06 h. The estimates of the gate for this
  kind of task are again several times too high (`significance`: 0.06 h against 0.01 h).
- The gate estimate varies between invocations of the same protocol on this machine, so the "estimate"
  column of the run-time table is the estimate at the start of the run that wrote the file. `size` was
  refused at 0.25 h and the run that wrote its file started with 0.14 h; `inverse` was 0.24 h, 0.17 h
  and 0.29 h in three invocations (and 33 h in one that spanned a sleep of the computer).
- The add-on `m5_methods` had a budget of 0.3 h, chosen after a gate estimate of 0.14 h for
  three seeds; with five seeds the gate estimated 0.24 h and the run measured 0.24 h. It was rerun at
  `a0a081d` after PCMCI's default `tau_max` gained a 30 ms floor (0.24 h again): every cell is the
  same, because at 100 Hz the floor is 3 samples and the BIC orders of these tasks are 5 to 7.
- A first run of `inverse` was discarded because the checkout was switched to another branch while it
  ran (its header recorded `d2e98d1`, the code that ran was a mixture). The file in the report is a
  rerun in a separate worktree at `0de0d7f`.

# Benchmark results (generated)

Generated by `scripts/make_report.py` from `benchmark_results/*.jsonl`; every number is computed from those files and nothing here is edited by hand. Cells are mean AUROC over seeds unless said otherwise; a difference is taken task by task (same simulation) and shown as mean, sd and number of pairs.

## Run time

| protocol | tasks | estimate (h) | measured (h) | workers |
|---|---|---|---|---|
| full | 300 | 0.37 | 0.55 | 6 |
| order | 200 | 0.28 | 0.44 | 6 |
| inverse | 32 | 0.29 | 0.28 | 6 |
| coupling | 360 | 0.33 | 0.40 | 6 |
| extent | 240 | 0.30 | 0.23 | 6 |
| size | 60 | 0.14 | 0.21 | 6 |
| leakage_adjacent | 20 | 0.03 | 0.03 | 6 |
| leakage_pairs | 90 | 0.04 | 0.04 | 6 |
| significance | 100 | 0.06 | 0.01 | 6 |
| noise | 54 | 0.03 | 0.02 | 6 |
| **total** | | | **2.22** | budget 2 h |

Add-on protocols (each has a budget of its own and is not in the total above):

| protocol | tasks | estimate (h) | measured (h) | workers |
|---|---|---|---|---|
| noise_hard | 108 | 0.05 | 0.03 | 6 |
| significance_power | 320 | 0.21 | 0.06 | 6 |
| m5_methods | 30 | 0.23 | 0.24 | 6 |
| asymptotic_calibration | 200 | 1.20 | 1.00 | 6 |

Environment of every results file: Python 3.11.16; numpy 2.4.6, scipy 1.17.1, mne 1.13.2, statsmodels 0.15.0, joblib 1.6.0, dirconn 0.0.1.dev0.

Code: the commits differ between the files: full 0de0d7f; order 0de0d7f; inverse 0de0d7f; coupling 0de0d7f; extent 0de0d7f; size 0de0d7f; leakage_adjacent 0de0d7f; leakage_pairs 0de0d7f; significance 0de0d7f; noise 0de0d7f; noise_hard 009c4d1; significance_power 3901193; m5_methods a0a081d; asymptotic_calibration d3537a1.

## Recovery by stage

Protocol `full`: five scenarios, SNR 0, -5 and -10 dB, 30 and 60 s, EEG, partly coherent sources, ten seeds, averaged over all of it; the last two columns are the mean difference to the region level, taken task by task (sd, pairs).

| estimator | roi | sensor | source | sensor - roi | source - roi |
|---|---|---|---|---|---|
| dtf/bic | 0.864 | 0.624 | 0.803 | -0.240 (0.237, n=300) | -0.061 (0.243, n=300) |
| dtf/fixed | 0.888 | 0.626 | 0.800 | -0.263 (0.218, n=300) | -0.088 (0.236, n=300) |
| gc/bic | 0.864 | 0.631 | 0.813 | -0.233 (0.257, n=300) | -0.051 (0.212, n=300) |
| gc/fixed | 0.888 | 0.629 | 0.823 | -0.259 (0.245, n=300) | -0.064 (0.202, n=300) |
| gc_pairwise/bic | 0.866 | 0.637 | 0.813 | -0.229 (0.209, n=300) | -0.053 (0.222, n=300) |
| gc_pairwise/fixed | 0.887 | 0.652 | 0.825 | -0.235 (0.191, n=300) | -0.062 (0.211, n=300) |
| gpdc/bic | 0.865 | 0.629 | 0.808 | -0.236 (0.250, n=300) | -0.057 (0.227, n=300) |
| gpdc/fixed | 0.891 | 0.622 | 0.817 | -0.268 (0.232, n=300) | -0.073 (0.223, n=300) |
| pdc/bic | 0.864 | 0.618 | 0.800 | -0.247 (0.249, n=300) | -0.065 (0.233, n=300) |
| pdc/fixed | 0.888 | 0.608 | 0.801 | -0.280 (0.233, n=300) | -0.087 (0.235, n=300) |

By condition (rows: scenario, SNR, data length):

Stage: roi.

| scenario | SNR (dB) | length (s) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gc_pairwise/bic | gc_pairwise/fixed | gpdc/bic | gpdc/fixed | pdc/bic | pdc/fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | -10 | 30 | 0.925 | 0.925 | 0.925 | 0.925 | 0.938 | 0.938 | 0.925 | 0.938 | 0.925 | 0.950 |
| chain | -10 | 60 | 0.875 | 0.963 | 0.875 | 0.975 | 0.887 | 0.950 | 0.863 | 0.975 | 0.875 | 0.975 |
| chain | -5 | 30 | 0.988 | 0.963 | 1.000 | 0.988 | 1.000 | 1.000 | 1.000 | 0.988 | 1.000 | 0.988 |
| chain | -5 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 0 | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collider | -10 | 30 | 0.670 | 0.900 | 0.670 | 0.895 | 0.685 | 0.865 | 0.670 | 0.875 | 0.670 | 0.880 |
| collider | -10 | 60 | 0.655 | 0.965 | 0.655 | 0.970 | 0.670 | 0.975 | 0.655 | 0.965 | 0.655 | 0.965 |
| collider | -5 | 30 | 0.995 | 1.000 | 0.995 | 1.000 | 0.995 | 1.000 | 0.995 | 1.000 | 0.995 | 1.000 |
| collider | -5 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collider | 0 | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collider | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| common_driver | -10 | 30 | 0.870 | 0.895 | 0.870 | 0.830 | 0.870 | 0.830 | 0.870 | 0.860 | 0.870 | 0.855 |
| common_driver | -10 | 60 | 0.970 | 0.960 | 0.970 | 0.935 | 0.970 | 0.940 | 0.970 | 0.955 | 0.970 | 0.955 |
| common_driver | -5 | 30 | 1.000 | 0.995 | 1.000 | 0.995 | 0.985 | 0.985 | 1.000 | 0.995 | 1.000 | 0.995 |
| common_driver | -5 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 0.995 | 0.985 | 1.000 | 1.000 | 1.000 | 1.000 |
| common_driver | 0 | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| common_driver | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| feedback_loop | -10 | 30 | 0.659 | 0.496 | 0.656 | 0.489 | 0.663 | 0.496 | 0.656 | 0.496 | 0.656 | 0.496 |
| feedback_loop | -10 | 60 | 0.563 | 0.552 | 0.563 | 0.552 | 0.563 | 0.537 | 0.563 | 0.552 | 0.563 | 0.548 |
| feedback_loop | -5 | 30 | 0.593 | 0.633 | 0.589 | 0.611 | 0.581 | 0.611 | 0.593 | 0.641 | 0.600 | 0.637 |
| feedback_loop | -5 | 60 | 0.689 | 0.715 | 0.685 | 0.719 | 0.689 | 0.711 | 0.693 | 0.719 | 0.693 | 0.719 |
| feedback_loop | 0 | 30 | 0.852 | 0.696 | 0.848 | 0.700 | 0.841 | 0.681 | 0.859 | 0.685 | 0.852 | 0.681 |
| feedback_loop | 0 | 60 | 0.978 | 0.926 | 0.959 | 0.937 | 0.952 | 0.926 | 0.963 | 0.937 | 0.967 | 0.930 |
| random_5 | -10 | 30 | 0.551 | 0.693 | 0.550 | 0.679 | 0.551 | 0.690 | 0.551 | 0.694 | 0.551 | 0.689 |
| random_5 | -10 | 60 | 0.551 | 0.743 | 0.551 | 0.747 | 0.555 | 0.752 | 0.551 | 0.740 | 0.548 | 0.728 |
| random_5 | -5 | 30 | 0.748 | 0.864 | 0.736 | 0.855 | 0.748 | 0.892 | 0.741 | 0.864 | 0.741 | 0.852 |
| random_5 | -5 | 60 | 0.878 | 0.878 | 0.885 | 0.886 | 0.882 | 0.917 | 0.887 | 0.891 | 0.883 | 0.879 |
| random_5 | 0 | 30 | 0.947 | 0.930 | 0.951 | 0.959 | 0.979 | 0.968 | 0.954 | 0.960 | 0.943 | 0.939 |
| random_5 | 0 | 60 | 0.964 | 0.954 | 0.980 | 0.986 | 0.983 | 0.980 | 0.980 | 0.988 | 0.974 | 0.974 |

Stage: sensor.

| scenario | SNR (dB) | length (s) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gc_pairwise/bic | gc_pairwise/fixed | gpdc/bic | gpdc/fixed | pdc/bic | pdc/fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | -10 | 30 | 0.950 | 0.938 | 0.925 | 0.963 | 0.875 | 0.863 | 0.950 | 0.950 | 0.963 | 0.950 |
| chain | -10 | 60 | 0.975 | 0.988 | 0.963 | 1.000 | 0.950 | 0.963 | 0.975 | 0.975 | 0.975 | 0.988 |
| chain | -5 | 30 | 0.975 | 0.988 | 1.000 | 1.000 | 0.925 | 0.963 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | -5 | 60 | 0.975 | 0.963 | 1.000 | 1.000 | 0.975 | 0.963 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 0 | 30 | 0.988 | 0.988 | 1.000 | 1.000 | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collider | -10 | 30 | 0.510 | 0.550 | 0.530 | 0.545 | 0.495 | 0.580 | 0.535 | 0.570 | 0.555 | 0.580 |
| collider | -10 | 60 | 0.570 | 0.560 | 0.595 | 0.550 | 0.555 | 0.540 | 0.585 | 0.555 | 0.605 | 0.570 |
| collider | -5 | 30 | 0.505 | 0.585 | 0.530 | 0.550 | 0.550 | 0.600 | 0.540 | 0.550 | 0.520 | 0.565 |
| collider | -5 | 60 | 0.610 | 0.635 | 0.640 | 0.650 | 0.620 | 0.625 | 0.625 | 0.655 | 0.635 | 0.650 |
| collider | 0 | 30 | 0.600 | 0.630 | 0.635 | 0.625 | 0.620 | 0.690 | 0.660 | 0.620 | 0.595 | 0.600 |
| collider | 0 | 60 | 0.615 | 0.645 | 0.630 | 0.660 | 0.640 | 0.675 | 0.660 | 0.670 | 0.605 | 0.635 |
| common_driver | -10 | 30 | 0.355 | 0.455 | 0.305 | 0.305 | 0.360 | 0.480 | 0.320 | 0.345 | 0.355 | 0.405 |
| common_driver | -10 | 60 | 0.530 | 0.525 | 0.435 | 0.395 | 0.500 | 0.490 | 0.475 | 0.425 | 0.530 | 0.495 |
| common_driver | -5 | 30 | 0.440 | 0.505 | 0.400 | 0.380 | 0.535 | 0.610 | 0.405 | 0.405 | 0.415 | 0.430 |
| common_driver | -5 | 60 | 0.495 | 0.485 | 0.390 | 0.380 | 0.630 | 0.635 | 0.420 | 0.425 | 0.425 | 0.425 |
| common_driver | 0 | 30 | 0.575 | 0.565 | 0.495 | 0.455 | 0.670 | 0.690 | 0.515 | 0.490 | 0.520 | 0.495 |
| common_driver | 0 | 60 | 0.585 | 0.605 | 0.450 | 0.465 | 0.700 | 0.700 | 0.500 | 0.520 | 0.500 | 0.505 |
| feedback_loop | -10 | 30 | 0.478 | 0.489 | 0.481 | 0.493 | 0.415 | 0.459 | 0.478 | 0.456 | 0.481 | 0.489 |
| feedback_loop | -10 | 60 | 0.515 | 0.478 | 0.556 | 0.556 | 0.493 | 0.489 | 0.548 | 0.533 | 0.526 | 0.485 |
| feedback_loop | -5 | 30 | 0.507 | 0.504 | 0.496 | 0.493 | 0.485 | 0.515 | 0.496 | 0.485 | 0.489 | 0.485 |
| feedback_loop | -5 | 60 | 0.515 | 0.515 | 0.496 | 0.500 | 0.507 | 0.500 | 0.496 | 0.504 | 0.493 | 0.500 |
| feedback_loop | 0 | 30 | 0.515 | 0.507 | 0.481 | 0.470 | 0.544 | 0.548 | 0.489 | 0.478 | 0.515 | 0.493 |
| feedback_loop | 0 | 60 | 0.500 | 0.504 | 0.470 | 0.485 | 0.541 | 0.552 | 0.463 | 0.474 | 0.478 | 0.485 |
| random_5 | -10 | 30 | 0.551 | 0.479 | 0.624 | 0.614 | 0.516 | 0.510 | 0.597 | 0.542 | 0.562 | 0.461 |
| random_5 | -10 | 60 | 0.593 | 0.541 | 0.633 | 0.607 | 0.527 | 0.482 | 0.622 | 0.557 | 0.595 | 0.511 |
| random_5 | -5 | 30 | 0.562 | 0.544 | 0.655 | 0.657 | 0.590 | 0.595 | 0.614 | 0.604 | 0.549 | 0.512 |
| random_5 | -5 | 60 | 0.584 | 0.567 | 0.673 | 0.669 | 0.562 | 0.568 | 0.621 | 0.610 | 0.556 | 0.536 |
| random_5 | 0 | 30 | 0.605 | 0.505 | 0.757 | 0.721 | 0.723 | 0.648 | 0.664 | 0.640 | 0.564 | 0.485 |
| random_5 | 0 | 60 | 0.553 | 0.528 | 0.689 | 0.689 | 0.620 | 0.639 | 0.619 | 0.626 | 0.522 | 0.505 |

Stage: source.

| scenario | SNR (dB) | length (s) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gc_pairwise/bic | gc_pairwise/fixed | gpdc/bic | gpdc/fixed | pdc/bic | pdc/fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | -10 | 30 | 0.825 | 0.863 | 0.775 | 0.838 | 0.787 | 0.825 | 0.762 | 0.863 | 0.800 | 0.887 |
| chain | -10 | 60 | 0.950 | 0.963 | 0.938 | 0.950 | 0.975 | 0.938 | 0.963 | 0.963 | 0.950 | 0.963 |
| chain | -5 | 30 | 0.988 | 0.988 | 0.988 | 1.000 | 1.000 | 1.000 | 0.988 | 1.000 | 0.988 | 1.000 |
| chain | -5 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 0 | 30 | 0.975 | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collider | -10 | 30 | 0.640 | 0.850 | 0.635 | 0.840 | 0.665 | 0.840 | 0.655 | 0.860 | 0.630 | 0.840 |
| collider | -10 | 60 | 0.930 | 0.945 | 0.955 | 0.950 | 0.955 | 0.965 | 0.955 | 0.970 | 0.940 | 0.940 |
| collider | -5 | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collider | -5 | 60 | 1.000 | 1.000 | 0.995 | 0.995 | 1.000 | 1.000 | 0.995 | 0.995 | 0.995 | 0.995 |
| collider | 0 | 30 | 0.995 | 1.000 | 0.970 | 0.980 | 1.000 | 1.000 | 0.975 | 0.980 | 0.960 | 0.975 |
| collider | 0 | 60 | 1.000 | 1.000 | 0.980 | 0.985 | 1.000 | 1.000 | 0.980 | 0.985 | 0.965 | 0.985 |
| common_driver | -10 | 30 | 0.880 | 0.920 | 0.865 | 0.905 | 0.855 | 0.915 | 0.880 | 0.915 | 0.895 | 0.930 |
| common_driver | -10 | 60 | 0.950 | 0.925 | 0.940 | 0.950 | 0.945 | 0.950 | 0.945 | 0.945 | 0.945 | 0.915 |
| common_driver | -5 | 30 | 1.000 | 1.000 | 0.980 | 0.970 | 0.995 | 0.985 | 0.980 | 0.965 | 0.975 | 0.965 |
| common_driver | -5 | 60 | 1.000 | 1.000 | 0.970 | 0.975 | 0.985 | 0.985 | 0.970 | 0.975 | 0.985 | 0.990 |
| common_driver | 0 | 30 | 1.000 | 1.000 | 0.975 | 0.975 | 1.000 | 1.000 | 0.965 | 0.970 | 0.975 | 0.985 |
| common_driver | 0 | 60 | 1.000 | 1.000 | 0.975 | 0.975 | 1.000 | 1.000 | 0.970 | 0.970 | 0.985 | 0.995 |
| feedback_loop | -10 | 30 | 0.637 | 0.604 | 0.637 | 0.615 | 0.622 | 0.626 | 0.648 | 0.626 | 0.622 | 0.604 |
| feedback_loop | -10 | 60 | 0.733 | 0.656 | 0.685 | 0.678 | 0.693 | 0.659 | 0.715 | 0.689 | 0.704 | 0.667 |
| feedback_loop | -5 | 30 | 0.852 | 0.774 | 0.841 | 0.789 | 0.837 | 0.793 | 0.848 | 0.804 | 0.844 | 0.770 |
| feedback_loop | -5 | 60 | 0.796 | 0.767 | 0.770 | 0.756 | 0.785 | 0.767 | 0.774 | 0.826 | 0.778 | 0.774 |
| feedback_loop | 0 | 30 | 0.778 | 0.733 | 0.744 | 0.726 | 0.774 | 0.759 | 0.748 | 0.737 | 0.748 | 0.733 |
| feedback_loop | 0 | 60 | 0.756 | 0.770 | 0.830 | 0.837 | 0.819 | 0.841 | 0.830 | 0.819 | 0.800 | 0.811 |
| random_5 | -10 | 30 | 0.447 | 0.403 | 0.476 | 0.486 | 0.409 | 0.465 | 0.460 | 0.462 | 0.458 | 0.408 |
| random_5 | -10 | 60 | 0.383 | 0.375 | 0.428 | 0.453 | 0.375 | 0.432 | 0.406 | 0.402 | 0.395 | 0.370 |
| random_5 | -5 | 30 | 0.416 | 0.361 | 0.508 | 0.501 | 0.480 | 0.509 | 0.455 | 0.433 | 0.429 | 0.358 |
| random_5 | -5 | 60 | 0.410 | 0.399 | 0.494 | 0.524 | 0.459 | 0.501 | 0.458 | 0.463 | 0.408 | 0.401 |
| random_5 | 0 | 30 | 0.352 | 0.349 | 0.504 | 0.509 | 0.494 | 0.499 | 0.445 | 0.439 | 0.386 | 0.361 |
| random_5 | 0 | 60 | 0.396 | 0.376 | 0.537 | 0.543 | 0.485 | 0.503 | 0.465 | 0.467 | 0.426 | 0.395 |

Saturation guard (the chain, best estimator per grid point): passed.

## Order setting

Fixed order (0.2 s) minus BIC order, the same simulation, in `full`, per method, stage and SNR (all scenarios and both lengths pooled):

| method | stage | SNR (dB) | mean difference | sd | pairs |
|---|---|---|---|---|---|
| dtf | roi | -10 | +0.080 | 0.209 | 100 |
| dtf | roi | -5 | +0.016 | 0.118 | 100 |
| dtf | roi | 0 | -0.023 | 0.066 | 100 |
| dtf | sensor | -10 | -0.002 | 0.104 | 100 |
| dtf | sensor | -5 | +0.012 | 0.062 | 100 |
| dtf | sensor | 0 | -0.006 | 0.048 | 100 |
| dtf | source | -10 | +0.013 | 0.150 | 100 |
| dtf | source | -5 | -0.017 | 0.051 | 100 |
| dtf | source | 0 | -0.004 | 0.031 | 100 |
| gc | roi | -10 | +0.071 | 0.221 | 100 |
| gc | roi | -5 | +0.016 | 0.108 | 100 |
| gc | roi | 0 | -0.016 | 0.065 | 100 |
| gc | sensor | -10 | -0.002 | 0.086 | 100 |
| gc | sensor | -5 | -0.000 | 0.048 | 100 |
| gc | sensor | 0 | -0.004 | 0.044 | 100 |
| gc | source | -10 | +0.033 | 0.149 | 100 |
| gc | source | -5 | -0.004 | 0.050 | 100 |
| gc | source | 0 | +0.001 | 0.029 | 100 |
| gc_pairwise | roi | -10 | +0.062 | 0.211 | 100 |
| gc_pairwise | roi | -5 | +0.021 | 0.105 | 100 |
| gc_pairwise | roi | 0 | -0.020 | 0.068 | 100 |
| gc_pairwise | sensor | -10 | +0.017 | 0.095 | 100 |
| gc_pairwise | sensor | -5 | +0.019 | 0.065 | 100 |
| gc_pairwise | sensor | 0 | +0.010 | 0.057 | 100 |
| gc_pairwise | source | -10 | +0.033 | 0.156 | 100 |
| gc_pairwise | source | -5 | -0.000 | 0.047 | 100 |
| gc_pairwise | source | 0 | +0.003 | 0.031 | 100 |
| gpdc | roi | -10 | +0.078 | 0.208 | 100 |
| gpdc | roi | -5 | +0.019 | 0.119 | 100 |
| gpdc | roi | 0 | -0.019 | 0.067 | 100 |
| gpdc | sensor | -10 | -0.018 | 0.097 | 100 |
| gpdc | sensor | -5 | +0.002 | 0.059 | 100 |
| gpdc | sensor | 0 | -0.005 | 0.040 | 100 |
| gpdc | source | -10 | +0.030 | 0.154 | 100 |
| gpdc | source | -5 | -0.001 | 0.054 | 100 |
| gpdc | source | 0 | -0.001 | 0.029 | 100 |
| pdc | roi | -10 | +0.076 | 0.206 | 100 |
| pdc | roi | -5 | +0.016 | 0.116 | 100 |
| pdc | roi | 0 | -0.021 | 0.069 | 100 |
| pdc | sensor | -10 | -0.021 | 0.098 | 100 |
| pdc | sensor | -5 | +0.002 | 0.055 | 100 |
| pdc | sensor | 0 | -0.009 | 0.051 | 100 |
| pdc | source | -10 | +0.018 | 0.147 | 100 |
| pdc | source | -5 | -0.015 | 0.059 | 100 |
| pdc | source | 0 | -0.000 | 0.035 | 100 |

## Conditional against pairwise Granger causality

Conditional minus pairwise GC (AUROC), the same simulation and order setting, in `full`, per scenario and stage:

| scenario | stage | mean difference | sd | pairs |
|---|---|---|---|---|
| chain | roi | -0.001 | 0.030 | 120 |
| chain | sensor | +0.032 | 0.075 | 120 |
| chain | source | -0.003 | 0.038 | 120 |
| collider | roi | -0.000 | 0.023 | 120 |
| collider | sensor | -0.004 | 0.094 | 120 |
| collider | source | -0.012 | 0.037 | 120 |
| common_driver | roi | +0.003 | 0.020 | 120 |
| common_driver | sensor | -0.179 | 0.123 | 120 |
| common_driver | source | -0.013 | 0.042 | 120 |
| feedback_loop | roi | +0.005 | 0.033 | 120 |
| feedback_loop | sensor | -0.006 | 0.083 | 120 |
| feedback_loop | source | -0.006 | 0.057 | 120 |
| random_5 | roi | -0.011 | 0.028 | 120 |
| random_5 | sensor | +0.084 | 0.052 | 120 |
| random_5 | source | +0.029 | 0.053 | 120 |

## Coupling strength

Protocol `coupling` (0.1 and 0.2) with the file strength 0.4 of `full`, same grid; mean over the lengths 30 and 60 s (rows: scenario, coupling, SNR).

Stage: roi.

| scenario | coupling | SNR (dB) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gc_pairwise/bic | gc_pairwise/fixed | gpdc/bic | gpdc/fixed | pdc/bic | pdc/fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | 0.1 | -10 | 0.500 | - | 0.500 | - | 0.494 | - | 0.500 | - | 0.500 | - |
| chain | 0.1 | -5 | 0.856 | - | 0.850 | - | 0.856 | - | 0.844 | - | 0.838 | - |
| chain | 0.1 | 0 | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - |
| chain | 0.2 | -10 | 0.600 | - | 0.600 | - | 0.613 | - | 0.600 | - | 0.600 | - |
| chain | 0.2 | -5 | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - |
| chain | 0.2 | 0 | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - |
| chain | 0.4 | -10 | 0.900 | 0.944 | 0.900 | 0.950 | 0.912 | 0.944 | 0.894 | 0.956 | 0.900 | 0.963 |
| chain | 0.4 | -5 | 0.994 | 0.981 | 1.000 | 0.994 | 1.000 | 0.994 | 1.000 | 0.994 | 1.000 | 0.994 |
| chain | 0.4 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collider | 0.1 | -10 | 0.590 | - | 0.590 | - | 0.585 | - | 0.588 | - | 0.588 | - |
| collider | 0.1 | -5 | 0.948 | - | 0.942 | - | 0.948 | - | 0.948 | - | 0.948 | - |
| collider | 0.1 | 0 | 0.997 | - | 0.995 | - | 0.995 | - | 0.995 | - | 0.995 | - |
| collider | 0.2 | -10 | 0.775 | - | 0.770 | - | 0.778 | - | 0.770 | - | 0.770 | - |
| collider | 0.2 | -5 | 0.978 | - | 0.978 | - | 0.980 | - | 0.978 | - | 0.978 | - |
| collider | 0.2 | 0 | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - |
| collider | 0.4 | -10 | 0.662 | 0.932 | 0.662 | 0.932 | 0.677 | 0.920 | 0.662 | 0.920 | 0.662 | 0.922 |
| collider | 0.4 | -5 | 0.997 | 1.000 | 0.997 | 1.000 | 0.997 | 1.000 | 0.997 | 1.000 | 0.997 | 1.000 |
| collider | 0.4 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| common_driver | 0.1 | -10 | 0.707 | - | 0.705 | - | 0.705 | - | 0.707 | - | 0.707 | - |
| common_driver | 0.1 | -5 | 0.968 | - | 0.965 | - | 0.968 | - | 0.968 | - | 0.968 | - |
| common_driver | 0.1 | 0 | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - |
| common_driver | 0.2 | -10 | 0.908 | - | 0.908 | - | 0.908 | - | 0.910 | - | 0.910 | - |
| common_driver | 0.2 | -5 | 0.990 | - | 0.990 | - | 0.985 | - | 0.990 | - | 0.988 | - |
| common_driver | 0.2 | 0 | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - |
| common_driver | 0.4 | -10 | 0.920 | 0.927 | 0.920 | 0.883 | 0.920 | 0.885 | 0.920 | 0.907 | 0.920 | 0.905 |
| common_driver | 0.4 | -5 | 1.000 | 0.997 | 1.000 | 0.997 | 0.990 | 0.985 | 1.000 | 0.997 | 1.000 | 0.997 |
| common_driver | 0.4 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Stage: sensor.

| scenario | coupling | SNR (dB) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gc_pairwise/bic | gc_pairwise/fixed | gpdc/bic | gpdc/fixed | pdc/bic | pdc/fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | 0.1 | -10 | 0.744 | - | 0.781 | - | 0.756 | - | 0.781 | - | 0.756 | - |
| chain | 0.1 | -5 | 0.850 | - | 0.869 | - | 0.919 | - | 0.869 | - | 0.875 | - |
| chain | 0.1 | 0 | 0.938 | - | 0.925 | - | 0.988 | - | 0.963 | - | 0.975 | - |
| chain | 0.2 | -10 | 0.856 | - | 0.863 | - | 0.863 | - | 0.887 | - | 0.850 | - |
| chain | 0.2 | -5 | 0.975 | - | 0.963 | - | 0.975 | - | 0.975 | - | 0.963 | - |
| chain | 0.2 | 0 | 0.931 | - | 0.963 | - | 0.994 | - | 0.956 | - | 0.963 | - |
| chain | 0.4 | -10 | 0.963 | 0.963 | 0.944 | 0.981 | 0.912 | 0.912 | 0.963 | 0.963 | 0.969 | 0.969 |
| chain | 0.4 | -5 | 0.975 | 0.975 | 1.000 | 1.000 | 0.950 | 0.963 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 0.4 | 0 | 0.994 | 0.994 | 1.000 | 1.000 | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collider | 0.1 | -10 | 0.495 | - | 0.530 | - | 0.463 | - | 0.537 | - | 0.517 | - |
| collider | 0.1 | -5 | 0.475 | - | 0.497 | - | 0.435 | - | 0.490 | - | 0.485 | - |
| collider | 0.1 | 0 | 0.520 | - | 0.522 | - | 0.440 | - | 0.510 | - | 0.500 | - |
| collider | 0.2 | -10 | 0.510 | - | 0.542 | - | 0.503 | - | 0.557 | - | 0.542 | - |
| collider | 0.2 | -5 | 0.538 | - | 0.535 | - | 0.515 | - | 0.545 | - | 0.520 | - |
| collider | 0.2 | 0 | 0.580 | - | 0.565 | - | 0.535 | - | 0.565 | - | 0.548 | - |
| collider | 0.4 | -10 | 0.540 | 0.555 | 0.562 | 0.547 | 0.525 | 0.560 | 0.560 | 0.562 | 0.580 | 0.575 |
| collider | 0.4 | -5 | 0.557 | 0.610 | 0.585 | 0.600 | 0.585 | 0.613 | 0.583 | 0.603 | 0.578 | 0.608 |
| collider | 0.4 | 0 | 0.607 | 0.637 | 0.633 | 0.642 | 0.630 | 0.682 | 0.660 | 0.645 | 0.600 | 0.617 |
| common_driver | 0.1 | -10 | 0.423 | - | 0.372 | - | 0.405 | - | 0.370 | - | 0.415 | - |
| common_driver | 0.1 | -5 | 0.458 | - | 0.413 | - | 0.455 | - | 0.435 | - | 0.470 | - |
| common_driver | 0.1 | 0 | 0.660 | - | 0.517 | - | 0.578 | - | 0.570 | - | 0.620 | - |
| common_driver | 0.2 | -10 | 0.445 | - | 0.405 | - | 0.460 | - | 0.390 | - | 0.432 | - |
| common_driver | 0.2 | -5 | 0.528 | - | 0.450 | - | 0.555 | - | 0.468 | - | 0.495 | - |
| common_driver | 0.2 | 0 | 0.645 | - | 0.512 | - | 0.623 | - | 0.568 | - | 0.588 | - |
| common_driver | 0.4 | -10 | 0.443 | 0.490 | 0.370 | 0.350 | 0.430 | 0.485 | 0.398 | 0.385 | 0.443 | 0.450 |
| common_driver | 0.4 | -5 | 0.468 | 0.495 | 0.395 | 0.380 | 0.583 | 0.622 | 0.412 | 0.415 | 0.420 | 0.427 |
| common_driver | 0.4 | 0 | 0.580 | 0.585 | 0.473 | 0.460 | 0.685 | 0.695 | 0.508 | 0.505 | 0.510 | 0.500 |

Stage: source.

| scenario | coupling | SNR (dB) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gc_pairwise/bic | gc_pairwise/fixed | gpdc/bic | gpdc/fixed | pdc/bic | pdc/fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | 0.1 | -10 | 0.700 | - | 0.656 | - | 0.688 | - | 0.644 | - | 0.688 | - |
| chain | 0.1 | -5 | 0.894 | - | 0.869 | - | 0.900 | - | 0.881 | - | 0.881 | - |
| chain | 0.1 | 0 | 0.875 | - | 0.869 | - | 0.875 | - | 0.869 | - | 0.875 | - |
| chain | 0.2 | -10 | 0.863 | - | 0.831 | - | 0.850 | - | 0.850 | - | 0.838 | - |
| chain | 0.2 | -5 | 0.963 | - | 0.900 | - | 0.963 | - | 0.912 | - | 0.944 | - |
| chain | 0.2 | 0 | 0.919 | - | 0.875 | - | 0.994 | - | 0.887 | - | 0.912 | - |
| chain | 0.4 | -10 | 0.887 | 0.912 | 0.856 | 0.894 | 0.881 | 0.881 | 0.863 | 0.912 | 0.875 | 0.925 |
| chain | 0.4 | -5 | 0.994 | 0.994 | 0.994 | 1.000 | 1.000 | 1.000 | 0.994 | 1.000 | 0.994 | 1.000 |
| chain | 0.4 | 0 | 0.988 | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collider | 0.1 | -10 | 0.727 | - | 0.722 | - | 0.725 | - | 0.740 | - | 0.720 | - |
| collider | 0.1 | -5 | 0.920 | - | 0.870 | - | 0.955 | - | 0.890 | - | 0.905 | - |
| collider | 0.1 | 0 | 0.880 | - | 0.840 | - | 0.937 | - | 0.845 | - | 0.857 | - |
| collider | 0.2 | -10 | 0.812 | - | 0.790 | - | 0.807 | - | 0.823 | - | 0.815 | - |
| collider | 0.2 | -5 | 0.985 | - | 0.967 | - | 0.997 | - | 0.975 | - | 0.972 | - |
| collider | 0.2 | 0 | 0.945 | - | 0.920 | - | 1.000 | - | 0.922 | - | 0.912 | - |
| collider | 0.4 | -10 | 0.785 | 0.897 | 0.795 | 0.895 | 0.810 | 0.902 | 0.805 | 0.915 | 0.785 | 0.890 |
| collider | 0.4 | -5 | 1.000 | 1.000 | 0.997 | 0.997 | 1.000 | 1.000 | 0.997 | 0.997 | 0.997 | 0.997 |
| collider | 0.4 | 0 | 0.997 | 1.000 | 0.975 | 0.983 | 1.000 | 1.000 | 0.978 | 0.982 | 0.962 | 0.980 |
| common_driver | 0.1 | -10 | 0.802 | - | 0.763 | - | 0.772 | - | 0.785 | - | 0.792 | - |
| common_driver | 0.1 | -5 | 0.965 | - | 0.932 | - | 0.955 | - | 0.940 | - | 0.955 | - |
| common_driver | 0.1 | 0 | 0.950 | - | 0.950 | - | 0.950 | - | 0.950 | - | 0.950 | - |
| common_driver | 0.2 | -10 | 0.872 | - | 0.863 | - | 0.878 | - | 0.870 | - | 0.870 | - |
| common_driver | 0.2 | -5 | 0.950 | - | 0.930 | - | 0.948 | - | 0.927 | - | 0.942 | - |
| common_driver | 0.2 | 0 | 0.997 | - | 0.960 | - | 1.000 | - | 0.952 | - | 0.962 | - |
| common_driver | 0.4 | -10 | 0.915 | 0.922 | 0.902 | 0.927 | 0.900 | 0.932 | 0.912 | 0.930 | 0.920 | 0.922 |
| common_driver | 0.4 | -5 | 1.000 | 1.000 | 0.975 | 0.972 | 0.990 | 0.985 | 0.975 | 0.970 | 0.980 | 0.978 |
| common_driver | 0.4 | 0 | 1.000 | 1.000 | 0.975 | 0.975 | 1.000 | 1.000 | 0.967 | 0.970 | 0.980 | 0.990 |

## Extent of the sources

Protocol `extent` (focal and patch) with the partly coherent sources of `full`, mean over the lengths (rows: scenario, source model, SNR).

Stage: roi.

| scenario | source model | SNR (dB) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gc_pairwise/bic | gc_pairwise/fixed | gpdc/bic | gpdc/fixed | pdc/bic | pdc/fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | focal | -10 | 0.731 | - | 0.731 | - | 0.750 | - | 0.731 | - | 0.731 | - |
| chain | focal | -5 | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - |
| chain | focal | 0 | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - |
| chain | partial | -10 | 0.900 | 0.944 | 0.900 | 0.950 | 0.912 | 0.944 | 0.894 | 0.956 | 0.900 | 0.963 |
| chain | partial | -5 | 0.994 | 0.981 | 1.000 | 0.994 | 1.000 | 0.994 | 1.000 | 0.994 | 1.000 | 0.994 |
| chain | partial | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | patch | -10 | 0.806 | - | 0.806 | - | 0.800 | - | 0.806 | - | 0.806 | - |
| chain | patch | -5 | 1.000 | - | 1.000 | - | 0.994 | - | 1.000 | - | 1.000 | - |
| chain | patch | 0 | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - |
| collider | focal | -10 | 0.820 | - | 0.820 | - | 0.825 | - | 0.820 | - | 0.820 | - |
| collider | focal | -5 | 0.988 | - | 0.993 | - | 0.990 | - | 0.990 | - | 0.988 | - |
| collider | focal | 0 | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - |
| collider | partial | -10 | 0.662 | 0.932 | 0.662 | 0.932 | 0.677 | 0.920 | 0.662 | 0.920 | 0.662 | 0.922 |
| collider | partial | -5 | 0.997 | 1.000 | 0.997 | 1.000 | 0.997 | 1.000 | 0.997 | 1.000 | 0.997 | 1.000 |
| collider | partial | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collider | patch | -10 | 0.785 | - | 0.788 | - | 0.795 | - | 0.785 | - | 0.785 | - |
| collider | patch | -5 | 0.975 | - | 0.982 | - | 0.972 | - | 0.982 | - | 0.982 | - |
| collider | patch | 0 | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - |

Stage: sensor.

| scenario | source model | SNR (dB) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gc_pairwise/bic | gc_pairwise/fixed | gpdc/bic | gpdc/fixed | pdc/bic | pdc/fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | focal | -10 | 0.713 | - | 0.681 | - | 0.562 | - | 0.706 | - | 0.688 | - |
| chain | focal | -5 | 0.713 | - | 0.725 | - | 0.537 | - | 0.694 | - | 0.713 | - |
| chain | focal | 0 | 0.688 | - | 0.650 | - | 0.519 | - | 0.606 | - | 0.606 | - |
| chain | partial | -10 | 0.963 | 0.963 | 0.944 | 0.981 | 0.912 | 0.912 | 0.963 | 0.963 | 0.969 | 0.969 |
| chain | partial | -5 | 0.975 | 0.975 | 1.000 | 1.000 | 0.950 | 0.963 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | partial | 0 | 0.994 | 0.994 | 1.000 | 1.000 | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | patch | -10 | 0.956 | - | 0.956 | - | 0.906 | - | 0.963 | - | 0.956 | - |
| chain | patch | -5 | 0.975 | - | 0.994 | - | 0.988 | - | 0.994 | - | 0.988 | - |
| chain | patch | 0 | 0.988 | - | 1.000 | - | 0.994 | - | 1.000 | - | 1.000 | - |
| collider | focal | -10 | 0.480 | - | 0.493 | - | 0.490 | - | 0.510 | - | 0.483 | - |
| collider | focal | -5 | 0.510 | - | 0.550 | - | 0.535 | - | 0.560 | - | 0.552 | - |
| collider | focal | 0 | 0.470 | - | 0.517 | - | 0.528 | - | 0.535 | - | 0.535 | - |
| collider | partial | -10 | 0.540 | 0.555 | 0.562 | 0.547 | 0.525 | 0.560 | 0.560 | 0.562 | 0.580 | 0.575 |
| collider | partial | -5 | 0.557 | 0.610 | 0.585 | 0.600 | 0.585 | 0.613 | 0.583 | 0.603 | 0.578 | 0.608 |
| collider | partial | 0 | 0.607 | 0.637 | 0.633 | 0.642 | 0.630 | 0.682 | 0.660 | 0.645 | 0.600 | 0.617 |
| collider | patch | -10 | 0.542 | - | 0.555 | - | 0.517 | - | 0.552 | - | 0.542 | - |
| collider | patch | -5 | 0.563 | - | 0.580 | - | 0.615 | - | 0.585 | - | 0.565 | - |
| collider | patch | 0 | 0.602 | - | 0.637 | - | 0.657 | - | 0.625 | - | 0.608 | - |

Stage: source.

| scenario | source model | SNR (dB) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gc_pairwise/bic | gc_pairwise/fixed | gpdc/bic | gpdc/fixed | pdc/bic | pdc/fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | focal | -10 | 0.775 | - | 0.769 | - | 0.762 | - | 0.713 | - | 0.794 | - |
| chain | focal | -5 | 0.912 | - | 0.950 | - | 0.938 | - | 0.931 | - | 0.938 | - |
| chain | focal | 0 | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - |
| chain | partial | -10 | 0.887 | 0.912 | 0.856 | 0.894 | 0.881 | 0.881 | 0.863 | 0.912 | 0.875 | 0.925 |
| chain | partial | -5 | 0.994 | 0.994 | 0.994 | 1.000 | 1.000 | 1.000 | 0.994 | 1.000 | 0.994 | 1.000 |
| chain | partial | 0 | 0.988 | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | patch | -10 | 0.906 | - | 0.894 | - | 0.931 | - | 0.906 | - | 0.887 | - |
| chain | patch | -5 | 1.000 | - | 0.988 | - | 1.000 | - | 0.994 | - | 1.000 | - |
| chain | patch | 0 | 0.988 | - | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - |
| collider | focal | -10 | 0.675 | - | 0.685 | - | 0.700 | - | 0.698 | - | 0.677 | - |
| collider | focal | -5 | 0.953 | - | 0.960 | - | 0.975 | - | 0.950 | - | 0.955 | - |
| collider | focal | 0 | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - | 1.000 | - |
| collider | partial | -10 | 0.785 | 0.897 | 0.795 | 0.895 | 0.810 | 0.902 | 0.805 | 0.915 | 0.785 | 0.890 |
| collider | partial | -5 | 1.000 | 1.000 | 0.997 | 0.997 | 1.000 | 1.000 | 0.997 | 0.997 | 0.997 | 0.997 |
| collider | partial | 0 | 0.997 | 1.000 | 0.975 | 0.983 | 1.000 | 1.000 | 0.978 | 0.982 | 0.962 | 0.980 |
| collider | patch | -10 | 0.873 | - | 0.857 | - | 0.890 | - | 0.867 | - | 0.853 | - |
| collider | patch | -5 | 0.997 | - | 0.990 | - | 1.000 | - | 0.997 | - | 0.993 | - |
| collider | patch | 0 | 0.995 | - | 0.965 | - | 1.000 | - | 0.965 | - | 0.955 | - |

## Network size

Protocol `size` (`random_10`) with `random_5` of `full`, mean over the lengths (rows: scenario, SNR).

Stage: roi.

| scenario | SNR (dB) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gc_pairwise/bic | gc_pairwise/fixed | gpdc/bic | gpdc/fixed | pdc/bic | pdc/fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random_10 | -10 | 0.658 | - | 0.656 | - | 0.664 | - | 0.656 | - | 0.656 | - |
| random_10 | -5 | 0.823 | - | 0.824 | - | 0.813 | - | 0.825 | - | 0.824 | - |
| random_10 | 0 | 0.932 | - | 0.948 | - | 0.900 | - | 0.950 | - | 0.941 | - |
| random_5 | -10 | 0.551 | 0.718 | 0.551 | 0.713 | 0.553 | 0.721 | 0.551 | 0.717 | 0.549 | 0.708 |
| random_5 | -5 | 0.813 | 0.871 | 0.811 | 0.871 | 0.815 | 0.904 | 0.814 | 0.877 | 0.812 | 0.866 |
| random_5 | 0 | 0.955 | 0.942 | 0.966 | 0.973 | 0.981 | 0.974 | 0.967 | 0.974 | 0.958 | 0.956 |

Stage: sensor.

| scenario | SNR (dB) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gc_pairwise/bic | gc_pairwise/fixed | gpdc/bic | gpdc/fixed | pdc/bic | pdc/fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random_10 | -10 | 0.480 | - | 0.470 | - | 0.462 | - | 0.470 | - | 0.481 | - |
| random_10 | -5 | 0.602 | - | 0.562 | - | 0.574 | - | 0.566 | - | 0.586 | - |
| random_10 | 0 | 0.661 | - | 0.581 | - | 0.662 | - | 0.585 | - | 0.608 | - |
| random_5 | -10 | 0.572 | 0.510 | 0.629 | 0.610 | 0.521 | 0.496 | 0.609 | 0.549 | 0.579 | 0.486 |
| random_5 | -5 | 0.573 | 0.555 | 0.664 | 0.663 | 0.576 | 0.581 | 0.617 | 0.607 | 0.553 | 0.524 |
| random_5 | 0 | 0.579 | 0.517 | 0.723 | 0.705 | 0.671 | 0.643 | 0.641 | 0.633 | 0.543 | 0.495 |

Stage: source.

| scenario | SNR (dB) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gc_pairwise/bic | gc_pairwise/fixed | gpdc/bic | gpdc/fixed | pdc/bic | pdc/fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random_10 | -10 | 0.489 | - | 0.492 | - | 0.445 | - | 0.482 | - | 0.488 | - |
| random_10 | -5 | 0.535 | - | 0.528 | - | 0.500 | - | 0.514 | - | 0.527 | - |
| random_10 | 0 | 0.601 | - | 0.607 | - | 0.635 | - | 0.572 | - | 0.589 | - |
| random_5 | -10 | 0.415 | 0.389 | 0.452 | 0.470 | 0.392 | 0.448 | 0.433 | 0.432 | 0.427 | 0.389 |
| random_5 | -5 | 0.413 | 0.380 | 0.501 | 0.512 | 0.470 | 0.505 | 0.457 | 0.448 | 0.419 | 0.380 |
| random_5 | 0 | 0.374 | 0.362 | 0.521 | 0.526 | 0.490 | 0.501 | 0.455 | 0.453 | 0.406 | 0.378 |

## Order protocol

Protocol `order`: scenarios with narrow-band coupling and long delays (rows: scenario, SNR, length).

Stage: roi.

| scenario | SNR (dB) | length (s) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gpdc/bic | gpdc/fixed |
|---|---|---|---|---|---|---|---|---|
| band_alpha | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_alpha | 0 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_alpha | 10 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_alpha | 10 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_beta | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_beta | 0 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_beta | 10 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_beta | 10 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 0 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 10 | 60 | 1.000 | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 10 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| delay_5 | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| delay_5 | 0 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| delay_5 | 10 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| delay_5 | 10 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| feedback_loop | 0 | 60 | 0.978 | 0.926 | 0.959 | 0.937 | 0.963 | 0.937 |
| feedback_loop | 0 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| feedback_loop | 10 | 60 | 1.000 | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 |
| feedback_loop | 10 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Stage: sensor.

| scenario | SNR (dB) | length (s) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gpdc/bic | gpdc/fixed |
|---|---|---|---|---|---|---|---|---|
| band_alpha | 0 | 60 | 0.800 | 0.960 | 0.940 | 1.000 | 0.940 | 1.000 |
| band_alpha | 0 | 240 | 0.980 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_alpha | 10 | 60 | 0.840 | 0.880 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_alpha | 10 | 240 | 0.960 | 0.960 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_beta | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_beta | 0 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_beta | 10 | 60 | 0.920 | 0.940 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_beta | 10 | 240 | 0.980 | 0.960 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 0 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 10 | 60 | 0.863 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 10 | 240 | 0.875 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 |
| delay_5 | 0 | 60 | 0.980 | 0.980 | 1.000 | 1.000 | 0.980 | 0.980 |
| delay_5 | 0 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| delay_5 | 10 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| delay_5 | 10 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| feedback_loop | 0 | 60 | 0.500 | 0.504 | 0.470 | 0.485 | 0.463 | 0.474 |
| feedback_loop | 0 | 240 | 0.526 | 0.533 | 0.485 | 0.485 | 0.474 | 0.485 |
| feedback_loop | 10 | 60 | 0.496 | 0.485 | 0.470 | 0.474 | 0.456 | 0.456 |
| feedback_loop | 10 | 240 | 0.493 | 0.489 | 0.467 | 0.463 | 0.452 | 0.448 |

Stage: source.

| scenario | SNR (dB) | length (s) | dtf/bic | dtf/fixed | gc/bic | gc/fixed | gpdc/bic | gpdc/fixed |
|---|---|---|---|---|---|---|---|---|
| band_alpha | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_alpha | 0 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_alpha | 10 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_alpha | 10 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_beta | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_beta | 0 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_beta | 10 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| band_beta | 10 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 0 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 10 | 60 | 0.975 | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | 10 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| delay_5 | 0 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| delay_5 | 0 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| delay_5 | 10 | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| delay_5 | 10 | 240 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| feedback_loop | 0 | 60 | 0.756 | 0.770 | 0.830 | 0.837 | 0.830 | 0.819 |
| feedback_loop | 0 | 240 | 0.781 | 0.781 | 0.844 | 0.867 | 0.852 | 0.848 |
| feedback_loop | 10 | 60 | 0.674 | 0.674 | 0.733 | 0.741 | 0.722 | 0.737 |
| feedback_loop | 10 | 240 | 0.674 | 0.663 | 0.770 | 0.767 | 0.756 | 0.744 |

Fixed order minus BIC order in `order`, per scenario, method and stage:

| scenario | method | stage | mean difference | sd | pairs |
|---|---|---|---|---|---|
| band_alpha | dtf | roi | +0.000 | 0.000 | 40 |
| band_alpha | dtf | sensor | +0.055 | 0.101 | 40 |
| band_alpha | dtf | source | +0.000 | 0.000 | 40 |
| band_alpha | gc | roi | +0.000 | 0.000 | 40 |
| band_alpha | gc | sensor | +0.015 | 0.053 | 40 |
| band_alpha | gc | source | +0.000 | 0.000 | 40 |
| band_alpha | gpdc | roi | +0.000 | 0.000 | 40 |
| band_alpha | gpdc | sensor | +0.015 | 0.053 | 40 |
| band_alpha | gpdc | source | +0.000 | 0.000 | 40 |
| band_beta | dtf | roi | +0.000 | 0.000 | 40 |
| band_beta | dtf | sensor | +0.000 | 0.064 | 40 |
| band_beta | dtf | source | +0.000 | 0.000 | 40 |
| band_beta | gc | roi | +0.000 | 0.000 | 40 |
| band_beta | gc | sensor | +0.000 | 0.000 | 40 |
| band_beta | gc | source | +0.000 | 0.000 | 40 |
| band_beta | gpdc | roi | +0.000 | 0.000 | 40 |
| band_beta | gpdc | sensor | +0.000 | 0.000 | 40 |
| band_beta | gpdc | source | +0.000 | 0.000 | 40 |
| chain | dtf | roi | -0.006 | 0.028 | 40 |
| chain | dtf | sensor | +0.003 | 0.020 | 40 |
| chain | dtf | source | +0.000 | 0.000 | 40 |
| chain | gc | roi | +0.000 | 0.000 | 40 |
| chain | gc | sensor | +0.000 | 0.000 | 40 |
| chain | gc | source | +0.000 | 0.000 | 40 |
| chain | gpdc | roi | +0.000 | 0.000 | 40 |
| chain | gpdc | sensor | +0.000 | 0.000 | 40 |
| chain | gpdc | source | +0.000 | 0.000 | 40 |
| delay_5 | dtf | roi | +0.000 | 0.000 | 40 |
| delay_5 | dtf | sensor | +0.000 | 0.000 | 40 |
| delay_5 | dtf | source | +0.000 | 0.000 | 40 |
| delay_5 | gc | roi | +0.000 | 0.000 | 40 |
| delay_5 | gc | sensor | +0.000 | 0.000 | 40 |
| delay_5 | gc | source | +0.000 | 0.000 | 40 |
| delay_5 | gpdc | roi | +0.000 | 0.000 | 40 |
| delay_5 | gpdc | sensor | +0.000 | 0.000 | 40 |
| delay_5 | gpdc | source | +0.000 | 0.000 | 40 |
| feedback_loop | dtf | roi | -0.014 | 0.048 | 40 |
| feedback_loop | dtf | sensor | -0.001 | 0.027 | 40 |
| feedback_loop | dtf | source | +0.001 | 0.032 | 40 |
| feedback_loop | gc | roi | -0.006 | 0.045 | 40 |
| feedback_loop | gc | sensor | +0.004 | 0.023 | 40 |
| feedback_loop | gc | source | +0.008 | 0.043 | 40 |
| feedback_loop | gpdc | roi | -0.006 | 0.041 | 40 |
| feedback_loop | gpdc | sensor | +0.005 | 0.027 | 40 |
| feedback_loop | gpdc | source | -0.003 | 0.036 | 40 |

## Inverse methods

Protocol `inverse`: minimum norm, LCMV, dSPM and eLORETA at stage 3 (columns: estimator and inverse method). The region and sensor stages do not depend on the inverse method.

Stage: source.

| scenario | modality | SNR (dB) | gc/bic@dspm | gc/bic@eloreta | gc/bic@lcmv | gc/bic@mne | gpdc/bic@dspm | gpdc/bic@eloreta | gpdc/bic@lcmv | gpdc/bic@mne |
|---|---|---|---|---|---|---|---|---|---|---|
| chain | eeg | 0 | 0.938 | 1.000 | 0.625 | 1.000 | 0.938 | 1.000 | 0.656 | 1.000 |
| chain | eeg | 10 | 0.875 | 1.000 | 0.500 | 1.000 | 0.875 | 1.000 | 0.469 | 1.000 |
| chain | grad | 0 | 1.000 | 1.000 | 0.812 | 1.000 | 1.000 | 1.000 | 0.781 | 1.000 |
| chain | grad | 10 | 1.000 | 1.000 | 0.344 | 1.000 | 1.000 | 1.000 | 0.344 | 1.000 |
| random_5 | eeg | 0 | 0.612 | 0.617 | 0.547 | 0.576 | 0.534 | 0.549 | 0.523 | 0.529 |
| random_5 | eeg | 10 | 0.643 | 0.727 | 0.372 | 0.703 | 0.581 | 0.654 | 0.388 | 0.641 |
| random_5 | grad | 0 | 0.776 | 0.797 | 0.437 | 0.784 | 0.781 | 0.815 | 0.456 | 0.786 |
| random_5 | grad | 10 | 0.688 | 0.766 | 0.479 | 0.776 | 0.724 | 0.799 | 0.456 | 0.781 |

## Leakage: the neighbour


### Neighbour leakage (leakage_adjacent)

Z -> X is the only edge; Y is X's neighbour. Relative scores are the score of an edge into Y over the score of Z -> X (mean over seeds, [min, max]).

| estimator | modality | SNR | stage | n | AUROC | direction | Z -> Y | X -> Y |
|---|---|---|---|---|---|---|---|---|
| dtf/bic | eeg | 10 | roi | 10 | 1.000 | 1.000 | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] |
| dtf/bic | eeg | 10 | sensor | 10 | 1.000 | 1.000 | 0.71 [0.67, 0.74] | 0.62 [0.54, 0.70] |
| dtf/bic | eeg | 10 | source | 10 | 0.400 | 1.000 | 1.78 [1.60, 1.95] | 2.58 [2.32, 2.94] |
| dtf/bic | grad | 10 | roi | 10 | 1.000 | 1.000 | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] |
| dtf/bic | grad | 10 | sensor | 10 | 1.000 | 1.000 | 0.34 [0.30, 0.41] | 0.65 [0.55, 0.71] |
| dtf/bic | grad | 10 | source | 10 | 1.000 | 1.000 | 0.29 [0.23, 0.32] | 0.67 [0.55, 0.86] |
| gc/bic | eeg | 10 | roi | 10 | 1.000 | 1.000 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| gc/bic | eeg | 10 | sensor | 10 | 1.000 | 1.000 | 0.63 [0.60, 0.66] | 0.31 [0.26, 0.35] |
| gc/bic | eeg | 10 | source | 10 | 0.800 | 1.000 | 1.40 [1.36, 1.48] | 0.62 [0.53, 0.70] |
| gc/bic | grad | 10 | roi | 10 | 1.000 | 1.000 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| gc/bic | grad | 10 | sensor | 10 | 1.000 | 1.000 | 0.33 [0.29, 0.37] | 0.58 [0.46, 0.66] |
| gc/bic | grad | 10 | source | 10 | 1.000 | 1.000 | 0.38 [0.28, 0.44] | 0.55 [0.49, 0.62] |
| gc_pairwise/bic | eeg | 10 | roi | 10 | 1.000 | 1.000 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| gc_pairwise/bic | eeg | 10 | sensor | 10 | 1.000 | 1.000 | 0.48 [0.44, 0.54] | 0.14 [0.13, 0.16] |
| gc_pairwise/bic | eeg | 10 | source | 10 | 0.800 | 1.000 | 2.07 [1.97, 2.22] | 0.43 [0.37, 0.51] |
| gc_pairwise/bic | grad | 10 | roi | 10 | 1.000 | 1.000 | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] |
| gc_pairwise/bic | grad | 10 | sensor | 10 | 1.000 | 1.000 | 0.13 [0.09, 0.16] | 0.39 [0.28, 0.45] |
| gc_pairwise/bic | grad | 10 | source | 10 | 1.000 | 1.000 | 0.16 [0.09, 0.20] | 0.33 [0.29, 0.39] |
| gpdc/bic | eeg | 10 | roi | 10 | 1.000 | 1.000 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |
| gpdc/bic | eeg | 10 | sensor | 10 | 0.880 | 1.000 | 0.59 [0.56, 0.63] | 1.00 [0.93, 1.08] |
| gpdc/bic | eeg | 10 | source | 10 | 0.400 | 1.000 | 1.44 [1.39, 1.52] | 2.94 [2.71, 3.12] |
| gpdc/bic | grad | 10 | roi | 10 | 1.000 | 1.000 | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] |
| gpdc/bic | grad | 10 | sensor | 10 | 1.000 | 1.000 | 0.30 [0.26, 0.33] | 0.75 [0.64, 0.80] |
| gpdc/bic | grad | 10 | source | 10 | 0.860 | 1.000 | 0.34 [0.25, 0.39] | 1.03 [0.83, 1.27] |
| pdc/bic | eeg | 10 | roi | 10 | 1.000 | 1.000 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |
| pdc/bic | eeg | 10 | sensor | 10 | 0.800 | 1.000 | 0.68 [0.66, 0.70] | 1.08 [1.00, 1.15] |
| pdc/bic | eeg | 10 | source | 10 | 0.400 | 1.000 | 1.45 [1.40, 1.49] | 2.48 [2.26, 2.63] |
| pdc/bic | grad | 10 | roi | 10 | 1.000 | 1.000 | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] |
| pdc/bic | grad | 10 | sensor | 10 | 1.000 | 1.000 | 0.37 [0.33, 0.40] | 0.83 [0.71, 0.89] |
| pdc/bic | grad | 10 | source | 10 | 0.980 | 1.000 | 0.35 [0.25, 0.40] | 0.91 [0.72, 1.15] |

Source minus region level, paired by seed (mean, sd over seeds):

| estimator | modality | SNR | n | AUROC | direction | Z -> Y | X -> Y |
|---|---|---|---|---|---|---|---|
| dtf/bic | eeg | 10 | 10 | -0.60 (0.00) | 0.00 (0.00) | 1.77 (0.12) | 2.58 (0.18) |
| dtf/bic | grad | 10 | 10 | 0.00 (0.00) | 0.00 (0.00) | 0.29 (0.03) | 0.67 (0.09) |
| gc/bic | eeg | 10 | 10 | -0.20 (0.00) | 0.00 (0.00) | 1.40 (0.04) | 0.62 (0.05) |
| gc/bic | grad | 10 | 10 | 0.00 (0.00) | 0.00 (0.00) | 0.38 (0.05) | 0.55 (0.05) |
| gc_pairwise/bic | eeg | 10 | 10 | -0.20 (0.00) | 0.00 (0.00) | 2.07 (0.08) | 0.43 (0.05) |
| gc_pairwise/bic | grad | 10 | 10 | 0.00 (0.00) | 0.00 (0.00) | 0.16 (0.03) | 0.33 (0.03) |
| gpdc/bic | eeg | 10 | 10 | -0.60 (0.00) | 0.00 (0.00) | 1.44 (0.04) | 2.94 (0.13) |
| gpdc/bic | grad | 10 | 10 | -0.14 (0.10) | 0.00 (0.00) | 0.34 (0.05) | 1.03 (0.12) |
| pdc/bic | eeg | 10 | 10 | -0.60 (0.00) | 0.00 (0.00) | 1.45 (0.03) | 2.48 (0.11) |
| pdc/bic | grad | 10 | 10 | -0.02 (0.06) | 0.00 (0.00) | 0.34 (0.05) | 0.91 (0.12) |

## Leakage: pairs against crosstalk


### Leakage and crosstalk

Region pairs without any edge. Spearman correlation between the score of a false edge x -> y and the leak of x into the series of y (the crosstalk matrix of the header), over the cells of all pairs; 95% percentile bootstrap over pairs. The region-level stage has no leakage and is the control.

| estimator | modality | stage | pairs | cells | Spearman | 95% interval |
|---|---|---|---|---|---|---|
| gc/bic | eeg | roi | 45 | 90 | 0.05 | [-0.16, 0.23] |
| gc/bic | eeg | sensor | 45 | 90 | 0.44 | [0.23, 0.61] |
| gc/bic | eeg | source | 45 | 90 | 0.62 | [0.47, 0.74] |
| gc/bic | grad | roi | 45 | 90 | 0.11 | [-0.13, 0.30] |
| gc/bic | grad | sensor | 45 | 90 | 0.51 | [0.33, 0.66] |
| gc/bic | grad | source | 45 | 90 | 0.68 | [0.53, 0.81] |
| gpdc/bic | eeg | roi | 45 | 90 | 0.05 | [-0.15, 0.24] |
| gpdc/bic | eeg | sensor | 45 | 90 | 0.45 | [0.26, 0.59] |
| gpdc/bic | eeg | source | 45 | 90 | 0.62 | [0.46, 0.74] |
| gpdc/bic | grad | roi | 45 | 90 | 0.04 | [-0.19, 0.27] |
| gpdc/bic | grad | sensor | 45 | 90 | 0.52 | [0.34, 0.66] |
| gpdc/bic | grad | source | 45 | 90 | 0.66 | [0.52, 0.79] |

## Significance tests


### Significance study

Datasets are simulated at region level (node activity plus white observation noise). A cell is *null* where the scenario has no edge. Rates are over cells; the intervals are bootstrapped over datasets (the cells of a dataset share its data), a share of datasets has a Wilson interval. A surrogate test with 39 surrogates has p-values in steps of 1/40, so a level below 0.025 cannot be reached (n/a).

BH is over the six cells of a dataset, q = 0.05.

False positives, cells without an edge:

| scenario | coupling | SNR | s | estimator | test | datasets | null cells | rate at 0.01 | rate at 0.05 | rate at 0.1 | mean p (expected) | datasets with a BH false rejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | 0 | 10 | 30 | gc/fixed | analytic | 25 | 150 | 0.013 [0.000, 0.033] | 0.053 [0.027, 0.087] | 0.127 [0.087, 0.167] | 0.453 (0.500) | 0.080 [0.022, 0.250] |
| chain | 0 | 10 | 30 | gc/fixed | surrogate:phase_randomize | 25 | 150 | n/a | 0.053 [0.020, 0.093] | 0.100 [0.060, 0.147] | 0.469 (0.512) | 0.000 [0.000, 0.133] |
| chain | 0 | 40 | 30 | gc/fixed | analytic | 25 | 150 | 0.007 [0.000, 0.020] | 0.047 [0.013, 0.087] | 0.100 [0.047, 0.153] | 0.513 (0.500) | 0.000 [0.000, 0.133] |
| chain | 0 | 40 | 30 | gc/fixed | surrogate:phase_randomize | 25 | 150 | n/a | 0.053 [0.027, 0.087] | 0.093 [0.053, 0.140] | 0.530 (0.512) | 0.000 [0.000, 0.133] |
| chain | 0.4 | 10 | 30 | gc/fixed | analytic | 25 | 100 | 0.050 [0.010, 0.090] | 0.150 [0.100, 0.200] | 0.230 [0.160, 0.310] | 0.354 (0.500) | 0.400 [0.234, 0.593] |
| chain | 0.4 | 10 | 30 | gc/fixed | surrogate:phase_randomize | 25 | 100 | n/a | 0.060 [0.020, 0.100] | 0.110 [0.060, 0.170] | 0.498 (0.512) | 0.000 [0.000, 0.133] |
| chain | 0.4 | 40 | 30 | gc/fixed | analytic | 25 | 100 | 0.010 [0.000, 0.030] | 0.030 [0.000, 0.060] | 0.090 [0.040, 0.150] | 0.533 (0.500) | 0.040 [0.007, 0.195] |
| chain | 0.4 | 40 | 30 | gc/fixed | surrogate:phase_randomize | 25 | 100 | n/a | 0.010 [0.000, 0.030] | 0.020 [0.000, 0.050] | 0.699 (0.512) | 0.000 [0.000, 0.133] |

Power, cells with a true edge:

| scenario | coupling | SNR | s | estimator | test | datasets | edge cells | power at 0.01 | power at 0.05 | power at 0.1 | BH power | BH false-discovery proportion | cells needed for BH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | 0.4 | 10 | 30 | gc/fixed | analytic | 25 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.140 | 1 |
| chain | 0.4 | 10 | 30 | gc/fixed | surrogate:phase_randomize | 25 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 3 |
| chain | 0.4 | 40 | 30 | gc/fixed | analytic | 25 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.013 | 1 |
| chain | 0.4 | 40 | 30 | gc/fixed | surrogate:phase_randomize | 25 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 | 3 |

## Noise: simulated, white and empty room


### Noise study

Every seed has up to three arms with the same sources: the simulated noise (1/f background sources plus white noise), white noise only (no background) and the real empty-room recording of the sample dataset (no background), all scaled to the same sensor SNR. Mean AUROC over seeds, and the difference between arms taken seed by seed (mean, sd, number of seeds). The empty-room windows are blocks of one 110 s recording and are reused by several seeds, so the noise samples are not independent even where the seeds are. The region-level stage does not see the sensor noise and is the control.

| modality | SNR | estimator | stage | simulated | white | empty room | white - simulated | empty room - white | empty room - simulated |
|---|---|---|---|---|---|---|---|---|---|
| grad | -10 | gc/bic | roi | 0.819 (n=9) | 0.819 (n=9) | 0.819 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| grad | -10 | gc/bic | sensor | 1.000 (n=9) | 1.000 (n=9) | 1.000 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| grad | -10 | gc/bic | source | 1.000 (n=9) | 1.000 (n=9) | 1.000 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| grad | -10 | gpdc/bic | roi | 0.819 (n=9) | 0.819 (n=9) | 0.819 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| grad | -10 | gpdc/bic | sensor | 1.000 (n=9) | 1.000 (n=9) | 1.000 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| grad | -10 | gpdc/bic | source | 1.000 (n=9) | 1.000 (n=9) | 1.000 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| mag | -10 | gc/bic | roi | 0.806 (n=9) | 0.806 (n=9) | 0.806 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| mag | -10 | gc/bic | sensor | 0.861 (n=9) | 0.708 (n=9) | 0.514 (n=9) | -0.153 (sd 0.205, n=9) | -0.194 (sd 0.178, n=9) | -0.347 (sd 0.150, n=9) |
| mag | -10 | gc/bic | source | 1.000 (n=9) | 1.000 (n=9) | 1.000 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| mag | -10 | gpdc/bic | roi | 0.806 (n=9) | 0.806 (n=9) | 0.806 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| mag | -10 | gpdc/bic | sensor | 0.875 (n=9) | 0.708 (n=9) | 0.472 (n=9) | -0.167 (sd 0.217, n=9) | -0.236 (sd 0.159, n=9) | -0.403 (sd 0.185, n=9) |
| mag | -10 | gpdc/bic | source | 1.000 (n=9) | 1.000 (n=9) | 1.000 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |

## Noise: harder SNR (add-on)


### Noise study

Every seed has up to three arms with the same sources: the simulated noise (1/f background sources plus white noise), white noise only (no background) and the real empty-room recording of the sample dataset (no background), all scaled to the same sensor SNR. Mean AUROC over seeds, and the difference between arms taken seed by seed (mean, sd, number of seeds). The empty-room windows are blocks of one 110 s recording and are reused by several seeds, so the noise samples are not independent even where the seeds are. The region-level stage does not see the sensor noise and is the control.

| modality | SNR | estimator | stage | simulated | white | empty room | white - simulated | empty room - white | empty room - simulated |
|---|---|---|---|---|---|---|---|---|---|
| grad | -15 | gc/bic | roi | 0.514 (n=9) | 0.514 (n=9) | 0.514 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| grad | -15 | gc/bic | sensor | 0.958 (n=9) | 0.889 (n=9) | 1.000 (n=9) | -0.069 (sd 0.127, n=9) | +0.111 (sd 0.098, n=9) | +0.042 (sd 0.088, n=9) |
| grad | -15 | gc/bic | source | 0.986 (n=9) | 0.903 (n=9) | 1.000 (n=9) | -0.083 (sd 0.153, n=9) | +0.097 (sd 0.137, n=9) | +0.014 (sd 0.042, n=9) |
| grad | -15 | gpdc/bic | roi | 0.514 (n=9) | 0.514 (n=9) | 0.514 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| grad | -15 | gpdc/bic | sensor | 0.986 (n=9) | 0.889 (n=9) | 1.000 (n=9) | -0.097 (sd 0.104, n=9) | +0.111 (sd 0.098, n=9) | +0.014 (sd 0.042, n=9) |
| grad | -15 | gpdc/bic | source | 0.986 (n=9) | 0.903 (n=9) | 1.000 (n=9) | -0.083 (sd 0.153, n=9) | +0.097 (sd 0.137, n=9) | +0.014 (sd 0.042, n=9) |
| grad | -20 | gc/bic | roi | 0.444 (n=9) | 0.444 (n=9) | 0.444 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| grad | -20 | gc/bic | sensor | 0.681 (n=9) | 0.583 (n=9) | 0.986 (n=9) | -0.097 (sd 0.278, n=9) | +0.403 (sd 0.317, n=9) | +0.306 (sd 0.226, n=9) |
| grad | -20 | gc/bic | source | 0.694 (n=9) | 0.514 (n=9) | 0.764 (n=9) | -0.181 (sd 0.235, n=9) | +0.250 (sd 0.207, n=9) | +0.069 (sd 0.243, n=9) |
| grad | -20 | gpdc/bic | roi | 0.444 (n=9) | 0.444 (n=9) | 0.444 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| grad | -20 | gpdc/bic | sensor | 0.722 (n=9) | 0.583 (n=9) | 0.986 (n=9) | -0.139 (sd 0.289, n=9) | +0.403 (sd 0.317, n=9) | +0.264 (sd 0.276, n=9) |
| grad | -20 | gpdc/bic | source | 0.667 (n=9) | 0.500 (n=9) | 0.792 (n=9) | -0.167 (sd 0.300, n=9) | +0.292 (sd 0.177, n=9) | +0.125 (sd 0.280, n=9) |
| mag | -15 | gc/bic | roi | 0.639 (n=9) | 0.639 (n=9) | 0.639 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| mag | -15 | gc/bic | sensor | 0.708 (n=9) | 0.667 (n=9) | 0.486 (n=9) | -0.042 (sd 0.207, n=9) | -0.181 (sd 0.266, n=9) | -0.222 (sd 0.174, n=9) |
| mag | -15 | gc/bic | source | 0.972 (n=9) | 0.833 (n=9) | 0.903 (n=9) | -0.139 (sd 0.202, n=9) | +0.069 (sd 0.273, n=9) | -0.069 (sd 0.141, n=9) |
| mag | -15 | gpdc/bic | roi | 0.639 (n=9) | 0.639 (n=9) | 0.639 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| mag | -15 | gpdc/bic | sensor | 0.750 (n=9) | 0.667 (n=9) | 0.597 (n=9) | -0.083 (sd 0.225, n=9) | -0.069 (sd 0.259, n=9) | -0.153 (sd 0.104, n=9) |
| mag | -15 | gpdc/bic | source | 0.972 (n=9) | 0.833 (n=9) | 0.847 (n=9) | -0.139 (sd 0.202, n=9) | +0.014 (sd 0.315, n=9) | -0.125 (sd 0.207, n=9) |
| mag | -20 | gc/bic | roi | 0.361 (n=9) | 0.361 (n=9) | 0.361 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| mag | -20 | gc/bic | sensor | 0.667 (n=9) | 0.597 (n=9) | 0.472 (n=9) | -0.069 (sd 0.434, n=9) | -0.125 (sd 0.405, n=9) | -0.194 (sd 0.167, n=9) |
| mag | -20 | gc/bic | source | 0.583 (n=9) | 0.667 (n=9) | 0.625 (n=9) | +0.083 (sd 0.331, n=9) | -0.042 (sd 0.286, n=9) | +0.042 (sd 0.265, n=9) |
| mag | -20 | gpdc/bic | roi | 0.361 (n=9) | 0.361 (n=9) | 0.361 (n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) | +0.000 (sd 0.000, n=9) |
| mag | -20 | gpdc/bic | sensor | 0.667 (n=9) | 0.597 (n=9) | 0.653 (n=9) | -0.069 (sd 0.415, n=9) | +0.056 (sd 0.410, n=9) | -0.014 (sd 0.159, n=9) |
| mag | -20 | gpdc/bic | source | 0.583 (n=9) | 0.667 (n=9) | 0.597 (n=9) | +0.083 (sd 0.319, n=9) | -0.069 (sd 0.349, n=9) | +0.014 (sd 0.321, n=9) |

## Significance tests: power and weak coupling (add-on)


### Significance study

Datasets are simulated at region level (node activity plus white observation noise). A cell is *null* where the scenario has no edge. Rates are over cells; the intervals are bootstrapped over datasets (the cells of a dataset share its data), a share of datasets has a Wilson interval. A surrogate test with 99 surrogates has p-values in steps of 1/100, so a level below 0.010 cannot be reached (n/a).

BH is over the six cells of a dataset, q = 0.05.

False positives, cells without an edge:

| scenario | coupling | SNR | s | estimator | test | datasets | null cells | rate at 0.01 | rate at 0.05 | rate at 0.1 | mean p (expected) | datasets with a BH false rejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | 0 | 10 | 30 | gc/fixed | analytic | 40 | 240 | 0.008 [0.000, 0.021] | 0.062 [0.033, 0.092] | 0.129 [0.096, 0.167] | 0.480 (0.500) | 0.050 [0.014, 0.165] |
| chain | 0 | 10 | 30 | gc/fixed | surrogate:phase_randomize | 40 | 240 | 0.008 [0.000, 0.021] | 0.058 [0.033, 0.083] | 0.117 [0.083, 0.150] | 0.486 (0.505) | 0.000 [0.000, 0.088] |
| chain | 0 | 40 | 30 | gc/fixed | analytic | 40 | 240 | 0.008 [0.000, 0.021] | 0.050 [0.025, 0.083] | 0.104 [0.067, 0.146] | 0.505 (0.500) | 0.025 [0.004, 0.129] |
| chain | 0 | 40 | 30 | gc/fixed | surrogate:phase_randomize | 40 | 240 | 0.004 [0.000, 0.013] | 0.054 [0.029, 0.083] | 0.117 [0.083, 0.154] | 0.508 (0.505) | 0.000 [0.000, 0.088] |
| chain | 0.1 | 10 | 30 | gc/fixed | analytic | 40 | 160 | 0.006 [0.000, 0.019] | 0.075 [0.031, 0.125] | 0.106 [0.056, 0.163] | 0.442 (0.500) | 0.100 [0.040, 0.231] |
| chain | 0.1 | 10 | 30 | gc/fixed | surrogate:phase_randomize | 40 | 160 | 0.006 [0.000, 0.019] | 0.050 [0.019, 0.087] | 0.075 [0.037, 0.119] | 0.466 (0.505) | 0.075 [0.026, 0.199] |
| chain | 0.1 | 40 | 30 | gc/fixed | analytic | 40 | 160 | 0.006 [0.000, 0.019] | 0.044 [0.019, 0.075] | 0.119 [0.069, 0.163] | 0.521 (0.500) | 0.050 [0.014, 0.165] |
| chain | 0.1 | 40 | 30 | gc/fixed | surrogate:phase_randomize | 40 | 160 | 0.000 [0.000, 0.000] | 0.044 [0.019, 0.075] | 0.100 [0.056, 0.150] | 0.549 (0.505) | 0.025 [0.004, 0.129] |
| chain | 0.2 | 10 | 30 | gc/fixed | analytic | 40 | 160 | 0.000 [0.000, 0.000] | 0.050 [0.019, 0.081] | 0.094 [0.056, 0.144] | 0.478 (0.500) | 0.050 [0.014, 0.165] |
| chain | 0.2 | 10 | 30 | gc/fixed | surrogate:phase_randomize | 40 | 160 | 0.000 [0.000, 0.000] | 0.019 [0.000, 0.037] | 0.087 [0.044, 0.138] | 0.528 (0.505) | 0.025 [0.004, 0.129] |
| chain | 0.2 | 40 | 30 | gc/fixed | analytic | 40 | 160 | 0.013 [0.000, 0.031] | 0.062 [0.031, 0.100] | 0.106 [0.056, 0.169] | 0.480 (0.500) | 0.075 [0.026, 0.199] |
| chain | 0.2 | 40 | 30 | gc/fixed | surrogate:phase_randomize | 40 | 160 | 0.006 [0.000, 0.019] | 0.037 [0.013, 0.069] | 0.069 [0.037, 0.113] | 0.542 (0.505) | 0.025 [0.004, 0.129] |
| chain | 0.4 | 10 | 30 | gc/fixed | analytic | 40 | 160 | 0.056 [0.025, 0.087] | 0.181 [0.131, 0.231] | 0.244 [0.181, 0.306] | 0.335 (0.500) | 0.475 [0.329, 0.625] |
| chain | 0.4 | 10 | 30 | gc/fixed | surrogate:phase_randomize | 40 | 160 | 0.031 [0.006, 0.056] | 0.069 [0.031, 0.113] | 0.131 [0.087, 0.181] | 0.479 (0.505) | 0.150 [0.071, 0.291] |
| chain | 0.4 | 40 | 30 | gc/fixed | analytic | 40 | 160 | 0.025 [0.006, 0.050] | 0.050 [0.019, 0.081] | 0.119 [0.075, 0.169] | 0.494 (0.500) | 0.100 [0.040, 0.231] |
| chain | 0.4 | 40 | 30 | gc/fixed | surrogate:phase_randomize | 40 | 160 | 0.006 [0.000, 0.019] | 0.019 [0.000, 0.038] | 0.037 [0.013, 0.069] | 0.658 (0.505) | 0.025 [0.004, 0.129] |

Power, cells with a true edge:

| scenario | coupling | SNR | s | estimator | test | datasets | edge cells | power at 0.01 | power at 0.05 | power at 0.1 | BH power | BH false-discovery proportion | cells needed for BH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | 0.1 | 10 | 30 | gc/fixed | analytic | 40 | 80 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.042 | 1 |
| chain | 0.1 | 10 | 30 | gc/fixed | surrogate:phase_randomize | 40 | 80 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.029 | 2 |
| chain | 0.1 | 40 | 30 | gc/fixed | analytic | 40 | 80 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.017 | 1 |
| chain | 0.1 | 40 | 30 | gc/fixed | surrogate:phase_randomize | 40 | 80 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.008 | 2 |
| chain | 0.2 | 10 | 30 | gc/fixed | analytic | 40 | 80 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.017 | 1 |
| chain | 0.2 | 10 | 30 | gc/fixed | surrogate:phase_randomize | 40 | 80 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.008 | 2 |
| chain | 0.2 | 40 | 30 | gc/fixed | analytic | 40 | 80 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.025 | 1 |
| chain | 0.2 | 40 | 30 | gc/fixed | surrogate:phase_randomize | 40 | 80 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.008 | 2 |
| chain | 0.4 | 10 | 30 | gc/fixed | analytic | 40 | 80 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.175 | 1 |
| chain | 0.4 | 10 | 30 | gc/fixed | surrogate:phase_randomize | 40 | 80 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.050 | 2 |
| chain | 0.4 | 40 | 30 | gc/fixed | analytic | 40 | 80 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.033 | 1 |
| chain | 0.4 | 40 | 30 | gc/fixed | surrogate:phase_randomize | 40 | 80 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.008 | 2 |

## Asymptotic PDC and rPDC tests: calibration (add-on)


### Significance study

Datasets are simulated at region level (node activity plus white observation noise). A cell is *null* where the scenario has no edge. Rates are over cells; the intervals are bootstrapped over datasets (the cells of a dataset share its data), a share of datasets has a Wilson interval. A surrogate test with 39 surrogates has p-values in steps of 1/40, so a level below 0.025 cannot be reached (n/a).


### Frequency-resolved tests

One p-value per pair and frequency. *Pointwise* rates are over pairs and the frequencies of the band (`all`: the whole grid); *after BH* a pair counts once if any frequency of the band survives a BH correction over those frequencies, q = 0.05. *Liberal*: the lower end of the interval of the pointwise rate at 0.05 is above 0.05.

False positives, pairs without an edge:

| scenario | coupling | SNR | s | estimator | test | datasets | band | freqs | null pairs | rate at 0.01 | rate at 0.05 | rate at 0.1 | liberal at 0.05 | pairs after BH | freqs needed for BH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | 0 | 10 | 120 | pdc/bic | analytic | 25 | all | 51 | 150 | 0.005 [0.000, 0.013] | 0.031 [0.019, 0.045] | 0.061 [0.042, 0.082] | no | 0.020 [0.000, 0.053] | 1 |
| chain | 0 | 10 | 120 | pdc/bic | analytic | 25 | alpha | 6 | 150 | 0.008 [0.000, 0.023] | 0.037 [0.011, 0.069] | 0.073 [0.040, 0.110] | no | 0.020 [0.000, 0.053] | 1 |
| chain | 0 | 10 | 120 | pdc/bic | analytic | 25 | beta | 18 | 150 | 0.000 [0.000, 0.000] | 0.009 [0.001, 0.019] | 0.043 [0.021, 0.066] | no | 0.000 [0.000, 0.000] | 1 |
| chain | 0 | 10 | 120 | pdc/bic | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.037 [0.022, 0.053] | 0.092 [0.067, 0.119] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0 | 10 | 120 | pdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.053 [0.020, 0.097] | 0.103 [0.064, 0.149] | no | 0.033 [0.007, 0.067] | 3 |
| chain | 0 | 10 | 120 | pdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.028 [0.014, 0.044] | 0.078 [0.050, 0.113] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0 | 10 | 120 | pdc/fixed | analytic | 25 | all | 51 | 150 | 0.004 [0.002, 0.006] | 0.027 [0.021, 0.034] | 0.077 [0.065, 0.089] | no | 0.000 [0.000, 0.000] | 1 |
| chain | 0 | 10 | 120 | pdc/fixed | analytic | 25 | alpha | 6 | 150 | 0.006 [0.000, 0.016] | 0.019 [0.003, 0.036] | 0.063 [0.036, 0.098] | no | 0.013 [0.000, 0.033] | 1 |
| chain | 0 | 10 | 120 | pdc/fixed | analytic | 25 | beta | 18 | 150 | 0.002 [0.000, 0.005] | 0.024 [0.015, 0.035] | 0.070 [0.055, 0.085] | no | 0.007 [0.000, 0.020] | 1 |
| chain | 0 | 10 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.051 [0.042, 0.061] | 0.107 [0.093, 0.123] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0 | 10 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.044 [0.024, 0.069] | 0.094 [0.059, 0.139] | no | 0.007 [0.000, 0.020] | 3 |
| chain | 0 | 10 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.051 [0.037, 0.065] | 0.108 [0.090, 0.125] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0 | 10 | 120 | pdc/low | analytic | 25 | all | 51 | 150 | 0.003 [0.000, 0.008] | 0.014 [0.005, 0.027] | 0.048 [0.028, 0.068] | no | 0.007 [0.000, 0.020] | 1 |
| chain | 0 | 10 | 120 | pdc/low | analytic | 25 | alpha | 6 | 150 | 0.000 [0.000, 0.000] | 0.009 [0.000, 0.024] | 0.030 [0.008, 0.057] | no | 0.007 [0.000, 0.020] | 1 |
| chain | 0 | 10 | 120 | pdc/low | analytic | 25 | beta | 18 | 150 | 0.000 [0.000, 0.001] | 0.009 [0.000, 0.023] | 0.030 [0.013, 0.050] | no | 0.007 [0.000, 0.020] | 1 |
| chain | 0 | 10 | 120 | pdc/low | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.047 [0.028, 0.065] | 0.105 [0.076, 0.135] | no | 0.007 [0.000, 0.020] | 26 |
| chain | 0 | 10 | 120 | pdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.049 [0.027, 0.073] | 0.108 [0.071, 0.147] | no | 0.027 [0.007, 0.053] | 3 |
| chain | 0 | 10 | 120 | pdc/low | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.040 [0.017, 0.068] | 0.102 [0.063, 0.139] | no | 0.007 [0.000, 0.020] | 9 |
| chain | 0 | 10 | 120 | rpdc/bic | analytic | 25 | all | 51 | 150 | 0.008 [0.000, 0.018] | 0.042 [0.024, 0.063] | 0.083 [0.059, 0.107] | no | 0.033 [0.000, 0.073] | 1 |
| chain | 0 | 10 | 120 | rpdc/bic | analytic | 25 | alpha | 6 | 150 | 0.013 [0.000, 0.040] | 0.064 [0.033, 0.099] | 0.107 [0.071, 0.146] | no | 0.060 [0.027, 0.100] | 1 |
| chain | 0 | 10 | 120 | rpdc/bic | analytic | 25 | beta | 18 | 150 | 0.008 [0.000, 0.018] | 0.033 [0.016, 0.055] | 0.073 [0.047, 0.098] | no | 0.020 [0.000, 0.040] | 1 |
| chain | 0 | 10 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.036 [0.019, 0.055] | 0.087 [0.062, 0.117] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0 | 10 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.066 [0.031, 0.108] | 0.104 [0.067, 0.146] | no | 0.033 [0.007, 0.073] | 3 |
| chain | 0 | 10 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.032 [0.015, 0.053] | 0.081 [0.050, 0.113] | no | 0.007 [0.000, 0.020] | 9 |
| chain | 0 | 10 | 120 | rpdc/fixed | analytic | 25 | all | 51 | 150 | 0.015 [0.011, 0.020] | 0.054 [0.045, 0.065] | 0.105 [0.092, 0.119] | no | 0.053 [0.020, 0.093] | 1 |
| chain | 0 | 10 | 120 | rpdc/fixed | analytic | 25 | alpha | 6 | 150 | 0.013 [0.002, 0.028] | 0.047 [0.023, 0.074] | 0.099 [0.064, 0.138] | no | 0.033 [0.007, 0.060] | 1 |
| chain | 0 | 10 | 120 | rpdc/fixed | analytic | 25 | beta | 18 | 150 | 0.012 [0.006, 0.019] | 0.053 [0.041, 0.068] | 0.104 [0.089, 0.120] | no | 0.040 [0.007, 0.080] | 1 |
| chain | 0 | 10 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.052 [0.042, 0.064] | 0.106 [0.093, 0.120] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0 | 10 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.047 [0.024, 0.074] | 0.091 [0.056, 0.130] | no | 0.007 [0.000, 0.020] | 3 |
| chain | 0 | 10 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.052 [0.039, 0.067] | 0.107 [0.091, 0.124] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0 | 10 | 120 | rpdc/low | analytic | 25 | all | 51 | 150 | 0.000 [0.000, 0.000] | 0.014 [0.001, 0.033] | 0.041 [0.015, 0.067] | no | 0.013 [0.000, 0.033] | 1 |
| chain | 0 | 10 | 120 | rpdc/low | analytic | 25 | alpha | 6 | 150 | 0.000 [0.000, 0.000] | 0.013 [0.000, 0.033] | 0.040 [0.013, 0.067] | no | 0.013 [0.000, 0.033] | 1 |
| chain | 0 | 10 | 120 | rpdc/low | analytic | 25 | beta | 18 | 150 | 0.000 [0.000, 0.000] | 0.013 [0.000, 0.033] | 0.040 [0.013, 0.067] | no | 0.013 [0.000, 0.033] | 1 |
| chain | 0 | 10 | 120 | rpdc/low | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.041 [0.014, 0.073] | 0.107 [0.067, 0.151] | no | 0.013 [0.000, 0.033] | 26 |
| chain | 0 | 10 | 120 | rpdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.040 [0.013, 0.080] | 0.107 [0.067, 0.147] | no | 0.013 [0.000, 0.033] | 3 |
| chain | 0 | 10 | 120 | rpdc/low | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.040 [0.007, 0.073] | 0.107 [0.067, 0.147] | no | 0.040 [0.013, 0.073] | 9 |
| chain | 0 | 10 | 30 | pdc/bic | analytic | 25 | all | 51 | 150 | 0.009 [0.001, 0.021] | 0.040 [0.018, 0.068] | 0.076 [0.049, 0.108] | no | 0.033 [0.007, 0.067] | 1 |
| chain | 0 | 10 | 30 | pdc/bic | analytic | 25 | alpha | 6 | 150 | 0.003 [0.000, 0.009] | 0.031 [0.007, 0.061] | 0.074 [0.041, 0.113] | no | 0.020 [0.000, 0.053] | 1 |
| chain | 0 | 10 | 30 | pdc/bic | analytic | 25 | beta | 18 | 150 | 0.003 [0.000, 0.007] | 0.029 [0.010, 0.048] | 0.072 [0.045, 0.101] | no | 0.007 [0.000, 0.020] | 1 |
| chain | 0 | 10 | 30 | pdc/bic | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.055 [0.026, 0.090] | 0.092 [0.058, 0.134] | no | 0.020 [0.000, 0.053] | 26 |
| chain | 0 | 10 | 30 | pdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.062 [0.031, 0.099] | 0.114 [0.076, 0.163] | no | 0.020 [0.000, 0.040] | 3 |
| chain | 0 | 10 | 30 | pdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.053 [0.024, 0.085] | 0.106 [0.070, 0.146] | no | 0.027 [0.000, 0.060] | 9 |
| chain | 0 | 10 | 30 | pdc/fixed | analytic | 25 | all | 51 | 150 | 0.002 [0.001, 0.004] | 0.027 [0.023, 0.032] | 0.078 [0.071, 0.084] | no | 0.000 [0.000, 0.000] | 1 |
| chain | 0 | 10 | 30 | pdc/fixed | analytic | 25 | alpha | 6 | 150 | 0.001 [0.000, 0.003] | 0.018 [0.007, 0.032] | 0.074 [0.056, 0.094] | no | 0.007 [0.000, 0.020] | 1 |
| chain | 0 | 10 | 30 | pdc/fixed | analytic | 25 | beta | 18 | 150 | 0.003 [0.000, 0.006] | 0.037 [0.026, 0.047] | 0.085 [0.068, 0.102] | no | 0.007 [0.000, 0.020] | 1 |
| chain | 0 | 10 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.054 [0.048, 0.060] | 0.109 [0.100, 0.117] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0 | 10 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.043 [0.027, 0.061] | 0.103 [0.076, 0.132] | no | 0.000 [0.000, 0.000] | 3 |
| chain | 0 | 10 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.066 [0.050, 0.081] | 0.117 [0.096, 0.136] | yes | 0.000 [0.000, 0.000] | 9 |
| chain | 0 | 10 | 30 | pdc/low | analytic | 25 | all | 51 | 150 | 0.003 [0.000, 0.006] | 0.021 [0.006, 0.038] | 0.070 [0.040, 0.102] | no | 0.013 [0.000, 0.033] | 1 |
| chain | 0 | 10 | 30 | pdc/low | analytic | 25 | alpha | 6 | 150 | 0.000 [0.000, 0.000] | 0.027 [0.007, 0.053] | 0.029 [0.007, 0.053] | no | 0.027 [0.007, 0.053] | 1 |
| chain | 0 | 10 | 30 | pdc/low | analytic | 25 | beta | 18 | 150 | 0.000 [0.000, 0.000] | 0.013 [0.002, 0.029] | 0.058 [0.027, 0.091] | no | 0.007 [0.000, 0.020] | 1 |
| chain | 0 | 10 | 30 | pdc/low | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.034 [0.016, 0.056] | 0.084 [0.060, 0.109] | no | 0.013 [0.000, 0.033] | 26 |
| chain | 0 | 10 | 30 | pdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.037 [0.013, 0.063] | 0.081 [0.047, 0.117] | no | 0.020 [0.000, 0.047] | 3 |
| chain | 0 | 10 | 30 | pdc/low | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.028 [0.007, 0.053] | 0.076 [0.045, 0.109] | no | 0.013 [0.000, 0.033] | 9 |
| chain | 0 | 10 | 30 | rpdc/bic | analytic | 25 | all | 51 | 150 | 0.019 [0.004, 0.040] | 0.068 [0.037, 0.100] | 0.112 [0.077, 0.152] | no | 0.053 [0.020, 0.093] | 1 |
| chain | 0 | 10 | 30 | rpdc/bic | analytic | 25 | alpha | 6 | 150 | 0.021 [0.003, 0.044] | 0.081 [0.042, 0.120] | 0.122 [0.081, 0.166] | no | 0.073 [0.040, 0.113] | 1 |
| chain | 0 | 10 | 30 | rpdc/bic | analytic | 25 | beta | 18 | 150 | 0.015 [0.003, 0.029] | 0.063 [0.035, 0.091] | 0.109 [0.069, 0.149] | no | 0.053 [0.027, 0.087] | 1 |
| chain | 0 | 10 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.052 [0.027, 0.081] | 0.106 [0.074, 0.138] | no | 0.027 [0.000, 0.060] | 26 |
| chain | 0 | 10 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.061 [0.032, 0.090] | 0.113 [0.076, 0.151] | no | 0.027 [0.007, 0.053] | 3 |
| chain | 0 | 10 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.054 [0.030, 0.082] | 0.101 [0.066, 0.142] | no | 0.033 [0.007, 0.060] | 9 |
| chain | 0 | 10 | 30 | rpdc/fixed | analytic | 25 | all | 51 | 150 | 0.013 [0.009, 0.016] | 0.055 [0.047, 0.062] | 0.109 [0.099, 0.118] | no | 0.053 [0.020, 0.087] | 1 |
| chain | 0 | 10 | 30 | rpdc/fixed | analytic | 25 | alpha | 6 | 150 | 0.010 [0.002, 0.019] | 0.056 [0.039, 0.073] | 0.117 [0.091, 0.141] | no | 0.033 [0.013, 0.060] | 1 |
| chain | 0 | 10 | 30 | rpdc/fixed | analytic | 25 | beta | 18 | 150 | 0.019 [0.012, 0.026] | 0.064 [0.050, 0.080] | 0.117 [0.099, 0.137] | no | 0.073 [0.040, 0.113] | 1 |
| chain | 0 | 10 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.054 [0.047, 0.061] | 0.111 [0.102, 0.119] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0 | 10 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.048 [0.029, 0.068] | 0.108 [0.081, 0.136] | no | 0.000 [0.000, 0.000] | 3 |
| chain | 0 | 10 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.067 [0.051, 0.082] | 0.117 [0.097, 0.137] | yes | 0.000 [0.000, 0.000] | 9 |
| chain | 0 | 10 | 30 | rpdc/low | analytic | 25 | all | 51 | 150 | 0.007 [0.000, 0.020] | 0.027 [0.007, 0.053] | 0.042 [0.015, 0.068] | no | 0.027 [0.007, 0.053] | 1 |
| chain | 0 | 10 | 30 | rpdc/low | analytic | 25 | alpha | 6 | 150 | 0.007 [0.000, 0.020] | 0.027 [0.007, 0.053] | 0.040 [0.013, 0.067] | no | 0.027 [0.007, 0.053] | 1 |
| chain | 0 | 10 | 30 | rpdc/low | analytic | 25 | beta | 18 | 150 | 0.007 [0.000, 0.020] | 0.027 [0.007, 0.053] | 0.040 [0.013, 0.067] | no | 0.027 [0.007, 0.053] | 1 |
| chain | 0 | 10 | 30 | rpdc/low | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.021 [0.001, 0.041] | 0.074 [0.035, 0.114] | no | 0.020 [0.000, 0.040] | 26 |
| chain | 0 | 10 | 30 | rpdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.020 [0.000, 0.047] | 0.073 [0.033, 0.113] | no | 0.020 [0.000, 0.040] | 3 |
| chain | 0 | 10 | 30 | rpdc/low | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.020 [0.000, 0.047] | 0.073 [0.033, 0.120] | no | 0.020 [0.000, 0.040] | 9 |
| chain | 0 | 40 | 120 | pdc/bic | analytic | 25 | all | 51 | 150 | 0.004 [0.000, 0.013] | 0.024 [0.008, 0.043] | 0.063 [0.030, 0.100] | no | 0.013 [0.000, 0.033] | 1 |
| chain | 0 | 40 | 120 | pdc/bic | analytic | 25 | alpha | 6 | 150 | 0.000 [0.000, 0.000] | 0.013 [0.000, 0.031] | 0.060 [0.030, 0.097] | no | 0.007 [0.000, 0.020] | 1 |
| chain | 0 | 40 | 120 | pdc/bic | analytic | 25 | beta | 18 | 150 | 0.005 [0.000, 0.016] | 0.021 [0.004, 0.041] | 0.059 [0.023, 0.103] | no | 0.013 [0.000, 0.033] | 1 |
| chain | 0 | 40 | 120 | pdc/bic | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.040 [0.018, 0.066] | 0.085 [0.045, 0.131] | no | 0.027 [0.007, 0.053] | 26 |
| chain | 0 | 40 | 120 | pdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.049 [0.023, 0.078] | 0.086 [0.042, 0.131] | no | 0.020 [0.000, 0.040] | 3 |
| chain | 0 | 40 | 120 | pdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.037 [0.013, 0.066] | 0.084 [0.040, 0.133] | no | 0.027 [0.007, 0.053] | 9 |
| chain | 0 | 40 | 120 | pdc/fixed | analytic | 25 | all | 51 | 150 | 0.002 [0.001, 0.003] | 0.024 [0.018, 0.030] | 0.066 [0.053, 0.077] | no | 0.000 [0.000, 0.000] | 1 |
| chain | 0 | 40 | 120 | pdc/fixed | analytic | 25 | alpha | 6 | 150 | 0.003 [0.000, 0.009] | 0.020 [0.007, 0.036] | 0.060 [0.038, 0.081] | no | 0.020 [0.000, 0.040] | 1 |
| chain | 0 | 40 | 120 | pdc/fixed | analytic | 25 | beta | 18 | 150 | 0.001 [0.000, 0.004] | 0.021 [0.015, 0.030] | 0.060 [0.044, 0.076] | no | 0.000 [0.000, 0.000] | 1 |
| chain | 0 | 40 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.048 [0.039, 0.059] | 0.099 [0.085, 0.113] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0 | 40 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.046 [0.026, 0.068] | 0.094 [0.068, 0.124] | no | 0.007 [0.000, 0.020] | 3 |
| chain | 0 | 40 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.052 [0.039, 0.067] | 0.097 [0.077, 0.115] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0 | 40 | 120 | pdc/low | analytic | 25 | all | 51 | 150 | 0.004 [0.000, 0.013] | 0.024 [0.008, 0.042] | 0.063 [0.032, 0.104] | no | 0.013 [0.000, 0.033] | 1 |
| chain | 0 | 40 | 120 | pdc/low | analytic | 25 | alpha | 6 | 150 | 0.000 [0.000, 0.000] | 0.013 [0.000, 0.032] | 0.060 [0.029, 0.101] | no | 0.007 [0.000, 0.020] | 1 |
| chain | 0 | 40 | 120 | pdc/low | analytic | 25 | beta | 18 | 150 | 0.005 [0.000, 0.016] | 0.021 [0.004, 0.041] | 0.059 [0.024, 0.100] | no | 0.013 [0.000, 0.033] | 1 |
| chain | 0 | 40 | 120 | pdc/low | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.040 [0.018, 0.063] | 0.085 [0.046, 0.128] | no | 0.027 [0.007, 0.053] | 26 |
| chain | 0 | 40 | 120 | pdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.049 [0.023, 0.079] | 0.086 [0.044, 0.129] | no | 0.020 [0.000, 0.040] | 3 |
| chain | 0 | 40 | 120 | pdc/low | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.037 [0.012, 0.064] | 0.084 [0.043, 0.135] | no | 0.027 [0.007, 0.053] | 9 |
| chain | 0 | 40 | 120 | rpdc/bic | analytic | 25 | all | 51 | 150 | 0.000 [0.000, 0.000] | 0.027 [0.007, 0.053] | 0.092 [0.040, 0.151] | no | 0.027 [0.007, 0.053] | 1 |
| chain | 0 | 40 | 120 | rpdc/bic | analytic | 25 | alpha | 6 | 150 | 0.000 [0.000, 0.000] | 0.027 [0.007, 0.053] | 0.093 [0.040, 0.153] | no | 0.027 [0.007, 0.053] | 1 |
| chain | 0 | 40 | 120 | rpdc/bic | analytic | 25 | beta | 18 | 150 | 0.000 [0.000, 0.000] | 0.027 [0.007, 0.053] | 0.093 [0.047, 0.153] | no | 0.027 [0.007, 0.053] | 1 |
| chain | 0 | 40 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.040 [0.008, 0.085] | 0.061 [0.021, 0.101] | no | 0.020 [0.000, 0.047] | 26 |
| chain | 0 | 40 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.040 [0.007, 0.080] | 0.060 [0.020, 0.100] | no | 0.020 [0.000, 0.040] | 3 |
| chain | 0 | 40 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.040 [0.007, 0.080] | 0.060 [0.020, 0.100] | no | 0.040 [0.007, 0.080] | 9 |
| chain | 0 | 40 | 120 | rpdc/fixed | analytic | 25 | all | 51 | 150 | 0.009 [0.006, 0.013] | 0.048 [0.039, 0.058] | 0.094 [0.081, 0.109] | no | 0.040 [0.013, 0.067] | 1 |
| chain | 0 | 40 | 120 | rpdc/fixed | analytic | 25 | alpha | 6 | 150 | 0.010 [0.000, 0.022] | 0.042 [0.024, 0.064] | 0.082 [0.059, 0.108] | no | 0.020 [0.000, 0.040] | 1 |
| chain | 0 | 40 | 120 | rpdc/fixed | analytic | 25 | beta | 18 | 150 | 0.009 [0.004, 0.014] | 0.049 [0.037, 0.063] | 0.092 [0.073, 0.110] | no | 0.027 [0.007, 0.053] | 1 |
| chain | 0 | 40 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.048 [0.040, 0.058] | 0.099 [0.086, 0.113] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0 | 40 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.046 [0.024, 0.070] | 0.092 [0.063, 0.124] | no | 0.007 [0.000, 0.020] | 3 |
| chain | 0 | 40 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.052 [0.038, 0.067] | 0.099 [0.080, 0.121] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0 | 40 | 120 | rpdc/low | analytic | 25 | all | 51 | 150 | 0.000 [0.000, 0.000] | 0.027 [0.007, 0.053] | 0.092 [0.040, 0.145] | no | 0.027 [0.007, 0.053] | 1 |
| chain | 0 | 40 | 120 | rpdc/low | analytic | 25 | alpha | 6 | 150 | 0.000 [0.000, 0.000] | 0.027 [0.007, 0.053] | 0.093 [0.047, 0.153] | no | 0.027 [0.007, 0.053] | 1 |
| chain | 0 | 40 | 120 | rpdc/low | analytic | 25 | beta | 18 | 150 | 0.000 [0.000, 0.000] | 0.027 [0.007, 0.053] | 0.093 [0.040, 0.153] | no | 0.027 [0.007, 0.053] | 1 |
| chain | 0 | 40 | 120 | rpdc/low | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.040 [0.007, 0.079] | 0.061 [0.021, 0.107] | no | 0.020 [0.000, 0.047] | 26 |
| chain | 0 | 40 | 120 | rpdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.040 [0.007, 0.080] | 0.060 [0.027, 0.107] | no | 0.020 [0.000, 0.040] | 3 |
| chain | 0 | 40 | 120 | rpdc/low | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.040 [0.007, 0.080] | 0.060 [0.020, 0.100] | no | 0.040 [0.007, 0.080] | 9 |
| chain | 0 | 40 | 30 | pdc/bic | analytic | 25 | all | 51 | 150 | 0.003 [0.000, 0.007] | 0.043 [0.020, 0.072] | 0.076 [0.045, 0.113] | no | 0.027 [0.007, 0.053] | 1 |
| chain | 0 | 40 | 30 | pdc/bic | analytic | 25 | alpha | 6 | 150 | 0.008 [0.000, 0.021] | 0.024 [0.007, 0.048] | 0.078 [0.040, 0.121] | no | 0.013 [0.000, 0.033] | 1 |
| chain | 0 | 40 | 30 | pdc/bic | analytic | 25 | beta | 18 | 150 | 0.000 [0.000, 0.000] | 0.032 [0.010, 0.060] | 0.076 [0.035, 0.117] | no | 0.013 [0.000, 0.033] | 1 |
| chain | 0 | 40 | 30 | pdc/bic | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.041 [0.016, 0.070] | 0.087 [0.042, 0.135] | no | 0.020 [0.000, 0.040] | 26 |
| chain | 0 | 40 | 30 | pdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.048 [0.014, 0.089] | 0.099 [0.050, 0.149] | no | 0.020 [0.000, 0.047] | 3 |
| chain | 0 | 40 | 30 | pdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.035 [0.008, 0.064] | 0.097 [0.049, 0.153] | no | 0.020 [0.000, 0.040] | 9 |
| chain | 0 | 40 | 30 | pdc/fixed | analytic | 25 | all | 51 | 150 | 0.002 [0.001, 0.004] | 0.024 [0.017, 0.031] | 0.065 [0.053, 0.078] | no | 0.000 [0.000, 0.000] | 1 |
| chain | 0 | 40 | 30 | pdc/fixed | analytic | 25 | alpha | 6 | 150 | 0.002 [0.000, 0.006] | 0.017 [0.008, 0.028] | 0.044 [0.028, 0.063] | no | 0.013 [0.000, 0.033] | 1 |
| chain | 0 | 40 | 30 | pdc/fixed | analytic | 25 | beta | 18 | 150 | 0.002 [0.000, 0.005] | 0.023 [0.012, 0.035] | 0.064 [0.046, 0.083] | no | 0.007 [0.000, 0.020] | 1 |
| chain | 0 | 40 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.047 [0.037, 0.056] | 0.095 [0.079, 0.112] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0 | 40 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.039 [0.020, 0.059] | 0.071 [0.050, 0.093] | no | 0.000 [0.000, 0.000] | 3 |
| chain | 0 | 40 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.052 [0.038, 0.068] | 0.103 [0.080, 0.126] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0 | 40 | 30 | pdc/low | analytic | 25 | all | 51 | 150 | 0.003 [0.000, 0.008] | 0.043 [0.022, 0.070] | 0.076 [0.040, 0.110] | no | 0.027 [0.007, 0.053] | 1 |
| chain | 0 | 40 | 30 | pdc/low | analytic | 25 | alpha | 6 | 150 | 0.008 [0.000, 0.021] | 0.024 [0.006, 0.048] | 0.078 [0.040, 0.119] | no | 0.013 [0.000, 0.033] | 1 |
| chain | 0 | 40 | 30 | pdc/low | analytic | 25 | beta | 18 | 150 | 0.000 [0.000, 0.000] | 0.032 [0.009, 0.059] | 0.076 [0.035, 0.120] | no | 0.013 [0.000, 0.033] | 1 |
| chain | 0 | 40 | 30 | pdc/low | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.041 [0.017, 0.069] | 0.087 [0.045, 0.138] | no | 0.020 [0.000, 0.047] | 26 |
| chain | 0 | 40 | 30 | pdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.048 [0.014, 0.088] | 0.099 [0.050, 0.153] | no | 0.020 [0.000, 0.047] | 3 |
| chain | 0 | 40 | 30 | pdc/low | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.035 [0.008, 0.067] | 0.097 [0.050, 0.149] | no | 0.020 [0.000, 0.040] | 9 |
| chain | 0 | 40 | 30 | rpdc/bic | analytic | 25 | all | 51 | 150 | 0.007 [0.000, 0.020] | 0.060 [0.027, 0.099] | 0.086 [0.047, 0.132] | no | 0.060 [0.027, 0.100] | 1 |
| chain | 0 | 40 | 30 | rpdc/bic | analytic | 25 | alpha | 6 | 150 | 0.007 [0.000, 0.020] | 0.060 [0.027, 0.093] | 0.087 [0.040, 0.133] | no | 0.060 [0.027, 0.100] | 1 |
| chain | 0 | 40 | 30 | rpdc/bic | analytic | 25 | beta | 18 | 150 | 0.007 [0.000, 0.020] | 0.060 [0.027, 0.100] | 0.087 [0.047, 0.133] | no | 0.060 [0.027, 0.100] | 1 |
| chain | 0 | 40 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.040 [0.008, 0.073] | 0.106 [0.054, 0.159] | no | 0.013 [0.000, 0.033] | 26 |
| chain | 0 | 40 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.040 [0.013, 0.073] | 0.107 [0.053, 0.160] | no | 0.013 [0.000, 0.033] | 3 |
| chain | 0 | 40 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.040 [0.007, 0.080] | 0.107 [0.053, 0.167] | no | 0.040 [0.013, 0.080] | 9 |
| chain | 0 | 40 | 30 | rpdc/fixed | analytic | 25 | all | 51 | 150 | 0.009 [0.006, 0.012] | 0.046 [0.035, 0.057] | 0.094 [0.077, 0.112] | no | 0.033 [0.007, 0.067] | 1 |
| chain | 0 | 40 | 30 | rpdc/fixed | analytic | 25 | alpha | 6 | 150 | 0.010 [0.002, 0.019] | 0.036 [0.020, 0.053] | 0.067 [0.046, 0.092] | no | 0.033 [0.007, 0.060] | 1 |
| chain | 0 | 40 | 30 | rpdc/fixed | analytic | 25 | beta | 18 | 150 | 0.011 [0.006, 0.019] | 0.045 [0.031, 0.059] | 0.090 [0.068, 0.113] | no | 0.047 [0.013, 0.087] | 1 |
| chain | 0 | 40 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.044 [0.034, 0.054] | 0.091 [0.075, 0.108] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0 | 40 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.042 [0.023, 0.064] | 0.073 [0.051, 0.097] | no | 0.000 [0.000, 0.000] | 3 |
| chain | 0 | 40 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.051 [0.036, 0.066] | 0.102 [0.079, 0.128] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0 | 40 | 30 | rpdc/low | analytic | 25 | all | 51 | 150 | 0.007 [0.000, 0.020] | 0.060 [0.027, 0.099] | 0.086 [0.047, 0.132] | no | 0.060 [0.027, 0.100] | 1 |
| chain | 0 | 40 | 30 | rpdc/low | analytic | 25 | alpha | 6 | 150 | 0.007 [0.000, 0.020] | 0.060 [0.027, 0.100] | 0.087 [0.040, 0.133] | no | 0.060 [0.027, 0.093] | 1 |
| chain | 0 | 40 | 30 | rpdc/low | analytic | 25 | beta | 18 | 150 | 0.007 [0.000, 0.020] | 0.060 [0.027, 0.100] | 0.087 [0.047, 0.133] | no | 0.060 [0.027, 0.100] | 1 |
| chain | 0 | 40 | 30 | rpdc/low | surrogate:phase_randomize | 25 | all | 51 | 150 | n/a | 0.040 [0.014, 0.079] | 0.106 [0.053, 0.159] | no | 0.013 [0.000, 0.033] | 26 |
| chain | 0 | 40 | 30 | rpdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 150 | n/a | 0.040 [0.013, 0.080] | 0.107 [0.053, 0.160] | no | 0.013 [0.000, 0.033] | 3 |
| chain | 0 | 40 | 30 | rpdc/low | surrogate:phase_randomize | 25 | beta | 18 | 150 | n/a | 0.040 [0.013, 0.080] | 0.107 [0.053, 0.167] | no | 0.040 [0.013, 0.080] | 9 |
| chain | 0.4 | 10 | 120 | pdc/bic | analytic | 25 | all | 51 | 100 | 0.291 [0.274, 0.308] | 0.415 [0.390, 0.445] | 0.483 [0.451, 0.516] | yes | 0.730 [0.690, 0.770] | 1 |
| chain | 0.4 | 10 | 120 | pdc/bic | analytic | 25 | alpha | 6 | 100 | 0.565 [0.532, 0.602] | 0.690 [0.660, 0.723] | 0.752 [0.718, 0.790] | yes | 0.720 [0.690, 0.750] | 1 |
| chain | 0.4 | 10 | 120 | pdc/bic | analytic | 25 | beta | 18 | 100 | 0.249 [0.223, 0.274] | 0.401 [0.369, 0.434] | 0.488 [0.453, 0.530] | yes | 0.520 [0.450, 0.590] | 1 |
| chain | 0.4 | 10 | 120 | pdc/bic | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.305 [0.275, 0.333] | 0.395 [0.359, 0.430] | yes | 0.160 [0.110, 0.200] | 26 |
| chain | 0.4 | 10 | 120 | pdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.297 [0.242, 0.347] | 0.452 [0.398, 0.503] | yes | 0.100 [0.050, 0.150] | 3 |
| chain | 0.4 | 10 | 120 | pdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.262 [0.233, 0.288] | 0.345 [0.318, 0.374] | yes | 0.170 [0.120, 0.210] | 9 |
| chain | 0.4 | 10 | 120 | pdc/fixed | analytic | 25 | all | 51 | 100 | 0.039 [0.030, 0.048] | 0.140 [0.124, 0.156] | 0.240 [0.225, 0.256] | yes | 0.180 [0.110, 0.250] | 1 |
| chain | 0.4 | 10 | 120 | pdc/fixed | analytic | 25 | alpha | 6 | 100 | 0.095 [0.057, 0.138] | 0.297 [0.245, 0.353] | 0.448 [0.405, 0.493] | yes | 0.260 [0.170, 0.350] | 1 |
| chain | 0.4 | 10 | 120 | pdc/fixed | analytic | 25 | beta | 18 | 100 | 0.039 [0.027, 0.052] | 0.143 [0.119, 0.167] | 0.254 [0.229, 0.282] | yes | 0.170 [0.090, 0.260] | 1 |
| chain | 0.4 | 10 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.151 [0.137, 0.166] | 0.244 [0.224, 0.265] | yes | 0.000 [0.000, 0.000] | 26 |
| chain | 0.4 | 10 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.140 [0.103, 0.182] | 0.265 [0.217, 0.315] | yes | 0.050 [0.010, 0.090] | 3 |
| chain | 0.4 | 10 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.153 [0.129, 0.177] | 0.234 [0.205, 0.261] | yes | 0.010 [0.000, 0.030] | 9 |
| chain | 0.4 | 10 | 120 | pdc/low | analytic | 25 | all | 51 | 100 | 0.993 [0.989, 0.996] | 0.998 [0.995, 1.000] | 0.999 [0.998, 1.000] | yes | 1.000 [1.000, 1.000] | 1 |
| chain | 0.4 | 10 | 120 | pdc/low | analytic | 25 | alpha | 6 | 100 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | yes | 1.000 [1.000, 1.000] | 1 |
| chain | 0.4 | 10 | 120 | pdc/low | analytic | 25 | beta | 18 | 100 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | yes | 1.000 [1.000, 1.000] | 1 |
| chain | 0.4 | 10 | 120 | pdc/low | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.981 [0.975, 0.987] | 0.992 [0.988, 0.996] | yes | 1.000 [1.000, 1.000] | 26 |
| chain | 0.4 | 10 | 120 | pdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.997 [0.992, 1.000] | 1.000 [1.000, 1.000] | yes | 0.980 [0.950, 1.000] | 3 |
| chain | 0.4 | 10 | 120 | pdc/low | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | yes | 1.000 [1.000, 1.000] | 9 |
| chain | 0.4 | 10 | 120 | rpdc/bic | analytic | 25 | all | 51 | 100 | 0.368 [0.351, 0.384] | 0.465 [0.440, 0.493] | 0.520 [0.488, 0.557] | yes | 0.740 [0.700, 0.770] | 1 |
| chain | 0.4 | 10 | 120 | rpdc/bic | analytic | 25 | alpha | 6 | 100 | 0.638 [0.607, 0.673] | 0.717 [0.685, 0.748] | 0.755 [0.723, 0.787] | yes | 0.730 [0.700, 0.750] | 1 |
| chain | 0.4 | 10 | 120 | rpdc/bic | analytic | 25 | beta | 18 | 100 | 0.338 [0.314, 0.360] | 0.467 [0.432, 0.509] | 0.523 [0.484, 0.566] | yes | 0.720 [0.670, 0.760] | 1 |
| chain | 0.4 | 10 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.417 [0.388, 0.446] | 0.497 [0.465, 0.531] | yes | 0.250 [0.210, 0.300] | 26 |
| chain | 0.4 | 10 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.615 [0.570, 0.662] | 0.695 [0.663, 0.728] | yes | 0.530 [0.450, 0.600] | 3 |
| chain | 0.4 | 10 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.399 [0.362, 0.437] | 0.489 [0.448, 0.532] | yes | 0.300 [0.240, 0.360] | 9 |
| chain | 0.4 | 10 | 120 | rpdc/fixed | analytic | 25 | all | 51 | 100 | 0.099 [0.086, 0.112] | 0.212 [0.194, 0.229] | 0.293 [0.278, 0.309] | yes | 0.500 [0.430, 0.570] | 1 |
| chain | 0.4 | 10 | 120 | rpdc/fixed | analytic | 25 | alpha | 6 | 100 | 0.220 [0.180, 0.260] | 0.405 [0.355, 0.457] | 0.495 [0.453, 0.540] | yes | 0.460 [0.390, 0.520] | 1 |
| chain | 0.4 | 10 | 120 | rpdc/fixed | analytic | 25 | beta | 18 | 100 | 0.097 [0.079, 0.115] | 0.217 [0.192, 0.245] | 0.308 [0.284, 0.331] | yes | 0.400 [0.330, 0.470] | 1 |
| chain | 0.4 | 10 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.144 [0.128, 0.158] | 0.232 [0.215, 0.249] | yes | 0.000 [0.000, 0.000] | 26 |
| chain | 0.4 | 10 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.207 [0.157, 0.258] | 0.342 [0.290, 0.395] | yes | 0.090 [0.040, 0.140] | 3 |
| chain | 0.4 | 10 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.134 [0.108, 0.161] | 0.232 [0.204, 0.258] | yes | 0.010 [0.000, 0.030] | 9 |
| chain | 0.4 | 10 | 120 | rpdc/low | analytic | 25 | all | 51 | 100 | 0.998 [0.997, 0.999] | 0.999 [0.998, 1.000] | 0.999 [0.999, 1.000] | yes | 1.000 [1.000, 1.000] | 1 |
| chain | 0.4 | 10 | 120 | rpdc/low | analytic | 25 | alpha | 6 | 100 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | yes | 1.000 [1.000, 1.000] | 1 |
| chain | 0.4 | 10 | 120 | rpdc/low | analytic | 25 | beta | 18 | 100 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | yes | 1.000 [1.000, 1.000] | 1 |
| chain | 0.4 | 10 | 120 | rpdc/low | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.999 [0.998, 1.000] | 1.000 [0.999, 1.000] | yes | 1.000 [1.000, 1.000] | 26 |
| chain | 0.4 | 10 | 120 | rpdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | yes | 1.000 [1.000, 1.000] | 3 |
| chain | 0.4 | 10 | 120 | rpdc/low | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | yes | 1.000 [1.000, 1.000] | 9 |
| chain | 0.4 | 10 | 30 | pdc/bic | analytic | 25 | all | 51 | 100 | 0.086 [0.061, 0.112] | 0.226 [0.189, 0.264] | 0.321 [0.274, 0.364] | yes | 0.270 [0.200, 0.350] | 1 |
| chain | 0.4 | 10 | 30 | pdc/bic | analytic | 25 | alpha | 6 | 100 | 0.095 [0.042, 0.155] | 0.262 [0.188, 0.345] | 0.382 [0.302, 0.458] | yes | 0.210 [0.140, 0.280] | 1 |
| chain | 0.4 | 10 | 30 | pdc/bic | analytic | 25 | beta | 18 | 100 | 0.096 [0.065, 0.132] | 0.229 [0.182, 0.276] | 0.324 [0.273, 0.381] | yes | 0.210 [0.150, 0.280] | 1 |
| chain | 0.4 | 10 | 30 | pdc/bic | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.089 [0.054, 0.134] | 0.160 [0.115, 0.217] | yes | 0.030 [0.000, 0.080] | 26 |
| chain | 0.4 | 10 | 30 | pdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.043 [0.017, 0.073] | 0.102 [0.050, 0.165] | no | 0.010 [0.000, 0.030] | 3 |
| chain | 0.4 | 10 | 30 | pdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.083 [0.047, 0.126] | 0.142 [0.094, 0.194] | no | 0.050 [0.010, 0.100] | 9 |
| chain | 0.4 | 10 | 30 | pdc/fixed | analytic | 25 | all | 51 | 100 | 0.004 [0.002, 0.007] | 0.041 [0.031, 0.051] | 0.099 [0.083, 0.116] | no | 0.010 [0.000, 0.030] | 1 |
| chain | 0.4 | 10 | 30 | pdc/fixed | analytic | 25 | alpha | 6 | 100 | 0.008 [0.000, 0.018] | 0.072 [0.040, 0.102] | 0.145 [0.102, 0.187] | no | 0.050 [0.020, 0.090] | 1 |
| chain | 0.4 | 10 | 30 | pdc/fixed | analytic | 25 | beta | 18 | 100 | 0.001 [0.000, 0.003] | 0.032 [0.018, 0.047] | 0.081 [0.060, 0.103] | no | 0.000 [0.000, 0.000] | 1 |
| chain | 0.4 | 10 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.072 [0.061, 0.083] | 0.125 [0.111, 0.142] | yes | 0.000 [0.000, 0.000] | 26 |
| chain | 0.4 | 10 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.030 [0.010, 0.057] | 0.060 [0.030, 0.097] | no | 0.000 [0.000, 0.000] | 3 |
| chain | 0.4 | 10 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.061 [0.041, 0.079] | 0.107 [0.081, 0.132] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0.4 | 10 | 30 | pdc/low | analytic | 25 | all | 51 | 100 | 0.826 [0.779, 0.869] | 0.933 [0.907, 0.955] | 0.966 [0.950, 0.981] | yes | 0.950 [0.910, 0.990] | 1 |
| chain | 0.4 | 10 | 30 | pdc/low | analytic | 25 | alpha | 6 | 100 | 0.745 [0.688, 0.800] | 0.952 [0.918, 0.980] | 0.978 [0.950, 1.000] | yes | 0.920 [0.870, 0.960] | 1 |
| chain | 0.4 | 10 | 30 | pdc/low | analytic | 25 | beta | 18 | 100 | 0.872 [0.809, 0.926] | 0.966 [0.942, 0.989] | 0.997 [0.993, 1.000] | yes | 0.940 [0.900, 0.980] | 1 |
| chain | 0.4 | 10 | 30 | pdc/low | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.848 [0.805, 0.885] | 0.899 [0.872, 0.925] | yes | 0.860 [0.800, 0.910] | 26 |
| chain | 0.4 | 10 | 30 | pdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.727 [0.682, 0.775] | 0.805 [0.760, 0.852] | yes | 0.590 [0.530, 0.650] | 3 |
| chain | 0.4 | 10 | 30 | pdc/low | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.872 [0.825, 0.911] | 0.936 [0.901, 0.965] | yes | 0.850 [0.790, 0.900] | 9 |
| chain | 0.4 | 10 | 30 | rpdc/bic | analytic | 25 | all | 51 | 100 | 0.147 [0.101, 0.196] | 0.290 [0.236, 0.346] | 0.375 [0.327, 0.424] | yes | 0.300 [0.230, 0.370] | 1 |
| chain | 0.4 | 10 | 30 | rpdc/bic | analytic | 25 | alpha | 6 | 100 | 0.140 [0.077, 0.205] | 0.325 [0.237, 0.418] | 0.443 [0.360, 0.522] | yes | 0.260 [0.170, 0.350] | 1 |
| chain | 0.4 | 10 | 30 | rpdc/bic | analytic | 25 | beta | 18 | 100 | 0.154 [0.104, 0.205] | 0.281 [0.234, 0.332] | 0.353 [0.303, 0.406] | yes | 0.320 [0.250, 0.390] | 1 |
| chain | 0.4 | 10 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.198 [0.149, 0.249] | 0.305 [0.253, 0.366] | yes | 0.130 [0.070, 0.200] | 26 |
| chain | 0.4 | 10 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.188 [0.115, 0.270] | 0.278 [0.203, 0.362] | yes | 0.100 [0.050, 0.160] | 3 |
| chain | 0.4 | 10 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.168 [0.118, 0.221] | 0.284 [0.228, 0.349] | yes | 0.100 [0.050, 0.150] | 9 |
| chain | 0.4 | 10 | 30 | rpdc/fixed | analytic | 25 | all | 51 | 100 | 0.020 [0.014, 0.026] | 0.077 [0.064, 0.090] | 0.138 [0.123, 0.156] | yes | 0.070 [0.030, 0.120] | 1 |
| chain | 0.4 | 10 | 30 | rpdc/fixed | analytic | 25 | alpha | 6 | 100 | 0.045 [0.025, 0.068] | 0.118 [0.082, 0.155] | 0.188 [0.138, 0.238] | yes | 0.140 [0.080, 0.200] | 1 |
| chain | 0.4 | 10 | 30 | rpdc/fixed | analytic | 25 | beta | 18 | 100 | 0.014 [0.008, 0.022] | 0.064 [0.044, 0.083] | 0.119 [0.095, 0.144] | no | 0.060 [0.020, 0.100] | 1 |
| chain | 0.4 | 10 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.055 [0.045, 0.066] | 0.101 [0.084, 0.116] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0.4 | 10 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.052 [0.030, 0.075] | 0.088 [0.055, 0.120] | no | 0.000 [0.000, 0.000] | 3 |
| chain | 0.4 | 10 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.038 [0.022, 0.054] | 0.078 [0.054, 0.104] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0.4 | 10 | 30 | rpdc/low | analytic | 25 | all | 51 | 100 | 0.932 [0.892, 0.971] | 0.975 [0.945, 0.994] | 0.996 [0.995, 0.998] | yes | 0.980 [0.950, 1.000] | 1 |
| chain | 0.4 | 10 | 30 | rpdc/low | analytic | 25 | alpha | 6 | 100 | 0.940 [0.890, 0.980] | 0.980 [0.950, 1.000] | 1.000 [1.000, 1.000] | yes | 0.980 [0.950, 1.000] | 1 |
| chain | 0.4 | 10 | 30 | rpdc/low | analytic | 25 | beta | 18 | 100 | 0.940 [0.900, 0.980] | 0.980 [0.950, 1.000] | 1.000 [1.000, 1.000] | yes | 0.980 [0.950, 1.000] | 1 |
| chain | 0.4 | 10 | 30 | rpdc/low | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.954 [0.915, 0.984] | 0.985 [0.965, 0.996] | yes | 0.910 [0.850, 0.960] | 26 |
| chain | 0.4 | 10 | 30 | rpdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.960 [0.920, 0.990] | 0.990 [0.970, 1.000] | yes | 0.880 [0.820, 0.930] | 3 |
| chain | 0.4 | 10 | 30 | rpdc/low | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.960 [0.920, 0.990] | 0.990 [0.970, 1.000] | yes | 0.960 [0.920, 0.990] | 9 |
| chain | 0.4 | 40 | 120 | pdc/bic | analytic | 25 | all | 51 | 100 | 0.027 [0.000, 0.060] | 0.068 [0.027, 0.118] | 0.124 [0.074, 0.174] | no | 0.050 [0.010, 0.090] | 1 |
| chain | 0.4 | 40 | 120 | pdc/bic | analytic | 25 | alpha | 6 | 100 | 0.020 [0.000, 0.050] | 0.050 [0.010, 0.100] | 0.068 [0.017, 0.130] | no | 0.050 [0.010, 0.100] | 1 |
| chain | 0.4 | 40 | 120 | pdc/bic | analytic | 25 | beta | 18 | 100 | 0.024 [0.000, 0.054] | 0.057 [0.019, 0.106] | 0.116 [0.065, 0.173] | no | 0.040 [0.010, 0.080] | 1 |
| chain | 0.4 | 40 | 120 | pdc/bic | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.003 [0.000, 0.008] | 0.003 [0.000, 0.009] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0.4 | 40 | 120 | pdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | no | 0.000 [0.000, 0.000] | 3 |
| chain | 0.4 | 40 | 120 | pdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.008 [0.000, 0.022] | 0.009 [0.000, 0.027] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0.4 | 40 | 120 | pdc/fixed | analytic | 25 | all | 51 | 100 | 0.002 [0.000, 0.004] | 0.024 [0.019, 0.031] | 0.074 [0.062, 0.088] | no | 0.010 [0.000, 0.030] | 1 |
| chain | 0.4 | 40 | 120 | pdc/fixed | analytic | 25 | alpha | 6 | 100 | 0.000 [0.000, 0.000] | 0.030 [0.007, 0.062] | 0.073 [0.038, 0.117] | no | 0.010 [0.000, 0.030] | 1 |
| chain | 0.4 | 40 | 120 | pdc/fixed | analytic | 25 | beta | 18 | 100 | 0.002 [0.000, 0.005] | 0.020 [0.011, 0.030] | 0.071 [0.046, 0.099] | no | 0.010 [0.000, 0.030] | 1 |
| chain | 0.4 | 40 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.066 [0.053, 0.080] | 0.112 [0.095, 0.130] | yes | 0.000 [0.000, 0.000] | 26 |
| chain | 0.4 | 40 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.015 [0.005, 0.027] | 0.037 [0.017, 0.063] | no | 0.000 [0.000, 0.000] | 3 |
| chain | 0.4 | 40 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.078 [0.053, 0.107] | 0.119 [0.091, 0.150] | yes | 0.000 [0.000, 0.000] | 9 |
| chain | 0.4 | 40 | 120 | pdc/low | analytic | 25 | all | 51 | 100 | 0.027 [0.000, 0.062] | 0.068 [0.025, 0.117] | 0.124 [0.077, 0.174] | no | 0.050 [0.020, 0.090] | 1 |
| chain | 0.4 | 40 | 120 | pdc/low | analytic | 25 | alpha | 6 | 100 | 0.020 [0.000, 0.050] | 0.050 [0.010, 0.110] | 0.068 [0.017, 0.133] | no | 0.050 [0.010, 0.100] | 1 |
| chain | 0.4 | 40 | 120 | pdc/low | analytic | 25 | beta | 18 | 100 | 0.024 [0.000, 0.052] | 0.057 [0.018, 0.104] | 0.116 [0.066, 0.171] | no | 0.040 [0.010, 0.080] | 1 |
| chain | 0.4 | 40 | 120 | pdc/low | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.060 [0.026, 0.103] | 0.083 [0.042, 0.128] | no | 0.060 [0.010, 0.110] | 26 |
| chain | 0.4 | 40 | 120 | pdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.045 [0.008, 0.090] | 0.068 [0.028, 0.123] | no | 0.030 [0.000, 0.070] | 3 |
| chain | 0.4 | 40 | 120 | pdc/low | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.056 [0.021, 0.095] | 0.085 [0.042, 0.134] | no | 0.070 [0.020, 0.130] | 9 |
| chain | 0.4 | 40 | 120 | rpdc/bic | analytic | 25 | all | 51 | 100 | 0.040 [0.010, 0.080] | 0.090 [0.030, 0.159] | 0.101 [0.042, 0.170] | no | 0.090 [0.030, 0.160] | 1 |
| chain | 0.4 | 40 | 120 | rpdc/bic | analytic | 25 | alpha | 6 | 100 | 0.040 [0.010, 0.080] | 0.090 [0.030, 0.160] | 0.100 [0.040, 0.170] | no | 0.090 [0.030, 0.160] | 1 |
| chain | 0.4 | 40 | 120 | rpdc/bic | analytic | 25 | beta | 18 | 100 | 0.040 [0.010, 0.080] | 0.090 [0.030, 0.170] | 0.100 [0.040, 0.170] | no | 0.090 [0.030, 0.160] | 1 |
| chain | 0.4 | 40 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.058 [0.017, 0.107] | 0.089 [0.037, 0.153] | no | 0.040 [0.010, 0.080] | 26 |
| chain | 0.4 | 40 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.052 [0.005, 0.110] | 0.067 [0.015, 0.130] | no | 0.030 [0.000, 0.060] | 3 |
| chain | 0.4 | 40 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.042 [0.008, 0.084] | 0.083 [0.032, 0.152] | no | 0.030 [0.000, 0.060] | 9 |
| chain | 0.4 | 40 | 120 | rpdc/fixed | analytic | 25 | all | 51 | 100 | 0.009 [0.005, 0.014] | 0.057 [0.046, 0.068] | 0.107 [0.091, 0.124] | no | 0.030 [0.000, 0.070] | 1 |
| chain | 0.4 | 40 | 120 | rpdc/fixed | analytic | 25 | alpha | 6 | 100 | 0.017 [0.003, 0.035] | 0.053 [0.022, 0.095] | 0.112 [0.072, 0.153] | no | 0.050 [0.010, 0.110] | 1 |
| chain | 0.4 | 40 | 120 | rpdc/fixed | analytic | 25 | beta | 18 | 100 | 0.009 [0.003, 0.016] | 0.053 [0.034, 0.072] | 0.107 [0.078, 0.139] | no | 0.040 [0.010, 0.080] | 1 |
| chain | 0.4 | 40 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.035 [0.026, 0.044] | 0.077 [0.064, 0.091] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0.4 | 40 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.015 [0.003, 0.032] | 0.035 [0.012, 0.063] | no | 0.010 [0.000, 0.030] | 3 |
| chain | 0.4 | 40 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.036 [0.022, 0.051] | 0.074 [0.051, 0.102] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0.4 | 40 | 120 | rpdc/low | analytic | 25 | all | 51 | 100 | 0.040 [0.010, 0.080] | 0.090 [0.030, 0.158] | 0.101 [0.042, 0.170] | no | 0.090 [0.040, 0.160] | 1 |
| chain | 0.4 | 40 | 120 | rpdc/low | analytic | 25 | alpha | 6 | 100 | 0.040 [0.010, 0.080] | 0.090 [0.030, 0.160] | 0.100 [0.040, 0.180] | no | 0.090 [0.030, 0.160] | 1 |
| chain | 0.4 | 40 | 120 | rpdc/low | analytic | 25 | beta | 18 | 100 | 0.040 [0.010, 0.080] | 0.090 [0.030, 0.160] | 0.100 [0.040, 0.180] | no | 0.090 [0.030, 0.160] | 1 |
| chain | 0.4 | 40 | 120 | rpdc/low | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.050 [0.010, 0.099] | 0.061 [0.011, 0.120] | no | 0.030 [0.000, 0.070] | 26 |
| chain | 0.4 | 40 | 120 | rpdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.050 [0.010, 0.100] | 0.060 [0.010, 0.120] | no | 0.030 [0.000, 0.070] | 3 |
| chain | 0.4 | 40 | 120 | rpdc/low | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.050 [0.010, 0.100] | 0.060 [0.010, 0.120] | no | 0.050 [0.010, 0.100] | 9 |
| chain | 0.4 | 40 | 30 | pdc/bic | analytic | 25 | all | 51 | 100 | 0.000 [0.000, 0.000] | 0.031 [0.005, 0.066] | 0.085 [0.043, 0.129] | no | 0.010 [0.000, 0.030] | 1 |
| chain | 0.4 | 40 | 30 | pdc/bic | analytic | 25 | alpha | 6 | 100 | 0.000 [0.000, 0.000] | 0.015 [0.000, 0.040] | 0.067 [0.027, 0.107] | no | 0.010 [0.000, 0.030] | 1 |
| chain | 0.4 | 40 | 30 | pdc/bic | analytic | 25 | beta | 18 | 100 | 0.000 [0.000, 0.000] | 0.023 [0.000, 0.056] | 0.070 [0.024, 0.133] | no | 0.010 [0.000, 0.030] | 1 |
| chain | 0.4 | 40 | 30 | pdc/bic | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.002] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0.4 | 40 | 30 | pdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | no | 0.000 [0.000, 0.000] | 3 |
| chain | 0.4 | 40 | 30 | pdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.000 [0.000, 0.000] | 0.003 [0.000, 0.007] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0.4 | 40 | 30 | pdc/fixed | analytic | 25 | all | 51 | 100 | 0.000 [0.000, 0.001] | 0.017 [0.011, 0.024] | 0.061 [0.045, 0.077] | no | 0.000 [0.000, 0.000] | 1 |
| chain | 0.4 | 40 | 30 | pdc/fixed | analytic | 25 | alpha | 6 | 100 | 0.000 [0.000, 0.000] | 0.018 [0.005, 0.035] | 0.053 [0.027, 0.087] | no | 0.000 [0.000, 0.000] | 1 |
| chain | 0.4 | 40 | 30 | pdc/fixed | analytic | 25 | beta | 18 | 100 | 0.000 [0.000, 0.000] | 0.019 [0.011, 0.029] | 0.070 [0.047, 0.095] | no | 0.000 [0.000, 0.000] | 1 |
| chain | 0.4 | 40 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.055 [0.041, 0.069] | 0.103 [0.084, 0.121] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0.4 | 40 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.013 [0.002, 0.028] | 0.023 [0.008, 0.043] | no | 0.000 [0.000, 0.000] | 3 |
| chain | 0.4 | 40 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.070 [0.048, 0.096] | 0.121 [0.090, 0.154] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0.4 | 40 | 30 | pdc/low | analytic | 25 | all | 51 | 100 | 0.000 [0.000, 0.000] | 0.031 [0.006, 0.068] | 0.085 [0.044, 0.132] | no | 0.010 [0.000, 0.030] | 1 |
| chain | 0.4 | 40 | 30 | pdc/low | analytic | 25 | alpha | 6 | 100 | 0.000 [0.000, 0.000] | 0.015 [0.000, 0.040] | 0.067 [0.030, 0.108] | no | 0.010 [0.000, 0.030] | 1 |
| chain | 0.4 | 40 | 30 | pdc/low | analytic | 25 | beta | 18 | 100 | 0.000 [0.000, 0.000] | 0.023 [0.000, 0.057] | 0.070 [0.023, 0.128] | no | 0.010 [0.000, 0.030] | 1 |
| chain | 0.4 | 40 | 30 | pdc/low | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.030 [0.008, 0.060] | 0.047 [0.017, 0.083] | no | 0.030 [0.000, 0.080] | 26 |
| chain | 0.4 | 40 | 30 | pdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.020 [0.000, 0.048] | 0.043 [0.013, 0.080] | no | 0.000 [0.000, 0.000] | 3 |
| chain | 0.4 | 40 | 30 | pdc/low | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.032 [0.006, 0.068] | 0.050 [0.019, 0.089] | no | 0.030 [0.000, 0.080] | 9 |
| chain | 0.4 | 40 | 30 | rpdc/bic | analytic | 25 | all | 51 | 100 | 0.000 [0.000, 0.000] | 0.059 [0.011, 0.118] | 0.100 [0.041, 0.169] | no | 0.060 [0.010, 0.120] | 1 |
| chain | 0.4 | 40 | 30 | rpdc/bic | analytic | 25 | alpha | 6 | 100 | 0.000 [0.000, 0.000] | 0.060 [0.020, 0.110] | 0.100 [0.040, 0.170] | no | 0.060 [0.010, 0.120] | 1 |
| chain | 0.4 | 40 | 30 | rpdc/bic | analytic | 25 | beta | 18 | 100 | 0.000 [0.000, 0.000] | 0.060 [0.020, 0.120] | 0.100 [0.040, 0.170] | no | 0.060 [0.020, 0.110] | 1 |
| chain | 0.4 | 40 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.016 [0.004, 0.032] | 0.062 [0.023, 0.108] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0.4 | 40 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.000 [0.000, 0.000] | 0.027 [0.007, 0.053] | no | 0.000 [0.000, 0.000] | 3 |
| chain | 0.4 | 40 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.021 [0.004, 0.041] | 0.065 [0.023, 0.116] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0.4 | 40 | 30 | rpdc/fixed | analytic | 25 | all | 51 | 100 | 0.005 [0.003, 0.008] | 0.042 [0.031, 0.054] | 0.091 [0.074, 0.109] | no | 0.010 [0.000, 0.030] | 1 |
| chain | 0.4 | 40 | 30 | rpdc/fixed | analytic | 25 | alpha | 6 | 100 | 0.005 [0.000, 0.012] | 0.040 [0.020, 0.063] | 0.080 [0.050, 0.115] | no | 0.040 [0.010, 0.080] | 1 |
| chain | 0.4 | 40 | 30 | rpdc/fixed | analytic | 25 | beta | 18 | 100 | 0.006 [0.002, 0.011] | 0.046 [0.029, 0.064] | 0.108 [0.082, 0.138] | no | 0.030 [0.000, 0.060] | 1 |
| chain | 0.4 | 40 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.025 [0.018, 0.033] | 0.063 [0.048, 0.078] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0.4 | 40 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.010 [0.003, 0.023] | 0.030 [0.013, 0.048] | no | 0.000 [0.000, 0.000] | 3 |
| chain | 0.4 | 40 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.028 [0.017, 0.040] | 0.064 [0.044, 0.086] | no | 0.000 [0.000, 0.000] | 9 |
| chain | 0.4 | 40 | 30 | rpdc/low | analytic | 25 | all | 51 | 100 | 0.000 [0.000, 0.000] | 0.059 [0.020, 0.109] | 0.100 [0.041, 0.169] | no | 0.060 [0.020, 0.110] | 1 |
| chain | 0.4 | 40 | 30 | rpdc/low | analytic | 25 | alpha | 6 | 100 | 0.000 [0.000, 0.000] | 0.060 [0.020, 0.110] | 0.100 [0.040, 0.180] | no | 0.060 [0.020, 0.110] | 1 |
| chain | 0.4 | 40 | 30 | rpdc/low | analytic | 25 | beta | 18 | 100 | 0.000 [0.000, 0.000] | 0.060 [0.020, 0.120] | 0.100 [0.040, 0.170] | no | 0.060 [0.020, 0.110] | 1 |
| chain | 0.4 | 40 | 30 | rpdc/low | surrogate:phase_randomize | 25 | all | 51 | 100 | n/a | 0.001 [0.000, 0.002] | 0.021 [0.001, 0.051] | no | 0.000 [0.000, 0.000] | 26 |
| chain | 0.4 | 40 | 30 | rpdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 100 | n/a | 0.000 [0.000, 0.000] | 0.020 [0.000, 0.050] | no | 0.000 [0.000, 0.000] | 3 |
| chain | 0.4 | 40 | 30 | rpdc/low | surrogate:phase_randomize | 25 | beta | 18 | 100 | n/a | 0.000 [0.000, 0.000] | 0.020 [0.000, 0.050] | no | 0.000 [0.000, 0.000] | 9 |

Power, pairs with a true edge:

| scenario | coupling | SNR | s | estimator | test | datasets | band | freqs | edge pairs | power at 0.01 | power at 0.05 | power at 0.1 | pairs after BH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | 0.4 | 10 | 120 | pdc/bic | analytic | 25 | all | 51 | 50 | 0.996 [0.991, 1.000] | 0.999 [0.998, 1.000] | 0.999 [0.998, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/bic | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/bic | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/bic | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 0.999 [0.996, 1.000] | 0.999 [0.998, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/fixed | analytic | 25 | all | 51 | 50 | 0.892 [0.883, 0.901] | 0.951 [0.942, 0.960] | 0.969 [0.963, 0.976] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/fixed | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/fixed | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 0.956 [0.948, 0.964] | 0.972 [0.967, 0.978] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/low | analytic | 25 | all | 51 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/low | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/low | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/low | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | pdc/low | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/bic | analytic | 25 | all | 51 | 50 | 0.999 [0.998, 1.000] | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/bic | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/bic | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/fixed | analytic | 25 | all | 51 | 50 | 0.940 [0.931, 0.949] | 0.970 [0.964, 0.976] | 0.977 [0.972, 0.982] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/fixed | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/fixed | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 0.963 [0.956, 0.969] | 0.976 [0.971, 0.982] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/low | analytic | 25 | all | 51 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/low | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/low | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/low | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 120 | rpdc/low | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/bic | analytic | 25 | all | 51 | 50 | 0.932 [0.918, 0.946] | 0.974 [0.960, 0.987] | 0.984 [0.974, 0.994] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/bic | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/bic | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/bic | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 0.933 [0.913, 0.953] | 0.959 [0.941, 0.976] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/fixed | analytic | 25 | all | 51 | 50 | 0.632 [0.618, 0.645] | 0.741 [0.725, 0.756] | 0.805 [0.791, 0.819] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/fixed | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/fixed | analytic | 25 | beta | 18 | 50 | 0.948 [0.933, 0.964] | 0.988 [0.979, 0.996] | 0.996 [0.989, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 0.736 [0.722, 0.749] | 0.789 [0.772, 0.804] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 0.979 [0.966, 0.991] | 0.992 [0.986, 0.998] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/low | analytic | 25 | all | 51 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/low | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/low | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/low | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | pdc/low | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/bic | analytic | 25 | all | 51 | 50 | 0.992 [0.987, 0.996] | 0.996 [0.994, 0.998] | 0.997 [0.995, 0.999] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/bic | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/bic | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 0.994 [0.990, 0.997] | 0.996 [0.993, 0.998] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/fixed | analytic | 25 | all | 51 | 50 | 0.713 [0.700, 0.727] | 0.794 [0.781, 0.807] | 0.841 [0.828, 0.854] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/fixed | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/fixed | analytic | 25 | beta | 18 | 50 | 0.982 [0.970, 0.992] | 0.992 [0.984, 0.998] | 0.997 [0.991, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 0.779 [0.766, 0.792] | 0.824 [0.812, 0.835] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 0.988 [0.978, 0.997] | 0.992 [0.986, 0.998] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/low | analytic | 25 | all | 51 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/low | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/low | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/low | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 10 | 30 | rpdc/low | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/bic | analytic | 25 | all | 51 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/bic | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/bic | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/bic | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/fixed | analytic | 25 | all | 51 | 50 | 0.981 [0.974, 0.989] | 0.997 [0.995, 0.999] | 0.999 [0.998, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/fixed | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/fixed | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 0.988 [0.982, 0.993] | 0.996 [0.994, 0.999] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/low | analytic | 25 | all | 51 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/low | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/low | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/low | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | pdc/low | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/bic | analytic | 25 | all | 51 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/bic | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/bic | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/fixed | analytic | 25 | all | 51 | 50 | 0.995 [0.991, 0.997] | 0.998 [0.996, 1.000] | 0.999 [0.998, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/fixed | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/fixed | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 0.997 [0.995, 0.999] | 0.998 [0.997, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/low | analytic | 25 | all | 51 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/low | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/low | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/low | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 120 | rpdc/low | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/bic | analytic | 25 | all | 51 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/bic | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/bic | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/bic | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 0.996 [0.990, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/fixed | analytic | 25 | all | 51 | 50 | 0.740 [0.722, 0.757] | 0.863 [0.845, 0.882] | 0.916 [0.901, 0.931] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/fixed | analytic | 25 | alpha | 6 | 50 | 0.997 [0.990, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/fixed | analytic | 25 | beta | 18 | 50 | 0.956 [0.942, 0.970] | 0.987 [0.979, 0.993] | 0.994 [0.988, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 0.832 [0.815, 0.849] | 0.882 [0.865, 0.898] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 0.969 [0.953, 0.982] | 0.986 [0.976, 0.994] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/low | analytic | 25 | all | 51 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/low | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/low | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/low | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | pdc/low | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/bic | analytic | 25 | all | 51 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/bic | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/bic | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/bic | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/fixed | analytic | 25 | all | 51 | 50 | 0.836 [0.820, 0.852] | 0.910 [0.894, 0.926] | 0.937 [0.925, 0.949] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/fixed | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/fixed | analytic | 25 | beta | 18 | 50 | 0.979 [0.966, 0.989] | 0.993 [0.987, 0.999] | 0.994 [0.988, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 0.896 [0.880, 0.914] | 0.929 [0.918, 0.942] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/fixed | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 0.989 [0.978, 0.997] | 0.996 [0.990, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/low | analytic | 25 | all | 51 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/low | analytic | 25 | alpha | 6 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/low | analytic | 25 | beta | 18 | 50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/low | surrogate:phase_randomize | 25 | all | 51 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/low | surrogate:phase_randomize | 25 | alpha | 6 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| chain | 0.4 | 40 | 30 | rpdc/low | surrogate:phase_randomize | 25 | beta | 18 | 50 | n/a | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

## Further methods on the head models (add-on)

Conditional Granger causality with the BIC order next to the further estimators, on three simple topologies. Scoring follows each method's convention (`dirconn.evaluation.convention_metrics`): `imcoh` is undirected (magnitude, an edge in either direction, no direction accuracy), `pcmci` is signed (magnitude), `trgc` and `psi` are net flows (a pair coupled both ways is left out). Mean AUROC over seeds:

Stage: roi.

| scenario | modality | gc/bic | imcoh/mt | mtgc/mt | pcmci/bic | psi/mt | te/bic | trgc/bic |
|---|---|---|---|---|---|---|---|---|
| chain | eeg | 1.000 | 0.500 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 |
| chain | grad | 1.000 | 0.600 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 |
| collider | eeg | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.980 | 1.000 |
| collider | grad | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.960 | 1.000 |
| common_driver | eeg | 1.000 | 0.875 | 1.000 | 1.000 | 0.950 | 1.000 | 1.000 |
| common_driver | grad | 1.000 | 0.875 | 1.000 | 1.000 | 0.950 | 1.000 | 1.000 |

Stage: sensor.

| scenario | modality | gc/bic | imcoh/mt | mtgc/mt | pcmci/bic | psi/mt | te/bic | trgc/bic |
|---|---|---|---|---|---|---|---|---|
| chain | eeg | 1.000 | 0.500 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 |
| chain | grad | 1.000 | 0.700 | 1.000 | 1.000 | 0.975 | 1.000 | 1.000 |
| collider | eeg | 0.610 | 0.975 | 0.670 | 0.560 | 0.990 | 0.650 | 0.910 |
| collider | grad | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| common_driver | eeg | 0.450 | 0.575 | 0.700 | 0.520 | 0.870 | 0.600 | 0.750 |
| common_driver | grad | 1.000 | 0.875 | 1.000 | 0.990 | 0.990 | 0.970 | 1.000 |

Stage: source.

| scenario | modality | gc/bic | imcoh/mt | mtgc/mt | pcmci/bic | psi/mt | te/bic | trgc/bic |
|---|---|---|---|---|---|---|---|---|
| chain | eeg | 1.000 | 0.500 | 1.000 | 0.925 | 0.850 | 0.975 | 1.000 |
| chain | grad | 1.000 | 0.500 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 |
| collider | eeg | 0.990 | 1.000 | 1.000 | 0.930 | 1.000 | 1.000 | 1.000 |
| collider | grad | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| common_driver | eeg | 0.980 | 0.700 | 1.000 | 0.940 | 0.860 | 1.000 | 1.000 |
| common_driver | grad | 1.000 | 0.875 | 1.000 | 1.000 | 0.950 | 1.000 | 1.000 |

Mean direction accuracy over seeds (NaN for the undirected `imcoh`):

Stage: roi.

| scenario | modality | gc/bic | imcoh/mt | mtgc/mt | pcmci/bic | psi/mt | te/bic | trgc/bic |
|---|---|---|---|---|---|---|---|---|
| chain | eeg | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | grad | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collider | eeg | 1.000 | - | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 |
| collider | grad | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| common_driver | eeg | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| common_driver | grad | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Stage: sensor.

| scenario | modality | gc/bic | imcoh/mt | mtgc/mt | pcmci/bic | psi/mt | te/bic | trgc/bic |
|---|---|---|---|---|---|---|---|---|
| chain | eeg | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chain | grad | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collider | eeg | 0.500 | - | 1.000 | 0.800 | 1.000 | 0.800 | 1.000 |
| collider | grad | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| common_driver | eeg | 1.000 | - | 1.000 | 0.900 | 1.000 | 0.900 | 0.900 |
| common_driver | grad | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Stage: source.

| scenario | modality | gc/bic | imcoh/mt | mtgc/mt | pcmci/bic | psi/mt | te/bic | trgc/bic |
|---|---|---|---|---|---|---|---|---|
| chain | eeg | 1.000 | - | 1.000 | 1.000 | 0.900 | 1.000 | 1.000 |
| chain | grad | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collider | eeg | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collider | grad | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| common_driver | eeg | 1.000 | - | 1.000 | 1.000 | 0.900 | 1.000 | 1.000 |
| common_driver | grad | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

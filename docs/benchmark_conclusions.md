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

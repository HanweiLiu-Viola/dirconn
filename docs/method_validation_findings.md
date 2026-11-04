## Findings

Read from the tables below (10 seeds, 10000 samples, region-level series). They describe these
scenarios and settings, not every dataset; re-run `scripts/validate_methods.py` after any change.

1. **Ranking is easy, specificity is not.** In the simple topologies (chain, common driver,
   collider, bidirectional pair, feedback loop) every method puts the true edges first, so AUROC is
   about 1 almost everywhere and says little. The separation shows the difference: in the chain,
   conditional Granger causality (`gc`) and `gpdc` keep the strongest spurious edge more than 300
   times below the weakest true one, `pdc` at least 9 times, but pairwise Granger causality
   (`gc_pairwise`) only 1.3 to 1.4 times and `dtf` below 1 (0.39 to 0.42).
2. **Pairwise Granger causality and DTF report indirect and shared-driver influence.** The indirect
   path A -> C of the chain scores about three quarters of the weakest true edge for `gc_pairwise`
   and 2.5 times the weakest true edge for `dtf`; in the common-driver network the pair of the two
   driven regions scores about 70 % (`gc_pairwise`). Conditioning (`gc`) or the partial measures
   (`pdc`, `gpdc`) remove these. Use `dtf` and the pairwise mode when "who reaches whom, directly or
   not" is the question, and `gc` or `gpdc` for direct influence.
3. **In the simple topologies the direction is right in every seed.** The exceptions are `dtf` in the
   three-region loop, where it gets two of the three edges right in nine of ten seeds (mean 0.70)
   because the loop gives every edge a strong indirect return path, `pdc` under a hidden driver
   (mean 0.90), and the limits of items 4 and 6.
4. **A zero-lag influence between memoryless regions cannot be found.** The separation of A -> B is
   0.56 to 0.59 for `gc`, `gc_pairwise` and `gpdc` and 0.02 to 0.04 for `dtf` and `pdc` (median over
   seeds), and the direction score is a coin flip (a mean of 0.85 against 0.75 expected by chance, 7
   hits in 10 seeds). With an oscillating source the same zero-lag edge is found in 6 of 6 seeds,
   through the source's own autocorrelation: the limit is "no temporal structure to exploit", not
   "zero delay".
5. **A hidden common driver creates a spurious edge.** `gc` still ranks the true edge first, but the
   spurious one is about 54 % as strong (separation 1.74 to 2.05); `gpdc` does better (3.8 to 4.7).
   `pdc` and `dtf` are shares of an outflow or inflow, and rank the spurious edge above the true
   one (separation 0.08 and 0.37, AUROC 0.6 and 0.8). No diagnostic can see this: it is a property
   of what was not recorded.
6. **Observation noise is the quiet failure.** At -10 dB the median separation falls to 1 (0.90 to
   1.06 depending on the method): a spurious edge is as strong as the weakest true one although the
   median AUROC is still 0.97. At -15 dB the median AUROC is 0.82 with a minimum of 0.5 (chance) and
   the selected order drops to 1. The only diagnostic that reacts is the order-stability check (added
   after the first version of this report): it warns in 6 of the 10 seeds at -15 dB, never at -10 dB;
   the sampling, stability and order-boundary checks did not warn in any run of this report. A
   clean-looking checklist is therefore no evidence that the data were clean.
7. **The six methods fail together under noise** because they share the same VAR fit: the choice
   between them does not repair a bad model, only the way the model is read.

The controls for zero-delay mixing (`trgc`, `imcoh`, `psi`) are scored here on the same
region-level series, which contain no mixing; these items say what they do when there is nothing
to control for. `trgc` and `psi` are net flows and are scored against the net truth, `imcoh` is
undirected (see Setup).

8. **TRGC keeps the direct-path behaviour of conditional Granger causality, with a smaller
   margin.** In the chain its separation is 4.2 to 340 (`gc`: above 1100), and it ranks and orients
   every simple topology correctly (AUROC 1, direction accuracy 1 in chain, common driver, collider,
   bidirectional pair and loop).
9. **A hidden driver hurts TRGC more than Granger causality.** The hidden region drives A and B with
   different delays, so A and B are related with a lag, which time reversal does not remove: the
   separation is 0.24 to 0.31 for `trgc` against 1.74 to 2.05 for `gc`, with AUROC 0.8 in every
   seed. TRGC removes zero-delay mixing, not unrecorded drivers.
10. **imcoh and PSI do not condition on other regions.** They report the indirect path of the chain
    (separation 0.70 to 0.81 for `imcoh`, 0.56 to 0.59 for `psi`), and `imcoh` the pair of the two
    driven regions in the common-driver network (0.87 to 1.01). Read them as pairwise measures, like
    `gc_pairwise` and `dtf`, not as direct influence.
11. **The controls are blind to zero-lag coupling by construction**: in `instantaneous` the separation
    is below 1 in 6 of 10 seeds for `trgc` (median 0.23), 8 for `imcoh` (0.88) and 10 for `psi`
    (0.05). That is what they are for: a zero-lag relation looks like mixing. Under observation noise `trgc` keeps a median
    separation of 1.43 at -10 dB where the VAR-based methods are at about 1, but falls to 0.10 at
    -15 dB; `imcoh` stays at about 0.76 and `psi` at 0.28 to 0.34 at both levels.
12. **Multitaper Granger causality (`mtgc`) reads like pairwise Granger causality.** It is pairwise
    by construction and reports the indirect path of the chain (separation 1.45 to 1.59; `gc_pairwise`
    1.30 to 1.40) and the shared-driver pair (1.09 to 1.21; `gc_pairwise` 1.42 to 1.47). Where it is
    right, its margin is smaller than that of the model-based estimates (collider 16 to 54 against
    above 360 for `gc_pairwise`), because an estimated spectrum leaves a positive floor on every
    non-edge. Under strong observation noise it held up better in these runs: at -15 dB its median
    separation is 0.90 and its median AUROC 0.95, against 0.18 and 0.82 for `gc_pairwise`, whose BIC
    order drops to 1 there; why is not established here (no model order to collapse is one
    candidate).
13. **PCMCI conditions away indirect and shared influence, and tests only lags up to `tau_max`.**
    In the chain, common-driver and collider networks its separation is 6.1 to 17.5 with AUROC 1 in
    every seed: the selected parents remove the indirect path and the shared-driver pair, as
    conditioning does for `gc`. Its default `tau_max` is the BIC order of the VAR model or 30 ms,
    whichever is longer (here 8 samples; the BIC order is 2). With the BIC order alone the
    bidirectional pair was weak (separation 0.05 to 1.13, median AUROC 0.80) because one edge has a
    delay of 12 ms (3 samples) that was never tested; with the 30 ms floor it is 0.67 to 2.2 with
    median AUROC 1.00, and at -15 dB the median AUROC rose from 0.80 to 0.95. The BIC order
    describes the fit of a model, not the longest coupling delay. In the feedback loop (weak couplings,
    all delays 2 samples) the separation stays about 1 (0.75 to 1.81).
14. **Transfer entropy (`te`) needs a target history as long as the signal's memory.** With the
    default history (the BIC order) it ranks and orients every simple topology correctly (median
    AUROC 1, direction accuracy 1) and, being pairwise, reports the indirect path of the chain
    (separation 1.55 to 1.88) and the shared driver (1.76 to 1.95) like `gc_pairwise`. A history of one
    sample on these oscillating regions got the direction wrong in most runs (direction accuracy 0 in
    the bidirectional pair and under the hidden driver, three seeds), which is why the default is the
    BIC order and not 1. Under strong observation noise the BIC order drops to 1 and the same failure
    returns: at -15 dB the median AUROC is 0.23 (0.1 to 0.6), worse than chance, i.e. the ranking is
    inverted, and at -10 dB it is 0.55. The raw estimate can also be slightly negative for a true edge.
15. **The renormalised PDC (`rpdc`) reads the VAR fit like `gc` and `gpdc`, not like `pdc`.** It is
    the direct influence at a frequency over its estimation error, not a share of an outflow, so
    the normalisation that hurts `pdc` is gone: in the common-driver network its median separation
    is 287 (`pdc` 1.26, `gc` 303), under the hidden driver it ranks the true edge first with a
    margin of 5.1 to 6.4 (`gpdc` 3.8 to 4.7, `pdc` 0.07 to 0.08), and the zero-lag edge scores like
    `gc` (median 0.59). It shares the fit, so it shares the failures of items 6 and 7 (median
    separation 0.90 at -10 dB, 0.19 at -15 dB). Its asymptotic threshold is calibrated in
    `docs/statistics.md`.

Not covered here, and covered elsewhere: mixing and leakage through a forward model (sensor level and
inverse-estimated regions: the simulation benchmark, and tutorial 9 for the controls), non-stationary
coupling (time-varying methods: the non-stationary validation), non-linear coupling and
conditional-independence methods (tutorials 11, 12 and 15).

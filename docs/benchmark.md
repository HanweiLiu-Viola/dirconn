# Benchmark on realistic head models

This page describes how the simulation benchmark is built and what each part of it measures. The
tables, and in front of them what they do and do not show, are in
[the benchmark results](benchmark_results.md).

The saved results (`benchmark_results/*.jsonl`) and the pages generated from them name the commit each
run was made at (for example `0de0d7f`). These identifiers belong to the development history that
preceded this public repository; they are kept so that the record of every run stays as it was
written, but they cannot be checked out here. The environment of every run (Python and library
versions) is in the header of its results file.

## What is compared

| Stage | Input of the estimator | What it isolates |
|---|---|---|
| 1. ROI series | the activity of each network node plus independent white noise at a stated SNR | the estimator, without any spatial mixing |
| 2. Sensors | one representative sensor per cortical node (a distinct sensor for each, chosen where the node's region is strongest), direct channels as recorded | the estimator after volume conduction, in the best case for the sensor level |
| 3. Source level | forward model, inverse solution, ROI extraction | the full pipeline |

The head model is the MNE `sample` subject: 7498 fixed-orientation sources, the 68 DK regions, and the
good channels of EEG (59), magnetometers (102) and gradiometers (203), which are separate modalities.
A region's topography is the mean of the gains of its sources. Sources follow the surface normal, so a
folded patch partly cancels: the topographies of the sources of one label correlate at about 0.06, and
regions differ 10 to 13 fold in how visible they are on the sensors (`HeadModel.visibility`). The
weakest region is at 0.22 to 0.30 of the median; none is invisible. EEG is average-referenced (see
below); the stored forward solution is not.

A single sensor sees a small part of a region: the strongest sensor holds a median 11 percent (EEG),
7 percent (magnetometers) and 9 percent (gradiometers) of a region's topography energy. Stage 2 is therefore not a typical sensor-space analysis; it reports the
`dominance` and the `purity` of each representative sensor so that this is visible in every result.

## Findings so far

**1. Unequal node power buries the weak node at sensor level.** In a chain every step amplifies the
power (node standard deviations 3.7, 16.4 and 82.3 in the test network). The sensor SNR is defined over
all sensors and is dominated by the strongest node, so the weakest node sat at about 3 dB at its
sensor while stage 1 gave it 20 dB. Without scaling, the sensor stage reached AUROC 0.75 and direction
accuracy 0.5 in every one of 10 seeds although the head model was nearly orthogonal; with every node
scaled to unit variance (`node_scale="unit"`, the default of `simulate_on_head`) it reached 1.0. Scaling a
node by a positive constant changes neither the network structure nor the truth. The purity of a sensor
must be weighted by node power: the leadfield alone gave 0.999.

**2. Model order decides the direction of a narrow-band influence in 1/f background.** Real pairs were
built from two real EEG channels of the sample recording (windows at different times, so the
background is independent): `target` is a causal 8 to 12 Hz filtered copy of `source`, delayed by 20 ms,
plus real noise. Over 24 random channel pairs and 40 s windows at 150 Hz:

| Estimator | Direction recovered (strength 1) | Direction recovered (strength 3) |
|---|---|---|
| GC, order 8 | 7 of 24 | 8 of 24 |
| GC, order 32 | 24 of 24 | 24 of 24 |
| GC, order 64 | 24 of 24 | 24 of 24 |
| GC, BIC up to 64 (chose about 63) | 24 of 24 | 24 of 24 |
| DTF, order 64 | 24 of 24 | 24 of 24 |
| gPDC, order 64 | 24 of 24 | 24 of 24 |

At order 8 the direction is reversed more often than chance (0.29 and 0.33 against 0.5). A
narrow-band influence on a long-memory background cannot be summarised by a short AR model, and the
other channel inherits the memory. The wrong answer does not look like a failure: at order 8 the
scores are small and the other diagnostics do not warn. Reproduce with `scripts/real_pair_study.py`.

**The order-stability check.** Because of this, GC, DTF, PDC and gPDC now refit the model at longer
orders (2, 4, 8, 16 and 32 times the chosen one, as far as the data allow) and use the exact F test
of Granger causality. A pair is flagged when, at some longer order, one direction is significant at
0.001 and the other is not significant at 0.05, while the chosen order favours the opposite
direction (`check_order_stability`, on by default; `dirconn.order_stability`). Measured on the real
pairs above (alpha 0.001):

| | Result |
|---|---|
| Independent real channel pairs, orders 2 and 8 | flagged 0 of 30 and 0 of 30 |
| Direction wrong at order 4, strength 1 / 3 | caught 13 of 17 / 6 of 10 |
| Direction wrong at order 8, strength 1 / 3 | caught 17 of 23 / 14 of 19 |
| Direction right (90 cases) | flagged 1 |
| Known scenarios of `docs/method_validation.md` (30 runs) | flagged 1: zero-lag coupling, pairwise GC |
| Extra cost | 2 to 4 times the fit it accompanies |

It misses about a quarter to a half of the reversed cases, so a clean result does not prove the
direction. The zero-lag case is a pair whose direction is not identifiable by construction. The
resampling tests (`surrogate_test`, `bootstrap`, `contrast_test`) switch the check off for the
resampled fits and keep it for the fit on the data. The F tests are not valid for non-stationary
data; the check assumes the stationarity diagnostic has been read.

**3. Stage 1 versus stage 2 on a clean head.** With node power equalised, the sensor stage matches the
ROI stage on a nearly orthogonal head model (AUROC 1.0 at 20 and 30 dB, with and without background
sources). Real head models are not orthogonal; the source-level stage and the sweeps measure how much is lost.

**4. The stored EEG forward solution is not average-referenced.** The column sums of its gain over the
channels are about 3.7 column norms: it is relative to an infinite reference and carries a large
common-mode component. Used as it is, every simulated electrode would share a signal that no
recording has, and the topographies of different regions would look more alike than they are. The
head model therefore average-references the EEG gain; the numbers above are for that. A beamformer
built from the stored forward needs the average-reference projector in its `info`: its weights then
sum to zero over the channels and give the same source estimate as MNE's own application.

**5. A beamformer's region series depends on how the region is summarised, and no summary is right
for every kind of source.** The sensor data of a chain of three regions were generated with three
source models (a focal source at one vertex, a partly coherent patch in which every vertex carries
0.7 of the node plus its own 1/f activity, and a fully coherent patch), then each region was
extracted. Median |correlation| of the extracted series with the true node activity, 10 dB, gradiometers
and EEG together (coupling 0 and 0.4 alike unless stated):

| Extraction | Focal | Partly coherent patch | Fully coherent patch |
|---|---|---|---|
| LCMV filter per vertex, first principal component of the region (`pca_flip`) | 0.03 to 0.09 | 0.02 | 0.01 |
| LCMV, mean of the sign-flipped vertices | 0.28 to 0.56 | 0.10 to 0.18 | 0.09 to 0.11 |
| LCMV, the vertex at the centre of the region | 0.99 (0.90 at coupling 0.4) | 0.05 | 0.02 |
| LCMV, one beamformer for the whole region (its mean topography as the lead field) | 0.01 to 0.06 | 0.97 (0.89 to 0.91 at 0.4) | 0.98 to 0.99 (0.90 at 0.4) |
| Minimum norm, `pca_flip` | 0.98 to 0.99 | 0.95 to 0.96 | 0.98 to 0.99 |

The LCMV filter itself is right: at the vertex where a single source is active its output correlates
0.99 to 1.00 with the truth at 10 dB. What fails is the summary of a region of 90 to 180 vertices: the
vertices away from the source carry noise, and their principal component is noise; with a coherent
patch the vertex-wise filters cancel each other. Correlation between the nodes is not the
cause (coupling 0 gives the same 0.01). Coupling of 0.4 costs the best extractions about 0.09.
Minimum norm is robust to the extent of the source. Sources are fixed-orientation and the minimum-norm
filters here use no depth weighting.

## Stage 3 and the leakage between regions

`evaluate_stages(..., inverse={})` adds stage 3: the sensors of the head model go through the inverse
method (default minimum norm, `pca_flip`) for every cortical node, and the estimator is applied to the
region series (direct nodes as recorded). On the easy test network (a chain of three regions, coupling
0.4, 6000 samples at 100 Hz, Granger causality of order 16, partly coherent sources) all three stages
recovered the network perfectly (AUROC 1.00 and direction accuracy 1.00 over 5 seeds, gradiometers and EEG,
10 dB and 0 dB). That setting saturates; the stages separate only in harder conditions, which the
sweeps cover.

`dirconn.inverse.crosstalk` measures how much one region leaks into another's series. On the sample head
model the nearest tenth of the region pairs leaks 0.38 to 0.69 of the own response and the farthest tenth
0.02 to 0.08 (minimum norm and LCMV, gradiometers and EEG); the Spearman correlation between the
distance of the region centres and the leakage is -0.34 to -0.47. A neighbouring region therefore
leaks into a node's series at about half its own strength, whatever the inverse method.

## Defaults of the benchmark

Two defaults were chosen after the measurements above.

- **Inverse method: minimum norm with `pca_flip`.** It is the only region summary that recovered the
  node under every source model (finding 5). LCMV, dSPM and eLORETA are selectable.
- **Source model: a partly coherent patch** (`simulate_on_head(source_model="partial")`). Every source of
  a region carries `c * node + sqrt(1 - c**2) * its own 1/f activity` with unit-variance parts and
  `c = 0.7`, so two sources of a region correlate at 0.49 and the sensors see their mean. The
  coherent part enters as `c` times the region's mean topography; the local part is activity of the
  region that is unrelated to the network and is counted in the signal. `source_model="focal"` (one source,
  the region's centre source) and `"patch"` (all sources identical) are the two extremes and become a
  sweep axis.

With the partly coherent default and minimum norm, the correlation of the extracted series with the
true node activity at 10 dB over 8 seeds and 3 nodes is: gradiometers median 0.90 (minimum 0.81), EEG 0.93
(0.89), magnetometers 0.88 (0.73); with whole-label sources it was 0.95 to 0.99. The coherence of the
region's own sources, not the inverse method, limits it.

## Scenario library

The scenarios of the sweeps are YAML files in `src/dirconn/scenarios/` (format: `dirconn.scenario`,
`version: 1`), loaded with `load_scenario("chain")`; `list_scenarios()` names them. A file holds the
network, the own dynamics of the nodes, defaults (100 Hz, 10 dB, 120 s) and the hidden nodes; difficulty
(SNR, length, coupling, source extent) is **not** in the files, it is an axis of the protocols.

The hand-written scenarios use a fixed pool of four regions, chosen by `scripts/pick_regions.py`: a
region qualifies when its visibility (norm of its topography relative to the median region) is at least
1.0 on EEG, magnetometers and gradiometers and at most 1.8 on the least visible of the three, and the
regions are taken in order of decreasing visibility when their centres are at least 60 mm from the ones
already taken: `transversetemporal-rh`, `cuneus-rh`, `transversetemporal-lh`,
`rostralanteriorcingulate-lh`. Equal-ish visibility keeps the baseline from being decided by an
accidentally weak region. The pool is small because of the distance rule (60 mm leaves four), which is
why scenarios with more than four nodes are the generated random graphs.

Shipped so far (hand-written): `chain`, `common_driver`, `collider`, `bidirectional`, `feedback_loop` (weak, 0.05:
a loop of resonant regions becomes unstable at much lower coupling than a one-way edge), `hidden_common_input`, `delay_1`/`delay_5`/`delay_zero_osc` (oscillators only: leakage at
zero delay is visible to a lagged method only when the source has a past), `band_theta`/`band_alpha`/`band_beta`/`band_gamma` (the gamma band stops at 45 Hz because 100 Hz
sampling ends at 50 Hz). The band and delay files have an unconnected node that provides null edges.

### Leakage, distance and artifact scenarios

The rules are in `dirconn.regions` (plain-array functions, tested on made-up data) and applied to the
sample head model by `scripts/pick_regions.py leakage` and `distance`; a sample-data test checks that
the shipped files are what the rules give. The leakage matrix is `dirconn.inverse.crosstalk` with the
defaults (minimum norm, `pca_flip`, 10 dB probe, seed 0); `matrix[y, x]` is the leak of region `x` into
the series of region `y`, relative to the response of that series to its own region.

- `leakage_adjacent`: Z -> X is the only edge, Y is X's neighbour that receives the strongest leak.
  A pair qualifies when its leak is above the 90th percentile of all region pairs on EEG (1.47) and on
  gradiometers (0.98) and both regions have visibility of at least 1.0; the pair with the highest
  smaller percentile rank wins: `cuneus-lh` -> `pericalcarine-lh`, 10 mm apart, leak 3.47 on EEG and 20.27
  on gradiometers (median leak over all pairs: 0.18 and 0.12). Values above 1 mean the neighbour's series
  holds more of the leaking region than of itself, because the leaking region is much more visible
  (1.6 against 1.0). Only 8 ordered pairs qualified (of 4556, with both regions at visibility 1.0 or more). Z is the most visible region with a
  visibility inside the pool window (at most 1.8) that is at least 60 mm from both and leaks into
  neither above the 90th percentile: `transversetemporal-rh` (76 and 78 mm).
- `distance_near`/`distance_mid`/`distance_far`: eligible regions have visibility 0.9 to 1.5 (23 of
  68); the distances of the 121 same-hemisphere pairs among them split at 45 and 67 mm; from each third
  the pair closest to the third's median is used: `parsorbitalis-rh` -> `temporalpole-rh` (34 mm),
  `posteriorcingulate-rh` -> `rostralanteriorcingulate-rh` (58 mm), `frontalpole-rh` ->
  `parahippocampal-rh` (91 mm). The leakage and the visibility are *not* held equal across the three,
  only the visibility window is; the distance scenarios therefore measure distance together with
  whatever leakage comes with it.
- `stimulation_artifact`: the baseline chain plus an `artifact` block (optional in version 1 files:
  `rate_hz`, `freq_hz`, `decay_ms`, `amplitude`, `anchor`, `n_sensors`). A stimulus every `1 / rate_hz`
  seconds (a whole number of samples) adds the same damped sine to the `n_sensors` sensors where the
  topography of node `anchor` is strongest, with that topography's signs, at `amplitude` squared times
  the power of the noise-free signal on those sensors. `dirconn.scenario.stimulation_artifact` makes the
  array; adding it to the sensor data is the runner's job (task 12). It does not exist at region level.

### Random sparse networks

`random_5`, `random_10` and `random_20` come from `dirconn.scenario.random_scenario` (seeds in
`RANDOM_SEEDS`; `scripts/make_random_scenarios.py` regenerates the files exactly). Per graph:
`round(1.5 * n)` edges on distinct unordered pairs (so a pair never has edges in both directions),
all but two from an earlier to a later node of a hidden order, so the graph is acyclic apart from two
feedback edges; strengths 0.2 to 0.4 (positive), delays 1 to 3 samples (10 to 30 ms), every node an
oscillator of 8 to 30 Hz with a 4 Hz bandwidth. A graph is thrown away and drawn again when its VAR
spectral radius is 0.95 or more, or when it has no cycle at all (a feedback edge does not always close
one). The regions are drawn from those with visibility of at least 0.5 on all three modalities (52 of
68), **without** the 60 mm spacing of the hand-written pool: twenty regions do not fit into that, and
neighbouring regions are part of the difficulty of a big network. For five regions 1.5 edges per node
is 8 of the 10 possible pairs, which is not very sparse.

Rejections of the shipped files (counted by the generator, also in each file's description):
`random_5` 0 for the spectral radius and 3 for lack of a cycle, `random_10` 4 and 3, `random_20` 0 and
0. A node on its own has a radius of 0.882 (4 Hz bandwidth at 100 Hz), so the limit of 0.95 only
bites through cycles; 4 of the 13 graphs drawn in total for the three files exceeded it (4, 8 and 1 draws for the three files, counting the accepted graph). The
numbers come from the three shipped seeds, not from a study of the generator.

## Protocols, results files and the budget gate

A protocol (`protocols/<name>.yaml`, `version: 1`, `dirconn.protocol`) lists axes; the runner
(`scripts/run_benchmark.py`, `dirconn.sweep`) takes their Cartesian product.

- **A task is one simulation**: a scenario, an SNR, a data length, a coupling strength, a source model,
  a modality and a seed number. The estimators are not part of the task: it simulates once through the
  head model and applies every estimator (and every inverse method) of the protocol to the same data
  at the three stages. So adding estimators does not multiply the simulations.
- **Axes**: `scenarios`, `snr_db`, `duration_s`, `coupling`, `source_model`, `modality`, plus
  `estimators`, `inverse`, `n_seeds`. An axis left out takes the scenario file's own default. `exclude`
  drops combinations. `coupling` is an **absolute** strength that replaces every edge (sign kept) and is
  only allowed for scenarios whose edges all share one strength; `0` removes all edges (a zero
  network: AUROC and friends are NaN by design, only false positives can be counted).
- **Estimator labels** are `<method>/<order>`: methods `gc`, `gc_pairwise`, `dtf`, `pdc`, `gpdc`; order
  `bic` (BIC up to the smaller of 40 and the order at which the model would break the rule of one
  coefficient per ten observations) or `fixed` (`round(0.2 * sfreq)` samples, 20 at 100 Hz, findings
  A24). The order-stability check is off in sweeps (it multiplies the cost of a fit by 2 to 4).
- **Seeds** come from the content of a task (a 60-bit hash of its readable key, through
  `RNGController`), not from its position: adding a value to an axis, or a seed, leaves every existing
  task's seed alone. The task id also contains a hash of the scenario as simulated, so an edited
  scenario file gives a new id (same seed, results not reused).
- **Results** go to `benchmark_results/<protocol>.jsonl`: a header (protocol settings that decide the
  numbers, versions, the gate measurement), one line per finished task, a summary line. Only the
  published `full` results are meant for git. A run resumes: finished tasks are skipped, failed ones are
  retried, a cut-off last line is ignored. A resume is refused (with the difference named) when the
  estimators, inverse methods, settings, master seed or a package version differ; the grid and the
  budget may change. A failing task is recorded with its traceback and the run goes on; the exit
  code is 1 when a task failed.
- **Hidden nodes and artifacts**: a scenario's `hidden` nodes take part in the choice of sensors and
  in the inverse step (they are in the head and leak into the others) but their series are not given
  to the estimators, and the truth is the subgraph between the nodes that remain, so an edge that a
  hidden common cause creates counts as a false positive. A stimulation `artifact` is added to the
  sensor data (`dirconn.benchmark.add_artifact`) after the SNR is fixed; the region-level stage never
  sees it and `snr_db` does not count it.
- **Time-varying scenarios are refused** when a protocol is loaded (`Network.has_intervals`: `burst`,
  `direction_flip`, `switch_on`, `switch_off`): the stages are scored against one static truth, so
  their numbers would be about another experiment. They belong to `scripts/run_nonstationary.py`.
- **Cards**: per estimator and stage the output of `evaluate_stages` (AUROC, AUPRC, direction
  accuracy, separation, selected order, diagnostics that warned) plus, for the spectral estimators,
  the AUROC at every frequency (`auroc_by_frequency`, four decimals, NaN where the truth has no edge
  or only edges), and the Spearman agreement of the stages. The summary table shows the mean AUROC,
  the mean direction accuracy and the median separation over the tasks with a defined AUROC.

### The saturation guard

Findings A29: in easy settings all three stages score perfectly and nothing can be said about the
methods. A protocol with `saturation_guard: true` (only `full`) must therefore reach from the easy
to the hard on the chain. For the sensor and the source stage separately, over the grid points
(SNR, length, coupling, source model, modality) and averaged over the seeds, the **best** estimator
must be below 0.99 mean AUROC somewhere (a hard point) and at or above it somewhere (an anchor).
The best estimator decides because some methods are never perfect on a chain for structural reasons
(DTF reports the indirect path A -> C: 0.875 to 0.94 where the others are perfect). The region level
is not judged. A failing run exits with code 3 and the report says which stage and where.

Where the chain breaks (a probe of 96 tasks, 4 seeds, EEG, best of conditional GC at both orders and
gPDC; mean AUROC at the ROI, sensor and source stage):

| source model, coupling | SNR | 60 s | 30 s |
|---|---|---|---|
| partial, file (0.4) | -5 dB | 1.00, 1.00, 1.00 | 1.00, 1.00, 1.00 |
| partial, file (0.4) | -10 dB | 0.97, 1.00, 0.97 | 0.97, 0.94, 0.81 |
| partial, file (0.4) | -15 dB | 0.66, 0.84, 0.53 | 0.62, 0.75, 0.72 |
| partial, 0.1 | -5 dB | 0.94, 0.91, 0.88 | 0.72, 0.91, 0.91 |
| focal, file (0.4) | -5 dB | 1.00, 0.81, 0.88 | 1.00, 0.62, 0.97 |
| focal, 0.1 | -5 dB | 1.00, 0.69, 1.00 | 0.69, 0.78, 1.00 |

In `quick` (60 s, 2 seeds) the chain was perfect at 0 dB for every estimator at every stage, and
at 10 dB only DTF was not; judging by that, the grid SNR 10 and 0 dB, 60 and 120 s (the second
version of `full`) would have failed the guard. That version was never run. With four seeds a mean AUROC of
the three-edge chain moves in coarse steps, so single cells of this table are rough.

First run of `full` (SNR 0, -5, -10 dB, lengths 30 and 60 s): the chain's best estimator is 1.000 at
0 dB and 30 s, and 0.963 (sensor) and 0.887 (source) at -10 dB and 30 s, so the guard passes.

### Leakage studies

Both use `store_scores: true`, which keeps the band-reduced score matrix of the analysed nodes in
every stage card (six decimals; off elsewhere because it makes the files larger), and are analysed by
`dirconn.leakage` from the saved results alone. A protocol of `kind: region_pairs` has no `scenarios`
and no `coupling` axis: `pairs` (`n`, `seed`, `regions`) gives `n` random pairs of the listed regions
(the first `n` of one seeded permutation of all pairs, so asking for more keeps the pairs already
chosen), each a two-node network without any edge, a 10 and a 20 Hz oscillator, coupling 0. The
results header of such a run holds the crosstalk matrix of every modality (`dirconn.inverse.crosstalk`,
defaults, seed 0).

**The neighbour** (`leakage_adjacent`: `Z -> X` is the only edge, `Y` is the neighbour into which `X`
leaks; SNR 10 dB, 60 s, 10 seeds, one estimator of each kind at BIC order). A score is read relative
to the score of the true edge `Z -> X` of the same card (0: nothing, 1: as strong as the real edge).
Mean over seeds; the range over seeds is narrow (for example 1.36 to 1.48 for conditional GC on EEG).

| stage | relative score of the false edges into Y | AUROC |
|---|---|---|
| region level | 0.00 to 0.01 for every estimator and modality | 1.00 everywhere |
| sensors, EEG | 0.14 to 1.08 | 1.00 (conditional and pairwise GC, DTF), 0.88 (gPDC), 0.80 (PDC) |
| sensors, gradiometers | 0.13 to 0.83 | 1.00 everywhere |
| sources, EEG | `Z -> Y` 1.40 to 2.07, `X -> Y` 0.43 to 2.94 | 0.80 (conditional and pairwise GC), 0.40 (DTF, PDC, gPDC) |
| sources, gradiometers | `Z -> Y` 0.16 to 0.38, `X -> Y` 0.33 to 1.03 | 1.00 (GC, DTF), 0.98 (PDC), 0.86 (gPDC) |

The direction accuracy of the true edge is 1.00 in every row. Per estimator, conditional GC at
the source level on EEG: `Z -> Y` 1.40, `X -> Y` 0.62; pairwise GC 2.07 and 0.43; DTF 1.78 and
2.58; PDC 1.45 and 2.48; gPDC 1.44 and 2.94. These are numbers for one pair of neighbours on one head model, and the
two modalities also differ in how visible the two regions are (1.6 and 1.0), so the table does not
say why EEG is worse than the gradiometers. SNR 0 dB and the fixed-order variants were cut to fit the
budget (the gate estimated 0.15 h for all ten labels at two SNRs; the shipped grid has 20 tasks).

**Pairs against crosstalk** (`leakage_pairs`: 45 random pairs of the 52 regions with a visibility of at
least 0.5 on all three modalities, EEG and gradiometers, SNR 10 dB, 60 s, one seed per pair,
conditional GC and gPDC at BIC order). The false edge `x -> y` of a pair is paired with `|crosstalk[y, x]|`,
the leak of `x` into the series of `y`, and `y -> x` with `|crosstalk[x, y]|`: 90 cells per modality
and stage. Spearman correlation, percentile bootstrap over pairs, 95% interval:

| stage | conditional GC | gPDC |
|---|---|---|
| region level (control) | EEG 0.05 [-0.15, 0.23], grad 0.11 [-0.13, 0.32] | EEG 0.05 [-0.15, 0.23], grad 0.04 [-0.19, 0.26] |
| sensors | EEG 0.44 [0.25, 0.60], grad 0.51 [0.33, 0.66] | EEG 0.45 [0.27, 0.60], grad 0.52 [0.35, 0.66] |
| sources | EEG 0.62 [0.45, 0.73], grad 0.68 [0.54, 0.80] | EEG 0.62 [0.46, 0.74], grad 0.66 [0.52, 0.79] |

Limits: 45 pairs, because the gate estimated 0.23 h for the planned 150 and the share is 0.06 h; the
pairs are drawn uniformly, so the strongly leaking end is thin (the median leak over all pairs is
0.18 on EEG and 0.12 on gradiometers); the crosstalk of a pair is also related to how visible its
regions are and how strong their activity is, which was not controlled, so the correlation says that
false edges follow the crosstalk matrix, not that the leak alone causes them; one head model.

### Significance study

`significance` (`kind: significance`, `dirconn.significance_study`) applies the tests of
`dirconn.significance_test` to simulated region-level data: node activity of a scenario plus white
observation noise, as at stage 1, without a head model (so no modality, source model or inverse
axis, and no head loading in the gate). Every task is one dataset; every test of `tests`
(`analytic`, `surrogate:<generator>`) is applied to it, and the p-value matrices are kept (six
decimals), so levels and corrections are applied afterwards from the saved results. Four worlds of
the chain, 25 datasets each, 30 s: no edge (coupling 0) and the chain (coupling 0.4), each with the
observation noise at 10 dB and at 40 dB (a control for what the noise does). Statistic: conditional
Granger causality at a fixed order of 0.2 s; surrogate test: 39 phase-randomised surrogates, whose
smallest p-value is 1/40 = 0.025, so level 0.01 cannot be reached (n/a), and p <= 0.05 has
probability exactly 0.05 under the null. Rates are over cells (150 null cells per row in the null
world, 100 null and 50 edge cells in the chain); the intervals are percentile bootstraps over
datasets because the six cells of a dataset share its data; a share of datasets has a Wilson interval.

Rate of rejections of a cell without an edge (uncorrected) and share of datasets with at least one
false rejection after Benjamini-Hochberg over the six cells, q = 0.05:

| world | test | rate at 0.05 | rate at 0.10 | mean p (expected) | datasets with a BH false rejection |
|---|---|---|---|---|---|
| no edge, 10 dB | analytic | 0.053 [0.027, 0.087] | 0.127 [0.080, 0.167] | 0.453 (0.500) | 0.080 [0.022, 0.250] |
| no edge, 10 dB | surrogate | 0.053 [0.020, 0.093] | 0.100 [0.060, 0.140] | 0.469 (0.512) | 0.000 [0.000, 0.133] |
| no edge, 40 dB | analytic | 0.047 [0.013, 0.080] | 0.100 [0.053, 0.160] | 0.513 (0.500) | 0.000 [0.000, 0.133] |
| no edge, 40 dB | surrogate | 0.053 [0.027, 0.080] | 0.093 [0.047, 0.140] | 0.530 (0.512) | 0.000 [0.000, 0.133] |
| chain, 10 dB | analytic | 0.150 [0.100, 0.210] | 0.230 [0.160, 0.310] | 0.354 (0.500) | 0.400 [0.234, 0.593] |
| chain, 10 dB | surrogate | 0.060 [0.020, 0.100] | 0.110 [0.060, 0.170] | 0.498 (0.512) | 0.000 [0.000, 0.133] |
| chain, 40 dB | analytic | 0.030 [0.000, 0.060] | 0.090 [0.040, 0.140] | 0.533 (0.500) | 0.040 [0.007, 0.195] |
| chain, 40 dB | surrogate | 0.010 [0.000, 0.030] | 0.020 [0.000, 0.050] | 0.700 (0.512) | 0.000 [0.000, 0.133] |

In the chain the null cells are the four cells that are not an edge (the reverse directions and
A -> C, which conditional Granger causality should not report). At level 0.01 the analytic test
rejected 0.013 [0.000, 0.033] and 0.007 [0.000, 0.020] of the null cells without an edge (10 and
40 dB) and 0.050 [0.010, 0.090] and 0.010 [0.000, 0.030] in the chain.

Power: both tests found both true edges in every dataset, at every level that is reachable and at
both noise levels (1.000 [1.000, 1.000]), so the chain at coupling 0.4 and 30 s is too easy to say
anything about power (the same saturation as in the probes of the protocol design). After BH the analytic test also found
them all (false-discovery proportion 0.140 at 10 dB, 0.013 at 40 dB); the surrogate test with 39
surrogates found **none**: its smallest p-value of 0.025 adjusted over six cells is 0.15, so BH can
reject only when at least three cells reach it (`cells needed for BH` in the summary), and the chain
has two edges. That is the resolution of the test, not a lack of power; about 119 surrogates would
let a single cell through.

Limits: 25 datasets per world, so the intervals are wide (the share of datasets with a BH false
rejection is 0 of 25 for most rows, [0.000, 0.133]); the chain at 30 s on one head-free setting;
conditional Granger causality at a fixed order only (the BIC order and the other estimators were
not tested); one surrogate generator (phase randomisation; circular shift not run); coupling 0.4
only, which saturates the power. The study cost 0.01 h; the gate estimated 0.05 h, because the
probe of the gate runs fresh worker processes and for a task of 0.8 s their start-up dominates
(efficiency 0.07): a cautious estimate for short region-level tasks.

### Noise study (the empty room)

`noise` is a task axis of the protocols with head-model tasks: `simulated` (the default: ten 1/f
background sources plus white noise), `white` (white noise only, no background) and `empty_room`
(the real recording `ernoise_raw.fif` of the sample dataset, no background). The three arms of a
seed share their sources (the noise is not part of the seed), so the difference between them is the
noise alone; `empty_room` exists for magnetometers and gradiometers only, because the dataset has
no empty-room recording for EEG. A task names its noise in its key only when it is not `simulated`,
so the ids and seeds of every earlier task are unchanged (a test holds the old values).

The recording is 110 s at 600 Hz. `dirconn.emptyroom` takes the channels of the head model in its
order, resamples to 100 Hz (MNE's FFT resampling), applies **none** of the three projectors stored in
the file and does no signal-space separation, and gives consecutive blocks of the length of the data
(three of 30 s). `simulate(noise_series=...)` uses it instead of the simulated noise and rescales it
so that the realised sensor SNR is exactly the target: only the shape of the noise in time and across
sensors differs, not its size. The two sensor types have very different noise (measured at 100 Hz:
mean absolute correlation between the series of two channels 0.57 for the magnetometers and 0.04 for
the gradiometers, lag-1 autocorrelation 0.89 and 0.55).

`noise`: chain, -10 dB (the point where the chain stops being perfect for EEG), 30 s,
partly coherent sources, conditional GC and gPDC at BIC order, nine seeds per arm and modality.
The window of a task is its seed index modulo 3, so the nine seeds are three source realisations on
each of three windows: **three independent noise samples, not nine**. Mean AUROC over the seeds
(n = 9) and the difference between arms taken seed by seed (mean, sd):

| modality, stage | simulated | white | empty room | empty room minus simulated |
|---|---|---|---|---|
| both, region level (control) | 0.819 grad, 0.806 mag | the same | the same | 0.000 (sd 0.000) |
| gradiometers, sensors and sources | 1.000 | 1.000 | 1.000 | 0.000 |
| magnetometers, sources | 1.000 | 1.000 | 1.000 | 0.000 |
| magnetometers, sensors, GC | 0.861 | 0.708 | 0.514 | -0.347 (sd 0.150) |
| magnetometers, sensors, gPDC | 0.875 | 0.708 | 0.472 | -0.403 (sd 0.185) |

For the magnetometers at the sensor stage the other differences were white minus simulated -0.153
(sd 0.205) for GC and -0.167 (sd 0.217) for gPDC, and empty room minus white -0.194 (sd 0.178) and
-0.236 (sd 0.159). The region-level result is identical in the three arms because the stage-1
observation noise does not depend on the sensor noise and the sources are shared.

Limits: **most cells are saturated** (every gradiometer cell and the magnetometer sources are 1.000
in all arms), so at -10 dB only the magnetometer sensor stage says anything; a harder SNR would be
needed for the rest. Three independent noise windows; one scenario (the chain); one SNR; the
projectors of the recording were not applied, so the magnetometer noise is the unprojected one; a
fourth arm with the empty-room noise *and* the simulated background (the nearest thing to a real
recording) was not run; no EEG. The cause of the magnetometer result is not established: no
control was run that could refute an explanation.

### The budget gate

`run_benchmark.py <protocol> --gate-only` times one task per network size and modality (heads loaded
first; time scaled linearly to 120 s of data; the slowest modality stands for the size), then runs the
heaviest task in every worker at once to **measure** the parallel efficiency, and adds the startup
time. The run refuses above the protocol's `budget_hours` unless `--force`.

The first version assumed an efficiency of 0.8. Checked against a real run it was wrong: `quick` on 13
workers was estimated at 0.01 h and took 0.057 h (207 s). The tasks themselves ran up to 7 times slower
when many ran at once (one task: 6 s alone, 16 s with 4 workers, 27 s with 7, 42 s with 13; 13 workers
were only 1.9 times as fast as one). Tasks were single-threaded (CPU/wall 0.8, one MKL
and one OpenMP thread), so this is not oversubscription; each worker holds 0.6 to 0.8 GB and the
machine (Core Ultra 5 125U, 15 W class, 15.4 GB, about 4.7 GB free) has a memory and a power limit,
which of the two matters more was not separated. With the measured efficiency (0.14) the estimate for
`quick` on 13 workers is 0.06 h, against 0.057 h measured. The default is now at most 6 workers.

Assumptions that remain: time linear in the data length; the efficiency measured on the heaviest
task holds for the others; startup counted once. Checked so far by actual run times: `quick` (0.06 h
estimated, 0.057 h measured, 13 workers), the 96-task probe (0.03 h and 0.03 h, 4 workers) and
`full` (0.36 h and 0.34 h, 6 workers).

The final run of the ten published protocols (code `0de0d7f`, 6 workers). *Estimate* is the gate
estimate at the start of the run that wrote the results file; *measured* is the wall-clock time of that
run. The same table is generated into `docs/benchmark_results.md` from the results headers.

| protocol | tasks | budget share | estimate | measured |
|---|---|---|---|---|
| `quick` | 24 | 0.2 h, daily regression, not counted | 0.06 h (13 workers) | 0.057 h |
| `full` | 300 | 0.42 h | 0.37 h | 0.55 h |
| `order` | 200 | 0.34 h | 0.28 h | 0.44 h |
| `inverse` | 32 | 0.22 h | 0.29 h (forced) | 0.28 h |
| `coupling` | 360 | 0.37 h | 0.33 h | 0.40 h |
| `extent` | 240 | 0.25 h | 0.30 h (forced) | 0.23 h |
| `size` | 60 | 0.18 h | 0.14 h (forced) | 0.21 h |
| `leakage_adjacent` | 20 | 0.04 h | 0.03 h | 0.03 h |
| `leakage_pairs` | 90 | 0.06 h | 0.04 h | 0.04 h |
| `significance` | 100 | 0.06 h | 0.06 h | 0.01 h |
| `noise` | 54 | 0.05 h | 0.03 h | 0.02 h |
| **all ten** | | **1.99 h** | | **2.22 h** |

**Add-on protocols**, run after the ten (decided on 2026-10-05): they do not count against
the 2 h, each has a budget of its own of at most 0.1 h, and the results of the ten were not touched. They
ran at later commits than the ten (the report says so).

| protocol | tasks | budget | estimate | measured | what it adds |
|---|---|---|---|---|---|
| `noise_hard` | 108 | 0.1 h | 0.05 h | 0.03 h | the noise study at -15 and -20 dB (`noise` has -10 dB, where most cells were at 1.000) |
| `significance_power` | 320 | 0.1 h | 0.21 h (forced) | 0.06 h | the significance study with coupling 0.1 and 0.2, 40 datasets and 99 surrogates (so that Benjamini-Hochberg can reject with two true edges) |

`significance_power` was first drafted with 50 datasets and two estimators (400 tasks, gate estimate
1.3 h), then shrunk to 40 datasets and `gc/fixed` only (gate estimate 0.22 h, still above the budget).
The gate estimate of this kind of task is again several times too high (the first study: 0.06 h against
0.01 h measured; this one 0.21 h against 0.06 h), because the tasks are short and the parallel efficiency
it measures is 0.07 to 0.09. The run was started deliberately with `--force`.

What the final run showed about the gate:

- The shares add up to 1.99 h (a test keeps the sum at or below 2 h), but the measured time of the ten
  protocols is 2.22 h: **the budget was exceeded by 11%.** Four protocols ran longer than estimated
  (`full` by a factor of 1.5, `order` 1.6, `coupling` 1.2, `size` 1.5), three clearly shorter
  (`extent`, `significance`, `noise`) and three about as estimated (`inverse` and the two leakage
  protocols).
- The estimate of one protocol is not stable between invocations on this machine. `size`: 0.25 h when it
  was refused, 0.14 h at the start of the run that wrote its file. `inverse`: 0.24, 0.17 and 0.29 h in
  three invocations. `extent`: 0.28 and 0.30 h. The gate times one task per size and modality and one
  parallel probe, so it inherits the variation of a 15 W laptop CPU.
- `extent`, `size` and `inverse` were above their share at the start and were run deliberately with
  `--force`, each after a look at its estimate. The results header does not record the flag.
- A computer that goes to sleep during the gate gives an absurd estimate (33 h for `inverse`) and the run
  refuses; this is the gate doing its job, but a sleep during a run inflates the recorded times. Keep
  the machine awake.
- The checkout must not be switched to another branch while a protocol runs: the header records the
  commit at the moment it is written, and the worker processes read the source files from disk when
  they start. One run of `inverse` was discarded for this (its header said `d2e98d1`, the code that ran
  was a mixture) and redone in a separate git worktree. Run the final protocols from a worktree of their
  own.

How the grid of `full` came about. A first `full` (6 scenarios including a 10-region network, SNR 10, 0,
-5, lengths 60 and 240 s, two source models, EEG and gradiometers, 1440 tasks) was estimated at 15 h. The
second version (five scenarios, SNR 10 and 0, lengths 60 and 120 s, EEG, 200 tasks, 0.91 h) would have
failed the saturation guard; the shipped one has SNR 0, -5 and -10 dB and lengths 30 and 60 s (300
tasks: shorter data are cheaper, which pays for the third SNR). The gradiometers, the coupling axis, the
focal and patch sources and the 10-region graph were taken out of `full` and run as the small protocols
`coupling`, `extent` and `size` with the five BIC settings only. What is not covered at all:
gradiometers in `full`, networks of 20 regions, data length 120 s (and 240 s outside `order`). The
random graph of 10 regions cost 44 to 57 s per 60 s task against 10 to 21 s for the others.

## Caveats

- The task recording is not resting state: it holds evoked responses and real connectivity. It is used
  as realistic background, with the signal and the noise taken from different times so that the truth
  of a constructed pair stays exact.
- Results here are on a handful of configurations. They show what can go wrong; the sweeps
  say how often.

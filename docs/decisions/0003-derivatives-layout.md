# 0003 — Derivatives layout of the participant pipeline

Date: 2026-11-17. Status: accepted. Revisit when BEP017 (generic connectivity schema) is merged or when
`DerivativesVersion` changes.

## Question

The participant pipeline turns the source-level ROI time series of a BIDS dataset into connectivity
results for each participant, run and band. Where are these results written, what does a result consist
of, and which metadata must travel with it so that a reader does not need `dirconn` to understand it?

## Layout

Results form a BIDS derivatives dataset named `dirconn` below the output directory:

```
<out>/dirconn/
  dataset_description.json
  sub-02/
    ses-01/                      (only when the input has a session)
      eeg/
        sub-02_ses-01_task-rest_run-1_desc-gc_conn.nc
        sub-02_ses-01_task-rest_run-1_desc-gc_conn.json
        sub-02_ses-01_task-rest_run-1_desc-gc_edges.tsv
```

Entities appear in the order `sub ses task run`, then `desc` (the estimator), then the suffix.
`dataset_description.json` has `DatasetType: "derivative"`, `BIDSVersion: "1.9.0"`, a `GeneratedBy`
entry (name, version, code URL, and the container when run in one) and
`Dirconn: {"DerivativesVersion": "1"}`. The version is bumped when a field of the sidecar changes
meaning, not when one is added.

## Sidecar fields

Names follow the BEP017 draft where it has an equivalent (`Measure`, `Directed`, `NodeLabels`,
`Sources`); fields it does not cover are `dirconn` specific.

| Field | Meaning |
|---|---|
| `Measure` | Estimator method name (`provenance.method`). |
| `Directed` | Whether `values[i, j]` differs from `values[j, i]`. |
| `Estimator` | The `desc` entity. |
| `Parameters` | Estimator parameters. |
| `NodeLabels`, `NodeKinds` | Node names; `cortical` or `direct`, omitted when unknown. |
| `Bands` | Band name to `[low, high]` in Hz (`null` when not recorded). |
| `SamplingFrequency` | Omitted when unknown. |
| `Description` | Orientation of the matrix: `values[i, j]` is the influence of node `i` on node `j`. |
| `Sources` | Input files relative to the BIDS root (absolute when outside it). |
| `Statistics` | One entry per band: test, surrogate generator, number of surrogates, seed, data history of the observed estimate. Arrays are never stored here. |
| `ValueIsMagnitude` | Present for complex-valued measures: the table holds magnitudes. |
| `TaskKey` | Hash of the pipeline task; lets a re-run skip finished results. |
| `DirconnVersion`, `DerivativesVersion` | Writer version and layout version. |

## The long table

`_edges.tsv` has one row per ordered pair and band with the columns
`source target band value p p_adj significant`. A measure without bands uses `band = broadband`. A
missing p-value or decision (no test run) is `n/a`. The table exists for people who open results in a
spreadsheet or `pandas` and for group-level tools that want a flat file; the NetCDF file remains the
source of truth.

## Not done

- **No h5 or zarr.** NetCDF4 is already the carrier of `Connectivity`, readable from every language
  that matters here, and carries the full provenance. BEP017 plans h5/zarr arrays, but the proposal is
  not settled; a second array format would have to be kept in step with it for no gain. A converter can
  be added when BEP017 is final.
- **No `_relmat` suffix.** The pipeline stores the four-dimensional result (source, target, frequency,
  time); a matrix-per-file suffix would lose the axes.
- **`leakage_pairs` is removed from the pipeline configuration.** A leakage correction or flag needs a
  head model or crosstalk functions, which real recordings do not come with. The option invited
  made-up inputs; the leakage diagnostics stay in `dirconn.diagnostics` for simulations.
- **`pybids` is not adopted.** File discovery already goes through the `mne-bids` based readers, a
  second indexer would disagree with it at the edges, and `pybids` has been a risk on Python 3.13.

## Relation to ADR 0001

ADR 0001 fixes the input: source-level ROI time series under `derivatives/dirconn-roi`, as FIF
files whose `channels.tsv` carries the extra `node_kind` column, and NetCDF for results (item 6 there). This ADR fixes the output and
reads the input only through `Sources` and `NodeKinds`; it does not change the ROI convention.

# 0001 — BIDS layout for source-level ROI time series

Date: 2026-10-03. Status: accepted. Revisit when BEP021 / BEP017 change.

## Question

Is there an official BIDS convention for source-level M/EEG ROI time series? If not, what layout does
`dirconn` read and write?

## What was checked

Sources were read through web search and page fetches; several results are summaries of pages rather than
verbatim text, so wording below is paraphrased and the stable-spec page for the `timeseries` suffix could not be fetched.

| Item | Finding |
|---|---|
| BEP038 | Titled **"Atlases"** (leads: J. Kent, P. Herholz, E. Duff, A. Galassi). Defines atlas layout, the `atlas` entity, `_dseg` / `_probseg`, and `atlases.tsv`-style metadata. It states that it does **not** cover time series or M/EEG source-space representations. Sources disagree on the merge date (one says v1.9.0, 2023-11; the PR page summary says 2026-01); unresolved and irrelevant to this decision. |
| BEP021 | "Common Electrophysiological Derivatives" (EEG, MEG, iEEG). Its text excludes data with extensive processing "such as source reconstruction". Derivatives must carry `desc`; file formats follow raw (FIF, EEGLAB `.set`, BrainVision work for epoched data; EDF/BDF do not). Still a proposal; work resumed, file format and annotations are blocking items. |
| BEP017 | "Generic BIDS connectivity data schema". Still a proposal, seeking reviewers. Plans h5 or zarr arrays for multi-dimensional matrices (e.g. dynamic connectivity). Relevant to how `dirconn` stores results later. |
| `_timeseries.tsv` | Defined in BEP012-derived imaging derivatives with the documented template `<pipeline>/sub-*/func/..._[desc-]_timeseries.tsv`; `SamplingFrequency` is required. Documented for `func` (fMRI); no M/EEG datatype directory is documented for it. |
| `channels.tsv` types | EEG list: AUDIO, EEG, EOG, ECG, EMG, EYEGAZE, GSR, HEOG, MISC, PPG, PUPIL, REF, RESP, SYSCLOCK, TEMP, TRIG, VEOG. MEG list: MEGMAG, MEGGRADAXIAL, MEGGRADPLANAR, MEGOTHER, MEGREF*, plus non-MEG types, HLU, FITERR. **No `SOURCE` or ROI type exists.** `MISC` exists. Extra columns are allowed if described in the sidecar JSON. |
| MNE-BIDS / Brainstorm | No evidence found that either exports source-level data with a `SOURCE` channel type. |

## Conclusion

**There is no official BIDS convention for source-level M/EEG ROI time series.** Notes supplied with the task claimed
otherwise; the claims that did not hold up:

- "BEP038 = source-level electrophysiology derivatives": it is the atlas BEP and excludes time series and M/EEG source space.
- `_timeseries.tsv` under `meg/` or `eeg/` and `.h5` files named `_desc-roi_meg.h5`: not documented for M/EEG; `.h5` is not an accepted raw M/EEG format.
- Sidecar fields `Space: DKI`, `SpatialReference`, `Atlas{Name,Type,URL}`, `InverseMethod`, `SourceOrientation`, `InverseAlgorithm`, `SourceSpace`: not found in any specification page checked (the example URL `https://harvard.edu` is not an atlas URL).
- `SOURCE` channel type written by MNE-BIDS/Brainstorm: not in the channel type lists and no tool documentation found.

One suggestion from those notes is sound and adopted as an option: a label table next to the data, which
BEP038's `_dseg` machinery is designed for.

## Decision

`dirconn` uses a **documented, dirconn-specific convention (v0)** built only from things the spec already allows,
and says so in every file it writes.

1. Dataset: a BIDS-Derivatives dataset (`dataset_description.json` with `DatasetType: derivative`, `GeneratedBy`), `desc` entity always present.
2. Data file: `sub-*/[ses-*/]{eeg|meg}/sub-*_task-*_desc-<label>_{eeg|meg}.fif` — FIF, as BEP021 allows formats of the raw data; one row per ROI or invasive channel.
   Epoched data is also FIF (`_epo.fif`-style content in the same container).
3. `*_channels.tsv`: `name` (DK label in MNE style, e.g. `inferiortemporal-lh`; invasive channels `STN-L`, `STN-R`), `type = MISC`,
   `units`, `sampling_frequency`, plus extra columns `region`, `hemisphere`, `atlas`, `node_kind` (`cortical` | `direct`), all described in `*_channels.json`.
4. Sidecar JSON: standard fields (`SamplingFrequency`, `TaskName`, `EEGReference` or MEG equivalents where applicable) and a namespaced object
   `"Dirconn": {"Convention": "0", "Atlas": "DK", "SourceSpace": ..., "InverseMethod": ..., "RoiExtraction": "mean_flip|pca_flip|...", "SourceOrientation": ..., "LeakageCorrection": ...}`.
   Namespacing avoids collisions with any future official fields.
5. Optional label table: `atlas-DK_dseg.tsv` following BEP038 conventions, used only if a source-space segmentation is shipped.
6. Truth for simulated datasets and results: NetCDF in the derivatives dataset (`desc-truth`, `desc-result`), described in the sidecar. Revisit if BEP017 settles on h5/zarr.
7. Reader: `read_bids(path, level="sensor"|"roi")`. `level="sensor"` is standard raw BIDS via `mne-bids`; `level="roi"` requires `Dirconn.Convention` and fails with a clear message otherwise.
8. Docs state: "This ROI layout is a dirconn convention, not part of the BIDS specification."

## Consequences

- Standard BIDS validators will accept the dataset structure but will not understand the `Dirconn` object; the extra `channels.tsv` columns must be declared in `channels.json` to stay valid.
- If BEP021 or a successor adds source-level data, migrate by adding a converter and bumping `Dirconn.Convention`.
- Tests: write-read round trip. The official BIDS validator is deliberately not used: it cannot interpret the `Dirconn` entry,
  and sensor-level data are already checked by `mne-bids` when read.
- Implementation notes: file naming and `dataset_description.json` come from `mne-bids`; data are double-precision FIF;
  epoched data are stored as concatenated trials described by `events.tsv` and the `Segmentation`, `NTrials`, `EpochLength`, `Tmin`
  sidecar entries; `run` is not zero-padded unless given as a string such as `"03"`.

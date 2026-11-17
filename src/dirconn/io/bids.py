"""BIDS input and output.

Sensor-level EEG/MEG/iEEG is plain BIDS and is read with ``mne-bids``. Region-level (ROI) time
series and invasive channels such as an STN LFP have no official BIDS convention; they are
stored in a BIDS-Derivatives dataset following the dirconn convention described in
``docs/decisions/0001-bids-roi-layout.md``. That layout is **not** part of the BIDS
specification and every file written here says so through the ``Dirconn`` sidecar entry.
"""

from __future__ import annotations

import json
import os
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from dirconn.data.channels import NODE_KINDS, ChannelMeta
from dirconn.data.timeseries import TimeSeries

CONVENTION = "0"
DATATYPES: tuple[str, ...] = ("eeg", "meg", "ieeg")
LEVELS: tuple[str, ...] = ("sensor", "roi")

_NA = "n/a"
_CHANNEL_COLUMN_DOCS: dict[str, dict[str, Any]] = {
    "region": {
        "Description": "Region label of the node, for example a Desikan-Killiany label or STN."
    },
    "hemisphere": {
        "Description": "Hemisphere of the node.",
        "Levels": {"lh": "left", "rh": "right", "mid": "midline"},
    },
    "atlas": {"Description": "Atlas the region label comes from; n/a for direct channels."},
    "node_kind": {
        "Description": "What the channel represents in the connectivity analysis.",
        "Levels": {
            "sensor": "scalp or MEG sensor",
            "cortical": "cortical region of interest extracted from a source estimate",
            "direct": "invasive channel used as recorded, for example an STN LFP",
        },
    },
}


def _check_datatype(datatype: str) -> None:
    if datatype not in DATATYPES:
        raise ValueError(f"datatype must be one of {DATATYPES}, got {datatype!r}.")


def _write_json(path: Path, content: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(content, handle, indent=4)
        handle.write("\n")


def _write_tsv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, sep="\t", index=False, na_rep=_NA, lineterminator="\n", encoding="utf-8")


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[_NA])


def _atlas_entry(channels: tuple[ChannelMeta, ...]) -> str | list[str] | None:
    atlases = sorted({ch.atlas for ch in channels if ch.atlas})
    if not atlases:
        return None
    return atlases[0] if len(atlases) == 1 else atlases


def write_roi_bids(
    ts: TimeSeries,
    root: str | os.PathLike[str],
    *,
    subject: str,
    task: str,
    description: str,
    session: str | None = None,
    run: int | str | None = None,
    datatype: str = "eeg",
    source_space: str | None = None,
    inverse_method: str | None = None,
    roi_extraction: str | None = None,
    source_orientation: str | None = None,
    leakage_correction: str | None = None,
    overwrite: bool = False,
) -> Any:
    """Write region-level time series as a BIDS-Derivatives dataset (dirconn convention v0).

    The data are stored as a double-precision FIF file. Epoched data are concatenated along
    time and described by an ``events.tsv`` file plus the ``Segmentation``, ``NTrials``,
    ``EpochLength`` and ``Tmin`` entries of the sidecar, so they can be restored exactly. Channel
    positions are not stored. Invalid samples (``ts.valid``) are written as NaN; read them back
    with ``read_bids(..., nan="mask")``. A ``predict`` mask is an analysis choice, not part of the
    recording, and is not written: such series are refused.

    Parameters
    ----------
    ts : TimeSeries
        Region-level series; the channel metadata (region, hemisphere, atlas, node kind) is
        written to ``channels.tsv``.
    root : path-like
        Root of the derivatives dataset; created if needed.
    subject, task : str
        BIDS entities.
    description : str
        Value of the ``desc`` entity, which is mandatory for derivatives (for example ``"dk"``).
    session, run : optional
        BIDS entities.
    datatype : {"eeg", "meg", "ieeg"}
        Directory and suffix to use, normally the modality the regions were reconstructed from.
    source_space, inverse_method, roi_extraction, source_orientation, leakage_correction : str
        Processing description recorded in the ``Dirconn`` sidecar entry; omitted when ``None``.
    overwrite : bool
        Replace existing files instead of raising ``FileExistsError``.

    Returns
    -------
    mne_bids.BIDSPath
        Path of the FIF file.
    """
    import mne
    from mne_bids import BIDSPath, make_dataset_description

    _check_datatype(datatype)
    if ts.predict is not None:
        raise ValueError(
            "write_roi_bids does not store a predict mask; write the series without it "
            "(dataclasses.replace(ts, predict=None)) and keep the mask with the analysis."
        )
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string (the BIDS desc entity).")
    root = Path(root)
    path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        description=description,
        root=root,
        datatype=datatype,
        suffix=datatype,
        extension=".fif",
    )
    if path.fpath.exists() and not overwrite:
        raise FileExistsError(f"{path.fpath} exists; pass overwrite=True to replace it.")
    path.mkdir()

    if not (root / "dataset_description.json").exists():
        make_dataset_description(
            path=root,
            name="dirconn region-level time series",
            dataset_type="derivative",
            generated_by=[
                {
                    "Name": "dirconn",
                    "Version": metadata.version("dirconn"),
                    "Description": "Region-level time series, dirconn convention "
                    f"{CONVENTION} (not part of the BIDS specification).",
                }
            ],
        )

    info = mne.create_info(list(ts.names), ts.sfreq, "misc")
    samples = np.asarray(ts.data)
    if ts.valid is not None:
        samples = np.where(ts.valid[:, None, :], samples, np.nan)
    raw = mne.io.RawArray(np.concatenate(list(samples), axis=1), info, verbose="error")
    raw.save(path.fpath, overwrite=True, fmt="double", verbose="error")

    meta: dict[str, Any] = {
        "Convention": CONVENTION,
        "Atlas": _atlas_entry(ts.channels),
        "Segmentation": ts.segmentation,
        "NTrials": ts.n_trials,
        "EpochLength": ts.n_times,
        "Tmin": ts.tmin,
        "SourceSpace": source_space,
        "InverseMethod": inverse_method,
        "RoiExtraction": roi_extraction,
        "SourceOrientation": source_orientation,
        "LeakageCorrection": leakage_correction,
    }
    sidecar = {
        "TaskName": task,
        "SamplingFrequency": ts.sfreq,
        "Dirconn": {k: v for k, v in meta.items() if v is not None},
    }
    _write_json(path.copy().update(extension=".json").fpath, sidecar)

    channels = pd.DataFrame(
        {
            "name": list(ts.names),
            "type": [ch.type.upper() for ch in ts.channels],
            "units": _NA,
            "region": [ch.region for ch in ts.channels],
            "hemisphere": [ch.hemisphere for ch in ts.channels],
            "atlas": [ch.atlas for ch in ts.channels],
            "node_kind": [ch.node_kind for ch in ts.channels],
        }
    )
    channels_path = path.copy().update(suffix="channels", extension=".tsv")
    _write_tsv(channels_path.fpath, channels)
    _write_json(channels_path.copy().update(extension=".json").fpath, _CHANNEL_COLUMN_DOCS)

    events_path = path.copy().update(suffix="events", extension=".tsv")
    if ts.segmentation == "epoched":
        length = ts.n_times / ts.sfreq
        events = pd.DataFrame(
            {
                "onset": np.arange(ts.n_trials) * length,
                "duration": length,
                "trial_type": "epoch",
            }
        )
        _write_tsv(events_path.fpath, events)
    elif events_path.fpath.exists():
        events_path.fpath.unlink()
    return path


def _infer_datatype(root: Path, subject: str, session: str | None) -> str:
    base = root / f"sub-{subject}"
    if session is not None:
        base = base / f"ses-{session}"
    found = [dt for dt in DATATYPES if (base / dt).is_dir()]
    if not found:
        raise FileNotFoundError(f"No {'/'.join(DATATYPES)} directory found in {base}.")
    if len(found) > 1:
        raise ValueError(f"Several datatypes found in {base} ({found}); pass datatype explicitly.")
    return found[0]


def read_bids(
    root: str | os.PathLike[str],
    *,
    subject: str,
    task: str,
    session: str | None = None,
    run: int | str | None = None,
    datatype: str | None = None,
    level: str = "sensor",
    description: str | None = None,
    nan: str = "raise",
    bad_annotations: str = "ignore",
) -> TimeSeries:
    """Read one recording from a BIDS dataset as a :class:`TimeSeries`.

    Parameters
    ----------
    root : path-like
        Root of the BIDS dataset, or of the derivatives dataset when ``level="roi"``.
    subject, task : str
        BIDS entities.
    session, run : optional
        BIDS entities.
    datatype : {"eeg", "meg", "ieeg"}, optional
        Inferred from the directory when only one exists.
    level : {"sensor", "roi"}
        ``"sensor"`` reads a standard raw BIDS recording through ``mne-bids``. ``"roi"`` reads a
        dataset written with :func:`write_roi_bids` and requires the ``Dirconn`` convention entry.
    description : str, optional
        The ``desc`` entity; required for ``level="roi"``.
    nan : {"raise", "mask"}
        What to do with NaN samples, as in :meth:`TimeSeries.from_array`: ``"mask"`` replaces
        them by 0 and marks them invalid in ``TimeSeries.valid`` (gaps in the recording).
    bad_annotations : {"ignore", "mask"}
        Sensor level only: ``"mask"`` also marks samples under ``BAD_`` annotations invalid (needs
        ``nan="mask"``). Ignored for ``level="roi"``, where invalid samples are stored as NaN.

    Returns
    -------
    TimeSeries
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}.")
    if nan not in ("raise", "mask"):
        raise ValueError(f"nan must be 'raise' or 'mask', got {nan!r}.")
    if datatype is not None:
        _check_datatype(datatype)
    if level == "roi" and not description:
        raise ValueError("description is required for level='roi' (the BIDS desc entity).")
    root = Path(root)
    datatype = datatype or _infer_datatype(root, subject, session)
    if level == "roi":
        assert description is not None
        return _read_roi(root, subject, task, session, run, datatype, description, nan)
    return _read_sensor(root, subject, task, session, run, datatype, nan, bad_annotations)


def _read_sensor(
    root: Path,
    subject: str,
    task: str,
    session: str | None,
    run: Any,
    datatype: str,
    nan: str,
    bad_annotations: str = "ignore",
) -> TimeSeries:
    from mne_bids import BIDSPath, read_raw_bids

    path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        root=root,
        datatype=datatype,
        suffix=datatype,
    )
    if not path.directory.is_dir():
        raise FileNotFoundError(f"{path.directory} does not exist.")
    raw = read_raw_bids(path, verbose="error")
    series = TimeSeries.from_mne(raw, nan=nan, bad_annotations=bad_annotations)

    # Epoched recordings are stored as concatenated trials; the sidecar says how to split them.
    sidecar_path = path.copy().update(extension=".json").fpath
    if sidecar_path.exists():
        with open(sidecar_path, encoding="utf-8") as handle:
            meta = json.load(handle).get("Dirconn")
        if isinstance(meta, dict) and meta.get("Segmentation") == "epoched":
            n_trials, length = int(meta["NTrials"]), int(meta["EpochLength"])
            data = _split_epochs(series.data[0], n_trials, length)
            valid = None
            if series.valid is not None:
                valid = series.valid[0].reshape(n_trials, length)
            return TimeSeries.from_array(
                data,
                series.sfreq,
                channels=series.channels,
                segmentation="epoched",
                tmin=float(meta.get("Tmin", 0.0)),
                valid=valid,
            )
    return series


def _split_epochs(data: np.ndarray, n_trials: int, length: int) -> np.ndarray:
    """Split ``(n_channels, n_trials * length)`` into trials ``(n_trials, n_channels, length)``."""
    if data.shape[1] != n_trials * length:
        raise ValueError(
            f"The data have {data.shape[1]} samples but the sidecar describes "
            f"{n_trials} epochs of {length} samples."
        )
    return data.reshape(data.shape[0], n_trials, length).transpose(1, 0, 2)


def _read_roi(
    root: Path,
    subject: str,
    task: str,
    session: str | None,
    run: Any,
    datatype: str,
    description: str,
    nan: str = "raise",
) -> TimeSeries:
    import mne
    from mne_bids import BIDSPath

    path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        description=description,
        root=root,
        datatype=datatype,
        suffix=datatype,
        extension=".fif",
    )
    if not path.fpath.exists():
        raise FileNotFoundError(f"{path.fpath} does not exist.")

    with open(path.copy().update(extension=".json").fpath, encoding="utf-8") as handle:
        sidecar = json.load(handle)
    meta = sidecar.get("Dirconn")
    if not isinstance(meta, dict):
        raise ValueError(
            f"{path.fpath.name} has no 'Dirconn' entry in its sidecar; it was not written with "
            "write_roi_bids. Use level='sensor' for standard BIDS recordings."
        )
    if meta.get("Convention") != CONVENTION:
        raise ValueError(
            f"Unsupported Dirconn convention version {meta.get('Convention')!r} "
            f"(this dirconn reads {CONVENTION!r})."
        )

    raw = mne.io.read_raw_fif(path.fpath, preload=True, verbose="error")
    table = _read_tsv(path.copy().update(suffix="channels", extension=".tsv").fpath)
    if list(table["name"]) != list(raw.ch_names):
        raise ValueError(
            "The names in channels.tsv do not match the channels of the data file "
            f"({list(table['name'])} vs {list(raw.ch_names)})."
        )

    def optional(value: Any) -> str | None:
        return None if pd.isna(value) else str(value)

    channels = []
    for _, row in table.iterrows():
        node_kind = optional(row.get("node_kind")) or NODE_KINDS[0]
        channels.append(
            ChannelMeta(
                str(row["name"]),
                type=str(row["type"]).lower(),
                node_kind=node_kind,
                region=optional(row.get("region")),
                hemisphere=optional(row.get("hemisphere")),
                atlas=optional(row.get("atlas")),
            )
        )

    data = raw.get_data()
    segmentation = meta["Segmentation"]
    if segmentation == "epoched":
        data = _split_epochs(data, int(meta["NTrials"]), int(meta["EpochLength"]))
    return TimeSeries.from_array(
        data,
        float(raw.info["sfreq"]),
        channels=channels,
        segmentation=segmentation,
        tmin=float(meta.get("Tmin", 0.0)),
        nan=nan,
    )


# The simulation layout lives in dirconn.io.simulation_bids, which imports helpers from this
# module. A module-level re-export would therefore depend on which of the two is imported first,
# so every name of the 0.1.0 module that moved there (SimulationRecord, write_simulation_bids,
# read_simulation_bids, SIMULATION_PIPELINE, SENSOR_UNIT, the private helpers) is resolved
# lazily (PEP 562).


def __getattr__(name: str) -> Any:
    if not name.startswith("__"):
        from dirconn.io import simulation_bids

        try:
            return getattr(simulation_bids, name)
        except AttributeError:
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    from dirconn.io import simulation_bids

    forwarded = {n for n in dir(simulation_bids) if not n.startswith("__")}
    return sorted(set(globals()) | forwarded)


if TYPE_CHECKING:
    from dirconn.io.simulation_bids import (  # noqa: F401
        SimulationRecord,
        read_simulation_bids,
        write_simulation_bids,
    )

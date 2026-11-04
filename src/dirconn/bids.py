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
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from dirconn.channels import NODE_KINDS, ChannelMeta
from dirconn.network import Network
from dirconn.provenance import Provenance
from dirconn.result import Connectivity
from dirconn.timeseries import TimeSeries

if TYPE_CHECKING:
    from dirconn.simulation import SimulatedData

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
    positions are not stored.

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
    raw = mne.io.RawArray(np.concatenate(list(ts.data), axis=1), info, verbose="error")
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

    Returns
    -------
    TimeSeries
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}.")
    if datatype is not None:
        _check_datatype(datatype)
    if level == "roi" and not description:
        raise ValueError("description is required for level='roi' (the BIDS desc entity).")
    root = Path(root)
    datatype = datatype or _infer_datatype(root, subject, session)
    if level == "roi":
        assert description is not None
        return _read_roi(root, subject, task, session, run, datatype, description)
    return _read_sensor(root, subject, task, session, run, datatype)


def _read_sensor(
    root: Path, subject: str, task: str, session: str | None, run: Any, datatype: str
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
    series = TimeSeries.from_mne(raw)

    # Epoched recordings are stored as concatenated trials; the sidecar says how to split them.
    sidecar_path = path.copy().update(extension=".json").fpath
    if sidecar_path.exists():
        with open(sidecar_path, encoding="utf-8") as handle:
            meta = json.load(handle).get("Dirconn")
        if isinstance(meta, dict) and meta.get("Segmentation") == "epoched":
            data = _split_epochs(series.data[0], int(meta["NTrials"]), int(meta["EpochLength"]))
            return TimeSeries.from_array(
                data,
                series.sfreq,
                channels=series.channels,
                segmentation="epoched",
                tmin=float(meta.get("Tmin", 0.0)),
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
    )


SIMULATION_PIPELINE = "dirconn-simulation"
SENSOR_UNIT = 1e-6
"""Sensor data of a simulation are written as microvolts: one simulated unit becomes 1 uV."""


@dataclass(frozen=True, eq=False)
class SimulationRecord:
    """A simulation read back from BIDS by :func:`read_simulation_bids`.

    Attributes
    ----------
    sensors : TimeSeries
        Sensor channels, in volts (the simulated unit is written as microvolts).
    direct : TimeSeries or None
        The direct (invasive) channels as recorded, with their noise; ``None`` without them.
    sources : TimeSeries
        Noise-free activity of every network node, in simulated units.
    truth : Connectivity
        Broadband ground truth.
    network : Network
        The network that generated the data.
    leadfield : ndarray, shape (n_sensors, n_cortical + n_background)
        Mixing matrix used by the simulation.
    provenance : Provenance
        The provenance of the simulation, seed included.
    """

    sensors: TimeSeries
    direct: TimeSeries | None
    sources: TimeSeries
    truth: Connectivity
    network: Network
    leadfield: np.ndarray
    provenance: Provenance

    def truth_at(self, freqs: Any = None, times: Any = None) -> Connectivity:
        """Ground truth on the frequency and time axes of the result to be evaluated."""
        return self.network.truth(freqs=freqs, times=times)


def _simulation_layout(
    root: Path, subject: str, task: str, session: str | None, run: Any
) -> dict[str, Path]:
    from mne_bids import BIDSPath

    derivatives = root / "derivatives" / SIMULATION_PIPELINE
    entities = dict(subject=subject, session=session, task=task, run=run)
    directory = BIDSPath(**entities, root=derivatives, datatype="eeg").directory

    def stem(description: str) -> str:
        path = BIDSPath(**entities, description=description, root=derivatives, datatype="eeg")
        return str(path.basename)

    return {
        "derivatives": derivatives,
        "directory": directory,
        "sources": directory / f"{stem('sources')}_eeg.fif",
        "direct": directory / f"{stem('direct')}_eeg.fif",
        "truth": directory / f"{stem('truth')}_connectivity.nc",
        "leadfield": directory / f"{stem('leadfield')}_leadfield.nc",
        "simulation": directory / f"{stem('simulation')}_simulation.json",
        "direct_stem": directory / stem("direct"),
    }


def write_simulation_bids(
    sim: SimulatedData,
    root: str | os.PathLike[str],
    *,
    subject: str,
    task: str,
    session: str | None = None,
    run: int | str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write a simulation as a raw BIDS dataset plus a derivatives dataset.

    The sensor channels go to a standard raw EEG dataset in ``root`` (BrainVision files, written
    by ``mne-bids``; the simulated unit is written as one microvolt). Epoched simulations are
    stored as concatenated trials with an ``events.tsv`` file and a ``Dirconn`` sidecar entry
    that lets :func:`read_bids` split them again. Everything else goes to
    ``root/derivatives/dirconn-simulation`` in the dirconn convention: the noise-free node
    activity (``desc-sources``), the direct channels as recorded (``desc-direct``), the
    broadband truth, the leadfield and a record holding the provenance.

    Nothing is written when a target already exists and ``overwrite`` is false.

    Parameters
    ----------
    sim : SimulatedData
        The simulation to store.
    root : path-like
        Root of the raw BIDS dataset.
    subject, task : str
        BIDS entities.
    session, run : optional
        BIDS entities.
    overwrite : bool
        Replace existing files.

    Returns
    -------
    dict of str to Path
        The files written: ``raw`` (the BrainVision header), ``sources``, ``direct`` (only with
        direct nodes), ``truth``, ``leadfield`` and ``simulation``.
    """
    import mne
    import xarray as xr
    from mne_bids import BIDSPath, write_raw_bids

    root = Path(root)
    layout = _simulation_layout(root, subject, task, session, run)
    raw_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        root=root,
        datatype="eeg",
        suffix="eeg",
    )
    header = raw_path.copy().update(extension=".vhdr").fpath

    channels = sim.data.channels
    sensor_idx = [i for i, ch in enumerate(channels) if ch.node_kind == "sensor"]
    direct_idx = [i for i, ch in enumerate(channels) if ch.node_kind == "direct"]

    targets = [
        header,
        layout["sources"],
        layout["truth"],
        layout["leadfield"],
        layout["simulation"],
    ]
    if direct_idx:
        targets.append(layout["direct"])
    existing = [p for p in targets if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{[p.name for p in existing]} already exist; pass overwrite=True to replace them."
        )

    n_trials, length = sim.data.n_trials, sim.data.n_times
    sensor_names = [channels[i].name for i in sensor_idx]
    continuous = np.concatenate(list(sim.data.data[:, sensor_idx]), axis=1) * SENSOR_UNIT
    raw = mne.io.RawArray(
        continuous, mne.create_info(sensor_names, sim.data.sfreq, "eeg"), verbose="error"
    )
    events = event_id = None
    if sim.data.segmentation == "epoched":
        events = np.array([[k * length, 0, 1] for k in range(n_trials)])
        event_id = {"epoch": 1}
    write_raw_bids(
        raw,
        raw_path,
        format="BrainVision",
        allow_preload=True,
        events=events,
        event_id=event_id,
        overwrite=overwrite,
        verbose="error",
    )
    if sim.data.segmentation == "epoched":
        sidecar_path = raw_path.copy().update(extension=".json").fpath
        with open(sidecar_path, encoding="utf-8") as handle:
            sidecar = json.load(handle)
        sidecar["Dirconn"] = {
            "Convention": CONVENTION,
            "Segmentation": "epoched",
            "NTrials": n_trials,
            "EpochLength": length,
            "Tmin": sim.data.tmin,
        }
        _write_json(sidecar_path, sidecar)

    derivatives = layout["derivatives"]
    write_roi_bids(
        sim.sources,
        derivatives,
        subject=subject,
        task=task,
        session=session,
        run=run,
        description="sources",
        datatype="eeg",
        overwrite=True,
    )
    if direct_idx:
        direct = TimeSeries.from_array(
            sim.data.data[:, direct_idx],
            sim.data.sfreq,
            channels=[channels[i] for i in direct_idx],
            segmentation=sim.data.segmentation,
            tmin=sim.data.tmin,
        )
        write_roi_bids(
            direct,
            derivatives,
            subject=subject,
            task=task,
            session=session,
            run=run,
            description="direct",
            datatype="eeg",
            overwrite=True,
        )
    else:
        for stale in layout["directory"].glob(f"{layout['direct_stem'].name}_*"):
            stale.unlink()

    sim.truth.to_netcdf(layout["truth"])

    cortical = [n.name for n in sim.network.nodes if n.node_kind == "cortical"]
    background = [f"bg{k:02d}" for k in range(sim.leadfield.shape[1] - len(cortical))]
    xr.Dataset(
        {"leadfield": (("sensor", "source"), np.asarray(sim.leadfield))},
        coords={"sensor": sensor_names, "source": cortical + background},
    ).to_netcdf(layout["leadfield"], engine="netcdf4")

    _write_json(
        layout["simulation"],
        {
            "Dirconn": {
                "Convention": CONVENTION,
                "Provenance": sim.provenance.to_dict(),
                "SensorUnit": "one simulated unit is written as 1 uV",
                "DirectNames": [channels[i].name for i in direct_idx],
            }
        },
    )

    paths = {
        "raw": header,
        "sources": layout["sources"],
        "truth": layout["truth"],
        "leadfield": layout["leadfield"],
        "simulation": layout["simulation"],
    }
    if direct_idx:
        paths["direct"] = layout["direct"]
    return paths


def read_simulation_bids(
    root: str | os.PathLike[str],
    *,
    subject: str,
    task: str,
    session: str | None = None,
    run: int | str | None = None,
) -> SimulationRecord:
    """Read a simulation written by :func:`write_simulation_bids`.

    Parameters
    ----------
    root : path-like
        Root of the raw BIDS dataset.
    subject, task : str
        BIDS entities.
    session, run : optional
        BIDS entities.

    Returns
    -------
    SimulationRecord
    """
    import xarray as xr

    root = Path(root)
    layout = _simulation_layout(root, subject, task, session, run)
    if not layout["simulation"].exists():
        raise FileNotFoundError(f"No simulation record found at {layout['simulation']}.")
    with open(layout["simulation"], encoding="utf-8") as handle:
        meta = json.load(handle).get("Dirconn")
    if not isinstance(meta, dict) or meta.get("Convention") != CONVENTION:
        raise ValueError(
            f"{layout['simulation'].name} is not a dirconn simulation record of convention "
            f"{CONVENTION!r}."
        )

    entities: dict[str, Any] = dict(
        subject=subject, task=task, session=session, run=run, datatype="eeg"
    )
    sensors = read_bids(root, level="sensor", **entities)
    derivatives = layout["derivatives"]
    sources = read_bids(derivatives, level="roi", description="sources", **entities)
    direct = None
    if meta.get("DirectNames"):
        direct = read_bids(derivatives, level="roi", description="direct", **entities)
    truth = Connectivity.from_netcdf(layout["truth"])
    with xr.open_dataset(layout["leadfield"], engine="netcdf4") as dataset:
        leadfield = np.array(dataset["leadfield"].values)
    provenance = Provenance.from_dict(meta["Provenance"])
    return SimulationRecord(
        sensors=sensors,
        direct=direct,
        sources=sources,
        truth=truth,
        network=Network.from_dict(provenance.parameters["network"]),
        leadfield=leadfield,
        provenance=provenance,
    )

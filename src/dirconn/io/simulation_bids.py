"""Simulated data in BIDS: write a simulation with its ground truth and read it back.

The layout follows the dirconn convention of :mod:`dirconn.io.bids`; the simulation is
stored in the ``dirconn-simulation`` derivatives pipeline next to the raw sensor dataset.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dirconn.data.network import Network
from dirconn.data.provenance import Provenance
from dirconn.data.result import Connectivity
from dirconn.data.timeseries import TimeSeries
from dirconn.io.bids import CONVENTION, _write_json, read_bids, write_roi_bids

if TYPE_CHECKING:
    from dirconn.simulation import SimulatedData


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

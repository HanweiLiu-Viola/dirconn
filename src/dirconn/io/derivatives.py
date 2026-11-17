"""Layout of the ``dirconn`` derivatives dataset (version 1), written by the participant pipeline.

A result is one NetCDF file (the full-spectrum :class:`~dirconn.Connectivity`), a JSON sidecar with
the fields a reader needs without opening it, and an optional long ``_edges.tsv`` table with one
row per ordered pair and band. See ``docs/decisions/0003-derivatives-layout.md``.

This module does not import :mod:`dirconn.workflows`; the significance results are described by the
:class:`BandTestLike` protocol.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

import dirconn
from dirconn.data.result import Connectivity

__all__ = [
    "DERIVATIVES_VERSION",
    "PIPELINE_NAME",
    "BandTestLike",
    "ResultFiles",
    "derivative_root",
    "filename",
    "read_edges",
    "result_task_key",
    "unit_dir",
    "write_dataset_description",
    "write_result",
]

PIPELINE_NAME = "dirconn"
DERIVATIVES_VERSION = "1"
BIDS_VERSION = "1.9.0"
CODE_URL = "https://github.com/HanweiLiu-Viola/dirconn"
BROADBAND = "broadband"
EDGE_COLUMNS = ("source", "target", "band", "value", "p", "p_adj", "significant")
_ENTITY_ORDER = ("sub", "ses", "task", "run")


class BandTestLike(Protocol):
    """What the writer reads of a significance test (``workflows.steps.test.BandTest``)."""

    band: str | None
    pvalues: Any
    adjusted: Any
    significant: Any
    test: str
    n_surrogates: int | None
    seed: int | None
    generator: dict[str, Any]
    observed_history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ResultFiles:
    """Paths written by :func:`write_result`.

    Parameters
    ----------
    nc : Path
        The NetCDF file.
    json : Path
        The sidecar.
    edges : Path or None
        The long table, ``None`` when it was not requested.
    """

    nc: Path
    json: Path
    edges: Path | None


def derivative_root(out_dir: Path) -> Path:
    """Return the derivatives dataset root, ``out_dir/"dirconn"``."""
    return Path(out_dir) / PIPELINE_NAME


def unit_dir(root: Path, entities: Mapping[str, str], datatype: str) -> Path:
    """Return ``root/sub-XX[/ses-YY]/<datatype>``."""
    path = Path(root) / f"sub-{entities['sub']}"
    if entities.get("ses"):
        path = path / f"ses-{entities['ses']}"
    return path / datatype


def filename(entities: Mapping[str, str], *, desc: str, suffix: str, ext: str) -> str:
    """Return a file name with entities in the order ``sub ses task run``, then ``desc``.

    Parameters
    ----------
    entities : mapping of str to str
        Entity values without the key prefix (``{"sub": "02", "task": "rest"}``); absent or empty
        entities are skipped.
    desc : str
        Value of the ``desc`` entity.
    suffix, ext : str
        Suffix (``"conn"``) and extension with its dot (``".nc"``).
    """
    parts = [f"{k}-{entities[k]}" for k in _ENTITY_ORDER if entities.get(k)]
    parts.append(f"desc-{desc}")
    return "_".join(parts) + f"_{suffix}{ext}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_dataset_description(
    root: Path, *, version: str, container: Mapping[str, str | None] | None
) -> None:
    """Write ``dataset_description.json`` of a derivatives dataset, replacing an existing one.

    Parameters
    ----------
    root : Path
        Dataset root; created when missing.
    version : str
        The ``dirconn`` version that generated the dataset.
    container : mapping or None
        Container description for ``GeneratedBy``; omitted when ``None``.
    """
    generated: dict[str, Any] = {"Name": PIPELINE_NAME, "Version": version, "CodeURL": CODE_URL}
    if container is not None:
        generated["Container"] = dict(container)
    _write_json(
        Path(root) / "dataset_description.json",
        {
            "Name": "dirconn derivatives",
            "BIDSVersion": BIDS_VERSION,
            "DatasetType": "derivative",
            "GeneratedBy": [generated],
            "Dirconn": {"DerivativesVersion": DERIVATIVES_VERSION},
        },
    )


def _plain(obj: Any) -> Any:
    """Convert numpy scalars and containers to strict JSON types."""
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _is_directed(conn: Connectivity) -> bool:
    from dirconn.methods.base import method_info

    info = method_info(conn.provenance.method)
    if info is not None:
        return bool(info.directed)
    v = np.abs(conn.values)
    return not bool(np.allclose(v, np.swapaxes(v, 0, 1), equal_nan=True))


def _source_name(path: Path, bids_dir: Path) -> str:
    try:
        return Path(path).relative_to(bids_dir).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _statistics(tests: Sequence[BandTestLike]) -> list[dict[str, Any]]:
    return [
        _plain(
            {
                "Band": t.band or BROADBAND,
                "Test": t.test,
                "Generator": t.generator,
                "NSurrogates": t.n_surrogates,
                "Seed": t.seed,
                "ObservedHistory": list(t.observed_history),
            }
        )
        for t in tests
    ]


def _pair_values(conn: Connectivity) -> np.ndarray:
    """Return the ``(source, target)`` values of a band-reduced result; magnitude if complex."""
    return np.abs(conn.values[:, :, 0, 0])


def _edge_rows(
    conn: Connectivity,
    bands: Mapping[str, Connectivity],
    tests: Sequence[BandTestLike],
) -> pd.DataFrame:
    by_band = {t.band or BROADBAND: t for t in tests}
    if bands:
        items = list(bands.items())
    elif conn.values.shape[2:] == (1, 1):
        items = [(BROADBAND, conn)]
    else:
        items = []
    rows: list[dict[str, Any]] = []
    for name, bc in items:
        vals = _pair_values(bc)
        test = by_band.get(name)
        for i, s in enumerate(bc.source):
            for j, t in enumerate(bc.target):
                if i == j:
                    continue
                row: dict[str, Any] = {
                    "source": s,
                    "target": t,
                    "band": name,
                    "value": float(vals[i, j]),
                    "p": None,
                    "p_adj": None,
                    "significant": None,
                }
                if test is not None:
                    row["p"] = float(np.asarray(test.pvalues)[i, j])
                    row["p_adj"] = float(np.asarray(test.adjusted)[i, j])
                    row["significant"] = bool(np.asarray(test.significant)[i, j])
                rows.append(row)
    df = pd.DataFrame(rows, columns=list(EDGE_COLUMNS))
    df["p"] = df["p"].astype("float64")
    df["p_adj"] = df["p_adj"].astype("float64")
    return df


def write_result(
    root: Path,
    entities: Mapping[str, str],
    datatype: str,
    *,
    desc: str,
    conn: Connectivity,
    bands: Mapping[str, Connectivity],
    tests: Sequence[BandTestLike],
    sources: Sequence[Path],
    bids_dir: Path,
    task_key: str,
    long_table: bool,
) -> ResultFiles:
    """Write the NetCDF file, the sidecar and (optionally) the long edge table of one result.

    Parameters
    ----------
    root : Path
        Derivatives dataset root (:func:`derivative_root`).
    entities : mapping of str to str
        BIDS entities of the unit (``sub``, optional ``ses``, ``task``, ``run``).
    datatype : str
        Datatype directory, for example ``"eeg"``.
    desc : str
        Estimator description, the ``desc`` entity.
    conn : Connectivity
        Full-spectrum result, stored as NetCDF.
    bands : mapping of str to Connectivity
        Band-reduced results (one frequency, one time); may be empty.
    tests : sequence of BandTestLike
        Significance tests, matched to bands by name (``None`` is ``"broadband"``).
    sources : sequence of Path
        Input files, recorded relative to ``bids_dir`` when possible.
    bids_dir : Path
        BIDS root the sources are relative to.
    task_key : str
        Key of the pipeline task, read back by :func:`result_task_key` for resuming.
    long_table : bool
        Whether to write ``_edges.tsv``. Complex values are written as magnitudes
        (``ValueIsMagnitude`` in the sidecar). A result without bands is tabulated only when it
        has no frequency or time axis; otherwise the table has no rows.

    Returns
    -------
    ResultFiles
        The written paths.
    """
    directory = unit_dir(root, entities, datatype)
    directory.mkdir(parents=True, exist_ok=True)
    nc = directory / filename(entities, desc=desc, suffix="conn", ext=".nc")
    sidecar = directory / filename(entities, desc=desc, suffix="conn", ext=".json")
    edges = directory / filename(entities, desc=desc, suffix="edges", ext=".tsv")

    conn.to_netcdf(nc)

    prov = conn.provenance
    meta: dict[str, Any] = {
        "Measure": prov.method,
        "Directed": _is_directed(conn),
        "Estimator": desc,
        "Parameters": prov.parameters,
        "NodeLabels": list(conn.source),
        "Description": (
            f"{prov.method} connectivity; values[i, j] is the influence of NodeLabels[i] "
            "on NodeLabels[j]."
        ),
        "Sources": [_source_name(p, bids_dir) for p in sources],
        "TaskKey": task_key,
        "DirconnVersion": dirconn.__version__,
        "DerivativesVersion": DERIVATIVES_VERSION,
        "Statistics": _statistics(tests),
    }
    meta["Bands"] = _band_limits(bands)
    kinds = prov.preprocessing.get("node_kinds")
    if kinds is not None:
        meta["NodeKinds"] = kinds
    sfreq = prov.preprocessing.get("sfreq")
    if sfreq is not None:
        meta["SamplingFrequency"] = sfreq
    if np.iscomplexobj(conn.values):
        meta["ValueIsMagnitude"] = True
    _write_json(sidecar, _plain(meta))

    if long_table:
        df = _edge_rows(conn, bands, tests)
        df.to_csv(edges, sep="\t", index=False, na_rep="n/a")
        return ResultFiles(nc=nc, json=sidecar, edges=edges)
    return ResultFiles(nc=nc, json=sidecar, edges=None)


def _band_limits(bands: Mapping[str, Connectivity]) -> dict[str, list[float] | None]:
    """Band limits in Hz from the ``reduce_freq`` provenance step; ``None`` when not recorded."""
    limits: dict[str, list[float] | None] = {}
    for name, bc in bands.items():
        limits[name] = None
        for step in reversed(bc.provenance.history):
            if step.get("op") == "reduce_freq":
                lo, hi = step["band"]
                limits[name] = [float(lo), float(hi)]
                break
    return limits


def read_edges(path: Path) -> pd.DataFrame:
    """Read an ``_edges.tsv`` table; ``n/a`` becomes NaN (``significant`` keeps booleans)."""
    return pd.read_csv(path, sep="\t", na_values=["n/a"])


def result_task_key(json_path: Path) -> str | None:
    """Return the ``TaskKey`` of a sidecar, or ``None`` if it is missing, unreadable or absent."""
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    key = data.get("TaskKey") if isinstance(data, dict) else None
    return key if isinstance(key, str) else None

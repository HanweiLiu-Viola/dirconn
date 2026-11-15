"""NetCDF input and output of :class:`~dirconn.Connectivity`.

The file stores the result as a ``Dataset`` with the dimensions ``source``, ``target``,
``freq`` and ``time``; complex values are split into ``values_real`` and ``values_imag``.
The provenance and the format version travel in the global attributes.
"""

from __future__ import annotations

import os

import numpy as np
import xarray as xr

from dirconn.data.provenance import Provenance
from dirconn.data.result import DIMS, FORMAT_VERSION, Connectivity


def write_connectivity(conn: Connectivity, path: str | os.PathLike[str]) -> None:
    """Write ``conn`` to a NetCDF file, provenance included.

    Complex values are stored as ``values_real`` and ``values_imag``.
    """
    coords = {
        "source": list(conn.source),
        "target": list(conn.target),
        "freq": conn.freqs,
        "time": conn.times,
    }
    if conn.is_complex:
        data_vars = {
            "values_real": (DIMS, np.ascontiguousarray(conn.values.real)),
            "values_imag": (DIMS, np.ascontiguousarray(conn.values.imag)),
        }
    else:
        data_vars = {"values": (DIMS, np.ascontiguousarray(conn.values))}
    dataset = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "dirconn_format_version": FORMAT_VERSION,
            "provenance": conn.provenance.to_json(),
        },
    )
    dataset.to_netcdf(os.fspath(path), engine="netcdf4")


def read_connectivity(path: str | os.PathLike[str]) -> Connectivity:
    """Read a result written by :func:`write_connectivity`."""
    file = os.fspath(path)
    if not os.path.exists(file):
        raise FileNotFoundError(file)
    with xr.open_dataset(file, engine="netcdf4") as ds:
        version = ds.attrs.get("dirconn_format_version")
        if version is None or str(version) != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported dirconn format version {version!r} (expected {FORMAT_VERSION!r})."
            )
        if "provenance" not in ds.attrs:
            raise ValueError("The file has no provenance attribute.")
        provenance = Provenance.from_json(str(ds.attrs["provenance"]))
        if "values_real" in ds:
            values = ds["values_real"].values + 1j * ds["values_imag"].values
        else:
            values = ds["values"].values
        return Connectivity(
            values=np.array(values),
            source=tuple(str(s) for s in ds["source"].values),
            target=tuple(str(t) for t in ds["target"].values),
            freqs=np.array(ds["freq"].values, dtype=float),
            times=np.array(ds["time"].values, dtype=float),
            provenance=provenance,
        )

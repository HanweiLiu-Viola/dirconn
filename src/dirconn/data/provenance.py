"""Provenance and assumption checklists attached to every result."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata
from typing import Any

import numpy as np

STANDARD_ASSUMPTIONS: tuple[str, ...] = (
    "stationarity",
    "linearity",
    "sampling_adequacy",
    "instantaneous_coupling",
    "causal_sufficiency",
    "filtering",
)
STATUSES: tuple[str, ...] = ("pass", "warn", "fail", "not_checked")
CLAIM = "predictive, not interventional"

_VERSION_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "xarray",
    "netCDF4",
    "mne",
    "mne-bids",
    "mne-connectivity",
    "statsmodels",
    "tigramite",
    "numba",
)


def _jsonable(obj: Any, where: str = "value") -> Any:
    """Return a new structure made only of strict-JSON types, or raise."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if not math.isfinite(float(obj)):
            raise ValueError(f"{where} must be finite to be stored as strict JSON, got {obj!r}.")
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist(), where)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, where) for v in obj]
    if isinstance(obj, Mapping):
        out = {}
        for key, val in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"{where} has a non-string key {key!r}; JSON needs string keys.")
            out[key] = _jsonable(val, f"{where}[{key!r}]")
        return out
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    raise TypeError(f"{where} of type {type(obj).__name__} is not JSON serialisable.")


def collect_versions() -> dict[str, str]:
    """Collect the Python version and the versions of installed relevant packages.

    Looking the versions up takes several milliseconds and they cannot change while the process
    runs, so they are read once; every call returns its own copy. Inside the Docker image the
    image's name and tag (``DIRCONN_IMAGE``) are recorded as ``image``, which names the commit.
    """
    return dict(_installed_versions())


@lru_cache(maxsize=1)
def _installed_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "dirconn": metadata.version("dirconn")}
    if os.environ.get("DIRCONN_IMAGE"):
        versions["image"] = os.environ["DIRCONN_IMAGE"]
    for name in _VERSION_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def array_hash(array: Any) -> str:
    """Return a SHA-256 hex digest of an array's dtype, shape and values.

    The digest does not depend on the memory layout, so a transposed view and its contiguous
    copy hash identically.
    """
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(repr(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class AssumptionCheck:
    """Outcome of one assumption diagnostic.

    Parameters
    ----------
    name : str
        Assumption name, for example ``"stationarity"``.
    status : {"pass", "warn", "fail", "not_checked"}
        Diagnostic outcome. Diagnostics inform; they never block a run.
    detail : str
        Human-readable explanation, for example the test used.
    value : float, optional
        Test statistic or p-value behind the status.
    """

    name: str
    status: str
    detail: str = ""
    value: float | None = None

    def __post_init__(self) -> None:
        """Validate the fields."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Assumption name must be a non-empty string.")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}.")
        if self.value is not None:
            value = float(self.value)
            if not math.isfinite(value):
                raise ValueError(f"value of {self.name!r} must be finite or None.")
            object.__setattr__(self, "value", value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AssumptionCheck:
        """Rebuild a check from :meth:`to_dict` output."""
        return cls(
            name=data["name"],
            status=data["status"],
            detail=data.get("detail", ""),
            value=data.get("value"),
        )


@dataclass(frozen=True)
class AssumptionChecklist:
    """An ordered set of assumption checks with unique names."""

    checks: tuple[AssumptionCheck, ...] = ()

    def __post_init__(self) -> None:
        """Validate that names are unique."""
        checks = tuple(self.checks)
        names = [c.name for c in checks]
        if len(set(names)) != len(names):
            raise ValueError("Assumption names must be unique.")
        object.__setattr__(self, "checks", checks)

    @classmethod
    def unchecked(cls) -> AssumptionChecklist:
        """Return the standard assumptions, all marked ``"not_checked"``."""
        return cls(tuple(AssumptionCheck(name, "not_checked") for name in STANDARD_ASSUMPTIONS))

    def get(self, name: str) -> AssumptionCheck:
        """Return the check called ``name``; raise ``KeyError`` if absent."""
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)

    def with_check(self, check: AssumptionCheck) -> AssumptionChecklist:
        """Return a new checklist where ``check`` replaces the entry of the same name."""
        replaced = False
        items = []
        for existing in self.checks:
            if existing.name == check.name:
                items.append(check)
                replaced = True
            else:
                items.append(existing)
        if not replaced:
            items.append(check)
        return AssumptionChecklist(tuple(items))

    @property
    def has_warnings(self) -> bool:
        """Whether any check has status ``"warn"`` or ``"fail"``."""
        return any(c.status in ("warn", "fail") for c in self.checks)

    def to_list(self) -> list[dict[str, Any]]:
        """Return a JSON-compatible list of dictionaries."""
        return [c.to_dict() for c in self.checks]

    @classmethod
    def from_list(cls, items: Sequence[Mapping[str, Any]]) -> AssumptionChecklist:
        """Rebuild a checklist from :meth:`to_list` output."""
        return cls(tuple(AssumptionCheck.from_dict(item) for item in items))


@dataclass(frozen=True)
class Provenance:
    """Everything needed to understand and reproduce a result.

    All free-form fields are normalised to strict JSON types on construction, so a stored
    provenance never aliases the caller's objects and always survives a JSON round trip.

    Parameters
    ----------
    method : str
        Name of the estimator that produced the result.
    parameters : dict
        Estimator parameters.
    model_order : int, optional
        Model order used, when the method has one.
    seed : int, optional
        Random seed, when the method or its statistics are stochastic.
    backend : str, optional
        The implementation actually used, for example ``"own numpy OLS MVAR"``.
    input_hash : str, optional
        :func:`array_hash` of the input data.
    preprocessing : dict
        Declared preprocessing (filter, reference, standardisation, ...).
    versions : dict
        Package versions; collected from the current environment by default.
    assumptions : AssumptionChecklist
        Assumption diagnostics; all ``"not_checked"`` by default.
    claim : str
        What the result claims. Always ``"predictive, not interventional"`` unless overridden.
    history : tuple of dict
        Post-processing steps applied after estimation, oldest first.
    details : dict
        Method-specific facts derived during estimation, for example the order-selection curves or
        the number of observations. Normalised to strict JSON like ``parameters``.
    """

    method: str
    parameters: dict[str, Any] = field(default_factory=dict)
    model_order: int | None = None
    seed: int | None = None
    backend: str | None = None
    input_hash: str | None = None
    preprocessing: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=collect_versions)
    assumptions: AssumptionChecklist = field(default_factory=AssumptionChecklist.unchecked)
    claim: str = CLAIM
    history: tuple[dict[str, Any], ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalise the fields."""
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("Provenance method must be a non-empty string.")
        set_ = object.__setattr__
        set_(self, "parameters", _jsonable(dict(self.parameters), "parameters"))
        set_(self, "preprocessing", _jsonable(dict(self.preprocessing), "preprocessing"))
        set_(self, "versions", _jsonable(dict(self.versions), "versions"))
        set_(self, "history", tuple(_jsonable(dict(step), "history") for step in self.history))
        set_(self, "details", _jsonable(dict(self.details), "details"))
        set_(self, "model_order", None if self.model_order is None else int(self.model_order))
        set_(self, "seed", None if self.seed is None else int(self.seed))

    def with_step(self, step: Mapping[str, Any]) -> Provenance:
        """Return a copy with a post-processing step appended to ``history``."""
        return Provenance(
            method=self.method,
            parameters=self.parameters,
            model_order=self.model_order,
            seed=self.seed,
            backend=self.backend,
            input_hash=self.input_hash,
            preprocessing=self.preprocessing,
            versions=self.versions,
            assumptions=self.assumptions,
            claim=self.claim,
            history=(*self.history, dict(step)),
            details=self.details,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "method": self.method,
            "parameters": self.parameters,
            "model_order": self.model_order,
            "seed": self.seed,
            "backend": self.backend,
            "input_hash": self.input_hash,
            "preprocessing": self.preprocessing,
            "versions": self.versions,
            "assumptions": self.assumptions.to_list(),
            "claim": self.claim,
            "history": list(self.history),
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Provenance:
        """Rebuild a provenance from :meth:`to_dict` output."""
        if "method" not in data:
            raise ValueError("Provenance data has no 'method' entry.")
        return cls(
            method=data["method"],
            parameters=data.get("parameters", {}),
            model_order=data.get("model_order"),
            seed=data.get("seed"),
            backend=data.get("backend"),
            input_hash=data.get("input_hash"),
            preprocessing=data.get("preprocessing", {}),
            versions=data.get("versions", {}),
            assumptions=AssumptionChecklist.from_list(data.get("assumptions", [])),
            claim=data.get("claim", CLAIM),
            history=tuple(data.get("history", ())),
            details=data.get("details", {}),
        )

    def to_json(self, indent: int | None = None) -> str:
        """Serialise to a strict JSON string."""
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Provenance:
        """Rebuild a provenance from :meth:`to_json` output."""
        return cls.from_dict(json.loads(text))

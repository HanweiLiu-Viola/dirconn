"""Typed pipeline configuration: pydantic models, TOML loading, overrides, hashes and dumping.

The configuration is a tree of frozen, strict models (unknown keys are errors). The defaults live
twice on purpose, in the model fields and in the commented ``_data/default_config.toml``, and a test
asserts the two agree. ``config_hash`` identifies a whole configuration, ``analysis_hash`` only the
sections that change numerical results (it keys per-task caches).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dirconn.methods.registry import REGISTRY

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib

__all__ = [
    "CORRECTION_METHODS",
    "SURROGATES",
    "ControlsConfig",
    "EstimationConfig",
    "ExecutionConfig",
    "InputConfig",
    "LockConfig",
    "OutputConfig",
    "PipelineConfig",
    "PrepareConfig",
    "StatisticsConfig",
    "analysis_hash",
    "apply_overrides",
    "config_hash",
    "dump_config",
    "load_config",
]

#: Surrogate generators selectable by name (``block_shuffle`` needs a block size, so it is not).
SURROGATES: tuple[str, ...] = (
    "phase_randomize",
    "aaft",
    "iaaft",
    "circular_shift",
    "trial_shuffle",
)
#: Multiple-comparison corrections accepted by ``dirconn.stats.correction.adjust_pvalues``.
CORRECTION_METHODS: tuple[str, ...] = ("fdr_bh", "bonferroni", "none")

_SCHEMA_VERSION = 1


_NULL = "null"


def _null_sentinel(value: Any) -> Any:
    """Map the string ``"null"`` (any case) to ``None``; TOML has no null of its own."""
    if isinstance(value, str) and value.strip().lower() == _NULL:
        return None
    return value


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputConfig(_Section):
    """What to analyse: node level, data types, node kinds and the parcellation description.

    Parameters
    ----------
    level : {"roi", "sensor"}
        Node level of the input time series.
    datatypes : tuple of str
        BIDS data types to include.
    node_kinds : tuple of str
        Kinds of nodes to keep (for example ``"cortical"``, ``"direct"``).
    description : str or None
        BIDS ``desc`` entity selecting the parcellation, or ``None`` for no filter.
    """

    level: Literal["roi", "sensor"] = "roi"
    datatypes: tuple[str, ...] = ("meg", "eeg", "ieeg")
    node_kinds: tuple[str, ...] = ("cortical", "direct")
    description: str | None = "DK"

    _description_null = field_validator("description", mode="before")(_null_sentinel)


class LockConfig(_Section):
    """Event-locked epoching.

    Parameters
    ----------
    events : tuple of str
        Event names to lock to; empty means continuous data.
    tmin, tmax : float
        Window around each event in seconds.
    """

    events: tuple[str, ...] = ()
    tmin: float = -1.0
    tmax: float = 1.0

    @model_validator(mode="after")
    def _check(self) -> LockConfig:
        if not self.tmin < self.tmax:
            raise ValueError(f"tmin must be below tmax, got ({self.tmin}, {self.tmax}).")
        return self


class PrepareConfig(_Section):
    """Preprocessing applied before estimation.

    Parameters
    ----------
    resample_hz : float or None
        Target sampling rate, or ``None`` to keep the original one.
    zscore : bool
        Standardise each channel.
    bad_segments : {"mask", "ignore"}
        Mask annotated bad segments out of the estimate, or ignore the annotations.
    pad_bad_seconds : float
        Extra seconds masked on both sides of each bad segment.
    lock : LockConfig
        Event-locked epoching.
    """

    resample_hz: float | None = Field(default=250.0, gt=0)
    zscore: bool = True
    bad_segments: Literal["mask", "ignore"] = "mask"
    pad_bad_seconds: float = Field(default=0.0, ge=0)
    lock: LockConfig = LockConfig()

    _resample_null = field_validator("resample_hz", mode="before")(_null_sentinel)


def _default_bands() -> dict[str, tuple[float, float]]:
    return {"theta": (4.0, 8.0), "beta_low": (13.0, 20.0), "beta_high": (20.0, 30.0)}


class EstimationConfig(_Section):
    """Connectivity estimation.

    Parameters
    ----------
    methods : tuple of str
        Method names from the registry.
    order : int or {"aic", "bic", "hqc", "cv"}
        Model order or the criterion that selects it.
    order_max : int
        Upper bound for criterion-based order selection.
    bands : dict of str to (float, float)
        Named frequency bands in Hz; each needs ``0 <= lo < hi``.
    reduce : {"mean", "median", "max", "peak_freq"}
        How a spectrum is reduced to one value per band.
    exclude_freqs : tuple of (float, float)
        Frequency ranges (for example line noise) excluded from band reduction.
    params : dict of str to dict
        Per-method keyword arguments; the TOML sub-table ``[estimation.<method>]``.
    """

    methods: tuple[str, ...] = ("gc",)
    order: int | Literal["aic", "bic", "hqc", "cv"] = "bic"
    order_max: int = Field(default=30, ge=1)
    bands: dict[str, tuple[float, float]] = _default_bands()
    reduce: Literal["mean", "median", "max", "peak_freq"] = "mean"
    exclude_freqs: tuple[tuple[float, float], ...] = ()
    params: dict[str, dict[str, Any]] = {}

    @model_validator(mode="before")
    @classmethod
    def _tables_to_params(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        known = set(cls.model_fields)
        moved = {k: v for k, v in data.items() if k not in known and isinstance(v, Mapping)}
        if not moved:
            return data
        out = {k: v for k, v in data.items() if k not in moved}
        out["params"] = {**dict(out.get("params") or {}), **moved}
        return out

    @field_validator("order")
    @classmethod
    def _order_positive(cls, value: int | str) -> int | str:
        if isinstance(value, int) and value < 1:
            raise ValueError(f"order must be at least 1, got {value}.")
        return value

    @model_validator(mode="after")
    def _check(self) -> EstimationConfig:
        unknown = [m for m in self.methods if m not in REGISTRY]
        if unknown:
            raise ValueError(f"unknown method(s) {unknown}; available: {sorted(REGISTRY)}.")
        if not self.methods:
            raise ValueError("methods must not be empty.")
        if not self.bands:
            raise ValueError("bands must not be empty.")
        for name, (lo, hi) in self.bands.items():
            if lo < 0 or not lo < hi:
                raise ValueError(f"band {name!r} must satisfy 0 <= lo < hi, got ({lo}, {hi}).")
        for lo, hi in self.exclude_freqs:
            if lo < 0 or not lo < hi:
                raise ValueError(
                    f"exclude_freqs range must satisfy 0 <= lo < hi, got ({lo}, {hi})."
                )
        stray = [m for m in self.params if m not in self.methods]
        if stray:
            raise ValueError(f"params given for method(s) {stray} that are not in methods.")
        return self


class StatisticsConfig(_Section):
    """Significance testing.

    Parameters
    ----------
    test : {"surrogate", "analytic", "none"}
        Kind of test.
    surrogate : str
        Surrogate generator name, one of ``SURROGATES``.
    n_surrogates : int
        Number of surrogates.
    alpha : float
        Significance level.
    correction : str
        Multiple-comparison correction, one of ``CORRECTION_METHODS``.
    """

    test: Literal["surrogate", "analytic", "none"] = "surrogate"
    surrogate: str = "trial_shuffle"
    n_surrogates: int = Field(default=200, ge=1)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    correction: str = "fdr_bh"

    @model_validator(mode="after")
    def _check(self) -> StatisticsConfig:
        if self.surrogate not in SURROGATES:
            raise ValueError(f"surrogate must be one of {SURROGATES}, got {self.surrogate!r}.")
        if self.correction not in CORRECTION_METHODS:
            raise ValueError(
                f"correction must be one of {CORRECTION_METHODS}, got {self.correction!r}."
            )
        return self


class ControlsConfig(_Section):
    """Control analyses.

    Parameters
    ----------
    time_reversal : bool
        Also compute time-reversed controls.
    spectral_range_check : bool
        Check that the requested bands lie within the usable spectral range.
    """

    time_reversal: bool = True
    spectral_range_check: bool = True


class OutputConfig(_Section):
    """Output options.

    Parameters
    ----------
    long_table : bool
        Write the long-format results table.
    """

    long_table: bool = True


class ExecutionConfig(_Section):
    """Execution options.

    Parameters
    ----------
    seed : int
        Base random seed.
    fail_fast : bool
        Stop at the first failing task instead of recording it and continuing.
    """

    seed: int = 20261115
    fail_fast: bool = False


class PipelineConfig(_Section):
    """The full pipeline configuration.

    Parameters
    ----------
    schema_version : int
        Must be 1.
    input, prepare, estimation, statistics, controls, output, execution
        The sections; see their classes.
    """

    schema_version: int = _SCHEMA_VERSION
    input: InputConfig = InputConfig()
    prepare: PrepareConfig = PrepareConfig()
    estimation: EstimationConfig = EstimationConfig()
    statistics: StatisticsConfig = StatisticsConfig()
    controls: ControlsConfig = ControlsConfig()
    output: OutputConfig = OutputConfig()
    execution: ExecutionConfig = ExecutionConfig()

    @model_validator(mode="after")
    def _check(self) -> PipelineConfig:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version}; expected {_SCHEMA_VERSION}."
            )
        fs = self.prepare.resample_hz
        if fs is not None:
            nyquist = fs / 2.0
            for name, (_, hi) in self.estimation.bands.items():
                if hi >= nyquist:
                    raise ValueError(
                        f"band {name!r} upper edge {hi} Hz is not below the Nyquist frequency "
                        f"{nyquist} Hz (resample_hz={fs})."
                    )
        return self


def load_config(path: Path | None) -> PipelineConfig:
    """Load a configuration from a TOML file.

    Parameters
    ----------
    path : Path or None
        TOML file. ``None`` loads the packaged ``default_config.toml``.

    Returns
    -------
    PipelineConfig
        The validated configuration. Sub-tables of ``[estimation]`` that are not declared fields
        (``[estimation.gc]``) become ``estimation.params``.
    """
    if path is None:
        text = (
            resources.files("dirconn.workflows._data")
            .joinpath("default_config.toml")
            .read_text(encoding="utf-8")
        )
    else:
        text = Path(path).read_text(encoding="utf-8")
    return PipelineConfig.model_validate(tomllib.loads(text))


def _parse_scalar(text: str) -> Any:
    low = text.strip().lower()
    if low == "null":
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def apply_overrides(cfg: PipelineConfig, overrides: Sequence[str]) -> PipelineConfig:
    """Return a copy of ``cfg`` with ``"dotted.path=value"`` overrides applied.

    Parameters
    ----------
    cfg : PipelineConfig
        Base configuration.
    overrides : sequence of str
        Each ``"section.field=value"``, split on the first ``=``. Fields holding a tuple or list
        take comma-separated values, ``null`` means ``None`` and booleans are ``true``/``false``.
        ``estimation.params.<method>.<key>=value`` creates the nested tables.

    Returns
    -------
    PipelineConfig
        The re-validated configuration.

    Raises
    ------
    ValueError
        For a malformed override or an unknown path. Invalid values raise pydantic's
        ``ValidationError`` (a ``ValueError`` subclass).
    """
    data = cfg.model_dump()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must have the form 'section.field=value'.")
        path, raw = item.split("=", 1)
        keys = path.strip().split(".")
        in_params = keys[:2] == ["estimation", "params"] and len(keys) >= 4
        node = data
        for i, key in enumerate(keys[:-1]):
            if not isinstance(node, dict):
                raise ValueError(f"unknown config path {path!r}.")
            if key not in node:
                if in_params and i >= 2:
                    node[key] = {}
                else:
                    raise ValueError(f"unknown config path {path!r}.")
            node = node[key]
        leaf = keys[-1]
        if not isinstance(node, dict) or (leaf not in node and not in_params):
            raise ValueError(f"unknown config path {path!r}.")
        if isinstance(node[leaf] if leaf in node else None, dict):
            raise ValueError(f"config path {path!r} is a section; set one of its fields.")
        current = node.get(leaf)
        if isinstance(current, (tuple, list)):
            node[leaf] = tuple(p.strip() for p in raw.split(",")) if raw.strip() else ()
        elif in_params:
            node[leaf] = _parse_scalar(raw)
        else:
            low = raw.strip().lower()
            node[leaf] = None if low == "null" else raw
    return PipelineConfig.model_validate(data)


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def config_hash(cfg: PipelineConfig) -> str:
    """Return a 16-hex-digit digest of the whole configuration.

    Parameters
    ----------
    cfg : PipelineConfig
        Configuration to hash.

    Returns
    -------
    str
        First 16 hex characters of the SHA-256 of the canonical JSON dump.
    """
    return _digest(cfg.model_dump(mode="json"))


def analysis_hash(cfg: PipelineConfig) -> str:
    """Return a digest of the sections that determine numerical results.

    Parameters
    ----------
    cfg : PipelineConfig
        Configuration to hash.

    Returns
    -------
    str
        Digest over ``prepare``, ``estimation``, ``statistics`` and ``controls`` only, so changing
        ``input``, ``output`` or ``execution`` leaves it unchanged.
    """
    dump = cfg.model_dump(mode="json")
    return _digest({k: dump[k] for k in ("prepare", "estimation", "statistics", "controls")})


def _scalar(value: Any) -> str:
    if value is None:
        return json.dumps(_NULL)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_scalar(v) for v in value if v is not None) + "]"
    if isinstance(value, (int, float)):
        return repr(value)
    raise TypeError(f"cannot write {type(value).__name__} value {value!r} as TOML.")


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(name: str) -> str:
    return name if _BARE_KEY.match(name) else json.dumps(name)


def _to_toml(mapping: Mapping[str, Any], _prefix: tuple[str, ...] = ()) -> str:
    """Serialise nested mappings as TOML: scalars first, then ``[a.b]`` sub-tables depth-first."""
    scalars = [(k, v) for k, v in mapping.items() if not isinstance(v, Mapping)]
    tables = [(k, v) for k, v in mapping.items() if isinstance(v, Mapping)]
    lines: list[str] = []
    if _prefix and (scalars or not tables):
        lines.append("[" + ".".join(_key(p) for p in _prefix) + "]")
    lines.extend(f"{_key(k)} = {_scalar(v)}" for k, v in scalars)
    out = "\n".join(lines) + "\n" if lines else ""
    for k, v in tables:
        sub = _to_toml(v, (*_prefix, k))
        if sub:
            out += ("\n" if out else "") + sub
    return out


def dump_config(cfg: PipelineConfig, path: Path) -> None:
    """Write ``cfg`` as TOML such that ``load_config`` reads it back equal.

    Parameters
    ----------
    cfg : PipelineConfig
        Configuration to write. ``None`` is written as the string ``"null"`` (read back as
        ``None`` by the nullable fields) and ``estimation.params`` is written as
        ``[estimation.<method>]`` sub-tables.
    path : Path
        Destination file.
    """
    data = cfg.model_dump()
    params = data["estimation"].pop("params")
    data["estimation"].update(params)
    Path(path).write_text(_to_toml(data), encoding="utf-8")

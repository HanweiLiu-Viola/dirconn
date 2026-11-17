"""Discovery of the BIDS recordings a pipeline run operates on.

A *unit* is one recording, identified by ``(subject, session, task, run)``. At ``level="roi"`` the
units are the region-level time series of a dirconn derivatives dataset; at ``level="sensor"`` they
are the raw recordings of the BIDS dataset itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mne_bids import find_matching_paths

__all__ = ["InputDiscoveryError", "InputUnit", "derivatives_root", "discover"]

_SUFFIXES = ["eeg", "meg", "ieeg"]
_EXTENSIONS = [".fif", ".vhdr", ".edf", ".set"]
_FILTER_KEYS = ("subject", "session", "task", "run")
_PREFIX = {"subject": "sub-", "session": "ses-", "task": "task-", "run": "run-"}


class InputDiscoveryError(ValueError):
    """Raised when the inputs of a run cannot be determined unambiguously."""


@dataclass(frozen=True)
class InputUnit:
    """One recording of a BIDS dataset.

    Parameters
    ----------
    subject, session, task, run : str or None
        BIDS entity values without the ``sub-`` / ``ses-`` / ``task-`` / ``run-`` prefix.
    datatype : str
        BIDS datatype directory, e.g. ``"eeg"``.
    level : str
        ``"roi"`` or ``"sensor"``.
    root : Path
        Root of the dataset the unit was found in.
    files : tuple of Path
        Absolute paths: the data file (and its BrainVision / EEGLAB data companions), then its
        ``.json`` sidecar, ``channels.tsv`` and ``events.tsv`` if present.
    sha256 : str
        Hex digest over the concatenated bytes of ``files``, in order.
    """

    subject: str
    session: str | None
    task: str
    run: str | None
    datatype: str
    level: str
    root: Path
    files: tuple[Path, ...]
    sha256: str

    def entities(self) -> dict[str, str]:
        """Entity values that are set, keyed ``sub``, ``ses``, ``task``, ``run``."""
        pairs = (
            ("sub", self.subject),
            ("ses", self.session),
            ("task", self.task),
            ("run", self.run),
        )
        return {key: value for key, value in pairs if value}

    def label(self) -> str:
        """Human-readable identifier, e.g. ``sub-01_ses-01_task-rest_run-1``."""
        return "_".join(f"{key}-{value}" for key, value in self.entities().items())


def derivatives_root(bids_dir: Path) -> Path:
    """Locate the dirconn derivatives dataset below ``bids_dir/derivatives``.

    Parameters
    ----------
    bids_dir : Path
        Root of the BIDS dataset.

    Returns
    -------
    Path
        The only derivatives directory whose ``GeneratedBy`` lists ``Name == "dirconn"``.

    Raises
    ------
    InputDiscoveryError
        If there is no such directory or more than one.
    """
    bids_dir = Path(bids_dir)
    found: list[Path] = []
    for description in sorted((bids_dir / "derivatives").glob("*/dataset_description.json")):
        try:
            data = json.loads(description.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        generated_by = data.get("GeneratedBy") if isinstance(data, dict) else None
        if not isinstance(generated_by, list):
            continue
        if any(isinstance(g, dict) and g.get("Name") == "dirconn" for g in generated_by):
            found.append(description.parent)
    if len(found) == 1:
        return found[0]
    expected = (
        'expected exactly one directory below "derivatives" whose dataset_description.json '
        'lists GeneratedBy Name == "dirconn"'
    )
    if not found:
        raise InputDiscoveryError(f"No dirconn derivatives found in {bids_dir}; {expected}.")
    names = ", ".join(p.name for p in found)
    raise InputDiscoveryError(
        f"Several dirconn derivatives found in {bids_dir} ({names}); {expected}."
    )


def _strip(value: object, key: str) -> str:
    text = str(value)
    prefix = _PREFIX[key]
    return text[len(prefix) :] if text.startswith(prefix) else text


def _normalise(values: Sequence[Any] | None, key: str) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise InputDiscoveryError(f"Filter {key!r} must be a list of values, got {values!r}.")
    return {_strip(v, key) for v in values}


def _read_filter_file(path: Path) -> dict[str, set[str]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputDiscoveryError(f"Cannot read filter file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise InputDiscoveryError(f"Filter file {path} must contain a JSON object.")
    unknown = sorted(set(data) - set(_FILTER_KEYS))
    if unknown:
        raise InputDiscoveryError(
            f"Filter file {path} has unknown keys {unknown}; allowed: {list(_FILTER_KEYS)}."
        )
    result: dict[str, set[str]] = {}
    for key, values in data.items():
        normalised = _normalise(values, key)
        assert normalised is not None
        result[key] = normalised
    return result


def _combine_filters(
    cli: dict[str, set[str] | None], from_file: dict[str, set[str]]
) -> dict[str, set[str]]:
    combined: dict[str, set[str]] = {}
    for key in _FILTER_KEYS:
        parts = [s for s in (cli.get(key), from_file.get(key)) if s is not None]
        if parts:
            combined[key] = set.intersection(*parts)
    return combined


def _sha256(files: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for file in files:
        content = file.read_bytes()
        digest.update(f"{file.name}\0{len(content)}\0".encode())
        digest.update(content)
    return digest.hexdigest()


def _unit_files(data_file: Path) -> tuple[Path, ...]:
    stem = data_file.stem
    base, _, _ = stem.rpartition("_")
    files = [data_file]
    data_companions: tuple[str, ...] = ()
    if data_file.suffix == ".vhdr":
        data_companions = (".eeg", ".vmrk")
    elif data_file.suffix == ".set":
        data_companions = (".fdt",)
    for ext in data_companions:
        data_part = data_file.with_suffix(ext)
        if data_part.is_file():
            files.append(data_part)
    companions = (
        data_file.with_suffix(".json"),
        data_file.with_name(f"{base}_channels.tsv"),
        data_file.with_name(f"{base}_events.tsv"),
    )
    for companion in companions:
        if companion.is_file():
            files.append(companion)
    return tuple(files)


def discover(
    bids_dir: Path,
    *,
    level: str,
    datatypes: Sequence[str],
    description: str | None = None,
    participant_labels: Sequence[str] | None = None,
    session_ids: Sequence[str] | None = None,
    task_ids: Sequence[str] | None = None,
    run_ids: Sequence[str] | None = None,
    filter_file: Path | None = None,
) -> list[InputUnit]:
    """Find the recordings of a run.

    Parameters
    ----------
    bids_dir : Path
        Root of the BIDS dataset.
    level : {"roi", "sensor"}
        ``"roi"`` reads the dirconn derivatives dataset, ``"sensor"`` the raw dataset.
    datatypes : sequence of str
        BIDS datatypes to search, e.g. ``["eeg"]``.
    description : str, optional
        Required ``desc-`` entity at ``level="roi"``; ``None`` accepts any.
    participant_labels, session_ids, task_ids, run_ids : sequence of str, optional
        Entity filters; values may carry the ``sub-`` / ``ses-`` / ``task-`` / ``run-`` prefix.
    filter_file : Path, optional
        JSON file with the keys ``subject session task run`` (lists of values). Combined with the
        other filters by intersection.

    Returns
    -------
    list of InputUnit
        Sorted by ``(subject, session, task, run)``.

    Raises
    ------
    InputDiscoveryError
        If nothing matches, the dataset layout is ambiguous, or a filter file is invalid.
    """
    if level not in ("roi", "sensor"):
        raise InputDiscoveryError(f"Unknown input level {level!r}; use 'roi' or 'sensor'.")
    bids_dir = Path(bids_dir)
    root = derivatives_root(bids_dir) if level == "roi" else bids_dir
    cli = {
        "subject": _normalise(participant_labels, "subject"),
        "session": _normalise(session_ids, "session"),
        "task": _normalise(task_ids, "task"),
        "run": _normalise(run_ids, "run"),
    }
    from_file = _read_filter_file(filter_file) if filter_file is not None else {}
    filters = _combine_filters(cli, from_file)

    paths = find_matching_paths(
        root,
        datatypes=list(datatypes),
        suffixes=_SUFFIXES,
        extensions=_EXTENSIONS,
        check=False,
    )
    groups: dict[tuple[str, str, str, str], list[Any]] = {}
    derivatives_dir = root / "derivatives"
    for path in paths:
        if level == "sensor" and derivatives_dir in Path(path.fpath).parents:
            continue  # derivatives nested in the raw dataset are not raw recordings
        if level == "roi" and description is not None and path.description != description:
            continue
        entities = {
            "subject": path.subject,
            "session": path.session,
            "task": path.task,
            "run": path.run,
        }
        if any(
            key in filters and (entities[key] or "") not in filters[key] for key in _FILTER_KEYS
        ):
            continue
        group_key = (
            entities["subject"] or "",
            entities["session"] or "",
            entities["task"] or "",
            entities["run"] or "",
        )
        groups.setdefault(group_key, []).append(path)

    units: list[InputUnit] = []
    for group_key in sorted(groups):
        members = groups[group_key]
        if len(members) > 1:
            names = ", ".join(sorted(m.fpath.name for m in members))
            raise InputDiscoveryError(
                f"Several data files for one recording ({names}); split or duplicate "
                "recordings are not supported."
            )
        path = members[0]
        files = _unit_files(Path(path.fpath).resolve())
        subject, session, task, run = group_key
        units.append(
            InputUnit(
                subject=subject,
                session=session or None,
                task=task,
                run=run or None,
                datatype=str(path.datatype),
                level=level,
                root=root,
                files=files,
                sha256=_sha256(files),
            )
        )
    if not units:
        applied: dict[str, Any] = {"level": level, "datatypes": list(datatypes), "root": str(root)}
        if level == "roi":
            applied["description"] = description
        applied.update({k: sorted(v) for k, v in filters.items()})
        text = ", ".join(f"{k}={v!r}" for k, v in applied.items())
        raise InputDiscoveryError(f"No recordings found with the filters: {text}.")
    return units

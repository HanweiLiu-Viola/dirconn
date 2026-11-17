"""Table of readers keyed by ``(datatype, level)``, used by the participant pipeline."""

from __future__ import annotations

from collections.abc import Callable, Collection
from pathlib import Path

from dirconn.data.timeseries import TimeSeries
from dirconn.io.bids import read_bids

Reader = Callable[..., TimeSeries]
"""``reader(root, *, subject, session, task, run, datatype, description, node_kinds,
bad_annotations)``."""

_DATATYPES = ("eeg", "meg", "ieeg")


def _roi_reader(
    root: Path,
    *,
    subject: str,
    session: str | None,
    task: str,
    run: int | str | None,
    datatype: str,
    description: str | None,
    node_kinds: Collection[str],
    bad_annotations: str = "mask",
) -> TimeSeries:
    # bad_annotations only applies to sensor recordings: NaN in an ROI file is always masked.
    series = read_bids(
        root,
        subject=subject,
        task=task,
        session=session,
        run=run,
        datatype=datatype,
        level="roi",
        description=description,
        nan="mask",
    )
    keep = [ch.name for ch in series.channels if ch.node_kind in node_kinds]
    if len(keep) < 2:
        present = sorted({ch.node_kind for ch in series.channels})
        raise ValueError(
            f"node_kinds={sorted(node_kinds)} leaves {len(keep)} channel(s) (at least 2 are "
            f"needed); the recording has node kinds {present}."
        )
    return series.pick(keep)


def _sensor_reader(
    root: Path,
    *,
    subject: str,
    session: str | None,
    task: str,
    run: int | str | None,
    datatype: str,
    description: str | None = None,
    node_kinds: Collection[str] = (),
    bad_annotations: str = "mask",
) -> TimeSeries:
    # description and node_kinds only apply to ROI derivatives.
    return read_bids(
        root,
        subject=subject,
        task=task,
        session=session,
        run=run,
        datatype=datatype,
        level="sensor",
        nan="mask",
        bad_annotations=bad_annotations,
    )


READERS: dict[tuple[str, str], Reader] = {
    **{(d, "roi"): _roi_reader for d in _DATATYPES},
    **{(d, "sensor"): _sensor_reader for d in _DATATYPES},
}


def read_unit(
    root: Path,
    *,
    subject: str,
    session: str | None,
    task: str,
    run: int | str | None,
    datatype: str,
    level: str,
    description: str | None,
    node_kinds: Collection[str],
    bad_annotations: str = "mask",
) -> TimeSeries:
    """Read one analysis unit with the reader registered for ``(datatype, level)``.

    Parameters
    ----------
    root : Path
        BIDS root (sensor level) or ROI derivatives root.
    subject, session, task, run, datatype : str or None
        Entities of the unit.
    level : {"roi", "sensor"}
        Node level; selects the reader together with ``datatype``.
    description : str or None
        ``desc`` entity of the ROI file (ROI level only).
    node_kinds : collection of str
        Node kinds to keep (ROI level only).
    bad_annotations : {"mask", "ignore"}
        Sensor level: ``"mask"`` marks ``BAD_`` annotated samples invalid, ``"ignore"`` keeps
        them as recorded. NaN samples are masked in both modes and at both levels.
    """
    try:
        reader = READERS[(datatype, level)]
    except KeyError:
        known = ", ".join(f"({d!r}, {lv!r})" for d, lv in sorted(READERS))
        raise KeyError(f"No reader for ({datatype!r}, {level!r}); known: {known}.") from None
    return reader(
        Path(root),
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype=datatype,
        description=description,
        node_kinds=node_kinds,
        bad_annotations=bad_annotations,
    )

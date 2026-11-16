"""Table of readers keyed by ``(datatype, level)``, used by the participant pipeline."""

from __future__ import annotations

from collections.abc import Callable, Collection
from pathlib import Path

from dirconn.data.timeseries import TimeSeries
from dirconn.io.bids import read_bids

Reader = Callable[..., TimeSeries]
"""``reader(root, *, subject, session, task, run, datatype, description, node_kinds)``."""

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
) -> TimeSeries:
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
        bad_annotations="mask",
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
) -> TimeSeries:
    """Read one analysis unit with the reader registered for ``(datatype, level)``."""
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
    )

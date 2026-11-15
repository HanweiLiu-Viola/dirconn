"""Access to the MNE ``sample`` dataset used by the benchmarks and tutorials.

The dataset is about 1.5 GB and is never downloaded implicitly. :func:`require_sample` looks for it
locally and, when it is missing, stops with instructions; it downloads only when called with
``download=True``.
"""

from __future__ import annotations

from pathlib import Path

SAMPLE_SIZE = "1.5 GB"

#: Files, relative to the dataset root, that the benchmark reads.
SAMPLE_FILES: tuple[str, ...] = (
    "MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif",
    "MEG/sample/sample_audvis_raw.fif",
    "MEG/sample/ernoise_raw.fif",
    "subjects/sample/label/lh.aparc.annot",
    "subjects/sample/label/rh.aparc.annot",
)


class SampleDataMissingError(FileNotFoundError):
    """The MNE ``sample`` dataset, or part of it, is not available locally."""


def _missing(root: Path) -> list[str]:
    if not root.is_dir():
        return list(SAMPLE_FILES)
    return [name for name in SAMPLE_FILES if not (root / name).is_file()]


def _instructions(missing: list[str], location: str) -> str:
    shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
    return (
        f"The MNE sample dataset ({SAMPLE_SIZE}) is not available {location}; missing: {shown}.\n"
        "It is never downloaded implicitly. To get it:\n"
        "  - call dirconn.datasets.require_sample(download=True), or\n"
        '  - run: python -c "import mne; mne.datasets.sample.data_path(download=True)"\n'
        "To keep it somewhere else, set the MNE_DATA environment variable (or pass path=) to an "
        "existing directory first."
    )


def require_sample(*, path: str | Path | None = None, download: bool = False) -> Path:
    """Return the root of the MNE ``sample`` dataset, or stop with instructions.

    Looks for the dataset locally and checks that the files the benchmark reads are present. It
    never changes MNE's global configuration (``update_path=False``).

    Parameters
    ----------
    path : path-like, optional
        Directory that contains (or will contain) ``MNE-sample-data``. By default MNE's own
        search order applies (``MNE_DATA``, its configuration, then ``~/mne_data``).
    download : bool
        ``True`` is the explicit confirmation to download the dataset (about 1.5 GB) when it is
        missing. Nothing is downloaded if the data are already complete.

    Returns
    -------
    pathlib.Path
        The ``MNE-sample-data`` directory.

    Raises
    ------
    SampleDataMissingError
        If the dataset is missing or incomplete and ``download`` is ``False``, if ``path`` does
        not exist, or if a download did not supply the files.
    TypeError
        If ``download`` is not a bool.
    """
    import mne

    if not isinstance(download, bool):
        raise TypeError(f"download must be a bool, got {download!r}.")
    if path is not None and not Path(path).is_dir():
        raise SampleDataMissingError(
            f"The directory {str(path)!r} does not exist; create it first or pass another path.\n"
            + _instructions(list(SAMPLE_FILES), "there")
        )

    def locate(*, allow_download: bool) -> Path:
        return Path(
            mne.datasets.sample.data_path(
                path=path, download=allow_download, update_path=False, verbose="ERROR"
            )
        )

    root = locate(allow_download=False)
    missing = _missing(root)
    if not missing:
        return root
    if not download:
        raise SampleDataMissingError(_instructions(missing, "locally"))

    root = locate(allow_download=True)
    still = _missing(root)
    if still:
        raise SampleDataMissingError(
            "The download finished but files are still missing: "
            + ", ".join(still[:5])
            + ".\n"
            + _instructions(still, f"in {root}")
        )
    return root

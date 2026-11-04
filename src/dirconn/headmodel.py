"""Head model of the MNE ``sample`` subject: sensors, DK regions and the leadfield between them.

A region (ROI) is a DK (``aparc``) label. Its topography on the sensors is the **mean of the gains
of the label's source-space vertices**, which is the field of a patch with unit average current
density. The sources are fixed-orientation dipoles along the cortical surface normal, so the
vertices of a folded patch partly cancel; regions therefore differ in how visible they are on the
sensors (``HeadModel.visibility``) and the loader reports that instead of hiding it by normalising
every region to the same strength. One global constant per modality brings the median region to a
topography norm of 1 so that simulated sensor signals are of order one.

EEG is average-referenced: the stored forward solution is relative to an infinite reference and
has a large common-mode component, which would make every simulated electrode share a signal that
no recording has. ``HeadModel.forward`` is the stored, unreferenced solution; an inverse method
must add the average-reference projector to its ``info`` (the beamformer weights then ignore the
common mode).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from dirconn.datasets import require_sample

MODALITIES: tuple[str, ...] = ("eeg", "mag", "grad")

_FORWARD_FILE = "MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"
_RAW_FILE = "MEG/sample/sample_audvis_raw.fif"

# arguments of mne.pick_types for each modality
_PICKS: dict[str, dict[str, Any]] = {
    "eeg": {"meg": False, "eeg": True},
    "mag": {"meg": "mag", "eeg": False},
    "grad": {"meg": "grad", "eeg": False},
}


def label_columns(
    vertno: Sequence[np.ndarray], label_vertices: np.ndarray, hemi: str
) -> np.ndarray:
    """Return the gain-matrix columns of the sources that lie inside a label.

    Parameters
    ----------
    vertno : (left, right)
        Vertex numbers of the source space of each hemisphere; the gain matrix holds the left
        sources first.
    label_vertices : ndarray
        Vertex numbers of the label (any vertices that are not sources are ignored).
    hemi : {"lh", "rh"}
    """
    if hemi not in ("lh", "rh"):
        raise ValueError(f"hemi must be 'lh' or 'rh', got {hemi!r}.")
    own = np.asarray(vertno[0 if hemi == "lh" else 1])
    offset = 0 if hemi == "lh" else len(vertno[0])
    inside = np.intersect1d(own, np.asarray(label_vertices))
    if inside.size == 0:
        raise ValueError(f"The label has no source in the {hemi} source space.")
    return offset + np.searchsorted(own, inside)


def average_reference(gain: np.ndarray) -> np.ndarray:
    """Return ``gain`` re-referenced to the average over the channels (columns sum to zero)."""
    gain = np.asarray(gain, dtype=float)
    return gain - gain.mean(axis=0, keepdims=True)


def aggregate_rois(gain: np.ndarray, columns: Sequence[np.ndarray]) -> np.ndarray:
    """Return the topography of each region: the mean of the gains of its sources.

    Returns an array of shape ``(n_channels, n_rois)``.
    """
    gain = np.asarray(gain, dtype=float)
    out = np.empty((gain.shape[0], len(columns)))
    for k, cols in enumerate(columns):
        cols = np.asarray(cols, dtype=int)
        if cols.size == 0:
            raise ValueError(f"Region {k} has no source.")
        if cols.min() < 0 or cols.max() >= gain.shape[1]:
            raise ValueError(f"Region {k} has columns outside the gain matrix.")
        out[:, k] = gain[:, cols].mean(axis=1)
    return out


def relative_visibility(roi_leadfield: np.ndarray) -> np.ndarray:
    """Return the norm of each region's topography relative to the median region."""
    norms = np.linalg.norm(np.asarray(roi_leadfield, dtype=float), axis=0)
    if (norms == 0).any():
        bad = np.flatnonzero(norms == 0).tolist()
        raise ValueError(f"Regions {bad} are invisible on the sensors (zero topography).")
    return norms / np.median(norms)


@dataclass(frozen=True, eq=False)
class HeadModel:
    """Sensors, regions and the leadfield of one modality.

    Attributes
    ----------
    modality : {"eeg", "mag", "grad"}
        Magnetometers and gradiometers are separate modalities and are never mixed.
    channel_names : tuple of str
        Good channels only.
    roi_names : tuple of str
        DK labels in MNE naming, for example ``"bankssts-lh"``.
    roi_leadfield : ndarray, shape (n_channels, n_rois)
        Topography of each region, divided by ``scale``.
    scale : float
        The raw median region norm that was divided out.
    visibility : ndarray, shape (n_rois,)
        Norm of each region's topography relative to the median region.
    gain : ndarray, shape (n_channels, n_sources)
        Gain of every source (fixed orientation), divided by ``scale``.
    source_roi : ndarray of int, shape (n_sources,)
        Index of the region that contains each source, or ``-1``.
    source_flip : ndarray of int, shape (n_sources,)
        Sign of each source relative to the mean orientation of its region (MNE's
        ``label_sign_flip``), used by ROI extraction; 0 for a source outside every region.
    roi_centre : ndarray of int, shape (n_rois,)
        For every region the source nearest to the centroid of its sources, for a focal source.
    forward : mne.Forward
        The fixed-orientation forward solution restricted to these channels, for the inverse step.
    info : mne.Info
        Description of exactly these channels for the inverse step: no bad channels, no SSP, and for
        EEG the average-reference projector. ``None`` for a made-up head model.
    provenance : dict
        Where the model comes from.
    """

    modality: str
    channel_names: tuple[str, ...]
    roi_names: tuple[str, ...]
    roi_leadfield: np.ndarray
    scale: float
    visibility: np.ndarray
    gain: np.ndarray
    source_roi: np.ndarray
    source_flip: np.ndarray
    roi_centre: np.ndarray
    forward: Any
    info: Any
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate the shapes and make the arrays read-only."""
        if self.modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {self.modality!r}.")
        roi_leadfield = np.asarray(self.roi_leadfield, dtype=float)
        gain = np.asarray(self.gain, dtype=float)
        visibility = np.asarray(self.visibility, dtype=float)
        source_roi = np.asarray(self.source_roi, dtype=int)
        source_flip = np.asarray(self.source_flip, dtype=int)
        roi_centre = np.asarray(self.roi_centre, dtype=int)
        n_channels, n_rois = len(self.channel_names), len(self.roi_names)
        if roi_leadfield.shape != (n_channels, n_rois):
            raise ValueError(
                f"roi_leadfield has shape {roi_leadfield.shape}, expected {(n_channels, n_rois)}."
            )
        if gain.ndim != 2 or gain.shape[0] != n_channels:
            raise ValueError(f"gain must have {n_channels} rows, got shape {gain.shape}.")
        if visibility.shape != (n_rois,):
            raise ValueError(f"visibility must have shape {(n_rois,)}, got {visibility.shape}.")
        if source_roi.shape != (gain.shape[1],):
            raise ValueError(
                f"source_roi must have one entry per source ({gain.shape[1]}), "
                f"got shape {source_roi.shape}."
            )
        if source_roi.size and (source_roi.min() < -1 or source_roi.max() >= n_rois):
            raise ValueError("source_roi entries must be -1 or a valid region index.")
        if source_flip.shape != source_roi.shape:
            raise ValueError(
                f"source_flip must have one entry per source ({source_roi.size}), "
                f"got shape {source_flip.shape}."
            )
        if roi_centre.shape != (n_rois,):
            raise ValueError(
                f"roi_centre must have one entry per region ({n_rois}), "
                f"got shape {roi_centre.shape}."
            )
        in_range = (roi_centre >= 0) & (roi_centre < source_roi.size)
        if (
            not in_range.all()
            or (source_roi[roi_centre[in_range]] != np.flatnonzero(in_range)).any()
        ):
            raise ValueError("Every entry of roi_centre must be a source of its own region.")
        outside = source_roi == -1
        if not np.isin(source_flip, (-1, 0, 1)).all() or ((source_flip == 0) != outside).any():
            raise ValueError(
                "source_flip must be -1 or 1 for the sources inside a region and 0 outside."
            )
        for name, array in (
            ("roi_leadfield", roi_leadfield),
            ("gain", gain),
            ("visibility", visibility),
            ("source_roi", source_roi),
            ("source_flip", source_flip),
            ("roi_centre", roi_centre),
        ):
            view = array.view()
            view.flags.writeable = False
            object.__setattr__(self, name, view)

    @property
    def n_channels(self) -> int:
        """Number of good channels."""
        return len(self.channel_names)

    @property
    def n_rois(self) -> int:
        """Number of regions."""
        return len(self.roi_names)

    @property
    def n_sources(self) -> int:
        """Number of sources in the source space."""
        return int(self.gain.shape[1])

    def __repr__(self) -> str:
        """Return a short summary."""
        return (
            f"HeadModel(modality={self.modality!r}, n_channels={self.n_channels}, "
            f"n_rois={self.n_rois}, n_sources={self.n_sources})"
        )


def load_head_model(modality: str, *, path: str | Path | None = None) -> HeadModel:
    """Load the head model of the MNE ``sample`` subject for one sensor type.

    The dataset is located with :func:`dirconn.datasets.require_sample`, so nothing is downloaded
    and a missing dataset stops with instructions. The result is cached per modality and dataset.

    Parameters
    ----------
    modality : {"eeg", "mag", "grad"}
        EEG, MEG magnetometers or MEG planar gradiometers.
    path : path-like, optional
        Passed to :func:`~dirconn.datasets.require_sample`.

    Returns
    -------
    HeadModel
    """
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}.")
    root = require_sample(path=path)
    return _load(modality, str(root))


def _info(root: Path, names: tuple[str, ...], modality: str) -> Any:
    """Return an ``mne.Info`` for exactly ``names``: no bads, no SSP, average reference for EEG."""
    import mne

    full = mne.io.read_info(root / _RAW_FILE, verbose="ERROR")
    info = mne.pick_info(full, [full["ch_names"].index(n) for n in names], verbose="ERROR")
    with info._unlock():
        info["bads"] = []
        info["projs"] = []
    if modality == "eeg":
        holder = mne.io.RawArray(np.zeros((len(names), 2)), info, verbose="ERROR")
        holder.set_eeg_reference("average", projection=True, verbose="ERROR")
        info = holder.info.copy()
    return info


@cache
def _load(modality: str, root_text: str) -> HeadModel:
    import mne

    root = Path(root_text)
    forward_file = root / _FORWARD_FILE
    fwd = mne.read_forward_solution(forward_file, verbose="ERROR")
    fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose="ERROR")

    info = fixed["info"]
    picks = mne.pick_types(info, exclude="bads", **_PICKS[modality])
    names = tuple(info["ch_names"][i] for i in picks)
    if list(fixed["sol"]["row_names"]) != list(info["ch_names"]):
        raise RuntimeError("The forward solution's rows do not follow its channel order.")
    picked = mne.pick_channels_forward(
        fixed, include=list(names), exclude=[], ordered=True, verbose="ERROR"
    )
    raw_gain = np.asarray(picked["sol"]["data"], dtype=float)
    if modality == "eeg":
        # The stored EEG forward is relative to an infinite reference and carries a large
        # common-mode component; recordings are average-referenced, so the head model is too.
        raw_gain = average_reference(raw_gain)

    vertno = [np.asarray(s["vertno"]) for s in picked["src"]]
    labels = [
        lab
        for lab in mne.read_labels_from_annot(
            "sample", parc="aparc", subjects_dir=root / "subjects", verbose="ERROR"
        )
        if not lab.name.startswith("unknown")
    ]
    columns = [label_columns(vertno, lab.vertices, lab.hemi) for lab in labels]
    raw_roi = aggregate_rois(raw_gain, columns)
    scale = float(np.median(np.linalg.norm(raw_roi, axis=0)))
    roi_leadfield = raw_roi / scale

    source_roi = np.full(raw_gain.shape[1], -1, dtype=int)
    source_flip = np.zeros(raw_gain.shape[1], dtype=int)
    for k, cols in enumerate(columns):
        if (source_roi[cols] != -1).any():
            raise RuntimeError("Two labels share a source; the parcellation must be disjoint.")
        source_roi[cols] = k
        source_flip[cols] = mne.label.label_sign_flip(labels[k], picked["src"])

    positions = np.concatenate(
        [np.asarray(s["rr"])[np.asarray(s["vertno"])] for s in picked["src"]]
    )
    roi_centre = np.array(
        [
            cols[np.argmin(np.linalg.norm(positions[cols] - positions[cols].mean(axis=0), axis=1))]
            for cols in columns
        ]
    )
    return HeadModel(
        modality=modality,
        channel_names=names,
        roi_names=tuple(lab.name for lab in labels),
        roi_leadfield=roi_leadfield,
        scale=scale,
        visibility=relative_visibility(roi_leadfield),
        gain=raw_gain / scale,
        source_roi=source_roi,
        source_flip=source_flip,
        roi_centre=roi_centre,
        forward=picked,
        info=_info(root, names, modality),
        provenance={
            "forward_file": str(forward_file),
            "subject": "sample",
            "parcellation": "aparc (Desikan-Killiany)",
            "orientation": "fixed, along the surface normal",
            "reference": "average" if modality == "eeg" else "none",
            "roi_topography": "mean of the gains of the label's source-space vertices",
            "scale": scale,
            "mne": mne.__version__,
        },
    )

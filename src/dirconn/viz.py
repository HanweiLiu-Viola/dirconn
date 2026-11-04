"""Figures that need more than a result: a grouping of the nodes, or the cortical surface.

``Connectivity.plot`` draws what the result alone contains (matrices, spectra, panels, small
graphs).
The functions here need something else and are therefore separate:

* :func:`chord_diagram` draws the connectivity *between groups of nodes* (for example the
  resting-state networks the regions belong to) with pyCirclize (extra ``plot``);
* :func:`degree_imbalance_map` colours the cortical surface by how much more a region sends than it
  receives, with MNE's 3D plotting (needs a 3D backend and the ``sample`` dataset's ``fsaverage``);
* :func:`yeo_networks` and :func:`yeo_groups` assign the Desikan-Killiany regions to the seven
  resting-state networks of Yeo et al. (2011) by the overlap of their vertices on ``fsaverage``.

Source/sink figures say something that is easy to over-read, so they carry a fixed caption
(:data:`GUARD_CAPTION`) that cannot be switched off. Heavy packages are imported when a figure is
drawn, never when :mod:`dirconn` is imported.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from dirconn.plotting import _select
from dirconn.result import Connectivity

#: Printed on every figure that shows sources and sinks.
GUARD_CAPTION = "Predictive, not causal. No leakage correction."

#: The seven networks of Yeo et al. (2011), as numbered in the FreeSurfer annotation.
YEO_NETWORKS: dict[str, str] = {
    "7Networks_1": "Visual",
    "7Networks_2": "Somatomotor",
    "7Networks_3": "Dorsal Attention",
    "7Networks_4": "Ventral Attention",
    "7Networks_5": "Limbic",
    "7Networks_6": "Frontoparietal",
    "7Networks_7": "Default",
}

#: A region whose largest network holds less than this share of its vertices is flagged.
MIN_DOMINANT_SHARE = 0.5


def dominant_networks(regions: Sequence[Any], networks: Sequence[Any]) -> pd.DataFrame:
    """Assign every region to the network that holds most of its vertices.

    Parameters
    ----------
    regions : sequence of mne.Label
        The regions (for example the Desikan-Killiany parcellation of ``fsaverage``).
    networks : sequence of mne.Label
        The networks, named ``7Networks_<k>-<hemi>``, and anything else (the medial wall), which
        never wins but counts against the share.

    Returns
    -------
    pandas.DataFrame
        Indexed by region name, with ``network`` (the name from :data:`YEO_NETWORKS`, or None when
        the region touches no network), ``share`` (that network's vertices over all the region's)
        and ``dominant`` (``share >= MIN_DOMINANT_SHARE``). Only networks of the region's own
        hemisphere are compared, and a tie goes to the first network.
    """
    rows = {}
    for region in regions:
        best, best_overlap = None, 0
        for network in networks:
            key = network.name.rsplit("-", 1)[0]
            if network.hemi != region.hemi or key not in YEO_NETWORKS:
                continue
            overlap = len(np.intersect1d(region.vertices, network.vertices))
            if overlap > best_overlap:
                best, best_overlap = YEO_NETWORKS[key], overlap
        share = best_overlap / len(region.vertices) if len(region.vertices) else 0.0
        rows[region.name] = {
            "network": best,
            "share": share,
            "dominant": best is not None and share >= MIN_DOMINANT_SHARE,
        }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["network", "share", "dominant"])


def _subjects_dir(subjects_dir: str | Path | None) -> Path:
    if subjects_dir is not None:
        return Path(subjects_dir)
    from dirconn.datasets import require_sample

    return require_sample() / "subjects"


@functools.lru_cache(maxsize=4)
def _yeo_table(subjects_dir: str, subject: str) -> pd.DataFrame:
    import mne

    label_dir = Path(subjects_dir) / subject / "label"
    needed = [
        label_dir / f"{hemi}.{parc}.annot"
        for hemi in ("lh", "rh")
        for parc in ("aparc", "Yeo2011_7Networks_N1000")
    ]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The annotation files of the Desikan-Killiany parcellation and of the Yeo networks "
            f"were not found for {subject!r}: {', '.join(missing)}. They come with the "
            "`fsaverage` subject of the MNE sample dataset."
        )
    regions = [
        label
        for label in mne.read_labels_from_annot(
            subject, parc="aparc", subjects_dir=subjects_dir, verbose="ERROR"
        )
        if not label.name.startswith("unknown")
    ]
    networks = mne.read_labels_from_annot(
        subject, parc="Yeo2011_7Networks_N1000", subjects_dir=subjects_dir, verbose="ERROR"
    )
    return dominant_networks(regions, networks)


def yeo_networks(
    *, subjects_dir: str | Path | None = None, subject: str = "fsaverage"
) -> pd.DataFrame:
    """Return the network of every Desikan-Killiany region, computed from the vertex overlap.

    Each region gets the Yeo network that holds most of its vertices on ``subject``; the table also
    gives the share and flags the regions whose largest network holds less than half of them (they
    straddle networks, and the label is then a coarse summary). Nothing is stored in the repository:
    both annotations are read from the ``fsaverage`` of the MNE sample dataset.

    Parameters
    ----------
    subjects_dir : path, optional
        The FreeSurfer subjects directory; the default is ``subjects`` of the MNE sample dataset
        (:func:`dirconn.datasets.require_sample`, which never downloads silently).
    subject : str
        The subject carrying both annotations.

    Returns
    -------
    pandas.DataFrame
        See :func:`dominant_networks`; indexed by region name such as ``"bankssts-lh"``.
    """
    return _yeo_table(str(_subjects_dir(subjects_dir)), subject).copy()


def yeo_groups(names: Sequence[str], *, subjects_dir: str | Path | None = None) -> dict[str, str]:
    """Return ``{region: network}`` for region names such as ``"precuneus-lh"``.

    The result is the ``groups`` argument of :func:`chord_diagram` and :func:`group_matrix`. A name
    that is not a region of the parcellation raises instead of being skipped.
    """
    table = yeo_networks(subjects_dir=subjects_dir)
    unknown = [name for name in names if name not in table.index]
    if unknown:
        raise ValueError(f"Not regions of the Desikan-Killiany parcellation: {', '.join(unknown)}.")
    without = [name for name in names if table.loc[name, "network"] is None]
    if without:
        raise ValueError(f"Regions without a network: {', '.join(without)}.")
    return {name: str(table.loc[name, "network"]) for name in names}


def group_matrix(
    conn: Connectivity,
    groups: Mapping[str, str],
    *,
    how: str = "mean",
    freq: float | None = None,
    band: str | Sequence[float] | None = None,
    time: float | None = None,
) -> pd.DataFrame:
    """Summarise a result between groups of nodes.

    Entry ``[g, h]`` is the mean (or sum) of ``values[i, j]`` over the nodes ``i`` of group ``g``
    and ``j`` of group ``h``, ``i != j``, ignoring undefined cells; it is NaN when there is no such
    pair (a group of one node has no value within itself). Rows are sources and columns are
    targets. The groups appear in the order of their first node. ``freq``, ``band`` and ``time``
    choose the slice as in :meth:`Connectivity.plot`.
    """
    if how not in ("mean", "sum"):
        raise ValueError(f"how must be 'mean' or 'sum', got {how!r}.")
    if conn.source != conn.target:
        raise ValueError(
            "A group summary needs the same nodes as sources and targets, in the same order."
        )
    if conn.is_complex:
        raise ValueError("Complex values cannot be summarised by group; take abs() or real first.")
    missing = [name for name in conn.source if name not in groups]
    if missing:
        raise ValueError(f"Nodes without a group: {', '.join(missing)}.")
    matrix, _ = _select(conn, freq, band, time)
    matrix = matrix.copy()
    np.fill_diagonal(matrix, np.nan)
    names = list(dict.fromkeys(groups[name] for name in conn.source))
    members = {g: [i for i, name in enumerate(conn.source) if groups[name] == g] for g in names}
    table = np.full((len(names), len(names)), np.nan)
    for a, g in enumerate(names):
        for b, h in enumerate(names):
            cells = matrix[np.ix_(members[g], members[h])]
            cells = cells[np.isfinite(cells)]
            if cells.size:
                table[a, b] = cells.sum() if how == "sum" else cells.mean()
    return pd.DataFrame(table, index=names, columns=names)


def chord_diagram(
    conn: Connectivity,
    groups: Mapping[str, str],
    *,
    measure: str = "values",
    how: str = "mean",
    freq: float | None = None,
    band: str | Sequence[float] | None = None,
    time: float | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    cmap: Any = "tab10",
    ax: Any = None,
) -> Any:
    """Draw the connectivity between groups of nodes as a directed chord diagram (pyCirclize).

    The values are summarised by :func:`group_matrix` (within-group entries are left out) and drawn
    with directed links, a ribbon from the source group to the target group as wide as the value.

    Parameters
    ----------
    conn, groups, how, freq, band, time
        As in :func:`group_matrix`.
    measure : {"values", "net_direction"}
        ``"values"`` draws the summarised values (they must be non-negative); ``"net_direction"``
        draws, for every pair of groups, only the stronger direction, with the width of the
        difference. That figure shows sources and sinks and carries :data:`GUARD_CAPTION`.
    figsize : tuple of float
        Size of the figure (ignored with ``ax``).
    cmap : str or dict
        Colours of the groups, as in pyCirclize.
    ax : matplotlib polar axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if measure not in ("values", "net_direction"):
        raise ValueError(f"measure must be 'values' or 'net_direction', got {measure!r}.")
    try:
        from pycirclize import Circos
    except ImportError as exc:
        raise ImportError(
            "The chord diagram needs pyCirclize. Install it with: pip install 'dirconn[plot]'."
        ) from exc
    table = group_matrix(conn, groups, how=how, freq=freq, band=band, time=time)
    values = np.nan_to_num(table.to_numpy(dtype=float, copy=True))
    np.fill_diagonal(values, 0.0)
    if measure == "values":
        if (values < 0).any():
            raise ValueError(
                "The values measure needs non-negative values; for a signed result such as a net "
                "direction use measure='net_direction' on the original values."
            )
    else:
        values = np.clip(values - values.T, 0.0, None)
    if not values.any():
        raise ValueError("The group summary is zero everywhere, so there is nothing to draw.")
    matrix = pd.DataFrame(values, index=table.index, columns=table.columns)
    circos = Circos.chord_diagram(
        matrix,
        space=5,
        cmap=cmap,
        label_kws={"size": 10},
        link_kws={"direction": 1, "ec": "black", "lw": 0.5},
    )
    fig = circos.plotfig(figsize=figsize, ax=ax)
    if ax is None:
        # leave room above and below the ring for the title and the caption
        fig.axes[0].set_position((0.1, 0.1, 0.8, 0.8))
    fig.text(
        0.5,
        0.97,
        f"{conn.provenance.method}: {measure.replace('_', ' ')} between groups ({how})",
        ha="center",
        va="top",
        fontsize=10,
    )
    if measure == "net_direction":
        fig.text(0.5, 0.02, GUARD_CAPTION, ha="center", va="bottom", fontsize=8)
    return fig


def region_values(values: Any) -> dict[str, float]:
    """Return ``{region: value}`` from a mapping, a Series or a result of ``degree_imbalance``.

    Region names must end in ``-lh`` or ``-rh``. Undefined (NaN) values are left out, so those
    regions stay uncoloured instead of being drawn as zero. A labelled array must have one value
    per node, that is a single frequency and a single time.
    """
    if isinstance(values, xr.DataArray):
        if values.dims != ("node", "freq", "time") or values.shape[1:] != (1, 1):
            raise ValueError(
                "An array of values needs the dimensions (node, freq, time) with one frequency "
                f"and one time; got dimensions {values.dims} and shape {values.shape}."
            )
        items = dict(
            zip(values.coords["node"].values.tolist(), values.values[:, 0, 0], strict=True)
        )
    elif isinstance(values, pd.Series):
        items = dict(values.items())
    else:
        items = dict(values)
    bad = [name for name in items if not str(name).endswith(("-lh", "-rh")) or len(str(name)) < 4]
    if bad:
        raise ValueError(
            "Region names must end in -lh or -rh (like 'precuneus-lh'); "
            f"got {', '.join(map(str, bad))}."
        )
    defined = {str(name): float(value) for name, value in items.items() if np.isfinite(value)}
    if not defined:
        raise ValueError("There is no defined value to draw.")
    return defined


def degree_imbalance_map(
    values: Any,
    *,
    subjects_dir: str | Path | None = None,
    hemi: str = "split",
    views: Sequence[str] = ("lateral", "medial"),
    vmax: float | None = None,
    degrees: str = "all edges, raw values",
    backend: str = "auto",
    size: tuple[int, int] = (1200, 750),
) -> Any:
    """Colour the cortical surface by ``log(D_out) - log(D_in)`` of every region (MNE 3D plot).

    Red means a region sends more than it receives (a source), blue the reverse; the scale is
    symmetric around zero. Regions without a value stay uncoloured. The figure carries
    :data:`GUARD_CAPTION`: the values describe predictive influence and the regions' signals were
    not corrected for leakage between neighbouring regions.

    Parameters
    ----------
    values : mapping, Series or DataArray
        A value per region, for example ``conn.degree_imbalance()`` of a result whose nodes are
        Desikan-Killiany regions (names like ``"precuneus-lh"``).
    subjects_dir : path, optional
        The FreeSurfer subjects directory with ``fsaverage``; the default is ``subjects`` of the MNE
        sample dataset.
    hemi, views, size
        As in ``SourceEstimate.plot``.
    vmax : float, optional
        Largest absolute value of the colour scale; default the largest absolute value drawn.
    degrees : str
        Says which edges the degrees were summed over; it is written into the title, so that the
        figure states the choice made in ``degree_imbalance`` (a mask or none).
    backend : str
        MNE's 3D backend (``"auto"`` follows MNE's configuration; it never changes it).

    Returns
    -------
    mne.viz.Brain
        Close it with ``brain.close()``; ``brain.screenshot()`` returns the image as an array.
    """
    import mne

    drawn = region_values(values)
    limit = float(vmax) if vmax is not None else max(abs(v) for v in drawn.values())
    if not limit > 0:
        raise ValueError(
            "All values are zero (or vmax is not positive), so there is nothing to draw."
        )
    root = _subjects_dir(subjects_dir)
    labels = mne.read_labels_from_annot(
        "fsaverage", parc="aparc", subjects_dir=str(root), verbose="ERROR"
    )
    by_name = {label.name: label for label in labels}
    unknown = [name for name in drawn if name not in by_name]
    if unknown:
        raise ValueError(f"Not regions of the Desikan-Killiany parcellation: {', '.join(unknown)}.")
    chosen = list(drawn)
    stc = mne.labels_to_stc(
        [by_name[name] for name in chosen],
        [drawn[name] for name in chosen],
        subject="fsaverage",
        verbose="ERROR",
    )
    brain = stc.plot(
        subject="fsaverage",
        subjects_dir=str(root),
        hemi=hemi,
        views=list(views),
        backend=backend,
        time_viewer=False,
        show_traces=False,
        colormap="RdBu_r",
        clim={"kind": "value", "pos_lims": [0.0, limit / 2, limit]},
        transparent=False,
        background="white",
        foreground="black",
        size=size,
        time_label=None,
        verbose="ERROR",
    )
    brain.add_text(
        0.02,
        0.93,
        f"log(D_out) - log(D_in) ({degrees})",
        "title",
        color="black",
        font_size=9,
    )
    brain.add_text(
        0.02, 0.88, "red = more source, blue = more sink", "legend", color="black", font_size=9
    )
    brain.add_text(0.02, 0.02, GUARD_CAPTION, "caption", color="black", font_size=9)
    return brain

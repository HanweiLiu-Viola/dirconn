"""Build the reportlets of a participant report from the outputs of the pipeline.

:func:`build_reportlets` collects one subject's derivatives, builds a reportlet for every figure
and table of the participant report (sections summary, input, diagnostics, connectivity, controls,
statistics and errors) and writes them next to the figures of each unit. Nothing is recomputed: the
reportlets only read the NetCDF results, JSON sidecars, edge tables, diagnostics and run records
that the pipeline wrote.

Every reportlet is built inside a guard. When its construction raises, a ``fail`` placeholder with
the same name takes its place and the error is listed in the ``summary/errors`` table, so one
broken figure never stops the report.

Subject-level reportlets (``summary/overview``, ``summary/errors``) are written with the label
``sub-<subject>`` into ``<root>/sub-<subject>/figures``; unit reportlets use the unit label and the
unit's own ``figures`` directory.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from dirconn.data.result import Connectivity
from dirconn.methods.registry import get_estimator
from dirconn.reports.collect import (
    CONTROL_DESC,
    MethodOutput,
    SubjectOutputs,
    UnitOutputs,
    collect_subject,
    selected_order,
)
from dirconn.reports.figures import (
    band_matrix_svg,
    directed_spectra_svg,
    net_direction_svg,
    order_curve_svg,
    psd_svg,
    pvalue_hist_svg,
    significant_over_time_svg,
    time_frequency_svg,
)
from dirconn.reports.reportlet import Reportlet, Status, worst_status, write_reportlet

__all__ = [
    "MAX_PAIRS",
    "MAX_SIGNIFICANT_ROWS",
    "SECTIONS",
    "build_reportlets",
    "statistics_table",
    "subject_reportlets",
    "unit_reportlets",
]

#: Section ids of the participant report, in display order.
SECTIONS = (
    "summary",
    "input",
    "diagnostics",
    "connectivity",
    "controls",
    "statistics",
    "methods",
    "errors",
)

#: Most node pairs drawn in one ``spectra`` or ``tf`` figure.
MAX_PAIRS = 6

#: Most rows of a ``significant`` table.
MAX_SIGNIFICANT_ROWS = 200

_DASH = "—"
_NOT_RECORDED = "not recorded"
_MIN_SPECTRUM_FREQS = 3
_ASSUMPTION_STATUS: dict[str, Status] = {
    "pass": "pass",
    "warn": "warn",
    "fail": "fail",
    "not_checked": "info",
}


@dataclass(frozen=True)
class _Body:
    """What a reportlet builder returns; the sink adds the name, section, desc and title."""

    caption: str
    status: Status = "info"
    svg: str | None = None
    table: list[dict[str, Any]] | None = None
    data: dict[str, Any] | None = None


Maker = Callable[[], _Body | None]


@dataclass
class _Sink:
    """Collects the reportlets of one unit (or subject) and guards every construction."""

    label: str
    items: list[Reportlet] = field(default_factory=list)

    def add(self, desc: str, name: str, section: str, title: str, make: Maker) -> None:
        """Run ``make``; add its reportlet, nothing when it returns ``None``, or a placeholder."""
        try:
            body = make()
            if body is None:
                return
            reportlet = Reportlet(
                name=name,
                section=section,
                desc=desc,
                title=title,
                caption=body.caption,
                status=body.status,
                svg=body.svg,
                table=body.table,
                data=body.data,
            )
        except Exception as exc:
            reason = _one_line(exc)
            reportlet = Reportlet(
                name=name,
                section=section,
                desc=desc,
                title=title,
                caption=f"could not be built: {reason}",
                status="fail",
                data={"build_error": reason, "unit": self.label},
            )
        self.items.append(reportlet)


def _one_line(exc: Exception) -> str:
    text = str(exc).strip()
    first = text.splitlines()[0] if text else ""
    return f"{type(exc).__name__}: {first}" if first else type(exc).__name__


# ---------------------------------------------------------------------------------------------
# small helpers


def _is_true(value: Any) -> bool:
    """Interpret a ``significant`` cell: bool, NaN or the strings "True", "False", "n/a"."""
    if isinstance(value, str):
        return value.strip().lower() == "true"
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
        return bool(value)
    except (TypeError, ValueError):
        return False


def _plain(value: Any) -> Any:
    """Return ``value`` as plain JSON data: numpy scalars to Python, NaN and inf to ``None``."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _records(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    present = [c for c in columns if c in frame.columns]
    return [_plain(row) for row in frame[present].to_dict("records")]


def _cfg(config: Mapping[str, Any] | None, *keys: str) -> Any:
    """Return ``config[k1][k2]...`` or ``None`` when a key is missing."""
    node: Any = config
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def _slug(text: str) -> str:
    """Make ``text`` usable as a reportlet name part (no ``_`` or ``/``)."""
    return text.replace("_", "-").replace("/", "-")


def _report_kind(desc: str) -> tuple[str, bool]:
    """Return ``(report_kind, known)``; an unknown method is treated as a matrix."""
    try:
        return get_estimator(desc).info.report_kind, True
    except KeyError:
        return "matrix", False


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass
class _Context:
    """One unit with a cache of its NetCDF results."""

    subject: SubjectOutputs
    unit: UnitOutputs
    _conns: dict[str, Connectivity] = field(default_factory=dict)

    @property
    def config(self) -> Mapping[str, Any] | None:
        return self.subject.config

    @property
    def outputs(self) -> list[MethodOutput]:
        """Method results (sorted by name) followed by the control."""
        listed = list(self.unit.methods.values())
        return [*listed, self.unit.control] if self.unit.control is not None else listed

    def conn(self, out: MethodOutput) -> Connectivity:
        if out.desc not in self._conns:
            self._conns[out.desc] = out.connectivity()
        return self._conns[out.desc]

    def nodes(self) -> tuple[list[str], list[str]]:
        """Node names and kinds from the first sidecar (kinds are ``unknown`` when missing)."""
        for out in self.outputs:
            names = out.sidecar.get("NodeLabels")
            if isinstance(names, list):
                kinds = out.sidecar.get("NodeKinds")
                if not (isinstance(kinds, list) and len(kinds) == len(names)):
                    kinds = ["unknown"] * len(names)
                return [str(n) for n in names], [str(k) for k in kinds]
        return [], []

    def bands(self) -> dict[str, list[float]]:
        """Band limits in Hz of the first sidecar that has any."""
        for out in self.outputs:
            raw = out.sidecar.get("Bands")
            if isinstance(raw, Mapping):
                limits = {str(k): list(v) for k, v in raw.items() if v}
                if limits:
                    return limits
        return {}


# ---------------------------------------------------------------------------------------------
# input


def _nodes_body(ctx: _Context) -> _Body | None:
    names, kinds = ctx.nodes()
    if not names:
        return None
    table = [{"name": n, "kind": k} for n, k in zip(names, kinds, strict=True)]
    counts: dict[str, int] = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    summary = ", ".join(f"{count} {kind}" for kind, count in counts.items())
    return _Body(f"{len(names)} nodes ({summary}).", table=table)


def _record_body(ctx: _Context) -> _Body | None:
    diag = ctx.unit.diagnostics
    if diag is None:
        return None
    fraction = diag.get("valid_fraction")
    std = diag.get("channel_std")
    table = (
        [{"channel": str(k), "std": _plain(v)} for k, v in std.items()]
        if isinstance(std, Mapping)
        else []
    )
    history: list[str] = []
    for out in ctx.outputs:
        steps = ctx.conn(out).provenance.preprocessing.get("data_history")
        if isinstance(steps, list):
            history = [str(s.get("step", "?")) if isinstance(s, Mapping) else str(s) for s in steps]
            break
    data = {
        "sfreq": _plain(diag.get("sfreq")),
        "n_trials": _plain(diag.get("n_trials")),
        "n_times": _plain(diag.get("n_times")),
        "valid_fraction": _plain(fraction),
        "data_history": history,
    }
    parts = [
        f"sampling frequency {data['sfreq']} Hz",
        f"{data['n_trials']} trial(s) of {data['n_times']} samples",
        (
            f"{100 * float(fraction):.0f}% of the samples are valid"
            if isinstance(fraction, (int, float))
            else "valid fraction not recorded"
        ),
        "preparation steps: " + (", ".join(history) if history else "none recorded"),
    ]
    low = isinstance(fraction, (int, float)) and fraction < 0.5
    status: Status = "warn" if low else "pass"
    suffix = " Fewer than half of the samples are valid." if low else ""
    return _Body("; ".join(parts).capitalize() + "." + suffix, status, table=table, data=data)


# ---------------------------------------------------------------------------------------------
# diagnostics


def _assumptions_body(ctx: _Context) -> _Body | None:
    rows: list[dict[str, Any]] = []
    statuses: list[Status] = []

    def add(method: str, name: str, status: str, detail: str) -> None:
        statuses.append(_ASSUMPTION_STATUS.get(status, "info"))
        rows.append(
            {
                "method": method,
                "check": name,
                "status": "not checked" if status == "not_checked" else status,
                "detail": detail,
            }
        )

    for out in ctx.outputs:
        for check in ctx.conn(out).provenance.assumptions.checks:
            add(out.desc, check.name, check.status, check.detail)
    for check in (ctx.unit.diagnostics or {}).get("checks") or []:
        add(
            "record", str(check.get("name")), str(check.get("status")), str(check.get("detail", ""))
        )
    if not rows:
        return None
    worst = worst_status(statuses)
    counts = {s: statuses.count(s) for s in ("fail", "warn")}
    caption = (
        f"{len(rows)} assumption checks over {len({r['method'] for r in rows})} results: "
        f"{counts['fail']} failed, {counts['warn']} warned. "
        "Diagnostics inform and never block a run."
    )
    return _Body(caption, worst, table=rows)


def _fixed_order(ctx: _Context) -> int | None:
    for out in ctx.outputs:
        order = ctx.conn(out).provenance.model_order
        if order is not None:
            return int(order)
    return None


def _order_body(ctx: _Context) -> _Body | None:
    diag = ctx.unit.diagnostics
    if diag is None:
        return None
    selection = diag.get("order_selection")
    if selection:
        order = selected_order(selection)
        orders = [int(o) for o in selection["orders"]]
        largest = max(orders)
        order_max = _cfg(ctx.config, "estimation", "order_max")
        caption = (
            f"Order chosen by the {selection['criterion']} criterion: {order} "
            f"(candidates {min(orders)} to {largest}"
            + (f"; order_max is {order_max}" if order_max is not None else "")
            + "). "
        )
        at_edge = order == largest
        caption += (
            "The selected order is the largest candidate order, so a higher order may fit better."
            if at_edge
            else "The selected order lies inside the candidate range."
        )
        return _Body(caption, "warn" if at_edge else "pass", svg=order_curve_svg(selection) or None)
    fixed = _fixed_order(ctx)
    if fixed is None:
        return _Body("No method of this record has a VAR order.")
    return _Body(f"Model order fixed at {fixed} by the configuration; no order was selected.")


def _psd_body(ctx: _Context) -> _Body | None:
    diag = ctx.unit.diagnostics
    spectral = None if diag is None else diag.get("spectral")
    if not spectral:
        return None
    assert diag is not None
    checks = [c for c in diag.get("checks") or [] if str(c.get("name", "")).startswith("spectral")]
    status = worst_status(_ASSUMPTION_STATUS.get(str(c.get("status")), "info") for c in checks)
    detail = " ".join(str(c.get("detail", "")) for c in checks)
    caption = (
        f"{detail} The dynamic-range threshold is uncalibrated, so read the status as a hint."
        if detail
        else "Power spectral density of the prepared channels; the dynamic-range threshold is "
        "uncalibrated."
    )
    return _Body(caption, status, svg=psd_svg(spectral, ctx.bands()) or None)


# ---------------------------------------------------------------------------------------------
# connectivity


def _band_names(edges: pd.DataFrame, sidecar: Mapping[str, Any]) -> list[str]:
    seen = list(dict.fromkeys(edges["band"].astype(str)))
    declared = list(sidecar.get("Bands") or {})
    return sorted(seen, key=lambda b: declared.index(b) if b in declared else len(declared))


def _choose_pairs(edges: pd.DataFrame | None, nodes: Sequence[str]) -> list[tuple[str, str]]:
    """Pick at most ``MAX_PAIRS`` ordered pairs: significant first, then by largest value."""
    ranked: dict[tuple[str, str], tuple[bool, float]] = {}
    if edges is not None and {"source", "target", "value"} <= set(edges.columns):
        flags = edges["significant"] if "significant" in edges.columns else [False] * len(edges)
        values = pd.to_numeric(edges["value"], errors="coerce")
        for src, tgt, value, flag in zip(
            edges["source"], edges["target"], values, flags, strict=True
        ):
            key = (str(src), str(tgt))
            if key[0] == key[1] or not math.isfinite(value):
                continue
            sig, top = ranked.get(key, (False, -math.inf))
            ranked[key] = (sig or _is_true(flag), max(top, float(value)))
    ordered = sorted(ranked, key=lambda k: (not ranked[k][0], -ranked[k][1], k[0], k[1]))
    pairs = ordered[:MAX_PAIRS]
    for src in nodes:
        for tgt in nodes:
            if len(pairs) >= MAX_PAIRS:
                return pairs
            if src != tgt and (src, tgt) not in ranked:
                pairs.append((src, tgt))
    return pairs


def _matrix_body(
    out: MethodOutput, edges: pd.DataFrame, band: str, nodes: Sequence[str], *, known: bool
) -> _Body:
    svg = band_matrix_svg(edges, band, nodes, title=f"{out.desc}, {band}")
    caption = (
        f"{out.desc} value of the {band} band; rows are sources and columns are targets. "
        "An asterisk marks an edge that is significant after correction."
    )
    if not known:
        caption += " The method is not registered, so it is shown as a matrix."
    return _Body(caption, svg=svg or None)


def _net_body(
    out: MethodOutput, edges: pd.DataFrame, band: str, kinds: Mapping[str, str]
) -> _Body | None:
    svg = net_direction_svg(edges, kinds, band)
    if not svg:
        return None
    caption = (
        f"Net direction of {out.desc} in the {band} band for pairs of nodes of different kinds: "
        "value(i to j) minus value(j to i)."
    )
    return _Body(caption, svg=svg)


def _spectra_body(
    ctx: _Context, out: MethodOutput, edges: pd.DataFrame | None, nodes: Sequence[str]
) -> _Body | None:
    conn = ctx.conn(out)
    if conn.freqs.shape[0] < _MIN_SPECTRUM_FREQS:
        return None
    pairs = _choose_pairs(edges, nodes)
    caption = (
        f"Both directions of {len(pairs)} node pairs over frequency. Pairs with a significant "
        f"band come first, then those with the largest band value (at most {MAX_PAIRS})."
    )
    return _Body(
        caption,
        svg=directed_spectra_svg(conn, pairs) or None,
        data={"pairs": [list(p) for p in pairs]},
    )


def _tf_body(
    ctx: _Context, out: MethodOutput, edges: pd.DataFrame | None, nodes: Sequence[str]
) -> _Body | None:
    conn = ctx.conn(out)
    if conn.times.shape[0] < 2:
        return None
    pairs = _choose_pairs(edges, nodes)
    caption = (
        f"Time-frequency maps of {out.desc} for {len(pairs)} node pairs "
        f"(at most {MAX_PAIRS}; pairs with large band values come first)."
    )
    return _Body(
        caption,
        svg=time_frequency_svg(conn, pairs) or None,
        data={"pairs": [list(p) for p in pairs]},
    )


def _add_connectivity(ctx: _Context, sink: _Sink, out: MethodOutput) -> None:
    kind, known = _report_kind(out.desc)
    names, kinds = ctx.nodes()
    kind_of = dict(zip(names, kinds, strict=True))
    edges = out.edge_table()
    timed = out.time_resolved
    desc, section = out.desc, "connectivity"
    if kind == "time_frequency":
        sink.add(
            desc,
            "tf",
            section,
            f"{desc.upper()} time-frequency",
            lambda: _tf_body(ctx, out, edges, names),
        )
        return
    if timed or edges is None:
        return
    for band in _band_names(edges, out.sidecar):
        title = f"{desc.upper()} {band}"
        sink.add(
            desc,
            f"matrix-{_slug(band)}",
            section,
            title,
            lambda band=band: _matrix_body(out, edges, band, names, known=known),  # type: ignore[misc]
        )
        if out.sidecar.get("Directed", True):
            sink.add(
                desc,
                f"net-{_slug(band)}",
                section,
                f"{desc.upper()} net direction, {band}",
                lambda band=band: _net_body(out, edges, band, kind_of),  # type: ignore[misc]
            )
    if kind == "spectrum":
        sink.add(
            desc,
            "spectra",
            section,
            f"{desc.upper()} spectra",
            lambda: _spectra_body(ctx, out, edges, names),
        )


# ---------------------------------------------------------------------------------------------
# statistics


def statistics_table(sidecar: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Turn the ``Statistics`` entries of a sidecar into display rows.

    Parameters
    ----------
    sidecar : mapping
        Parsed ``*_conn.json`` sidecar.

    Returns
    -------
    list of dict
        One row per entry with ``band``, ``test``, ``generator`` (the ``source`` sentence for a
        built-in test), ``block`` (block size when the generator has one), ``n_surrogates``,
        ``seed`` and ``p_shape`` (shape joined with ``×``). Missing or null values are ``"—"``.
    """

    def show(value: Any) -> Any:
        return _DASH if value is None else _plain(value)

    rows = []
    for entry in sidecar.get("Statistics") or []:
        generator = entry.get("Generator") or {}
        name = generator.get("name")
        label = generator.get("source") if name == "builtin" else name
        params = generator.get("params") or {}
        shape = entry.get("PValueShape")
        rows.append(
            {
                "band": show(entry.get("Band")),
                "test": show(entry.get("Test")),
                "generator": show(label if label is not None else name),
                "block": show(params.get("block_size")),
                "n_surrogates": show(entry.get("NSurrogates")),
                "seed": show(entry.get("Seed")),
                "p_shape": "×".join(str(int(s)) for s in shape) if shape else _DASH,
            }
        )
    return rows


def _tests_body(ctx: _Context, out: MethodOutput) -> _Body:
    rows = statistics_table(out.sidecar)
    if rows:
        correction = _cfg(ctx.config, "statistics", "correction")
        alpha = _cfg(ctx.config, "statistics", "alpha")
        parts = []
        if correction is not None:
            parts.append(f"p values are corrected with {correction}")
        if alpha is not None:
            parts.append(f"edges are significant at alpha = {alpha}")
        caption = "; ".join(parts).capitalize() + "." if parts else "Tests run for this result."
        return _Body(caption, table=rows)
    if out.statistics_note is not None:
        return _Body(out.statistics_note)
    if _cfg(ctx.config, "statistics", "test") in (None, "none"):
        return _Body("No test configured (statistics.test = none).")
    return _Body("No test is recorded in the sidecar of this result.")


def _significant_body(out: MethodOutput, edges: pd.DataFrame) -> _Body | None:
    if "significant" not in edges.columns:
        return None
    rows = edges.loc[edges["significant"].map(_is_true)].copy()
    total = len(rows)
    rows["_p_adj"] = pd.to_numeric(rows["p_adj"], errors="coerce") if "p_adj" in rows else np.nan
    rows["_value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows = rows.sort_values(["_p_adj", "_value"], ascending=[True, False], na_position="last")
    shown = rows.head(MAX_SIGNIFICANT_ROWS)
    table = _records(shown, ["source", "target", "band", "time", "value", "p", "p_adj"])
    caption = (
        f"{len(shown)} of {total} significant rows shown, ordered by adjusted p value and then "
        "by value."
    )
    return _Body(caption, table=table)


def _pvalues_body(edges: pd.DataFrame) -> _Body | None:
    svg = pvalue_hist_svg(edges)
    if not svg:
        return None
    return _Body("Histogram of the p values of all pairs, bands and time points.", svg=svg)


def _sig_time_body(edges: pd.DataFrame) -> _Body | None:
    svg = significant_over_time_svg(edges)
    if not svg:
        return None
    return _Body("Number of significant pairs at each time point, one line per band.", svg=svg)


def _add_statistics(ctx: _Context, sink: _Sink, out: MethodOutput) -> None:
    desc, section = out.desc, "statistics"
    label = desc.upper()
    sink.add(desc, "tests", section, f"{label} tests", lambda: _tests_body(ctx, out))
    if not out.sidecar.get("Statistics"):
        return
    edges = out.edge_table()
    if edges is None:
        return
    sink.add(
        desc,
        "significant",
        section,
        f"{label} significant edges",
        lambda: _significant_body(out, edges),
    )
    sink.add(desc, "pvalues", section, f"{label} p values", lambda: _pvalues_body(edges))
    if "time" in edges.columns:
        sink.add(
            desc,
            "sig-time",
            section,
            f"{label} significant pairs over time",
            lambda: _sig_time_body(edges),
        )


# ---------------------------------------------------------------------------------------------
# controls


def _trgc_net(ctx: _Context, control: MethodOutput) -> tuple[np.ndarray, list[str]]:
    conn = ctx.conn(control)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        net = np.nanmean(np.asarray(conn.values, dtype=float), axis=(2, 3))
    return net, [str(n) for n in conn.source]


def _trgc_frame(net: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    rows = [
        (names[i], names[j], "broadband", float(net[i, j]))
        for i in range(len(names))
        for j in range(len(names))
        if i != j
    ]
    return pd.DataFrame(rows, columns=["source", "target", "band", "value"])


def _trgc_matrix_body(ctx: _Context, control: MethodOutput) -> _Body:
    net, names = _trgc_net(ctx, control)
    svg = band_matrix_svg(_trgc_frame(net, names), "broadband", names, title="TRGC, broadband")
    caption = (
        "Time-reversed Granger causality, the net flow averaged over frequency; a positive value "
        "means the row node drives the column node."
    )
    return _Body(caption, svg=svg or None)


def _trgc_agreement_body(ctx: _Context, control: MethodOutput) -> _Body | None:
    gc = ctx.unit.methods.get("gc")
    edges = None if gc is None else gc.edge_table()
    if edges is None or "time" in edges.columns:
        return None
    broadband = edges.loc[edges["band"].astype(str) == "broadband"]
    if broadband.empty:
        return None
    gc_value = {
        (str(s), str(t)): float(v)
        for s, t, v in zip(
            broadband["source"], broadband["target"], broadband["value"], strict=True
        )
    }
    net, names = _trgc_net(ctx, control)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = float(net[i, j])
            b = gc_value.get((names[i], names[j]), math.nan) - gc_value.get(
                (names[j], names[i]), math.nan
            )
            agree = (
                bool(np.sign(a) == np.sign(b)) if math.isfinite(a) and math.isfinite(b) else None
            )
            rows.append(
                {
                    "source": names[i],
                    "target": names[j],
                    "trgc_net": _plain(a),
                    "gc_net": _plain(b),
                    "agree": agree,
                }
            )
    checked = [r for r in rows if r["agree"] is not None]
    agreeing = sum(1 for r in checked if r["agree"])
    status: Status = "warn" if agreeing < len(checked) else "pass"
    caption = (
        f"{agreeing} of {len(checked)} node pairs have the same net direction in time-reversed "
        "and ordinary Granger causality; a disagreement can point to instantaneous mixing."
    )
    return _Body(caption, status, table=rows)


def _add_controls(ctx: _Context, sink: _Sink) -> None:
    control = ctx.unit.control
    if control is None:
        return
    sink.add(
        CONTROL_DESC, "matrix", "controls", "TRGC net flow", lambda: _trgc_matrix_body(ctx, control)
    )
    sink.add(
        CONTROL_DESC,
        "agreement",
        "controls",
        "TRGC and GC agreement",
        lambda: _trgc_agreement_body(ctx, control),
    )


# ---------------------------------------------------------------------------------------------
# unit and subject


def _build_unit(subject: SubjectOutputs, unit: UnitOutputs) -> list[Reportlet]:
    ctx = _Context(subject, unit)
    sink = _Sink(unit.label)
    sink.add("prep", "nodes", "input", "Nodes", lambda: _nodes_body(ctx))
    sink.add("prep", "record", "input", "Recording", lambda: _record_body(ctx))
    sink.add(
        "diag", "assumptions", "diagnostics", "Assumption checks", lambda: _assumptions_body(ctx)
    )
    sink.add("diag", "order", "diagnostics", "Model order", lambda: _order_body(ctx))
    sink.add("diag", "psd", "diagnostics", "Spectral range", lambda: _psd_body(ctx))
    for out in unit.methods.values():
        _add_connectivity(ctx, sink, out)
    _add_controls(ctx, sink)
    for out in unit.methods.values():
        _add_statistics(ctx, sink, out)
    return sink.items


def unit_reportlets(subject: SubjectOutputs, unit: UnitOutputs) -> list[Reportlet]:
    """Build the reportlets of one unit (input, diagnostics, connectivity, controls, statistics).

    A reportlet whose construction raises becomes a ``fail`` placeholder with the same name; the
    exception is named in its caption and in the ``summary/errors`` table.

    Parameters
    ----------
    subject : SubjectOutputs
        The subject the unit belongs to (supplies configuration, runs and failures).
    unit : UnitOutputs
        The unit.

    Returns
    -------
    list of Reportlet
        Reportlets in section order; results that are absent give no reportlet (see
        :func:`subject_reportlets` for the reasons).
    """
    return _build_unit(subject, unit)


def _missing_rows(subject: SubjectOutputs, unit: UnitOutputs) -> list[dict[str, Any]]:
    """Rows ``{kind, label, reportlet, reason, run_dir, crash_file}`` for absent outputs."""
    root = subject.root
    failed = {f.label: f for f in subject.failures}
    config = subject.config

    def row(reportlet: str, reason: str, failure_label: str | None = None) -> dict[str, Any]:
        failure = failed.get(failure_label) if failure_label else None
        crash = (
            ""
            if failure is None or failure.crash_file is None
            else _relative(failure.crash_file, root)
        )
        return {
            "kind": "missing reportlet",
            "label": unit.label,
            "reportlet": reportlet,
            "reason": reason,
            "run_dir": "" if failure is None else _relative(failure.run_dir, root),
            "crash_file": crash,
        }

    rows: list[dict[str, Any]] = []
    expected = _cfg(config, "estimation", "methods")
    for method in expected if isinstance(expected, list) else []:
        if method not in unit.methods:
            key = f"{unit.label}:{method}"
            rows.append(row(f"{method}/*", "task failed" if key in failed else "not run", key))
    if unit.control is None:
        if _cfg(config, "controls", "time_reversal") is False:
            reason, key = "time_reversal is off", ""
        else:
            key = f"{unit.label}:controls"
            reason = "controls task failed" if key in failed else "controls task failed or not run"
        rows.append(row("trgc/matrix", reason, key))
        if "gc" in unit.methods:
            rows.append(row("trgc/agreement", reason, key))
    diagnostics = unit.diagnostics
    if diagnostics is None:
        key = f"{unit.label}:controls"
        reason = "diagnostics missing (controls task failed or not run)"
        rows.extend(row(name, reason, key) for name in ("prep/record", "diag/order", "diag/psd"))
    elif not diagnostics.get("spectral"):
        if _cfg(config, "controls", "spectral_range_check") is False:
            rows.append(row("diag/psd", "spectral_range_check is off"))
        else:
            rows.append(row("diag/psd", "spectral diagnostics not recorded"))
    return rows


def _errors_body(subject: SubjectOutputs, unit_reports: Mapping[str, Sequence[Reportlet]]) -> _Body:
    root = subject.root
    rows: list[dict[str, Any]] = [
        {
            "kind": "task failed",
            "label": f.label,
            "reportlet": "",
            "reason": "task failed",
            "run_dir": _relative(f.run_dir, root),
            "crash_file": "" if f.crash_file is None else _relative(f.crash_file, root),
        }
        for f in subject.failures
    ]
    n_failed = len(rows)
    for unit in subject.units:
        rows.extend(_missing_rows(subject, unit))
    n_build = 0
    for unit in subject.units:
        for r in unit_reports.get(unit.label, ()):
            if r.data and "build_error" in r.data:
                n_build += 1
                rows.append(
                    {
                        "kind": "build failed",
                        "label": unit.label,
                        "reportlet": f"{r.desc}/{r.name}",
                        "reason": f"could not be built: {r.data['build_error']}",
                        "run_dir": "",
                        "crash_file": "",
                    }
                )
    n_missing = len(rows) - n_failed - n_build
    status: Status = "fail" if n_failed or n_build else ("info" if n_missing else "pass")
    if not rows:
        caption = "No task failed and no reportlet is missing."
    else:
        caption = (
            f"{n_failed} failed task(s), {n_build} reportlet(s) that could not be built and "
            f"{n_missing} reportlet(s) that are absent."
        )
    return _Body(caption, status, table=rows, data={"failed": n_failed, "built": n_build})


def _overview_body(
    subject: SubjectOutputs, reportlets: Iterable[Reportlet], errors_status: Status
) -> _Body:
    statuses = [r.status for r in reportlets]
    statuses.append(errors_status)
    rows = []
    for unit in subject.units:
        run = subject.latest_run_for(unit.label)
        provenance = (run.provenance if run is not None else None) or {}
        entry: Mapping[str, Any] = next(
            (i for i in provenance.get("inputs") or [] if i.get("label") == unit.label), {}
        )
        image = provenance.get("image") or {}
        bands = sorted({b for o in unit.methods.values() for b in (o.sidecar.get("Bands") or {})})
        rows.append(
            {
                "unit": unit.label,
                "input_files": [PurePosixPath(str(f)).name for f in entry.get("files") or []],
                "sha256": entry.get("sha256") or _NOT_RECORDED,
                "methods": ", ".join(sorted(unit.methods)),
                "bands": ", ".join(bands),
                "started": provenance.get("started") or _NOT_RECORDED,
                "finished": provenance.get("finished") or _NOT_RECORDED,
                "dirconn": (provenance.get("versions") or {}).get("dirconn") or _NOT_RECORDED,
                "image_tag": image.get("tag") or _NOT_RECORDED,
                "image_revision": image.get("revision") or _NOT_RECORDED,
                "git": provenance.get("git") or _NOT_RECORDED,
            }
        )
    worst = worst_status(statuses)
    caption = (
        f"{len(rows)} unit(s) of subject {subject.subject}; the status is the worst of the "
        f"{len(statuses)} reportlets of the subject."
    )
    return _Body(caption, worst, table=_plain(rows))


def subject_reportlets(
    subject: SubjectOutputs, unit_reports: Mapping[str, Sequence[Reportlet]] | None = None
) -> list[Reportlet]:
    """Build the subject-level reportlets ``summary/overview`` and ``summary/errors``.

    Parameters
    ----------
    subject : SubjectOutputs
        The collected outputs of the subject.
    unit_reports : mapping of str to sequence of Reportlet, optional
        The unit reportlets by unit label, as built by :func:`unit_reportlets`. They give the
        worst status of the overview and the build failures of the errors table. When omitted
        they are built here (which renders every figure again), so pass them when you have them.

    Returns
    -------
    list of Reportlet
        ``summary/errors`` first, then ``summary/overview`` (its status is the worst of every
        reportlet of the subject).
    """
    if unit_reports is None:
        unit_reports = {u.label: _build_unit(subject, u) for u in subject.units}
    label = f"sub-{subject.subject}"
    sink = _Sink(label)
    sink.add("summary", "errors", "errors", "Errors", lambda: _errors_body(subject, unit_reports))
    errors = sink.items[-1]
    everything = [r for reports in unit_reports.values() for r in reports]
    sink.add(
        "summary",
        "overview",
        "summary",
        "Overview",
        lambda: _overview_body(subject, everything, errors.status),
    )
    return sink.items


# ---------------------------------------------------------------------------------------------
# writing


def _clear(figures_dir: Path, label: str) -> None:
    """Delete the reportlet files of ``label`` (never diagnostics or other labels)."""
    if not figures_dir.is_dir():
        return
    prefix = f"{label}_desc-"
    for path in figures_dir.iterdir():
        if path.name.startswith(prefix) and path.suffix in (".json", ".svg"):
            path.unlink()


def build_reportlets(root: Path, subject: str) -> list[Path]:
    """Build and write all reportlets of one subject.

    The reportlet files of each label written here are deleted first (diagnostics files and other
    labels stay), so a rebuild gives the same files as a first build.

    Parameters
    ----------
    root : Path
        Dataset root (``<out_dir>/dirconn``).
    subject : str
        Subject label without the ``sub-`` prefix.

    Returns
    -------
    list of Path
        The files written. Empty when the subject has neither results nor failed tasks.
    """
    root = Path(root)
    outputs = collect_subject(root, subject)
    if not outputs.units and not outputs.failures:
        return []
    unit_reports = {u.label: _build_unit(outputs, u) for u in outputs.units}
    written: list[Path] = []
    for unit in outputs.units:
        _clear(unit.figures_dir, unit.label)
        for reportlet in unit_reports[unit.label]:
            written.extend(write_reportlet(unit.figures_dir, unit.label, reportlet))
    label = f"sub-{subject}"
    subject_dir = root / label / "figures"
    _clear(subject_dir, label)
    for reportlet in subject_reportlets(outputs, unit_reports):
        written.extend(write_reportlet(subject_dir, label, reportlet))
    return written

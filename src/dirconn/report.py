"""The generated benchmark report: tables from the saved results, nothing typed by hand.

Every number of ``docs/benchmark_results.md`` is computed here from the results files that the
protocols wrote (``benchmark_results/<protocol>.jsonl``). The hand-written reading of the numbers
is a separate file that :func:`build_report` puts in front; it is never edited by the generator.
The tables are flat Markdown: a grid of mean AUROC (rows are conditions, columns estimators), the
differences between two settings taken task by task (mean, sd and the number of pairs), and the
run time against the estimate of the budget gate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dirconn.protocol import (
    _leakage_sections,
    _noise_section,
    _significance_section,
    read_results,
    saturation_report,
)
from dirconn.scenario import load_scenario

PUBLISHED: tuple[str, ...] = (
    "full",
    "order",
    "inverse",
    "coupling",
    "extent",
    "size",
    "leakage_adjacent",
    "leakage_pairs",
    "significance",
    "noise",
)
"""The protocols of the published run, in the order of the report."""
ADDONS: tuple[str, ...] = (
    "noise_hard",
    "significance_power",
    "m5_methods",
    "asymptotic_calibration",
)
"""Protocols run after the published run. Each has a budget of its own (at most 0.1 h, except
``m5_methods`` with 0.3 h and ``asymptotic_calibration`` with 1.5 h, set from the gate estimates
of its design) and is listed apart
from the total of the ten; the report takes them when their results file exists and does not
need them."""
STAGES: tuple[str, ...] = ("roi", "sensor", "source")
TOTAL_BUDGET_HOURS = 2.0

Published = Mapping[
    str, tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any] | None]
]


def load_published(
    directory: str | Path, names: Sequence[str] = PUBLISHED
) -> dict[str, tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]]:
    """Read the results file of every protocol in ``names`` from ``directory``.

    Raises
    ------
    FileNotFoundError
        Naming every protocol whose file is missing.
    """
    directory = Path(directory)
    missing = [n for n in names if not (directory / f"{n}.jsonl").is_file()]
    if missing:
        raise FileNotFoundError(f"No results file in {directory} for: {', '.join(missing)}.")
    return {n: read_results(directory / f"{n}.jsonl") for n in names}


def load_add_ons(
    directory: str | Path,
) -> dict[str, tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]]:
    """Read the results file of each add-on protocol found in ``directory``; none is required."""
    directory = Path(directory)
    return {
        n: read_results(directory / f"{n}.jsonl")
        for n in ADDONS
        if (directory / f"{n}.jsonl").is_file()
    }


def _effective_coupling(scenario: str, coupling: float | None) -> float | None:
    """Return the strength of a task's edges: the coupling set, else the file's if it is single."""
    if coupling is not None:
        return float(coupling)
    try:
        strengths = {abs(e.strength) for e in load_scenario(scenario).network.edges}
    except ValueError:
        return None  # not a shipped scenario, for example a region pair
    return strengths.pop() if len(strengths) == 1 else None


def flatten(protocol: str, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one row per finished task, estimator and stage.

    A row carries the conditions of the task (``scenario``, ``snr_db``, ``duration_s``,
    ``coupling`` as the strength of its edges, ``source_model``, ``modality``, ``noise``,
    ``seed_index``), the estimator split into ``label``, ``method``, ``order`` and ``inverse``
    (``None`` unless the label has ``@method``), a ``column`` that joins label and inverse, the
    ``stage`` and its ``auroc``, ``direction_accuracy`` and ``separation``.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "ok":
            continue
        task = record["task"]
        coupling = _effective_coupling(task["scenario"], task.get("coupling"))
        for full_label, payload in record.get("result", {}).items():
            label, _, inverse = full_label.partition("@")
            method, _, order = label.partition("/")
            for stage in STAGES:
                card = payload.get(stage)
                if not isinstance(card, Mapping) or "auroc" not in card:
                    continue
                rows.append(
                    {
                        "protocol": protocol,
                        "task_id": record["id"],
                        "scenario": task["scenario"],
                        "snr_db": task["snr_db"],
                        "duration_s": task["duration_s"],
                        "coupling": coupling,
                        "source_model": task["source_model"],
                        "modality": task["modality"],
                        "noise": task.get("noise", "simulated"),
                        "seed_index": task["seed_index"],
                        "label": label,
                        "method": method,
                        "order": order,
                        "inverse": inverse or None,
                        "column": full_label,
                        "stage": stage,
                        "auroc": float(card["auroc"]),
                        "direction_accuracy": float(card.get("direction_accuracy", math.nan)),
                        "separation": float(card.get("separation", math.nan)),
                    }
                )
    return rows


def _sort_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(
        (0, float(v), "")
        if isinstance(v, (int, float)) and not isinstance(v, bool)
        else (1, 0.0, "" if v is None else str(v))
        for v in key
    )


def mean_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    row_keys: Sequence[str],
    columns: Sequence[str],
    column_field: str = "column",
    value: str = "auroc",
) -> dict[tuple[Any, ...], dict[str, tuple[float, int]]]:
    """Return the mean of ``value`` and the number of defined values per row key and column.

    Rows with an undefined (NaN) value do not count; a cell with none is ``(nan, 0)``.
    """
    values: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    for row in rows:
        if row["stage"] != stage:
            continue
        key = tuple(row[k] for k in row_keys)
        cell = values.setdefault(key, {c: [] for c in columns})
        column = row[column_field]
        if column in cell and math.isfinite(row[value]):
            cell[column].append(row[value])
    return {
        key: {c: ((float(np.mean(v)), len(v)) if v else (math.nan, 0)) for c, v in cells.items()}
        for key, cells in values.items()
    }


def _cell(value: Any) -> str:
    if value is None:
        return "file"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def grid_markdown(
    grid: Mapping[tuple[Any, ...], Mapping[str, tuple[float, int]]],
    *,
    row_names: Sequence[str],
    columns: Sequence[str],
    digits: int = 3,
) -> str:
    """Return a Markdown table of a :func:`mean_grid`: one row per key, one column per estimator."""
    lines = [
        "| " + " | ".join([*row_names, *columns]) + " |",
        "|" + "---|" * (len(row_names) + len(columns)),
    ]
    for key in sorted(grid, key=_sort_key):
        cells = ["-" if grid[key][c][1] == 0 else f"{grid[key][c][0]:.{digits}f}" for c in columns]
        lines.append("| " + " | ".join([*(_cell(k) for k in key), *cells]) + " |")
    return "\n".join(lines)


def paired_differences(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    plus: Any,
    minus: Any,
    match: Sequence[str],
    group_by: Sequence[str],
    value: str = "auroc",
) -> dict[tuple[Any, ...], dict[str, float]]:
    """Return the difference of two settings taken pair by pair, per group.

    Rows with ``row[field] == plus`` are paired with the row with ``row[field] == minus`` that has
    the same values of the ``match`` fields; the difference is plus minus minus. Per group of the
    ``group_by`` fields of the plus row: ``mean``, ``sd`` (ddof 1, NaN for a single pair) and ``n``.
    Pairs with an undefined value on either side are dropped.
    """
    minus_values: dict[tuple[Any, ...], float] = {}
    for row in rows:
        if row[field] == minus:
            minus_values[tuple(row[m] for m in match)] = row[value]
    diffs: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        if row[field] != plus:
            continue
        other = minus_values.get(tuple(row[m] for m in match))
        if other is None or not (math.isfinite(other) and math.isfinite(row[value])):
            continue
        diffs.setdefault(tuple(row[g] for g in group_by), []).append(row[value] - other)
    return {
        key: {
            "mean": float(np.mean(d)),
            "sd": float(np.std(d, ddof=1)) if len(d) > 1 else math.nan,
            "n": len(d),
        }
        for key, d in diffs.items()
    }


def _diff_markdown(
    diffs: Mapping[tuple[Any, ...], Mapping[str, float]], *, row_names: Sequence[str]
) -> str:
    lines = [
        "| " + " | ".join([*row_names, "mean difference", "sd", "pairs"]) + " |",
        "|" + "---|" * (len(row_names) + 3),
    ]
    for key in sorted(diffs, key=_sort_key):
        d = diffs[key]
        sd = "nan" if math.isnan(d["sd"]) else f"{d['sd']:.3f}"
        lines.append(
            "| "
            + " | ".join([*(_cell(k) for k in key), f"{d['mean']:+.3f}", sd, str(d["n"])])
            + " |"
        )
    return "\n".join(lines)


def _hours(seconds: float) -> str:
    return f"{seconds / 3600.0:.2f}"


def runtime_markdown(published: Published) -> str:
    """Return the table of estimated and measured run time per protocol, and the environment."""
    lines = [
        "| protocol | tasks | estimate (h) | measured (h) | workers |",
        "|---|---|---|---|---|",
    ]
    add_on_rows: list[str] = []
    total, known = 0.0, True
    for name, (header, records, final) in published.items():
        gate = header.get("gate") or {}
        estimate = gate.get("estimate_hours")
        measured = "unknown" if final is None else _hours(float(final["seconds"]))
        n_ok = sum(1 for r in records if r.get("status") == "ok")
        row = (
            f"| {name} | {n_ok} | {'?' if estimate is None else f'{float(estimate):.2f}'} "
            f"| {measured} | {gate.get('workers', '?')} |"
        )
        if name in ADDONS:
            add_on_rows.append(row)  # own budget: not part of the total of the ten
            continue
        if final is None:
            known = False
        else:
            total += float(final["seconds"])
        lines.append(row)
    lines.append(
        f"| **total** | | | **{_hours(total)}**{'' if known else ' (some unknown)'} "
        f"| budget {TOTAL_BUDGET_HOURS:g} h |"
    )
    if add_on_rows:
        lines += [
            "",
            "Add-on protocols (each has a budget of its own and is not in the total above):",
            "",
            "| protocol | tasks | estimate (h) | measured (h) | workers |",
            "|---|---|---|---|---|",
            *add_on_rows,
        ]
    envs = {n: dict(h.get("environment", {})) for n, (h, _, _) in published.items()}
    comparable = {n: {k: v for k, v in e.items() if k != "platform"} for n, e in envs.items()}
    reference = next(iter(comparable.values()), {})
    differing = sorted(
        {
            k
            for e in comparable.values()
            for k in set(e) | set(reference)
            if e.get(k) != reference.get(k)
        }
    )
    lines.append("")
    if not differing:
        shown = ", ".join(f"{k} {v}" for k, v in reference.items() if k != "python")
        lines.append(
            f"Environment of every results file: Python {reference.get('python', '?')}; {shown}."
        )
    else:
        for key in differing:
            seen = "; ".join(f"{n} {e.get(key, '-')}" for n, e in comparable.items())
            lines.append(f"The versions of {key} differ between the files: {seen}.")
    codes = {n: h.get("code") for n, (h, _, _) in published.items()}
    lines.append("")
    if any(c is None for c in codes.values()):
        missing = ", ".join(n for n, c in codes.items() if c is None)
        lines.append(f"Code: the commit of the code is unknown for {missing}.")
    else:
        commits = {n: c["git_commit"] for n, c in codes.items() if c}
        if len(set(commits.values())) == 1:
            lines.append(f"Code: git {next(iter(commits.values()))[:7]} for every results file.")
        else:
            seen = "; ".join(f"{n} {commit[:7]}" for n, commit in commits.items())
            lines.append(f"Code: the commits differ between the files: {seen}.")
        dirty = [n for n, c in codes.items() if c and c.get("git_dirty")]
        if dirty:
            lines.append(f"Uncommitted changes in the tree when these ran: {', '.join(dirty)}.")
    return "\n".join(lines)


def _labels(rows: Sequence[Mapping[str, Any]], names: Sequence[str] | None = None) -> list[str]:
    seen: list[str] = []
    for row in rows:
        if row["column"] not in seen:
            seen.append(row["column"])
    if names is None:
        return sorted(seen)
    return [c for c in names if c in seen]


def _stage_grids(
    rows: Sequence[Mapping[str, Any]],
    *,
    row_keys: Sequence[str],
    row_names: Sequence[str],
    stages: Sequence[str] = STAGES,
) -> list[str]:
    columns = _labels(rows)
    out: list[str] = []
    for stage in stages:
        grid = mean_grid(rows, stage=stage, row_keys=row_keys, columns=columns)
        out += [
            f"Stage: {stage}.",
            "",
            grid_markdown(grid, row_names=row_names, columns=columns),
            "",
        ]
    return out


def build_report(published: Published, conclusions: str = "") -> str:
    """Return the Markdown of the benchmark results, with ``conclusions`` in front."""
    parts: list[str] = []
    if conclusions.strip():
        parts += [conclusions.strip(), ""]
    parts += [
        "# Benchmark results (generated)",
        "",
        "Generated by `scripts/make_report.py` from `benchmark_results/*.jsonl`; every number is "
        "computed from those files and nothing here is edited by hand. Cells are mean AUROC over "
        "seeds unless said otherwise; a difference is taken task by task (same simulation) and "
        "shown as mean, sd and number of pairs.",
        "",
    ]
    absent = [n for n in PUBLISHED if n not in published]
    if absent:
        parts += [f"Not run, so absent from this report: {', '.join(absent)}.", ""]
    parts += ["## Run time", "", runtime_markdown(published), ""]

    full_rows = flatten("full", published["full"][1]) if "full" in published else []
    if full_rows:
        by_stage = [
            "| estimator | " + " | ".join(STAGES) + " | sensor - roi | source - roi |",
            "|" + "---|" * (len(STAGES) + 3),
        ]
        columns = _labels(full_rows)
        means = {
            s: mean_grid(full_rows, stage=s, row_keys=("protocol",), columns=columns)[("full",)]
            for s in STAGES
        }
        stage_diffs = {
            (s,): paired_differences(
                full_rows,
                field="stage",
                plus=s,
                minus="roi",
                match=("task_id", "column"),
                group_by=("column",),
            )
            for s in ("sensor", "source")
        }
        for c in columns:
            cells = [f"{means[s][c][0]:.3f}" if means[s][c][1] else "-" for s in STAGES]
            diffs = [
                (lambda d: f"{d['mean']:+.3f} ({d['sd']:.3f}, n={d['n']})" if d else "-")(
                    stage_diffs[(s,)].get((c,))
                )
                for s in ("sensor", "source")
            ]
            by_stage.append(f"| {c} | " + " | ".join([*cells, *diffs]) + " |")
        parts += [
            "## Recovery by stage",
            "",
            "Protocol `full`: five scenarios, SNR 0, -5 and -10 dB, 30 and 60 s, EEG, partly "
            "coherent sources, ten seeds, averaged over all of it; the last two columns are the "
            "mean difference to the region level, taken task by task (sd, pairs).",
            "",
            *by_stage,
            "",
            "By condition (rows: scenario, SNR, data length):",
            "",
            *_stage_grids(
                full_rows,
                row_keys=("scenario", "snr_db", "duration_s"),
                row_names=("scenario", "SNR (dB)", "length (s)"),
            ),
        ]
        guard = saturation_report(published["full"][1])
        parts += [
            "Saturation guard (the chain, best estimator per grid point): "
            + ("passed." if guard.ok else "**violated**.")
            + "".join(f" {p}" for p in guard.problems),
            "",
        ]

        order_diffs = paired_differences(
            full_rows,
            field="order",
            plus="fixed",
            minus="bic",
            match=("task_id", "method", "inverse", "stage"),
            group_by=("method", "stage", "snr_db"),
        )
        parts += [
            "## Order setting",
            "",
            "Fixed order (0.2 s) minus BIC order, the same simulation, in `full`, per method, "
            "stage "
            "and SNR (all scenarios and both lengths pooled):",
            "",
            _diff_markdown(order_diffs, row_names=("method", "stage", "SNR (dB)")),
            "",
        ]
        pair_diffs = paired_differences(
            full_rows,
            field="method",
            plus="gc",
            minus="gc_pairwise",
            match=("task_id", "order", "inverse", "stage"),
            group_by=("scenario", "stage"),
        )
        parts += [
            "## Conditional against pairwise Granger causality",
            "",
            "Conditional minus pairwise GC (AUROC), the same simulation and order setting, in "
            "`full`, "
            "per scenario and stage:",
            "",
            _diff_markdown(pair_diffs, row_names=("scenario", "stage")),
            "",
        ]
    else:
        parts += ["## Recovery by stage", "", "Protocol `full` was not run.", ""]

    parts += _slice_sections(published, full_rows)
    parts += _study_sections(published)
    parts += _extended_section(published)
    return "\n".join(parts).rstrip("\n") + "\n"


def _extended_section(published: Published) -> list[str]:
    """Return the section of the further methods on the head models (add-on): AUROC, direction."""
    if "m5_methods" not in published:
        return []  # an add-on is optional: no section and no "not run" line
    rows = flatten("m5_methods", published["m5_methods"][1])
    columns = _labels(rows)
    out = [
        "## Further methods on the head models (add-on)",
        "",
        "Conditional Granger causality with the BIC order next to the further estimators, on three "
        "simple topologies. Scoring follows each method's convention (`dirconn.evaluation."
        "convention_metrics`): `imcoh` is undirected (magnitude, an edge in either direction, "
        "no direction accuracy), `pcmci` is signed (magnitude), `trgc` and `psi` are net flows "
        "(a pair coupled both ways is left out). Mean AUROC over seeds:",
        "",
    ]
    out += _stage_grids(rows, row_keys=("scenario", "modality"), row_names=("scenario", "modality"))
    out += ["Mean direction accuracy over seeds (NaN for the undirected `imcoh`):", ""]
    for stage in STAGES:
        grid = mean_grid(
            rows,
            stage=stage,
            row_keys=("scenario", "modality"),
            columns=columns,
            value="direction_accuracy",
        )
        out += [
            f"Stage: {stage}.",
            "",
            grid_markdown(grid, row_names=("scenario", "modality"), columns=columns),
            "",
        ]
    return out


def _slice_sections(published: Published, full_rows: list[dict[str, Any]]) -> list[str]:
    """Return the sections of the slices of `full`: coupling, extent, size, order, inverse."""
    out: list[str] = []

    def rows_of(name: str) -> list[dict[str, Any]]:
        return flatten(name, published[name][1]) if name in published else []

    def section(
        title: str,
        text: str,
        name: str,
        rows: list[dict[str, Any]],
        keys: Sequence[str],
        names: Sequence[str],
    ) -> None:
        out.extend([f"## {title}", "", text, ""])
        if name not in published:
            out.extend([f"Protocol `{name}` was not run.", ""])
        else:
            out.extend(_stage_grids(rows, row_keys=keys, row_names=names))

    coupling = rows_of("coupling") + [
        r
        for r in full_rows
        if r["scenario"] in ("chain", "collider", "common_driver") and r["coupling"] == 0.4
    ]
    section(
        "Coupling strength",
        "Protocol `coupling` (0.1 and 0.2) with the file strength 0.4 of `full`, same grid; mean "
        "over "
        "the lengths 30 and 60 s (rows: scenario, coupling, SNR).",
        "coupling",
        coupling,
        ("scenario", "coupling", "snr_db"),
        ("scenario", "coupling", "SNR (dB)"),
    )
    extent = rows_of("extent") + [
        r
        for r in full_rows
        if r["scenario"] in ("chain", "collider") and r["source_model"] == "partial"
    ]
    section(
        "Extent of the sources",
        "Protocol `extent` (focal and patch) with the partly coherent sources of `full`, mean over "
        "the lengths (rows: scenario, source model, SNR).",
        "extent",
        extent,
        ("scenario", "source_model", "snr_db"),
        ("scenario", "source model", "SNR (dB)"),
    )
    size = rows_of("size") + [r for r in full_rows if r["scenario"] == "random_5"]
    section(
        "Network size",
        "Protocol `size` (`random_10`) with `random_5` of `full`, mean over the lengths (rows: "
        "scenario, SNR).",
        "size",
        size,
        ("scenario", "snr_db"),
        ("scenario", "SNR (dB)"),
    )
    order_rows = rows_of("order")
    section(
        "Order protocol",
        "Protocol `order`: scenarios with narrow-band coupling and long delays (rows: scenario, "
        "SNR, "
        "length).",
        "order",
        order_rows,
        ("scenario", "snr_db", "duration_s"),
        ("scenario", "SNR (dB)", "length (s)"),
    )
    if order_rows:
        diffs = paired_differences(
            order_rows,
            field="order",
            plus="fixed",
            minus="bic",
            match=("task_id", "method", "inverse", "stage"),
            group_by=("scenario", "method", "stage"),
        )
        out += [
            "Fixed order minus BIC order in `order`, per scenario, method and stage:",
            "",
            _diff_markdown(diffs, row_names=("scenario", "method", "stage")),
            "",
        ]
    inverse = rows_of("inverse")
    out += [
        "## Inverse methods",
        "",
        "Protocol `inverse`: minimum norm, LCMV, dSPM and eLORETA at stage 3 (columns: estimator "
        "and "
        "inverse method). The region and sensor stages do not depend on the inverse method.",
        "",
    ]
    if "inverse" not in published:
        out += ["Protocol `inverse` was not run.", ""]
    else:
        out += _stage_grids(
            inverse,
            row_keys=("scenario", "modality", "snr_db"),
            row_names=("scenario", "modality", "SNR (dB)"),
            stages=("source",),
        )
    return out


def _study_sections(published: Published) -> list[str]:
    """Leakage, significance and noise: the sections of the summaries, from their own files."""
    out: list[str] = []
    for name, title in (
        ("leakage_adjacent", "Leakage: the neighbour"),
        ("leakage_pairs", "Leakage: pairs against crosstalk"),
        ("significance", "Significance tests"),
        ("noise", "Noise: simulated, white and empty room"),
        ("noise_hard", "Noise: harder SNR (add-on)"),
        ("significance_power", "Significance tests: power and weak coupling (add-on)"),
        ("asymptotic_calibration", "Asymptotic PDC and rPDC tests: calibration (add-on)"),
    ):
        if name in ADDONS and name not in published:
            continue  # an add-on is optional: no section and no "not run" line
        out += [f"## {title}", ""]
        if name not in published:
            out += [f"Protocol `{name}` was not run.", ""]
            continue
        header, records, _ = published[name]
        sections: list[str] = []
        for build in (_leakage_sections, _significance_section):
            sections += build(header, records)
        if name in ("noise", "noise_hard"):
            sections += _noise_section(records)
        # the sections carry their own "## " headings; demote them under this one
        out += [
            line.replace("## ", "### ", 1) if line.startswith("## ") else line for line in sections
        ]
        out.append("")
    return out

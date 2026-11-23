"""Methods boilerplate and reference list for a ``dirconn`` derivatives dataset.

The text describes what the pipeline did (configuration, selected model orders, the parameters
the pipeline filled in, the tests and controls) and cites the method papers from the bundled
``_data/references.json``. References are numbered in order of first citation. The functions
:func:`make_boilerplate` and :func:`render` are pure; :func:`write_citation_files` reads a dataset
root and writes ``logs/CITATION.{html,md,tex,bib}``.
"""

from __future__ import annotations

import html
import json
import os
import re
import statistics
import sys
import tempfile
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata, resources
from pathlib import Path
from typing import Any, Literal

from dirconn.methods.registry import REGISTRY
from dirconn.reports.collect import collect_subject, list_subjects, selected_order

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

__all__ = [
    "Boilerplate",
    "ResultFacts",
    "collect_facts",
    "make_boilerplate",
    "reference_entries",
    "render",
    "write_citation_files",
]

CLAIM = "These are predictive, not interventional, measures of directed interaction."
_SELF = "dirconn"
_CONTROL = "trgc"
_EN_DASH = "–"

_NAMES = {
    "gc": "Granger causality",
    "dtf": "Directed transfer function",
    "pdc": "Partial directed coherence",
    "gpdc": "Generalised partial directed coherence",
    "rpdc": "Renormalised partial directed coherence",
    "ssgc": "State-space Granger causality",
    "mtgc": "Multitaper nonparametric Granger causality",
    "imcoh": "Imaginary part of coherency",
    "psi": "Phase slope index",
    "te": "Transfer entropy",
    "pcmci": "PCMCI",
    "adtf": "Adaptive directed transfer function",
    "tpdc": "Time-varying partial directed coherence",
    "wdtf": "Windowed directed transfer function",
}
#: Methods with a built-in test: the name used in the text and the test.
_BUILTIN_TESTS = {
    "pcmci": ("pcmci", "MCI test"),
    "te": ("multivariate transfer entropy", "permutation test"),
}
_SURROGATES = {
    "auto": "block shuffle on continuous and trial shuffle on epoched recordings",
}


@dataclass(frozen=True)
class ResultFacts:
    """Summary of one result file (``.nc`` plus sidecar).

    Parameters
    ----------
    method : str
        Method name (``desc`` entity).
    label : str
        Label of the unit (recording) the result belongs to.
    model_order : int or None
        ``provenance.model_order`` of the result.
    parameters : dict
        ``provenance.parameters`` of the result, including the pipeline defaults such as
        ``segment_length`` and ``window``.
    tests : tuple of str
        Distinct ``Test`` values of the sidecar ``Statistics`` entries.
    untested : bool
        Whether the sidecar has a ``StatisticsNote`` (the result was not tested).
    generators : tuple of (str, int or None)
        Distinct surrogate generators of the ``Statistics`` entries: the generator name and
        its ``block_size`` in samples (``None`` without one). Built-in and analytic tests
        have no surrogate generator and are not listed.
    """

    method: str
    label: str
    model_order: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    tests: tuple[str, ...] = ()
    untested: bool = False
    generators: tuple[tuple[str, int | None], ...] = ()


@dataclass(frozen=True)
class Boilerplate:
    """Methods text and its references.

    Parameters
    ----------
    paragraphs : tuple of str
        Plain-text English paragraphs; citations are written as ``[n]``.
    references : tuple of str
        Reference keys, numbered ``1..n`` in this order (order of first citation).
    version : str
        ``dirconn`` version, shown with the software reference.
    """

    paragraphs: tuple[str, ...]
    references: tuple[str, ...]
    version: str = ""


# --------------------------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _references() -> dict[str, dict[str, Any]]:
    text = (resources.files("dirconn.reports") / "_data" / "references.json").read_text("utf-8")
    refs: dict[str, dict[str, Any]] = json.loads(text)
    return refs


@lru_cache(maxsize=1)
def _defaults() -> dict[str, Any]:
    text = (resources.files("dirconn.workflows") / "_data" / "default_config.toml").read_text(
        "utf-8"
    )
    return tomllib.loads(text)


def _option(config: Mapping[str, Any], section: str, key: str) -> Any:
    """Return ``config[section][key]``, falling back to the packaged default."""
    value = (config.get(section) or {}).get(key)
    if value is None:
        value = (_defaults().get(section) or {}).get(key)
    return value


def _section(config: Mapping[str, Any], section: str, key: str) -> Mapping[str, Any]:
    value = _option(config, section, key)
    return value if isinstance(value, Mapping) else {}


class _Numbering:
    """Assign reference numbers in order of first citation."""

    def __init__(self) -> None:
        self.keys: list[str] = []

    def cite(self, keys: Sequence[str]) -> str:
        known = [k for k in keys if k in _references()]
        if not known:
            return ""
        numbers = []
        for key in known:
            if key not in self.keys:
                self.keys.append(key)
            numbers.append(self.keys.index(key) + 1)
        return "[" + ", ".join(str(n) for n in sorted(numbers)) + "]"


# --------------------------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------------------------


def _seconds(value: Any) -> str:
    return f"{float(value):g} s"


def _join(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.lower() == "null")


def _preparation(config: Mapping[str, Any]) -> str:
    parts: list[str] = []
    resample = _option(config, "prepare", "resample_hz")
    if not _is_null(resample):
        parts.append(f"Signals were resampled to {float(resample):g} Hz.")
    if _option(config, "prepare", "zscore"):
        parts.append("Each channel was z-scored.")
    if _option(config, "prepare", "bad_segments") == "mask":
        pad = _option(config, "prepare", "pad_bad_seconds")
        text = "Annotated bad segments were masked out of the estimates"
        if isinstance(pad, int | float) and pad > 0:
            text += f" with {_seconds(pad)} of padding around each segment"
        parts.append(text + ".")
    else:
        parts.append("Bad-segment annotations were ignored.")
    lock = _section(config, "prepare", "lock")
    events = lock.get("events") or []
    if events:
        names = _join([f"'{e}'" for e in events])
        tmin, tmax = lock.get("tmin", -1.0), lock.get("tmax", 1.0)
        parts.append(
            f"Data were epoched around the events {names} ({float(tmin):g} to {float(tmax):g} s)."
        )
    else:
        parts.append("Continuous recordings were analysed.")
    return " ".join(parts)


def _all_recordings(n: int) -> str:
    """Phrase for every one of ``n`` recordings: "the one recording", "both ..." or "all N ..."."""
    if n == 1:
        return "the one recording"
    return "both recordings" if n == 2 else f"all {n} recordings"


def _orders(
    config: Mapping[str, Any],
    facts: Sequence[ResultFacts],
    selections: Mapping[str, Mapping[str, Any] | None] | None,
) -> str | None:
    order_cfg = _option(config, "estimation", "order")
    chosen: list[int] = []
    criterion: str | None = None
    for selection in (selections or {}).values():
        if not selection:
            continue
        try:
            chosen.append(selected_order(selection))
        except ValueError:
            continue
        criterion = criterion or str(selection.get("criterion"))
    order_max = _option(config, "estimation", "order_max")
    if chosen:
        lo, hi, n = min(chosen), max(chosen), len(chosen)
        if lo == hi:
            summary = f"selected order {lo} in " + _all_recordings(n)
        else:
            median = f"{statistics.median(chosen):g}"
            summary = f"selected orders {lo}{_EN_DASH}{hi}, median {median} over {n} recordings"
        at_limit = sum(1 for o in chosen if o == order_max)
        if at_limit == n:
            summary += "; this is the largest order searched"
        elif at_limit:
            summary += (
                f"; in {at_limit} of {n} recordings the selected order was the largest order "
                "searched"
            )
        return (
            f"The model order of the autoregressive models was selected once per recording by "
            f"{criterion} up to {order_max}; {summary}."
        )
    fixed = sorted({f.model_order for f in facts if f.model_order is not None})
    if fixed:
        listing = ", ".join(str(o) for o in fixed)
        return f"The model order of the autoregressive models was fixed at {listing}."
    if isinstance(order_cfg, int) and not isinstance(order_cfg, bool):
        return f"The model order of the autoregressive models was fixed at {order_cfg}."
    if isinstance(order_cfg, str):
        return (
            f"The model order of the autoregressive models was selected once per recording by "
            f"{order_cfg} up to {order_max}."
        )
    return None


def _bands_text(config: Mapping[str, Any]) -> str:
    bands = _section(config, "estimation", "bands")
    items = []
    for name, edges in bands.items():
        if isinstance(edges, Sequence) and len(edges) == 2:
            items.append(f"{name} ({float(edges[0]):g}{_EN_DASH}{float(edges[1]):g} Hz)")
    return ", ".join(items)


def _method_sentence(
    method: str,
    facts: Sequence[ResultFacts],
    config: Mapping[str, Any],
    numbering: _Numbering,
) -> str:
    cls = REGISTRY.get(method)
    info = cls.info if cls is not None else None
    name = _NAMES.get(method, method)
    cites = numbering.cite(info.citations) if info is not None else ""
    head = f"{name} ({method})" + (f" {cites}" if cites else "")
    text = f"{head} was estimated"
    if info is not None and info.time_resolved:
        text += " in sliding windows"
    if info is not None and info.frequency_resolved:
        bands = _bands_text(config)
        reduce = _option(config, "estimation", "reduce")
        if bands and method == "psi":
            text += f" in each of the bands {bands}"
        elif bands:
            text += f" and reduced ({reduce}) to the bands {bands}"
    values: dict[str, list[str]] = {}
    for key in ("segment_length", "window", "step"):
        for f in facts:
            value = f.parameters.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                formatted = _seconds(value)
                if formatted not in values.setdefault(key, []):
                    values[key].append(formatted)
    labels = {"segment_length": "segment length", "window": "window", "step": "step"}
    clauses = [f"{labels[k]} {' or '.join(v)}" for k, v in values.items()]
    if clauses:
        text += " with " + _join(clauses)
    return text + "."


def _methods(
    config: Mapping[str, Any], facts: Sequence[ResultFacts], numbering: _Numbering
) -> str | None:
    order: list[str] = []
    by_method: dict[str, list[ResultFacts]] = {}
    for f in facts:
        if f.method == _CONTROL:
            continue
        if f.method not in by_method:
            order.append(f.method)
        by_method.setdefault(f.method, []).append(f)
    if not order:
        return None
    sentences = [_method_sentence(m, by_method[m], config, numbering) for m in order]
    return " ".join(sentences)


def _generators_text(facts: Sequence[ResultFacts]) -> str:
    """Describe the surrogate generators recorded in the sidecars; empty when none is recorded."""
    seen: dict[str, set[int]] = {}
    for f in facts:
        for name, block in f.generators:
            seen.setdefault(name, set())
            if block is not None:
                seen[name].add(block)
    items = []
    for name, blocks in seen.items():
        text = name.replace("_", " ")
        if blocks:
            sizes = " or ".join(str(b) for b in sorted(blocks))
            text += f", blocks of {sizes} samples"
        items.append(text)
    return " and ".join(items)


def _tests(config: Mapping[str, Any], facts: Sequence[ResultFacts], numbering: _Numbering) -> str:
    test = _option(config, "statistics", "test")
    if test == "none":
        return "Statistical testing was disabled: no statistical test was run."
    alpha = _option(config, "statistics", "alpha")
    parts: list[str] = []
    if test == "surrogate":
        n = _option(config, "statistics", "n_surrogates")
        gen = str(_option(config, "statistics", "surrogate"))
        gen_text = _generators_text(facts) or _SURROGATES.get(gen, gen.replace("_", " "))
        cite = numbering.cite(("Theiler_1992",))
        parts.append(
            f"Significance was assessed against {n} surrogates ({gen_text})"
            f"{' ' + cite if cite else ''}."
        )
        parts.append(
            "Surrogate statistics used one set of surrogates per recording and method, "
            "shared by all bands and fitted with the recording's selected order."
        )
    else:
        parts.append("Significance was assessed with analytic tests.")
    correction = _option(config, "statistics", "correction")
    if correction == "fdr_bh":
        cite = numbering.cite(("Benjamini_1995",))
        parts.append(
            "P-values were corrected for multiple comparisons by the Benjamini-Hochberg false "
            f"discovery rate procedure{' ' + cite if cite else ''} at alpha = {alpha}."
        )
    elif correction == "bonferroni":
        parts.append(f"P-values were Bonferroni-corrected at alpha = {alpha}.")
    else:
        parts.append(f"P-values were not corrected for multiple comparisons (alpha = {alpha}).")
    untested = sorted({f.method for f in facts if f.untested and f.method != _CONTROL})
    if untested:
        parts.append(
            f"For {_join(untested)}, time-resolved results on continuous recordings were not "
            "tested and are descriptive."
        )
    builtin = [
        m for m in _BUILTIN_TESTS if any(f.method == m and "builtin" in f.tests for f in facts)
    ]
    if builtin:
        names = _join([_BUILTIN_TESTS[m][0] for m in builtin])
        tests = "; ".join(_BUILTIN_TESTS[m][1] for m in builtin)
        verb = "uses its" if len(builtin) == 1 else "use their"
        parts.append(f"{names} {verb} own p-values ({tests}).")
    return " ".join(parts)


def _controls(config: Mapping[str, Any], numbering: _Numbering) -> str | None:
    parts: list[str] = []
    if _option(config, "controls", "time_reversal"):
        cls = REGISTRY.get(_CONTROL)
        cite = numbering.cite(cls.info.citations) if cls is not None else ""
        parts.append(
            "Time-reversed Granger causality was computed as a control for zero-delay mixing"
            f"{' ' + cite if cite else ''}."
        )
    if _option(config, "controls", "spectral_range_check"):
        parts.append(
            "The band edges were checked against the usable spectral range of each recording "
            "(spectral dynamic range check)."
        )
    return " ".join(parts) or None


def make_boilerplate(
    config: Mapping[str, Any],
    facts: Sequence[ResultFacts],
    *,
    version: str,
    order_selections: Mapping[str, Mapping[str, Any] | None] | None = None,
) -> Boilerplate:
    """Compose the methods paragraphs and number the references.

    Parameters
    ----------
    config : mapping
        Parsed ``dirconn.toml`` (sections ``prepare``, ``estimation``, ``statistics``,
        ``controls``). Missing keys take the packaged defaults.
    facts : sequence of ResultFacts
        One entry per result file; the ``trgc`` control is not described as a method.
    version : str
        ``dirconn`` version to cite.
    order_selections : mapping, optional
        Unit label to the ``order_selection`` of its diagnostics (``None`` for a fixed order).

    Returns
    -------
    Boilerplate
        Paragraphs with ``[n]`` citations and the reference keys in order of first citation.
    """
    numbering = _Numbering()
    paragraphs: list[str] = [_preparation(config)]
    orders = _orders(config, facts, order_selections)
    if orders:
        paragraphs.append(orders)
    methods = _methods(config, facts, numbering)
    if methods:
        paragraphs.append(methods)
    paragraphs.append(_tests(config, facts, numbering))
    controls = _controls(config, numbering)
    if controls:
        paragraphs.append(controls)
    cite = numbering.cite((_SELF,))
    paragraphs.append(
        f"{CLAIM} Analyses were run with dirconn {version} {cite}.".replace(" .", ".")
    )
    return Boilerplate(tuple(paragraphs), tuple(numbering.keys), version)


# --------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------


def _initials(given: str) -> str:
    tokens = [t for t in re.split(r"[\s\-]+", given) if t]
    return " ".join(t[0] + "." for t in tokens)


def _author(name: str) -> str:
    family, sep, given = name.partition(", ")
    return f"{family}, {_initials(given)}" if sep else family


def _entry_parts(bp: Boilerplate, key: str) -> tuple[str, str, str, str]:
    """Return (authors, title, source, doi) as plain text with an en dash in the pages."""
    ref = _references()[key]
    authors = "; ".join(_author(a) for a in ref["authors"])
    source = str(ref.get("journal") or ref.get("booktitle") or "")
    if ref.get("volume"):
        source += f" {ref['volume']}"
        if ref.get("issue"):
            source += f"({ref['issue']})"
    if ref.get("pages"):
        source += f", {str(ref['pages']).replace('-', _EN_DASH)}"
    if key == _SELF and bp.version:
        source = f"Software, version {bp.version}"
    head = f"{authors} ({ref['year']}). {ref['title']}."
    return head, source, str(ref["doi"]), key


def _doi_url(doi: str) -> str:
    return "https://doi.org/" + urllib.parse.quote(doi, safe="/()-._~:;")


def reference_entries(bp: Boilerplate) -> list[tuple[str, str | None]]:
    """Return the formatted references of ``bp`` in citation order.

    Parameters
    ----------
    bp : Boilerplate
        The text and reference keys.

    Returns
    -------
    list of tuple of (str, str or None)
        For each reference, its text without markup (authors, year, title and source, ending in
        a full stop) and the DOI link (``https://doi.org/...``), ``None`` when it has no DOI.
        Entry ``i`` is the reference numbered ``i + 1``.
    """
    entries: list[tuple[str, str | None]] = []
    for key in bp.references:
        head, source, doi, _ = _entry_parts(bp, key)
        entries.append((f"{head} {source}.", _doi_url(doi) if doi else None))
    return entries


_TEX = {
    "\\": "\\textbackslash{}",
    "&": "\\&",
    "%": "\\%",
    "$": "\\$",
    "#": "\\#",
    "_": "\\_",
    "{": "\\{",
    "}": "\\}",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
}
_TEX_RE = re.compile("[" + re.escape("".join(_TEX)) + "]")


def _tex(text: str) -> str:
    return _TEX_RE.sub(lambda m: _TEX[m.group(0)], text).replace(_EN_DASH, "--")


_BIB_RE = re.compile(r"[&%$#_{}]")


def _bib_value(text: str) -> str:
    return _BIB_RE.sub(lambda m: "\\" + m.group(0), text)


def _render_md(bp: Boilerplate) -> str:
    lines = ["# Methods", ""]
    for paragraph in bp.paragraphs:
        lines += [paragraph, ""]
    lines += ["## References", ""]
    for n, key in enumerate(bp.references, start=1):
        head, source, doi, _ = _entry_parts(bp, key)
        lines.append(f"{n}. {head} {source}. <{_doi_url(doi)}>")
    return "\n".join(lines) + "\n"


def _render_html(bp: Boilerplate) -> str:
    esc = html.escape
    items = []
    for key in bp.references:
        head, source, doi, _ = _entry_parts(bp, key)
        items.append(
            f'<li id="ref-{esc(key)}">{esc(head)} {esc(source)}. '
            f'<a href="{esc(_doi_url(doi))}">doi:{esc(doi)}</a></li>'
        )
    paragraphs = "\n".join(f"<p>{esc(p)}</p>" for p in bp.paragraphs)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>dirconn methods</title>\n"
        "<style>body{font-family:sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem;"
        "line-height:1.5}li{margin-bottom:.4rem}</style>\n"
        "</head>\n<body>\n<h1>Methods</h1>\n"
        f"{paragraphs}\n<h2>References</h2>\n<ol>\n"
        + "\n".join(items)
        + "\n</ol>\n</body>\n</html>\n"
    )


def _render_tex(bp: Boilerplate) -> str:
    out = [
        "\\documentclass{article}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage[T1]{fontenc}",
        "\\begin{document}",
        "\\section*{Methods}",
        "",
    ]
    for paragraph in bp.paragraphs:
        out += [_tex(paragraph), ""]
    out += ["\\begin{thebibliography}{99}"]
    for key in bp.references:
        head, source, doi, _ = _entry_parts(bp, key)
        out.append(f"\\bibitem{{{key}}} {_tex(head)} {_tex(source)}. doi:{_tex(doi)}")
    out += ["\\end{thebibliography}", "\\end{document}", ""]
    return "\n".join(out)


def _render_bib(bp: Boilerplate) -> str:
    blocks = []
    for key in bp.references:
        ref = _references()[key]
        kind = {"article": "article", "inproceedings": "inproceedings", "inbook": "incollection"}
        entry_type = kind.get(ref["type"], "misc")
        fields: list[tuple[str, str]] = [
            ("author", " and ".join(_bib_value(a) for a in ref["authors"])),
            ("title", _bib_value(ref["title"])),
        ]
        if "journal" in ref:
            fields.append(("journal", _bib_value(ref["journal"])))
        if "booktitle" in ref:
            fields.append(("booktitle", _bib_value(ref["booktitle"])))
        fields.append(("year", str(ref["year"])))
        for src, dst in (("volume", "volume"), ("issue", "number")):
            if ref.get(src):
                fields.append((dst, _bib_value(str(ref[src]))))
        if ref.get("pages"):
            fields.append(("pages", str(ref["pages"]).replace("-", "--")))
        if key == _SELF:
            if bp.version:
                fields.append(("version", _bib_value(bp.version)))
            fields.append(("note", "Software"))
        fields.append(("doi", str(ref["doi"])))
        body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
        blocks.append(f"@{entry_type}{{{key},\n{body}\n}}")
    return "\n\n".join(blocks) + "\n"


def render(bp: Boilerplate, fmt: Literal["html", "md", "tex", "bib"]) -> str:
    """Render ``bp`` as HTML, Markdown, LaTeX or BibTeX.

    Parameters
    ----------
    bp : Boilerplate
        The text and reference keys.
    fmt : {"html", "md", "tex", "bib"}
        ``"html"`` is a self-contained page without scripts or external resources (the only links
        are DOI links); ``"tex"`` a small document with ``thebibliography``; ``"bib"`` the entries
        only (software and book chapters become ``@misc`` and ``@incollection``).

    Returns
    -------
    str
        The rendered document.

    Raises
    ------
    ValueError
        If ``fmt`` is unknown.
    """
    if fmt == "md":
        return _render_md(bp)
    if fmt == "html":
        return _render_html(bp)
    if fmt == "tex":
        return _render_tex(bp)
    if fmt == "bib":
        return _render_bib(bp)
    raise ValueError(f"unknown format {fmt!r}; use 'html', 'md', 'tex' or 'bib'")


# --------------------------------------------------------------------------------------------
# dataset level
# --------------------------------------------------------------------------------------------


def collect_facts(
    root: Path, subjects: Sequence[str] | None = None
) -> tuple[list[ResultFacts], dict[str, Mapping[str, Any] | None]]:
    """Read the result facts and order selections of a dataset.

    Parameters
    ----------
    root : Path
        Dataset root (``<out_dir>/dirconn``).
    subjects : sequence of str, optional
        Subject labels without ``sub-``; all subjects below ``root`` when omitted.

    Returns
    -------
    facts : list of ResultFacts
        One entry per method result (the ``trgc`` control is not included).
    order_selections : dict
        Unit label to the ``order_selection`` of its diagnostics (``None`` for a fixed order);
        units without a diagnostics file are absent.
    """
    root = Path(root)
    facts: list[ResultFacts] = []
    selections: dict[str, Mapping[str, Any] | None] = {}
    for subject in list_subjects(root) if subjects is None else subjects:
        for unit in collect_subject(root, subject).units:
            if unit.diagnostics is not None:
                selections[unit.label] = unit.diagnostics.get("order_selection")
            for desc, output in unit.methods.items():
                provenance = output.connectivity().provenance
                statistics_entries = output.sidecar.get("Statistics") or []
                tests = sorted(
                    {
                        str(s["Test"])
                        for s in statistics_entries
                        if isinstance(s, dict) and "Test" in s
                    }
                )
                generators = []
                for s in statistics_entries:
                    gen = s.get("Generator") if isinstance(s, dict) else None
                    if not isinstance(gen, dict) or gen.get("name") in (
                        None,
                        "builtin",
                        "analytic",
                    ):
                        continue
                    name = gen["name"]
                    block = (gen.get("params") or {}).get("block_size")
                    item = (str(name), None if block is None else int(block))
                    if item not in generators:
                        generators.append(item)
                facts.append(
                    ResultFacts(
                        method=desc,
                        label=unit.label,
                        model_order=provenance.model_order,
                        parameters=dict(provenance.parameters),
                        tests=tuple(tests),
                        untested=output.statistics_note is not None,
                        generators=tuple(sorted(generators, key=lambda g: (g[0], g[1] or 0))),
                    )
                )
    return facts, selections


def write_citation_files(root: Path) -> list[Path]:
    """Write ``logs/CITATION.{html,md,tex,bib}`` for the dataset at ``root``.

    Parameters
    ----------
    root : Path
        Dataset root (``<out_dir>/dirconn``) holding ``dirconn.toml`` and the subject results.

    Returns
    -------
    list of Path
        The four files written (each replaced atomically).

    Raises
    ------
    FileNotFoundError
        If ``root/dirconn.toml`` does not exist.
    """
    root = Path(root)
    config_path = root / "dirconn.toml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"no dirconn.toml in {root}; run the participant stage first to create the dataset"
        )
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    facts, selections = collect_facts(root)
    bp = make_boilerplate(
        config, facts, version=metadata.version("dirconn"), order_selections=selections
    )
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in ("html", "md", "tex", "bib"):
        target = logs / f"CITATION.{fmt}"
        descriptor, name = tempfile.mkstemp(dir=logs, prefix=f".CITATION.{fmt}.", suffix=".tmp")
        temp = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(render(bp, fmt))
            os.replace(temp, target)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        written.append(target)
    return written

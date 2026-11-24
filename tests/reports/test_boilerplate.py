"""Tests for ``dirconn.reports.boilerplate``."""

from __future__ import annotations

import json
import re
import shutil
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from dirconn.reports.boilerplate import (
    Boilerplate,
    ResultFacts,
    collect_facts,
    make_boilerplate,
    reference_entries,
    render,
    write_citation_files,
)

CLAIM = "These are predictive, not interventional, measures of directed interaction."

CONFIG: dict[str, Any] = {
    "prepare": {"resample_hz": 250.0, "zscore": True, "bad_segments": "mask"},
    "estimation": {"methods": ["gc"], "order": "bic", "order_max": 6, "bands": {"beta": [13, 30]}},
    "statistics": {
        "test": "surrogate",
        "surrogate": "auto",
        "n_surrogates": 19,
        "alpha": 0.05,
        "correction": "fdr_bh",
    },
    "controls": {"time_reversal": True, "spectral_range_check": True},
}


def _facts(method: str = "gc", **kw: Any) -> ResultFacts:
    base: dict[str, Any] = {
        "method": method,
        "label": "sub-01_task-rest",
        "model_order": 3,
        "parameters": {},
        "tests": ("surrogate",),
        "untested": False,
    }
    base.update(kw)
    return ResultFacts(**base)


def _selection(order: int) -> dict[str, Any]:
    orders = list(range(1, max(order, 5) + 1))
    bic = [1.0 if o != order else 0.0 for o in orders]
    return {"criterion": "bic", "orders": orders, "bic": bic}


def _text(bp: Boilerplate) -> str:
    return "\n".join(bp.paragraphs)


def _refs() -> dict[str, Any]:
    text = (resources.files("dirconn.reports") / "_data" / "references.json").read_text("utf-8")
    refs: dict[str, Any] = json.loads(text)
    return refs


def test_every_reference_has_a_doi() -> None:
    refs = _refs()
    assert len(refs) >= 25
    for key, ref in refs.items():
        assert ref["doi"].startswith("10."), key
        assert ref["authors"] and ref["title"] and ref["year"], key
        assert ref["type"] in {"article", "inproceedings", "inbook", "misc", "software"}, key
        if ref["type"] == "article":
            assert ref["journal"], key
        elif ref["type"] in {"inproceedings", "inbook"}:
            assert ref["booktitle"], key


def test_boilerplate_states_the_claim() -> None:
    bp = make_boilerplate(CONFIG, [_facts()], version="1.2.3")
    text = _text(bp)
    assert CLAIM in text
    assert "dirconn 1.2.3" in text
    assert "dirconn" in bp.references


def test_numbering_follows_first_citation() -> None:
    bp = make_boilerplate(CONFIG, [_facts("gc"), _facts("pdc")], version="1.0")
    text = _text(bp)
    assert bp.references[:3] == ("Granger_1969", "Geweke_1982", "Geweke_1984")
    assert len(set(bp.references)) == len(bp.references)
    numbers = [int(n) for group in re.findall(r"\[([\d, ]+)\]", text) for n in group.split(",")]
    first_seen: list[int] = []
    for n in numbers:
        if n not in first_seen:
            first_seen.append(n)
    assert first_seen == sorted(first_seen)
    assert max(numbers) == len(bp.references)


def test_orders_are_summarised() -> None:
    selections = {"a": _selection(3), "b": _selection(3), "c": _selection(5)}
    bp = make_boilerplate(CONFIG, [_facts()], version="1", order_selections=selections)
    text = _text(bp)
    assert "selected once per recording by bic up to 6" in text
    assert "3–5" in text
    assert "median 3" in text
    assert "over 3 recordings" in text
    assert "largest order searched" not in text

    same = make_boilerplate(
        CONFIG, [_facts()], version="1", order_selections={"a": _selection(3), "b": _selection(3)}
    )
    assert "selected order 3 in both recordings" in _text(same)
    assert "median" not in _text(same)
    one = make_boilerplate(CONFIG, [_facts()], version="1", order_selections={"a": _selection(3)})
    assert "selected order 3 in the one recording" in _text(one)

    some = make_boilerplate(
        CONFIG,
        [_facts()],
        version="1",
        order_selections={"a": _selection(6), "b": _selection(2), "c": _selection(3)},
    )
    assert "in 1 of 3 recordings the selected order was the largest order searched" in _text(some)
    every = make_boilerplate(
        CONFIG,
        [_facts()],
        version="1",
        order_selections={"a": _selection(6), "b": _selection(6)},
    )
    assert "selected order 6 in both recordings; this is the largest order searched" in _text(every)

    fixed = make_boilerplate(
        CONFIG, [_facts(model_order=4)], version="1", order_selections={"a": None, "b": None}
    )
    assert "fixed at 4" in _text(fixed)


def test_untested_and_builtin_are_stated() -> None:
    bp = make_boilerplate(
        CONFIG,
        [_facts("wdtf", untested=True, tests=()), _facts("pcmci", tests=("builtin",))],
        version="1",
    )
    text = _text(bp)
    assert "time-resolved results on continuous recordings were not tested" in text
    assert "are descriptive" in text
    assert "pcmci uses its own p-values (MCI test)" in text
    assert "transfer entropy" not in text
    plain = _text(make_boilerplate(CONFIG, [_facts()], version="1"))
    assert "not tested" not in plain
    assert "their own p-values" not in plain


def test_pipeline_defaults_are_stated() -> None:
    facts = [
        _facts("mtgc", parameters={"segment_length": 2.0}),
        _facts("wdtf", parameters={"window": 0.5, "step": 0.25}),
    ]
    text = _text(make_boilerplate(CONFIG, facts, version="1"))
    assert "2 s" in text
    assert "0.5 s" in text
    assert "0.25 s" in text


def test_tests_and_controls_are_stated() -> None:
    bp = make_boilerplate(CONFIG, [_facts()], version="1")
    text = _text(bp)
    assert "19 surrogates" in text
    assert "one set of surrogates per recording and method" in text
    assert "Benjamini_1995" in bp.references and "Theiler_1992" in bp.references
    assert {"Haufe_2013", "Winkler_2016"} <= set(bp.references)
    none_cfg = {**CONFIG, "statistics": {"test": "none"}, "controls": {}}
    bp2 = make_boilerplate(none_cfg, [_facts()], version="1")
    assert "no statistical test was run" in _text(bp2)
    assert "Theiler_1992" not in bp2.references


def test_render_formats() -> None:
    bp = make_boilerplate(
        CONFIG, [_facts("gc"), _facts("psi"), _facts("adtf"), _facts("tpdc")], version="1.2.3"
    )
    html = render(bp, "html")
    assert "<ol>" in html and "<!DOCTYPE html>" in html
    assert "<script" not in html
    assert set(re.findall(r'(?:href|src)="(https?://[^"/]+)', html)) == {"https://doi.org"}
    assert '<a href="https://doi.org/10.5281/zenodo.24036991">' in html
    md = render(bp, "md")
    assert "1. " in md and CLAIM in md
    tex = render(bp, "tex")
    assert "\\begin{thebibliography}" in tex and "\\bibitem{Granger_1969}" in tex
    bib = render(bp, "bib")
    entries = re.findall(r"^@(\w+)\{", bib, flags=re.M)
    assert len(entries) == len(bp.references)
    allowed = {"article", "inproceedings", "inbook", "incollection", "misc", "software"}
    assert set(entries) <= allowed
    assert bib.count("doi = {") == len(entries)
    assert "version = {1.2.3}" in bib
    with pytest.raises(ValueError):
        render(bp, "docx")  # type: ignore[arg-type]


def test_render_escapes_special_characters() -> None:
    bp = Boilerplate(paragraphs=("a <b> & 50% of c_d [1].",), references=("Granger_1969",))
    assert "&lt;b&gt; &amp;" in render(bp, "html")
    tex = render(bp, "tex")
    assert "50\\% of c\\_d" in tex and "\\&" in tex


def test_write_citation_files(pipeline_run: Any, tmp_path: Path) -> None:
    root = tmp_path / "dirconn"
    shutil.copytree(pipeline_run.root, root)
    paths = write_citation_files(root)
    assert {p.name for p in paths} == {
        "CITATION.html",
        "CITATION.md",
        "CITATION.tex",
        "CITATION.bib",
    }
    assert all(p.parent == root / "logs" and p.is_file() for p in paths)
    md = (root / "logs" / "CITATION.md").read_text(encoding="utf-8")
    assert "selected once per recording by bic" in md
    assert CLAIM in md
    html5lib = pytest.importorskip("html5lib")
    html5lib.parse((root / "logs" / "CITATION.html").read_text(encoding="utf-8"))
    assert not list((root / "logs").glob("*.tmp"))


def test_write_citation_files_needs_the_configuration(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dirconn.toml"):
        write_citation_files(tmp_path)


def test_no_reference_is_a_comment_or_rejoinder() -> None:
    text = (resources.files("dirconn.reports") / "_data" / "references.json").read_text("utf-8")
    for key, ref in json.loads(text).items():
        title = re.sub(r"[^\w\s]", "", ref["title"]).strip().lower()
        assert not title.endswith(("rejoinder", "comment")), key


def test_collect_facts_by_subject(pipeline_run: Any) -> None:
    facts_all, selections_all = collect_facts(pipeline_run.root)
    facts_one, selections_one = collect_facts(pipeline_run.root, ["01"])
    assert facts_one and len(facts_one) < len(facts_all)
    assert {f.label.split("_")[0] for f in facts_one} == {"sub-01"}
    assert set(selections_one) < set(selections_all)
    assert all(label.startswith("sub-01_") for label in selections_one)
    assert collect_facts(pipeline_run.root, []) == ([], {})


def test_reference_entries() -> None:
    bp = make_boilerplate(CONFIG, [_facts()], version="1.2.3")
    entries = reference_entries(bp)
    assert len(entries) == len(bp.references)
    for text, url in entries:
        assert text and "<" not in text
        assert url is None or url.startswith("https://doi.org/")
    assert "Software, version 1.2.3" in entries[-1][0]
    md = render(bp, "md")
    for n, (text, url) in enumerate(entries, start=1):
        assert f"{n}. {text}" in md and f"<{url}>" in md


def test_builtin_sentence_names_only_the_methods_that_ran() -> None:
    both = _text(
        make_boilerplate(
            CONFIG,
            [_facts("pcmci", tests=("builtin",)), _facts("te", tests=("builtin",))],
            version="1",
        )
    )
    assert (
        "pcmci and multivariate transfer entropy use their own p-values "
        "(MCI test; permutation test)" in both
    )
    only_te = _text(make_boilerplate(CONFIG, [_facts("te", tests=("builtin",))], version="1"))
    assert "multivariate transfer entropy uses its own p-values (permutation test)" in only_te
    assert "pcmci uses" not in only_te


def test_surrogate_sentence_follows_the_recorded_generators() -> None:
    cont = _text(
        make_boilerplate(CONFIG, [_facts(generators=(("block_shuffle", 1250),))], version="1")
    )
    assert "block shuffle" in cont and "1250 samples" in cont
    assert "trial shuffle" not in cont
    epoched = _text(
        make_boilerplate(CONFIG, [_facts(generators=(("trial_shuffle", None),))], version="1")
    )
    assert "trial shuffle" in epoched and "block shuffle" not in epoched
    mixed = _text(
        make_boilerplate(
            CONFIG,
            [
                _facts(generators=(("block_shuffle", 1250),)),
                _facts(generators=(("trial_shuffle", None),)),
            ],
            version="1",
        )
    )
    assert "block shuffle" in mixed and "trial shuffle" in mixed


def test_collect_facts_records_the_generators(pipeline_run: Any) -> None:
    facts, _ = collect_facts(pipeline_run.root, ["01"])
    gc = [f for f in facts if f.method == "gc"]
    assert gc and all(f.generators == (("block_shuffle", 1250),) for f in gc)


def test_psi_is_described_per_band() -> None:
    config = {**CONFIG, "estimation": {**CONFIG["estimation"], "methods": ["psi"]}}
    text = _text(make_boilerplate(config, [_facts("psi")], version="1"))
    assert "reduced" not in text
    assert "in each of the bands beta (13" in text


def test_failed_citation_write_leaves_no_temp_file(
    pipeline_run: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    out = tmp_path / "out"
    shutil.copytree(pipeline_run.root.parent, out)
    root = out / "dirconn"

    def broken(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        write_citation_files(root)
    assert not list((root / "logs").glob("*.tmp"))

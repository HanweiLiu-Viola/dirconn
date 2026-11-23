import inspect

import pytest

from dirconn.methods.base import REPORT_KINDS, Estimator, MethodInfo
from dirconn.methods.registry import REGISTRY, get_estimator, list_methods

EXPECTED_NAMES = {
    "gc", "dtf", "pdc", "gpdc", "rpdc", "ssgc", "mtgc", "imcoh", "psi", "trgc", "te", "pcmci",
    "adtf", "tpdc", "wdtf",
}  # fmt: skip


def _concrete_estimators() -> set[type[Estimator]]:
    found: set[type[Estimator]] = set()
    stack = list(Estimator.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if inspect.isabstract(cls) or cls.__module__.startswith("tests"):
            continue
        found.add(cls)
    return found


def test_registry_lists_every_concrete_estimator():
    assert set(REGISTRY.values()) == _concrete_estimators()
    assert set(REGISTRY) == EXPECTED_NAMES
    for name, cls in REGISTRY.items():
        assert cls.info.name == name


def test_get_estimator_returns_the_class():
    assert get_estimator("gc") is REGISTRY["gc"]


def test_get_estimator_unknown_name_lists_known_names():
    with pytest.raises(KeyError) as excinfo:
        get_estimator("nope")
    message = str(excinfo.value)
    assert "nope" in message
    for name in EXPECTED_NAMES:
        assert name in message


def test_list_methods_sorted_by_name():
    infos = list_methods()
    assert [i.name for i in infos] == sorted(EXPECTED_NAMES)
    assert all(isinstance(i, MethodInfo) for i in infos)


def test_report_kind_is_set_on_every_method():
    for info in list_methods():
        assert info.report_kind in REPORT_KINDS
    kinds = {i.name: i.report_kind for i in list_methods()}
    assert kinds["trgc"] == "control"
    assert kinds["gc"] == "matrix"
    assert kinds["pdc"] == "spectrum"
    assert kinds["wdtf"] == "time_frequency"


def test_invalid_report_kind_is_rejected():
    with pytest.raises(ValueError, match="report_kind"):
        MethodInfo(name="x", backend="y", directed=True, report_kind="bogus")


def test_citations_are_normalised_to_a_tuple():
    info = MethodInfo(name="x", backend="y", directed=True, citations=["a", "b"])  # type: ignore[arg-type]
    assert info.citations == ("a", "b")


def test_every_method_cites_known_references():
    import json
    from importlib import resources

    text = (resources.files("dirconn.reports") / "_data" / "references.json").read_text("utf-8")
    refs = json.loads(text)
    for name, cls in REGISTRY.items():
        assert cls.info.citations, name
        for key in cls.info.citations:
            assert key in refs, (name, key)

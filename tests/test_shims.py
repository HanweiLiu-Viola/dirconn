"""Old module paths keep working (with a DeprecationWarning) after the subpackage move."""

from __future__ import annotations

import importlib
import re
import sys
import warnings
from pathlib import Path

import pytest

# old -> new; every restructure task adds its moved modules here.
MOVED = {
    "dirconn.timeseries": "dirconn.data.timeseries",
    "dirconn.channels": "dirconn.data.channels",
    "dirconn.network": "dirconn.data.network",
    "dirconn.result": "dirconn.data.result",
    "dirconn.provenance": "dirconn.data.provenance",
    "dirconn.bands": "dirconn.data.bands",
    "dirconn.rng": "dirconn.data.rng",
    "dirconn.exceptions": "dirconn.data.exceptions",
    "dirconn.bids": "dirconn.io.bids",
    "dirconn.estimator": "dirconn.methods.base",
    "dirconn.mvar": "dirconn.methods.mvar",
    "dirconn.granger": "dirconn.methods.granger",
    "dirconn.statespace": "dirconn.methods.statespace",
    "dirconn.spectral": "dirconn.methods.spectral",
    "dirconn.spectral_estimators": "dirconn.methods.spectral_estimators",
    "dirconn.multitaper_granger": "dirconn.methods.multitaper_granger",
    "dirconn.coherency": "dirconn.methods.coherency",
    "dirconn.time_reversal": "dirconn.methods.time_reversal",
    "dirconn.transfer_entropy": "dirconn.methods.transfer_entropy",
    "dirconn.pcmci": "dirconn.methods.pcmci",
    "dirconn.tvmvar": "dirconn.methods.timevarying.tvmvar",
    "dirconn.timevarying": "dirconn.methods.timevarying.timevarying",
    "dirconn.windowed": "dirconn.methods.timevarying.windowed",
    "dirconn.nonstationary": "dirconn.validation.nonstationary",
    "dirconn.surrogates": "dirconn.stats.surrogates",
    "dirconn.surrogate_test": "dirconn.stats.surrogate_test",
    "dirconn.significance": "dirconn.stats.significance",
    "dirconn.analytic": "dirconn.stats.analytic",
    "dirconn.bootstrap": "dirconn.stats.bootstrap",
    "dirconn.contrast": "dirconn.stats.contrast",
    "dirconn.correction": "dirconn.stats.correction",
    "dirconn.baseline": "dirconn.stats.baseline",
    "dirconn.order_stability": "dirconn.diagnostics.order_stability",
    "dirconn.leakage": "dirconn.diagnostics.leakage",
    "dirconn.regions": "dirconn.diagnostics.regions",
    "dirconn.scenario": "dirconn.simulation.scenario",
    "dirconn.headmodel": "dirconn.simulation.headmodel",
    "dirconn.inverse": "dirconn.simulation.inverse",
    "dirconn.emptyroom": "dirconn.simulation.emptyroom",
    "dirconn.protocol": "dirconn.validation.protocol",
    "dirconn.sweep": "dirconn.validation.sweep",
    "dirconn.benchmark": "dirconn.validation.benchmark",
    "dirconn.evaluation": "dirconn.validation.evaluation",
    "dirconn.noise_study": "dirconn.validation.noise_study",
    "dirconn.significance_study": "dirconn.validation.significance_study",
    "dirconn.report": "dirconn.validation.benchmark_report",
    "dirconn.plotting": "dirconn.viz",
}

SRC = Path(__file__).resolve().parents[1] / "src" / "dirconn"
TESTS = Path(__file__).resolve().parent


@pytest.mark.parametrize(("old", "new"), MOVED.items())
def test_old_path_warns_and_forwards(old, new):
    sys.modules.pop(old, None)
    with pytest.warns(DeprecationWarning, match="moved to"):
        mod = importlib.import_module(old)
    target = importlib.import_module(new)
    names = getattr(target, "__all__", None)
    if names is None:
        names = [n for n in dir(target) if not n.startswith("_")]
    assert names
    for name in names:
        assert getattr(mod, name) is getattr(target, name)


def test_no_internal_module_imports_a_shim():
    import dirconn

    last = [old.rsplit(".", 1)[-1] for old in MOVED]
    # A module that shares its name with a public function (``bootstrap``, ``pcmci``,
    # ``surrogate_test``) is only a shim import in the ``from dirconn.<m> import`` and
    # ``import dirconn.<m>`` forms; elsewhere ``dirconn.<m>`` and ``from dirconn import <m>``
    # name the function.
    plain = "|".join(n for n in last if n not in dirconn.__all__)
    shared = "|".join(n for n in last if n in dirconn.__all__)
    dotted = re.compile(
        r"\bdirconn\.(" + plain + r")\b|\b(?:from|import) dirconn\.(" + shared + r")\b"
    )
    from_import = re.compile(r"^\s*from dirconn import [^\n]*\b(" + plain + r")\b")
    skip = {SRC / f"{old.rsplit('.', 1)[-1]}.py" for old in MOVED}
    skip |= {SRC / "_compat.py", Path(__file__).resolve()}
    offenders = []
    for path in [*SRC.rglob("*.py"), *TESTS.rglob("*.py")]:
        if path in skip:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if dotted.search(line) or from_import.search(line):
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("name", ["bootstrap", "pcmci", "surrogate_test"])
def test_function_named_old_paths_keep_the_function(name):
    import dirconn

    sys.modules.pop(f"dirconn.{name}", None)
    with pytest.warns(DeprecationWarning, match="moved to"):
        shim = importlib.import_module(f"dirconn.{name}")
    assert callable(getattr(dirconn, name))  # still the function
    assert getattr(shim, name) is getattr(dirconn, name)  # the shim forwards to it


def test_datasets_alias_is_the_io_module():
    import dirconn
    import dirconn.io.datasets

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        import dirconn.datasets
    assert dirconn.datasets is dirconn.io.datasets


def test_scenarios_ship_with_simulation_subpackage():
    from dirconn.simulation.scenario import SCENARIO_DIR, list_scenarios

    assert SCENARIO_DIR.is_dir() and "chain" in list_scenarios()


def test_simulation_package_keeps_the_old_module_names_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from dirconn.simulation import NodeDynamics, simulate
    sim = importlib.import_module("dirconn.simulation.simulation")
    assert simulate is sim.simulate and NodeDynamics is sim.NodeDynamics


def test_validation_package_forwards_old_names_with_warning():
    import dirconn.validation as v

    target = importlib.import_module("dirconn.validation.method_validation")
    with pytest.warns(DeprecationWarning, match="moved to dirconn.validation.method_validation"):
        assert v.simulate_scenario is target.simulate_scenario
    with pytest.raises(AttributeError):
        v.no_such_name  # noqa: B018


def test_validation_submodules_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from dirconn.validation import method_validation, protocol

    assert protocol is importlib.import_module("dirconn.validation.protocol")
    assert method_validation is importlib.import_module("dirconn.validation.method_validation")


def test_viz_public_names_unchanged():
    import dirconn.viz as viz

    for name in [
        "dominant_networks",
        "yeo_networks",
        "yeo_groups",
        "group_matrix",
        "chord_diagram",
        "region_values",
        "degree_imbalance_map",
        "connectivity_grid",
        "directed_spectrum",
        "truth_comparison",
        "time_frequency",
        "network_graph",
        "plot",
    ]:
        assert callable(getattr(viz, name)), name
        assert name in viz.__all__
    assert isinstance(viz.PLOT_KINDS, tuple)
    for name in ["GUARD_CAPTION", "YEO_NETWORKS", "MIN_DOMINANT_SHARE"]:
        assert hasattr(viz, name), name
        assert not callable(getattr(viz, name)), name


def test_plot_dispatch_kinds():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from dirconn import Connectivity
    from dirconn.viz import PLOT_KINDS, plot

    conn = Connectivity.from_netcdf(TESTS / "data" / "connectivity_main_0e2b04e.nc")
    # the fixture is a (3, 3) result with no frequency or time axis, so directed_spectrum and
    # time_frequency (both need a frequency axis) cannot draw it and are skipped here
    kwargs = {
        "connectivity_grid": {},
        "network_graph": {"threshold": 0.0},
        "truth_comparison": {"truth": conn},  # a truth on the same nodes and axes
    }
    try:
        for kind in PLOT_KINDS:
            if kind in kwargs:
                assert plot(conn, kind, **kwargs[kind]) is not None
        with pytest.raises(ValueError, match="Unknown plot kind"):
            plot(conn, "no_such_kind")
    finally:
        plt.close("all")

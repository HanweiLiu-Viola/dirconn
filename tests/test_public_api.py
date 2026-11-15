import importlib
import warnings
from pathlib import Path

import dirconn
from dirconn import Connectivity

FIXTURE = Path(__file__).parent / "data" / "connectivity_main_0e2b04e.nc"

PUBLIC_NAMES = frozenset(
    [
        "ADTF",
        "AnalyticTestResult",
        "AssumptionCheck",
        "AssumptionChecklist",
        "AsymptoticTestResult",
        "BANDS",
        "BootstrapResult",
        "ChannelMeta",
        "Connectivity",
        "ContrastResult",
        "DTF",
        "DirconnWarning",
        "Edge",
        "Estimator",
        "GC",
        "GPDC",
        "ImCoh",
        "MTGC",
        "MethodInfo",
        "Network",
        "NodeDynamics",
        "OrderSelection",
        "PCMCI",
        "PCMCITestResult",
        "PDC",
        "PSI",
        "Provenance",
        "RPDC",
        "SSGC",
        "SimulatedData",
        "SimulationRecord",
        "SurrogateTestResult",
        "TE",
        "TPDC",
        "TRGC",
        "TimeReversalAsymmetry",
        "TimeSeries",
        "VarFit",
        "WDTF",
        "__version__",
        "adjust_pvalues",
        "adtf",
        "analytic_gc_test",
        "analytic_pdc_test",
        "analytic_rpdc_test",
        "baseline_normalise",
        "bootstrap",
        "contrast_test",
        "coupled_nonlinear",
        "datasets",
        "direct_node",
        "dk_node",
        "dtf",
        "fit_var",
        "gc",
        "gpdc",
        "imcoh",
        "mtgc",
        "pcmci",
        "pcmci_test",
        "pdc",
        "psi",
        "read_bids",
        "read_simulation_bids",
        "rpdc",
        "select_order",
        "significance_test",
        "simulate",
        "ssgc",
        "surrogate_test",
        "te",
        "time_reversal_asymmetry",
        "tpdc",
        "trgc",
        "viz",
        "wdtf",
        "write_roi_bids",
        "write_simulation_bids",
    ]
)


def test_all_unchanged():
    assert frozenset(dirconn.__all__) == PUBLIC_NAMES


def test_every_public_name_importable():
    for name in PUBLIC_NAMES:
        assert getattr(dirconn, name) is not None


def test_importing_dirconn_emits_no_deprecation_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.reload(dirconn)


def test_netcdf_written_before_restructure_loads():
    conn = Connectivity.from_netcdf(FIXTURE)
    assert conn.provenance.method == "gc"
    assert conn.values.shape[:2] == (3, 3)

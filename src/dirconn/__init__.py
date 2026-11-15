"""dirconn: directed functional connectivity for neurophysiology signals.

The core package depends on numpy, scipy, pandas, xarray, netcdf4, mne and mne-bids. MNE is
imported lazily by the modules that need it; tigramite, statsmodels, mne-connectivity and
plotting libraries are optional extras.
"""

import sys
import types
from importlib import metadata
from typing import Any

from dirconn import viz
from dirconn.data.bands import BANDS
from dirconn.data.channels import ChannelMeta
from dirconn.data.exceptions import DirconnWarning
from dirconn.data.network import Edge, Network, direct_node, dk_node
from dirconn.data.provenance import AssumptionCheck, AssumptionChecklist, Provenance
from dirconn.data.result import Connectivity
from dirconn.data.timeseries import TimeSeries
from dirconn.io import datasets
from dirconn.io.bids import read_bids, write_roi_bids
from dirconn.io.simulation_bids import (
    SimulationRecord,
    read_simulation_bids,
    write_simulation_bids,
)
from dirconn.methods.base import Estimator, MethodInfo
from dirconn.methods.coherency import PSI, ImCoh, imcoh, psi
from dirconn.methods.granger import GC, gc
from dirconn.methods.multitaper_granger import MTGC, mtgc
from dirconn.methods.mvar import OrderSelection, VarFit, fit_var, select_order
from dirconn.methods.pcmci import PCMCI, PCMCITestResult, pcmci, pcmci_test
from dirconn.methods.spectral_estimators import (
    DTF,
    GPDC,
    PDC,
    RPDC,
    SSGC,
    dtf,
    gpdc,
    pdc,
    rpdc,
    ssgc,
)
from dirconn.methods.time_reversal import TRGC, TimeReversalAsymmetry, time_reversal_asymmetry, trgc
from dirconn.methods.timevarying.timevarying import ADTF, TPDC, adtf, tpdc
from dirconn.methods.timevarying.windowed import WDTF, wdtf
from dirconn.methods.transfer_entropy import TE, te
from dirconn.simulation import NodeDynamics, SimulatedData, coupled_nonlinear, simulate
from dirconn.stats.analytic import (
    AnalyticTestResult,
    AsymptoticTestResult,
    analytic_gc_test,
    analytic_pdc_test,
    analytic_rpdc_test,
)
from dirconn.stats.baseline import baseline_normalise
from dirconn.stats.bootstrap import BootstrapResult, bootstrap
from dirconn.stats.contrast import ContrastResult, contrast_test
from dirconn.stats.correction import adjust_pvalues
from dirconn.stats.significance import significance_test
from dirconn.stats.surrogate_test import SurrogateTestResult, surrogate_test

# ``import dirconn.datasets`` must keep resolving to this module, without a deprecation warning.
# (Placed after the imports so that ruff E402 holds; nothing above imports ``dirconn.datasets``.)
sys.modules["dirconn.datasets"] = datasets

__version__: str = metadata.version("dirconn")

__all__ = [
    "ADTF",
    "BANDS",
    "DTF",
    "GPDC",
    "MTGC",
    "PCMCI",
    "PCMCITestResult",
    "PDC",
    "PSI",
    "RPDC",
    "SSGC",
    "TPDC",
    "TRGC",
    "WDTF",
    "AnalyticTestResult",
    "AsymptoticTestResult",
    "AssumptionCheck",
    "AssumptionChecklist",
    "BootstrapResult",
    "ChannelMeta",
    "ContrastResult",
    "Connectivity",
    "DirconnWarning",
    "Edge",
    "Estimator",
    "GC",
    "ImCoh",
    "MethodInfo",
    "Network",
    "NodeDynamics",
    "OrderSelection",
    "Provenance",
    "SimulatedData",
    "SimulationRecord",
    "SurrogateTestResult",
    "TE",
    "TimeReversalAsymmetry",
    "TimeSeries",
    "VarFit",
    "__version__",
    "datasets",
    "adjust_pvalues",
    "adtf",
    "analytic_gc_test",
    "analytic_pdc_test",
    "analytic_rpdc_test",
    "baseline_normalise",
    "bootstrap",
    "contrast_test",
    "coupled_nonlinear",
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


class _Package(types.ModuleType):
    """The package module: a deprecated submodule import may not replace a public function.

    Importing the old path ``dirconn`` + ``.bootstrap`` loads the shim and then binds it to
    ``dirconn.bootstrap``, which is the function; before the subpackage move the real
    submodule was imported during package initialisation and the function always won.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        current = self.__dict__.get(name)
        if isinstance(value, types.ModuleType) and name in __all__ and callable(current):
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _Package

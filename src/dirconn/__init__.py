"""dirconn: directed functional connectivity for neurophysiology signals.

The core package depends on numpy, scipy, pandas, xarray, netcdf4, mne and mne-bids. MNE is
imported lazily by the modules that need it; tigramite, statsmodels, mne-connectivity and
plotting libraries are optional extras.
"""

from importlib import metadata

from dirconn import datasets, viz
from dirconn.analytic import (
    AnalyticTestResult,
    AsymptoticTestResult,
    analytic_gc_test,
    analytic_pdc_test,
    analytic_rpdc_test,
)
from dirconn.bands import BANDS
from dirconn.baseline import baseline_normalise
from dirconn.bids import (
    SimulationRecord,
    read_bids,
    read_simulation_bids,
    write_roi_bids,
    write_simulation_bids,
)
from dirconn.bootstrap import BootstrapResult, bootstrap
from dirconn.channels import ChannelMeta
from dirconn.coherency import PSI, ImCoh, imcoh, psi
from dirconn.contrast import ContrastResult, contrast_test
from dirconn.correction import adjust_pvalues
from dirconn.estimator import Estimator, MethodInfo
from dirconn.exceptions import DirconnWarning
from dirconn.granger import GC, gc
from dirconn.multitaper_granger import MTGC, mtgc
from dirconn.mvar import OrderSelection, VarFit, fit_var, select_order
from dirconn.network import Edge, Network, direct_node, dk_node
from dirconn.pcmci import PCMCI, PCMCITestResult, pcmci, pcmci_test
from dirconn.provenance import AssumptionCheck, AssumptionChecklist, Provenance
from dirconn.result import Connectivity
from dirconn.significance import significance_test
from dirconn.simulation import NodeDynamics, SimulatedData, coupled_nonlinear, simulate
from dirconn.spectral_estimators import DTF, GPDC, PDC, RPDC, dtf, gpdc, pdc, rpdc
from dirconn.surrogate_test import SurrogateTestResult, surrogate_test
from dirconn.time_reversal import TRGC, TimeReversalAsymmetry, time_reversal_asymmetry, trgc
from dirconn.timeseries import TimeSeries
from dirconn.timevarying import ADTF, TPDC, adtf, tpdc
from dirconn.transfer_entropy import TE, te
from dirconn.windowed import WDTF, wdtf

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

API reference
=============

All names in this page are importable from ``dirconn`` unless a module is given.

Data structures
---------------

.. currentmodule:: dirconn

.. autosummary::
   :toctree: generated
   :nosignatures:

   TimeSeries
   ChannelMeta
   Connectivity
   Provenance
   AssumptionChecklist
   AssumptionCheck

Simulation
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Network
   Edge
   dk_node
   direct_node
   NodeDynamics
   simulate
   SimulatedData
   coupled_nonlinear

Estimators
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Estimator
   MethodInfo
   GC
   DTF
   PDC
   GPDC
   RPDC
   SSGC
   ADTF
   TPDC
   WDTF
   TRGC
   ImCoh
   PSI
   MTGC
   PCMCI
   TE
   fit_var
   select_order
   VarFit
   OrderSelection

Every estimator class has a lower-case shorthand that builds the estimator and fits it in one call:

.. autofunction:: dirconn.gc

.. autofunction:: dirconn.dtf

.. autofunction:: dirconn.pdc

.. autofunction:: dirconn.gpdc

.. autofunction:: dirconn.rpdc

.. autofunction:: dirconn.ssgc

.. autofunction:: dirconn.adtf

.. autofunction:: dirconn.tpdc

.. autofunction:: dirconn.wdtf

.. autofunction:: dirconn.trgc

.. autofunction:: dirconn.imcoh

.. autofunction:: dirconn.psi

.. autofunction:: dirconn.mtgc

.. autofunction:: dirconn.pcmci

.. autofunction:: dirconn.te

The state-space form behind ``GC(method="state_space")`` and :class:`~dirconn.SSGC`, for VAR
coefficients known in advance (for example the truth of a simulation):

.. autosummary::
   :toctree: generated
   :nosignatures:

   methods.statespace.var_to_state_space
   methods.statespace.reduced_model
   methods.statespace.state_space_gc
   methods.statespace.spectral_state_space_gc
   methods.statespace.InnovationsForm

A heuristic that compares any estimator with itself on time-reversed data (not an estimator, not a
test):

.. autosummary::
   :toctree: generated
   :nosignatures:

   time_reversal_asymmetry
   TimeReversalAsymmetry

Statistics
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   significance_test
   surrogate_test
   SurrogateTestResult
   analytic_gc_test
   AnalyticTestResult
   analytic_pdc_test
   analytic_rpdc_test
   AsymptoticTestResult
   pcmci_test
   PCMCITestResult
   bootstrap
   BootstrapResult
   contrast_test
   ContrastResult
   adjust_pvalues
   baseline_normalise

Figures
-------

``Connectivity.plot(kind=...)`` draws the result alone (matrices, spectra, panels and small graphs;
see :meth:`dirconn.Connectivity.plot`). The functions of :mod:`dirconn.viz` need more than a result:
a grouping of the nodes, or the cortical surface.

.. currentmodule:: dirconn.viz

.. autosummary::
   :toctree: generated
   :nosignatures:

   chord_diagram
   degree_imbalance_map
   group_matrix
   yeo_networks
   yeo_groups
   region_values
   dominant_networks

Files and data
--------------

.. currentmodule:: dirconn

.. autosummary::
   :toctree: generated
   :nosignatures:

   read_bids
   write_roi_bids
   read_simulation_bids
   write_simulation_bids
   SimulationRecord
   datasets.require_sample

Modules the tutorials use
-------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   simulation.scenario.load_scenario
   simulation.scenario.list_scenarios
   simulation.simulation.simulate_on_head
   simulation.headmodel.load_head_model
   simulation.inverse.roi_series
   simulation.inverse.crosstalk
   diagnostics.regions.centre_distances

Modules
-------

The package is split into eight subpackages of ``dirconn`` (see
:doc:`decisions/0002-package-layout`). The old top-level module paths (such as ``dirconn.granger``
or ``dirconn.plotting``) are deprecated shims that warn on import and are removed in dirconn 0.4.0.

.. currentmodule:: dirconn

.. autosummary::
   :toctree: generated

   data
   io
   methods
   stats
   diagnostics
   simulation
   validation
   viz

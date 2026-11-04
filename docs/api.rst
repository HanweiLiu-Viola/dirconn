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

.. autofunction:: dirconn.adtf

.. autofunction:: dirconn.tpdc

.. autofunction:: dirconn.wdtf

.. autofunction:: dirconn.trgc

.. autofunction:: dirconn.imcoh

.. autofunction:: dirconn.psi

.. autofunction:: dirconn.mtgc

.. autofunction:: dirconn.pcmci

.. autofunction:: dirconn.te

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

   scenario.load_scenario
   scenario.list_scenarios
   simulation.simulate_on_head
   headmodel.load_head_model
   inverse.roi_series
   inverse.crosstalk
   regions.centre_distances

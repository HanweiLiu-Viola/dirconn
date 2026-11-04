dirconn
=======

Directed functional connectivity for neurophysiology signals (EEG, MEG, iEEG, LFP) at sensor and
region level.

.. note::

   dirconn is alpha software. It estimates *predictive* directed influence (Granger causality and
   related measures). It does not establish interventional causality, and every result records the
   assumptions it relies on.

Where to start
--------------

* :doc:`auto_tutorials/index`: sixteen tutorials, from a first directed connectivity to the reading of
  sources and sinks, the controls for zero-delay mixing, model-free spectral Granger causality, PCMCI,
  transfer entropy (including one hypothesis on the EEG sources of a recording and the multivariate
  mode on an indirect path and a shared driver), three methods compared on the same simulated
  regions, and the renormalised PDC with its asymptotic threshold. Each ends on what its result does
  and does not show.
* :doc:`statistics` and :doc:`surrogates`: which test answers which question, and what each
  surrogate keeps.
* :doc:`api`: every public class and function.

Install
-------

.. code-block:: bash

   pip install dirconn                  # the package
   pip install "dirconn[plot,stats]"    # with plotting and the statsmodels-based tests
   pip install -e ".[dev,all]"          # from a clone, for development

Plots need the ``plot`` extra (matplotlib and, for the chord diagram of :mod:`dirconn.viz`,
pyCirclize); the map of the cortical surface needs ``plot-extra`` and MNE's 3D plotting.

Tutorials that need the MNE sample dataset
------------------------------------------

Tutorials 1 to 6, 9 to 12, 15 and 16 use simulated data only. Tutorials 7, 8, 13 and 14 need the MNE
``sample`` dataset (about 1.5 GB). dirconn **never downloads it without being asked**: the tutorials
call :func:`dirconn.datasets.require_sample`, which looks for the data locally and, if it is missing,
stops with the size, the ``MNE_DATA`` setting and the download command.

How the methods were validated
------------------------------

Every method is scored on simulated networks with a known truth. :doc:`method_validation` covers
small region-level networks (chain, common driver, collider, feedback loop, zero-lag coupling, hidden
driver, observation noise); :doc:`benchmark` describes the benchmark through the head models of the
MNE ``sample`` subject, :doc:`benchmark_results` holds what they do and do not show, written by hand,
in front of its tables, which are generated from the saved results. :doc:`timevarying` and
:doc:`nonstationary_validation` do the same for the time-varying methods.

.. toctree::
   :hidden:
   :maxdepth: 2

   auto_tutorials/index
   api

.. toctree::
   :hidden:
   :caption: Background
   :maxdepth: 1

   statistics
   surrogates
   timevarying
   method_validation
   nonstationary_validation
   benchmark
   Benchmark conclusions and results <benchmark_results>
   decisions/0001-bids-roi-layout

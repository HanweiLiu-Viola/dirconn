Outputs
=======

``dirconn-bids BIDS_DIR OUTPUT_DIR participant`` writes a BIDS derivatives dataset to
``OUTPUT_DIR/dirconn`` (layout version 1, :doc:`decisions/0003-derivatives-layout`) and keeps its
intermediate files in the work directory (``--work-dir``, default ``OUTPUT_DIR/work``).

The tree
--------

.. code-block:: text

   OUTPUT_DIR/
     dirconn/
       dataset_description.json
       dirconn.toml                                  configuration of the first run
       logs/
         run-20261117T094834-12184/                  one directory per run
           dirconn.log
           progress.jsonl
           crash-<task key>.txt                      one per failed task
           provenance.json
           environment-lock.txt
           dirconn.toml                              effective configuration of this run
       sub-01/
         eeg/
           sub-01_task-rest_run-1_desc-gc_conn.nc
           sub-01_task-rest_run-1_desc-gc_conn.json
           sub-01_task-rest_run-1_desc-gc_edges.tsv
           sub-01_task-rest_run-1_desc-trgc_conn.nc    controls.time_reversal = true
           sub-01_task-rest_run-1_desc-trgc_conn.json
           sub-01_task-rest_run-1_desc-trgc_edges.tsv
           figures/
             sub-01_task-rest_run-1_diagnostics.json
       sub-02/
         ses-01/                                     only when the input has a session
           eeg/
             ...
     work/
       prepared-<hash>.npz                           prepared-series cache
       prepared-<hash>.json

File names carry the entities of the input recording in the order ``sub ses task run``, then
``desc-<method>`` (the name in ``estimation.methods``, or ``trgc`` for the time-reversal control),
then the suffix. The data type directory is that of the input.

Dataset files
-------------

``dataset_description.json`` and ``dirconn.toml`` are written by the first run, when they do not exist
yet, and are never rewritten. The description has ``Name`` ``"dirconn derivatives"``,
``BIDSVersion`` ``"1.9.0"``, ``DatasetType`` ``"derivative"``, one ``GeneratedBy`` entry (``Name``
``"dirconn"``, ``Version``, ``CodeURL`` and, inside the Docker image, ``Container`` with ``Type``,
``Tag`` and ``Revision``) and ``"Dirconn": {"DerivativesVersion": "1"}``, which marks the dataset as a
connectivity output, so it is never taken for an input. ``dirconn.toml`` is the configuration that
later runs are checked against (see :doc:`usage`); it can be passed back with ``--config``.

One result
----------

Each method gives three files per recording.

``_conn.nc``
   The full result, a :class:`~dirconn.Connectivity` in NetCDF4 with the axes ``source``,
   ``target``, ``freq`` and ``time`` and its whole provenance (method, parameters, assumption checks,
   data history). It is the source of truth; read it with :meth:`dirconn.Connectivity.from_netcdf`.
   ``values[i, j]`` is the influence of node ``i`` on node ``j``. Excluded frequencies
   (``estimation.exclude_freqs``) are NaN.

``_conn.json``
   The sidecar, readable without dirconn:

   .. list-table::
      :header-rows: 1
      :widths: 25 75

      * - Field
        - Meaning
      * - ``Measure``
        - Method name recorded in the provenance.
      * - ``Directed``
        - Whether ``values[i, j]`` and ``values[j, i]`` differ in general.
      * - ``Estimator``
        - The ``desc`` entity.
      * - ``Parameters``
        - The estimator's parameters, as recorded in the provenance.
      * - ``NodeLabels``
        - Node names, in the order of both axes.
      * - ``NodeKinds``
        - Kind of each node (``cortical`` or ``direct``); present when the provenance records it.
      * - ``Bands``
        - Band name to ``[low, high]`` in Hz; empty for a method without frequency resolution.
      * - ``SamplingFrequency``
        - Present when the provenance records it.
      * - ``Description``
        - The orientation of the matrix, in words.
      * - ``Sources``
        - Input files, relative to ``BIDS_DIR`` (absolute when outside it).
      * - ``Statistics``
        - One entry per test: ``Band`` (``"broadband"`` for a method without bands), ``Test``
          (``"surrogate"`` or ``"analytic"``), ``Generator`` (``name``, the null hypothesis
          ``null`` and ``params``, for example the block size), ``NSurrogates``, ``Seed`` and
          ``ObservedHistory``, the data history of the observed value (with the block counts for
          block shuffling). Empty when ``statistics.test = "none"``. No arrays are stored here.
      * - ``ValueIsMagnitude``
        - Present for complex-valued measures: the edge table holds magnitudes.
      * - ``TaskKey``
        - Key of the task that wrote the result; a later run skips the task when it matches.
      * - ``DirconnVersion``, ``DerivativesVersion``
        - Version of dirconn that wrote the file, and of the layout (``"1"``).

``_edges.tsv``
   Written when ``output.long_table`` is on: one row per ordered pair of distinct nodes and band,
   tab-separated, with the columns

   =============== ==============================================================================
   ``source``      Node the influence comes from.
   ``target``      Node it goes to.
   ``band``        Band name; ``broadband`` for a method without frequency resolution.
   ``value``       The band value (magnitude for complex measures).
   ``p``           Raw p-value of the band's test.
   ``p_adj``       Corrected p-value (``statistics.correction``).
   ``significant`` ``True`` when ``p_adj < statistics.alpha``, else ``False``.
   =============== ==============================================================================

   The test columns are ``n/a`` without a test (``statistics.test = "none"``, and always for
   ``desc-trgc``). :func:`dirconn.io.derivatives.read_edges` reads the table with ``n/a`` as NaN. A
   frequency-resolved result without bands, or a time-resolved one, gives a table with the header
   only; the NetCDF file has the values.

The controls of a recording
---------------------------

The ``controls`` task of every recording writes

* ``figures/<recording>_diagnostics.json``: ``valid_fraction`` (share of valid samples after
  preparation), ``n_trials``, ``n_times``, ``sfreq``, ``channel_std`` (standard deviation of each
  channel over its valid samples, ``null`` for a channel without any), ``spectral`` (the spectral
  dynamic range per channel and band, ``null`` when ``controls.spectral_range_check`` is off),
  ``checks`` (the assumption checks drawn from it, with ``name``, ``status``, ``detail`` and
  ``value``) and ``task_key``;
* with ``controls.time_reversal``, the ``desc-trgc`` result (time-reversed Granger causality, no
  tests), written after the diagnostics.

The run directory
-----------------

Every run creates ``logs/run-<start time>-<pid>/``, the start time in UTC as ``YYYYmmddTHHMMSS``,
with ``-2``, ``-3``, ... appended when a run of the same process started in the same second. Runs
never share files, so jobs writing to the same output directory do not overwrite each other's logs.

``dirconn.log``
   The log of the run at debug level (stderr shows what ``-v`` asks for). Messages emitted inside
   worker processes (``--nprocs`` above 1) do not reach this file; the task outcomes are logged by the
   main process.
``progress.jsonl``
   One JSON object per task, appended when the task ends: ``key``, ``label`` (``<recording>:<method>``
   or ``<recording>:controls``), ``status`` (``done``, ``skipped`` or ``failed``), ``seconds``,
   ``crash`` (the crash file name, or ``null``) and ``finished`` (UTC time).
``crash-<key>.txt``
   For each failed task: its label, its payload and the traceback.
``provenance.json``
   Written when the run ends, with the keys

   .. list-table::
      :header-rows: 1
      :widths: 20 80

      * - Key
        - Content
      * - ``command``
        - The command line, as a list.
      * - ``started``, ``finished``
        - Start and end in ISO 8601, UTC.
      * - ``hostname``, ``platform``, ``python``
        - The machine, the operating system and the Python version.
      * - ``versions``
        - Versions of Python, dirconn and the installed scientific packages it records.
      * - ``image``
        - ``tag`` and ``revision`` of the Docker image (from ``DIRCONN_IMAGE_TAG`` and
          ``DIRCONN_BUILD_REVISION``), ``null`` outside it.
      * - ``git``
        - Commit of the dirconn source tree for a source install, otherwise ``null``.
      * - ``inputs``
        - One entry per recording: ``label``, ``files`` and their ``sha256``.
      * - ``config_hash``, ``analysis_hash``
        - Hashes of the configuration (see :doc:`configuration`).
      * - ``resources``
        - ``nprocs``, ``omp_nthreads``, ``mem_mb`` and ``work_dir``.
      * - ``tasks``
        - Counts ``done``, ``skipped`` and ``failed``.
``environment-lock.txt``
   The installed Python distributions as sorted ``name==version`` lines; inside the Docker image a
   copy of the image's ``/opt/environment-lock.txt``.
``dirconn.toml``
   The effective configuration of this run (``--config`` with the ``--set`` overrides applied),
   which may differ from the dataset's in its ``output`` and ``execution`` sections.

The work directory
------------------

``prepared-<hash>.npz`` and its ``.json`` sidecar hold a prepared record (data, masks, channels,
history), shared by the tasks of the recording within and across runs. The hash covers the input
files, the ``prepare`` section, ``input.node_kinds``, the dirconn version and the events file, and an
unreadable entry is recomputed. Nothing in the dataset refers to the work directory; it can be
deleted.

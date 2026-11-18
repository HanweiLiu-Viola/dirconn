Pipeline configuration
======================

What ``dirconn-bids`` computes is set by one TOML file, given with ``--config`` (see :doc:`usage`).
Every key is optional: an omitted key takes its default, and a file with only

.. code-block:: toml

   [estimation]
   methods = ["gc", "pdc"]

is a complete configuration. Unknown keys and sections are errors, and so are values of the wrong
type or out of range. ``dirconn-bids validate-config FILE`` checks a file without running anything
and prints its hash.

The file
--------

* **Sections.** ``[input]``, ``[prepare]`` (with ``[prepare.lock]``), ``[estimation]`` (with
  ``[estimation.bands]`` and one sub-table per method), ``[statistics]``, ``[controls]``, ``[output]``
  and ``[execution]``, plus the top-level ``schema_version``.
* **No null in TOML.** TOML has no null value, so the nullable fields (``prepare.resample_hz`` and
  ``input.description``) are switched off with the string ``"null"`` (any case), in the file or with
  ``--set key=null``. The configuration that dirconn writes back (``dirconn.toml`` in the output)
  uses the same string.
* **Per-method options.** A sub-table of ``[estimation]`` that is not one of its fields holds the
  keyword arguments of the estimator of that name:

  .. code-block:: toml

     [estimation]
     methods = ["gc"]

     [estimation.gc]
     method = "state_space"

  It is stored as ``estimation.params.gc.method``. Options may be given only for methods listed in
  ``estimation.methods``; an option the estimator does not have makes its task fail.
  ``dirconn-bids list-methods`` lists the method names, and :doc:`api` the options of each
  estimator. ``estimation.order`` and ``estimation.order_max`` are passed as ``order`` and
  ``max_order`` to the estimators that have these options, unless the sub-table sets them.
* **Checks across sections.** With ``prepare.resample_hz`` set, every band's upper edge must lie below
  half of it (the Nyquist frequency). ``statistics.test = "analytic"`` is refused when
  ``estimation.methods`` lists a frequency-resolved method (see ``statistics.test``).

Hashes
------

``config_hash`` (printed by ``validate-config`` and recorded in ``provenance.json``) is a digest of
the whole configuration. ``analysis_hash`` covers only the sections that change the numbers:
``prepare``, ``estimation``, ``statistics`` and ``controls``. It is part of every task key, and an
output directory refuses a run whose analysis hash or ``input`` section differ from the ones it was
created with; ``output`` and ``execution`` may change between runs (see :doc:`usage`).

Fields
------

Types are TOML types; a list of two numbers is written ``[4.0, 8.0]``.

``schema_version``
   Integer, default ``1``. Version of the configuration format; only 1 is accepted.

``[input]``: what is read
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Field
     - Type
     - Default
     - Meaning
   * - ``input.level``
     - ``"roi"`` or ``"sensor"``
     - ``"roi"``
     - ``"roi"`` reads the region-level series of the dirconn derivatives below
       ``BIDS_DIR/derivatives``; ``"sensor"`` reads the raw recordings.
   * - ``input.datatypes``
     - list of strings
     - ``["meg", "eeg", "ieeg"]``
     - BIDS data types searched.
   * - ``input.node_kinds``
     - list of strings
     - ``["cortical", "direct"]``
     - Kinds of region-level nodes kept (the ``node_kind`` column of ``channels.tsv``; ``direct``
       is a recorded channel such as an LFP contact). At least two nodes must remain. Ignored at
       ``level = "sensor"``.
   * - ``input.description``
     - string or ``"null"``
     - ``"DK"``
     - The ``desc`` entity of the region-level files (the parcellation); ``"null"`` accepts any.
       Ignored at ``level = "sensor"``.

``[prepare]``: preparation before estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The steps run in this order: padding of the bad segments, resampling, z-score, event lock. Each is
recorded in the data history that every result carries in its provenance.

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Field
     - Type
     - Default
     - Meaning
   * - ``prepare.resample_hz``
     - number > 0, or ``"null"``
     - ``250.0``
     - Target sampling rate in Hz; ``"null"`` keeps the recorded rate, and so does a rate equal to
       it. The ratio to the recorded rate must be exactly representable as a fraction with a
       denominator up to 10000, otherwise the task fails and names the nearest possible rate.
       After resampling, the samples within the filter's half-length of a bad segment are
       invalid. The filter transients at both ends of the record (and of every epoch) are cut
       off rather than masked: ``guard_samples`` samples go from each end, ``tmin`` moves later by
       the same time, and the times of the remaining samples are unchanged (10 samples, 40 ms, when
       downsampling to 250 Hz). A record without bad segments therefore carries no mask after
       resampling. Data that is already epoched gets shorter epochs: -1 to 1 s at 1000 Hz becomes
       -0.96 to 0.96 s at 250 Hz. The resample entry of the data history records
       ``guard_samples`` and the new ``tmin``.
   * - ``prepare.zscore``
     - boolean
     - ``true``
     - Standardise each channel over its valid samples. A channel with zero variance makes the
       task fail.
   * - ``prepare.bad_segments``
     - ``"mask"`` or ``"ignore"``
     - ``"mask"``
     - ``"mask"`` leaves the ``BAD_`` annotations (and the resampling guards around them) out of the
       estimate through the validity mask (see :doc:`masks`); methods without mask support then refuse the
       record (``supports_mask`` in ``list-methods``). ``"ignore"`` drops the mask after
       resampling, so every method runs and the bad samples enter the estimate: at
       ``input.level = "sensor"`` the recording is read without the ``BAD_`` mask, so these are
       the samples as recorded. A region-level file stores its bad samples as NaN, and the
       recorded values cannot be recovered from it; in ``"ignore"`` mode they enter the estimate
       as zeros.
   * - ``prepare.pad_bad_seconds``
     - number >= 0
     - ``0.0``
     - Extra seconds masked on both sides of every bad segment (``"mask"`` mode only).
   * - ``prepare.lock.events``
     - list of strings
     - ``[]``
     - ``trial_type`` values of the recording's ``_events.tsv`` to cut epochs around; empty keeps the
       record continuous. Epochs that cross the record edges or contain invalid samples are dropped.
       At ``input.level = "roi"`` the events come from the ``_events.tsv`` next to the
       region-level file, which :func:`~dirconn.io.write_roi_bids` writes only for epoched records
       (one ``epoch`` event per epoch); a continuous region-level record has none unless one is
       added.
   * - ``prepare.lock.tmin``
     - number
     - ``-1.0``
     - Start of the epoch relative to the event, in seconds.
   * - ``prepare.lock.tmax``
     - number
     - ``1.0``
     - End of the epoch relative to the event, in seconds; must be above ``tmin``.

``[estimation]``: connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Field
     - Type
     - Default
     - Meaning
   * - ``estimation.methods``
     - list of strings, not empty
     - ``["gc"]``
     - Methods to run, by their registry names (``dirconn-bids list-methods``). Each gives one task
       and one ``desc-<method>`` result per recording. Control estimators (``trgc``, report kind
       ``control``) are refused here: the time-reversal control runs through
       ``controls.time_reversal``.
   * - ``estimation.order``
     - integer >= 1, or ``"aic"``, ``"bic"``, ``"hqc"``, ``"cv"``
     - ``"bic"``
     - Model order of the model-based methods, or the criterion that selects it.
   * - ``estimation.order_max``
     - integer >= 1
     - ``30``
     - Largest order a criterion may select.
   * - ``estimation.bands.<name>``
     - ``[low, high]`` in Hz, ``0 <= low < high``
     - ``theta = [4.0, 8.0]``, ``beta_low = [13.0, 20.0]``, ``beta_high = [20.0, 30.0]``
     - Named frequency bands; at least one. A frequency-resolved result is reduced to one value
       per band and tested per band. Giving ``[estimation.bands]`` replaces the default bands.
   * - ``estimation.reduce``
     - ``"mean"``, ``"median"``, ``"max"`` or ``"peak_freq"``
     - ``"mean"``
     - How a spectrum is reduced to the value of a band.
   * - ``estimation.exclude_freqs``
     - list of ``[low, high]`` in Hz
     - ``[]``
     - Frequency ranges (for example line noise, ``[[49.0, 51.0]]``) set to NaN in the spectrum
       before the band reduction.
   * - ``estimation.params.<method>.<key>``
     - any TOML value
     - none
     - Keyword argument ``<key>`` of the estimator ``<method>``, written as the sub-table
       ``[estimation.<method>]`` (see above).

``[statistics]``: significance tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every test is corrected over the pairs of its band; the choice of test and surrogate is discussed in
:doc:`statistics` and :doc:`surrogates`.

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Field
     - Type
     - Default
     - Meaning
   * - ``statistics.test``
     - ``"surrogate"``, ``"analytic"`` or ``"none"``
     - ``"surrogate"``
     - ``"surrogate"`` tests every band (or the single value of a method without frequency
       resolution) against surrogates; ``"analytic"`` uses the estimator's asymptotic test (a task
       fails for estimators without one); ``"none"`` runs no test (the p-value columns of the
       edge table are ``n/a``). The pipeline runs the analytic test only for methods without
       frequency resolution (one test, no band): a configuration that combines it with a
       frequency-resolved method is refused.
   * - ``statistics.surrogate``
     - ``"auto"`` or a generator name
     - ``"auto"``
     - ``"auto"`` is ``block_shuffle`` for continuous records and ``trial_shuffle`` for epoched
       ones, the two generators that carry masks. Otherwise one of ``phase_randomize``, ``aaft``,
       ``iaaft``, ``circular_shift``, ``trial_shuffle``, ``block_shuffle``. With block shuffling the
       observed value of the test is computed on the blocks, so it differs a little from the
       stored estimate on the unbroken record.
   * - ``statistics.block_seconds``
     - number > 0
     - ``5.0``
     - Block length of ``block_shuffle`` in seconds. Blocks are cut inside every run of valid
       samples; the numbers of blocks kept are recorded with the test.
   * - ``statistics.n_surrogates``
     - integer >= 1
     - ``200``
     - Number of surrogates per test.
   * - ``statistics.alpha``
     - number in (0, 1)
     - ``0.05``
     - Significance level applied to the corrected p-values.
   * - ``statistics.correction``
     - ``"fdr_bh"``, ``"bonferroni"`` or ``"none"``
     - ``"fdr_bh"``
     - Multiple-comparison correction over the pairs of a band.

``[controls]``: control analyses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run once per recording by its ``controls`` task.

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Field
     - Type
     - Default
     - Meaning
   * - ``controls.time_reversal``
     - boolean
     - ``true``
     - Also compute time-reversed Granger causality (TRGC, with ``estimation.order`` and
       ``estimation.order_max``) and write it as ``desc-trgc``, without tests.
   * - ``controls.spectral_range_check``
     - boolean
     - ``true``
     - Measure the spectral dynamic range of each channel across the bands and record it, with an
       assumption check, in the diagnostics file of the recording.

``[output]``
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Field
     - Type
     - Default
     - Meaning
   * - ``output.long_table``
     - boolean
     - ``true``
     - Write the long ``_edges.tsv`` table next to every result (see :doc:`outputs`).

``[execution]``
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Field
     - Type
     - Default
     - Meaning
   * - ``execution.seed``
     - integer
     - ``20261115``
     - Base seed of the significance tests. Each task derives its own seed from it and its task
       key, and band ``i`` of a task uses that seed plus ``i``; the seeds are recorded in the
       sidecars. It is not part of the task keys: changing it does not redo finished tasks.
   * - ``execution.fail_fast``
     - boolean
     - ``false``
     - Stop at the first failing task instead of recording it and going on. With ``--nprocs`` above
       1 the tasks run in batches of that size and the run stops after the batch with the failure.

The default file
----------------

The built-in default (``src/dirconn/workflows/_data/default_config.toml``), with a comment on
every key:

.. code-block:: toml

   # dirconn participant pipeline: default configuration.
   # Every key is optional in a user file; omitted keys take the values below.
   # TOML has no null: write the string "null" (or `--set key=null`) to turn a nullable option
   # (prepare.resample_hz, input.description) off.

   schema_version = 1

   [input]
   level = "roi"                          # "roi" (parcellated sources) or "sensor"
   datatypes = ["meg", "eeg", "ieeg"]     # BIDS data types to include
   node_kinds = ["cortical", "direct"]    # kinds of nodes to keep
   description = "DK"                     # parcellation (BIDS desc entity)

   [prepare]
   resample_hz = 250.0                    # target sampling rate; every band edge must be < half of it
   zscore = true                          # standardise each channel
   bad_segments = "mask"                  # "mask" bad annotations out of the estimate, or "ignore"
   pad_bad_seconds = 0.0                  # extra seconds masked around each bad segment

   [prepare.lock]
   events = []                            # event names to lock to; empty = continuous data
   tmin = -1.0                            # window start relative to the event, seconds
   tmax = 1.0                             # window end relative to the event, seconds

   [estimation]
   methods = ["gc"]                       # names from the method registry
   order = "bic"                          # integer, or "aic" / "bic" / "hqc" / "cv"
   order_max = 30                         # upper bound for criterion-based order selection
   reduce = "mean"                        # band reduction: "mean", "median", "max", "peak_freq"
   exclude_freqs = []                     # [[lo, hi], ...] ranges left out of band reduction

   # Per-method options go in sub-tables named after the method, for example:
   # [estimation.gc]
   # method = "state_space"

   [estimation.bands]                     # named bands in Hz, lo < hi
   theta = [4.0, 8.0]
   beta_low = [13.0, 20.0]
   beta_high = [20.0, 30.0]

   [statistics]
   test = "surrogate"                     # "surrogate", "analytic" or "none"
   surrogate = "auto"                     # auto = block_shuffle (continuous) / trial_shuffle (epoched);
                                          # or phase_randomize, aaft, iaaft, circular_shift, trial_shuffle, block_shuffle
   block_seconds = 5.0                    # block length of block_shuffle, in seconds
   n_surrogates = 200
   alpha = 0.05
   correction = "fdr_bh"                  # "fdr_bh", "bonferroni" or "none"

   [controls]
   time_reversal = true                   # also compute time-reversed controls
   spectral_range_check = true            # check bands against the usable spectral range

   [output]
   long_table = true                      # write the long-format results table

   [execution]
   seed = 20261115
   fail_fast = false                      # stop at the first failing task

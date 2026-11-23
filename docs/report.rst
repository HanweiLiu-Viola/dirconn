The participant report
======================

After every participant run, ``dirconn-bids`` writes one page for each subject of the run,
``OUTPUT_DIR/dirconn/sub-XX.html``, and the methods text ``OUTPUT_DIR/dirconn/logs/CITATION.*``.
The report puts what the pipeline wrote for a subject (inputs, diagnostics, results, tests, failures)
in one place, so that a problem is noticed and a result can be traced to the files and settings
behind it. It is a reading aid. It does not decide whether the assumptions of a method hold for your
recording, and it never changes a result: the report is built from the derivatives that are already
on disk and computes no connectivity. :doc:`auto_tutorials/plot_18_bids_app_report` reads one report
step by step; the design is recorded in :doc:`decisions/0004-report-spec`.

When it is written
------------------

* At the end of ``dirconn-bids BIDS_DIR OUTPUT_DIR participant``, once the tasks have finished (exit
  code 0 or 1), for every subject that had a recording in the run, and then the methods text of the
  whole output dataset. A run that skipped every task (a repeat) writes them again.
* ``dirconn-bids BIDS_DIR OUTPUT_DIR participant --reports-only`` writes them again from the files in
  ``OUTPUT_DIR/dirconn`` without reading the BIDS dataset or the configuration and without computing
  anything. Use it after updating dirconn, or to rebuild a report that failed.
* ``--boilerplate-only`` writes only ``logs/CITATION.*``. See :doc:`usage` for both options.

A subject's report covers every recording of that subject in the output dataset, including those of
earlier runs. If a report cannot be written, the command prints the reason, leaves
``crash-report-<subject>.txt`` (or ``crash-report-citation.txt``) in the run directory (with
``--reports-only``: in ``logs/reports-<UTC time>-<pid>/``), exits with code 1 and goes on with the
other subjects. The task results are never touched by a report failure.

The page
--------

``sub-XX.html`` is one self-contained file: the style sheet and the figures (SVG) are inside it, it
contains no script, and it loads nothing from the network, so it can be copied, mailed or archived on
its own. The only links that leave the page are the DOI links in the reference list of the Methods
section. The time-frequency maps of time-resolved methods are drawn as raster images inside their
SVG, which keeps a report small (a test keeps the report of the three-recording test dataset below
5 MB); axes and labels stay vector. A table with more than 50 rows is folded into a ``<details>``
element.

The top of the page shows the overall status and a table of contents with the status of each section.
Sections that hold results of recordings (Input, Diagnostics, Connectivity, Controls, Statistics) list
the recordings one after the other, each under its label such as ``sub-01_task-rest_run-1``.

The eight sections
------------------

Each section answers one question. The reportlets named below are the figures and tables as the report
builds them; ``<method>`` is a name from ``estimation.methods`` and ``<band>`` a band name.

Summary
   *Which recordings does this report cover, and what produced them?* ``summary/overview``: one row per
   recording with its input files, the SHA-256 over them, the methods and bands that have results,
   the start and end of the run that wrote them, the dirconn version, the Docker image tag and
   revision (``not recorded`` outside the image) and the git commit of a source install. The status is
   the worst of every reportlet of the subject.

Input
   *What was analysed?* ``prep/nodes`` (node names and kinds) and ``prep/record`` (sampling
   frequency, number of trials and samples, the share of valid samples after preparation, the
   preparation steps, and the standard deviation of each channel after preparation; the spread before
   the z-score is not recorded).

Diagnostics
   *Does the recording suit the models that were fitted?* ``diag/assumptions`` lists every assumption
   check of every result of the recording (stationarity, residual whiteness, sampling adequacy, model
   order selection and so on, as each method records them) and the checks drawn from the recording
   itself, with their status and detail text. ``diag/order`` is the selection of the VAR order: the
   curves of the information criteria and the selected order. The order is chosen once per recording,
   so there is one figure per recording; with a fixed ``estimation.order`` it is a one-line note.
   ``diag/psd`` is the power spectral density of the prepared channels with the analysed bands, shown
   when ``controls.spectral_range_check`` is on.

Connectivity
   *What did each method find?* For a method with an edge table: ``<method>/matrix-<band>``, the band
   value of every ordered pair as a matrix with significant edges marked; and, for a directed method,
   ``<method>/net-<band>``, the net direction ``value(i to j) - value(j to i)`` of the node pairs of
   different kinds (it needs at least two kinds, for example cortical and deep nodes). A method whose
   ``report_kind`` is ``spectrum`` (see ``dirconn-bids list-methods``) also gets
   ``<method>/spectra`` when its result has at least three frequencies: both directions of up to six
   node pairs over frequency. A method whose ``report_kind`` is ``time_frequency`` (``adtf``,
   ``tpdc``, ``wdtf``) gets ``<method>/tf``, maps for up to six node pairs, instead of the
   matrices. The pairs are chosen by the rule in the caption:
   significant pairs first, then the largest band values.

Controls
   *Do time-reversed and ordinary Granger causality agree on the direction?* ``trgc/matrix`` is the
   net time-reversed Granger causality, averaged over frequency. ``trgc/agreement`` compares its sign
   with the sign of ``gc(i to j) - gc(j to i)`` for every pair (it needs a ``gc`` result and the
   ``broadband`` rows of its edge table). The section is empty when ``controls.time_reversal`` is off.

Statistics
   *How was each result tested, and what is significant?* ``<method>/tests`` lists the tests that were
   run: band, test, generator, block size, number of surrogates, seed and the shape of the p-value
   array (a built-in test shows its source sentence instead of a generator). ``<method>/significant``
   lists the significant rows ordered by adjusted p value and then by value, at most 200;
   ``<method>/pvalues`` is the histogram of the p values; ``<method>/sig-time`` is the number of
   significant pairs over time, for time-resolved results that were tested. Results that were not
   tested are described under "Described, not tested" below.

Methods
   *What was done, in words?* The methods paragraphs of the subject, written from the configuration,
   the model orders that were selected, the parameters dirconn filled in and the tests and controls,
   followed by the references with their DOI links. It is the text of ``logs/CITATION.md`` restricted
   to this subject's results; the files in ``logs/`` cover the whole dataset (see below). It has no
   status.

Errors
   *What went wrong, and what is missing?* See "The Errors section" below.

Statuses
--------

Every reportlet has one of four statuses. A section shows the worst status of its reportlets and the
page the worst of the subject, ordered ``FAIL`` > ``WARN`` > ``PASS`` > ``INFO``.

``PASS``
   A check ran and found nothing to flag.
``WARN``
   A check ran and crossed a rule of thumb. The result is still there; the caption or table says which
   rule. A warning is a reason to look, not a verdict.
``FAIL``
   A check found the model unusable (an unstable VAR model), or a task or a reportlet could not be
   made.
``INFO``
   Not a check (a matrix, a table of the tests that were run), or a check that was not run
   (``not checked``).

The report assigns them as follows.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Reportlet
     - Status
   * - ``prep/record``
     - ``WARN`` when fewer than half of the samples are valid, else ``PASS``.
   * - ``diag/assumptions``
     - The worst status of its rows; a check that was not run counts as ``INFO``.
   * - ``diag/order``
     - ``WARN`` when the selected order is the largest candidate order (a higher order may fit better),
       else ``PASS``; ``INFO`` when the order is fixed.
   * - ``diag/psd``
     - The status of the spectral check of the recording (``WARN`` above the dynamic-range threshold).
   * - ``trgc/agreement``
     - ``WARN`` when at least one pair disagrees in sign, else ``PASS``. A pair counts as
       disagreeing even when one of the two values is tiny.
   * - ``summary/errors``
     - ``FAIL`` when a task failed or a reportlet could not be built; ``INFO`` when it only lists
       reportlets that are absent for an expected reason; ``PASS`` when it is empty.
   * - ``summary/overview``
     - The worst status of every other reportlet of the subject.
   * - All others
     - ``INFO``.

The assumption checks inside ``diag/assumptions`` and ``diag/psd`` come from the estimators and
diagnostics. Their rules are, as of this version: more than one VAR coefficient per ten observations
(sampling adequacy, ``WARN``); a spectral radius of the fitted VAR model of 0.99 or more (``WARN``) or
1 or more (``FAIL``); residual autocorrelation that is significant (p < 0.05) and above 0.1 in
absolute value (residual whiteness, ``WARN``); a spectral dynamic range above 60 dB (``WARN``); a
selected order at the largest candidate order (``WARN``). **These thresholds are rules of thumb that
have not been calibrated** for these methods or for your data. A recording can pass all of them and
still be unsuitable, and one that warns can be fine. The report never stops a run and never removes a
result because of a status.

Described, not tested
---------------------

A time-resolved result (``adtf``, ``tpdc``, ``wdtf``) on a *continuous* recording is not tested. The
default test, ``statistics.surrogate = "auto"``, shuffles blocks of 5 s and fits the observed value on
them, which leaves no time points of the result to test. The Statistics section then shows the reason
instead of a table: the reason of the sidecar's ``StatisticsNote`` (it starts "not tested: block
shuffling fits the observed value on blocks of the record ... the result is descriptive"), with status
``INFO``; there is no p-value histogram and no list of significant edges. This is expected
and is not an error: the edge table of the result has ``n/a`` in its test columns and its sidecar has
a ``StatisticsNote`` (see :doc:`outputs`). Read such a result as a description of how the measure
changes over time. On epoched data every time point is tested against shuffled trials, and the
``sig-time`` figure shows the number of significant pairs over time.

The Errors section
------------------

``summary/errors`` has one row per problem, with the kind, the recording label, the reportlet
(``<desc>/<name>``, where ``<method>/*`` stands for all reportlets of a method), the reason, and the
run directory and crash file (relative to ``OUTPUT_DIR/dirconn``) of a failed task.

``task failed``
   The last progress line of the task in the run logs says it failed; ``crash_file`` is the traceback.
   A task that failed once and was done by a later run is not listed.
``missing reportlet``
   A reportlet that the report expects is absent. The reason is one of: ``task failed`` or ``not run`` (a method of ``estimation.methods`` without a result);
   ``time_reversal is off``; ``controls task failed`` or ``controls task failed or not run`` (no
   time-reversal result); ``spectral_range_check is off`` or ``spectral diagnostics not recorded``;
   ``diagnostics missing (controls task failed or not run)``; or ``not produced`` (found while the
   page was assembled, when nothing else explains it).
``build failed``
   A reportlet raised while it was built. A placeholder with status ``FAIL`` and the reason in its
   caption takes its place, so one broken figure does not stop the report.
``boilerplate failed``
   The methods text of the subject could not be made; the Methods section says so.

A reportlet that is absent because the option behind it is off is not a failure: such rows give the
section status ``INFO``. A time-resolved result that was not tested is not listed either.

Files
-----

Reportlets are ordinary files, written when the report is built and replaced each time. Per
recording, next to its diagnostics file::

   sub-01/eeg/figures/sub-01_task-rest_run-1_desc-<desc>_<name>.json
   sub-01/eeg/figures/sub-01_task-rest_run-1_desc-<desc>_<name>.svg      only when it has a figure

with ``desc`` a method name or ``prep``, ``diag`` or ``trgc``, and ``name`` as above
(``matrix-alpha``, ``order``, ``tests`` and so on). The subject-level reportlets (``overview`` and
``errors``, ``desc-summary``) are ``sub-01/figures/sub-01_desc-summary_overview.json`` and
``..._errors.json``. The JSON holds ``name``, ``section``, ``desc``, ``title``, ``caption``,
``status``, ``table`` (rows as plain values) and ``data``; the SVG is the figure. They can be read
without dirconn, or with ``dirconn.reports.reportlet.read_reportlets``.

The methods text is in ``logs/CITATION.html``, ``.md``, ``.tex`` and ``.bib``: the same paragraphs and
references in four formats, covering every subject of the dataset. Its sentence on the model
order summarises all recordings (range and median of the selected orders). Its last paragraph says
that the measures are predictive, not interventional.

Reading a report
----------------

* Start at the top: the overall status and the section badges. ``FAIL`` in Errors means a task did
  not finish; ``WARN`` in Diagnostics means a rule of thumb was crossed, and the table names it.
* Check the Input section before the results: how many samples were valid and which nodes were
  analysed.
* Read the order figure: a selected order at the edge of the search (``WARN``) means the search
  range may be too small. ``estimation.order_max`` sets the range.
* Read the Statistics section before the Connectivity figures: which test, how many surrogates and
  which results were not tested. With few surrogates a corrected test cannot reject anything (the
  smallest p-value is ``1 / (n_surrogates + 1)``).
* Trust nothing in the report more than the assumptions of the method it shows: every figure is a
  predictive measure, not an interventional one.

Command line
============

``dirconn-bids`` runs the dirconn pipeline on a BIDS dataset, following the BIDS App convention::

   dirconn-bids BIDS_DIR OUTPUT_DIR participant [options]

It reads the region-level (or sensor-level) time series of every selected recording, prepares them,
estimates directed connectivity with the configured methods, tests it, runs the controls and writes a
derivatives dataset to ``OUTPUT_DIR/dirconn`` (see :doc:`outputs`). What is computed is set by a TOML
file (see :doc:`configuration`); the command line selects the inputs and the resources.

Only the participant level exists in dirconn 0.2.0; the group level arrives in dirconn 0.3.0 (until
then it stops with exit code 2 and the message "available in dirconn 0.3.0"). After every participant
run the command writes a report page ``sub-XX.html`` for each subject of the run and the methods
boilerplate ``logs/CITATION.*`` into the output dataset (see :doc:`report`). ``--reports-only`` and
``--boilerplate-only`` write them again from existing outputs without computing anything.

Install
-------

.. code-block:: bash

   pip install "dirconn[app]"

The ``app`` extra adds pydantic, joblib and (on Python 3.10) tomli. Without it, ``dirconn-bids
--version``, ``dirconn-bids --help`` and ``dirconn-bids list-methods`` still work; every other command
stops with exit code 2 and the hint ``pip install "dirconn[app]"``.

A first run
-----------

.. code-block:: bash

   dirconn-bids validate-config my_config.toml        # prints OK <config hash>
   dirconn-bids /data/bids /data/out participant \
       --participant-label 01 02 --config my_config.toml --nprocs 4 -v

When it finishes, the command prints one line to stderr::

   dirconn-bids: done 12, skipped 0, failed 0; run directory /data/out/dirconn/logs/run-20261117T094834-12184

Each recording (a *unit*, one ``sub``/``ses``/``task``/``run`` combination) gives one task per
method in ``estimation.methods`` plus one ``controls`` task (diagnostics and, when switched on, the
time-reversal control), so a run has *units* × (*methods* + 1) tasks. A failing method never takes the
controls or the other methods of its unit down with it.

Positional arguments
--------------------

``BIDS_DIR``
   Root of the BIDS dataset. At ``input.level = "roi"`` (the default) the inputs are read from the one
   directory below ``BIDS_DIR/derivatives`` whose ``dataset_description.json`` lists ``GeneratedBy``
   ``Name == "dirconn"`` (the region-level series of :doc:`decisions/0001-bids-roi-layout`);
   connectivity outputs of dirconn and the output dataset of the run itself are skipped. At
   ``input.level = "sensor"`` the raw recordings of ``BIDS_DIR`` are read, and anything below
   ``BIDS_DIR/derivatives`` is skipped.

``OUTPUT_DIR``
   The results go to ``OUTPUT_DIR/dirconn``. ``OUTPUT_DIR`` may not be ``BIDS_DIR`` itself nor lie
   inside it, except below ``BIDS_DIR/derivatives``: ``dirconn-bids /data/bids /data/bids/derivatives
   participant`` writes ``/data/bids/derivatives/dirconn`` and can be run again on the same dataset.

``participant`` | ``group``
   The analysis level; ``group`` is not available yet (exit code 2).

A BIDS directory whose name is a subcommand (``validate-config`` or ``list-methods``) must be written
with a path prefix, for example ``./list-methods``, or it is read as the subcommand.

Selecting the inputs
--------------------

``--participant-label LABEL [LABEL ...]``
   Subjects to process, with or without ``sub-``; default all.
``--session-id ID [ID ...]``
   Sessions, with or without ``ses-``.
``--task-id ID [ID ...]``
   Tasks, with or without ``task-``.
``--run-id ID [ID ...]``
   Runs, with or without ``run-``.
``--bids-filter-file FILE``
   A JSON object whose keys are among ``subject``, ``session``, ``task`` and ``run``, each holding a
   list of values (prefixes allowed)::

      {"subject": ["01", "02"], "task": ["rest"]}

   Unknown keys and values that are not lists are errors. The file is **intersected** with the other
   filters: a recording must pass both, so ``--participant-label 01`` with a file listing subjects
   ``02`` and ``03`` selects nothing (and stops with exit code 2, "No recordings found"). Some BIDS
   Apps let the file replace the command-line filters; dirconn does not.

A recording with a session is skipped by ``--session-id`` filters that do not list it; one without a
session passes no ``--session-id`` filter. Several data files for one recording (split files) are an
error. Recordings are processed in the order (subject, session, task, run).

Configuration
-------------

``--config FILE``
   The TOML configuration; without it, the built-in default (see :doc:`configuration`). Unknown keys
   are errors.
``--set KEY=VALUE``
   Override one field by its dotted path, after the file is read; may be repeated, and the overrides
   are applied in order before the whole configuration is validated again. The value is split on the
   first ``=``:

   * a field holding a list takes comma-separated values (``estimation.methods=gc,pdc``,
     ``prepare.lock.events=go,stop``); an empty value gives an empty list
     (``estimation.exclude_freqs=``). An existing band can be changed this way
     (``estimation.bands.theta=4,7``); a new band, or ``exclude_freqs`` ranges, must be written in the
     TOML file;
   * other fields take the text, converted and checked by the configuration model
     (``estimation.order=5``, ``prepare.zscore=false``, ``statistics.alpha=0.01``);
   * ``null`` (any case) turns a nullable field off (``prepare.resample_hz=null``,
     ``input.description=null``);
   * ``estimation.params.<method>.<key>=value`` sets a keyword argument of an estimator and creates
     the table if needed; the value is read as ``null``, ``true``/``false``, an integer, a float, or
     else kept as text (lists cannot be given this way). The method must be in
     ``estimation.methods``.

   A malformed override, an unknown path or a path that names a whole section stops the command with
   exit code 2 before anything runs; the first such override is reported. Invalid values are reported
   one line per problem with their dotted path.

Resources
---------

``--work-dir DIR``
   Intermediate files; default ``OUTPUT_DIR/work``. It holds the prepared-series cache
   (``prepared-<hash>.npz`` with a ``.json`` sidecar) shared by the tasks of a unit. The key of a cache
   entry covers the content of the unit's files, the ``prepare`` section, ``input.node_kinds``, the
   dirconn version and the ``_events.tsv`` file, so a stale entry is never reused; the directory can be
   deleted after a run.
``--nprocs N``
   Worker processes (default 1). With more than one, tasks run in parallel processes (joblib, loky
   backend).
``--omp-nthreads N``
   Threads per worker for the numerical libraries (default 1). It sets ``OMP_NUM_THREADS``,
   ``OPENBLAS_NUM_THREADS``, ``MKL_NUM_THREADS`` and ``NUMBA_NUM_THREADS``, which the worker
   processes read when they start. The main process has already set up its BLAS thread pool (numpy is
   imported with dirconn), so with ``--nprocs 1`` the setting reaches only libraries that read the
   variables later; set the variables in the environment before starting ``dirconn-bids`` to control
   that pool too. The number of threads can change floating-point sums in the last digits.
``--mem-mb N``
   Memory limit in MB. It is recorded in ``provenance.json``; nothing enforces it.
``--resource-monitor``
   Not implemented in dirconn 0.2.0: a warning is printed and the option is ignored.

Other options
-------------

``--skip-bids-validation``
   Do not run ``bids-validator``. Otherwise it is run on ``BIDS_DIR`` when it is on the ``PATH`` (a
   warning is printed when it is not); a rejection stops the command with exit code 3.
``--reports-only``
   Rebuild the reports (``sub-XX.html``, :doc:`report`) and the methods boilerplate from the existing outputs of
   ``OUTPUT_DIR/dirconn``; nothing is recomputed. The subjects are the ``--participant-label`` values
   (default: every ``sub-*`` directory of the dataset). ``BIDS_DIR`` is not read, ``bids-validator`` is
   not run, and ``--config``, ``--set`` and the input filters are ignored with a warning. Exit code 2
   when ``OUTPUT_DIR/dirconn`` has no ``dirconn.toml`` or a requested subject has no results.
``--boilerplate-only``
   Rewrite only the methods boilerplate (``logs/CITATION.*``) from the existing outputs, under the
   same conditions. It excludes ``--reports-only``.
``-v``, ``--verbose``
   More log output on stderr: ``-v`` information (one line per task), ``-vv`` debug messages. The
   log file ``dirconn.log`` of the run always receives debug messages.
``--version``
   Print ``dirconn-bids <version>``.
``-h``, ``--help``
   Print the options.

Subcommands
-----------

``dirconn-bids validate-config FILE``
   Check a TOML configuration and print ``OK <config hash>``; errors go to stderr, one per line, and
   give exit code 2.
``dirconn-bids list-methods``
   List the methods that ``estimation.methods`` accepts, one per line after a header, with the columns
   ``name``, ``report_kind``, ``frequency_resolved``, ``supports_mask`` and ``backend``.

Reports and the methods text
----------------------------

After the tasks of a participant run have finished (exit code 0 or 1, not when the run stopped with
exit code 2), the command writes, for every subject that had a recording in the run,
``OUTPUT_DIR/dirconn/sub-XX.html`` (:doc:`report` explains what it shows), and then the methods text
of the whole output dataset, ``OUTPUT_DIR/dirconn/logs/CITATION.html``, ``.md``, ``.tex`` and ``.bib``.
Each file is written to a temporary name and moved into place, so a reader never sees half a file. The
report of a subject covers all recordings of that subject in the output dataset, also those of earlier
runs. The command prints one line per report (``dirconn-bids: report <path>``) and one for the methods
text to stderr.

The reports are built from the files that are already in the dataset. They compute no connectivity,
and nothing a report does can change a result. When a report or the methods text cannot be written,
the command prints the reason and the name of the crash file, goes on with the other subjects and ends
with exit code 1; the traceback is in ``crash-report-<subject>.txt`` (or
``crash-report-citation.txt``) in the run directory.

``--reports-only`` and ``--boilerplate-only`` do the same without a run:

* They read ``OUTPUT_DIR/dirconn`` only. ``BIDS_DIR`` must be given but is not read,
  ``bids-validator`` is not run, and ``--config``, ``--set``, ``--session-id``, ``--task-id``,
  ``--run-id`` and ``--bids-filter-file`` are ignored with a warning (also ``--participant-label``
  with ``--boilerplate-only``). The ``app`` extra is needed for both.
* ``--reports-only`` writes the report of each subject in ``--participant-label`` (default: every
  ``sub-*`` directory of the dataset) and then the methods text; ``--boilerplate-only`` writes the
  methods text only. The methods text always covers every subject of the dataset.
* They exit with 2 when ``OUTPUT_DIR/dirconn`` has no ``dirconn.toml`` (it is not an output dataset
  of ``dirconn-bids``), when a requested subject has no results, when the dataset has no subject
  with ``--reports-only``, and when both options are given. Failures leave crash files in
  ``logs/reports-<UTC time>-<pid>/`` and give exit code 1.
* Nothing is computed and the result files are not touched; the figures are drawn again.

Use ``--reports-only`` after updating dirconn (a newer version may draw the report differently) or to
rebuild a report that failed.

Exit codes
----------

=====  =================================================================================================
Code   Meaning
=====  =================================================================================================
0      Every task succeeded or was already done.
1      At least one task failed. The outputs of the other tasks are kept; each failure leaves a
       ``crash-<key>.txt`` file in the run directory, and its label is logged. Also 1 when a report
       could not be written (``crash-report-<subject>.txt`` in the run directory, or in
       ``logs/reports-<time>-<pid>/`` with ``--reports-only``); the results are kept.
2      The run did not start: invalid arguments, configuration or ``--set`` override; ``BIDS_DIR``
       missing; an output location inside ``BIDS_DIR``; no matching recordings, an ambiguous layout or
       an invalid filter file; an output dataset written with a different configuration, or whose
       stored ``dirconn.toml`` is not a valid configuration; the ``app`` extra missing; the group level;
       with ``--reports-only`` or ``--boilerplate-only``, an output dataset or subject that does not
       exist, or both options together; ``bids-validator`` failing to start. Or the run stopped on a file system
       error in its own files (dataset files, run directory, log, provenance; the message starts with
       ``the run stopped:``), before or while the tasks ran: tasks that had finished keep their
       results and are skipped by the next run.
3      ``bids-validator`` rejected the dataset.
=====  =================================================================================================

Running again and resuming
--------------------------

A run can be repeated on the same ``OUTPUT_DIR``, for example after an interruption, after adding
recordings or with other participants. Every task has a key computed from the content of the unit's
input files, the analysis hash of the configuration (the ``prepare``, ``estimation``, ``statistics``
and ``controls`` sections), the method (or ``controls``) and the dirconn version. A task whose last
written file already carries its key (the ``TaskKey`` of the result sidecar; for the controls task the
``desc-trgc`` sidecar, or the ``task_key`` of the diagnostics file when the time-reversal control is
off) is skipped and counted as ``skipped``. Changed input files or a new dirconn version therefore
redo the affected tasks, and a task that failed or was interrupted is run again.

The first run writes the configuration to ``OUTPUT_DIR/dirconn/dirconn.toml``. A later run is refused
(exit code 2, nothing written) when its ``analysis`` sections or its ``input`` section differ from that
file: results of different analyses never share a dataset. Use a new output directory, or run again
with the stored file (``--config OUTPUT_DIR/dirconn/dirconn.toml``). The ``output`` and ``execution``
sections may change between runs; each run records its own effective configuration in its run
directory. They are not part of the task keys, so changing ``execution.seed`` or
``output.long_table`` does not redo tasks that are already done.

Several jobs (for example one per participant on a cluster) may write to the same ``OUTPUT_DIR``:
every run has its own run directory, the result files of different units do not overlap, and the
dataset files and cache entries are moved into place atomically. Start the first job alone, or wait
until ``OUTPUT_DIR/dirconn/dirconn.toml`` exists, before starting the others: two runs that start at
the same moment on an empty directory can both pass the configuration check.

Each job also rewrites ``logs/CITATION.*`` from the subjects that exist when it finishes, so the text
written by an early job does not cover the subjects of the later ones. Run
``dirconn-bids BIDS_DIR OUTPUT_DIR participant --boilerplate-only`` once after the last job to get the
text for the whole dataset (``--reports-only`` for the pages is not needed: each job writes the pages
of its own subjects).

Running in Docker
-----------------

The Docker image (see the README) has ``dirconn-bids`` as its entrypoint, so the arguments are those
above. Mount the BIDS dataset read-only at ``/data`` and an output directory at ``/out``::

   docker run --rm -v /path/to/bids:/data:ro -v /path/to/out:/out \
     ghcr.io/hanweiliu-viola/dirconn:<tag> /data /out participant --participant-label 01

The report is then ``/path/to/out/dirconn/sub-01.html``, and ``provenance.json`` of the run names the
image tag and revision (``image``). To rebuild the report from an existing output, give the same
mounts and add ``--reports-only``. The image does not contain ``bids-validator``: the command warns
that the dataset is not validated and goes on, so ``--skip-bids-validation`` is not needed. The
container runs as an unprivileged user (UID 1000); on Linux add ``--user $(id -u):$(id -g)`` when the
output directory belongs to someone else. Without arguments the image prints the help.

Where to look after a run
-------------------------

``OUTPUT_DIR/dirconn/logs/run-<UTC start time>-<pid>/`` holds the log, the progress lines, the crash
files and the provenance of that run; :doc:`outputs` describes every file. ``OUTPUT_DIR/dirconn/sub-XX.html`` is the report of
a subject (:doc:`report`).

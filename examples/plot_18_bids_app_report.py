"""
==========================================================
Reading the participant report of dirconn-bids
==========================================================

Tutorial 17 runs the ``dirconn-bids`` command on a BIDS dataset and reads the result files by hand:
the configuration, the edge tables, the diagnostics and ``provenance.json``. It does not explain
them again here. This tutorial is about the one thing the command writes after every run that
tutorial 17 does not read: the participant report, ``sub-XX.html``, and the methods text in
``logs/CITATION.md``.

It simulates a small network, runs three methods on two subjects, and then looks at the report
through the files it is built from: which sections it has, which checks have which status, why a
time-resolved result is marked as not tested, what a failed method looks like, and how to rebuild
the report without computing anything again. The HTML page itself is a file to open in a browser;
the tutorial prints what is in it instead of showing screenshots.

It needs the ``app`` extra (``pip install "dirconn[app]"``).
"""

# %%
# Setup
# -----
# Three nodes: two cortical regions and one deep (STN) contact. Two influences are true,
# precuneus -> postcentral and postcentral -> STN, each with a 12 ms delay. Two subjects with one
# resting recording each, written as a BIDS dataset exactly as in tutorial 17 (30 s per recording
# in the quick version, 60 s otherwise).
import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dirconn
from dirconn import Edge, Network, direct_node, dk_node, write_roi_bids
from dirconn.simulation import NodeDynamics, simulate

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")

cortical = ["precuneus-lh", "postcentral-lh"]
nodes = [dk_node(n, region=n.removesuffix("-lh"), hemisphere="lh", atlas="DK") for n in cortical]
nodes.append(direct_node("STN-L", region="STN", hemisphere="lh"))
names = [*cortical, "STN-L"]
network = Network(
    nodes=nodes,
    edges=[
        Edge("precuneus-lh", "postcentral-lh", strength=0.4, delay_ms=12.0),
        Edge("postcentral-lh", "STN-L", strength=0.4, delay_ms=12.0),
    ],
    name="report",
)
dynamics = {
    name: NodeDynamics(kind="oscillator", freq_hz=freq)
    for name, freq in zip(names, (10.0, 20.0, 8.0), strict=True)
}

SFREQ = 250.0
DURATION_S = 30.0 if QUICK else 60.0
work = Path(tempfile.mkdtemp(prefix="dirconn_report_"))
bids_dir = work / "bids"
for index, subject in enumerate(["01", "02"]):
    sim = simulate(
        network,
        sfreq=SFREQ,
        n_times=int(DURATION_S * SFREQ),
        seed=SEED + index,
        dynamics=dynamics,
        snr_db=10.0,
        n_background=0,
        node_scale="unit",
    )
    write_roi_bids(
        sim.roi_observations(snr_db=10.0, seed=SEED + 100 + index),
        bids_dir / "derivatives" / "dirconn-roi",
        subject=subject,
        task="rest",
        description="DK",
    )
(bids_dir / "dataset_description.json").write_text(
    '{"Name": "dirconn report", "BIDSVersion": "1.9.0"}', encoding="utf-8"
)

# %%
# A run that produces a report
# ----------------------------
# Three methods: ``gc`` (Granger causality, one value per pair), ``pdc`` (frequency resolved) and
# ``adtf`` (time resolved). The VAR order is chosen by BIC, once per recording, among the orders
# 1 to ``order_max`` (12 here, to keep the search short; the default is 30). The number of
# surrogates is small so that the tutorial is quick; with it nothing can be significant after
# correction (the smallest possible p-value, ``1 / (n_surrogates + 1)``, is above the corrected
# threshold), which is fine here because this tutorial reads the report, not the edges.
N_SURROGATES = 19 if QUICK else 99
config = f"""
schema_version = 1

[estimation]
methods = ["gc", "pdc", "adtf"]
order = "bic"
order_max = 12
bands = {{ alpha = [8.0, 12.0], beta = [13.0, 30.0] }}

[statistics]
n_surrogates = {N_SURROGATES}
"""
config_file = work / "dirconn.toml"
config_file.write_text(config, encoding="utf-8")

from dirconn.cli.run import main as dirconn_bids  # noqa: E402

out_dir = work / "out"
argv = [str(bids_dir), str(out_dir), "participant", "--config", str(config_file)]
argv += ["--skip-bids-validation", "--nprocs", "2" if QUICK else "4"]
exit_code = dirconn_bids(argv)
print("exit code", exit_code)

root = out_dir / "dirconn"
for page in sorted(root.glob("sub-*.html")):
    print(page.relative_to(out_dir).as_posix(), f"{page.stat().st_size / 1e3:.0f} kB")

# %%
# Where the report is and what it answers
# ---------------------------------------
# ``out/dirconn/sub-01.html`` is one self-contained page: the style sheet and the figures are inside
# it and nothing is loaded from the network, so it can be copied, mailed or archived alone. Its
# eight sections each answer one question:
#
# * **Summary**: which recordings, with which input files (and their SHA-256), methods, bands, run
#   times, dirconn version and Docker image; the status of the whole subject.
# * **Input**: which nodes (and of which kind) were analysed, and how much of each recording was
#   valid after preparation.
# * **Diagnostics**: do the recordings suit a VAR model? Assumption checks, the selected model
#   order and, when the spectral check is on, the power spectral density.
# * **Connectivity**: what did each method find? Band matrices, net direction, spectra and
#   time-frequency maps.
# * **Controls**: do time-reversed and ordinary Granger causality agree on the direction?
# * **Statistics**: how was each result tested, and which edges are significant?
# * **Methods**: the methods text and the references (also in ``logs/CITATION.*``).
# * **Errors**: which tasks failed, and which parts of the report are missing and why.
#
# Every figure and table is a *reportlet*: a JSON file (and an SVG file when it has a figure) next
# to the unit's diagnostics. The report only arranges them, so the files can be read without a
# browser. A reportlet carries a title, a caption and a status.
from dirconn.reports.collect import collect_subject, selected_order
from dirconn.reports.reportlet import read_reportlets

outputs = collect_subject(root, "01")
rows = []
for label, figures_dir in [
    *((u.label, u.figures_dir) for u in outputs.units),
    ("sub-01", root / "sub-01" / "figures"),
]:
    for r in read_reportlets(figures_dir, label):
        rows.append(
            {
                "section": r.section,
                "unit": label,
                "reportlet": f"{r.desc}/{r.name}",
                "status": r.status,
            }
        )
listing = pd.DataFrame(rows)
print(listing.to_string(index=False))

# %%
# A few captions of the first recording. A caption says what the
# figure shows and which rule produced the status.
unit = outputs.units[0]
by_name = {f"{r.desc}/{r.name}": r for r in read_reportlets(unit.figures_dir, unit.label)}
for key in ("prep/record", "diag/order", "gc/tests"):
    print(f"{key}: {by_name[key].caption}\n")

# %%
# One figure of the report, drawn here from the file it is made from: the model-order selection
# of one recording, read from its diagnostics file. The curves are the information criteria for
# every candidate order; the selected order is the minimum of the chosen criterion.
diagnostics = json.loads(
    (unit.figures_dir / f"{unit.label}_diagnostics.json").read_text(encoding="utf-8")
)
selection = diagnostics["order_selection"]
chosen = selected_order(selection)
fig, ax = plt.subplots(figsize=(5.0, 3.2), layout="constrained")
for criterion in ("aic", "bic", "hqc"):
    if selection.get(criterion) is not None:
        ax.plot(selection["orders"], selection[criterion], marker="o", ms=3, label=criterion)
ax.axvline(chosen, color="k", ls="--", lw=1, label=f"selected ({selection['criterion']}): {chosen}")
ax.set_xlabel("VAR order")
ax.set_ylabel("criterion value")
ax.set_title(unit.label, fontsize=9)
ax.legend(fontsize=8)
print(
    "selected order", chosen, "of candidates", selection["orders"][0], "to", selection["orders"][-1]
)

# %%
# How to read the statuses
# ------------------------
# Each reportlet has one of four statuses, and each section and the whole report show the worst
# one present (``FAIL`` over ``WARN`` over ``PASS`` over ``INFO``):
#
# * ``PASS``: a check ran and found nothing to flag.
# * ``WARN``: a check ran and crossed a rule of thumb. The result is still written and the report
#   says which rule: a warning is a reason to look, not a verdict.
# * ``FAIL``: a check found the model unusable (for example an unstable VAR model), or a task or a
#   reportlet could not be made.
# * ``INFO``: not a check (a matrix, a table of the tests that were run), or a check that was not
#   run.
#
# The thresholds behind ``WARN`` are rules of thumb that nobody calibrated for these methods:
# more than one VAR coefficient per ten observations, a spectral radius of 0.99 or more, residual
# autocorrelation that is both significant and above 0.1, a spectral dynamic range above 60 dB,
# fewer than half of the samples valid, and a selected order at the largest order that was tried.
# A recording can pass all of them and still be unsuitable, and one that warns can be fine.
# These are the checks of sub-01, with their status.
assumptions = pd.DataFrame(by_name["diag/assumptions"].table)
print(assumptions[["method", "check", "status"]].to_string(index=False))

# %%
# Everything that is not a plain PASS or INFO in this subject, with its caption:
flagged = [
    (u.label, f"{r.desc}/{r.name}", r.status, r.caption)
    for u in outputs.units
    for r in read_reportlets(u.figures_dir, u.label)
    if r.status in ("warn", "fail")
]
if flagged:
    for label, key, status, caption in flagged:
        print(f"{status.upper()} {label} {key}: {caption}\n")
else:
    print("No reportlet of this subject is WARN or FAIL.")
for row in assumptions[assumptions["status"].isin(["warn", "fail"])].itertuples():
    print(f"{row.status.upper()} {row.method} {row.check}: {row.detail}")

# %%
# Results that are described, not tested
# --------------------------------------
# ``adtf`` has a value for every time point. On a continuous recording the default surrogate test
# shuffles 5 s blocks and fits the observed value on them, which leaves no time points of the
# result, so the result is not tested. The report says so in the Statistics section (status
# ``INFO``) instead of showing an empty table, and it is not an error: its edge table has ``n/a``
# in the test columns and its sidecar has a ``StatisticsNote``. Read the time course as a
# description. On epoched data every time point is tested against shuffled trials.
print("adtf:", by_name["adtf/tests"].caption)
print("gc tests:")
print(pd.DataFrame(by_name["gc/tests"].table).to_string(index=False))

# %%
# A method that fails
# -------------------
# A failing task never takes the others down: the command exits with 1, the other results are kept
# and the report still gets written. Here ``gc`` is given a model order of 100000, which no
# recording of this size can support. A run that changes the configuration needs its own output
# directory (a second run of the other configuration in ``out`` would be refused with exit code 2,
# because results of different analyses never share a dataset), so this one goes to ``out_failed``.
failing_config = (
    config.replace('methods = ["gc", "pdc", "adtf"]', 'methods = ["gc", "pdc"]')
    + "\n[estimation.gc]\norder = 100000\n"
)
failing_file = work / "failing.toml"
failing_file.write_text(failing_config, encoding="utf-8")
failed_dir = work / "out_failed"
failed_argv = [str(bids_dir), str(failed_dir), "participant", "--config", str(failing_file)]
failed_argv += ["--skip-bids-validation", "--participant-label", "01", "--nprocs", "2"]
print("exit code", dirconn_bids(failed_argv))

# %%
# The Errors section lists the failed task with its crash file (the traceback is in that file)
# and every part of the report that is missing because of it. The status of the section, and of
# the whole report, is ``FAIL``. The ``pdc`` results are there as usual.
failed_root = failed_dir / "dirconn"
failed_outputs = collect_subject(failed_root, "01")
errors = next(
    r for r in read_reportlets(failed_root / "sub-01" / "figures", "sub-01") if r.name == "errors"
)
print(errors.status.upper(), "-", errors.caption)
print(
    pd.DataFrame(errors.table)[["kind", "label", "reportlet", "reason", "crash_file"]].to_string()
)
print("methods with results:", sorted(failed_outputs.units[0].methods))

# %%
# Rebuilding the report and the methods text
# ------------------------------------------
# ``--reports-only`` writes ``sub-XX.html`` and ``logs/CITATION.*`` again from the outputs that are
# already there. It reads the output dataset only, not the BIDS dataset or a configuration, and
# computes nothing: the result files keep their modification times. Use it after a dirconn
# update that changes the report, or to rebuild a report that failed. ``--boilerplate-only`` writes
# only ``logs/CITATION.*``; on a cluster, where every job handles one subject, run it once at the
# end so that the text covers the whole dataset.
results = sorted(root.glob("sub-*/**/*_conn.json"))
before = {p: p.stat().st_mtime_ns for p in results}
page_before = (root / "sub-01.html").stat().st_mtime_ns
print("exit code", dirconn_bids([str(bids_dir), str(out_dir), "participant", "--reports-only"]))
print("results untouched:", before == {p: p.stat().st_mtime_ns for p in results})
print("report rewritten:", (root / "sub-01.html").stat().st_mtime_ns > page_before)
print(
    "exit code",
    dirconn_bids([str(bids_dir), str(out_dir), "participant", "--boilerplate-only"]),
)

# %%
# ``logs/CITATION.md`` is the methods paragraph for a manuscript, written from what the run did
# (the configuration, the model orders that were selected, the parameters dirconn filled in, the
# tests and the controls), with the references; ``CITATION.html``, ``.tex`` and ``.bib`` hold the
# same text for other uses. The first paragraphs:
citation = (root / "logs" / "CITATION.md").read_text(encoding="utf-8")
print("\n\n".join(citation.split("\n\n")[:3]))

# %%
# The same command in the Docker image
# ------------------------------------
# The image has ``dirconn-bids`` as its entrypoint, so the report comes out of a container the
# same way. This tutorial does not run Docker. In a shell, with the BIDS dataset mounted read-only
# and an output directory mounted for writing::
#
#     docker run --rm -v /path/to/bids:/data:ro -v /path/to/out:/out IMAGE /data /out participant
#
# The report is then ``/path/to/out/dirconn/sub-01.html``, and the run's ``provenance.json`` names
# the image tag and revision.

# %%
# What this does and does not show
# --------------------------------
# * The report is a reading aid. It puts the diagnostics, the figures, the tests and the methods
#   text of a subject in one place, so that a problem is noticed and a result can be traced to the
#   files and settings behind it. It does not decide whether the assumptions of a method hold for
#   your recording.
# * The statuses follow the rules above. They are rules of thumb that were not calibrated, so a
#   ``PASS`` is not a guarantee and a ``WARN`` is not a rejection. ``INFO`` means that nothing was
#   judged.
# * A figure shows what the method computed. In this tutorial the surrogate test is too small to
#   reject anything after correction, and the time-resolved ``adtf`` result is not tested.
# * All of these are predictive, not interventional, measures. Each result lists the
#   assumptions it checked (``not_checked`` means not tested, not that it holds):
conn = dirconn.Connectivity.from_netcdf(next(unit.figures_dir.parent.glob("*desc-gc_conn.nc")))
for check in conn.provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")
plt.close("all")

"""
==========================================================
One command, every method: dirconn as a BIDS App
==========================================================

Tutorials 1 to 16 call the library from Python: simulate or load a recording, build an estimator,
fit it, test it, plot it, one method at a time. Since 0.2.0 the same analysis of a whole dataset is
one command::

    dirconn-bids BIDS_DIR OUTPUT_DIR participant --config dirconn.toml

It finds the recordings of a BIDS dataset, prepares them, runs every configured method with its
significance test and the time-reversal control, and writes the results as a BIDS-Derivatives
dataset with provenance. This tutorial simulates a small network with a known truth, writes it
as a BIDS dataset, runs eight directed-connectivity methods on it with one command (and says what
the other six cost or need), and then reads what the command wrote: the edge tables, the result
files, the diagnostics and the run's logs.

It needs the ``app`` extra (``pip install "dirconn[app]"``).
"""

# %%
# Setup
# -----
# Four nodes: three cortical regions and one deep (STN) contact, as in a DBS recording. Two
# influences are true, precuneus -> postcentral and postcentral -> STN, each with a 12 ms delay.
# The pair precuneus -> STN is linked only through postcentral (indirect).
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

cortical = ["precuneus-lh", "postcentral-lh", "superiorfrontal-lh"]
nodes = [dk_node(n, region=n.removesuffix("-lh"), hemisphere="lh", atlas="DK") for n in cortical]
nodes.append(direct_node("STN-L", region="STN", hemisphere="lh"))
names = [*cortical, "STN-L"]
true_edges = {("precuneus-lh", "postcentral-lh"), ("postcentral-lh", "STN-L")}
network = Network(
    nodes=nodes,
    edges=[Edge(s, t, strength=0.4, delay_ms=12.0) for s, t in sorted(true_edges)],
    name="showcase",
)
dynamics = {
    name: NodeDynamics(kind="oscillator", freq_hz=freq)
    for name, freq in zip(names, (10.0, 20.0, 15.0, 8.0), strict=True)
}

# %%
# The input: a BIDS dataset
# -------------------------
# ``dirconn-bids`` reads BIDS. Region-level series (here simulated; in practice from a source
# reconstruction) are stored as a BIDS-Derivatives dataset with :func:`~dirconn.write_roi_bids`;
# sensor-level EEG/MEG/iEEG in BIDS works as well (``--set input.level=sensor``). Two subjects with
# one resting recording each, 250 Hz, observed with white noise at 10 dB.
SFREQ = 250.0
DURATION_S = 30.0 if QUICK else 60.0
work = Path(tempfile.mkdtemp(prefix="dirconn_showcase_"))
bids_dir = work / "bids"
out_dir = work / "out"
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
# A minimal raw dataset around the derivatives, as BIDS requires.
(bids_dir / "dataset_description.json").write_text(
    '{"Name": "dirconn showcase", "BIDSVersion": "1.9.0"}', encoding="utf-8"
)
for path in sorted(bids_dir.rglob("*")):
    if path.is_file():
        print(path.relative_to(bids_dir).as_posix())

# %%
# The configuration
# -----------------
# The analysis is a TOML file (``dirconn-bids validate-config FILE`` checks it; omitted keys keep
# their defaults, listed in the documentation's configuration page). Three things here are not
# defaults, and each matters:
#
# * **Methods.** Eight of the fourteen directed methods of ``dirconn-bids list-methods``
#   (``trgc`` is the time-reversal control and runs by itself): every method that finishes in
#   about a minute per recording. The others are left out for time or, for one, because it fails:
#
#   - ``adtf``, ``tpdc``, ``mtgc``, ``pcmci``: with 239 surrogates and 60 s, adtf took about 20 min
#     and tpdc about 28 min per recording on a workstation (4 workers); mtgc and pcmci had not
#     finished after 34 min. Add them to ``methods`` (mtgc also needs ``segment_length``, as imcoh
#     below) when the time is there.
#   - ``te``: one pairwise fit of these 60 s took more than nine minutes; the surrogate test repeats
#     it for every surrogate and band. Use it for a chosen hypothesis (tutorials 12, 13, 15).
#   - ``psi`` returns one value for a band of its own, which dirconn 0.2.0.dev1 then tries to reduce
#     to every configured band, so its task fails unless the configuration has a single band.
# * **A fixed model order.** With ``order = "bic"`` each VAR fit selects its order and checks that
#   the choice is stable, 5 to 8 s per fit here; with a fixed order 0.4 s. The surrogate test
#   refits every surrogate, so this decides between minutes and an hour. Choose the order once
#   (tutorial 2) and fix it.
# * **Parameters without defaults.** The multitaper methods (mtgc, imcoh, psi) need
#   ``segment_length`` on continuous data, because it sets the frequency resolution; the windowed
#   DTF needs its window. Without them these tasks fail (and the others still finish, with exit
#   code 1).
# * **Enough surrogates.** With 12 directed pairs per band, a corrected test can only reject when
#   the smallest possible p-value, ``1 / (n_surrogates + 1)``, is below ``0.05 / 12``: at least 239
#   surrogates. With 19 nothing is ever significant (dirconn warns about it).
METHODS = ["gc", "dtf", "pdc", "gpdc", "rpdc", "ssgc", "imcoh", "wdtf"]
config = f"""
schema_version = 1

[estimation]
methods = {METHODS!r}
order = 6
bands = {{ alpha = [8.0, 12.0], beta_low = [13.0, 20.0], beta_high = [20.0, 30.0] }}

[estimation.imcoh]
segment_length = 2.0
[estimation.wdtf]
window = 2.0
step = 1.0

[statistics]
n_surrogates = 239
"""
config_file = work / "dirconn.toml"
config_file.write_text(config.replace("'", '"'), encoding="utf-8")
print(config_file.read_text(encoding="utf-8"))

# %%
# The one command
# ---------------
# In a shell::
#
#     dirconn-bids bids out participant --config dirconn.toml --skip-bids-validation --nprocs 4
#
# Here it runs in-process with the same arguments; the return value is the exit code (0 every task
# succeeded; 1 a task failed and the other results are kept; 2 invalid input or configuration).
from dirconn.cli.run import main as dirconn_bids  # noqa: E402

argv = [str(bids_dir), str(out_dir), "participant", "--config", str(config_file)]
argv += ["--skip-bids-validation", "--nprocs", "2" if QUICK else "4"]
print("dirconn-bids " + " ".join(argv))
exit_code = dirconn_bids(argv)
print("exit code", exit_code)

# %%
# What the command wrote
# ----------------------
# One BIDS-Derivatives dataset, ``out/dirconn``. Per recording and method: the full result
# (``_conn.nc``, NetCDF with its provenance), a sidecar (``_conn.json``, written last: it marks the
# task as done, so a rerun skips it) and a long edge table (``_edges.tsv``). Per recording:
# ``figures/<recording>_diagnostics.json`` and the time-reversal control ``desc-trgc``. Per run:
# ``logs/run-<time>-<pid>/`` with the log, the progress of each task, the effective configuration,
# the installed packages and ``provenance.json``.
root = out_dir / "dirconn"
for path in sorted(root.rglob("*")):
    if path.is_file() and "sub-02" not in path.as_posix():
        print(path.relative_to(root).as_posix())

# %%
# The edge tables
# ---------------
# One row per directed pair and band (or one ``broadband`` row for methods without a frequency
# axis): the band value, its p-value, the corrected p-value and the decision. Time-resolved
# methods (wdtf here; adtf, tpdc) have a header-only table; their values are in ``_conn.nc``.
unit = root / "sub-01" / "eeg"
edges = pd.read_csv(unit / "sub-01_task-rest_desc-gc_edges.tsv", sep="\t")
print(edges.to_string(index=False))

# %%
# All methods against the truth
# -----------------------------
# For every method with an edge table, the beta-low band (13-20 Hz; ``broadband`` for gc, which
# has no frequency axis) of subject 01 as a matrix, true influences framed. The values of different methods are on
# different scales and mean different things (a share of a spectrum, a variance ratio, a
# coherence); compare where each method puts its largest values, not the numbers.
tables = {}
for method in METHODS:
    path = unit / f"sub-01_task-rest_desc-{method}_edges.tsv"
    table = pd.read_csv(path, sep="\t")
    if len(table):
        band = "beta_low" if "beta_low" in set(table["band"]) else table["band"].iloc[0]
        tables[method] = table[table["band"] == band]

n_cols = 4
n_rows = int(np.ceil(len(tables) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows), layout="constrained")
for ax in np.ravel(axes)[len(tables) :]:
    ax.axis("off")
summary = []
for ax, (method, table) in zip(np.ravel(axes), tables.items(), strict=False):
    matrix = np.full((len(names), len(names)), np.nan)
    for row in table.itertuples():
        matrix[names.index(row.source), names.index(row.target)] = row.value
    ax.imshow(matrix, cmap="viridis")
    for source, target in true_edges:
        ax.add_patch(
            plt.Rectangle(
                (names.index(target) - 0.5, names.index(source) - 0.5),
                1,
                1,
                fill=False,
                lw=2,
                edgecolor="red",
            )
        )
    ax.set_title(f"{method} ({table['band'].iloc[0]})", fontsize=9)
    ax.set_xticks(range(len(names)), [n.split("-")[0][:6] for n in names], rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), [n.split("-")[0][:6] for n in names], fontsize=7)
    is_true = [(r.source, r.target) in true_edges for r in table.itertuples()]
    true_values = table["value"][is_true]
    other_values = table["value"][[not t for t in is_true]]
    summary.append(
        {
            "method": method,
            "band": table["band"].iloc[0],
            "true edges above every other pair": bool(true_values.min() > other_values.max()),
            "significant true": int(table["significant"][is_true].astype(str).eq("True").sum()),
            "significant other": int(
                table["significant"][[not t for t in is_true]].astype(str).eq("True").sum()
            ),
        }
    )
fig.suptitle("rows: source, columns: target; red: true influence")
print(pd.DataFrame(summary).to_string(index=False))

# %%
# Diagnostics, control and provenance
# -----------------------------------
# The diagnostics file holds the share of valid samples, each channel's spread and the spectral
# dynamic range check; the time-reversal control is an ordinary result (``desc-trgc``); every run
# directory has ``provenance.json`` (command, times, host, versions, inputs and their hashes).
import json  # noqa: E402

diagnostics = json.loads((unit / "figures" / "sub-01_task-rest_diagnostics.json").read_text())
print({k: diagnostics[k] for k in ("valid_fraction", "n_times", "sfreq")})
for check in diagnostics["checks"]:
    print("diagnostic:", check["name"], check["status"])
run_dir = sorted((root / "logs").glob("run-*"))[-1]
provenance = json.loads((run_dir / "provenance.json").read_text())
print("command:", " ".join(provenance["command"]))
print("tasks:", provenance["tasks"])
progress = pd.read_json(run_dir / "progress.jsonl", lines=True)
print(progress[["label", "status", "seconds"]].to_string(index=False))

# %%
# What this does and does not show
# --------------------------------
# * One command ran eight methods on every recording of the dataset, tested each band against
#   surrogates and wrote results that a second run would skip. It does not choose the methods,
#   bands or model order for you: the defaults are a starting point, set in ``--config`` or
#   ``--set``, and the run's ``dirconn.toml`` records what was used.
# * The comparison above is one short simulation with a known truth, not an evaluation of the
#   methods (see the benchmark in the documentation). Methods answer different questions: imcoh
#   is not conditional on the other nodes and is blind to zero-lag coupling, gc is not spectral,
#   and DTF counts indirect paths (precuneus -> STN here goes through postcentral).
# * In the full version every method found both true influences significant, and also three or
#   four pairs that are not true influences: all of them among the three nodes of the chain
#   (the indirect precuneus -> STN and reversed directions), none involving the unconnected
#   superiorfrontal. With observation noise the best VAR of coupled signals has small real terms
#   where the network has none, and 60 s are enough to find them (tutorial 16). imcoh gives both
#   directions of a pair the same value, so its "significant" says coupled, not which way.
# * The surrogate test here shuffles 5 s blocks of a 60 s record (12 blocks); with 30 s (the quick
#   version) only 6 blocks are left to shuffle, which limits how small a p-value can get.
# * All of these are predictive, not interventional, measures. Each result lists the
#   assumptions it checked (``not_checked`` means not tested, not that it holds):
conn = dirconn.Connectivity.from_netcdf(unit / "sub-01_task-rest_desc-gc_conn.nc")
for check in conn.provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")
plt.close("all")

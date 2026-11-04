"""
================================================================
Multivariate transfer entropy: indirect paths and shared drivers
================================================================

Pairwise transfer entropy (tutorial 12) looks at two signals at a time. When A drives B and B
drives C, the past of A tells something about C, because it passes through B, and a pairwise
measure reports A -> C. When A drives both B and C, B and C share information that neither sends
to the other. Tutorial 5 showed the same two traps for Granger causality and the control: put the
other regions into the condition.

For transfer entropy the full condition is costly: every added signal adds dimensions, and the
nearest-neighbour estimator degrades quickly with the dimension. The **multivariate** mode of
:class:`~dirconn.TE` (its default) instead *selects* the past samples that carry information
about each target, one at a time, and tests every step by permutation (the algorithm of IDTxl,
Novelli et al. 2019). The result is a network of selected sources with their transfer entropy;
a source that is not selected is zero.
"""

# %%
# Setup
# -----
# The two shipped scenarios of tutorial 5, simulated at the region level. ``N_PERM`` is the number
# of permutations of every test: IDTxl's defaults (200 and 500) are the library's defaults too,
# but they take long, and 99 already reaches a level of 0.01 per test.
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dirconn
from dirconn import TE
from dirconn.scenario import load_scenario

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
DURATION_S = 15.0 if QUICK else 40.0
MAX_ORDER = 4 if QUICK else 6  # candidate lags of the target past
N_PERM = 19 if QUICK else 99
PERMUTATIONS = {
    "n_perm_max_stat": N_PERM,
    "n_perm_min_stat": N_PERM,
    "n_perm_omnibus": N_PERM,
    "n_perm_max_seq": N_PERM,
}
WORKERS = -1  # the targets in parallel processes (needs joblib, extra "parallel")
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")


def run_scenario(name):
    """Simulate a shipped scenario; return its specification, the simulation and the region data."""
    spec = load_scenario(name)
    sim = dirconn.simulate(
        spec.network,
        sfreq=spec.sfreq,
        n_times=int(DURATION_S * spec.sfreq),
        seed=SEED,
        dynamics=spec.dynamics,
        snr_db=spec.snr_db,
        n_background=0,
        node_scale="unit",
    )
    return spec, sim, sim.roi_observations(snr_db=spec.snr_db, seed=SEED + 1)


def both_modes(data):
    """Pairwise and multivariate transfer entropy of the same data."""
    return {
        "pairwise": TE(mode="pairwise", max_delay=3, workers=WORKERS).fit(data),
        "multivariate (default)": TE(
            max_order=MAX_ORDER, max_delay=3, workers=WORKERS, seed=SEED, **PERMUTATIONS
        ).fit(data),
    }


def show(truth, estimates, title):
    """The truth and one panel per estimate, each with its own colour scale."""
    fig, axes = plt.subplots(
        1, 1 + len(estimates), figsize=(4.6 * (1 + len(estimates)), 4.4), layout="constrained"
    )
    truth.plot(ax=axes[0], colorbar=False)
    axes[0].set_title("truth")
    for ax, (label, result) in zip(axes[1:], estimates.items(), strict=True):
        result.plot(ax=ax)
        ax.set_title(label)
    fig.suptitle(title)


def values_table(estimates, cells):
    """The value of some named cells in several estimates; ``cells`` is {label: (source, target)}."""
    rows = {}
    for label, (source, target) in cells.items():
        rows[label] = {
            name: result.values[result.source.index(source), result.target.index(target), 0, 0]
            for name, result in estimates.items()
        }
    return pd.DataFrame(rows).T


def selection_table(result):
    """What the multivariate mode selected for every target, with its omnibus test."""
    details = result.provenance.details
    names = result.source
    rows = {}
    for j, target in enumerate(names):
        rows[target] = {
            "target lags": details["selected_target_lags"][j],
            "sources (lag)": [f"{names[i]} ({lag})" for i, lag in details["selected_sources"][j]],
            "omnibus TE": round(details["omnibus"][j]["te"], 3),
            "omnibus p": round(details["omnibus"][j]["p"], 3),
        }
    return pd.DataFrame(rows).T


# %%
# An indirect path
# ----------------
# In the chain A drives B and B drives C, each one sample (10 ms) later; A does not drive C.
spec, sim, data = run_scenario("chain")
A, B, C = spec.network.names
estimates = both_modes(data)
show(sim.truth, estimates, "chain: A -> B -> C")
print(
    values_table(estimates, {"A -> C (no such influence)": (A, C), "A -> B (real)": (A, B)}).round(
        3
    )
)

# %%
# The pairwise mode reports A -> C. The multivariate mode, once it has selected the past of B for
# C, finds nothing left in the past of A, and A -> C is zero. The selection per target:
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", None)
print(selection_table(estimates["multivariate (default)"]))

# %%
# A shared driver
# ---------------
# In ``common_driver`` region A drives B (10 ms) and C (20 ms); B and C are not connected, and D
# is unconnected.
spec, sim, data = run_scenario("common_driver")
A, B, C, D = spec.network.names
estimates = both_modes(data)
show(sim.truth, estimates, "common driver: A -> B and A -> C")
print(
    values_table(estimates, {"B -> C (no such influence)": (B, C), "A -> C (real)": (A, C)}).round(
        3
    )
)
print(selection_table(estimates["multivariate (default)"]))

# %%
# What this does and does not show
# --------------------------------
# * On two simulated networks the multivariate mode removed the indirect link of a chain and the
#   link between two regions that share a driver, which the pairwise mode reported. This is one
#   seed of each (a demonstration, not an evaluation).
# * The result is already a decision: every selected source passed its permutation tests at
#   ``alpha`` (0.05 by default). ``fdr=True`` adds a correction over the targets. Running
#   :func:`dirconn.significance_test` on top would test again.
# * It is still predictive information transfer between the *recorded* signals: a driver that was
#   not recorded makes the regions it drives look linked (tutorial 5), whatever the mode. Each
#   result lists its assumptions; ``not_checked`` means the library did not test one, not that it
#   holds:
for check in estimates["multivariate (default)"].provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")

# %%
# * It is slow: every step estimates the information of every candidate for every permutation.
#   ``workers`` runs the targets in parallel, and ``device="cuda"`` computes the neighbour
#   searches on a GPU with PyTorch (extra ``gpu``).
plt.close("all")

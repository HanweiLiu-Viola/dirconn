"""
==============================================
When the picture misleads: indirect and hidden
==============================================

A directed connectivity estimate can show an influence that is not there. This tutorial builds
three situations in which that happens, each with a known truth, and for each one runs a control
that shows *why*: the same data analysed in a way that removes the false influence. It also shows
the one situation where no analysis of the recorded data can remove it, and why a significant p-value
does not help.

The tutorial is about a way of reading results, so all values come from short simulations of
three or four regions. They show that these effects exist, not how large they are in general.
"""

# %%
# Setup
# -----
# Three shipped scenarios are used: ``chain`` (A drives B, B drives C), ``common_driver`` (A drives B
# and C) and ``hidden_common_input`` (a region H, which will not be recorded, drives A and B; A
# also drives C). We compare two Granger causality estimators:
#
# * the **conditional** one (the default, ``conditional=True``) asks whether the past of the
#   source helps to predict the target *beyond what the past of all other recorded regions does*;
# * the **pairwise** one (``conditional=False``) looks at two regions at a time and ignores the rest.
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dirconn
from dirconn import GC, Connectivity
from dirconn.scenario import load_scenario

QUICK = os.environ.get("DIRCONN_QUICK") == "1"  # the tests set this; readers get the full version
SEED = 0
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")


def run_scenario(name):
    """Simulate a shipped scenario; return the simulation and the region data of every node."""
    spec = load_scenario(name)
    sim = dirconn.simulate(
        spec.network,
        sfreq=spec.sfreq,
        n_times=int((30.0 if QUICK else spec.duration_s) * spec.sfreq),
        seed=SEED,
        dynamics=spec.dynamics,
        snr_db=spec.snr_db,
        n_background=0,
        node_scale="unit",
    )
    return spec, sim, sim.roi_observations(snr_db=spec.snr_db, seed=SEED + 1)


def truth_of(sim, names):
    """The truth restricted to the regions that are analysed."""
    index = [sim.truth.source.index(name) for name in names]
    return Connectivity.from_array(
        sim.truth.values[np.ix_(index, index)],
        source=list(names),
        provenance=sim.truth.provenance,
    )


def estimate(data, *, conditional):
    return GC(order="bic", conditional=conditional, check_order_stability=False).fit(data)


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


# %%
# An indirect path
# ----------------
# In the chain A drives B and B drives C; A does not drive C. The pairwise estimator sees only A
# and C, and A's signal does help to predict C, because it passes through B.
spec, sim, data = run_scenario("chain")
A, B, C = spec.network.names
estimates = {
    "conditional (default)": estimate(data, conditional=True),
    "pairwise": estimate(data, conditional=False),
}
show(sim.truth, estimates, "chain: A -> B -> C")
print(
    values_table(estimates, {"A -> C (no such influence)": (A, C), "A -> B (real)": (A, B)}).round(
        3
    )
)

# %%
# The pairwise estimator reports a clear influence from A to C. The conditional one, which has
# the past of B at its disposal, does not. That is the control: the same data, with the missing
# piece supplied, and the false influence is gone, so it came from the path through B.

# %%
# A shared driver
# ---------------
# In ``common_driver`` region A drives B and C, which are not connected to each other. B and C get
# the same signal from A, 10 ms and 20 ms late, so the past of B helps to predict C.
spec, sim, data = run_scenario("common_driver")
A, B, C, D = spec.network.names
estimates = {
    "conditional (default)": estimate(data, conditional=True),
    "pairwise": estimate(data, conditional=False),
}
show(sim.truth, estimates, "common driver: A -> B and A -> C")
print(
    values_table(estimates, {"B -> C (no such influence)": (B, C), "A -> C (real)": (A, C)}).round(
        3
    )
)

# %%
# Again the pairwise estimator reports an influence between the two regions that share a driver,
# and the conditional one, which has the driver in the model, does not.

# %%
# A driver you did not record
# ---------------------------
# Now the driver is *missing*. In ``hidden_common_input`` an unrecorded region H drives A and B,
# and A drives C. We analyse only A, B and C, as one would with a real recording in which H was not
# measured.
spec, sim, all_regions = run_scenario("hidden_common_input")
A, B, C = spec.analysed
(H,) = spec.hidden
seen = all_regions.pick(list(spec.analysed))
print("analysed:", seen.names)
estimates = {
    "conditional (default)": estimate(seen, conditional=True),
    "pairwise": estimate(seen, conditional=False),
}
show(truth_of(sim, spec.analysed), estimates, "H drives A and B; only A, B and C are recorded")
print(
    values_table(estimates, {"A -> B (no such influence)": (A, B), "A -> C (real)": (A, C)}).round(
        3
    )
)

# %%
# Both estimators report an influence between A and B, which have no connection and share a hidden
# driver. Conditioning cannot help, because H is not among the recorded regions.
#
# The control that shows where it comes from needs the one thing a real recording does not have:
# the hidden region. Here we have it, because we simulated it. With H in the data, the conditional
# estimator removes the influence between A and B:
with_hidden = {"conditional, H recorded": estimate(all_regions, conditional=True)}
print(
    values_table(
        with_hidden, {"A -> B (no such influence)": (A, B), "H -> A (real)": (H, A)}
    ).round(3)
)

# %%
# **Significance does not help.** Tutorial 4 showed a test that asks whether a value is more than
# noise. Applied to the recorded data, it marks the false influence like the real ones, because
# the false influence is not noise: there *is* a relation between A and B, it just does not run
# from one to the other.
surrogate = dirconn.significance_test(
    GC(order=5, check_order_stability=False),
    seen,
    test="surrogate",
    surrogate="phase_randomize",
    n_surrogates=199,
    seed=SEED + 2,
)
significant = surrogate.significant()[:, :, 0, 0]
print(
    "A -> B significant:", bool(significant[0, 1]), "  A -> C significant:", bool(significant[0, 2])
)

# %%
# The library cannot tell either. Every result carries a check named ``causal_sufficiency``, the
# assumption that no unrecorded region drives two of the recorded ones, and the library marks it
# as not checked.
check = estimates["conditional (default)"].provenance.assumptions.get("causal_sufficiency")
print(f"{check.name}: {check.status}")

# %%
# What this does and does not show
# --------------------------------
# * Each false influence above was removed by a control that supplied what the estimate lacked: the
#   middle region of the chain, the shared driver, the hidden region. That is the evidence that
#   these influences come from a missing piece and not from noise. In a real recording the missing
#   piece may be one you do not know about, and then no control of this kind exists.
# * What **conditioning** does and does not do: it handles third regions that are recorded; it
#   cannot handle one that is not. Whether conditioning is the better choice in other settings
#   (many regions, little data) is not shown here; with many regions the conditional model has more
#   coefficients to estimate from the same data.
# * Every estimate is **predictive**. A value from A to C says that the past of A helps to predict
#   C given what was included; it does not say that A acts on C.
# * Every result lists the assumptions it rests on; ``not_checked`` means that the library did not
#   test an assumption, not that it holds:
for check in estimates["conditional (default)"].provenance.assumptions.checks:
    print(f"{check.name:24s} {check.status:12s} {check.detail}")

# %%
# * Not shown here: other ways to be misled, such as leakage between neighbouring regions when
#   region signals are estimated from the sensors (tutorial 7), influences that switch on and off
#   (tutorial 6), or coupling at zero delay.
plt.close("all")

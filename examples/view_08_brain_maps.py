"""
Open the two brain maps of tutorial 8 as interactive 3D windows.

Tutorial 8 shows the maps as pictures, because a tutorial page cannot be rotated. This script
computes the same thing and opens each map in a window of its own, where you can rotate, zoom and
change the view: the sources and sinks of the shipped scenario ``random_10``, once from the regions'
own series (the oracle) and once from regions estimated from simulated EEG. Both maps use the same
colour scale; red is a source, blue a sink, and the caption on every map is on purpose.

Run it from a terminal (it opens windows, so it is not run by the tests or by the documentation)::

    python examples/view_08_brain_maps.py

``DIRCONN_QUICK=1`` uses 30 s of data instead of 60 s. It needs the MNE ``sample`` dataset, which is
never downloaded without being asked (``dirconn.datasets.require_sample``), and MNE's 3D plotting
with a Qt binding. Close both windows to end the script.
"""

import os

import numpy as np
from qtpy import QtWidgets

import dirconn
from dirconn import viz
from dirconn.datasets import require_sample
from dirconn.headmodel import load_head_model
from dirconn.inverse import roi_series
from dirconn.scenario import load_scenario
from dirconn.simulation import simulate_on_head

QUICK = os.environ.get("DIRCONN_QUICK") == "1"
SEED = 0
ORDER = 4 if QUICK else 5
print(f"dirconn {dirconn.__version__}, numpy {np.__version__}")
print("sample data:", require_sample())  # stops with instructions when the data are missing

spec = load_scenario("random_10")
names = list(spec.network.names)
head = load_head_model("eeg")
print("simulating ...", flush=True)
sim = simulate_on_head(
    spec.network,
    head,
    sfreq=spec.sfreq,
    n_times=int((30.0 if QUICK else 60.0) * spec.sfreq),
    seed=SEED,
    dynamics=spec.dynamics,
    snr_db=spec.snr_db,
    n_background=10,
)
oracle = sim.roi_observations(snr_db=spec.snr_db, seed=SEED + 1)
print("estimating the regions from the sensors ...", flush=True)
estimated = roi_series(sim.data, head, names)
print("Granger causality ...", flush=True)
imbalance = {
    "regions' own series (oracle)": dirconn.gc(
        oracle, order=ORDER, check_order_stability=False
    ).degree_imbalance(),
    "regions estimated from the sensors": dirconn.gc(
        estimated, order=ORDER, check_order_stability=False
    ).degree_imbalance(),
}
vmax = max(np.nanmax(np.abs(values.values)) for values in imbalance.values())

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
brains = []
for label, values in imbalance.items():
    print("opening:", label, flush=True)
    brain = viz.degree_imbalance_map(
        values, vmax=vmax, degrees="all edges, raw values", backend="pyvistaqt"
    )
    brain.add_text(0.02, 0.83, label, "which", color="black", font_size=10)
    brains.append(brain)
print("The windows are open; close them to end the script.", flush=True)
app.exec()

"""Choose the regions of the head-model scenarios.

The rules are in ``dirconn.diagnostics.regions``; this script applies them to the MNE sample head
model and
prints the result, which the scenario files and ``tests/simulation/test_scenario_library.py`` carry.

``leakage`` lists the leakiest neighbour pairs X -> Y, then the far, quiet region Z of the best
pair (``leakage_adjacent``) and of every pair: the geometries of the ``leakage_adjacent_*``
scenarios of the leakage controls (issue #9) are pairs 1 to 4 (counted from 0) with their Z.

    python tools/pick_regions.py pool        # the four regions of the hand-written scenarios
    python tools/pick_regions.py leakage     # leaking neighbour pairs and a far, quiet region
                                             # for each (the leakage_adjacent scenarios)
    python tools/pick_regions.py distance    # one pair per distance tercile

Needs the MNE sample dataset.
"""

from __future__ import annotations

import argparse

import numpy as np

from dirconn.diagnostics.regions import (
    centre_distances,
    distance_strata,
    distant_region,
    leakage_pairs,
    min_visibility,
    pool,
)
from dirconn.simulation.headmodel import load_head_model
from dirconn.simulation.inverse import crosstalk


def show_pool(heads: dict, args: argparse.Namespace) -> None:
    """Print the pool of visible, well separated regions."""
    head = heads["eeg"]
    visibility = min_visibility(heads.values())
    distances = centre_distances(head)
    chosen = pool(visibility, distances, low=args.low, high=args.high, min_distance=args.distance)
    print("centre distances (mm):")
    print(np.round(distances[np.ix_(chosen, chosen)]).astype(int))
    for i in chosen:
        print(f"{head.roi_names[i]:32s} {visibility[i]:.2f}")


def show_leakage(heads: dict, args: argparse.Namespace) -> None:
    """Print the leakiest neighbour pairs, the far, quiet region of the best one and of each."""
    head = heads["eeg"]
    visibility = min_visibility(heads.values())
    distances = centre_distances(head)
    matrices = {m: crosstalk(heads[m], seed=0).matrix for m in ("eeg", "grad")}
    pairs = leakage_pairs(matrices, visibility)
    print(f"{len(pairs)} pairs above the 90th percentile of leakage on EEG and gradiometers")
    for pair in pairs[:8]:
        x, y = pair.source, pair.receiver
        leaks = {m: abs(matrices[m][y, x]) for m in matrices}
        print(
            f"{head.roi_names[x]} -> {head.roi_names[y]}: score {pair.score:.4f}, leak "
            f"{leaks['eeg']:.2f} (EEG) {leaks['grad']:.2f} (grad), {distances[x, y]:.0f} mm"
        )
    best = pairs[0]
    z = distant_region((best.source, best.receiver), matrices, visibility, distances)
    print(
        f"far, quiet region: {head.roi_names[z]} (visibility {visibility[z]:.2f}, "
        f"{distances[z, best.source]:.0f} and {distances[z, best.receiver]:.0f} mm away)"
    )
    print("far, quiet region of every pair (Z, X, Y of a leakage_adjacent scenario):")
    for k, pair in enumerate(pairs):
        x, y = pair.source, pair.receiver
        try:
            z = distant_region((x, y), matrices, visibility, distances)
        except ValueError:
            print(f"{k}: {head.roi_names[x]} -> {head.roi_names[y]}: no far, quiet region")
            continue
        print(
            f"{k}: Z {head.roi_names[z]}, X {head.roi_names[x]}, Y {head.roi_names[y]} "
            f"(visibility {visibility[z]:.2f}, {distances[z, x]:.0f} and {distances[z, y]:.0f} mm "
            "away)"
        )


def show_distance(heads: dict, args: argparse.Namespace) -> None:
    """Print one pair per distance tercile."""
    head = heads["eeg"]
    visibility = min_visibility(heads.values())
    distances = centre_distances(head)
    strata = distance_strata(visibility, distances, head.roi_names)
    print(f"tercile bounds: {strata.bounds[0]:.0f} mm and {strata.bounds[1]:.0f} mm")
    for label, (i, j) in strata.pairs.items():
        print(
            f"{label:5s} {head.roi_names[i]} -> {head.roi_names[j]}: {distances[i, j]:.0f} mm, "
            f"visibility {visibility[i]:.2f} and {visibility[j]:.2f}"
        )


def main() -> None:
    """Run the chosen subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pool_parser = sub.add_parser("pool")
    pool_parser.add_argument("--low", type=float, default=1.0)
    pool_parser.add_argument("--high", type=float, default=1.8)
    pool_parser.add_argument("--distance", type=float, default=60.0)
    pool_parser.set_defaults(run=show_pool)
    sub.add_parser("leakage").set_defaults(run=show_leakage)
    sub.add_parser("distance").set_defaults(run=show_distance)
    args = parser.parse_args()
    heads = {m: load_head_model(m) for m in ("eeg", "mag", "grad")}
    args.run(heads, args)


if __name__ == "__main__":
    main()

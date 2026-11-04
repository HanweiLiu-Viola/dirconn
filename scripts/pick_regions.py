"""Choose the regions of the head-model scenarios.

The rules are in ``dirconn.regions``; this script applies them to the MNE sample head model and
prints the result, which the scenario files and ``tests/test_scenario_library.py`` carry.

    python scripts/pick_regions.py pool        # the four regions of the hand-written scenarios
    python scripts/pick_regions.py leakage     # leaking neighbour pair and a far, quiet region
    python scripts/pick_regions.py distance    # one pair per distance tercile

Needs the MNE sample dataset.
"""

from __future__ import annotations

import argparse

import numpy as np

from dirconn.headmodel import load_head_model
from dirconn.inverse import crosstalk
from dirconn.regions import (
    centre_distances,
    distance_strata,
    distant_region,
    leakage_pairs,
    min_visibility,
    pool,
)


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
    """Print the leakiest neighbour pairs and the far, quiet region for the best one."""
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

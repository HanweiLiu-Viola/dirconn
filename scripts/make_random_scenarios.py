"""Write the shipped random scenarios ``random_5``, ``random_10`` and ``random_20``.

The regions are drawn from those whose visibility is at least ``--min-visibility`` times the median
region on EEG, magnetometers and gradiometers (the smallest of the three counts), without the
60 mm spacing of the hand-written pool: 20 regions do not fit into that, and neighbouring regions in
a large network are part of the difficulty. The graphs come from
``dirconn.scenario.random_scenario`` with the seeds in ``RANDOM_SEEDS``, so rerunning this script
reproduces the files exactly. Needs the MNE sample dataset.

    python scripts/make_random_scenarios.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dirconn.headmodel import load_head_model
from dirconn.scenario import RANDOM_SEEDS, SCENARIO_DIR, random_scenario


def candidate_regions(min_visibility: float) -> tuple[str, ...]:
    """Return the regions visible enough on all three modalities, in head-model order."""
    heads = [load_head_model(m) for m in ("eeg", "mag", "grad")]
    visibility = np.min([h.visibility for h in heads], axis=0)
    return tuple(
        name
        for name, value in zip(heads[0].roi_names, visibility, strict=True)
        if value >= min_visibility
    )


def main() -> None:
    """Generate the files and print how many graphs were rejected for each."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=SCENARIO_DIR)
    args = parser.parse_args()
    regions = candidate_regions(args.min_visibility)
    print(f"{len(regions)} candidate regions with visibility >= {args.min_visibility:g}")
    for name, seed in RANDOM_SEEDS.items():
        n_nodes = int(name.split("_")[1])
        result = random_scenario(n_nodes, seed=seed, regions=regions)
        result.spec.to_yaml(args.out / f"{name}.yaml")
        print(
            f"{name}: seed {seed}, {result.n_rejected} graphs rejected for the spectral radius, "
            f"{result.n_acyclic} redrawn for lack of a cycle"
        )


if __name__ == "__main__":
    main()

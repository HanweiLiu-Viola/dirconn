"""Write ``protocols/leakage_pairs.yaml``, the region-pair leakage study.

The protocol lists the regions the pairs are drawn from: those whose visibility is at least
``--min-visibility`` times the median region on EEG, magnetometers and gradiometers (the smallest
of the three counts), the same candidates as the random scenarios. The list is written into the
file, so the protocol needs no head model to be expanded; rerunning this script reproduces the file
exactly. Needs the MNE sample dataset.

    python scripts/make_leakage_pairs_protocol.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_random_scenarios import candidate_regions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "protocols" / "leakage_pairs.yaml"
N_PAIRS = 45
PAIR_SEED = 14
MIN_VISIBILITY = 0.5


def build(regions: tuple[str, ...], n: int = N_PAIRS, seed: int = PAIR_SEED) -> dict[str, Any]:
    """Return the protocol as a dictionary, with the regions the pairs are drawn from."""
    return {
        "version": 1,
        "name": "leakage_pairs",
        "description": (
            f"What leaks between regions: {n} random pairs of regions without any edge, each a "
            "two-node network, EEG and gradiometers. Whatever an estimator finds between the "
            "series of a pair at source level is leakage (the region level is the control); the "
            "score of the false edge x -> y is correlated with the leak of x into the series of y "
            "(crosstalk matrix in the results header). Conditional Granger causality and gPDC, "
            "BIC order, one seed per pair; 45 pairs because the gate estimated 0.23 h for 150. "
            f"The regions are those with a visibility of at least {MIN_VISIBILITY:g} on all three "
            "modalities (scripts/make_leakage_pairs_protocol.py)."
        ),
        "kind": "region_pairs",
        "pairs": {"n": n, "seed": seed, "regions": list(regions)},
        "estimators": ["gc/bic", "gpdc/bic"],
        "n_seeds": 1,
        "snr_db": [10],
        "duration_s": [60],
        "source_model": ["partial"],
        "modality": ["eeg", "grad"],
        "settings": {"store_scores": True},
        "budget_hours": 0.06,
    }


def main() -> None:
    """Write the protocol file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-visibility", type=float, default=MIN_VISIBILITY)
    args = parser.parse_args()
    regions = candidate_regions(args.min_visibility)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(build(regions), handle, sort_keys=False, width=100)
    print(f"{len(regions)} candidate regions, {N_PAIRS} pairs: written to {args.out}")


if __name__ == "__main__":
    main()

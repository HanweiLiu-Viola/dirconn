"""Write a region-pair leakage protocol: ``leakage_pairs`` or its add-on ``leakage_pairs_controls``.

The protocol lists the regions the pairs are drawn from: those whose visibility is at least
``--min-visibility`` times the median region on EEG, magnetometers and gradiometers (the smallest
of the three counts), the same candidates as the random scenarios. The list is written into the
file, so the protocol needs no head model to be expanded; rerunning this script reproduces the file
exactly. Needs the MNE sample dataset.

``--name`` picks one of the protocols in :data:`PRESETS` (number of pairs, estimators, budget and
description); ``--n``, ``--estimators`` and ``--budget-hours`` override the preset. Both protocols
draw their pairs with the same seed from the same regions, and the pairs are a prefix of one
permutation, so the first 45 pairs of ``leakage_pairs_controls`` are those of ``leakage_pairs``.

    python tools/make_leakage_pairs_protocol.py                                # leakage_pairs
    python tools/make_leakage_pairs_protocol.py --name leakage_pairs_controls  # issue #9
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
PROTOCOL_DIR = ROOT / "validation" / "protocols"
DEFAULT_OUT = PROTOCOL_DIR / "leakage_pairs.yaml"
N_PAIRS = 45
PAIR_SEED = 14
MIN_VISIBILITY = 0.5


def _pairs_description(n: int, min_visibility: float) -> str:
    return (
        f"What leaks between regions: {n} random pairs of regions without any edge, each a "
        "two-node network, EEG and gradiometers. Whatever an estimator finds between the "
        "series of a pair at source level is leakage (the region level is the control); the "
        "score of the false edge x -> y is correlated with the leak of x into the series of y "
        "(crosstalk matrix in the results header). Conditional Granger causality and gPDC, "
        "BIC order, one seed per pair; 45 pairs because the gate estimated 0.23 h for 150. "
        f"The regions are those with a visibility of at least {min_visibility:g} on all three "
        "modalities (tools/make_leakage_pairs_protocol.py)."
    )


def _controls_description(n: int, min_visibility: float) -> str:
    return (
        f"The leakage controls (issue #9), an add-on to leakage_pairs: {n} random pairs of "
        "regions without any edge, each a two-node network, EEG and gradiometers, drawn from the "
        "same regions with the same pair seed, so the first 45 pairs are those of leakage_pairs. "
        "Conditional Granger causality as the reference and the three controls against leakage: "
        "time-reversed GC (BIC order), imaginary coherence and the phase slope index "
        "(multitaper), one seed per pair. A pair is a false positive of an estimator when its "
        "strongest |score| at sensor or source level is above the 95th percentile of the same estimator's "
        "strongest |score| at region level; the controls are compared with GC pair by pair "
        "(McNemar). The scores are kept. The regions are those with a visibility of at least "
        f"{min_visibility:g} on all three modalities (tools/make_leakage_pairs_protocol.py "
        "--name leakage_pairs_controls)."
    )


# name -> number of pairs, estimators, budget in hours, description(n, min_visibility)
PRESETS: dict[str, dict[str, Any]] = {
    "leakage_pairs": {
        "n": N_PAIRS,
        "estimators": ["gc/bic", "gpdc/bic"],
        "budget_hours": 0.06,
        "description": _pairs_description,
    },
    "leakage_pairs_controls": {
        "n": 150,
        "estimators": ["gc/bic", "trgc/bic", "imcoh/mt", "psi/mt"],
        "budget_hours": 2.0,
        "description": _controls_description,
    },
}


def build(
    regions: tuple[str, ...],
    n: int | None = None,
    seed: int = PAIR_SEED,
    *,
    name: str = "leakage_pairs",
    estimators: list[str] | None = None,
    budget_hours: float | None = None,
    min_visibility: float = MIN_VISIBILITY,
) -> dict[str, Any]:
    """Return the protocol as a dictionary, with the regions the pairs are drawn from.

    ``name`` is a key of :data:`PRESETS`; ``n``, ``estimators`` and ``budget_hours`` default to
    its values.
    """
    preset = PRESETS[name]
    n = preset["n"] if n is None else n
    return {
        "version": 1,
        "name": name,
        "description": preset["description"](n, min_visibility),
        "kind": "region_pairs",
        "pairs": {"n": n, "seed": seed, "regions": list(regions)},
        "estimators": list(preset["estimators"] if estimators is None else estimators),
        "n_seeds": 1,
        "snr_db": [10],
        "duration_s": [60],
        "source_model": ["partial"],
        "modality": ["eeg", "grad"],
        "settings": {"store_scores": True},
        "budget_hours": preset["budget_hours"] if budget_hours is None else budget_hours,
    }


def main() -> None:
    """Write the protocol file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", choices=sorted(PRESETS), default="leakage_pairs")
    parser.add_argument("--out", type=Path, help="default: validation/protocols/<name>.yaml")
    parser.add_argument("--min-visibility", type=float, default=MIN_VISIBILITY)
    parser.add_argument("--n", type=int, help="number of pairs (default: the preset's)")
    parser.add_argument("--estimators", nargs="+", help="estimator labels (default: the preset's)")
    parser.add_argument("--budget-hours", type=float, help="budget (default: the preset's)")
    args = parser.parse_args()
    out = args.out if args.out is not None else PROTOCOL_DIR / f"{args.name}.yaml"
    regions = candidate_regions(args.min_visibility)
    protocol = build(
        regions,
        args.n,
        name=args.name,
        estimators=args.estimators,
        budget_hours=args.budget_hours,
        min_visibility=args.min_visibility,
    )
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False, width=100)
    print(f"{len(regions)} candidate regions, {protocol['pairs']['n']} pairs: written to {out}")


if __name__ == "__main__":
    main()

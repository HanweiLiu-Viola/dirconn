"""Compare two runs of scripts/validate_methods.py, e.g. the Docker image against the published one.

Rows are matched by scenario, method and seed. The script prints how many rows agree exactly, the
largest absolute difference of every score, and every scenario and method whose median over the
seeds differs (the benchmark asserts on multi-seed medians, so those are what would change a
conclusion). It exits 1 when the two runs do not cover the same rows or the same settings.

Usage::

    python scripts/validate_methods.py --results run.json --output run.md     # e.g. in the image
    python scripts/compare_validation.py run.json                             # vs the published
    python scripts/compare_validation.py run.json --reference other.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORES = ("auroc", "auprc", "direction_accuracy", "separation", "order")


def _load(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {(r["scenario"], r["method"], r["seed"]): r for r in data["rows"]}
    return data["meta"], rows


def _value(row: dict, score: str) -> float:
    value = row.get(score)
    return float("nan") if value is None else float(value)


def _median(values: list[float]) -> float:
    """Median of the defined values; NaN (score undefined for the scenario) when there are none."""
    defined = [v for v in values if np.isfinite(v)]
    return float(np.median(defined)) if defined else float("nan")


def compare(reference: Path, candidate: Path) -> tuple[int, str]:
    """Return the exit code and a markdown report of the differences."""
    ref_meta, ref = _load(reference)
    new_meta, new = _load(candidate)
    lines = [
        f"# Validation runs compared\n\nreference: `{reference}`  \ncandidate: `{candidate}`\n"
    ]
    for name, meta in (("reference", ref_meta), ("candidate", new_meta)):
        env = ", ".join(f"{k} {v}" for k, v in meta.get("environment", {}).items())
        lines.append(f"- {name}: {env}")
    settings = ("n_times", "seeds", "sfreq", "spectral_band")
    if any(ref_meta.get(k) != new_meta.get(k) for k in settings) or set(ref) != set(new):
        lines.append("\nThe runs differ in their settings or rows; they cannot be compared.")
        return 1, "\n".join(lines)

    exact = sum(all(np.isclose(_value(ref[k], s), _value(new[k], s), rtol=0, atol=0,
                               equal_nan=True) for s in SCORES) for k in ref)  # fmt: skip
    lines.append(f"\n{exact} of {len(ref)} rows agree exactly.\n")
    lines.append("| score | largest absolute difference |\n|---|---|")
    for score in SCORES:
        diff = np.array([abs(_value(ref[k], score) - _value(new[k], score)) for k in ref])
        largest = np.nanmax(diff) if np.isfinite(diff).any() else float("nan")
        lines.append(f"| {score} | {largest:.3g} |")

    medians: dict[tuple[str, str], list[tuple[dict, dict]]] = defaultdict(list)
    for key in ref:
        medians[key[:2]].append((ref[key], new[key]))
    changed = []
    for (scenario, method), pairs in sorted(medians.items()):
        for score in ("auroc", "auprc", "direction_accuracy"):
            a = _median([_value(r, score) for r, _ in pairs])
            b = _median([_value(n, score) for _, n in pairs])
            if not np.isclose(a, b, rtol=0, atol=1e-12, equal_nan=True):
                changed.append(f"| {scenario} | {method} | {score} | {a:.4g} | {b:.4g} |")
    if changed:
        lines.append("\nMedians over the seeds that differ:\n")
        lines.append("| scenario | method | score | reference | candidate |\n|---|---|---|---|---|")
        lines.extend(changed)
    else:
        lines.append("\nNo median over the seeds differs.")
    return 0, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the comparison; return 1 when the runs cannot be compared."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "docs" / "method_validation_results.json"
    )
    args = parser.parse_args(argv)
    code, report = compare(args.reference, args.candidate)
    print(report)
    return code


if __name__ == "__main__":
    sys.exit(main())

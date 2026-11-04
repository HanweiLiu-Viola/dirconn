"""Regenerate ``docs/method_validation.md``: what each method can and cannot see.

Runs every method of ``dirconn.validation.METHODS`` on every scenario of ``SCENARIOS`` for several
seeds, scores the estimates against the known networks and writes the tables. The raw rows are kept
next to the report so that the text can be rendered again without simulating anything::

    python scripts/validate_methods.py                     # full run, about ten minutes
    python scripts/validate_methods.py --seeds 3           # quick look
    python scripts/validate_methods.py --from-results      # re-render from the saved rows

The hand-written reading of the numbers lives in ``docs/method_validation_findings.md`` and is
inserted at the top of the report when the file exists.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dirconn.validation import load_rows, render_report, run_all, save_rows

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Run the validation, or re-render a saved one, and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds (0, 1, ...)")
    parser.add_argument("--n-times", type=int, default=10000, help="samples per simulation")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "method_validation.md")
    parser.add_argument(
        "--results", type=Path, default=ROOT / "docs" / "method_validation_results.json"
    )
    parser.add_argument(
        "--findings", type=Path, default=ROOT / "docs" / "method_validation_findings.md"
    )
    parser.add_argument(
        "--from-results", action="store_true", help="re-render the saved rows without running"
    )
    args = parser.parse_args(argv)

    if args.from_results:
        rows, meta = load_rows(args.results)
        n_times, seeds, env = meta["n_times"], meta["seeds"], meta["environment"]
    else:
        seeds = list(range(args.seeds))
        n_times, env = args.n_times, None
        started = time.time()

        def progress(message: str) -> None:
            print(f"[{time.time() - started:6.0f}s] {message}", flush=True)

        rows = run_all(seeds=seeds, n_times=n_times, progress=progress)
        save_rows(rows, args.results, n_times=n_times, seeds=seeds)
        print(f"rows written to {args.results}")

    findings = args.findings.read_text(encoding="utf-8") if args.findings.exists() else ""
    report = render_report(rows, n_times=n_times, seeds=seeds, findings=findings, env=env)
    args.output.write_text(report, encoding="utf-8", newline="\n")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

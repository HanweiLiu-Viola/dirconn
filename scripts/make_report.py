"""Write ``docs/benchmark_results.md`` from the published results of the benchmark.

Every table is computed from ``benchmark_results/<protocol>.jsonl`` (``dirconn.report``); nothing in
the generated part is typed by hand. The hand-written reading of the numbers is
``docs/benchmark_conclusions.md``; when it exists it is put in front of the tables unchanged::

    python scripts/make_report.py                    # needs the results of all ten protocols
    python scripts/make_report.py --allow-missing    # what exists, with the others named as not run

The results come from ``python scripts/run_benchmark.py <protocol>``; the environment they were
made in is in ``constraints/benchmark.txt`` and in the header of every results file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dirconn.protocol import read_results
from dirconn.report import PUBLISHED, build_report, load_add_ons, load_published

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Build the report and write it; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", type=Path, default=ROOT / "benchmark_results")
    parser.add_argument(
        "--conclusions", type=Path, default=ROOT / "docs" / "benchmark_conclusions.md"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "benchmark_results.md")
    parser.add_argument(
        "--allow-missing", action="store_true", help="build the report from the files that exist"
    )
    args = parser.parse_args(argv)
    try:
        published = load_published(args.results_dir)
    except FileNotFoundError as error:
        if not args.allow_missing:
            print(f"error: {error} (--allow-missing builds the rest)", file=sys.stderr)
            return 2
        published = {
            name: read_results(args.results_dir / f"{name}.jsonl")
            for name in PUBLISHED
            if (args.results_dir / f"{name}.jsonl").is_file()
        }
    add_ons = load_add_ons(args.results_dir)
    conclusions = args.conclusions.read_text(encoding="utf-8") if args.conclusions.is_file() else ""
    text = build_report({**published, **add_ons}, conclusions)
    args.output.write_text(text, encoding="utf-8", newline="\n")
    extra = f", plus {len(add_ons)} add-on{'s' if len(add_ons) > 1 else ''}" if add_ons else ""
    print(f"{len(published)} of {len(PUBLISHED)} protocols{extra}, written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run the non-stationary sweep, or re-render the report of one that has run.

Four three-region scenarios switch a coupling on, off, around or briefly on; ADTF, TPDC and WDTF
are scored with the time-resolved metrics of ``dirconn.evaluation``. A dataset (scenario, SNR,
seed) is one unit of work; the methods all see the same series::

    python scripts/run_nonstationary.py --gate-only      # time one dataset, measure the efficiency
    python scripts/run_nonstationary.py --tier quick     # 8 datasets: the daily regression
    python scripts/run_nonstationary.py                  # the full grid: gate, run, report
    python scripts/run_nonstationary.py --from-results   # re-render without simulating

Finished datasets are appended to ``<results>.partial.jsonl`` as they come in, so an interrupted run
resumes where it stopped; the partial file is removed when the results file is written. The gate
**measures** the parallel efficiency on this machine (one dataset per worker, all at once) instead
of assuming it, estimates the wall-clock time, and the run records estimate and actual in the
results header so that the two can be compared (an assumed efficiency was once off by a factor
of six).
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":  # BLAS threads must be limited before numpy is imported
    for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from dirconn.nonstationary import (  # noqa: E402
    N_SEEDS,
    SNR_DB,
    datasets,
    load_results,
    render_report,
    run_dataset,
    save_results,
)

ROOT = Path(__file__).resolve().parents[1]
GATE_TASK = ("switch_on", 10.0, 0)
Task = tuple[str, float, int]


def _noop(_: int) -> int:
    return os.getpid()


def gate(jobs: int, n_datasets: int) -> dict[str, Any]:
    """Time one dataset alone, then ``jobs`` of them at once, and estimate the run."""
    gate_started = time.perf_counter()
    single = float(run_dataset(GATE_TASK)["seconds"])
    if jobs <= 1:
        efficiency, startup = 1.0, 0.0
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            started = time.perf_counter()
            list(pool.map(_noop, range(jobs)))
            startup = time.perf_counter() - started  # processes started, dirconn not yet imported
            started = time.perf_counter()
            list(pool.map(run_dataset, [GATE_TASK] * jobs))
            wall = time.perf_counter() - started
        efficiency = min(1.0, single / wall)
    estimate = startup + math.ceil(n_datasets / jobs) * single / efficiency
    return {
        "single_dataset_s": single,
        "jobs": jobs,
        "efficiency": efficiency,
        "startup_s": startup,
        "n_datasets": n_datasets,
        "estimate_s": estimate,
        "gate_wall_s": time.perf_counter() - gate_started,
    }


def _key(task: list[Any]) -> Task:
    return (str(task[0]), float(task[1]), int(task[2]))


def run(tasks: list[Task], jobs: int, partial: Path) -> list[dict[str, Any]]:
    """Run the datasets that are not in ``partial`` yet; return the rows of all of them."""
    done: dict[Task, list[dict[str, Any]]] = {}
    if partial.exists():
        with open(partial, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    done[_key(item["task"])] = item["rows"]
    todo = [t for t in tasks if t not in done]
    print(f"{len(done)} datasets already done, {len(todo)} to run", flush=True)
    started = time.perf_counter()

    def keep(item: dict[str, Any]) -> None:
        task = _key(item["task"])
        done[task] = item["rows"]
        with open(partial, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(item) + "\n")
        print(
            f"[{time.perf_counter() - started:6.0f}s] {len(done)}/{len(tasks)} {task} "
            f"({item['seconds']:.0f} s)",
            flush=True,
        )

    if jobs <= 1:
        for task in todo:
            keep(run_dataset(task))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for future in as_completed([pool.submit(run_dataset, t) for t in todo]):
                keep(future.result())
    return [row for task in tasks for row in done[task]]


def main(argv: list[str] | None = None) -> int:
    """Run the sweep, or re-render a saved one, and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tier", choices=("full", "quick"), default="full")
    parser.add_argument("--seeds", type=int, default=None, help=f"default {N_SEEDS} (quick: 2)")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--results", type=Path, default=ROOT / "docs" / "nonstationary_results.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs" / "nonstationary_validation.md"
    )
    parser.add_argument(
        "--findings", type=Path, default=ROOT / "docs" / "nonstationary_findings.md"
    )
    parser.add_argument("--from-results", action="store_true", help="re-render, run nothing")
    parser.add_argument("--gate-only", action="store_true", help="measure, estimate, run nothing")
    args = parser.parse_args(argv)

    if args.from_results:
        rows, meta = load_results(args.results)
    else:
        quick = args.tier == "quick"
        seeds = args.seeds if args.seeds is not None else (2 if quick else N_SEEDS)
        tasks = datasets(seeds=seeds, snr_db=(10.0,) if quick else SNR_DB)
        measured = gate(args.jobs, len(tasks))
        print(
            f"gate: one dataset {measured['single_dataset_s']:.1f} s alone, efficiency "
            f"{measured['efficiency']:.2f} with {measured['jobs']} workers, startup "
            f"{measured['startup_s']:.1f} s -> estimate for {len(tasks)} datasets: "
            f"{measured['estimate_s'] / 60:.1f} min",
            flush=True,
        )
        if args.gate_only:
            return 0
        started = time.perf_counter()
        partial = args.results.with_suffix(".partial.jsonl")
        rows = run(tasks, args.jobs, partial)
        wall = time.perf_counter() - started
        meta = {
            "tier": args.tier,
            "seeds": seeds,
            "gate": measured,
            "actual_wall_s": wall,
            "estimate_over_actual": measured["estimate_s"] / wall if wall > 0 else None,
        }
        save_results(rows, args.results, meta=meta)
        partial.unlink(missing_ok=True)
        rows, meta = load_results(args.results)
        print(
            f"run: {wall / 60:.1f} min against an estimate of "
            f"{measured['estimate_s'] / 60:.1f} min; results written to {args.results}"
        )

    findings = args.findings.read_text(encoding="utf-8") if args.findings.exists() else ""
    report = render_report(rows, meta, findings=findings)
    args.output.write_text(report, encoding="utf-8", newline="\n")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

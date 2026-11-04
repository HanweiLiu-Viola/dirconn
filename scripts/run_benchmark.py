"""Run a benchmark protocol, or re-render the results of one that has run.

A protocol (``protocols/<name>.yaml``, see ``dirconn.protocol``) lists scenarios and axes; the
runner simulates every task once through the head model and scores all estimators on it. Results
go to ``benchmark_results/<name>.jsonl``, one line per finished task, so an interrupted run resumes
where it stopped::

    python scripts/run_benchmark.py quick --dry-run        # count the tasks, run nothing
    python scripts/run_benchmark.py quick --gate-only      # time one task per network size
    python scripts/run_benchmark.py quick                  # gate, then run what is not done yet
    python scripts/run_benchmark.py quick --from-results   # re-render the saved results

Before a run the budget gate times one task per network size and estimates the wall-clock time of
the whole protocol; above the protocol's ``budget_hours`` it refuses unless ``--force``. A task
that fails is recorded and the run goes on; the exit code is 1 when any task failed, and 3 when a
protocol with a saturation guard has run but fails it (a grid on which even the best estimator is
perfect everywhere on the easy chain). Needs the MNE sample dataset, and ``joblib`` for more than
one worker.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":  # BLAS threads must be limited before numpy is imported
    for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
from collections import Counter  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from dirconn.protocol import (  # noqa: E402
    DEFAULT_EFFICIENCY,
    Protocol,
    check_budget,
    estimate_hours,
    expand,
    load_protocol,
    read_results,
    render_summary,
    run_protocol,
    saturation_report,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "protocols"
RESULTS_DIR = ROOT / "benchmark_results"


def resolve_protocol(name: str) -> Path:
    """Return the file of a protocol given as a path or as a bare name in ``protocols/``."""
    path = Path(name)
    if path.suffix and path.exists():
        return path
    candidate = PROTOCOL_DIR / f"{name}.yaml"
    if candidate.exists():
        return candidate
    known = sorted(p.stem for p in PROTOCOL_DIR.glob("*.yaml")) if PROTOCOL_DIR.exists() else []
    raise FileNotFoundError(f"No protocol {name!r}; known protocols: {known}.")


def _fail(message: str) -> int:
    """Print an error after whatever was printed before it, and return the exit code 2."""
    sys.stdout.flush()
    print(f"error: {message}", file=sys.stderr, flush=True)
    return 2


def default_workers() -> tuple[int, str | None]:
    """Return a modest worker count, or 1 with a note when joblib is missing.

    One worker fewer than the CPUs but at most 6: every worker holds the head models (about 0.7
    GB), and on the development machine 13 workers were only 1.9 times as fast as one (findings
    A32). The gate measures the efficiency of whatever count is used.
    """
    if importlib.util.find_spec("joblib") is None:
        return 1, "joblib is not installed: running with one worker."
    return max(1, min(6, (os.cpu_count() or 2) - 1)), None


def _saturation_exit(guard: bool, records: list[dict[str, Any]]) -> int:
    """Return 3 and print the problems when a protocol with a saturation guard fails it."""
    if not guard:
        return 0
    report = saturation_report(records)
    if report.ok:
        return 0
    sys.stdout.flush()
    for problem in report.problems:
        print(f"saturation guard: {problem}", file=sys.stderr, flush=True)
    return 3


def _gate_for(
    protocol: Protocol,
    tasks: list[Any],
    old_gate: dict[str, Any] | None,
    workers: int,
    measure: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Reuse the timings of the results file (same worker count), or measure them."""
    if old_gate and old_gate.get("workers") == workers:
        seconds = {int(k): float(v) for k, v in old_gate["seconds_per_task"].items()}
        try:
            hours = estimate_hours(
                tasks,
                seconds,
                reference_duration_s=old_gate["reference_duration_s"],
                workers=workers,
                efficiency=old_gate.get("efficiency", DEFAULT_EFFICIENCY),
                startup_seconds=old_gate.get("startup_seconds", 0.0),
            )
        except ValueError:  # the grid has a network size that was never timed
            pass
        else:
            return {**old_gate, "n_tasks": len(tasks), "estimate_hours": hours}
    return measure(protocol, workers=workers, progress=lambda m: print(m, flush=True))


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., dict[str, Any]] | None = None,
    gate_measure: Callable[..., dict[str, Any]] | None = None,
    crosstalk_fn: Callable[..., dict[str, Any]] | None = None,
) -> int:
    """Run the command line; ``runner``, ``gate_measure`` and ``crosstalk_fn`` are test hooks."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("protocol", help="a protocol name in protocols/ or a path to a YAML file")
    parser.add_argument("--results", type=Path, help="results file (default: by protocol name)")
    parser.add_argument("--workers", type=int, help="processes (default: CPUs - 1, at most 6)")
    parser.add_argument("--dry-run", action="store_true", help="count the tasks and stop")
    parser.add_argument("--gate-only", action="store_true", help="measure the gate and stop")
    parser.add_argument("--from-results", action="store_true", help="render saved results")
    parser.add_argument("--output", type=Path, help="write the rendered summary here")
    parser.add_argument("--force", action="store_true", help="run even over the budget")
    parser.add_argument("--new-file", action="store_true", help="replace an existing results file")
    args = parser.parse_args(argv)

    try:
        protocol = load_protocol(resolve_protocol(args.protocol))
    except (FileNotFoundError, ValueError) as error:
        return _fail(str(error))
    results = args.results or RESULTS_DIR / f"{protocol.name}.jsonl"

    if args.from_results:
        if not results.exists():
            return _fail(f"no results file at {results}.")
        header, records, final = read_results(results)
        text = render_summary(header, records, final)
        if args.output:
            args.output.write_text(text, encoding="utf-8", newline="\n")
        else:
            print(text)
        return _saturation_exit(bool(header.get("saturation_guard")), records)

    tasks = expand(protocol)
    if args.dry_run:
        print(f"{protocol.name}: {len(tasks)} tasks")
        for label, counter in (
            ("scenario", Counter(t.scenario.split(":")[0] for t in tasks)),
            ("modality", Counter(t.modality for t in tasks)),
            ("noise", Counter(t.noise for t in tasks) if len(protocol.noise) > 1 else {}),
        ):
            if not counter:
                continue
            print(f"  by {label}: " + ", ".join(f"{k} {v}" for k, v in counter.items()))
        print(f"  estimators ({len(protocol.estimators)}): {', '.join(protocol.estimators)}")
        return 0

    from dirconn.sweep import crosstalk_matrices, make_runner, measure_gate

    workers = args.workers
    if workers is None:
        workers, note = default_workers()
        if note:
            print(note)

    old_gate = None
    old_header: dict[str, Any] = {}
    if results.exists() and not args.new_file:
        try:
            old_header = read_results(results)[0]
            old_gate = old_header.get("gate")
        except ValueError:
            old_gate = None  # run_protocol reports a file that is not a results file
    gate = _gate_for(protocol, tasks, old_gate, workers, gate_measure or measure_gate)
    estimate = float(gate["estimate_hours"])
    print(
        f"{protocol.name}: {len(tasks)} tasks, estimated {estimate:.2f} h on {workers} worker(s) "
        f"(budget {protocol.budget_hours:g} h)."
    )
    try:
        check_budget(estimate, protocol.budget_hours, force=args.force)
    except ValueError as error:
        return _fail(f"{error} (--force goes ahead)")
    if args.gate_only:
        return 0

    header_extra = None
    if protocol.kind == "region_pairs" and "crosstalk" not in old_header:
        modalities = sorted(set(protocol.modality))
        print(f"crosstalk matrices for {modalities} ...", flush=True)
        header_extra = {"crosstalk": (crosstalk_fn or crosstalk_matrices)(modalities)}
    try:
        summary = run_protocol(
            protocol,
            runner or make_runner(protocol),
            results,
            workers=workers,
            new_file=args.new_file,
            gate=gate,
            header_extra=header_extra,
            progress=lambda m: print(m, flush=True),
        )
    except ValueError as error:
        return _fail(str(error).replace("new_file=True", "--new-file"))
    print(
        f"{summary.n_done} done, {summary.n_skipped} skipped, {summary.n_failed} failed; "
        f"actual {summary.seconds / 3600.0:.2f} h this session, estimated {estimate:.2f} h "
        f"for the whole grid. Results: {results}"
    )
    if summary.n_failed:
        return 1
    return _saturation_exit(protocol.saturation_guard, read_results(results)[1])


if __name__ == "__main__":
    sys.exit(main())

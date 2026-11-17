"""Entry point of the ``dirconn-bids`` command.

``dirconn-bids BIDS_DIR OUTPUT_DIR participant [options]`` runs the participant level:
configuration (``--config`` and ``--set``), optional BIDS validation, input discovery and
`dirconn.workflows.participant.run_participant`. ``dirconn-bids validate-config FILE`` and
``dirconn-bids list-methods`` are subcommands (`dirconn.cli.tools`). The group level,
``--reports-only`` and ``--boilerplate-only`` arrive in dirconn 0.3.0.

The module imports with the core dependencies only, so ``--version`` and ``--help`` work
without the ``app`` extra; the pipeline modules are imported once a run needs them.

Exit codes: 0 every task succeeded; 1 at least one task failed; 2 invalid arguments,
configuration, inputs or output location, a missing ``app`` extra, or a file system error
before the tasks ran; 3 BIDS validation failed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from dirconn.cli.parser import PROG, SUBCOMMANDS, build_parser, build_subparser
from dirconn.cli.tools import (
    config_errors,
    error,
    list_methods,
    load_or_report,
    require_app,
    validate_config,
)

__all__ = ["THREAD_VARIABLES", "main"]

#: Environment variables set from ``--omp-nthreads``.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)

_LATER = "available in dirconn 0.3.0"


def _warn(message: str) -> None:
    print(f"{PROG}: warning: {message}", file=sys.stderr)


def _subcommand(argv: Sequence[str]) -> int:
    args = build_subparser().parse_args(argv)
    if args.command == "validate-config":
        return validate_config(args.config)
    return list_methods()


def _bids_validation(bids_dir: Path) -> int | None:
    """Run ``bids-validator`` when installed; return 3 if it rejects the dataset, 2 on OSError."""
    exe = shutil.which("bids-validator")
    if exe is None:
        _warn("bids-validator not found; the dataset is not validated")
        return None
    try:
        returncode = subprocess.run([exe, str(bids_dir)], check=False).returncode
    except OSError as exc:
        error(f"cannot run bids-validator ({exe}): {exc}")
        return 2
    if returncode != 0:
        error(f"bids-validator rejected {bids_dir} (use --skip-bids-validation to skip)")
        return 3
    return None


def _output_location_error(bids_dir: Path, out_dir: Path) -> str | None:
    """Return why ``out_dir`` may not be used with ``bids_dir``, or ``None`` (resolved paths)."""
    if out_dir == bids_dir:
        return f"the output directory {out_dir} is the BIDS directory"
    derivatives = bids_dir / "derivatives"
    if out_dir.is_relative_to(bids_dir) and not out_dir.is_relative_to(derivatives):
        return (
            f"the output directory {out_dir} lies inside the BIDS directory {bids_dir}; "
            f"use a directory outside it or below {derivatives}"
        )
    return None


def _participant(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Run the participant level; requires the ``app`` extra."""
    from dirconn.io.derivatives import derivative_root
    from dirconn.workflows.config import apply_overrides
    from dirconn.workflows.inputs import InputDiscoveryError, discover
    from dirconn.workflows.participant import ConfigMismatchError, run_participant
    from dirconn.workflows.provenance import Resources

    cfg = load_or_report(args.config)
    if cfg is None:
        return 2
    try:
        cfg = apply_overrides(cfg, args.overrides)
    except ValueError as exc:
        for message in config_errors(exc):
            error(f"--set: {message}")
        return 2

    bids_dir = args.bids_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not bids_dir.is_dir():
        error(f"BIDS directory {bids_dir} does not exist")
        return 2
    problem = _output_location_error(bids_dir, out_dir)
    if problem is not None:
        error(problem)
        return 2
    if args.resource_monitor:
        _warn("--resource-monitor is not implemented in dirconn 0.2.0; ignored")
    if not args.skip_bids_validation:
        code = _bids_validation(bids_dir)
        if code is not None:
            return code

    try:
        units = discover(
            bids_dir,
            level=cfg.input.level,
            datatypes=cfg.input.datatypes,
            description=cfg.input.description,
            participant_labels=args.participant_label,
            session_ids=args.session_id,
            task_ids=args.task_id,
            run_ids=args.run_id,
            filter_file=args.bids_filter_file,
            exclude=derivative_root(out_dir),
        )
    except InputDiscoveryError as exc:
        error(str(exc))
        return 2

    work_dir = (args.work_dir if args.work_dir is not None else out_dir / "work").resolve()
    resources = Resources(
        nprocs=args.nprocs, omp_nthreads=args.omp_nthreads, mem_mb=args.mem_mb, work_dir=work_dir
    )
    try:
        summary = run_participant(
            cfg,
            bids_dir=bids_dir,
            out_dir=out_dir,
            units=units,
            resources=resources,
            argv=[PROG, *argv],
            verbosity=args.verbose,
        )
    except ConfigMismatchError as exc:
        error(str(exc))
        return 2
    except OSError as exc:
        # Task errors stay inside the executor; an OSError here comes from setting up the run.
        error(f"cannot set up the run in {out_dir}: {exc}")
        return 2
    print(
        f"{PROG}: done {summary.done}, skipped {summary.skipped}, failed {summary.failed}; "
        f"run directory {summary.run_dir}",
        file=sys.stderr,
    )
    return summary.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``dirconn-bids`` command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Command line arguments without the program name; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        The exit code: 0 every task succeeded; 1 at least one task failed; 2 invalid
        arguments, configuration, inputs or output location (``OUTPUT_DIR`` may not be
        ``BIDS_DIR`` or lie inside it outside ``BIDS_DIR/derivatives``), a missing ``app``
        extra, or a file system error before the tasks ran; 3 BIDS validation failed.
        argparse errors exit with 2 through ``SystemExit``.

    Notes
    -----
    ``--omp-nthreads`` is written to `THREAD_VARIABLES` for the worker processes. The thread
    pools of this process are already set up (numpy is imported with dirconn), so with
    ``--nprocs 1`` it applies only to libraries that read the variables later.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        return _subcommand(argv)
    args = build_parser().parse_args(argv)

    for name in THREAD_VARIABLES:
        os.environ[name] = str(args.omp_nthreads)

    if args.analysis_level == "group":
        error(f"the group level is {_LATER}")
        return 2
    for flag, value in (
        ("--reports-only", args.reports_only),
        ("--boilerplate-only", args.boilerplate_only),
    ):
        if value:
            error(f"{flag} is {_LATER}")
            return 2
    if not require_app():
        return 2
    return _participant(args, argv)


if __name__ == "__main__":
    raise SystemExit(main())

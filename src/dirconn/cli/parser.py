"""Argument parsers of the ``dirconn-bids`` command.

`build_parser` is the BIDS App interface (``BIDS_DIR OUTPUT_DIR {participant,group}`` and
options); `build_subparser` holds the subcommands in `SUBCOMMANDS`, which take the place of the
positional arguments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dirconn

__all__ = ["PROG", "SUBCOMMANDS", "build_parser", "build_subparser"]

PROG = "dirconn-bids"
#: First arguments that select a subcommand instead of a BIDS App run.
SUBCOMMANDS: tuple[str, ...] = ("validate-config", "list-methods")


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def _add_version(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--version", action="version", version=f"%(prog)s {dirconn.__version__}")


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of a BIDS App run.

    Returns
    -------
    argparse.ArgumentParser
        Positional ``bids_dir``, ``out_dir`` and ``analysis_level`` (``participant`` or
        ``group``) and the options of the BIDS App convention (see ``dirconn-bids --help``).
        Entity filters accept values with or without their ``sub-`` / ``ses-`` / ``task-`` /
        ``run-`` prefix.
    """
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "dirconn BIDS App: directed connectivity of a BIDS dataset. Subcommands: "
            f"{PROG} validate-config FILE, {PROG} list-methods."
        ),
        epilog=(
            "Exit codes: 0 every task succeeded; 1 at least one task failed (the other "
            "outputs are kept); 2 invalid arguments, configuration, inputs or output location, "
            "or a file system error before the tasks ran (nothing run); "
            "3 BIDS validation failed."
        ),
    )
    parser.add_argument("bids_dir", type=Path, help="Root of the BIDS dataset.")
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Output directory; results go to OUT_DIR/dirconn. Outside BIDS_DIR, or below "
        "BIDS_DIR/derivatives.",
    )
    parser.add_argument("analysis_level", choices=("participant", "group"), help="Analysis level.")

    filters = parser.add_argument_group("input selection")
    filters.add_argument(
        "--participant-label",
        nargs="+",
        metavar="LABEL",
        help="Subjects to process (with or without 'sub-'); default all.",
    )
    filters.add_argument(
        "--session-id", nargs="+", metavar="ID", help="Sessions (with or without 'ses-')."
    )
    filters.add_argument(
        "--task-id", nargs="+", metavar="ID", help="Tasks (with or without 'task-')."
    )
    filters.add_argument("--run-id", nargs="+", metavar="ID", help="Runs (with or without 'run-').")
    filters.add_argument(
        "--bids-filter-file",
        type=Path,
        metavar="FILE",
        help="JSON file with lists under 'subject', 'session', 'task', 'run'; intersected with "
        "the other filters.",
    )

    config = parser.add_argument_group("configuration")
    config.add_argument(
        "--config", type=Path, metavar="FILE", help="TOML configuration; default built in."
    )
    config.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="Override one configuration field by its dotted path, e.g. "
        "estimation.methods=gc,pdc or estimation.params.gc.order=5; may be repeated.",
    )

    resources = parser.add_argument_group("resources")
    resources.add_argument(
        "--work-dir",
        type=Path,
        metavar="DIR",
        help="Intermediate files (prepared-series cache); default OUT_DIR/work.",
    )
    resources.add_argument(
        "--nprocs", type=_positive_int, default=1, metavar="N", help="Worker processes (1)."
    )
    resources.add_argument(
        "--omp-nthreads",
        type=_positive_int,
        default=1,
        metavar="N",
        help="Threads per worker for the numerical libraries (1).",
    )
    resources.add_argument(
        "--mem-mb", type=_positive_int, metavar="N", help="Memory limit in MB (recorded only)."
    )
    resources.add_argument(
        "--resource-monitor",
        action="store_true",
        help="Record CPU time and peak memory per step (not implemented in dirconn 0.2.0).",
    )

    other = parser.add_argument_group("other")
    other.add_argument(
        "--skip-bids-validation", action="store_true", help="Do not run bids-validator."
    )
    other.add_argument(
        "--reports-only",
        action="store_true",
        help="Rebuild the reports (sub-XX.html) and the methods boilerplate from the existing "
        "outputs; nothing is recomputed.",
    )
    other.add_argument(
        "--boilerplate-only",
        action="store_true",
        help="Rewrite only the methods boilerplate (logs/CITATION.*) from the existing outputs.",
    )
    other.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="More log output on stderr: -v information, -vv debug messages.",
    )
    _add_version(parser)
    return parser


def build_subparser() -> argparse.ArgumentParser:
    """Return the parser of the subcommands in `SUBCOMMANDS`.

    Returns
    -------
    argparse.ArgumentParser
        Sets ``command`` to the subcommand; ``validate-config`` also sets ``config``.
    """
    parser = argparse.ArgumentParser(prog=PROG, description="dirconn BIDS App subcommands.")
    _add_version(parser)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    validate = sub.add_parser(
        "validate-config", help="Check a TOML configuration and print its hash."
    )
    validate.add_argument("config", type=Path, metavar="FILE", help="TOML configuration.")
    sub.add_parser("list-methods", help="List the connectivity methods.")
    return parser

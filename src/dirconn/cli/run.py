"""Entry point of the ``dirconn-bids`` command.

Only ``--version`` and ``--help`` exist so far; every other invocation reports that the command
is not implemented yet.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import dirconn


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``dirconn-bids`` command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Command line arguments without the program name; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        The process exit code: 0 on success, 2 for an invocation that is not implemented yet.
    """
    parser = argparse.ArgumentParser(prog="dirconn-bids", description="dirconn BIDS App.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {dirconn.__version__}")
    parser.parse_known_args(argv)
    print("dirconn-bids: error: not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

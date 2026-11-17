"""Run-level provenance of a pipeline run: resources, environment, inputs and task counts."""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from dirconn.data.provenance import collect_versions
from dirconn.workflows.executor import RunSummary
from dirconn.workflows.inputs import InputUnit

__all__ = [
    "DIRCONN_VERSION",
    "Resources",
    "git_revision",
    "run_provenance",
    "write_environment_lock",
]

#: ``dirconn.__version__``, read without importing the top-level package (which pulls in the
#: simulation and plotting modules the workflows must not depend on).
DIRCONN_VERSION: str = metadata.version("dirconn")
#: The ``dirconn`` package directory; ``PACKAGE_DIR.parent.parent`` is the checkout root of a
#: source install.
PACKAGE_DIR = Path(__file__).resolve().parents[1]

#: Lock file baked into the Docker image.
IMAGE_LOCK_FILE = Path("/opt/environment-lock.txt")


@dataclass(frozen=True)
class Resources:
    """Compute resources of a run.

    Parameters
    ----------
    nprocs : int
        Number of worker processes.
    omp_nthreads : int
        Threads per process for the numerical libraries.
    mem_mb : int or None
        Memory limit in MB, ``None`` when unspecified.
    work_dir : Path
        Directory for intermediate files (the prepared-series cache).
    """

    nprocs: int
    omp_nthreads: int
    mem_mb: int | None
    work_dir: Path


def git_revision(path: Path) -> str | None:
    """Return the ``HEAD`` commit of the git checkout whose top level is ``path``.

    Parameters
    ----------
    path : Path
        The expected top-level directory of the checkout (for the package itself
        ``PACKAGE_DIR.parent.parent``, the root of a source tree).

    Returns
    -------
    str or None
        The commit hash; ``None`` when ``git rev-parse --show-toplevel`` run in ``path`` names
        another directory (an installed package inside some other repository, or a
        subdirectory), when ``path`` is not in a checkout, when git is missing, or when a
        command fails or takes longer than 5 seconds.
    """
    path = Path(path)

    def git(*args: str) -> str:
        out = subprocess.run(
            ["git", *args],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()

    try:
        top = git("rev-parse", "--show-toplevel")
        if not top or not _same_dir(Path(top), path):
            return None
        revision = git("rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return revision or None


def _same_dir(a: Path, b: Path) -> bool:
    return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def _iso(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat()


def run_provenance(
    *,
    argv: Sequence[str],
    cfg_hash: str,
    analysis_hash: str,
    units: Sequence[InputUnit],
    started: datetime,
    finished: datetime,
    resources: Resources,
    summary: RunSummary,
) -> dict[str, Any]:
    """Describe a finished run.

    Parameters
    ----------
    argv : sequence of str
        The command line.
    cfg_hash, analysis_hash : str
        ``config_hash`` and ``analysis_hash`` of the configuration.
    units : sequence of InputUnit
        The inputs of the run.
    started, finished : datetime
        Start and end of the run; naive values are taken as UTC.
    resources : Resources
        Compute resources.
    summary : RunSummary
        Task counts.

    Returns
    -------
    dict
        JSON-serialisable record with the keys ``command started finished hostname platform
        python versions image git inputs config_hash analysis_hash resources tasks``.
    """
    res = asdict(resources)
    res["work_dir"] = str(resources.work_dir)
    return {
        "command": list(argv),
        "started": _iso(started),
        "finished": _iso(finished),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": collect_versions(),
        "image": {
            "tag": os.environ.get("DIRCONN_IMAGE_TAG"),
            "revision": os.environ.get("DIRCONN_BUILD_REVISION"),
        },
        "git": git_revision(PACKAGE_DIR.parent.parent),
        "inputs": [
            {
                "label": unit.label(),
                "files": [Path(f).as_posix() for f in unit.files],
                "sha256": unit.sha256,
            }
            for unit in units
        ],
        "config_hash": cfg_hash,
        "analysis_hash": analysis_hash,
        "resources": res,
        "tasks": {"done": summary.done, "skipped": summary.skipped, "failed": summary.failed},
    }


def write_environment_lock(root: Path) -> None:
    """Write ``root/environment-lock.txt``.

    Parameters
    ----------
    root : Path
        Derivatives dataset root.

    Notes
    -----
    Inside the Docker image the image's lock file (``/opt/environment-lock.txt``) is copied;
    elsewhere the installed distributions are listed as sorted ``name==version`` lines.
    """
    target = Path(root) / "environment-lock.txt"
    if IMAGE_LOCK_FILE.is_file():
        shutil.copyfile(IMAGE_LOCK_FILE, target)
        return
    lines = sorted(
        {
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in metadata.distributions()
            if dist.metadata["Name"]
        },
        key=str.lower,
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

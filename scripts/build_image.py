"""Build the Docker image of dirconn, labelled with the commit it was built from, and push it.

The tag is ``<version>-<short sha>``; a build from a tree with uncommitted changes gets ``-dirty``
appended and is refused by ``--push``, because the image would not match any commit. ``--push``
needs a ``docker login ghcr.io`` first (a token with ``write:packages``).

Usage::

    python scripts/build_image.py --dry-run     # print the commands, run nothing
    python scripts/build_image.py               # build ghcr.io/hanweiliu-viola/dirconn:<tag>
    python scripts/build_image.py --push        # build, then push the tag
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "ghcr.io/hanweiliu-viola/dirconn"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def image_tag(version: str, sha: str, dirty: bool) -> str:
    """Return the tag of an image built from ``sha`` (Docker tags allow no ``+``)."""
    tag = f"{version}-{sha[:7]}".replace("+", "-")
    return f"{tag}-dirty" if dirty else tag


def build_command(image: str, sha: str, version: str, date: str, dirty: bool) -> list[str]:
    """Return the ``docker build`` command with the provenance build arguments."""
    return [
        "docker", "build",
        "--build-arg", f"GIT_SHA={sha}",
        "--build-arg", f"GIT_DIRTY={int(dirty)}",
        "--build-arg", f"IMAGE={image}",
        "--build-arg", f"BUILD_DATE={date}",
        "--build-arg", f"VERSION={version}",
        "-t", image,
        str(ROOT),
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    """Build (and optionally push) the image; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--push", action="store_true", help="push the tag after building")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = parser.parse_args(argv)

    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    if args.push and dirty:
        print("Refusing to push an image built from uncommitted changes; commit first.")
        return 1
    image = f"{REPOSITORY}:{image_tag(version, sha, dirty)}"
    date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    commands = [build_command(image, sha, version, date, dirty)]
    if args.push:
        commands.append(["docker", "push", image])
    for command in commands:
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    print(image)
    return 0


if __name__ == "__main__":
    sys.exit(main())

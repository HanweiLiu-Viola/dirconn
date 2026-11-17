"""Tests for the run-level provenance helpers."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dirconn.workflows.provenance import PACKAGE_DIR, git_revision

SOURCE_ROOT = PACKAGE_DIR.parent.parent


def test_git_revision_is_none_for_a_plain_or_missing_directory(tmp_path: Path) -> None:
    # tmp_path may lie inside a checkout (the basetemp is in the source tree); it is never the
    # top level, so the top-level check rejects it either way.
    assert git_revision(tmp_path) is None
    assert git_revision(tmp_path / "missing") is None


def test_git_revision_is_none_below_the_top_level() -> None:
    # PACKAGE_DIR is inside the checkout but is not its top level.
    assert git_revision(PACKAGE_DIR) is None


def test_git_revision_is_none_for_another_repository(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=other, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        cwd=other,
        check=True,
    )
    # The repository has a HEAD, so only the top-level check can reject the nested directory
    # (an installed package inside some other repository, say).
    nested = other / "lib" / "site-packages"
    nested.mkdir(parents=True)
    assert git_revision(nested) is None
    head = git_revision(other)
    assert head is not None and re.fullmatch(r"[0-9a-f]{40}", head)


def test_git_revision_of_the_source_tree() -> None:
    if not (SOURCE_ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip("not running from a git checkout")
    revision = git_revision(SOURCE_ROOT)
    assert revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision)

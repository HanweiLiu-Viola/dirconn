"""Fixtures shared by the report tests."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from dirconn.reports.build import build_reportlets


@pytest.fixture(scope="session")
def reportlets_root(pipeline_run: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Dataset root of a copy of ``pipeline_run`` with both subjects' reportlets written once.

    Read-only; a test that changes files copies ``root.parent`` first.
    """
    out = tmp_path_factory.mktemp("reportlets") / "out"
    shutil.copytree(pipeline_run.root.parent, out)
    root = out / "dirconn"
    for subject in ("01", "02"):
        assert build_reportlets(root, subject)
    return root

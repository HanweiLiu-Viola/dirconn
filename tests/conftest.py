"""Shared fixtures."""

import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def bids_tiny() -> Path:
    """Root of the committed tiny BIDS dataset; read-only, never write into it."""
    return FIXTURES / "bids_tiny"


@pytest.fixture
def bids_tiny_copy(bids_tiny: Path, tmp_path: Path) -> Path:
    """A private, writable copy of the tiny BIDS dataset."""
    target = tmp_path / "bids_tiny"
    shutil.copytree(bids_tiny, target)
    return target

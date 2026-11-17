"""Numerical regression of the participant pipeline against a stored golden file.

The golden file is written by ``tools/make_golden.py``; regenerate it only for an intended change
of the numbers.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from dirconn.workflows.log import close_logging

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).parent / "golden" / "bids_tiny_gc_order3.json"


def _load_tool():
    spec = importlib.util.spec_from_file_location("make_golden", ROOT / "tools" / "make_golden.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_golden = _load_tool()


@pytest.fixture(autouse=True)
def _restore_dirconn_logger() -> Iterator[None]:
    logger = logging.getLogger("dirconn")
    state = (list(logger.handlers), logger.level, logger.propagate)
    yield
    close_logging()
    logger.handlers[:] = state[0]
    logger.setLevel(state[1])
    logger.propagate = state[2]


def test_golden_config_is_the_documented_one():
    cfg = make_golden.golden_config()
    assert cfg.estimation.methods == ("gc",)
    assert cfg.estimation.order == 3
    assert cfg.statistics.test == "none"
    assert cfg.prepare.resample_hz is None
    assert cfg.prepare.zscore is True
    assert cfg.controls.time_reversal is False


def test_gc_values_match_golden(bids_tiny_copy: Path, tmp_path: Path):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    values = make_golden.golden_values(bids_tiny_copy, tmp_path / "out")

    assert sorted(values) == sorted(golden)
    for label, pairs in golden.items():
        assert sorted(values[label]) == sorted(pairs), label
        keys = sorted(pairs)
        expected = np.array([pairs[k] for k in keys])
        actual = np.array([values[label][k] for k in keys])
        assert np.allclose(actual, expected, atol=1e-8, rtol=0), (
            label,
            float(np.max(np.abs(actual - expected))),
        )

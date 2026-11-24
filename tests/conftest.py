"""Shared fixtures."""

import logging
import shutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class PipelineRun:
    """One finished participant run of a session pipeline fixture (read-only for tests).

    Attributes
    ----------
    root : Path
        Output dataset root, ``<out>/dirconn``.
    bids : Path
        The private copy of ``bids_tiny`` that was the input.
    cfg : Any
        The ``PipelineConfig`` that was run.
    units : list
        The ``InputUnit`` objects that were run.
    summary : Any
        The ``ParticipantSummary`` of the run.
    """

    root: Path
    bids: Path
    cfg: Any
    units: list[Any]
    summary: Any


def _run_pipeline(
    bids_tiny: Path,
    factory: pytest.TempPathFactory,
    name: str,
    toml: str,
    *,
    only_rest: bool,
    max_units: int | None = None,
) -> PipelineRun:
    """Run ``run_participant`` over a private copy of ``bids_tiny`` and return the result."""
    from dirconn.data.exceptions import DirconnWarning
    from dirconn.workflows.config import load_config
    from dirconn.workflows.inputs import discover
    from dirconn.workflows.log import close_logging
    from dirconn.workflows.participant import run_participant
    from dirconn.workflows.provenance import Resources

    tmp = factory.mktemp(name)
    bids = tmp / "bids_tiny"
    shutil.copytree(bids_tiny, bids)
    config_path = tmp / "config.toml"
    config_path.write_text(toml, encoding="utf-8")
    cfg = load_config(config_path)
    units = discover(
        bids,
        level=cfg.input.level,
        datatypes=cfg.input.datatypes,
        description=cfg.input.description,
    )
    if only_rest:
        units = [u for u in units if u.task == "rest"]
    if max_units is not None:
        units = units[:max_units]
    out = tmp / "out"
    resources = Resources(nprocs=1, omp_nthreads=1, mem_mb=None, work_dir=out / "work")
    logger = logging.getLogger("dirconn")
    state = (list(logger.handlers), logger.level, logger.propagate)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DirconnWarning)
            summary = run_participant(
                cfg,
                bids_dir=bids,
                out_dir=out,
                units=units,
                resources=resources,
                argv=["dirconn", str(bids), str(out), "participant"],
                verbosity=1,
            )
    finally:
        close_logging()
        logger.handlers[:] = state[0]
        logger.setLevel(state[1])
        logger.propagate = state[2]
    return PipelineRun(out / "dirconn", bids, cfg, units, summary)


@pytest.fixture(scope="session")
def pipeline_run(bids_tiny: Path, tmp_path_factory: pytest.TempPathFactory) -> PipelineRun:
    """ROI-level run (five units; gc, pdc, adtf; order chosen by BIC) with ``PIPELINE_CONFIG``.

    Returns a `PipelineRun` (``root`` is the dataset root ``<out>/dirconn``); read-only.
    """
    from tests.reports._configs import PIPELINE_CONFIG

    return _run_pipeline(bids_tiny, tmp_path_factory, "pipeline", PIPELINE_CONFIG, only_rest=False)


@pytest.fixture(scope="session")
def pipeline_run_with_failure(
    bids_tiny: Path, tmp_path_factory: pytest.TempPathFactory
) -> PipelineRun:
    """Like `pipeline_run` with ``FAILURE_CONFIG``: every gc task fails (order 100000).

    Returns a `PipelineRun`; read-only.
    """
    from tests.reports._configs import FAILURE_CONFIG

    return _run_pipeline(bids_tiny, tmp_path_factory, "failure", FAILURE_CONFIG, only_rest=False)


@pytest.fixture(scope="session")
def pipeline_run_epoched(bids_tiny: Path, tmp_path_factory: pytest.TempPathFactory) -> PipelineRun:
    """Sensor-level, event-locked run of the ``rest`` unit(s) with ``EPOCHED_CONFIG``.

    Fixed order, so ``order_selection`` is null; wdtf has p values per time point.
    Returns a `PipelineRun`; read-only.
    """
    from tests.reports._configs import EPOCHED_CONFIG

    return _run_pipeline(bids_tiny, tmp_path_factory, "epoched", EPOCHED_CONFIG, only_rest=True)


@pytest.fixture(scope="session")
def make_pipeline_run(
    bids_tiny: Path, tmp_path_factory: pytest.TempPathFactory
) -> Callable[..., PipelineRun]:
    """Factory ``(name, toml, *, only_rest=False, max_units=None) -> PipelineRun`` for extra runs.

    Each call runs `run_participant` over a new private copy of ``bids_tiny`` (keeping the first
    ``max_units`` units after the ``only_rest`` filter). The environment at call time applies, so a
    test can set ``DIRCONN_IMAGE_TAG`` with ``monkeypatch`` first. Outputs are read-only for tests
    that share them; copy them before modifying.
    """

    def make(
        name: str, toml: str, *, only_rest: bool = False, max_units: int | None = None
    ) -> PipelineRun:
        return _run_pipeline(
            bids_tiny, tmp_path_factory, name, toml, only_rest=only_rest, max_units=max_units
        )

    return make

"""Generate the numerical golden file of the participant pipeline.

Runs `dirconn.workflows.participant.run_participant` on a copy of ``tests/fixtures/bids_tiny``
with conditional Granger causality at order 3, no significance test, no resampling, z-scored
channels and no time-reversal control, and writes the ``desc-gc`` values of every unit as
``{unit label: {"source|target": value}}`` (values as ``repr(float)``, diagonal left out).
``tests/workflows/test_golden.py`` reruns the same pipeline and compares with this file.

Regenerate only when a change of the numbers is intended, and say why in the commit message.

Usage::

    python tools/make_golden.py --out tests/workflows/golden/bids_tiny_gc_order3.json
"""

import argparse
import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from dirconn.data.exceptions import DirconnWarning
from dirconn.data.result import Connectivity
from dirconn.io.derivatives import derivative_root, filename, unit_dir
from dirconn.workflows.config import PipelineConfig, apply_overrides
from dirconn.workflows.inputs import discover
from dirconn.workflows.participant import run_participant
from dirconn.workflows.provenance import Resources

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "bids_tiny"

#: The configuration of the golden run, as ``--set`` overrides of the built-in default.
GOLDEN_OVERRIDES: tuple[str, ...] = (
    "estimation.methods=gc",
    "estimation.order=3",
    "statistics.test=none",
    "prepare.resample_hz=null",
    "prepare.zscore=true",
    "controls.time_reversal=false",
)


def golden_config() -> PipelineConfig:
    """Return the configuration of the golden run."""
    return apply_overrides(PipelineConfig(), GOLDEN_OVERRIDES)


def golden_values(bids_dir: Path, out_dir: Path) -> dict[str, dict[str, float]]:
    """Run the golden pipeline on ``bids_dir`` and return the ``desc-gc`` values per unit.

    Parameters
    ----------
    bids_dir : Path
        A writable copy of the tiny BIDS dataset.
    out_dir : Path
        Output directory of the run (must not hold results of another configuration).

    Returns
    -------
    dict
        ``{unit label: {"source|target": value}}`` for every ordered pair of distinct nodes.
    """
    cfg = golden_config()
    units = discover(
        bids_dir,
        level=cfg.input.level,
        datatypes=cfg.input.datatypes,
        description=cfg.input.description,
    )
    resources = Resources(nprocs=1, omp_nthreads=1, mem_mb=None, work_dir=out_dir / "work")
    with warnings.catch_warnings():
        # order 3 is too low for the fixture on purpose; the fit warnings are expected
        warnings.simplefilter("ignore", DirconnWarning)
        summary = run_participant(
            cfg,
            bids_dir=bids_dir,
            out_dir=out_dir,
            units=units,
            resources=resources,
            argv=["make_golden.py"],
            verbosity=0,
        )
    if summary.failed:
        raise RuntimeError(f"golden run failed: {summary.failures}")
    root = derivative_root(out_dir)
    values: dict[str, dict[str, float]] = {}
    for unit in units:
        entities = unit.entities()
        nc = unit_dir(root, entities, unit.datatype) / filename(
            entities, desc="gc", suffix="conn", ext=".nc"
        )
        conn = Connectivity.from_netcdf(nc)
        matrix = np.asarray(conn.values)[:, :, 0, 0]
        values[unit.label()] = {
            f"{s}|{t}": float(matrix[i, j])
            for i, s in enumerate(conn.source)
            for j, t in enumerate(conn.target)
            if i != j
        }
    return values


def main(argv=None) -> int:
    """Write the golden file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="Golden JSON file to write.")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        bids = Path(tmp) / "bids_tiny"
        shutil.copytree(FIXTURE, bids)
        values = golden_values(bids, Path(tmp) / "out")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # json writes floats as repr(float), which reads back as the same double
    args.out.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(values)} units", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

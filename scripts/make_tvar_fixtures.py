"""Write ``tests/data/tvar_neurrow.npz``: inputs and outputs of the old time-varying MVAR kernel.

The fixtures let the tests of :mod:`dirconn.tvmvar` compare against the numerical reference of the
old package (``neurrow``, never modified) without importing it, so CI does not need it. Every case
starts from **zero coefficients** (``init="zeros"``): the old kernel seeds from a statsmodels fit
and the new one from its own OLS core, so a common explicit start is the only way to compare the
recursions themselves. Rerunning this script reproduces the file exactly.

    python scripts/make_tvar_fixtures.py --neurrow C:/Projects/neurrow-toolkit/src
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "tvar_neurrow.npz"

N_NODES = 3
N_TIMES = 400


def simulate(order: int, switch: int | None, seed: int) -> np.ndarray:
    """Three nodes; node 0 drives node 1, and after ``switch`` node 2 drives node 0 instead."""
    rng = np.random.default_rng(seed)
    first = np.zeros((order, N_NODES, N_NODES))
    first[0] = [[0.5, 0.0, 0.0], [0.5, 0.4, 0.0], [0.0, 0.0, 0.3]]
    if order > 1:
        first[1] = [[-0.2, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]
    second = first.copy()
    second[0, 1, 0] = 0.0
    second[0, 0, 2] = 0.5
    x = np.zeros((N_NODES, N_TIMES))
    for t in range(order, N_TIMES):
        coefs = first if switch is None or t < switch else second
        x[:, t] = sum(coefs[k] @ x[:, t - k - 1] for k in range(order))
        x[:, t] += 0.5 * rng.standard_normal(N_NODES)
    return x


CASES: dict[str, dict] = {
    "dual_o1": dict(order=1, update="dual", forgetting=0.98, burn_in=40, time_step=1),
    "dual_o2_step5": dict(order=2, update="dual", forgetting=0.99, burn_in=40, time_step=5),
    "rls_o2": dict(order=2, update="rls", forgetting=0.97, burn_in=40, time_step=1),
    "dekf_o2_default_noise": dict(order=2, update="dekf", forgetting=0.99, burn_in=40, time_step=1),
    "dekf_o1_explicit_noise": dict(
        order=1,
        update="dekf",
        forgetting=0.98,
        measurement_noise=0.2,
        obs_noise=0.5,
        burn_in=40,
        time_step=1,
    ),
}


def main() -> None:
    """Run the old kernel on every case and write the archive."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--neurrow", type=Path, required=True, help="the old package's src folder")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    sys.path.insert(0, str(args.neurrow))
    from neurrow.connectivity.effective.tv_mvar import fit_kalman_tvar

    arrays: dict[str, np.ndarray] = {}
    for seed, (name, case) in enumerate(CASES.items()):
        x = simulate(case["order"], N_TIMES // 2, seed=100 + seed)
        fit = fit_kalman_tvar(x, init="zeros", standardize=None, **case)
        arrays[f"{name}__x"] = x
        arrays[f"{name}__coefs"] = fit.coefs
        arrays[f"{name}__noise_cov"] = fit.noise_cov
        arrays[f"{name}__sample_index"] = fit.sample_index
    arrays["cases"] = np.array(json.dumps(CASES))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB, {len(CASES)} cases)")


if __name__ == "__main__":
    main()

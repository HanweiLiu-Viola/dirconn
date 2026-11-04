"""Check the torch backend of the KSG estimator on a GPU and time it against the CPU path.

Run on a machine with a CUDA (or XPU / MPS) build of PyTorch::

    pip install -e ".[dev,parallel,gpu]"
    python scripts/gpu_check.py                 # device "cuda"
    python scripts/gpu_check.py --device xpu
    pytest -m gpu                               # the marked tests

It prints, for a few data lengths, the time of one conditional mutual information on the CPU
path (scipy cKDTree) and on the device in float64 and float32, and the differences from the CPU
value; then one multivariate transfer entropy on both paths. float64 must agree within 1e-10;
float32 is expected within about 1 %. Record the measured times before
stating any speed-up.
"""

from __future__ import annotations

import argparse
import platform
import time
import warnings
from functools import partial

import numpy as np


def _time(function, repeats: int) -> tuple[float, float]:
    value = function()  # warm-up (kernel compilation, transfers)
    t0 = time.perf_counter()
    for _ in range(repeats):
        value = function()
    return (time.perf_counter() - t0) / repeats, value


def main() -> None:
    """Time and compare the device against the CPU path; exit 1 on a mismatch."""
    import torch

    from dirconn import DirconnWarning, TimeSeries
    from dirconn._ksg import ksg_cmi
    from dirconn.transfer_entropy import TE

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2000, 10000, 30000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--te-times", type=int, default=5000, help="samples of the TE check")
    args = parser.parse_args()
    device = args.device
    print(f"torch {torch.__version__}, {platform.platform()}")
    if device.startswith("cuda"):
        print(f"device: {torch.cuda.get_device_name(torch.device(device))}")

    rng = np.random.default_rng(0)
    failures = 0
    for n in args.lengths:
        z = rng.standard_normal((n, 4))
        x = rng.standard_normal(n) + 0.5 * z[:, 0]
        y = 0.4 * x + rng.standard_normal(n) + 0.5 * z[:, 1]
        cpu_s, cpu = _time(partial(ksg_cmi, x, y, z), args.repeats)
        print(f"n = {n}: cpu {cpu_s:.3f} s", flush=True)
        for dtype, tolerance in (("float64", 1e-10), ("float32", 0.01 * abs(cpu) + 1e-3)):
            dev_s, value = _time(
                partial(ksg_cmi, x, y, z, device=device, dtype=dtype), args.repeats
            )
            ok = abs(value - cpu) <= tolerance
            failures += not ok
            print(
                f"  {device} {dtype}: {dev_s:.3f} s ({cpu_s / dev_s:.1f}x), "
                f"difference {abs(value - cpu):.2e} {'ok' if ok else 'FAIL'}",
                flush=True,
            )

    # one multivariate TE on a chain of four channels
    n_times = args.te_times
    e = rng.standard_normal((4, n_times))
    s = np.zeros((4, n_times))
    for t in range(1, n_times):
        s[0, t] = 0.5 * s[0, t - 1] + e[0, t]
        for c in range(1, 4):
            s[c, t] = 0.3 * s[c, t - 1] + 0.6 * s[c - 1, t - 1] + e[c, t]
    data = TimeSeries.from_array(s, 100.0)
    params = {"max_order": 3, "max_delay": 2}
    params.update({f"n_perm_{t}": 49 for t in ("max_stat", "min_stat", "omnibus", "max_seq")})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirconnWarning)
        t0 = time.perf_counter()
        cpu = TE(**params).fit(data)
        cpu_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        gpu = TE(device=device, **params).fit(data)
        gpu_s = time.perf_counter() - t0
    difference = float(np.nanmax(np.abs(gpu.values - cpu.values)))
    ok = difference <= 1e-10 and (
        cpu.provenance.details["selected_sources"] == gpu.provenance.details["selected_sources"]
    )
    failures += not ok
    print(
        f"multivariate TE, 4 channels x {n_times}: cpu {cpu_s:.0f} s, {device} {gpu_s:.0f} s, "
        f"largest difference {difference:.2e} {'ok' if ok else 'FAIL'}"
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

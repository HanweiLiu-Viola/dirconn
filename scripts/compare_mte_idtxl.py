r"""Compare dirconn's multivariate transfer entropy with IDTxl's ``MultivariateTE``.

Three steps, the middle one inside a container that has IDTxl 1.6.1 and JIDT (the image
``neurrow-toolkit:0.2.0``)::

    python scripts/compare_mte_idtxl.py export <dir>      # dirconn environment: write series.npz
    docker run --rm -v <dir>:/work -v C:/Projects/dirconn/scripts:/scripts -w /work \\
        neurrow-toolkit:0.2.0 python /scripts/compare_mte_idtxl.py idtxl /work
    python scripts/compare_mte_idtxl.py compare <dir>     # dirconn environment: compare

Both use the same candidates (target lags 1..MAX_ORDER, source lags 1..MAX_DELAY), the same rows
(from the longest lag on), the KSG estimator with 4 neighbours on standardised variables, and the
same numbers of permutations. The permutations themselves differ, so p-values differ and only the
selection of strong links is expected to agree. Where both select the same variables, the omnibus
transfer entropy is the same conditional mutual information and has to agree to rounding.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

MAX_ORDER = 4
MAX_DELAY = 3
N_PERM = 99
N_TIMES = 3000


def _datasets() -> dict[str, np.ndarray]:
    """Three systems with known links, each (n_channels, n_times)."""
    from dirconn.simulation import coupled_nonlinear

    out = {}
    rng = np.random.default_rng(0)
    x = np.zeros((3, N_TIMES))
    e = rng.standard_normal((3, N_TIMES))
    for t in range(1, N_TIMES):  # X -> Y -> Z
        x[0, t] = 0.5 * x[0, t - 1] + e[0, t]
        x[1, t] = 0.3 * x[1, t - 1] + 0.7 * x[0, t - 1] + e[1, t]
        x[2, t] = 0.3 * x[2, t - 1] + 0.7 * x[1, t - 1] + e[2, t]
    out["chain"] = x
    rng = np.random.default_rng(1)
    x = np.zeros((3, N_TIMES))
    e = rng.standard_normal((3, N_TIMES))
    for t in range(2, N_TIMES):  # Z -> X (lag 1), Z -> Y (lag 2)
        x[2, t] = 0.6 * x[2, t - 1] + e[2, t]
        x[0, t] = 0.8 * x[2, t - 1] + e[0, t]
        x[1, t] = 0.8 * x[2, t - 2] + e[1, t]
    out["common_driver"] = x
    data, _ = coupled_nonlinear(N_TIMES, sfreq=100.0, seed=0)
    out["quadratic"] = np.asarray(data.data)[0]
    return out


def export(folder: Path) -> None:
    """Write the series of the three systems to ``folder/series.npz``."""
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "series.npz", **_datasets())
    print(f"written {folder / 'series.npz'}")


def run_idtxl(folder: Path) -> None:
    """Run IDTxl's MultivariateTE on every series; write ``idtxl_mte.json``."""
    import math

    np.math = math  # type: ignore[attr-defined]  # IDTxl 1.6.1 still calls np.math (numpy < 2)
    from idtxl.data import Data
    from idtxl.multivariate_te import MultivariateTE

    settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "kraskov_k": 4,
        "normalise": True,
        "noise_level": 0,
        "theiler_t": 0,
        "algorithm_num": 1,
        "max_lag_sources": MAX_DELAY,
        "min_lag_sources": 1,
        "max_lag_target": MAX_ORDER,
        "tau_sources": 1,
        "tau_target": 1,
        "n_perm_max_stat": N_PERM,
        "n_perm_min_stat": N_PERM,
        "n_perm_omnibus": N_PERM,
        "n_perm_max_seq": N_PERM,
        "alpha_max_stat": 0.05,
        "alpha_min_stat": 0.05,
        "alpha_omnibus": 0.05,
        "alpha_max_seq": 0.05,
        # one record: IDTxl shuffles the samples in time (dirconn shifts them circularly)
        "permute_in_time": True,
        "perm_type": "random",
        "verbose": False,
    }
    series = np.load(folder / "series.npz")
    out = {}
    for name in series.files:
        t0 = time.perf_counter()
        data = Data(series[name], dim_order="ps", normalise=False)
        results = MultivariateTE().analyse_network(settings, data)
        targets = {}
        for j in range(series[name].shape[0]):
            r = results.get_single_target(j, fdr=False)
            targets[j] = {
                "target_past": sorted(int(lag) for _, lag in r["selected_vars_target"]),
                "sources": sorted([int(p), int(lag)] for p, lag in r["selected_vars_sources"]),
                "omnibus_te": None if r["omnibus_te"] is None else float(r["omnibus_te"]),
            }
        out[name] = {"targets": targets, "seconds": time.perf_counter() - t0}
        print(f"{name}: {out[name]['seconds']:.0f} s", flush=True)
    (folder / "idtxl_mte.json").write_text(json.dumps(out, indent=1))


def compare(folder: Path) -> None:
    """Run dirconn on the same series and print the selections side by side."""
    import warnings

    from dirconn import DirconnWarning, TimeSeries
    from dirconn._ksg import ksg_cmi
    from dirconn.transfer_entropy import TE

    reference = json.loads((folder / "idtxl_mte.json").read_text())
    series = np.load(folder / "series.npz")
    start = max(MAX_ORDER, MAX_DELAY)
    for name in series.files:
        x = series[name]
        estimator = TE(
            max_order=MAX_ORDER,
            max_delay=MAX_DELAY,
            n_perm_max_stat=N_PERM,
            n_perm_min_stat=N_PERM,
            n_perm_omnibus=N_PERM,
            n_perm_max_seq=N_PERM,
        )
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DirconnWarning)
            details = estimator.fit(TimeSeries.from_array(x, 100.0)).provenance.details
        seconds = time.perf_counter() - t0
        print(f"== {name}: dirconn {seconds:.0f} s, IDTxl {reference[name]['seconds']:.0f} s")
        for j in range(x.shape[0]):
            ours = {
                "target_past": sorted(details["selected_target_lags"][j]),
                "sources": sorted(details["selected_sources"][j]),
            }
            theirs = reference[name]["targets"][str(j)]
            same = ours["target_past"] == theirs["target_past"]
            same = same and ours["sources"] == theirs["sources"]
            line = f"target {j}: dirconn {ours}, IDTxl {theirs['target_past']} {theirs['sources']}"
            print(("same  " if same else "DIFF  ") + line)
            if same and ours["sources"]:
                # the omnibus TE of the common selection, from dirconn's estimator
                rows = np.arange(start, x.shape[-1])
                present = x[j, rows]
                past = np.stack([x[j, rows - lag] for lag in ours["target_past"]], axis=1)
                sources = np.stack([x[c, rows - lag] for c, lag in ours["sources"]], axis=1)
                value = ksg_cmi(present, sources, past)
                print(
                    f"      omnibus TE dirconn {value:.12f}, IDTxl {theirs['omnibus_te']:.12f}, "
                    f"difference {abs(value - theirs['omnibus_te']):.2e}"
                )


if __name__ == "__main__":
    step, where = sys.argv[1], Path(sys.argv[2])
    {"export": export, "idtxl": run_idtxl, "compare": compare}[step](where)

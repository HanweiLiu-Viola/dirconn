"""Direction recovery on constructed pairs in real EEG background (the full study).

Takes random pairs of channels of the MNE sample EEG recording and non-overlapping 40 s windows,
builds ``source -> target`` pairs with a 20 ms delay and an 8 to 12 Hz influence
(:func:`dirconn.benchmark.constructed_pair`) and prints, for several estimators and model orders,
how often the known direction is recovered. Needs the sample dataset (see
``dirconn.datasets.require_sample``). Takes about two minutes.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np

from dirconn import DTF, GC, GPDC, Estimator
from dirconn.benchmark import constructed_pair, evaluate_pair, real_eeg_series
from dirconn.timeseries import TimeSeries

DURATION = 40.0  # seconds; six non-overlapping slots fit into the 278 s recording
N_SLOTS = 6
N_PAIRS = 24
BAND = (8.0, 12.0)
DELAY_MS = 20.0


def draw_pairs(real: TimeSeries, n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Draw ``n`` channel pairs and two different time slots for each."""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        channels = rng.choice(real.names, size=2, replace=False)
        slots = rng.choice(N_SLOTS, size=2, replace=False)
        pairs.append(
            {
                "source_channel": str(channels[0]),
                "noise_channel": str(channels[1]),
                "source_start": float(slots[0] * DURATION),
                "noise_start": float(slots[1] * DURATION),
                "duration": DURATION,
            }
        )
    return pairs


def configurations() -> list[tuple[str, Estimator, tuple[float, float] | None]]:
    """Return the estimators compared, with the band each one is scored in."""
    return [
        ("GC order 8", GC(order=8), None),
        ("GC order 32", GC(order=32), None),
        ("GC order 64", GC(order=64), None),
        ("GC bic <= 64", GC(order="bic", max_order=64), None),
        ("DTF order 64", DTF(order=64), BAND),
        ("gPDC order 64", GPDC(order=64), BAND),
    ]


def main() -> None:
    """Run the study and print one line per estimator and strength."""
    warnings.simplefilter("ignore")
    start = time.time()
    real = real_eeg_series()
    print(f"{real.n_channels} channels, {real.duration:.0f} s at {real.sfreq:g} Hz")
    pairs = draw_pairs(real, N_PAIRS)
    for strength in (1.0, 3.0):
        print(f"== strength {strength:g}")
        series = [
            constructed_pair(real, delay_ms=DELAY_MS, strength=strength, band=BAND, **pair)
            for pair in pairs
        ]
        for name, estimator, band in configurations():
            out = [evaluate_pair(estimator, s, truth, band=band) for s, truth in series]
            accuracy = float(np.mean([o["direction_accuracy"] for o in out]))
            ratio = float(
                np.median(
                    [o["scores"]["forward"] / max(o["scores"]["reverse"], 1e-12) for o in out]
                )
            )
            order = float(np.median([o["order"] or 0 for o in out]))
            print(
                f"  {name:<14} direction accuracy {accuracy:.2f} "
                f"({round(accuracy * len(out))}/{len(out)}) | median forward/reverse {ratio:.1f} "
                f"| median order {order:g}"
            )
    print(f"total {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()

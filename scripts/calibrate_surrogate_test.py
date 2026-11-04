"""Type I error of the surrogate test on independent channels (the full calibration experiment).

Prints, per surrogate generator, the number of p-values, rejections at 0.05 and 0.10 against the
expected numbers, and the mean p-value. Takes about half a minute.
"""

from __future__ import annotations

import warnings

import numpy as np

from dirconn import GC, TimeSeries
from dirconn.surrogate_test import surrogate_test
from dirconn.surrogates import CircularShift, PhaseRandomize

N_DATASETS = 100
N_SURROGATES = 39  # p <= 0.05 and p <= 0.10 then have probability exactly 0.05 and 0.10


def independent_data(seed: int) -> TimeSeries:
    """Two independent AR(1) channels, two epochs."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 2, 150))
    for t in range(1, 150):
        x[..., t] += 0.5 * x[..., t - 1]
    return TimeSeries.from_array(x, 100.0, segmentation="epoched")


def main() -> None:
    """Run the experiment and print one line per generator."""
    warnings.simplefilter("ignore")
    off_diagonal = ~np.eye(2, dtype=bool)
    for generator in (PhaseRandomize(), CircularShift(min_shift=10)):
        p = np.concatenate(
            [
                surrogate_test(
                    GC(order=2), independent_data(s), generator, n_surrogates=N_SURROGATES, seed=s
                )
                .pvalues[off_diagonal]
                .ravel()
                for s in range(N_DATASETS)
            ]
        )
        print(
            f"{generator.name}: {p.size} p-values, "
            f"rejections at 0.05: {(p <= 0.05).sum()} (expected {0.05 * p.size:.0f}), "
            f"at 0.10: {(p <= 0.10).sum()} (expected {0.10 * p.size:.0f}), "
            f"mean p {p.mean():.3f}"
        )


if __name__ == "__main__":
    main()

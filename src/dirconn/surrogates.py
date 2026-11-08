"""Surrogate data: copies of a recording that keep some structure and destroy the rest.

Each generator keeps a defined property of the data (power spectra, amplitude distribution,
autocorrelation, trial content) and removes what is being tested, the coupling between channels.
The test that results from comparing the data with its surrogates answers a narrower question
than "is there connectivity": it rejects the null named in ``generator.null`` and nothing more.
``generator.limitations`` lists how that null can be rejected for reasons other than coupling.

Data with ``valid`` or ``predict`` masks (see :class:`~dirconn.TimeSeries`) can only be shuffled by
whole trials: :class:`TrialShuffle` carries the masks with the trials. The generators that move
samples within a trial (phase randomisation, AAFT, IAAFT, circular shifts, block shuffling) would
mix invalid samples into valid ones and refuse masked data.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from dirconn.exceptions import DirconnWarning
from dirconn.rng import _as_count
from dirconn.timeseries import TimeSeries


class SurrogateGenerator(ABC):
    """Base class: one surrogate per call, driven by an explicit random generator.

    Attributes
    ----------
    name : str
        Short identifier, used in provenance records.
    null : str
        The null hypothesis that a test against these surrogates examines.
    limitations : tuple of str
        Ways in which that null can fail without any coupling between the channels.
    """

    name: str
    null: str
    limitations: tuple[str, ...]
    supports_mask: bool = False

    def _check_masks(self, data: TimeSeries) -> None:
        """Raise ``ValueError`` for masked data unless the generator carries the masks."""
        if data.has_mask and not self.supports_mask:
            raise ValueError(
                f"The {self.name} surrogate does not support data with valid or predict masks: "
                "it moves samples within a trial, so gaps and predicted samples would no longer "
                "line up. Use TrialShuffle, which keeps whole trials with their masks."
            )

    def prepare(self, data: TimeSeries) -> TimeSeries:
        """Return the data the test is computed on; surrogates are generated from this.

        Most generators keep the data as they are. A generator that re-segments the record (see
        :class:`BlockShuffle`) returns the re-segmented data here, so that the observed result and
        the surrogates are computed on identically shaped input.
        """
        return data

    @abstractmethod
    def generate_one(self, data: TimeSeries, rng: np.random.Generator) -> TimeSeries:
        """Return one surrogate of ``data`` with the same shape and metadata."""

    def generate(self, data: TimeSeries, rngs: Sequence[np.random.Generator]) -> list[TimeSeries]:
        """Return one surrogate per generator, in the order of ``rngs``."""
        rngs = list(rngs)
        if not rngs:
            raise ValueError("Pass at least one random generator.")
        return [self.generate_one(data, rng) for rng in rngs]

    def params(self) -> dict[str, Any]:
        """Return the constructor parameters that change the result."""
        return {}

    def describe(self) -> dict[str, Any]:
        """Return the name, null hypothesis and parameters for a provenance record."""
        return {"name": self.name, "null": self.null, "params": self.params()}

    def __repr__(self) -> str:
        """Return the class name with its parameters."""
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({args})"


def limitations_table(generators: Sequence[SurrogateGenerator] | None = None) -> str:
    """Return the null hypothesis and the limitations of each generator as a Markdown table.

    The table in ``docs/surrogates.md`` is this output; a test keeps the two identical.
    """
    if generators is None:
        # The texts do not depend on the block size, so any valid size describes BlockShuffle.
        generators = [
            PhaseRandomize(),
            AAFT(),
            IAAFT(),
            CircularShift(),
            TrialShuffle(),
            BlockShuffle(block_size=1),
        ]
    rows = [
        "| Surrogate | Null hypothesis | How the null can be rejected without coupling |",
        "|---|---|---|",
    ]
    for g in generators:
        limits = "<br>".join(f"{i}. {text}" for i, text in enumerate(g.limitations, start=1))
        rows.append(f"| `{g.name}` | {g.null} | {limits} |")
    return "\n".join(rows)


def _shuffle_units(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the first axis of every channel independently; ``values`` is (units, channels, t)."""
    out = np.empty_like(values)
    for ch in range(values.shape[1]):
        out[:, ch] = values[rng.permutation(values.shape[0]), ch]
    return out


def _with_data(data: TimeSeries, values: np.ndarray) -> TimeSeries:
    return replace(data, data=values)


def _ranks(x: np.ndarray) -> np.ndarray:
    """Return the rank of every sample along the last axis (ties broken by position)."""
    order = np.argsort(x, axis=-1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(x.shape[-1]) * np.ones_like(order), axis=-1)
    return ranks


def _randomize_phases(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Give every row independent uniform Fourier phases; the amplitudes stay exactly."""
    n_times = x.shape[-1]
    spectrum = np.fft.rfft(x, axis=-1)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
    randomized = np.abs(spectrum) * np.exp(1j * phases)
    # The mean and, for even lengths, the Nyquist term must stay real, and keeping them as they
    # are also keeps their sign (taking the magnitude would flip a negative mean).
    randomized[..., 0] = spectrum[..., 0]
    if n_times % 2 == 0:
        randomized[..., -1] = spectrum[..., -1]
    return np.fft.irfft(randomized, n=n_times, axis=-1)


def _aaft(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Gaussian values arranged in the rank order of the data carry the data's linear structure
    # in a Gaussian marginal; randomizing their phases is then legitimate.
    gaussian = np.sort(rng.standard_normal(x.shape), axis=-1)
    gaussianized = np.take_along_axis(gaussian, _ranks(x), axis=-1)
    randomized = _randomize_phases(gaussianized, rng)
    return np.take_along_axis(np.sort(x, axis=-1), _ranks(randomized), axis=-1)


class PhaseRandomize(SurrogateGenerator):
    """Randomize the Fourier phases of every channel and trial independently.

    The amplitude spectrum, and with it the autocorrelation and the mean, are kept exactly. All
    structure between channels is destroyed.
    """

    name = "phase_randomize"
    null = "connectivity is explained by individual-channel power spectra"
    limitations = (
        "Assumes each channel is a stationary linear Gaussian process; amplitude fluctuations "
        "over time and non-Gaussian amplitude distributions are not kept.",
        "Destroys phase relations within a channel (bursts, waveform shape, cross-frequency "
        "coupling), so a channel with such structure can look significant for that reason alone.",
        "A trial-locked evoked response is also scrambled; for epoched data this can inflate "
        "significance when the coupling is driven by the evoked response.",
    )

    def generate_one(self, data: TimeSeries, rng: np.random.Generator) -> TimeSeries:
        """Return one phase-randomized copy of ``data``."""
        self._check_masks(data)
        return _with_data(data, _randomize_phases(data.data, rng))


class AAFT(SurrogateGenerator):
    """Amplitude-adjusted Fourier transform surrogate.

    Keeps the amplitude distribution of each channel and trial exactly, and its power spectrum
    approximately (the rank reordering at the end perturbs it).
    """

    name = "aaft"
    null = "connectivity is explained by spectra and marginal amplitude distributions"
    limitations = (
        "The spectrum is matched only approximately and the error is systematically biased "
        "towards a flatter spectrum; IAAFT matches it better.",
        "Assumes the data are a monotone static transformation of a linear Gaussian process.",
        "Destroys phase relations within a channel and trial-locked evoked responses, like "
        "phase randomization.",
    )

    def generate_one(self, data: TimeSeries, rng: np.random.Generator) -> TimeSeries:
        """Return one AAFT copy of ``data``."""
        self._check_masks(data)
        return _with_data(data, _aaft(data.data, rng))


class IAAFT(SurrogateGenerator):
    """Iterative amplitude-adjusted Fourier transform surrogate.

    Alternates between imposing the original amplitude spectrum and the original amplitude
    distribution until the rank order stops changing. The distribution is exact; the spectrum
    is as close as the iteration gets. A run that reaches ``n_iter`` first warns.

    Parameters
    ----------
    n_iter : int
        Maximum number of iterations per call, at least 1.
    """

    name = "iaaft"
    null = AAFT.null
    limitations = (
        "Matches the spectrum only up to the iteration's residual error, and the residual is "
        "not zero even at convergence.",
        "The iteration can stop at a rank-order fixed point or reach n_iter without "
        "converging; the second case is reported as a warning.",
        "Same stationarity and phase-structure assumptions as AAFT; evoked responses are "
        "scrambled.",
    )

    def __init__(self, n_iter: int = 100) -> None:
        self.n_iter = _as_count(n_iter, "n_iter", minimum=1)

    def params(self) -> dict[str, Any]:
        """Return ``{"n_iter": ...}``."""
        return {"n_iter": self.n_iter}

    def generate_one(self, data: TimeSeries, rng: np.random.Generator) -> TimeSeries:
        """Return one IAAFT copy of ``data``."""
        self._check_masks(data)
        x = data.data
        sorted_x = np.sort(x, axis=-1)
        target_amplitude = np.abs(np.fft.rfft(x, axis=-1))
        current = _aaft(x, rng)
        converged = False
        for _ in range(self.n_iter):
            phases = np.angle(np.fft.rfft(current, axis=-1))
            adjusted = np.fft.irfft(target_amplitude * np.exp(1j * phases), n=x.shape[-1], axis=-1)
            updated = np.take_along_axis(sorted_x, _ranks(adjusted), axis=-1)
            converged = bool(np.array_equal(updated, current))
            current = updated
            if converged:
                break
        if not converged:
            warnings.warn(
                f"IAAFT did not converge within n_iter={self.n_iter} iterations; the surrogate "
                "matches the original spectrum less well. Increase n_iter or use AAFT.",
                DirconnWarning,
                stacklevel=2,
            )
        return _with_data(data, current)


class TrialShuffle(SurrogateGenerator):
    """Permute the trials of every channel independently.

    Each channel keeps whole trials but the trial that goes with trial ``k`` of another channel
    is drawn at random, so any trial-by-trial relation between channels is destroyed while
    everything inside a trial is kept. Needs at least two trials.

    Masks travel with the trials. A surrogate trial combines samples of different trials, one per
    channel, so a time point is valid only where it is valid in every trial that a channel
    contributes, and likewise for ``predict``. With masks that differ between trials the
    surrogates therefore have fewer regression rows than the data; with the same mask in every
    trial they have the same.
    """

    name = "trial_shuffle"
    supports_mask = True
    null = "there is no stable trial-to-trial coupling"
    limitations = (
        "Coupling that comes from a shared evoked response is kept in every surrogate, so the "
        "test cannot detect it; it asks only about trial-specific (induced) coupling.",
        "With few trials the number of distinct surrogates is small (n_trials! per channel "
        "beyond the first), which limits the smallest attainable p-value, and a random "
        "permutation may leave a channel's order unchanged.",
        "Needs trials that are exchangeable: a drift or learning effect across trials makes "
        "the shuffled data differ from the original for reasons unrelated to coupling.",
    )

    def generate_one(self, data: TimeSeries, rng: np.random.Generator) -> TimeSeries:
        """Return one trial-shuffled copy of ``data``."""
        if data.n_trials < 2:
            raise ValueError(
                "TrialShuffle needs at least 2 trials to shuffle, got "
                f"{data.n_trials} (continuous data have a single trial)."
            )
        if not data.has_mask:
            return _with_data(data, _shuffle_units(data.data, rng))
        values = np.empty_like(data.data)
        valid = np.ones((data.n_trials, data.n_times), dtype=bool)
        predict = np.ones((data.n_trials, data.n_times), dtype=bool)
        for ch in range(data.n_channels):
            order = rng.permutation(data.n_trials)  # the same draws as _shuffle_units
            values[:, ch] = data.data[order, ch]
            if data.valid is not None:
                valid &= data.valid[order]
            if data.predict is not None:
                predict &= data.predict[order]
        return replace(data, data=values, valid=valid, predict=predict)


class CircularShift(SurrogateGenerator):
    """Rotate every channel and trial by its own random number of samples.

    Autocorrelation, spectrum and amplitude distribution of each series are kept (apart from
    the one seam where the end joins the start); the alignment between channels is destroyed.

    Parameters
    ----------
    min_shift : int
        Smallest allowed rotation in samples, in either direction, at least 1. Choose it longer
        than the autocorrelation time so the shifted series is not still aligned with the
        original. Needs ``n_times >= 2 * min_shift``.
    """

    name = "circular_shift"
    null = "between-channel coupling is explained by within-channel autocorrelation"
    limitations = (
        "The seam where the end of a trial is joined to its start is a discontinuity that "
        "affects model-based estimators such as MVAR fits; long trials make it negligible.",
        "Assumes stationarity within a trial: a slow trend or a change of variance is moved "
        "to another place in the trial, which changes the cross-channel statistics.",
        "A trial-locked evoked response is shifted away from its latency in every channel, so "
        "coupling that is driven by a shared evoked response is destroyed with it.",
    )

    def __init__(self, min_shift: int = 1) -> None:
        self.min_shift = _as_count(min_shift, "min_shift", minimum=1)

    def params(self) -> dict[str, Any]:
        """Return ``{"min_shift": ...}``."""
        return {"min_shift": self.min_shift}

    def generate_one(self, data: TimeSeries, rng: np.random.Generator) -> TimeSeries:
        """Return one circularly shifted copy of ``data``."""
        self._check_masks(data)
        n_times = data.n_times
        if n_times < 2 * self.min_shift:
            raise ValueError(
                f"n_times={n_times} is too short for min_shift={self.min_shift}: it needs "
                f"n_times >= 2 * min_shift."
            )
        shifts = rng.integers(
            self.min_shift, n_times - self.min_shift + 1, size=data.data.shape[:2]
        )
        source = (np.arange(n_times) - shifts[..., np.newaxis]) % n_times
        return _with_data(data, np.take_along_axis(data.data, source, axis=-1))


class BlockShuffle(SurrogateGenerator):
    """Cut the record into blocks and permute the blocks of every channel independently.

    The data are cut into non-overlapping blocks of ``block_size`` samples (every trial of epoched
    data is cut separately, and samples after the last whole block are dropped); the blocks become
    the trials of the prepared data, so a model fitted on them never regresses across a block
    boundary. Each channel then gets its own random order of the blocks. Everything inside a block
    is kept; the alignment of the blocks between channels is destroyed. This is the surrogate for
    continuous data when circular shifting is unsuitable.

    Use it through :func:`dirconn.surrogate_test`, which calls :meth:`prepare` first so that the
    observed result is computed on the same blocks.

    Parameters
    ----------
    block_size : int
        Block length in samples, at least 1. Choose it longer than the memory of the process
        (model order, autocorrelation time); the data must contain at least two blocks.
    """

    name = "block_shuffle"
    null = (
        "between-channel coupling is explained by within-block structure "
        "(blocks are exchangeable within a channel)"
    )
    limitations = (
        "The observed result is computed on the blocks too (as separate trials), so a model "
        "loses the regression rows across block boundaries; with blocks short relative to the "
        "model order this costs data and changes the estimate.",
        "Assumes stationarity and exchangeable blocks: a drift, a change of variance or a "
        "block-locked event makes shuffled blocks differ from the original for reasons unrelated "
        "to coupling.",
        "Dependence longer than a block is destroyed in the surrogates, so the null also "
        "includes 'no coupling on time scales longer than block_size'; too short a block "
        "inflates significance.",
        "A trial-locked evoked response is moved to other blocks in some channels and not in "
        "others, so coupling driven by it is destroyed with it.",
    )

    def __init__(self, block_size: int) -> None:
        self.block_size = _as_count(block_size, "block_size", minimum=1)

    def params(self) -> dict[str, Any]:
        """Return ``{"block_size": ...}``."""
        return {"block_size": self.block_size}

    def prepare(self, data: TimeSeries) -> TimeSeries:
        """Return the blocks of ``data`` as the trials of an epoched series."""
        self._check_masks(data)
        size = self.block_size
        per_trial = data.n_times // size
        if per_trial * data.n_trials < 2:
            raise ValueError(
                f"block_size={size} gives {per_trial * data.n_trials} block(s) from "
                f"{data.n_trials} trial(s) of {data.n_times} samples; block shuffling needs "
                "at least two blocks."
            )
        used = per_trial * size
        blocks = (
            data.data[:, :, :used]
            .reshape(data.n_trials, data.n_channels, per_trial, size)
            .transpose(0, 2, 1, 3)
            .reshape(data.n_trials * per_trial, data.n_channels, size)
        )
        return TimeSeries(blocks, data.sfreq, data.channels, "epoched", 0.0)

    def generate_one(self, data: TimeSeries, rng: np.random.Generator) -> TimeSeries:
        """Return one surrogate of data that went through :meth:`prepare`."""
        self._check_masks(data)
        if data.segmentation != "epoched" or data.n_times != self.block_size:
            raise ValueError(
                f"BlockShuffle expects blocks of {self.block_size} samples as trials; pass the "
                "data through prepare() first (surrogate_test does this)."
            )
        return _with_data(data, _shuffle_units(data.data, rng))

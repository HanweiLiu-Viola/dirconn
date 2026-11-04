import warnings

import numpy as np
import pytest

from dirconn import DirconnWarning, TimeSeries
from dirconn.channels import ChannelMeta
from dirconn.rng import RNGController
from dirconn.surrogates import (
    AAFT,
    IAAFT,
    BlockShuffle,
    CircularShift,
    PhaseRandomize,
    SurrogateGenerator,
    TrialShuffle,
)

ALL = [PhaseRandomize, AAFT, IAAFT, CircularShift, TrialShuffle]


def ar1_series(n_trials=6, n_channels=3, n_times=200, seed=0, phi=0.8):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_trials, n_channels, n_times))
    x = np.empty_like(noise)
    x[..., 0] = noise[..., 0]
    for t in range(1, n_times):
        x[..., t] = phi * x[..., t - 1] + noise[..., t]
    channels = tuple(ChannelMeta(f"c{i}", type="eeg") for i in range(n_channels))
    return TimeSeries(x, 100.0, channels, "epoched", tmin=-0.5)


def skewed_series(**kwargs):
    ts = ar1_series(**kwargs)
    return TimeSeries(np.exp(ts.data / 2.0), ts.sfreq, ts.channels, ts.segmentation, ts.tmin)


def amplitude(data):
    return np.abs(np.fft.rfft(data, axis=-1))


def circular_autocorrelation(data):
    spectrum = np.abs(np.fft.rfft(data, axis=-1)) ** 2
    return np.fft.irfft(spectrum, n=data.shape[-1], axis=-1)


def rng(seed=0):
    return np.random.default_rng(seed)


@pytest.mark.parametrize("cls", ALL)
class TestEveryGenerator:
    def test_it_is_a_generator_with_a_stated_null_and_limits(self, cls):
        generator = cls()
        assert isinstance(generator, SurrogateGenerator)
        assert generator.name and isinstance(generator.name, str)
        assert generator.null.strip()
        assert generator.limitations and all(s.strip() for s in generator.limitations)

    def test_the_description_is_ready_for_a_provenance_record(self, cls):
        info = cls().describe()
        assert info["name"] == cls().name
        assert info["null"] == cls().null
        assert isinstance(info["params"], dict)

    def test_it_keeps_the_shape_and_all_metadata(self, cls):
        ts = ar1_series()
        out = cls().generate_one(ts, rng())
        assert isinstance(out, TimeSeries)
        assert out is not ts
        assert out.data.shape == ts.data.shape
        assert out.sfreq == ts.sfreq
        assert out.channels == ts.channels
        assert out.segmentation == ts.segmentation
        assert out.tmin == ts.tmin
        assert np.isfinite(out.data).all()

    def test_it_does_not_change_the_input(self, cls):
        ts = ar1_series()
        before = ts.data.copy()
        cls().generate_one(ts, rng())
        np.testing.assert_array_equal(ts.data, before)

    def test_it_is_reproducible_and_depends_on_the_generator(self, cls):
        ts = ar1_series()
        a = cls().generate_one(ts, rng(1))
        b = cls().generate_one(ts, rng(1))
        c = cls().generate_one(ts, rng(2))
        np.testing.assert_array_equal(a.data, b.data)
        assert not np.array_equal(a.data, c.data)

    def test_it_actually_changes_the_data(self, cls):
        ts = ar1_series()
        assert not np.array_equal(cls().generate_one(ts, rng()).data, ts.data)

    def test_generate_returns_one_series_per_generator(self, cls):
        ts = ar1_series()
        rngs = RNGController(0).spawn(3)
        out = cls().generate(ts, rngs)
        assert len(out) == 3
        singles = [cls().generate_one(ts, g) for g in RNGController(0).spawn(3)]
        for a, b in zip(out, singles, strict=True):
            np.testing.assert_array_equal(a.data, b.data)

    def test_generate_needs_at_least_one_generator(self, cls):
        with pytest.raises(ValueError, match="at least one"):
            cls().generate(ar1_series(), [])


@pytest.mark.parametrize("n_times", [200, 201])
class TestPhaseRandomize:
    def test_the_amplitude_spectrum_is_kept_exactly(self, n_times):
        ts = ar1_series(n_times=n_times)
        out = PhaseRandomize().generate_one(ts, rng())
        np.testing.assert_allclose(amplitude(out.data), amplitude(ts.data), atol=1e-8, rtol=1e-9)

    def test_the_autocorrelation_and_mean_are_kept(self, n_times):
        ts = ar1_series(n_times=n_times)
        out = PhaseRandomize().generate_one(ts, rng())
        np.testing.assert_allclose(
            circular_autocorrelation(out.data),
            circular_autocorrelation(ts.data),
            atol=1e-7 * np.abs(circular_autocorrelation(ts.data)).max(),
        )
        np.testing.assert_allclose(out.data.mean(axis=-1), ts.data.mean(axis=-1), atol=1e-10)

    def test_a_negative_mean_keeps_its_sign(self, n_times):
        ts = ar1_series(n_times=n_times)
        shifted = TimeSeries(ts.data - 5.0, ts.sfreq, ts.channels, ts.segmentation)
        out = PhaseRandomize().generate_one(shifted, rng())
        np.testing.assert_allclose(out.data.mean(axis=-1), shifted.data.mean(axis=-1), atol=1e-10)

    def test_phases_are_independent_between_channels_and_trials(self, n_times):
        # Two identical channels in every trial stop being identical.
        base = ar1_series(n_channels=1, n_times=n_times).data
        ts = TimeSeries(
            np.concatenate([base, base], axis=1),
            100.0,
            (ChannelMeta("a"), ChannelMeta("b")),
            "epoched",
        )
        out = PhaseRandomize().generate_one(ts, rng()).data
        assert not np.allclose(out[:, 0], out[:, 1])
        assert not np.allclose(out[0, 0], out[1, 0])


class TestPhaseRandomizeCoherence:
    def test_coupling_between_channels_is_destroyed(self):
        # y is x delayed by 3 samples: strong cross-correlation at lag 3 before, none after.
        n = 4000
        x = ar1_series(n_trials=1, n_channels=1, n_times=n, phi=0.3).data[0, 0]
        y = np.roll(x, 3)
        ts = TimeSeries(
            np.stack([x, y])[np.newaxis], 100.0, (ChannelMeta("x"), ChannelMeta("y")), "continuous"
        )

        def xcorr(a, b):
            return np.corrcoef(a, b)[0, 1]

        observed = abs(xcorr(np.roll(ts.data[0, 0], 3), ts.data[0, 1]))
        assert observed > 0.99
        surrogate = [
            abs(xcorr(np.roll(s.data[0, 0], 3), s.data[0, 1]))
            for s in PhaseRandomize().generate(ts, RNGController(0).spawn(20))
        ]
        # Chance level for n = 4000 independent samples is about 1/sqrt(n) = 0.016.
        assert max(surrogate) < 0.1


def relative_spectrum_error(original, surrogate):
    a, b = amplitude(original.data), amplitude(surrogate.data)
    return float((np.linalg.norm(a - b, axis=-1) / np.linalg.norm(a, axis=-1)).mean())


class TestAAFT:
    def test_the_spectrum_is_approximately_kept(self):
        # Measured over 10 seeds: AAFT 0.21 to 0.25. The relative error of an AAFT that loses the
        # autocorrelation (rank matching in the wrong direction) is 0.36 to 0.43, so 0.3 separates
        # a correct implementation from that mistake.
        for seed in range(5):
            ts = skewed_series(seed=seed)
            out = AAFT().generate_one(ts, RNGController(seed).child(0))
            assert relative_spectrum_error(ts, out) < 0.3

    def test_the_marginal_distribution_is_kept_exactly(self):
        ts = skewed_series()
        out = AAFT().generate_one(ts, rng())
        np.testing.assert_array_equal(np.sort(out.data, axis=-1), np.sort(ts.data, axis=-1))

    def test_each_channel_and_trial_gets_its_own_sequence(self):
        base = skewed_series(n_channels=1).data
        ts = TimeSeries(
            np.concatenate([base, base], axis=1),
            100.0,
            (ChannelMeta("a"), ChannelMeta("b")),
            "epoched",
        )
        out = AAFT().generate_one(ts, rng()).data
        assert not np.array_equal(out[:, 0], out[:, 1])


class TestIAAFT:
    def test_the_marginal_distribution_is_kept_exactly(self):
        ts = skewed_series()
        out = IAAFT().generate_one(ts, rng())
        np.testing.assert_array_equal(np.sort(out.data, axis=-1), np.sort(ts.data, axis=-1))

    def test_the_spectrum_is_matched_much_better_than_by_aaft(self):
        # Measured over 10 seeds: IAAFT 0.025 to 0.041, AAFT 0.21 to 0.25 (skewed AR(1) data).
        for seed in range(5):
            ts = skewed_series(seed=seed)
            iaaft = IAAFT().generate_one(ts, RNGController(seed).child(0))
            aaft = AAFT().generate_one(ts, RNGController(seed).child(0))
            assert relative_spectrum_error(ts, iaaft) < 0.08
            assert relative_spectrum_error(ts, iaaft) < 0.5 * relative_spectrum_error(ts, aaft)

    def test_the_parameters_are_recorded(self):
        assert IAAFT(n_iter=40).describe()["params"] == {"n_iter": 40}

    @pytest.mark.parametrize("n_iter", [0, -3, 2.5, True])
    def test_the_iteration_limit_must_be_a_positive_integer(self, n_iter):
        with pytest.raises((TypeError, ValueError)):
            IAAFT(n_iter=n_iter)

    def test_it_warns_when_the_iteration_limit_stops_it_before_convergence(self):
        ts = skewed_series()
        with pytest.warns(DirconnWarning, match="converge"):
            IAAFT(n_iter=1).generate_one(ts, rng())

    def test_it_stays_quiet_when_it_converges(self):
        ts = skewed_series()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DirconnWarning)
            IAAFT(n_iter=1000).generate_one(ts, rng())


class TestTrialShuffle:
    def test_every_channel_holds_the_original_trials_in_a_new_order(self):
        ts = ar1_series(n_trials=8)
        out = TrialShuffle().generate_one(ts, rng()).data
        for ch in range(ts.n_channels):
            original = sorted(map(tuple, ts.data[:, ch]))
            shuffled = sorted(map(tuple, out[:, ch]))
            assert original == shuffled

    def test_the_trial_pairing_between_channels_is_broken(self):
        base = ar1_series(n_trials=20, n_channels=1).data
        ts = TimeSeries(
            np.concatenate([base, base], axis=1),
            100.0,
            (ChannelMeta("a"), ChannelMeta("b")),
            "epoched",
        )
        out = TrialShuffle().generate_one(ts, rng()).data
        paired = sum(np.array_equal(out[t, 0], out[t, 1]) for t in range(20))
        assert paired < 20

    def test_it_needs_at_least_two_trials(self):
        ts = ar1_series(n_trials=1)
        with pytest.raises(ValueError, match="at least 2 trials"):
            TrialShuffle().generate_one(ts, rng())

    def test_continuous_data_are_refused_with_the_reason(self):
        ts = TimeSeries(ar1_series(n_trials=1).data, 100.0, ar1_series().channels, "continuous")
        with pytest.raises(ValueError, match="trial"):
            TrialShuffle().generate_one(ts, rng())


class TestCircularShift:
    @staticmethod
    def shift_of(original, shifted):
        matches = [k for k in range(len(original)) if np.array_equal(np.roll(original, k), shifted)]
        assert len(matches) == 1
        return matches[0]

    def test_every_series_is_a_rotation_of_the_original(self):
        ts = ar1_series()
        out = CircularShift().generate_one(ts, rng()).data
        for trial in range(ts.n_trials):
            for ch in range(ts.n_channels):
                self.shift_of(ts.data[trial, ch], out[trial, ch])

    def test_the_shifts_respect_the_minimum_in_both_directions(self):
        ts = ar1_series(n_trials=4, n_channels=3, n_times=40)
        min_shift = 10
        found = []
        for generator in RNGController(0).spawn(60):
            out = CircularShift(min_shift=min_shift).generate_one(ts, generator).data
            for trial in range(ts.n_trials):
                for ch in range(ts.n_channels):
                    found.append(self.shift_of(ts.data[trial, ch], out[trial, ch]))
        assert min(found) >= min_shift
        assert max(found) <= ts.n_times - min_shift
        # Both ends are reachable.
        assert min(found) == min_shift
        assert max(found) == ts.n_times - min_shift

    def test_channels_and_trials_are_shifted_independently(self):
        ts = ar1_series(n_trials=5, n_channels=4, n_times=300)
        out = CircularShift().generate_one(ts, rng()).data
        shifts = {
            self.shift_of(ts.data[t, c], out[t, c])
            for t in range(ts.n_trials)
            for c in range(ts.n_channels)
        }
        assert len(shifts) > 1

    def test_the_autocorrelation_is_kept(self):
        ts = ar1_series()
        out = CircularShift().generate_one(ts, rng()).data
        np.testing.assert_allclose(
            circular_autocorrelation(out), circular_autocorrelation(ts.data), rtol=1e-8, atol=1e-8
        )

    @pytest.mark.parametrize("min_shift", [0, -1, 2.5, True])
    def test_the_minimum_shift_must_be_a_positive_integer(self, min_shift):
        with pytest.raises((TypeError, ValueError)):
            CircularShift(min_shift=min_shift)

    def test_the_series_must_be_long_enough_for_two_directions(self):
        ts = ar1_series(n_times=19)
        with pytest.raises(ValueError, match="n_times"):
            CircularShift(min_shift=10).generate_one(ts, rng())


class TestDocumentedLimitations:
    def test_the_table_lists_every_generator_with_its_null_and_limits(self):
        from dirconn.surrogates import limitations_table

        table = limitations_table()
        for generator in [cls() for cls in ALL] + [BlockShuffle(block_size=50)]:
            assert f"`{generator.name}`" in table
            assert generator.null in table
            for text in generator.limitations:
                assert text in table

    def test_the_documentation_contains_the_generated_table(self):
        from pathlib import Path

        from dirconn.surrogates import limitations_table

        doc = Path(__file__).parent.parent / "docs" / "surrogates.md"
        assert limitations_table() in doc.read_text(encoding="utf-8")


class TestBlockShuffle:
    @staticmethod
    def continuous(n_times=1000, n_channels=3, seed=0):
        x = np.random.default_rng(seed).standard_normal((n_channels, n_times))
        return TimeSeries.from_array(x, 100.0)

    def test_prepare_cuts_the_record_into_blocks_stored_as_trials(self):
        ts = self.continuous(1030)
        prepared = BlockShuffle(block_size=50).prepare(ts)
        assert prepared.segmentation == "epoched"
        assert (prepared.n_trials, prepared.n_channels, prepared.n_times) == (20, 3, 50)
        for k in range(20):
            np.testing.assert_array_equal(prepared.data[k], ts.data[0, :, k * 50 : (k + 1) * 50])
        assert prepared.channels == ts.channels and prepared.sfreq == ts.sfreq

    def test_prepare_cuts_every_trial_of_epoched_data(self):
        ts = ar1_series(n_trials=3, n_times=100)
        prepared = BlockShuffle(block_size=40).prepare(ts)
        assert (prepared.n_trials, prepared.n_times) == (6, 40)  # 2 whole blocks per trial
        np.testing.assert_array_equal(prepared.data[1], ts.data[0, :, 40:80])
        np.testing.assert_array_equal(prepared.data[2], ts.data[1, :, :40])

    def test_other_generators_leave_the_data_as_they_are(self):
        ts = ar1_series()
        for cls in ALL:
            assert cls().prepare(ts) is ts

    def test_every_channel_keeps_all_blocks_in_a_new_order(self):
        generator = BlockShuffle(block_size=50)
        prepared = generator.prepare(self.continuous())
        out = generator.generate_one(prepared, rng())
        for ch in range(prepared.n_channels):
            original = sorted(map(tuple, prepared.data[:, ch]))
            assert original == sorted(map(tuple, out.data[:, ch]))
        assert not np.array_equal(out.data, prepared.data)

    def test_the_pairing_of_blocks_between_channels_is_broken(self):
        base = self.continuous(1000, n_channels=1).data[0]
        ts = TimeSeries(
            np.concatenate([base, base], axis=0)[np.newaxis],
            100.0,
            (ChannelMeta("a"), ChannelMeta("b")),
            "continuous",
        )
        generator = BlockShuffle(block_size=50)
        out = generator.generate_one(generator.prepare(ts), rng()).data
        assert sum(np.array_equal(out[k, 0], out[k, 1]) for k in range(20)) < 20

    def test_it_refuses_data_that_were_not_prepared(self):
        with pytest.raises(ValueError, match="prepare"):
            BlockShuffle(block_size=50).generate_one(self.continuous(), rng())

    def test_at_least_two_blocks_are_needed(self):
        with pytest.raises(ValueError, match="two blocks"):
            BlockShuffle(block_size=60).prepare(self.continuous(100))

    @pytest.mark.parametrize("block_size", [0, -4, 2.5, True])
    def test_the_block_size_must_be_a_positive_integer(self, block_size):
        with pytest.raises((TypeError, ValueError)):
            BlockShuffle(block_size=block_size)

    def test_the_block_size_has_no_default(self):
        with pytest.raises(TypeError):
            BlockShuffle()  # type: ignore[call-arg]

    def test_it_states_its_null_and_limits_and_records_its_parameter(self):
        generator = BlockShuffle(block_size=25)
        assert generator.name == "block_shuffle"
        assert generator.null.strip() and len(generator.limitations) >= 3
        assert generator.describe()["params"] == {"block_size": 25}

    def test_it_is_reproducible(self):
        generator = BlockShuffle(block_size=50)
        prepared = generator.prepare(self.continuous())
        a = generator.generate_one(prepared, rng(1)).data
        np.testing.assert_array_equal(a, generator.generate_one(prepared, rng(1)).data)
        assert not np.array_equal(a, generator.generate_one(prepared, rng(2)).data)

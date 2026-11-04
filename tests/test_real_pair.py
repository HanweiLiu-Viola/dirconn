import numpy as np
import pytest

from dirconn import GC, DirconnWarning, TimeSeries
from dirconn.benchmark import (
    constructed_pair,
    delayed_bandpass,
    evaluate_pair,
    real_eeg_series,
)
from helpers import SFREQ, fake_recording


def pair(**kwargs):
    params = {
        "source_channel": "E0",
        "noise_channel": "E3",
        "source_start": 0.0,
        "noise_start": 100.0,
        "duration": 40.0,
        "delay_ms": 20.0,
        "strength": 1.0,
        "band": (8.0, 12.0),
    }
    params.update(kwargs)
    return constructed_pair(fake_recording(), **params)


class TestConstruction:
    def test_it_is_a_continuous_two_channel_series_of_the_requested_length(self):
        series, truth = pair()
        assert series.segmentation == "continuous" and series.n_channels == 2
        assert series.n_times == int(40.0 * SFREQ) and series.sfreq == SFREQ
        assert series.names == ("source", "target")
        np.testing.assert_array_equal(truth, [[np.nan, 1.0], [0.0, np.nan]])

    def test_the_source_is_the_real_channel_standardised(self):
        rec = fake_recording()
        series, _ = constructed_pair(
            rec,
            source_channel="E0",
            noise_channel="E3",
            source_start=10.0,
            noise_start=120.0,
            duration=30.0,
            delay_ms=20.0,
        )
        n = int(30.0 * SFREQ)
        raw = rec.data[0, rec.names.index("E0"), int(10 * SFREQ) : int(10 * SFREQ) + n]
        expected = (raw - raw.mean()) / raw.std()
        np.testing.assert_allclose(series.data[0, 0], expected, rtol=1e-10)

    def test_the_target_is_the_delayed_filtered_source_plus_independent_real_noise(self):
        rec = fake_recording()
        kwargs = {
            "source_channel": "E0",
            "noise_channel": "E3",
            "source_start": 0.0,
            "noise_start": 100.0,
            "duration": 40.0,
            "delay_ms": 20.0,
            "band": (8.0, 12.0),
        }
        weak, _ = constructed_pair(rec, strength=0.5, **kwargs)
        strong, _ = constructed_pair(rec, strength=2.0, **kwargs)
        # B = strength * F + C with the same F and C, so the difference isolates F
        f = (strong.data[0, 1] - weak.data[0, 1]) / 1.5
        c = weak.data[0, 1] - 0.5 * f
        noise = rec.data[0, rec.names.index("E3"), int(100 * SFREQ) : int(100 * SFREQ) + f.size]
        np.testing.assert_allclose(c, (noise - noise.mean()) / noise.std(), rtol=1e-8)
        assert f.std() == pytest.approx(1.0, rel=0.05)

    def test_the_source_channel_is_not_touched_by_the_construction(self):
        a, _ = pair(strength=0.1)
        b, _ = pair(strength=5.0)
        np.testing.assert_array_equal(a.data[0, 0], b.data[0, 0])

    def test_it_is_deterministic(self):
        a, _ = pair()
        b, _ = pair()
        np.testing.assert_array_equal(a.data, b.data)


class TestDelayedBandpass:
    def test_it_is_causal_and_starts_after_the_delay(self):
        # a perturbation at sample t0 must not change the output before t0 + delay
        rng = np.random.default_rng(0)
        x = rng.standard_normal(3000)
        t0, delay = 1000, 6
        bumped = x.copy()
        bumped[t0] += 5.0
        base = delayed_bandpass(x, sfreq=SFREQ, band=(8.0, 12.0), delay_samples=delay)
        other = delayed_bandpass(bumped, sfreq=SFREQ, band=(8.0, 12.0), delay_samples=delay)
        change = np.abs(other - base)
        assert change[: t0 + delay].max() == 0.0
        assert change[t0 + delay :].max() > 0.0

    def test_the_first_samples_are_zero_and_the_length_is_kept(self):
        x = np.random.default_rng(1).standard_normal(500)
        out = delayed_bandpass(x, sfreq=SFREQ, band=(8.0, 12.0), delay_samples=4)
        assert out.shape == x.shape and (out[:4] == 0.0).all()

    def test_it_keeps_the_band_and_removes_the_rest(self):
        t = np.arange(6000) / SFREQ
        inside, outside = np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 40 * t)
        kept = delayed_bandpass(inside, sfreq=SFREQ, band=(8.0, 12.0), delay_samples=3)
        removed = delayed_bandpass(outside, sfreq=SFREQ, band=(8.0, 12.0), delay_samples=3)
        assert kept[500:].std() > 0.5 and removed[500:].std() < 0.05


class TestErrors:
    def test_overlapping_windows_are_refused_because_the_noise_must_be_independent(self):
        with pytest.raises(ValueError, match="overlap"):
            pair(source_start=0.0, noise_start=30.0, duration=40.0)

    def test_a_window_outside_the_recording_is_refused(self):
        with pytest.raises(ValueError, match="recording"):
            pair(noise_start=290.0)

    def test_the_two_channels_must_differ_and_exist(self):
        with pytest.raises(ValueError, match="differ"):
            pair(noise_channel="E0")
        with pytest.raises(ValueError, match="E9"):
            pair(source_channel="E9")

    def test_the_delay_must_be_a_whole_number_of_samples_and_positive(self):
        with pytest.raises(ValueError, match="delay"):
            pair(delay_ms=0.0)
        with pytest.raises(ValueError, match="whole number"):
            pair(delay_ms=21.0)

    def test_the_band_must_lie_below_the_nyquist_frequency(self):
        with pytest.raises(ValueError, match="band"):
            pair(band=(8.0, 80.0))
        with pytest.raises(ValueError, match="band"):
            pair(band=(12.0, 8.0))

    @pytest.mark.parametrize("strength", [0.0, -1.0, float("nan")])
    def test_the_strength_must_be_positive(self, strength):
        with pytest.raises(ValueError, match="strength"):
            pair(strength=strength)

    def test_the_recording_must_be_continuous(self):
        epochs = TimeSeries.from_array(
            np.zeros((3, 2, 100)), SFREQ, segmentation="epoched", names=["a", "b"]
        )
        with pytest.raises(ValueError, match="continuous"):
            constructed_pair(
                epochs,
                source_channel="a",
                noise_channel="b",
                source_start=0.0,
                noise_start=0.5,
                duration=0.2,
                delay_ms=20.0,
            )


class TestEvaluatePair:
    """Measured on the stand-in recording (1/f noise, coupling at 8 to 12 Hz, 20 ms delay).

    Model order decides the direction. With ``strength`` 3 the forward / reverse GC ratio is 0.04
    to 0.23 at orders 2 to 8 (the wrong direction), 3.8 at order 16, 21 at 32 and 25 at 64; for
    DTF at orders 8, 32 and 64 it is 0.0, 20 and 20. A narrow-band influence on a 1/f background
    has a long memory, which a short AR model hands to the other channel.
    """

    def test_gc_finds_the_constructed_direction_with_a_long_enough_model(self):
        series, truth = pair(strength=3.0)
        out = evaluate_pair(GC(order=64), series, truth)
        assert out["direction_accuracy"] == 1.0 and out["auroc"] == 1.0
        assert out["scores"]["forward"] > 10 * out["scores"]["reverse"]

    def test_a_model_that_is_too_short_gets_the_direction_wrong(self):
        series, truth = pair(strength=3.0)
        out = evaluate_pair(GC(order=4), series, truth)
        assert out["direction_accuracy"] == 0.0
        assert out["scores"]["reverse"] > out["scores"]["forward"]

    def test_a_spectral_estimator_needs_the_band_of_the_construction(self):
        from dirconn import DTF

        series, truth = pair(strength=3.0)
        with pytest.raises(ValueError, match="band"):
            evaluate_pair(DTF(order=64), series, truth)
        out = evaluate_pair(DTF(order=64), series, truth, band=(8.0, 12.0))
        assert out["direction_accuracy"] == 1.0
        assert out["scores"]["forward"] > 10 * out["scores"]["reverse"]

    def test_diagnostics_are_collected_not_raised(self):
        import warnings

        series, truth = pair(strength=3.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DirconnWarning)
            out = evaluate_pair(GC(order=64), series, truth)
        assert isinstance(out["diagnostics"], list)


@pytest.fixture(scope="module")
def real():
    return real_eeg_series()


@pytest.mark.sample_data
class TestRealRecording:
    def test_it_is_the_good_eeg_channels_filtered_and_resampled(self, real):
        assert real.segmentation == "continuous"
        assert real.n_channels == 59 and "EEG 053" not in real.names
        assert real.sfreq == 150.0
        assert 270 < real.duration < 280
        assert np.isfinite(real.data).all()

    def test_loading_twice_returns_the_cached_series(self, real):
        assert real_eeg_series() is real

    def test_the_known_direction_is_recovered_in_real_background_with_a_long_enough_model(
        self, real
    ):
        # Full study (scripts/real_pair_study.py): 24 random channel pairs and non-overlapping 40 s
        # windows, 20 ms delay, band 8 to 12 Hz. Direction accuracy at strength 1 and 3:
        #   GC order 8: 0.29 and 0.33 (below chance: the direction comes out reversed)
        #   GC order 32, 64 and BIC up to 64 (it chose about 63), DTF order 64, gPDC order 64: 1.00
        from dirconn import DTF

        rng = np.random.default_rng(0)
        duration, n_slots = 40.0, 6
        accuracy = {"GC8": [], "GC64": [], "DTF64": []}
        for _ in range(8):
            channels = rng.choice(real.names, size=2, replace=False)
            slots = rng.choice(n_slots, size=2, replace=False)
            series, truth = constructed_pair(
                real,
                source_channel=str(channels[0]),
                noise_channel=str(channels[1]),
                source_start=float(slots[0] * duration),
                noise_start=float(slots[1] * duration),
                duration=duration,
                delay_ms=20.0,
                strength=1.0,
            )
            accuracy["GC8"].append(evaluate_pair(GC(order=8), series, truth)["direction_accuracy"])
            accuracy["GC64"].append(
                evaluate_pair(GC(order=64), series, truth)["direction_accuracy"]
            )
            accuracy["DTF64"].append(
                evaluate_pair(DTF(order=64), series, truth, band=(8.0, 12.0))["direction_accuracy"]
            )
        assert np.mean(accuracy["GC64"]) == 1.0 and np.mean(accuracy["DTF64"]) == 1.0
        assert np.mean(accuracy["GC8"]) <= 0.5

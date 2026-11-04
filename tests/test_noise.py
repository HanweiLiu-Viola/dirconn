"""Real noise in the simulation, the empty-room recording, and the noise study."""

import numpy as np
import pytest

from dirconn.simulation import simulate_on_head
from helpers import ROIS, chain, head

N_TIMES = 2000
NET = chain(ROIS[0], ROIS[1], ROIS[2])


def coloured(shape, seed=0):
    """Noise that is neither white nor equal on every sensor: smoothed, with a per-sensor scale."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    kernel = np.ones(7) / 7.0
    x = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), -1, x)
    return x * rng.uniform(0.5, 3.0, size=(1, shape[1], 1))


def simulate(model, **kwargs):
    params = {
        "sfreq": 100.0,
        "n_times": N_TIMES,
        "seed": 3,
        "snr_db": 0.0,
        "n_background": 0,
        "source_model": "patch",
    }
    params.update(kwargs)
    return simulate_on_head(NET, model, **params)


@pytest.fixture(scope="module")
def model():
    return head(n_channels=10)


class TestNoiseSeries:
    def test_the_given_noise_is_the_noise_of_the_simulation_up_to_one_scale(self, model):
        given = coloured((1, 10, N_TIMES))
        sim = simulate(model, noise_series=given)
        ratio = np.asarray(sim.noise) / given
        np.testing.assert_allclose(ratio, ratio.flat[0], rtol=1e-9)
        assert ratio.flat[0] > 0

    @pytest.mark.parametrize("snr_db", [10.0, 0.0, -10.0])
    def test_the_realised_snr_is_exactly_the_target(self, model, snr_db):
        sim = simulate(model, noise_series=coloured((1, 10, N_TIMES)), snr_db=snr_db)
        assert sim.snr_db == pytest.approx(snr_db, abs=1e-9)

    def test_the_sources_and_the_signal_do_not_depend_on_the_noise(self, model):
        with_noise = simulate(model, noise_series=coloured((1, 10, N_TIMES)))
        without = simulate(model)
        np.testing.assert_array_equal(
            np.asarray(with_noise.sources.data), np.asarray(without.sources.data)
        )
        np.testing.assert_array_equal(np.asarray(with_noise.signal), np.asarray(without.signal))
        assert not np.array_equal(np.asarray(with_noise.noise), np.asarray(without.noise))

    def test_the_data_are_the_signal_plus_the_noise(self, model):
        sim = simulate(model, noise_series=coloured((1, 10, N_TIMES)))
        np.testing.assert_allclose(
            np.asarray(sim.data.data), np.asarray(sim.signal) + np.asarray(sim.noise), atol=1e-12
        )

    def test_it_needs_the_background_to_be_off(self, model):
        with pytest.raises(ValueError, match="n_background"):
            simulate(model, noise_series=coloured((1, 10, N_TIMES)), n_background=5)

    @pytest.mark.parametrize("shape", [(1, 9, N_TIMES), (1, 10, N_TIMES - 1), (2, 10, N_TIMES)])
    def test_the_shape_must_be_trials_sensors_times(self, model, shape):
        with pytest.raises(ValueError, match="noise_series"):
            simulate(model, noise_series=coloured(shape))

    def test_it_must_be_finite(self, model):
        bad = coloured((1, 10, N_TIMES))
        bad[0, 0, 5] = np.nan
        with pytest.raises(ValueError, match="noise_series"):
            simulate(model, noise_series=bad)

    def test_it_must_not_be_all_zero(self, model):
        with pytest.raises(ValueError, match="noise_series"):
            simulate(model, noise_series=np.zeros((1, 10, N_TIMES)))

    def test_the_provenance_says_whether_the_noise_was_given(self, model):
        given = simulate(model, noise_series=coloured((1, 10, N_TIMES)))
        assert given.provenance.parameters["noise_series"] is True
        assert simulate(model).provenance.parameters["noise_series"] is False

    def test_two_trials_take_a_noise_of_two_trials(self, model):
        sim = simulate(model, n_trials=2, noise_series=coloured((2, 10, N_TIMES)), n_times=N_TIMES)
        assert np.asarray(sim.noise).shape == (2, 10, N_TIMES)
        assert sim.snr_db == pytest.approx(0.0, abs=1e-9)


@pytest.fixture(scope="module")
def heads():
    from dirconn.headmodel import load_head_model

    return {m: load_head_model(m) for m in ("mag", "grad", "eeg")}


@pytest.mark.sample_data
class TestEmptyRoom:
    def reference(self, head, sfreq=100.0):
        """The recording read and resampled by MNE directly, in the order of the head's channels."""
        import mne

        from dirconn.datasets import require_sample

        raw = mne.io.read_raw_fif(
            require_sample() / "MEG" / "sample" / "ernoise_raw.fif", preload=True, verbose="ERROR"
        )
        raw.pick(list(head.channel_names)).resample(sfreq, verbose="ERROR")
        return np.vstack([raw.get_data(picks=[raw.ch_names.index(n)]) for n in head.channel_names])

    @pytest.mark.parametrize(("modality", "n_channels"), [("mag", 102), ("grad", 203)])
    def test_it_is_the_whole_recording_of_the_channels_of_the_head_at_100_hz(
        self, heads, modality, n_channels
    ):
        from dirconn.emptyroom import empty_room_series

        data = empty_room_series(heads[modality])
        assert data.shape[0] == n_channels
        assert abs(data.shape[1] - 11000) <= 1  # 110 s
        assert np.isfinite(data).all()

    @pytest.mark.parametrize("modality", ["mag", "grad"])
    def test_channel_order_and_resampling_are_those_of_mne(self, heads, modality):
        from dirconn.emptyroom import empty_room_series

        data = empty_room_series(heads[modality])
        np.testing.assert_allclose(data, self.reference(heads[modality]), rtol=1e-9, atol=1e-25)

    def test_the_order_is_the_one_of_the_head_even_when_it_differs_from_the_file(self, heads):
        import dataclasses

        from dirconn.emptyroom import empty_room_series

        # the head's channels happen to be in the file's order; reverse them so the order matters
        reversed_head = dataclasses.replace(
            heads["grad"], channel_names=tuple(reversed(heads["grad"].channel_names))
        )
        data = empty_room_series(reversed_head)
        np.testing.assert_allclose(data, self.reference(reversed_head), rtol=1e-9, atol=1e-25)
        np.testing.assert_array_equal(data, empty_room_series(heads["grad"])[::-1])

    def test_the_recording_is_not_projected_or_rescaled(self, heads):
        from dirconn.emptyroom import empty_room_series

        data = empty_room_series(heads["grad"])
        # grad noise of this recording is of the order of 1e-12 T/m; a scaled copy would not be
        assert 1e-13 < np.median(np.sqrt((data**2).mean(axis=1))) < 1e-10

    def test_only_meg_has_an_empty_room(self, heads):
        from dirconn.emptyroom import empty_room_series

        with pytest.raises(ValueError, match="MEG"):
            empty_room_series(heads["eeg"])

    def test_windows_are_consecutive_blocks_that_do_not_overlap(self, heads):
        from dirconn.emptyroom import empty_room_series, empty_room_window, n_windows

        head_ = heads["grad"]
        full = empty_room_series(head_)
        assert n_windows(head_, 3000) == 3
        for k in range(3):
            window = empty_room_window(head_, k, 3000)
            assert window.shape == (1, 203, 3000)
            np.testing.assert_array_equal(window[0], full[:, k * 3000 : (k + 1) * 3000])

    def test_a_window_past_the_recording_and_a_duration_longer_than_it_are_errors(self, heads):
        from dirconn.emptyroom import empty_room_window, n_windows

        with pytest.raises(ValueError, match="window"):
            empty_room_window(heads["grad"], 3, 3000)
        assert n_windows(heads["grad"], 20000) == 0
        with pytest.raises(ValueError, match="110"):
            empty_room_window(heads["grad"], 0, 20000)

    def test_the_array_is_cached_and_read_only(self, heads):
        from dirconn.emptyroom import empty_room_series

        a = empty_room_series(heads["mag"])
        assert empty_room_series(heads["mag"]) is a
        assert not a.flags.writeable

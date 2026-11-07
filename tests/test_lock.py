"""Event locking and matched control locks (issue #7)."""

import warnings

import numpy as np
import pytest

from dirconn import GC, DirconnWarning, TimeSeries

SFREQ = 100.0


def record(n_times=2000, seed=0, **masks):
    rng = np.random.default_rng(seed)
    return TimeSeries.from_array(rng.standard_normal((3, n_times)), SFREQ, **masks)


class TestLock:
    def test_epochs_are_cut_around_the_events(self):
        data = record()
        locked = data.lock([2.0, 5.5, 10.0], tmin=-0.2, tmax=0.3)
        assert locked.segmentation == "epoched"
        assert locked.n_trials == 3 and locked.n_times == 51
        assert locked.tmin == pytest.approx(-0.2)
        assert locked.names == data.names and locked.sfreq == SFREQ
        for k, onset in enumerate([200, 550, 1000]):
            np.testing.assert_array_equal(locked.data[k], data.data[0][:, onset - 20 : onset + 31])

    def test_events_are_rounded_to_the_nearest_sample(self):
        data = record()
        locked = data.lock([2.004, 2.006], tmin=0.0, tmax=0.1)
        np.testing.assert_array_equal(locked.data[0], data.data[0][:, 200:211])
        np.testing.assert_array_equal(locked.data[1], data.data[0][:, 201:212])

    def test_locks_that_cross_the_edges_are_dropped_and_recorded(self):
        locked = record().lock([0.1, 5.0, 19.9], tmin=-0.2, tmax=0.3)
        assert locked.n_trials == 1
        step = locked.history[-1]
        assert step["step"] == "lock"
        assert step["events"] == [5.0]
        assert step["dropped_edge"] == [0.1, 19.9]
        assert step["dropped_invalid"] == []

    def test_locks_that_overlap_invalid_samples_are_dropped_and_recorded(self):
        valid = np.ones(2000, dtype=bool)
        valid[505:531] = False
        data = record(valid=valid)
        locked = data.lock([5.0, 5.4, 10.0], tmin=-0.2, tmax=0.1)
        assert locked.history[-1]["dropped_invalid"] == [5.0, 5.4]
        assert locked.history[-1]["events"] == [10.0]
        assert locked.valid is None  # every kept sample is valid

    def test_the_predict_mask_travels_with_the_locks(self):
        predict = np.zeros(2000, dtype=bool)
        predict[1000:1010] = True
        locked = record(predict=predict).lock([10.0], tmin=-0.1, tmax=0.2)
        np.testing.assert_array_equal(locked.predict[0], predict[990:1021])

    def test_the_time_axis_of_the_record_is_respected(self):
        rng = np.random.default_rng(0)
        data = TimeSeries.from_array(rng.standard_normal((2, 500)), SFREQ, tmin=10.0)
        locked = data.lock([12.0], tmin=0.0, tmax=0.05)
        np.testing.assert_array_equal(locked.data[0], data.data[0][:, 200:206])

    def test_epoched_data_cannot_be_locked(self):
        rng = np.random.default_rng(0)
        data = TimeSeries.from_array(
            rng.standard_normal((2, 2, 100)), SFREQ, segmentation="epoched"
        )
        with pytest.raises(ValueError, match="continuous"):
            data.lock([0.5], tmin=0.0, tmax=0.1)

    def test_no_lock_left_raises(self):
        with pytest.raises(ValueError, match="No event"):
            record().lock([0.0, 19.99], tmin=-0.2, tmax=0.2)

    @pytest.mark.parametrize(
        ("events", "tmin", "tmax", "match"),
        [
            ([1.0], 0.3, 0.2, "tmin"),
            ([], -0.1, 0.1, "event"),
            ([[1.0]], -0.1, 0.1, "one-dimensional"),
            ([np.nan], -0.1, 0.1, "finite"),
        ],
    )
    def test_invalid_arguments_are_rejected(self, events, tmin, tmax, match):
        with pytest.raises(ValueError, match=match):
            record().lock(events, tmin=tmin, tmax=tmax)

    def test_the_lock_is_recorded_in_the_provenance_of_a_result(self):
        locked = record(n_times=6000).lock(np.arange(2.0, 58.0, 1.0), tmin=-0.2, tmax=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DirconnWarning)
            result = GC(order=2).fit(locked)
        steps = result.provenance.preprocessing["data_history"]
        assert steps[-1]["step"] == "lock"
        assert len(steps[-1]["events"]) == 56
        result.provenance.to_dict()  # strictly JSON serialisable

    def test_the_history_survives_pick(self):
        locked = record().lock([5.0], tmin=0.0, tmax=0.1)
        assert locked.pick(["ch1"]).history == locked.history


class TestControlLocks:
    def eligible(self):
        mask = np.zeros(2000, dtype=bool)
        mask[300:700] = True
        mask[1200:1900] = True
        return mask

    def test_the_same_seed_gives_the_same_locks(self):
        data = record()
        a = data.control_locks(self.eligible(), 12, tmin=-0.2, tmax=0.3, seed=3)
        b = data.control_locks(self.eligible(), 12, tmin=-0.2, tmax=0.3, seed=3)
        c = data.control_locks(self.eligible(), 12, tmin=-0.2, tmax=0.3, seed=4)
        assert a.n_trials == 12
        np.testing.assert_array_equal(a.data, b.data)
        assert a.history == b.history
        assert a.history[-1]["events"] != c.history[-1]["events"]

    def test_onsets_are_eligible_and_windows_are_valid_and_inside(self):
        valid = np.ones(2000, dtype=bool)
        valid[400:450] = False
        eligible = self.eligible()
        eligible[1890:] = True  # windows would cross the end
        data = record(valid=valid)
        locks = data.control_locks(eligible, 30, tmin=-0.2, tmax=0.3, seed=0)
        step = locks.history[-1]
        assert step["step"] == "control_locks" and step["seed"] == 0 and step["n"] == 30
        onsets = np.round(np.asarray(step["events"]) * SFREQ).astype(int)
        assert eligible[onsets].all()
        for onset in onsets:
            assert onset - 20 >= 0 and onset + 30 < 2000
            assert valid[onset - 20 : onset + 31].all()
        assert onsets.tolist() == sorted(onsets.tolist())
        assert len(set(onsets.tolist())) == 30

    def test_too_few_candidates_raise(self):
        eligible = np.zeros(2000, dtype=bool)
        eligible[1000:1005] = True
        with pytest.raises(ValueError, match="5 eligible"):
            record().control_locks(eligible, 6, tmin=-0.1, tmax=0.1, seed=0)

    def test_the_mask_must_cover_the_record(self):
        with pytest.raises(ValueError, match="eligible"):
            record().control_locks(np.ones(10, dtype=bool), 2, tmin=0.0, tmax=0.1, seed=0)

    def test_a_seed_is_required(self):
        with pytest.raises(TypeError):
            record().control_locks(self.eligible(), 2, tmin=0.0, tmax=0.1)  # type: ignore[call-arg]

    def test_controls_match_a_lock_in_shape(self):
        data = record()
        locked = data.lock([4.0, 8.0, 15.0], tmin=-0.2, tmax=0.3)
        controls = data.control_locks(self.eligible(), locked.n_trials, tmin=-0.2, tmax=0.3, seed=0)
        assert controls.data.shape == locked.data.shape
        assert controls.tmin == locked.tmin

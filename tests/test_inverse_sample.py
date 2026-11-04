"""Tests of the spatial filter and region series that need the MNE sample dataset.

Run with ``pytest -m sample_data``.
"""

import numpy as np
import pytest

from dirconn.inverse import ROI_MODES, extract_roi

pytestmark = pytest.mark.sample_data

NODES = ("superiorfrontal-lh", "precuneus-rh", "lateraloccipital-lh")
MNE_NAMES = {"mne": "MNE", "dspm": "dSPM", "eloreta": "eLORETA"}


def simulated(modality, n_times=1500, snr_db=5.0, seed=1, n_trials=1):
    from dirconn.headmodel import load_head_model
    from dirconn.simulation import simulate_on_head
    from helpers import chain

    head = load_head_model(modality)
    sim = simulate_on_head(
        chain(*NODES),
        head,
        sfreq=100.0,
        n_times=n_times,
        n_trials=n_trials,
        seed=seed,
        snr_db=snr_db,
        n_background=30,
    )
    return head, sim


def mne_source_estimate(head, data, method):
    """MNE's own source estimate of ``data``, for comparison (average reference for EEG)."""
    import mne

    info = head.info.copy()
    with info._unlock():
        info["sfreq"] = 100.0
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    if head.modality == "eeg":
        raw.set_eeg_reference("average", projection=True, verbose="ERROR")
    if method == "lcmv":
        cov = mne.compute_raw_covariance(raw, method="empirical", verbose="ERROR")
        filters = mne.beamformer.make_lcmv(
            raw.info,
            head.forward,
            cov,
            reg=0.05,
            pick_ori=None,
            weight_norm="unit-noise-gain",
            rank="info",
            verbose="ERROR",
        )
        return mne.beamformer.apply_lcmv_raw(raw, filters, verbose="ERROR").data
    # the noise scale is the mean channel variance of the (for EEG, average-referenced) data;
    # dSPM depends on it
    referenced = data - data.mean(axis=0, keepdims=True) if head.modality == "eeg" else data
    noise = mne.Covariance(
        np.eye(head.n_channels) * float(np.mean(np.var(referenced, axis=1))),
        list(head.channel_names),
        [],
        [],
        1000,
    )
    inverse = mne.minimum_norm.make_inverse_operator(
        raw.info,
        head.forward,
        noise,
        loose=0.0,
        depth=None,
        fixed=True,
        rank="info",
        verbose="ERROR",
    )
    return mne.minimum_norm.apply_inverse_raw(
        raw,
        inverse,
        lambda2=1.0 / 9.0,
        method=MNE_NAMES[method],
        pick_ori=None,
        verbose="ERROR",
    ).data


class TestSpatialFilter:
    @pytest.mark.parametrize("method", ["lcmv", "mne", "dspm"])
    @pytest.mark.parametrize("modality", ["eeg", "grad"])
    def test_the_filter_gives_mnes_own_source_estimate(self, modality, method):
        from dirconn.inverse import spatial_filter

        head, sim = simulated(modality)
        data = sim.data.data[:, : head.n_channels]
        weights = spatial_filter(head, data, method=method)
        assert weights.shape == (head.n_sources, head.n_channels)
        expected = mne_source_estimate(head, data[0], method)
        np.testing.assert_allclose(
            weights @ data[0], expected, atol=1e-9 * np.abs(expected).max(), rtol=0
        )

    def test_eloreta_also_matches_mne(self):
        from dirconn.inverse import spatial_filter

        head, sim = simulated("eeg", n_times=600)
        data = sim.data.data[:, : head.n_channels]
        weights = spatial_filter(head, data, method="eloreta")
        expected = mne_source_estimate(head, data[0], "eloreta")
        np.testing.assert_allclose(
            weights @ data[0], expected, atol=1e-9 * np.abs(expected).max(), rtol=0
        )

    def test_eeg_filters_ignore_the_common_mode(self):
        from dirconn.inverse import spatial_filter

        head, sim = simulated("eeg")
        data = sim.data.data[:, : head.n_channels]
        for method in ("lcmv", "mne"):
            weights = spatial_filter(head, data, method=method)
            assert np.abs(weights.sum(axis=1)).max() < 1e-9 * np.abs(weights).sum(axis=1).max()

    def test_the_regularisation_matters(self):
        from dirconn.inverse import spatial_filter

        head, sim = simulated("grad")
        data = sim.data.data[:, : head.n_channels]
        a = spatial_filter(head, data, method="lcmv", reg=0.01)
        b = spatial_filter(head, data, method="lcmv", reg=0.5)
        assert not np.allclose(a, b)

    def test_a_head_model_without_forward_or_info_is_refused(self):
        from dirconn.inverse import spatial_filter
        from helpers import head as made_up_head

        with pytest.raises(ValueError, match="forward"):
            spatial_filter(made_up_head(), np.ones((1, 12, 100)), method="lcmv")

    def test_invalid_arguments_are_refused(self):
        from dirconn.inverse import spatial_filter

        head, sim = simulated("grad", n_times=300)
        data = sim.data.data[:, : head.n_channels]
        with pytest.raises(ValueError, match="method"):
            spatial_filter(head, data, method="sloreta")
        with pytest.raises(ValueError, match="channels"):
            spatial_filter(head, data[:, :10], method="lcmv")
        with pytest.raises(ValueError, match="reg"):
            spatial_filter(head, data, method="lcmv", reg=-1.0)


class TestDefaults:
    def test_the_default_method_is_minimum_norm(self):
        from dirconn.inverse import roi_series, spatial_filter

        head, sim = simulated("grad", n_times=600)
        data = sim.data.data[:, : head.n_channels]
        np.testing.assert_array_equal(
            spatial_filter(head, data), spatial_filter(head, data, method="mne")
        )
        sensors = sim.data.pick(list(head.channel_names))
        np.testing.assert_array_equal(
            roi_series(sensors, head, NODES).data,
            roi_series(sensors, head, NODES, method="mne", mode="pca_flip").data,
        )


class TestRoiSeries:
    @pytest.mark.parametrize("mode", ROI_MODES)
    def test_the_extraction_matches_mnes_extract_label_time_course(self, mode):
        import mne

        from dirconn.datasets import require_sample
        from dirconn.headmodel import load_head_model

        head = load_head_model("grad")
        data = np.random.default_rng(0).standard_normal((head.n_sources, 200))
        src = head.forward["src"]
        stc = mne.SourceEstimate(
            data, vertices=[s["vertno"] for s in src], tmin=0.0, tstep=0.01, verbose="ERROR"
        )
        labels = {
            lab.name: lab
            for lab in mne.read_labels_from_annot(
                "sample",
                parc="aparc",
                subjects_dir=require_sample() / "subjects",
                verbose="ERROR",
            )
        }
        for name in ("bankssts-lh", "precuneus-rh", "insula-lh"):
            expected = mne.extract_label_time_course(
                stc, labels[name], src, mode=mode, verbose="ERROR"
            )[0]
            idx = np.flatnonzero(head.source_roi == head.roi_names.index(name))
            ours = extract_roi(data[idx][None], head.source_flip[idx], mode)[0]
            np.testing.assert_allclose(ours, expected, rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize("modality", ["grad", "eeg"])
    def test_minimum_norm_series_follow_the_true_node_activity(self, modality):
        from dirconn.inverse import roi_series

        # Measured over 8 seeds x 3 nodes at 10 dB (coupled chain, partly coherent sources, c = 0.7,
        # the default): |r| median 0.90 and minimum 0.81 (gradiometers), 0.93 and 0.89 (EEG),
        # 0.88 and 0.73 (magnetometers). With whole-label sources it was 0.95 to 0.99.
        head, sim = simulated(modality, n_times=3000, snr_db=10.0)
        sensors = sim.data.pick(list(head.channel_names))
        out = roi_series(sensors, head, NODES, method="mne", mode="pca_flip")
        assert out.names == NODES and out.segmentation == sensors.segmentation
        correlation = [
            abs(np.corrcoef(out.data[0, k], sim.sources.data[0, k])[0, 1]) for k in range(3)
        ]
        assert min(correlation) > 0.65 and np.median(correlation) > 0.8

    def test_the_vertex_wise_beamformer_with_pca_flip_recovers_almost_nothing(self):
        # Measured (docs/benchmark.md): the standard recipe, an LCMV filter per vertex and
        # the first principal component of the vertices of the region, gives |r| of 0.01 to 0.09
        # with focal, partly coherent and whole-label sources alike, at every coupling strength,
        # although the filter is right at the vertex where the source is (|r| 0.99). If this test
        # fails because the recipe improved, update the documentation, not the bound.
        from dirconn.inverse import roi_series

        head, sim = simulated("grad", n_times=3000, snr_db=10.0)
        sensors = sim.data.pick(list(head.channel_names))
        out = roi_series(sensors, head, NODES, method="lcmv", mode="pca_flip")
        correlation = [
            abs(np.corrcoef(out.data[0, k], sim.sources.data[0, k])[0, 1]) for k in range(3)
        ]
        assert max(correlation) < 0.2

    def test_epoched_data_keep_their_shape_and_trials(self):
        from dirconn.inverse import roi_series

        head, sim = simulated("grad", n_times=400, n_trials=6, seed=2)
        out = roi_series(sim.data.pick(list(head.channel_names)), head, NODES)
        assert out.data.shape == (6, 3, 400) and out.segmentation == "epoched"

    def test_the_input_must_be_the_channels_of_the_head_model_in_order(self):
        from dirconn.inverse import roi_series

        head, sim = simulated("grad", n_times=300)
        shuffled = sim.data.pick(list(head.channel_names)[::-1])
        with pytest.raises(ValueError, match="channels"):
            roi_series(shuffled, head, NODES)

    def test_unknown_regions_are_named(self):
        from dirconn.inverse import roi_series

        head, sim = simulated("grad", n_times=300)
        sensors = sim.data.pick(list(head.channel_names))
        with pytest.raises(ValueError, match="notaregion-lh"):
            roi_series(sensors, head, ["notaregion-lh", NODES[0]])

import numpy as np
import pytest

from dirconn import Edge, Network, direct_node, dk_node, simulate
from dirconn.headmodel import HeadModel
from dirconn.simulation import simulate_on_head
from helpers import ROIS, chain, head


def run(model=None, network=None, **kwargs):
    params = {"sfreq": 100.0, "n_times": 600, "seed": 3, "snr_db": 10.0}
    params.update(kwargs)
    return simulate_on_head(network or chain(ROIS[0], ROIS[1]), model or head(), **params)


class TestMixing:
    def test_the_sensors_are_the_head_models_channels_with_their_type(self):
        for modality in ("eeg", "mag", "grad"):
            sim = run(model=head(modality))
            assert sim.data.names == head(modality).channel_names
            assert {c.type for c in sim.data.channels} == {modality}

    def test_the_signal_is_the_roi_topography_times_the_node_activity(self):
        model = head()
        sim = run(model=model, network=chain(ROIS[2], ROIS[0], ROIS[3]), source_model="patch")
        columns = [model.roi_names.index(n) for n in (ROIS[2], ROIS[0], ROIS[3])]
        expected = np.einsum("sk,bkt->bst", model.roi_leadfield[:, columns], sim.sources.data)
        np.testing.assert_allclose(sim.signal, expected, rtol=1e-12, atol=1e-12)

    def test_background_sources_are_vertices_outside_the_nodes_regions(self):
        model = head()
        sim = run(model=model, n_background=6)
        n_nodes = 2
        background = sim.leadfield[:, n_nodes:]
        assert background.shape[1] == 6
        columns = [
            int(np.flatnonzero(np.all(np.isclose(model.gain, col[:, None]), axis=0))[0])
            for col in background.T
        ]
        assert all(model.source_roi[c] not in (0, 1) for c in columns)  # not in the nodes' regions
        assert len(set(columns)) == 6  # drawn without replacement

    def test_other_regions_of_the_atlas_can_be_background_too(self):
        # Sources inside regions that are not network nodes are allowed as background.
        model = head()
        sim = run(model=model, n_background=60)
        columns = {
            int(np.flatnonzero(np.all(np.isclose(model.gain, col[:, None]), axis=0))[0])
            for col in sim.leadfield[:, 2:].T
        }
        in_other_roi = [c for c in columns if model.source_roi[c] >= 2]
        assert in_other_roi
        assert all(model.source_roi[c] not in (0, 1) for c in columns)

    def test_it_equals_simulate_with_the_assembled_leadfield(self):
        model = head()
        sim = run(model=model, n_background=5, seed=9, source_model="patch")
        direct = simulate(
            sim.network,
            sfreq=100.0,
            n_times=600,
            seed=9,
            snr_db=10.0,
            n_background=5,
            node_scale="unit",
            leadfield=np.asarray(sim.leadfield),
        )
        np.testing.assert_array_equal(sim.data.data, direct.data.data)
        np.testing.assert_array_equal(sim.sources.data, direct.sources.data)

    def test_the_realised_snr_is_the_target(self):
        sim = run(snr_db=-5.0)
        assert sim.snr_db == pytest.approx(-5.0, abs=1e-9)

    def test_same_seed_same_data_and_the_seed_is_required(self):
        a, b, c = run(seed=1), run(seed=1), run(seed=2)
        np.testing.assert_array_equal(a.data.data, b.data.data)
        assert not np.array_equal(a.data.data, c.data.data)
        with pytest.raises(TypeError):
            simulate_on_head(chain(ROIS[0], ROIS[1]), head(), sfreq=100.0, n_times=100)  # type: ignore[call-arg]


class TestDirectNodes:
    def test_a_direct_node_bypasses_the_head_model_and_follows_the_sensors(self):
        nodes = [dk_node(ROIS[0]), direct_node("STN-L", region="STN", hemisphere="lh")]
        net = Network(nodes=nodes, edges=[Edge(ROIS[0], "STN-L", strength=0.4, delay_ms=10.0)])
        sim = run(network=net)
        model = head()
        assert sim.data.n_channels == model.n_channels + 1
        assert sim.data.names[-1] == "STN-L"
        assert sim.leadfield.shape[1] == 1 + 10  # one cortical node + the default background


class TestErrors:
    def test_a_node_that_is_not_a_region_of_the_head_model_is_named(self):
        net = chain(ROIS[0], "frontalpole-lh")
        with pytest.raises(ValueError, match="frontalpole-lh"):
            run(network=net)

    def test_too_few_sources_for_the_background_is_an_error(self):
        model = head(n_sources=45)  # 45 sources, 16 of them inside the two nodes' regions
        with pytest.raises(ValueError, match="background"):
            run(model=model, n_background=30)
        assert run(model=model, n_background=29).data.n_channels == model.n_channels

    def test_the_head_must_be_a_headmodel(self):
        with pytest.raises(TypeError, match="HeadModel"):
            simulate_on_head(
                chain(ROIS[0], ROIS[1]), np.ones((3, 3)), sfreq=100.0, n_times=100, seed=0
            )  # type: ignore[arg-type]


class TestProvenance:
    def test_it_records_the_head_model_and_the_choices_made(self):
        model = head()
        sim = run(model=model, n_background=4)
        params = sim.provenance.parameters
        assert params["head_model"]["modality"] == "eeg"
        assert params["head_model"]["forward_file"] == "made up"
        assert params["head_model"]["n_channels"] == model.n_channels
        assert params["leadfield"] == "head model"
        assert len(params["head_model"]["background_sources"]) == 4
        assert params["head_model"]["node_visibility"] == {
            n: pytest.approx(float(model.visibility[model.roi_names.index(n)]))
            for n in (ROIS[0], ROIS[1])
        }


@pytest.mark.sample_data
class TestOnTheSampleHeadModel:
    @pytest.mark.parametrize("modality, n_channels", [("eeg", 59), ("mag", 102), ("grad", 203)])
    def test_a_chain_through_the_real_head_model(self, modality, n_channels):
        from dirconn.headmodel import load_head_model

        model = load_head_model(modality)
        names = ("superiorfrontal-lh", "precuneus-rh", "lateraloccipital-lh")
        net = chain(*names)
        sim = simulate_on_head(
            net, model, sfreq=200.0, n_times=2000, seed=4, snr_db=0.0, n_background=30
        )
        assert sim.data.n_channels == n_channels
        assert {c.type for c in sim.data.channels} == {modality}
        assert sim.snr_db == pytest.approx(0.0, abs=1e-9)
        rows = [model.roi_names.index(n) for n in names]
        # the default source model is partly coherent (c = 0.7): the nodes enter with 0.7 of the
        # region's mean topography; the rest of the region's activity is local
        np.testing.assert_allclose(sim.leadfield[:, :3], 0.7 * model.roi_leadfield[:, rows])
        assert np.isfinite(sim.data.data).all()
        assert sim.provenance.parameters["head_model"]["modality"] == modality

    @pytest.mark.parametrize("source_model", ["patch", "focal"])
    def test_the_other_source_models_use_the_regions_topography_or_centre_source(
        self, source_model
    ):
        from dirconn.headmodel import load_head_model

        model = load_head_model("grad")
        names = ("superiorfrontal-lh", "precuneus-rh", "lateraloccipital-lh")
        sim = simulate_on_head(
            chain(*names),
            model,
            sfreq=100.0,
            n_times=500,
            seed=4,
            n_background=5,
            source_model=source_model,
        )
        rows = [model.roi_names.index(n) for n in names]
        expected = (
            model.roi_leadfield[:, rows]
            if source_model == "patch"
            else model.gain[:, model.roi_centre[rows]]
        )
        np.testing.assert_allclose(sim.leadfield[:, :3], expected)


def identical_vertex_head(n_channels=8, seed=0):
    """Five regions of eight sources; every source of a region has the same gain."""
    rng = np.random.default_rng(seed)
    topo = rng.standard_normal((n_channels, 5))
    gain = np.zeros((n_channels, 45))
    source_roi = np.full(45, -1)
    for k in range(5):
        gain[:, k * 8 : (k + 1) * 8] = topo[:, [k]]
        source_roi[k * 8 : (k + 1) * 8] = k
    gain[:, 40:] = rng.standard_normal((n_channels, 5))
    norms = np.linalg.norm(topo, axis=0)
    return HeadModel(
        modality="eeg",
        channel_names=tuple(f"E{i}" for i in range(n_channels)),
        roi_names=ROIS,
        roi_leadfield=topo,
        scale=1.0,
        visibility=norms / np.median(norms),
        gain=gain,
        source_roi=source_roi,
        source_flip=np.where(source_roi >= 0, 1, 0),
        roi_centre=np.array([k * 8 + 3 for k in range(5)]),
        forward=None,
        info=None,
        provenance={"forward_file": "made up"},
    )


class TestSourceModels:
    def test_the_default_is_a_partly_coherent_patch_with_coherence_seven_tenths(self):
        sim = run(n_background=3)
        info = sim.provenance.parameters["head_model"]
        assert info["source_model"] == "partial" and info["coherence"] == 0.7

    def test_a_patch_is_the_whole_labels_mean_topography_times_the_activity(self):
        model = head()
        sim = run(model=model, source_model="patch")
        assert sim.provenance.parameters["head_model"]["coherence"] is None
        expected = np.einsum("sk,bkt->bst", model.roi_leadfield[:, [0, 1]], sim.sources.data)
        np.testing.assert_allclose(sim.signal, expected, rtol=1e-12, atol=1e-12)

    def test_a_focal_source_has_the_gain_of_the_regions_centre_source(self):
        model = head()
        sim = run(model=model, source_model="focal")
        expected = np.einsum(
            "sk,bkt->bst", model.gain[:, model.roi_centre[[0, 1]]], sim.sources.data
        )
        np.testing.assert_allclose(sim.signal, expected, rtol=1e-12, atol=1e-12)

    def test_full_coherence_is_the_patch(self):
        model = head()
        partial = run(model=model, source_model="partial", coherence=1.0)
        patch = run(model=model, source_model="patch")
        np.testing.assert_allclose(partial.signal, patch.signal, rtol=1e-12, atol=1e-12)

    def test_a_partial_patch_is_the_coherent_part_plus_independent_local_activity(self):
        model = head()
        c = 0.6
        sim = run(model=model, source_model="partial", coherence=c, n_times=8000)
        coherent = np.einsum("sk,bkt->bst", c * model.roi_leadfield[:, [0, 1]], sim.sources.data)
        local = sim.signal - coherent
        assert np.abs(local).max() > 0  # there is local activity
        # it is unrelated to the node activity (8000 samples: chance correlation about 0.011)
        for k in range(2):
            for sensor in range(model.n_channels):
                r = np.corrcoef(local[0, sensor], sim.sources.data[0, k])[0, 1]
                assert abs(r) < 0.06

    def test_the_local_activity_has_the_variance_of_the_mean_of_independent_vertices(self):
        # Every vertex of a region has the same gain g, so the sensor sees g times the mean of the
        # vertices: a variance of g**2 * (1 - c**2) / n_sources for the local part. The variance of
        # 1/f activity fluctuates a lot from one realisation to the next (a single seed was off by
        # a standard deviation of 12% in 40 trials), so the ratio is averaged over 40 seeds: its
        # mean was 1.002 with a standard error of 0.013.
        model = identical_vertex_head()
        net = chain(ROIS[0], ROIS[1])
        c = 0.7
        expected = (
            (1 - c**2) / 8 * (model.roi_leadfield[:, 0] ** 2 + model.roi_leadfield[:, 1] ** 2)
        )
        ratios = []
        for seed in range(40):
            sim = simulate_on_head(
                net,
                model,
                sfreq=100.0,
                n_times=20000,
                seed=seed,
                snr_db=40.0,
                n_background=0,
                source_model="partial",
                coherence=c,
            )
            design = sim.sources.data[0].T
            fit = design @ np.linalg.lstsq(design, sim.signal[0].T, rcond=None)[0]
            ratios.append((sim.signal[0].T - fit).var(axis=0) / expected)
        assert np.mean(ratios) == pytest.approx(1.0, abs=0.05)

    def test_the_local_activity_is_part_of_the_signal_and_the_snr_stays_exact(self):
        sim = run(source_model="partial", snr_db=-4.0)
        assert sim.snr_db == pytest.approx(-4.0, abs=1e-9)

    def test_same_seed_same_data_for_every_model(self):
        for model_name in ("partial", "focal", "patch"):
            a = run(seed=5, source_model=model_name)
            b = run(seed=5, source_model=model_name)
            np.testing.assert_array_equal(a.data.data, b.data.data)

    def test_epoched_data_get_independent_local_activity_in_every_trial(self):
        # 1/f series are strongly autocorrelated, so one pair of trials can correlate by chance at
        # 0.1; the mean over 59 neighbouring pairs of 60 trials must be close to zero
        model = head()
        sim = run(model=model, source_model="partial", n_trials=60, n_times=300)
        coherent = np.einsum("sk,bkt->bst", 0.7 * model.roi_leadfield[:, [0, 1]], sim.sources.data)
        local = sim.signal - coherent
        r = [np.corrcoef(local[i].ravel(), local[i + 1].ravel())[0, 1] for i in range(59)]
        assert abs(np.mean(r)) < 0.05
        assert not np.allclose(local[0], local[1])

    def test_invalid_models_and_coherences_are_refused_with_the_reason(self):
        with pytest.raises(ValueError, match="partial.*focal.*patch"):
            run(source_model="smeared")
        with pytest.raises(ValueError, match="coherence"):
            run(source_model="focal", coherence=0.5)
        with pytest.raises(ValueError, match="coherence"):
            run(source_model="patch", coherence=0.5)
        for bad in (0.0, -0.2, 1.5, float("nan")):
            with pytest.raises(ValueError, match="coherence"):
                run(source_model="partial", coherence=bad)

    def test_the_provenance_records_the_source_model(self):
        info = run(source_model="partial", coherence=0.4).provenance.parameters["head_model"]
        assert info["source_model"] == "partial" and info["coherence"] == 0.4
        assert run(source_model="focal").provenance.parameters["head_model"]["coherence"] is None


class TestExtraSignal:
    def test_it_is_added_to_the_signal_and_counted_in_the_snr(self):
        base = dict(sfreq=100.0, n_times=600, seed=3, snr_db=6.0, n_background=2)
        net = chain(ROIS[0], ROIS[1])
        lead = np.random.default_rng(0).standard_normal((12, 4))
        plain = simulate(net, leadfield=lead, **base)
        extra = np.random.default_rng(1).standard_normal(plain.signal.shape)
        out = simulate(net, leadfield=lead, extra_signal=extra, **base)
        np.testing.assert_allclose(out.signal, plain.signal + extra)
        assert out.snr_db == pytest.approx(6.0, abs=1e-9)
        assert out.provenance.parameters["extra_signal"] is True
        assert plain.provenance.parameters["extra_signal"] is False

    def test_the_shape_must_match(self):
        net = chain(ROIS[0], ROIS[1])
        lead = np.random.default_rng(0).standard_normal((12, 4))
        with pytest.raises(ValueError, match="extra_signal"):
            simulate(
                net,
                leadfield=lead,
                extra_signal=np.zeros((1, 11, 600)),
                sfreq=100.0,
                n_times=600,
                seed=0,
                n_background=2,
            )

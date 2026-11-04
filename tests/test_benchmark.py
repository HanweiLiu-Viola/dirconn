import numpy as np
import pytest

from dirconn import DTF, GC, Edge, Network, direct_node, dk_node
from dirconn.benchmark import (
    evaluate_stages,
    representative_sensors,
    roi_observations,
    sensor_series,
)
from dirconn.simulation import simulate_on_head
from helpers import ROIS, chain, head


def orthogonal_topographies(n_channels=10):
    """Region k lives on sensors 2k (strongly) and 2k + 1 (a little less), nowhere else."""
    topo = np.full((n_channels, 5), 0.02)
    for k in range(5):
        topo[2 * k, k], topo[2 * k + 1, k] = 1.0, 0.8
    return topo


def run_sim(model, network, **kwargs):
    # These tests are about the stages and about node power, not about the source model, so they
    # use whole-label sources; the findings in docs/benchmark.md were measured that way.
    params = {
        "sfreq": 100.0,
        "n_times": 4000,
        "seed": 3,
        "snr_db": 20.0,
        "n_background": 5,
        "source_model": "patch",
    }
    params.update(kwargs)
    return simulate_on_head(network, model, **params)


class TestRepresentativeSensors:
    def test_each_node_gets_the_sensor_where_its_region_is_strongest(self):
        model = head(n_channels=10, topographies=orthogonal_topographies())
        rep = representative_sensors(model, chain(ROIS[0], ROIS[1], ROIS[2]))
        assert rep.node_names == (ROIS[0], ROIS[1], ROIS[2])
        assert rep.sensor_names == ("EEG00", "EEG02", "EEG04")
        assert rep.sensor_index == (0, 2, 4)

    def test_two_regions_wanting_the_same_sensor_get_distinct_sensors(self):
        topo = np.full((6, 5), 0.0)
        topo[:2, 0] = [1.0, 0.9]  # region 0 peaks on sensor 0 but is nearly as strong on 1
        topo[:2, 1] = [1.0, 0.2]  # region 1 peaks on sensor 0 and has nothing on 1
        topo[2:, 2] = 1.0
        model = head(n_channels=6, topographies=topo + 1e-3)
        rep = representative_sensors(model, chain(ROIS[0], ROIS[1]))
        assert len(set(rep.sensor_index)) == 2
        # the assignment that maximises the total share gives sensor 1 to region 0
        assert rep.sensor_index == (1, 0)

    def test_dominance_and_purity_are_the_shares_defined_in_the_docstring(self):
        topo = np.zeros((4, 5)) + 1e-9
        topo[:, 0] = [3.0, 1.0, 0.0, 0.0]
        topo[:, 1] = [1.0, 2.0, 1.5, 0.0]
        model = head(n_channels=4, topographies=topo)
        rep = representative_sensors(model, chain(ROIS[0], ROIS[1]))
        assert rep.sensor_index == (0, 1)
        # dominance: the sensor's share of the region's topography energy
        np.testing.assert_allclose(rep.dominance, [9.0 / (9 + 1), 4.0 / (1 + 4 + 2.25)], rtol=1e-6)
        # purity: the region's share of the energy that the network's regions put on the sensor
        np.testing.assert_allclose(rep.purity, [9.0 / (9 + 1), 4.0 / (4 + 1)], rtol=1e-6)

    def test_purity_weights_the_regions_by_the_power_of_their_nodes(self):
        # Topographies are almost orthogonal (a leak of 0.02), but a node 100 times stronger than
        # its neighbours still contaminates their sensors: 0.02 ** 2 * 100 is not negligible.
        model = head(n_channels=10, topographies=orthogonal_topographies())
        net = chain(ROIS[0], ROIS[1], ROIS[2])
        equal = representative_sensors(model, net)
        weighted = representative_sensors(model, net, node_power=[1.0, 100.0, 100.0])
        np.testing.assert_allclose(equal.purity, 1.0 / (1.0 + 2 * 0.0004), rtol=1e-3)
        assert weighted.purity[0] == pytest.approx(1.0 / (1.0 + 2 * 0.0004 * 100.0), rel=1e-3)
        assert weighted.purity[0] < 0.93 < equal.purity[0]

    @pytest.mark.parametrize("power", [[1.0, 2.0], [1.0, 0.0, 1.0], [1.0, -1.0, 1.0]])
    def test_node_power_needs_one_positive_value_per_cortical_node(self, power):
        model = head(n_channels=10, topographies=orthogonal_topographies())
        with pytest.raises(ValueError, match="node_power"):
            representative_sensors(model, chain(ROIS[0], ROIS[1], ROIS[2]), node_power=power)

    def test_a_well_separated_head_is_purer_than_a_mixed_one(self):
        net = chain(ROIS[0], ROIS[1], ROIS[2])
        clean = representative_sensors(
            head(n_channels=10, topographies=orthogonal_topographies()), net
        )
        mixed_topo = np.random.default_rng(0).standard_normal((10, 5))
        mixed = representative_sensors(head(n_channels=10, topographies=mixed_topo), net)
        assert clean.purity.min() > 0.9
        assert mixed.purity.mean() < clean.purity.mean()

    def test_direct_nodes_have_no_representative_sensor(self):
        nodes = [dk_node(ROIS[0]), dk_node(ROIS[1]), direct_node("STN-L", region="STN")]
        net = Network(nodes=nodes, edges=[Edge(ROIS[0], "STN-L", 0.4, 10.0)])
        rep = representative_sensors(head(), net)
        assert rep.node_names == (ROIS[0], ROIS[1])

    def test_more_nodes_than_sensors_is_an_error(self):
        with pytest.raises(ValueError, match="sensors"):
            representative_sensors(head(n_channels=2), chain(ROIS[0], ROIS[1], ROIS[2]))

    def test_a_node_that_is_not_a_region_is_named(self):
        with pytest.raises(ValueError, match="frontalpole-lh"):
            representative_sensors(head(), chain(ROIS[0], "frontalpole-lh"))

    def test_a_network_without_cortical_nodes_is_an_error(self):
        net = Network(nodes=[direct_node("A"), direct_node("B")], edges=[Edge("A", "B", 0.4, 10.0)])
        with pytest.raises(ValueError, match="cortical"):
            representative_sensors(head(), net)


class TestRoiObservations:
    def test_it_is_the_node_activity_plus_noise_at_the_requested_snr(self):
        sim = run_sim(head(), chain(ROIS[0], ROIS[1], ROIS[2]))
        obs = roi_observations(sim, snr_db=7.0, seed=1)
        assert obs.names == (ROIS[0], ROIS[1], ROIS[2])
        noise = obs.data - sim.sources.data
        for k in range(3):
            snr = 10 * np.log10(np.mean(sim.sources.data[:, k] ** 2) / np.mean(noise[:, k] ** 2))
            assert snr == pytest.approx(7.0, abs=1e-9)

    def test_it_is_reproducible_and_the_seed_is_required(self):
        sim = run_sim(head(), chain(ROIS[0], ROIS[1]))
        a = roi_observations(sim, snr_db=5.0, seed=4)
        b = roi_observations(sim, snr_db=5.0, seed=4)
        c = roi_observations(sim, snr_db=5.0, seed=5)
        np.testing.assert_array_equal(a.data, b.data)
        assert not np.array_equal(a.data, c.data)
        with pytest.raises(TypeError):
            roi_observations(sim, snr_db=5.0)  # type: ignore[call-arg]

    def test_direct_nodes_are_included_in_the_network_order(self):
        nodes = [dk_node(ROIS[0]), direct_node("STN-L", region="STN"), dk_node(ROIS[1])]
        net = Network(nodes=nodes, edges=[Edge(ROIS[0], "STN-L", 0.4, 10.0)])
        obs = roi_observations(run_sim(head(), net), snr_db=10.0, seed=0)
        assert obs.names == (ROIS[0], "STN-L", ROIS[1])


class TestSensorSeries:
    def test_the_channels_follow_the_node_order_with_direct_nodes_as_they_are(self):
        model = head(n_channels=10, topographies=orthogonal_topographies())
        nodes = [dk_node(ROIS[1]), direct_node("STN-L", region="STN"), dk_node(ROIS[0])]
        net = Network(nodes=nodes, edges=[Edge(ROIS[1], "STN-L", 0.4, 10.0)])
        sim = run_sim(model, net)
        rep = representative_sensors(model, net)
        series = sensor_series(sim, rep)
        assert series.names == ("EEG02", "STN-L", "EEG00")
        for name in series.names:
            np.testing.assert_array_equal(
                series.data[:, series.names.index(name)],
                sim.data.data[:, sim.data.names.index(name)],
            )


@pytest.fixture(scope="module")
def clean_chain():
    model = head(n_channels=10, topographies=orthogonal_topographies())
    net = chain(ROIS[0], ROIS[1], ROIS[2])
    return model, net, run_sim(model, net)


class TestEvaluateStages:
    @pytest.fixture
    def clean(self, clean_chain):
        return clean_chain

    def test_both_stages_recover_a_chain_through_a_clean_head(self, clean):
        model, net, sim = clean
        out = evaluate_stages(GC(order=2), sim, model, seed=0)
        assert out["roi"]["auroc"] == 1.0 and out["sensor"]["auroc"] == 1.0
        assert out["roi"]["direction_accuracy"] == 1.0
        assert out["sensor"]["direction_accuracy"] == 1.0

    def test_the_result_has_both_stages_the_agreement_and_the_sensor_facts(self, clean):
        model, net, sim = clean
        out = evaluate_stages(GC(order=2), sim, model, seed=0)
        assert set(out) >= {"roi", "sensor", "agreement", "representative", "band"}
        assert out["band"] is None  # a time-domain estimator needs no band
        # measured over 10 seeds, unit node power: minimum 0.83 at 30 dB, 1.0 at 20 dB
        assert out["agreement"]["spearman"] > 0.7
        assert out["representative"]["sensors"] == ["EEG00", "EEG02", "EEG04"]
        assert len(out["representative"]["dominance"]) == 3
        assert len(out["representative"]["node_power"]) == 3
        np.testing.assert_allclose(out["representative"]["node_power"], 1.0, rtol=1e-9)
        assert out["roi"]["order"] is not None

    def test_a_spectral_estimator_is_scored_in_the_band_of_the_edges(self, clean):
        model, _, _ = clean
        names = (ROIS[0], ROIS[1], ROIS[2])
        net = Network(
            nodes=[dk_node(n) for n in names],
            edges=[
                Edge(names[0], names[1], 0.4, 10.0, (6.0, 14.0)),
                Edge(names[1], names[2], 0.4, 10.0, (6.0, 14.0)),
            ],
        )
        sim = run_sim(model, net)
        out = evaluate_stages(DTF(order=2), sim, model, seed=0)
        assert out["band"] == (6.0, 14.0)
        assert out["roi"]["band"] == (6.0, 14.0)
        assert out["roi"]["auroc"] > 0.8

    def test_a_spectral_estimator_on_a_broadband_network_needs_a_band(self, clean):
        model, _, sim = clean
        with pytest.raises(ValueError, match="broadband"):
            evaluate_stages(DTF(order=2), sim, model, seed=0)
        out = evaluate_stages(DTF(order=2), sim, model, band=(5.0, 20.0), seed=0)
        assert out["band"] == (5.0, 20.0)

    def test_the_roi_snr_defaults_to_the_one_of_the_simulation_and_can_be_set(self, clean):
        model, _, sim = clean
        default = evaluate_stages(GC(order=2), sim, model, seed=0)
        noisy = evaluate_stages(GC(order=2), sim, model, seed=0, roi_snr_db=-20.0)
        assert default["roi_snr_db"] == 20.0 and noisy["roi_snr_db"] == -20.0
        assert noisy["roi"]["separation"] < default["roi"]["separation"]

    def test_it_is_reproducible_and_the_seed_is_required(self, clean):
        model, _, sim = clean
        a = evaluate_stages(GC(order=2), sim, model, seed=2)
        b = evaluate_stages(GC(order=2), sim, model, seed=2)
        assert (
            a["roi"]["auroc"] == b["roi"]["auroc"]
            and a["roi"]["separation"] == b["roi"]["separation"]
        )
        with pytest.raises(TypeError):
            evaluate_stages(GC(order=2), sim, model)  # type: ignore[call-arg]

    def test_types_are_checked(self, clean):
        model, _, sim = clean
        with pytest.raises(TypeError, match="Estimator"):
            evaluate_stages(lambda d: d, sim, model, seed=0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="HeadModel"):
            evaluate_stages(GC(order=2), sim, np.ones((2, 2)), seed=0)  # type: ignore[arg-type]


def test_unequal_node_power_is_what_degrades_the_sensor_stage():
    # Measured: in a chain every step amplifies, the sensor SNR is dominated by the strongest
    # node and the weakest one is buried (about 3 dB at its sensor against 20 dB nominal). Measured
    # over 10 seeds without background: AUROC 0.75 and direction accuracy 0.5 at the sensor stage in
    # every seed without scaling, AUROC 1.0 with unit node power. The ROI stage is 1.0 in both.
    model = head(n_channels=10, topographies=orthogonal_topographies())
    net = chain(ROIS[0], ROIS[1], ROIS[2])
    for seed in range(3):
        raw = evaluate_stages(
            GC(order=2),
            run_sim(model, net, seed=seed, n_background=0, node_scale="none"),
            model,
            seed=seed,
        )
        unit = evaluate_stages(
            GC(order=2),
            run_sim(model, net, seed=seed, n_background=0, node_scale="unit"),
            model,
            seed=seed,
        )
        assert raw["roi"]["auroc"] == 1.0 and unit["roi"]["auroc"] == 1.0
        assert raw["sensor"]["auroc"] < 1.0
        assert unit["sensor"]["auroc"] == 1.0
        assert (
            max(raw["representative"]["node_power"]) / min(raw["representative"]["node_power"]) > 50
        )


# --- task 13: hidden nodes, the stimulation artifact, the per-frequency curve -------------------

from dataclasses import dataclass  # noqa: E402

from dirconn import Connectivity  # noqa: E402
from dirconn.benchmark import add_artifact  # noqa: E402
from dirconn.estimator import Estimator, MethodInfo  # noqa: E402
from dirconn.scenario import ArtifactSpec, stimulation_artifact  # noqa: E402
from helpers import Spy  # noqa: E402

A, B, C, H = ROIS[0], ROIS[1], ROIS[2], ROIS[3]


@dataclass(frozen=True)
class Oracle(Estimator):
    """Returns a fixed score matrix over the channels it gets: a stand-in for a perfect method."""

    info = MethodInfo(name="oracle", backend="test stub", directed=True)
    edges: tuple = ()

    def _fit(self, data):
        n = data.n_channels
        values = np.zeros((n, n))
        for i, j in self.edges:
            values[i, j] = 1.0
        np.fill_diagonal(values, np.nan)
        return Connectivity.from_array(values, source=data.names, provenance=self._provenance(data))


@pytest.fixture(scope="module")
def hidden_driver():
    """H drives A and B, A drives C. H is simulated but hidden from the methods."""
    model = head(n_channels=10, topographies=orthogonal_topographies())
    net = Network(
        nodes=[dk_node(n) for n in (H, A, B, C)],
        edges=[
            Edge(H, A, 0.5, 10.0),
            Edge(H, B, 0.5, 20.0),
            Edge(A, C, 0.4, 10.0),
        ],
    )
    return model, net, run_sim(model, net)


class TestHiddenNodes:
    def test_the_estimators_see_only_the_analysed_nodes_at_every_stage(self, hidden_driver):
        model, _, sim = hidden_driver
        spy = Spy()
        out = evaluate_stages(spy, sim, model, seed=0, hidden=[H])
        assert len(spy.seen) == 2 and all(d.n_channels == 3 for d in spy.seen)
        assert spy.seen[0].names == (A, B, C)
        assert out["analysed"] == [A, B, C] and out["hidden"] == [H]
        assert out["agreement"]["n_pairs"] == 6  # 3 nodes, off-diagonal

    def test_the_hidden_node_still_takes_part_in_the_choice_of_sensors(self, hidden_driver):
        model, _, sim = hidden_driver
        out = evaluate_stages(Spy(), sim, model, seed=0, hidden=[H])
        assert out["representative"]["nodes"] == [H, A, B, C]
        assert len(set(out["representative"]["sensors"])) == 4

    def test_the_truth_is_the_subgraph_between_the_analysed_nodes(self, hidden_driver):
        model, _, sim = hidden_driver
        # in (A, B, C) the only true edge is A -> C (positions 0 -> 2); H -> A and H -> B are gone
        perfect = Oracle(edges=((0, 2),))
        out = evaluate_stages(perfect, sim, model, seed=0, hidden=[H])
        assert out["roi"]["auroc"] == 1.0 and out["sensor"]["auroc"] == 1.0
        assert out["roi"]["direction_accuracy"] == 1.0
        # the spurious A -> B (a hidden common cause) is a false positive: scoring it as an edge
        # makes the AUROC drop below one
        fooled = Oracle(edges=((0, 2), (0, 1)))
        assert evaluate_stages(fooled, sim, model, seed=0, hidden=[H])["roi"]["auroc"] < 1.0

    def test_without_hidden_nodes_nothing_changes(self, clean_chain):
        model, _, sim = clean_chain
        plain = evaluate_stages(GC(order=2), sim, model, seed=0)
        empty = evaluate_stages(GC(order=2), sim, model, seed=0, hidden=[])
        assert plain["roi"]["auroc"] == empty["roi"]["auroc"]
        assert empty["analysed"] == [ROIS[0], ROIS[1], ROIS[2]] and empty["hidden"] == []

    def test_a_hidden_name_that_is_not_a_node_is_named(self, clean_chain):
        model, _, sim = clean_chain
        with pytest.raises(ValueError, match="nowhere-lh"):
            evaluate_stages(GC(order=2), sim, model, seed=0, hidden=["nowhere-lh"])

    def test_hiding_all_but_one_node_is_refused(self, clean_chain):
        model, _, sim = clean_chain
        with pytest.raises(ValueError, match="hidden"):
            evaluate_stages(GC(order=2), sim, model, seed=0, hidden=[ROIS[0], ROIS[1]])


SPEC = ArtifactSpec(rate_hz=2.0, freq_hz=30.0, decay_ms=40.0, amplitude=1.0, anchor=B, n_sensors=3)


class TestAddArtifact:
    def test_it_adds_the_artifact_to_the_sensors_of_its_group_only(self, clean_chain):
        model, _, sim = clean_chain
        out = add_artifact(sim, model, SPEC)
        diff = np.asarray(out.data.data) - np.asarray(sim.data.data)
        changed = np.flatnonzero(np.abs(diff).sum(axis=(0, 2)) > 0)
        assert len(changed) == 3
        topography = model.roi_leadfield[:, model.roi_names.index(B)]
        strongest = np.argsort(-np.abs(topography))[:3]
        assert sorted(changed.tolist()) == sorted(strongest.tolist())

    def test_it_has_the_amplitude_relative_to_the_signal_of_the_group(self, clean_chain):
        model, _, sim = clean_chain
        spec = ArtifactSpec(2.0, 30.0, 40.0, 0.5, B, 3)
        out = add_artifact(sim, model, spec)
        diff = np.asarray(out.data.data) - np.asarray(sim.data.data)
        group = np.abs(diff).sum(axis=(0, 2)) > 0
        ratio = np.mean(diff[:, group] ** 2) / np.mean(np.asarray(sim.signal)[:, group] ** 2)
        assert ratio == pytest.approx(0.5**2, rel=1e-6)

    def test_it_equals_the_scenario_helper_on_the_sensor_signal(self, clean_chain):
        model, _, sim = clean_chain
        topography = model.roi_leadfield[:, model.roi_names.index(B)]
        expected = stimulation_artifact(SPEC, topography, sim.signal, sim.data.sfreq)
        out = add_artifact(sim, model, SPEC)
        np.testing.assert_allclose(
            np.asarray(out.data.data) - np.asarray(sim.data.data), expected, atol=1e-12
        )

    def test_the_original_simulation_signal_noise_and_snr_are_untouched(self, clean_chain):
        model, _, sim = clean_chain
        before = np.array(sim.data.data)
        out = add_artifact(sim, model, SPEC)
        np.testing.assert_array_equal(np.asarray(sim.data.data), before)
        assert out.signal is sim.signal and out.noise is sim.noise
        assert out.snr_db == sim.snr_db

    def test_the_region_level_stage_does_not_see_it_but_the_sensor_stage_does(self, clean_chain):
        model, _, sim = clean_chain
        out = add_artifact(sim, model, SPEC)
        np.testing.assert_array_equal(
            np.asarray(roi_observations(out, snr_db=10.0, seed=0).data),
            np.asarray(roi_observations(sim, snr_db=10.0, seed=0).data),
        )
        rep = representative_sensors(model, out.network)
        assert not np.array_equal(
            np.asarray(sensor_series(out, rep).data), np.asarray(sensor_series(sim, rep).data)
        )

    def test_the_provenance_records_the_artifact(self, clean_chain):
        model, _, sim = clean_chain
        out = add_artifact(sim, model, SPEC)
        assert out.provenance.parameters["artifact"]["anchor"] == B
        assert "artifact" not in sim.provenance.parameters

    def test_an_anchor_that_is_not_a_region_of_the_head_is_named(self, clean_chain):
        model, _, sim = clean_chain
        bad = ArtifactSpec(2.0, 30.0, 40.0, 1.0, ROIS[0], 3)
        object.__setattr__(bad, "anchor", "nowhere-lh")
        with pytest.raises(ValueError, match="nowhere-lh"):
            add_artifact(sim, model, bad)


class TestByFrequency:
    @pytest.fixture
    def banded(self, clean_chain):
        model, _, _ = clean_chain
        names = (ROIS[0], ROIS[1], ROIS[2])
        net = Network(
            nodes=[dk_node(n) for n in names],
            edges=[
                Edge(names[0], names[1], 0.4, 10.0, (6.0, 14.0)),
                Edge(names[1], names[2], 0.4, 10.0, (6.0, 14.0)),
            ],
        )
        return model, run_sim(model, net)

    def test_a_spectral_estimator_gets_a_curve_in_every_stage_card(self, banded):
        model, sim = banded
        out = evaluate_stages(DTF(order=2), sim, model, seed=0, by_frequency=True)
        for stage in ("roi", "sensor"):
            curve = out[stage]["auroc_by_frequency"]
            assert len(curve["freqs"]) == len(curve["auroc"]) > 10
            assert curve["freqs"] == sorted(curve["freqs"])
            freqs, values = np.array(curve["freqs"]), np.array(curve["auroc"])
            assert np.isfinite(values[(freqs >= 6) & (freqs <= 14)]).all()
            assert np.isnan(values[freqs > 30]).all()  # a banded truth has no AUROC outside it

    def test_the_values_are_rounded_to_four_decimals(self, clean_chain):
        # five nodes give 4 edges against 16 non-edges, so an AUROC is a multiple of 1/64 =
        # 0.015625: it has more than four decimals unless it is rounded
        model, _, _ = clean_chain
        names = ROIS
        net = Network(
            nodes=[dk_node(n) for n in names],
            edges=[
                Edge(a, b, 0.4, 10.0, (6.0, 14.0))
                for a, b in zip(names[:-1], names[1:], strict=True)
            ],
        )
        sim = run_sim(model, net)
        out = evaluate_stages(DTF(order=2), sim, model, seed=0, by_frequency=True, roi_snr_db=-5.0)
        values = np.array(out["roi"]["auroc_by_frequency"]["auroc"])
        values = values[np.isfinite(values)]
        # the test is only sensitive if some value is not a short decimal to begin with
        assert (np.abs(values - np.round(values, 2)) > 1e-3).any()
        np.testing.assert_allclose(values, np.round(values, 4), atol=0)

    def test_it_is_off_by_default_and_absent_for_time_domain_estimators(self, banded):
        model, sim = banded
        assert "auroc_by_frequency" not in evaluate_stages(DTF(order=2), sim, model, seed=0)["roi"]
        out = evaluate_stages(GC(order=2), sim, model, seed=0, by_frequency=True)
        assert "auroc_by_frequency" not in out["roi"]


class TestStoreScores:
    def test_it_is_off_by_default(self, clean_chain):
        model, _, sim = clean_chain
        out = evaluate_stages(GC(order=2), sim, model, seed=0)
        assert "scores" not in out["roi"] and "scores" not in out["sensor"]

    def test_every_stage_card_gets_the_score_matrix_of_the_analysed_nodes(self, clean_chain):
        model, _, sim = clean_chain
        out = evaluate_stages(GC(order=2), sim, model, seed=0, store_scores=True)
        for stage in ("roi", "sensor"):
            matrix = np.array(out[stage]["scores"])
            assert matrix.shape == (3, 3)
            assert np.isnan(np.diag(matrix)).all()
        # the chain A -> B -> C: the true edges score higher than the reverse ones, at both stages
        roi = np.array(out["roi"]["scores"])
        assert roi[0, 1] > roi[1, 0] and roi[1, 2] > roi[2, 1]

    def test_the_matrix_is_the_one_the_agreement_is_computed_from(self, clean_chain):
        from scipy.stats import spearmanr

        model, _, sim = clean_chain
        out = evaluate_stages(GC(order=2), sim, model, seed=0, store_scores=True)
        off = ~np.eye(3, dtype=bool)
        a = np.array(out["roi"]["scores"])[off]
        b = np.array(out["sensor"]["scores"])[off]
        assert spearmanr(a, b).statistic == pytest.approx(out["agreement"]["spearman"], abs=1e-4)

    def test_a_spectral_estimator_gets_the_band_reduced_matrix(self, clean_chain):
        model, _, sim = clean_chain
        out = evaluate_stages(DTF(order=2), sim, model, seed=0, band=(5.0, 20.0), store_scores=True)
        matrix = np.array(out["roi"]["scores"])
        assert matrix.shape == (3, 3) and np.isfinite(matrix[~np.eye(3, dtype=bool)]).all()

    def test_with_hidden_nodes_the_matrix_covers_the_analysed_nodes_only(self, hidden_driver):
        model, _, sim = hidden_driver
        out = evaluate_stages(GC(order=2), sim, model, seed=0, hidden=[H], store_scores=True)
        assert np.array(out["roi"]["scores"]).shape == (3, 3)
        assert out["analysed"] == [A, B, C]

    def test_the_values_are_rounded_to_six_decimals(self, clean_chain):
        model, _, sim = clean_chain
        out = evaluate_stages(GC(order=2), sim, model, seed=0, store_scores=True)
        values = np.array(out["roi"]["scores"])
        values = values[np.isfinite(values)]
        np.testing.assert_allclose(values, np.round(values, 6), atol=0)
        assert (np.abs(values - np.round(values, 3)) > 1e-7).any()  # not vacuous

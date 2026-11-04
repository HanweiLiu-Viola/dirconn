"""Stage 3 of the benchmark (forward model, inverse solution, region series); needs sample data."""

import numpy as np
import pytest

from dirconn import GC, Edge, Network, direct_node, dk_node
from dirconn.benchmark import evaluate_stages, source_series
from dirconn.headmodel import load_head_model
from dirconn.simulation import simulate_on_head
from helpers import chain

pytestmark = pytest.mark.sample_data

NODES = ("superiorfrontal-lh", "precuneus-rh", "lateraloccipital-lh")


@pytest.fixture(scope="module")
def grad():
    return load_head_model("grad")


@pytest.fixture(scope="module")
def sim(grad):
    return simulate_on_head(
        chain(*NODES), grad, sfreq=100.0, n_times=6000, seed=1, snr_db=10.0, n_background=30
    )


class TestSourceSeries:
    def test_the_series_follow_the_network_order_with_direct_nodes_as_recorded(self, grad):
        nodes = [dk_node(NODES[1]), direct_node("STN-L", region="STN"), dk_node(NODES[0])]
        net = Network(nodes=nodes, edges=[Edge(NODES[1], "STN-L", 0.4, 10.0)])
        sim = simulate_on_head(net, grad, sfreq=100.0, n_times=1500, seed=2, n_background=10)
        series = source_series(sim, grad)
        assert series.names == (NODES[1], "STN-L", NODES[0])
        np.testing.assert_array_equal(
            series.data[0, 1], sim.data.data[0, sim.data.names.index("STN-L")]
        )

    def test_the_default_is_minimum_norm_with_pca_flip(self, grad, sim):
        a = source_series(sim, grad)
        b = source_series(sim, grad, method="mne", mode="pca_flip")
        np.testing.assert_array_equal(a.data, b.data)

    def test_the_method_changes_the_series(self, grad, sim):
        a = source_series(sim, grad, method="mne")
        b = source_series(sim, grad, method="dspm")
        assert not np.allclose(a.data, b.data)


class TestEvaluateStagesWithTheSourceStage:
    @pytest.fixture(scope="class")
    def result(self, grad, sim):
        return evaluate_stages(
            GC(order=16, check_order_stability=False), sim, grad, seed=0, inverse={}
        )

    def test_all_three_stages_recover_an_easy_chain(self, result):
        # Measured over 5 seeds, gradiometers and EEG, 10 dB and 0 dB, 6000 samples, chain of three
        # regions with coupling 0.4: AUROC 1.00 and direction accuracy 1.00 at every stage. The
        # setting is too easy to separate the stages; the benchmark sweeps are not.
        for stage in ("roi", "sensor", "source"):
            assert result[stage]["auroc"] == 1.0 and result[stage]["direction_accuracy"] == 1.0

    def test_the_result_records_the_inverse_settings_and_the_agreement(self, result):
        assert result["inverse"] == {"method": "mne", "reg": None, "mode": "pca_flip"}
        assert "spearman_roi_source" in result["agreement"]
        assert result["agreement"]["spearman_roi_source"] > 0.5

    def test_without_the_inverse_argument_there_is_no_third_stage(self, grad, sim):
        out = evaluate_stages(GC(order=16, check_order_stability=False), sim, grad, seed=0)
        assert "source" not in out and "inverse" not in out

    def test_the_inverse_settings_are_passed_on_and_checked(self, grad, sim):
        est = GC(order=16, check_order_stability=False)
        out = evaluate_stages(
            est, sim, grad, seed=0, inverse={"method": "dspm", "mode": "mean_flip", "reg": 0.2}
        )
        assert out["inverse"] == {"method": "dspm", "reg": 0.2, "mode": "mean_flip"}
        with pytest.raises(ValueError, match="inverse"):
            evaluate_stages(est, sim, grad, seed=0, inverse={"methd": "mne"})
        with pytest.raises(TypeError, match="inverse"):
            evaluate_stages(est, sim, grad, seed=0, inverse="mne")  # type: ignore[arg-type]

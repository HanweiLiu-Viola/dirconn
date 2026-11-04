import math

import numpy as np
import pytest

from dirconn import Estimator, TimeSeries
from dirconn.evaluation import separation
from dirconn.simulation import network_var
from dirconn.spectral import spectral_radius
from dirconn.validation import (
    METHODS,
    SCENARIOS,
    SFREQ,
    VAR_METHODS,
    A,
    B,
    C,
    D,
    H,
    environment,
    evaluate,
    evaluate_series,
    load_rows,
    render_report,
    run_all,
    save_rows,
    simulate_scenario,
    truth_matrix,
)

EXPECTED_SCENARIOS = [
    "chain",
    "common_driver",
    "collider",
    "bidirectional",
    "feedback_loop",
    "instantaneous",
    "hidden_driver",
    "noise_m10db",
    "noise_m15db",
]


class TestScenarioDefinitions:
    def test_the_reference_scenarios_and_methods(self):
        assert list(SCENARIOS) == EXPECTED_SCENARIOS
        assert list(METHODS) == [
            "gc",
            "gc_pairwise",
            "dtf",
            "pdc",
            "rpdc",
            "gpdc",
            "trgc",
            "imcoh",
            "psi",
            "mtgc",
            "pcmci",
            "te",
        ]
        assert VAR_METHODS == ("gc", "gc_pairwise", "dtf", "pdc", "rpdc", "gpdc")
        assert all(isinstance(factory(), Estimator) for factory in METHODS.values())

    @pytest.mark.parametrize("name", EXPECTED_SCENARIOS)
    def test_every_scenario_is_well_formed_and_stable(self, name):
        scenario = SCENARIOS[name]
        assert scenario.name == name
        assert scenario.description and scenario.expectation
        assert set(scenario.analysed) <= set(scenario.network.names)
        truth = truth_matrix(scenario)
        assert truth.shape == (len(scenario.analysed),) * 2
        assert np.isnan(np.diag(truth)).all()
        model = network_var(scenario.network, SFREQ, scenario.dynamics)
        assert spectral_radius(model.lagged) < 0.98  # near-unit-root scenarios would be unreliable

    def test_every_scenario_has_a_true_edge_and_a_non_edge_among_the_analysed_nodes(self):
        for scenario in SCENARIOS.values():
            truth = np.nan_to_num(truth_matrix(scenario)) != 0
            off = ~np.eye(truth.shape[0], dtype=bool)
            assert truth[off].any() and (~truth[off]).any(), scenario.name

    def test_the_hidden_driver_is_not_analysed_and_its_edges_are_not_in_the_truth(self):
        scenario = SCENARIOS["hidden_driver"]
        assert H in scenario.network.names and H not in scenario.analysed
        truth = np.nan_to_num(truth_matrix(scenario))
        index = {name: i for i, name in enumerate(scenario.analysed)}
        assert truth.sum() == 1.0 and truth[index[A], index[C]] == 1.0

    def test_white_nodes_get_white_dynamics_and_the_rest_oscillate(self):
        dynamics = SCENARIOS["instantaneous"].dynamics
        assert dynamics[A].kind == dynamics[B].kind == "white"
        assert dynamics[C].kind == dynamics[D].kind == "oscillator"
        assert SCENARIOS["chain"].dynamics[A].kind == "oscillator"

    def test_the_isolated_node_is_part_of_every_scenario_except_the_hidden_driver_case(self):
        for name, scenario in SCENARIOS.items():
            if name != "hidden_driver":
                assert D in scenario.analysed


class TestSimulatedSeries:
    def test_the_analysed_regions_come_back_in_order(self):
        series = simulate_scenario(SCENARIOS["hidden_driver"], seed=0, n_times=2000)
        assert isinstance(series, TimeSeries)
        assert series.names == (A, B, C)
        assert series.n_times == 2000 and np.isfinite(series.data).all()

    def test_the_same_seed_gives_the_same_data(self):
        first = simulate_scenario(SCENARIOS["chain"], seed=3, n_times=1000)
        again = simulate_scenario(SCENARIOS["chain"], seed=3, n_times=1000)
        np.testing.assert_array_equal(first.data, again.data)

    def test_observation_noise_has_exactly_the_requested_snr_on_every_channel(self):
        clean_scenario = SCENARIOS["chain"]
        noisy_scenario = SCENARIOS["noise_m10db"]
        clean = simulate_scenario(clean_scenario, seed=1, n_times=4000)
        noisy = simulate_scenario(noisy_scenario, seed=1, n_times=4000)
        noise = noisy.data - clean.data
        snr = 10 * np.log10(np.mean(clean.data**2, axis=(0, 2)) / np.mean(noise**2, axis=(0, 2)))
        np.testing.assert_allclose(snr, -10.0, atol=1e-9)


class TestSeparation:
    def test_it_is_the_weakest_true_edge_over_the_strongest_non_edge(self):
        truth = np.array([[np.nan, 1, 0], [0, np.nan, 1], [0, 0, np.nan]])
        scores = np.array([[0.0, 0.8, 0.2], [0.1, 0.0, 0.5], [0.4, 0.3, 0.0]])
        assert separation(scores, truth) == pytest.approx(0.5 / 0.4)

    def test_below_one_means_a_spurious_edge_beats_a_true_one(self):
        truth = np.array([[np.nan, 1], [0, np.nan]])
        assert separation(np.array([[0.0, 0.2], [0.6, 0.0]]), truth) == pytest.approx(0.2 / 0.6)

    def test_it_is_infinite_when_every_non_edge_scores_zero(self):
        truth = np.array([[np.nan, 1], [0, np.nan]])
        assert separation(np.array([[0.0, 0.3], [0.0, 0.0]]), truth) == math.inf

    def test_it_is_nan_without_edges_or_without_non_edges(self):
        assert math.isnan(separation(np.ones((2, 2)), np.zeros((2, 2))))
        assert math.isnan(separation(np.ones((2, 2)), np.ones((2, 2))))

    def test_cells_without_a_finite_score_are_left_out(self):
        truth = np.array([[np.nan, 1.0, 0.0], [0.0, np.nan, 1.0], [0.0, 0.0, np.nan]])
        scores = np.array([[0.0, 0.8, 0.2], [np.nan, 0.0, np.nan], [0.1, 0.1, 0.0]])
        assert separation(scores, truth) == pytest.approx(0.8 / 0.2)


class TestExtendedScoring:
    def test_the_m5_methods_are_flagged_as_net_or_undirected(self):
        assert METHODS["trgc"]().info.antisymmetric and METHODS["trgc"]().info.directed
        assert METHODS["psi"]().info.antisymmetric and METHODS["psi"]().info.directed
        assert METHODS["imcoh"]().info.antisymmetric and not METHODS["imcoh"]().info.directed
        assert not any(METHODS[m]().info.antisymmetric for m in VAR_METHODS)

    def test_a_net_measure_leaves_a_pair_coupled_both_ways_out(self):
        scenario = SCENARIOS["bidirectional"]  # A <-> B and C -> D
        series = simulate_scenario(scenario, seed=0, n_times=4000)
        result = evaluate_series(scenario, "trgc", series)
        assert result["auroc"] == 1.0  # only C -> D against the non-edges
        assert result["separation"] > 1.0  # finite and positive: A <-> B is not scored
        assert result["direction_accuracy"] == 1.0

    def test_a_true_edge_pointing_the_wrong_way_counts_as_zero(self, monkeypatch):
        import dirconn.validation as validation

        scenario = SCENARIOS["chain"]  # A -> B -> C, D isolated
        series = simulate_scenario(scenario, seed=0, n_times=2000)
        flipped = np.zeros((4, 4))
        flipped[1, 0], flipped[0, 1] = 0.5, -0.5  # A -> B reported as B -> A
        flipped[1, 2], flipped[2, 1] = 0.5, -0.5
        np.fill_diagonal(flipped, np.nan)
        monkeypatch.setattr(validation, "score_matrix", lambda est, data: (flipped, {}))
        assert evaluate_series(scenario, "trgc", series)["separation"] == 0.0

    def test_score_matrix_takes_the_magnitude_of_an_undirected_measure(self):
        from dataclasses import dataclass

        from dirconn import Connectivity
        from dirconn.estimator import MethodInfo
        from dirconn.validation import score_matrix

        @dataclass(frozen=True)
        class Signed(Estimator):
            info = MethodInfo(name="signed", backend="test stub", directed=False)

            def _fit(self, data):
                values = np.array([[np.nan, -0.4, 0.1], [0.4, np.nan, -0.2], [-0.1, 0.2, np.nan]])
                return Connectivity.from_array(
                    values, source=data.names, provenance=self._provenance(data)
                )

        series = simulate_scenario(SCENARIOS["chain"], seed=0, n_times=500).pick([0, 1, 2])
        scores, _ = score_matrix(Signed(), series)
        expected = np.array([[np.nan, 0.4, 0.1], [0.4, np.nan, 0.2], [0.1, 0.2, np.nan]])
        np.testing.assert_allclose(scores, expected)

    def test_an_undirected_measure_counts_the_reverse_cell_of_a_true_edge(self, monkeypatch):
        import dirconn.validation as validation

        scenario = SCENARIOS["chain"]  # A -> B -> C, D isolated
        series = simulate_scenario(scenario, seed=0, n_times=500)
        scores = np.full((4, 4), 0.1)
        scores[1, 0] = scores[0, 1] = scores[2, 1] = scores[1, 2] = 0.9  # both orders of each pair
        np.fill_diagonal(scores, np.nan)
        monkeypatch.setattr(validation, "score_matrix", lambda est, data: (scores, {}))
        assert evaluate_series(scenario, "imcoh", series)["auroc"] == 1.0
        assert evaluate_series(scenario, "gc", series)["auroc"] < 1.0  # directed truth

    def test_an_undirected_measure_counts_every_pair_once(self, monkeypatch):
        import dirconn.validation as validation

        scenario = SCENARIOS["chain"]  # A - B and B - C undirected; A - C, D - x are non-edges
        series = simulate_scenario(scenario, seed=0, n_times=500)
        scores = np.array(
            [[np.nan, 0.9, 0.5, 0.1], [0.0, np.nan, 0.5, 0.1], [0.95, 0.0, np.nan, 0.1], [0.0] * 4]
        )
        scores[3, 3] = np.nan
        monkeypatch.setattr(validation, "score_matrix", lambda est, data: (scores, {}))
        # only the upper triangle counts (the lower one would rank the non-edge A - C first):
        # AB 0.9 (edge), AC 0.5, BC 0.5 (edge), AD/BD/CD 0.1 -> AP = (1 + 2/3) / 2
        assert evaluate_series(scenario, "imcoh", series)["auprc"] == pytest.approx(5 / 6)

    def test_an_undirected_measure_is_scored_on_its_magnitude_without_direction(self):
        scenario = SCENARIOS["collider"]
        series = simulate_scenario(scenario, seed=0, n_times=4000)
        result = evaluate_series(scenario, "imcoh", series)
        assert math.isnan(result["direction_accuracy"])
        assert result["order"] is None


class TestRunAndReport:
    def test_evaluate_returns_the_metrics_and_facts_of_a_fit(self):
        result = evaluate(SCENARIOS["collider"], "gc", seed=0, n_times=4000)
        assert set(result) == {
            "auroc",
            "auprc",
            "direction_accuracy",
            "separation",
            "order",
            "diagnostic_warnings",
        }
        assert 0.0 <= result["auroc"] <= 1.0
        assert isinstance(result["order"], int) and isinstance(result["diagnostic_warnings"], list)

    def test_run_all_shares_one_simulation_per_scenario_and_seed(self):
        rows = run_all(seeds=[0, 1], n_times=3000, scenarios=["chain"], methods=["gc", "pdc"])
        assert len(rows) == 4
        assert {(r["scenario"], r["method"], r["seed"]) for r in rows} == {
            ("chain", "gc", 0),
            ("chain", "gc", 1),
            ("chain", "pdc", 0),
            ("chain", "pdc", 1),
        }
        assert all("auroc" in r for r in rows)

    def test_run_all_reports_progress(self):
        seen = []
        run_all(seeds=[0], n_times=2000, scenarios=["chain"], methods=["gc"], progress=seen.append)
        assert seen and "chain" in seen[0]

    def test_unknown_names_are_rejected(self):
        with pytest.raises(ValueError, match="scenario"):
            run_all(seeds=[0], scenarios=["nope"])
        with pytest.raises(ValueError, match="method"):
            run_all(seeds=[0], methods=["nope"])

    def test_the_rows_survive_a_save_and_load(self, tmp_path):
        rows = run_all(seeds=[0], n_times=2000, scenarios=["chain"], methods=["gc"])
        rows[0]["separation"] = math.inf
        path = tmp_path / "rows.json"
        save_rows(rows, path, n_times=2000, seeds=[0])
        loaded, meta = load_rows(path)
        assert loaded == rows
        assert meta["n_times"] == 2000 and meta["seeds"] == [0]

    def test_the_report_contains_the_tables_the_scenario_text_and_the_environment(self):
        rows = run_all(
            seeds=[0, 1], n_times=3000, scenarios=["chain", "collider"], methods=["gc", "pdc"]
        )
        text = render_report(rows, n_times=3000, seeds=[0, 1], findings="FINDINGS-MARKER")
        for needle in (
            "# Method validation",
            "FINDINGS-MARKER",
            "### chain",
            "### collider",
            "| gc ",
            "| pdc ",
            "AUROC",
            "separation",
            "numpy",
            "dirconn",
            SCENARIOS["chain"].expectation[:40],
        ):
            assert needle in text, needle
        assert "noise_m15db" not in text  # only the scenarios that were run

    def test_the_environment_names_the_versions_that_matter(self):
        env = environment()
        assert {"python", "numpy", "scipy", "dirconn", "platform"} <= set(env)


@pytest.fixture(scope="module")
def known():
    """One 10000-sample simulation (seed 0) per scenario, scored for every method."""
    out = {}
    for name in ("chain", "common_driver", "feedback_loop", "instantaneous", "hidden_driver"):
        series = simulate_scenario(SCENARIOS[name], seed=0, n_times=10000)
        out[name] = {m: evaluate_series(SCENARIOS[name], m, series) for m in METHODS}
    series = simulate_scenario(SCENARIOS["noise_m15db"], seed=0, n_times=10000)
    out["noise_m15db"] = {m: evaluate_series(SCENARIOS["noise_m15db"], m, series) for m in METHODS}
    return out


class TestKnownBehaviour:
    # Thresholds are taken from docs/method_validation.md (10 seeds, 10000 samples): each one sits
    # between the observed values of the methods it separates, over all seeds, with margin. They
    # guard the findings of that report against regressions; they are not a benchmark.

    def test_the_chain_is_found_directly_by_conditioned_and_partial_measures(self, known):
        sep = {m: r["separation"] for m, r in known["chain"].items()}
        assert sep["gc"] > 100 and sep["gpdc"] > 100  # observed over seeds: > 340
        assert sep["pdc"] > 5  # observed: 9.4 to 84
        assert all(known["chain"][m]["direction_accuracy"] == 1.0 for m in VAR_METHODS)

    def test_pairwise_granger_and_dtf_report_the_indirect_path_of_the_chain(self, known):
        sep = {m: r["separation"] for m, r in known["chain"].items()}
        assert (
            1.0 < sep["gc_pairwise"] < 2.0
        )  # observed: 1.30 to 1.40, the indirect edge is 70-75 %
        assert (
            sep["dtf"] < 1.0
        )  # observed: 0.39 to 0.42, the indirect edge is stronger than a true one

    def test_pairwise_granger_reports_the_shared_driver_edge(self, known):
        sep = {m: r["separation"] for m, r in known["common_driver"].items()}
        assert sep["gc"] > 50  # observed: > 186
        assert 1.0 < sep["gc_pairwise"] < 3.0  # observed: 1.42 to 1.47

    def test_dtf_gets_the_direction_of_a_loop_edge_wrong(self, known):
        accuracy = {m: r["direction_accuracy"] for m, r in known["feedback_loop"].items()}
        assert accuracy["dtf"] < 1.0  # observed: 0.67 in 9 of 10 seeds
        assert all(accuracy[m] == 1.0 for m in ("gc", "gc_pairwise", "pdc", "gpdc"))

    def test_a_zero_lag_edge_between_memoryless_regions_is_not_found(self, known):
        sep = {m: known["instantaneous"][m]["separation"] for m in VAR_METHODS}
        assert all(value < 1.0 for value in sep.values())  # observed here: 0.02 to 0.67
        assert sep["dtf"] < 0.5 and sep["pdc"] < 0.5  # observed over seeds: below 0.11

    def test_a_hidden_driver_creates_an_edge_that_the_shares_rank_above_the_true_one(self, known):
        sep = {m: r["separation"] for m, r in known["hidden_driver"].items()}
        assert 1.2 < sep["gc"] < 3.0  # observed: 1.74 to 2.05, the spurious edge is about half
        assert sep["gpdc"] > 2.0  # observed: 3.8 to 4.7
        assert sep["pdc"] < 0.5 and sep["dtf"] < 0.6  # observed: <= 0.08 and <= 0.43

    def test_strong_observation_noise_breaks_every_method_and_no_diagnostic_notices(self, known):
        for method in VAR_METHODS:
            result = known["noise_m15db"][method]
            assert result["separation"] < 1.0, method  # observed here: 0.18 to 0.19
            assert result["order"] == 1, method
            assert result["diagnostic_warnings"] == [], method

    def test_rpdc_is_not_a_share_and_keeps_the_true_edge_of_a_hidden_driver_first(self, known):
        sep = {
            s: known[s]["rpdc"]["separation"] for s in ("chain", "common_driver", "hidden_driver")
        }
        assert sep["chain"] > 1000  # observed over seeds: 2325 to 12745
        assert sep["common_driver"] > 50  # observed: 169 to 438 (pdc: 0.65 to 3.6)
        assert sep["hidden_driver"] > 2.0  # observed: 5.1 to 6.4 (pdc: 0.07 to 0.08)

    def test_trgc_finds_direct_paths_but_not_past_a_hidden_driver(self, known):
        # observed over 10 seeds: chain 4.2 to 340; hidden driver 0.24 to 0.31 (gc 1.74 to 2.05)
        assert known["chain"]["trgc"]["separation"] > 2.0
        assert known["chain"]["trgc"]["direction_accuracy"] == 1.0
        assert known["hidden_driver"]["trgc"]["separation"] < 0.6
        assert known["hidden_driver"]["gc"]["separation"] > 1.2

    def test_imcoh_and_psi_report_the_indirect_path_of_the_chain(self, known):
        # observed: imcoh 0.70 to 0.81, psi 0.56 to 0.59 (they do not condition on B)
        assert known["chain"]["imcoh"]["separation"] < 1.0
        assert known["chain"]["psi"]["separation"] < 1.0
        assert known["chain"]["psi"]["direction_accuracy"] == 1.0

    def test_mtgc_reports_the_indirect_path_like_pairwise_granger(self, known):
        # observed over 10 seeds: chain 1.45 to 1.59 (gc_pairwise 1.30 to 1.40), always oriented
        assert 1.0 < known["chain"]["mtgc"]["separation"] < 2.5
        assert known["chain"]["mtgc"]["direction_accuracy"] == 1.0
        # -15 dB: mtgc 0.74 to 1.2, gc_pairwise 0.026 to 1.68 (seed 0: 0.18)
        assert (
            known["noise_m15db"]["mtgc"]["separation"]
            > known["noise_m15db"]["gc_pairwise"]["separation"]
        )

    def test_pcmci_removes_indirect_paths_but_misses_a_delay_beyond_tau_max(self, known):
        # observed over 10 seeds: chain 6.7 to 22.7; bidirectional 0.05 to 1.13 (tau_max = 2 < 3)
        assert known["chain"]["pcmci"]["separation"] > 3.0
        assert known["common_driver"]["pcmci"]["separation"] > 3.0

    def test_pcmci_finds_the_longer_delay_once_tau_max_covers_it(self, monkeypatch):
        # bidirectional: B -> A has a 3-sample delay, the BIC order is 2. Seed 1, measured:
        # separation 0.05 with the BIC order alone and 1.64 with the default 30 ms floor (8 samples)
        import dirconn.validation as validation
        from dirconn import PCMCI

        scenario = SCENARIOS["bidirectional"]
        series = simulate_scenario(scenario, seed=1, n_times=10000)
        default = evaluate_series(scenario, "pcmci", series)
        assert default["order"] == 8 and default["separation"] > 1.0
        monkeypatch.setitem(
            validation.METHODS, "pcmci", lambda: PCMCI(tau_max="bic", min_lag_seconds=0.0)
        )
        without_floor = evaluate_series(scenario, "pcmci", series)
        assert without_floor["order"] == 2 and without_floor["separation"] < 0.5

    def test_te_with_the_bic_history_orients_the_chain_and_inverts_under_strong_noise(self, known):
        # observed over 10 seeds: chain separation 1.55 to 1.88, direction 1; -15 dB AUROC 0.1-0.6
        assert known["chain"]["te"]["separation"] > 1.2
        assert known["chain"]["te"]["direction_accuracy"] == 1.0
        assert known["noise_m15db"]["te"]["auroc"] < 0.5  # seed 0: 0.2, an inverted ranking

    def test_psi_does_not_see_a_zero_lag_edge(self, known):
        assert known["instantaneous"]["psi"]["separation"] < 1.0  # observed: below 0.85

    def test_no_diagnostic_but_the_order_stability_check_warned_in_any_known_scenario(self, known):
        warned = {
            (name, method): [d for d in r["diagnostic_warnings"] if d != "order_stability"]
            for name, results in known.items()
            for method, r in results.items()
        }
        assert {key: names for key, names in warned.items() if names} == {}

    def test_the_order_stability_check_fires_rarely_on_the_known_scenarios(self, known):
        # 1 of 30 runs: zero-lag coupling with pairwise Granger causality. The direction of a
        # zero-lag edge is not identifiable, so the direction at the chosen order (2) and at a
        # longer one (32) need not agree; the check names the pair. If this changes, look at why.
        flagged = sorted(
            (name, method)
            for name, results in known.items()
            for method, r in results.items()
            if "order_stability" in r["diagnostic_warnings"]
        )
        assert flagged == [("instantaneous", "gc_pairwise")]

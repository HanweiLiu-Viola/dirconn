"""Seeded random sparse scenarios."""

import numpy as np
import pytest

from dirconn.scenario import (
    RandomScenario,
    ScenarioSpec,
    list_scenarios,
    load_scenario,
    random_scenario,
)
from dirconn.simulation import network_var, simulate

REGIONS = tuple(f"region{i:02d}-{'lh' if i % 2 else 'rh'}" for i in range(40))
SHIPPED = {"random_5": 5, "random_10": 10, "random_20": 20}


def make(n=10, seed=0, **kwargs) -> RandomScenario:
    return random_scenario(n, seed=seed, regions=REGIONS, **kwargs)


class TestGenerator:
    def test_the_same_seed_gives_the_same_scenario(self):
        assert make(seed=3).spec == make(seed=3).spec

    def test_another_seed_gives_another_graph_and_other_regions(self):
        a, b = make(seed=1).spec, make(seed=2).spec
        assert a.network.edges != b.network.edges
        assert a.network.names != b.network.names

    @pytest.mark.parametrize("n", [5, 10, 20])
    def test_size_and_edge_count(self, n):
        spec = make(n).spec
        assert spec.network.n_nodes == n
        assert len(spec.network.edges) == round(1.5 * n)
        assert len(set(spec.network.names)) == n
        assert set(spec.network.names) <= set(REGIONS)

    def test_edges_are_in_range_and_never_pair_up_in_both_directions(self):
        spec = make(20, seed=4).spec
        pairs = set()
        for edge in spec.network.edges:
            assert 0.2 <= edge.strength <= 0.4
            assert edge.delay_ms in (10.0, 20.0, 30.0)
            assert edge.band is None
            assert edge.source != edge.target
            pair = frozenset((edge.source, edge.target))
            assert pair not in pairs
            pairs.add(pair)

    def test_the_graph_has_the_requested_number_of_feedback_edges(self):
        for back in (0, 2, 3):
            result = make(10, seed=5, n_back_edges=back)
            assert result.n_back_edges == back
            # an acyclic graph has a topological order; with `back` edges removed it must exist
            order = {name: i for i, name in enumerate(result.order)}
            wrong = [e for e in result.spec.network.edges if order[e.source] > order[e.target]]
            assert len(wrong) == back

    def test_feedback_edges_always_close_a_cycle_and_the_redraws_are_counted(self):
        results = [make(10, seed=k) for k in range(10)]
        for result in results:
            adjacency = result.spec.network.adjacency().astype(int)
            reach = np.linalg.matrix_power(adjacency + np.eye(10, dtype=int), 10) > 0
            # a cycle: some node reaches another node that reaches it back
            assert (reach & reach.T & ~np.eye(10, dtype=bool)).any()
        assert sum(r.n_acyclic for r in results) > 0

    def test_zero_back_edges_gives_a_dag(self):
        spec = make(10, seed=6, n_back_edges=0).spec
        adjacency = spec.network.adjacency().astype(int)
        power = np.linalg.matrix_power(adjacency, spec.network.n_nodes)
        assert not power.any()  # nilpotent: no cycle

    @pytest.mark.parametrize("n", [5, 10, 20])
    def test_the_spectral_radius_is_below_the_limit(self, n):
        spec = make(n, seed=7).spec
        radius = network_var(spec.network, spec.sfreq, spec.dynamics).spectral_radius
        assert radius < 0.95

    def test_a_tight_limit_rejects_graphs_and_counts_them(self):
        # a node on its own has a radius of 0.8819 (bandwidth 4 Hz at 100 Hz); a graph only
        # raises it through a cycle, and a feedback edge does not always close one
        seeds = range(10)
        loose = [make(10, seed=k, max_radius=0.99) for k in seeds]
        tight = [make(10, seed=k, max_radius=0.8825, max_tries=5000) for k in seeds]
        assert sum(r.n_rejected for r in tight) > sum(r.n_rejected for r in loose)
        for result in tight:
            spec = result.spec
            radius = network_var(spec.network, spec.sfreq, spec.dynamics).spectral_radius
            assert radius < 0.8825

    def test_an_impossible_limit_fails_after_max_tries(self):
        with pytest.raises(ValueError, match="max_tries"):
            make(10, seed=9, max_radius=0.1, max_tries=20)

    def test_the_dynamics_are_oscillators_below_nyquist(self):
        spec = make(10, seed=10).spec
        for dyn in spec.dynamics.values():
            assert dyn.kind == "oscillator"
            assert 8.0 <= dyn.freq_hz <= 30.0

    def test_the_description_records_the_seed_and_the_rejections(self):
        result = make(10, seed=11)
        assert "seed 11" in result.spec.description
        assert f"{result.n_rejected} rejected" in result.spec.description

    def test_the_name_says_the_size(self):
        assert make(10).spec.name == "random_10"
        assert random_scenario(5, seed=0, regions=REGIONS, name="mine").spec.name == "mine"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_nodes": 1}, "n_nodes"),
            ({"n_nodes": 41}, "regions"),
            ({"n_nodes": 5, "n_back_edges": -1}, "n_back_edges"),
            ({"n_nodes": 5, "n_back_edges": 9}, "n_back_edges"),
            ({"n_nodes": 5, "edges_per_node": 0.0}, "edges_per_node"),
            ({"n_nodes": 5, "edges_per_node": 3.0}, "edges_per_node"),
            ({"n_nodes": 5, "strength": (0.4, 0.2)}, "strength"),
            ({"n_nodes": 5, "delay_samples": (0, 3)}, "delay_samples"),
            ({"n_nodes": 5, "delay_samples": (3, 1)}, "delay_samples"),
            ({"n_nodes": 5, "max_radius": 1.0}, "max_radius"),
        ],
    )
    def test_bad_arguments_name_the_argument(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            random_scenario(seed=0, regions=REGIONS, **kwargs)

    def test_duplicate_regions_are_refused(self):
        with pytest.raises(ValueError, match="regions"):
            random_scenario(3, seed=0, regions=("a-lh", "a-lh", "b-lh", "c-lh"))

    def test_the_result_survives_a_yaml_round_trip(self, tmp_path):
        spec = make(10, seed=12).spec
        path = tmp_path / "random.yaml"
        spec.to_yaml(path)
        assert load_scenario(path) == spec
        assert ScenarioSpec.from_dict(spec.to_dict()) == spec


class TestShippedFiles:
    def test_the_three_sizes_are_shipped(self):
        assert set(SHIPPED) <= set(list_scenarios())

    @pytest.mark.parametrize(("name", "n"), sorted(SHIPPED.items()))
    def test_size_stability_and_description(self, name, n):
        spec = load_scenario(name)
        assert spec.network.n_nodes == n
        assert len(spec.network.edges) == round(1.5 * n)
        radius = network_var(spec.network, spec.sfreq, spec.dynamics).spectral_radius
        assert radius < 0.95
        assert "seed" in spec.description and "rejected" in spec.description
        assert "has a cycle" in spec.description
        adjacency = spec.network.adjacency().astype(int)
        reach = np.linalg.matrix_power(adjacency + np.eye(n, dtype=int), n) > 0
        assert (reach & reach.T & ~np.eye(n, dtype=bool)).any()

    @pytest.mark.parametrize("name", sorted(SHIPPED))
    def test_a_short_simulation_runs_and_equalises_the_node_power(self, name):
        spec = load_scenario(name)
        sim = simulate(
            spec.network,
            sfreq=spec.sfreq,
            n_times=3000,
            seed=0,
            dynamics=spec.dynamics,
            n_background=spec.n_background,
            background_fraction=spec.background_fraction,
            node_scale="unit",
        )
        power = np.var(np.asarray(sim.sources.data), axis=(0, 2))
        assert power == pytest.approx(np.ones(spec.network.n_nodes), rel=1e-6)
        truth = np.asarray(sim.truth.values)[:, :, 0, 0]
        np.testing.assert_array_equal(np.nan_to_num(truth) > 0, spec.network.adjacency())


@pytest.mark.sample_data
class TestShippedFilesOnTheSampleHead:
    @pytest.fixture(scope="module")
    def candidates(self):
        from dirconn.headmodel import load_head_model

        heads = [load_head_model(m) for m in ("eeg", "mag", "grad")]
        visibility = np.min([h.visibility for h in heads], axis=0)
        return tuple(n for n, v in zip(heads[0].roi_names, visibility, strict=True) if v >= 0.5)

    def test_the_candidate_set_is_large_enough_for_20_nodes(self, candidates):
        assert len(candidates) >= 20

    @pytest.mark.parametrize("name", sorted(SHIPPED))
    def test_the_file_is_what_the_generator_gives_for_its_seed(self, name, candidates):
        from dirconn.scenario import RANDOM_SEEDS

        regenerated = random_scenario(SHIPPED[name], seed=RANDOM_SEEDS[name], regions=candidates)
        assert regenerated.spec == load_scenario(name)

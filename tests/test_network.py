import dataclasses
import warnings

import numpy as np
import pytest

from dirconn import ChannelMeta, Connectivity, DirconnWarning
from dirconn.network import Edge, Network, direct_node, dk_node


def make_network(**overrides):
    nodes = [
        dk_node("inferiortemporal-lh"),
        dk_node("superiorparietal-rh"),
        direct_node("STN-L", region="STN", hemisphere="lh"),
    ]
    edges = [
        Edge("STN-L", "inferiortemporal-lh", strength=0.3, delay_ms=10.0),
        Edge(
            "inferiortemporal-lh",
            "superiorparietal-rh",
            strength=-0.2,
            delay_ms=20.0,
            band=(13, 30),
        ),
    ]
    kwargs = dict(nodes=nodes, edges=edges, name="demo")
    kwargs.update(overrides)
    return Network(**kwargs)


class TestNodeHelpers:
    def test_dk_node_derives_region_hemisphere_and_atlas_from_the_label(self):
        node = dk_node("inferiortemporal-lh")
        assert node == ChannelMeta(
            "inferiortemporal-lh",
            type="misc",
            node_kind="cortical",
            region="inferiortemporal",
            hemisphere="lh",
            atlas="DK",
        )

    def test_dk_node_accepts_overrides(self):
        assert dk_node("bankssts-rh", atlas="custom").atlas == "custom"

    @pytest.mark.parametrize("name", ["bankssts", "bankssts-left", "-lh"])
    def test_dk_node_rejects_labels_without_a_hemisphere_suffix(self, name):
        with pytest.raises(ValueError, match="hemisphere"):
            dk_node(name)

    def test_direct_node(self):
        node = direct_node("STN-L", region="STN", hemisphere="lh")
        assert node.node_kind == "direct"
        assert node.region == "STN"
        assert node.hemisphere == "lh"
        assert node.atlas is None
        assert node.type == "misc"


class TestEdge:
    def test_defaults_are_instantaneous_and_broadband(self):
        edge = Edge("a", "b", strength=1.0)
        assert edge.delay_ms == 0.0
        assert edge.band is None

    def test_band_is_stored_as_a_float_tuple(self):
        assert Edge("a", "b", strength=1.0, band=[8, 13]).band == (8.0, 13.0)

    def test_negative_strength_is_allowed(self):
        assert Edge("a", "b", strength=-0.5).strength == -0.5

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            Edge("a", "b", strength=1.0).strength = 2.0  # type: ignore[misc]

    def test_self_loops_are_rejected(self):
        with pytest.raises(ValueError, match="self"):
            Edge("a", "a", strength=1.0)

    @pytest.mark.parametrize("strength", [0.0, float("nan"), float("inf")])
    def test_strength_must_be_finite_and_non_zero(self, strength):
        with pytest.raises(ValueError, match="strength"):
            Edge("a", "b", strength=strength)

    @pytest.mark.parametrize("delay", [-1.0, float("nan"), float("inf")])
    def test_delay_must_be_finite_and_non_negative(self, delay):
        with pytest.raises(ValueError, match="delay_ms"):
            Edge("a", "b", strength=1.0, delay_ms=delay)

    @pytest.mark.parametrize("band", [(13, 8), (8, 8), (-1, 5), (1, float("inf")), (1, 2, 3), (5,)])
    def test_invalid_band_is_rejected(self, band):
        with pytest.raises(ValueError, match="band"):
            Edge("a", "b", strength=1.0, band=band)

    def test_delay_in_samples_when_aligned(self):
        assert Edge("a", "b", strength=1.0, delay_ms=10.0).delay_samples(1000.0) == 10
        assert Edge("a", "b", strength=1.0, delay_ms=0.0).delay_samples(250.0) == 0

    def test_misaligned_delay_is_an_error_by_default(self):
        edge = Edge("a", "b", strength=1.0, delay_ms=10.0)
        with pytest.raises(ValueError, match="not a whole number of samples"):
            edge.delay_samples(250.0)

    def test_sub_sample_delay_is_misaligned(self):
        with pytest.raises(ValueError, match="not a whole number of samples"):
            Edge("a", "b", strength=1.0, delay_ms=1.0).delay_samples(100.0)

    def test_rounding_can_be_requested_and_warns(self):
        edge = Edge("a", "b", strength=1.0, delay_ms=10.0)
        with pytest.warns(DirconnWarning, match="rounded"):
            assert edge.delay_samples(250.0, rounding="round") == 3

    def test_aligned_delay_does_not_warn_when_rounding(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert Edge("a", "b", strength=1.0, delay_ms=8.0).delay_samples(250.0, "round") == 2

    @pytest.mark.parametrize("sfreq", [0.0, -1.0, float("nan")])
    def test_sampling_frequency_must_be_positive(self, sfreq):
        with pytest.raises(ValueError, match="sfreq"):
            Edge("a", "b", strength=1.0).delay_samples(sfreq)

    def test_unknown_rounding_mode_is_rejected(self):
        with pytest.raises(ValueError, match="rounding"):
            Edge("a", "b", strength=1.0).delay_samples(100.0, rounding="floor")


class TestNetworkValidation:
    def test_valid_network(self):
        net = make_network()
        assert net.n_nodes == 3
        assert net.names == ("inferiortemporal-lh", "superiorparietal-rh", "STN-L")
        assert net.name == "demo"

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            make_network().name = "x"  # type: ignore[misc]

    def test_at_least_two_nodes(self):
        with pytest.raises(ValueError, match="two nodes"):
            Network(nodes=[dk_node("bankssts-lh")], edges=[])

    def test_node_names_must_be_unique(self):
        with pytest.raises(ValueError, match="unique"):
            Network(nodes=[dk_node("bankssts-lh"), dk_node("bankssts-lh")], edges=[])

    def test_sensor_nodes_are_rejected(self):
        with pytest.raises(ValueError, match="node_kind"):
            Network(nodes=[ChannelMeta("Cz"), dk_node("bankssts-lh")], edges=[])

    def test_edges_must_reference_known_nodes(self):
        with pytest.raises(ValueError, match="unknown node"):
            make_network(edges=[Edge("STN-L", "nowhere", strength=1.0)])
        with pytest.raises(ValueError, match="unknown node"):
            make_network(edges=[Edge("nowhere", "STN-L", strength=1.0)])

    def test_exact_duplicate_edges_are_rejected(self):
        edge = Edge("STN-L", "bankssts-lh", strength=1.0)
        nodes = [direct_node("STN-L"), dk_node("bankssts-lh")]
        with pytest.raises(ValueError, match="duplicate"):
            Network(nodes=nodes, edges=[edge, Edge("STN-L", "bankssts-lh", strength=2.0)])

    def test_same_pair_with_different_bands_is_allowed(self):
        nodes = [direct_node("STN-L"), dk_node("bankssts-lh")]
        net = Network(
            nodes=nodes,
            edges=[
                Edge("STN-L", "bankssts-lh", strength=1.0, band=(4, 8)),
                Edge("STN-L", "bankssts-lh", strength=1.0, band=(13, 30)),
            ],
        )
        assert len(net.edges) == 2

    def test_a_network_without_edges_is_valid(self):
        net = Network(nodes=[dk_node("bankssts-lh"), dk_node("bankssts-rh")], edges=[])
        assert not net.adjacency().any()

    def test_adjacency_is_indexed_source_by_target(self):
        adjacency = make_network().adjacency()
        assert adjacency.dtype == bool
        assert adjacency[2, 0]  # STN-L -> inferiortemporal-lh
        assert not adjacency[0, 2]
        assert adjacency[0, 1]
        assert adjacency.sum() == 2


class TestTruth:
    def test_broadband_truth_has_fixed_dimensions_and_marks_edges(self):
        truth = make_network().truth()
        assert isinstance(truth, Connectivity)
        assert truth.values.shape == (3, 3, 1, 1)
        assert truth.source == truth.target == make_network().names
        assert not truth.is_frequency_resolved
        assert not truth.is_time_resolved
        matrix = truth.values[:, :, 0, 0]
        assert matrix[2, 0] == 1.0
        assert matrix[0, 2] == 0.0
        assert matrix[0, 1] == 1.0
        assert matrix[1, 0] == 0.0

    def test_diagonal_is_undefined(self):
        matrix = make_network().truth().values[:, :, 0, 0]
        assert np.isnan(np.diag(matrix)).all()
        assert np.isfinite(matrix[~np.eye(3, dtype=bool)]).all()

    def test_banded_edges_are_active_only_inside_their_band(self):
        freqs = [5.0, 10.0, 13.0, 20.0, 30.0, 35.0]
        truth = make_network().truth(freqs=freqs)
        assert truth.values.shape == (3, 3, 6, 1)
        np.testing.assert_array_equal(truth.freqs, freqs)
        banded = truth.values[0, 1, :, 0]
        np.testing.assert_array_equal(banded, [0, 0, 1, 1, 1, 0])  # 13 and 30 are inclusive
        broadband = truth.values[2, 0, :, 0]
        np.testing.assert_array_equal(broadband, np.ones(6))

    def test_several_bands_on_one_pair_are_combined(self):
        nodes = [direct_node("STN-L"), dk_node("bankssts-lh")]
        net = Network(
            nodes=nodes,
            edges=[
                Edge("STN-L", "bankssts-lh", strength=1.0, band=(4, 8)),
                Edge("STN-L", "bankssts-lh", strength=1.0, band=(13, 30)),
            ],
        )
        truth = net.truth(freqs=[6.0, 10.0, 20.0])
        np.testing.assert_array_equal(truth.values[0, 1, :, 0], [1, 0, 1])

    def test_static_truth_is_repeated_over_the_given_times(self):
        truth = make_network().truth(times=[0.0, 0.5, 1.0])
        assert truth.values.shape == (3, 3, 1, 3)
        assert truth.is_time_resolved
        np.testing.assert_array_equal(truth.values[2, 0, 0, :], np.ones(3))

    def test_frequencies_and_times_can_be_combined(self):
        truth = make_network().truth(freqs=[10.0, 20.0], times=[0.0, 1.0])
        assert truth.values.shape == (3, 3, 2, 2)

    def test_invalid_frequency_axis_is_rejected(self):
        with pytest.raises(ValueError, match="increasing"):
            make_network().truth(freqs=[10.0, 5.0])

    def test_provenance_describes_the_network(self):
        truth = make_network().truth()
        assert truth.provenance.method == "truth"
        assert truth.provenance.backend == "Network.truth"
        assert truth.provenance.parameters["name"] == "demo"
        assert len(truth.provenance.parameters["edges"]) == 2


class TestSerialisation:
    def test_dict_round_trip(self):
        net = make_network()
        assert Network.from_dict(net.to_dict()) == net

    def test_yaml_round_trip(self, tmp_path):
        net = make_network()
        path = tmp_path / "net.yaml"
        net.to_yaml(path)
        assert Network.from_yaml(path) == net

    def test_minimal_yaml_infers_dk_metadata(self, tmp_path):
        path = tmp_path / "net.yaml"
        path.write_text(
            "name: chain\n"
            "nodes:\n"
            "  - name: inferiortemporal-lh\n"
            "  - name: STN-L\n"
            "    kind: direct\n"
            "    region: STN\n"
            "    hemisphere: lh\n"
            "edges:\n"
            "  - source: STN-L\n"
            "    target: inferiortemporal-lh\n"
            "    strength: 0.3\n"
            "    delay_ms: 10\n"
            "    band: [13, 30]\n",
            encoding="utf-8",
        )
        net = Network.from_yaml(path)
        node = net.nodes[0]
        assert (node.node_kind, node.region, node.hemisphere, node.atlas) == (
            "cortical",
            "inferiortemporal",
            "lh",
            "DK",
        )
        assert net.nodes[1].node_kind == "direct"
        assert net.edges[0].band == (13.0, 30.0)
        assert net.name == "chain"

    def test_explicit_null_disables_inference(self):
        net = Network.from_dict(
            {
                "nodes": [
                    {"name": "bankssts-lh", "region": None, "hemisphere": None, "atlas": None},
                    {"name": "bankssts-rh"},
                ],
                "edges": [],
            }
        )
        node = net.nodes[0]
        assert (node.region, node.hemisphere, node.atlas) == (None, None, None)

    def test_a_cortical_node_without_a_dk_label_needs_no_metadata(self):
        net = Network.from_dict(
            {
                "nodes": [{"name": "V1"}, {"name": "V2"}],
                "edges": [{"source": "V1", "target": "V2", "strength": 1}],
            }
        )
        assert net.nodes[0].region is None
        assert net.nodes[0].node_kind == "cortical"

    @pytest.mark.parametrize(
        "mutate, text",
        [
            (lambda d: d.update(extra=1), "extra"),
            (lambda d: d["nodes"][0].update(colour="red"), "colour"),
            (lambda d: d["edges"][0].update(delay=5), "delay"),
        ],
    )
    def test_unknown_keys_are_rejected(self, mutate, text):
        data = {
            "nodes": [{"name": "bankssts-lh"}, {"name": "bankssts-rh"}],
            "edges": [{"source": "bankssts-lh", "target": "bankssts-rh", "strength": 1.0}],
        }
        mutate(data)
        with pytest.raises(ValueError, match=text):
            Network.from_dict(data)

    def test_missing_required_sections_are_reported(self):
        with pytest.raises(ValueError, match="nodes"):
            Network.from_dict({"edges": []})
        with pytest.raises(ValueError, match="source"):
            Network.from_dict(
                {
                    "nodes": [{"name": "bankssts-lh"}, {"name": "bankssts-rh"}],
                    "edges": [{"target": "x", "strength": 1}],
                }
            )
        with pytest.raises(ValueError, match="name"):
            Network.from_dict({"nodes": [{"kind": "cortical"}, {"name": "b"}], "edges": []})

    def test_unknown_node_kind_is_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            Network.from_dict(
                {"nodes": [{"name": "a", "kind": "sensor"}, {"name": "b"}], "edges": []}
            )

    def test_edges_section_is_optional(self):
        net = Network.from_dict({"nodes": [{"name": "bankssts-lh"}, {"name": "bankssts-rh"}]})
        assert net.edges == ()

    def test_yaml_with_a_non_mapping_top_level_is_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            Network.from_yaml(path)

    def test_missing_yaml_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Network.from_yaml(tmp_path / "nope.yaml")

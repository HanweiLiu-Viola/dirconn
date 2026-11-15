import dataclasses

import pytest

from dirconn import ChannelMeta


def test_defaults_describe_a_plain_sensor_channel():
    ch = ChannelMeta("Cz")
    assert ch.name == "Cz"
    assert ch.type == "misc"
    assert ch.node_kind == "sensor"
    assert ch.region is None
    assert ch.hemisphere is None
    assert ch.atlas is None
    assert ch.pos is None


def test_type_is_normalised_to_lower_case():
    assert ChannelMeta("Cz", type="EEG").type == "eeg"


def test_roi_and_direct_nodes_carry_region_metadata():
    roi = ChannelMeta(
        "inferiortemporal-lh",
        node_kind="cortical",
        region="inferiortemporal",
        hemisphere="lh",
        atlas="DK",
    )
    stn = ChannelMeta("STN-L", node_kind="direct", region="STN", hemisphere="lh")
    assert roi.node_kind == "cortical"
    assert stn.node_kind == "direct"


def test_position_is_stored_as_a_float_tuple():
    ch = ChannelMeta("Cz", pos=[0, 0, 0.1])
    assert ch.pos == (0.0, 0.0, 0.1)


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_name_is_rejected(name):
    with pytest.raises(ValueError, match="name"):
        ChannelMeta(name)


def test_unknown_hemisphere_is_rejected():
    with pytest.raises(ValueError, match="hemisphere"):
        ChannelMeta("x", hemisphere="left")


def test_unknown_node_kind_is_rejected():
    with pytest.raises(ValueError, match="node_kind"):
        ChannelMeta("x", node_kind="source")


@pytest.mark.parametrize("pos", [(0.0, 1.0), (0.0, 1.0, 2.0, 3.0), (0.0, float("nan"), 1.0)])
def test_invalid_position_is_rejected(pos):
    with pytest.raises(ValueError, match="pos"):
        ChannelMeta("x", pos=pos)


def test_channel_meta_is_immutable():
    ch = ChannelMeta("Cz")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ch.name = "Pz"  # type: ignore[misc]

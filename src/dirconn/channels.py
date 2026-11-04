"""Channel metadata shared by sensors, cortical regions and invasive channels."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

HEMISPHERES: tuple[str, ...] = ("lh", "rh", "mid")
NODE_KINDS: tuple[str, ...] = ("sensor", "cortical", "direct")


@dataclass(frozen=True)
class ChannelMeta:
    """Description of one channel (or node) of a time series.

    Parameters
    ----------
    name : str
        Unique channel name, for example ``"Cz"``, ``"inferiortemporal-lh"`` or ``"STN-L"``.
    type : str
        Channel type, stored in lower case (``"eeg"``, ``"mag"``, ``"grad"``, ``"misc"``, ...).
    node_kind : {"sensor", "cortical", "direct"}
        ``"sensor"`` for scalp or MEG sensors, ``"cortical"`` for a cortical region of
        interest, ``"direct"`` for an invasive channel used as-is (for example an STN LFP).
    region : str, optional
        Region label, for example ``"inferiortemporal"`` or ``"STN"``.
    hemisphere : {"lh", "rh", "mid"}, optional
        Hemisphere of the region.
    atlas : str, optional
        Atlas the region label comes from, for example ``"DK"``.
    pos : tuple of float, optional
        Position as three finite coordinates in metres.
    """

    name: str
    type: str = "misc"
    node_kind: str = "sensor"
    region: str | None = None
    hemisphere: str | None = None
    atlas: str | None = None
    pos: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        """Validate and normalise the fields."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Channel name must be a non-empty string.")
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError(f"Channel type of {self.name!r} must be a non-empty string.")
        object.__setattr__(self, "type", self.type.strip().lower())
        if self.node_kind not in NODE_KINDS:
            raise ValueError(f"node_kind must be one of {NODE_KINDS}, got {self.node_kind!r}.")
        if self.hemisphere is not None and self.hemisphere not in HEMISPHERES:
            raise ValueError(f"hemisphere must be one of {HEMISPHERES}, got {self.hemisphere!r}.")
        if self.pos is not None:
            object.__setattr__(self, "pos", _as_position(self.pos))


def _as_position(pos: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(v) for v in pos)
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"pos must be three finite coordinates, got {tuple(pos)!r}.")
    return values

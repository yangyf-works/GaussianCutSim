"""GaussianCutSim cutting / block_delta implementation."""
from __future__ import annotations

from dataclasses import dataclass as _rt_dataclass
import numpy as np
from cutsim.core.commands import CutCommand
from cutsim.core.splat_cloud import SplatCloud
from cutsim.cutting.playback import BlockPlayback


@_rt_dataclass
class RealtimeBlockDelta:
    block_index: int
    command: CutCommand
    sweep: object
    base_h: float
    before: SplatCloud
    removed_mask: np.ndarray
    removed: SplatCloud
    # Reserved for future GEOMETRIC partial-splat splitting during cutting.
    # Do not confuse this with the render-only 2-edge Half2DGS children.
    split_children: SplatCloud
    split_parent_ids: np.ndarray
    split_birth_u: np.ndarray
    cut_u: np.ndarray
    added_direct: SplatCloud
    added_grid: SplatCloud
    vertex_refined: SplatCloud  # debug-only 3-face candidates forced to h/4 Local Grid
    added: SplatCloud
    birth_u: np.ndarray
    candidate_cloud: SplatCloud
    direct_mask: np.ndarray
    fallback_mask: np.ndarray
    discard_mask: np.ndarray
    fallback_items: list
    stats: dict

    def playback(self):
        if not hasattr(self, '_playback'):
            self._playback = BlockPlayback(self)
        return self._playback

    def visible_cloud(self, u: float) -> SplatCloud:
        return self.playback().visible_cloud(u)

"""GaussianCutSim cutting / footprint implementation."""
from __future__ import annotations

from cutsim.rendering.clipping import _gs_radius
from cutsim.rendering.settings import RenderSettings


_CUT_FOOTPRINT_SETTINGS = RenderSettings(sigma_scale=0.75, intersection_sigma=2.0)


def _cut_footprint_radius(support,seam_weight):
    return _gs_radius(support,seam_weight,_CUT_FOOTPRINT_SETTINGS)

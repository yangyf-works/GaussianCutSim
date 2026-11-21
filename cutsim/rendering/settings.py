"""GaussianCutSim rendering / settings implementation."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from cutsim.rendering.opaque_coverage import DEFAULT_DISPLAY_SCALE


@dataclass(frozen=True)
class RenderSettings:
    """2DGS display defaults for the current viewer."""
    sigma_scale: float = DEFAULT_DISPLAY_SCALE
    alpha: float = 0.99
    ambient: float = 0.50
    diffuse: float = 0.50
    specular: float = 0.50
    half_clip: bool = True
    native_two_clip: bool = True  # Preserve Gaussian; at most two native clipping planes
    clip_stock_bounds: bool = True  # Workpiece only; tool-local previews opt out.
    tool_id_color: bool = True
    resolution_color: bool = False
    resolution_base_h: float = 0.5
    resolution_seam_scale: float = 1.20
    # Numerical conditioning floor for Half2DGS candidate intersections.
    half_min_angle_deg: float = 5.0
    clip_valleys: bool = True  # Both ridge and valley boundaries are physical edges.
    valley_min_angle_deg: float = 5.0  # Same angular conditioning as ridges by default.
    # Native clipping: bounded density, wider boundary groups, diverse second group.
    intersection_sigma: float = 2.0
    edge_center_band: float = 0.01
    similar_edge_angle_deg: float = 15.0  # 0 disables boundary grouping.
    clip_position_floor: float = 0.10
    clip_diversity_floor: float = 0.15
    clip_boundary_samples: int = 16
    # Debug quality/cost control: larger floor stops drawing-only subdivision earlier.
    # Independent of geometric splitting, GS density and the 2-sigma search.
    draw_subdivision_floor_scale: float = 1.0
    # One render Half2DGS owns one clipping edge.
    half_double_edge_split: bool = True
    half_double_edge_offset_scale: float = 0.34
    half_double_edge_min_sigma_scale: float = 0.78
    half_double_edge_opacity_scale: float = 0.72
    half_double_edge_elliptical: bool = True
    half_double_edge_guard_sigma: float = 2.5
    half_double_edge_center_sigma_scale: float = 0.50
    half_double_edge_max_children: int = 8  # 親1枚あたりの上限

    def __post_init__(self):
        from math import isfinite
        value = float(self.draw_subdivision_floor_scale)
        if not isfinite(value) or not 1. <= value <= 16.:
            raise ValueError('Drawing subdivision floor scale must be between 1 and 16')

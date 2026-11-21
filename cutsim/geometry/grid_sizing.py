"""GaussianCutSim geometry / grid_sizing implementation."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
import numpy as np


@dataclass(frozen=True)
class GridSizingSettings:
    """Cut-dependent isotropic base-grid policy.

    The tool diameter is first divided into ``tool_width_cells`` nominal cells.
    For a translating cut, the actual pitch is then nudged so the commanded move
    length is an integer number of those SAME cubic cells.  This keeps U/V/W
    isotropic while making the grid follow the physical scale of each cut.
    """
    tool_width_cells: int = 18
    min_cell: float = 0.05
    max_cell: float = 1.00
    fixed_cell: float | None = None


def cut_dependent_grid_cell(sweep, settings: GridSizingSettings):
    if settings.fixed_cell is not None:
        h = float(settings.fixed_cell)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("fixed grid cell must be finite and > 0")
        return h, {
            'grid_sizing_mode': 'fixed override',
            'grid_nominal_cell': h,
            'tool_diameter': float(sweep.diameter),
            'move_length': float(sweep.move_length),
            'tool_width_cells_actual': float(sweep.diameter / h),
            'move_cells_actual': float(sweep.move_length / h) if sweep.move_length > 0 else 0.0,
            'move_cells_integer': int(round(sweep.move_length / h)) if sweep.move_length > 0 else 0,
        }

    cells = max(2, int(settings.tool_width_cells))
    lo = max(float(settings.min_cell), 1e-6)
    hi = max(lo, float(settings.max_cell))
    nominal = float(np.clip(float(sweep.diameter) / cells, lo, hi))
    length = float(sweep.move_length)
    h = nominal
    n_move = 0

    # Preserve the tool-width-derived resolution, but choose the nearest integer
    # move subdivision so the start->end distance itself is exactly N*h.
    if length > max(0.5 * nominal, 1e-12):
        n_move = max(1, int(round(length / nominal)))
        h = length / n_move
        if h < lo:
            n_move = max(1, int(math.floor(length / lo + 1e-12)))
            h = length / n_move if n_move else nominal
        if h > hi:
            n_move = max(1, int(math.ceil(length / hi - 1e-12)))
            h = length / n_move
        h = float(np.clip(h, lo, hi))

    return h, {
        'grid_sizing_mode': 'cut-scaled',
        'grid_nominal_cell': nominal,
        'tool_diameter': float(sweep.diameter),
        'move_length': length,
        'tool_width_cells_actual': float(sweep.diameter / h),
        'move_cells_actual': float(length / h) if length > 0 else 0.0,
        'move_cells_integer': int(n_move),
    }

"""GaussianCutSim core / commands implementation."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import List, Tuple
from numbers import Real
import numpy as np
from cutsim.core.sweep import BallSweep, FlatSweep
from cutsim.core.tool_sampling import ToolGSSampling
from cutsim.core.tool_pose import DEFAULT_TOOL_AXIS, normalized_axis, tool_basis


CUTTING_RANGE_MARGIN_MM = 1e-4


@dataclass(frozen=True)
class CuttingRange:
    """Nominal axial interval and a calculation-only margin, in tip-local mm."""
    start: float
    end: float
    margin_mm: float = CUTTING_RANGE_MARGIN_MM

    def __post_init__(self):
        if (not isinstance(self.start, Real) or not isinstance(self.end, Real)
                or isinstance(self.start, (bool, np.bool_)) or isinstance(self.end, (bool, np.bool_))
                or not np.isfinite(self.start) or not np.isfinite(self.end)
                or not 0. <= self.start < self.end):
            raise ValueError('cutting_range requires finite 0 <= start < end (mm from tool tip)')
        if (not isinstance(self.margin_mm, Real) or isinstance(self.margin_mm, (bool, np.bool_))
                or not np.isfinite(self.margin_mm) or self.margin_mm < 0.
                or not np.isfinite(float(self.end)+float(self.margin_mm))):
            raise ValueError('cutting_range.margin_mm must be finite and nonnegative, with a finite expanded end')

    def calculation_bounds(self):
        """Expand axial clipping only; the physical tip remains at zero."""
        return max(0., float(self.start)-float(self.margin_mm)), float(self.end)+float(self.margin_mm)

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        if (not isinstance(value, dict) or not {'start', 'end'} <= value.keys()
                or set(value)-{'start', 'end', 'margin_mm'}):
            raise ValueError('cutting_range must contain start and end, with optional margin_mm')
        return cls(**value)


@dataclass(frozen=True)
class CutCommand:
    a: Tuple[float, ...]
    b: Tuple[float, ...]
    radius: float
    depth: float | None = None
    tool_id: int = 1
    tool_type: str = "flat"
    cutting_range: CuttingRange | None = None
    gs_sampling: ToolGSSampling = ToolGSSampling()
    tool_axis: Tuple[float, ...] = DEFAULT_TOOL_AXIS

    def __post_init__(self):
        object.__setattr__(self, 'tool_axis', normalized_axis(self.tool_axis))
        object.__setattr__(self, 'gs_sampling', ToolGSSampling.parse(self.gs_sampling))
        if self.cutting_range is not None:
            object.__setattr__(self, 'cutting_range', CuttingRange.parse(self.cutting_range))
        a, b = np.asarray(self.a, float), np.asarray(self.b, float)
        if a.shape not in ((2,), (3,)) or b.shape != a.shape:
            raise ValueError('Cut endpoints must both be XY or both be XYZ')
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Cut endpoints must be finite')
        if not np.isfinite(self.radius) or self.radius <= 0.:
            raise ValueError('Tool radius must be finite and positive')
        if len(a) == 2 and (self.depth is None or not np.isfinite(self.depth)):
            raise ValueError('XY commands require a finite depth')
        if len(a) == 3 and self.depth is not None:
            raise ValueError('XYZ commands specify Z in each endpoint; omit depth')
        if self.tool_axis != DEFAULT_TOOL_AXIS and (len(a) != 3 or self.cutting_range is None):
            raise ValueError('Indexed cuts require XYZ endpoints and a finite cutting_range')


def command_endpoints(cmd):
    """World tool-tip endpoints from XYZ coordinates or XY coordinates plus depth."""
    a, b = np.asarray(cmd.a, np.float64), np.asarray(cmd.b, np.float64)
    if len(a) == 2:
        return np.r_[a, cmd.depth], np.r_[b, cmd.depth]
    return a.copy(), b.copy()


def command_sweep(cmd: CutCommand, stock_top: float):
    if cmd.tool_axis != DEFAULT_TOOL_AXIS:
        from cutsim.core.indexed_sweep import IndexedSweep
        local, basis, origin = command_in_tool_frame(cmd)
        return IndexedSweep(command_sweep(local, float('inf')), basis, origin)
    if cmd.cutting_range is not None:
        from cutsim.core.cutting_sweep import CuttingSweep
        a, b = command_endpoints(cmd)
        cutting = cmd.cutting_range
        low, high = cutting.calculation_bounds()
        kind = cmd.tool_type.lower()
        # Reuse the established XY sampler when the finite upper cap is above
        # all stock. Every representation uses the same expanded axial bounds.
        top = float(a[2]+high)
        if a[2] == b[2] and top > float(stock_top):
            if kind == 'flat':
                return FlatSweep(a[:2], b[:2], cmd.radius, float(a[2]+low), top, cmd.tool_id)
            if kind == 'ball' and low == 0. and high >= 2.*cmd.radius:
                return BallSweep(a[:2], b[:2], cmd.radius, float(a[2]), top, cmd.tool_id)
        return CuttingSweep(a, b, cmd.radius, max(a[2], b[2])+high,
                            cmd.tool_id, kind, low, high)
    if len(cmd.a) == 3:
        from cutsim.core.linear_sweep import LinearSweep3D
        a, b = command_endpoints(cmd)
        kind = str(cmd.tool_type).lower()
        if kind not in ('flat', 'ball'):
            raise ValueError(f"Unsupported tool_type: {cmd.tool_type!r}")
        if a[2] != b[2]:
            return LinearSweep3D(a, b, cmd.radius,
                max(float(stock_top), float(max(a[2], b[2])+2.*cmd.radius))+6.,
                cmd.tool_id, kind)
        cmd = replace(cmd, a=tuple(a[:2]), b=tuple(b[:2]), depth=float(a[2]))
        if a[2] >= float(stock_top)+6.:
            stock_top = float(a[2]+2.*cmd.radius)
    a = np.array(cmd.a, np.float64)
    b = np.array(cmd.b, np.float64)
    z_hi = float(stock_top) + 6.0
    kind = str(cmd.tool_type).lower()
    if kind == "flat":
        return FlatSweep(a, b, cmd.radius, cmd.depth, z_hi, cmd.tool_id)
    if kind == "ball":
        return BallSweep(a, b, cmd.radius, cmd.depth, z_hi, cmd.tool_id)
    raise ValueError(f"Unsupported tool_type: {cmd.tool_type!r}")


def command_in_tool_frame(command):
    """A fixed-axis cut in tool coordinates; the zero-tilt path stays identical."""
    basis = tool_basis(command.tool_axis)
    if command.tool_axis == DEFAULT_TOOL_AXIS:
        return command, basis, np.zeros(3)
    a, b = command_endpoints(command)
    delta = (b-a)@basis
    delta[np.abs(delta)<64.*np.finfo(float).eps*max(1.,np.linalg.norm(delta))] = 0.
    return replace(command, a=(0., 0., 0.), b=tuple(delta), depth=None,
                   tool_axis=DEFAULT_TOOL_AXIS), basis, a

"""Immutable per-tool GS sampling settings and tools-file defaults."""
from dataclasses import dataclass, replace
from numbers import Integral, Real
import math


@dataclass(frozen=True)
class ToolGSSampling:
    mode: str = 'auto'
    pitch_mm: float | None = None
    ball_angle_deg: float | None = None
    diameter_divisions: int | None = None

    def __post_init__(self):
        if self.mode not in ('auto', 'fixed'):
            raise ValueError('gs_sampling.mode must be auto or fixed')
        if self.mode == 'fixed':
            if (not isinstance(self.pitch_mm, Real) or isinstance(self.pitch_mm, bool)
                    or not math.isfinite(self.pitch_mm) or self.pitch_mm <= 0.):
                raise ValueError('Fixed gs_sampling requires a finite positive pitch_mm')
        elif self.pitch_mm is not None:
            raise ValueError('Auto gs_sampling must omit pitch_mm')
        if self.ball_angle_deg is not None and (
                not isinstance(self.ball_angle_deg, Real) or isinstance(self.ball_angle_deg, bool)
                or not math.isfinite(self.ball_angle_deg) or not 0. < self.ball_angle_deg <= 90.):
            raise ValueError('gs_sampling.ball_angle_deg must be null or a finite angle in (0, 90] degrees')
        if self.diameter_divisions is not None and (
                not isinstance(self.diameter_divisions, Integral) or isinstance(self.diameter_divisions, bool)
                or self.diameter_divisions < 2):
            raise ValueError('gs_sampling.diameter_divisions must be null or an integer >= 2')

    @classmethod
    def parse(cls, value=None, defaults=None):
        defaults = cls() if defaults is None else defaults
        if value is None:
            return defaults
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict) or set(value)-{'mode', 'pitch_mm', 'ball_angle_deg', 'diameter_divisions'}:
            raise ValueError('gs_sampling accepts only mode, pitch_mm, ball_angle_deg and diameter_divisions')
        mode = value.get('mode', 'fixed' if 'pitch_mm' in value else defaults.mode)
        pitch = value.get('pitch_mm', defaults.pitch_mm) if mode == 'fixed' else value.get('pitch_mm')
        return cls(mode, pitch, value.get('ball_angle_deg', defaults.ball_angle_deg),
                   value.get('diameter_divisions', defaults.diameter_divisions))

    def resolve_pitch(self, automatic_pitch, gs_density=1.):
        from cutsim.geometry.sampling_density import sampling_pitch
        return sampling_pitch(self.pitch_mm if self.mode == 'fixed' else automatic_pitch, gs_density)

    def resolve_sweep_pitch(self, sweep, grid_sizing, gs_density=1.):
        """Resolve GS pitch using a tool-specific auto divisor or the simulator default."""
        if self.mode == 'fixed':
            return self.resolve_pitch(None, gs_density)
        from cutsim.geometry.grid_sizing import cut_dependent_grid_cell
        if self.diameter_divisions is not None:
            grid_sizing = replace(grid_sizing, tool_width_cells=int(self.diameter_divisions))
        automatic_pitch, _ = cut_dependent_grid_cell(sweep, grid_sizing)
        return self.resolve_pitch(automatic_pitch, gs_density)

    def resolve_ball_angle(self, gs_density=1.):
        from cutsim.geometry.sampling_density import validate_gs_density
        density = validate_gs_density(gs_density)
        return (None if self.ball_angle_deg is None else
                min(90., float(self.ball_angle_deg)/math.sqrt(density)))

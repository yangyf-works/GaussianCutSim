"""One density control for surface sampling and bounded geometric approximation."""
from dataclasses import dataclass
from math import isfinite, sqrt

GS_DENSITY_PERCENTAGES = (10, 25, 50, 100, 200, 400, 800)

def validate_gs_density(value):
    value = float(value)
    if not isfinite(value) or not GS_DENSITY_PERCENTAGES[0]/100. <= value <= GS_DENSITY_PERCENTAGES[-1]/100.:
        raise ValueError('GS density multiplier must be between 0.10 and 8.0 (10-800%)')
    return value


def sampling_pitch(base_pitch, density=1.):
    """Area density: multiply both surface pitch and support by 1/sqrt(D).

    Exactly 1.0 preserves baseline arithmetic. See DensityPolicy for the
    approximation scales; numerical/contact tolerances and 2-sigma stay fixed.
    """
    density = validate_gs_density(density)
    return float(base_pitch) if density == 1. else float(base_pitch) / sqrt(density)


@dataclass(frozen=True)
class DensityPolicy:
    density: float = 1.

    def __post_init__(self):
        object.__setattr__(self, 'density', validate_gs_density(self.density))

    @property
    def spacing_scale(self): return 1./sqrt(self.density)

    @property
    def chord_scale(self):
        # Tangent/chord error grows quadratically with the sampling interval.
        return 1./self.density

    def minimum_support(self, baseline): return sampling_pitch(baseline, self.density)

    def boundary_pitch(self, baseline):
        # Intersections remain exact; only their surface sampling interval
        # changes. Keeping a baseline lattice here defeats area-density control.
        return sampling_pitch(baseline, self.density)

    def boundary_support(self, baseline):
        # Boundary anchors cover a line, not the surrounding surface.
        # Keep the narrow boundary band instead
        # of increasing circular area and triggering more boundary splits.
        # The regular surface lattice supplies the surrounding face coverage.
        return float(baseline) if self.density<1. else sampling_pitch(baseline,self.density)

    def draw_floor(self, regions, baseline=.005):
        target = sampling_pitch(baseline, self.density)
        # Enlarging a Top-2 leaf at a third boundary can create a fin. Keep the
        # baseline floor for these critical domains; never discard that plane.
        critical = any(len(region) > 2 for region in regions)
        return min(target, baseline) if critical else target

"""Surface-coordinate ray cells for finite ball/flat tools, in millimetres.

The grid is indexed by meridian arc length (or cap radius) and azimuth.
Ball rays originate at the ball centre; cylinder rays at the corresponding
point on the tool axis. No Cartesian voxel lattice is constructed.
"""
from dataclasses import dataclass
from numbers import Integral
import math
import numpy as np

from cutsim.core.commands import command_endpoints


def _frozen(value, dtype=float):
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class ToolProfileGrid:
    origins: np.ndarray
    directions: np.ndarray
    surface_distance: np.ndarray
    charts: np.ndarray
    cell_bounds: np.ndarray  # meridian/radius low, high, azimuth low, high
    ring_offsets: np.ndarray
    tip: np.ndarray
    basis: np.ndarray
    radius: float
    pitch: float
    kind: str
    axial_bounds: tuple
    ray_bounds: np.ndarray | None = None

    @property
    def surface_points(self):
        return self.origins + self.surface_distance[:, None] * self.directions

    def points(self, ray_ids, distance):
        return self.origins[ray_ids] + np.asarray(distance)[..., None] * self.directions[ray_ids]

    @classmethod
    def from_command(cls, command, pitch=None, *, max_rays=200_000,ray_margin=None,gs_density=1.):
        """Use the command's initial tip, fixed tool axis and nominal range.

        Cutting margins still belong to the analytic sweep, not to sampling.
        An indexed command translates at one constant orientation.
        """
        if command.cutting_range is None:
            raise ValueError('Tool-profile grids require an explicit finite cutting_range')
        tip, _ = command_endpoints(command)
        from cutsim.core.tool_pose import tool_basis
        axial = command.cutting_range
        if pitch is None:
            from cutsim.core.commands import command_sweep
            from cutsim.geometry.grid_sizing import GridSizingSettings
            sweep=command_sweep(command,0.)
            pitch=command.gs_sampling.resolve_sweep_pitch(sweep,GridSizingSettings(),gs_density)
            if ray_margin is None:ray_margin=pitch
        return cls.build(command.tool_type.lower(), command.radius, (axial.start, axial.end),
                         pitch, tip=tip, basis=tool_basis(command.tool_axis),
                         max_rays=max_rays,ray_margin=ray_margin,
                         ball_angle_deg=command.gs_sampling.resolve_ball_angle(gs_density))

    @classmethod
    def build(cls, kind, radius, axial_bounds, pitch, *, tip=(0., 0., 0.),
              basis=None, max_rays=200_000,ray_margin=None,ball_angle_deg=None,
              include_lower_cap=True, include_upper_cap=False):
        """Build immutable cells at a fixed pose; pitch is on the tool surface.

        Each meridian step and azimuthal arc at its sample are <= pitch.
        Ring populations decrease towards the pole, avoiding a dense pole.
        The lower cap samples the bottom, including truncated balls. The axial
        sweep can request its exposed upper cap or skip an internal lower cap.
        A positive ray_margin requests forward finite rays ending that many mm
        past the surface. None retains the unbounded reference-grid convention.
        """
        tip = np.asarray(tip, float)
        basis = np.eye(3) if basis is None else np.asarray(basis, float)
        axial = np.asarray(axial_bounds, float)
        if kind not in ('ball', 'flat'):
            raise ValueError('Tool-profile kind must be ball or flat')
        if not np.isfinite(radius) or radius <= 0 or not np.isfinite(pitch) or pitch <= 0:
            raise ValueError('Positive finite radius and pitch required')
        if ray_margin is not None and (not np.isfinite(ray_margin) or ray_margin<=0):
            raise ValueError('Positive finite profile ray margin required')
        if ball_angle_deg is not None and (not np.isfinite(ball_angle_deg) or not 0<ball_angle_deg<=90):
            raise ValueError('Ball angle must be finite and in (0,90] degrees')
        if axial.shape != (2,) or not np.isfinite(axial).all() or not 0 <= axial[0] < axial[1]:
            raise ValueError('Finite axial bounds must satisfy 0 <= low < high')
        if tip.shape != (3,) or not np.isfinite(tip).all():
            raise ValueError('Finite 3D tool tip required')
        if (basis.shape != (3, 3) or not np.isfinite(basis).all()
                or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-12, rtol=0.)
                or not np.isclose(np.linalg.det(basis), 1., atol=1e-12, rtol=0.)):
            raise ValueError('Right-handed orthonormal tool basis required')
        if not isinstance(max_rays, Integral) or isinstance(max_rays, bool) or max_rays < 1:
            raise ValueError('Positive integer max_rays required')
        origins, directions, distances, charts, cells = [], [], [], [], []
        offsets = [0]

        def rings(chart, lower, upper, evaluate, edges=None):
            if upper <= lower:
                return
            count = max(1, math.ceil((upper-lower)/pitch)) if edges is None else len(edges)-1
            if 4*count > max_rays-offsets[-1]:
                raise ValueError('Tool-profile ray budget exceeded')
            step = (upper-lower)/count
            for row in range(count):
                lo, hi = (lower+row*step, lower+(row+1)*step) if edges is None else edges[row:row+2]
                radial, height, normal_radial, normal_z, distance = evaluate((lo+hi)/2)
                population = max(4, math.ceil(2*math.pi*radial/pitch))
                if offsets[-1]+population > max_rays:
                    raise ValueError('Tool-profile ray budget exceeded')
                phi0 = np.arange(population)*2*math.pi/population
                phi1 = (np.arange(population)+1)*2*math.pi/population
                phi = (phi0+phi1)/2
                radial_direction = np.column_stack((np.cos(phi), np.sin(phi), np.zeros(population)))
                normal = normal_radial*radial_direction
                normal[:, 2] = normal_z
                surface = radial*radial_direction
                surface[:, 2] = height
                origins.append(surface-distance*normal)
                directions.append(normal)
                distances.append(np.full(population, distance))
                charts.extend([chart]*population)
                cells.append(np.column_stack((np.full(population, lo), np.full(population, hi), phi0, phi1)))
                offsets.append(offsets[-1]+population)

        low, high = axial
        if kind == 'ball' and low < radius:
            from cutsim.geometry.tool_section import ball_section_angles
            lower = radius*math.acos(np.clip(1-low/radius, 0., 1.))
            upper = radius*math.acos(np.clip(1-min(high, radius)/radius, 0., 1.))
            edges=(None if ball_angle_deg is None else
                   radius*ball_section_angles(radius,low,high,pitch,ball_angle_deg))
            rings('ball', lower, upper, lambda s: (
                radius*math.sin(s/radius), radius*(1-math.cos(s/radius)),
                math.sin(s/radius), -math.cos(s/radius), radius),edges)
        cylinder_low = max(low, radius) if kind == 'ball' else low
        rings('cylinder', cylinder_low, high, lambda z: (radius, z, 1., 0., radius))

        def cross_radius(z):
            return math.sqrt(max(0., z*(2*radius-z))) if kind == 'ball' and z < radius else radius

        for chart, z, sign, enabled in (('lower_cap', low, -1., include_lower_cap),
                                       ('upper_cap', high, 1., include_upper_cap)):
            if not enabled:
                continue
            def cap(r,z=z,sign=sign):
                distance=0.
                if ray_margin is not None:
                    # Start inside this axial column, including truncated balls.
                    bottom=max(low,radius-math.sqrt(max(0.,radius*radius-r*r))) if kind=='ball' else low
                    middle=.5*(bottom+high)
                    distance=abs(z-middle)
                return r,z,0.,sign,distance
            rings(chart, 0., cross_radius(z),cap)
        local_origins, local_directions = np.concatenate(origins), np.concatenate(directions)
        distance=np.concatenate(distances)
        domain=None if ray_margin is None else _frozen(np.column_stack((np.zeros(len(distance)),distance+ray_margin)))
        return cls(_frozen(tip+local_origins@basis.T), _frozen(local_directions@basis.T),
                   _frozen(distance), _frozen(charts, 'U12'),
                   _frozen(np.concatenate(cells)), _frozen(offsets, np.int64),
                   _frozen(tip), _frozen(basis), float(radius), float(pitch), kind, tuple(axial),domain)

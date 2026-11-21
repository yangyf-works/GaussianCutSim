"""GaussianCutSim geometry / tool_profile implementation."""
from __future__ import annotations

import math
import numpy as np
import time
from cutsim.core.commands import command_sweep
from cutsim.core.splat_cloud import SplatCloud
from cutsim.core.sweep import BallSweep, FlatSweep, capsule2d_parameter


DIRECT_SMOOTH_SCALE = 1.50


DIRECT_RIM_SCALE = 1.00


def _centered_samples(lo: float, hi: float, target_step: float) -> np.ndarray:
    """Cell-centred 1-D samples with approximately ``target_step`` spacing."""
    lo = float(lo); hi = float(hi); target_step = max(float(target_step), 1e-9)
    if hi <= lo + 1e-12:
        return np.array([(lo + hi) * 0.5], dtype=np.float64)
    n = max(1, int(math.ceil((hi - lo) / target_step)))
    step = (hi - lo) / n
    return lo + (np.arange(n, dtype=np.float64) + 0.5) * step


def _sweep_uv_frame(sweep):
    dxy = np.asarray(sweep.b_xy - sweep.a_xy, dtype=np.float64)
    length = float(np.linalg.norm(dxy))
    if length < 1e-12:
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        u = np.array([dxy[0] / length, dxy[1] / length, 0.0], dtype=np.float64)
    v = np.array([-u[1], u[0], 0.0], dtype=np.float64)
    w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return u, v, w, length


def _capsule_perimeter_xy(sweep, h: float):
    """Sample the 2-D capsule perimeter and return cutter-outward XY normals."""
    u, v, _w, length = _sweep_uv_frame(sweep)
    u2, v2 = u[:2], v[:2]
    a = np.asarray(sweep.a_xy, dtype=np.float64)
    radius = float(sweep.radius)
    pts = []
    outward = []

    if length < 1e-12:
        n_ang = max(12, int(math.ceil(2.0 * math.pi * radius / max(h, 1e-9))))
        ang = (np.arange(n_ang, dtype=np.float64) + 0.5) * (2.0 * math.pi / n_ang)
        dirs = np.cos(ang)[:, None] * u2 + np.sin(ang)[:, None] * v2
        pts.append(a + radius * dirs)
        outward.append(dirs)
    else:
        t = _centered_samples(0.0, length, h)
        line = a + t[:, None] * u2
        pts.extend([line + radius * v2, line - radius * v2])
        outward.extend([np.broadcast_to(v2, line.shape), np.broadcast_to(-v2, line.shape)])

        n_cap = max(6, int(math.ceil(math.pi * radius / max(h, 1e-9))))
        da = math.pi / n_cap
        # Start cap: outward directions have non-positive U component.
        ang0 = math.pi * 0.5 + (np.arange(n_cap, dtype=np.float64) + 0.5) * da
        d0 = np.cos(ang0)[:, None] * u2 + np.sin(ang0)[:, None] * v2
        pts.append(a + radius * d0)
        outward.append(d0)
        # End cap: outward directions have non-negative U component.
        b = np.asarray(sweep.b_xy, dtype=np.float64)
        ang1 = -math.pi * 0.5 + (np.arange(n_cap, dtype=np.float64) + 0.5) * da
        d1 = np.cos(ang1)[:, None] * u2 + np.sin(ang1)[:, None] * v2
        pts.append(b + radius * d1)
        outward.append(d1)

    return np.concatenate(pts, axis=0), np.concatenate(outward, axis=0)


def _append_probe_parts(parts, positions, normals, h: float, tool_id: int):
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if not len(positions):
        return
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
    parts.append(SplatCloud(
        positions,
        normals,
        np.full(len(positions), float(h), dtype=np.float64),
        np.full(len(positions), int(tool_id), dtype=np.int32),
        np.full(len(positions), -2, dtype=np.int16),
    ))


def _dedupe_probe_cloud(cloud: SplatCloud, h: float) -> SplatCloud:
    if not len(cloud.positions):
        return cloud
    quant = max(float(h) * 2e-4, 1e-8)
    key = np.column_stack([
        np.rint(cloud.positions / quant).astype(np.int64),
        np.rint(cloud.normals * 1000.0).astype(np.int64),
    ])
    _, first = np.unique(key, axis=0, return_index=True)
    first.sort()
    return SplatCloud(cloud.positions[first], cloud.normals[first], cloud.support[first],
                      cloud.tool_ids[first], cloud.refine_levels[first], cloud.seam_weight[first])


def analytic_sweep_surface_candidates(sweep, h: float,
                                      stock_lo: np.ndarray, stock_hi: np.ndarray) -> SplatCloud:
    """Generate candidate 2DGS centres directly on the newest analytic sweep.

    This is intentionally independent of the local SDF grid.  It samples only
    boundary families that can become a machined surface (bottom/side for a flat
    end sweep; lower rounded capsule + upper side for a ball-end sweep).  The
    old-material probe later decides whether each candidate is safely usable,
    near a Boolean intersection, or completely outside already-existing material.
    """
    if hasattr(sweep, 'a_xyz'):
        from cutsim.geometry.linear_surface import linear_surface_candidates
        return linear_surface_candidates(sweep, h, stock_lo, stock_hi)
    h = max(float(h), 1e-9)
    u, v, w, length = _sweep_uv_frame(sweep)
    u2, v2 = u[:2], v[:2]
    parts = []
    radius = float(sweep.radius)

    # Keep vertical sampling near the stock only.  The command's z_hi deliberately
    # extends above the stock and would otherwise create many guaranteed rejects.
    z_pad = 2.0 * h
    z_stock_lo = float(stock_lo[2]) - z_pad
    z_stock_hi = float(stock_hi[2]) + z_pad

    if isinstance(sweep, FlatSweep):
        # Bottom planar face: sample the full swept 2-D capsule area.
        qu = _centered_samples(-radius, length + radius, h)
        qv = _centered_samples(-radius, radius, h)
        U, V = np.meshgrid(qu, qv, indexing='ij')
        xy = (np.asarray(sweep.a_xy)[None, :] + U.reshape(-1, 1) * u2 +
              V.reshape(-1, 1) * v2)
        _t, nearest = capsule2d_parameter(xy, np.asarray(sweep.a_xy), np.asarray(sweep.b_xy))
        keep = np.linalg.norm(xy - nearest, axis=1) <= radius + 1e-10
        p = np.column_stack([xy[keep], np.full(np.count_nonzero(keep), float(sweep.z_lo))])
        n = np.broadcast_to(np.array([0.0, 0.0, 1.0]), p.shape).copy()
        _append_probe_parts(parts, p, n, h, sweep.tool_id)

        # Vertical capsule side.  Material-outward is opposite cutter-outward.
        z0 = max(float(sweep.z_lo), z_stock_lo)
        z1 = min(float(sweep.z_hi), z_stock_hi)
        if z1 > z0 + 1e-12:
            perimeter, out_xy = _capsule_perimeter_xy(sweep, h)
            zz = _centered_samples(z0, z1, h)
            p = np.repeat(perimeter, len(zz), axis=0)
            zcol = np.tile(zz, len(perimeter))
            p = np.column_stack([p, zcol])
            no = np.repeat(out_xy, len(zz), axis=0)
            n = np.column_stack([-no, np.zeros(len(no))])
            _append_probe_parts(parts, p, n, h, sweep.tool_id)

    elif isinstance(sweep, BallSweep):
        center_z = float(sweep.center_z)
        if length < 1e-12:
            # Stationary ball: lower hemisphere.
            n_lat = max(4, int(math.ceil(0.5 * math.pi * radius / h)))
            n_az = max(12, int(math.ceil(2.0 * math.pi * radius / h)))
            psi = (np.arange(n_lat, dtype=np.float64) + 0.5) * (0.5 * math.pi / n_lat)
            phi = (np.arange(n_az, dtype=np.float64) + 0.5) * (2.0 * math.pi / n_az)
            P, A = np.meshgrid(psi, phi, indexing='ij')
            d = (np.sin(P).reshape(-1, 1) * np.cos(A).reshape(-1, 1) * u +
                 np.sin(P).reshape(-1, 1) * np.sin(A).reshape(-1, 1) * v -
                 np.cos(P).reshape(-1, 1) * w)
            c = np.array([sweep.a_xy[0], sweep.a_xy[1], center_z], dtype=np.float64)
            p = c + radius * d
            _append_probe_parts(parts, p, -d, h, sweep.tool_id)
        else:
            # Interior strip of the swept lower capsule.
            t = _centered_samples(0.0, length, h)
            n_ang = max(6, int(math.ceil(math.pi * radius / h)))
            ang = math.pi + (np.arange(n_ang, dtype=np.float64) + 0.5) * (math.pi / n_ang)
            T, A = np.meshgrid(t, ang, indexing='ij')
            center = (np.array([sweep.a_xy[0], sweep.a_xy[1], center_z])[None, :] +
                      T.reshape(-1, 1) * u)
            d = (np.cos(A).reshape(-1, 1) * v + np.sin(A).reshape(-1, 1) * w)
            p = center + radius * d
            _append_probe_parts(parts, p, -d, h, sweep.tool_id)

            # Start/end spherical quarter-caps: endpoint half-space AND z<=center_z.
            n_alpha = max(3, int(math.ceil(0.5 * math.pi * radius / h)))
            n_beta = max(6, int(math.ceil(math.pi * radius / h)))
            beta = math.pi + (np.arange(n_beta, dtype=np.float64) + 0.5) * (math.pi / n_beta)
            for is_start in (True, False):
                if is_start:
                    alpha = (0.5 * math.pi +
                             (np.arange(n_alpha, dtype=np.float64) + 0.5) *
                             (0.5 * math.pi / n_alpha))
                    c = np.array([sweep.a_xy[0], sweep.a_xy[1], center_z], dtype=np.float64)
                else:
                    alpha = ((np.arange(n_alpha, dtype=np.float64) + 0.5) *
                             (0.5 * math.pi / n_alpha))
                    c = np.array([sweep.b_xy[0], sweep.b_xy[1], center_z], dtype=np.float64)
                AL, BE = np.meshgrid(alpha, beta, indexing='ij')
                d = (np.cos(AL).reshape(-1, 1) * u +
                     (np.sin(AL) * np.cos(BE)).reshape(-1, 1) * v +
                     (np.sin(AL) * np.sin(BE)).reshape(-1, 1) * w)
                p = c + radius * d
                _append_probe_parts(parts, p, -d, h, sweep.tool_id)

        # Vertical side above the ball equator, sampled on the 2-D capsule perimeter.
        z0 = max(center_z, z_stock_lo)
        z1 = min(float(sweep.z_hi), z_stock_hi)
        if z1 > z0 + 1e-12:
            perimeter, out_xy = _capsule_perimeter_xy(sweep, h)
            zz = _centered_samples(z0, z1, h)
            pxy = np.repeat(perimeter, len(zz), axis=0)
            p = np.column_stack([pxy, np.tile(zz, len(perimeter))])
            no = np.repeat(out_xy, len(zz), axis=0)
            n = np.column_stack([-no, np.zeros(len(no))])
            _append_probe_parts(parts, p, n, h, sweep.tool_id)
    else:
        raise TypeError(f"Unsupported sweep type: {type(sweep).__name__}")

    out = SplatCloud.empty()
    for part in parts:
        out.append(part)
    return _dedupe_probe_cloud(out, h)


def _cloud_from_parts(parts) -> SplatCloud:
    out = SplatCloud.empty()
    for part in parts:
        out.append(part)
    return out


def _append_surface_part(parts, positions, normals, support: float, tool_id: int,
                         refine_level: int = -2):
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if not len(positions):
        return
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
    parts.append(SplatCloud(
        positions, normals,
        np.full(len(positions), float(support), dtype=np.float64),
        np.full(len(positions), int(tool_id), dtype=np.int32),
        np.full(len(positions), int(refine_level), dtype=np.int16)))


def _dedupe_splat_cloud(cloud: SplatCloud, quant: float) -> SplatCloud:
    if not len(cloud.positions):
        return cloud
    q = max(float(quant), 1e-8)
    key = np.column_stack([
        np.rint(cloud.positions / q).astype(np.int64),
        np.rint(cloud.normals * 1000.0).astype(np.int64),
    ])
    _, first = np.unique(key, axis=0, return_index=True)
    first.sort()
    return SplatCloud(cloud.positions[first], cloud.normals[first], cloud.support[first],
                      cloud.tool_ids[first], cloud.refine_levels[first])


def _flat_formal_surface_candidates(sweep: FlatSweep, h: float,
                                    stock_lo: np.ndarray, stock_hi: np.ndarray) -> SplatCloud:
    """Tool-specific Flat End sampling.

    Flat faces may use larger Direct 2DGS in the smooth interior, but the sharp
    90-degree bottom/side corner needs a denser transition zone.  Therefore:
      * bottom interior     -> 3h
      * bottom near rim     -> h
      * straight side upper -> 2h
      * straight side lower -> h
      * curved end walls    -> h
      * explicit rim pair   -> h/2 with coincident bottom/side normals
    """
    h = max(float(h), 1e-9)
    u, v, _w, length = _sweep_uv_frame(sweep)
    u2, v2 = u[:2], v[:2]
    a = np.asarray(sweep.a_xy, dtype=np.float64)
    b = np.asarray(sweep.b_xy, dtype=np.float64)
    r = float(sweep.radius)
    parts = []

    def _inside_capsule(xy: np.ndarray):
        _t, nearest = capsule2d_parameter(xy, a, b)
        rr = np.linalg.norm(xy - nearest, axis=1)
        return rr <= r + 1e-10, np.maximum(r - rr, 0.0), nearest, rr

    # Bottom interior: keep only points comfortably away from the sharp rim.
    hb = 3.0 * h
    rim_band = max(1.75 * h, 0.55 * hb)
    qu = _centered_samples(-r, length + r, hb)
    qv = _centered_samples(-r, r, hb)
    U, V = np.meshgrid(qu, qv, indexing='ij')
    xy = a[None, :] + U.reshape(-1, 1) * u2 + V.reshape(-1, 1) * v2
    inside, dist_to_rim, _nearest, _rr = _inside_capsule(xy)
    keep = inside & (dist_to_rim >= rim_band)
    p = np.column_stack([xy[keep], np.full(np.count_nonzero(keep), float(sweep.z_lo))])
    n = np.broadcast_to(np.array([0.0, 0.0, 1.0]), p.shape).copy()
    _append_surface_part(parts, p, n, hb, sweep.tool_id)

    # Bottom transition annulus near the rim.  This reduces the size jump from 3h
    # interior splats to the exact sharp-rim pair and helps prevent visible gaps.
    hbt = h
    qu = _centered_samples(-r, length + r, hbt)
    qv = _centered_samples(-r, r, hbt)
    U, V = np.meshgrid(qu, qv, indexing='ij')
    xy = a[None, :] + U.reshape(-1, 1) * u2 + V.reshape(-1, 1) * v2
    inside, dist_to_rim, _nearest, _rr = _inside_capsule(xy)
    rim_guard = min(0.45*h, 0.20*r)
    keep = inside & (dist_to_rim >= rim_guard) & (dist_to_rim < max(2.25 * h, rim_band + 0.25 * h))
    p = np.column_stack([xy[keep], np.full(np.count_nonzero(keep), float(sweep.z_lo))])
    n = np.broadcast_to(np.array([0.0, 0.0, 1.0]), p.shape).copy()
    _append_surface_part(parts, p, n, hbt, sweep.tool_id)

    z0 = max(float(sweep.z_lo), float(stock_lo[2]) - 2*h)
    z1 = min(float(sweep.z_hi), float(stock_hi[2]) + 2*h)
    if z1 > z0 + 1e-12:
        lower_band_hi = min(z1, z0 + 2.0 * h)

        # Long straight walls: use h near the floor/rim, then 2h higher up.
        tu_h = _centered_samples(0.0, length, h) if length > 1e-12 else np.empty(0)
        side_guard=min(z1,z0+rim_guard)
        zz_low = _centered_samples(side_guard, lower_band_hi, h) if lower_band_hi > side_guard + 1e-12 else np.empty(0)
        if len(tu_h) and len(zz_low):
            line = a + tu_h[:, None] * u2
            for side_sign in (+1.0, -1.0):
                xy_side = line + side_sign * r * v2
                pxy = np.repeat(xy_side, len(zz_low), axis=0)
                pp = np.column_stack([pxy, np.tile(zz_low, len(xy_side))])
                cutter_out = side_sign * v2
                nn = np.broadcast_to(np.array([-cutter_out[0], -cutter_out[1], 0.0]), pp.shape).copy()
                _append_surface_part(parts, pp, nn, h, sweep.tool_id)

        hs = 2.0 * h
        z_upper0 = min(z1, z0 + 1.5 * h)
        tu = _centered_samples(0.0, length, hs) if length > 1e-12 else np.empty(0)
        zz = _centered_samples(z_upper0, z1, hs) if z1 > z_upper0 + 1e-12 else np.empty(0)
        if len(tu) and len(zz):
            line = a + tu[:, None] * u2
            for side_sign in (+1.0, -1.0):
                xy_side = line + side_sign * r * v2
                pxy = np.repeat(xy_side, len(zz), axis=0)
                pp = np.column_stack([pxy, np.tile(zz, len(xy_side))])
                cutter_out = side_sign * v2
                nn = np.broadcast_to(np.array([-cutter_out[0], -cutter_out[1], 0.0]), pp.shape).copy()
                _append_surface_part(parts, pp, nn, hs, sweep.tool_id)

        # Curved end walls retain base h because curvature is 1/R.
        n_cap = max(6, int(math.ceil(math.pi * r / h)))
        da = math.pi / n_cap
        caps = []
        if length < 1e-12:
            nang = max(12, int(math.ceil(2*math.pi*r/h)))
            ang = (np.arange(nang)+0.5)*(2*math.pi/nang)
            dirs = np.cos(ang)[:,None]*u2 + np.sin(ang)[:,None]*v2
            caps.append((a + r*dirs, dirs))
        else:
            ang0 = math.pi*0.5 + (np.arange(n_cap)+0.5)*da
            d0 = np.cos(ang0)[:,None]*u2 + np.sin(ang0)[:,None]*v2
            caps.append((a + r*d0, d0))
            ang1 = -math.pi*0.5 + (np.arange(n_cap)+0.5)*da
            d1 = np.cos(ang1)[:,None]*u2 + np.sin(ang1)[:,None]*v2
            caps.append((b + r*d1, d1))
        zz_h = _centered_samples(min(z1,z0+rim_guard), z1, h)
        for per, out_xy in caps:
            pxy = np.repeat(per, len(zz_h), axis=0)
            pp = np.column_stack([pxy, np.tile(zz_h, len(per))])
            no = np.repeat(out_xy, len(zz_h), axis=0)
            nn = np.column_stack([-no, np.zeros(len(no))])
            _append_surface_part(parts, pp, nn, h, sweep.tool_id)

    # Explicit sharp bottom/side rim.  Same positions, two incident normals.
    hr = 0.5 * h
    perimeter, cutter_out = _capsule_perimeter_xy(sweep, hr)
    rp = np.column_stack([perimeter, np.full(len(perimeter), float(sweep.z_lo))])
    bottom_n = np.broadcast_to(np.array([0.0, 0.0, 1.0]), rp.shape).copy()
    side_n = np.column_stack([-cutter_out, np.zeros(len(cutter_out))])
    _append_surface_part(parts, rp, bottom_n, hr, sweep.tool_id)
    _append_surface_part(parts, rp, side_n, hr, sweep.tool_id)

    return _dedupe_splat_cloud(_cloud_from_parts(parts), h*2e-4)


def _ball_endpoint_cap_rings(sweep: BallSweep, h: float, is_start: bool):
    """Nearly equal surface-distance samples on one lower spherical quarter-cap."""
    u, v, w, _length = _sweep_uv_frame(sweep)
    r = float(sweep.radius)
    cxy = sweep.a_xy if is_start else sweep.b_xy
    c = np.array([cxy[0], cxy[1], sweep.center_z], dtype=np.float64)
    n_alpha = max(2, int(math.ceil(0.5 * math.pi * r / h)))
    dalpha = 0.5 * math.pi / n_alpha
    points, normals = [], []
    for i in range(n_alpha):
        if is_start:
            alpha = math.pi*0.5 + (i+0.5)*dalpha
        else:
            alpha = (i+0.5)*dalpha
        ring_r = r * max(math.sin(alpha), 1e-9)
        n_beta = max(3, int(math.ceil(math.pi * ring_r / h)))
        beta = math.pi + (np.arange(n_beta, dtype=np.float64)+0.5)*(math.pi/n_beta)
        d = (math.cos(alpha)*u[None,:] +
             math.sin(alpha)*np.cos(beta)[:,None]*v[None,:] +
             math.sin(alpha)*np.sin(beta)[:,None]*w[None,:])
        points.append(c[None,:] + r*d)
        normals.append(-d)
    return np.concatenate(points, axis=0), np.concatenate(normals, axis=0)


def _stationary_lower_hemisphere_rings(sweep: BallSweep, h: float):
    u, v, w, _ = _sweep_uv_frame(sweep)
    r = float(sweep.radius)
    c = np.array([sweep.a_xy[0], sweep.a_xy[1], sweep.center_z], dtype=np.float64)
    n_lat = max(2, int(math.ceil(0.5*math.pi*r/h)))
    dpsi = 0.5*math.pi/n_lat
    points, normals = [], []
    for i in range(n_lat):
        psi = (i+0.5)*dpsi
        ring_r = r*math.sin(psi)
        n_phi = max(6, int(math.ceil(2*math.pi*ring_r/h)))
        phi = (np.arange(n_phi,dtype=np.float64)+0.5)*(2*math.pi/n_phi)
        d = (math.sin(psi)*np.cos(phi)[:,None]*u[None,:] +
             math.sin(psi)*np.sin(phi)[:,None]*v[None,:] -
             math.cos(psi)*w[None,:])
        points.append(c[None,:]+r*d)
        normals.append(-d)
    return np.concatenate(points,axis=0), np.concatenate(normals,axis=0)


def _ball_formal_surface_candidates(sweep: BallSweep, h: float,
                                    stock_lo: np.ndarray, stock_hi: np.ndarray) -> SplatCloud:
    """Ball End sampling with approximately uniform *surface-distance* spacing h."""
    h = max(float(h), 1e-9)
    u, _v, _w, length = _sweep_uv_frame(sweep)
    parts = []
    r = float(sweep.radius)
    center_z = float(sweep.center_z)

    if length < 1e-12:
        p, n = _stationary_lower_hemisphere_rings(sweep, h)
        _append_surface_part(parts, p, n, h, sweep.tool_id)
    else:
        # Central lower swept half-cylinder: U arc-length x theta arc-length.
        t = _centered_samples(0.0, length, h)
        n_ang = max(6, int(math.ceil(math.pi*r/h)))
        ang = math.pi + (np.arange(n_ang,dtype=np.float64)+0.5)*(math.pi/n_ang)
        T, A = np.meshgrid(t, ang, indexing='ij')
        center = np.array([sweep.a_xy[0], sweep.a_xy[1], center_z])[None,:] + T.reshape(-1,1)*u
        # Reuse v/w from frame explicitly.
        _u, v, w, _ = _sweep_uv_frame(sweep)
        d = np.cos(A).reshape(-1,1)*v + np.sin(A).reshape(-1,1)*w
        p = center + r*d
        _append_surface_part(parts, p, -d, h, sweep.tool_id)
        for is_start in (True, False):
            p, n = _ball_endpoint_cap_rings(sweep, h, is_start)
            _append_surface_part(parts, p, n, h, sweep.tool_id)

    # Smooth upper cylindrical wall.  Keep h because circumferential curvature is 1/R.
    z0 = max(center_z, float(stock_lo[2])-2*h)
    z1 = min(float(sweep.z_hi), float(stock_hi[2])+2*h)
    if z1 > z0 + 1e-12:
        perimeter, out_xy = _capsule_perimeter_xy(sweep, h)
        zz = _centered_samples(z0, z1, h)
        pxy = np.repeat(perimeter, len(zz), axis=0)
        pp = np.column_stack([pxy, np.tile(zz, len(perimeter))])
        no = np.repeat(out_xy, len(zz), axis=0)
        nn = np.column_stack([-no, np.zeros(len(no))])
        _append_surface_part(parts, pp, nn, h, sweep.tool_id)

    return _dedupe_splat_cloud(_cloud_from_parts(parts), h*2e-4)


def formal_direct_surface_candidates(sweep, h: float,
                                     stock_lo: np.ndarray, stock_hi: np.ndarray) -> SplatCloud:
    if hasattr(sweep, 'a_xyz'):
        from cutsim.geometry.linear_surface import linear_surface_candidates
        return linear_surface_candidates(sweep, h, stock_lo, stock_hi)
    if isinstance(sweep, FlatSweep):
        return _flat_formal_surface_candidates(sweep, h, stock_lo, stock_hi)
    if isinstance(sweep, BallSweep):
        return _ball_formal_surface_candidates(sweep, h, stock_lo, stock_hi)
    raise TypeError(f"Unsupported sweep type: {type(sweep).__name__}")


class ToolProfileCache:
    """Reusable *section-2DGS* cache for rotational cutters.

    The cache stores compact cutter-local 2-D 2DGS templates rather than full
    command-specific 3-D clouds.  A template contains section position and normal
    coefficients in the local V-W plane.  During a cut the template is only:

        1) oriented by the current sweep frame,
        2) translated along A->B for the straight swept body, or
        3) rotated around an endpoint for a start/end cap.

    Flat-End bottom candidates are constructed from a cached bottom section line
    plus cached rotational half-disk templates: tool section 2DGS x tool motion.

    Local Grid + analytic CSG + Bisection resolve Boolean intersections and
    thin/sharp ambiguous regions.
    """
    def __init__(self, enabled: bool = True):
        self.enabled=bool(enabled)
        self._cache={}
        self.hits=0
        self.misses=0
        self.prewarm_ms=0.0
        self.prewarm_profile_values=0
        self.last_expand_probe_ms=0.0
        self.last_expand_direct_ms=0.0

    @staticmethod
    def _f(x):
        return float(x)

    def clear(self):
        self._cache.clear(); self.hits=0; self.misses=0
        self.prewarm_ms=0.0; self.prewarm_profile_values=0
        self.last_expand_probe_ms=0.0; self.last_expand_direct_ms=0.0

    def _get(self,key,builder):
        if self.enabled and key in self._cache:
            self.hits+=1
            return self._cache[key]
        value=builder()
        self.misses+=1
        if self.enabled:
            self._cache[key]=value
        return value

    def centered_transverse(self, radius, step):
        r=self._f(radius); h=self._f(step)
        return self._get(('transverse',r,h),lambda:_centered_samples(-r,r,h))

    def cutting_section(self, command, pitch, ball_angle_deg=None):
        """Reusable physical-tool section, independent of its path and pose."""
        from cutsim.geometry.tool_section import cutting_section_profile
        cutting = command.cutting_range
        if cutting is None:
            raise ValueError('A finite cutting_range is required for a cutting section')
        if ball_angle_deg is None:
            ball_angle_deg = command.gs_sampling.resolve_ball_angle()
        if command.tool_type.lower() != 'ball':
            ball_angle_deg = None
        key = ('cutting_section', command.tool_type.lower(), float(command.radius),
               float(cutting.start), float(cutting.end), float(pitch), ball_angle_deg)
        return self._get(key, lambda: cutting_section_profile(command.tool_type.lower(), command.radius,
                         cutting.start, cutting.end, pitch, ball_angle_deg))

    def flat_bottom_profile(self, radius, step, v_lo=None, v_hi=None):
        """Cached Flat-End bottom section 2DGS in local (v,z,nv,nz).

        z is relative to the tool bottom.  The material-side normal for the cutter
        bottom is +W, hence (nv,nz)=(0,+1).
        """
        r=self._f(radius); h=max(self._f(step),1e-9)
        lo=-r if v_lo is None else max(-r,self._f(v_lo))
        hi= r if v_hi is None else min( r,self._f(v_hi))
        key=('flat_bottom_profile',r,h,lo,hi)
        def build():
            if hi <= lo + 1e-12:
                return np.empty((0,4),dtype=np.float64)
            vv=_centered_samples(lo,hi,h)
            return np.column_stack([vv,np.zeros(len(vv)),np.zeros(len(vv)),np.ones(len(vv))])
        return self._get(key,build)

    def flat_cap_patch(self, radius, step, radial_lo, radial_hi, end):
        """Cached rotational bottom-cap 2DGS centres in local endpoint (u,v).

        ``end``: -1 full disk, 0 start half-disk (u<=0), 1 end half-disk (u>=0).
        Concentric rings use Ntheta proportional to radius so surface spacing remains
        close to ``step`` instead of becoming excessively dense near the centre.
        """
        r=self._f(radius); h=max(self._f(step),1e-9)
        r0=max(0.0,min(r,self._f(radial_lo))); r1=max(r0,min(r,self._f(radial_hi)))
        end=int(end)
        key=('flat_cap_patch',r,h,r0,r1,end)
        def build():
            if r1 <= r0 + 1e-12:
                return np.empty((0,2),dtype=np.float64)
            # Cell-centred radial 2DGS.  Add the true centre only when this patch owns it.
            rho=_centered_samples(r0,r1,h)
            chunks=[]
            if r0 <= 1e-12:
                chunks.append(np.zeros((1,2),dtype=np.float64))
            if end == -1:
                amin,span=0.0,2.0*math.pi
            elif end == 0:
                amin,span=0.5*math.pi,math.pi
            else:
                amin,span=-0.5*math.pi,math.pi
            for rr in rho:
                n=max(1,int(math.ceil(span*max(float(rr),0.25*h)/h)))
                aa=amin+(np.arange(n,dtype=np.float64)+0.5)*(span/n)
                chunks.append(np.column_stack([rr*np.cos(aa),rr*np.sin(aa)]))
            return np.concatenate(chunks,axis=0) if chunks else np.empty((0,2),dtype=np.float64)
        return self._get(key,build)

    def circle_dirs(self, radius, step):
        r=self._f(radius); h=self._f(step)
        def build():
            n=max(12,int(math.ceil(2.0*math.pi*r/max(h,1e-9))))
            a=(np.arange(n,dtype=np.float64)+0.5)*(2.0*math.pi/n)
            return np.column_stack([np.cos(a),np.sin(a)])
        return self._get(('circle',r,h),build)

    def capsule_cap_dirs(self, radius, step, end):
        r=self._f(radius); h=self._f(step); end=int(end)
        def build():
            n=max(6,int(math.ceil(math.pi*r/max(h,1e-9))))
            da=math.pi/n
            if end==0:
                a=math.pi*0.5+(np.arange(n,dtype=np.float64)+0.5)*da
            else:
                a=-math.pi*0.5+(np.arange(n,dtype=np.float64)+0.5)*da
            return np.column_stack([np.cos(a),np.sin(a)])
        return self._get(('capsule_cap',r,h,end),build)

    def ball_lower_meridian(self, radius, step, ball_angle_deg=None):
        """Cached Ball-End lower radial meridian 2DGS.

        Columns are ``(rho, z, n_rho, n_z)`` in tool-local cylindrical coordinates.
        ``rho>=0`` and ``z<=0`` are relative to the ball centre.  The material-side
        normal is stored directly, so the same local 2DGS can be mirrored for the
        straight V-W section or rotated around W for a spherical endpoint cap.
        """
        r=self._f(radius); h=max(self._f(step),1e-9)
        def build():
            if ball_angle_deg is None:
                n=max(3,int(math.ceil(0.5*math.pi*r/h)))
                psi=(np.arange(n,dtype=np.float64)+0.5)*(0.5*math.pi/n)
            else:
                psi=np.deg2rad(np.r_[np.arange(0.,90.,ball_angle_deg),90.])
            sr=np.sin(psi); cz=np.cos(psi)
            # Cutter surface direction=(+rho,-z); material normal is its negative.
            return np.column_stack([r*sr,-r*cz,-sr,cz])
        key = (('ball_lower_meridian',r,h) if ball_angle_deg is None else
               ('ball_lower_angle',r,float(ball_angle_deg)))
        return self._get(key,build)

    def ball_side_meridian(self, radius, step, z_rel_lo, z_rel_hi):
        """Cached upper cylindrical side profile in (rho,z,nrho,nz)."""
        r=self._f(radius); h=max(self._f(step),1e-9)
        lo=self._f(z_rel_lo); hi=self._f(z_rel_hi)
        key=('ball_side_meridian',r,h,lo,hi)
        def build():
            if hi <= lo + 1e-12:
                return np.empty((0,4),dtype=np.float64)
            zz=_centered_samples(lo,hi,h)
            return np.column_stack([np.full(len(zz),r),zz,
                                    -np.ones(len(zz)),np.zeros(len(zz))])
        return self._get(key,build)

    def prepare_profile(self,sweep,h,stock_lo=None,stock_hi=None,ball_angle_deg=None):
        r=float(sweep.radius); h=max(float(h),1e-9)
        if isinstance(sweep,BallSweep):
            # Dense h profile remains available for probes.  Direct smooth surfaces
            # use a coarser reusable profile; Local Grid retains fine complexity.
            self.ball_lower_meridian(r,h,ball_angle_deg)
            hd=max(h,float(DIRECT_SMOOTH_SCALE)*h)
            self.ball_lower_meridian(r,hd,ball_angle_deg)
            if stock_lo is not None and stock_hi is not None:
                cz=float(sweep.center_z)
                for hh in (h,hd):
                    z0=max(cz,float(stock_lo[2])-2.0*hh)
                    z1=min(float(sweep.z_hi),float(stock_hi[2])+2.0*hh)
                    self.ball_side_meridian(r,hh,z0-cz,z1-cz)
        elif isinstance(sweep,FlatSweep):
            self.centered_transverse(r,h)
            self.circle_dirs(r,h)
            self.capsule_cap_dirs(r,h,0); self.capsule_cap_dirs(r,h,1)
            self.capsule_cap_dirs(r,0.5*h,0); self.capsule_cap_dirs(r,0.5*h,1)
            hr=max(0.5*h,float(DIRECT_RIM_SCALE)*h)
            self.capsule_cap_dirs(r,hr,0); self.capsule_cap_dirs(r,hr,1)
            hb=3.0*h
            hf=max(h,float(DIRECT_SMOOTH_SCALE)*h)
            hr=max(0.5*h,float(DIRECT_RIM_SCALE)*h)
            rim_band=max(1.75*h,0.55*hb)
            rim_depth=max(2.25*h,rim_band+0.25*h)
            interior=max(0.0,r-rim_band)
            rim_inner=max(0.0,r-rim_depth)
            rim_guard=min(0.45*h,0.20*r)
            rim_outer=max(rim_inner,r-rim_guard)
            self.flat_bottom_profile(r,h)  # dense probe profile
            if rim_outer>rim_inner+1e-12:
                self.flat_bottom_profile(r,hf,-rim_outer,-rim_inner)
                self.flat_bottom_profile(r,hf,rim_inner,rim_outer)
            if interior>1e-12:
                self.flat_bottom_profile(r,hb,-interior,interior)
            for end in (-1,0,1):
                self.flat_cap_patch(r,h,0.0,r,end)
                if rim_outer>rim_inner+1e-12:
                    self.flat_cap_patch(r,hf,rim_inner,rim_outer,end)
                if interior>1e-12:
                    self.flat_cap_patch(r,hb,0.0,interior,end)
        return None

    def prewarm_program(self,program,stock_top,stock_lo,stock_hi,grid_sizing,gs_density=1.):
        if not self.enabled:
            self.prewarm_ms=0.0; self.prewarm_profile_values=0; return
        t0=time.perf_counter(); h0=self.hits; m0=self.misses
        for cmd in program:
            sweep=command_sweep(cmd,stock_top)
            pitch = cmd.gs_sampling.resolve_sweep_pitch(sweep, grid_sizing, gs_density)
            angle = cmd.gs_sampling.resolve_ball_angle(gs_density)
            self.prepare_profile(sweep,pitch,stock_lo,stock_hi,ball_angle_deg=angle)
            if cmd.cutting_range is not None:
                self.cutting_section(cmd, pitch, angle)
        self.hits=h0; self.misses=m0
        self.prewarm_ms=(time.perf_counter()-t0)*1000.0
        self.prewarm_profile_values=self.profile_values

    @property
    def entries(self):
        return len(self._cache)

    @property
    def profile_values(self):
        n=0
        for v in self._cache.values():
            if isinstance(v,np.ndarray): n+=int(v.size)
            elif isinstance(v,(tuple,list)):
                for x in v:
                    if isinstance(x,np.ndarray): n+=int(x.size)
        return n

    @property
    def samples(self):
        return self.profile_values

    @property
    def prewarm_samples(self):
        return self.prewarm_profile_values

    def get_probe(self,sweep,h,stock_lo,stock_hi,*,ball_angle_deg=None):
        t0=time.perf_counter()
        if hasattr(sweep, 'a_xyz'):
            from cutsim.geometry.linear_surface import linear_surface_candidates
            cloud = linear_surface_candidates(sweep, h, stock_lo, stock_hi, self, ball_angle_deg=ball_angle_deg)
            ms = (time.perf_counter()-t0)*1000.
            self.last_expand_probe_ms = ms
            return cloud, True, ms
        self.prepare_profile(sweep,h,stock_lo,stock_hi,ball_angle_deg)
        cloud=_profile_probe_surface_candidates(self,sweep,h,stock_lo,stock_hi,ball_angle_deg)
        ms=(time.perf_counter()-t0)*1000.0
        self.last_expand_probe_ms=ms
        return cloud, True, ms

    def get_direct(self,sweep,h,stock_lo,stock_hi,*,ball_angle_deg=None):
        t0=time.perf_counter()
        if hasattr(sweep, 'a_xyz'):
            from cutsim.geometry.linear_surface import linear_surface_candidates
            cloud = linear_surface_candidates(sweep, h, stock_lo, stock_hi, self, ball_angle_deg=ball_angle_deg)
            ms = (time.perf_counter()-t0)*1000.
            self.last_expand_direct_ms = ms
            return cloud, True, ms
        self.prepare_profile(sweep,h,stock_lo,stock_hi,ball_angle_deg)
        cloud=_profile_direct_surface_candidates(self,sweep,h,stock_lo,stock_hi,ball_angle_deg)
        ms=(time.perf_counter()-t0)*1000.0
        self.last_expand_direct_ms=ms
        return cloud, True, ms


def _profile_capsule_perimeter_xy(cache: ToolProfileCache, sweep, h: float):
    """Capsule perimeter from cached straight-section + rotational cap templates."""
    u,v,_w,length=_sweep_uv_frame(sweep); u2,v2=u[:2],v[:2]
    a=np.asarray(sweep.a_xy,dtype=np.float64); b=np.asarray(sweep.b_xy,dtype=np.float64)
    r=float(sweep.radius); pts=[]; outward=[]
    if length < 1e-12:
        d=cache.circle_dirs(r,h)
        dirs=d[:,0,None]*u2+d[:,1,None]*v2
        return a+r*dirs,dirs
    t=_centered_samples(0.0,length,h)
    line=a+t[:,None]*u2
    pts.extend([line+r*v2,line-r*v2])
    outward.extend([np.broadcast_to(v2,line.shape),np.broadcast_to(-v2,line.shape)])
    for end,c in ((0,a),(1,b)):
        d=cache.capsule_cap_dirs(r,h,end)
        dirs=d[:,0,None]*u2+d[:,1,None]*v2
        pts.append(c+r*dirs); outward.append(dirs)
    return np.concatenate(pts,axis=0),np.concatenate(outward,axis=0)


def _local_dirs_to_world(d,u,v,w):
    d=np.asarray(d,dtype=np.float64).reshape(-1,3)
    return d[:,0,None]*u+d[:,1,None]*v+d[:,2,None]*w


def _expand_vz_profile(profile, t, origin, u, v, w):
    """Instantiate cached V-W section 2DGS along straight sweep coordinate t."""
    prof=np.asarray(profile,dtype=np.float64).reshape(-1,4)
    tt=np.asarray(t,dtype=np.float64).reshape(-1)
    if not len(prof) or not len(tt):
        return np.empty((0,3),dtype=np.float64),np.empty((0,3),dtype=np.float64)
    nprof=len(prof)
    tr=np.repeat(tt,nprof)
    q=np.tile(prof,(len(tt),1))
    o=np.asarray(origin,dtype=np.float64).reshape(3)
    p=o[None,:]+tr[:,None]*u+q[:,0,None]*v+q[:,1,None]*w
    n=q[:,2,None]*v+q[:,3,None]*w
    return p,n


def _radial_meridian_to_vz_section(meridian):
    """Mirror one rho>=0 radial profile into the full local V-W section."""
    m=np.asarray(meridian,dtype=np.float64).reshape(-1,4)
    if not len(m):
        return np.empty((0,4),dtype=np.float64)
    # Positive V uses the cached radial basis directly.  Negative V mirrors both
    # position rho and its radial normal component; z/nz stay unchanged.
    pos=np.column_stack([m[:,0],m[:,1],m[:,2],m[:,3]])
    neg=np.column_stack([-m[:,0],m[:,1],-m[:,2],m[:,3]])
    return np.concatenate([neg[::-1],pos],axis=0)


def _rotate_radial_profile_world(profile, center, u, v, w, step, end):
    """Rotate cached radial 2DGS around tool axis W.

    ``end=-1`` creates a full revolution (stationary tool), ``end=0`` the start
    half-cap (U<=0), and ``end=1`` the end half-cap (U>=0).  Angular sample count
    grows with ring radius so neighbouring 2DGS remain approximately ``step`` apart.
    """
    m=np.asarray(profile,dtype=np.float64).reshape(-1,4)
    if not len(m):
        return np.empty((0,3),dtype=np.float64),np.empty((0,3),dtype=np.float64)
    h=max(float(step),1e-9); end=int(end)
    if end==-1:
        amin,span=0.0,2.0*math.pi
    elif end==0:
        amin,span=0.5*math.pi,math.pi
    elif end==1:
        amin,span=-0.5*math.pi,math.pi
    else:
        raise ValueError(f'unsupported rotational end={end}')
    c=np.asarray(center,dtype=np.float64).reshape(3)
    pparts=[]; nparts=[]
    for rho,z,nrho,nz in m:
        rr=abs(float(rho))
        ntheta=max(1,int(math.ceil(span*max(rr,0.25*h)/h)))
        aa=amin+(np.arange(ntheta,dtype=np.float64)+0.5)*(span/ntheta)
        radial=np.cos(aa)[:,None]*u+np.sin(aa)[:,None]*v
        pparts.append(c[None,:]+float(rho)*radial+float(z)*w)
        nparts.append(float(nrho)*radial+float(nz)*w)
    return np.concatenate(pparts,axis=0),np.concatenate(nparts,axis=0)


def _ball_profile_surface_parts(cache, sweep, h, stock_lo, stock_hi, ball_angle_deg=None):
    """Return Ball-End lower/side local radial profiles for this stock window."""
    r=float(sweep.radius); cz=float(sweep.center_z); h=max(float(h),1e-9)
    lower=cache.ball_lower_meridian(r,h,ball_angle_deg)
    z0=max(cz,float(stock_lo[2])-2.0*h)
    z1=min(float(sweep.z_hi),float(stock_hi[2])+2.0*h)
    side=cache.ball_side_meridian(r,h,z0-cz,z1-cz) if z1>z0+1e-12 else np.empty((0,4),dtype=np.float64)
    return lower,side


def _flat_cap_world(cache, sweep, step, radial_lo, radial_hi, end, z):
    """Instantiate one cached rotational Flat-End bottom patch in world coordinates."""
    u,v,_w,_length=_sweep_uv_frame(sweep)
    cxy=np.asarray(sweep.a_xy if end in (-1,0) else sweep.b_xy,dtype=np.float64)
    local=cache.flat_cap_patch(sweep.radius,step,radial_lo,radial_hi,end)
    if not len(local):
        return np.empty((0,3),dtype=np.float64)
    pp=np.array([cxy[0],cxy[1],float(z)],dtype=np.float64)[None,:] + local[:,0,None]*u + local[:,1,None]*v
    return pp


def _profile_probe_surface_candidates(cache: ToolProfileCache, sweep, h: float,
                                      stock_lo: np.ndarray, stock_hi: np.ndarray, ball_angle_deg=None) -> SplatCloud:
    """Broad-phase candidates from cached section 2DGS x tool motion."""
    h=max(float(h),1e-9); u,v,w,length=_sweep_uv_frame(sweep)
    a=np.asarray(sweep.a_xy,dtype=np.float64); r=float(sweep.radius); parts=[]
    zpad=2.0*h; zsl=float(stock_lo[2])-zpad; zsh=float(stock_hi[2])+zpad
    if isinstance(sweep,FlatSweep):
        # Bottom: cached V-W section line x translation, plus profile x rotation caps.
        if length < 1e-12:
            pp=_flat_cap_world(cache,sweep,h,0.0,r,-1,sweep.z_lo)
            nn=np.broadcast_to(np.array([0.,0.,1.]),pp.shape).copy()
            _append_probe_parts(parts,pp,nn,h,sweep.tool_id)
        else:
            prof=cache.flat_bottom_profile(r,h)
            tt=_centered_samples(0.0,length,h)
            origin=np.array([a[0],a[1],float(sweep.z_lo)],dtype=np.float64)
            pp,nn=_expand_vz_profile(prof,tt,origin,u,v,w)
            _append_probe_parts(parts,pp,nn,h,sweep.tool_id)
            for end in (0,1):
                pp=_flat_cap_world(cache,sweep,h,0.0,r,end,sweep.z_lo)
                nn=np.broadcast_to(np.array([0.,0.,1.]),pp.shape).copy()
                _append_probe_parts(parts,pp,nn,h,sweep.tool_id)
        z0=max(float(sweep.z_lo),zsl); z1=min(float(sweep.z_hi),zsh)
        if z1>z0+1e-12:
            per,outxy=_profile_capsule_perimeter_xy(cache,sweep,h)
            zz=_centered_samples(z0,z1,h)
            pxy=np.repeat(per,len(zz),axis=0)
            pp=np.column_stack([pxy,np.tile(zz,len(per))])
            no=np.repeat(outxy,len(zz),axis=0)
            nn=np.column_stack([-no,np.zeros(len(no))])
            _append_probe_parts(parts,pp,nn,h,sweep.tool_id)
    elif isinstance(sweep,BallSweep):
        cz=float(sweep.center_z)
        center0=np.array([a[0],a[1],cz],dtype=np.float64)
        lower,side=_ball_profile_surface_parts(cache,sweep,h,stock_lo,stock_hi,ball_angle_deg)
        curve_h = h if ball_angle_deg is None else r*math.radians(ball_angle_deg)
        if length<1e-12:
            # Stationary Ball-End: rotate the very same cached radial profiles.
            for prof in (lower,side):
                spacing = curve_h if prof is lower else h
                pp,nn=_rotate_radial_profile_world(prof,center0,u,v,w,spacing,-1)
                _append_probe_parts(parts,pp,nn,spacing,sweep.tool_id)
        else:
            tt=_centered_samples(0.0,length,h)
            # Straight body = mirrored local V-W section x translation.
            for prof in (lower,side):
                full=_radial_meridian_to_vz_section(prof)
                pp,nn=_expand_vz_profile(full,tt,center0,u,v,w)
                _append_probe_parts(parts,pp,nn,max(h,curve_h) if prof is lower else h,sweep.tool_id)
            # Start/end surfaces = the SAME radial profiles x rotation.
            for end,cxy in ((0,sweep.a_xy),(1,sweep.b_xy)):
                c=np.array([cxy[0],cxy[1],cz],dtype=np.float64)
                for prof in (lower,side):
                    spacing = curve_h if prof is lower else h
                    pp,nn=_rotate_radial_profile_world(prof,c,u,v,w,spacing,end)
                    _append_probe_parts(parts,pp,nn,spacing,sweep.tool_id)
    else:
        raise TypeError(type(sweep).__name__)
    return _dedupe_probe_cloud(_cloud_from_parts(parts),h)


def _profile_direct_surface_candidates(cache: ToolProfileCache, sweep, h: float,
                                       stock_lo: np.ndarray, stock_hi: np.ndarray, ball_angle_deg=None) -> SplatCloud:
    """Direct candidates: cached section 2DGS x translation/rotation."""
    h=max(float(h),1e-9); u,v,w,length=_sweep_uv_frame(sweep)
    a=np.asarray(sweep.a_xy,dtype=np.float64); r=float(sweep.radius); parts=[]
    if isinstance(sweep,FlatSweep):
        hb=3.0*h
        hf=max(h,float(DIRECT_SMOOTH_SCALE)*h)
        hr=max(0.5*h,float(DIRECT_RIM_SCALE)*h)
        rim_band=max(1.75*h,0.55*hb)
        rim_depth=max(2.25*h,rim_band+0.25*h)
        interior=max(0.0,r-rim_band)
        rim_inner=max(0.0,r-rim_depth)
        # Ownership guard: the explicit h/2 sharp-rim pair owns the immediate
        # edge neighbourhood.  Ordinary h bottom/side samples start slightly inward.
        rim_guard=min(0.45*h,0.20*r)
        rim_outer=max(rim_inner,r-rim_guard)
        origin=np.array([a[0],a[1],float(sweep.z_lo)],dtype=np.float64)

        # Smooth bottom interior: one cached coarse section template x translation.
        if interior>1e-12:
            prof=cache.flat_bottom_profile(r,hb,-interior,interior)
            if length<1e-12:
                pp=_flat_cap_world(cache,sweep,hb,0.0,interior,-1,sweep.z_lo)
                nn=np.broadcast_to(np.array([0.,0.,1.]),pp.shape).copy()
            else:
                tt=_centered_samples(0.0,length,hb)
                pp,nn=_expand_vz_profile(prof,tt,origin,u,v,w)
                for end in (0,1):
                    cp=_flat_cap_world(cache,sweep,hb,0.0,interior,end,sweep.z_lo)
                    cn=np.broadcast_to(np.array([0.,0.,1.]),cp.shape).copy()
                    _append_surface_part(parts,cp,cn,hb,sweep.tool_id)
            _append_surface_part(parts,pp,nn,hb,sweep.tool_id)

        # Near-rim bottom band. Leave a small ownership guard for the explicit
        # h/2 rim pair instead of stacking h and h/2 samples at the same edge.
        if rim_outer>rim_inner+1e-12:
            if length<1e-12:
                pp=_flat_cap_world(cache,sweep,hf,rim_inner,rim_outer,-1,sweep.z_lo)
                nn=np.broadcast_to(np.array([0.,0.,1.]),pp.shape).copy()
                _append_surface_part(parts,pp,nn,hf,sweep.tool_id)
            else:
                tt=_centered_samples(0.0,length,hf)
                for lo,hi in ((-rim_outer,-rim_inner),(rim_inner,rim_outer)):
                    prof=cache.flat_bottom_profile(r,hf,lo,hi)
                    pp,nn=_expand_vz_profile(prof,tt,origin,u,v,w)
                    _append_surface_part(parts,pp,nn,hf,sweep.tool_id)
                for end in (0,1):
                    pp=_flat_cap_world(cache,sweep,hf,rim_inner,rim_outer,end,sweep.z_lo)
                    nn=np.broadcast_to(np.array([0.,0.,1.]),pp.shape).copy()
                    _append_surface_part(parts,pp,nn,hf,sweep.tool_id)

        # Side wall families remain section/profile driven as before.
        z0=max(float(sweep.z_lo),float(stock_lo[2])-2*h); z1=min(float(sweep.z_hi),float(stock_hi[2])+2*h)
        if z1>z0+1e-12:
            lowhi=min(z1,z0+2.0*h)
            side_guard=min(z1,z0+rim_guard)
            th=_centered_samples(0.0,length,hf) if length>1e-12 else np.empty(0)
            zz=_centered_samples(side_guard,lowhi,hf) if lowhi>side_guard+1e-12 else np.empty(0)
            if len(th) and len(zz):
                line=a+th[:,None]*u[:2]
                for ss in (+1.,-1.):
                    xys=line+ss*r*v[:2]; pxy=np.repeat(xys,len(zz),axis=0)
                    pp=np.column_stack([pxy,np.tile(zz,len(xys))]); co=ss*v[:2]
                    nn=np.broadcast_to(np.array([-co[0],-co[1],0.]),pp.shape).copy()
                    _append_surface_part(parts,pp,nn,hf,sweep.tool_id)
            hs=max(2.0*h,hf); zu=min(z1,z0+1.5*h)
            tt=_centered_samples(0.0,length,hs) if length>1e-12 else np.empty(0)
            zz=_centered_samples(zu,z1,hs) if z1>zu+1e-12 else np.empty(0)
            if len(tt) and len(zz):
                line=a+tt[:,None]*u[:2]
                for ss in (+1.,-1.):
                    xys=line+ss*r*v[:2]; pxy=np.repeat(xys,len(zz),axis=0)
                    pp=np.column_stack([pxy,np.tile(zz,len(xys))]); co=ss*v[:2]
                    nn=np.broadcast_to(np.array([-co[0],-co[1],0.]),pp.shape).copy()
                    _append_surface_part(parts,pp,nn,hs,sweep.tool_id)
            zz_h=_centered_samples(min(z1,z0+rim_guard),z1,hf)
            if length<1e-12:
                d=cache.circle_dirs(r,hf); dirs=d[:,0,None]*u[:2]+d[:,1,None]*v[:2]
                caps=[(a+r*dirs,dirs)]
            else:
                caps=[]
                for end,c in ((0,np.asarray(sweep.a_xy)),(1,np.asarray(sweep.b_xy))):
                    d=cache.capsule_cap_dirs(r,hf,end); dirs=d[:,0,None]*u[:2]+d[:,1,None]*v[:2]
                    caps.append((c+r*dirs,dirs))
            for per,outxy in caps:
                pxy=np.repeat(per,len(zz_h),axis=0)
                pp=np.column_stack([pxy,np.tile(zz_h,len(per))])
                no=np.repeat(outxy,len(zz_h),axis=0)
                _append_surface_part(parts,pp,np.column_stack([-no,np.zeros(len(no))]),hf,sweep.tool_id)

        # Sharp rim remains denser than smooth Direct surfaces, but h/2 everywhere
        # was excessive once complex intersections moved to Local Grid.
        per,outxy=_profile_capsule_perimeter_xy(cache,sweep,hr)
        rp=np.column_stack([per,np.full(len(per),float(sweep.z_lo))])
        _append_surface_part(parts,rp,np.broadcast_to(np.array([0.,0.,1.]),rp.shape).copy(),hr,sweep.tool_id)
        _append_surface_part(parts,rp,np.column_stack([-outxy,np.zeros(len(outxy))]),hr,sweep.tool_id)

    elif isinstance(sweep,BallSweep):
        cz=float(sweep.center_z)
        hd=max(h,float(DIRECT_SMOOTH_SCALE)*h)
        center0=np.array([a[0],a[1],cz],dtype=np.float64)
        lower,side=_ball_profile_surface_parts(cache,sweep,hd,stock_lo,stock_hi,ball_angle_deg)
        curve_h = hd if ball_angle_deg is None else r*math.radians(ball_angle_deg)
        if length<1e-12:
            for prof in (lower,side):
                spacing = curve_h if prof is lower else hd
                pp,nn=_rotate_radial_profile_world(prof,center0,u,v,w,spacing,-1)
                _append_surface_part(parts,pp,nn,spacing,sweep.tool_id)
        else:
            tt=_centered_samples(0.0,length,hd)
            for prof in (lower,side):
                full=_radial_meridian_to_vz_section(prof)
                pp,nn=_expand_vz_profile(full,tt,center0,u,v,w)
                _append_surface_part(parts,pp,nn,max(hd,curve_h) if prof is lower else hd,sweep.tool_id)
            for end,cxy in ((0,sweep.a_xy),(1,sweep.b_xy)):
                c=np.array([cxy[0],cxy[1],cz],dtype=np.float64)
                for prof in (lower,side):
                    spacing = curve_h if prof is lower else hd
                    pp,nn=_rotate_radial_profile_world(prof,c,u,v,w,spacing,end)
                    _append_surface_part(parts,pp,nn,spacing,sweep.tool_id)
    else:
        raise TypeError(type(sweep).__name__)
    return _dedupe_splat_cloud(_cloud_from_parts(parts),h*2e-4)


class ToolSurfaceCache:
    """Precompute newest-tool surface candidates outside the cut-step hot path.

    The hybrid algorithm uses analytic surface samples as a broad phase.
    The cache stores candidate positions, normals, support and feature levels by an
    exact sweep/grid signature.  A fixed demo/program can be fully prewarmed during
    reset, so execute_next() normally performs only dictionary lookup for candidate
    generation.

    Intersections, thin remnants and sharp Boolean features use
    Local Grid + analytic SDF + Bisection.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._probe = {}
        self._direct = {}
        self.hits = 0
        self.misses = 0
        self.prewarm_ms = 0.0
        self.prewarm_samples = 0

    @staticmethod
    def _f(x):
        # Inputs come from the program and cut-dependent sizing. Keeping exact float
        # values avoids changing sampling geometry through cache-key quantization.
        return float(x)

    def clear(self):
        self._probe.clear()
        self._direct.clear()
        self.hits = 0
        self.misses = 0
        self.prewarm_ms = 0.0
        self.prewarm_samples = 0

    def _key(self, sweep, h: float, stock_lo: np.ndarray, stock_hi: np.ndarray):
        base = (
            str(sweep.kind),
            tuple(self._f(v) for v in np.asarray(sweep.a_xy).reshape(2)),
            tuple(self._f(v) for v in np.asarray(sweep.b_xy).reshape(2)),
            self._f(sweep.radius), self._f(h),
            self._f(stock_lo[2]), self._f(stock_hi[2]),
        )
        if hasattr(sweep, 'a_xyz'):
            return base + (type(sweep).__name__, tuple(sweep.a_xyz), tuple(sweep.b_xyz),
                           self._f(sweep.z_hi), int(sweep.tool_id),
                           getattr(sweep, 'cutting_start', None), getattr(sweep, 'cutting_end', None))
        if isinstance(sweep, FlatSweep):
            return base + (self._f(sweep.z_lo), self._f(sweep.z_hi), int(sweep.tool_id))
        if isinstance(sweep, BallSweep):
            return base + (self._f(sweep.tip_z), self._f(sweep.z_hi), int(sweep.tool_id))
        raise TypeError(f"Unsupported sweep type: {type(sweep).__name__}")

    def _get(self, mode: str, sweep, h: float, stock_lo: np.ndarray, stock_hi: np.ndarray, ball_angle_deg=None):
        table = self._probe if mode == 'probe' else self._direct
        key = self._key(sweep, h, stock_lo, stock_hi)
        if ball_angle_deg is not None and sweep.kind == 'ball':
            key += ('ball_angle_deg', float(ball_angle_deg))
        if self.enabled and key in table:
            self.hits += 1
            return table[key], True, 0.0

        t0 = time.perf_counter()
        if ball_angle_deg is not None and sweep.kind == 'ball':
            profiles = ToolProfileCache(False)
            cloud = (profiles.get_probe if mode == 'probe' else profiles.get_direct)(
                sweep, h, stock_lo, stock_hi, ball_angle_deg=ball_angle_deg)[0]
        elif mode == 'probe':
            cloud = analytic_sweep_surface_candidates(sweep, h, stock_lo, stock_hi)
        elif mode == 'direct':
            cloud = formal_direct_surface_candidates(sweep, h, stock_lo, stock_hi)
        else:
            raise ValueError(f"unknown tool surface cache mode: {mode!r}")
        build_ms = (time.perf_counter() - t0) * 1000.0
        self.misses += 1
        if self.enabled:
            table[key] = cloud
        return cloud, False, build_ms

    def get_probe(self, sweep, h: float, stock_lo: np.ndarray, stock_hi: np.ndarray, *, ball_angle_deg=None):
        return self._get('probe', sweep, h, stock_lo, stock_hi, ball_angle_deg)

    def get_direct(self, sweep, h: float, stock_lo: np.ndarray, stock_hi: np.ndarray, *, ball_angle_deg=None):
        return self._get('direct', sweep, h, stock_lo, stock_hi, ball_angle_deg)

    def prewarm_program(self, program, stock_top: float, stock_lo: np.ndarray,
                        stock_hi: np.ndarray, grid_sizing) -> None:
        if not self.enabled:
            self.prewarm_ms = 0.0
            self.prewarm_samples = 0
            return
        t0 = time.perf_counter()
        samples = 0
        # Do not count prewarm population as runtime cache hits/misses.
        hits0, misses0 = self.hits, self.misses
        for cmd in program:
            sweep = command_sweep(cmd, stock_top)
            h = cmd.gs_sampling.resolve_sweep_pitch(sweep, grid_sizing)
            angle = cmd.gs_sampling.resolve_ball_angle()
            probe, _, _ = self._get('probe', sweep, h, stock_lo, stock_hi, angle)
            direct, _, _ = self._get('direct', sweep, h, stock_lo, stock_hi, angle)
            samples += len(probe.positions) + len(direct.positions)
        self.hits, self.misses = hits0, misses0
        self.prewarm_ms = (time.perf_counter() - t0) * 1000.0
        self.prewarm_samples = int(samples)

    @property
    def entries(self) -> int:
        return len(self._probe) + len(self._direct)

    @property
    def samples(self) -> int:
        return int(sum(len(c.positions) for c in self._probe.values()) +
                   sum(len(c.positions) for c in self._direct.values()))

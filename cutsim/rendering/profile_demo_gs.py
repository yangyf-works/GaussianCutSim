"""Render-only 2DGS fitted to recorded Dexel endpoints, without resampling cuts.

Positions and normals come exclusively from the saved calculation. Circular
sigma estimates the tangent footprint of each tool-profile cell. It is capped
at grazing incidence; this is a surfel preview, not a watertight reconstruction.
"""
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from cutsim.rendering.clip_search import CLIP_SIGMA

from cutsim.geometry.frames import tangent_frames, rotation_matrices_to_quaternions
from cutsim.rendering.gs_schema import validate_gaussian_attributes


@dataclass
class GaussianSnapshot:
    attributes: dict
    u: np.ndarray
    v: np.ndarray
    preparation_ms: float


def profile_sigma(case, endpoints, normals):
    """Project the two cell derivatives onto the recorded endpoint plane."""
    ids = endpoints['ray_ids']
    cells = case.grid['cell_bounds'][ids]
    chart = case.grid['charts'][ids]
    radius = float(case.metadata['radius'])
    pitch = float(case.metadata['pitch'])
    rotation = np.asarray(case.metadata['rotation'])
    return profile_sigma_arrays(chart,cells,endpoints['distances'],case.grid['directions'][ids],
                                normals,radius,pitch,rotation)


def profile_sigma_arrays(chart,cells,distance,direction,normals,radius,pitch,rotation=None,feeds=None):
    """Batched footprint fit, also supporting an atlas of different tools."""
    count=len(cells)
    radius=np.broadcast_to(radius,(count,))
    pitch=np.broadcast_to(pitch,(count,))
    rotation=np.eye(3) if rotation is None else rotation
    s = cells[:, :2].mean(axis=1)
    phi = cells[:, 2:].mean(axis=1)
    radial = np.column_stack((np.cos(phi), np.sin(phi), np.zeros(count)))
    azimuth = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros(count)))
    a = radial.copy()
    b = s[:, None] * azimuth
    ball = chart == 'ball'
    a[ball] = (distance[ball] / radius[ball] * np.cos(s[ball] / radius[ball]))[:,None] * radial[ball]
    a[ball, 2] = distance[ball] / radius[ball] * np.sin(s[ball] / radius[ball])
    b[ball] = (distance[ball] * np.sin(s[ball] / radius[ball]))[:, None] * azimuth[ball]
    cylinder = chart == 'cylinder'
    a[cylinder] = [0., 0., 1.]
    b[cylinder] = distance[cylinder, None] * azimuth[cylinder]
    strip=np.char.startswith(chart,'strip_')
    if strip.any():
        if feeds is None:raise ValueError('Feed vectors required for swept profile cells')
        side=np.column_stack((-feeds[:,1],feeds[:,0],np.zeros(count)))
        a[strip]=feeds[strip];b[strip]=side[strip]
        b[chart=='strip_side']=[0.,0.,1.]
        curved=chart=='strip_ball';theta=phi[curved]/radius[curved]
        b[curved]=(distance[curved]/radius[curved]*np.cos(theta))[:,None]*side[curved]
        b[curved,2]=distance[curved]/radius[curved]*np.sin(theta)
    incidence = np.sum(normals * direction, axis=1)
    safe = np.where(incidence < 0., -1., 1.) * np.maximum(np.abs(incidence), 1e-8)
    widths = []
    for derivative, step in ((a, cells[:, 1]-cells[:, 0]), (b, cells[:, 3]-cells[:, 2])):
        derivative = derivative @ rotation.T
        tangent = derivative - direction * (np.sum(normals*derivative, axis=1)/safe)[:, None]
        widths.append(np.linalg.norm(tangent, axis=1) * step)
    spacing = np.hypot(*widths) / np.sqrt(2.)
    ramp = chart == 'ramp'
    # Ramp cells already store their two tangent-plane lengths in millimetres.
    spacing[ramp] = np.hypot(cells[ramp, 1]-cells[ramp, 0],
                             cells[ramp, 3]-cells[ramp, 2]) / np.sqrt(2.)
    return np.clip(.75 * spacing, .15*pitch, 4.*pitch)


def endpoint_surface_ids(case, endpoints, normals):
    """Keep stock faces and generating sweeps separate during GPU smoothing."""
    owners = endpoints['owners']
    identity = np.zeros(len(owners), np.int32)
    local_normal = normals @ np.asarray(case.metadata['rotation'])
    local_point = (endpoints['positions']-case.metadata['shift']) @ np.asarray(case.metadata['rotation'])
    # The six box faces are known independently of the original GS tile IDs.
    for axis in range(3):
        for side, sign in enumerate((-1., 1.)):
            bound = case.metadata['stock_low' if side == 0 else 'stock_high'][axis]
            match = ((owners < 0) & (sign*local_normal[:, axis] > 1.-1e-8)
                     & (np.abs(local_point[:, axis]-bound) < 1e-8))
            identity[match] = 1 + 2*axis + side
    cut = owners >= 0
    # IDs are never shared by different cutting passes. Separate planar caps
    # from side surfaces; native normal/depth gates also protect sharp joins.
    patch = np.zeros(len(owners), np.int32)
    if case.metadata['kind'] == 'flat':
        patch[normals[:, 2] > 1.-1e-8] = 1
    patch[normals[:, 2] < -1.+1e-8] = 2
    identity[cut] = 7 + 3*owners[cut] + patch[cut]
    return identity[:, None]


def build_snapshot(case, stage):
    started = perf_counter()
    endpoints = case.frames[stage].endpoints
    normals = np.asarray(endpoints['normals'], dtype=float)
    length = np.linalg.norm(normals, axis=1)
    if np.any(length < 1e-12):
        raise ValueError('2DGS requires nonzero endpoint normals')
    normals = normals / length[:, None]
    u, v = tangent_frames(normals)
    quaternion = rotation_matrices_to_quaternions(np.stack((u, v, normals), axis=2))
    attributes = dict(positions=endpoints['positions'].astype(np.float32),
                      rot=quaternion.astype(np.float32),
                      scale=profile_sigma(case, endpoints, normals).astype(np.float32)[:, None],
                      surface_color=np.ones((len(normals), 3), np.float32),
                      surface_id=endpoint_surface_ids(case, endpoints, normals))
    validate_gaussian_attributes(attributes)
    for value in attributes.values():
        value.flags.writeable = False
    return GaussianSnapshot(attributes, u, v, (perf_counter()-started)*1000)


class GaussianSnapshotCache:
    """Bounded cache: case/stage changes fit once; camera changes fit nothing."""
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.last_preparation_ms = 0.

    def get(self, case, stage):
        key = (id(case), stage)
        self.last_preparation_ms = 0.
        if key not in self.entries:
            snapshot = build_snapshot(case, stage)
            self.entries[key] = (case, snapshot)
            self.last_preparation_ms = snapshot.preparation_ms
        self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return self.entries[key][1]


def draw_attributes(case, snapshot, selected, color, size=1., clip_y=None):
    """Clip finite stock/section planes in the native normalized GS coordinates.

Two planes are uploaded. If a third plane intersects the circular support,
shrink the radius so the remaining planes cannot leak outside the box/section.
Cut-to-cut curved boundaries remain a local tangent approximation.
"""
    if not np.isfinite(size) or not .25 <= size <= 2.:
        raise ValueError('GS display size must be between 0.25 and 2')
    result = {name: value[selected].copy() for name, value in snapshot.attributes.items()}
    count = len(result['positions'])
    result['surface_color'][:] = color
    sigma = result['scale'][:, 0].astype(float) * size
    u, v = snapshot.u[selected], snapshot.v[selected]
    rotation = np.asarray(case.metadata['rotation'])
    # Snapshot centers have only upload precision. Plane distances use their
    # FP32 values too, so clipping remains aligned with the uploaded primitive.
    p = result['positions'].astype(float)
    local = (p-case.metadata['shift']) @ rotation
    distances, angles = [], []
    for axis in range(3):
        upper = case.metadata['stock_high'][axis]
        if axis == 1 and clip_y is not None:
            upper = min(upper, clip_y)
        for sign, bound in ((1., case.metadata['stock_low'][axis]),
                            (-1., upper)):
            gradient = sign*rotation[:, axis]
            x, y = u @ gradient, v @ gradient
            norm = np.hypot(x, y)
            clearance = np.maximum(sign*(local[:, axis]-bound), 0.)
            distances.append(np.divide(clearance, norm, out=np.full(count, np.inf), where=norm>1e-8))
            angles.append(np.arctan2(y, x))
    distances, angles = np.column_stack(distances), np.column_stack(angles)
    order = np.argsort(distances, axis=1, kind='stable')
    rows = np.arange(count)
    sigma = np.maximum(1e-7, np.minimum(sigma, distances[rows, order[:, 2]] / (CLIP_SIGMA+.00001)))
    result['scale'] = sigma.astype(np.float32)[:, None]
    for slot in range(2):
        distance = distances[rows, order[:, slot]]
        active = distance < CLIP_SIGMA*sigma
        angle = (angles[rows, order[:, slot]]+np.pi) % (2*np.pi)-np.pi
        encoded = angle.astype(np.float32)
        encoded[encoded >= np.float32(np.pi)] = -np.float32(np.pi)
        result[f'clip{slot+1}_angle'] = np.where(active, encoded, 4.).astype(np.float32)[:, None]
        result[f'clip{slot+1}_split'] = np.where(active, -distance/sigma, 0.).astype(np.float32)[:, None]
    validate_gaussian_attributes(result)
    return result

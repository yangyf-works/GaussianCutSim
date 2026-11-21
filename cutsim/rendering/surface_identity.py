"""Conservative analytic surface identity for render-only interpolation.

Zero means unknown: it must never permit blending between separate splats.
An ID is scoped to a material history whose sweep indices are stable prefixes.
These labels do not alter geometry, clipping, colors, or cutting decisions.
"""
from __future__ import annotations

import numpy as np

from cutsim.core.sweep import BallSweep, FlatSweep, capsule2d_parameter


def surface_ids(cloud, material):
    """Label only unambiguous generating faces using provenance AND geometry.

    IDs 1..6 are the six stock planes. A sweep uses 7+3*index+patch,
    where patch 0 is the side (including the tangent-continuous ball bottom),
    1 is a flat bottom, and 2 is the top. Different passes by the same tool
    never share an ID. Unsupported or ambiguous surfaces remain zero.
    """
    p = np.asarray(cloud.positions, dtype=np.float64)
    count = len(p)
    provenance = getattr(material, '_gs_surface_provenance', None)
    explicit = np.zeros(count, dtype=np.int32)
    if provenance is not None and hasattr(cloud, '_local_ids'):
        from cutsim.core.surface_provenance import SurfaceProvenance
        explicit = (provenance.lookup(cloud._local_ids) if isinstance(provenance,SurfaceProvenance)
                    else np.array([provenance.get(int(sid), 0) for sid in cloud._local_ids], dtype=np.int32))
    known = (explicit > 0) & (explicit < 7+3*len(material.sweeps))
    # The old final override already makes these exact generating identities
    # authoritative. Do not infer them again from every historical sweep.
    if np.all(known):
        return explicit.reshape(-1, 1)
    n = np.asarray(cloud.normals, dtype=np.float64)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    identity = np.zeros(count, dtype=np.int32)
    ambiguous = np.zeros(count, dtype=bool)
    if not count:
        return identity.reshape(-1, 1)
    # This tolerance establishes membership, not a license to blend nearby
    # surfaces: two matching candidates explicitly invalidate the identity.
    tol = np.maximum(.04 * np.asarray(cloud.support), 1e-6)
    alignment = np.cos(np.deg2rad(25.))
    tools = np.asarray(cloud.tool_ids)

    def assign(rows, match, value):
        ids = rows[np.asarray(match, dtype=bool)]
        ambiguous[ids] |= (identity[ids] != 0) & (identity[ids] != value)
        identity[ids] = value

    rows = np.flatnonzero((tools == 0) & ~known)
    for axis in range(3):
        for side, bound in enumerate((material.stock_lo[axis], material.stock_hi[axis])):
            sign = -1. if side == 0 else 1.
            match = ((np.abs(p[rows, axis] - bound) <= tol[rows]) &
                     (sign * n[rows, axis] >= alignment))
            # Check the finite face domain as well as its supporting plane.
            match &= np.all(p[rows] >= material.stock_lo - tol[rows, None], axis=1)
            match &= np.all(p[rows] <= material.stock_hi + tol[rows, None], axis=1)
            assign(rows, match, 1 + 2 * axis + side)

    for sweep_index, sweep in enumerate(material.sweeps):
        if hasattr(sweep, 'a_xyz'):
            rows = np.flatnonzero((tools > 0) & (tools == sweep.tool_id) & ~known)
            if len(rows):
                f, gradient = sweep.sdf_gradient(p[rows])
                gradient /= np.maximum(np.linalg.norm(gradient, axis=1)[:, None], 1e-30)
                match = (np.abs(f) <= tol[rows]) & (-np.sum(gradient*n[rows], axis=1) >= alignment)
                aliases = getattr(material, '_surface_aliases', ())
                canonical = aliases[sweep_index] if sweep_index < len(aliases) else sweep_index
                patch = np.zeros(len(rows), np.int32)
                if sweep.kind == 'flat': patch[n[rows, 2] > .9] = 1
                patch[(n[rows, 2] < -.9) & (np.abs(p[rows, 2]-sweep.z_hi) < tol[rows])] = 2
                for k in range(3): assign(rows, match & (patch == k), 7+3*canonical+k)
            continue
        if not isinstance(sweep, (FlatSweep, BallSweep)):
            continue
        rows = np.flatnonzero((tools > 0) & (tools == sweep.tool_id) & ~known)
        if not len(rows):
            continue
        lo, hi = sweep.world_aabb()
        inside = (np.all(p[rows] >= lo - tol[rows, None], axis=1) &
                  np.all(p[rows] <= hi + tol[rows, None], axis=1))
        rows = rows[inside]
        if not len(rows):
            continue
        q, normal, eps = p[rows], n[rows], tol[rows]
        _, nearest = capsule2d_parameter(q[:, :2], sweep.a_xy, sweep.b_xy)
        radial_vector = q[:, :2] - nearest
        radial_length = np.linalg.norm(radial_vector, axis=1)
        radial = radial_length - sweep.radius
        radial_normal = -radial_vector / np.maximum(radial_length[:, None], 1e-30)
        aliases = getattr(material, '_surface_aliases', ())
        canonical = aliases[sweep_index] if sweep_index < len(aliases) else sweep_index
        base_id = 7 + 3 * canonical
        side_lo = sweep.center_z if isinstance(sweep, BallSweep) else sweep.z_lo
        side = ((np.abs(radial) <= eps) & (radial_length > 1e-14) &
                (q[:, 2] >= side_lo - eps) & (q[:, 2] <= sweep.z_hi + eps) &
                (np.sum(radial_normal * normal[:, :2], axis=1) >= alignment))
        assign(rows, side, base_id)
        top = ((radial <= eps) & (np.abs(q[:, 2] - sweep.z_hi) <= eps) &
               (-normal[:, 2] >= alignment))
        assign(rows, top, base_id + 2)
        if isinstance(sweep, FlatSweep):
            bottom = ((radial <= eps) & (np.abs(q[:, 2] - sweep.z_lo) <= eps) &
                      (normal[:, 2] >= alignment))
            assign(rows, bottom, base_id + 1)
        else:
            vector = np.column_stack((radial_vector, q[:, 2] - sweep.center_z))
            length = np.linalg.norm(vector, axis=1)
            face_normal = -vector / np.maximum(length[:, None], 1e-30)
            rounded = ((np.abs(length - sweep.radius) <= eps) & (length > 1e-14) &
                       (q[:, 2] <= sweep.center_z + eps) &
                       (np.sum(face_normal * normal, axis=1) >= alignment))
            assign(rows, rounded, base_id)
    identity[ambiguous] = 0
    identity[known] = explicit[known]
    return identity.reshape(-1, 1)

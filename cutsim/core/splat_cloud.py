"""GaussianCutSim core / splat_cloud implementation."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
import numpy as np


@dataclass
class SplatCloud:
    positions: np.ndarray
    normals: np.ndarray
    support: np.ndarray
    tool_ids: np.ndarray
    refine_levels: np.ndarray
    seam_weight: np.ndarray | None = None
    inherited_clip: np.ndarray | None = None  # World halfplanes: 2 disjoint branches x 2 planes.

    sweep_domains: np.ndarray | None = None  # None, or tuple of world halfplane arrays (disjoint OR).

    def __post_init__(self):
        if self.sweep_domains is None:
            self.sweep_domains=np.empty(len(self.positions),dtype=object);self.sweep_domains[:]=None
        elif len(self.sweep_domains)!=len(self.positions):raise ValueError('sweep_domains length mismatch')
        else:self.sweep_domains=np.asarray(self.sweep_domains,dtype=object).copy()
        if self.inherited_clip is None:
            self.inherited_clip=np.zeros((len(self.positions),2,2,4),dtype=np.float64)
            self.inherited_clip[:,0,:,3]=1.  # unrestricted first region
            self.inherited_clip[:,1,:,3]=-1. # empty second region
        else:
            self.inherited_clip=np.asarray(self.inherited_clip,dtype=np.float64).reshape(-1,2,2,4).copy()
            if len(self.inherited_clip)!=len(self.positions) or not np.isfinite(self.inherited_clip).all():
                raise ValueError('Invalid inherited clip planes')
        if self.seam_weight is None:
            self.seam_weight = np.zeros(len(self.positions), dtype=np.float64)
        else:
            self.seam_weight = np.asarray(self.seam_weight, dtype=np.float64).reshape(-1)
            if len(self.seam_weight) != len(self.positions):
                raise ValueError("seam_weight length mismatch")

    @classmethod
    def empty(cls) -> "SplatCloud":
        return cls(np.empty((0, 3), np.float64), np.empty((0, 3), np.float64),
                   np.empty(0, np.float64), np.empty(0, np.int32),
                   np.empty(0, np.int16), np.empty(0, np.float64))

    @classmethod
    def _from_arrays(cls, arrays):
        """Internal arrays derived from validated clouds, without a second copy.

        Callers must own writable arrays or explicitly publish read-only views.
        Public construction still validates and owns clipping/domain buffers.
        """
        out = cls.__new__(cls)
        for field, value in zip(fields(cls), arrays):
            setattr(out, field.name, value)
        return out

    def append(self, other: "SplatCloud") -> None:
        if not len(other.positions):
            return
        if not len(self.positions):
            self.positions = other.positions.copy()
            self.normals = other.normals.copy()
            self.support = other.support.copy()
            self.tool_ids = other.tool_ids.copy()
            self.refine_levels = other.refine_levels.copy()
            self.seam_weight = other.seam_weight.copy()
            self.inherited_clip = other.inherited_clip.copy()
            self.sweep_domains = other.sweep_domains.copy()
            return
        self.positions = np.concatenate([self.positions, other.positions])
        self.normals = np.concatenate([self.normals, other.normals])
        self.support = np.concatenate([self.support, other.support])
        self.tool_ids = np.concatenate([self.tool_ids, other.tool_ids])
        self.refine_levels = np.concatenate([self.refine_levels, other.refine_levels])
        self.seam_weight = np.concatenate([self.seam_weight, other.seam_weight])
        self.inherited_clip = np.concatenate([self.inherited_clip, other.inherited_clip])
        self.sweep_domains = np.concatenate([self.sweep_domains, other.sweep_domains])

    def keep(self, mask: np.ndarray) -> None:
        self.positions = self.positions[mask]
        self.normals = self.normals[mask]
        self.support = self.support[mask]
        self.tool_ids = self.tool_ids[mask]
        self.refine_levels = self.refine_levels[mask]
        self.seam_weight = self.seam_weight[mask]
        self.inherited_clip = self.inherited_clip[mask]
        self.sweep_domains = self.sweep_domains[mask]


def face_samples(origin: np.ndarray, u: np.ndarray, v: np.ndarray,
                 nu: int, nv: int, du: float, dv: float,
                 normal: np.ndarray, support: float) -> SplatCloud:
    iu = (np.arange(nu) + 0.5) - nu * 0.5
    iv = (np.arange(nv) + 0.5) - nv * 0.5
    uu, vv = np.meshgrid(iu * du, iv * dv, indexing="ij")
    p = origin + uu.reshape(-1, 1) * u + vv.reshape(-1, 1) * v
    n = np.tile(normal, (len(p), 1))
    return SplatCloud(p, n, np.full(len(p), support), np.zeros(len(p), np.int32),
                      np.full(len(p), -1, np.int16))


def analytic_box_surface_splats(lo: np.ndarray, hi: np.ndarray, spacing: float) -> SplatCloud:
    """Initial stock 2DGS sampled directly on analytic box faces.

    This is surface sampling only; it does not create a 3-D material grid.
    """
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    sx, sy, sz = hi - lo
    nx = max(1, int(math.ceil(sx / spacing)))
    ny = max(1, int(math.ceil(sy / spacing)))
    nz = max(1, int(math.ceil(sz / spacing)))
    dx, dy, dz = sx / nx, sy / ny, sz / nz
    clouds = [
        face_samples(np.array([(x0+x1)/2, (y0+y1)/2, z1]), np.array([1.,0,0]), np.array([0,1.,0]), nx, ny, dx, dy, np.array([0,0,1.]), spacing, ),
        face_samples(np.array([(x0+x1)/2, (y0+y1)/2, z0]), np.array([1.,0,0]), np.array([0,1.,0]), nx, ny, dx, dy, np.array([0,0,-1.]), spacing, ),
        face_samples(np.array([x0, (y0+y1)/2, (z0+z1)/2]), np.array([0,1.,0]), np.array([0,0,1.]), ny, nz, dy, dz, np.array([-1.,0,0]), spacing, ),
        face_samples(np.array([x1, (y0+y1)/2, (z0+z1)/2]), np.array([0,1.,0]), np.array([0,0,1.]), ny, nz, dy, dz, np.array([1.,0,0]), spacing, ),
        face_samples(np.array([(x0+x1)/2, y0, (z0+z1)/2]), np.array([1.,0,0]), np.array([0,0,1.]), nx, nz, dx, dz, np.array([0,-1.,0]), spacing, ),
        face_samples(np.array([(x0+x1)/2, y1, (z0+z1)/2]), np.array([1.,0,0]), np.array([0,0,1.]), nx, nz, dx, dz, np.array([0,1.,0]), spacing, ),
    ]
    out = SplatCloud.empty()
    for c in clouds:
        out.append(c)
    return out


def _rt_copy_cloud(c: SplatCloud) -> SplatCloud:
    out=SplatCloud._from_arrays((c.positions.copy(), c.normals.copy(), c.support.copy(),
                   c.tool_ids.copy(), c.refine_levels.copy(), c.seam_weight.copy(),c.inherited_clip.copy(),c.sweep_domains.copy()))
    if hasattr(c,'_profile_attributes'): out._profile_attributes={k:v.copy() for k,v in c._profile_attributes.items()}
    if hasattr(c,'_local_ids'): out._local_ids=c._local_ids.copy()
    if hasattr(c,'_local_state'): out._local_state=c._local_state
    return out


def _rt_snapshot_cloud(c: SplatCloud) -> SplatCloud:
    """Publish a completed state without copying its numeric GS buffers.

    Cutting constructs new retained/added arrays; it never edits this state.
    Freeze the source too so accidental in-place edits fail instead of corrupting
    earlier block endpoints. Nested domain planes are immutable as well.
    """
    from cutsim.core.chunked_cloud import ChunkedCloud
    if isinstance(c,ChunkedCloud):
        from copy import copy
        return copy(c)
    arrays = []
    for field in fields(SplatCloud):
        value = getattr(c, field.name)
        value.flags.writeable = False
        view = value.view()
        view.flags.writeable = False
        arrays.append(view)
    for domain in c.sweep_domains:
        if domain is not None:
            for plane in domain:
                plane.flags.writeable = False
    out = SplatCloud._from_arrays(arrays)
    if hasattr(c, '_profile_attributes'): out._profile_attributes=c._profile_attributes
    if hasattr(c, '_local_ids'):
        c._local_ids.flags.writeable = False
        out._local_ids = c._local_ids.view()
    if hasattr(c, '_local_state'): out._local_state = c._local_state
    return out


def _rt_subset_cloud(c: SplatCloud, mask) -> SplatCloud:
    from cutsim.core.local_state import _local_take
    m=np.asarray(mask,dtype=bool).reshape(-1)
    return _local_take(c,m)


def _rt_concat_clouds(*clouds) -> SplatCloud:
    valid=[c for c in clouds if c is not None and len(c.positions)]
    if not valid:
        return SplatCloud.empty()
    out=SplatCloud._from_arrays((
        np.concatenate([c.positions for c in valid],axis=0),
        np.concatenate([c.normals for c in valid],axis=0),
        np.concatenate([c.support for c in valid],axis=0),
        np.concatenate([c.tool_ids for c in valid],axis=0),
        np.concatenate([c.refine_levels for c in valid],axis=0),
        np.concatenate([c.seam_weight for c in valid],axis=0),
        np.concatenate([c.inherited_clip for c in valid],axis=0),
        np.concatenate([c.sweep_domains for c in valid]),
    ))

    if all(hasattr(c,'_profile_attributes') for c in valid):
        out._profile_attributes={k:np.concatenate([c._profile_attributes[k] for c in valid]) for k in valid[0]._profile_attributes}
    if all(hasattr(c,'_local_ids') for c in valid):
        out._local_ids=np.concatenate([c._local_ids for c in valid])
    return out

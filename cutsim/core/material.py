"""GaussianCutSim core / material implementation."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import List, Tuple
import numpy as np
from cutsim.core.scipy_support import _HAVE_SCIPY_GRAPH, cKDTree
from cutsim.core.spatial_index import SweepSpatialIndex, _HistoryPrefix
from cutsim.core.sweep import BallSweep, FlatSweep, box_sdf


class AnalyticMaterial:
    """Material truth = stock box minus analytic tool sweeps.

    Analytic material truth is stored in a persistent spatial hierarchy.
    Bounds prune whole branches; playback prefixes share immutable tree roots.
    A query overlapping many old cuts can still visit many leaves.
    """
    def __init__(self, stock_lo: np.ndarray, stock_hi: np.ndarray):
        self.stock_lo = np.asarray(stock_lo, dtype=np.float64)
        self.stock_hi = np.asarray(stock_hi, dtype=np.float64)
        self.sweeps: List[object] = []
        self._sweep_index=SweepSpatialIndex()
        self._grid_cache = None
        self.last_sweep_sdf_requested=0
        self.last_sweep_sdf_evaluated=0
        self.last_sweep_index_nodes=0
        self.last_sweep_index_pairs=0

    def add_sweep(self, sweep) -> None:
        # Own immutable geometry: persistent roots and append-time cache tokens
        # must not become stale when a caller reuses or edits input arrays.
        if type(sweep) in (FlatSweep,BallSweep):
            a=np.array(sweep.a_xy,copy=True); b=np.array(sweep.b_xy,copy=True)
            a.flags.writeable=False; b.flags.writeable=False
            sweep=replace(sweep,a_xy=a,b_xy=b)
        self.sweeps.append(sweep)
        self._sweep_index.add(sweep)
        if self._grid_cache is not None:
            self._grid_cache.invalidate()

    def grid_values(self, frame, axes, world_nodes):
        if self._grid_cache is None:
            return self.value(world_nodes)
        return self._grid_cache.evaluate(self,frame,axes,world_nodes)

    def grid_values_lattice(self, frame, anchor, ijk, h, world_nodes, raw_requested=None):
        if self._grid_cache is None:
            return self.value(world_nodes)
        return self._grid_cache.evaluate_lattice(
            self,frame,anchor,ijk,h,world_nodes,raw_requested=raw_requested)

    def snapshot(self, upto=None):
        """Share the immutable spatial root and history prefix for playback."""
        count=len(self.sweeps) if upto is None else max(0,min(int(upto),len(self.sweeps)))
        out=AnalyticMaterial(self.stock_lo.copy(),self.stock_hi.copy())
        out.sweeps=_HistoryPrefix(self.sweeps,count)
        out._sweep_index=self._sweep_index.prefix(count)
        if hasattr(self,'_gs_density'):
            out._gs_density=self._gs_density
        if hasattr(self,'_boundary_work'):
            out._boundary_work=self._boundary_work
        if hasattr(self,'_surface_aliases'):
            out._surface_aliases=tuple(self._surface_aliases[:count])
        if hasattr(self,'_gs_surface_provenance'):
            # Stable IDs are never reused. This append-only map can be shared
            # by old playback prefixes without changing any existing entry.
            out._gs_surface_provenance=self._gs_surface_provenance
        return out

    def query_sweep_ids(self,lo,hi,*,original_boxes=True):
        return self._sweep_index.query_aabb(np.asarray(lo),np.asarray(hi),original_boxes=original_boxes)

    def value(self, points: np.ndarray, upto: int | None = None) -> np.ndarray:
        p=np.asarray(points,dtype=np.float64).reshape(-1,3)
        stock=box_sdf(p,self.stock_lo,self.stock_hi)
        limit=len(self.sweeps) if upto is None else max(0,min(int(upto),len(self.sweeps)))
        if (limit and not self._sweep_index.supported[limit-1] and
                any(getattr(self.sweeps[i], 'euclidean_distance', True) is False for i in range(limit))):
            # General XYZ quadric unions supply a Lipschitz field. Euclidean
            # AABB distance is not a lower bound for that field outside the box.
            # Preserve the exact CSG field rather than applying distance pruning.
            value = stock.copy()
            for i in range(limit):
                value = np.maximum(value, -self.sweeps[i].sdf(p))
            self.last_sweep_sdf_requested += len(p)*limit
            self.last_sweep_sdf_evaluated += len(p)*limit
            return value
        value,evaluated=self._sweep_index.material_value(p,stock,self.sweeps,limit)
        self.last_sweep_sdf_requested+=len(p)*limit
        self.last_sweep_sdf_evaluated+=evaluated
        self.last_sweep_index_nodes+=self._sweep_index.last_value_nodes
        self.last_sweep_index_pairs+=self._sweep_index.last_value_pairs
        return value

    def gradient(self, points: np.ndarray, eps: float) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        n = np.empty_like(p)
        for axis in range(3):
            d = np.zeros(3, dtype=np.float64)
            d[axis] = eps
            n[:, axis] = (self.value(p + d) - self.value(p - d)) / (2.0 * eps)
        norm = np.linalg.norm(n, axis=1)
        bad = norm < 1e-14
        n[~bad] /= norm[~bad, None]
        n[bad] = np.array([0.0, 0.0, 1.0])
        return n


class SplatSurfaceField:
    """Local signed-distance approximation reconstructed from the current 2DGS surface.

    This field deliberately targets only the narrow local-grid band used during
    incremental cutting.  It is NOT intended to be a globally exact SDF.

    The nearest splat supplies a reference normal. Nearby splats whose normals are
    broadly consistent with that reference are blended with an MLS-like Gaussian
    weight in tangent space. This avoids averaging unrelated faces across sharp
    corners while still reducing nearest-splat discontinuities on smooth regions.
    """
    def __init__(self, cloud, k: int = 12, support_reach: float = 2.75,
                 normal_cos: float = 0.50):
        self.positions=np.asarray(cloud.positions,dtype=np.float64).reshape(-1,3)
        self.normals=np.asarray(cloud.normals,dtype=np.float64).reshape(-1,3)
        self.support=np.asarray(cloud.support,dtype=np.float64).reshape(-1)
        self.tool_ids=np.asarray(cloud.tool_ids,dtype=np.int32).reshape(-1)
        self.k=max(1,int(k))
        self.support_reach=max(float(support_reach),1.0)
        self.normal_cos=float(np.clip(normal_cos,-1.0,1.0))
        self.query_points=0
        self.query_neighbors=0
        self.tree=(cKDTree(self.positions) if (_HAVE_SCIPY_GRAPH and len(self.positions)) else None)

    def _query_ids(self, points):
        p=np.asarray(points,dtype=np.float64).reshape(-1,3)
        if not len(p):
            return np.empty((0,0),dtype=np.int64),np.empty((0,0),dtype=np.float64)
        if not len(self.positions):
            return np.empty((len(p),0),dtype=np.int64),np.empty((len(p),0),dtype=np.float64)
        kk=min(self.k,len(self.positions))
        if self.tree is not None:
            dist,ids=self.tree.query(p,k=kk)
            if kk==1:
                dist=dist[:,None]; ids=ids[:,None]
            return np.asarray(ids,dtype=np.int64),np.asarray(dist,dtype=np.float64)
        # SciPy-free fallback: chunked brute-force nearest neighbours.
        ids=np.empty((len(p),kk),dtype=np.int64)
        dist=np.empty((len(p),kk),dtype=np.float64)
        chunk=256
        for start in range(0,len(p),chunk):
            q=p[start:start+chunk]
            d2=np.sum((q[:,None,:]-self.positions[None,:,:])**2,axis=2)
            part=np.argpartition(d2,kk-1,axis=1)[:,:kk]
            dd=np.take_along_axis(d2,part,axis=1)
            order=np.argsort(dd,axis=1)
            part=np.take_along_axis(part,order,axis=1)
            dd=np.take_along_axis(dd,order,axis=1)
            ids[start:start+len(q)]=part
            dist[start:start+len(q)]=np.sqrt(np.maximum(dd,0.0))
        return ids,dist

    def value(self, points):
        p=np.asarray(points,dtype=np.float64).reshape(-1,3)
        if not len(p):
            return np.empty(0,dtype=np.float64)
        if not len(self.positions):
            return np.full(len(p),np.inf,dtype=np.float64)
        ids,center_dist=self._query_ids(p)
        self.query_points+=len(p)
        self.query_neighbors+=int(ids.size)
        pp=self.positions[ids]
        nn=self.normals[ids]
        ss=np.maximum(self.support[ids],1e-8)
        r=p[:,None,:]-pp
        nd=np.sum(r*nn,axis=2)
        tangent=r-nd[:,:,None]*nn
        td2=np.sum(tangent*tangent,axis=2)

        # Use the nearest centre's normal as a local face-family reference.
        ref_n=nn[:,0,:]
        align=np.sum(nn*ref_n[:,None,:],axis=2)
        same_face=align>=self.normal_cos
        within=td2 <= (self.support_reach*ss)**2

        # Tangent-space Gaussian matches the 2DGS surface interpretation; a mild
        # centre-distance term suppresses far neighbours with very large support.
        wt=np.exp(-0.5*td2/(ss*ss))
        wc=np.exp(-0.5*(center_dist/np.maximum(1.5*ss,1e-8))**2)
        w=wt*wc*same_face*within
        den=np.sum(w,axis=1)
        sdf=np.sum(w*nd,axis=1)/np.maximum(den,1e-30)
        # If all filtered neighbours were rejected, keep the nearest oriented plane.
        bad=den<=1e-20
        if np.any(bad):
            sdf[bad]=nd[bad,0]
        return sdf

    def nearest_tool_ids(self, points):
        p=np.asarray(points,dtype=np.float64).reshape(-1,3)
        if not len(p) or not len(self.positions):
            return np.zeros(len(p),dtype=np.int32)
        ids,_=self._query_ids(p)
        return self.tool_ids[ids[:,0]]


class SplatSurfaceCutMaterial:
    """Post-cut local field = current 2DGS surface minus the newest tool sweep.

    This provides the minimal material interface consumed by the existing local-grid
    extractor, root bisection, and fallback normal code while avoiding replay of all
    historical sweeps for local-grid SDF samples.
    """
    def __init__(self, stock_lo, stock_hi, surface_cloud, active_sweep,
                 k: int = 12, support_reach: float = 2.75, normal_cos: float = 0.50):
        self.stock_lo=np.asarray(stock_lo,dtype=np.float64)
        self.stock_hi=np.asarray(stock_hi,dtype=np.float64)
        self.surface=SplatSurfaceField(surface_cloud,k=k,support_reach=support_reach,
                                       normal_cos=normal_cos)
        self.active_sweep=active_sweep
        # split_face_samples() enumerates analytic incident faces from this list.
        # Only the newest cutter is analytic here; older material comes from 2DGS.
        self.sweeps=[active_sweep]
        self._grid_cache=None

    def old_value(self, points):
        return self.surface.value(points)

    def value(self, points, upto=None):
        p=np.asarray(points,dtype=np.float64).reshape(-1,3)
        old=self.surface.value(p)
        if self.active_sweep is None:
            return old
        return np.maximum(old,-self.active_sweep.sdf(p))

    def grid_values(self, frame, axes, world_nodes):
        if self._grid_cache is None:
            return self.value(world_nodes)
        return self._grid_cache.evaluate(self,frame,axes,world_nodes)

    def grid_values_lattice(self, frame, anchor, ijk, h, world_nodes, raw_requested=None):
        if self._grid_cache is None:
            return self.value(world_nodes)
        return self._grid_cache.evaluate_lattice(
            self,frame,anchor,ijk,h,world_nodes,raw_requested=raw_requested)

    def gradient(self, points, eps):
        p=np.asarray(points,dtype=np.float64).reshape(-1,3)
        g=np.empty_like(p)
        ee=max(float(eps),1e-6)
        for axis in range(3):
            d=np.zeros(3,dtype=np.float64); d[axis]=ee
            g[:,axis]=(self.value(p+d)-self.value(p-d))/(2.0*ee)
        norm=np.linalg.norm(g,axis=1)
        bad=norm<1e-14
        g[~bad]/=norm[~bad,None]
        if np.any(bad):
            g[bad]=np.array([0.0,0.0,1.0])
        return g

    def surface_tool_ids(self, points):
        return self.surface.nearest_tool_ids(points)

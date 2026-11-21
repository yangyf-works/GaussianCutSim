"""GaussianCutSim cutting / spatial_index implementation."""
from __future__ import annotations

import numpy as np
from cutsim.core.sweep import BallSweep
from cutsim.cutting.footprint import _CUT_FOOTPRINT_SETTINGS, _cut_footprint_radius
from cutsim.geometry.frames import tangent_frames


class _CutReferenceIndex:
    def __init__(self,index,before,removed,split):
        self.index=index; self.before=before; self.kept=np.flatnonzero(~removed)
        self.removed=removed; self.split=split
        self.extra=_RealtimeSplatIndex(split,index.cell_size) if len(split.positions) else None

    def query(self,lo,hi,positions):
        old=self.index.query(lo,hi,self.before.positions)
        old=old[~self.removed[old]]
        rows=np.searchsorted(self.kept,old)
        if self.extra is not None:
            rows=np.r_[rows,len(self.kept)+self.extra.query(lo,hi,self.split.positions)]
        return rows


def _local_nearest_within(index,points,queries,radii,k):
    ids=np.zeros((len(queries),k),dtype=np.int64)
    dist=np.full((len(queries),k),np.inf)
    for j,(p,r) in enumerate(zip(queries,radii)):
        rows=index.query(p-r-1e-10,p+r+1e-10,points)
        if not len(rows): continue
        d2=np.sum((points[rows]-p)**2,axis=1)
        order=np.lexsort((rows,d2))[:k]; rows=rows[order]; d2=d2[order]
        ids[j,:len(rows)]=rows; dist[j,:len(rows)]=np.sqrt(np.maximum(d2,0.))
    return dist,ids


class _NominalFootprintIndex:
    """Incremental broad phase conservatively enclosing the maximum display disks.

    Membership covers each cell's actual AABB, including the classifier tolerance.
    Queries visit only sweep-overlapped buckets; no global support maximum or scan.
    IDs are supplied by the persistent centre index and survive dense compaction.
    """
    def __init__(self,cell_size):
        self.cell_size=float(cell_size)
        self.buckets={}; self.entries={}
        self.bounds_lo=None; self.bounds_hi=None
        self.last_query_cells=0; self.last_query_points=0

    def _keys(self,lower,upper):
        for x in range(int(lower[0]),int(upper[0])+1):
            for y in range(int(lower[1]),int(upper[1])+1):
                for z in range(int(lower[2]),int(upper[2])+1):
                    yield (x,y,z)

    def add(self,ids,positions,normals,support):
        if not len(ids): return
        u,v=tangent_frames(np.asarray(normals,dtype=np.float64))
        support=np.asarray(support,dtype=np.float64)
        # Conservative seam maximum; actual seam radius is used in the narrow phase.
        radius=3.*np.asarray(support)*1.0*max(1.,_CUT_FOOTPRINT_SETTINGS.resolution_seam_scale)
        extent=radius[:,None]*np.sqrt(u*u+v*v)
        extent+=np.maximum(.015*support,1e-7)[:,None]
        lo=np.asarray(positions)-extent; hi=np.asarray(positions)+extent
        lower=np.floor(lo/self.cell_size).astype(np.int64)
        upper=np.floor(hi/self.cell_size).astype(np.int64)
        self.bounds_lo=lower.min(axis=0) if self.bounds_lo is None else np.minimum(self.bounds_lo,lower.min(axis=0))
        self.bounds_hi=upper.max(axis=0) if self.bounds_hi is None else np.maximum(self.bounds_hi,upper.max(axis=0))
        for i,stable in enumerate(ids):
            stable=int(stable)
            keys=tuple(self._keys(lower[i],upper[i]))
            self.entries[stable]=(lo[i],hi[i],keys)
            for key in keys: self.buckets.setdefault(key,set()).add(stable)

    def remove(self,ids):
        for stable in ids:
            stable=int(stable)
            for key in self.entries.pop(stable)[2]:
                bucket=self.buckets[key]; bucket.remove(stable)
                if not bucket: del self.buckets[key]

    def query(self,lo,hi,ordered_ids):
        self.last_query_cells=0; self.last_query_points=0
        if not self.entries: return np.empty(0,dtype=np.int64)
        lower=np.maximum(np.floor(lo/self.cell_size).astype(np.int64),self.bounds_lo)
        upper=np.minimum(np.floor(hi/self.cell_size).astype(np.int64),self.bounds_hi)
        found=set()
        for key in self._keys(lower,upper):
            self.last_query_cells+=1
            found.update(self.buckets.get(key,()))
        self.last_query_points=len(found)
        stable=sorted(i for i in found if np.all(self.entries[i][0]<=hi) and np.all(self.entries[i][1]>=lo))
        return np.searchsorted(ordered_ids,np.asarray(stable,dtype=np.int64))

    def query_line(self,origin,direction,start,end,ordered_ids):
        """Closed segment/finite-footprint AABB query by grid-cell traversal.

        Visit every touched cell, including simultaneous boundary crossings and
        lines lying on cell planes. There is no neighbour/count budget. A vector
        slab test removes bucket-only candidates before the GS polygon test.
        """
        self.last_query_cells=0;self.last_query_points=0
        if not self.entries or start>end:return np.empty(0,np.int64)
        origin=np.asarray(origin,float);direction=np.asarray(direction,float)
        if not np.isfinite([start,end]).all() or not np.any(direction):
            raise ValueError('A finite nonzero line segment is required')
        point=origin+start*direction
        cell=np.floor(point/self.cell_size).astype(np.int64)
        step=np.sign(direction).astype(np.int64)
        crossing=np.full(3,np.inf);increment=np.full(3,np.inf)
        for axis in np.flatnonzero(step):
            boundary=(cell[axis]+(step[axis]>0))*self.cell_size
            crossing[axis]=(boundary-origin[axis])/direction[axis]
            increment[axis]=self.cell_size/abs(direction[axis])
        # Include the negative neighbour when a parallel coordinate is exactly
        # on a grid plane. The narrow slab test still decides actual contact.
        parallel_planes=[axis for axis in range(3) if not step[axis] and
                         abs(point[axis]/self.cell_size-round(point[axis]/self.cell_size))<1e-12]
        offsets=[np.zeros(3,np.int64)]
        for axis in parallel_planes:
            extra=[]
            for offset in offsets:
                q=offset.copy();q[axis]-=1;extra.append(q)
            offsets.extend(extra)
        visited=set();found=set()
        def visit(current):
            for offset in offsets:
                key=tuple(map(int,current+offset))
                if key not in visited:
                    visited.add(key);found.update(self.buckets.get(key,()))
        visit(cell)
        while True:
            parameter=float(crossing.min())
            error=64*np.finfo(float).eps*max(1.,abs(parameter),abs(end))
            if not np.isfinite(parameter) or parameter>end+error:break
            axes=np.flatnonzero(crossing<=parameter+error)
            touched=[cell.copy()]
            for axis in axes:
                extra=[]
                for current in touched:
                    q=current.copy();q[axis]+=step[axis];extra.append(q)
                touched.extend(extra)
            for current in touched:visit(current)
            cell[axes]+=step[axes];crossing[axes]+=increment[axes]
        self.last_query_cells=len(visited);self.last_query_points=len(found)
        if not found:return np.empty(0,np.int64)
        stable=np.array(sorted(found),np.int64)
        lower=np.array([self.entries[int(i)][0] for i in stable])
        upper=np.array([self.entries[int(i)][1] for i in stable])
        lo=np.full(len(stable),start);hi=np.full(len(stable),end)
        for axis in range(3):
            if direction[axis]==0.:
                lo[(origin[axis]<lower[:,axis])|(origin[axis]>upper[:,axis])]=np.inf
            else:
                a=(lower[:,axis]-origin[axis])/direction[axis]
                b=(upper[:,axis]-origin[axis])/direction[axis]
                lo=np.maximum(lo,np.minimum(a,b));hi=np.minimum(hi,np.maximum(a,b))
        return np.searchsorted(ordered_ids,stable[lo<=hi])


def _rt_footprint_candidates(index,cloud,sweep):
    """Conservative local admission for the 2.0-sigma circular classifier."""
    lo,hi=sweep.world_aabb()
    if type(sweep) is BallSweep:
        hi=hi.copy(); hi[2]=max(hi[2],sweep.center_z+sweep.radius)
    pad=64*np.finfo(float).eps*max(float(np.max(np.abs(lo))),float(np.max(np.abs(hi))),1.)
    rows=index.footprints.query(lo-pad,hi+pad,index.ids)
    support=cloud.support[rows]
    # Flat/Ball SDFs are 1-Lipschitz; all disk probes lie within this radius.
    reach=_cut_footprint_radius(support,cloud.seam_weight[rows])+np.maximum(.015*support,1e-7)
    selected=rows[np.asarray(sweep.sdf(cloud.positions[rows]))<=reach+pad] if len(rows) else rows
    mask=np.zeros(len(cloud.positions),dtype=bool); mask[selected]=True
    return mask,rows


def _two_edge_centres(arr,exclude_multi=False):
    """Parents with two selected boundaries; render-only copies are not counted.

    Top-2 parents are included. When both overlays are enabled, their existing
    magenta marker takes priority so coincident debug points do not z-fight.
    """
    selected=np.asarray(arr['clip_edge_count'])==2
    if exclude_multi: selected&=np.asarray(arr['clip_candidate_count'])<3
    return arr['positions'][selected]


def _multi_edge_centres(arr):
    """Actual parent centres with >=3 geometric ridge candidates BEFORE selection."""
    return arr['positions'][np.asarray(arr['clip_candidate_count'])>=3]


class _RealtimeSplatIndex:
    """Persistent fixed-cell membership for exact cutting broad-phase queries.

    Stable IDs are independent of dense array rows. Only removed/new points
    update buckets; searchsorted maps queried IDs to current array rows. Dense
    ID compaction is still linear, as is the simulator's cloud serialization.
    """
    def __init__(self,cloud,cell_size,track_footprints=False):
        self.cell_size=max(float(cell_size),1e-6)
        self.footprints=_NominalFootprintIndex(self.cell_size) if track_footprints else None
        self.buckets={}; self.id_cells={}; self.ids=np.empty(0,dtype=np.int64)
        self.next_id=0; self.bounds_lo=None; self.bounds_hi=None
        self.dirty_cells=set(); self.last_query_cells=0; self.last_query_points=0
        self._append(cloud.positions,cloud.normals,cloud.support)

    def _append(self,positions,normals=None,support=None):
        count=len(positions)
        ids=np.arange(self.next_id,self.next_id+count,dtype=np.int64)
        self.next_id+=count
        if self.footprints is not None:
            self.footprints.add(ids,positions,normals,support)
        if count:
            keys=np.floor(np.asarray(positions)/self.cell_size).astype(np.int64)
            lo=keys.min(axis=0); hi=keys.max(axis=0)
            self.bounds_lo=lo if self.bounds_lo is None else np.minimum(self.bounds_lo,lo)
            self.bounds_hi=hi if self.bounds_hi is None else np.maximum(self.bounds_hi,hi)
            for stable_id,key in zip(ids,keys):
                key=tuple(map(int,key)); stable_id=int(stable_id)
                self.buckets.setdefault(key,set()).add(stable_id)
                self.id_cells[stable_id]=key; self.dirty_cells.add(key)
            self.ids=np.concatenate((self.ids,ids))

    def update(self,removed,new_positions,new_normals=None,new_support=None):
        if self.footprints is not None:
            if len(new_positions) and (new_normals is None or new_support is None):
                raise ValueError('Footprint index updates require normals and support')
            self.footprints.remove(self.ids[removed])
        self.dirty_cells=set()
        for stable_id in self.ids[removed]:
            stable_id=int(stable_id); key=self.id_cells.pop(stable_id)
            bucket=self.buckets[key]; bucket.remove(stable_id)
            if not bucket: del self.buckets[key]
            self.dirty_cells.add(key)
        self.ids=self.ids[~removed]
        self._append(new_positions,new_normals,new_support)

    def query(self,lo,hi,positions):
        self.last_query_cells=0; self.last_query_points=0
        if not len(self.ids): return np.empty(0,dtype=np.int64)
        lower=np.maximum(np.floor(np.asarray(lo)/self.cell_size).astype(np.int64),self.bounds_lo)
        upper=np.minimum(np.floor(np.asarray(hi)/self.cell_size).astype(np.int64),self.bounds_hi)
        found=[]
        for x in range(int(lower[0]),int(upper[0])+1):
            for y in range(int(lower[1]),int(upper[1])+1):
                for z in range(int(lower[2]),int(upper[2])+1):
                    self.last_query_cells+=1
                    found.extend(self.buckets.get((x,y,z),()))
        if not found: return np.empty(0,dtype=np.int64)
        stable=np.sort(np.asarray(found,dtype=np.int64))
        rows=np.searchsorted(self.ids,stable)
        p=positions[rows]; keep=np.all(p>=lo,axis=1)&np.all(p<=hi,axis=1)
        self.last_query_points=len(rows)
        return rows[keep]

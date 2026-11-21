"""GaussianCutSim core / local_state implementation."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import numpy as np
from cutsim.core.splat_cloud import SplatCloud
from cutsim.core.sweep import BallSweep


_LOCAL_FIELDS=('positions','normals','support','tool_ids','refine_levels','seam_weight','inherited_clip','sweep_domains')


_LOCAL_POSITION_QUANT=1e-6


def _local_take(cloud,rows):
    from cutsim.core.chunked_cloud import ChunkedCloud
    if isinstance(cloud,ChunkedCloud):return cloud.take(rows)
    arrays=[]
    for k in _LOCAL_FIELDS:
        source=np.asarray(getattr(cloud,k)); selected=source[rows]
        # Advanced indexing already owns a copy; slices still need ownership.
        arrays.append(selected.copy() if not selected.flags.writeable or np.shares_memory(source,selected) else selected)
    out=SplatCloud._from_arrays(arrays)
    if hasattr(cloud,'_profile_attributes'):
        out._profile_attributes={k:v[rows].copy() for k,v in cloud._profile_attributes.items()}
    if hasattr(cloud,'_local_ids'):
        out._local_ids=np.asarray(cloud._local_ids)[rows].copy()
    return out


class _LocalCloudState:
    """Geometry event; changing playback branches requires only intervening deltas."""
    def __init__(self,parent,removed,added,sweep=None):
        self.parent=parent; self.removed=removed; self.added=added; self.sweep=sweep
        self.depth=0 if parent is None else parent.depth+1
        self.root=self if parent is None else parent.root


def _local_initial_state(cloud,cell_size=3.):
    if not hasattr(cloud,'_local_ids'): cloud._local_ids=np.arange(len(cloud.positions),dtype=np.int64)
    state=_LocalCloudState(None,SplatCloud.empty(),cloud)
    state.cell_size=max(float(cell_size),1e-5)
    cloud._local_state=state
    return state


def _local_partial_sweep(sweep,u):
    if u<=1e-12: return None
    if u>=1.-1e-12: return sweep
    a=np.asarray(sweep.a_xy); b=a+(np.asarray(sweep.b_xy)-a)*u
    return replace(sweep,a_xy=a.copy(),b_xy=b)


class _LocalPointStore:
    """Stable ID -> geometry plus persistent centre buckets; edits touch local buckets."""
    def __init__(self,cell):
        self.cell=float(cell); self.points={}; self.buckets={}; self.groups={}
        self.max_h=0.; self.visited_points=0
        self.lower=None; self.upper=None
        self.bucket_arrays={}
        self._slots={}; self._free_slots=[]; self._next_slot=0
        self.geometry=np.empty(0,dtype=[('positions','f8',(3,)),('normals','f8',(3,)),
            ('support','f8'),('tool_ids','i4'),('refine_levels','i4'),('seam_weight','f8'),('inherited_clip','f8',(2,2,4)),('sweep_domains',object)])

    def slots(self, ids):
        return np.fromiter((self._slots[int(sid)] for sid in ids), dtype=np.int64)

    def key(self,p): return tuple(np.floor(p/self.cell).astype(np.int64))
    def group_key(self,p): return tuple(np.rint(p/_LOCAL_POSITION_QUANT).astype(np.int64))

    def add(self,cloud):
        stable=np.asarray(getattr(cloud,'_local_ids',()),dtype=np.int64)
        if len(stable):
            for sid in stable:
                sid=int(sid)
                if sid not in self._slots:
                    if self._free_slots: slot=self._free_slots.pop()
                    else: slot=self._next_slot; self._next_slot+=1
                    self._slots[sid]=slot
            slots=self.slots(stable)
            required=self._next_slot
            if required>len(self.geometry):
                grown=np.empty(max(required,2*len(self.geometry),1024),dtype=self.geometry.dtype)
                grown[:len(self.geometry)]=self.geometry; self.geometry=grown
            for name in _LOCAL_FIELDS: self.geometry[name][slots]=getattr(cloud,name)
        if len(cloud.positions):
            lo=cloud.positions.min(axis=0); hi=cloud.positions.max(axis=0)
            self.lower=lo if self.lower is None else np.minimum(self.lower,lo)
            self.upper=hi if self.upper is None else np.maximum(self.upper,hi)
        # Identical elementwise arithmetic, batched before the Python ID loop.
        # No quantization rule or spatial bucket size changes.
        cell_keys=np.floor(cloud.positions/self.cell).astype(np.int64)
        point_keys=np.rint(cloud.positions/_LOCAL_POSITION_QUANT).astype(np.int64)
        for row,sid in enumerate(getattr(cloud,'_local_ids',())):
            sid=int(sid)
            values=(cloud.positions[row].copy(),cloud.normals[row].copy(),float(cloud.support[row]),
                    int(cloud.tool_ids[row]),int(cloud.refine_levels[row]),float(cloud.seam_weight[row]))
            self.points[sid]=values; self.max_h=max(self.max_h,float(values[2]))
            key=tuple(cell_keys[row]); self.buckets.setdefault(key,set()).add(sid)
            self.bucket_arrays.pop(key,None)
            self.groups.setdefault(tuple(point_keys[row]),set()).add(sid)

    def remove(self,ids):
        for sid in ids:
            sid=int(sid); row=self.points.pop(sid)
            slot=self._slots.pop(sid)
            self.geometry['sweep_domains'][slot]=None
            self._free_slots.append(slot)
            self.bucket_arrays.pop(self.key(row[0]),None)
            for mapping,key in ((self.buckets,self.key(row[0])),(self.groups,self.group_key(row[0]))):
                mapping[key].remove(sid)
                if not mapping[key]: del mapping[key]

    def box(self,lo,hi):
        if self.lower is None: return np.empty(0,dtype=np.int64)
        lo=np.maximum(lo,self.lower); hi=np.minimum(hi,self.upper)
        if np.any(lo>hi): return np.empty(0,dtype=np.int64)
        a=np.floor(lo/self.cell).astype(np.int64); b=np.floor(hi/self.cell).astype(np.int64)
        found=[]
        for x in range(int(a[0]),int(b[0])+1):
            for y in range(int(a[1]),int(b[1])+1):
                for z in range(int(a[2]),int(b[2])+1):
                    key=(x,y,z); bucket=self.buckets.get(key)
                    if not bucket: continue
                    cached=self.bucket_arrays.get(key)
                    if cached is None:
                        ids=np.array(sorted(bucket),dtype=np.int64)
                        cached=(ids,self.geometry['positions'][self.slots(ids)]); self.bucket_arrays[key]=cached
                    ids,p=cached; self.visited_points+=len(ids)
                    found.append(cached)
        if not found: return np.empty(0,dtype=np.int64)
        # Apply the identical inclusive box predicate once, rather than launch
        # tiny NumPy reductions separately for every spatial bucket.
        ids=np.concatenate([part[0] for part in found])
        positions=np.concatenate([part[1] for part in found])
        hit=np.all(positions>=lo,axis=1)&np.all(positions<=hi,axis=1)
        return np.sort(ids[hit])

    def cloud(self,ids):
        ids=np.asarray(ids,dtype=np.int64)
        if not len(ids):
            out=SplatCloud.empty()
        else:
            slots=self.slots(ids)
            out=SplatCloud(*(self.geometry[name][slots] for name in _LOCAL_FIELDS))
        out._local_ids=ids.copy()
        # Physical coincident parents form independent preparation units. Children
        # of unrelated parents cannot introduce a batch-dependent sharp override.
        out._local_groups=np.rint(out.positions/_LOCAL_POSITION_QUANT).astype(np.int64)
        return out

    def neighbours(self,seeds,settings):
        result=set()
        reach=(3.000001 if settings.native_two_clip else 2.5)*float(settings.sigma_scale)*max(1.,float(settings.resolution_seam_scale))
        # This is ONE hop from the supplied seeds, never recursive flood filling.
        groups={}
        for sid in seeds:
            if int(sid) not in self.points: continue
            groups.setdefault(self.key(self.points[int(sid)][0]),[]).append(int(sid))
        for group in groups.values():
            for start in range(0,len(group),128):
                chosen=group[start:start+128]
                p=self.geometry['positions'][self.slots(chosen)]; h=self.geometry['support'][self.slots(chosen)]
                radius=reach*(float(h.max())+self.max_h)+1e-6
                ids=self.box(p.min(axis=0)-radius-1e-10,p.max(axis=0)+radius+1e-10)
                if not len(ids): continue
                q=self.geometry['positions'][self.slots(ids)]; hs=self.geometry['support'][self.slots(ids)]
                d2=np.sum((q[None,:,:]-p[:,None,:])**2,axis=2)
                hit=np.any(d2<=(reach*(h[:,None]+hs[None,:])+1e-6)**2,axis=0)
                result.update(map(int,ids[hit]))
        return result

    def material_dependents(self,sweep,settings):
        if sweep is None or not self.points: return set()
        lo,hi=sweep.world_aabb(); hi=hi.copy()
        if type(sweep) is BallSweep: hi[2]=max(hi[2],sweep.center_z+sweep.radius)
        # Analytic bound derived from candidate probes, displaced split children,
        # continuation probes and sampled fitting disks, not a fixed cell halo.
        factor=max(3.*settings.sigma_scale+.15,
                   (2.5+abs(settings.half_double_edge_offset_scale))*settings.sigma_scale+.12)
        if settings.native_two_clip:
            factor=3.*settings.sigma_scale*max(1.,settings.resolution_seam_scale)+.15
        r=factor*self.max_h+4e-6
        ids=self.box(lo-r,hi+r)
        if not len(ids): return set()
        p=self.geometry['positions'][self.slots(ids)]
        radii=factor*self.geometry['support'][self.slots(ids)]+4e-6
        hit=np.all(p+radii[:,None]>=lo,axis=1)&np.all(p-radii[:,None]<=hi,axis=1)
        return set(map(int,ids[hit]))

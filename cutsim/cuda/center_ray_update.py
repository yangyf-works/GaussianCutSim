"""Analytic tool-surface generation without Dexel/old-GS hit associations.

Material evidence comes from currently supported finite analytic faces.
Old-surface cutting and geometric replacement are separate downstream stages.
"""
from copy import copy
from time import perf_counter
import numpy as np


def center_grid(grid,sweep,box,*,xp=np,margin=None):
    """Prepare finite rays while preserving their tool surface samples.

    Flat tools keep axial cap rays and horizontal side rays. Ball tools
    move cap-ray origins onto the tool axis and retain the cap normals.
    """
    result=copy(grid);o=xp.asarray(grid.origins);d=xp.asarray(grid.directions)
    distance=xp.asarray(grid.surface_distance);direction=d
    if sweep.kind == 'ball':
        o=o.copy();points=o+distance[:,None]*d
        charts=grid.charts
        cap=xp.isin(charts,xp.asarray([2,3,6])) if xp is not np else np.isin(charts,['lower_cap','upper_cap','strip_cap'])
        a=np.asarray(sweep.a_xyz if hasattr(sweep,'a_xyz') else [*sweep.a_xy,sweep.z_lo])
        b=np.asarray(sweep.b_xyz if hasattr(sweep,'b_xyz') else [*sweep.b_xy,sweep.z_lo])
        axis_direction=np.asarray(sweep.basis[:,2] if hasattr(sweep,'local') else [0.,0.,1.])
        feed=b-a;feed=feed-axis_direction*(feed@axis_direction)
        length=float(feed@feed)
        along=xp.clip(((points-xp.asarray(a))*xp.asarray(feed)).sum(axis=1)/max(length,1e-30),0,1)
        on_axis=xp.asarray(a)+along[:,None]*xp.asarray(feed)
        projected=on_axis+((o-on_axis)*xp.asarray(axis_direction)).sum(axis=1)[:,None]*xp.asarray(axis_direction)
        o=xp.where(cap[:,None],projected,o)
        delta=points-o;distance=xp.linalg.norm(delta,axis=1)
        direction=delta/xp.maximum(distance[:,None],1e-30)
    extra=xp.asarray(grid.pitches) if margin is None else xp.full(len(o),margin)
    low=xp.zeros(len(o));high=distance+extra
    valid=xp.ones(len(o),dtype=bool)
    for k in range(3):
        parallel=xp.abs(direction[:,k])<1e-15
        denom=xp.where(parallel,1.,direction[:,k])
        u=(box[0][k]-o[:,k])/denom;v=(box[1][k]-o[:,k])/denom
        low=xp.maximum(low,xp.where(parallel,-xp.inf,xp.minimum(u,v)))
        high=xp.minimum(high,xp.where(parallel,xp.inf,xp.maximum(u,v)))
        valid &= ~parallel|((o[:,k]>=box[0][k])&(o[:,k]<=box[1][k]))
    result.origins=xp.ascontiguousarray(o);result.directions=xp.ascontiguousarray(direction)
    result.surface_distance=xp.ascontiguousarray(distance)
    result.tool_normals=xp.ascontiguousarray(-d)
    result.ray_bounds=xp.ascontiguousarray(xp.column_stack((low,xp.where(valid,high,low-1))))
    result.center_rays=True
    return result


class CenterRayModel:
    center_ray_update=True

    def __init__(self,surface,grid,*,parent=None,device_input=False,capture=False):
        started=perf_counter();self.surface=surface;self.runtime=surface.rt;self.grid=grid
        self.readback_bytes=0;self.closed=False;self.parent=parent
        rt=self.runtime;cp=rt.cp
        with rt.context():
            self._origins=cp.asarray(grid.origins);self._directions=cp.asarray(grid.directions)
            self._distance=cp.asarray(grid.surface_distance);self._normals=cp.asarray(grid.tool_normals)
            # This zero mask is workspace for stage 3, never a generation decision.
            self.removed=parent.removed if parent else cp.zeros(len(surface.snapshot.endpoints['positions']),cp.uint32)
            self.added=cp.zeros(len(grid.origins),cp.uint32)
            self.stats=dict(surface.stats,hits=0,merged_hits=0,nearest_seed_rays=0,inferred_crossings=0,
                generation_hit_queries=0,generation_hit_links_retained=0,no_hit_rays=None,
                initialization_s=perf_counter()-started,seed_payload_readback_bytes=0,
                seed_control_readback_bytes=0,bvh_build_ms=surface.build_ms,gs_pack_upload_ms=surface.pack_upload_ms)

    def _download(self,a):
        result=self.runtime.cp.asnumpy(a,stream=self.runtime.stream);self.readback_bytes+=result.nbytes;return result

    def step_device(self,sweep):
        started=perf_counter();rt=self.runtime;cp=rt.cp;self.sweep=sweep
        with rt.context():
            from cutsim.cuda.analytic_ray_generation import select_generation
            select_generation(self)
            self.selected=cp.flatnonzero(self.added)
            rt.stream.synchronize()
        self.stats['last_device_step_s']=perf_counter()-started
        return len(self.selected)

    def replace_parent_rays(self,ray_ids):
        """Replace sampling cells without any old-GS correspondence to update."""
        with self.runtime.context():
            self.added[self.runtime.cp.asarray(ray_ids)]=0
            self.selected=self.runtime.cp.flatnonzero(self.added)

    def endpoints_device(self,**_):
        cp=self.runtime.cp;r=self.selected
        return dict(positions=self._origins[r]+self._distance[r,None]*self._directions[r],
            normals=self._normals[r],ray_ids=r,distances=self._distance[r],owners=cp.zeros(len(r),cp.int64),
            tool_ids=cp.full(len(r),self.sweep.tool_id,cp.int32))

    def close(self):
        for name in ('_origins','_directions','_distance','_normals','removed','added','selected','generation_evidence'):
            setattr(self,name,None)
        self.surface=None;self.parent=None;self.closed=True

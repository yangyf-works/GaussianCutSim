"""Exact draw ownership at a vertical flat plunge / horizontal flat sweep join.

The shared cylinder is divided by the plane through its axis perpendicular to
the horizontal motion. Only this analytic identity is used; no angular or
distance tolerance merges unrelated faces. CPU material domains are untouched.
"""
from copy import copy
import numpy as np
from cutsim.core.sweep import FlatSweep
from cutsim.core.regions import ExactSurfaceDomains


def cap_junctions(material):
    cylinders={};result=[]
    aliases=getattr(material,'_surface_aliases',())
    def identity(i):return 7+3*(aliases[i] if i<len(aliases) else i)
    for i,s in enumerate(material.sweeps):
        if isinstance(s,FlatSweep) and np.array_equal(s.a_xy,s.b_xy):
            center=s.a_xy;low=s.z_lo;high=s.z_hi
        elif (getattr(s,'kind',None)=='flat' and hasattr(s,'a_xyz') and
              np.array_equal(s.a_xyz[:2],s.b_xyz[:2])):
            center=s.a_xyz[:2];low=s.z_lo
            high=s.z_hi
        else:
            if not isinstance(s,FlatSweep):continue
            motion=s.b_xy-s.a_xy;length=np.linalg.norm(motion)
            if not length:continue
            for center,direction in ((s.a_xy,motion/length),(s.b_xy,-motion/length)):
                key=(s.tool_id,s.radius,*center)
                for old,low,high in cylinders.get(key,()):
                    lo=max(low,s.z_lo);hi=min(high,s.z_hi)
                    if lo>=hi or old==identity(i):continue
                    plane=np.r_[direction,0.,-direction@center]
                    result.append((old,identity(i),lo,hi,plane))
            continue
        cylinders.setdefault((s.tool_id,s.radius,*center),[]).append((identity(i),low,high))
    return result


def bound_cap_junctions(cloud,attributes,material):
    """Intersect draw domains with complementary surface ownership regions.

    Outside the common vertical interval all existing domains remain valid.
    Within it the old cylinder owns the back half; the subsequent horizontal
    sweep owns the forward half. Preserve all stock/previous-cut constraints.
    """
    pairs=cap_junctions(material)
    if not pairs:return cloud
    out=copy(cloud);out.sweep_domains=cloud.sweep_domains.copy()
    labels=attributes['surface_id'][:,0]
    lateral=np.abs(cloud.normals[:,2])<1e-12
    for old,new,lo,hi,plane in pairs:
        for identity,sign in ((old,-1.),(new,1.)):
            for row in np.flatnonzero(lateral&(labels==identity)):
                previous=out.sweep_domains[row]
                # Unknown peer-GS clipping must not be replaced by explicit
                # ownership constraints. Only authoritative analytic domains.
                if previous is None:continue
                alternatives=(sign*plane,np.array([0.,0.,-1.,lo]),np.array([0.,0.,1.,-hi]))
                domain=ExactSurfaceDomains(np.vstack((region,boundary))
                    for region in previous for boundary in alternatives)
                for name in ('stock_planes','source_planes'):
                    planes=getattr(previous,name,None)
                    if planes is not None:setattr(domain,name,planes)
                out.sweep_domains[row]=domain
    return out

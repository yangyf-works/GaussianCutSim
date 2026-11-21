"""Reuse translation-independent plane projection; preserve scalar arithmetic."""
import numpy as np
from cutsim.rendering.exact_draw_cache import ExactDrawCache, array_key

projection_cache=ExactDrawCache(16*1024*1024)
description_cache=ExactDrawCache(32*1024*1024)


def describe_domain(domain,position,scale,u,v,exact,*,_projected=None):
    projected=([project_region(world,position,scale,u,v) for world in domain]
               if _projected is None else _projected)
    # The existing exact single-region reducer removes these planes before
    # polygon clipping. Do this before the cache key, allowing translated flat
    # faces/straight edges to reuse the identical local result. Never apply it
    # before Boolean expansion of a multi-region union.
    if exact and len(projected)==1 and projected[0] is not None:
        projected[0]=[q for q in projected[0] if q[2]<3.+1e-6]
    key=(exact,tuple(None if q is None else array_key(np.asarray(q).reshape(-1,3)) for q in projected))
    result=description_cache.get(key)
    if result is None:
        from cutsim.rendering.clipping import _describe_sweep_domain
        result=_describe_sweep_domain(domain,position,scale,u,v,exact,_projected=projected)
        description_cache.put(key,result)
    return result


def project_region(world,position,scale,u,v):
    normals=world[:,:3]
    key=(array_key(normals),array_key(scale[:2]),array_key(u),array_key(v))
    template=projection_cache.get(key)
    if template is None:
        xy=np.column_stack((normals@u*scale[0],normals@v*scale[1]))
        lengths=np.array([np.linalg.norm(row) for row in xy])
        parallel=np.flatnonzero(lengths<1e-10)
        active=np.flatnonzero(lengths>=1e-10)
        directions=np.zeros_like(xy);directions[active]=xy[active]/lengths[active,None]
        close=np.zeros((len(xy),len(xy)),bool)
        for offset,i in enumerate(active):
            for j in active[:offset]:close[i,j]=np.linalg.norm(directions[j]-directions[i])<1e-7
        # Normals within 1e-7 merge. The winning *whole plane*, including its
        # own direction, must replace the representative when clearance wins.
        # Store pairwise direction comparisons instead of assuming transitivity.
        template=(directions,lengths,parallel,active,close)
        projection_cache.put(key,template)
    directions,lengths,parallel,active,close=template
    constants=normals@position+world[:,3]
    if len(parallel) and np.any(constants[parallel]<-1e-8):return None
    selected=[];clearances=constants[active]/lengths[active]
    values={int(i):value for i,value in zip(active,clearances)}
    for i in active:
        match=next((j for j,old in enumerate(selected) if close[i,old]),None)
        if match is None:selected.append(i)
        elif values[i]<values[selected[match]]:selected[match]=i
    return [np.array([*directions[i],values[i]]) for i in selected]

"""Finite observation domains; their caps are not physical material surfaces."""
import numpy as np

WINDOW_BOUNDARY_OWNER = np.iinfo(np.int64).min


def surface_distances(grid):
    value=np.asarray(getattr(grid,'surface_distance',None),dtype=float)
    if value.shape!=(len(grid.origins),) or not np.isfinite(value).all() or np.any(value<0):
        raise ValueError('Finite nonnegative surface distance required for each ray')
    return np.ascontiguousarray(value)


def intersect_box(origins,directions,domain,low,high):
    """Clip finite rays to a closed AABB, including parallel/on-face rays."""
    result=np.array(domain,copy=True,dtype=float)
    for axis in range(3):
        o,d=origins[:,axis],directions[:,axis]
        moving=d!=0
        a=np.divide(low[axis]-o,d,out=np.full(len(o),-np.inf),where=moving)
        b=np.divide(high[axis]-o,d,out=np.full(len(o),np.inf),where=moving)
        result[:,0]=np.maximum(result[:,0],np.minimum(a,b))
        result[:,1]=np.minimum(result[:,1],np.maximum(a,b))
        result[(~moving)&((o<low[axis])|(o>high[axis])),1]=-np.inf
    if np.any(np.asarray(low)>=high):result[:,1]=-np.inf
    return result,result[:,0]<result[:,1]


def ray_bounds(grid):
    value=getattr(grid,'ray_bounds',None)
    if value is None:return None
    value=np.asarray(value,dtype=float)
    if value.shape!=(len(grid.origins),2) or not np.isfinite(value).all() or np.any(value[:,0]>=value[:,1]):
        raise ValueError('Finite ordered (N,2) profile ray bounds required')
    result=value.copy();result.flags.writeable=False
    return result


def clip_line(line,low,high):
    result=[]
    for a,b,left,right in line:
        lo,hi=max(a,low),min(b,high)
        if lo<hi:
            result.append((lo,hi,WINDOW_BOUNDARY_OWNER if a<low else left,
                           WINDOW_BOUNDARY_OWNER if b>high else right))
    return tuple(result)

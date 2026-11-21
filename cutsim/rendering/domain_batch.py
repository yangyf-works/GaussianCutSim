"""Batched drawing-domain normalization/reduction; Float64, no fastmath.

The NumPy world-to-local products retain the reference arithmetic. Variable
length local constraints are packed with offsets, then independent parents
are normalized, merged, sorted and clipped in a bounded native batch.
Boolean unions and non-exact reductions use ordered domain processing
after native normalization; their local results are cached.
"""
import os
import numpy as np
from cutsim.geometry.strip_kernels import njit, prange, enabled as native_enabled, configure_threads
from cutsim.rendering.clip_polygon_kernel import render_clip_polygon


def enabled():
    return (native_enabled() and os.environ.get('CUTSIM_PREP_ACCEL','1')!='0'
            and os.environ.get('CUTSIM_BOUNDARY_JIT','1')!='0')


@njit(cache=True,nogil=True)
def reduce_one(planes,complete=True):
    reduced=np.empty_like(planes)
    count=0
    for plane in planes:
        length=np.sqrt(np.dot(plane[:2],plane[:2]))
        if length<1e-10:
            if plane[2]<-1e-8:return reduced[:0],True
            continue
        p=plane/length
        match=-1
        for j in range(count):
            delta=reduced[j,:2]-p[:2]
            if np.sqrt(np.dot(delta,delta))<1e-7:
                match=j
                break
        if match<0:
            reduced[count]=p
            count+=1
        elif p[2]<reduced[match,2]:
            reduced[match]=p
    if not complete:return reduced[:count],False
    # Same single-conjunction canonical order as _sweep_expand_union_regions.
    for i in range(1,count):
        p=reduced[i].copy()
        j=i-1
        while j>=0:
            q=reduced[j]
            greater=(q[0]>p[0] or (q[0]==p[0] and
                     (q[1]>p[1] or (q[1]==p[1] and q[2]>p[2]))))
            if not greater:break
            reduced[j+1]=q
            j-=1
        reduced[j+1]=p
    n=0
    for i in range(count):
        if reduced[i,2]<3.+1e-6:
            reduced[n]=reduced[i]
            n+=1
    q=reduced[:n].copy()
    radius=3.+1e-6
    polygon=np.array([[-radius,-radius],[radius,-radius],[radius,radius],[-radius,radius]])
    for plane in q:
        polygon=render_clip_polygon(polygon,plane)
        if not len(polygon):return q[:0],True
    if not n:return q,False
    signed=polygon@q[:,:2].T+q[:,2]
    kept=0
    for j in range(n):
        minimum=np.min(signed[:,j])
        if minimum<=1e-7:
            reduced[kept]=q[j]
            kept+=1
    return reduced[:kept],False


@njit(cache=True,nogil=True,parallel=True)
def reduce_many(packed,offsets,complete=None):
    output=np.empty_like(packed)
    counts=np.zeros(len(offsets)-1,np.int64)
    empty=np.zeros(len(counts),np.bool_)
    for i in prange(len(counts)):
        start,end=offsets[i],offsets[i+1]
        q,invalid=reduce_one(packed[start:end],complete is None or complete[i])
        output[start:start+len(q)]=q
        counts[i]=len(q)
        empty[i]=invalid
    return output,counts,empty


@njit(cache=True,nogil=True)
def line_peak_scores(groups):
    """Feasible boundary intervals for Top-2; parent workers already parallel.

    Quadrature/transcendentals and canonical tie selection stay in the existing
    NumPy caller, preserving its summation and comparison order.
    """
    result=np.full((len(groups),groups.shape[1]),-np.inf)
    for row in range(len(groups)):
        q=groups[row]
        for index in range(len(q)):
            p=q[index]
            foot=-p[2]*p[:2]
            tangent=np.array([-p[1],p[0]])
            lo,hi=-np.inf,np.inf
            valid=True
            for j in range(len(q)):
                if j==index:continue
                b=q[j]
                slope=np.dot(b[:2],tangent)
                offset=np.dot(b[:2],foot)+b[2]
                if abs(slope)<1e-12:
                    if offset < -1e-9:
                        valid=False
                        break
                elif slope>0:lo=max(lo,-offset/slope)
                else:hi=min(hi,-offset/slope)
            if valid and lo<=hi+1e-9:
                # np.clip also defines the near-contact case lo > hi.
                t=min(max(0.,lo),hi)
                result[row,index]=-.5*(p[2]**2+t*t)
    return result


def describe_many(cloud,arr,u,v,ids):
    """Normalize every branch; never reduce before multi-branch Boolean union."""
    from cutsim.core.regions import ExactSurfaceDomains
    from cutsim.rendering.domain_projection import describe_domain
    result={}
    rows=list(ids)
    if not rows:return result
    configure_threads()
    # Bounded packing even for large initial workpieces. No geometry budget.
    for begin in range(0,len(rows),2048):
        batch=rows[begin:begin+2048]
        local=[]
        offsets=[0]
        parent_offsets=[0]
        complete=[]
        for i in batch:
            domain=cloud.sweep_domains[i]
            single=isinstance(domain,ExactSurfaceDomains) and len(domain)==1
            for world in domain:
                local.append(np.column_stack((world[:,:3]@u[i]*arr['scale'][i,0],
                    world[:,:3]@v[i]*arr['scale'][i,1],world[:,:3]@arr['positions'][i]+world[:,3])))
                offsets.append(offsets[-1]+len(world))
                complete.append(single)
            parent_offsets.append(len(local))
        offsets=np.asarray(offsets,np.int64)
        packed=np.concatenate(local) if local else np.empty((0,3))
        output,counts,empty=reduce_many(packed,offsets,np.asarray(complete,np.bool_))
        for k,i in enumerate(batch):
            start,end=parent_offsets[k:k+2]
            domain=cloud.sweep_domains[i]
            if start<end and complete[start]:
                if empty[start]:result[i]=([],[],[],[])
                else:
                    q=output[offsets[start]:offsets[start]+counts[start]].copy()
                    regions=[q]
                    selected,indices=(None,None) if len(q)>2 else ([q.copy()],[list(range(len(q)))])
                    result[i]=(regions,regions,selected,indices)
            else:
                # Cached descriptions must not retain the whole batch through
                # a small ndarray view (the LRU accounts owned branch bytes).
                projected=[None if empty[j] else output[offsets[j]:offsets[j]+counts[j]].copy()
                           for j in range(start,end)]
                result[i]=describe_domain(domain,arr['positions'][i],arr['scale'][i],u[i],v[i],
                    isinstance(domain,ExactSurfaceDomains),_projected=projected)
    return result

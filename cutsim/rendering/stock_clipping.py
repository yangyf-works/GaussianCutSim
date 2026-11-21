"""Analytic stock limits, independent of the 2-sigma GS/GS edge search."""
import copy
import numpy as np
from cutsim.core.regions import ExactSurfaceDomains


def stock_planes(material,position,normal):
    normal=np.asarray(normal,float);normal=normal/max(np.linalg.norm(normal),1e-30)
    planes=[]
    for axis in range(3):
        tangent=np.sqrt(max(0.,1.-normal[axis]**2))
        for sign,bound in ((1.,material.stock_lo[axis]),(-1.,material.stock_hi[axis])):
            clearance=sign*(position[axis]-bound)
            if tangent<=1e-8:
                # A coincident stock face is constant on the GS. Float32
                # quaternion roundoff must not manufacture a cutting edge.
                if clearance < -1e-7:return np.array([[0.,0.,0.,-1.]])
                continue
            p=np.zeros(4);p[axis]=sign;p[3]=-sign*bound;planes.append(p)
    return np.asarray(planes,float).reshape(-1,4)


def merge_boundaries(domain,planes):
    """Intersect a union with finite-face limits, preserving mandatory bounds."""
    if not len(planes):return domain
    if domain is None:domain=(np.empty((0,4)),)
    branches=[]
    for q in domain:
        q=np.asarray(q,float).reshape(-1,4)
        extra=[p for p in planes if not np.any(np.all(q==p,axis=1))]
        branches.append(np.vstack((q,extra)) if extra else q)
    result=ExactSurfaceDomains(tuple(branches))
    for name in ('stock_planes','source_planes'):
        value=getattr(domain,name,None)
        if value is not None:setattr(result,name,value)
    return result


def merge_stock(domain,planes):
    result=merge_boundaries(domain,planes)
    if len(planes):result.stock_planes=planes
    return result


def complete_clip_bounds(planes,bounds):
    """Use spare native slots for finite-face bounds, even outside 3 sigma.

    The renderer's screen-space subpixel filter can accept UV beyond that
    disk at grazing angles. Keeping an inactive source bound costs no extra
    GS and prevents that filter from extrapolating below the generating floor.
    Existing selected edges are never replaced.
    """
    result=np.asarray(planes).reshape(-1,3)
    for plane in bounds:
        if len(result)>=2:break
        if any(np.linalg.norm(q[:2]-plane[:2])<1e-8 and q[2]<=plane[2]+1e-8 for q in result):continue
        result=np.vstack((result,plane))
    return result


def bound_drawing_cloud(arr,cloud,material):
    """Add stock limits after ordinary GS/GS selection, keeping both kinds.

    Only rows near stock faces need work. The extra .015 mm is the maximum
    3-sigma reach of a .005 mm minimum-size child, not a moved stock boundary.
    CPU domains and geometric positions remain untouched.
    """
    from cutsim.rendering.gaussian_attributes import _render_axes
    n=np.asarray(cloud.normals,float);n=n/np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
    tangent=np.sqrt(np.maximum(0.,1.-n*n))
    radius=3.*np.max(arr['scale'][:,:2],axis=1)+.015
    distance=np.minimum(cloud.positions-material.stock_lo,material.stock_hi-cloud.positions)
    mask=np.any(((tangent>1e-8)&(distance<=radius[:,None]*tangent+1e-7))|(distance< -1e-7),axis=1)
    rows=np.flatnonzero(mask)
    if not len(rows):return cloud
    out=copy.copy(cloud);out.sweep_domains=cloud.sweep_domains.copy()
    u,v=_render_axes(arr['rot'][rows])
    for local,i in enumerate(rows):
        stock=stock_planes(material,cloud.positions[i],cloud.normals[i])
        domain=out.sweep_domains[i]
        if domain is None:
            if arr['clip_empty'][i]:domain=()
            else:
                pending=(getattr(arr,'approximated',{}).get(int(i),{}).get('fallback_regions_local')
                         if arr['clip_pending'][i] else None)
                if pending is not None:local_regions=pending
                else:
                    values=[[arr[k][i,0] for k in ('clip1_angle','clip1_split','clip2_angle','clip2_split')]]
                    if arr['clip_concave'][i]:values.append(arr['clip_second_values'][i])
                    local_regions=[np.asarray([[np.cos(float(a)),np.sin(float(a)),-float(b)]
                        for a,b in np.asarray(branch).reshape(2,2) if a!=4.]).reshape(-1,3) for branch in values]
                domain=[]
                for branch in local_regions:
                    planes=[]
                    for x,y,constant in branch:
                        direction=x*u[local]/arr['scale'][i,0]+y*v[local]/arr['scale'][i,1]
                        p=np.r_[direction,constant-direction@arr['positions'][i]]
                        length=np.linalg.norm(direction)
                        if length>1e-14:
                            p=p/length
                            # Canonicalize Float32 round-trips of an existing
                            # stock edge (2e-7 normal, 2e-6 mm offset), not a
                            # neighbouring geometric boundary. This prevents
                            # duplicate almost-parallel constraints/splitting.
                            match=np.flatnonzero((np.linalg.norm(stock[:,:3]-p[:3],axis=1)<=2e-7)&
                                                 (np.abs(stock[:,3]-p[3])<=2e-6))
                            if len(match):p=stock[match[0]]
                        planes.append(p)
                    domain.append(np.asarray(planes).reshape(-1,4))
        out.sweep_domains[i]=merge_stock(domain,stock)
    return out


def terminal_stock_selection(candidates,stock,origin,inverse,position,ratio):
    """Keep stock planes intersecting the emitted child's actual 3-sigma disk."""
    from cutsim.rendering.clipping import _sweep_top_two
    if stock is None or not len(stock):
        chosen,indices=_sweep_top_two(candidates)
        return chosen,indices,candidates
    center=inverse@(position.astype(float)-origin)
    hard=np.column_stack((stock[:,:2],(stock[:,:2]@center+stock[:,2])/ratio))
    hard=hard[hard[:,2]<3.+1e-7]
    required=[];q=np.asarray(candidates)
    for p in hard:
        match=np.flatnonzero(np.all(np.abs(q-p)<=1e-8*np.maximum(1.,np.abs(p)),axis=1))
        if len(match):index=int(match[0])
        else:index=len(q);q=np.vstack((q,p))
        if index not in required:required.append(index)
    if len(required)>2:
        raise ValueError('Terminal GS intersects more than two mandatory stock planes; finer subdivision is required')
    if len(required)==2:return q[required].copy(),required,q
    chosen,indices=_sweep_top_two(q,mandatory=required[0] if required else None)
    return chosen,indices,q

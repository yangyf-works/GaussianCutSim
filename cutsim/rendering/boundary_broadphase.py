"""Conservative rejection of empty GS peer searches, never a clipping rule."""
import math
import numpy as np


def possible_peer_rows(cloud, rows, store, radii, settings):
    """Cardinal normal groups and arbitrary-normal spatial groups have boxes.

    A box encloses every peer centre in its group and uses the largest actual
    intersection radius. Rejected rows cannot pass the existing distance AND
    normal predicate. Arbitrary normals receive no angular rejection.
    Rebuild from live records each time: no stale geometry/cache dependencies.
    """
    from cutsim.rendering.clipping import _gs_radius
    rows=np.asarray(rows,dtype=np.int64)
    if len(rows)<4096:return rows
    ids=np.fromiter(store.points,dtype=np.int64,count=len(store.points))
    slots=store.slots(ids)
    positions=store.geometry['positions'][slots]
    normals=store.geometry['normals'][slots]
    peer_radii=_gs_radius(store.geometry['support'][slots],store.geometry['seam_weight'][slots],settings)
    if not len(ids):return rows[:0]
    p=cloud.positions[rows];r=radii[rows]
    if not (np.isfinite(positions).all() and np.isfinite(peer_radii).all() and np.isfinite(r).all()):return rows
    own=cloud.normals[rows]
    own=own/np.maximum(np.linalg.norm(own,axis=1)[:,None],1e-30)
    other=np.ones(len(ids),bool);keep=np.zeros(len(rows),bool)
    threshold=math.sin(math.radians(settings.half_min_angle_deg))**2-1e-12
    # Extra broad-phase slack only, in mm, scaled to coordinate magnitude.
    # Exact 2-sigma distance/angle predicates below remain authoritative.
    slack=1e-10*max(1.,float(np.max(np.abs(positions))),float(np.max(np.abs(p))))
    def admit(mask,angled):
        if not np.any(mask):return
        q=positions[mask];lo=q.min(axis=0);hi=q.max(axis=0)
        distance=np.maximum(np.maximum(lo-p,p-hi),0.)
        reach=r+peer_radii[mask].max()+slack
        keep[:] |= angled & (np.sum(distance*distance,axis=1)<=reach*reach+1e-12)
    for axis in range(3):
        for sign in (1.,-1.):
            n=np.zeros(3);n[axis]=sign
            mask=np.all(normals==n,axis=1);other &= ~mask
            admit(mask,1.-own[:,axis]**2>=threshold)
    # Tilted edge/tool normals must not share one box spanning the entire
    # workpiece. Partition their centres spatially; angular tests remain
    # deliberately conservative for these arbitrary normal groups.
    if np.any(other):
        indices=np.flatnonzero(other);q=positions[indices]
        lo=q.min(axis=0);span=np.maximum(q.max(axis=0)-lo,1e-12)
        cells=np.minimum(3,np.floor(4*(q-lo)/span).astype(np.int64))
        codes=cells[:,0]+4*cells[:,1]+16*cells[:,2]
        for code in np.unique(codes):
            mask=np.zeros(len(ids),bool);mask[indices[codes==code]]=True
            admit(mask,True)
    return rows[keep]

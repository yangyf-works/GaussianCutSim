import numpy as np
"""Shared finite-sweep bounds (no persistent Dexel grid)."""
from cutsim.core.sweep import BallSweep

def bounds(sweep):
    lo, hi = sweep.world_aabb()
    if isinstance(sweep, BallSweep):
        hi[2] = max(hi[2], sweep.center_z+sweep.radius)
    return lo, hi


def _sections(part, points, axis):
    """Intersect a batch of axis-parallel infinite lines with one convex part."""
    lo = np.full(len(points), -np.inf)
    hi = np.full(len(points), np.inf)
    for plane in part.planes:
        slope = plane[axis]
        value = points@plane[:3]+plane[3]
        if slope > 0.:
            lo = np.maximum(lo, -value/slope)
        elif slope < 0.:
            hi = np.minimum(hi, -value/slope)
        else:
            hi[value < 0.] = -np.inf
    if part.matrix is not None:
        delta = points-part.center
        a = part.matrix[axis,axis]
        b = delta@part.matrix[axis]
        c = np.sum((delta@part.matrix)*delta,axis=1)-part.radius**2
        if a > 0.:
            disc = b*b-a*c
            root = np.sqrt(np.maximum(0.,disc))
            # Stable quadratic roots, including intersections near the origin.
            q = -b-np.copysign(root,b)
            t1 = q/a
            t2 = np.divide(c,q,out=t1.copy(),where=q!=0.)
            lo = np.maximum(lo,np.minimum(t1,t2))
            hi = np.minimum(hi,np.maximum(t1,t2))
            hi[disc <= 0.] = -np.inf
        else:
            hi[b>0.] = np.minimum(hi[b>0.],-c[b>0.]/(2.*b[b>0.]))
            lo[b<0.] = np.maximum(lo[b<0.],-c[b<0.]/(2.*b[b<0.]))
            hi[(b==0.) & (c>0.)] = -np.inf
    return lo,hi

def _subtract(intervals, a, b, sid):
    """Intervals carry the origin of both faces (-1 for stock)."""
    result = []
    for lo,hi,left,right in intervals:
        if a>=hi or b<=lo:
            result.append((lo,hi,left,right))
        else:
            if a>lo:
                result.append((lo,a,left,sid))
            if b<hi:
                result.append((b,hi,sid,right))
    return tuple(result)

"""Common 2-sigma search and clipping domain; Gaussian sigma is unchanged."""
import numpy as np
from cutsim.cutting.local_dexel import bounds


# Include neighbouring Gaussian tails in clipping, independently of residual R.
CLIP_SIGMA = 2.
CLIP_SEARCH_SIGMA = CLIP_SIGMA


def update_candidates(positions,u,v,sigma,sweep):
    """Conservative broad phase for the oriented 2-sigma tangent discs.

    AABB overlap can include false positives; actual cutter/clip calculations
    still decide the boundary. In particular, a centre outside the sweep must
    not exclude a disc whose edge reaches it. Existing clipping can only
    reduce this enclosing box and is handled by the downstream clip merge.
    """
    radius=CLIP_SEARCH_SIGMA*np.asarray(sigma,dtype=float)
    reach=radius[:,None]*np.sqrt(np.asarray(u)**2+np.asarray(v)**2)
    lo,hi=bounds(sweep)
    eps=1e-9*(1.+np.abs(positions)+reach)
    return (radius>0)&np.all((positions+reach+eps>=lo)&(positions-reach-eps<=hi),axis=1)


def boundary_probes(positions,u,v,sigma):
    """Two rings at 1 and 2 sigma; same 16-probe ordering as before."""
    angles=np.tile(np.arange(8)*np.pi/4,2)
    radii=np.repeat([.5,1.],8)[:,None]*CLIP_SEARCH_SIGMA*np.asarray(sigma)[None]
    return positions[None]+radii[:,:,None]*(
        np.cos(angles)[:,None,None]*u[None]+np.sin(angles)[:,None,None]*v[None])

from cutsim.geometry.strip_kernels import njit
import numpy as np

@njit(cache=True, nogil=True)
def render_clip_polygon(polygon, plane):
    if len(polygon) == 0: return polygon
    signed = polygon @ plane[:2] + plane[2]
    out = np.empty((2*len(polygon),2)); count=0
    for i in range(len(polygon)):
        end=polygon[i]; start=polygon[i-1]
        a,b=signed[i-1],signed[i]
        if (a>=0.) != (b>=0.):
            out[count]=start+(end-start)*(a/(a-b)); count+=1
        if b>=0.:
            out[count]=end; count+=1
    return out[:count].copy()

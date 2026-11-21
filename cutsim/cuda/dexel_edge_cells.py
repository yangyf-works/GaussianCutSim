"""Identify generated Dexel cells on a generation boundary or chart edge."""
import numpy as np
from numba import njit


@njit(cache=True)
def _edge_interval_hits(group, axis, high, coordinate, start, end,
                        qgroup, qaxis, qhigh, qcoordinate, lower, upper):
    """Binary-search generated edge intervals without Python per-cell loops."""
    extent = end.copy()
    for i in range(1, len(end)):
        if (group[i], axis[i], high[i], coordinate[i]) == (group[i-1], axis[i-1], high[i-1], coordinate[i-1]):
            extent[i] = max(extent[i-1], end[i])
    result = np.zeros(len(lower), np.bool_)
    for row in range(len(lower)):
        target = (qgroup[row], qaxis[row], qhigh[row], qcoordinate[row], upper[row]-1e-8)
        lo, hi = 0, len(start)
        while lo < hi:
            mid = (lo+hi)//2
            key = (group[mid], axis[mid], high[mid], coordinate[mid], start[mid])
            if key < target:
                lo = mid+1
            else:
                hi = mid
        i = lo-1
        if i >= 0 and (group[i], axis[i], high[i], coordinate[i]) == target[:4]:
            result[row] = extent[i] > lower[row]+1e-8
    return result


def adjacent_unseeded_cells(atlas, accepted, candidates):
    """Missing base cells sharing a positive-length edge with an accepted cell.

    Adjacency uses the original chart cells, not GS distance or clipped support.
    Pose/chart boundaries stay separate; periodic azimuth wraps. Unequal ring
    populations may share only part of an edge. Corner-only contact is excluded.
    """
    accepted = np.asarray(accepted, np.int64)
    candidates = np.asarray(candidates, np.int64)
    if not len(accepted) or not len(candidates):
        return np.empty(0, np.int64)
    n = len(atlas.origins)
    pose_ids = np.zeros(n, np.int64)
    start = 0
    for i, pose in enumerate(atlas.poses):
        end = min(n, start + pose['rays'])
        pose_ids[start:end] = i
        start = end
    bounds = np.asarray(atlas.cell_bounds)
    charts = np.asarray(atlas.charts)
    _, chart_ids = np.unique(charts, return_inverse=True)
    side_ids = np.zeros(n, np.int64)
    separate = np.isin(charts, ('strip_side', 'strip_cap'))
    if separate.any():
        _, sides = np.unique(np.round(atlas.directions[separate], 8), axis=0, return_inverse=True)
        side_ids[separate] = sides + 1
    group_ids = (pose_ids * (int(chart_ids.max())+1) + chart_ids) * (int(side_ids.max())+1) + side_ids
    periodic = ~np.char.startswith(charts, 'strip_')
    valid = (charts != 'ramp') & (bounds[:,1] > bounds[:,0]) & (bounds[:,3] > bounds[:,2])
    accepted = accepted[valid[accepted]]
    candidates = candidates[valid[candidates]]
    if not len(accepted) or not len(candidates):
        return np.empty(0, np.int64)
    dtype = np.dtype([('group', np.int64), ('axis', np.int8), ('high', np.int8),
                      ('coordinate', np.float64), ('start', np.float64)])

    def edges(rows, *, query=False):
        cells = bounds[rows]
        keys = np.empty(4*len(rows), dtype=dtype)
        keys['group'] = np.repeat(group_ids[rows], 4)
        keys['axis'] = np.tile([0,0,1,1], len(rows))
        keys['high'] = np.tile([1,0,1,0] if query else [0,1,0,1], len(rows))
        coordinate = cells.reshape(-1).copy()
        seam = np.repeat(periodic[rows], 4) & (keys['axis']==1) & (abs(coordinate-2*np.pi)<1e-8)
        coordinate[seam] = 0.
        keys['coordinate'] = np.round(coordinate, 9)
        lower = cells[:,[2,2,0,0]].reshape(-1)
        upper = cells[:,[3,3,1,1]].reshape(-1)
        # Query the last generated interval starting strictly before this end.
        keys['start'] = upper-1e-8 if query else lower
        return keys, lower, upper

    # Vectorized edge lookup avoids Python work per cell. Prefix maxima within
    # each edge key also handle overlapping/unequal generated intervals.
    generated, _, upper = edges(accepted)
    order = np.lexsort(tuple(generated[name] for name in reversed(generated.dtype.names)))
    generated, upper = generated[order], upper[order]
    queries, lower, query_upper = edges(candidates, query=True)
    adjacent = _edge_interval_hits(
        *(generated[name].copy() for name in generated.dtype.names), upper,
        *(queries[name].copy() for name in ('group','axis','high','coordinate')), lower, query_upper)
    return candidates[adjacent.reshape(-1,4).any(axis=1)]



def edge_generated_cells(atlas, ray_ids, *, refinement=None):
    """Return a mask in GS order, including partially covered child-cell edges.

    Neighbours must share a pose, chart and strip side. A side is interior
    only if generated neighbours cover its complete parameter interval.
    Chart joins are intentionally included; periodic azimuth seams are joined.
    """
    ids=np.asarray(ray_ids,np.int64);n=len(atlas.origins)
    if np.any(ids<0) or np.any(ids>=n):raise ValueError('Expected local generated ray IDs')
    pose_ids=np.zeros(n,np.int64);start=0
    for i,pose in enumerate(atlas.poses):
        end=min(n,start+pose['rays']);pose_ids[start:end]=i;start=end
    refinements=refinement if isinstance(refinement,(list,tuple)) else (refinement,)
    for descriptor in refinements:
        if descriptor is None:continue
        base=descriptor['base_rays'];parents=np.asarray(descriptor['parents'])
        pose_ids[base:base+len(parents)]=pose_ids[parents]
    bounds=np.asarray(atlas.cell_bounds)[ids]
    charts=np.asarray(atlas.charts)[ids]
    groups=[]
    for row,chart in zip(ids,charts):
        # strip_side uses identical parameters on the two opposite walls.
        side=tuple(np.round(atlas.directions[row],8)) if chart=='strip_side' else ()
        groups.append((int(pose_ids[row]),str(chart),side))
    edges={};sides=[]
    for i,(group,b,chart) in enumerate(zip(groups,bounds,charts)):
        row=[]
        for axis in range(2):
            lo,hi=b[2*(1-axis):2*(1-axis)+2]
            for high in (0,1):
                x=b[2*axis+high]
                if axis==1 and not chart.startswith('strip_') and abs(x-2*np.pi)<1e-8:x=0.
                key=(group,axis,round(float(x),9))
                edges.setdefault((*key,high),[]).append((float(lo),float(hi)))
                row.append((key,high,float(lo),float(hi)))
        sides.append(row)
    result=np.zeros(len(ids),bool)
    for i,row in enumerate(sides):
        if charts[i]=='ramp' or np.any(bounds[i,[1,3]]<=bounds[i,[0,2]]):
            result[i]=True;continue
        for key,high,lo,hi in row:
            end=lo
            for a,b in sorted(edges.get((*key,1-high),())):
                if a>end+1e-8:break
                if b>end:end=b
                if end>=hi-1e-8:break
            if end<hi-1e-8:result[i]=True;break
    return result

"""Read-only picking of the displayed, clipped GS and its generating motion."""
import numpy as np

from cutsim.rendering.gaussian_attributes import _render_axes


def draw_owners(records, count):
    """Follow native clip expansion without adding attributes to GPU uploads."""
    owners = records.get('machining_owner')
    if owners is None:
        return np.full(count, -1, np.int32)
    n = len(owners)
    joins = np.asarray(records.get('clip_concave', np.zeros(n, bool)))
    hidden = (np.asarray(records.get('clip_pending', np.zeros(n, bool))) |
              records.get('clip_empty', False))
    parents = np.repeat(np.arange(n), 1 + joins.astype(int))
    parents = parents[~hidden[parents]]
    result = np.full(count, -1, np.int32)
    if len(parents) <= count:
        result[:len(parents)] = owners[parents]
    return result


def pick_gs(draw, origin, direction, max_distance=np.inf):
    """Intersect native 3-sigma tangent disks, including both clipping edges.

    Stock/unknown GS participate in occlusion too. Return the nearest visible
    disk and hit position, rather than selecting a nearby screen-space centre.
    Work is chunked to keep temporary memory bounded for large models.
    """
    if draw is None:
        return None
    origin, direction = np.asarray(origin, float), np.asarray(direction, float)
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length <= 1e-12:
        return None
    direction = direction / length
    best, distance = None, float(max_distance)
    for start in range(0, len(draw['positions']), 65536):
        rows = slice(start, start + 65536)
        p = draw['positions'][rows].astype(float)
        u, v = _render_axes(draw['rot'][rows])
        normal = np.cross(u, v)
        denominator = normal @ direction
        t = np.divide(np.sum((p-origin)*normal, axis=1), denominator,
                      out=np.full(len(p), np.inf), where=np.abs(denominator)>1e-10)
        candidates = np.flatnonzero((t >= 0.) & (t < distance))
        if not len(candidates):
            continue
        delta = origin + t[candidates, None]*direction - p[candidates]
        sigma = draw['scale'][rows][candidates, 0]
        x = np.sum(delta*u[candidates], axis=1)/sigma
        y = np.sum(delta*v[candidates], axis=1)/sigma
        keep = x*x + y*y < 9.
        for slot in (1, 2):
            if f'clip{slot}_angle' not in draw:
                continue
            angle = draw[f'clip{slot}_angle'][rows][candidates, 0]
            split = draw[f'clip{slot}_split'][rows][candidates, 0]
            keep &= (angle == 4.) | (np.cos(angle)*x + np.sin(angle)*y >= split)
        valid = candidates[keep]
        if len(valid):
            row = valid[np.argmin(t[valid])]
            distance = t[row]
            best = (start + int(row), origin + distance*direction)
    return best


def gs_outline(draw, row, toward_camera, segments=128):
    """Closed contour of the same clipped 3-sigma disk used for picking."""
    from cutsim.rendering.clip_fallback import _clip_polygon
    angles = np.arange(segments)*2.*np.pi/segments
    polygon = 3.*np.column_stack((np.cos(angles), np.sin(angles)))
    for slot in (1, 2):
        if f'clip{slot}_angle' not in draw:
            continue
        angle = float(draw[f'clip{slot}_angle'][row, 0])
        if angle != 4.:
            split = float(draw[f'clip{slot}_split'][row, 0])
            polygon = _clip_polygon(polygon, np.array([np.cos(angle), np.sin(angle), -split]))
    if len(polygon) < 3:
        return np.empty((0, 3))
    u, v = _render_axes(draw['rot'][row:row+1])
    normal = np.cross(u[0], v[0])
    if normal @ toward_camera < 0.:
        normal = -normal
    sigma = float(draw['scale'][row, 0])
    # Separate the line very slightly from the surface to avoid depth flicker.
    center = draw['positions'][row] + normal*max(sigma*.002, 1e-5)
    return center + sigma*(polygon[:, :1]*u[0] + polygon[:, 1:]*v[0])


def inspection_tip(command, point):
    from cutsim.core.commands import command_endpoints
    start, end = command_endpoints(command)
    target = np.asarray(point, float).copy()
    if command.tool_type.lower() == 'ball':
        target -= command.radius*np.asarray(command.tool_axis)
    move = end - start
    length2 = float(move @ move)
    fraction = np.clip((target-start) @ move / length2, 0., 1.) if length2 else 0.
    return start + fraction*move

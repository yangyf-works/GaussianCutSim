"""Generate every existing child-clip collar probe in one native batch."""
import numpy as np
from cutsim.geometry.strip_kernels import njit


@njit(cache=True, nogil=True)
def clip_collars(centres, ratios, angles, splits, cosine, sine, phase, budget):
    n = 12 if phase else 8
    result = np.empty((min(budget, len(centres)*2*3*n), 2), np.float64)
    count = 0
    local = np.empty((n, 2), np.float64)
    for row in range(len(centres)):
        for edge in range(2):
            split = splits[row, edge]
            if angles[row, edge] == 4. or abs(split) >= 3.:
                continue
            nx, ny = cosine[row, edge], sine[row, edge]
            half = np.sqrt(9.-split*split)
            accepted = 0
            for j in range(n):
                t = -half+2.*half*(j+(.61 if phase else .17))/n
                x, y = split*nx-t*ny, split*ny+t*nx
                valid = True
                for other in range(2):
                    valid = valid and (angles[row, other] == 4. or
                        x*cosine[row, other]+y*sine[row, other] >= splits[row, other]-2e-6)
                if valid:
                    local[accepted, 0], local[accepted, 1] = x, y
                    accepted += 1
            for offset in (-2e-7, 0., 2e-7):
                if count+accepted > budget:
                    return None
                for j in range(accepted):
                    result[count, 0] = centres[row, 0]+ratios[row]*local[j, 0]+offset*nx
                    result[count, 1] = centres[row, 1]+ratios[row]*local[j, 1]+offset*ny
                    count += 1
    return result[:count].copy()


def child_clip_probes(draw, centres, ratios, phase, budget):
    angles = np.column_stack((draw['clip1_angle'][:, 0], draw['clip2_angle'][:, 0])).astype(float)
    splits = np.column_stack((draw['clip1_split'][:, 0], draw['clip2_split'][:, 0])).astype(float)
    return clip_collars(centres, ratios, angles, splits, np.cos(angles), np.sin(angles), phase, budget)

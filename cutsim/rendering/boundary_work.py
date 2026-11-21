"""Bounded, block-local sharing of exact geometric boundary work.

Keys contain the actual ordered nearby geometry, never rounded positions or a
nearest-K approximation. Keys also contain the exact surviving CPU domains of
both parents and peers. Values are private read-only results, not draw packets;
drawing domains are applied afresh by every caller.
"""
import sys
import numpy as np


BOUNDARY_SETTINGS = (
    'sigma_scale', 'alpha', 'resolution_seam_scale', 'half_min_angle_deg',
    'edge_center_band', 'valley_min_angle_deg', 'clip_valleys', 'intersection_sigma',
    'clip_position_floor', 'clip_diversity_floor', 'clip_boundary_samples',
    'similar_edge_angle_deg',
)


class BoundaryRows(list):
    work_key = None


class PreparedBoundary:
    """Only the outputs consumed by packing; do not retain per-pair segments."""
    def __init__(self, selection, normals):
        self.selection = selection
        self.normals = normals

    def __len__(self):
        return len(self.selection[0])


def _owned_bytes(value):
    if isinstance(value, PreparedBoundary):
        return sys.getsizeof(value) + 256 + _owned_bytes(value.selection) + _owned_bytes(value.normals)
    if isinstance(value, np.ndarray):
        return sys.getsizeof(value) + (value.nbytes if value.base is not None else 0)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(_owned_bytes(k)+_owned_bytes(v) for k,v in value.items())
    if isinstance(value, (list, tuple)):
        return sys.getsizeof(value) + sum(_owned_bytes(v) for v in value)
    return sys.getsizeof(value)


class BoundaryWork:
    def __init__(self, max_bytes=64*1024*1024):
        self.max_bytes = int(max_bytes)
        self.clear()

    def clear(self):
        self.entries = {}
        self.bytes = 0
        self.hits = self.misses = self.selection_hits = 0

    def get(self, key):
        result = self.entries.get(key)
        if result is None:
            self.misses += 1
        else:
            self.hits += 1
        return result

    def put(self, key, rows):
        if self.bytes + len(key[1]) + len(key[2]) + 128 > self.max_bytes:
            return
        size = _owned_bytes(key) + _owned_bytes(rows) + 128
        if self.bytes + size <= self.max_bytes:
            self.entries[key] = rows
            self.bytes += size

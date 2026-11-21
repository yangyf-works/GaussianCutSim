"""Short-lived exact clipping reuse across selections of immutable GS parents.

The conservative dependency test decides which parents require recomputation
through finite-disk intersection, Top-2 selection and fallback construction.
"""
import numpy as np

from cutsim.core.local_state import _LocalPointStore, _local_take
from cutsim.core.scipy_support import cKDTree, _HAVE_SCIPY_GRAPH
from cutsim.core.splat_cloud import _rt_concat_clouds
from cutsim.rendering.clipping import NativeClipArrays, _gs_radius, _pack_native_clip


def peer_dirty_mask(parents, changed, settings):
    """Conservative dependency on changed finite disks, with no neighbour cap.

    Explicit domains don't read peers. Parallel disks cannot create a boundary.
    Numerical collars are deliberately broader than the packer's predicates.
    This does NOT change its intersection_sigma (normally 2 sigma).
    """
    dirty = np.zeros(len(parents.positions), bool)
    if not settings.half_clip or not len(changed.positions):
        return dirty
    rows = np.array([i for i, d in enumerate(parents.sweep_domains)
                     if getattr(d, 'render_domains', d) is None], np.int64)
    if not len(rows):
        return dirty
    radius = _gs_radius(parents.support, parents.seam_weight, settings)
    other = _gs_radius(changed.support, changed.seam_weight, settings)
    normals = changed.normals / np.maximum(np.linalg.norm(changed.normals, axis=1)[:, None], 1e-30)
    tree = cKDTree(changed.positions) if _HAVE_SCIPY_GRAPH else None
    threshold = np.sin(np.deg2rad(settings.half_min_angle_deg)) ** 2 - 1e-10
    for start in range(0, len(rows), 128):
        batch = rows[start:start+128]
        reach = radius[batch] + float(other.max()) + 1e-6
        neighbours = (tree.query_ball_point(parents.positions[batch], reach) if tree is not None else
                      [np.arange(len(other)) for _ in batch])
        for row, near in zip(batch, neighbours):
            if not len(near):
                continue
            near = np.asarray(near, np.int64)
            n = parents.normals[row] / max(np.linalg.norm(parents.normals[row]), 1e-30)
            cosine = normals[near] @ n
            delta = changed.positions[near] - parents.positions[row]
            dirty[row] = np.any(((1.-cosine*cosine) >= threshold) &
                (np.sum(delta*delta, axis=1) <= (radius[row]+other[near]+1e-6)**2))
    return dirty


def _records(arr):
    names = [k for k in arr if k != 'half_double_parent']
    out = np.empty(len(arr['positions']), dtype=[(k, arr[k].dtype, arr[k].shape[1:]) for k in names])
    for k in names:
        out[k] = arr[k]
    return out


class SelectionClipCache:
    """A cache for one fixed material/settings and immutable keyed source rows.

    Keys identify source rows, not current dense indices. Callers may select,
    remove, reorder or append rows, but must give changed attributes a NEW key.
    Lifetime is a single edge/sweep supplementation call, never machining history.
    """
    def __init__(self, material, settings):
        self.material = material
        self.settings = settings
        self.previous = None
        self.keys = np.empty(0, np.int64)
        self.records = None
        self.ready = np.zeros(0, bool)
        self.metadata = {}
        self.last_stats = {}

    def get(self, reference, keys, rows, *, cell=None):
        keys = np.asarray(keys, np.int64)
        rows = np.asarray(rows, np.int64)
        if len(keys) != len(reference.positions) or len(np.unique(keys)) != len(keys) or np.any(keys < 0):
            raise ValueError('Unique nonnegative keys are required for immutable reference rows')
        required = int(keys.max(initial=-1))+1
        if required > len(self.ready):
            self.ready = np.pad(self.ready, (0, required-len(self.ready)))
        changed_parts = []
        if self.previous is not None:
            removed = ~np.isin(self.keys, keys)
            added = ~np.isin(keys, self.keys)
            changed_parts.extend((_local_take(self.previous, np.flatnonzero(removed)),
                                  _local_take(reference, np.flatnonzero(added))))
            # Order can affect floating-point reductions for coincident/tied
            # peers. Reordering invalidates every parent that can see them.
            old_common = self.keys[~removed]; new_common = keys[~added]
            if not np.array_equal(old_common, new_common):
                moved = old_common[old_common != new_common]
                changed_parts.append(_local_take(reference, np.flatnonzero(np.isin(keys, moved))))
            # Invalidate all cached surviving parents, including rows not
            # requested this time but potentially requested on a later pass.
            active = np.flatnonzero(self.ready[keys])
            if len(active):
                affected = peer_dirty_mask(_local_take(reference, active), _rt_concat_clouds(*changed_parts), self.settings)
                self.ready[keys[active[affected]]] = False
            self.ready[self.keys[removed]] = False
        dirty = rows[~self.ready[keys[rows]]]
        if len(dirty) or self.records is None:
            indexed = _local_take(reference, np.arange(len(keys)))
            indexed._local_ids = np.arange(len(keys), dtype=np.int64)
            if cell is None:
                cell = max(4.*float(np.median(reference.support)), 1e-5) if len(keys) else 1.
            store = _LocalPointStore(cell); store.add(indexed)
            part = _pack_native_clip(_local_take(indexed, dirty), self.material, self.settings, reference_store=store)
            records = _records(part)
            if self.records is None or len(self.records) < required:
                previous = self.records
                self.records = np.empty(required, records.dtype)
                if previous is not None:
                    self.records[:len(previous)] = previous
            self.records[keys[dirty]] = records
            for i, key in enumerate(keys[dirty]):
                if i in part.approximated:
                    self.metadata[int(key)] = part.approximated[i]
                else:
                    self.metadata.pop(int(key), None)
            self.ready[keys[dirty]] = True
        selected = self.records[keys[rows]]
        out = NativeClipArrays({k: selected[k] for k in selected.dtype.names})
        out['source_ids'] = np.arange(len(rows), dtype=np.int64)
        out['half_double_parent'] = np.zeros(len(rows), bool)
        out.pending = {}
        out.approximated = {i: self.metadata[int(key)] for i, key in enumerate(keys[rows]) if int(key) in self.metadata}
        self.last_stats = dict(prepared=len(dirty), reused=len(rows)-len(dirty))
        self.previous, self.keys = reference, keys.copy()
        return out

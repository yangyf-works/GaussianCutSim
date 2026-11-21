"""Select completed GS records by tool progress; never evaluate intermediate cuts."""
import numpy as np

from cutsim.core.local_state import _local_take
from cutsim.core.splat_cloud import _rt_concat_clouds, _rt_subset_cloud
from cutsim.cutting.classification import _rt_motion_u


class BlockPlayback:
    def __init__(self, delta):
        self.before = delta.before
        self._unchanged=bool(getattr(delta,'stats',{}).get('empty_material_move',False))
        self.after = getattr(delta, '_local_final_cloud', None)
        if self.after is None:
            self.after = _rt_concat_clouds(
                _rt_subset_cloud(delta.before, ~delta.removed_mask),
                delta.split_children, delta.added)
        # Endpoint publication needs neither the motion projection nor an
        # O(N log N) ordering of unchanged parents. Build that exact ordering
        # only when an intermediate playback position actually requests it.
        self._schedule_inputs = (delta.removed_mask, delta.cut_u, delta.sweep,
                                 delta.split_birth_u, delta.birth_u)
        after_count = self.after.row_count if hasattr(self.after,'row_count') else len(self.after.positions)
        expected_count = (len(delta.removed_mask)-np.count_nonzero(delta.removed_mask)
                          +len(delta.split_birth_u)+len(delta.birth_u))
        if expected_count != after_count:
            raise ValueError('Playback final rows must be retained + split + added')
        self._schedule = None
        self._key = None
        self._rows = None
        self._cloud_key = None
        self._cloud = None

    def _prepare_schedule(self):
        if self._schedule is not None:
            return self._schedule
        removed, cut_u, sweep, split_birth, birth_u = self._schedule_inputs
        retained = np.flatnonzero(~removed)
        # Surviving parents also have completed clipping/domain changes. Switch
        # their before/after records at the same projected position, without fitting.
        retire = np.array(cut_u, copy=True)
        retire[retained] = _rt_motion_u(sweep, self.before.positions[retained])
        birth = np.concatenate((retire[retained], split_birth, birth_u))
        if len(birth) != len(self.after.positions):
            raise ValueError('Playback final rows must be retained + split + added')
        remove_order = np.argsort(retire, kind='stable')
        add_order = np.argsort(birth, kind='stable')
        self._schedule = (remove_order, add_order, retire[remove_order], birth[add_order])
        return self._schedule

    def prepare_motion(self):
        """Prepare once on the receiver thread, before publishing to the UI."""
        if self._unchanged:return
        self._prepare_schedule()

    remove_order = property(lambda self: self._prepare_schedule()[0])
    add_order = property(lambda self: self._prepare_schedule()[1])
    remove_times = property(lambda self: self._prepare_schedule()[2])
    add_times = property(lambda self: self._prepare_schedule()[3])

    def rows(self, u):
        u = float(np.clip(u, 0., 1.))
        if self._unchanged:
            key = ('after',) if u>=1. else ('before',)
        elif u <= 0.:
            key = ('before',)
        elif u >= 1.:
            key = ('after',)
        else:
            key = (int(np.searchsorted(self.remove_times, u, side='right')),
                   int(np.searchsorted(self.add_times, u, side='right')))
        if key != self._key:
            if key == ('before',):
                old = np.arange(len(self.before.positions), dtype=np.int64)
                new = np.empty(0, dtype=np.int64)
            elif key == ('after',):
                old = np.empty(0, dtype=np.int64)
                new = np.arange(len(self.after.positions), dtype=np.int64)
            else:
                keep = np.ones(len(self.before.positions), dtype=bool)
                keep[self.remove_order[:key[0]]] = False
                added = np.zeros(len(self.after.positions), dtype=bool)
                added[self.add_order[:key[1]]] = True
                old, new = np.flatnonzero(keep), np.flatnonzero(added)
            self._key, self._rows = key, (old, new)
        return self._rows

    def visible_cloud(self, u):
        old, new = self.rows(u)
        if self._key == ('before',):
            return self.before
        if self._key == ('after',):
            return self.after
        if self._cloud_key != self._key:
            self._cloud = _rt_concat_clouds(_local_take(self.before, old), _local_take(self.after, new))
            self._cloud_key = self._key
        return self._cloud

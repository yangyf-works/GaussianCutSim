"""Persistent, lossless GS endpoint with lazy flat-array compatibility."""
import numpy as np
from cutsim.core.splat_cloud import SplatCloud
from cutsim.core.frozen_rows import FrozenRows, BUCKET_SIZE, bucket_ranges

FIELDS = ('positions', 'normals', 'support', 'tool_ids', 'refine_levels',
          'seam_weight', 'inherited_clip', 'sweep_domains')


class ChunkedCloud(SplatCloud):
    def __init__(self, chunks):
        self.chunks = tuple(chunks)
        self._flat = {}
        self._ends = np.cumsum([len(c.ids) for c in self.chunks], dtype=np.int64)

    @property
    def row_count(self):
        return int(self._ends[-1]) if len(self._ends) else 0

    def __getattribute__(self, name):
        if name in FIELDS or name == '_local_ids':
            flat = object.__getattribute__(self, '_flat')
            if name not in flat:
                chunks = object.__getattribute__(self, 'chunks')
                values = [c.ids if name == '_local_ids' else c.arrays[name] for c in chunks]
                if values:
                    value = np.concatenate(values)
                else:
                    value = (np.empty(0, np.int64) if name == '_local_ids'
                             else getattr(SplatCloud.empty(), name))
                value.flags.writeable = False
                flat[name] = value
            return flat[name]
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in FIELDS or name == '_local_ids':
            raise AttributeError('Completed chunked GS geometry is immutable')
        super().__setattr__(name, value)

    def __copy__(self):
        out = object.__new__(type(self))
        out.__dict__ = self.__dict__.copy()
        out._flat = self._flat.copy()
        return out

    def __getstate__(self):
        # Dense compatibility views must not turn a local archive write into
        # another scan/copy of the complete endpoint.
        return {k: v for k, v in vars(self).items() if k != '_flat'}

    def __setstate__(self, state):
        self.__dict__.update(state); self._flat = {}

    def dense_view(self):
        """Mutable wrapper for standalone/debug projection; buffers stay shared.

        Projection code owns its replacement domain arrays. The completed
        geometric endpoint and its shared buckets remain immutable.
        """
        out=SplatCloud._from_arrays([getattr(self,k) for k in FIELDS])
        out._local_ids=self._local_ids
        if hasattr(self,'_local_state'):out._local_state=self._local_state
        return out

    def take(self, rows):
        count = int(self._ends[-1]) if len(self._ends) else 0
        if isinstance(rows, slice):
            rows = np.arange(count, dtype=np.int64)[rows]
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        rows = rows.astype(np.int64, copy=False).reshape(-1)
        rows = np.where(rows < 0, rows+count, rows)
        if np.any((rows < 0) | (rows >= count)):
            raise IndexError('GS row outside endpoint')
        template = self.chunks[0].arrays if self.chunks else vars(SplatCloud.empty())
        arrays = {k: np.empty((len(rows),)+template[k].shape[1:], template[k].dtype) for k in FIELDS}
        ids = np.empty(len(rows), np.int64)
        owners = np.searchsorted(self._ends, rows, side='right')
        for owner in np.unique(owners):
            selected = np.flatnonzero(owners == owner)
            local = rows[selected] - (self._ends[owner-1] if owner else 0)
            chunk = self.chunks[owner]
            for k in FIELDS:
                arrays[k][selected] = chunk.arrays[k][local]
            ids[selected] = chunk.ids[local]
        out = SplatCloud._from_arrays([arrays[k] for k in FIELDS]); out._local_ids = ids
        return out

    @classmethod
    def update(cls, before, removed_ids, added):
        """Rebuild only buckets touched by removals or monotonically born IDs."""
        chunks = {int(c.ids[0])//BUCKET_SIZE: c for c in before.chunks} if isinstance(before, cls) else {}
        if not isinstance(before, cls) and len(before.positions):
            # Used only when importing a dense endpoint. The first Dexel
            # publication replaces all initial display GS, so bypass it there.
            for key, a, b in bucket_ranges(before._local_ids):
                chunks[key] = FrozenRows(before._local_ids[a:b], {k: getattr(before,k)[a:b] for k in FIELDS})
        removed = np.sort(np.asarray(removed_ids, np.int64))
        removals = {key: removed[a:b] for key,a,b in bucket_ranges(removed)}
        births = {key:(a,b) for key,a,b in bucket_ranges(added._local_ids)}
        for key in removals.keys() | births.keys():
            old = chunks.get(key)
            keep = ~np.isin(old.ids, removals.get(key, ())) if old is not None else np.zeros(0, bool)
            a,b = births.get(key, (0,0))
            ids = np.concatenate((old.ids[keep] if old is not None else np.empty(0,np.int64), added._local_ids[a:b]))
            if not len(ids):
                chunks.pop(key, None); continue
            arrays = {k: np.concatenate((old.arrays[k][keep], getattr(added,k)[a:b]))
                      if old is not None else getattr(added,k)[a:b] for k in FIELDS}
            chunks[key] = FrozenRows(ids, arrays)
        return cls([chunks[key] for key in sorted(chunks)])

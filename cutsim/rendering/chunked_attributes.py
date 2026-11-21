"""Lazy native attributes backed by immutable changed-only row buckets."""
import numpy as np
from weakref import WeakKeyDictionary
from cutsim.rendering.clipping import NativeClipArrays

_chunk_stats = WeakKeyDictionary()


class ChunkedAttributes(NativeClipArrays):
    _immutable_chunks = True
    _clip_draws_resolved = True

    def __init__(self, chunks, density=1.):
        self.chunks = tuple(chunks)
        self.gs_density = density
        self.pending = {}
        self._approximated = None
        self._flat = {}
        offset = 0
        for chunk in self.chunks:
            offset += len(chunk.ids)
        self.row_count = offset
        names = tuple(self.chunks[0].arrays) if self.chunks else ()
        # Preserve the dict API used throughout the renderer. Values are
        # materialized only through the public accessors below.
        dict.__init__(self, ((name,None) for name in names))
        dict.__setitem__(self, 'source_ids', None)
        dict.__setitem__(self, 'half_double_parent', None)

    @property
    def approximated(self):
        if self._approximated is None:
            self._approximated = {}
            offset = 0
            for chunk in self.chunks:
                self._approximated.update((offset+row, item) for row,item in chunk.diagnostics.items())
                offset += len(chunk.ids)
        return self._approximated

    def native_stats(self):
        """Reduce cached scalar summaries, never flatten untouched row buffers."""
        from types import SimpleNamespace
        from cutsim.rendering.clipping import _native_clip_stats
        from cutsim.rendering.gaussian_attributes import _edge_stats_empty
        total = _edge_stats_empty()
        constants = {'clip_native','clip_fixed_top_two','clip_ridge_selection','clip_intersection_rejection'}
        maxima = {'clip_omitted_mass_max','clip_partition_guard_max'}
        for chunk in self.chunks:
            stats = _chunk_stats.get(chunk)
            if stats is None:
                part = NativeClipArrays(chunk.arrays)
                part.approximated = chunk.diagnostics
                context = SimpleNamespace()
                _native_clip_stats(part, context)
                stats = context.last_edge_split_stats
                _chunk_stats[chunk] = stats
            for key, value in stats.items():
                if key in constants: total[key] = value
                elif key in maxima: total[key] = max(total.get(key,0), value)
                else: total[key] = total.get(key,0) + value
        return total

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        if key not in self._flat:
            if key == 'source_ids': value = np.arange(self.row_count,dtype=np.int64)
            elif key == 'half_double_parent': value = np.zeros(self.row_count,bool)
            else: value = np.concatenate([chunk.arrays[key] for chunk in self.chunks])
            value.flags.writeable = False
            self._flat[key] = value
        return self._flat[key]

    def __iter__(self):
        # dict(endpoint) must use __getitem__, not CPython's fast dict copy
        # of the private placeholders.
        return dict.__iter__(self)

    def get(self, key, default=None):
        return self[key] if key in self else default

    def items(self):
        return ((key,self[key]) for key in self)

    def values(self):
        return (self[key] for key in self)

    def copy(self):
        out = NativeClipArrays(self.items())
        out.pending = self.pending; out.approximated = self.approximated
        out.gs_density = self.gs_density
        return out

    def __reduce_ex__(self, protocol):
        # No flat buffers or dict placeholders in IPC/history. Reconstruct
        # compatibility arrays only when the consumer actually draws them.
        return (type(self), (self.chunks,self.gs_density))


def freeze_endpoint(endpoint):
    if not getattr(endpoint, '_immutable_chunks', False):
        for value in endpoint.values():
            value.flags.writeable = False
